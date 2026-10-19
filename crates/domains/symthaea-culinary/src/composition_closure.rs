// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Proximate composition closure diagnostic for food composition profiles.
//!
//! This checks whether declared composition components plausibly sum to 100 g
//! per 100 g edible fresh-weight food. It is a diagnostic, not proof of the
//! correctness of any one component, a nutrient assay, or a food-safety claim.
//! Process-boundary mass conservation belongs to PIE-001; this module does not
//! implement a second process mass-balance engine.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const AVAILABLE_PLUS_FIBRE: &[&str] = &[
    "water", "protein", "fat", "available_carbohydrate", "dietary_fibre", "ash", "alcohol",
];
const TOTAL_CARBOHYDRATE: &[&str] = &[
    "water", "protein", "fat", "total_carbohydrate", "ash", "alcohol",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositionClosureDispositionV1 {
    /// The entire possible sum lies in 97–103 g per 100 g.
    WithinPreferred,
    /// The entire sum lies within 95–105 g, but not wholly within 97–103 g.
    WithinAcceptable,
    /// The entire sum lies outside the acceptable 95–105 g interval.
    OutsideAcceptable,
    /// Missing components or uncertainty prevent classification.
    Indeterminate,
}

impl CompositionClosureDispositionV1 {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WithinPreferred => "within_preferred",
            Self::WithinAcceptable => "within_acceptable",
            Self::OutsideAcceptable => "outside_acceptable",
            Self::Indeterminate => "indeterminate",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompositionClosureResultV1 {
    pub disposition: CompositionClosureDispositionV1,
    /// None when a required component has no point estimate or is unavailable.
    pub point_sum_g_per_100g: Option<f64>,
    /// Sum of known component lower bounds. Missing components are nonnegative.
    pub lower_sum_g_per_100g: f64,
    /// None when one or more required components are unavailable.
    pub upper_sum_g_per_100g: Option<f64>,
    pub deviation_from_100_g_per_100g: Option<f64>,
    pub unavailable_components: Vec<String>,
    pub below_quantification_limit_components: Vec<String>,
    /// Metadata only; this does not authenticate external sources.
    pub all_component_evidence_marked_qualified: bool,
}

fn non_empty<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str).filter(|text| !text.trim().is_empty())
}

fn number(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64).filter(|n| n.is_finite())
}

fn valid_timestamp(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() < 20
        || bytes.get(4) != Some(&b'-')
        || bytes.get(7) != Some(&b'-')
        || bytes.get(10) != Some(&b'T')
        || bytes.get(13) != Some(&b':')
        || bytes.get(16) != Some(&b':')
    {
        return false;
    }
    let parse = |start: usize, end: usize| {
        std::str::from_utf8(&bytes[start..end]).ok()?.parse::<u32>().ok()
    };
    let (Some(year), Some(month), Some(day), Some(hour), Some(minute), Some(second)) = (
        parse(0, 4), parse(5, 7), parse(8, 10), parse(11, 13), parse(14, 16), parse(17, 19),
    ) else {
        return false;
    };
    if !(1..=12).contains(&month) || hour > 23 || minute > 59 || second > 60 {
        return false;
    }
    let leap = year % 400 == 0 || (year % 4 == 0 && year % 100 != 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    if day == 0 || day > days {
        return false;
    }
    let tz_start = if bytes.last() == Some(&b'Z') {
        bytes.len() - 1
    } else {
        if bytes.len() < 25 {
            return false;
        }
        let start = bytes.len() - 6;
        if !matches!(bytes[start], b'+' | b'-') || bytes[start + 3] != b':' {
            return false;
        }
        let (Some(h), Some(m)) = (parse(start + 1, start + 3), parse(start + 4, start + 6)) else {
            return false;
        };
        if h > 23 || m > 59 {
            return false;
        }
        start
    };
    if tz_start == 19 {
        return true;
    }
    bytes.get(19) == Some(&b'.')
        && tz_start > 20
        && bytes[20..tz_start].iter().all(|b| b.is_ascii_digit())
}

fn validate_evidence(value: &Value, label: &str, errors: &mut Vec<String>) -> bool {
    let source_ref = non_empty(value, "source_ref");
    let source_version = non_empty(value, "source_version");
    let known_at = non_empty(value, "known_at");
    let status = non_empty(value, "evidence_status");
    if source_ref.is_none() {
        errors.push(format!("{label}.source_ref is required"));
    }
    if source_version.is_none() {
        errors.push(format!("{label}.source_version is required"));
    }
    if !known_at.is_some_and(valid_timestamp) {
        errors.push(format!("{label}.known_at must be a valid RFC3339 timestamp"));
    }
    if !matches!(status, Some("synthetic" | "unreviewed" | "reviewed" | "qualified")) {
        errors.push(format!("{label}.evidence_status is invalid"));
    }
    if status == Some("qualified")
        && source_ref.is_some_and(|reference| reference.starts_with("urn:synthetic:"))
    {
        errors.push(format!("{label} cannot qualify an explicitly synthetic source"));
        return false;
    }
    status == Some("qualified")
        && source_ref.is_some()
        && source_version.is_some()
        && known_at.is_some_and(valid_timestamp)
}

fn classify(lower: f64, upper: f64) -> CompositionClosureDispositionV1 {
    if lower >= 97.0 && upper <= 103.0 {
        CompositionClosureDispositionV1::WithinPreferred
    } else if lower >= 95.0 && upper <= 105.0 {
        CompositionClosureDispositionV1::WithinAcceptable
    } else if upper < 95.0 || lower > 105.0 {
        CompositionClosureDispositionV1::OutsideAcceptable
    } else {
        CompositionClosureDispositionV1::Indeterminate
    }
}

/// Check closure for g/100 g edible fresh-weight composition.
///
/// Exactly one carbohydrate convention must be selected:
/// available carbohydrate plus separate dietary fibre, or total carbohydrate
/// that already includes fibre. Unavailable components force Indeterminate;
/// their absence is never silently treated as a numeric zero. Explicit below-
/// quantification-limit values are represented by [0, LOQ], with no point value.
pub fn check_proximate_composition_closure_v1(
    profile: &Value,
) -> Result<CompositionClosureResultV1, Vec<String>> {
    let mut errors = Vec::new();
    if profile.get("schema_version").and_then(Value::as_str) != Some("1.0.0") {
        errors.push("schema_version must be 1.0.0".to_owned());
    }
    if non_empty(profile, "profile_id").is_none() {
        errors.push("profile_id is required".to_owned());
    }
    if non_empty(profile, "basis") != Some("per_100g_edible_fresh_weight") {
        errors.push("basis must be per_100g_edible_fresh_weight".to_owned());
    }
    if non_empty(profile, "food_state").is_none() {
        errors.push("food_state is required".to_owned());
    }
    let required: &[&str] = match non_empty(profile, "carbohydrate_basis") {
        Some("available_plus_fibre") => AVAILABLE_PLUS_FIBRE,
        Some("total_includes_fibre") => TOTAL_CARBOHYDRATE,
        _ => {
            errors.push("carbohydrate_basis must be available_plus_fibre or total_includes_fibre".to_owned());
            &[]
        }
    };
    let required_set: BTreeSet<&str> = required.iter().copied().collect();
    let components: &[Value] = match profile.get("components").and_then(Value::as_array) {
        Some(items) => items.as_slice(),
        None => {
            errors.push("components must be an array".to_owned());
            &[]
        }
    };
    let unavailable: &[Value] = match profile.get("unavailable_components").and_then(Value::as_array) {
        Some(items) => items.as_slice(),
        None => {
            errors.push("unavailable_components must be an array".to_owned());
            &[]
        }
    };

    let profile_basis = non_empty(profile, "basis");
    let profile_state = non_empty(profile, "food_state");
    let mut seen = BTreeSet::new();
    let mut values: BTreeMap<String, (f64, Option<f64>, f64)> = BTreeMap::new();
    let mut point_complete = true;
    let mut all_qualified = true;
    let mut below_loq = Vec::new();
    let mut unavailable_ids = Vec::new();

    for (index, component) in components.iter().enumerate() {
        let label = format!("components[{index}]");
        let Some(id) = non_empty(component, "component_id") else {
            errors.push(format!("{label}.component_id is required"));
            continue;
        };
        if !required_set.contains(id) {
            errors.push(format!("{label} component_id {id:?} is not part of the selected carbohydrate convention"));
            continue;
        }
        if !seen.insert(id.to_owned()) {
            errors.push(format!("duplicate component entry {id:?}"));
            continue;
        }
        if non_empty(component, "unit") != Some("g/100g") {
            errors.push(format!("{label}.unit must be g/100g"));
        }
        if non_empty(component, "basis") != profile_basis {
            errors.push(format!("{label}.basis must match profile basis"));
        }
        if non_empty(component, "food_state") != profile_state {
            errors.push(format!("{label}.food_state must match profile food_state"));
        }
        if !validate_evidence(
            component.get("evidence").unwrap_or(&Value::Null),
            &format!("{label}.evidence"),
            &mut errors,
        ) {
            all_qualified = false;
        }
        match non_empty(component, "measurement_status") {
            Some("measured" | "calculated") => {
                match (number(component, "value"), number(component, "lower"), number(component, "upper")) {
                    (Some(value), Some(lower), Some(upper))
                        if lower >= 0.0 && lower <= value && value <= upper =>
                    {
                        values.insert(id.to_owned(), (lower, Some(value), upper));
                    }
                    _ => errors.push(format!("{label} interval must satisfy 0 <= lower <= value <= upper")),
                }
            }
            Some("below_quantification_limit") => {
                let lower = number(component, "lower");
                let upper = number(component, "upper");
                if !component.get("value").is_some_and(Value::is_null)
                    || lower != Some(0.0)
                    || !upper.is_some_and(|v| v.is_finite() && v > 0.0)
                {
                    errors.push(format!("{label} below_quantification_limit requires value=null, lower=0, upper>0"));
                } else if let Some(upper) = upper {
                    values.insert(id.to_owned(), (0.0, None, upper));
                    below_loq.push(id.to_owned());
                    point_complete = false;
                }
            }
            _ => errors.push(format!("{label}.measurement_status must be measured, calculated, or below_quantification_limit")),
        }
    }

    for (index, item) in unavailable.iter().enumerate() {
        let label = format!("unavailable_components[{index}]");
        let Some(id) = non_empty(item, "component_id") else {
            errors.push(format!("{label}.component_id is required"));
            continue;
        };
        if !required_set.contains(id) {
            errors.push(format!("{label} component_id {id:?} is not part of the selected carbohydrate convention"));
            continue;
        }
        if !seen.insert(id.to_owned()) {
            errors.push(format!("component {id:?} is recorded more than once"));
            continue;
        }
        if !matches!(
            non_empty(item, "reason"),
            Some("not_analyzed" | "not_reported" | "unsupported_basis" | "conflicting_sources" | "unknown")
        ) {
            errors.push(format!("{label}.reason is invalid"));
        }
        if non_empty(item, "detail").is_none() {
            errors.push(format!("{label}.detail is required"));
        }
        if !validate_evidence(
            item.get("evidence").unwrap_or(&Value::Null),
            &format!("{label}.evidence"),
            &mut errors,
        ) {
            all_qualified = false;
        }
        unavailable_ids.push(id.to_owned());
        point_complete = false;
    }

    for id in required {
        if !seen.contains(*id) {
            errors.push(format!("required component {id:?} must be measured or explicitly listed unavailable"));
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let lower_sum: f64 = values.values().map(|v| v.0).sum();
    let complete = unavailable_ids.is_empty();
    let upper_sum = if complete {
        Some(values.values().map(|v| v.2).sum::<f64>())
    } else {
        None
    };
    let point_sum = if complete && point_complete {
        Some(values.values().map(|v| v.1.expect("point completeness checked")).sum::<f64>())
    } else {
        None
    };
    if !lower_sum.is_finite()
        || upper_sum.is_some_and(|v| !v.is_finite())
        || point_sum.is_some_and(|v| !v.is_finite())
    {
        return Err(vec!["summed proximate composition is non-finite".to_owned()]);
    }
    let disposition = match upper_sum {
        Some(upper) => classify(lower_sum, upper),
        None if lower_sum > 105.0 => CompositionClosureDispositionV1::OutsideAcceptable,
        None => CompositionClosureDispositionV1::Indeterminate,
    };
    if let (Some(point), Some(upper)) = (point_sum, upper_sum) {
        if point < lower_sum || point > upper {
            return Err(vec!["component intervals do not contain the point sum".to_owned()]);
        }
    }
    Ok(CompositionClosureResultV1 {
        disposition,
        point_sum_g_per_100g: point_sum,
        lower_sum_g_per_100g: lower_sum,
        upper_sum_g_per_100g: upper_sum,
        deviation_from_100_g_per_100g: point_sum.map(|sum| sum - 100.0),
        unavailable_components: unavailable_ids,
        below_quantification_limit_components: below_loq,
        all_component_evidence_marked_qualified: all_qualified,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../fixtures/composition-closure-v1/synthetic-profile.json"
        ))
        .expect("synthetic profile fixture must parse")
    }

    #[test]
    fn complete_profile_inside_preferred_band_is_classified() {
        let result = check_proximate_composition_closure_v1(&fixture()).unwrap();
        assert_eq!(result.disposition, CompositionClosureDispositionV1::WithinPreferred);
        assert_eq!(result.point_sum_g_per_100g, Some(101.0));
        assert!(!result.all_component_evidence_marked_qualified);
    }

    #[test]
    fn broad_interval_is_indeterminate_not_balanced() {
        let mut value = fixture();
        value["components"][0]["lower"] = json!(56.0);
        value["components"][0]["upper"] = json!(68.0);
        let result = check_proximate_composition_closure_v1(&value).unwrap();
        assert_eq!(result.disposition, CompositionClosureDispositionV1::Indeterminate);
    }

    #[test]
    fn unavailable_component_does_not_hide_a_known_lower_sum_above_limit() {
        let mut value = fixture();
        value["components"][0]["value"] = json!(70.0);
        value["components"][0]["lower"] = json!(70.0);
        value["components"][0]["upper"] = json!(70.0);
        value["components"].as_array_mut().unwrap().pop();
        value["unavailable_components"] = serde_json::json!([{
            "component_id": "alcohol",
            "reason": "not_analyzed",
            "detail": "synthetic missingness test",
            "evidence": {
                "source_ref": "urn:synthetic:missing:alcohol",
                "source_version": "fixture-v1",
                "known_at": "2026-10-18T00:00:00Z",
                "evidence_status": "synthetic"
            }
        }]);
        let result = check_proximate_composition_closure_v1(&value).unwrap();
        assert_eq!(result.disposition, CompositionClosureDispositionV1::OutsideAcceptable);
        assert!(result.upper_sum_g_per_100g.is_none());
    }

    #[test]
    fn acceptable_band_is_distinct_from_preferred() {
        let mut value = fixture();
        value["components"][0]["value"] = json!(65.0);
        value["components"][0]["lower"] = json!(65.0);
        value["components"][0]["upper"] = json!(65.0);
        for component in value["components"].as_array_mut().unwrap().iter_mut().skip(1) {
            let point = component["value"].as_f64().unwrap();
            component["lower"] = json!(point);
            component["upper"] = json!(point);
        }
        let result = check_proximate_composition_closure_v1(&value).unwrap();
        assert_eq!(result.point_sum_g_per_100g, Some(104.0));
        assert_eq!(result.disposition, CompositionClosureDispositionV1::WithinAcceptable);
    }

    #[test]
    fn fully_outside_interval_is_flagged() {
        let mut value = fixture();
        value["components"][0]["value"] = json!(70.0);
        value["components"][0]["lower"] = json!(70.0);
        value["components"][0]["upper"] = json!(70.0);
        let result = check_proximate_composition_closure_v1(&value).unwrap();
        assert_eq!(result.point_sum_g_per_100g, Some(109.0));
        assert_eq!(result.disposition, CompositionClosureDispositionV1::OutsideAcceptable);
    }

    #[test]
    fn unavailable_component_makes_closure_indeterminate_not_zero() {
        let mut value = fixture();
        value["components"].as_array_mut().unwrap().pop();
        value["unavailable_components"] = json!([{
            "component_id":"alcohol",
            "reason":"not_analyzed",
            "detail":"synthetic missingness test",
            "evidence":{
                "source_ref":"urn:synthetic:missing:alcohol",
                "source_version":"fixture-v1",
                "known_at":"2026-10-18T00:00:00Z",
                "evidence_status":"synthetic"
            }
        }]);
        let result = check_proximate_composition_closure_v1(&value).unwrap();
        assert_eq!(result.disposition, CompositionClosureDispositionV1::Indeterminate);
        assert!(result.point_sum_g_per_100g.is_none());
        assert!(result.upper_sum_g_per_100g.is_none());
        assert_eq!(result.unavailable_components, vec!["alcohol"]);
    }

    #[test]
    fn below_quantification_limit_is_an_interval_not_zero() {
        let mut value = fixture();
        value["components"][6]["measurement_status"] = json!("below_quantification_limit");
        value["components"][6]["value"] = Value::Null;
        value["components"][6]["lower"] = json!(0.0);
        value["components"][6]["upper"] = json!(0.1);
        let result = check_proximate_composition_closure_v1(&value).unwrap();
        assert!(result.point_sum_g_per_100g.is_none());
        assert_eq!(result.below_quantification_limit_components, vec!["alcohol"]);
        assert_eq!(result.disposition, CompositionClosureDispositionV1::WithinPreferred);
    }

    #[test]
    fn incompatible_carbohydrate_convention_is_rejected() {
        let mut value = fixture();
        value["carbohydrate_basis"] = json!("total_includes_fibre");
        assert!(check_proximate_composition_closure_v1(&value).is_err());
    }

    #[test]
    fn basis_or_state_mismatch_is_rejected() {
        let mut value = fixture();
        value["components"][1]["food_state"] = json!("raw");
        assert!(check_proximate_composition_closure_v1(&value)
            .unwrap_err().iter().any(|e| e.contains("food_state")));
    }

    #[test]
    fn synthetic_source_cannot_be_relabelled_qualified() {
        let mut value = fixture();
        value["components"][0]["evidence"]["evidence_status"] = json!("qualified");
        assert!(check_proximate_composition_closure_v1(&value)
            .unwrap_err().iter().any(|e| e.contains("cannot qualify an explicitly synthetic")));
    }

    #[test]
    fn duplicate_component_ids_are_rejected() {
        let mut value = fixture();
        value["components"][1]["component_id"] = json!("water");
        let errors = check_proximate_composition_closure_v1(&value).unwrap_err();
        assert!(errors.iter().any(|e| e.contains("duplicate component entry")));
    }

    #[test]
    fn invalid_calendar_timestamp_is_rejected() {
        let mut value = fixture();
        value["components"][0]["evidence"]["known_at"] = json!("2025-02-29T12:00:00Z");
        assert!(check_proximate_composition_closure_v1(&value)
            .unwrap_err().iter().any(|e| e.contains("valid RFC3339")));
    }

    #[test]
    fn total_carbohydrate_mode_uses_one_non_overlapping_carbohydrate_row() {
        let mut value = fixture();
        value["carbohydrate_basis"] = json!("total_includes_fibre");
        value["components"].as_array_mut().unwrap().retain(|c| {
            c.get("component_id").and_then(Value::as_str) != Some("available_carbohydrate")
                && c.get("component_id").and_then(Value::as_str) != Some("dietary_fibre")
        });
        value["components"].as_array_mut().unwrap().push(json!({
            "component_id":"total_carbohydrate","measurement_status":"measured",
            "value":23.0,"lower":22.8,"upper":23.2,"unit":"g/100g",
            "basis":"per_100g_edible_fresh_weight","food_state":"prepared",
            "evidence":{
                "source_ref":"urn:synthetic:component:total-carbohydrate",
                "source_version":"fixture-v1","known_at":"2026-10-18T00:00:00Z",
                "evidence_status":"synthetic"
            }
        }));
        let result = check_proximate_composition_closure_v1(&value).unwrap();
        assert_eq!(result.disposition, CompositionClosureDispositionV1::WithinPreferred);
    }
}
 
    #[test]
    fn widening_a_component_interval_never_strengthens_closure() {
        let mut value = fixture();
        let center = value["components"][0]["value"].as_f64().unwrap();
        let mut seen_acceptable = false;
        let mut seen_indeterminate = false;
        for spread in 0..=8 {
            let spread = spread as f64;
            value["components"][0]["lower"] = json!((center - spread).max(0.0));
            value["components"][0]["upper"] = json!(center + spread);
            let result = check_proximate_composition_closure_v1(&value).unwrap();
            match result.disposition {
                CompositionClosureDispositionV1::WithinPreferred => {
                    assert!(!seen_acceptable && !seen_indeterminate);
                }
                CompositionClosureDispositionV1::WithinAcceptable => {
                    seen_acceptable = true;
                    assert!(!seen_indeterminate);
                }
                CompositionClosureDispositionV1::Indeterminate => {
                    seen_indeterminate = true;
                }
                CompositionClosureDispositionV1::OutsideAcceptable => {
                    panic!("widening an interval around a valid point sum must not manufacture outside evidence");
                }
            }
        }
    }
