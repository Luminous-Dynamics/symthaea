// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bearing food-quantity conversions.
//!
//! This contract keeps identity matching separate from quantity transformation.
//! It checks a declared chain of mass/volume transformations, pins every input
//! and step to a source version, propagates conservative factor intervals, and
//! refuses to call a calculation qualified when any evidence is unqualified.
//! It does not authenticate source references, estimate nutrient retention, or
//! establish that a source's empirical factor is representative.

use serde_json::Value;
use std::collections::BTreeSet;

/// Computed values after a valid FoodQuantityConversionV1 chain.
#[derive(Debug, Clone, PartialEq)]
pub struct FoodQuantityConversionResultV1 {
    pub amount: f64,
    pub lower: f64,
    pub upper: f64,
    pub unit: String,
    pub basis: String,
    pub food_state: String,
}

fn non_empty<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
}

fn number(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64).filter(|n| n.is_finite())
}

/// Canonical factors expressed in grams for mass or millilitres for volume.
/// Household measures are intentionally excluded because their realized
/// quantity is food-, vessel-, and region-dependent.
fn unit_definition(unit: &str) -> Option<(&'static str, f64)> {
    match unit {
        "kg" => Some(("mass", 1_000.0)),
        "g" => Some(("mass", 1.0)),
        "mg" => Some(("mass", 0.001)),
        "ug" => Some(("mass", 0.000_001)),
        "oz" => Some(("mass", 28.349_523_125)),
        "lb" => Some(("mass", 453.592_37)),
        "L" => Some(("volume", 1_000.0)),
        "mL" => Some(("volume", 1.0)),
        "uL" => Some(("volume", 0.001)),
        _ => None,
    }
}

fn timestamp_looks_rfc3339(value: &str) -> bool {
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
    if bytes.last() == Some(&b'Z') {
        return true;
    }
    bytes.len() >= 25
        && matches!(bytes.get(bytes.len() - 6).copied(), Some(b'+' | b'-'))
        && bytes.get(bytes.len() - 3) == Some(&b':')
}

fn validate_evidence<'a>(
    value: &'a Value,
    label: &str,
    errors: &mut Vec<String>,
) -> Option<&'a str> {
    if non_empty(value, "source_ref").is_none() {
        errors.push(format!("{label}.source_ref must be a non-empty source locator"));
    }
    if non_empty(value, "source_version").is_none() {
        errors.push(format!("{label}.source_version must pin a source release/version"));
    }
    match non_empty(value, "known_at") {
        Some(timestamp) if timestamp_looks_rfc3339(timestamp) => {}
        _ => errors.push(format!("{label}.known_at must be an RFC3339 timestamp")),
    }
    match non_empty(value, "evidence_status") {
        Some(status)
            if matches!(status, "synthetic" | "unreviewed" | "reviewed" | "qualified") =>
        {
            Some(status)
        }
        _ => {
            errors.push(format!(
                "{label}.evidence_status must be synthetic, unreviewed, reviewed, or qualified"
            ));
            None
        }
    }
}

/// Validate the semantic invariants of a FoodQuantityConversionV1 chain.
///
/// Ok means the declared arithmetic, step continuity, source-version fields,
/// factor intervals, and output claim agree with each other. It does not
/// validate a complete JSON Schema, fetch cited sources, verify permissions,
/// or prove that empirical conversion factors describe the real food/process.
pub fn validate_food_quantity_conversion_v1(
    assessment: &Value,
) -> Result<FoodQuantityConversionResultV1, Vec<String>> {
    let mut errors = Vec::new();

    if assessment.get("schema_version").and_then(Value::as_str) != Some("1.0.0") {
        errors.push("schema_version must be 1.0.0".to_owned());
    }
    if non_empty(assessment, "conversion_id").is_none() {
        errors.push("conversion_id must be a non-empty string".to_owned());
    }
    if non_empty(assessment, "quantity_kind") != Some("food_quantity") {
        errors.push("quantity_kind must be food_quantity; nutrient retention is not modelled here".to_owned());
    }
    let claim_status = non_empty(assessment, "claim_status");
    if !matches!(claim_status, Some("calculation_only" | "review_required" | "qualified")) {
        errors.push("claim_status must be calculation_only, review_required, or qualified".to_owned());
    }
    if non_empty(assessment, "rationale").is_none() {
        errors.push("rationale must state the intended transformation and its limits".to_owned());
    }

    let null = Value::Null;
    let input = assessment.get("input").unwrap_or(&null);
    let input_status = validate_evidence(input, "input", &mut errors);
    let input_unit = non_empty(input, "unit");
    let input_basis = non_empty(input, "basis");
    let input_state = non_empty(input, "food_state");
    if input_unit.is_none() {
        errors.push("input.unit is required".to_owned());
    }
    if input_basis.is_none() {
        errors.push("input.basis is required".to_owned());
    }
    if input_state.is_none() {
        errors.push("input.food_state is required".to_owned());
    }
    if let Some(unit) = input_unit {
        if unit_definition(unit).is_none() {
            errors.push(format!("unsupported input unit {unit:?}; household measures need a food-specific mass/volume reference"));
        }
    }

    let input_amount = number(input, "amount");
    let input_lower = number(input, "lower");
    let input_upper = number(input, "upper");
    for (field, value) in [
        ("input.amount", input_amount),
        ("input.lower", input_lower),
        ("input.upper", input_upper),
    ] {
        if value.is_none() {
            errors.push(format!("{field} must be a finite JSON number"));
        }
    }
    let mut amount = input_amount.unwrap_or(0.0);
    let mut lower = input_lower.unwrap_or(0.0);
    let mut upper = input_upper.unwrap_or(0.0);
    if amount < 0.0 || lower < 0.0 || upper < 0.0 || lower > amount || amount > upper {
        errors.push("input interval must satisfy 0 <= lower <= amount <= upper".to_owned());
    }

    let mut current_unit = input_unit.unwrap_or("<missing>").to_owned();
    let mut current_basis = input_basis.unwrap_or("<missing>").to_owned();
    let mut current_state = input_state.unwrap_or("<missing>").to_owned();
    let mut step_ids = BTreeSet::new();
    let mut evidence_statuses = Vec::new();
    if let Some(status) = input_status {
        evidence_statuses.push(status.to_owned());
    }

    let steps: &[Value] = match assessment.get("steps").and_then(Value::as_array) {
        Some(steps) => steps.as_slice(),
        None => {
            errors.push("steps must be an array".to_owned());
            &[]
        }
    };
    if steps.is_empty() {
        errors.push("steps must contain at least one explicit transformation".to_owned());
    }

    for (index, step) in steps.iter().enumerate() {
        let label = format!("steps[{index}]");
        if let Some(status) = validate_evidence(step, &label, &mut errors) {
            evidence_statuses.push(status.to_owned());
        }
        let Some(step_id) = non_empty(step, "step_id") else {
            errors.push(format!("{label}.step_id must be non-empty"));
            continue;
        };
        if !step_ids.insert(step_id.to_owned()) {
            errors.push(format!("duplicate step_id {step_id:?}"));
        }
        let Some(kind) = non_empty(step, "kind") else {
            errors.push(format!("{label}.kind is required"));
            continue;
        };
        let Some(input_unit) = non_empty(step, "input_unit") else {
            errors.push(format!("{label}.input_unit is required"));
            continue;
        };
        let Some(output_unit) = non_empty(step, "output_unit") else {
            errors.push(format!("{label}.output_unit is required"));
            continue;
        };
        let Some(input_basis) = non_empty(step, "input_basis") else {
            errors.push(format!("{label}.input_basis is required"));
            continue;
        };
        let Some(output_basis) = non_empty(step, "output_basis") else {
            errors.push(format!("{label}.output_basis is required"));
            continue;
        };
        let Some(input_state) = non_empty(step, "input_state") else {
            errors.push(format!("{label}.input_state is required"));
            continue;
        };
        let Some(output_state) = non_empty(step, "output_state") else {
            errors.push(format!("{label}.output_state is required"));
            continue;
        };

        if input_unit != current_unit {
            errors.push(format!("{label} input unit {input_unit:?} does not continue prior unit {current_unit:?}"));
        }
        if input_basis != current_basis {
            errors.push(format!("{label} input basis {input_basis:?} does not continue prior basis {current_basis:?}"));
        }
        if input_state != current_state {
            errors.push(format!("{label} input state {input_state:?} does not continue prior food state {current_state:?}"));
        }

        let factor = number(step, "factor");
        let factor_lower = number(step, "factor_lower");
        let factor_upper = number(step, "factor_upper");
        if factor.is_none() || factor_lower.is_none() || factor_upper.is_none() {
            errors.push(format!("{label} factor/factor_lower/factor_upper must be finite numbers"));
        }
        let (factor, factor_lower, factor_upper) = (
            factor.unwrap_or(0.0),
            factor_lower.unwrap_or(0.0),
            factor_upper.unwrap_or(0.0),
        );
        if factor_lower <= 0.0
            || factor_lower > factor
            || factor > factor_upper
            || !factor_upper.is_finite()
        {
            errors.push(format!("{label} factor interval must satisfy 0 < factor_lower <= factor <= factor_upper"));
        }

        let input_definition = unit_definition(input_unit);
        let output_definition = unit_definition(output_unit);
        match kind {
            "unit_scale" => {
                match (input_definition, output_definition) {
                    (Some((input_dimension, input_scale)), Some((output_dimension, output_scale))) => {
                        if input_dimension != output_dimension {
                            errors.push(format!("{label} unit_scale cannot cross dimensions"));
                        } else {
                            let expected = input_scale / output_scale;
                            if !nearly_equal(factor, expected)
                                || !nearly_equal(factor_lower, expected)
                                || !nearly_equal(factor_upper, expected)
                            {
                                errors.push(format!("{label} unit_scale factor must exactly match the versioned unit table (expected {expected})"));
                            }
                        }
                    }
                    _ => errors.push(format!("{label} unit_scale uses an unsupported unit")),
                }
                if input_basis != output_basis || input_state != output_state {
                    errors.push(format!("{label} unit_scale must preserve food basis and state"));
                }
            }
            "edible_fraction" => {
                if input_definition.map(|v| v.0) != Some("mass")
                    || output_definition.map(|v| v.0) != Some("mass")
                {
                    errors.push(format!("{label} edible_fraction requires mass units"));
                }
                if input_basis != "as_purchased" || output_basis != "edible_portion" {
                    errors.push(format!("{label} edible_fraction must map as_purchased to edible_portion"));
                }
                if input_state != output_state {
                    errors.push(format!("{label} edible_fraction must preserve food state"));
                }
                if factor_upper > 1.0 {
                    errors.push(format!("{label} edible_fraction cannot exceed 1.0"));
                }
            }
            "process_yield" => {
                if input_definition.map(|v| v.0) != Some("mass")
                    || output_definition.map(|v| v.0) != Some("mass")
                {
                    errors.push(format!("{label} process_yield requires mass units"));
                }
                if input_basis != output_basis {
                    errors.push(format!("{label} process_yield must preserve the declared denominator basis"));
                }
                if input_state == output_state {
                    errors.push(format!("{label} process_yield must declare a changed food state"));
                }
            }
            "density" => {
                if input_definition.map(|v| v.0) != Some("volume")
                    || output_definition.map(|v| v.0) != Some("mass")
                {
                    errors.push(format!("{label} density must convert volume to mass"));
                }
                if input_basis != output_basis || input_state != output_state {
                    errors.push(format!("{label} density must preserve food basis and state"));
                }
            }
            _ => errors.push(format!("{label} unsupported transformation kind {kind:?}")),
        }

        // Positive-factor interval multiplication conservatively bounds each
        // sequential output, assuming each declared interval contains its factor.
        // Correlation/calibration quality must be assessed separately.
        if factor > 0.0 && factor_lower > 0.0 && factor_upper >= factor_lower {
            amount *= factor;
            lower *= factor_lower;
            upper *= factor_upper;
        }
        current_unit = output_unit.to_owned();
        current_basis = output_basis.to_owned();
        current_state = output_state.to_owned();
    }

    if claim_status == Some("qualified")
        && evidence_statuses.iter().any(|status| status != "qualified")
    {
        errors.push("claim_status qualified requires qualified input and qualified evidence for every step".to_owned());
    }

    let expected = assessment.get("expected_output").unwrap_or(&null);
    let expected_amount = number(expected, "amount");
    let expected_lower = number(expected, "lower");
    let expected_upper = number(expected, "upper");
    for (field, value) in [
        ("expected_output.amount", expected_amount),
        ("expected_output.lower", expected_lower),
        ("expected_output.upper", expected_upper),
    ] {
        if value.is_none() {
            errors.push(format!("{field} must be a finite JSON number"));
        }
    }
    if let (Some(expected_amount), Some(expected_lower), Some(expected_upper)) =
        (expected_amount, expected_lower, expected_upper)
    {
        if expected_lower < 0.0
            || expected_lower > expected_amount
            || expected_amount > expected_upper
        {
            errors.push("expected_output interval must satisfy 0 <= lower <= amount <= upper".to_owned());
        }
        if !nearly_equal(amount, expected_amount)
            || !nearly_equal(lower, expected_lower)
            || !nearly_equal(upper, expected_upper)
        {
            errors.push(format!(
                "expected_output disagrees with recomputed amount/interval (computed {amount} [{lower}, {upper}])"
            ));
        }
    }

    for (field, computed) in [
        ("unit", current_unit.as_str()),
        ("basis", current_basis.as_str()),
        ("food_state", current_state.as_str()),
    ] {
        if non_empty(expected, field) != Some(computed) {
            errors.push(format!("expected_output.{field} must equal the final transformation output"));
        }
    }

    if errors.is_empty() {
        Ok(FoodQuantityConversionResultV1 {
            amount,
            lower,
            upper,
            unit: current_unit,
            basis: current_basis,
            food_state: current_state,
        })
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../fixtures/food-quantity-conversion-v1/estimated-potato-yield.json"
        ))
        .expect("synthetic conversion fixture must parse")
    }

    #[test]
    fn computes_a_source_pinned_chain_and_propagates_bounds() {
        let result = validate_food_quantity_conversion_v1(&fixture()).unwrap();
        assert!(nearly_equal(result.amount, 0.18655));
        assert!(nearly_equal(result.lower, 0.176));
        assert!(nearly_equal(result.upper, 0.1974));
        assert_eq!(result.unit, "kg");
        assert_eq!(result.basis, "edible_portion");
        assert_eq!(result.food_state, "boiled");
    }

    #[test]
    fn output_is_recomputed_not_trusted() {
        let mut value = fixture();
        value["expected_output"]["amount"] = json!(0.2);
        assert!(validate_food_quantity_conversion_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("disagrees with recomputed")));
    }

    #[test]
    fn variable_factors_require_source_provenance() {
        let mut value = fixture();
        value["steps"][0]["source_ref"] = Value::Null;
        assert!(validate_food_quantity_conversion_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("source_ref")));
    }

    #[test]
    fn metric_unit_scale_must_match_the_pinned_unit_table() {
        let mut value = fixture();
        value["steps"][2]["factor"] = json!(0.01);
        value["steps"][2]["factor_lower"] = json!(0.01);
        value["steps"][2]["factor_upper"] = json!(0.01);
        assert!(validate_food_quantity_conversion_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("versioned unit table")));
    }

    #[test]
    fn factor_uncertainty_must_contain_the_point_estimate() {
        let mut value = fixture();
        value["steps"][0]["factor_lower"] = json!(0.9);
        value["steps"][0]["factor_upper"] = json!(0.95);
        assert!(validate_food_quantity_conversion_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("factor interval")));
    }

    #[test]
    fn qualified_claim_cannot_use_synthetic_yield_evidence() {
        let mut value = fixture();
        value["claim_status"] = json!("qualified");
        assert!(validate_food_quantity_conversion_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("requires qualified input and qualified evidence")));
    }

    #[test]
    fn broken_basis_chain_is_rejected() {
        let mut value = fixture();
        value["steps"][1]["input_basis"] = json!("dry_matter");
        assert!(validate_food_quantity_conversion_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("does not continue prior basis")));
    }

    #[test]
    fn edible_fraction_cannot_exceed_one() {
        let mut value = fixture();
        value["steps"][0]["factor"] = json!(1.2);
        value["steps"][0]["factor_lower"] = json!(1.1);
        value["steps"][0]["factor_upper"] = json!(1.3);
        assert!(validate_food_quantity_conversion_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("cannot exceed 1.0")));
    }

    #[test]
    fn process_yield_must_represent_a_state_change() {
        let mut value = fixture();
        value["steps"][1]["output_state"] = json!("raw");
        assert!(validate_food_quantity_conversion_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("must declare a changed food state")));
    }

    #[test]
    fn household_volume_units_are_not_silently_accepted() {
        let mut value = fixture();
        value["input"]["unit"] = json!("cup");
        assert!(validate_food_quantity_conversion_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("unsupported input unit")));
    }
}
