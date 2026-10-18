// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bearing recipe nutrient accounting.
//!
//! This module separates nutrient amounts from food mass. Ingredient composition
//! is expressed per 100 g edible portion; per-ingredient nutrient-retention
//! factors change nutrient amount; measured final recipe mass is used only for
//! normalization. Yield must not be applied again to nutrient amount.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// A computed nutrient total and concentration per 100 g of finished recipe.
#[derive(Debug, Clone, PartialEq)]
pub struct NutrientOutputV1 {
    pub nutrient_id: String,
    pub unit: String,
    pub retained_total: f64,
    pub retained_lower: f64,
    pub retained_upper: f64,
    pub per_100g_final: f64,
    pub per_100g_lower: f64,
    pub per_100g_upper: f64,
}

/// Validated outputs for a NutrientAccountingV1 calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct NutrientAccountingResultV1 {
    pub outputs: Vec<NutrientOutputV1>,
}

fn non_empty<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
}

fn number(value: &Value, key: &str) -> Option<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
}

fn interval(
    value: &Value,
    label: &str,
    errors: &mut Vec<String>,
    strictly_positive_lower: bool,
) -> Option<(f64, f64, f64)> {
    let point = number(value, "value");
    let lower = number(value, "lower");
    let upper = number(value, "upper");
    if point.is_none() || lower.is_none() || upper.is_none() {
        errors.push(format!("{label} requires finite value/lower/upper numbers"));
        return None;
    }
    let (point, lower, upper) = (point.unwrap(), lower.unwrap(), upper.unwrap());
    if lower < 0.0
        || point < lower
        || point > upper
        || (strictly_positive_lower && lower <= 0.0)
    {
        let condition = if strictly_positive_lower {
            "0 < lower <= value <= upper"
        } else {
            "0 <= lower <= value <= upper"
        };
        errors.push(format!("{label} interval must satisfy {condition}"));
        return None;
    }
    Some((point, lower, upper))
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
    let parse = |start: usize, end: usize| {
        std::str::from_utf8(&bytes[start..end])
            .ok()
            .and_then(|text| text.parse::<u32>().ok())
    };
    let (Some(year), Some(month), Some(day), Some(hour), Some(minute), Some(second)) = (
        parse(0, 4), parse(5, 7), parse(8, 10),
        parse(11, 13), parse(14, 16), parse(17, 19),
    ) else {
        return false;
    };
    if !(1..=12).contains(&month)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return false;
    }
    let leap_year = year % 400 == 0 || (year % 4 == 0 && year % 100 != 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return false,
    };
    if day == 0 || day > days_in_month {
        return false;
    }

    let timezone_start = if bytes.last() == Some(&b'Z') {
        bytes.len() - 1
    } else {
        if bytes.len() < 25 {
            return false;
        }
        let start = bytes.len() - 6;
        if !matches!(bytes[start], b'+' | b'-')
            || bytes[start + 3] != b':'
        {
            return false;
        }
        let (Some(offset_hour), Some(offset_minute)) = (
            parse(start + 1, start + 3),
            parse(start + 4, start + 6),
        ) else {
            return false;
        };
        if offset_hour > 23 || offset_minute > 59 {
            return false;
        }
        start
    };

    if timezone_start == 19 {
        return true;
    }
    bytes.get(19) == Some(&b'.')
        && timezone_start > 20
        && bytes[20..timezone_start].iter().all(|byte| byte.is_ascii_digit())
}

/// Returns true only for complete, non-synthetic qualified evidence.
fn validate_evidence(
    evidence: &Value,
    label: &str,
    errors: &mut Vec<String>,
) -> bool {
    let source_ref = non_empty(evidence, "source_ref");
    if source_ref.is_none() {
        errors.push(format!("{label}.source_ref is required"));
    }
    if non_empty(evidence, "source_version").is_none() {
        errors.push(format!("{label}.source_version is required"));
    }
    match non_empty(evidence, "known_at") {
        Some(timestamp) if timestamp_looks_rfc3339(timestamp) => {}
        _ => errors.push(format!("{label}.known_at must be a structurally valid RFC3339 timestamp")),
    }
    let status = non_empty(evidence, "evidence_status");
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
        && non_empty(evidence, "source_version").is_some()
        && non_empty(evidence, "known_at").is_some_and(timestamp_looks_rfc3339)
}

fn nearly_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-9_f64.max(1.0e-9 * left.abs().max(right.abs()))
}

type Interval = (f64, f64, f64);
type IngredientNutrients = BTreeMap<String, (String, Interval)>;

/// Validate and recompute a NutrientAccountingV1 recipe calculation.
///
/// Composition values are quantities per 100 g edible ingredient. The retained
/// amount for each ingredient is composition × edible mass / 100 × nutrient
/// retention factor. Totals are then normalized by final measured recipe mass.
/// The validator does not authenticate sources, select retention factors, or
/// establish empirical accuracy, legal-label suitability, or clinical safety.
pub fn validate_nutrient_accounting_v1(
    assessment: &Value,
) -> Result<NutrientAccountingResultV1, Vec<String>> {
    let mut errors = Vec::new();
    let mut all_evidence_qualified = true;

    if assessment.get("schema_version").and_then(Value::as_str) != Some("1.0.0") {
        errors.push("schema_version must be 1.0.0".to_owned());
    }
    if non_empty(assessment, "calculation_id").is_none() {
        errors.push("calculation_id is required".to_owned());
    }
    if non_empty(assessment, "process_id").is_none() {
        errors.push("process_id is required to scope retention evidence".to_owned());
    }
    if non_empty(assessment, "final_food_state").is_none() {
        errors.push("final_food_state is required to scope the prepared dish".to_owned());
    }
    let final_food_state = non_empty(assessment, "final_food_state");
    let process_id = non_empty(assessment, "process_id");
    let claim_status = non_empty(assessment, "claim_status");
    if !matches!(claim_status, Some("calculation_only" | "review_required" | "qualified")) {
        errors.push("claim_status must be calculation_only, review_required, or qualified".to_owned());
    }
    if non_empty(assessment, "rationale").is_none() {
        errors.push("rationale must document assumptions and limits".to_owned());
    }

    let null = Value::Null;
    let ingredients: &[Value] = match assessment.get("ingredients").and_then(Value::as_array) {
        Some(ingredients) if !ingredients.is_empty() => ingredients.as_slice(),
        _ => {
            errors.push("ingredients must be a non-empty array".to_owned());
            &[]
        }
    };
    let mut ingredient_ids = BTreeSet::new();
    let mut masses: BTreeMap<String, Interval> = BTreeMap::new();
    let mut nutrient_by_ingredient: BTreeMap<String, IngredientNutrients> = BTreeMap::new();

    for (index, ingredient) in ingredients.iter().enumerate() {
        let label = format!("ingredients[{index}]");
        let Some(ingredient_id) = non_empty(ingredient, "ingredient_id") else {
            errors.push(format!("{label}.ingredient_id is required"));
            continue;
        };
        if !ingredient_ids.insert(ingredient_id.to_owned()) {
            errors.push(format!("duplicate ingredient_id {ingredient_id:?}"));
            continue;
        }
        let mass = interval(
            ingredient.get("edible_mass_g").unwrap_or(&null),
            &format!("{label}.edible_mass_g"),
            &mut errors,
            true,
        );
        if let Some(mass) = mass {
            masses.insert(ingredient_id.to_owned(), mass);
        }
        if !validate_evidence(
            ingredient.get("mass_evidence").unwrap_or(&null),
            &format!("{label}.mass_evidence"),
            &mut errors,
        ) {
            all_evidence_qualified = false;
        }
        if non_empty(ingredient, "composition_basis") != Some("per_100g_edible") {
            errors.push(format!("{label}.composition_basis must be per_100g_edible"));
        }
        if non_empty(ingredient, "composition_state").is_none() {
            errors.push(format!("{label}.composition_state is required to bind composition data to a food state"));
        }
        if !validate_evidence(
            ingredient.get("composition_source").unwrap_or(&null),
            &format!("{label}.composition_source"),
            &mut errors,
        ) {
            all_evidence_qualified = false;
        }

        let mut nutrient_map = IngredientNutrients::new();
        match ingredient.get("nutrients").and_then(Value::as_array) {
            Some(nutrients) if !nutrients.is_empty() => {
                for (nutrient_index, nutrient) in nutrients.iter().enumerate() {
                    let nutrient_label = format!("{label}.nutrients[{nutrient_index}]");
                    let Some(nutrient_id) = non_empty(nutrient, "nutrient_id") else {
                        errors.push(format!("{nutrient_label}.nutrient_id is required"));
                        continue;
                    };
                    let Some(unit) = non_empty(nutrient, "unit") else {
                        errors.push(format!("{nutrient_label}.unit is required"));
                        continue;
                    };
                    let Some(value) = interval(
                        nutrient.get("per_100g").unwrap_or(&null),
                        &format!("{nutrient_label}.per_100g"),
                        &mut errors,
                        false,
                    ) else {
                        continue;
                    };
                    if nutrient_map
                        .insert(nutrient_id.to_owned(), (unit.to_owned(), value))
                        .is_some()
                    {
                        errors.push(format!(
                            "{label} contains duplicate nutrient_id {nutrient_id:?}"
                        ));
                    }
                }
            }
            _ => errors.push(format!("{label}.nutrients must be a non-empty array")),
        }
        nutrient_by_ingredient.insert(ingredient_id.to_owned(), nutrient_map);
    }

    let final_mass = interval(
        assessment.get("final_food_mass_g").unwrap_or(&null),
        "final_food_mass_g",
        &mut errors,
        true,
    );
    if !validate_evidence(
        assessment.get("final_mass_evidence").unwrap_or(&null),
        "final_mass_evidence",
        &mut errors,
    ) {
        all_evidence_qualified = false;
    }

    let retentions: &[Value] = match assessment.get("nutrient_retention").and_then(Value::as_array) {
        Some(retentions) => retentions.as_slice(),
        None => {
            errors.push("nutrient_retention must be an array".to_owned());
            &[]
        }
    };
    let mut retention_by_key: BTreeMap<(String, String), Interval> = BTreeMap::new();
    for (index, retention) in retentions.iter().enumerate() {
        let label = format!("nutrient_retention[{index}]");
        let Some(ingredient_id) = non_empty(retention, "ingredient_id") else {
            errors.push(format!("{label}.ingredient_id is required"));
            continue;
        };
        let Some(nutrient_id) = non_empty(retention, "nutrient_id") else {
            errors.push(format!("{label}.nutrient_id is required"));
            continue;
        };
        if non_empty(retention, "process_id") != process_id {
            errors.push(format!("{label}.process_id does not match the recipe process_id"));
        }
        let retention_input_state = non_empty(retention, "input_state");
        let retention_output_state = non_empty(retention, "output_state");
        if retention_input_state.is_none() || retention_output_state.is_none() {
            errors.push(format!("{label} requires explicit input_state and output_state"));
        }
        if let Some(ingredient) = ingredients.iter().find(|candidate| {
            non_empty(candidate, "ingredient_id") == Some(ingredient_id)
        }) {
            if retention_input_state != non_empty(ingredient, "composition_state") {
                errors.push(format!("{label}.input_state does not match the ingredient composition_state"));
            }
        }
        if retention_output_state != final_food_state {
            errors.push(format!("{label}.output_state does not match final_food_state"));
        }
        let key = (ingredient_id.to_owned(), nutrient_id.to_owned());
        if !ingredient_ids.contains(ingredient_id) {
            errors.push(format!("{label} references unknown ingredient {ingredient_id:?}"));
        }
        let factor_value = number(retention, "value");
        let factor_lower = number(retention, "lower");
        let factor_upper = number(retention, "upper");
        let factor = match (factor_value, factor_lower, factor_upper) {
            (Some(value), Some(lower), Some(upper))
                if lower >= 0.0 && lower <= value && value <= upper && upper <= 1.0 =>
            {
                Some((value, lower, upper))
            }
            _ => {
                errors.push(format!("{label} factor interval must satisfy 0 <= lower <= value <= upper <= 1"));
                None
            }
        };
        if !validate_evidence(
            retention.get("evidence").unwrap_or(&null),
            &format!("{label}.evidence"),
            &mut errors,
        ) {
            all_evidence_qualified = false;
        }
        if let Some(factor) = factor {
            if retention_by_key.insert(key.clone(), factor).is_some() {
                errors.push(format!("duplicate retention entry for {key:?}"));
            }
        }
        if nutrient_by_ingredient
            .get(ingredient_id)
            .is_none_or(|nutrients| !nutrients.contains_key(nutrient_id))
        {
            errors.push(format!("{label} has no matching ingredient composition value"));
        }
    }

    if claim_status == Some("qualified") && !all_evidence_qualified {
        errors.push(
            "qualified calculation requires qualified evidence for every mass, composition and retention input"
                .to_owned(),
        );
    }

    let expected_outputs: &[Value] = match assessment.get("expected_outputs").and_then(Value::as_array) {
        Some(outputs) if !outputs.is_empty() => outputs.as_slice(),
        _ => {
            errors.push("expected_outputs must be a non-empty array".to_owned());
            &[]
        }
    };
    let mut output_ids = BTreeSet::new();
    let mut results = Vec::new();

    for (index, expected) in expected_outputs.iter().enumerate() {
        let label = format!("expected_outputs[{index}]");
        let Some(nutrient_id) = non_empty(expected, "nutrient_id") else {
            errors.push(format!("{label}.nutrient_id is required"));
            continue;
        };
        if !output_ids.insert(nutrient_id.to_owned()) {
            errors.push(format!("duplicate expected output for nutrient {nutrient_id:?}"));
            continue;
        }
        let Some(unit) = non_empty(expected, "unit") else {
            errors.push(format!("{label}.unit is required"));
            continue;
        };
        let Some(final_mass) = final_mass else {
            errors.push("cannot calculate normalized nutrient values without a valid final food mass".to_owned());
            continue;
        };

        let mut total = 0.0;
        let mut total_lower = 0.0;
        let mut total_upper = 0.0;
        let mut complete = true;
        for ingredient_id in &ingredient_ids {
            let Some((ingredient_unit, composition)) = nutrient_by_ingredient
                .get(ingredient_id)
                .and_then(|nutrients| nutrients.get(nutrient_id))
            else {
                errors.push(format!(
                    "nutrient {nutrient_id:?} is missing for ingredient {ingredient_id:?}; absence is not assumed to mean zero"
                ));
                complete = false;
                continue;
            };
            if ingredient_unit != unit {
                errors.push(format!(
                    "nutrient {nutrient_id:?} has inconsistent units {ingredient_unit:?} and {unit:?}"
                ));
                complete = false;
                continue;
            }
            let Some(mass) = masses.get(ingredient_id).copied() else {
                errors.push(format!("ingredient {ingredient_id:?} has no valid edible mass"));
                complete = false;
                continue;
            };
            let Some(factor) = retention_by_key.get(&(ingredient_id.clone(), nutrient_id.to_owned())).copied() else {
                errors.push(format!(
                    "missing retention factor for ingredient {ingredient_id:?}, nutrient {nutrient_id:?}"
                ));
                complete = false;
                continue;
            };

            let contribution = composition.0 * mass.0 / 100.0 * factor.0;
            let contribution_lower = composition.1 * mass.1 / 100.0 * factor.1;
            let contribution_upper = composition.2 * mass.2 / 100.0 * factor.2;
            total += contribution;
            total_lower += contribution_lower;
            total_upper += contribution_upper;
            if !total.is_finite() || !total_lower.is_finite() || !total_upper.is_finite() {
                errors.push(format!("{label} overflowed while aggregating nutrient contributions"));
                complete = false;
                break;
            }
        }
        if !complete {
            continue;
        }
        let per_100g_final = total * 100.0 / final_mass.0;
        let per_100g_lower = total_lower * 100.0 / final_mass.2;
        let per_100g_upper = total_upper * 100.0 / final_mass.1;
        if !per_100g_final.is_finite()
            || !per_100g_lower.is_finite()
            || !per_100g_upper.is_finite()
        {
            errors.push(format!("{label} normalized output is not finite"));
            continue;
        }
        if per_100g_lower > per_100g_final || per_100g_final > per_100g_upper {
            errors.push(format!("{label} computed uncertainty interval does not contain the point estimate"));
            continue;
        }

        let retained_expected = interval(
            expected.get("retained_total").unwrap_or(&null),
            &format!("{label}.retained_total"),
            &mut errors,
            false,
        );
        let concentration_expected = interval(
            expected.get("per_100g_final").unwrap_or(&null),
            &format!("{label}.per_100g_final"),
            &mut errors,
            false,
        );
        if let Some((value, lower, upper)) = retained_expected {
            if !nearly_equal(value, total)
                || !nearly_equal(lower, total_lower)
                || !nearly_equal(upper, total_upper)
            {
                errors.push(format!("{label}.retained_total disagrees with recomputation"));
            }
        }
        if let Some((value, lower, upper)) = concentration_expected {
            if !nearly_equal(value, per_100g_final)
                || !nearly_equal(lower, per_100g_lower)
                || !nearly_equal(upper, per_100g_upper)
            {
                errors.push(format!("{label}.per_100g_final disagrees with recomputation"));
            }
        }
        results.push(NutrientOutputV1 {
            nutrient_id: nutrient_id.to_owned(),
            unit: unit.to_owned(),
            retained_total: total,
            retained_lower: total_lower,
            retained_upper: total_upper,
            per_100g_final,
            per_100g_lower,
            per_100g_upper,
        });
    }

    if errors.is_empty() {
        results.sort_by(|left, right| left.nutrient_id.cmp(&right.nutrient_id));
        Ok(NutrientAccountingResultV1 { outputs: results })
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
            "../fixtures/nutrient-accounting-v1/synthetic-rice-pea-recipe.json"
        ))
        .expect("synthetic nutrient-accounting fixture must parse")
    }

    #[test]
    fn computes_retained_nutrient_then_normalizes_by_final_recipe_mass() {
        let result = validate_nutrient_accounting_v1(&fixture()).unwrap();
        let protein = &result.outputs[0];
        assert_eq!(protein.nutrient_id, "protein");
        assert!(nearly_equal(protein.retained_total, 12.225));
        assert!(nearly_equal(protein.per_100g_final, 12.225 * 100.0 / 330.0));
    }

    #[test]
    fn retained_nutrient_total_is_not_scaled_by_weight_yield_again() {
        let result = validate_nutrient_accounting_v1(&fixture()).unwrap();
        assert!(nearly_equal(result.outputs[0].retained_total, 12.225));
    }

    #[test]
    fn output_tampering_is_rejected() {
        let mut value = fixture();
        value["expected_outputs"][0]["retained_total"]["value"] = json!(999.0);
        assert!(validate_nutrient_accounting_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("retained_total disagrees")));
    }

    #[test]
    fn missing_nutrient_is_not_silently_assumed_to_be_zero() {
        let mut value = fixture();
        value["ingredients"][1]["nutrients"] = json!([]);
        assert!(validate_nutrient_accounting_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("missing for ingredient")));
    }

    #[test]
    fn final_mass_interval_must_be_positive() {
        let mut value = fixture();
        value["final_food_mass_g"]["lower"] = json!(0.0);
        assert!(validate_nutrient_accounting_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("interval must satisfy 0 <")));
    }

    #[test]
    fn retention_factors_cannot_exceed_one_in_v1() {
        let mut value = fixture();
        value["nutrient_retention"][0]["value"] = json!(1.1);
        value["nutrient_retention"][0]["upper"] = json!(1.2);
        assert!(validate_nutrient_accounting_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("factor interval must satisfy")));
    }

    #[test]
    fn missing_retention_factor_for_any_ingredient_fails_closed() {
        let mut value = fixture();
        value["nutrient_retention"].as_array_mut().unwrap().pop();
        assert!(validate_nutrient_accounting_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("missing retention factor")));
    }

    #[test]
    fn qualified_claim_cannot_use_synthetic_evidence() {
        let mut value = fixture();
        value["claim_status"] = json!("qualified");
        assert!(validate_nutrient_accounting_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("qualified calculation requires")));
    }

    #[test]
    fn malformed_evidence_timestamp_is_rejected() {
        let mut value = fixture();
        value["ingredients"][0]["mass_evidence"]["known_at"] = json!("sometime");
        assert!(validate_nutrient_accounting_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("RFC3339")));
    }

    #[test]
    fn duplicate_ingredient_ids_are_rejected() {
        let mut value = fixture();
        value["ingredients"][1]["ingredient_id"] = json!("rice");
        assert!(validate_nutrient_accounting_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("duplicate ingredient_id")));
    }

    #[test]
    fn every_retention_entry_must_refer_to_a_matching_composition() {
        let mut value = fixture();
        value["nutrient_retention"][0]["nutrient_id"] = json!("vitamin-c");
        assert!(validate_nutrient_accounting_v1(&value)
            .unwrap_err()
            .iter()
            .any(|error| error.contains("no matching ingredient composition")));
    }
}
