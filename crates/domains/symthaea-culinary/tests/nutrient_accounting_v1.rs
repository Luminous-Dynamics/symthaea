use serde_json::{Value, json};
use symthaea_culinary::nutrient_accounting::validate_nutrient_accounting_v1;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../fixtures/nutrient-accounting-v1/synthetic-rice-pea-recipe.json"
    ))
    .expect("synthetic nutrient accounting fixture must parse")
}

#[test]
fn synthetic_recipe_recomputes_total_and_per_100g_output() {
    let result = validate_nutrient_accounting_v1(&fixture()).unwrap();
    assert_eq!(result.outputs.len(), 1);
    let protein = &result.outputs[0];
    assert_eq!(protein.nutrient_id, "protein");
    assert!((protein.retained_total - 12.225).abs() < 1.0e-9);
    assert!((protein.per_100g_final - 12.225 * 100.0 / 330.0).abs() < 1.0e-9);
}

#[test]
fn output_tampering_is_rejected() {
    let mut value = fixture();
    value["expected_outputs"][0]["per_100g_final"]["value"] = json!(99.0);
    assert!(validate_nutrient_accounting_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("per_100g_final disagrees")));
}

#[test]
fn missing_composition_is_not_assumed_to_be_zero() {
    let mut value = fixture();
    value["ingredients"][1]["nutrients"] = json!([]);
    assert!(validate_nutrient_accounting_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("nutrients must be a non-empty array")));
}

#[test]
fn every_ingredient_needs_a_retention_factor_for_each_requested_nutrient() {
    let mut value = fixture();
    value["nutrient_retention"].as_array_mut().unwrap().pop();
    assert!(validate_nutrient_accounting_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("missing retention factor")));
}

#[test]
fn a_retention_factor_above_one_is_rejected_in_v1() {
    let mut value = fixture();
    value["nutrient_retention"][0]["value"] = json!(1.1);
    value["nutrient_retention"][0]["upper"] = json!(1.2);
    assert!(validate_nutrient_accounting_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("factor interval must satisfy")));
}

#[test]
fn a_qualified_calculation_cannot_rely_on_synthetic_evidence() {
    let mut value = fixture();
    value["claim_status"] = json!("qualified");
    assert!(validate_nutrient_accounting_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("qualified calculation requires")));
}

#[test]
fn malformed_source_timestamps_are_rejected() {
    let mut value = fixture();
    value["final_mass_evidence"]["known_at"] = json!("sometime");
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
fn retention_must_reference_a_real_ingredient_and_composition() {
    let mut value = fixture();
    value["nutrient_retention"][0]["ingredient_id"] = json!("ghost");
    assert!(validate_nutrient_accounting_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("unknown ingredient")));
}

#[test]
fn inconsistent_units_for_the_same_nutrient_are_rejected() {
    let mut value = fixture();
    value["ingredients"][1]["nutrients"][0]["unit"] = json!("mg");
    assert!(validate_nutrient_accounting_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("inconsistent units")));
}

#[test]
fn final_food_mass_uncertainty_bounds_are_used_in_concentration() {
    let result = validate_nutrient_accounting_v1(&fixture()).unwrap();
    let output = &result.outputs[0];
    assert!(output.per_100g_lower < output.per_100g_final);
    assert!(output.per_100g_final < output.per_100g_upper);
}

#[test]
fn ingredient_order_does_not_change_result() {
    let mut value = fixture();
    let expected = validate_nutrient_accounting_v1(&value).unwrap();
    value["ingredients"].as_array_mut().unwrap().reverse();
    value["nutrient_retention"].as_array_mut().unwrap().reverse();
    let actual = validate_nutrient_accounting_v1(&value).unwrap();
    assert_eq!(expected, actual);
}

#[test]
fn retention_factor_must_be_pinned_to_the_same_process() {
    let mut value = fixture();
    value["nutrient_retention"][0]["process_id"] = json!("frying-v1");
    assert!(validate_nutrient_accounting_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("does not match the recipe process_id")));
}

#[test]
fn retention_factor_input_state_must_match_composition_state() {
    let mut value = fixture();
    value["nutrient_retention"][0]["input_state"] = json!("fried");
    assert!(validate_nutrient_accounting_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("does not match the ingredient composition_state")));
}

#[test]
fn retention_factor_output_state_must_match_final_recipe_state() {
    let mut value = fixture();
    value["nutrient_retention"][0]["output_state"] = json!("raw");
    assert!(validate_nutrient_accounting_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("does not match final_food_state")));
}
