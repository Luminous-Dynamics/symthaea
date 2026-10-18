use serde_json::{Value, json};
use symthaea_culinary::food_conversion::validate_food_quantity_conversion_v1;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../fixtures/food-quantity-conversion-v1/estimated-potato-yield.json"
    ))
    .expect("synthetic conversion fixture must parse")
}

#[test]
fn synthetic_conversion_matches_independent_expected_arithmetic() {
    let result = validate_food_quantity_conversion_v1(&fixture()).unwrap();
    assert!((result.amount - 0.18655).abs() < 1.0e-9);
    assert!((result.lower - 0.176).abs() < 1.0e-9);
    assert!((result.upper - 0.1974).abs() < 1.0e-9);
}

#[test]
fn an_input_interval_that_does_not_contain_its_point_is_rejected() {
    let mut value = fixture();
    value["input"]["lower"] = json!(251.0);
    assert!(validate_food_quantity_conversion_v1(&value).is_err());
}

#[test]
fn conversion_cannot_skip_a_step_state() {
    let mut value = fixture();
    value["steps"][1]["input_state"] = json!("steamed");
    assert!(validate_food_quantity_conversion_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("does not continue prior food state")));
}

#[test]
fn a_density_step_must_convert_volume_to_mass() {
    let mut value = fixture();
    value["steps"][0]["kind"] = json!("density");
    assert!(validate_food_quantity_conversion_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("density must convert volume to mass")));
}

#[test]
fn unsupported_conversion_kinds_fail_closed() {
    let mut value = fixture();
    value["steps"][0]["kind"] = json!("magic_nutrition_factor");
    assert!(validate_food_quantity_conversion_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("unsupported transformation kind")));
}

#[test]
fn timestamps_are_required_for_evidence_as_known_at_a_time() {
    let mut value = fixture();
    value["steps"][1]["known_at"] = json!("sometime");
    assert!(validate_food_quantity_conversion_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("RFC3339 timestamp")));
}

#[test]
fn synthetic_source_locators_cannot_be_relabelled_qualified() {
    let mut value = fixture();
    value["claim_status"] = json!("qualified");
    value["input"]["evidence_status"] = json!("qualified");
    value["steps"][0]["evidence_status"] = json!("qualified");
    value["steps"][1]["evidence_status"] = json!("qualified");
    let errors = validate_food_quantity_conversion_v1(&value).unwrap_err();
    assert!(errors.iter().any(|error| error.contains("source_ref is synthetic")));
}

#[test]
fn output_interval_is_recomputed() {
    let mut value = fixture();
    value["expected_output"]["upper"] = json!(0.2);
    assert!(validate_food_quantity_conversion_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("disagrees with recomputed")));
}

#[test]
fn conversion_does_not_claim_nutrient_retention_support() {
    let mut value = fixture();
    value["quantity_kind"] = json!("nutrient_amount");
    assert!(validate_food_quantity_conversion_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("nutrient retention is not modelled")));
}

#[test]
fn duplicate_step_ids_are_rejected() {
    let mut value = fixture();
    value["steps"][1]["step_id"] = json!("trim-edible-fraction");
    assert!(validate_food_quantity_conversion_v1(&value)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("duplicate step_id")));
}
