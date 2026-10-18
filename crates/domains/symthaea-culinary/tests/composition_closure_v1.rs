use serde_json::{Value, json};
use symthaea_culinary::composition_closure::{
    CompositionClosureDispositionV1 as D, check_proximate_composition_closure_v1,
};

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../fixtures/composition-closure-v1/synthetic-profile.json"
    ))
    .expect("synthetic composition profile must parse")
}

#[test]
fn closure_classification_is_recomputed_from_component_intervals() {
    let result = check_proximate_composition_closure_v1(&fixture()).unwrap();
    assert_eq!(result.disposition, D::WithinPreferred);
    assert_eq!(result.point_sum_g_per_100g, Some(101.0));
    assert!((result.lower_sum_g_per_100g - 99.6).abs() < 1.0e-9);
    assert!((result.upper_sum_g_per_100g.unwrap() - 102.4).abs() < 1.0e-9);
}

#[test]
fn unavailable_component_cannot_be_treated_as_zero() {
    let mut value = fixture();
    value["components"].as_array_mut().unwrap().pop();
    value["unavailable_components"] = json!([{
        "component_id":"alcohol","reason":"not_analyzed","detail":"test missingness"
    }]);
    let result = check_proximate_composition_closure_v1(&value).unwrap();
    assert_eq!(result.disposition, D::Indeterminate);
    assert!(result.upper_sum_g_per_100g.is_none());
}

#[test]
fn below_quantification_limit_has_interval_without_point_value() {
    let mut value = fixture();
    value["components"][6]["measurement_status"] = json!("below_quantification_limit");
    value["components"][6]["value"] = Value::Null;
    value["components"][6]["lower"] = json!(0.0);
    value["components"][6]["upper"] = json!(0.1);
    let result = check_proximate_composition_closure_v1(&value).unwrap();
    assert!(result.point_sum_g_per_100g.is_none());
    assert_eq!(result.disposition, D::WithinPreferred);
}

#[test]
fn selected_carbohydrate_convention_cannot_double_count_fibre() {
    let mut value = fixture();
    value["carbohydrate_basis"] = json!("total_includes_fibre");
    assert!(check_proximate_composition_closure_v1(&value).is_err());
}

#[test]
fn source_food_state_must_match_profile_state() {
    let mut value = fixture();
    value["components"][0]["food_state"] = json!("raw");
    assert!(check_proximate_composition_closure_v1(&value).is_err());
}

#[test]
fn synthetic_source_cannot_be_relabelled_qualified() {
    let mut value = fixture();
    value["components"][0]["evidence"]["evidence_status"] = json!("qualified");
    assert!(check_proximate_composition_closure_v1(&value).is_err());
}

#[test]
fn omission_without_explicit_unavailable_reason_is_rejected() {
    let mut value = fixture();
    value["components"].as_array_mut().unwrap().pop();
    let errors = check_proximate_composition_closure_v1(&value).unwrap_err();
    assert!(errors.iter().any(|error| error.contains("must be measured or explicitly listed unavailable")));
}

#[test]
fn duplicate_component_id_is_rejected() {
    let mut value = fixture();
    value["components"][1]["component_id"] = json!("water");
    assert!(check_proximate_composition_closure_v1(&value)
        .unwrap_err().iter().any(|error| error.contains("duplicate component entry")));
}

#[test]
fn reordering_components_does_not_change_closure() {
    let mut value = fixture();
    let expected = check_proximate_composition_closure_v1(&value).unwrap();
    value["components"].as_array_mut().unwrap().reverse();
    let actual = check_proximate_composition_closure_v1(&value).unwrap();
    assert_eq!(expected, actual);
}

#[test]
fn broad_uncertainty_must_not_strengthen_closure_disposition() {
    let mut value = fixture();
    value["components"][0]["lower"] = json!(56.0);
    value["components"][0]["upper"] = json!(68.0);
    let result = check_proximate_composition_closure_v1(&value).unwrap();
    assert_eq!(result.disposition, D::Indeterminate);
}
