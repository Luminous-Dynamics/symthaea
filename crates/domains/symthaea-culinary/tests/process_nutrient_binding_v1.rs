// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use symthaea_culinary::process_nutrient_binding::validate_nutrient_process_binding_v1;

fn fixture(path: &str) -> Value {
    serde_json::from_str(path).unwrap()
}

#[test]
fn compatible_trace_and_nutrient_accounting_join_exactly() {
    let trace = fixture(include_str!(
        "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-trace.json"
    ));
    let assessment = fixture(include_str!(
        "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-nutrients.json"
    ));
    let binding = fixture(include_str!(
        "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-binding.json"
    ));
    let result = validate_nutrient_process_binding_v1(&trace, &assessment, &binding).unwrap();
    assert_eq!(result.final_food_mass_g, 260.0);
    assert_eq!(result.ingredient_binding_count, 2);
}

#[test]
fn process_identity_mismatch_is_not_comparable() {
    let trace = fixture(include_str!(
        "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-trace.json"
    ));
    let mut assessment = fixture(include_str!(
        "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-nutrients.json"
    ));
    let binding = fixture(include_str!(
        "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-binding.json"
    ));
    assessment["process_id"] = Value::String("different-batch".into());
    assert!(validate_nutrient_process_binding_v1(&trace, &assessment, &binding).is_err());
}

#[test]
fn final_product_mass_mismatch_is_not_comparable() {
    let trace = fixture(include_str!(
        "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-trace.json"
    ));
    let mut assessment = fixture(include_str!(
        "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-nutrients.json"
    ));
    let binding = fixture(include_str!(
        "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-binding.json"
    ));
    assessment["final_food_mass_g"]["value"] = serde_json::json!(261.0);
    assert!(validate_nutrient_process_binding_v1(&trace, &assessment, &binding).is_err());
}
