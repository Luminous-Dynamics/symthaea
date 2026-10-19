// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-contract bindings between process-trace streams and nutrient accounting.
//!
//! This module does not recalculate nutrients or mass closure. It checks that the
//! process identity, final food state, final-product mass/evidence, and listed
//! ingredient masses/evidence refer to the same records in both contracts.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

use crate::nutrient_accounting::validate_nutrient_accounting_v1;
use crate::process_trace::validate_culinary_process_trace_v1;

/// A successful cross-contract consistency check; not a nutrition qualification.
#[derive(Debug, Clone, PartialEq)]
pub struct NutrientProcessBindingResultV1 {
    pub trace_id: String,
    pub process_id: String,
    pub final_product_stream_id: String,
    pub ingredient_binding_count: usize,
    pub final_food_mass_g: f64,
}

fn non_empty<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty())
}

fn finite_number(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64).filter(|v| v.is_finite())
}

fn interval_matches(left: &Value, right: &Value) -> bool {
    ["value", "lower", "upper"]
        .iter()
        .all(|key| finite_number(left, key).is_some() && finite_number(right, key).is_some()
            && finite_number(left, key) == finite_number(right, key))
}

fn evidence_matches(left: &Value, right: &Value) -> bool {
    ["source_ref", "source_version", "known_at", "evidence_status"]
        .iter()
        .all(|key| non_empty(left, key) == non_empty(right, key)
            && non_empty(left, key).is_some())
}

fn stream_by_id<'a>(trace: &'a Value, id: &str) -> Option<&'a Value> {
    trace
        .get("streams")?
        .as_array()?
        .iter()
        .find(|stream| non_empty(stream, "stream_id") == Some(id))
}

/// Validate a binding that joins a process trace, nutrient-accounting record,
/// and explicit cross-reference manifest.
///
/// Fail-closed joins:
/// - process IDs and prepared-food states must match;
/// - final_food_mass_g and final_mass_evidence must exactly match the chosen
///   terminal edible-product stream's mass interval and evidence record;
/// - every nutrient-accounting ingredient must map to a distinct external
///   ingredient/seasoning/oil input with the same mass interval and evidence;
/// - every trace boundary input marked ingredient/seasoning/oil must be mapped.
///
/// This gate does not authenticate evidence, evaluate nutrient retention, infer
/// water/oil absorption, calculate mass balance, or confer qualified status.
pub fn validate_nutrient_process_binding_v1(
    trace: &Value,
    assessment: &Value,
    binding: &Value,
) -> Result<NutrientProcessBindingResultV1, Vec<String>> {
    let mut errors = Vec::new();

    if let Err(mut nested) = validate_culinary_process_trace_v1(trace) {
        errors.extend(nested.drain(..).map(|e| format!("trace: {e}")));
    }
    if let Err(mut nested) = validate_nutrient_accounting_v1(assessment) {
        errors.extend(nested.drain(..).map(|e| format!("nutrient_accounting: {e}")));
    }

    if non_empty(binding, "schema_version") != Some("1.0.0") {
        errors.push("binding.schema_version must be 1.0.0".to_owned());
    }
    let trace_id = non_empty(trace, "trace_id");
    let process_id = non_empty(trace, "process_id");
    if trace_id.is_none() || non_empty(binding, "trace_id") != trace_id {
        errors.push("binding.trace_id must exactly match trace.trace_id".to_owned());
    }
    if process_id.is_none() || non_empty(assessment, "process_id") != process_id {
        errors.push("nutrient accounting process_id must match trace.process_id".to_owned());
    }
    if non_empty(assessment, "final_food_state") != non_empty(trace, "food_state_end")
        || non_empty(trace, "food_state_end").is_none()
    {
        errors.push("nutrient final_food_state must match trace.food_state_end".to_owned());
    }

    let final_stream_id = non_empty(binding, "final_product_stream_id");
    let final_stream = final_stream_id.and_then(|id| stream_by_id(trace, id));
    match (final_stream_id, final_stream) {
        (Some(id), Some(stream)) => {
            if non_empty(stream, "material_class") != Some("edible_product") {
                errors.push(format!("final stream {id:?} must have material_class edible_product"));
            }
            if non_empty(stream, "disposition") != Some("retained")
                || non_empty(stream, "to_step_id") != Some("external")
                || non_empty(stream, "from_step_id") == Some("external")
            {
                errors.push(format!("final stream {id:?} must be a retained terminal output"));
            }
            let trace_mass = stream.get("mass_g").unwrap_or(&Value::Null);
            if !matches!(non_empty(trace_mass, "status"), Some("measured" | "estimated")) {
                errors.push("selected final product mass must be measured or estimated, not unknown".to_owned());
            }
            if !interval_matches(
                assessment.get("final_food_mass_g").unwrap_or(&Value::Null),
                trace_mass,
            ) {
                errors.push("final_food_mass_g interval must exactly match selected product stream mass_g".to_owned());
            }
            if !evidence_matches(
                assessment.get("final_mass_evidence").unwrap_or(&Value::Null),
                stream.get("evidence").unwrap_or(&Value::Null),
            ) {
                errors.push("final_mass_evidence must exactly match selected product stream evidence".to_owned());
            }
        }
        (None, _) => errors.push("binding.final_product_stream_id is required".to_owned()),
        (_, None) => errors.push("binding final product stream ID does not resolve".to_owned()),
    }

    let ingredients: &[Value] = match assessment.get("ingredients").and_then(Value::as_array) {
        Some(items) => items.as_slice(),
        None => {
            errors.push("nutrient accounting ingredients must be an array".to_owned());
            &[]
        }
    };
    let mut ingredient_by_id = BTreeMap::<String, &Value>::new();
    for ingredient in ingredients {
        if let Some(id) = non_empty(ingredient, "ingredient_id") {
            ingredient_by_id.insert(id.to_owned(), ingredient);
        }
    }

    let mappings: &[Value] = match binding.get("ingredient_stream_bindings").and_then(Value::as_array) {
        Some(items) => items.as_slice(),
        None => {
            errors.push("binding.ingredient_stream_bindings must be an array".to_owned());
            &[]
        }
    };

    let mut bound_ingredients = BTreeSet::<String>::new();
    let mut bound_streams = BTreeSet::<String>::new();
    const NUTRIENT_BEARING_INPUTS: &[&str] = &["ingredient", "seasoning", "added_oil"];

    for (index, mapping) in mappings.iter().enumerate() {
        let label = format!("ingredient_stream_bindings[{index}]");
        let ingredient_id = non_empty(mapping, "ingredient_id");
        let stream_id = non_empty(mapping, "stream_id");
        let (Some(ingredient_id), Some(stream_id)) = (ingredient_id, stream_id) else {
            errors.push(format!("{label} requires non-empty ingredient_id and stream_id"));
            continue;
        };
        if !bound_ingredients.insert(ingredient_id.to_owned()) {
            errors.push(format!("ingredient {ingredient_id:?} is mapped more than once"));
        }
        if !bound_streams.insert(stream_id.to_owned()) {
            errors.push(format!("stream {stream_id:?} is mapped to more than one nutrient ingredient"));
        }

        let Some(ingredient) = ingredient_by_id.get(ingredient_id).copied() else {
            errors.push(format!("mapping refers to unknown nutrient ingredient {ingredient_id:?}"));
            continue;
        };
        let Some(stream) = stream_by_id(trace, stream_id) else {
            errors.push(format!("{label} refers to unknown stream {stream_id:?}"));
            continue;
        };
        if !NUTRIENT_BEARING_INPUTS.contains(&non_empty(stream, "material_class").unwrap_or("")) {
            errors.push(format!("stream {stream_id:?} is not a declared ingredient, seasoning, or added-oil input"));
        }
        if non_empty(stream, "from_step_id") != Some("external")
            || non_empty(stream, "to_step_id") == Some("external")
        {
            errors.push(format!("ingredient stream {stream_id:?} must be an external boundary input"));
        }
        if !matches!(non_empty(stream, "disposition"), Some("consumed" | "transformed")) {
            errors.push(format!("ingredient stream {stream_id:?} must be consumed or transformed"));
        }
        if !interval_matches(
            ingredient.get("edible_mass_g").unwrap_or(&Value::Null),
            stream.get("mass_g").unwrap_or(&Value::Null),
        ) {
            errors.push(format!("ingredient {ingredient_id:?} edible_mass_g must exactly match stream {stream_id:?} mass_g"));
        }
        if !evidence_matches(
            ingredient.get("mass_evidence").unwrap_or(&Value::Null),
            stream.get("evidence").unwrap_or(&Value::Null),
        ) {
            errors.push(format!("ingredient {ingredient_id:?} mass_evidence must exactly match stream {stream_id:?} evidence"));
        }
    }

    for ingredient_id in ingredient_by_id.keys() {
        if !bound_ingredients.contains(ingredient_id) {
            errors.push(format!("nutrient ingredient {ingredient_id:?} has no process-stream binding"));
        }
    }

    if let Some(streams) = trace.get("streams").and_then(Value::as_array) {
        for stream in streams {
            if non_empty(stream, "from_step_id") == Some("external")
                && NUTRIENT_BEARING_INPUTS.contains(&non_empty(stream, "material_class").unwrap_or(""))
            {
                let id = non_empty(stream, "stream_id").unwrap_or("<missing>");
                if !bound_streams.contains(id) {
                    errors.push(format!("nutrient-bearing boundary stream {id:?} is absent from the nutrient ingredient mapping"));
                }
            }
        }
    }

    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }

    let final_food_mass_g = final_stream
        .and_then(|stream| stream.get("mass_g"))
        .and_then(|mass| finite_number(mass, "value"))
        .unwrap_or(0.0);

    Ok(NutrientProcessBindingResultV1 {
        trace_id: trace_id.unwrap_or_default().to_owned(),
        process_id: process_id.unwrap_or_default().to_owned(),
        final_product_stream_id: final_stream_id.unwrap_or_default().to_owned(),
        ingredient_binding_count: mappings.len(),
        final_food_mass_g,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(path: &str) -> Value {
        serde_json::from_str(path).unwrap()
    }

    fn valid_inputs() -> (Value, Value, Value) {
        let trace: Value = read(include_str!(
            "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-trace.json"
        ));
        let assessment: Value = read(include_str!(
            "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-nutrients.json"
        ));
        let binding: Value = read(include_str!(
            "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-binding.json"
        ));
        (trace, assessment, binding)
    }

    #[test]
    fn matching_contracts_bind_process_state_masses_and_evidence() {
        let (trace, assessment, binding) = valid_inputs();
        let result = validate_nutrient_process_binding_v1(&trace, &assessment, &binding).unwrap();
        assert_eq!(result.trace_id, "synthetic-rice-pea-process-trace-v1");
        assert_eq!(result.process_id, "synthetic-boil-with-water-v1");
        assert_eq!(result.final_product_stream_id, "cooked-meal");
        assert_eq!(result.ingredient_binding_count, 2);
        assert_eq!(result.final_food_mass_g, 260.0);
    }

    #[test]
    fn process_id_mismatch_fails_closed() {
        let (trace, mut assessment, binding) = valid_inputs();
        assessment["process_id"] = Value::String("other-process".into());
        assert!(validate_nutrient_process_binding_v1(&trace, &assessment, &binding).is_err());
    }

    #[test]
    fn final_food_state_mismatch_fails_closed() {
        let (trace, mut assessment, binding) = valid_inputs();
        assessment["final_food_state"] = Value::String("raw".into());
        assert!(validate_nutrient_process_binding_v1(&trace, &assessment, &binding).is_err());
    }

    #[test]
    fn final_mass_interval_mismatch_fails_closed() {
        let (trace, mut assessment, binding) = valid_inputs();
        assessment["final_food_mass_g"]["value"] = serde_json::json!(261.0);
        assert!(validate_nutrient_process_binding_v1(&trace, &assessment, &binding).is_err());
    }

    #[test]
    fn final_mass_evidence_mismatch_fails_closed() {
        let (trace, mut assessment, binding) = valid_inputs();
        assessment["final_mass_evidence"]["source_ref"] = Value::String("urn:synthetic:other".into());
        assert!(validate_nutrient_process_binding_v1(&trace, &assessment, &binding).is_err());
    }

    #[test]
    fn ingredient_mass_mismatch_fails_closed() {
        let (trace, mut assessment, binding) = valid_inputs();
        assessment["ingredients"][0]["edible_mass_g"]["value"] = serde_json::json!(151.0);
        assert!(validate_nutrient_process_binding_v1(&trace, &assessment, &binding).is_err());
    }

    #[test]
    fn ingredient_mass_evidence_mismatch_fails_closed() {
        let (trace, mut assessment, binding) = valid_inputs();
        assessment["ingredients"][0]["mass_evidence"]["source_ref"] = Value::String("urn:synthetic:other".into());
        assert!(validate_nutrient_process_binding_v1(&trace, &assessment, &binding).is_err());
    }

    #[test]
    fn missing_ingredient_binding_fails_closed() {
        let (trace, assessment, mut binding) = valid_inputs();
        binding["ingredient_stream_bindings"].as_array_mut().unwrap().pop();
        assert!(validate_nutrient_process_binding_v1(&trace, &assessment, &binding).is_err());
    }

    #[test]
    fn duplicate_stream_binding_fails_closed() {
        let (trace, assessment, mut binding) = valid_inputs();
        binding["ingredient_stream_bindings"][1]["stream_id"] =
            binding["ingredient_stream_bindings"][0]["stream_id"].clone();
        assert!(validate_nutrient_process_binding_v1(&trace, &assessment, &binding).is_err());
    }

    #[test]
    fn non_product_final_stream_fails_closed() {
        let (trace, assessment, mut binding) = valid_inputs();
        binding["final_product_stream_id"] = Value::String("evaporated-water".into());
        assert!(validate_nutrient_process_binding_v1(&trace, &assessment, &binding).is_err());
    }

    #[test]
    fn wrong_ingredient_stream_type_fails_closed() {
        let (mut trace, assessment, binding) = valid_inputs();
        trace["streams"][0]["material_class"] = Value::String("added_water".into());
        assert!(validate_nutrient_process_binding_v1(&trace, &assessment, &binding).is_err());
    }

    #[test]
    fn unrelated_boundary_ingredient_cannot_be_silently_omitted() {
        let (mut trace, assessment, binding) = valid_inputs();
        let mut extra = trace["streams"][0].clone();
        extra["stream_id"] = Value::String("extra-ingredient".into());
        trace["streams"].as_array_mut().unwrap().push(extra);
        trace["steps"][0]["input_stream_ids"].as_array_mut().unwrap()
            .push(Value::String("extra-ingredient".into()));
        assert!(validate_nutrient_process_binding_v1(&trace, &assessment, &binding).is_err());
    }
}
