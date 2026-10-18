#![deny(unsafe_code)]
// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent contract checks for display-manufacturing route profiles.
//! This validates planning metadata only; it does not execute processes.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const MANIFEST_TEXT: &str =
    include_str!("../../../docs/engineering/display-manufacturing-route-profiles-v0.1.json");
const SCHEMA_TEXT: &str =
    include_str!("../../../docs/engineering/display-manufacturing-route-profiles.schema.json");

fn parse(text: &str, name: &str, errors: &mut Vec<String>) -> Option<Value> {
    match serde_json::from_str(text) {
        Ok(value) => Some(value),
        Err(error) => {
            errors.push(format!("{name} JSON parse error: {error}"));
            None
        }
    }
}

fn validate_contract(manifest_text: &str, schema_text: &str) -> Vec<String> {
    let mut errors = Vec::new();
    let Some(schema) = parse(schema_text, "route schema", &mut errors) else {
        return errors;
    };
    let Some(manifest) = parse(manifest_text, "route manifest", &mut errors) else {
        return errors;
    };

    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        errors.push("route schema must declare JSON Schema Draft 2020-12".to_owned());
    }

    match jsonschema::draft202012::new(&schema) {
        Ok(validator) => {
            for error in validator.iter_errors(&manifest) {
                errors.push(format!("route JSON Schema violation: {error}"));
            }
        }
        Err(error) => errors.push(format!("route schema compilation failed: {error}")),
    }

    if manifest.get("corpus_id").and_then(Value::as_str) != Some("VIS-MFG-ROUTE-CORPUS-001") {
        errors.push("route corpus identity mismatch".to_owned());
    }
    if manifest.get("status").and_then(Value::as_str) != Some("UNQUALIFIED") {
        errors.push("route corpus must remain UNQUALIFIED".to_owned());
    }

    let sources: BTreeSet<String> = manifest
        .get("reference_sources")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|source| source.get("id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();
    let source_count = manifest
        .get("reference_sources")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    if sources.len() != source_count {
        errors.push("reference source IDs must be unique".to_owned());
    }

    let Some(routes) = manifest.get("routes").and_then(Value::as_array) else {
        errors.push("routes must be an array".to_owned());
        errors.sort();
        errors.dedup();
        return errors;
    };
    if routes.len() != 4 {
        errors.push("v0.1 route corpus must contain exactly four frozen route profiles".to_owned());
    }

    let mut route_ids = BTreeSet::new();
    let mut component_ids = BTreeSet::new();
    for route in routes {
        let route_id = route
            .get("route_id")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        if !route_ids.insert(route_id.to_owned()) {
            errors.push(format!("duplicate route ID: {route_id}"));
        }

        let route_sources: Vec<&str> = route
            .get("source_ids")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        if route_sources.is_empty()
            || route_sources
                .iter()
                .any(|source_id| !sources.contains(*source_id))
        {
            errors.push(format!(
                "route {route_id} has empty or unresolved source references"
            ));
        }

        let Some(components) = route.get("component_interfaces").and_then(Value::as_array) else {
            errors.push(format!(
                "route {route_id} component_interfaces must be an array"
            ));
            continue;
        };
        if components.len() < 4 {
            errors.push(format!(
                "route {route_id} has too few component interface contracts"
            ));
        }
        for component in components {
            let component_id = component
                .get("component_id")
                .and_then(Value::as_str)
                .unwrap_or("<missing>");
            if !component_ids.insert(component_id.to_owned()) {
                errors.push(format!("duplicate global component ID: {component_id}"));
            }
            if component.get("evidence_status").and_then(Value::as_str) != Some("UNQUALIFIED") {
                errors.push(format!(
                    "component {component_id} cannot self-promote evidence status"
                ));
            }
            let component_sources: Vec<&str> = component
                .get("reference_source_ids")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            if component_sources.is_empty()
                || component_sources
                    .iter()
                    .any(|source_id| !sources.contains(*source_id))
            {
                errors.push(format!(
                    "component {component_id} has empty or unresolved source references"
                ));
            }
            let owners: Vec<&str> = component
                .get("process_semantics_owners")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            if owners.is_empty()
                || owners.iter().any(|owner| {
                    !owner.starts_with("https://github.com/Luminous-Dynamics/symthaea/issues/")
                })
            {
                errors.push(format!(
                    "component {component_id} lacks canonical process owners"
                ));
            }
        }

        let Some(stages) = route.get("stages").and_then(Value::as_array) else {
            errors.push(format!("route {route_id} stages must be an array"));
            continue;
        };

        let mut stage_sequence = BTreeMap::<String, u64>::new();
        for (index, stage) in stages.iter().enumerate() {
            let stage_id = stage
                .get("stage_id")
                .and_then(Value::as_str)
                .unwrap_or("<missing>");
            let sequence = stage.get("sequence").and_then(Value::as_u64);
            if sequence != Some((index + 1) as u64) {
                errors.push(format!(
                    "route {route_id} stage {stage_id} is not in its frozen topological sequence"
                ));
            }
            if stage_sequence
                .insert(stage_id.to_owned(), sequence.unwrap_or(0))
                .is_some()
            {
                errors.push(format!(
                    "duplicate stage ID in route {route_id}: {stage_id}"
                ));
            }
            if stage.get("evidence_status").and_then(Value::as_str) != Some("UNQUALIFIED") {
                errors.push(format!(
                    "stage {stage_id} cannot self-promote evidence status"
                ));
            }
            if stage
                .get("process_semantics_owner")
                .and_then(Value::as_str)
                .is_none_or(|value| {
                    !value.starts_with("https://github.com/Luminous-Dynamics/symthaea/issues/")
                })
            {
                errors.push(format!(
                    "stage {stage_id} lacks a canonical process-semantics owner"
                ));
            }
        }

        for stage in stages {
            let stage_id = stage
                .get("stage_id")
                .and_then(Value::as_str)
                .unwrap_or("<missing>");
            let current_sequence = stage.get("sequence").and_then(Value::as_u64).unwrap_or(0);
            let Some(groups) = stage.get("predecessor_groups").and_then(Value::as_array) else {
                errors.push(format!("stage {stage_id} has no predecessor_groups array"));
                continue;
            };

            let mut seen_predecessors = BTreeSet::new();
            for group in groups {
                let mode = group.get("mode").and_then(Value::as_str);
                let predecessors = group.get("predecessors").and_then(Value::as_array);
                if !matches!(mode, Some("all_of" | "one_of_conditioned")) {
                    errors.push(format!(
                        "stage {stage_id} has an invalid predecessor group mode"
                    ));
                    continue;
                }
                let Some(predecessors) = predecessors else {
                    errors.push(format!(
                        "stage {stage_id} predecessor group has no predecessors"
                    ));
                    continue;
                };
                if predecessors.is_empty() {
                    errors.push(format!(
                        "stage {stage_id} contains an empty predecessor group"
                    ));
                }
                if mode == Some("one_of_conditioned")
                    && predecessors.iter().any(|edge| {
                        edge.get("condition")
                            .and_then(Value::as_str)
                            .is_none_or(str::is_empty)
                    })
                {
                    errors.push(format!(
                        "stage {stage_id} has an unconditioned alternative predecessor"
                    ));
                }
                for edge in predecessors {
                    let predecessor = edge
                        .get("stage_id")
                        .and_then(Value::as_str)
                        .unwrap_or("<missing>");
                    if !seen_predecessors.insert(predecessor.to_owned()) {
                        errors.push(format!(
                            "stage {stage_id} repeats predecessor {predecessor}"
                        ));
                    }
                    match stage_sequence.get(predecessor) {
                        None => errors.push(format!(
                            "stage {stage_id} has unresolved predecessor {predecessor}"
                        )),
                        Some(predecessor_sequence) if *predecessor_sequence >= current_sequence => {
                            errors.push(format!(
                                "stage {stage_id} has a forward dependency on {predecessor}"
                            ));
                        }
                        Some(_) => {}
                    }
                }
            }
        }
    }

    errors.sort();
    errors.dedup();
    errors
}

#[test]
fn frozen_display_manufacturing_routes_validate_against_schema_and_dag_contract() {
    let errors = validate_contract(MANIFEST_TEXT, SCHEMA_TEXT);
    assert!(
        errors.is_empty(),
        "route contract validation errors: {errors:#?}"
    );
}

#[test]
fn rejects_promoted_route_or_stage_status() {
    let mut manifest: Value = serde_json::from_str(MANIFEST_TEXT).expect("fixture manifest JSON");
    manifest["routes"][0]["stages"][0]["evidence_status"] = Value::String("QUALIFIED".to_owned());
    let text = serde_json::to_string(&manifest).expect("mutated manifest serializes");
    assert!(!validate_contract(&text, SCHEMA_TEXT).is_empty());
}

#[test]
fn rejects_promoted_component_status_and_unresolved_component_source() {
    let mut promoted: Value = serde_json::from_str(MANIFEST_TEXT).expect("fixture manifest JSON");
    promoted["routes"][0]["component_interfaces"][0]["evidence_status"] =
        Value::String("QUALIFIED".to_owned());
    let promoted_text = serde_json::to_string(&promoted).expect("promoted fixture serializes");
    assert!(!validate_contract(&promoted_text, SCHEMA_TEXT).is_empty());

    let mut unresolved: Value = serde_json::from_str(MANIFEST_TEXT).expect("fixture manifest JSON");
    unresolved["routes"][0]["component_interfaces"][0]["reference_source_ids"][0] =
        Value::String("SRC-NOT-DEFINED".to_owned());
    let unresolved_text =
        serde_json::to_string(&unresolved).expect("unresolved fixture serializes");
    assert!(!validate_contract(&unresolved_text, SCHEMA_TEXT).is_empty());
}

#[test]
fn rejects_unresolved_source_and_predecessor_references() {
    let mut source_mutation: Value =
        serde_json::from_str(MANIFEST_TEXT).expect("fixture manifest JSON");
    source_mutation["routes"][0]["source_ids"][0] = Value::String("SRC-NOT-DEFINED".to_owned());
    let source_text = serde_json::to_string(&source_mutation).expect("source mutation serializes");
    assert!(!validate_contract(&source_text, SCHEMA_TEXT).is_empty());

    let mut predecessor_mutation: Value =
        serde_json::from_str(MANIFEST_TEXT).expect("fixture manifest JSON");
    predecessor_mutation["routes"][0]["stages"][2]["predecessor_groups"][0]["predecessors"][0]["stage_id"] =
        Value::String("MICLED-99".to_owned());
    let predecessor_text =
        serde_json::to_string(&predecessor_mutation).expect("predecessor mutation serializes");
    assert!(!validate_contract(&predecessor_text, SCHEMA_TEXT).is_empty());
}

#[test]
fn rejects_duplicate_stage_ids_and_non_topological_dependencies() {
    let mut duplicate_mutation: Value =
        serde_json::from_str(MANIFEST_TEXT).expect("fixture manifest JSON");
    let first_stage_id = duplicate_mutation["routes"][0]["stages"][0]["stage_id"].clone();
    duplicate_mutation["routes"][0]["stages"][1]["stage_id"] = first_stage_id;
    let duplicate_text =
        serde_json::to_string(&duplicate_mutation).expect("duplicate mutation serializes");
    assert!(!validate_contract(&duplicate_text, SCHEMA_TEXT).is_empty());

    let mut order_mutation: Value =
        serde_json::from_str(MANIFEST_TEXT).expect("fixture manifest JSON");
    order_mutation["routes"][0]["stages"][0]["predecessor_groups"] = serde_json::json!([{
        "mode":"all_of",
        "predecessors":[{
            "stage_id":"MICLED-02",
            "condition":"deliberately forward dependency for negative control"
        }]
    }]);
    let order_text = serde_json::to_string(&order_mutation).expect("order mutation serializes");
    assert!(!validate_contract(&order_text, SCHEMA_TEXT).is_empty());
}
