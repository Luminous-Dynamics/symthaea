// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bearing trace of material streams through one culinary batch.
//!
//! This module validates identity, references, direction, ordering, and explicit
//! quantity missingness. It deliberately does not calculate mass closure. A
//! structurally exportable trace is not evidence that mass balances, chemistry,
//! nutrition, food safety, or sensory quality are correct.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const OPERATIONS: &[&str] = &[
    "prepare", "wash", "trim", "mix", "heat", "cool", "ferment", "filter",
    "transfer", "serve", "other",
];
const MATERIAL_CLASSES: &[&str] = &[
    "ingredient", "added_water", "added_oil", "seasoning", "working_fluid",
    "intermediate_food", "edible_product", "coproduct", "byproduct",
    "discarded_liquid", "spent_oil", "trim", "utensil_residue",
    "evaporated_water", "recycled_material", "waste", "unknown",
];
const DISPOSITIONS: &[&str] = &[
    "consumed", "transformed", "retained", "coproduct", "byproduct",
    "discarded", "evaporated", "recycled", "unknown",
];

/// Structural validation summary. Export readiness is not a balance verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CulinaryProcessTraceResultV1 {
    pub step_count: usize,
    pub stream_count: usize,
    pub boundary_input_count: usize,
    pub boundary_output_count: usize,
    /// True means every declared stream is connected and has bounded mass data,
    /// and the boundary inventory is claimed complete. It does not mean balanced.
    pub mass_balance_export_ready: bool,
    /// Reflects declared evidence labels only; source authenticity is not checked.
    pub all_evidence_records_marked_qualified: bool,
    pub blockers: Vec<String>,
}

fn non_empty<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty())
}

fn number(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64).filter(|n| n.is_finite())
}

fn digits(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes.iter().all(u8::is_ascii_digit)
}

fn rfc3339_shape(value: &str) -> bool {
    let b = value.as_bytes();
    if b.len() < 20
        || b.get(4) != Some(&b'-')
        || b.get(7) != Some(&b'-')
        || b.get(10) != Some(&b'T')
        || b.get(13) != Some(&b':')
        || b.get(16) != Some(&b':')
        || !digits(&b[0..4])
        || !digits(&b[5..7])
        || !digits(&b[8..10])
        || !digits(&b[11..13])
        || !digits(&b[14..16])
        || !digits(&b[17..19])
    {
        return false;
    }

    let tz_start = if b.last() == Some(&b'Z') {
        b.len() - 1
    } else if b.len() >= 25 {
        let start = b.len() - 6;
        if !matches!(b[start], b'+' | b'-')
            || b[start + 3] != b':'
            || !digits(&b[start + 1..start + 3])
            || !digits(&b[start + 4..start + 6])
        {
            return false;
        }
        start
    } else {
        return false;
    };

    if tz_start == 19 {
        return true;
    }
    b.get(19) == Some(&b'.') && tz_start > 20 && digits(&b[20..tz_start])
}

fn validate_evidence(evidence: &Value, label: &str, errors: &mut Vec<String>) -> bool {
    let source = non_empty(evidence, "source_ref");
    let version = non_empty(evidence, "source_version");
    let known_at = non_empty(evidence, "known_at");
    let status = non_empty(evidence, "evidence_status");

    if source.is_none() {
        errors.push(format!("{label}.source_ref is required"));
    }
    if version.is_none() {
        errors.push(format!("{label}.source_version is required"));
    }
    if !known_at.is_some_and(rfc3339_shape) {
        errors.push(format!("{label}.known_at must have an RFC3339 timestamp shape"));
    }
    if !matches!(status, Some("synthetic" | "unreviewed" | "reviewed" | "qualified")) {
        errors.push(format!("{label}.evidence_status is invalid"));
    }

    if status == Some("qualified")
        && source.is_some_and(|reference| reference.starts_with("urn:synthetic:"))
    {
        errors.push(format!("{label} cannot mark a synthetic source qualified"));
        return false;
    }

    status == Some("qualified")
        && source.is_some()
        && version.is_some()
        && known_at.is_some_and(rfc3339_shape)
}

fn string_array(value: &Value, key: &str, label: &str, errors: &mut Vec<String>) -> Vec<String> {
    let Some(items) = value.get(key).and_then(Value::as_array) else {
        errors.push(format!("{label}.{key} must be an array"));
        return Vec::new();
    };
    if items.is_empty() {
        errors.push(format!("{label}.{key} must not be empty"));
    }
    let mut out = Vec::with_capacity(items.len());
    let mut seen = BTreeSet::new();
    for (index, item) in items.iter().enumerate() {
        match item.as_str().filter(|s| !s.trim().is_empty()) {
            Some(id) => {
                if !seen.insert(id.to_owned()) {
                    errors.push(format!("{label}.{key}[{index}] duplicates stream id {id:?}"));
                }
                out.push(id.to_owned());
            }
            None => errors.push(format!("{label}.{key}[{index}] must be non-empty text")),
        }
    }
    out
}

fn validate_mass(
    mass: &Value,
    label: &str,
    errors: &mut Vec<String>,
    blockers: &mut Vec<String>,
) -> bool {
    if non_empty(mass, "unit") != Some("g") {
        errors.push(format!("{label}.unit must be g"));
    }
    match non_empty(mass, "status") {
        Some("measured" | "estimated") => {
            let (Some(value), Some(lower), Some(upper)) = (
                number(mass, "value"),
                number(mass, "lower"),
                number(mass, "upper"),
            ) else {
                errors.push(format!("{label} measured/estimated mass needs finite value/lower/upper"));
                return false;
            };
            if lower < 0.0 || lower > value || value > upper {
                errors.push(format!("{label} must satisfy 0 <= lower <= value <= upper"));
                return false;
            }
            if mass.get("unknown_reason").is_some() {
                errors.push(format!("{label} must not include unknown_reason when mass is bounded"));
            }
            true
        }
        Some("unknown") => {
            let explicit_nulls = ["value", "lower", "upper"]
                .iter()
                .all(|key| mass.get(key).is_some_and(Value::is_null));
            if !explicit_nulls {
                errors.push(format!("{label} with status unknown must set value/lower/upper to null"));
            }
            if non_empty(mass, "unknown_reason").is_none() {
                errors.push(format!("{label} with status unknown requires unknown_reason"));
            }
            blockers.push(format!("{label} has an explicit unknown quantity"));
            false
        }
        _ => {
            errors.push(format!("{label}.status must be measured, estimated, or unknown"));
            false
        }
    }
}

/// Validate one source-bearing CulinaryProcessTraceV1 JSON record.
///
/// The returned readiness flag only says the trace can be mapped to the shared
/// PIE-001 mass-balance interface. This function never computes a balance and
/// never authenticates evidence sources or the real-world completeness claim.
pub fn validate_culinary_process_trace_v1(
    trace: &Value,
) -> Result<CulinaryProcessTraceResultV1, Vec<String>> {
    let mut errors = Vec::new();
    let mut blockers = Vec::new();
    let mut all_evidence_qualified = true;

    if non_empty(trace, "schema_version") != Some("1.0.0") {
        errors.push("schema_version must be 1.0.0".to_owned());
    }
    if non_empty(trace, "trace_id").is_none() {
        errors.push("trace_id is required".to_owned());
    }
    if non_empty(trace, "process_id").is_none() {
        errors.push("process_id is required".to_owned());
    }
    if non_empty(trace, "batch_basis") != Some("single_batch") {
        errors.push("batch_basis must be single_batch".to_owned());
    }
    if non_empty(trace, "mass_unit") != Some("g") {
        errors.push("mass_unit must be g".to_owned());
    }
    if non_empty(trace, "food_state_start").is_none()
        || non_empty(trace, "food_state_end").is_none()
    {
        errors.push("food_state_start and food_state_end are required".to_owned());
    }

    let boundary_status = non_empty(trace, "boundary_inventory_status");
    if !matches!(boundary_status, Some("claimed_complete" | "known_incomplete" | "unknown")) {
        errors.push("boundary_inventory_status is invalid".to_owned());
    }
    if boundary_status != Some("claimed_complete") {
        blockers.push(format!(
            "boundary inventory is {:?}, not claimed_complete",
            boundary_status.unwrap_or("missing")
        ));
    }
    if !validate_evidence(
        trace.get("boundary_inventory_evidence").unwrap_or(&Value::Null),
        "boundary_inventory_evidence",
        &mut errors,
    ) {
        all_evidence_qualified = false;
    }

    let steps: &[Value] = match trace.get("steps").and_then(Value::as_array) {
        Some(items) if !items.is_empty() => items.as_slice(),
        _ => {
            errors.push("steps must be a non-empty array".to_owned());
            &[]
        }
    };
    let streams: &[Value] = match trace.get("streams").and_then(Value::as_array) {
        Some(items) if !items.is_empty() => items.as_slice(),
        _ => {
            errors.push("streams must be a non-empty array".to_owned());
            &[]
        }
    };

    let mut step_order = BTreeMap::<String, usize>::new();
    let mut step_inputs = BTreeMap::<String, Vec<String>>::new();
    let mut step_outputs = BTreeMap::<String, Vec<String>>::new();

    for (index, step) in steps.iter().enumerate() {
        let label = format!("steps[{index}]");
        let Some(step_id) = non_empty(step, "step_id") else {
            errors.push(format!("{label}.step_id is required"));
            continue;
        };
        if step_order.insert(step_id.to_owned(), index).is_some() {
            errors.push(format!("duplicate step_id {step_id:?}"));
            continue;
        }
        if !non_empty(step, "operation").is_some_and(|op| OPERATIONS.contains(&op)) {
            errors.push(format!("{label}.operation is invalid"));
        }
        step_inputs.insert(
            step_id.to_owned(),
            string_array(step, "input_stream_ids", &label, &mut errors),
        );
        step_outputs.insert(
            step_id.to_owned(),
            string_array(step, "output_stream_ids", &label, &mut errors),
        );
        if !validate_evidence(
            step.get("evidence").unwrap_or(&Value::Null),
            &format!("{label}.evidence"),
            &mut errors,
        ) {
            all_evidence_qualified = false;
        }
    }

    let mut stream_indices = BTreeMap::<String, usize>::new();
    let mut from_steps = Vec::with_capacity(streams.len());
    let mut to_steps = Vec::with_capacity(streams.len());
    let mut masses_bounded = true;
    let mut boundary_input_count = 0;
    let mut boundary_output_count = 0;

    for (index, stream) in streams.iter().enumerate() {
        let label = format!("streams[{index}]");
        let Some(stream_id) = non_empty(stream, "stream_id") else {
            errors.push(format!("{label}.stream_id is required"));
            from_steps.push(None);
            to_steps.push(None);
            masses_bounded = false;
            continue;
        };
        if stream_indices.insert(stream_id.to_owned(), index).is_some() {
            errors.push(format!("duplicate stream_id {stream_id:?}"));
        }

        if !non_empty(stream, "material_class")
            .is_some_and(|kind| MATERIAL_CLASSES.contains(&kind))
        {
            errors.push(format!("{label}.material_class is invalid"));
        }
        if !non_empty(stream, "disposition")
            .is_some_and(|kind| DISPOSITIONS.contains(&kind))
        {
            errors.push(format!("{label}.disposition is invalid"));
        }

        let from = non_empty(stream, "from_step_id");
        let to = non_empty(stream, "to_step_id");
        match (from, to) {
            (Some("external"), Some(target)) if target != "external" => {
                boundary_input_count += 1;
                if !step_order.contains_key(target) {
                    errors.push(format!("{label} targets unknown step {target:?}"));
                }
                from_steps.push(None);
                to_steps.push(Some(target.to_owned()));
            }
            (Some(source), Some("external")) if source != "external" => {
                boundary_output_count += 1;
                if !step_order.contains_key(source) {
                    errors.push(format!("{label} starts at unknown step {source:?}"));
                }
                from_steps.push(Some(source.to_owned()));
                to_steps.push(None);
            }
            (Some(source), Some(target)) if source != "external" && target != "external" => {
                if source == target {
                    errors.push(format!("{label} cannot loop within the same step"));
                }
                if !step_order.contains_key(source) {
                    errors.push(format!("{label} starts at unknown step {source:?}"));
                }
                if !step_order.contains_key(target) {
                    errors.push(format!("{label} targets unknown step {target:?}"));
                }
                if let (Some(source_order), Some(target_order)) =
                    (step_order.get(source), step_order.get(target))
                {
                    if source_order >= target_order {
                        errors.push(format!("{label} violates declared step order"));
                    }
                }
                from_steps.push(Some(source.to_owned()));
                to_steps.push(Some(target.to_owned()));
            }
            _ => {
                errors.push(format!("{label} must connect external-to-step, step-to-step, or step-to-external"));
                from_steps.push(None);
                to_steps.push(None);
            }
        }

        if !validate_mass(
            stream.get("mass_g").unwrap_or(&Value::Null),
            &format!("{label}.mass_g"),
            &mut errors,
            &mut blockers,
        ) {
            masses_bounded = false;
        }
        if !validate_evidence(
            stream.get("evidence").unwrap_or(&Value::Null),
            &format!("{label}.evidence"),
            &mut errors,
        ) {
            all_evidence_qualified = false;
        }
    }

    let mut used_as_input = BTreeSet::<String>::new();
    let mut used_as_output = BTreeSet::<String>::new();

    for (step_id, ids) in &step_inputs {
        for id in ids {
            if !stream_indices.contains_key(id) {
                errors.push(format!("step {step_id:?} references unknown input stream {id:?}"));
                continue;
            }
            if !used_as_input.insert(id.clone()) {
                errors.push(format!("stream {id:?} is consumed by more than one step"));
            }
            let index = stream_indices[id];
            if to_steps.get(index).and_then(Option::as_deref) != Some(step_id.as_str()) {
                errors.push(format!("stream {id:?} destination disagrees with step input list"));
            }
        }
    }
    for (step_id, ids) in &step_outputs {
        for id in ids {
            if !stream_indices.contains_key(id) {
                errors.push(format!("step {step_id:?} references unknown output stream {id:?}"));
                continue;
            }
            if !used_as_output.insert(id.clone()) {
                errors.push(format!("stream {id:?} is produced by more than one step"));
            }
            let index = stream_indices[id];
            if from_steps.get(index).and_then(Option::as_deref) != Some(step_id.as_str()) {
                errors.push(format!("stream {id:?} source disagrees with step output list"));
            }
        }
    }

    for (index, stream) in streams.iter().enumerate() {
        let Some(stream_id) = non_empty(stream, "stream_id") else {
            continue;
        };
        if from_steps.get(index).and_then(Option::as_deref).is_some()
            && !used_as_output.contains(stream_id)
        {
            errors.push(format!("internal/output stream {stream_id:?} is not listed by its producer"));
        }
        if to_steps.get(index).and_then(Option::as_deref).is_some()
            && !used_as_input.contains(stream_id)
        {
            errors.push(format!("internal/input stream {stream_id:?} is not listed by its consumer"));
        }
    }

    if boundary_input_count == 0 {
        errors.push("at least one external boundary input stream is required".to_owned());
    }
    if boundary_output_count == 0 {
        errors.push("at least one external boundary output stream is required".to_owned());
    }

    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }

    blockers.sort();
    blockers.dedup();
    Ok(CulinaryProcessTraceResultV1 {
        step_count: steps.len(),
        stream_count: streams.len(),
        boundary_input_count,
        boundary_output_count,
        mass_balance_export_ready: blockers.is_empty() && masses_bounded,
        all_evidence_records_marked_qualified: all_evidence_qualified,
        blockers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../fixtures/process-trace-v1/synthetic-cook-batch.json"
        ))
        .unwrap()
    }

    #[test]
    fn synthetic_fixture_is_connected_and_exportable_but_not_qualified() {
        let result = validate_culinary_process_trace_v1(&fixture()).unwrap();
        assert_eq!(result.step_count, 2);
        assert_eq!(result.stream_count, 8);
        assert_eq!(result.boundary_input_count, 3);
        assert_eq!(result.boundary_output_count, 4);
        assert!(result.mass_balance_export_ready);
        assert!(!result.all_evidence_records_marked_qualified);
        assert!(result.blockers.is_empty());
    }

    #[test]
    fn unknown_mass_is_explicit_and_blocks_export_readiness() {
        let mut trace = fixture();
        let stream = trace["streams"].as_array_mut().unwrap()
            .iter_mut().find(|s| s["stream_id"] == "evaporated-water").unwrap();
        stream["mass_g"]["status"] = Value::String("unknown".into());
        stream["mass_g"]["value"] = Value::Null;
        stream["mass_g"]["lower"] = Value::Null;
        stream["mass_g"]["upper"] = Value::Null;
        stream["mass_g"]["unknown_reason"] = Value::String("not measured in this synthetic run".into());
        let result = validate_culinary_process_trace_v1(&trace).unwrap();
        assert!(!result.mass_balance_export_ready);
        assert!(result.blockers.iter().any(|b| b.contains("explicit unknown quantity")));
    }

    #[test]
    fn missing_unknown_reason_is_invalid() {
        let mut trace = fixture();
        trace["streams"][0]["mass_g"]["status"] = Value::String("unknown".into());
        trace["streams"][0]["mass_g"]["value"] = Value::Null;
        trace["streams"][0]["mass_g"]["lower"] = Value::Null;
        trace["streams"][0]["mass_g"]["upper"] = Value::Null;
        assert!(validate_culinary_process_trace_v1(&trace).is_err());
    }

    #[test]
    fn orphaned_declared_stream_is_rejected() {
        let mut trace = fixture();
        trace["steps"][0]["output_stream_ids"] = serde_json::json!(["prepared-staple"]);
        assert!(validate_culinary_process_trace_v1(&trace).is_err());
    }

    #[test]
    fn dangling_step_reference_is_rejected() {
        let mut trace = fixture();
        trace["steps"][1]["input_stream_ids"].as_array_mut().unwrap()
            .push(Value::String("missing-water".into()));
        assert!(validate_culinary_process_trace_v1(&trace).is_err());
    }

    #[test]
    fn duplicate_stream_ids_are_rejected() {
        let mut trace = fixture();
        let first = trace["streams"][0].clone();
        trace["streams"].as_array_mut().unwrap().push(first);
        assert!(validate_culinary_process_trace_v1(&trace).is_err());
    }

    #[test]
    fn synthetic_evidence_cannot_be_relabelled_qualified() {
        let mut trace = fixture();
        trace["streams"][0]["evidence"]["evidence_status"] = Value::String("qualified".into());
        assert!(validate_culinary_process_trace_v1(&trace).is_err());
    }

    #[test]
    fn missing_source_metadata_is_rejected() {
        let mut trace = fixture();
        trace["streams"][0]["evidence"]["source_ref"] = Value::Null;
        assert!(validate_culinary_process_trace_v1(&trace).is_err());
    }

    #[test]
    fn non_gram_stream_mass_is_rejected() {
        let mut trace = fixture();
        trace["streams"][0]["mass_g"]["unit"] = Value::String("kg".into());
        assert!(validate_culinary_process_trace_v1(&trace).is_err());
    }

    #[test]
    fn step_order_must_follow_material_flow() {
        let mut trace = fixture();
        trace["streams"][4]["from_step_id"] = Value::String("cook".into());
        trace["streams"][4]["to_step_id"] = Value::String("prepare".into());
        assert!(validate_culinary_process_trace_v1(&trace).is_err());
    }

    #[test]
    fn unknown_boundary_inventory_is_explicit_but_not_exportable() {
        let mut trace = fixture();
        trace["boundary_inventory_status"] = Value::String("unknown".into());
        let result = validate_culinary_process_trace_v1(&trace).unwrap();
        assert!(!result.mass_balance_export_ready);
        assert!(!result.blockers.is_empty());
    }

    #[test]
    fn stream_array_order_does_not_change_validity_summary() {
        let mut trace = fixture();
        let expected = validate_culinary_process_trace_v1(&trace).unwrap();
        trace["streams"].as_array_mut().unwrap().reverse();
        let actual = validate_culinary_process_trace_v1(&trace).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn widening_mass_uncertainty_does_not_create_a_balance_verdict() {
        let mut trace = fixture();
        trace["streams"][0]["mass_g"]["lower"] = serde_json::json!(0.0);
        let result = validate_culinary_process_trace_v1(&trace).unwrap();
        assert!(result.mass_balance_export_ready);
        // Result type intentionally contains no balanced/unbalanced field.
    }
}
