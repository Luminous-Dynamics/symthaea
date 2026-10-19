// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Independent test-only oracle. Keep these rules separate from process_trace.rs:
// the purpose is to detect disagreement rather than call the implementation twice.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_culinary::process_trace::validate_culinary_process_trace_v1;

#[derive(Debug, PartialEq)]
struct OracleReport {
    step_count: usize,
    stream_count: usize,
    boundary_input_count: usize,
    boundary_output_count: usize,
    mass_balance_export_ready: bool,
    all_evidence_records_marked_qualified: bool,
    boundary_input_point_g: Option<f64>,
    boundary_output_point_g: Option<f64>,
}

fn text<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::as_str).filter(|s| !s.trim().is_empty())
}

fn timestamp_shape(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 20
        && b.get(4) == Some(&b'-')
        && b.get(7) == Some(&b'-')
        && b.get(10) == Some(&b'T')
        && b.get(13) == Some(&b':')
        && b.get(16) == Some(&b':')
        && b.get(19).is_some_and(|x| *x == b'.' || *x == b'Z' || *x == b'+' || *x == b'-')
}

fn evidence_ok(e: &Value, errors: &mut Vec<String>) -> bool {
    let source = text(e, "source_ref");
    let version = text(e, "source_version");
    let time = text(e, "known_at");
    let status = text(e, "evidence_status");
    if source.is_none() || version.is_none() || !time.is_some_and(timestamp_shape)
        || !matches!(status, Some("synthetic" | "unreviewed" | "reviewed" | "qualified"))
    {
        errors.push("oracle: incomplete or malformed evidence metadata".to_owned());
    }
    if status == Some("qualified") && source.is_some_and(|s| s.starts_with("urn:synthetic:")) {
        errors.push("oracle: synthetic evidence cannot be marked qualified".to_owned());
        return false;
    }
    status == Some("qualified") && source.is_some() && version.is_some() && time.is_some()
}

fn point_sum(streams: &[Value], input: bool) -> Option<f64> {
    let mut sum = 0.0;
    for stream in streams {
        let boundary = if input {
            text(stream, "from_step_id") == Some("external")
        } else {
            text(stream, "to_step_id") == Some("external")
        };
        if !boundary {
            continue;
        }
        if !matches!(text(&stream["mass_g"], "status"), Some("measured" | "estimated")) {
            return None;
        }
        sum += stream["mass_g"]["value"].as_f64()?;
    }
    Some(sum)
}

fn independent_oracle(trace: &Value) -> Result<OracleReport, Vec<String>> {
    let mut errors = Vec::new();
    let mut blockers = Vec::new();
    let mut all_qualified = true;
    if text(trace, "schema_version") != Some("1.0.0")
        || text(trace, "batch_basis") != Some("single_batch")
        || text(trace, "mass_unit") != Some("g")
        || text(trace, "trace_id").is_none()
        || text(trace, "process_id").is_none()
        || text(trace, "food_state_start").is_none()
        || text(trace, "food_state_end").is_none()
    {
        errors.push("oracle: invalid trace header".to_owned());
    }
    let inventory = text(trace, "boundary_inventory_status");
    if !matches!(inventory, Some("claimed_complete" | "known_incomplete" | "unknown")) {
        errors.push("oracle: invalid inventory declaration".to_owned());
    }
    if inventory != Some("claimed_complete") {
        blockers.push("oracle: boundary inventory is not claimed complete".to_owned());
    }
    all_qualified &= evidence_ok(
        trace.get("boundary_inventory_evidence").unwrap_or(&Value::Null),
        &mut errors,
    );

    let Some(steps) = trace.get("steps").and_then(Value::as_array) else {
        return Err(vec!["oracle: steps is not an array".to_owned()]);
    };
    let Some(streams) = trace.get("streams").and_then(Value::as_array) else {
        return Err(vec!["oracle: streams is not an array".to_owned()]);
    };
    if steps.is_empty() || streams.is_empty() {
        errors.push("oracle: steps and streams must not be empty".to_owned());
    }

    let mut step_order = BTreeMap::new();
    let mut step_inputs = BTreeMap::<String, Vec<String>>::new();
    let mut step_outputs = BTreeMap::<String, Vec<String>>::new();
    for (index, step) in steps.iter().enumerate() {
        let Some(id) = text(step, "step_id") else {
            errors.push("oracle: step lacks an id".to_owned());
            continue;
        };
        if step_order.insert(id.to_owned(), index).is_some() {
            errors.push(format!("oracle: duplicate step id {id}"));
        }
        if text(step, "operation").is_none() {
            errors.push(format!("oracle: step {id} lacks operation"));
        }
        let read_refs = |key: &str, errors: &mut Vec<String>| -> Vec<String> {
            let Some(items) = step.get(key).and_then(Value::as_array) else {
                errors.push(format!("oracle: {id} has no {key} array"));
                return Vec::new();
            };
            let mut result = Vec::new();
            let mut seen = BTreeSet::new();
            for item in items {
                if let Some(item_id) = item.as_str().filter(|s| !s.trim().is_empty()) {
                    if !seen.insert(item_id.to_owned()) {
                        errors.push(format!("oracle: duplicate {key} reference {item_id}"));
                    }
                    result.push(item_id.to_owned());
                } else {
                    errors.push(format!("oracle: malformed {key} reference"));
                }
            }
            if result.is_empty() {
                errors.push(format!("oracle: step {id} has empty {key}"));
            }
            result
        };
        step_inputs.insert(id.to_owned(), read_refs("input_stream_ids", &mut errors));
        step_outputs.insert(id.to_owned(), read_refs("output_stream_ids", &mut errors));
        all_qualified &= evidence_ok(
            step.get("evidence").unwrap_or(&Value::Null),
            &mut errors,
        );
    }

    let mut by_id = BTreeMap::<String, &Value>::new();
    for stream in streams {
        if let Some(id) = text(stream, "stream_id") {
            if by_id.insert(id.to_owned(), stream).is_some() {
                errors.push(format!("oracle: duplicate stream id {id}"));
            }
        } else {
            errors.push("oracle: stream lacks id".to_owned());
        }
    }

    let mut inputs_seen = BTreeSet::new();
    let mut outputs_seen = BTreeSet::new();
    for (step_id, refs) in &step_inputs {
        for id in refs {
            match by_id.get(id) {
                Some(stream) if text(stream, "to_step_id") == Some(step_id.as_str()) => {
                    if !inputs_seen.insert(id.clone()) {
                        errors.push(format!("oracle: stream {id} is consumed more than once"));
                    }
                }
                Some(_) => errors.push(format!("oracle: input endpoint mismatch for {id}")),
                None => errors.push(format!("oracle: missing input stream {id}")),
            }
        }
    }
    for (step_id, refs) in &step_outputs {
        for id in refs {
            match by_id.get(id) {
                Some(stream) if text(stream, "from_step_id") == Some(step_id.as_str()) => {
                    if !outputs_seen.insert(id.clone()) {
                        errors.push(format!("oracle: stream {id} is produced more than once"));
                    }
                }
                Some(_) => errors.push(format!("oracle: output endpoint mismatch for {id}")),
                None => errors.push(format!("oracle: missing output stream {id}")),
            }
        }
    }

    let mut boundary_inputs = 0;
    let mut boundary_outputs = 0;
    for stream in streams {
        let id = text(stream, "stream_id").unwrap_or("<missing>");
        let from = text(stream, "from_step_id");
        let to = text(stream, "to_step_id");
        match (from, to) {
            (Some("external"), Some(target)) if target != "external" => {
                boundary_inputs += 1;
                if !step_order.contains_key(target) || !inputs_seen.contains(id) {
                    errors.push(format!("oracle: invalid boundary input {id}"));
                }
            }
            (Some(source), Some("external")) if source != "external" => {
                boundary_outputs += 1;
                if !step_order.contains_key(source) || !outputs_seen.contains(id) {
                    errors.push(format!("oracle: invalid boundary output {id}"));
                }
            }
            (Some(source), Some(target)) if source != "external" && target != "external" => {
                if !step_order.get(source).zip(step_order.get(target))
                    .is_some_and(|(a, b)| a < b)
                {
                    errors.push(format!("oracle: internal flow {id} violates order"));
                }
                if !outputs_seen.contains(id) || !inputs_seen.contains(id) {
                    errors.push(format!("oracle: internal flow {id} is orphaned"));
                }
            }
            _ => errors.push(format!("oracle: invalid endpoints on stream {id}")),
        }

        let mass = &stream["mass_g"];
        match text(mass, "status") {
            Some("measured" | "estimated") => {
                let (v, lo, hi) = (
                    mass.get("value").and_then(Value::as_f64),
                    mass.get("lower").and_then(Value::as_f64),
                    mass.get("upper").and_then(Value::as_f64),
                );
                if let (Some(v), Some(lo), Some(hi)) = (v, lo, hi) {
                    if !v.is_finite() || !lo.is_finite() || !hi.is_finite()
                        || lo < 0.0 || lo > v || v > hi || text(mass, "unit") != Some("g")
                    {
                        errors.push(format!("oracle: invalid mass interval on {id}"));
                    }
                    if mass.get("unknown_reason").is_some() {
                        errors.push(format!("oracle: bounded mass {id} has unknown_reason"));
                    }
                } else {
                    errors.push(format!("oracle: bounded mass missing numbers on {id}"));
                }
            }
            Some("unknown") => {
                if !["value", "lower", "upper"].iter().all(|key| mass.get(key).is_some_and(Value::is_null))
                    || text(mass, "unknown_reason").is_none()
                    || text(mass, "unit") != Some("g")
                {
                    errors.push(format!("oracle: unknown mass on {id} is not explicit"));
                } else {
                    blockers.push(format!("oracle: stream {id} quantity is unknown"));
                }
            }
            _ => errors.push(format!("oracle: invalid mass status on {id}")),
        }
        all_qualified &= evidence_ok(stream.get("evidence").unwrap_or(&Value::Null), &mut errors);
    }

    if boundary_inputs == 0 || boundary_outputs == 0 {
        errors.push("oracle: at least one input and output boundary stream required".to_owned());
    }
    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }
    blockers.sort();
    blockers.dedup();
    Ok(OracleReport {
        step_count: steps.len(),
        stream_count: streams.len(),
        boundary_input_count: boundary_inputs,
        boundary_output_count: boundary_outputs,
        mass_balance_export_ready: blockers.is_empty(),
        all_evidence_records_marked_qualified: all_qualified,
        boundary_input_point_g: point_sum(streams, true),
        boundary_output_point_g: point_sum(streams, false),
    })
}

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../fixtures/process-trace-v1/synthetic-cook-batch.json"
    )).unwrap()
}

#[test]
fn independent_oracle_agrees_on_fixture_without_claiming_conservation() {
    let trace = fixture();
    let expected = independent_oracle(&trace).unwrap();
    let actual = validate_culinary_process_trace_v1(&trace).unwrap();
    assert_eq!(actual.step_count, expected.step_count);
    assert_eq!(actual.stream_count, expected.stream_count);
    assert_eq!(actual.boundary_input_count, expected.boundary_input_count);
    assert_eq!(actual.boundary_output_count, expected.boundary_output_count);
    assert_eq!(actual.mass_balance_export_ready, expected.mass_balance_export_ready);
    assert_eq!(actual.all_evidence_records_marked_qualified, expected.all_evidence_records_marked_qualified);
    assert_eq!(expected.boundary_input_point_g, Some(308.0));
    assert_eq!(expected.boundary_output_point_g, Some(308.0));
}

#[test]
fn independent_oracle_preserves_unknown_as_missing_not_zero() {
    let mut trace = fixture();
    trace["streams"][6]["mass_g"] = serde_json::json!({
        "status":"unknown", "value":null, "lower":null, "upper":null,
        "unit":"g", "unknown_reason":"not weighed in this synthetic example"
    });
    let expected = independent_oracle(&trace).unwrap();
    let actual = validate_culinary_process_trace_v1(&trace).unwrap();
    assert!(!expected.mass_balance_export_ready);
    assert!(!actual.mass_balance_export_ready);
    assert!(expected.boundary_output_point_g.is_none());
}

#[test]
fn independent_oracle_and_implementation_reject_negative_controls() {
    let mut dangling = fixture();
    dangling["steps"][1]["input_stream_ids"].as_array_mut().unwrap()
        .push(Value::String("not-a-stream".to_owned()));
    assert!(independent_oracle(&dangling).is_err());
    assert!(validate_culinary_process_trace_v1(&dangling).is_err());

    let mut spoofed = fixture();
    spoofed["streams"][0]["evidence"]["evidence_status"] =
        Value::String("qualified".to_owned());
    assert!(independent_oracle(&spoofed).is_err());
    assert!(validate_culinary_process_trace_v1(&spoofed).is_err());
}
