// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Verify causal invariants in one Symthaea interaction trace or its fixtures.
//!
//! This validates event-to-event relationships. It is not a JSON Schema
//! validator, dataset scorer, device attestation tool, or observer trust root.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

use serde_json::Value;
use symthaea_soma::embodiment_interaction_trace::{
    InteractionTraceSummary, InteractionTraceVerifier, INTERACTION_TRACE_SCHEMA_V1,
};

const FIXTURE_SCHEMA_V1: &str = "symthaea.interaction-trace-fixtures.v1";

fn usage(program: &std::ffi::OsStr) {
    eprintln!(
        "Usage: {} <interaction-trace-or-fixtures.json>",
        PathBuf::from(program).display()
    );
}

fn verify_input(input: &str) -> Result<(InteractionTraceSummary, String), String> {
    let value: Value =
        serde_json::from_str(input).map_err(|error| format!("invalid JSON: {error}"))?;
    if value.get("schema").and_then(Value::as_str) == Some(INTERACTION_TRACE_SCHEMA_V1) {
        let summary = InteractionTraceVerifier::verify_value(&value)
            .map_err(|error| format!("causal verification failed: {error}"))?;
        return Ok((summary, "single trace".to_owned()));
    }

    if value.get("fixture_schema").and_then(Value::as_str) != Some(FIXTURE_SCHEMA_V1) {
        return Err("unknown trace or fixture schema".to_owned());
    }

    let valid_traces = value
        .get("valid_traces")
        .and_then(Value::as_array)
        .filter(|traces| !traces.is_empty())
        .ok_or_else(|| "fixture bundle has no valid_traces".to_owned())?;
    let invalid_cases = value
        .get("invalid_cases")
        .and_then(Value::as_array)
        .filter(|cases| !cases.is_empty())
        .ok_or_else(|| "fixture bundle has no invalid_cases".to_owned())?;
    let mut summary = InteractionTraceSummary::default();

    for (index, trace) in valid_traces.iter().enumerate() {
        let result = InteractionTraceVerifier::verify_value(trace)
            .map_err(|error| format!("valid fixture {index} failed: {error}"))?;
        summary.event_count += result.event_count;
        summary.input_chunks += result.input_chunks;
        summary.semantic_cues += result.semantic_cues;
        summary.turn_decisions += result.turn_decisions;
        summary.authorizations += result.authorizations;
        summary.submitted_outputs += result.submitted_outputs;
        summary.presented_frames += result.presented_frames;
        summary.stopped_outputs += result.stopped_outputs;
        summary.grounding_probes += result.grounding_probes;
    }

    for (index, case) in invalid_cases.iter().enumerate() {
        let name = case
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("unnamed");
        let trace = case
            .get("trace")
            .ok_or_else(|| format!("negative fixture {index} ({name}) has no trace"))?;
        if let Ok(result) = InteractionTraceVerifier::verify_value(trace) {
            return Err(format!(
                "negative fixture {index} ({name}) unexpectedly passed with {} events",
                result.event_count
            ));
        }
    }

    Ok((
        summary,
        format!(
            "fixture bundle: {} valid trace(s), {} negative case(s) rejected",
            valid_traces.len(),
            invalid_cases.len()
        ),
    ))
}

fn main() {
    let mut args = env::args_os();
    let program = args
        .next()
        .unwrap_or_else(|| "verify_interaction_trace".into());
    let Some(path) = args.next() else {
        usage(&program);
        process::exit(2);
    };
    if args.next().is_some() {
        usage(&program);
        process::exit(2);
    }

    let path = PathBuf::from(path);
    let input = match fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("cannot read {}: {error}", path.display());
            process::exit(2);
        }
    };

    match verify_input(&input) {
        Ok((summary, mode)) => {
            println!(
                "valid {mode}: events={} inputs={} cues={} turn_decisions={} authorizations={} submitted={} frames={} stopped={} grounding_probes={}",
                summary.event_count,
                summary.input_chunks,
                summary.semantic_cues,
                summary.turn_decisions,
                summary.authorizations,
                summary.submitted_outputs,
                summary.presented_frames,
                summary.stopped_outputs,
                summary.grounding_probes,
            );
        }
        Err(error) => {
            eprintln!("invalid causal trace {}: {error}", path.display());
            process::exit(1);
        }
    }
}
