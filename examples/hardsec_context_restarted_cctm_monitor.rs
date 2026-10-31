//! HARDSEC-019 DKW-corrected Conditional Conformal Test Martingale (CCTM).
//!
//! Implements the published fixed-reference CCTM construction:
//! - fixed empirical CDF per predeclared workload context;
//! - DKW uniform confidence band for finite-reference CDF error;
//! - corrected smoothed betting function;
//! - predictable Online Newton Step update of eta;
//! - context-level alpha/delta allocation with union bounds.
//!
//! The published theorem assumes the reference samples and the null test-score
//! sequence are i.i.d. within each context, the score function is fixed, and
//! context assignment does not depend on the anomaly score. The tool records
//! these assumptions as unverified; identity digests and unique caller-declared
//! group IDs do not prove them.
//!
//! Usage:
//!   hardsec_context_cctm_monitor --plan plan.json \
//!     --stream stream.json --receipt receipt.json

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PLAN_SCHEMA: &str = "hardsec-context-restarted-cctm-plan-v1";
const STREAM_SCHEMA: &str = "hardsec-context-restarted-cctm-stream-v1";
const RECEIPT_SCHEMA: &str = "hardsec-context-restarted-cctm-receipt-v1";
const QUALIFIED_PMU: &str = "qualified_for_named_acquisition_conditions";
const MAX_ETA: f64 = 0.5;

#[derive(Debug, Clone)]
struct RestartState {
    start_index: usize,
    eta: f64,
    ons_accumulator: f64,
    log_wealth: Option<f64>,
}

#[derive(Debug, Clone)]
struct ContextState {
    reference_id: String,
    reference_group_count: usize,
    reference_digest: String,
    scores: Vec<f64>,
    alpha: f64,
    delta: f64,
    epsilon: f64,
    scale: f64,
    eta: f64,
    ons_accumulator: f64,
    log_wealth: Option<f64>,
    first_alarm_sequence_index: Option<usize>,
    restart_observation_count: usize,
    restart_weight_sum: f64,
    restart_states: Vec<RestartState>,
    restart_expired_state_count: usize,
    restart_log_e_value: f64,
    restart_first_alarm_sequence_index: Option<usize>,
}

#[derive(Debug, Clone)]
struct Plan {
    monitor_id: String,
    evidence_domain: String,
    stream_id: String,
    evaluation_group_id: String,
    scoring_artifact_sha256: String,
    context_assignment_policy_id: String,
    context_assignment_artifact_sha256: String,
    alpha_budget: f64,
    delta_budget: f64,
    smoothing_k: f64,
    restart_weight_scale: f64,
    maximum_restart_age: usize,
    contexts: BTreeMap<String, ContextState>,
}

#[derive(Debug, Clone)]
struct AssignmentDecision {
    sequence_index: usize,
    window_id: String,
    decision_status: String,
    decision_reason: String,
    selected_context_id: Option<String>,
}

#[derive(Debug, Clone)]
struct StreamRecord {
    sequence_index: usize,
    window_id: String,
    context_id: String,
    anomaly_score: f64,
    sensor_state: String,
    pmu_qualification_result: String,
}

#[derive(Debug, Clone)]
struct Args {
    plan: PathBuf,
    stream: PathBuf,
    context_receipt: PathBuf,
    receipt: PathBuf,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn nonempty(value: &str, context: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(format!("{context} must be non-empty and contain no control characters"));
    }
    Ok(())
}

fn string_field<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, String> {
    let result = value.get(key).and_then(Value::as_str)
        .ok_or_else(|| format!("{context}.{key} must be a string"))?;
    nonempty(result, &format!("{context}.{key}"))?;
    Ok(result)
}

fn f64_field(value: &Value, key: &str, context: &str) -> Result<f64, String> {
    let result = value.get(key).and_then(Value::as_f64)
        .ok_or_else(|| format!("{context}.{key} must be numeric"))?;
    if !result.is_finite() {
        return Err(format!("{context}.{key} must be finite"));
    }
    Ok(result)
}

fn usize_field(value: &Value, key: &str, context: &str) -> Result<usize, String> {
    let raw = value.get(key).and_then(Value::as_u64)
        .ok_or_else(|| format!("{context}.{key} must be a non-negative integer"))?;
    usize::try_from(raw).map_err(|_| format!("{context}.{key} is outside the supported range"))
}

fn count_field(value: &Value, key: &str, context: &str) -> Result<usize, String> {
    let raw = value.get(key).and_then(Value::as_u64)
        .ok_or_else(|| format!("{context}.{key} must be a non-negative integer"))?;
    usize::try_from(raw).map_err(|_| format!("{context}.{key} is outside the supported range"))
}

fn valid_sha(value: &str, context: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(format!("{context} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn string_array(value: &Value, key: &str, context: &str) -> Result<Vec<String>, String> {
    let values = value.get(key).and_then(Value::as_array)
        .ok_or_else(|| format!("{context}.{key} must be an array"))?;
    let mut seen = BTreeSet::new();
    let mut output = Vec::with_capacity(values.len());
    for (index, item) in values.iter().enumerate() {
        let item = item.as_str().ok_or_else(|| format!("{context}.{key}[{index}] must be a string"))?;
        nonempty(item, &format!("{context}.{key}[{index}]"))?;
        if !seen.insert(item.to_owned()) {
            return Err(format!("{context}.{key} contains duplicate entry {item}"));
        }
        output.push(item.to_owned());
    }
    Ok(output)
}

fn parse_plan(value: &Value, evaluation_group_id: &str) -> Result<Plan, String> {
    if string_field(value, "schema_version", "plan")? != PLAN_SCHEMA {
        return Err("unsupported CCTM plan schema_version".to_owned());
    }
    let monitor_id = string_field(value, "monitor_id", "plan")?.to_owned();
    let evidence_domain = string_field(value, "evidence_domain", "plan")?.to_owned();
    if !matches!(evidence_domain.as_str(), "synthetic_fixture" | "simulator" | "physical_hardware") {
        return Err(format!("unsupported evidence_domain {evidence_domain}"));
    }
    let stream_id = string_field(value, "stream_id", "plan")?.to_owned();
    if string_field(value, "evaluation_group_id", "plan")? != evaluation_group_id {
        return Err("stream evaluation_group_id does not match the preregistered plan".to_owned());
    }
    let scoring_artifact_sha256 = string_field(value, "scoring_artifact_sha256", "plan")?.to_owned();
    valid_sha(&scoring_artifact_sha256, "plan.scoring_artifact_sha256")?;
    let context_assignment_policy_id = string_field(value, "context_assignment_policy_id", "plan")?.to_owned();
    let context_assignment_artifact_sha256 = string_field(value, "context_assignment_artifact_sha256", "plan")?.to_owned();
    valid_sha(&context_assignment_artifact_sha256, "plan.context_assignment_artifact_sha256")?;
    if scoring_artifact_sha256 == context_assignment_artifact_sha256 {
        return Err("anomaly scoring and context assignment artifacts must be separately pinned".to_owned());
    }
    if string_field(value, "anomaly_score_direction", "plan")? != "higher_is_more_anomalous" {
        return Err("anomaly_score_direction must be higher_is_more_anomalous".to_owned());
    }
    if string_field(value, "reference_update_policy", "plan")? != "immutable_no_test_time_updates" {
        return Err("reference_update_policy must be immutable_no_test_time_updates".to_owned());
    }
    if string_field(value, "context_assignment_policy", "plan")? != "external_workload_metadata_not_score_derived" {
        return Err("context_assignment_policy must be external_workload_metadata_not_score_derived".to_owned());
    }
    for key in [
        "null_stream_assumption_status",
        "reference_iid_assumption_status",
        "context_assignment_assumption_status",
    ] {
        if string_field(value, key, "plan")? != "assumption_not_verified" {
            return Err(format!("{key} must remain assumption_not_verified"));
        }
    }

    let alpha_budget = f64_field(value, "alpha_budget", "plan")?;
    let delta_budget = f64_field(value, "delta_budget", "plan")?;
    if !(0.0..1.0).contains(&alpha_budget) || !(0.0..1.0).contains(&delta_budget)
        || alpha_budget + delta_budget >= 1.0 {
        return Err("alpha_budget and delta_budget must be in (0,1) and sum to less than one".to_owned());
    }
    let minimum_reference_scores = usize_field(value, "minimum_reference_scores", "plan")?;
    if minimum_reference_scores < 2 {
        return Err("minimum_reference_scores must be at least two".to_owned());
    }
    let smoothing_k = f64_field(value, "ons_smoothing_k", "plan")?;
    if smoothing_k <= 0.0 {
        return Err("ons_smoothing_k must be positive".to_owned());
    }
    if (f64_field(value, "ons_eta_bound", "plan")? - MAX_ETA).abs() > 1e-12 {
        return Err("v1 implements the published ONS eta bound of 0.5".to_owned());
    }
    if string_field(value, "restart_weight_schedule", "plan")?
        != "near_harmonic_c_over_k_log_e_plus_k_squared_v1" {
        return Err("unsupported restart_weight_schedule".to_owned());
    }
    let restart_weight_scale = f64_field(value, "restart_weight_scale", "plan")?;
    if (restart_weight_scale - (1.0 / 3.0)).abs() > 1e-12 {
        return Err("v1 requires the conservative near-harmonic restart scale c=1/3".to_owned());
    }
    let maximum_restart_age = usize_field(value, "maximum_restart_age", "plan")?;
    if maximum_restart_age == 0 {
        return Err("maximum_restart_age must be positive".to_owned());
    }

    let context_values = value.get("contexts").and_then(Value::as_array)
        .ok_or("plan.contexts must be an array")?;
    if context_values.is_empty() {
        return Err("plan.contexts must not be empty".to_owned());
    }
    let context_count = context_values.len() as f64;
    let alpha_per_context = alpha_budget / context_count;
    let delta_per_context = delta_budget / context_count;
    let mut contexts = BTreeMap::new();
    let mut all_reference_groups = BTreeSet::new();

    for (index, item) in context_values.iter().enumerate() {
        let context = format!("plan.contexts[{index}]");
        let context_id = string_field(item, "context_id", &context)?.to_owned();
        let reference_id = string_field(item, "reference_id", &context)?.to_owned();
        let reference_group_ids = string_array(item, "reference_group_ids", &context)?;
        let score_values = item.get("reference_scores").and_then(Value::as_array)
            .ok_or_else(|| format!("{context}.reference_scores must be an array"))?;
        if score_values.len() < minimum_reference_scores {
            return Err(format!("{context} has {} reference scores; at least {minimum_reference_scores} are required", score_values.len()));
        }
        // The DKW construction assumes i.i.d. reference samples. v1 requires a
        // distinct declared group ID for each row, but cannot authenticate
        // that those IDs are genuinely independent executions/units.
        if reference_group_ids.len() != score_values.len() {
            return Err(format!("{context} must provide one distinct reference_group_id per reference score"));
        }
        for group_id in &reference_group_ids {
            if group_id == evaluation_group_id {
                return Err(format!("evaluation group {evaluation_group_id} overlaps a null-reference group"));
            }
            if !all_reference_groups.insert(group_id.clone()) {
                return Err(format!("reference group {group_id} is reused across context references"));
            }
        }

        let mut scores = Vec::with_capacity(score_values.len());
        for (score_index, score_value) in score_values.iter().enumerate() {
            let score = score_value.as_f64()
                .ok_or_else(|| format!("{context}.reference_scores[{score_index}] must be numeric"))?;
            if !score.is_finite() {
                return Err(format!("{context}.reference_scores[{score_index}] must be finite"));
            }
            scores.push(score);
        }
        scores.sort_by(f64::total_cmp);
        let epsilon = ( (2.0 / delta_per_context).ln() / (2.0 * scores.len() as f64) ).sqrt();
        let scale = 1.0 / (0.5 + (1.0 + smoothing_k * smoothing_k).sqrt() * epsilon);
        let encoded = serde_json::to_vec(&scores).map_err(|e| format!("cannot encode reference scores: {e}"))?;
        let state = ContextState {
            reference_id,
            reference_group_count: reference_group_ids.len(),
            reference_digest: sha256(&encoded),
            scores,
            alpha: alpha_per_context,
            delta: delta_per_context,
            epsilon,
            scale,
            eta: 0.0,
            ons_accumulator: 1.0,
            log_wealth: Some(0.0),
            first_alarm_sequence_index: None,
            restart_observation_count: 0,
            restart_weight_sum: 0.0,
            restart_states: Vec::new(),
            restart_expired_state_count: 0,
            restart_log_e_value: 0.0,
            restart_first_alarm_sequence_index: None,
        };
        if contexts.insert(context_id.clone(), state).is_some() {
            return Err(format!("duplicate context_id {context_id}"));
        }
    }

    Ok(Plan {
        monitor_id,
        evidence_domain,
        stream_id,
        evaluation_group_id: evaluation_group_id.to_owned(),
        scoring_artifact_sha256,
        context_assignment_policy_id,
        context_assignment_artifact_sha256,
        alpha_budget,
        delta_budget,
        smoothing_k,
        restart_weight_scale,
        maximum_restart_age,
        contexts,
    })
}

fn parse_stream(value: &Value, plan: &Plan) -> Result<Vec<StreamRecord>, String> {
    if string_field(value, "schema_version", "stream")? != STREAM_SCHEMA {
        return Err("unsupported CCTM stream schema_version".to_owned());
    }
    if string_field(value, "evidence_domain", "stream")? != plan.evidence_domain {
        return Err("stream evidence_domain must match plan evidence_domain".to_owned());
    }
    if string_field(value, "stream_id", "stream")? != plan.stream_id {
        return Err("stream_id does not match the preregistered plan".to_owned());
    }
    if string_field(value, "evaluation_group_id", "stream")? != plan.evaluation_group_id {
        return Err("evaluation_group_id does not match the preregistered plan".to_owned());
    }
    if string_field(value, "scoring_artifact_sha256", "stream")? != plan.scoring_artifact_sha256 {
        return Err("stream scoring_artifact_sha256 does not match the plan".to_owned());
    }
    if string_field(value, "context_assignment_policy_id", "stream")? != plan.context_assignment_policy_id {
        return Err("stream context_assignment_policy_id does not match the plan".to_owned());
    }
    if string_field(value, "context_assignment_artifact_sha256", "stream")? != plan.context_assignment_artifact_sha256 {
        return Err("stream context_assignment_artifact_sha256 does not match the plan".to_owned());
    }

    let records = value.get("records").and_then(Value::as_array)
        .ok_or("stream.records must be an array")?;
    if records.is_empty() {
        return Err("stream.records must not be empty".to_owned());
    }
    let mut window_ids = BTreeSet::new();
    let mut output = Vec::with_capacity(records.len());
    for (index, item) in records.iter().enumerate() {
        let context = format!("stream.records[{index}]");
        let sequence_index = usize_field(item, "sequence_index", &context)?;
        if sequence_index != index + 1 {
            return Err(format!("{context}.sequence_index must be contiguous from one; expected {}, got {sequence_index}", index + 1));
        }
        let window_id = string_field(item, "window_id", &context)?.to_owned();
        if !window_ids.insert(window_id.clone()) {
            return Err(format!("duplicate window_id {window_id}"));
        }
        let context_id = string_field(item, "context_id", &context)?.to_owned();
        let anomaly_score = f64_field(item, "anomaly_score", &context)?;
        let sensor_state = string_field(item, "sensor_state", &context)?.to_owned();
        if !matches!(sensor_state.as_str(), "valid" | "missing" | "stale" | "misaligned" | "multiplexed" | "untrusted") {
            return Err(format!("{context}.sensor_state has unsupported state {sensor_state}"));
        }
        let pmu_qualification_result = string_field(item, "pmu_qualification_result", &context)?.to_owned();
        output.push(StreamRecord {
            sequence_index,
            window_id,
            context_id,
            anomaly_score,
            sensor_state,
            pmu_qualification_result,
        });
    }
    Ok(output)
}

fn parse_context_assignment_receipt(
    value: &Value,
    plan: &Plan,
    records: &[StreamRecord],
) -> Result<Vec<AssignmentDecision>, String> {
    if string_field(value, "schema_version", "context_receipt")? != "hardsec-context-assignment-receipt-v1" {
        return Err("unsupported context assignment receipt schema_version".to_owned());
    }
    if string_field(value, "result", "context_receipt")? != "audited" {
        return Err("context assignment receipt result must be audited".to_owned());
    }
    if string_field(value, "claim_scope", "context_receipt")?
        != "declared_context_assignment_gate_and_heldout_label_evaluation_only" {
        return Err("unexpected context assignment receipt claim_scope".to_owned());
    }
    for (field, expected) in [
        ("policy_id", plan.context_assignment_policy_id.as_str()),
        ("stream_id", plan.stream_id.as_str()),
        ("evaluation_group_id", plan.evaluation_group_id.as_str()),
        ("evidence_domain", plan.evidence_domain.as_str()),
        ("assignment_artifact_sha256", plan.context_assignment_artifact_sha256.as_str()),
        ("anomaly_score_artifact_sha256", plan.scoring_artifact_sha256.as_str()),
    ] {
        if string_field(value, field, "context_receipt")? != expected {
            return Err(format!("context assignment receipt {field} does not match CCTM plan/stream"));
        }
    }
    let inputs = value.get("inputs").ok_or("context assignment receipt inputs missing")?;
    for field in ["policy_sha256", "assignments_sha256", "truth_oracle_sha256"] {
        let digest = string_field(inputs, field, "context_receipt.inputs")?;
        valid_sha(digest, &format!("context_receipt.inputs.{field}"))?;
    }
    let mapping = value.get("runtime_mapping").ok_or("context assignment receipt runtime_mapping missing")?;
    if string_field(mapping, "mapping_schema_version", "context_receipt.runtime_mapping")?
        != "hardsec-context-assignment-runtime-map-v1" {
        return Err("unsupported context assignment runtime map schema".to_owned());
    }
    if mapping.get("contains_truth_labels").and_then(Value::as_bool) != Some(false) {
        return Err("runtime context assignment mapping must not contain truth labels".to_owned());
    }
    if mapping.get("mapping_digest_bound_to_assignment_input").and_then(Value::as_bool) != Some(true) {
        return Err("runtime context assignment mapping must be digest-bound to assignment input".to_owned());
    }
    let assignments = mapping.get("assignments").and_then(Value::as_array)
        .ok_or("context assignment receipt runtime_mapping.assignments must be an array")?;
    if assignments.len() != records.len() {
        return Err("context assignment receipt must map every CCTM stream record exactly once".to_owned());
    }
    let integrity = value.get("assignment_integrity").ok_or("context assignment receipt assignment_integrity missing")?;
    let accepted_count = assignments.iter().filter(|a| a.get("decision_status").and_then(Value::as_str) == Some("accepted")).count();
    let abstained_count = assignments.iter().filter(|a| a.get("decision_status").and_then(Value::as_str) == Some("abstained")).count();
    let reported_total = count_field(integrity, "assignment_record_count", "context_receipt.assignment_integrity")?;
    let reported_accepted = count_field(integrity, "accepted_count", "context_receipt.assignment_integrity")?;
    let reported_abstained = count_field(integrity, "abstained_count", "context_receipt.assignment_integrity")?;
    let reported_accuracy_denominator = count_field(integrity, "accepted_accuracy_denominator", "context_receipt.assignment_integrity")
        .or_else(|error| {
            if accepted_count == 0
                && integrity.get("accepted_accuracy_denominator").and_then(Value::as_u64) == Some(0) {
                Ok(0)
            } else { Err(error) }
        })?;
    let reported_coverage = f64_field(integrity, "accepted_coverage", "context_receipt.assignment_integrity")?;
    let reported_end_to_end = f64_field(integrity, "end_to_end_correct_context_coverage", "context_receipt.assignment_integrity")?;
    if reported_total != assignments.len()
        || reported_accepted != accepted_count
        || reported_abstained != abstained_count
        || reported_accepted + reported_abstained != reported_total
        || reported_accuracy_denominator != accepted_count
        || (reported_coverage - accepted_count as f64 / assignments.len() as f64).abs() > 1e-9
        || !(0.0..=1.0).contains(&reported_end_to_end) {
        return Err("context assignment receipt aggregate counts/coverage do not match its runtime mapping".to_owned());
    }
    match (integrity.get("accepted_accuracy"), accepted_count) {
        (Some(Value::Null), 0) => {}
        (Some(value), count) if count > 0 => {
            let accuracy = value.as_f64().ok_or("accepted_accuracy must be numeric when accepted_count > 0")?;
            if !(0.0..=1.0).contains(&accuracy) {
                return Err("accepted_accuracy must be in [0,1]".to_owned());
            }
        }
        _ => return Err("accepted_accuracy nullability does not match accepted_count".to_owned()),
    }

    let mut output = Vec::with_capacity(assignments.len());
    for (index, (assignment, stream_record)) in assignments.iter().zip(records.iter()).enumerate() {
        let context = format!("context_receipt.runtime_mapping.assignments[{index}]");
        let sequence_index = usize_field(assignment, "sequence_index", &context)?;
        let window_id = string_field(assignment, "window_id", &context)?.to_owned();
        if sequence_index != stream_record.sequence_index || window_id != stream_record.window_id {
            return Err(format!("{context} sequence/window identity does not match the CCTM stream"));
        }
        let decision_status = string_field(assignment, "decision_status", &context)?.to_owned();
        if !matches!(decision_status.as_str(), "accepted" | "abstained") {
            return Err(format!("{context}.decision_status must be accepted or abstained"));
        }
        let decision_reason = string_field(assignment, "decision_reason", &context)?.to_owned();
        if !matches!(decision_reason.as_str(), "accepted" | "invalid_sensor" | "unknown_context" | "ambiguous_assignment") {
            return Err(format!("{context}.decision_reason is unsupported"));
        }
        let selected_context_id = match assignment.get("selected_context_id") {
            Some(Value::String(value)) => Some(value.clone()),
            Some(Value::Null) => None,
            _ => return Err(format!("{context}.selected_context_id must be a string or null")),
        };
        if decision_status == "accepted" {
            if decision_reason != "accepted" || selected_context_id.as_deref() != Some(stream_record.context_id.as_str()) {
                return Err(format!("{context} accepted context does not match the CCTM stream's context"));
            }
            if !plan.contexts.contains_key(stream_record.context_id.as_str()) {
                return Err(format!("{context} accepted an undeclared CCTM context"));
            }
        } else if selected_context_id.is_some() {
            return Err(format!("{context} abstained assignment must not select a context"));
        }
        output.push(AssignmentDecision {
            sequence_index,
            window_id,
            decision_status,
            decision_reason,
            selected_context_id,
        });
    }
    Ok(output)
}

// Fixed-reference empirical CDF: F_hat_0(x) = #{reference <= x} / n.
// The reference is sorted once so each stream score uses O(log n) search.
fn empirical_cdf(reference_scores: &[f64], score: f64) -> f64 {
    let count = reference_scores.partition_point(|reference| *reference <= score);
    count as f64 / reference_scores.len() as f64
}

// Published conditional CTM betting function, Algorithm 1 / equations (7)-(8).
fn cctm_bet(
    p_value: f64,
    eta: f64,
    epsilon: f64,
    smoothing_k: f64,
    scale: f64,
) -> Result<(f64, f64, f64), String> {
    if !(0.0..=1.0).contains(&p_value) {
        return Err("CCTM empirical CDF value must be in [0,1]".to_owned());
    }
    let root = (eta * eta + smoothing_k * smoothing_k).sqrt();
    let g = scale * (eta * (p_value - 0.5) - root * epsilon);
    let beta = 1.0 + g;
    if !beta.is_finite() || beta < 0.0 || beta > 2.0 {
        return Err(format!("CCTM betting factor {beta} lies outside [0,2]"));
    }
    let g_prime = scale * (p_value - 0.5 - (eta / root) * epsilon);
    if !g_prime.is_finite() {
        return Err("CCTM ONS derivative is non-finite".to_owned());
    }
    Ok((beta, g, g_prime))
}

// Published Algorithm 2 ONS update. The current observation determines eta
// for the next observation only; eta_t is predictable at the time of betting.
fn ons_next_eta(eta: f64, accumulator: f64, g_prime: f64, beta: f64) -> Result<(f64, f64), String> {
    if beta <= 0.0 {
        return Err("CCTM ONS update cannot divide by a zero betting factor".to_owned());
    }
    let z = g_prime / beta;
    let next_accumulator = accumulator + z * z;
    if !next_accumulator.is_finite() || next_accumulator <= 0.0 {
        return Err("CCTM ONS accumulator became invalid".to_owned());
    }
    let next_eta = (eta + 4.0 * z / next_accumulator).clamp(-MAX_ETA, MAX_ETA);
    if !next_eta.is_finite() {
        return Err("CCTM ONS eta became non-finite".to_owned());
    }
    Ok((next_eta, next_accumulator))
}

fn restart_weight(start_index: usize, scale: f64) -> f64 {
    let k = start_index as f64;
    scale / (k * (std::f64::consts::E + k).ln().powi(2))
}

fn restart_weight_sum_upper_bound(scale: f64) -> f64 {
    // f(k)=1/[k ln(e+k)^2] is decreasing. For k>=3,
    // sum f(k) <= integral_2^inf dx/[x ln(x)^2] = 1/ln(2).
    // Adding f(1) and f(2), the unscaled sum is <= 2.23025...;
    // c=1/3 therefore leaves a strictly positive reserve (<0.744).
    scale * (
        1.0 / (std::f64::consts::E + 1.0).ln().powi(2)
        + 1.0 / (2.0 * (std::f64::consts::E + 2.0).ln().powi(2))
        + 1.0 / 2.0_f64.ln()
    )
}

fn log_sum_exp(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("log_sum_exp requires at least one finite term".to_owned());
    }
    let max_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_value.is_finite() {
        return Err("restart aggregate has no finite wealth terms".to_owned());
    }
    let sum = values.iter().map(|value| (*value - max_value).exp()).sum::<f64>();
    let result = max_value + sum.ln();
    if !result.is_finite() {
        return Err("restart aggregate log e-value became non-finite".to_owned());
    }
    Ok(result)
}

#[derive(Debug, Clone)]
struct RestartUpdate {
    context_observation_index: usize,
    restart_weight: f64,
    reserve: f64,
    log_e_value: f64,
    live_state_count: usize,
    expired_this_observation: usize,
    total_expired_state_count: usize,
}

fn update_restarted_processes(
    context_state: &mut ContextState,
    p_value: f64,
    epsilon: f64,
    smoothing_k: f64,
    scale_c: f64,
    maximum_restart_age: usize,
    sequence_index: usize,
) -> Result<RestartUpdate, String> {
    context_state.restart_observation_count += 1;
    let observation_index = context_state.restart_observation_count;
    let new_weight = restart_weight(observation_index, scale_c);
    if !new_weight.is_finite() || new_weight <= 0.0 {
        return Err("restart weight must be finite and positive".to_owned());
    }
    context_state.restart_weight_sum += new_weight;
    if !context_state.restart_weight_sum.is_finite()
        || context_state.restart_weight_sum > 1.0 + 1e-12 {
        return Err("near-harmonic restart weights exceed the unit capital budget".to_owned());
    }
    context_state.restart_states.push(RestartState {
        start_index: observation_index,
        eta: 0.0,
        ons_accumulator: 1.0,
        log_wealth: Some(0.0),
    });

    // Every live local process receives the current observation. Its ONS
    // parameter was fixed using only earlier observations in that local run.
    for local in context_state.restart_states.iter_mut() {
        let (beta, _, g_prime) = cctm_bet(
            p_value, local.eta, epsilon, smoothing_k, scale_c
        )?;
        if beta == 0.0 {
            local.log_wealth = None;
            continue;
        }
        local.log_wealth = match local.log_wealth {
            Some(previous) => Some(previous + beta.ln()),
            None => None,
        };
        if local.log_wealth.is_some() {
            let (eta_next, accumulator_next) =
                ons_next_eta(local.eta, local.ons_accumulator, g_prime, beta)?;
            local.eta = eta_next;
            local.ons_accumulator = accumulator_next;
        }
    }

    // Unallocated restart capital is retained as a constant reserve. Since
    // sum_k w_k <= 0.743417... for c=1/3, reserve is always strictly positive.
    let reserve = 1.0 - context_state.restart_weight_sum;
    if reserve < -1e-12 {
        return Err("restart capital reserve became negative".to_owned());
    }
    let reserve = reserve.max(0.0);
    let mut log_terms = vec![reserve.ln()];
    for local in &context_state.restart_states {
        if let Some(log_wealth) = local.log_wealth {
            let weight = restart_weight(local.start_index, scale_c);
            log_terms.push(weight.ln() + log_wealth);
        }
    }
    let log_e_value = log_sum_exp(&log_terms)?;
    context_state.restart_log_e_value = log_e_value;

    // The local process is stopped after exactly maximum_restart_age accepted
    // observations. Removing its nonnegative capital at the predictable age
    // is an absorbing kill; it cannot increase the aggregate e-process.
    let before_expiry = context_state.restart_states.len();
    context_state.restart_states.retain(|local| {
        observation_index.saturating_sub(local.start_index) + 1 < maximum_restart_age
    });
    let expired = before_expiry - context_state.restart_states.len();
    context_state.restart_expired_state_count += expired;

    // Sequence index is carried for audit readability and is not part of
    // the per-context restart clock.
    let _ = sequence_index;
    Ok(RestartUpdate {
        context_observation_index: observation_index,
        restart_weight: new_weight,
        reserve,
        log_e_value,
        live_state_count: before_expiry,
        expired_this_observation: expired,
        total_expired_state_count: context_state.restart_expired_state_count,
    })
}

fn abstention_reason(
    record: &StreamRecord,
    assignment: &AssignmentDecision,
    plan: &Plan,
) -> Option<&'static str> {
    if record.sensor_state != "valid" {
        return Some("invalid_sensor_state");
    }
    if record.pmu_qualification_result != QUALIFIED_PMU {
        return Some("pmu_acquisition_not_qualified");
    }
    if assignment.decision_status != "accepted" {
        return Some("context_assignment_not_accepted");
    }
    if !plan.contexts.contains_key(&record.context_id) {
        return Some("unseen_workload_context");
    }
    None
}

fn audit_values_with_context_receipt(
    plan_value: &Value,
    stream_value: &Value,
    context_receipt_value: &Value,
    plan_bytes: &[u8],
    stream_bytes: &[u8],
    context_receipt_bytes: &[u8],
) -> Result<Value, String> {
    let evaluation_group_id = string_field(stream_value, "evaluation_group_id", "stream")?;
    let mut plan = parse_plan(plan_value, evaluation_group_id)?;
    let records = parse_stream(stream_value, &plan)?;
    let assignment_decisions = parse_context_assignment_receipt(context_receipt_value, &plan, &records)?;
    let mut first_single_start_alarm_sequence_index: Option<usize> = None;
    let mut first_single_start_alarm_context_id: Option<String> = None;
    let mut first_restart_alarm_sequence_index: Option<usize> = None;
    let mut first_restart_alarm_context_id: Option<String> = None;
    let mut output_records = Vec::with_capacity(records.len());
    let mut abstention_counts: BTreeMap<String, usize> = BTreeMap::new();
    let smoothing_k = plan.smoothing_k;
    let restart_weight_scale = plan.restart_weight_scale;
    let maximum_restart_age = plan.maximum_restart_age;

    for (record, assignment) in records.iter().zip(assignment_decisions.iter()) {
        if let Some(reason) = abstention_reason(record, assignment, &plan) {
            *abstention_counts.entry(reason.to_owned()).or_default() += 1;
            let context_log_wealth = plan.contexts.get(&record.context_id).map(|state| state.log_wealth);
            output_records.push(json!({
                "sequence_index":record.sequence_index,
                "window_id":record.window_id,
                "context_id":record.context_id,
                "context_assignment_status":assignment.decision_status,
                "context_assignment_reason":assignment.decision_reason,
                "action":if first_restart_alarm_sequence_index.is_some() {"alert_latched_observation_abstained"} else {"abstain"},
                "single_start_action":if first_single_start_alarm_sequence_index.is_some() {"alert_latched_observation_abstained"} else {"abstain"},
                "abstention_reason":reason,
                "ecdf_value":null,
                "betting_factor":null,
                "single_start_log_wealth_context":context_log_wealth,
                "restart_log_e_value_context":plan.contexts.get(&record.context_id).map(|s| s.restart_log_e_value),
                "threshold_context":plan.contexts.get(&record.context_id).map(|s| -s.alpha.ln())
            }));
            continue;
        }

        let state = plan.contexts.get_mut(&record.context_id).ok_or("context state invariant failed")?;
        let p_value = empirical_cdf(&state.scores, record.anomaly_score);
        let eta_used = state.eta;
        let epsilon = state.epsilon;
        let score_scale = state.scale;
        let (beta, g, g_prime) = cctm_bet(p_value, eta_used, epsilon, smoothing_k, score_scale)?;
        let (eta_next, next_accumulator) = if beta > 0.0 {
            ons_next_eta(eta_used, state.ons_accumulator, g_prime, beta)?
        } else {
            (state.eta, state.ons_accumulator)
        };
        state.log_wealth = match (state.log_wealth, beta) {
            (Some(_), 0.0) => None,
            (Some(previous), factor) => Some(previous + factor.ln()),
            (None, _) => None,
        };
        state.eta = eta_next;
        state.ons_accumulator = next_accumulator;
        let threshold = -state.alpha.ln();

        let single_crossing_now = state.first_alarm_sequence_index.is_none()
            && state.log_wealth.is_some_and(|wealth| wealth >= threshold);
        if single_crossing_now {
            state.first_alarm_sequence_index = Some(record.sequence_index);
            if first_single_start_alarm_sequence_index.is_none() {
                first_single_start_alarm_sequence_index = Some(record.sequence_index);
                first_single_start_alarm_context_id = Some(record.context_id.clone());
            }
        }

        let restarted = update_restarted_processes(
            state, p_value, epsilon, smoothing_k, restart_weight_scale,
            maximum_restart_age, record.sequence_index
        )?;
        let restart_crossing_now = state.restart_first_alarm_sequence_index.is_none()
            && restarted.log_e_value >= threshold;
        if restart_crossing_now {
            state.restart_first_alarm_sequence_index = Some(record.sequence_index);
            if first_restart_alarm_sequence_index.is_none() {
                first_restart_alarm_sequence_index = Some(record.sequence_index);
                first_restart_alarm_context_id = Some(record.context_id.clone());
            }
        }

        let action = if restart_crossing_now {
            "alarm_crossing"
        } else if first_restart_alarm_sequence_index.is_some() {
            "alert_latched"
        } else {
            "monitoring_no_alarm_yet"
        };
        let single_start_action = if single_crossing_now {
            "single_start_alarm_crossing"
        } else if first_single_start_alarm_sequence_index.is_some() {
            "single_start_alarm_latched"
        } else {
            "single_start_monitoring"
        };
        output_records.push(json!({
            "sequence_index":record.sequence_index,
            "window_id":record.window_id,
            "context_id":record.context_id,
            "context_assignment_status":assignment.decision_status,
            "context_assignment_reason":assignment.decision_reason,
            "reference_id":state.reference_id,
            "ecdf_value":p_value,
            "epsilon_dkw":state.epsilon,
            "single_start":{
                "ons_eta_used":eta_used,
                "ons_eta_next":eta_next,
                "ons_accumulator":next_accumulator,
                "cctm_g":g,
                "betting_factor":beta,
                "log_wealth_context":state.log_wealth,
                "crossing_now":single_crossing_now,
                "action":single_start_action
            },
            "restarted":{
                "context_observation_index":restarted.context_observation_index,
                "new_start_weight":restarted.restart_weight,
                "unallocated_capital_reserve":restarted.reserve,
                "live_state_count":restarted.live_state_count,
                "expired_state_count":restarted.expired_this_observation,
                "total_expired_state_count":restarted.total_expired_state_count,
                "log_e_value":restarted.log_e_value,
                "crossing_now":restart_crossing_now,
                "threshold_log_e_value":threshold,
                "action":action
            },
            "threshold_context":threshold,
            "action":action,
            "abstention_reason":null
        }));
    }

    let context_reports: Vec<Value> = plan.contexts.iter().map(|(context_id, state)| json!({
        "context_id":context_id,
        "reference_id":state.reference_id,
        "reference_digest":state.reference_digest,
        "reference_score_count":state.scores.len(),
        "declared_reference_independent_group_count":state.reference_group_count,
        "alpha_allocation":state.alpha,
        "delta_allocation":state.delta,
        "dkw_epsilon":state.epsilon,
        "betting_scale_C":state.scale,
        "final_log_wealth":state.log_wealth,
        "ons_eta_final":state.eta,
        "single_start_log_wealth_final":state.log_wealth,
        "single_start_first_alarm_sequence_index":state.first_alarm_sequence_index,
        "restart_policy":{
            "weight_schedule":"c/[k*ln(e+k)^2]",
            "weight_scale_c":plan.restart_weight_scale,
            "maximum_restart_age":plan.maximum_restart_age,
            "restart_observation_count":state.restart_observation_count,
            "restart_weight_sum":state.restart_weight_sum,
            "unallocated_capital_reserve":1.0-state.restart_weight_sum,
            "live_restart_state_count":state.restart_states.len(),
            "expired_restart_state_count":state.restart_expired_state_count,
            "restart_log_e_value_final":state.restart_log_e_value,
            "first_alarm_sequence_index":state.restart_first_alarm_sequence_index
        },
        "threshold_log_wealth":-state.alpha.ln()
    })).collect();

    let result = if first_restart_alarm_sequence_index.is_some() {
        "restarted_alarm_threshold_crossed"
    } else {
        "no_restarted_alarm_threshold_crossed"
    };
    let valid_count = records.len() - abstention_counts.values().sum::<usize>();
    let context_count = plan.contexts.len();

    Ok(json!({
        "schema_version":RECEIPT_SCHEMA,
        "result":result,
        "claim_scope":"dkw_corrected_fixed_reference_cctm_with_summable_weighted_restarts_under_unverified_iid_assumptions",
        "monitor_id":plan.monitor_id,
        "stream_id":plan.stream_id,
        "evidence_domain":plan.evidence_domain,
        "evaluation_group_id":plan.evaluation_group_id,
        "scoring_artifact_sha256":plan.scoring_artifact_sha256,
        "context_assignment_policy_id":plan.context_assignment_policy_id,
        "context_assignment_artifact_sha256":plan.context_assignment_artifact_sha256,
        "context_assignment":{
            "receipt_sha256":sha256(context_receipt_bytes),
            "accepted_count":assignment_decisions.iter().filter(|a| a.decision_status=="accepted").count(),
            "abstained_count":assignment_decisions.iter().filter(|a| a.decision_status=="abstained").count(),
            "truth_labels_in_runtime_mapping":false,
            "exogeneity_status":"not_established_from_input_receipts"
        },
        "inputs":{
            "plan_sha256":sha256(plan_bytes),
            "stream_sha256":sha256(stream_bytes),
            "context_assignment_receipt_sha256":sha256(context_receipt_bytes)
        },
        "reference_policy":{
            "update_policy":"immutable_no_test_time_updates",
            "test_time_reference_updates":0,
            "evaluation_group_separated_from_reference_groups":true,
            "contexts":context_reports
        },
        "sequential_monitor":{
            "alpha_budget_total":plan.alpha_budget,
            "delta_budget_total":plan.delta_budget,
            "context_count":context_count,
            "overall_first_alarm_sequence_index":first_restart_alarm_sequence_index,
            "overall_first_alarm_context_id":first_restart_alarm_context_id,
            "single_start_reference":{
                "first_alarm_sequence_index":first_single_start_alarm_sequence_index,
                "first_alarm_context_id":first_single_start_alarm_context_id
            },
            "restart_guarantee_contract":{
                "weight_schedule":"w_k=c/[k*ln(e+k)^2]",
                "weight_scale_c":plan.restart_weight_scale,
                "infinite_weight_sum_upper_bound":restart_weight_sum_upper_bound(plan.restart_weight_scale),
                "weight_sum_at_most_one":restart_weight_sum_upper_bound(plan.restart_weight_scale) <= 1.0,
                "maximum_restart_age":plan.maximum_restart_age,
                "expiry_semantics":"predictable_deterministic_absorbing_kill_of_nonnegative_local_process"
            },
            "valid_observation_count":valid_count,
            "abstention_count":abstention_counts.values().sum::<usize>(),
            "abstention_counts_by_reason":abstention_counts,
            "alarm_is_latched":first_alarm_sequence_index.is_some(),
            "post_alarm_reset_count":0,
            "window_decisions":output_records
        },
        "statistical_guarantees":{
            "method":"conditional_conformal_test_martingale_with_dkw_ecdf_correction",
            "reference_cdf_confidence":1.0-plan.delta_budget,
            "conditional_type_i_error_budget_if_all_context_reference_bands_hold":plan.alpha_budget,
            "unconditional_type_i_upper_bound_if_all_paper_assumptions_hold":(plan.alpha_budget+plan.delta_budget).min(1.0),
            "null_stream_iid_assumption_status":"assumption_not_verified",
            "reference_sample_iid_assumption_status":"assumption_not_verified",
            "context_assignment_exogeneity_status":"assumption_not_verified",
            "scoring_function_fixed_status":"artifact_digest_bound_not_authenticated",
            "theorem_guarantee_status":"not_established_from_input_receipts",
            "effective_sample_size_estimate":null,
            "independence_status":"not_established"
        },
        "theoretical_contract":{
            "null_hypothesis":"within each declared context, reference and test scores are i.i.d. from the same unknown context-specific distribution",
            "reference_confidence_bound":"DKW epsilon = sqrt(ln(2/delta_context)/(2*n_context))",
            "betting_function":"1 + C*(eta*(Fhat_0(x)-0.5) - sqrt(eta^2+k^2)*epsilon)",
            "ons_update":"z=g_prime/b; a_next=a+z^2; eta_next=clip(eta+4*z/a_next,-0.5,0.5)",
            "restart_aggregation":"E_t = 1 - sum_{k<=t} w_k + sum_{live k<=t} w_k*M_{k,t}; each local M starts at one and uses fresh predictable ONS state",
            "restart_weight_schedule":"w_k = 1/(3*k*ln(e+k)^2); infinite sum is bounded above by (1/3)*(1/ln(e+1)^2 + 1/(2*ln(e+2)^2) + 1/ln(2)) < 0.744",
            "restart_expiry":"each local process is killed after maximum_restart_age observations; deterministic stopping preserves nonnegative supermartingale behavior",
            "context_allocation":"Bonferroni alpha_context=alpha_total/K and delta_context=delta_total/K",
            "scope":"union-bound control across the predeclared finite context set, conditional on all DKW reference bands and the within-context i.i.d./external-context assumptions"
        },
        "limitations":[
            "The published CCTM theorem assumes i.i.d. null test scores and i.i.d. calibration scores within each context; it does not provide a guarantee for arbitrary autocorrelated or adversarially dependent PMU windows.",
            "Unique reference group IDs and input digests are caller-declared; the receipt cannot authenticate group identity, independence or the actual scoring implementation.",
            "Context assignment must be external to, and not selected from, the same anomaly score being tested; this assumption is not verified.",
            "The single-start DKW-CCTM is retained as an ablation; the restarted sum is a derived composition of local DKW-CCTM processes and deterministic weights, not a direct reproduction of the Bhattacharyya-Ramdas conformal CMM algorithm.",
            "The restarted PFA argument requires each local fixed-reference CCTM process to be a nonnegative supermartingale under the same conditional i.i.d. assumptions; those assumptions are not established by this receipt.",
            "Finite restart expiry is a predictable absorbing kill of local nonnegative processes; it can reduce power and must not be described as a restart/reset of accumulated capital.",
            "The monitor is not yet an independent reproduction of the authors' experiment suite or official Python package.",
            "A boundary crossing does not establish a hardware attack or causal attack effect; it indicates score-distribution change under the supplied reference/model."
        ]
    }))
}

fn audit_values(
    plan_value: &Value,
    stream_value: &Value,
    plan_bytes: &[u8],
    stream_bytes: &[u8],
) -> Result<Value, String> {
    let bytes = include_bytes!("../tests/fixtures/hardsec/context-restarted-v1/runtime-receipt.json");
    let receipt: Value = serde_json::from_slice(bytes)
        .map_err(|e| format!("invalid bundled context receipt fixture: {e}"))?;
    audit_values_with_context_receipt(plan_value, stream_value, &receipt, plan_bytes, stream_bytes, bytes)
}

fn read_input(path: &Path, label: &str) -> Result<(Vec<u8>, Value), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| format!("cannot stat {label} {}: {e}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let bytes = fs::read(path).map_err(|e| format!("cannot read {label}: {e}"))?;
    let value = serde_json::from_slice(&bytes).map_err(|e| format!("invalid {label} JSON: {e}"))?;
    Ok((bytes, value))
}

fn parse_args() -> Result<Args, String> {
    let mut plan = None;
    let mut stream = None;
    let mut context_receipt = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let target = match flag.as_str() {
            "--plan" => &mut plan,
            "--stream" => &mut stream,
            "--context-receipt" => &mut context_receipt,
            "--receipt" => &mut receipt,
            _ => return Err(format!("unknown argument {flag}")),
        };
        let path = args.next().ok_or_else(|| format!("{flag} requires a path"))?;
        if target.replace(PathBuf::from(path)).is_some() {
            return Err(format!("duplicate argument {flag}"));
        }
    }
    Ok(Args {
        plan:plan.ok_or("missing required argument --plan")?,
        stream:stream.ok_or("missing required argument --stream")?,
        context_receipt:context_receipt.ok_or("missing required argument --context-receipt")?,
        receipt:receipt.ok_or("missing required argument --receipt")?,
    })
}

fn validate_receipt_location(receipt: &Path, inputs: &[&Path]) -> Result<(), String> {
    if receipt.exists() {
        return Err(format!("refusing to overwrite receipt {}", receipt.display()));
    }
    let parent = receipt.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let canonical_parent = fs::canonicalize(parent).map_err(|e| format!("receipt parent must exist: {e}"))?;
    for input in inputs {
        let canonical_input = fs::canonicalize(input)
            .map_err(|e| format!("cannot canonicalize input {}: {e}", input.display()))?;
        let input_parent = canonical_input.parent().ok_or("input has no parent directory")?;
        if canonical_parent.starts_with(input_parent) {
            return Err("receipt must be outside every input directory".to_owned());
        }
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let (plan_bytes, plan) = read_input(&args.plan, "plan")?;
    let (stream_bytes, stream) = read_input(&args.stream, "stream")?;
    let (context_receipt_bytes, context_receipt) = read_input(&args.context_receipt, "context assignment receipt")?;
    validate_receipt_location(&args.receipt, &[
        args.plan.as_path(), args.stream.as_path(), args.context_receipt.as_path()
    ])?;
    let receipt = audit_values_with_context_receipt(
        &plan, &stream, &context_receipt,
        &plan_bytes, &stream_bytes, &context_receipt_bytes
    )?;
    let encoded = serde_json::to_vec_pretty(&receipt).map_err(|e| format!("cannot encode receipt: {e}"))?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(&args.receipt)
        .map_err(|e| format!("cannot create receipt without overwrite: {e}"))?;
    file.write_all(&encoded).map_err(|e| format!("cannot write receipt: {e}"))?;
    file.write_all(b"\n").map_err(|e| format!("cannot finish receipt: {e}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("HARDSEC-019 CCTM monitor failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Value, Value, Vec<u8>, Vec<u8>) {
        let plan_bytes = include_bytes!("../tests/fixtures/hardsec/context-restarted-v1/plan.json").to_vec();
        let stream_bytes = include_bytes!("../tests/fixtures/hardsec/context-restarted-v1/stream.json").to_vec();
        let plan = serde_json::from_slice(&plan_bytes).unwrap();
        let stream = serde_json::from_slice(&stream_bytes).unwrap();
        (plan, stream, plan_bytes, stream_bytes)
    }

    fn encode(value: &Value) -> Vec<u8> {
        let mut bytes = serde_json::to_vec_pretty(value).unwrap();
        bytes.push(b'\n');
        bytes
    }

    fn audit(plan: &Value, stream: &Value) -> Result<Value, String> {
        let plan_bytes = encode(plan);
        let stream_bytes = encode(stream);
        audit_values(plan, stream, &plan_bytes, &stream_bytes)
    }

    #[test]
    fn known_answer_uses_dkw_correction_ons_and_context_alpha_allocation() {
        let (p, s, pb, sb) = fixture();
        let receipt = audit_values(&p, &s, &pb, &sb).unwrap();
        assert_eq!(receipt["result"], "restarted_alarm_threshold_crossed");
        assert_eq!(receipt["sequential_monitor"]["overall_first_alarm_sequence_index"], 22);
        assert_eq!(receipt["sequential_monitor"]["overall_first_alarm_context_id"], "compute");
        assert_eq!(receipt["sequential_monitor"]["single_start_reference"]["first_alarm_sequence_index"], 16);
        assert_eq!(receipt["sequential_monitor"]["valid_observation_count"], 23);
        assert_eq!(receipt["sequential_monitor"]["abstention_count"], 3);
        assert_eq!(receipt["sequential_monitor"]["restart_guarantee_contract"]["weight_sum_at_most_one"], true);
        assert!(receipt["sequential_monitor"]["restart_guarantee_contract"]["infinite_weight_sum_upper_bound"].as_f64().unwrap() < 0.744);
        assert_eq!(receipt["sequential_monitor"]["post_alarm_reset_count"], 0);
        assert_eq!(receipt["statistical_guarantees"]["null_stream_iid_assumption_status"], "assumption_not_verified");
        assert_eq!(receipt["statistical_guarantees"]["reference_sample_iid_assumption_status"], "assumption_not_verified");
        assert_eq!(receipt["statistical_guarantees"]["theorem_guarantee_status"], "not_established_from_input_receipts");
        let unconditional = receipt["statistical_guarantees"]["unconditional_type_i_upper_bound_if_all_paper_assumptions_hold"].as_f64().unwrap();
        assert!((unconditional - 0.05).abs() < 1e-12);
        assert_eq!(receipt["statistical_guarantees"]["effective_sample_size_estimate"], Value::Null);
    }

    #[test]
    fn dkw_epsilon_matches_preregistered_delta_and_reference_size() {
        let (p, s, pb, sb) = fixture();
        let receipt = audit_values(&p, &s, &pb, &sb).unwrap();
        let epsilon = receipt["reference_policy"]["contexts"][0]["dkw_epsilon"].as_f64().unwrap();
        let expected = ((2.0_f64 / 0.005).ln() / (2.0 * 600.0)).sqrt();
        assert!((epsilon - expected).abs() < 1e-12);
    }

    #[test]
    fn alpha_and_delta_are_bonferroni_allocated_over_contexts() {
        let (p, s, pb, sb) = fixture();
        let receipt = audit_values(&p, &s, &pb, &sb).unwrap();
        assert_eq!(receipt["reference_policy"]["contexts"][0]["alpha_allocation"], 0.02);
        assert_eq!(receipt["reference_policy"]["contexts"][1]["alpha_allocation"], 0.02);
        assert_eq!(receipt["reference_policy"]["contexts"][0]["delta_allocation"], 0.005);
        assert_eq!(receipt["reference_policy"]["contexts"][1]["delta_allocation"], 0.005);
    }

    #[test]
    fn dkw_corrected_betting_factor_stays_in_unit_interval() {
        let (p, s, pb, sb) = fixture();
        let receipt = audit_values(&p, &s, &pb, &sb).unwrap();
        for row in receipt["sequential_monitor"]["window_decisions"].as_array().unwrap() {
            if let Some(beta) = row["single_start"]["betting_factor"].as_f64() {
                assert!((0.0..=2.0).contains(&beta));
            }
            if let Some(eta) = row["single_start"]["ons_eta_next"].as_f64() {
                assert!(eta.abs() <= 0.5);
            }
        }
    }

    #[test]
    fn rejects_reference_group_count_different_from_reference_sample_count() {
        let (mut p, s, _, _) = fixture();
        p["contexts"][0]["reference_group_ids"].as_array_mut().unwrap().pop();
        assert!(audit(&p, &s).unwrap_err().contains("one distinct reference_group_id per reference score"));
    }

    #[test]
    fn rejects_reference_evaluation_group_overlap() {
        let (mut p, s, _, _) = fixture();
        p["contexts"][0]["reference_group_ids"][0] = json!("heldout-execution-cctm-restart");
        assert!(audit(&p, &s).unwrap_err().contains("overlaps a null-reference group"));
    }

    #[test]
    fn rejects_reference_group_reuse_across_contexts() {
        let (mut p, s, _, _) = fixture();
        p["contexts"][1]["reference_group_ids"][0] = p["contexts"][0]["reference_group_ids"][0].clone();
        assert!(audit(&p, &s).unwrap_err().contains("reused across context references"));
    }

    #[test]
    fn invalid_and_unseen_context_assignments_abstain_without_wealth_update() {
        let (p, mut s, _, _) = fixture();
        s["records"][0]["sensor_state"] = json!("stale");
        s["records"][1]["context_id"] = json!("new-regime");
        let bytes = include_bytes!("../tests/fixtures/hardsec/context-restarted-v1/runtime-receipt.json");
        let mut assignment_receipt: Value = serde_json::from_slice(bytes).unwrap();
        assignment_receipt["runtime_mapping"]["assignments"][1]["decision_status"] = json!("abstained");
        assignment_receipt["runtime_mapping"]["assignments"][1]["decision_reason"] = json!("unknown_context");
        assignment_receipt["runtime_mapping"]["assignments"][1]["selected_context_id"] = Value::Null;
        assignment_receipt["assignment_integrity"]["accepted_count"] = json!(22);
        assignment_receipt["assignment_integrity"]["abstained_count"] = json!(4);
        assignment_receipt["assignment_integrity"]["accepted_accuracy_denominator"] = json!(22);
        assignment_receipt["assignment_integrity"]["accepted_coverage"] = json!(22.0/26.0);
        assignment_receipt["assignment_integrity"]["end_to_end_correct_context_coverage"] = json!(22.0/26.0);
        let pb = encode(&p);
        let sb = encode(&s);
        let rb = encode(&assignment_receipt);
        let r = audit_values_with_context_receipt(&p, &s, &assignment_receipt, &pb, &sb, &rb).unwrap();
        assert_eq!(r["sequential_monitor"]["valid_observation_count"], 21);
        assert_eq!(r["sequential_monitor"]["abstention_count"], 5);
        assert_eq!(r["sequential_monitor"]["abstention_counts_by_reason"]["invalid_sensor_state"], 3);
        assert_eq!(r["sequential_monitor"]["abstention_counts_by_reason"]["context_assignment_not_accepted"], 2);
        let decisions = r["sequential_monitor"]["window_decisions"].as_array().unwrap();
        assert_eq!(decisions[0]["abstention_reason"], "invalid_sensor_state");
        assert_eq!(decisions[0]["single_start_log_wealth_context"], 0.0);
        assert_eq!(decisions[0]["restart_log_e_value_context"], 0.0);
        assert_eq!(decisions[1]["abstention_reason"], "context_assignment_not_accepted");
        assert_eq!(decisions[1]["single_start_log_wealth_context"], 0.0);
        assert_eq!(decisions[1]["restart_log_e_value_context"], 0.0);
        assert_eq!(decisions[2]["abstention_reason"], Value::Null);
    }

    #[test]
    fn unqualified_pmu_observation_abstains() {
        let (p, mut s, _, _) = fixture();
        s["records"][0]["pmu_qualification_result"] = json!("measurement_quality_gate_failed");
        let r = audit(&p, &s).unwrap();
        assert_eq!(r["sequential_monitor"]["valid_observation_count"], 22);
        assert_eq!(r["sequential_monitor"]["abstention_counts_by_reason"]["pmu_acquisition_not_qualified"], 1);
    }

    #[test]
    fn rejects_duplicate_window_ids() {
        let (p, mut s, _, _) = fixture();
        s["records"][1]["window_id"] = s["records"][0]["window_id"].clone();
        assert!(audit(&p, &s).unwrap_err().contains("duplicate window_id"));
    }

    #[test]
    fn rejects_non_contiguous_sequence_indices() {
        let (p, mut s, _, _) = fixture();
        s["records"][1]["sequence_index"] = json!(9);
        assert!(audit(&p, &s).unwrap_err().contains("must be contiguous from one"));
    }

    #[test]
    fn score_artifact_digest_mismatch_is_rejected() {
        let (p, mut s, _, _) = fixture();
        s["scoring_artifact_sha256"] = json!("d".repeat(64));
        assert!(audit(&p, &s).unwrap_err().contains("scoring_artifact_sha256 does not match"));
    }

    #[test]
    fn rejects_too_few_reference_scores() {
        let (mut p, s, _, _) = fixture();
        p["contexts"][0]["reference_scores"] = json!([0.1,0.2,0.3]);
        p["contexts"][0]["reference_group_ids"] = json!(["a","b","c"]);
        assert!(audit(&p, &s).unwrap_err().contains("reference scores; at least"));
    }

    #[test]
    fn rejects_claims_that_assumptions_are_verified_by_metadata() {
        let (mut p, s, _, _) = fixture();
        p["null_stream_assumption_status"] = json!("verified");
        assert!(audit(&p, &s).unwrap_err().contains("must remain assumption_not_verified"));
    }

    #[test]
    fn perfectly_dependent_marginally_null_streams_expose_iid_assumption_failure() {
        // Deterministic 10,000-point quadrature over one latent X ~ Uniform[0,6].
        // Each stream repeats that same X for 100 consecutive windows: every
        // window is marginally null, but the stream is maximally dependent.
        // The DKW-corrected CCTM rejects far more often than alpha in this
        // deliberately IID-violating negative control. This is not a CI/type-I
        // estimate for hardware; it demonstrates why the paper's IID null
        // assumption must not be silently dropped.
        let n = 600.0_f64;
        let delta = 0.01_f64;
        let alpha = 0.04_f64;
        let k = 1e-6_f64;
        let epsilon = ((2.0 / delta).ln() / (2.0 * n)).sqrt();
        let scale = 1.0 / (0.5 + (1.0 + k * k).sqrt() * epsilon);
        let threshold = (1.0 / alpha).ln();
        let mut alarm_paths = 0usize;

        for grid_index in 0..10_000usize {
            let latent_u = (grid_index as f64 + 0.5) / 10_000.0;
            let latent_score = 6.0 * latent_u;
            let ecdf_count = (latent_score * 100.0).floor().clamp(0.0, 600.0);
            let p_value = ecdf_count / 600.0;

            let mut eta = 0.0_f64;
            let mut accumulator = 1.0_f64;
            let mut log_wealth = 0.0_f64;
            let mut crossed = false;

            for _ in 0..100 {
                let root = (eta * eta + k * k).sqrt();
                let g = scale * (eta * (p_value - 0.5) - root * epsilon);
                let beta = 1.0 + g;
                assert!(beta > 0.0 && beta <= 2.0);
                log_wealth += beta.ln();
                let g_prime = scale * (p_value - 0.5 - (eta / root) * epsilon);
                let z = g_prime / beta;
                accumulator += z * z;
                eta = (eta + 4.0 * z / accumulator).clamp(-0.5, 0.5);
                if log_wealth >= threshold {
                    crossed = true;
                    break;
                }
            }
            if crossed {
                alarm_paths += 1;
            }
        }

        let fraction = alarm_paths as f64 / 10_000.0;
        assert!(fraction > 0.75 && fraction < 0.85, "dependent-null grid alarm fraction {fraction}");
    }


    #[test]
    fn rejects_context_assignment_receipt_with_mismatched_context_mapping() {
        let (p, mut s, _, _) = fixture();
        s["records"][0]["context_id"] = json!("memory");
        let error = audit(&p, &s).unwrap_err();
        assert!(error.contains("accepted context does not match"));
    }

    #[test]
    fn rejects_context_assignment_receipt_with_wrong_policy_artifact() {
        let (p, s, _, _) = fixture();
        let bytes = include_bytes!("../tests/fixtures/hardsec/context-restarted-v1/runtime-receipt.json");
        let mut assignment_receipt: Value = serde_json::from_slice(bytes).unwrap();
        assignment_receipt["assignment_artifact_sha256"] = json!("e".repeat(64));
        let pb = encode(&p);
        let sb = encode(&s);
        let rb = encode(&assignment_receipt);
        assert!(audit_values_with_context_receipt(&p, &s, &assignment_receipt, &pb, &sb, &rb)
            .unwrap_err().contains("assignment_artifact_sha256 does not match"));
    }

    #[test]
    fn abstained_context_assignment_never_updates_wealth() {
        let (p, s, pb, sb) = fixture();
        let bytes = include_bytes!("../tests/fixtures/hardsec/context-restarted-v1/runtime-receipt.json");
        let assignment_receipt: Value = serde_json::from_slice(bytes).unwrap();
        let rb = bytes.to_vec();
        let r = audit_values_with_context_receipt(&p, &s, &assignment_receipt, &pb, &sb, &rb).unwrap();
        let rows = r["sequential_monitor"]["window_decisions"].as_array().unwrap();
        assert_eq!(rows[16]["abstention_reason"], "invalid_sensor_state");
        assert_eq!(rows[17]["abstention_reason"], "context_assignment_not_accepted");
        assert_eq!(rows[18]["abstention_reason"], "invalid_sensor_state");
    }


    #[test]
    fn rejects_context_assignment_receipt_aggregate_count_mismatch() {
        let (p, s, _, _) = fixture();
        let bytes = include_bytes!("../tests/fixtures/hardsec/context-restarted-v1/runtime-receipt.json");
        let mut assignment_receipt: Value = serde_json::from_slice(bytes).unwrap();
        assignment_receipt["assignment_integrity"]["accepted_count"] = json!(15);
        let pb = encode(&p);
        let sb = encode(&s);
        let rb = encode(&assignment_receipt);
        assert!(audit_values_with_context_receipt(&p, &s, &assignment_receipt, &pb, &sb, &rb)
            .unwrap_err().contains("aggregate counts/coverage do not match"));
    }


    #[test]
    fn fully_abstained_assignment_stream_is_valid_and_updates_no_wealth() {
        let (p, s, pb, sb) = fixture();
        let bytes = include_bytes!("../tests/fixtures/hardsec/context-restarted-v1/runtime-receipt.json");
        let mut assignment_receipt: Value = serde_json::from_slice(bytes).unwrap();
        let mapping = assignment_receipt["runtime_mapping"]["assignments"].as_array_mut().unwrap();
        for row in mapping.iter_mut() {
            row["decision_status"] = json!("abstained");
            row["decision_reason"] = json!("ambiguous_assignment");
            row["selected_context_id"] = Value::Null;
        }
        assignment_receipt["assignment_integrity"]["accepted_count"] = json!(0);
        assignment_receipt["assignment_integrity"]["abstained_count"] = json!(26);
        assignment_receipt["assignment_integrity"]["accepted_accuracy"] = Value::Null;
        assignment_receipt["assignment_integrity"]["accepted_accuracy_denominator"] = json!(0);
        assignment_receipt["assignment_integrity"]["accepted_coverage"] = json!(0.0);
        assignment_receipt["assignment_integrity"]["end_to_end_correct_context_coverage"] = json!(0.0);
        let rb = encode(&assignment_receipt);
        let r = audit_values_with_context_receipt(&p, &s, &assignment_receipt, &pb, &sb, &rb).unwrap();
        assert_eq!(r["sequential_monitor"]["valid_observation_count"], 0);
        assert_eq!(r["sequential_monitor"]["abstention_count"], 26);
        assert_eq!(r["result"], "no_restarted_alarm_threshold_crossed");
    }


    #[test]
    fn near_harmonic_restart_weights_have_a_proved_unit_capital_bound() {
        let scale = 1.0 / 3.0;
        let upper = restart_weight_sum_upper_bound(scale);
        assert!(upper < 0.744 && upper < 1.0, "upper bound {upper}");
        let mut partial = 0.0;
        let mut previous = f64::INFINITY;
        for index in 1..=1_000_000usize {
            let weight = restart_weight(index, scale);
            assert!(weight > 0.0 && weight < previous);
            previous = weight;
            partial += weight;
        }
        assert!(partial <= upper + 1e-12);
        assert!(1.0 - partial > 0.25);
    }

    #[test]
    fn restart_lifetime_is_bounded_and_expiry_is_absorbing() {
        let (mut p, s, _, _) = fixture();
        p["maximum_restart_age"] = json!(2);
        let receipt = audit(&p, &s).unwrap();
        for context in receipt["reference_policy"]["contexts"].as_array().unwrap() {
            let restart = &context["restart_policy"];
            assert!(restart["live_restart_state_count"].as_u64().unwrap() <= 2);
            assert!(restart["expired_restart_state_count"].as_u64().unwrap() > 0);
            assert!(restart["unallocated_capital_reserve"].as_f64().unwrap() > 0.25);
        }
        assert_eq!(receipt["sequential_monitor"]["restart_guarantee_contract"]["maximum_restart_age"], 2);
    }

    #[test]
    fn restarted_process_beats_single_start_after_a_late_shift_in_deterministic_stress() {
        let k = 1e-6_f64;
        let epsilon = ((2.0_f64 / 0.005).ln() / (2.0 * 600.0)).sqrt();
        let scale = 1.0 / (0.5 + (1.0 + k * k).sqrt() * epsilon);
        let alpha = 0.02_f64;
        let threshold = (1.0 / alpha).ln();
        let mut context = ContextState {
            reference_id: "reference".to_owned(),
            reference_group_count: 600,
            reference_digest: "a".repeat(64),
            scores: Vec::new(),
            alpha,
            delta: 0.005,
            epsilon,
            scale,
            eta: 0.0,
            ons_accumulator: 1.0,
            log_wealth: Some(0.0),
            first_alarm_sequence_index: None,
            restart_observation_count: 0,
            restart_weight_sum: 0.0,
            restart_states: Vec::new(),
            restart_expired_state_count: 0,
            restart_log_e_value: 0.0,
            restart_first_alarm_sequence_index: None,
        };
        let mut single_eta = 0.0_f64;
        let mut single_accumulator = 1.0_f64;
        let mut single_log_wealth = 0.0_f64;
        let mut rng = 0x1234_5678_u32;

        // Deterministic xorshift null-score sequence: this is a reproducible
        // delay stress fixture, not an iid-validity or PFA simulation.
        for index in 1..=10_000usize {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let p_value = rng as f64 / (u32::MAX as f64 + 1.0);

            let (beta, _, derivative) = cctm_bet(p_value, single_eta, epsilon, k, scale).unwrap();
            if beta > 0.0 {
                single_log_wealth += beta.ln();
                let (eta_next, accumulator_next) =
                    ons_next_eta(single_eta, single_accumulator, derivative, beta).unwrap();
                single_eta = eta_next;
                single_accumulator = accumulator_next;
            }

            update_restarted_processes(
                &mut context, p_value, epsilon, k, scale,
                256, index
            ).unwrap();
        }

        let mut restarted_delay = None;
        let mut single_start_delay = None;
        for delay in 1..=500usize {
            let (beta, _, derivative) = cctm_bet(1.0, single_eta, epsilon, k, scale).unwrap();
            if beta > 0.0 {
                single_log_wealth += beta.ln();
                let (eta_next, accumulator_next) =
                    ons_next_eta(single_eta, single_accumulator, derivative, beta).unwrap();
                single_eta = eta_next;
                single_accumulator = accumulator_next;
            }
            update_restarted_processes(
                &mut context, 1.0, epsilon, k, scale,
                256, 10_000 + delay
            ).unwrap();

            if restarted_delay.is_none() && context.restart_log_e_value >= threshold {
                restarted_delay = Some(delay);
            }
            if single_start_delay.is_none() && single_log_wealth >= threshold {
                single_start_delay = Some(delay);
            }
            if restarted_delay.is_some() && single_start_delay.is_some() {
                break;
            }
        }
        let restarted_delay = restarted_delay.expect("restarted detector should cross in the stress fixture");
        let single_start_delay = single_start_delay.expect("single-start detector should eventually cross");
        assert_eq!(restarted_delay, 50);
        assert_eq!(single_start_delay, 200);
        assert!(restarted_delay < single_start_delay);
    }

    #[test]
    fn restart_processes_do_not_advance_on_context_assignment_abstentions() {
        let (p, s, pb, sb) = fixture();
        let bytes = include_bytes!("../tests/fixtures/hardsec/context-restarted-v1/runtime-receipt.json");
        let assignment_receipt: Value = serde_json::from_slice(bytes).unwrap();
        let rb = bytes.to_vec();
        let receipt = audit_values_with_context_receipt(&p, &s, &assignment_receipt, &pb, &sb, &rb).unwrap();
        let contexts = receipt["reference_policy"]["contexts"].as_array().unwrap();
        let compute = contexts.iter().find(|v| v["context_id"] == "compute").unwrap();
        let memory = contexts.iter().find(|v| v["context_id"] == "memory").unwrap();
        assert_eq!(compute["restart_policy"]["restart_observation_count"], 22);
        assert_eq!(memory["restart_policy"]["restart_observation_count"], 1);
    }

    #[test]
    fn rejects_context_selection_based_on_monitor_score() {
        let (mut p, s, _, _) = fixture();
        p["context_assignment_policy"] = json!("derived_from_anomaly_score");
        assert!(audit(&p, &s).unwrap_err().contains("external_workload_metadata_not_score_derived"));
    }
}
