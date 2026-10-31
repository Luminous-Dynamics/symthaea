//! HARDSEC-021 randomized-PIT DKW-corrected Conditional Conformal Test Martingale.
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
//!   hardsec_context_cctm_randomized_pit --plan plan.json \
//!     --stream stream.json --receipt receipt.json

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PLAN_SCHEMA: &str = "hardsec-context-cctm-rpit-plan-v1";
const STREAM_SCHEMA: &str = "hardsec-context-cctm-rpit-stream-v1";
const RECEIPT_SCHEMA: &str = "hardsec-context-cctm-rpit-receipt-v1";
const QUALIFIED_PMU: &str = "qualified_for_named_acquisition_conditions";
const MAX_ETA: f64 = 0.5;

#[derive(Debug, Clone)]
struct ContextState {
    reference_id: String,
    reference_group_count: usize,
    reference_digest: String,
    reference_duplicate_score_group_count: usize,
    reference_duplicate_score_extra_occurrences: usize,
    scores: Vec<f64>,
    alpha: f64,
    delta: f64,
    epsilon: f64,
    scale: f64,
    eta: f64,
    ons_accumulator: f64,
    log_wealth: Option<f64>,
    first_alarm_sequence_index: Option<usize>,
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
    pit_randomizer_source_sha256: String,
    alpha_budget: f64,
    delta_budget: f64,
    smoothing_k: f64,
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
    randomizer_id: String,
    pit_randomizer_uniform: f64,
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
    let pit_randomizer_source_sha256 = string_field(value, "pit_randomizer_source_sha256", "plan")?.to_owned();
    valid_sha(&pit_randomizer_source_sha256, "plan.pit_randomizer_source_sha256")?;
    if string_field(value, "pit_randomization_policy", "plan")? != "independent_uniform_after_score_acquisition_v1" {
        return Err("unsupported PIT randomization policy".to_owned());
    }
    if string_field(value, "pit_randomizer_independence_status", "plan")? != "assumption_not_verified" {
        return Err("PIT randomizer independence must remain assumption_not_verified".to_owned());
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
        let mut duplicate_score_group_count = 0usize;
        let mut duplicate_score_extra_occurrences = 0usize;
        let mut position = 0usize;
        while position < scores.len() {
            let mut next = position + 1;
            while next < scores.len() && scores[next] == scores[position] {
                next += 1;
            }
            if next - position > 1 {
                duplicate_score_group_count += 1;
                duplicate_score_extra_occurrences += next - position - 1;
            }
            position = next;
        }
        let encoded = serde_json::to_vec(&scores).map_err(|e| format!("cannot encode reference scores: {e}"))?;
        let state = ContextState {
            reference_id,
            reference_group_count: reference_group_ids.len(),
            reference_digest: sha256(&encoded),
            reference_duplicate_score_group_count: duplicate_score_group_count,
            reference_duplicate_score_extra_occurrences: duplicate_score_extra_occurrences,
            scores,
            alpha: alpha_per_context,
            delta: delta_per_context,
            epsilon,
            scale,
            eta: 0.0,
            ons_accumulator: 1.0,
            log_wealth: Some(0.0),
            first_alarm_sequence_index: None,
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
        pit_randomizer_source_sha256,
        alpha_budget,
        delta_budget,
        smoothing_k,
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
    if string_field(value, "pit_randomizer_source_sha256", "stream")? != plan.pit_randomizer_source_sha256 {
        return Err("stream PIT randomizer source digest does not match the plan".to_owned());
    }

    let records = value.get("records").and_then(Value::as_array)
        .ok_or("stream.records must be an array")?;
    if records.is_empty() {
        return Err("stream.records must not be empty".to_owned());
    }
    let mut window_ids = BTreeSet::new();
    let mut randomizer_ids = BTreeSet::new();
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
        let randomizer_id = string_field(item, "randomizer_id", &context)?.to_owned();
        if !randomizer_ids.insert(randomizer_id.clone()) {
            return Err(format!("duplicate PIT randomizer_id {randomizer_id}"));
        }
        let pit_randomizer_uniform = f64_field(item, "pit_randomizer_uniform", &context)?;
        if !(0.0..1.0).contains(&pit_randomizer_uniform) {
            return Err(format!("{context}.pit_randomizer_uniform must be in [0,1)"));
        }
        output.push(StreamRecord {
            sequence_index,
            window_id,
            context_id,
            anomaly_score,
            sensor_state,
            pmu_qualification_result,
            randomizer_id,
            pit_randomizer_uniform,
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
    if count_field(integrity, "assignment_record_count", "context_receipt.assignment_integrity")? != assignments.len()
        || count_field(integrity, "accepted_count", "context_receipt.assignment_integrity")? != accepted_count
        || count_field(integrity, "abstained_count", "context_receipt.assignment_integrity")? != abstained_count {
        return Err("context assignment receipt aggregate counts do not match its runtime mapping".to_owned());
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
// Randomized empirical PIT for arbitrary score distributions, including atoms.
// For a sorted reference set R, let L=#(r<x), H=#(r<=x). Then
// Fhat(x-)+U*(Fhat(x)-Fhat(x-)) = (L+U*(H-L))/n.
// If U is independent Uniform(0,1), the population randomized PIT is uniform
// even for discrete P. A DKW band on F also bounds its left limits; convex
// interpolation retains the same epsilon. Independence of U is an assumption,
// not something the input receipt can authenticate.
fn randomized_empirical_pit(reference_scores: &[f64], score: f64, uniform: f64) -> Result<(f64, usize, usize), String> {
    if !(0.0..1.0).contains(&uniform) {
        return Err("randomized PIT input U must be in [0,1)".to_owned());
    }
    let lower_count = reference_scores.partition_point(|reference| *reference < score);
    let upper_count = reference_scores.partition_point(|reference| *reference <= score);
    let equal_count = upper_count - lower_count;
    let p_value = (lower_count as f64 + uniform * equal_count as f64) / reference_scores.len() as f64;
    if !p_value.is_finite() || !(0.0..=1.0).contains(&p_value) {
        return Err("randomized PIT value outside [0,1]".to_owned());
    }
    Ok((p_value, lower_count, equal_count))
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
    let mut first_alarm_sequence_index: Option<usize> = None;
    let mut first_alarm_context_id: Option<String> = None;
    let mut output_records = Vec::with_capacity(records.len());
    let mut abstention_counts: BTreeMap<String, usize> = BTreeMap::new();

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
                "action":if first_alarm_sequence_index.is_some() {"alert_latched_observation_abstained"} else {"abstain"},
                "abstention_reason":reason,
                "randomized_pit_value":null,
                "betting_factor":null,
                "log_wealth_context":context_log_wealth,
                "threshold_context":plan.contexts.get(&record.context_id).map(|s| -s.alpha.ln())
            }));
            continue;
        }

        let state = plan.contexts.get_mut(&record.context_id).ok_or("context state invariant failed")?;
        let (p_value, reference_lower_count, reference_equal_count) =
            randomized_empirical_pit(&state.scores, record.anomaly_score, record.pit_randomizer_uniform)?;
        let eta_used = state.eta;
        let (beta, g, g_prime) = cctm_bet(p_value, eta_used, state.epsilon, plan.smoothing_k, state.scale)?;
        let (eta_next, next_accumulator) = ons_next_eta(eta_used, state.ons_accumulator, g_prime, beta)?;
        state.log_wealth = match (state.log_wealth, beta) {
            (Some(_), 0.0) => None, // exact zero wealth is represented as JSON null, never non-finite JSON.
            (Some(previous), factor) => Some(previous + factor.ln()),
            (None, _) => None, // zero wealth cannot recover under multiplicative betting.
        };
        state.eta = eta_next;
        state.ons_accumulator = next_accumulator;
        let threshold = -state.alpha.ln();

        let crossing_now = state.first_alarm_sequence_index.is_none()
            && state.log_wealth.is_some_and(|wealth| wealth >= threshold);
        if crossing_now {
            state.first_alarm_sequence_index = Some(record.sequence_index);
            if first_alarm_sequence_index.is_none() {
                first_alarm_sequence_index = Some(record.sequence_index);
                first_alarm_context_id = Some(record.context_id.clone());
            }
        }
        let action = if crossing_now {
            "alarm_crossing"
        } else if first_alarm_sequence_index.is_some() {
            "alert_latched"
        } else {
            "monitoring_no_alarm_yet"
        };
        output_records.push(json!({
            "sequence_index":record.sequence_index,
            "window_id":record.window_id,
            "context_id":record.context_id,
            "context_assignment_status":assignment.decision_status,
            "context_assignment_reason":assignment.decision_reason,
            "reference_id":state.reference_id,
            "randomized_pit_value":p_value,
            "randomized_pit_method":"Fhat(x_minus)+U*(Fhat(x)-Fhat(x_minus))",
            "randomizer_id":record.randomizer_id,
            "pit_randomizer_source_sha256":plan.pit_randomizer_source_sha256,
            "reference_lower_count":reference_lower_count,
            "reference_equal_count":reference_equal_count,
            "epsilon_dkw":state.epsilon,
            "ons_eta_used":eta_used,
            "ons_eta_next":eta_next,
            "ons_accumulator":next_accumulator,
            "cctm_g":g,
            "betting_factor":beta,
            "log_wealth_context":state.log_wealth,
            "threshold_context":threshold,
            "action":action,
            "abstention_reason":null
        }));
    }

    let mut test_score_values: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for record in &records {
        test_score_values.entry(record.context_id.clone()).or_default().push(record.anomaly_score);
    }
    let test_score_tie_diagnostics: Vec<Value> = test_score_values.iter().map(|(context_id, values)| {
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        let mut groups = 0usize;
        let mut extra = 0usize;
        let mut position = 0usize;
        while position < sorted.len() {
            let mut next = position + 1;
            while next < sorted.len() && sorted[next] == sorted[position] {
                next += 1;
            }
            if next - position > 1 {
                groups += 1;
                extra += next - position - 1;
            }
            position = next;
        }
        json!({
            "context_id":context_id,
            "stream_score_count":sorted.len(),
            "test_duplicate_score_group_count":groups,
            "test_duplicate_score_extra_occurrences":extra
        })
    }).collect();

    let context_reports: Vec<Value> = plan.contexts.iter().map(|(context_id, state)| json!({
        "context_id":context_id,
        "reference_id":state.reference_id,
        "reference_digest":state.reference_digest,
        "reference_score_count":state.scores.len(),
        "reference_duplicate_score_group_count":state.reference_duplicate_score_group_count,
        "reference_duplicate_score_extra_occurrences":state.reference_duplicate_score_extra_occurrences,
        "declared_reference_independent_group_count":state.reference_group_count,
        "alpha_allocation":state.alpha,
        "delta_allocation":state.delta,
        "dkw_epsilon":state.epsilon,
        "betting_scale_C":state.scale,
        "final_log_wealth":state.log_wealth,
        "ons_eta_final":state.eta,
        "ons_accumulator_final":state.ons_accumulator,
        "first_alarm_sequence_index":state.first_alarm_sequence_index,
        "threshold_log_wealth":-state.alpha.ln()
    })).collect();

    let result = if first_alarm_sequence_index.is_some() {
        "alarm_threshold_crossed"
    } else {
        "no_alarm_threshold_crossed"
    };
    let valid_count = records.len() - abstention_counts.values().sum::<usize>();
    let context_count = plan.contexts.len();

    Ok(json!({
        "schema_version":RECEIPT_SCHEMA,
        "result":result,
        "claim_scope":"published_dkw_corrected_cctm_mechanics_with_unverified_iid_assumptions",
        "monitor_id":plan.monitor_id,
        "stream_id":plan.stream_id,
        "evidence_domain":plan.evidence_domain,
        "evaluation_group_id":plan.evaluation_group_id,
        "scoring_artifact_sha256":plan.scoring_artifact_sha256,
        "pit_randomizer_source_sha256":plan.pit_randomizer_source_sha256,
        "pit_randomization_policy":"independent_uniform_after_score_acquisition_v1",
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
            "overall_first_alarm_sequence_index":first_alarm_sequence_index,
            "overall_first_alarm_context_id":first_alarm_context_id,
            "valid_observation_count":valid_count,
            "abstention_count":abstention_counts.values().sum::<usize>(),
            "abstention_counts_by_reason":abstention_counts,
            "alarm_is_latched":first_alarm_sequence_index.is_some(),
            "post_alarm_reset_count":0,
            "window_decisions":output_records
        },
        "statistical_guarantees":{
            "method":"conditional_conformal_test_martingale_with_dkw_randomized_pit",
            "randomized_pit_atom_safe":true,
            "randomizer_source_artifact_sha256":plan.pit_randomizer_source_sha256,
            "randomizer_independence_status":"assumption_not_verified",
            "reference_and_test_score_tie_diagnostics":{
                "reference":context_reports.iter().map(|r| json!({
                    "context_id":r["context_id"],
                    "duplicate_score_group_count":r["reference_duplicate_score_group_count"],
                    "duplicate_score_extra_occurrences":r["reference_duplicate_score_extra_occurrences"]
                })).collect::<Vec<_>>(),
                "test_stream":test_score_tie_diagnostics
            },
            "continuous_score_distribution_required":false,
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
            "reference_confidence_bound":"DKW epsilon = sqrt(ln(2/delta_context)/(2*n_context)); left limits share the same sup-norm bound",
            "randomized_pit":"Fhat_0(x_minus)+U*(Fhat_0(x)-Fhat_0(x_minus))",
            "betting_function":"1 + C*(eta*(randomized_pit-0.5) - sqrt(eta^2+k^2)*epsilon)",
            "ons_update":"z=g_prime/b; a_next=a+z^2; eta_next=clip(eta+4*z/a_next,-0.5,0.5)",
            "context_allocation":"Bonferroni alpha_context=alpha_total/K and delta_context=delta_total/K",
            "scope":"union-bound control across the predeclared finite context set, conditional on all DKW reference bands and the within-context i.i.d./external-context assumptions"
        },
        "limitations":[
            "The published CCTM theorem assumes i.i.d. null test scores and i.i.d. calibration scores within each context; it does not provide a guarantee for arbitrary autocorrelated or adversarially dependent PMU windows.",
            "Unique reference group IDs and input digests are caller-declared; the receipt cannot authenticate group identity, independence or the actual scoring implementation.",
            "Context assignment must be external to, and not selected from, the same anomaly score being tested; this assumption is not verified.",
            "The randomized PIT is valid only when each U_t is conditionally independent Uniform(0,1) after score acquisition and independent of the entire score/history/context process; the source digest does not prove this.",
            "The randomized-PIT interpolation is a derived extension of the paper CCTM, not a verbatim implementation of the authors' ECDF software; its proof audit and reference implementation are separate deliverables.",
            "The monitor is not yet an independent reproduction of the authors' experiment suite or a comparison against their Python package.",
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
    let bytes = include_bytes!("../tests/fixtures/hardsec/context-assignment-v1/runtime-receipt.json");
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
        let plan_bytes = include_bytes!("../tests/fixtures/hardsec/context-cctm-rpit-v1/plan.json").to_vec();
        let stream_bytes = include_bytes!("../tests/fixtures/hardsec/context-cctm-rpit-v1/stream.json").to_vec();
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
        assert_eq!(receipt["result"], "alarm_threshold_crossed");
        assert_eq!(receipt["sequential_monitor"]["overall_first_alarm_sequence_index"], 16);
        assert_eq!(receipt["sequential_monitor"]["overall_first_alarm_context_id"], "compute");
        assert_eq!(receipt["sequential_monitor"]["valid_observation_count"], 16);
        assert_eq!(receipt["sequential_monitor"]["abstention_count"], 3);
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
            if let Some(beta) = row["betting_factor"].as_f64() {
                assert!((0.0..=2.0).contains(&beta));
            }
            if let Some(eta) = row["ons_eta_next"].as_f64() {
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
        p["contexts"][0]["reference_group_ids"][0] = json!("heldout-execution-cctm-01");
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
        let (p, mut s, pb, sb) = fixture();
        s["records"][0]["sensor_state"] = json!("stale");
        s["records"][1]["context_id"] = json!("new-regime");
        let bytes = include_bytes!("../tests/fixtures/hardsec/context-assignment-v1/runtime-receipt.json");
        let mut assignment_receipt: Value = serde_json::from_slice(bytes).unwrap();
        assignment_receipt["runtime_mapping"]["assignments"][1]["decision_status"] = json!("abstained");
        assignment_receipt["runtime_mapping"]["assignments"][1]["decision_reason"] = json!("unknown_context");
        assignment_receipt["runtime_mapping"]["assignments"][1]["selected_context_id"] = Value::Null;
        let rb = encode(&assignment_receipt);
        let r = audit_values_with_context_receipt(&p, &s, &assignment_receipt, &pb, &sb, &rb).unwrap();
        assert_eq!(r["sequential_monitor"]["valid_observation_count"], 0);
        assert_eq!(r["sequential_monitor"]["abstention_count"], 2);
        assert_eq!(r["sequential_monitor"]["abstention_counts_by_reason"]["invalid_sensor_state"], 1);
        assert_eq!(r["sequential_monitor"]["abstention_counts_by_reason"]["context_assignment_not_accepted"], 1);
        assert_eq!(r["reference_policy"]["contexts"][0]["final_log_wealth"], 0.0);
    }

    #[test]
    fn unqualified_pmu_observation_abstains() {
        let (p, mut s, _, _) = fixture();
        s["records"][0]["pmu_qualification_result"] = json!("measurement_quality_gate_failed");
        let r = audit(&p, &s).unwrap();
        assert_eq!(r["sequential_monitor"]["valid_observation_count"], 15);
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
        let bytes = include_bytes!("../tests/fixtures/hardsec/context-assignment-v1/runtime-receipt.json");
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
        let bytes = include_bytes!("../tests/fixtures/hardsec/context-assignment-v1/runtime-receipt.json");
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
        let bytes = include_bytes!("../tests/fixtures/hardsec/context-assignment-v1/runtime-receipt.json");
        let mut assignment_receipt: Value = serde_json::from_slice(bytes).unwrap();
        assignment_receipt["assignment_integrity"]["accepted_count"] = json!(15);
        let pb = encode(&p);
        let sb = encode(&s);
        let rb = encode(&assignment_receipt);
        assert!(audit_values_with_context_receipt(&p, &s, &assignment_receipt, &pb, &sb, &rb)
            .unwrap_err().contains("aggregate counts do not match"));
    }


    #[test]
    fn fully_abstained_assignment_stream_is_valid_and_updates_no_wealth() {
        let (p, s, pb, sb) = fixture();
        let bytes = include_bytes!("../tests/fixtures/hardsec/context-assignment-v1/runtime-receipt.json");
        let mut assignment_receipt: Value = serde_json::from_slice(bytes).unwrap();
        let mapping = assignment_receipt["runtime_mapping"]["assignments"].as_array_mut().unwrap();
        for row in mapping.iter_mut() {
            row["decision_status"] = json!("abstained");
            row["decision_reason"] = json!("ambiguous_assignment");
            row["selected_context_id"] = Value::Null;
        }
        assignment_receipt["assignment_integrity"]["accepted_count"] = json!(0);
        assignment_receipt["assignment_integrity"]["abstained_count"] = json!(19);
        assignment_receipt["assignment_integrity"]["accepted_accuracy"] = Value::Null;
        assignment_receipt["assignment_integrity"]["accepted_accuracy_denominator"] = json!(0);
        let rb = encode(&assignment_receipt);
        let r = audit_values_with_context_receipt(&p, &s, &assignment_receipt, &pb, &sb, &rb).unwrap();
        assert_eq!(r["sequential_monitor"]["valid_observation_count"], 0);
        assert_eq!(r["sequential_monitor"]["abstention_count"], 19);
        assert_eq!(r["result"], "no_alarm_threshold_crossed");
    }


    #[test]
    fn randomized_pit_interpolates_within_reference_atoms() {
        let reference = vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let (middle_of_one, lower1, mass1) = randomized_empirical_pit(&reference, 1.0, 0.5).unwrap();
        assert_eq!(lower1, 0);
        assert_eq!(mass1, 3);
        assert!((middle_of_one - 0.25).abs() < 1e-12);
        let (inside_two, lower2, mass2) = randomized_empirical_pit(&reference, 2.0, 0.25).unwrap();
        assert_eq!(lower2, 3);
        assert_eq!(mass2, 2);
        assert!((inside_two - (3.5 / 6.0)).abs() < 1e-12);
    }

    #[test]
    fn randomized_pit_endpoints_span_left_and_right_cdf_limits() {
        let reference = vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let (left, _, _) = randomized_empirical_pit(&reference, 1.0, 0.0).unwrap();
        let (right, _, _) = randomized_empirical_pit(&reference, 1.0, 0.999999999).unwrap();
        assert_eq!(left, 0.0);
        assert!(right > 0.499999 && right < 0.5);
    }

    #[test]
    fn randomized_pit_rejects_out_of_range_randomizer() {
        let reference = vec![1.0, 1.0, 2.0];
        assert!(randomized_empirical_pit(&reference, 1.0, 1.0).is_err());
        assert!(randomized_empirical_pit(&reference, 1.0, -0.1).is_err());
    }

    #[test]
    fn reference_atom_diagnostics_are_reported() {
        let (mut p, s, _, _) = fixture();
        p["contexts"][0]["reference_scores"] = json!([
            1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
            2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0
        ]);
        p["contexts"][0]["reference_group_ids"] = json!([
            "g1","g2","g3","g4","g5","g6","g7","g8","g9","g10",
            "g11","g12","g13","g14","g15","g16","g17","g18","g19","g20"
        ]);
        p["minimum_reference_scores"] = json!(2);
        let r = audit(&p, &s).unwrap();
        assert_eq!(r["reference_policy"]["contexts"][0]["reference_duplicate_score_group_count"], 2);
        assert_eq!(r["reference_policy"]["contexts"][0]["reference_duplicate_score_extra_occurrences"], 18);
    }

    #[test]
    fn rejects_randomizer_source_digest_mismatch() {
        let (p, mut s, _, _) = fixture();
        s["pit_randomizer_source_sha256"] = json!("f".repeat(64));
        assert!(audit(&p, &s).unwrap_err().contains("PIT randomizer source digest does not match"));
    }

    #[test]
    fn rejects_duplicate_randomizer_id() {
        let (p, mut s, _, _) = fixture();
        s["records"][1]["randomizer_id"] = s["records"][0]["randomizer_id"].clone();
        assert!(audit(&p, &s).unwrap_err().contains("duplicate PIT randomizer_id"));
    }

    #[test]
    fn randomizer_independence_remains_an_unverified_assumption() {
        let (p, s, _, _) = fixture();
        let r = audit(&p, &s).unwrap();
        assert_eq!(r["statistical_guarantees"]["randomizer_independence_status"], "assumption_not_verified");
        assert_eq!(r["statistical_guarantees"]["randomized_pit_atom_safe"], true);
    }


    #[test]
    fn exact_discrete_pit_control_spreads_atoms_uniformly() {
        // Discrete null with masses .5, .3, .2. The randomized PIT fills
        // [0,.5], [.5,.8], [.8,1] in proportion to those atom masses.
        // Stratified deterministic U grids verify the probability-integral
        // transform algebra without pretending to sample a CSPRNG.
        let reference = vec![
            1.0,1.0,1.0,1.0,1.0,
            2.0,2.0,2.0,
            3.0,3.0
        ];
        let mut randomized = Vec::new();
        let atom_groups = [(1.0, 5usize), (2.0, 3usize), (3.0, 2usize)];
        let grid_units_per_probability_mass = 1_000usize;
        let mut naive = Vec::new();
        for (score, mass) in atom_groups {
            let lower = reference.iter().filter(|v| **v < score).count() as f64 / reference.len() as f64;
            let atom_mass = mass as f64 / reference.len() as f64;
            let upper = lower + atom_mass;
            naive.push(upper);
            let grid_per_atom = mass * grid_units_per_probability_mass;
            for index in 0..grid_per_atom {
                let uniform = (index as f64 + 0.5) / grid_per_atom as f64;
                let (value, _, observed_mass) = randomized_empirical_pit(&reference, score, uniform).unwrap();
                assert_eq!(observed_mass, mass);
                randomized.push(value);
            }
        }
        randomized.sort_by(f64::total_cmp);
        let n = randomized.len() as f64;
        let mut ks = 0.0_f64;
        for (index, value) in randomized.iter().enumerate() {
            let expected = (index as f64 + 0.5) / n;
            ks = ks.max((value - expected).abs());
        }
        assert!(ks <= 0.000051, "randomized PIT stratified-grid KS error {ks}");
        assert_eq!(naive, vec![0.5,0.8,1.0]);
        assert!(naive[0] < naive[1] && naive[1] < naive[2]);
    }

    #[test]
    fn dkw_endpoint_interpolation_never_exceeds_same_sup_norm_error() {
        // Endpoint errors bounded by epsilon imply every convex interpolation
        // of the left/right CDF endpoints has error bounded by epsilon.
        let epsilon = 0.07_f64;
        let reference_left = 0.31_f64;
        let reference_right = 0.59_f64;
        let true_left = reference_left - epsilon * 0.9;
        let true_right = reference_right + epsilon * 0.8;
        for index in 0..=1000 {
            let u = index as f64 / 1000.0;
            let empirical = reference_left + u * (reference_right - reference_left);
            let truth = true_left + u * (true_right - true_left);
            assert!((empirical - truth).abs() <= epsilon + 1e-12);
        }
    }

    #[test]
    fn rejects_context_selection_based_on_monitor_score() {
        let (mut p, s, _, _) = fixture();
        p["context_assignment_policy"] = json!("derived_from_anomaly_score");
        assert!(audit(&p, &s).unwrap_err().contains("external_workload_metadata_not_score_derived"));
    }
}
