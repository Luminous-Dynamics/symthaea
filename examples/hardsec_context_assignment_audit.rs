//! HARDSEC-020 workload-context assignment auditor.
//!
//! Audits whether each score stream window was assigned to a declared workload
//! context by an independently pinned, frozen assignment source. The held-out
//! truth oracle is used only for offline evaluation; its labels are never copied
//! into the runtime assignment receipt consumed by HARDSEC-019.
//!
//! Usage:
//!   hardsec_context_assignment_audit --policy policy.json \
//!     --assignments assignments.json --truth truth.json --receipt receipt.json

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const POLICY_SCHEMA: &str = "hardsec-context-assignment-policy-v1";
const ASSIGNMENTS_SCHEMA: &str = "hardsec-context-assignment-stream-v1";
const TRUTH_SCHEMA: &str = "hardsec-context-assignment-truth-v1";
const RECEIPT_SCHEMA: &str = "hardsec-context-assignment-receipt-v1";

#[derive(Debug, Clone)]
struct Policy {
    policy_id: String,
    evidence_domain: String,
    stream_id: String,
    evaluation_group_id: String,
    assignment_mode: String,
    assignment_artifact_sha256: String,
    anomaly_score_artifact_sha256: String,
    allowed_contexts: BTreeSet<String>,
    minimum_top1_confidence: f64,
    minimum_top1_top2_margin: f64,
}

#[derive(Debug, Clone)]
struct CandidateScore {
    context_id: String,
    probability: f64,
}

#[derive(Debug, Clone)]
struct Assignment {
    sequence_index: usize,
    window_id: String,
    candidate_scores: Vec<CandidateScore>,
    top1_context_id: String,
    top1_confidence: f64,
    top1_top2_margin: f64,
    sensor_state: String,
    decision_status: String,
    decision_reason: String,
    selected_context_id: Option<String>,
}

#[derive(Debug, Clone)]
struct TruthRecord {
    sequence_index: usize,
    window_id: String,
    true_context_id: String,
}

#[derive(Debug, Clone)]
struct Args {
    policy: PathBuf,
    assignments: PathBuf,
    truth: PathBuf,
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

fn number_field(value: &Value, key: &str, context: &str) -> Result<f64, String> {
    let result = value.get(key).and_then(Value::as_f64)
        .ok_or_else(|| format!("{context}.{key} must be numeric"))?;
    if !result.is_finite() {
        return Err(format!("{context}.{key} must be finite"));
    }
    Ok(result)
}

fn usize_field(value: &Value, key: &str, context: &str) -> Result<usize, String> {
    let raw = value.get(key).and_then(Value::as_u64)
        .ok_or_else(|| format!("{context}.{key} must be a positive integer"))?;
    let result = usize::try_from(raw).map_err(|_| format!("{context}.{key} is outside supported range"))?;
    if result == 0 {
        return Err(format!("{context}.{key} must be positive"));
    }
    Ok(result)
}

fn valid_sha(value: &str, context: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(format!("{context} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn string_set(value: &Value, key: &str, context: &str) -> Result<BTreeSet<String>, String> {
    let records = value.get(key).and_then(Value::as_array)
        .ok_or_else(|| format!("{context}.{key} must be an array"))?;
    let mut output = BTreeSet::new();
    for (index, item) in records.iter().enumerate() {
        let item = item.as_str().ok_or_else(|| format!("{context}.{key}[{index}] must be a string"))?;
        nonempty(item, &format!("{context}.{key}[{index}]"))?;
        if !output.insert(item.to_owned()) {
            return Err(format!("{context}.{key} contains duplicate value {item}"));
        }
    }
    Ok(output)
}

fn parse_policy(value: &Value) -> Result<Policy, String> {
    if string_field(value, "schema_version", "policy")? != POLICY_SCHEMA {
        return Err("unsupported context policy schema_version".to_owned());
    }
    let policy_id = string_field(value, "policy_id", "policy")?.to_owned();
    let evidence_domain = string_field(value, "evidence_domain", "policy")?.to_owned();
    if !matches!(evidence_domain.as_str(), "synthetic_fixture" | "simulator" | "physical_hardware") {
        return Err("unsupported policy evidence_domain".to_owned());
    }
    let stream_id = string_field(value, "stream_id", "policy")?.to_owned();
    let evaluation_group_id = string_field(value, "evaluation_group_id", "policy")?.to_owned();
    let assignment_mode = string_field(value, "assignment_mode", "policy")?.to_owned();
    if !matches!(assignment_mode.as_str(), "external_metadata" | "frozen_classifier") {
        return Err("assignment_mode must be external_metadata or frozen_classifier".to_owned());
    }
    let assignment_artifact_sha256 = string_field(value, "assignment_artifact_sha256", "policy")?.to_owned();
    valid_sha(&assignment_artifact_sha256, "policy.assignment_artifact_sha256")?;
    let anomaly_score_artifact_sha256 = string_field(value, "anomaly_score_artifact_sha256", "policy")?.to_owned();
    valid_sha(&anomaly_score_artifact_sha256, "policy.anomaly_score_artifact_sha256")?;
    if assignment_artifact_sha256 == anomaly_score_artifact_sha256 {
        return Err("context assignment artifact and anomaly-score artifact must be separately pinned".to_owned());
    }
    if value.get("assignment_not_derived_from_anomaly_score").and_then(Value::as_bool) != Some(true)
        || value.get("assignment_not_derived_from_sequential_wealth").and_then(Value::as_bool) != Some(true) {
        return Err("context assignment must be declared independent of anomaly score and sequential wealth".to_owned());
    }
    if value.get("frozen_before_evaluation").and_then(Value::as_bool) != Some(true) {
        return Err("context assignment policy must be frozen before evaluation".to_owned());
    }
    let allowed_contexts = string_set(value, "allowed_contexts", "policy")?;
    if allowed_contexts.len() < 2 {
        return Err("policy must declare at least two allowed contexts".to_owned());
    }
    let minimum_top1_confidence = number_field(value, "minimum_top1_confidence", "policy")?;
    let minimum_top1_top2_margin = number_field(value, "minimum_top1_top2_margin", "policy")?;
    if !(0.0..=1.0).contains(&minimum_top1_confidence)
        || !(0.0..=1.0).contains(&minimum_top1_top2_margin) {
        return Err("context confidence and margin thresholds must be in [0,1]".to_owned());
    }

    let training = string_set(value, "classifier_training_group_ids", "policy")?;
    let calibration = string_set(value, "classifier_calibration_group_ids", "policy")?;
    if training.contains(&evaluation_group_id) || calibration.contains(&evaluation_group_id) {
        return Err("evaluation group overlaps context assignment training/calibration groups".to_owned());
    }
    if !training.is_disjoint(&calibration) {
        return Err("context assignment training and calibration groups overlap".to_owned());
    }

    Ok(Policy {
        policy_id, evidence_domain, stream_id, evaluation_group_id, assignment_mode,
        assignment_artifact_sha256, anomaly_score_artifact_sha256, allowed_contexts,
        minimum_top1_confidence, minimum_top1_top2_margin
    })
}

fn parse_assignments(value: &Value, policy: &Policy) -> Result<Vec<Assignment>, String> {
    if string_field(value, "schema_version", "assignments")? != ASSIGNMENTS_SCHEMA {
        return Err("unsupported assignments schema_version".to_owned());
    }
    for (field, expected) in [
        ("evidence_domain", policy.evidence_domain.as_str()),
        ("policy_id", policy.policy_id.as_str()),
        ("stream_id", policy.stream_id.as_str()),
        ("evaluation_group_id", policy.evaluation_group_id.as_str()),
        ("assignment_artifact_sha256", policy.assignment_artifact_sha256.as_str()),
    ] {
        if string_field(value, field, "assignments")? != expected {
            return Err(format!("assignments.{field} does not match the policy"));
        }
    }

    let records = value.get("records").and_then(Value::as_array)
        .ok_or("assignments.records must be an array")?;
    if records.is_empty() {
        return Err("assignments.records must not be empty".to_owned());
    }
    let mut seen_windows = BTreeSet::new();
    let mut output = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        let context = format!("assignments.records[{index}]");
        let sequence_index = usize_field(record, "sequence_index", &context)?;
        if sequence_index != index + 1 {
            return Err(format!("{context}.sequence_index must be contiguous from one"));
        }
        let window_id = string_field(record, "window_id", &context)?.to_owned();
        if !seen_windows.insert(window_id.clone()) {
            return Err(format!("duplicate assignment window_id {window_id}"));
        }
        let candidate_values = record.get("candidate_scores").and_then(Value::as_array)
            .ok_or_else(|| format!("{context}.candidate_scores must be an array"))?;
        if candidate_values.len() < 2 {
            return Err(format!("{context} must provide at least two candidate context scores"));
        }
        let mut candidate_scores = Vec::with_capacity(candidate_values.len());
        let mut candidate_ids = BTreeSet::new();
        let mut probability_sum = 0.0;
        for (candidate_index, candidate) in candidate_values.iter().enumerate() {
            let candidate_context = format!("{context}.candidate_scores[{candidate_index}]");
            let context_id = string_field(candidate, "context_id", &candidate_context)?.to_owned();
            let probability = number_field(candidate, "probability", &candidate_context)?;
            if !(0.0..=1.0).contains(&probability) {
                return Err(format!("{candidate_context}.probability must be within [0,1]"));
            }
            if !candidate_ids.insert(context_id.clone()) {
                return Err(format!("{context} repeats candidate context {context_id}"));
            }
            probability_sum += probability;
            candidate_scores.push(CandidateScore { context_id, probability });
        }
        if (probability_sum - 1.0).abs() > 1e-9 {
            return Err(format!("{context}.candidate_scores probabilities must sum to one"));
        }
        candidate_scores.sort_by(|a,b| b.probability.total_cmp(&a.probability));
        let top1_context_id = string_field(record, "top1_context_id", &context)?.to_owned();
        let top1_confidence = number_field(record, "top1_confidence", &context)?;
        let top1_top2_margin = number_field(record, "top1_top2_margin", &context)?;
        let expected_margin = candidate_scores[0].probability - candidate_scores[1].probability;
        if top1_context_id != candidate_scores[0].context_id
            || (top1_confidence - candidate_scores[0].probability).abs() > 1e-9
            || (top1_top2_margin - expected_margin).abs() > 1e-9 {
            return Err(format!("{context} top-1 label/confidence/margin do not match candidate scores"));
        }
        let sensor_state = string_field(record, "sensor_state", &context)?.to_owned();
        if !matches!(sensor_state.as_str(), "valid" | "missing" | "stale" | "misaligned" | "multiplexed" | "untrusted") {
            return Err(format!("{context} has unsupported sensor_state"));
        }
        let decision_status = string_field(record, "decision_status", &context)?.to_owned();
        if !matches!(decision_status.as_str(), "accepted" | "abstained") {
            return Err(format!("{context}.decision_status must be accepted or abstained"));
        }
        let decision_reason = string_field(record, "decision_reason", &context)?.to_owned();
        if !matches!(decision_reason.as_str(), "accepted" | "invalid_sensor" | "unknown_context" | "ambiguous_assignment") {
            return Err(format!("{context}.decision_reason is unsupported"));
        }
        let selected_context_id = match record.get("selected_context_id") {
            Some(Value::String(value)) => Some(value.clone()),
            Some(Value::Null) => None,
            _ => return Err(format!("{context}.selected_context_id must be a string or null")),
        };

        let valid_sensor = sensor_state == "valid";
        let context_known = policy.allowed_contexts.contains(&top1_context_id);
        let confident = top1_confidence >= policy.minimum_top1_confidence
            && top1_top2_margin >= policy.minimum_top1_top2_margin;
        let (expected_status, expected_reason, expected_selected) = if !valid_sensor {
            ("abstained", "invalid_sensor", None)
        } else if !context_known {
            ("abstained", "unknown_context", None)
        } else if !confident {
            ("abstained", "ambiguous_assignment", None)
        } else {
            ("accepted", "accepted", Some(top1_context_id.as_str()))
        };
        if decision_status != expected_status || decision_reason != expected_reason {
            return Err(format!("{context} decision status/reason is inconsistent with sensor, vocabulary, confidence and margin gates"));
        }
        if selected_context_id.as_deref() != expected_selected {
            return Err(format!("{context}.selected_context_id does not match the required fail-closed decision"));
        }

        output.push(Assignment {
            sequence_index, window_id, candidate_scores, top1_context_id,
            top1_confidence, top1_top2_margin, sensor_state, decision_status,
            decision_reason, selected_context_id
        });
    }
    Ok(output)
}

fn parse_truth(value: &Value, policy: &Policy, assignments: &[Assignment]) -> Result<Vec<TruthRecord>, String> {
    if string_field(value, "schema_version", "truth")? != TRUTH_SCHEMA {
        return Err("unsupported truth oracle schema_version".to_owned());
    }
    if string_field(value, "evidence_domain", "truth")? != policy.evidence_domain
        || string_field(value, "stream_id", "truth")? != policy.stream_id
        || string_field(value, "evaluation_group_id", "truth")? != policy.evaluation_group_id {
        return Err("truth oracle identity does not match the held-out assignment stream".to_owned());
    }
    let records = value.get("records").and_then(Value::as_array)
        .ok_or("truth.records must be an array")?;
    if records.len() != assignments.len() {
        return Err("truth oracle must cover every assignment record exactly once".to_owned());
    }
    let mut output = Vec::with_capacity(records.len());
    let mut seen = BTreeSet::new();
    for (index, record) in records.iter().enumerate() {
        let context = format!("truth.records[{index}]");
        let sequence_index = usize_field(record, "sequence_index", &context)?;
        let window_id = string_field(record, "window_id", &context)?.to_owned();
        let true_context_id = string_field(record, "true_context_id", &context)?.to_owned();
        if sequence_index != index + 1 || sequence_index != assignments[index].sequence_index
            || window_id != assignments[index].window_id {
            return Err(format!("{context} does not match the exact assignment sequence/window identity"));
        }
        if !seen.insert(window_id.clone()) {
            return Err(format!("duplicate truth window_id {window_id}"));
        }
        if true_context_id != "unknown" && !policy.allowed_contexts.contains(&true_context_id) {
            return Err(format!("{context}.true_context_id is not a declared context or unknown"));
        }
        output.push(TruthRecord { sequence_index, window_id, true_context_id });
    }
    Ok(output)
}

fn audit_values(
    policy_value: &Value,
    assignment_value: &Value,
    truth_value: &Value,
    policy_bytes: &[u8],
    assignment_bytes: &[u8],
    truth_bytes: &[u8],
) -> Result<Value, String> {
    let policy = parse_policy(policy_value)?;
    let assignments = parse_assignments(assignment_value, &policy)?;
    let truth = parse_truth(truth_value, &policy, &assignments)?;

    let mut accepted = 0usize;
    let mut abstained = 0usize;
    let mut correct = 0usize;
    let mut incorrect = 0usize;
    let mut abstention_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut per_context: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut confusion: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut cctm_mapping = Vec::with_capacity(assignments.len());

    for (assignment, truth_record) in assignments.iter().zip(truth.iter()) {
        let is_accepted = assignment.decision_status == "accepted";
        if is_accepted {
            accepted += 1;
            if assignment.selected_context_id.as_deref() == Some(truth_record.true_context_id.as_str()) {
                correct += 1;
            } else {
                incorrect += 1;
            }
        } else {
            abstained += 1;
            *abstention_counts.entry(assignment.decision_reason.clone()).or_default() += 1;
        }

        let stats = per_context.entry(truth_record.true_context_id.clone()).or_default();
        *stats.entry("truth_count".to_owned()).or_default() += 1;
        if is_accepted {
            if assignment.selected_context_id.as_deref() == Some(truth_record.true_context_id.as_str()) {
                *stats.entry("correct_accepted_count".to_owned()).or_default() += 1;
            } else {
                *stats.entry("incorrect_accepted_count".to_owned()).or_default() += 1;
            }
        } else {
            *stats.entry("abstained_count".to_owned()).or_default() += 1;
        }

        let predicted_key = assignment.selected_context_id.clone().unwrap_or_else(|| "abstained".to_owned());
        *confusion.entry(truth_record.true_context_id.clone()).or_default()
            .entry(predicted_key).or_default() += 1;

        cctm_mapping.push(json!({
            "sequence_index":assignment.sequence_index,
            "window_id":assignment.window_id,
            "decision_status":assignment.decision_status,
            "decision_reason":assignment.decision_reason,
            "selected_context_id":assignment.selected_context_id
        }));
    }

    let total = assignments.len();
    let accepted_accuracy = if accepted == 0 { None } else { Some(correct as f64 / accepted as f64) };
    let accepted_coverage = accepted as f64 / total as f64;
    let end_to_end_correct_context_coverage = correct as f64 / total as f64;

    let per_context_reports: Vec<Value> = per_context.iter().map(|(context, counts)| {
        let truth_count = *counts.get("truth_count").unwrap_or(&0);
        let correct_count = *counts.get("correct_accepted_count").unwrap_or(&0);
        let incorrect_count = *counts.get("incorrect_accepted_count").unwrap_or(&0);
        let abstained_count = *counts.get("abstained_count").unwrap_or(&0);
        json!({
            "true_context_id":context,
            "truth_count":truth_count,
            "correct_accepted_count":correct_count,
            "incorrect_accepted_count":incorrect_count,
            "abstained_count":abstained_count,
            "end_to_end_correct_assignment_recall":if truth_count==0 {Value::Null} else {json!(correct_count as f64 / truth_count as f64)}
        })
    }).collect();

    let worst_declared_context_recall = policy.allowed_contexts.iter().filter_map(|context| {
        per_context.get(context).map(|counts| {
            let denominator = *counts.get("truth_count").unwrap_or(&0);
            let numerator = *counts.get("correct_accepted_count").unwrap_or(&0);
            if denominator == 0 { None } else { Some(numerator as f64 / denominator as f64) }
        }).flatten()
    }).reduce(f64::min);

    Ok(json!({
        "schema_version":RECEIPT_SCHEMA,
        "result":"audited",
        "claim_scope":"declared_context_assignment_gate_and_heldout_label_evaluation_only",
        "policy_id":policy.policy_id,
        "stream_id":policy.stream_id,
        "evaluation_group_id":policy.evaluation_group_id,
        "evidence_domain":policy.evidence_domain,
        "assignment_mode":policy.assignment_mode,
        "assignment_artifact_sha256":policy.assignment_artifact_sha256,
        "anomaly_score_artifact_sha256":policy.anomaly_score_artifact_sha256,
        "inputs":{
            "policy_sha256":sha256(policy_bytes),
            "assignments_sha256":sha256(assignment_bytes),
            "truth_oracle_sha256":sha256(truth_bytes)
        },
        "assignment_integrity":{
            "assignment_record_count":total,
            "accepted_count":accepted,
            "abstained_count":abstained,
            "accepted_coverage":accepted_coverage,
            "accepted_accuracy":accepted_accuracy,
            "accepted_accuracy_denominator":accepted,
            "end_to_end_correct_context_coverage":end_to_end_correct_context_coverage,
            "incorrect_accepted_count":incorrect,
            "abstention_counts_by_reason":abstention_counts,
            "per_true_context":per_context_reports,
            "worst_declared_context_end_to_end_recall":worst_declared_context_recall,
            "confusion_matrix":confusion
        },
        "runtime_mapping":{
            "mapping_schema_version":"hardsec-context-assignment-runtime-map-v1",
            "assignments":cctm_mapping,
            "contains_truth_labels":false,
            "mapping_digest_bound_to_assignment_input":true
        },
        "assumptions":{
            "assignment_not_derived_from_anomaly_score":true,
            "assignment_not_derived_from_sequential_wealth":true,
            "policy_frozen_before_evaluation":true,
            "group_disjointness_checked_by_declared_ids":true,
            "assignment_artifact_provenance":"caller_declared_not_authenticated",
            "context_assignment_exogeneity_status":"not_established_from_input_receipts",
            "truth_oracle_validity_status":"not_established_from_input_receipts"
        },
        "limitations":[
            "The policy, assignment artifact digest, training/calibration group IDs and truth labels are caller-declared and are not authenticated by this auditor.",
            "This audit verifies candidate-score arithmetic and policy gates; it cannot prove that raw metadata or classifier outputs were actually produced by the pinned artifact.",
            "Accepted accuracy is measured only on accepted assignments; coverage and end-to-end correct-context coverage are reported separately.",
            "Context assignment correctness does not prove within-context i.i.d. score behavior or CCTM's DKW assumptions.",
            "The runtime mapping omits truth labels and can be supplied to a sequential monitor; it is not a secure attestation."
        ]
    }))
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
    let mut policy = None;
    let mut assignments = None;
    let mut truth = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let target = match flag.as_str() {
            "--policy" => &mut policy,
            "--assignments" => &mut assignments,
            "--truth" => &mut truth,
            "--receipt" => &mut receipt,
            _ => return Err(format!("unknown argument {flag}")),
        };
        let path = args.next().ok_or_else(|| format!("{flag} requires a path"))?;
        if target.replace(PathBuf::from(path)).is_some() {
            return Err(format!("duplicate argument {flag}"));
        }
    }
    Ok(Args {
        policy:policy.ok_or("missing required argument --policy")?,
        assignments:assignments.ok_or("missing required argument --assignments")?,
        truth:truth.ok_or("missing required argument --truth")?,
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
    let (policy_bytes, policy) = read_input(&args.policy, "policy")?;
    let (assignment_bytes, assignments) = read_input(&args.assignments, "assignments")?;
    let (truth_bytes, truth) = read_input(&args.truth, "truth oracle")?;
    validate_receipt_location(&args.receipt, &[
        args.policy.as_path(), args.assignments.as_path(), args.truth.as_path()
    ])?;
    let receipt = audit_values(
        &policy, &assignments, &truth,
        &policy_bytes, &assignment_bytes, &truth_bytes
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
            eprintln!("HARDSEC-020 context assignment audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Value, Value, Value, Vec<u8>, Vec<u8>, Vec<u8>) {
        let policy_bytes = include_bytes!("../tests/fixtures/hardsec/context-assignment-v1/policy.json").to_vec();
        let assignments_bytes = include_bytes!("../tests/fixtures/hardsec/context-assignment-v1/assignments.json").to_vec();
        let truth_bytes = include_bytes!("../tests/fixtures/hardsec/context-assignment-v1/truth-oracle.json").to_vec();
        let policy = serde_json::from_slice(&policy_bytes).unwrap();
        let assignments = serde_json::from_slice(&assignments_bytes).unwrap();
        let truth = serde_json::from_slice(&truth_bytes).unwrap();
        (policy, assignments, truth, policy_bytes, assignments_bytes, truth_bytes)
    }

    fn encode(value: &Value) -> Vec<u8> {
        let mut bytes = serde_json::to_vec_pretty(value).unwrap();
        bytes.push(b'\n');
        bytes
    }

    fn audit(policy: &Value, assignments: &Value, truth: &Value) -> Result<Value, String> {
        let pb = encode(policy); let ab = encode(assignments); let tb = encode(truth);
        audit_values(policy, assignments, truth, &pb, &ab, &tb)
    }

    #[test]
    fn known_answer_reports_coverage_accuracy_and_truth_free_runtime_map() {
        let (p,a,t,pb,ab,tb)=fixture();
        let r=audit_values(&p,&a,&t,&pb,&ab,&tb).unwrap();
        assert_eq!(r["result"],"audited");
        assert_eq!(r["assignment_integrity"]["assignment_record_count"],19);
        assert_eq!(r["assignment_integrity"]["accepted_count"],16);
        assert_eq!(r["assignment_integrity"]["abstained_count"],3);
        assert_eq!(r["assignment_integrity"]["accepted_accuracy"],1.0);
        assert_eq!(r["assignment_integrity"]["accepted_accuracy_denominator"],16);
        assert!((r["assignment_integrity"]["accepted_coverage"].as_f64().unwrap() - 16.0/19.0).abs() < 1e-12);
        assert!((r["assignment_integrity"]["end_to_end_correct_context_coverage"].as_f64().unwrap() - 16.0/19.0).abs() < 1e-12);
        assert_eq!(r["runtime_mapping"]["contains_truth_labels"],false);
        assert_eq!(r["runtime_mapping"]["assignments"].as_array().unwrap().len(),19);
        assert_eq!(r["assumptions"]["context_assignment_exogeneity_status"],"not_established_from_input_receipts");
    }

    #[test]
    fn rejects_context_assignment_derived_from_anomaly_score() {
        let (mut p,a,t,_,_,_)=fixture();
        p["assignment_not_derived_from_anomaly_score"]=json!(false);
        assert!(audit(&p,&a,&t).unwrap_err().contains("independent of anomaly score"));
    }

    #[test]
    fn rejects_reuse_of_anomaly_score_artifact_as_context_artifact() {
        let (mut p,a,t,_,_,_)=fixture();
        p["assignment_artifact_sha256"]=p["anomaly_score_artifact_sha256"].clone();
        assert!(audit(&p,&a,&t).unwrap_err().contains("separately pinned"));
    }

    #[test]
    fn rejects_training_evaluation_group_overlap() {
        let (mut p,a,t,_,_,_)=fixture();
        p["classifier_training_group_ids"][0]=p["evaluation_group_id"].clone();
        assert!(audit(&p,&a,&t).unwrap_err().contains("evaluation group overlaps"));
    }

    #[test]
    fn rejects_training_calibration_group_overlap() {
        let (mut p,a,t,_,_,_)=fixture();
        p["classifier_calibration_group_ids"][0]=p["classifier_training_group_ids"][0].clone();
        assert!(audit(&p,&a,&t).unwrap_err().contains("training and calibration groups overlap"));
    }

    #[test]
    fn rejects_probability_sum_not_equal_to_one() {
        let (p,mut a,t,_,_,_)=fixture();
        a["records"][0]["candidate_scores"][0]["probability"]=json!(0.5);
        assert!(audit(&p,&a,&t).unwrap_err().contains("must sum to one"));
    }

    #[test]
    fn rejects_candidate_label_confidence_or_margin_mismatch() {
        let (p,mut a,t,_,_,_)=fixture();
        a["records"][0]["top1_confidence"]=json!(0.5);
        assert!(audit(&p,&a,&t).unwrap_err().contains("do not match candidate scores"));
    }

    #[test]
    fn rejects_low_confidence_record_marked_accepted() {
        let (p,mut a,t,_,_,_)=fixture();
        a["records"][0]["candidate_scores"]=json!([
            {"context_id":"compute","probability":0.6},
            {"context_id":"memory","probability":0.4}
        ]);
        a["records"][0]["top1_confidence"]=json!(0.6);
        a["records"][0]["top1_top2_margin"]=json!(0.2);
        assert!(audit(&p,&a,&t).unwrap_err().contains("decision status/reason"));
    }

    #[test]
    fn invalid_sensor_must_abstain() {
        let (p,mut a,t,_,_,_)=fixture();
        a["records"][16]["decision_status"]=json!("accepted");
        a["records"][16]["decision_reason"]=json!("accepted");
        a["records"][16]["selected_context_id"]=json!("compute");
        assert!(audit(&p,&a,&t).unwrap_err().contains("decision status/reason"));
    }

    #[test]
    fn rejects_duplicate_window_identity() {
        let (p,mut a,t,_,_,_)=fixture();
        a["records"][1]["window_id"]=a["records"][0]["window_id"].clone();
        assert!(audit(&p,&a,&t).unwrap_err().contains("duplicate assignment window_id"));
    }

    #[test]
    fn rejects_missing_truth_oracle_record() {
        let (p,a,mut t,_,_,_)=fixture();
        t["records"].as_array_mut().unwrap().pop();
        assert!(audit(&p,&a,&t).unwrap_err().contains("cover every assignment record"));
    }

    #[test]
    fn rejects_truth_window_order_mismatch() {
        let (p,a,mut t,_,_,_)=fixture();
        t["records"][0]["window_id"]=json!("wrong-window");
        assert!(audit(&p,&a,&t).unwrap_err().contains("does not match the exact assignment"));
    }

    #[test]
    fn rejects_duplicate_candidate_context() {
        let (p,mut a,t,_,_,_)=fixture();
        a["records"][0]["candidate_scores"][1]["context_id"]=json!("compute");
        assert!(audit(&p,&a,&t).unwrap_err().contains("repeats candidate context"));
    }
}
