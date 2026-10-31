//! HARDSEC-021 fixed-length episode packer and weighted-restarted conformal detector.
//!
//! Statistical unit: one fixed-length episode from one caller-declared execution
//! group. Correlated windows inside an episode are collapsed to a fixed max score
//! before sequential testing. The monitor consumes the truth-free HARDSEC-020
//! window-context receipt and fails closed on mixed/abstained context assignments.
//!
//! Under an exchangeable/i.i.d. stream of independent episode scores, a fixed
//! score function, independent U(0,1) tie randomizers and the specified betting
//! family, randomized online conformal p-values are i.i.d. uniform. The weighted
//! restarted sum with summable deterministic weights has a distribution-free
//! PFA bound. This executable cannot prove episode independence, randomizer
//! independence, score immutability or context exogeneity from caller metadata.
//!
//! Near-harmonic weights:
//!   w_k = 1 / (2*k*(1+ln(k))^2)
//! For f(x)=1/[x(1+ln x)^2], f is decreasing and
//!   sum_{k>=1} f(k) <= f(1) + integral_1^infinity f(x)dx = 1 + 1 = 2,
//! hence sum w_k <= 1.
//!
//! Usage:
//!   hardsec_episode_restarted_monitor --plan plan.json --episodes episodes.json
//!     --context-receipt receipt.json --receipt output.json

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PLAN_SCHEMA: &str = "hardsec-episode-restarted-plan-v1";
const STREAM_SCHEMA: &str = "hardsec-episode-window-stream-v1";
const ASSIGNMENT_SCHEMA: &str = "hardsec-context-assignment-receipt-v1";
const RECEIPT_SCHEMA: &str = "hardsec-episode-restarted-receipt-v1";
const QUALIFIED_PMU: &str = "qualified_for_named_acquisition_conditions";

#[derive(Debug, Clone)]
struct BettingTerm {
    theta: f64,
    prior_weight: f64,
}

#[derive(Debug, Clone)]
struct Plan {
    monitor_id: String,
    evidence_domain: String,
    stream_id: String,
    evaluation_group_id: String,
    score_artifact_sha256: String,
    context_assignment_policy_id: String,
    context_assignment_artifact_sha256: String,
    randomizer_source_sha256: String,
    windows_per_episode: usize,
    alpha_total: f64,
    contexts: BTreeSet<String>,
    betting_family: Vec<BettingTerm>,
    training_groups: BTreeSet<String>,
    selection_groups: BTreeSet<String>,
    evaluation_groups: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct Window {
    window_index: usize,
    source_sequence_index: usize,
    window_id: String,
    anomaly_score: f64,
    sensor_state: String,
    pmu_qualification_result: String,
    source_offset_start: usize,
    source_offset_end: usize,
}

#[derive(Debug, Clone)]
struct Episode {
    sequence_index: usize,
    episode_id: String,
    execution_group_id: String,
    source_trace_sha256: String,
    program_binary_sha256: String,
    device_config_id: String,
    randomizer_u: f64,
    windows: Vec<Window>,
}

#[derive(Debug, Clone)]
struct Assignment {
    source_sequence_index: usize,
    window_id: String,
    decision_status: String,
    decision_reason: String,
    selected_context_id: Option<String>,
}

#[derive(Debug, Clone)]
struct ContextState {
    alpha: f64,
    history_transformed_scores: Vec<f64>,
    log_weighted_products: Vec<f64>,
    log_fixed_start_products: Vec<f64>,
    valid_episode_count: usize,
    first_restarted_alarm_sequence: Option<usize>,
    first_fixed_start_alarm_sequence: Option<usize>,
    latest_log_restarted_sum: f64,
    latest_log_fixed_start: f64,
}

#[derive(Debug, Clone)]
struct Args {
    plan: PathBuf,
    episodes: PathBuf,
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

fn number_field(value: &Value, key: &str, context: &str) -> Result<f64, String> {
    let result = value.get(key).and_then(Value::as_f64)
        .ok_or_else(|| format!("{context}.{key} must be numeric"))?;
    if !result.is_finite() {
        return Err(format!("{context}.{key} must be finite"));
    }
    Ok(result)
}

fn count_field(value: &Value, key: &str, context: &str) -> Result<usize, String> {
    let raw = value.get(key).and_then(Value::as_u64)
        .ok_or_else(|| format!("{context}.{key} must be a non-negative integer"))?;
    usize::try_from(raw).map_err(|_| format!("{context}.{key} is outside supported range"))
}

fn valid_sha(value: &str, context: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(format!("{context} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn unique_string_set(value: &Value, key: &str, context: &str) -> Result<BTreeSet<String>, String> {
    let values = value.get(key).and_then(Value::as_array)
        .ok_or_else(|| format!("{context}.{key} must be an array"))?;
    let mut output = BTreeSet::new();
    for (index, item) in values.iter().enumerate() {
        let item = item.as_str().ok_or_else(|| format!("{context}.{key}[{index}] must be a string"))?;
        nonempty(item, &format!("{context}.{key}[{index}]"))?;
        if !output.insert(item.to_owned()) {
            return Err(format!("{context}.{key} contains duplicate value {item}"));
        }
    }
    Ok(output)
}

fn parse_plan(value: &Value) -> Result<Plan, String> {
    if string_field(value, "schema_version", "plan")? != PLAN_SCHEMA {
        return Err("unsupported episode plan schema_version".to_owned());
    }
    for (field, expected) in [
        ("aggregation_policy", "max_anomaly_score"),
        ("invalid_window_policy", "abstain_entire_episode"),
        ("restart_weight_policy", "near_harmonic_summable"),
        ("restart_weight_formula", "w_k=1/(2*k*(1+ln(k))^2)"),
        ("pfa_statistic", "weighted_restarted_sum"),
    ] {
        if string_field(value, field, "plan")? != expected {
            return Err(format!("unsupported plan.{field}"));
        }
    }
    for field in [
        "fixed_before_evaluation",
        "score_artifact_frozen",
        "require_contiguous_window_indices",
        "require_single_context_per_episode",
        "randomizers_strictly_between_zero_and_one",
    ] {
        if value.get(field).and_then(Value::as_bool) != Some(true) {
            return Err(format!("plan.{field} must be true"));
        }
    }
    if string_field(value, "randomizer_independence_status", "plan")? != "assumption_not_verified" {
        return Err("randomizer independence must remain assumption_not_verified".to_owned());
    }

    let evidence_domain = string_field(value, "evidence_domain", "plan")?.to_owned();
    if !matches!(evidence_domain.as_str(), "synthetic_fixture" | "simulator" | "physical_hardware") {
        return Err("unsupported episode plan evidence_domain".to_owned());
    }
    let score_artifact_sha256 = string_field(value, "score_artifact_sha256", "plan")?.to_owned();
    let context_assignment_artifact_sha256 =
        string_field(value, "context_assignment_artifact_sha256", "plan")?.to_owned();
    let randomizer_source_sha256 = string_field(value, "randomizer_source_sha256", "plan")?.to_owned();
    valid_sha(&score_artifact_sha256, "plan.score_artifact_sha256")?;
    valid_sha(&context_assignment_artifact_sha256, "plan.context_assignment_artifact_sha256")?;
    valid_sha(&randomizer_source_sha256, "plan.randomizer_source_sha256")?;
    if score_artifact_sha256 == context_assignment_artifact_sha256
        || score_artifact_sha256 == randomizer_source_sha256
        || context_assignment_artifact_sha256 == randomizer_source_sha256 {
        return Err("score, context-assignment and randomizer artifacts must have separate digests".to_owned());
    }

    let contexts = unique_string_set(value, "contexts", "plan")?;
    if contexts.len() < 2 {
        return Err("at least two contexts are required for the preregistered multi-context plan".to_owned());
    }
    let alpha_total = number_field(value, "alpha_budget_total", "plan")?;
    if !(0.0..1.0).contains(&alpha_total) {
        return Err("alpha_budget_total must be strictly between zero and one".to_owned());
    }
    let windows_per_episode = count_field(value, "windows_per_episode", "plan")?;
    if windows_per_episode < 2 {
        return Err("windows_per_episode must be at least two".to_owned());
    }

    let betting_values = value.get("betting_family").and_then(Value::as_array)
        .ok_or("plan.betting_family must be an array")?;
    if betting_values.is_empty() {
        return Err("plan.betting_family must not be empty".to_owned());
    }
    let mut betting_family = Vec::with_capacity(betting_values.len());
    let mut prior_sum = 0.0;
    let mut seen_theta = BTreeSet::new();
    for (index, item) in betting_values.iter().enumerate() {
        let context = format!("plan.betting_family[{index}]");
        let theta = number_field(item, "theta", &context)?;
        let prior_weight = number_field(item, "prior_weight", &context)?;
        if !(0.0..1.0).contains(&theta) || !(0.0..1.0).contains(&prior_weight) {
            return Err(format!("{context}: theta and prior_weight must be in (0,1)"));
        }
        if !seen_theta.insert(theta.to_bits()) {
            return Err(format!("{context}: duplicate theta"));
        }
        prior_sum += prior_weight;
        betting_family.push(BettingTerm { theta, prior_weight });
    }
    if (prior_sum - 1.0).abs() > 1e-12 {
        return Err(format!("betting-family prior weights must sum to one; observed {prior_sum}"));
    }

    let training_groups = unique_string_set(value, "score_model_training_execution_group_ids", "plan")?;
    let selection_groups = unique_string_set(value, "score_model_selection_execution_group_ids", "plan")?;
    let evaluation_groups = unique_string_set(value, "evaluation_execution_group_ids", "plan")?;
    if training_groups.is_empty() || selection_groups.is_empty() || evaluation_groups.is_empty() {
        return Err("training, selection and evaluation execution-group sets must be non-empty".to_owned());
    }
    if !training_groups.is_disjoint(&selection_groups)
        || !training_groups.is_disjoint(&evaluation_groups)
        || !selection_groups.is_disjoint(&evaluation_groups) {
        return Err("score-model training, selection and evaluation execution groups overlap".to_owned());
    }

    Ok(Plan {
        monitor_id: string_field(value, "monitor_id", "plan")?.to_owned(),
        evidence_domain,
        stream_id: string_field(value, "stream_id", "plan")?.to_owned(),
        evaluation_group_id: string_field(value, "evaluation_group_id", "plan")?.to_owned(),
        score_artifact_sha256,
        context_assignment_policy_id: string_field(value, "context_assignment_policy_id", "plan")?.to_owned(),
        context_assignment_artifact_sha256,
        randomizer_source_sha256,
        windows_per_episode,
        alpha_total,
        contexts,
        betting_family,
        training_groups,
        selection_groups,
        evaluation_groups,
    })
}

fn parse_episodes(value: &Value, plan: &Plan) -> Result<Vec<Episode>, String> {
    if string_field(value, "schema_version", "episodes")? != STREAM_SCHEMA {
        return Err("unsupported episode stream schema_version".to_owned());
    }
    for (field, expected) in [
        ("evidence_domain", plan.evidence_domain.as_str()),
        ("stream_id", plan.stream_id.as_str()),
        ("evaluation_group_id", plan.evaluation_group_id.as_str()),
        ("score_artifact_sha256", plan.score_artifact_sha256.as_str()),
        ("context_assignment_policy_id", plan.context_assignment_policy_id.as_str()),
        ("context_assignment_artifact_sha256", plan.context_assignment_artifact_sha256.as_str()),
        ("randomizer_source_sha256", plan.randomizer_source_sha256.as_str()),
    ] {
        if string_field(value, field, "episodes")? != expected {
            return Err(format!("episodes.{field} does not match the preregistered plan"));
        }
    }
    let records = value.get("records").and_then(Value::as_array)
        .ok_or("episodes.records must be an array")?;
    if records.is_empty() {
        return Err("episodes.records must not be empty".to_owned());
    }
    if records.len() != plan.evaluation_groups.len() {
        return Err("episode count must equal the preregistered evaluation execution-group count".to_owned());
    }

    let mut episode_ids = BTreeSet::new();
    let mut execution_groups = BTreeSet::new();
    let mut trace_digests = BTreeSet::new();
    let mut window_ids = BTreeSet::new();
    let mut source_sequence_ids = BTreeSet::new();
    let mut output = Vec::with_capacity(records.len());
    let mut expected_source_sequence = 1usize;

    for (index, item) in records.iter().enumerate() {
        let context = format!("episodes.records[{index}]");
        let sequence_index = count_field(item, "sequence_index", &context)?;
        if sequence_index != index + 1 {
            return Err(format!("{context}.sequence_index must be contiguous from one"));
        }
        let episode_id = string_field(item, "episode_id", &context)?.to_owned();
        if !episode_ids.insert(episode_id.clone()) {
            return Err(format!("duplicate episode_id {episode_id}"));
        }
        let execution_group_id = string_field(item, "execution_group_id", &context)?.to_owned();
        if !plan.evaluation_groups.contains(&execution_group_id) {
            return Err(format!("{context} execution group is not in the preregistered evaluation set"));
        }
        if !execution_groups.insert(execution_group_id.clone()) {
            return Err(format!("execution group {execution_group_id} appears in more than one episode"));
        }
        let source_trace_sha256 = string_field(item, "source_trace_sha256", &context)?.to_owned();
        valid_sha(&source_trace_sha256, &format!("{context}.source_trace_sha256"))?;
        if !trace_digests.insert(source_trace_sha256.clone()) {
            return Err(format!("source trace digest {source_trace_sha256} is reused across episodes"));
        }
        let program_binary_sha256 = string_field(item, "program_binary_sha256", &context)?.to_owned();
        valid_sha(&program_binary_sha256, &format!("{context}.program_binary_sha256"))?;
        let device_config_id = string_field(item, "device_config_id", &context)?.to_owned();
        let randomizer_u = number_field(item, "randomizer_u", &context)?;
        if !(0.0..1.0).contains(&randomizer_u) {
            return Err(format!("{context}.randomizer_u must be strictly between zero and one"));
        }

        let window_values = item.get("windows").and_then(Value::as_array)
            .ok_or_else(|| format!("{context}.windows must be an array"))?;
        if window_values.len() != plan.windows_per_episode {
            return Err(format!("{context} has {} windows; exactly {} are required", window_values.len(), plan.windows_per_episode));
        }
        let mut windows = Vec::with_capacity(window_values.len());
        let mut previous_end = None;
        let mut window_width = None;
        for (window_index, window_value) in window_values.iter().enumerate() {
            let wc = format!("{context}.windows[{window_index}]");
            let parsed_window_index = count_field(window_value, "window_index", &wc)?;
            if parsed_window_index != window_index {
                return Err(format!("{wc}.window_index must be contiguous from zero"));
            }
            let source_sequence_index = count_field(window_value, "source_sequence_index", &wc)?;
            if source_sequence_index != expected_source_sequence {
                return Err(format!("{wc}.source_sequence_index must be contiguous across the episode stream; expected {expected_source_sequence}, got {source_sequence_index}"));
            }
            expected_source_sequence += 1;
            if !source_sequence_ids.insert(source_sequence_index) {
                return Err(format!("duplicate source sequence index {source_sequence_index}"));
            }
            let window_id = string_field(window_value, "window_id", &wc)?.to_owned();
            if !window_ids.insert(window_id.clone()) {
                return Err(format!("duplicate or cross-episode window_id {window_id}"));
            }
            let anomaly_score = number_field(window_value, "anomaly_score", &wc)?;
            let sensor_state = string_field(window_value, "sensor_state", &wc)?.to_owned();
            if !matches!(sensor_state.as_str(), "valid" | "missing" | "stale" | "misaligned" | "multiplexed" | "untrusted") {
                return Err(format!("{wc}.sensor_state is unsupported"));
            }
            let pmu_qualification_result = string_field(window_value, "pmu_qualification_result", &wc)?.to_owned();
            let source_offset_start = count_field(window_value, "source_offset_start", &wc)?;
            let source_offset_end = count_field(window_value, "source_offset_end", &wc)?;
            if window_index == 0 && source_offset_start != 0 {
                return Err(format!("{wc} first source window must begin at offset zero"));
            }
            if source_offset_end <= source_offset_start {
                return Err(format!("{wc} source offsets must have positive width"));
            }
            if let Some(end) = previous_end {
                if source_offset_start != end {
                    return Err(format!("{wc} has a gap or overlap with the previous source window"));
                }
            }
            let width = source_offset_end - source_offset_start;
            if let Some(expected_width) = window_width {
                if width != expected_width {
                    return Err(format!("{wc} has a variable window width within the fixed episode"));
                }
            } else {
                window_width = Some(width);
            }
            previous_end = Some(source_offset_end);

            windows.push(Window {
                window_index: parsed_window_index,
                source_sequence_index,
                window_id,
                anomaly_score,
                sensor_state,
                pmu_qualification_result,
                source_offset_start,
                source_offset_end,
            });
        }

        output.push(Episode {
            sequence_index,
            episode_id,
            execution_group_id,
            source_trace_sha256,
            program_binary_sha256,
            device_config_id,
            randomizer_u,
            windows,
        });
    }
    if execution_groups != plan.evaluation_groups {
        return Err("observed execution groups do not exactly match the preregistered evaluation set".to_owned());
    }
    Ok(output)
}

fn parse_context_receipt(
    value: &Value,
    plan: &Plan,
    episodes: &[Episode],
) -> Result<BTreeMap<String, Assignment>, String> {
    if string_field(value, "schema_version", "context_receipt")? != ASSIGNMENT_SCHEMA
        || string_field(value, "result", "context_receipt")? != "audited"
        || string_field(value, "claim_scope", "context_receipt")?
            != "declared_context_assignment_gate_and_heldout_label_evaluation_only" {
        return Err("context assignment receipt schema/result/claim scope is unsupported".to_owned());
    }
    for (field, expected) in [
        ("policy_id", plan.context_assignment_policy_id.as_str()),
        ("stream_id", plan.stream_id.as_str()),
        ("evaluation_group_id", plan.evaluation_group_id.as_str()),
        ("evidence_domain", plan.evidence_domain.as_str()),
        ("assignment_artifact_sha256", plan.context_assignment_artifact_sha256.as_str()),
        ("anomaly_score_artifact_sha256", plan.score_artifact_sha256.as_str()),
    ] {
        if string_field(value, field, "context_receipt")? != expected {
            return Err(format!("context receipt {field} does not match plan/episode stream"));
        }
    }
    let inputs = value.get("inputs").ok_or("context receipt inputs missing")?;
    for field in ["policy_sha256", "assignments_sha256", "truth_oracle_sha256"] {
        let digest = string_field(inputs, field, "context_receipt.inputs")?;
        valid_sha(digest, &format!("context_receipt.inputs.{field}"))?;
    }
    let mapping = value.get("runtime_mapping").ok_or("context receipt runtime_mapping missing")?;
    if string_field(mapping, "mapping_schema_version", "runtime_mapping")?
        != "hardsec-context-assignment-runtime-map-v1"
        || mapping.get("contains_truth_labels").and_then(Value::as_bool) != Some(false)
        || mapping.get("mapping_digest_bound_to_assignment_input").and_then(Value::as_bool) != Some(true) {
        return Err("context runtime mapping must use the truth-free, digest-bound HARDSEC-020 schema".to_owned());
    }
    let assignment_values = mapping.get("assignments").and_then(Value::as_array)
        .ok_or("runtime mapping assignments must be an array")?;
    let expected_window_count: usize = episodes.iter().map(|e| e.windows.len()).sum();
    if assignment_values.len() != expected_window_count {
        return Err("context assignment receipt must map every episode source window exactly once".to_owned());
    }
    let integrity = value.get("assignment_integrity").ok_or("context receipt assignment_integrity missing")?;
    let accepted_count = assignment_values.iter().filter(|a| a.get("decision_status").and_then(Value::as_str)==Some("accepted")).count();
    let abstained_count = assignment_values.iter().filter(|a| a.get("decision_status").and_then(Value::as_str)==Some("abstained")).count();
    if count_field(integrity, "assignment_record_count", "assignment_integrity")? != assignment_values.len()
        || count_field(integrity, "accepted_count", "assignment_integrity")? != accepted_count
        || count_field(integrity, "abstained_count", "assignment_integrity")? != abstained_count {
        return Err("context receipt aggregate counts do not match the per-window mapping".to_owned());
    }

    let mut map = BTreeMap::new();
    for (index, item) in assignment_values.iter().enumerate() {
        let context = format!("runtime_mapping.assignments[{index}]");
        let source_sequence_index = count_field(item, "sequence_index", &context)?;
        if !mapped_source_sequences.insert(source_sequence_index) {
            return Err(format!("duplicate context assignment source sequence index {source_sequence_index}"));
        }
        let window_id = string_field(item, "window_id", &context)?.to_owned();
        let decision_status = string_field(item, "decision_status", &context)?.to_owned();
        let decision_reason = string_field(item, "decision_reason", &context)?.to_owned();
        if !matches!(decision_status.as_str(), "accepted" | "abstained")
            || !matches!(decision_reason.as_str(), "accepted" | "invalid_sensor" | "unknown_context" | "ambiguous_assignment") {
            return Err(format!("{context} has unsupported decision status/reason"));
        }
        let selected_context_id = match item.get("selected_context_id") {
            Some(Value::String(v)) => Some(v.clone()),
            Some(Value::Null) => None,
            _ => return Err(format!("{context}.selected_context_id must be string or null")),
        };
        if (decision_status == "accepted" && (decision_reason != "accepted" || selected_context_id.is_none()))
            || (decision_status == "abstained" && selected_context_id.is_some()) {
            return Err(format!("{context} selected context is inconsistent with its decision"));
        }
        let assignment = Assignment {
            source_sequence_index,
            window_id: window_id.clone(),
            decision_status,
            decision_reason,
            selected_context_id,
        };
        if map.insert(window_id.clone(), assignment).is_some() {
            return Err(format!("duplicate context assignment window_id {window_id}"));
        }
    }

    for episode in episodes {
        for window in &episode.windows {
            let assignment = map.get(&window.window_id)
                .ok_or_else(|| format!("context receipt omits source window {}", window.window_id))?;
            if assignment.source_sequence_index != window.source_sequence_index {
                return Err(format!("context receipt source sequence mismatch for {}", window.window_id));
            }
        }
    }
    Ok(map)
}

fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY { return b; }
    if b == f64::NEG_INFINITY { return a; }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

fn log_sum_exp(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("log_sum_exp requires at least one value".to_owned());
    }
    let m = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return Ok(m);
    }
    if !m.is_finite() {
        return Err("log_sum_exp received a non-finite value".to_owned());
    }
    let sum: f64 = values.iter().map(|value| (*value - m).exp()).sum();
    let result = m + sum.ln();
    if !result.is_finite() {
        return Err("log_sum_exp result is non-finite".to_owned());
    }
    Ok(result)
}

fn near_harmonic_restart_weight(k: usize) -> f64 {
    let k = k as f64;
    1.0 / (2.0 * k * (1.0 + k.ln()).powi(2))
}

fn online_randomized_conformal_p(
    previous_transformed_scores: &[f64],
    anomaly_score: f64,
    randomizer_u: f64,
) -> Result<f64, String> {
    if !(0.0..1.0).contains(&randomizer_u) {
        return Err("randomizer_u must be strictly between zero and one".to_owned());
    }
    if !anomaly_score.is_finite() {
        return Err("episode anomaly score must be finite".to_owned());
    }
    // Negate scores so larger anomaly scores occupy the lower tail.
    let transformed = -anomaly_score;
    let less_count = previous_transformed_scores.iter().filter(|old| **old < transformed).count();
    // Include the current item in its own tie block.
    let equal_count = previous_transformed_scores.iter().filter(|old| **old == transformed).count() + 1;
    let t = previous_transformed_scores.len() + 1;
    let p = (less_count as f64 + randomizer_u * equal_count as f64) / t as f64;
    if !p.is_finite() || !(0.0..1.0).contains(&p) {
        return Err(format!("randomized conformal p-value is outside (0,1): {p}"));
    }
    Ok(p)
}

impl ContextState {
    fn new(alpha: f64, betting_family: &[BettingTerm]) -> Self {
        Self {
            alpha,
            history_transformed_scores: Vec::new(),
            log_weighted_products: vec![f64::NEG_INFINITY; betting_family.len()],
            log_fixed_start_products: vec![0.0; betting_family.len()],
            valid_episode_count: 0,
            first_restarted_alarm_sequence: None,
            first_fixed_start_alarm_sequence: None,
            latest_log_restarted_sum: f64::NEG_INFINITY,
            latest_log_fixed_start: 0.0,
        }
    }

    fn update(
        &mut self,
        score: f64,
        randomizer_u: f64,
        sequence_index: usize,
        betting_family: &[BettingTerm],
    ) -> Result<(f64, f64, f64, f64, bool, bool), String> {
        let p = online_randomized_conformal_p(&self.history_transformed_scores, score, randomizer_u)?;
        let transformed = -score;
        self.history_transformed_scores.push(transformed);
        self.valid_episode_count += 1;
        let k = self.valid_episode_count;
        let log_weight = near_harmonic_restart_weight(k).ln();

        for (index, term) in betting_family.iter().enumerate() {
            let log_f = term.theta.ln() + (term.theta - 1.0) * p.ln();
            // A_{j,t}=f_j(p_t)*(A_{j,t-1}+w_t), the O(J) recurrence
            // for sum_k w_k * product_{r=k}^t f_j(p_r).
            self.log_weighted_products[index] =
                log_f + log_add_exp(self.log_weighted_products[index], log_weight);
            self.log_fixed_start_products[index] += log_f;
        }
        let restarted_components: Vec<f64> = betting_family.iter().enumerate()
            .map(|(i, term)| term.prior_weight.ln() + self.log_weighted_products[i])
            .collect();
        let fixed_components: Vec<f64> = betting_family.iter().enumerate()
            .map(|(i, term)| term.prior_weight.ln() + self.log_fixed_start_products[i])
            .collect();
        let log_restarted_sum = log_sum_exp(&restarted_components)?;
        let log_fixed_start = log_sum_exp(&fixed_components)?;
        self.latest_log_restarted_sum = log_restarted_sum;
        self.latest_log_fixed_start = log_fixed_start;

        let threshold = -self.alpha.ln();
        let restarted_crossing = self.first_restarted_alarm_sequence.is_none()
            && log_restarted_sum >= threshold;
        let fixed_crossing = self.first_fixed_start_alarm_sequence.is_none()
            && log_fixed_start >= threshold;
        if restarted_crossing {
            self.first_restarted_alarm_sequence = Some(sequence_index);
        }
        if fixed_crossing {
            self.first_fixed_start_alarm_sequence = Some(sequence_index);
        }

        Ok((p, log_restarted_sum, log_fixed_start, threshold, restarted_crossing, fixed_crossing))
    }
}

fn audit_values(
    plan_value: &Value,
    episode_value: &Value,
    context_receipt_value: &Value,
    plan_bytes: &[u8],
    episode_bytes: &[u8],
    context_receipt_bytes: &[u8],
) -> Result<Value, String> {
    let plan = parse_plan(plan_value)?;
    let episodes = parse_episodes(episode_value, &plan)?;
    let assignments = parse_context_receipt(context_receipt_value, &plan, &episodes)?;
    let alpha_per_context = plan.alpha_total / plan.contexts.len() as f64;
    let mut states: BTreeMap<String, ContextState> = plan.contexts.iter()
        .map(|context| (context.clone(), ContextState::new(alpha_per_context, &plan.betting_family)))
        .collect();

    let mut total_source_windows = 0usize;
    let mut total_accepted_episodes = 0usize;
    let mut total_abstained_episodes = 0usize;
    let mut abstention_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut output_episodes = Vec::with_capacity(episodes.len());
    let mut overall_first_alarm_sequence: Option<usize> = None;
    let mut overall_first_alarm_context: Option<String> = None;

    for episode in &episodes {
        total_source_windows += episode.windows.len();
        let mut assigned_context: Option<String> = None;
        let mut assignment_failure: Option<&'static str> = None;
        for window in &episode.windows {
            let assignment = assignments.get(&window.window_id).ok_or("assignment map lookup invariant failed")?;
            if assignment.decision_status != "accepted" {
                assignment_failure = Some("context_assignment_not_accepted");
                break;
            }
            let selected = assignment.selected_context_id.as_ref().ok_or("accepted assignment missing selected context")?;
            if !plan.contexts.contains(selected) {
                assignment_failure = Some("unseen_workload_context");
                break;
            }
            if let Some(prior) = &assigned_context {
                if prior != selected {
                    assignment_failure = Some("mixed_context_episode");
                    break;
                }
            } else {
                assigned_context = Some(selected.clone());
            }
        }

        let invalid_sensor = episode.windows.iter().any(|w| w.sensor_state != "valid");
        let unqualified_pmu = episode.windows.iter().any(|w| w.pmu_qualification_result != QUALIFIED_PMU);
        let abstention_reason = if invalid_sensor {
            Some("invalid_sensor_state")
        } else if unqualified_pmu {
            Some("pmu_acquisition_not_qualified")
        } else if let Some(reason) = assignment_failure {
            Some(reason)
        } else if assigned_context.is_none() {
            Some("missing_context_assignment")
        } else {
            None
        };

        if let Some(reason) = abstention_reason {
            total_abstained_episodes += 1;
            *abstention_counts.entry(reason.to_owned()).or_default() += 1;
            output_episodes.push(json!({
                "sequence_index":episode.sequence_index,
                "episode_id":episode.episode_id,
                "execution_group_id":episode.execution_group_id,
                "source_trace_sha256":episode.source_trace_sha256,
                "program_binary_sha256":episode.program_binary_sha256,
                "device_config_id":episode.device_config_id,
                "source_window_count":episode.windows.len(),
                "aggregated_episode_score":null,
                "context_id":assigned_context,
                "action":"abstain",
                "abstention_reason":reason,
                "context_valid_episode_index":null,
                "randomizer_u":episode.randomizer_u,
                "conformal_p_value":null,
                "log_restarted_sum":null,
                "log_fixed_start_cmm":null,
                "threshold_log_wealth":null
            }));
            continue;
        }

        let context_id = assigned_context.ok_or("accepted episode must have one context")?;
        let episode_score = episode.windows.iter().map(|w| w.anomaly_score)
            .fold(f64::NEG_INFINITY, f64::max);
        if !episode_score.is_finite() {
            return Err(format!("episode {} aggregate score is non-finite", episode.episode_id));
        }
        let state = states.get_mut(&context_id).ok_or("context state invariant failed")?;
        let (p_value, log_restarted_sum, log_fixed_start, threshold, restarted_crossing, fixed_crossing) =
            state.update(episode_score, episode.randomizer_u, episode.sequence_index, &plan.betting_family)?;
        total_accepted_episodes += 1;
        if restarted_crossing && overall_first_alarm_sequence.is_none() {
            overall_first_alarm_sequence = Some(episode.sequence_index);
            overall_first_alarm_context = Some(context_id.clone());
        }
        let action = if restarted_crossing {
            "restarted_alarm_crossing"
        } else if state.first_restarted_alarm_sequence.is_some() {
            "restarted_alarm_latched"
        } else {
            "monitoring_no_restarted_alarm"
        };

        output_episodes.push(json!({
            "sequence_index":episode.sequence_index,
            "episode_id":episode.episode_id,
            "execution_group_id":episode.execution_group_id,
            "source_trace_sha256":episode.source_trace_sha256,
            "program_binary_sha256":episode.program_binary_sha256,
            "device_config_id":episode.device_config_id,
            "source_window_count":episode.windows.len(),
            "aggregated_episode_score":episode_score,
            "aggregation_policy":"max_anomaly_score",
            "context_id":context_id,
            "action":action,
            "abstention_reason":null,
            "context_valid_episode_index":state.valid_episode_count,
            "randomizer_u":episode.randomizer_u,
            "conformal_p_value":p_value,
            "log_restarted_sum":log_restarted_sum,
            "log_fixed_start_cmm":log_fixed_start,
            "threshold_log_wealth":threshold,
            "fixed_start_baseline_crossing_now":fixed_crossing
        }));
    }

    let context_reports: Vec<Value> = states.iter().map(|(context, state)| json!({
        "context_id":context,
        "alpha_allocation":state.alpha,
        "threshold":1.0/state.alpha,
        "valid_episode_count":state.valid_episode_count,
        "first_restarted_alarm_sequence_index":state.first_restarted_alarm_sequence,
        "first_fixed_start_cmm_alarm_sequence_index":state.first_fixed_start_alarm_sequence,
        "final_log_restarted_sum":state.latest_log_restarted_sum,
        "final_log_fixed_start_cmm":state.latest_log_fixed_start
    })).collect();

    let result = if overall_first_alarm_sequence.is_some() {
        "restarted_alarm_threshold_crossed"
    } else {
        "no_restarted_alarm_threshold_crossed"
    };

    Ok(json!({
        "schema_version":RECEIPT_SCHEMA,
        "result":result,
        "claim_scope":"fixed_episode_packaging_and_weighted_restarted_conformal_monitor_mechanics_only",
        "monitor_id":plan.monitor_id,
        "stream_id":plan.stream_id,
        "evaluation_group_id":plan.evaluation_group_id,
        "evidence_domain":plan.evidence_domain,
        "score_artifact_sha256":plan.score_artifact_sha256,
        "context_assignment_policy_id":plan.context_assignment_policy_id,
        "context_assignment_artifact_sha256":plan.context_assignment_artifact_sha256,
        "randomizer_source_sha256":plan.randomizer_source_sha256,
        "inputs":{
            "plan_sha256":sha256(plan_bytes),
            "episode_stream_sha256":sha256(episode_bytes),
            "context_assignment_receipt_sha256":sha256(context_receipt_bytes)
        },
        "episode_integrity":{
            "episode_count":episodes.len(),
            "accepted_episode_count":total_accepted_episodes,
            "abstained_episode_count":total_abstained_episodes,
            "source_window_count":total_source_windows,
            "windows_per_accepted_episode":plan.windows_per_episode,
            "one_episode_per_unique_execution_group":true,
            "source_window_ids_unique":true,
            "source_offsets_contiguous_within_episode":true,
            "source_offsets_start_at_zero":true,
            "evaluation_execution_groups_disjoint_from_model_training_and_selection":true,
            "declared_execution_group_independence":"not_established_from_input_receipts"
        },
        "context_assignment":{
            "receipt_sha256":sha256(context_receipt_bytes),
            "assignment_record_count":assignments.len(),
            "truth_labels_in_runtime_mapping":false,
            "one_context_per_accepted_episode_required":true,
            "mixed_context_episode_policy":"abstain_entire_episode"
        },
        "sequential_monitor":{
            "method":"weighted_restarted_conformal_mixture",
            "pfa_alpha_total":plan.alpha_total,
            "context_count":plan.contexts.len(),
            "overall_first_alarm_sequence_index":overall_first_alarm_sequence,
            "overall_first_alarm_context_id":overall_first_alarm_context,
            "accepted_episode_count":total_accepted_episodes,
            "abstained_episode_count":total_abstained_episodes,
            "abstention_counts_by_reason":abstention_counts,
            "primary_statistic":"S_t=sum_{k=1}^t w_k*M_{k,t}",
            "restart_weight_formula":"1/(2*k*(1+ln(k))^2)",
            "restart_weight_sum_upper_bound":1.0,
            "betting_family":plan.betting_family.iter().map(|v| json!({"theta":v.theta,"prior_weight":v.prior_weight})).collect::<Vec<_>>(),
            "context_results":context_reports,
            "episode_decisions":output_episodes
        },
        "statistical_guarantees":{
            "pfa_bound_if_all_theorem_assumptions_hold":plan.alpha_total,
            "episode_score_exchangeability_status":"assumption_not_verified",
            "cross_execution_independence_status":"not_established",
            "randomizer_independence_status":"assumption_not_verified",
            "score_artifact_fixed_status":"digest_bound_not_authenticated",
            "context_exogeneity_status":"not_established_from_input_receipts",
            "episode_abstention_noninformative_status":"assumption_not_verified",
            "theorem_guarantee_status":"not_established_from_input_receipts",
            "effective_sample_size_estimate":null
        },
        "theoretical_contract":{
            "null_unit":"one fixed-length episode per independent execution group",
            "within_episode_windows":"may be correlated; collapsed to one max score before testing",
            "online_p_values":"randomized sequential conformal ranks on the fixed episode-score stream",
            "randomizer":"U(0,1) independent of episode scores and past; supplied in input but not authenticated",
            "weighted_restarted_sum":"sum_k w_k * integral product_{r=k}^t f_theta(p_r) dPi(theta)",
            "weight_schedule":"w_k=1/(2*k*(1+ln(k))^2)",
            "weight_sum_proof":"decreasing f(x)=1/[x(1+ln x)^2]; sum f(k)<=f(1)+integral_1^infinity f(x)dx=2; therefore sum w_k<=1",
            "pfa_threshold":"1/alpha_context, Bonferroni alpha_context=alpha_total/number_of_contexts",
            "efficient_recurrence":"A[j,t]=f_j(p_t)*(A[j,t-1]+w_t), so weighted sum update is O(number_of_bets) per valid episode",
            "delay_unit":"independent execution episodes, not PMU windows"
        },
        "limitations":[
            "Unique execution IDs and disjoint declared group IDs do not prove that executions are independent or exchangeable.",
            "The p-value randomizers are caller-supplied; their independence from scores and past is not authenticated.",
            "The maximum episode score is fixed-length and preregistered, but aggregation does not authenticate the source trace or acquisition process.",
            "The PFA bound applies only under the independent/exchangeable episode null, fixed score function, independent randomizer and valid context-assignment assumptions.",
            "Abstaining on invalid episodes can be informative; the non-informative episode-selection assumption is not verified.",
            "The fixed-start CMM score is a comparison diagnostic; the restarted weighted sum is the only primary alarm statistic in v1.",
            "A distribution-change alarm does not identify a hardware attack or establish causal attack effect."
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
    let mut plan = None;
    let mut episodes = None;
    let mut context_receipt = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let target = match flag.as_str() {
            "--plan" => &mut plan,
            "--episodes" => &mut episodes,
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
        episodes:episodes.ok_or("missing required argument --episodes")?,
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
    let (plan_bytes, plan_value) = read_input(&args.plan, "plan")?;
    let (episode_bytes, episode_value) = read_input(&args.episodes, "episode stream")?;
    let (context_bytes, context_value) = read_input(&args.context_receipt, "context assignment receipt")?;
    validate_receipt_location(&args.receipt, &[
        args.plan.as_path(), args.episodes.as_path(), args.context_receipt.as_path()
    ])?;
    let receipt = audit_values(
        &plan_value, &episode_value, &context_value,
        &plan_bytes, &episode_bytes, &context_bytes
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
            eprintln!("HARDSEC-021 episode restarted monitor failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Value, Value, Value, Vec<u8>, Vec<u8>, Vec<u8>) {
        let pb = include_bytes!("../tests/fixtures/hardsec/episode-restarted-v1/plan.json").to_vec();
        let eb = include_bytes!("../tests/fixtures/hardsec/episode-restarted-v1/episodes.json").to_vec();
        let cb = include_bytes!("../tests/fixtures/hardsec/episode-restarted-v1/context-assignment-receipt.json").to_vec();
        let p = serde_json::from_slice(&pb).unwrap();
        let e = serde_json::from_slice(&eb).unwrap();
        let c = serde_json::from_slice(&cb).unwrap();
        (p,e,c,pb,eb,cb)
    }

    fn encode(value: &Value) -> Vec<u8> {
        let mut bytes = serde_json::to_vec_pretty(value).unwrap();
        bytes.push(b'\n');
        bytes
    }

    fn audit(p: &Value, e: &Value, c: &Value) -> Result<Value, String> {
        let pb=encode(p);let eb=encode(e);let cb=encode(c);
        audit_values(p,e,c,&pb,&eb,&cb)
    }

    #[test]
    fn known_answer_collapses_windows_and_crosses_after_five_attack_episodes() {
        let (p,e,c,pb,eb,cb)=fixture();
        let r=audit_values(&p,&e,&c,&pb,&eb,&cb).unwrap();
        assert_eq!(r["result"],"restarted_alarm_threshold_crossed");
        assert_eq!(r["episode_integrity"]["episode_count"],132);
        assert_eq!(r["episode_integrity"]["accepted_episode_count"],132);
        assert_eq!(r["episode_integrity"]["source_window_count"],528);
        assert_eq!(r["sequential_monitor"]["overall_first_alarm_sequence_index"],65);
        assert_eq!(r["sequential_monitor"]["overall_first_alarm_context_id"],"compute");
        assert_eq!(r["sequential_monitor"]["context_results"][0]["context_id"],"compute");
        assert_eq!(r["sequential_monitor"]["context_results"][0]["first_restarted_alarm_sequence_index"],65);
        assert_eq!(r["statistical_guarantees"]["episode_score_exchangeability_status"],"assumption_not_verified");
        assert_eq!(r["statistical_guarantees"]["randomizer_independence_status"],"assumption_not_verified");
        assert_eq!(r["statistical_guarantees"]["theorem_guarantee_status"],"not_established_from_input_receipts");
        assert_eq!(r["statistical_guarantees"]["effective_sample_size_estimate"],Value::Null);
    }

    #[test]
    fn restart_weight_schedule_has_proven_unit_sum_upper_bound() {
        let finite_sum: f64 = (1..1_000_000).map(near_harmonic_restart_weight).sum();
        assert!(finite_sum < 1.0);
        assert!(near_harmonic_restart_weight(1) > near_harmonic_restart_weight(2));
        assert!(near_harmonic_restart_weight(2) > near_harmonic_restart_weight(10));
    }

    #[test]
    fn online_randomized_conformal_p_value_matches_known_rank_and_ties() {
        let previous=vec![-4.0,-3.0,-2.0,-2.0];
        // Current transformed score -2.0: two prior values are strictly less,
        // three values are tied including the current item; t=5.
        let p=online_randomized_conformal_p(&previous,2.0,0.25).unwrap();
        assert!((p-0.55).abs()<1e-12);
    }

    #[test]
    fn recurrence_matches_brute_force_weighted_restarted_sum() {
        let family=vec![
            BettingTerm{theta:0.15,prior_weight:0.25},
            BettingTerm{theta:0.30,prior_weight:0.25},
            BettingTerm{theta:0.50,prior_weight:0.25},
            BettingTerm{theta:0.75,prior_weight:0.25},
        ];
        let ps=[0.2,0.4,0.01,0.07,0.03,0.5];
        let mut state=ContextState::new(0.05,&family);
        for (i,p) in ps.iter().enumerate() {
            // Build the recurrence directly from a valid randomized rank input
            // by using the betting recurrence helper below.
            let t=i+1;
            for (j,term) in family.iter().enumerate() {
                let logf=term.theta.ln()+(term.theta-1.0)*p.ln();
                state.log_weighted_products[j]=logf+log_add_exp(state.log_weighted_products[j],near_harmonic_restart_weight(t).ln());
                state.log_fixed_start_products[j]+=logf;
            }
            state.valid_episode_count=t;
        }
        let recurrence_components:Vec<f64>=family.iter().enumerate()
            .map(|(j,term)|term.prior_weight.ln()+state.log_weighted_products[j]).collect();
        let recurrence=log_sum_exp(&recurrence_components).unwrap();
        let mut brute_terms=Vec::new();
        for k in 1..=ps.len() {
            let mut mixture_components=Vec::new();
            for term in &family {
                let log_product: f64=ps[(k-1)..].iter().map(|p|term.theta.ln()+(term.theta-1.0)*p.ln()).sum();
                mixture_components.push(term.prior_weight.ln()+log_product);
            }
            brute_terms.push(near_harmonic_restart_weight(k).ln()+log_sum_exp(&mixture_components).unwrap());
        }
        let brute=log_sum_exp(&brute_terms).unwrap();
        assert!((recurrence-brute).abs()<1e-12);
    }

    #[test]
    fn rejects_variable_episode_length() {
        let (p,mut e,c,_,_,_)=fixture();
        e["records"][0]["windows"].as_array_mut().unwrap().pop();
        assert!(audit(&p,&e,&c).unwrap_err().contains("exactly 4 are required"));
    }

    #[test]
    fn rejects_duplicate_execution_group_reuse() {
        let (p,mut e,c,_,_,_)=fixture();
        e["records"][1]["execution_group_id"]=e["records"][0]["execution_group_id"].clone();
        assert!(audit(&p,&e,&c).unwrap_err().contains("appears in more than one episode"));
    }

    #[test]
    fn rejects_cross_episode_window_reuse() {
        let (p,mut e,c,_,_,_)=fixture();
        e["records"][1]["windows"][0]["window_id"]=e["records"][0]["windows"][0]["window_id"].clone();
        assert!(audit(&p,&e,&c).unwrap_err().contains("duplicate or cross-episode window_id"));
    }

    #[test]
    fn rejects_source_offset_overlap_and_gap() {
        let (p,mut e,c,_,_,_)=fixture();
        e["records"][0]["windows"][1]["source_offset_start"]=json!(50);
        assert!(audit(&p,&e,&c).unwrap_err().contains("gap or overlap"));
    }

    #[test]
    fn invalid_window_abstains_entire_episode() {
        let (p,mut e,c,_,_,_)=fixture();
        e["records"][0]["windows"][2]["sensor_state"]=json!("stale");
        let r=audit(&p,&e,&c).unwrap();
        assert_eq!(r["sequential_monitor"]["episode_decisions"][0]["action"],"abstain");
        assert_eq!(r["sequential_monitor"]["episode_decisions"][0]["abstention_reason"],"invalid_sensor_state");
        assert_eq!(r["episode_integrity"]["abstained_episode_count"],1);
        assert_eq!(r["episode_integrity"]["accepted_episode_count"],131);
    }

    #[test]
    fn rejects_context_receipt_assignment_artifact_mismatch() {
        let (p,e,mut c,_,_,_)=fixture();
        c["assignment_artifact_sha256"]=json!("f".repeat(64));
        assert!(audit(&p,&e,&c).unwrap_err().contains("assignment_artifact_sha256 does not match"));
    }

    #[test]
    fn mixed_context_episode_abstains_instead_of_splitting_independent_unit() {
        let (p,e,mut c,_,_,_)=fixture();
        c["runtime_mapping"]["assignments"][1]["selected_context_id"]=json!("memory");
        let r=audit(&p,&e,&c).unwrap();
        assert_eq!(r["sequential_monitor"]["episode_decisions"][0]["abstention_reason"],"mixed_context_episode");
    }

    #[test]
    fn abstained_context_window_abstains_whole_episode() {
        let (p,e,mut c,_,_,_)=fixture();
        c["runtime_mapping"]["assignments"][0]["decision_status"]=json!("abstained");
        c["runtime_mapping"]["assignments"][0]["decision_reason"]=json!("ambiguous_assignment");
        c["runtime_mapping"]["assignments"][0]["selected_context_id"]=Value::Null;
        c["assignment_integrity"]["accepted_count"]=json!(527);
        c["assignment_integrity"]["abstained_count"]=json!(1);
        let r=audit(&p,&e,&c).unwrap();
        assert_eq!(r["sequential_monitor"]["episode_decisions"][0]["abstention_reason"],"context_assignment_not_accepted");
    }

    #[test]
    fn rejects_randomizer_outside_open_unit_interval() {
        let (p,mut e,c,_,_,_)=fixture();
        e["records"][0]["randomizer_u"]=json!(0.0);
        assert!(audit(&p,&e,&c).unwrap_err().contains("strictly between zero and one"));
    }

    #[test]
    fn rejects_training_selection_evaluation_group_leakage() {
        let (mut p,e,c,_,_,_)=fixture();
        p["score_model_selection_execution_group_ids"][0]=p["score_model_training_execution_group_ids"][0].clone();
        assert!(audit(&p,&e,&c).unwrap_err().contains("training, selection and evaluation execution groups overlap"));
    }


    #[test]
    fn rejects_truth_labels_in_the_runtime_context_mapping() {
        let (p,e,mut c,_,_,_)=fixture();
        c["runtime_mapping"]["assignments"][0]["true_context_id"]=json!("compute");
        assert!(audit(&p,&e,&c).unwrap_err().contains("exactly the five declared fields"));
    }

    #[test]
    fn rejects_non_contiguous_context_assignment_source_sequence() {
        let (p,e,mut c,_,_,_)=fixture();
        c["runtime_mapping"]["assignments"][1]["sequence_index"]=json!(9);
        assert!(audit(&p,&e,&c).unwrap_err().contains("sequence_index must be contiguous from one"));
    }

    #[test]
    fn rejects_missing_assignment_window() {
        let (p,e,mut c,_,_,_)=fixture();
        c["runtime_mapping"]["assignments"].as_array_mut().unwrap().pop();
        c["assignment_integrity"]["assignment_record_count"]=json!(527);
        c["assignment_integrity"]["accepted_count"]=json!(527);
        assert!(audit(&p,&e,&c).unwrap_err().contains("must map every episode source window exactly once"));
    }

    #[test]
    fn rejects_duplicate_source_trace_digest_between_executions() {
        let (p,mut e,c,_,_,_)=fixture();
        e["records"][1]["source_trace_sha256"]=e["records"][0]["source_trace_sha256"].clone();
        assert!(audit(&p,&e,&c).unwrap_err().contains("is reused across episodes"));
    }

    #[test]
    fn rejects_randomizer_artifact_digest_mismatch() {
        let (p,mut e,c,_,_,_)=fixture();
        e["randomizer_source_sha256"]=json!("f".repeat(64));
        assert!(audit(&p,&e,&c).unwrap_err().contains("episodes.randomizer_source_sha256"));
    }

    #[test]
    fn late_change_restarted_sum_is_faster_than_fixed_start_cmm_in_episode_units() {
        let family=vec![
            BettingTerm{theta:0.15,prior_weight:0.25},
            BettingTerm{theta:0.30,prior_weight:0.25},
            BettingTerm{theta:0.50,prior_weight:0.25},
            BettingTerm{theta:0.75,prior_weight:0.25},
        ];
        let mut rng=123456_u64;
        let mut values=Vec::with_capacity(1200);
        for _ in 0..1000 {
            rng=(1664525_u64*rng+1013904223_u64)%4_294_967_296_u64;
            values.push(rng as f64/4_294_967_296.0);
        }
        values.extend((0..200).map(|i|2.0+i as f64*0.01));
        let mut random_rng=98765_u64;
        let mut state=ContextState::new(0.025,&family);
        let mut classic_alarm=None;
        let mut restarted_alarm=None;
        let mut hist=Vec::new();
        let mut classic_logs=vec![0.0_f64;family.len()];
        for (i,score) in values.iter().enumerate() {
            random_rng=(1664525_u64*random_rng+1013904223_u64)%4_294_967_296_u64;
            let u=(random_rng as f64/4_294_967_296.0).clamp(0.0000001,0.9999999);
            let transformed=-*score;
            let p=online_randomized_conformal_p(&hist,*score,u).unwrap();
            hist.push(transformed);
            for (j,term) in family.iter().enumerate() {
                classic_logs[j]+=term.theta.ln()+(term.theta-1.0)*p.ln();
            }
            let fixed_components:Vec<f64>=family.iter().enumerate()
                .map(|(j,term)|term.prior_weight.ln()+classic_logs[j]).collect();
            let fixed=log_sum_exp(&fixed_components).unwrap();
            if classic_alarm.is_none() && fixed>=-(0.025_f64).ln() {
                classic_alarm=Some(i+1);
            }
            // This independent reference loop tests the same recurrence update
            // used by the production context processor.
            let (_p,_restarted,_fixed,_threshold,restarted_crossing,_fixed_crossing)=
                state.update(*score,u,i+1,&family).unwrap();
            if restarted_alarm.is_none() && restarted_crossing {
                restarted_alarm=Some(i+1);
            }
        }
        assert_eq!(restarted_alarm,Some(1004));
        assert_eq!(classic_alarm,Some(1025));
        assert!(restarted_alarm.unwrap()-1000 < classic_alarm.unwrap()-1000);
    }

    #[test]
    fn weight_schedule_is_summable_and_recent_starts_remain_nonzero() {
        let sum:f64=(1..1_000_000).map(near_harmonic_restart_weight).sum();
        assert!(sum<1.0);
        assert!(near_harmonic_restart_weight(100_000)>0.0);
    }
}
