//! HARDSEC-019 fixed-reference context-conditioned sequential monitor.
//!
//! Computes context-specific upper-tail empirical p-values against immutable
//! null references and accumulates a predeclared mixture of decreasing
//! p-value betting functions in log space. Invalid or unqualified telemetry
//! abstains and does not update evidence. An unseen context also abstains.
//!
//! Crucial claim ceiling: the e-process false-alarm guarantee requires each
//! p-value to be conditionally super-uniform given the complete past and
//! declared context. This executable does not establish that assumption for
//! temporally dependent PMU streams. The output explicitly reports this
//! guarantee as not established from the supplied inputs.
//!
//! Usage:
//!   hardsec_context_sequential_monitor --plan plan.json \
//!     --stream stream.json --receipt receipt.json

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PLAN_SCHEMA: &str = "hardsec-context-sequential-plan-v1";
const STREAM_SCHEMA: &str = "hardsec-context-sequential-stream-v1";
const RECEIPT_SCHEMA: &str = "hardsec-context-sequential-monitor-receipt-v1";
const QUALIFIED_PMU: &str = "qualified_for_named_acquisition_conditions";

#[derive(Debug, Clone)]
struct BettingTerm {
    lambda: f64,
    weight: f64,
}

#[derive(Debug, Clone)]
struct ContextReference {
    reference_id: String,
    reference_group_ids: Vec<String>,
    scores: Vec<f64>,
    digest: String,
}

#[derive(Debug, Clone)]
struct Plan {
    monitor_id: String,
    evidence_domain: String,
    stream_id: String,
    evaluation_group_id: String,
    alpha_budget: f64,
    minimum_reference_scores: usize,
    betting_mixture: Vec<BettingTerm>,
    contexts: BTreeMap<String, ContextReference>,
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
        return Err("unsupported plan schema_version".to_owned());
    }
    let monitor_id = string_field(value, "monitor_id", "plan")?.to_owned();
    let evidence_domain = string_field(value, "evidence_domain", "plan")?.to_owned();
    if !matches!(evidence_domain.as_str(), "synthetic_fixture" | "simulator" | "physical_hardware") {
        return Err(format!("unsupported evidence_domain {evidence_domain}"));
    }
    let stream_id = string_field(value, "stream_id", "plan")?.to_owned();
    let declared_evaluation_group = string_field(value, "evaluation_group_id", "plan")?;
    if declared_evaluation_group != evaluation_group_id {
        return Err("stream evaluation_group_id does not match the preregistered plan".to_owned());
    }
    if string_field(value, "reference_update_policy", "plan")? != "immutable_no_test_time_updates" {
        return Err("reference_update_policy must be immutable_no_test_time_updates".to_owned());
    }
    if string_field(value, "pmu_qualification_required", "plan")? != QUALIFIED_PMU {
        return Err("unsupported pmu_qualification_required status".to_owned());
    }
    if string_field(value, "conditional_superuniformity_status", "plan")? != "assumption_not_verified" {
        return Err("v1 must explicitly mark conditional super-uniformity as assumption_not_verified".to_owned());
    }

    let alpha_budget = f64_field(value, "alpha_budget", "plan")?;
    if !(0.0..1.0).contains(&alpha_budget) {
        return Err("alpha_budget must be strictly between zero and one".to_owned());
    }
    let minimum_reference_scores = usize_field(value, "minimum_reference_scores", "plan")?;
    if minimum_reference_scores < 2 {
        return Err("minimum_reference_scores must be at least two".to_owned());
    }

    let betting_values = value.get("betting_mixture").and_then(Value::as_array)
        .ok_or("plan.betting_mixture must be an array")?;
    if betting_values.is_empty() {
        return Err("plan.betting_mixture must not be empty".to_owned());
    }
    let mut betting_mixture = Vec::with_capacity(betting_values.len());
    let mut weight_sum = 0.0;
    for (index, term) in betting_values.iter().enumerate() {
        let context = format!("plan.betting_mixture[{index}]");
        let lambda = f64_field(term, "lambda", &context)?;
        let weight = f64_field(term, "weight", &context)?;
        if !(0.0..1.0).contains(&lambda) || !(0.0..1.0).contains(&weight) {
            return Err(format!("{context}: lambda and weight must be in (0,1)"));
        }
        weight_sum += weight;
        betting_mixture.push(BettingTerm { lambda, weight });
    }
    if (weight_sum - 1.0).abs() > 1e-12 {
        return Err(format!("betting mixture weights must sum to one (observed {weight_sum})"));
    }

    let contexts_value = value.get("contexts").and_then(Value::as_array)
        .ok_or("plan.contexts must be an array")?;
    if contexts_value.is_empty() {
        return Err("plan.contexts must not be empty".to_owned());
    }
    let mut contexts = BTreeMap::new();
    let mut all_reference_groups = BTreeSet::new();
    for (index, item) in contexts_value.iter().enumerate() {
        let context = format!("plan.contexts[{index}]");
        let context_id = string_field(item, "context_id", &context)?.to_owned();
        let reference_id = string_field(item, "reference_id", &context)?.to_owned();
        let reference_group_ids = string_array(item, "reference_group_ids", &context)?;
        if reference_group_ids.is_empty() {
            return Err(format!("{context}.reference_group_ids must not be empty"));
        }
        for group_id in &reference_group_ids {
            if group_id == evaluation_group_id {
                return Err(format!("evaluation group {evaluation_group_id} overlaps a null-reference group"));
            }
            if !all_reference_groups.insert(group_id.clone()) {
                return Err(format!("reference group {group_id} is reused across context references"));
            }
        }
        let score_values = item.get("reference_scores").and_then(Value::as_array)
            .ok_or_else(|| format!("{context}.reference_scores must be an array"))?;
        if score_values.len() < minimum_reference_scores {
            return Err(format!("{context} has {} reference scores; at least {minimum_reference_scores} are required", score_values.len()));
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
        let encoded = serde_json::to_vec(&scores).map_err(|e| format!("cannot encode reference scores: {e}"))?;
        let reference = ContextReference {
            reference_id,
            reference_group_ids,
            scores,
            digest: sha256(&encoded),
        };
        if contexts.insert(context_id.clone(), reference).is_some() {
            return Err(format!("duplicate context_id {context_id}"));
        }
    }

    Ok(Plan {
        monitor_id,
        evidence_domain,
        stream_id,
        evaluation_group_id: evaluation_group_id.to_owned(),
        alpha_budget,
        minimum_reference_scores,
        betting_mixture,
        contexts,
    })
}

fn parse_stream(value: &Value, plan: &Plan) -> Result<Vec<StreamRecord>, String> {
    if string_field(value, "schema_version", "stream")? != STREAM_SCHEMA {
        return Err("unsupported stream schema_version".to_owned());
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

// Conservative empirical upper-tail rank. Ties are counted in the tail:
// p = (1 + #{reference score >= observation score}) / (n + 1).
fn empirical_upper_tail_p_value(reference_scores: &[f64], score: f64) -> f64 {
    let tail_count = reference_scores.iter().filter(|reference| **reference >= score).count();
    (1.0 + tail_count as f64) / (reference_scores.len() as f64 + 1.0)
}

// log(sum_j w_j * lambda_j * p^(lambda_j - 1)), evaluated via log-sum-exp.
// For p uniform on [0,1], each component has expectation one. The mixture is
// decreasing in p and has expectation <= 1 for super-uniform p-values.
fn log_mixture_bet(p_value: f64, mixture: &[BettingTerm]) -> Result<f64, String> {
    if !(0.0..=1.0).contains(&p_value) || p_value == 0.0 {
        return Err("p_value must be in (0,1]".to_owned());
    }
    let log_p = p_value.ln();
    let terms: Vec<f64> = mixture.iter()
        .map(|term| term.weight.ln() + term.lambda.ln() + (term.lambda - 1.0) * log_p)
        .collect();
    let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = terms.iter().map(|value| (value - max).exp()).sum();
    let result = max + sum_exp.ln();
    if !result.is_finite() {
        return Err("mixture betting function produced a non-finite log value".to_owned());
    }
    Ok(result)
}

fn abstention_reason(record: &StreamRecord, plan: &Plan) -> Option<&'static str> {
    if record.sensor_state != "valid" {
        return Some("invalid_sensor_state");
    }
    if record.pmu_qualification_result != QUALIFIED_PMU {
        return Some("pmu_acquisition_not_qualified");
    }
    if !plan.contexts.contains_key(&record.context_id) {
        return Some("unseen_workload_context");
    }
    None
}

fn audit_values(plan_value: &Value, stream_value: &Value, plan_bytes: &[u8], stream_bytes: &[u8]) -> Result<Value, String> {
    let evaluation_group_id = string_field(stream_value, "evaluation_group_id", "stream")?;
    let plan = parse_plan(plan_value, evaluation_group_id)?;
    let records = parse_stream(stream_value, &plan)?;
    let log_threshold = -plan.alpha_budget.ln();

    let mut log_wealth = 0.0_f64;
    let mut first_alarm_index: Option<usize> = None;
    let mut output_records = Vec::with_capacity(records.len());
    let mut abstention_counts: BTreeMap<String, usize> = BTreeMap::new();

    for record in &records {
        if let Some(reason) = abstention_reason(record, &plan) {
            *abstention_counts.entry(reason.to_owned()).or_default() += 1;
            output_records.push(json!({
                "sequence_index":record.sequence_index,
                "window_id":record.window_id,
                "context_id":record.context_id,
                "action":if first_alarm_index.is_some() {"alert_latched_observation_abstained"} else {"abstain"},
                "abstention_reason":reason,
                "p_value":null,
                "log_e_value":null,
                "log_wealth":log_wealth,
                "threshold_log_wealth":log_threshold
            }));
            continue;
        }

        let reference = plan.contexts.get(&record.context_id)
            .ok_or("context validation invariant failed")?;
        let p_value = empirical_upper_tail_p_value(&reference.scores, record.anomaly_score);
        let log_e_value = log_mixture_bet(p_value, &plan.betting_mixture)?;
        log_wealth += log_e_value;
        if !log_wealth.is_finite() {
            return Err("cumulative log wealth became non-finite".to_owned());
        }

        let crossing_now = first_alarm_index.is_none() && log_wealth >= log_threshold;
        if crossing_now {
            first_alarm_index = Some(record.sequence_index);
        }
        let action = if crossing_now {
            "alarm_crossing"
        } else if first_alarm_index.is_some() {
            "alert_latched"
        } else {
            "monitoring_no_alarm_yet"
        };

        output_records.push(json!({
            "sequence_index":record.sequence_index,
            "window_id":record.window_id,
            "context_id":record.context_id,
            "reference_id":reference.reference_id,
            "reference_score_count":reference.scores.len(),
            "reference_digest":reference.digest,
            "action":action,
            "abstention_reason":null,
            "p_value":p_value,
            "log_e_value":log_e_value,
            "log_wealth":log_wealth,
            "threshold_log_wealth":log_threshold
        }));
    }

    let reference_reports: Vec<Value> = plan.contexts.iter().map(|(context_id, reference)| json!({
        "context_id":context_id,
        "reference_id":reference.reference_id,
        "reference_digest":reference.digest,
        "reference_score_count":reference.scores.len(),
        "declared_reference_group_count":reference.reference_group_ids.len(),
        "reference_group_ids":reference.reference_group_ids
    })).collect();

    let result = if first_alarm_index.is_some() {
        "alarm_threshold_crossed"
    } else {
        "no_alarm_threshold_crossed"
    };

    Ok(json!({
        "schema_version":RECEIPT_SCHEMA,
        "result":result,
        "claim_scope":"fixed_reference_sequential_monitor_mechanics_only",
        "monitor_id":plan.monitor_id,
        "stream_id":plan.stream_id,
        "evidence_domain":plan.evidence_domain,
        "evaluation_group_id":plan.evaluation_group_id,
        "inputs":{
            "plan_sha256":sha256(plan_bytes),
            "stream_sha256":sha256(stream_bytes)
        },
        "reference_policy":{
            "update_policy":"immutable_no_test_time_updates",
            "reference_context_count":plan.contexts.len(),
            "references":reference_reports,
            "test_time_reference_updates":0,
            "evaluation_group_separated_from_reference_groups":true
        },
        "sequential_monitor":{
            "alpha_budget":plan.alpha_budget,
            "log_wealth_threshold":log_threshold,
            "final_log_wealth":log_wealth,
            "first_alarm_sequence_index":first_alarm_index,
            "valid_observation_count":records.len() - abstention_counts.values().sum::<usize>(),
            "abstention_count":abstention_counts.values().sum::<usize>(),
            "abstention_counts_by_reason":abstention_counts,
            "post_alarm_reset_count":0,
            "alarm_is_latched":first_alarm_index.is_some(),
            "window_decisions":output_records
        },
        "betting_function":{
            "type":"fixed_mixture_of_power_p_value_bets",
            "mixture":plan.betting_mixture.iter().map(|term| json!({"lambda":term.lambda,"weight":term.weight})).collect::<Vec<_>>(),
            "formula":"sum_j weight_j * lambda_j * p^(lambda_j - 1)",
            "conditional_expectation_requirement":"each p_t must be super-uniform given the complete past and declared context",
            "e_process_statement":"under conditional super-uniformity, the product process is a non-negative supermartingale and crossing 1/alpha has Ville bound alpha"
        },
        "statistical_guarantees":{
            "conditional_superuniformity_status":"assumption_not_verified",
            "sequential_false_alarm_guarantee_status":"not_established_from_inputs",
            "ordinary_exchangeable_reference_ranks_suffice_under_arbitrary_temporal_dependence":false,
            "effective_sample_size_estimate":null,
            "independence_status":"not_established"
        },
        "limitations":[
            "The empirical upper-tail p-values are rank-based against a fixed context reference; their marginal exchangeability validity does not establish conditional super-uniformity given the complete past of a temporally dependent PMU stream.",
            "The anytime-valid false-alarm statement is conditional on a filtration-conditional super-uniformity assumption that this executable does not verify.",
            "Calibration reference group IDs and evaluation group IDs are caller-declared; digest binding does not authenticate their identity or independence.",
            "This implementation is a fixed-reference e-process candidate, not a reproduction of the 2026 conditional conformal test martingale construction.",
            "The receipt does not use attack labels, establish attack-effect detection, estimate false-alert rates, compare SOTA, prove platform transfer or certify hardware security."
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
    let mut stream = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let target = match flag.as_str() {
            "--plan" => &mut plan,
            "--stream" => &mut stream,
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
    validate_receipt_location(&args.receipt, &[args.plan.as_path(), args.stream.as_path()])?;
    let receipt = audit_values(&plan, &stream, &plan_bytes, &stream_bytes)?;
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
            eprintln!("HARDSEC-019 sequential monitor failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Value, Value, Vec<u8>, Vec<u8>) {
        let plan_bytes = include_bytes!("../tests/fixtures/hardsec/context-sequential-v1/plan.json").to_vec();
        let stream_bytes = include_bytes!("../tests/fixtures/hardsec/context-sequential-v1/stream.json").to_vec();
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
    fn known_answer_crosses_boundary_and_preserves_claim_ceiling() {
        let (plan, stream, pb, sb) = fixture();
        let receipt = audit_values(&plan, &stream, &pb, &sb).unwrap();
        assert_eq!(receipt["result"], "alarm_threshold_crossed");
        assert_eq!(receipt["sequential_monitor"]["first_alarm_sequence_index"], 4);
        assert_eq!(receipt["sequential_monitor"]["valid_observation_count"], 4);
        assert_eq!(receipt["sequential_monitor"]["abstention_count"], 3);
        assert_eq!(receipt["sequential_monitor"]["post_alarm_reset_count"], 0);
        assert_eq!(receipt["statistical_guarantees"]["conditional_superuniformity_status"], "assumption_not_verified");
        assert_eq!(receipt["statistical_guarantees"]["sequential_false_alarm_guarantee_status"], "not_established_from_inputs");
        assert_eq!(receipt["statistical_guarantees"]["effective_sample_size_estimate"], Value::Null);
    }

    #[test]
    fn unseen_context_abstains_without_wealth_update() {
        let (p, mut s, _, _) = fixture();
        s["records"] = json!([{
            "sequence_index":1,"window_id":"unknown-context-window","context_id":"new-regime",
            "anomaly_score":99.0,"sensor_state":"valid","pmu_qualification_result":QUALIFIED_PMU
        }]);
        let r = audit(&p, &s).unwrap();
        assert_eq!(r["result"], "no_alarm_threshold_crossed");
        assert_eq!(r["sequential_monitor"]["final_log_wealth"], 0.0);
        assert_eq!(r["sequential_monitor"]["abstention_counts_by_reason"]["unseen_workload_context"], 1);
    }

    #[test]
    fn stale_and_untrusted_sensor_records_do_not_update_wealth() {
        let (p, mut s, _, _) = fixture();
        s["records"] = json!([
            {"sequence_index":1,"window_id":"stale","context_id":"compute","anomaly_score":99.0,"sensor_state":"stale","pmu_qualification_result":QUALIFIED_PMU},
            {"sequence_index":2,"window_id":"untrusted","context_id":"compute","anomaly_score":99.0,"sensor_state":"untrusted","pmu_qualification_result":QUALIFIED_PMU}
        ]);
        let r = audit(&p, &s).unwrap();
        assert_eq!(r["sequential_monitor"]["final_log_wealth"], 0.0);
        assert_eq!(r["sequential_monitor"]["abstention_count"], 2);
    }

    #[test]
    fn unqualified_pmu_record_abstains_even_when_sensor_state_is_valid() {
        let (p, mut s, _, _) = fixture();
        s["records"] = json!([{
            "sequence_index":1,"window_id":"unqualified","context_id":"compute",
            "anomaly_score":99.0,"sensor_state":"valid","pmu_qualification_result":"measurement_quality_gate_failed"
        }]);
        let r = audit(&p, &s).unwrap();
        assert_eq!(r["sequential_monitor"]["final_log_wealth"], 0.0);
        assert_eq!(r["sequential_monitor"]["abstention_counts_by_reason"]["pmu_acquisition_not_qualified"], 1);
    }

    #[test]
    fn reference_and_evaluation_group_overlap_is_rejected() {
        let (mut p, s, _, _) = fixture();
        p["contexts"][0]["reference_group_ids"] = json!(["heldout-execution-01"]);
        assert!(audit(&p, &s).unwrap_err().contains("overlaps a null-reference group"));
    }

    #[test]
    fn duplicate_window_identity_is_rejected() {
        let (p, mut s, _, _) = fixture();
        s["records"][1]["window_id"] = s["records"][0]["window_id"].clone();
        assert!(audit(&p, &s).unwrap_err().contains("duplicate window_id"));
    }

    #[test]
    fn non_contiguous_sequence_index_is_rejected() {
        let (p, mut s, _, _) = fixture();
        s["records"][1]["sequence_index"] = json!(4);
        assert!(audit(&p, &s).unwrap_err().contains("must be contiguous from one"));
    }

    #[test]
    fn malformed_betting_weights_are_rejected() {
        let (mut p, s, _, _) = fixture();
        p["betting_mixture"][0]["weight"] = json!(0.1);
        assert!(audit(&p, &s).unwrap_err().contains("weights must sum to one"));
    }

    #[test]
    fn insufficient_reference_support_is_rejected() {
        let (mut p, s, _, _) = fixture();
        p["contexts"][0]["reference_scores"] = json!([0.1,0.2,0.3]);
        assert!(audit(&p, &s).unwrap_err().contains("reference scores; at least"));
    }

    #[test]
    fn context_specific_tail_rank_is_conservative_on_ties() {
        let (mut p, mut s, _, _) = fixture();
        p["contexts"][0]["reference_scores"] = json!([1.0,1.0,2.0]);
        p["minimum_reference_scores"] = json!(2);
        s["records"] = json!([{
            "sequence_index":1,"window_id":"tie-window","context_id":"compute",
            "anomaly_score":1.0,"sensor_state":"valid","pmu_qualification_result":QUALIFIED_PMU
        }]);
        let r = audit(&p, &s).unwrap();
        assert_eq!(r["sequential_monitor"]["window_decisions"][0]["p_value"], 1.0);
    }

    #[test]
    fn fixed_reference_digest_is_stable_and_test_time_updates_are_zero() {
        let (p, s, pb, sb) = fixture();
        let first = audit_values(&p, &s, &pb, &sb).unwrap();
        let second = audit_values(&p, &s, &pb, &sb).unwrap();
        assert_eq!(first["reference_policy"]["references"], second["reference_policy"]["references"]);
        assert_eq!(first["reference_policy"]["test_time_reference_updates"], 0);
    }

    #[test]
    fn wrong_evidence_domain_is_rejected() {
        let (p, mut s, _, _) = fixture();
        s["evidence_domain"] = json!("physical_hardware");
        assert!(audit(&p, &s).unwrap_err().contains("evidence_domain must match"));
    }

    #[test]
    fn invalid_sensor_state_is_rejected_not_silently_accepted() {
        let (p, mut s, _, _) = fixture();
        s["records"][0]["sensor_state"] = json!("sort-of-valid");
        assert!(audit(&p, &s).unwrap_err().contains("unsupported state"));
    }
}
