//! HARDSEC-012 adaptive-evasion campaign evaluator.
//!
//! Detector predictions, evaluator case metadata, and independently verified
//! attack-effect truth are supplied as three separate files. The evaluator
//! joins them by case_id, validates declared group holdouts, and computes
//! attack-effect-conditioned recall, benign false-alert burden, sensor
//! integrity, adaptation-stratum metrics, and measured acquisition/inference
//! cost. It does not generate attacks or validate the authenticity of oracles.
//!
//! Usage:
//!   hardsec_adaptive_evasion_audit --predictions predictions.json \
//!       --manifest evaluator-manifest.json --truth evaluator-truth.json \
//!       --receipt receipt.json

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const CLAIM_SCOPE: &str = "conditional_campaign_metrics_for_named_domain_and_declared_holdouts";

#[derive(Debug, Clone)]
struct Prediction {
    case_id: String,
    response: String,
    score: Option<f64>,
    acquisition_ns: Option<u64>,
    inference_ns: Option<u64>,
    telemetry_bytes: Option<u64>,
    peak_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
struct CaseMeta {
    case_id: String,
    split: String,
    program_group_id: String,
    execution_group_id: String,
    device_config_id: String,
    attack_family_id: String,
    adaptation_family_id: String,
    adaptation_level: u64,
    sensor_state: String,
}

#[derive(Debug, Clone)]
struct Truth {
    case_id: String,
    class: String,
    effect: String,
}

#[derive(Debug, Clone, Default)]
struct RecallCounts {
    cases: usize,
    detected: usize,
    cleared: usize,
    abstained: usize,
}

#[derive(Debug, Clone)]
struct Args {
    predictions: PathBuf,
    manifest: PathBuf,
    truth: PathBuf,
    receipt: PathBuf,
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn nonempty_id(value: &str, context: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(format!("{context} must be non-empty and contain no control characters"));
    }
    Ok(())
}

fn str_field<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, String> {
    let result = value.get(key).and_then(Value::as_str)
        .ok_or_else(|| format!("{context}.{key} must be a string"))?;
    nonempty_id(result, &format!("{context}.{key}"))?;
    Ok(result)
}

fn optional_u64(value: &Value, key: &str, context: &str) -> Result<Option<u64>, String> {
    let field = value.get(key).ok_or_else(|| format!("{context}.{key} is required"))?;
    if field.is_null() {
        return Ok(None);
    }
    field.as_u64().map(Some)
        .ok_or_else(|| format!("{context}.{key} must be a non-negative integer or null"))
}

fn validate_sha(value: &str, context: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(format!("{context} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn digest_field(value: &Value, key: &str, context: &str) -> Result<(), String> {
    let digest = str_field(value, key, context)?;
    validate_sha(digest, &format!("{context}.{key}"))
}

fn parse_predictions(value: &Value) -> Result<(String, String, String, Vec<Prediction>), String> {
    if str_field(value, "schema_version", "predictions")? != "hardsec-adaptive-predictions-v1" {
        return Err("unsupported predictions schema_version".to_owned());
    }
    let campaign_id = str_field(value, "campaign_id", "predictions")?.to_owned();
    let model_id = str_field(value, "model_id", "predictions")?.to_owned();
    digest_field(value, "model_sha256", "predictions")?;
    digest_field(value, "threshold_profile_sha256", "predictions")?;
    let values = value.get("cases").and_then(Value::as_array)
        .ok_or_else(|| "predictions.cases must be an array".to_owned())?;
    if values.is_empty() {
        return Err("predictions.cases must not be empty".to_owned());
    }
    let mut seen = BTreeSet::new();
    let mut output = Vec::with_capacity(values.len());
    for (index, item) in values.iter().enumerate() {
        let ctx = format!("predictions.cases[{index}]");
        let case_id = str_field(item, "case_id", &ctx)?.to_owned();
        if !seen.insert(case_id.clone()) {
            return Err(format!("duplicate prediction case_id {case_id}"));
        }
        let response = str_field(item, "response", &ctx)?.to_owned();
        if !matches!(response.as_str(), "alert" | "clear" | "abstain" | "sensor_invalid") {
            return Err(format!("{ctx}.response has unsupported value {response}"));
        }
        let score_value = item.get("score").ok_or_else(|| format!("{ctx}.score is required"))?;
        let score = if score_value.is_null() {
            None
        } else {
            let score = score_value.as_f64().ok_or_else(|| format!("{ctx}.score must be a number or null"))?;
            if !(0.0..=1.0).contains(&score) || !score.is_finite() {
                return Err(format!("{ctx}.score must be finite and within [0, 1]"));
            }
            Some(score)
        };
        let cost = item.get("cost").ok_or_else(|| format!("{ctx}.cost is required"))?;
        output.push(Prediction {
            case_id,
            response,
            score,
            acquisition_ns: optional_u64(cost, "acquisition_ns", &format!("{ctx}.cost"))?,
            inference_ns: optional_u64(cost, "inference_ns", &format!("{ctx}.cost"))?,
            telemetry_bytes: optional_u64(cost, "telemetry_bytes", &format!("{ctx}.cost"))?,
            peak_memory_bytes: optional_u64(cost, "peak_memory_bytes", &format!("{ctx}.cost"))?,
        });
    }
    let model_sha = str_field(value, "model_sha256", "predictions")?.to_owned();
    Ok((campaign_id, model_id, model_sha, output))
}

fn axis_value<'a>(case: &'a CaseMeta, axis: &str) -> &'a str {
    match axis {
        "program" => &case.program_group_id,
        "execution" => &case.execution_group_id,
        "device_config" => &case.device_config_id,
        "attack_family" => &case.attack_family_id,
        "adaptation_family" => &case.adaptation_family_id,
        _ => unreachable!("holdout axis validated before lookup"),
    }
}

fn parse_manifest(value: &Value) -> Result<(String, String, Vec<String>, Vec<CaseMeta>), String> {
    if str_field(value, "schema_version", "manifest")? != "hardsec-adaptive-case-manifest-v1" {
        return Err("unsupported manifest schema_version".to_owned());
    }
    let campaign_id = str_field(value, "campaign_id", "manifest")?.to_owned();
    let domain = str_field(value, "evidence_domain", "manifest")?.to_owned();
    if !matches!(domain.as_str(), "synthetic_fixture" | "simulator" | "physical_hardware") {
        return Err(format!("unsupported evidence_domain {domain}"));
    }
    let platform_id = str_field(value, "platform_id", "manifest")?.to_owned();
    let axes_values = value.get("holdout_axes").and_then(Value::as_array)
        .ok_or_else(|| "manifest.holdout_axes must be an array".to_owned())?;
    let allowed_axes = ["program", "execution", "device_config", "attack_family", "adaptation_family"];
    let mut axes = BTreeSet::new();
    for item in axes_values {
        let axis = item.as_str().ok_or("manifest.holdout_axes entries must be strings")?;
        if !allowed_axes.contains(&axis) {
            return Err(format!("unsupported holdout axis {axis}"));
        }
        if !axes.insert(axis.to_owned()) {
            return Err(format!("duplicate holdout axis {axis}"));
        }
    }
    for required in ["program", "execution", "attack_family", "adaptation_family"] {
        if !axes.contains(required) {
            return Err(format!("manifest.holdout_axes must include {required}"));
        }
    }
    let values = value.get("cases").and_then(Value::as_array)
        .ok_or_else(|| "manifest.cases must be an array".to_owned())?;
    let mut seen = BTreeSet::new();
    let mut output = Vec::with_capacity(values.len());
    for (index, item) in values.iter().enumerate() {
        let ctx = format!("manifest.cases[{index}]");
        let case_id = str_field(item, "case_id", &ctx)?.to_owned();
        if !seen.insert(case_id.clone()) {
            return Err(format!("duplicate manifest case_id {case_id}"));
        }
        let split = str_field(item, "split", &ctx)?.to_owned();
        if !matches!(split.as_str(), "train" | "validation" | "test") {
            return Err(format!("{ctx}.split must be train, validation or test"));
        }
        let sensor_state = str_field(item, "sensor_state", &ctx)?.to_owned();
        if !matches!(sensor_state.as_str(), "valid" | "missing" | "stale" | "misaligned" | "multiplexed" | "untrusted") {
            return Err(format!("{ctx}.sensor_state is unsupported"));
        }
        let adaptation_level = item.get("adaptation_level").and_then(Value::as_u64)
            .ok_or_else(|| format!("{ctx}.adaptation_level must be a non-negative integer"))?;
        let meta = CaseMeta {
            case_id,
            split,
            program_group_id: str_field(item, "program_group_id", &ctx)?.to_owned(),
            execution_group_id: str_field(item, "execution_group_id", &ctx)?.to_owned(),
            device_config_id: str_field(item, "device_config_id", &ctx)?.to_owned(),
            attack_family_id: str_field(item, "attack_family_id", &ctx)?.to_owned(),
            adaptation_family_id: str_field(item, "adaptation_family_id", &ctx)?.to_owned(),
            adaptation_level,
            sensor_state,
        };
        if meta.adaptation_family_id == "none" && meta.adaptation_level != 0 {
            return Err(format!("{ctx}: adaptation_family_id=none requires adaptation_level=0"));
        }
        output.push(meta);
    }
    for split in ["train", "validation", "test"] {
        if !output.iter().any(|case| case.split == split) {
            return Err(format!("manifest must contain at least one {split} case"));
        }
    }
    Ok((campaign_id, domain, axes.into_iter().collect(), output))
}

fn parse_truth(value: &Value) -> Result<(String, BTreeMap<String, Truth>), String> {
    if str_field(value, "schema_version", "truth")? != "hardsec-adaptive-oracle-v1" {
        return Err("unsupported truth schema_version".to_owned());
    }
    let campaign_id = str_field(value, "campaign_id", "truth")?.to_owned();
    let values = value.get("cases").and_then(Value::as_array)
        .ok_or_else(|| "truth.cases must be an array".to_owned())?;
    let mut output = BTreeMap::new();
    for (index, item) in values.iter().enumerate() {
        let ctx = format!("truth.cases[{index}]");
        let case_id = str_field(item, "case_id", &ctx)?.to_owned();
        let class = str_field(item, "class", &ctx)?.to_owned();
        let effect = str_field(item, "attack_effect", &ctx)?.to_owned();
        digest_field(item, "outcome_evidence_sha256", &ctx)?;
        if !matches!(class.as_str(), "malicious" | "benign") {
            return Err(format!("{ctx}.class must be malicious or benign"));
        }
        if !matches!(effect.as_str(), "observed" | "not_observed" | "unknown" | "not_applicable") {
            return Err(format!("{ctx}.attack_effect is unsupported"));
        }
        if (class == "benign" && effect != "not_applicable")
            || (class == "malicious" && effect == "not_applicable") {
            return Err(format!("{ctx}: class and attack_effect semantics conflict"));
        }
        let truth = Truth { case_id: case_id.clone(), class, effect };
        if output.insert(case_id.clone(), truth).is_some() {
            return Err(format!("duplicate truth case_id {case_id}"));
        }
    }
    if output.is_empty() {
        return Err("truth.cases must not be empty".to_owned());
    }
    Ok((campaign_id, output))
}

fn add_recall(stats: &mut RecallCounts, response: &str) {
    stats.cases += 1;
    match response {
        "alert" => stats.detected += 1,
        "clear" => stats.cleared += 1,
        "abstain" | "sensor_invalid" => stats.abstained += 1,
        _ => unreachable!("response validated before aggregation"),
    }
}

fn recall(stats: &RecallCounts) -> Option<f64> {
    if stats.cases == 0 {
        None
    } else {
        Some(stats.detected as f64 / stats.cases as f64)
    }
}

fn macro_recall(groups: &BTreeMap<String, RecallCounts>) -> (Option<f64>, Option<f64>, usize) {
    let values: Vec<f64> = groups.values().filter_map(recall).collect();
    if values.is_empty() {
        return (None, None, 0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let worst = values.iter().copied().fold(f64::INFINITY, f64::min);
    (Some(mean), Some(worst), values.len())
}

fn percentile(sorted: &[u64], q: f64) -> Option<u64> {
    if sorted.is_empty() {
        return None;
    }
    let idx = ((sorted.len() - 1) as f64 * q).ceil() as usize;
    sorted.get(idx).copied()
}

fn audit_values(
    predictions: &Value,
    manifest: &Value,
    truth_value: &Value,
    predictions_bytes: &[u8],
    manifest_bytes: &[u8],
    truth_bytes: &[u8],
) -> Result<Value, String> {
    let (prediction_campaign, model_id, model_sha, prediction_rows) = parse_predictions(predictions)?;
    let (manifest_campaign, evidence_domain, holdout_axes, manifest_rows) = parse_manifest(manifest)?;
    let (truth_campaign, truth_rows) = parse_truth(truth_value)?;
    if prediction_campaign != manifest_campaign || prediction_campaign != truth_campaign {
        return Err("predictions, manifest and truth campaign_id values must match".to_owned());
    }

    let mut metadata = BTreeMap::new();
    for row in manifest_rows {
        metadata.insert(row.case_id.clone(), row);
    }
    let mut prediction_map = BTreeMap::new();
    for row in prediction_rows {
        prediction_map.insert(row.case_id.clone(), row);
    }
    let test_ids: BTreeSet<String> = metadata.values()
        .filter(|case| case.split == "test")
        .map(|case| case.case_id.clone()).collect();
    let prediction_ids: BTreeSet<String> = prediction_map.keys().cloned().collect();
    let truth_ids: BTreeSet<String> = truth_rows.keys().cloned().collect();
    if test_ids != prediction_ids || test_ids != truth_ids {
        return Err("predictions and truth must contain exactly the manifest's test case IDs".to_owned());
    }

    // Check that the declared training/validation/test holdout axes are disjoint.
    let mut holdout_reports = Vec::new();
    for axis in &holdout_axes {
        let mut split_by_value: BTreeMap<String, String> = BTreeMap::new();
        for case in metadata.values() {
            let group_value = axis_value(case, axis);
            if axis == "adaptation_family" && group_value == "none" {
                continue;
            }
            if let Some(previous) = split_by_value.insert(group_value.to_owned(), case.split.clone()) {
                if previous != case.split {
                    return Err(format!("declared {axis} holdout group {group_value} crosses {previous} and {}", case.split));
                }
            }
        }
        holdout_reports.push(json!({
            "axis": axis,
            "unique_group_count": split_by_value.len(),
            "disjoint_across_train_validation_test": true
        }));
    }

    let mut success_attack_cases = 0usize;
    let mut successful_attack_alerts = 0usize;
    let mut successful_attack_clears = 0usize;
    let mut successful_attack_abstentions = 0usize;
    let mut failed_attack_attempts = 0usize;
    let mut failed_attack_alerts = 0usize;
    let mut unknown_effect_attempts = 0usize;
    let mut unknown_effect_alerts = 0usize;
    let mut benign_cases = 0usize;
    let mut benign_alerts = 0usize;
    let mut benign_clears = 0usize;
    let mut benign_abstentions = 0usize;
    let mut valid_benign_cases = 0usize;
    let mut valid_benign_clears = 0usize;
    let mut invalid_benign_cases = 0usize;
    let mut all_alerts = 0usize;
    let mut all_abstentions = 0usize;
    let mut invalid_sensor_cases = 0usize;
    let mut invalid_sensor_clear_verdicts = 0usize;

    let mut adaptation_groups: BTreeMap<String, RecallCounts> = BTreeMap::new();
    let mut program_groups: BTreeMap<String, RecallCounts> = BTreeMap::new();
    let mut execution_groups: BTreeMap<String, RecallCounts> = BTreeMap::new();
    let mut benign_by_program: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();

    let mut total_costs = Vec::new();
    let mut measured_acquisition_sum = 0u64;
    let mut measured_inference_sum = 0u64;
    let mut measured_cost_cases = 0usize;
    let mut telemetry_bytes_sum = 0u64;
    let mut telemetry_bytes_cases = 0usize;
    let mut max_memory = 0u64;
    let mut max_memory_cases = 0usize;

    for case_id in &test_ids {
        let meta = metadata.get(case_id).ok_or("test metadata lookup failed")?;
        let prediction = prediction_map.get(case_id).ok_or("prediction lookup failed")?;
        let truth = truth_rows.get(case_id).ok_or("truth lookup failed")?;
        if truth.case_id != prediction.case_id {
            return Err(format!("joined case_id mismatch for {case_id}"));
        }

        let invalid_sensor = meta.sensor_state != "valid";
        if invalid_sensor {
            invalid_sensor_cases += 1;
            if prediction.response == "clear" {
                invalid_sensor_clear_verdicts += 1;
            }
        }
        if prediction.response == "alert" {
            all_alerts += 1;
        }
        if matches!(prediction.response.as_str(), "abstain" | "sensor_invalid") {
            all_abstentions += 1;
        }

        match truth.class.as_str() {
            "malicious" => match truth.effect.as_str() {
                "observed" => {
                    success_attack_cases += 1;
                    let key = format!("{}|{}", meta.adaptation_family_id, meta.adaptation_level);
                    add_recall(adaptation_groups.entry(key).or_default(), &prediction.response);
                    add_recall(program_groups.entry(meta.program_group_id.clone()).or_default(), &prediction.response);
                    add_recall(execution_groups.entry(meta.execution_group_id.clone()).or_default(), &prediction.response);
                    match prediction.response.as_str() {
                        "alert" => successful_attack_alerts += 1,
                        "clear" => successful_attack_clears += 1,
                        "abstain" | "sensor_invalid" => successful_attack_abstentions += 1,
                        _ => unreachable!(),
                    }
                }
                "not_observed" => {
                    failed_attack_attempts += 1;
                    if prediction.response == "alert" {
                        failed_attack_alerts += 1;
                    }
                }
                "unknown" => {
                    unknown_effect_attempts += 1;
                    if prediction.response == "alert" {
                        unknown_effect_alerts += 1;
                    }
                }
                _ => return Err(format!("invalid malicious attack_effect for {case_id}")),
            },
            "benign" => {
                benign_cases += 1;
                if invalid_sensor {
                    invalid_benign_cases += 1;
                } else {
                    valid_benign_cases += 1;
                }
                let counts = benign_by_program.entry(meta.program_group_id.clone()).or_insert((0,0,0));
                counts.0 += 1;
                match prediction.response.as_str() {
                    "alert" => {
                        benign_alerts += 1;
                        counts.1 += 1;
                    }
                    "clear" => {
                        benign_clears += 1;
                        counts.2 += 1;
                        if !invalid_sensor {
                            valid_benign_clears += 1;
                        }
                    }
                    "abstain" | "sensor_invalid" => benign_abstentions += 1,
                    _ => unreachable!(),
                }
            }
            _ => unreachable!("truth class validated before aggregation"),
        }

        if let (Some(acquisition), Some(inference)) = (prediction.acquisition_ns, prediction.inference_ns) {
            let total = acquisition.checked_add(inference)
                .ok_or_else(|| format!("end-to-end cost overflow for {case_id}"))?;
            measured_acquisition_sum = measured_acquisition_sum.checked_add(acquisition)
                .ok_or("acquisition-cost sum overflow")?;
            measured_inference_sum = measured_inference_sum.checked_add(inference)
                .ok_or("inference-cost sum overflow")?;
            measured_cost_cases += 1;
            total_costs.push(total);
        }
        if let Some(bytes) = prediction.telemetry_bytes {
            telemetry_bytes_sum = telemetry_bytes_sum.checked_add(bytes)
                .ok_or("telemetry byte sum overflow")?;
            telemetry_bytes_cases += 1;
        }
        if let Some(memory) = prediction.peak_memory_bytes {
            max_memory = max_memory.max(memory);
            max_memory_cases += 1;
        }
    }

    let (adaptation_macro, adaptation_worst, adaptation_group_count) = macro_recall(&adaptation_groups);

    // Calibration is for malicious-attempt versus benign labels, not for
    // successful attack effects. It is unavailable unless every test case
    // has a probability score.
    let all_scores_available = test_ids.iter().all(|id| prediction_map.get(id).and_then(|p| p.score).is_some());
    let mut brier_sum = 0.0f64;
    let mut log_loss_sum = 0.0f64;
    let mut ece_bins = vec![(0usize, 0.0f64, 0.0f64); 10];
    if all_scores_available {
        for case_id in &test_ids {
            let predicted = prediction_map.get(case_id).and_then(|p| p.score).ok_or("score completeness check failed")?;
            let actual = if truth_rows.get(case_id).ok_or("truth lookup failed")?.class == "malicious" { 1.0 } else { 0.0 };
            let error = predicted - actual;
            brier_sum += error * error;
            let clipped = predicted.clamp(1e-15, 1.0 - 1e-15);
            log_loss_sum += if actual == 1.0 { -clipped.ln() } else { -(1.0 - clipped).ln() };
            let bin = ((predicted * 10.0).floor() as usize).min(9);
            ece_bins[bin].0 += 1;
            ece_bins[bin].1 += predicted;
            ece_bins[bin].2 += actual;
        }
    }
    let calibration = if all_scores_available {
        let count = test_ids.len() as f64;
        let ece = ece_bins.iter().filter(|bin| bin.0 > 0).map(|bin| {
            (bin.0 as f64 / count) * ((bin.1 / bin.0 as f64) - (bin.2 / bin.0 as f64)).abs()
        }).sum::<f64>();
        json!({
            "target_semantics":"malicious_attempt_vs_benign_not_successful_effect",
            "scored_cases":test_ids.len(),
            "brier_score":brier_sum/count,
            "log_loss":log_loss_sum/count,
            "ten_bin_ece":ece
        })
    } else {
        json!({
            "target_semantics":"malicious_attempt_vs_benign_not_successful_effect",
            "scored_cases":test_ids.iter().filter(|id| prediction_map.get(*id).and_then(|p| p.score).is_some()).count(),
            "brier_score":null,
            "log_loss":null,
            "ten_bin_ece":null,
            "unavailable_reason":"at least one detector score is null or missing"
        })
    };
    let (program_macro, program_worst, program_group_count) = macro_recall(&program_groups);
    let (execution_macro, execution_worst, execution_group_count) = macro_recall(&execution_groups);
    let attack_recall = if success_attack_cases == 0 { None } else {
        Some(successful_attack_alerts as f64 / success_attack_cases as f64)
    };
    let benign_false_alert_rate = if benign_cases == 0 { None } else {
        Some(benign_alerts as f64 / benign_cases as f64)
    };
    let benign_true_negative_rate = if valid_benign_cases == 0 { None } else {
        Some(valid_benign_clears as f64 / valid_benign_cases as f64)
    };
    let conservative_alert_precision = if all_alerts == 0 { None } else {
        Some(successful_attack_alerts as f64 / all_alerts as f64)
    };
    let abstention_rate = if test_ids.is_empty() { None } else {
        Some(all_abstentions as f64 / test_ids.len() as f64)
    };
    total_costs.sort_unstable();
    let all_costs_measured = measured_cost_cases == test_ids.len();
    let resource_cost = json!({
        "all_cases_have_acquisition_and_inference_cost": all_costs_measured,
        "cases_with_both_measured": measured_cost_cases,
        "total_test_cases": test_ids.len(),
        "acquisition_ns_sum": if all_costs_measured { Some(measured_acquisition_sum) } else { None },
        "inference_ns_sum": if all_costs_measured { Some(measured_inference_sum) } else { None },
        "end_to_end_ns_sum": if all_costs_measured { Some(measured_acquisition_sum.checked_add(measured_inference_sum).ok_or("total cost overflow")?) } else { None },
        "end_to_end_ns_p50": if all_costs_measured { percentile(&total_costs, 0.50) } else { None },
        "end_to_end_ns_p95": if all_costs_measured { percentile(&total_costs, 0.95) } else { None },
        "all_case_telemetry_bytes_measured": telemetry_bytes_cases == test_ids.len(),
        "telemetry_bytes_sum": if telemetry_bytes_cases == test_ids.len() { Some(telemetry_bytes_sum) } else { None },
        "peak_memory_measurement_cases": max_memory_cases,
        "peak_memory_bytes_max": if max_memory_cases == test_ids.len() { Some(max_memory) } else { None },
        "energy": "not_measured_by_this_evaluator"
    });

    let by_adaptation: Vec<Value> = adaptation_groups.iter().map(|(group, stats)| json!({
        "adaptation_family_and_level": group,
        "successful_attack_cases": stats.cases,
        "detected": stats.detected,
        "cleared": stats.cleared,
        "abstained": stats.abstained,
        "attack_effect_recall": recall(stats)
    })).collect();
    let by_program: Vec<Value> = program_groups.iter().map(|(group, stats)| json!({
        "program_group_id": group,
        "successful_attack_cases": stats.cases,
        "detected": stats.detected,
        "cleared": stats.cleared,
        "abstained": stats.abstained,
        "attack_effect_recall": recall(stats)
    })).collect();
    let by_execution: Vec<Value> = execution_groups.iter().map(|(group, stats)| json!({
        "execution_group_id": group,
        "successful_attack_cases": stats.cases,
        "detected": stats.detected,
        "cleared": stats.cleared,
        "abstained": stats.abstained,
        "attack_effect_recall": recall(stats)
    })).collect();
    let benign_groups: Vec<Value> = benign_by_program.iter().map(|(group, values)| json!({
        "program_group_id": group,
        "benign_cases": values.0,
        "false_alerts": values.1,
        "clear_verdicts": values.2,
        "false_alert_rate": if values.0 == 0 { None } else { Some(values.1 as f64 / values.0 as f64) }
    })).collect();

    Ok(json!({
        "schema_version": "hardsec-adaptive-evasion-receipt-v1",
        "result": "evaluated",
        "claim_scope": CLAIM_SCOPE,
        "campaign": {
            "campaign_id": prediction_campaign_id(predictions)?,
            "evidence_domain": evidence_domain,
            "platform_id": str_field(manifest, "platform_id", "manifest")?,
            "model_id": model_id,
            "model_sha256": model_sha,
            "threshold_profile_sha256": str_field(predictions, "threshold_profile_sha256", "predictions")?,
            "declared_holdout_axes": holdout_axes,
            "holdout_audit": holdout_reports
        },
        "inputs": {
            "predictions_sha256": sha256(predictions_bytes),
            "manifest_sha256": sha256(manifest_bytes),
            "truth_sha256": sha256(truth_bytes)
        },
        "primary_metrics": {
            "successful_attack_effect_cases": success_attack_cases,
            "successful_attack_alerts": successful_attack_alerts,
            "successful_attack_clear_misses": successful_attack_clears,
            "successful_attack_abstentions": successful_attack_abstentions,
            "attack_effect_conditioned_recall": attack_recall,
            "failed_attack_attempts_excluded_from_success_recall": failed_attack_attempts,
            "failed_attack_attempts_alerted": failed_attack_alerts,
            "unknown_attack_effect_attempts_excluded": unknown_effect_attempts,
            "unknown_effect_attempts_alerted": unknown_effect_alerts,
            "benign_cases": benign_cases,
            "benign_false_alerts": benign_alerts,
            "benign_clear_verdicts": benign_clears,
            "benign_abstentions": benign_abstentions,
            "benign_false_alert_rate": benign_false_alert_rate,
            "benign_true_negative_rate_on_valid_sensor_cases": benign_true_negative_rate,
            "benign_valid_sensor_cases": valid_benign_cases,
            "benign_invalid_sensor_cases": invalid_benign_cases,
            "conservative_effect_confirmed_alert_precision": conservative_alert_precision,
            "all_alerts": all_alerts,
            "abstention_rate": abstention_rate
        },
        "generalization": {
            "adaptation_family_level": by_adaptation,
            "adaptation_family_macro_recall": adaptation_macro,
            "worst_adaptation_family_recall": adaptation_worst,
            "adaptation_group_count": adaptation_group_count,
            "program_group_level": by_program,
            "program_macro_recall": program_macro,
            "worst_program_recall": program_worst,
            "program_group_count": program_group_count,
            "execution_group_level": by_execution,
            "execution_macro_recall": execution_macro,
            "worst_execution_recall": execution_worst,
            "execution_group_count": execution_group_count,
            "benign_false_alert_by_program": benign_groups
        },
        "calibration": calibration,
        "sensor_integrity": {
            "invalid_sensor_cases": invalid_sensor_cases,
            "invalid_sensor_clear_verdicts": invalid_sensor_clear_verdicts,
            "integrity_outcome": if invalid_sensor_clear_verdicts == 0 { "no_silent_clear_observed" } else { "integrity_violation_observed" }
        },
        "resource_cost": resource_cost,
        "limitations": [
            "Attack-effect recall is conditional on the evaluator-supplied observed-effect labels and their evidence hashes.",
            "The evaluator cannot authenticate the truth oracle or prove that attack-effect labels are complete.",
            "Failed or unknown-effect attack attempts are not counted as successful attacks or as successful detector evasions.",
            "A single campaign contains one evidence domain; simulator, synthetic and physical results must not be pooled.",
            "Holdout checks validate caller-declared group IDs, not the truth of source lineage metadata.",
            "Cost summaries are emitted only where the corresponding per-case measurements are complete; energy is not inferred.",
            "The receipt is unsigned and does not establish exploitability, prevention, attribution, silicon security or certification."
        ]
    }))
}

fn prediction_campaign_id(value: &Value) -> Result<&str, String> {
    str_field(value, "campaign_id", "predictions")
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
    let mut values = BTreeMap::new();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if !matches!(flag.as_str(), "--predictions" | "--manifest" | "--truth" | "--receipt") {
            return Err(format!("unknown argument {flag}"));
        }
        let path = args.next().ok_or_else(|| format!("{flag} requires a path"))?;
        if values.insert(flag.clone(), PathBuf::from(path)).is_some() {
            return Err(format!("duplicate argument {flag}"));
        }
    }
    let take = |key: &str| values.get(key).cloned().ok_or_else(|| format!("missing required argument {key}"));
    Ok(Args {
        predictions: take("--predictions")?,
        manifest: take("--manifest")?,
        truth: take("--truth")?,
        receipt: take("--receipt")?,
    })
}

fn validate_receipt_location(receipt: &Path, inputs: &[&Path]) -> Result<(), String> {
    if receipt.exists() {
        return Err(format!("refusing to overwrite receipt {}", receipt.display()));
    }
    let parent = receipt.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let canonical_parent = fs::canonicalize(parent)
        .map_err(|e| format!("receipt parent must exist: {e}"))?;
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
    let (predictions_bytes, predictions) = read_input(&args.predictions, "predictions")?;
    let (manifest_bytes, manifest) = read_input(&args.manifest, "manifest")?;
    let (truth_bytes, truth) = read_input(&args.truth, "truth")?;
    validate_receipt_location(&args.receipt, &[&args.predictions, &args.manifest, &args.truth])?;
    let receipt = audit_values(
        &predictions, &manifest, &truth,
        &predictions_bytes, &manifest_bytes, &truth_bytes,
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
            eprintln!("HARDSEC-012 adaptive-evasion audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_values() -> (Value, Value, Value, Vec<u8>, Vec<u8>, Vec<u8>) {
        let predictions = json!({
            "schema_version":"hardsec-adaptive-predictions-v1",
            "campaign_id":"hs012-synthetic-campaign-01",
            "model_id":"fixture-rule-detector",
            "model_sha256":"a".repeat(64),
            "threshold_profile_sha256":"b".repeat(64),
            "cases":[
                {"case_id":"c1","response":"alert","score":0.91,"cost":{"acquisition_ns":100,"inference_ns":20,"telemetry_bytes":128,"peak_memory_bytes":2048}},
                {"case_id":"c2","response":"clear","score":0.12,"cost":{"acquisition_ns":100,"inference_ns":20,"telemetry_bytes":128,"peak_memory_bytes":2048}},
                {"case_id":"c3","response":"abstain","score":null,"cost":{"acquisition_ns":100,"inference_ns":20,"telemetry_bytes":128,"peak_memory_bytes":2048}},
                {"case_id":"c4","response":"alert","score":0.88,"cost":{"acquisition_ns":100,"inference_ns":20,"telemetry_bytes":128,"peak_memory_bytes":2048}},
                {"case_id":"c5","response":"alert","score":0.77,"cost":{"acquisition_ns":100,"inference_ns":20,"telemetry_bytes":128,"peak_memory_bytes":2048}},
                {"case_id":"c6","response":"clear","score":0.08,"cost":{"acquisition_ns":100,"inference_ns":20,"telemetry_bytes":128,"peak_memory_bytes":2048}},
                {"case_id":"c7","response":"alert","score":0.81,"cost":{"acquisition_ns":100,"inference_ns":20,"telemetry_bytes":128,"peak_memory_bytes":2048}},
                {"case_id":"c8","response":"clear","score":0.07,"cost":{"acquisition_ns":100,"inference_ns":20,"telemetry_bytes":128,"peak_memory_bytes":2048}},
                {"case_id":"c9","response":"sensor_invalid","score":null,"cost":{"acquisition_ns":100,"inference_ns":20,"telemetry_bytes":128,"peak_memory_bytes":2048}}
            ]
        });
        let manifest = json!({
            "schema_version":"hardsec-adaptive-case-manifest-v1",
            "campaign_id":"hs012-synthetic-campaign-01",
            "evidence_domain":"synthetic_fixture",
            "platform_id":"synthetic-cpu-model-v1",
            "holdout_axes":["program","execution","attack_family","adaptation_family"],
            "cases":[
                {"case_id":"tr1","split":"train","program_group_id":"train-p","execution_group_id":"train-e","device_config_id":"cpu-a","attack_family_id":"spectre-base","adaptation_family_id":"none","adaptation_level":0,"sensor_state":"valid"},
                {"case_id":"va1","split":"validation","program_group_id":"val-p","execution_group_id":"val-e","device_config_id":"cpu-a","attack_family_id":"spectre-base","adaptation_family_id":"none","adaptation_level":0,"sensor_state":"valid"},
                {"case_id":"c1","split":"test","program_group_id":"p1","execution_group_id":"e1","device_config_id":"cpu-a","attack_family_id":"spectre-evasive","adaptation_family_id":"timing","adaptation_level":1,"sensor_state":"valid"},
                {"case_id":"c2","split":"test","program_group_id":"p2","execution_group_id":"e2","device_config_id":"cpu-a","attack_family_id":"spectre-evasive","adaptation_family_id":"interleave","adaptation_level":2,"sensor_state":"valid"},
                {"case_id":"c3","split":"test","program_group_id":"p3","execution_group_id":"e3","device_config_id":"cpu-a","attack_family_id":"spectre-evasive","adaptation_family_id":"timing","adaptation_level":2,"sensor_state":"untrusted"},
                {"case_id":"c4","split":"test","program_group_id":"p4","execution_group_id":"e4","device_config_id":"cpu-a","attack_family_id":"spectre-evasive","adaptation_family_id":"timing","adaptation_level":3,"sensor_state":"valid"},
                {"case_id":"c5","split":"test","program_group_id":"p5","execution_group_id":"e5","device_config_id":"cpu-a","attack_family_id":"spectre-evasive","adaptation_family_id":"interleave","adaptation_level":1,"sensor_state":"valid"},
                {"case_id":"c6","split":"test","program_group_id":"p6","execution_group_id":"e6","device_config_id":"cpu-a","attack_family_id":"none","adaptation_family_id":"none","adaptation_level":0,"sensor_state":"valid"},
                {"case_id":"c7","split":"test","program_group_id":"p7","execution_group_id":"e7","device_config_id":"cpu-a","attack_family_id":"none","adaptation_family_id":"none","adaptation_level":0,"sensor_state":"valid"},
                {"case_id":"c8","split":"test","program_group_id":"p8","execution_group_id":"e8","device_config_id":"cpu-a","attack_family_id":"none","adaptation_family_id":"none","adaptation_level":0,"sensor_state":"misaligned"},
                {"case_id":"c9","split":"test","program_group_id":"p9","execution_group_id":"e9","device_config_id":"cpu-a","attack_family_id":"none","adaptation_family_id":"none","adaptation_level":0,"sensor_state":"missing"}
            ]
        });
        let truth = json!({
            "schema_version":"hardsec-adaptive-oracle-v1",
            "campaign_id":"hs012-synthetic-campaign-01",
            "cases":[
                {"case_id":"c1","class":"malicious","attack_effect":"observed","outcome_evidence_sha256":"1".repeat(64)},
                {"case_id":"c2","class":"malicious","attack_effect":"observed","outcome_evidence_sha256":"2".repeat(64)},
                {"case_id":"c3","class":"malicious","attack_effect":"observed","outcome_evidence_sha256":"3".repeat(64)},
                {"case_id":"c4","class":"malicious","attack_effect":"not_observed","outcome_evidence_sha256":"4".repeat(64)},
                {"case_id":"c5","class":"malicious","attack_effect":"unknown","outcome_evidence_sha256":"5".repeat(64)},
                {"case_id":"c6","class":"benign","attack_effect":"not_applicable","outcome_evidence_sha256":"6".repeat(64)},
                {"case_id":"c7","class":"benign","attack_effect":"not_applicable","outcome_evidence_sha256":"7".repeat(64)},
                {"case_id":"c8","class":"benign","attack_effect":"not_applicable","outcome_evidence_sha256":"8".repeat(64)},
                {"case_id":"c9","class":"benign","attack_effect":"not_applicable","outcome_evidence_sha256":"9".repeat(64)}
            ]
        });
        let pb = serde_json::to_vec(&predictions).unwrap();
        let mb = serde_json::to_vec(&manifest).unwrap();
        let tb = serde_json::to_vec(&truth).unwrap();
        (predictions,manifest,truth,pb,mb,tb)
    }

    #[test]
    fn known_answer_separates_success_failed_unknown_and_benign_cases() {
        let (p,m,t,pb,mb,tb) = fixture_values();
        let receipt = audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap();
        let metrics = &receipt["primary_metrics"];
        assert_eq!(metrics["successful_attack_effect_cases"], 3);
        assert_eq!(metrics["successful_attack_alerts"], 1);
        assert_eq!(metrics["attack_effect_conditioned_recall"], 1.0/3.0);
        assert_eq!(metrics["failed_attack_attempts_excluded_from_success_recall"], 1);
        assert_eq!(metrics["unknown_attack_effect_attempts_excluded"], 1);
        assert_eq!(metrics["benign_cases"], 4);
        assert_eq!(metrics["benign_false_alert_rate"], 0.25);
    }

    #[test]
    fn invalid_sensor_clear_is_reported_as_integrity_violation() {
        let (p,m,t,pb,mb,tb) = fixture_values();
        let receipt = audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap();
        assert_eq!(receipt["sensor_integrity"]["invalid_sensor_cases"], 3);
        assert_eq!(receipt["sensor_integrity"]["invalid_sensor_clear_verdicts"], 1);
        assert_eq!(receipt["sensor_integrity"]["integrity_outcome"], "integrity_violation_observed");
    }

    #[test]
    fn failed_attack_attempt_is_not_counted_as_successful_evasion() {
        let (p,m,t,pb,mb,tb) = fixture_values();
        let receipt = audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap();
        assert_eq!(receipt["primary_metrics"]["failed_attack_attempts_alerted"], 1);
        assert_eq!(receipt["primary_metrics"]["successful_attack_effect_cases"], 3);
    }

    #[test]
    fn rejects_group_holdout_leakage() {
        let (p,mut m,t,pb,_,tb) = fixture_values();
        m["cases"][0]["program_group_id"] = json!("p1");
        let mb = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap_err().contains("crosses"));
    }

    #[test]
    fn rejects_missing_test_prediction() {
        let (mut p,m,t,_,mb,tb) = fixture_values();
        p["cases"].as_array_mut().unwrap().pop();
        let pb = serde_json::to_vec(&p).unwrap();
        assert!(audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap_err().contains("exactly the manifest's test case IDs"));
    }

    #[test]
    fn rejects_duplicate_prediction_case_id() {
        let (mut p,m,t,_,mb,tb) = fixture_values();
        let first = p["cases"][0].clone();
        p["cases"].as_array_mut().unwrap().push(first);
        let pb = serde_json::to_vec(&p).unwrap();
        assert!(audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap_err().contains("duplicate prediction case_id"));
    }

    #[test]
    fn rejects_benign_case_with_attack_effect() {
        let (p,m,mut t,pb,mb,_) = fixture_values();
        t["cases"][5]["attack_effect"] = json!("observed");
        let tb = serde_json::to_vec(&t).unwrap();
        assert!(audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap_err().contains("semantics conflict"));
    }

    #[test]
    fn rejects_unknown_effect_label_without_coercion() {
        let (p,m,mut t,pb,mb,_) = fixture_values();
        t["cases"][0]["attack_effect"] = json!("probably_successful");
        let tb = serde_json::to_vec(&t).unwrap();
        assert!(audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap_err().contains("unsupported"));
    }

    #[test]
    fn refuses_to_claim_cost_when_any_case_lacks_acquisition_or_inference() {
        let (mut p,m,t,_,mb,tb) = fixture_values();
        p["cases"][0]["cost"]["acquisition_ns"] = Value::Null;
        let pb = serde_json::to_vec(&p).unwrap();
        let receipt = audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap();
        assert_eq!(receipt["resource_cost"]["all_cases_have_acquisition_and_inference_cost"], false);
        assert_eq!(receipt["resource_cost"]["end_to_end_ns_sum"], Value::Null);
    }

    #[test]
    fn computes_complete_measured_cost_separately() {
        let (p,m,t,pb,mb,tb) = fixture_values();
        let receipt = audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap();
        assert_eq!(receipt["resource_cost"]["all_cases_have_acquisition_and_inference_cost"], true);
        assert_eq!(receipt["resource_cost"]["acquisition_ns_sum"], 900);
        assert_eq!(receipt["resource_cost"]["inference_ns_sum"], 180);
        assert_eq!(receipt["resource_cost"]["end_to_end_ns_sum"], 1080);
    }

    #[test]
    fn reports_adaptation_strata_separately() {
        let (p,m,t,pb,mb,tb) = fixture_values();
        let receipt = audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap();
        assert_eq!(receipt["generalization"]["adaptation_group_count"], 3);
        assert_eq!(receipt["generalization"]["adaptation_family_level"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn calibration_is_reported_only_when_every_detector_score_exists() {
        let (mut p,m,t,_,mb,tb) = fixture_values();
        p["cases"][2]["score"] = json!(0.53);
        p["cases"][8]["score"] = json!(0.04);
        let pb = serde_json::to_vec(&p).unwrap();
        let receipt = audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap();
        assert_eq!(receipt["calibration"]["scored_cases"], 9);
        assert!(receipt["calibration"]["brier_score"].as_f64().is_some());
        assert_eq!(receipt["calibration"]["target_semantics"], "malicious_attempt_vs_benign_not_successful_effect");
    }

    #[test]
    fn valid_sensor_benign_tnr_excludes_invalid_sensor_clear_verdicts() {
        let (p,m,t,pb,mb,tb) = fixture_values();
        let receipt = audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap();
        assert_eq!(receipt["primary_metrics"]["benign_valid_sensor_cases"], 2);
        assert_eq!(receipt["primary_metrics"]["benign_invalid_sensor_cases"], 2);
        assert_eq!(receipt["primary_metrics"]["benign_true_negative_rate_on_valid_sensor_cases"], 0.5);
    }

    #[test]
    fn rejects_campaign_domain_pooling_at_case_level_by_contract() {
        let (p,mut m,t,pb,_,tb) = fixture_values();
        m["evidence_domain"] = json!("combined_physical_and_simulator");
        let mb = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap_err().contains("unsupported evidence_domain"));
    }
}
