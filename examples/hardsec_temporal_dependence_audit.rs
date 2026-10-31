//! HARDSEC-022 functional-dependence and block-coverage auditor.
//!
//! This tool audits a *declared* causal-innovation/FDM contract and computes
//! non-overlapping block-based uncertainty diagnostics for a fixed score
//! threshold. A numerically consistent FDM envelope is not authenticated by
//! this tool; the innovation model, coupling proof and stationarity assumptions
//! remain unverified. The block standard error is post-hoc calibration analysis,
//! not an anytime-valid CCTM false-alarm or e-process guarantee.
//!
//! Usage:
//!   hardsec_temporal_dependence_audit --model model.json \
//!     --data scores.json --receipt receipt.json

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const MODEL_SCHEMA: &str = "hardsec-temporal-dependence-model-v1";
const DATA_SCHEMA: &str = "hardsec-temporal-dependence-data-v1";
const RECEIPT_SCHEMA: &str = "hardsec-temporal-dependence-receipt-v1";
const EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
struct GeometricFdm {
    amplitude: f64,
    rho: f64,
    lag_upper_bounds: Vec<f64>,
    proof_artifact_sha256: String,
    proof_reference: String,
    theta_sum_upper_bound: f64,
    tail_upper_bound: f64,
}

#[derive(Debug, Clone)]
enum FdmContract {
    SummableGeometric(GeometricFdm),
    NotBounded,
}

#[derive(Debug, Clone)]
struct Model {
    dependence_model_id: String,
    evidence_domain: String,
    stream_id: String,
    calibration_execution_group_id: String,
    evaluation_execution_group_id: String,
    calibration_device_boot_group_id: String,
    evaluation_device_boot_group_id: String,
    context_id: String,
    scoring_artifact_sha256: String,
    alpha: f64,
    block_lengths: Vec<usize>,
    primary_block_length: usize,
    nominal_level: f64,
    fdm: FdmContract,
}

#[derive(Debug, Clone)]
struct Data {
    evidence_domain: String,
    stream_id: String,
    context_id: String,
    scoring_artifact_sha256: String,
    calibration_execution_group_id: String,
    evaluation_execution_group_id: String,
    calibration_device_boot_group_id: String,
    evaluation_device_boot_group_id: String,
    calibration_scores: Vec<f64>,
    evaluation_scores: Vec<f64>,
}

#[derive(Debug, Clone)]
struct Args {
    model: PathBuf,
    data: PathBuf,
    receipt: PathBuf,
}

#[derive(Debug, Clone)]
struct BlockReport {
    block_length: usize,
    block_count: usize,
    complete_calibration_count: usize,
    discarded_calibration_count: usize,
    calibration_block_mean: f64,
    sigma_hat_covariance_sum: f64,
    iid_standard_error: f64,
    block_standard_error: f64,
    realized_evaluation_coverage: f64,
    standardized_coverage_deviation: Option<f64>,
    normal_critical_value: f64,
    normal_interval_lower: f64,
    normal_interval_upper: f64,
    diagnostic_status: &'static str,
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

fn valid_sha(value: &str, context: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(format!("{context} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn finite_scores(value: &Value, key: &str, context: &str) -> Result<Vec<f64>, String> {
    let values = value.get(key).and_then(Value::as_array)
        .ok_or_else(|| format!("{context}.{key} must be an array"))?;
    if values.is_empty() {
        return Err(format!("{context}.{key} must not be empty"));
    }
    let mut output = Vec::with_capacity(values.len());
    for (index, item) in values.iter().enumerate() {
        let score = item.as_f64()
            .ok_or_else(|| format!("{context}.{key}[{index}] must be numeric"))?;
        if !score.is_finite() {
            return Err(format!("{context}.{key}[{index}] must be finite"));
        }
        output.push(score);
    }
    Ok(output)
}

fn parse_fdm(value: &Value) -> Result<FdmContract, String> {
    let status = string_field(value, "status", "model.dependence_bound")?;
    match status {
        "not_bounded" => {
            if value.get("envelope_type").is_some()
                || value.get("lag_upper_bounds").is_some() {
                return Err("not_bounded FDM contract must not supply a theorem-level envelope".to_owned());
            }
            Ok(FdmContract::NotBounded)
        }
        "summable_geometric_claimed" => {
            if string_field(value, "envelope_type", "model.dependence_bound")? != "geometric" {
                return Err("v1 supports only a geometric summable FDM envelope".to_owned());
            }
            let amplitude = f64_field(value, "amplitude", "model.dependence_bound")?;
            let rho = f64_field(value, "rho", "model.dependence_bound")?;
            if amplitude <= 0.0 || !(0.0..1.0).contains(&rho) {
                return Err("geometric FDM requires amplitude > 0 and rho in (0,1)".to_owned());
            }
            let proof_artifact_sha256 = string_field(value, "proof_artifact_sha256", "model.dependence_bound")?.to_owned();
            valid_sha(&proof_artifact_sha256, "model.dependence_bound.proof_artifact_sha256")?;
            let proof_reference = string_field(value, "proof_reference", "model.dependence_bound")?.to_owned();
            if string_field(value, "per_lag_bound_proof_status", "model.dependence_bound")?
                != "assumption_not_verified" {
                return Err("per-lag FDM proof status must remain assumption_not_verified".to_owned());
            }
            let lag_values = value.get("lag_upper_bounds").and_then(Value::as_array)
                .ok_or("model.dependence_bound.lag_upper_bounds must be an array")?;
            if lag_values.is_empty() {
                return Err("at least one per-lag FDM upper bound is required".to_owned());
            }
            let mut lag_upper_bounds = Vec::with_capacity(lag_values.len());
            let mut partial_sum = 0.0;
            for (index, item) in lag_values.iter().enumerate() {
                let context = format!("model.dependence_bound.lag_upper_bounds[{index}]");
                let lag = usize_field(item, "lag", &context)?;
                if lag != index {
                    return Err(format!("{context}.lag must be contiguous from zero; expected {index}"));
                }
                let theta = f64_field(item, "theta_upper_bound", &context)?;
                if !(0.0..=1.0).contains(&theta) {
                    return Err(format!("{context}.theta_upper_bound must be in [0,1]"));
                }
                let envelope = amplitude * rho.powi(i32::try_from(lag).map_err(|_| "lag index exceeds supported range")?);
                if theta > envelope + EPS {
                    return Err(format!("{context}.theta_upper_bound exceeds declared geometric envelope"));
                }
                partial_sum += theta;
                if !partial_sum.is_finite() {
                    return Err("FDM partial sum became non-finite".to_owned());
                }
                lag_upper_bounds.push(theta);
            }
            let last_lag = lag_upper_bounds.len() - 1;
            let tail_upper_bound = amplitude * rho.powi(i32::try_from(last_lag + 1).map_err(|_| "lag index exceeds supported range")?)
                / (1.0 - rho);
            let theta_sum_upper_bound = partial_sum + tail_upper_bound;
            if !tail_upper_bound.is_finite() || !theta_sum_upper_bound.is_finite() {
                return Err("FDM infinite-sum upper bound is non-finite".to_owned());
            }
            Ok(FdmContract::SummableGeometric(GeometricFdm {
                amplitude,
                rho,
                lag_upper_bounds,
                proof_artifact_sha256,
                proof_reference,
                theta_sum_upper_bound,
                tail_upper_bound,
            }))
        }
        _ => Err(format!("unsupported dependence_bound.status {status}")),
    }
}

fn parse_model(value: &Value) -> Result<Model, String> {
    if string_field(value, "schema_version", "model")? != MODEL_SCHEMA {
        return Err("unsupported temporal dependence model schema_version".to_owned());
    }
    let dependence_model_id = string_field(value, "dependence_model_id", "model")?.to_owned();
    let evidence_domain = string_field(value, "evidence_domain", "model")?.to_owned();
    if !matches!(evidence_domain.as_str(), "synthetic_fixture" | "simulator" | "physical_hardware") {
        return Err(format!("unsupported evidence_domain {evidence_domain}"));
    }
    let stream_id = string_field(value, "stream_id", "model")?.to_owned();
    let calibration_execution_group_id = string_field(value, "calibration_execution_group_id", "model")?.to_owned();
    let evaluation_execution_group_id = string_field(value, "evaluation_execution_group_id", "model")?.to_owned();
    let calibration_device_boot_group_id = string_field(value, "calibration_device_boot_group_id", "model")?.to_owned();
    let evaluation_device_boot_group_id = string_field(value, "evaluation_device_boot_group_id", "model")?.to_owned();
    if calibration_execution_group_id == evaluation_execution_group_id {
        return Err("calibration and evaluation execution groups must be distinct".to_owned());
    }
    if calibration_device_boot_group_id == evaluation_device_boot_group_id {
        return Err("calibration and evaluation device-boot groups must be distinct".to_owned());
    }
    let context_id = string_field(value, "context_id", "model")?.to_owned();
    let scoring_artifact_sha256 = string_field(value, "scoring_artifact_sha256", "model")?.to_owned();
    valid_sha(&scoring_artifact_sha256, "model.scoring_artifact_sha256")?;
    if string_field(value, "anomaly_score_direction", "model")? != "higher_is_more_anomalous" {
        return Err("only higher_is_more_anomalous score direction is supported".to_owned());
    }
    let alpha = f64_field(value, "target_miscoverage_alpha", "model")?;
    if !(0.0..1.0).contains(&alpha) {
        return Err("target_miscoverage_alpha must be in (0,1)".to_owned());
    }
    for key in ["predictor_fixed_status", "score_marginal_continuity_status"] {
        if string_field(value, key, "model")? != "assumption_not_verified" {
            return Err(format!("{key} must remain assumption_not_verified"));
        }
    }
    let causal = value.get("causal_innovation_model")
        .ok_or("model.causal_innovation_model missing")?;
    let model_sha = string_field(causal, "model_artifact_sha256", "model.causal_innovation_model")?;
    valid_sha(model_sha, "model.causal_innovation_model.model_artifact_sha256")?;
    for key in ["iid_innovations_status", "stationarity_status", "causal_representation_status"] {
        if string_field(causal, key, "model.causal_innovation_model")? != "assumption_not_verified" {
            return Err(format!("causal innovation {key} must remain assumption_not_verified"));
        }
    }

    let dependence_value = value.get("dependence_bound")
        .ok_or("model.dependence_bound missing")?;
    let fdm = parse_fdm(dependence_value)?;

    let block = value.get("block_diagnostic").ok_or("model.block_diagnostic missing")?;
    if string_field(block, "block_inference_assumption_status", "model.block_diagnostic")?
        != "assumption_not_verified" {
        return Err("block inference assumptions must remain assumption_not_verified".to_owned());
    }
    let block_lengths_value = block.get("block_lengths").and_then(Value::as_array)
        .ok_or("model.block_diagnostic.block_lengths must be an array")?;
    if block_lengths_value.is_empty() {
        return Err("at least one block length is required".to_owned());
    }
    let mut block_lengths = Vec::with_capacity(block_lengths_value.len());
    let mut seen_lengths = BTreeSet::new();
    for (index, item) in block_lengths_value.iter().enumerate() {
        let length = item.as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| format!("block_lengths[{index}] must be a positive integer"))?;
        if length == 0 || !seen_lengths.insert(length) {
            return Err(format!("block_lengths[{index}] must be positive and unique"));
        }
        block_lengths.push(length);
    }
    let primary_block_length = usize_field(block, "primary_block_length", "model.block_diagnostic")?;
    if !block_lengths.contains(&primary_block_length) {
        return Err("primary_block_length must appear in block_lengths".to_owned());
    }
    let nominal_level = f64_field(block, "nominal_two_sided_level", "model.block_diagnostic")?;
    if ![0.10, 0.05, 0.01].iter().any(|level| (nominal_level - level).abs() < EPS) {
        return Err("nominal_two_sided_level must be 0.10, 0.05 or 0.01".to_owned());
    }

    Ok(Model {
        dependence_model_id, evidence_domain, stream_id,
        calibration_execution_group_id, evaluation_execution_group_id,
        calibration_device_boot_group_id, evaluation_device_boot_group_id,
        context_id, scoring_artifact_sha256, alpha, block_lengths,
        primary_block_length, nominal_level, fdm,
    })
}

fn parse_data(value: &Value, model: &Model) -> Result<Data, String> {
    if string_field(value, "schema_version", "data")? != DATA_SCHEMA {
        return Err("unsupported temporal dependence data schema_version".to_owned());
    }
    let evidence_domain = string_field(value, "evidence_domain", "data")?.to_owned();
    let stream_id = string_field(value, "stream_id", "data")?.to_owned();
    let context_id = string_field(value, "context_id", "data")?.to_owned();
    let scoring_artifact_sha256 = string_field(value, "scoring_artifact_sha256", "data")?.to_owned();
    valid_sha(&scoring_artifact_sha256, "data.scoring_artifact_sha256")?;
    let calibration_execution_group_id = string_field(value, "calibration_execution_group_id", "data")?.to_owned();
    let evaluation_execution_group_id = string_field(value, "evaluation_execution_group_id", "data")?.to_owned();
    let calibration_device_boot_group_id = string_field(value, "calibration_device_boot_group_id", "data")?.to_owned();
    let evaluation_device_boot_group_id = string_field(value, "evaluation_device_boot_group_id", "data")?.to_owned();
    for (field, actual, expected) in [
        ("evidence_domain", evidence_domain.as_str(), model.evidence_domain.as_str()),
        ("stream_id", stream_id.as_str(), model.stream_id.as_str()),
        ("context_id", context_id.as_str(), model.context_id.as_str()),
        ("scoring_artifact_sha256", scoring_artifact_sha256.as_str(), model.scoring_artifact_sha256.as_str()),
        ("calibration_execution_group_id", calibration_execution_group_id.as_str(), model.calibration_execution_group_id.as_str()),
        ("evaluation_execution_group_id", evaluation_execution_group_id.as_str(), model.evaluation_execution_group_id.as_str()),
        ("calibration_device_boot_group_id", calibration_device_boot_group_id.as_str(), model.calibration_device_boot_group_id.as_str()),
        ("evaluation_device_boot_group_id", evaluation_device_boot_group_id.as_str(), model.evaluation_device_boot_group_id.as_str()),
    ] {
        if actual != expected {
            return Err(format!("data.{field} does not match the dependence model"));
        }
    }
    let calibration_scores = finite_scores(value, "calibration_scores", "data")?;
    let evaluation_scores = finite_scores(value, "evaluation_scores", "data")?;
    if calibration_scores.len() < 20 {
        return Err("at least 20 calibration scores are required for block diagnostics".to_owned());
    }
    if evaluation_scores.len() < 2 {
        return Err("at least two evaluation scores are required".to_owned());
    }
    for length in &model.block_lengths {
        if calibration_scores.len() / length < 2 {
            return Err(format!("block length {length} leaves fewer than two complete calibration blocks"));
        }
    }
    Ok(Data {
        evidence_domain, stream_id, context_id, scoring_artifact_sha256,
        calibration_execution_group_id, evaluation_execution_group_id,
        calibration_device_boot_group_id, evaluation_device_boot_group_id,
        calibration_scores, evaluation_scores,
    })
}

fn normal_critical_value(nominal_level: f64) -> f64 {
    if (nominal_level - 0.10).abs() < EPS {
        1.644_853_626_951_472_2
    } else if (nominal_level - 0.05).abs() < EPS {
        1.959_963_984_540_054
    } else {
        2.575_829_303_548_900_4
    }
}

fn empirical_upper_quantile(scores: &[f64], alpha: f64) -> Result<f64, String> {
    let n = scores.len();
    let rank = (((n + 1) as f64) * (1.0 - alpha)).ceil() as usize;
    if rank == 0 || rank > n {
        return Err(format!("calibration support n={n} is insufficient for alpha={alpha}"));
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(sorted[rank - 1])
}

fn block_report(
    calibration_scores: &[f64],
    evaluation_scores: &[f64],
    cutoff: f64,
    target_coverage: f64,
    block_length: usize,
    nominal_level: f64,
) -> Result<BlockReport, String> {
    let n = calibration_scores.len();
    let m = evaluation_scores.len();
    let block_count = n / block_length;
    if block_count < 2 {
        return Err(format!("block length {block_length} leaves fewer than two complete blocks"));
    }
    let complete_count = block_count * block_length;
    let mut block_means = Vec::with_capacity(block_count);
    for block_index in 0..block_count {
        let start = block_index * block_length;
        let end = start + block_length;
        let successes = calibration_scores[start..end]
            .iter().filter(|score| **score <= cutoff).count();
        block_means.push(successes as f64 / block_length as f64);
    }
    let mean = block_means.iter().sum::<f64>() / block_count as f64;
    let squared_deviation_sum = block_means.iter()
        .map(|block_mean| (block_mean - mean).powi(2))
        .sum::<f64>();
    let sigma_squared = (block_length as f64 / block_count as f64) * squared_deviation_sum;
    if !sigma_squared.is_finite() || sigma_squared < -EPS {
        return Err("block covariance-sum estimator is invalid".to_owned());
    }
    let sigma_hat_covariance_sum = sigma_squared.max(0.0);
    let sigma = sigma_hat_covariance_sum.sqrt();
    let iid_standard_error = (target_coverage * (1.0 - target_coverage)
        * (1.0 / n as f64 + 1.0 / m as f64)).sqrt();
    let block_standard_error = sigma * (1.0 / n as f64 + 1.0 / m as f64).sqrt();
    let realized_count = evaluation_scores.iter().filter(|score| **score <= cutoff).count();
    let realized_coverage = realized_count as f64 / m as f64;
    let standardized = if block_standard_error > 0.0 {
        Some((realized_coverage - target_coverage) / block_standard_error)
    } else {
        None
    };
    let critical = normal_critical_value(nominal_level);
    // Null prediction band for the future realized coverage is centered on
    // the nominal target, not on the observed evaluation coverage.
    let lower = (target_coverage - critical * block_standard_error).max(0.0);
    let upper = (target_coverage + critical * block_standard_error).min(1.0);
    let diagnostic_status = if block_count >= 10 && block_standard_error > 0.0 {
        "asymptotic_normal_diagnostic_only_assumptions_unverified"
    } else {
        "variance_estimate_only_too_few_blocks_or_zero_standard_error"
    };
    Ok(BlockReport {
        block_length, block_count,
        complete_calibration_count: complete_count,
        discarded_calibration_count: n - complete_count,
        calibration_block_mean: mean,
        sigma_hat_covariance_sum,
        iid_standard_error,
        block_standard_error,
        realized_evaluation_coverage: realized_coverage,
        standardized_coverage_deviation: standardized,
        normal_critical_value: critical,
        normal_interval_lower: lower,
        normal_interval_upper: upper,
        diagnostic_status,
    })
}

fn fdm_coverage_bound(theta_sum: f64, n: usize, alpha: f64) -> Value {
    let n_f = n as f64;
    let condition = 2.0 / n_f + (2.0 * theta_sum * theta_sum / n_f).cbrt();
    let required = alpha.min(1.0 - alpha);
    if condition < required {
        let error = 2.0 / n_f + 3.0 * (theta_sum * theta_sum / (4.0 * n_f)).cbrt();
        json!({
            "status":"numeric_condition_satisfied_assumptions_unverified",
            "assumption_condition_value":condition,
            "assumption_condition_threshold":required,
            "marginal_coverage_error_upper_bound":error
        })
    } else {
        json!({
            "status":"numeric_condition_not_satisfied",
            "assumption_condition_value":condition,
            "assumption_condition_threshold":required,
            "marginal_coverage_error_upper_bound":null
        })
    }
}

fn audit_values(
    model_value: &Value,
    data_value: &Value,
    model_bytes: &[u8],
    data_bytes: &[u8],
) -> Result<Value, String> {
    let model = parse_model(model_value)?;
    let data = parse_data(data_value, &model)?;
    let cutoff = empirical_upper_quantile(&data.calibration_scores, model.alpha)?;
    let target = 1.0 - model.alpha;
    let block_reports: Vec<BlockReport> = model.block_lengths.iter().map(|length| {
        block_report(
            &data.calibration_scores, &data.evaluation_scores,
            cutoff, target, *length, model.nominal_level
        )
    }).collect::<Result<Vec<_>, _>>()?;
    let primary = block_reports.iter().find(|report| report.block_length == model.primary_block_length)
        .ok_or("primary block length report missing")?;

    let fdm_summary = match &model.fdm {
        FdmContract::SummableGeometric(fdm) => json!({
            "status":"model_bound_summable_fdm_contract_consistent_not_authenticated",
            "envelope_type":"geometric",
            "amplitude":fdm.amplitude,
            "rho":fdm.rho,
            "lag_count":fdm.lag_upper_bounds.len(),
            "lag_upper_bounds":fdm.lag_upper_bounds.clone(),
            "finite_partial_sum":fdm.lag_upper_bounds.iter().sum::<f64>(),
            "tail_upper_bound_after_last_lag":fdm.tail_upper_bound,
            "theta_sum_upper_bound":fdm.theta_sum_upper_bound,
            "proof_artifact_sha256":fdm.proof_artifact_sha256,
            "proof_reference":fdm.proof_reference,
            "proof_status":"assumption_not_verified",
            "marginal_coverage_bound_from_zhai_theorem_1":fdm_coverage_bound(
                fdm.theta_sum_upper_bound, data.calibration_scores.len(), model.alpha
            )
        }),
        FdmContract::NotBounded => json!({
            "status":"not_bounded",
            "theta_sum_upper_bound":null,
            "marginal_coverage_bound_from_zhai_theorem_1":null
        }),
    };

    let report_values: Vec<Value> = block_reports.iter().map(|report| json!({
        "block_length":report.block_length,
        "block_count":report.block_count,
        "complete_calibration_count":report.complete_calibration_count,
        "discarded_calibration_count":report.discarded_calibration_count,
        "calibration_block_mean":report.calibration_block_mean,
        "sigma_hat_covariance_sum":report.sigma_hat_covariance_sum,
        "iid_standard_error":report.iid_standard_error,
        "block_standard_error":report.block_standard_error,
        "standard_error_ratio_block_over_iid":if report.iid_standard_error > 0.0 {
            Some(report.block_standard_error / report.iid_standard_error)
        } else { None },
        "realized_evaluation_coverage":report.realized_evaluation_coverage,
        "target_coverage":target,
        "standardized_coverage_deviation":report.standardized_coverage_deviation,
        "normal_critical_value":report.normal_critical_value,
        "null_realized_coverage_interval_lower":report.normal_interval_lower,
        "null_realized_coverage_interval_upper":report.normal_interval_upper,
        "observed_coverage_inside_null_interval":report.realized_evaluation_coverage >= report.normal_interval_lower
            && report.realized_evaluation_coverage <= report.normal_interval_upper,
        "diagnostic_status":report.diagnostic_status
    })).collect();

    Ok(json!({
        "schema_version":RECEIPT_SCHEMA,
        "result":"audited",
        "claim_scope":"declared_fdm_contract_consistency_and_posthoc_block_coverage_diagnostics_only",
        "dependence_model_id":model.dependence_model_id,
        "evidence_domain":model.evidence_domain,
        "stream_id":model.stream_id,
        "context_id":model.context_id,
        "score_artifact_sha256":model.scoring_artifact_sha256,
        "groups":{
            "calibration_execution_group_id":model.calibration_execution_group_id,
            "evaluation_execution_group_id":model.evaluation_execution_group_id,
            "calibration_device_boot_group_id":model.calibration_device_boot_group_id,
            "evaluation_device_boot_group_id":model.evaluation_device_boot_group_id,
            "execution_groups_distinct":true,
            "device_boot_groups_distinct":true,
            "independence_authenticated":false
        },
        "inputs":{
            "model_sha256":sha256(model_bytes),
            "data_sha256":sha256(data_bytes)
        },
        "calibration":{
            "score_count":data.calibration_scores.len(),
            "target_miscoverage_alpha":model.alpha,
            "target_coverage":target,
            "upper_order_statistic_rank":(((data.calibration_scores.len()+1) as f64)*target).ceil() as usize,
            "fixed_cutoff":cutoff,
            "calibration_coverage_at_cutoff":data.calibration_scores.iter().filter(|score| **score<=cutoff).count() as f64 / data.calibration_scores.len() as f64
        },
        "evaluation":{
            "score_count":data.evaluation_scores.len(),
            "realized_coverage_at_fixed_cutoff":data.evaluation_scores.iter().filter(|score| **score<=cutoff).count() as f64 / data.evaluation_scores.len() as f64
        },
        "functional_dependence":{
            "indicator_definition":"I_t = 1{score_t <= fixed calibration cutoff}",
            "causal_innovation_model_status":"assumption_not_verified",
            "iid_innovations_status":"assumption_not_verified",
            "stationarity_status":"assumption_not_verified",
            "score_marginal_continuity_status":"assumption_not_verified",
            "predictor_fixed_status":"assumption_not_verified",
            "fdm_contract":fdm_summary,
            "sequential_ctm_validity_inferred":false,
            "reason":"FDM coverage results concern marginal score-threshold coverage under their assumptions; they do not establish a conditional supermartingale or sequential CCTM PFA guarantee."
        },
        "block_diagnostic":{
            "method":"non_overlapping_calibration_indicator_batch_means",
            "primary_block_length":model.primary_block_length,
            "primary_block_report":report_values.iter().find(|report| report["block_length"]==model.primary_block_length).cloned(),
            "block_length_sensitivity":report_values,
            "normal_approximation_status":"posthoc_asymptotic_diagnostic_only_assumptions_unverified",
            "student_t_small_block_diagnostic_status":"not_implemented_v1",
            "not_anytime_valid":true
        },
        "limitations":[
            "The FDM bounds are caller-declared and checked for arithmetic/envelope consistency; the auditor does not derive the innovation coupling proof from physical traces or authenticate the model artifact.",
            "The non-asymptotic marginal coverage bound is conditional on stationarity, causal innovation representation, score-indicator FDM and the theorem's continuity/regularity assumptions; all are marked assumption_not_verified.",
            "The block standard error and normal interval are post-hoc asymptotic diagnostics, not finite-sample distribution-free intervals and not sequential e-values.",
            "An estimated autocorrelation function or a fitted AR parameter is not a certified upper bound on the functional dependence of threshold indicators.",
            "The output does not establish independence of calibration/evaluation executions, even when caller-declared execution and boot identifiers differ.",
            "No result from this auditor can be plugged into the HARDSEC-019/021 CCTM PFA guarantee without a separate proof of conditional super-uniformity or a valid supermartingale construction under the same dependence model."
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
    let mut model = None;
    let mut data = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let target = match flag.as_str() {
            "--model" => &mut model,
            "--data" => &mut data,
            "--receipt" => &mut receipt,
            _ => return Err(format!("unknown argument {flag}")),
        };
        let path = args.next().ok_or_else(|| format!("{flag} requires a path"))?;
        if target.replace(PathBuf::from(path)).is_some() {
            return Err(format!("duplicate argument {flag}"));
        }
    }
    Ok(Args {
        model:model.ok_or("missing required argument --model")?,
        data:data.ok_or("missing required argument --data")?,
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
    let (model_bytes, model) = read_input(&args.model, "model")?;
    let (data_bytes, data) = read_input(&args.data, "data")?;
    validate_receipt_location(&args.receipt, &[args.model.as_path(), args.data.as_path()])?;
    let receipt = audit_values(&model, &data, &model_bytes, &data_bytes)?;
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
            eprintln!("HARDSEC-022 temporal dependence audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Value, Value, Vec<u8>, Vec<u8>) {
        let model_bytes = include_bytes!("../tests/fixtures/hardsec/temporal-dependence-v1/model.json").to_vec();
        let data_bytes = include_bytes!("../tests/fixtures/hardsec/temporal-dependence-v1/data.json").to_vec();
        let model = serde_json::from_slice(&model_bytes).unwrap();
        let data = serde_json::from_slice(&data_bytes).unwrap();
        (model, data, model_bytes, data_bytes)
    }

    fn encode(value: &Value) -> Vec<u8> {
        let mut bytes = serde_json::to_vec_pretty(value).unwrap();
        bytes.push(b'\n');
        bytes
    }

    fn audit(model: &Value, data: &Value) -> Result<Value, String> {
        let mb = encode(model);
        let db = encode(data);
        audit_values(model, data, &mb, &db)
    }

    #[test]
    fn known_answer_reports_fdm_bound_and_block_sensitivity() {
        let (model,data,mb,db)=fixture();
        let receipt=audit_values(&model,&data,&mb,&db).unwrap();
        assert_eq!(receipt["result"],"audited");
        assert_eq!(receipt["calibration"]["score_count"],2000);
        assert_eq!(receipt["evaluation"]["score_count"],2000);
        assert!((receipt["calibration"]["fixed_cutoff"].as_f64().unwrap()-0.9).abs()<1e-12);
        assert!((receipt["evaluation"]["realized_coverage_at_fixed_cutoff"].as_f64().unwrap()-0.9).abs()<1e-12);
        let fdm=&receipt["functional_dependence"]["fdm_contract"];
        assert_eq!(fdm["status"],"model_bound_summable_fdm_contract_consistent_not_authenticated");
        assert!((fdm["theta_sum_upper_bound"].as_f64().unwrap()-0.25).abs()<1e-10);
        assert_eq!(fdm["proof_status"],"assumption_not_verified");
        assert_eq!(receipt["functional_dependence"]["sequential_ctm_validity_inferred"],false);
        assert_eq!(receipt["block_diagnostic"]["block_length_sensitivity"].as_array().unwrap().len(),3);
        assert_eq!(receipt["block_diagnostic"]["not_anytime_valid"],true);
        assert_eq!(receipt["functional_dependence"]["fdm_contract"]["marginal_coverage_bound_from_zhai_theorem_1"]["status"],"numeric_condition_satisfied_assumptions_unverified");
    }

    #[test]
    fn rejects_calibration_evaluation_execution_overlap() {
        let (mut model,data,_,_)=fixture();
        model["evaluation_execution_group_id"]=model["calibration_execution_group_id"].clone();
        assert!(audit(&model,&data).unwrap_err().contains("execution groups must be distinct"));
    }

    #[test]
    fn rejects_calibration_evaluation_boot_overlap() {
        let (mut model,data,_,_)=fixture();
        model["evaluation_device_boot_group_id"]=model["calibration_device_boot_group_id"].clone();
        assert!(audit(&model,&data).unwrap_err().contains("device-boot groups must be distinct"));
    }

    #[test]
    fn rejects_score_artifact_mismatch() {
        let (model,mut data,_,_)=fixture();
        data["scoring_artifact_sha256"]=json!("f".repeat(64));
        assert!(audit(&model,&data).unwrap_err().contains("scoring_artifact_sha256 does not match"));
    }

    #[test]
    fn rejects_unverified_claims_marked_as_verified() {
        let (mut model,data,_,_)=fixture();
        model["causal_innovation_model"]["iid_innovations_status"]=json!("verified");
        assert!(audit(&model,&data).unwrap_err().contains("iid_innovations_status"));
    }

    #[test]
    fn rejects_nondecreasing_or_gapped_lag_identity() {
        let (mut model,data,_,_)=fixture();
        model["dependence_bound"]["lag_upper_bounds"][2]["lag"]=json!(3);
        assert!(audit(&model,&data).unwrap_err().contains("contiguous from zero"));
    }

    #[test]
    fn rejects_per_lag_bound_above_geometric_envelope() {
        let (mut model,data,_,_)=fixture();
        model["dependence_bound"]["lag_upper_bounds"][2]["theta_upper_bound"]=json!(0.2);
        assert!(audit(&model,&data).unwrap_err().contains("exceeds declared geometric envelope"));
    }

    #[test]
    fn rejects_invalid_geometric_decay_parameter() {
        let (mut model,data,_,_)=fixture();
        model["dependence_bound"]["rho"]=json!(1.0);
        assert!(audit(&model,&data).unwrap_err().contains("rho in (0,1)"));
    }

    #[test]
    fn not_bounded_contract_does_not_emit_theta_or_coverage_bound() {
        let (mut model,data,_,_)=fixture();
        model["dependence_bound"]=json!({"status":"not_bounded"});
        let receipt=audit(&model,&data).unwrap();
        assert_eq!(receipt["functional_dependence"]["fdm_contract"]["status"],"not_bounded");
        assert_eq!(receipt["functional_dependence"]["fdm_contract"]["theta_sum_upper_bound"],Value::Null);
        assert_eq!(receipt["functional_dependence"]["fdm_contract"]["marginal_coverage_bound_from_zhai_theorem_1"],Value::Null);
    }

    #[test]
    fn rejects_not_bounded_contract_that_smuggles_an_envelope() {
        let (mut model,data,_,_)=fixture();
        model["dependence_bound"]=json!({"status":"not_bounded","lag_upper_bounds":[]});
        assert!(audit(&model,&data).unwrap_err().contains("must not supply a theorem-level envelope"));
    }

    #[test]
    fn rejects_block_length_with_fewer_than_two_complete_blocks() {
        let (mut model,data,_,_)=fixture();
        model["block_diagnostic"]["block_lengths"]=json!([2000]);
        model["block_diagnostic"]["primary_block_length"]=json!(2000);
        assert!(audit(&model,&data).unwrap_err().contains("fewer than two complete calibration blocks"));
    }

    #[test]
    fn rejects_duplicate_block_lengths() {
        let (mut model,data,_,_)=fixture();
        model["block_diagnostic"]["block_lengths"]=json!([25,25]);
        assert!(audit(&model,&data).unwrap_err().contains("positive and unique"));
    }

    #[test]
    fn rejects_empty_evaluation_scores() {
        let (model,mut data,_,_)=fixture();
        data["evaluation_scores"]=json!([]);
        assert!(audit(&model,&data).unwrap_err().contains("must not be empty"));
    }

    #[test]
    fn reports_iid_and_block_standard_errors_separately() {
        let (model,data,_,_)=fixture();
        let receipt=audit(&model,&data).unwrap();
        let reports=receipt["block_diagnostic"]["block_length_sensitivity"].as_array().unwrap();
        for report in reports {
            assert!(report["iid_standard_error"].as_f64().unwrap()>0.0);
            assert!(report["block_standard_error"].as_f64().unwrap()>0.0);
            assert!(report["block_count"].as_u64().unwrap()>=10);
        }
    }


    #[test]
    fn block_se_exposes_dependence_in_fixture_relative_to_iid_formula() {
        let (model,data,_,_)=fixture();
        let receipt=audit(&model,&data).unwrap();
        for report in receipt["block_diagnostic"]["block_length_sensitivity"].as_array().unwrap() {
            let block_se=report["block_standard_error"].as_f64().unwrap();
            let iid_se=report["iid_standard_error"].as_f64().unwrap();
            assert!(block_se > 2.0*iid_se, "block length {} failed to expose dependence", report["block_length"]);
            assert_eq!(report["observed_coverage_inside_null_interval"],true);
        }
    }

    #[test]
    fn fdm_marginal_coverage_bound_known_answer_is_separate_from_sequential_pfa() {
        let (model,data,_,_)=fixture();
        let receipt=audit(&model,&data).unwrap();
        let fdm=&receipt["functional_dependence"]["fdm_contract"];
        let coverage=&fdm["marginal_coverage_bound_from_zhai_theorem_1"];
        assert!((coverage["marginal_coverage_error_upper_bound"].as_f64().unwrap()-0.06052753944880749).abs()<1e-10);
        assert_eq!(receipt["functional_dependence"]["sequential_ctm_validity_inferred"],false);
    }

    #[test]
    fn normal_diagnostic_is_explicitly_non_anytime_valid() {
        let (model,data,_,_)=fixture();
        let receipt=audit(&model,&data).unwrap();
        assert_eq!(receipt["block_diagnostic"]["normal_approximation_status"],"posthoc_asymptotic_diagnostic_only_assumptions_unverified");
        assert_eq!(receipt["block_diagnostic"]["not_anytime_valid"],true);
        assert_eq!(receipt["functional_dependence"]["sequential_ctm_validity_inferred"],false);
    }

    #[test]
    fn rejects_wrong_stream_or_context_identity() {
        let (model,mut data,_,_)=fixture();
        data["context_id"]=json!("memory");
        assert!(audit(&model,&data).unwrap_err().contains("context_id does not match"));
    }

    #[test]
    fn rejects_alpha_outside_unit_interval() {
        let (mut model,data,_,_)=fixture();
        model["target_miscoverage_alpha"]=json!(0.0);
        assert!(audit(&model,&data).unwrap_err().contains("must be in (0,1)"));
    }

    #[test]
    fn rejects_model_scoring_digest_with_invalid_format() {
        let (mut model,data,_,_)=fixture();
        model["scoring_artifact_sha256"]=json!("not-a-sha");
        assert!(audit(&model,&data).unwrap_err().contains("lowercase SHA-256"));
    }

    #[test]
    fn rejects_lag_bound_outside_indicator_norm_range() {
        let (mut model,data,_,_)=fixture();
        model["dependence_bound"]["lag_upper_bounds"][0]["theta_upper_bound"]=json!(1.1);
        assert!(audit(&model,&data).unwrap_err().contains("must be in [0,1]"));
    }

    #[test]
    fn reports_numerical_fdm_condition_separately_from_assumption_status() {
        let (model,data,_,_)=fixture();
        let receipt=audit(&model,&data).unwrap();
        let bound=&receipt["functional_dependence"]["fdm_contract"]["marginal_coverage_bound_from_zhai_theorem_1"];
        assert!(bound["assumption_condition_value"].as_f64().unwrap()<bound["assumption_condition_threshold"].as_f64().unwrap());
        assert_eq!(receipt["functional_dependence"]["iid_innovations_status"],"assumption_not_verified");
    }
}
