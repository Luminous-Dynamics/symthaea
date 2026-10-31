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
const GROUP_BOOTSTRAP_REPLICATES: usize = 10_000;
const MIN_GROUPS_FOR_BOOTSTRAP_INTERVAL: usize = 10;

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
    baseline_predictions: Option<PathBuf>,
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
        if (meta.adaptation_family_id == "none" && meta.adaptation_level != 0)
            || (meta.adaptation_family_id != "none" && meta.adaptation_level == 0) {
            return Err(format!("{ctx}: adaptation_family_id=none requires level 0; a named adaptation family requires a positive level"));
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

fn next_bootstrap_u64(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    value.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// Deterministic equal-group percentile bootstrap. The seed deliberately
/// depends only on campaign ID, grouping axis and algorithm version, so
/// matched model arms with the same ordered group IDs use common resamples.
/// Finite-sample two-sided Hoeffding interval for a mean of independent
/// bounded group-level observations. This is conservative but has a
/// distribution-free coverage guarantee conditional on group independence.
fn hoeffding_mean_interval(
    values: &[f64],
    lower_bound: f64,
    upper_bound: f64,
    alpha: f64,
    assumption: &str,
) -> Value {
    if values.is_empty() || !(lower_bound < upper_bound) || !(0.0 < alpha && alpha < 1.0) {
        return json!({
            "method": "two_sided_hoeffding_bounded_group_mean_v1",
            "status": "unavailable_invalid_or_empty_support",
            "group_count": values.len(),
            "lower": null,
            "upper": null
        });
    }
    if values.iter().any(|value| !value.is_finite() || *value < lower_bound || *value > upper_bound) {
        return json!({
            "method": "two_sided_hoeffding_bounded_group_mean_v1",
            "status": "unavailable_values_outside_declared_bounds",
            "group_count": values.len(),
            "lower": null,
            "upper": null
        });
    }
    let count = values.len();
    let point = values.iter().sum::<f64>() / count as f64;
    let epsilon = (upper_bound - lower_bound)
        * ((2.0 / alpha).ln() / (2.0 * count as f64)).sqrt();
    json!({
        "method": "two_sided_hoeffding_bounded_group_mean_v1",
        "status": "available_finite_sample_bound",
        "point_estimate": point,
        "group_count": count,
        "confidence_level": 1.0 - alpha,
        "lower": (point - epsilon).max(lower_bound),
        "upper": (point + epsilon).min(upper_bound),
        "declared_value_bounds": [lower_bound, upper_bound],
        "epsilon": epsilon,
        "assumption": assumption,
        "interpretation": "Finite-sample distribution-free bound for the mean of independent bounded group-level observations. Independence/exchangeability of caller-declared groups is not proven by this evaluator."
    })
}

fn group_bootstrap_interval(
    groups: &BTreeMap<String, RecallCounts>,
    campaign_id: &str,
    axis: &str,
) -> Value {
    let values: Vec<f64> = groups.values().filter_map(recall).collect();
    let group_count = values.len();
    let point_estimate = if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / group_count as f64)
    };
    let seed_material = format!("{campaign_id}|{axis}|hardsec-group-bootstrap-v1");
    let seed_digest = Sha256::digest(seed_material.as_bytes());
    let seed_sha256 = seed_digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    let hoeffding = hoeffding_mean_interval(
        &values,
        0.0,
        1.0,
        0.05,
        "caller-declared independent group-level rates bounded to [0,1]",
    );

    if group_count < MIN_GROUPS_FOR_BOOTSTRAP_INTERVAL {
        return json!({
            "method": "deterministic_equal_group_percentile_bootstrap_v1",
            "status": "unavailable_insufficient_independent_groups",
            "point_estimate": point_estimate,
            "independent_group_count": group_count,
            "minimum_independent_groups": MIN_GROUPS_FOR_BOOTSTRAP_INTERVAL,
            "bootstrap_replicates": GROUP_BOOTSTRAP_REPLICATES,
            "nominal_percentile_level": 0.95,
            "lower_percentile_95": null,
            "upper_percentile_95": null,
            "seed_sha256": seed_sha256,
            "seed_scope": "campaign_id + grouping_axis + algorithm_version",
            "resampling_unit": "caller_declared_independent_group",
            "assumption": "group identifiers encode independent exchangeable units",
            "coverage_guarantee": false,
            "interpretation": "No percentile-bootstrap interval is reported below the minimum group count; this is a descriptive resampling interval, not a finite-sample coverage guarantee. The separate Hoeffding bound is the conservative distribution-free interval conditional on independent bounded groups.",
            "distribution_free_hoeffding_95": hoeffding
        });
    }

    let mut state = u64::from_le_bytes(
        seed_digest[..8]
            .try_into()
            .expect("SHA-256 has at least eight bytes"),
    );
    if state == 0 {
        state = 0x9E37_79B9_7F4A_7C15;
    }
    let mut bootstrap_means = Vec::with_capacity(GROUP_BOOTSTRAP_REPLICATES);
    for _ in 0..GROUP_BOOTSTRAP_REPLICATES {
        let mut sum = 0.0f64;
        for _ in 0..group_count {
            let draw = next_bootstrap_u64(&mut state);
            let index = ((draw as u128 * group_count as u128) >> 64) as usize;
            sum += values[index];
        }
        bootstrap_means.push(sum / group_count as f64);
    }
    bootstrap_means.sort_by(f64::total_cmp);
    let lower_index = (((GROUP_BOOTSTRAP_REPLICATES - 1) as f64) * 0.025).floor() as usize;
    let upper_index = (((GROUP_BOOTSTRAP_REPLICATES - 1) as f64) * 0.975).ceil() as usize;

    json!({
        "method": "deterministic_equal_group_percentile_bootstrap_v1",
        "status": "available_descriptive_interval",
        "point_estimate": point_estimate,
        "independent_group_count": group_count,
        "minimum_independent_groups": MIN_GROUPS_FOR_BOOTSTRAP_INTERVAL,
        "bootstrap_replicates": GROUP_BOOTSTRAP_REPLICATES,
        "nominal_percentile_level": 0.95,
        "lower_percentile_95": bootstrap_means[lower_index],
        "upper_percentile_95": bootstrap_means[upper_index],
        "quantile_rule": "lower_floor_upper_ceil_order_statistic",
        "coverage_guarantee": false,
        "seed_sha256": seed_sha256,
        "seed_scope": "campaign_id + grouping_axis + algorithm_version",
        "resampling_unit": "caller_declared_independent_group",
        "assumption": "group identifiers encode independent exchangeable units",
        "interpretation": "Descriptive cluster-bootstrap percentile range only; it is not a finite-sample 95% coverage guarantee. Use the separate Hoeffding bound for a conservative distribution-free interval conditional on independent bounded groups.",
        "distribution_free_hoeffding_95": hoeffding
    })
}

fn binomial_lower_tail(k: usize, n: usize, probability: f64) -> f64 {
    if k >= n {
        return 1.0;
    }
    if k == 0 {
        return (1.0 - probability).powf(n as f64);
    }
    let mut log_choose = 0.0f64;
    for i in 1..=k {
        log_choose += ((n - k + i) as f64).ln() - (i as f64).ln();
    }
    let log_mass = log_choose
        + (k as f64) * probability.ln()
        + ((n - k) as f64) * (-probability).ln_1p();
    let mut term = log_mass.exp();
    let mut total = term;
    for i in (1..=k).rev() {
        term *= (i as f64 / (n - i + 1) as f64) * ((1.0 - probability) / probability);
        total += term;
        if term <= total * 1e-16 {
            break;
        }
    }
    total.min(1.0)
}

/// Exact one-sided Clopper-Pearson upper endpoint, obtained by inverting
/// P[X <= observed_failures] = alpha for X ~ Binomial(n, p).
fn clopper_pearson_upper(observed_failures: usize, groups: usize, alpha: f64) -> Option<f64> {
    if groups == 0 || observed_failures > groups || !(0.0 < alpha && alpha < 1.0) {
        return None;
    }
    if observed_failures == groups {
        return Some(1.0);
    }
    if observed_failures == 0 {
        return Some(1.0 - alpha.powf(1.0 / groups as f64));
    }
    let mut lower = observed_failures as f64 / groups as f64;
    let mut upper = 1.0f64;
    for _ in 0..80 {
        let midpoint = (lower + upper) * 0.5;
        if binomial_lower_tail(observed_failures, groups, midpoint) > alpha {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    Some(upper)
}

fn group_failure_bound(
    groups: &BTreeMap<String, RecallCounts>,
    endpoint: &str,
    failure_semantics: &str,
) -> Value {
    let group_count = groups.len();
    let failures = groups.values().filter(|stats| match failure_semantics {
        "any_attack_miss" => stats.cleared + stats.abstained > 0,
        "any_benign_false_alert" => stats.detected > 0,
        _ => unreachable!("group failure semantics fixed by caller"),
    }).count();
    let familywise_alpha = 0.05f64;
    let adjusted_alpha = familywise_alpha / 6.0;
    let upper_bound = clopper_pearson_upper(failures, group_count, adjusted_alpha);
    json!({
        "endpoint": endpoint,
        "failure_semantics": failure_semantics,
        "status": if upper_bound.is_some() { "available_conditional_group_bound" } else { "unavailable_no_independent_groups" },
        "groups_with_any_failure": failures,
        "independent_group_count": group_count,
        "observed_group_failure_fraction": if group_count == 0 { None } else { Some(failures as f64 / group_count as f64) },
        "upper_confidence_bound": upper_bound,
        "confidence_level_per_bound": 1.0 - adjusted_alpha,
        "familywise_alpha": familywise_alpha,
        "bonferroni_endpoint_count": 6,
        "adjusted_alpha_per_endpoint": adjusted_alpha,
        "bernoulli_unit": "caller_declared_independent_group",
        "assumption": "groups are independent and exchangeable for this endpoint",
        "interpretation": "Bound concerns the probability a new group contains at least one defined failure; it is not a per-case FNR/FPR and does not authenticate group independence."
    })
}

fn group_metric_map(
    receipt: &Value,
    section: &str,
    group_id_key: &str,
    metric_key: &str,
    context: &str,
) -> Result<BTreeMap<String, f64>, String> {
    let rows = receipt.get("generalization")
        .and_then(|value| value.get(section))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{context} group rows are unavailable"))?;
    let mut output = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        let row_context = format!("{context}[{index}]");
        let group_id = str_field(row, group_id_key, &row_context)?.to_owned();
        let metric = row.get(metric_key).and_then(Value::as_f64)
            .ok_or_else(|| format!("{row_context}.{metric_key} must be numeric"))?;
        if !metric.is_finite() || !(0.0..=1.0).contains(&metric) {
            return Err(format!("{row_context}.{metric_key} must be finite and within [0, 1]"));
        }
        if output.insert(group_id.clone(), metric).is_some() {
            return Err(format!("{context} contains duplicate group ID {group_id}"));
        }
    }
    Ok(output)
}

fn paired_group_metric_delta(
    candidate: &BTreeMap<String, f64>,
    baseline: &BTreeMap<String, f64>,
    campaign_id: &str,
    axis: &str,
    metric: &str,
    higher_is_better: bool,
) -> Result<Value, String> {
    let candidate_ids: BTreeSet<&String> = candidate.keys().collect();
    let baseline_ids: BTreeSet<&String> = baseline.keys().collect();
    if candidate_ids != baseline_ids {
        return Err(format!(
            "paired comparison requires identical {axis} group IDs; candidate={} baseline={}",
            candidate_ids.len(),
            baseline_ids.len()
        ));
    }
    if candidate.is_empty() {
        return Err(format!("paired comparison has no {axis} groups"));
    }

    let mut deltas = Vec::with_capacity(candidate.len());
    let mut by_group = Vec::with_capacity(candidate.len());
    for (group_id, candidate_value) in candidate {
        let baseline_value = baseline.get(group_id)
            .ok_or_else(|| format!("baseline missing paired {axis} group {group_id}"))?;
        let improvement = if higher_is_better {
            candidate_value - baseline_value
        } else {
            baseline_value - candidate_value
        };
        deltas.push(improvement);
        by_group.push(json!({
            "group_id": group_id,
            "candidate_value": candidate_value,
            "baseline_value": baseline_value,
            "improvement_delta": improvement
        }));
    }

    let group_count = deltas.len();
    let point = deltas.iter().sum::<f64>() / group_count as f64;
    let seed_material = format!(
        "{campaign_id}|{axis}|{metric}|hardsec-paired-group-bootstrap-v1"
    );
    let seed_digest = Sha256::digest(seed_material.as_bytes());
    let seed_sha256 = seed_digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    let hoeffding = hoeffding_mean_interval(
        &deltas,
        -1.0,
        1.0,
        0.05,
        "paired candidate-minus-baseline group deltas bounded to [-1,1], independent across declared groups",
    );

    let (lower, upper, status) = if group_count < MIN_GROUPS_FOR_BOOTSTRAP_INTERVAL {
        (None, None, "unavailable_insufficient_independent_groups")
    } else {
        let mut state = u64::from_le_bytes(
            seed_digest[..8]
                .try_into()
                .expect("SHA-256 has at least eight bytes"),
        );
        if state == 0 {
            state = 0x9E37_79B9_7F4A_7C15;
        }
        let mut samples = Vec::with_capacity(GROUP_BOOTSTRAP_REPLICATES);
        for _ in 0..GROUP_BOOTSTRAP_REPLICATES {
            let mut sum = 0.0f64;
            for _ in 0..group_count {
                let draw = next_bootstrap_u64(&mut state);
                let index = ((draw as u128 * group_count as u128) >> 64) as usize;
                sum += deltas[index];
            }
            samples.push(sum / group_count as f64);
        }
        samples.sort_by(f64::total_cmp);
        let lower_index = (((GROUP_BOOTSTRAP_REPLICATES - 1) as f64) * 0.025).floor() as usize;
        let upper_index = (((GROUP_BOOTSTRAP_REPLICATES - 1) as f64) * 0.975).ceil() as usize;
        (
            Some(samples[lower_index]),
            Some(samples[upper_index]),
            "available_paired_group_bootstrap_interval",
        )
    };

    let candidate_macro = candidate.values().sum::<f64>() / group_count as f64;
    let baseline_macro = baseline.values().sum::<f64>() / group_count as f64;
    Ok(json!({
        "metric": metric,
        "direction": if higher_is_better { "higher_is_better" } else { "lower_is_better" },
        "status": status,
        "independent_group_count": group_count,
        "candidate_macro_estimate": candidate_macro,
        "baseline_macro_estimate": baseline_macro,
        "improvement_point_estimate": point,
        "improvement_lower_percentile_95": lower,
        "improvement_upper_percentile_95": upper,
        "bootstrap_coverage_guarantee": false,
        "bootstrap_interval_type": "descriptive_percentile_range_no_coverage_guarantee",
        "distribution_free_hoeffding_95": hoeffding,
        "bootstrap_replicates": GROUP_BOOTSTRAP_REPLICATES,
        "minimum_independent_groups": MIN_GROUPS_FOR_BOOTSTRAP_INTERVAL,
        "seed_sha256": seed_sha256,
        "seed_scope": "campaign_id + grouping_axis + metric + algorithm_version",
        "resampling_unit": "paired_independent_group_delta",
        "assumption": "candidate and baseline values are paired on identical caller-declared independent exchangeable groups",
        "interpretation": "Positive improvement favors the candidate. The percentile-bootstrap interval is descriptive only; the separate Hoeffding bound is finite-sample distribution-free conditional on independent paired groups.",
        "by_group": by_group
    }))
}

fn paired_model_comparison(
    candidate_receipt: &Value,
    baseline_receipt: &Value,
    campaign_id: &str,
    baseline_predictions_sha256: &str,
) -> Result<Value, String> {
    let candidate_campaign = candidate_receipt.get("campaign")
        .ok_or("candidate receipt campaign section missing")?;
    let baseline_campaign = baseline_receipt.get("campaign")
        .ok_or("baseline receipt campaign section missing")?;
    if str_field(candidate_campaign, "campaign_id", "candidate.campaign")? != campaign_id
        || str_field(baseline_campaign, "campaign_id", "baseline.campaign")? != campaign_id
    {
        return Err("candidate and baseline campaign IDs must match".to_owned());
    }
    let candidate_model = str_field(candidate_campaign, "model_id", "candidate.campaign")?;
    let baseline_model = str_field(baseline_campaign, "model_id", "baseline.campaign")?;
    let candidate_domain = str_field(candidate_campaign, "evidence_domain", "candidate.campaign")?;
    let baseline_domain = str_field(baseline_campaign, "evidence_domain", "baseline.campaign")?;
    let candidate_model_sha256 = str_field(candidate_campaign, "model_sha256", "candidate.campaign")?;
    let baseline_model_sha256 = str_field(baseline_campaign, "model_sha256", "baseline.campaign")?;
    let candidate_threshold_sha256 = str_field(candidate_campaign, "threshold_profile_sha256", "candidate.campaign")?;
    let baseline_threshold_sha256 = str_field(baseline_campaign, "threshold_profile_sha256", "baseline.campaign")?;
    if candidate_domain != baseline_domain {
        return Err("paired candidate/baseline evidence domains must match".to_owned());
    }

    let comparisons = [
        (
            "attack_family_level",
            "attack_family_id",
            "attack_effect_recall",
            true,
        ),
        (
            "adaptation_family_level",
            "adaptation_family_and_level",
            "attack_effect_recall",
            true,
        ),
        (
            "adaptation_family",
            "adaptation_family_id",
            "attack_effect_recall",
            true,
        ),
        (
            "program_group_level",
            "program_group_id",
            "attack_effect_recall",
            true,
        ),
        (
            "execution_group_level",
            "execution_group_id",
            "attack_effect_recall",
            true,
        ),
        (
            "benign_false_alert_by_program",
            "program_group_id",
            "false_alert_rate",
            false,
        ),
    ];
    let mut group_metrics = serde_json::Map::new();
    for (section, group_key, metric_key, higher_is_better) in comparisons {
        let candidate_values = group_metric_map(
            candidate_receipt,
            section,
            group_key,
            metric_key,
            &format!("candidate.{section}"),
        )?;
        let baseline_values = group_metric_map(
            baseline_receipt,
            section,
            group_key,
            metric_key,
            &format!("baseline.{section}"),
        )?;
        let delta = paired_group_metric_delta(
            &candidate_values,
            &baseline_values,
            campaign_id,
            section,
            metric_key,
            higher_is_better,
        )?;
        group_metrics.insert(section.to_owned(), delta);
    }
    Ok(json!({
        "schema_version": "hardsec-paired-model-comparison-v1",
        "result": "paired",
        "campaign_id": campaign_id,
        "evidence_domain": candidate_domain,
        "candidate_model_id": candidate_model,
        "candidate_model_sha256": candidate_model_sha256,
        "baseline_model_id": baseline_model,
        "baseline_model_sha256": baseline_model_sha256,
        "candidate_predictions_sha256": candidate_receipt.get("inputs").and_then(|v| v.get("predictions_sha256")),
        "baseline_predictions_sha256": baseline_predictions_sha256,
        "candidate_threshold_profile_sha256": candidate_threshold_sha256,
        "baseline_threshold_profile_sha256": baseline_threshold_sha256,
        "threshold_profile_digests_equal": candidate_threshold_sha256 == baseline_threshold_sha256,
        "operating_point_comparability": {
            "status": "not_verified_by_paired_ledger",
            "matched_false_alert_budget_verified": false,
            "required_before_superiority_claim": "Freeze both thresholds using validation groups only; predeclare an alert budget; verify that both candidate and baseline are evaluated at the same operating constraint on held-out groups. Matching threshold digests alone is not proof of matched false-alert burden."
        },
        "pairing_rule": "same manifest, oracle, test case IDs and exact ordered independent group IDs for each metric",
        "decision": "descriptive_paired_comparison_no_superiority_claim",
        "group_metrics": group_metrics,
        "limitations": [
            "This compares the supplied prediction ledgers on the same cases and group IDs; it does not authenticate either model or its threshold selection.",
            "Group bootstrap intervals are descriptive and conditional on caller-declared independent/exchangeable groups.",
            "A positive paired delta is not a SOTA claim; matched platform, tuning, acquisition and detector operating budgets are separate obligations."
        ]
    }))
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
            if matches!(axis.as_str(), "adaptation_family" | "attack_family") && group_value == "none" {
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
    let mut valid_benign_alerts = 0usize;
    let mut valid_benign_clears = 0usize;
    let mut invalid_benign_cases = 0usize;
    let mut all_alerts = 0usize;
    let mut all_abstentions = 0usize;
    let mut invalid_sensor_cases = 0usize;
    let mut invalid_sensor_clear_verdicts = 0usize;

    let mut attack_family_groups: BTreeMap<String, RecallCounts> = BTreeMap::new();
    let mut adaptation_groups: BTreeMap<String, RecallCounts> = BTreeMap::new();
    let mut adaptation_family_groups: BTreeMap<String, RecallCounts> = BTreeMap::new();
    let mut program_groups: BTreeMap<String, RecallCounts> = BTreeMap::new();
    let mut execution_groups: BTreeMap<String, RecallCounts> = BTreeMap::new();
    let mut benign_by_program: BTreeMap<String, RecallCounts> = BTreeMap::new();

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
                    if meta.attack_family_id == "none" {
                        return Err(format!("malicious observed-effect case {case_id} cannot use attack_family_id=none"));
                    }
                    success_attack_cases += 1;
                    let key = format!("{}|{}", meta.adaptation_family_id, meta.adaptation_level);
                    add_recall(attack_family_groups.entry(meta.attack_family_id.clone()).or_default(), &prediction.response);
                    add_recall(adaptation_groups.entry(key).or_default(), &prediction.response);
                    add_recall(adaptation_family_groups.entry(meta.adaptation_family_id.clone()).or_default(), &prediction.response);
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
                    if meta.attack_family_id == "none" {
                        return Err(format!("malicious failed-attempt case {case_id} cannot use attack_family_id=none"));
                    }
                    failed_attack_attempts += 1;
                    if prediction.response == "alert" {
                        failed_attack_alerts += 1;
                    }
                }
                "unknown" => {
                    if meta.attack_family_id == "none" {
                        return Err(format!("malicious unknown-effect case {case_id} cannot use attack_family_id=none"));
                    }
                    unknown_effect_attempts += 1;
                    if prediction.response == "alert" {
                        unknown_effect_alerts += 1;
                    }
                }
                _ => return Err(format!("invalid malicious attack_effect for {case_id}")),
            },
            "benign" => {
                if meta.attack_family_id != "none" || meta.adaptation_family_id != "none" || meta.adaptation_level != 0 {
                    return Err(format!("benign case {case_id} must use attack/adaptation family none and level 0"));
                }
                benign_cases += 1;
                if invalid_sensor {
                    invalid_benign_cases += 1;
                } else {
                    valid_benign_cases += 1;
                }
                add_recall(benign_by_program.entry(meta.program_group_id.clone()).or_default(), &prediction.response);
                match prediction.response.as_str() {
                    "alert" => {
                        benign_alerts += 1;
                        if !invalid_sensor {
                            valid_benign_alerts += 1;
                        }
                    }
                    "clear" => {
                        benign_clears += 1;
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

    let (attack_family_macro, attack_family_worst, attack_family_count) = macro_recall(&attack_family_groups);
    let (adaptation_macro, adaptation_worst, adaptation_group_count) = macro_recall(&adaptation_groups);
    let (adaptation_family_macro, adaptation_family_worst, adaptation_family_count) = macro_recall(&adaptation_family_groups);

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
    let benign_false_alert_rate_valid_sensor = if valid_benign_cases == 0 { None } else {
        Some(valid_benign_alerts as f64 / valid_benign_cases as f64)
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

    let by_attack_family: Vec<Value> = attack_family_groups.iter().map(|(group, stats)| json!({
        "attack_family_id": group,
        "successful_attack_cases": stats.cases,
        "detected": stats.detected,
        "cleared": stats.cleared,
        "abstained": stats.abstained,
        "attack_effect_recall": recall(stats)
    })).collect();
    let by_adaptation_family: Vec<Value> = adaptation_family_groups.iter().map(|(group, stats)| json!({
        "adaptation_family_id": group,
        "successful_attack_cases": stats.cases,
        "detected": stats.detected,
        "cleared": stats.cleared,
        "abstained": stats.abstained,
        "attack_effect_recall": recall(stats)
    })).collect();
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
    let benign_groups: Vec<Value> = benign_by_program.iter().map(|(group, stats)| json!({
        "program_group_id": group,
        "benign_cases": stats.cases,
        "false_alerts": stats.detected,
        "clear_verdicts": stats.cleared,
        "abstentions": stats.abstained,
        "false_alert_rate": recall(stats)
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
            "benign_false_alert_rate_on_valid_sensor_cases": benign_false_alert_rate_valid_sensor,
            "benign_true_negative_rate_on_valid_sensor_cases": benign_true_negative_rate,
            "benign_valid_sensor_cases": valid_benign_cases,
            "benign_invalid_sensor_cases": invalid_benign_cases,
            "conservative_effect_confirmed_alert_precision": conservative_alert_precision,
            "all_alerts": all_alerts,
            "abstention_rate": abstention_rate
        },
        "generalization": {
            "attack_family_level": by_attack_family,
            "attack_family_macro_recall": attack_family_macro,
            "worst_attack_family_recall": attack_family_worst,
            "attack_family_group_count": attack_family_count,
            "adaptation_family_level": by_adaptation,
            "adaptation_family_level_macro_recall": adaptation_macro,
            "worst_adaptation_family_level_recall": adaptation_worst,
            "adaptation_family_level_group_count": adaptation_group_count,
            "adaptation_family": by_adaptation_family,
            "adaptation_family_macro_recall": adaptation_family_macro,
            "worst_adaptation_family_recall": adaptation_family_worst,
            "adaptation_family_count": adaptation_family_count,
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
        "statistical_support": {
            "method": "deterministic_equal_group_percentile_bootstrap_v1",
            "replicates": GROUP_BOOTSTRAP_REPLICATES,
            "minimum_independent_groups": MIN_GROUPS_FOR_BOOTSTRAP_INTERVAL,
            "common_random_numbers": "seed depends on campaign_id + grouping_axis + algorithm_version, not prediction bytes",
            "assumption": "caller-declared group identifiers represent independent exchangeable units",
            "warning": "Intervals are descriptive and are not simultaneous across axes; pre-register one primary axis or apply an explicit multiplicity procedure.",
            "attack_family_macro_recall": group_bootstrap_interval(&attack_family_groups, &prediction_campaign, "attack_family"),
            "adaptation_family_level_macro_recall": group_bootstrap_interval(&adaptation_groups, &prediction_campaign, "adaptation_family_level"),
            "adaptation_family_macro_recall": group_bootstrap_interval(&adaptation_family_groups, &prediction_campaign, "adaptation_family"),
            "program_macro_recall": group_bootstrap_interval(&program_groups, &prediction_campaign, "program"),
            "execution_macro_recall": group_bootstrap_interval(&execution_groups, &prediction_campaign, "execution"),
            "benign_program_macro_false_alert_rate": group_bootstrap_interval(&benign_by_program, &prediction_campaign, "benign_program"),
            "group_failure_bounds": {
                "attack_family": group_failure_bound(&attack_family_groups, "attack_family", "any_attack_miss"),
                "adaptation_family_level": group_failure_bound(&adaptation_groups, "adaptation_family_level", "any_attack_miss"),
                "adaptation_family": group_failure_bound(&adaptation_family_groups, "adaptation_family", "any_attack_miss"),
                "program": group_failure_bound(&program_groups, "program", "any_attack_miss"),
                "execution": group_failure_bound(&execution_groups, "execution", "any_attack_miss"),
                "benign_program": group_failure_bound(&benign_by_program, "benign_program", "any_benign_false_alert")
            }
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
        if !matches!(flag.as_str(), "--predictions" | "--baseline-predictions" | "--manifest" | "--truth" | "--receipt") {
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
        baseline_predictions: values.get("--baseline-predictions").cloned(),
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
    let mut input_paths = vec![args.predictions.as_path(), args.manifest.as_path(), args.truth.as_path()];
    if let Some(baseline_path) = args.baseline_predictions.as_ref() {
        input_paths.push(baseline_path.as_path());
    }
    validate_receipt_location(&args.receipt, &input_paths)?;
    let mut receipt = audit_values(
        &predictions, &manifest, &truth,
        &predictions_bytes, &manifest_bytes, &truth_bytes,
    )?;
    if let Some(baseline_path) = args.baseline_predictions.as_ref() {
        let (baseline_bytes, baseline_predictions) = read_input(baseline_path, "baseline predictions")?;
        let baseline_campaign_id = prediction_campaign_id(&baseline_predictions)?;
        let candidate_campaign_id = prediction_campaign_id(&predictions)?;
        if baseline_campaign_id != candidate_campaign_id {
            return Err("candidate and baseline prediction campaign IDs must match".to_owned());
        }
        let baseline_receipt = audit_values(
            &baseline_predictions,
            &manifest,
            &truth,
            &baseline_bytes,
            &manifest_bytes,
            &truth_bytes,
        )?;
        let comparison = paired_model_comparison(
            &receipt,
            &baseline_receipt,
            candidate_campaign_id,
            &sha256(&baseline_bytes),
        )?;
        receipt["paired_baseline_comparison"] = comparison;
    }
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
                {"case_id":"tr1","split":"train","program_group_id":"train-p","execution_group_id":"train-e","device_config_id":"cpu-a","attack_family_id":"spectre-baseline-train","adaptation_family_id":"none","adaptation_level":0,"sensor_state":"valid"},
                {"case_id":"va1","split":"validation","program_group_id":"val-p","execution_group_id":"val-e","device_config_id":"cpu-a","attack_family_id":"spectre-baseline-validation","adaptation_family_id":"none","adaptation_level":0,"sensor_state":"valid"},
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
        let attack_recall = metrics["attack_effect_conditioned_recall"].as_f64().unwrap();
        assert!((attack_recall - 1.0/3.0).abs() < 1e-12);
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
    fn rejects_attack_family_holdout_leakage() {
        let (p,mut m,t,pb,_,tb) = fixture_values();
        m["cases"][0]["attack_family_id"] = json!("spectre-evasive");
        let mb = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap_err().contains("attack_family holdout"));
    }

    #[test]
    fn rejects_adaptation_family_holdout_leakage() {
        let (p,mut m,t,pb,_,tb) = fixture_values();
        m["cases"][0]["adaptation_family_id"] = json!("timing");
        m["cases"][0]["adaptation_level"] = json!(1);
        let mb = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap_err().contains("adaptation_family holdout"));
    }

    #[test]
    fn rejects_malicious_case_with_none_attack_family() {
        let (p,mut m,t,pb,_,tb) = fixture_values();
        m["cases"][2]["attack_family_id"] = json!("none");
        let mb = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap_err().contains("cannot use attack_family_id=none"));
    }

    #[test]
    fn rejects_benign_case_tagged_with_attack_family() {
        let (p,mut m,t,pb,_,tb) = fixture_values();
        m["cases"][5]["attack_family_id"] = json!("evasive-spectre");
        let mb = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap_err().contains("benign case"));
    }

    #[test]
    fn rejects_named_adaptation_family_at_zero_level() {
        let (p,mut m,t,pb,_,tb) = fixture_values();
        m["cases"][2]["adaptation_level"] = json!(0);
        let mb = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap_err().contains("named adaptation family requires a positive level"));
    }

    #[test]
    fn reports_attack_family_macro_recall() {
        let (p,m,t,pb,mb,tb) = fixture_values();
        let receipt = audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap();
        assert_eq!(receipt["generalization"]["attack_family_group_count"], 1);
        assert_eq!(receipt["generalization"]["attack_family_level"].as_array().unwrap().len(), 1);
        assert_eq!(receipt["generalization"]["attack_family_macro_recall"], 1.0/3.0);
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
        assert_eq!(receipt["generalization"]["adaptation_family_level_group_count"], 3);
        assert_eq!(receipt["generalization"]["adaptation_family_level"].as_array().unwrap().len(), 3);
        assert_eq!(receipt["generalization"]["adaptation_family_count"], 2);
        assert_eq!(receipt["generalization"]["adaptation_family"].as_array().unwrap().len(), 2);
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
        assert_eq!(receipt["primary_metrics"]["benign_false_alert_rate_on_valid_sensor_cases"], 0.5);
        assert_eq!(receipt["primary_metrics"]["benign_true_negative_rate_on_valid_sensor_cases"], 0.5);
    }

    fn test_binomial_pmf(k: usize, n: usize, probability: f64) -> f64 {
        if probability == 0.0 {
            return if k == 0 { 1.0 } else { 0.0 };
        }
        if probability == 1.0 {
            return if k == n { 1.0 } else { 0.0 };
        }
        if k > n {
            return 0.0;
        }
        let terms = k.min(n - k);
        let mut log_choose = 0.0f64;
        for index in 1..=terms {
            log_choose += ((n - terms + index) as f64).ln() - (index as f64).ln();
        }
        (log_choose
            + (k as f64) * probability.ln()
            + ((n - k) as f64) * (-probability).ln_1p())
            .exp()
    }

    fn test_binomial_quantile(n: usize, probability: f64, quantile: f64) -> usize {
        (0..=n)
            .find(|count| binomial_lower_tail(*count, n, probability) >= quantile)
            .unwrap_or(n)
    }

    fn exact_binary_percentile_bootstrap_coverage(n: usize, truth: f64) -> f64 {
        let mut coverage = 0.0f64;
        for observed_successes in 0..=n {
            let observed_rate = observed_successes as f64 / n as f64;
            let lower = test_binomial_quantile(n, observed_rate, 0.025) as f64 / n as f64;
            let upper = test_binomial_quantile(n, observed_rate, 0.975) as f64 / n as f64;
            if lower <= truth && truth <= upper {
                coverage += test_binomial_pmf(observed_successes, n, truth);
            }
        }
        coverage
    }

    #[test]
    fn percentile_bootstrap_undercoverage_is_detected_at_boundary_rates() {
        // Exact enumeration of the Bernoulli-group sampling distribution;
        // this is not a Monte Carlo approximation.
        assert!((exact_binary_percentile_bootstrap_coverage(10, 0.05) - 0.4002345628237306).abs() < 1e-12);
        assert!((exact_binary_percentile_bootstrap_coverage(10, 0.10) - 0.6496866225).abs() < 1e-12);
        assert!((exact_binary_percentile_bootstrap_coverage(10, 0.50) - 0.978515625).abs() < 1e-12);
        assert!((exact_binary_percentile_bootstrap_coverage(10, 0.90) - 0.6496866225).abs() < 1e-12);
        assert!((exact_binary_percentile_bootstrap_coverage(10, 0.95) - 0.4002345628237306).abs() < 1e-12);
    }

    #[test]
    fn group_bootstrap_with_fewer_than_ten_groups_is_unavailable() {
        let mut groups = BTreeMap::new();
        for index in 0..9 {
            groups.insert(format!("g{index}"), RecallCounts {
                cases: 10,
                detected: 8,
                cleared: 2,
                abstained: 0,
            });
        }
        let interval = group_bootstrap_interval(&groups, "campaign-a", "program");
        assert_eq!(interval["status"], "unavailable_insufficient_independent_groups");
        assert_eq!(interval["independent_group_count"], 9);
        assert_eq!(interval["lower_percentile_95"], Value::Null);
        assert_eq!(interval["upper_percentile_95"], Value::Null);
        assert_eq!(interval["distribution_free_hoeffding_95"]["status"], "available_finite_sample_bound");
        assert_eq!(interval["distribution_free_hoeffding_95"]["group_count"], 9);
        assert_eq!(interval["coverage_guarantee"], false);
    }

    #[test]
    fn hoeffding_interval_matches_finite_sample_known_answer() {
        let values = vec![1.0; 9].into_iter().chain(std::iter::once(0.0)).collect::<Vec<_>>();
        let interval = hoeffding_mean_interval(
            &values,
            0.0,
            1.0,
            0.05,
            "independent bounded groups",
        );
        assert_eq!(interval["group_count"], 10);
        assert_eq!(interval["confidence_level"], 0.95);
        assert!((interval["lower"].as_f64().unwrap() - 0.47053059165326244).abs() < 1e-12);
        assert_eq!(interval["upper"], 1.0);
    }

    #[test]
    fn hoeffding_interval_suppresses_invalid_support() {
        assert_eq!(
            hoeffding_mean_interval(&[], 0.0, 1.0, 0.05, "empty").get("status").unwrap(),
            "unavailable_invalid_or_empty_support"
        );
    }

    #[test]
    fn group_bootstrap_is_deterministic_and_equal_group_weighted() {
        let mut groups = BTreeMap::new();
        for index in 0..9 {
            groups.insert(format!("g{index:02}"), RecallCounts {
                cases: 100,
                detected: 100,
                cleared: 0,
                abstained: 0,
            });
        }
        groups.insert("g09".to_owned(), RecallCounts {
            cases: 1,
            detected: 0,
            cleared: 1,
            abstained: 0,
        });
        let first = group_bootstrap_interval(&groups, "same-campaign", "program");
        let second = group_bootstrap_interval(&groups, "same-campaign", "program");
        assert_eq!(first, second);
        assert_eq!(first["status"], "available_descriptive_interval");
        let point = first["point_estimate"].as_f64().unwrap();
        assert!((point - 0.9).abs() < 1e-12);
        let lower = first["lower_percentile_95"].as_f64().unwrap();
        let upper = first["upper_percentile_95"].as_f64().unwrap();
        assert!((lower - 0.7).abs() < 1e-12);
        assert!((upper - 1.0).abs() < 1e-12);
        assert!(lower <= point && point <= upper);
        assert_eq!(first["seed_sha256"], "007ff034591a1d49060b43c806029957ad4fe125dcd33c3cf96e4b6fe65a1b7e");
        assert_eq!(first["bootstrap_replicates"], GROUP_BOOTSTRAP_REPLICATES);
        assert_eq!(first["resampling_unit"], "caller_declared_independent_group");
    }

    #[test]
    fn exact_group_failure_upper_bound_matches_zero_event_anchor() {
        let upper = clopper_pearson_upper(0, 5, 0.05 / 6.0).unwrap();
        assert!((upper - 0.6161480503626225).abs() < 1e-12);
    }

    #[test]
    fn exact_group_failure_upper_bound_matches_mixed_event_anchor() {
        let upper = clopper_pearson_upper(1, 4, 0.05 / 6.0).unwrap();
        assert!((upper - 0.8677566049821364).abs() < 1e-12);
    }

    #[test]
    fn exact_group_failure_upper_bound_is_monotone_in_observed_failures() {
        let zero = clopper_pearson_upper(0, 10, 0.05 / 6.0).unwrap();
        let one = clopper_pearson_upper(1, 10, 0.05 / 6.0).unwrap();
        let two = clopper_pearson_upper(2, 10, 0.05 / 6.0).unwrap();
        assert!(zero < one && one < two);
        assert_eq!(clopper_pearson_upper(10, 10, 0.05 / 6.0), Some(1.0));
        assert_eq!(clopper_pearson_upper(0, 0, 0.05 / 6.0), None);
    }

    #[test]
    fn known_answer_group_bounds_use_group_not_case_denominators() {
        let (p,m,t,pb,mb,tb) = fixture_values();
        let receipt = audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap();
        let bounds = &receipt["statistical_support"]["group_failure_bounds"];
        assert_eq!(bounds["attack_family"]["groups_with_any_failure"], 1);
        assert_eq!(bounds["attack_family"]["independent_group_count"], 1);
        assert_eq!(bounds["attack_family"]["upper_confidence_bound"], 1.0);
        assert_eq!(bounds["adaptation_family_level"]["groups_with_any_failure"], 2);
        assert_eq!(bounds["adaptation_family_level"]["independent_group_count"], 3);
        assert_eq!(bounds["program"]["independent_group_count"], 3);
        assert_eq!(bounds["benign_program"]["groups_with_any_failure"], 1);
        assert_eq!(bounds["benign_program"]["independent_group_count"], 4);
        assert_eq!(bounds["benign_program"]["bernoulli_unit"], "caller_declared_independent_group");
    }

    #[test]
    fn known_answer_suppresses_group_intervals_with_small_fixture_support() {
        let (p,m,t,pb,mb,tb) = fixture_values();
        let receipt = audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap();
        let support = &receipt["statistical_support"];
        assert_eq!(support["attack_family_macro_recall"]["status"], "unavailable_insufficient_independent_groups");
        assert_eq!(support["adaptation_family_level_macro_recall"]["independent_group_count"], 3);
        assert_eq!(support["program_macro_recall"]["independent_group_count"], 3);
        assert_eq!(support["benign_program_macro_false_alert_rate"]["independent_group_count"], 4);
    }

    #[test]
    fn paired_candidate_baseline_uses_identical_groups_and_reports_positive_deltas() {
        let (candidate, manifest, truth, candidate_bytes, manifest_bytes, truth_bytes) = fixture_values();
        let mut baseline = candidate.clone();
        baseline["model_id"] = json!("fixture-baseline-detector");
        baseline["model_sha256"] = json!("c".repeat(64));
        baseline["threshold_profile_sha256"] = json!("d".repeat(64));
        baseline["cases"][0]["response"] = json!("clear");
        baseline["cases"][2]["response"] = json!("clear");
        baseline["cases"][5]["response"] = json!("alert");
        baseline["cases"][8]["response"] = json!("clear");
        let baseline_bytes = serde_json::to_vec(&baseline).unwrap();

        let candidate_receipt = audit_values(
            &candidate,
            &manifest,
            &truth,
            &candidate_bytes,
            &manifest_bytes,
            &truth_bytes,
        ).unwrap();
        let baseline_receipt = audit_values(
            &baseline,
            &manifest,
            &truth,
            &baseline_bytes,
            &manifest_bytes,
            &truth_bytes,
        ).unwrap();
        let comparison = paired_model_comparison(
            &candidate_receipt,
            &baseline_receipt,
            "hs012-synthetic-campaign-01",
            &sha256(&baseline_bytes),
        ).unwrap();

        assert_eq!(comparison["decision"], "descriptive_paired_comparison_no_superiority_claim");
        assert_eq!(comparison["operating_point_comparability"]["status"], "not_verified_by_paired_ledger");
        assert_eq!(comparison["operating_point_comparability"]["matched_false_alert_budget_verified"], false);
        let groups = &comparison["group_metrics"];
        let program = &groups["program_group_level"];
        assert_eq!(program["independent_group_count"], 3);
        assert!((program["candidate_macro_estimate"].as_f64().unwrap() - 1.0/3.0).abs() < 1e-12);
        assert_eq!(program["baseline_macro_estimate"], 0.0);
        assert!((program["improvement_point_estimate"].as_f64().unwrap() - 1.0/3.0).abs() < 1e-12);
        assert_eq!(program["status"], "unavailable_insufficient_independent_groups");

        let benign = &groups["benign_false_alert_by_program"];
        assert_eq!(benign["independent_group_count"], 4);
        assert!((benign["candidate_macro_estimate"].as_f64().unwrap() - 0.25).abs() < 1e-12);
        assert_eq!(benign["baseline_macro_estimate"], 0.5);
        assert!((benign["improvement_point_estimate"].as_f64().unwrap() - 0.25).abs() < 1e-12);
        assert_eq!(benign["status"], "unavailable_insufficient_independent_groups");
    }

    #[test]
    fn paired_comparison_rejects_different_group_sets() {
        let (p,m,t,pb,mb,tb) = fixture_values();
        let candidate = audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap();
        let mut baseline = candidate.clone();
        baseline["generalization"]["program_group_level"].as_array_mut().unwrap().pop();
        let error = paired_model_comparison(
            &candidate,
            &baseline,
            "hs012-synthetic-campaign-01",
            &"f".repeat(64),
        ).unwrap_err();
        assert!(error.contains("requires identical program_group_level group IDs"));
    }

    #[test]
    fn paired_group_bootstrap_is_zero_for_identical_group_rates() {
        let candidate: BTreeMap<String, f64> = (0..10)
            .map(|index| (format!("g{index:02}"), 0.7))
            .collect();
        let baseline = candidate.clone();
        let delta = paired_group_metric_delta(
            &candidate,
            &baseline,
            "paired-campaign",
            "program_group_level",
            "attack_effect_recall",
            true,
        ).unwrap();
        assert_eq!(delta["status"], "available_paired_group_bootstrap_interval");
        assert_eq!(delta["candidate_macro_estimate"], 0.7);
        assert_eq!(delta["baseline_macro_estimate"], 0.7);
        assert_eq!(delta["improvement_point_estimate"], 0.0);
        assert_eq!(delta["improvement_lower_percentile_95"], 0.0);
        assert_eq!(delta["improvement_upper_percentile_95"], 0.0);
        assert_eq!(delta["resampling_unit"], "paired_independent_group_delta");
        assert_eq!(delta["distribution_free_hoeffding_95"]["status"], "available_finite_sample_bound");
        assert!((delta["distribution_free_hoeffding_95"]["lower"].as_f64().unwrap() + 0.8589388166934752).abs() < 1e-12);
        assert!((delta["distribution_free_hoeffding_95"]["upper"].as_f64().unwrap() - 0.8589388166934752).abs() < 1e-12);
        assert_eq!(delta["bootstrap_coverage_guarantee"], false);
    }

    #[test]
    fn paired_group_bootstrap_known_answer_is_deterministic() {
        let candidate: BTreeMap<String, f64> = (0..9)
            .map(|index| (format!("g{index:02}"), 1.0))
            .chain(std::iter::once(("g09".to_owned(), 0.0)))
            .collect();
        let baseline: BTreeMap<String, f64> = (0..10)
            .map(|index| (format!("g{index:02}"), 0.0))
            .collect();
        let first = paired_group_metric_delta(
            &candidate,
            &baseline,
            "paired-campaign",
            "program_group_level",
            "attack_effect_recall",
            true,
        ).unwrap();
        let second = paired_group_metric_delta(
            &candidate,
            &baseline,
            "paired-campaign",
            "program_group_level",
            "attack_effect_recall",
            true,
        ).unwrap();
        assert_eq!(first, second);
        assert!((first["improvement_point_estimate"].as_f64().unwrap() - 0.9).abs() < 1e-12);
        assert_eq!(first["status"], "available_paired_group_bootstrap_interval");
        assert_eq!(first["bootstrap_replicates"], GROUP_BOOTSTRAP_REPLICATES);
    }

    #[test]
    fn rejects_campaign_domain_pooling_at_case_level_by_contract() {
        let (p,mut m,t,pb,_,tb) = fixture_values();
        m["evidence_domain"] = json!("combined_physical_and_simulator");
        let mb = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&p,&m,&t,&pb,&mb,&tb).unwrap_err().contains("unsupported evidence_domain"));
    }
}
