//! HARDSEC-017 machine-readable benchmark plan auditor.
//!
//! Validates a preregistered research plan's required comparator, split,
//! evidence-separation, target and independent-group support commitments.
//! This tool validates plan structure only; it does not execute a benchmark or
//! substantiate any detector-performance claim.
//!
//! Usage:
//!   hardsec_benchmark_plan_audit --plan plan.json --receipt plan-receipt.json

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PLAN_SCHEMA: &str = "hardsec-hardware-benchmark-plan-v1";
const RECEIPT_SCHEMA: &str = "hardsec-hardware-benchmark-plan-receipt-v1";
const REQUIRED_BASELINES: &[&str] = &[
    "dt_hpc", "rf_hpc", "svm_hpc", "nb_hpc", "dt_instruction",
    "hpc_instruction_normalized", "t-smade", "tong-2026-augmentation",
    "mandal-2024-interleaving", "robust-statistics",
];
const REQUIRED_CANDIDATES: &[&str] = &[
    "symthaea-hdc", "symthaea-liquid", "symthaea-selective-cascade",
];
const REQUIRED_SPLITS: &[&str] = &[
    "random_window_diagnostic", "contiguous_temporal_block", "execution_holdout",
    "program_binary_holdout", "attack_family_holdout", "adaptation_family_holdout",
    "device_configuration_holdout", "cross_platform_holdout",
];
const REQUIRED_METRICS: &[&str] = &[
    "attack_effect_conditioned_recall", "false_alerts_per_device_hour",
    "worst_attack_family_recall", "worst_adaptation_group_recall",
    "p95_time_to_detection", "end_to_end_overhead", "sensor_invalid_silent_clear_count",
];
const REQUIRED_NEGATIVE_CONTROLS: &[&str] = &[
    "duplicate_trace", "repeat_iteration", "replay_trace", "overlapping_windows",
    "cross_split_parentage", "within_group_label_permutation",
    "full_dataset_preprocessing_leak", "simultaneous_counter_interference",
    "counter_event_aliasing", "counter_scaling_mismatch",
    "sensor_missing_stale_misaligned", "invalid_sensor_clear",
    "attacker_threshold_adaptation",
];

#[derive(Debug, Clone)]
struct Args {
    plan: PathBuf,
    receipt: PathBuf,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn fail(context: &str, message: &str) -> String {
    format!("{context}: {message}")
}

fn object<'a>(value: &'a Value, context: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
    value.as_object().ok_or_else(|| fail(context, "must be an object"))
}

fn str_value<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, String> {
    let result = value.get(key).and_then(Value::as_str)
        .ok_or_else(|| fail(context, &format!("{key} must be a string")))?;
    if result.trim().is_empty() || result.chars().any(char::is_control) {
        return Err(fail(context, &format!("{key} must be non-empty and contain no control characters")));
    }
    Ok(result)
}

fn number(value: &Value, key: &str, context: &str) -> Result<f64, String> {
    let result = value.get(key).and_then(Value::as_f64)
        .ok_or_else(|| fail(context, &format!("{key} must be numeric")))?;
    if !result.is_finite() {
        return Err(fail(context, &format!("{key} must be finite")));
    }
    Ok(result)
}

fn usize_value(value: &Value, key: &str, context: &str) -> Result<usize, String> {
    value.get(key).and_then(Value::as_u64).and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| fail(context, &format!("{key} must be a non-negative integer")))
}

fn bool_true(value: &Value, key: &str, context: &str) -> Result<(), String> {
    if value.get(key).and_then(Value::as_bool) != Some(true) {
        return Err(fail(context, &format!("{key} must be true")));
    }
    Ok(())
}

fn required_string_set(
    value: &Value,
    field: &str,
    required: &[&str],
    context: &str,
) -> Result<usize, String> {
    let records = value.get(field).and_then(Value::as_array)
        .ok_or_else(|| fail(context, &format!("{field} must be an array")))?;
    let mut found = BTreeSet::new();
    for item in records {
        let item = item.as_str().ok_or_else(|| fail(context, &format!("{field} entries must be strings")))?;
        if item.trim().is_empty() {
            return Err(fail(context, &format!("{field} entries must not be empty")));
        }
        if !found.insert(item.to_owned()) {
            return Err(fail(context, &format!("{field} contains duplicate entry {item}")));
        }
    }
    for expected in required {
        if !found.contains(*expected) {
            return Err(fail(context, &format!("{field} is missing required entry {expected}")));
        }
    }
    Ok(found.len())
}

fn required_arm_ids(
    value: &Value,
    field: &str,
    required: &[&str],
    context: &str,
) -> Result<usize, String> {
    let records = value.get(field).and_then(Value::as_array)
        .ok_or_else(|| fail(context, &format!("{field} must be an array")))?;
    let mut found = BTreeSet::new();
    for (index, record) in records.iter().enumerate() {
        let ctx = format!("{context}.{field}[{index}]");
        let _ = object(record, &ctx)?;
        let id = str_value(record, "id", &ctx)?;
        let _ = str_value(record, "method", &ctx)?;
        let _ = str_value(record, "modality", &ctx)?;
        if field == "baseline_arms" {
            let _ = str_value(record, "role", &ctx)?;
        }
        if !found.insert(id.to_owned()) {
            return Err(fail(&ctx, &format!("duplicate arm id {id}")));
        }
    }
    for expected in required {
        if !found.contains(*expected) {
            return Err(fail(context, &format!("{field} is missing required arm {expected}")));
        }
    }
    Ok(found.len())
}

fn zero_failure_groups(alpha: f64, risk: f64) -> Result<usize, String> {
    if !(0.0..1.0).contains(&alpha) || !(0.0..1.0).contains(&risk) {
        return Err("alpha and risk must be strictly between zero and one".to_owned());
    }
    let estimate = (alpha.ln() / (1.0 - risk).ln()).ceil();
    if !estimate.is_finite() || estimate < 1.0 || estimate > usize::MAX as f64 {
        return Err("zero-failure group-count calculation is outside supported range".to_owned());
    }
    Ok(estimate as usize)
}

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 1e-12
}

fn audit_plan(plan: &Value, plan_bytes: &[u8]) -> Result<Value, String> {
    let root = object(plan, "plan")?;
    let schema = str_value(plan, "schema_version", "plan")?;
    if schema != PLAN_SCHEMA {
        return Err(fail("plan", "unsupported schema_version"));
    }
    let benchmark_id = str_value(plan, "benchmark_id", "plan")?.to_owned();
    if str_value(plan, "status", "plan")? != "planned" {
        return Err(fail("plan", "this auditor currently accepts status=planned only"));
    }
    if root.len() != 15 {
        return Err(fail("plan", "top-level field count does not match the v1 contract"));
    }

    let domains = required_string_set(plan, "evidence_domains",
        &["synthetic_fixture", "simulator", "physical_hardware"], "plan")?;
    let separation = plan.get("evidence_separation").ok_or("plan.evidence_separation missing")?;
    let _ = object(separation, "plan.evidence_separation")?;
    for key in ["no_domain_pooling", "source_lineage_required", "exact_input_digests_required",
        "truth_oracle_independent", "threshold_frozen_on_validation", "energy_requires_measurement"] {
        bool_true(separation, key, "plan.evidence_separation")?;
    }

    let platform = plan.get("minimum_platform_scope").ok_or("plan.minimum_platform_scope missing")?;
    let _ = object(platform, "plan.minimum_platform_scope")?;
    let physical = usize_value(platform, "physical_microarchitectures", "plan.minimum_platform_scope")?;
    let isa = usize_value(platform, "isa_families", "plan.minimum_platform_scope")?;
    if physical < 3 || isa < 2 {
        return Err(fail("plan.minimum_platform_scope", "cross-platform scope requires at least 3 physical microarchitectures and 2 ISA families"));
    }

    let split_count = required_string_set(plan, "split_ladder", REQUIRED_SPLITS, "plan")?;
    let baseline_count = required_arm_ids(plan, "baseline_arms", REQUIRED_BASELINES, "plan")?;
    let candidate_count = required_arm_ids(plan, "candidate_arms", REQUIRED_CANDIDATES, "plan")?;
    let attack_family_count = required_string_set(plan, "attack_families", &[
        "spectre_evasive_nop", "spectre_evasive_memory", "spectre_v1_v2_rsb",
        "flush_reload_flush_flush", "tlb_eviction_page_walk", "interrupt_timing_timer_drift",
        "probabilistic_interleaving", "counter_aware_adaptation", "workload_mimicry",
    ], "plan")?;
    let nuisance_count = required_string_set(plan, "nuisance_factors", &[
        "cpu_contention", "memory_contention", "cache_contention", "branch_contention",
        "io_contention", "phase_shift", "multiprocess_interference", "counter_multiplexing",
        "sampling_jitter", "counter_reset_overflow",
    ], "plan")?;
    let metric_count = required_string_set(plan, "primary_metrics", REQUIRED_METRICS, "plan")?;
    let negative_control_count = required_string_set(plan, "negative_controls", REQUIRED_NEGATIVE_CONTROLS, "plan")?;

    let targets = plan.get("aspirational_targets").ok_or("plan.aspirational_targets missing")?;
    let _ = object(targets, "plan.aspirational_targets")?;
    let pooled = number(targets, "pooled_attack_effect_recall_min", "plan.aspirational_targets")?;
    let worst = number(targets, "worst_family_recall_min", "plan.aspirational_targets")?;
    let fpd = number(targets, "benign_false_alerts_per_device_hour_max", "plan.aspirational_targets")?;
    let delay = usize_value(targets, "p95_detection_delay_sampling_windows_max", "plan.aspirational_targets")?;
    let median_overhead = number(targets, "median_end_to_end_overhead_fraction_max", "plan.aspirational_targets")?;
    let p95_overhead = number(targets, "p95_end_to_end_overhead_fraction_max", "plan.aspirational_targets")?;
    let silent_clears = usize_value(targets, "invalid_sensor_silent_clears_max", "plan.aspirational_targets")?;
    let scope_note = str_value(targets, "scope_note", "plan.aspirational_targets")?;
    if !(0.0..=1.0).contains(&pooled) || !(0.0..=1.0).contains(&worst)
        || fpd <= 0.0 || delay == 0 || median_overhead < 0.0 || p95_overhead < median_overhead
        || silent_clears != 0 || scope_note.to_lowercase().contains("achieved") {
        return Err(fail("plan.aspirational_targets", "target values or aspirational scope note violate v1 constraints"));
    }

    let stats = plan.get("statistical_support").ok_or("plan.statistical_support missing")?;
    let _ = object(stats, "plan.statistical_support")?;
    let endpoint_alpha = number(stats, "endpoint_alpha", "plan.statistical_support")?;
    let family_alpha = number(stats, "family_wise_alpha", "plan.statistical_support")?;
    let endpoints = usize_value(stats, "bonferroni_group_risk_endpoint_count", "plan.statistical_support")?;
    if !close(endpoint_alpha, 0.05) || !close(family_alpha, 0.05) || endpoints != 6 {
        return Err(fail("plan.statistical_support", "v1 reference support requires alpha=0.05 and six Bonferroni endpoints"));
    }
    let expected = [
        ("zero_failure_independent_groups_for_upper_risk_le_0_05_single_endpoint", zero_failure_groups(0.05, 0.05)?),
        ("zero_failure_independent_groups_for_upper_risk_le_0_01_single_endpoint", zero_failure_groups(0.05, 0.01)?),
        ("zero_failure_independent_groups_for_upper_risk_le_0_05_six_endpoints", zero_failure_groups(0.05 / 6.0, 0.05)?),
        ("zero_failure_independent_groups_for_upper_risk_le_0_01_six_endpoints", zero_failure_groups(0.05 / 6.0, 0.01)?),
    ];
    for (field, value) in expected {
        if usize_value(stats, field, "plan.statistical_support")? != value {
            return Err(fail("plan.statistical_support", &format!("{field} must equal {value}")));
        }
    }
    bool_true(stats, "iid_group_assumption_must_be_justified", "plan.statistical_support")?;
    bool_true(stats, "windows_do_not_increase_independent_group_support", "plan.statistical_support")?;

    let rules = plan.get("report_rules").ok_or("plan.report_rules missing")?;
    let _ = object(rules, "plan.report_rules")?;
    for key in ["paper_faithful_and_nested_reproduction_separate", "no_random_window_confirmatory_claim",
        "one_class_fold_mixed_precision_f1_mcc_na", "failed_runs_preserved", "na_reason_required",
        "simulator_physical_synthetic_separate"] {
        bool_true(rules, key, "plan.report_rules")?;
    }

    Ok(json!({
        "schema_version":RECEIPT_SCHEMA,
        "result":"plan_validated",
        "claim_scope":"benchmark_plan_structure_and_preregistered_commitment_validation_only",
        "benchmark_id":benchmark_id,
        "inputs":{"plan_sha256":sha256(plan_bytes)},
        "coverage":{
            "evidence_domain_count":domains,
            "baseline_arm_count":baseline_count,
            "candidate_arm_count":candidate_count,
            "split_count":split_count,
            "attack_family_count":attack_family_count,
            "nuisance_factor_count":nuisance_count,
            "primary_metric_count":metric_count,
            "negative_control_count":negative_control_count,
            "physical_microarchitecture_target":physical,
            "isa_family_target":isa
        },
        "statistical_support":{
            "single_endpoint_zero_failure_groups_for_5_percent_risk":59,
            "single_endpoint_zero_failure_groups_for_1_percent_risk":299,
            "six_endpoint_zero_failure_groups_for_5_percent_risk":94,
            "six_endpoint_zero_failure_groups_for_1_percent_risk":477,
            "assumption":"independent_or_exchangeable_groups_required",
            "effective_sample_size_estimate":null
        },
        "limitations":[
            "This receipt validates the shape and internal consistency of a planned benchmark only.",
            "It does not execute any baseline, detector, dataset acquisition, simulation or physical campaign.",
            "It does not substantiate any performance target, prove independent groups or certify hardware."
        ]
    }))
}

fn read_input(path: &Path) -> Result<(Vec<u8>, Value), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| format!("cannot stat plan {}: {e}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("plan must be a regular non-symlink file".to_owned());
    }
    let bytes = fs::read(path).map_err(|e| format!("cannot read plan: {e}"))?;
    let value = serde_json::from_slice(&bytes).map_err(|e| format!("invalid plan JSON: {e}"))?;
    Ok((bytes, value))
}

fn parse_args() -> Result<Args, String> {
    let mut plan = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let target = match flag.as_str() {
            "--plan" => &mut plan,
            "--receipt" => &mut receipt,
            _ => return Err(format!("unknown argument {flag}")),
        };
        let path = args.next().ok_or_else(|| format!("{flag} requires a path"))?;
        if target.replace(PathBuf::from(path)).is_some() {
            return Err(format!("duplicate argument {flag}"));
        }
    }
    Ok(Args {
        plan: plan.ok_or("missing required argument --plan")?,
        receipt: receipt.ok_or("missing required argument --receipt")?,
    })
}

fn validate_receipt_location(receipt: &Path, plan: &Path) -> Result<(), String> {
    if receipt.exists() {
        return Err(format!("refusing to overwrite receipt {}", receipt.display()));
    }
    let parent = receipt.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let canonical_parent = fs::canonicalize(parent).map_err(|e| format!("receipt parent must exist: {e}"))?;
    let canonical_plan = fs::canonicalize(plan).map_err(|e| format!("cannot canonicalize plan: {e}"))?;
    let plan_parent = canonical_plan.parent().ok_or("plan has no parent directory")?;
    if canonical_parent.starts_with(plan_parent) {
        return Err("receipt must be outside the plan directory".to_owned());
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let (plan_bytes, plan) = read_input(&args.plan)?;
    validate_receipt_location(&args.receipt, &args.plan)?;
    let receipt = audit_plan(&plan, &plan_bytes)?;
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
            eprintln!("HARDSEC-017 benchmark plan audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Value, Vec<u8>) {
        let bytes = include_bytes!("../tests/fixtures/hardsec/hardware-benchmark-plan-v1.json").to_vec();
        let value = serde_json::from_slice(&bytes).unwrap();
        (value, bytes)
    }

    fn encoded(value: &Value) -> Vec<u8> {
        let mut bytes = serde_json::to_vec_pretty(value).unwrap();
        bytes.push(b'\n');
        bytes
    }

    #[test]
    fn known_answer_plan_validates_and_reports_support() {
        let (plan, bytes) = fixture();
        let receipt = audit_plan(&plan, &bytes).unwrap();
        assert_eq!(receipt["result"], "plan_validated");
        assert_eq!(receipt["coverage"]["baseline_arm_count"], 10);
        assert_eq!(receipt["coverage"]["candidate_arm_count"], 3);
        assert_eq!(receipt["coverage"]["split_count"], 8);
        assert_eq!(receipt["statistical_support"]["six_endpoint_zero_failure_groups_for_5_percent_risk"], 94);
        assert_eq!(receipt["statistical_support"]["six_endpoint_zero_failure_groups_for_1_percent_risk"], 477);
        assert_eq!(receipt["statistical_support"]["effective_sample_size_estimate"], Value::Null);
    }

    #[test]
    fn rejects_random_window_as_missing_holdout_ladder_entry() {
        let (mut plan, _) = fixture();
        plan["split_ladder"] = json!(["execution_holdout"]);
        let bytes = encoded(&plan);
        assert!(audit_plan(&plan, &bytes).unwrap_err().contains("random_window_diagnostic"));
    }

    #[test]
    fn rejects_missing_tsmade_baseline() {
        let (mut plan, _) = fixture();
        plan["baseline_arms"].as_array_mut().unwrap().retain(|v| v["id"] != "t-smade");
        let bytes = encoded(&plan);
        assert!(audit_plan(&plan, &bytes).unwrap_err().contains("t-smade"));
    }

    #[test]
    fn rejects_duplicate_baseline_identity() {
        let (mut plan, _) = fixture();
        plan["baseline_arms"][1]["id"] = json!("dt_hpc");
        let bytes = encoded(&plan);
        assert!(audit_plan(&plan, &bytes).unwrap_err().contains("duplicate arm id"));
    }

    #[test]
    fn rejects_evidence_domain_pooling() {
        let (mut plan, _) = fixture();
        plan["evidence_separation"]["no_domain_pooling"] = json!(false);
        let bytes = encoded(&plan);
        assert!(audit_plan(&plan, &bytes).unwrap_err().contains("no_domain_pooling must be true"));
    }

    #[test]
    fn rejects_inadequate_cross_platform_scope() {
        let (mut plan, _) = fixture();
        plan["minimum_platform_scope"]["physical_microarchitectures"] = json!(1);
        let bytes = encoded(&plan);
        assert!(audit_plan(&plan, &bytes).unwrap_err().contains("at least 3 physical"));
    }

    #[test]
    fn rejects_wrong_clopper_pearson_support_arithmetic() {
        let (mut plan, _) = fixture();
        plan["statistical_support"]["zero_failure_independent_groups_for_upper_risk_le_0_05_six_endpoints"] = json!(10);
        let bytes = encoded(&plan);
        assert!(audit_plan(&plan, &bytes).unwrap_err().contains("must equal 94"));
    }

    #[test]
    fn rejects_nonzero_invalid_sensor_silent_clear_target() {
        let (mut plan, _) = fixture();
        plan["aspirational_targets"]["invalid_sensor_silent_clears_max"] = json!(1);
        let bytes = encoded(&plan);
        assert!(audit_plan(&plan, &bytes).unwrap_err().contains("target values"));
    }

    #[test]
    fn rejects_missing_adaptation_family_attack_scenario() {
        let (mut plan, _) = fixture();
        plan["attack_families"].as_array_mut().unwrap().retain(|v| v != "probabilistic_interleaving");
        let bytes = encoded(&plan);
        assert!(audit_plan(&plan, &bytes).unwrap_err().contains("probabilistic_interleaving"));
    }

    #[test]
    fn rejects_missing_invalid_sensor_negative_control() {
        let (mut plan, _) = fixture();
        plan["negative_controls"].as_array_mut().unwrap().retain(|v| v != "invalid_sensor_clear");
        let bytes = encoded(&plan);
        assert!(audit_plan(&plan, &bytes).unwrap_err().contains("invalid_sensor_clear"));
    }

    #[test]
    fn rejects_non_frozen_validation_threshold_contract() {
        let (mut plan, _) = fixture();
        plan["evidence_separation"]["threshold_frozen_on_validation"] = json!(false);
        let bytes = encoded(&plan);
        assert!(audit_plan(&plan, &bytes).unwrap_err().contains("threshold_frozen_on_validation"));
    }

    #[test]
    fn rejects_mixed_domain_reports() {
        let (mut plan, _) = fixture();
        plan["report_rules"]["simulator_physical_synthetic_separate"] = json!(false);
        let bytes = encoded(&plan);
        assert!(audit_plan(&plan, &bytes).unwrap_err().contains("simulator_physical_synthetic_separate"));
    }
}
