// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed analysis for the pre-registered Proof Insight Transfer benchmark.
//!
//! Inputs are immutable run configuration, the exact protocol and split files,
//! a validation-only baseline-selection receipt, and JSONL target outcomes.
//! This computes no evidence from candidate annotations; it only analyzes
//! outcomes supplied with explicit receipts and matched per-arm budgets.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Component, Path},
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const RUN_CONFIG_SCHEMA: &str = "pitb.run-config.v1";
const RUN_RESULTS_SCHEMA: &str = "pitb.target-outcome.v0";
const ANALYSIS_REPORT_SCHEMA: &str = "pitb.analysis-report.v0";
const EXPECTED_PROTOCOL_ID: &str = "proof-insight-transfer-v0";
const EXPECTED_UPSTREAM_COMMIT: &str = "1581d4cbdd69a0faf34b2089614fe2a814316a6a";
const EXPECTED_BOOTSTRAP_REPLICATES: usize = 10_000;
// Conservative minimum, not a substitute for run-specific simulation-based power.
const MIN_CONFIRMATORY_FAMILY_CLUSTERS: usize = 20;
const POWER_ASSOCIATION_SIMULTANEOUS_Z: f64 = 2.394;
const REQUIRED_EXPOSED_VALIDATION_FAMILY: &str = "012";
const REQUIRED_EXPOSED_TEST_FAMILY: &str = "089";
const EXPECTED_SPLIT_SEED: &str = "pitb-v0-openai-math-2026-10-19";
const EXPECTED_SPLIT_ALGORITHM: &str = "sha256(seed || NUL || family_id), sort by digest then family ID, floor 70% train and floor 85% train+validation";
const ARMS: [&str; 9] = ["A", "B", "C", "D", "E", "F", "G", "H", "I"];
const BASELINE_SELECTION_ARMS: [&str; 6] = ["A", "B", "C", "D", "E", "I"];
const TREATMENT_ARM: &str = "F";

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RunConfig {
    schema_version: String,
    protocol_id: String,
    protocol_sha256: String,
    split_manifest_sha256: String,
    partition_digest_sha256: String,
    baseline_selection_receipt_sha256: String,
    known_exposed_validation_family_ids: Vec<String>,
    known_exposed_test_family_ids: Vec<String>,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
    model_version: String,
    checker_version: String,
    budget: Budget,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    power_report_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Budget {
    max_model_calls: u32,
    max_tokens: u64,
    max_wall_ms: u64,
    max_cost_microusd: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaselineSelectionReceipt {
    schema_version: String,
    selected_split: String,
    metric: String,
    selected_arm: String,
    frozen_before_test: bool,
    test_outcomes_seen_before_selection: bool,
    tie_break_rule: String,
    validation_results_sha256: String,
    arm_scores: BTreeMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct SplitManifest {
    upstream: UpstreamMetadata,
    split_plan: SplitPlan,
    families: Vec<SplitFamily>,
}

#[derive(Debug, Deserialize)]
struct UpstreamMetadata {
    commit: String,
}

#[derive(Debug, Deserialize)]
struct SplitPlan {
    seed: String,
    algorithm: String,
    train_percent: u8,
    validation_percent: u8,
    test_percent: u8,
    counts: SplitCounts,
    partition_digest_sha256: String,
}

#[derive(Debug, Deserialize)]
struct SplitCounts {
    train: usize,
    validation: usize,
    test: usize,
}

#[derive(Debug, Deserialize)]
struct SplitFamily {
    family_id: String,
    split: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetOutcome {
    schema_version: String,
    target_id: String,
    family_id: String,
    split: String,
    eligibility: Eligibility,
    #[serde(default)]
    source_commit: Option<String>,
    #[serde(default)]
    theorem_id: Option<String>,
    #[serde(default)]
    artifact_path: Option<String>,
    #[serde(default)]
    artifact_sha256: Option<String>,
    #[serde(default)]
    lean_toolchain: Option<String>,
    #[serde(default)]
    mathlib_revision: Option<String>,
    #[serde(default)]
    statement_sha256: Option<String>,
    #[serde(default)]
    eligibility_receipt_sha256: Option<String>,
    #[serde(default)]
    statement_review_receipt_sha256: Option<String>,
    #[serde(default)]
    proof_audit_receipt_sha256: Option<String>,
    #[serde(default)]
    successor_test_receipt_sha256: Option<String>,
    #[serde(default)]
    mutation_receipt_sha256: Option<String>,
    #[serde(default)]
    exclusion_reason: Option<String>,
    arms: Vec<ArmOutcome>,
}

#[derive(Debug, Deserialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum Eligibility {
    Eligible,
    Excluded,
    Unknown,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum OutcomeStatus {
    VerifiedSuccess,
    ProofOnly,
    Failed,
    Timeout,
    NotAttempted,
    ToolUnavailable,
    OutOfScope,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmOutcome {
    arm: String,
    status: OutcomeStatus,
    checker_passed: bool,
    semantic_gate_passed: bool,
    model_version: String,
    checker_version: String,
    budget: Budget,
    model_calls_used: u32,
    tokens_used: u64,
    wall_ms: u64,
    cost_microusd: u64,
    config_sha256: String,
    input_sha256: String,
    retrieval_log_sha256: String,
    checker_receipt_sha256: String,
    semantic_gate_receipt_sha256: String,
}

#[derive(Debug, Serialize)]
struct AnalysisReport {
    schema_version: &'static str,
    protocol_id: String,
    protocol_sha256: String,
    run_config_sha256: String,
    power_report_sha256: String,
    validation_results_sha256: String,
    results_jsonl_sha256: String,
    analyzer_source_sha256: String,
    split_manifest_sha256: String,
    partition_digest_sha256: String,
    baseline_selection_receipt_sha256: String,
    upstream_commit: String,
    selected_baseline_arm: String,
    treatment_arm: &'static str,
    total_test_families_in_frozen_split: usize,
    eligible_validation_target_count: usize,
    excluded_validation_target_count: usize,
    unknown_validation_target_count: usize,
    validation_baseline_scores: BTreeMap<String, f64>,
    eligible_target_count: usize,
    eligible_family_count: usize,
    excluded_target_count: usize,
    unknown_eligibility_target_count: usize,
    known_exposed_validation_family_ids: Vec<String>,
    known_exposed_test_family_ids: Vec<String>,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
    family_aggregates: Vec<FamilyAggregate>,
    arm_metrics: Vec<ArmMetrics>,
    primary_comparison: PrimaryComparison,
    external_replication: &'static str,
    interpretation: String,
}

#[derive(Debug, Serialize)]
struct FamilyAggregate {
    family_id: String,
    target_count: usize,
    treatment_semantic_gate_successes: usize,
    baseline_semantic_gate_successes: usize,
    paired_delta_sum: i64,
    paired_effect_percentage_points: f64,
}

#[derive(Debug, Serialize)]
struct ArmMetrics {
    arm: String,
    target_count: usize,
    checker_successes: usize,
    semantic_gate_successes: usize,
    checker_success_rate: f64,
    semantic_gate_success_rate: f64,
    total_model_calls: u64,
    total_tokens: u64,
    total_wall_ms: u128,
    total_cost_microusd: u128,
    mean_wall_ms_per_target: f64,
    mean_tokens_per_target: f64,
    cost_microusd_per_verified_solve: Option<f64>,
}

#[derive(Debug, Serialize)]
struct PrimaryComparison {
    metric: &'static str,
    confidence_interval_method: &'static str,
    baseline_arm: String,
    treatment_arm: &'static str,
    paired_absolute_effect_percentage_points: f64,
    confidence_level: f64,
    clustered_bootstrap_lower_percentage_points: f64,
    clustered_bootstrap_upper_percentage_points: f64,
    target_pairs: usize,
    family_clusters: usize,
    minimum_family_clusters_for_confirmatory_claim: usize,
    minimum_effect_percentage_points: f64,
    confidence_interval_informative: bool,
    confidence_interval_excludes_zero_positive: bool,
    preregistered_utility_gate_passed: bool,
    gate_decision: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct FamilyTargetCount {
    family_id: String,
    eligible_target_count: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct PowerScenario {
    status: String,
    true_treatment_effect_percentage_points: f64,
    assumed_family_icc: f64,
    pair_association_fraction: f64,
    study_simulations: usize,
    bootstrap_replicates_per_simulated_study: usize,
    utility_gate_passes: usize,
    degenerate_interval_trials: usize,
    estimated_utility_gate_power: Option<f64>,
    monte_carlo_standard_error: Option<f64>,
    estimated_power_lower_95: Option<f64>,
    estimated_power_upper_95: Option<f64>,
    null_effect_false_claim_probability_upper_simultaneous_95: Option<f64>,
    mean_observed_effect_percentage_points: Option<f64>,
    mean_lower_95ci_percentage_points: Option<f64>,
    interpretation: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct WeakNullScenario {
    status: String,
    true_target_weighted_mean_effect_percentage_points: f64,
    assumed_family_icc: f64,
    pair_association_fraction: f64,
    requested_family_effect_heterogeneity_sd_percentage_points: f64,
    study_simulations: usize,
    bootstrap_replicates_per_simulated_study: usize,
    false_claims: usize,
    degenerate_interval_trials: usize,
    estimated_false_claim_probability: Option<f64>,
    monte_carlo_standard_error: Option<f64>,
    false_claim_probability_lower_95: Option<f64>,
    false_claim_probability_upper_95: Option<f64>,
    false_claim_probability_upper_simultaneous_95: Option<f64>,
    mean_observed_effect_percentage_points: Option<f64>,
    mean_realized_family_effect_heterogeneity_sd_percentage_points: Option<f64>,
    interpretation: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct PowerReport {
    schema_version: String,
    protocol_id: String,
    protocol_sha256: String,
    run_design_sha256: String,
    validation_results_sha256: String,
    baseline_selection_receipt_sha256: String,
    split_manifest_sha256: String,
    partition_digest_sha256: String,
    analyzer_source_sha256: String,
    upstream_commit: String,
    selected_baseline_arm: String,
    known_exposed_validation_family_ids: Vec<String>,
    known_exposed_test_family_ids: Vec<String>,
    eligible_validation_target_count: usize,
    eligible_validation_family_count: usize,
    validation_baseline_semantic_success_rate: f64,
    family_target_counts: Vec<FamilyTargetCount>,
    assumed_family_icc_grid: Vec<f64>,
    pair_association_fraction_grid: Vec<f64>,
    true_effect_grid_percentage_points: Vec<f64>,
    study_simulations: usize,
    bootstrap_replicates_per_simulated_study: usize,
    simulation_seed: u64,
    confirmatory_minimum_family_count: usize,
    utility_effect_threshold_percentage_points: f64,
    required_power_at_20pp_and_icc_020: f64,
    maximum_false_claim_probability_at_zero_effect: f64,
    null_effect_false_claim_probability_max: Option<f64>,
    null_effect_false_claim_probability_upper_simultaneous_95_max: Option<f64>,
    weak_null_family_effect_heterogeneity_sd_grid_percentage_points: Vec<f64>,
    weak_null_pair_association_fraction_grid: Vec<f64>,
    weak_null_scenarios: Vec<WeakNullScenario>,
    weak_null_false_claim_probability_max: Option<f64>,
    weak_null_false_claim_probability_upper_simultaneous_95_max: Option<f64>,
    estimated_power_at_20pp_and_icc_020: Option<f64>,
    estimated_power_at_20pp_and_icc_020_lower_95: Option<f64>,
    estimated_power_at_20pp_and_icc_020_lower_simultaneous_95: Option<f64>,
    power_gate_passed: bool,
    scenarios: Vec<PowerScenario>,
    model_assumptions: Vec<String>,
    interpretation: String,
}

struct SimulatedFamily {
    target_count: usize,
    paired_delta_sum: i64,
}

fn run_design_sha256(config: &RunConfig) -> Result<String> {
    let mut design = config.clone();
    design.power_report_sha256 = None;
    let bytes = serde_json::to_vec(&design).context("serialize canonical run design")?;
    Ok(sha256_hex(&bytes))
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("proof-insight analysis rejected: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn execute() -> Result<()> {
    let args = parse_args(env::args().skip(1).collect())?;
    match args.get("--mode").map(String::as_str).unwrap_or("analyze") {
        "analyze" => execute_analysis(&args),
        "power" => execute_power(&args),
        mode => bail!("unsupported mode {mode}; expected analyze or power"),
    }
}

fn execute_analysis(args: &BTreeMap<String, String>) -> Result<()> {
    let config_bytes = read_arg(args, "--config")?;
    let results_bytes = read_arg(args, "--results")?;
    let validation_bytes = read_arg(args, "--validation-results")?;
    let power_report_bytes = read_arg(args, "--power-report")?;
    let protocol_bytes = read_arg(args, "--protocol")?;
    let split_bytes = read_arg(args, "--splits")?;
    let receipt_bytes = read_arg(args, "--baseline-receipt")?;

    let config: RunConfig =
        serde_json::from_slice(&config_bytes).context("parse frozen run config JSON")?;
    validate_config(&config, &protocol_bytes, &split_bytes, &receipt_bytes)?;
    let expected_power_digest = config
        .power_report_sha256
        .as_deref()
        .context("frozen run config lacks power_report_sha256; run validation-only power planning before test analysis")?;
    if expected_power_digest != sha256_hex(&power_report_bytes) {
        bail!("power report digest does not match the frozen run config");
    }

    let split_manifest: SplitManifest =
        serde_json::from_slice(&split_bytes).context("parse frozen split manifest JSON")?;
    let receipt: BaselineSelectionReceipt =
        serde_json::from_slice(&receipt_bytes).context("parse baseline selection receipt JSON")?;
    validate_baseline_receipt(&receipt, &validation_bytes)?;

    let families: BTreeMap<String, String> = split_manifest
        .families
        .iter()
        .map(|family| (family.family_id.clone(), family.split.clone()))
        .collect();

    let validation_outcomes = parse_target_jsonl(&validation_bytes, "validation")?;
    let eligible_validation =
        validate_outcomes(&validation_outcomes, &config, &families, "validation")?;
    if eligible_validation.is_empty() {
        bail!("no eligible validation targets supplied to select the baseline");
    }
    let validation_scores = compute_baseline_scores(&eligible_validation)?;
    validate_baseline_scores(&receipt, &validation_scores)?;
    let power_report: PowerReport = serde_json::from_slice(&power_report_bytes)
        .context("parse frozen power-planning report JSON")?;
    validate_power_report(
        &power_report,
        &config,
        &validation_bytes,
        &split_manifest,
        &receipt,
        &eligible_validation,
        &validation_scores,
    )?;
    let selected_baseline = receipt.selected_arm.clone();

    let results_digest = sha256_hex(&results_bytes);
    let outcomes = parse_target_jsonl(&results_bytes, "test")?;
    let eligible = validate_outcomes(&outcomes, &config, &families, "test")?;
    if eligible.is_empty() {
        bail!("no eligible held-out targets remain; report exclusions, do not compute a win");
    }

    let eligible_family_count = eligible
        .iter()
        .map(|target| target.family_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let family_aggregates = compute_family_aggregates(&eligible, &selected_baseline)?;
    let arm_metrics = compute_arm_metrics(&eligible);
    let (effect, lower, upper, pairs, clusters) = paired_cluster_bootstrap(
        &family_aggregates,
        config.bootstrap_replicates,
        config.bootstrap_seed,
    )?;

    let (interval_informative, interval_positive, passed, gate_reason) =
        confirmatory_utility_gate(effect, lower, upper, clusters);
    let gate_decision = gate_reason.to_string();

    let interpretation = if passed {
        "The paired, family-clustered estimate met the preregistered effect and informative-interval gates on this test partition. This is within-benchmark utility only; it is not external/SOTA evidence."
    } else if clusters < MIN_CONFIRMATORY_FAMILY_CLUSTERS {
        "The eligible independent-family count is below the v0 floor of 20. Treat this as an underpowered pilot; the floor itself is not evidence of adequate power, so simulation-based power planning is still required."
    } else if !interval_informative {
        "The paired cluster-bootstrap interval is zero-width and has no empirical resolution. No utility claim is permitted even if its point estimate exceeds the effect threshold; inspect the component outcomes and collect additional independent evidence."
    } else {
        "The preregistered utility gate was not met. Report the estimate, interval, denominators, and failures; do not claim a win or relax the gate after seeing test outcomes."
    }
    .to_string();

    let report = AnalysisReport {
        schema_version: ANALYSIS_REPORT_SCHEMA,
        protocol_id: config.protocol_id,
        protocol_sha256: config.protocol_sha256,
        run_config_sha256: sha256_hex(&config_bytes),
        power_report_sha256: sha256_hex(&power_report_bytes),
        validation_results_sha256: sha256_hex(&validation_bytes),
        results_jsonl_sha256: results_digest,
        analyzer_source_sha256: sha256_hex(include_bytes!("analyze_pitb_results.rs")),
        split_manifest_sha256: config.split_manifest_sha256,
        partition_digest_sha256: config.partition_digest_sha256,
        baseline_selection_receipt_sha256: config.baseline_selection_receipt_sha256,
        upstream_commit: split_manifest.upstream.commit,
        selected_baseline_arm: selected_baseline.clone(),
        treatment_arm: TREATMENT_ARM,
        total_test_families_in_frozen_split: split_manifest
            .families
            .iter()
            .filter(|family| family.split == "test")
            .count(),
        eligible_validation_target_count: eligible_validation.len(),
        excluded_validation_target_count: validation_outcomes
            .iter()
            .filter(|row| row.eligibility == Eligibility::Excluded)
            .count(),
        unknown_validation_target_count: validation_outcomes
            .iter()
            .filter(|row| row.eligibility == Eligibility::Unknown)
            .count(),
        validation_baseline_scores: validation_scores,
        eligible_target_count: eligible.len(),
        eligible_family_count,
        excluded_target_count: outcomes
            .iter()
            .filter(|row| row.eligibility == Eligibility::Excluded)
            .count(),
        unknown_eligibility_target_count: outcomes
            .iter()
            .filter(|row| row.eligibility == Eligibility::Unknown)
            .count(),
        known_exposed_validation_family_ids: config.known_exposed_validation_family_ids.clone(),
        known_exposed_test_family_ids: config.known_exposed_test_family_ids.clone(),
        bootstrap_replicates: config.bootstrap_replicates,
        bootstrap_seed: config.bootstrap_seed,
        family_aggregates,
        arm_metrics,
        primary_comparison: PrimaryComparison {
            metric: "semantic_gate_verified_exact_target_success",
            confidence_interval_method: "paired target-delta percentile bootstrap; resample families with replacement; SplitMix64; linear-interpolated 2.5th/97.5th quantiles",
            baseline_arm: selected_baseline,
            treatment_arm: TREATMENT_ARM,
            paired_absolute_effect_percentage_points: effect * 100.0,
            confidence_level: 0.95,
            clustered_bootstrap_lower_percentage_points: lower * 100.0,
            clustered_bootstrap_upper_percentage_points: upper * 100.0,
            target_pairs: pairs,
            family_clusters: clusters,
            minimum_family_clusters_for_confirmatory_claim: MIN_CONFIRMATORY_FAMILY_CLUSTERS,
            minimum_effect_percentage_points: 10.0,
            confidence_interval_informative: interval_informative,
            confidence_interval_excludes_zero_positive: interval_positive,
            preregistered_utility_gate_passed: passed,
            gate_decision,
        },
        external_replication: "not_evaluated",
        interpretation,
    };
    let out_path = args.get("--out").context("missing --out path")?;
    let mut serialized = serde_json::to_vec_pretty(&report).context("serialize analysis report")?;
    serialized.push(b'\n');
    fs::write(out_path, serialized)
        .with_context(|| format!("write analysis report to {out_path}"))?;
    println!(
        "PITB_ANALYSIS_COMPLETE targets={} families={} baseline={} effect_pp={:.3} ci95=[{:.3},{:.3}] gate={}",
        report.eligible_target_count,
        report.eligible_family_count,
        report.selected_baseline_arm,
        report
            .primary_comparison
            .paired_absolute_effect_percentage_points,
        report
            .primary_comparison
            .clustered_bootstrap_lower_percentage_points,
        report
            .primary_comparison
            .clustered_bootstrap_upper_percentage_points,
        report.primary_comparison.gate_decision,
    );
    Ok(())
}

fn execute_power(args: &BTreeMap<String, String>) -> Result<()> {
    let config_bytes = read_arg(args, "--config")?;
    let validation_bytes = read_arg(args, "--validation-results")?;
    let protocol_bytes = read_arg(args, "--protocol")?;
    let split_bytes = read_arg(args, "--splits")?;
    let receipt_bytes = read_arg(args, "--baseline-receipt")?;
    let simulations = args
        .get("--power-simulations")
        .context("missing --power-simulations")?
        .parse::<usize>()
        .context("parse --power-simulations")?;
    let bootstrap_replicates = args
        .get("--power-bootstrap-replicates")
        .context("missing --power-bootstrap-replicates")?
        .parse::<usize>()
        .context("parse --power-bootstrap-replicates")?;

    let config: RunConfig =
        serde_json::from_slice(&config_bytes).context("parse frozen run config JSON")?;
    if config.power_report_sha256.is_some() {
        bail!("power planning must use the run design before its power-report digest is added");
    }
    validate_config(&config, &protocol_bytes, &split_bytes, &receipt_bytes)?;
    let split_manifest: SplitManifest =
        serde_json::from_slice(&split_bytes).context("parse frozen split manifest JSON")?;
    let receipt: BaselineSelectionReceipt =
        serde_json::from_slice(&receipt_bytes).context("parse baseline-selection receipt JSON")?;
    validate_baseline_receipt(&receipt, &validation_bytes)?;
    let frozen_families: BTreeMap<String, String> = split_manifest
        .families
        .iter()
        .map(|family| (family.family_id.clone(), family.split.clone()))
        .collect();
    let validation_outcomes = parse_target_jsonl(&validation_bytes, "validation")?;
    let eligible_validation = validate_outcomes(
        &validation_outcomes,
        &config,
        &frozen_families,
        "validation",
    )?;
    if eligible_validation.is_empty() {
        bail!("no eligible validation targets supplied for power planning");
    }
    let baseline_scores = compute_baseline_scores(&eligible_validation)?;
    validate_baseline_scores(&receipt, &baseline_scores)?;
    let baseline_rate = *baseline_scores
        .get(&receipt.selected_arm)
        .context("selected validation baseline has no computed score")?;
    let mut family_sizes: BTreeMap<String, usize> = BTreeMap::new();
    for target in &eligible_validation {
        *family_sizes.entry(target.family_id.clone()).or_default() += 1;
    }
    if family_sizes.is_empty() || family_sizes.values().any(|count| *count == 0) {
        bail!("validation-derived family-size distribution is empty or invalid");
    }
    let family_target_counts: Vec<FamilyTargetCount> = family_sizes
        .into_iter()
        .map(|(family_id, eligible_target_count)| FamilyTargetCount {
            family_id,
            eligible_target_count,
        })
        .collect();

    let effect_grid = [0.0_f64, 0.10, 0.15, 0.20];
    let icc_grid = [0.0_f64, 0.05, 0.10, 0.20];
    let pair_association_grid = [0.0_f64, 0.5, 0.9];
    let mut scenarios = Vec::with_capacity(
        effect_grid.len() * icc_grid.len() * pair_association_grid.len(),
    );
    for (effect_index, effect) in effect_grid.iter().copied().enumerate() {
        for (icc_index, icc) in icc_grid.iter().copied().enumerate() {
            for (association_index, association) in pair_association_grid.iter().copied().enumerate() {
                let scenario_index = effect_index * icc_grid.len() * pair_association_grid.len()
                    + icc_index * pair_association_grid.len()
                    + association_index;
                let scenario_seed = config.bootstrap_seed.wrapping_add(
                    scenario_index as u64 * 0x9E37_79B9_7F4A_7C15,
                );
                scenarios.push(simulate_power_scenario(
                    &family_target_counts,
                    baseline_rate,
                    effect,
                    icc,
                    association,
                    simulations,
                    bootstrap_replicates,
                    scenario_seed,
                )?);
            }
        }
    }
    let weak_null_sd_grid_pp = [5.0_f64, 10.0, 20.0];
    let weak_null_pair_association_grid = pair_association_grid;
    let mut weak_null_scenarios = Vec::with_capacity(
        weak_null_sd_grid_pp.len() * icc_grid.len() * weak_null_pair_association_grid.len(),
    );
    for (sd_index, sd_pp) in weak_null_sd_grid_pp.iter().copied().enumerate() {
        for (icc_index, icc) in icc_grid.iter().copied().enumerate() {
            for (association_index, association) in
                weak_null_pair_association_grid.iter().copied().enumerate()
            {
                let scenario_index = sd_index * icc_grid.len() * weak_null_pair_association_grid.len()
                    + icc_index * weak_null_pair_association_grid.len()
                    + association_index;
                let scenario_seed = config
                    .bootstrap_seed
                    .wrapping_add(0xD1B5_4A32_D192_ED03)
                    .wrapping_add(scenario_index as u64 * 0x9E37_79B9_7F4A_7C15);
                weak_null_scenarios.push(simulate_weak_null_scenario(
                    &family_target_counts,
                    baseline_rate,
                    sd_pp / 100.0,
                    icc,
                    association,
                    simulations,
                    bootstrap_replicates,
                    scenario_seed,
                )?);
            }
        }
    }
    let weak_null_false_claim_probability_max = weak_null_scenarios
        .iter()
        .filter_map(|scenario| scenario.estimated_false_claim_probability)
        .reduce(f64::max);
    let weak_null_false_claim_probability_upper_simultaneous_95_max = weak_null_scenarios
        .iter()
        .filter_map(|scenario| scenario.false_claim_probability_upper_simultaneous_95)
        .reduce(f64::max);
    let null_effect_false_claim_probability_max = scenarios
        .iter()
        .filter(|scenario| scenario.true_treatment_effect_percentage_points == 0.0)
        .filter_map(|scenario| scenario.estimated_utility_gate_power)
        .reduce(f64::max);
    let null_effect_false_claim_probability_upper_simultaneous_95_max = scenarios
        .iter()
        .filter(|scenario| scenario.true_treatment_effect_percentage_points == 0.0)
        .filter_map(|scenario| scenario.null_effect_false_claim_probability_upper_simultaneous_95)
        .reduce(f64::max);
    let twenty_point_scenarios: Vec<&PowerScenario> = scenarios
        .iter()
        .filter(|scenario| {
            (scenario.true_treatment_effect_percentage_points - 20.0).abs() < f64::EPSILON
                && (scenario.assumed_family_icc - 0.20).abs() < f64::EPSILON
        })
        .collect();
    let all_worst_case_associations_estimated =
        twenty_point_scenarios.len() == pair_association_grid.len()
            && twenty_point_scenarios.iter().all(|scenario| {
                scenario.estimated_utility_gate_power.is_some()
                    && scenario.estimated_power_lower_95.is_some()
            });
    // Require the power floor at the worst registered pairing dependence, not
    // just whichever scenario happens to appear first in the report.
    let estimated_power_at_20pp_and_icc_020 = if all_worst_case_associations_estimated {
        twenty_point_scenarios
            .iter()
            .filter_map(|scenario| scenario.estimated_utility_gate_power)
            .reduce(f64::min)
    } else {
        None
    };
    let estimated_power_at_20pp_and_icc_020_lower_95 = if all_worst_case_associations_estimated {
        twenty_point_scenarios
            .iter()
            .filter_map(|scenario| scenario.estimated_power_lower_95)
            .reduce(f64::min)
    } else {
        None
    };
    // The confirmatory design requirement is simultaneous across all three
    // preregistered dependence settings, not a minimum of three unadjusted
    // per-cell 95% lower bounds.
    let estimated_power_at_20pp_and_icc_020_lower_simultaneous_95 =
        if all_worst_case_associations_estimated {
            twenty_point_scenarios
                .iter()
                .map(|scenario| {
                    simultaneous_power_lower_bound(
                        scenario.utility_gate_passes,
                        scenario.study_simulations,
                    )
                })
                .reduce(f64::min)
        } else {
            None
        };
    let power_gate_passed = family_target_counts.len() >= MIN_CONFIRMATORY_FAMILY_CLUSTERS
        && estimated_power_at_20pp_and_icc_020_lower_simultaneous_95
            .is_some_and(|power| power >= 0.80)
        && null_effect_false_claim_probability_upper_simultaneous_95_max
            .is_some_and(|rate| rate <= 0.05)
        && weak_null_false_claim_probability_upper_simultaneous_95_max
            .is_some_and(|rate| rate <= 0.05);

    let interpretation = if power_gate_passed {
        "The validation-derived design met the preregistered Monte Carlo confidence-bound gates for a +20pp effect at ICC 0.20, the strict-null scenarios, and the family-heterogeneous weak-null grid. This is a design-planning result, not evidence that insight transfer succeeds."
    } else if family_target_counts.len() < MIN_CONFIRMATORY_FAMILY_CLUSTERS {
        "The validation-derived eligible sample has fewer than 20 independent families. The design is underpowered by the frozen project floor and cannot support a confirmatory claim."
    } else {
        "The simulated design did not meet the power gate, strict-null calibration, or family-heterogeneous weak-null false-claim gate, or the +20pp scenario was infeasible at the observed baseline. Expand the separately sourced eligible set or report the eventual experiment as a pilot; do not lower the gates."
    }
    .to_string();

    let design_digest = run_design_sha256(&config)?;
    let report = PowerReport {
        schema_version: "pitb.power-report.v2".to_string(),
        protocol_id: config.protocol_id,
        protocol_sha256: config.protocol_sha256,
        run_design_sha256: design_digest,
        validation_results_sha256: sha256_hex(&validation_bytes),
        baseline_selection_receipt_sha256: config.baseline_selection_receipt_sha256,
        split_manifest_sha256: config.split_manifest_sha256,
        partition_digest_sha256: config.partition_digest_sha256,
        analyzer_source_sha256: sha256_hex(include_bytes!("analyze_pitb_results.rs")),
        upstream_commit: split_manifest.upstream.commit,
        selected_baseline_arm: receipt.selected_arm,
        known_exposed_validation_family_ids: config.known_exposed_validation_family_ids.clone(),
        known_exposed_test_family_ids: config.known_exposed_test_family_ids.clone(),
        eligible_validation_target_count: eligible_validation.len(),
        eligible_validation_family_count: family_target_counts.len(),
        validation_baseline_semantic_success_rate: baseline_rate,
        family_target_counts,
        assumed_family_icc_grid: icc_grid.to_vec(),
        pair_association_fraction_grid: pair_association_grid.to_vec(),
        true_effect_grid_percentage_points: effect_grid.map(|value| value * 100.0).to_vec(),
        study_simulations: simulations,
        bootstrap_replicates_per_simulated_study: bootstrap_replicates,
        simulation_seed: config.bootstrap_seed,
        confirmatory_minimum_family_count: MIN_CONFIRMATORY_FAMILY_CLUSTERS,
        utility_effect_threshold_percentage_points: 10.0,
        required_power_at_20pp_and_icc_020: 0.80,
        maximum_false_claim_probability_at_zero_effect: 0.05,
        null_effect_false_claim_probability_max,
        null_effect_false_claim_probability_upper_simultaneous_95_max,
        weak_null_family_effect_heterogeneity_sd_grid_percentage_points: weak_null_sd_grid_pp.to_vec(),
        weak_null_pair_association_fraction_grid: weak_null_pair_association_grid.to_vec(),
        weak_null_scenarios,
        weak_null_false_claim_probability_max,
        weak_null_false_claim_probability_upper_simultaneous_95_max,
        estimated_power_at_20pp_and_icc_020,
        estimated_power_at_20pp_and_icc_020_lower_95,
        estimated_power_at_20pp_and_icc_020_lower_simultaneous_95,
        power_gate_passed,
        scenarios,
        model_assumptions: vec![
            "Family sizes are taken from eligible validation outcomes and reused as the projected evaluation-family-size distribution; test eligibility may differ.".to_string(),
            "A beta-binomial family random effect induces the specified baseline Bernoulli ICC; treatment success shares the same latent family difficulty with its marginal success rate shifted by the scenario effect.".to_string(),
            "The strict-null and positive-effect power grid samples paired outcomes using the same bivariate Bernoulli coupling as the weak-null grid. Its association-fraction levels are 0, 0.5, and 0.9; zero gives conditional independence and higher values approach the Frechet upper bound. This is a scenario grid, not an empirically estimated correlation.".to_string(),
            "A separate weak-null grid assigns normally generated family-level treatment-effect offsets, recenters them to exactly zero target-weighted mean within each simulated study, and uniformly scales them only as needed to keep both arm probabilities in [0,1]. The requested effect SD grid is 5, 10, and 20 percentage points, ICC grid is 0, 0.05, 0.10, 0.20, and pair-association fractions are 0, 0.5, 0.9; boundary scaling can reduce realized heterogeneity and is reported.".to_string(),
            "Weak-null paired outcomes use a bivariate Bernoulli coupling with the prescribed marginal success rates; the association fraction interpolates joint-success probability from independence to the Frechet upper bound and is not a Pearson correlation. The 36-cell weak-null grid tests false-claim probability when positive and negative family effects cancel on average, under those explicit pair-association and ICC values.".to_string(),
            "This finite family-size stress model does not cover selective missingness, alternative family-size mechanisms, or every dependence structure and is not proof of nominal coverage universally.".to_string(),
            "Each simulated study uses the same target-weighted paired effect, family-with-replacement bootstrap, linear-interpolated percentile lower bound, +10pp point-estimate gate, positive lower-bound gate, and 20-family floor as the confirmatory analyzer. Bootstrap resample count is separately reported.".to_string(),
            "Power estimates and their Monte Carlo standard errors are design simulations only. The strict null has 4 ICCs × 3 association fractions = 12 cells; the heterogeneous weak null has 3 effect SDs × 4 ICCs × 3 association fractions = 36 cells. The one-sided Wilson upper bound uses z=3.08 across all 48 registered null cells, conservatively exceeding the Bonferroni critical value for alpha=0.05/48. The +20pp/ICC .20 design gate uses the minimum two-sided 95% Bonferroni Wilson lower bound (z=2.394) across all three preregistered association settings. These are model-conditional design diagnostics, not measured proof-insight utility or an SOTA comparison.".to_string(),
        ],
        interpretation,
    };
    let out_path = args.get("--out").context("missing --out path")?;
    let mut serialized = serde_json::to_vec_pretty(&report).context("serialize power report")?;
    serialized.push(b'\n');
    fs::write(out_path, serialized).with_context(|| format!("write power report to {out_path}"))?;
    println!(
        "PITB_POWER_PLAN_COMPLETE baseline={} validation_targets={} eligible_families={} p0={:.3} simulations={} bootstrap_replicates={} worst_assoc_power20pp_icc020={} worst_assoc_lower95={} worst_assoc_lower_simultaneous95={} strict_null_false_claim_upper_simultaneous95={} weak_null_false_claim_upper_simultaneous95={} gate={}",
        report.selected_baseline_arm,
        report.eligible_validation_target_count,
        report.eligible_validation_family_count,
        report.validation_baseline_semantic_success_rate,
        report.study_simulations,
        report.bootstrap_replicates_per_simulated_study,
        report
            .estimated_power_at_20pp_and_icc_020
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "infeasible_or_unestimated".to_string()),
        report
            .estimated_power_at_20pp_and_icc_020_lower_95
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "unestimated".to_string()),
        report
            .estimated_power_at_20pp_and_icc_020_lower_simultaneous_95
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "unestimated".to_string()),
        report
            .null_effect_false_claim_probability_upper_simultaneous_95_max
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "unestimated".to_string()),
        report
            .weak_null_false_claim_probability_upper_simultaneous_95_max
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "unestimated".to_string()),
        if report.power_gate_passed {
            "pass"
        } else {
            "no_go"
        }
    );
    Ok(())
}

fn wilson_interval(successes: usize, trials: usize) -> (f64, f64) {
    wilson_interval_with_z(successes, trials, 1.959_963_984_540_054_f64)
}

fn wilson_interval_with_z(successes: usize, trials: usize, z: f64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z_squared = z * z;
    let denominator = 1.0 + z_squared / n;
    let center = (p + z_squared / (2.0 * n)) / denominator;
    let half_width = z * (p * (1.0 - p) / n + z_squared / (4.0 * n * n)).sqrt() / denominator;
    (
        (center - half_width).max(0.0),
        (center + half_width).min(1.0),
    )
}

// The registered null grid has 48 cells: 4 strict-null ICCs × 3
// paired-outcome association fractions plus 3 family-effect SDs × 4 ICCs × 3
// association fractions for the heterogeneous weak null. z=3.08 conservatively
// exceeds the one-sided Bonferroni critical value for alpha=.05/48.
fn simultaneous_null_upper_bound(successes: usize, trials: usize) -> f64 {
    wilson_interval_with_z(successes, trials, 3.08).1
}

// Conservative two-sided 95% Bonferroni lower Wilson bound across the three
// preregistered paired-outcome-association settings for the +20pp / ICC .20
// design gate. z=2.394 exceeds Phi^-1(1 - .05/(2*3)).
fn simultaneous_power_lower_bound(successes: usize, trials: usize) -> f64 {
    wilson_interval_with_z(successes, trials, POWER_ASSOCIATION_SIMULTANEOUS_Z).0
}

fn simulate_power_scenario(
    family_target_counts: &[FamilyTargetCount],
    baseline_rate: f64,
    true_effect: f64,
    icc: f64,
    pair_association_fraction: f64,
    simulations: usize,
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<PowerScenario> {
    if family_target_counts.is_empty() {
        bail!("power simulation requires at least one eligible family");
    }
    if !(0.0..1.0).contains(&icc) {
        bail!("family ICC must lie in [0,1)");
    }
    if !pair_association_fraction.is_finite()
        || !(0.0..=1.0).contains(&pair_association_fraction)
    {
        bail!("pair-association fraction must lie in [0,1]");
    }
    if !baseline_rate.is_finite()
        || !(0.0..=1.0).contains(&baseline_rate)
        || !true_effect.is_finite()
        || !(0.0..=1.0).contains(&true_effect)
    {
        bail!("baseline rate and true effect must be finite probabilities");
    }
    if simulations == 0 || bootstrap_replicates == 0 {
        bail!("simulation and bootstrap replicate counts must be positive");
    }
    let treatment_rate = baseline_rate + true_effect;
    if treatment_rate > 1.0 {
        return Ok(PowerScenario {
            status: "infeasible_effect_above_ceiling".to_string(),
            true_treatment_effect_percentage_points: true_effect * 100.0,
            assumed_family_icc: icc,
            pair_association_fraction,
            study_simulations: simulations,
            bootstrap_replicates_per_simulated_study: bootstrap_replicates,
            utility_gate_passes: 0,
            degenerate_interval_trials: 0,
            estimated_utility_gate_power: None,
            monte_carlo_standard_error: None,
            estimated_power_lower_95: None,
            estimated_power_upper_95: None,
            null_effect_false_claim_probability_upper_simultaneous_95: None,
            mean_observed_effect_percentage_points: None,
            mean_lower_95ci_percentage_points: None,
            interpretation: "The requested positive effect would imply a treatment success rate above 100% at this validation baseline.".to_string(),
        });
    }

    let clusters = family_target_counts.len();
    let total_targets: usize = family_target_counts
        .iter()
        .map(|family| family.eligible_target_count)
        .sum();
    let mut rng = SplitMix64 { state: seed };
    let mut passes = 0_usize;
    let mut degenerate_interval_trials = 0_usize;
    let mut sum_effect = 0.0;
    let mut sum_lower = 0.0;
    for _ in 0..simulations {
        let simulated_families = simulate_matched_study(
            family_target_counts,
            baseline_rate,
            treatment_rate,
            icc,
            pair_association_fraction,
            &mut rng,
        )?;
        let total_delta: i64 = simulated_families
            .iter()
            .map(|family| family.paired_delta_sum)
            .sum();
        let observed_effect = total_delta as f64 / total_targets as f64;
        let [lower, upper] = simulated_cluster_bootstrap_interval(
            &simulated_families,
            bootstrap_replicates,
            &mut rng,
        );
        let interval_informative = upper - lower > f64::EPSILON;
        if !interval_informative {
            degenerate_interval_trials += 1;
        }
        sum_effect += observed_effect;
        sum_lower += lower;
        if confirmatory_utility_gate(observed_effect, lower, upper, clusters).2 {
            passes += 1;
        }
    }
    let power = passes as f64 / simulations as f64;
    Ok(PowerScenario {
        status: "estimated".to_string(),
        true_treatment_effect_percentage_points: true_effect * 100.0,
        assumed_family_icc: icc,
        pair_association_fraction,
        study_simulations: simulations,
        bootstrap_replicates_per_simulated_study: bootstrap_replicates,
        utility_gate_passes: passes,
        degenerate_interval_trials,
        estimated_utility_gate_power: Some(power),
        monte_carlo_standard_error: Some((power * (1.0 - power) / simulations as f64).sqrt()),
        estimated_power_lower_95: Some(wilson_interval(passes, simulations).0),
        estimated_power_upper_95: Some(wilson_interval(passes, simulations).1),
        null_effect_false_claim_probability_upper_simultaneous_95: if true_effect.abs() <= 1e-12 {
            Some(simultaneous_null_upper_bound(passes, simulations))
        } else {
            None
        },
        mean_observed_effect_percentage_points: Some(sum_effect / simulations as f64 * 100.0),
        mean_lower_95ci_percentage_points: Some(sum_lower / simulations as f64 * 100.0),
        interpretation: if clusters < MIN_CONFIRMATORY_FAMILY_CLUSTERS {
            "Power is structurally zero for the confirmatory gate because the eligible validation design contains fewer than 20 independent family clusters.".to_string()
        } else {
            "Estimated fraction of synthetic matched studies passing the complete primary utility gate under the declared baseline, effect, ICC, and validation-derived family sizes.".to_string()
        },
    })
}

fn simulate_weak_null_scenario(
    family_target_counts: &[FamilyTargetCount],
    baseline_rate: f64,
    requested_effect_sd: f64,
    icc: f64,
    pair_association_fraction: f64,
    simulations: usize,
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<WeakNullScenario> {
    if family_target_counts.is_empty()
        || family_target_counts
            .iter()
            .any(|family| family.eligible_target_count == 0)
    {
        bail!("weak-null calibration requires nonempty family sizes");
    }
    if !baseline_rate.is_finite() || !(0.0..=1.0).contains(&baseline_rate) {
        bail!("weak-null baseline rate must be a finite probability");
    }
    if !requested_effect_sd.is_finite() || requested_effect_sd < 0.0 {
        bail!("weak-null family-effect SD must be finite and nonnegative");
    }
    if !icc.is_finite() || !(0.0..1.0).contains(&icc) {
        bail!("weak-null family ICC must lie in [0,1)");
    }
    if !pair_association_fraction.is_finite()
        || !(0.0..=1.0).contains(&pair_association_fraction)
    {
        bail!("weak-null pair-association fraction must lie in [0,1]");
    }
    if simulations == 0 || bootstrap_replicates == 0 {
        bail!("weak-null simulation and bootstrap counts must be positive");
    }

    let total_targets: usize = family_target_counts
        .iter()
        .map(|family| family.eligible_target_count)
        .sum();
    if total_targets == 0 {
        bail!("weak-null calibration has zero eligible targets");
    }

    let mut rng = SplitMix64 { state: seed };
    let mut false_claims = 0_usize;
    let mut degenerate_interval_trials = 0_usize;
    let mut sum_effect = 0.0;
    let mut sum_realized_effect_sd = 0.0;
    for _ in 0..simulations {
        let (simulated_families, realized_effect_sd) = simulate_weak_null_matched_study(
            family_target_counts,
            baseline_rate,
            requested_effect_sd,
            icc,
            pair_association_fraction,
            &mut rng,
        )?;
        let total_delta: i64 = simulated_families
            .iter()
            .map(|family| family.paired_delta_sum)
            .sum();
        let observed_effect = total_delta as f64 / total_targets as f64;
        let [lower, upper] = simulated_cluster_bootstrap_interval(
            &simulated_families,
            bootstrap_replicates,
            &mut rng,
        );
        if upper - lower <= f64::EPSILON {
            degenerate_interval_trials += 1;
        }
        sum_effect += observed_effect;
        sum_realized_effect_sd += realized_effect_sd;
        if confirmatory_utility_gate(
            observed_effect,
            lower,
            upper,
            family_target_counts.len(),
        )
        .2
        {
            false_claims += 1;
        }
    }

    let false_claim_probability = false_claims as f64 / simulations as f64;
    let (false_claim_lower, false_claim_upper) = wilson_interval(false_claims, simulations);
    Ok(WeakNullScenario {
        status: "estimated".to_string(),
        true_target_weighted_mean_effect_percentage_points: 0.0,
        assumed_family_icc: icc,
        pair_association_fraction,
        requested_family_effect_heterogeneity_sd_percentage_points: requested_effect_sd * 100.0,
        study_simulations: simulations,
        bootstrap_replicates_per_simulated_study: bootstrap_replicates,
        false_claims,
        degenerate_interval_trials,
        estimated_false_claim_probability: Some(false_claim_probability),
        monte_carlo_standard_error: Some(
            (false_claim_probability * (1.0 - false_claim_probability) / simulations as f64)
                .sqrt(),
        ),
        false_claim_probability_lower_95: Some(false_claim_lower),
        false_claim_probability_upper_95: Some(false_claim_upper),
        false_claim_probability_upper_simultaneous_95: Some(
            simultaneous_null_upper_bound(false_claims, simulations),
        ),
        mean_observed_effect_percentage_points: Some(sum_effect / simulations as f64 * 100.0),
        mean_realized_family_effect_heterogeneity_sd_percentage_points: Some(
            sum_realized_effect_sd / simulations as f64 * 100.0,
        ),
        interpretation: "Weak-null calibration: treatment effects vary by family but are recentered to exactly zero target-weighted mean within each simulated study. The paired Bernoulli coupling varies from independent potential outcomes to the Frechet upper bound per the registered association fraction. This does not cover every dependence, selective-missingness, or family-size-generating mechanism.".to_string(),
    })
}

fn simulate_weak_null_matched_study(
    family_target_counts: &[FamilyTargetCount],
    baseline_rate: f64,
    requested_effect_sd: f64,
    icc: f64,
    pair_association_fraction: f64,
    rng: &mut SplitMix64,
) -> Result<(Vec<SimulatedFamily>, f64)> {
    let baseline_rates: Vec<f64> = family_target_counts
        .iter()
        .map(|_| sample_family_rate(baseline_rate, icc, rng))
        .collect();
    let treatment_effects = sample_centered_weak_null_family_effects(
        family_target_counts,
        &baseline_rates,
        requested_effect_sd,
        rng,
    )?;
    let realized_effect_sd = weighted_family_effect_sd(&treatment_effects, family_target_counts);
    let mut families = Vec::with_capacity(family_target_counts.len());

    for ((family, baseline_family_rate), effect) in family_target_counts
        .iter()
        .zip(baseline_rates)
        .zip(treatment_effects)
    {
        let treatment_family_rate = (baseline_family_rate + effect).clamp(0.0, 1.0);
        let joint_success_probability = joint_success_probability_at_association(
            baseline_family_rate,
            treatment_family_rate,
            pair_association_fraction,
        )?;
        let mut paired_delta_sum = 0_i64;
        for _ in 0..family.eligible_target_count {
            let (baseline_success, treatment_success) = sample_paired_bernoulli_from_joint_probability(
                baseline_family_rate,
                treatment_family_rate,
                joint_success_probability,
                rng,
            );
            paired_delta_sum += (if treatment_success { 1_i64 } else { 0_i64 })
                - (if baseline_success { 1_i64 } else { 0_i64 });
        }
        families.push(SimulatedFamily {
            target_count: family.eligible_target_count,
            paired_delta_sum,
        });
    }
    Ok((families, realized_effect_sd))
}

fn joint_success_probability_at_association(
    baseline_probability: f64,
    treatment_probability: f64,
    pair_association_fraction: f64,
) -> Result<f64> {
    if !baseline_probability.is_finite()
        || !(0.0..=1.0).contains(&baseline_probability)
        || !treatment_probability.is_finite()
        || !(0.0..=1.0).contains(&treatment_probability)
    {
        bail!("paired Bernoulli marginals must be finite probabilities");
    }
    if !pair_association_fraction.is_finite()
        || !(0.0..=1.0).contains(&pair_association_fraction)
    {
        bail!("pair-association fraction must lie in [0,1]");
    }

    // Association fraction 0 gives independent potential outcomes. Fraction 1
    // reaches the Frechet upper bound P(Y0=1,Y1=1)=min(p0,p1). Intermediate
    // values interpolate joint success probability; this is not a Pearson rho.
    let independent_joint = baseline_probability * treatment_probability;
    let lower_frechet = (baseline_probability + treatment_probability - 1.0).max(0.0);
    let upper_frechet = baseline_probability.min(treatment_probability);
    let joint = independent_joint
        + pair_association_fraction * (upper_frechet - independent_joint);
    // Roundoff near fraction 1 must not violate the Frechet bounds used by the
    // categorical sampler's debug assertions.
    Ok(joint.clamp(lower_frechet, upper_frechet))
}

fn sample_paired_bernoulli_from_joint_probability(
    baseline_probability: f64,
    treatment_probability: f64,
    joint_success_probability: f64,
    rng: &mut SplitMix64,
) -> (bool, bool) {
    debug_assert!((0.0..=1.0).contains(&baseline_probability));
    debug_assert!((0.0..=1.0).contains(&treatment_probability));
    debug_assert!((0.0..=baseline_probability.min(treatment_probability))
        .contains(&joint_success_probability));
    let draw = rng.uniform01();
    if draw < joint_success_probability {
        (true, true)
    } else if draw < baseline_probability {
        (true, false)
    } else if draw < baseline_probability + treatment_probability - joint_success_probability {
        (false, true)
    } else {
        (false, false)
    }
}

fn sample_centered_weak_null_family_effects(
    family_target_counts: &[FamilyTargetCount],
    baseline_family_rates: &[f64],
    requested_effect_sd: f64,
    rng: &mut SplitMix64,
) -> Result<Vec<f64>> {
    if family_target_counts.is_empty()
        || family_target_counts.len() != baseline_family_rates.len()
        || family_target_counts
            .iter()
            .any(|family| family.eligible_target_count == 0)
    {
        bail!("weak-null family counts and baseline rates must be nonempty and aligned");
    }
    if !requested_effect_sd.is_finite() || requested_effect_sd < 0.0 {
        bail!("requested weak-null effect SD must be finite and nonnegative");
    }
    if baseline_family_rates
        .iter()
        .any(|rate| !rate.is_finite() || !(0.0..=1.0).contains(rate))
    {
        bail!("weak-null baseline family rates must be finite probabilities");
    }
    if requested_effect_sd == 0.0 {
        return Ok(vec![0.0; family_target_counts.len()]);
    }

    let total_targets: f64 = family_target_counts
        .iter()
        .map(|family| family.eligible_target_count as f64)
        .sum();
    let raw: Vec<f64> = (0..family_target_counts.len())
        .map(|_| rng.normal())
        .collect();
    let weighted_mean = raw
        .iter()
        .zip(family_target_counts)
        .map(|(value, family)| value * family.eligible_target_count as f64)
        .sum::<f64>()
        / total_targets;
    let centered: Vec<f64> = raw.iter().map(|value| value - weighted_mean).collect();
    let centered_sd = weighted_family_effect_sd(&centered, family_target_counts);
    if centered_sd <= f64::EPSILON {
        return Ok(vec![0.0; family_target_counts.len()]);
    }

    let standardized: Vec<f64> = centered
        .iter()
        .map(|value| value / centered_sd)
        .collect();
    // A single common scale factor preserves the exact target-weighted zero
    // mean while keeping every treatment probability inside [0,1]. The
    // realized effect SD is reported because boundary constraints can shrink
    // the requested heterogeneity.
    let mut feasible_scale = 1.0_f64;
    for (standard_effect, baseline_rate) in standardized.iter().zip(baseline_family_rates) {
        let proposed_effect = standard_effect * requested_effect_sd;
        if proposed_effect > 0.0 {
            feasible_scale = feasible_scale.min((1.0 - baseline_rate) / proposed_effect);
        } else if proposed_effect < 0.0 {
            feasible_scale = feasible_scale.min(baseline_rate / -proposed_effect);
        }
    }
    feasible_scale = feasible_scale.clamp(0.0, 1.0);
    Ok(standardized
        .iter()
        .map(|effect| effect * requested_effect_sd * feasible_scale)
        .collect())
}

fn weighted_family_effect_sd(
    effects: &[f64],
    family_target_counts: &[FamilyTargetCount],
) -> f64 {
    if effects.is_empty() || effects.len() != family_target_counts.len() {
        return 0.0;
    }
    let total_targets: f64 = family_target_counts
        .iter()
        .map(|family| family.eligible_target_count as f64)
        .sum();
    if total_targets <= 0.0 {
        return 0.0;
    }
    let mean = effects
        .iter()
        .zip(family_target_counts)
        .map(|(effect, family)| effect * family.eligible_target_count as f64)
        .sum::<f64>()
        / total_targets;
    let variance = effects
        .iter()
        .zip(family_target_counts)
        .map(|(effect, family)| {
            let difference = effect - mean;
            difference * difference * family.eligible_target_count as f64
        })
        .sum::<f64>()
        / total_targets;
    variance.max(0.0).sqrt()
}

fn simulate_matched_study(
    family_target_counts: &[FamilyTargetCount],
    baseline_rate: f64,
    treatment_rate: f64,
    icc: f64,
    pair_association_fraction: f64,
    rng: &mut SplitMix64,
) -> Result<Vec<SimulatedFamily>> {
    family_target_counts
        .iter()
        .map(|family| {
            let baseline_family_rate = sample_family_rate(baseline_rate, icc, rng);
            let treatment_family_rate = treatment_rate_given_baseline_family(
                baseline_family_rate,
                baseline_rate,
                treatment_rate,
            );
            let joint_success_probability = joint_success_probability_at_association(
                baseline_family_rate,
                treatment_family_rate,
                pair_association_fraction,
            )?;
            let mut delta_sum = 0_i64;
            for _ in 0..family.eligible_target_count {
                let (baseline_success, treatment_success) =
                    sample_paired_bernoulli_from_joint_probability(
                        baseline_family_rate,
                        treatment_family_rate,
                        joint_success_probability,
                        rng,
                    );
                delta_sum += (if treatment_success { 1_i64 } else { 0_i64 })
                    - (if baseline_success { 1_i64 } else { 0_i64 });
            }
            Ok(SimulatedFamily {
                target_count: family.eligible_target_count,
                paired_delta_sum: delta_sum,
            })
        })
        .collect()
}

fn sample_family_rate(mean: f64, icc: f64, rng: &mut SplitMix64) -> f64 {
    if mean <= 0.0 || mean >= 1.0 || icc <= 0.0 {
        return mean;
    }
    let concentration = 1.0 / icc - 1.0;
    let alpha = mean * concentration;
    let beta = (1.0 - mean) * concentration;
    let x = sample_gamma(alpha, rng);
    let y = sample_gamma(beta, rng);
    if x + y <= 0.0 { mean } else { x / (x + y) }
}

fn sample_gamma(shape: f64, rng: &mut SplitMix64) -> f64 {
    if shape < 1.0 {
        return sample_gamma(shape + 1.0, rng) * rng.uniform01().powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = rng.normal();
        let v0 = 1.0 + c * x;
        if v0 <= 0.0 {
            continue;
        }
        let v = v0 * v0 * v0;
        let u = rng.uniform01();
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

fn treatment_rate_given_baseline_family(
    baseline_family_rate: f64,
    baseline_rate: f64,
    treatment_rate: f64,
) -> f64 {
    if treatment_rate >= baseline_rate && baseline_rate < 1.0 {
        1.0 - (1.0 - baseline_family_rate) * (1.0 - treatment_rate) / (1.0 - baseline_rate)
    } else if treatment_rate < baseline_rate && baseline_rate > 0.0 {
        baseline_family_rate * treatment_rate / baseline_rate
    } else {
        treatment_rate
    }
}

fn simulated_cluster_bootstrap_interval(
    families: &[SimulatedFamily],
    replicates: usize,
    rng: &mut SplitMix64,
) -> [f64; 2] {
    let mut values = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut delta_sum = 0_i64;
        let mut target_count = 0_usize;
        for _ in 0..families.len() {
            let sampled = &families[rng.index(families.len())];
            delta_sum += sampled.paired_delta_sum;
            target_count += sampled.target_count;
        }
        values.push(delta_sum as f64 / target_count as f64);
    }
    values.sort_by(f64::total_cmp);
    [quantile(&values, 0.025), quantile(&values, 0.975)]
}

fn parse_args(args: Vec<String>) -> Result<BTreeMap<String, String>> {
    let mut parsed = BTreeMap::new();
    let mut iter = args.into_iter();
    while let Some(flag) = iter.next() {
        if ![
            "--mode",
            "--config",
            "--results",
            "--validation-results",
            "--protocol",
            "--splits",
            "--baseline-receipt",
            "--power-report",
            "--out",
            "--power-simulations",
            "--power-bootstrap-replicates",
        ]
        .contains(&flag.as_str())
        {
            bail!("unknown argument {flag}");
        }
        let value = iter
            .next()
            .with_context(|| format!("missing value after {flag}"))?;
        if parsed.insert(flag.clone(), value).is_some() {
            bail!("duplicate argument {flag}");
        }
    }

    let mode = parsed
        .get("--mode")
        .map(String::as_str)
        .unwrap_or("analyze");
    if !["analyze", "power"].contains(&mode) {
        bail!("unsupported mode {mode}; expected analyze or power");
    }
    for required in [
        "--config",
        "--validation-results",
        "--protocol",
        "--splits",
        "--baseline-receipt",
        "--out",
    ] {
        if !parsed.contains_key(required) {
            bail!("missing required argument {required}");
        }
    }

    match mode {
        "analyze" => {
            if !parsed.contains_key("--results") {
                bail!("analyze mode requires --results");
            }
            if !parsed.contains_key("--power-report") {
                bail!("analyze mode requires a content-pinned --power-report");
            }
            if parsed.contains_key("--power-simulations")
                || parsed.contains_key("--power-bootstrap-replicates")
            {
                bail!("power simulation options are only valid in --mode power");
            }
        }
        "power" => {
            if parsed.contains_key("--results") {
                bail!("power mode must not receive held-out test outcomes");
            }
            if parsed.contains_key("--power-report") {
                bail!("power mode creates the report; it must not consume one");
            }
            let simulations = parsed
                .get("--power-simulations")
                .context("power mode requires --power-simulations")?
                .parse::<usize>()
                .context("--power-simulations must be a positive integer")?;
            let bootstrap_replicates = parsed
                .get("--power-bootstrap-replicates")
                .context("power mode requires --power-bootstrap-replicates")?
                .parse::<usize>()
                .context("--power-bootstrap-replicates must be a positive integer")?;
            if simulations < 1_000 {
                bail!("confirmatory power planning requires at least 1000 simulated studies");
            }
            if bootstrap_replicates < 500 {
                bail!(
                    "power planning requires at least 500 bootstrap replicates per simulated study"
                );
            }
        }
        _ => unreachable!("mode was validated above"),
    }
    Ok(parsed)
}

fn read_arg(args: &BTreeMap<String, String>, key: &str) -> Result<Vec<u8>> {
    let path = args.get(key).with_context(|| format!("missing {key}"))?;
    fs::read(path).with_context(|| format!("read {key} from {path}"))
}

fn validate_config(
    config: &RunConfig,
    protocol_bytes: &[u8],
    split_bytes: &[u8],
    receipt_bytes: &[u8],
) -> Result<()> {
    if config.schema_version != RUN_CONFIG_SCHEMA {
        bail!("unsupported run config schema {}", config.schema_version);
    }
    if config.protocol_id != EXPECTED_PROTOCOL_ID {
        bail!("unexpected protocol id {}", config.protocol_id);
    }
    if let Some(power_digest) = config.power_report_sha256.as_deref() {
        validate_sha256(power_digest).context("invalid power_report_sha256")?;
    }
    for (field, value) in [
        ("protocol_sha256", config.protocol_sha256.as_str()),
        (
            "split_manifest_sha256",
            config.split_manifest_sha256.as_str(),
        ),
        (
            "partition_digest_sha256",
            config.partition_digest_sha256.as_str(),
        ),
        (
            "baseline_selection_receipt_sha256",
            config.baseline_selection_receipt_sha256.as_str(),
        ),
    ] {
        validate_sha256(value).with_context(|| format!("invalid {field}"))?;
    }
    if config.protocol_sha256 != sha256_hex(protocol_bytes) {
        bail!("evaluation protocol digest mismatch; do not analyze against a changed protocol");
    }
    if config.split_manifest_sha256 != sha256_hex(split_bytes) {
        bail!("split manifest digest mismatch; do not analyze against a changed split file");
    }
    if config.baseline_selection_receipt_sha256 != sha256_hex(receipt_bytes) {
        bail!("baseline selection receipt digest mismatch");
    }
    let split_manifest: SplitManifest =
        serde_json::from_slice(split_bytes).context("parse frozen split manifest JSON")?;
    validate_frozen_partition(&split_manifest, &config.partition_digest_sha256)?;

    let exposed_splits: BTreeMap<String, String> = split_manifest
        .families
        .iter()
        .map(|family| (family.family_id.clone(), family.split.clone()))
        .collect();
    validate_exposure_quarantine(
        &config.known_exposed_validation_family_ids,
        &config.known_exposed_test_family_ids,
        &exposed_splits,
    )?;
    if config.bootstrap_replicates != EXPECTED_BOOTSTRAP_REPLICATES {
        bail!(
            "protocol requires exactly {EXPECTED_BOOTSTRAP_REPLICATES} bootstrap replicates, got {}",
            config.bootstrap_replicates
        );
    }
    if config.model_version.trim().is_empty() || config.checker_version.trim().is_empty() {
        bail!("model and checker versions must be explicit");
    }
    Ok(())
}

fn validate_exposure_quarantine(
    validation_ids: &[String],
    test_ids: &[String],
    frozen_families: &BTreeMap<String, String>,
) -> Result<()> {
    if validation_ids.is_empty() || !validation_ids.iter().any(|id| id == REQUIRED_EXPOSED_VALIDATION_FAMILY) {
        bail!(
            "known-exposed validation family {REQUIRED_EXPOSED_VALIDATION_FAMILY} must remain excluded from validation baseline selection"
        );
    }
    if test_ids.is_empty() || !test_ids.iter().any(|id| id == REQUIRED_EXPOSED_TEST_FAMILY) {
        bail!(
            "known-exposed test family {REQUIRED_EXPOSED_TEST_FAMILY} must remain excluded from confirmatory scoring"
        );
    }

    let validation_unique: BTreeSet<&str> = validation_ids.iter().map(String::as_str).collect();
    if validation_unique.len() != validation_ids.len() {
        bail!("known_exposed_validation_family_ids contains duplicates");
    }
    let test_unique: BTreeSet<&str> = test_ids.iter().map(String::as_str).collect();
    if test_unique.len() != test_ids.len() {
        bail!("known_exposed_test_family_ids contains duplicates");
    }
    if validation_unique.intersection(&test_unique).next().is_some() {
        bail!("a family cannot be listed as exposed in both validation and test");
    }

    for exposed in validation_ids {
        if frozen_families.get(exposed).map(String::as_str) != Some("validation") {
            bail!(
                "known-exposed validation family {exposed} is absent from or not assigned to the frozen validation split"
            );
        }
    }
    for exposed in test_ids {
        if frozen_families.get(exposed).map(String::as_str) != Some("test") {
            bail!(
                "known-exposed test family {exposed} is absent from or not assigned to the frozen test split"
            );
        }
    }
    Ok(())
}

fn validate_frozen_partition(manifest: &SplitManifest, expected_config_digest: &str) -> Result<()> {
    if manifest.upstream.commit != EXPECTED_UPSTREAM_COMMIT {
        bail!(
            "unexpected upstream commit {}; expected {}",
            manifest.upstream.commit,
            EXPECTED_UPSTREAM_COMMIT
        );
    }
    if manifest.families.len() != 372 {
        bail!(
            "frozen split must contain exactly 372 families, got {}",
            manifest.families.len()
        );
    }
    let plan = &manifest.split_plan;
    if plan.seed != EXPECTED_SPLIT_SEED || plan.algorithm != EXPECTED_SPLIT_ALGORITHM {
        bail!("frozen split seed or assignment algorithm differs from the registered version");
    }
    if (
        plan.train_percent,
        plan.validation_percent,
        plan.test_percent,
    ) != (70, 15, 15)
    {
        bail!("frozen split percentages must remain exactly 70/15/15");
    }
    if (plan.counts.train, plan.counts.validation, plan.counts.test) != (260, 56, 56) {
        bail!("frozen split plan count metadata differs from the expected 260/56/56 partition");
    }

    let expected = expected_split_assignments(&manifest.families, &plan.seed)?;
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for family in &manifest.families {
        let expected_split = expected
            .get(&family.family_id)
            .with_context(|| format!("no computed assignment for family {}", family.family_id))?;
        if &family.split != expected_split {
            bail!(
                "family {} has split {}, but deterministic assignment requires {}",
                family.family_id,
                family.split,
                expected_split
            );
        }
        *counts.entry(family.split.as_str()).or_default() += 1;
    }
    for (split, expected_count) in [("train", 260), ("validation", 56), ("test", 56)] {
        if counts.get(split).copied().unwrap_or_default() != expected_count {
            bail!(
                "frozen split count mismatch for {split}: expected {expected_count}, got {}",
                counts.get(split).copied().unwrap_or_default()
            );
        }
    }

    let recomputed_digest = partition_digest_sha256(&manifest.families);
    if recomputed_digest != plan.partition_digest_sha256 {
        bail!("partition digest in manifest is incorrect: recomputed {recomputed_digest}");
    }
    if recomputed_digest != expected_config_digest {
        bail!("partition digest does not match the frozen run config");
    }
    Ok(())
}

fn expected_split_assignments(
    families: &[SplitFamily],
    seed: &str,
) -> Result<BTreeMap<String, String>> {
    let mut unique_ids = BTreeSet::new();
    let mut ranked: Vec<(Vec<u8>, String)> = Vec::with_capacity(families.len());
    for family in families {
        if family.family_id.trim().is_empty() {
            bail!("frozen family IDs must not be empty");
        }
        if !unique_ids.insert(family.family_id.as_str()) {
            bail!("duplicate family ID in frozen split: {}", family.family_id);
        }
        let mut hasher = Sha256::new();
        hasher.update(seed.as_bytes());
        hasher.update([0]);
        hasher.update(family.family_id.as_bytes());
        ranked.push((hasher.finalize().to_vec(), family.family_id.clone()));
    }
    ranked.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

    let train_count = ranked.len() * 70 / 100;
    let validation_end = ranked.len() * 85 / 100;
    Ok(ranked
        .into_iter()
        .enumerate()
        .map(|(index, (_, family_id))| {
            let split = if index < train_count {
                "train"
            } else if index < validation_end {
                "validation"
            } else {
                "test"
            };
            (family_id, split.to_string())
        })
        .collect())
}

fn partition_digest_sha256(families: &[SplitFamily]) -> String {
    let mut rows: Vec<(&str, &str)> = families
        .iter()
        .map(|family| (family.family_id.as_str(), family.split.as_str()))
        .collect();
    rows.sort_unstable_by(|left, right| left.0.cmp(right.0));
    let mut material = String::new();
    for (family_id, split) in rows {
        material.push_str(family_id);
        material.push('\t');
        material.push_str(split);
        material.push('\n');
    }
    sha256_hex(material.as_bytes())
}

fn parse_target_jsonl(bytes: &[u8], split: &str) -> Result<Vec<TargetOutcome>> {
    let text = String::from_utf8(bytes.to_vec())
        .with_context(|| format!("{split} target results must be UTF-8"))?;
    let mut outcomes = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: TargetOutcome = serde_json::from_str(line)
            .with_context(|| format!("parse {split} result JSONL line {}", line_index + 1))?;
        outcomes.push(row);
    }
    if outcomes.is_empty() {
        bail!("no {split} target outcome rows supplied");
    }
    Ok(outcomes)
}

fn compute_baseline_scores(targets: &[&TargetOutcome]) -> Result<BTreeMap<String, f64>> {
    if targets.is_empty() {
        bail!("cannot select a baseline from zero eligible validation targets");
    }
    let mut scores = BTreeMap::new();
    for arm_id in BASELINE_SELECTION_ARMS {
        let mut successes = 0_usize;
        for target in targets {
            let arm = target
                .arms
                .iter()
                .find(|arm| arm.arm == arm_id)
                .with_context(|| {
                    format!(
                        "validation target {} lacks baseline arm {arm_id}",
                        target.target_id
                    )
                })?;
            successes += if arm.semantic_gate_passed { 1 } else { 0 };
        }
        scores.insert(arm_id.to_string(), successes as f64 / targets.len() as f64);
    }
    Ok(scores)
}

fn validate_baseline_scores(
    receipt: &BaselineSelectionReceipt,
    computed_scores: &BTreeMap<String, f64>,
) -> Result<()> {
    for arm_id in BASELINE_SELECTION_ARMS {
        let claimed = receipt
            .arm_scores
            .get(arm_id)
            .with_context(|| format!("baseline receipt lacks score for arm {arm_id}"))?;
        let computed = computed_scores
            .get(arm_id)
            .with_context(|| format!("computed validation score missing for arm {arm_id}"))?;
        if (claimed - computed).abs() > 1e-12 {
            bail!(
                "validation score for arm {arm_id} is {claimed}, but raw validation outcomes yield {computed}"
            );
        }
    }
    let selected = computed_scores
        .iter()
        .max_by(|(arm_a, score_a), (arm_b, score_b)| {
            score_a.total_cmp(score_b).then_with(|| arm_b.cmp(arm_a))
        })
        .map(|(arm, _)| arm.as_str())
        .context("computed validation scores are empty")?;
    if selected != receipt.selected_arm {
        bail!(
            "baseline receipt selected {}, but raw validation outcomes select {selected}",
            receipt.selected_arm
        );
    }
    Ok(())
}

fn validate_power_report(
    report: &PowerReport,
    config: &RunConfig,
    validation_bytes: &[u8],
    split_manifest: &SplitManifest,
    receipt: &BaselineSelectionReceipt,
    eligible_validation: &[&TargetOutcome],
    validation_scores: &BTreeMap<String, f64>,
) -> Result<()> {
    if report.schema_version != "pitb.power-report.v2" {
        bail!("unsupported power report schema {}", report.schema_version);
    }
    if report.protocol_id != EXPECTED_PROTOCOL_ID
        || report.protocol_sha256 != config.protocol_sha256
    {
        bail!("power report protocol ID/digest differs from the frozen run");
    }
    if report.run_design_sha256 != run_design_sha256(config)? {
        bail!("power report was produced for a different canonical run design");
    }
    if report.validation_results_sha256 != sha256_hex(validation_bytes) {
        bail!("power report validation JSONL digest does not match supplied validation data");
    }
    if report.baseline_selection_receipt_sha256 != config.baseline_selection_receipt_sha256 {
        bail!("power report baseline receipt does not match the frozen validation baseline");
    }
    if report.split_manifest_sha256 != config.split_manifest_sha256
        || report.partition_digest_sha256 != config.partition_digest_sha256
        || report.upstream_commit != split_manifest.upstream.commit
    {
        bail!("power report source/split provenance differs from the frozen run");
    }
    if report.analyzer_source_sha256 != sha256_hex(include_bytes!("analyze_pitb_results.rs")) {
        bail!("power report was generated by a different analyzer source revision");
    }
    if report.selected_baseline_arm != receipt.selected_arm {
        bail!("power report baseline arm does not match the validation-selected baseline");
    }
    if report.known_exposed_validation_family_ids != config.known_exposed_validation_family_ids
        || report.known_exposed_test_family_ids != config.known_exposed_test_family_ids
    {
        bail!("power report exposure quarantine differs from the frozen run config");
    }
    let selected_rate = validation_scores
        .get(&receipt.selected_arm)
        .context("selected baseline has no independently recomputed validation score")?;
    if (report.validation_baseline_semantic_success_rate - selected_rate).abs() > 1e-12 {
        bail!("power report baseline rate differs from raw validation outcomes");
    }
    if report.eligible_validation_target_count != eligible_validation.len()
        || report.eligible_validation_family_count
            != eligible_validation
                .iter()
                .map(|target| target.family_id.as_str())
                .collect::<BTreeSet<_>>()
                .len()
    {
        bail!("power report validation denominator does not match the validated targets");
    }

    let mut expected_family_sizes: BTreeMap<String, usize> = BTreeMap::new();
    for target in eligible_validation {
        *expected_family_sizes
            .entry(target.family_id.clone())
            .or_default() += 1;
    }
    let expected_family_rows: Vec<FamilyTargetCount> = expected_family_sizes
        .into_iter()
        .map(|(family_id, eligible_target_count)| FamilyTargetCount {
            family_id,
            eligible_target_count,
        })
        .collect();
    if report.family_target_counts.len() != expected_family_rows.len()
        || report
            .family_target_counts
            .iter()
            .zip(&expected_family_rows)
            .any(|(observed, expected)| {
                observed.family_id != expected.family_id
                    || observed.eligible_target_count != expected.eligible_target_count
            })
    {
        bail!("power report family-size distribution differs from raw validation rows");
    }

    if report.study_simulations < 1_000 || report.bootstrap_replicates_per_simulated_study < 500 {
        bail!("power report has fewer simulations/bootstrap resamples than the registered floor");
    }
    if report.simulation_seed != config.bootstrap_seed
        || report.confirmatory_minimum_family_count != MIN_CONFIRMATORY_FAMILY_CLUSTERS
        || (report.utility_effect_threshold_percentage_points - 10.0).abs() > 1e-12
        || (report.required_power_at_20pp_and_icc_020 - 0.80).abs() > 1e-12
        || (report.maximum_false_claim_probability_at_zero_effect - 0.05).abs() > 1e-12
    {
        bail!("power report does not encode the preregistered seed and gate thresholds");
    }

    let expected_effects = [0.0_f64, 10.0, 15.0, 20.0];
    let expected_iccs = [0.0_f64, 0.05, 0.10, 0.20];
    let expected_pair_associations = [0.0_f64, 0.5, 0.9];
    if report.true_effect_grid_percentage_points.len() != expected_effects.len()
        || report
            .true_effect_grid_percentage_points
            .iter()
            .zip(expected_effects)
            .any(|(actual, expected)| (*actual - expected).abs() > 1e-12)
        || report.assumed_family_icc_grid.len() != expected_iccs.len()
        || report
            .assumed_family_icc_grid
            .iter()
            .zip(expected_iccs)
            .any(|(actual, expected)| (*actual - expected).abs() > 1e-12)
        || report.pair_association_fraction_grid.len() != expected_pair_associations.len()
        || report
            .pair_association_fraction_grid
            .iter()
            .zip(expected_pair_associations)
            .any(|(actual, expected)| (*actual - expected).abs() > 1e-12)
        || report.scenarios.len()
            != expected_effects.len() * expected_iccs.len() * expected_pair_associations.len()
    {
        bail!("power report effect/ICC/pair-association grid differs from the preregistered 4x4x3 grid");
    }

    let mut observed_scenarios = BTreeSet::new();
    for scenario in &report.scenarios {
        if scenario.study_simulations != report.study_simulations
            || scenario.bootstrap_replicates_per_simulated_study
                != report.bootstrap_replicates_per_simulated_study
            || !expected_effects.iter().any(|effect| {
                (scenario.true_treatment_effect_percentage_points - effect).abs() <= 1e-12
            })
            || !expected_iccs
                .iter()
                .any(|icc| (scenario.assumed_family_icc - icc).abs() <= 1e-12)
            || !expected_pair_associations
                .iter()
                .any(|association| (scenario.pair_association_fraction - association).abs() <= 1e-12)
            || !observed_scenarios.insert((
                (scenario.true_treatment_effect_percentage_points * 1000.0).round() as i32,
                (scenario.assumed_family_icc * 1000.0).round() as i32,
                (scenario.pair_association_fraction * 1000.0).round() as i32,
            ))
        {
            bail!("power report contains duplicate or unregistered scenarios");
        }
        match scenario.status.as_str() {
            "estimated" => {
                let power = scenario
                    .estimated_utility_gate_power
                    .context("estimated power scenario has no power estimate")?;
                let lower = scenario
                    .estimated_power_lower_95
                    .context("estimated power scenario has no lower Wilson bound")?;
                let upper = scenario
                    .estimated_power_upper_95
                    .context("estimated power scenario has no upper Wilson bound")?;
                let standard_error = scenario
                    .monte_carlo_standard_error
                    .context("estimated power scenario has no Monte Carlo standard error")?;
                let expected_simultaneous_null_upper = if
                    scenario.true_treatment_effect_percentage_points.abs() <= 1e-12
                {
                    Some(simultaneous_null_upper_bound(
                        scenario.utility_gate_passes,
                        report.study_simulations,
                    ))
                } else {
                    None
                };
                if [power, lower, upper, standard_error]
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
                    || lower > power + 1e-12
                    || power > upper + 1e-12
                    || scenario.utility_gate_passes > report.study_simulations
                    || scenario.degenerate_interval_trials > report.study_simulations
                    || scenario.utility_gate_passes
                        > report.study_simulations - scenario.degenerate_interval_trials
                    || !same_optional_probability(
                        scenario.null_effect_false_claim_probability_upper_simultaneous_95,
                        expected_simultaneous_null_upper,
                    )
                {
                    bail!("power scenario has inconsistent probability or confidence bounds");
                }
            }
            "infeasible_effect_above_ceiling" => {
                if scenario.estimated_utility_gate_power.is_some()
                    || scenario.estimated_power_lower_95.is_some()
                    || scenario.estimated_power_upper_95.is_some()
                    || scenario.null_effect_false_claim_probability_upper_simultaneous_95.is_some()
                {
                    bail!("infeasible power scenario must not report power or null-calibration bounds");
                }
            }
            other => bail!("unknown power scenario status {other}"),
        }
    }

    // Re-run the frozen validation-only simulations before accepting the receipt.
    // A content hash pins a report; it does not prove the report's numbers were
    // actually computed from the pinned family sizes, seed, and decision rule.
    for (effect_index, effect_pp) in expected_effects.iter().copied().enumerate() {
        for (icc_index, icc) in expected_iccs.iter().copied().enumerate() {
            for (association_index, association) in expected_pair_associations.iter().copied().enumerate() {
                let observed = report
                    .scenarios
                    .iter()
                    .find(|scenario| {
                        (scenario.true_treatment_effect_percentage_points - effect_pp).abs() <= 1e-12
                            && (scenario.assumed_family_icc - icc).abs() <= 1e-12
                            && (scenario.pair_association_fraction - association).abs() <= 1e-12
                    })
                    .context("power report lacks a required effect/ICC/pair-association scenario")?;
                let scenario_index = effect_index * expected_iccs.len() * expected_pair_associations.len()
                    + icc_index * expected_pair_associations.len()
                    + association_index;
                let scenario_seed = config.bootstrap_seed.wrapping_add(
                    scenario_index as u64 * 0x9E37_79B9_7F4A_7C15,
                );
                let expected = simulate_power_scenario(
                    &expected_family_rows,
                    *selected_rate,
                    effect_pp / 100.0,
                    icc,
                    association,
                    report.study_simulations,
                    report.bootstrap_replicates_per_simulated_study,
                    scenario_seed,
                )?;
                if observed.status != expected.status
                    || observed.utility_gate_passes != expected.utility_gate_passes
                    || observed.degenerate_interval_trials != expected.degenerate_interval_trials
                    || !same_optional_probability(
                        observed.estimated_utility_gate_power,
                        expected.estimated_utility_gate_power,
                    )
                    || !same_optional_probability(
                        observed.monte_carlo_standard_error,
                        expected.monte_carlo_standard_error,
                    )
                    || !same_optional_probability(
                        observed.estimated_power_lower_95,
                        expected.estimated_power_lower_95,
                    )
                    || !same_optional_probability(
                        observed.estimated_power_upper_95,
                        expected.estimated_power_upper_95,
                    )
                    || !same_optional_probability(
                        observed.null_effect_false_claim_probability_upper_simultaneous_95,
                        expected.null_effect_false_claim_probability_upper_simultaneous_95,
                    )
                    || !same_optional_probability(
                        observed.mean_observed_effect_percentage_points,
                        expected.mean_observed_effect_percentage_points,
                    )
                    || !same_optional_probability(
                        observed.mean_lower_95ci_percentage_points,
                        expected.mean_lower_95ci_percentage_points,
                    )
                {
                    bail!(
                        "power report scenario +{effect_pp}pp/ICC {icc}/pair-association {association} cannot be reproduced from the pinned validation rows and seed"
                    );
                }
            }
        }
    }

    let expected_weak_null_sds_pp = [5.0_f64, 10.0, 20.0];
    let expected_pair_associations = [0.0_f64, 0.5, 0.9];
    if report.weak_null_family_effect_heterogeneity_sd_grid_percentage_points.len()
        != expected_weak_null_sds_pp.len()
        || report
            .weak_null_family_effect_heterogeneity_sd_grid_percentage_points
            .iter()
            .zip(expected_weak_null_sds_pp)
            .any(|(actual, expected)| (*actual - expected).abs() > 1e-12)
        || report.weak_null_pair_association_fraction_grid.len()
            != expected_pair_associations.len()
        || report
            .weak_null_pair_association_fraction_grid
            .iter()
            .zip(expected_pair_associations)
            .any(|(actual, expected)| (*actual - expected).abs() > 1e-12)
        || report.weak_null_scenarios.len()
            != expected_weak_null_sds_pp.len() * expected_iccs.len() * expected_pair_associations.len()
    {
        bail!("weak-null report does not contain the preregistered 3x4x3 heterogeneity/ICC/association grid");
    }

    let mut observed_weak_null_scenarios = BTreeSet::new();
    for scenario in &report.weak_null_scenarios {
        let probability = scenario
            .estimated_false_claim_probability
            .context("weak-null scenario lacks a false-claim estimate")?;
        let lower = scenario
            .false_claim_probability_lower_95
            .context("weak-null scenario lacks a lower Wilson bound")?;
        let upper = scenario
            .false_claim_probability_upper_95
            .context("weak-null scenario lacks an upper Wilson bound")?;
        let simultaneous_upper = scenario
            .false_claim_probability_upper_simultaneous_95
            .context("weak-null scenario lacks a simultaneous upper Wilson bound")?;
        let standard_error = scenario
            .monte_carlo_standard_error
            .context("weak-null scenario lacks its Monte Carlo standard error")?;
        let mean_effect = scenario
            .mean_observed_effect_percentage_points
            .context("weak-null scenario lacks its mean observed effect")?;
        let realized_sd = scenario
            .mean_realized_family_effect_heterogeneity_sd_percentage_points
            .context("weak-null scenario lacks its realized heterogeneity diagnostic")?;
        if scenario.status != "estimated"
            || scenario.true_target_weighted_mean_effect_percentage_points.abs() > 1e-12
            || scenario.study_simulations != report.study_simulations
            || scenario.bootstrap_replicates_per_simulated_study
                != report.bootstrap_replicates_per_simulated_study
            || !expected_weak_null_sds_pp.iter().any(|sd| {
                (scenario.requested_family_effect_heterogeneity_sd_percentage_points - sd).abs()
                    <= 1e-12
            })
            || !expected_iccs
                .iter()
                .any(|icc| (scenario.assumed_family_icc - icc).abs() <= 1e-12)
            || !expected_pair_associations.iter().any(|association| {
                (scenario.pair_association_fraction - association).abs() <= 1e-12
            })
            || !observed_weak_null_scenarios.insert((
                (scenario.requested_family_effect_heterogeneity_sd_percentage_points * 1000.0)
                    .round() as i32,
                (scenario.assumed_family_icc * 1000.0).round() as i32,
                (scenario.pair_association_fraction * 1000.0).round() as i32,
            ))
            || scenario.false_claims > report.study_simulations
            || scenario.degenerate_interval_trials > report.study_simulations
            || scenario.false_claims
                > report.study_simulations - scenario.degenerate_interval_trials
            || [probability, lower, upper, simultaneous_upper, standard_error]
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
            || lower > probability + 1e-12
            || probability > upper + 1e-12
            || upper > simultaneous_upper + 1e-12
            || (simultaneous_upper
                - simultaneous_null_upper_bound(scenario.false_claims, report.study_simulations))
                .abs()
                > 1e-12
            || !mean_effect.is_finite()
            || !realized_sd.is_finite()
            || realized_sd < 0.0
            || realized_sd
                > scenario.requested_family_effect_heterogeneity_sd_percentage_points + 1e-8
        {
            bail!("weak-null scenario has an invalid grid key, count, or probability bound");
        }
    }

    // Reproduce all family-heterogeneous, zero-average-effect stress scenarios
    // from the same validation-derived family sizes, configuration and seed.
    for (sd_index, requested_sd_pp) in expected_weak_null_sds_pp.iter().copied().enumerate() {
        for (icc_index, icc) in expected_iccs.iter().copied().enumerate() {
            for (association_index, pair_association_fraction) in
                expected_pair_associations.iter().copied().enumerate()
            {
                let scenario_index = sd_index * expected_iccs.len() * expected_pair_associations.len()
                    + icc_index * expected_pair_associations.len()
                    + association_index;
                let scenario_seed = config
                    .bootstrap_seed
                    .wrapping_add(0xD1B5_4A32_D192_ED03)
                    .wrapping_add(scenario_index as u64 * 0x9E37_79B9_7F4A_7C15);
                let observed = report
                    .weak_null_scenarios
                    .iter()
                    .find(|scenario| {
                        (scenario.requested_family_effect_heterogeneity_sd_percentage_points
                            - requested_sd_pp)
                            .abs()
                            <= 1e-12
                            && (scenario.assumed_family_icc - icc).abs() <= 1e-12
                            && (scenario.pair_association_fraction - pair_association_fraction).abs()
                                <= 1e-12
                    })
                    .context("power report lacks a required weak-null scenario")?;
                let expected = simulate_weak_null_scenario(
                    &expected_family_rows,
                    *selected_rate,
                    requested_sd_pp / 100.0,
                    icc,
                    pair_association_fraction,
                    report.study_simulations,
                    report.bootstrap_replicates_per_simulated_study,
                    scenario_seed,
                )?;
                if observed.false_claims != expected.false_claims
                    || observed.degenerate_interval_trials != expected.degenerate_interval_trials
                    || !same_optional_probability(
                        observed.estimated_false_claim_probability,
                        expected.estimated_false_claim_probability,
                    )
                    || !same_optional_probability(
                        observed.monte_carlo_standard_error,
                        expected.monte_carlo_standard_error,
                    )
                    || !same_optional_probability(
                        observed.false_claim_probability_lower_95,
                        expected.false_claim_probability_lower_95,
                    )
                    || !same_optional_probability(
                        observed.false_claim_probability_upper_95,
                        expected.false_claim_probability_upper_95,
                    )
                    || !same_optional_probability(
                        observed.false_claim_probability_upper_simultaneous_95,
                        expected.false_claim_probability_upper_simultaneous_95,
                    )
                    || !same_optional_probability(
                        observed.mean_observed_effect_percentage_points,
                        expected.mean_observed_effect_percentage_points,
                    )
                    || !same_optional_probability(
                        observed.mean_realized_family_effect_heterogeneity_sd_percentage_points,
                        expected.mean_realized_family_effect_heterogeneity_sd_percentage_points,
                    )
                {
                    bail!(
                        "weak-null scenario +{requested_sd_pp}pp SD/ICC {icc}/pair-association {pair_association_fraction} cannot be reproduced from the frozen validation rows and seed"
                    );
                }
            }
        }
    }

    let max_weak_null_false_claim_probability = report
        .weak_null_scenarios
        .iter()
        .filter_map(|scenario| scenario.estimated_false_claim_probability)
        .reduce(f64::max);
    let max_weak_null_false_claim_upper_simultaneous_95 = report
        .weak_null_scenarios
        .iter()
        .filter_map(|scenario| scenario.false_claim_probability_upper_simultaneous_95)
        .reduce(f64::max);
    if !same_optional_probability(
        report.weak_null_false_claim_probability_max,
        max_weak_null_false_claim_probability,
    ) || !same_optional_probability(
        report.weak_null_false_claim_probability_upper_simultaneous_95_max,
        max_weak_null_false_claim_upper_simultaneous_95,
    ) {
        bail!("weak-null false-claim maxima do not match scenario rows");
    }

    let max_null_power = report
        .scenarios
        .iter()
        .filter(|scenario| scenario.true_treatment_effect_percentage_points.abs() <= 1e-12)
        .filter_map(|scenario| scenario.estimated_utility_gate_power)
        .reduce(f64::max);
    let max_null_upper_simultaneous_95 = report
        .scenarios
        .iter()
        .filter(|scenario| scenario.true_treatment_effect_percentage_points.abs() <= 1e-12)
        .filter_map(|scenario| scenario.null_effect_false_claim_probability_upper_simultaneous_95)
        .reduce(f64::max);
    if !same_optional_probability(
        report.null_effect_false_claim_probability_max,
        max_null_power,
    ) || !same_optional_probability(
        report.null_effect_false_claim_probability_upper_simultaneous_95_max,
        max_null_upper_simultaneous_95,
    ) {
        bail!("power report null-effect false-claim maxima do not match its scenario rows");
    }

    let worst_case_scenarios: Vec<&PowerScenario> = report
        .scenarios
        .iter()
        .filter(|scenario| {
            (scenario.true_treatment_effect_percentage_points - 20.0).abs() <= 1e-12
                && (scenario.assumed_family_icc - 0.20).abs() <= 1e-12
        })
        .collect();
    let all_worst_case_associations_estimated =
        worst_case_scenarios.len() == expected_pair_associations.len()
            && worst_case_scenarios.iter().all(|scenario| {
                scenario.estimated_utility_gate_power.is_some()
                    && scenario.estimated_power_lower_95.is_some()
            });
    let recomputed_worst_power = if all_worst_case_associations_estimated {
        worst_case_scenarios
            .iter()
            .filter_map(|scenario| scenario.estimated_utility_gate_power)
            .reduce(f64::min)
    } else {
        None
    };
    let recomputed_worst_lower = if all_worst_case_associations_estimated {
        worst_case_scenarios
            .iter()
            .filter_map(|scenario| scenario.estimated_power_lower_95)
            .reduce(f64::min)
    } else {
        None
    };
    let recomputed_worst_simultaneous_lower = if all_worst_case_associations_estimated {
        worst_case_scenarios
            .iter()
            .map(|scenario| {
                simultaneous_power_lower_bound(
                    scenario.utility_gate_passes,
                    scenario.study_simulations,
                )
            })
            .reduce(f64::min)
    } else {
        None
    };
    if !same_optional_probability(
        report.estimated_power_at_20pp_and_icc_020,
        recomputed_worst_power,
    ) || !same_optional_probability(
        report.estimated_power_at_20pp_and_icc_020_lower_95,
        recomputed_worst_lower,
    ) || !same_optional_probability(
        report.estimated_power_at_20pp_and_icc_020_lower_simultaneous_95,
        recomputed_worst_simultaneous_lower,
    ) {
        bail!("power report headline power/bounds must equal the worst registered association case at +20pp/ICC 0.20");
    }

    let recomputed_gate = report.eligible_validation_family_count
        >= MIN_CONFIRMATORY_FAMILY_CLUSTERS
        && report
            .estimated_power_at_20pp_and_icc_020_lower_simultaneous_95
            .is_some_and(|value| value >= 0.80)
        && report
            .null_effect_false_claim_probability_upper_simultaneous_95_max
            .is_some_and(|value| value <= 0.05)
        && report
            .weak_null_false_claim_probability_upper_simultaneous_95_max
            .is_some_and(|value| value <= 0.05);
    if report.power_gate_passed != recomputed_gate || !report.power_gate_passed {
        bail!("preregistered power-planning no-go: confirmatory held-out scoring is blocked");
    }
    Ok(())
}

fn same_optional_probability(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => (left - right).abs() <= 1e-12,
        (None, None) => true,
        _ => false,
    }
}

fn validate_baseline_receipt(
    receipt: &BaselineSelectionReceipt,
    validation_bytes: &[u8],
) -> Result<()> {
    if receipt.schema_version != "pitb.baseline-selection.v0" {
        bail!("unsupported baseline-selection receipt schema");
    }
    if receipt.selected_split != "validation" || receipt.metric != "semantic_gate_success_rate" {
        bail!("baseline must be selected on validation using semantic-gate success rate");
    }
    if !receipt.frozen_before_test || receipt.test_outcomes_seen_before_selection {
        bail!("baseline selection was not demonstrably frozen before test evaluation");
    }
    if receipt.tie_break_rule != "highest score; alphabetical arm ID on exact tie" {
        bail!("unexpected or unspecified validation baseline tie-break rule");
    }
    validate_sha256(&receipt.validation_results_sha256)
        .context("invalid validation_results_sha256 in baseline receipt")?;
    if receipt.validation_results_sha256 != sha256_hex(validation_bytes) {
        bail!("validation outcome digest does not match the baseline-selection receipt");
    }
    let score_keys: BTreeSet<&str> = receipt.arm_scores.keys().map(String::as_str).collect();
    let expected_keys: BTreeSet<&str> = BASELINE_SELECTION_ARMS.into_iter().collect();
    if score_keys != expected_keys {
        bail!("baseline receipt must include exactly validation scores for arms A through E and I");
    }
    for (arm, score) in &receipt.arm_scores {
        if !score.is_finite() || !(0.0..=1.0).contains(score) {
            bail!("validation score for arm {arm} must be finite and within [0,1]");
        }
    }
    let selected = receipt
        .arm_scores
        .iter()
        .max_by(|(arm_a, score_a), (arm_b, score_b)| {
            score_a
                .total_cmp(score_b)
                // max_by picks the greater element, so reverse the name order
                // to give the alphabetically first arm the tie.
                .then_with(|| arm_b.cmp(arm_a))
        })
        .map(|(arm, _)| arm.as_str())
        .context("baseline receipt contains no scores")?;
    if selected != receipt.selected_arm {
        bail!(
            "selected baseline {} does not match the validation scores (expected {selected})",
            receipt.selected_arm
        );
    }
    if !BASELINE_SELECTION_ARMS.contains(&receipt.selected_arm.as_str()) {
        bail!("selected baseline must be one of A through E or I");
    }
    Ok(())
}

fn validate_outcomes<'a>(
    outcomes: &'a [TargetOutcome],
    config: &RunConfig,
    frozen_families: &BTreeMap<String, String>,
    required_split: &str,
) -> Result<Vec<&'a TargetOutcome>> {
    let mut target_ids = BTreeSet::new();
    let mut eligible_statement_digests = BTreeSet::new();
    let mut arm_config_digests: BTreeMap<String, String> = BTreeMap::new();
    let mut represented_families = BTreeSet::new();
    let mut eligible = Vec::new();

    for target in outcomes {
        if target.schema_version != RUN_RESULTS_SCHEMA {
            bail!(
                "target {} has unsupported result schema {}",
                target.target_id,
                target.schema_version
            );
        }
        if target.target_id.trim().is_empty() || !target_ids.insert(target.target_id.as_str()) {
            bail!("empty or duplicate target_id {}", target.target_id);
        }
        if target.split != required_split {
            bail!(
                "target {} is assigned to {}, but this analysis stage accepts {required_split} rows only",
                target.target_id,
                target.split
            );
        }
        let frozen_split = frozen_families
            .get(&target.family_id)
            .with_context(|| format!("unknown family_id {}", target.family_id))?;
        if frozen_split != required_split {
            bail!(
                "target {} family {} is assigned to {frozen_split}, not {required_split}",
                target.target_id,
                target.family_id
            );
        }
        represented_families.insert(target.family_id.as_str());

        match target.eligibility {
            Eligibility::Eligible => {
                if config
                    .known_exposed_validation_family_ids
                    .contains(&target.family_id)
                    || config
                        .known_exposed_test_family_ids
                        .contains(&target.family_id)
                {
                    bail!(
                        "known-exposed family {} was included as eligible",
                        target.family_id
                    );
                }
                let statement_sha = target
                    .statement_sha256
                    .as_deref()
                    .context("eligible target lacks exact statement SHA-256")?;
                validate_sha256(statement_sha)
                    .with_context(|| format!("target {} statement digest", target.target_id))?;
                if !eligible_statement_digests.insert(statement_sha) {
                    bail!(
                        "duplicate eligible statement digest {}; investigate cross-family leakage",
                        statement_sha
                    );
                }
                if target
                    .exclusion_reason
                    .as_deref()
                    .is_some_and(|s| !s.trim().is_empty())
                {
                    bail!(
                        "eligible target {} has an exclusion reason",
                        target.target_id
                    );
                }
                if target.family_id.trim().is_empty() {
                    bail!(
                        "eligible target {} has an empty family id",
                        target.family_id
                    );
                }
                let source_commit = target
                    .source_commit
                    .as_deref()
                    .context("eligible target lacks pinned source commit")?;
                if source_commit != EXPECTED_UPSTREAM_COMMIT {
                    bail!(
                        "target {} source commit {} differs from frozen upstream {}",
                        target.target_id,
                        source_commit,
                        EXPECTED_UPSTREAM_COMMIT
                    );
                }
                let theorem_id = target.theorem_id.as_deref().unwrap_or_default().trim();
                let artifact_path = target
                    .artifact_path
                    .as_deref()
                    .context("eligible target lacks repository-relative artifact path")?;
                if theorem_id.is_empty()
                    || target
                        .lean_toolchain
                        .as_deref()
                        .unwrap_or_default()
                        .trim()
                        .is_empty()
                    || target
                        .mathlib_revision
                        .as_deref()
                        .unwrap_or_default()
                        .trim()
                        .is_empty()
                {
                    bail!(
                        "eligible target {} must identify theorem, Lean toolchain and Mathlib revision",
                        target.target_id
                    );
                }
                validate_repo_relative_path(artifact_path)
                    .with_context(|| format!("target {} artifact path", target.target_id))?;
                for (label, digest) in [
                    ("artifact", target.artifact_sha256.as_deref()),
                    (
                        "eligibility receipt",
                        target.eligibility_receipt_sha256.as_deref(),
                    ),
                    (
                        "statement review receipt",
                        target.statement_review_receipt_sha256.as_deref(),
                    ),
                    (
                        "proof audit receipt",
                        target.proof_audit_receipt_sha256.as_deref(),
                    ),
                    (
                        "successor-test receipt",
                        target.successor_test_receipt_sha256.as_deref(),
                    ),
                    (
                        "mutation receipt",
                        target.mutation_receipt_sha256.as_deref(),
                    ),
                ] {
                    validate_sha256(digest.context(label)?)
                        .with_context(|| format!("target {} {label} digest", target.target_id))?;
                }
                validate_arm_set(target, config, &mut arm_config_digests)?;
                eligible.push(target);
            }
            Eligibility::Excluded | Eligibility::Unknown => {
                let reason = target
                    .exclusion_reason
                    .as_deref()
                    .unwrap_or_default()
                    .trim();
                if reason.is_empty() {
                    bail!(
                        "excluded/unknown target {} must retain an explicit reason",
                        target.target_id
                    );
                }
                if !target.arms.is_empty() {
                    bail!(
                        "excluded/unknown target {} must not contain scored arm outcomes",
                        target.target_id
                    );
                }
            }
        }
    }

    let missing_families: Vec<&str> = frozen_families
        .iter()
        .filter(|(family_id, split)| {
            split.as_str() == required_split && !represented_families.contains(family_id.as_str())
        })
        .map(|(family_id, _)| family_id.as_str())
        .collect();
    if !missing_families.is_empty() {
        bail!(
            "frozen {required_split} families lack an explicit outcome/exclusion row: {}",
            missing_families.join(", ")
        );
    }
    Ok(eligible)
}

fn validate_arm_set(
    target: &TargetOutcome,
    config: &RunConfig,
    arm_config_digests: &mut BTreeMap<String, String>,
) -> Result<()> {
    let mut seen = BTreeSet::new();
    for arm in &target.arms {
        if !ARMS.contains(&arm.arm.as_str()) || !seen.insert(arm.arm.as_str()) {
            bail!(
                "target {} contains unknown or duplicate arm {}",
                target.target_id,
                arm.arm
            );
        }
        if arm.model_version != config.model_version
            || arm.checker_version != config.checker_version
        {
            bail!(
                "target {} arm {} model/checker version differs from frozen config",
                target.target_id,
                arm.arm
            );
        }
        if !same_budget(&arm.budget, &config.budget) {
            bail!(
                "target {} arm {} has a budget different from the frozen matched budget",
                target.target_id,
                arm.arm
            );
        }
        for (label, digest) in [
            ("config", arm.config_sha256.as_str()),
            ("input", arm.input_sha256.as_str()),
            ("retrieval log", arm.retrieval_log_sha256.as_str()),
            ("checker receipt", arm.checker_receipt_sha256.as_str()),
            (
                "semantic-gate receipt",
                arm.semantic_gate_receipt_sha256.as_str(),
            ),
        ] {
            validate_sha256(digest).with_context(|| {
                format!("target {} arm {} {label} digest", target.target_id, arm.arm)
            })?;
        }
        match arm_config_digests.insert(arm.arm.clone(), arm.config_sha256.clone()) {
            Some(previous) if previous != arm.config_sha256 => {
                bail!(
                    "arm {} changed configuration across held-out targets",
                    arm.arm
                );
            }
            _ => {}
        }
        if arm.model_calls_used > config.budget.max_model_calls
            || arm.tokens_used > config.budget.max_tokens
            || arm.wall_ms > config.budget.max_wall_ms
            || arm.cost_microusd > config.budget.max_cost_microusd
        {
            bail!(
                "target {} arm {} exceeded a frozen task budget",
                target.target_id,
                arm.arm
            );
        }
        match arm.status {
            OutcomeStatus::VerifiedSuccess if !arm.checker_passed || !arm.semantic_gate_passed => {
                bail!(
                    "target {} arm {} claims verified_success without both checker and semantic gates",
                    target.target_id,
                    arm.arm
                );
            }
            OutcomeStatus::ProofOnly if !arm.checker_passed || arm.semantic_gate_passed => {
                bail!(
                    "target {} arm {} proof_only requires checker pass and semantic-gate failure",
                    target.target_id,
                    arm.arm
                );
            }
            OutcomeStatus::Failed
            | OutcomeStatus::Timeout
            | OutcomeStatus::NotAttempted
            | OutcomeStatus::ToolUnavailable
            | OutcomeStatus::OutOfScope
                if arm.checker_passed || arm.semantic_gate_passed =>
            {
                bail!(
                    "target {} arm {} has a passed gate inconsistent with status {:?}",
                    target.target_id,
                    arm.arm,
                    arm.status
                );
            }
            _ => {}
        }
    }
    let missing: Vec<&str> = ARMS
        .iter()
        .copied()
        .filter(|expected| !seen.contains(expected))
        .collect();
    if !missing.is_empty() {
        bail!(
            "eligible target {} lacks matched arms: {}",
            target.target_id,
            missing.join(", ")
        );
    }
    Ok(())
}

fn same_budget(left: &Budget, right: &Budget) -> bool {
    left.max_model_calls == right.max_model_calls
        && left.max_tokens == right.max_tokens
        && left.max_wall_ms == right.max_wall_ms
        && left.max_cost_microusd == right.max_cost_microusd
}

fn compute_arm_metrics(targets: &[&TargetOutcome]) -> Vec<ArmMetrics> {
    ARMS.iter()
        .map(|arm_name| {
            let arms: Vec<&ArmOutcome> = targets
                .iter()
                .flat_map(|target| target.arms.iter())
                .filter(|arm| arm.arm == *arm_name)
                .collect();
            let target_count = arms.len();
            let checker_successes = arms.iter().filter(|arm| arm.checker_passed).count();
            let semantic_gate_successes =
                arms.iter().filter(|arm| arm.semantic_gate_passed).count();
            let total_tokens: u128 = arms.iter().map(|arm| u128::from(arm.tokens_used)).sum();
            let total_wall_ms: u128 = arms.iter().map(|arm| u128::from(arm.wall_ms)).sum();
            let total_cost_microusd: u128 =
                arms.iter().map(|arm| u128::from(arm.cost_microusd)).sum();
            let total_model_calls: u64 =
                arms.iter().map(|arm| u64::from(arm.model_calls_used)).sum();
            ArmMetrics {
                arm: (*arm_name).to_string(),
                target_count,
                checker_successes,
                semantic_gate_successes,
                checker_success_rate: ratio(checker_successes, target_count),
                semantic_gate_success_rate: ratio(semantic_gate_successes, target_count),
                total_model_calls,
                total_tokens: total_tokens.min(u128::from(u64::MAX)) as u64,
                total_wall_ms,
                total_cost_microusd,
                mean_wall_ms_per_target: total_wall_ms as f64 / target_count as f64,
                mean_tokens_per_target: total_tokens as f64 / target_count as f64,
                cost_microusd_per_verified_solve: if semantic_gate_successes == 0 {
                    None
                } else {
                    Some(total_cost_microusd as f64 / semantic_gate_successes as f64)
                },
            }
        })
        .collect()
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator as f64
}

fn compute_family_aggregates(
    targets: &[&TargetOutcome],
    baseline_arm: &str,
) -> Result<Vec<FamilyAggregate>> {
    let mut aggregates: BTreeMap<String, FamilyAggregate> = BTreeMap::new();
    for target in targets {
        let treatment = target
            .arms
            .iter()
            .find(|arm| arm.arm == TREATMENT_ARM)
            .context("treatment arm F missing after validation")?;
        let baseline = target
            .arms
            .iter()
            .find(|arm| arm.arm == baseline_arm)
            .context("selected baseline arm missing after validation")?;
        let treatment_success = treatment.semantic_gate_passed;
        let baseline_success = baseline.semantic_gate_passed;
        let delta = (if treatment_success { 1_i64 } else { 0_i64 })
            - (if baseline_success { 1_i64 } else { 0_i64 });
        let aggregate = aggregates
            .entry(target.family_id.clone())
            .or_insert_with(|| FamilyAggregate {
                family_id: target.family_id.clone(),
                target_count: 0,
                treatment_semantic_gate_successes: 0,
                baseline_semantic_gate_successes: 0,
                paired_delta_sum: 0,
                paired_effect_percentage_points: 0.0,
            });
        aggregate.target_count += 1;
        aggregate.treatment_semantic_gate_successes += if treatment_success { 1 } else { 0 };
        aggregate.baseline_semantic_gate_successes += if baseline_success { 1 } else { 0 };
        aggregate.paired_delta_sum += delta;
    }
    for aggregate in aggregates.values_mut() {
        if aggregate.target_count == 0 {
            bail!(
                "family {} has no eligible target pairs",
                aggregate.family_id
            );
        }
        aggregate.paired_effect_percentage_points =
            aggregate.paired_delta_sum as f64 / aggregate.target_count as f64 * 100.0;
    }
    Ok(aggregates.into_values().collect())
}

fn confirmatory_utility_gate(
    effect: f64,
    lower: f64,
    upper: f64,
    clusters: usize,
) -> (bool, bool, bool, &'static str) {
    // A nonzero positive estimate is insufficient when all component-bootstrap
    // replicates collapse to one value: such an interval has no empirical
    // resolution and must never pass the confirmatory utility gate.
    let interval_informative = upper - lower > f64::EPSILON;
    // Keep the literal interval fact separate from whether it is informative:
    // [a, a] with a > 0 excludes zero mathematically but cannot qualify a claim.
    let interval_positive = lower > 0.0;
    let effect_gate = effect >= 0.10;
    let passed = interval_informative
        && interval_positive
        && effect_gate
        && clusters >= MIN_CONFIRMATORY_FAMILY_CLUSTERS;
    let decision = if clusters < MIN_CONFIRMATORY_FAMILY_CLUSTERS {
        "underpowered_too_few_independent_family_clusters"
    } else if !interval_informative {
        "no_claim_degenerate_bootstrap_interval"
    } else if passed {
        "pass_preregistered_utility_gate"
    } else {
        "no_claim_preregistered_utility_gate_not_met"
    };
    (interval_informative, interval_positive, passed, decision)
}

fn paired_cluster_bootstrap(
    families: &[FamilyAggregate],
    replicates: usize,
    seed: u64,
) -> Result<(f64, f64, f64, usize, usize)> {
    if families.is_empty() {
        bail!("no family clusters available for bootstrap");
    }
    if replicates != EXPECTED_BOOTSTRAP_REPLICATES {
        bail!(
            "protocol requires exactly {EXPECTED_BOOTSTRAP_REPLICATES} bootstrap replicates, got {replicates}"
        );
    }
    let total_delta: i64 = families.iter().map(|family| family.paired_delta_sum).sum();
    let total_pairs: usize = families.iter().map(|family| family.target_count).sum();
    if total_pairs == 0 {
        bail!("no target pairs available for bootstrap");
    }
    let estimate = total_delta as f64 / total_pairs as f64;
    let mut rng = SplitMix64 { state: seed };
    let mut bootstrapped = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut sum = 0_i64;
        let mut n = 0_usize;
        for _ in 0..families.len() {
            let sampled = &families[rng.index(families.len())];
            sum += sampled.paired_delta_sum;
            n += sampled.target_count;
        }
        if n == 0 {
            bail!("bootstrap replicate contained no target pairs");
        }
        bootstrapped.push(sum as f64 / n as f64);
    }
    bootstrapped.sort_by(f64::total_cmp);
    let lower = quantile(&bootstrapped, 0.025);
    let upper = quantile(&bootstrapped, 0.975);
    Ok((estimate, lower, upper, total_pairs, families.len()))
}

fn quantile(sorted_values: &[f64], probability: f64) -> f64 {
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }
    let position = probability * (sorted_values.len() - 1) as f64;
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;
    let fraction = position - lower_index as f64;
    sorted_values[lower_index]
        + fraction * (sorted_values[upper_index] - sorted_values[lower_index])
}

fn validate_repo_relative_path(value: &str) -> Result<()> {
    let path = Path::new(value);
    if value.trim().is_empty() || path.is_absolute() || value.contains('\\') {
        bail!("expected a non-empty repository-relative POSIX path");
    }
    if path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        bail!("path may not escape the repository root");
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64 || !value.chars().all(|character| character.is_ascii_hexdigit()) {
        bail!("expected a 64-character SHA-256 hex digest");
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper_exclusive: usize) -> usize {
        let bound = upper_exclusive as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }

    fn uniform01(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform01().ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * self.uniform01();
        radius * angle.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_split_assignment_and_digest_are_order_independent() {
        let mut families: Vec<SplitFamily> = (1..=20)
            .map(|id| SplitFamily {
                family_id: format!("{id:03}"),
                split: String::new(),
            })
            .collect();
        let assignments = expected_split_assignments(&families, EXPECTED_SPLIT_SEED)
            .expect("unique IDs should produce assignments");
        for family in &mut families {
            family.split = assignments
                .get(&family.family_id)
                .expect("family should be assigned")
                .clone();
        }
        let counts = families.iter().fold(BTreeMap::new(), |mut map, family| {
            *map.entry(family.split.as_str()).or_insert(0_usize) += 1;
            map
        });
        assert_eq!(counts.get("train"), Some(&14));
        assert_eq!(counts.get("validation"), Some(&3));
        assert_eq!(counts.get("test"), Some(&3));

        let digest = partition_digest_sha256(&families);
        families.reverse();
        assert_eq!(partition_digest_sha256(&families), digest);

        families[0].split = match families[0].split.as_str() {
            "train" => "test".to_string(),
            _ => "train".to_string(),
        };
        assert_ne!(partition_digest_sha256(&families), digest);
    }

    #[test]
    fn duplicate_family_ids_are_rejected_by_split_assignment() {
        let families = vec![
            SplitFamily {
                family_id: "dup".to_string(),
                split: "train".to_string(),
            },
            SplitFamily {
                family_id: "dup".to_string(),
                split: "test".to_string(),
            },
        ];
        assert!(
            expected_split_assignments(&families, EXPECTED_SPLIT_SEED)
                .expect_err("duplicate IDs must fail")
                .to_string()
                .contains("duplicate family ID")
        );
    }

    #[test]
    fn run_design_digest_excludes_power_receipt_to_avoid_a_hash_cycle() {
        let mut config = test_config();
        let base_digest = run_design_sha256(&config).expect("base run design should serialize");
        config.power_report_sha256 = Some("f".repeat(64));
        let final_digest = run_design_sha256(&config)
            .expect("adding receipt hash must not change the design digest");
        assert_eq!(base_digest, final_digest);
    }

    #[test]
    fn splitmix_sampling_is_deterministic() {
        let mut left = SplitMix64 { state: 20261020 };
        let mut right = SplitMix64 { state: 20261020 };
        for _ in 0..256 {
            assert_eq!(left.index(56), right.index(56));
        }
    }

    #[test]
    fn quantile_uses_linear_interpolation() {
        let values = [0.0, 10.0, 20.0, 30.0];
        assert!((quantile(&values, 0.5) - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn baseline_selection_uses_alphabetical_tie_break() {
        let receipt = BaselineSelectionReceipt {
            schema_version: "pitb.baseline-selection.v0".to_string(),
            selected_split: "validation".to_string(),
            metric: "semantic_gate_success_rate".to_string(),
            selected_arm: "A".to_string(),
            frozen_before_test: true,
            test_outcomes_seen_before_selection: false,
            tie_break_rule: "highest score; alphabetical arm ID on exact tie".to_string(),
            validation_results_sha256: "d".repeat(64),
            arm_scores: ["A", "B", "C", "D", "E", "I"]
                .into_iter()
                .map(|arm| (arm.to_string(), 0.5))
                .collect(),
        };
        let validation_bytes = b"synthetic validation rows".as_slice();
        let mut receipt = receipt;
        receipt.validation_results_sha256 = sha256_hex(validation_bytes);
        validate_baseline_receipt(&receipt, validation_bytes)
            .expect("valid baseline receipt and validation hash should pass");
    }

    #[test]
    fn baseline_scores_are_recomputed_from_raw_validation_outcomes() {
        let mut target = test_target("012", "validation-target", false, false);
        target.split = "validation".to_string();
        let retrieval = target
            .arms
            .iter_mut()
            .find(|arm| arm.arm == "I")
            .expect("arm I should exist");
        retrieval.status = OutcomeStatus::VerifiedSuccess;
        retrieval.checker_passed = true;
        retrieval.semantic_gate_passed = true;
        let targets = vec![&target];
        let scores = compute_baseline_scores(&targets).expect("baseline scores from real rows");
        assert_eq!(scores.get("I"), Some(&1.0));
        assert_eq!(scores.get("A"), Some(&0.0));

        let mut receipt = BaselineSelectionReceipt {
            schema_version: "pitb.baseline-selection.v0".to_string(),
            selected_split: "validation".to_string(),
            metric: "semantic_gate_success_rate".to_string(),
            selected_arm: "I".to_string(),
            frozen_before_test: true,
            test_outcomes_seen_before_selection: false,
            tie_break_rule: "highest score; alphabetical arm ID on exact tie".to_string(),
            validation_results_sha256: "e".repeat(64),
            arm_scores: scores.clone(),
        };
        validate_baseline_scores(&receipt, &scores)
            .expect("receipt should agree with raw validation outcomes");
        receipt.selected_arm = "A".to_string();
        assert!(validate_baseline_scores(&receipt, &scores).is_err());
    }

    #[test]
    fn mismatched_statement_digests_are_not_silently_normalized() {
        assert!(validate_sha256("not-a-digest").is_err());
        assert!(validate_sha256(&"a".repeat(64)).is_ok());
    }

    #[test]
    fn repository_paths_cannot_escape_the_checkout() {
        assert!(validate_repo_relative_path("lean/OAI/Target.lean").is_ok());
        assert!(validate_repo_relative_path("../outside.lean").is_err());
        assert!(validate_repo_relative_path("/absolute/path.lean").is_err());
        assert!(validate_repo_relative_path("lean\\..\\outside.lean").is_err());
    }

    #[test]
    fn positive_degenerate_interval_cannot_pass_confirmatory_gate() {
        let (informative, excludes_zero, passed, decision) =
            confirmatory_utility_gate(0.25, 0.25, 0.25, MIN_CONFIRMATORY_FAMILY_CLUSTERS);
        assert!(!informative);
        assert!(excludes_zero, "the interval is numerically positive even though it is uninformative");
        assert!(!passed);
        assert_eq!(decision, "no_claim_degenerate_bootstrap_interval");
    }

    #[test]
    fn identical_positive_family_effects_cannot_pass_after_real_bootstrap() {
        let families: Vec<FamilyAggregate> = (0..MIN_CONFIRMATORY_FAMILY_CLUSTERS)
            .map(|index| FamilyAggregate {
                family_id: format!("family-{index:02}"),
                target_count: 1,
                treatment_semantic_gate_successes: 1,
                baseline_semantic_gate_successes: 0,
                paired_delta_sum: 1,
                paired_effect_percentage_points: 100.0,
            })
            .collect();

        let (effect, lower, upper, target_pairs, clusters) =
            paired_cluster_bootstrap(&families, EXPECTED_BOOTSTRAP_REPLICATES, 20261028)
                .expect("the full bootstrap should accept valid family aggregates");
        assert_eq!((effect, lower, upper), (1.0, 1.0, 1.0));
        assert_eq!(target_pairs, MIN_CONFIRMATORY_FAMILY_CLUSTERS);
        assert_eq!(clusters, MIN_CONFIRMATORY_FAMILY_CLUSTERS);

        let (informative, excludes_zero, passed, decision) =
            confirmatory_utility_gate(effect, lower, upper, clusters);
        assert!(!informative);
        assert!(excludes_zero);
        assert!(!passed);
        assert_eq!(decision, "no_claim_degenerate_bootstrap_interval");
    }

    #[test]
    fn informative_positive_interval_can_pass_only_with_effect_and_cluster_gates() {
        let (informative, excludes_zero, passed, decision) =
            confirmatory_utility_gate(0.25, 0.12, 0.31, MIN_CONFIRMATORY_FAMILY_CLUSTERS);
        assert!(informative);
        assert!(excludes_zero);
        assert!(passed);
        assert_eq!(decision, "pass_preregistered_utility_gate");

        let (_, _, passed_below_effect, decision_below_effect) =
            confirmatory_utility_gate(0.09, 0.01, 0.15, MIN_CONFIRMATORY_FAMILY_CLUSTERS);
        assert!(!passed_below_effect);
        assert_eq!(decision_below_effect, "no_claim_preregistered_utility_gate_not_met");

        let (_, _, passed_underpowered, decision_underpowered) =
            confirmatory_utility_gate(0.25, 0.12, 0.31, MIN_CONFIRMATORY_FAMILY_CLUSTERS - 1);
        assert!(!passed_underpowered);
        assert_eq!(decision_underpowered, "underpowered_too_few_independent_family_clusters");
    }

    #[test]
    fn clustered_bootstrap_is_deterministic_and_publishes_family_aggregates() {
        let first = test_target("090", "target-090-a", true, false);
        let second = test_target("091", "target-091-a", true, false);
        let targets = vec![&first, &second];
        let aggregates = compute_family_aggregates(&targets, "A").expect("matched target outcomes");
        assert_eq!(aggregates.len(), 2);
        assert_eq!(aggregates[0].target_count, 1);
        assert_eq!(aggregates[0].paired_delta_sum, 1);
        assert_eq!(aggregates[0].paired_effect_percentage_points, 100.0);

        let left = paired_cluster_bootstrap(&aggregates, EXPECTED_BOOTSTRAP_REPLICATES, 20261020)
            .expect("valid family bootstrap");
        let right = paired_cluster_bootstrap(&aggregates, EXPECTED_BOOTSTRAP_REPLICATES, 20261020)
            .expect("same seed is repeatable");
        assert_eq!(left, right);
        assert_eq!(left, (1.0, 1.0, 1.0, 2, 2));
    }

    #[test]
    fn exposure_quarantine_accepts_required_ids_on_their_frozen_splits() {
        let frozen = BTreeMap::from([
            ("012".to_string(), "validation".to_string()),
            ("089".to_string(), "test".to_string()),
            ("099".to_string(), "test".to_string()),
            ("017".to_string(), "validation".to_string()),
        ]);
        let validation = vec!["012".to_string(), "017".to_string()];
        let test = vec!["089".to_string(), "099".to_string()];
        validate_exposure_quarantine(&validation, &test, &frozen)
            .expect("separate exposure lists should match the frozen partition");
    }

    #[test]
    fn exposure_quarantine_rejects_each_required_id_in_the_wrong_split() {
        let frozen = BTreeMap::from([
            ("012".to_string(), "validation".to_string()),
            ("017".to_string(), "validation".to_string()),
            ("089".to_string(), "test".to_string()),
        ]);
        let validation_has_test_id = vec!["012".to_string(), "089".to_string()];
        let test = vec!["089".to_string()];
        assert!(validate_exposure_quarantine(&validation_has_test_id, &test, &frozen)
            .expect_err("test family 089 must not enter validation exposure list")
            .to_string().contains("assigned to the frozen validation split"));

        let validation = vec!["012".to_string()];
        let test_has_validation_id = vec!["089".to_string(), "017".to_string()];
        assert!(validate_exposure_quarantine(&validation, &test_has_validation_id, &frozen)
            .expect_err("validation family 017 must not enter test exposure list")
            .to_string().contains("assigned to the frozen test split"));
    }

    #[test]
    fn exposure_quarantine_rejects_missing_required_ids_duplicates_and_overlap() {
        let frozen = BTreeMap::from([
            ("012".to_string(), "validation".to_string()),
            ("089".to_string(), "test".to_string()),
        ]);
        let missing_validation = Vec::<String>::new();
        let test = vec!["089".to_string()];
        assert!(validate_exposure_quarantine(&missing_validation, &test, &frozen).is_err());

        let validation = vec!["012".to_string(), "012".to_string()];
        assert!(validate_exposure_quarantine(&validation, &test, &frozen)
            .expect_err("duplicate validation IDs must fail")
            .to_string().contains("duplicates"));

        let overlap_validation = vec!["012".to_string(), "089".to_string()];
        let overlap_test = vec!["089".to_string()];
        assert!(validate_exposure_quarantine(&overlap_validation, &overlap_test, &frozen)
            .expect_err("shared IDs must fail")
            .to_string().contains("a family cannot be listed as exposed in both validation and test"));
    }

    #[test]
    fn known_exposed_family_cannot_enter_confirmatory_denominator() {
        let target = test_target(REQUIRED_EXPOSED_TEST_FAMILY, "target-exposed", false, false);
        let config = test_config();
        let frozen_families =
            BTreeMap::from([(REQUIRED_EXPOSED_TEST_FAMILY.to_string(), "test".to_string())]);
        let error = validate_outcomes(&[target], &config, &frozen_families, "test")
            .expect_err("known-exposed family must be rejected");
        assert!(error.to_string().contains("known-exposed family 089"));
    }

    #[test]
    fn known_exposed_validation_family_cannot_enter_baseline_denominator() {
        let mut target = test_target(
            REQUIRED_EXPOSED_VALIDATION_FAMILY,
            "target-exposed-validation",
            false,
            false,
        );
        target.split = "validation".to_string();
        let config = test_config();
        let frozen_families = BTreeMap::from([(
            REQUIRED_EXPOSED_VALIDATION_FAMILY.to_string(),
            "validation".to_string(),
        )]);
        let error = validate_outcomes(&[target], &config, &frozen_families, "validation")
            .expect_err("known-exposed validation family must not enter baseline tuning");
        assert!(error
            .to_string()
            .contains("known-exposed family 012 was included as eligible"));
    }

    #[test]
    fn known_exposed_validation_family_can_be_kept_as_explicit_exclusion() {
        let mut target = test_target(
            REQUIRED_EXPOSED_VALIDATION_FAMILY,
            "target-exposed-validation-excluded",
            false,
            false,
        );
        target.split = "validation".to_string();
        target.eligibility = Eligibility::Excluded;
        target.exclusion_reason = Some(
            "Known researcher exposure; excluded from validation baseline selection".to_string(),
        );
        target.arms.clear();
        let config = test_config();
        let frozen_families = BTreeMap::from([(
            REQUIRED_EXPOSED_VALIDATION_FAMILY.to_string(),
            "validation".to_string(),
        )]);
        let eligible = validate_outcomes(&[target], &config, &frozen_families, "validation")
            .expect("explicit exposure exclusion remains auditable");
        assert!(eligible.is_empty());
    }

    #[test]
    fn every_frozen_test_family_needs_an_outcome_or_exclusion_row() {
        let target = test_target("090", "target-090", false, false);
        let config = test_config();
        let frozen_families = BTreeMap::from([
            ("090".to_string(), "test".to_string()),
            ("091".to_string(), "test".to_string()),
        ]);
        let error = validate_outcomes(&[target], &config, &frozen_families, "test")
            .expect_err("missing test family must be surfaced rather than dropped");
        assert!(error.to_string().contains("091"));
        assert!(error.to_string().contains("explicit outcome/exclusion row"));
    }

    #[test]
    fn arm_budget_mismatch_is_rejected_before_scoring() {
        let mut target = test_target("090", "target-budget-mismatch", false, false);
        target.arms[0].budget.max_tokens += 1;
        let config = test_config();
        let frozen_families = BTreeMap::from([("090".to_string(), "test".to_string())]);
        let error = validate_outcomes(&[target], &config, &frozen_families, "test")
            .expect_err("unequal arm budget must be rejected");
        assert!(
            error
                .to_string()
                .contains("different from the frozen matched budget")
        );
    }

    #[test]
    fn simulated_cluster_bootstrap_reports_both_endpoints_and_detects_degeneracy() {
        let families: Vec<SimulatedFamily> = (0..MIN_CONFIRMATORY_FAMILY_CLUSTERS)
            .map(|_| SimulatedFamily {
                target_count: 1,
                paired_delta_sum: 1,
            })
            .collect();
        let mut rng = SplitMix64 { state: 20261028 };
        let interval = simulated_cluster_bootstrap_interval(&families, 1_000, &mut rng);
        assert_eq!(interval, [1.0, 1.0]);
        let (informative, _, passed, reason) =
            confirmatory_utility_gate(1.0, interval[0], interval[1], families.len());
        assert!(!informative);
        assert!(!passed);
        assert_eq!(reason, "no_claim_degenerate_bootstrap_interval");
    }

    #[test]
    fn weak_null_family_effects_have_zero_target_weighted_mean_and_valid_rates() {
        let family_target_counts: Vec<FamilyTargetCount> = (0..24)
            .map(|index| FamilyTargetCount {
                family_id: format!("weak-null-{index:02}"),
                eligible_target_count: 1 + index % 5,
            })
            .collect();
        let baseline_rates: Vec<f64> = (0..24)
            .map(|index| 0.10 + ((index * 7) % 8) as f64 * 0.10)
            .collect();
        let mut rng = SplitMix64 { state: 20261028 };
        let effects = sample_centered_weak_null_family_effects(
            &family_target_counts,
            &baseline_rates,
            0.20,
            &mut rng,
        )
        .expect("aligned family rates should produce a weak-null effect vector");

        let weighted_effect_sum: f64 = effects
            .iter()
            .zip(&family_target_counts)
            .map(|(effect, family)| effect * family.eligible_target_count as f64)
            .sum();
        assert!(weighted_effect_sum.abs() < 1e-10);
        assert!(effects
            .iter()
            .zip(&baseline_rates)
            .all(|(effect, baseline)| (0.0..=1.0).contains(&(*baseline + *effect))));
        let realized_sd = weighted_family_effect_sd(&effects, &family_target_counts);
        assert!(realized_sd > 0.0);
        assert!(realized_sd <= 0.20 + 1e-12);
    }

    #[test]
    fn joint_success_probability_interpolates_independence_to_frechet_upper_bound() {
        let baseline = 0.25;
        let treatment = 0.45;
        let independent = joint_success_probability_at_association(baseline, treatment, 0.0)
            .expect("independent coupling should be valid");
        let middle = joint_success_probability_at_association(baseline, treatment, 0.5)
            .expect("intermediate coupling should be valid");
        let maximal = joint_success_probability_at_association(baseline, treatment, 1.0)
            .expect("upper Frechet coupling should be valid");
        assert!((independent - baseline * treatment).abs() < 1e-12);
        assert!((middle - (baseline * treatment + 0.5 * (baseline.min(treatment) - baseline * treatment))).abs() < 1e-12);
        assert!((maximal - baseline.min(treatment)).abs() < 1e-12);
        assert!(joint_success_probability_at_association(baseline, treatment, -0.1).is_err());
        assert!(joint_success_probability_at_association(baseline, treatment, 1.1).is_err());
    }

    #[test]
    fn paired_bernoulli_sampler_preserves_both_marginals_and_joint_success_probability() {
        let baseline = 0.25;
        let treatment = 0.45;
        for (index, association) in [0.0_f64, 0.5, 0.9].into_iter().enumerate() {
            let joint = joint_success_probability_at_association(baseline, treatment, association)
                .expect("registered association fraction should be valid");
            let mut rng = SplitMix64 { state: 20261028 + index as u64 };
            let trials = 60_000usize;
            let mut baseline_successes = 0usize;
            let mut treatment_successes = 0usize;
            let mut joint_successes = 0usize;
            for _ in 0..trials {
                let (base, treated) = sample_paired_bernoulli_from_joint_probability(
                    baseline, treatment, joint, &mut rng,
                );
                baseline_successes += usize::from(base);
                treatment_successes += usize::from(treated);
                joint_successes += usize::from(base && treated);
            }
            let empirical_base = baseline_successes as f64 / trials as f64;
            let empirical_treatment = treatment_successes as f64 / trials as f64;
            let empirical_joint = joint_successes as f64 / trials as f64;
            assert!((empirical_base - baseline).abs() < 0.01);
            assert!((empirical_treatment - treatment).abs() < 0.01);
            assert!((empirical_joint - joint).abs() < 0.01);
        }
    }

    #[test]
    fn weak_null_calibration_replays_deterministically_and_counts_degenerate_intervals() {
        let families: Vec<FamilyTargetCount> = (0..24)
            .map(|index| FamilyTargetCount {
                family_id: format!("weak-family-{index:02}"),
                eligible_target_count: 1 + index % 4,
            })
            .collect();
        let left = simulate_weak_null_scenario(
            &families, 0.35, 0.10, 0.20, 0.5, 64, 128, 20261028,
        )
        .expect("weak-null calibration should execute");
        let right = simulate_weak_null_scenario(
            &families, 0.35, 0.10, 0.20, 0.5, 64, 128, 20261028,
        )
        .expect("same weak-null seed must replay identically");

        assert_eq!(left.status, "estimated");
        assert_eq!(left.true_target_weighted_mean_effect_percentage_points, 0.0);
        assert_eq!(left.false_claims, right.false_claims);
        assert_eq!(left.degenerate_interval_trials, right.degenerate_interval_trials);
        assert_eq!(
            left.estimated_false_claim_probability,
            right.estimated_false_claim_probability
        );
        assert_eq!(
            left.false_claim_probability_lower_95,
            right.false_claim_probability_lower_95
        );
        assert_eq!(
            left.false_claim_probability_upper_95,
            right.false_claim_probability_upper_95
        );
        assert_eq!(
            left.false_claim_probability_upper_simultaneous_95,
            right.false_claim_probability_upper_simultaneous_95
        );
        assert_eq!(
            left.mean_observed_effect_percentage_points,
            right.mean_observed_effect_percentage_points
        );
        assert_eq!(
            left.mean_realized_family_effect_heterogeneity_sd_percentage_points,
            right.mean_realized_family_effect_heterogeneity_sd_percentage_points
        );
        assert!(left.false_claims <= 64 - left.degenerate_interval_trials);
        assert!(left.mean_realized_family_effect_heterogeneity_sd_percentage_points
            .is_some_and(|sd| sd > 0.0 && sd <= 10.0 + 1e-8));
        assert!(left.estimated_false_claim_probability
            .is_some_and(|rate| (0.0..=1.0).contains(&rate)));
        assert!(left.false_claim_probability_lower_95
            .zip(left.estimated_false_claim_probability)
            .zip(left.false_claim_probability_upper_95)
            .is_some_and(|((lower, rate), upper)| lower <= rate && rate <= upper));
        assert!(left.false_claim_probability_upper_simultaneous_95
            .zip(left.false_claim_probability_upper_95)
            .is_some_and(|(adjusted, per_cell)| adjusted >= per_cell));
    }

    #[test]
    fn strict_null_power_grid_spans_registered_pair_association() {
        let families: Vec<FamilyTargetCount> = (0..24)
            .map(|index| FamilyTargetCount {
                family_id: format!("assoc-family-{index:02}"),
                eligible_target_count: 1 + index % 3,
            })
            .collect();
        for (index, association) in [0.0_f64, 0.5, 0.9].into_iter().enumerate() {
            let scenario = simulate_power_scenario(
                &families,
                0.35,
                0.0,
                0.20,
                association,
                64,
                128,
                20261028 + index as u64,
            )
            .expect("registered association scenario should be estimable");
            assert_eq!(scenario.pair_association_fraction, association);
            assert_eq!(scenario.status, "estimated");
            assert!(scenario.null_effect_false_claim_probability_upper_simultaneous_95.is_some());
        }
    }

    #[test]
    fn strict_null_scenario_emits_the_simultaneous_upper_bound() {
        let families: Vec<FamilyTargetCount> = (0..24)
            .map(|index| FamilyTargetCount {
                family_id: format!("null-family-{index:02}"),
                eligible_target_count: 1 + index % 3,
            })
            .collect();
        let scenario = simulate_power_scenario(
            &families, 0.35, 0.0, 0.20, 0.0, 64, 128, 20261028,
        )
        .expect("strict-null scenario should be estimated");
        assert_eq!(scenario.status, "estimated");
        assert_eq!(
            scenario.null_effect_false_claim_probability_upper_simultaneous_95,
            Some(simultaneous_null_upper_bound(scenario.utility_gate_passes, 64))
        );
        assert!(scenario.null_effect_false_claim_probability_upper_simultaneous_95
            .zip(scenario.estimated_power_upper_95)
            .is_some_and(|(adjusted, per_cell)| adjusted >= per_cell));
    }

    #[test]
    fn power_simulation_is_reproducible_for_a_frozen_seed() {
        let families: Vec<FamilyTargetCount> = (0..24)
            .map(|index| FamilyTargetCount {
                family_id: format!("family-{index:02}"),
                eligible_target_count: if index % 3 == 0 { 2 } else { 1 },
            })
            .collect();
        let left = simulate_power_scenario(&families, 0.25, 0.20, 0.10, 0.0, 64, 100, 20261020)
            .expect("synthetic power scenario should run");
        let right = simulate_power_scenario(&families, 0.25, 0.20, 0.10, 0.0, 64, 100, 20261020)
            .expect("repeating a frozen seed must be deterministic");
        assert_eq!(left.status, "estimated");
        assert_eq!(left.utility_gate_passes, right.utility_gate_passes);
        assert_eq!(
            left.degenerate_interval_trials,
            right.degenerate_interval_trials
        );
        assert_eq!(
            left.estimated_utility_gate_power,
            right.estimated_utility_gate_power
        );
        assert_eq!(
            left.mean_observed_effect_percentage_points,
            right.mean_observed_effect_percentage_points
        );
        assert_eq!(
            left.mean_lower_95ci_percentage_points,
            right.mean_lower_95ci_percentage_points
        );
        assert!(
            left.estimated_utility_gate_power
                .is_some_and(|power| (0.0..=1.0).contains(&power))
        );
        let power = left.estimated_utility_gate_power.expect("estimated power");
        assert!(
            left.estimated_power_lower_95
                .is_some_and(|lower| lower <= power)
        );
        assert!(
            left.estimated_power_upper_95
                .is_some_and(|upper| upper >= power)
        );
    }

    #[test]
    fn simultaneous_power_lower_bound_is_more_conservative_than_per_cell_bound() {
        let per_cell = wilson_interval(850, 1_000).0;
        let simultaneous = simultaneous_power_lower_bound(850, 1_000);
        assert!(simultaneous < per_cell);
        assert!(simultaneous > 0.80);
        assert_eq!(simultaneous_power_lower_bound(0, 0), 0.0);
    }

    #[test]
    fn null_grid_uses_conservative_simultaneous_wilson_upper_bound() {
        let per_cell_upper = wilson_interval(2, 1_000).1;
        let simultaneous_upper = simultaneous_null_upper_bound(2, 1_000);
        assert!(simultaneous_upper >= per_cell_upper);
        assert!((0.0..=1.0).contains(&simultaneous_upper));
        assert!(simultaneous_null_upper_bound(0, 1_000) <= 0.05);
    }

    #[test]
    fn wilson_interval_brackets_observed_bernoulli_rate() {
        let (lower, upper) = wilson_interval(80, 100);
        assert!(lower < 0.80);
        assert!(upper > 0.80);
        let (all_lower, all_upper) = wilson_interval(100, 100);
        assert!(all_lower <= 1.0);
        assert_eq!(all_upper, 1.0);
        assert_eq!(wilson_interval(0, 0), (0.0, 1.0));
    }

    #[test]
    fn treatment_family_probabilities_preserve_the_marginal_alternative_rate() {
        let baseline = 0.25;
        let treatment = 0.45;
        let latent_baseline_rates = [0.0, 0.5];
        let mean_treatment = latent_baseline_rates
            .iter()
            .map(|family_rate| {
                treatment_rate_given_baseline_family(*family_rate, baseline, treatment)
            })
            .sum::<f64>()
            / latent_baseline_rates.len() as f64;
        assert!((mean_treatment - treatment).abs() < 1e-12);
        assert!(latent_baseline_rates.iter().all(|family_rate| {
            let treatment_family_rate =
                treatment_rate_given_baseline_family(*family_rate, baseline, treatment);
            (0.0..=1.0).contains(&treatment_family_rate)
        }));
    }

    fn test_config() -> RunConfig {
        RunConfig {
            schema_version: RUN_CONFIG_SCHEMA.to_string(),
            protocol_id: EXPECTED_PROTOCOL_ID.to_string(),
            protocol_sha256: "a".repeat(64),
            split_manifest_sha256: "b".repeat(64),
            partition_digest_sha256: "c".repeat(64),
            baseline_selection_receipt_sha256: "d".repeat(64),
            known_exposed_validation_family_ids: vec![
                REQUIRED_EXPOSED_VALIDATION_FAMILY.to_string(),
            ],
            known_exposed_test_family_ids: vec![REQUIRED_EXPOSED_TEST_FAMILY.to_string()],
            bootstrap_replicates: EXPECTED_BOOTSTRAP_REPLICATES,
            bootstrap_seed: 20261020,
            model_version: "test-model-v1".to_string(),
            checker_version: "lean-checker-test-v1".to_string(),
            budget: Budget {
                max_model_calls: 8,
                max_tokens: 16_000,
                max_wall_ms: 300_000,
                max_cost_microusd: 500_000,
            },
            power_report_sha256: None,
        }
    }

    fn test_target(
        family_id: &str,
        target_id: &str,
        treatment_success: bool,
        baseline_success: bool,
    ) -> TargetOutcome {
        let budget = Budget {
            max_model_calls: 8,
            max_tokens: 16_000,
            max_wall_ms: 300_000,
            max_cost_microusd: 500_000,
        };
        let arms = ARMS
            .iter()
            .map(|arm_id| {
                let success = match *arm_id {
                    "A" => baseline_success,
                    "F" => treatment_success,
                    _ => false,
                };
                ArmOutcome {
                    arm: (*arm_id).to_string(),
                    status: if success {
                        OutcomeStatus::VerifiedSuccess
                    } else {
                        OutcomeStatus::Failed
                    },
                    checker_passed: success,
                    semantic_gate_passed: success,
                    model_version: "test-model-v1".to_string(),
                    checker_version: "lean-checker-test-v1".to_string(),
                    budget: budget.clone(),
                    model_calls_used: 1,
                    tokens_used: 100,
                    wall_ms: 100,
                    cost_microusd: 100,
                    config_sha256: sha256_hex(format!("fixed-config-{arm_id}").as_bytes()),
                    input_sha256: sha256_hex(format!("input-{target_id}-{arm_id}").as_bytes()),
                    retrieval_log_sha256: sha256_hex(
                        format!("retrieval-{target_id}-{arm_id}").as_bytes(),
                    ),
                    checker_receipt_sha256: sha256_hex(
                        format!("checker-{target_id}-{arm_id}").as_bytes(),
                    ),
                    semantic_gate_receipt_sha256: sha256_hex(
                        format!("semantic-{target_id}-{arm_id}").as_bytes(),
                    ),
                }
            })
            .collect();

        TargetOutcome {
            schema_version: RUN_RESULTS_SCHEMA.to_string(),
            target_id: target_id.to_string(),
            family_id: family_id.to_string(),
            split: "test".to_string(),
            eligibility: Eligibility::Eligible,
            source_commit: Some(EXPECTED_UPSTREAM_COMMIT.to_string()),
            theorem_id: Some(format!("Test.{target_id}")),
            artifact_path: Some(format!("lean/Test/{target_id}.lean")),
            artifact_sha256: Some(sha256_hex(format!("artifact-{target_id}").as_bytes())),
            lean_toolchain: Some("leanprover/lean4:fixed-test-toolchain".to_string()),
            mathlib_revision: Some(EXPECTED_UPSTREAM_COMMIT.to_string()),
            statement_sha256: Some(sha256_hex(format!("statement-{target_id}").as_bytes())),
            eligibility_receipt_sha256: Some(sha256_hex(b"eligibility")),
            statement_review_receipt_sha256: Some(sha256_hex(b"statement-review")),
            proof_audit_receipt_sha256: Some(sha256_hex(b"proof-audit")),
            successor_test_receipt_sha256: Some(sha256_hex(b"successor-test")),
            mutation_receipt_sha256: Some(sha256_hex(b"mutation")),
            exclusion_reason: None,
            arms,
        }
    }
}
