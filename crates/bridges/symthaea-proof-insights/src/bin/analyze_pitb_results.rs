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

const RUN_CONFIG_SCHEMA: &str = "pitb.run-config.v0";
const RUN_RESULTS_SCHEMA: &str = "pitb.target-outcome.v0";
const ANALYSIS_REPORT_SCHEMA: &str = "pitb.analysis-report.v0";
const EXPECTED_PROTOCOL_ID: &str = "proof-insight-transfer-v0";
const EXPECTED_UPSTREAM_COMMIT: &str = "1581d4cbdd69a0faf34b2089614fe2a814316a6a";
const EXPECTED_BOOTSTRAP_REPLICATES: usize = 10_000;
const REQUIRED_EXPOSED_FAMILY: &str = "089";
const ARMS: [&str; 8] = ["A", "B", "C", "D", "E", "F", "G", "H"];
const SIMPLE_BASELINES: [&str; 5] = ["A", "B", "C", "D", "E"];
const TREATMENT_ARM: &str = "F";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunConfig {
    schema_version: String,
    protocol_id: String,
    protocol_sha256: String,
    split_manifest_sha256: String,
    partition_digest_sha256: String,
    baseline_selection_receipt_sha256: String,
    known_exposed_family_ids: Vec<String>,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
    model_version: String,
    checker_version: String,
    budget: Budget,
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
    partition_digest_sha256: String,
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
    results_jsonl_sha256: String,
    split_manifest_sha256: String,
    partition_digest_sha256: String,
    baseline_selection_receipt_sha256: String,
    upstream_commit: String,
    selected_baseline_arm: String,
    treatment_arm: &'static str,
    total_test_families_in_frozen_split: usize,
    eligible_target_count: usize,
    eligible_family_count: usize,
    excluded_target_count: usize,
    unknown_eligibility_target_count: usize,
    known_exposed_family_ids: Vec<String>,
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
    minimum_effect_percentage_points: f64,
    confidence_interval_excludes_zero_positive: bool,
    preregistered_utility_gate_passed: bool,
    gate_decision: String,
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
    let config_bytes = read_arg(&args, "--config")?;
    let results_bytes = read_arg(&args, "--results")?;
    let protocol_bytes = read_arg(&args, "--protocol")?;
    let split_bytes = read_arg(&args, "--splits")?;
    let receipt_bytes = read_arg(&args, "--baseline-receipt")?;

    let config: RunConfig =
        serde_json::from_slice(&config_bytes).context("parse frozen run config JSON")?;
    validate_config(&config, &protocol_bytes, &split_bytes, &receipt_bytes)?;

    let split_manifest: SplitManifest =
        serde_json::from_slice(&split_bytes).context("parse frozen split manifest JSON")?;
    let receipt: BaselineSelectionReceipt =
        serde_json::from_slice(&receipt_bytes).context("parse baseline selection receipt JSON")?;
    validate_baseline_receipt(&receipt)?;
    let selected_baseline = receipt.selected_arm.clone();

    let results_digest = sha256_hex(&results_bytes);
    let result_text = String::from_utf8(results_bytes).context("results JSONL must be UTF-8")?;
    let mut outcomes = Vec::new();
    for (line_index, line) in result_text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: TargetOutcome = serde_json::from_str(line)
            .with_context(|| format!("parse results JSONL line {}", line_index + 1))?;
        outcomes.push(row);
    }
    if outcomes.is_empty() {
        bail!("no target outcome rows supplied; no analysis can be claimed");
    }

    let families: BTreeMap<String, String> = split_manifest
        .families
        .iter()
        .map(|family| (family.family_id.clone(), family.split.clone()))
        .collect();
    let eligible = validate_outcomes(&outcomes, &config, &families)?;
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

    let interval_positive = lower > 0.0;
    let effect_gate = effect >= 0.10;
    let passed = interval_positive && effect_gate && clusters >= 2;
    let gate_decision = if clusters < 2 {
        "underpowered_fewer_than_two_family_clusters"
    } else if passed {
        "pass_preregistered_utility_gate"
    } else {
        "no_claim_preregistered_utility_gate_not_met"
    }
    .to_string();

    let interpretation = if passed {
        "The paired, family-clustered estimate met the preregistered effect and interval gates on this test partition. This is within-benchmark utility only; it is not external/SOTA evidence."
    } else if clusters < 2 {
        "Too few eligible family clusters for a meaningful clustered interval. Treat this as an underpowered pilot; do not claim transfer utility."
    } else {
        "The preregistered utility gate was not met. Report the estimate, interval, denominators, and failures; do not claim a win or relax the gate after seeing test outcomes."
    }
    .to_string();

    let report = AnalysisReport {
        schema_version: ANALYSIS_REPORT_SCHEMA,
        protocol_id: config.protocol_id,
        protocol_sha256: config.protocol_sha256,
        run_config_sha256: sha256_hex(&config_bytes),
        results_jsonl_sha256: results_digest,
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
        known_exposed_family_ids: config.known_exposed_family_ids,
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
            minimum_effect_percentage_points: 10.0,
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

fn parse_args(args: Vec<String>) -> Result<BTreeMap<String, String>> {
    let mut parsed = BTreeMap::new();
    let mut iter = args.into_iter();
    while let Some(flag) = iter.next() {
        if ![
            "--config",
            "--results",
            "--protocol",
            "--splits",
            "--baseline-receipt",
            "--out",
        ]
        .contains(&flag.as_str())
        {
            bail!("unknown argument {flag}");
        }
        let value = iter
            .next()
            .with_context(|| format!("missing path after {flag}"))?;
        if parsed.insert(flag.clone(), value).is_some() {
            bail!("duplicate argument {flag}");
        }
    }
    for required in [
        "--config",
        "--results",
        "--protocol",
        "--splits",
        "--baseline-receipt",
        "--out",
    ] {
        if !parsed.contains_key(required) {
            bail!("missing required argument {required}");
        }
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
    if split_manifest.upstream.commit != EXPECTED_UPSTREAM_COMMIT {
        bail!(
            "unexpected upstream commit {}; expected {}",
            split_manifest.upstream.commit,
            EXPECTED_UPSTREAM_COMMIT
        );
    }
    if split_manifest.families.len() != 372 {
        bail!(
            "frozen split must contain exactly 372 families, got {}",
            split_manifest.families.len()
        );
    }
    let mut family_ids = BTreeSet::new();
    let mut split_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for family in &split_manifest.families {
        if !family_ids.insert(family.family_id.as_str()) {
            bail!("duplicate family id in frozen split: {}", family.family_id);
        }
        if !["train", "validation", "test"].contains(&family.split.as_str()) {
            bail!(
                "unknown split label {} for family {}",
                family.split,
                family.family_id
            );
        }
        *split_counts.entry(family.split.as_str()).or_default() += 1;
    }
    for (split, expected_count) in [("train", 260), ("validation", 56), ("test", 56)] {
        if split_counts.get(split).copied().unwrap_or_default() != expected_count {
            bail!(
                "frozen split count mismatch for {split}: expected {expected_count}, got {}",
                split_counts.get(split).copied().unwrap_or_default()
            );
        }
    }
    if config.partition_digest_sha256 != split_manifest.split_plan.partition_digest_sha256 {
        bail!("partition digest does not match the frozen split manifest");
    }
    let exposed_splits: BTreeMap<&str, &str> = split_manifest
        .families
        .iter()
        .map(|family| (family.family_id.as_str(), family.split.as_str()))
        .collect();
    for exposed in &config.known_exposed_family_ids {
        if exposed_splits.get(exposed.as_str()).copied() != Some("test") {
            bail!(
                "known-exposed family {exposed} is absent from or not assigned to the frozen test split"
            );
        }
    }
    if config.bootstrap_replicates != EXPECTED_BOOTSTRAP_REPLICATES {
        bail!(
            "protocol requires exactly {EXPECTED_BOOTSTRAP_REPLICATES} bootstrap replicates, got {}",
            config.bootstrap_replicates
        );
    }
    if config.model_version.trim().is_empty() || config.checker_version.trim().is_empty() {
        bail!("model and checker versions must be explicit");
    }
    if config.known_exposed_family_ids.is_empty()
        || !config
            .known_exposed_family_ids
            .iter()
            .any(|family| family == REQUIRED_EXPOSED_FAMILY)
    {
        bail!(
            "known-exposed family {REQUIRED_EXPOSED_FAMILY} must remain excluded from confirmatory scoring"
        );
    }
    let exposed_unique: BTreeSet<&str> = config
        .known_exposed_family_ids
        .iter()
        .map(String::as_str)
        .collect();
    if exposed_unique.len() != config.known_exposed_family_ids.len() {
        bail!("known_exposed_family_ids contains duplicates");
    }
    Ok(())
}

fn validate_baseline_receipt(receipt: &BaselineSelectionReceipt) -> Result<()> {
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
    let score_keys: BTreeSet<&str> = receipt.arm_scores.keys().map(String::as_str).collect();
    let expected_keys: BTreeSet<&str> = SIMPLE_BASELINES.into_iter().collect();
    if score_keys != expected_keys {
        bail!("baseline receipt must include exactly validation scores for arms A through E");
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
    if !SIMPLE_BASELINES.contains(&receipt.selected_arm.as_str()) {
        bail!("selected baseline must be one of A through E");
    }
    Ok(())
}

fn validate_outcomes(
    outcomes: &[TargetOutcome],
    config: &RunConfig,
    frozen_families: &BTreeMap<String, String>,
) -> Result<Vec<&TargetOutcome>> {
    let mut target_ids = BTreeSet::new();
    let mut eligible_statement_digests = BTreeSet::new();
    let mut arm_config_digests: BTreeMap<String, String> = BTreeMap::new();
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
        if target.split != "test" {
            bail!(
                "target {} is assigned to {}, but this analyzer accepts test rows only",
                target.target_id,
                target.split
            );
        }
        let frozen_split = frozen_families
            .get(&target.family_id)
            .with_context(|| format!("unknown family_id {}", target.family_id))?;
        if frozen_split != "test" {
            bail!(
                "target {} family {} is assigned to {frozen_split}, not test",
                target.target_id,
                target.family_id
            );
        }

        match target.eligibility {
            Eligibility::Eligible => {
                if config.known_exposed_family_ids.contains(&target.family_id) {
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
        if let Some(previous) =
            arm_config_digests.insert(arm.arm.clone(), arm.config_sha256.clone())
        {
            if previous != arm.config_sha256 {
                bail!(
                    "arm {} changed configuration across held-out targets",
                    arm.arm
                );
            }
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
        aggregate.treatment_semantic_gate_successes += usize::from(treatment_success);
        aggregate.baseline_semantic_gate_successes += usize::from(baseline_success);
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
            arm_scores: ["A", "B", "C", "D", "E"]
                .into_iter()
                .map(|arm| (arm.to_string(), 0.5))
                .collect(),
        };
        validate_baseline_receipt(&receipt)
            .expect("alphabetically first tied baseline should pass");
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
}
