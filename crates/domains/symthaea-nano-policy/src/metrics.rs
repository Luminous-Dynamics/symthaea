//! Independent post-outcome metrics for frozen nanotechnology benchmark runs.
//!
//! This evaluator consumes observations after candidate selection. Its input ledger must
//! remain separate from the selector request; outcomes must never be fed back into a
//! policy within the same frozen evaluation.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// Post-run input-ledger contract version.
pub const OUTCOME_LEDGER_SCHEMA_VERSION: &str = "nano-outcome-ledger-v1";
/// Machine-readable metrics-report contract version.
pub const METRICS_REPORT_SCHEMA_VERSION: &str = "nano-benchmark-metrics-v2";
/// Maximum policy runs compared in one ledger.
pub const MAX_ARMS: usize = 8;
/// Maximum target queries per arm.
pub const MAX_TARGET_QUERIES: usize = 2_000;
/// Maximum oracle attempts across one arm.
pub const MAX_TOTAL_OBSERVATIONS_PER_ARM: usize = 500_000;
/// Maximum oracle attempts allowed per target query.
pub const MAX_QUERY_BUDGET: usize = 1_000;
/// Maximum permitted non-negative loss or cost.
pub const MAX_NUMERIC_VALUE: f64 = 1.0e12;
/// Maximum encoded outcome ledger size.
pub const MAX_LEDGER_BYTES: usize = 64 * 1024 * 1024;
/// Default deterministic group-bootstrap replicate count.
pub const DEFAULT_BOOTSTRAP_REPLICATES: usize = 2_000;
/// Minimum bootstrap replicates for an interval.
pub const MIN_BOOTSTRAP_REPLICATES: usize = 100;
/// Minimum independent groups required before reporting a confidence interval.
pub const MIN_INDEPENDENT_GROUPS_FOR_INTERVAL: usize = 5;
/// Maximum bootstrap replicates.
pub const MAX_BOOTSTRAP_REPLICATES: usize = 10_000;

/// One post-selection oracle attempt. This type belongs in the outcome ledger only.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OracleObservation {
    /// One-based, contiguous query rank within the target query.
    pub query_rank: usize,
    /// Candidate chosen before its outcome was known.
    pub candidate_id: String,
    /// Outcome status of this oracle attempt.
    pub status: ObservationStatus,
    /// SHA-256 of the exact raw oracle response or durable failure receipt for this attempt.
    /// The evaluator validates digest shape but does not open or recompute this artifact.
    pub observation_artifact_sha256: String,
    /// Task loss; required for evaluated attempts and forbidden for failed attempts.
    #[serde(default)]
    pub loss: Option<f64>,
    /// All-in cost for this selection/evaluation attempt in the ledger's one unit. Include selector/runtime and oracle costs when claiming end-to-end decision efficiency; failed attempts count.
    pub cost: f64,
}

/// State of one oracle attempt.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservationStatus {
    /// Oracle returned a usable finite task loss.
    Evaluated,
    /// Candidate was invalid under the frozen task/oracle contract.
    InvalidCandidate,
    /// Oracle returned no usable result.
    OracleFailed,
    /// Oracle attempt exceeded its declared time limit.
    Timeout,
}

/// One frozen target-query scenario and the attempts made against it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TargetQuery {
    /// Stable target-query identity, identical across every policy arm.
    pub target_query_id: String,
    /// Independent resampling group declared in the frozen evaluation manifest.
    pub group_id: String,
    /// Chronological oracle attempts; early stopping is permitted.
    pub observations: Vec<OracleObservation>,
}

/// Frozen outcome ledger comparing multiple policies on identical targets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OutcomeLedger {
    /// Must equal OUTCOME_LEDGER_SCHEMA_VERSION.
    pub schema_version: String,
    /// Human-readable benchmark identifier.
    pub benchmark_id: String,
    /// SHA-256 of the frozen task, loss, units, threshold and constraint contract.
    pub task_contract_sha256: String,
    /// SHA-256 of the frozen target/query and group-split manifest.
    pub evaluation_manifest_sha256: String,
    /// SHA-256 of the source, data-rights and artifact-provenance manifest.
    pub source_manifest_sha256: String,
    /// Optional SHA-256 of the independently verified distribution-shift manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shift_manifest_sha256: Option<String>,
    /// Task loss name; lower is better.
    pub loss_name: String,
    /// Units/normalization definition for the loss.
    pub loss_unit: String,
    /// Predeclared success rule: an evaluated loss less than or equal to this threshold.
    pub success_threshold: f64,
    /// Shared maximum number of oracle attempts per target query.
    pub query_budget: usize,
    /// Common declared unit for per-attempt cost.
    pub cost_unit: String,
    /// Arm used as the baseline for paired comparisons.
    pub baseline_arm_id: String,
    /// Minimum practically meaningful absolute improvement in success rate.
    pub minimum_success_rate_delta: f64,
    /// Minimum practically meaningful absolute improvement in normalized success AUC.
    pub minimum_success_auc_delta: f64,
    /// Minimum practically meaningful reduction in best loss at the shared final budget.
    pub minimum_best_loss_reduction: f64,
    /// Policy runs. All arms must contain exactly the same target IDs and group IDs.
    pub arms: Vec<OutcomeArm>,
}

/// A single policy run and its post-outcome observation ledger.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OutcomeArm {
    /// Unique run/arm identity.
    pub arm_id: String,
    /// Human-readable policy identifier.
    pub policy_id: String,
    /// Immutable 40-character lowercase Git revision of the policy/adapter.
    pub policy_revision: String,
    /// SHA-256 of the exact resolved run manifest/configuration.
    pub run_manifest_sha256: String,
    /// SHA-256 of the selector output/receipt artifact.
    pub selection_receipt_sha256: String,
    /// Per-target query ledgers after the run has completed.
    pub target_queries: Vec<TargetQuery>,
}

/// One point on the budgeted success-rate curve.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SuccessCurvePoint {
    /// Oracle attempt budget (one-based).
    pub query_count: usize,
    /// Fraction of all target queries that have reached success by this budget.
    pub success_rate: f64,
}

/// Per-target metrics; failures and repeated candidate queries remain in the denominator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TargetMetrics {
    /// Paired target-query ID.
    pub target_query_id: String,
    /// Declared resampling group.
    pub group_id: String,
    /// All oracle attempts, including failed and invalid calls.
    pub oracle_attempts: usize,
    /// Attempts returning a usable loss.
    pub evaluated_attempts: usize,
    /// Attempts after the first appearance of a repeated candidate ID in this target.
    pub repeated_candidate_attempts: usize,
    /// Best observed loss by budget, or null if all calls failed.
    pub best_loss_at_budget: Option<f64>,
    /// Whether any evaluated candidate reached the success threshold by budget.
    pub success_at_budget: bool,
    /// Mean of the success indicator over every budget point from 1 to query_budget.
    pub normalized_success_auc: f64,
    /// Cost of every attempted query, including failed attempts.
    pub total_cost: f64,
    /// Cumulative cost up to the first successful observation, if any.
    pub cost_to_success: Option<f64>,
}

/// Outcome summary for one declared independent group (for worst-group auditing).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GroupMetrics {
    /// Stable group identifier from the frozen evaluation/shift manifest.
    pub group_id: String,
    /// Number of target queries assigned to this group.
    pub target_query_count: usize,
    /// Number of target queries with at least one usable outcome.
    pub covered_target_queries: usize,
    /// Covered target queries divided by this group's target count.
    pub target_coverage_at_budget: f64,
    /// Number of target queries that reached success by the final budget.
    pub successful_target_queries: usize,
    /// Success rate at the final budget, with fully failed targets retained in the denominator.
    pub success_rate_at_budget: f64,
    /// Mean success indicator over every point from 1..=query_budget.
    pub normalized_success_auc: f64,
    /// All-in recorded cost divided by this group's target count.
    pub mean_cost_per_target_query: f64,
    /// Success fraction at each shared query budget for this group.
    pub success_curve: Vec<SuccessCurvePoint>,
}

/// Aggregate metrics for one policy arm.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ArmMetrics {
    /// Run/arm identity.
    pub arm_id: String,
    /// Policy identifier.
    pub policy_id: String,
    /// Number of paired target queries.
    pub target_query_count: usize,
    /// Number of caller-declared independent groups.
    pub independent_group_count: usize,
    /// Total oracle attempts, including failure statuses.
    pub oracle_attempts: usize,
    /// Oracle attempts yielding usable losses.
    pub evaluated_attempts: usize,
    /// Invalid, failed or timed-out oracle attempts.
    pub failed_attempts: usize,
    /// Evaluated attempts divided by all attempts; null when no attempts occurred.
    pub valid_query_rate: Option<f64>,
    /// Repeated candidate attempts divided by all attempts; null with zero attempts.
    pub repeated_candidate_query_rate: Option<f64>,
    /// Number of target queries with at least one usable outcome.
    pub covered_target_queries: usize,
    /// Covered targets divided by all target queries.
    pub target_coverage_at_budget: f64,
    /// Number of target queries that reached success by the budget.
    pub successful_target_queries: usize,
    /// Successful target queries divided by all target queries, including fully failed targets.
    pub success_rate_at_budget: f64,
    /// Mean success rate over budget points 1..=query_budget.
    pub normalized_success_auc: f64,
    /// Success fraction at each common query budget.
    pub success_curve: Vec<SuccessCurvePoint>,
    /// Lowest final-budget success rate across independent groups.
    pub worst_group_success_rate_at_budget: f64,
    /// Lowest normalized success AUC across independent groups.
    pub worst_group_normalized_success_auc: f64,
    /// Lowest target coverage across independent groups.
    pub worst_group_target_coverage_at_budget: f64,
    /// Per-group outcomes for shift-severity joins and worst-group audits.
    pub group_metrics: Vec<GroupMetrics>,
    /// Mean best loss conditional on target coverage.
    pub mean_best_loss_given_coverage: Option<f64>,
    /// Median best loss conditional on target coverage.
    pub median_best_loss_given_coverage: Option<f64>,
    /// Mean observed oracle cost per target query; failures contribute cost.
    pub mean_cost_per_target_query: f64,
    /// Median cost to first success among successful target queries only.
    pub median_cost_to_success: Option<f64>,
    /// Per-target results for denominator audits and independent re-analysis.
    pub targets: Vec<TargetMetrics>,
}

/// Metric for a paired comparison against the named baseline.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonMetric {
    /// Candidate arm's success-rate difference at the final budget.
    SuccessRateAtBudget,
    /// Candidate arm's normalized success-AUC difference.
    NormalizedSuccessAuc,
    /// Baseline best loss minus candidate best loss at the final budget; positive favors candidate.
    BestLossReductionAtBudget,
    /// Minimum per-group candidate-minus-baseline success-rate delta across all declared groups.
    WorstGroupSuccessRateAtBudget,
    /// Minimum per-group candidate-minus-baseline normalized success-AUC delta across all declared groups.
    WorstGroupNormalizedSuccessAuc,
}

/// Comparison interpretation against the predeclared practical-effect margin.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonDisposition {
    /// Point estimate meets the positive practical margin and the 95% interval excludes zero.
    ImprovementBeyondMargin,
    /// Point estimate meets the negative practical margin and the 95% interval excludes zero.
    RegressionBeyondMargin,
    /// The interval overlaps practical neutrality.
    Inconclusive,
    /// Fewer than five groups: report estimate, but no confidence interval.
    InsufficientIndependentGroups,
    /// At least one paired target lacks an evaluated outcome in either arm for the continuous-loss comparison.
    InsufficientTargetCoverage,
    /// At least one group has fewer than two paired targets, so a within-group interval is not estimable.
    InsufficientWithinGroupReplication,
}

/// Paired group-bootstrap result; positive deltas favour the candidate arm.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MetricComparison {
    /// Baseline arm identity.
    pub baseline_arm_id: String,
    /// Candidate arm identity.
    pub comparison_arm_id: String,
    /// Compared metric.
    pub metric: ComparisonMetric,
    /// Paired effect; for ordinary metrics this is the pooled mean difference, while for worst-group metrics it is the minimum group-level mean difference. Loss reduction is baseline minus candidate, so positive is better.
    /// With incomplete loss coverage this is descriptive over covered targets only; no interval is emitted.
    pub estimated_delta: f64,
    /// Percentile 95% interval lower bound; null if groups are insufficient.
    pub confidence_interval_lower: Option<f64>,
    /// Percentile 95% interval upper bound; null if groups are insufficient.
    pub confidence_interval_upper: Option<f64>,
    /// Confidence level.
    pub confidence_level: f64,
    /// Predeclared practical margin.
    pub practical_margin: f64,
    /// Number of declared groups represented by paired target values for this metric.
    pub independent_group_count: usize,
    /// Paired targets with a metric value in both arms (continuous loss requires complete coverage).
    pub covered_target_count: usize,
    /// Total paired targets in the evaluation contract.
    pub total_target_count: usize,
    /// Number of deterministic bootstrap replicates.
    pub bootstrap_replicates: usize,
    /// Classification under the stated margin.
    pub disposition: ComparisonDisposition,
}

/// Complete report bound to the exact outcome-ledger bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkReport {
    /// Must equal METRICS_REPORT_SCHEMA_VERSION.
    pub schema_version: String,
    /// Evaluator package version.
    pub evaluator_version: String,
    /// SHA-256 of the exact raw JSON input bytes.
    pub outcome_ledger_sha256: String,
    /// Benchmark name copied from the ledger.
    pub benchmark_id: String,
    /// Frozen task-contract digest.
    pub task_contract_sha256: String,
    /// Frozen evaluation-manifest digest.
    pub evaluation_manifest_sha256: String,
    /// Source/provenance-manifest digest.
    pub source_manifest_sha256: String,
    /// Shift-manifest digest carried forward for audit joins, if the run is shift-aware.
    pub shift_manifest_sha256: Option<String>,
    /// Loss name.
    pub loss_name: String,
    /// Loss units/definition.
    pub loss_unit: String,
    /// Declared success threshold.
    pub success_threshold: f64,
    /// Shared query budget.
    pub query_budget: usize,
    /// Declared cost unit.
    pub cost_unit: String,
    /// Baseline arm.
    pub baseline_arm_id: String,
    /// Bootstrap configuration.
    pub bootstrap_replicates: usize,
    /// Bootstrap PRNG seed.
    pub bootstrap_seed: u64,
    /// Aggregate and per-target outputs for every arm.
    pub arms: Vec<ArmMetrics>,
    /// Pairwise comparisons of all non-baseline arms against the baseline.
    pub comparisons: Vec<MetricComparison>,
    /// Explicit limitations on claim strength.
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone)]
struct TargetComputation {
    metrics: TargetMetrics,
    success_by_budget: Vec<bool>,
}

#[derive(Debug, Clone)]
struct ArmComputation {
    metrics: ArmMetrics,
    targets: BTreeMap<String, TargetComputation>,
}

/// Parse, validate and evaluate an exact post-outcome ledger.
///
/// The raw input bytes are hashed internally before parsing so callers cannot
/// accidentally supply a digest that does not correspond to the evaluated data.
/// Keep this outcome ledger separate from selector inputs.
pub fn evaluate_json_ledger(
    raw_ledger: &[u8],
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
) -> Result<BenchmarkReport, String> {
    if raw_ledger.len() > MAX_LEDGER_BYTES {
        return Err(format!(
            "ledger exceeds the {MAX_LEDGER_BYTES}-byte limit"
        ));
    }
    let ledger_sha256 = sha256_hex(raw_ledger);
    let ledger: OutcomeLedger = serde_json::from_slice(raw_ledger)
        .map_err(|error| format!("invalid outcome-ledger JSON: {error}"))?;
    validate_ledger(&ledger)?;

    if !(MIN_BOOTSTRAP_REPLICATES..=MAX_BOOTSTRAP_REPLICATES)
        .contains(&bootstrap_replicates)
    {
        return Err(format!(
            "bootstrap replicates must be between {MIN_BOOTSTRAP_REPLICATES} and {MAX_BOOTSTRAP_REPLICATES}"
        ));
    }

    let computations: Vec<ArmComputation> = ledger
        .arms
        .iter()
        .map(|arm| evaluate_arm(&ledger, arm))
        .collect();
    let baseline_index = ledger
        .arms
        .iter()
        .position(|arm| arm.arm_id == ledger.baseline_arm_id)
        .ok_or_else(|| "baseline_arm_id is not present in arms".to_string())?;
    let baseline = &computations[baseline_index];

    let mut comparisons = Vec::with_capacity((computations.len() - 1) * 5);
    for (index, candidate) in computations.iter().enumerate() {
        if index == baseline_index {
            continue;
        }
        comparisons.push(compare_arms(
            &ledger.baseline_arm_id,
            &baseline.targets,
            &candidate.metrics.arm_id,
            &candidate.targets,
            ComparisonMetric::SuccessRateAtBudget,
            ledger.minimum_success_rate_delta,
            bootstrap_replicates,
            bootstrap_seed,
        )?);
        comparisons.push(compare_arms(
            &ledger.baseline_arm_id,
            &baseline.targets,
            &candidate.metrics.arm_id,
            &candidate.targets,
            ComparisonMetric::NormalizedSuccessAuc,
            ledger.minimum_success_auc_delta,
            bootstrap_replicates,
            bootstrap_seed,
        )?);
        comparisons.push(compare_arms(
            &ledger.baseline_arm_id,
            &baseline.targets,
            &candidate.metrics.arm_id,
            &candidate.targets,
            ComparisonMetric::BestLossReductionAtBudget,
            ledger.minimum_best_loss_reduction,
            bootstrap_replicates,
            bootstrap_seed,
        )?);
        comparisons.push(compare_worst_group_arms(
            &ledger.baseline_arm_id,
            &baseline.targets,
            &candidate.metrics.arm_id,
            &candidate.targets,
            ComparisonMetric::WorstGroupSuccessRateAtBudget,
            ledger.minimum_success_rate_delta,
            bootstrap_replicates,
            bootstrap_seed,
        )?);
        comparisons.push(compare_worst_group_arms(
            &ledger.baseline_arm_id,
            &baseline.targets,
            &candidate.metrics.arm_id,
            &candidate.targets,
            ComparisonMetric::WorstGroupNormalizedSuccessAuc,
            ledger.minimum_success_auc_delta,
            bootstrap_replicates,
            bootstrap_seed,
        )?);
    }

    Ok(BenchmarkReport {
        schema_version: METRICS_REPORT_SCHEMA_VERSION.to_string(),
        evaluator_version: env!("CARGO_PKG_VERSION").to_string(),
        outcome_ledger_sha256: ledger_sha256,
        benchmark_id: ledger.benchmark_id,
        task_contract_sha256: ledger.task_contract_sha256,
        evaluation_manifest_sha256: ledger.evaluation_manifest_sha256,
        source_manifest_sha256: ledger.source_manifest_sha256,
        shift_manifest_sha256: ledger.shift_manifest_sha256,
        loss_name: ledger.loss_name,
        loss_unit: ledger.loss_unit,
        success_threshold: ledger.success_threshold,
        query_budget: ledger.query_budget,
        cost_unit: ledger.cost_unit,
        baseline_arm_id: ledger.baseline_arm_id,
        bootstrap_replicates,
        bootstrap_seed,
        arms: computations.into_iter().map(|arm| arm.metrics).collect(),
        comparisons,
        limitations: vec![
            "Success and loss semantics are only as valid as the frozen task contract; this evaluator does not validate the physics.".to_string(),
            "Group independence is asserted by the external evaluation manifest and cannot be proven from IDs alone.".to_string(),
            "An optional shift-manifest digest is carried through as a declaration; a separate artifact auditor must verify the manifest bytes and split membership.".to_string(),
            "The evaluator uses provided post-outcome observations; it does not run a selector, oracle, simulator, or experiment.".to_string(),
            "The evaluator checks observation artifact digest shape but does not dereference artifacts or recompute each loss from raw spectra/predictions; a separate scientific checker must validate those links.".to_string(),
            "A report does not establish transfer to another material family, physical regime, laboratory, or application.".to_string(),
            "The selector's structural receipt must be independently checked by receipt::verify_receipt; this metrics evaluator does not replace that check.".to_string(),
            "Pairwise 95% intervals are pointwise and unadjusted for multiple arms/metrics; use a single preregistered primary comparison for confirmatory claims or apply a declared multiplicity adjustment externally.".to_string(),
        ],
    })
}

fn validate_ledger(ledger: &OutcomeLedger) -> Result<(), String> {
    if ledger.schema_version != OUTCOME_LEDGER_SCHEMA_VERSION {
        return Err(format!(
            "schema_version must be {OUTCOME_LEDGER_SCHEMA_VERSION}"
        ));
    }
    validate_text(&ledger.benchmark_id, "benchmark_id", 256)?;
    validate_text(&ledger.loss_name, "loss_name", 128)?;
    validate_text(&ledger.loss_unit, "loss_unit", 128)?;
    validate_text(&ledger.cost_unit, "cost_unit", 128)?;
    validate_text(&ledger.baseline_arm_id, "baseline_arm_id", 128)?;

    if !is_hex_digest(&ledger.task_contract_sha256, 64)
        || !is_hex_digest(&ledger.evaluation_manifest_sha256, 64)
        || !is_hex_digest(&ledger.source_manifest_sha256, 64)
    {
        return Err(
            "task, evaluation and source manifest digests must be 64 lowercase hexadecimal characters"
                .to_string(),
        );
    }
    if ledger
        .shift_manifest_sha256
        .as_deref()
        .is_some_and(|digest| !is_hex_digest(digest, 64))
    {
        return Err(
            "shift_manifest_sha256 must be 64 lowercase hexadecimal characters when present"
                .to_string(),
        );
    }
    if !valid_nonnegative(ledger.success_threshold) {
        return Err("success_threshold must be finite and within the allowed non-negative range".to_string());
    }
    if ledger.query_budget == 0 || ledger.query_budget > MAX_QUERY_BUDGET {
        return Err(format!("query_budget must be in 1..={MAX_QUERY_BUDGET}"));
    }
    for (name, margin) in [
        ("minimum_success_rate_delta", ledger.minimum_success_rate_delta),
        ("minimum_success_auc_delta", ledger.minimum_success_auc_delta),
    ] {
        if !margin.is_finite() || !(0.0..=1.0).contains(&margin) {
            return Err(format!("{name} must be finite and within [0, 1]"));
        }
    }
    if !valid_nonnegative(ledger.minimum_best_loss_reduction) {
        return Err("minimum_best_loss_reduction must be finite and within the allowed non-negative range".to_string());
    }
    if !(2..=MAX_ARMS).contains(&ledger.arms.len()) {
        return Err(format!("arms must contain between 2 and {MAX_ARMS} policy runs"));
    }

    let mut seen_arm_ids = BTreeSet::new();
    let mut expected_targets: Option<BTreeMap<String, String>> = None;
    let mut baseline_present = false;

    for arm in &ledger.arms {
        validate_text(&arm.arm_id, "arm_id", 128)?;
        validate_text(&arm.policy_id, "policy_id", 128)?;
        if !seen_arm_ids.insert(arm.arm_id.as_str()) {
            return Err(format!("duplicate arm_id {:?}", arm.arm_id));
        }
        baseline_present |= arm.arm_id == ledger.baseline_arm_id;

        if !is_hex_digest(&arm.policy_revision, 40)
            || !is_hex_digest(&arm.run_manifest_sha256, 64)
            || !is_hex_digest(&arm.selection_receipt_sha256, 64)
        {
            return Err(format!(
                "arm {:?} has invalid policy revision or run/receipt digest",
                arm.arm_id
            ));
        }
        if arm.target_queries.is_empty() || arm.target_queries.len() > MAX_TARGET_QUERIES {
            return Err(format!(
                "arm {:?} target_queries must contain 1..={MAX_TARGET_QUERIES} entries",
                arm.arm_id
            ));
        }

        let mut target_map = BTreeMap::new();
        let mut observation_total = 0usize;
        for target in &arm.target_queries {
            validate_text(&target.target_query_id, "target_query_id", 256)?;
            validate_text(&target.group_id, "group_id", 256)?;
            if target_map
                .insert(target.target_query_id.clone(), target.group_id.clone())
                .is_some()
            {
                return Err(format!(
                    "arm {:?} repeats target_query_id {:?}",
                    arm.arm_id, target.target_query_id
                ));
            }
            if target.observations.len() > ledger.query_budget {
                return Err(format!(
                    "target query {:?} exceeds the shared query budget",
                    target.target_query_id
                ));
            }
            observation_total = observation_total
                .checked_add(target.observations.len())
                .ok_or_else(|| "total observation count overflow".to_string())?;

            for (index, observation) in target.observations.iter().enumerate() {
                validate_text(&observation.candidate_id, "candidate_id", 256)?;
                if observation.query_rank != index + 1 {
                    return Err(format!(
                        "target query {:?} query_rank must be contiguous from 1",
                        target.target_query_id
                    ));
                }
                if !is_hex_digest(&observation.observation_artifact_sha256, 64) {
                    return Err(format!(
                        "target query {:?} has invalid observation artifact digest",
                        target.target_query_id
                    ));
                }
                if !valid_nonnegative(observation.cost) {
                    return Err(format!(
                        "target query {:?} has invalid cost",
                        target.target_query_id
                    ));
                }

                match (observation.status, observation.loss) {
                    (ObservationStatus::Evaluated, Some(loss))
                        if valid_nonnegative(loss) => {}
                    (ObservationStatus::Evaluated, _) => {
                        return Err(format!(
                            "evaluated query in {:?} must have a finite non-negative loss",
                            target.target_query_id
                        ));
                    }
                    (_, None) => {}
                    (_, Some(_)) => {
                        return Err(format!(
                            "non-evaluated query in {:?} must not contain a loss",
                            target.target_query_id
                        ));
                    }
                }
            }
        }
        if observation_total > MAX_TOTAL_OBSERVATIONS_PER_ARM {
            return Err(format!(
                "arm {:?} exceeds total observation limit {MAX_TOTAL_OBSERVATIONS_PER_ARM}",
                arm.arm_id
            ));
        }

        if let Some(expected) = &expected_targets {
            if &target_map != expected {
                return Err(format!(
                    "arm {:?} must use the exact same target_query_id/group_id set as every other arm",
                    arm.arm_id
                ));
            }
        } else {
            expected_targets = Some(target_map);
        }
    }

    if !baseline_present {
        return Err("baseline_arm_id must identify one of the provided arms".to_string());
    }
    Ok(())
}

fn validate_text(value: &str, field: &str, max_bytes: usize) -> Result<(), String> {
    if value.trim().is_empty()
        || value.len() > max_bytes
        || value.chars().any(char::is_control)
    {
        return Err(format!(
            "{field} must be nonempty, bounded, and contain no control characters"
        ));
    }
    Ok(())
}

fn valid_nonnegative(value: f64) -> bool {
    value.is_finite() && (0.0..=MAX_NUMERIC_VALUE).contains(&value)
}

fn is_hex_digest(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn evaluate_arm(ledger: &OutcomeLedger, arm: &OutcomeArm) -> ArmComputation {
    let targets: BTreeMap<String, TargetComputation> = arm
        .target_queries
        .iter()
        .map(|target| {
            (
                target.target_query_id.clone(),
                evaluate_target(ledger, target),
            )
        })
        .collect();

    let target_count = targets.len();
    let mut success_curve = Vec::with_capacity(ledger.query_budget);
    for query_count in 1..=ledger.query_budget {
        let successful = targets
            .values()
            .filter(|target| target.success_by_budget[query_count - 1])
            .count();
        success_curve.push(SuccessCurvePoint {
            query_count,
            success_rate: successful as f64 / target_count as f64,
        });
    }

    let oracle_attempts: usize = targets
        .values()
        .map(|target| target.metrics.oracle_attempts)
        .sum();
    let evaluated_attempts: usize = targets
        .values()
        .map(|target| target.metrics.evaluated_attempts)
        .sum();
    let repeated_candidate_attempts: usize = targets
        .values()
        .map(|target| target.metrics.repeated_candidate_attempts)
        .sum();
    let total_cost: f64 = targets
        .values()
        .map(|target| target.metrics.total_cost)
        .sum();
    let best_losses: Vec<f64> = targets
        .values()
        .filter_map(|target| target.metrics.best_loss_at_budget)
        .collect();
    let costs_to_success: Vec<f64> = targets
        .values()
        .filter_map(|target| target.metrics.cost_to_success)
        .collect();
    let successful_targets = targets
        .values()
        .filter(|target| target.metrics.success_at_budget)
        .count();
    let covered_targets = best_losses.len();
    let normalized_success_auc = success_curve
        .iter()
        .map(|point| point.success_rate)
        .sum::<f64>()
        / ledger.query_budget as f64;
    let group_metrics = evaluate_group_metrics(&targets, ledger.query_budget);
    let worst_group_success_rate_at_budget = group_metrics
        .iter()
        .map(|group| group.success_rate_at_budget)
        .fold(f64::INFINITY, f64::min);
    let worst_group_normalized_success_auc = group_metrics
        .iter()
        .map(|group| group.normalized_success_auc)
        .fold(f64::INFINITY, f64::min);
    let worst_group_target_coverage_at_budget = group_metrics
        .iter()
        .map(|group| group.target_coverage_at_budget)
        .fold(f64::INFINITY, f64::min);

    let metrics = ArmMetrics {
        arm_id: arm.arm_id.clone(),
        policy_id: arm.policy_id.clone(),
        target_query_count: target_count,
        independent_group_count: targets
            .values()
            .map(|target| target.metrics.group_id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        oracle_attempts,
        evaluated_attempts,
        failed_attempts: oracle_attempts - evaluated_attempts,
        valid_query_rate: (oracle_attempts > 0)
            .then_some(evaluated_attempts as f64 / oracle_attempts as f64),
        repeated_candidate_query_rate: (oracle_attempts > 0)
            .then_some(repeated_candidate_attempts as f64 / oracle_attempts as f64),
        covered_target_queries: covered_targets,
        target_coverage_at_budget: covered_targets as f64 / target_count as f64,
        successful_target_queries: successful_targets,
        success_rate_at_budget: successful_targets as f64 / target_count as f64,
        normalized_success_auc,
        success_curve,
        worst_group_success_rate_at_budget,
        worst_group_normalized_success_auc,
        worst_group_target_coverage_at_budget,
        group_metrics,
        mean_best_loss_given_coverage: mean(&best_losses),
        median_best_loss_given_coverage: median(best_losses),
        mean_cost_per_target_query: total_cost / target_count as f64,
        median_cost_to_success: median(costs_to_success),
        targets: targets.values().map(|target| target.metrics.clone()).collect(),
    };

    ArmComputation { metrics, targets }
}

fn evaluate_group_metrics(
    targets: &BTreeMap<String, TargetComputation>,
    query_budget: usize,
) -> Vec<GroupMetrics> {
    let mut targets_by_group: BTreeMap<String, Vec<&TargetComputation>> = BTreeMap::new();
    for target in targets.values() {
        targets_by_group
            .entry(target.metrics.group_id.clone())
            .or_default()
            .push(target);
    }

    targets_by_group
        .into_iter()
        .map(|(group_id, group_targets)| {
            let target_count = group_targets.len();
            let covered_target_queries = group_targets
                .iter()
                .filter(|target| target.metrics.best_loss_at_budget.is_some())
                .count();
            let successful_target_queries = group_targets
                .iter()
                .filter(|target| target.metrics.success_at_budget)
                .count();
            let total_cost = group_targets
                .iter()
                .map(|target| target.metrics.total_cost)
                .sum::<f64>();
            let success_curve = (1..=query_budget)
                .map(|query_count| {
                    let successful = group_targets
                        .iter()
                        .filter(|target| target.success_by_budget[query_count - 1])
                        .count();
                    SuccessCurvePoint {
                        query_count,
                        success_rate: successful as f64 / target_count as f64,
                    }
                })
                .collect::<Vec<_>>();
            let normalized_success_auc = success_curve
                .iter()
                .map(|point| point.success_rate)
                .sum::<f64>()
                / query_budget as f64;

            GroupMetrics {
                group_id,
                target_query_count: target_count,
                covered_target_queries,
                target_coverage_at_budget: covered_target_queries as f64 / target_count as f64,
                successful_target_queries,
                success_rate_at_budget: successful_target_queries as f64 / target_count as f64,
                normalized_success_auc,
                mean_cost_per_target_query: total_cost / target_count as f64,
                success_curve,
            }
        })
        .collect()
}

fn evaluate_target(ledger: &OutcomeLedger, target: &TargetQuery) -> TargetComputation {
    let mut best_loss: Option<f64> = None;
    let mut cost_to_success: Option<f64> = None;
    let mut total_cost = 0.0;
    let mut evaluated_attempts = 0usize;
    let mut repeated_candidate_attempts = 0usize;
    let mut seen_candidates = HashSet::new();
    let mut success_by_budget = Vec::with_capacity(ledger.query_budget);

    for query_count in 1..=ledger.query_budget {
        if let Some(observation) = target.observations.get(query_count - 1) {
            total_cost += observation.cost;
            if !seen_candidates.insert(observation.candidate_id.as_str()) {
                repeated_candidate_attempts += 1;
            }
            if let Some(loss) = observation.loss {
                evaluated_attempts += 1;
                best_loss = Some(best_loss.map_or(loss, |previous| previous.min(loss)));
            }
            if cost_to_success.is_none()
                && best_loss.is_some_and(|loss| loss <= ledger.success_threshold)
            {
                cost_to_success = Some(total_cost);
            }
        }
        success_by_budget.push(best_loss.is_some_and(|loss| loss <= ledger.success_threshold));
    }

    let successful_budget_points = success_by_budget
        .iter()
        .filter(|success| **success)
        .count();
    let metrics = TargetMetrics {
        target_query_id: target.target_query_id.clone(),
        group_id: target.group_id.clone(),
        oracle_attempts: target.observations.len(),
        evaluated_attempts,
        repeated_candidate_attempts,
        best_loss_at_budget: best_loss,
        success_at_budget: best_loss.is_some_and(|loss| loss <= ledger.success_threshold),
        normalized_success_auc: successful_budget_points as f64 / ledger.query_budget as f64,
        total_cost,
        cost_to_success,
    };

    TargetComputation {
        metrics,
        success_by_budget,
    }
}

fn compare_worst_group_arms(
    baseline_arm_id: &str,
    baseline: &BTreeMap<String, TargetComputation>,
    candidate_arm_id: &str,
    candidate: &BTreeMap<String, TargetComputation>,
    metric: ComparisonMetric,
    practical_margin: f64,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
) -> Result<MetricComparison, String> {
    if !matches!(
        metric,
        ComparisonMetric::WorstGroupSuccessRateAtBudget
            | ComparisonMetric::WorstGroupNormalizedSuccessAuc
    ) {
        return Err("compare_worst_group_arms requires a worst-group metric".to_string());
    }

    let total_target_count = baseline.len();
    let mut deltas_by_group: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (target_id, baseline_target) in baseline {
        let candidate_target = candidate
            .get(target_id)
            .ok_or_else(|| format!("comparison arm lacks target {target_id:?}"))?;
        if candidate_target.metrics.group_id != baseline_target.metrics.group_id {
            return Err(format!(
                "target {target_id:?} belongs to different groups across comparison arms"
            ));
        }
        let delta = match metric {
            ComparisonMetric::WorstGroupSuccessRateAtBudget => {
                (if candidate_target.metrics.success_at_budget { 1.0 } else { 0.0 })
                    - (if baseline_target.metrics.success_at_budget { 1.0 } else { 0.0 })
            }
            ComparisonMetric::WorstGroupNormalizedSuccessAuc => {
                candidate_target.metrics.normalized_success_auc
                    - baseline_target.metrics.normalized_success_auc
            }
            _ => unreachable!("metric variant checked at function entry"),
        };
        deltas_by_group
            .entry(baseline_target.metrics.group_id.clone())
            .or_default()
            .push(delta);
    }

    if deltas_by_group.is_empty() {
        return Err("worst-group comparison requires at least one paired target".to_string());
    }
    let covered_target_count = deltas_by_group.values().map(Vec::len).sum::<usize>();
    let group_means = deltas_by_group
        .iter()
        .map(|(group_id, deltas)| {
            (
                group_id.clone(),
                deltas.iter().sum::<f64>() / deltas.len() as f64,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let estimated_delta = group_means
        .values()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let group_count = deltas_by_group.len();

    let no_interval = |disposition| MetricComparison {
        baseline_arm_id: baseline_arm_id.to_string(),
        comparison_arm_id: candidate_arm_id.to_string(),
        metric,
        estimated_delta,
        confidence_interval_lower: None,
        confidence_interval_upper: None,
        confidence_level: 0.95,
        practical_margin,
        independent_group_count: group_count,
        covered_target_count,
        total_target_count,
        bootstrap_replicates,
        disposition,
    };

    if group_count < MIN_INDEPENDENT_GROUPS_FOR_INTERVAL {
        return Ok(no_interval(
            ComparisonDisposition::InsufficientIndependentGroups,
        ));
    }
    if deltas_by_group.values().any(|deltas| deltas.len() < 2) {
        return Ok(no_interval(
            ComparisonDisposition::InsufficientWithinGroupReplication,
        ));
    }

    // The groups are the predeclared worst-group set, not a random sample to be
    // averaged away. Resample paired targets within every group, build a
    // Bonferroni-adjusted percentile interval for each group, and bound the
    // minimum of group effects using the minimum of simultaneous group bounds.
    let tail_probability = 0.025 / group_count as f64;
    let metric_salt = match metric {
        ComparisonMetric::WorstGroupSuccessRateAtBudget => 0x5753_5543_4300_0004,
        ComparisonMetric::WorstGroupNormalizedSuccessAuc => 0x5753_4155_4300_0005,
        _ => unreachable!("metric variant checked at function entry"),
    };
    let mut group_lower_bounds = Vec::with_capacity(group_count);
    let mut group_upper_bounds = Vec::with_capacity(group_count);

    for (group_id, deltas) in &deltas_by_group {
        let mut random = SplitMix64::new(
            bootstrap_seed
                ^ stable_hash64(candidate_arm_id.as_bytes())
                ^ stable_hash64(group_id.as_bytes())
                ^ metric_salt,
        );
        let mut bootstrap_means = Vec::with_capacity(bootstrap_replicates);
        for _ in 0..bootstrap_replicates {
            let mut sum = 0.0;
            for _ in 0..deltas.len() {
                sum += deltas[random.index(deltas.len())];
            }
            bootstrap_means.push(sum / deltas.len() as f64);
        }
        bootstrap_means.sort_by(f64::total_cmp);
        let last = bootstrap_means.len() - 1;
        let lower_index = ((last as f64) * tail_probability).floor() as usize;
        let upper_index = ((last as f64) * (1.0 - tail_probability)).ceil() as usize;
        group_lower_bounds.push(bootstrap_means[lower_index]);
        group_upper_bounds.push(bootstrap_means[upper_index.min(last)]);
    }

    let lower = group_lower_bounds
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let upper = group_upper_bounds
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let disposition = if estimated_delta >= practical_margin && lower > 0.0 {
        ComparisonDisposition::ImprovementBeyondMargin
    } else if estimated_delta <= -practical_margin && upper < 0.0 {
        ComparisonDisposition::RegressionBeyondMargin
    } else {
        ComparisonDisposition::Inconclusive
    };

    Ok(MetricComparison {
        baseline_arm_id: baseline_arm_id.to_string(),
        comparison_arm_id: candidate_arm_id.to_string(),
        metric,
        estimated_delta,
        confidence_interval_lower: Some(lower),
        confidence_interval_upper: Some(upper),
        confidence_level: 0.95,
        practical_margin,
        independent_group_count: group_count,
        covered_target_count,
        total_target_count,
        bootstrap_replicates,
        disposition,
    })
}

fn compare_arms(
    baseline_arm_id: &str,
    baseline: &BTreeMap<String, TargetComputation>,
    candidate_arm_id: &str,
    candidate: &BTreeMap<String, TargetComputation>,
    metric: ComparisonMetric,
    practical_margin: f64,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
) -> Result<MetricComparison, String> {
    let total_target_count = baseline.len();
    let mut covered_target_count = 0usize;
    let mut deltas_by_group: BTreeMap<String, Vec<f64>> = BTreeMap::new();

    for (target_id, baseline_target) in baseline {
        let candidate_target = candidate
            .get(target_id)
            .ok_or_else(|| format!("comparison arm lacks target {target_id:?}"))?;
        let delta = match metric {
            ComparisonMetric::SuccessRateAtBudget | ComparisonMetric::NormalizedSuccessAuc => {
                let candidate_value = target_metric(&candidate_target.metrics, metric)
                    .ok_or_else(|| "paired target metric was unavailable".to_string())?;
                let baseline_value = target_metric(&baseline_target.metrics, metric)
                    .ok_or_else(|| "paired baseline metric was unavailable".to_string())?;
                Some(candidate_value - baseline_value)
            }
            ComparisonMetric::BestLossReductionAtBudget => {
                match (
                    baseline_target.metrics.best_loss_at_budget,
                    candidate_target.metrics.best_loss_at_budget,
                ) {
                    (Some(baseline_loss), Some(candidate_loss)) => {
                        Some(baseline_loss - candidate_loss)
                    }
                    _ => None,
                }
            }
            ComparisonMetric::WorstGroupSuccessRateAtBudget
            | ComparisonMetric::WorstGroupNormalizedSuccessAuc => {
                return Err("worst-group comparisons must use compare_worst_group_arms".to_string());
            }
        };

        if let Some(delta) = delta {
            deltas_by_group
                .entry(baseline_target.metrics.group_id.clone())
                .or_default()
                .push(delta);
            covered_target_count += 1;
        }
    }

    let group_deltas: Vec<Vec<f64>> = deltas_by_group.into_values().collect();
    let estimated_delta = if covered_target_count == 0 {
        0.0
    } else {
        group_deltas.iter().flatten().sum::<f64>() / covered_target_count as f64
    };

    // Do not silently drop failed-all-attempts targets from a continuous-loss
    // comparison. Report a descriptive delta over the covered subset, but do not
    // compute an interval or classify an improvement unless coverage is complete.
    if metric == ComparisonMetric::BestLossReductionAtBudget
        && covered_target_count != total_target_count
    {
        return Ok(MetricComparison {
            baseline_arm_id: baseline_arm_id.to_string(),
            comparison_arm_id: candidate_arm_id.to_string(),
            metric,
            estimated_delta,
            confidence_interval_lower: None,
            confidence_interval_upper: None,
            confidence_level: 0.95,
            practical_margin,
            independent_group_count: group_deltas.len(),
            covered_target_count,
            total_target_count,
            bootstrap_replicates,
            disposition: ComparisonDisposition::InsufficientTargetCoverage,
        });
    }

    if group_deltas.len() < MIN_INDEPENDENT_GROUPS_FOR_INTERVAL {
        return Ok(MetricComparison {
            baseline_arm_id: baseline_arm_id.to_string(),
            comparison_arm_id: candidate_arm_id.to_string(),
            metric,
            estimated_delta,
            confidence_interval_lower: None,
            confidence_interval_upper: None,
            confidence_level: 0.95,
            practical_margin,
            independent_group_count: group_deltas.len(),
            covered_target_count,
            total_target_count,
            bootstrap_replicates,
            disposition: ComparisonDisposition::InsufficientIndependentGroups,
        });
    }

    let metric_salt = match metric {
        ComparisonMetric::SuccessRateAtBudget => 0x5355_4343_4553_5301,
        ComparisonMetric::NormalizedSuccessAuc => 0x4155_4300_0000_0002,
        ComparisonMetric::BestLossReductionAtBudget => 0x4C4F_5353_5245_4403,
        ComparisonMetric::WorstGroupSuccessRateAtBudget => 0x5753_5543_4300_0004,
        ComparisonMetric::WorstGroupNormalizedSuccessAuc => 0x5753_4155_4300_0005,
    };
    let mut random = SplitMix64::new(
        bootstrap_seed ^ stable_hash64(candidate_arm_id.as_bytes()) ^ metric_salt,
    );
    let mut sampled_means = Vec::with_capacity(bootstrap_replicates);
    for _ in 0..bootstrap_replicates {
        let mut sum = 0.0;
        let mut sampled_target_count = 0usize;
        for _ in 0..group_deltas.len() {
            let sampled_group = &group_deltas[random.index(group_deltas.len())];
            sum += sampled_group.iter().sum::<f64>();
            sampled_target_count += sampled_group.len();
        }
        sampled_means.push(sum / sampled_target_count as f64);
    }
    sampled_means.sort_by(f64::total_cmp);
    let last = sampled_means.len() - 1;
    let lower_index = ((last as f64) * 0.025).floor() as usize;
    let upper_index = ((last as f64) * 0.975).ceil() as usize;
    let lower = sampled_means[lower_index];
    let upper = sampled_means[upper_index];

    let disposition = if estimated_delta >= practical_margin && lower > 0.0 {
        ComparisonDisposition::ImprovementBeyondMargin
    } else if estimated_delta <= -practical_margin && upper < 0.0 {
        ComparisonDisposition::RegressionBeyondMargin
    } else {
        ComparisonDisposition::Inconclusive
    };

    Ok(MetricComparison {
        baseline_arm_id: baseline_arm_id.to_string(),
        comparison_arm_id: candidate_arm_id.to_string(),
        metric,
        estimated_delta,
        confidence_interval_lower: Some(lower),
        confidence_interval_upper: Some(upper),
        confidence_level: 0.95,
        practical_margin,
        independent_group_count: group_deltas.len(),
        covered_target_count,
        total_target_count,
        bootstrap_replicates,
        disposition,
    })
}

fn target_metric(target: &TargetMetrics, metric: ComparisonMetric) -> Option<f64> {
    match metric {
        ComparisonMetric::SuccessRateAtBudget => Some(if target.success_at_budget { 1.0 } else { 0.0 }),
        ComparisonMetric::NormalizedSuccessAuc => Some(target.normalized_success_auc),
        ComparisonMetric::BestLossReductionAtBudget
        | ComparisonMetric::WorstGroupSuccessRateAtBudget
        | ComparisonMetric::WorstGroupNormalizedSuccessAuc => None,
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn stable_hash64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper_bound: usize) -> usize {
        debug_assert!(upper_bound > 0);
        let bound = upper_bound as u64;
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
    use serde_json::{json, Value};

    fn hex(character: char, length: usize) -> String {
        character.to_string().repeat(length)
    }

    fn observation(
        rank: usize,
        candidate_id: &str,
        loss: Option<f64>,
        status: ObservationStatus,
    ) -> OracleObservation {
        OracleObservation {
            query_rank: rank,
            candidate_id: candidate_id.to_string(),
            status,
            observation_artifact_sha256: hex('9', 64),
            loss,
            cost: 1.0,
        }
    }

    fn target(id: &str, group: &str, observations: Vec<OracleObservation>) -> TargetQuery {
        TargetQuery {
            target_query_id: id.to_string(),
            group_id: group.to_string(),
            observations,
        }
    }

    fn arm(id: &str, policy: &str, target_queries: Vec<TargetQuery>) -> OutcomeArm {
        OutcomeArm {
            arm_id: id.to_string(),
            policy_id: policy.to_string(),
            policy_revision: hex('a', 40),
            run_manifest_sha256: hex('b', 64),
            selection_receipt_sha256: hex('c', 64),
            target_queries,
        }
    }

    fn fixture() -> OutcomeLedger {
        OutcomeLedger {
            schema_version: OUTCOME_LEDGER_SCHEMA_VERSION.to_string(),
            benchmark_id: "synthetic-unit-fixture".to_string(),
            task_contract_sha256: hex('d', 64),
            evaluation_manifest_sha256: hex('e', 64),
            source_manifest_sha256: hex('f', 64),
            shift_manifest_sha256: None,
            loss_name: "synthetic_loss".to_string(),
            loss_unit: "dimensionless".to_string(),
            success_threshold: 0.01,
            query_budget: 3,
            cost_unit: "oracle_call".to_string(),
            baseline_arm_id: "baseline".to_string(),
            minimum_success_rate_delta: 0.05,
            minimum_success_auc_delta: 0.05,
            minimum_best_loss_reduction: 0.05,
            arms: vec![
                arm(
                    "baseline",
                    "random_seeded",
                    vec![
                        target("t1", "g1", vec![
                            observation(1, "a", Some(0.03), ObservationStatus::Evaluated),
                            observation(2, "b", Some(0.02), ObservationStatus::Evaluated),
                            observation(3, "c", Some(0.02), ObservationStatus::Evaluated),
                        ]),
                        target("t2", "g2", vec![
                            observation(1, "d", Some(0.05), ObservationStatus::Evaluated),
                            observation(2, "e", Some(0.008), ObservationStatus::Evaluated),
                        ]),
                        target("t3", "g3", vec![
                            observation(1, "f", None, ObservationStatus::OracleFailed),
                            observation(2, "g", Some(0.05), ObservationStatus::Evaluated),
                        ]),
                        target("t4", "g4", vec![
                            observation(1, "h", Some(0.02), ObservationStatus::Evaluated),
                            observation(2, "i", Some(0.03), ObservationStatus::Evaluated),
                            observation(3, "j", Some(0.009), ObservationStatus::Evaluated),
                        ]),
                        target("t5", "g5", vec![
                            observation(1, "k", Some(0.03), ObservationStatus::Evaluated),
                        ]),
                    ],
                ),
                arm(
                    "candidate",
                    "farthest_first",
                    vec![
                        target("t1", "g1", vec![
                            observation(1, "a1", Some(0.01), ObservationStatus::Evaluated),
                        ]),
                        target("t2", "g2", vec![
                            observation(1, "b1", Some(0.011), ObservationStatus::Evaluated),
                            observation(2, "b2", Some(0.009), ObservationStatus::Evaluated),
                        ]),
                        target("t3", "g3", vec![
                            observation(1, "c1", Some(0.004), ObservationStatus::Evaluated),
                        ]),
                        target("t4", "g4", vec![
                            observation(1, "d1", Some(0.018), ObservationStatus::Evaluated),
                            observation(2, "d1", Some(0.018), ObservationStatus::Evaluated),
                            observation(3, "d2", Some(0.019), ObservationStatus::Evaluated),
                        ]),
                        target("t5", "g5", vec![
                            observation(1, "e1", Some(0.02), ObservationStatus::Evaluated),
                        ]),
                    ],
                ),
            ],
        }
    }

    fn evaluate_fixture(ledger: &OutcomeLedger) -> BenchmarkReport {
        let bytes = serde_json::to_vec(ledger).expect("serialize fixture");
        evaluate_json_ledger(&bytes, 200, 17).expect("valid synthetic ledger")
    }

    #[test]
    fn computes_budget_success_coverage_failures_and_cost() {
        let report = evaluate_fixture(&fixture());
        let baseline = report.arms.iter().find(|arm| arm.arm_id == "baseline").unwrap();
        let candidate = report.arms.iter().find(|arm| arm.arm_id == "candidate").unwrap();

        assert!((baseline.success_rate_at_budget - 0.4).abs() < 1e-12);
        assert!((candidate.success_rate_at_budget - 0.6).abs() < 1e-12);
        assert_eq!(baseline.oracle_attempts, 11);
        assert_eq!(baseline.failed_attempts, 1);
        assert!((baseline.valid_query_rate.unwrap() - (10.0 / 11.0)).abs() < 1e-12);
        assert_eq!(
            candidate.targets.iter().map(|target| target.repeated_candidate_attempts).sum::<usize>(),
            1
        );
        assert!((candidate.target_coverage_at_budget - 1.0).abs() < 1e-12);
        assert!((baseline.mean_cost_per_target_query - 2.2).abs() < 1e-12);
    }

    #[test]
    fn carries_shift_manifest_digest_into_metrics_report() {
        let mut ledger = fixture();
        ledger.shift_manifest_sha256 = Some(hex('8', 64));

        let report = evaluate_fixture(&ledger);
        assert_eq!(
            report.shift_manifest_sha256.as_deref(),
            ledger.shift_manifest_sha256.as_deref()
        );
    }

    #[test]
    fn rejects_malformed_shift_manifest_digest() {
        let mut ledger = fixture();
        ledger.shift_manifest_sha256 = Some("not-a-sha256".to_string());
        let bytes = serde_json::to_vec(&ledger).expect("serialize malformed fixture");

        let error = evaluate_json_ledger(&bytes, 200, 17).expect_err("invalid shift digest");
        assert!(error.contains("shift_manifest_sha256"));
    }

    #[test]
    fn worst_group_comparison_rejects_hidden_pooled_win() {
        let mut ledger = fixture();
        for arm in &mut ledger.arms {
            let mut replicate_targets = Vec::new();
            for target in &arm.target_queries {
                let mut replicate = target.clone();
                replicate.target_query_id = format!("{}-replicate", target.target_query_id);
                let success = match (arm.arm_id.as_str(), target.target_query_id.as_str()) {
                    ("baseline", "t4") => true,
                    ("candidate", "t2" | "t3" | "t5") => true,
                    _ => false,
                };
                let loss = if success { 0.005 } else { 0.02 };
                let candidate_id = format!("{}-replicate-candidate", target.target_query_id);
                replicate.observations = vec![observation(
                    1,
                    &candidate_id,
                    Some(loss),
                    ObservationStatus::Evaluated,
                )];
                replicate_targets.push(replicate);
            }
            arm.target_queries.extend(replicate_targets);
        }

        let report = evaluate_fixture(&ledger);
        let pooled = report
            .comparisons
            .iter()
            .find(|comparison| comparison.metric == ComparisonMetric::SuccessRateAtBudget)
            .expect("pooled success comparison");
        let worst = report
            .comparisons
            .iter()
            .find(|comparison| {
                comparison.metric == ComparisonMetric::WorstGroupSuccessRateAtBudget
            })
            .expect("worst-group success comparison");

        assert!(pooled.estimated_delta > 0.0, "pooled improvement should be positive");
        assert!(worst.estimated_delta < 0.0, "weakest group should regress");
        assert_eq!(worst.independent_group_count, 5);
        assert_eq!(worst.covered_target_count, 10);
        assert_eq!(worst.disposition, ComparisonDisposition::RegressionBeyondMargin);
        assert!(
            worst.confidence_interval_upper.unwrap() < 0.0,
            "simultaneous worst-group upper confidence bound must stay below zero"
        );
    }

    #[test]
    fn reports_per_group_curves_and_worst_group_regression() {
        let mut ledger = fixture();
        for arm in &mut ledger.arms {
            for target in &mut arm.target_queries {
                target.group_id = match target.target_query_id.as_str() {
                    "t1" | "t2" | "t3" => "strong_group".to_string(),
                    _ => "weak_group".to_string(),
                };
            }
        }

        let report = evaluate_fixture(&ledger);
        let baseline = report
            .arms
            .iter()
            .find(|arm| arm.arm_id == "baseline")
            .expect("baseline arm");
        let candidate = report
            .arms
            .iter()
            .find(|arm| arm.arm_id == "candidate")
            .expect("candidate arm");

        assert_eq!(candidate.group_metrics.len(), 2);
        let weak_group = candidate
            .group_metrics
            .iter()
            .find(|group| group.group_id == "weak_group")
            .expect("weak group metrics");
        assert_eq!(weak_group.target_query_count, 2);
        assert_eq!(weak_group.successful_target_queries, 0);
        assert_eq!(weak_group.success_rate_at_budget, 0.0);
        assert_eq!(weak_group.success_curve.len(), ledger.query_budget);
        assert_eq!(candidate.worst_group_success_rate_at_budget, 0.0);
        assert!(baseline.worst_group_success_rate_at_budget > 0.0);
        assert!(
            candidate.worst_group_success_rate_at_budget
                < baseline.worst_group_success_rate_at_budget
        );
        assert_eq!(
            candidate.worst_group_success_rate_at_budget,
            candidate
                .group_metrics
                .iter()
                .map(|group| group.success_rate_at_budget)
                .fold(f64::INFINITY, f64::min)
        );
    }

    #[test]
    fn computes_normalized_success_auc_from_full_budget_curve() {
        let report = evaluate_fixture(&fixture());
        let baseline = report.arms.iter().find(|arm| arm.arm_id == "baseline").unwrap();
        let candidate = report.arms.iter().find(|arm| arm.arm_id == "candidate").unwrap();

        assert!((baseline.normalized_success_auc - 0.2).abs() < 1e-12);
        assert!((candidate.normalized_success_auc - (8.0 / 15.0)).abs() < 1e-12);
        assert_eq!(baseline.success_curve.len(), 3);
    }

    #[test]
    fn exact_input_bytes_are_bound_to_report_digest() {
        let ledger = fixture();
        let bytes = serde_json::to_vec(&ledger).expect("serialize fixture");
        let report = evaluate_json_ledger(&bytes, 200, 17).expect("evaluate");
        assert_eq!(report.outcome_ledger_sha256, sha256_hex(&bytes));
        let mut changed = bytes.clone();
        changed.push(b' ');
        let changed_report = evaluate_json_ledger(&changed, 200, 17).expect("valid JSON with whitespace");
        assert_ne!(report.outcome_ledger_sha256, changed_report.outcome_ledger_sha256);
    }

    #[test]
    fn paired_group_bootstrap_is_deterministic() {
        let bytes = serde_json::to_vec(&fixture()).expect("serialize fixture");
        let first = evaluate_json_ledger(&bytes, 500, 17).expect("first evaluation");
        let second = evaluate_json_ledger(&bytes, 500, 17).expect("second evaluation");
        assert_eq!(first, second);

        let comparison = first
            .comparisons
            .iter()
            .find(|comparison| comparison.metric == ComparisonMetric::SuccessRateAtBudget)
            .unwrap();
        assert!((comparison.estimated_delta - 0.2).abs() < 1e-12);
        assert_eq!(comparison.independent_group_count, 5);
        assert!(comparison.confidence_interval_lower.is_some());
        assert!(comparison.confidence_interval_upper.is_some());
    }

    #[test]
    fn refuses_confidence_interval_with_too_few_declared_groups() {
        let mut ledger = fixture();
        for arm in &mut ledger.arms {
            for target in &mut arm.target_queries {
                target.group_id = "one-group".to_string();
            }
        }
        let report = evaluate_fixture(&ledger);
        assert!(report.comparisons.iter().all(|comparison| {
            comparison.disposition == ComparisonDisposition::InsufficientIndependentGroups
                && comparison.confidence_interval_lower.is_none()
                && comparison.confidence_interval_upper.is_none()
        }));
    }

    #[test]
    fn classifies_improvement_only_when_margin_is_met_and_interval_excludes_zero() {
        let mut ledger = fixture();
        for target in &mut ledger.arms[0].target_queries {
            for observation in &mut target.observations {
                if observation.status == ObservationStatus::Evaluated {
                    observation.loss = Some(0.5);
                }
            }
        }
        for target in &mut ledger.arms[1].target_queries {
            for observation in &mut target.observations {
                if observation.status == ObservationStatus::Evaluated {
                    observation.loss = Some(0.001);
                }
            }
        }

        let report = evaluate_fixture(&ledger);
        assert!(report.comparisons.iter().filter(|comparison| {
            matches!(
                comparison.metric,
                ComparisonMetric::SuccessRateAtBudget
                    | ComparisonMetric::NormalizedSuccessAuc
                    | ComparisonMetric::BestLossReductionAtBudget
            )
        }).all(|comparison| {
            comparison.disposition == ComparisonDisposition::ImprovementBeyondMargin
                && comparison.estimated_delta >= comparison.practical_margin
                && comparison.confidence_interval_lower.unwrap() > 0.0
        }));
    }

    #[test]
    fn computes_paired_best_loss_reduction_when_every_target_is_covered() {
        let mut ledger = fixture();
        ledger.minimum_best_loss_reduction = 0.05;
        for target in &mut ledger.arms[0].target_queries {
            for observation in &mut target.observations {
                if observation.status == ObservationStatus::Evaluated {
                    observation.loss = Some(0.5);
                }
            }
        }
        for target in &mut ledger.arms[1].target_queries {
            for observation in &mut target.observations {
                if observation.status == ObservationStatus::Evaluated {
                    observation.loss = Some(0.001);
                }
            }
        }

        let report = evaluate_fixture(&ledger);
        let comparison = report
            .comparisons
            .iter()
            .find(|comparison| comparison.metric == ComparisonMetric::BestLossReductionAtBudget)
            .unwrap();
        assert!((comparison.estimated_delta - 0.499).abs() < 1e-12);
        assert_eq!(comparison.covered_target_count, 5);
        assert_eq!(comparison.total_target_count, 5);
        assert_eq!(comparison.disposition, ComparisonDisposition::ImprovementBeyondMargin);
        assert!(comparison.confidence_interval_lower.unwrap() > 0.0);
    }

    #[test]
    fn refuses_loss_reduction_interval_when_one_target_has_no_valid_outcome() {
        let mut ledger = fixture();
        ledger.arms[1].target_queries[2].observations = vec![observation(
            1,
            "all-failed",
            None,
            ObservationStatus::OracleFailed,
        )];
        let report = evaluate_fixture(&ledger);
        let comparison = report
            .comparisons
            .iter()
            .find(|comparison| comparison.metric == ComparisonMetric::BestLossReductionAtBudget)
            .unwrap();
        assert_eq!(comparison.disposition, ComparisonDisposition::InsufficientTargetCoverage);
        assert_eq!(comparison.covered_target_count, 4);
        assert_eq!(comparison.total_target_count, 5);
        assert!(comparison.confidence_interval_lower.is_none());
        assert!(comparison.confidence_interval_upper.is_none());
    }

    #[test]
    fn rejects_mismatched_paired_target_sets() {
        let mut ledger = fixture();
        ledger.arms[1].target_queries.pop();
        let bytes = serde_json::to_vec(&ledger).unwrap();
        assert!(evaluate_json_ledger(&bytes, 200, 17)
            .unwrap_err()
            .contains("same target_query_id/group_id set"));
    }

    #[test]
    fn rejects_noncontiguous_query_ranks() {
        let mut ledger = fixture();
        ledger.arms[0].target_queries[0].observations[1].query_rank = 3;
        let bytes = serde_json::to_vec(&ledger).unwrap();
        assert!(evaluate_json_ledger(&bytes, 200, 17)
            .unwrap_err()
            .contains("contiguous"));
    }

    #[test]
    fn rejects_loss_for_failed_observation() {
        let mut ledger = fixture();
        ledger.arms[0].target_queries[2].observations[0].loss = Some(0.5);
        let bytes = serde_json::to_vec(&ledger).unwrap();
        assert!(evaluate_json_ledger(&bytes, 200, 17)
            .unwrap_err()
            .contains("must not contain a loss"));
    }

    #[test]
    fn checked_in_outcome_fixture_and_json_schemas_are_parseable() {
        let raw = include_bytes!("../fixtures/synthetic-outcome-ledger.json");
        let report = evaluate_json_ledger(raw, 200, 17).expect("checked-in synthetic ledger");
        assert_eq!(report.schema_version, METRICS_REPORT_SCHEMA_VERSION);
        assert_eq!(report.arms.len(), 2);
        assert_eq!(report.arms.iter().find(|arm| arm.arm_id == "baseline").unwrap().oracle_attempts, 11);

        let ledger_schema: Value =
            serde_json::from_str(include_str!("../schema/outcome-ledger.schema.json")).unwrap();
        let report_schema: Value =
            serde_json::from_str(include_str!("../schema/benchmark-metrics-report.schema.json")).unwrap();
        assert_eq!(ledger_schema["additionalProperties"], false);
        assert_eq!(report_schema["additionalProperties"], false);
        assert_eq!(ledger_schema["properties"]["schema_version"]["const"], OUTCOME_LEDGER_SCHEMA_VERSION);
        assert!(
            ledger_schema["properties"]["arms"]["items"]["properties"]["target_queries"]["items"]["properties"]["observations"]["items"]["required"]
                .as_array()
                .unwrap()
                .contains(&json!("observation_artifact_sha256"))
        );
        assert_eq!(report_schema["properties"]["schema_version"]["const"], METRICS_REPORT_SCHEMA_VERSION);
        let declared_metrics = report_schema["properties"]["comparisons"]["items"]["properties"]["metric"]["enum"]
            .as_array()
            .unwrap();
        for metric_name in [
            "best_loss_reduction_at_budget",
            "worst_group_success_rate_at_budget",
            "worst_group_normalized_success_auc",
        ] {
            assert!(declared_metrics.contains(&json!(metric_name)));
        }
        assert!(
            report_schema["properties"]["comparisons"]["items"]["properties"]["disposition"]["enum"]
                .as_array()
                .unwrap()
                .contains(&json!("insufficient_within_group_replication"))
        );
        assert!(
            ledger_schema["required"]
                .as_array()
                .unwrap()
                .contains(&json!("minimum_best_loss_reduction"))
        );

        let report_value = serde_json::to_value(report).expect("serialize report");
        assert_eq!(report_value["schema_version"], METRICS_REPORT_SCHEMA_VERSION);
        assert_eq!(report_value["arms"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn rejects_unknown_outcome_fields() {
        let mut value: Value = serde_json::to_value(fixture()).unwrap();
        value.as_object_mut().unwrap().insert("hidden_target_labels".to_string(), json!([1.0, 2.0]));
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(evaluate_json_ledger(&bytes, 200, 17).unwrap_err().contains("invalid outcome-ledger JSON"));
    }

    #[test]
    fn rejects_invalid_observation_artifact_digest() {
        let mut ledger = fixture();
        ledger.arms[0].target_queries[0].observations[0].observation_artifact_sha256 =
            "not-a-sha256".to_string();
        let bytes = serde_json::to_vec(&ledger).unwrap();
        assert!(evaluate_json_ledger(&bytes, 200, 17)
            .unwrap_err()
            .contains("invalid observation artifact digest"));
    }

    #[test]
    fn rejects_uppercase_or_malformed_manifest_digests() {
        let mut ledger = fixture();
        ledger.task_contract_sha256 = "A".repeat(64);
        let bytes = serde_json::to_vec(&ledger).unwrap();
        assert!(evaluate_json_ledger(&bytes, 200, 17).unwrap_err().contains("lowercase hexadecimal"));
    }

    #[test]
    fn rejects_missing_baseline_arm() {
        let mut ledger = fixture();
        ledger.baseline_arm_id = "missing".to_string();
        let bytes = serde_json::to_vec(&ledger).unwrap();
        assert!(evaluate_json_ledger(&bytes, 200, 17).unwrap_err().contains("baseline_arm_id"));
    }
}
