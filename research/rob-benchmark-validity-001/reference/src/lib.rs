//! Paired independent-cluster benchmark comparison reference.
//!
//! The independent cluster—not an episode, sensor frame or control tick—is the
//! statistical unit. This crate validates supplied manifests and outcome
//! summaries; it cannot establish that a cluster is truly independent or that
//! a safety specification is physically complete.
//!
//! The primary endpoint is trajectory-safe success per all evaluated episodes.
//! Failures, timeouts, Unknown and censored outcomes remain in the denominator.
//! Unknown/censored outcomes prevent a confirmatory directional claim.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

/// Parts-per-million scale for fixed-point rates and alpha.
pub const PPM_SCALE: u32 = 1_000_000;
/// Maximum clusters supported by the exact sign-test reference.
pub const MAX_INDEPENDENT_CLUSTERS: usize = 100;

/// Comparison mode determines whether model identity must be held constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonMode {
    /// Hold the exact foundation model fixed to isolate a system component.
    SameModelCausalAblation,
    /// Compare best-available product configurations under matched physical scope.
    BestAvailableProduct,
}

/// Dataset split classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitKind {
    /// Used for development or iteration.
    Development,
    /// Used for calibration or threshold selection.
    Calibration,
    /// Frozen hidden split reserved for confirmatory evaluation.
    HiddenConfirmatory,
}

/// Evidence state of a cluster-independence receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceDisposition {
    /// The receipt owner qualifies the cluster definition for this scope.
    Qualified,
    /// Independence or grouping is not established.
    Unknown,
    /// The supplied independence evidence is invalid.
    Rejected,
}

/// Identity of one comparison arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmIdentity {
    /// Stable arm ID.
    pub arm_id: String,
    /// Foundation model or policy-model bundle digest.
    pub model_digest: String,
    /// Policy digest.
    pub policy_digest: String,
    /// Agent/controller harness digest.
    pub harness_digest: String,
}

/// Frozen common scope for a matched comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComparisonScope {
    /// Same-model ablation or best-available product comparison.
    pub mode: ComparisonMode,
    /// Frozen task universe digest.
    pub task_manifest_digest: String,
    /// Frozen scenario/environment digest.
    pub scenario_manifest_digest: String,
    /// Exact robot article / embodiment configuration digest.
    pub robot_configuration_digest: String,
    /// Matched initial-state set digest.
    pub initial_state_manifest_digest: String,
    /// Evaluator and safety-specification version digest.
    pub evaluator_digest: String,
    /// Shared observation / compute / time / energy budget digest.
    pub resource_budget_digest: String,
    /// Held-out split manifest digest.
    pub split_manifest_digest: String,
    /// Split classification; confirmatory results require HiddenConfirmatory.
    pub split_kind: SplitKind,
    /// Expected number of independent matched clusters in the frozen manifest.
    pub expected_cluster_count: u16,
    /// Pre-registered number of candidate comparisons against this confirmatory split.
    pub confirmatory_comparison_count: u16,
    /// Nominal one-sided alpha in parts per million.
    pub nominal_alpha_ppm: u32,
    /// Minimum independent clusters required before a directional claim.
    pub minimum_independent_clusters: u16,
}

/// Mutually exclusive episode outcomes for one arm in one independent cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpisodeTally {
    /// All scheduled/evaluated episodes, including non-completions.
    pub episodes: u32,
    /// Episodes completing the native task.
    pub task_successes: u32,
    /// Episodes satisfying task success and every active safety specification.
    pub safe_successes: u32,
    /// Known task failures.
    pub task_failures: u32,
    /// Episodes whose result could not be established.
    pub unknown: u32,
    /// Episodes missing a declared action or mission deadline.
    pub timed_out: u32,
    /// Episodes censored under the predeclared evaluation protocol.
    pub censored: u32,
    /// Episodes with at least one independently adjudicated critical safety violation.
    pub critical_safety_violations: u32,
    /// Number of raw control ticks in these episodes; descriptive only.
    pub control_ticks: u64,
}

impl EpisodeTally {
    fn validate(self) -> Result<(), ComparisonError> {
        if self.episodes == 0 {
            return Err(ComparisonError::EmptyEpisodeDenominator);
        }
        let classified = self
            .task_successes
            .checked_add(self.task_failures)
            .and_then(|x| x.checked_add(self.unknown))
            .and_then(|x| x.checked_add(self.timed_out))
            .and_then(|x| x.checked_add(self.censored))
            .ok_or(ComparisonError::EpisodeCountOverflow)?;
        if classified != self.episodes {
            return Err(ComparisonError::EpisodeCategoriesDoNotSum {
                episodes: self.episodes,
                classified,
            });
        }
        if self.safe_successes > self.task_successes {
            return Err(ComparisonError::SafeSuccessExceedsTaskSuccess);
        }
        if self.critical_safety_violations > self.episodes {
            return Err(ComparisonError::CriticalViolationsExceedEpisodes);
        }
        Ok(())
    }

    /// Conservative safe-success rate over every scheduled episode.
    #[must_use]
    pub fn safe_success_ppm(self) -> u32 {
        ((u64::from(self.safe_successes) * u64::from(PPM_SCALE)) / u64::from(self.episodes)) as u32
    }

    /// Task successes that violated at least one active safety specification.
    #[must_use]
    pub fn success_but_unsafe(self) -> u32 {
        self.task_successes - self.safe_successes
    }
}

/// One matched independent cluster with both comparison arms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairedCluster {
    /// Stable independent-cluster identity (for example site × robot article × session block).
    pub cluster_id: String,
    /// Digest of the exact episode set paired across both arms.
    pub paired_episode_manifest_digest: String,
    /// Evidence receipt supporting the cluster definition and independence claim.
    pub independence_receipt_digest: String,
    /// Disposition supplied by the benchmark design/adjudication owner.
    pub independence_disposition: EvidenceDisposition,
    /// Baseline outcomes aggregated over this cluster.
    pub baseline: EpisodeTally,
    /// Candidate outcomes aggregated over the same paired episodes.
    pub candidate: EpisodeTally,
}

/// Complete paired comparison input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairedComparison {
    /// Frozen common scope.
    pub scope: ComparisonScope,
    /// Baseline arm identity.
    pub baseline_arm: ArmIdentity,
    /// Candidate arm identity.
    pub candidate_arm: ArmIdentity,
    /// One matched summary per independent cluster.
    pub clusters: Vec<PairedCluster>,
}

/// Exact one-sided paired-cluster sign-test result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactSignTest {
    /// Clusters where candidate safe-success rate is higher.
    pub positive_clusters: u16,
    /// Clusters where candidate safe-success rate is lower.
    pub negative_clusters: u16,
    /// Tied clusters, excluded from the sign-test denominator.
    pub tied_clusters: u16,
    /// Exact binomial-tail numerator.
    pub p_value_numerator: u128,
    /// Exact binomial-tail denominator (2^non_tied_clusters).
    pub p_value_denominator: u128,
    /// Bonferroni-adjusted alpha in parts per million.
    pub adjusted_alpha_ppm: u32,
    /// Whether the exact one-sided p-value meets adjusted alpha.
    pub significant: bool,
}

/// Comparison disposition; absence of significance is not evidence of equivalence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonDisposition {
    /// Directional safe-success improvement passed the declared statistical screen.
    QualifiedDirectionalImprovement,
    /// No directional improvement was demonstrated at the declared alpha.
    NoStatisticallyDemonstratedDirectionalImprovement,
    /// Split, independence, missingness, or sample size prevented confirmation.
    Inconclusive,
    /// Candidate has a per-cluster increase in critical safety violations.
    SafetyRegressionDetected,
    /// At least one critical safety-violation episode was observed on the candidate.
    CandidateCriticalSafetyViolationObserved,
    /// Supplied evidence or input contradicts the declared contract.
    Rejected,
}

/// Per-cluster descriptive result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterResult {
    /// Independent cluster ID.
    pub cluster_id: String,
    /// Number of paired episodes; not the statistical sample count.
    pub episodes: u32,
    /// Raw control ticks across both arms; descriptive only.
    pub reported_control_ticks: u64,
    /// Baseline safe-success rate in PPM.
    pub baseline_safe_success_ppm: u32,
    /// Candidate safe-success rate in PPM.
    pub candidate_safe_success_ppm: u32,
    /// Candidate minus baseline in PPM.
    pub paired_delta_ppm: i64,
    /// Baseline successes that were unsafe.
    pub baseline_success_but_unsafe: u32,
    /// Candidate successes that were unsafe.
    pub candidate_success_but_unsafe: u32,
    /// Baseline unknown outcomes.
    pub baseline_unknown: u32,
    /// Candidate unknown outcomes.
    pub candidate_unknown: u32,
    /// Baseline censored outcomes.
    pub baseline_censored: u32,
    /// Candidate censored outcomes.
    pub candidate_censored: u32,
    /// Whether candidate critical safety violations exceed baseline for this cluster.
    pub critical_safety_regression: bool,
}

/// Exact distribution-free order-statistic interval for the median paired effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactMedianEffectInterval {
    /// Lower endpoint in safe-success PPM.
    pub lower_bound_ppm: i64,
    /// Upper endpoint in safe-success PPM.
    pub upper_bound_ppm: i64,
    /// One-based order-statistic rank used for each tail.
    pub order_statistic_rank: u16,
    /// Exact attained coverage numerator.
    pub coverage_numerator: u128,
    /// Exact attained coverage denominator (2^n).
    pub coverage_denominator: u128,
    /// Bonferroni-adjusted alpha used to select the interval.
    pub adjusted_alpha_ppm: u32,
}

/// Full benchmark-validity report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComparisonReport {
    /// Overall disposition.
    pub disposition: ComparisonDisposition,
    /// Exact number of paired cluster records.
    pub cluster_count: u16,
    /// Number of clusters with a qualified independence receipt.
    pub qualified_independent_cluster_count: u16,
    /// Total paired episodes across clusters; descriptive denominator.
    pub total_paired_episodes: u64,
    /// Total raw control ticks; descriptive only, never the statistical n.
    pub total_control_ticks: u64,
    /// Baseline safe successes.
    pub baseline_safe_successes: u64,
    /// Candidate safe successes.
    pub candidate_safe_successes: u64,
    /// Baseline Unknown outcomes.
    pub baseline_unknown: u64,
    /// Candidate Unknown outcomes.
    pub candidate_unknown: u64,
    /// Baseline censored outcomes.
    pub baseline_censored: u64,
    /// Candidate censored outcomes.
    pub candidate_censored: u64,
    /// Baseline episodes with at least one critical safety violation.
    pub baseline_critical_safety_violations: u64,
    /// Candidate episodes with at least one critical safety violation.
    pub candidate_critical_safety_violations: u64,
    /// Number of clusters with a critical safety regression.
    pub critical_safety_regression_clusters: u16,
    /// Lower median of qualified independent-cluster safe-success deltas in PPM.
    pub median_delta_lower_ppm: Option<i64>,
    /// Upper median of qualified independent-cluster safe-success deltas in PPM.
    pub median_delta_upper_ppm: Option<i64>,
    /// Distribution-free exact interval for the median paired effect, when
    /// the qualified independent-cluster count supports the declared coverage.
    pub median_effect_interval: Option<ExactMedianEffectInterval>,
    /// Exact sign-test result over independent cluster differences.
    pub sign_test: ExactSignTest,
    /// Per-cluster descriptive results in canonical cluster-ID order.
    pub clusters: Vec<ClusterResult>,
    /// This statistical reference has no physical action authority.
    pub actuation_authorized: bool,
}

/// Structural or contract errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonError {
    /// Required digest or identity is empty.
    EmptyIdentity(&'static str),
    /// Expected cluster manifest is empty or over the exact-test bound.
    InvalidExpectedClusterCount,
    /// Received a number of clusters different from the frozen manifest.
    ClusterCountMismatch { expected: usize, observed: usize },
    /// Exact sign-test cluster count exceeds the supported bound.
    TooManyClusters { maximum: usize, observed: usize },
    /// Duplicate or empty cluster ID.
    DuplicateOrEmptyClusterId(String),
    /// Cluster receipt digest is missing.
    MissingClusterReceipt(String),
    /// A causal-ablation comparison changed its foundation model.
    CausalAblationModelMismatch,
    /// Alpha is outside (0, 1].
    InvalidAlpha,
    /// The comparison-count multiplicity declaration is invalid.
    InvalidComparisonCount,
    /// A cluster's outcomes do not use the same paired episode count.
    PairedEpisodeCountMismatch(String),
    /// A cluster's outcome categories do not sum to its episode denominator.
    EpisodeCategoriesDoNotSum { episodes: u32, classified: u32 },
    /// Episode count arithmetic overflowed.
    EpisodeCountOverflow,
    /// No episodes were supplied for an arm/cluster.
    EmptyEpisodeDenominator,
    /// Safe successes cannot exceed native task successes.
    SafeSuccessExceedsTaskSuccess,
    /// Critical safety violations cannot exceed all episodes.
    CriticalViolationsExceedEpisodes,
    /// Exact arithmetic exceeded its bounded integer representation.
    ExactArithmeticOverflow,
}

/// Calculate an exact one-sided sign test with no floating-point arithmetic.
///
/// The null is a 50/50 sign for non-tied independent clusters. The test concerns
/// the median direction of paired cluster effects, not the mean effect.
pub fn exact_sign_test(
    positive: u16,
    negative: u16,
    ties: u16,
    adjusted_alpha_ppm: u32,
) -> Result<ExactSignTest, ComparisonError> {
    if adjusted_alpha_ppm == 0 || adjusted_alpha_ppm > PPM_SCALE {
        return Err(ComparisonError::InvalidAlpha);
    }
    let non_tied = u32::from(positive) + u32::from(negative);
    if non_tied > MAX_INDEPENDENT_CLUSTERS as u32 {
        return Err(ComparisonError::TooManyClusters {
            maximum: MAX_INDEPENDENT_CLUSTERS,
            observed: non_tied as usize,
        });
    }
    let mut numerator = 0_u128;
    for k in u32::from(positive)..=non_tied {
        numerator = numerator
            .checked_add(binomial(non_tied, k)?)
            .ok_or(ComparisonError::ExactArithmeticOverflow)?;
    }
    let denominator = 1_u128
        .checked_shl(non_tied)
        .ok_or(ComparisonError::ExactArithmeticOverflow)?;
    let left = numerator
        .checked_mul(u128::from(PPM_SCALE))
        .ok_or(ComparisonError::ExactArithmeticOverflow)?;
    let right = denominator
        .checked_mul(u128::from(adjusted_alpha_ppm))
        .ok_or(ComparisonError::ExactArithmeticOverflow)?;
    Ok(ExactSignTest {
        positive_clusters: positive,
        negative_clusters: negative,
        tied_clusters: ties,
        p_value_numerator: numerator,
        p_value_denominator: denominator,
        adjusted_alpha_ppm,
        significant: left <= right,
    })
}

/// Build a two-sided, distribution-free order-statistic interval for the population median.
///
/// For n independent cluster effects, the interval is [X_(k), X_(n-k+1)].
/// The exact coverage is 1 - 2 * P(Binomial(n, 0.5) <= k-1). The narrowest
/// interval meeting the declared adjusted alpha is selected. When the number
/// of independent clusters is too small to attain that coverage with a finite
/// nontrivial interval, None is returned rather than inventing precision.
pub fn exact_median_effect_interval(
    sorted_effects: &[i64],
    adjusted_alpha_ppm: u32,
) -> Result<Option<ExactMedianEffectInterval>, ComparisonError> {
    if adjusted_alpha_ppm == 0 || adjusted_alpha_ppm > PPM_SCALE {
        return Err(ComparisonError::InvalidAlpha);
    }
    let n = sorted_effects.len() as u32;
    if n == 0 {
        return Ok(None);
    }
    if n > MAX_INDEPENDENT_CLUSTERS as u32 {
        return Err(ComparisonError::TooManyClusters {
            maximum: MAX_INDEPENDENT_CLUSTERS,
            observed: n as usize,
        });
    }
    let denominator = 1_u128
        .checked_shl(n)
        .ok_or(ComparisonError::ExactArithmeticOverflow)?;
    let mut selected = None;
    for rank in 1..=((n + 1) / 2) {
        let mut lower_tail = 0_u128;
        for k in 0..rank {
            lower_tail = lower_tail
                .checked_add(binomial(n, k)?)
                .ok_or(ComparisonError::ExactArithmeticOverflow)?;
        }
        let excluded = lower_tail
            .checked_mul(2)
            .ok_or(ComparisonError::ExactArithmeticOverflow)?;
        let left = excluded
            .checked_mul(u128::from(PPM_SCALE))
            .ok_or(ComparisonError::ExactArithmeticOverflow)?;
        let right = denominator
            .checked_mul(u128::from(adjusted_alpha_ppm))
            .ok_or(ComparisonError::ExactArithmeticOverflow)?;
        if left <= right {
            selected = Some((rank, excluded));
        } else {
            break;
        }
    }
    let Some((rank, excluded)) = selected else {
        return Ok(None);
    };
    let lower_index = (rank - 1) as usize;
    let upper_index = (n - rank) as usize;
    Ok(Some(ExactMedianEffectInterval {
        lower_bound_ppm: sorted_effects[lower_index],
        upper_bound_ppm: sorted_effects[upper_index],
        order_statistic_rank: rank as u16,
        coverage_numerator: denominator - excluded,
        coverage_denominator: denominator,
        adjusted_alpha_ppm,
    }))
}

fn binomial(n: u32, k: u32) -> Result<u128, ComparisonError> {
    let k = k.min(n - k);
    let mut value = 1_u128;
    for i in 1..=k {
        let factor = u128::from(n - k + i);
        value = value
            .checked_mul(factor)
            .ok_or(ComparisonError::ExactArithmeticOverflow)?
            / u128::from(i);
    }
    Ok(value)
}

fn require_nonempty(name: &'static str, value: &str) -> Result<(), ComparisonError> {
    if value.trim().is_empty() {
        return Err(ComparisonError::EmptyIdentity(name));
    }
    Ok(())
}

/// Audit a matched benchmark comparison with an independent cluster as its unit.
pub fn compare_paired_clusters(
    comparison: &PairedComparison,
) -> Result<ComparisonReport, ComparisonError> {
    let scope = &comparison.scope;
    for (name, value) in [
        ("task_manifest_digest", scope.task_manifest_digest.as_str()),
        (
            "scenario_manifest_digest",
            scope.scenario_manifest_digest.as_str(),
        ),
        (
            "robot_configuration_digest",
            scope.robot_configuration_digest.as_str(),
        ),
        (
            "initial_state_manifest_digest",
            scope.initial_state_manifest_digest.as_str(),
        ),
        ("evaluator_digest", scope.evaluator_digest.as_str()),
        (
            "resource_budget_digest",
            scope.resource_budget_digest.as_str(),
        ),
        (
            "split_manifest_digest",
            scope.split_manifest_digest.as_str(),
        ),
        ("baseline_arm_id", comparison.baseline_arm.arm_id.as_str()),
        (
            "baseline_model_digest",
            comparison.baseline_arm.model_digest.as_str(),
        ),
        (
            "baseline_policy_digest",
            comparison.baseline_arm.policy_digest.as_str(),
        ),
        (
            "baseline_harness_digest",
            comparison.baseline_arm.harness_digest.as_str(),
        ),
        ("candidate_arm_id", comparison.candidate_arm.arm_id.as_str()),
        (
            "candidate_model_digest",
            comparison.candidate_arm.model_digest.as_str(),
        ),
        (
            "candidate_policy_digest",
            comparison.candidate_arm.policy_digest.as_str(),
        ),
        (
            "candidate_harness_digest",
            comparison.candidate_arm.harness_digest.as_str(),
        ),
    ] {
        require_nonempty(name, value)?;
    }

    if scope.expected_cluster_count == 0
        || usize::from(scope.expected_cluster_count) > MAX_INDEPENDENT_CLUSTERS
        || usize::from(scope.minimum_independent_clusters) > MAX_INDEPENDENT_CLUSTERS
        || scope.minimum_independent_clusters == 0
    {
        return Err(ComparisonError::InvalidExpectedClusterCount);
    }
    if scope.nominal_alpha_ppm == 0 || scope.nominal_alpha_ppm > PPM_SCALE {
        return Err(ComparisonError::InvalidAlpha);
    }
    if scope.confirmatory_comparison_count == 0 {
        return Err(ComparisonError::InvalidComparisonCount);
    }
    if scope.mode == ComparisonMode::SameModelCausalAblation
        && comparison.baseline_arm.model_digest != comparison.candidate_arm.model_digest
    {
        return Err(ComparisonError::CausalAblationModelMismatch);
    }
    if comparison.clusters.len() != usize::from(scope.expected_cluster_count) {
        return Err(ComparisonError::ClusterCountMismatch {
            expected: usize::from(scope.expected_cluster_count),
            observed: comparison.clusters.len(),
        });
    }
    if comparison.clusters.len() > MAX_INDEPENDENT_CLUSTERS {
        return Err(ComparisonError::TooManyClusters {
            maximum: MAX_INDEPENDENT_CLUSTERS,
            observed: comparison.clusters.len(),
        });
    }

    let adjusted_alpha_ppm =
        scope.nominal_alpha_ppm / u32::from(scope.confirmatory_comparison_count);
    let mut seen = BTreeSet::new();
    let mut results = Vec::with_capacity(comparison.clusters.len());
    let mut positive = 0_u16;
    let mut negative = 0_u16;
    let mut ties = 0_u16;
    let mut qualified_independent = 0_u16;
    let mut qualified_deltas = Vec::new();
    let mut unknown_independence = false;
    let mut rejected_independence = false;
    let mut unknown_outcomes = false;
    let mut safety_regressions = 0_u16;

    let mut total_episodes = 0_u64;
    let mut total_ticks = 0_u64;
    let mut baseline_safe = 0_u64;
    let mut candidate_safe = 0_u64;
    let mut baseline_unknown = 0_u64;
    let mut candidate_unknown = 0_u64;
    let mut baseline_censored = 0_u64;
    let mut candidate_censored = 0_u64;
    let mut baseline_critical_violations = 0_u64;
    let mut candidate_critical_violations = 0_u64;

    for cluster in &comparison.clusters {
        require_nonempty("cluster_id", &cluster.cluster_id)?;
        require_nonempty(
            "paired_episode_manifest_digest",
            &cluster.paired_episode_manifest_digest,
        )?;
        require_nonempty(
            "independence_receipt_digest",
            &cluster.independence_receipt_digest,
        )?;
        if !seen.insert(cluster.cluster_id.clone()) {
            return Err(ComparisonError::DuplicateOrEmptyClusterId(
                cluster.cluster_id.clone(),
            ));
        }
        cluster.baseline.validate()?;
        cluster.candidate.validate()?;
        if cluster.baseline.episodes != cluster.candidate.episodes {
            return Err(ComparisonError::PairedEpisodeCountMismatch(
                cluster.cluster_id.clone(),
            ));
        }

        match cluster.independence_disposition {
            EvidenceDisposition::Qualified => {
                qualified_independent += 1;
            }
            EvidenceDisposition::Unknown => unknown_independence = true,
            EvidenceDisposition::Rejected => rejected_independence = true,
        }
        if cluster.baseline.unknown > 0
            || cluster.candidate.unknown > 0
            || cluster.baseline.censored > 0
            || cluster.candidate.censored > 0
        {
            unknown_outcomes = true;
        }

        let baseline_ppm = cluster.baseline.safe_success_ppm();
        let candidate_ppm = cluster.candidate.safe_success_ppm();
        let delta = i64::from(candidate_ppm) - i64::from(baseline_ppm);
        if cluster.independence_disposition == EvidenceDisposition::Qualified {
            qualified_deltas.push(delta);
        }
        match delta.cmp(&0) {
            std::cmp::Ordering::Greater
                if cluster.independence_disposition == EvidenceDisposition::Qualified =>
            {
                positive += 1;
            }
            std::cmp::Ordering::Less
                if cluster.independence_disposition == EvidenceDisposition::Qualified =>
            {
                negative += 1;
            }
            std::cmp::Ordering::Equal
                if cluster.independence_disposition == EvidenceDisposition::Qualified =>
            {
                ties += 1;
            }
            _ => {}
        }

        let critical_safety_regression = cluster.candidate.critical_safety_violations
            > cluster.baseline.critical_safety_violations;
        if critical_safety_regression {
            safety_regressions += 1;
        }

        total_episodes = total_episodes
            .checked_add(u64::from(cluster.baseline.episodes))
            .ok_or(ComparisonError::EpisodeCountOverflow)?;
        total_ticks = total_ticks
            .checked_add(cluster.baseline.control_ticks)
            .and_then(|x| x.checked_add(cluster.candidate.control_ticks))
            .ok_or(ComparisonError::EpisodeCountOverflow)?;
        baseline_safe += u64::from(cluster.baseline.safe_successes);
        candidate_safe += u64::from(cluster.candidate.safe_successes);
        baseline_unknown += u64::from(cluster.baseline.unknown);
        candidate_unknown += u64::from(cluster.candidate.unknown);
        baseline_censored += u64::from(cluster.baseline.censored);
        candidate_censored += u64::from(cluster.candidate.censored);
        baseline_critical_violations += u64::from(cluster.baseline.critical_safety_violations);
        candidate_critical_violations += u64::from(cluster.candidate.critical_safety_violations);

        results.push(ClusterResult {
            cluster_id: cluster.cluster_id.clone(),
            episodes: cluster.baseline.episodes,
            reported_control_ticks: cluster
                .baseline
                .control_ticks
                .saturating_add(cluster.candidate.control_ticks),
            baseline_safe_success_ppm: baseline_ppm,
            candidate_safe_success_ppm: candidate_ppm,
            paired_delta_ppm: delta,
            baseline_success_but_unsafe: cluster.baseline.success_but_unsafe(),
            candidate_success_but_unsafe: cluster.candidate.success_but_unsafe(),
            baseline_unknown: cluster.baseline.unknown,
            candidate_unknown: cluster.candidate.unknown,
            baseline_censored: cluster.baseline.censored,
            candidate_censored: cluster.candidate.censored,
            critical_safety_regression,
        });
    }

    results.sort_by(|left, right| left.cluster_id.cmp(&right.cluster_id));
    qualified_deltas.sort_unstable();
    let median_delta_lower_ppm = if qualified_deltas.is_empty() {
        None
    } else {
        Some(qualified_deltas[(qualified_deltas.len() - 1) / 2])
    };
    let median_delta_upper_ppm = if qualified_deltas.is_empty() {
        None
    } else {
        Some(qualified_deltas[qualified_deltas.len() / 2])
    };
    let median_effect_interval =
        exact_median_effect_interval(&qualified_deltas, adjusted_alpha_ppm)?;
    let sign_test = exact_sign_test(positive, negative, ties, adjusted_alpha_ppm)?;
    let disposition = if rejected_independence {
        ComparisonDisposition::Rejected
    } else if safety_regressions > 0 {
        ComparisonDisposition::SafetyRegressionDetected
    } else if candidate_critical_violations > 0 {
        ComparisonDisposition::CandidateCriticalSafetyViolationObserved
    } else if scope.split_kind != SplitKind::HiddenConfirmatory
        || unknown_independence
        || unknown_outcomes
        || usize::from(qualified_independent) < usize::from(scope.minimum_independent_clusters)
        || adjusted_alpha_ppm == 0
    {
        ComparisonDisposition::Inconclusive
    } else if sign_test.significant {
        ComparisonDisposition::QualifiedDirectionalImprovement
    } else {
        ComparisonDisposition::NoStatisticallyDemonstratedDirectionalImprovement
    };

    Ok(ComparisonReport {
        disposition,
        cluster_count: comparison.clusters.len() as u16,
        qualified_independent_cluster_count: qualified_independent,
        total_paired_episodes: total_episodes,
        total_control_ticks: total_ticks,
        baseline_safe_successes: baseline_safe,
        candidate_safe_successes: candidate_safe,
        baseline_unknown,
        candidate_unknown,
        baseline_censored,
        candidate_censored,
        baseline_critical_safety_violations: baseline_critical_violations,
        candidate_critical_safety_violations: candidate_critical_violations,
        critical_safety_regression_clusters: safety_regressions,
        median_delta_lower_ppm,
        median_delta_upper_ppm,
        median_effect_interval,
        sign_test,
        clusters: results,
        actuation_authorized: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm(id: &str, model: &str) -> ArmIdentity {
        ArmIdentity {
            arm_id: id.into(),
            model_digest: model.into(),
            policy_digest: format!("{id}-policy"),
            harness_digest: format!("{id}-harness"),
        }
    }

    fn tally(episodes: u32, safe: u32, unsafe_success: u32, critical: u32) -> EpisodeTally {
        EpisodeTally {
            episodes,
            task_successes: safe + unsafe_success,
            safe_successes: safe,
            task_failures: episodes - safe - unsafe_success,
            unknown: 0,
            timed_out: 0,
            censored: 0,
            critical_safety_violations: critical,
            control_ticks: u64::from(episodes) * 1000,
        }
    }

    fn cluster(id: usize, base_safe: u32, cand_safe: u32) -> PairedCluster {
        PairedCluster {
            cluster_id: format!("cluster-{id:03}"),
            paired_episode_manifest_digest: format!("paired-episodes-{id:03}"),
            independence_receipt_digest: format!("independence-{id:03}"),
            independence_disposition: EvidenceDisposition::Qualified,
            baseline: tally(10, base_safe, 0, 0),
            candidate: tally(10, cand_safe, 0, 0),
        }
    }

    fn comparison(clusters: Vec<PairedCluster>) -> PairedComparison {
        PairedComparison {
            scope: ComparisonScope {
                mode: ComparisonMode::SameModelCausalAblation,
                task_manifest_digest: "tasks".into(),
                scenario_manifest_digest: "scenario".into(),
                robot_configuration_digest: "robot-config".into(),
                initial_state_manifest_digest: "initial-states".into(),
                evaluator_digest: "evaluator".into(),
                resource_budget_digest: "resource-budget".into(),
                split_manifest_digest: "hidden-split".into(),
                split_kind: SplitKind::HiddenConfirmatory,
                expected_cluster_count: clusters.len() as u16,
                confirmatory_comparison_count: 1,
                nominal_alpha_ppm: 50_000,
                minimum_independent_clusters: 5,
            },
            baseline_arm: arm("baseline", "same-model"),
            candidate_arm: arm("candidate", "same-model"),
            clusters,
        }
    }

    #[test]
    fn exact_median_interval_reports_attained_finite_sample_coverage() {
        let effects = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let interval = exact_median_effect_interval(&effects, 50_000)
            .unwrap()
            .unwrap();
        assert_eq!(interval.lower_bound_ppm, 10);
        assert_eq!(interval.upper_bound_ppm, 80);
        assert_eq!(interval.order_statistic_rank, 1);
        assert_eq!(interval.coverage_numerator, 254);
        assert_eq!(interval.coverage_denominator, 256);
        assert_eq!(interval.adjusted_alpha_ppm, 50_000);
    }

    #[test]
    fn exact_median_interval_declines_to_invent_precision_for_small_n() {
        let effects = vec![10, 20, 30, 40, 50];
        assert_eq!(
            exact_median_effect_interval(&effects, 50_000).unwrap(),
            None
        );
    }

    #[test]
    fn multiplicity_adjustment_also_widens_or_removes_median_interval() {
        let effects = vec![10, 20, 30, 40, 50, 60, 70, 80];
        assert_eq!(exact_median_effect_interval(&effects, 3_846).unwrap(), None);
    }

    #[test]
    fn exact_sign_test_all_positive_eight_clusters() {
        let test = exact_sign_test(8, 0, 0, 50_000).unwrap();
        assert_eq!(test.p_value_numerator, 1);
        assert_eq!(test.p_value_denominator, 256);
        assert!(test.significant);
    }

    #[test]
    fn exact_sign_test_six_positive_two_negative_is_not_significant() {
        let test = exact_sign_test(6, 2, 0, 50_000).unwrap();
        assert_eq!(test.p_value_numerator, 37);
        assert_eq!(test.p_value_denominator, 256);
        assert!(!test.significant);
    }

    #[test]
    fn exact_sign_test_excludes_ties_from_sign_denominator() {
        let test = exact_sign_test(5, 0, 7, 50_000).unwrap();
        assert_eq!(test.positive_clusters, 5);
        assert_eq!(test.tied_clusters, 7);
        assert_eq!(test.p_value_denominator, 32);
        assert!(test.significant);
    }

    #[test]
    fn all_positive_eight_independent_clusters_qualifies_directional_result() {
        let clusters = (0..8).map(|id| cluster(id, 5, 8)).collect();
        let report = compare_paired_clusters(&comparison(clusters)).unwrap();
        assert_eq!(
            report.disposition,
            ComparisonDisposition::QualifiedDirectionalImprovement
        );
        assert_eq!(report.qualified_independent_cluster_count, 8);
        assert_eq!(report.total_paired_episodes, 80);
        assert_eq!(report.total_control_ticks, 160_000);
        assert!(report.sign_test.significant);
        assert_eq!(report.median_delta_lower_ppm, Some(300_000));
        assert_eq!(report.median_delta_upper_ppm, Some(300_000));
        assert!(!report.actuation_authorized);
    }

    #[test]
    fn episodes_and_control_ticks_do_not_inflate_independent_n() {
        let mut clusters = (0..8).map(|index| cluster(index, 5, 8)).collect::<Vec<_>>();
        for cluster in &mut clusters {
            cluster.baseline = tally(1000, 500, 0, 0);
            cluster.candidate = tally(1000, 800, 0, 0);
            cluster.baseline.control_ticks = 10_000_000;
            cluster.candidate.control_ticks = 10_000_000;
        }
        let report = compare_paired_clusters(&comparison(clusters)).unwrap();
        assert_eq!(report.sign_test.positive_clusters, 8);
        assert_eq!(report.sign_test.p_value_denominator, 256);
        assert_eq!(report.total_paired_episodes, 8_000);
        assert_eq!(report.total_control_ticks, 160_000_000);
    }

    #[test]
    fn unknown_and_censored_outcomes_stay_in_denominators_and_block_confirmation() {
        let mut clusters = (0..8).map(|id| cluster(id, 5, 8)).collect::<Vec<_>>();
        clusters[0].candidate.unknown = 1;
        clusters[0].candidate.task_failures -= 1;
        clusters[1].baseline.censored = 1;
        clusters[1].baseline.task_failures -= 1;
        let report = compare_paired_clusters(&comparison(clusters)).unwrap();
        assert_eq!(report.disposition, ComparisonDisposition::Inconclusive);
        assert_eq!(report.candidate_unknown, 1);
        assert_eq!(report.baseline_censored, 1);
        assert_eq!(report.total_paired_episodes, 80);
    }

    #[test]
    fn any_per_cluster_critical_safety_regression_blocks_directional_claim() {
        let mut clusters = (0..8).map(|id| cluster(id, 5, 9)).collect::<Vec<_>>();
        clusters[3].candidate.critical_safety_violations = 1;
        let report = compare_paired_clusters(&comparison(clusters)).unwrap();
        assert_eq!(
            report.disposition,
            ComparisonDisposition::SafetyRegressionDetected
        );
        assert_eq!(report.critical_safety_regression_clusters, 1);
    }

    #[test]
    fn any_candidate_critical_violation_blocks_safe_improvement_even_if_below_baseline() {
        let mut clusters = (0..8).map(|id| cluster(id, 5, 9)).collect::<Vec<_>>();
        clusters[3].baseline.critical_safety_violations = 2;
        clusters[3].candidate.critical_safety_violations = 1;
        let report = compare_paired_clusters(&comparison(clusters)).unwrap();
        assert_eq!(report.critical_safety_regression_clusters, 0);
        assert_eq!(report.baseline_critical_safety_violations, 2);
        assert_eq!(report.candidate_critical_safety_violations, 1);
        assert_eq!(
            report.disposition,
            ComparisonDisposition::CandidateCriticalSafetyViolationObserved
        );
    }

    #[test]
    fn development_split_never_produces_confirmatory_improvement() {
        let mut spec = comparison((0..8).map(|id| cluster(id, 5, 9)).collect());
        spec.scope.split_kind = SplitKind::Development;
        let report = compare_paired_clusters(&spec).unwrap();
        assert_eq!(report.disposition, ComparisonDisposition::Inconclusive);
    }

    #[test]
    fn unknown_independence_receipt_prevents_confirmatory_improvement() {
        let mut spec = comparison((0..8).map(|id| cluster(id, 5, 9)).collect());
        spec.clusters[2].independence_disposition = EvidenceDisposition::Unknown;
        let report = compare_paired_clusters(&spec).unwrap();
        assert_eq!(report.disposition, ComparisonDisposition::Inconclusive);
        assert_eq!(report.qualified_independent_cluster_count, 7);
    }

    #[test]
    fn rejected_independence_receipt_rejects_comparison() {
        let mut spec = comparison((0..8).map(|id| cluster(id, 5, 9)).collect());
        spec.clusters[2].independence_disposition = EvidenceDisposition::Rejected;
        let report = compare_paired_clusters(&spec).unwrap();
        assert_eq!(report.disposition, ComparisonDisposition::Rejected);
    }

    #[test]
    fn repeated_confirmatory_comparisons_receive_bonferroni_alpha() {
        let mut spec = comparison((0..8).map(|id| cluster(id, 5, 9)).collect());
        spec.scope.confirmatory_comparison_count = 13;
        let report = compare_paired_clusters(&spec).unwrap();
        assert_eq!(report.sign_test.adjusted_alpha_ppm, 3_846);
        assert_eq!(
            report.disposition,
            ComparisonDisposition::NoStatisticallyDemonstratedDirectionalImprovement
        );
    }

    #[test]
    fn same_model_ablation_rejects_model_mismatch() {
        let mut spec = comparison((0..8).map(|id| cluster(id, 5, 9)).collect());
        spec.candidate_arm.model_digest = "different-model".into();
        assert_eq!(
            compare_paired_clusters(&spec),
            Err(ComparisonError::CausalAblationModelMismatch)
        );
    }

    #[test]
    fn best_available_product_mode_allows_model_mismatch_but_keeps_common_scope() {
        let mut spec = comparison((0..8).map(|id| cluster(id, 5, 9)).collect());
        spec.scope.mode = ComparisonMode::BestAvailableProduct;
        spec.candidate_arm.model_digest = "different-model".into();
        let report = compare_paired_clusters(&spec).unwrap();
        assert_eq!(
            report.disposition,
            ComparisonDisposition::QualifiedDirectionalImprovement
        );
    }

    #[test]
    fn duplicate_clusters_are_rejected() {
        let mut clusters = (0..8).map(|id| cluster(id, 5, 9)).collect::<Vec<_>>();
        clusters[1].cluster_id = clusters[0].cluster_id.clone();
        assert!(matches!(
            compare_paired_clusters(&comparison(clusters)),
            Err(ComparisonError::DuplicateOrEmptyClusterId(_))
        ));
    }

    #[test]
    fn episode_categories_must_sum_to_all_scheduled_episodes() {
        let mut clusters = (0..8).map(|id| cluster(id, 5, 9)).collect::<Vec<_>>();
        clusters[0].candidate.task_failures -= 1;
        assert!(matches!(
            compare_paired_clusters(&comparison(clusters)),
            Err(ComparisonError::EpisodeCategoriesDoNotSum { .. })
        ));
    }

    #[test]
    fn paired_arms_must_share_episode_denominator() {
        let mut clusters = (0..8).map(|id| cluster(id, 5, 9)).collect::<Vec<_>>();
        clusters[0].candidate.episodes = 9;
        clusters[0].candidate.task_failures = 0;
        assert!(matches!(
            compare_paired_clusters(&comparison(clusters)),
            Err(ComparisonError::PairedEpisodeCountMismatch(_))
        ));
    }

    #[test]
    fn unsafe_successes_are_not_safe_successes() {
        let mut clusters = (0..8).map(|id| cluster(id, 5, 9)).collect::<Vec<_>>();
        clusters[0].candidate = tally(10, 5, 4, 1);
        let report = compare_paired_clusters(&comparison(clusters)).unwrap();
        assert_eq!(report.clusters[0].candidate_success_but_unsafe, 4);
        assert_eq!(report.clusters[0].candidate_safe_success_ppm, 500_000);
    }

    #[test]
    fn cluster_result_order_is_canonical() {
        let mut a = (0..8).map(|id| cluster(id, 5, 9)).collect::<Vec<_>>();
        let mut b = a.clone();
        b.reverse();
        let report_a = compare_paired_clusters(&comparison(a.clone())).unwrap();
        let report_b = compare_paired_clusters(&comparison(b)).unwrap();
        assert_eq!(report_a, report_b);
        a.rotate_left(3);
        assert_eq!(compare_paired_clusters(&comparison(a)).unwrap(), report_a);
    }

    #[test]
    fn exact_test_refuses_more_than_one_hundred_clusters() {
        assert!(matches!(
            exact_sign_test(101, 0, 0, 50_000),
            Err(ComparisonError::TooManyClusters { .. })
        ));
    }
}
