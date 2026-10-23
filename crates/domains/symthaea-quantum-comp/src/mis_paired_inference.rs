//! Paired, dependency-free inference for quantum-informed optimization studies.
//!
//! These helpers are more explicit than the lightweight normal-approximation summary:
//! they expose a distribution-free order-statistic interval for a population median and
//! a paired sign-flip randomization test for the mean paired effect. They do not repair
//! pseudoreplication: the input rows must be independent units (for graph benchmarks,
//! normally one prespecified aggregate per graph), not nested shots treated as graphs.
//!
//! The sign-flip test is exact under its sharp-null/exchangeability assumptions. It is
//! not interchangeable with the exact sign test, and it does not adjust for multiple
//! comparisons. Monte Carlo p-values use the plus-one correction and a deterministic
//! seeded generator. Neither helper establishes quantum advantage by itself.

use core::fmt;
use crate::rng::XorShift64;

/// Highest exact sign-flip enumeration size accepted by the default implementation.
pub const MAX_EXACT_SIGN_FLIP_PAIRS: usize = 20;

/// Configuration for a paired mean-effect sign-flip randomization test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MisSignFlipConfig {
    /// Enumerate every sign assignment when the number of pairs is at most this value.
    /// Must be in 1..=MAX_EXACT_SIGN_FLIP_PAIRS.
    pub exact_max_pairs: usize,
    /// Number of seeded random sign assignments when exact enumeration is too large.
    /// Must be positive even when a particular call takes the exact path.
    pub monte_carlo_draws: usize,
    /// Seed for the deterministic pseudo-random sign assignments.
    pub seed: u64,
}

impl Default for MisSignFlipConfig {
    fn default() -> Self {
        Self {
            exact_max_pairs: MAX_EXACT_SIGN_FLIP_PAIRS,
            monte_carlo_draws: 100_000,
            seed: 0x5143_4f4d_5053_5441,
        }
    }
}

/// Method used to calculate the sign-flip p-value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MisSignFlipMethod {
    /// Every one of the 2^n paired sign assignments was enumerated.
    ExactEnumeration,
    /// Random sign assignments were drawn using the recorded seed.
    MonteCarlo { draws: usize, seed: u64 },
}

/// Result of a two-sided paired sign-flip test of the mean difference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MisSignFlipResult {
    /// Number of independent paired units supplied to the test.
    pub pair_count: usize,
    /// Number of tested sign assignments at least as extreme as the observed statistic.
    pub extreme_assignments: usize,
    /// Total number of sign assignments evaluated.
    pub permutations_evaluated: usize,
    /// Two-sided p-value; Monte Carlo mode uses the plus-one correction.
    pub p_value_two_sided: f64,
    /// Exact enumeration or Monte Carlo metadata, including seed and draw count.
    pub method: MisSignFlipMethod,
    /// Approximate Monte Carlo standard error of the uncorrected exceedance proportion.
    /// None for exact enumeration.
    pub monte_carlo_standard_error: Option<f64>,
}

/// Distribution-free order-statistic interval for a population median.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MisMedianInterval {
    /// Number of independent observations supplied.
    pub sample_count: usize,
    /// Requested minimum confidence coverage.
    pub requested_confidence: f64,
    /// Lower order statistic, or None when the sample is too small for finite bounds.
    pub lower: Option<f64>,
    /// Upper order statistic, or None when the sample is too small for finite bounds.
    pub upper: Option<f64>,
    /// Selected one-based order statistic k for [X_(k), X_(n-k+1)].
    pub order_statistic_k: Option<usize>,
    /// Distribution-free coverage of the selected interval, or None if unavailable.
    pub achieved_coverage: Option<f64>,
}

/// Settings for hierarchical, graph-clustered bootstrap of a paired mean difference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MisClusterBootstrapConfig {
    /// Number of hierarchical bootstrap replicates; must be positive.
    pub replicates: usize,
    /// Percentile interval confidence, strictly between zero and one.
    pub confidence: f64,
    /// Seed for deterministic cluster and within-cluster resampling.
    pub seed: u64,
}

/// Percentile interval from a graph-clustered paired bootstrap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MisClusterBootstrapResult {
    /// Number of independent graph/instance clusters.
    pub cluster_count: usize,
    /// Total paired observations across clusters.
    pub observation_count: usize,
    /// Number of bootstrap draws used.
    pub replicates: usize,
    /// Requested interval confidence.
    pub confidence: f64,
    /// Equal-cluster-weighted point estimate of the mean paired difference.
    pub point_estimate: f64,
    /// Lower percentile bound from the bootstrap distribution.
    pub lower: f64,
    /// Upper percentile bound from the bootstrap distribution.
    pub upper: f64,
    /// Seed used for deterministic resampling.
    pub seed: u64,
}

/// Invalid inference input or configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisInferenceError {
    /// A paired test or median interval needs at least one observation.
    EmptySample,
    /// A sample contains NaN or infinity at the indicated zero-based index.
    NonFiniteObservation { index: usize },
    /// Confidence must be finite and strictly between zero and one.
    InvalidConfidence,
    /// Exact-enumeration limit is outside 1..=MAX_EXACT_SIGN_FLIP_PAIRS.
    InvalidExactLimit,
    /// Monte Carlo mode requires a positive draw count.
    ZeroMonteCarloDraws,
    /// A cluster bootstrap needs at least two independent clusters.
    InsufficientClusters { cluster_count: usize },
    /// A graph cluster has no paired observations.
    EmptyCluster { cluster_index: usize },
    /// A clustered observation is NaN or infinity.
    NonFiniteClusterObservation { cluster_index: usize, observation_index: usize },
    /// Cluster bootstrap requires a positive replicate count.
    ZeroBootstrapReplicates,
    /// A p-value is NaN, infinite, below zero, or greater than one at the indicated index.
    InvalidPValue { index: usize },
}

impl fmt::Display for MisInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySample => write!(f, "sample must not be empty"),
            Self::NonFiniteObservation { index } => {
                write!(f, "observation at index {index} is not finite")
            }
            Self::InvalidConfidence => {
                write!(f, "confidence must be finite and strictly between zero and one")
            }
            Self::InvalidExactLimit => {
                write!(f, "exact limit must be in 1..={MAX_EXACT_SIGN_FLIP_PAIRS}")
            }
            Self::ZeroMonteCarloDraws => write!(f, "Monte Carlo draw count must be positive"),
            Self::InsufficientClusters { cluster_count } => write!(f, "cluster bootstrap needs at least two clusters, got {cluster_count}"),
            Self::EmptyCluster { cluster_index } => write!(f, "cluster {cluster_index} has no paired observations"),
            Self::NonFiniteClusterObservation { cluster_index, observation_index } => write!(f, "cluster {cluster_index} observation {observation_index} is not finite"),
            Self::ZeroBootstrapReplicates => write!(f, "cluster bootstrap replicate count must be positive"),
            Self::InvalidPValue { index } => write!(f, "p-value at index {index} must be finite and in [0, 1]"),
        }
    }
}

impl std::error::Error for MisInferenceError {}

/// Tests whether the mean paired difference is unusually far from zero under sign flips.
///
/// Positive values mean the first policy is larger than the second. The statistic is
/// the absolute mean difference; the implementation rescales all values by their
/// maximum absolute magnitude before summing to avoid overflow. Sign flips are exact
/// under a sharp null that makes pairwise signs exchangeable (commonly justified by a
/// paired randomized assignment or a symmetric null distribution). The test assumes
/// independent pairs. For graph benchmarks with repeated runs per graph, aggregate by
/// graph according to a preregistered rule or use a cluster-aware analysis first.
///
/// The result is a two-sided test. It is intentionally separate from the exact sign test:
/// this procedure uses effect magnitude, while the sign test uses only win/loss signs.
/// Multiple-comparison correction is not applied here.
///
/// For pair counts at or below config.exact_max_pairs, all 2^n sign assignments are
/// enumerated. Above it, config.monte_carlo_draws assignments are sampled with the
/// configured seed and the p-value is (extreme + 1) / (draws + 1).
pub fn paired_sign_flip_test(
    differences: &[f64],
    config: MisSignFlipConfig,
) -> Result<MisSignFlipResult, MisInferenceError> {
    if differences.is_empty() {
        return Err(MisInferenceError::EmptySample);
    }
    if !(1..=MAX_EXACT_SIGN_FLIP_PAIRS).contains(&config.exact_max_pairs) {
        return Err(MisInferenceError::InvalidExactLimit);
    }
    if config.monte_carlo_draws == 0 {
        return Err(MisInferenceError::ZeroMonteCarloDraws);
    }
    for (index, value) in differences.iter().enumerate() {
        if !value.is_finite() {
            return Err(MisInferenceError::NonFiniteObservation { index });
        }
    }

    // Scaling does not change the sign-flip ordering of the absolute mean statistic.
    let scale = differences
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let scaled: Vec<f64> = if scale == 0.0 {
        vec![0.0; differences.len()]
    } else {
        differences.iter().map(|value| value / scale).collect()
    };
    let observed = scaled.iter().sum::<f64>().abs();
    let tolerance = 32.0 * f64::EPSILON * (scaled.len() as f64).max(1.0);
    let threshold = (observed - tolerance).max(0.0);

    if differences.len() <= config.exact_max_pairs {
        let assignments = 1usize << differences.len();
        let mut extreme = 0usize;
        for mask in 0..assignments {
            let mut signed_sum = 0.0;
            for (index, value) in scaled.iter().enumerate() {
                if mask & (1usize << index) == 0 {
                    signed_sum += value;
                } else {
                    signed_sum -= value;
                }
            }
            if signed_sum.abs() >= threshold {
                extreme += 1;
            }
        }
        return Ok(MisSignFlipResult {
            pair_count: differences.len(),
            extreme_assignments: extreme,
            permutations_evaluated: assignments,
            p_value_two_sided: extreme as f64 / assignments as f64,
            method: MisSignFlipMethod::ExactEnumeration,
            monte_carlo_standard_error: None,
        });
    }

    let mut rng = crate::rng::XorShift64::new(config.seed);
    let mut extreme = 0usize;
    for _ in 0..config.monte_carlo_draws {
        let mut signed_sum = 0.0;
        for value in &scaled {
            if rng.next_u64() & (1_u64 << 63) == 0 {
                signed_sum += value;
            } else {
                signed_sum -= value;
            }
        }
        if signed_sum.abs() >= threshold {
            extreme += 1;
        }
    }
    let rate = extreme as f64 / config.monte_carlo_draws as f64;
    let p_value = (extreme as f64 + 1.0) / (config.monte_carlo_draws as f64 + 1.0);
    Ok(MisSignFlipResult {
        pair_count: differences.len(),
        extreme_assignments: extreme,
        permutations_evaluated: config.monte_carlo_draws,
        p_value_two_sided: p_value,
        method: MisSignFlipMethod::MonteCarlo {
            draws: config.monte_carlo_draws,
            seed: config.seed,
        },
        monte_carlo_standard_error: Some(
            (rate * (1.0 - rate) / config.monte_carlo_draws as f64).sqrt(),
        ),
    })
}

/// Builds a central order-statistic confidence interval for a population median.
///
/// For sorted observations x, the candidate interval is
/// [x[k - 1], x[n - k]], with one-based k. Under independent sampling from a continuous
/// population, its coverage is 1 - 2 * P(Binomial(n, 0.5) <= k - 1); ties make the
/// interval conservative. The routine chooses the narrowest interval whose coverage
/// reaches the requested confidence. If the available sample cannot support that
/// confidence with finite order-statistic bounds, the returned lower/upper/k/coverage
/// are all None rather than implying unattainable precision.
///
/// This is a confidence interval for the population median, not a confidence interval
/// for the mean and not a substitute for cluster-aware uncertainty estimation.
pub fn median_order_statistic_interval(
    observations: &[f64],
    confidence: f64,
) -> Result<MisMedianInterval, MisInferenceError> {
    if observations.is_empty() {
        return Err(MisInferenceError::EmptySample);
    }
    if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0 {
        return Err(MisInferenceError::InvalidConfidence);
    }
    for (index, value) in observations.iter().enumerate() {
        if !value.is_finite() {
            return Err(MisInferenceError::NonFiniteObservation { index });
        }
    }

    let mut sorted = observations.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let mut log_probability = -(n as f64) * core::f64::consts::LN_2;
    let mut lower_tail = 0.0_f64;
    let mut selected: Option<(usize, f64)> = None;

    for j in 0..=(n / 2) {
        let k = j + 1;
        if k > n - k + 1 {
            break;
        }
        if j > 0 {
            log_probability += ((n - j + 1) as f64).ln() - (j as f64).ln();
        }
        lower_tail += log_probability.exp();
        let coverage = (1.0 - 2.0 * lower_tail).clamp(0.0, 1.0);
        if coverage + 8.0 * f64::EPSILON >= confidence {
            selected = Some((k, coverage));
        } else {
            // Coverage decreases as k increases, so no later order statistic can qualify.
            break;
        }
    }

    let (lower, upper, k, achieved_coverage) = match selected {
        Some((k, coverage)) => (
            Some(sorted[k - 1]),
            Some(sorted[n - k]),
            Some(k),
            Some(coverage),
        ),
        None => (None, None, None, None),
    };
    Ok(MisMedianInterval {
        sample_count: n,
        requested_confidence: confidence,
        lower,
        upper,
        order_statistic_k: k,
        achieved_coverage,
    })
}

/// Apply the Holm-Bonferroni step-down family-wise-error correction.
///
/// Input p-values must already come from valid, predeclared tests. The returned vector
/// is in the same order as the input and contains adjusted p-values in [0, 1]. Empty
/// input returns an empty vector. Ties are resolved stably by input index for
/// deterministic output. Holm's procedure controls family-wise error under arbitrary
/// dependence provided the underlying individual p-values are valid; it does not fix
/// post-hoc selection of the comparison family or invalid/exchangeability-violating tests.
pub fn holm_bonferroni_adjust(p_values: &[f64]) -> Result<Vec<f64>, MisInferenceError> {
    for (index, p_value) in p_values.iter().enumerate() {
        if !p_value.is_finite() || !(0.0..=1.0).contains(p_value) {
            return Err(MisInferenceError::InvalidPValue { index });
        }
    }
    let mut ordered: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    ordered.sort_by(|left, right| {
        left.1.total_cmp(&right.1).then_with(|| left.0.cmp(&right.0))
    });

    let mut adjusted = vec![0.0_f64; p_values.len()];
    let mut running_max: f64 = 0.0;
    let family_size = p_values.len() as f64;
    for (rank, (original_index, p_value)) in ordered.into_iter().enumerate() {
        let step_adjusted = ((family_size - rank as f64) * p_value).min(1.0);
        running_max = running_max.max(step_adjusted);
        adjusted[original_index] = running_max;
    }
    Ok(adjusted)
}

/// Compute an equal-cluster-weighted mean without summing unscaled extreme values.
fn stable_mean(values: &[f64]) -> f64 {
    let scale = values.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    let normalized_mean = (values.iter().map(|value| value / scale).sum::<f64>()
        / values.len() as f64)
        .clamp(-1.0, 1.0);
    scale * normalized_mean
}

/// Linear-interpolated percentile from an ascending sample. Weighted interpolation avoids
/// subtracting opposite-sign extreme finite values, which could overflow.
fn percentile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;
    let fraction = position - lower_index as f64;
    sorted[lower_index] * (1.0 - fraction) + sorted[upper_index] * fraction
}

/// Estimate an equal-graph-weighted mean paired effect with a hierarchical bootstrap.
///
/// The outer stage samples independent graph clusters with replacement. For each sampled
/// graph, the inner stage samples that graph's paired repetition-level differences with
/// replacement, preserving the original number of repetitions for that graph. Cluster
/// means are weighted equally in each bootstrap replicate, so a graph with many seeds
/// does not count as many independent graphs.
///
/// This is an approximate percentile bootstrap, not an exact finite-sample confidence
/// procedure. It assumes graphs are independent sampling units and that within-graph
/// paired repetitions represent the intended resampling population. Pre-register the
/// graph-level estimand and use enough graphs/replicates for the scientific question.
/// It does not adjust for multiple comparisons and cannot repair leakage between graph
/// families.
pub fn hierarchical_cluster_bootstrap_mean(
    clusters: &[Vec<f64>],
    config: MisClusterBootstrapConfig,
) -> Result<MisClusterBootstrapResult, MisInferenceError> {
    if clusters.is_empty() {
        return Err(MisInferenceError::EmptySample);
    }
    if clusters.len() < 2 {
        return Err(MisInferenceError::InsufficientClusters {
            cluster_count: clusters.len(),
        });
    }
    if config.replicates == 0 {
        return Err(MisInferenceError::ZeroBootstrapReplicates);
    }
    if !config.confidence.is_finite() || config.confidence <= 0.0 || config.confidence >= 1.0 {
        return Err(MisInferenceError::InvalidConfidence);
    }

    let mut observation_count = 0usize;
    let mut cluster_means = Vec::with_capacity(clusters.len());
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        if cluster.is_empty() {
            return Err(MisInferenceError::EmptyCluster { cluster_index });
        }
        for (observation_index, value) in cluster.iter().enumerate() {
            if !value.is_finite() {
                return Err(MisInferenceError::NonFiniteClusterObservation {
                    cluster_index,
                    observation_index,
                });
            }
        }
        observation_count += cluster.len();
        cluster_means.push(stable_mean(cluster));
    }
    let point_estimate = stable_mean(&cluster_means);

    let mut rng = XorShift64::new(config.seed);
    let mut bootstrap_estimates = Vec::with_capacity(config.replicates);
    let mut resampled_cluster_means = Vec::with_capacity(clusters.len());
    for _ in 0..config.replicates {
        resampled_cluster_means.clear();
        for _ in 0..clusters.len() {
            let cluster_index = rng.next_usize(clusters.len()).unwrap_or(0);
            let cluster = &clusters[cluster_index];
            let mut within_sample = Vec::with_capacity(cluster.len());
            for _ in 0..cluster.len() {
                let observation_index = rng.next_usize(cluster.len()).unwrap_or(0);
                within_sample.push(cluster[observation_index]);
            }
            resampled_cluster_means.push(stable_mean(&within_sample));
        }
        bootstrap_estimates.push(stable_mean(&resampled_cluster_means));
    }
    bootstrap_estimates.sort_by(f64::total_cmp);

    let alpha = (1.0 - config.confidence) / 2.0;
    Ok(MisClusterBootstrapResult {
        cluster_count: clusters.len(),
        observation_count,
        replicates: config.replicates,
        confidence: config.confidence,
        point_estimate,
        lower: percentile(&bootstrap_estimates, alpha),
        upper: percentile(&bootstrap_estimates, 1.0 - alpha),
        seed: config.seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_sign_flip_matches_hand_calculated_three_pair_example() {
        let result = paired_sign_flip_test(
            &[1.0, 1.0, 1.0],
            MisSignFlipConfig {
                exact_max_pairs: 20,
                monte_carlo_draws: 1,
                seed: 42,
            },
        )
        .unwrap();
        assert_eq!(result.method, MisSignFlipMethod::ExactEnumeration);
        assert_eq!(result.permutations_evaluated, 8);
        assert_eq!(result.extreme_assignments, 2);
        assert!((result.p_value_two_sided - 0.25).abs() < 1e-12);
        assert_eq!(result.monte_carlo_standard_error, None);
    }

    #[test]
    fn exact_sign_flip_is_one_for_zero_effects_and_handles_large_finite_values() {
        let zero = paired_sign_flip_test(&[0.0, 0.0], MisSignFlipConfig::default()).unwrap();
        assert_eq!(zero.p_value_two_sided, 1.0);

        let large = paired_sign_flip_test(
            &[f64::MAX, f64::MAX],
            MisSignFlipConfig::default(),
        )
        .unwrap();
        assert!(large.p_value_two_sided.is_finite());
        assert!((large.p_value_two_sided - 0.5).abs() < 1e-12);
    }

    #[test]
    fn monte_carlo_sign_flip_is_replayable_and_uses_plus_one() {
        let config = MisSignFlipConfig {
            exact_max_pairs: 2,
            monte_carlo_draws: 2_000,
            seed: 991,
        };
        let differences = [1.0, 0.5, -0.3, 0.9, -0.1];
        let a = paired_sign_flip_test(&differences, config).unwrap();
        let b = paired_sign_flip_test(&differences, config).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.permutations_evaluated, 2_000);
        assert!(a.p_value_two_sided > 0.0 && a.p_value_two_sided <= 1.0);
        assert!(a.monte_carlo_standard_error.is_some());
    }

    #[test]
    fn sign_flip_rejects_non_finite_and_invalid_configuration() {
        assert_eq!(
            paired_sign_flip_test(&[], MisSignFlipConfig::default()),
            Err(MisInferenceError::EmptySample)
        );
        assert_eq!(
            paired_sign_flip_test(
                &[1.0, f64::NAN],
                MisSignFlipConfig::default()
            ),
            Err(MisInferenceError::NonFiniteObservation { index: 1 })
        );
        assert_eq!(
            paired_sign_flip_test(
                &[1.0],
                MisSignFlipConfig {
                    exact_max_pairs: 21,
                    ..MisSignFlipConfig::default()
                }
            ),
            Err(MisInferenceError::InvalidExactLimit)
        );
    }

    #[test]
    fn median_interval_has_hand_calculated_coverage_and_width() {
        let result = median_order_statistic_interval(&[6.0, 1.0, 2.0, 5.0, 4.0, 3.0], 0.95)
            .unwrap();
        assert_eq!(result.lower, Some(1.0));
        assert_eq!(result.upper, Some(6.0));
        assert_eq!(result.order_statistic_k, Some(1));
        assert!((result.achieved_coverage.unwrap() - 0.96875).abs() < 1e-12);
    }

    #[test]
    fn median_interval_reports_insufficient_finite_coverage_honestly() {
        let result = median_order_statistic_interval(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.95)
            .unwrap();
        assert_eq!(result.lower, None);
        assert_eq!(result.upper, None);
        assert_eq!(result.order_statistic_k, None);
        assert_eq!(result.achieved_coverage, None);
    }

    #[test]
    fn median_interval_rejects_bad_confidence_and_non_finite_values() {
        assert_eq!(
            median_order_statistic_interval(&[1.0, 2.0], 1.0),
            Err(MisInferenceError::InvalidConfidence)
        );
        assert_eq!(
            median_order_statistic_interval(&[1.0, f64::INFINITY], 0.95),
            Err(MisInferenceError::NonFiniteObservation { index: 1 })
        );
    }

    #[test]
    fn holm_bonferroni_matches_hand_computed_family_and_preserves_input_order() {
        let adjusted = holm_bonferroni_adjust(&[0.04, 0.01, 0.03, 1.0]).unwrap();
        assert_eq!(adjusted, vec![0.09, 0.04, 0.09, 1.0]);
        assert_eq!(holm_bonferroni_adjust(&[]).unwrap(), Vec::<f64>::new());

        let ties = holm_bonferroni_adjust(&[0.01, 0.01, 0.04]).unwrap();
        assert_eq!(ties, vec![0.03, 0.03, 0.04]);
    }

    #[test]
    fn holm_bonferroni_rejects_invalid_p_values_and_keeps_monotone_sorted_adjustments() {
        for bad in [-0.01, 1.01, f64::NAN, f64::INFINITY] {
            assert_eq!(
                holm_bonferroni_adjust(&[0.05, bad, 0.2]),
                Err(MisInferenceError::InvalidPValue { index: 1 })
            );
        }
        let raw = [0.2, 0.001, 0.05, 0.03, 0.8];
        let adjusted = holm_bonferroni_adjust(&raw).unwrap();
        assert!(adjusted.iter().all(|p| (0.0..=1.0).contains(p)));
        let mut pairs: Vec<(f64, f64)> = raw.iter().copied().zip(adjusted.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert!(pairs.windows(2).all(|window| window[0].1 <= window[1].1));
    }

    #[test]
    fn hierarchical_cluster_bootstrap_is_deterministic_and_graph_weighted() {
        let clusters = vec![vec![10.0; 16], vec![0.0]];
        let config = MisClusterBootstrapConfig {
            replicates: 2_000,
            confidence: 0.95,
            seed: 991,
        };
        let first = hierarchical_cluster_bootstrap_mean(&clusters, config).unwrap();
        let second = hierarchical_cluster_bootstrap_mean(&clusters, config).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.cluster_count, 2);
        assert_eq!(first.observation_count, 17);
        assert_eq!(first.replicates, 2_000);
        assert_eq!(first.point_estimate, 5.0);
        assert_ne!(first.point_estimate, clusters.iter().flatten().sum::<f64>() / 17.0);
        assert!(first.lower <= first.point_estimate);
        assert!(first.point_estimate <= first.upper);
    }

    #[test]
    fn hierarchical_cluster_bootstrap_rejects_pseudoreplication_and_bad_rows() {
        let config = MisClusterBootstrapConfig {
            replicates: 50,
            confidence: 0.95,
            seed: 7,
        };
        assert_eq!(
            hierarchical_cluster_bootstrap_mean(&[], config),
            Err(MisInferenceError::EmptySample)
        );
        assert_eq!(
            hierarchical_cluster_bootstrap_mean(&[vec![1.0, 2.0]], config),
            Err(MisInferenceError::InsufficientClusters { cluster_count: 1 })
        );
        assert_eq!(
            hierarchical_cluster_bootstrap_mean(&[vec![1.0], vec![]], config),
            Err(MisInferenceError::EmptyCluster { cluster_index: 1 })
        );
        assert_eq!(
            hierarchical_cluster_bootstrap_mean(&[vec![1.0], vec![f64::NAN]], config),
            Err(MisInferenceError::NonFiniteClusterObservation {
                cluster_index: 1,
                observation_index: 0,
            })
        );
        assert_eq!(
            hierarchical_cluster_bootstrap_mean(
                &[vec![1.0], vec![2.0]],
                MisClusterBootstrapConfig { replicates: 0, ..config },
            ),
            Err(MisInferenceError::ZeroBootstrapReplicates)
        );
    }

    #[test]
    fn median_interval_coverage_matches_integer_binomial_reference_for_small_samples() {
        fn reference_coverage(n: usize, k: usize) -> f64 {
            // Independent integer binomial recurrence for P(Binomial(n, 0.5) <= k - 1).
            let mut coefficient = 1_u128;
            let mut lower_tail = 0_u128;
            for j in 0..k {
                if j > 0 {
                    coefficient = coefficient * (n - j + 1) as u128 / j as u128;
                }
                lower_tail += coefficient;
            }
            let all_assignments = (1_u128 << n) as f64;
            (1.0 - 2.0 * lower_tail as f64 / all_assignments).clamp(0.0, 1.0)
        }

        for n in 1..=32 {
            let ordered: Vec<f64> = (0..n).map(|i| i as f64).collect();
            for confidence in [0.50, 0.75, 0.90, 0.95, 0.99] {
                let result = median_order_statistic_interval(&ordered, confidence).unwrap();
                match result.order_statistic_k {
                    Some(k) => {
                        let expected = reference_coverage(n, k);
                        assert!(
                            (result.achieved_coverage.unwrap() - expected).abs() < 1e-12,
                            "coverage mismatch for n={n}, k={k}, confidence={confidence}"
                        );
                        assert!(
                            expected + 1e-12 >= confidence,
                            "reported finite interval undercovers for n={n}, k={k}"
                        );
                        if 2 * k + 1 <= n {
                            assert!(
                                reference_coverage(n, k + 1) + 1e-12 < confidence,
                                "interval is not the narrowest qualifying interval for n={n}, k={k}"
                            );
                        }
                    }
                    None => {
                        assert!(
                            reference_coverage(n, 1) + 1e-12 < confidence,
                            "finite order-statistic interval should be available for n={n}"
                        );
                        assert_eq!(result.lower, None);
                        assert_eq!(result.upper, None);
                        assert_eq!(result.achieved_coverage, None);
                    }
                }
            }
        }
    }

}
