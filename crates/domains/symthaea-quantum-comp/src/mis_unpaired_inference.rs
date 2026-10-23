//! Independent-sample inference for distribution-level MIS algorithm comparisons.
//!
//! Use these routines when two algorithms produce outcomes on independently sampled
//! graph sets and no exact workload pairing exists (for example, comparing a
//! deferred-decision random-regular graph procedure with a completed-graph solver).
//! Do not treat matching seed integers as evidence of paired workloads.
//!
//! The permutation routine uses the absolute Welch-studentized mean difference. Exact
//! assignment enumeration computes the full label-permutation reference distribution,
//! but is an exact randomization test only under the stronger exchangeability null.
//! Studentization supports asymptotic inference when group variances differ; it does
//! not make small-sample permutation inference exact under a mere equal-means null.
//!
//! The bootstrap resamples independently within each arm and produces a percentile
//! interval for the difference of means. Inputs should contain one independent
//! observation per graph (or a preregistered graph-level aggregate), not repeated
//! within-graph trials treated as new independent graphs. Multiplicity correction is
//! supplied separately by the Holm helper in mis_paired_inference.

use core::fmt;

use crate::rng::XorShift64;

/// Hard maximum pooled observation count for exact label-assignment enumeration.
pub const MAX_EXACT_UNPAIRED_TOTAL: usize = 16;

/// Configuration for a two-sided independent-sample permutation test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MisUnpairedPermutationConfig {
    /// Use exact label-assignment enumeration when total observations do not exceed this
    /// value. Must be in 2..=MAX_EXACT_UNPAIRED_TOTAL.
    pub exact_max_total: usize,
    /// Seeded Monte Carlo label permutations used when exact enumeration is too large.
    pub monte_carlo_draws: usize,
    /// Deterministic pseudo-random seed for Monte Carlo permutations.
    pub seed: u64,
}

impl Default for MisUnpairedPermutationConfig {
    fn default() -> Self {
        Self {
            exact_max_total: MAX_EXACT_UNPAIRED_TOTAL,
            monte_carlo_draws: 100_000,
            seed: 0x554e_5041_4952_4544,
        }
    }
}

/// Method used for the independent-sample permutation reference distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MisUnpairedPermutationMethod {
    /// Every allocation of the pooled observations to groups of the original sizes.
    ExactLabelEnumeration,
    /// Seeded Monte Carlo label assignments with the plus-one p-value correction.
    MonteCarlo { draws: usize, seed: u64 },
}

/// Summary from the independent-sample Welch-studentized permutation test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MisUnpairedPermutationResult {
    /// Number of independent observations in sample A.
    pub sample_count_a: usize,
    /// Number of independent observations in sample B.
    pub sample_count_b: usize,
    /// Observed mean of sample A.
    pub sample_mean_a: f64,
    /// Observed mean of sample B.
    pub sample_mean_b: f64,
    /// Estimated effect, mean(A) - mean(B), in the original units.
    pub mean_difference_a_minus_b: f64,
    /// Absolute Welch-studentized statistic observed on the actual assignment.
    pub absolute_welch_statistic: f64,
    /// Assignments with statistic at least as extreme as observed.
    pub extreme_assignments: usize,
    /// Number of label assignments evaluated (not Monte Carlo uncertainty draws).
    pub assignments_evaluated: usize,
    /// Two-sided p-value for the chosen reference distribution.
    pub p_value_two_sided: f64,
    /// Whether the reference distribution was enumerated or sampled.
    pub method: MisUnpairedPermutationMethod,
    /// Approximate Monte Carlo standard error of the uncorrected exceedance proportion.
    /// None for exact label enumeration.
    pub monte_carlo_standard_error: Option<f64>,
}

/// Configuration for independent percentile-bootstrap inference on a mean difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MisUnpairedBootstrapConfig {
    /// Number of independent bootstrap replicates; must be positive.
    pub replicates: usize,
    /// Deterministic pseudo-random seed.
    pub seed: u64,
    /// Requested percentile-interval confidence, validated by the routine.
    pub confidence_millionths: u32,
}

impl Default for MisUnpairedBootstrapConfig {
    fn default() -> Self {
        Self {
            replicates: 20_000,
            seed: 0x4253_5452_4150_3031,
            confidence_millionths: 95_000,
        }
    }
}

/// Independent bootstrap percentile interval for a mean difference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MisUnpairedBootstrapResult {
    /// Number of independent observations in sample A.
    pub sample_count_a: usize,
    /// Number of independent observations in sample B.
    pub sample_count_b: usize,
    /// Observed mean of sample A.
    pub sample_mean_a: f64,
    /// Observed mean of sample B.
    pub sample_mean_b: f64,
    /// Observed effect, mean(A) - mean(B), in the original units.
    pub mean_difference_a_minus_b: f64,
    /// Lower endpoint of the independent percentile-bootstrap interval.
    pub lower: f64,
    /// Upper endpoint of the independent percentile-bootstrap interval.
    pub upper: f64,
    /// Sample standard deviation of bootstrap mean differences.
    pub bootstrap_standard_error: f64,
    /// Requested interval confidence.
    pub confidence: f64,
    /// Number of replicates.
    pub replicates: usize,
    /// Seed used by the deterministic bootstrap.
    pub seed: u64,
}

/// Invalid independent-sample inference input or configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisUnpairedInferenceError {
    /// Either sample has fewer than two independent observations.
    InsufficientSample { group: char, count: usize },
    /// A non-finite observation was found at the specified zero-based group index.
    NonFiniteObservation { group: char, index: usize },
    /// Exact enumeration limit is outside 2..=MAX_EXACT_UNPAIRED_TOTAL.
    InvalidExactLimit,
    /// Monte Carlo mode requires a positive draw count.
    ZeroMonteCarloDraws,
    /// Bootstrap mode requires a positive replicate count.
    ZeroBootstrapReplicates,
    /// Confidence must be finite and strictly between zero and one.
    InvalidConfidence,
    /// The requested effect or interval endpoints cannot be represented as a finite f64.
    NonFiniteStatistic,
}

impl fmt::Display for MisUnpairedInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientSample { group, count } => {
                write!(f, "sample {group} needs at least two observations, got {count}")
            }
            Self::NonFiniteObservation { group, index } => {
                write!(f, "sample {group} observation {index} is not finite")
            }
            Self::InvalidExactLimit => write!(
                f,
                "exact pooled size must be in 2..={MAX_EXACT_UNPAIRED_TOTAL}"
            ),
            Self::ZeroMonteCarloDraws => write!(f, "Monte Carlo draw count must be positive"),
            Self::ZeroBootstrapReplicates => write!(f, "bootstrap replicate count must be positive"),
            Self::InvalidConfidence => {
                write!(f, "confidence must be finite and strictly between zero and one")
            }
            Self::NonFiniteStatistic => {
                write!(f, "the requested statistic exceeds finite f64 range")
            }
        }
    }
}

impl std::error::Error for MisUnpairedInferenceError {}

fn validated_scale(
    a: &[f64],
    b: &[f64],
) -> Result<f64, MisUnpairedInferenceError> {
    for (group, values) in [('A', a), ('B', b)] {
        if values.len() < 2 {
            return Err(MisUnpairedInferenceError::InsufficientSample {
                group,
                count: values.len(),
            });
        }
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(MisUnpairedInferenceError::NonFiniteObservation { group, index });
            }
        }
    }
    Ok(a.iter()
        .chain(b.iter())
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max))
}

fn scaled_values(values: &[f64], scale: f64) -> Vec<f64> {
    if scale == 0.0 {
        vec![0.0; values.len()]
    } else {
        values.iter().map(|value| value / scale).collect()
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_variance(values: &[f64], avg: f64) -> f64 {
    values.iter().map(|value| (value - avg) * (value - avg)).sum::<f64>()
        / (values.len() - 1) as f64
}

fn mean_difference_scaled(a: &[f64], b: &[f64]) -> f64 {
    mean(a) - mean(b)
}

/// Welch-studentized mean difference. Data are scaled to [-1, 1] before this helper.
/// Infinite values occur only when the pooled group assignments yield zero estimated
/// standard error but a non-zero difference in means; such assignments are handled
/// deterministically in the permutation comparison.
fn welch_statistic_scaled(a: &[f64], b: &[f64]) -> f64 {
    let mean_a = mean(a);
    let mean_b = mean(b);
    let difference = mean_a - mean_b;
    let se_squared =
        sample_variance(a, mean_a) / a.len() as f64
        + sample_variance(b, mean_b) / b.len() as f64;
    if se_squared == 0.0 {
        if difference == 0.0 {
            0.0
        } else {
            difference.signum() * f64::INFINITY
        }
    } else {
        difference / se_squared.sqrt()
    }
}

fn random_index(rng: &mut XorShift64, upper: usize) -> usize {
    // Lemire's multiply-high mapping with rejection avoids modulo bias. Use the high
    // part of the XorShift64* output rather than weak low bits for bounded draws.
    let upper64 = upper as u64;
    let threshold = upper64.wrapping_neg() % upper64;
    loop {
        let product = u128::from(rng.next_u64()) * u128::from(upper64);
        let low = product as u64;
        if low >= threshold {
            return (product >> 64) as usize;
        }
    }
}

fn shuffle_in_place(values: &mut [f64], rng: &mut XorShift64) {
    for i in (1..values.len()).rev() {
        let j = random_index(rng, i + 1);
        values.swap(i, j);
    }
}

fn quantile_sorted(sorted: &[f64], probability: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;
    if lower_index == upper_index {
        sorted[lower_index]
    } else {
        let fraction = position - lower_index as f64;
        sorted[lower_index] * (1.0 - fraction) + sorted[upper_index] * fraction
    }
}

fn rescale_statistic(value: f64, scale: f64) -> Result<f64, MisUnpairedInferenceError> {
    let original = value * scale;
    if original.is_finite() {
        Ok(original)
    } else {
        Err(MisUnpairedInferenceError::NonFiniteStatistic)
    }
}

/// Compare two independent samples using a two-sided Welch-studentized permutation test.
///
/// The statistic is the absolute Welch t-like statistic for mean(A)-mean(B), recalculated
/// for every label assignment. If the pooled sample fits within the exact cap, every
/// fixed-size group allocation is enumerated. Otherwise, fixed-size labels are shuffled
/// using the configured seed; Monte Carlo p-values use the plus-one correction.
///
/// **Assumptions:** exact label enumeration is a finite exact reference calculation, but
/// yields an exact randomization p-value only under the sharp exchangeability/equal-distribution
/// null. Under unequal variances with only equal means, the Welch-studentized permutation
/// statistic is asymptotically more defensible than an unstudentized label permutation,
/// but is not an exact small-sample test. Groups should contain independent graph-level
/// observations. This routine does not match instances, correct family dependence, or
/// adjust for multiple comparisons.
///
/// For N small, use this alongside an independent bootstrap interval and report raw group
/// distributions and the observed effect. A p-value alone is not practical-value evidence.
pub fn unpaired_mean_permutation_test(
    sample_a: &[f64],
    sample_b: &[f64],
    config: MisUnpairedPermutationConfig,
) -> Result<MisUnpairedPermutationResult, MisUnpairedInferenceError> {
    let scale = validated_scale(sample_a, sample_b)?;
    if !(2..=MAX_EXACT_UNPAIRED_TOTAL).contains(&config.exact_max_total) {
        return Err(MisUnpairedInferenceError::InvalidExactLimit);
    }
    if config.monte_carlo_draws == 0 {
        return Err(MisUnpairedInferenceError::ZeroMonteCarloDraws);
    }

    let a = scaled_values(sample_a, scale);
    let b = scaled_values(sample_b, scale);
    let observed_scaled = mean_difference_scaled(&a, &b);
    let observed_statistic = welch_statistic_scaled(&a, &b);
    let abs_observed = observed_statistic.abs();
    let tolerance = 64.0 * f64::EPSILON * abs_observed.max(1.0);
    let threshold = (abs_observed - tolerance).max(0.0);
    let mean_a = rescale_statistic(mean(&a), scale)?;
    let mean_b = rescale_statistic(mean(&b), scale)?;
    let effect = rescale_statistic(observed_scaled, scale)?;

    let total = a.len() + b.len();
    let mut pooled = a;
    pooled.extend(b);

    if total <= config.exact_max_total {
        let total_assignments = 1usize << total;
        let mut extreme = 0usize;
        let mut evaluated = 0usize;
        let mut group_a = Vec::with_capacity(sample_a.len());
        let mut group_b = Vec::with_capacity(sample_b.len());

        for mask in 0..total_assignments {
            if mask.count_ones() as usize != sample_a.len() {
                continue;
            }
            group_a.clear();
            group_b.clear();
            for (index, value) in pooled.iter().enumerate() {
                if mask & (1usize << index) != 0 {
                    group_a.push(*value);
                } else {
                    group_b.push(*value);
                }
            }
            let statistic = welch_statistic_scaled(&group_a, &group_b).abs();
            if statistic >= threshold {
                extreme += 1;
            }
            evaluated += 1;
        }

        return Ok(MisUnpairedPermutationResult {
            sample_count_a: sample_a.len(),
            sample_count_b: sample_b.len(),
            sample_mean_a: mean_a,
            sample_mean_b: mean_b,
            mean_difference_a_minus_b: effect,
            absolute_welch_statistic: abs_observed,
            extreme_assignments: extreme,
            assignments_evaluated: evaluated,
            p_value_two_sided: extreme as f64 / evaluated as f64,
            method: MisUnpairedPermutationMethod::ExactLabelEnumeration,
            monte_carlo_standard_error: None,
        });
    }

    let mut rng = XorShift64::new(config.seed);
    let mut permuted = pooled.clone();
    let mut extreme = 0usize;
    for _ in 0..config.monte_carlo_draws {
        shuffle_in_place(&mut permuted, &mut rng);
        let statistic = welch_statistic_scaled(
            &permuted[..sample_a.len()],
            &permuted[sample_a.len()..],
        ).abs();
        if statistic >= threshold {
            extreme += 1;
        }
    }
    let rate = extreme as f64 / config.monte_carlo_draws as f64;
    let p_value = (extreme as f64 + 1.0) / (config.monte_carlo_draws as f64 + 1.0);
    Ok(MisUnpairedPermutationResult {
        sample_count_a: sample_a.len(),
        sample_count_b: sample_b.len(),
        sample_mean_a: mean_a,
        sample_mean_b: mean_b,
        mean_difference_a_minus_b: effect,
        absolute_welch_statistic: abs_observed,
        extreme_assignments: extreme,
        assignments_evaluated: config.monte_carlo_draws,
        p_value_two_sided: p_value,
        method: MisUnpairedPermutationMethod::MonteCarlo {
            draws: config.monte_carlo_draws,
            seed: config.seed,
        },
        monte_carlo_standard_error: Some(
            (rate * (1.0 - rate) / config.monte_carlo_draws as f64).sqrt(),
        ),
    })
}

/// Bootstrap a percentile interval for mean(A)-mean(B), resampling both independent
/// samples separately with replacement.
///
/// The observed effect is reported separately from the bootstrap interval. The interval
/// is an approximate percentile interval, not an exact confidence interval. It assumes
/// independent sampling within each arm and one independent observation per analysis unit.
/// For MIS studies, do not pass repeated trials from one graph as if they were new graphs;
/// aggregate at graph level or use a hierarchical cluster model instead.
///
/// Confidence is configured in millionths to keep the config deterministic/Eq-friendly:
/// 95_000 means 0.95; 99_000 means 0.99.
pub fn unpaired_mean_bootstrap_interval(
    sample_a: &[f64],
    sample_b: &[f64],
    config: MisUnpairedBootstrapConfig,
) -> Result<MisUnpairedBootstrapResult, MisUnpairedInferenceError> {
    let scale = validated_scale(sample_a, sample_b)?;
    if config.replicates == 0 {
        return Err(MisUnpairedInferenceError::ZeroBootstrapReplicates);
    }
    if config.confidence_millionths == 0 || config.confidence_millionths >= 1_000_000 {
        return Err(MisUnpairedInferenceError::InvalidConfidence);
    }

    let a = scaled_values(sample_a, scale);
    let b = scaled_values(sample_b, scale);
    let mean_a = mean(&a);
    let mean_b = mean(&b);
    let effect = rescale_statistic(mean_a - mean_b, scale)?;
    let confidence = f64::from(config.confidence_millionths) / 1_000_000.0;
    let alpha = (1.0 - confidence) / 2.0;
    let mut rng = XorShift64::new(config.seed);
    let mut differences = Vec::with_capacity(config.replicates);

    for _ in 0..config.replicates {
        let mut sum_a = 0.0_f64;
        for _ in 0..a.len() {
            sum_a += a[random_index(&mut rng, a.len())];
        }
        let mut sum_b = 0.0_f64;
        for _ in 0..b.len() {
            sum_b += b[random_index(&mut rng, b.len())];
        }
        differences.push(sum_a / a.len() as f64 - sum_b / b.len() as f64);
    }

    let boot_mean = mean(&differences);
    let boot_variance = sample_variance(&differences, boot_mean);
    let mut sorted = differences;
    sorted.sort_by(f64::total_cmp);
    let lower = rescale_statistic(quantile_sorted(&sorted, alpha), scale)?;
    let upper = rescale_statistic(quantile_sorted(&sorted, 1.0 - alpha), scale)?;
    let standard_error = rescale_statistic(boot_variance.sqrt(), scale)?;

    Ok(MisUnpairedBootstrapResult {
        sample_count_a: sample_a.len(),
        sample_count_b: sample_b.len(),
        sample_mean_a: rescale_statistic(mean_a, scale)?,
        sample_mean_b: rescale_statistic(mean_b, scale)?,
        mean_difference_a_minus_b: effect,
        lower,
        upper,
        bootstrap_standard_error: standard_error,
        confidence,
        replicates: config.replicates,
        seed: config.seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_unpaired_permutation_uses_all_fixed_size_assignments() {
        let result = unpaired_mean_permutation_test(
            &[0.8, 0.9],
            &[0.1, 0.2],
            MisUnpairedPermutationConfig {
                exact_max_total: 4,
                monte_carlo_draws: 1,
                seed: 42,
            },
        ).unwrap();
        assert_eq!(result.method, MisUnpairedPermutationMethod::ExactLabelEnumeration);
        assert_eq!(result.assignments_evaluated, 6);
        assert_eq!(result.extreme_assignments, 2);
        assert!((result.p_value_two_sided - (1.0 / 3.0)).abs() < 1e-12);
        assert!((result.mean_difference_a_minus_b - 0.7).abs() < 1e-12);
        assert_eq!(result.monte_carlo_standard_error, None);
    }

    #[test]
    fn unpaired_monte_carlo_permutation_is_deterministic_and_plus_one_corrected() {
        let config = MisUnpairedPermutationConfig {
            exact_max_total: 4,
            monte_carlo_draws: 2_000,
            seed: 991,
        };
        let a = unpaired_mean_permutation_test(&[0.1, 0.4, 0.8, 0.9], &[0.05, 0.2, 0.3, 0.7], config).unwrap();
        let b = unpaired_mean_permutation_test(&[0.1, 0.4, 0.8, 0.9], &[0.05, 0.2, 0.3, 0.7], config).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.assignments_evaluated, 2_000);
        assert!(a.p_value_two_sided > 0.0 && a.p_value_two_sided <= 1.0);
        assert!(a.monte_carlo_standard_error.is_some());
    }

    #[test]
    fn unpaired_bootstrap_replays_and_reports_effect_and_interval() {
        let config = MisUnpairedBootstrapConfig {
            replicates: 2_000,
            seed: 1234,
            confidence_millionths: 95_000,
        };
        let a = unpaired_mean_bootstrap_interval(&[1.0, 2.0, 3.0, 4.0], &[0.0, 1.0, 2.0, 3.0], config).unwrap();
        let b = unpaired_mean_bootstrap_interval(&[1.0, 2.0, 3.0, 4.0], &[0.0, 1.0, 2.0, 3.0], config).unwrap();
        assert_eq!(a, b);
        assert!((a.mean_difference_a_minus_b - 1.0).abs() < 1e-12);
        assert!(a.lower < a.upper);
        assert!(a.bootstrap_standard_error > 0.0);
        assert_eq!(a.confidence, 0.95);
    }

    #[test]
    fn unpaired_inference_fails_closed_on_small_or_invalid_samples() {
        assert_eq!(
            unpaired_mean_permutation_test(&[1.0], &[0.0, 1.0], MisUnpairedPermutationConfig::default()),
            Err(MisUnpairedInferenceError::InsufficientSample { group: 'A', count: 1 })
        );
        assert_eq!(
            unpaired_mean_permutation_test(&[1.0, f64::NAN], &[0.0, 1.0], MisUnpairedPermutationConfig::default()),
            Err(MisUnpairedInferenceError::NonFiniteObservation { group: 'A', index: 1 })
        );
        assert_eq!(
            unpaired_mean_permutation_test(&[1.0, 2.0], &[0.0, 1.0], MisUnpairedPermutationConfig { exact_max_total: 1, ..MisUnpairedPermutationConfig::default() }),
            Err(MisUnpairedInferenceError::InvalidExactLimit)
        );
        assert_eq!(
            unpaired_mean_bootstrap_interval(&[1.0, 2.0], &[0.0, 1.0], MisUnpairedBootstrapConfig { replicates: 0, ..MisUnpairedBootstrapConfig::default() }),
            Err(MisUnpairedInferenceError::ZeroBootstrapReplicates)
        );
        assert_eq!(
            unpaired_mean_bootstrap_interval(&[1.0, 2.0], &[0.0, 1.0], MisUnpairedBootstrapConfig { confidence_millionths: 1_000_000, ..MisUnpairedBootstrapConfig::default() }),
            Err(MisUnpairedInferenceError::InvalidConfidence)
        );
    }
}
