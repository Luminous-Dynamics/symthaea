//! Exact finite-support distributional comparison for temporal exposure outcomes.
//!
//! This module is a deterministic empirical-distribution reference. It reports
//! lower-cost first-order CDF dominance and upper-tail stop-loss ordering on a
//! preregistered common finite support. It does not perform statistical inference,
//! estimate population dominance, or claim that empirical ordering generalizes.

use std::fmt;

pub const MAX_GRID_BINS: usize = 100_000;
pub const MAX_OBSERVATIONS_PER_ARM: u64 = 1_000_000;
const QUANTILE_LEVELS: [(u32, u32); 4] = [(1, 2), (9, 10), (95, 100), (99, 100)];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FiniteGridDistribution {
    /// Common ordered exposure support in microseconds.
    pub support_us: Vec<u64>,
    /// Number of observed episodes at each support value.
    pub counts: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DominanceDirection {
    pub treatment_dominates_control: bool,
    pub control_dominates_treatment: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DistributionalThreshold {
    pub threshold_us: u64,
    pub treatment_cdf_count: u64,
    pub control_cdf_count: u64,
    /// Numerator of E[(exposure - threshold)_+], denominator = arm sample count.
    pub treatment_stop_loss_numerator: u128,
    pub control_stop_loss_numerator: u128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalDistributionalComparison {
    pub support_us: Vec<u64>,
    pub treatment_sample_count: u64,
    pub control_sample_count: u64,
    pub treatment_mean_numerator: u128,
    pub control_mean_numerator: u128,
    pub treatment_mean_lower: bool,
    /// Quantile order: median, p90, p95, p99 (nearest-rank empirical quantiles).
    pub treatment_quantiles_us: [u64; 4],
    pub control_quantiles_us: [u64; 4],
    /// For lower-is-better exposure, F_treatment(x) >= F_control(x) at every x.
    pub first_order_lower_cost: DominanceDirection,
    /// For lower-is-better exposure, E[(X_treatment - t)_+] <=
    /// E[(X_control - t)_+] at every declared threshold t.
    pub upper_tail_stop_loss: DominanceDirection,
    pub thresholds: Vec<DistributionalThreshold>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DistributionalError {
    EmptySupport,
    TooManySupportBins { actual: usize, maximum: usize },
    SupportCountMismatch { support: usize, counts: usize },
    UnsortedOrDuplicateSupport,
    ZeroTotalObservations { arm: &'static str },
    TooManyObservations { arm: &'static str, actual: u64, maximum: u64 },
    SupportMismatch,
    ArithmeticOverflow,
}

impl fmt::Display for DistributionalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySupport => write!(f, "finite support must not be empty"),
            Self::TooManySupportBins { actual, maximum } =>
                write!(f, "support has {actual} bins; maximum is {maximum}"),
            Self::SupportCountMismatch { support, counts } =>
                write!(f, "support length {support} differs from count length {counts}"),
            Self::UnsortedOrDuplicateSupport =>
                write!(f, "support values must be strictly increasing and unique"),
            Self::ZeroTotalObservations { arm } =>
                write!(f, "{arm} distribution has zero observations"),
            Self::TooManyObservations { arm, actual, maximum } =>
                write!(f, "{arm} distribution has {actual} observations; maximum is {maximum}"),
            Self::SupportMismatch => write!(f, "treatment and control must use the same frozen support"),
            Self::ArithmeticOverflow => write!(f, "exact distributional arithmetic overflowed"),
        }
    }
}

impl std::error::Error for DistributionalError {}

/// Compare empirical exposure distributions on one frozen finite support.
///
/// First-order lower-cost dominance checks the complete empirical CDF. The separate
/// stop-loss ordering checks expected exceedance beyond every support threshold and
/// therefore exposes tail regressions that a mean or sign-only comparison can miss.
/// All comparisons are exact integer cross-products; no floating-point tolerance is
/// used. A false dominance flag means only that this empirical sample did not satisfy
/// the declared ordering, not that the distributions are equivalent.
pub fn compare_temporal_exposure_distributions(
    treatment: &FiniteGridDistribution,
    control: &FiniteGridDistribution,
) -> Result<TemporalDistributionalComparison, DistributionalError> {
    let treatment_total = validate_distribution(treatment, "treatment")?;
    let control_total = validate_distribution(control, "control")?;
    if treatment.support_us != control.support_us {
        return Err(DistributionalError::SupportMismatch);
    }

    let support = &treatment.support_us;
    let mut treatment_cdf = Vec::with_capacity(support.len());
    let mut control_cdf = Vec::with_capacity(support.len());
    let mut treatment_running = 0_u64;
    let mut control_running = 0_u64;
    for index in 0..support.len() {
        treatment_running = treatment_running
            .checked_add(treatment.counts[index])
            .ok_or(DistributionalError::ArithmeticOverflow)?;
        control_running = control_running
            .checked_add(control.counts[index])
            .ok_or(DistributionalError::ArithmeticOverflow)?;
        treatment_cdf.push(treatment_running);
        control_cdf.push(control_running);
    }

    let mut thresholds = Vec::with_capacity(support.len());
    let mut treatment_first_ge_all = true;
    let mut control_first_ge_all = true;
    let mut treatment_first_strict = false;
    let mut control_first_strict = false;
    let mut treatment_stop_le_all = true;
    let mut control_stop_le_all = true;
    let mut treatment_stop_strict = false;
    let mut control_stop_strict = false;

    let treatment_mean_numerator = weighted_sum(support, &treatment.counts)?;
    let control_mean_numerator = weighted_sum(support, &control.counts)?;
    let treatment_mean_lower = treatment_mean_numerator
        .checked_mul(u128::from(control_total))
        .ok_or(DistributionalError::ArithmeticOverflow)?
        < control_mean_numerator
            .checked_mul(u128::from(treatment_total))
            .ok_or(DistributionalError::ArithmeticOverflow)?;

    for index in 0..support.len() {
        let treatment_cdf_cross = u128::from(treatment_cdf[index])
            .checked_mul(u128::from(control_total))
            .ok_or(DistributionalError::ArithmeticOverflow)?;
        let control_cdf_cross = u128::from(control_cdf[index])
            .checked_mul(u128::from(treatment_total))
            .ok_or(DistributionalError::ArithmeticOverflow)?;

        let t_first_ge = treatment_cdf_cross >= control_cdf_cross;
        let c_first_ge = treatment_cdf_cross <= control_cdf_cross;
        treatment_first_ge_all &= t_first_ge;
        control_first_ge_all &= c_first_ge;
        treatment_first_strict |= treatment_cdf_cross > control_cdf_cross;
        control_first_strict |= treatment_cdf_cross < control_cdf_cross;

        let treatment_stop = stop_loss_numerator(support, &treatment.counts, index)?;
        let control_stop = stop_loss_numerator(support, &control.counts, index)?;
        let treatment_stop_cross = treatment_stop
            .checked_mul(u128::from(control_total))
            .ok_or(DistributionalError::ArithmeticOverflow)?;
        let control_stop_cross = control_stop
            .checked_mul(u128::from(treatment_total))
            .ok_or(DistributionalError::ArithmeticOverflow)?;

        let t_stop_le = treatment_stop_cross <= control_stop_cross;
        let c_stop_le = treatment_stop_cross >= control_stop_cross;
        treatment_stop_le_all &= t_stop_le;
        control_stop_le_all &= c_stop_le;
        treatment_stop_strict |= treatment_stop_cross < control_stop_cross;
        control_stop_strict |= treatment_stop_cross > control_stop_cross;

        thresholds.push(DistributionalThreshold {
            threshold_us: support[index],
            treatment_cdf_count: treatment_cdf[index],
            control_cdf_count: control_cdf[index],
            treatment_stop_loss_numerator: treatment_stop,
            control_stop_loss_numerator: control_stop,
        });
    }

    Ok(TemporalDistributionalComparison {
        support_us: support.clone(),
        treatment_sample_count: treatment_total,
        control_sample_count: control_total,
        treatment_mean_numerator,
        control_mean_numerator,
        treatment_mean_lower,
        treatment_quantiles_us: quantiles(support, &treatment.counts, treatment_total)?,
        control_quantiles_us: quantiles(support, &control.counts, control_total)?,
        first_order_lower_cost: DominanceDirection {
            treatment_dominates_control: treatment_first_ge_all && treatment_first_strict,
            control_dominates_treatment: control_first_ge_all && control_first_strict,
        },
        upper_tail_stop_loss: DominanceDirection {
            treatment_dominates_control: treatment_stop_le_all && treatment_stop_strict,
            control_dominates_treatment: control_stop_le_all && control_stop_strict,
        },
        thresholds,
    })
}

fn validate_distribution(
    distribution: &FiniteGridDistribution,
    arm: &'static str,
) -> Result<u64, DistributionalError> {
    if distribution.support_us.is_empty() {
        return Err(DistributionalError::EmptySupport);
    }
    if distribution.support_us.len() > MAX_GRID_BINS {
        return Err(DistributionalError::TooManySupportBins {
            actual: distribution.support_us.len(),
            maximum: MAX_GRID_BINS,
        });
    }
    if distribution.support_us.len() != distribution.counts.len() {
        return Err(DistributionalError::SupportCountMismatch {
            support: distribution.support_us.len(),
            counts: distribution.counts.len(),
        });
    }
    if distribution.support_us.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(DistributionalError::UnsortedOrDuplicateSupport);
    }
    let total = distribution.counts.iter().try_fold(0_u64, |sum, count| {
        sum.checked_add(*count).ok_or(DistributionalError::ArithmeticOverflow)
    })?;
    if total == 0 {
        return Err(DistributionalError::ZeroTotalObservations { arm });
    }
    if total > MAX_OBSERVATIONS_PER_ARM {
        return Err(DistributionalError::TooManyObservations {
            arm,
            actual: total,
            maximum: MAX_OBSERVATIONS_PER_ARM,
        });
    }
    Ok(total)
}

fn weighted_sum(support: &[u64], counts: &[u64]) -> Result<u128, DistributionalError> {
    support.iter().zip(counts).try_fold(0_u128, |sum, (value, count)| {
        let term = u128::from(*value)
            .checked_mul(u128::from(*count))
            .ok_or(DistributionalError::ArithmeticOverflow)?;
        sum.checked_add(term).ok_or(DistributionalError::ArithmeticOverflow)
    })
}

/// Exact numerator of the empirical stop-loss mean E[(X - threshold)_+].
fn stop_loss_numerator(
    support: &[u64],
    counts: &[u64],
    threshold_index: usize,
) -> Result<u128, DistributionalError> {
    let threshold = support[threshold_index];
    support.iter().zip(counts).skip(threshold_index + 1)
        .try_fold(0_u128, |sum, (value, count)| {
            let excess = value - threshold;
            let term = u128::from(excess)
                .checked_mul(u128::from(*count))
                .ok_or(DistributionalError::ArithmeticOverflow)?;
            sum.checked_add(term).ok_or(DistributionalError::ArithmeticOverflow)
        })
}

fn quantiles(
    support: &[u64],
    counts: &[u64],
    total: u64,
) -> Result<[u64; 4], DistributionalError> {
    let mut output = [0_u64; 4];
    for (position, (numerator, denominator)) in QUANTILE_LEVELS.into_iter().enumerate() {
        let rank_numerator = u128::from(numerator)
            .checked_mul(u128::from(total))
            .ok_or(DistributionalError::ArithmeticOverflow)?;
        let rank = rank_numerator
            .checked_add(u128::from(denominator - 1))
            .ok_or(DistributionalError::ArithmeticOverflow)?
            / u128::from(denominator);
        let rank = u64::try_from(rank.max(1)).map_err(|_| DistributionalError::ArithmeticOverflow)?;
        let mut cumulative = 0_u64;
        let mut selected = None;
        for (value, count) in support.iter().zip(counts) {
            cumulative = cumulative.checked_add(*count).ok_or(DistributionalError::ArithmeticOverflow)?;
            if cumulative >= rank {
                selected = Some(*value);
                break;
            }
        }
        output[position] = selected.ok_or(DistributionalError::ArithmeticOverflow)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distribution(support: &[u64], counts: &[u64]) -> FiniteGridDistribution {
        FiniteGridDistribution {
            support_us: support.to_vec(),
            counts: counts.to_vec(),
        }
    }

    #[test]
    fn strict_lower_cost_first_order_dominance_is_detected_exactly() {
        let treatment = distribution(&[0, 10, 20], &[5, 5, 0]);
        let control = distribution(&[0, 10, 20], &[0, 5, 5]);
        let result = compare_temporal_exposure_distributions(&treatment, &control).unwrap();
        assert!(result.first_order_lower_cost.treatment_dominates_control);
        assert!(!result.first_order_lower_cost.control_dominates_treatment);
        assert!(result.upper_tail_stop_loss.treatment_dominates_control);
    }

    #[test]
    fn crossing_cdfs_do_not_create_a_false_first_order_claim() {
        let treatment = distribution(&[0, 1, 2], &[5, 0, 5]);
        let control = distribution(&[0, 1, 2], &[0, 10, 0]);
        let result = compare_temporal_exposure_distributions(&treatment, &control).unwrap();
        assert!(!result.first_order_lower_cost.treatment_dominates_control);
        assert!(!result.first_order_lower_cost.control_dominates_treatment);
    }

    #[test]
    fn equal_mean_different_tails_are_not_misreported_as_dominance() {
        let treatment = distribution(&[0, 1, 2], &[5, 0, 5]);
        let control = distribution(&[0, 1, 2], &[0, 10, 0]);
        let result = compare_temporal_exposure_distributions(&treatment, &control).unwrap();
        assert_eq!(result.treatment_mean_numerator * u128::from(result.control_sample_count),
                   result.control_mean_numerator * u128::from(result.treatment_sample_count));
        assert!(!result.first_order_lower_cost.treatment_dominates_control);
        assert!(!result.first_order_lower_cost.control_dominates_treatment);
        assert!(!result.upper_tail_stop_loss.treatment_dominates_control);
        assert!(!result.upper_tail_stop_loss.control_dominates_treatment);
    }

    #[test]
    fn lower_mean_can_coexist_with_a_worse_p99_and_tail_stop_loss() {
        let treatment = distribution(&[0, 10, 100], &[95, 0, 5]);
        let control = distribution(&[0, 10, 100], &[0, 100, 0]);
        let result = compare_temporal_exposure_distributions(&treatment, &control).unwrap();
        assert!(result.treatment_mean_lower);
        assert!(result.treatment_quantiles_us[3] > result.control_quantiles_us[3]);
        assert!(!result.first_order_lower_cost.treatment_dominates_control);
        assert!(result.upper_tail_stop_loss.control_dominates_treatment);
    }

    #[test]
    fn identical_distributions_are_not_reported_as_strict_dominance() {
        let arm = distribution(&[0, 10, 50], &[2, 3, 5]);
        let result = compare_temporal_exposure_distributions(&arm, &arm).unwrap();
        assert!(!result.first_order_lower_cost.treatment_dominates_control);
        assert!(!result.first_order_lower_cost.control_dominates_treatment);
        assert!(!result.upper_tail_stop_loss.treatment_dominates_control);
        assert!(!result.upper_tail_stop_loss.control_dominates_treatment);
    }

    #[test]
    fn nonuniform_support_gaps_are_handled_by_exact_excess_costs() {
        let treatment = distribution(&[0, 5, 100], &[9, 1, 0]);
        let control = distribution(&[0, 5, 100], &[0, 10, 0]);
        let result = compare_temporal_exposure_distributions(&treatment, &control).unwrap();
        assert!(result.first_order_lower_cost.treatment_dominates_control);
        assert!(result.upper_tail_stop_loss.treatment_dominates_control);
        assert_eq!(result.thresholds[1].treatment_stop_loss_numerator, 0);
        assert_eq!(result.thresholds[1].control_stop_loss_numerator, 0);
    }

    #[test]
    fn mismatched_support_and_malformed_distributions_fail_closed() {
        let treatment = distribution(&[0, 10], &[5, 5]);
        let mismatch = distribution(&[0, 20], &[5, 5]);
        assert_eq!(
            compare_temporal_exposure_distributions(&treatment, &mismatch),
            Err(DistributionalError::SupportMismatch)
        );
        let unsorted = distribution(&[10, 0], &[5, 5]);
        assert_eq!(
            compare_temporal_exposure_distributions(&unsorted, &unsorted),
            Err(DistributionalError::UnsortedOrDuplicateSupport)
        );
        let zero = distribution(&[0, 10], &[0, 0]);
        assert_eq!(
            compare_temporal_exposure_distributions(&zero, &zero),
            Err(DistributionalError::ZeroTotalObservations { arm: "treatment" })
        );
    }

    #[test]
    fn exact_means_and_nearest_rank_quantiles_are_preserved() {
        let treatment = distribution(&[0, 10, 100], &[95, 0, 5]);
        let control = distribution(&[0, 10, 100], &[0, 100, 0]);
        let result = compare_temporal_exposure_distributions(&treatment, &control).unwrap();
        assert_eq!(result.treatment_mean_numerator, 500);
        assert_eq!(result.control_mean_numerator, 1_000);
        assert_eq!(result.treatment_quantiles_us, [0, 0, 0, 100]);
        assert_eq!(result.control_quantiles_us, [10, 10, 10, 10]);
    }
}
