#![forbid(unsafe_code)]

//! Multi-resolution anytime-valid monitoring of a fixed predictive-rank reference.
//!
//! The monitor builds a deterministic dyadic Haar contrast basis over the
//! n+1 possible one-based predictive-rank buckets. Every contrast is scaled
//! to [-1/2, 1/2], centered against the exact predictive law
//!
//!     pi[t,j] = (1 + N[t-1,j]) / (n + t),
//!
//! and assigned positive- and negative-direction fixed-bet e-processes.
//! A normalized, pre-registered mixture across resolutions and directions
//! preserves the anytime null guarantee under the same fixed-reference
//! exchangeability assumptions as CMAQ-031.
//!
//! This crate accepts already-qualified ranks. It does not authenticate score
//! receipts, tie-breaking, common-cause closure, cluster independence, or
//! upstream rank construction. Those remain CMAQ-031/032/evidence-authority
//! responsibilities.

use std::f64::consts::LN_2;

const BET_LAMBDA: f64 = 0.5;
const MAX_RANK_BUCKETS: usize = 1 << 20;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpectrumError {
    InvalidCalibrationCount,
    InvalidMiscoverage,
    TooManyRankBuckets,
    InvalidRank,
    CounterOverflow,
    ArithmeticOverflow,
    InvalidPredictivePayoff,
    InvalidBettingFactor,
    EmptyFeatureBank,
    AlreadyQuarantined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    Positive,
    Negative,
}

impl Direction {
    fn sign(self) -> i128 {
        match self {
            Self::Positive => 1,
            Self::Negative => -1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct HaarContrast {
    start: usize,
    midpoint: usize,
    end: usize,
    resolution_level: usize,
}

impl HaarContrast {
    /// Returns twice the contrast value, so each bucket code is -1, 0 or +1.
    fn code_for_bucket(&self, bucket_zero_based: usize) -> i128 {
        if (self.start..self.midpoint).contains(&bucket_zero_based) {
            1
        } else if (self.midpoint..self.end).contains(&bucket_zero_based) {
            -1
        } else {
            0
        }
    }

    fn code_for_rank(&self, rank_one_based: usize) -> i128 {
        self.code_for_bucket(rank_one_based - 1)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Component {
    contrast: HaarContrast,
    direction: Direction,
    /// Log of the fixed, pre-registered mixture weight.
    log_weight: f64,
    /// Log wealth of this directional e-process.
    log_wealth: f64,
    last_centered_payoff: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpectrumUpdate {
    Accepted {
        rank_one_based: usize,
        log_mixture_wealth: f64,
        component_count: usize,
    },
    Quarantined {
        rank_one_based: usize,
        log_mixture_wealth: f64,
        threshold_log_wealth: f64,
        component_count: usize,
    },
}

/// Fixed-reference, multi-resolution rank-spectrum monitor.
///
/// The reference is represented by its calibration count and must be supplied
/// by an upstream qualified evidence boundary. Rank buckets are one-based:
/// 1 through calibration_count + 1.
#[derive(Clone, Debug, PartialEq)]
pub struct RankSpectrumMonitor {
    calibration_count: usize,
    alpha_numerator: u32,
    alpha_denominator: u32,
    rank_counts: Vec<u64>,
    observations_seen: u64,
    components: Vec<Component>,
    resolution_levels: usize,
    quarantined: bool,
}

impl RankSpectrumMonitor {
    /// Creates a monitor with deterministic Haar contrasts and normalized
    /// equal mass per active resolution and per betting direction.
    ///
    /// The basis is frozen from calibration_count before any deployment rank
    /// is observed. For m = calibration_count + 1 rank buckets, the pruned
    /// dyadic tree contains exactly m-1 nonempty contrasts and 2(m-1)
    /// directional e-processes.
    pub fn new(
        calibration_count: usize,
        alpha_numerator: u32,
        alpha_denominator: u32,
    ) -> Result<Self, SpectrumError> {
        if calibration_count == 0 {
            return Err(SpectrumError::InvalidCalibrationCount);
        }
        if alpha_numerator == 0
            || alpha_denominator == 0
            || alpha_numerator >= alpha_denominator
        {
            return Err(SpectrumError::InvalidMiscoverage);
        }

        let rank_buckets = calibration_count
            .checked_add(1)
            .ok_or(SpectrumError::ArithmeticOverflow)?;
        if rank_buckets > MAX_RANK_BUCKETS {
            return Err(SpectrumError::TooManyRankBuckets);
        }
        let padded_buckets = rank_buckets
            .checked_next_power_of_two()
            .ok_or(SpectrumError::ArithmeticOverflow)?;
        let maximum_levels = padded_buckets.trailing_zeros() as usize;
        if maximum_levels == 0 {
            return Err(SpectrumError::EmptyFeatureBank);
        }

        let mut levels: Vec<Vec<HaarContrast>> = Vec::with_capacity(maximum_levels);
        for resolution_level in 0..maximum_levels {
            let width = padded_buckets >> resolution_level;
            let node_count = padded_buckets / width;
            let mut contrasts = Vec::new();

            for node in 0..node_count {
                let start = node
                    .checked_mul(width)
                    .ok_or(SpectrumError::ArithmeticOverflow)?;
                let midpoint = start
                    .checked_add(width / 2)
                    .ok_or(SpectrumError::ArithmeticOverflow)?;
                let end_unclipped = start
                    .checked_add(width)
                    .ok_or(SpectrumError::ArithmeticOverflow)?;
                let end = end_unclipped.min(rank_buckets);

                // Padding may make one side empty. Such a contrast carries no
                // information and is omitted from the frozen basis.
                if start < rank_buckets && midpoint < rank_buckets && end > midpoint {
                    contrasts.push(HaarContrast {
                        start,
                        midpoint,
                        end,
                        resolution_level,
                    });
                }
            }

            if !contrasts.is_empty() {
                levels.push(contrasts);
            }
        }

        let resolution_levels = levels.len();
        if resolution_levels == 0 {
            return Err(SpectrumError::EmptyFeatureBank);
        }

        let mut components = Vec::with_capacity(
            rank_buckets
                .checked_sub(1)
                .and_then(|count| count.checked_mul(2))
                .ok_or(SpectrumError::ArithmeticOverflow)?,
        );

        for contrasts in levels {
            let per_component_log_weight =
                -(resolution_levels as f64).ln() - LN_2 - (contrasts.len() as f64).ln();

            for contrast in contrasts {
                components.push(Component {
                    contrast: contrast.clone(),
                    direction: Direction::Positive,
                    log_weight: per_component_log_weight,
                    log_wealth: 0.0,
                    last_centered_payoff: 0.0,
                });
                components.push(Component {
                    contrast,
                    direction: Direction::Negative,
                    log_weight: per_component_log_weight,
                    log_wealth: 0.0,
                    last_centered_payoff: 0.0,
                });
            }
        }

        if components.len() != (rank_buckets - 1) * 2 {
            return Err(SpectrumError::EmptyFeatureBank);
        }

        let monitor = Self {
            calibration_count,
            alpha_numerator,
            alpha_denominator,
            rank_counts: vec![0; rank_buckets],
            observations_seen: 0,
            components,
            resolution_levels,
            quarantined: false,
        };

        let initial_log_mixture = monitor.log_mixture_wealth()?;
        if initial_log_mixture.abs() > 1e-10 {
            return Err(SpectrumError::ArithmeticOverflow);
        }

        Ok(monitor)
    }

    /// Adds one already-qualified rank against the immutable calibration set.
    ///
    /// The payoff for every Haar feature is centered against the exact
    /// predictive rank distribution before the current rank is counted.
    /// State is committed only after all feature factors and mixture arithmetic
    /// have been validated.
    pub fn observe_rank(
        &mut self,
        rank_one_based: usize,
    ) -> Result<SpectrumUpdate, SpectrumError> {
        if self.quarantined {
            return Err(SpectrumError::AlreadyQuarantined);
        }
        if rank_one_based == 0 || rank_one_based > self.calibration_count + 1 {
            return Err(SpectrumError::InvalidRank);
        }

        let next_seen = self
            .observations_seen
            .checked_add(1)
            .ok_or(SpectrumError::CounterOverflow)?;
        let rank_index = rank_one_based - 1;
        let next_rank_count = self.rank_counts[rank_index]
            .checked_add(1)
            .ok_or(SpectrumError::CounterOverflow)?;

        // One prefix scan per observation makes evaluation O(m) rather than
        // rescanning all m buckets for each of O(m) Haar contrasts.
        let mut weighted_prefix = Vec::with_capacity(self.rank_counts.len() + 1);
        weighted_prefix.push(0i128);
        for count in &self.rank_counts {
            let weight = i128::from(*count)
                .checked_add(1)
                .ok_or(SpectrumError::ArithmeticOverflow)?;
            let next = weighted_prefix
                .last()
                .copied()
                .ok_or(SpectrumError::ArithmeticOverflow)?
                .checked_add(weight)
                .ok_or(SpectrumError::ArithmeticOverflow)?;
            weighted_prefix.push(next);
        }

        let mut next_log_wealths = Vec::with_capacity(self.components.len());
        let mut next_payoffs = Vec::with_capacity(self.components.len());

        for component in &self.components {
            let base_payoff = self.centered_payoff_from_prefix(
                &component.contrast,
                rank_one_based,
                next_seen,
                &weighted_prefix,
            )?;
            let payoff = base_payoff * component.direction.sign() as f64;
            if !payoff.is_finite() || !(-1.0 - 1e-12..=1.0 + 1e-12).contains(&payoff) {
                return Err(SpectrumError::InvalidPredictivePayoff);
            }

            let factor = 1.0 + BET_LAMBDA * payoff;
            if !factor.is_finite() || factor < 0.5 - 1e-12 {
                return Err(SpectrumError::InvalidBettingFactor);
            }

            let next_log_wealth = component.log_wealth + factor.ln();
            if !next_log_wealth.is_finite() {
                return Err(SpectrumError::ArithmeticOverflow);
            }
            next_log_wealths.push(next_log_wealth);
            next_payoffs.push(payoff);
        }

        let log_mixture = self.log_mixture_from(&next_log_wealths)?;
        let threshold = (f64::from(self.alpha_denominator)
            / f64::from(self.alpha_numerator))
            .ln();
        if !log_mixture.is_finite() || !threshold.is_finite() {
            return Err(SpectrumError::ArithmeticOverflow);
        }

        // Commit only after all validation and arithmetic succeeds.
        self.rank_counts[rank_index] = next_rank_count;
        self.observations_seen = next_seen;
        for ((component, next_log_wealth), next_payoff) in self
            .components
            .iter_mut()
            .zip(next_log_wealths)
            .zip(next_payoffs)
        {
            component.log_wealth = next_log_wealth;
            component.last_centered_payoff = next_payoff;
        }

        if log_mixture >= threshold {
            self.quarantined = true;
            Ok(SpectrumUpdate::Quarantined {
                rank_one_based,
                log_mixture_wealth: log_mixture,
                threshold_log_wealth: threshold,
                component_count: self.components.len(),
            })
        } else {
            Ok(SpectrumUpdate::Accepted {
                rank_one_based,
                log_mixture_wealth: log_mixture,
                component_count: self.components.len(),
            })
        }
    }

    #[cfg(test)]
    fn centered_payoff(
        &self,
        contrast: &HaarContrast,
        rank_one_based: usize,
        observation_index_one_based: u64,
    ) -> Result<f64, SpectrumError> {
        let mut weighted_prefix = Vec::with_capacity(self.rank_counts.len() + 1);
        weighted_prefix.push(0i128);
        for count in &self.rank_counts {
            let weight = i128::from(*count)
                .checked_add(1)
                .ok_or(SpectrumError::ArithmeticOverflow)?;
            let next = weighted_prefix
                .last()
                .copied()
                .ok_or(SpectrumError::ArithmeticOverflow)?
                .checked_add(weight)
                .ok_or(SpectrumError::ArithmeticOverflow)?;
            weighted_prefix.push(next);
        }
        self.centered_payoff_from_prefix(
            contrast,
            rank_one_based,
            observation_index_one_based,
            &weighted_prefix,
        )
    }

    fn centered_payoff_from_prefix(
        &self,
        contrast: &HaarContrast,
        rank_one_based: usize,
        observation_index_one_based: u64,
        weighted_prefix: &[i128],
    ) -> Result<f64, SpectrumError> {
        if weighted_prefix.len() != self.rank_counts.len() + 1 {
            return Err(SpectrumError::InvalidPredictivePayoff);
        }
        let total = (self.calibration_count as u128)
            .checked_add(u128::from(observation_index_one_based))
            .ok_or(SpectrumError::ArithmeticOverflow)?;
        let total_i128 = i128::try_from(total).map_err(|_| SpectrumError::ArithmeticOverflow)?;

        // The Haar code is +1 on [start, midpoint), -1 on [midpoint, end),
        // and zero elsewhere. Prefix sums produce its predictable mean in O(1).
        let positive_mass = weighted_prefix[contrast.midpoint]
            .checked_sub(weighted_prefix[contrast.start])
            .ok_or(SpectrumError::ArithmeticOverflow)?;
        let negative_mass = weighted_prefix[contrast.end]
            .checked_sub(weighted_prefix[contrast.midpoint])
            .ok_or(SpectrumError::ArithmeticOverflow)?;
        let weighted_code_sum = positive_mass
            .checked_sub(negative_mass)
            .ok_or(SpectrumError::ArithmeticOverflow)?;

        // Haar values are code/2. Keep predictive centering exact until f64.
        let observed_code = contrast.code_for_rank(rank_one_based);
        let observed_term = observed_code
            .checked_mul(total_i128)
            .ok_or(SpectrumError::ArithmeticOverflow)?;
        let numerator = observed_term
            .checked_sub(weighted_code_sum)
            .ok_or(SpectrumError::ArithmeticOverflow)?;
        let denominator = total
            .checked_mul(2)
            .ok_or(SpectrumError::ArithmeticOverflow)?;

        let payoff = numerator as f64 / denominator as f64;
        if !payoff.is_finite() || !(-1.0 - 1e-12..=1.0 + 1e-12).contains(&payoff) {
            return Err(SpectrumError::InvalidPredictivePayoff);
        }
        Ok(payoff.clamp(-1.0, 1.0))
    }

    fn log_mixture_from(&self, log_wealths: &[f64]) -> Result<f64, SpectrumError> {
        if log_wealths.len() != self.components.len() || log_wealths.is_empty() {
            return Err(SpectrumError::ArithmeticOverflow);
        }
        let weighted_logs: Vec<f64> = self
            .components
            .iter()
            .zip(log_wealths)
            .map(|(component, log_wealth)| component.log_weight + log_wealth)
            .collect();
        let maximum = weighted_logs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !maximum.is_finite() {
            return Err(SpectrumError::ArithmeticOverflow);
        }
        let scaled_sum: f64 = weighted_logs
            .iter()
            .map(|value| (value - maximum).exp())
            .sum();
        let result = maximum + scaled_sum.ln();
        if !result.is_finite() {
            return Err(SpectrumError::ArithmeticOverflow);
        }
        Ok(result)
    }

    /// Current log of the normalized mixture wealth.
    pub fn log_mixture_wealth(&self) -> Result<f64, SpectrumError> {
        let log_wealths: Vec<f64> = self
            .components
            .iter()
            .map(|component| component.log_wealth)
            .collect();
        self.log_mixture_from(&log_wealths)
    }

    pub fn calibration_count(&self) -> usize {
        self.calibration_count
    }

    pub fn observations_seen(&self) -> u64 {
        self.observations_seen
    }

    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    pub fn resolution_levels(&self) -> usize {
        self.resolution_levels
    }

    pub fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    /// Diagnostic only: returns the sum of the fixed mixture weights.
    pub fn mixture_weight_sum(&self) -> f64 {
        self.components
            .iter()
            .map(|component| component.log_weight.exp())
            .sum()
    }

    /// Number of positive-direction Haar contrasts at each active resolution.
    pub fn contrast_counts_by_resolution(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.resolution_levels];
        for component in &self.components {
            if component.direction == Direction::Positive {
                counts[component.contrast.resolution_level] += 1;
            }
        }
        counts
    }

    /// Diagnostic snapshot of the last centered payoff for every directional component.
    pub fn last_centered_payoffs(&self) -> Vec<f64> {
        self.components
            .iter()
            .map(|component| component.last_centered_payoff)
            .collect()
    }

    #[cfg(test)]
    fn contrasts(&self) -> Vec<&HaarContrast> {
        self.components
            .iter()
            .filter(|component| component.direction == Direction::Positive)
            .map(|component| &component.contrast)
            .collect()
    }

    #[cfg(test)]
    fn test_payoff(
        &self,
        contrast: &HaarContrast,
        rank_one_based: usize,
        observation_index_one_based: u64,
    ) -> Result<f64, SpectrumError> {
        self.centered_payoff(contrast, rank_one_based, observation_index_one_based)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_basis_separates_every_pair_of_rank_buckets() {
        for calibration_count in [1, 2, 3, 18, 49, 99] {
            let monitor = RankSpectrumMonitor::new(calibration_count, 1, 20).unwrap();
            let contrasts = monitor.contrasts();
            let buckets = calibration_count + 1;

            for left in 0..buckets {
                for right in (left + 1)..buckets {
                    assert!(
                        contrasts.iter().any(|contrast| {
                            contrast.code_for_bucket(left) != contrast.code_for_bucket(right)
                        }),
                        "unseparated buckets {left} and {right} for n={calibration_count}"
                    );
                }
            }
        }
    }

    #[test]
    fn pruned_tree_has_two_directions_for_each_of_m_minus_one_contrasts() {
        for calibration_count in [1, 2, 3, 18, 49, 99, 999] {
            let monitor = RankSpectrumMonitor::new(calibration_count, 1, 20).unwrap();
            assert_eq!(monitor.component_count(), 2 * calibration_count);
        }
    }

    #[test]
    fn mixture_weights_are_normalized_and_initial_wealth_is_one() {
        for calibration_count in [1, 18, 49, 99, 999] {
            let monitor = RankSpectrumMonitor::new(calibration_count, 1, 20).unwrap();
            assert!((monitor.mixture_weight_sum() - 1.0).abs() < 1e-12);
            assert!(monitor.log_mixture_wealth().unwrap().abs() < 1e-10);
        }
    }

    #[test]
    fn every_feature_payoff_is_conditionally_centered() {
        let mut monitor = RankSpectrumMonitor::new(18, 1, 20).unwrap();
        for rank in [2, 4, 4, 9, 13, 19] {
            let update = monitor.observe_rank(rank).unwrap();
            assert!(matches!(update, SpectrumUpdate::Accepted { .. }));
        }

        let next_t = monitor.observations_seen() + 1;
        let denominator = (monitor.calibration_count() as u64) + next_t;
        for contrast in monitor.contrasts() {
            let mut expectation = 0.0;
            for rank in 1..=monitor.calibration_count() + 1 {
                let probability =
                    (1 + monitor.rank_counts[rank - 1]) as f64 / denominator as f64;
                expectation += probability
                    * monitor
                        .test_payoff(contrast, rank, next_t)
                        .unwrap();
            }
            assert!(expectation.abs() < 1e-12, "expectation={expectation}");
        }
    }

    #[test]
    fn betting_factors_remain_nonnegative_for_all_ranks() {
        let mut monitor = RankSpectrumMonitor::new(99, 1, 20).unwrap();
        for rank in 1..=100 {
            let t = monitor.observations_seen() + 1;
            for contrast in monitor.contrasts() {
                let payoff = monitor.test_payoff(contrast, rank, t).unwrap();
                assert!((-1.0..=1.0).contains(&payoff));
                assert!(1.0 + BET_LAMBDA * payoff >= 0.5);
                assert!(1.0 - BET_LAMBDA * payoff >= 0.5);
            }
        }
    }

    #[test]
    fn invalid_rank_does_not_mutate_monitor_state() {
        let mut monitor = RankSpectrumMonitor::new(19, 1, 20).unwrap();
        let before = monitor.clone();
        assert_eq!(monitor.observe_rank(0), Err(SpectrumError::InvalidRank));
        assert_eq!(monitor, before);
        assert_eq!(
            monitor.observe_rank(21),
            Err(SpectrumError::InvalidRank)
        );
        assert_eq!(monitor, before);
    }

    #[test]
    fn trimodal_rank_grid_challenge_triggers_quarantine() {
        let mut monitor = RankSpectrumMonitor::new(99, 1, 20).unwrap();
        let mut quarantined_at = None;
        for step in 0..500 {
            let rank = match step % 3 {
                0 => 11,
                1 => 51,
                _ => 91,
            };
            if matches!(
                monitor.observe_rank(rank).unwrap(),
                SpectrumUpdate::Quarantined { .. }
            ) {
                quarantined_at = Some(step + 1);
                break;
            }
        }
        assert!(quarantined_at.is_some());
        assert!(quarantined_at.unwrap() <= 500);
        assert!(monitor.is_quarantined());
        assert_eq!(
            monitor.observe_rank(51),
            Err(SpectrumError::AlreadyQuarantined)
        );
    }

    #[test]
    fn invalid_configuration_fails_closed() {
        assert_eq!(
            RankSpectrumMonitor::new(0, 1, 20),
            Err(SpectrumError::InvalidCalibrationCount)
        );
        assert_eq!(
            RankSpectrumMonitor::new(19, 0, 20),
            Err(SpectrumError::InvalidMiscoverage)
        );
        assert_eq!(
            RankSpectrumMonitor::new(19, 1, 1),
            Err(SpectrumError::InvalidMiscoverage)
        );
        assert_eq!(
            RankSpectrumMonitor::new(MAX_RANK_BUCKETS, 1, 20),
            Err(SpectrumError::TooManyRankBuckets)
        );
    }
}
