#![forbid(unsafe_code)]

//! Restart-time mixture over multi-resolution predictive-rank e-processes.
//!
//! Each frozen Haar contrast is conditionally centered by the exact predictive
//! rank law for a fixed calibration reference. For every contrast, direction,
//! and pre-registered betting rate, a nonnegative e-process is restarted at
//! every observation. The restarts are combined with a summable heavy-tailed
//! start-time prior and an unspent-cash residual. Feature, direction, and rate
//! accounts are then combined by a fixed normalized mixture.
//!
//! The construction is a PRM composition experiment, not a reproduction of the
//! parameter-free CCTM universal-portfolio algorithm. It accepts already
//! qualified ranks and does not authenticate score custody, ties, common-cause
//! closure, block independence, or physical safety.

use std::f64::consts::LN_2;

const BETTING_RATES: [f64; 3] = [0.25, 0.5, 1.0];
const MAX_RANK_BUCKETS: usize = 1 << 18;
const NEG_INFINITY: f64 = f64::NEG_INFINITY;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestartError {
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

#[derive(Clone, Debug, PartialEq)]
struct HaarContrast {
    start: usize,
    midpoint: usize,
    end: usize,
    contrasts_at_level: usize,
    active_resolution_levels: usize,
}

impl HaarContrast {
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

    fn log_feature_weight(&self) -> f64 {
        -(self.active_resolution_levels as f64).ln()
            - (self.contrasts_at_level as f64).ln()
            - 2.0f64.ln()
            - (BETTING_RATES.len() as f64).ln()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Component {
    contrast: HaarContrast,
    direction: i8,
    betting_rate: f64,
    log_weight: f64,
    /// Log of the aggregate wealth of all started accounts for this tuple.
    log_active_wealth: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RestartUpdate {
    Accepted {
        rank_one_based: usize,
        log_mixture_wealth: f64,
        feature_count: usize,
        component_count: usize,
    },
    Quarantined {
        rank_one_based: usize,
        log_mixture_wealth: f64,
        threshold_log_wealth: f64,
        feature_count: usize,
        component_count: usize,
    },
}

/// Restart-time-universal fixed-reference predictive-rank spectrum monitor.
///
/// For calibration size n, there are n+1 possible rank buckets and exactly n
/// nonempty dyadic Haar contrasts. Each contrast has two directions and three
/// fixed betting rates, yielding 6n restarted component mixtures.
///
/// The rank reference and score-to-rank mapping must be frozen upstream. Ranks
/// are one-based in [1, n+1].
#[derive(Clone, Debug, PartialEq)]
pub struct RestartedRankSpectrumMonitor {
    calibration_count: usize,
    alpha_numerator: u32,
    alpha_denominator: u32,
    rank_counts: Vec<u64>,
    observations_seen: u64,
    components: Vec<Component>,
    feature_count: usize,
    quarantined: bool,
}

impl RestartedRankSpectrumMonitor {
    pub fn new(
        calibration_count: usize,
        alpha_numerator: u32,
        alpha_denominator: u32,
    ) -> Result<Self, RestartError> {
        if calibration_count == 0 {
            return Err(RestartError::InvalidCalibrationCount);
        }
        if alpha_numerator == 0 || alpha_denominator == 0 || alpha_numerator >= alpha_denominator {
            return Err(RestartError::InvalidMiscoverage);
        }

        let rank_buckets = calibration_count
            .checked_add(1)
            .ok_or(RestartError::ArithmeticOverflow)?;
        if rank_buckets > MAX_RANK_BUCKETS {
            return Err(RestartError::TooManyRankBuckets);
        }

        let padded_buckets = rank_buckets
            .checked_next_power_of_two()
            .ok_or(RestartError::ArithmeticOverflow)?;
        let maximum_levels = padded_buckets.trailing_zeros() as usize;
        if maximum_levels == 0 {
            return Err(RestartError::EmptyFeatureBank);
        }

        let mut levels: Vec<Vec<HaarContrast>> = Vec::with_capacity(maximum_levels);
        for resolution_level in 0..maximum_levels {
            let width = padded_buckets >> resolution_level;
            let node_count = padded_buckets / width;
            let mut ranges = Vec::new();

            for node in 0..node_count {
                let start = node
                    .checked_mul(width)
                    .ok_or(RestartError::ArithmeticOverflow)?;
                let midpoint = start
                    .checked_add(width / 2)
                    .ok_or(RestartError::ArithmeticOverflow)?;
                let end = start
                    .checked_add(width)
                    .ok_or(RestartError::ArithmeticOverflow)?
                    .min(rank_buckets);

                // A padded node with an empty side has no rank information.
                if start < rank_buckets && midpoint < rank_buckets && end > midpoint {
                    ranges.push((start, midpoint, end));
                }
            }

            if !ranges.is_empty() {
                let count = ranges.len();
                levels.push(
                    ranges
                        .into_iter()
                        .map(|(start, midpoint, end)| HaarContrast {
                            start,
                            midpoint,
                            end,
                            contrasts_at_level: count,
                            active_resolution_levels: 0,
                        })
                        .collect(),
                );
            }
        }

        let active_resolution_levels = levels.len();
        if active_resolution_levels == 0 {
            return Err(RestartError::EmptyFeatureBank);
        }

        let feature_count = levels.iter().map(Vec::len).sum::<usize>();
        if feature_count != calibration_count {
            return Err(RestartError::EmptyFeatureBank);
        }
        let component_count = feature_count
            .checked_mul(2)
            .and_then(|value| value.checked_mul(BETTING_RATES.len()))
            .ok_or(RestartError::ArithmeticOverflow)?;

        let mut components = Vec::with_capacity(component_count);
        for level in levels {
            for mut contrast in level {
                contrast.active_resolution_levels = active_resolution_levels;
                let log_weight = contrast.log_feature_weight();
                for direction in [1i8, -1i8] {
                    for betting_rate in BETTING_RATES {
                        components.push(Component {
                            contrast: contrast.clone(),
                            direction,
                            betting_rate,
                            log_weight,
                            log_active_wealth: NEG_INFINITY,
                        });
                    }
                }
            }
        }

        if components.len() != component_count {
            return Err(RestartError::EmptyFeatureBank);
        }

        let monitor = Self {
            calibration_count,
            alpha_numerator,
            alpha_denominator,
            rank_counts: vec![0; rank_buckets],
            observations_seen: 0,
            components,
            feature_count,
            quarantined: false,
        };
        let initial = monitor.log_mixture_wealth()?;
        if initial.abs() > 1e-10 {
            return Err(RestartError::ArithmeticOverflow);
        }
        Ok(monitor)
    }

    /// Adds an already-qualified rank and returns the updated mixture wealth.
    ///
    /// The active restart mass is updated as
    ///
    /// A_t = (A_{t-1} + pi_t) * (1 + lambda * direction * Z_t),
    ///
    /// while unstarted wealth remains in cash:
    ///
    /// r_t = ln(2) / ln(t+2).
    ///
    /// The rank-count and wealth state are committed only after every component
    /// and the final log-mixture have passed validation.
    pub fn observe_rank(&mut self, rank_one_based: usize) -> Result<RestartUpdate, RestartError> {
        if self.quarantined {
            return Err(RestartError::AlreadyQuarantined);
        }
        if rank_one_based == 0 || rank_one_based > self.calibration_count + 1 {
            return Err(RestartError::InvalidRank);
        }

        let next_seen = self
            .observations_seen
            .checked_add(1)
            .ok_or(RestartError::CounterOverflow)?;
        let rank_index = rank_one_based - 1;
        let next_rank_count = self.rank_counts[rank_index]
            .checked_add(1)
            .ok_or(RestartError::CounterOverflow)?;

        // Predictable rank law: (1 + previous bucket count) / (n + t).
        // Build its weighted prefix sum once per observation.
        let mut weighted_prefix = Vec::with_capacity(self.rank_counts.len() + 1);
        weighted_prefix.push(0i128);
        for count in &self.rank_counts {
            let weight = i128::from(*count)
                .checked_add(1)
                .ok_or(RestartError::ArithmeticOverflow)?;
            let next = weighted_prefix
                .last()
                .copied()
                .ok_or(RestartError::ArithmeticOverflow)?
                .checked_add(weight)
                .ok_or(RestartError::ArithmeticOverflow)?;
            weighted_prefix.push(next);
        }

        let (start_mass, residual_mass) = restart_masses(next_seen)?;
        let log_start_mass = start_mass.ln();
        let log_residual_mass = residual_mass.ln();
        let mut next_active_wealths = Vec::with_capacity(self.components.len());

        for component in &self.components {
            let payoff = centered_payoff_from_prefix(
                self.calibration_count,
                &component.contrast,
                rank_one_based,
                next_seen,
                &weighted_prefix,
            )?;
            if !payoff.is_finite() || !(-1.0 - 1e-12..=1.0 + 1e-12).contains(&payoff) {
                return Err(RestartError::InvalidPredictivePayoff);
            }

            let signed_payoff = payoff * f64::from(component.direction);
            let factor = 1.0 + component.betting_rate * signed_payoff;
            if !factor.is_finite() || factor < -1e-12 {
                return Err(RestartError::InvalidBettingFactor);
            }

            // New mass pi_t enters in cash immediately before round t. A zero
            // factor maps active wealth to zero; -infinity is a valid log state.
            let active_before_bet = log_add_exp(component.log_active_wealth, log_start_mass);
            let next_active = if factor <= 0.0 {
                NEG_INFINITY
            } else {
                active_before_bet + factor.ln()
            };
            if next_active.is_nan() || next_active == f64::INFINITY {
                return Err(RestartError::ArithmeticOverflow);
            }
            next_active_wealths.push(next_active);
        }

        let log_mixture = self.log_mixture_from_active(&next_active_wealths, log_residual_mass)?;
        let threshold = (f64::from(self.alpha_denominator) / f64::from(self.alpha_numerator)).ln();
        if !log_mixture.is_finite() || !threshold.is_finite() {
            return Err(RestartError::ArithmeticOverflow);
        }

        // Commit only after the whole update is known to be valid.
        self.rank_counts[rank_index] = next_rank_count;
        self.observations_seen = next_seen;
        for (component, next_active) in self.components.iter_mut().zip(next_active_wealths) {
            component.log_active_wealth = next_active;
        }

        if log_mixture >= threshold {
            self.quarantined = true;
            Ok(RestartUpdate::Quarantined {
                rank_one_based,
                log_mixture_wealth: log_mixture,
                threshold_log_wealth: threshold,
                feature_count: self.feature_count,
                component_count: self.components.len(),
            })
        } else {
            Ok(RestartUpdate::Accepted {
                rank_one_based,
                log_mixture_wealth: log_mixture,
                feature_count: self.feature_count,
                component_count: self.components.len(),
            })
        }
    }

    pub fn log_mixture_wealth(&self) -> Result<f64, RestartError> {
        let residual = if self.observations_seen == 0 {
            1.0
        } else {
            let (_, residual) = restart_masses(self.observations_seen)?;
            residual
        };
        self.log_mixture_from_active(
            &self
                .components
                .iter()
                .map(|component| component.log_active_wealth)
                .collect::<Vec<_>>(),
            residual.ln(),
        )
    }

    fn log_mixture_from_active(
        &self,
        active_wealths: &[f64],
        log_residual_mass: f64,
    ) -> Result<f64, RestartError> {
        if active_wealths.len() != self.components.len() {
            return Err(RestartError::ArithmeticOverflow);
        }
        let mut log_mixture = NEG_INFINITY;
        for (component, log_active) in self.components.iter().zip(active_wealths) {
            let log_local_wealth = log_add_exp(*log_active, log_residual_mass);
            let weighted = component.log_weight + log_local_wealth;
            log_mixture = log_add_exp(log_mixture, weighted);
        }
        if log_mixture.is_nan() || log_mixture == f64::INFINITY {
            return Err(RestartError::ArithmeticOverflow);
        }
        Ok(log_mixture)
    }

    pub fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    pub fn observations_seen(&self) -> u64 {
        self.observations_seen
    }

    pub fn calibration_reference_count(&self) -> usize {
        self.calibration_count
    }

    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    pub fn threshold_log_wealth(&self) -> f64 {
        (f64::from(self.alpha_denominator) / f64::from(self.alpha_numerator)).ln()
    }
}

fn centered_payoff_from_prefix(
    calibration_count: usize,
    contrast: &HaarContrast,
    rank_one_based: usize,
    observation_index_one_based: u64,
    weighted_prefix: &[i128],
) -> Result<f64, RestartError> {
    if rank_one_based == 0
        || rank_one_based > calibration_count + 1
        || weighted_prefix.len() != calibration_count + 2
    {
        return Err(RestartError::InvalidRank);
    }
    let total = (calibration_count as u128)
        .checked_add(u128::from(observation_index_one_based))
        .ok_or(RestartError::ArithmeticOverflow)?;
    let total_i128 = i128::try_from(total).map_err(|_| RestartError::ArithmeticOverflow)?;

    let positive_mass = weighted_prefix[contrast.midpoint]
        .checked_sub(weighted_prefix[contrast.start])
        .ok_or(RestartError::ArithmeticOverflow)?;
    let negative_mass = weighted_prefix[contrast.end]
        .checked_sub(weighted_prefix[contrast.midpoint])
        .ok_or(RestartError::ArithmeticOverflow)?;
    let expected_code_numerator = positive_mass
        .checked_sub(negative_mass)
        .ok_or(RestartError::ArithmeticOverflow)?;
    let observed_code = contrast.code_for_rank(rank_one_based);
    let centered =
        (observed_code as f64 - expected_code_numerator as f64 / total_i128 as f64) * 0.5;
    if !centered.is_finite() {
        return Err(RestartError::ArithmeticOverflow);
    }
    Ok(centered)
}

fn restart_masses(start_time_one_based: u64) -> Result<(f64, f64), RestartError> {
    if start_time_one_based == 0 {
        return Err(RestartError::ArithmeticOverflow);
    }
    let time = start_time_one_based as f64;
    let first_time = time + 1.0;
    let second_time = time + 2.0;
    let first_log = first_time.ln();
    let second_log = second_time.ln();
    // Stable form of 1/ln(t+1) - 1/ln(t+2): subtracting the
    // reciprocals directly rounds to zero for sufficiently large t.
    let log_increment = (1.0 / first_time).ln_1p();
    let start_mass = LN_2 * log_increment / (first_log * second_log);
    let residual_mass = LN_2 / second_log;
    if !start_mass.is_finite()
        || !residual_mass.is_finite()
        || start_mass <= 0.0
        || residual_mass <= 0.0
        || start_mass > residual_mass
    {
        return Err(RestartError::ArithmeticOverflow);
    }
    Ok((start_mass, residual_mass))
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    if left == NEG_INFINITY {
        return right;
    }
    if right == NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor() -> RestartedRankSpectrumMonitor {
        RestartedRankSpectrumMonitor::new(15, 1, 20).expect("valid monitor")
    }

    #[test]
    fn initial_mixture_has_unit_wealth_and_expected_component_count() {
        let monitor = make_monitor();
        assert_eq!(monitor.feature_count(), 15);
        assert_eq!(monitor.component_count(), 90);
        assert!(monitor.log_mixture_wealth().unwrap().abs() < 1e-10);
    }

    #[test]
    fn heavy_tailed_start_mass_and_residual_telescope_to_one() {
        let mut started = 0.0;
        for time in 1..=100_000 {
            let (mass, residual) = restart_masses(time).unwrap();
            started += mass;
            assert!((started + residual - 1.0).abs() < 2e-10);
        }
    }

    #[test]
    fn neutral_restart_recurrence_never_creates_or_loses_wealth() {
        let mut log_active = NEG_INFINITY;
        for time in 1..=10_000 {
            let (start_mass, residual_mass) = restart_masses(time).unwrap();
            log_active = log_add_exp(log_active, start_mass.ln());
            let total = log_add_exp(log_active, residual_mass.ln()).exp();
            assert!((total - 1.0).abs() < 2e-10);
        }
    }

    #[test]
    fn rank_feature_payoffs_are_exactly_predictively_centered() {
        let monitor = make_monitor();
        let contrast = &monitor.components[0].contrast;
        let counts = [2u64, 0, 4, 1, 3, 0, 0, 2, 1, 0, 1, 0, 0, 0, 1, 2];
        let mut prefix = vec![0i128];
        for count in counts {
            prefix.push(prefix.last().copied().unwrap() + i128::from(count) + 1);
        }
        let t = 18u64;
        let denominator = 15.0 + t as f64;
        let mut weighted_payoff = 0.0;
        for rank in 1..=16 {
            let mass = (prefix[rank] - prefix[rank - 1]) as f64 / denominator;
            let payoff = centered_payoff_from_prefix(15, contrast, rank, t, &prefix).unwrap();
            weighted_payoff += mass * payoff;
        }
        assert!(weighted_payoff.abs() < 1e-12);
    }

    #[test]
    fn every_registered_rate_has_a_nonnegative_factor_over_all_ranks() {
        let monitor = make_monitor();
        let prefix = (0..=16).map(|value| value as i128).collect::<Vec<_>>();
        for component in &monitor.components {
            for rank in 1..=16 {
                let payoff =
                    centered_payoff_from_prefix(15, &component.contrast, rank, 1, &prefix).unwrap();
                let factor = 1.0 + component.betting_rate * payoff * f64::from(component.direction);
                assert!(factor >= -1e-12);
            }
        }
    }

    #[test]
    fn valid_observation_updates_all_restart_accounts_atomically() {
        let mut monitor = make_monitor();
        let update = monitor.observe_rank(16).unwrap();
        assert!(matches!(
            update,
            RestartUpdate::Accepted { .. } | RestartUpdate::Quarantined { .. }
        ));
        assert_eq!(monitor.observations_seen(), 1);
        assert_eq!(monitor.rank_counts[15], 1);
        assert!(monitor.log_mixture_wealth().unwrap().is_finite());
    }

    #[test]
    fn invalid_rank_does_not_mutate_any_state() {
        let mut monitor = make_monitor();
        let before = monitor.clone();
        assert_eq!(monitor.observe_rank(0), Err(RestartError::InvalidRank));
        assert_eq!(monitor, before);
        assert_eq!(monitor.observe_rank(17), Err(RestartError::InvalidRank));
        assert_eq!(monitor, before);
    }

    #[test]
    fn counter_overflow_fails_before_state_mutation() {
        let mut monitor = make_monitor();
        monitor.observations_seen = u64::MAX;
        let before_counts = monitor.rank_counts.clone();
        let before_components = monitor.components.clone();
        assert_eq!(monitor.observe_rank(1), Err(RestartError::CounterOverflow));
        assert_eq!(monitor.rank_counts, before_counts);
        assert_eq!(monitor.components, before_components);
        assert_eq!(monitor.observations_seen, u64::MAX);
    }

    #[test]
    fn log_domain_represents_zero_accounts_and_enforces_quarantine() {
        assert_eq!(log_add_exp(NEG_INFINITY, NEG_INFINITY), NEG_INFINITY);
        let mut monitor = make_monitor();
        // Force a valid, very high pre-existing e-value to exercise thresholding
        // without relying on a particular random path.
        for component in &mut monitor.components {
            component.log_active_wealth = 100.0;
        }
        let result = monitor.observe_rank(1).unwrap();
        assert!(matches!(result, RestartUpdate::Quarantined { .. }));
        assert!(monitor.is_quarantined());
        assert_eq!(
            monitor.observe_rank(1),
            Err(RestartError::AlreadyQuarantined)
        );
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        assert_eq!(
            RestartedRankSpectrumMonitor::new(0, 1, 20).err(),
            Some(RestartError::InvalidCalibrationCount)
        );
        assert_eq!(
            RestartedRankSpectrumMonitor::new(10, 0, 20).err(),
            Some(RestartError::InvalidMiscoverage)
        );
        assert_eq!(
            RestartedRankSpectrumMonitor::new(10, 1, 1).err(),
            Some(RestartError::InvalidMiscoverage)
        );
    }

    #[test]
    fn restart_mass_remains_positive_at_the_u64_horizon_limit() {
        let (start_mass, residual_mass) = restart_masses(u64::MAX).unwrap();
        assert!(start_mass > 0.0);
        assert!(residual_mass > 0.0);
        assert!(start_mass < residual_mass);
    }

    #[test]
    fn late_start_prior_retains_positive_cash_for_a_late_change() {
        let (start_mass, residual_mass) = restart_masses(8_001).unwrap();
        assert!(start_mass > 0.0);
        assert!(residual_mass > 0.0);
        let time = 8_001.0f64;
        let asymptotic_cost = time.ln() + 2.0 * time.ln().ln() - LN_2.ln();
        assert!((-start_mass.ln() - asymptotic_cost).abs() < 1.0);
    }
}
