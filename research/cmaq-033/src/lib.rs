#![forbid(unsafe_code)]

//! ONS-adaptive predictive-rank betting for a frozen calibration reference.
//!
//! This crate implements Order and Dispersion PRM betting using the same
//! fixed-reference predictive rank law as CMAQ-031. It is an experimental
//! betting kernel: callers must obtain ranks from an independently validated
//! reference-comparison and evidence-ingestion boundary. It does not authenticate
//! score receipts, tie-break randomness or cluster independence.

use cmaq_031_anytime_calibration_monitor::{centered_rank_payoff, MonitorError, RankFeature};

#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveComponent {
    pub feature: RankFeature,
    /// Predictable bet used on the next rank.
    pub lambda: f64,
    /// A_t = sum of squared ONS gradients.
    pub squared_gradient_sum: f64,
    pub log_wealth: f64,
    pub last_centered_payoff: f64,
    /// Predictable radius computed before the next observation.
    pub projection_bound: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OnsUpdate {
    Accepted {
        rank_one_based: usize,
        log_mixture_wealth: f64,
        components: Vec<AdaptiveComponent>,
    },
    Quarantined {
        rank_one_based: usize,
        log_mixture_wealth: f64,
        threshold_log_wealth: f64,
        components: Vec<AdaptiveComponent>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OnsError {
    InvalidCalibrationCount,
    InvalidMiscoverage,
    InvalidRank,
    CounterOverflow,
    ArithmeticOverflow,
    InvalidPayoff,
    InvalidBettingFactor,
    AlreadyQuarantined,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OnsPredictiveRankMonitor {
    calibration_count: usize,
    alpha_numerator: u32,
    alpha_denominator: u32,
    rank_counts: Vec<u64>,
    observations_seen: u64,
    components: Vec<AdaptiveComponent>,
    quarantined: bool,
}

const ONS_STEP: f64 = 4.5;
const MIN_FACTOR: f64 = 0.25;
const ROUNDING_GUARD: f64 = 16.0 * f64::EPSILON;

impl OnsPredictiveRankMonitor {
    /// Creates an ONS portfolio with the canonical Order and Dispersion features.
    ///
    /// Signed ONS bets already adapt to positive and negative departures;
    /// separate opposite-sign features would duplicate the same betting process.
    pub fn new(
        calibration_count: usize,
        alpha_numerator: u32,
        alpha_denominator: u32,
    ) -> Result<Self, OnsError> {
        if calibration_count == 0 {
            return Err(OnsError::InvalidCalibrationCount);
        }
        if alpha_numerator == 0
            || alpha_denominator == 0
            || alpha_numerator >= alpha_denominator
        {
            return Err(OnsError::InvalidMiscoverage);
        }

        let rank_counts = vec![0; calibration_count + 1];
        let order_bound = projection_bound(
            calibration_count,
            &rank_counts,
            1,
            RankFeature::UpperOrder,
        )?;
        let dispersion_bound = projection_bound(
            calibration_count,
            &rank_counts,
            1,
            RankFeature::TailDispersion,
        )?;
        let components = vec![
            AdaptiveComponent {
                feature: RankFeature::UpperOrder,
                lambda: 0.0,
                squared_gradient_sum: 0.0,
                log_wealth: 0.0,
                last_centered_payoff: 0.0,
                projection_bound: order_bound,
            },
            AdaptiveComponent {
                feature: RankFeature::TailDispersion,
                lambda: 0.0,
                squared_gradient_sum: 0.0,
                log_wealth: 0.0,
                last_centered_payoff: 0.0,
                projection_bound: dispersion_bound,
            },
        ];

        Ok(Self {
            calibration_count,
            alpha_numerator,
            alpha_denominator,
            rank_counts,
            observations_seen: 0,
            components,
            quarantined: false,
        })
    }

    /// Adds an already-computed rank from the fixed calibration reference.
    ///
    /// Rank one is the lowest placement; n+1 is the highest. The caller must
    /// establish that the rank came from an independently reconciled cluster
    /// block, with independently randomized tie-breaking where required.
    pub fn observe_rank(&mut self, rank_one_based: usize) -> Result<OnsUpdate, OnsError> {
        if self.quarantined {
            return Err(OnsError::AlreadyQuarantined);
        }
        if rank_one_based == 0 || rank_one_based > self.calibration_count + 1 {
            return Err(OnsError::InvalidRank);
        }
        let next_observation = self
            .observations_seen
            .checked_add(1)
            .ok_or(OnsError::CounterOverflow)?;
        let mut next_components = self.components.clone();

        for component in &mut next_components {
            let payoff = centered_rank_payoff(
                self.calibration_count,
                &self.rank_counts,
                next_observation,
                rank_one_based,
                component.feature,
            )
            .map_err(map_monitor_error)?;
            if !payoff.is_finite() || !(-1.0..=1.0).contains(&payoff) {
                return Err(OnsError::InvalidPayoff);
            }
            let factor = 1.0 + component.lambda * payoff;
            if !factor.is_finite() || factor < MIN_FACTOR - 1e-12 {
                return Err(OnsError::InvalidBettingFactor);
            }
            let gradient = payoff / factor;
            if !gradient.is_finite() {
                return Err(OnsError::ArithmeticOverflow);
            }
            let next_gradient_sum = component.squared_gradient_sum + gradient * gradient;
            if !next_gradient_sum.is_finite() {
                return Err(OnsError::ArithmeticOverflow);
            }
            component.log_wealth += factor.ln();
            if !component.log_wealth.is_finite() {
                return Err(OnsError::ArithmeticOverflow);
            }
            component.last_centered_payoff = payoff;
            component.squared_gradient_sum = next_gradient_sum;
        }

        let rank_index = rank_one_based - 1;
        let next_count = self.rank_counts[rank_index]
            .checked_add(1)
            .ok_or(OnsError::CounterOverflow)?;
        let mut next_counts = self.rank_counts.clone();
        next_counts[rank_index] = next_count;
        let next_step = next_observation
            .checked_add(1)
            .ok_or(OnsError::CounterOverflow)?;

        for index in 0..next_components.len() {
            let bound = projection_bound(
                self.calibration_count,
                &next_counts,
                next_step,
                next_components[index].feature,
            )?;
            let previous = &self.components[index];
            let current = &mut next_components[index];
            let gradient = current.last_centered_payoff
                / (1.0 + previous.lambda * current.last_centered_payoff);
            let candidate =
                previous.lambda + ONS_STEP * gradient / (1.0 + current.squared_gradient_sum);
            current.lambda = candidate.clamp(-bound, bound);
            current.projection_bound = bound;
            if !current.lambda.is_finite()
                || current.lambda.abs() > current.projection_bound + f64::EPSILON
            {
                return Err(OnsError::ArithmeticOverflow);
            }
        }

        let log_mixture_wealth = log_mean_exp(
            &next_components
                .iter()
                .map(|component| component.log_wealth)
                .collect::<Vec<_>>(),
        )?;
        let threshold_log_wealth =
            (f64::from(self.alpha_denominator) / f64::from(self.alpha_numerator)).ln();

        // State is committed only after every feature, bound and mixture succeeds.
        self.rank_counts = next_counts;
        self.observations_seen = next_observation;
        self.components = next_components;

        if log_mixture_wealth >= threshold_log_wealth {
            self.quarantined = true;
            Ok(OnsUpdate::Quarantined {
                rank_one_based,
                log_mixture_wealth,
                threshold_log_wealth,
                components: self.components.clone(),
            })
        } else {
            Ok(OnsUpdate::Accepted {
                rank_one_based,
                log_mixture_wealth,
                components: self.components.clone(),
            })
        }
    }

    pub fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    pub fn observations_seen(&self) -> u64 {
        self.observations_seen
    }

    pub fn rank_counts(&self) -> &[u64] {
        &self.rank_counts
    }

    pub fn components(&self) -> &[AdaptiveComponent] {
        &self.components
    }

    pub fn log_mixture_wealth(&self) -> Result<f64, OnsError> {
        log_mean_exp(
            &self
                .components
                .iter()
                .map(|component| component.log_wealth)
                .collect::<Vec<_>>(),
        )
    }
}

fn projection_bound(
    calibration_count: usize,
    counts: &[u64],
    observation_index_one_based: u64,
    feature: RankFeature,
) -> Result<f64, OnsError> {
    if calibration_count == 0
        || counts.len() != calibration_count + 1
        || observation_index_one_based == 0
    {
        return Err(OnsError::InvalidPayoff);
    }
    let total_u128 = (calibration_count as u128)
        .checked_add(u128::from(observation_index_one_based))
        .ok_or(OnsError::ArithmeticOverflow)?;
    let total = i128::try_from(total_u128).map_err(|_| OnsError::ArithmeticOverflow)?;
    let mut weighted_sum = 0i128;
    for (index, count) in counts.iter().enumerate() {
        let weight = i128::try_from(
            u128::from(*count)
                .checked_add(1)
                .ok_or(OnsError::ArithmeticOverflow)?,
        )
        .map_err(|_| OnsError::ArithmeticOverflow)?;
        let feature_value = feature_numerator(index, calibration_count, feature)?;
        weighted_sum = weighted_sum
            .checked_add(
                weight
                    .checked_mul(feature_value)
                    .ok_or(OnsError::ArithmeticOverflow)?,
            )
            .ok_or(OnsError::ArithmeticOverflow)?;
    }
    let denominator = (calibration_count as u128)
        .checked_mul(2)
        .and_then(|value| value.checked_mul(total_u128))
        .ok_or(OnsError::ArithmeticOverflow)?;
    let denominator = denominator as f64;
    let mut maximum = 0i128;
    for index in 0..=calibration_count {
        let centered = feature_numerator(index, calibration_count, feature)?
            .checked_mul(total)
            .and_then(|value| value.checked_sub(weighted_sum))
            .ok_or(OnsError::ArithmeticOverflow)?;
        maximum = maximum.max(
            centered
                .checked_abs()
                .ok_or(OnsError::ArithmeticOverflow)?,
        );
    }
    let maximum = maximum as f64;
    if !denominator.is_finite() || denominator <= 0.0 || !maximum.is_finite() {
        return Err(OnsError::ArithmeticOverflow);
    }
    let bound = if maximum == 0.0 {
        1.0
    } else {
        (0.75 / (maximum / denominator)).min(1.0)
    };
    let conservative = if bound < 1.0 {
        (bound * (1.0 - ROUNDING_GUARD)).next_down()
    } else {
        1.0
    };
    if !conservative.is_finite() || !(0.0..=1.0).contains(&conservative) {
        return Err(OnsError::ArithmeticOverflow);
    }
    Ok(conservative)
}

fn feature_numerator(
    rank_index_zero_based: usize,
    calibration_count: usize,
    feature: RankFeature,
) -> Result<i128, OnsError> {
    let index = i128::try_from(rank_index_zero_based).map_err(|_| OnsError::ArithmeticOverflow)?;
    let n = i128::try_from(calibration_count).map_err(|_| OnsError::ArithmeticOverflow)?;
    let order_offset = index
        .checked_mul(2)
        .and_then(|value| value.checked_sub(n))
        .ok_or(OnsError::ArithmeticOverflow)?;
    match feature {
        RankFeature::UpperOrder => Ok(order_offset),
        RankFeature::LowerOrder => Ok(-order_offset),
        RankFeature::TailDispersion => order_offset
            .checked_abs()
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_sub(n))
            .ok_or(OnsError::ArithmeticOverflow),
        RankFeature::CenterConcentration => order_offset
            .checked_abs()
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_sub(n))
            .and_then(|value| value.checked_neg())
            .ok_or(OnsError::ArithmeticOverflow),
    }
}

fn log_mean_exp(values: &[f64]) -> Result<f64, OnsError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(OnsError::ArithmeticOverflow);
    }
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values.iter().map(|value| (value - maximum).exp()).sum();
    let result = maximum + (sum / values.len() as f64).ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(OnsError::ArithmeticOverflow)
    }
}

fn map_monitor_error(error: MonitorError) -> OnsError {
    match error {
        MonitorError::CounterOverflow => OnsError::CounterOverflow,
        MonitorError::ArithmeticOverflow => OnsError::ArithmeticOverflow,
        _ => OnsError::InvalidPayoff,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictive_rank_feature_payoffs_are_conditionally_centered() {
        let n = 9;
        let mut counts = vec![0_u64; n + 1];
        counts[1] = 2;
        counts[6] = 1;
        let total = (n + 4) as f64;
        for feature in [RankFeature::UpperOrder, RankFeature::TailDispersion] {
            let expected = (0..=n)
                .map(|rank| {
                    ((1 + counts[rank]) as f64 / total)
                        * centered_rank_payoff(n, &counts, 4, rank + 1, feature).unwrap()
                })
                .sum::<f64>();
            assert!(expected.abs() < 1e-12);
        }
    }

    #[test]
    fn ons_projection_bound_covers_every_possible_next_rank() {
        let mut monitor = OnsPredictiveRankMonitor::new(13, 1, 20).unwrap();
        for rank in [13, 1, 7, 12, 2, 8, 8, 13, 1] {
            for component in monitor.components() {
                assert!(component.lambda.abs() <= component.projection_bound + 1e-12);
                for candidate_rank in 1..=14 {
                    let payoff = centered_rank_payoff(
                        13,
                        monitor.rank_counts(),
                        monitor.observations_seen() + 1,
                        candidate_rank,
                        component.feature,
                    )
                    .unwrap();
                    let factor = 1.0 + component.lambda * payoff;
                    assert!(factor >= MIN_FACTOR - 1e-12);
                }
            }
            let update = monitor.observe_rank(rank).unwrap();
            let components = match update {
                OnsUpdate::Accepted { components, .. }
                | OnsUpdate::Quarantined { components, .. } => components,
            };
            for component in components {
                assert!(component.lambda.abs() <= component.projection_bound + 1e-12);
            }
        }
    }

    #[test]
    fn ons_uses_only_past_payoff_to_choose_next_bet() {
        let mut monitor = OnsPredictiveRankMonitor::new(17, 1, 20).unwrap();
        assert_eq!(monitor.components()[0].lambda, 0.0);
        let first = monitor.observe_rank(18).unwrap();
        let components = match first {
            OnsUpdate::Accepted { components, .. }
            | OnsUpdate::Quarantined { components, .. } => components,
        };
        assert!(components[0].lambda > 0.0);
        assert!(components[1].lambda > 0.0);

        let mut lower_shift = OnsPredictiveRankMonitor::new(17, 1, 20).unwrap();
        let lower = lower_shift.observe_rank(1).unwrap();
        let lower_components = match lower {
            OnsUpdate::Accepted { components, .. }
            | OnsUpdate::Quarantined { components, .. } => components,
        };
        assert!(lower_components[0].lambda < 0.0);
    }

    #[test]
    fn symmetric_tail_mass_increases_dispersion_bet() {
        let mut monitor = OnsPredictiveRankMonitor::new(20, 1, 20).unwrap();
        for rank in [1, 21, 1, 21, 1, 21, 1, 21] {
            let _ = monitor.observe_rank(rank).unwrap();
        }
        assert!(monitor.components()[1].lambda > 0.0);
    }

    #[test]
    fn centre_concentration_is_the_opposite_dispersion_direction() {
        let n = 20;
        let mut counts = vec![0_u64; n + 1];
        counts[n / 2] = 3;
        let t = 4;
        for rank in [8, 9, 10, 11, 12] {
            let tail =
                centered_rank_payoff(n, &counts, t, rank, RankFeature::TailDispersion).unwrap();
            let center = centered_rank_payoff(
                n,
                &counts,
                t,
                rank,
                RankFeature::CenterConcentration,
            )
            .unwrap();
            assert!((tail + center).abs() < 1e-12);
        }
    }

    #[test]
    fn extreme_persistent_shift_triggers_permanent_quarantine() {
        let mut monitor = OnsPredictiveRankMonitor::new(29, 1, 20).unwrap();
        let mut alarmed = false;
        for _ in 0..500 {
            match monitor.observe_rank(29).unwrap() {
                OnsUpdate::Quarantined { .. } => {
                    alarmed = true;
                    break;
                }
                OnsUpdate::Accepted { .. } => {}
            }
        }
        assert!(alarmed);
        assert!(monitor.is_quarantined());
        assert_eq!(
            monitor.observe_rank(29),
            Err(OnsError::AlreadyQuarantined)
        );
    }

    #[test]
    fn invalid_rank_does_not_mutate_monitor() {
        let mut monitor = OnsPredictiveRankMonitor::new(8, 1, 20).unwrap();
        let before = monitor.clone();
        assert_eq!(monitor.observe_rank(0), Err(OnsError::InvalidRank));
        assert_eq!(monitor, before);
        assert_eq!(monitor.observe_rank(10), Err(OnsError::InvalidRank));
        assert_eq!(monitor, before);
    }

    #[test]
    fn current_factor_stays_above_one_quarter() {
        let mut monitor = OnsPredictiveRankMonitor::new(11, 1, 20).unwrap();
        for rank in [1, 11, 1, 11, 1, 11, 1, 11, 1, 11] {
            let before = monitor.components().to_vec();
            let next = monitor.observe_rank(rank).unwrap();
            let after = match next {
                OnsUpdate::Accepted { components, .. }
                | OnsUpdate::Quarantined { components, .. } => components,
            };
            for (old, new) in before.iter().zip(&after) {
                let factor = 1.0 + old.lambda * new.last_centered_payoff;
                assert!(factor >= MIN_FACTOR - 1e-12);
            }
        }
    }
}
