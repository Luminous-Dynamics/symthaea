// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Predictive-rank martingale core for anytime-valid shift monitoring.
//!
//! This module implements an Order/Dispersion predictive-rank portfolio based on Kuang & Xia,
//! “Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales” (September
//! 2026 preprint). The reference sample is fixed for the entire monitor epoch. This is distinct
//! from repeatedly applying a batch conformal p-value, and distinct from the finite-horizon
//! Bonferroni baseline in sequential_adequacy.
//!
//! Validity depends on the data protocol, not on this Rust type alone. The paper's anytime
//! type-I guarantee is marginal over the jointly sampled calibration reference and online stream;
//! it is not generally conditional on the exact realized numeric reference sample. Under the null,
//! calibration and online scores must be i.i.d./exchangeable under a frozen scoring rule, and tie
//! breakers for atomic scores must be independent Uniform(0,1) draws. The martingale filtration is
//! the history of ranks, not arbitrary information about raw scores. Callers pass tie breakers
//! explicitly so their source and provenance can be recorded. This implementation validates their
//! numeric range but cannot prove randomness, independence, or exchangeability.
//!
//! In an adaptive experiment, query choices can make raw residual scores non-identically
//! distributed even when the physical law is correct. Do not feed such residuals directly into
//! this PRM unless a predeclared transformation and the actual data protocol justify the required
//! joint-exchangeability null. For known conditional predictive distributions, a prequential PIT
//! martingale is a separate and more appropriate hypothesis test.
//!
//! The portfolio is 0.5 times the Order e-process plus 0.5 times the Dispersion e-process. Each
//! constituent uses the paper's online Newton step betting update. Wealth is accumulated in log
//! space. Crossing wealth 1/alpha raises a terminal alarm. A non-alarm is not proof of adequacy;
//! either feature can have low power against departures that it does not contrast with.

use std::cmp::Ordering;

/// Configuration, protocol, or arithmetic errors for predictive-rank monitoring.
#[derive(Clone, Debug, PartialEq)]
pub enum PredictiveRankError {
    /// The significance level must be finite and strictly between zero and one.
    InvalidSignificanceLevel,
    /// At least one calibration score is required.
    EmptyCalibrationReference,
    /// There must be one tie breaker for every calibration score.
    CalibrationTieBreakerLengthMismatch {
        /// Number of calibration scores supplied.
        score_count: usize,
        /// Number of calibration tie breakers supplied.
        tie_breaker_count: usize,
    },
    /// A calibration score is not finite.
    NonFiniteCalibrationScore {
        /// Zero-based index in the caller-supplied calibration array.
        index: usize,
    },
    /// A calibration tie breaker is not finite or is outside [0,1).
    InvalidCalibrationTieBreaker {
        /// Zero-based index in the caller-supplied calibration array.
        index: usize,
    },
    /// Two calibration entries have the same (score, tie-breaker) key.
    DuplicateCalibrationKey {
        /// Zero-based position of the duplicate in the sorted reference.
        sorted_index: usize,
    },
    /// An online score is not finite.
    InvalidAuditScore,
    /// An online tie breaker is not finite or is outside [0,1).
    InvalidAuditTieBreaker,
    /// The online (score, tie-breaker) key exactly matches a fixed reference key.
    AuditKeyCollidesWithReference {
        /// Zero-based position of the matching item in the sorted reference.
        sorted_index: usize,
    },
    /// The monitor has raised an alarm and cannot be reused in this epoch.
    AlreadyAlarmed,
    /// An intermediate computation produced an invalid or non-finite value.
    NumericalFailure,
    /// The online observation counter or a rank count would overflow.
    ObservationCounterOverflow,
}

/// Terminal state of one predictive-rank monitoring epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictiveRankState {
    /// More observations may be processed under the frozen scoring/reference protocol.
    Monitoring,
    /// The portfolio crossed its predeclared threshold; start investigation or a new epoch.
    Alarmed,
}

/// Decision emitted for the latest online score.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictiveRankDecision {
    /// The portfolio has not crossed its alarm threshold.
    Continue,
    /// The portfolio crossed 1/alpha; this epoch is terminal.
    Alarm,
}

/// Observable result for one online score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictiveRankResult {
    /// One-based index of this online observation within the current epoch.
    pub audit_index: usize,
    /// One-based rank, in 1..=n+1, among the frozen reference and this score's randomized key.
    pub rank: usize,
    /// Order-feature conditional payoff after subtracting its predictive-rank null expectation.
    pub order_payoff_z: f64,
    /// Dispersion-feature conditional payoff after subtracting its predictive-rank null expectation.
    pub dispersion_payoff_z: f64,
    /// Natural log wealth of the Order e-process.
    pub order_log_wealth: f64,
    /// Natural log wealth of the Dispersion e-process.
    pub dispersion_log_wealth: f64,
    /// Natural log wealth of the 50/50 Order/Dispersion portfolio, computed stably.
    pub portfolio_log_wealth: f64,
    /// Natural log of the alarm wealth threshold, -ln(alpha).
    pub log_alarm_threshold: f64,
    /// Policy decision after evaluating this observation.
    pub decision: PredictiveRankDecision,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FeatureState {
    lambda: f64,
    log_wealth: f64,
    squared_gradient_sum: f64,
}

impl FeatureState {
    fn initial() -> Self {
        Self {
            lambda: 0.0,
            log_wealth: 0.0,
            squared_gradient_sum: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum RankFeature {
    Order,
    Dispersion,
}

impl RankFeature {
    fn value(self, category: usize, reference_count: usize) -> f64 {
        let u = category as f64 / reference_count as f64;
        match self {
            Self::Order => u - 0.5,
            Self::Dispersion => (2.0 * u - 1.0).abs() - 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FeatureStats {
    mean: f64,
    bet_bound: f64,
}

#[derive(Clone, Copy, Debug)]
struct FeatureObservation {
    log_wealth: f64,
    gradient: f64,
    squared_gradient_sum: f64,
}

/// Fixed-reference predictive-rank martingale with an Order/Dispersion portfolio.
///
/// Calibration entries are sorted by score and then tie breaker. Each online observation is
/// ranked against this frozen reference, while the conditional rank law adapts to the history
/// of earlier ranks. No calibration observations are added or replaced after an alarm or shift.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictiveRankMartingale {
    sorted_reference: Vec<(f64, f64)>,
    rank_counts: Vec<usize>,
    order: FeatureState,
    dispersion: FeatureState,
    alpha: f64,
    log_alarm_threshold: f64,
    observations_seen: usize,
    state: PredictiveRankState,
}

impl PredictiveRankMartingale {
    /// Construct an Order/Dispersion predictive-rank portfolio.
    ///
    /// Calibration scores and tie breakers must be paired. For atomic scores, tie breakers
    /// should be independent Uniform(0,1) draws, independent of the scores and the online
    /// tie breakers. For continuous scores the tie breakers should still be supplied to keep
    /// the ordering protocol explicit, though ties should occur with probability zero.
    pub fn new(
        calibration_scores: &[f64],
        calibration_tie_breakers: &[f64],
        alpha: f64,
    ) -> Result<Self, PredictiveRankError> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(PredictiveRankError::InvalidSignificanceLevel);
        }
        if calibration_scores.is_empty() {
            return Err(PredictiveRankError::EmptyCalibrationReference);
        }
        if calibration_scores.len() != calibration_tie_breakers.len() {
            return Err(PredictiveRankError::CalibrationTieBreakerLengthMismatch {
                score_count: calibration_scores.len(),
                tie_breaker_count: calibration_tie_breakers.len(),
            });
        }
        let category_count = calibration_scores
            .len()
            .checked_add(1)
            .ok_or(PredictiveRankError::ObservationCounterOverflow)?;

        let mut sorted_reference = Vec::with_capacity(calibration_scores.len());
        for (index, (score, tie_breaker)) in calibration_scores
            .iter()
            .copied()
            .zip(calibration_tie_breakers.iter().copied())
            .enumerate()
        {
            if !score.is_finite() {
                return Err(PredictiveRankError::NonFiniteCalibrationScore { index });
            }
            if !valid_tie_breaker(tie_breaker) {
                return Err(PredictiveRankError::InvalidCalibrationTieBreaker { index });
            }
            sorted_reference.push((score, tie_breaker));
        }

        sorted_reference.sort_by(compare_keys);
        for index in 1..sorted_reference.len() {
            if compare_keys(&sorted_reference[index - 1], &sorted_reference[index])
                == Ordering::Equal
            {
                return Err(PredictiveRankError::DuplicateCalibrationKey {
                    sorted_index: index,
                });
            }
        }

        Ok(Self {
            sorted_reference,
            rank_counts: vec![0; category_count],
            order: FeatureState::initial(),
            dispersion: FeatureState::initial(),
            alpha,
            log_alarm_threshold: -alpha.ln(),
            observations_seen: 0,
            state: PredictiveRankState::Monitoring,
        })
    }

    /// Score one independent online observation.
    ///
    /// The caller supplies the tie breaker so a run receipt can preserve its generator/source.
    /// The fixed reference is never updated. Invalid inputs return an error without consuming
    /// the observation index or modifying monitor state.
    pub fn observe(
        &mut self,
        audit_score: f64,
        audit_tie_breaker: f64,
    ) -> Result<PredictiveRankResult, PredictiveRankError> {
        if self.state == PredictiveRankState::Alarmed {
            return Err(PredictiveRankError::AlreadyAlarmed);
        }
        if !audit_score.is_finite() {
            return Err(PredictiveRankError::InvalidAuditScore);
        }
        if !valid_tie_breaker(audit_tie_breaker) {
            return Err(PredictiveRankError::InvalidAuditTieBreaker);
        }

        let audit_index = self
            .observations_seen
            .checked_add(1)
            .ok_or(PredictiveRankError::ObservationCounterOverflow)?;
        // A non-alarming observation also needs the next time index for the predictable ONS
        // projection. Reject the practically unreachable overflow before mutating this epoch.
        let next_audit_index = audit_index
            .checked_add(1)
            .ok_or(PredictiveRankError::ObservationCounterOverflow)?;

        let key = (audit_score, audit_tie_breaker);
        let reference_count_le = self.sorted_reference.partition_point(|reference_key| {
            compare_keys(reference_key, &key) != Ordering::Greater
        });
        if reference_count_le > 0
            && compare_keys(&self.sorted_reference[reference_count_le - 1], &key) == Ordering::Equal
        {
            return Err(PredictiveRankError::AuditKeyCollidesWithReference {
                sorted_index: reference_count_le - 1,
            });
        }
        let rank_category = reference_count_le;
        let rank = rank_category + 1;
        let n = self.sorted_reference.len();

        let order_stats = feature_stats(&self.rank_counts, n, audit_index, RankFeature::Order)?;
        let dispersion_stats =
            feature_stats(&self.rank_counts, n, audit_index, RankFeature::Dispersion)?;
        let order_z = RankFeature::Order.value(rank_category, n) - order_stats.mean;
        let dispersion_z = RankFeature::Dispersion.value(rank_category, n) - dispersion_stats.mean;

        let order_observation = feature_observation(self.order, order_z, order_stats.bet_bound)?;
        let dispersion_observation =
            feature_observation(self.dispersion, dispersion_z, dispersion_stats.bet_bound)?;
        let portfolio_log_wealth = log_mean_exp(
            order_observation.log_wealth,
            dispersion_observation.log_wealth,
        )?;
        let decision = if portfolio_log_wealth >= self.log_alarm_threshold {
            PredictiveRankDecision::Alarm
        } else {
            PredictiveRankDecision::Continue
        };

        let result = PredictiveRankResult {
            audit_index,
            rank,
            order_payoff_z: order_z,
            dispersion_payoff_z: dispersion_z,
            order_log_wealth: order_observation.log_wealth,
            dispersion_log_wealth: dispersion_observation.log_wealth,
            portfolio_log_wealth,
            log_alarm_threshold: self.log_alarm_threshold,
            decision,
        };

        if decision == PredictiveRankDecision::Alarm {
            // Store the just-observed wealth, then freeze the epoch. As in the paper's
            // algorithm, an alarm is checked before the next ONS projection/count update.
            self.order = FeatureState {
                lambda: self.order.lambda,
                log_wealth: order_observation.log_wealth,
                squared_gradient_sum: order_observation.squared_gradient_sum,
            };
            self.dispersion = FeatureState {
                lambda: self.dispersion.lambda,
                log_wealth: dispersion_observation.log_wealth,
                squared_gradient_sum: dispersion_observation.squared_gradient_sum,
            };
            self.observations_seen = audit_index;
            self.state = PredictiveRankState::Alarmed;
            return Ok(result);
        }

        let mut next_counts = self.rank_counts.clone();
        next_counts[rank_category] = next_counts[rank_category]
            .checked_add(1)
            .ok_or(PredictiveRankError::ObservationCounterOverflow)?;
        let next_order_stats =
            feature_stats(&next_counts, n, next_audit_index, RankFeature::Order)?;
        let next_dispersion_stats =
            feature_stats(&next_counts, n, next_audit_index, RankFeature::Dispersion)?;
        let next_order_lambda = next_lambda(
            self.order.lambda,
            order_observation,
            next_order_stats.bet_bound,
        )?;
        let next_dispersion_lambda = next_lambda(
            self.dispersion.lambda,
            dispersion_observation,
            next_dispersion_stats.bet_bound,
        )?;

        self.rank_counts = next_counts;
        self.order = FeatureState {
            lambda: next_order_lambda,
            log_wealth: order_observation.log_wealth,
            squared_gradient_sum: order_observation.squared_gradient_sum,
        };
        self.dispersion = FeatureState {
            lambda: next_dispersion_lambda,
            log_wealth: dispersion_observation.log_wealth,
            squared_gradient_sum: dispersion_observation.squared_gradient_sum,
        };
        self.observations_seen = audit_index;
        Ok(result)
    }

    /// Number of fixed calibration entries.
    pub fn calibration_count(&self) -> usize {
        self.sorted_reference.len()
    }

    /// Number of valid online observations consumed in this epoch.
    pub fn observations_seen(&self) -> usize {
        self.observations_seen
    }

    /// The configured one-sided significance level for the portfolio alarm threshold.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Natural-log alarm threshold, equal to -ln(alpha).
    pub fn log_alarm_threshold(&self) -> f64 {
        self.log_alarm_threshold
    }

    /// Current terminal state of this monitoring epoch.
    pub fn state(&self) -> PredictiveRankState {
        self.state
    }
}

fn valid_tie_breaker(value: f64) -> bool {
    value.is_finite() && (0.0..1.0).contains(&value)
}

fn compare_keys(left: &(f64, f64), right: &(f64, f64)) -> Ordering {
    let score_order = if left.0 < right.0 {
        Ordering::Less
    } else if left.0 > right.0 {
        Ordering::Greater
    } else {
        Ordering::Equal
    };
    score_order.then_with(|| {
        if left.1 < right.1 {
            Ordering::Less
        } else if left.1 > right.1 {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    })
}

fn feature_stats(
    rank_counts: &[usize],
    reference_count: usize,
    audit_index: usize,
    feature: RankFeature,
) -> Result<FeatureStats, PredictiveRankError> {
    let denominator = reference_count as f64 + audit_index as f64;
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(PredictiveRankError::NumericalFailure);
    }

    // Conditional predictive-rank probabilities: pi_j = (1 + N_{t-1,j}) / (n+t).
    let mut mean = 0.0;
    for (category, count) in rank_counts.iter().copied().enumerate() {
        let probability = (1.0 + count as f64) / denominator;
        mean += probability * feature.value(category, reference_count);
    }
    if !mean.is_finite() {
        return Err(PredictiveRankError::NumericalFailure);
    }

    let mut max_deviation = 0.0_f64;
    for category in 0..rank_counts.len() {
        let deviation = (feature.value(category, reference_count) - mean).abs();
        max_deviation = max_deviation.max(deviation);
    }
    let bet_bound = if max_deviation > 0.0 {
        (3.0 / (4.0 * max_deviation)).min(1.0)
    } else {
        1.0
    };
    if !bet_bound.is_finite() || bet_bound <= 0.0 {
        return Err(PredictiveRankError::NumericalFailure);
    }

    Ok(FeatureStats { mean, bet_bound })
}

fn feature_observation(
    state: FeatureState,
    z: f64,
    current_bet_bound: f64,
) -> Result<FeatureObservation, PredictiveRankError> {
    if !z.is_finite()
        || !current_bet_bound.is_finite()
        || current_bet_bound <= 0.0
        || state.lambda.abs() > current_bet_bound + 1e-12
    {
        return Err(PredictiveRankError::NumericalFailure);
    }

    let factor = 1.0 + state.lambda * z;
    // The paper's projection bound ensures 1 + lambda*z >= 1/4. Keep a strict positivity
    // check too, so corrupted states or floating-point defects cannot create negative wealth.
    if !factor.is_finite() || factor <= 0.0 {
        return Err(PredictiveRankError::NumericalFailure);
    }
    let log_wealth = state.log_wealth + factor.ln();
    let gradient = z / factor;
    let squared_gradient_sum = state.squared_gradient_sum + gradient * gradient;
    if !log_wealth.is_finite() || !gradient.is_finite() || !squared_gradient_sum.is_finite() {
        return Err(PredictiveRankError::NumericalFailure);
    }

    Ok(FeatureObservation {
        log_wealth,
        gradient,
        squared_gradient_sum,
    })
}

fn next_lambda(
    current_lambda: f64,
    observation: FeatureObservation,
    next_bet_bound: f64,
) -> Result<f64, PredictiveRankError> {
    if !current_lambda.is_finite()
        || !observation.gradient.is_finite()
        || !observation.squared_gradient_sum.is_finite()
        || !next_bet_bound.is_finite()
        || next_bet_bound <= 0.0
    {
        return Err(PredictiveRankError::NumericalFailure);
    }
    // Online Newton step from Kuang & Xia: project the update onto the next predictable bound.
    let candidate =
        current_lambda + 4.5 * observation.gradient / (1.0 + observation.squared_gradient_sum);
    if !candidate.is_finite() {
        return Err(PredictiveRankError::NumericalFailure);
    }
    Ok(candidate.clamp(-next_bet_bound, next_bet_bound))
}

fn log_mean_exp(left: f64, right: f64) -> Result<f64, PredictiveRankError> {
    if !left.is_finite() || !right.is_finite() {
        return Err(PredictiveRankError::NumericalFailure);
    }
    let maximum = left.max(right);
    let result = maximum + ((left - maximum).exp() + (right - maximum).exp()).ln() - 2.0_f64.ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(PredictiveRankError::NumericalFailure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monitor_with_scores(scores: &[f64], alpha: f64) -> PredictiveRankMartingale {
        let tie_breakers = vec![0.5; scores.len()];
        PredictiveRankMartingale::new(scores, &tie_breakers, alpha)
            .expect("fixture should be a valid fixed reference")
    }

    fn sequential_scores(count: usize) -> Vec<f64> {
        (0..count).map(|index| index as f64).collect()
    }

    #[test]
    fn rejects_invalid_alpha_reference_and_tie_breakers() {
        let score = [1.0, 2.0];
        let ties = [0.2, 0.3];
        assert!(matches!(
            PredictiveRankMartingale::new(&score, &ties, 0.0),
            Err(PredictiveRankError::InvalidSignificanceLevel)
        ));
        assert!(matches!(
            PredictiveRankMartingale::new(&score, &ties, f64::NAN),
            Err(PredictiveRankError::InvalidSignificanceLevel)
        ));
        assert!(matches!(
            PredictiveRankMartingale::new(&[], &[], 0.05),
            Err(PredictiveRankError::EmptyCalibrationReference)
        ));
        assert!(matches!(
            PredictiveRankMartingale::new(&score, &[0.2], 0.05),
            Err(PredictiveRankError::CalibrationTieBreakerLengthMismatch { .. })
        ));
        assert!(matches!(
            PredictiveRankMartingale::new(&[1.0, f64::INFINITY], &ties, 0.05),
            Err(PredictiveRankError::NonFiniteCalibrationScore { index: 1 })
        ));
        assert!(matches!(
            PredictiveRankMartingale::new(&score, &[0.2, 1.0], 0.05),
            Err(PredictiveRankError::InvalidCalibrationTieBreaker { index: 1 })
        ));
        assert!(matches!(
            PredictiveRankMartingale::new(&[1.0, 1.0], &[0.5, 0.5], 0.05),
            Err(PredictiveRankError::DuplicateCalibrationKey { .. })
        ));
    }

    #[test]
    fn randomized_tie_breakers_define_order_within_atomic_scores() {
        let calibration_scores = [1.0, 1.0, 1.0];
        let calibration_ties = [0.1, 0.5, 0.9];
        let mut monitor =
            PredictiveRankMartingale::new(&calibration_scores, &calibration_ties, 0.05).unwrap();

        let result = monitor.observe(1.0, 0.4).unwrap();
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn initial_predictive_rank_payoffs_are_centered_over_all_ranks() {
        let calibration = [0.0, 1.0, 2.0, 3.0, 4.0];
        let audits = [-1.0, 0.5, 1.5, 2.5, 3.5, 5.0];
        let mut order_sum = 0.0;
        let mut dispersion_sum = 0.0;
        for audit in audits {
            let mut monitor = monitor_with_scores(&calibration, 0.05);
            let result = monitor.observe(audit, 0.5).unwrap();
            order_sum += result.order_payoff_z;
            dispersion_sum += result.dispersion_payoff_z;
        }
        assert!(order_sum.abs() < 1e-12, "order sum={order_sum}");
        assert!(
            dispersion_sum.abs() < 1e-12,
            "dispersion sum={dispersion_sum}"
        );
    }

    #[test]
    fn persistent_upper_rank_shift_triggers_an_anytime_portfolio_alarm() {
        let mut monitor = monitor_with_scores(&sequential_scores(100), 0.05);
        let mut alarm_index = None;
        for _ in 0..100 {
            let result = monitor.observe(1000.0, 0.5).unwrap();
            assert_eq!(result.rank, 101);
            assert!(result.order_log_wealth.is_finite());
            assert!(result.dispersion_log_wealth.is_finite());
            assert!(result.portfolio_log_wealth.is_finite());
            if result.decision == PredictiveRankDecision::Alarm {
                alarm_index = Some(result.audit_index);
                break;
            }
        }
        assert!(alarm_index.is_some(), "upper-rank shift was not detected");
        assert!(alarm_index.unwrap() <= 100);
        assert_eq!(monitor.state(), PredictiveRankState::Alarmed);
        assert!(matches!(
            monitor.observe(1000.0, 0.5),
            Err(PredictiveRankError::AlreadyAlarmed)
        ));
    }

    #[test]
    fn dispersion_feature_detects_symmetric_extreme_ranks() {
        let mut monitor = monitor_with_scores(&sequential_scores(100), 0.05);
        let mut dispersion_dominated = false;
        let mut alarm_index = None;
        for index in 0..100 {
            let score = if index % 2 == 0 { -1.0 } else { 1000.0 };
            let result = monitor.observe(score, 0.5).unwrap();
            if result.dispersion_log_wealth > result.order_log_wealth + 1e-8 {
                dispersion_dominated = true;
            }
            if result.decision == PredictiveRankDecision::Alarm {
                alarm_index = Some(result.audit_index);
                break;
            }
        }
        assert!(
            dispersion_dominated,
            "symmetric extremes should contribute differently to the dispersion feature"
        );
        assert!(
            alarm_index.is_some(),
            "symmetric extreme-rank stream was not detected by the portfolio"
        );
    }

    fn next_unit_interval(state: &mut u64) -> f64 {
        // Deterministic test generator only; not suitable for production tie breaking.
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        let value = (*state).wrapping_mul(0x2545_F491_4F6C_DD1D);
        (value >> 11) as f64 / ((1_u64 << 53) as f64)
    }

    fn sample_predictive_null_rank(
        state: &mut u64,
        prior_rank_counts: &[usize],
        reference_count: usize,
        audit_index: usize,
    ) -> usize {
        let target = next_unit_interval(state)
            * (reference_count as f64 + audit_index as f64);
        let mut cumulative_mass = 0.0;
        for (category, count) in prior_rank_counts.iter().copied().enumerate() {
            cumulative_mass += 1.0 + count as f64;
            if target < cumulative_mass {
                return category;
            }
        }
        // Roundoff can only reach this fallback when target lies infinitesimally above the
        // final cumulative sum. The final category preserves the rank range.
        prior_rank_counts.len() - 1
    }

    #[test]
    fn predictive_null_rank_stream_smoke_test_respects_anytime_alarm_budget() {
        const STREAMS: usize = 1_000;
        const HORIZON: usize = 300;
        const REFERENCE_COUNT: usize = 20;

        let mut rng = 0xD1B5_4A32_D192_ED03_u64;
        let calibration_scores = sequential_scores(REFERENCE_COUNT);
        let calibration_ties = vec![0.5; REFERENCE_COUNT];
        let mut alarms = 0_usize;

        for _ in 0..STREAMS {
            let mut monitor =
                PredictiveRankMartingale::new(&calibration_scores, &calibration_ties, 0.05)
                    .unwrap();
            let mut rank_counts = vec![0_usize; REFERENCE_COUNT + 1];

            for audit_index in 1..=HORIZON {
                // Generate from pi[t,j] = (1 + N[t-1,j]) / (n+t), the exact null
                // predictive rank law on which the PRM martingale proof depends.
                let category = sample_predictive_null_rank(
                    &mut rng,
                    &rank_counts,
                    REFERENCE_COUNT,
                    audit_index,
                );
                // Half-integer scores realize every possible rank against the integer-valued
                // fixed reference and avoid calibration-key collisions in this rank-process test.
                let result = monitor.observe(category as f64 - 0.5, 0.5).unwrap();
                if result.decision == PredictiveRankDecision::Alarm {
                    alarms += 1;
                    break;
                }
                rank_counts[category] += 1;
            }
        }

        let empirical_alarm_rate = alarms as f64 / STREAMS as f64;
        // Fixed-seed smoke bound, deliberately looser than the nominal 0.05 guarantee. This
        // catches gross errors in rank probabilities, wealth updates, or portfolio thresholding;
        // it is not a proof or a confidence interval for anytime validity.
        assert!(
            empirical_alarm_rate <= 0.10,
            "null-rank smoke-test alarm rate was {empirical_alarm_rate:.4}"
        );
    }

    #[test]
    fn invalid_observations_do_not_consume_the_epoch() {
        let mut monitor = monitor_with_scores(&sequential_scores(20), 0.05);
        assert!(matches!(
            monitor.observe(f64::NAN, 0.5),
            Err(PredictiveRankError::InvalidAuditScore)
        ));
        assert!(matches!(
            monitor.observe(25.0, 1.0),
            Err(PredictiveRankError::InvalidAuditTieBreaker)
        ));
        assert!(matches!(
            monitor.observe(10.0, 0.5),
            Err(PredictiveRankError::AuditKeyCollidesWithReference { .. })
        ));
        assert_eq!(monitor.observations_seen(), 0);
        assert_eq!(monitor.state(), PredictiveRankState::Monitoring);
        let result = monitor.observe(25.0, 0.5).unwrap();
        assert_eq!(result.audit_index, 1);
    }
}
