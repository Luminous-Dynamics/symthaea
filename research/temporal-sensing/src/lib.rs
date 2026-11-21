//! Candidate-side causal observation scheduling controls.
//!
//! All decision inputs are candidate-observable features derived from the
//! received history. There are no evaluator labels, future samples, sealed
//! schedule seeds or hidden trajectories in this API. This package chooses
//! when to request an observation; it does not claim to reconstruct events
//! that were never observed.

use std::fmt;

const PPM: u128 = 1_000_000;

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SensingFeatures {
    /// Candidate-estimated chance/severity score for a near-term event.
    pub event_risk_ppm: u32,
    pub uncertainty_ppm: u32,
    pub novelty_ppm: u32,
    /// Residual-trigger score computed only from prior observations.
    pub past_residual_ppm: u32,
    /// Candidate-estimated wrong-belief exposure avoided by observing now.
    pub predicted_ted_reduction_us: u64,
    /// Candidate-estimated standard deviation of opportunity loss over current
    /// belief states, in microseconds. This is not evaluator-side hidden truth.
    pub opportunity_loss_stddev_us: u64,
    /// Candidate-estimated predictive-model confidence after posterior feedback,
    /// in PPM. Age/horizon decay should be incorporated by the candidate estimator.
    pub model_confidence_ppm: u32,
    /// Candidate-estimated share of the current ignorance exposure that this
    /// specific observation can resolve, in PPM. Zero means the sensor is
    /// uninformative about the uncertainty being penalized.
    pub predicted_ignorance_resolution_ppm: u32,
    /// Candidate-visible horizon over which model ignorance could expose future
    /// decisions to wrong-belief loss, in microseconds.
    pub prediction_horizon_us: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SensingContext {
    pub now_us: u64,
    pub last_observation_us: u64,
    pub minimum_gap_us: u64,
    pub maximum_gap_us: u64,
    pub estimated_observation_energy_nj: u64,
    pub remaining_observations: u64,
    pub remaining_energy_nj: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SensingPolicy {
    /// Fixed-rate observation with an upper bound on the inter-observation gap.
    FixedInterval { interval_us: u64 },
    /// Reproducible jittered schedule baseline. The seed is candidate-side,
    /// not an evaluator schedule key.
    JitteredInterval {
        interval_us: u64,
        jitter_per_mille: u16,
        seed: u64,
    },
    /// Choose between two rates from candidate-visible event/uncertainty signals.
    MultiRate {
        fast_interval_us: u64,
        slow_interval_us: u64,
        activity_threshold_ppm: u32,
    },
    EventThreshold { threshold_ppm: u32 },
    NoveltyThreshold { threshold_ppm: u32 },
    UncertaintyThreshold { threshold_ppm: u32 },
    PastResidualThreshold { threshold_ppm: u32 },
    /// Observe when expected avoided TED per nJ exceeds this threshold.
    /// The threshold is in microseconds/nJ, scaled by 1e6.
    ValueOfInformation { minimum_us_per_nj_ppm: u64 },
    /// Candidate-side risk-adjusted VOI ablation. It adds a risk premium for
    /// opportunity-loss variability across belief states and an ignorance
    /// surcharge for low predictive-model confidence over the current horizon.
    /// All coefficients are fixed-point PPM values in [0, 1_000_000].
    RiskAdjustedValueOfInformation {
        minimum_adjusted_us_per_nj_ppm: u64,
        risk_aversion_ppm: u32,
        ignorance_weight_ppm: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TriggerReason {
    FixedSchedule,
    FastActivity,
    SlowActivity,
    EventRisk,
    Novelty,
    Uncertainty,
    PastResidual,
    ValueOfInformation,
    RiskAdjustedValueOfInformation,
    MaximumGap,
    MinimumGap,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BudgetBlock {
    ObservationCountExhausted,
    EnergyInsufficient,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SensingDecision {
    ObserveNow {
        reason: TriggerReason,
        estimated_energy_nj: u64,
    },
    Wait {
        delay_us: u64,
        due_at_us: u64,
        reason: TriggerReason,
    },
    BudgetBlocked {
        reason: BudgetBlock,
        must_observe_by_us: u64,
        /// True when the maximum-gap deadline has already passed.
        deadline_missed: bool,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SensingError {
    ClockWentBackwards,
    InvalidGapBounds {
        minimum_gap_us: u64,
        maximum_gap_us: u64,
    },
    ZeroObservationEnergy,
    ZeroInterval,
    InvalidProbability {
        field: &'static str,
        value: u32,
    },
    InvalidJitter { jitter_per_mille: u16 },
    TimestampOverflow,
    ArithmeticOverflow,
}

impl fmt::Display for SensingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClockWentBackwards => write!(f, "current time precedes the last observation"),
            Self::InvalidGapBounds { minimum_gap_us, maximum_gap_us } => write!(
                f,
                "invalid observation gap bounds: min {minimum_gap_us}, max {maximum_gap_us}"
            ),
            Self::ZeroObservationEnergy => write!(f, "estimated observation energy must be nonzero"),
            Self::ZeroInterval => write!(f, "policy observation intervals must be nonzero"),
            Self::InvalidProbability { field, value } => {
                write!(f, "{field}={value} exceeds the PPM maximum 1000000")
            }
            Self::InvalidJitter { jitter_per_mille } => {
                write!(f, "jitter {jitter_per_mille} per mille must be below 1000")
            }
            Self::TimestampOverflow => write!(f, "next observation timestamp overflowed"),
            Self::ArithmeticOverflow => write!(f, "value-of-information arithmetic overflowed"),
        }
    }
}

impl std::error::Error for SensingError {}

/// Decide whether to observe now or when to observe next.
///
/// The decision is causal: it uses only the supplied current timestamp,
/// previous-observation timestamp and candidate-estimated features. The
/// maximum inter-observation gap is a hard schedule bound; it is not a
/// guarantee against aliasing or events shorter than the observation gap.
pub fn decide_next_observation(
    policy: SensingPolicy,
    context: SensingContext,
    features: SensingFeatures,
) -> Result<SensingDecision, SensingError> {
    if context.now_us < context.last_observation_us {
        return Err(SensingError::ClockWentBackwards);
    }
    if context.maximum_gap_us == 0
        || context.minimum_gap_us > context.maximum_gap_us
    {
        return Err(SensingError::InvalidGapBounds {
            minimum_gap_us: context.minimum_gap_us,
            maximum_gap_us: context.maximum_gap_us,
        });
    }
    if context.estimated_observation_energy_nj == 0 {
        return Err(SensingError::ZeroObservationEnergy);
    }
    validate_features(features)?;
    validate_policy(policy)?;

    let deadline_us = context
        .last_observation_us
        .checked_add(context.maximum_gap_us)
        .ok_or(SensingError::TimestampOverflow)?;
    let elapsed_us = context.now_us - context.last_observation_us;
    let maximum_gap_reached = elapsed_us >= context.maximum_gap_us;
    let remaining_to_maximum_us = context.maximum_gap_us.saturating_sub(elapsed_us);

    let (mut proposed_delay_us, mut reason) = if maximum_gap_reached {
        (0, TriggerReason::MaximumGap)
    } else {
        proposed_delay(policy, context, features, remaining_to_maximum_us)?
    };

    proposed_delay_us = proposed_delay_us.min(remaining_to_maximum_us);

    if proposed_delay_us == 0 && elapsed_us < context.minimum_gap_us {
        proposed_delay_us = context.minimum_gap_us - elapsed_us;
        reason = TriggerReason::MinimumGap;
    }

    let budget_block = if context.remaining_observations == 0 {
        Some(BudgetBlock::ObservationCountExhausted)
    } else if context.remaining_energy_nj < context.estimated_observation_energy_nj {
        Some(BudgetBlock::EnergyInsufficient)
    } else {
        None
    };

    if let Some(block_reason) = budget_block {
        return Ok(SensingDecision::BudgetBlocked {
            reason: block_reason,
            must_observe_by_us: deadline_us,
            deadline_missed: maximum_gap_reached,
        });
    }

    if proposed_delay_us == 0 {
        return Ok(SensingDecision::ObserveNow {
            reason,
            estimated_energy_nj: context.estimated_observation_energy_nj,
        });
    }

    let due_at_us = context
        .now_us
        .checked_add(proposed_delay_us)
        .ok_or(SensingError::TimestampOverflow)?;
    Ok(SensingDecision::Wait {
        delay_us: proposed_delay_us,
        due_at_us,
        reason,
    })
}

fn proposed_delay(
    policy: SensingPolicy,
    context: SensingContext,
    features: SensingFeatures,
    remaining_to_maximum_us: u64,
) -> Result<(u64, TriggerReason), SensingError> {
    let elapsed = context.now_us - context.last_observation_us;
    let (target_interval, reason) = match policy {
        SensingPolicy::FixedInterval { interval_us } => {
            (interval_us, TriggerReason::FixedSchedule)
        }
        SensingPolicy::JitteredInterval {
            interval_us,
            jitter_per_mille,
            seed,
        } => {
            let max_jitter = u128::from(interval_us)
                .checked_mul(u128::from(jitter_per_mille))
                .ok_or(SensingError::ArithmeticOverflow)?
                / 1_000;
            let width = max_jitter
                .checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .ok_or(SensingError::ArithmeticOverflow)?;
            let random = u128::from(mix64(seed ^ context.last_observation_us.rotate_left(19)));
            let offset = random % width;
            let jittered = u128::from(interval_us) + offset - max_jitter;
            let jittered_interval = u64::try_from(jittered)
                .map_err(|_| SensingError::ArithmeticOverflow)?;
            (jittered_interval, TriggerReason::FixedSchedule)
        }
        SensingPolicy::MultiRate {
            fast_interval_us,
            slow_interval_us,
            activity_threshold_ppm,
        } => {
            let high_activity = features.event_risk_ppm >= activity_threshold_ppm
                || features.uncertainty_ppm >= activity_threshold_ppm;
            if high_activity {
                (fast_interval_us, TriggerReason::FastActivity)
            } else {
                (slow_interval_us, TriggerReason::SlowActivity)
            }
        }
        SensingPolicy::EventThreshold { threshold_ppm } => {
            if features.event_risk_ppm >= threshold_ppm {
                (elapsed, TriggerReason::EventRisk)
            } else {
                (context.maximum_gap_us, TriggerReason::MaximumGap)
            }
        }
        SensingPolicy::NoveltyThreshold { threshold_ppm } => {
            if features.novelty_ppm >= threshold_ppm {
                (elapsed, TriggerReason::Novelty)
            } else {
                (context.maximum_gap_us, TriggerReason::MaximumGap)
            }
        }
        SensingPolicy::UncertaintyThreshold { threshold_ppm } => {
            if features.uncertainty_ppm >= threshold_ppm {
                (elapsed, TriggerReason::Uncertainty)
            } else {
                (context.maximum_gap_us, TriggerReason::MaximumGap)
            }
        }
        SensingPolicy::PastResidualThreshold { threshold_ppm } => {
            if features.past_residual_ppm >= threshold_ppm {
                (elapsed, TriggerReason::PastResidual)
            } else {
                (context.maximum_gap_us, TriggerReason::MaximumGap)
            }
        }
        SensingPolicy::ValueOfInformation { minimum_us_per_nj_ppm } => {
            let scaled_value = u128::from(features.predicted_ted_reduction_us)
                .checked_mul(PPM)
                .ok_or(SensingError::ArithmeticOverflow)?;
            let value_per_nj_ppm =
                scaled_value / u128::from(context.estimated_observation_energy_nj);
            if value_per_nj_ppm >= u128::from(minimum_us_per_nj_ppm) {
                (elapsed, TriggerReason::ValueOfInformation)
            } else {
                (context.maximum_gap_us, TriggerReason::MaximumGap)
            }
        }
        SensingPolicy::RiskAdjustedValueOfInformation {
            minimum_adjusted_us_per_nj_ppm,
            risk_aversion_ppm,
            ignorance_weight_ppm,
        } => {
            let diagnostics = risk_adjusted_voi_diagnostics(
                features,
                context.estimated_observation_energy_nj,
                risk_aversion_ppm,
                ignorance_weight_ppm,
            )?;
            if diagnostics.adjusted_value_per_nj_ppm
                >= u128::from(minimum_adjusted_us_per_nj_ppm)
            {
                (elapsed, TriggerReason::RiskAdjustedValueOfInformation)
            } else {
                (context.maximum_gap_us, TriggerReason::MaximumGap)
            }
        }
    };

    let delay = target_interval.saturating_sub(elapsed);
    Ok((delay.min(remaining_to_maximum_us), reason))
}

/// Auditable utility components for the candidate-side risk-adjusted VOI arm.
///
/// All inputs are candidate estimates from received history/posterior feedback.
/// This diagnostic is not an evaluator outcome and is not the IEEE paper's exact
/// value function. The fields are suitable for a work-ledger diagnostic record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RiskAdjustedVoIDiagnostics {
    pub expected_ted_reduction_us: u64,
    pub opportunity_loss_stddev_us: u64,
    pub model_confidence_ppm: u32,
    pub prediction_horizon_us: u64,
    pub risk_aversion_ppm: u32,
    pub risk_premium_us: u128,
    pub ignorance_exposure_us: u128,
    pub predicted_ignorance_resolution_ppm: u32,
    pub resolvable_ignorance_exposure_us: u128,
    pub ignorance_weight_ppm: u32,
    pub ignorance_penalty_us: u128,
    pub adjusted_benefit_us: u128,
    pub estimated_observation_energy_nj: u64,
    /// Fixed-point adjusted microseconds/nJ, scaled by 1,000,000.
    pub adjusted_value_per_nj_ppm: u128,
}

/// Compute candidate-visible risk/ignorance-adjusted VOI diagnostics.
///
/// The policy uses the same function as this diagnostic, so the recorded
/// components and scheduling threshold cannot silently drift apart.
pub fn risk_adjusted_voi_diagnostics(
    features: SensingFeatures,
    estimated_observation_energy_nj: u64,
    risk_aversion_ppm: u32,
    ignorance_weight_ppm: u32,
) -> Result<RiskAdjustedVoIDiagnostics, SensingError> {
    validate_features(features)?;
    if estimated_observation_energy_nj == 0 {
        return Err(SensingError::ZeroObservationEnergy);
    }
    for (field, value) in [
        ("risk_aversion_ppm", risk_aversion_ppm),
        ("ignorance_weight_ppm", ignorance_weight_ppm),
    ] {
        if u128::from(value) > PPM {
            return Err(SensingError::InvalidProbability { field, value });
        }
    }

    let risk_premium_us = u128::from(features.opportunity_loss_stddev_us)
        .checked_mul(u128::from(risk_aversion_ppm))
        .ok_or(SensingError::ArithmeticOverflow)?
        / PPM;
    let confidence_shortfall = PPM
        .checked_sub(u128::from(features.model_confidence_ppm))
        .ok_or(SensingError::ArithmeticOverflow)?;
    let ignorance_exposure_us = u128::from(features.prediction_horizon_us)
        .checked_mul(confidence_shortfall)
        .ok_or(SensingError::ArithmeticOverflow)?
        / PPM;
    let resolvable_ignorance_exposure_us = ignorance_exposure_us
        .checked_mul(u128::from(features.predicted_ignorance_resolution_ppm))
        .ok_or(SensingError::ArithmeticOverflow)?
        / PPM;
    let ignorance_penalty_us = resolvable_ignorance_exposure_us
        .checked_mul(u128::from(ignorance_weight_ppm))
        .ok_or(SensingError::ArithmeticOverflow)?
        / PPM;
    let adjusted_benefit_us = u128::from(features.predicted_ted_reduction_us)
        .checked_add(risk_premium_us)
        .and_then(|value| value.checked_add(ignorance_penalty_us))
        .ok_or(SensingError::ArithmeticOverflow)?;
    let adjusted_value_per_nj_ppm = adjusted_benefit_us
        .checked_mul(PPM)
        .ok_or(SensingError::ArithmeticOverflow)?
        / u128::from(estimated_observation_energy_nj);

    Ok(RiskAdjustedVoIDiagnostics {
        expected_ted_reduction_us: features.predicted_ted_reduction_us,
        opportunity_loss_stddev_us: features.opportunity_loss_stddev_us,
        model_confidence_ppm: features.model_confidence_ppm,
        prediction_horizon_us: features.prediction_horizon_us,
        risk_aversion_ppm,
        risk_premium_us,
        ignorance_exposure_us,
        predicted_ignorance_resolution_ppm: features.predicted_ignorance_resolution_ppm,
        resolvable_ignorance_exposure_us,
        ignorance_weight_ppm,
        ignorance_penalty_us,
        adjusted_benefit_us,
        estimated_observation_energy_nj,
        adjusted_value_per_nj_ppm,
    })
}

fn validate_features(features: SensingFeatures) -> Result<(), SensingError> {
    for (field, value) in [
        ("event_risk_ppm", features.event_risk_ppm),
        ("uncertainty_ppm", features.uncertainty_ppm),
        ("novelty_ppm", features.novelty_ppm),
        ("past_residual_ppm", features.past_residual_ppm),
        ("model_confidence_ppm", features.model_confidence_ppm),
        ("predicted_ignorance_resolution_ppm", features.predicted_ignorance_resolution_ppm),
    ] {
        if u128::from(value) > PPM {
            return Err(SensingError::InvalidProbability { field, value });
        }
    }
    Ok(())
}

fn validate_policy(policy: SensingPolicy) -> Result<(), SensingError> {
    let valid_interval = |value: u64| value > 0;
    let valid_threshold = |value: u32| u128::from(value) <= PPM;
    match policy {
        SensingPolicy::FixedInterval { interval_us } if !valid_interval(interval_us) => {
            Err(SensingError::ZeroInterval)
        }
        SensingPolicy::JitteredInterval { interval_us, jitter_per_mille, .. }
            if !valid_interval(interval_us) || jitter_per_mille >= 1_000 =>
        {
            if interval_us == 0 {
                Err(SensingError::ZeroInterval)
            } else {
                Err(SensingError::InvalidJitter { jitter_per_mille })
            }
        }
        SensingPolicy::MultiRate {
            fast_interval_us,
            slow_interval_us,
            activity_threshold_ppm,
        } if !valid_interval(fast_interval_us)
            || !valid_interval(slow_interval_us)
            || !valid_threshold(activity_threshold_ppm) =>
        {
            if fast_interval_us == 0 || slow_interval_us == 0 {
                Err(SensingError::ZeroInterval)
            } else {
                Err(SensingError::InvalidProbability {
                    field: "activity_threshold_ppm",
                    value: activity_threshold_ppm,
                })
            }
        }
        SensingPolicy::EventThreshold { threshold_ppm } if !valid_threshold(threshold_ppm) => {
            Err(SensingError::InvalidProbability { field: "threshold_ppm", value: threshold_ppm })
        }
        SensingPolicy::NoveltyThreshold { threshold_ppm } if !valid_threshold(threshold_ppm) => {
            Err(SensingError::InvalidProbability { field: "threshold_ppm", value: threshold_ppm })
        }
        SensingPolicy::UncertaintyThreshold { threshold_ppm } if !valid_threshold(threshold_ppm) => {
            Err(SensingError::InvalidProbability { field: "threshold_ppm", value: threshold_ppm })
        }
        SensingPolicy::PastResidualThreshold { threshold_ppm } if !valid_threshold(threshold_ppm) => {
            Err(SensingError::InvalidProbability { field: "threshold_ppm", value: threshold_ppm })
        }
        SensingPolicy::RiskAdjustedValueOfInformation {
            risk_aversion_ppm,
            ignorance_weight_ppm,
            ..
        } if !valid_threshold(risk_aversion_ppm) || !valid_threshold(ignorance_weight_ppm) => {
            if !valid_threshold(risk_aversion_ppm) {
                Err(SensingError::InvalidProbability {
                    field: "risk_aversion_ppm",
                    value: risk_aversion_ppm,
                })
            } else {
                Err(SensingError::InvalidProbability {
                    field: "ignorance_weight_ppm",
                    value: ignorance_weight_ppm,
                })
            }
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(now_us: u64) -> SensingContext {
        SensingContext {
            now_us,
            last_observation_us: 0,
            minimum_gap_us: 10,
            maximum_gap_us: 100,
            estimated_observation_energy_nj: 50,
            remaining_observations: 20,
            remaining_energy_nj: 10_000,
        }
    }

    fn features() -> SensingFeatures {
        SensingFeatures {
            event_risk_ppm: 0,
            uncertainty_ppm: 100_000,
            novelty_ppm: 100_000,
            past_residual_ppm: 0,
            predicted_ted_reduction_us: 100,
            opportunity_loss_stddev_us: 0,
            model_confidence_ppm: 1_000_000,
            predicted_ignorance_resolution_ppm: 1_000_000,
            prediction_horizon_us: 0,
        }
    }

    #[test]
    fn fixed_interval_waits_until_next_period() {
        let decision = decide_next_observation(
            SensingPolicy::FixedInterval { interval_us: 20 },
            context(5),
            features(),
        ).unwrap();
        assert_eq!(decision, SensingDecision::Wait {
            delay_us: 15,
            due_at_us: 20,
            reason: TriggerReason::FixedSchedule,
        });
    }

    #[test]
    fn fixed_interval_observes_when_period_has_elapsed() {
        let decision = decide_next_observation(
            SensingPolicy::FixedInterval { interval_us: 20 },
            context(20),
            features(),
        ).unwrap();
        assert_eq!(decision, SensingDecision::ObserveNow {
            reason: TriggerReason::FixedSchedule,
            estimated_energy_nj: 50,
        });
    }

    #[test]
    fn jittered_schedule_is_deterministic_and_within_declared_band() {
        let policy = SensingPolicy::JitteredInterval {
            interval_us: 100,
            jitter_per_mille: 200,
            seed: 0xA11C_E001,
        };
        let first = decide_next_observation(policy, context(0), features()).unwrap();
        let repeated = decide_next_observation(policy, context(0), features()).unwrap();
        assert_eq!(first, repeated);
        match first {
            SensingDecision::Wait { delay_us, .. } => {
                assert!((80..=120).contains(&delay_us));
            }
            other => panic!("expected a wait decision, got {other:?}"),
        }
    }

    #[test]
    fn event_trigger_observes_now_when_candidate_risk_crosses_threshold() {
        let mut current = features();
        current.event_risk_ppm = 900_000;
        let decision = decide_next_observation(
            SensingPolicy::EventThreshold { threshold_ppm: 800_000 },
            context(20),
            current,
        ).unwrap();
        assert!(matches!(
            decision,
            SensingDecision::ObserveNow { reason: TriggerReason::EventRisk, .. }
        ));
    }

    #[test]
    fn multirate_policy_selects_fast_interval_for_high_activity() {
        let mut current = features();
        current.uncertainty_ppm = 900_000;
        let decision = decide_next_observation(
            SensingPolicy::MultiRate {
                fast_interval_us: 20,
                slow_interval_us: 80,
                activity_threshold_ppm: 700_000,
            },
            context(5),
            current,
        ).unwrap();
        assert_eq!(decision, SensingDecision::Wait {
            delay_us: 15,
            due_at_us: 20,
            reason: TriggerReason::FastActivity,
        });
    }

    #[test]
    fn maximum_gap_forces_observation_even_without_trigger() {
        let decision = decide_next_observation(
            SensingPolicy::EventThreshold { threshold_ppm: 900_000 },
            context(100),
            features(),
        ).unwrap();
        assert!(matches!(
            decision,
            SensingDecision::ObserveNow { reason: TriggerReason::MaximumGap, .. }
        ));
    }

    #[test]
    fn minimum_gap_defers_an_early_trigger() {
        let mut current = features();
        current.event_risk_ppm = 1_000_000;
        let mut ctx = context(5);
        ctx.minimum_gap_us = 20;
        let decision = decide_next_observation(
            SensingPolicy::EventThreshold { threshold_ppm: 900_000 },
            ctx,
            current,
        ).unwrap();
        assert_eq!(decision, SensingDecision::Wait {
            delay_us: 15,
            due_at_us: 20,
            reason: TriggerReason::MinimumGap,
        });
    }

    #[test]
    fn voi_policy_uses_candidate_value_and_observation_energy() {
        let decision = decide_next_observation(
            SensingPolicy::ValueOfInformation { minimum_us_per_nj_ppm: 1_000_000 },
            context(20),
            features(),
        ).unwrap();
        assert!(matches!(
            decision,
            SensingDecision::ObserveNow { reason: TriggerReason::ValueOfInformation, .. }
        ));
    }

    #[test]
    fn risk_adjusted_voi_triggers_on_belief_state_loss_variance() {
        let mut current = features();
        current.predicted_ted_reduction_us = 100;
        current.opportunity_loss_stddev_us = 1_000;
        current.model_confidence_ppm = 1_000_000;
        current.prediction_horizon_us = 10_000;

        // Plain VOI is 100/50 = 2 us/nJ and would wait at this threshold.
        assert!(matches!(
            decide_next_observation(
                SensingPolicy::ValueOfInformation {
                    minimum_us_per_nj_ppm: 10_000_000,
                },
                context(20),
                current,
            )
            .unwrap(),
            SensingDecision::Wait { .. }
        ));

        // The full-PPM risk premium adds the 1,000-us candidate-estimated
        // opportunity-loss standard deviation, making adjusted benefit 1,100 us.
        assert!(matches!(
            decide_next_observation(
                SensingPolicy::RiskAdjustedValueOfInformation {
                    minimum_adjusted_us_per_nj_ppm: 10_000_000,
                    risk_aversion_ppm: 1_000_000,
                    ignorance_weight_ppm: 0,
                },
                context(20),
                current,
            )
            .unwrap(),
            SensingDecision::ObserveNow {
                reason: TriggerReason::RiskAdjustedValueOfInformation,
                ..
            }
        ));
    }

    #[test]
    fn risk_adjusted_voi_adds_model_ignorance_exposure_as_confidence_decays() {
        let mut current = features();
        current.predicted_ted_reduction_us = 100;
        current.opportunity_loss_stddev_us = 0;
        current.model_confidence_ppm = 0;
        current.prediction_horizon_us = 10_000;

        // At zero current model confidence, the candidate estimates up to
        // 10,000 us of ignorance exposure over this horizon.
        assert!(matches!(
            decide_next_observation(
                SensingPolicy::RiskAdjustedValueOfInformation {
                    minimum_adjusted_us_per_nj_ppm: 50_000_000,
                    risk_aversion_ppm: 0,
                    ignorance_weight_ppm: 1_000_000,
                },
                context(20),
                current,
            )
            .unwrap(),
            SensingDecision::ObserveNow {
                reason: TriggerReason::RiskAdjustedValueOfInformation,
                ..
            }
        ));
    }

    #[test]
    fn risk_adjusted_voi_diagnostics_expose_each_fixed_point_component() {
        let mut current = features();
        current.predicted_ted_reduction_us = 100;
        current.opportunity_loss_stddev_us = 2_000;
        current.model_confidence_ppm = 500_000;
        current.prediction_horizon_us = 10_000;

        let diagnostics =
            risk_adjusted_voi_diagnostics(current, 50, 250_000, 200_000).unwrap();
        assert_eq!(diagnostics.expected_ted_reduction_us, 100);
        assert_eq!(diagnostics.risk_premium_us, 500);
        assert_eq!(diagnostics.ignorance_exposure_us, 5_000);
        assert_eq!(diagnostics.ignorance_penalty_us, 1_000);
        assert_eq!(diagnostics.adjusted_benefit_us, 1_600);
        assert_eq!(diagnostics.estimated_observation_energy_nj, 50);
        assert_eq!(diagnostics.adjusted_value_per_nj_ppm, 32_000_000);

        assert_eq!(
            risk_adjusted_voi_diagnostics(current, 0, 0, 0),
            Err(SensingError::ZeroObservationEnergy)
        );
    }

    #[test]
    fn low_confidence_does_not_reward_an_uninformative_observation() {
        let mut current = features();
        current.predicted_ted_reduction_us = 0;
        current.opportunity_loss_stddev_us = 0;
        current.model_confidence_ppm = 0;
        current.predicted_ignorance_resolution_ppm = 0;
        current.prediction_horizon_us = 10_000;

        let diagnostics =
            risk_adjusted_voi_diagnostics(current, 50, 0, 1_000_000).unwrap();
        assert_eq!(diagnostics.ignorance_exposure_us, 10_000);
        assert_eq!(diagnostics.resolvable_ignorance_exposure_us, 0);
        assert_eq!(diagnostics.ignorance_penalty_us, 0);
        assert_eq!(diagnostics.adjusted_benefit_us, 0);

        assert!(matches!(
            decide_next_observation(
                SensingPolicy::RiskAdjustedValueOfInformation {
                    minimum_adjusted_us_per_nj_ppm: 1,
                    risk_aversion_ppm: 0,
                    ignorance_weight_ppm: 1_000_000,
                },
                context(20),
                current,
            )
            .unwrap(),
            SensingDecision::Wait {
                reason: TriggerReason::MaximumGap,
                ..
            }
        ));
    }

    #[test]
    fn ignorance_surcharge_scales_with_candidate_estimated_resolvability() {
        let mut current = features();
        current.predicted_ted_reduction_us = 0;
        current.opportunity_loss_stddev_us = 0;
        current.model_confidence_ppm = 0;
        current.predicted_ignorance_resolution_ppm = 500_000;
        current.prediction_horizon_us = 10_000;

        let diagnostics =
            risk_adjusted_voi_diagnostics(current, 50, 0, 1_000_000).unwrap();
        assert_eq!(diagnostics.ignorance_exposure_us, 10_000);
        assert_eq!(diagnostics.predicted_ignorance_resolution_ppm, 500_000);
        assert_eq!(diagnostics.resolvable_ignorance_exposure_us, 5_000);
        assert_eq!(diagnostics.ignorance_penalty_us, 5_000);
    }

    #[test]
    fn predicted_ignorance_resolvability_must_be_bounded() {
        let mut current = features();
        current.predicted_ignorance_resolution_ppm = 1_000_001;
        assert_eq!(
            risk_adjusted_voi_diagnostics(current, 50, 0, 0),
            Err(SensingError::InvalidProbability {
                field: "predicted_ignorance_resolution_ppm",
                value: 1_000_001,
            })
        );
    }

    #[test]
    fn fully_confident_model_has_no_ignorance_surcharge() {
        let mut current = features();
        current.predicted_ted_reduction_us = 100;
        current.opportunity_loss_stddev_us = 0;
        current.model_confidence_ppm = 1_000_000;
        current.prediction_horizon_us = 10_000;

        assert!(matches!(
            decide_next_observation(
                SensingPolicy::RiskAdjustedValueOfInformation {
                    minimum_adjusted_us_per_nj_ppm: 10_000_000,
                    risk_aversion_ppm: 0,
                    ignorance_weight_ppm: 1_000_000,
                },
                context(20),
                current,
            )
            .unwrap(),
            SensingDecision::Wait {
                reason: TriggerReason::MaximumGap,
                ..
            }
        ));
    }

    #[test]
    fn risk_adjusted_voi_rejects_unbounded_coefficients_and_confidence() {
        let invalid_policy = SensingPolicy::RiskAdjustedValueOfInformation {
            minimum_adjusted_us_per_nj_ppm: 1,
            risk_aversion_ppm: 1_000_001,
            ignorance_weight_ppm: 0,
        };
        assert_eq!(
            decide_next_observation(invalid_policy, context(20), features()),
            Err(SensingError::InvalidProbability {
                field: "risk_aversion_ppm",
                value: 1_000_001,
            })
        );

        assert_eq!(
            decide_next_observation(
                SensingPolicy::RiskAdjustedValueOfInformation {
                    minimum_adjusted_us_per_nj_ppm: 1,
                    risk_aversion_ppm: 0,
                    ignorance_weight_ppm: 1_000_001,
                },
                context(20),
                features(),
            ),
            Err(SensingError::InvalidProbability {
                field: "ignorance_weight_ppm",
                value: 1_000_001,
            })
        );

        let mut invalid_features = features();
        invalid_features.model_confidence_ppm = 1_000_001;
        assert_eq!(
            decide_next_observation(
                SensingPolicy::RiskAdjustedValueOfInformation {
                    minimum_adjusted_us_per_nj_ppm: 1,
                    risk_aversion_ppm: 0,
                    ignorance_weight_ppm: 0,
                },
                context(20),
                invalid_features,
            ),
            Err(SensingError::InvalidProbability {
                field: "model_confidence_ppm",
                value: 1_000_001,
            })
        );
    }

    #[test]
    fn observation_and_energy_budgets_fail_closed() {
        let mut no_observations = context(100);
        no_observations.remaining_observations = 0;
        assert!(matches!(
            decide_next_observation(
                SensingPolicy::FixedInterval { interval_us: 30 },
                no_observations,
                features(),
            ).unwrap(),
            SensingDecision::BudgetBlocked {
                reason: BudgetBlock::ObservationCountExhausted,
                must_observe_by_us: 100,
                deadline_missed: true,
            }
        ));

        let mut no_energy = context(100);
        no_energy.remaining_energy_nj = 49;
        assert!(matches!(
            decide_next_observation(
                SensingPolicy::FixedInterval { interval_us: 30 },
                no_energy,
                features(),
            ).unwrap(),
            SensingDecision::BudgetBlocked {
                reason: BudgetBlock::EnergyInsufficient,
                must_observe_by_us: 100,
                deadline_missed: true,
            }
        ));
    }

    #[test]
    fn novelty_and_past_residual_thresholds_use_only_candidate_history() {
        let mut current = features();
        current.novelty_ppm = 950_000;
        let novelty = decide_next_observation(
            SensingPolicy::NoveltyThreshold { threshold_ppm: 900_000 },
            context(20),
            current,
        ).unwrap();
        assert!(matches!(novelty,
            SensingDecision::ObserveNow { reason: TriggerReason::Novelty, .. }
        ));

        current.novelty_ppm = 0;
        current.past_residual_ppm = 950_000;
        let residual = decide_next_observation(
            SensingPolicy::PastResidualThreshold { threshold_ppm: 900_000 },
            context(20),
            current,
        ).unwrap();
        assert!(matches!(residual,
            SensingDecision::ObserveNow { reason: TriggerReason::PastResidual, .. }
        ));
    }

    #[test]
    fn multirate_policy_uses_slow_interval_below_activity_threshold() {
        let decision = decide_next_observation(
            SensingPolicy::MultiRate {
                fast_interval_us: 20,
                slow_interval_us: 80,
                activity_threshold_ppm: 700_000,
            },
            context(5),
            features(),
        ).unwrap();
        assert_eq!(decision, SensingDecision::Wait {
            delay_us: 75,
            due_at_us: 80,
            reason: TriggerReason::SlowActivity,
        });
    }

    #[test]
    fn invalid_zero_intervals_and_timestamp_overflow_are_rejected() {
        assert_eq!(
            decide_next_observation(
                SensingPolicy::FixedInterval { interval_us: 0 },
                context(5),
                features(),
            ),
            Err(SensingError::ZeroInterval)
        );

        assert_eq!(
            decide_next_observation(
                SensingPolicy::JitteredInterval {
                    interval_us: 100,
                    jitter_per_mille: 1_000,
                    seed: 7,
                },
                context(5),
                features(),
            ),
            Err(SensingError::InvalidJitter { jitter_per_mille: 1_000 })
        );

        let mut overflow = context(u64::MAX - 5);
        overflow.last_observation_us = u64::MAX - 10;
        overflow.maximum_gap_us = 20;
        overflow.minimum_gap_us = 1;
        assert_eq!(
            decide_next_observation(
                SensingPolicy::FixedInterval { interval_us: 1 },
                overflow,
                features(),
            ),
            Err(SensingError::TimestampOverflow)
        );
    }

    #[test]
    fn irregular_schedule_clamps_policy_wait_to_maximum_gap() {
        let decision = decide_next_observation(
            SensingPolicy::FixedInterval { interval_us: 1_000 },
            context(40),
            features(),
        ).unwrap();
        assert_eq!(decision, SensingDecision::Wait {
            delay_us: 60,
            due_at_us: 100,
            reason: TriggerReason::FixedSchedule,
        });
    }

    #[test]
    fn invalid_clock_gap_energy_and_features_fail_closed() {
        let mut clock = context(5);
        clock.last_observation_us = 6;
        assert_eq!(
            decide_next_observation(SensingPolicy::FixedInterval { interval_us: 10 }, clock, features()),
            Err(SensingError::ClockWentBackwards)
        );

        let mut gaps = context(5);
        gaps.minimum_gap_us = 200;
        assert!(matches!(
            decide_next_observation(SensingPolicy::FixedInterval { interval_us: 10 }, gaps, features()),
            Err(SensingError::InvalidGapBounds { .. })
        ));

        let mut invalid_features = features();
        invalid_features.novelty_ppm = 1_000_001;
        assert!(matches!(
            decide_next_observation(
                SensingPolicy::FixedInterval { interval_us: 10 },
                context(5),
                invalid_features
            ),
            Err(SensingError::InvalidProbability { .. })
        ));
    }
}
