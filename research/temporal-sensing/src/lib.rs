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
    /// Observe when predicted avoided TED per nJ exceeds this fixed-point
    /// threshold (microseconds/nJ, scaled by 1e6).
    ValueOfInformation { minimum_us_per_nj_ppm: u64 },
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
    };

    let delay = target_interval.saturating_sub(elapsed);
    Ok((delay.min(remaining_to_maximum_us), reason))
}

fn validate_features(features: SensingFeatures) -> Result<(), SensingError> {
    for (field, value) in [
        ("event_risk_ppm", features.event_risk_ppm),
        ("uncertainty_ppm", features.uncertainty_ppm),
        ("novelty_ppm", features.novelty_ppm),
        ("past_residual_ppm", features.past_residual_ppm),
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
