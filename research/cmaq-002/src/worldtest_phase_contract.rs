//! MARA-specific WorldTest exploration contract.
//!
//! Every candidate-issued action and reset carries the exact IEEE-754 reward
//! bits returned by the pinned MARA environment. The reference exploration
//! phase transmits this field but returns constant zero. Nonzero and
//! non-finite values fail closed before the generic trace diagnostic runs.

use super::worldtest_exploration::{
    evaluate_worldtest_exploration, ExplorationEvent, ExplorationTrace,
    WorldTestExplorationCertificate, WorldTestExplorationError,
};
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaraExplorationEvent {
    Action {
        action_id: String,
        reward_bits: u64,
    },
    Reset {
        reward_bits: u64,
    },
    BeginTest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaraExplorationTrace {
    pub events: Vec<MaraExplorationEvent>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorldTestPhaseContractError {
    MissingBeginTestMarker,
    MissingGoToTestAction,
    GoToTestNotImmediatelyBeforeMarker,
    DuplicateGoToTestAction,
    NonFiniteExplorationReward {
        event_index: usize,
        reward_bits: u64,
    },
    NonZeroExplorationReward {
        event_index: usize,
        reward_bits: u64,
    },
    Trace(WorldTestExplorationError),
}

impl fmt::Display for WorldTestPhaseContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBeginTestMarker => write!(formatter, "WorldTest trace is missing the terminal BeginTest marker"),
            Self::MissingGoToTestAction => write!(formatter, "WorldTest exploration must include the candidate-issued go-to-test action"),
            Self::GoToTestNotImmediatelyBeforeMarker => write!(formatter, "go-to-test must be the final charged exploration action immediately before BeginTest"),
            Self::DuplicateGoToTestAction => write!(formatter, "WorldTest trace contains more than one go-to-test action"),
            Self::NonFiniteExplorationReward { event_index, reward_bits } => write!(
                formatter,
                "exploration event {event_index} returned a non-finite reward (IEEE-754 bits {reward_bits:#018x})"
            ),
            Self::NonZeroExplorationReward { event_index, reward_bits } => write!(
                formatter,
                "exploration event {event_index} returned a nonzero reward (IEEE-754 bits {reward_bits:#018x})"
            ),
            Self::Trace(error) => write!(formatter, "invalid WorldTest exploration trace: {error}"),
        }
    }
}

impl std::error::Error for WorldTestPhaseContractError {}

fn validate_zero_reward(
    event_index: usize,
    reward_bits: u64,
) -> Result<(), WorldTestPhaseContractError> {
    let reward = f64::from_bits(reward_bits);
    if !reward.is_finite() {
        return Err(WorldTestPhaseContractError::NonFiniteExplorationReward {
            event_index,
            reward_bits,
        });
    }
    if reward != 0.0 {
        return Err(WorldTestPhaseContractError::NonZeroExplorationReward {
            event_index,
            reward_bits,
        });
    }
    Ok(())
}

/// Qualify a MARA exploration trace. Each Action and Reset event includes the
/// exact reward bits observed at that environment step. Exactly one go-to-test
/// action must be immediately followed by the trusted BeginTest marker. The
/// go-to-test action is charged by the underlying generic evaluator.
pub fn evaluate_mara_worldtest_exploration(
    trace: &MaraExplorationTrace,
) -> Result<WorldTestExplorationCertificate, WorldTestPhaseContractError> {
    let Some(last) = trace.events.last() else {
        return Err(WorldTestPhaseContractError::MissingBeginTestMarker);
    };
    if !matches!(last, MaraExplorationEvent::BeginTest) {
        return Err(WorldTestPhaseContractError::MissingBeginTestMarker);
    }

    let mut generic_events = Vec::with_capacity(trace.events.len());
    let mut go_to_test_positions = Vec::new();

    for (index, event) in trace.events.iter().enumerate() {
        match event {
            MaraExplorationEvent::Action {
                action_id,
                reward_bits,
            } => {
                validate_zero_reward(index, *reward_bits)?;
                if action_id == "go-to-test" {
                    go_to_test_positions.push(index);
                }
                generic_events.push(ExplorationEvent::Action(action_id.clone()));
            }
            MaraExplorationEvent::Reset { reward_bits } => {
                validate_zero_reward(index, *reward_bits)?;
                generic_events.push(ExplorationEvent::Reset);
            }
            MaraExplorationEvent::BeginTest => {
                generic_events.push(ExplorationEvent::BeginTest);
            }
        }
    }

    if go_to_test_positions.is_empty() {
        return Err(WorldTestPhaseContractError::MissingGoToTestAction);
    }
    if go_to_test_positions.len() != 1 {
        return Err(WorldTestPhaseContractError::DuplicateGoToTestAction);
    }
    if go_to_test_positions[0] + 2 != trace.events.len() {
        return Err(WorldTestPhaseContractError::GoToTestNotImmediatelyBeforeMarker);
    }

    let generic_trace = ExplorationTrace {
        events: generic_events,
    };
    evaluate_worldtest_exploration(&generic_trace)
        .map_err(WorldTestPhaseContractError::Trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(value: &str) -> MaraExplorationEvent {
        MaraExplorationEvent::Action {
            action_id: value.to_owned(),
            reward_bits: 0.0_f64.to_bits(),
        }
    }

    fn reset() -> MaraExplorationEvent {
        MaraExplorationEvent::Reset {
            reward_bits: 0.0_f64.to_bits(),
        }
    }

    fn valid_trace() -> MaraExplorationTrace {
        MaraExplorationTrace {
            events: vec![
                action("left"),
                reset(),
                action("right"),
                action("go-to-test"),
                MaraExplorationEvent::BeginTest,
            ],
        }
    }

    #[test]
    fn go_to_test_is_required_and_charged_as_an_interaction() {
        let certificate = evaluate_mara_worldtest_exploration(&valid_trace()).unwrap();
        assert_eq!(certificate.exploration_action_count, 3);
        assert_eq!(certificate.reset_count, 1);
        assert_eq!(certificate.reset_share_ppm, 250_000);
    }

    #[test]
    fn missing_go_to_test_action_is_rejected() {
        let trace = MaraExplorationTrace {
            events: vec![action("left"), MaraExplorationEvent::BeginTest],
        };
        assert_eq!(
            evaluate_mara_worldtest_exploration(&trace),
            Err(WorldTestPhaseContractError::MissingGoToTestAction)
        );
    }

    #[test]
    fn go_to_test_must_be_the_final_action_before_marker() {
        let trace = MaraExplorationTrace {
            events: vec![
                action("go-to-test"),
                action("left"),
                MaraExplorationEvent::BeginTest,
            ],
        };
        assert_eq!(
            evaluate_mara_worldtest_exploration(&trace),
            Err(WorldTestPhaseContractError::GoToTestNotImmediatelyBeforeMarker)
        );
    }

    #[test]
    fn duplicate_go_to_test_actions_are_rejected() {
        let trace = MaraExplorationTrace {
            events: vec![
                action("go-to-test"),
                action("go-to-test"),
                MaraExplorationEvent::BeginTest,
            ],
        };
        assert_eq!(
            evaluate_mara_worldtest_exploration(&trace),
            Err(WorldTestPhaseContractError::DuplicateGoToTestAction)
        );
    }

    #[test]
    fn missing_terminal_marker_is_rejected() {
        let trace = MaraExplorationTrace {
            events: vec![action("go-to-test")],
        };
        assert_eq!(
            evaluate_mara_worldtest_exploration(&trace),
            Err(WorldTestPhaseContractError::MissingBeginTestMarker)
        );
    }

    #[test]
    fn rejects_nonzero_reward_on_ordinary_action() {
        let mut trace = valid_trace();
        if let MaraExplorationEvent::Action { reward_bits, .. } = &mut trace.events[0] {
            *reward_bits = 0.25_f64.to_bits();
        }
        assert_eq!(
            evaluate_mara_worldtest_exploration(&trace),
            Err(WorldTestPhaseContractError::NonZeroExplorationReward {
                event_index: 0,
                reward_bits: 0.25_f64.to_bits(),
            })
        );
    }

    #[test]
    fn rejects_nonzero_reward_on_reset() {
        let mut trace = valid_trace();
        if let MaraExplorationEvent::Reset { reward_bits } = &mut trace.events[1] {
            *reward_bits = 1.0_f64.to_bits();
        }
        assert_eq!(
            evaluate_mara_worldtest_exploration(&trace),
            Err(WorldTestPhaseContractError::NonZeroExplorationReward {
                event_index: 1,
                reward_bits: 1.0_f64.to_bits(),
            })
        );
    }

    #[test]
    fn rejects_nan_and_infinite_rewards() {
        for (index, reward_bits) in [
            (0_usize, f64::NAN.to_bits()),
            (2_usize, f64::INFINITY.to_bits()),
        ] {
            let mut trace = valid_trace();
            if let MaraExplorationEvent::Action {
                reward_bits: actual,
                ..
            } = &mut trace.events[index]
            {
                *actual = reward_bits;
            }
            assert!(matches!(
                evaluate_mara_worldtest_exploration(&trace),
                Err(WorldTestPhaseContractError::NonFiniteExplorationReward {
                    event_index,
                    ..
                }) if event_index == index
            ));
        }
    }

    #[test]
    fn negative_zero_is_numerically_zero_and_is_accepted() {
        let mut trace = valid_trace();
        if let MaraExplorationEvent::Action { reward_bits, .. } = &mut trace.events[0] {
            *reward_bits = (-0.0_f64).to_bits();
        }
        assert!(evaluate_mara_worldtest_exploration(&trace).is_ok());
    }
}
