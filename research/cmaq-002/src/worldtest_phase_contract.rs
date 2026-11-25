//! WorldTest adapter phase-exit gate.
//!
//! The MARA reference environment exposes "go-to-test" as an actual candidate
//! action. It is charged as an exploration interaction. BeginTest is adapter
//! metadata, not a free agent action. This wrapper rejects traces that omit,
//! duplicate, or move the phase-exit action.

use super::worldtest_exploration::{
    evaluate_worldtest_exploration, ExplorationEvent, ExplorationTrace,
    WorldTestExplorationCertificate, WorldTestExplorationError,
};
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorldTestPhaseContractError {
    MissingBeginTestMarker,
    MissingGoToTestAction,
    GoToTestNotImmediatelyBeforeMarker,
    DuplicateGoToTestAction,
    Trace(WorldTestExplorationError),
}

impl fmt::Display for WorldTestPhaseContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBeginTestMarker => write!(formatter, "WorldTest trace is missing the terminal BeginTest marker"),
            Self::MissingGoToTestAction => write!(formatter, "WorldTest exploration must include the candidate-issued go-to-test action"),
            Self::GoToTestNotImmediatelyBeforeMarker => write!(formatter, "go-to-test must be the final charged exploration action immediately before BeginTest"),
            Self::DuplicateGoToTestAction => write!(formatter, "WorldTest trace contains more than one go-to-test action"),
            Self::Trace(error) => write!(formatter, "invalid WorldTest exploration trace: {error}"),
        }
    }
}

impl std::error::Error for WorldTestPhaseContractError {}

/// Evaluate an exploration trace only after checking the MARA phase-exit
/// contract. The final charged action is Action("go-to-test"); the following
/// BeginTest marker is appended by the trusted adapter when the phase changes.
pub fn evaluate_mara_worldtest_exploration(
    trace: &ExplorationTrace,
) -> Result<WorldTestExplorationCertificate, WorldTestPhaseContractError> {
    let Some(last) = trace.events.last() else {
        return Err(WorldTestPhaseContractError::MissingBeginTestMarker);
    };
    if !matches!(last, ExplorationEvent::BeginTest) {
        return Err(WorldTestPhaseContractError::MissingBeginTestMarker);
    }

    let positions: Vec<usize> = trace
        .events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| match event {
            ExplorationEvent::Action(action_id) if action_id == "go-to-test" => Some(index),
            _ => None,
        })
        .collect();

    if positions.is_empty() {
        return Err(WorldTestPhaseContractError::MissingGoToTestAction);
    }
    if positions.len() != 1 {
        return Err(WorldTestPhaseContractError::DuplicateGoToTestAction);
    }
    if positions[0] + 2 != trace.events.len() {
        return Err(WorldTestPhaseContractError::GoToTestNotImmediatelyBeforeMarker);
    }

    evaluate_worldtest_exploration(trace).map_err(WorldTestPhaseContractError::Trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(value: &str) -> ExplorationEvent {
        ExplorationEvent::Action(value.to_owned())
    }

    fn valid_trace() -> ExplorationTrace {
        ExplorationTrace {
            events: vec![
                action("left"),
                ExplorationEvent::Reset,
                action("right"),
                action("go-to-test"),
                ExplorationEvent::BeginTest,
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
        let trace = ExplorationTrace {
            events: vec![action("left"), ExplorationEvent::BeginTest],
        };
        assert_eq!(
            evaluate_mara_worldtest_exploration(&trace),
            Err(WorldTestPhaseContractError::MissingGoToTestAction)
        );
    }

    #[test]
    fn go_to_test_must_be_the_final_action_before_marker() {
        let trace = ExplorationTrace {
            events: vec![
                action("go-to-test"),
                action("left"),
                ExplorationEvent::BeginTest,
            ],
        };
        assert_eq!(
            evaluate_mara_worldtest_exploration(&trace),
            Err(WorldTestPhaseContractError::GoToTestNotImmediatelyBeforeMarker)
        );
    }

    #[test]
    fn duplicate_go_to_test_actions_are_rejected() {
        let trace = ExplorationTrace {
            events: vec![
                action("go-to-test"),
                action("go-to-test"),
                ExplorationEvent::BeginTest,
            ],
        };
        assert_eq!(
            evaluate_mara_worldtest_exploration(&trace),
            Err(WorldTestPhaseContractError::DuplicateGoToTestAction)
        );
    }

    #[test]
    fn missing_terminal_marker_is_rejected() {
        let trace = ExplorationTrace {
            events: vec![action("go-to-test")],
        };
        assert_eq!(
            evaluate_mara_worldtest_exploration(&trace),
            Err(WorldTestPhaseContractError::MissingBeginTestMarker)
        );
    }
}
