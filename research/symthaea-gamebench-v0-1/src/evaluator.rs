//! Evaluator-owned F1 episode runner.
//!
//! The candidate receives observations and action vocabulary only. The exact
//! BFS reference and true transition state remain inside this module. The
//! report exposes replay evidence and aggregate regret, never the oracle plan.

use std::fmt::Display;

use crate::{
    Action, CandidateFeedback, CandidateInput, Game, GridSpec, InteractiveCandidate, Observation,
    ObservationMode, Position, State, TerminalStatus, Trace, validate_decision,
};

const F1_ACTIONS: [Action; 4] = [Action::North, Action::East, Action::South, Action::West];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct F1EpisodeReport {
    /// Environment-computed task outcome, independent of candidate protocol status.
    pub success: bool,
    /// True only when candidate reset/decision/feedback all completed validly.
    pub candidate_protocol_valid: bool,
    pub terminal_status: TerminalStatus,
    pub actions_attempted: u32,
    pub optimal_actions: Option<u32>,
    /// Present only for successful episodes with a reachable exact optimum.
    pub excess_actions: Option<i64>,
    pub final_state: State,
    pub events: Vec<crate::Event>,
    /// Candidate errors or invalid decisions terminate the episode. They are
    /// reported rather than silently repaired or removed from denominators.
    pub candidate_error: Option<String>,
}

pub fn evaluate_f1<C>(
    spec: GridSpec,
    candidate: &mut C,
) -> F1EpisodeReport
where
    C: InteractiveCandidate<Observation, Action>,
    C::Error: Display,
{
    // Exact shortest-path answer is evaluator-private and never passed to the
    // candidate. It is used only to compute the final path-cost regret.
    let optimal_actions = super::optimal_path(&spec)
        .and_then(|path| u32::try_from(path.len()).ok());

    let mut game = Game::new(spec.clone());
    let mut trace = Trace::default();
    let mut candidate_error = None;
    let mut actions_attempted = 0_u32;

    if let Err(error) = candidate.reset() {
        candidate_error = Some(format!("candidate reset failed: {error}"));
    }

    while game.status() == TerminalStatus::Running && candidate_error.is_none() {
        let before = game.observe(ObservationMode::Full);
        let input = CandidateInput {
            observation: &before,
            legal_actions: &F1_ACTIONS,
            remaining_actions: spec.max_actions().saturating_sub(game.state().elapsed_actions),
            episode_step: game.state().elapsed_actions,
        };

        let decision = match candidate.decide(input) {
            Ok(decision) => decision,
            Err(error) => {
                candidate_error = Some(format!("candidate decision failed: {error}"));
                break;
            }
        };

        if let Err(error) = validate_decision(&decision, &F1_ACTIONS) {
            candidate_error = Some(format!("candidate decision invalid: {error}"));
            break;
        }

        let action = decision.action;
        if let Err(error) = trace.record(&mut game, action) {
            candidate_error = Some(format!("environment trace failed: {error}"));
            break;
        }
        actions_attempted = actions_attempted.saturating_add(1);

        let after = game.observe(ObservationMode::Full);
        if let Err(error) = candidate.observe_transition(CandidateFeedback {
            previous_observation: &before,
            action: &action,
            next_observation: &after,
        }) {
            candidate_error = Some(format!("candidate transition feedback failed: {error}"));
            break;
        }
    }

    let final_state = game.state();
    let terminal_status = game.status();
    let success = terminal_status == TerminalStatus::Success;
    let candidate_protocol_valid = candidate_error.is_none();
    let excess_actions = if success {
        optimal_actions.map(|optimal| i64::from(actions_attempted) - i64::from(optimal))
    } else {
        None
    };

    F1EpisodeReport {
        success,
        candidate_protocol_valid,
        terminal_status,
        actions_attempted,
        optimal_actions,
        excess_actions,
        final_state,
        events: trace.events().to_vec(),
        candidate_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cell, GreedyManhattanCandidate, GridSpec};

    fn open_fixture() -> GridSpec {
        GridSpec::new(
            3,
            3,
            vec![
                Cell::Floor, Cell::Floor, Cell::Floor,
                Cell::Floor, Cell::Floor, Cell::Floor,
                Cell::Floor, Cell::Floor, Cell::Goal,
            ],
            Position::new(0, 0),
            Position::new(2, 2),
            8,
        )
        .expect("valid open map")
    }

    #[test]
    fn greedy_candidate_completes_episode_and_matches_exact_path_cost() {
        let spec = open_fixture();
        let mut candidate = GreedyManhattanCandidate;
        let report = evaluate_f1(spec, &mut candidate);

        assert!(report.success);
        assert_eq!(report.terminal_status, TerminalStatus::Success);
        assert_eq!(report.actions_attempted, 4);
        assert_eq!(report.optimal_actions, Some(4));
        assert_eq!(report.excess_actions, Some(0));
        assert_eq!(report.events.len(), 4);
        assert_eq!(report.candidate_error, None);
    }

    #[derive(Debug)]
    struct DecisionFailureCandidate;

    impl InteractiveCandidate<Observation, Action> for DecisionFailureCandidate {
        type Error = std::io::Error;

        fn reset(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }

        fn decide(
            &mut self,
            _input: CandidateInput<'_, Observation, Action>,
        ) -> Result<crate::CandidateDecision<Action>, Self::Error> {
            Err(std::io::Error::other("synthetic decision failure"))
        }

        fn observe_transition(
            &mut self,
            _feedback: CandidateFeedback<'_, Observation, Action>,
        ) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    #[test]
    fn candidate_decision_failure_is_preserved_before_environment_mutation() {
        let spec = open_fixture();
        let mut candidate = DecisionFailureCandidate;
        let report = evaluate_f1(spec, &mut candidate);
        assert!(!report.success);
        assert!(!report.candidate_protocol_valid);
        assert_eq!(report.actions_attempted, 0);
        assert!(report.events.is_empty());
        assert!(report.candidate_error.is_some());
    }

    #[derive(Debug)]
    struct FeedbackFailureCandidate {
        pending: Option<Action>,
    }

    impl InteractiveCandidate<Observation, Action> for FeedbackFailureCandidate {
        type Error = std::io::Error;

        fn reset(&mut self) -> Result<(), Self::Error> {
            self.pending = None;
            Ok(())
        }

        fn decide(
            &mut self,
            _input: CandidateInput<'_, Observation, Action>,
        ) -> Result<crate::CandidateDecision<Action>, Self::Error> {
            let action = Action::East;
            self.pending = Some(action);
            Ok(crate::CandidateDecision {
                action,
                confidence: None,
                forecasts: Vec::new(),
            })
        }

        fn observe_transition(
            &mut self,
            _feedback: CandidateFeedback<'_, Observation, Action>,
        ) -> Result<(), Self::Error> {
            Err(std::io::Error::other("synthetic feedback failure"))
        }
    }

    #[test]
    fn failed_feedback_is_reported_and_does_not_count_as_success() {
        let spec = open_fixture();
        let mut candidate = FeedbackFailureCandidate { pending: None };
        let report = evaluate_f1(spec, &mut candidate);
        assert!(!report.success);
        assert!(!report.candidate_protocol_valid);
        assert_eq!(report.actions_attempted, 1);
        assert!(report.candidate_error.is_some());
        assert_eq!(report.events.len(), 1);
    }

    #[derive(Debug)]
    struct TerminalFeedbackFailureCandidate;

    impl InteractiveCandidate<Observation, Action> for TerminalFeedbackFailureCandidate {
        type Error = std::io::Error;

        fn reset(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }

        fn decide(
            &mut self,
            _input: CandidateInput<'_, Observation, Action>,
        ) -> Result<crate::CandidateDecision<Action>, Self::Error> {
            Ok(crate::CandidateDecision {
                action: Action::East,
                confidence: None,
                forecasts: Vec::new(),
            })
        }

        fn observe_transition(
            &mut self,
            _feedback: CandidateFeedback<'_, Observation, Action>,
        ) -> Result<(), Self::Error> {
            Err(std::io::Error::other("feedback failed after task completion"))
        }
    }

    #[test]
    fn task_outcome_and_candidate_protocol_validity_are_separate() {
        let spec = GridSpec::new(
            2,
            1,
            vec![Cell::Floor, Cell::Goal],
            Position::new(0, 0),
            Position::new(1, 0),
            2,
        )
        .expect("valid one-move map");
        let mut candidate = TerminalFeedbackFailureCandidate;
        let report = evaluate_f1(spec, &mut candidate);
        assert!(report.success);
        assert!(!report.candidate_protocol_valid);
        assert_eq!(report.actions_attempted, 1);
        assert_eq!(report.candidate_error.as_deref(), Some("candidate transition feedback failed: feedback failed after task completion"));
    }

    #[test]
    fn successful_report_events_replay_against_original_spec() {
        let spec = open_fixture();
        let mut candidate = GreedyManhattanCandidate;
        let report = evaluate_f1(spec.clone(), &mut candidate);
        let replay = Trace::replay(spec, &report.events).expect("evaluator replay");
        assert_eq!(replay.status(), TerminalStatus::Success);
        assert_eq!(replay.state(), report.final_state);
    }
}
