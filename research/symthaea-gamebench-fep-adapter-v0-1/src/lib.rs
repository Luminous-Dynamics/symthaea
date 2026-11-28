#![forbid(unsafe_code)]

//! Narrow candidate adapter for Symthaea's existing FEP ActiveInferenceAgent.
//!
//! This is a real Symthaea submodule baseline, not the complete HDC+CfC
//! cognitive loop. It receives only the F1 observation/action contract.
//! The evaluator owns Game, GridSpec, exact BFS, and all transition truth.

use std::error::Error;
use std::fmt::{Display, Formatter};

mod hdc_candidate;
pub use hdc_candidate::{HdcCandidateError, HdcSpatialCandidate};

use symthaea_fep::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation as FepObservation,
};
use symthaea_gamebench::{
    Action, CandidateDecision, CandidateFeedback, CandidateInput, FeatureForecast, ForecastMass,
    ForecastValue, InteractiveCandidate, Observation, validate_decision,
};

const ACTION_COUNT: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FepAdapterError {
    ZeroDimension,
    MissingGoal,
    IncompleteActionVocabulary,
    NoRemainingBudget,
    DecisionActionMismatch,
    MissingPendingTransition,
    ObservationGeometryChanged,
}

impl Display for FepAdapterError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "navigation dimensions must be nonzero"),
            Self::MissingGoal => write!(f, "FEP navigation baseline requires a visible goal"),
            Self::IncompleteActionVocabulary => {
                write!(f, "F1 must offer all four movement attempts")
            }
            Self::NoRemainingBudget => write!(f, "candidate was called with no remaining actions"),
            Self::DecisionActionMismatch => write!(f, "transition feedback action differs from selected action"),
            Self::MissingPendingTransition => write!(f, "transition feedback arrived without a pending action"),
            Self::ObservationGeometryChanged => write!(f, "grid geometry changed within an episode"),
        }
    }
}

impl Error for FepAdapterError {}

pub struct FepNavigationCandidate {
    width: u16,
    height: u16,
    agent: ActiveInferenceAgent,
    belief_is_current: bool,
    pending_action: Option<Action>,
}

impl FepNavigationCandidate {
    pub fn new(width: u16, height: u16, seed: u64) -> Result<Self, FepAdapterError> {
        if width == 0 || height == 0 {
            return Err(FepAdapterError::ZeroDimension);
        }
        let agent = build_agent(width, height, seed);
        Ok(Self {
            width,
            height,
            agent,
            belief_is_current: false,
            pending_action: None,
        })
    }

    fn encode_observation(
        &mut self,
        observation: &Observation,
    ) -> Result<FepObservation, FepAdapterError> {
        if observation.width != self.width || observation.height != self.height {
            return Err(FepAdapterError::ObservationGeometryChanged);
        }
        let goal = observation.known_goal.ok_or(FepAdapterError::MissingGoal)?;
        let x_denominator = f64::from(self.width.saturating_sub(1)).max(1.0);
        let y_denominator = f64::from(self.height.saturating_sub(1)).max(1.0);

        // Goal preferences are supplied separately to the FEP agent.
        self.agent.set_goals(
            vec![
                f64::from(goal.x) / x_denominator,
                f64::from(goal.y) / y_denominator,
            ],
            4.0,
        );
        Ok(FepObservation::new(
            vec![
                f64::from(observation.position.x) / x_denominator,
                f64::from(observation.position.y) / y_denominator,
            ],
            1.0,
            "gamebench/f1-navigation",
        ))
    }

    fn index_to_action(index: usize) -> Option<Action> {
        match index {
            0 => Some(Action::North),
            1 => Some(Action::East),
            2 => Some(Action::South),
            3 => Some(Action::West),
            _ => None,
        }
    }

    fn predicted_coordinate(mean: f64, extent: u16) -> i64 {
        let max_index = i64::from(extent.saturating_sub(1));
        (mean.clamp(0.0, 1.0) * max_index as f64).round() as i64
    }
    fn coordinate_forecast(
        mean: f64,
        precision: f64,
        extent: u16,
    ) -> Vec<ForecastMass> {
        let max_index = i64::from(extent.saturating_sub(1));
        let normalized_mean = if mean.is_finite() {
            mean.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let mean_index = normalized_mean * max_index as f64;
        let center = mean_index.round() as i64;
        let start = (center - 2).max(0);
        let end = (center + 2).min(max_index);
        let precision = if precision.is_finite() {
            precision.clamp(1e-6, 1e6)
        } else {
            1.0
        };
        let scale = if max_index == 0 {
            0.0
        } else {
            precision.sqrt() / max_index as f64
        };

        let mut weighted: Vec<(i64, f64)> = (start..=end)
            .map(|coordinate| {
                let z = (coordinate as f64 - mean_index) * scale;
                (coordinate, (-0.5 * z * z).exp())
            })
            .collect();
        let total: f64 = weighted.iter().map(|(_, weight)| *weight).sum();
        if total <= 0.0 || !total.is_finite() {
            return vec![ForecastMass {
                value: ForecastValue::Integer(center.clamp(0, max_index)),
                probability: 1.0,
            }];
        }
        weighted
            .drain(..)
            .map(|(coordinate, weight)| ForecastMass {
                value: ForecastValue::Integer(coordinate),
                probability: weight / total,
            })
            .collect()
    }

}

impl InteractiveCandidate<Observation, Action> for FepNavigationCandidate {
    type Error = FepAdapterError;

    fn reset(&mut self, episode_id: u64) -> Result<(), Self::Error> {
        self.agent = build_agent(self.width, self.height, episode_id);
        self.belief_is_current = false;
        self.pending_action = None;
        Ok(())
    }

    fn decide(
        &mut self,
        input: CandidateInput<'_, Observation, Action>,
    ) -> Result<CandidateDecision<Action>, Self::Error> {
        if input.remaining_actions == 0 {
            return Err(FepAdapterError::NoRemainingBudget);
        }
        if ![Action::North, Action::East, Action::South, Action::West]
            .iter()
            .all(|action| input.legal_actions.contains(action))
        {
            return Err(FepAdapterError::IncompleteActionVocabulary);
        }
        if self.pending_action.is_some() {
            return Err(FepAdapterError::MissingPendingTransition);
        }

        let observation = self.encode_observation(input.observation)?;
        if !self.belief_is_current {
            self.agent.perceive(&observation);
            self.belief_is_current = true;
        }

        let selection = self.agent.select_action();
        let action = Self::index_to_action(selection.action)
            .ok_or(FepAdapterError::DecisionActionMismatch)?;
        self.agent.act(selection.action);
        self.pending_action = Some(action);
        self.belief_is_current = false;

        let predicted_x = selection.predicted_state.mean.first().copied().unwrap_or(0.5);
        let predicted_y = selection.predicted_state.mean.get(1).copied().unwrap_or(0.5);
        let precision_x = selection.predicted_state.precision.first().copied().unwrap_or(1.0);
        let precision_y = selection.predicted_state.precision.get(1).copied().unwrap_or(1.0);

        let decision = CandidateDecision {
            action,
            // The FEP softmax is a relative action-selection distribution, not
            // a calibrated probability of task success. Leave confidence
            // unavailable until a held-out calibration map is qualified.
            confidence: None,
            forecasts: vec![
                FeatureForecast {
                    feature_id: 0,
                    outcomes: Self::coordinate_forecast(predicted_x, precision_x, self.width),
                },
                FeatureForecast {
                    feature_id: 1,
                    outcomes: Self::coordinate_forecast(predicted_y, precision_y, self.height),
                },
            ],
        };
        validate_decision(&decision, input.legal_actions)
            .map_err(|_| FepAdapterError::DecisionActionMismatch)?;
        Ok(decision)
    }

    fn observe_transition(
        &mut self,
        feedback: CandidateFeedback<'_, Observation, Action>,
    ) -> Result<(), Self::Error> {
        if self.pending_action != Some(*feedback.action) {
            return Err(FepAdapterError::DecisionActionMismatch);
        }
        if feedback.previous_observation.width != self.width
            || feedback.previous_observation.height != self.height
            || feedback.next_observation.width != self.width
            || feedback.next_observation.height != self.height
        {
            return Err(FepAdapterError::ObservationGeometryChanged);
        }
        // Consume only the next candidate-visible observation. This FEP
        // perception step consumes the previous belief/action for TD learning.
        let next_observation = self.encode_observation(feedback.next_observation)?;
        self.agent.perceive(&next_observation);
        self.belief_is_current = true;
        self.pending_action = None;
        Ok(())
    }
}

fn build_agent(width: u16, height: u16, seed: u64) -> ActiveInferenceAgent {
    let mut config = ActiveInferenceAgentConfig::default();
    config.state_dim = 2;
    config.obs_dim = 2;
    config.num_actions = ACTION_COUNT;
    config.inference_iterations = 5;
    config.planning_horizon = 1;

    let mut agent = ActiveInferenceAgent::new(config);

    // Use a simple identity transition prior with action-conditioned movement
    // biases. Online TD learning can then refine the model from observations.
    for action in 0..ACTION_COUNT {
        agent.model.transition_matrices[action] = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    }
    let dx = if width > 1 {
        1.0 / f64::from(width - 1)
    } else {
        0.0
    };
    let dy = if height > 1 {
        1.0 / f64::from(height - 1)
    } else {
        0.0
    };
    agent.set_transition_bias(0, vec![0.0, -dy]);
    agent.set_transition_bias(1, vec![dx, 0.0]);
    agent.set_transition_bias(2, vec![0.0, dy]);
    agent.set_transition_bias(3, vec![-dx, 0.0]);
    agent.set_rng_seed(seed);
    agent
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_gamebench::{
        Cell, Game, GridSpec, Position, TerminalStatus,
    };

    fn fixture() -> GridSpec {
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
            12,
        )
        .expect("valid open navigation map")
    }

    #[test]
    fn fep_candidate_emits_valid_action_and_prediction() {
        let spec = fixture();
        let game = Game::new(spec);
        let observation = game.observe(symthaea_gamebench::ObservationMode::Full);
        let actions = [Action::North, Action::East, Action::South, Action::West];
        let mut candidate = FepNavigationCandidate::new(3, 3, 42).expect("candidate");
        let decision = candidate
            .decide(CandidateInput {
                observation: &observation,
                legal_actions: &actions,
                remaining_actions: 12,
                episode_step: 0,
            })
            .expect("decision");
        assert!(actions.contains(&decision.action));
        assert_eq!(decision.forecasts.len(), 2);
        assert!((0.0..=1.0).contains(&decision.confidence));
    }

    #[test]
    fn fep_forecasts_are_normalized_and_confidence_is_unavailable() {
        let game = Game::new(fixture());
        let observation = game.observe(symthaea_gamebench::ObservationMode::Full);
        let actions = [Action::North, Action::East, Action::South, Action::West];
        let mut candidate = FepNavigationCandidate::new(3, 3, 421).expect("candidate");
        let decision = candidate
            .decide(CandidateInput {
                observation: &observation,
                legal_actions: &actions,
                remaining_actions: 12,
                episode_step: 0,
            })
            .expect("decision");
        assert_eq!(decision.confidence, None);
        for forecast in decision.forecasts {
            let sum: f64 = forecast.outcomes.iter().map(|mass| mass.probability).sum();
            assert!((sum - 1.0).abs() <= 1e-6);
            assert!(forecast.outcomes.iter().all(|mass| (0.0..=1.0).contains(&mass.probability)));
        }
    }

    #[test]
    fn fep_candidate_closes_each_observation_transition() {
        let spec = fixture();
        let mut game = Game::new(spec);
        let actions = [Action::North, Action::East, Action::South, Action::West];
        let mut candidate = FepNavigationCandidate::new(3, 3, 11).expect("candidate");
        let before = game.observe(symthaea_gamebench::ObservationMode::Full);
        let decision = candidate
            .decide(CandidateInput {
                observation: &before,
                legal_actions: &actions,
                remaining_actions: 12,
                episode_step: 0,
            })
            .expect("first decision");
        game.step(decision.action).expect("environment accepts action");
        let after = game.observe(symthaea_gamebench::ObservationMode::Full);
        candidate
            .observe_transition(CandidateFeedback {
                previous_observation: &before,
                action: &decision.action,
                next_observation: &after,
            })
            .expect("transition feedback");

        let next = candidate
            .decide(CandidateInput {
                observation: &after,
                legal_actions: &actions,
                remaining_actions: 11,
                episode_step: 1,
            })
            .expect("second decision");
        assert!(actions.contains(&next.action));
    }

    #[test]
    fn fep_candidate_rejects_a_second_decision_without_feedback() {
        let game = Game::new(fixture());
        let observation = game.observe(symthaea_gamebench::ObservationMode::Full);
        let actions = [Action::North, Action::East, Action::South, Action::West];
        let mut candidate = FepNavigationCandidate::new(3, 3, 12).expect("candidate");
        candidate
            .decide(CandidateInput {
                observation: &observation,
                legal_actions: &actions,
                remaining_actions: 12,
                episode_step: 0,
            })
            .expect("first decision");
        assert_eq!(
            candidate.decide(CandidateInput {
                observation: &observation,
                legal_actions: &actions,
                remaining_actions: 11,
                episode_step: 1,
            }),
            Err(FepAdapterError::MissingPendingTransition)
        );
    }

    #[test]
    fn reset_restores_a_fresh_fep_agent_with_declared_seed() {
        let mut candidate = FepNavigationCandidate::new(3, 3, 1).expect("candidate");
        assert!(candidate.reset(99).is_ok());
        assert!(candidate.pending_action.is_none());
        assert!(!candidate.belief_is_current);
    }

    #[test]
    fn terminal_episode_can_still_supply_one_final_transition() {
        let spec = GridSpec::new(
            2,
            1,
            vec![Cell::Floor, Cell::Goal],
            Position::new(0, 0),
            Position::new(1, 0),
            2,
        )
        .expect("valid terminal fixture");
        let mut game = Game::new(spec);
        let before = game.observe(symthaea_gamebench::ObservationMode::Full);
        let mut candidate = FepNavigationCandidate::new(2, 1, 3).expect("candidate");
        candidate.pending_action = Some(Action::East);
        game.step(Action::East).expect("goal entry");
        assert_eq!(game.status(), TerminalStatus::Success);
        let after = game.observe(symthaea_gamebench::ObservationMode::Full);
        candidate
            .observe_transition(CandidateFeedback {
                previous_observation: &before,
                action: &Action::East,
                next_observation: &after,
            })
            .expect("final observation feedback");
        assert!(candidate.pending_action.is_none());
    }
}
