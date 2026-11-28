#![forbid(unsafe_code)]

//! Temporal HDC-LTC candidate using the real symthaea-core
//! HdcLtcUnifiedNeuron. This is a small stateful temporal baseline, not a
//! complete Symthaea HDC+CfC or production controller.

use std::error::Error;
use std::fmt::{Display, Formatter};

use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_gamebench::{
    Action, CandidateDecision, CandidateFeedback, CandidateInput, FeatureForecast, ForecastMass,
    ForecastValue, InteractiveCandidate, Observation, validate_decision,
};

use crate::hdc_candidate::SpatialHdcEncoder;

const EVOLUTION_DT: f32 = 0.1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HdcLtcAdapterError {
    IncompleteMap,
    MissingGoal,
    EmptyActionSet,
    DecisionWithoutFeedback,
    FeedbackWithoutPendingAction,
    ActionMismatch,
    GeometryChanged,
    InvalidDecision,
}

impl Display for HdcLtcAdapterError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IncompleteMap => write!(f, "HDC-LTC baseline requires a complete O1 map"),
            Self::MissingGoal => write!(f, "HDC-LTC baseline requires a visible goal"),
            Self::EmptyActionSet => write!(f, "HDC-LTC candidate received no offered actions"),
            Self::DecisionWithoutFeedback => write!(f, "HDC-LTC candidate awaits previous transition feedback"),
            Self::FeedbackWithoutPendingAction => write!(f, "HDC-LTC transition feedback has no pending action"),
            Self::ActionMismatch => write!(f, "HDC-LTC feedback action differs from selected action"),
            Self::GeometryChanged => write!(f, "grid geometry changed during HDC-LTC episode"),
            Self::InvalidDecision => write!(f, "HDC-LTC candidate produced an invalid decision"),
        }
    }
}

impl Error for HdcLtcAdapterError {}

pub struct HdcLtcSpatialCandidate {
    width: u16,
    height: u16,
    seed: u64,
    encoder: SpatialHdcEncoder,
    neuron: HdcLtcUnifiedNeuron,
    initialized: bool,
    pending_action: Option<Action>,
}

impl HdcLtcSpatialCandidate {
    pub fn new(width: u16, height: u16, seed: u64) -> Result<Self, HdcLtcAdapterError> {
        let encoder = SpatialHdcEncoder::new(width, height, seed)
            .map_err(|_| HdcLtcAdapterError::GeometryChanged)?;
        let neuron = build_neuron(seed);
        Ok(Self {
            width,
            height,
            seed,
            encoder,
            neuron,
            initialized: false,
            pending_action: None,
        })
    }

    fn next_position(observation: &Observation, action: Action) -> (u16, u16) {
        let current = observation.position;
        let candidate = match action {
            Action::North => current.y.checked_sub(1).map(|y| (current.x, y)),
            Action::East => current.x.checked_add(1).map(|x| (x, current.y)),
            Action::South => current.y.checked_add(1).map(|y| (current.x, y)),
            Action::West => current.x.checked_sub(1).map(|x| (x, current.y)),
        };

        let Some((x, y)) = candidate else {
            return (current.x, current.y);
        };
        if x >= observation.width || y >= observation.height {
            return (current.x, current.y);
        }
        let is_wall = observation.visible_cells.iter().any(|cell| {
            cell.position.x == x
                && cell.position.y == y
                && cell.cell == symthaea_gamebench::Cell::Wall
        });
        if is_wall {
            (current.x, current.y)
        } else {
            (x, y)
        }
    }

    fn validate_observation(&self, observation: &Observation) -> Result<(), HdcLtcAdapterError> {
        if observation.width != self.width || observation.height != self.height {
            return Err(HdcLtcAdapterError::GeometryChanged);
        }
        let area = usize::from(self.width)
            .checked_mul(usize::from(self.height))
            .ok_or(HdcLtcAdapterError::IncompleteMap)?;
        if observation.visible_cells.len() != area {
            return Err(HdcLtcAdapterError::IncompleteMap);
        }
        Ok(())
    }
}

impl InteractiveCandidate<Observation, Action> for HdcLtcSpatialCandidate {
    type Error = HdcLtcAdapterError;

    fn reset(&mut self, episode_id: u64) -> Result<(), Self::Error> {
        self.seed = episode_id;
        self.neuron = build_neuron(episode_id);
        self.initialized = false;
        self.pending_action = None;
        Ok(())
    }

    fn decide(
        &mut self,
        input: CandidateInput<'_, Observation, Action>,
    ) -> Result<CandidateDecision<Action>, Self::Error> {
        if self.pending_action.is_some() {
            return Err(HdcLtcAdapterError::DecisionWithoutFeedback);
        }
        self.validate_observation(input.observation)?;
        let goal = input.observation.known_goal.ok_or(HdcLtcAdapterError::MissingGoal)?;
        if input.legal_actions.is_empty() {
            return Err(HdcLtcAdapterError::EmptyActionSet);
        }

        let current_position_hv = self
            .encoder
            .encode(input.observation.position.x, input.observation.position.y);
        if !self.initialized {
            self.neuron.set_state(current_position_hv);
            self.initialized = true;
        }

        let goal_hv = self.encoder.encode(goal.x, goal.y);
        let mut best: Option<(f32, Action, (u16, u16))> = None;
        for action in input.legal_actions {
            let next_position = Self::next_position(input.observation, *action);
            let candidate_input = self.encoder.encode(next_position.0, next_position.1);
            let mut trial_neuron = self.neuron.clone();
            trial_neuron.evolve_closed_form(EVOLUTION_DT, &candidate_input);
            let score = trial_neuron.state().similarity(&goal_hv);
            if best.as_ref().is_none_or(|(best_score, _, _)| score > *best_score) {
                best = Some((score, *action, next_position));
            }
        }

        let (_, action, (predicted_x, predicted_y)) =
            best.ok_or(HdcLtcAdapterError::EmptyActionSet)?;
        let decision = CandidateDecision {
            action,
            // HDC similarity and action ranking are not calibrated success
            // probabilities. Leave confidence unavailable.
            confidence: None,
            forecasts: vec![
                FeatureForecast {
                    feature_id: 0,
                    outcomes: vec![ForecastMass {
                        value: ForecastValue::Integer(i64::from(predicted_x)),
                        probability: 1.0,
                    }],
                },
                FeatureForecast {
                    feature_id: 1,
                    outcomes: vec![ForecastMass {
                        value: ForecastValue::Integer(i64::from(predicted_y)),
                        probability: 1.0,
                    }],
                },
            ],
        };
        validate_decision(&decision, input.legal_actions)
            .map_err(|_| HdcLtcAdapterError::InvalidDecision)?;
        self.pending_action = Some(action);
        Ok(decision)
    }

    fn observe_transition(
        &mut self,
        feedback: CandidateFeedback<'_, Observation, Action>,
    ) -> Result<(), Self::Error> {
        let pending = self
            .pending_action
            .ok_or(HdcLtcAdapterError::FeedbackWithoutPendingAction)?;
        if pending != *feedback.action {
            return Err(HdcLtcAdapterError::ActionMismatch);
        }
        self.validate_observation(feedback.previous_observation)?;
        self.validate_observation(feedback.next_observation)?;

        // Update temporal state from the *observed* next position rather than
        // committing any hypothetical action rollout from decide().
        let observed = self.encoder.encode(
            feedback.next_observation.position.x,
            feedback.next_observation.position.y,
        );
        self.neuron.evolve_closed_form(EVOLUTION_DT, &observed);
        self.pending_action = None;
        Ok(())
    }
}

fn build_neuron(seed: u64) -> HdcLtcUnifiedNeuron {
    let config = UnifiedConfig::default();
    HdcLtcUnifiedNeuron::new(config, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_gamebench::{Cell, Game, GridSpec, ObservationMode, Position};

    fn fixture() -> Observation {
        let spec = GridSpec::new(
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
        .expect("valid map");
        Game::new(spec).observe(ObservationMode::Full)
    }

    #[test]
    fn hdc_ltc_candidate_emits_an_offered_action() {
        let observation = fixture();
        let actions = [Action::North, Action::East, Action::South, Action::West];
        let mut candidate = HdcLtcSpatialCandidate::new(3, 3, 42).expect("candidate");
        let decision = candidate
            .decide(CandidateInput {
                observation: &observation,
                legal_actions: &actions,
                remaining_actions: 12,
                episode_step: 0,
            })
            .expect("decision");
        assert!(actions.contains(&decision.action));
        assert_eq!(decision.confidence, None);
    }

    #[test]
    fn hdc_ltc_candidate_consumes_actual_transition_feedback() {
        let spec = GridSpec::new(
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
        .expect("valid map");
        let mut game = Game::new(spec);
        let actions = [Action::North, Action::East, Action::South, Action::West];
        let mut candidate = HdcLtcSpatialCandidate::new(3, 3, 43).expect("candidate");
        let before = game.observe(ObservationMode::Full);
        let decision = candidate
            .decide(CandidateInput {
                observation: &before,
                legal_actions: &actions,
                remaining_actions: 12,
                episode_step: 0,
            })
            .expect("decision");
        game.step(decision.action).expect("step");
        let after = game.observe(ObservationMode::Full);
        candidate
            .observe_transition(CandidateFeedback {
                previous_observation: &before,
                action: &decision.action,
                next_observation: &after,
            })
            .expect("feedback");
        assert!(candidate.pending_action.is_none());
    }

    #[test]
    fn hdc_ltc_candidate_requires_feedback_before_next_action() {
        let observation = fixture();
        let actions = [Action::North, Action::East, Action::South, Action::West];
        let mut candidate = HdcLtcSpatialCandidate::new(3, 3, 44).expect("candidate");
        candidate
            .decide(CandidateInput {
                observation: &observation,
                legal_actions: &actions,
                remaining_actions: 12,
                episode_step: 0,
            })
            .expect("first action");
        assert!(matches!(
            candidate.decide(CandidateInput {
                observation: &observation,
                legal_actions: &actions,
                remaining_actions: 11,
                episode_step: 1,
            }),
            Err(HdcLtcAdapterError::DecisionWithoutFeedback)
        ));
    }

    #[test]
    fn hdc_ltc_reset_clears_temporal_state_and_pending_action() {
        let observation = fixture();
        let actions = [Action::North, Action::East, Action::South, Action::West];
        let mut candidate = HdcLtcSpatialCandidate::new(3, 3, 45).expect("candidate");
        candidate
            .decide(CandidateInput {
                observation: &observation,
                legal_actions: &actions,
                remaining_actions: 12,
                episode_step: 0,
            })
            .expect("decision");
        candidate.reset(46).expect("reset");
        assert!(!candidate.initialized);
        assert!(candidate.pending_action.is_none());
    }
}
