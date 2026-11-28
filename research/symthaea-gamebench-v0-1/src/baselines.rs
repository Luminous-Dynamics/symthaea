//! Simple policy baselines using the same observation/action-only contract.
//!
//! These are candidate policies, not evaluator oracles. The greedy navigation
//! baseline requires a complete O1 map and performs only one-step Manhattan
//! improvement; it does not search routes or retain memory.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{
    Action, CandidateDecision, CandidateFeedback, CandidateInput, FeatureForecast, ForecastMass,
    ForecastValue, InteractiveCandidate, Observation, validate_decision,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaselineError {
    EmptyActionSet,
    NoRemainingBudget,
    IncompleteMap,
    MissingGoal,
    InvalidDecision,
}

impl Display for BaselineError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyActionSet => write!(f, "no candidate actions were offered"),
            Self::NoRemainingBudget => write!(f, "candidate was called with no remaining actions"),
            Self::IncompleteMap => write!(f, "greedy O1 baseline requires a complete map"),
            Self::MissingGoal => write!(f, "greedy O1 baseline requires a visible goal"),
            Self::InvalidDecision => write!(f, "baseline produced an invalid candidate decision"),
        }
    }
}

impl Error for BaselineError {}

#[derive(Debug, Clone)]
pub struct RandomLegalCandidate {
    base_seed: u64,
    rng_state: u64,
    episode_counter: u64,
}

impl RandomLegalCandidate {
    pub fn new(seed: u64) -> Self {
        Self {
            base_seed: nonzero_seed(seed),
            rng_state: nonzero_seed(seed),
            episode_counter: 0,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        self.rng_state
    }
}

impl InteractiveCandidate<Observation, Action> for RandomLegalCandidate {
    type Error = BaselineError;

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.episode_counter = self.episode_counter.wrapping_add(1);
        self.rng_state = nonzero_seed(
            self.base_seed ^ self.episode_counter.wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        Ok(())
    }

    fn decide(
        &mut self,
        input: CandidateInput<'_, Observation, Action>,
    ) -> Result<CandidateDecision<Action>, Self::Error> {
        if input.remaining_actions == 0 {
            return Err(BaselineError::NoRemainingBudget);
        }
        if input.legal_actions.is_empty() {
            return Err(BaselineError::EmptyActionSet);
        }
        let index = (self.next_u64() as usize) % input.legal_actions.len();
        let decision = CandidateDecision {
            action: input.legal_actions[index],
            confidence: None,
            forecasts: Vec::new(),
        };
        validate_decision(&decision, input.legal_actions)
            .map_err(|_| BaselineError::InvalidDecision)?;
        Ok(decision)
    }

    fn observe_transition(
        &mut self,
        _feedback: CandidateFeedback<'_, Observation, Action>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct GreedyManhattanCandidate;

impl InteractiveCandidate<Observation, Action> for GreedyManhattanCandidate {
    type Error = BaselineError;

    fn reset(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn decide(
        &mut self,
        input: CandidateInput<'_, Observation, Action>,
    ) -> Result<CandidateDecision<Action>, Self::Error> {
        if input.remaining_actions == 0 {
            return Err(BaselineError::NoRemainingBudget);
        }
        let observation = input.observation;
        let goal = observation.known_goal.ok_or(BaselineError::MissingGoal)?;
        let area = usize::from(observation.width)
            .checked_mul(usize::from(observation.height))
            .ok_or(BaselineError::IncompleteMap)?;
        if observation.visible_cells.len() != area {
            return Err(BaselineError::IncompleteMap);
        }

        let mut best: Option<(u32, Action, (u16, u16))> = None;
        for action in input.legal_actions {
            let destination = next_position(observation, *action);
            let distance = u32::from(destination.0.abs_diff(goal.x))
                + u32::from(destination.1.abs_diff(goal.y));
            let replace = best.as_ref().is_none_or(|(best_distance, _, _)| {
                distance < *best_distance
            });
            if replace {
                best = Some((distance, *action, destination));
            }
        }

        let (_, action, (predicted_x, predicted_y)) =
            best.ok_or(BaselineError::EmptyActionSet)?;
        let decision = CandidateDecision {
            action,
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
            .map_err(|_| BaselineError::InvalidDecision)?;
        Ok(decision)
    }

    fn observe_transition(
        &mut self,
        _feedback: CandidateFeedback<'_, Observation, Action>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn nonzero_seed(seed: u64) -> u64 {
    if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    }
}

fn next_position(observation: &Observation, action: Action) -> (u16, u16) {
    let x = observation.position.x;
    let y = observation.position.y;
    let candidate = match action {
        Action::North => y.checked_sub(1).map(|next_y| (x, next_y)),
        Action::East => x.checked_add(1).map(|next_x| (next_x, y)),
        Action::South => y.checked_add(1).map(|next_y| (x, next_y)),
        Action::West => x.checked_sub(1).map(|next_x| (next_x, y)),
    };

    let Some((next_x, next_y)) = candidate else {
        return (x, y);
    };
    if next_x >= observation.width || next_y >= observation.height {
        return (x, y);
    }
    if observation.visible_cells.iter().any(|cell| {
        cell.position.x == next_x && cell.position.y == next_y && cell.cell == crate::Cell::Wall
    }) {
        return (x, y);
    }
    (next_x, next_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cell, Game, GridSpec, ObservationMode, Position};

    fn full_observation() -> Observation {
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
        .expect("valid open grid");
        Game::new(spec).observe(ObservationMode::Full)
    }

    #[test]
    fn seeded_random_candidate_is_reproducible_after_reset() {
        let observation = full_observation();
        let actions = [Action::North, Action::East, Action::South, Action::West];
        let mut first_candidate = RandomLegalCandidate::new(7);
        let mut replay_candidate = RandomLegalCandidate::new(7);
        first_candidate.reset().expect("first reset");
        replay_candidate.reset().expect("replay reset");
        let input = CandidateInput {
            observation: &observation,
            legal_actions: &actions,
            remaining_actions: 12,
            episode_step: 0,
        };
        let first = first_candidate.decide(input).expect("random action");
        let replayed = replay_candidate.decide(input).expect("replayed random action");
        assert_eq!(first.action, replayed.action);
    }

    #[test]
    fn random_candidate_rejects_empty_action_set() {
        let observation = full_observation();
        let mut candidate = RandomLegalCandidate::new(8);
        assert_eq!(
            candidate.decide(CandidateInput {
                observation: &observation,
                legal_actions: &[],
                remaining_actions: 12,
                episode_step: 0,
            }),
            Err(BaselineError::EmptyActionSet)
        );
    }

    #[test]
    fn greedy_baseline_selects_a_one_step_manhattan_improvement() {
        let observation = full_observation();
        let actions = [Action::North, Action::East, Action::South, Action::West];
        let mut candidate = GreedyManhattanCandidate;
        let decision = candidate
            .decide(CandidateInput {
                observation: &observation,
                legal_actions: &actions,
                remaining_actions: 12,
                episode_step: 0,
            })
            .expect("greedy action");
        assert_eq!(decision.action, Action::East);
        assert_eq!(decision.confidence, None);
        assert_eq!(decision.forecasts[0].outcomes[0].value, ForecastValue::Integer(1));
    }

    #[test]
    fn policies_reject_zero_remaining_budget() {
        let observation = full_observation();
        let actions = [Action::North, Action::East, Action::South, Action::West];
        let input = CandidateInput {
            observation: &observation,
            legal_actions: &actions,
            remaining_actions: 0,
            episode_step: 0,
        };
        let mut random = RandomLegalCandidate::new(9);
        assert_eq!(
            random.decide(input),
            Err(BaselineError::NoRemainingBudget)
        );
        let mut greedy = GreedyManhattanCandidate;
        assert_eq!(
            greedy.decide(input),
            Err(BaselineError::NoRemainingBudget)
        );
    }

    #[test]
    fn greedy_baseline_refuses_partial_map_instead_of_assuming_unknown_is_floor() {
        let spec = GridSpec::new(
            3, 1,
            vec![Cell::Floor, Cell::Floor, Cell::Goal],
            Position::new(0, 0),
            Position::new(2, 0),
            4,
        ).expect("valid grid");
        let observation = Game::new(spec).observe(ObservationMode::ManhattanRadius(0));
        let actions = [Action::East, Action::West];
        let mut candidate = GreedyManhattanCandidate;
        assert_eq!(
            candidate.decide(CandidateInput {
                observation: &observation,
                legal_actions: &actions,
                remaining_actions: 4,
                episode_step: 0,
            }),
            Err(BaselineError::IncompleteMap)
        );
    }
}
