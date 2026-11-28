//! Candidate-side frontier exploration policy for the F4 partial-map track.
//!
//! The policy receives only current observations and maintains its own
//! ExploreMemory. It never sees the environment specification or hidden cells.

use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{
    CandidateDecision, CandidateFeedback, CandidateInput, ExploreAction, ExploreMemory,
    ExploreMemoryError, ExploreObservation, ExploreState, ExploreStepError, ExploreTransition,
    ExploreGame, ExploreSpec, FeatureForecast, ForecastMass, ForecastValue, InteractiveCandidate,
    Position, validate_decision,
};
use crate::Action;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryExplorerError {
    Memory(ExploreMemoryError),
    EmptyActionSet,
    NoRemainingBudget,
    DecisionWithoutFeedback,
    FeedbackWithoutPendingAction,
    ActionMismatch,
    NoReachableFrontier,
    NoKnownRouteToGoal,
    InvalidDecision,
}

impl Display for MemoryExplorerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Memory(error) => write!(f, "map memory rejected observation: {error}"),
            Self::EmptyActionSet => write!(f, "explorer received no offered actions"),
            Self::NoRemainingBudget => write!(f, "explorer received no remaining action budget"),
            Self::DecisionWithoutFeedback => write!(f, "explorer awaits transition feedback"),
            Self::FeedbackWithoutPendingAction => write!(f, "explorer received feedback without a pending action"),
            Self::ActionMismatch => write!(f, "feedback action differs from pending exploration action"),
            Self::NoReachableFrontier => write!(f, "no known reachable exploration frontier remains"),
            Self::NoKnownRouteToGoal => write!(f, "visible goal has no route through known traversable cells"),
            Self::InvalidDecision => write!(f, "explorer produced an invalid decision"),
        }
    }
}

impl Error for MemoryExplorerError {}

impl From<ExploreMemoryError> for MemoryExplorerError {
    fn from(value: ExploreMemoryError) -> Self {
        Self::Memory(value)
    }
}

#[derive(Debug, Clone)]
pub struct MemoryFrontierCandidate {
    memory: ExploreMemory,
    last_observation: Option<ExploreObservation>,
    pending_action: Option<ExploreAction>,
}

impl MemoryFrontierCandidate {
    pub fn new(width: u16, height: u16) -> Result<Self, MemoryExplorerError> {
        Ok(Self {
            memory: ExploreMemory::new(width, height)?,
            last_observation: None,
            pending_action: None,
        })
    }

    pub fn memory(&self) -> &ExploreMemory {
        &self.memory
    }

    fn path_to(&self, start: Position, target: Position) -> Option<Vec<Action>> {
        if start == target {
            return Some(Vec::new());
        }

        let mut queue = VecDeque::from([start]);
        let mut seen = HashSet::from([start]);
        let mut parent: HashMap<Position, (Position, Action)> = HashMap::new();

        while let Some(position) = queue.pop_front() {
            for action in [Action::North, Action::East, Action::South, Action::West] {
                let Some(next) = adjacent(position, action) else {
                    continue;
                };
                if next.x >= self.memory.width() || next.y >= self.memory.height() {
                    continue;
                }
                if !self
                    .memory
                    .known_cell(next)
                    .is_some_and(|cell| cell != crate::Cell::Wall)
                {
                    continue;
                }
                if !seen.insert(next) {
                    continue;
                }
                parent.insert(next, (position, action));
                if next == target {
                    let mut cursor = target;
                    let mut path = Vec::new();
                    while cursor != start {
                        let (previous, step) = *parent.get(&cursor)?;
                        path.push(step);
                        cursor = previous;
                    }
                    path.reverse();
                    return Some(path);
                }
                queue.push_back(next);
            }
        }
        None
    }

    fn next_policy_action(
        &self,
        observation: &ExploreObservation,
        offered: &[ExploreAction],
    ) -> Result<ExploreAction, MemoryExplorerError> {
        let current = observation.position;

        let goal = self
            .memory
            .known_cells()
            .iter()
            .find_map(|(position, cell)| (*cell == crate::Cell::Goal).then_some(*position));

        if let Some(goal_position) = goal {
            if let Some(path) = self.path_to(current, goal_position) {
                if let Some(first) = path.first() {
                    let action = ExploreAction::Move(*first);
                    if offered.contains(&action) {
                        return Ok(action);
                    }
                    return Err(MemoryExplorerError::InvalidDecision);
                }
            }
        }

        // Scan when currently standing at a frontier, unless this exact
        // observation is already the result of a scan. That avoids free or
        // repeated sensing loops.
        if !observation.scan_was_used_for_this_observation
            && self.memory.frontier_positions().contains(&current)
            && offered.contains(&ExploreAction::Scan)
        {
            return Ok(ExploreAction::Scan);
        }

        let mut choices: Vec<(usize, Position, Vec<Action>)> = self
            .memory
            .frontier_positions()
            .into_iter()
            .filter(|position| *position != current)
            .filter_map(|position| {
                self.path_to(current, position)
                    .map(|path| (path.len(), position, path))
            })
            .collect();
        choices.sort_by_key(|(distance, position, _)| (*distance, *position));

        if let Some((_, _, path)) = choices.into_iter().next() {
            if let Some(first) = path.first() {
                let action = ExploreAction::Move(*first);
                if offered.contains(&action) {
                    return Ok(action);
                }
            }
        }

        if offered.contains(&ExploreAction::Scan)
            && !observation.scan_was_used_for_this_observation
        {
            return Ok(ExploreAction::Scan);
        }

        if goal.is_some() {
            Err(MemoryExplorerError::NoKnownRouteToGoal)
        } else {
            Err(MemoryExplorerError::NoReachableFrontier)
        }
    }
}

impl InteractiveCandidate<ExploreObservation, ExploreAction> for MemoryFrontierCandidate {
    type Error = MemoryExplorerError;

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.memory.reset();
        self.last_observation = None;
        self.pending_action = None;
        Ok(())
    }

    fn decide(
        &mut self,
        input: CandidateInput<'_, ExploreObservation, ExploreAction>,
    ) -> Result<CandidateDecision<ExploreAction>, Self::Error> {
        if self.pending_action.is_some() {
            return Err(MemoryExplorerError::DecisionWithoutFeedback);
        }
        if input.remaining_actions == 0 {
            return Err(MemoryExplorerError::NoRemainingBudget);
        }
        if input.legal_actions.is_empty() {
            return Err(MemoryExplorerError::EmptyActionSet);
        }
        if self.last_observation.as_ref() != Some(input.observation) {
            self.memory.remember(input.observation)?;
            self.last_observation = Some(input.observation.clone());
        }

        let action = self.next_policy_action(input.observation, input.legal_actions)?;
        let decision = CandidateDecision {
            action,
            confidence: None,
            forecasts: Vec::new(),
        };
        validate_decision(&decision, input.legal_actions)
            .map_err(|_| MemoryExplorerError::InvalidDecision)?;
        self.pending_action = Some(action);
        Ok(decision)
    }

    fn observe_transition(
        &mut self,
        feedback: CandidateFeedback<'_, ExploreObservation, ExploreAction>,
    ) -> Result<(), Self::Error> {
        let pending = self
            .pending_action
            .ok_or(MemoryExplorerError::FeedbackWithoutPendingAction)?;
        if pending != *feedback.action {
            return Err(MemoryExplorerError::ActionMismatch);
        }
        self.memory.remember(feedback.next_observation)?;
        self.last_observation = Some(feedback.next_observation.clone());
        self.pending_action = None;
        Ok(())
    }
}

fn adjacent(position: Position, action: Action) -> Option<Position> {
    match action {
        Action::North => Some(Position::new(position.x, position.y.checked_sub(1)?)),
        Action::East => Some(Position::new(position.x.checked_add(1)?, position.y)),
        Action::South => Some(Position::new(position.x, position.y.checked_add(1)?)),
        Action::West => Some(Position::new(position.x.checked_sub(1)?, position.y)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cell, ExploreGame, ExploreSpec};

    fn fixture(max_actions: u32) -> ExploreSpec {
        ExploreSpec::new(
            7,
            3,
            vec![
                Cell::Wall, Cell::Wall, Cell::Wall, Cell::Wall, Cell::Wall, Cell::Wall, Cell::Wall,
                Cell::Floor, Cell::Floor, Cell::Floor, Cell::Wall, Cell::Floor, Cell::Floor, Cell::Goal,
                Cell::Floor, Cell::Floor, Cell::Floor, Cell::Floor, Cell::Floor, Cell::Floor, Cell::Floor,
            ],
            Position::new(0, 1),
            Position::new(6, 1),
            max_actions,
            1,
            4,
        )
        .expect("valid navigable F4 map")
    }

    #[test]
    fn memory_frontier_candidate_can_reach_goal_on_partial_map() {
        let mut game = ExploreGame::new(fixture(60));
        let mut candidate = MemoryFrontierCandidate::new(7, 3).expect("candidate");
        candidate.reset().expect("reset");
        let offered = [
            ExploreAction::Move(Action::North),
            ExploreAction::Move(Action::East),
            ExploreAction::Move(Action::South),
            ExploreAction::Move(Action::West),
            ExploreAction::Scan,
        ];

        while game.status() == crate::TerminalStatus::Running {
            let before = game.observe();
            let decision = candidate
                .decide(CandidateInput {
                    observation: &before,
                    legal_actions: &offered,
                    remaining_actions: 60 - game.state().elapsed_actions,
                    episode_step: game.state().elapsed_actions,
                })
                .expect("candidate decision");
            let transition = game.step(decision.action).expect("environment step");
            candidate
                .observe_transition(CandidateFeedback {
                    previous_observation: &before,
                    action: &decision.action,
                    next_observation: &transition.observation,
                })
                .expect("candidate memory update");
            assert!(candidate.memory().observation_count() > 0);
            if game.state().elapsed_actions > 60 {
                panic!("F4 candidate exceeded action budget");
            }
        }

        assert_eq!(game.status(), crate::TerminalStatus::Success);
        assert!(candidate.memory().known_cell(Position::new(6, 1)) == Some(Cell::Goal));
        assert!(candidate.memory().coverage() > 0.0);
    }

    #[test]
    fn memory_frontier_candidate_requires_feedback_before_next_decision() {
        let game = ExploreGame::new(fixture(20));
        let observation = game.observe();
        let offered = [
            ExploreAction::Move(Action::North),
            ExploreAction::Move(Action::East),
            ExploreAction::Move(Action::South),
            ExploreAction::Move(Action::West),
            ExploreAction::Scan,
        ];
        let mut candidate = MemoryFrontierCandidate::new(7, 3).expect("candidate");
        let first = candidate
            .decide(CandidateInput {
                observation: &observation,
                legal_actions: &offered,
                remaining_actions: 20,
                episode_step: 0,
            })
            .expect("first decision");
        assert_eq!(
            candidate.decide(CandidateInput {
                observation: &observation,
                legal_actions: &offered,
                remaining_actions: 19,
                episode_step: 1,
            }),
            Err(MemoryExplorerError::DecisionWithoutFeedback)
        );
        assert!(offered.contains(&first.action));
    }

    #[test]
    fn memory_frontier_candidate_rejects_zero_budget() {
        let game = ExploreGame::new(fixture(20));
        let observation = game.observe();
        let offered = [ExploreAction::Scan];
        let mut candidate = MemoryFrontierCandidate::new(7, 3).expect("candidate");
        assert_eq!(
            candidate.decide(CandidateInput {
                observation: &observation,
                legal_actions: &offered,
                remaining_actions: 0,
                episode_step: 0,
            }),
            Err(MemoryExplorerError::NoRemainingBudget)
        );
    }
}
