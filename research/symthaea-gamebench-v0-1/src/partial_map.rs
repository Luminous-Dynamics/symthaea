//! Partially observable map and active scan fixture (GameBench F4).
//!
//! Observations contain only the current field of view. The environment does
//! not carry the agent's observation history; candidate-side memory must do so.
//! A scan consumes one action and temporarily widens the current observation.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{Action, Cell, Position, TerminalStatus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExploreSpecError {
    ZeroDimension,
    CellCountMismatch { expected: usize, actual: usize },
    PositionOutOfBounds(&'static str),
    ZeroActionLimit,
    StartBlocked,
    GoalNotMarked,
}

impl Display for ExploreSpecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "exploration map dimensions must be nonzero"),
            Self::CellCountMismatch { expected, actual } => {
                write!(f, "exploration map needs {expected} cells, received {actual}")
            }
            Self::PositionOutOfBounds(name) => {
                write!(f, "exploration {name} position is outside the map")
            }
            Self::ZeroActionLimit => write!(f, "maximum action count must be nonzero"),
            Self::StartBlocked => write!(f, "exploration start position cannot be a wall"),
            Self::GoalNotMarked => write!(f, "exploration goal cell must be marked as Goal"),
        }
    }
}

impl Error for ExploreSpecError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExploreSpec {
    width: u16,
    height: u16,
    cells: Vec<Cell>,
    start: Position,
    goal: Position,
    max_actions: u32,
    view_radius: u16,
    scan_radius: u16,
}

impl ExploreSpec {
    pub fn new(
        width: u16,
        height: u16,
        cells: Vec<Cell>,
        start: Position,
        goal: Position,
        max_actions: u32,
        view_radius: u16,
        scan_radius: u16,
    ) -> Result<Self, ExploreSpecError> {
        if width == 0 || height == 0 {
            return Err(ExploreSpecError::ZeroDimension);
        }
        let expected = usize::from(width) * usize::from(height);
        if cells.len() != expected {
            return Err(ExploreSpecError::CellCountMismatch {
                expected,
                actual: cells.len(),
            });
        }
        if max_actions == 0 {
            return Err(ExploreSpecError::ZeroActionLimit);
        }
        if start.x >= width || start.y >= height {
            return Err(ExploreSpecError::PositionOutOfBounds("start"));
        }
        if goal.x >= width || goal.y >= height {
            return Err(ExploreSpecError::PositionOutOfBounds("goal"));
        }

        let cell_at = |position: Position| {
            let index = usize::from(position.y) * usize::from(width) + usize::from(position.x);
            cells[index]
        };
        if cell_at(start) == Cell::Wall {
            return Err(ExploreSpecError::StartBlocked);
        }
        if cell_at(goal) != Cell::Goal {
            return Err(ExploreSpecError::GoalNotMarked);
        }

        Ok(Self {
            width,
            height,
            cells,
            start,
            goal,
            max_actions,
            view_radius,
            scan_radius: scan_radius.max(view_radius),
        })
    }

    fn cell_at(&self, position: Position) -> Option<Cell> {
        if position.x >= self.width || position.y >= self.height {
            return None;
        }
        let index = usize::from(position.y) * usize::from(self.width) + usize::from(position.x);
        self.cells.get(index).copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExploreState {
    pub position: Position,
    pub elapsed_actions: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExploreAction {
    Move(Action),
    Scan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservedCell {
    pub position: Position,
    pub cell: Cell,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExploreObservation {
    pub position: Position,
    pub visible_cells: Vec<ObservedCell>,
    pub remaining_actions: u32,
    pub scan_was_used_for_this_observation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExploreTransition {
    pub state: ExploreState,
    pub moved: bool,
    pub scanned: bool,
    pub status: TerminalStatus,
    pub observation: ExploreObservation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExploreStepError {
    AlreadyTerminal(TerminalStatus),
}

impl Display for ExploreStepError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyTerminal(status) => {
                write!(f, "cannot explore after terminal status {status:?}")
            }
        }
    }
}

impl Error for ExploreStepError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExploreGame {
    spec: ExploreSpec,
    state: ExploreState,
    status: TerminalStatus,
}

impl ExploreGame {
    pub fn new(spec: ExploreSpec) -> Self {
        let state = ExploreState {
            position: spec.start,
            elapsed_actions: 0,
        };
        let status = if state.position == spec.goal {
            TerminalStatus::Success
        } else {
            TerminalStatus::Running
        };
        Self { spec, state, status }
    }

    pub const fn state(&self) -> ExploreState {
        self.state
    }

    pub const fn status(&self) -> TerminalStatus {
        self.status
    }

    /// Returns a fresh local observation only; no previous observation is
    /// embedded. Absence from visible_cells means unknown, not floor.
    pub fn observe(&self) -> ExploreObservation {
        self.observe_with_radius(self.spec.view_radius, false)
    }

    pub fn step(
        &mut self,
        action: ExploreAction,
    ) -> Result<ExploreTransition, ExploreStepError> {
        if self.status != TerminalStatus::Running {
            return Err(ExploreStepError::AlreadyTerminal(self.status));
        }

        let mut moved = false;
        let scanned = matches!(action, ExploreAction::Scan);
        let observation_radius = if scanned {
            self.spec.scan_radius
        } else {
            self.spec.view_radius
        };

        if let ExploreAction::Move(direction) = action {
            if let Some(destination) = adjacent(self.state.position, direction) {
                if self.spec.cell_at(destination).is_some_and(|cell| cell != Cell::Wall) {
                    self.state.position = destination;
                    moved = true;
                }
            }
        }

        // Scanning, successful movement, and blocked movement all consume one
        // action. This prevents free repeated observation.
        self.state.elapsed_actions += 1;
        self.status = if self.state.position == self.spec.goal {
            TerminalStatus::Success
        } else if self.state.elapsed_actions >= self.spec.max_actions {
            TerminalStatus::ActionLimit
        } else {
            TerminalStatus::Running
        };

        Ok(ExploreTransition {
            state: self.state,
            moved,
            scanned,
            status: self.status,
            observation: self.observe_with_radius(observation_radius, scanned),
        })
    }

    fn observe_with_radius(
        &self,
        radius: u16,
        scan_was_used: bool,
    ) -> ExploreObservation {
        let mut visible_cells = Vec::new();
        for y in 0..self.spec.height {
            for x in 0..self.spec.width {
                let position = Position::new(x, y);
                if self.state.position.manhattan_distance(position) <= u32::from(radius) {
                    if let Some(cell) = self.spec.cell_at(position) {
                        visible_cells.push(ObservedCell { position, cell });
                    }
                }
            }
        }

        ExploreObservation {
            position: self.state.position,
            visible_cells,
            remaining_actions: self.spec.max_actions - self.state.elapsed_actions,
            scan_was_used_for_this_observation: scan_was_used,
        }
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

    fn fixture(max_actions: u32) -> ExploreSpec {
        // P . # . G. The wall at x=2 and goal at x=4 are not initially visible.
        ExploreSpec::new(
            5,
            1,
            vec![Cell::Floor, Cell::Floor, Cell::Wall, Cell::Floor, Cell::Goal],
            Position::new(0, 0),
            Position::new(4, 0),
            max_actions,
            1,
            2,
        )
        .expect("valid partially observable fixture")
    }

    #[test]
    fn initial_observation_does_not_disclose_distant_wall_or_goal() {
        let game = ExploreGame::new(fixture(12));
        let observation = game.observe();
        assert_eq!(observation.position, Position::new(0, 0));
        assert_eq!(observation.visible_cells.len(), 2);
        assert!(!observation.visible_cells.iter().any(|cell| {
            cell.position == Position::new(2, 0) || cell.position == Position::new(4, 0)
        }));
    }

    #[test]
    fn scan_costs_one_action_and_reveals_only_local_extended_view() {
        let mut game = ExploreGame::new(fixture(12));
        let transition = game.step(ExploreAction::Scan).expect("scan consumes a turn");
        assert!(transition.scanned);
        assert!(!transition.moved);
        assert_eq!(transition.state.elapsed_actions, 1);
        assert_eq!(transition.observation.visible_cells.len(), 3);
        assert!(transition.observation.visible_cells.iter().any(|cell| {
            cell.position == Position::new(2, 0) && cell.cell == Cell::Wall
        }));
        assert!(!transition.observation.visible_cells.iter().any(|cell| {
            cell.position == Position::new(4, 0)
        }));
    }

    #[test]
    fn observations_are_current_views_not_accumulated_memory() {
        let mut game = ExploreGame::new(fixture(12));
        let scan = game.step(ExploreAction::Scan).expect("scan");
        assert!(scan.observation.visible_cells.iter().any(|cell| {
            cell.position == Position::new(0, 0)
        }));

        game.step(ExploreAction::Move(Action::East)).expect("move to x=1");
        let moved = game.step(ExploreAction::Move(Action::East)).expect("blocked at wall x=2");
        assert_eq!(moved.state.position, Position::new(1, 0));
        assert!(!moved.observation.visible_cells.iter().any(|cell| {
            cell.position == Position::new(4, 0)
        }));
        // The environment never attaches the earlier scan result to this view.
        assert!(!moved.observation.scan_was_used_for_this_observation);
    }

    #[test]
    fn wall_collision_consumes_action_and_preserves_position() {
        let mut game = ExploreGame::new(fixture(2));
        game.step(ExploreAction::Scan).expect("scan");
        let transition = game
            .step(ExploreAction::Move(Action::East))
            .expect("blocked move is still a legal attempt");
        assert!(!transition.moved);
        assert_eq!(transition.state.position, Position::new(0, 0));
        assert_eq!(transition.state.elapsed_actions, 2);
        assert_eq!(transition.status, TerminalStatus::ActionLimit);
    }

    #[test]
    fn invalid_specifications_fail_closed() {
        let mismatch = ExploreSpec::new(
            2, 2, vec![Cell::Floor], Position::new(0, 0), Position::new(1, 1), 4, 1, 2,
        );
        assert!(matches!(mismatch, Err(ExploreSpecError::CellCountMismatch { .. })));

        let blocked_start = ExploreSpec::new(
            2, 1, vec![Cell::Wall, Cell::Goal], Position::new(0, 0), Position::new(1, 0), 4, 1, 2,
        );
        assert_eq!(blocked_start, Err(ExploreSpecError::StartBlocked));
    }
}
