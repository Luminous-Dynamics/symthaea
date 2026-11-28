//! Partially observable map and active scan fixture (GameBench F4).
//!
//! Observations contain only the current field of view. The environment does
//! not carry the agent's observation history; candidate-side memory must do so.
//! A scan consumes one action and temporarily widens the current observation.

use std::collections::{HashMap, HashSet};
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
    #[allow(clippy::too_many_arguments)]
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
    pub width: u16,
    pub height: u16,
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
            width: self.spec.width,
            height: self.spec.height,
            position: self.state.position,
            visible_cells,
            remaining_actions: self.spec.max_actions - self.state.elapsed_actions,
            scan_was_used_for_this_observation: scan_was_used,
        }
    }
}

/// Candidate-side map memory. It stores only cells that arrived in observations;
/// unknown locations remain absent and are never converted to floor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExploreMemory {
    width: u16,
    height: u16,
    known_cells: HashMap<Position, Cell>,
    visited_positions: HashSet<Position>,
    observation_count: u64,
    current_position: Option<Position>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExploreMemoryError {
    ZeroDimension,
    GeometryMismatch {
        expected_width: u16,
        expected_height: u16,
        actual_width: u16,
        actual_height: u16,
    },
    PositionOutOfBounds(&'static str),
    AgentPositionNotObserved,
    DuplicateObservationCell(Position),
    ConflictingObservation {
        position: Position,
        previous: Cell,
        observed: Cell,
    },
    ObservationCountOverflow,
}

impl Display for ExploreMemoryError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "map-memory dimensions must be nonzero"),
            Self::GeometryMismatch {
                expected_width,
                expected_height,
                actual_width,
                actual_height,
            } => write!(
                f,
                "observation geometry {actual_width}x{actual_height} does not match memory {expected_width}x{expected_height}"
            ),
            Self::PositionOutOfBounds(name) => {
                write!(f, "observed {name} position is outside remembered map")
            }
            Self::AgentPositionNotObserved => write!(f, "current agent position is absent from its observation"),
            Self::DuplicateObservationCell(position) => write!(f, "observation contains duplicate cell ({}, {})", position.x, position.y),
            Self::ConflictingObservation {
                position,
                previous,
                observed,
            } => write!(
                f,
                "conflicting cell at ({}, {}): remembered {previous:?}, observed {observed:?}",
                position.x, position.y
            ),
            Self::ObservationCountOverflow => write!(f, "map-memory observation counter overflowed"),
        }
    }
}

impl Error for ExploreMemoryError {}

impl ExploreMemory {
    pub fn new(width: u16, height: u16) -> Result<Self, ExploreMemoryError> {
        if width == 0 || height == 0 {
            return Err(ExploreMemoryError::ZeroDimension);
        }
        Ok(Self {
            width,
            height,
            known_cells: HashMap::new(),
            visited_positions: HashSet::new(),
            observation_count: 0,
            current_position: None,
        })
    }

    pub fn reset(&mut self) {
        self.known_cells.clear();
        self.visited_positions.clear();
        self.observation_count = 0;
        self.current_position = None;
    }

    /// Merge a candidate-visible observation into episodic map memory.
    /// Contradictory observations fail closed rather than silently replacing
    /// prior knowledge; a future belief-state variant can model uncertainty.
    pub fn remember(
        &mut self,
        observation: &ExploreObservation,
    ) -> Result<(), ExploreMemoryError> {
        if observation.width != self.width || observation.height != self.height {
            return Err(ExploreMemoryError::GeometryMismatch {
                expected_width: self.width,
                expected_height: self.height,
                actual_width: observation.width,
                actual_height: observation.height,
            });
        }

        if observation.position.x >= self.width || observation.position.y >= self.height {
            return Err(ExploreMemoryError::PositionOutOfBounds("agent"));
        }
        if !observation
            .visible_cells
            .iter()
            .any(|cell| cell.position == observation.position)
        {
            return Err(ExploreMemoryError::AgentPositionNotObserved);
        }

        let mut current_observation = HashMap::new();
        for visible in &observation.visible_cells {
            if visible.position.x >= self.width || visible.position.y >= self.height {
                return Err(ExploreMemoryError::PositionOutOfBounds("cell"));
            }
            if current_observation.insert(visible.position, visible.cell).is_some() {
                return Err(ExploreMemoryError::DuplicateObservationCell(visible.position));
            }
            if let Some(previous) = self.known_cells.get(&visible.position) {
                if *previous != visible.cell {
                    return Err(ExploreMemoryError::ConflictingObservation {
                        position: visible.position,
                        previous: *previous,
                        observed: visible.cell,
                    });
                }
            }
        }

        if current_observation.get(&observation.position) == Some(&Cell::Wall) {
            return Err(ExploreMemoryError::AgentPositionNotObserved);
        }

        let next_count = self
            .observation_count
            .checked_add(1)
            .ok_or(ExploreMemoryError::ObservationCountOverflow)?;

        for visible in &observation.visible_cells {
            self.known_cells.insert(visible.position, visible.cell);
        }
        self.visited_positions.insert(observation.position);
        self.current_position = Some(observation.position);
        self.observation_count = next_count;
        Ok(())
    }

    pub fn known_cell(&self, position: Position) -> Option<Cell> {
        self.known_cells.get(&position).copied()
    }

    pub fn known_cells(&self) -> &HashMap<Position, Cell> {
        &self.known_cells
    }

    pub fn visited_positions(&self) -> &HashSet<Position> {
        &self.visited_positions
    }

    pub const fn current_position(&self) -> Option<Position> {
        self.current_position
    }

    pub const fn observation_count(&self) -> u64 {
        self.observation_count
    }

    pub fn coverage(&self) -> f64 {
        let total = usize::from(self.width) * usize::from(self.height);
        self.known_cells.len() as f64 / total as f64
    }

    /// Known traversable cells adjacent to at least one unknown in-bounds
    /// cell. These are candidate exploration frontiers, not guaranteed routes.
    pub fn frontier_positions(&self) -> Vec<Position> {
        let mut frontier = Vec::new();
        for (position, cell) in &self.known_cells {
            if *cell == Cell::Wall {
                continue;
            }
            let neighbors = [
                position.y.checked_sub(1).map(|y| Position::new(position.x, y)),
                position.x.checked_add(1).map(|x| Position::new(x, position.y)),
                position.y.checked_add(1).map(|y| Position::new(position.x, y)),
                position.x.checked_sub(1).map(|x| Position::new(x, position.y)),
            ];
            let touches_unknown = neighbors.into_iter().flatten().any(|neighbor| {
                neighbor.x < self.width && neighbor.y < self.height
                    && !self.known_cells.contains_key(&neighbor)
            });
            if touches_unknown {
                frontier.push(*position);
            }
        }
        frontier.sort_unstable();
        frontier
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
        // The direct middle corridor is blocked at (3,1), but a lower
        // route remains available. The distant goal is not initially visible.
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
        .expect("valid partially observable fixture")
    }

    #[test]
    fn initial_observation_does_not_disclose_distant_wall_or_goal() {
        let game = ExploreGame::new(fixture(12));
        let observation = game.observe();
        assert_eq!(observation.position, Position::new(0, 1));
        assert_eq!(observation.visible_cells.len(), 4);
        assert!(!observation.visible_cells.iter().any(|cell| {
            cell.position == Position::new(3, 1) || cell.position == Position::new(6, 1)
        }));
    }

    #[test]
    fn scan_costs_one_action_and_reveals_only_local_extended_view() {
        let mut game = ExploreGame::new(fixture(12));
        let transition = game.step(ExploreAction::Scan).expect("scan consumes a turn");
        assert!(transition.scanned);
        assert!(!transition.moved);
        assert_eq!(transition.state.elapsed_actions, 1);
        assert_eq!(transition.observation.visible_cells.len(), 13);
        assert!(transition.observation.visible_cells.iter().any(|cell| {
            cell.position == Position::new(3, 1) && cell.cell == Cell::Wall
        }));
        assert!(!transition.observation.visible_cells.iter().any(|cell| {
            cell.position == Position::new(6, 1)
        }));
    }

    #[test]
    fn observations_are_current_views_not_accumulated_memory() {
        let mut game = ExploreGame::new(fixture(12));
        let scan = game.step(ExploreAction::Scan).expect("scan");
        assert!(scan.observation.visible_cells.iter().any(|cell| {
            cell.position == Position::new(0, 1)
        }));

        game.step(ExploreAction::Move(Action::East)).expect("move to x=1");
        let moved = game.step(ExploreAction::Move(Action::East)).expect("move to x=2");
        assert_eq!(moved.state.position, Position::new(2, 0));
        assert!(!moved.observation.visible_cells.iter().any(|cell| {
            cell.position == Position::new(0, 0)
        }));
        // The environment never attaches the earlier scan result to this view.
        assert!(!moved.observation.scan_was_used_for_this_observation);
    }

    #[test]
    fn wall_collision_consumes_action_and_preserves_position() {
        let mut game = ExploreGame::new(fixture(2));
        game.step(ExploreAction::Scan).expect("scan");
        let transition = game
            .step(ExploreAction::Move(Action::West))
            .expect("blocked border move is still a legal attempt");
        assert!(!transition.moved);
        assert_eq!(transition.state.position, Position::new(0, 1));
        assert_eq!(transition.state.elapsed_actions, 2);
        assert_eq!(transition.status, TerminalStatus::ActionLimit);
    }

    fn one_cell_observation(
        position: Position,
        cell: Cell,
        width: u16,
        height: u16,
    ) -> ExploreObservation {
        ExploreObservation {
            width,
            height,
            position,
            visible_cells: vec![ObservedCell { position, cell }],
            remaining_actions: 5,
            scan_was_used_for_this_observation: false,
        }
    }

    #[test]
    fn candidate_memory_accumulates_cells_that_environment_views_discard() {
        let mut game = ExploreGame::new(fixture(12));
        let mut memory = ExploreMemory::new(7, 1).expect("memory");
        let first = game.observe();
        memory.remember(&first).expect("remember first view");
        let after_scan = game.step(ExploreAction::Scan).expect("scan");
        memory.remember(&after_scan.observation).expect("remember scan view");
        game.step(ExploreAction::Move(Action::East)).expect("move to x=1");
        game.step(ExploreAction::Move(Action::East)).expect("move to x=2");
        let later = game.observe();
        memory.remember(&later).expect("remember later view");

        assert!(memory.known_cell(Position::new(0, 0)).is_some());
        assert!(memory.known_cell(Position::new(3, 1)).is_some());
        assert!(memory.known_cell(Position::new(6, 1)).is_none());
        assert!(later.visible_cells.iter().all(|cell| cell.position.x != 0));
        assert_eq!(memory.observation_count(), 3);
        assert!(memory.coverage() > 0.0);
    }

    #[test]
    fn unknown_cells_remain_unknown_not_implicitly_floor() {
        let mut memory = ExploreMemory::new(7, 1).expect("memory");
        memory
            .remember(&one_cell_observation(Position::new(0, 0), Cell::Floor, 7, 1))
            .expect("remember visible cell");
        assert_eq!(memory.known_cell(Position::new(1, 0)), None);
        assert_eq!(memory.known_cell(Position::new(0, 0)), Some(Cell::Floor));
    }

    #[test]
    fn map_memory_rejects_conflicting_reobservation() {
        let mut memory = ExploreMemory::new(3, 1).expect("memory");
        memory
            .remember(&one_cell_observation(Position::new(1, 0), Cell::Wall, 3, 1))
            .expect("remember wall");
        assert_eq!(
            memory.remember(&one_cell_observation(
                Position::new(1, 0),
                Cell::Floor,
                3,
                1
            )),
            Err(ExploreMemoryError::ConflictingObservation {
                position: Position::new(1, 0),
                previous: Cell::Wall,
                observed: Cell::Floor,
            })
        );
        assert_eq!(memory.known_cell(Position::new(1, 0)), Some(Cell::Wall));
    }

    #[test]
    fn map_memory_rejects_geometry_mismatch_and_out_of_bounds_cells() {
        let mut memory = ExploreMemory::new(3, 1).expect("memory");
        assert!(matches!(
            memory.remember(&one_cell_observation(
                Position::new(0, 0),
                Cell::Floor,
                4,
                1
            )),
            Err(ExploreMemoryError::GeometryMismatch { .. })
        ));
        assert_eq!(
            memory.remember(&one_cell_observation(
                Position::new(3, 0),
                Cell::Floor,
                3,
                1
            )),
            Err(ExploreMemoryError::PositionOutOfBounds("agent"))
        );
    }

    #[test]
    fn map_memory_rejects_duplicate_cells_and_missing_agent_position() {
        let mut memory = ExploreMemory::new(3, 1).expect("memory");
        let duplicate = ExploreObservation {
            width: 3,
            height: 1,
            position: Position::new(0, 0),
            visible_cells: vec![
                ObservedCell { position: Position::new(0, 0), cell: Cell::Floor },
                ObservedCell { position: Position::new(0, 0), cell: Cell::Wall },
            ],
            remaining_actions: 4,
            scan_was_used_for_this_observation: false,
        };
        assert_eq!(
            memory.remember(&duplicate),
            Err(ExploreMemoryError::DuplicateObservationCell(Position::new(0, 0)))
        );

        let missing_agent = ExploreObservation {
            width: 3,
            height: 1,
            position: Position::new(0, 0),
            visible_cells: vec![ObservedCell {
                position: Position::new(1, 0),
                cell: Cell::Floor,
            }],
            remaining_actions: 4,
            scan_was_used_for_this_observation: false,
        };
        assert_eq!(
            memory.remember(&missing_agent),
            Err(ExploreMemoryError::AgentPositionNotObserved)
        );
        assert!(memory.known_cells().is_empty());
    }

    #[test]
    fn map_memory_frontiers_are_known_traversable_cells_adjacent_to_unknown() {
        let mut memory = ExploreMemory::new(3, 1).expect("memory");
        memory
            .remember(&ExploreObservation {
                width: 3,
                height: 1,
                position: Position::new(0, 0),
                visible_cells: vec![
                    ObservedCell { position: Position::new(0, 0), cell: Cell::Floor },
                    ObservedCell { position: Position::new(1, 0), cell: Cell::Wall },
                ],
                remaining_actions: 4,
                scan_was_used_for_this_observation: false,
            })
            .expect("remember two cells");
        assert_eq!(memory.frontier_positions(), vec![Position::new(0, 0)]);
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
