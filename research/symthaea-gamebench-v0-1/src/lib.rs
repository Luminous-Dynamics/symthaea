#![forbid(unsafe_code)]

//! Standalone, deterministic game-environment primitives for Symthaea GameBench.
//!
//! V0.1 implements navigation (F1), Sokoban-like pushing (F2), keys/doors (F3),
//! partial-map exploration (F4), resource crafting (F5), and hidden mechanics (F6).
//! It is a benchmark substrate, not an agent,
//! game-playing result, or production runtime component.

mod baselines;
pub use baselines::{BaselineError, GreedyManhattanCandidate, RandomLegalCandidate};

mod candidate;
pub use candidate::{
    CandidateDecision, CandidateFeedback, CandidateInput, DecisionValidationError,
    FeatureForecast, ForecastMass, ForecastValue, InteractiveCandidate, validate_decision,
};

mod hidden_mechanics;
pub use hidden_mechanics::{
    HiddenAction, HiddenMechanicsError, HiddenMechanicsGame, HiddenMechanicsObservation,
    HiddenMechanicsState, HiddenMechanicsTransition, SwitchEffect,
};

mod partial_map;
pub use partial_map::{
    ExploreAction, ExploreGame, ExploreObservation, ExploreSpec, ExploreSpecError,
    ExploreState, ExploreStepError, ExploreTransition, ObservedCell,
};

mod prerequisites;
pub use prerequisites::{
    DoorSpec, KeySpec, PrerequisiteGame, PrerequisiteSpec, PrerequisiteSpecError,
    PrerequisiteState, PrerequisiteStepError, PrerequisiteTransition,
};

mod resource_chain;
pub use resource_chain::{
    CraftingAction, CraftingGame, CraftingSpec, CraftingSpecError, CraftingState,
    CraftingStepError, CraftingTransition, Ingredient, RecipeSpec, ResourceKind, ResourcePile,
};

mod sokoban;
pub use sokoban::{
    SokobanGame, SokobanSpec, SokobanSpecError, SokobanState, SokobanStepError,
    SokobanTransition,
};

use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Position {
    pub x: u16,
    pub y: u16,
}

impl Position {
    pub const fn new(x: u16, y: u16) -> Self {
        Self { x, y }
    }

    pub fn manhattan_distance(self, other: Self) -> u32 {
        u32::from(self.x.abs_diff(other.x)) + u32::from(self.y.abs_diff(other.y))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cell {
    Wall,
    Floor,
    Goal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecError {
    ZeroDimension,
    AreaOverflow,
    CellCountMismatch { expected: usize, actual: usize },
    StartOutOfBounds,
    GoalOutOfBounds,
    StartBlocked,
    GoalNotMarked,
    ZeroActionLimit,
}

impl Display for SpecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "grid dimensions must be nonzero"),
            Self::AreaOverflow => write!(f, "grid area overflows addressable memory"),
            Self::CellCountMismatch { expected, actual } => {
                write!(f, "grid needs {expected} cells, received {actual}")
            }
            Self::StartOutOfBounds => write!(f, "start position is outside the grid"),
            Self::GoalOutOfBounds => write!(f, "goal position is outside the grid"),
            Self::StartBlocked => write!(f, "start position cannot be a wall"),
            Self::GoalNotMarked => write!(f, "goal position must contain Cell::Goal"),
            Self::ZeroActionLimit => write!(f, "maximum action count must be nonzero"),
        }
    }
}

impl Error for SpecError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridSpec {
    width: u16,
    height: u16,
    cells: Vec<Cell>,
    start: Position,
    goal: Position,
    max_actions: u32,
}

impl GridSpec {
    pub fn new(
        width: u16,
        height: u16,
        cells: Vec<Cell>,
        start: Position,
        goal: Position,
        max_actions: u32,
    ) -> Result<Self, SpecError> {
        if width == 0 || height == 0 {
            return Err(SpecError::ZeroDimension);
        }
        let expected = usize::from(width)
            .checked_mul(usize::from(height))
            .ok_or(SpecError::AreaOverflow)?;
        if cells.len() != expected {
            return Err(SpecError::CellCountMismatch {
                expected,
                actual: cells.len(),
            });
        }
        if start.x >= width || start.y >= height {
            return Err(SpecError::StartOutOfBounds);
        }
        if goal.x >= width || goal.y >= height {
            return Err(SpecError::GoalOutOfBounds);
        }
        if max_actions == 0 {
            return Err(SpecError::ZeroActionLimit);
        }

        let start_index = usize::from(start.y) * usize::from(width) + usize::from(start.x);
        let goal_index = usize::from(goal.y) * usize::from(width) + usize::from(goal.x);
        if cells[start_index] == Cell::Wall {
            return Err(SpecError::StartBlocked);
        }
        if cells[goal_index] != Cell::Goal {
            return Err(SpecError::GoalNotMarked);
        }

        Ok(Self {
            width,
            height,
            cells,
            start,
            goal,
            max_actions,
        })
    }

    pub const fn width(&self) -> u16 {
        self.width
    }

    pub const fn height(&self) -> u16 {
        self.height
    }

    pub const fn start(&self) -> Position {
        self.start
    }

    pub const fn goal(&self) -> Position {
        self.goal
    }

    pub const fn max_actions(&self) -> u32 {
        self.max_actions
    }

    fn cell_at(&self, position: Position) -> Option<Cell> {
        if position.x >= self.width || position.y >= self.height {
            return None;
        }
        let index = usize::from(position.y) * usize::from(self.width) + usize::from(position.x);
        self.cells.get(index).copied()
    }

    fn neighbor(&self, position: Position, action: Action) -> Option<Position> {
        let candidate = match action {
            Action::North => Position::new(position.x, position.y.checked_sub(1)?),
            Action::East => Position::new(position.x.checked_add(1)?, position.y),
            Action::South => Position::new(position.x, position.y.checked_add(1)?),
            Action::West => Position::new(position.x.checked_sub(1)?, position.y),
        };
        if candidate.x >= self.width || candidate.y >= self.height {
            return None;
        }
        (self.cell_at(candidate) != Some(Cell::Wall)).then_some(candidate)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    North,
    East,
    South,
    West,
}

const SEARCH_ACTIONS: [Action; 4] = [Action::North, Action::East, Action::South, Action::West];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct State {
    pub position: Position,
    pub elapsed_actions: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalStatus {
    Running,
    Success,
    ActionLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transition {
    pub state: State,
    pub moved: bool,
    pub status: TerminalStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameError {
    AlreadyTerminal(TerminalStatus),
}

impl Display for GameError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyTerminal(status) => {
                write!(f, "cannot step an episode after terminal status {status:?}")
            }
        }
    }
}

impl Error for GameError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Game {
    spec: GridSpec,
    state: State,
    status: TerminalStatus,
}

impl Game {
    pub fn new(spec: GridSpec) -> Self {
        let state = State {
            position: spec.start,
            elapsed_actions: 0,
        };
        let status = if state.position == spec.goal {
            TerminalStatus::Success
        } else {
            TerminalStatus::Running
        };
        Self {
            spec,
            state,
            status,
        }
    }

    pub const fn state(&self) -> State {
        self.state
    }

    pub const fn status(&self) -> TerminalStatus {
        self.status
    }

    pub fn step(&mut self, action: Action) -> Result<Transition, GameError> {
        if self.status != TerminalStatus::Running {
            return Err(GameError::AlreadyTerminal(self.status));
        }

        let mut moved = false;
        if let Some(next) = self.spec.neighbor(self.state.position, action) {
            self.state.position = next;
            moved = true;
        }

        // Every attempted action consumes budget, including wall collisions.
        self.state.elapsed_actions += 1;
        self.status = if self.state.position == self.spec.goal {
            TerminalStatus::Success
        } else if self.state.elapsed_actions >= self.spec.max_actions {
            TerminalStatus::ActionLimit
        } else {
            TerminalStatus::Running
        };

        Ok(Transition {
            state: self.state,
            moved,
            status: self.status,
        })
    }

    pub fn observe(&self, mode: ObservationMode) -> Observation {
        let mut visible_cells = Vec::new();
        let mut known_goal = None;

        for y in 0..self.spec.height {
            for x in 0..self.spec.width {
                let position = Position::new(x, y);
                let visible = match mode {
                    ObservationMode::Full => true,
                    ObservationMode::ManhattanRadius(radius) => {
                        self.state.position.manhattan_distance(position) <= u32::from(radius)
                    }
                };
                if visible {
                    if let Some(cell) = self.spec.cell_at(position) {
                        visible_cells.push(VisibleCell { position, cell });
                    }
                    if position == self.spec.goal {
                        known_goal = Some(position);
                    }
                }
            }
        }

        Observation {
            width: self.spec.width,
            height: self.spec.height,
            position: self.state.position,
            visible_cells,
            known_goal,
            remaining_actions: self.spec.max_actions - self.state.elapsed_actions,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationMode {
    Full,
    ManhattanRadius(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisibleCell {
    pub position: Position,
    pub cell: Cell,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    pub width: u16,
    pub height: u16,
    pub position: Position,
    pub visible_cells: Vec<VisibleCell>,
    pub known_goal: Option<Position>,
    pub remaining_actions: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    pub sequence: u32,
    pub action: Action,
    pub before: State,
    pub after: State,
    pub moved: bool,
    pub status: TerminalStatus,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Trace {
    events: Vec<Event>,
}

impl Trace {
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    pub fn record(&mut self, game: &mut Game, action: Action) -> Result<Transition, TraceError> {
        let sequence = u32::try_from(self.events.len()).map_err(|_| TraceError::SequenceOverflow)?;
        let before = game.state();
        let transition = game.step(action).map_err(TraceError::Step)?;
        self.events.push(Event {
            sequence,
            action,
            before,
            after: transition.state,
            moved: transition.moved,
            status: transition.status,
        });
        Ok(transition)
    }

    pub fn replay(spec: GridSpec, events: &[Event]) -> Result<Game, ReplayError> {
        let mut game = Game::new(spec);

        for (index, event) in events.iter().enumerate() {
            let expected_sequence =
                u32::try_from(index).map_err(|_| ReplayError::SequenceOverflow)?;
            if event.sequence != expected_sequence {
                return Err(ReplayError::WrongSequence {
                    expected: expected_sequence,
                    actual: event.sequence,
                });
            }
            if event.before != game.state() {
                return Err(ReplayError::BeforeStateMismatch {
                    sequence: event.sequence,
                });
            }

            let transition = game
                .step(event.action)
                .map_err(|error| ReplayError::StepFailed {
                    sequence: event.sequence,
                    error,
                })?;
            if transition.state != event.after
                || transition.moved != event.moved
                || transition.status != event.status
            {
                return Err(ReplayError::TransitionMismatch {
                    sequence: event.sequence,
                });
            }
        }

        Ok(game)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceError {
    SequenceOverflow,
    Step(GameError),
}

impl Display for TraceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SequenceOverflow => write!(f, "trace sequence exceeds u32"),
            Self::Step(error) => write!(f, "cannot record step: {error}"),
        }
    }
}

impl Error for TraceError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    SequenceOverflow,
    WrongSequence { expected: u32, actual: u32 },
    BeforeStateMismatch { sequence: u32 },
    StepFailed { sequence: u32, error: GameError },
    TransitionMismatch { sequence: u32 },
}

impl Display for ReplayError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SequenceOverflow => write!(f, "replay sequence exceeds u32"),
            Self::WrongSequence { expected, actual } => {
                write!(f, "event sequence {actual} does not match {expected}")
            }
            Self::BeforeStateMismatch { sequence } => {
                write!(f, "event {sequence} has an incorrect pre-state")
            }
            Self::StepFailed { sequence, error } => {
                write!(f, "event {sequence} cannot be replayed: {error}")
            }
            Self::TransitionMismatch { sequence } => {
                write!(f, "event {sequence} has an incorrect transition result")
            }
        }
    }
}

impl Error for ReplayError {}

/// Exact breadth-first shortest-path oracle for the deterministic navigation
/// fixture. This is evaluator-only reference logic, not an agent action API.
#[allow(dead_code)]
pub(crate) fn optimal_path(spec: &GridSpec) -> Option<Vec<Action>> {
    let start = State {
        position: spec.start,
        elapsed_actions: 0,
    };
    if start.position == spec.goal {
        return Some(Vec::new());
    }

    let mut queue = VecDeque::from([start]);
    let mut seen = HashSet::from([start]);
    let mut parents: HashMap<State, (State, Action)> = HashMap::new();
    let mut goal_state = None;

    while let Some(state) = queue.pop_front() {
        if state.elapsed_actions >= spec.max_actions {
            continue;
        }

        for action in SEARCH_ACTIONS {
            let Some(position) = spec.neighbor(state.position, action) else {
                continue;
            };
            let next = State {
                position,
                elapsed_actions: state.elapsed_actions + 1,
            };
            if !seen.insert(next) {
                continue;
            }
            parents.insert(next, (state, action));

            if position == spec.goal {
                goal_state = Some(next);
                break;
            }
            queue.push_back(next);
        }

        if goal_state.is_some() {
            break;
        }
    }

    let mut cursor = goal_state?;
    let mut path = Vec::new();
    while cursor != start {
        let (previous, action) = *parents.get(&cursor)?;
        path.push(action);
        cursor = previous;
    }
    path.reverse();
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(max_actions: u32) -> GridSpec {
        // S .# .
        // . . .
        // . # G
        // The start is (0,0), the goal is (2,2), and the exact shortest
        // route is South, East, East, South (four actions).
        let cells = vec![
            Cell::Floor,
            Cell::Wall,
            Cell::Floor,
            Cell::Floor,
            Cell::Floor,
            Cell::Floor,
            Cell::Floor,
            Cell::Wall,
            Cell::Goal,
        ];
        GridSpec::new(
            3,
            3,
            cells,
            Position::new(0, 0),
            Position::new(2, 2),
            max_actions,
        )
        .expect("fixture must be valid")
    }

    #[test]
    fn rejects_zero_dimensions() {
        let result = GridSpec::new(
            0,
            1,
            vec![Cell::Goal],
            Position::new(0, 0),
            Position::new(0, 0),
            1,
        );
        assert_eq!(result, Err(SpecError::ZeroDimension));
    }

    #[test]
    fn rejects_cell_count_mismatch() {
        let result = GridSpec::new(
            2,
            2,
            vec![Cell::Floor, Cell::Goal],
            Position::new(0, 0),
            Position::new(1, 1),
            4,
        );
        assert_eq!(
            result,
            Err(SpecError::CellCountMismatch {
                expected: 4,
                actual: 2
            })
        );
    }

    #[test]
    fn rejects_blocked_start_and_unmarked_goal() {
        let blocked_start = GridSpec::new(
            2,
            1,
            vec![Cell::Wall, Cell::Goal],
            Position::new(0, 0),
            Position::new(1, 0),
            2,
        );
        assert_eq!(blocked_start, Err(SpecError::StartBlocked));

        let unmarked_goal = GridSpec::new(
            2,
            1,
            vec![Cell::Floor, Cell::Floor],
            Position::new(0, 0),
            Position::new(1, 0),
            2,
        );
        assert_eq!(unmarked_goal, Err(SpecError::GoalNotMarked));
    }

    #[test]
    fn exact_bfs_returns_shortest_path_and_game_reaches_goal() {
        let spec = fixture(8);
        let path = optimal_path(&spec).expect("goal is reachable");
        assert_eq!(path.len(), 4);

        let mut game = Game::new(spec);
        let mut trace = Trace::default();
        for action in path {
            trace.record(&mut game, action).expect("legal step");
        }

        assert_eq!(game.status(), TerminalStatus::Success);
        assert_eq!(game.state().position, Position::new(2, 2));
        assert_eq!(game.state().elapsed_actions, 4);

        let replayed = Trace::replay(fixture(8), trace.events()).expect("trace replays");
        assert_eq!(replayed.state(), game.state());
        assert_eq!(replayed.status(), game.status());
    }

    #[test]
    fn wall_collision_consumes_action_without_moving() {
        let mut game = Game::new(fixture(5));
        let transition = game.step(Action::East).expect("step is accepted");
        assert!(!transition.moved);
        assert_eq!(transition.state.position, Position::new(0, 0));
        assert_eq!(transition.state.elapsed_actions, 1);
    }

    #[test]
    fn action_limit_is_terminal_and_fail_closed() {
        let mut game = Game::new(fixture(1));
        let transition = game.step(Action::North).expect("first action is accepted");
        assert_eq!(transition.status, TerminalStatus::ActionLimit);
        assert_eq!(
            game.step(Action::South),
            Err(GameError::AlreadyTerminal(TerminalStatus::ActionLimit))
        );
    }

    #[test]
    fn partial_observation_does_not_reveal_hidden_goal() {
        let game = Game::new(fixture(8));
        let observation = game.observe(ObservationMode::ManhattanRadius(1));
        assert_eq!(observation.position, Position::new(0, 0));
        assert_eq!(observation.visible_cells.len(), 3);
        assert_eq!(observation.known_goal, None);
        assert_eq!(observation.remaining_actions, 8);
    }

    #[test]
    fn full_observation_reveals_complete_map() {
        let game = Game::new(fixture(8));
        let observation = game.observe(ObservationMode::Full);
        assert_eq!(observation.visible_cells.len(), 9);
        assert_eq!(observation.known_goal, Some(Position::new(2, 2)));
    }

    #[test]
    fn replay_rejects_tampered_transition() {
        let spec = fixture(8);
        let mut game = Game::new(spec.clone());
        let mut trace = Trace::default();
        trace.record(&mut game, Action::South).expect("first step");
        trace.record(&mut game, Action::East).expect("second step");

        let mut tampered = trace.events().to_vec();
        tampered[1].after.position = Position::new(2, 1);

        assert_eq!(
            Trace::replay(spec, &tampered),
            Err(ReplayError::TransitionMismatch { sequence: 1 })
        );
    }

    #[test]
    fn replay_rejects_noncontiguous_event_sequence() {
        let spec = fixture(8);
        let mut game = Game::new(spec.clone());
        let mut trace = Trace::default();
        trace.record(&mut game, Action::South).expect("first step");

        let mut tampered = trace.events().to_vec();
        tampered[0].sequence = 2;

        assert_eq!(
            Trace::replay(spec, &tampered),
            Err(ReplayError::WrongSequence {
                expected: 0,
                actual: 2
            })
        );
    }

    #[test]
    fn exact_oracle_reports_unreachable_goal() {
        let cells = vec![
            Cell::Floor,
            Cell::Wall,
            Cell::Goal,
        ];
        let spec = GridSpec::new(
            3,
            1,
            cells,
            Position::new(0, 0),
            Position::new(2, 0),
            10,
        )
        .expect("valid but unsolvable map");
        assert_eq!(optimal_path(&spec), None);
    }

    #[test]
    fn start_at_goal_is_immediately_successful() {
        let spec = GridSpec::new(
            1,
            1,
            vec![Cell::Goal],
            Position::new(0, 0),
            Position::new(0, 0),
            1,
        )
        .expect("valid zero-distance fixture");
        let game = Game::new(spec.clone());
        assert_eq!(game.status(), TerminalStatus::Success);
        assert_eq!(optimal_path(&spec), Some(Vec::new()));
    }
}
