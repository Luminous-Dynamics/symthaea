//! Small deterministic Sokoban-like pushing environment (GameBench F2).
//!
//! This module is intentionally independent of agent logic. The optimal-path
//! search is crate-private and remains an evaluator-side diagnostic.

use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{Action, Position, TerminalStatus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SokobanSpecError {
    ZeroDimension,
    AreaOverflow,
    ZeroActionLimit,
    PositionOutOfBounds(&'static str),
    DuplicateWall,
    WallOverlapsObject(&'static str),
    PlayerOverlapsCrate,
}

impl Display for SokobanSpecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "Sokoban dimensions must be nonzero"),
            Self::AreaOverflow => write!(f, "Sokoban board area overflows addressable memory"),
            Self::ZeroActionLimit => write!(f, "Sokoban maximum action count must be nonzero"),
            Self::PositionOutOfBounds(name) => {
                write!(f, "Sokoban {name} position is outside the board")
            }
            Self::DuplicateWall => write!(f, "Sokoban wall list contains duplicates"),
            Self::WallOverlapsObject(name) => {
                write!(f, "Sokoban wall overlaps {name}")
            }
            Self::PlayerOverlapsCrate => write!(f, "Sokoban player and crate cannot overlap"),
        }
    }
}

impl Error for SokobanSpecError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SokobanSpec {
    width: u16,
    height: u16,
    walls: HashSet<Position>,
    player_start: Position,
    crate_start: Position,
    goal: Position,
    max_actions: u32,
}

impl SokobanSpec {
    pub fn new(
        width: u16,
        height: u16,
        walls: Vec<Position>,
        player_start: Position,
        crate_start: Position,
        goal: Position,
        max_actions: u32,
    ) -> Result<Self, SokobanSpecError> {
        if width == 0 || height == 0 {
            return Err(SokobanSpecError::ZeroDimension);
        }
        usize::from(width)
            .checked_mul(usize::from(height))
            .ok_or(SokobanSpecError::AreaOverflow)?;
        if max_actions == 0 {
            return Err(SokobanSpecError::ZeroActionLimit);
        }

        let walls_set: HashSet<_> = walls.iter().copied().collect();
        if walls_set.len() != walls.len() {
            return Err(SokobanSpecError::DuplicateWall);
        }

        for (name, position) in [
            ("player", player_start),
            ("crate", crate_start),
            ("goal", goal),
        ] {
            if position.x >= width || position.y >= height {
                return Err(SokobanSpecError::PositionOutOfBounds(name));
            }
            if walls_set.contains(&position) {
                return Err(SokobanSpecError::WallOverlapsObject(name));
            }
        }
        if player_start == crate_start {
            return Err(SokobanSpecError::PlayerOverlapsCrate);
        }

        Ok(Self {
            width,
            height,
            walls: walls_set,
            player_start,
            crate_start,
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

    pub const fn goal(&self) -> Position {
        self.goal
    }

    pub const fn max_actions(&self) -> u32 {
        self.max_actions
    }

    pub fn is_wall(&self, position: Position) -> bool {
        self.walls.contains(&position)
    }

    fn in_bounds(&self, position: Position) -> bool {
        position.x < self.width && position.y < self.height
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SokobanState {
    pub player: Position,
    pub crate_position: Position,
    pub elapsed_actions: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SokobanTransition {
    pub state: SokobanState,
    pub player_moved: bool,
    pub crate_pushed: bool,
    pub status: TerminalStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SokobanStepError {
    AlreadyTerminal(TerminalStatus),
}

impl Display for SokobanStepError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyTerminal(status) => {
                write!(f, "cannot step Sokoban after terminal status {status:?}")
            }
        }
    }
}

impl Error for SokobanStepError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SokobanGame {
    spec: SokobanSpec,
    state: SokobanState,
    status: TerminalStatus,
}

impl SokobanGame {
    pub fn new(spec: SokobanSpec) -> Self {
        let state = SokobanState {
            player: spec.player_start,
            crate_position: spec.crate_start,
            elapsed_actions: 0,
        };
        let status = if state.crate_position == spec.goal {
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

    pub const fn state(&self) -> SokobanState {
        self.state
    }

    pub const fn status(&self) -> TerminalStatus {
        self.status
    }

    pub fn step(&mut self, action: Action) -> Result<SokobanTransition, SokobanStepError> {
        if self.status != TerminalStatus::Running {
            return Err(SokobanStepError::AlreadyTerminal(self.status));
        }

        let transition = simulate(&self.spec, self.state, action);
        self.state = transition.state;
        self.status = transition.status;
        Ok(transition)
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

fn simulate(spec: &SokobanSpec, state: SokobanState, action: Action) -> SokobanTransition {
    let mut next_state = state;
    let mut player_moved = false;
    let mut crate_pushed = false;

    if let Some(next_player) = adjacent(state.player, action) {
        if spec.in_bounds(next_player) && !spec.walls.contains(&next_player) {
            if next_player == state.crate_position {
                if let Some(next_crate) = adjacent(state.crate_position, action) {
                    if spec.in_bounds(next_crate) && !spec.walls.contains(&next_crate) {
                        next_state.player = state.crate_position;
                        next_state.crate_position = next_crate;
                        player_moved = true;
                        crate_pushed = true;
                    }
                }
            } else {
                next_state.player = next_player;
                player_moved = true;
            }
        }
    }

    next_state.elapsed_actions += 1;
    let status = if next_state.crate_position == spec.goal {
        TerminalStatus::Success
    } else if next_state.elapsed_actions >= spec.max_actions {
        TerminalStatus::ActionLimit
    } else {
        TerminalStatus::Running
    };

    SokobanTransition {
        state: next_state,
        player_moved,
        crate_pushed,
        status,
    }
}

/// Exact BFS diagnostic for the evaluator on small deterministic fixtures.
/// This function is crate-private so an agent linked against the library
/// cannot call the oracle as part of its policy.
#[allow(dead_code)]
pub(crate) fn optimal_sokoban_path(spec: &SokobanSpec) -> Option<Vec<Action>> {
    const ACTIONS: [Action; 4] = [Action::North, Action::East, Action::South, Action::West];

    let start = SokobanState {
        player: spec.player_start,
        crate_position: spec.crate_start,
        elapsed_actions: 0,
    };
    if start.crate_position == spec.goal {
        return Some(Vec::new());
    }

    let mut queue = VecDeque::from([start]);
    let mut seen = HashSet::from([start]);
    let mut parents: HashMap<SokobanState, (SokobanState, Action)> = HashMap::new();
    let mut goal_state = None;

    while let Some(state) = queue.pop_front() {
        if state.elapsed_actions >= spec.max_actions {
            continue;
        }

        for action in ACTIONS {
            let transition = simulate(spec, state, action);
            let next = transition.state;
            if !transition.player_moved || !seen.insert(next) {
                continue;
            }
            parents.insert(next, (state, action));

            if transition.status == TerminalStatus::Success {
                goal_state = Some(next);
                break;
            }
            if transition.status == TerminalStatus::Running {
                queue.push_back(next);
            }
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

    fn push_fixture() -> SokobanSpec {
        // #####
        // #PBG#
        // #####
        // East pushes the box directly onto the goal.
        let walls = vec![
            Position::new(0, 0),
            Position::new(1, 0),
            Position::new(2, 0),
            Position::new(3, 0),
            Position::new(4, 0),
            Position::new(0, 1),
            Position::new(4, 1),
            Position::new(0, 2),
            Position::new(1, 2),
            Position::new(2, 2),
            Position::new(3, 2),
            Position::new(4, 2),
        ];
        SokobanSpec::new(
            5,
            3,
            walls,
            Position::new(1, 1),
            Position::new(2, 1),
            Position::new(3, 1),
            8,
        )
        .expect("valid push fixture")
    }

    #[test]
    fn pushes_box_and_reaches_goal() {
        let mut game = SokobanGame::new(push_fixture());
        let transition = game.step(Action::East).expect("push is a valid action");
        assert!(transition.player_moved);
        assert!(transition.crate_pushed);
        assert_eq!(transition.state.player, Position::new(2, 1));
        assert_eq!(transition.state.crate_position, Position::new(3, 1));
        assert_eq!(transition.status, TerminalStatus::Success);
    }

    #[test]
    fn blocked_push_does_not_move_player_or_crate() {
        let walls = vec![
            Position::new(0, 0),
            Position::new(1, 0),
            Position::new(2, 0),
            Position::new(3, 0),
            Position::new(4, 0),
            Position::new(0, 1),
            Position::new(4, 1),
            Position::new(0, 2),
            Position::new(1, 2),
            Position::new(2, 2),
            Position::new(3, 2),
            Position::new(4, 2),
        ];
        let spec = SokobanSpec::new(
            5,
            3,
            walls,
            Position::new(2, 1),
            Position::new(3, 1),
            Position::new(1, 1),
            4,
        )
        .expect("valid blocked-push fixture");
        let mut game = SokobanGame::new(spec);
        let transition = game.step(Action::East).expect("blocked push still consumes action");
        assert!(!transition.player_moved);
        assert!(!transition.crate_pushed);
        assert_eq!(transition.state.player, Position::new(2, 1));
        assert_eq!(transition.state.crate_position, Position::new(3, 1));
        assert_eq!(transition.state.elapsed_actions, 1);
    }

    #[test]
    fn exact_oracle_finds_one_push_solution() {
        let spec = push_fixture();
        assert_eq!(optimal_sokoban_path(&spec), Some(vec![Action::East]));
    }

    #[test]
    fn exact_oracle_detects_wall_trapped_box() {
        let walls = vec![
            Position::new(0, 0),
            Position::new(1, 0),
            Position::new(2, 0),
            Position::new(3, 0),
            Position::new(4, 0),
            Position::new(0, 1),
            Position::new(4, 1),
            Position::new(0, 2),
            Position::new(1, 2),
            Position::new(2, 2),
            Position::new(3, 2),
            Position::new(4, 2),
        ];
        let spec = SokobanSpec::new(
            5,
            3,
            walls,
            Position::new(2, 1),
            Position::new(3, 1),
            Position::new(1, 1),
            12,
        )
        .expect("valid but unsolvable fixture");
        assert_eq!(optimal_sokoban_path(&spec), None);
    }

    #[test]
    fn rejects_player_crate_overlap_and_duplicate_walls() {
        let overlap = SokobanSpec::new(
            3,
            3,
            Vec::new(),
            Position::new(1, 1),
            Position::new(1, 1),
            Position::new(2, 2),
            4,
        );
        assert_eq!(overlap, Err(SokobanSpecError::PlayerOverlapsCrate));

        let duplicate = SokobanSpec::new(
            3,
            3,
            vec![Position::new(0, 0), Position::new(0, 0)],
            Position::new(1, 1),
            Position::new(1, 2),
            Position::new(2, 2),
            4,
        );
        assert_eq!(duplicate, Err(SokobanSpecError::DuplicateWall));
    }
}
