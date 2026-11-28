//! Deterministic keys, doors, and prerequisite planning fixture (GameBench F3).
//!
//! Keys are single-use: entering a locked door consumes its matching key and
//! permanently unlocks that door. Blocked attempts consume one action. The
//! exact path search is crate-private evaluator logic.

use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{Action, Position, TerminalStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeySpec {
    pub id: u8,
    pub position: Position,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DoorSpec {
    pub id: u8,
    pub position: Position,
    pub key_id: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrerequisiteSpecError {
    ZeroDimension,
    ZeroActionLimit,
    PositionOutOfBounds(&'static str),
    DuplicateWall,
    DuplicateKeyId,
    DuplicateDoorId,
    DuplicateKeyPosition,
    DuplicateDoorPosition,
    KeyIdOutOfRange(u8),
    DoorIdOutOfRange(u8),
    DoorReferencesUnknownKey(u8),
    WallOverlapsObject(&'static str),
    PlayerOverlapsObject(&'static str),
    GoalOverlapsObject(&'static str),
}

impl Display for PrerequisiteSpecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "board dimensions must be nonzero"),
            Self::ZeroActionLimit => write!(f, "maximum action count must be nonzero"),
            Self::PositionOutOfBounds(name) => write!(f, "{name} is outside the board"),
            Self::DuplicateWall => write!(f, "wall list contains duplicate positions"),
            Self::DuplicateKeyId => write!(f, "key IDs must be unique"),
            Self::DuplicateDoorId => write!(f, "door IDs must be unique"),
            Self::DuplicateKeyPosition => write!(f, "key positions must be unique"),
            Self::DuplicateDoorPosition => write!(f, "door positions must be unique"),
            Self::KeyIdOutOfRange(id) => write!(f, "key ID {id} must be below 64"),
            Self::DoorIdOutOfRange(id) => write!(f, "door ID {id} must be below 64"),
            Self::DoorReferencesUnknownKey(id) => {
                write!(f, "door references unknown key ID {id}")
            }
            Self::WallOverlapsObject(name) => write!(f, "wall overlaps {name}"),
            Self::PlayerOverlapsObject(name) => write!(f, "player overlaps {name}"),
            Self::GoalOverlapsObject(name) => write!(f, "goal overlaps {name}"),
        }
    }
}

impl Error for PrerequisiteSpecError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrerequisiteSpec {
    width: u16,
    height: u16,
    walls: HashSet<Position>,
    keys: HashMap<Position, u8>,
    doors: HashMap<Position, DoorSpec>,
    player_start: Position,
    goal: Position,
    max_actions: u32,
}

impl PrerequisiteSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: u16,
        height: u16,
        walls: Vec<Position>,
        keys: Vec<KeySpec>,
        doors: Vec<DoorSpec>,
        player_start: Position,
        goal: Position,
        max_actions: u32,
    ) -> Result<Self, PrerequisiteSpecError> {
        if width == 0 || height == 0 {
            return Err(PrerequisiteSpecError::ZeroDimension);
        }
        if max_actions == 0 {
            return Err(PrerequisiteSpecError::ZeroActionLimit);
        }

        let walls_set: HashSet<_> = walls.iter().copied().collect();
        if walls_set.len() != walls.len() {
            return Err(PrerequisiteSpecError::DuplicateWall);
        }

        let mut keys_by_position = HashMap::new();
        let mut key_ids = HashSet::new();
        for key in keys {
            if key.id >= 64 {
                return Err(PrerequisiteSpecError::KeyIdOutOfRange(key.id));
            }
            if !key_ids.insert(key.id) {
                return Err(PrerequisiteSpecError::DuplicateKeyId);
            }
            if keys_by_position.insert(key.position, key.id).is_some() {
                return Err(PrerequisiteSpecError::DuplicateKeyPosition);
            }
        }

        let mut doors_by_position = HashMap::new();
        let mut door_ids = HashSet::new();
        for door in doors {
            if door.id >= 64 {
                return Err(PrerequisiteSpecError::DoorIdOutOfRange(door.id));
            }
            if !door_ids.insert(door.id) {
                return Err(PrerequisiteSpecError::DuplicateDoorId);
            }
            if !key_ids.contains(&door.key_id) {
                return Err(PrerequisiteSpecError::DoorReferencesUnknownKey(
                    door.key_id,
                ));
            }
            if doors_by_position.insert(door.position, door).is_some() {
                return Err(PrerequisiteSpecError::DuplicateDoorPosition);
            }
        }

        for (name, position) in [("player", player_start), ("goal", goal)]
            .into_iter()
            .chain(keys_by_position.keys().copied().map(|position| ("key", position)))
            .chain(doors_by_position.keys().copied().map(|position| ("door", position)))
        {
            if position.x >= width || position.y >= height {
                return Err(PrerequisiteSpecError::PositionOutOfBounds(name));
            }
            if walls_set.contains(&position) {
                return Err(PrerequisiteSpecError::WallOverlapsObject(name));
            }
        }

        if keys_by_position.contains_key(&player_start)
            || doors_by_position.contains_key(&player_start)
        {
            return Err(PrerequisiteSpecError::PlayerOverlapsObject("key or door"));
        }
        if keys_by_position.contains_key(&goal) {
            return Err(PrerequisiteSpecError::GoalOverlapsObject("key"));
        }
        if doors_by_position.contains_key(&goal) {
            return Err(PrerequisiteSpecError::GoalOverlapsObject("door"));
        }

        Ok(Self {
            width,
            height,
            walls: walls_set,
            keys: keys_by_position,
            doors: doors_by_position,
            player_start,
            goal,
            max_actions,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrerequisiteState {
    pub player: Position,
    pub key_inventory: u64,
    pub unlocked_doors: u64,
    pub elapsed_actions: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrerequisiteTransition {
    pub state: PrerequisiteState,
    pub moved: bool,
    pub collected_key: Option<u8>,
    pub unlocked_door: Option<u8>,
    pub status: TerminalStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrerequisiteStepError {
    AlreadyTerminal(TerminalStatus),
}

impl Display for PrerequisiteStepError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyTerminal(status) => {
                write!(f, "cannot step after terminal status {status:?}")
            }
        }
    }
}

impl Error for PrerequisiteStepError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrerequisiteGame {
    spec: PrerequisiteSpec,
    state: PrerequisiteState,
    status: TerminalStatus,
}

impl PrerequisiteGame {
    pub fn new(spec: PrerequisiteSpec) -> Self {
        let state = PrerequisiteState {
            player: spec.player_start,
            key_inventory: 0,
            unlocked_doors: 0,
            elapsed_actions: 0,
        };
        let status = if state.player == spec.goal {
            TerminalStatus::Success
        } else {
            TerminalStatus::Running
        };
        Self { spec, state, status }
    }

    pub const fn state(&self) -> PrerequisiteState {
        self.state
    }

    pub const fn status(&self) -> TerminalStatus {
        self.status
    }

    pub fn step(
        &mut self,
        action: Action,
    ) -> Result<PrerequisiteTransition, PrerequisiteStepError> {
        if self.status != TerminalStatus::Running {
            return Err(PrerequisiteStepError::AlreadyTerminal(self.status));
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

fn simulate(
    spec: &PrerequisiteSpec,
    state: PrerequisiteState,
    action: Action,
) -> PrerequisiteTransition {
    let mut next = state;
    let mut moved = false;
    let mut collected_key = None;
    let mut unlocked_door = None;

    if let Some(destination) = adjacent(state.player, action) {
        let in_bounds = destination.x < spec.width && destination.y < spec.height;
        if in_bounds && !spec.walls.contains(&destination) {
            let mut passable = true;
            if let Some(door) = spec.doors.get(&destination) {
                let door_bit = 1_u64 << door.id;
                if next.unlocked_doors & door_bit == 0 {
                    let key_bit = 1_u64 << door.key_id;
                    if next.key_inventory & key_bit == 0 {
                        passable = false;
                    } else {
                        next.key_inventory &= !key_bit;
                        next.unlocked_doors |= door_bit;
                        unlocked_door = Some(door.id);
                    }
                }
            }
            if passable {
                next.player = destination;
                moved = true;
                if let Some(key_id) = spec.keys.get(&destination).copied() {
                    let key_bit = 1_u64 << key_id;
                    if next.key_inventory & key_bit == 0 {
                        next.key_inventory |= key_bit;
                        collected_key = Some(key_id);
                    }
                }
            }
        }
    }

    next.elapsed_actions += 1;
    let status = if next.player == spec.goal {
        TerminalStatus::Success
    } else if next.elapsed_actions >= spec.max_actions {
        TerminalStatus::ActionLimit
    } else {
        TerminalStatus::Running
    };

    PrerequisiteTransition {
        state: next,
        moved,
        collected_key,
        unlocked_door,
        status,
    }
}

/// Crate-private exact shortest-path reference for evaluator use only.
#[allow(dead_code)]
pub(crate) fn optimal_prerequisite_path(
    spec: &PrerequisiteSpec,
) -> Option<Vec<Action>> {
    const ACTIONS: [Action; 4] = [Action::North, Action::East, Action::South, Action::West];

    let start = PrerequisiteState {
        player: spec.player_start,
        key_inventory: 0,
        unlocked_doors: 0,
        elapsed_actions: 0,
    };
    if start.player == spec.goal {
        return Some(Vec::new());
    }

    let mut queue = VecDeque::from([start]);
    let mut seen = HashSet::from([start]);
    let mut parents: HashMap<PrerequisiteState, (PrerequisiteState, Action)> = HashMap::new();
    let mut goal = None;

    while let Some(state) = queue.pop_front() {
        if state.elapsed_actions >= spec.max_actions {
            continue;
        }
        for action in ACTIONS {
            let transition = simulate(spec, state, action);
            let next = transition.state;
            if !transition.moved || !seen.insert(next) {
                continue;
            }
            parents.insert(next, (state, action));
            if transition.status == TerminalStatus::Success {
                goal = Some(next);
                break;
            }
            if transition.status == TerminalStatus::Running {
                queue.push_back(next);
            }
        }
        if goal.is_some() {
            break;
        }
    }

    let mut cursor = goal?;
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

    fn fixture() -> PrerequisiteSpec {
        // P -> K -> D -> G. The key is single-use and unlocks door 0.
        PrerequisiteSpec::new(
            4,
            1,
            Vec::new(),
            vec![KeySpec { id: 0, position: Position::new(1, 0) }],
            vec![DoorSpec { id: 0, position: Position::new(2, 0), key_id: 0 }],
            Position::new(0, 0),
            Position::new(3, 0),
            8,
        )
        .expect("valid prerequisite fixture")
    }

    #[test]
    fn collect_key_then_unlock_door_then_reach_goal() {
        let mut game = PrerequisiteGame::new(fixture());
        let key = game.step(Action::East).expect("key pickup");
        assert_eq!(key.collected_key, Some(0));
        assert_eq!(key.state.key_inventory, 1);

        let door = game.step(Action::East).expect("door unlock");
        assert_eq!(door.unlocked_door, Some(0));
        assert_eq!(door.state.key_inventory, 0);
        assert_eq!(door.state.unlocked_doors, 1);

        let goal = game.step(Action::East).expect("goal entry");
        assert_eq!(goal.status, TerminalStatus::Success);
    }

    #[test]
    fn locked_door_without_key_blocks_movement() {
        let spec = PrerequisiteSpec::new(
            3, 1, Vec::new(), Vec::new(),
            vec![DoorSpec { id: 0, position: Position::new(1, 0), key_id: 0 }],
            Position::new(0, 0), Position::new(2, 0), 5,
        ).expect("valid locked-door control");
        let mut game = PrerequisiteGame::new(spec);
        let transition = game.step(Action::East).expect("blocked move consumes an action");
        assert!(!transition.moved);
        assert_eq!(transition.state.player, Position::new(0, 0));
        assert_eq!(transition.state.elapsed_actions, 1);
    }

    #[test]
    fn already_unlocked_door_is_passable_without_a_second_key() {
        let mut game = PrerequisiteGame::new(fixture());
        game.step(Action::East).expect("collect");
        game.step(Action::East).expect("unlock");
        game.step(Action::West).expect("step back through unlocked door");
        let result = game.step(Action::East).expect("re-enter unlocked door");
        assert!(result.moved);
        assert_eq!(result.unlocked_door, None);
        assert_eq!(result.state.key_inventory, 0);
    }

    #[test]
    fn exact_oracle_finds_key_door_dependency_path() {
        assert_eq!(
            optimal_prerequisite_path(&fixture()),
            Some(vec![Action::East, Action::East, Action::East])
        );
    }

    #[test]
    fn rejects_duplicate_ids_and_unknown_key_references() {
        let duplicate_key_ids = PrerequisiteSpec::new(
            4, 1, Vec::new(),
            vec![
                KeySpec { id: 0, position: Position::new(1, 0) },
                KeySpec { id: 0, position: Position::new(2, 0) },
            ],
            Vec::new(), Position::new(0, 0), Position::new(3, 0), 8,
        );
        assert_eq!(duplicate_key_ids, Err(PrerequisiteSpecError::DuplicateKeyId));

        let unknown_key = PrerequisiteSpec::new(
            3, 1, Vec::new(), Vec::new(),
            vec![DoorSpec { id: 0, position: Position::new(1, 0), key_id: 7 }],
            Position::new(0, 0), Position::new(2, 0), 8,
        );
        assert_eq!(
            unknown_key,
            Err(PrerequisiteSpecError::DoorReferencesUnknownKey(7))
        );
    }
}
