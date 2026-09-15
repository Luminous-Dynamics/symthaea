// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical ARC-AGI-3 observation/action contract.
//!
//! This module is deliberately narrow. It normalizes the public ARC-AGI-3 wire
//! representation into deterministic semantic types while keeping provenance
//! (`game_id`, `guid`, client-supplied reasoning) out of policy-authoritative
//! canonical bytes.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// ARC-AGI-3 game lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Arc3GameState {
    NotPlayed,
    NotFinished,
    Win,
    GameOver,
}

/// Standard ARC-AGI-3 action identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum Arc3ActionKind {
    Reset = 0,
    Action1 = 1,
    Action2 = 2,
    Action3 = 3,
    Action4 = 4,
    Action5 = 5,
    Action6 = 6,
    Action7 = 7,
}

impl Arc3ActionKind {
    /// Convert an official numeric action id to a typed action.
    pub fn from_id(id: u8) -> Result<Self, Arc3ProtocolError> {
        match id {
            0 => Ok(Self::Reset),
            1 => Ok(Self::Action1),
            2 => Ok(Self::Action2),
            3 => Ok(Self::Action3),
            4 => Ok(Self::Action4),
            5 => Ok(Self::Action5),
            6 => Ok(Self::Action6),
            7 => Ok(Self::Action7),
            other => Err(Arc3ProtocolError::InvalidActionId(other.to_string())),
        }
    }

    /// Numeric id used by the official ARC-AGI-3 protocol.
    pub const fn id(self) -> u8 {
        self as u8
    }

    /// Whether this action requires `(x, y)` coordinates.
    pub const fn requires_coordinates(self) -> bool {
        matches!(self, Self::Action6)
    }
}

/// Canonical action used by Symthaea cognition and evidence receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Arc3Action {
    pub kind: Arc3ActionKind,
    /// Coordinate payload for `ACTION6`; absent for all simple actions.
    pub coordinates: Option<Arc3Coordinates>,
}

impl Arc3Action {
    pub fn simple(kind: Arc3ActionKind) -> Result<Self, Arc3ProtocolError> {
        if kind.requires_coordinates() {
            return Err(Arc3ProtocolError::MissingCoordinates);
        }
        Ok(Self {
            kind,
            coordinates: None,
        })
    }

    pub fn action6(x: u8, y: u8) -> Result<Self, Arc3ProtocolError> {
        if x > 63 || y > 63 {
            return Err(Arc3ProtocolError::CoordinateOutOfRange { x, y });
        }
        Ok(Self {
            kind: Arc3ActionKind::Action6,
            coordinates: Some(Arc3Coordinates { x, y }),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Arc3Coordinates {
    pub x: u8,
    pub y: u8,
}

/// Raw action metadata accepted from official API/recording JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct Arc3WireActionInput {
    pub id: Value,
    #[serde(default)]
    pub data: BTreeMap<String, Value>,
    #[serde(default)]
    pub reasoning: Option<Value>,
}

/// Raw observation accepted from the official ARC-AGI-3 JSON shape.
#[derive(Debug, Clone, Deserialize)]
pub struct Arc3WireObservation {
    #[serde(default)]
    pub game_id: String,
    pub frame: Vec<Vec<Vec<u8>>>,
    pub state: Arc3GameState,
    #[serde(default)]
    pub levels_completed: u16,
    #[serde(default)]
    pub win_levels: u16,
    #[serde(default)]
    pub action_input: Option<Arc3WireActionInput>,
    #[serde(default)]
    pub guid: Option<String>,
    #[serde(default)]
    pub full_reset: bool,
    #[serde(default)]
    pub available_actions: Vec<u8>,
}

/// Provenance retained for evidence bookkeeping but excluded from semantic bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arc3Provenance {
    pub game_id: String,
    pub guid: Option<String>,
}

/// Policy-authoritative observation semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Arc3Observation {
    pub frame: Vec<Vec<Vec<u8>>>,
    pub state: Arc3GameState,
    pub levels_completed: u16,
    pub win_levels: u16,
    pub last_action: Option<Arc3Action>,
    pub full_reset: bool,
    /// Sorted, duplicate-free official action ids.
    pub available_actions: Vec<Arc3ActionKind>,
}

/// Validated observation plus non-authoritative provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arc3Envelope {
    pub semantic: Arc3Observation,
    pub provenance: Arc3Provenance,
}

impl Arc3Envelope {
    /// Parse and validate the official JSON representation.
    pub fn from_json(json: &str) -> Result<Self, Arc3ProtocolError> {
        let wire: Arc3WireObservation = serde_json::from_str(json)
            .map_err(|error| Arc3ProtocolError::Json(error.to_string()))?;
        Self::try_from(wire)
    }

    /// Deterministic semantic bytes used for evidence commitments.
    pub fn semantic_bytes(&self) -> Result<Vec<u8>, Arc3ProtocolError> {
        serde_json::to_vec(&self.semantic)
            .map_err(|error| Arc3ProtocolError::Json(error.to_string()))
    }

    /// BLAKE3 commitment over semantic bytes only.
    pub fn semantic_hash(&self) -> Result<blake3::Hash, Arc3ProtocolError> {
        Ok(blake3::hash(&self.semantic_bytes()?))
    }
}

impl TryFrom<Arc3WireObservation> for Arc3Envelope {
    type Error = Arc3ProtocolError;

    fn try_from(wire: Arc3WireObservation) -> Result<Self, Self::Error> {
        validate_frame(&wire.frame)?;

        if wire.levels_completed > 254 {
            return Err(Arc3ProtocolError::LevelCountOutOfRange(
                wire.levels_completed,
            ));
        }
        if wire.win_levels > 254 {
            return Err(Arc3ProtocolError::LevelCountOutOfRange(wire.win_levels));
        }

        let last_action = wire
            .action_input
            .as_ref()
            .map(normalize_action_input)
            .transpose()?;

        let mut unique_actions = BTreeSet::new();
        for id in wire.available_actions {
            let action = Arc3ActionKind::from_id(id)?;
            if !unique_actions.insert(action) {
                return Err(Arc3ProtocolError::DuplicateAvailableAction(id));
            }
        }

        Ok(Self {
            semantic: Arc3Observation {
                frame: wire.frame,
                state: wire.state,
                levels_completed: wire.levels_completed,
                win_levels: wire.win_levels,
                last_action,
                full_reset: wire.full_reset,
                available_actions: unique_actions.into_iter().collect(),
            },
            provenance: Arc3Provenance {
                game_id: wire.game_id,
                guid: wire.guid,
            },
        })
    }
}

fn normalize_action_input(input: &Arc3WireActionInput) -> Result<Arc3Action, Arc3ProtocolError> {
    let kind = parse_action_kind(&input.id)?;

    for key in input.data.keys() {
        let allowed = key == "game_id" || (kind == Arc3ActionKind::Action6 && (key == "x" || key == "y"));
        if !allowed {
            return Err(Arc3ProtocolError::UnexpectedActionField(key.clone()));
        }
    }

    if kind.requires_coordinates() {
        let x = parse_coordinate(input.data.get("x"), "x")?;
        let y = parse_coordinate(input.data.get("y"), "y")?;
        Arc3Action::action6(x, y)
    } else {
        if input.data.contains_key("x") || input.data.contains_key("y") {
            return Err(Arc3ProtocolError::UnexpectedCoordinates(kind));
        }
        Arc3Action::simple(kind)
    }
}

fn parse_action_kind(value: &Value) -> Result<Arc3ActionKind, Arc3ProtocolError> {
    if let Some(id) = value.as_u64() {
        let id = u8::try_from(id)
            .map_err(|_| Arc3ProtocolError::InvalidActionId(id.to_string()))?;
        return Arc3ActionKind::from_id(id);
    }

    if let Some(name) = value.as_str() {
        return match name.to_ascii_uppercase().as_str() {
            "RESET" => Ok(Arc3ActionKind::Reset),
            "ACTION1" => Ok(Arc3ActionKind::Action1),
            "ACTION2" => Ok(Arc3ActionKind::Action2),
            "ACTION3" => Ok(Arc3ActionKind::Action3),
            "ACTION4" => Ok(Arc3ActionKind::Action4),
            "ACTION5" => Ok(Arc3ActionKind::Action5),
            "ACTION6" => Ok(Arc3ActionKind::Action6),
            "ACTION7" => Ok(Arc3ActionKind::Action7),
            other => Err(Arc3ProtocolError::InvalidActionId(other.to_owned())),
        };
    }

    Err(Arc3ProtocolError::InvalidActionId(value.to_string()))
}

fn parse_coordinate(value: Option<&Value>, field: &'static str) -> Result<u8, Arc3ProtocolError> {
    let value = value.ok_or(Arc3ProtocolError::MissingActionField(field))?;
    let raw = value
        .as_u64()
        .ok_or(Arc3ProtocolError::InvalidActionField(field))?;
    let coordinate = u8::try_from(raw).map_err(|_| Arc3ProtocolError::InvalidActionField(field))?;
    if coordinate > 63 {
        return Err(Arc3ProtocolError::InvalidActionField(field));
    }
    Ok(coordinate)
}

fn validate_frame(frame: &[Vec<Vec<u8>>]) -> Result<(), Arc3ProtocolError> {
    if frame.is_empty() {
        return Err(Arc3ProtocolError::EmptyFrame);
    }

    for (frame_index, grid) in frame.iter().enumerate() {
        if grid.is_empty() || grid.len() > 64 {
            return Err(Arc3ProtocolError::InvalidGridDimensions { frame_index });
        }
        let width = grid[0].len();
        if width == 0 || width > 64 || grid.iter().any(|row| row.len() != width) {
            return Err(Arc3ProtocolError::InvalidGridDimensions { frame_index });
        }
        for (y, row) in grid.iter().enumerate() {
            for (x, &cell) in row.iter().enumerate() {
                if cell > 15 {
                    return Err(Arc3ProtocolError::InvalidCell {
                        frame_index,
                        x,
                        y,
                        value: cell,
                    });
                }
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Arc3ProtocolError {
    Json(String),
    EmptyFrame,
    InvalidGridDimensions {
        frame_index: usize,
    },
    InvalidCell {
        frame_index: usize,
        x: usize,
        y: usize,
        value: u8,
    },
    InvalidActionId(String),
    DuplicateAvailableAction(u8),
    MissingCoordinates,
    CoordinateOutOfRange {
        x: u8,
        y: u8,
    },
    MissingActionField(&'static str),
    InvalidActionField(&'static str),
    UnexpectedActionField(String),
    UnexpectedCoordinates(Arc3ActionKind),
    LevelCountOutOfRange(u16),
}

impl fmt::Display for Arc3ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(f, "invalid ARC-3 JSON: {error}"),
            Self::EmptyFrame => write!(f, "ARC-3 observation contains no frame grids"),
            Self::InvalidGridDimensions { frame_index } => {
                write!(f, "invalid ARC-3 grid dimensions at frame {frame_index}")
            }
            Self::InvalidCell {
                frame_index,
                x,
                y,
                value,
            } => write!(
                f,
                "invalid ARC-3 cell value {value} at frame {frame_index} ({x},{y})"
            ),
            Self::InvalidActionId(id) => write!(f, "invalid ARC-3 action id {id}"),
            Self::DuplicateAvailableAction(id) => {
                write!(f, "duplicate ARC-3 available action id {id}")
            }
            Self::MissingCoordinates => write!(f, "ACTION6 requires coordinates"),
            Self::CoordinateOutOfRange { x, y } => {
                write!(f, "ACTION6 coordinates out of range: ({x},{y})")
            }
            Self::MissingActionField(field) => write!(f, "missing ARC-3 action field {field}"),
            Self::InvalidActionField(field) => write!(f, "invalid ARC-3 action field {field}"),
            Self::UnexpectedActionField(field) => {
                write!(f, "unexpected ARC-3 action field {field}")
            }
            Self::UnexpectedCoordinates(kind) => {
                write!(f, "unexpected coordinates for ARC-3 action {kind:?}")
            }
            Self::LevelCountOutOfRange(value) => {
                write!(f, "ARC-3 level count out of range: {value}")
            }
        }
    }
}

impl std::error::Error for Arc3ProtocolError {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn observation_json(game_id: &str, guid: &str, reasoning: Value) -> String {
        json!({
            "game_id": game_id,
            "state": "NOT_FINISHED",
            "levels_completed": 1,
            "win_levels": 7,
            "action_input": {
                "id": "ACTION1",
                "data": {},
                "reasoning": reasoning
            },
            "guid": guid,
            "full_reset": false,
            "available_actions": [6, 1, 4, 2],
            "frame": [
                [[0, 1, 0], [2, 3, 2]],
                [[0, 1, 0], [2, 4, 2]]
            ]
        })
        .to_string()
    }

    #[test]
    fn parses_official_recording_shape() {
        let envelope = Arc3Envelope::from_json(&observation_json(
            "ls20-016295f7601e",
            "episode-guid",
            json!({"thought": "move up"}),
        ))
        .unwrap();

        assert_eq!(envelope.semantic.state, Arc3GameState::NotFinished);
        assert_eq!(envelope.semantic.levels_completed, 1);
        assert_eq!(envelope.semantic.frame.len(), 2);
        assert_eq!(
            envelope.semantic.available_actions,
            vec![
                Arc3ActionKind::Action1,
                Arc3ActionKind::Action2,
                Arc3ActionKind::Action4,
                Arc3ActionKind::Action6,
            ]
        );
        assert_eq!(
            envelope.semantic.last_action,
            Some(Arc3Action {
                kind: Arc3ActionKind::Action1,
                coordinates: None,
            })
        );
    }

    #[test]
    fn provenance_and_reasoning_do_not_change_semantic_commitment() {
        let a = Arc3Envelope::from_json(&observation_json(
            "game-a-v1",
            "guid-a",
            json!({"private_note": "alpha"}),
        ))
        .unwrap();
        let b = Arc3Envelope::from_json(&observation_json(
            "game-b-v99",
            "guid-b",
            json!({"private_note": "beta"}),
        ))
        .unwrap();

        assert_ne!(a.provenance, b.provenance);
        assert_eq!(a.semantic, b.semantic);
        assert_eq!(a.semantic_hash().unwrap(), b.semantic_hash().unwrap());
    }

    #[test]
    fn semantic_change_changes_commitment() {
        let a = Arc3Envelope::from_json(&observation_json(
            "game-a-v1",
            "guid-a",
            Value::Null,
        ))
        .unwrap();

        let mut changed: Value = serde_json::from_str(&observation_json(
            "game-a-v1",
            "guid-a",
            Value::Null,
        ))
        .unwrap();
        changed["frame"][1][1][1] = json!(5);
        let b = Arc3Envelope::from_json(&changed.to_string()).unwrap();

        assert_ne!(a.semantic_hash().unwrap(), b.semantic_hash().unwrap());
    }

    #[test]
    fn symbolic_and_numeric_action_ids_normalize_identically() {
        let symbolic = Arc3Envelope::from_json(&observation_json(
            "game-a",
            "guid-a",
            Value::Null,
        ))
        .unwrap();

        let mut numeric_json: Value = serde_json::from_str(&observation_json(
            "game-a",
            "guid-a",
            Value::Null,
        ))
        .unwrap();
        numeric_json["action_input"]["id"] = json!(1);
        let numeric = Arc3Envelope::from_json(&numeric_json.to_string()).unwrap();

        assert_eq!(symbolic.semantic, numeric.semantic);
        assert_eq!(symbolic.semantic_hash().unwrap(), numeric.semantic_hash().unwrap());
    }

    #[test]
    fn action6_requires_valid_coordinates() {
        let action = Arc3WireActionInput {
            id: json!("ACTION6"),
            data: BTreeMap::from([("x".to_string(), json!(63)), ("y".to_string(), json!(0))]),
            reasoning: None,
        };
        assert_eq!(
            normalize_action_input(&action).unwrap(),
            Arc3Action {
                kind: Arc3ActionKind::Action6,
                coordinates: Some(Arc3Coordinates { x: 63, y: 0 }),
            }
        );

        let invalid = Arc3WireActionInput {
            id: json!("ACTION6"),
            data: BTreeMap::from([("x".to_string(), json!(64)), ("y".to_string(), json!(0))]),
            reasoning: None,
        };
        assert!(normalize_action_input(&invalid).is_err());
    }

    #[test]
    fn invalid_cell_is_rejected() {
        let mut value: Value = serde_json::from_str(&observation_json(
            "game-a",
            "guid-a",
            Value::Null,
        ))
        .unwrap();
        value["frame"][0][0][0] = json!(16);
        assert!(Arc3Envelope::from_json(&value.to_string()).is_err());
    }

    #[test]
    fn duplicate_available_action_is_rejected() {
        let mut value: Value = serde_json::from_str(&observation_json(
            "game-a",
            "guid-a",
            Value::Null,
        ))
        .unwrap();
        value["available_actions"] = json!([1, 1, 2]);
        assert!(matches!(
            Arc3Envelope::from_json(&value.to_string()),
            Err(Arc3ProtocolError::DuplicateAvailableAction(1))
        ));
    }
}
