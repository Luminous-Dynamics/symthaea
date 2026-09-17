// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Minimal canonical ARC-AGI-3 observation/action contract.
//!
//! This crate is deliberately small so the benchmark protocol can be qualified
//! independently of the much larger psychological benchmark dependency graph.
//! The wire parser is strict: unknown fields fail closed, while provenance-only
//! identifiers are retained outside policy-authoritative bytes.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const ARC3_SEMANTIC_SCHEMA_V1: &[u8] = b"symthaea.arc3.semantic-observation.v1\0";
pub const ARC3_PROTOCOL_REFERENCE_ARCENGINE_COMMIT: &str =
    "b495c6acaf253c9681cd7b75c4299d352e9ce6f8";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Arc3GameState {
    NotPlayed,
    NotFinished,
    Win,
    GameOver,
}

impl Arc3GameState {
    const fn code(self) -> u8 {
        match self {
            Self::NotPlayed => 0,
            Self::NotFinished => 1,
            Self::Win => 2,
            Self::GameOver => 3,
        }
    }
}

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

    pub const fn id(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Arc3Coordinates {
    pub x: u8,
    pub y: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Arc3Action {
    pub kind: Arc3ActionKind,
    pub coordinates: Option<Arc3Coordinates>,
}

impl Arc3Action {
    pub fn simple(kind: Arc3ActionKind) -> Result<Self, Arc3ProtocolError> {
        if kind == Arc3ActionKind::Action6 {
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

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireActionInput {
    id: Value,
    #[serde(default)]
    data: BTreeMap<String, Value>,
    #[serde(default)]
    reasoning: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireObservation {
    #[serde(default)]
    game_id: String,
    frame: Vec<Vec<Vec<u8>>>,
    state: Arc3GameState,
    #[serde(default)]
    levels_completed: u16,
    #[serde(default)]
    win_levels: u16,
    #[serde(default)]
    action_input: Option<WireActionInput>,
    #[serde(default)]
    guid: Option<String>,
    #[serde(default)]
    full_reset: bool,
    #[serde(default)]
    available_actions: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arc3Provenance {
    pub game_id: String,
    pub guid: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Arc3Observation {
    pub frame: Vec<Vec<Vec<u8>>>,
    pub state: Arc3GameState,
    pub levels_completed: u16,
    pub win_levels: u16,
    pub last_action: Option<Arc3Action>,
    pub full_reset: bool,
    pub available_actions: Vec<Arc3ActionKind>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arc3Envelope {
    pub semantic: Arc3Observation,
    pub provenance: Arc3Provenance,
}

impl Arc3Envelope {
    pub fn from_json(json: &str) -> Result<Self, Arc3ProtocolError> {
        let wire: WireObservation =
            serde_json::from_str(json).map_err(|e| Arc3ProtocolError::Json(e.to_string()))?;
        Self::try_from(wire)
    }

    pub fn semantic_bytes(&self) -> Result<Vec<u8>, Arc3ProtocolError> {
        validate_observation(&self.semantic)?;
        let mut out = Vec::new();
        out.extend_from_slice(ARC3_SEMANTIC_SCHEMA_V1);
        out.push(self.semantic.state.code());
        out.extend_from_slice(&self.semantic.levels_completed.to_be_bytes());
        out.extend_from_slice(&self.semantic.win_levels.to_be_bytes());
        out.push(u8::from(self.semantic.full_reset));

        match &self.semantic.last_action {
            None => out.push(0),
            Some(action) => {
                out.push(1);
                out.push(action.kind.id());
                if let Some(xy) = action.coordinates {
                    out.extend_from_slice(&[xy.x, xy.y]);
                }
            }
        }

        out.push(self.semantic.available_actions.len() as u8);
        out.extend(self.semantic.available_actions.iter().map(|a| a.id()));
        out.extend_from_slice(&(self.semantic.frame.len() as u32).to_be_bytes());
        for grid in &self.semantic.frame {
            out.extend_from_slice(&[grid.len() as u8, grid[0].len() as u8]);
            for row in grid {
                out.extend_from_slice(row);
            }
        }
        Ok(out)
    }

    pub fn semantic_hash(&self) -> Result<blake3::Hash, Arc3ProtocolError> {
        Ok(blake3::hash(&self.semantic_bytes()?))
    }
}

impl TryFrom<WireObservation> for Arc3Envelope {
    type Error = Arc3ProtocolError;

    fn try_from(wire: WireObservation) -> Result<Self, Self::Error> {
        validate_frame(&wire.frame)?;
        validate_level_count(wire.levels_completed)?;
        validate_level_count(wire.win_levels)?;

        let last_action = wire.action_input.as_ref().map(parse_action).transpose()?;
        if let Some(input) = &wire.action_input {
            if let Some(value) = input.data.get("game_id") {
                let action_game = value
                    .as_str()
                    .ok_or(Arc3ProtocolError::InvalidActionField("game_id"))?;
                if !wire.game_id.is_empty()
                    && !action_game.is_empty()
                    && action_game != wire.game_id
                {
                    return Err(Arc3ProtocolError::ActionGameIdMismatch);
                }
            }
            let _ = &input.reasoning;
        }

        let mut actions = BTreeSet::new();
        for id in wire.available_actions {
            let action = Arc3ActionKind::from_id(id)?;
            if !actions.insert(action) {
                return Err(Arc3ProtocolError::DuplicateAvailableAction(id));
            }
        }

        let semantic = Arc3Observation {
            frame: wire.frame,
            state: wire.state,
            levels_completed: wire.levels_completed,
            win_levels: wire.win_levels,
            last_action,
            full_reset: wire.full_reset,
            available_actions: actions.into_iter().collect(),
        };
        validate_observation(&semantic)?;

        Ok(Self {
            semantic,
            provenance: Arc3Provenance {
                game_id: wire.game_id,
                guid: wire.guid,
            },
        })
    }
}

fn parse_action(input: &WireActionInput) -> Result<Arc3Action, Arc3ProtocolError> {
    let kind = parse_action_kind(&input.id)?;
    for (key, value) in &input.data {
        let allowed =
            key == "game_id" || (kind == Arc3ActionKind::Action6 && (key == "x" || key == "y"));
        if !allowed {
            return Err(Arc3ProtocolError::UnexpectedActionField(key.clone()));
        }
        if key == "game_id" && !value.is_string() {
            return Err(Arc3ProtocolError::InvalidActionField("game_id"));
        }
    }
    if kind == Arc3ActionKind::Action6 {
        Arc3Action::action6(coord(&input.data, "x")?, coord(&input.data, "y")?)
    } else {
        Arc3Action::simple(kind)
    }
}

fn parse_action_kind(value: &Value) -> Result<Arc3ActionKind, Arc3ProtocolError> {
    if let Some(id) = value.as_u64() {
        return Arc3ActionKind::from_id(
            u8::try_from(id).map_err(|_| Arc3ProtocolError::InvalidActionId(id.to_string()))?,
        );
    }
    let name = value
        .as_str()
        .ok_or_else(|| Arc3ProtocolError::InvalidActionId(value.to_string()))?;
    match name.to_ascii_uppercase().as_str() {
        "RESET" => Ok(Arc3ActionKind::Reset),
        "ACTION1" => Ok(Arc3ActionKind::Action1),
        "ACTION2" => Ok(Arc3ActionKind::Action2),
        "ACTION3" => Ok(Arc3ActionKind::Action3),
        "ACTION4" => Ok(Arc3ActionKind::Action4),
        "ACTION5" => Ok(Arc3ActionKind::Action5),
        "ACTION6" => Ok(Arc3ActionKind::Action6),
        "ACTION7" => Ok(Arc3ActionKind::Action7),
        other => Err(Arc3ProtocolError::InvalidActionId(other.to_owned())),
    }
}

fn coord(data: &BTreeMap<String, Value>, key: &'static str) -> Result<u8, Arc3ProtocolError> {
    let n = data
        .get(key)
        .and_then(Value::as_u64)
        .ok_or(Arc3ProtocolError::InvalidActionField(key))?;
    let n = u8::try_from(n).map_err(|_| Arc3ProtocolError::InvalidActionField(key))?;
    if n > 63 {
        return Err(Arc3ProtocolError::InvalidActionField(key));
    }
    Ok(n)
}

fn validate_observation(o: &Arc3Observation) -> Result<(), Arc3ProtocolError> {
    validate_frame(&o.frame)?;
    validate_level_count(o.levels_completed)?;
    validate_level_count(o.win_levels)?;
    if let Some(action) = &o.last_action {
        match (action.kind, action.coordinates) {
            (Arc3ActionKind::Action6, Some(xy)) if xy.x <= 63 && xy.y <= 63 => {}
            (Arc3ActionKind::Action6, None) => return Err(Arc3ProtocolError::MissingCoordinates),
            (Arc3ActionKind::Action6, Some(xy)) => {
                return Err(Arc3ProtocolError::CoordinateOutOfRange { x: xy.x, y: xy.y });
            }
            (_, Some(_)) => return Err(Arc3ProtocolError::UnexpectedCoordinates),
            (_, None) => {}
        }
    }
    if o.available_actions.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Arc3ProtocolError::NonCanonicalAvailableActions);
    }
    Ok(())
}

fn validate_level_count(n: u16) -> Result<(), Arc3ProtocolError> {
    if n > 254 {
        Err(Arc3ProtocolError::LevelCountOutOfRange(n))
    } else {
        Ok(())
    }
}

fn validate_frame(frame: &[Vec<Vec<u8>>]) -> Result<(), Arc3ProtocolError> {
    if frame.is_empty() {
        return Err(Arc3ProtocolError::EmptyFrame);
    }
    for (i, grid) in frame.iter().enumerate() {
        if grid.is_empty() || grid.len() > 64 {
            return Err(Arc3ProtocolError::InvalidGridDimensions(i));
        }
        let width = grid[0].len();
        if width == 0 || width > 64 || grid.iter().any(|row| row.len() != width) {
            return Err(Arc3ProtocolError::InvalidGridDimensions(i));
        }
        if grid.iter().flatten().any(|&cell| cell > 15) {
            return Err(Arc3ProtocolError::InvalidCellValue(i));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Arc3ProtocolError {
    Json(String),
    EmptyFrame,
    InvalidGridDimensions(usize),
    InvalidCellValue(usize),
    InvalidActionId(String),
    DuplicateAvailableAction(u8),
    MissingCoordinates,
    CoordinateOutOfRange { x: u8, y: u8 },
    InvalidActionField(&'static str),
    UnexpectedActionField(String),
    UnexpectedCoordinates,
    ActionGameIdMismatch,
    NonCanonicalAvailableActions,
    LevelCountOutOfRange(u16),
}

impl fmt::Display for Arc3ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ARC-3 protocol error: {self:?}")
    }
}

impl std::error::Error for Arc3ProtocolError {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture(game: &str, guid: &str, reasoning: Value) -> String {
        json!({
            "game_id": game,
            "state": "NOT_FINISHED",
            "levels_completed": 1,
            "win_levels": 7,
            "action_input": {"id": "ACTION1", "data": {}, "reasoning": reasoning},
            "guid": guid,
            "full_reset": false,
            "available_actions": [6, 1, 4, 2],
            "frame": [[[0,1,0],[2,3,2]], [[0,1,0],[2,4,2]]]
        })
        .to_string()
    }

    #[test]
    fn official_shape_normalizes() {
        let e = Arc3Envelope::from_json(&fixture("ls20-v1", "g", Value::Null)).unwrap();
        assert_eq!(e.semantic.state, Arc3GameState::NotFinished);
        assert_eq!(
            e.semantic.available_actions,
            vec![
                Arc3ActionKind::Action1,
                Arc3ActionKind::Action2,
                Arc3ActionKind::Action4,
                Arc3ActionKind::Action6
            ]
        );
    }

    #[test]
    fn provenance_and_reasoning_are_not_semantic() {
        let a = Arc3Envelope::from_json(&fixture("a", "ga", json!({"note":"a"}))).unwrap();
        let b = Arc3Envelope::from_json(&fixture("b", "gb", json!({"note":"b"}))).unwrap();
        assert_ne!(a.provenance, b.provenance);
        assert_eq!(a.semantic_hash().unwrap(), b.semantic_hash().unwrap());
    }

    #[test]
    fn canonical_bytes_are_versioned_and_non_json() {
        let e = Arc3Envelope::from_json(&fixture("a", "g", Value::Null)).unwrap();
        let bytes = e.semantic_bytes().unwrap();
        assert!(bytes.starts_with(ARC3_SEMANTIC_SCHEMA_V1));
        assert_ne!(bytes.first().copied(), Some(b'{'));
    }

    #[test]
    fn semantic_change_changes_hash() {
        let a = Arc3Envelope::from_json(&fixture("a", "g", Value::Null)).unwrap();
        let mut v: Value = serde_json::from_str(&fixture("a", "g", Value::Null)).unwrap();
        v["frame"][1][1][1] = json!(5);
        let b = Arc3Envelope::from_json(&v.to_string()).unwrap();
        assert_ne!(a.semantic_hash().unwrap(), b.semantic_hash().unwrap());
    }

    #[test]
    fn unknown_wire_field_fails_closed() {
        let mut v: Value = serde_json::from_str(&fixture("a", "g", Value::Null)).unwrap();
        v["future_field"] = json!(1);
        assert!(matches!(
            Arc3Envelope::from_json(&v.to_string()),
            Err(Arc3ProtocolError::Json(_))
        ));
    }

    #[test]
    fn action6_bounds_and_game_id_consistency_are_enforced() {
        let mut v: Value = serde_json::from_str(&fixture("a", "g", Value::Null)).unwrap();
        v["action_input"] = json!({
            "id":"ACTION6",
            "data":{"game_id":"a","x":63,"y":0},
            "reasoning":null
        });
        let e = Arc3Envelope::from_json(&v.to_string()).unwrap();
        assert_eq!(e.semantic.last_action.unwrap().coordinates.unwrap().x, 63);

        v["action_input"]["data"]["game_id"] = json!("b");
        assert!(matches!(
            Arc3Envelope::from_json(&v.to_string()),
            Err(Arc3ProtocolError::ActionGameIdMismatch)
        ));
    }
}
