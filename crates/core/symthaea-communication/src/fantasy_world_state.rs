// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fiction-only world-state and continuity semantics for adult fantasy dialogue.
//!
//! This module stores opaque narrative references and provenance, not raw erotic
//! transcript content. Its state is explicitly fictional and non-authoritative.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const FANTASY_WORLD_STATE_SCHEMA_V1: &str =
    "symthaea.communication.fantasy-world-state.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FantasyRealityNamespaceV1 {
    FictionOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FantasyFactSourceV1 {
    ParticipantAuthored,
    SymthaeaAuthored,
    SharedNarrative,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FantasyCharacterV1 {
    character_id: String,
    persona_ref: String,
    introduced_turn: u64,
}

impl FantasyCharacterV1 {
    pub fn new(
        character_id: impl Into<String>,
        persona_ref: impl Into<String>,
        introduced_turn: u64,
    ) -> Result<Self, FantasyWorldStateErrorV1> {
        Ok(Self {
            character_id: canonical_id(character_id.into(), 128)?,
            persona_ref: canonical_ref(persona_ref.into(), 512)?,
            introduced_turn,
        })
    }

    pub fn character_id(&self) -> &str {
        &self.character_id
    }

    pub fn persona_ref(&self) -> &str {
        &self.persona_ref
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FantasyFactV1 {
    fact_id: String,
    semantic_key: String,
    value_ref: String,
    source: FantasyFactSourceV1,
    introduced_turn: u64,
    supersedes_fact_id: Option<String>,
}

impl FantasyFactV1 {
    pub fn new(
        fact_id: impl Into<String>,
        semantic_key: impl Into<String>,
        value_ref: impl Into<String>,
        source: FantasyFactSourceV1,
        introduced_turn: u64,
        supersedes_fact_id: Option<String>,
    ) -> Result<Self, FantasyWorldStateErrorV1> {
        Ok(Self {
            fact_id: canonical_id(fact_id.into(), 256)?,
            semantic_key: canonical_id(semantic_key.into(), 256)?,
            value_ref: canonical_ref(value_ref.into(), 1024)?,
            source,
            introduced_turn,
            supersedes_fact_id: supersedes_fact_id
                .map(|id| canonical_id(id, 256))
                .transpose()?,
        })
    }

    pub fn fact_id(&self) -> &str {
        &self.fact_id
    }

    pub fn semantic_key(&self) -> &str {
        &self.semantic_key
    }

    pub fn value_ref(&self) -> &str {
        &self.value_ref
    }

    pub const fn source(&self) -> FantasyFactSourceV1 {
        self.source
    }

    pub fn supersedes_fact_id(&self) -> Option<&str> {
        self.supersedes_fact_id.as_deref()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FantasyThreadStateV1 {
    Open,
    Resolved,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FantasyNarrativeThreadV1 {
    thread_id: String,
    thread_ref: String,
    opened_turn: u64,
    resolved_turn: Option<u64>,
    state: FantasyThreadStateV1,
}

impl FantasyNarrativeThreadV1 {
    pub fn new(
        thread_id: impl Into<String>,
        thread_ref: impl Into<String>,
        opened_turn: u64,
    ) -> Result<Self, FantasyWorldStateErrorV1> {
        Ok(Self {
            thread_id: canonical_id(thread_id.into(), 256)?,
            thread_ref: canonical_ref(thread_ref.into(), 1024)?,
            opened_turn,
            resolved_turn: None,
            state: FantasyThreadStateV1::Open,
        })
    }

    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    pub const fn state(&self) -> FantasyThreadStateV1 {
        self.state
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FantasySceneStateV1 {
    scene_id: String,
    scene_epoch: u64,
    tone_ref: Option<String>,
}

impl FantasySceneStateV1 {
    pub fn new(
        scene_id: impl Into<String>,
        scene_epoch: u64,
        tone_ref: Option<String>,
    ) -> Result<Self, FantasyWorldStateErrorV1> {
        if scene_epoch == 0 {
            return Err(FantasyWorldStateErrorV1::InvalidSceneEpoch);
        }
        Ok(Self {
            scene_id: canonical_id(scene_id.into(), 256)?,
            scene_epoch,
            tone_ref: tone_ref.map(|value| canonical_ref(value, 512)).transpose()?,
        })
    }

    pub fn scene_id(&self) -> &str {
        &self.scene_id
    }

    pub const fn scene_epoch(&self) -> u64 {
        self.scene_epoch
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FantasyWorldStateErrorV1 {
    InvalidWorldId,
    InvalidSessionId,
    InvalidId,
    InvalidReference,
    InvalidSceneEpoch,
    DuplicateCharacterId,
    DuplicateFactId,
    DuplicateThreadId,
    UnknownSupersededFact,
    SupersededSemanticKeyMismatch,
    FactAlreadySuperseded,
    UnknownThread,
    ThreadAlreadyResolved,
    StaleSceneEpoch,
}

#[derive(Debug)]
pub struct FantasyWorldStateV1 {
    world_id: String,
    session_id: String,
    namespace: FantasyRealityNamespaceV1,
    scene: FantasySceneStateV1,
    characters: BTreeMap<String, FantasyCharacterV1>,
    facts: BTreeMap<String, FantasyFactV1>,
    superseded_fact_ids: BTreeSet<String>,
    threads: BTreeMap<String, FantasyNarrativeThreadV1>,
    callback_refs: BTreeSet<String>,
}

impl FantasyWorldStateV1 {
    pub fn new(
        world_id: impl Into<String>,
        session_id: impl Into<String>,
        scene: FantasySceneStateV1,
    ) -> Result<Self, FantasyWorldStateErrorV1> {
        let world_id = canonical_id(world_id.into(), 256)
            .map_err(|_| FantasyWorldStateErrorV1::InvalidWorldId)?;
        let session_id = canonical_id(session_id.into(), 256)
            .map_err(|_| FantasyWorldStateErrorV1::InvalidSessionId)?;
        Ok(Self {
            world_id,
            session_id,
            namespace: FantasyRealityNamespaceV1::FictionOnly,
            scene,
            characters: BTreeMap::new(),
            facts: BTreeMap::new(),
            superseded_fact_ids: BTreeSet::new(),
            threads: BTreeMap::new(),
            callback_refs: BTreeSet::new(),
        })
    }

    pub fn world_id(&self) -> &str {
        &self.world_id
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub const fn namespace(&self) -> FantasyRealityNamespaceV1 {
        self.namespace
    }

    pub fn scene(&self) -> &FantasySceneStateV1 {
        &self.scene
    }

    pub fn advance_scene(
        &mut self,
        next_scene: FantasySceneStateV1,
    ) -> Result<(), FantasyWorldStateErrorV1> {
        if next_scene.scene_epoch <= self.scene.scene_epoch {
            return Err(FantasyWorldStateErrorV1::StaleSceneEpoch);
        }
        self.scene = next_scene;
        Ok(())
    }

    pub fn add_character(
        &mut self,
        character: FantasyCharacterV1,
    ) -> Result<(), FantasyWorldStateErrorV1> {
        if self.characters.contains_key(&character.character_id) {
            return Err(FantasyWorldStateErrorV1::DuplicateCharacterId);
        }
        self.characters
            .insert(character.character_id.clone(), character);
        Ok(())
    }

    pub fn record_fact(
        &mut self,
        fact: FantasyFactV1,
    ) -> Result<(), FantasyWorldStateErrorV1> {
        if self.facts.contains_key(&fact.fact_id) {
            return Err(FantasyWorldStateErrorV1::DuplicateFactId);
        }
        if let Some(previous_id) = fact.supersedes_fact_id.as_deref() {
            let previous = self
                .facts
                .get(previous_id)
                .ok_or(FantasyWorldStateErrorV1::UnknownSupersededFact)?;
            if previous.semantic_key != fact.semantic_key {
                return Err(FantasyWorldStateErrorV1::SupersededSemanticKeyMismatch);
            }
            if self.superseded_fact_ids.contains(previous_id) {
                return Err(FantasyWorldStateErrorV1::FactAlreadySuperseded);
            }
            self.superseded_fact_ids.insert(previous_id.to_owned());
        }
        self.facts.insert(fact.fact_id.clone(), fact);
        Ok(())
    }

    pub fn current_fact_for_key(&self, semantic_key: &str) -> Option<&FantasyFactV1> {
        self.facts
            .values()
            .filter(|fact| {
                fact.semantic_key == semantic_key
                    && !self.superseded_fact_ids.contains(&fact.fact_id)
            })
            .max_by_key(|fact| fact.introduced_turn)
    }

    pub fn open_thread(
        &mut self,
        thread: FantasyNarrativeThreadV1,
    ) -> Result<(), FantasyWorldStateErrorV1> {
        if self.threads.contains_key(&thread.thread_id) {
            return Err(FantasyWorldStateErrorV1::DuplicateThreadId);
        }
        self.threads.insert(thread.thread_id.clone(), thread);
        Ok(())
    }

    pub fn resolve_thread(
        &mut self,
        thread_id: &str,
        resolved_turn: u64,
    ) -> Result<(), FantasyWorldStateErrorV1> {
        let thread = self
            .threads
            .get_mut(thread_id)
            .ok_or(FantasyWorldStateErrorV1::UnknownThread)?;
        if thread.state == FantasyThreadStateV1::Resolved {
            return Err(FantasyWorldStateErrorV1::ThreadAlreadyResolved);
        }
        thread.state = FantasyThreadStateV1::Resolved;
        thread.resolved_turn = Some(resolved_turn);
        Ok(())
    }

    pub fn thread(&self, thread_id: &str) -> Option<&FantasyNarrativeThreadV1> {
        self.threads.get(thread_id)
    }

    pub fn add_callback_ref(
        &mut self,
        callback_ref: impl Into<String>,
    ) -> Result<bool, FantasyWorldStateErrorV1> {
        let callback_ref = canonical_ref(callback_ref.into(), 1024)?;
        Ok(self.callback_refs.insert(callback_ref))
    }

    pub fn callback_refs(&self) -> &BTreeSet<String> {
        &self.callback_refs
    }
}

fn canonical_id(value: String, max_len: usize) -> Result<String, FantasyWorldStateErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > max_len {
        return Err(FantasyWorldStateErrorV1::InvalidId);
    }
    Ok(value)
}

fn canonical_ref(value: String, max_len: usize) -> Result<String, FantasyWorldStateErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > max_len {
        return Err(FantasyWorldStateErrorV1::InvalidReference);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> FantasyWorldStateV1 {
        FantasyWorldStateV1::new(
            "world-a",
            "session-a",
            FantasySceneStateV1::new("scene-1", 1, Some("tone:warm".into())).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn world_state_is_permanently_fiction_namespaced() {
        let world = world();
        assert_eq!(world.namespace(), FantasyRealityNamespaceV1::FictionOnly);
    }

    #[test]
    fn fact_provenance_and_supersession_are_preserved() {
        let mut world = world();
        world
            .record_fact(
                FantasyFactV1::new(
                    "fact-1",
                    "location.current",
                    "fiction:location-a",
                    FantasyFactSourceV1::ParticipantAuthored,
                    1,
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        world
            .record_fact(
                FantasyFactV1::new(
                    "fact-2",
                    "location.current",
                    "fiction:location-b",
                    FantasyFactSourceV1::SharedNarrative,
                    2,
                    Some("fact-1".into()),
                )
                .unwrap(),
            )
            .unwrap();
        let current = world.current_fact_for_key("location.current").unwrap();
        assert_eq!(current.fact_id(), "fact-2");
        assert_eq!(current.source(), FantasyFactSourceV1::SharedNarrative);
        assert_eq!(world.facts.get("fact-1").unwrap().value_ref(), "fiction:location-a");
    }

    #[test]
    fn supersession_cannot_cross_semantic_keys() {
        let mut world = world();
        world
            .record_fact(
                FantasyFactV1::new(
                    "fact-1",
                    "location.current",
                    "fiction:location-a",
                    FantasyFactSourceV1::SharedNarrative,
                    1,
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            world.record_fact(
                FantasyFactV1::new(
                    "fact-2",
                    "character.mood",
                    "fiction:mood-a",
                    FantasyFactSourceV1::SharedNarrative,
                    2,
                    Some("fact-1".into()),
                )
                .unwrap(),
            ),
            Err(FantasyWorldStateErrorV1::SupersededSemanticKeyMismatch)
        );
    }

    #[test]
    fn unresolved_threads_have_explicit_lifecycle() {
        let mut world = world();
        world
            .open_thread(FantasyNarrativeThreadV1::new("thread-1", "fiction:thread", 1).unwrap())
            .unwrap();
        assert_eq!(
            world.thread("thread-1").unwrap().state(),
            FantasyThreadStateV1::Open
        );
        world.resolve_thread("thread-1", 5).unwrap();
        assert_eq!(
            world.thread("thread-1").unwrap().state(),
            FantasyThreadStateV1::Resolved
        );
    }

    #[test]
    fn scene_epoch_must_move_forward() {
        let mut world = world();
        assert_eq!(
            world.advance_scene(FantasySceneStateV1::new("scene-old", 1, None).unwrap()),
            Err(FantasyWorldStateErrorV1::StaleSceneEpoch)
        );
        world
            .advance_scene(FantasySceneStateV1::new("scene-2", 2, None).unwrap())
            .unwrap();
        assert_eq!(world.scene().scene_id(), "scene-2");
    }
}
