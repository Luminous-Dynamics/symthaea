// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent world lifecycle evidence: snapshot, suspend, resume, revisit and archive.
//!
//! This module is host-neutral. It proves continuity and authority boundaries,
//! but it does not serialize a host world, stop/start a runtime, or mint authority.

use serde::{Deserialize, Serialize};

use crate::{
    digest::{DigestAlgorithm, TypedDigest},
    presence::WorldPresenceSession,
    types::{RealityLayer, WorldDescriptor, WorldOrigin, WorldRelation},
};

pub const WORLD_SNAPSHOT_DIGEST_DOMAIN: &str = "symthaea.world-snapshot.v1";
pub const WORLD_REVISIT_DIGEST_DOMAIN: &str = "symthaea.world-revisit.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorldLifecycleState {
    Active,
    Suspended,
    Archived,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorldLifecycleTransition {
    Suspend,
    Resume,
    Archive,
}

impl WorldLifecycleTransition {
    pub fn expected_states(self) -> (WorldLifecycleState, WorldLifecycleState) {
        match self {
            Self::Suspend => (WorldLifecycleState::Active, WorldLifecycleState::Suspended),
            Self::Resume => (WorldLifecycleState::Suspended, WorldLifecycleState::Active),
            Self::Archive => (WorldLifecycleState::Suspended, WorldLifecycleState::Archived),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldSnapshotManifest {
    pub schema_version: u32,
    pub snapshot_id: String,
    pub world: WorldDescriptor,
    pub genesis_digest: TypedDigest,
    pub state_digest: TypedDigest,
    pub ledger_head_digest: TypedDigest,
    pub host_artifact_digest: TypedDigest,
    pub frame: Option<u64>,
    pub previous_snapshot_digest: Option<TypedDigest>,
}

impl WorldSnapshotManifest {
    pub fn validate(&self) -> Result<(), WorldLifecycleError> {
        if self.schema_version == 0 {
            return Err(WorldLifecycleError::InvalidSchemaVersion);
        }
        if self.snapshot_id.trim().is_empty() {
            return Err(WorldLifecycleError::MissingSnapshotId);
        }
        self.world
            .validate()
            .map_err(|error| WorldLifecycleError::InvalidWorld(error.to_string()))?;
        for digest in [
            &self.genesis_digest,
            &self.state_digest,
            &self.ledger_head_digest,
            &self.host_artifact_digest,
        ] {
            digest
                .validate()
                .map_err(|error| WorldLifecycleError::InvalidDigest(error.to_string()))?;
        }
        if let Some(previous) = &self.previous_snapshot_digest {
            previous
                .validate()
                .map_err(|error| WorldLifecycleError::InvalidDigest(error.to_string()))?;
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<TypedDigest, WorldLifecycleError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        feed(&mut hasher, b"symthaea.world-snapshot.v1");
        hasher.update(&self.schema_version.to_le_bytes());
        feed(&mut hasher, self.snapshot_id.as_bytes());
        feed_world(&mut hasher, &self.world);
        for digest in [
            &self.genesis_digest,
            &self.state_digest,
            &self.ledger_head_digest,
            &self.host_artifact_digest,
        ] {
            feed_typed_digest(&mut hasher, digest);
        }
        feed_optional_frame(&mut hasher, self.frame);
        feed_optional_typed_digest(&mut hasher, self.previous_snapshot_digest.as_ref());
        TypedDigest::new(
            WORLD_SNAPSHOT_DIGEST_DOMAIN,
            DigestAlgorithm::Blake3,
            hasher.finalize().to_hex().to_string(),
        )
        .map_err(|error| WorldLifecycleError::InvalidDigest(error.to_string()))
    }

    pub fn validate_successor(&self, previous: &Self) -> Result<(), WorldLifecycleError> {
        self.validate()?;
        previous.validate()?;
        if self.world != previous.world {
            return Err(WorldLifecycleError::WorldMismatch);
        }
        let expected_previous = previous.digest()?;
        let actual_previous = self
            .previous_snapshot_digest
            .as_ref()
            .ok_or(WorldLifecycleError::MissingPreviousSnapshotDigest)?;
        if !actual_previous.same_typed_value(&expected_previous) {
            return Err(WorldLifecycleError::PreviousSnapshotMismatch);
        }
        if let (Some(previous_frame), Some(current_frame)) = (previous.frame, self.frame) {
            if current_frame < previous_frame {
                return Err(WorldLifecycleError::FrameRegression);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldLifecycleReceipt {
    pub transition_id: String,
    pub world: WorldDescriptor,
    pub transition: WorldLifecycleTransition,
    pub from_state: WorldLifecycleState,
    pub to_state: WorldLifecycleState,
    pub snapshot_digest: TypedDigest,
    pub state_digest: TypedDigest,
    pub frame: Option<u64>,
    pub authority_receipt_digest: Option<TypedDigest>,
}

impl WorldLifecycleReceipt {
    pub fn validate(&self) -> Result<(), WorldLifecycleError> {
        if self.transition_id.trim().is_empty() {
            return Err(WorldLifecycleError::MissingTransitionId);
        }
        self.world
            .validate()
            .map_err(|error| WorldLifecycleError::InvalidWorld(error.to_string()))?;
        self.snapshot_digest
            .validate()
            .map_err(|error| WorldLifecycleError::InvalidDigest(error.to_string()))?;
        self.state_digest
            .validate()
            .map_err(|error| WorldLifecycleError::InvalidDigest(error.to_string()))?;
        self.authority_receipt_digest
            .as_ref()
            .ok_or(WorldLifecycleError::MissingAuthorityReceipt)?
            .validate()
            .map_err(|error| WorldLifecycleError::InvalidDigest(error.to_string()))?;
        let (expected_from, expected_to) = self.transition.expected_states();
        if self.from_state != expected_from || self.to_state != expected_to {
            return Err(WorldLifecycleError::InvalidTransitionStates);
        }
        Ok(())
    }

    pub fn validate_against_snapshot(
        &self,
        snapshot: &WorldSnapshotManifest,
    ) -> Result<(), WorldLifecycleError> {
        self.validate()?;
        snapshot.validate()?;
        if self.world != snapshot.world {
            return Err(WorldLifecycleError::WorldMismatch);
        }
        if !self
            .snapshot_digest
            .same_typed_value(&snapshot.digest()?)
        {
            return Err(WorldLifecycleError::SnapshotDigestMismatch);
        }
        if !self.state_digest.same_typed_value(&snapshot.state_digest) {
            return Err(WorldLifecycleError::StateContinuityMismatch);
        }
        if let (Some(snapshot_frame), Some(transition_frame)) = (snapshot.frame, self.frame) {
            if transition_frame < snapshot_frame {
                return Err(WorldLifecycleError::FrameRegression);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldRevisitReceipt {
    pub receipt_id: String,
    pub world: WorldDescriptor,
    pub agent_id: String,
    pub prior_session_id: String,
    pub resumed_session_id: String,
    pub snapshot_digest: TypedDigest,
    pub state_digest: TypedDigest,
    pub prior_exit_frame: Option<u64>,
    pub resumed_entry_frame: Option<u64>,
}

impl WorldRevisitReceipt {
    pub fn prove(
        receipt_id: impl Into<String>,
        snapshot: &WorldSnapshotManifest,
        prior: &WorldPresenceSession,
        resumed: &WorldPresenceSession,
    ) -> Result<Self, WorldLifecycleError> {
        let receipt_id = receipt_id.into();
        if receipt_id.trim().is_empty() {
            return Err(WorldLifecycleError::MissingRevisitId);
        }
        snapshot.validate()?;
        prior
            .validate()
            .map_err(|error| WorldLifecycleError::InvalidPresence(error.to_string()))?;
        resumed
            .validate()
            .map_err(|error| WorldLifecycleError::InvalidPresence(error.to_string()))?;
        if prior.is_open() {
            return Err(WorldLifecycleError::PriorPresenceStillOpen);
        }
        if !resumed.is_open() {
            return Err(WorldLifecycleError::ResumedPresenceAlreadyClosed);
        }
        if prior.session_id == resumed.session_id {
            return Err(WorldLifecycleError::ReusedPresenceSessionId);
        }
        if prior.world != snapshot.world || resumed.world != snapshot.world {
            return Err(WorldLifecycleError::WorldMismatch);
        }
        if prior.agent_id != resumed.agent_id {
            return Err(WorldLifecycleError::AgentMismatch);
        }
        let prior_exit = prior
            .exit_state_digest
            .as_ref()
            .ok_or(WorldLifecycleError::PriorPresenceMissingExitState)?;
        if !prior_exit.same_typed_value(&snapshot.state_digest)
            || !resumed
                .entry_state_digest
                .same_typed_value(&snapshot.state_digest)
        {
            return Err(WorldLifecycleError::StateContinuityMismatch);
        }
        if let (Some(snapshot_frame), Some(entry_frame)) = (snapshot.frame, resumed.entered_frame) {
            if entry_frame < snapshot_frame {
                return Err(WorldLifecycleError::FrameRegression);
            }
        }
        let receipt = Self {
            receipt_id,
            world: snapshot.world.clone(),
            agent_id: prior.agent_id.clone(),
            prior_session_id: prior.session_id.clone(),
            resumed_session_id: resumed.session_id.clone(),
            snapshot_digest: snapshot.digest()?,
            state_digest: snapshot.state_digest.clone(),
            prior_exit_frame: prior.exited_frame,
            resumed_entry_frame: resumed.entered_frame,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn validate(&self) -> Result<(), WorldLifecycleError> {
        for value in [
            self.receipt_id.as_str(),
            self.agent_id.as_str(),
            self.prior_session_id.as_str(),
            self.resumed_session_id.as_str(),
        ] {
            if value.trim().is_empty() {
                return Err(WorldLifecycleError::MissingRevisitIdentity);
            }
        }
        if self.prior_session_id == self.resumed_session_id {
            return Err(WorldLifecycleError::ReusedPresenceSessionId);
        }
        self.world
            .validate()
            .map_err(|error| WorldLifecycleError::InvalidWorld(error.to_string()))?;
        self.snapshot_digest
            .validate()
            .map_err(|error| WorldLifecycleError::InvalidDigest(error.to_string()))?;
        self.state_digest
            .validate()
            .map_err(|error| WorldLifecycleError::InvalidDigest(error.to_string()))?;
        if let (Some(exit_frame), Some(entry_frame)) =
            (self.prior_exit_frame, self.resumed_entry_frame)
        {
            if entry_frame < exit_frame {
                return Err(WorldLifecycleError::FrameRegression);
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<TypedDigest, WorldLifecycleError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        feed(&mut hasher, b"symthaea.world-revisit.v1");
        feed(&mut hasher, self.receipt_id.as_bytes());
        feed_world(&mut hasher, &self.world);
        feed(&mut hasher, self.agent_id.as_bytes());
        feed(&mut hasher, self.prior_session_id.as_bytes());
        feed(&mut hasher, self.resumed_session_id.as_bytes());
        feed_typed_digest(&mut hasher, &self.snapshot_digest);
        feed_typed_digest(&mut hasher, &self.state_digest);
        feed_optional_frame(&mut hasher, self.prior_exit_frame);
        feed_optional_frame(&mut hasher, self.resumed_entry_frame);
        TypedDigest::new(
            WORLD_REVISIT_DIGEST_DOMAIN,
            DigestAlgorithm::Blake3,
            hasher.finalize().to_hex().to_string(),
        )
        .map_err(|error| WorldLifecycleError::InvalidDigest(error.to_string()))
    }
}

fn feed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn feed_optional_frame(hasher: &mut blake3::Hasher, frame: Option<u64>) {
    match frame {
        Some(frame) => {
            hasher.update(&[1]);
            hasher.update(&frame.to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn feed_optional_typed_digest(hasher: &mut blake3::Hasher, digest: Option<&TypedDigest>) {
    match digest {
        Some(digest) => {
            hasher.update(&[1]);
            feed_typed_digest(hasher, digest);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn feed_typed_digest(hasher: &mut blake3::Hasher, digest: &TypedDigest) {
    feed(hasher, digest.domain.as_bytes());
    match &digest.algorithm {
        DigestAlgorithm::Blake3 => {
            hasher.update(&[0]);
        }
        DigestAlgorithm::Sha256 => {
            hasher.update(&[1]);
        }
        DigestAlgorithm::Other(name) => {
            hasher.update(&[2]);
            feed(hasher, name.as_bytes());
        }
    }
    feed(hasher, digest.value.as_bytes());
}

fn feed_world(hasher: &mut blake3::Hasher, world: &WorldDescriptor) {
    feed(hasher, world.world_id.0.as_bytes());
    feed(hasher, world.lineage_id.0.as_bytes());
    hasher.update(&[layer_tag(world.layer)]);
    feed(hasher, world.creator_id.as_bytes());
    hasher.update(&world.generation_depth.to_le_bytes());
    match &world.parent {
        Some(parent) => {
            hasher.update(&[1]);
            feed(hasher, parent.world_id.0.as_bytes());
            feed(hasher, parent.lineage_id.0.as_bytes());
            hasher.update(&[relation_tag(&parent.relation)]);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match &world.origin {
        WorldOrigin::PhysicalSensorium => {
            hasher.update(&[0]);
        }
        WorldOrigin::DigitalHost { host_kind } => {
            hasher.update(&[1]);
            feed(hasher, host_kind.as_bytes());
        }
        WorldOrigin::CounterfactualBranch => {
            hasher.update(&[2]);
        }
        WorldOrigin::ReplayArtifact => {
            hasher.update(&[3]);
        }
        WorldOrigin::DreamEngine => {
            hasher.update(&[4]);
        }
        WorldOrigin::ImportedExternal { source } => {
            hasher.update(&[5]);
            feed(hasher, source.as_bytes());
        }
        WorldOrigin::Unknown => {
            hasher.update(&[6]);
        }
    }
}

fn layer_tag(layer: RealityLayer) -> u8 {
    match layer {
        RealityLayer::PhysicalGrounded => 0,
        RealityLayer::DigitalCommitted => 1,
        RealityLayer::Counterfactual => 2,
        RealityLayer::Replay => 3,
        RealityLayer::Dream => 4,
        RealityLayer::Imported => 5,
        RealityLayer::Unknown => 6,
    }
}

fn relation_tag(relation: &WorldRelation) -> u8 {
    match relation {
        WorldRelation::CounterfactualOf => 0,
        WorldRelation::ReplayOf => 1,
        WorldRelation::DreamedFrom => 2,
        WorldRelation::ImportedFrom => 3,
        WorldRelation::SpawnedFrom => 4,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorldLifecycleError {
    #[error("lifecycle schema version must be non-zero")]
    InvalidSchemaVersion,
    #[error("snapshot id may not be empty")]
    MissingSnapshotId,
    #[error("lifecycle transition id may not be empty")]
    MissingTransitionId,
    #[error("revisit receipt id may not be empty")]
    MissingRevisitId,
    #[error("revisit receipt contains an empty identity")]
    MissingRevisitIdentity,
    #[error("invalid world descriptor: {0}")]
    InvalidWorld(String),
    #[error("invalid typed digest: {0}")]
    InvalidDigest(String),
    #[error("invalid presence session: {0}")]
    InvalidPresence(String),
    #[error("world descriptors differ across a lifecycle continuity boundary")]
    WorldMismatch,
    #[error("snapshot successor is missing its previous snapshot digest")]
    MissingPreviousSnapshotDigest,
    #[error("snapshot successor does not reference the exact previous snapshot")]
    PreviousSnapshotMismatch,
    #[error("lifecycle frame regressed")]
    FrameRegression,
    #[error("authority-bearing lifecycle transition lacks an external authority receipt")]
    MissingAuthorityReceipt,
    #[error("lifecycle transition from/to states do not match the declared transition")]
    InvalidTransitionStates,
    #[error("lifecycle receipt references a different snapshot")]
    SnapshotDigestMismatch,
    #[error("persisted/resumed state does not equal the snapshot state as a typed digest")]
    StateContinuityMismatch,
    #[error("prior presence session is still open")]
    PriorPresenceStillOpen,
    #[error("resumed presence session is already closed")]
    ResumedPresenceAlreadyClosed,
    #[error("revisit reused the prior presence session id")]
    ReusedPresenceSessionId,
    #[error("revisit changed agent identity")]
    AgentMismatch,
    #[error("prior presence session has no exit state")]
    PriorPresenceMissingExitState,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        presence::PresenceCapability,
        types::{WorldId, WorldLineageId},
    };

    fn d(domain: &str) -> TypedDigest {
        TypedDigest::blake3(domain, domain.as_bytes()).unwrap()
    }

    fn world() -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId("garden".into()),
            lineage_id: WorldLineageId("garden-lineage".into()),
            layer: RealityLayer::DigitalCommitted,
            origin: WorldOrigin::DigitalHost {
                host_kind: "symtropy".into(),
            },
            parent: None,
            generation_depth: 0,
            creator_id: "symthaea".into(),
        }
    }

    fn snapshot() -> WorldSnapshotManifest {
        WorldSnapshotManifest {
            schema_version: 1,
            snapshot_id: "snap-1".into(),
            world: world(),
            genesis_digest: d("genesis.v1"),
            state_digest: d("state.v1"),
            ledger_head_digest: d("ledger.v1"),
            host_artifact_digest: d("artifact.v1"),
            frame: Some(100),
            previous_snapshot_digest: None,
        }
    }

    fn presence(session: &str, entry: TypedDigest, exit: Option<TypedDigest>) -> WorldPresenceSession {
        WorldPresenceSession {
            session_id: session.into(),
            agent_id: "symthaea".into(),
            world: world(),
            embodiment_id: "camera-body".into(),
            sensor_suite_digest: d("sensors.v1"),
            action_surface_digest: d("actions.v1"),
            capabilities: vec![
                PresenceCapability::Observe,
                PresenceCapability::Enter,
                PresenceCapability::Fork,
                PresenceCapability::Propose,
            ],
            authority_receipt_digest: None,
            entry_state_digest: entry,
            exit_state_digest: exit.clone(),
            entered_frame: Some(90),
            exited_frame: exit.map(|_| 100),
        }
    }

    #[test]
    fn snapshot_digest_binds_full_world_provenance() {
        let a = snapshot();
        let mut b = a.clone();
        b.world.creator_id = "different".into();
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }

    #[test]
    fn successor_requires_exact_previous_snapshot_digest() {
        let first = snapshot();
        let mut second = first.clone();
        second.snapshot_id = "snap-2".into();
        second.frame = Some(120);
        second.previous_snapshot_digest = Some(first.digest().unwrap());
        second.validate_successor(&first).unwrap();
        second.previous_snapshot_digest = Some(d("wrong.v1"));
        assert_eq!(
            second.validate_successor(&first),
            Err(WorldLifecycleError::PreviousSnapshotMismatch)
        );
    }

    #[test]
    fn lifecycle_transition_requires_external_authority() {
        let snap = snapshot();
        let receipt = WorldLifecycleReceipt {
            transition_id: "suspend-1".into(),
            world: snap.world.clone(),
            transition: WorldLifecycleTransition::Suspend,
            from_state: WorldLifecycleState::Active,
            to_state: WorldLifecycleState::Suspended,
            snapshot_digest: snap.digest().unwrap(),
            state_digest: snap.state_digest.clone(),
            frame: Some(100),
            authority_receipt_digest: None,
        };
        assert_eq!(
            receipt.validate_against_snapshot(&snap),
            Err(WorldLifecycleError::MissingAuthorityReceipt)
        );
    }

    #[test]
    fn resume_rejects_same_bytes_in_different_state_domain() {
        let snap = snapshot();
        let wrong_state = TypedDigest::new(
            "other-state.v1",
            snap.state_digest.algorithm.clone(),
            snap.state_digest.value.clone(),
        )
        .unwrap();
        let receipt = WorldLifecycleReceipt {
            transition_id: "resume-1".into(),
            world: snap.world.clone(),
            transition: WorldLifecycleTransition::Resume,
            from_state: WorldLifecycleState::Suspended,
            to_state: WorldLifecycleState::Active,
            snapshot_digest: snap.digest().unwrap(),
            state_digest: wrong_state,
            frame: Some(101),
            authority_receipt_digest: Some(d("authority.v1")),
        };
        assert_eq!(
            receipt.validate_against_snapshot(&snap),
            Err(WorldLifecycleError::StateContinuityMismatch)
        );
    }

    #[test]
    fn revisit_proves_exit_snapshot_entry_state_continuity() {
        let snap = snapshot();
        let prior = presence(
            "presence-a",
            d("older-state.v1"),
            Some(snap.state_digest.clone()),
        );
        let mut resumed = presence("presence-b", snap.state_digest.clone(), None);
        resumed.entered_frame = Some(101);
        let receipt = WorldRevisitReceipt::prove("revisit-1", &snap, &prior, &resumed).unwrap();
        assert_eq!(receipt.world, snap.world);
        assert!(receipt.digest().is_ok());
    }

    #[test]
    fn archived_world_is_not_a_valid_resume_source() {
        let (from, to) = WorldLifecycleTransition::Resume.expected_states();
        assert_eq!(from, WorldLifecycleState::Suspended);
        assert_eq!(to, WorldLifecycleState::Active);
        assert_ne!(from, WorldLifecycleState::Archived);
    }
}
