// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ordered lifecycle state machine for one exact persisted world snapshot.

use serde::{Deserialize, Serialize};

use crate::{
    digest::{DigestAlgorithm, TypedDigest},
    lifecycle::{
        WorldLifecycleError, WorldLifecycleReceipt, WorldLifecycleState, WorldSnapshotManifest,
    },
    types::WorldDescriptor,
};

pub const WORLD_LIFECYCLE_TIMELINE_DIGEST_DOMAIN: &str = "symthaea.world-lifecycle-timeline.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldLifecycleTimeline {
    pub world: WorldDescriptor,
    pub snapshot_digest: TypedDigest,
    pub current_state: WorldLifecycleState,
    receipts: Vec<WorldLifecycleReceipt>,
}

impl WorldLifecycleTimeline {
    pub fn new(snapshot: &WorldSnapshotManifest) -> Result<Self, WorldLifecycleTimelineError> {
        snapshot
            .validate()
            .map_err(WorldLifecycleTimelineError::Lifecycle)?;
        Ok(Self {
            world: snapshot.world.clone(),
            snapshot_digest: snapshot
                .digest()
                .map_err(WorldLifecycleTimelineError::Lifecycle)?,
            current_state: WorldLifecycleState::Active,
            receipts: Vec::new(),
        })
    }

    pub fn receipts(&self) -> &[WorldLifecycleReceipt] {
        &self.receipts
    }

    pub fn append(
        &mut self,
        snapshot: &WorldSnapshotManifest,
        receipt: WorldLifecycleReceipt,
    ) -> Result<(), WorldLifecycleTimelineError> {
        if self.world != snapshot.world || receipt.world != self.world {
            return Err(WorldLifecycleTimelineError::WorldMismatch);
        }
        let snapshot_digest = snapshot
            .digest()
            .map_err(WorldLifecycleTimelineError::Lifecycle)?;
        if !self.snapshot_digest.same_typed_value(&snapshot_digest) {
            return Err(WorldLifecycleTimelineError::SnapshotMismatch);
        }
        receipt
            .validate_against_snapshot(snapshot)
            .map_err(WorldLifecycleTimelineError::Lifecycle)?;
        if receipt.from_state != self.current_state {
            return Err(WorldLifecycleTimelineError::StateMachineMismatch {
                expected: self.current_state,
                actual: receipt.from_state,
            });
        }
        self.current_state = receipt.to_state;
        self.receipts.push(receipt);
        Ok(())
    }

    pub fn verify(
        &self,
        snapshot: &WorldSnapshotManifest,
    ) -> Result<(), WorldLifecycleTimelineError> {
        if self.world != snapshot.world {
            return Err(WorldLifecycleTimelineError::WorldMismatch);
        }
        let expected_snapshot = snapshot
            .digest()
            .map_err(WorldLifecycleTimelineError::Lifecycle)?;
        if !self.snapshot_digest.same_typed_value(&expected_snapshot) {
            return Err(WorldLifecycleTimelineError::SnapshotMismatch);
        }
        let mut state = WorldLifecycleState::Active;
        for receipt in &self.receipts {
            receipt
                .validate_against_snapshot(snapshot)
                .map_err(WorldLifecycleTimelineError::Lifecycle)?;
            if receipt.from_state != state {
                return Err(WorldLifecycleTimelineError::StateMachineMismatch {
                    expected: state,
                    actual: receipt.from_state,
                });
            }
            state = receipt.to_state;
        }
        if state != self.current_state {
            return Err(WorldLifecycleTimelineError::TerminalStateMismatch);
        }
        Ok(())
    }

    pub fn digest(
        &self,
        snapshot: &WorldSnapshotManifest,
    ) -> Result<TypedDigest, WorldLifecycleTimelineError> {
        self.verify(snapshot)?;
        let mut hasher = blake3::Hasher::new();
        feed(&mut hasher, b"symthaea.world-lifecycle-timeline.v1");
        feed(&mut hasher, self.world.world_id.0.as_bytes());
        feed(&mut hasher, self.world.lineage_id.0.as_bytes());
        feed_typed_digest(&mut hasher, &self.snapshot_digest);
        hasher.update(&[state_tag(self.current_state)]);
        hasher.update(&(self.receipts.len() as u64).to_le_bytes());
        for receipt in &self.receipts {
            feed(&mut hasher, receipt.transition_id.as_bytes());
            hasher.update(&[transition_tag(receipt.transition)]);
            hasher.update(&[state_tag(receipt.from_state)]);
            hasher.update(&[state_tag(receipt.to_state)]);
            feed_typed_digest(&mut hasher, &receipt.snapshot_digest);
            feed_typed_digest(&mut hasher, &receipt.state_digest);
            match receipt.frame {
                Some(frame) => {
                    hasher.update(&[1]);
                    hasher.update(&frame.to_le_bytes());
                }
                None => {
                    hasher.update(&[0]);
                }
            }
            if let Some(authority) = &receipt.authority_receipt_digest {
                hasher.update(&[1]);
                feed_typed_digest(&mut hasher, authority);
            } else {
                hasher.update(&[0]);
            }
        }
        TypedDigest::new(
            WORLD_LIFECYCLE_TIMELINE_DIGEST_DOMAIN,
            DigestAlgorithm::Blake3,
            hasher.finalize().to_hex().to_string(),
        )
        .map_err(|error| WorldLifecycleTimelineError::Digest(error.to_string()))
    }
}

fn feed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
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

fn state_tag(state: WorldLifecycleState) -> u8 {
    match state {
        WorldLifecycleState::Active => 0,
        WorldLifecycleState::Suspended => 1,
        WorldLifecycleState::Archived => 2,
    }
}

fn transition_tag(transition: crate::lifecycle::WorldLifecycleTransition) -> u8 {
    match transition {
        crate::lifecycle::WorldLifecycleTransition::Suspend => 0,
        crate::lifecycle::WorldLifecycleTransition::Resume => 1,
        crate::lifecycle::WorldLifecycleTransition::Archive => 2,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorldLifecycleTimelineError {
    #[error("lifecycle receipt rejected: {0}")]
    Lifecycle(#[from] WorldLifecycleError),
    #[error("lifecycle timeline world differs from snapshot/receipt world")]
    WorldMismatch,
    #[error("lifecycle timeline is bound to a different snapshot")]
    SnapshotMismatch,
    #[error("lifecycle transition started from {actual:?}, expected {expected:?}")]
    StateMachineMismatch {
        expected: WorldLifecycleState,
        actual: WorldLifecycleState,
    },
    #[error("lifecycle timeline terminal state does not match replayed transitions")]
    TerminalStateMismatch,
    #[error("invalid lifecycle timeline digest: {0}")]
    Digest(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lifecycle::{WorldLifecycleTransition, WorldSnapshotManifest},
        types::{RealityLayer, WorldId, WorldLineageId, WorldOrigin},
    };

    fn d(domain: &str) -> TypedDigest {
        TypedDigest::blake3(domain, domain.as_bytes()).unwrap()
    }

    fn snapshot() -> WorldSnapshotManifest {
        WorldSnapshotManifest {
            schema_version: 1,
            snapshot_id: "snap".into(),
            world: WorldDescriptor {
                world_id: WorldId("garden".into()),
                lineage_id: WorldLineageId("garden-lineage".into()),
                layer: RealityLayer::DigitalCommitted,
                origin: WorldOrigin::DigitalHost { host_kind: "symtropy".into() },
                parent: None,
                generation_depth: 0,
                creator_id: "symthaea".into(),
            },
            genesis_digest: d("genesis.v1"),
            state_digest: d("state.v1"),
            ledger_head_digest: d("ledger.v1"),
            host_artifact_digest: d("artifact.v1"),
            frame: Some(10),
            previous_snapshot_digest: None,
        }
    }

    fn receipt(
        snapshot: &WorldSnapshotManifest,
        transition: WorldLifecycleTransition,
        id: &str,
    ) -> WorldLifecycleReceipt {
        let (from_state, to_state) = transition.expected_states();
        WorldLifecycleReceipt {
            transition_id: id.into(),
            world: snapshot.world.clone(),
            transition,
            from_state,
            to_state,
            snapshot_digest: snapshot.digest().unwrap(),
            state_digest: snapshot.state_digest.clone(),
            frame: snapshot.frame,
            authority_receipt_digest: Some(d("authority.v1")),
        }
    }

    #[test]
    fn suspend_resume_sequence_verifies() {
        let snapshot = snapshot();
        let mut timeline = WorldLifecycleTimeline::new(&snapshot).unwrap();
        timeline
            .append(&snapshot, receipt(&snapshot, WorldLifecycleTransition::Suspend, "s"))
            .unwrap();
        timeline
            .append(&snapshot, receipt(&snapshot, WorldLifecycleTransition::Resume, "r"))
            .unwrap();
        assert_eq!(timeline.current_state, WorldLifecycleState::Active);
        timeline.verify(&snapshot).unwrap();
        assert!(timeline.digest(&snapshot).is_ok());
    }

    #[test]
    fn archived_world_cannot_resume() {
        let snapshot = snapshot();
        let mut timeline = WorldLifecycleTimeline::new(&snapshot).unwrap();
        timeline
            .append(&snapshot, receipt(&snapshot, WorldLifecycleTransition::Suspend, "s"))
            .unwrap();
        timeline
            .append(&snapshot, receipt(&snapshot, WorldLifecycleTransition::Archive, "a"))
            .unwrap();
        assert_eq!(timeline.current_state, WorldLifecycleState::Archived);
        assert!(timeline
            .append(&snapshot, receipt(&snapshot, WorldLifecycleTransition::Resume, "r"))
            .is_err());
    }
}
