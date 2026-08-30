// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Snapshot-bound world fork evidence.

use serde::{Deserialize, Serialize};

use crate::{
    digest::TypedDigest,
    lifecycle::WorldSnapshotManifest,
    types::{RealityLayer, WorldDescriptor, WorldRelation},
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldSnapshotForkReceipt {
    pub fork_id: String,
    pub source_snapshot_digest: TypedDigest,
    pub parent_world: WorldDescriptor,
    pub child_world: WorldDescriptor,
    pub child_initial_state_digest: TypedDigest,
    pub child_genesis_digest: TypedDigest,
    /// Persisting a child is authority-bearing; ephemeral counterfactual forks
    /// may remain unpersisted without such authority.
    pub persisted: bool,
    pub persist_authority_receipt_digest: Option<TypedDigest>,
}

impl WorldSnapshotForkReceipt {
    pub fn validate_against_snapshot(
        &self,
        snapshot: &WorldSnapshotManifest,
    ) -> Result<(), WorldSnapshotForkError> {
        if self.fork_id.trim().is_empty() {
            return Err(WorldSnapshotForkError::MissingForkId);
        }
        snapshot
            .validate()
            .map_err(|error| WorldSnapshotForkError::Lifecycle(error.to_string()))?;
        self.parent_world
            .validate()
            .map_err(|error| WorldSnapshotForkError::InvalidWorld(error.to_string()))?;
        self.child_world
            .validate()
            .map_err(|error| WorldSnapshotForkError::InvalidWorld(error.to_string()))?;
        for digest in [
            &self.source_snapshot_digest,
            &self.child_initial_state_digest,
            &self.child_genesis_digest,
        ] {
            digest
                .validate()
                .map_err(|error| WorldSnapshotForkError::InvalidDigest(error.to_string()))?;
        }
        if let Some(authority) = &self.persist_authority_receipt_digest {
            authority
                .validate()
                .map_err(|error| WorldSnapshotForkError::InvalidDigest(error.to_string()))?;
        }
        if self.persisted && self.persist_authority_receipt_digest.is_none() {
            return Err(WorldSnapshotForkError::MissingPersistAuthority);
        }
        if self.parent_world != snapshot.world {
            return Err(WorldSnapshotForkError::ParentWorldMismatch);
        }
        if !self
            .source_snapshot_digest
            .same_typed_value(&snapshot.digest().map_err(|error| {
                WorldSnapshotForkError::Lifecycle(error.to_string())
            })?)
        {
            return Err(WorldSnapshotForkError::SnapshotDigestMismatch);
        }
        if !self
            .child_initial_state_digest
            .same_typed_value(&snapshot.state_digest)
        {
            return Err(WorldSnapshotForkError::InitialStateMismatch);
        }
        let parent = self
            .child_world
            .parent
            .as_ref()
            .ok_or(WorldSnapshotForkError::ChildMissingParent)?;
        if parent.world_id != snapshot.world.world_id
            || parent.lineage_id != snapshot.world.lineage_id
            || self.child_world.generation_depth != snapshot.world.generation_depth + 1
        {
            return Err(WorldSnapshotForkError::ChildParentMismatch);
        }
        let relation_ok = matches!(
            (self.child_world.layer, &parent.relation),
            (RealityLayer::Counterfactual, WorldRelation::CounterfactualOf)
                | (RealityLayer::DigitalCommitted, WorldRelation::SpawnedFrom)
        );
        if !relation_ok {
            return Err(WorldSnapshotForkError::UnsupportedForkRelation);
        }
        if self.child_world.world_id == snapshot.world.world_id
            && self.child_world.lineage_id == snapshot.world.lineage_id
        {
            return Err(WorldSnapshotForkError::ChildReusesParentIdentity);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorldSnapshotForkError {
    #[error("fork id may not be empty")]
    MissingForkId,
    #[error("lifecycle snapshot rejected fork: {0}")]
    Lifecycle(String),
    #[error("invalid world descriptor: {0}")]
    InvalidWorld(String),
    #[error("invalid typed digest: {0}")]
    InvalidDigest(String),
    #[error("persisted fork requires an external persist authority receipt")]
    MissingPersistAuthority,
    #[error("fork parent does not exactly match the source snapshot world")]
    ParentWorldMismatch,
    #[error("fork references a different source snapshot")]
    SnapshotDigestMismatch,
    #[error("fork child initial state differs from the source snapshot state")]
    InitialStateMismatch,
    #[error("fork child has no parent reference")]
    ChildMissingParent,
    #[error("fork child parent/depth does not match the source snapshot")]
    ChildParentMismatch,
    #[error("fork relation/layer must be CounterfactualOf or committed SpawnedFrom")]
    UnsupportedForkRelation,
    #[error("fork child may not reuse the exact parent world/lineage identity")]
    ChildReusesParentIdentity,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        digest::TypedDigest,
        lifecycle::WorldSnapshotManifest,
        types::{WorldId, WorldLineageId, WorldOrigin, WorldParentRef},
    };

    fn d(domain: &str) -> TypedDigest {
        TypedDigest::blake3(domain, domain.as_bytes()).unwrap()
    }

    fn parent_world() -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId("garden".into()),
            lineage_id: WorldLineageId("garden-lineage".into()),
            layer: RealityLayer::DigitalCommitted,
            origin: WorldOrigin::DigitalHost { host_kind: "symtropy".into() },
            parent: None,
            generation_depth: 0,
            creator_id: "symthaea".into(),
        }
    }

    fn snapshot() -> WorldSnapshotManifest {
        WorldSnapshotManifest {
            schema_version: 1,
            snapshot_id: "snap".into(),
            world: parent_world(),
            genesis_digest: d("genesis.v1"),
            state_digest: d("state.v1"),
            ledger_head_digest: d("ledger.v1"),
            host_artifact_digest: d("artifact.v1"),
            frame: Some(10),
            previous_snapshot_digest: None,
        }
    }

    fn child(layer: RealityLayer, relation: WorldRelation) -> WorldDescriptor {
        let parent = parent_world();
        WorldDescriptor {
            world_id: WorldId("child".into()),
            lineage_id: WorldLineageId("child-lineage".into()),
            layer,
            origin: if layer == RealityLayer::Counterfactual {
                WorldOrigin::CounterfactualBranch
            } else {
                WorldOrigin::DigitalHost { host_kind: "symtropy".into() }
            },
            parent: Some(WorldParentRef {
                world_id: parent.world_id,
                lineage_id: parent.lineage_id,
                relation,
            }),
            generation_depth: 1,
            creator_id: "symthaea".into(),
        }
    }

    #[test]
    fn ephemeral_counterfactual_fork_needs_no_persist_authority() {
        let snapshot = snapshot();
        let receipt = WorldSnapshotForkReceipt {
            fork_id: "fork-a".into(),
            source_snapshot_digest: snapshot.digest().unwrap(),
            parent_world: snapshot.world.clone(),
            child_world: child(RealityLayer::Counterfactual, WorldRelation::CounterfactualOf),
            child_initial_state_digest: snapshot.state_digest.clone(),
            child_genesis_digest: d("child-genesis.v1"),
            persisted: false,
            persist_authority_receipt_digest: None,
        };
        receipt.validate_against_snapshot(&snapshot).unwrap();
    }

    #[test]
    fn persisted_committed_fork_requires_authority() {
        let snapshot = snapshot();
        let mut receipt = WorldSnapshotForkReceipt {
            fork_id: "fork-b".into(),
            source_snapshot_digest: snapshot.digest().unwrap(),
            parent_world: snapshot.world.clone(),
            child_world: child(RealityLayer::DigitalCommitted, WorldRelation::SpawnedFrom),
            child_initial_state_digest: snapshot.state_digest.clone(),
            child_genesis_digest: d("child-genesis.v1"),
            persisted: true,
            persist_authority_receipt_digest: None,
        };
        assert_eq!(
            receipt.validate_against_snapshot(&snapshot),
            Err(WorldSnapshotForkError::MissingPersistAuthority)
        );
        receipt.persist_authority_receipt_digest = Some(d("authority.v1"));
        receipt.validate_against_snapshot(&snapshot).unwrap();
    }
}
