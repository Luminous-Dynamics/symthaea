// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit boundary for materializing a counterfactual into a committed digital world.
//!
//! A commit never changes the counterfactual world's provenance. Instead it
//! records that an authorized mutation of the parent committed world reproduced
//! a selected counterfactual state.

use serde::{Deserialize, Serialize};

use crate::types::{RealityLayer, WorldDescriptor, WorldRelation};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterfactualCommitReceipt {
    pub source_world: WorldDescriptor,
    pub target_world: WorldDescriptor,
    pub source_state_digest: String,
    pub target_before_state_digest: String,
    pub target_after_state_digest: String,
    /// Digest of the authority decision/capability that permitted the real
    /// mutation. The reality ledger does not mint that authority itself.
    pub authority_receipt_digest: String,
    pub actor_id: String,
}

impl CounterfactualCommitReceipt {
    pub fn validate(&self) -> Result<(), RealityCommitError> {
        self.source_world
            .validate()
            .map_err(|error| RealityCommitError::InvalidWorld(error.to_string()))?;
        self.target_world
            .validate()
            .map_err(|error| RealityCommitError::InvalidWorld(error.to_string()))?;
        if self.source_world.layer != RealityLayer::Counterfactual {
            return Err(RealityCommitError::SourceNotCounterfactual);
        }
        if self.target_world.layer != RealityLayer::DigitalCommitted {
            return Err(RealityCommitError::TargetNotDigitalCommitted);
        }
        let parent = self
            .source_world
            .parent
            .as_ref()
            .ok_or(RealityCommitError::SourceMissingParent)?;
        if parent.world_id != self.target_world.world_id
            || parent.lineage_id != self.target_world.lineage_id
            || parent.relation != WorldRelation::CounterfactualOf
        {
            return Err(RealityCommitError::SourceIsNotCounterfactualOfTarget);
        }
        if self.source_state_digest.trim().is_empty()
            || self.target_before_state_digest.trim().is_empty()
            || self.target_after_state_digest.trim().is_empty()
        {
            return Err(RealityCommitError::MissingStateDigest);
        }
        if self.authority_receipt_digest.trim().is_empty() {
            return Err(RealityCommitError::MissingAuthorityReceipt);
        }
        if self.actor_id.trim().is_empty() {
            return Err(RealityCommitError::MissingActor);
        }
        if self.source_state_digest != self.target_after_state_digest {
            return Err(RealityCommitError::CommittedStateDoesNotMatchCounterfactual);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RealityCommitError {
    #[error("invalid world descriptor: {0}")]
    InvalidWorld(String),
    #[error("commit source must remain explicitly Counterfactual")]
    SourceNotCounterfactual,
    #[error("v1 commit target must be DigitalCommitted")]
    TargetNotDigitalCommitted,
    #[error("counterfactual source is missing its parent")]
    SourceMissingParent,
    #[error("counterfactual source is not a branch of the target world")]
    SourceIsNotCounterfactualOfTarget,
    #[error("state digests must be non-empty")]
    MissingStateDigest,
    #[error("commit requires an external authority receipt digest")]
    MissingAuthorityReceipt,
    #[error("commit actor id may not be empty")]
    MissingActor,
    #[error("committed target state does not equal the selected counterfactual state")]
    CommittedStateDoesNotMatchCounterfactual,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        WorldId, WorldLineageId, WorldOrigin, WorldParentRef,
    };

    fn target() -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId("studio".into()),
            lineage_id: WorldLineageId("studio-lineage".into()),
            layer: RealityLayer::DigitalCommitted,
            origin: WorldOrigin::DigitalHost { host_kind: "bevy".into() },
            parent: None,
            generation_depth: 0,
            creator_id: "host".into(),
        }
    }

    fn source(target: &WorldDescriptor) -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId("ghost-a".into()),
            lineage_id: WorldLineageId("ghost-lineage".into()),
            layer: RealityLayer::Counterfactual,
            origin: WorldOrigin::CounterfactualBranch,
            parent: Some(WorldParentRef {
                world_id: target.world_id.clone(),
                lineage_id: target.lineage_id.clone(),
                relation: WorldRelation::CounterfactualOf,
            }),
            generation_depth: 1,
            creator_id: "ghost-engine".into(),
        }
    }

    #[test]
    fn authorized_matching_counterfactual_can_be_materialized_without_relabeling_it() {
        let target = target();
        let source = source(&target);
        let receipt = CounterfactualCommitReceipt {
            source_world: source.clone(),
            target_world: target,
            source_state_digest: "state-a".into(),
            target_before_state_digest: "state-base".into(),
            target_after_state_digest: "state-a".into(),
            authority_receipt_digest: "authority".into(),
            actor_id: "art-port".into(),
        };
        receipt.validate().unwrap();
        assert_eq!(source.layer, RealityLayer::Counterfactual);
    }

    #[test]
    fn dream_cannot_skip_into_committed_world_through_counterfactual_commit_gate() {
        let target = target();
        let mut source = source(&target);
        source.layer = RealityLayer::Dream;
        source.origin = WorldOrigin::DreamEngine;
        source.parent.as_mut().unwrap().relation = WorldRelation::DreamedFrom;
        let receipt = CounterfactualCommitReceipt {
            source_world: source,
            target_world: target,
            source_state_digest: "state-a".into(),
            target_before_state_digest: "state-base".into(),
            target_after_state_digest: "state-a".into(),
            authority_receipt_digest: "authority".into(),
            actor_id: "art-port".into(),
        };
        assert_eq!(receipt.validate(), Err(RealityCommitError::SourceNotCounterfactual));
    }
}
