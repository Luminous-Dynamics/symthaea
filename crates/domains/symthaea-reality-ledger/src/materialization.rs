// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed-digest materialization gate for promoting a selected counterfactual
//! into an externally authorized committed digital-world mutation.

use serde::{Deserialize, Serialize};

use crate::{digest::TypedDigest, types::{RealityLayer, WorldDescriptor, WorldRelation}};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypedCounterfactualCommitReceipt {
    pub source_world: WorldDescriptor,
    pub target_world: WorldDescriptor,
    pub source_state_digest: TypedDigest,
    pub target_before_state_digest: TypedDigest,
    pub target_after_state_digest: TypedDigest,
    pub authority_receipt_digest: TypedDigest,
    pub actor_id: String,
}

impl TypedCounterfactualCommitReceipt {
    pub fn validate(&self) -> Result<(), TypedMaterializationError> {
        self.source_world
            .validate()
            .map_err(|error| TypedMaterializationError::InvalidWorld(error.to_string()))?;
        self.target_world
            .validate()
            .map_err(|error| TypedMaterializationError::InvalidWorld(error.to_string()))?;
        for digest in [
            &self.source_state_digest,
            &self.target_before_state_digest,
            &self.target_after_state_digest,
            &self.authority_receipt_digest,
        ] {
            digest
                .validate()
                .map_err(|error| TypedMaterializationError::InvalidDigest(error.to_string()))?;
        }
        if self.actor_id.trim().is_empty() {
            return Err(TypedMaterializationError::MissingActor);
        }
        if self.source_world.layer != RealityLayer::Counterfactual {
            return Err(TypedMaterializationError::SourceNotCounterfactual);
        }
        if self.target_world.layer != RealityLayer::DigitalCommitted {
            return Err(TypedMaterializationError::TargetNotDigitalCommitted);
        }
        let parent = self
            .source_world
            .parent
            .as_ref()
            .ok_or(TypedMaterializationError::SourceMissingParent)?;
        if parent.world_id != self.target_world.world_id
            || parent.lineage_id != self.target_world.lineage_id
            || parent.relation != WorldRelation::CounterfactualOf
        {
            return Err(TypedMaterializationError::WrongTargetParent);
        }
        if !self
            .source_state_digest
            .same_typed_value(&self.target_after_state_digest)
        {
            return Err(TypedMaterializationError::CommittedStateMismatch);
        }
        if !same_state_type(
            &self.target_before_state_digest,
            &self.target_after_state_digest,
        ) {
            return Err(TypedMaterializationError::TargetStateTypeChanged);
        }
        Ok(())
    }
}

fn same_state_type(a: &TypedDigest, b: &TypedDigest) -> bool {
    a.domain == b.domain && a.algorithm == b.algorithm
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TypedMaterializationError {
    #[error("invalid world descriptor: {0}")]
    InvalidWorld(String),
    #[error("invalid typed digest: {0}")]
    InvalidDigest(String),
    #[error("materialization actor id may not be empty")]
    MissingActor,
    #[error("materialization source must remain Counterfactual")]
    SourceNotCounterfactual,
    #[error("materialization target must be DigitalCommitted")]
    TargetNotDigitalCommitted,
    #[error("counterfactual source is missing parent provenance")]
    SourceMissingParent,
    #[error("counterfactual source is not a branch of the target world")]
    WrongTargetParent,
    #[error("committed after-state does not exactly equal selected counterfactual state")]
    CommittedStateMismatch,
    #[error("target before/after digests changed state domain or digest algorithm")]
    TargetStateTypeChanged,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{digest::{DigestAlgorithm, TypedDigest}, types::{WorldId, WorldLineageId, WorldOrigin, WorldParentRef}};

    fn d(domain: &str, value: &str) -> TypedDigest {
        TypedDigest::new(domain, DigestAlgorithm::Blake3, value).unwrap()
    }

    fn target() -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId("studio".into()),
            lineage_id: WorldLineageId("studio-lineage".into()),
            layer: RealityLayer::DigitalCommitted,
            origin: WorldOrigin::DigitalHost { host_kind: "symtropy".into() },
            parent: None,
            generation_depth: 0,
            creator_id: "host".into(),
        }
    }

    fn source(target: &WorldDescriptor) -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId("ghost-a".into()),
            lineage_id: WorldLineageId("ghost-a-lineage".into()),
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
    fn identical_hex_in_wrong_domain_cannot_materialize() {
        let target = target();
        let receipt = TypedCounterfactualCommitReceipt {
            source_world: source(&target),
            target_world: target,
            source_state_digest: d("symtropy.scene-state.v1", "same"),
            target_before_state_digest: d("symtropy.scene-state.v1", "before"),
            target_after_state_digest: d("some.other.state.v1", "same"),
            authority_receipt_digest: d("authority.receipt.v1", "authority"),
            actor_id: "art-port".into(),
        };
        assert_eq!(receipt.validate(), Err(TypedMaterializationError::CommittedStateMismatch));
    }
}
