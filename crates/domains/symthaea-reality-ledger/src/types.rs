// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed provenance for worlds, simulations, dreams, replays and physical observations.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorldId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorldLineageId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RealityRecordId(pub String);

impl WorldId {
    pub fn validate(&self) -> Result<(), RealityTypeError> {
        validate_id(&self.0, RealityTypeError::EmptyWorldId)
    }
}

impl WorldLineageId {
    pub fn validate(&self) -> Result<(), RealityTypeError> {
        validate_id(&self.0, RealityTypeError::EmptyLineageId)
    }
}

impl RealityRecordId {
    pub fn validate(&self) -> Result<(), RealityTypeError> {
        validate_id(&self.0, RealityTypeError::EmptyRecordId)
    }
}

fn validate_id(value: &str, empty: RealityTypeError) -> Result<(), RealityTypeError> {
    if value.trim().is_empty() {
        return Err(empty);
    }
    if value.len() > 512 {
        return Err(RealityTypeError::IdentifierTooLong);
    }
    Ok(())
}

/// What kind of reality a world represents.
///
/// These labels are provenance classes, not claims about consciousness or
/// metaphysical status. In particular, `DigitalCommitted` means authoritative
/// within a declared digital world, not physically real.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RealityLayer {
    /// Evidence originating in independently identified physical sensors or
    /// instruments. The evidence may still be noisy or wrong.
    PhysicalGrounded,
    /// A persistent authoritative digital/simulated world.
    DigitalCommitted,
    /// A hypothetical branch that must never be silently remembered as having
    /// happened in its parent world.
    Counterfactual,
    /// A replay or reconstruction of a prior world state.
    Replay,
    /// An internally generated dream/imagination world.
    Dream,
    /// A world imported from an external author/source whose grounding is not
    /// established by this ledger.
    Imported,
    /// The layer could not be resolved. Unknown is never treated as grounded.
    Unknown,
}

impl RealityLayer {
    pub fn is_hypothetical(self) -> bool {
        matches!(self, Self::Counterfactual | Self::Dream)
    }

    pub fn is_committed_world(self) -> bool {
        matches!(self, Self::PhysicalGrounded | Self::DigitalCommitted)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorldOrigin {
    PhysicalSensorium,
    DigitalHost { host_kind: String },
    CounterfactualBranch,
    ReplayArtifact,
    DreamEngine,
    ImportedExternal { source: String },
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorldRelation {
    CounterfactualOf,
    ReplayOf,
    DreamedFrom,
    ImportedFrom,
    SpawnedFrom,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorldParentRef {
    pub world_id: WorldId,
    pub lineage_id: WorldLineageId,
    pub relation: WorldRelation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldDescriptor {
    pub world_id: WorldId,
    pub lineage_id: WorldLineageId,
    pub layer: RealityLayer,
    pub origin: WorldOrigin,
    pub parent: Option<WorldParentRef>,
    pub generation_depth: u32,
    /// Stable identity of the process/agent/operator that instantiated the
    /// descriptor. This is provenance only, not authority.
    pub creator_id: String,
}

impl WorldDescriptor {
    pub fn validate(&self) -> Result<(), RealityTypeError> {
        self.world_id.validate()?;
        self.lineage_id.validate()?;
        if self.creator_id.trim().is_empty() {
            return Err(RealityTypeError::EmptyCreatorId);
        }
        if self.creator_id.len() > 512 {
            return Err(RealityTypeError::IdentifierTooLong);
        }
        match (self.layer, self.parent.as_ref()) {
            (RealityLayer::Counterfactual, None)
            | (RealityLayer::Replay, None)
            | (RealityLayer::Dream, None) => return Err(RealityTypeError::DerivedWorldMissingParent),
            _ => {}
        }
        if self.parent.is_none() && self.generation_depth != 0 {
            return Err(RealityTypeError::RootWorldHasNonzeroDepth);
        }
        if self.parent.is_some() && self.generation_depth == 0 {
            return Err(RealityTypeError::DerivedWorldHasZeroDepth);
        }
        if let Some(parent) = &self.parent {
            parent.world_id.validate()?;
            parent.lineage_id.validate()?;
            if parent.world_id == self.world_id && parent.lineage_id == self.lineage_id {
                return Err(RealityTypeError::SelfParent);
            }
            let relation_matches_layer = matches!(
                (self.layer, &parent.relation),
                (RealityLayer::Counterfactual, WorldRelation::CounterfactualOf)
                    | (RealityLayer::Replay, WorldRelation::ReplayOf)
                    | (RealityLayer::Dream, WorldRelation::DreamedFrom)
                    | (RealityLayer::Imported, WorldRelation::ImportedFrom)
                    | (RealityLayer::DigitalCommitted, WorldRelation::SpawnedFrom)
                    | (RealityLayer::Unknown, WorldRelation::SpawnedFrom)
                    | (RealityLayer::PhysicalGrounded, WorldRelation::SpawnedFrom)
            );
            if !relation_matches_layer {
                return Err(RealityTypeError::LayerRelationMismatch);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceSource {
    PhysicalSensor { sensor_id: String },
    DigitalWorldObservation { host_id: String },
    CounterfactualSimulation { engine_id: String },
    Replay { artifact_id: String },
    DreamGeneration { engine_id: String },
    Imported { source_id: String },
    DerivedComputation { processor_id: String },
    Unknown,
}

impl EvidenceSource {
    pub fn validate(&self) -> Result<(), RealityTypeError> {
        let id = match self {
            Self::PhysicalSensor { sensor_id } => Some(sensor_id),
            Self::DigitalWorldObservation { host_id } => Some(host_id),
            Self::CounterfactualSimulation { engine_id } => Some(engine_id),
            Self::Replay { artifact_id } => Some(artifact_id),
            Self::DreamGeneration { engine_id } => Some(engine_id),
            Self::Imported { source_id } => Some(source_id),
            Self::DerivedComputation { processor_id } => Some(processor_id),
            Self::Unknown => None,
        };
        if id.is_some_and(|id| id.trim().is_empty()) {
            return Err(RealityTypeError::EmptyEvidenceSourceId);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RealityTypeError {
    #[error("world id may not be empty")]
    EmptyWorldId,
    #[error("world lineage id may not be empty")]
    EmptyLineageId,
    #[error("reality record id may not be empty")]
    EmptyRecordId,
    #[error("creator id may not be empty")]
    EmptyCreatorId,
    #[error("evidence source id may not be empty")]
    EmptyEvidenceSourceId,
    #[error("identifier exceeds the supported length")]
    IdentifierTooLong,
    #[error("counterfactual, replay and dream worlds require a parent")]
    DerivedWorldMissingParent,
    #[error("root worlds must have generation depth zero")]
    RootWorldHasNonzeroDepth,
    #[error("derived worlds must have nonzero generation depth")]
    DerivedWorldHasZeroDepth,
    #[error("a world may not be its own parent")]
    SelfParent,
    #[error("world layer and parent relation are inconsistent")]
    LayerRelationMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> WorldDescriptor {
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

    #[test]
    fn committed_root_is_valid() {
        root().validate().unwrap();
    }

    #[test]
    fn counterfactual_requires_parent_and_matching_relation() {
        let mut child = root();
        child.world_id = WorldId("ghost-a".into());
        child.layer = RealityLayer::Counterfactual;
        child.generation_depth = 1;
        assert_eq!(child.validate(), Err(RealityTypeError::DerivedWorldMissingParent));
        child.parent = Some(WorldParentRef {
            world_id: WorldId("studio".into()),
            lineage_id: WorldLineageId("studio-lineage".into()),
            relation: WorldRelation::CounterfactualOf,
        });
        child.validate().unwrap();
    }

    #[test]
    fn unknown_is_not_committed_or_hypothetical() {
        assert!(!RealityLayer::Unknown.is_committed_world());
        assert!(!RealityLayer::Unknown.is_hypothetical());
    }
}
