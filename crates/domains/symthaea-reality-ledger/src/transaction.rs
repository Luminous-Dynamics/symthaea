// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transactional alignment contract for multi-plane world observations.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{digest::TypedDigest, types::{WorldDescriptor, WorldId, WorldLineageId}};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ObservationPlane {
    Color,
    Depth,
    ObjectId,
    Motion,
    Audio,
    SemanticScene,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationArtifactReceipt {
    pub plane: ObservationPlane,
    pub world_id: WorldId,
    pub lineage_id: WorldLineageId,
    pub revision_id: String,
    pub frame: u64,
    pub state_digest: TypedDigest,
    pub artifact_digest: TypedDigest,
    pub camera_id: Option<String>,
    pub fidelity_id: Option<String>,
}

impl ObservationArtifactReceipt {
    pub fn validate(&self) -> Result<(), ObservationTransactionError> {
        self.world_id
            .validate()
            .map_err(|error| ObservationTransactionError::InvalidIdentity(error.to_string()))?;
        self.lineage_id
            .validate()
            .map_err(|error| ObservationTransactionError::InvalidIdentity(error.to_string()))?;
        if self.revision_id.trim().is_empty() {
            return Err(ObservationTransactionError::MissingRevision);
        }
        self.state_digest
            .validate()
            .map_err(|error| ObservationTransactionError::InvalidDigest(error.to_string()))?;
        self.artifact_digest
            .validate()
            .map_err(|error| ObservationTransactionError::InvalidDigest(error.to_string()))?;
        if let ObservationPlane::Custom(name) = &self.plane {
            if name.trim().is_empty() {
                return Err(ObservationTransactionError::EmptyCustomPlane);
            }
        }
        for value in [self.camera_id.as_deref(), self.fidelity_id.as_deref()].into_iter().flatten() {
            if value.trim().is_empty() {
                return Err(ObservationTransactionError::EmptyOptionalIdentity);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldObservationBundle {
    pub bundle_id: String,
    pub world: WorldDescriptor,
    pub revision_id: String,
    pub frame: u64,
    pub state_digest: TypedDigest,
    pub camera_id: Option<String>,
    pub fidelity_id: Option<String>,
    pub required_planes: Vec<ObservationPlane>,
    pub receipts: Vec<ObservationArtifactReceipt>,
}

impl WorldObservationBundle {
    pub fn validate(&self) -> Result<(), ObservationTransactionError> {
        if self.bundle_id.trim().is_empty() {
            return Err(ObservationTransactionError::MissingBundleId);
        }
        self.world
            .validate()
            .map_err(|error| ObservationTransactionError::InvalidWorld(error.to_string()))?;
        if self.revision_id.trim().is_empty() {
            return Err(ObservationTransactionError::MissingRevision);
        }
        self.state_digest
            .validate()
            .map_err(|error| ObservationTransactionError::InvalidDigest(error.to_string()))?;

        let mut required = BTreeSet::new();
        for plane in &self.required_planes {
            if let ObservationPlane::Custom(name) = plane {
                if name.trim().is_empty() {
                    return Err(ObservationTransactionError::EmptyCustomPlane);
                }
            }
            if !required.insert(plane.clone()) {
                return Err(ObservationTransactionError::DuplicateRequiredPlane);
            }
        }
        if required.is_empty() {
            return Err(ObservationTransactionError::NoRequiredPlanes);
        }

        let mut observed = BTreeSet::new();
        for receipt in &self.receipts {
            receipt.validate()?;
            if !observed.insert(receipt.plane.clone()) {
                return Err(ObservationTransactionError::DuplicateObservedPlane);
            }
            if receipt.world_id != self.world.world_id
                || receipt.lineage_id != self.world.lineage_id
                || receipt.revision_id != self.revision_id
                || receipt.frame != self.frame
                || !receipt.state_digest.same_typed_value(&self.state_digest)
                || receipt.camera_id != self.camera_id
                || receipt.fidelity_id != self.fidelity_id
            {
                return Err(ObservationTransactionError::PlaneMisalignment);
            }
        }

        if !required.is_subset(&observed) {
            return Err(ObservationTransactionError::MissingRequiredPlane);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ObservationTransactionError {
    #[error("bundle id may not be empty")]
    MissingBundleId,
    #[error("invalid world descriptor: {0}")]
    InvalidWorld(String),
    #[error("invalid identity: {0}")]
    InvalidIdentity(String),
    #[error("revision id may not be empty")]
    MissingRevision,
    #[error("invalid typed digest: {0}")]
    InvalidDigest(String),
    #[error("custom observation plane name may not be empty")]
    EmptyCustomPlane,
    #[error("optional camera/fidelity identity may not be empty when present")]
    EmptyOptionalIdentity,
    #[error("required planes may not contain duplicates")]
    DuplicateRequiredPlane,
    #[error("observation bundle must require at least one plane")]
    NoRequiredPlanes,
    #[error("receipts may not contain duplicate planes")]
    DuplicateObservedPlane,
    #[error("observation plane does not match the bundle world/revision/frame/state/camera/fidelity")]
    PlaneMisalignment,
    #[error("one or more required observation planes are missing")]
    MissingRequiredPlane,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{digest::TypedDigest, types::{RealityLayer, WorldId, WorldLineageId, WorldOrigin}};

    fn d(domain: &str, value: &str) -> TypedDigest {
        TypedDigest::new(domain, crate::digest::DigestAlgorithm::Blake3, value).unwrap()
    }

    fn world() -> WorldDescriptor {
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

    fn receipt(plane: ObservationPlane, state: &TypedDigest) -> ObservationArtifactReceipt {
        ObservationArtifactReceipt {
            plane,
            world_id: WorldId("studio".into()),
            lineage_id: WorldLineageId("studio-lineage".into()),
            revision_id: "r1".into(),
            frame: 10,
            state_digest: state.clone(),
            artifact_digest: d("artifact.v1", "artifact"),
            camera_id: Some("camera-a".into()),
            fidelity_id: Some("cognitive".into()),
        }
    }

    #[test]
    fn mixed_state_planes_fail_closed() {
        let state = d("symtropy.scene-state.v1", "state-a");
        let mut depth = receipt(ObservationPlane::Depth, &state);
        depth.state_digest = d("symtropy.scene-state.v1", "state-b");
        let bundle = WorldObservationBundle {
            bundle_id: "bundle".into(),
            world: world(),
            revision_id: "r1".into(),
            frame: 10,
            state_digest: state.clone(),
            camera_id: Some("camera-a".into()),
            fidelity_id: Some("cognitive".into()),
            required_planes: vec![ObservationPlane::Color, ObservationPlane::Depth],
            receipts: vec![receipt(ObservationPlane::Color, &state), depth],
        };
        assert_eq!(bundle.validate(), Err(ObservationTransactionError::PlaneMisalignment));
    }

    #[test]
    fn missing_required_plane_is_not_silently_accepted() {
        let state = d("symtropy.scene-state.v1", "state-a");
        let bundle = WorldObservationBundle {
            bundle_id: "bundle".into(),
            world: world(),
            revision_id: "r1".into(),
            frame: 10,
            state_digest: state.clone(),
            camera_id: Some("camera-a".into()),
            fidelity_id: Some("cognitive".into()),
            required_planes: vec![ObservationPlane::Color, ObservationPlane::Depth],
            receipts: vec![receipt(ObservationPlane::Color, &state)],
        };
        assert_eq!(bundle.validate(), Err(ObservationTransactionError::MissingRequiredPlane));
    }
}
