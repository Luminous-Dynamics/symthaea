// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Claim-level grounding independent of the reality layer containing the claim.

use serde::{Deserialize, Serialize};

use crate::{digest::TypedDigest, types::{EvidenceSource, RealityLayer, WorldDescriptor}};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClaimGrounding {
    DirectWorldObservation,
    DerivedInference,
    Simulated,
    Replayed,
    Dreamed,
    Imported,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealityClaimEnvelope {
    pub world: WorldDescriptor,
    pub grounding: ClaimGrounding,
    pub source: EvidenceSource,
    /// Digest of the proposition/claim representation.
    pub statement_digest: TypedDigest,
    /// Digest of the evidence payload used to support the claim.
    pub evidence_digest: TypedDigest,
}

impl RealityClaimEnvelope {
    pub fn validate(&self) -> Result<(), RealityClaimError> {
        self.world
            .validate()
            .map_err(|error| RealityClaimError::InvalidWorld(error.to_string()))?;
        self.source
            .validate()
            .map_err(|error| RealityClaimError::InvalidSource(error.to_string()))?;
        self.statement_digest
            .validate()
            .map_err(|error| RealityClaimError::InvalidDigest(error.to_string()))?;
        self.evidence_digest
            .validate()
            .map_err(|error| RealityClaimError::InvalidDigest(error.to_string()))?;
        validate_grounding(self.world.layer, &self.source, self.grounding)
    }
}

pub fn validate_grounding(
    layer: RealityLayer,
    source: &EvidenceSource,
    grounding: ClaimGrounding,
) -> Result<(), RealityClaimError> {
    let compatible = matches!(
        (layer, source, grounding),
        (
            RealityLayer::PhysicalGrounded,
            EvidenceSource::PhysicalSensor { .. },
            ClaimGrounding::DirectWorldObservation
        )
            | (
                RealityLayer::DigitalCommitted,
                EvidenceSource::DigitalWorldObservation { .. },
                ClaimGrounding::DirectWorldObservation
            )
            | (
                _,
                EvidenceSource::DerivedComputation { .. },
                ClaimGrounding::DerivedInference
            )
            | (
                RealityLayer::Counterfactual,
                EvidenceSource::CounterfactualSimulation { .. },
                ClaimGrounding::Simulated
            )
            | (
                RealityLayer::Replay,
                EvidenceSource::Replay { .. },
                ClaimGrounding::Replayed
            )
            | (
                RealityLayer::Dream,
                EvidenceSource::DreamGeneration { .. },
                ClaimGrounding::Dreamed
            )
            | (
                RealityLayer::Imported,
                EvidenceSource::Imported { .. },
                ClaimGrounding::Imported
            )
            | (
                RealityLayer::Unknown,
                EvidenceSource::Unknown,
                ClaimGrounding::Unknown
            )
    );

    if compatible {
        Ok(())
    } else {
        Err(RealityClaimError::GroundingSourceLayerMismatch)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RealityClaimError {
    #[error("invalid world descriptor: {0}")]
    InvalidWorld(String),
    #[error("invalid evidence source: {0}")]
    InvalidSource(String),
    #[error("invalid typed digest: {0}")]
    InvalidDigest(String),
    #[error("claim grounding, evidence source and world layer are incompatible")]
    GroundingSourceLayerMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{digest::DigestAlgorithm, types::{WorldId, WorldLineageId, WorldOrigin}};

    fn digital_world() -> WorldDescriptor {
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

    fn digest(domain: &str) -> TypedDigest {
        TypedDigest::new(domain, DigestAlgorithm::Blake3, "abc").unwrap()
    }

    #[test]
    fn physical_derived_computation_cannot_claim_direct_observation() {
        let result = validate_grounding(
            RealityLayer::PhysicalGrounded,
            &EvidenceSource::DerivedComputation { processor_id: "vision".into() },
            ClaimGrounding::DirectWorldObservation,
        );
        assert_eq!(result, Err(RealityClaimError::GroundingSourceLayerMismatch));
    }

    #[test]
    fn direct_digital_observation_is_distinct_from_physical_observation() {
        let claim = RealityClaimEnvelope {
            world: digital_world(),
            grounding: ClaimGrounding::DirectWorldObservation,
            source: EvidenceSource::DigitalWorldObservation { host_id: "symtropy".into() },
            statement_digest: digest("statement.v1"),
            evidence_digest: digest("gpu-frame.v1"),
        };
        claim.validate().unwrap();
        assert_eq!(claim.world.layer, RealityLayer::DigitalCommitted);
    }
}
