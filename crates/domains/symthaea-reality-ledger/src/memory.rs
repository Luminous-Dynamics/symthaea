// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Memory-admission policy that preserves world provenance.
//!
//! This is intentionally not a truth oracle. A physical sensor can be wrong,
//! and a digital simulation can be internally inconsistent. The gate only
//! prevents stronger provenance claims than the source supports.

use serde::{Deserialize, Serialize};

use crate::{
    ledger::{RealityLedgerError, RealityRecord},
    types::{EvidenceSource, RealityLayer, WorldId, WorldLineageId},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryAdmissionClass {
    /// A memory explicitly bound to a physical-grounded world observation.
    /// This does not mean the observation is infallible.
    PhysicalWorldBound,
    /// A memory explicitly bound to an authoritative digital world.
    DigitalWorldBound,
    /// A derived computation associated with a committed world. The inference
    /// must remain distinguishable from direct observation.
    CommittedWorldDerived,
    /// A counterfactual or dream memory that may be recalled as imagination
    /// but must never be represented as having happened in the parent world.
    HypotheticalOnly,
    /// A replay describes historical/reconstructed content rather than a new
    /// observation in the current world.
    ReplayOnly,
    /// Imported content whose external grounding is not established here.
    ImportedUnverified,
    /// Provenance is unresolved; hold rather than upgrade.
    UnknownHold,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryAdmissionReceipt {
    pub record_id: String,
    pub record_digest: String,
    pub world_id: WorldId,
    pub lineage_id: WorldLineageId,
    pub class: MemoryAdmissionClass,
    pub may_claim_happened_in_current_world: bool,
    pub may_claim_physically_observed: bool,
}

pub fn assess_memory_admission(
    record: &RealityRecord,
) -> Result<MemoryAdmissionReceipt, MemoryAdmissionError> {
    record.validate().map_err(MemoryAdmissionError::Ledger)?;
    let class = match (record.world.layer, &record.source) {
        (RealityLayer::PhysicalGrounded, EvidenceSource::PhysicalSensor { .. }) => {
            MemoryAdmissionClass::PhysicalWorldBound
        }
        (RealityLayer::PhysicalGrounded, EvidenceSource::DerivedComputation { .. })
        | (RealityLayer::DigitalCommitted, EvidenceSource::DerivedComputation { .. }) => {
            MemoryAdmissionClass::CommittedWorldDerived
        }
        (RealityLayer::DigitalCommitted, EvidenceSource::DigitalWorldObservation { .. }) => {
            MemoryAdmissionClass::DigitalWorldBound
        }
        (RealityLayer::Counterfactual, _) | (RealityLayer::Dream, _) => {
            MemoryAdmissionClass::HypotheticalOnly
        }
        (RealityLayer::Replay, _) => MemoryAdmissionClass::ReplayOnly,
        (RealityLayer::Imported, _) => MemoryAdmissionClass::ImportedUnverified,
        (RealityLayer::Unknown, _) => MemoryAdmissionClass::UnknownHold,
        // `RealityRecord::validate` should already reject these pairings.
        _ => return Err(MemoryAdmissionError::ImpossibleSourceLayerCombination),
    };

    let may_claim_happened_in_current_world = matches!(
        class,
        MemoryAdmissionClass::PhysicalWorldBound
            | MemoryAdmissionClass::DigitalWorldBound
            | MemoryAdmissionClass::CommittedWorldDerived
    );
    let may_claim_physically_observed = class == MemoryAdmissionClass::PhysicalWorldBound;

    Ok(MemoryAdmissionReceipt {
        record_id: record.record_id.0.clone(),
        record_digest: record.digest().map_err(MemoryAdmissionError::Ledger)?,
        world_id: record.world.world_id.clone(),
        lineage_id: record.world.lineage_id.clone(),
        class,
        may_claim_happened_in_current_world,
        may_claim_physically_observed,
    })
}

impl MemoryAdmissionReceipt {
    /// Require that a memory remain in a hypothetical/dream namespace.
    pub fn require_hypothetical_only(&self) -> Result<(), MemoryAdmissionError> {
        if self.class == MemoryAdmissionClass::HypotheticalOnly
            && !self.may_claim_happened_in_current_world
            && !self.may_claim_physically_observed
        {
            Ok(())
        } else {
            Err(MemoryAdmissionError::NotHypotheticalOnly)
        }
    }

    /// Require the strongest provenance class supported by this v1 contract.
    pub fn require_physical_observation(&self) -> Result<(), MemoryAdmissionError> {
        if self.class == MemoryAdmissionClass::PhysicalWorldBound
            && self.may_claim_happened_in_current_world
            && self.may_claim_physically_observed
        {
            Ok(())
        } else {
            Err(MemoryAdmissionError::NotPhysicalObservation)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MemoryAdmissionError {
    #[error("reality ledger error: {0}")]
    Ledger(#[from] RealityLedgerError),
    #[error("validated reality record reached an impossible source/layer combination")]
    ImpossibleSourceLayerCombination,
    #[error("memory receipt is not confined to hypothetical recall")]
    NotHypotheticalOnly,
    #[error("memory receipt is not a direct physical-world observation")]
    NotPhysicalObservation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ledger::RealityRecordKind,
        types::{
            RealityRecordId, WorldDescriptor, WorldId, WorldLineageId, WorldOrigin,
            WorldParentRef, WorldRelation,
        },
    };

    fn record(layer: RealityLayer, source: EvidenceSource) -> RealityRecord {
        let parent = layer.is_hypothetical().then(|| WorldParentRef {
            world_id: WorldId("studio".into()),
            lineage_id: WorldLineageId("studio-lineage".into()),
            relation: if layer == RealityLayer::Dream {
                WorldRelation::DreamedFrom
            } else {
                WorldRelation::CounterfactualOf
            },
        });
        RealityRecord {
            record_id: RealityRecordId("record".into()),
            sequence: 0,
            world: WorldDescriptor {
                world_id: WorldId("world".into()),
                lineage_id: WorldLineageId("lineage".into()),
                layer,
                origin: match layer {
                    RealityLayer::PhysicalGrounded => WorldOrigin::PhysicalSensorium,
                    RealityLayer::DigitalCommitted => WorldOrigin::DigitalHost { host_kind: "bevy".into() },
                    RealityLayer::Counterfactual => WorldOrigin::CounterfactualBranch,
                    RealityLayer::Dream => WorldOrigin::DreamEngine,
                    _ => WorldOrigin::Unknown,
                },
                parent,
                generation_depth: if layer.is_hypothetical() { 1 } else { 0 },
                creator_id: "creator".into(),
            },
            kind: RealityRecordKind::MemoryCandidate,
            source,
            revision_id: None,
            frame: None,
            content_digest: "payload".into(),
            previous_record_digest: None,
        }
    }

    #[test]
    fn counterfactual_memory_cannot_claim_parent_world_occurrence() {
        let receipt = assess_memory_admission(&record(
            RealityLayer::Counterfactual,
            EvidenceSource::CounterfactualSimulation { engine_id: "ghost".into() },
        ))
        .unwrap();
        receipt.require_hypothetical_only().unwrap();
        assert!(!receipt.may_claim_happened_in_current_world);
        assert!(!receipt.may_claim_physically_observed);
    }

    #[test]
    fn physical_sensor_memory_keeps_physical_provenance() {
        let receipt = assess_memory_admission(&record(
            RealityLayer::PhysicalGrounded,
            EvidenceSource::PhysicalSensor { sensor_id: "camera".into() },
        ))
        .unwrap();
        receipt.require_physical_observation().unwrap();
    }

    #[test]
    fn derived_physical_record_is_not_direct_sensor_memory() {
        let receipt = assess_memory_admission(&record(
            RealityLayer::PhysicalGrounded,
            EvidenceSource::DerivedComputation { processor_id: "vision".into() },
        ))
        .unwrap();
        assert_eq!(receipt.class, MemoryAdmissionClass::CommittedWorldDerived);
        assert!(!receipt.may_claim_physically_observed);
    }
}
