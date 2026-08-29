// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only provenance ledger for observations, simulations and world events.

use serde::{Deserialize, Serialize};

use crate::types::{
    EvidenceSource, RealityLayer, RealityRecordId, RealityTypeError, WorldDescriptor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RealityRecordKind {
    Observation,
    DerivedInference,
    Action,
    Creation,
    MemoryCandidate,
    WorldTransition,
    Diagnostic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealityRecord {
    pub record_id: RealityRecordId,
    pub sequence: u64,
    pub world: WorldDescriptor,
    pub kind: RealityRecordKind,
    pub source: EvidenceSource,
    /// Host-owned revision identity. The ledger does not interpret the value.
    pub revision_id: Option<String>,
    /// Host-owned deterministic frame/tick coordinate.
    pub frame: Option<u64>,
    /// Digest of the evidence payload/artifact. Raw payloads remain outside
    /// this host-neutral ledger.
    pub content_digest: String,
    pub previous_record_digest: Option<String>,
}

impl RealityRecord {
    pub fn validate(&self) -> Result<(), RealityLedgerError> {
        self.record_id.validate().map_err(RealityLedgerError::Type)?;
        self.world.validate().map_err(RealityLedgerError::Type)?;
        self.source.validate().map_err(RealityLedgerError::Type)?;
        if self.content_digest.trim().is_empty() {
            return Err(RealityLedgerError::EmptyContentDigest);
        }
        if self.revision_id.as_ref().is_some_and(|id| id.trim().is_empty()) {
            return Err(RealityLedgerError::EmptyRevisionId);
        }
        validate_source_layer(self.world.layer, &self.source)?;
        Ok(())
    }

    pub fn digest(&self) -> Result<String, RealityLedgerError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        feed(&mut hasher, b"symthaea.reality-record.v1");
        feed(&mut hasher, self.record_id.0.as_bytes());
        hasher.update(&self.sequence.to_le_bytes());
        feed_world(&mut hasher, &self.world);
        hasher.update(&[record_kind_tag(self.kind)]);
        feed_source(&mut hasher, &self.source);
        feed_optional(&mut hasher, self.revision_id.as_deref());
        match self.frame {
            Some(frame) => {
                hasher.update(&[1]);
                hasher.update(&frame.to_le_bytes());
            }
            None => hasher.update(&[0]),
        };
        feed(&mut hasher, self.content_digest.as_bytes());
        feed_optional(&mut hasher, self.previous_record_digest.as_deref());
        Ok(hasher.finalize().to_hex().to_string())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealityLedger {
    records: Vec<RealityRecord>,
}

impl RealityLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn records(&self) -> &[RealityRecord] {
        &self.records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn last_digest(&self) -> Result<Option<String>, RealityLedgerError> {
        self.records.last().map(RealityRecord::digest).transpose()
    }

    pub fn append(&mut self, record: RealityRecord) -> Result<String, RealityLedgerError> {
        record.validate()?;
        let expected_sequence = self.records.len() as u64;
        if record.sequence != expected_sequence {
            return Err(RealityLedgerError::SequenceMismatch {
                expected: expected_sequence,
                actual: record.sequence,
            });
        }
        let expected_previous = self.last_digest()?;
        if record.previous_record_digest != expected_previous {
            return Err(RealityLedgerError::PreviousDigestMismatch);
        }
        if self.records.iter().any(|prior| prior.record_id == record.record_id) {
            return Err(RealityLedgerError::DuplicateRecordId(record.record_id.0));
        }
        let digest = record.digest()?;
        self.records.push(record);
        Ok(digest)
    }

    pub fn verify(&self) -> Result<String, RealityLedgerError> {
        let mut expected_previous: Option<String> = None;
        for (index, record) in self.records.iter().enumerate() {
            record.validate()?;
            if record.sequence != index as u64 {
                return Err(RealityLedgerError::SequenceMismatch {
                    expected: index as u64,
                    actual: record.sequence,
                });
            }
            if record.previous_record_digest != expected_previous {
                return Err(RealityLedgerError::PreviousDigestMismatch);
            }
            expected_previous = Some(record.digest()?);
        }
        expected_previous.ok_or(RealityLedgerError::EmptyLedger)
    }
}

fn validate_source_layer(layer: RealityLayer, source: &EvidenceSource) -> Result<(), RealityLedgerError> {
    let compatible = matches!(
        (layer, source),
        (RealityLayer::PhysicalGrounded, EvidenceSource::PhysicalSensor { .. })
            | (RealityLayer::PhysicalGrounded, EvidenceSource::DerivedComputation { .. })
            | (RealityLayer::DigitalCommitted, EvidenceSource::DigitalWorldObservation { .. })
            | (RealityLayer::DigitalCommitted, EvidenceSource::DerivedComputation { .. })
            | (RealityLayer::Counterfactual, EvidenceSource::CounterfactualSimulation { .. })
            | (RealityLayer::Counterfactual, EvidenceSource::DerivedComputation { .. })
            | (RealityLayer::Replay, EvidenceSource::Replay { .. })
            | (RealityLayer::Replay, EvidenceSource::DerivedComputation { .. })
            | (RealityLayer::Dream, EvidenceSource::DreamGeneration { .. })
            | (RealityLayer::Dream, EvidenceSource::DerivedComputation { .. })
            | (RealityLayer::Imported, EvidenceSource::Imported { .. })
            | (RealityLayer::Imported, EvidenceSource::DerivedComputation { .. })
            | (RealityLayer::Unknown, EvidenceSource::Unknown)
            | (RealityLayer::Unknown, EvidenceSource::DerivedComputation { .. })
    );
    if compatible {
        Ok(())
    } else {
        Err(RealityLedgerError::SourceLayerMismatch)
    }
}

fn feed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn feed_optional(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            feed(hasher, value.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
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
        crate::types::WorldOrigin::PhysicalSensorium => hasher.update(&[0]),
        crate::types::WorldOrigin::DigitalHost { host_kind } => {
            hasher.update(&[1]);
            feed(hasher, host_kind.as_bytes());
        }
        crate::types::WorldOrigin::CounterfactualBranch => hasher.update(&[2]),
        crate::types::WorldOrigin::ReplayArtifact => hasher.update(&[3]),
        crate::types::WorldOrigin::DreamEngine => hasher.update(&[4]),
        crate::types::WorldOrigin::ImportedExternal { source } => {
            hasher.update(&[5]);
            feed(hasher, source.as_bytes());
        }
        crate::types::WorldOrigin::Unknown => hasher.update(&[6]),
    };
}

fn feed_source(hasher: &mut blake3::Hasher, source: &EvidenceSource) {
    match source {
        EvidenceSource::PhysicalSensor { sensor_id } => {
            hasher.update(&[0]);
            feed(hasher, sensor_id.as_bytes());
        }
        EvidenceSource::DigitalWorldObservation { host_id } => {
            hasher.update(&[1]);
            feed(hasher, host_id.as_bytes());
        }
        EvidenceSource::CounterfactualSimulation { engine_id } => {
            hasher.update(&[2]);
            feed(hasher, engine_id.as_bytes());
        }
        EvidenceSource::Replay { artifact_id } => {
            hasher.update(&[3]);
            feed(hasher, artifact_id.as_bytes());
        }
        EvidenceSource::DreamGeneration { engine_id } => {
            hasher.update(&[4]);
            feed(hasher, engine_id.as_bytes());
        }
        EvidenceSource::Imported { source_id } => {
            hasher.update(&[5]);
            feed(hasher, source_id.as_bytes());
        }
        EvidenceSource::DerivedComputation { processor_id } => {
            hasher.update(&[6]);
            feed(hasher, processor_id.as_bytes());
        }
        EvidenceSource::Unknown => {
            hasher.update(&[7]);
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

fn relation_tag(relation: &crate::types::WorldRelation) -> u8 {
    match relation {
        crate::types::WorldRelation::CounterfactualOf => 0,
        crate::types::WorldRelation::ReplayOf => 1,
        crate::types::WorldRelation::DreamedFrom => 2,
        crate::types::WorldRelation::ImportedFrom => 3,
        crate::types::WorldRelation::SpawnedFrom => 4,
    }
}

fn record_kind_tag(kind: RealityRecordKind) -> u8 {
    match kind {
        RealityRecordKind::Observation => 0,
        RealityRecordKind::DerivedInference => 1,
        RealityRecordKind::Action => 2,
        RealityRecordKind::Creation => 3,
        RealityRecordKind::MemoryCandidate => 4,
        RealityRecordKind::WorldTransition => 5,
        RealityRecordKind::Diagnostic => 6,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RealityLedgerError {
    #[error("reality type error: {0}")]
    Type(#[from] RealityTypeError),
    #[error("content digest may not be empty")]
    EmptyContentDigest,
    #[error("revision id may not be empty when supplied")]
    EmptyRevisionId,
    #[error("evidence source is inconsistent with the declared reality layer")]
    SourceLayerMismatch,
    #[error("ledger sequence mismatch: expected {expected}, got {actual}")]
    SequenceMismatch { expected: u64, actual: u64 },
    #[error("previous record digest does not match the current ledger head")]
    PreviousDigestMismatch,
    #[error("duplicate reality record id: {0}")]
    DuplicateRecordId(String),
    #[error("an empty ledger is not verified evidence")]
    EmptyLedger,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{WorldId, WorldLineageId, WorldOrigin};

    fn world() -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId("studio".into()),
            lineage_id: WorldLineageId("lineage".into()),
            layer: RealityLayer::DigitalCommitted,
            origin: WorldOrigin::DigitalHost { host_kind: "bevy".into() },
            parent: None,
            generation_depth: 0,
            creator_id: "host".into(),
        }
    }

    fn record(sequence: u64, previous: Option<String>) -> RealityRecord {
        RealityRecord {
            record_id: RealityRecordId(format!("r-{sequence}")),
            sequence,
            world: world(),
            kind: RealityRecordKind::Observation,
            source: EvidenceSource::DigitalWorldObservation { host_id: "bevy".into() },
            revision_id: Some("rev-1".into()),
            frame: Some(sequence),
            content_digest: format!("artifact-{sequence}"),
            previous_record_digest: previous,
        }
    }

    #[test]
    fn append_only_chain_verifies() {
        let mut ledger = RealityLedger::new();
        let d0 = ledger.append(record(0, None)).unwrap();
        let d1 = ledger.append(record(1, Some(d0))).unwrap();
        assert_eq!(ledger.verify().unwrap(), d1);
    }

    #[test]
    fn chain_rejects_missing_previous_digest() {
        let mut ledger = RealityLedger::new();
        ledger.append(record(0, None)).unwrap();
        assert_eq!(ledger.append(record(1, None)), Err(RealityLedgerError::PreviousDigestMismatch));
    }

    #[test]
    fn physical_layer_cannot_claim_dream_source() {
        let mut bad = record(0, None);
        bad.world.layer = RealityLayer::PhysicalGrounded;
        bad.world.origin = WorldOrigin::PhysicalSensorium;
        bad.source = EvidenceSource::DreamGeneration { engine_id: "dream".into() };
        assert_eq!(bad.validate(), Err(RealityLedgerError::SourceLayerMismatch));
    }
}
