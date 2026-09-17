// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded canonical wire envelope for EKM-038 schema-bound revision history.
//!
//! This envelope contains only explicit typed policy/decision semantics. It has
//! no `Debug` strings and does not construct a live history or mutation handle.

use super::belief_revision_receipt::BeliefRevisionReceiptId;
use super::belief_revision_schema_persistence::BeliefRevisionSchemaHistoryCapsuleV1;
use super::belief_revision_snapshot::{
    BeliefRevisionDecisionSnapshotV1, BeliefRevisionFailureSnapshotV1,
    BeliefRevisionPolicySchemaV1, BeliefRevisionSnapshotVersion,
    KnowledgeWeightRoutingFailureSnapshotV1,
};
use super::claim_evidence::{ClaimId, EvidenceId};
use super::epistemic_vector::UncertaintyDimension;
use super::knowledge_weight_routing::{KnowledgeWeightDimension, KnowledgeWeightSource};
use std::error::Error;
use std::fmt;

const MAGIC: &[u8] = b"SYMTHAEA-EKM-SCHEMA-WIRE";
const VERSION: u16 = 1;
const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
const MAX_RECORDS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefRevisionSchemaWireVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefRevisionSchemaWireEncoding {
    ExplicitPolicyDecisionV1,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefRevisionSchemaWireRecordV1 {
    pub receipt_id: BeliefRevisionReceiptId,
    pub claim_id: ClaimId,
    pub proposed_delta: f32,
    pub evaluated_at_cycle: u64,
    pub policy_schema: BeliefRevisionPolicySchemaV1,
    pub decision_snapshot: BeliefRevisionDecisionSnapshotV1,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefRevisionSchemaWireSnapshotV1 {
    pub version: BeliefRevisionSchemaWireVersion,
    pub encoding: BeliefRevisionSchemaWireEncoding,
    pub captured_at_cycle: u64,
    pub linked_revision_capture_cycle: u64,
    pub linked_revision_count: u64,
    pub linked_next_receipt_id: BeliefRevisionReceiptId,
    pub records: Vec<BeliefRevisionSchemaWireRecordV1>,
    pub wire_checksum: [u8; 32],
}

pub struct BeliefRevisionSchemaWireV1;

impl BeliefRevisionSchemaWireV1 {
    pub fn encode(
        capsule: &BeliefRevisionSchemaHistoryCapsuleV1,
    ) -> Result<Vec<u8>, BeliefRevisionSchemaWireError> {
        let mut payload = Writer::new();
        payload.u8(1); // encoding tag
        payload.u64(capsule.captured_at_cycle());
        payload.u64(capsule.linked_revision_capture_cycle());
        payload.usize(capsule.linked_revision_count())?;
        payload.u64(capsule.linked_next_receipt_id().0);
        payload.count(capsule.records().len())?;

        for record in capsule.records() {
            payload.u64(record.receipt_id.0);
            payload.u64(record.claim_id.0);
            payload.f32(record.proposed_delta);
            payload.u64(record.evaluated_at_cycle);
            encode_policy(&mut payload, &record.policy_schema)?;
            encode_decision(&mut payload, &record.decision_snapshot)?;
        }

        let payload = payload.finish();
        if payload.len() > MAX_PAYLOAD_BYTES {
            return Err(BeliefRevisionSchemaWireError::PayloadTooLarge(payload.len()));
        }
        let payload_len = u64::try_from(payload.len())
            .map_err(|_| BeliefRevisionSchemaWireError::LengthOverflow)?;
        let checksum = checksum(VERSION, payload_len, &payload);

        let mut out = Vec::with_capacity(MAGIC.len() + 2 + 8 + payload.len() + 32);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&payload_len.to_le_bytes());
        out.extend_from_slice(&payload);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    pub fn decode(
        bytes: &[u8],
    ) -> Result<BeliefRevisionSchemaWireSnapshotV1, BeliefRevisionSchemaWireError> {
        let minimum = MAGIC.len() + 2 + 8 + 32;
        if bytes.len() < minimum {
            return Err(BeliefRevisionSchemaWireError::Truncated);
        }
        if &bytes[..MAGIC.len()] != MAGIC {
            return Err(BeliefRevisionSchemaWireError::BadMagic);
        }
        let version_offset = MAGIC.len();
        let version = u16::from_le_bytes([
            bytes[version_offset],
            bytes[version_offset + 1],
        ]);
        if version != VERSION {
            return Err(BeliefRevisionSchemaWireError::UnsupportedVersion(version));
        }
        let len_offset = version_offset + 2;
        let payload_len_u64 = u64::from_le_bytes(
            bytes[len_offset..len_offset + 8]
                .try_into()
                .expect("slice width checked"),
        );
        let payload_len = usize::try_from(payload_len_u64)
            .map_err(|_| BeliefRevisionSchemaWireError::LengthOverflow)?;
        if payload_len > MAX_PAYLOAD_BYTES {
            return Err(BeliefRevisionSchemaWireError::PayloadTooLarge(payload_len));
        }
        let expected_total = minimum
            .checked_add(payload_len)
            .ok_or(BeliefRevisionSchemaWireError::LengthOverflow)?;
        if bytes.len() != expected_total {
            return Err(BeliefRevisionSchemaWireError::EnvelopeLengthMismatch {
                declared_payload: payload_len,
                actual_total: bytes.len(),
            });
        }
        let payload_start = len_offset + 8;
        let payload_end = payload_start + payload_len;
        let payload = &bytes[payload_start..payload_end];
        let actual_checksum: [u8; 32] = bytes[payload_end..]
            .try_into()
            .expect("checksum width checked");
        let expected_checksum = checksum(VERSION, payload_len_u64, payload);
        if actual_checksum != expected_checksum {
            return Err(BeliefRevisionSchemaWireError::ChecksumMismatch);
        }

        let mut reader = Reader::new(payload);
        let encoding = match reader.u8()? {
            1 => BeliefRevisionSchemaWireEncoding::ExplicitPolicyDecisionV1,
            tag => return Err(BeliefRevisionSchemaWireError::UnknownEncodingTag(tag)),
        };
        let captured_at_cycle = reader.u64()?;
        let linked_revision_capture_cycle = reader.u64()?;
        let linked_revision_count = reader.u64()?;
        let linked_next_receipt_id = BeliefRevisionReceiptId(reader.u64()?);
        let record_count = reader.count()?;
        let mut records = Vec::with_capacity(record_count);
        for _ in 0..record_count {
            let receipt_id = BeliefRevisionReceiptId(reader.u64()?);
            let claim_id = ClaimId(reader.u64()?);
            let proposed_delta = reader.f32()?;
            if !proposed_delta.is_finite() || !(-1.0..=1.0).contains(&proposed_delta) {
                return Err(BeliefRevisionSchemaWireError::InvalidDelta(proposed_delta));
            }
            let evaluated_at_cycle = reader.u64()?;
            let policy_schema = decode_policy(&mut reader)?;
            let decision_snapshot = decode_decision(&mut reader)?;
            records.push(BeliefRevisionSchemaWireRecordV1 {
                receipt_id,
                claim_id,
                proposed_delta,
                evaluated_at_cycle,
                policy_schema,
                decision_snapshot,
            });
        }
        if !reader.is_finished() {
            return Err(BeliefRevisionSchemaWireError::TrailingPayloadBytes(
                reader.remaining(),
            ));
        }

        Ok(BeliefRevisionSchemaWireSnapshotV1 {
            version: BeliefRevisionSchemaWireVersion::V1,
            encoding,
            captured_at_cycle,
            linked_revision_capture_cycle,
            linked_revision_count,
            linked_next_receipt_id,
            records,
            wire_checksum: actual_checksum,
        })
    }
}

fn encode_policy(
    writer: &mut Writer,
    policy: &BeliefRevisionPolicySchemaV1,
) -> Result<(), BeliefRevisionSchemaWireError> {
    writer.f32(policy.max_abs_delta);
    writer.usize(policy.min_declared_provenance_roots)?;
    writer.bool(policy.require_calibration);
    writer.u64(policy.min_calibration_samples);
    writer.f64(policy.max_calibration_ece);
    writer.bool(policy.require_uncertainty_assessment);
    writer.bool(policy.require_current_uncertainty);
    writer.bool(policy.block_strengthen_with_unresolved_contradictions);
    writer.bool(policy.require_intervention_for_causal_strengthen);
    writer.count(policy.strengthen_uncertainty_caps.len())?;
    for (dimension, maximum) in &policy.strengthen_uncertainty_caps {
        writer.u8(uncertainty_tag(*dimension));
        writer.f64(*maximum);
    }
    Ok(())
}

fn decode_policy(reader: &mut Reader<'_>) -> Result<BeliefRevisionPolicySchemaV1, BeliefRevisionSchemaWireError> {
    let max_abs_delta = reader.f32()?;
    let min_roots = reader.usize()?;
    let require_calibration = reader.bool()?;
    let min_samples = reader.u64()?;
    let max_ece = reader.f64()?;
    let require_assessment = reader.bool()?;
    let require_current = reader.bool()?;
    let block_contradictions = reader.bool()?;
    let require_intervention = reader.bool()?;
    let cap_count = reader.count()?;

    let mut policy = BeliefRevisionPolicySchemaV1::new(
        max_abs_delta,
        min_roots,
        require_calibration,
        min_samples,
        max_ece,
    )
    .map_err(|_| BeliefRevisionSchemaWireError::InvalidPolicy)?
    .with_uncertainty_requirements(require_assessment, require_current)
    .block_strengthen_with_unresolved_contradictions(block_contradictions)
    .require_intervention_for_causal_strengthen(require_intervention);

    let mut previous_tag = 0u8;
    for _ in 0..cap_count {
        let tag = reader.u8()?;
        if tag <= previous_tag {
            return Err(BeliefRevisionSchemaWireError::NonCanonicalUncertaintyCaps);
        }
        previous_tag = tag;
        let dimension = parse_uncertainty(tag)?;
        let maximum = reader.f64()?;
        policy = policy
            .with_strengthen_uncertainty_cap(dimension, maximum)
            .map_err(|_| BeliefRevisionSchemaWireError::InvalidPolicy)?;
    }
    Ok(policy)
}

fn encode_decision(
    writer: &mut Writer,
    decision: &BeliefRevisionDecisionSnapshotV1,
) -> Result<(), BeliefRevisionSchemaWireError> {
    writer.bool(decision.eligible);
    writer.usize(decision.declared_provenance_root_count)?;
    writer.count(decision.failures.len())?;
    for failure in &decision.failures {
        encode_failure(writer, failure)?;
    }
    Ok(())
}

fn decode_decision(reader: &mut Reader<'_>) -> Result<BeliefRevisionDecisionSnapshotV1, BeliefRevisionSchemaWireError> {
    let eligible = reader.bool()?;
    let declared_provenance_root_count = reader.usize()?;
    let failure_count = reader.count()?;
    let mut failures = Vec::with_capacity(failure_count);
    for _ in 0..failure_count {
        failures.push(decode_failure(reader)?);
    }
    Ok(BeliefRevisionDecisionSnapshotV1 {
        version: BeliefRevisionSnapshotVersion::V1,
        eligible,
        declared_provenance_root_count,
        failures,
    })
}

fn encode_failure(
    writer: &mut Writer,
    failure: &BeliefRevisionFailureSnapshotV1,
) -> Result<(), BeliefRevisionSchemaWireError> {
    match failure {
        BeliefRevisionFailureSnapshotV1::WeightRoutingDenied(failures) => {
            writer.u8(1);
            writer.count(failures.len())?;
            for failure in failures {
                match failure {
                    KnowledgeWeightRoutingFailureSnapshotV1::SourceCannotUpdateDimension {
                        source,
                        dimension,
                    } => {
                        writer.u8(1);
                        writer.u8(source_tag(*source));
                        writer.u8(weight_dimension_tag(*dimension));
                    }
                    KnowledgeWeightRoutingFailureSnapshotV1::EpistemicUpdateHasNoEvidenceBasis => {
                        writer.u8(2);
                    }
                }
            }
        }
        BeliefRevisionFailureSnapshotV1::UnknownClaim(id) => {
            writer.u8(2);
            writer.u64(id.0);
        }
        BeliefRevisionFailureSnapshotV1::UnknownEvidence(id) => {
            writer.u8(3);
            writer.u64(id.0);
        }
        BeliefRevisionFailureSnapshotV1::EvidenceForDifferentClaim {
            evidence_id,
            expected_claim,
            actual_claim,
        } => {
            writer.u8(4);
            writer.u64(evidence_id.0);
            writer.u64(expected_claim.0);
            writer.u64(actual_claim.0);
        }
        BeliefRevisionFailureSnapshotV1::PositiveDeltaLacksSupportingEvidence => writer.u8(5),
        BeliefRevisionFailureSnapshotV1::PositiveDeltaIncludesContradictingEvidence => writer.u8(6),
        BeliefRevisionFailureSnapshotV1::NegativeDeltaLacksContradictingEvidence => writer.u8(7),
        BeliefRevisionFailureSnapshotV1::NegativeDeltaIncludesSupportingEvidence => writer.u8(8),
        BeliefRevisionFailureSnapshotV1::DeclaredProvenanceRootsBelowMinimum { required, actual } => {
            writer.u8(9);
            writer.usize(*required)?;
            writer.usize(*actual)?;
        }
        BeliefRevisionFailureSnapshotV1::DeltaExceedsPolicy { maximum, actual } => {
            writer.u8(10);
            writer.f32(*maximum);
            writer.f32(*actual);
        }
        BeliefRevisionFailureSnapshotV1::CalibrationMissing => writer.u8(11),
        BeliefRevisionFailureSnapshotV1::CalibrationSamplesBelowMinimum { required, actual } => {
            writer.u8(12);
            writer.u64(*required);
            writer.u64(*actual);
        }
        BeliefRevisionFailureSnapshotV1::CalibrationEceAboveMaximum { maximum, actual } => {
            writer.u8(13);
            writer.f64(*maximum);
            writer.f64(*actual);
        }
        BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentMissing => writer.u8(14),
        BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentForDifferentClaim {
            expected_claim,
            actual_claim,
        } => {
            writer.u8(15);
            writer.u64(expected_claim.0);
            writer.u64(actual_claim.0);
        }
        BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentPredatesEvidence {
            assessment_cycle,
            latest_evidence_cycle,
        } => {
            writer.u8(16);
            writer.u64(*assessment_cycle);
            writer.u64(*latest_evidence_cycle);
        }
        BeliefRevisionFailureSnapshotV1::RequiredUncertaintyDimensionUnassessed(dimension) => {
            writer.u8(17);
            writer.u8(uncertainty_tag(*dimension));
        }
        BeliefRevisionFailureSnapshotV1::UncertaintyAboveMaximum {
            dimension,
            maximum,
            actual,
        } => {
            writer.u8(18);
            writer.u8(uncertainty_tag(*dimension));
            writer.f64(*maximum);
            writer.f64(*actual);
        }
        BeliefRevisionFailureSnapshotV1::UnresolvedContradictionBlocksStrengthen { count } => {
            writer.u8(19);
            writer.usize(*count)?;
        }
        BeliefRevisionFailureSnapshotV1::CausalStrengthenLacksInterventionalBasis => writer.u8(20),
    }
    Ok(())
}

fn decode_failure(reader: &mut Reader<'_>) -> Result<BeliefRevisionFailureSnapshotV1, BeliefRevisionSchemaWireError> {
    Ok(match reader.u8()? {
        1 => {
            let count = reader.count()?;
            let mut failures = Vec::with_capacity(count);
            for _ in 0..count {
                failures.push(match reader.u8()? {
                    1 => KnowledgeWeightRoutingFailureSnapshotV1::SourceCannotUpdateDimension {
                        source: parse_source(reader.u8()?)?,
                        dimension: parse_weight_dimension(reader.u8()?)?,
                    },
                    2 => KnowledgeWeightRoutingFailureSnapshotV1::EpistemicUpdateHasNoEvidenceBasis,
                    tag => return Err(BeliefRevisionSchemaWireError::UnknownRoutingFailureTag(tag)),
                });
            }
            BeliefRevisionFailureSnapshotV1::WeightRoutingDenied(failures)
        }
        2 => BeliefRevisionFailureSnapshotV1::UnknownClaim(ClaimId(reader.u64()?)),
        3 => BeliefRevisionFailureSnapshotV1::UnknownEvidence(EvidenceId(reader.u64()?)),
        4 => BeliefRevisionFailureSnapshotV1::EvidenceForDifferentClaim {
            evidence_id: EvidenceId(reader.u64()?),
            expected_claim: ClaimId(reader.u64()?),
            actual_claim: ClaimId(reader.u64()?),
        },
        5 => BeliefRevisionFailureSnapshotV1::PositiveDeltaLacksSupportingEvidence,
        6 => BeliefRevisionFailureSnapshotV1::PositiveDeltaIncludesContradictingEvidence,
        7 => BeliefRevisionFailureSnapshotV1::NegativeDeltaLacksContradictingEvidence,
        8 => BeliefRevisionFailureSnapshotV1::NegativeDeltaIncludesSupportingEvidence,
        9 => BeliefRevisionFailureSnapshotV1::DeclaredProvenanceRootsBelowMinimum {
            required: reader.usize()?,
            actual: reader.usize()?,
        },
        10 => {
            let maximum = checked_unit_f32(reader.f32()?)?;
            let actual = checked_unit_f32(reader.f32()?)?;
            BeliefRevisionFailureSnapshotV1::DeltaExceedsPolicy { maximum, actual }
        }
        11 => BeliefRevisionFailureSnapshotV1::CalibrationMissing,
        12 => BeliefRevisionFailureSnapshotV1::CalibrationSamplesBelowMinimum {
            required: reader.u64()?,
            actual: reader.u64()?,
        },
        13 => {
            let maximum = checked_unit_f64(reader.f64()?)?;
            let actual = checked_unit_f64(reader.f64()?)?;
            BeliefRevisionFailureSnapshotV1::CalibrationEceAboveMaximum { maximum, actual }
        }
        14 => BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentMissing,
        15 => BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentForDifferentClaim {
            expected_claim: ClaimId(reader.u64()?),
            actual_claim: ClaimId(reader.u64()?),
        },
        16 => BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentPredatesEvidence {
            assessment_cycle: reader.u64()?,
            latest_evidence_cycle: reader.u64()?,
        },
        17 => BeliefRevisionFailureSnapshotV1::RequiredUncertaintyDimensionUnassessed(
            parse_uncertainty(reader.u8()?)?,
        ),
        18 => BeliefRevisionFailureSnapshotV1::UncertaintyAboveMaximum {
            dimension: parse_uncertainty(reader.u8()?)?,
            maximum: checked_unit_f64(reader.f64()?)?,
            actual: checked_unit_f64(reader.f64()?)?,
        },
        19 => BeliefRevisionFailureSnapshotV1::UnresolvedContradictionBlocksStrengthen {
            count: reader.usize()?,
        },
        20 => BeliefRevisionFailureSnapshotV1::CausalStrengthenLacksInterventionalBasis,
        tag => return Err(BeliefRevisionSchemaWireError::UnknownDecisionFailureTag(tag)),
    })
}

fn uncertainty_tag(value: UncertaintyDimension) -> u8 {
    match value {
        UncertaintyDimension::Epistemic => 1,
        UncertaintyDimension::Aleatoric => 2,
        UncertaintyDimension::Ontological => 3,
        UncertaintyDimension::DistributionShift => 4,
    }
}

fn parse_uncertainty(tag: u8) -> Result<UncertaintyDimension, BeliefRevisionSchemaWireError> {
    match tag {
        1 => Ok(UncertaintyDimension::Epistemic),
        2 => Ok(UncertaintyDimension::Aleatoric),
        3 => Ok(UncertaintyDimension::Ontological),
        4 => Ok(UncertaintyDimension::DistributionShift),
        tag => Err(BeliefRevisionSchemaWireError::UnknownUncertaintyTag(tag)),
    }
}

fn source_tag(value: KnowledgeWeightSource) -> u8 {
    match value {
        KnowledgeWeightSource::AdmittedEvidence => 1,
        KnowledgeWeightSource::Retrieval => 2,
        KnowledgeWeightSource::SimilarityMatch => 3,
        KnowledgeWeightSource::DreamReplay => 4,
        KnowledgeWeightSource::CausalConsolidation => 5,
        KnowledgeWeightSource::MemoryDecay => 6,
        KnowledgeWeightSource::TaskRelevance => 7,
    }
}

fn parse_source(tag: u8) -> Result<KnowledgeWeightSource, BeliefRevisionSchemaWireError> {
    match tag {
        1 => Ok(KnowledgeWeightSource::AdmittedEvidence),
        2 => Ok(KnowledgeWeightSource::Retrieval),
        3 => Ok(KnowledgeWeightSource::SimilarityMatch),
        4 => Ok(KnowledgeWeightSource::DreamReplay),
        5 => Ok(KnowledgeWeightSource::CausalConsolidation),
        6 => Ok(KnowledgeWeightSource::MemoryDecay),
        7 => Ok(KnowledgeWeightSource::TaskRelevance),
        tag => Err(BeliefRevisionSchemaWireError::UnknownSourceTag(tag)),
    }
}

fn weight_dimension_tag(value: KnowledgeWeightDimension) -> u8 {
    match value {
        KnowledgeWeightDimension::EpistemicSupport => 1,
        KnowledgeWeightDimension::Accessibility => 2,
        KnowledgeWeightDimension::Retention => 3,
        KnowledgeWeightDimension::Consolidation => 4,
    }
}

fn parse_weight_dimension(tag: u8) -> Result<KnowledgeWeightDimension, BeliefRevisionSchemaWireError> {
    match tag {
        1 => Ok(KnowledgeWeightDimension::EpistemicSupport),
        2 => Ok(KnowledgeWeightDimension::Accessibility),
        3 => Ok(KnowledgeWeightDimension::Retention),
        4 => Ok(KnowledgeWeightDimension::Consolidation),
        tag => Err(BeliefRevisionSchemaWireError::UnknownWeightDimensionTag(tag)),
    }
}

fn checked_unit_f32(value: f32) -> Result<f32, BeliefRevisionSchemaWireError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(BeliefRevisionSchemaWireError::InvalidUnitF32(value));
    }
    Ok(value)
}

fn checked_unit_f64(value: f64) -> Result<f64, BeliefRevisionSchemaWireError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(BeliefRevisionSchemaWireError::InvalidUnitF64(value));
    }
    Ok(value)
}

fn checksum(version: u16, payload_len: u64, payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-revision-schema-wire-v1");
    hasher.update(&version.to_le_bytes());
    hasher.update(&payload_len.to_le_bytes());
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }
    fn finish(self) -> Vec<u8> {
        self.bytes
    }
    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }
    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }
    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn usize(&mut self, value: usize) -> Result<(), BeliefRevisionSchemaWireError> {
        self.u64(u64::try_from(value).map_err(|_| BeliefRevisionSchemaWireError::LengthOverflow)?);
        Ok(())
    }
    fn count(&mut self, value: usize) -> Result<(), BeliefRevisionSchemaWireError> {
        if value > MAX_RECORDS {
            return Err(BeliefRevisionSchemaWireError::RecordCountTooLarge(value));
        }
        self.usize(value)
    }
    fn f32(&mut self, value: f32) {
        self.bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    fn f64(&mut self, value: f64) {
        self.bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take(&mut self, len: usize) -> Result<&'a [u8], BeliefRevisionSchemaWireError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(BeliefRevisionSchemaWireError::LengthOverflow)?;
        if end > self.bytes.len() {
            return Err(BeliefRevisionSchemaWireError::Truncated);
        }
        let out = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(out)
    }
    fn u8(&mut self) -> Result<u8, BeliefRevisionSchemaWireError> {
        Ok(self.take(1)?[0])
    }
    fn bool(&mut self) -> Result<bool, BeliefRevisionSchemaWireError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(BeliefRevisionSchemaWireError::InvalidBoolean(value)),
        }
    }
    fn u64(&mut self) -> Result<u64, BeliefRevisionSchemaWireError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .expect("reader returned exact u64 width"),
        ))
    }
    fn usize(&mut self) -> Result<usize, BeliefRevisionSchemaWireError> {
        usize::try_from(self.u64()?).map_err(|_| BeliefRevisionSchemaWireError::LengthOverflow)
    }
    fn count(&mut self) -> Result<usize, BeliefRevisionSchemaWireError> {
        let value = self.usize()?;
        if value > MAX_RECORDS {
            return Err(BeliefRevisionSchemaWireError::RecordCountTooLarge(value));
        }
        Ok(value)
    }
    fn f32(&mut self) -> Result<f32, BeliefRevisionSchemaWireError> {
        Ok(f32::from_bits(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .expect("reader returned exact f32 width"),
        )))
    }
    fn f64(&mut self) -> Result<f64, BeliefRevisionSchemaWireError> {
        Ok(f64::from_bits(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .expect("reader returned exact f64 width"),
        )))
    }
    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefRevisionSchemaWireError {
    Truncated,
    BadMagic,
    UnsupportedVersion(u16),
    UnknownEncodingTag(u8),
    EnvelopeLengthMismatch {
        declared_payload: usize,
        actual_total: usize,
    },
    ChecksumMismatch,
    PayloadTooLarge(usize),
    RecordCountTooLarge(usize),
    LengthOverflow,
    InvalidBoolean(u8),
    InvalidDelta(f32),
    InvalidPolicy,
    NonCanonicalUncertaintyCaps,
    UnknownUncertaintyTag(u8),
    UnknownSourceTag(u8),
    UnknownWeightDimensionTag(u8),
    UnknownRoutingFailureTag(u8),
    UnknownDecisionFailureTag(u8),
    InvalidUnitF32(f32),
    InvalidUnitF64(f64),
    TrailingPayloadBytes(usize),
}

impl fmt::Display for BeliefRevisionSchemaWireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "belief revision schema wire invalid: {self:?}")
    }
}

impl Error for BeliefRevisionSchemaWireError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistory,
        BeliefRevisionHistoryCapsuleV1, BeliefRevisionSchemaHistoryV1, ClaimKind,
        EpistemicLedger, EpistemicRevisionProposal, EpistemicSupportStore, EvidenceKind,
        EvidencePolarity,
    };

    fn capsule() -> BeliefRevisionSchemaHistoryCapsuleV1 {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("lab", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                provenance,
                2,
                None,
                None,
            )
            .unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.1, vec![evidence], "measurement")
            .unwrap();
        let schema = BeliefRevisionPolicySchemaV1::new(0.2, 1, true, 10, 0.2)
            .unwrap()
            .with_uncertainty_requirements(true, true)
            .with_strengthen_uncertainty_cap(UncertaintyDimension::Epistemic, 0.4)
            .unwrap();
        let mut receipts = BeliefRevisionHistory::new();
        let mut schemas = BeliefRevisionSchemaHistoryV1::new();
        schemas
            .evaluate_and_record(
                &mut receipts,
                &ledger,
                &proposal,
                &schema,
                None,
                None,
                3,
            )
            .unwrap();
        let store = EpistemicSupportStore::new();
        let mutations = BeliefMutationPersistenceCapsuleV1::capture(&store, &[], 4).unwrap();
        let revisions = BeliefRevisionHistoryCapsuleV1::capture(&receipts, &mutations, 4).unwrap();
        BeliefRevisionSchemaHistoryCapsuleV1::capture(&schemas, &receipts, &revisions, 4).unwrap()
    }

    #[test]
    fn typed_schema_wire_round_trip_preserves_semantics() {
        let capsule = capsule();
        let bytes = BeliefRevisionSchemaWireV1::encode(&capsule).unwrap();
        let decoded = BeliefRevisionSchemaWireV1::decode(&bytes).unwrap();
        assert_eq!(decoded.captured_at_cycle, capsule.captured_at_cycle());
        assert_eq!(decoded.linked_revision_count as usize, capsule.linked_revision_count());
        assert_eq!(decoded.records.len(), 1);
        assert_eq!(decoded.records[0].receipt_id, capsule.records()[0].receipt_id);
        assert_eq!(decoded.records[0].policy_schema, capsule.records()[0].policy_schema);
        assert_eq!(decoded.records[0].decision_snapshot, capsule.records()[0].decision_snapshot);
    }

    #[test]
    fn one_byte_tamper_fails_checksum() {
        let capsule = capsule();
        let mut bytes = BeliefRevisionSchemaWireV1::encode(&capsule).unwrap();
        let payload_start = MAGIC.len() + 2 + 8;
        bytes[payload_start] ^= 0x01;
        assert_eq!(
            BeliefRevisionSchemaWireV1::decode(&bytes).unwrap_err(),
            BeliefRevisionSchemaWireError::ChecksumMismatch
        );
    }

    #[test]
    fn unsupported_version_fails_before_payload_parse() {
        let capsule = capsule();
        let mut bytes = BeliefRevisionSchemaWireV1::encode(&capsule).unwrap();
        let offset = MAGIC.len();
        bytes[offset..offset + 2].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            BeliefRevisionSchemaWireV1::decode(&bytes).unwrap_err(),
            BeliefRevisionSchemaWireError::UnsupportedVersion(2)
        );
    }
}
