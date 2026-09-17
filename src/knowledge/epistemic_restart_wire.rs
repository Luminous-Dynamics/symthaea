// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned, bounded wire envelope for EKM restart capsules.
//!
//! This module serializes an [`EpistemicRestartCapsuleV1`] into deterministic
//! bytes with explicit framing and a domain-separated BLAKE3 checksum. Decoding
//! produces an untrusted, read-only snapshot DTO only. It deliberately does not
//! construct an EKM capsule, quarantine image, writable ledger, support store,
//! revision history, or activation handle.
//!
//! V1 still carries revision policy/decision internals as opaque length-prefixed
//! `Debug` text because those fields are intentionally private in EKM-024/025.
//! The encoding enum names that limitation explicitly; V1 is therefore not
//! claimed as a durable cross-version receipt schema.

use super::belief_mutation_firewall::{BeliefMutationReceiptId, EpistemicSupportStore};
use super::belief_revision_receipt::{
    BeliefRevisionReceiptId, RevisionEvidenceReference, RevisionEvidenceSnapshot,
};
use super::claim_evidence::{
    ClaimId, ClaimKind, EvidenceId, EvidenceKind, EvidencePolarity, ProvenanceId,
};
use super::epistemic_restart_capsule::EpistemicRestartCapsuleV1;
use super::epistemic_vector::UncertaintyDimension;
use super::knowledge_weight_routing::BoundedWeight;
use std::error::Error;
use std::fmt;

const MAGIC: &[u8] = b"SYMTHAEA-EKM-WIRE";
const VERSION: u16 = 1;
const MAX_PAYLOAD_BYTES: usize = 256 * 1024 * 1024;
const MAX_STRING_BYTES: usize = 16 * 1024 * 1024;
const MAX_RECORDS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRestartWireVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRestartWireEncoding {
    /// Ledger, mutation, evidence-basis, calibration, and uncertainty fields are
    /// explicitly typed. Revision policy and gate-decision internals remain opaque
    /// version-pinned Debug strings in this first envelope.
    ExplicitFieldsWithOpaqueRevisionPolicyDecisionV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireManifestSummaryV1 {
    pub captured_at_cycle: u64,
    pub claim_count: u64,
    pub evidence_count: u64,
    pub provenance_count: u64,
    pub next_claim_id: ClaimId,
    pub next_evidence_id: EvidenceId,
    pub next_provenance_id: ProvenanceId,
    pub ledger_digest: [u8; 32],
    pub mutation_capsule_digest: [u8; 32],
    pub revision_capsule_digest: [u8; 32],
    pub manifest_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireProvenanceV1 {
    pub id: ProvenanceId,
    pub source_label: String,
    pub source_uri: Option<String>,
    pub content_hash: Option<String>,
    pub recorded_at_cycle: u64,
    pub parent_ids: Vec<ProvenanceId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireClaimV1 {
    pub id: ClaimId,
    pub statement: String,
    pub kind: ClaimKind,
    pub domain: Option<String>,
    pub scope: Option<String>,
    pub created_at_cycle: u64,
    pub evidence_ids: Vec<EvidenceId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireEvidenceV1 {
    pub id: EvidenceId,
    pub claim_id: ClaimId,
    pub kind: EvidenceKind,
    pub polarity: EvidencePolarity,
    pub provenance_id: ProvenanceId,
    pub observed_at_cycle: u64,
    pub context: Option<String>,
    pub method: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WireSupportStateV1 {
    pub claim_id: ClaimId,
    pub baseline_support: BoundedWeight,
    pub current_support: BoundedWeight,
    pub revision: u64,
    pub initialized_at_cycle: u64,
    pub last_updated_cycle: u64,
    pub last_mutation_id: Option<BeliefMutationReceiptId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WireMutationV1 {
    pub id: BeliefMutationReceiptId,
    pub source_revision_receipt_id: BeliefRevisionReceiptId,
    pub claim_id: ClaimId,
    pub proposed_delta: f32,
    pub support_before: BoundedWeight,
    pub support_after: BoundedWeight,
    pub state_revision_before: u64,
    pub state_revision_after: u64,
    pub authorization_id: String,
    pub authority_label: String,
    pub authorized_at_cycle: u64,
    pub applied_at_cycle: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireRevisionBasisV1 {
    pub requested_id: EvidenceId,
    pub snapshot: Option<RevisionEvidenceSnapshot>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WireUncertaintyAssessmentV1 {
    pub claim_id: ClaimId,
    pub epistemic: Option<f64>,
    pub aleatoric: Option<f64>,
    pub ontological: Option<f64>,
    pub distribution_shift: Option<f64>,
    pub basis_evidence_ids: Vec<EvidenceId>,
    pub assessed_at_cycle: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WireRevisionReceiptV1 {
    pub id: BeliefRevisionReceiptId,
    pub claim_id: ClaimId,
    pub proposed_delta: f32,
    pub rationale: String,
    pub basis: Vec<WireRevisionBasisV1>,
    pub duplicate_basis_evidence_ids: Vec<EvidenceId>,
    /// Opaque V1 compatibility field; not a stable cross-version schema.
    pub policy_debug: String,
    pub calibration: Option<(u64, f64)>,
    pub uncertainty: Option<WireUncertaintyAssessmentV1>,
    pub decision_eligible: bool,
    pub declared_provenance_root_count: u64,
    /// Opaque V1 compatibility field; not a stable cross-version schema.
    pub decision_debug: String,
    pub evaluated_at_cycle: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicRestartWireSnapshotV1 {
    pub version: EpistemicRestartWireVersion,
    pub encoding: EpistemicRestartWireEncoding,
    pub captured_at_cycle: u64,
    pub manifest: WireManifestSummaryV1,
    pub provenance: Vec<WireProvenanceV1>,
    pub claims: Vec<WireClaimV1>,
    pub evidence: Vec<WireEvidenceV1>,
    pub support_states: Vec<WireSupportStateV1>,
    pub mutations: Vec<WireMutationV1>,
    pub consumed_authorization_count: u64,
    pub revision_next_receipt_id: BeliefRevisionReceiptId,
    pub revisions: Vec<WireRevisionReceiptV1>,
    pub wire_checksum: [u8; 32],
}

pub struct EpistemicRestartWireV1;

impl EpistemicRestartWireV1 {
    pub fn encode(capsule: &EpistemicRestartCapsuleV1) -> Result<Vec<u8>, EpistemicRestartWireError> {
        let mut payload = WireWriter::new();
        payload.u8(1); // encoding tag
        payload.u64(capsule.captured_at_cycle());

        let manifest = capsule.manifest();
        let lineage = manifest.ledger_lineage();
        payload.u64(manifest.captured_at_cycle());
        payload.usize(lineage.claim_count)?;
        payload.usize(lineage.evidence_count)?;
        payload.usize(lineage.provenance_count)?;
        payload.u64(lineage.next_claim_id.0);
        payload.u64(lineage.next_evidence_id.0);
        payload.u64(lineage.next_provenance_id.0);
        payload.bytes32(manifest.ledger_digest().as_bytes());
        payload.bytes32(manifest.mutation_capsule_digest().as_bytes());
        payload.bytes32(manifest.revision_capsule_digest().as_bytes());
        payload.bytes32(manifest.manifest_digest().as_bytes());

        let ledger = capsule.ledger_payload();
        payload.count(ledger.provenance().len())?;
        for record in ledger.provenance() {
            payload.u64(record.id.0);
            payload.string(&record.source_label)?;
            payload.optional_string(record.source_uri.as_deref())?;
            payload.optional_string(record.content_hash.as_deref())?;
            payload.u64(record.recorded_at_cycle);
            payload.count(record.parent_ids.len())?;
            for parent in &record.parent_ids {
                payload.u64(parent.0);
            }
        }

        payload.count(ledger.claims().len())?;
        for record in ledger.claims() {
            payload.u64(record.id.0);
            payload.string(&record.statement)?;
            payload.u8(claim_kind_tag(record.kind));
            payload.optional_string(record.domain.as_deref())?;
            payload.optional_string(record.scope.as_deref())?;
            payload.u64(record.created_at_cycle);
            payload.count(record.evidence_ids.len())?;
            for id in &record.evidence_ids {
                payload.u64(id.0);
            }
        }

        payload.count(ledger.evidence().len())?;
        for record in ledger.evidence() {
            payload.u64(record.id.0);
            payload.u64(record.claim_id.0);
            payload.u8(evidence_kind_tag(record.kind));
            payload.u8(evidence_polarity_tag(record.polarity));
            payload.u64(record.provenance_id.0);
            payload.u64(record.observed_at_cycle);
            payload.optional_string(record.context.as_deref())?;
            payload.optional_string(record.method.as_deref())?;
        }

        let mutations = capsule.mutation_capsule();
        payload.count(mutations.states().len())?;
        for state in mutations.states() {
            payload.u64(state.claim_id.0);
            payload.f32(state.baseline_support.get());
            payload.f32(state.current_support.get());
            payload.u64(state.revision);
            payload.u64(state.initialized_at_cycle);
            payload.u64(state.last_updated_cycle);
            match state.last_mutation_id {
                Some(id) => {
                    payload.bool(true);
                    payload.u64(id.0);
                }
                None => payload.bool(false),
            }
        }

        payload.count(mutations.mutations().len())?;
        for mutation in mutations.mutations() {
            payload.u64(mutation.id.0);
            payload.u64(mutation.source_revision_receipt_id.0);
            payload.u64(mutation.claim_id.0);
            payload.f32(mutation.proposed_delta);
            payload.f32(mutation.support_before.get());
            payload.f32(mutation.support_after.get());
            payload.u64(mutation.state_revision_before);
            payload.u64(mutation.state_revision_after);
            payload.string(&mutation.authorization_id)?;
            payload.string(&mutation.authority_label)?;
            payload.u64(mutation.authorized_at_cycle);
            payload.u64(mutation.applied_at_cycle);
        }
        payload.usize(mutations.consumed_authorization_count())?;

        let revisions = capsule.revision_capsule();
        payload.u64(revisions.next_receipt_id().0);
        payload.count(revisions.receipts().len())?;
        for receipt in revisions.receipts() {
            payload.u64(receipt.id().0);
            payload.u64(receipt.claim_id().0);
            payload.f32(receipt.proposed_delta());
            payload.string(receipt.rationale())?;
            payload.count(receipt.basis().len())?;
            for basis in receipt.basis() {
                encode_revision_basis(&mut payload, basis)?;
            }
            payload.count(receipt.duplicate_basis_evidence_ids().len())?;
            for id in receipt.duplicate_basis_evidence_ids() {
                payload.u64(id.0);
            }
            payload.string(&format!("{:?}", receipt.policy()))?;
            match receipt.calibration() {
                Some(calibration) => {
                    payload.bool(true);
                    payload.u64(calibration.sample_count);
                    payload.f64(calibration.ece);
                }
                None => payload.bool(false),
            }
            match receipt.uncertainty() {
                Some(assessment) => {
                    payload.bool(true);
                    payload.u64(assessment.claim_id.0);
                    encode_optional_uncertainty(
                        &mut payload,
                        assessment.vector.get(UncertaintyDimension::Epistemic).map(|v| v.get()),
                    );
                    encode_optional_uncertainty(
                        &mut payload,
                        assessment.vector.get(UncertaintyDimension::Aleatoric).map(|v| v.get()),
                    );
                    encode_optional_uncertainty(
                        &mut payload,
                        assessment.vector.get(UncertaintyDimension::Ontological).map(|v| v.get()),
                    );
                    encode_optional_uncertainty(
                        &mut payload,
                        assessment
                            .vector
                            .get(UncertaintyDimension::DistributionShift)
                            .map(|v| v.get()),
                    );
                    payload.count(assessment.basis_evidence_ids.len())?;
                    for id in &assessment.basis_evidence_ids {
                        payload.u64(id.0);
                    }
                    payload.u64(assessment.assessed_at_cycle);
                }
                None => payload.bool(false),
            }
            payload.bool(receipt.decision().eligible());
            payload.usize(receipt.decision().declared_provenance_root_count())?;
            payload.string(&format!("{:?}", receipt.decision()))?;
            payload.u64(receipt.evaluated_at_cycle());
        }

        let payload = payload.finish();
        if payload.len() > MAX_PAYLOAD_BYTES {
            return Err(EpistemicRestartWireError::PayloadTooLarge(payload.len()));
        }
        let payload_len = u64::try_from(payload.len())
            .map_err(|_| EpistemicRestartWireError::LengthOverflow)?;
        let checksum = wire_checksum(VERSION, payload_len, &payload);

        let mut out = Vec::with_capacity(MAGIC.len() + 2 + 8 + payload.len() + 32);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&payload_len.to_le_bytes());
        out.extend_from_slice(&payload);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<EpistemicRestartWireSnapshotV1, EpistemicRestartWireError> {
        let minimum = MAGIC.len() + 2 + 8 + 32;
        if bytes.len() < minimum {
            return Err(EpistemicRestartWireError::Truncated);
        }
        if &bytes[..MAGIC.len()] != MAGIC {
            return Err(EpistemicRestartWireError::BadMagic);
        }
        let version_offset = MAGIC.len();
        let version = u16::from_le_bytes([
            bytes[version_offset],
            bytes[version_offset + 1],
        ]);
        if version != VERSION {
            return Err(EpistemicRestartWireError::UnsupportedVersion(version));
        }
        let len_offset = version_offset + 2;
        let payload_len_u64 = u64::from_le_bytes(
            bytes[len_offset..len_offset + 8]
                .try_into()
                .expect("slice length checked"),
        );
        let payload_len = usize::try_from(payload_len_u64)
            .map_err(|_| EpistemicRestartWireError::LengthOverflow)?;
        if payload_len > MAX_PAYLOAD_BYTES {
            return Err(EpistemicRestartWireError::PayloadTooLarge(payload_len));
        }
        let expected_len = minimum
            .checked_add(payload_len)
            .ok_or(EpistemicRestartWireError::LengthOverflow)?;
        if bytes.len() != expected_len {
            return Err(EpistemicRestartWireError::EnvelopeLengthMismatch {
                declared_payload: payload_len,
                actual_total: bytes.len(),
            });
        }
        let payload_start = len_offset + 8;
        let payload_end = payload_start + payload_len;
        let payload = &bytes[payload_start..payload_end];
        let checksum: [u8; 32] = bytes[payload_end..]
            .try_into()
            .expect("checksum length checked by envelope length");
        let expected_checksum = wire_checksum(version, payload_len_u64, payload);
        if checksum != expected_checksum {
            return Err(EpistemicRestartWireError::ChecksumMismatch);
        }

        let mut reader = WireReader::new(payload);
        let encoding = match reader.u8()? {
            1 => EpistemicRestartWireEncoding::ExplicitFieldsWithOpaqueRevisionPolicyDecisionV1,
            tag => return Err(EpistemicRestartWireError::UnknownEncoding(tag)),
        };
        let captured_at_cycle = reader.u64()?;
        let manifest_cycle = reader.u64()?;
        let manifest = WireManifestSummaryV1 {
            captured_at_cycle: manifest_cycle,
            claim_count: reader.u64()?,
            evidence_count: reader.u64()?,
            provenance_count: reader.u64()?,
            next_claim_id: ClaimId(reader.u64()?),
            next_evidence_id: EvidenceId(reader.u64()?),
            next_provenance_id: ProvenanceId(reader.u64()?),
            ledger_digest: reader.bytes32()?,
            mutation_capsule_digest: reader.bytes32()?,
            revision_capsule_digest: reader.bytes32()?,
            manifest_digest: reader.bytes32()?,
        };
        if manifest_cycle != captured_at_cycle {
            return Err(EpistemicRestartWireError::CaptureCycleMismatch {
                envelope: captured_at_cycle,
                manifest: manifest_cycle,
            });
        }

        let provenance_count = reader.count()?;
        let mut provenance = Vec::with_capacity(provenance_count);
        for _ in 0..provenance_count {
            let id = ProvenanceId(reader.u64()?);
            let source_label = reader.string()?;
            let source_uri = reader.optional_string()?;
            let content_hash = reader.optional_string()?;
            let recorded_at_cycle = reader.u64()?;
            let parent_count = reader.count()?;
            let mut parent_ids = Vec::with_capacity(parent_count);
            for _ in 0..parent_count {
                parent_ids.push(ProvenanceId(reader.u64()?));
            }
            provenance.push(WireProvenanceV1 {
                id,
                source_label,
                source_uri,
                content_hash,
                recorded_at_cycle,
                parent_ids,
            });
        }

        let claim_count = reader.count()?;
        let mut claims = Vec::with_capacity(claim_count);
        for _ in 0..claim_count {
            let id = ClaimId(reader.u64()?);
            let statement = reader.string()?;
            let kind = parse_claim_kind(reader.u8()?)?;
            let domain = reader.optional_string()?;
            let scope = reader.optional_string()?;
            let created_at_cycle = reader.u64()?;
            let evidence_count = reader.count()?;
            let mut evidence_ids = Vec::with_capacity(evidence_count);
            for _ in 0..evidence_count {
                evidence_ids.push(EvidenceId(reader.u64()?));
            }
            claims.push(WireClaimV1 {
                id,
                statement,
                kind,
                domain,
                scope,
                created_at_cycle,
                evidence_ids,
            });
        }

        let evidence_count = reader.count()?;
        let mut evidence = Vec::with_capacity(evidence_count);
        for _ in 0..evidence_count {
            evidence.push(WireEvidenceV1 {
                id: EvidenceId(reader.u64()?),
                claim_id: ClaimId(reader.u64()?),
                kind: parse_evidence_kind(reader.u8()?)?,
                polarity: parse_evidence_polarity(reader.u8()?)?,
                provenance_id: ProvenanceId(reader.u64()?),
                observed_at_cycle: reader.u64()?,
                context: reader.optional_string()?,
                method: reader.optional_string()?,
            });
        }

        let state_count = reader.count()?;
        let mut support_states = Vec::with_capacity(state_count);
        for _ in 0..state_count {
            let claim_id = ClaimId(reader.u64()?);
            let baseline_support = parse_weight(reader.f32()?)?;
            let current_support = parse_weight(reader.f32()?)?;
            let revision = reader.u64()?;
            let initialized_at_cycle = reader.u64()?;
            let last_updated_cycle = reader.u64()?;
            let last_mutation_id = if reader.bool()? {
                Some(BeliefMutationReceiptId(reader.u64()?))
            } else {
                None
            };
            support_states.push(WireSupportStateV1 {
                claim_id,
                baseline_support,
                current_support,
                revision,
                initialized_at_cycle,
                last_updated_cycle,
                last_mutation_id,
            });
        }

        let mutation_count = reader.count()?;
        let mut mutations = Vec::with_capacity(mutation_count);
        for _ in 0..mutation_count {
            mutations.push(WireMutationV1 {
                id: BeliefMutationReceiptId(reader.u64()?),
                source_revision_receipt_id: BeliefRevisionReceiptId(reader.u64()?),
                claim_id: ClaimId(reader.u64()?),
                proposed_delta: reader.f32()?,
                support_before: parse_weight(reader.f32()?)?,
                support_after: parse_weight(reader.f32()?)?,
                state_revision_before: reader.u64()?,
                state_revision_after: reader.u64()?,
                authorization_id: reader.string()?,
                authority_label: reader.string()?,
                authorized_at_cycle: reader.u64()?,
                applied_at_cycle: reader.u64()?,
            });
        }
        let consumed_authorization_count = reader.u64()?;

        let revision_next_receipt_id = BeliefRevisionReceiptId(reader.u64()?);
        let revision_count = reader.count()?;
        let mut revisions = Vec::with_capacity(revision_count);
        for _ in 0..revision_count {
            let id = BeliefRevisionReceiptId(reader.u64()?);
            let claim_id = ClaimId(reader.u64()?);
            let proposed_delta = reader.f32()?;
            let rationale = reader.string()?;
            let basis_count = reader.count()?;
            let mut basis = Vec::with_capacity(basis_count);
            for _ in 0..basis_count {
                basis.push(decode_revision_basis(&mut reader)?);
            }
            let duplicate_count = reader.count()?;
            let mut duplicate_basis_evidence_ids = Vec::with_capacity(duplicate_count);
            for _ in 0..duplicate_count {
                duplicate_basis_evidence_ids.push(EvidenceId(reader.u64()?));
            }
            let policy_debug = reader.string()?;
            let calibration = if reader.bool()? {
                let sample_count = reader.u64()?;
                let ece = reader.f64()?;
                if !ece.is_finite() || !(0.0..=1.0).contains(&ece) {
                    return Err(EpistemicRestartWireError::InvalidCalibration(ece));
                }
                Some((sample_count, ece))
            } else {
                None
            };
            let uncertainty = if reader.bool()? {
                let claim_id = ClaimId(reader.u64()?);
                let epistemic = decode_optional_uncertainty(&mut reader)?;
                let aleatoric = decode_optional_uncertainty(&mut reader)?;
                let ontological = decode_optional_uncertainty(&mut reader)?;
                let distribution_shift = decode_optional_uncertainty(&mut reader)?;
                let basis_count = reader.count()?;
                let mut basis_evidence_ids = Vec::with_capacity(basis_count);
                for _ in 0..basis_count {
                    basis_evidence_ids.push(EvidenceId(reader.u64()?));
                }
                let assessed_at_cycle = reader.u64()?;
                Some(WireUncertaintyAssessmentV1 {
                    claim_id,
                    epistemic,
                    aleatoric,
                    ontological,
                    distribution_shift,
                    basis_evidence_ids,
                    assessed_at_cycle,
                })
            } else {
                None
            };
            let decision_eligible = reader.bool()?;
            let declared_provenance_root_count = reader.u64()?;
            let decision_debug = reader.string()?;
            let evaluated_at_cycle = reader.u64()?;
            revisions.push(WireRevisionReceiptV1 {
                id,
                claim_id,
                proposed_delta,
                rationale,
                basis,
                duplicate_basis_evidence_ids,
                policy_debug,
                calibration,
                uncertainty,
                decision_eligible,
                declared_provenance_root_count,
                decision_debug,
                evaluated_at_cycle,
            });
        }

        if !reader.is_finished() {
            return Err(EpistemicRestartWireError::TrailingPayloadBytes(
                reader.remaining(),
            ));
        }

        Ok(EpistemicRestartWireSnapshotV1 {
            version: EpistemicRestartWireVersion::V1,
            encoding,
            captured_at_cycle,
            manifest,
            provenance,
            claims,
            evidence,
            support_states,
            mutations,
            consumed_authorization_count,
            revision_next_receipt_id,
            revisions,
            wire_checksum: checksum,
        })
    }
}

fn encode_revision_basis(
    writer: &mut WireWriter,
    basis: &RevisionEvidenceReference,
) -> Result<(), EpistemicRestartWireError> {
    writer.u64(basis.requested_id.0);
    match &basis.snapshot {
        Some(snapshot) => {
            writer.bool(true);
            writer.u64(snapshot.evidence_id.0);
            writer.u64(snapshot.claim_id.0);
            writer.u8(evidence_kind_tag(snapshot.kind));
            writer.u8(evidence_polarity_tag(snapshot.polarity));
            writer.u64(snapshot.provenance_id.0);
            writer.u64(snapshot.observed_at_cycle);
            writer.optional_string(snapshot.context.as_deref())?;
            writer.optional_string(snapshot.method.as_deref())?;
        }
        None => writer.bool(false),
    }
    Ok(())
}

fn decode_revision_basis(
    reader: &mut WireReader<'_>,
) -> Result<WireRevisionBasisV1, EpistemicRestartWireError> {
    let requested_id = EvidenceId(reader.u64()?);
    let snapshot = if reader.bool()? {
        Some(RevisionEvidenceSnapshot {
            evidence_id: EvidenceId(reader.u64()?),
            claim_id: ClaimId(reader.u64()?),
            kind: parse_evidence_kind(reader.u8()?)?,
            polarity: parse_evidence_polarity(reader.u8()?)?,
            provenance_id: ProvenanceId(reader.u64()?),
            observed_at_cycle: reader.u64()?,
            context: reader.optional_string()?,
            method: reader.optional_string()?,
        })
    } else {
        None
    };
    Ok(WireRevisionBasisV1 {
        requested_id,
        snapshot,
    })
}

fn encode_optional_uncertainty(writer: &mut WireWriter, value: Option<f64>) {
    match value {
        Some(value) => {
            writer.bool(true);
            writer.f64(value);
        }
        None => writer.bool(false),
    }
}

fn decode_optional_uncertainty(
    reader: &mut WireReader<'_>,
) -> Result<Option<f64>, EpistemicRestartWireError> {
    if !reader.bool()? {
        return Ok(None);
    }
    let value = reader.f64()?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(EpistemicRestartWireError::InvalidUncertainty(value));
    }
    Ok(Some(value))
}

fn wire_checksum(version: u16, payload_len: u64, payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-wire-checksum-v1");
    hasher.update(&version.to_le_bytes());
    hasher.update(&payload_len.to_le_bytes());
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

struct WireWriter {
    bytes: Vec<u8>,
}

impl WireWriter {
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

    fn usize(&mut self, value: usize) -> Result<(), EpistemicRestartWireError> {
        let value = u64::try_from(value).map_err(|_| EpistemicRestartWireError::LengthOverflow)?;
        self.u64(value);
        Ok(())
    }

    fn count(&mut self, value: usize) -> Result<(), EpistemicRestartWireError> {
        if value > MAX_RECORDS {
            return Err(EpistemicRestartWireError::RecordCountTooLarge(value));
        }
        self.usize(value)
    }

    fn f32(&mut self, value: f32) {
        self.bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    fn f64(&mut self, value: f64) {
        self.bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    fn bytes32(&mut self, value: [u8; 32]) {
        self.bytes.extend_from_slice(&value);
    }

    fn string(&mut self, value: &str) -> Result<(), EpistemicRestartWireError> {
        let bytes = value.as_bytes();
        if bytes.len() > MAX_STRING_BYTES {
            return Err(EpistemicRestartWireError::StringTooLarge(bytes.len()));
        }
        self.usize(bytes.len())?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn optional_string(&mut self, value: Option<&str>) -> Result<(), EpistemicRestartWireError> {
        match value {
            Some(value) => {
                self.bool(true);
                self.string(value)?;
            }
            None => self.bool(false),
        }
        Ok(())
    }
}

struct WireReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> WireReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], EpistemicRestartWireError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(EpistemicRestartWireError::LengthOverflow)?;
        if end > self.bytes.len() {
            return Err(EpistemicRestartWireError::Truncated);
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, EpistemicRestartWireError> {
        Ok(self.take(1)?[0])
    }

    fn bool(&mut self) -> Result<bool, EpistemicRestartWireError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(EpistemicRestartWireError::InvalidBoolean(value)),
        }
    }

    fn u64(&mut self) -> Result<u64, EpistemicRestartWireError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .expect("reader returned exact u64 width"),
        ))
    }

    fn count(&mut self) -> Result<usize, EpistemicRestartWireError> {
        let raw = self.u64()?;
        let value = usize::try_from(raw).map_err(|_| EpistemicRestartWireError::LengthOverflow)?;
        if value > MAX_RECORDS {
            return Err(EpistemicRestartWireError::RecordCountTooLarge(value));
        }
        Ok(value)
    }

    fn f32(&mut self) -> Result<f32, EpistemicRestartWireError> {
        Ok(f32::from_bits(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .expect("reader returned exact f32 width"),
        )))
    }

    fn f64(&mut self) -> Result<f64, EpistemicRestartWireError> {
        Ok(f64::from_bits(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .expect("reader returned exact f64 width"),
        )))
    }

    fn bytes32(&mut self) -> Result<[u8; 32], EpistemicRestartWireError> {
        Ok(self
            .take(32)?
            .try_into()
            .expect("reader returned exact digest width"))
    }

    fn string(&mut self) -> Result<String, EpistemicRestartWireError> {
        let len = self.u64()?;
        let len = usize::try_from(len).map_err(|_| EpistemicRestartWireError::LengthOverflow)?;
        if len > MAX_STRING_BYTES {
            return Err(EpistemicRestartWireError::StringTooLarge(len));
        }
        let bytes = self.take(len)?;
        let value = std::str::from_utf8(bytes)
            .map_err(|_| EpistemicRestartWireError::InvalidUtf8)?;
        Ok(value.to_owned())
    }

    fn optional_string(&mut self) -> Result<Option<String>, EpistemicRestartWireError> {
        if self.bool()? {
            Ok(Some(self.string()?))
        } else {
            Ok(None)
        }
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }
}

fn parse_weight(value: f32) -> Result<BoundedWeight, EpistemicRestartWireError> {
    BoundedWeight::new(value).map_err(|_| EpistemicRestartWireError::InvalidWeight(value))
}

fn claim_kind_tag(value: ClaimKind) -> u8 {
    match value {
        ClaimKind::Descriptive => 1,
        ClaimKind::Predictive => 2,
        ClaimKind::Causal => 3,
        ClaimKind::Counterfactual => 4,
        ClaimKind::Procedural => 5,
        ClaimKind::Normative => 6,
    }
}

fn parse_claim_kind(tag: u8) -> Result<ClaimKind, EpistemicRestartWireError> {
    match tag {
        1 => Ok(ClaimKind::Descriptive),
        2 => Ok(ClaimKind::Predictive),
        3 => Ok(ClaimKind::Causal),
        4 => Ok(ClaimKind::Counterfactual),
        5 => Ok(ClaimKind::Procedural),
        6 => Ok(ClaimKind::Normative),
        _ => Err(EpistemicRestartWireError::UnknownClaimKind(tag)),
    }
}

fn evidence_kind_tag(value: EvidenceKind) -> u8 {
    match value {
        EvidenceKind::Report => 1,
        EvidenceKind::Observation => 2,
        EvidenceKind::Measurement => 3,
        EvidenceKind::Intervention => 4,
        EvidenceKind::Replication => 5,
        EvidenceKind::Simulation => 6,
        EvidenceKind::Deduction => 7,
        EvidenceKind::ToolResult => 8,
    }
}

fn parse_evidence_kind(tag: u8) -> Result<EvidenceKind, EpistemicRestartWireError> {
    match tag {
        1 => Ok(EvidenceKind::Report),
        2 => Ok(EvidenceKind::Observation),
        3 => Ok(EvidenceKind::Measurement),
        4 => Ok(EvidenceKind::Intervention),
        5 => Ok(EvidenceKind::Replication),
        6 => Ok(EvidenceKind::Simulation),
        7 => Ok(EvidenceKind::Deduction),
        8 => Ok(EvidenceKind::ToolResult),
        _ => Err(EpistemicRestartWireError::UnknownEvidenceKind(tag)),
    }
}

fn evidence_polarity_tag(value: EvidencePolarity) -> u8 {
    match value {
        EvidencePolarity::Supports => 1,
        EvidencePolarity::Contradicts => 2,
        EvidencePolarity::Contextualizes => 3,
    }
}

fn parse_evidence_polarity(tag: u8) -> Result<EvidencePolarity, EpistemicRestartWireError> {
    match tag {
        1 => Ok(EvidencePolarity::Supports),
        2 => Ok(EvidencePolarity::Contradicts),
        3 => Ok(EvidencePolarity::Contextualizes),
        _ => Err(EpistemicRestartWireError::UnknownEvidencePolarity(tag)),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicRestartWireError {
    BadMagic,
    UnsupportedVersion(u16),
    UnknownEncoding(u8),
    PayloadTooLarge(usize),
    StringTooLarge(usize),
    RecordCountTooLarge(usize),
    LengthOverflow,
    EnvelopeLengthMismatch {
        declared_payload: usize,
        actual_total: usize,
    },
    Truncated,
    ChecksumMismatch,
    InvalidUtf8,
    InvalidBoolean(u8),
    UnknownClaimKind(u8),
    UnknownEvidenceKind(u8),
    UnknownEvidencePolarity(u8),
    InvalidWeight(f32),
    InvalidCalibration(f64),
    InvalidUncertainty(f64),
    CaptureCycleMismatch {
        envelope: u64,
        manifest: u64,
    },
    TrailingPayloadBytes(usize),
}

impl fmt::Display for EpistemicRestartWireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epistemic restart wire invalid: {self:?}")
    }
}

impl Error for EpistemicRestartWireError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationAuthority, BeliefMutationAuthorization,
        BeliefMutationAuthorizationDecision, BeliefMutationPersistenceCapsuleV1,
        BeliefRevisionHistory, BeliefRevisionHistoryCapsuleV1, BeliefRevisionPolicy,
        EpistemicLedger, EpistemicLedgerInventoryV1, EpistemicRevisionProposal,
        EpistemicRestartCapsuleV1, EvidenceKind, EvidencePolarity,
    };

    fn capsule() -> EpistemicRestartCapsuleV1 {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("lab", Some("lab://a".into()), Some("hash".into()), 1, vec![])
            .unwrap();
        let claim = ledger.add_claim(
            "X predicts Y",
            ClaimKind::Predictive,
            Some("test".into()),
            None,
            1,
        );
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                provenance,
                2,
                Some("fixture".into()),
                Some("protocol-v1".into()),
            )
            .unwrap();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
            .unwrap();
        let rejected = EpistemicRevisionProposal::new(claim, -0.10, vec![evidence], "reject")
            .unwrap();
        let mut history = BeliefRevisionHistory::new();
        let mut authority = BeliefMutationAuthority::new();
        let prepared = authority
            .prepare(&ledger, &mut history, &proposal, &policy, None, None, 3)
            .unwrap();
        history
            .evaluate_and_record(&ledger, &rejected, &policy, None, None, 4)
            .unwrap();
        let mut store = EpistemicSupportStore::new();
        store
            .register_claim(&ledger, claim, BoundedWeight::new(0.5).unwrap(), 3)
            .unwrap();
        let authorization = BeliefMutationAuthorization::new(
            "auth-1",
            "test-authority",
            BeliefMutationAuthorizationDecision::Approved,
            4,
            prepared.receipt(),
            store.state(claim).unwrap(),
        )
        .unwrap();
        authority
            .apply(&ledger, &mut store, &prepared, &authorization, 5)
            .unwrap();
        let mutations = BeliefMutationPersistenceCapsuleV1::capture(&store, &[claim], 6).unwrap();
        let revisions = BeliefRevisionHistoryCapsuleV1::capture(&history, &mutations, 6).unwrap();
        let inventory =
            EpistemicLedgerInventoryV1::new(vec![claim], vec![evidence], vec![provenance]).unwrap();
        EpistemicRestartCapsuleV1::capture(&ledger, &inventory, &mutations, &revisions, 6).unwrap()
    }

    #[test]
    fn roundtrip_preserves_structured_restart_snapshot() {
        let capsule = capsule();
        let bytes = EpistemicRestartWireV1::encode(&capsule).unwrap();
        let snapshot = EpistemicRestartWireV1::decode(&bytes).unwrap();
        assert_eq!(snapshot.version, EpistemicRestartWireVersion::V1);
        assert_eq!(snapshot.captured_at_cycle, 6);
        assert_eq!(snapshot.manifest.manifest_digest, capsule.manifest_digest().as_bytes());
        assert_eq!(snapshot.provenance.len(), 1);
        assert_eq!(snapshot.claims.len(), 1);
        assert_eq!(snapshot.evidence.len(), 1);
        assert_eq!(snapshot.support_states.len(), 1);
        assert_eq!(snapshot.mutations.len(), 1);
        assert_eq!(snapshot.revisions.len(), 2);
        assert!(snapshot.revisions[0].decision_eligible);
        assert!(!snapshot.revisions[1].decision_eligible);
        assert!(!snapshot.revisions[0].policy_debug.is_empty());
        assert!(!snapshot.revisions[0].decision_debug.is_empty());
    }

    #[test]
    fn one_byte_payload_tamper_fails_checksum() {
        let capsule = capsule();
        let mut bytes = EpistemicRestartWireV1::encode(&capsule).unwrap();
        let payload_start = MAGIC.len() + 2 + 8;
        bytes[payload_start + 3] ^= 0x01;
        assert_eq!(
            EpistemicRestartWireV1::decode(&bytes).unwrap_err(),
            EpistemicRestartWireError::ChecksumMismatch
        );
    }

    #[test]
    fn unsupported_version_fails_before_payload_decode() {
        let capsule = capsule();
        let mut bytes = EpistemicRestartWireV1::encode(&capsule).unwrap();
        let offset = MAGIC.len();
        bytes[offset..offset + 2].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            EpistemicRestartWireV1::decode(&bytes).unwrap_err(),
            EpistemicRestartWireError::UnsupportedVersion(2)
        );
    }

    #[test]
    fn truncated_envelope_fails_closed() {
        let capsule = capsule();
        let mut bytes = EpistemicRestartWireV1::encode(&capsule).unwrap();
        bytes.pop();
        assert!(matches!(
            EpistemicRestartWireV1::decode(&bytes),
            Err(EpistemicRestartWireError::EnvelopeLengthMismatch { .. })
                | Err(EpistemicRestartWireError::Truncated)
        ));
    }

    #[test]
    fn trailing_bytes_are_not_ignored() {
        let capsule = capsule();
        let mut bytes = EpistemicRestartWireV1::encode(&capsule).unwrap();
        bytes.push(0);
        assert!(matches!(
            EpistemicRestartWireV1::decode(&bytes),
            Err(EpistemicRestartWireError::EnvelopeLengthMismatch { .. })
        ));
    }
}
