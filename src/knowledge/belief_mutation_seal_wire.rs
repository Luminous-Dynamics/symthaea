// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded canonical wire envelope for EKM-056 mutation-time evidence seals.
//!
//! Decoding produces an untrusted DTO only. It does not construct an EKM-056
//! capsule, replay a mutation, hydrate writable state, or authorize restart.
//! Cross-checking the record digests against the base restart mutation history is
//! intentionally deferred to a separate semantic validator.

use super::belief_mutation_firewall::BeliefMutationReceiptId;
use super::belief_mutation_seal_persistence::BeliefMutationEvidenceSealCapsuleV1;
use super::belief_revision_receipt::{BeliefRevisionReceiptId, RevisionEvidenceSnapshot};
use super::belief_mutation_transaction::SealedClaimSnapshot;
use super::claim_evidence::{
    ClaimId, ClaimKind, EvidenceId, EvidenceKind, EvidencePolarity, ProvenanceId,
};
use std::error::Error;
use std::fmt;

const MAGIC: &[u8] = b"SYMTHAEA-EKM-MUT-SEAL";
const VERSION: u16 = 1;
const MAX_PAYLOAD_BYTES: usize = 256 * 1024 * 1024;
const MAX_STRING_BYTES: usize = 16 * 1024 * 1024;
const MAX_RECORDS: usize = 1_000_000;
const MAX_TOTAL_EVIDENCE: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefMutationSealWireVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefMutationSealWireEncoding {
    ExplicitMutationSealFieldsV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireBeliefMutationEvidenceSealV1 {
    pub mutation_id: BeliefMutationReceiptId,
    pub source_revision_receipt_id: BeliefRevisionReceiptId,
    pub claim_id: ClaimId,
    pub sealed_at_cycle: u64,
    pub applied_at_cycle: u64,
    pub claim: SealedClaimSnapshot,
    pub evidence: Vec<RevisionEvidenceSnapshot>,
    pub record_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeliefMutationSealWireSnapshotV1 {
    pub version: BeliefMutationSealWireVersion,
    pub encoding: BeliefMutationSealWireEncoding,
    pub captured_at_cycle: u64,
    pub linked_mutation_capture_cycle: u64,
    pub records: Vec<WireBeliefMutationEvidenceSealV1>,
    pub claimed_capsule_digest: [u8; 32],
    pub wire_checksum: [u8; 32],
}

pub struct BeliefMutationSealWireV1;

impl BeliefMutationSealWireV1 {
    pub fn encode(
        capsule: &BeliefMutationEvidenceSealCapsuleV1,
    ) -> Result<Vec<u8>, BeliefMutationSealWireError> {
        if capsule.records().len() > MAX_RECORDS {
            return Err(BeliefMutationSealWireError::TooManyRecords {
                actual: capsule.records().len(),
                maximum: MAX_RECORDS,
            });
        }
        let total_evidence = capsule
            .records()
            .iter()
            .try_fold(0usize, |total, record| {
                total
                    .checked_add(record.seal().evidence().len())
                    .ok_or(BeliefMutationSealWireError::LengthOverflow)
            })?;
        if total_evidence > MAX_TOTAL_EVIDENCE {
            return Err(BeliefMutationSealWireError::TooManyEvidenceRecords {
                actual: total_evidence,
                maximum: MAX_TOTAL_EVIDENCE,
            });
        }

        let mut payload = Vec::new();
        write_u8(&mut payload, 1); // encoding tag
        write_u64(&mut payload, capsule.captured_at_cycle());
        write_u64(&mut payload, capsule.linked_mutation_capture_cycle());
        write_len(&mut payload, capsule.records().len())?;
        for record in capsule.records() {
            write_u64(&mut payload, record.mutation_id().0);
            write_u64(&mut payload, record.source_revision_receipt_id().0);
            write_u64(&mut payload, record.claim_id().0);
            write_u64(&mut payload, record.sealed_at_cycle());
            write_u64(&mut payload, record.applied_at_cycle());
            write_claim(&mut payload, record.seal().claim())?;
            write_len(&mut payload, record.seal().evidence().len())?;
            for evidence in record.seal().evidence() {
                write_evidence(&mut payload, evidence)?;
            }
            payload.extend_from_slice(&record.record_digest().as_bytes());
        }
        payload.extend_from_slice(&capsule.capsule_digest().as_bytes());

        if payload.len() > MAX_PAYLOAD_BYTES {
            return Err(BeliefMutationSealWireError::PayloadTooLarge {
                actual: payload.len(),
                maximum: MAX_PAYLOAD_BYTES,
            });
        }

        let checksum = checksum(&payload);
        let mut out = Vec::with_capacity(
            MAGIC.len()
                + 2
                + 8
                + payload.len()
                + checksum.len(),
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(
            &u64::try_from(payload.len())
                .map_err(|_| BeliefMutationSealWireError::LengthOverflow)?
                .to_le_bytes(),
        );
        out.extend_from_slice(&payload);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<BeliefMutationSealWireSnapshotV1, BeliefMutationSealWireError> {
        let fixed = MAGIC.len() + 2 + 8 + 32;
        if bytes.len() < fixed {
            return Err(BeliefMutationSealWireError::TruncatedEnvelope);
        }
        if &bytes[..MAGIC.len()] != MAGIC {
            return Err(BeliefMutationSealWireError::InvalidMagic);
        }
        let mut offset = MAGIC.len();
        let version = read_u16_raw(bytes, &mut offset)?;
        if version != VERSION {
            return Err(BeliefMutationSealWireError::UnsupportedVersion(version));
        }
        let payload_len_u64 = read_u64_raw(bytes, &mut offset)?;
        let payload_len = usize::try_from(payload_len_u64)
            .map_err(|_| BeliefMutationSealWireError::LengthOverflow)?;
        if payload_len > MAX_PAYLOAD_BYTES {
            return Err(BeliefMutationSealWireError::PayloadTooLarge {
                actual: payload_len,
                maximum: MAX_PAYLOAD_BYTES,
            });
        }
        let expected_len = offset
            .checked_add(payload_len)
            .and_then(|value| value.checked_add(32))
            .ok_or(BeliefMutationSealWireError::LengthOverflow)?;
        if bytes.len() != expected_len {
            return Err(BeliefMutationSealWireError::EnvelopeLengthMismatch {
                declared: expected_len,
                actual: bytes.len(),
            });
        }
        let payload = &bytes[offset..offset + payload_len];
        let claimed_checksum: [u8; 32] = bytes[offset + payload_len..]
            .try_into()
            .map_err(|_| BeliefMutationSealWireError::TruncatedEnvelope)?;
        let actual_checksum = checksum(payload);
        if claimed_checksum != actual_checksum {
            return Err(BeliefMutationSealWireError::ChecksumMismatch);
        }

        let mut reader = Reader::new(payload);
        let encoding = match reader.u8()? {
            1 => BeliefMutationSealWireEncoding::ExplicitMutationSealFieldsV1,
            tag => return Err(BeliefMutationSealWireError::UnknownEncoding(tag)),
        };
        let captured_at_cycle = reader.u64()?;
        let linked_mutation_capture_cycle = reader.u64()?;
        let record_count = reader.len(MAX_RECORDS)?;
        let mut records = Vec::with_capacity(record_count);
        let mut total_evidence = 0usize;
        for _ in 0..record_count {
            let mutation_id = BeliefMutationReceiptId(reader.u64()?);
            let source_revision_receipt_id = BeliefRevisionReceiptId(reader.u64()?);
            let claim_id = ClaimId(reader.u64()?);
            let sealed_at_cycle = reader.u64()?;
            let applied_at_cycle = reader.u64()?;
            let claim = reader.claim()?;
            let evidence_count = reader.len(MAX_RECORDS)?;
            total_evidence = total_evidence
                .checked_add(evidence_count)
                .ok_or(BeliefMutationSealWireError::LengthOverflow)?;
            if total_evidence > MAX_TOTAL_EVIDENCE {
                return Err(BeliefMutationSealWireError::TooManyEvidenceRecords {
                    actual: total_evidence,
                    maximum: MAX_TOTAL_EVIDENCE,
                });
            }
            let mut evidence = Vec::with_capacity(evidence_count);
            for _ in 0..evidence_count {
                evidence.push(reader.evidence()?);
            }
            let record_digest = reader.bytes32()?;
            records.push(WireBeliefMutationEvidenceSealV1 {
                mutation_id,
                source_revision_receipt_id,
                claim_id,
                sealed_at_cycle,
                applied_at_cycle,
                claim,
                evidence,
                record_digest,
            });
        }
        let claimed_capsule_digest = reader.bytes32()?;
        if !reader.finished() {
            return Err(BeliefMutationSealWireError::TrailingPayloadBytes);
        }

        Ok(BeliefMutationSealWireSnapshotV1 {
            version: BeliefMutationSealWireVersion::V1,
            encoding,
            captured_at_cycle,
            linked_mutation_capture_cycle,
            records,
            claimed_capsule_digest,
            wire_checksum: claimed_checksum,
        })
    }
}

fn write_claim(
    out: &mut Vec<u8>,
    claim: &SealedClaimSnapshot,
) -> Result<(), BeliefMutationSealWireError> {
    write_u64(out, claim.claim_id.0);
    write_string(out, &claim.statement)?;
    write_u8(out, claim_kind_tag(claim.kind));
    write_optional_string(out, claim.domain.as_deref())?;
    write_optional_string(out, claim.scope.as_deref())?;
    write_u64(out, claim.created_at_cycle);
    Ok(())
}

fn write_evidence(
    out: &mut Vec<u8>,
    evidence: &RevisionEvidenceSnapshot,
) -> Result<(), BeliefMutationSealWireError> {
    write_u64(out, evidence.evidence_id.0);
    write_u64(out, evidence.claim_id.0);
    write_u8(out, evidence_kind_tag(evidence.kind));
    write_u8(out, evidence_polarity_tag(evidence.polarity));
    write_u64(out, evidence.provenance_id.0);
    write_u64(out, evidence.observed_at_cycle);
    write_optional_string(out, evidence.context.as_deref())?;
    write_optional_string(out, evidence.method.as_deref())?;
    Ok(())
}

fn write_optional_string(
    out: &mut Vec<u8>,
    value: Option<&str>,
) -> Result<(), BeliefMutationSealWireError> {
    match value {
        Some(value) => {
            write_u8(out, 1);
            write_string(out, value)?;
        }
        None => write_u8(out, 0),
    }
    Ok(())
}

fn write_string(out: &mut Vec<u8>, value: &str) -> Result<(), BeliefMutationSealWireError> {
    if value.len() > MAX_STRING_BYTES {
        return Err(BeliefMutationSealWireError::StringTooLarge {
            actual: value.len(),
            maximum: MAX_STRING_BYTES,
        });
    }
    write_len(out, value.len())?;
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn write_len(out: &mut Vec<u8>, value: usize) -> Result<(), BeliefMutationSealWireError> {
    write_u64(
        out,
        u64::try_from(value).map_err(|_| BeliefMutationSealWireError::LengthOverflow)?,
    );
    Ok(())
}

fn write_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn read_u16_raw(bytes: &[u8], offset: &mut usize) -> Result<u16, BeliefMutationSealWireError> {
    let end = offset
        .checked_add(2)
        .ok_or(BeliefMutationSealWireError::LengthOverflow)?;
    let raw: [u8; 2] = bytes
        .get(*offset..end)
        .ok_or(BeliefMutationSealWireError::TruncatedEnvelope)?
        .try_into()
        .map_err(|_| BeliefMutationSealWireError::TruncatedEnvelope)?;
    *offset = end;
    Ok(u16::from_le_bytes(raw))
}

fn read_u64_raw(bytes: &[u8], offset: &mut usize) -> Result<u64, BeliefMutationSealWireError> {
    let end = offset
        .checked_add(8)
        .ok_or(BeliefMutationSealWireError::LengthOverflow)?;
    let raw: [u8; 8] = bytes
        .get(*offset..end)
        .ok_or(BeliefMutationSealWireError::TruncatedEnvelope)?
        .try_into()
        .map_err(|_| BeliefMutationSealWireError::TruncatedEnvelope)?;
    *offset = end;
    Ok(u64::from_le_bytes(raw))
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], BeliefMutationSealWireError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(BeliefMutationSealWireError::LengthOverflow)?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or(BeliefMutationSealWireError::TruncatedPayload)?;
        self.offset = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, BeliefMutationSealWireError> {
        Ok(self.take(1)?[0])
    }

    fn u64(&mut self) -> Result<u64, BeliefMutationSealWireError> {
        let raw: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| BeliefMutationSealWireError::TruncatedPayload)?;
        Ok(u64::from_le_bytes(raw))
    }

    fn len(&mut self, maximum: usize) -> Result<usize, BeliefMutationSealWireError> {
        let value = usize::try_from(self.u64()?)
            .map_err(|_| BeliefMutationSealWireError::LengthOverflow)?;
        if value > maximum {
            return Err(BeliefMutationSealWireError::CountTooLarge {
                actual: value,
                maximum,
            });
        }
        Ok(value)
    }

    fn string(&mut self) -> Result<String, BeliefMutationSealWireError> {
        let len = self.len(MAX_STRING_BYTES)?;
        let bytes = self.take(len)?;
        let value = std::str::from_utf8(bytes)
            .map_err(|_| BeliefMutationSealWireError::InvalidUtf8)?;
        Ok(value.to_owned())
    }

    fn optional_string(&mut self) -> Result<Option<String>, BeliefMutationSealWireError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.string()?)),
            tag => Err(BeliefMutationSealWireError::InvalidOptionTag(tag)),
        }
    }

    fn bytes32(&mut self) -> Result<[u8; 32], BeliefMutationSealWireError> {
        self.take(32)?
            .try_into()
            .map_err(|_| BeliefMutationSealWireError::TruncatedPayload)
    }

    fn claim(&mut self) -> Result<SealedClaimSnapshot, BeliefMutationSealWireError> {
        Ok(SealedClaimSnapshot {
            claim_id: ClaimId(self.u64()?),
            statement: self.string()?,
            kind: claim_kind_from_tag(self.u8()?)?,
            domain: self.optional_string()?,
            scope: self.optional_string()?,
            created_at_cycle: self.u64()?,
        })
    }

    fn evidence(&mut self) -> Result<RevisionEvidenceSnapshot, BeliefMutationSealWireError> {
        Ok(RevisionEvidenceSnapshot {
            evidence_id: EvidenceId(self.u64()?),
            claim_id: ClaimId(self.u64()?),
            kind: evidence_kind_from_tag(self.u8()?)?,
            polarity: evidence_polarity_from_tag(self.u8()?)?,
            provenance_id: ProvenanceId(self.u64()?),
            observed_at_cycle: self.u64()?,
            context: self.optional_string()?,
            method: self.optional_string()?,
        })
    }
}

fn checksum(payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-belief-mutation-seal-wire-v1");
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

fn claim_kind_tag(kind: ClaimKind) -> u8 {
    match kind {
        ClaimKind::Descriptive => 1,
        ClaimKind::Predictive => 2,
        ClaimKind::Causal => 3,
        ClaimKind::Counterfactual => 4,
        ClaimKind::Procedural => 5,
        ClaimKind::Normative => 6,
    }
}

fn claim_kind_from_tag(tag: u8) -> Result<ClaimKind, BeliefMutationSealWireError> {
    match tag {
        1 => Ok(ClaimKind::Descriptive),
        2 => Ok(ClaimKind::Predictive),
        3 => Ok(ClaimKind::Causal),
        4 => Ok(ClaimKind::Counterfactual),
        5 => Ok(ClaimKind::Procedural),
        6 => Ok(ClaimKind::Normative),
        _ => Err(BeliefMutationSealWireError::UnknownClaimKind(tag)),
    }
}

fn evidence_kind_tag(kind: EvidenceKind) -> u8 {
    match kind {
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

fn evidence_kind_from_tag(tag: u8) -> Result<EvidenceKind, BeliefMutationSealWireError> {
    match tag {
        1 => Ok(EvidenceKind::Report),
        2 => Ok(EvidenceKind::Observation),
        3 => Ok(EvidenceKind::Measurement),
        4 => Ok(EvidenceKind::Intervention),
        5 => Ok(EvidenceKind::Replication),
        6 => Ok(EvidenceKind::Simulation),
        7 => Ok(EvidenceKind::Deduction),
        8 => Ok(EvidenceKind::ToolResult),
        _ => Err(BeliefMutationSealWireError::UnknownEvidenceKind(tag)),
    }
}

fn evidence_polarity_tag(polarity: EvidencePolarity) -> u8 {
    match polarity {
        EvidencePolarity::Supports => 1,
        EvidencePolarity::Contradicts => 2,
        EvidencePolarity::Contextualizes => 3,
    }
}

fn evidence_polarity_from_tag(tag: u8) -> Result<EvidencePolarity, BeliefMutationSealWireError> {
    match tag {
        1 => Ok(EvidencePolarity::Supports),
        2 => Ok(EvidencePolarity::Contradicts),
        3 => Ok(EvidencePolarity::Contextualizes),
        _ => Err(BeliefMutationSealWireError::UnknownEvidencePolarity(tag)),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeliefMutationSealWireError {
    InvalidMagic,
    UnsupportedVersion(u16),
    UnknownEncoding(u8),
    UnknownClaimKind(u8),
    UnknownEvidenceKind(u8),
    UnknownEvidencePolarity(u8),
    InvalidOptionTag(u8),
    InvalidUtf8,
    TruncatedEnvelope,
    TruncatedPayload,
    TrailingPayloadBytes,
    ChecksumMismatch,
    LengthOverflow,
    EnvelopeLengthMismatch { declared: usize, actual: usize },
    PayloadTooLarge { actual: usize, maximum: usize },
    StringTooLarge { actual: usize, maximum: usize },
    TooManyRecords { actual: usize, maximum: usize },
    TooManyEvidenceRecords { actual: usize, maximum: usize },
    CountTooLarge { actual: usize, maximum: usize },
}

impl fmt::Display for BeliefMutationSealWireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "belief mutation seal wire rejected: {self:?}")
    }
}

impl Error for BeliefMutationSealWireError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{BeliefMutationPersistenceCapsuleV1, EpistemicLedger, EpistemicSupportStore};

    #[test]
    fn empty_seal_capsule_round_trips_as_untrusted_snapshot() {
        let ledger = EpistemicLedger::new();
        let store = EpistemicSupportStore::new();
        let mutations = BeliefMutationPersistenceCapsuleV1::capture(&store, &[], 1).unwrap();
        let capsule = BeliefMutationEvidenceSealCapsuleV1::capture(&[], &mutations, &ledger, 1)
            .unwrap();
        let bytes = BeliefMutationSealWireV1::encode(&capsule).unwrap();
        let decoded = BeliefMutationSealWireV1::decode(&bytes).unwrap();
        assert_eq!(decoded.captured_at_cycle, 1);
        assert_eq!(decoded.linked_mutation_capture_cycle, 1);
        assert!(decoded.records.is_empty());
        assert_eq!(decoded.claimed_capsule_digest, capsule.capsule_digest().as_bytes());
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let ledger = EpistemicLedger::new();
        let store = EpistemicSupportStore::new();
        let mutations = BeliefMutationPersistenceCapsuleV1::capture(&store, &[], 1).unwrap();
        let capsule = BeliefMutationEvidenceSealCapsuleV1::capture(&[], &mutations, &ledger, 1)
            .unwrap();
        let mut bytes = BeliefMutationSealWireV1::encode(&capsule).unwrap();
        bytes.push(0);
        assert!(matches!(
            BeliefMutationSealWireV1::decode(&bytes),
            Err(BeliefMutationSealWireError::EnvelopeLengthMismatch { .. })
        ));
    }
}
