// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Thin restart-v2 bundle framing over existing base-state and typed-schema wires.
//!
//! EKM-034 remains the source of truth for base-state serialization. EKM-040
//! remains the source of truth for typed policy/decision serialization. This
//! module length-prefixes those two complete envelopes, binds the claimed EKM-039
//! V2 digest, and adds an outer corruption checksum.
//!
//! Decoding still produces untrusted snapshots only. Cross-component semantic
//! validation and wire-to-quarantine conversion are deliberately separate.

use super::belief_revision_schema_wire::{
    BeliefRevisionSchemaWireError, BeliefRevisionSchemaWireSnapshotV1,
    BeliefRevisionSchemaWireV1,
};
use super::epistemic_restart_capsule_v2::EpistemicRestartCapsuleV2;
use super::epistemic_restart_wire::{
    EpistemicRestartWireError, EpistemicRestartWireSnapshotV1, EpistemicRestartWireV1,
};
use std::error::Error;
use std::fmt;

const MAGIC: &[u8] = b"SYMTHAEA-EKM-RESTART-V2-BUNDLE";
const VERSION: u16 = 2;
const MAX_BASE_WIRE_BYTES: usize = 300 * 1024 * 1024;
const MAX_SCHEMA_WIRE_BYTES: usize = 80 * 1024 * 1024;
const MAX_BUNDLE_PAYLOAD_BYTES: usize = 384 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRestartWireV2Version {
    V2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRestartWireV2Encoding {
    BaseV1PlusTypedSchemaV1,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicRestartWireSnapshotV2 {
    pub version: EpistemicRestartWireV2Version,
    pub encoding: EpistemicRestartWireV2Encoding,
    pub base: EpistemicRestartWireSnapshotV1,
    pub revision_schemas: BeliefRevisionSchemaWireSnapshotV1,
    /// Claimed EKM-039 typed restart-v2 digest. It is not trusted merely because
    /// it was checksum-protected by this outer envelope.
    pub claimed_v2_digest: [u8; 32],
    pub outer_checksum: [u8; 32],
}

pub struct EpistemicRestartWireV2;

impl EpistemicRestartWireV2 {
    pub fn encode(
        capsule: &EpistemicRestartCapsuleV2,
    ) -> Result<Vec<u8>, EpistemicRestartWireV2Error> {
        capsule
            .verify()
            .map_err(EpistemicRestartWireV2Error::Capsule)?;
        let base_wire = EpistemicRestartWireV1::encode(capsule.base_v1())
            .map_err(EpistemicRestartWireV2Error::BaseWire)?;
        let schema_wire = BeliefRevisionSchemaWireV1::encode(capsule.revision_schema_capsule())
            .map_err(EpistemicRestartWireV2Error::SchemaWire)?;

        if base_wire.len() > MAX_BASE_WIRE_BYTES {
            return Err(EpistemicRestartWireV2Error::BaseWireTooLarge(
                base_wire.len(),
            ));
        }
        if schema_wire.len() > MAX_SCHEMA_WIRE_BYTES {
            return Err(EpistemicRestartWireV2Error::SchemaWireTooLarge(
                schema_wire.len(),
            ));
        }

        let mut payload = Writer::new();
        payload.u8(1); // encoding tag
        payload.bytes(&base_wire)?;
        payload.bytes(&schema_wire)?;
        payload.bytes32(capsule.capsule_digest().as_bytes());
        let payload = payload.finish();
        if payload.len() > MAX_BUNDLE_PAYLOAD_BYTES {
            return Err(EpistemicRestartWireV2Error::PayloadTooLarge(payload.len()));
        }

        let payload_len = u64::try_from(payload.len())
            .map_err(|_| EpistemicRestartWireV2Error::LengthOverflow)?;
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
    ) -> Result<EpistemicRestartWireSnapshotV2, EpistemicRestartWireV2Error> {
        let minimum = MAGIC.len() + 2 + 8 + 32;
        if bytes.len() < minimum {
            return Err(EpistemicRestartWireV2Error::Truncated);
        }
        if &bytes[..MAGIC.len()] != MAGIC {
            return Err(EpistemicRestartWireV2Error::BadMagic);
        }

        let version_offset = MAGIC.len();
        let version = u16::from_le_bytes([
            bytes[version_offset],
            bytes[version_offset + 1],
        ]);
        if version != VERSION {
            return Err(EpistemicRestartWireV2Error::UnsupportedVersion(version));
        }

        let len_offset = version_offset + 2;
        let payload_len_u64 = u64::from_le_bytes(
            bytes[len_offset..len_offset + 8]
                .try_into()
                .expect("slice width checked"),
        );
        let payload_len = usize::try_from(payload_len_u64)
            .map_err(|_| EpistemicRestartWireV2Error::LengthOverflow)?;
        if payload_len > MAX_BUNDLE_PAYLOAD_BYTES {
            return Err(EpistemicRestartWireV2Error::PayloadTooLarge(payload_len));
        }
        let expected_total = minimum
            .checked_add(payload_len)
            .ok_or(EpistemicRestartWireV2Error::LengthOverflow)?;
        if bytes.len() != expected_total {
            return Err(EpistemicRestartWireV2Error::EnvelopeLengthMismatch {
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
            return Err(EpistemicRestartWireV2Error::ChecksumMismatch);
        }

        let mut reader = Reader::new(payload);
        let encoding = match reader.u8()? {
            1 => EpistemicRestartWireV2Encoding::BaseV1PlusTypedSchemaV1,
            tag => return Err(EpistemicRestartWireV2Error::UnknownEncodingTag(tag)),
        };
        let base_wire = reader.bytes(MAX_BASE_WIRE_BYTES)?;
        let schema_wire = reader.bytes(MAX_SCHEMA_WIRE_BYTES)?;
        let claimed_v2_digest = reader.bytes32()?;
        if !reader.is_finished() {
            return Err(EpistemicRestartWireV2Error::TrailingPayloadBytes(
                reader.remaining(),
            ));
        }

        let base = EpistemicRestartWireV1::decode(base_wire)
            .map_err(EpistemicRestartWireV2Error::BaseWire)?;
        let revision_schemas = BeliefRevisionSchemaWireV1::decode(schema_wire)
            .map_err(EpistemicRestartWireV2Error::SchemaWire)?;

        Ok(EpistemicRestartWireSnapshotV2 {
            version: EpistemicRestartWireV2Version::V2,
            encoding,
            base,
            revision_schemas,
            claimed_v2_digest,
            outer_checksum: actual_checksum,
        })
    }
}

fn checksum(version: u16, payload_len: u64, payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-v2-wire-bundle");
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

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn bytes(&mut self, value: &[u8]) -> Result<(), EpistemicRestartWireV2Error> {
        let len = u64::try_from(value.len())
            .map_err(|_| EpistemicRestartWireV2Error::LengthOverflow)?;
        self.u64(len);
        self.bytes.extend_from_slice(value);
        Ok(())
    }

    fn bytes32(&mut self, value: [u8; 32]) {
        self.bytes.extend_from_slice(&value);
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
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

    fn take(&mut self, len: usize) -> Result<&'a [u8], EpistemicRestartWireV2Error> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(EpistemicRestartWireV2Error::LengthOverflow)?;
        if end > self.bytes.len() {
            return Err(EpistemicRestartWireV2Error::Truncated);
        }
        let out = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, EpistemicRestartWireV2Error> {
        Ok(self.take(1)?[0])
    }

    fn u64(&mut self) -> Result<u64, EpistemicRestartWireV2Error> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .expect("reader returned exact u64 width"),
        ))
    }

    fn bytes(&mut self, maximum: usize) -> Result<&'a [u8], EpistemicRestartWireV2Error> {
        let len = usize::try_from(self.u64()?)
            .map_err(|_| EpistemicRestartWireV2Error::LengthOverflow)?;
        if len > maximum {
            return Err(EpistemicRestartWireV2Error::SubEnvelopeTooLarge {
                maximum,
                actual: len,
            });
        }
        self.take(len)
    }

    fn bytes32(&mut self) -> Result<[u8; 32], EpistemicRestartWireV2Error> {
        Ok(self
            .take(32)?
            .try_into()
            .expect("reader returned exact digest width"))
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicRestartWireV2Error {
    Capsule(super::epistemic_restart_capsule_v2::EpistemicRestartCapsuleV2Error),
    BaseWire(EpistemicRestartWireError),
    SchemaWire(BeliefRevisionSchemaWireError),
    Truncated,
    BadMagic,
    UnsupportedVersion(u16),
    UnknownEncodingTag(u8),
    EnvelopeLengthMismatch {
        declared_payload: usize,
        actual_total: usize,
    },
    ChecksumMismatch,
    BaseWireTooLarge(usize),
    SchemaWireTooLarge(usize),
    PayloadTooLarge(usize),
    SubEnvelopeTooLarge {
        maximum: usize,
        actual: usize,
    },
    LengthOverflow,
    TrailingPayloadBytes(usize),
}

impl fmt::Display for EpistemicRestartWireV2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epistemic restart v2 wire bundle invalid: {self:?}")
    }
}

impl Error for EpistemicRestartWireV2Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistory,
        BeliefRevisionHistoryCapsuleV1, BeliefRevisionPolicySchemaV1,
        BeliefRevisionSchemaHistoryCapsuleV1, BeliefRevisionSchemaHistoryV1,
        ClaimKind, EpistemicLedger, EpistemicLedgerInventoryV1,
        EpistemicRestartCapsuleV1, EpistemicRestartCapsuleV2, EpistemicRevisionProposal,
        EpistemicSupportStore, EvidenceKind, EvidencePolarity,
    };

    fn capsule() -> EpistemicRestartCapsuleV2 {
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
        let inventory = EpistemicLedgerInventoryV1::new(
            vec![claim],
            vec![evidence],
            vec![provenance],
        )
        .unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.1, vec![evidence], "measurement")
            .unwrap();
        let schema = BeliefRevisionPolicySchemaV1::new(0.2, 1, false, 0, 1.0).unwrap();
        let mut receipts = BeliefRevisionHistory::new();
        let mut schema_history = BeliefRevisionSchemaHistoryV1::new();
        schema_history
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
        let schema_capsule = BeliefRevisionSchemaHistoryCapsuleV1::capture(
            &schema_history,
            &receipts,
            &revisions,
            4,
        )
        .unwrap();
        let base = EpistemicRestartCapsuleV1::capture(
            &ledger,
            &inventory,
            &mutations,
            &revisions,
            4,
        )
        .unwrap();
        EpistemicRestartCapsuleV2::capture(&base, &schema_capsule).unwrap()
    }

    #[test]
    fn bundle_round_trip_preserves_both_subsnapshots() {
        let capsule = capsule();
        let bytes = EpistemicRestartWireV2::encode(&capsule).unwrap();
        let decoded = EpistemicRestartWireV2::decode(&bytes).unwrap();
        assert_eq!(
            decoded.claimed_v2_digest,
            capsule.capsule_digest().as_bytes()
        );
        assert_eq!(decoded.base.captured_at_cycle, capsule.captured_at_cycle());
        assert_eq!(
            decoded.revision_schemas.captured_at_cycle,
            capsule.captured_at_cycle()
        );
        assert_eq!(decoded.revision_schemas.records.len(), 1);
    }

    #[test]
    fn one_byte_outer_tamper_fails_before_subwire_parse() {
        let capsule = capsule();
        let mut bytes = EpistemicRestartWireV2::encode(&capsule).unwrap();
        let payload_start = MAGIC.len() + 2 + 8;
        bytes[payload_start] ^= 0x01;
        assert_eq!(
            EpistemicRestartWireV2::decode(&bytes).unwrap_err(),
            EpistemicRestartWireV2Error::ChecksumMismatch
        );
    }

    #[test]
    fn unsupported_outer_version_is_rejected() {
        let capsule = capsule();
        let mut bytes = EpistemicRestartWireV2::encode(&capsule).unwrap();
        let offset = MAGIC.len();
        bytes[offset..offset + 2].copy_from_slice(&3u16.to_le_bytes());
        assert_eq!(
            EpistemicRestartWireV2::decode(&bytes).unwrap_err(),
            EpistemicRestartWireV2Error::UnsupportedVersion(3)
        );
    }
}
