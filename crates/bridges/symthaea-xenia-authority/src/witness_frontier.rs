// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent verifier for Xenia witness-frontier anchor/currentness evidence.
//!
//! This module intentionally has no dependency on Xenia source crates. It mirrors
//! the commitment-only V1 protocol introduced by xenia-peer #293 and verifies:
//!
//! - exact Symthaea witness-frontier statement and operation commitments;
//! - Xenia source identity derived from the trusted ledger key + anchor policy;
//! - the durable anchor's Ed25519 signature;
//! - a verifier-challenge-bound fresh current-frontier observation;
//! - exact observation-to-anchor fingerprint/frontier equality;
//! - non-regressing signed Xenia ledger context.
//!
//! The result is evidence chronology only. It never creates or restores execution
//! authority, capability uses, budgets, retries, or action admission.

use ed25519_dalek::{Signature, Verifier as _, VerifyingKey};
use serde::{Deserialize, Serialize};
use symthaea_authority::Digest32;
use thiserror::Error;

use crate::protocol::{ED25519_SIGNATURE_ALGORITHM, XeniaSignatureEnvelopeV1};

pub const XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION: u16 = 1;
pub const SYMTHAEA_WITNESS_FRONTIER_STATEMENT_SCHEMA_VERSION: u16 = 1;
pub const SYMTHAEA_WITNESS_ANCHOR_OPERATION_DOMAIN: &[u8] =
    b"symthaea.qualification-witness.anchor-operation.v1\0";
pub const SYMTHAEA_WITNESS_FRONTIER_STATEMENT_DOMAIN: &[u8] =
    b"symthaea.qualification-witness.sequence-frontier.v1\0";
pub const XENIA_WITNESS_FRONTIER_ANCHOR_DOMAIN: &[u8] =
    b"xenia.witness-frontier-anchor.v1\0";
pub const XENIA_WITNESS_FRONTIER_ANCHOR_FINGERPRINT_DOMAIN: &[u8] =
    b"xenia.witness-frontier-anchor-fingerprint.v1\0";
pub const XENIA_WITNESS_FRONTIER_OBSERVATION_DOMAIN: &[u8] =
    b"xenia.witness-frontier-observation.v1\0";
pub const XENIA_WITNESS_FRONTIER_OBSERVATION_FINGERPRINT_DOMAIN: &[u8] =
    b"xenia.witness-frontier-observation-fingerprint.v1\0";
pub const XENIA_WITNESS_FRONTIER_SOURCE_DOMAIN: &[u8] =
    b"xenia.witness-frontier-source-id.v1\0";

const ZERO16: [u8; 16] = [0; 16];
const ZERO32: [u8; 32] = [0; 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaWitnessFrontierAnchorTargetV1 {
    pub schema_version: u16,
    pub operation_id: [u8; 32],
    pub source_id: [u8; 16],
    pub source_epoch: u64,
    pub anchor_policy_digest: [u8; 32],
    pub witness_id: [u8; 16],
    pub high_watermark: u64,
    pub reservation_head: [u8; 32],
    pub frontier_statement_digest: [u8; 32],
}

impl XeniaWitnessFrontierAnchorTargetV1 {
    pub fn validate(&self) -> Result<(), XeniaWitnessFrontierError> {
        if self.schema_version != XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION
            || self.operation_id == ZERO32
            || self.source_id == ZERO16
            || self.source_epoch == 0
            || self.anchor_policy_digest == ZERO32
            || self.witness_id == ZERO16
            || self.high_watermark == 0
            || self.reservation_head == ZERO32
            || self.frontier_statement_digest == ZERO32
        {
            return Err(XeniaWitnessFrontierError::MalformedTarget);
        }
        if self.frontier_statement_digest != self.recompute_frontier_statement_digest() {
            return Err(XeniaWitnessFrontierError::FrontierStatementDigestMismatch);
        }
        if self.operation_id != self.recompute_operation_id() {
            return Err(XeniaWitnessFrontierError::OperationIdMismatch);
        }
        Ok(())
    }

    pub fn recompute_frontier_statement_digest(&self) -> [u8; 32] {
        witness_frontier_statement_digest(
            self.witness_id,
            self.high_watermark,
            self.reservation_head,
        )
    }

    pub fn canonical_operation_message(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(224);
        bytes.extend_from_slice(SYMTHAEA_WITNESS_ANCHOR_OPERATION_DOMAIN);
        bytes.extend_from_slice(&XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION.to_be_bytes());
        bytes.extend_from_slice(&self.source_id);
        bytes.extend_from_slice(&self.source_epoch.to_be_bytes());
        bytes.extend_from_slice(&self.anchor_policy_digest);
        bytes.extend_from_slice(&self.witness_id);
        bytes.extend_from_slice(&self.high_watermark.to_be_bytes());
        bytes.extend_from_slice(&self.reservation_head);
        bytes.extend_from_slice(&self.frontier_statement_digest);
        bytes
    }

    pub fn recompute_operation_id(&self) -> [u8; 32] {
        *blake3::hash(&self.canonical_operation_message()).as_bytes()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaSignedWitnessFrontierAnchorV1 {
    pub schema_version: u16,
    pub target: XeniaWitnessFrontierAnchorTargetV1,
    pub anchor_sequence: u64,
    pub previous_anchor_fingerprint: [u8; 32],
    pub ledger_entry_count: u64,
    pub ledger_head_hash: [u8; 32],
    pub ledger_public_key: [u8; 32],
    pub issued_at_unix_s: u64,
    pub signature: XeniaSignatureEnvelopeV1,
}

impl XeniaSignedWitnessFrontierAnchorV1 {
    pub fn canonical_message(&self) -> Result<Vec<u8>, XeniaWitnessFrontierError> {
        self.validate_unsigned_shape()?;
        let mut bytes = Vec::with_capacity(384);
        bytes.extend_from_slice(XENIA_WITNESS_FRONTIER_ANCHOR_DOMAIN);
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        bytes.extend_from_slice(&self.target.canonical_operation_message());
        bytes.extend_from_slice(&self.target.operation_id);
        bytes.extend_from_slice(&self.anchor_sequence.to_be_bytes());
        bytes.extend_from_slice(&self.previous_anchor_fingerprint);
        bytes.extend_from_slice(&self.ledger_entry_count.to_be_bytes());
        bytes.extend_from_slice(&self.ledger_head_hash);
        bytes.extend_from_slice(&self.ledger_public_key);
        bytes.extend_from_slice(&self.issued_at_unix_s.to_be_bytes());
        Ok(bytes)
    }

    pub fn verify_with_trusted_key(
        &self,
        trusted_ledger_public_key: [u8; 32],
    ) -> Result<(), XeniaWitnessFrontierError> {
        if self.ledger_public_key != trusted_ledger_public_key {
            return Err(XeniaWitnessFrontierError::LedgerKeyMismatch);
        }
        verify_ed25519_envelope(
            trusted_ledger_public_key,
            &self.canonical_message()?,
            &self.signature,
        )
    }

    pub fn fingerprint(&self) -> Result<[u8; 32], XeniaWitnessFrontierError> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(XENIA_WITNESS_FRONTIER_ANCHOR_FINGERPRINT_DOMAIN);
        hasher.update(&self.canonical_message()?);
        hasher.update(self.signature.algorithm.as_bytes());
        hasher.update(&(self.signature.signature.len() as u64).to_be_bytes());
        hasher.update(&self.signature.signature);
        Ok(*hasher.finalize().as_bytes())
    }

    fn validate_unsigned_shape(&self) -> Result<(), XeniaWitnessFrontierError> {
        if self.schema_version != XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION
            || self.anchor_sequence == 0
            || self.ledger_entry_count == 0
            || self.ledger_head_hash == ZERO32
            || self.ledger_public_key == ZERO32
            || self.issued_at_unix_s == 0
        {
            return Err(XeniaWitnessFrontierError::MalformedAnchor);
        }
        self.target.validate()?;
        let expected_source = derive_xenia_witness_frontier_source_id(
            self.ledger_public_key,
            self.target.anchor_policy_digest,
        )?;
        if self.target.source_id != expected_source {
            return Err(XeniaWitnessFrontierError::SourceBindingMismatch);
        }
        if self.anchor_sequence == 1 {
            if self.previous_anchor_fingerprint != ZERO32 {
                return Err(XeniaWitnessFrontierError::PreviousAnchorMismatch);
            }
        } else if self.previous_anchor_fingerprint == ZERO32 {
            return Err(XeniaWitnessFrontierError::PreviousAnchorMismatch);
        }
        validate_signature_envelope(&self.signature)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaWitnessFrontierAnchorSummaryV1 {
    pub anchor_sequence: u64,
    pub anchor_fingerprint: [u8; 32],
    pub operation_id: [u8; 32],
    pub high_watermark: u64,
    pub reservation_head: [u8; 32],
    pub frontier_statement_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaSignedWitnessFrontierObservationV1 {
    pub schema_version: u16,
    pub source_id: [u8; 16],
    pub source_epoch: u64,
    pub anchor_policy_digest: [u8; 32],
    pub witness_id: [u8; 16],
    pub challenge: [u8; 32],
    pub observed_at_unix_s: u64,
    pub current: Option<XeniaWitnessFrontierAnchorSummaryV1>,
    pub ledger_entry_count: u64,
    pub ledger_head_hash: [u8; 32],
    pub ledger_public_key: [u8; 32],
    pub signature: XeniaSignatureEnvelopeV1,
}

impl XeniaSignedWitnessFrontierObservationV1 {
    pub fn canonical_message(&self) -> Result<Vec<u8>, XeniaWitnessFrontierError> {
        self.validate_unsigned_shape()?;
        let mut bytes = Vec::with_capacity(384);
        bytes.extend_from_slice(XENIA_WITNESS_FRONTIER_OBSERVATION_DOMAIN);
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        bytes.extend_from_slice(&self.source_id);
        bytes.extend_from_slice(&self.source_epoch.to_be_bytes());
        bytes.extend_from_slice(&self.anchor_policy_digest);
        bytes.extend_from_slice(&self.witness_id);
        bytes.extend_from_slice(&self.challenge);
        bytes.extend_from_slice(&self.observed_at_unix_s.to_be_bytes());
        match self.current {
            None => bytes.push(0),
            Some(current) => {
                bytes.push(1);
                bytes.extend_from_slice(&current.anchor_sequence.to_be_bytes());
                bytes.extend_from_slice(&current.anchor_fingerprint);
                bytes.extend_from_slice(&current.operation_id);
                bytes.extend_from_slice(&current.high_watermark.to_be_bytes());
                bytes.extend_from_slice(&current.reservation_head);
                bytes.extend_from_slice(&current.frontier_statement_digest);
            }
        }
        bytes.extend_from_slice(&self.ledger_entry_count.to_be_bytes());
        bytes.extend_from_slice(&self.ledger_head_hash);
        bytes.extend_from_slice(&self.ledger_public_key);
        Ok(bytes)
    }

    pub fn verify_with_trusted_key(
        &self,
        trusted_ledger_public_key: [u8; 32],
    ) -> Result<(), XeniaWitnessFrontierError> {
        if self.ledger_public_key != trusted_ledger_public_key {
            return Err(XeniaWitnessFrontierError::LedgerKeyMismatch);
        }
        verify_ed25519_envelope(
            trusted_ledger_public_key,
            &self.canonical_message()?,
            &self.signature,
        )
    }

    pub fn fingerprint(&self) -> Result<[u8; 32], XeniaWitnessFrontierError> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(XENIA_WITNESS_FRONTIER_OBSERVATION_FINGERPRINT_DOMAIN);
        hasher.update(&self.canonical_message()?);
        hasher.update(self.signature.algorithm.as_bytes());
        hasher.update(&(self.signature.signature.len() as u64).to_be_bytes());
        hasher.update(&self.signature.signature);
        Ok(*hasher.finalize().as_bytes())
    }

    fn validate_unsigned_shape(&self) -> Result<(), XeniaWitnessFrontierError> {
        if self.schema_version != XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION
            || self.source_id == ZERO16
            || self.source_epoch == 0
            || self.anchor_policy_digest == ZERO32
            || self.witness_id == ZERO16
            || self.challenge == ZERO32
            || self.observed_at_unix_s == 0
            || self.ledger_entry_count == 0
            || self.ledger_head_hash == ZERO32
            || self.ledger_public_key == ZERO32
        {
            return Err(XeniaWitnessFrontierError::MalformedObservation);
        }
        let expected_source = derive_xenia_witness_frontier_source_id(
            self.ledger_public_key,
            self.anchor_policy_digest,
        )?;
        if self.source_id != expected_source {
            return Err(XeniaWitnessFrontierError::SourceBindingMismatch);
        }
        if let Some(current) = self.current {
            if current.anchor_sequence == 0
                || current.anchor_fingerprint == ZERO32
                || current.operation_id == ZERO32
                || current.high_watermark == 0
                || current.reservation_head == ZERO32
                || current.frontier_statement_digest == ZERO32
                || current.frontier_statement_digest
                    != witness_frontier_statement_digest(
                        self.witness_id,
                        current.high_watermark,
                        current.reservation_head,
                    )
            {
                return Err(XeniaWitnessFrontierError::MalformedObservation);
            }
        }
        validate_signature_envelope(&self.signature)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XeniaWitnessFrontierExpectationV1 {
    pub trusted_ledger_public_key: [u8; 32],
    pub source_epoch: u64,
    pub anchor_policy_digest: [u8; 32],
    pub witness_id: [u8; 16],
    pub challenge: [u8; 32],
}

impl XeniaWitnessFrontierExpectationV1 {
    pub fn source_id(self) -> Result<[u8; 16], XeniaWitnessFrontierError> {
        derive_xenia_witness_frontier_source_id(
            self.trusted_ledger_public_key,
            self.anchor_policy_digest,
        )
    }

    fn validate(self) -> Result<(), XeniaWitnessFrontierError> {
        if self.trusted_ledger_public_key == ZERO32
            || self.source_epoch == 0
            || self.anchor_policy_digest == ZERO32
            || self.witness_id == ZERO16
            || self.challenge == ZERO32
        {
            return Err(XeniaWitnessFrontierError::MalformedExpectation);
        }
        self.source_id()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XeniaWitnessObservationFreshnessV1 {
    /// Earliest plausible current wall-clock instant from a trusted time source.
    pub earliest_now_unix_s: u64,
    /// Latest plausible current wall-clock instant from a trusted time source.
    pub latest_now_unix_s: u64,
    pub max_age_s: u64,
    pub max_future_skew_s: u64,
}

impl XeniaWitnessObservationFreshnessV1 {
    fn validate(self) -> Result<(), XeniaWitnessFrontierError> {
        if self.earliest_now_unix_s == 0
            || self.latest_now_unix_s == 0
            || self.earliest_now_unix_s > self.latest_now_unix_s
        {
            return Err(XeniaWitnessFrontierError::MalformedFreshnessWindow);
        }
        Ok(())
    }
}

/// Transport-neutral fields produced only after the exact Xenia anchor and fresh
/// observation have been independently verified.
#[derive(Debug)]
pub struct VerifiedXeniaWitnessFrontierV1 {
    source_id: [u8; 16],
    source_epoch: u64,
    source_sequence: u64,
    witness_id: [u8; 16],
    high_watermark: u64,
    reservation_head: Digest32,
    frontier_statement_digest: Digest32,
    freshness_evidence_digest: Digest32,
    anchor_fingerprint: Digest32,
    operation_id: Digest32,
    observed_at_unix_s: u64,
}

impl VerifiedXeniaWitnessFrontierV1 {
    pub fn source_id(&self) -> [u8; 16] { self.source_id }
    pub fn source_epoch(&self) -> u64 { self.source_epoch }
    pub fn source_sequence(&self) -> u64 { self.source_sequence }
    pub fn witness_id(&self) -> [u8; 16] { self.witness_id }
    pub fn high_watermark(&self) -> u64 { self.high_watermark }
    pub fn reservation_head(&self) -> Digest32 { self.reservation_head }
    pub fn frontier_statement_digest(&self) -> Digest32 { self.frontier_statement_digest }
    pub fn freshness_evidence_digest(&self) -> Digest32 { self.freshness_evidence_digest }
    pub fn anchor_fingerprint(&self) -> Digest32 { self.anchor_fingerprint }
    pub fn operation_id(&self) -> Digest32 { self.operation_id }
    pub fn observed_at_unix_s(&self) -> u64 { self.observed_at_unix_s }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_xenia_witness_frontier_v1(
    anchor: &XeniaSignedWitnessFrontierAnchorV1,
    observation: &XeniaSignedWitnessFrontierObservationV1,
    expected: XeniaWitnessFrontierExpectationV1,
    freshness: XeniaWitnessObservationFreshnessV1,
) -> Result<VerifiedXeniaWitnessFrontierV1, XeniaWitnessFrontierError> {
    expected.validate()?;
    freshness.validate()?;
    let expected_source_id = expected.source_id()?;

    anchor.verify_with_trusted_key(expected.trusted_ledger_public_key)?;
    observation.verify_with_trusted_key(expected.trusted_ledger_public_key)?;

    if anchor.target.source_id != expected_source_id
        || anchor.target.source_epoch != expected.source_epoch
        || anchor.target.anchor_policy_digest != expected.anchor_policy_digest
        || anchor.target.witness_id != expected.witness_id
    {
        return Err(XeniaWitnessFrontierError::AnchorBindingMismatch);
    }

    if observation.source_id != expected_source_id
        || observation.source_epoch != expected.source_epoch
        || observation.anchor_policy_digest != expected.anchor_policy_digest
        || observation.witness_id != expected.witness_id
        || observation.challenge != expected.challenge
    {
        return Err(XeniaWitnessFrontierError::ObservationBindingMismatch);
    }

    let oldest_acceptable = freshness.latest_now_unix_s.saturating_sub(freshness.max_age_s);
    let latest_acceptable = freshness
        .earliest_now_unix_s
        .saturating_add(freshness.max_future_skew_s);
    if observation.observed_at_unix_s < oldest_acceptable
        || observation.observed_at_unix_s > latest_acceptable
    {
        return Err(XeniaWitnessFrontierError::ObservationStaleOrFuture);
    }
    if observation.observed_at_unix_s < anchor.issued_at_unix_s {
        return Err(XeniaWitnessFrontierError::ObservationPredatesAnchor);
    }

    let current = observation
        .current
        .ok_or(XeniaWitnessFrontierError::ObservationMissingCurrentAnchor)?;
    let anchor_fingerprint = anchor.fingerprint()?;
    let expected_summary = XeniaWitnessFrontierAnchorSummaryV1 {
        anchor_sequence: anchor.anchor_sequence,
        anchor_fingerprint,
        operation_id: anchor.target.operation_id,
        high_watermark: anchor.target.high_watermark,
        reservation_head: anchor.target.reservation_head,
        frontier_statement_digest: anchor.target.frontier_statement_digest,
    };
    if current != expected_summary {
        return Err(XeniaWitnessFrontierError::ObservationCurrentAnchorMismatch);
    }

    if observation.ledger_entry_count < anchor.ledger_entry_count
        || (observation.ledger_entry_count == anchor.ledger_entry_count
            && observation.ledger_head_hash != anchor.ledger_head_hash)
    {
        return Err(XeniaWitnessFrontierError::LedgerContextRegression);
    }

    let freshness_evidence_digest = Digest32(observation.fingerprint()?);
    Ok(VerifiedXeniaWitnessFrontierV1 {
        source_id: expected_source_id,
        source_epoch: expected.source_epoch,
        source_sequence: anchor.anchor_sequence,
        witness_id: anchor.target.witness_id,
        high_watermark: anchor.target.high_watermark,
        reservation_head: Digest32(anchor.target.reservation_head),
        frontier_statement_digest: Digest32(anchor.target.frontier_statement_digest),
        freshness_evidence_digest,
        anchor_fingerprint: Digest32(anchor_fingerprint),
        operation_id: Digest32(anchor.target.operation_id),
        observed_at_unix_s: observation.observed_at_unix_s,
    })
}

pub fn derive_xenia_witness_frontier_source_id(
    ledger_public_key: [u8; 32],
    anchor_policy_digest: [u8; 32],
) -> Result<[u8; 16], XeniaWitnessFrontierError> {
    if ledger_public_key == ZERO32 || anchor_policy_digest == ZERO32 {
        return Err(XeniaWitnessFrontierError::MalformedExpectation);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(XENIA_WITNESS_FRONTIER_SOURCE_DOMAIN);
    hasher.update(&ledger_public_key);
    hasher.update(&anchor_policy_digest);
    let digest = hasher.finalize();
    let mut source_id = [0u8; 16];
    source_id.copy_from_slice(&digest.as_bytes()[..16]);
    if source_id == ZERO16 {
        return Err(XeniaWitnessFrontierError::MalformedExpectation);
    }
    Ok(source_id)
}

pub fn witness_frontier_statement_digest(
    witness_id: [u8; 16],
    high_watermark: u64,
    reservation_head: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(128);
    bytes.extend_from_slice(SYMTHAEA_WITNESS_FRONTIER_STATEMENT_DOMAIN);
    bytes.extend_from_slice(&SYMTHAEA_WITNESS_FRONTIER_STATEMENT_SCHEMA_VERSION.to_be_bytes());
    bytes.extend_from_slice(&witness_id);
    bytes.extend_from_slice(&high_watermark.to_be_bytes());
    bytes.extend_from_slice(&reservation_head);
    *blake3::hash(&bytes).as_bytes()
}

fn validate_signature_envelope(
    envelope: &XeniaSignatureEnvelopeV1,
) -> Result<(), XeniaWitnessFrontierError> {
    if envelope.algorithm != ED25519_SIGNATURE_ALGORITHM || envelope.signature.len() != 64 {
        return Err(XeniaWitnessFrontierError::UnsupportedSignatureEnvelope);
    }
    Ok(())
}

fn verify_ed25519_envelope(
    public_key: [u8; 32],
    message: &[u8],
    envelope: &XeniaSignatureEnvelopeV1,
) -> Result<(), XeniaWitnessFrontierError> {
    validate_signature_envelope(envelope)?;
    let key = VerifyingKey::from_bytes(&public_key)
        .map_err(|_| XeniaWitnessFrontierError::BadLedgerPublicKey)?;
    let signature_bytes: [u8; 64] = envelope
        .signature
        .as_slice()
        .try_into()
        .map_err(|_| XeniaWitnessFrontierError::BadSignature)?;
    key.verify(message, &Signature::from_bytes(&signature_bytes))
        .map_err(|_| XeniaWitnessFrontierError::BadSignature)
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum XeniaWitnessFrontierError {
    #[error("malformed Xenia witness-frontier target")]
    MalformedTarget,
    #[error("Xenia witness frontier statement digest mismatch")]
    FrontierStatementDigestMismatch,
    #[error("Xenia witness anchor operation id mismatch")]
    OperationIdMismatch,
    #[error("malformed Xenia signed witness anchor")]
    MalformedAnchor,
    #[error("Xenia source identity does not match trusted key and policy")]
    SourceBindingMismatch,
    #[error("Xenia previous-anchor fingerprint is inconsistent with sequence")]
    PreviousAnchorMismatch,
    #[error("unsupported Xenia witness signature envelope")]
    UnsupportedSignatureEnvelope,
    #[error("Xenia witness ledger public key is malformed")]
    BadLedgerPublicKey,
    #[error("Xenia witness signature verification failed")]
    BadSignature,
    #[error("Xenia witness evidence uses another ledger key")]
    LedgerKeyMismatch,
    #[error("malformed Xenia witness-frontier observation")]
    MalformedObservation,
    #[error("malformed Xenia witness-frontier expectation")]
    MalformedExpectation,
    #[error("malformed trusted freshness window")]
    MalformedFreshnessWindow,
    #[error("Xenia durable anchor does not match expected source/witness")]
    AnchorBindingMismatch,
    #[error("Xenia fresh observation does not match expected challenge/source/witness")]
    ObservationBindingMismatch,
    #[error("Xenia witness observation is stale or implausibly future-dated")]
    ObservationStaleOrFuture,
    #[error("Xenia witness observation predates the anchor it claims is current")]
    ObservationPredatesAnchor,
    #[error("Xenia witness observation contains no current anchor")]
    ObservationMissingCurrentAnchor,
    #[error("Xenia observation current summary does not equal the exact signed anchor")]
    ObservationCurrentAnchorMismatch,
    #[error("Xenia signed ledger context regressed relative to the anchor")]
    LedgerContextRegression,
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::{Signer, SigningKey};

    use super::*;

    const POLICY: [u8; 32] = [0x33; 32];
    const WITNESS: [u8; 16] = [0x44; 16];
    const RESERVATION_HEAD: [u8; 32] = [0x55; 32];
    const CHALLENGE: [u8; 32] = [0x66; 32];

    fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&[7; 32])
    }

    fn envelope(signature: [u8; 64]) -> XeniaSignatureEnvelopeV1 {
        XeniaSignatureEnvelopeV1 {
            algorithm: ED25519_SIGNATURE_ALGORITHM.to_string(),
            signature: signature.to_vec(),
        }
    }

    fn target() -> XeniaWitnessFrontierAnchorTargetV1 {
        let key = signing_key().verifying_key().to_bytes();
        let source_id = derive_xenia_witness_frontier_source_id(key, POLICY).unwrap();
        let frontier_statement_digest =
            witness_frontier_statement_digest(WITNESS, 9, RESERVATION_HEAD);
        let mut target = XeniaWitnessFrontierAnchorTargetV1 {
            schema_version: XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION,
            operation_id: ZERO32,
            source_id,
            source_epoch: 3,
            anchor_policy_digest: POLICY,
            witness_id: WITNESS,
            high_watermark: 9,
            reservation_head: RESERVATION_HEAD,
            frontier_statement_digest,
        };
        target.operation_id = target.recompute_operation_id();
        target
    }

    fn signed_anchor() -> XeniaSignedWitnessFrontierAnchorV1 {
        let key = signing_key();
        let mut anchor = XeniaSignedWitnessFrontierAnchorV1 {
            schema_version: XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION,
            target: target(),
            anchor_sequence: 1,
            previous_anchor_fingerprint: ZERO32,
            ledger_entry_count: 12,
            ledger_head_hash: [0x77; 32],
            ledger_public_key: key.verifying_key().to_bytes(),
            issued_at_unix_s: 1_000,
            signature: envelope([0; 64]),
        };
        let signature = key.sign(&anchor.canonical_message().unwrap()).to_bytes();
        anchor.signature = envelope(signature);
        anchor
    }

    fn signed_observation(anchor: &XeniaSignedWitnessFrontierAnchorV1) -> XeniaSignedWitnessFrontierObservationV1 {
        let key = signing_key();
        let mut observation = XeniaSignedWitnessFrontierObservationV1 {
            schema_version: XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION,
            source_id: anchor.target.source_id,
            source_epoch: anchor.target.source_epoch,
            anchor_policy_digest: anchor.target.anchor_policy_digest,
            witness_id: anchor.target.witness_id,
            challenge: CHALLENGE,
            observed_at_unix_s: 1_010,
            current: Some(XeniaWitnessFrontierAnchorSummaryV1 {
                anchor_sequence: anchor.anchor_sequence,
                anchor_fingerprint: anchor.fingerprint().unwrap(),
                operation_id: anchor.target.operation_id,
                high_watermark: anchor.target.high_watermark,
                reservation_head: anchor.target.reservation_head,
                frontier_statement_digest: anchor.target.frontier_statement_digest,
            }),
            ledger_entry_count: 13,
            ledger_head_hash: [0x88; 32],
            ledger_public_key: key.verifying_key().to_bytes(),
            signature: envelope([0; 64]),
        };
        let signature = key.sign(&observation.canonical_message().unwrap()).to_bytes();
        observation.signature = envelope(signature);
        observation
    }

    fn expectation() -> XeniaWitnessFrontierExpectationV1 {
        XeniaWitnessFrontierExpectationV1 {
            trusted_ledger_public_key: signing_key().verifying_key().to_bytes(),
            source_epoch: 3,
            anchor_policy_digest: POLICY,
            witness_id: WITNESS,
            challenge: CHALLENGE,
        }
    }

    fn freshness() -> XeniaWitnessObservationFreshnessV1 {
        XeniaWitnessObservationFreshnessV1 {
            earliest_now_unix_s: 1_010,
            latest_now_unix_s: 1_012,
            max_age_s: 30,
            max_future_skew_s: 2,
        }
    }

    #[test]
    fn exact_anchor_and_fresh_observation_verify() {
        let anchor = signed_anchor();
        let observation = signed_observation(&anchor);
        let verified = verify_xenia_witness_frontier_v1(
            &anchor,
            &observation,
            expectation(),
            freshness(),
        )
        .unwrap();
        assert_eq!(verified.source_sequence(), 1);
        assert_eq!(verified.high_watermark(), 9);
        assert_eq!(verified.reservation_head(), Digest32(RESERVATION_HEAD));
        assert_eq!(verified.anchor_fingerprint(), Digest32(anchor.fingerprint().unwrap()));
        assert_eq!(
            verified.freshness_evidence_digest(),
            Digest32(observation.fingerprint().unwrap())
        );
    }

    #[test]
    fn challenge_substitution_is_rejected() {
        let anchor = signed_anchor();
        let observation = signed_observation(&anchor);
        let mut expected = expectation();
        expected.challenge[0] ^= 1;
        assert_eq!(
            verify_xenia_witness_frontier_v1(&anchor, &observation, expected, freshness()),
            Err(XeniaWitnessFrontierError::ObservationBindingMismatch)
        );
    }

    #[test]
    fn stale_observation_is_rejected() {
        let anchor = signed_anchor();
        let observation = signed_observation(&anchor);
        let stale = XeniaWitnessObservationFreshnessV1 {
            earliest_now_unix_s: 2_000,
            latest_now_unix_s: 2_001,
            max_age_s: 30,
            max_future_skew_s: 2,
        };
        assert_eq!(
            verify_xenia_witness_frontier_v1(&anchor, &observation, expectation(), stale),
            Err(XeniaWitnessFrontierError::ObservationStaleOrFuture)
        );
    }

    #[test]
    fn current_summary_substitution_is_rejected_even_when_resigned() {
        let anchor = signed_anchor();
        let mut observation = signed_observation(&anchor);
        observation.current.as_mut().unwrap().operation_id[0] ^= 1;
        let key = signing_key();
        observation.signature = envelope(key.sign(&observation.canonical_message().unwrap()).to_bytes());
        assert_eq!(
            verify_xenia_witness_frontier_v1(
                &anchor,
                &observation,
                expectation(),
                freshness(),
            ),
            Err(XeniaWitnessFrontierError::ObservationCurrentAnchorMismatch)
        );
    }

    #[test]
    fn source_relabelling_is_rejected_before_trust() {
        let mut anchor = signed_anchor();
        anchor.target.source_id[0] ^= 1;
        let key = signing_key();
        anchor.signature = envelope(key.sign(&anchor.canonical_message().unwrap()).to_bytes());
        assert!(matches!(
            verify_xenia_witness_frontier_v1(
                &anchor,
                &signed_observation(&signed_anchor()),
                expectation(),
                freshness(),
            ),
            Err(XeniaWitnessFrontierError::SourceBindingMismatch)
                | Err(XeniaWitnessFrontierError::AnchorBindingMismatch)
        ));
    }

    #[test]
    fn ledger_context_regression_is_rejected() {
        let anchor = signed_anchor();
        let mut observation = signed_observation(&anchor);
        observation.ledger_entry_count = anchor.ledger_entry_count - 1;
        let key = signing_key();
        observation.signature = envelope(key.sign(&observation.canonical_message().unwrap()).to_bytes());
        assert_eq!(
            verify_xenia_witness_frontier_v1(
                &anchor,
                &observation,
                expectation(),
                freshness(),
            ),
            Err(XeniaWitnessFrontierError::LedgerContextRegression)
        );
    }

    #[test]
    fn signature_tampering_is_rejected() {
        let anchor = signed_anchor();
        let mut observation = signed_observation(&anchor);
        observation.signature.signature[0] ^= 1;
        assert_eq!(
            verify_xenia_witness_frontier_v1(
                &anchor,
                &observation,
                expectation(),
                freshness(),
            ),
            Err(XeniaWitnessFrontierError::BadSignature)
        );
    }
}
