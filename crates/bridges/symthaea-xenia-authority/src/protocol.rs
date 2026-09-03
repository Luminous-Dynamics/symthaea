// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent Symthaea representation of Xenia agent authority V1.
//!
//! This file intentionally does not depend on Xenia crates. Its canonical bytes
//! must remain compatible with the frozen `xenia-agent-capability-authorization-v1`
//! contract; the cross-repository signature fixture detects drift.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const AGENT_CAPABILITY_AUTHORIZATION_SCHEMA_VERSION: u16 = 1;
pub const AGENT_CAPABILITY_AUTHORIZATION_DOMAIN: &[u8] =
    b"xenia.agent-capability-authorization.v1\0";
pub const AGENT_CAPABILITY_ATTESTATION_SCHEMA: &str = "xenia-agent-capability-attestation-v1";
pub const XENIA_LEDGER_CHECKPOINT_SCHEMA: &str = "xenia-ledger-checkpoint-v1";
pub const ED25519_SIGNATURE_ALGORITHM: &str = "ed25519-rfc8032";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum TranscriptSignatureSuiteV1 {
    Ed25519Rfc8032 = 1,
    MlDsa65Fips204 = 2,
    MlDsa87Fips204 = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaCheckpointAnchorV1 {
    pub sequence: u64,
    pub digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaAgentAuthorizationV1 {
    pub schema_version: u16,
    pub authorization_id: [u8; 16],
    pub session_id: [u8; 16],
    pub session_transcript_hash: [u8; 32],
    pub session_signature_suite: TranscriptSignatureSuiteV1,
    pub capability_digest: [u8; 32],
    pub executor_workload_digest: [u8; 32],
    pub authority_epoch: u64,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub nonce: [u8; 16],
    pub ledger_entry_count: u64,
    pub ledger_head_hash: [u8; 32],
    pub prior_checkpoint: Option<XeniaCheckpointAnchorV1>,
}

impl XeniaAgentAuthorizationV1 {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.schema_version != AGENT_CAPABILITY_AUTHORIZATION_SCHEMA_VERSION {
            return Err(ProtocolError::UnsupportedAuthorizationSchema);
        }
        if self.authorization_id == [0; 16] {
            return Err(ProtocolError::ZeroAuthorizationId);
        }
        if self.session_id == [0; 16] {
            return Err(ProtocolError::ZeroSessionId);
        }
        if self.session_transcript_hash == [0; 32] {
            return Err(ProtocolError::ZeroSessionTranscriptHash);
        }
        if self.capability_digest == [0; 32] {
            return Err(ProtocolError::ZeroCapabilityDigest);
        }
        if self.executor_workload_digest == [0; 32] {
            return Err(ProtocolError::ZeroExecutorWorkloadDigest);
        }
        if self.expires_at_unix_s <= self.issued_at_unix_s {
            return Err(ProtocolError::InvalidValidityWindow);
        }
        if self.nonce == [0; 16] {
            return Err(ProtocolError::ZeroNonce);
        }
        if self.ledger_entry_count == 0 || self.ledger_head_hash == [0; 32] {
            return Err(ProtocolError::MissingLedgerFrontier);
        }
        if self
            .prior_checkpoint
            .is_some_and(|anchor| anchor.digest == [0; 32])
        {
            return Err(ProtocolError::ZeroCheckpointDigest);
        }
        Ok(())
    }

    pub fn canonical_message(&self) -> Result<Vec<u8>, ProtocolError> {
        self.validate()?;
        let mut out = Vec::with_capacity(320);
        out.extend_from_slice(AGENT_CAPABILITY_AUTHORIZATION_DOMAIN);
        out.extend_from_slice(&self.schema_version.to_be_bytes());
        out.extend_from_slice(&self.authorization_id);
        out.extend_from_slice(&self.session_id);
        out.extend_from_slice(&self.session_transcript_hash);
        out.push(self.session_signature_suite as u8);
        out.extend_from_slice(&self.capability_digest);
        out.extend_from_slice(&self.executor_workload_digest);
        out.extend_from_slice(&self.authority_epoch.to_be_bytes());
        out.extend_from_slice(&self.issued_at_unix_s.to_be_bytes());
        out.extend_from_slice(&self.expires_at_unix_s.to_be_bytes());
        out.extend_from_slice(&self.nonce);
        out.extend_from_slice(&self.ledger_entry_count.to_be_bytes());
        out.extend_from_slice(&self.ledger_head_hash);
        match self.prior_checkpoint {
            None => out.push(0),
            Some(anchor) => {
                out.push(1);
                out.extend_from_slice(&anchor.sequence.to_be_bytes());
                out.extend_from_slice(&anchor.digest);
            }
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaSignatureEnvelopeV1 {
    pub algorithm: String,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaAgentCapabilityAttestationV1 {
    pub schema: String,
    pub authorization: XeniaAgentAuthorizationV1,
    pub ledger_public_key_fingerprint: [u8; 32],
    pub signature: XeniaSignatureEnvelopeV1,
}

/// Privacy-minimized signed Xenia ledger freshness checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaLedgerCheckpointV1 {
    pub schema: String,
    pub entry_count: u64,
    pub head_hash: [u8; 32],
    pub ledger_public_key: [u8; 32],
    pub timestamp_unix_secs: u64,
    /// Ed25519 signature; V1 verifier requires exactly 64 bytes.
    pub signature: Vec<u8>,
}

impl XeniaLedgerCheckpointV1 {
    pub fn signature_message(&self) -> Result<Vec<u8>, ProtocolError> {
        if self.schema != XENIA_LEDGER_CHECKPOINT_SCHEMA {
            return Err(ProtocolError::UnsupportedLedgerCheckpointSchema);
        }
        let mut message = Vec::with_capacity(64 + XENIA_LEDGER_CHECKPOINT_SCHEMA.len());
        message.extend_from_slice(b"xenia:ledger-checkpoint:v1");
        message.push(0);
        message.extend_from_slice(XENIA_LEDGER_CHECKPOINT_SCHEMA.as_bytes());
        message.push(0);
        message.extend_from_slice(&self.entry_count.to_be_bytes());
        message.extend_from_slice(&self.head_hash);
        message.extend_from_slice(&self.ledger_public_key);
        message.extend_from_slice(&self.timestamp_unix_secs.to_be_bytes());
        Ok(message)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProtocolError {
    #[error("unsupported Xenia agent authorization schema")]
    UnsupportedAuthorizationSchema,
    #[error("authorization id must not be zero")]
    ZeroAuthorizationId,
    #[error("session id must not be zero")]
    ZeroSessionId,
    #[error("session transcript hash must not be zero")]
    ZeroSessionTranscriptHash,
    #[error("capability digest must not be zero")]
    ZeroCapabilityDigest,
    #[error("executor workload digest must not be zero")]
    ZeroExecutorWorkloadDigest,
    #[error("authorization validity window is invalid")]
    InvalidValidityWindow,
    #[error("authorization nonce must not be zero")]
    ZeroNonce,
    #[error("authorization requires a non-empty Xenia ledger frontier")]
    MissingLedgerFrontier,
    #[error("prior checkpoint digest must not be zero")]
    ZeroCheckpointDigest,
    #[error("unsupported Xenia ledger checkpoint schema")]
    UnsupportedLedgerCheckpointSchema,
}
