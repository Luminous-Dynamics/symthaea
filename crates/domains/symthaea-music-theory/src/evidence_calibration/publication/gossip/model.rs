// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persisted authenticated-gossip models.

use serde::{Deserialize, Serialize};

use crate::evidence_calibration::{
    CalibrationPublicationCatalogCheckpoint, CalibrationSignerIdentity,
};

pub const CALIBRATION_PUBLICATION_GOSSIP_PAYLOAD_VERSION: &str =
    "score-evidence-calibration-publication-gossip-payload-v1";
pub const CALIBRATION_SIGNED_PUBLICATION_GOSSIP_VERSION: &str =
    "score-evidence-calibration-signed-publication-gossip-v1";
pub const CALIBRATION_PUBLICATION_GOSSIP_LEDGER_VERSION: &str =
    "score-evidence-calibration-publication-gossip-ledger-v1";
pub const CALIBRATION_PUBLICATION_GOSSIP_CONFLICT_PROOF_VERSION: &str =
    "score-evidence-calibration-publication-gossip-conflict-proof-v1";
pub const CALIBRATION_PUBLICATION_GOSSIP_AUDIT_VERSION: &str =
    "score-evidence-calibration-publication-gossip-audit-v1";
pub const CALIBRATION_PUBLICATION_GOSSIP_CONFLICT_AUDIT_VERSION: &str =
    "score-evidence-calibration-publication-gossip-conflict-audit-v1";

pub(crate) const GOSSIP_PAYLOAD_DOMAIN: &[u8] =
    b"symthaea.score-evidence.publication-gossip-payload.v1\0";
pub(crate) const GOSSIP_ENVELOPE_DOMAIN: &[u8] =
    b"symthaea.score-evidence.signed-publication-gossip.v1\0";
pub(crate) const GOSSIP_LEDGER_DOMAIN: &[u8] =
    b"symthaea.score-evidence.publication-gossip-ledger.v1\0";
pub(crate) const GOSSIP_CONFLICT_DOMAIN: &[u8] =
    b"symthaea.score-evidence.publication-gossip-conflict-proof.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationGossipPayload {
    pub payload_version: String,
    pub observer_id: String,
    pub checkpoint: CalibrationPublicationCatalogCheckpoint,
    pub previous_observed_checkpoint_sha256: Option<String>,
    pub witness_policy_epoch_sha256: String,
    pub observed_epoch: u64,
    pub payload_sha256: String,
}

impl CalibrationPublicationGossipPayload {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GOSSIP_PAYLOAD_DOMAIN);
        push_field(&mut bytes, &self.payload_version);
        push_field(&mut bytes, &self.observer_id);
        push_field(&mut bytes, &self.checkpoint.checkpoint_sha256);
        push_optional_field(
            &mut bytes,
            self.previous_observed_checkpoint_sha256.as_deref(),
        );
        push_field(&mut bytes, &self.witness_policy_epoch_sha256);
        bytes.extend_from_slice(&self.observed_epoch.to_le_bytes());
        bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationSignedPublicationGossip {
    pub envelope_version: String,
    pub payload: CalibrationPublicationGossipPayload,
    pub signer: CalibrationSignerIdentity,
    pub signature_hex: String,
    pub envelope_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationGossipLedger {
    pub ledger_version: String,
    pub catalog_id: String,
    pub authority_id: String,
    pub statements: Vec<CalibrationSignedPublicationGossip>,
    pub ledger_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationPublicationGossipConflictKind {
    ObserverRollback,
    ObserverEquivocation,
    AuthorityEquivocation,
    CheckpointFork,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationGossipConflictProof {
    pub proof_version: String,
    pub kind: CalibrationPublicationGossipConflictKind,
    pub first: CalibrationSignedPublicationGossip,
    pub second: CalibrationSignedPublicationGossip,
    pub proof_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationPublicationGossipIssueCode {
    LedgerVersionMismatch,
    EmptyIdentity,
    PayloadVersionMismatch,
    EnvelopeVersionMismatch,
    EmptyObserverIdentity,
    EmptySignerIdentity,
    InvalidSignatureHex,
    CheckpointVersionMismatch,
    CheckpointSha256Mismatch,
    CheckpointIdentityMismatch,
    InvalidPolicyEpochSha256,
    PayloadSha256Mismatch,
    EnvelopeSha256Mismatch,
    DuplicateStatement,
    DuplicateObserverCheckpoint,
    PreviousObservationMismatch,
    ObserverEpochRegression,
    ObserverRollback,
    ObserverEquivocation,
    AuthorityEquivocation,
    CheckpointFork,
    SignatureRejected,
    LedgerSha256Mismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationGossipIssue {
    pub code: CalibrationPublicationGossipIssueCode,
    pub observer_id: Option<String>,
    pub checkpoint_sha256: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationGossipAuditReport {
    pub audit_version: String,
    pub structurally_valid: bool,
    pub signatures_authenticated: bool,
    pub authenticated_statements: u64,
    pub rollback_detected: bool,
    pub observer_equivocation_detected: bool,
    pub authority_equivocation_detected: bool,
    pub fork_detected: bool,
    pub issues: Vec<CalibrationPublicationGossipIssue>,
}

impl CalibrationPublicationGossipAuditReport {
    pub fn integrity_valid(&self) -> bool {
        self.structurally_valid
            && self.issues.iter().all(|issue| {
                matches!(
                    issue.code,
                    CalibrationPublicationGossipIssueCode::ObserverRollback
                        | CalibrationPublicationGossipIssueCode::ObserverEquivocation
                        | CalibrationPublicationGossipIssueCode::AuthorityEquivocation
                        | CalibrationPublicationGossipIssueCode::CheckpointFork
                )
            })
    }

    pub fn conflict_free(&self) -> bool {
        !self.rollback_detected
            && !self.observer_equivocation_detected
            && !self.authority_equivocation_detected
            && !self.fork_detected
    }

    pub fn accepted(&self) -> bool {
        self.integrity_valid() && self.signatures_authenticated && self.conflict_free()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationPublicationGossipConflictIssueCode {
    ProofVersionMismatch,
    FirstStatementInvalid,
    SecondStatementInvalid,
    ConflictKindMismatch,
    ProofSha256Mismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationGossipConflictIssue {
    pub code: CalibrationPublicationGossipConflictIssueCode,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationGossipConflictAuditReport {
    pub audit_version: String,
    pub issues: Vec<CalibrationPublicationGossipConflictIssue>,
}

impl CalibrationPublicationGossipConflictAuditReport {
    pub fn valid(&self) -> bool {
        self.issues.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "details", rename_all = "snake_case")]
pub enum CalibrationPublicationGossipError {
    InvalidLedger { issues: usize },
    InvalidStatement { issues: usize },
    SignatureRejected,
    DuplicateStatement,
    IdentityMismatch,
    NoConflict,
}

impl std::fmt::Display for CalibrationPublicationGossipError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLedger { issues } => {
                write!(formatter, "gossip ledger audit failed with {issues} issues")
            }
            Self::InvalidStatement { issues } => {
                write!(formatter, "gossip statement audit failed with {issues} issues")
            }
            Self::SignatureRejected => write!(formatter, "external gossip signature verifier rejected the statement"),
            Self::DuplicateStatement => write!(formatter, "gossip statement already exists"),
            Self::IdentityMismatch => write!(formatter, "gossip statement and ledger identities differ"),
            Self::NoConflict => write!(formatter, "the supplied gossip statements do not prove the requested conflict"),
        }
    }
}

impl std::error::Error for CalibrationPublicationGossipError {}

pub trait CalibrationPublicationGossipVerifier {
    type Error: std::fmt::Display;

    fn verify(
        &self,
        payload: &[u8],
        signer: &CalibrationSignerIdentity,
        signature: &[u8],
    ) -> Result<(), Self::Error>;
}

fn push_field(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn push_optional_field(bytes: &mut Vec<u8>, value: Option<&str>) {
    match value {
        None => bytes.push(0),
        Some(value) => {
            bytes.push(1);
            push_field(bytes, value);
        }
    }
}
