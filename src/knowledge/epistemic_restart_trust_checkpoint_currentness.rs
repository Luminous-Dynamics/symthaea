// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Current-head attestation for the protected EKM restart trust-context checkpoint.
//!
//! EKM-054 proves one exact joint trust-context checkpoint is protected and valid,
//! but a generic signature or historical protected checkpoint does not prove that
//! checkpoint is still the deployment's current/latest head. This module adds a
//! separate provider contract whose evidence semantics explicitly mean monotonic
//! current-head state.
//!
//! Currentness is time-bounded audit evidence only. It does not mutate trusted
//! state, authorize activation, or expose any restart sandbox capability.

use crate::knowledge::epistemic_restart_trust_checkpoint::{
    RestartTrustContextCheckpointError, VerifiedRestartTrustContextCheckpointV1,
};
use std::error::Error;
use std::fmt;

pub const MAX_TRUST_CONTEXT_CURRENTNESS_AUTHORITY_ID_BYTES: usize = 256;
pub const MAX_TRUST_CONTEXT_CURRENTNESS_PROOF_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartTrustContextCurrentnessVersion {
    V1,
}

/// Evidence classes whose provider contract explicitly means current monotonic
/// head state. A generic detached signature is deliberately absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartTrustContextCurrentnessEvidenceKindV1 {
    MonotonicProtectedState,
    HardwareMonotonicCounter,
    TransparencyLogHead,
    CurrentHeadWitnessQuorum,
}

impl RestartTrustContextCurrentnessEvidenceKindV1 {
    fn tag(self) -> u8 {
        match self {
            Self::MonotonicProtectedState => 1,
            Self::HardwareMonotonicCounter => 2,
            Self::TransparencyLogHead => 3,
            Self::CurrentHeadWitnessQuorum => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RestartTrustContextCurrentnessDigestV1([u8; 32]);

impl RestartTrustContextCurrentnessDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
        }
        out
    }
}

/// Exact current-head question presented to the deployment provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartTrustContextCurrentnessStatementV1 {
    version: RestartTrustContextCurrentnessVersion,
    deployment_id: String,
    trust_domain_id: String,
    checkpoint_sequence: u64,
    checkpoint_statement_digest: [u8; 32],
    context_digest: [u8; 32],
    context_committed_at_cycle: u64,
    anchor_sequence: u64,
    anchor_capture_cycle: u64,
    verifier_trust_snapshot_sequence: u64,
    checkpoint_verified_at_cycle: u64,
    checkpoint_expires_at_cycle: u64,
    attested_at_cycle: u64,
    expires_at_cycle: u64,
    authority_id: String,
    evidence_kind: RestartTrustContextCurrentnessEvidenceKindV1,
}

impl RestartTrustContextCurrentnessStatementV1 {
    pub fn new(
        checkpoint: &VerifiedRestartTrustContextCheckpointV1,
        attested_at_cycle: u64,
        expires_at_cycle: u64,
        authority_id: impl Into<String>,
        evidence_kind: RestartTrustContextCurrentnessEvidenceKindV1,
    ) -> Result<Self, RestartTrustContextCurrentnessError> {
        checkpoint
            .verify_internal()
            .map_err(RestartTrustContextCurrentnessError::CheckpointRejected)?;
        if checkpoint.trusted_state_mutated()
            || checkpoint.quarantine_construction_authorized()
            || checkpoint.writable_hydration_authorized()
            || checkpoint.activation_authorized()
        {
            return Err(RestartTrustContextCurrentnessError::UnexpectedCheckpointAuthority);
        }
        if attested_at_cycle < checkpoint.verified_at_cycle() {
            return Err(RestartTrustContextCurrentnessError::AttestationPredatesCheckpoint {
                attested_at_cycle,
                checkpoint_verified_at_cycle: checkpoint.verified_at_cycle(),
            });
        }
        if attested_at_cycle >= checkpoint.statement().expires_at_cycle() {
            return Err(RestartTrustContextCurrentnessError::CheckpointExpiredBeforeAttestation {
                attested_at_cycle,
                checkpoint_expires_at_cycle: checkpoint.statement().expires_at_cycle(),
            });
        }
        if expires_at_cycle <= attested_at_cycle
            || expires_at_cycle > checkpoint.statement().expires_at_cycle()
        {
            return Err(RestartTrustContextCurrentnessError::InvalidValidityWindow {
                attested_at_cycle,
                expires_at_cycle,
                checkpoint_expires_at_cycle: checkpoint.statement().expires_at_cycle(),
            });
        }

        let statement = Self {
            version: RestartTrustContextCurrentnessVersion::V1,
            deployment_id: checkpoint.statement().deployment_id().to_string(),
            trust_domain_id: checkpoint.statement().trust_domain_id().to_string(),
            checkpoint_sequence: checkpoint.statement().sequence(),
            checkpoint_statement_digest: checkpoint.statement_digest().as_bytes(),
            context_digest: checkpoint.statement().context_digest().as_bytes(),
            context_committed_at_cycle: checkpoint.statement().context_committed_at_cycle(),
            anchor_sequence: checkpoint.statement().anchor_sequence(),
            anchor_capture_cycle: checkpoint.statement().anchor_capture_cycle(),
            verifier_trust_snapshot_sequence: checkpoint
                .statement()
                .verifier_trust_snapshot_sequence(),
            checkpoint_verified_at_cycle: checkpoint.verified_at_cycle(),
            checkpoint_expires_at_cycle: checkpoint.statement().expires_at_cycle(),
            attested_at_cycle,
            expires_at_cycle,
            authority_id: authority_id.into(),
            evidence_kind,
        };
        statement.validate_shape()?;
        Ok(statement)
    }

    pub fn version(&self) -> RestartTrustContextCurrentnessVersion {
        self.version
    }
    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }
    pub fn trust_domain_id(&self) -> &str {
        &self.trust_domain_id
    }
    pub fn checkpoint_sequence(&self) -> u64 {
        self.checkpoint_sequence
    }
    pub fn checkpoint_statement_digest(&self) -> [u8; 32] {
        self.checkpoint_statement_digest
    }
    pub fn context_digest(&self) -> [u8; 32] {
        self.context_digest
    }
    pub fn context_committed_at_cycle(&self) -> u64 {
        self.context_committed_at_cycle
    }
    pub fn anchor_sequence(&self) -> u64 {
        self.anchor_sequence
    }
    pub fn anchor_capture_cycle(&self) -> u64 {
        self.anchor_capture_cycle
    }
    pub fn verifier_trust_snapshot_sequence(&self) -> u64 {
        self.verifier_trust_snapshot_sequence
    }
    pub fn checkpoint_verified_at_cycle(&self) -> u64 {
        self.checkpoint_verified_at_cycle
    }
    pub fn checkpoint_expires_at_cycle(&self) -> u64 {
        self.checkpoint_expires_at_cycle
    }
    pub fn attested_at_cycle(&self) -> u64 {
        self.attested_at_cycle
    }
    pub fn expires_at_cycle(&self) -> u64 {
        self.expires_at_cycle
    }
    pub fn authority_id(&self) -> &str {
        &self.authority_id
    }
    pub fn evidence_kind(&self) -> RestartTrustContextCurrentnessEvidenceKindV1 {
        self.evidence_kind
    }

    fn validate_shape(&self) -> Result<(), RestartTrustContextCurrentnessError> {
        if self.checkpoint_sequence == 0
            || self.anchor_sequence == 0
            || self.verifier_trust_snapshot_sequence == 0
        {
            return Err(RestartTrustContextCurrentnessError::InvalidEmbeddedSequence);
        }
        validate_identifier(&self.deployment_id)?;
        validate_identifier(&self.trust_domain_id)?;
        if self.authority_id.trim().is_empty()
            || self.authority_id != self.authority_id.trim()
            || self.authority_id.len() > MAX_TRUST_CONTEXT_CURRENTNESS_AUTHORITY_ID_BYTES
        {
            return Err(RestartTrustContextCurrentnessError::InvalidAuthorityId);
        }
        if self.context_committed_at_cycle < self.anchor_capture_cycle {
            return Err(RestartTrustContextCurrentnessError::ContextCommitPredatesAnchor);
        }
        if self.attested_at_cycle < self.checkpoint_verified_at_cycle {
            return Err(RestartTrustContextCurrentnessError::AttestationPredatesCheckpoint {
                attested_at_cycle: self.attested_at_cycle,
                checkpoint_verified_at_cycle: self.checkpoint_verified_at_cycle,
            });
        }
        if self.attested_at_cycle >= self.checkpoint_expires_at_cycle {
            return Err(RestartTrustContextCurrentnessError::CheckpointExpiredBeforeAttestation {
                attested_at_cycle: self.attested_at_cycle,
                checkpoint_expires_at_cycle: self.checkpoint_expires_at_cycle,
            });
        }
        if self.expires_at_cycle <= self.attested_at_cycle
            || self.expires_at_cycle > self.checkpoint_expires_at_cycle
        {
            return Err(RestartTrustContextCurrentnessError::InvalidValidityWindow {
                attested_at_cycle: self.attested_at_cycle,
                expires_at_cycle: self.expires_at_cycle,
                checkpoint_expires_at_cycle: self.checkpoint_expires_at_cycle,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartTrustContextCurrentnessEvidenceV1 {
    statement: RestartTrustContextCurrentnessStatementV1,
    proof: Vec<u8>,
}

impl RestartTrustContextCurrentnessEvidenceV1 {
    pub fn new(
        statement: RestartTrustContextCurrentnessStatementV1,
        proof: Vec<u8>,
    ) -> Result<Self, RestartTrustContextCurrentnessError> {
        statement.validate_shape()?;
        if proof.is_empty() {
            return Err(RestartTrustContextCurrentnessError::EmptyProof);
        }
        if proof.len() > MAX_TRUST_CONTEXT_CURRENTNESS_PROOF_BYTES {
            return Err(RestartTrustContextCurrentnessError::ProofTooLarge {
                actual: proof.len(),
                maximum: MAX_TRUST_CONTEXT_CURRENTNESS_PROOF_BYTES,
            });
        }
        Ok(Self { statement, proof })
    }

    pub fn statement(&self) -> &RestartTrustContextCurrentnessStatementV1 {
        &self.statement
    }
    pub fn proof(&self) -> &[u8] {
        &self.proof
    }
}

/// Returning true means the provider attests that the exact EKM-054 checkpoint
/// named by the canonical statement is the current monotonic/head state under the
/// selected evidence kind at `attested_at_cycle`.
pub trait RestartTrustContextCurrentnessVerifierV1 {
    fn verify_current_restart_trust_context_checkpoint(
        &self,
        evidence_kind: RestartTrustContextCurrentnessEvidenceKindV1,
        authority_id: &str,
        statement_digest: [u8; 32],
        proof: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRestartTrustContextCurrentnessV1 {
    statement: RestartTrustContextCurrentnessStatementV1,
    statement_digest: RestartTrustContextCurrentnessDigestV1,
    proof_digest: [u8; 32],
    verified_at_cycle: u64,
    current_head_proven: bool,
    trusted_state_mutated: bool,
    activation_preflight_authorized: bool,
    activation_authorized: bool,
    trusted_checkpoint_commit_authorized: bool,
}

impl VerifiedRestartTrustContextCurrentnessV1 {
    pub fn statement(&self) -> &RestartTrustContextCurrentnessStatementV1 {
        &self.statement
    }
    pub fn statement_digest(&self) -> RestartTrustContextCurrentnessDigestV1 {
        self.statement_digest
    }
    pub fn proof_digest(&self) -> [u8; 32] {
        self.proof_digest
    }
    pub fn verified_at_cycle(&self) -> u64 {
        self.verified_at_cycle
    }
    pub fn current_head_proven(&self) -> bool {
        self.current_head_proven
    }
    pub fn trusted_state_mutated(&self) -> bool {
        self.trusted_state_mutated
    }
    pub fn activation_preflight_authorized(&self) -> bool {
        self.activation_preflight_authorized
    }
    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }
    pub fn trusted_checkpoint_commit_authorized(&self) -> bool {
        self.trusted_checkpoint_commit_authorized
    }

    pub fn verify_internal(&self) -> Result<(), RestartTrustContextCurrentnessError> {
        self.statement.validate_shape()?;
        if digest_restart_trust_context_currentness_statement(&self.statement)?
            != self.statement_digest
        {
            return Err(RestartTrustContextCurrentnessError::StatementDigestMismatch);
        }
        if !self.current_head_proven
            || self.trusted_state_mutated
            || self.activation_preflight_authorized
            || self.activation_authorized
            || self.trusted_checkpoint_commit_authorized
        {
            return Err(RestartTrustContextCurrentnessError::UnexpectedAuthority);
        }
        Ok(())
    }
}

pub fn verify_restart_trust_context_currentness(
    evidence: &RestartTrustContextCurrentnessEvidenceV1,
    checkpoint: &VerifiedRestartTrustContextCheckpointV1,
    observed_at_cycle: u64,
    verifier: &dyn RestartTrustContextCurrentnessVerifierV1,
) -> Result<VerifiedRestartTrustContextCurrentnessV1, RestartTrustContextCurrentnessError> {
    checkpoint
        .verify_internal()
        .map_err(RestartTrustContextCurrentnessError::CheckpointRejected)?;
    evidence.statement.validate_shape()?;
    validate_statement_binding(&evidence.statement, checkpoint)?;

    if observed_at_cycle < evidence.statement.attested_at_cycle {
        return Err(RestartTrustContextCurrentnessError::ObservationPredatesAttestation {
            observed_at_cycle,
            attested_at_cycle: evidence.statement.attested_at_cycle,
        });
    }
    if observed_at_cycle >= evidence.statement.expires_at_cycle {
        return Err(RestartTrustContextCurrentnessError::CurrentnessEvidenceExpired {
            observed_at_cycle,
            expires_at_cycle: evidence.statement.expires_at_cycle,
        });
    }

    let statement_digest = digest_restart_trust_context_currentness_statement(&evidence.statement)?;
    let accepted = verifier
        .verify_current_restart_trust_context_checkpoint(
            evidence.statement.evidence_kind,
            &evidence.statement.authority_id,
            statement_digest.as_bytes(),
            &evidence.proof,
        )
        .map_err(RestartTrustContextCurrentnessError::VerificationProvider)?;
    if !accepted {
        return Err(RestartTrustContextCurrentnessError::ProofRejected);
    }

    let mut proof_hasher = blake3::Hasher::new();
    proof_hasher.update(b"symthaea-ekm-restart-trust-context-currentness-proof-v1");
    proof_hasher.update(&evidence.proof);
    let receipt = VerifiedRestartTrustContextCurrentnessV1 {
        statement: evidence.statement.clone(),
        statement_digest,
        proof_digest: *proof_hasher.finalize().as_bytes(),
        verified_at_cycle: observed_at_cycle,
        current_head_proven: true,
        trusted_state_mutated: false,
        activation_preflight_authorized: false,
        activation_authorized: false,
        trusted_checkpoint_commit_authorized: false,
    };
    receipt.verify_internal()?;
    Ok(receipt)
}

pub fn digest_restart_trust_context_currentness_statement(
    statement: &RestartTrustContextCurrentnessStatementV1,
) -> Result<RestartTrustContextCurrentnessDigestV1, RestartTrustContextCurrentnessError> {
    statement.validate_shape()?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-trust-context-currentness-v1");
    hasher.update(&[1]);
    hash_bytes(&mut hasher, statement.deployment_id.as_bytes())?;
    hash_bytes(&mut hasher, statement.trust_domain_id.as_bytes())?;
    hasher.update(&statement.checkpoint_sequence.to_le_bytes());
    hasher.update(&statement.checkpoint_statement_digest);
    hasher.update(&statement.context_digest);
    hasher.update(&statement.context_committed_at_cycle.to_le_bytes());
    hasher.update(&statement.anchor_sequence.to_le_bytes());
    hasher.update(&statement.anchor_capture_cycle.to_le_bytes());
    hasher.update(&statement.verifier_trust_snapshot_sequence.to_le_bytes());
    hasher.update(&statement.checkpoint_verified_at_cycle.to_le_bytes());
    hasher.update(&statement.checkpoint_expires_at_cycle.to_le_bytes());
    hasher.update(&statement.attested_at_cycle.to_le_bytes());
    hasher.update(&statement.expires_at_cycle.to_le_bytes());
    hash_bytes(&mut hasher, statement.authority_id.as_bytes())?;
    hasher.update(&[statement.evidence_kind.tag()]);
    Ok(RestartTrustContextCurrentnessDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn validate_statement_binding(
    statement: &RestartTrustContextCurrentnessStatementV1,
    checkpoint: &VerifiedRestartTrustContextCheckpointV1,
) -> Result<(), RestartTrustContextCurrentnessError> {
    if statement.deployment_id != checkpoint.statement().deployment_id()
        || statement.trust_domain_id != checkpoint.statement().trust_domain_id()
        || statement.checkpoint_sequence != checkpoint.statement().sequence()
        || statement.checkpoint_statement_digest != checkpoint.statement_digest().as_bytes()
        || statement.context_digest != checkpoint.statement().context_digest().as_bytes()
        || statement.context_committed_at_cycle != checkpoint.statement().context_committed_at_cycle()
        || statement.anchor_sequence != checkpoint.statement().anchor_sequence()
        || statement.anchor_capture_cycle != checkpoint.statement().anchor_capture_cycle()
        || statement.verifier_trust_snapshot_sequence
            != checkpoint.statement().verifier_trust_snapshot_sequence()
        || statement.checkpoint_verified_at_cycle != checkpoint.verified_at_cycle()
        || statement.checkpoint_expires_at_cycle != checkpoint.statement().expires_at_cycle()
    {
        return Err(RestartTrustContextCurrentnessError::CheckpointBindingMismatch);
    }
    if checkpoint.trusted_state_mutated()
        || checkpoint.quarantine_construction_authorized()
        || checkpoint.writable_hydration_authorized()
        || checkpoint.activation_authorized()
    {
        return Err(RestartTrustContextCurrentnessError::UnexpectedCheckpointAuthority);
    }
    Ok(())
}

fn validate_identifier(value: &str) -> Result<(), RestartTrustContextCurrentnessError> {
    if value.trim().is_empty() || value != value.trim() || value.len() > 256 {
        return Err(RestartTrustContextCurrentnessError::InvalidIdentifier);
    }
    Ok(())
}

fn hash_bytes(
    hasher: &mut blake3::Hasher,
    bytes: &[u8],
) -> Result<(), RestartTrustContextCurrentnessError> {
    let len = u64::try_from(bytes.len())
        .map_err(|_| RestartTrustContextCurrentnessError::LengthOverflow)?;
    hasher.update(&len.to_le_bytes());
    hasher.update(bytes);
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartTrustContextCurrentnessError {
    CheckpointRejected(RestartTrustContextCheckpointError),
    InvalidEmbeddedSequence,
    InvalidIdentifier,
    InvalidAuthorityId,
    ContextCommitPredatesAnchor,
    AttestationPredatesCheckpoint {
        attested_at_cycle: u64,
        checkpoint_verified_at_cycle: u64,
    },
    CheckpointExpiredBeforeAttestation {
        attested_at_cycle: u64,
        checkpoint_expires_at_cycle: u64,
    },
    InvalidValidityWindow {
        attested_at_cycle: u64,
        expires_at_cycle: u64,
        checkpoint_expires_at_cycle: u64,
    },
    EmptyProof,
    ProofTooLarge {
        actual: usize,
        maximum: usize,
    },
    CheckpointBindingMismatch,
    UnexpectedCheckpointAuthority,
    ObservationPredatesAttestation {
        observed_at_cycle: u64,
        attested_at_cycle: u64,
    },
    CurrentnessEvidenceExpired {
        observed_at_cycle: u64,
        expires_at_cycle: u64,
    },
    VerificationProvider(String),
    ProofRejected,
    StatementDigestMismatch,
    UnexpectedAuthority,
    LengthOverflow,
}

impl fmt::Display for RestartTrustContextCurrentnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "restart trust-context currentness rejected: {self:?}")
    }
}

impl Error for RestartTrustContextCurrentnessError {}
