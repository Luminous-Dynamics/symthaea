// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Protected checkpoint evidence for sealed epistemic restart trust contexts.
//!
//! EKM-052/053 bind the restart anchor and verifier checkpoint into one opaque
//! deployment trust epoch, but that in-memory context still needs an external,
//! rollback-resistant trust story. This module defines the provider boundary and
//! canonical evidence statement for protecting that exact context digest.
//!
//! The module does not choose a filesystem, TPM, secure element, signature
//! algorithm, transparency log, witness quorum, or remote service. Those remain
//! deployment concerns behind [`RestartTrustContextCheckpointVerifierV1`].
//!
//! A verified checkpoint receipt is audit evidence only. It grants no quarantine,
//! hydration, activation, or trusted-state mutation authority.

use super::epistemic_restart_quarantine_facade::{
    RestartQuarantineFacadeError, RestartTrustContextDigestV1, RestartTrustContextHandleV1,
};
use std::error::Error;
use std::fmt;

pub const MAX_RESTART_TRUST_CHECKPOINT_AUTHORITY_ID_BYTES: usize = 256;
pub const MAX_RESTART_TRUST_CHECKPOINT_PROOF_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartTrustContextCheckpointVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartTrustContextCheckpointEvidenceKindV1 {
    ProtectedStorage,
    SignedCheckpoint,
    HardwareAttestation,
    TransparencyCheckpoint,
    WitnessQuorum,
}

impl RestartTrustContextCheckpointEvidenceKindV1 {
    fn tag(self) -> u8 {
        match self {
            Self::ProtectedStorage => 1,
            Self::SignedCheckpoint => 2,
            Self::HardwareAttestation => 3,
            Self::TransparencyCheckpoint => 4,
            Self::WitnessQuorum => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RestartTrustContextCheckpointDigestV1([u8; 32]);

impl RestartTrustContextCheckpointDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

/// Canonical statement an external protection provider must verify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartTrustContextCheckpointStatementV1 {
    version: RestartTrustContextCheckpointVersion,
    sequence: u64,
    deployment_id: String,
    trust_domain_id: String,
    context_digest: RestartTrustContextDigestV1,
    context_committed_at_cycle: u64,
    anchor_sequence: u64,
    anchor_capture_cycle: u64,
    verifier_trust_snapshot_sequence: u64,
    previous_checkpoint_digest: Option<RestartTrustContextCheckpointDigestV1>,
    issued_at_cycle: u64,
    expires_at_cycle: u64,
    authority_id: String,
    evidence_kind: RestartTrustContextCheckpointEvidenceKindV1,
}

impl RestartTrustContextCheckpointStatementV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sequence: u64,
        context: &RestartTrustContextHandleV1,
        previous_checkpoint_digest: Option<RestartTrustContextCheckpointDigestV1>,
        issued_at_cycle: u64,
        expires_at_cycle: u64,
        authority_id: impl Into<String>,
        evidence_kind: RestartTrustContextCheckpointEvidenceKindV1,
    ) -> Result<Self, RestartTrustContextCheckpointError> {
        context
            .verify()
            .map_err(RestartTrustContextCheckpointError::ContextRejected)?;
        let statement = Self {
            version: RestartTrustContextCheckpointVersion::V1,
            sequence,
            deployment_id: context.deployment_id().to_string(),
            trust_domain_id: context.trust_domain_id().to_string(),
            context_digest: context.digest(),
            context_committed_at_cycle: context.committed_at_cycle(),
            anchor_sequence: context.anchor_sequence(),
            anchor_capture_cycle: context.anchor_capture_cycle(),
            verifier_trust_snapshot_sequence: context.verifier_trust_snapshot_sequence(),
            previous_checkpoint_digest,
            issued_at_cycle,
            expires_at_cycle,
            authority_id: authority_id.into(),
            evidence_kind,
        };
        statement.validate_shape()?;
        Ok(statement)
    }

    pub fn version(&self) -> RestartTrustContextCheckpointVersion {
        self.version
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }

    pub fn trust_domain_id(&self) -> &str {
        &self.trust_domain_id
    }

    pub fn context_digest(&self) -> RestartTrustContextDigestV1 {
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

    pub fn previous_checkpoint_digest(&self) -> Option<RestartTrustContextCheckpointDigestV1> {
        self.previous_checkpoint_digest
    }

    pub fn issued_at_cycle(&self) -> u64 {
        self.issued_at_cycle
    }

    pub fn expires_at_cycle(&self) -> u64 {
        self.expires_at_cycle
    }

    pub fn authority_id(&self) -> &str {
        &self.authority_id
    }

    pub fn evidence_kind(&self) -> RestartTrustContextCheckpointEvidenceKindV1 {
        self.evidence_kind
    }

    fn validate_shape(&self) -> Result<(), RestartTrustContextCheckpointError> {
        if self.sequence == 0 {
            return Err(RestartTrustContextCheckpointError::InvalidSequence);
        }
        if (self.sequence == 1) != self.previous_checkpoint_digest.is_none() {
            return Err(RestartTrustContextCheckpointError::InvalidPredecessorShape);
        }
        if self.anchor_sequence == 0 || self.verifier_trust_snapshot_sequence == 0 {
            return Err(RestartTrustContextCheckpointError::InvalidEmbeddedSequence);
        }
        if self.context_committed_at_cycle < self.anchor_capture_cycle {
            return Err(RestartTrustContextCheckpointError::ContextCommitPredatesAnchor {
                context_committed_at_cycle: self.context_committed_at_cycle,
                anchor_capture_cycle: self.anchor_capture_cycle,
            });
        }
        if self.issued_at_cycle < self.context_committed_at_cycle {
            return Err(RestartTrustContextCheckpointError::IssuePredatesContextCommit {
                issued_at_cycle: self.issued_at_cycle,
                context_committed_at_cycle: self.context_committed_at_cycle,
            });
        }
        if self.issued_at_cycle >= self.expires_at_cycle {
            return Err(RestartTrustContextCheckpointError::InvalidValidityWindow);
        }
        validate_identifier(
            &self.deployment_id,
            RestartTrustContextCheckpointField::DeploymentId,
        )?;
        validate_identifier(
            &self.trust_domain_id,
            RestartTrustContextCheckpointField::TrustDomainId,
        )?;
        if self.authority_id.trim().is_empty()
            || self.authority_id != self.authority_id.trim()
            || self.authority_id.len() > MAX_RESTART_TRUST_CHECKPOINT_AUTHORITY_ID_BYTES
        {
            return Err(RestartTrustContextCheckpointError::InvalidAuthorityId);
        }
        Ok(())
    }
}

/// Opaque external proof over one exact trust-context checkpoint statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartTrustContextCheckpointEvidenceV1 {
    statement: RestartTrustContextCheckpointStatementV1,
    proof: Vec<u8>,
}

impl RestartTrustContextCheckpointEvidenceV1 {
    pub fn new(
        statement: RestartTrustContextCheckpointStatementV1,
        proof: Vec<u8>,
    ) -> Result<Self, RestartTrustContextCheckpointError> {
        statement.validate_shape()?;
        if proof.is_empty() {
            return Err(RestartTrustContextCheckpointError::EmptyProof);
        }
        if proof.len() > MAX_RESTART_TRUST_CHECKPOINT_PROOF_BYTES {
            return Err(RestartTrustContextCheckpointError::ProofTooLarge {
                actual: proof.len(),
                maximum: MAX_RESTART_TRUST_CHECKPOINT_PROOF_BYTES,
            });
        }
        Ok(Self { statement, proof })
    }

    pub fn statement(&self) -> &RestartTrustContextCheckpointStatementV1 {
        &self.statement
    }

    pub fn proof(&self) -> &[u8] {
        &self.proof
    }
}

/// External integration seam for protected-storage, signature, hardware,
/// transparency-log, or witness-quorum verification.
pub trait RestartTrustContextCheckpointVerifierV1 {
    fn verify_trust_context_checkpoint(
        &self,
        evidence_kind: RestartTrustContextCheckpointEvidenceKindV1,
        authority_id: &str,
        statement_digest: [u8; 32],
        proof: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRestartTrustContextCheckpointV1 {
    statement: RestartTrustContextCheckpointStatementV1,
    statement_digest: RestartTrustContextCheckpointDigestV1,
    proof_digest: [u8; 32],
    verified_at_cycle: u64,
    trusted_state_mutated: bool,
    quarantine_construction_authorized: bool,
    writable_hydration_authorized: bool,
    activation_authorized: bool,
}

impl VerifiedRestartTrustContextCheckpointV1 {
    pub fn statement(&self) -> &RestartTrustContextCheckpointStatementV1 {
        &self.statement
    }

    pub fn statement_digest(&self) -> RestartTrustContextCheckpointDigestV1 {
        self.statement_digest
    }

    pub fn proof_digest(&self) -> [u8; 32] {
        self.proof_digest
    }

    pub fn verified_at_cycle(&self) -> u64 {
        self.verified_at_cycle
    }

    pub fn trusted_state_mutated(&self) -> bool {
        self.trusted_state_mutated
    }

    pub fn quarantine_construction_authorized(&self) -> bool {
        self.quarantine_construction_authorized
    }

    pub fn writable_hydration_authorized(&self) -> bool {
        self.writable_hydration_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn verify_internal(&self) -> Result<(), RestartTrustContextCheckpointError> {
        self.statement.validate_shape()?;
        if digest_trust_context_checkpoint_statement(&self.statement)? != self.statement_digest {
            return Err(RestartTrustContextCheckpointError::StatementDigestMismatch);
        }
        if self.trusted_state_mutated
            || self.quarantine_construction_authorized
            || self.writable_hydration_authorized
            || self.activation_authorized
        {
            return Err(RestartTrustContextCheckpointError::UnexpectedAuthority);
        }
        Ok(())
    }
}

pub fn verify_restart_trust_context_checkpoint(
    evidence: &RestartTrustContextCheckpointEvidenceV1,
    context: &RestartTrustContextHandleV1,
    observed_at_cycle: u64,
    verifier: &dyn RestartTrustContextCheckpointVerifierV1,
) -> Result<VerifiedRestartTrustContextCheckpointV1, RestartTrustContextCheckpointError> {
    context
        .verify()
        .map_err(RestartTrustContextCheckpointError::ContextRejected)?;
    evidence.statement.validate_shape()?;
    if evidence.statement.deployment_id != context.deployment_id()
        || evidence.statement.trust_domain_id != context.trust_domain_id()
        || evidence.statement.context_digest != context.digest()
        || evidence.statement.context_committed_at_cycle != context.committed_at_cycle()
        || evidence.statement.anchor_sequence != context.anchor_sequence()
        || evidence.statement.anchor_capture_cycle != context.anchor_capture_cycle()
        || evidence.statement.verifier_trust_snapshot_sequence
            != context.verifier_trust_snapshot_sequence()
    {
        return Err(RestartTrustContextCheckpointError::ContextBindingMismatch);
    }
    if observed_at_cycle < evidence.statement.issued_at_cycle {
        return Err(RestartTrustContextCheckpointError::NotYetValid {
            observed_at_cycle,
            issued_at_cycle: evidence.statement.issued_at_cycle,
        });
    }
    if observed_at_cycle >= evidence.statement.expires_at_cycle {
        return Err(RestartTrustContextCheckpointError::Expired {
            observed_at_cycle,
            expires_at_cycle: evidence.statement.expires_at_cycle,
        });
    }

    let statement_digest = digest_trust_context_checkpoint_statement(&evidence.statement)?;
    let accepted = verifier
        .verify_trust_context_checkpoint(
            evidence.statement.evidence_kind,
            &evidence.statement.authority_id,
            statement_digest.as_bytes(),
            &evidence.proof,
        )
        .map_err(RestartTrustContextCheckpointError::VerificationProvider)?;
    if !accepted {
        return Err(RestartTrustContextCheckpointError::ProofRejected);
    }

    let mut proof_hasher = blake3::Hasher::new();
    proof_hasher.update(b"symthaea-ekm-restart-trust-context-checkpoint-proof-v1");
    proof_hasher.update(&evidence.proof);
    let receipt = VerifiedRestartTrustContextCheckpointV1 {
        statement: evidence.statement.clone(),
        statement_digest,
        proof_digest: *proof_hasher.finalize().as_bytes(),
        verified_at_cycle: observed_at_cycle,
        trusted_state_mutated: false,
        quarantine_construction_authorized: false,
        writable_hydration_authorized: false,
        activation_authorized: false,
    };
    receipt.verify_internal()?;
    Ok(receipt)
}

pub fn digest_trust_context_checkpoint_statement(
    statement: &RestartTrustContextCheckpointStatementV1,
) -> Result<RestartTrustContextCheckpointDigestV1, RestartTrustContextCheckpointError> {
    statement.validate_shape()?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-trust-context-checkpoint-v1");
    hasher.update(&[1]);
    hasher.update(&statement.sequence.to_le_bytes());
    hash_bytes(&mut hasher, statement.deployment_id.as_bytes())?;
    hash_bytes(&mut hasher, statement.trust_domain_id.as_bytes())?;
    hasher.update(&statement.context_digest.as_bytes());
    hasher.update(&statement.context_committed_at_cycle.to_le_bytes());
    hasher.update(&statement.anchor_sequence.to_le_bytes());
    hasher.update(&statement.anchor_capture_cycle.to_le_bytes());
    hasher.update(&statement.verifier_trust_snapshot_sequence.to_le_bytes());
    match statement.previous_checkpoint_digest {
        Some(previous) => {
            hasher.update(&[1]);
            hasher.update(&previous.as_bytes());
        }
        None => hasher.update(&[0]),
    }
    hasher.update(&statement.issued_at_cycle.to_le_bytes());
    hasher.update(&statement.expires_at_cycle.to_le_bytes());
    hash_bytes(&mut hasher, statement.authority_id.as_bytes())?;
    hasher.update(&[statement.evidence_kind.tag()]);
    Ok(RestartTrustContextCheckpointDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartTrustContextCheckpointContinuityDispositionV1 {
    ForwardProgress,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartTrustContextCheckpointContinuityFailureV1 {
    DeploymentChanged,
    TrustDomainChanged,
    SequenceGap { expected: u64, actual: u64 },
    PreviousCheckpointMismatch,
    ContextDigestReplay,
    ContextCommitCycleDidNotAdvance { previous: u64, candidate: u64 },
    AnchorSequenceRollback { previous: u64, candidate: u64 },
    AnchorCaptureCycleRollback { previous: u64, candidate: u64 },
    VerifierTrustSnapshotRollback { previous: u64, candidate: u64 },
    VerificationCycleRollback { previous: u64, candidate: u64 },
    InvalidPreviousReceipt,
    InvalidCandidateReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartTrustContextCheckpointContinuityDecisionV1 {
    disposition: RestartTrustContextCheckpointContinuityDispositionV1,
    failures: Vec<RestartTrustContextCheckpointContinuityFailureV1>,
    further_review_eligible: bool,
    trusted_state_mutated: bool,
    quarantine_construction_authorized: bool,
    writable_hydration_authorized: bool,
    activation_authorized: bool,
}

impl RestartTrustContextCheckpointContinuityDecisionV1 {
    pub fn disposition(&self) -> RestartTrustContextCheckpointContinuityDispositionV1 {
        self.disposition
    }

    pub fn failures(&self) -> &[RestartTrustContextCheckpointContinuityFailureV1] {
        &self.failures
    }

    pub fn further_review_eligible(&self) -> bool {
        self.further_review_eligible
    }

    pub fn trusted_state_mutated(&self) -> bool {
        self.trusted_state_mutated
    }

    pub fn quarantine_construction_authorized(&self) -> bool {
        self.quarantine_construction_authorized
    }

    pub fn writable_hydration_authorized(&self) -> bool {
        self.writable_hydration_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }
}

pub struct RestartTrustContextCheckpointContinuityGateV1;

impl RestartTrustContextCheckpointContinuityGateV1 {
    pub fn evaluate(
        previous: &VerifiedRestartTrustContextCheckpointV1,
        candidate: &VerifiedRestartTrustContextCheckpointV1,
    ) -> RestartTrustContextCheckpointContinuityDecisionV1 {
        let mut failures = Vec::new();
        if previous.verify_internal().is_err() {
            failures.push(RestartTrustContextCheckpointContinuityFailureV1::InvalidPreviousReceipt);
        }
        if candidate.verify_internal().is_err() {
            failures.push(RestartTrustContextCheckpointContinuityFailureV1::InvalidCandidateReceipt);
        }

        let prior = previous.statement();
        let next = candidate.statement();
        if next.deployment_id != prior.deployment_id {
            failures.push(RestartTrustContextCheckpointContinuityFailureV1::DeploymentChanged);
        }
        if next.trust_domain_id != prior.trust_domain_id {
            failures.push(RestartTrustContextCheckpointContinuityFailureV1::TrustDomainChanged);
        }
        match prior.sequence.checked_add(1) {
            Some(expected) if next.sequence != expected => failures.push(
                RestartTrustContextCheckpointContinuityFailureV1::SequenceGap {
                    expected,
                    actual: next.sequence,
                },
            ),
            None => failures.push(
                RestartTrustContextCheckpointContinuityFailureV1::SequenceGap {
                    expected: u64::MAX,
                    actual: next.sequence,
                },
            ),
            _ => {}
        }
        if next.previous_checkpoint_digest != Some(previous.statement_digest) {
            failures.push(
                RestartTrustContextCheckpointContinuityFailureV1::PreviousCheckpointMismatch,
            );
        }
        if next.context_digest == prior.context_digest {
            failures.push(RestartTrustContextCheckpointContinuityFailureV1::ContextDigestReplay);
        }
        if next.context_committed_at_cycle <= prior.context_committed_at_cycle {
            failures.push(
                RestartTrustContextCheckpointContinuityFailureV1::ContextCommitCycleDidNotAdvance {
                    previous: prior.context_committed_at_cycle,
                    candidate: next.context_committed_at_cycle,
                },
            );
        }
        if next.anchor_sequence < prior.anchor_sequence {
            failures.push(
                RestartTrustContextCheckpointContinuityFailureV1::AnchorSequenceRollback {
                    previous: prior.anchor_sequence,
                    candidate: next.anchor_sequence,
                },
            );
        }
        if next.anchor_capture_cycle < prior.anchor_capture_cycle {
            failures.push(
                RestartTrustContextCheckpointContinuityFailureV1::AnchorCaptureCycleRollback {
                    previous: prior.anchor_capture_cycle,
                    candidate: next.anchor_capture_cycle,
                },
            );
        }
        if next.verifier_trust_snapshot_sequence < prior.verifier_trust_snapshot_sequence {
            failures.push(
                RestartTrustContextCheckpointContinuityFailureV1::VerifierTrustSnapshotRollback {
                    previous: prior.verifier_trust_snapshot_sequence,
                    candidate: next.verifier_trust_snapshot_sequence,
                },
            );
        }
        if candidate.verified_at_cycle < previous.verified_at_cycle {
            failures.push(
                RestartTrustContextCheckpointContinuityFailureV1::VerificationCycleRollback {
                    previous: previous.verified_at_cycle,
                    candidate: candidate.verified_at_cycle,
                },
            );
        }

        let disposition = if failures.is_empty() {
            RestartTrustContextCheckpointContinuityDispositionV1::ForwardProgress
        } else {
            RestartTrustContextCheckpointContinuityDispositionV1::Rejected
        };
        RestartTrustContextCheckpointContinuityDecisionV1 {
            disposition,
            failures,
            further_review_eligible: matches!(
                disposition,
                RestartTrustContextCheckpointContinuityDispositionV1::ForwardProgress
            ),
            trusted_state_mutated: false,
            quarantine_construction_authorized: false,
            writable_hydration_authorized: false,
            activation_authorized: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartTrustContextCheckpointField {
    DeploymentId,
    TrustDomainId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartTrustContextCheckpointError {
    ContextRejected(RestartQuarantineFacadeError),
    InvalidSequence,
    InvalidPredecessorShape,
    InvalidEmbeddedSequence,
    InvalidIdentifier(RestartTrustContextCheckpointField),
    InvalidAuthorityId,
    ContextCommitPredatesAnchor {
        context_committed_at_cycle: u64,
        anchor_capture_cycle: u64,
    },
    IssuePredatesContextCommit {
        issued_at_cycle: u64,
        context_committed_at_cycle: u64,
    },
    InvalidValidityWindow,
    EmptyProof,
    ProofTooLarge { actual: usize, maximum: usize },
    ContextBindingMismatch,
    NotYetValid { observed_at_cycle: u64, issued_at_cycle: u64 },
    Expired { observed_at_cycle: u64, expires_at_cycle: u64 },
    VerificationProvider(String),
    ProofRejected,
    StatementDigestMismatch,
    UnexpectedAuthority,
    LengthOverflow,
}

impl fmt::Display for RestartTrustContextCheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "restart trust-context checkpoint rejected: {self:?}")
    }
}

impl Error for RestartTrustContextCheckpointError {}

fn validate_identifier(
    value: &str,
    field: RestartTrustContextCheckpointField,
) -> Result<(), RestartTrustContextCheckpointError> {
    if value.trim().is_empty() || value != value.trim() || value.len() > 256 {
        return Err(RestartTrustContextCheckpointError::InvalidIdentifier(field));
    }
    Ok(())
}

fn hash_bytes(
    hasher: &mut blake3::Hasher,
    bytes: &[u8],
) -> Result<(), RestartTrustContextCheckpointError> {
    let len = u64::try_from(bytes.len())
        .map_err(|_| RestartTrustContextCheckpointError::LengthOverflow)?;
    hasher.update(&len.to_le_bytes());
    hasher.update(bytes);
    Ok(())
}

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn statement(
        sequence: u64,
        context_byte: u8,
        committed_at_cycle: u64,
        anchor_sequence: u64,
        anchor_capture_cycle: u64,
        verifier_sequence: u64,
        previous: Option<RestartTrustContextCheckpointDigestV1>,
    ) -> RestartTrustContextCheckpointStatementV1 {
        RestartTrustContextCheckpointStatementV1 {
            version: RestartTrustContextCheckpointVersion::V1,
            sequence,
            deployment_id: "deployment-a".into(),
            trust_domain_id: "restart-trust".into(),
            context_digest: RestartTrustContextDigestV1([context_byte; 32]),
            context_committed_at_cycle: committed_at_cycle,
            anchor_sequence,
            anchor_capture_cycle,
            verifier_trust_snapshot_sequence: verifier_sequence,
            previous_checkpoint_digest: previous,
            issued_at_cycle: committed_at_cycle,
            expires_at_cycle: committed_at_cycle + 100,
            authority_id: "checkpoint-authority".into(),
            evidence_kind: RestartTrustContextCheckpointEvidenceKindV1::ProtectedStorage,
        }
    }

    fn verified(
        statement: RestartTrustContextCheckpointStatementV1,
        verified_at_cycle: u64,
    ) -> VerifiedRestartTrustContextCheckpointV1 {
        let statement_digest = digest_trust_context_checkpoint_statement(&statement).unwrap();
        VerifiedRestartTrustContextCheckpointV1 {
            statement,
            statement_digest,
            proof_digest: [9; 32],
            verified_at_cycle,
            trusted_state_mutated: false,
            quarantine_construction_authorized: false,
            writable_hydration_authorized: false,
            activation_authorized: false,
        }
    }

    #[test]
    fn statement_digest_is_deterministic() {
        let first = statement(1, 1, 10, 3, 8, 4, None);
        let second = first.clone();
        assert_eq!(
            digest_trust_context_checkpoint_statement(&first).unwrap(),
            digest_trust_context_checkpoint_statement(&second).unwrap()
        );
    }

    #[test]
    fn verified_receipt_never_grants_restore_authority() {
        let receipt = verified(statement(1, 1, 10, 3, 8, 4, None), 10);
        receipt.verify_internal().unwrap();
        assert!(!receipt.trusted_state_mutated());
        assert!(!receipt.quarantine_construction_authorized());
        assert!(!receipt.writable_hydration_authorized());
        assert!(!receipt.activation_authorized());
    }

    #[test]
    fn continuity_accepts_strict_forward_progress() {
        let first = verified(statement(1, 1, 10, 3, 8, 4, None), 10);
        let second_statement = statement(
            2,
            2,
            12,
            4,
            11,
            5,
            Some(first.statement_digest()),
        );
        let second = verified(second_statement, 12);
        let decision = RestartTrustContextCheckpointContinuityGateV1::evaluate(&first, &second);
        assert_eq!(
            decision.disposition(),
            RestartTrustContextCheckpointContinuityDispositionV1::ForwardProgress
        );
        assert!(decision.further_review_eligible());
        assert!(!decision.trusted_state_mutated());
    }

    #[test]
    fn continuity_rejects_context_replay_and_trust_rollback() {
        let first = verified(statement(1, 1, 10, 3, 8, 4, None), 10);
        let second_statement = statement(
            2,
            1,
            11,
            2,
            7,
            3,
            Some(first.statement_digest()),
        );
        let second = verified(second_statement, 11);
        let decision = RestartTrustContextCheckpointContinuityGateV1::evaluate(&first, &second);
        assert_eq!(
            decision.disposition(),
            RestartTrustContextCheckpointContinuityDispositionV1::Rejected
        );
        assert!(decision
            .failures()
            .contains(&RestartTrustContextCheckpointContinuityFailureV1::ContextDigestReplay));
        assert!(decision.failures().iter().any(|failure| matches!(
            failure,
            RestartTrustContextCheckpointContinuityFailureV1::AnchorSequenceRollback { .. }
        )));
        assert!(decision.failures().iter().any(|failure| matches!(
            failure,
            RestartTrustContextCheckpointContinuityFailureV1::VerifierTrustSnapshotRollback { .. }
        )));
    }
}
