// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Externally protected checkpoint for restart mutation-time evidence seals.
//!
//! EKM-056 retains the complete EKM-028 evidence census for every persisted
//! belief mutation. EKM-057/058 can serialize and cross-check that sidecar, but
//! an untrusted sidecar can still omit historical non-basis evidence and recompute
//! sidecar-local digests. This module binds the *original* EKM-056 capsule digest
//! into an externally verifiable statement alongside the exact protected restart
//! trust checkpoint, anchor, and validation receipt.
//!
//! A verified receipt proves only that the configured external provider accepted
//! this exact combined statement. It does not construct a seal capsule, hydrate
//! writable state, advance trusted state, or authorize activation.

use super::belief_mutation_seal_persistence::BeliefMutationEvidenceSealCapsuleV1;
use super::epistemic_restart_anchor::{
    digest_restart_anchor_statement, RestartAnchorEvidenceError, VerifiedRestartAnchorEvidenceV1,
};
use super::epistemic_restart_trust_checkpoint::{
    RestartTrustContextCheckpointEvidenceKindV1, RestartTrustContextCheckpointError,
    VerifiedRestartTrustContextCheckpointV1, MAX_RESTART_TRUST_CHECKPOINT_AUTHORITY_ID_BYTES,
    MAX_RESTART_TRUST_CHECKPOINT_PROOF_BYTES,
};
use super::epistemic_restart_validation_receipt::EpistemicRestartValidationReceiptV1;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartMutationSealCheckpointVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RestartMutationSealCheckpointDigestV1([u8; 32]);

impl RestartMutationSealCheckpointDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

/// Canonical child statement binding the protected restart lineage to the
/// original EKM-056 mutation-evidence-seal capsule digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartMutationSealCheckpointStatementV1 {
    version: RestartMutationSealCheckpointVersion,
    sequence: u64,
    deployment_id: String,
    trust_domain_id: String,
    parent_trust_checkpoint_sequence: u64,
    parent_trust_checkpoint_digest: [u8; 32],
    trust_context_digest: [u8; 32],
    anchor_sequence: u64,
    anchor_statement_digest: [u8; 32],
    restart_captured_at_cycle: u64,
    restart_validation_receipt_digest: [u8; 32],
    restart_v2_digest: [u8; 32],
    mutation_seal_capture_cycle: u64,
    linked_mutation_capture_cycle: u64,
    mutation_count: u64,
    mutation_seal_capsule_digest: [u8; 32],
    previous_checkpoint_digest: Option<RestartMutationSealCheckpointDigestV1>,
    issued_at_cycle: u64,
    expires_at_cycle: u64,
    authority_id: String,
    evidence_kind: RestartTrustContextCheckpointEvidenceKindV1,
}

impl RestartMutationSealCheckpointStatementV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sequence: u64,
        parent: &VerifiedRestartTrustContextCheckpointV1,
        anchor: &VerifiedRestartAnchorEvidenceV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        seal_capsule: &BeliefMutationEvidenceSealCapsuleV1,
        previous_checkpoint_digest: Option<RestartMutationSealCheckpointDigestV1>,
        issued_at_cycle: u64,
        expires_at_cycle: u64,
        authority_id: impl Into<String>,
        evidence_kind: RestartTrustContextCheckpointEvidenceKindV1,
    ) -> Result<Self, RestartMutationSealCheckpointError> {
        validate_source_bindings(parent, anchor, restart_receipt, seal_capsule, issued_at_cycle)?;
        let statement = Self {
            version: RestartMutationSealCheckpointVersion::V1,
            sequence,
            deployment_id: parent.statement().deployment_id().to_string(),
            trust_domain_id: parent.statement().trust_domain_id().to_string(),
            parent_trust_checkpoint_sequence: parent.statement().sequence(),
            parent_trust_checkpoint_digest: parent.statement_digest().as_bytes(),
            trust_context_digest: parent.statement().context_digest().as_bytes(),
            anchor_sequence: anchor.statement().sequence(),
            anchor_statement_digest: anchor.statement_digest().as_bytes(),
            restart_captured_at_cycle: restart_receipt.captured_at_cycle(),
            restart_validation_receipt_digest: restart_receipt.receipt_digest().as_bytes(),
            restart_v2_digest: restart_receipt.claimed_v2_digest(),
            mutation_seal_capture_cycle: seal_capsule.captured_at_cycle(),
            linked_mutation_capture_cycle: seal_capsule.linked_mutation_capture_cycle(),
            mutation_count: u64::try_from(seal_capsule.records().len())
                .map_err(|_| RestartMutationSealCheckpointError::LengthOverflow)?,
            mutation_seal_capsule_digest: seal_capsule.capsule_digest().as_bytes(),
            previous_checkpoint_digest,
            issued_at_cycle,
            expires_at_cycle,
            authority_id: authority_id.into(),
            evidence_kind,
        };
        statement.validate_shape()?;
        Ok(statement)
    }

    pub fn version(&self) -> RestartMutationSealCheckpointVersion {
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

    pub fn parent_trust_checkpoint_sequence(&self) -> u64 {
        self.parent_trust_checkpoint_sequence
    }

    pub fn parent_trust_checkpoint_digest(&self) -> [u8; 32] {
        self.parent_trust_checkpoint_digest
    }

    pub fn trust_context_digest(&self) -> [u8; 32] {
        self.trust_context_digest
    }

    pub fn anchor_sequence(&self) -> u64 {
        self.anchor_sequence
    }

    pub fn anchor_statement_digest(&self) -> [u8; 32] {
        self.anchor_statement_digest
    }

    pub fn restart_captured_at_cycle(&self) -> u64 {
        self.restart_captured_at_cycle
    }

    pub fn restart_validation_receipt_digest(&self) -> [u8; 32] {
        self.restart_validation_receipt_digest
    }

    pub fn restart_v2_digest(&self) -> [u8; 32] {
        self.restart_v2_digest
    }

    pub fn mutation_seal_capture_cycle(&self) -> u64 {
        self.mutation_seal_capture_cycle
    }

    pub fn linked_mutation_capture_cycle(&self) -> u64 {
        self.linked_mutation_capture_cycle
    }

    pub fn mutation_count(&self) -> u64 {
        self.mutation_count
    }

    pub fn mutation_seal_capsule_digest(&self) -> [u8; 32] {
        self.mutation_seal_capsule_digest
    }

    pub fn previous_checkpoint_digest(&self) -> Option<RestartMutationSealCheckpointDigestV1> {
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

    fn validate_shape(&self) -> Result<(), RestartMutationSealCheckpointError> {
        if self.sequence == 0 {
            return Err(RestartMutationSealCheckpointError::InvalidSequence);
        }
        if (self.sequence == 1) != self.previous_checkpoint_digest.is_none() {
            return Err(RestartMutationSealCheckpointError::InvalidPredecessorShape);
        }
        if self.parent_trust_checkpoint_sequence == 0 || self.anchor_sequence == 0 {
            return Err(RestartMutationSealCheckpointError::InvalidParentSequence);
        }
        if self.linked_mutation_capture_cycle != self.restart_captured_at_cycle {
            return Err(RestartMutationSealCheckpointError::MutationEpochMismatch {
                restart_capture_cycle: self.restart_captured_at_cycle,
                mutation_capture_cycle: self.linked_mutation_capture_cycle,
            });
        }
        if self.mutation_seal_capture_cycle < self.linked_mutation_capture_cycle {
            return Err(RestartMutationSealCheckpointError::SealCapturePredatesMutationEpoch {
                seal_capture_cycle: self.mutation_seal_capture_cycle,
                mutation_capture_cycle: self.linked_mutation_capture_cycle,
            });
        }
        if self.issued_at_cycle < self.mutation_seal_capture_cycle {
            return Err(RestartMutationSealCheckpointError::IssuePredatesSealCapture {
                issued_at_cycle: self.issued_at_cycle,
                seal_capture_cycle: self.mutation_seal_capture_cycle,
            });
        }
        if self.issued_at_cycle >= self.expires_at_cycle {
            return Err(RestartMutationSealCheckpointError::InvalidValidityWindow);
        }
        validate_identifier(&self.deployment_id)?;
        validate_identifier(&self.trust_domain_id)?;
        if self.authority_id.trim().is_empty()
            || self.authority_id != self.authority_id.trim()
            || self.authority_id.len() > MAX_RESTART_TRUST_CHECKPOINT_AUTHORITY_ID_BYTES
        {
            return Err(RestartMutationSealCheckpointError::InvalidAuthorityId);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartMutationSealCheckpointEvidenceV1 {
    statement: RestartMutationSealCheckpointStatementV1,
    proof: Vec<u8>,
}

impl RestartMutationSealCheckpointEvidenceV1 {
    pub fn new(
        statement: RestartMutationSealCheckpointStatementV1,
        proof: Vec<u8>,
    ) -> Result<Self, RestartMutationSealCheckpointError> {
        statement.validate_shape()?;
        if proof.is_empty() {
            return Err(RestartMutationSealCheckpointError::EmptyProof);
        }
        if proof.len() > MAX_RESTART_TRUST_CHECKPOINT_PROOF_BYTES {
            return Err(RestartMutationSealCheckpointError::ProofTooLarge {
                actual: proof.len(),
                maximum: MAX_RESTART_TRUST_CHECKPOINT_PROOF_BYTES,
            });
        }
        Ok(Self { statement, proof })
    }

    pub fn statement(&self) -> &RestartMutationSealCheckpointStatementV1 {
        &self.statement
    }

    pub fn proof(&self) -> &[u8] {
        &self.proof
    }
}

/// Deployment seam for protected storage, signatures, hardware attestation,
/// transparency logs, or witness-quorum verification over the combined statement.
pub trait RestartMutationSealCheckpointVerifierV1 {
    fn verify_restart_mutation_seal_checkpoint(
        &self,
        evidence_kind: RestartTrustContextCheckpointEvidenceKindV1,
        authority_id: &str,
        statement_digest: [u8; 32],
        proof: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRestartMutationSealCheckpointV1 {
    statement: RestartMutationSealCheckpointStatementV1,
    statement_digest: RestartMutationSealCheckpointDigestV1,
    proof_digest: [u8; 32],
    verified_at_cycle: u64,
    seal_capsule_digest_protected: bool,
    trusted_state_mutated: bool,
    capsule_construction_authorized: bool,
    writable_hydration_authorized: bool,
    activation_authorized: bool,
}

impl VerifiedRestartMutationSealCheckpointV1 {
    pub fn statement(&self) -> &RestartMutationSealCheckpointStatementV1 {
        &self.statement
    }

    pub fn statement_digest(&self) -> RestartMutationSealCheckpointDigestV1 {
        self.statement_digest
    }

    pub fn proof_digest(&self) -> [u8; 32] {
        self.proof_digest
    }

    pub fn verified_at_cycle(&self) -> u64 {
        self.verified_at_cycle
    }

    pub fn seal_capsule_digest_protected(&self) -> bool {
        self.seal_capsule_digest_protected
    }

    pub fn trusted_state_mutated(&self) -> bool {
        self.trusted_state_mutated
    }

    pub fn capsule_construction_authorized(&self) -> bool {
        self.capsule_construction_authorized
    }

    pub fn writable_hydration_authorized(&self) -> bool {
        self.writable_hydration_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn verify_internal(&self) -> Result<(), RestartMutationSealCheckpointError> {
        self.statement.validate_shape()?;
        if digest_restart_mutation_seal_checkpoint_statement(&self.statement)?
            != self.statement_digest
        {
            return Err(RestartMutationSealCheckpointError::StatementDigestMismatch);
        }
        if !self.seal_capsule_digest_protected
            || self.trusted_state_mutated
            || self.capsule_construction_authorized
            || self.writable_hydration_authorized
            || self.activation_authorized
        {
            return Err(RestartMutationSealCheckpointError::UnexpectedAuthority);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_restart_mutation_seal_checkpoint(
    evidence: &RestartMutationSealCheckpointEvidenceV1,
    parent: &VerifiedRestartTrustContextCheckpointV1,
    anchor: &VerifiedRestartAnchorEvidenceV1,
    restart_receipt: &EpistemicRestartValidationReceiptV1,
    seal_capsule: &BeliefMutationEvidenceSealCapsuleV1,
    observed_at_cycle: u64,
    verifier: &dyn RestartMutationSealCheckpointVerifierV1,
) -> Result<VerifiedRestartMutationSealCheckpointV1, RestartMutationSealCheckpointError> {
    evidence.statement.validate_shape()?;
    validate_source_bindings(
        parent,
        anchor,
        restart_receipt,
        seal_capsule,
        evidence.statement.issued_at_cycle,
    )?;
    validate_statement_bindings(
        &evidence.statement,
        parent,
        anchor,
        restart_receipt,
        seal_capsule,
    )?;

    if observed_at_cycle < evidence.statement.issued_at_cycle {
        return Err(RestartMutationSealCheckpointError::NotYetValid {
            observed_at_cycle,
            issued_at_cycle: evidence.statement.issued_at_cycle,
        });
    }
    if observed_at_cycle >= evidence.statement.expires_at_cycle {
        return Err(RestartMutationSealCheckpointError::Expired {
            observed_at_cycle,
            expires_at_cycle: evidence.statement.expires_at_cycle,
        });
    }

    let statement_digest =
        digest_restart_mutation_seal_checkpoint_statement(&evidence.statement)?;
    let accepted = verifier
        .verify_restart_mutation_seal_checkpoint(
            evidence.statement.evidence_kind,
            &evidence.statement.authority_id,
            statement_digest.as_bytes(),
            &evidence.proof,
        )
        .map_err(RestartMutationSealCheckpointError::VerificationProvider)?;
    if !accepted {
        return Err(RestartMutationSealCheckpointError::ProofRejected);
    }

    let mut proof_hasher = blake3::Hasher::new();
    proof_hasher.update(b"symthaea-ekm-restart-mutation-seal-checkpoint-proof-v1");
    proof_hasher.update(&evidence.proof);
    let receipt = VerifiedRestartMutationSealCheckpointV1 {
        statement: evidence.statement.clone(),
        statement_digest,
        proof_digest: *proof_hasher.finalize().as_bytes(),
        verified_at_cycle: observed_at_cycle,
        seal_capsule_digest_protected: true,
        trusted_state_mutated: false,
        capsule_construction_authorized: false,
        writable_hydration_authorized: false,
        activation_authorized: false,
    };
    receipt.verify_internal()?;
    Ok(receipt)
}

pub fn digest_restart_mutation_seal_checkpoint_statement(
    statement: &RestartMutationSealCheckpointStatementV1,
) -> Result<RestartMutationSealCheckpointDigestV1, RestartMutationSealCheckpointError> {
    statement.validate_shape()?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-mutation-seal-checkpoint-v1");
    hasher.update(&[1]);
    hasher.update(&statement.sequence.to_le_bytes());
    hash_bytes(&mut hasher, statement.deployment_id.as_bytes())?;
    hash_bytes(&mut hasher, statement.trust_domain_id.as_bytes())?;
    hasher.update(&statement.parent_trust_checkpoint_sequence.to_le_bytes());
    hasher.update(&statement.parent_trust_checkpoint_digest);
    hasher.update(&statement.trust_context_digest);
    hasher.update(&statement.anchor_sequence.to_le_bytes());
    hasher.update(&statement.anchor_statement_digest);
    hasher.update(&statement.restart_captured_at_cycle.to_le_bytes());
    hasher.update(&statement.restart_validation_receipt_digest);
    hasher.update(&statement.restart_v2_digest);
    hasher.update(&statement.mutation_seal_capture_cycle.to_le_bytes());
    hasher.update(&statement.linked_mutation_capture_cycle.to_le_bytes());
    hasher.update(&statement.mutation_count.to_le_bytes());
    hasher.update(&statement.mutation_seal_capsule_digest);
    match statement.previous_checkpoint_digest {
        Some(previous) => {
            hasher.update(&[1]);
            hasher.update(&previous.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&statement.issued_at_cycle.to_le_bytes());
    hasher.update(&statement.expires_at_cycle.to_le_bytes());
    hash_bytes(&mut hasher, statement.authority_id.as_bytes())?;
    hasher.update(&[evidence_kind_tag(statement.evidence_kind)]);
    Ok(RestartMutationSealCheckpointDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn validate_source_bindings(
    parent: &VerifiedRestartTrustContextCheckpointV1,
    anchor: &VerifiedRestartAnchorEvidenceV1,
    restart_receipt: &EpistemicRestartValidationReceiptV1,
    seal_capsule: &BeliefMutationEvidenceSealCapsuleV1,
    issued_at_cycle: u64,
) -> Result<(), RestartMutationSealCheckpointError> {
    parent
        .verify_internal()
        .map_err(RestartMutationSealCheckpointError::ParentCheckpointRejected)?;
    let anchor_digest = digest_restart_anchor_statement(anchor.statement())
        .map_err(RestartMutationSealCheckpointError::AnchorRejected)?;
    if anchor_digest != anchor.statement_digest() {
        return Err(RestartMutationSealCheckpointError::AnchorStatementDigestMismatch);
    }
    if anchor.quarantine_construction_authorized() || anchor.activation_authorized() {
        return Err(RestartMutationSealCheckpointError::UnexpectedSourceAuthority);
    }
    if restart_receipt.authority().quarantine_construction_authorized()
        || restart_receipt.authority().activation_authorized()
    {
        return Err(RestartMutationSealCheckpointError::UnexpectedSourceAuthority);
    }

    if parent.statement().anchor_sequence() != anchor.statement().sequence()
        || parent.statement().anchor_capture_cycle() != anchor.statement().captured_at_cycle()
    {
        return Err(RestartMutationSealCheckpointError::ParentAnchorMismatch);
    }
    if anchor.statement().receipt_digest() != restart_receipt.receipt_digest()
        || anchor.statement().captured_at_cycle() != restart_receipt.captured_at_cycle()
    {
        return Err(RestartMutationSealCheckpointError::AnchorReceiptMismatch);
    }
    if seal_capsule.linked_mutation_capture_cycle() != restart_receipt.captured_at_cycle() {
        return Err(RestartMutationSealCheckpointError::MutationEpochMismatch {
            restart_capture_cycle: restart_receipt.captured_at_cycle(),
            mutation_capture_cycle: seal_capsule.linked_mutation_capture_cycle(),
        });
    }
    if seal_capsule.captured_at_cycle() < seal_capsule.linked_mutation_capture_cycle() {
        return Err(RestartMutationSealCheckpointError::SealCapturePredatesMutationEpoch {
            seal_capture_cycle: seal_capsule.captured_at_cycle(),
            mutation_capture_cycle: seal_capsule.linked_mutation_capture_cycle(),
        });
    }
    if issued_at_cycle < parent.verified_at_cycle()
        || issued_at_cycle < anchor.verified_at_cycle()
        || issued_at_cycle < seal_capsule.captured_at_cycle()
    {
        return Err(RestartMutationSealCheckpointError::IssuePredatesProtectedInputs);
    }
    if issued_at_cycle >= parent.statement().expires_at_cycle()
        || issued_at_cycle >= anchor.statement().expires_at_cycle()
    {
        return Err(RestartMutationSealCheckpointError::ParentEvidenceExpiredAtIssue);
    }
    Ok(())
}

fn validate_statement_bindings(
    statement: &RestartMutationSealCheckpointStatementV1,
    parent: &VerifiedRestartTrustContextCheckpointV1,
    anchor: &VerifiedRestartAnchorEvidenceV1,
    restart_receipt: &EpistemicRestartValidationReceiptV1,
    seal_capsule: &BeliefMutationEvidenceSealCapsuleV1,
) -> Result<(), RestartMutationSealCheckpointError> {
    if statement.deployment_id != parent.statement().deployment_id()
        || statement.trust_domain_id != parent.statement().trust_domain_id()
        || statement.parent_trust_checkpoint_sequence != parent.statement().sequence()
        || statement.parent_trust_checkpoint_digest != parent.statement_digest().as_bytes()
        || statement.trust_context_digest != parent.statement().context_digest().as_bytes()
        || statement.anchor_sequence != anchor.statement().sequence()
        || statement.anchor_statement_digest != anchor.statement_digest().as_bytes()
        || statement.restart_captured_at_cycle != restart_receipt.captured_at_cycle()
        || statement.restart_validation_receipt_digest != restart_receipt.receipt_digest().as_bytes()
        || statement.restart_v2_digest != restart_receipt.claimed_v2_digest()
        || statement.mutation_seal_capture_cycle != seal_capsule.captured_at_cycle()
        || statement.linked_mutation_capture_cycle != seal_capsule.linked_mutation_capture_cycle()
        || statement.mutation_count
            != u64::try_from(seal_capsule.records().len())
                .map_err(|_| RestartMutationSealCheckpointError::LengthOverflow)?
        || statement.mutation_seal_capsule_digest != seal_capsule.capsule_digest().as_bytes()
    {
        return Err(RestartMutationSealCheckpointError::StatementBindingMismatch);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartMutationSealCheckpointContinuityDispositionV1 {
    ForwardProgress,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartMutationSealCheckpointContinuityFailureV1 {
    DeploymentChanged,
    TrustDomainChanged,
    SequenceGap { expected: u64, actual: u64 },
    PreviousCheckpointMismatch,
    ParentTrustCheckpointRollback { previous: u64, candidate: u64 },
    AnchorSequenceRollback { previous: u64, candidate: u64 },
    RestartCaptureDidNotAdvance { previous: u64, candidate: u64 },
    SealCapsuleReplay,
    VerificationCycleRollback { previous: u64, candidate: u64 },
    InvalidPreviousReceipt,
    InvalidCandidateReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartMutationSealCheckpointContinuityDecisionV1 {
    disposition: RestartMutationSealCheckpointContinuityDispositionV1,
    failures: Vec<RestartMutationSealCheckpointContinuityFailureV1>,
    further_review_eligible: bool,
    trusted_state_mutated: bool,
    writable_hydration_authorized: bool,
    activation_authorized: bool,
}

impl RestartMutationSealCheckpointContinuityDecisionV1 {
    pub fn disposition(&self) -> RestartMutationSealCheckpointContinuityDispositionV1 {
        self.disposition
    }

    pub fn failures(&self) -> &[RestartMutationSealCheckpointContinuityFailureV1] {
        &self.failures
    }

    pub fn further_review_eligible(&self) -> bool {
        self.further_review_eligible
    }

    pub fn trusted_state_mutated(&self) -> bool {
        self.trusted_state_mutated
    }

    pub fn writable_hydration_authorized(&self) -> bool {
        self.writable_hydration_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }
}

pub struct RestartMutationSealCheckpointContinuityGateV1;

impl RestartMutationSealCheckpointContinuityGateV1 {
    pub fn evaluate(
        previous: &VerifiedRestartMutationSealCheckpointV1,
        candidate: &VerifiedRestartMutationSealCheckpointV1,
    ) -> RestartMutationSealCheckpointContinuityDecisionV1 {
        let mut failures = Vec::new();
        if previous.verify_internal().is_err() {
            failures.push(RestartMutationSealCheckpointContinuityFailureV1::InvalidPreviousReceipt);
        }
        if candidate.verify_internal().is_err() {
            failures.push(RestartMutationSealCheckpointContinuityFailureV1::InvalidCandidateReceipt);
        }
        let prior = previous.statement();
        let next = candidate.statement();
        if next.deployment_id != prior.deployment_id {
            failures.push(RestartMutationSealCheckpointContinuityFailureV1::DeploymentChanged);
        }
        if next.trust_domain_id != prior.trust_domain_id {
            failures.push(RestartMutationSealCheckpointContinuityFailureV1::TrustDomainChanged);
        }
        match prior.sequence.checked_add(1) {
            Some(expected) if next.sequence != expected => failures.push(
                RestartMutationSealCheckpointContinuityFailureV1::SequenceGap {
                    expected,
                    actual: next.sequence,
                },
            ),
            None => failures.push(RestartMutationSealCheckpointContinuityFailureV1::SequenceGap {
                expected: u64::MAX,
                actual: next.sequence,
            }),
            _ => {}
        }
        if next.previous_checkpoint_digest != Some(previous.statement_digest) {
            failures.push(
                RestartMutationSealCheckpointContinuityFailureV1::PreviousCheckpointMismatch,
            );
        }
        if next.parent_trust_checkpoint_sequence < prior.parent_trust_checkpoint_sequence {
            failures.push(
                RestartMutationSealCheckpointContinuityFailureV1::ParentTrustCheckpointRollback {
                    previous: prior.parent_trust_checkpoint_sequence,
                    candidate: next.parent_trust_checkpoint_sequence,
                },
            );
        }
        if next.anchor_sequence < prior.anchor_sequence {
            failures.push(
                RestartMutationSealCheckpointContinuityFailureV1::AnchorSequenceRollback {
                    previous: prior.anchor_sequence,
                    candidate: next.anchor_sequence,
                },
            );
        }
        if next.restart_captured_at_cycle <= prior.restart_captured_at_cycle {
            failures.push(
                RestartMutationSealCheckpointContinuityFailureV1::RestartCaptureDidNotAdvance {
                    previous: prior.restart_captured_at_cycle,
                    candidate: next.restart_captured_at_cycle,
                },
            );
        }
        if next.mutation_seal_capsule_digest == prior.mutation_seal_capsule_digest {
            failures.push(RestartMutationSealCheckpointContinuityFailureV1::SealCapsuleReplay);
        }
        if candidate.verified_at_cycle < previous.verified_at_cycle {
            failures.push(
                RestartMutationSealCheckpointContinuityFailureV1::VerificationCycleRollback {
                    previous: previous.verified_at_cycle,
                    candidate: candidate.verified_at_cycle,
                },
            );
        }

        let disposition = if failures.is_empty() {
            RestartMutationSealCheckpointContinuityDispositionV1::ForwardProgress
        } else {
            RestartMutationSealCheckpointContinuityDispositionV1::Rejected
        };
        RestartMutationSealCheckpointContinuityDecisionV1 {
            disposition,
            failures,
            further_review_eligible: matches!(
                disposition,
                RestartMutationSealCheckpointContinuityDispositionV1::ForwardProgress
            ),
            trusted_state_mutated: false,
            writable_hydration_authorized: false,
            activation_authorized: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartMutationSealCheckpointError {
    ParentCheckpointRejected(RestartTrustContextCheckpointError),
    AnchorRejected(RestartAnchorEvidenceError),
    InvalidSequence,
    InvalidPredecessorShape,
    InvalidParentSequence,
    InvalidAuthorityId,
    InvalidIdentifier,
    InvalidValidityWindow,
    MutationEpochMismatch { restart_capture_cycle: u64, mutation_capture_cycle: u64 },
    SealCapturePredatesMutationEpoch { seal_capture_cycle: u64, mutation_capture_cycle: u64 },
    IssuePredatesSealCapture { issued_at_cycle: u64, seal_capture_cycle: u64 },
    IssuePredatesProtectedInputs,
    ParentEvidenceExpiredAtIssue,
    AnchorStatementDigestMismatch,
    UnexpectedSourceAuthority,
    ParentAnchorMismatch,
    AnchorReceiptMismatch,
    StatementBindingMismatch,
    EmptyProof,
    ProofTooLarge { actual: usize, maximum: usize },
    NotYetValid { observed_at_cycle: u64, issued_at_cycle: u64 },
    Expired { observed_at_cycle: u64, expires_at_cycle: u64 },
    VerificationProvider(String),
    ProofRejected,
    StatementDigestMismatch,
    UnexpectedAuthority,
    LengthOverflow,
}

impl fmt::Display for RestartMutationSealCheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "restart mutation-seal checkpoint rejected: {self:?}")
    }
}

impl Error for RestartMutationSealCheckpointError {}

fn evidence_kind_tag(kind: RestartTrustContextCheckpointEvidenceKindV1) -> u8 {
    match kind {
        RestartTrustContextCheckpointEvidenceKindV1::ProtectedStorage => 1,
        RestartTrustContextCheckpointEvidenceKindV1::SignedCheckpoint => 2,
        RestartTrustContextCheckpointEvidenceKindV1::HardwareAttestation => 3,
        RestartTrustContextCheckpointEvidenceKindV1::TransparencyCheckpoint => 4,
        RestartTrustContextCheckpointEvidenceKindV1::WitnessQuorum => 5,
    }
}

fn validate_identifier(value: &str) -> Result<(), RestartMutationSealCheckpointError> {
    if value.trim().is_empty() || value != value.trim() || value.len() > 256 {
        return Err(RestartMutationSealCheckpointError::InvalidIdentifier);
    }
    Ok(())
}

fn hash_bytes(
    hasher: &mut blake3::Hasher,
    bytes: &[u8],
) -> Result<(), RestartMutationSealCheckpointError> {
    let len = u64::try_from(bytes.len())
        .map_err(|_| RestartMutationSealCheckpointError::LengthOverflow)?;
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
        restart_cycle: u64,
        seal_byte: u8,
        previous: Option<RestartMutationSealCheckpointDigestV1>,
    ) -> RestartMutationSealCheckpointStatementV1 {
        RestartMutationSealCheckpointStatementV1 {
            version: RestartMutationSealCheckpointVersion::V1,
            sequence,
            deployment_id: "deployment-a".into(),
            trust_domain_id: "restart-trust".into(),
            parent_trust_checkpoint_sequence: sequence,
            parent_trust_checkpoint_digest: [10 + sequence as u8; 32],
            trust_context_digest: [20 + sequence as u8; 32],
            anchor_sequence: sequence,
            anchor_statement_digest: [30 + sequence as u8; 32],
            restart_captured_at_cycle: restart_cycle,
            restart_validation_receipt_digest: [40 + sequence as u8; 32],
            restart_v2_digest: [50 + sequence as u8; 32],
            mutation_seal_capture_cycle: restart_cycle,
            linked_mutation_capture_cycle: restart_cycle,
            mutation_count: sequence,
            mutation_seal_capsule_digest: [seal_byte; 32],
            previous_checkpoint_digest: previous,
            issued_at_cycle: restart_cycle,
            expires_at_cycle: restart_cycle + 100,
            authority_id: "seal-checkpoint-authority".into(),
            evidence_kind: RestartTrustContextCheckpointEvidenceKindV1::ProtectedStorage,
        }
    }

    fn verified(
        statement: RestartMutationSealCheckpointStatementV1,
        verified_at_cycle: u64,
    ) -> VerifiedRestartMutationSealCheckpointV1 {
        let statement_digest = digest_restart_mutation_seal_checkpoint_statement(&statement)
            .unwrap();
        VerifiedRestartMutationSealCheckpointV1 {
            statement,
            statement_digest,
            proof_digest: [7; 32],
            verified_at_cycle,
            seal_capsule_digest_protected: true,
            trusted_state_mutated: false,
            capsule_construction_authorized: false,
            writable_hydration_authorized: false,
            activation_authorized: false,
        }
    }

    #[test]
    fn statement_digest_is_deterministic() {
        let first = statement(1, 10, 1, None);
        let second = first.clone();
        assert_eq!(
            digest_restart_mutation_seal_checkpoint_statement(&first).unwrap(),
            digest_restart_mutation_seal_checkpoint_statement(&second).unwrap()
        );
    }

    #[test]
    fn verified_receipt_protects_digest_but_grants_no_restore_authority() {
        let receipt = verified(statement(1, 10, 1, None), 10);
        receipt.verify_internal().unwrap();
        assert!(receipt.seal_capsule_digest_protected());
        assert!(!receipt.trusted_state_mutated());
        assert!(!receipt.capsule_construction_authorized());
        assert!(!receipt.writable_hydration_authorized());
        assert!(!receipt.activation_authorized());
    }

    #[test]
    fn continuity_accepts_strict_forward_progress() {
        let first = verified(statement(1, 10, 1, None), 10);
        let second_statement = statement(2, 11, 2, Some(first.statement_digest()));
        let second = verified(second_statement, 11);
        let decision = RestartMutationSealCheckpointContinuityGateV1::evaluate(&first, &second);
        assert_eq!(
            decision.disposition(),
            RestartMutationSealCheckpointContinuityDispositionV1::ForwardProgress
        );
        assert!(decision.further_review_eligible());
        assert!(!decision.writable_hydration_authorized());
        assert!(!decision.activation_authorized());
    }

    #[test]
    fn continuity_rejects_restart_and_seal_replay() {
        let first = verified(statement(1, 10, 1, None), 10);
        let mut second_statement = statement(2, 10, 1, Some(first.statement_digest()));
        second_statement.parent_trust_checkpoint_sequence = 0;
        let second_digest = digest_restart_mutation_seal_checkpoint_statement(&second_statement);
        assert!(second_digest.is_err());
    }
}
