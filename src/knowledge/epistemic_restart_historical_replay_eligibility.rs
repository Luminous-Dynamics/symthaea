// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only eligibility gate for later historical restart reconstruction review.
//!
//! EKM-060 proves that one supplied mutation-evidence sidecar reproduces the
//! externally protected EKM-056 source capsule. EKM-061 separately proves, under
//! an explicit current-head provider contract and validity window, that the exact
//! EKM-059 checkpoint is current. This module requires both receipts to bind the
//! same checkpoint before declaring the restart eligible for *isolated historical
//! projection review*.
//!
//! Eligibility is not replay authority. No historical ledger is constructed here,
//! no belief mutation is executed, no writable state is hydrated, and no live or
//! trusted state is changed.

use super::belief_mutation_seal_wire::BeliefMutationSealWireSnapshotV1;
use super::epistemic_restart_mutation_seal_admission::{
    ProtectedMutationSealAdmissionError, ProtectedMutationSealAdmissionV1,
};
use super::epistemic_restart_mutation_seal_checkpoint::{
    RestartMutationSealCheckpointError, VerifiedRestartMutationSealCheckpointV1,
};
use super::epistemic_restart_mutation_seal_currentness::{
    RestartMutationSealCurrentnessError, VerifiedRestartMutationSealCurrentnessV1,
};
use super::epistemic_restart_validation_receipt::EpistemicRestartValidationReceiptV1;
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoricalReplayEligibilityVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HistoricalReplayEligibilityDigestV1([u8; 32]);

impl HistoricalReplayEligibilityDigestV1 {
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

/// Immutable review receipt proving that protected sidecar equivalence and
/// current-head evidence refer to the exact same EKM-059 checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricalReplayEligibilityReceiptV1 {
    version: HistoricalReplayEligibilityVersion,
    restart_capture_cycle: u64,
    seal_capture_cycle: u64,
    mutation_count: usize,
    protected_checkpoint_digest: [u8; 32],
    protected_seal_capsule_digest: [u8; 32],
    protected_admission_digest: [u8; 32],
    currentness_statement_digest: [u8; 32],
    currentness_proof_digest: [u8; 32],
    currentness_attested_at_cycle: u64,
    currentness_expires_at_cycle: u64,
    reviewed_at_cycle: u64,
    protected_source_equivalence_verified: bool,
    checkpoint_currentness_verified: bool,
    isolated_historical_projection_review_eligible: bool,
    historical_replay_authorized: bool,
    capsule_construction_authorized: bool,
    writable_hydration_authorized: bool,
    activation_authorized: bool,
    receipt_digest: HistoricalReplayEligibilityDigestV1,
}

impl HistoricalReplayEligibilityReceiptV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        admission: &ProtectedMutationSealAdmissionV1,
        currentness: &VerifiedRestartMutationSealCurrentnessV1,
        reviewed_at_cycle: u64,
    ) -> Result<Self, HistoricalReplayEligibilityError> {
        checkpoint
            .verify_internal()
            .map_err(HistoricalReplayEligibilityError::CheckpointRejected)?;
        admission
            .verify_against(
                restart,
                seals,
                restart_receipt,
                checkpoint,
                admission.observed_at_cycle(),
            )
            .map_err(HistoricalReplayEligibilityError::AdmissionRejected)?;
        currentness
            .verify_internal()
            .map_err(HistoricalReplayEligibilityError::CurrentnessRejected)?;

        if reviewed_at_cycle < admission.observed_at_cycle() {
            return Err(HistoricalReplayEligibilityError::ReviewPredatesAdmission {
                reviewed_at_cycle,
                admission_observed_at_cycle: admission.observed_at_cycle(),
            });
        }
        if reviewed_at_cycle < currentness.verified_at_cycle() {
            return Err(HistoricalReplayEligibilityError::ReviewPredatesCurrentnessVerification {
                reviewed_at_cycle,
                currentness_verified_at_cycle: currentness.verified_at_cycle(),
            });
        }
        if reviewed_at_cycle >= currentness.statement().expires_at_cycle() {
            return Err(HistoricalReplayEligibilityError::CurrentnessExpired {
                reviewed_at_cycle,
                expires_at_cycle: currentness.statement().expires_at_cycle(),
            });
        }

        if !admission.cross_component_consistent()
            || !admission.protected_source_capsule_equivalent()
            || !admission.historical_census_completeness_protected()
            || admission.protected_checkpoint_currentness_independently_proven()
            || admission.capsule_construction_authorized()
            || admission.writable_hydration_authorized()
            || admission.activation_authorized()
        {
            return Err(HistoricalReplayEligibilityError::UnexpectedAdmissionClaims);
        }
        if !currentness.current_head_proven()
            || currentness.trusted_state_mutated()
            || currentness.historical_replay_authorized()
            || currentness.writable_hydration_authorized()
            || currentness.activation_authorized()
        {
            return Err(HistoricalReplayEligibilityError::UnexpectedCurrentnessClaims);
        }
        if !checkpoint.seal_capsule_digest_protected()
            || checkpoint.trusted_state_mutated()
            || checkpoint.capsule_construction_authorized()
            || checkpoint.writable_hydration_authorized()
            || checkpoint.activation_authorized()
        {
            return Err(HistoricalReplayEligibilityError::UnexpectedCheckpointAuthority);
        }

        let checkpoint_digest = checkpoint.statement_digest().as_bytes();
        if admission.protected_checkpoint_digest() != checkpoint_digest
            || currentness.statement().checkpoint_statement_digest() != checkpoint_digest
        {
            return Err(HistoricalReplayEligibilityError::CheckpointDigestMismatch);
        }
        if admission.restart_capture_cycle() != checkpoint.statement().restart_captured_at_cycle()
            || currentness.statement().restart_capture_cycle()
                != checkpoint.statement().restart_captured_at_cycle()
        {
            return Err(HistoricalReplayEligibilityError::RestartEpochMismatch);
        }
        if admission.recomputed_seal_capsule_digest()
            != checkpoint.statement().mutation_seal_capsule_digest()
            || currentness.statement().mutation_seal_capsule_digest()
                != checkpoint.statement().mutation_seal_capsule_digest()
        {
            return Err(HistoricalReplayEligibilityError::SealCapsuleDigestMismatch);
        }
        if currentness.statement().checkpoint_sequence() != checkpoint.statement().sequence()
            || currentness.statement().checkpoint_verified_at_cycle()
                != checkpoint.verified_at_cycle()
            || currentness.statement().checkpoint_expires_at_cycle()
                != checkpoint.statement().expires_at_cycle()
        {
            return Err(HistoricalReplayEligibilityError::CurrentnessCheckpointBindingMismatch);
        }

        let mut receipt = Self {
            version: HistoricalReplayEligibilityVersion::V1,
            restart_capture_cycle: admission.restart_capture_cycle(),
            seal_capture_cycle: admission.seal_capture_cycle(),
            mutation_count: admission.mutation_count(),
            protected_checkpoint_digest: checkpoint_digest,
            protected_seal_capsule_digest: admission.recomputed_seal_capsule_digest(),
            protected_admission_digest: admission.admission_digest().as_bytes(),
            currentness_statement_digest: currentness.statement_digest().as_bytes(),
            currentness_proof_digest: currentness.proof_digest(),
            currentness_attested_at_cycle: currentness.statement().attested_at_cycle(),
            currentness_expires_at_cycle: currentness.statement().expires_at_cycle(),
            reviewed_at_cycle,
            protected_source_equivalence_verified: true,
            checkpoint_currentness_verified: true,
            isolated_historical_projection_review_eligible: true,
            historical_replay_authorized: false,
            capsule_construction_authorized: false,
            writable_hydration_authorized: false,
            activation_authorized: false,
            receipt_digest: HistoricalReplayEligibilityDigestV1([0; 32]),
        };
        receipt.receipt_digest = digest_receipt(&receipt)?;
        Ok(receipt)
    }

    pub fn version(&self) -> HistoricalReplayEligibilityVersion {
        self.version
    }

    pub fn restart_capture_cycle(&self) -> u64 {
        self.restart_capture_cycle
    }

    pub fn seal_capture_cycle(&self) -> u64 {
        self.seal_capture_cycle
    }

    pub fn mutation_count(&self) -> usize {
        self.mutation_count
    }

    pub fn protected_checkpoint_digest(&self) -> [u8; 32] {
        self.protected_checkpoint_digest
    }

    pub fn protected_seal_capsule_digest(&self) -> [u8; 32] {
        self.protected_seal_capsule_digest
    }

    pub fn reviewed_at_cycle(&self) -> u64 {
        self.reviewed_at_cycle
    }

    pub fn protected_source_equivalence_verified(&self) -> bool {
        self.protected_source_equivalence_verified
    }

    pub fn checkpoint_currentness_verified(&self) -> bool {
        self.checkpoint_currentness_verified
    }

    pub fn isolated_historical_projection_review_eligible(&self) -> bool {
        self.isolated_historical_projection_review_eligible
    }

    pub fn historical_replay_authorized(&self) -> bool {
        self.historical_replay_authorized
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

    pub fn receipt_digest(&self) -> HistoricalReplayEligibilityDigestV1 {
        self.receipt_digest
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_against(
        &self,
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        admission: &ProtectedMutationSealAdmissionV1,
        currentness: &VerifiedRestartMutationSealCurrentnessV1,
        reviewed_at_cycle: u64,
    ) -> Result<(), HistoricalReplayEligibilityError> {
        let live = Self::evaluate(
            restart,
            seals,
            restart_receipt,
            checkpoint,
            admission,
            currentness,
            reviewed_at_cycle,
        )?;
        if &live != self {
            return Err(HistoricalReplayEligibilityError::ReceiptMismatch);
        }
        if digest_receipt(self)? != self.receipt_digest {
            return Err(HistoricalReplayEligibilityError::ReceiptDigestMismatch);
        }
        Ok(())
    }
}

fn digest_receipt(
    receipt: &HistoricalReplayEligibilityReceiptV1,
) -> Result<HistoricalReplayEligibilityDigestV1, HistoricalReplayEligibilityError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-historical-replay-eligibility-v1");
    hasher.update(&[1]);
    hasher.update(&receipt.restart_capture_cycle.to_le_bytes());
    hasher.update(&receipt.seal_capture_cycle.to_le_bytes());
    let mutation_count = u64::try_from(receipt.mutation_count)
        .map_err(|_| HistoricalReplayEligibilityError::LengthOverflow)?;
    hasher.update(&mutation_count.to_le_bytes());
    hasher.update(&receipt.protected_checkpoint_digest);
    hasher.update(&receipt.protected_seal_capsule_digest);
    hasher.update(&receipt.protected_admission_digest);
    hasher.update(&receipt.currentness_statement_digest);
    hasher.update(&receipt.currentness_proof_digest);
    hasher.update(&receipt.currentness_attested_at_cycle.to_le_bytes());
    hasher.update(&receipt.currentness_expires_at_cycle.to_le_bytes());
    hasher.update(&receipt.reviewed_at_cycle.to_le_bytes());
    hasher.update(&[u8::from(receipt.protected_source_equivalence_verified)]);
    hasher.update(&[u8::from(receipt.checkpoint_currentness_verified)]);
    hasher.update(&[u8::from(
        receipt.isolated_historical_projection_review_eligible,
    )]);
    hasher.update(&[u8::from(receipt.historical_replay_authorized)]);
    hasher.update(&[u8::from(receipt.capsule_construction_authorized)]);
    hasher.update(&[u8::from(receipt.writable_hydration_authorized)]);
    hasher.update(&[u8::from(receipt.activation_authorized)]);
    Ok(HistoricalReplayEligibilityDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

#[derive(Debug)]
pub enum HistoricalReplayEligibilityError {
    CheckpointRejected(RestartMutationSealCheckpointError),
    AdmissionRejected(ProtectedMutationSealAdmissionError),
    CurrentnessRejected(RestartMutationSealCurrentnessError),
    ReviewPredatesAdmission {
        reviewed_at_cycle: u64,
        admission_observed_at_cycle: u64,
    },
    ReviewPredatesCurrentnessVerification {
        reviewed_at_cycle: u64,
        currentness_verified_at_cycle: u64,
    },
    CurrentnessExpired {
        reviewed_at_cycle: u64,
        expires_at_cycle: u64,
    },
    UnexpectedAdmissionClaims,
    UnexpectedCurrentnessClaims,
    UnexpectedCheckpointAuthority,
    CheckpointDigestMismatch,
    RestartEpochMismatch,
    SealCapsuleDigestMismatch,
    CurrentnessCheckpointBindingMismatch,
    ReceiptMismatch,
    ReceiptDigestMismatch,
    LengthOverflow,
}

impl fmt::Display for HistoricalReplayEligibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "historical replay eligibility rejected: {self:?}")
    }
}

impl Error for HistoricalReplayEligibilityError {}
