// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only admission of a mutation-evidence-seal sidecar against protected source evidence.
//!
//! EKM-058 proves cross-component consistency but deliberately cannot prove that
//! an untrusted sidecar retained the complete historical non-basis evidence census.
//! EKM-059 protects the original EKM-056 seal-capsule digest through an external
//! deployment verifier. This module composes those boundaries: it re-runs EKM-058,
//! re-verifies the exact restart receipt against the exact restart snapshot, and
//! requires the independently recomputed seal-capsule digest to equal the exact
//! externally protected EKM-059 commitment.
//!
//! The result is an immutable audit receipt only. It does not construct an EKM-056
//! capsule, hydrate writable state, mutate trusted state, or authorize activation.

use super::belief_mutation_seal_wire::BeliefMutationSealWireSnapshotV1;
use super::belief_mutation_seal_wire_validation::{
    BeliefMutationSealValidationError, BeliefMutationSealValidationReportV1,
    BeliefMutationSealWireValidator,
};
use super::epistemic_restart_mutation_seal_checkpoint::{
    RestartMutationSealCheckpointError, VerifiedRestartMutationSealCheckpointV1,
};
use super::epistemic_restart_validation_receipt::{
    EpistemicRestartValidationReceiptError, EpistemicRestartValidationReceiptV1,
};
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtectedMutationSealAdmissionVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProtectedMutationSealAdmissionDigestV1([u8; 32]);

impl ProtectedMutationSealAdmissionDigestV1 {
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

/// Audit-only receipt proving that one supplied seal sidecar reproduces the exact
/// EKM-056 capsule digest protected by one verified EKM-059 checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectedMutationSealAdmissionV1 {
    version: ProtectedMutationSealAdmissionVersion,
    restart_capture_cycle: u64,
    seal_capture_cycle: u64,
    mutation_count: usize,
    sealed_evidence_count: usize,
    restart_validation_receipt_digest: [u8; 32],
    restart_v2_digest: [u8; 32],
    restart_outer_checksum: [u8; 32],
    seal_wire_checksum: [u8; 32],
    recomputed_seal_capsule_digest: [u8; 32],
    protected_checkpoint_digest: [u8; 32],
    protected_checkpoint_proof_digest: [u8; 32],
    observed_at_cycle: u64,
    cross_component_consistent: bool,
    protected_source_capsule_equivalent: bool,
    historical_census_completeness_protected: bool,
    capsule_construction_authorized: bool,
    writable_hydration_authorized: bool,
    activation_authorized: bool,
    admission_digest: ProtectedMutationSealAdmissionDigestV1,
}

impl ProtectedMutationSealAdmissionV1 {
    pub fn validate_and_admit(
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        protected_checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        observed_at_cycle: u64,
    ) -> Result<Self, ProtectedMutationSealAdmissionError> {
        restart_receipt
            .verify_against(restart)
            .map_err(ProtectedMutationSealAdmissionError::RestartReceiptRejected)?;
        protected_checkpoint
            .verify_internal()
            .map_err(ProtectedMutationSealAdmissionError::ProtectedCheckpointRejected)?;

        if observed_at_cycle < protected_checkpoint.verified_at_cycle() {
            return Err(ProtectedMutationSealAdmissionError::ObservationPredatesProtection {
                observed_at_cycle,
                protected_at_cycle: protected_checkpoint.verified_at_cycle(),
            });
        }
        if observed_at_cycle >= protected_checkpoint.statement().expires_at_cycle() {
            return Err(ProtectedMutationSealAdmissionError::ProtectedCheckpointExpired {
                observed_at_cycle,
                expires_at_cycle: protected_checkpoint.statement().expires_at_cycle(),
            });
        }

        let validation = BeliefMutationSealWireValidator::validate(restart, seals)
            .map_err(ProtectedMutationSealAdmissionError::SealValidationRejected)?;
        Self::from_validated(
            restart,
            seals,
            restart_receipt,
            protected_checkpoint,
            &validation,
            observed_at_cycle,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_validated(
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        protected_checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        validation: &BeliefMutationSealValidationReportV1,
        observed_at_cycle: u64,
    ) -> Result<Self, ProtectedMutationSealAdmissionError> {
        if !validation.cross_component_consistent() {
            return Err(ProtectedMutationSealAdmissionError::CrossComponentConsistencyMissing);
        }
        // EKM-058 must remain conservative on its own. EKM-060 is specifically
        // the layer that supplies the missing externally protected source digest.
        if validation.historical_census_completeness_independently_proven() {
            return Err(ProtectedMutationSealAdmissionError::UnexpectedIndependentCompletenessClaim);
        }
        if validation.capsule_construction_authorized()
            || validation.hydration_authorized()
            || validation.activation_authorized()
            || !protected_checkpoint.seal_capsule_digest_protected()
            || protected_checkpoint.trusted_state_mutated()
            || protected_checkpoint.capsule_construction_authorized()
            || protected_checkpoint.writable_hydration_authorized()
            || protected_checkpoint.activation_authorized()
        {
            return Err(ProtectedMutationSealAdmissionError::UnexpectedAuthority);
        }

        let statement = protected_checkpoint.statement();
        let mutation_count = validation.mutation_count();
        let protected_mutation_count = usize::try_from(statement.mutation_count())
            .map_err(|_| ProtectedMutationSealAdmissionError::LengthOverflow)?;

        if validation.base_capture_cycle() != restart.base.captured_at_cycle
            || validation.base_capture_cycle() != restart_receipt.captured_at_cycle()
            || validation.base_capture_cycle() != statement.restart_captured_at_cycle()
            || validation.base_capture_cycle() != statement.linked_mutation_capture_cycle()
            || validation.base_capture_cycle() != seals.linked_mutation_capture_cycle
        {
            return Err(ProtectedMutationSealAdmissionError::RestartEpochMismatch);
        }
        if validation.seal_capture_cycle() != seals.captured_at_cycle
            || validation.seal_capture_cycle() != statement.mutation_seal_capture_cycle()
        {
            return Err(ProtectedMutationSealAdmissionError::SealEpochMismatch);
        }
        if mutation_count != restart.base.mutations.len()
            || mutation_count != seals.records.len()
            || mutation_count != protected_mutation_count
        {
            return Err(ProtectedMutationSealAdmissionError::MutationCountMismatch {
                validated: mutation_count,
                restart: restart.base.mutations.len(),
                sidecar: seals.records.len(),
                protected: protected_mutation_count,
            });
        }

        if restart_receipt.receipt_digest().as_bytes()
            != statement.restart_validation_receipt_digest()
        {
            return Err(ProtectedMutationSealAdmissionError::RestartReceiptDigestMismatch);
        }
        if restart.claimed_v2_digest != restart_receipt.claimed_v2_digest()
            || restart.claimed_v2_digest != statement.restart_v2_digest()
        {
            return Err(ProtectedMutationSealAdmissionError::RestartV2DigestMismatch);
        }

        let recomputed_seal_capsule_digest = validation.recomputed_capsule_digest();
        if recomputed_seal_capsule_digest != seals.claimed_capsule_digest {
            return Err(ProtectedMutationSealAdmissionError::SidecarCapsuleDigestMismatch);
        }
        if recomputed_seal_capsule_digest != statement.mutation_seal_capsule_digest() {
            return Err(ProtectedMutationSealAdmissionError::ProtectedCapsuleDigestMismatch);
        }

        let mut receipt = Self {
            version: ProtectedMutationSealAdmissionVersion::V1,
            restart_capture_cycle: validation.base_capture_cycle(),
            seal_capture_cycle: validation.seal_capture_cycle(),
            mutation_count,
            sealed_evidence_count: validation.sealed_evidence_count(),
            restart_validation_receipt_digest: restart_receipt.receipt_digest().as_bytes(),
            restart_v2_digest: restart.claimed_v2_digest,
            restart_outer_checksum: restart.outer_checksum,
            seal_wire_checksum: seals.wire_checksum,
            recomputed_seal_capsule_digest,
            protected_checkpoint_digest: protected_checkpoint.statement_digest().as_bytes(),
            protected_checkpoint_proof_digest: protected_checkpoint.proof_digest(),
            observed_at_cycle,
            cross_component_consistent: true,
            protected_source_capsule_equivalent: true,
            historical_census_completeness_protected: true,
            capsule_construction_authorized: false,
            writable_hydration_authorized: false,
            activation_authorized: false,
            admission_digest: ProtectedMutationSealAdmissionDigestV1([0; 32]),
        };
        receipt.admission_digest = digest_admission(&receipt)?;
        Ok(receipt)
    }

    pub fn version(&self) -> ProtectedMutationSealAdmissionVersion {
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

    pub fn sealed_evidence_count(&self) -> usize {
        self.sealed_evidence_count
    }

    pub fn restart_validation_receipt_digest(&self) -> [u8; 32] {
        self.restart_validation_receipt_digest
    }

    pub fn restart_v2_digest(&self) -> [u8; 32] {
        self.restart_v2_digest
    }

    pub fn recomputed_seal_capsule_digest(&self) -> [u8; 32] {
        self.recomputed_seal_capsule_digest
    }

    pub fn protected_checkpoint_digest(&self) -> [u8; 32] {
        self.protected_checkpoint_digest
    }

    pub fn observed_at_cycle(&self) -> u64 {
        self.observed_at_cycle
    }

    pub fn cross_component_consistent(&self) -> bool {
        self.cross_component_consistent
    }

    /// True means the supplied EKM-057 semantic capsule digest is exactly the
    /// digest externally protected by EKM-059 for this restart lineage.
    pub fn protected_source_capsule_equivalent(&self) -> bool {
        self.protected_source_capsule_equivalent
    }

    /// True only in the qualified, conditional sense that the complete census in
    /// the original EKM-056 source capsule is protected by EKM-059 and the supplied
    /// sidecar independently reproduces that exact capsule digest.
    pub fn historical_census_completeness_protected(&self) -> bool {
        self.historical_census_completeness_protected
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

    pub fn admission_digest(&self) -> ProtectedMutationSealAdmissionDigestV1 {
        self.admission_digest
    }

    /// Re-run every source validator and protected binding against the exact
    /// supplied inputs. This remains a read-only equivalence check.
    pub fn verify_against(
        &self,
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        protected_checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        observed_at_cycle: u64,
    ) -> Result<(), ProtectedMutationSealAdmissionError> {
        let live = Self::validate_and_admit(
            restart,
            seals,
            restart_receipt,
            protected_checkpoint,
            observed_at_cycle,
        )?;
        if &live != self {
            return Err(ProtectedMutationSealAdmissionError::AdmissionReceiptMismatch);
        }
        if digest_admission(self)? != self.admission_digest {
            return Err(ProtectedMutationSealAdmissionError::AdmissionDigestMismatch);
        }
        Ok(())
    }
}

fn digest_admission(
    receipt: &ProtectedMutationSealAdmissionV1,
) -> Result<ProtectedMutationSealAdmissionDigestV1, ProtectedMutationSealAdmissionError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-protected-mutation-seal-admission-v1");
    hasher.update(&[1]);
    hasher.update(&receipt.restart_capture_cycle.to_le_bytes());
    hasher.update(&receipt.seal_capture_cycle.to_le_bytes());
    hash_usize(&mut hasher, receipt.mutation_count)?;
    hash_usize(&mut hasher, receipt.sealed_evidence_count)?;
    hasher.update(&receipt.restart_validation_receipt_digest);
    hasher.update(&receipt.restart_v2_digest);
    hasher.update(&receipt.restart_outer_checksum);
    hasher.update(&receipt.seal_wire_checksum);
    hasher.update(&receipt.recomputed_seal_capsule_digest);
    hasher.update(&receipt.protected_checkpoint_digest);
    hasher.update(&receipt.protected_checkpoint_proof_digest);
    hasher.update(&receipt.observed_at_cycle.to_le_bytes());
    hasher.update(&[u8::from(receipt.cross_component_consistent)]);
    hasher.update(&[u8::from(receipt.protected_source_capsule_equivalent)]);
    hasher.update(&[u8::from(
        receipt.historical_census_completeness_protected,
    )]);
    hasher.update(&[u8::from(receipt.capsule_construction_authorized)]);
    hasher.update(&[u8::from(receipt.writable_hydration_authorized)]);
    hasher.update(&[u8::from(receipt.activation_authorized)]);
    Ok(ProtectedMutationSealAdmissionDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn hash_usize(
    hasher: &mut blake3::Hasher,
    value: usize,
) -> Result<(), ProtectedMutationSealAdmissionError> {
    let value = u64::try_from(value).map_err(|_| ProtectedMutationSealAdmissionError::LengthOverflow)?;
    hasher.update(&value.to_le_bytes());
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtectedMutationSealAdmissionError {
    RestartReceiptRejected(EpistemicRestartValidationReceiptError),
    ProtectedCheckpointRejected(RestartMutationSealCheckpointError),
    SealValidationRejected(BeliefMutationSealValidationError),
    ObservationPredatesProtection { observed_at_cycle: u64, protected_at_cycle: u64 },
    ProtectedCheckpointExpired { observed_at_cycle: u64, expires_at_cycle: u64 },
    CrossComponentConsistencyMissing,
    UnexpectedIndependentCompletenessClaim,
    UnexpectedAuthority,
    RestartEpochMismatch,
    SealEpochMismatch,
    MutationCountMismatch {
        validated: usize,
        restart: usize,
        sidecar: usize,
        protected: usize,
    },
    RestartReceiptDigestMismatch,
    RestartV2DigestMismatch,
    SidecarCapsuleDigestMismatch,
    ProtectedCapsuleDigestMismatch,
    AdmissionReceiptMismatch,
    AdmissionDigestMismatch,
    LengthOverflow,
}

impl fmt::Display for ProtectedMutationSealAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "protected mutation-seal admission rejected: {self:?}")
    }
}

impl Error for ProtectedMutationSealAdmissionError {}
