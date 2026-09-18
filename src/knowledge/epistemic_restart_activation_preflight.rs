// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Non-authorizing activation-preflight composition receipt for epistemic restart.
//!
//! EKM-067 provides a sealed split-state sandbox and EKM-068 fences the actual
//! live EKM state. This module composes those with candidate current-head evidence
//! and a protected joint trust-context checkpoint.
//!
//! Important: EKM-054 proves that the joint trust-context checkpoint is protected
//! and unexpired, but its evidence taxonomy includes mechanisms (for example a
//! generic signature) that do not independently prove that checkpoint is the
//! deployment's current/latest head. EKM-069 therefore does **not** mark an
//! activation transaction review eligible yet. A separate current-head attestation
//! for the EKM-054 checkpoint is required by a later tranche.

use crate::knowledge::belief_mutation_firewall::EpistemicSupportStore;
use crate::knowledge::belief_mutation_seal_wire::BeliefMutationSealWireSnapshotV1;
use crate::knowledge::belief_revision_receipt::BeliefRevisionHistory;
use crate::knowledge::belief_revision_schema_history::BeliefRevisionSchemaHistoryV1;
use crate::knowledge::claim_evidence::EpistemicLedger;
use crate::knowledge::epistemic_restart_continuity::live_epoch_fence::{
    LiveEpistemicEpochFenceError, LiveEpistemicEpochFenceV1,
};
use crate::knowledge::epistemic_restart_historical_firewall_replay::HistoricalFirewallReplayReportV1;
use crate::knowledge::epistemic_restart_historical_projection::HistoricalEvidenceProjectionV1;
use crate::knowledge::epistemic_restart_historical_replay_eligibility::HistoricalReplayEligibilityReceiptV1;
use crate::knowledge::epistemic_restart_manifest::EpistemicLedgerInventoryV1;
use crate::knowledge::epistemic_restart_mutation_seal_admission::ProtectedMutationSealAdmissionV1;
use crate::knowledge::epistemic_restart_mutation_seal_checkpoint::VerifiedRestartMutationSealCheckpointV1;
use crate::knowledge::epistemic_restart_mutation_seal_currentness::{
    RestartMutationSealCurrentnessError, VerifiedRestartMutationSealCurrentnessV1,
};
use crate::knowledge::epistemic_restart_quarantine_facade::ReadOnlyEpistemicRestartQuarantineV2;
use crate::knowledge::epistemic_restart_revision_audit_restoration::ImmutableRevisionAuditRestorationV1;
use crate::knowledge::epistemic_restart_split_state_sandbox::{
    SealedSplitStateHydrationSandboxV1, SplitStateHydrationSandboxError,
};
use crate::knowledge::epistemic_restart_support_hydration_eligibility::SupportHydrationEligibilityReceiptV1;
use crate::knowledge::epistemic_restart_trust_checkpoint::{
    RestartTrustContextCheckpointError, VerifiedRestartTrustContextCheckpointV1,
};
use crate::knowledge::epistemic_restart_validation_receipt::EpistemicRestartValidationReceiptV1;
use crate::knowledge::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationPreflightVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivationPreflightDigestV1([u8; 32]);

impl ActivationPreflightDigestV1 {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationPreflightReceiptV1 {
    version: ActivationPreflightVersion,
    preflight_at_cycle: u64,
    restart_capture_cycle: u64,
    sandbox_digest: [u8; 32],
    live_epoch_fence_digest: [u8; 32],
    live_epoch_continuity_digest: [u8; 32],
    currentness_statement_digest: [u8; 32],
    currentness_proof_digest: [u8; 32],
    trust_checkpoint_statement_digest: [u8; 32],
    trust_checkpoint_proof_digest: [u8; 32],
    source_sandbox_rederived: bool,
    live_epoch_unchanged: bool,
    candidate_current_head_proven: bool,
    protected_trust_checkpoint_valid: bool,
    protected_trust_checkpoint_currentness_independently_proven: bool,
    trust_context_current_head_attestation_required: bool,
    trusted_state_mutated: bool,
    activation_transaction_review_eligible: bool,
    live_state_swap_authorized: bool,
    rollback_authorized: bool,
    activation_authorized: bool,
    trusted_checkpoint_commit_authorized: bool,
    receipt_digest: ActivationPreflightDigestV1,
}

impl ActivationPreflightReceiptV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        mutation_checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        admission: &ProtectedMutationSealAdmissionV1,
        currentness: &VerifiedRestartMutationSealCurrentnessV1,
        eligibility: &HistoricalReplayEligibilityReceiptV1,
        projection: &HistoricalEvidenceProjectionV1,
        replay_report: &HistoricalFirewallReplayReportV1,
        quarantine: &ReadOnlyEpistemicRestartQuarantineV2,
        trust_checkpoint: &VerifiedRestartTrustContextCheckpointV1,
        support_eligibility: &SupportHydrationEligibilityReceiptV1,
        audit: &ImmutableRevisionAuditRestorationV1,
        sandbox: &SealedSplitStateHydrationSandboxV1,
        live_fence: &LiveEpistemicEpochFenceV1,
        live_ledger: &EpistemicLedger,
        live_inventory: &EpistemicLedgerInventoryV1,
        live_store: &EpistemicSupportStore,
        live_history: &BeliefRevisionHistory,
        live_schema_history: &BeliefRevisionSchemaHistoryV1,
        preflight_at_cycle: u64,
    ) -> Result<Self, ActivationPreflightError> {
        sandbox.verify().map_err(ActivationPreflightError::SandboxRejected)?;
        currentness.verify_internal().map_err(ActivationPreflightError::CurrentnessRejected)?;
        trust_checkpoint.verify_internal().map_err(ActivationPreflightError::TrustCheckpointRejected)?;

        if preflight_at_cycle < sandbox.sandboxed_at_cycle() {
            return Err(ActivationPreflightError::PreflightPredatesSandbox { preflight_at_cycle, sandboxed_at_cycle: sandbox.sandboxed_at_cycle() });
        }
        if preflight_at_cycle < live_fence.established_at_cycle() {
            return Err(ActivationPreflightError::PreflightPredatesLiveFence { preflight_at_cycle, fence_cycle: live_fence.established_at_cycle() });
        }
        if preflight_at_cycle < currentness.verified_at_cycle() {
            return Err(ActivationPreflightError::PreflightPredatesCurrentnessVerification { preflight_at_cycle, verified_at_cycle: currentness.verified_at_cycle() });
        }
        if preflight_at_cycle >= currentness.statement().expires_at_cycle() {
            return Err(ActivationPreflightError::CandidateCurrentnessExpired { preflight_at_cycle, expires_at_cycle: currentness.statement().expires_at_cycle() });
        }
        if preflight_at_cycle < trust_checkpoint.verified_at_cycle() {
            return Err(ActivationPreflightError::PreflightPredatesTrustCheckpointVerification { preflight_at_cycle, verified_at_cycle: trust_checkpoint.verified_at_cycle() });
        }
        if preflight_at_cycle >= trust_checkpoint.statement().expires_at_cycle() {
            return Err(ActivationPreflightError::TrustCheckpointExpired { preflight_at_cycle, expires_at_cycle: trust_checkpoint.statement().expires_at_cycle() });
        }
        if !currentness.current_head_proven()
            || currentness.trusted_state_mutated()
            || currentness.historical_replay_authorized()
            || currentness.writable_hydration_authorized()
            || currentness.activation_authorized()
        {
            return Err(ActivationPreflightError::UnexpectedCurrentnessAuthority);
        }
        if trust_checkpoint.trusted_state_mutated()
            || trust_checkpoint.quarantine_construction_authorized()
            || trust_checkpoint.writable_hydration_authorized()
            || trust_checkpoint.activation_authorized()
        {
            return Err(ActivationPreflightError::UnexpectedTrustCheckpointAuthority);
        }
        if sandbox.writable_state_export_authorized() || sandbox.activation_authorized() {
            return Err(ActivationPreflightError::UnexpectedSandboxAuthority);
        }
        if currentness.statement().restart_capture_cycle() != sandbox.restart_capture_cycle() {
            return Err(ActivationPreflightError::CandidateCaptureCycleMismatch { currentness: currentness.statement().restart_capture_cycle(), sandbox: sandbox.restart_capture_cycle() });
        }

        let rederived = SealedSplitStateHydrationSandboxV1::hydrate_sealed(
            restart, seals, restart_receipt, mutation_checkpoint, admission, currentness,
            eligibility, projection, replay_report, quarantine, trust_checkpoint,
            support_eligibility, audit.clone(), sandbox.sandboxed_at_cycle(),
        )
        .map_err(ActivationPreflightError::SandboxRederivationRejected)?;
        if rederived.sandbox_digest() != sandbox.sandbox_digest() {
            return Err(ActivationPreflightError::SandboxRederivationMismatch);
        }

        let live_continuity = live_fence
            .recheck_live_unchanged(
                sandbox, live_ledger, live_inventory, live_store, live_history,
                live_schema_history, preflight_at_cycle,
            )
            .map_err(ActivationPreflightError::LiveEpochRejected)?;
        if !live_continuity.live_state_unchanged()
            || live_continuity.activation_preflight_authorized()
            || live_continuity.activation_authorized()
            || live_continuity.trusted_checkpoint_commit_authorized()
        {
            return Err(ActivationPreflightError::UnexpectedLiveContinuityAuthority);
        }

        let mut receipt = Self {
            version: ActivationPreflightVersion::V1,
            preflight_at_cycle,
            restart_capture_cycle: restart.base.captured_at_cycle,
            sandbox_digest: sandbox.sandbox_digest().as_bytes(),
            live_epoch_fence_digest: live_fence.fence_digest().as_bytes(),
            live_epoch_continuity_digest: live_continuity.receipt_digest().as_bytes(),
            currentness_statement_digest: currentness.statement_digest().as_bytes(),
            currentness_proof_digest: currentness.proof_digest(),
            trust_checkpoint_statement_digest: trust_checkpoint.statement_digest().as_bytes(),
            trust_checkpoint_proof_digest: trust_checkpoint.proof_digest(),
            source_sandbox_rederived: true,
            live_epoch_unchanged: true,
            candidate_current_head_proven: true,
            protected_trust_checkpoint_valid: true,
            protected_trust_checkpoint_currentness_independently_proven: false,
            trust_context_current_head_attestation_required: true,
            trusted_state_mutated: false,
            activation_transaction_review_eligible: false,
            live_state_swap_authorized: false,
            rollback_authorized: false,
            activation_authorized: false,
            trusted_checkpoint_commit_authorized: false,
            receipt_digest: ActivationPreflightDigestV1([0; 32]),
        };
        receipt.receipt_digest = digest_preflight(&receipt)?;
        Ok(receipt)
    }

    pub fn version(&self) -> ActivationPreflightVersion { self.version }
    pub fn preflight_at_cycle(&self) -> u64 { self.preflight_at_cycle }
    pub fn restart_capture_cycle(&self) -> u64 { self.restart_capture_cycle }
    pub fn source_sandbox_rederived(&self) -> bool { self.source_sandbox_rederived }
    pub fn live_epoch_unchanged(&self) -> bool { self.live_epoch_unchanged }
    pub fn candidate_current_head_proven(&self) -> bool { self.candidate_current_head_proven }
    pub fn protected_trust_checkpoint_valid(&self) -> bool { self.protected_trust_checkpoint_valid }
    pub fn protected_trust_checkpoint_currentness_independently_proven(&self) -> bool { self.protected_trust_checkpoint_currentness_independently_proven }
    pub fn trust_context_current_head_attestation_required(&self) -> bool { self.trust_context_current_head_attestation_required }
    pub fn trusted_state_mutated(&self) -> bool { self.trusted_state_mutated }
    pub fn activation_transaction_review_eligible(&self) -> bool { self.activation_transaction_review_eligible }
    pub fn live_state_swap_authorized(&self) -> bool { self.live_state_swap_authorized }
    pub fn rollback_authorized(&self) -> bool { self.rollback_authorized }
    pub fn activation_authorized(&self) -> bool { self.activation_authorized }
    pub fn trusted_checkpoint_commit_authorized(&self) -> bool { self.trusted_checkpoint_commit_authorized }
    pub fn receipt_digest(&self) -> ActivationPreflightDigestV1 { self.receipt_digest }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_against(
        &self,
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        mutation_checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        admission: &ProtectedMutationSealAdmissionV1,
        currentness: &VerifiedRestartMutationSealCurrentnessV1,
        eligibility: &HistoricalReplayEligibilityReceiptV1,
        projection: &HistoricalEvidenceProjectionV1,
        replay_report: &HistoricalFirewallReplayReportV1,
        quarantine: &ReadOnlyEpistemicRestartQuarantineV2,
        trust_checkpoint: &VerifiedRestartTrustContextCheckpointV1,
        support_eligibility: &SupportHydrationEligibilityReceiptV1,
        audit: &ImmutableRevisionAuditRestorationV1,
        sandbox: &SealedSplitStateHydrationSandboxV1,
        live_fence: &LiveEpistemicEpochFenceV1,
        live_ledger: &EpistemicLedger,
        live_inventory: &EpistemicLedgerInventoryV1,
        live_store: &EpistemicSupportStore,
        live_history: &BeliefRevisionHistory,
        live_schema_history: &BeliefRevisionSchemaHistoryV1,
        preflight_at_cycle: u64,
    ) -> Result<(), ActivationPreflightError> {
        let live = Self::evaluate(
            restart, seals, restart_receipt, mutation_checkpoint, admission, currentness,
            eligibility, projection, replay_report, quarantine, trust_checkpoint,
            support_eligibility, audit, sandbox, live_fence, live_ledger, live_inventory,
            live_store, live_history, live_schema_history, preflight_at_cycle,
        )?;
        if &live != self { return Err(ActivationPreflightError::ReceiptMismatch); }
        if digest_preflight(self)? != self.receipt_digest { return Err(ActivationPreflightError::ReceiptDigestMismatch); }
        Ok(())
    }
}

fn digest_preflight(receipt: &ActivationPreflightReceiptV1) -> Result<ActivationPreflightDigestV1, ActivationPreflightError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-activation-preflight-v1");
    hasher.update(&[1]);
    hasher.update(&receipt.preflight_at_cycle.to_le_bytes());
    hasher.update(&receipt.restart_capture_cycle.to_le_bytes());
    hasher.update(&receipt.sandbox_digest);
    hasher.update(&receipt.live_epoch_fence_digest);
    hasher.update(&receipt.live_epoch_continuity_digest);
    hasher.update(&receipt.currentness_statement_digest);
    hasher.update(&receipt.currentness_proof_digest);
    hasher.update(&receipt.trust_checkpoint_statement_digest);
    hasher.update(&receipt.trust_checkpoint_proof_digest);
    hasher.update(&[u8::from(receipt.source_sandbox_rederived)]);
    hasher.update(&[u8::from(receipt.live_epoch_unchanged)]);
    hasher.update(&[u8::from(receipt.candidate_current_head_proven)]);
    hasher.update(&[u8::from(receipt.protected_trust_checkpoint_valid)]);
    hasher.update(&[u8::from(receipt.protected_trust_checkpoint_currentness_independently_proven)]);
    hasher.update(&[u8::from(receipt.trust_context_current_head_attestation_required)]);
    hasher.update(&[u8::from(receipt.trusted_state_mutated)]);
    hasher.update(&[u8::from(receipt.activation_transaction_review_eligible)]);
    hasher.update(&[u8::from(receipt.live_state_swap_authorized)]);
    hasher.update(&[u8::from(receipt.rollback_authorized)]);
    hasher.update(&[u8::from(receipt.activation_authorized)]);
    hasher.update(&[u8::from(receipt.trusted_checkpoint_commit_authorized)]);
    Ok(ActivationPreflightDigestV1(*hasher.finalize().as_bytes()))
}

#[derive(Debug)]
pub enum ActivationPreflightError {
    SandboxRejected(SplitStateHydrationSandboxError),
    SandboxRederivationRejected(SplitStateHydrationSandboxError),
    CurrentnessRejected(RestartMutationSealCurrentnessError),
    TrustCheckpointRejected(RestartTrustContextCheckpointError),
    LiveEpochRejected(LiveEpistemicEpochFenceError),
    PreflightPredatesSandbox { preflight_at_cycle: u64, sandboxed_at_cycle: u64 },
    PreflightPredatesLiveFence { preflight_at_cycle: u64, fence_cycle: u64 },
    PreflightPredatesCurrentnessVerification { preflight_at_cycle: u64, verified_at_cycle: u64 },
    CandidateCurrentnessExpired { preflight_at_cycle: u64, expires_at_cycle: u64 },
    PreflightPredatesTrustCheckpointVerification { preflight_at_cycle: u64, verified_at_cycle: u64 },
    TrustCheckpointExpired { preflight_at_cycle: u64, expires_at_cycle: u64 },
    UnexpectedCurrentnessAuthority,
    UnexpectedTrustCheckpointAuthority,
    UnexpectedSandboxAuthority,
    UnexpectedLiveContinuityAuthority,
    CandidateCaptureCycleMismatch { currentness: u64, sandbox: u64 },
    SandboxRederivationMismatch,
    ReceiptMismatch,
    ReceiptDigestMismatch,
}

impl fmt::Display for ActivationPreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "activation preflight rejected: {self:?}") }
}
impl Error for ActivationPreflightError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_digest_binds_missing_trust_currentness_and_non_authority() {
        let mut receipt = ActivationPreflightReceiptV1 {
            version: ActivationPreflightVersion::V1,
            preflight_at_cycle: 10,
            restart_capture_cycle: 7,
            sandbox_digest: [1; 32],
            live_epoch_fence_digest: [2; 32],
            live_epoch_continuity_digest: [3; 32],
            currentness_statement_digest: [4; 32],
            currentness_proof_digest: [5; 32],
            trust_checkpoint_statement_digest: [6; 32],
            trust_checkpoint_proof_digest: [7; 32],
            source_sandbox_rederived: true,
            live_epoch_unchanged: true,
            candidate_current_head_proven: true,
            protected_trust_checkpoint_valid: true,
            protected_trust_checkpoint_currentness_independently_proven: false,
            trust_context_current_head_attestation_required: true,
            trusted_state_mutated: false,
            activation_transaction_review_eligible: false,
            live_state_swap_authorized: false,
            rollback_authorized: false,
            activation_authorized: false,
            trusted_checkpoint_commit_authorized: false,
            receipt_digest: ActivationPreflightDigestV1([0; 32]),
        };
        let first = digest_preflight(&receipt).unwrap();
        assert!(receipt.source_sandbox_rederived());
        assert!(receipt.candidate_current_head_proven());
        assert!(receipt.protected_trust_checkpoint_valid());
        assert!(!receipt.protected_trust_checkpoint_currentness_independently_proven());
        assert!(receipt.trust_context_current_head_attestation_required());
        assert!(!receipt.activation_transaction_review_eligible());
        assert!(!receipt.live_state_swap_authorized());
        assert!(!receipt.activation_authorized());
        assert!(!receipt.trusted_checkpoint_commit_authorized());
        receipt.preflight_at_cycle += 1;
        let second = digest_preflight(&receipt).unwrap();
        assert_ne!(first, second);
    }
}
