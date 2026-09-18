// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only eligibility receipt for a later atomic restart activation transaction.
//!
//! EKM-069 intentionally leaves transaction review ineligible because EKM-054
//! protected-checkpoint validity does not prove current-head status. EKM-070 adds
//! that missing current-head attestation. This module composes both at one exact
//! review cycle while still granting no swap, rollback, activation, or checkpoint
//! commit authority.

use crate::knowledge::belief_mutation_firewall::EpistemicSupportStore;
use crate::knowledge::belief_mutation_seal_wire::BeliefMutationSealWireSnapshotV1;
use crate::knowledge::belief_revision_receipt::BeliefRevisionHistory;
use crate::knowledge::belief_revision_schema_history::BeliefRevisionSchemaHistoryV1;
use crate::knowledge::claim_evidence::EpistemicLedger;
use crate::knowledge::epistemic_restart_continuity::activation_preflight::{
    ActivationPreflightError, ActivationPreflightReceiptV1,
};
use crate::knowledge::epistemic_restart_continuity::live_epoch_fence::LiveEpistemicEpochFenceV1;
use crate::knowledge::epistemic_restart_continuity::trust_checkpoint_currentness::{
    RestartTrustContextCurrentnessError, VerifiedRestartTrustContextCurrentnessV1,
};
use crate::knowledge::epistemic_restart_historical_firewall_replay::HistoricalFirewallReplayReportV1;
use crate::knowledge::epistemic_restart_historical_projection::HistoricalEvidenceProjectionV1;
use crate::knowledge::epistemic_restart_historical_replay_eligibility::HistoricalReplayEligibilityReceiptV1;
use crate::knowledge::epistemic_restart_manifest::EpistemicLedgerInventoryV1;
use crate::knowledge::epistemic_restart_mutation_seal_admission::ProtectedMutationSealAdmissionV1;
use crate::knowledge::epistemic_restart_mutation_seal_checkpoint::VerifiedRestartMutationSealCheckpointV1;
use crate::knowledge::epistemic_restart_mutation_seal_currentness::VerifiedRestartMutationSealCurrentnessV1;
use crate::knowledge::epistemic_restart_quarantine_facade::ReadOnlyEpistemicRestartQuarantineV2;
use crate::knowledge::epistemic_restart_revision_audit_restoration::ImmutableRevisionAuditRestorationV1;
use crate::knowledge::epistemic_restart_split_state_sandbox::SealedSplitStateHydrationSandboxV1;
use crate::knowledge::epistemic_restart_support_hydration_eligibility::SupportHydrationEligibilityReceiptV1;
use crate::knowledge::epistemic_restart_trust_checkpoint::VerifiedRestartTrustContextCheckpointV1;
use crate::knowledge::epistemic_restart_validation_receipt::EpistemicRestartValidationReceiptV1;
use crate::knowledge::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationTransactionReviewEligibilityVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivationTransactionReviewEligibilityDigestV1([u8; 32]);

impl ActivationTransactionReviewEligibilityDigestV1 {
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

/// Complete read-only review receipt. `activation_transaction_review_eligible`
/// means only that a later atomic transaction design may be evaluated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationTransactionReviewEligibilityReceiptV1 {
    version: ActivationTransactionReviewEligibilityVersion,
    reviewed_at_cycle: u64,
    restart_capture_cycle: u64,
    fresh_preflight_digest: [u8; 32],
    trust_context_currentness_statement_digest: [u8; 32],
    trust_context_currentness_proof_digest: [u8; 32],
    live_epoch_unchanged_at_review: bool,
    candidate_current_head_proven_at_review: bool,
    trust_context_current_head_proven_at_review: bool,
    exact_trust_checkpoint_binding_proven: bool,
    activation_transaction_review_eligible: bool,
    live_state_lock_acquired: bool,
    compare_and_swap_implemented: bool,
    time_of_check_use_window_closed: bool,
    live_state_swap_authorized: bool,
    rollback_authorized: bool,
    activation_authorized: bool,
    trusted_state_mutated: bool,
    trusted_checkpoint_commit_authorized: bool,
    receipt_digest: ActivationTransactionReviewEligibilityDigestV1,
}

impl ActivationTransactionReviewEligibilityReceiptV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        mutation_checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        admission: &ProtectedMutationSealAdmissionV1,
        candidate_currentness: &VerifiedRestartMutationSealCurrentnessV1,
        eligibility: &HistoricalReplayEligibilityReceiptV1,
        projection: &HistoricalEvidenceProjectionV1,
        replay_report: &HistoricalFirewallReplayReportV1,
        quarantine: &ReadOnlyEpistemicRestartQuarantineV2,
        trust_checkpoint: &VerifiedRestartTrustContextCheckpointV1,
        trust_currentness: &VerifiedRestartTrustContextCurrentnessV1,
        support_eligibility: &SupportHydrationEligibilityReceiptV1,
        audit: &ImmutableRevisionAuditRestorationV1,
        sandbox: &SealedSplitStateHydrationSandboxV1,
        live_fence: &LiveEpistemicEpochFenceV1,
        live_ledger: &EpistemicLedger,
        live_inventory: &EpistemicLedgerInventoryV1,
        live_store: &EpistemicSupportStore,
        live_history: &BeliefRevisionHistory,
        live_schema_history: &BeliefRevisionSchemaHistoryV1,
        reviewed_at_cycle: u64,
    ) -> Result<Self, ActivationTransactionReviewEligibilityError> {
        // Recompute EKM-069 at the current review cycle rather than accepting a
        // stale detached preflight receipt.
        let preflight = ActivationPreflightReceiptV1::evaluate(
            restart,
            seals,
            restart_receipt,
            mutation_checkpoint,
            admission,
            candidate_currentness,
            eligibility,
            projection,
            replay_report,
            quarantine,
            trust_checkpoint,
            support_eligibility,
            audit,
            sandbox,
            live_fence,
            live_ledger,
            live_inventory,
            live_store,
            live_history,
            live_schema_history,
            reviewed_at_cycle,
        )
        .map_err(ActivationTransactionReviewEligibilityError::PreflightRejected)?;

        trust_currentness
            .verify_internal()
            .map_err(ActivationTransactionReviewEligibilityError::TrustCurrentnessRejected)?;

        if reviewed_at_cycle < trust_currentness.verified_at_cycle() {
            return Err(
                ActivationTransactionReviewEligibilityError::ReviewPredatesTrustCurrentness {
                    reviewed_at_cycle,
                    verified_at_cycle: trust_currentness.verified_at_cycle(),
                },
            );
        }
        if reviewed_at_cycle >= trust_currentness.statement().expires_at_cycle() {
            return Err(
                ActivationTransactionReviewEligibilityError::TrustCurrentnessExpired {
                    reviewed_at_cycle,
                    expires_at_cycle: trust_currentness.statement().expires_at_cycle(),
                },
            );
        }
        if reviewed_at_cycle >= candidate_currentness.statement().expires_at_cycle() {
            return Err(
                ActivationTransactionReviewEligibilityError::CandidateCurrentnessExpired {
                    reviewed_at_cycle,
                    expires_at_cycle: candidate_currentness.statement().expires_at_cycle(),
                },
            );
        }
        if reviewed_at_cycle >= trust_checkpoint.statement().expires_at_cycle() {
            return Err(
                ActivationTransactionReviewEligibilityError::TrustCheckpointExpired {
                    reviewed_at_cycle,
                    expires_at_cycle: trust_checkpoint.statement().expires_at_cycle(),
                },
            );
        }

        if trust_currentness.statement().checkpoint_statement_digest()
            != trust_checkpoint.statement_digest().as_bytes()
            || trust_currentness.statement().checkpoint_sequence()
                != trust_checkpoint.statement().sequence()
            || trust_currentness.statement().context_digest()
                != trust_checkpoint.statement().context_digest().as_bytes()
            || trust_currentness.statement().deployment_id()
                != trust_checkpoint.statement().deployment_id()
            || trust_currentness.statement().trust_domain_id()
                != trust_checkpoint.statement().trust_domain_id()
        {
            return Err(
                ActivationTransactionReviewEligibilityError::TrustCurrentnessCheckpointMismatch,
            );
        }

        if !preflight.source_sandbox_rederived()
            || !preflight.live_epoch_unchanged()
            || !preflight.candidate_current_head_proven()
            || !preflight.protected_trust_checkpoint_valid()
            || preflight.protected_trust_checkpoint_currentness_independently_proven()
            || !preflight.trust_context_current_head_attestation_required()
            || preflight.trusted_state_mutated()
            || preflight.activation_transaction_review_eligible()
            || preflight.live_state_swap_authorized()
            || preflight.rollback_authorized()
            || preflight.activation_authorized()
            || preflight.trusted_checkpoint_commit_authorized()
        {
            return Err(
                ActivationTransactionReviewEligibilityError::UnexpectedPreflightClaims,
            );
        }
        if !trust_currentness.current_head_proven()
            || trust_currentness.trusted_state_mutated()
            || trust_currentness.activation_preflight_authorized()
            || trust_currentness.activation_authorized()
            || trust_currentness.trusted_checkpoint_commit_authorized()
        {
            return Err(
                ActivationTransactionReviewEligibilityError::UnexpectedTrustCurrentnessAuthority,
            );
        }

        let mut receipt = Self {
            version: ActivationTransactionReviewEligibilityVersion::V1,
            reviewed_at_cycle,
            restart_capture_cycle: preflight.restart_capture_cycle(),
            fresh_preflight_digest: preflight.receipt_digest().as_bytes(),
            trust_context_currentness_statement_digest: trust_currentness
                .statement_digest()
                .as_bytes(),
            trust_context_currentness_proof_digest: trust_currentness.proof_digest(),
            live_epoch_unchanged_at_review: true,
            candidate_current_head_proven_at_review: true,
            trust_context_current_head_proven_at_review: true,
            exact_trust_checkpoint_binding_proven: true,
            activation_transaction_review_eligible: true,
            live_state_lock_acquired: false,
            compare_and_swap_implemented: false,
            time_of_check_use_window_closed: false,
            live_state_swap_authorized: false,
            rollback_authorized: false,
            activation_authorized: false,
            trusted_state_mutated: false,
            trusted_checkpoint_commit_authorized: false,
            receipt_digest: ActivationTransactionReviewEligibilityDigestV1([0; 32]),
        };
        receipt.receipt_digest = digest_receipt(&receipt)?;
        Ok(receipt)
    }

    pub fn version(&self) -> ActivationTransactionReviewEligibilityVersion {
        self.version
    }
    pub fn reviewed_at_cycle(&self) -> u64 {
        self.reviewed_at_cycle
    }
    pub fn restart_capture_cycle(&self) -> u64 {
        self.restart_capture_cycle
    }
    pub fn live_epoch_unchanged_at_review(&self) -> bool {
        self.live_epoch_unchanged_at_review
    }
    pub fn candidate_current_head_proven_at_review(&self) -> bool {
        self.candidate_current_head_proven_at_review
    }
    pub fn trust_context_current_head_proven_at_review(&self) -> bool {
        self.trust_context_current_head_proven_at_review
    }
    pub fn exact_trust_checkpoint_binding_proven(&self) -> bool {
        self.exact_trust_checkpoint_binding_proven
    }
    pub fn activation_transaction_review_eligible(&self) -> bool {
        self.activation_transaction_review_eligible
    }
    pub fn live_state_lock_acquired(&self) -> bool {
        self.live_state_lock_acquired
    }
    pub fn compare_and_swap_implemented(&self) -> bool {
        self.compare_and_swap_implemented
    }
    pub fn time_of_check_use_window_closed(&self) -> bool {
        self.time_of_check_use_window_closed
    }
    pub fn live_state_swap_authorized(&self) -> bool {
        self.live_state_swap_authorized
    }
    pub fn rollback_authorized(&self) -> bool {
        self.rollback_authorized
    }
    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }
    pub fn trusted_state_mutated(&self) -> bool {
        self.trusted_state_mutated
    }
    pub fn trusted_checkpoint_commit_authorized(&self) -> bool {
        self.trusted_checkpoint_commit_authorized
    }
    pub fn receipt_digest(&self) -> ActivationTransactionReviewEligibilityDigestV1 {
        self.receipt_digest
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_against(
        &self,
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        mutation_checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        admission: &ProtectedMutationSealAdmissionV1,
        candidate_currentness: &VerifiedRestartMutationSealCurrentnessV1,
        eligibility: &HistoricalReplayEligibilityReceiptV1,
        projection: &HistoricalEvidenceProjectionV1,
        replay_report: &HistoricalFirewallReplayReportV1,
        quarantine: &ReadOnlyEpistemicRestartQuarantineV2,
        trust_checkpoint: &VerifiedRestartTrustContextCheckpointV1,
        trust_currentness: &VerifiedRestartTrustContextCurrentnessV1,
        support_eligibility: &SupportHydrationEligibilityReceiptV1,
        audit: &ImmutableRevisionAuditRestorationV1,
        sandbox: &SealedSplitStateHydrationSandboxV1,
        live_fence: &LiveEpistemicEpochFenceV1,
        live_ledger: &EpistemicLedger,
        live_inventory: &EpistemicLedgerInventoryV1,
        live_store: &EpistemicSupportStore,
        live_history: &BeliefRevisionHistory,
        live_schema_history: &BeliefRevisionSchemaHistoryV1,
        reviewed_at_cycle: u64,
    ) -> Result<(), ActivationTransactionReviewEligibilityError> {
        let live = Self::evaluate(
            restart,
            seals,
            restart_receipt,
            mutation_checkpoint,
            admission,
            candidate_currentness,
            eligibility,
            projection,
            replay_report,
            quarantine,
            trust_checkpoint,
            trust_currentness,
            support_eligibility,
            audit,
            sandbox,
            live_fence,
            live_ledger,
            live_inventory,
            live_store,
            live_history,
            live_schema_history,
            reviewed_at_cycle,
        )?;
        if &live != self {
            return Err(ActivationTransactionReviewEligibilityError::ReceiptMismatch);
        }
        if digest_receipt(self)? != self.receipt_digest {
            return Err(ActivationTransactionReviewEligibilityError::ReceiptDigestMismatch);
        }
        Ok(())
    }
}

fn digest_receipt(
    receipt: &ActivationTransactionReviewEligibilityReceiptV1,
) -> Result<ActivationTransactionReviewEligibilityDigestV1, ActivationTransactionReviewEligibilityError>
{
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-activation-transaction-review-eligibility-v1");
    hasher.update(&[1]);
    hasher.update(&receipt.reviewed_at_cycle.to_le_bytes());
    hasher.update(&receipt.restart_capture_cycle.to_le_bytes());
    hasher.update(&receipt.fresh_preflight_digest);
    hasher.update(&receipt.trust_context_currentness_statement_digest);
    hasher.update(&receipt.trust_context_currentness_proof_digest);
    hasher.update(&[u8::from(receipt.live_epoch_unchanged_at_review)]);
    hasher.update(&[u8::from(
        receipt.candidate_current_head_proven_at_review,
    )]);
    hasher.update(&[u8::from(
        receipt.trust_context_current_head_proven_at_review,
    )]);
    hasher.update(&[u8::from(receipt.exact_trust_checkpoint_binding_proven)]);
    hasher.update(&[u8::from(receipt.activation_transaction_review_eligible)]);
    hasher.update(&[u8::from(receipt.live_state_lock_acquired)]);
    hasher.update(&[u8::from(receipt.compare_and_swap_implemented)]);
    hasher.update(&[u8::from(receipt.time_of_check_use_window_closed)]);
    hasher.update(&[u8::from(receipt.live_state_swap_authorized)]);
    hasher.update(&[u8::from(receipt.rollback_authorized)]);
    hasher.update(&[u8::from(receipt.activation_authorized)]);
    hasher.update(&[u8::from(receipt.trusted_state_mutated)]);
    hasher.update(&[u8::from(receipt.trusted_checkpoint_commit_authorized)]);
    Ok(ActivationTransactionReviewEligibilityDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

#[derive(Debug)]
pub enum ActivationTransactionReviewEligibilityError {
    PreflightRejected(ActivationPreflightError),
    TrustCurrentnessRejected(RestartTrustContextCurrentnessError),
    ReviewPredatesTrustCurrentness {
        reviewed_at_cycle: u64,
        verified_at_cycle: u64,
    },
    TrustCurrentnessExpired {
        reviewed_at_cycle: u64,
        expires_at_cycle: u64,
    },
    CandidateCurrentnessExpired {
        reviewed_at_cycle: u64,
        expires_at_cycle: u64,
    },
    TrustCheckpointExpired {
        reviewed_at_cycle: u64,
        expires_at_cycle: u64,
    },
    TrustCurrentnessCheckpointMismatch,
    UnexpectedPreflightClaims,
    UnexpectedTrustCurrentnessAuthority,
    ReceiptMismatch,
    ReceiptDigestMismatch,
}

impl fmt::Display for ActivationTransactionReviewEligibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "activation transaction review eligibility rejected: {self:?}"
        )
    }
}

impl Error for ActivationTransactionReviewEligibilityError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_binds_review_eligibility_without_swap_authority() {
        let mut receipt = ActivationTransactionReviewEligibilityReceiptV1 {
            version: ActivationTransactionReviewEligibilityVersion::V1,
            reviewed_at_cycle: 12,
            restart_capture_cycle: 7,
            fresh_preflight_digest: [1; 32],
            trust_context_currentness_statement_digest: [2; 32],
            trust_context_currentness_proof_digest: [3; 32],
            live_epoch_unchanged_at_review: true,
            candidate_current_head_proven_at_review: true,
            trust_context_current_head_proven_at_review: true,
            exact_trust_checkpoint_binding_proven: true,
            activation_transaction_review_eligible: true,
            live_state_lock_acquired: false,
            compare_and_swap_implemented: false,
            time_of_check_use_window_closed: false,
            live_state_swap_authorized: false,
            rollback_authorized: false,
            activation_authorized: false,
            trusted_state_mutated: false,
            trusted_checkpoint_commit_authorized: false,
            receipt_digest: ActivationTransactionReviewEligibilityDigestV1([0; 32]),
        };
        let first = digest_receipt(&receipt).unwrap();
        assert!(receipt.activation_transaction_review_eligible());
        assert!(!receipt.live_state_lock_acquired());
        assert!(!receipt.compare_and_swap_implemented());
        assert!(!receipt.time_of_check_use_window_closed());
        assert!(!receipt.live_state_swap_authorized());
        assert!(!receipt.activation_authorized());
        assert!(!receipt.trusted_checkpoint_commit_authorized());
        receipt.reviewed_at_cycle += 1;
        let second = digest_receipt(&receipt).unwrap();
        assert_ne!(first, second);
    }
}
