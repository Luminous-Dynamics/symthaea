// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Read-only rollback/equivocation checks for validated restart receipts.
//!
//! A semantically valid restart receipt can still be stale. This module compares
//! a candidate EKM-044 receipt against a caller-supplied trusted checkpoint and
//! classifies idempotent replay, forward progress, rollback, or same-epoch
//! equivocation. It grants no restore, quarantine, or activation authority.

#[path = "epistemic_restart_live_epoch_fence.rs"]
pub mod live_epoch_fence;
#[path = "epistemic_restart_activation_preflight.rs"]
pub mod activation_preflight;
#[path = "epistemic_restart_trust_checkpoint_currentness.rs"]
pub mod trust_checkpoint_currentness;
#[path = "epistemic_restart_activation_review_eligibility.rs"]
pub mod activation_review_eligibility;

use super::epistemic_restart_validation_receipt::{
    EpistemicRestartValidationReceiptDigest, EpistemicRestartValidationReceiptV1,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrustedRestartValidationAnchorV1 {
    captured_at_cycle: u64,
    receipt_digest: EpistemicRestartValidationReceiptDigest,
}

impl TrustedRestartValidationAnchorV1 {
    /// Create an anchor only from an already validated EKM-044 receipt.
    ///
    /// The caller remains responsible for storing this anchor in a location whose
    /// rollback properties are appropriate for the deployment. This type does not
    /// itself provide durable or tamper-resistant storage.
    pub fn from_receipt(receipt: &EpistemicRestartValidationReceiptV1) -> Self {
        Self {
            captured_at_cycle: receipt.captured_at_cycle(),
            receipt_digest: receipt.receipt_digest(),
        }
    }

    /// Crate-internal bridge for EKM-046 after external anchor evidence has
    /// already been verified. Public callers cannot use this to mint anchors.
    pub(crate) fn from_verified_parts(
        captured_at_cycle: u64,
        receipt_digest: EpistemicRestartValidationReceiptDigest,
    ) -> Self {
        Self {
            captured_at_cycle,
            receipt_digest,
        }
    }

    pub fn captured_at_cycle(self) -> u64 {
        self.captured_at_cycle
    }

    pub fn receipt_digest(self) -> EpistemicRestartValidationReceiptDigest {
        self.receipt_digest
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartContinuityDispositionV1 {
    /// The exact same validated state is being presented again.
    IdempotentReplay,
    /// A different validated state advances beyond the trusted capture cycle.
    ForwardProgress,
    /// The candidate predates the trusted checkpoint.
    Rollback,
    /// A different validated state claims the same logical capture cycle.
    SameCycleEquivocation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartContinuityDecisionV1 {
    disposition: RestartContinuityDispositionV1,
    candidate_cycle: u64,
    anchor_cycle: u64,
    candidate_digest: EpistemicRestartValidationReceiptDigest,
    anchor_digest: EpistemicRestartValidationReceiptDigest,
    further_review_eligible: bool,
    quarantine_construction_authorized: bool,
    activation_authorized: bool,
}

impl RestartContinuityDecisionV1 {
    pub fn disposition(self) -> RestartContinuityDispositionV1 {
        self.disposition
    }

    pub fn candidate_cycle(self) -> u64 {
        self.candidate_cycle
    }

    pub fn anchor_cycle(self) -> u64 {
        self.anchor_cycle
    }

    pub fn candidate_digest(self) -> EpistemicRestartValidationReceiptDigest {
        self.candidate_digest
    }

    pub fn anchor_digest(self) -> EpistemicRestartValidationReceiptDigest {
        self.anchor_digest
    }

    /// True only for an exact replay or monotonic forward progress. This means
    /// only "eligible for the next review layer"; it is not restore authority.
    pub fn further_review_eligible(self) -> bool {
        self.further_review_eligible
    }

    pub fn quarantine_construction_authorized(self) -> bool {
        self.quarantine_construction_authorized
    }

    pub fn activation_authorized(self) -> bool {
        self.activation_authorized
    }
}

pub struct RestartContinuityGateV1;

impl RestartContinuityGateV1 {
    pub fn evaluate(
        anchor: TrustedRestartValidationAnchorV1,
        candidate: &EpistemicRestartValidationReceiptV1,
    ) -> RestartContinuityDecisionV1 {
        let candidate_cycle = candidate.captured_at_cycle();
        let candidate_digest = candidate.receipt_digest();
        let anchor_cycle = anchor.captured_at_cycle;
        let anchor_digest = anchor.receipt_digest;

        let disposition = if candidate_digest == anchor_digest {
            RestartContinuityDispositionV1::IdempotentReplay
        } else if candidate_cycle < anchor_cycle {
            RestartContinuityDispositionV1::Rollback
        } else if candidate_cycle == anchor_cycle {
            RestartContinuityDispositionV1::SameCycleEquivocation
        } else {
            RestartContinuityDispositionV1::ForwardProgress
        };

        let further_review_eligible = matches!(
            disposition,
            RestartContinuityDispositionV1::IdempotentReplay
                | RestartContinuityDispositionV1::ForwardProgress
        );

        RestartContinuityDecisionV1 {
            disposition,
            candidate_cycle,
            anchor_cycle,
            candidate_digest,
            anchor_digest,
            further_review_eligible,
            quarantine_construction_authorized: false,
            activation_authorized: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistory,
        BeliefRevisionHistoryCapsuleV1, BeliefRevisionPolicySchemaV1,
        BeliefRevisionSchemaHistoryCapsuleV1, BeliefRevisionSchemaHistoryV1, ClaimKind,
        EpistemicLedger, EpistemicLedgerInventoryV1, EpistemicRestartCapsuleV1,
        EpistemicRestartCapsuleV2, EpistemicRestartValidationReceiptV1, EpistemicRestartWireV2,
        EpistemicRevisionProposal, EpistemicSupportStore, EvidenceKind, EvidencePolarity,
    };

    fn receipt(capture_cycle: u64, statement: &str) -> EpistemicRestartValidationReceiptV1 {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("lab", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim(statement, ClaimKind::Predictive, None, None, 1);
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                provenance,
                2,
                None,
                None,
            )
            .unwrap();
        let inventory = EpistemicLedgerInventoryV1::new(
            vec![claim],
            vec![evidence],
            vec![provenance],
        )
        .unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.1, vec![evidence], "measurement")
            .unwrap();
        let schema = BeliefRevisionPolicySchemaV1::new(0.2, 1, false, 0, 1.0).unwrap();
        let mut receipts = BeliefRevisionHistory::new();
        let mut schema_history = BeliefRevisionSchemaHistoryV1::new();
        schema_history
            .evaluate_and_record(
                &mut receipts,
                &ledger,
                &proposal,
                &schema,
                None,
                None,
                3,
            )
            .unwrap();
        let store = EpistemicSupportStore::new();
        let mutations =
            BeliefMutationPersistenceCapsuleV1::capture(&store, &[], capture_cycle).unwrap();
        let revisions =
            BeliefRevisionHistoryCapsuleV1::capture(&receipts, &mutations, capture_cycle).unwrap();
        let schemas = BeliefRevisionSchemaHistoryCapsuleV1::capture(
            &schema_history,
            &receipts,
            &revisions,
            capture_cycle,
        )
        .unwrap();
        let base = EpistemicRestartCapsuleV1::capture(
            &ledger,
            &inventory,
            &mutations,
            &revisions,
            capture_cycle,
        )
        .unwrap();
        let v2 = EpistemicRestartCapsuleV2::capture(&base, &schemas).unwrap();
        let bytes = EpistemicRestartWireV2::encode(&v2).unwrap();
        let snapshot = EpistemicRestartWireV2::decode(&bytes).unwrap();
        EpistemicRestartValidationReceiptV1::validate_and_capture(&snapshot).unwrap()
    }

    #[test]
    fn exact_receipt_is_idempotent_replay() {
        let current = receipt(4, "X predicts Y");
        let anchor = TrustedRestartValidationAnchorV1::from_receipt(&current);
        let decision = RestartContinuityGateV1::evaluate(anchor, &current);
        assert_eq!(
            decision.disposition(),
            RestartContinuityDispositionV1::IdempotentReplay
        );
        assert!(decision.further_review_eligible());
        assert!(!decision.quarantine_construction_authorized());
        assert!(!decision.activation_authorized());
    }

    #[test]
    fn older_valid_receipt_is_rollback() {
        let old = receipt(4, "X predicts Y");
        let current = receipt(5, "X predicts Y");
        let anchor = TrustedRestartValidationAnchorV1::from_receipt(&current);
        let decision = RestartContinuityGateV1::evaluate(anchor, &old);
        assert_eq!(decision.disposition(), RestartContinuityDispositionV1::Rollback);
        assert!(!decision.further_review_eligible());
    }

    #[test]
    fn different_receipt_same_cycle_is_equivocation() {
        let left = receipt(4, "X predicts Y");
        let right = receipt(4, "X predicts Z");
        let anchor = TrustedRestartValidationAnchorV1::from_receipt(&left);
        let decision = RestartContinuityGateV1::evaluate(anchor, &right);
        assert_eq!(
            decision.disposition(),
            RestartContinuityDispositionV1::SameCycleEquivocation
        );
        assert!(!decision.further_review_eligible());
    }

    #[test]
    fn newer_different_receipt_is_forward_progress_only() {
        let old = receipt(4, "X predicts Y");
        let newer = receipt(5, "X predicts Z");
        let anchor = TrustedRestartValidationAnchorV1::from_receipt(&old);
        let decision = RestartContinuityGateV1::evaluate(anchor, &newer);
        assert_eq!(
            decision.disposition(),
            RestartContinuityDispositionV1::ForwardProgress
        );
        assert!(decision.further_review_eligible());
        assert!(!decision.quarantine_construction_authorized());
        assert!(!decision.activation_authorized());
    }
}
