// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Read-only persistence contract for EKM belief-revision decision history.
//!
//! EKM-030 captures applied epistemic-support mutations, but restart safety also
//! requires the full EKM-025 decision lineage: eligible and rejected decisions,
//! their immutable evidence/policy/calibration/uncertainty snapshots, and the next
//! receipt identifier. Otherwise a restarted process could reuse a historical
//! `BeliefRevisionReceiptId` for a different decision.
//!
//! This module exports and validates that lineage. It deliberately does not
//! hydrate [`BeliefRevisionHistory`] and performs no file/database I/O.

use super::belief_mutation_persistence::BeliefMutationPersistenceCapsuleV1;
use super::belief_revision_receipt::{
    BeliefRevisionHistory, BeliefRevisionReceipt, BeliefRevisionReceiptId,
};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefRevisionPersistenceVersion {
    V1,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefRevisionHistoryCapsuleV1 {
    version: BeliefRevisionPersistenceVersion,
    captured_at_cycle: u64,
    receipts: Vec<BeliefRevisionReceipt>,
    next_receipt_id: BeliefRevisionReceiptId,
    linked_mutation_capture_cycle: u64,
    linked_mutation_count: usize,
}

impl BeliefRevisionHistoryCapsuleV1 {
    /// Capture the complete append-only decision history and bind it to one
    /// already-validated EKM-030 mutation capsule.
    pub fn capture(
        history: &BeliefRevisionHistory,
        mutation_capsule: &BeliefMutationPersistenceCapsuleV1,
        captured_at_cycle: u64,
    ) -> Result<Self, BeliefRevisionPersistenceError> {
        if mutation_capsule.captured_at_cycle() > captured_at_cycle {
            return Err(BeliefRevisionPersistenceError::MutationCapsulePostdatesCapture {
                mutation_capture_cycle: mutation_capsule.captured_at_cycle(),
                revision_capture_cycle: captured_at_cycle,
            });
        }

        let receipts = history.receipts().to_vec();
        let next_receipt_id = validate_receipt_chain(&receipts, captured_at_cycle)?;
        validate_mutation_links(&receipts, mutation_capsule)?;

        Ok(Self {
            version: BeliefRevisionPersistenceVersion::V1,
            captured_at_cycle,
            receipts,
            next_receipt_id,
            linked_mutation_capture_cycle: mutation_capsule.captured_at_cycle(),
            linked_mutation_count: mutation_capsule.mutations().len(),
        })
    }

    pub fn version(&self) -> BeliefRevisionPersistenceVersion {
        self.version
    }

    pub fn captured_at_cycle(&self) -> u64 {
        self.captured_at_cycle
    }

    pub fn receipts(&self) -> &[BeliefRevisionReceipt] {
        &self.receipts
    }

    pub fn next_receipt_id(&self) -> BeliefRevisionReceiptId {
        self.next_receipt_id
    }

    pub fn linked_mutation_capture_cycle(&self) -> u64 {
        self.linked_mutation_capture_cycle
    }

    pub fn linked_mutation_count(&self) -> usize {
        self.linked_mutation_count
    }

    /// Validate that a still-live history and EKM-030 capsule remain exactly
    /// represented by this decision-history capsule.
    pub fn validate_live(
        &self,
        history: &BeliefRevisionHistory,
        mutation_capsule: &BeliefMutationPersistenceCapsuleV1,
        observed_at_cycle: u64,
    ) -> Result<(), BeliefRevisionPersistenceError> {
        if observed_at_cycle < self.captured_at_cycle {
            return Err(BeliefRevisionPersistenceError::ObservationPredatesCapsule {
                observed_at_cycle,
                captured_at_cycle: self.captured_at_cycle,
            });
        }

        let live = Self::capture(history, mutation_capsule, observed_at_cycle)?;
        if live.receipts != self.receipts {
            return Err(BeliefRevisionPersistenceError::LiveDecisionHistoryMismatch);
        }
        if live.next_receipt_id != self.next_receipt_id {
            return Err(BeliefRevisionPersistenceError::LiveNextReceiptIdMismatch {
                expected: self.next_receipt_id,
                actual: live.next_receipt_id,
            });
        }
        if live.linked_mutation_capture_cycle != self.linked_mutation_capture_cycle
            || live.linked_mutation_count != self.linked_mutation_count
        {
            return Err(BeliefRevisionPersistenceError::LinkedMutationCapsuleMismatch);
        }
        Ok(())
    }
}

fn validate_receipt_chain(
    receipts: &[BeliefRevisionReceipt],
    captured_at_cycle: u64,
) -> Result<BeliefRevisionReceiptId, BeliefRevisionPersistenceError> {
    let mut expected = 1u64;
    let mut previous_evaluation_cycle = None;

    for receipt in receipts {
        if receipt.id().0 != expected {
            return Err(BeliefRevisionPersistenceError::ReceiptIdSequenceBroken {
                expected: BeliefRevisionReceiptId(expected),
                actual: receipt.id(),
            });
        }
        if receipt.evaluated_at_cycle() > captured_at_cycle {
            return Err(BeliefRevisionPersistenceError::CapturePredatesDecision {
                receipt_id: receipt.id(),
                captured_at_cycle,
                evaluated_at_cycle: receipt.evaluated_at_cycle(),
            });
        }
        if let Some(previous) = previous_evaluation_cycle {
            if receipt.evaluated_at_cycle() < previous {
                return Err(BeliefRevisionPersistenceError::DecisionCyclesRegressed {
                    previous_cycle: previous,
                    current_receipt: receipt.id(),
                    current_cycle: receipt.evaluated_at_cycle(),
                });
            }
        }
        previous_evaluation_cycle = Some(receipt.evaluated_at_cycle());
        expected = expected
            .checked_add(1)
            .ok_or(BeliefRevisionPersistenceError::ReceiptIdExhausted)?;
    }

    Ok(BeliefRevisionReceiptId(expected))
}

fn validate_mutation_links(
    receipts: &[BeliefRevisionReceipt],
    mutation_capsule: &BeliefMutationPersistenceCapsuleV1,
) -> Result<(), BeliefRevisionPersistenceError> {
    for mutation in mutation_capsule.mutations() {
        let receipt = receipts
            .iter()
            .find(|receipt| receipt.id() == mutation.source_revision_receipt_id)
            .ok_or(BeliefRevisionPersistenceError::MutationMissingDecision {
                mutation_id: mutation.id,
                source_receipt_id: mutation.source_revision_receipt_id,
            })?;

        if !receipt.eligible() {
            return Err(BeliefRevisionPersistenceError::MutationReferencesRejectedDecision {
                mutation_id: mutation.id,
                source_receipt_id: mutation.source_revision_receipt_id,
            });
        }
        if receipt.claim_id() != mutation.claim_id {
            return Err(BeliefRevisionPersistenceError::MutationDecisionClaimMismatch {
                mutation_id: mutation.id,
                receipt_claim: receipt.claim_id(),
                mutation_claim: mutation.claim_id,
            });
        }
        if (receipt.proposed_delta() - mutation.proposed_delta).abs() > 1e-6 {
            return Err(BeliefRevisionPersistenceError::MutationDecisionDeltaMismatch {
                mutation_id: mutation.id,
                receipt_delta: receipt.proposed_delta(),
                mutation_delta: mutation.proposed_delta,
            });
        }
        if mutation.authorized_at_cycle < receipt.evaluated_at_cycle() {
            return Err(BeliefRevisionPersistenceError::AuthorizationPredatesDecision {
                mutation_id: mutation.id,
                receipt_id: receipt.id(),
                evaluated_at_cycle: receipt.evaluated_at_cycle(),
                authorized_at_cycle: mutation.authorized_at_cycle,
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefRevisionPersistenceError {
    MutationCapsulePostdatesCapture {
        mutation_capture_cycle: u64,
        revision_capture_cycle: u64,
    },
    ReceiptIdSequenceBroken {
        expected: BeliefRevisionReceiptId,
        actual: BeliefRevisionReceiptId,
    },
    ReceiptIdExhausted,
    CapturePredatesDecision {
        receipt_id: BeliefRevisionReceiptId,
        captured_at_cycle: u64,
        evaluated_at_cycle: u64,
    },
    DecisionCyclesRegressed {
        previous_cycle: u64,
        current_receipt: BeliefRevisionReceiptId,
        current_cycle: u64,
    },
    MutationMissingDecision {
        mutation_id: super::belief_mutation_firewall::BeliefMutationReceiptId,
        source_receipt_id: BeliefRevisionReceiptId,
    },
    MutationReferencesRejectedDecision {
        mutation_id: super::belief_mutation_firewall::BeliefMutationReceiptId,
        source_receipt_id: BeliefRevisionReceiptId,
    },
    MutationDecisionClaimMismatch {
        mutation_id: super::belief_mutation_firewall::BeliefMutationReceiptId,
        receipt_claim: super::claim_evidence::ClaimId,
        mutation_claim: super::claim_evidence::ClaimId,
    },
    MutationDecisionDeltaMismatch {
        mutation_id: super::belief_mutation_firewall::BeliefMutationReceiptId,
        receipt_delta: f32,
        mutation_delta: f32,
    },
    AuthorizationPredatesDecision {
        mutation_id: super::belief_mutation_firewall::BeliefMutationReceiptId,
        receipt_id: BeliefRevisionReceiptId,
        evaluated_at_cycle: u64,
        authorized_at_cycle: u64,
    },
    ObservationPredatesCapsule {
        observed_at_cycle: u64,
        captured_at_cycle: u64,
    },
    LiveDecisionHistoryMismatch,
    LiveNextReceiptIdMismatch {
        expected: BeliefRevisionReceiptId,
        actual: BeliefRevisionReceiptId,
    },
    LinkedMutationCapsuleMismatch,
}

impl fmt::Display for BeliefRevisionPersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "belief revision persistence capsule invalid: {self:?}")
    }
}

impl Error for BeliefRevisionPersistenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationAuthority, BeliefMutationAuthorization,
        BeliefMutationAuthorizationDecision, BeliefMutationPersistenceCapsuleV1,
        BeliefRevisionPolicy, BoundedWeight, ClaimKind, EpistemicLedger,
        EpistemicRevisionProposal, EpistemicSupportStore, EvidenceKind, EvidencePolarity,
    };

    fn ledger_fixture() -> (EpistemicLedger, super::super::claim_evidence::ClaimId, super::super::claim_evidence::EvidenceId) {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("lab", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                provenance,
                2,
                Some("fixture".into()),
                Some("protocol-v1".into()),
            )
            .unwrap();
        (ledger, claim, evidence)
    }

    #[test]
    fn preserves_eligible_and_rejected_decisions_and_next_id() {
        let (ledger, claim, evidence) = ledger_fixture();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        let mut history = BeliefRevisionHistory::new();

        let eligible = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
            .unwrap();
        history
            .evaluate_and_record(&ledger, &eligible, &policy, None, None, 3)
            .unwrap();
        let rejected = EpistemicRevisionProposal::new(
            claim,
            -0.10,
            vec![evidence],
            "wrong direction",
        )
        .unwrap();
        history
            .evaluate_and_record(&ledger, &rejected, &policy, None, None, 4)
            .unwrap();

        let mut store = EpistemicSupportStore::new();
        store
            .register_claim(&ledger, claim, BoundedWeight::new(0.50).unwrap(), 2)
            .unwrap();
        let mutations =
            BeliefMutationPersistenceCapsuleV1::capture(&store, &[claim], 5).unwrap();
        let capsule = BeliefRevisionHistoryCapsuleV1::capture(&history, &mutations, 5).unwrap();

        assert_eq!(capsule.receipts().len(), 2);
        assert!(capsule.receipts()[0].eligible());
        assert!(!capsule.receipts()[1].eligible());
        assert_eq!(capsule.next_receipt_id(), BeliefRevisionReceiptId(3));
        capsule.validate_live(&history, &mutations, 6).unwrap();
    }

    #[test]
    fn applied_mutation_must_link_to_eligible_decision() {
        let (ledger, claim, evidence) = ledger_fixture();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
            .unwrap();
        let mut history = BeliefRevisionHistory::new();
        let mut authority = BeliefMutationAuthority::new();
        let prepared = authority
            .prepare(&ledger, &mut history, &proposal, &policy, None, None, 3)
            .unwrap();
        let mut store = EpistemicSupportStore::new();
        store
            .register_claim(&ledger, claim, BoundedWeight::new(0.50).unwrap(), 3)
            .unwrap();
        let authorization = BeliefMutationAuthorization::new(
            "auth-1",
            "test-authority",
            BeliefMutationAuthorizationDecision::Approved,
            4,
            prepared.receipt(),
            store.state(claim).unwrap(),
        )
        .unwrap();
        authority
            .apply(&ledger, &mut store, &prepared, &authorization, 5)
            .unwrap();

        let mutations =
            BeliefMutationPersistenceCapsuleV1::capture(&store, &[claim], 6).unwrap();
        let capsule = BeliefRevisionHistoryCapsuleV1::capture(&history, &mutations, 6).unwrap();
        assert_eq!(capsule.linked_mutation_count(), 1);
        assert_eq!(capsule.next_receipt_id(), BeliefRevisionReceiptId(2));
    }

    #[test]
    fn mutation_without_its_decision_lineage_fails_closed() {
        let (ledger, claim, evidence) = ledger_fixture();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
            .unwrap();
        let mut complete_history = BeliefRevisionHistory::new();
        let mut authority = BeliefMutationAuthority::new();
        let prepared = authority
            .prepare(
                &ledger,
                &mut complete_history,
                &proposal,
                &policy,
                None,
                None,
                3,
            )
            .unwrap();
        let mut store = EpistemicSupportStore::new();
        store
            .register_claim(&ledger, claim, BoundedWeight::new(0.50).unwrap(), 3)
            .unwrap();
        let authorization = BeliefMutationAuthorization::new(
            "auth-1",
            "test-authority",
            BeliefMutationAuthorizationDecision::Approved,
            4,
            prepared.receipt(),
            store.state(claim).unwrap(),
        )
        .unwrap();
        authority
            .apply(&ledger, &mut store, &prepared, &authorization, 5)
            .unwrap();
        let mutations =
            BeliefMutationPersistenceCapsuleV1::capture(&store, &[claim], 6).unwrap();

        let empty_history = BeliefRevisionHistory::new();
        assert!(matches!(
            BeliefRevisionHistoryCapsuleV1::capture(&empty_history, &mutations, 6),
            Err(BeliefRevisionPersistenceError::MutationMissingDecision { .. })
        ));
    }

    #[test]
    fn later_decision_makes_older_capsule_stale() {
        let (ledger, claim, evidence) = ledger_fixture();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
            .unwrap();
        let mut history = BeliefRevisionHistory::new();
        history
            .evaluate_and_record(&ledger, &proposal, &policy, None, None, 3)
            .unwrap();
        let mut store = EpistemicSupportStore::new();
        store
            .register_claim(&ledger, claim, BoundedWeight::new(0.50).unwrap(), 2)
            .unwrap();
        let mutations =
            BeliefMutationPersistenceCapsuleV1::capture(&store, &[claim], 3).unwrap();
        let capsule = BeliefRevisionHistoryCapsuleV1::capture(&history, &mutations, 3).unwrap();

        history
            .evaluate_and_record(&ledger, &proposal, &policy, None, None, 4)
            .unwrap();
        assert!(matches!(
            capsule.validate_live(&history, &mutations, 5),
            Err(BeliefRevisionPersistenceError::LiveDecisionHistoryMismatch)
        ));
    }
}
