// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent post-mutation verification for EKM-026 belief updates.
//!
//! A successful mutation call is not itself proof that only the intended state
//! changed. This module snapshots the relevant isolated support-store and ledger
//! state before mutation, then checks the returned mutation receipt and resulting
//! state afterward.
//!
//! Verification is observational. It cannot repair, roll back, or authorize a
//! mutation.

use super::belief_mutation_firewall::{
    BeliefMutationAuthorization, BeliefMutationOutcome, BeliefMutationReceiptId,
    EpistemicSupportState, EpistemicSupportStore,
};
use super::belief_revision_receipt::BeliefRevisionReceipt;
use super::claim_evidence::{ClaimId, EpistemicLedger};

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefMutationSnapshot {
    claim_id: ClaimId,
    target_state: EpistemicSupportState,
    store_state_count: usize,
    mutation_history_len: usize,
    consumed_authorization_count: usize,
    ledger_claim_count: usize,
    ledger_evidence_count: usize,
    ledger_provenance_count: usize,
}

impl BeliefMutationSnapshot {
    pub fn capture(
        ledger: &EpistemicLedger,
        store: &EpistemicSupportStore,
        claim_id: ClaimId,
    ) -> Result<Self, BeliefMutationVerificationError> {
        let target_state = store
            .state(claim_id)
            .cloned()
            .ok_or(BeliefMutationVerificationError::StateNotRegistered(claim_id))?;
        Ok(Self {
            claim_id,
            target_state,
            store_state_count: store.len(),
            mutation_history_len: store.history().len(),
            consumed_authorization_count: store.consumed_authorization_count(),
            ledger_claim_count: ledger.claim_count(),
            ledger_evidence_count: ledger.evidence_count(),
            ledger_provenance_count: ledger.provenance_count(),
        })
    }

    pub fn claim_id(&self) -> ClaimId {
        self.claim_id
    }

    pub fn target_state(&self) -> &EpistemicSupportState {
        &self.target_state
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeliefMutationVerificationError {
    StateNotRegistered(ClaimId),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefMutationInvariantFailure {
    SourceReceiptClaimMismatch {
        snapshot_claim: ClaimId,
        receipt_claim: ClaimId,
    },
    MutationReceiptSourceMismatch,
    MutationReceiptClaimMismatch,
    MutationReceiptDeltaMismatch {
        expected: f32,
        actual: f32,
    },
    MutationReceiptAuthorizationMismatch,
    LedgerClaimCountChanged {
        before: usize,
        after: usize,
    },
    LedgerEvidenceCountChanged {
        before: usize,
        after: usize,
    },
    LedgerProvenanceCountChanged {
        before: usize,
        after: usize,
    },
    StoreStateCountChanged {
        before: usize,
        after: usize,
    },
    MutationHistoryLengthMismatch {
        expected: usize,
        actual: usize,
    },
    AuthorizationCountMismatch {
        minimum: usize,
        maximum: usize,
        actual: usize,
    },
    MissingTargetState(ClaimId),
    SupportBeforeMismatch {
        expected: f32,
        actual: f32,
    },
    SupportAfterMismatch {
        expected: f32,
        actual: f32,
    },
    StateRevisionBeforeMismatch {
        expected: u64,
        actual: u64,
    },
    StateRevisionAfterMismatch {
        expected: u64,
        actual: u64,
    },
    StateLastMutationMismatch {
        expected: BeliefMutationReceiptId,
        actual: Option<BeliefMutationReceiptId>,
    },
    StateLastUpdatedCycleMismatch {
        expected: u64,
        actual: u64,
    },
    AppliedMutationReceiptMissing(BeliefMutationReceiptId),
    AppliedMutationReceiptDuplicated(BeliefMutationReceiptId),
    ReplayStateRegressed {
        receipt_revision: u64,
        current_revision: u64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefMutationVerificationReport {
    failures: Vec<BeliefMutationInvariantFailure>,
}

impl BeliefMutationVerificationReport {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    pub fn failures(&self) -> &[BeliefMutationInvariantFailure] {
        &self.failures
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BeliefMutationVerifier;

impl BeliefMutationVerifier {
    pub fn verify(
        before: &BeliefMutationSnapshot,
        ledger: &EpistemicLedger,
        store: &EpistemicSupportStore,
        source_revision_receipt: &BeliefRevisionReceipt,
        authorization: &BeliefMutationAuthorization,
        outcome: &BeliefMutationOutcome,
    ) -> BeliefMutationVerificationReport {
        let mut failures = Vec::new();
        let mutation = outcome.receipt();

        if source_revision_receipt.claim_id() != before.claim_id {
            failures.push(BeliefMutationInvariantFailure::SourceReceiptClaimMismatch {
                snapshot_claim: before.claim_id,
                receipt_claim: source_revision_receipt.claim_id(),
            });
        }
        if mutation.source_revision_receipt_id() != source_revision_receipt.id() {
            failures.push(BeliefMutationInvariantFailure::MutationReceiptSourceMismatch);
        }
        if mutation.claim_id() != before.claim_id {
            failures.push(BeliefMutationInvariantFailure::MutationReceiptClaimMismatch);
        }
        if (mutation.proposed_delta() - source_revision_receipt.proposed_delta()).abs() > 1e-6 {
            failures.push(BeliefMutationInvariantFailure::MutationReceiptDeltaMismatch {
                expected: source_revision_receipt.proposed_delta(),
                actual: mutation.proposed_delta(),
            });
        }
        if mutation.authorization_id() != authorization.authorization_id()
            && outcome.applied_new_revision()
        {
            failures.push(BeliefMutationInvariantFailure::MutationReceiptAuthorizationMismatch);
        }

        if ledger.claim_count() != before.ledger_claim_count {
            failures.push(BeliefMutationInvariantFailure::LedgerClaimCountChanged {
                before: before.ledger_claim_count,
                after: ledger.claim_count(),
            });
        }
        if ledger.evidence_count() != before.ledger_evidence_count {
            failures.push(BeliefMutationInvariantFailure::LedgerEvidenceCountChanged {
                before: before.ledger_evidence_count,
                after: ledger.evidence_count(),
            });
        }
        if ledger.provenance_count() != before.ledger_provenance_count {
            failures.push(BeliefMutationInvariantFailure::LedgerProvenanceCountChanged {
                before: before.ledger_provenance_count,
                after: ledger.provenance_count(),
            });
        }
        if store.len() != before.store_state_count {
            failures.push(BeliefMutationInvariantFailure::StoreStateCountChanged {
                before: before.store_state_count,
                after: store.len(),
            });
        }

        let expected_history_len = before.mutation_history_len
            + usize::from(outcome.applied_new_revision());
        if store.history().len() != expected_history_len {
            failures.push(BeliefMutationInvariantFailure::MutationHistoryLengthMismatch {
                expected: expected_history_len,
                actual: store.history().len(),
            });
        }

        let (min_auth, max_auth) = if outcome.applied_new_revision() {
            (
                before.consumed_authorization_count + 1,
                before.consumed_authorization_count + 1,
            )
        } else {
            (
                before.consumed_authorization_count,
                before.consumed_authorization_count + 1,
            )
        };
        let actual_auth = store.consumed_authorization_count();
        if actual_auth < min_auth || actual_auth > max_auth {
            failures.push(BeliefMutationInvariantFailure::AuthorizationCountMismatch {
                minimum: min_auth,
                maximum: max_auth,
                actual: actual_auth,
            });
        }

        let current = store.state(before.claim_id);
        match current {
            None => failures.push(BeliefMutationInvariantFailure::MissingTargetState(
                before.claim_id,
            )),
            Some(current) if outcome.applied_new_revision() => {
                if mutation.support_before() != before.target_state.support() {
                    failures.push(BeliefMutationInvariantFailure::SupportBeforeMismatch {
                        expected: before.target_state.support().get(),
                        actual: mutation.support_before().get(),
                    });
                }
                let expected_after = before.target_state.support().get()
                    + source_revision_receipt.proposed_delta();
                if (mutation.support_after().get() - expected_after).abs() > 1e-6 {
                    failures.push(BeliefMutationInvariantFailure::SupportAfterMismatch {
                        expected: expected_after,
                        actual: mutation.support_after().get(),
                    });
                }
                if mutation.state_revision_before() != before.target_state.revision() {
                    failures.push(
                        BeliefMutationInvariantFailure::StateRevisionBeforeMismatch {
                            expected: before.target_state.revision(),
                            actual: mutation.state_revision_before(),
                        },
                    );
                }
                let expected_revision = before.target_state.revision() + 1;
                if mutation.state_revision_after() != expected_revision
                    || current.revision() != expected_revision
                {
                    failures.push(BeliefMutationInvariantFailure::StateRevisionAfterMismatch {
                        expected: expected_revision,
                        actual: current.revision(),
                    });
                }
                if current.support() != mutation.support_after() {
                    failures.push(BeliefMutationInvariantFailure::SupportAfterMismatch {
                        expected: mutation.support_after().get(),
                        actual: current.support().get(),
                    });
                }
                if current.last_mutation_id() != Some(mutation.id()) {
                    failures.push(BeliefMutationInvariantFailure::StateLastMutationMismatch {
                        expected: mutation.id(),
                        actual: current.last_mutation_id(),
                    });
                }
                if current.last_updated_cycle() != mutation.applied_at_cycle() {
                    failures.push(
                        BeliefMutationInvariantFailure::StateLastUpdatedCycleMismatch {
                            expected: mutation.applied_at_cycle(),
                            actual: current.last_updated_cycle(),
                        },
                    );
                }
            }
            Some(current) => {
                // An idempotent replay may occur after later legitimate revisions;
                // it must never move the state backwards or append another copy.
                if current.revision() < mutation.state_revision_after() {
                    failures.push(BeliefMutationInvariantFailure::ReplayStateRegressed {
                        receipt_revision: mutation.state_revision_after(),
                        current_revision: current.revision(),
                    });
                }
            }
        }

        let matching_history_count = store
            .history()
            .iter()
            .filter(|candidate| candidate.id() == mutation.id())
            .count();
        if matching_history_count == 0 {
            failures.push(BeliefMutationInvariantFailure::AppliedMutationReceiptMissing(
                mutation.id(),
            ));
        } else if matching_history_count > 1 {
            failures.push(
                BeliefMutationInvariantFailure::AppliedMutationReceiptDuplicated(mutation.id()),
            );
        }

        BeliefMutationVerificationReport { failures }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationAuthorizationDecision, BeliefMutationFirewall, BeliefRevisionHistory,
        BeliefRevisionPolicy, BoundedWeight, ClaimKind, EpistemicRevisionProposal,
        EvidenceKind, EvidencePolarity,
    };

    fn fixture() -> (
        EpistemicLedger,
        BeliefRevisionReceipt,
        EpistemicSupportStore,
        BeliefMutationAuthorization,
    ) {
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
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
            .unwrap();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        let mut history = BeliefRevisionHistory::new();
        let receipt_id = history
            .evaluate_and_record(&ledger, &proposal, &policy, None, None, 3)
            .unwrap();
        let receipt = history.get(receipt_id).unwrap().clone();
        let mut store = EpistemicSupportStore::new();
        store
            .register_claim(&ledger, claim, BoundedWeight::new(0.50).unwrap(), 3)
            .unwrap();
        let authorization = BeliefMutationAuthorization::new(
            "auth-1",
            "test-authority",
            BeliefMutationAuthorizationDecision::Approved,
            4,
            &receipt,
            store.state(claim).unwrap(),
        )
        .unwrap();
        (ledger, receipt, store, authorization)
    }

    #[test]
    fn clean_application_verifies() {
        let (ledger, receipt, mut store, authorization) = fixture();
        let before = BeliefMutationSnapshot::capture(&ledger, &store, receipt.claim_id()).unwrap();
        let mut firewall = BeliefMutationFirewall::new();
        let outcome = firewall
            .apply(&ledger, &mut store, &receipt, &authorization, 5)
            .unwrap();
        let report = BeliefMutationVerifier::verify(
            &before,
            &ledger,
            &store,
            &receipt,
            &authorization,
            &outcome,
        );
        assert!(report.passed(), "{:?}", report.failures());
    }

    #[test]
    fn idempotent_replay_verifies_without_second_history_entry() {
        let (ledger, receipt, mut store, authorization) = fixture();
        let mut firewall = BeliefMutationFirewall::new();
        firewall
            .apply(&ledger, &mut store, &receipt, &authorization, 5)
            .unwrap();
        let before = BeliefMutationSnapshot::capture(&ledger, &store, receipt.claim_id()).unwrap();
        let replay = firewall
            .apply(&ledger, &mut store, &receipt, &authorization, 6)
            .unwrap();
        let report = BeliefMutationVerifier::verify(
            &before,
            &ledger,
            &store,
            &receipt,
            &authorization,
            &replay,
        );
        assert!(report.passed(), "{:?}", report.failures());
        assert!(!replay.applied_new_revision());
    }

    #[test]
    fn unrelated_store_registration_after_mutation_is_detected() {
        let (mut ledger, receipt, mut store, authorization) = fixture();
        let before = BeliefMutationSnapshot::capture(&ledger, &store, receipt.claim_id()).unwrap();
        let mut firewall = BeliefMutationFirewall::new();
        let outcome = firewall
            .apply(&ledger, &mut store, &receipt, &authorization, 5)
            .unwrap();

        let other = ledger.add_claim("unrelated", ClaimKind::Descriptive, None, None, 6);
        store
            .register_claim(&ledger, other, BoundedWeight::new(0.40).unwrap(), 6)
            .unwrap();

        let report = BeliefMutationVerifier::verify(
            &before,
            &ledger,
            &store,
            &receipt,
            &authorization,
            &outcome,
        );
        assert!(!report.passed());
        assert!(report.failures().iter().any(|failure| matches!(
            failure,
            BeliefMutationInvariantFailure::LedgerClaimCountChanged { .. }
                | BeliefMutationInvariantFailure::StoreStateCountChanged { .. }
        )));
    }

    #[test]
    fn unrelated_ledger_evidence_after_mutation_is_detected() {
        let (mut ledger, receipt, mut store, authorization) = fixture();
        let before = BeliefMutationSnapshot::capture(&ledger, &store, receipt.claim_id()).unwrap();
        let mut firewall = BeliefMutationFirewall::new();
        let outcome = firewall
            .apply(&ledger, &mut store, &receipt, &authorization, 5)
            .unwrap();

        let provenance = ledger
            .add_provenance("unrelated", None, None, 6, vec![])
            .unwrap();
        let other = ledger.add_claim("other", ClaimKind::Descriptive, None, None, 6);
        ledger
            .add_evidence(
                other,
                EvidenceKind::Report,
                EvidencePolarity::Supports,
                provenance,
                6,
                None,
                None,
            )
            .unwrap();

        let report = BeliefMutationVerifier::verify(
            &before,
            &ledger,
            &store,
            &receipt,
            &authorization,
            &outcome,
        );
        assert!(!report.passed());
        assert!(report.failures().iter().any(|failure| matches!(
            failure,
            BeliefMutationInvariantFailure::LedgerClaimCountChanged { .. }
                | BeliefMutationInvariantFailure::LedgerEvidenceCountChanged { .. }
                | BeliefMutationInvariantFailure::LedgerProvenanceCountChanged { .. }
        )));
    }
}
