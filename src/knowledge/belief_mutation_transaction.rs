// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Full-evidence-census transaction wrapper for EKM belief mutation.
//!
//! EKM-026 rejects evidence whose *observation cycle* postdates a revision
//! decision, but that alone cannot detect a record inserted later while carrying
//! an older observation timestamp. This module closes that gap by sealing the
//! complete claim-evidence census in the same logical cycle as the EKM-025
//! revision receipt, then requiring exact census equality immediately before
//! mutation.
//!
//! The transaction coordinator does not add new mutation authority. It composes
//! the existing EKM-026 firewall with the EKM-027 independent verifier.

use super::belief_mutation_firewall::{
    BeliefMutationAuthorization, BeliefMutationError, BeliefMutationFirewall,
    BeliefMutationOutcome, EpistemicSupportStore,
};
use super::belief_mutation_verifier::{
    BeliefMutationSnapshot, BeliefMutationVerificationError, BeliefMutationVerificationReport,
    BeliefMutationVerifier,
};
use super::belief_revision_receipt::{
    BeliefRevisionReceipt, BeliefRevisionReceiptId, RevisionEvidenceSnapshot,
};
use super::claim_evidence::{ClaimId, ClaimKind, EpistemicLedger, EvidenceId};
use std::collections::HashSet;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealedClaimSnapshot {
    pub claim_id: ClaimId,
    pub statement: String,
    pub kind: ClaimKind,
    pub domain: Option<String>,
    pub scope: Option<String>,
    pub created_at_cycle: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeliefRevisionEvidenceSeal {
    source_revision_receipt_id: BeliefRevisionReceiptId,
    claim: SealedClaimSnapshot,
    evidence: Vec<RevisionEvidenceSnapshot>,
    sealed_at_cycle: u64,
}

impl BeliefRevisionEvidenceSeal {
    pub fn capture(
        ledger: &EpistemicLedger,
        receipt: &BeliefRevisionReceipt,
        sealed_at_cycle: u64,
    ) -> Result<Self, BeliefMutationSealError> {
        if !receipt.eligible() {
            return Err(BeliefMutationSealError::RevisionReceiptNotEligible(
                receipt.id(),
            ));
        }
        if sealed_at_cycle != receipt.evaluated_at_cycle() {
            return Err(BeliefMutationSealError::SealCycleDoesNotMatchEvaluation {
                sealed_at_cycle,
                evaluated_at_cycle: receipt.evaluated_at_cycle(),
            });
        }

        let claim = ledger
            .claim(receipt.claim_id())
            .ok_or(BeliefMutationSealError::UnknownClaim(receipt.claim_id()))?;
        let sealed_claim = SealedClaimSnapshot {
            claim_id: claim.id,
            statement: claim.statement.clone(),
            kind: claim.kind,
            domain: claim.domain.clone(),
            scope: claim.scope.clone(),
            created_at_cycle: claim.created_at_cycle,
        };

        let mut seen = HashSet::new();
        let mut evidence = Vec::with_capacity(claim.evidence_ids.len());
        for evidence_id in &claim.evidence_ids {
            if !seen.insert(*evidence_id) {
                return Err(BeliefMutationSealError::DuplicateClaimEvidenceId(
                    *evidence_id,
                ));
            }
            let record = ledger
                .evidence(*evidence_id)
                .ok_or(BeliefMutationSealError::MissingClaimEvidence(*evidence_id))?;
            if record.claim_id != claim.id {
                return Err(BeliefMutationSealError::EvidenceForDifferentClaim {
                    evidence_id: *evidence_id,
                    expected_claim: claim.id,
                    actual_claim: record.claim_id,
                });
            }
            if record.observed_at_cycle > sealed_at_cycle {
                return Err(BeliefMutationSealError::EvidencePostdatesSeal {
                    evidence_id: *evidence_id,
                    observed_at_cycle: record.observed_at_cycle,
                    sealed_at_cycle,
                });
            }
            evidence.push(snapshot(record));
        }
        evidence.sort_by_key(|record| record.evidence_id);

        // The receipt's explicit basis must be a semantic subset of the full
        // census being sealed.
        for reference in receipt.basis() {
            let expected = reference
                .snapshot
                .as_ref()
                .ok_or(BeliefMutationSealError::MissingReceiptBasisEvidence(
                    reference.requested_id,
                ))?;
            let live = evidence
                .iter()
                .find(|record| record.evidence_id == reference.requested_id)
                .ok_or(BeliefMutationSealError::MissingReceiptBasisEvidence(
                    reference.requested_id,
                ))?;
            if live != expected {
                return Err(BeliefMutationSealError::ReceiptBasisMismatch(
                    reference.requested_id,
                ));
            }
        }

        Ok(Self {
            source_revision_receipt_id: receipt.id(),
            claim: sealed_claim,
            evidence,
            sealed_at_cycle,
        })
    }

    pub fn source_revision_receipt_id(&self) -> BeliefRevisionReceiptId {
        self.source_revision_receipt_id
    }

    pub fn claim(&self) -> &SealedClaimSnapshot {
        &self.claim
    }

    pub fn evidence(&self) -> &[RevisionEvidenceSnapshot] {
        &self.evidence
    }

    pub fn sealed_at_cycle(&self) -> u64 {
        self.sealed_at_cycle
    }

    pub fn verify_live(
        &self,
        ledger: &EpistemicLedger,
        receipt: &BeliefRevisionReceipt,
    ) -> Result<(), BeliefMutationSealError> {
        if receipt.id() != self.source_revision_receipt_id
            || receipt.claim_id() != self.claim.claim_id
        {
            return Err(BeliefMutationSealError::SealReceiptMismatch);
        }

        let claim = ledger
            .claim(self.claim.claim_id)
            .ok_or(BeliefMutationSealError::UnknownClaim(self.claim.claim_id))?;
        let live_claim = SealedClaimSnapshot {
            claim_id: claim.id,
            statement: claim.statement.clone(),
            kind: claim.kind,
            domain: claim.domain.clone(),
            scope: claim.scope.clone(),
            created_at_cycle: claim.created_at_cycle,
        };
        if live_claim != self.claim {
            return Err(BeliefMutationSealError::ClaimChanged(self.claim.claim_id));
        }

        let mut live_ids = claim.evidence_ids.clone();
        live_ids.sort_unstable();
        let sealed_ids = self
            .evidence
            .iter()
            .map(|record| record.evidence_id)
            .collect::<Vec<_>>();
        if live_ids != sealed_ids {
            return Err(BeliefMutationSealError::EvidenceCensusChanged {
                sealed: sealed_ids,
                live: live_ids,
            });
        }

        for expected in &self.evidence {
            let live = ledger
                .evidence(expected.evidence_id)
                .ok_or(BeliefMutationSealError::MissingClaimEvidence(
                    expected.evidence_id,
                ))?;
            if snapshot(live) != *expected {
                return Err(BeliefMutationSealError::EvidenceRecordChanged(
                    expected.evidence_id,
                ));
            }
        }
        Ok(())
    }
}

fn snapshot(record: &super::claim_evidence::EvidenceRecord) -> RevisionEvidenceSnapshot {
    RevisionEvidenceSnapshot {
        evidence_id: record.id,
        claim_id: record.claim_id,
        kind: record.kind,
        polarity: record.polarity,
        provenance_id: record.provenance_id,
        observed_at_cycle: record.observed_at_cycle,
        context: record.context.clone(),
        method: record.method.clone(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeliefMutationSealError {
    RevisionReceiptNotEligible(BeliefRevisionReceiptId),
    SealCycleDoesNotMatchEvaluation {
        sealed_at_cycle: u64,
        evaluated_at_cycle: u64,
    },
    UnknownClaim(ClaimId),
    DuplicateClaimEvidenceId(EvidenceId),
    MissingClaimEvidence(EvidenceId),
    MissingReceiptBasisEvidence(EvidenceId),
    ReceiptBasisMismatch(EvidenceId),
    EvidenceForDifferentClaim {
        evidence_id: EvidenceId,
        expected_claim: ClaimId,
        actual_claim: ClaimId,
    },
    EvidencePostdatesSeal {
        evidence_id: EvidenceId,
        observed_at_cycle: u64,
        sealed_at_cycle: u64,
    },
    SealReceiptMismatch,
    ClaimChanged(ClaimId),
    EvidenceCensusChanged {
        sealed: Vec<EvidenceId>,
        live: Vec<EvidenceId>,
    },
    EvidenceRecordChanged(EvidenceId),
}

impl fmt::Display for BeliefMutationSealError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "belief mutation evidence seal failed: {self:?}")
    }
}

impl Error for BeliefMutationSealError {}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefMutationTransactionError {
    Seal(BeliefMutationSealError),
    Snapshot(BeliefMutationVerificationError),
    Mutation(BeliefMutationError),
}

impl fmt::Display for BeliefMutationTransactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Seal(error) => write!(f, "{error}"),
            Self::Snapshot(error) => write!(f, "belief mutation snapshot failed: {error:?}"),
            Self::Mutation(error) => write!(f, "belief mutation failed: {error}"),
        }
    }
}

impl Error for BeliefMutationTransactionError {}

impl From<BeliefMutationSealError> for BeliefMutationTransactionError {
    fn from(value: BeliefMutationSealError) -> Self {
        Self::Seal(value)
    }
}

impl From<BeliefMutationVerificationError> for BeliefMutationTransactionError {
    fn from(value: BeliefMutationVerificationError) -> Self {
        Self::Snapshot(value)
    }
}

impl From<BeliefMutationError> for BeliefMutationTransactionError {
    fn from(value: BeliefMutationError) -> Self {
        Self::Mutation(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefMutationTransactionOutcome {
    mutation: BeliefMutationOutcome,
    verification: BeliefMutationVerificationReport,
}

impl BeliefMutationTransactionOutcome {
    pub fn mutation(&self) -> &BeliefMutationOutcome {
        &self.mutation
    }

    pub fn verification(&self) -> &BeliefMutationVerificationReport {
        &self.verification
    }

    pub fn verified(&self) -> bool {
        self.verification.passed()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BeliefMutationTransactionCoordinator;

impl BeliefMutationTransactionCoordinator {
    pub fn apply(
        seal: &BeliefRevisionEvidenceSeal,
        ledger: &EpistemicLedger,
        store: &mut EpistemicSupportStore,
        revision_receipt: &BeliefRevisionReceipt,
        authorization: &BeliefMutationAuthorization,
        firewall: &mut BeliefMutationFirewall,
        mutation_cycle: u64,
    ) -> Result<BeliefMutationTransactionOutcome, BeliefMutationTransactionError> {
        // All checks before this point are non-mutating. A seal mismatch must not
        // partially change support state.
        seal.verify_live(ledger, revision_receipt)?;
        let before = BeliefMutationSnapshot::capture(
            ledger,
            store,
            revision_receipt.claim_id(),
        )?;

        let mutation = firewall.apply(
            ledger,
            store,
            revision_receipt,
            authorization,
            mutation_cycle,
        )?;

        // A verification failure occurs after the mutation and is therefore
        // returned as data, not converted into an error that might imply nothing
        // changed. Higher authority must halt/escalate; this layer never repairs.
        let verification = BeliefMutationVerifier::verify(
            &before,
            ledger,
            store,
            revision_receipt,
            authorization,
            &mutation,
        );

        Ok(BeliefMutationTransactionOutcome {
            mutation,
            verification,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationAuthorizationDecision, BeliefRevisionHistory, BeliefRevisionPolicy,
        BoundedWeight, ClaimKind, EpistemicRevisionProposal, EvidenceKind, EvidencePolarity,
    };

    fn fixture() -> (
        EpistemicLedger,
        BeliefRevisionReceipt,
        BeliefRevisionEvidenceSeal,
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
        let seal = BeliefRevisionEvidenceSeal::capture(&ledger, &receipt, 3).unwrap();
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
        (ledger, receipt, seal, store, authorization)
    }

    #[test]
    fn clean_sealed_transaction_applies_and_verifies() {
        let (ledger, receipt, seal, mut store, authorization) = fixture();
        let mut firewall = BeliefMutationFirewall::new();
        let outcome = BeliefMutationTransactionCoordinator::apply(
            &seal,
            &ledger,
            &mut store,
            &receipt,
            &authorization,
            &mut firewall,
            5,
        )
        .unwrap();
        assert!(outcome.mutation().applied_new_revision());
        assert!(outcome.verified(), "{:?}", outcome.verification().failures());
    }

    #[test]
    fn late_inserted_old_dated_evidence_invalidates_seal_before_mutation() {
        let (mut ledger, receipt, seal, mut store, authorization) = fixture();
        let provenance = ledger
            .add_provenance("late-ingest", None, None, 4, vec![])
            .unwrap();
        // Inserted after the seal but intentionally carrying an old observation
        // cycle. EKM-026's cycle-only freshness check cannot distinguish this;
        // the full census seal must.
        ledger
            .add_evidence(
                receipt.claim_id(),
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                provenance,
                2,
                Some("late old-dated record".into()),
                Some("protocol-v0".into()),
            )
            .unwrap();

        let mut firewall = BeliefMutationFirewall::new();
        let error = BeliefMutationTransactionCoordinator::apply(
            &seal,
            &ledger,
            &mut store,
            &receipt,
            &authorization,
            &mut firewall,
            5,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BeliefMutationTransactionError::Seal(
                BeliefMutationSealError::EvidenceCensusChanged { .. }
            )
        ));
        assert!(store.history().is_empty());
        assert!((store.state(receipt.claim_id()).unwrap().support().get() - 0.50).abs() < 1e-6);
    }

    #[test]
    fn seal_must_be_created_in_same_logical_cycle_as_revision_decision() {
        let (ledger, receipt, _, _, _) = fixture();
        assert_eq!(
            BeliefRevisionEvidenceSeal::capture(&ledger, &receipt, 4).unwrap_err(),
            BeliefMutationSealError::SealCycleDoesNotMatchEvaluation {
                sealed_at_cycle: 4,
                evaluated_at_cycle: 3,
            }
        );
    }

    #[test]
    fn seal_is_bound_to_exact_revision_receipt() {
        let (ledger, receipt, seal, _, _) = fixture();
        let support = ledger.claim(receipt.claim_id()).unwrap().evidence_ids[0];
        let proposal = EpistemicRevisionProposal::new(
            receipt.claim_id(),
            0.05,
            vec![support],
            "second revision decision",
        )
        .unwrap();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        let mut history = BeliefRevisionHistory::new();
        let second_id = history
            .evaluate_and_record(&ledger, &proposal, &policy, None, None, 3)
            .unwrap();
        let second = history.get(second_id).unwrap();
        assert_eq!(
            seal.verify_live(&ledger, second).unwrap_err(),
            BeliefMutationSealError::SealReceiptMismatch
        );
    }
}
