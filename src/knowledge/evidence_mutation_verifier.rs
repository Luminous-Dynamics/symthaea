// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Post-mutation verification for evidence ingestion.
//!
//! The mutation firewall is intentionally narrow, but a successful call should
//! still be independently checked. This module snapshots the relevant ledger
//! state before ingestion and verifies afterward that only the expected evidence
//! linkage changed.
//!
//! Verification is observational. It performs no mutation and cannot repair a
//! failed invariant automatically.

use super::claim_evidence::{ClaimId, EpistemicLedger, EvidenceId, EvidenceKind, EvidencePolarity};
use super::evidence_mutation_firewall::{
    EvidenceDraftIdentity, EvidenceIngestionOutcome, EvidenceIngestionReceipt,
};
use super::receipt_admission::AdmissibleEvidenceDraft;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceMutationSnapshot {
    claim_id: ClaimId,
    claim_count: usize,
    evidence_count: usize,
    provenance_count: usize,
    target_evidence_ids: Vec<EvidenceId>,
    interventional_support_count: usize,
}

impl EvidenceMutationSnapshot {
    pub fn capture(
        ledger: &EpistemicLedger,
        draft: &AdmissibleEvidenceDraft,
    ) -> Result<Self, EvidenceMutationVerificationError> {
        let claim = ledger
            .claim(draft.claim_id())
            .ok_or(EvidenceMutationVerificationError::UnknownClaim(
                draft.claim_id(),
            ))?;
        if ledger.provenance(draft.provenance_id()).is_none() {
            return Err(EvidenceMutationVerificationError::UnknownProvenance(
                draft.provenance_id().0,
            ));
        }

        let mut target_evidence_ids = claim.evidence_ids.clone();
        target_evidence_ids.sort_unstable();

        Ok(Self {
            claim_id: draft.claim_id(),
            claim_count: ledger.claim_count(),
            evidence_count: ledger.evidence_count(),
            provenance_count: ledger.provenance_count(),
            target_evidence_ids,
            interventional_support_count: ledger.interventional_support_count(draft.claim_id()),
        })
    }

    pub fn claim_id(&self) -> ClaimId {
        self.claim_id
    }

    pub fn claim_count(&self) -> usize {
        self.claim_count
    }

    pub fn evidence_count(&self) -> usize {
        self.evidence_count
    }

    pub fn provenance_count(&self) -> usize {
        self.provenance_count
    }

    pub fn target_evidence_ids(&self) -> &[EvidenceId] {
        &self.target_evidence_ids
    }

    pub fn interventional_support_count(&self) -> usize {
        self.interventional_support_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceMutationVerificationError {
    UnknownClaim(ClaimId),
    /// Kept as a raw numeric ID to avoid widening this verifier's public type
    /// surface solely for snapshot-construction errors.
    UnknownProvenance(u64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceMutationInvariantFailure {
    ReceiptClaimMismatch {
        snapshot_claim: ClaimId,
        receipt_claim: ClaimId,
    },
    ClaimCountChanged {
        before: usize,
        after: usize,
    },
    ProvenanceCountChanged {
        before: usize,
        after: usize,
    },
    EvidenceCountMismatch {
        expected: usize,
        actual: usize,
    },
    TargetEvidenceIdsMismatch {
        expected: Vec<EvidenceId>,
        actual: Vec<EvidenceId>,
    },
    DuplicateTargetEvidenceId(EvidenceId),
    AlreadySatisfiedEvidenceWasNotPresentBefore(EvidenceId),
    InsertedEvidenceAlreadyPresentBefore(EvidenceId),
    MissingReceiptEvidence(EvidenceId),
    ReceiptEvidenceMismatch(EvidenceId),
    InterventionalSupportMismatch {
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceMutationVerificationReport {
    failures: Vec<EvidenceMutationInvariantFailure>,
}

impl EvidenceMutationVerificationReport {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    pub fn failures(&self) -> &[EvidenceMutationInvariantFailure] {
        &self.failures
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct EvidenceMutationVerifier;

impl EvidenceMutationVerifier {
    pub fn verify(
        before: &EvidenceMutationSnapshot,
        ledger: &EpistemicLedger,
        outcome: &EvidenceIngestionOutcome,
    ) -> EvidenceMutationVerificationReport {
        let receipt = outcome.receipt();
        let identity = receipt.draft_identity();
        let mut failures = Vec::new();

        if identity.claim_id != before.claim_id {
            failures.push(EvidenceMutationInvariantFailure::ReceiptClaimMismatch {
                snapshot_claim: before.claim_id,
                receipt_claim: identity.claim_id,
            });
        }

        if ledger.claim_count() != before.claim_count {
            failures.push(EvidenceMutationInvariantFailure::ClaimCountChanged {
                before: before.claim_count,
                after: ledger.claim_count(),
            });
        }
        if ledger.provenance_count() != before.provenance_count {
            failures.push(EvidenceMutationInvariantFailure::ProvenanceCountChanged {
                before: before.provenance_count,
                after: ledger.provenance_count(),
            });
        }

        let inserted = outcome.inserted_new_record();
        let expected_evidence_count = before.evidence_count + usize::from(inserted);
        if ledger.evidence_count() != expected_evidence_count {
            failures.push(EvidenceMutationInvariantFailure::EvidenceCountMismatch {
                expected: expected_evidence_count,
                actual: ledger.evidence_count(),
            });
        }

        let receipt_id = receipt.evidence_id();
        let was_present_before = before.target_evidence_ids.contains(&receipt_id);
        if inserted && was_present_before {
            failures.push(EvidenceMutationInvariantFailure::InsertedEvidenceAlreadyPresentBefore(
                receipt_id,
            ));
        }
        if !inserted && !was_present_before {
            failures.push(
                EvidenceMutationInvariantFailure::AlreadySatisfiedEvidenceWasNotPresentBefore(
                    receipt_id,
                ),
            );
        }

        let mut expected_target_ids = before.target_evidence_ids.clone();
        if inserted {
            expected_target_ids.push(receipt_id);
        }
        expected_target_ids.sort_unstable();

        let mut actual_target_ids = ledger
            .claim(before.claim_id)
            .map(|claim| claim.evidence_ids.clone())
            .unwrap_or_default();
        actual_target_ids.sort_unstable();

        if expected_target_ids != actual_target_ids {
            failures.push(EvidenceMutationInvariantFailure::TargetEvidenceIdsMismatch {
                expected: expected_target_ids,
                actual: actual_target_ids.clone(),
            });
        }

        let mut seen = HashSet::new();
        for evidence_id in &actual_target_ids {
            if !seen.insert(*evidence_id) {
                failures.push(EvidenceMutationInvariantFailure::DuplicateTargetEvidenceId(
                    *evidence_id,
                ));
            }
        }

        match ledger.evidence(receipt_id) {
            None => failures.push(EvidenceMutationInvariantFailure::MissingReceiptEvidence(
                receipt_id,
            )),
            Some(record) if !record_matches_receipt(record, receipt) => failures.push(
                EvidenceMutationInvariantFailure::ReceiptEvidenceMismatch(receipt_id),
            ),
            Some(_) => {}
        }

        let expected_interventional_support = before.interventional_support_count
            + usize::from(
                inserted
                    && identity.polarity == EvidencePolarity::Supports
                    && matches!(identity.kind, EvidenceKind::Intervention | EvidenceKind::Replication),
            );
        let actual_interventional_support = ledger.interventional_support_count(before.claim_id);
        if actual_interventional_support != expected_interventional_support {
            failures.push(EvidenceMutationInvariantFailure::InterventionalSupportMismatch {
                expected: expected_interventional_support,
                actual: actual_interventional_support,
            });
        }

        EvidenceMutationVerificationReport { failures }
    }
}

fn record_matches_receipt(
    record: &super::claim_evidence::EvidenceRecord,
    receipt: &EvidenceIngestionReceipt,
) -> bool {
    let identity: &EvidenceDraftIdentity = receipt.draft_identity();
    let expected_context = Some(format!("inquiry-result: {}", identity.result_summary));
    let expected_method = Some(format!(
        "preregistered-decision[{}]: {}",
        identity.decision_rule_label, identity.decision_criterion
    ));

    record.claim_id == identity.claim_id
        && record.kind == identity.kind
        && record.polarity == identity.polarity
        && record.provenance_id == identity.provenance_id
        && record.observed_at_cycle == identity.observed_at_cycle
        && record.context == expected_context
        && record.method == expected_method
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        ClaimKind, DecisionInterpretation, EvidenceMutationAuthorization,
        IgnoranceFrontier, InquiryContractBuilder, InquiryPreregistration, InquiryRequest,
        InquiryResultReceipt, JournaledEvidenceMutationFirewall, MutationAuthorizationDecision,
        PreregisteredDecisionRule, ReceiptAdmissionGate, ReceiptAdmissionPolicy,
    };

    fn rules() -> Vec<PreregisteredDecisionRule> {
        vec![
            PreregisteredDecisionRule::new(
                "supports",
                "measure > upper",
                DecisionInterpretation::SupportsClaim,
            ),
            PreregisteredDecisionRule::new(
                "contradicts",
                "measure < lower",
                DecisionInterpretation::ContradictsClaim,
            ),
            PreregisteredDecisionRule::new(
                "inconclusive",
                "otherwise",
                DecisionInterpretation::Inconclusive,
            ),
        ]
    }

    fn fixture(
        kind: EvidenceKind,
    ) -> (
        EpistemicLedger,
        AdmissibleEvidenceDraft,
        EvidenceMutationAuthorization,
    ) {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("experiment", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let frontier = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        let plan = InquiryContractBuilder::from_frontier(&ledger, &frontier).unwrap();
        let contract = plan
            .contracts
            .into_iter()
            .find(|contract| contract.request == InquiryRequest::SeekDiscriminatingEvidence)
            .unwrap();
        let preregistration = InquiryPreregistration::new(
            &contract,
            "bounded comparison",
            "prediction error delta",
            rules(),
            "fixed sample budget",
            vec![],
            vec![],
            5,
        )
        .unwrap();
        let receipt = InquiryResultReceipt::new(
            &ledger,
            &contract,
            &preregistration,
            kind,
            provenance,
            6,
            "observed result",
            "supports",
        )
        .unwrap();
        let policy = ReceiptAdmissionPolicy::new(vec![kind], true, true, false);
        let admission = ReceiptAdmissionGate::evaluate(&ledger, &receipt, &policy);
        let draft = admission.draft().unwrap().clone();
        let authorization = EvidenceMutationAuthorization::new(
            "auth-1",
            "test-authority",
            MutationAuthorizationDecision::Approved,
            7,
            &draft,
        )
        .unwrap();
        (ledger, draft, authorization)
    }

    #[test]
    fn clean_insertion_verifies_only_expected_mutation() {
        let (mut ledger, draft, authorization) = fixture(EvidenceKind::Measurement);
        let before = EvidenceMutationSnapshot::capture(&ledger, &draft).unwrap();
        let mut firewall = JournaledEvidenceMutationFirewall::new();
        let outcome = firewall
            .ingest(&mut ledger, &draft, &authorization, 8)
            .unwrap();

        let report = EvidenceMutationVerifier::verify(&before, &ledger, &outcome);
        assert!(report.passed(), "{:?}", report.failures());
    }

    #[test]
    fn idempotent_replay_verifies_no_second_mutation() {
        let (mut ledger, draft, authorization) = fixture(EvidenceKind::Measurement);
        let mut firewall = JournaledEvidenceMutationFirewall::new();
        firewall
            .ingest(&mut ledger, &draft, &authorization, 8)
            .unwrap();
        let before_replay = EvidenceMutationSnapshot::capture(&ledger, &draft).unwrap();
        let replay = firewall
            .ingest(&mut ledger, &draft, &authorization, 9)
            .unwrap();

        let report = EvidenceMutationVerifier::verify(&before_replay, &ledger, &replay);
        assert!(report.passed(), "{:?}", report.failures());
        assert!(!replay.inserted_new_record());
    }

    #[test]
    fn report_ingestion_cannot_increase_interventional_support() {
        let (mut ledger, draft, authorization) = fixture(EvidenceKind::Report);
        let before = EvidenceMutationSnapshot::capture(&ledger, &draft).unwrap();
        let mut firewall = JournaledEvidenceMutationFirewall::new();
        let outcome = firewall
            .ingest(&mut ledger, &draft, &authorization, 8)
            .unwrap();

        let report = EvidenceMutationVerifier::verify(&before, &ledger, &outcome);
        assert!(report.passed(), "{:?}", report.failures());
        assert_eq!(ledger.interventional_support_count(draft.claim_id()), 0);
    }

    #[test]
    fn unrelated_extra_mutation_is_detected() {
        let (mut ledger, draft, authorization) = fixture(EvidenceKind::Measurement);
        let before = EvidenceMutationSnapshot::capture(&ledger, &draft).unwrap();
        let mut firewall = JournaledEvidenceMutationFirewall::new();
        let outcome = firewall
            .ingest(&mut ledger, &draft, &authorization, 8)
            .unwrap();

        let other_source = ledger
            .add_provenance("unrelated", None, None, 9, vec![])
            .unwrap();
        let other_claim = ledger.add_claim("unrelated claim", ClaimKind::Descriptive, None, None, 9);
        ledger
            .add_evidence(
                other_claim,
                EvidenceKind::Report,
                EvidencePolarity::Supports,
                other_source,
                9,
                None,
                None,
            )
            .unwrap();

        let report = EvidenceMutationVerifier::verify(&before, &ledger, &outcome);
        assert!(!report.passed());
        assert!(report.failures().iter().any(|failure| matches!(
            failure,
            EvidenceMutationInvariantFailure::ClaimCountChanged { .. }
                | EvidenceMutationInvariantFailure::ProvenanceCountChanged { .. }
                | EvidenceMutationInvariantFailure::EvidenceCountMismatch { .. }
        )));
    }
}
