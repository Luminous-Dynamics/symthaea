// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Non-mutating admission gate from inquiry result receipts to evidence drafts.
//!
//! A receipt is not automatically knowledge. This module evaluates a receipt
//! against an explicit caller-supplied policy and, when eligible, produces an
//! immutable evidence draft. It never calls `EpistemicLedger::add_evidence`.

use super::claim_evidence::{
    ClaimId, EpistemicLedger, EvidenceKind, EvidencePolarity, ProvenanceId,
};
use super::inquiry_preregistration::DecisionInterpretation;
use super::inquiry_result::InquiryResultReceipt;

/// Explicit policy for turning a validated result receipt into an evidence draft.
///
/// There is intentionally no `Default`: callers must choose what evidence kinds
/// and interpretations are admissible in their context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptAdmissionPolicy {
    accepted_kinds: Vec<EvidenceKind>,
    admit_supporting_results: bool,
    admit_contradicting_results: bool,
    admit_inconclusive_as_contextual: bool,
}

impl ReceiptAdmissionPolicy {
    pub fn new(
        accepted_kinds: Vec<EvidenceKind>,
        admit_supporting_results: bool,
        admit_contradicting_results: bool,
        admit_inconclusive_as_contextual: bool,
    ) -> Self {
        let mut deduped = Vec::new();
        for kind in accepted_kinds {
            if !deduped.contains(&kind) {
                deduped.push(kind);
            }
        }
        Self {
            accepted_kinds: deduped,
            admit_supporting_results,
            admit_contradicting_results,
            admit_inconclusive_as_contextual,
        }
    }

    pub fn accepted_kinds(&self) -> &[EvidenceKind] {
        &self.accepted_kinds
    }

    pub fn admit_supporting_results(&self) -> bool {
        self.admit_supporting_results
    }

    pub fn admit_contradicting_results(&self) -> bool {
        self.admit_contradicting_results
    }

    pub fn admit_inconclusive_as_contextual(&self) -> bool {
        self.admit_inconclusive_as_contextual
    }
}

/// Immutable candidate evidence record. It has not been inserted into the ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissibleEvidenceDraft {
    claim_id: ClaimId,
    kind: EvidenceKind,
    polarity: EvidencePolarity,
    provenance_id: ProvenanceId,
    observed_at_cycle: u64,
    result_summary: String,
    decision_rule_label: String,
    decision_criterion: String,
}

impl AdmissibleEvidenceDraft {
    pub fn claim_id(&self) -> ClaimId {
        self.claim_id
    }

    pub fn kind(&self) -> EvidenceKind {
        self.kind
    }

    pub fn polarity(&self) -> EvidencePolarity {
        self.polarity
    }

    pub fn provenance_id(&self) -> ProvenanceId {
        self.provenance_id
    }

    pub fn observed_at_cycle(&self) -> u64 {
        self.observed_at_cycle
    }

    pub fn result_summary(&self) -> &str {
        &self.result_summary
    }

    pub fn decision_rule_label(&self) -> &str {
        &self.decision_rule_label
    }

    pub fn decision_criterion(&self) -> &str {
        &self.decision_criterion
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiptAdmissionFailure {
    UnknownClaim(ClaimId),
    UnknownProvenance(ProvenanceId),
    EvidenceKindNotAllowed(EvidenceKind),
    SupportingResultsDisabled,
    ContradictingResultsDisabled,
    InconclusiveContextualizationDisabled,
}

/// Auditable, non-mutating result of evaluating one receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptAdmissionDecision {
    eligible: bool,
    failures: Vec<ReceiptAdmissionFailure>,
    draft: Option<AdmissibleEvidenceDraft>,
}

impl ReceiptAdmissionDecision {
    pub fn eligible(&self) -> bool {
        self.eligible
    }

    pub fn failures(&self) -> &[ReceiptAdmissionFailure] {
        &self.failures
    }

    pub fn draft(&self) -> Option<&AdmissibleEvidenceDraft> {
        self.draft.as_ref()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ReceiptAdmissionGate;

impl ReceiptAdmissionGate {
    pub fn evaluate(
        ledger: &EpistemicLedger,
        receipt: &InquiryResultReceipt,
        policy: &ReceiptAdmissionPolicy,
    ) -> ReceiptAdmissionDecision {
        let mut failures = Vec::new();

        if ledger.claim(receipt.claim_id()).is_none() {
            failures.push(ReceiptAdmissionFailure::UnknownClaim(receipt.claim_id()));
        }
        if ledger.provenance(receipt.provenance_id()).is_none() {
            failures.push(ReceiptAdmissionFailure::UnknownProvenance(
                receipt.provenance_id(),
            ));
        }
        if !policy.accepted_kinds.contains(&receipt.evidence_kind()) {
            failures.push(ReceiptAdmissionFailure::EvidenceKindNotAllowed(
                receipt.evidence_kind(),
            ));
        }

        let polarity = match receipt.interpretation() {
            DecisionInterpretation::SupportsClaim => {
                if !policy.admit_supporting_results {
                    failures.push(ReceiptAdmissionFailure::SupportingResultsDisabled);
                }
                EvidencePolarity::Supports
            }
            DecisionInterpretation::ContradictsClaim => {
                if !policy.admit_contradicting_results {
                    failures.push(ReceiptAdmissionFailure::ContradictingResultsDisabled);
                }
                EvidencePolarity::Contradicts
            }
            DecisionInterpretation::Inconclusive => {
                if !policy.admit_inconclusive_as_contextual {
                    failures.push(
                        ReceiptAdmissionFailure::InconclusiveContextualizationDisabled,
                    );
                }
                EvidencePolarity::Contextualizes
            }
        };

        let eligible = failures.is_empty();
        let draft = eligible.then(|| AdmissibleEvidenceDraft {
            claim_id: receipt.claim_id(),
            kind: receipt.evidence_kind(),
            polarity,
            provenance_id: receipt.provenance_id(),
            observed_at_cycle: receipt.observed_at_cycle(),
            result_summary: receipt.result_summary().to_string(),
            decision_rule_label: receipt.decision_rule_label().to_string(),
            decision_criterion: receipt.decision_criterion().to_string(),
        });

        ReceiptAdmissionDecision {
            eligible,
            failures,
            draft,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        ClaimKind, DecisionInterpretation, IgnoranceFrontier, InquiryContractBuilder,
        InquiryPreregistration, InquiryRequest, InquiryResultReceipt, PreregisteredDecisionRule,
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

    fn receipt_fixture(
        rule_label: &str,
        evidence_kind: EvidenceKind,
    ) -> (EpistemicLedger, InquiryResultReceipt) {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("source", None, None, 1, vec![])
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
            evidence_kind,
            provenance,
            6,
            "observed result",
            rule_label,
        )
        .unwrap();
        (ledger, receipt)
    }

    #[test]
    fn eligible_receipt_produces_draft_without_mutating_ledger() {
        let (ledger, receipt) = receipt_fixture("supports", EvidenceKind::Measurement);
        let evidence_before = ledger.evidence_count();
        let policy = ReceiptAdmissionPolicy::new(
            vec![EvidenceKind::Measurement],
            true,
            true,
            false,
        );

        let decision = ReceiptAdmissionGate::evaluate(&ledger, &receipt, &policy);
        assert!(decision.eligible());
        let draft = decision.draft().unwrap();
        assert_eq!(draft.kind(), EvidenceKind::Measurement);
        assert_eq!(draft.polarity(), EvidencePolarity::Supports);
        assert_eq!(ledger.evidence_count(), evidence_before);
    }

    #[test]
    fn contradicting_receipt_stays_contradicting() {
        let (ledger, receipt) = receipt_fixture("contradicts", EvidenceKind::Measurement);
        let policy = ReceiptAdmissionPolicy::new(
            vec![EvidenceKind::Measurement],
            true,
            true,
            false,
        );
        let decision = ReceiptAdmissionGate::evaluate(&ledger, &receipt, &policy);
        assert_eq!(
            decision.draft().unwrap().polarity(),
            EvidencePolarity::Contradicts
        );
    }

    #[test]
    fn inconclusive_result_requires_explicit_contextualization_opt_in() {
        let (ledger, receipt) = receipt_fixture("inconclusive", EvidenceKind::Observation);
        let deny = ReceiptAdmissionPolicy::new(
            vec![EvidenceKind::Observation],
            true,
            true,
            false,
        );
        let denied = ReceiptAdmissionGate::evaluate(&ledger, &receipt, &deny);
        assert!(!denied.eligible());
        assert!(denied.failures().contains(
            &ReceiptAdmissionFailure::InconclusiveContextualizationDisabled
        ));
        assert!(denied.draft().is_none());

        let allow = ReceiptAdmissionPolicy::new(
            vec![EvidenceKind::Observation],
            true,
            true,
            true,
        );
        let admitted = ReceiptAdmissionGate::evaluate(&ledger, &receipt, &allow);
        assert_eq!(
            admitted.draft().unwrap().polarity(),
            EvidencePolarity::Contextualizes
        );
    }

    #[test]
    fn evidence_kind_policy_fails_closed() {
        let (ledger, receipt) = receipt_fixture("supports", EvidenceKind::Report);
        let policy = ReceiptAdmissionPolicy::new(
            vec![EvidenceKind::Measurement],
            true,
            true,
            false,
        );
        let decision = ReceiptAdmissionGate::evaluate(&ledger, &receipt, &policy);
        assert!(!decision.eligible());
        assert!(decision.failures().contains(
            &ReceiptAdmissionFailure::EvidenceKindNotAllowed(EvidenceKind::Report)
        ));
    }

    #[test]
    fn report_receipt_never_changes_evidence_kind_into_intervention() {
        let (ledger, receipt) = receipt_fixture("supports", EvidenceKind::Report);
        let policy = ReceiptAdmissionPolicy::new(vec![EvidenceKind::Report], true, true, false);
        let decision = ReceiptAdmissionGate::evaluate(&ledger, &receipt, &policy);
        let draft = decision.draft().unwrap();
        assert_eq!(draft.kind(), EvidenceKind::Report);
        assert_ne!(draft.kind(), EvidenceKind::Intervention);
    }

    #[test]
    fn policy_can_refuse_supporting_results_without_suppressing_audit_failure() {
        let (ledger, receipt) = receipt_fixture("supports", EvidenceKind::Measurement);
        let policy = ReceiptAdmissionPolicy::new(
            vec![EvidenceKind::Measurement],
            false,
            true,
            false,
        );
        let decision = ReceiptAdmissionGate::evaluate(&ledger, &receipt, &policy);
        assert!(!decision.eligible());
        assert!(decision
            .failures()
            .contains(&ReceiptAdmissionFailure::SupportingResultsDisabled));
        assert!(decision.draft().is_none());
    }
}
