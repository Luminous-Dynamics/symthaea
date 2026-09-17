// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Read-only conformance observer for the canonical EKM evidence-record binding.
//!
//! EKM-020 freezes `EvidenceRecordBinding::v1()` as the compatibility target for
//! evidence insertion, restart replay, and post-mutation verification. This module
//! checks the records those existing paths produce without changing their runtime
//! behavior. It is therefore safe to use before the writer/journal/verifier are
//! mechanically migrated onto the canonical helper.
//!
//! A passing report means only that the inspected records conform to the selected
//! ledger binding version. It does not qualify the scientific evidence, authorize
//! a mutation, update confidence, or promote a causal relation.

use super::claim_evidence::{EpistemicLedger, EvidenceId};
use super::evidence_mutation_firewall::EvidenceIngestionOutcome;
use super::evidence_mutation_journal::EvidenceMutationJournal;
use super::evidence_record_binding::{EvidenceRecordBinding, EvidenceRecordBindingVersion};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceBindingConformanceFailure {
    MissingReceiptEvidence(EvidenceId),
    ReceiptRecordMismatch(EvidenceId),
    JournalEvidenceMissing {
        authorization_id: String,
        evidence_id: EvidenceId,
    },
    JournalRecordMismatch {
        authorization_id: String,
        evidence_id: EvidenceId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceBindingConformanceReport {
    version: EvidenceRecordBindingVersion,
    checked_receipt_records: usize,
    checked_journal_records: usize,
    failures: Vec<EvidenceBindingConformanceFailure>,
}

impl EvidenceBindingConformanceReport {
    pub fn version(&self) -> EvidenceRecordBindingVersion {
        self.version
    }

    pub fn checked_receipt_records(&self) -> usize {
        self.checked_receipt_records
    }

    pub fn checked_journal_records(&self) -> usize {
        self.checked_journal_records
    }

    pub fn failures(&self) -> &[EvidenceBindingConformanceFailure] {
        &self.failures
    }

    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Pure observer for exact ledger-binding compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceBindingConformanceObserver {
    binding: EvidenceRecordBinding,
}

impl Default for EvidenceBindingConformanceObserver {
    fn default() -> Self {
        Self::v1()
    }
}

impl EvidenceBindingConformanceObserver {
    pub const fn v1() -> Self {
        Self {
            binding: EvidenceRecordBinding::v1(),
        }
    }

    pub const fn version(self) -> EvidenceRecordBindingVersion {
        self.binding.version()
    }

    /// Check the ledger record named by one mutation outcome against the exact
    /// draft identity carried by its receipt.
    pub fn inspect_outcome(
        self,
        ledger: &EpistemicLedger,
        outcome: &EvidenceIngestionOutcome,
    ) -> EvidenceBindingConformanceReport {
        let receipt = outcome.receipt();
        let evidence_id = receipt.evidence_id();
        let mut failures = Vec::new();

        match ledger.evidence(evidence_id) {
            None => failures.push(EvidenceBindingConformanceFailure::MissingReceiptEvidence(
                evidence_id,
            )),
            Some(record) if !self.binding.matches(record, receipt.draft_identity()) => {
                failures.push(EvidenceBindingConformanceFailure::ReceiptRecordMismatch(
                    evidence_id,
                ));
            }
            Some(_) => {}
        }

        EvidenceBindingConformanceReport {
            version: self.binding.version(),
            checked_receipt_records: 1,
            checked_journal_records: 0,
            failures,
        }
    }

    /// Check every exported journal entry against the ledger record it claims to
    /// bind. The journal itself is not modified.
    pub fn inspect_journal(
        self,
        ledger: &EpistemicLedger,
        journal: &EvidenceMutationJournal,
    ) -> EvidenceBindingConformanceReport {
        let snapshot = journal.snapshot();
        let mut failures = Vec::new();

        for entry in snapshot.entries() {
            let evidence_id = entry.evidence_id();
            match ledger.evidence(evidence_id) {
                None => failures.push(
                    EvidenceBindingConformanceFailure::JournalEvidenceMissing {
                        authorization_id: entry.authorization_id().to_string(),
                        evidence_id,
                    },
                ),
                Some(record)
                    if !self.binding.matches(record, entry.draft_identity()) =>
                {
                    failures.push(
                        EvidenceBindingConformanceFailure::JournalRecordMismatch {
                            authorization_id: entry.authorization_id().to_string(),
                            evidence_id,
                        },
                    );
                }
                Some(_) => {}
            }
        }

        EvidenceBindingConformanceReport {
            version: self.binding.version(),
            checked_receipt_records: 0,
            checked_journal_records: snapshot.entries().len(),
            failures,
        }
    }

    /// Inspect both the immediate mutation outcome and the current journal in one
    /// report. This is observational and performs no repair when a mismatch exists.
    pub fn inspect_pipeline(
        self,
        ledger: &EpistemicLedger,
        outcome: &EvidenceIngestionOutcome,
        journal: &EvidenceMutationJournal,
    ) -> EvidenceBindingConformanceReport {
        let outcome_report = self.inspect_outcome(ledger, outcome);
        let journal_report = self.inspect_journal(ledger, journal);
        let mut failures = outcome_report.failures;
        failures.extend(journal_report.failures);

        EvidenceBindingConformanceReport {
            version: self.binding.version(),
            checked_receipt_records: outcome_report.checked_receipt_records,
            checked_journal_records: journal_report.checked_journal_records,
            failures,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        ClaimKind, DecisionInterpretation, EvidenceKind, EvidenceMutationAuthorization,
        EvidenceMutationSnapshot, EvidenceMutationVerifier, IgnoranceFrontier,
        InquiryContractBuilder, InquiryPreregistration, InquiryRequest, InquiryResultReceipt,
        JournaledEvidenceMutationFirewall, MutationAuthorizationDecision,
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

    fn fixture() -> (
        EpistemicLedger,
        crate::knowledge::AdmissibleEvidenceDraft,
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
        let result = InquiryResultReceipt::new(
            &ledger,
            &contract,
            &preregistration,
            EvidenceKind::Measurement,
            provenance,
            6,
            "observed result",
            "supports",
        )
        .unwrap();
        let policy = ReceiptAdmissionPolicy::new(vec![EvidenceKind::Measurement], true, true, false);
        let admission = ReceiptAdmissionGate::evaluate(&ledger, &result, &policy);
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
    fn current_mutation_writer_conforms_to_v1_without_runtime_migration() {
        let (mut ledger, draft, authorization) = fixture();
        let before = EvidenceMutationSnapshot::capture(&ledger, &draft).unwrap();
        let mut firewall = JournaledEvidenceMutationFirewall::new();
        let outcome = firewall
            .ingest(&mut ledger, &draft, &authorization, 8)
            .unwrap();

        let report = EvidenceBindingConformanceObserver::v1().inspect_pipeline(
            &ledger,
            &outcome,
            firewall.journal(),
        );
        assert!(report.passed(), "{:?}", report.failures());
        assert_eq!(report.version(), EvidenceRecordBindingVersion::V1);
        assert_eq!(report.checked_receipt_records(), 1);
        assert_eq!(report.checked_journal_records(), 1);

        // The pre-existing independent mutation verifier must agree as well.
        let mutation_report = EvidenceMutationVerifier::verify(&before, &ledger, &outcome);
        assert!(mutation_report.passed(), "{:?}", mutation_report.failures());
    }

    #[test]
    fn idempotent_replay_remains_v1_conformant() {
        let (mut ledger, draft, authorization) = fixture();
        let mut firewall = JournaledEvidenceMutationFirewall::new();
        firewall
            .ingest(&mut ledger, &draft, &authorization, 8)
            .unwrap();
        let replay = firewall
            .ingest(&mut ledger, &draft, &authorization, 9)
            .unwrap();
        assert!(!replay.inserted_new_record());

        let report = EvidenceBindingConformanceObserver::v1().inspect_pipeline(
            &ledger,
            &replay,
            firewall.journal(),
        );
        assert!(report.passed(), "{:?}", report.failures());
    }

    #[test]
    fn tampered_record_is_reported_not_repaired() {
        let (mut ledger, draft, authorization) = fixture();
        let mut firewall = JournaledEvidenceMutationFirewall::new();
        let outcome = firewall
            .ingest(&mut ledger, &draft, &authorization, 8)
            .unwrap();
        let receipt = outcome.receipt();

        // Build a separate ledger with the same IDs but a non-canonical record.
        let mut divergent = EpistemicLedger::new();
        let provenance = divergent
            .add_provenance("experiment", None, None, 1, vec![])
            .unwrap();
        let claim = divergent.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        assert_eq!(claim, draft.claim_id());
        assert_eq!(provenance, draft.provenance_id());
        let evidence_id = divergent
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                crate::knowledge::EvidencePolarity::Supports,
                provenance,
                6,
                Some("non-canonical-context".into()),
                Some("non-canonical-method".into()),
            )
            .unwrap();
        assert_eq!(evidence_id, receipt.evidence_id());

        let report = EvidenceBindingConformanceObserver::v1().inspect_outcome(
            &divergent,
            &outcome,
        );
        assert!(!report.passed());
        assert_eq!(
            report.failures(),
            &[EvidenceBindingConformanceFailure::ReceiptRecordMismatch(
                receipt.evidence_id()
            )]
        );
        assert_eq!(divergent.evidence_count(), 1);
    }
}
