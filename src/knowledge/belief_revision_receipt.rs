// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Immutable audit receipts for shadow belief-revision decisions.
//!
//! A receipt records the exact proposal, policy, calibration snapshot, uncertainty
//! assessment, evidence records, and gate decision used at one evaluation point.
//! It records eligible and rejected decisions alike. It does not apply a weight
//! update and carries no mutation authority.

use super::belief_revision_gate::{
    BeliefRevisionDecision, BeliefRevisionGate, BeliefRevisionPolicy, CalibrationSnapshot,
    EpistemicRevisionProposal,
};
use super::claim_evidence::{
    ClaimId, EpistemicLedger, EvidenceId, EvidenceKind, EvidencePolarity, ProvenanceId,
};
use super::epistemic_vector::ClaimUncertaintyAssessment;
use std::collections::HashSet;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BeliefRevisionReceiptId(pub u64);

/// Immutable copy of the ledger fields that existed when a basis record was
/// evaluated. The live evidence record is not referenced by pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevisionEvidenceSnapshot {
    pub evidence_id: EvidenceId,
    pub claim_id: ClaimId,
    pub kind: EvidenceKind,
    pub polarity: EvidencePolarity,
    pub provenance_id: ProvenanceId,
    pub observed_at_cycle: u64,
    pub context: Option<String>,
    pub method: Option<String>,
}

/// One unique evidence ID requested by the proposal, together with the record
/// that resolved at evaluation time. `None` preserves unknown IDs in rejected
/// proposals rather than making them disappear from audit history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevisionEvidenceReference {
    pub requested_id: EvidenceId,
    pub snapshot: Option<RevisionEvidenceSnapshot>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefRevisionReceipt {
    id: BeliefRevisionReceiptId,
    claim_id: ClaimId,
    proposed_delta: f32,
    rationale: String,
    /// Unique basis IDs in first-seen order. Duplicate attempts are reported
    /// separately and never duplicated in the snapshotted basis.
    basis: Vec<RevisionEvidenceReference>,
    duplicate_basis_evidence_ids: Vec<EvidenceId>,
    policy: BeliefRevisionPolicy,
    calibration: Option<CalibrationSnapshot>,
    uncertainty: Option<ClaimUncertaintyAssessment>,
    decision: BeliefRevisionDecision,
    evaluated_at_cycle: u64,
}

impl BeliefRevisionReceipt {
    pub fn id(&self) -> BeliefRevisionReceiptId {
        self.id
    }

    pub fn claim_id(&self) -> ClaimId {
        self.claim_id
    }

    pub fn proposed_delta(&self) -> f32 {
        self.proposed_delta
    }

    pub fn rationale(&self) -> &str {
        &self.rationale
    }

    pub fn basis(&self) -> &[RevisionEvidenceReference] {
        &self.basis
    }

    pub fn duplicate_basis_evidence_ids(&self) -> &[EvidenceId] {
        &self.duplicate_basis_evidence_ids
    }

    pub fn policy(&self) -> &BeliefRevisionPolicy {
        &self.policy
    }

    pub fn calibration(&self) -> Option<CalibrationSnapshot> {
        self.calibration
    }

    pub fn uncertainty(&self) -> Option<&ClaimUncertaintyAssessment> {
        self.uncertainty.as_ref()
    }

    pub fn decision(&self) -> &BeliefRevisionDecision {
        &self.decision
    }

    pub fn evaluated_at_cycle(&self) -> u64 {
        self.evaluated_at_cycle
    }

    pub fn eligible(&self) -> bool {
        self.decision.eligible()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeliefRevisionReceiptError {
    EvaluationPredatesClaim {
        evaluated_at_cycle: u64,
        claim_created_at_cycle: u64,
    },
    EvaluationPredatesBasisEvidence {
        evidence_id: EvidenceId,
        evaluated_at_cycle: u64,
        evidence_observed_at_cycle: u64,
    },
    EvaluationPredatesUncertaintyAssessment {
        evaluated_at_cycle: u64,
        assessment_cycle: u64,
    },
}

impl fmt::Display for BeliefRevisionReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EvaluationPredatesClaim {
                evaluated_at_cycle,
                claim_created_at_cycle,
            } => write!(
                f,
                "belief revision evaluation cycle {evaluated_at_cycle} predates claim creation cycle {claim_created_at_cycle}"
            ),
            Self::EvaluationPredatesBasisEvidence {
                evidence_id,
                evaluated_at_cycle,
                evidence_observed_at_cycle,
            } => write!(
                f,
                "belief revision evaluation cycle {evaluated_at_cycle} predates basis evidence {} observed at cycle {evidence_observed_at_cycle}",
                evidence_id.0
            ),
            Self::EvaluationPredatesUncertaintyAssessment {
                evaluated_at_cycle,
                assessment_cycle,
            } => write!(
                f,
                "belief revision evaluation cycle {evaluated_at_cycle} predates uncertainty assessment cycle {assessment_cycle}"
            ),
        }
    }
}

impl Error for BeliefRevisionReceiptError {}

/// Append-only in-memory audit history for shadow revision decisions.
///
/// Persistence is deliberately not implemented here. A later persistence layer
/// can serialize these immutable receipts once the representation is qualified.
#[derive(Debug, Clone)]
pub struct BeliefRevisionHistory {
    receipts: Vec<BeliefRevisionReceipt>,
    next_id: u64,
}

impl Default for BeliefRevisionHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl BeliefRevisionHistory {
    pub fn new() -> Self {
        Self {
            receipts: Vec::new(),
            next_id: 1,
        }
    }

    pub fn evaluate_and_record(
        &mut self,
        ledger: &EpistemicLedger,
        proposal: &EpistemicRevisionProposal,
        policy: &BeliefRevisionPolicy,
        calibration: Option<CalibrationSnapshot>,
        uncertainty: Option<&ClaimUncertaintyAssessment>,
        evaluated_at_cycle: u64,
    ) -> Result<BeliefRevisionReceiptId, BeliefRevisionReceiptError> {
        if let Some(claim) = ledger.claim(proposal.claim_id) {
            if evaluated_at_cycle < claim.created_at_cycle {
                return Err(BeliefRevisionReceiptError::EvaluationPredatesClaim {
                    evaluated_at_cycle,
                    claim_created_at_cycle: claim.created_at_cycle,
                });
            }
        }

        if let Some(assessment) = uncertainty {
            if evaluated_at_cycle < assessment.assessed_at_cycle {
                return Err(
                    BeliefRevisionReceiptError::EvaluationPredatesUncertaintyAssessment {
                        evaluated_at_cycle,
                        assessment_cycle: assessment.assessed_at_cycle,
                    },
                );
            }
        }

        let mut seen = HashSet::new();
        let mut duplicate_basis_evidence_ids = Vec::new();
        let mut basis = Vec::new();
        for evidence_id in &proposal.update.basis_evidence_ids {
            if !seen.insert(*evidence_id) {
                if !duplicate_basis_evidence_ids.contains(evidence_id) {
                    duplicate_basis_evidence_ids.push(*evidence_id);
                }
                continue;
            }

            let snapshot = ledger.evidence(*evidence_id).map(|record| RevisionEvidenceSnapshot {
                evidence_id: record.id,
                claim_id: record.claim_id,
                kind: record.kind,
                polarity: record.polarity,
                provenance_id: record.provenance_id,
                observed_at_cycle: record.observed_at_cycle,
                context: record.context.clone(),
                method: record.method.clone(),
            });

            if let Some(record) = &snapshot {
                if evaluated_at_cycle < record.observed_at_cycle {
                    return Err(BeliefRevisionReceiptError::EvaluationPredatesBasisEvidence {
                        evidence_id: *evidence_id,
                        evaluated_at_cycle,
                        evidence_observed_at_cycle: record.observed_at_cycle,
                    });
                }
            }

            basis.push(RevisionEvidenceReference {
                requested_id: *evidence_id,
                snapshot,
            });
        }

        // Compute the decision inside the receipt boundary rather than accepting a
        // caller-supplied decision that could have been evaluated against different inputs.
        let decision = BeliefRevisionGate::evaluate(
            ledger,
            proposal,
            policy,
            calibration,
            uncertainty,
        );

        let id = BeliefRevisionReceiptId(self.next_id);
        self.next_id += 1;
        self.receipts.push(BeliefRevisionReceipt {
            id,
            claim_id: proposal.claim_id,
            proposed_delta: proposal.update.delta.get(),
            rationale: proposal.update.rationale.clone(),
            basis,
            duplicate_basis_evidence_ids,
            policy: policy.clone(),
            calibration,
            uncertainty: uncertainty.cloned(),
            decision,
            evaluated_at_cycle,
        });
        Ok(id)
    }

    pub fn get(&self, id: BeliefRevisionReceiptId) -> Option<&BeliefRevisionReceipt> {
        self.receipts.iter().find(|receipt| receipt.id == id)
    }

    pub fn receipts(&self) -> &[BeliefRevisionReceipt] {
        &self.receipts
    }

    pub fn receipts_for_claim(&self, claim_id: ClaimId) -> Vec<&BeliefRevisionReceipt> {
        self.receipts
            .iter()
            .filter(|receipt| receipt.claim_id == claim_id)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.receipts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.receipts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefRevisionFailure, EvidencePolarity, EpistemicLedger, EvidenceKind,
    };

    fn fixture(
        polarity: EvidencePolarity,
    ) -> (
        EpistemicLedger,
        ClaimId,
        EvidenceId,
        BeliefRevisionPolicy,
    ) {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("lab", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim(
            "X predicts Y",
            super::super::claim_evidence::ClaimKind::Predictive,
            None,
            None,
            1,
        );
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                polarity,
                provenance,
                2,
                Some("fixture".into()),
                Some("protocol-v1".into()),
            )
            .unwrap();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        (ledger, claim, evidence, policy)
    }

    #[test]
    fn records_eligible_and_rejected_decisions() {
        let (ledger, claim, support, policy) = fixture(EvidencePolarity::Supports);
        let mut history = BeliefRevisionHistory::new();

        let eligible = EpistemicRevisionProposal::new(claim, 0.10, vec![support], "support")
            .unwrap();
        let eligible_id = history
            .evaluate_and_record(&ledger, &eligible, &policy, None, None, 3)
            .unwrap();
        assert!(history.get(eligible_id).unwrap().eligible());

        let rejected = EpistemicRevisionProposal::new(claim, -0.10, vec![support], "wrong direction")
            .unwrap();
        let rejected_id = history
            .evaluate_and_record(&ledger, &rejected, &policy, None, None, 3)
            .unwrap();
        let rejected_receipt = history.get(rejected_id).unwrap();
        assert!(!rejected_receipt.eligible());
        assert!(rejected_receipt
            .decision()
            .failures()
            .contains(&BeliefRevisionFailure::NegativeDeltaLacksContradictingEvidence));
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn duplicate_basis_ids_are_snapshotted_once_and_reported() {
        let (ledger, claim, support, policy) = fixture(EvidencePolarity::Supports);
        let proposal = EpistemicRevisionProposal::new(
            claim,
            0.10,
            vec![support, support, support],
            "duplicate basis",
        )
        .unwrap();
        let mut history = BeliefRevisionHistory::new();
        let id = history
            .evaluate_and_record(&ledger, &proposal, &policy, None, None, 3)
            .unwrap();
        let receipt = history.get(id).unwrap();
        assert_eq!(receipt.basis().len(), 1);
        assert_eq!(receipt.duplicate_basis_evidence_ids(), &[support]);
    }

    #[test]
    fn evidence_snapshot_preserves_semantics_at_evaluation_time() {
        let (ledger, claim, support, policy) = fixture(EvidencePolarity::Supports);
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![support], "support")
            .unwrap();
        let mut history = BeliefRevisionHistory::new();
        let id = history
            .evaluate_and_record(&ledger, &proposal, &policy, None, None, 3)
            .unwrap();
        let snapshot = history.get(id).unwrap().basis()[0].snapshot.as_ref().unwrap();
        assert_eq!(snapshot.evidence_id, support);
        assert_eq!(snapshot.claim_id, claim);
        assert_eq!(snapshot.kind, EvidenceKind::Measurement);
        assert_eq!(snapshot.polarity, EvidencePolarity::Supports);
        assert_eq!(snapshot.observed_at_cycle, 2);
        assert_eq!(snapshot.context.as_deref(), Some("fixture"));
        assert_eq!(snapshot.method.as_deref(), Some("protocol-v1"));
    }

    #[test]
    fn unknown_evidence_can_be_recorded_for_rejected_audit_history() {
        let (ledger, claim, _, policy) = fixture(EvidencePolarity::Supports);
        let missing = EvidenceId(9999);
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![missing], "missing basis")
            .unwrap();
        let mut history = BeliefRevisionHistory::new();
        let id = history
            .evaluate_and_record(&ledger, &proposal, &policy, None, None, 3)
            .unwrap();
        let receipt = history.get(id).unwrap();
        assert!(!receipt.eligible());
        assert_eq!(receipt.basis()[0].requested_id, missing);
        assert!(receipt.basis()[0].snapshot.is_none());
        assert!(receipt
            .decision()
            .failures()
            .contains(&BeliefRevisionFailure::UnknownEvidence(missing)));
    }

    #[test]
    fn evaluation_cannot_predate_known_basis_evidence() {
        let (ledger, claim, support, policy) = fixture(EvidencePolarity::Supports);
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![support], "support")
            .unwrap();
        let mut history = BeliefRevisionHistory::new();
        assert_eq!(
            history
                .evaluate_and_record(&ledger, &proposal, &policy, None, None, 1)
                .unwrap_err(),
            BeliefRevisionReceiptError::EvaluationPredatesBasisEvidence {
                evidence_id: support,
                evaluated_at_cycle: 1,
                evidence_observed_at_cycle: 2,
            }
        );
        assert!(history.is_empty());
    }

    #[test]
    fn recording_decision_does_not_mutate_ledger() {
        let (ledger, claim, support, policy) = fixture(EvidencePolarity::Supports);
        let claim_count = ledger.claim_count();
        let evidence_count = ledger.evidence_count();
        let provenance_count = ledger.provenance_count();
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![support], "support")
            .unwrap();
        let mut history = BeliefRevisionHistory::new();
        history
            .evaluate_and_record(&ledger, &proposal, &policy, None, None, 3)
            .unwrap();
        assert_eq!(ledger.claim_count(), claim_count);
        assert_eq!(ledger.evidence_count(), evidence_count);
        assert_eq!(ledger.provenance_count(), provenance_count);
    }
}
