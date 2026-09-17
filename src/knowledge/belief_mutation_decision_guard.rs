// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Same-call decision/seal guard for EKM-028 belief mutation transactions.
//!
//! `BeliefRevisionEvidenceSeal::capture` protects a previously created revision
//! receipt, but a caller could otherwise evaluate a decision and delay seal
//! capture until after the ledger changed. This guard makes revision evaluation,
//! complete temporal validation, and evidence-census sealing one operation while
//! the ledger is borrowed immutably.

use super::belief_mutation_transaction::{BeliefMutationSealError, BeliefRevisionEvidenceSeal};
use super::belief_revision_gate::{
    BeliefRevisionPolicy, CalibrationSnapshot, EpistemicRevisionProposal,
};
use super::belief_revision_receipt::{
    BeliefRevisionHistory, BeliefRevisionReceiptError, BeliefRevisionReceiptId,
};
use super::claim_evidence::{ClaimId, EpistemicLedger, ProvenanceId};
use super::epistemic_vector::ClaimUncertaintyAssessment;
use std::collections::HashSet;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefMutationDecisionGuardError {
    RevisionReceipt(BeliefRevisionReceiptError),
    RevisionReceiptMissing(BeliefRevisionReceiptId),
    UnknownClaim(ClaimId),
    UnknownProvenance(ProvenanceId),
    ProvenancePostdatesDecision {
        provenance_id: ProvenanceId,
        recorded_at_cycle: u64,
        decision_cycle: u64,
    },
    Seal(BeliefMutationSealError),
}

impl fmt::Display for BeliefMutationDecisionGuardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RevisionReceipt(error) => write!(f, "belief revision receipt failed: {error}"),
            Self::RevisionReceiptMissing(id) => {
                write!(f, "belief revision receipt {} is missing after evaluation", id.0)
            }
            Self::UnknownClaim(id) => write!(f, "unknown claim {}", id.0),
            Self::UnknownProvenance(id) => write!(f, "unknown provenance {}", id.0),
            Self::ProvenancePostdatesDecision {
                provenance_id,
                recorded_at_cycle,
                decision_cycle,
            } => write!(
                f,
                "provenance {} recorded at cycle {recorded_at_cycle} postdates decision cycle {decision_cycle}",
                provenance_id.0
            ),
            Self::Seal(error) => write!(f, "belief mutation seal failed: {error}"),
        }
    }
}

impl Error for BeliefMutationDecisionGuardError {}

impl From<BeliefRevisionReceiptError> for BeliefMutationDecisionGuardError {
    fn from(value: BeliefRevisionReceiptError) -> Self {
        Self::RevisionReceipt(value)
    }
}

impl From<BeliefMutationSealError> for BeliefMutationDecisionGuardError {
    fn from(value: BeliefMutationSealError) -> Self {
        Self::Seal(value)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BeliefMutationDecisionGuard;

impl BeliefMutationDecisionGuard {
    /// Evaluate a belief revision and seal the complete claim-evidence census in
    /// the same call.
    ///
    /// The immutable ledger borrow spans both operations. This means safe callers
    /// cannot insert evidence between the revision decision and seal capture.
    pub fn evaluate_and_seal(
        ledger: &EpistemicLedger,
        history: &mut BeliefRevisionHistory,
        proposal: &EpistemicRevisionProposal,
        policy: &BeliefRevisionPolicy,
        calibration: Option<CalibrationSnapshot>,
        uncertainty: Option<&ClaimUncertaintyAssessment>,
        evaluated_at_cycle: u64,
    ) -> Result<BeliefRevisionEvidenceSeal, BeliefMutationDecisionGuardError> {
        let receipt_id = history.evaluate_and_record(
            ledger,
            proposal,
            policy,
            calibration,
            uncertainty,
            evaluated_at_cycle,
        )?;
        let receipt = history
            .get(receipt_id)
            .ok_or(BeliefMutationDecisionGuardError::RevisionReceiptMissing(
                receipt_id,
            ))?;

        validate_claim_provenance_time(ledger, receipt.claim_id(), evaluated_at_cycle)?;
        BeliefRevisionEvidenceSeal::capture(ledger, receipt, evaluated_at_cycle).map_err(Into::into)
    }
}

fn validate_claim_provenance_time(
    ledger: &EpistemicLedger,
    claim_id: ClaimId,
    decision_cycle: u64,
) -> Result<(), BeliefMutationDecisionGuardError> {
    let claim = ledger
        .claim(claim_id)
        .ok_or(BeliefMutationDecisionGuardError::UnknownClaim(claim_id))?;
    let mut visited = HashSet::new();
    for evidence_id in &claim.evidence_ids {
        if let Some(evidence) = ledger.evidence(*evidence_id) {
            validate_provenance_ancestry(
                ledger,
                evidence.provenance_id,
                decision_cycle,
                &mut visited,
            )?;
        }
    }
    Ok(())
}

fn validate_provenance_ancestry(
    ledger: &EpistemicLedger,
    provenance_id: ProvenanceId,
    decision_cycle: u64,
    visited: &mut HashSet<ProvenanceId>,
) -> Result<(), BeliefMutationDecisionGuardError> {
    if !visited.insert(provenance_id) {
        return Ok(());
    }
    let provenance = ledger
        .provenance(provenance_id)
        .ok_or(BeliefMutationDecisionGuardError::UnknownProvenance(
            provenance_id,
        ))?;
    if provenance.recorded_at_cycle > decision_cycle {
        return Err(
            BeliefMutationDecisionGuardError::ProvenancePostdatesDecision {
                provenance_id,
                recorded_at_cycle: provenance.recorded_at_cycle,
                decision_cycle,
            },
        );
    }
    for parent in &provenance.parent_ids {
        validate_provenance_ancestry(ledger, *parent, decision_cycle, visited)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationSealError, ClaimKind, EvidenceKind, EvidencePolarity,
    };

    fn policy() -> BeliefRevisionPolicy {
        BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap()
    }

    #[test]
    fn evaluation_and_seal_share_one_decision_cycle() {
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
                None,
                None,
            )
            .unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
            .unwrap();
        let mut history = BeliefRevisionHistory::new();

        let seal = BeliefMutationDecisionGuard::evaluate_and_seal(
            &ledger,
            &mut history,
            &proposal,
            &policy(),
            None,
            None,
            3,
        )
        .unwrap();

        assert_eq!(seal.sealed_at_cycle(), 3);
        assert_eq!(history.len(), 1);
        assert_eq!(seal.source_revision_receipt_id().0, 1);
        assert_eq!(seal.evidence().len(), 1);
    }

    #[test]
    fn future_dated_provenance_ancestor_fails_closed() {
        let mut ledger = EpistemicLedger::new();
        let future_root = ledger
            .add_provenance("future-root", None, None, 50, vec![])
            .unwrap();
        let derived = ledger
            .add_provenance("derived", None, None, 2, vec![future_root])
            .unwrap();
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                derived,
                2,
                None,
                None,
            )
            .unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
            .unwrap();
        let mut history = BeliefRevisionHistory::new();

        assert!(matches!(
            BeliefMutationDecisionGuard::evaluate_and_seal(
                &ledger,
                &mut history,
                &proposal,
                &policy(),
                None,
                None,
                3,
            ),
            Err(BeliefMutationDecisionGuardError::ProvenancePostdatesDecision {
                provenance_id,
                ..
            }) if provenance_id == future_root
        ));
    }

    #[test]
    fn future_dated_contextual_evidence_is_rejected_by_same_call_seal() {
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
                None,
                None,
            )
            .unwrap();
        ledger
            .add_evidence(
                claim,
                EvidenceKind::Observation,
                EvidencePolarity::Contextualizes,
                provenance,
                9,
                None,
                None,
            )
            .unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
            .unwrap();
        let mut history = BeliefRevisionHistory::new();

        assert!(matches!(
            BeliefMutationDecisionGuard::evaluate_and_seal(
                &ledger,
                &mut history,
                &proposal,
                &policy(),
                None,
                None,
                3,
            ),
            Err(BeliefMutationDecisionGuardError::Seal(
                BeliefMutationSealError::EvidencePostdatesSeal { .. }
            ))
        ));
    }

    #[test]
    fn rejected_revision_is_recorded_but_cannot_be_sealed() {
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
                None,
                None,
            )
            .unwrap();
        let proposal = EpistemicRevisionProposal::new(
            claim,
            -0.10,
            vec![evidence],
            "wrong-direction negative revision",
        )
        .unwrap();
        let mut history = BeliefRevisionHistory::new();

        assert!(matches!(
            BeliefMutationDecisionGuard::evaluate_and_seal(
                &ledger,
                &mut history,
                &proposal,
                &policy(),
                None,
                None,
                3,
            ),
            Err(BeliefMutationDecisionGuardError::Seal(
                BeliefMutationSealError::RevisionReceiptNotEligible(_)
            ))
        ));
        assert_eq!(history.len(), 1);
        assert!(!history.receipts()[0].eligible());
    }
}
