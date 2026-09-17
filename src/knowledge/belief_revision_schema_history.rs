// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Schema-bound audit history for belief-revision decisions.
//!
//! EKM-036 defines explicit policy and decision snapshot schemas. This module
//! binds those schemas to the existing immutable revision receipts at evaluation
//! time so future persistence never needs to infer policy semantics afterward.
//!
//! The wrapper is audit-only. It delegates the actual gate evaluation and receipt
//! creation to [`BeliefRevisionHistory`] and adds no belief-mutation authority.

use super::belief_revision_gate::{
    BeliefRevisionPolicyError, CalibrationSnapshot, EpistemicRevisionProposal,
};
use super::belief_revision_receipt::{
    BeliefRevisionHistory, BeliefRevisionReceiptError, BeliefRevisionReceiptId,
};
use super::belief_revision_snapshot::{
    BeliefRevisionDecisionSnapshotV1, BeliefRevisionPolicySchemaV1,
};
use super::claim_evidence::{ClaimId, EpistemicLedger};
use super::epistemic_vector::ClaimUncertaintyAssessment;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct SchemaBoundBeliefRevisionRecordV1 {
    pub receipt_id: BeliefRevisionReceiptId,
    pub claim_id: ClaimId,
    pub proposed_delta: f32,
    pub evaluated_at_cycle: u64,
    pub policy_schema: BeliefRevisionPolicySchemaV1,
    pub decision_snapshot: BeliefRevisionDecisionSnapshotV1,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BeliefRevisionSchemaHistoryV1 {
    records: Vec<SchemaBoundBeliefRevisionRecordV1>,
}

impl BeliefRevisionSchemaHistoryV1 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn records(&self) -> &[SchemaBoundBeliefRevisionRecordV1] {
        &self.records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn get(
        &self,
        id: BeliefRevisionReceiptId,
    ) -> Option<&SchemaBoundBeliefRevisionRecordV1> {
        self.records.iter().find(|record| record.receipt_id == id)
    }

    /// Evaluate through the existing receipt path while binding the canonical
    /// policy schema and typed decision snapshot to the same immutable receipt ID.
    ///
    /// The histories must already be exactly aligned. This is checked before the
    /// underlying receipt history is mutated so a divergence never becomes a
    /// partially recorded schema sidecar.
    pub fn evaluate_and_record(
        &mut self,
        history: &mut BeliefRevisionHistory,
        ledger: &EpistemicLedger,
        proposal: &EpistemicRevisionProposal,
        policy_schema: &BeliefRevisionPolicySchemaV1,
        calibration: Option<CalibrationSnapshot>,
        uncertainty: Option<&ClaimUncertaintyAssessment>,
        evaluated_at_cycle: u64,
    ) -> Result<BeliefRevisionReceiptId, BeliefRevisionSchemaHistoryError> {
        self.validate_alignment(history)?;

        let policy = policy_schema
            .build_policy()
            .map_err(BeliefRevisionSchemaHistoryError::Policy)?;
        let receipt_id = history
            .evaluate_and_record(
                ledger,
                proposal,
                &policy,
                calibration,
                uncertainty,
                evaluated_at_cycle,
            )
            .map_err(BeliefRevisionSchemaHistoryError::Receipt)?;

        let expected_id = BeliefRevisionReceiptId(self.records.len() as u64 + 1);
        if receipt_id != expected_id {
            return Err(BeliefRevisionSchemaHistoryError::UnexpectedReceiptId {
                expected: expected_id,
                actual: receipt_id,
            });
        }
        let receipt = history
            .get(receipt_id)
            .ok_or(BeliefRevisionSchemaHistoryError::MissingReceipt(receipt_id))?;

        let record = SchemaBoundBeliefRevisionRecordV1 {
            receipt_id,
            claim_id: receipt.claim_id(),
            proposed_delta: receipt.proposed_delta(),
            evaluated_at_cycle: receipt.evaluated_at_cycle(),
            policy_schema: policy_schema.clone(),
            decision_snapshot: BeliefRevisionDecisionSnapshotV1::capture(receipt.decision()),
        };
        self.records.push(record);
        Ok(receipt_id)
    }

    /// Verify that every schema sidecar still describes the exact immutable
    /// decision receipt at the same ID. No mutation occurs.
    pub fn validate_alignment(
        &self,
        history: &BeliefRevisionHistory,
    ) -> Result<(), BeliefRevisionSchemaHistoryError> {
        if history.len() != self.records.len() {
            return Err(BeliefRevisionSchemaHistoryError::HistoryLengthMismatch {
                receipts: history.len(),
                schema_records: self.records.len(),
            });
        }

        for (index, record) in self.records.iter().enumerate() {
            let expected_id = BeliefRevisionReceiptId(index as u64 + 1);
            if record.receipt_id != expected_id {
                return Err(BeliefRevisionSchemaHistoryError::SchemaIdSequenceBroken {
                    expected: expected_id,
                    actual: record.receipt_id,
                });
            }
            let receipt = history
                .get(record.receipt_id)
                .ok_or(BeliefRevisionSchemaHistoryError::MissingReceipt(
                    record.receipt_id,
                ))?;
            if receipt.claim_id() != record.claim_id {
                return Err(BeliefRevisionSchemaHistoryError::ClaimMismatch {
                    receipt_id: record.receipt_id,
                    receipt_claim: receipt.claim_id(),
                    schema_claim: record.claim_id,
                });
            }
            if receipt.proposed_delta().to_bits() != record.proposed_delta.to_bits() {
                return Err(BeliefRevisionSchemaHistoryError::DeltaMismatch(
                    record.receipt_id,
                ));
            }
            if receipt.evaluated_at_cycle() != record.evaluated_at_cycle {
                return Err(BeliefRevisionSchemaHistoryError::EvaluationCycleMismatch {
                    receipt_id: record.receipt_id,
                    receipt_cycle: receipt.evaluated_at_cycle(),
                    schema_cycle: record.evaluated_at_cycle,
                });
            }
            let live_decision = BeliefRevisionDecisionSnapshotV1::capture(receipt.decision());
            if live_decision != record.decision_snapshot {
                return Err(BeliefRevisionSchemaHistoryError::DecisionSnapshotMismatch(
                    record.receipt_id,
                ));
            }
            let rebuilt_policy = record
                .policy_schema
                .build_policy()
                .map_err(BeliefRevisionSchemaHistoryError::Policy)?;
            if receipt.policy() != &rebuilt_policy {
                return Err(BeliefRevisionSchemaHistoryError::PolicySchemaMismatch(
                    record.receipt_id,
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefRevisionSchemaHistoryError {
    Policy(BeliefRevisionPolicyError),
    Receipt(BeliefRevisionReceiptError),
    HistoryLengthMismatch {
        receipts: usize,
        schema_records: usize,
    },
    UnexpectedReceiptId {
        expected: BeliefRevisionReceiptId,
        actual: BeliefRevisionReceiptId,
    },
    MissingReceipt(BeliefRevisionReceiptId),
    SchemaIdSequenceBroken {
        expected: BeliefRevisionReceiptId,
        actual: BeliefRevisionReceiptId,
    },
    ClaimMismatch {
        receipt_id: BeliefRevisionReceiptId,
        receipt_claim: ClaimId,
        schema_claim: ClaimId,
    },
    DeltaMismatch(BeliefRevisionReceiptId),
    EvaluationCycleMismatch {
        receipt_id: BeliefRevisionReceiptId,
        receipt_cycle: u64,
        schema_cycle: u64,
    },
    DecisionSnapshotMismatch(BeliefRevisionReceiptId),
    PolicySchemaMismatch(BeliefRevisionReceiptId),
}

impl fmt::Display for BeliefRevisionSchemaHistoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "belief revision schema history invalid: {self:?}")
    }
}

impl Error for BeliefRevisionSchemaHistoryError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{ClaimKind, EvidenceKind, EvidencePolarity};

    fn fixture() -> (
        EpistemicLedger,
        ClaimId,
        super::super::claim_evidence::EvidenceId,
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
                None,
                None,
            )
            .unwrap();
        (ledger, claim, evidence)
    }

    #[test]
    fn evaluation_records_receipt_and_schema_atomically_when_aligned() {
        let (ledger, claim, evidence) = fixture();
        let proposal = EpistemicRevisionProposal::new(claim, 0.1, vec![evidence], "measurement")
            .unwrap();
        let schema = BeliefRevisionPolicySchemaV1::new(0.2, 1, false, 0, 1.0).unwrap();
        let mut receipts = BeliefRevisionHistory::new();
        let mut schemas = BeliefRevisionSchemaHistoryV1::new();

        let id = schemas
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
        assert_eq!(id, BeliefRevisionReceiptId(1));
        assert_eq!(receipts.len(), 1);
        assert_eq!(schemas.len(), 1);
        schemas.validate_alignment(&receipts).unwrap();
    }

    #[test]
    fn raw_receipt_insertion_is_detected_before_sidecar_mutation() {
        let (ledger, claim, evidence) = fixture();
        let proposal = EpistemicRevisionProposal::new(claim, 0.1, vec![evidence], "measurement")
            .unwrap();
        let schema = BeliefRevisionPolicySchemaV1::new(0.2, 1, false, 0, 1.0).unwrap();
        let policy = schema.build_policy().unwrap();
        let mut receipts = BeliefRevisionHistory::new();
        receipts
            .evaluate_and_record(&ledger, &proposal, &policy, None, None, 3)
            .unwrap();
        let mut schemas = BeliefRevisionSchemaHistoryV1::new();

        assert_eq!(
            schemas
                .evaluate_and_record(
                    &mut receipts,
                    &ledger,
                    &proposal,
                    &schema,
                    None,
                    None,
                    3,
                )
                .unwrap_err(),
            BeliefRevisionSchemaHistoryError::HistoryLengthMismatch {
                receipts: 1,
                schema_records: 0,
            }
        );
        assert!(schemas.is_empty());
        assert_eq!(receipts.len(), 1);
    }

    #[test]
    fn policy_schema_is_verified_against_immutable_receipt_policy() {
        let (ledger, claim, evidence) = fixture();
        let proposal = EpistemicRevisionProposal::new(claim, 0.1, vec![evidence], "measurement")
            .unwrap();
        let schema = BeliefRevisionPolicySchemaV1::new(0.2, 1, false, 0, 1.0).unwrap();
        let mut receipts = BeliefRevisionHistory::new();
        let mut schemas = BeliefRevisionSchemaHistoryV1::new();
        let id = schemas
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

        schemas.records[0].policy_schema =
            BeliefRevisionPolicySchemaV1::new(0.05, 1, false, 0, 1.0).unwrap();
        assert_eq!(
            schemas.validate_alignment(&receipts).unwrap_err(),
            BeliefRevisionSchemaHistoryError::PolicySchemaMismatch(id)
        );
    }
}
