// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persistence capsule for schema-bound belief-revision history.
//!
//! EKM-037 binds canonical policy schemas to immutable revision receipts at
//! evaluation time. This module makes that binding restart-capturable while
//! failing closed on mixed legacy/new histories.
//!
//! The capsule is export/validation only. It does not hydrate history, mutate
//! support, or activate restored state.

use super::belief_revision_persistence::{
    BeliefRevisionHistoryCapsuleV1, BeliefRevisionPersistenceVersion,
};
use super::belief_revision_receipt::{BeliefRevisionHistory, BeliefRevisionReceiptId};
use super::belief_revision_schema_history::{
    BeliefRevisionSchemaHistoryError, BeliefRevisionSchemaHistoryV1,
    SchemaBoundBeliefRevisionRecordV1,
};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefRevisionSchemaPersistenceVersion {
    V1,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefRevisionSchemaHistoryCapsuleV1 {
    version: BeliefRevisionSchemaPersistenceVersion,
    captured_at_cycle: u64,
    linked_revision_capture_cycle: u64,
    linked_revision_count: usize,
    linked_next_receipt_id: BeliefRevisionReceiptId,
    records: Vec<SchemaBoundBeliefRevisionRecordV1>,
}

impl BeliefRevisionSchemaHistoryCapsuleV1 {
    pub fn capture(
        schema_history: &BeliefRevisionSchemaHistoryV1,
        live_history: &BeliefRevisionHistory,
        revision_capsule: &BeliefRevisionHistoryCapsuleV1,
        captured_at_cycle: u64,
    ) -> Result<Self, BeliefRevisionSchemaPersistenceError> {
        if revision_capsule.version() != BeliefRevisionPersistenceVersion::V1 {
            return Err(BeliefRevisionSchemaPersistenceError::UnsupportedRevisionVersion);
        }
        if revision_capsule.captured_at_cycle() != captured_at_cycle {
            return Err(BeliefRevisionSchemaPersistenceError::CaptureCycleMismatch {
                requested: captured_at_cycle,
                revision_capsule: revision_capsule.captured_at_cycle(),
            });
        }

        schema_history
            .validate_alignment(live_history)
            .map_err(BeliefRevisionSchemaPersistenceError::SchemaHistory)?;

        if revision_capsule.receipts().len() != schema_history.len() {
            return Err(BeliefRevisionSchemaPersistenceError::LegacyReceiptWithoutSchema {
                revision_receipts: revision_capsule.receipts().len(),
                schema_records: schema_history.len(),
            });
        }
        if live_history.len() != revision_capsule.receipts().len() {
            return Err(BeliefRevisionSchemaPersistenceError::LiveRevisionCountMismatch {
                live: live_history.len(),
                persisted: revision_capsule.receipts().len(),
            });
        }

        for (receipt, schema) in revision_capsule
            .receipts()
            .iter()
            .zip(schema_history.records())
        {
            if receipt.id() != schema.receipt_id {
                return Err(BeliefRevisionSchemaPersistenceError::ReceiptIdMismatch {
                    receipt: receipt.id(),
                    schema: schema.receipt_id,
                });
            }
            if receipt.claim_id() != schema.claim_id {
                return Err(BeliefRevisionSchemaPersistenceError::ClaimMismatch(
                    schema.receipt_id,
                ));
            }
            if receipt.proposed_delta().to_bits() != schema.proposed_delta.to_bits() {
                return Err(BeliefRevisionSchemaPersistenceError::DeltaMismatch(
                    schema.receipt_id,
                ));
            }
            if receipt.evaluated_at_cycle() != schema.evaluated_at_cycle {
                return Err(BeliefRevisionSchemaPersistenceError::EvaluationCycleMismatch(
                    schema.receipt_id,
                ));
            }
            if receipt.evaluated_at_cycle() > captured_at_cycle {
                return Err(BeliefRevisionSchemaPersistenceError::ReceiptPostdatesCapture {
                    receipt_id: receipt.id(),
                    receipt_cycle: receipt.evaluated_at_cycle(),
                    capture_cycle: captured_at_cycle,
                });
            }
            if schema.decision_snapshot
                != super::belief_revision_snapshot::BeliefRevisionDecisionSnapshotV1::capture(
                    receipt.decision(),
                )
            {
                return Err(BeliefRevisionSchemaPersistenceError::DecisionMismatch(
                    schema.receipt_id,
                ));
            }
            let rebuilt = schema
                .policy_schema
                .build_policy()
                .map_err(BeliefRevisionSchemaPersistenceError::Policy)?;
            if receipt.policy() != &rebuilt {
                return Err(BeliefRevisionSchemaPersistenceError::PolicyMismatch(
                    schema.receipt_id,
                ));
            }
        }

        Ok(Self {
            version: BeliefRevisionSchemaPersistenceVersion::V1,
            captured_at_cycle,
            linked_revision_capture_cycle: revision_capsule.captured_at_cycle(),
            linked_revision_count: revision_capsule.receipts().len(),
            linked_next_receipt_id: revision_capsule.next_receipt_id(),
            records: schema_history.records().to_vec(),
        })
    }

    pub fn version(&self) -> BeliefRevisionSchemaPersistenceVersion {
        self.version
    }

    pub fn captured_at_cycle(&self) -> u64 {
        self.captured_at_cycle
    }

    pub fn linked_revision_capture_cycle(&self) -> u64 {
        self.linked_revision_capture_cycle
    }

    pub fn linked_revision_count(&self) -> usize {
        self.linked_revision_count
    }

    pub fn linked_next_receipt_id(&self) -> BeliefRevisionReceiptId {
        self.linked_next_receipt_id
    }

    pub fn records(&self) -> &[SchemaBoundBeliefRevisionRecordV1] {
        &self.records
    }

    pub fn validate_live(
        &self,
        schema_history: &BeliefRevisionSchemaHistoryV1,
        live_history: &BeliefRevisionHistory,
        revision_capsule: &BeliefRevisionHistoryCapsuleV1,
        observed_at_cycle: u64,
    ) -> Result<(), BeliefRevisionSchemaPersistenceError> {
        if observed_at_cycle < self.captured_at_cycle {
            return Err(BeliefRevisionSchemaPersistenceError::ObservationPredatesCapture {
                observed: observed_at_cycle,
                captured: self.captured_at_cycle,
            });
        }
        let live = Self::capture(
            schema_history,
            live_history,
            revision_capsule,
            self.captured_at_cycle,
        )?;
        if live != *self {
            return Err(BeliefRevisionSchemaPersistenceError::LiveCapsuleMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefRevisionSchemaPersistenceError {
    UnsupportedRevisionVersion,
    CaptureCycleMismatch {
        requested: u64,
        revision_capsule: u64,
    },
    SchemaHistory(BeliefRevisionSchemaHistoryError),
    LegacyReceiptWithoutSchema {
        revision_receipts: usize,
        schema_records: usize,
    },
    LiveRevisionCountMismatch {
        live: usize,
        persisted: usize,
    },
    ReceiptIdMismatch {
        receipt: BeliefRevisionReceiptId,
        schema: BeliefRevisionReceiptId,
    },
    ClaimMismatch(BeliefRevisionReceiptId),
    DeltaMismatch(BeliefRevisionReceiptId),
    EvaluationCycleMismatch(BeliefRevisionReceiptId),
    ReceiptPostdatesCapture {
        receipt_id: BeliefRevisionReceiptId,
        receipt_cycle: u64,
        capture_cycle: u64,
    },
    DecisionMismatch(BeliefRevisionReceiptId),
    Policy(super::belief_revision_gate::BeliefRevisionPolicyError),
    PolicyMismatch(BeliefRevisionReceiptId),
    ObservationPredatesCapture {
        observed: u64,
        captured: u64,
    },
    LiveCapsuleMismatch,
}

impl fmt::Display for BeliefRevisionSchemaPersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "belief revision schema persistence invalid: {self:?}")
    }
}

impl Error for BeliefRevisionSchemaPersistenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationPersistenceCapsuleV1, BeliefRevisionPolicySchemaV1, ClaimKind,
        EpistemicLedger, EpistemicRevisionProposal, EpistemicSupportStore, EvidenceKind,
        EvidencePolarity,
    };

    fn fixture() -> (
        EpistemicLedger,
        BeliefRevisionHistory,
        BeliefRevisionSchemaHistoryV1,
        BeliefRevisionHistoryCapsuleV1,
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
        let proposal = EpistemicRevisionProposal::new(claim, 0.1, vec![evidence], "measurement")
            .unwrap();
        let schema = BeliefRevisionPolicySchemaV1::new(0.2, 1, false, 0, 1.0).unwrap();
        let mut receipts = BeliefRevisionHistory::new();
        let mut schemas = BeliefRevisionSchemaHistoryV1::new();
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
            .unwrap();

        // EKM-031 requires a linked mutation capsule. An empty support store is
        // sufficient here because no belief mutation is being exercised.
        let store = EpistemicSupportStore::new();
        let mutations = BeliefMutationPersistenceCapsuleV1::capture(&store, &[], 4).unwrap();
        let revision_capsule =
            BeliefRevisionHistoryCapsuleV1::capture(&receipts, &mutations, 4).unwrap();
        (ledger, receipts, schemas, revision_capsule)
    }

    #[test]
    fn complete_schema_history_can_be_captured() {
        let (_ledger, receipts, schemas, revision_capsule) = fixture();
        let capsule = BeliefRevisionSchemaHistoryCapsuleV1::capture(
            &schemas,
            &receipts,
            &revision_capsule,
            4,
        )
        .unwrap();
        assert_eq!(capsule.records().len(), 1);
        assert_eq!(capsule.linked_revision_count(), 1);
        assert_eq!(capsule.linked_next_receipt_id(), BeliefRevisionReceiptId(2));
        capsule
            .validate_live(&schemas, &receipts, &revision_capsule, 5)
            .unwrap();
    }

    #[test]
    fn legacy_receipt_without_schema_fails_closed() {
        let (ledger, mut receipts, schemas, _revision_capsule) = fixture();
        let claim = ledger.claim(super::super::claim_evidence::ClaimId(1)).unwrap().id;
        let evidence = ledger.evidence_for_claim(claim)[0].id;
        let proposal = EpistemicRevisionProposal::new(claim, 0.05, vec![evidence], "legacy")
            .unwrap();
        let policy = BeliefRevisionPolicySchemaV1::new(0.2, 1, false, 0, 1.0)
            .unwrap()
            .build_policy()
            .unwrap();
        receipts
            .evaluate_and_record(&ledger, &proposal, &policy, None, None, 3)
            .unwrap();

        let store = EpistemicSupportStore::new();
        let mutations = BeliefMutationPersistenceCapsuleV1::capture(&store, &[], 4).unwrap();
        let revision_capsule =
            BeliefRevisionHistoryCapsuleV1::capture(&receipts, &mutations, 4).unwrap();
        assert_eq!(
            BeliefRevisionSchemaHistoryCapsuleV1::capture(
                &schemas,
                &receipts,
                &revision_capsule,
                4,
            )
            .unwrap_err(),
            BeliefRevisionSchemaPersistenceError::SchemaHistory(
                BeliefRevisionSchemaHistoryError::HistoryLengthMismatch {
                    receipts: 2,
                    schema_records: 1,
                }
            )
        );
    }
}
