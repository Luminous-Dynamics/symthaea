// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent semantic validation for EKM-040 typed schema-wire snapshots.
//!
//! The EKM-040 parser proves framing, bounded decoding and checksum integrity.
//! This module treats the decoded DTO as still untrusted and validates the
//! semantic invariants that can be established from the schema sidecar alone.
//! Evidence-dependent claims are intentionally deferred to a later top-level
//! restart-v2 bundle validator where the base ledger and receipts are present.

use super::belief_revision_receipt::BeliefRevisionReceiptId;
use super::belief_revision_schema_wire::{
    BeliefRevisionSchemaWireEncoding, BeliefRevisionSchemaWireSnapshotV1,
    BeliefRevisionSchemaWireVersion,
};
use super::belief_revision_snapshot::{
    uncertainty_dimension_tag, BeliefRevisionFailureSnapshotV1,
    BeliefRevisionPolicySchemaV1, BeliefRevisionSnapshotVersion,
    KnowledgeWeightRoutingFailureSnapshotV1,
};
use super::epistemic_vector::UncertaintyDimension;
use super::knowledge_weight_routing::{KnowledgeWeightDimension, KnowledgeWeightSource};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeliefRevisionSchemaWireValidationReport {
    pub captured_at_cycle: u64,
    pub record_count: usize,
    pub eligible_count: usize,
    pub rejected_count: usize,
    pub failure_count: usize,
}

pub struct BeliefRevisionSchemaWireValidator;

impl BeliefRevisionSchemaWireValidator {
    pub fn validate(
        snapshot: &BeliefRevisionSchemaWireSnapshotV1,
    ) -> Result<BeliefRevisionSchemaWireValidationReport, BeliefRevisionSchemaWireValidationError>
    {
        if snapshot.version != BeliefRevisionSchemaWireVersion::V1 {
            return Err(BeliefRevisionSchemaWireValidationError::UnsupportedVersion);
        }
        if snapshot.encoding != BeliefRevisionSchemaWireEncoding::ExplicitPolicyDecisionV1 {
            return Err(BeliefRevisionSchemaWireValidationError::UnsupportedEncoding);
        }
        if snapshot.captured_at_cycle != snapshot.linked_revision_capture_cycle {
            return Err(BeliefRevisionSchemaWireValidationError::CaptureCycleMismatch {
                schema_cycle: snapshot.captured_at_cycle,
                linked_revision_cycle: snapshot.linked_revision_capture_cycle,
            });
        }
        if snapshot.linked_revision_count != snapshot.records.len() as u64 {
            return Err(BeliefRevisionSchemaWireValidationError::LinkedRevisionCountMismatch {
                linked: snapshot.linked_revision_count,
                records: snapshot.records.len(),
            });
        }
        let expected_next = u64::try_from(snapshot.records.len())
            .map_err(|_| BeliefRevisionSchemaWireValidationError::LengthOverflow)?
            .checked_add(1)
            .ok_or(BeliefRevisionSchemaWireValidationError::ReceiptIdOverflow)?;
        if snapshot.linked_next_receipt_id != BeliefRevisionReceiptId(expected_next) {
            return Err(BeliefRevisionSchemaWireValidationError::NextReceiptIdMismatch {
                expected: BeliefRevisionReceiptId(expected_next),
                actual: snapshot.linked_next_receipt_id,
            });
        }

        let mut eligible_count = 0usize;
        let mut failure_count = 0usize;
        for (index, record) in snapshot.records.iter().enumerate() {
            let expected_id = BeliefRevisionReceiptId(
                u64::try_from(index)
                    .map_err(|_| BeliefRevisionSchemaWireValidationError::LengthOverflow)?
                    .checked_add(1)
                    .ok_or(BeliefRevisionSchemaWireValidationError::ReceiptIdOverflow)?,
            );
            if record.receipt_id != expected_id {
                return Err(BeliefRevisionSchemaWireValidationError::ReceiptIdSequenceBroken {
                    expected: expected_id,
                    actual: record.receipt_id,
                });
            }
            if record.evaluated_at_cycle > snapshot.captured_at_cycle {
                return Err(BeliefRevisionSchemaWireValidationError::ReceiptPostdatesCapture {
                    receipt_id: record.receipt_id,
                    evaluated_at_cycle: record.evaluated_at_cycle,
                    captured_at_cycle: snapshot.captured_at_cycle,
                });
            }
            if !record.proposed_delta.is_finite()
                || !(-1.0..=1.0).contains(&record.proposed_delta)
            {
                return Err(BeliefRevisionSchemaWireValidationError::InvalidDelta {
                    receipt_id: record.receipt_id,
                    delta: record.proposed_delta,
                });
            }

            validate_policy(record.receipt_id, &record.policy_schema)?;
            validate_decision(
                record.receipt_id,
                record.proposed_delta,
                &record.policy_schema,
                &record.decision_snapshot,
            )?;

            if record.decision_snapshot.eligible {
                eligible_count += 1;
            }
            failure_count = failure_count
                .checked_add(record.decision_snapshot.failures.len())
                .ok_or(BeliefRevisionSchemaWireValidationError::LengthOverflow)?;
        }

        Ok(BeliefRevisionSchemaWireValidationReport {
            captured_at_cycle: snapshot.captured_at_cycle,
            record_count: snapshot.records.len(),
            eligible_count,
            rejected_count: snapshot.records.len() - eligible_count,
            failure_count,
        })
    }
}

fn validate_policy(
    receipt_id: BeliefRevisionReceiptId,
    policy: &BeliefRevisionPolicySchemaV1,
) -> Result<(), BeliefRevisionSchemaWireValidationError> {
    policy
        .build_policy()
        .map_err(|_| BeliefRevisionSchemaWireValidationError::InvalidPolicy(receipt_id))?;

    let mut previous_tag = 0u8;
    for (dimension, maximum) in &policy.strengthen_uncertainty_caps {
        let tag = uncertainty_dimension_tag(*dimension);
        if tag <= previous_tag {
            return Err(
                BeliefRevisionSchemaWireValidationError::NonCanonicalUncertaintyCaps(receipt_id),
            );
        }
        previous_tag = tag;
        if !maximum.is_finite() || !(0.0..=1.0).contains(maximum) {
            return Err(BeliefRevisionSchemaWireValidationError::InvalidPolicy(receipt_id));
        }
    }
    Ok(())
}

fn validate_decision(
    receipt_id: BeliefRevisionReceiptId,
    proposed_delta: f32,
    policy: &BeliefRevisionPolicySchemaV1,
    decision: &super::belief_revision_snapshot::BeliefRevisionDecisionSnapshotV1,
) -> Result<(), BeliefRevisionSchemaWireValidationError> {
    if decision.version != BeliefRevisionSnapshotVersion::V1 {
        return Err(BeliefRevisionSchemaWireValidationError::UnsupportedDecisionVersion(
            receipt_id,
        ));
    }
    if decision.eligible != decision.failures.is_empty() {
        return Err(BeliefRevisionSchemaWireValidationError::EligibilityFailureMismatch {
            receipt_id,
            eligible: decision.eligible,
            failures: decision.failures.len(),
        });
    }

    for failure in &decision.failures {
        validate_failure(receipt_id, proposed_delta, policy, decision, failure)?;
    }
    Ok(())
}

fn validate_failure(
    receipt_id: BeliefRevisionReceiptId,
    proposed_delta: f32,
    policy: &BeliefRevisionPolicySchemaV1,
    decision: &super::belief_revision_snapshot::BeliefRevisionDecisionSnapshotV1,
    failure: &BeliefRevisionFailureSnapshotV1,
) -> Result<(), BeliefRevisionSchemaWireValidationError> {
    match failure {
        BeliefRevisionFailureSnapshotV1::WeightRoutingDenied(failures) => {
            if failures.is_empty() {
                return Err(BeliefRevisionSchemaWireValidationError::EmptyRoutingFailure(
                    receipt_id,
                ));
            }
            for failure in failures {
                validate_routing_failure(receipt_id, failure)?;
            }
        }
        BeliefRevisionFailureSnapshotV1::UnknownClaim(_)
        | BeliefRevisionFailureSnapshotV1::UnknownEvidence(_)
        | BeliefRevisionFailureSnapshotV1::EvidenceForDifferentClaim { .. } => {}
        BeliefRevisionFailureSnapshotV1::PositiveDeltaLacksSupportingEvidence
        | BeliefRevisionFailureSnapshotV1::PositiveDeltaIncludesContradictingEvidence => {
            if proposed_delta <= 0.0 {
                return Err(BeliefRevisionSchemaWireValidationError::FailureDirectionMismatch(
                    receipt_id,
                ));
            }
        }
        BeliefRevisionFailureSnapshotV1::NegativeDeltaLacksContradictingEvidence
        | BeliefRevisionFailureSnapshotV1::NegativeDeltaIncludesSupportingEvidence => {
            if proposed_delta >= 0.0 {
                return Err(BeliefRevisionSchemaWireValidationError::FailureDirectionMismatch(
                    receipt_id,
                ));
            }
        }
        BeliefRevisionFailureSnapshotV1::DeclaredProvenanceRootsBelowMinimum {
            required,
            actual,
        } => {
            if *required != policy.min_declared_provenance_roots
                || *actual != decision.declared_provenance_root_count
                || actual >= required
            {
                return Err(
                    BeliefRevisionSchemaWireValidationError::ProvenanceThresholdFailureMismatch(
                        receipt_id,
                    ),
                );
            }
        }
        BeliefRevisionFailureSnapshotV1::DeltaExceedsPolicy { maximum, actual } => {
            if maximum.to_bits() != policy.max_abs_delta.to_bits()
                || actual.to_bits() != proposed_delta.abs().to_bits()
                || *actual <= *maximum
            {
                return Err(BeliefRevisionSchemaWireValidationError::DeltaFailureMismatch(
                    receipt_id,
                ));
            }
        }
        BeliefRevisionFailureSnapshotV1::CalibrationMissing => {
            if !policy.require_calibration || proposed_delta == 0.0 {
                return Err(BeliefRevisionSchemaWireValidationError::CalibrationFailureMismatch(
                    receipt_id,
                ));
            }
        }
        BeliefRevisionFailureSnapshotV1::CalibrationSamplesBelowMinimum { required, actual } => {
            if !policy.require_calibration
                || proposed_delta == 0.0
                || *required != policy.min_calibration_samples
                || actual >= required
            {
                return Err(BeliefRevisionSchemaWireValidationError::CalibrationFailureMismatch(
                    receipt_id,
                ));
            }
        }
        BeliefRevisionFailureSnapshotV1::CalibrationEceAboveMaximum { maximum, actual } => {
            if !policy.require_calibration
                || proposed_delta == 0.0
                || maximum.to_bits() != policy.max_calibration_ece.to_bits()
                || actual <= maximum
            {
                return Err(BeliefRevisionSchemaWireValidationError::CalibrationFailureMismatch(
                    receipt_id,
                ));
            }
        }
        BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentMissing => {
            if !uncertainty_policy_active(policy) {
                return Err(BeliefRevisionSchemaWireValidationError::UncertaintyFailureMismatch(
                    receipt_id,
                ));
            }
        }
        BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentForDifferentClaim { .. }
        | BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentPredatesEvidence { .. } => {
            if !uncertainty_policy_active(policy) {
                return Err(BeliefRevisionSchemaWireValidationError::UncertaintyFailureMismatch(
                    receipt_id,
                ));
            }
        }
        BeliefRevisionFailureSnapshotV1::RequiredUncertaintyDimensionUnassessed(dimension) => {
            if proposed_delta <= 0.0 || policy_cap(policy, *dimension).is_none() {
                return Err(BeliefRevisionSchemaWireValidationError::UncertaintyFailureMismatch(
                    receipt_id,
                ));
            }
        }
        BeliefRevisionFailureSnapshotV1::UncertaintyAboveMaximum {
            dimension,
            maximum,
            actual,
        } => {
            let Some(policy_maximum) = policy_cap(policy, *dimension) else {
                return Err(BeliefRevisionSchemaWireValidationError::UncertaintyFailureMismatch(
                    receipt_id,
                ));
            };
            if proposed_delta <= 0.0
                || maximum.to_bits() != policy_maximum.to_bits()
                || actual <= maximum
            {
                return Err(BeliefRevisionSchemaWireValidationError::UncertaintyFailureMismatch(
                    receipt_id,
                ));
            }
        }
        BeliefRevisionFailureSnapshotV1::UnresolvedContradictionBlocksStrengthen { count } => {
            if proposed_delta <= 0.0
                || !policy.block_strengthen_with_unresolved_contradictions
                || *count == 0
            {
                return Err(BeliefRevisionSchemaWireValidationError::ContradictionFailureMismatch(
                    receipt_id,
                ));
            }
        }
        BeliefRevisionFailureSnapshotV1::CausalStrengthenLacksInterventionalBasis => {
            if proposed_delta <= 0.0 || !policy.require_intervention_for_causal_strengthen {
                return Err(BeliefRevisionSchemaWireValidationError::CausalFailureMismatch(
                    receipt_id,
                ));
            }
        }
    }
    Ok(())
}

fn validate_routing_failure(
    receipt_id: BeliefRevisionReceiptId,
    failure: &KnowledgeWeightRoutingFailureSnapshotV1,
) -> Result<(), BeliefRevisionSchemaWireValidationError> {
    match failure {
        KnowledgeWeightRoutingFailureSnapshotV1::SourceCannotUpdateDimension {
            source,
            dimension,
        } => {
            if source_may_target(*source, *dimension) {
                return Err(BeliefRevisionSchemaWireValidationError::RoutingFailureMismatch(
                    receipt_id,
                ));
            }
        }
        KnowledgeWeightRoutingFailureSnapshotV1::EpistemicUpdateHasNoEvidenceBasis => {}
    }
    Ok(())
}

fn source_may_target(source: KnowledgeWeightSource, dimension: KnowledgeWeightDimension) -> bool {
    match source {
        KnowledgeWeightSource::AdmittedEvidence => {
            dimension == KnowledgeWeightDimension::EpistemicSupport
        }
        KnowledgeWeightSource::Retrieval | KnowledgeWeightSource::SimilarityMatch => {
            dimension == KnowledgeWeightDimension::Accessibility
        }
        KnowledgeWeightSource::DreamReplay | KnowledgeWeightSource::CausalConsolidation => matches!(
            dimension,
            KnowledgeWeightDimension::Retention | KnowledgeWeightDimension::Consolidation
        ),
        KnowledgeWeightSource::MemoryDecay => dimension == KnowledgeWeightDimension::Retention,
        KnowledgeWeightSource::TaskRelevance => {
            dimension == KnowledgeWeightDimension::Accessibility
        }
    }
}

fn uncertainty_policy_active(policy: &BeliefRevisionPolicySchemaV1) -> bool {
    policy.require_uncertainty_assessment
        || policy.require_current_uncertainty
        || !policy.strengthen_uncertainty_caps.is_empty()
}

fn policy_cap(
    policy: &BeliefRevisionPolicySchemaV1,
    dimension: UncertaintyDimension,
) -> Option<f64> {
    policy
        .strengthen_uncertainty_caps
        .iter()
        .find(|(candidate, _)| *candidate == dimension)
        .map(|(_, maximum)| *maximum)
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefRevisionSchemaWireValidationError {
    UnsupportedVersion,
    UnsupportedEncoding,
    CaptureCycleMismatch {
        schema_cycle: u64,
        linked_revision_cycle: u64,
    },
    LinkedRevisionCountMismatch {
        linked: u64,
        records: usize,
    },
    NextReceiptIdMismatch {
        expected: BeliefRevisionReceiptId,
        actual: BeliefRevisionReceiptId,
    },
    ReceiptIdSequenceBroken {
        expected: BeliefRevisionReceiptId,
        actual: BeliefRevisionReceiptId,
    },
    ReceiptPostdatesCapture {
        receipt_id: BeliefRevisionReceiptId,
        evaluated_at_cycle: u64,
        captured_at_cycle: u64,
    },
    InvalidDelta {
        receipt_id: BeliefRevisionReceiptId,
        delta: f32,
    },
    InvalidPolicy(BeliefRevisionReceiptId),
    NonCanonicalUncertaintyCaps(BeliefRevisionReceiptId),
    UnsupportedDecisionVersion(BeliefRevisionReceiptId),
    EligibilityFailureMismatch {
        receipt_id: BeliefRevisionReceiptId,
        eligible: bool,
        failures: usize,
    },
    EmptyRoutingFailure(BeliefRevisionReceiptId),
    RoutingFailureMismatch(BeliefRevisionReceiptId),
    FailureDirectionMismatch(BeliefRevisionReceiptId),
    ProvenanceThresholdFailureMismatch(BeliefRevisionReceiptId),
    DeltaFailureMismatch(BeliefRevisionReceiptId),
    CalibrationFailureMismatch(BeliefRevisionReceiptId),
    UncertaintyFailureMismatch(BeliefRevisionReceiptId),
    ContradictionFailureMismatch(BeliefRevisionReceiptId),
    CausalFailureMismatch(BeliefRevisionReceiptId),
    LengthOverflow,
    ReceiptIdOverflow,
}

impl fmt::Display for BeliefRevisionSchemaWireValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "belief revision schema wire semantics invalid: {self:?}")
    }
}

impl Error for BeliefRevisionSchemaWireValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistory,
        BeliefRevisionHistoryCapsuleV1, BeliefRevisionPolicySchemaV1,
        BeliefRevisionSchemaHistoryCapsuleV1, BeliefRevisionSchemaHistoryV1,
        BeliefRevisionSchemaWireV1, ClaimKind, EpistemicLedger, EpistemicRevisionProposal,
        EpistemicSupportStore, EvidenceKind, EvidencePolarity,
    };

    fn snapshot() -> BeliefRevisionSchemaWireSnapshotV1 {
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
        let store = EpistemicSupportStore::new();
        let mutations = BeliefMutationPersistenceCapsuleV1::capture(&store, &[], 4).unwrap();
        let revisions = BeliefRevisionHistoryCapsuleV1::capture(&receipts, &mutations, 4).unwrap();
        let capsule = BeliefRevisionSchemaHistoryCapsuleV1::capture(
            &schemas,
            &receipts,
            &revisions,
            4,
        )
        .unwrap();
        let bytes = BeliefRevisionSchemaWireV1::encode(&capsule).unwrap();
        BeliefRevisionSchemaWireV1::decode(&bytes).unwrap()
    }

    #[test]
    fn valid_schema_snapshot_passes_semantic_validation() {
        let snapshot = snapshot();
        let report = BeliefRevisionSchemaWireValidator::validate(&snapshot).unwrap();
        assert_eq!(report.record_count, 1);
        assert_eq!(report.eligible_count, 1);
        assert_eq!(report.rejected_count, 0);
    }

    #[test]
    fn eligibility_cannot_disagree_with_failure_list() {
        let mut snapshot = snapshot();
        snapshot.records[0].decision_snapshot.eligible = false;
        assert!(matches!(
            BeliefRevisionSchemaWireValidator::validate(&snapshot),
            Err(BeliefRevisionSchemaWireValidationError::EligibilityFailureMismatch { .. })
        ));
    }

    #[test]
    fn delta_failure_payload_must_match_policy_and_proposal() {
        let mut snapshot = snapshot();
        snapshot.records[0].decision_snapshot.eligible = false;
        snapshot.records[0].decision_snapshot.failures = vec![
            BeliefRevisionFailureSnapshotV1::DeltaExceedsPolicy {
                maximum: 0.2,
                actual: 0.7,
            },
        ];
        assert_eq!(
            BeliefRevisionSchemaWireValidator::validate(&snapshot).unwrap_err(),
            BeliefRevisionSchemaWireValidationError::DeltaFailureMismatch(
                BeliefRevisionReceiptId(1)
            )
        );
    }

    #[test]
    fn receipt_ids_must_be_contiguous() {
        let mut snapshot = snapshot();
        snapshot.records[0].receipt_id = BeliefRevisionReceiptId(2);
        assert!(matches!(
            BeliefRevisionSchemaWireValidator::validate(&snapshot),
            Err(BeliefRevisionSchemaWireValidationError::ReceiptIdSequenceBroken { .. })
        ));
    }
}
