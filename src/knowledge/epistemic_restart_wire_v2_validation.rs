// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-component semantic validation for EKM-042 restart-v2 bundle snapshots.
//!
//! This validator composes the independent EKM-035 and EKM-041 validators, then
//! verifies that the base receipt history and typed schema sidecar describe the
//! same decisions. It reconstructs an internal verification-only ledger and
//! re-runs the existing proposal-only belief gate from typed inputs.
//!
//! The validator returns only a report. It does not construct a restart capsule,
//! quarantine image, writable history/store, authorization, or activation handle.

use super::belief_revision_gate::{BeliefRevisionGate, CalibrationSnapshot, EpistemicRevisionProposal};
use super::belief_revision_schema_wire_validation::{
    BeliefRevisionSchemaWireValidationError, BeliefRevisionSchemaWireValidator,
};
use super::belief_revision_snapshot::{
    knowledge_weight_dimension_tag, knowledge_weight_source_tag, uncertainty_dimension_tag,
    BeliefRevisionDecisionSnapshotV1, BeliefRevisionFailureSnapshotV1,
    KnowledgeWeightRoutingFailureSnapshotV1,
};
use super::claim_evidence::EpistemicLedger;
use super::epistemic_restart_wire::EpistemicRestartWireSnapshotV1;
use super::epistemic_restart_wire_v2::{
    EpistemicRestartWireSnapshotV2, EpistemicRestartWireV2Encoding, EpistemicRestartWireV2Version,
};
use super::epistemic_restart_wire_validation::{
    EpistemicRestartWireValidationError, EpistemicRestartWireValidator,
};
use super::epistemic_vector::{ClaimUncertaintyAssessment, EpistemicVector, UncertaintyDimension};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicRestartWireV2ValidationReport {
    pub captured_at_cycle: u64,
    pub revision_count: usize,
    pub eligible_revision_count: usize,
    pub rejected_revision_count: usize,
    /// True means the claimed V2 digest correctly binds the embedded V1 manifest
    /// digest to the independently canonical typed-schema digest. It does NOT mean
    /// the legacy embedded V1 manifest digest was independently re-derived here.
    pub claimed_digest_binding_valid: bool,
}

pub struct EpistemicRestartWireV2Validator;

impl EpistemicRestartWireV2Validator {
    pub fn validate(
        snapshot: &EpistemicRestartWireSnapshotV2,
    ) -> Result<EpistemicRestartWireV2ValidationReport, EpistemicRestartWireV2ValidationError> {
        if snapshot.version != EpistemicRestartWireV2Version::V2 {
            return Err(EpistemicRestartWireV2ValidationError::UnsupportedVersion);
        }
        if snapshot.encoding != EpistemicRestartWireV2Encoding::BaseV1PlusTypedSchemaV1 {
            return Err(EpistemicRestartWireV2ValidationError::UnsupportedEncoding);
        }

        EpistemicRestartWireValidator::validate(&snapshot.base)
            .map_err(EpistemicRestartWireV2ValidationError::BaseSemantic)?;
        BeliefRevisionSchemaWireValidator::validate(&snapshot.revision_schemas)
            .map_err(EpistemicRestartWireV2ValidationError::SchemaSemantic)?;

        if snapshot.base.captured_at_cycle != snapshot.revision_schemas.captured_at_cycle {
            return Err(EpistemicRestartWireV2ValidationError::CaptureCycleMismatch {
                base: snapshot.base.captured_at_cycle,
                schema: snapshot.revision_schemas.captured_at_cycle,
            });
        }
        if snapshot.base.revisions.len() != snapshot.revision_schemas.records.len() {
            return Err(EpistemicRestartWireV2ValidationError::RevisionCountMismatch {
                base: snapshot.base.revisions.len(),
                schema: snapshot.revision_schemas.records.len(),
            });
        }
        if snapshot.base.revision_next_receipt_id
            != snapshot.revision_schemas.linked_next_receipt_id
        {
            return Err(EpistemicRestartWireV2ValidationError::NextReceiptIdMismatch);
        }

        let ledger = rebuild_verification_ledger(&snapshot.base)?;
        let mut eligible = 0usize;
        for (base_receipt, schema_record) in snapshot
            .base
            .revisions
            .iter()
            .zip(&snapshot.revision_schemas.records)
        {
            if base_receipt.id != schema_record.receipt_id
                || base_receipt.claim_id != schema_record.claim_id
                || base_receipt.proposed_delta.to_bits() != schema_record.proposed_delta.to_bits()
                || base_receipt.evaluated_at_cycle != schema_record.evaluated_at_cycle
            {
                return Err(EpistemicRestartWireV2ValidationError::ReceiptIdentityMismatch(
                    base_receipt.id.0,
                ));
            }

            let policy = schema_record
                .policy_schema
                .build_policy()
                .map_err(|_| EpistemicRestartWireV2ValidationError::PolicyRebuildRejected(
                    base_receipt.id.0,
                ))?;
            let basis_ids = base_receipt
                .basis
                .iter()
                .map(|basis| basis.requested_id)
                .collect::<Vec<_>>();
            let proposal = EpistemicRevisionProposal::new(
                base_receipt.claim_id,
                base_receipt.proposed_delta,
                basis_ids,
                base_receipt.rationale.clone(),
            )
            .map_err(|_| EpistemicRestartWireV2ValidationError::ProposalRebuildRejected(
                base_receipt.id.0,
            ))?;
            let calibration = rebuild_calibration(base_receipt.id.0, base_receipt.calibration)?;
            let uncertainty = rebuild_uncertainty(base_receipt.id.0, &ledger, base_receipt)?;

            let decision = BeliefRevisionGate::evaluate(
                &ledger,
                &proposal,
                &policy,
                calibration,
                uncertainty.as_ref(),
            );
            let decision_snapshot = BeliefRevisionDecisionSnapshotV1::capture(&decision);
            if decision_snapshot != schema_record.decision_snapshot {
                return Err(EpistemicRestartWireV2ValidationError::RecomputedDecisionMismatch(
                    base_receipt.id.0,
                ));
            }
            if base_receipt.decision_eligible != decision_snapshot.eligible {
                return Err(EpistemicRestartWireV2ValidationError::BaseEligibilityMismatch(
                    base_receipt.id.0,
                ));
            }
            let root_count = u64::try_from(decision_snapshot.declared_provenance_root_count)
                .map_err(|_| EpistemicRestartWireV2ValidationError::LengthOverflow)?;
            if base_receipt.declared_provenance_root_count != root_count {
                return Err(EpistemicRestartWireV2ValidationError::BaseRootCountMismatch(
                    base_receipt.id.0,
                ));
            }
            if decision_snapshot.eligible {
                eligible += 1;
            }
        }

        let schema_digest = digest_schema_snapshot(&snapshot.revision_schemas)?;
        let expected_v2 = bind_v2_digest(snapshot.base.manifest.manifest_digest, schema_digest);
        if expected_v2 != snapshot.claimed_v2_digest {
            return Err(EpistemicRestartWireV2ValidationError::ClaimedV2DigestMismatch);
        }

        Ok(EpistemicRestartWireV2ValidationReport {
            captured_at_cycle: snapshot.base.captured_at_cycle,
            revision_count: snapshot.base.revisions.len(),
            eligible_revision_count: eligible,
            rejected_revision_count: snapshot.base.revisions.len() - eligible,
            claimed_digest_binding_valid: true,
        })
    }
}

fn rebuild_verification_ledger(
    base: &EpistemicRestartWireSnapshotV1,
) -> Result<EpistemicLedger, EpistemicRestartWireV2ValidationError> {
    let mut ledger = EpistemicLedger::new();
    for record in &base.provenance {
        let id = ledger
            .add_provenance(
                record.source_label.clone(),
                record.source_uri.clone(),
                record.content_hash.clone(),
                record.recorded_at_cycle,
                record.parent_ids.clone(),
            )
            .map_err(|_| EpistemicRestartWireV2ValidationError::LedgerRebuildRejected)?;
        if id != record.id {
            return Err(EpistemicRestartWireV2ValidationError::LedgerRebuildIdMismatch);
        }
    }
    for record in &base.claims {
        let id = ledger.add_claim(
            record.statement.clone(),
            record.kind,
            record.domain.clone(),
            record.scope.clone(),
            record.created_at_cycle,
        );
        if id != record.id {
            return Err(EpistemicRestartWireV2ValidationError::LedgerRebuildIdMismatch);
        }
    }
    for record in &base.evidence {
        let id = ledger
            .add_evidence(
                record.claim_id,
                record.kind,
                record.polarity,
                record.provenance_id,
                record.observed_at_cycle,
                record.context.clone(),
                record.method.clone(),
            )
            .map_err(|_| EpistemicRestartWireV2ValidationError::LedgerRebuildRejected)?;
        if id != record.id {
            return Err(EpistemicRestartWireV2ValidationError::LedgerRebuildIdMismatch);
        }
    }
    Ok(ledger)
}

fn rebuild_calibration(
    receipt_id: u64,
    calibration: Option<(u64, f64)>,
) -> Result<Option<CalibrationSnapshot>, EpistemicRestartWireV2ValidationError> {
    calibration
        .map(|(sample_count, ece)| {
            CalibrationSnapshot::new(sample_count, ece).map_err(|_| {
                EpistemicRestartWireV2ValidationError::CalibrationRebuildRejected(receipt_id)
            })
        })
        .transpose()
}

fn rebuild_uncertainty(
    receipt_id: u64,
    ledger: &EpistemicLedger,
    receipt: &super::epistemic_restart_wire::WireRevisionReceiptV1,
) -> Result<Option<ClaimUncertaintyAssessment>, EpistemicRestartWireV2ValidationError> {
    let Some(wire) = &receipt.uncertainty else {
        return Ok(None);
    };
    if wire.claim_id != receipt.claim_id {
        return Err(EpistemicRestartWireV2ValidationError::UncertaintyRebuildRejected(
            receipt_id,
        ));
    }
    let mut vector = EpistemicVector::new();
    for (dimension, value) in [
        (UncertaintyDimension::Epistemic, wire.epistemic),
        (UncertaintyDimension::Aleatoric, wire.aleatoric),
        (UncertaintyDimension::Ontological, wire.ontological),
        (
            UncertaintyDimension::DistributionShift,
            wire.distribution_shift,
        ),
    ] {
        if let Some(value) = value {
            vector
                .set(dimension, value)
                .map_err(|_| EpistemicRestartWireV2ValidationError::UncertaintyRebuildRejected(
                    receipt_id,
                ))?;
        }
    }
    ClaimUncertaintyAssessment::new(
        ledger,
        wire.claim_id,
        vector,
        wire.basis_evidence_ids.clone(),
        wire.assessed_at_cycle,
    )
    .map(Some)
    .map_err(|_| EpistemicRestartWireV2ValidationError::UncertaintyRebuildRejected(receipt_id))
}

fn digest_schema_snapshot(
    snapshot: &super::belief_revision_schema_wire::BeliefRevisionSchemaWireSnapshotV1,
) -> Result<[u8; 32], EpistemicRestartWireV2ValidationError> {
    let mut h = CanonicalHasher::new(b"symthaea-ekm-revision-schema-history-v1");
    h.u64(snapshot.captured_at_cycle);
    h.u64(snapshot.linked_revision_capture_cycle);
    h.u64(snapshot.linked_revision_count);
    h.u64(snapshot.linked_next_receipt_id.0);
    h.usize(snapshot.records.len())?;
    for record in &snapshot.records {
        h.u64(record.receipt_id.0);
        h.u64(record.claim_id.0);
        h.f32(record.proposed_delta);
        h.u64(record.evaluated_at_cycle);

        let policy = &record.policy_schema;
        h.f32(policy.max_abs_delta);
        h.usize(policy.min_declared_provenance_roots)?;
        h.bool(policy.require_calibration);
        h.u64(policy.min_calibration_samples);
        h.f64(policy.max_calibration_ece);
        h.bool(policy.require_uncertainty_assessment);
        h.bool(policy.require_current_uncertainty);
        h.bool(policy.block_strengthen_with_unresolved_contradictions);
        h.bool(policy.require_intervention_for_causal_strengthen);
        h.usize(policy.strengthen_uncertainty_caps.len())?;
        for (dimension, maximum) in &policy.strengthen_uncertainty_caps {
            h.u8(uncertainty_dimension_tag(*dimension));
            h.f64(*maximum);
        }
        digest_decision(&mut h, &record.decision_snapshot)?;
    }
    Ok(h.finish())
}

fn digest_decision(
    h: &mut CanonicalHasher,
    decision: &BeliefRevisionDecisionSnapshotV1,
) -> Result<(), EpistemicRestartWireV2ValidationError> {
    h.bool(decision.eligible);
    h.usize(decision.declared_provenance_root_count)?;
    h.usize(decision.failures.len())?;
    for failure in &decision.failures {
        match failure {
            BeliefRevisionFailureSnapshotV1::WeightRoutingDenied(failures) => {
                h.u8(1);
                h.usize(failures.len())?;
                for failure in failures {
                    match failure {
                        KnowledgeWeightRoutingFailureSnapshotV1::SourceCannotUpdateDimension {
                            source,
                            dimension,
                        } => {
                            h.u8(1);
                            h.u8(knowledge_weight_source_tag(*source));
                            h.u8(knowledge_weight_dimension_tag(*dimension));
                        }
                        KnowledgeWeightRoutingFailureSnapshotV1::EpistemicUpdateHasNoEvidenceBasis => {
                            h.u8(2);
                        }
                    }
                }
            }
            BeliefRevisionFailureSnapshotV1::UnknownClaim(id) => {
                h.u8(2);
                h.u64(id.0);
            }
            BeliefRevisionFailureSnapshotV1::UnknownEvidence(id) => {
                h.u8(3);
                h.u64(id.0);
            }
            BeliefRevisionFailureSnapshotV1::EvidenceForDifferentClaim {
                evidence_id,
                expected_claim,
                actual_claim,
            } => {
                h.u8(4);
                h.u64(evidence_id.0);
                h.u64(expected_claim.0);
                h.u64(actual_claim.0);
            }
            BeliefRevisionFailureSnapshotV1::PositiveDeltaLacksSupportingEvidence => h.u8(5),
            BeliefRevisionFailureSnapshotV1::PositiveDeltaIncludesContradictingEvidence => h.u8(6),
            BeliefRevisionFailureSnapshotV1::NegativeDeltaLacksContradictingEvidence => h.u8(7),
            BeliefRevisionFailureSnapshotV1::NegativeDeltaIncludesSupportingEvidence => h.u8(8),
            BeliefRevisionFailureSnapshotV1::DeclaredProvenanceRootsBelowMinimum {
                required,
                actual,
            } => {
                h.u8(9);
                h.usize(*required)?;
                h.usize(*actual)?;
            }
            BeliefRevisionFailureSnapshotV1::DeltaExceedsPolicy { maximum, actual } => {
                h.u8(10);
                h.f32(*maximum);
                h.f32(*actual);
            }
            BeliefRevisionFailureSnapshotV1::CalibrationMissing => h.u8(11),
            BeliefRevisionFailureSnapshotV1::CalibrationSamplesBelowMinimum { required, actual } => {
                h.u8(12);
                h.u64(*required);
                h.u64(*actual);
            }
            BeliefRevisionFailureSnapshotV1::CalibrationEceAboveMaximum { maximum, actual } => {
                h.u8(13);
                h.f64(*maximum);
                h.f64(*actual);
            }
            BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentMissing => h.u8(14),
            BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentForDifferentClaim {
                expected_claim,
                actual_claim,
            } => {
                h.u8(15);
                h.u64(expected_claim.0);
                h.u64(actual_claim.0);
            }
            BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentPredatesEvidence {
                assessment_cycle,
                latest_evidence_cycle,
            } => {
                h.u8(16);
                h.u64(*assessment_cycle);
                h.u64(*latest_evidence_cycle);
            }
            BeliefRevisionFailureSnapshotV1::RequiredUncertaintyDimensionUnassessed(dimension) => {
                h.u8(17);
                h.u8(uncertainty_dimension_tag(*dimension));
            }
            BeliefRevisionFailureSnapshotV1::UncertaintyAboveMaximum {
                dimension,
                maximum,
                actual,
            } => {
                h.u8(18);
                h.u8(uncertainty_dimension_tag(*dimension));
                h.f64(*maximum);
                h.f64(*actual);
            }
            BeliefRevisionFailureSnapshotV1::UnresolvedContradictionBlocksStrengthen { count } => {
                h.u8(19);
                h.usize(*count)?;
            }
            BeliefRevisionFailureSnapshotV1::CausalStrengthenLacksInterventionalBasis => h.u8(20),
        }
    }
    Ok(())
}

fn bind_v2_digest(base_manifest_digest: [u8; 32], schema_digest: [u8; 32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-capsule-v2");
    hasher.update(&base_manifest_digest);
    hasher.update(&schema_digest);
    *hasher.finalize().as_bytes()
}

struct CanonicalHasher(blake3::Hasher);

impl CanonicalHasher {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(domain.len() as u64).to_le_bytes());
        hasher.update(domain);
        Self(hasher)
    }

    fn u8(&mut self, value: u8) {
        self.0.update(&[value]);
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u64(&mut self, value: u64) {
        self.0.update(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) -> Result<(), EpistemicRestartWireV2ValidationError> {
        self.u64(
            u64::try_from(value)
                .map_err(|_| EpistemicRestartWireV2ValidationError::LengthOverflow)?,
        );
        Ok(())
    }

    fn f32(&mut self, value: f32) {
        self.0.update(&value.to_bits().to_le_bytes());
    }

    fn f64(&mut self, value: f64) {
        self.0.update(&value.to_bits().to_le_bytes());
    }

    fn finish(self) -> [u8; 32] {
        *self.0.finalize().as_bytes()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicRestartWireV2ValidationError {
    UnsupportedVersion,
    UnsupportedEncoding,
    BaseSemantic(EpistemicRestartWireValidationError),
    SchemaSemantic(BeliefRevisionSchemaWireValidationError),
    CaptureCycleMismatch { base: u64, schema: u64 },
    RevisionCountMismatch { base: usize, schema: usize },
    NextReceiptIdMismatch,
    ReceiptIdentityMismatch(u64),
    LedgerRebuildRejected,
    LedgerRebuildIdMismatch,
    PolicyRebuildRejected(u64),
    ProposalRebuildRejected(u64),
    CalibrationRebuildRejected(u64),
    UncertaintyRebuildRejected(u64),
    RecomputedDecisionMismatch(u64),
    BaseEligibilityMismatch(u64),
    BaseRootCountMismatch(u64),
    ClaimedV2DigestMismatch,
    LengthOverflow,
}

impl fmt::Display for EpistemicRestartWireV2ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epistemic restart v2 bundle semantics invalid: {self:?}")
    }
}

impl Error for EpistemicRestartWireV2ValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistory,
        BeliefRevisionHistoryCapsuleV1, BeliefRevisionPolicySchemaV1,
        BeliefRevisionSchemaHistoryCapsuleV1, BeliefRevisionSchemaHistoryV1,
        ClaimKind, EpistemicLedgerInventoryV1, EpistemicRestartCapsuleV1,
        EpistemicRestartCapsuleV2, EpistemicRestartWireV2, EpistemicSupportStore,
        EvidenceKind, EvidencePolarity,
    };

    fn snapshot() -> EpistemicRestartWireSnapshotV2 {
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
        let inventory = EpistemicLedgerInventoryV1::new(
            vec![claim],
            vec![evidence],
            vec![provenance],
        )
        .unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.1, vec![evidence], "measurement")
            .unwrap();
        let schema = BeliefRevisionPolicySchemaV1::new(0.2, 1, false, 0, 1.0).unwrap();
        let mut receipts = BeliefRevisionHistory::new();
        let mut schema_history = BeliefRevisionSchemaHistoryV1::new();
        schema_history
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
        let schemas = BeliefRevisionSchemaHistoryCapsuleV1::capture(
            &schema_history,
            &receipts,
            &revisions,
            4,
        )
        .unwrap();
        let base = EpistemicRestartCapsuleV1::capture(&ledger, &inventory, &mutations, &revisions, 4)
            .unwrap();
        let v2 = EpistemicRestartCapsuleV2::capture(&base, &schemas).unwrap();
        let bytes = EpistemicRestartWireV2::encode(&v2).unwrap();
        EpistemicRestartWireV2::decode(&bytes).unwrap()
    }

    #[test]
    fn valid_v2_bundle_recomputes_gate_and_digest_binding() {
        let snapshot = snapshot();
        let report = EpistemicRestartWireV2Validator::validate(&snapshot).unwrap();
        assert_eq!(report.revision_count, 1);
        assert_eq!(report.eligible_revision_count, 1);
        assert!(report.claimed_digest_binding_valid);
    }

    #[test]
    fn typed_policy_tamper_changes_recomputed_decision_or_digest() {
        let mut snapshot = snapshot();
        snapshot.revision_schemas.records[0].policy_schema.max_abs_delta = 0.05;
        assert!(EpistemicRestartWireV2Validator::validate(&snapshot).is_err());
    }

    #[test]
    fn claimed_v2_digest_is_not_trusted() {
        let mut snapshot = snapshot();
        snapshot.claimed_v2_digest[0] ^= 1;
        assert_eq!(
            EpistemicRestartWireV2Validator::validate(&snapshot).unwrap_err(),
            EpistemicRestartWireV2ValidationError::ClaimedV2DigestMismatch
        );
    }
}
