// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed restart-v2 capsule binding EKM-033 state to EKM-038 policy semantics.
//!
//! V1 remains unchanged and preserves the original restart contract. V2 wraps a
//! verified V1 capsule plus complete schema-bound revision history, and adds a
//! domain-separated digest over explicit policy and decision fields only.
//!
//! No activation or writable hydration is introduced here.

use super::belief_revision_gate::BeliefRevisionFailure;
use super::belief_revision_schema_persistence::BeliefRevisionSchemaHistoryCapsuleV1;
use super::belief_revision_snapshot::{
    knowledge_weight_dimension_tag, knowledge_weight_source_tag, uncertainty_dimension_tag,
    BeliefRevisionDecisionSnapshotV1, BeliefRevisionFailureSnapshotV1,
    KnowledgeWeightRoutingFailureSnapshotV1,
};
use super::epistemic_restart_capsule::{
    EpistemicRestartCapsuleError, EpistemicRestartCapsuleV1, QuarantinedEpistemicRestartV1,
};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRestartCapsuleV2Version {
    V2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpistemicRestartV2Digest([u8; 32]);

impl EpistemicRestartV2Digest {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicRestartCapsuleV2 {
    version: EpistemicRestartCapsuleV2Version,
    base: EpistemicRestartCapsuleV1,
    revision_schemas: BeliefRevisionSchemaHistoryCapsuleV1,
    schema_digest: EpistemicRestartV2Digest,
    capsule_digest: EpistemicRestartV2Digest,
}

impl EpistemicRestartCapsuleV2 {
    pub fn capture(
        base: &EpistemicRestartCapsuleV1,
        revision_schemas: &BeliefRevisionSchemaHistoryCapsuleV1,
    ) -> Result<Self, EpistemicRestartCapsuleV2Error> {
        validate_links(base, revision_schemas)?;
        let schema_digest = digest_schema_history(revision_schemas)?;
        let capsule_digest = digest_capsule(base, schema_digest);
        Ok(Self {
            version: EpistemicRestartCapsuleV2Version::V2,
            base: base.clone(),
            revision_schemas: revision_schemas.clone(),
            schema_digest,
            capsule_digest,
        })
    }

    pub fn version(&self) -> EpistemicRestartCapsuleV2Version {
        self.version
    }

    pub fn captured_at_cycle(&self) -> u64 {
        self.base.captured_at_cycle()
    }

    pub fn base_v1(&self) -> &EpistemicRestartCapsuleV1 {
        &self.base
    }

    pub fn revision_schema_capsule(&self) -> &BeliefRevisionSchemaHistoryCapsuleV1 {
        &self.revision_schemas
    }

    pub fn schema_digest(&self) -> EpistemicRestartV2Digest {
        self.schema_digest
    }

    pub fn capsule_digest(&self) -> EpistemicRestartV2Digest {
        self.capsule_digest
    }

    pub fn verify(&self) -> Result<(), EpistemicRestartCapsuleV2Error> {
        validate_links(&self.base, &self.revision_schemas)?;
        let schema_digest = digest_schema_history(&self.revision_schemas)?;
        if schema_digest != self.schema_digest {
            return Err(EpistemicRestartCapsuleV2Error::SchemaDigestMismatch);
        }
        let capsule_digest = digest_capsule(&self.base, schema_digest);
        if capsule_digest != self.capsule_digest {
            return Err(EpistemicRestartCapsuleV2Error::CapsuleDigestMismatch);
        }
        Ok(())
    }

    pub fn quarantine(
        &self,
    ) -> Result<QuarantinedEpistemicRestartV2, EpistemicRestartCapsuleV2Error> {
        self.verify()?;
        let base = self.base.quarantine().map_err(EpistemicRestartCapsuleV2Error::Base)?;
        validate_quarantine_links(&base, &self.revision_schemas)?;
        Ok(QuarantinedEpistemicRestartV2 {
            base,
            revision_schemas: self.revision_schemas.clone(),
            schema_digest: self.schema_digest,
            capsule_digest: self.capsule_digest,
        })
    }
}

#[derive(Debug, Clone)]
pub struct QuarantinedEpistemicRestartV2 {
    base: QuarantinedEpistemicRestartV1,
    revision_schemas: BeliefRevisionSchemaHistoryCapsuleV1,
    schema_digest: EpistemicRestartV2Digest,
    capsule_digest: EpistemicRestartV2Digest,
}

impl QuarantinedEpistemicRestartV2 {
    pub fn base_v1(&self) -> &QuarantinedEpistemicRestartV1 {
        &self.base
    }

    pub fn revision_schema_capsule(&self) -> &BeliefRevisionSchemaHistoryCapsuleV1 {
        &self.revision_schemas
    }

    pub fn schema_digest(&self) -> EpistemicRestartV2Digest {
        self.schema_digest
    }

    pub fn capsule_digest(&self) -> EpistemicRestartV2Digest {
        self.capsule_digest
    }

    pub fn verify(&self) -> Result<(), EpistemicRestartCapsuleV2Error> {
        self.base.verify().map_err(EpistemicRestartCapsuleV2Error::Base)?;
        validate_quarantine_links(&self.base, &self.revision_schemas)?;
        let schema_digest = digest_schema_history(&self.revision_schemas)?;
        if schema_digest != self.schema_digest {
            return Err(EpistemicRestartCapsuleV2Error::SchemaDigestMismatch);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-ekm-restart-capsule-v2");
        hasher.update(&self.base.manifest_digest().as_bytes());
        hasher.update(&schema_digest.0);
        let actual = EpistemicRestartV2Digest(*hasher.finalize().as_bytes());
        if actual != self.capsule_digest {
            return Err(EpistemicRestartCapsuleV2Error::CapsuleDigestMismatch);
        }
        Ok(())
    }
}

fn validate_links(
    base: &EpistemicRestartCapsuleV1,
    schemas: &BeliefRevisionSchemaHistoryCapsuleV1,
) -> Result<(), EpistemicRestartCapsuleV2Error> {
    if base.captured_at_cycle() != schemas.captured_at_cycle()
        || base.revision_capsule().captured_at_cycle() != schemas.linked_revision_capture_cycle()
    {
        return Err(EpistemicRestartCapsuleV2Error::CaptureEpochMismatch);
    }
    if base.revision_capsule().receipts().len() != schemas.linked_revision_count()
        || base.revision_capsule().receipts().len() != schemas.records().len()
    {
        return Err(EpistemicRestartCapsuleV2Error::RevisionCountMismatch);
    }
    if base.revision_capsule().next_receipt_id() != schemas.linked_next_receipt_id() {
        return Err(EpistemicRestartCapsuleV2Error::NextReceiptIdMismatch);
    }

    for (receipt, schema) in base
        .revision_capsule()
        .receipts()
        .iter()
        .zip(schemas.records())
    {
        if receipt.id() != schema.receipt_id
            || receipt.claim_id() != schema.claim_id
            || receipt.proposed_delta().to_bits() != schema.proposed_delta.to_bits()
            || receipt.evaluated_at_cycle() != schema.evaluated_at_cycle
        {
            return Err(EpistemicRestartCapsuleV2Error::RevisionSchemaMismatch(
                receipt.id().0,
            ));
        }
        let rebuilt = schema
            .policy_schema
            .build_policy()
            .map_err(|_| EpistemicRestartCapsuleV2Error::PolicyRebuildRejected(receipt.id().0))?;
        if receipt.policy() != &rebuilt {
            return Err(EpistemicRestartCapsuleV2Error::PolicyMismatch(receipt.id().0));
        }
        let decision = BeliefRevisionDecisionSnapshotV1::capture(receipt.decision());
        if decision != schema.decision_snapshot {
            return Err(EpistemicRestartCapsuleV2Error::DecisionMismatch(receipt.id().0));
        }
    }
    Ok(())
}

fn validate_quarantine_links(
    base: &QuarantinedEpistemicRestartV1,
    schemas: &BeliefRevisionSchemaHistoryCapsuleV1,
) -> Result<(), EpistemicRestartCapsuleV2Error> {
    if base.revision_receipts().len() != schemas.records().len() {
        return Err(EpistemicRestartCapsuleV2Error::RevisionCountMismatch);
    }
    for (receipt, schema) in base.revision_receipts().iter().zip(schemas.records()) {
        if receipt.id() != schema.receipt_id
            || receipt.claim_id() != schema.claim_id
            || receipt.proposed_delta().to_bits() != schema.proposed_delta.to_bits()
            || receipt.evaluated_at_cycle() != schema.evaluated_at_cycle
        {
            return Err(EpistemicRestartCapsuleV2Error::RevisionSchemaMismatch(
                receipt.id().0,
            ));
        }
        let rebuilt = schema
            .policy_schema
            .build_policy()
            .map_err(|_| EpistemicRestartCapsuleV2Error::PolicyRebuildRejected(receipt.id().0))?;
        if receipt.policy() != &rebuilt {
            return Err(EpistemicRestartCapsuleV2Error::PolicyMismatch(receipt.id().0));
        }
        if BeliefRevisionDecisionSnapshotV1::capture(receipt.decision()) != schema.decision_snapshot {
            return Err(EpistemicRestartCapsuleV2Error::DecisionMismatch(receipt.id().0));
        }
    }
    Ok(())
}

fn digest_capsule(
    base: &EpistemicRestartCapsuleV1,
    schema_digest: EpistemicRestartV2Digest,
) -> EpistemicRestartV2Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-capsule-v2");
    hasher.update(&base.manifest_digest().as_bytes());
    hasher.update(&schema_digest.0);
    EpistemicRestartV2Digest(*hasher.finalize().as_bytes())
}

fn digest_schema_history(
    schemas: &BeliefRevisionSchemaHistoryCapsuleV1,
) -> Result<EpistemicRestartV2Digest, EpistemicRestartCapsuleV2Error> {
    let mut h = CanonicalHasher::new(b"symthaea-ekm-revision-schema-history-v1");
    h.u64(schemas.captured_at_cycle());
    h.u64(schemas.linked_revision_capture_cycle());
    h.usize(schemas.linked_revision_count())?;
    h.u64(schemas.linked_next_receipt_id().0);
    h.usize(schemas.records().len())?;

    for record in schemas.records() {
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
    Ok(EpistemicRestartV2Digest(h.finish()))
}

fn digest_decision(
    h: &mut CanonicalHasher,
    decision: &BeliefRevisionDecisionSnapshotV1,
) -> Result<(), EpistemicRestartCapsuleV2Error> {
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

struct CanonicalHasher(blake3::Hasher);

impl CanonicalHasher {
    fn new(domain: &[u8]) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(&(domain.len() as u64).to_le_bytes());
        h.update(domain);
        Self(h)
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

    fn usize(&mut self, value: usize) -> Result<(), EpistemicRestartCapsuleV2Error> {
        self.u64(u64::try_from(value).map_err(|_| EpistemicRestartCapsuleV2Error::LengthOverflow)?);
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
pub enum EpistemicRestartCapsuleV2Error {
    Base(EpistemicRestartCapsuleError),
    CaptureEpochMismatch,
    RevisionCountMismatch,
    NextReceiptIdMismatch,
    RevisionSchemaMismatch(u64),
    PolicyRebuildRejected(u64),
    PolicyMismatch(u64),
    DecisionMismatch(u64),
    SchemaDigestMismatch,
    CapsuleDigestMismatch,
    LengthOverflow,
}

impl fmt::Display for EpistemicRestartCapsuleV2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epistemic restart capsule v2 invalid: {self:?}")
    }
}

impl Error for EpistemicRestartCapsuleV2Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistory, BeliefRevisionHistoryCapsuleV1,
        BeliefRevisionPolicySchemaV1, BeliefRevisionSchemaHistoryCapsuleV1,
        BeliefRevisionSchemaHistoryV1, ClaimKind, EpistemicLedger, EpistemicLedgerInventoryV1,
        EpistemicRevisionProposal, EpistemicSupportStore, EvidenceKind, EvidencePolarity,
    };

    #[test]
    fn typed_v2_capsule_round_trips_through_quarantine() {
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
        let revision_capsule =
            BeliefRevisionHistoryCapsuleV1::capture(&receipts, &mutations, 4).unwrap();
        let schema_capsule = BeliefRevisionSchemaHistoryCapsuleV1::capture(
            &schema_history,
            &receipts,
            &revision_capsule,
            4,
        )
        .unwrap();
        let base = EpistemicRestartCapsuleV1::capture(
            &ledger,
            &inventory,
            &mutations,
            &revision_capsule,
            4,
        )
        .unwrap();
        let v2 = EpistemicRestartCapsuleV2::capture(&base, &schema_capsule).unwrap();
        v2.verify().unwrap();
        let quarantine = v2.quarantine().unwrap();
        quarantine.verify().unwrap();
        assert_eq!(quarantine.revision_schema_capsule().records().len(), 1);
        assert_eq!(v2.capsule_digest().to_hex().len(), 64);
    }

    #[test]
    fn v2_digest_is_deterministic_for_same_components() {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("lab", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("X exists", ClaimKind::Descriptive, None, None, 1);
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Observation,
                EvidencePolarity::Supports,
                provenance,
                2,
                None,
                None,
            )
            .unwrap();
        let inventory = EpistemicLedgerInventoryV1::new(vec![claim], vec![evidence], vec![provenance])
            .unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.05, vec![evidence], "observation")
            .unwrap();
        let schema = BeliefRevisionPolicySchemaV1::new(0.2, 1, false, 0, 1.0).unwrap();
        let mut receipts = BeliefRevisionHistory::new();
        let mut schema_history = BeliefRevisionSchemaHistoryV1::new();
        schema_history
            .evaluate_and_record(&mut receipts, &ledger, &proposal, &schema, None, None, 3)
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
        let left = EpistemicRestartCapsuleV2::capture(&base, &schemas).unwrap();
        let right = EpistemicRestartCapsuleV2::capture(&base, &schemas).unwrap();
        assert_eq!(left.schema_digest(), right.schema_digest());
        assert_eq!(left.capsule_digest(), right.capsule_digest());
    }
}
