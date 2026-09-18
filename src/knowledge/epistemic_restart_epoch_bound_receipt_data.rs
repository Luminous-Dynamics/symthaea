// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable data representation for the future epoch-bound revision receipt V2.
//!
//! EKM-075 defines the semantic migration contract. This module defines the exact
//! typed data shape and canonical digest for that future receipt while deliberately
//! providing no production constructor. Historical V1 receipts therefore cannot be
//! upgraded into operational V2 authority through this module.

use crate::knowledge::belief_revision_receipt::BeliefRevisionReceiptId;
use crate::knowledge::belief_revision_snapshot::{
    knowledge_weight_dimension_tag, knowledge_weight_source_tag, uncertainty_dimension_tag,
    BeliefRevisionDecisionSnapshotV1, BeliefRevisionFailureSnapshotV1,
    BeliefRevisionPolicySchemaV1, BeliefRevisionSnapshotVersion,
    KnowledgeWeightRoutingFailureSnapshotV1,
};
use crate::knowledge::claim_evidence::{ClaimId, EvidenceId, EvidenceKind, EvidencePolarity};
use crate::knowledge::epistemic_restart_continuity::epoch_bound_receipt_schema::{
    EpochBoundRevisionReceiptSchemaContractV2, EpochBoundRevisionReceiptSchemaError,
};
use crate::knowledge::epistemic_restart_wire::{
    WireRevisionBasisV1, WireUncertaintyAssessmentV1,
};
use std::collections::HashSet;
use std::error::Error;
use std::fmt;

const MAX_RECEIPT_BASIS_RECORDS: usize = 1_000_000;
const MAX_DUPLICATE_BASIS_IDS: usize = 1_000_000;
const MAX_DECISION_FAILURES: usize = 1_000_000;
const MAX_ROUTING_FAILURES: usize = 1_000_000;
const MAX_RATIONALE_BYTES: usize = 16 * 1024 * 1024;
const MAX_OPTIONAL_TEXT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochBoundRevisionReceiptDataVersion {
    V2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpochBoundRevisionReceiptDigestV2([u8; 32]);

impl EpochBoundRevisionReceiptDigestV2 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
        }
        out
    }
}

/// Exact immutable semantic data for a future operational revision receipt V2.
///
/// There is intentionally no production constructor in EKM-076. A later authority
/// tranche must construct this only from a fresh evaluation under the currently
/// committed authority epoch. Archival V1 state is not a construction source.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochBoundRevisionReceiptDataV2 {
    version: EpochBoundRevisionReceiptDataVersion,
    receipt_id: BeliefRevisionReceiptId,
    claim_id: ClaimId,
    proposed_delta: f32,
    rationale: String,
    basis: Vec<WireRevisionBasisV1>,
    duplicate_basis_evidence_ids: Vec<EvidenceId>,
    policy_schema: BeliefRevisionPolicySchemaV1,
    calibration: Option<(u64, f64)>,
    uncertainty: Option<WireUncertaintyAssessmentV1>,
    decision_snapshot: BeliefRevisionDecisionSnapshotV1,
    evaluated_at_cycle: u64,
    authority_epoch_digest: [u8; 32],
    authority_epoch_sequence: u64,
    receipt_digest: EpochBoundRevisionReceiptDigestV2,
}

impl EpochBoundRevisionReceiptDataV2 {
    pub fn version(&self) -> EpochBoundRevisionReceiptDataVersion {
        self.version
    }
    pub fn receipt_id(&self) -> BeliefRevisionReceiptId {
        self.receipt_id
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
    pub fn basis(&self) -> &[WireRevisionBasisV1] {
        &self.basis
    }
    pub fn duplicate_basis_evidence_ids(&self) -> &[EvidenceId] {
        &self.duplicate_basis_evidence_ids
    }
    pub fn policy_schema(&self) -> &BeliefRevisionPolicySchemaV1 {
        &self.policy_schema
    }
    pub fn calibration(&self) -> Option<(u64, f64)> {
        self.calibration
    }
    pub fn uncertainty(&self) -> Option<&WireUncertaintyAssessmentV1> {
        self.uncertainty.as_ref()
    }
    pub fn decision_snapshot(&self) -> &BeliefRevisionDecisionSnapshotV1 {
        &self.decision_snapshot
    }
    pub fn evaluated_at_cycle(&self) -> u64 {
        self.evaluated_at_cycle
    }
    pub fn authority_epoch_digest(&self) -> [u8; 32] {
        self.authority_epoch_digest
    }
    pub fn authority_epoch_sequence(&self) -> u64 {
        self.authority_epoch_sequence
    }
    pub fn receipt_digest(&self) -> EpochBoundRevisionReceiptDigestV2 {
        self.receipt_digest
    }

    /// Possessing this passive DTO is never mutation authority.
    pub fn mutation_authority(&self) -> bool {
        false
    }

    /// EKM-076 does not connect the DTO to the current mutation facade.
    pub fn accepted_by_current_mutation_facade(&self) -> bool {
        false
    }

    pub fn verify(&self) -> Result<(), EpochBoundRevisionReceiptDataError> {
        validate_data(self)?;
        if digest_data(self)? != self.receipt_digest {
            return Err(EpochBoundRevisionReceiptDataError::ReceiptDigestMismatch);
        }
        Ok(())
    }
}

fn validate_data(
    data: &EpochBoundRevisionReceiptDataV2,
) -> Result<(), EpochBoundRevisionReceiptDataError> {
    EpochBoundRevisionReceiptSchemaContractV2::canonical()
        .verify()
        .map_err(EpochBoundRevisionReceiptDataError::SchemaContract)?;

    if data.version != EpochBoundRevisionReceiptDataVersion::V2 {
        return Err(EpochBoundRevisionReceiptDataError::UnsupportedVersion);
    }
    if data.receipt_id.0 == 0 {
        return Err(EpochBoundRevisionReceiptDataError::InvalidReceiptId);
    }
    if data.authority_epoch_sequence == 0 {
        return Err(EpochBoundRevisionReceiptDataError::InvalidAuthorityEpochSequence);
    }
    if !data.proposed_delta.is_finite() || !(-1.0..=1.0).contains(&data.proposed_delta) {
        return Err(EpochBoundRevisionReceiptDataError::InvalidProposedDelta(
            data.proposed_delta,
        ));
    }
    if data.rationale.len() > MAX_RATIONALE_BYTES {
        return Err(EpochBoundRevisionReceiptDataError::RationaleTooLarge {
            actual: data.rationale.len(),
            maximum: MAX_RATIONALE_BYTES,
        });
    }
    if data.basis.len() > MAX_RECEIPT_BASIS_RECORDS {
        return Err(EpochBoundRevisionReceiptDataError::TooManyBasisRecords {
            actual: data.basis.len(),
            maximum: MAX_RECEIPT_BASIS_RECORDS,
        });
    }
    if data.duplicate_basis_evidence_ids.len() > MAX_DUPLICATE_BASIS_IDS {
        return Err(EpochBoundRevisionReceiptDataError::TooManyDuplicateBasisIds {
            actual: data.duplicate_basis_evidence_ids.len(),
            maximum: MAX_DUPLICATE_BASIS_IDS,
        });
    }

    let mut basis_ids = HashSet::new();
    for basis in &data.basis {
        if !basis_ids.insert(basis.requested_id) {
            return Err(EpochBoundRevisionReceiptDataError::DuplicateUniqueBasisId(
                basis.requested_id,
            ));
        }
        if let Some(snapshot) = &basis.snapshot {
            if snapshot.evidence_id != basis.requested_id {
                return Err(EpochBoundRevisionReceiptDataError::BasisSnapshotIdMismatch {
                    requested: basis.requested_id,
                    snapshot: snapshot.evidence_id,
                });
            }
            if snapshot.observed_at_cycle > data.evaluated_at_cycle {
                return Err(EpochBoundRevisionReceiptDataError::EvaluationPredatesBasisEvidence {
                    evidence_id: snapshot.evidence_id,
                    evaluated_at_cycle: data.evaluated_at_cycle,
                    observed_at_cycle: snapshot.observed_at_cycle,
                });
            }
            validate_optional_text(snapshot.context.as_deref())?;
            validate_optional_text(snapshot.method.as_deref())?;
        }
    }

    let mut duplicate_ids = HashSet::new();
    for id in &data.duplicate_basis_evidence_ids {
        if !duplicate_ids.insert(*id) {
            return Err(EpochBoundRevisionReceiptDataError::DuplicateDuplicateBasisId(*id));
        }
        if !basis_ids.contains(id) {
            return Err(EpochBoundRevisionReceiptDataError::DuplicateBasisIdMissingFromBasis(*id));
        }
    }

    data.policy_schema
        .build_policy()
        .map_err(|_| EpochBoundRevisionReceiptDataError::InvalidPolicySchema)?;
    let mut previous_dimension_tag = 0u8;
    for (dimension, maximum) in &data.policy_schema.strengthen_uncertainty_caps {
        let tag = uncertainty_dimension_tag(*dimension);
        if tag <= previous_dimension_tag {
            return Err(EpochBoundRevisionReceiptDataError::NonCanonicalUncertaintyCaps);
        }
        previous_dimension_tag = tag;
        validate_unit_f64(*maximum)?;
    }

    if let Some((_, ece)) = data.calibration {
        validate_unit_f64(ece)?;
    }
    if let Some(uncertainty) = &data.uncertainty {
        for value in [
            uncertainty.epistemic,
            uncertainty.aleatoric,
            uncertainty.ontological,
            uncertainty.distribution_shift,
        ]
        .into_iter()
        .flatten()
        {
            validate_unit_f64(value)?;
        }
        if uncertainty.assessed_at_cycle > data.evaluated_at_cycle {
            return Err(
                EpochBoundRevisionReceiptDataError::EvaluationPredatesUncertaintyAssessment {
                    evaluated_at_cycle: data.evaluated_at_cycle,
                    assessed_at_cycle: uncertainty.assessed_at_cycle,
                },
            );
        }
        if uncertainty.basis_evidence_ids.len() > MAX_RECEIPT_BASIS_RECORDS {
            return Err(EpochBoundRevisionReceiptDataError::TooManyUncertaintyBasisIds {
                actual: uncertainty.basis_evidence_ids.len(),
                maximum: MAX_RECEIPT_BASIS_RECORDS,
            });
        }
    }

    if data.decision_snapshot.version != BeliefRevisionSnapshotVersion::V1 {
        return Err(EpochBoundRevisionReceiptDataError::UnsupportedDecisionSnapshotVersion);
    }
    if data.decision_snapshot.failures.len() > MAX_DECISION_FAILURES {
        return Err(EpochBoundRevisionReceiptDataError::TooManyDecisionFailures {
            actual: data.decision_snapshot.failures.len(),
            maximum: MAX_DECISION_FAILURES,
        });
    }
    if data.decision_snapshot.eligible != data.decision_snapshot.failures.is_empty() {
        return Err(EpochBoundRevisionReceiptDataError::DecisionEligibilityMismatch);
    }
    for failure in &data.decision_snapshot.failures {
        validate_failure(failure)?;
    }

    Ok(())
}

fn validate_optional_text(value: Option<&str>) -> Result<(), EpochBoundRevisionReceiptDataError> {
    if let Some(value) = value {
        if value.len() > MAX_OPTIONAL_TEXT_BYTES {
            return Err(EpochBoundRevisionReceiptDataError::OptionalTextTooLarge {
                actual: value.len(),
                maximum: MAX_OPTIONAL_TEXT_BYTES,
            });
        }
    }
    Ok(())
}

fn validate_failure(
    failure: &BeliefRevisionFailureSnapshotV1,
) -> Result<(), EpochBoundRevisionReceiptDataError> {
    match failure {
        BeliefRevisionFailureSnapshotV1::WeightRoutingDenied(failures) => {
            if failures.len() > MAX_ROUTING_FAILURES {
                return Err(EpochBoundRevisionReceiptDataError::TooManyRoutingFailures {
                    actual: failures.len(),
                    maximum: MAX_ROUTING_FAILURES,
                });
            }
        }
        BeliefRevisionFailureSnapshotV1::DeltaExceedsPolicy { maximum, actual } => {
            validate_unit_f32(*maximum)?;
            if !actual.is_finite() || !(0.0..=1.0).contains(actual) {
                return Err(EpochBoundRevisionReceiptDataError::InvalidFailureFloat);
            }
        }
        BeliefRevisionFailureSnapshotV1::CalibrationEceAboveMaximum { maximum, actual }
        | BeliefRevisionFailureSnapshotV1::UncertaintyAboveMaximum {
            maximum, actual, ..
        } => {
            validate_unit_f64(*maximum)?;
            validate_unit_f64(*actual)?;
        }
        _ => {}
    }
    Ok(())
}

fn validate_unit_f32(value: f32) -> Result<(), EpochBoundRevisionReceiptDataError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(EpochBoundRevisionReceiptDataError::InvalidFailureFloat);
    }
    Ok(())
}

fn validate_unit_f64(value: f64) -> Result<(), EpochBoundRevisionReceiptDataError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(EpochBoundRevisionReceiptDataError::InvalidUnitFloat(value));
    }
    Ok(())
}

fn digest_data(
    data: &EpochBoundRevisionReceiptDataV2,
) -> Result<EpochBoundRevisionReceiptDigestV2, EpochBoundRevisionReceiptDataError> {
    validate_data(data)?;
    let schema = EpochBoundRevisionReceiptSchemaContractV2::canonical();
    let mut writer =
        CanonicalDigestWriter::new(b"symthaea-ekm-epoch-bound-revision-receipt-data-v2");
    writer.u8(2);
    writer.bytes32(schema.schema_digest().as_bytes());
    writer.u64(data.receipt_id.0);
    writer.u64(data.claim_id.0);
    writer.f32(data.proposed_delta);
    writer.string(&data.rationale)?;
    writer.count(data.basis.len())?;
    for basis in &data.basis {
        encode_basis(&mut writer, basis)?;
    }
    writer.count(data.duplicate_basis_evidence_ids.len())?;
    for id in &data.duplicate_basis_evidence_ids {
        writer.u64(id.0);
    }
    encode_policy(&mut writer, &data.policy_schema)?;
    match data.calibration {
        Some((sample_count, ece)) => {
            writer.bool(true);
            writer.u64(sample_count);
            writer.f64(ece);
        }
        None => writer.bool(false),
    }
    match &data.uncertainty {
        Some(uncertainty) => {
            writer.bool(true);
            encode_uncertainty(&mut writer, uncertainty)?;
        }
        None => writer.bool(false),
    }
    encode_decision(&mut writer, &data.decision_snapshot)?;
    writer.u64(data.evaluated_at_cycle);
    writer.bytes32(data.authority_epoch_digest);
    writer.u64(data.authority_epoch_sequence);
    Ok(EpochBoundRevisionReceiptDigestV2(writer.finish()))
}

fn encode_basis(
    writer: &mut CanonicalDigestWriter,
    basis: &WireRevisionBasisV1,
) -> Result<(), EpochBoundRevisionReceiptDataError> {
    writer.u64(basis.requested_id.0);
    match &basis.snapshot {
        Some(snapshot) => {
            writer.bool(true);
            writer.u64(snapshot.evidence_id.0);
            writer.u64(snapshot.claim_id.0);
            writer.u8(evidence_kind_tag(snapshot.kind));
            writer.u8(evidence_polarity_tag(snapshot.polarity));
            writer.u64(snapshot.provenance_id.0);
            writer.u64(snapshot.observed_at_cycle);
            writer.optional_string(snapshot.context.as_deref())?;
            writer.optional_string(snapshot.method.as_deref())?;
        }
        None => writer.bool(false),
    }
    Ok(())
}

fn encode_policy(
    writer: &mut CanonicalDigestWriter,
    policy: &BeliefRevisionPolicySchemaV1,
) -> Result<(), EpochBoundRevisionReceiptDataError> {
    writer.f32(policy.max_abs_delta);
    writer.usize(policy.min_declared_provenance_roots)?;
    writer.bool(policy.require_calibration);
    writer.u64(policy.min_calibration_samples);
    writer.f64(policy.max_calibration_ece);
    writer.bool(policy.require_uncertainty_assessment);
    writer.bool(policy.require_current_uncertainty);
    writer.bool(policy.block_strengthen_with_unresolved_contradictions);
    writer.bool(policy.require_intervention_for_causal_strengthen);
    writer.count(policy.strengthen_uncertainty_caps.len())?;
    for (dimension, maximum) in &policy.strengthen_uncertainty_caps {
        writer.u8(uncertainty_dimension_tag(*dimension));
        writer.f64(*maximum);
    }
    Ok(())
}

fn encode_uncertainty(
    writer: &mut CanonicalDigestWriter,
    uncertainty: &WireUncertaintyAssessmentV1,
) -> Result<(), EpochBoundRevisionReceiptDataError> {
    writer.u64(uncertainty.claim_id.0);
    writer.optional_f64(uncertainty.epistemic);
    writer.optional_f64(uncertainty.aleatoric);
    writer.optional_f64(uncertainty.ontological);
    writer.optional_f64(uncertainty.distribution_shift);
    writer.count(uncertainty.basis_evidence_ids.len())?;
    for id in &uncertainty.basis_evidence_ids {
        writer.u64(id.0);
    }
    writer.u64(uncertainty.assessed_at_cycle);
    Ok(())
}

fn encode_decision(
    writer: &mut CanonicalDigestWriter,
    decision: &BeliefRevisionDecisionSnapshotV1,
) -> Result<(), EpochBoundRevisionReceiptDataError> {
    writer.bool(decision.eligible);
    writer.usize(decision.declared_provenance_root_count)?;
    writer.count(decision.failures.len())?;
    for failure in &decision.failures {
        encode_failure(writer, failure)?;
    }
    Ok(())
}

fn encode_failure(
    writer: &mut CanonicalDigestWriter,
    failure: &BeliefRevisionFailureSnapshotV1,
) -> Result<(), EpochBoundRevisionReceiptDataError> {
    match failure {
        BeliefRevisionFailureSnapshotV1::WeightRoutingDenied(failures) => {
            writer.u8(1);
            writer.count(failures.len())?;
            for failure in failures {
                match failure {
                    KnowledgeWeightRoutingFailureSnapshotV1::SourceCannotUpdateDimension {
                        source,
                        dimension,
                    } => {
                        writer.u8(1);
                        writer.u8(knowledge_weight_source_tag(*source));
                        writer.u8(knowledge_weight_dimension_tag(*dimension));
                    }
                    KnowledgeWeightRoutingFailureSnapshotV1::EpistemicUpdateHasNoEvidenceBasis => {
                        writer.u8(2);
                    }
                }
            }
        }
        BeliefRevisionFailureSnapshotV1::UnknownClaim(id) => {
            writer.u8(2);
            writer.u64(id.0);
        }
        BeliefRevisionFailureSnapshotV1::UnknownEvidence(id) => {
            writer.u8(3);
            writer.u64(id.0);
        }
        BeliefRevisionFailureSnapshotV1::EvidenceForDifferentClaim {
            evidence_id,
            expected_claim,
            actual_claim,
        } => {
            writer.u8(4);
            writer.u64(evidence_id.0);
            writer.u64(expected_claim.0);
            writer.u64(actual_claim.0);
        }
        BeliefRevisionFailureSnapshotV1::PositiveDeltaLacksSupportingEvidence => writer.u8(5),
        BeliefRevisionFailureSnapshotV1::PositiveDeltaIncludesContradictingEvidence => writer.u8(6),
        BeliefRevisionFailureSnapshotV1::NegativeDeltaLacksContradictingEvidence => writer.u8(7),
        BeliefRevisionFailureSnapshotV1::NegativeDeltaIncludesSupportingEvidence => writer.u8(8),
        BeliefRevisionFailureSnapshotV1::DeclaredProvenanceRootsBelowMinimum { required, actual } => {
            writer.u8(9);
            writer.usize(*required)?;
            writer.usize(*actual)?;
        }
        BeliefRevisionFailureSnapshotV1::DeltaExceedsPolicy { maximum, actual } => {
            writer.u8(10);
            writer.f32(*maximum);
            writer.f32(*actual);
        }
        BeliefRevisionFailureSnapshotV1::CalibrationMissing => writer.u8(11),
        BeliefRevisionFailureSnapshotV1::CalibrationSamplesBelowMinimum { required, actual } => {
            writer.u8(12);
            writer.u64(*required);
            writer.u64(*actual);
        }
        BeliefRevisionFailureSnapshotV1::CalibrationEceAboveMaximum { maximum, actual } => {
            writer.u8(13);
            writer.f64(*maximum);
            writer.f64(*actual);
        }
        BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentMissing => writer.u8(14),
        BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentForDifferentClaim {
            expected_claim,
            actual_claim,
        } => {
            writer.u8(15);
            writer.u64(expected_claim.0);
            writer.u64(actual_claim.0);
        }
        BeliefRevisionFailureSnapshotV1::UncertaintyAssessmentPredatesEvidence {
            assessment_cycle,
            latest_evidence_cycle,
        } => {
            writer.u8(16);
            writer.u64(*assessment_cycle);
            writer.u64(*latest_evidence_cycle);
        }
        BeliefRevisionFailureSnapshotV1::RequiredUncertaintyDimensionUnassessed(dimension) => {
            writer.u8(17);
            writer.u8(uncertainty_dimension_tag(*dimension));
        }
        BeliefRevisionFailureSnapshotV1::UncertaintyAboveMaximum {
            dimension,
            maximum,
            actual,
        } => {
            writer.u8(18);
            writer.u8(uncertainty_dimension_tag(*dimension));
            writer.f64(*maximum);
            writer.f64(*actual);
        }
        BeliefRevisionFailureSnapshotV1::UnresolvedContradictionBlocksStrengthen { count } => {
            writer.u8(19);
            writer.usize(*count)?;
        }
        BeliefRevisionFailureSnapshotV1::CausalStrengthenLacksInterventionalBasis => writer.u8(20),
    }
    Ok(())
}

fn evidence_kind_tag(kind: EvidenceKind) -> u8 {
    match kind {
        EvidenceKind::Report => 1,
        EvidenceKind::Observation => 2,
        EvidenceKind::Measurement => 3,
        EvidenceKind::Intervention => 4,
        EvidenceKind::Replication => 5,
        EvidenceKind::Simulation => 6,
        EvidenceKind::Deduction => 7,
        EvidenceKind::ToolResult => 8,
    }
}

fn evidence_polarity_tag(polarity: EvidencePolarity) -> u8 {
    match polarity {
        EvidencePolarity::Supports => 1,
        EvidencePolarity::Contradicts => 2,
        EvidencePolarity::Contextualizes => 3,
    }
}

struct CanonicalDigestWriter {
    hasher: blake3::Hasher,
}

impl CanonicalDigestWriter {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(domain);
        Self { hasher }
    }
    fn u8(&mut self, value: u8) {
        self.hasher.update(&[value]);
    }
    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }
    fn u64(&mut self, value: u64) {
        self.hasher.update(&value.to_le_bytes());
    }
    fn usize(&mut self, value: usize) -> Result<(), EpochBoundRevisionReceiptDataError> {
        self.u64(
            u64::try_from(value).map_err(|_| EpochBoundRevisionReceiptDataError::LengthOverflow)?,
        );
        Ok(())
    }
    fn count(&mut self, value: usize) -> Result<(), EpochBoundRevisionReceiptDataError> {
        self.usize(value)
    }
    fn f32(&mut self, value: f32) {
        self.hasher.update(&value.to_bits().to_le_bytes());
    }
    fn f64(&mut self, value: f64) {
        self.hasher.update(&value.to_bits().to_le_bytes());
    }
    fn bytes32(&mut self, value: [u8; 32]) {
        self.hasher.update(&value);
    }
    fn string(&mut self, value: &str) -> Result<(), EpochBoundRevisionReceiptDataError> {
        self.usize(value.len())?;
        self.hasher.update(value.as_bytes());
        Ok(())
    }
    fn optional_string(
        &mut self,
        value: Option<&str>,
    ) -> Result<(), EpochBoundRevisionReceiptDataError> {
        match value {
            Some(value) => {
                self.bool(true);
                self.string(value)?;
            }
            None => self.bool(false),
        }
        Ok(())
    }
    fn optional_f64(&mut self, value: Option<f64>) {
        match value {
            Some(value) => {
                self.bool(true);
                self.f64(value);
            }
            None => self.bool(false),
        }
    }
    fn finish(self) -> [u8; 32] {
        *self.hasher.finalize().as_bytes()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EpochBoundRevisionReceiptDataError {
    SchemaContract(EpochBoundRevisionReceiptSchemaError),
    UnsupportedVersion,
    InvalidReceiptId,
    InvalidAuthorityEpochSequence,
    InvalidProposedDelta(f32),
    RationaleTooLarge { actual: usize, maximum: usize },
    TooManyBasisRecords { actual: usize, maximum: usize },
    TooManyDuplicateBasisIds { actual: usize, maximum: usize },
    DuplicateUniqueBasisId(EvidenceId),
    BasisSnapshotIdMismatch { requested: EvidenceId, snapshot: EvidenceId },
    EvaluationPredatesBasisEvidence {
        evidence_id: EvidenceId,
        evaluated_at_cycle: u64,
        observed_at_cycle: u64,
    },
    OptionalTextTooLarge { actual: usize, maximum: usize },
    DuplicateDuplicateBasisId(EvidenceId),
    DuplicateBasisIdMissingFromBasis(EvidenceId),
    InvalidPolicySchema,
    NonCanonicalUncertaintyCaps,
    InvalidUnitFloat(f64),
    EvaluationPredatesUncertaintyAssessment {
        evaluated_at_cycle: u64,
        assessed_at_cycle: u64,
    },
    TooManyUncertaintyBasisIds { actual: usize, maximum: usize },
    UnsupportedDecisionSnapshotVersion,
    TooManyDecisionFailures { actual: usize, maximum: usize },
    DecisionEligibilityMismatch,
    TooManyRoutingFailures { actual: usize, maximum: usize },
    InvalidFailureFloat,
    LengthOverflow,
    ReceiptDigestMismatch,
}

impl fmt::Display for EpochBoundRevisionReceiptDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epoch-bound revision receipt data invalid: {self:?}")
    }
}
impl Error for EpochBoundRevisionReceiptDataError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(epoch_sequence: u64) -> EpochBoundRevisionReceiptDataV2 {
        let policy_schema = BeliefRevisionPolicySchemaV1::new(0.2, 1, false, 0, 1.0).unwrap();
        let decision_snapshot = BeliefRevisionDecisionSnapshotV1 {
            version: BeliefRevisionSnapshotVersion::V1,
            eligible: true,
            declared_provenance_root_count: 1,
            failures: vec![],
        };
        let mut out = EpochBoundRevisionReceiptDataV2 {
            version: EpochBoundRevisionReceiptDataVersion::V2,
            receipt_id: BeliefRevisionReceiptId(42),
            claim_id: ClaimId(7),
            proposed_delta: 0.1,
            rationale: "fresh evaluation under active epoch".into(),
            basis: vec![],
            duplicate_basis_evidence_ids: vec![],
            policy_schema,
            calibration: None,
            uncertainty: None,
            decision_snapshot,
            evaluated_at_cycle: 100,
            authority_epoch_digest: [9; 32],
            authority_epoch_sequence: epoch_sequence,
            receipt_digest: EpochBoundRevisionReceiptDigestV2([0; 32]),
        };
        out.receipt_digest = digest_data(&out).unwrap();
        out
    }

    #[test]
    fn typed_v2_data_verifies_but_grants_no_authority() {
        let receipt = fixture(3);
        receipt.verify().unwrap();
        assert_eq!(receipt.authority_epoch_sequence(), 3);
        assert!(!receipt.mutation_authority());
        assert!(!receipt.accepted_by_current_mutation_facade());
    }

    #[test]
    fn epoch_binding_changes_canonical_receipt_identity() {
        let left = fixture(3);
        let right = fixture(4);
        assert_ne!(left.receipt_digest(), right.receipt_digest());
    }

    #[test]
    fn zero_epoch_sequence_is_rejected() {
        let mut receipt = fixture(1);
        receipt.authority_epoch_sequence = 0;
        assert_eq!(
            receipt.verify().unwrap_err(),
            EpochBoundRevisionReceiptDataError::InvalidAuthorityEpochSequence
        );
    }

    #[test]
    fn digest_covers_rationale_and_epoch_digest() {
        let base = fixture(3);
        let mut changed_rationale = base.clone();
        changed_rationale.rationale.push_str(" changed");
        changed_rationale.receipt_digest = digest_data(&changed_rationale).unwrap();
        assert_ne!(base.receipt_digest(), changed_rationale.receipt_digest());

        let mut changed_epoch = base.clone();
        changed_epoch.authority_epoch_digest[0] ^= 0x80;
        changed_epoch.receipt_digest = digest_data(&changed_epoch).unwrap();
        assert_ne!(base.receipt_digest(), changed_epoch.receipt_digest());
    }
}
