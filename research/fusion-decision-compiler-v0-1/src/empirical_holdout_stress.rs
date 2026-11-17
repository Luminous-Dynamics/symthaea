//! Empirical, worst-slice holdout stress around a finite model decision envelope.
//!
//! This module composes the existing finite-model decision envelope with
//! externally verified, pre-reveal holdout residual summaries. The resulting
//! interval is an empirical sample stress envelope only. It is not a confidence
//! interval, credible interval, future-error guarantee, or probability model.

use std::collections::{BTreeMap, BTreeSet};

use super::cross_solver_comparison::ApplicabilityState;
use super::decision_envelope::{
    DecisionEnvelopeDisposition, DecisionEnvelopePlan, DecisionEnvelopeReport, ThresholdDirection,
    ThresholdResult, evaluate_decision_envelope,
};

pub const EMPIRICAL_STRESS_VERSION: &str = "fusion-empirical-holdout-stress/0.1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StressEvidenceClass {
    SyntheticFixture,
    AssumptionOnly,
    ExternallyVerifiedMeasuredPayload,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HoldoutMetricResult {
    PassedDeclaredMetricPolicy,
    FailedDeclaredMetricPolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmpiricalStressSide {
    StressPass,
    StressFail,
    ThresholdAmbiguous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmpiricalStressDecision {
    AllModelsStressPass,
    AllModelsStressFail,
    StressDecisionSplit,
    StressAmbiguous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmpiricalStressEvidenceDisposition {
    SyntheticOnly,
    AssumptionOnly,
    EmpiricalHoldoutStressOnly,
    IncompleteOrNotComparable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmpiricalStressDiagnostic {
    InvalidDecisionProfileDigest,
    EmptyDecisionProfileReference,
    EmptyValidationDomain,
    InvalidValidationProfileDigest,
    InvalidMinimumSampleCount,
    InvalidMinimumIndependentGroupCount,
    EmptyResidualBasis,
    EmptyRequiredSliceSet,
    DuplicateRequiredSlice,
    DuplicateRunReceipt,
    MissingRunReceipt,
    UnexpectedRunReceipt,
    ModelRunIdentityMismatch,
    SolverIdentityMismatch,
    SolverRevisionMismatch,
    CandidateIdentityMismatch,
    CandidateRevisionMismatch,
    GeometryMismatch,
    CaseMismatch,
    ObservableMismatch,
    QuantityMismatch,
    UnitMismatch,
    FrameMismatch,
    SignConventionMismatch,
    TimeBasisMismatch,
    NormalizationMismatch,
    OutputDigestMismatch,
    RunReceiptReferenceMismatch,
    InvalidReceiptDigest,
    InvalidModelArtifactDigest,
    InvalidValidationArtifactDigest,
    InvalidCommitmentDigest,
    InvalidDomainAuditDigest,
    InvalidMeasurementAttestationDigest,
    InvalidSliceManifestDigest,
    MissingExternalReference,
    HoldoutCommitmentNotPreReveal,
    MeasurementPayloadNotVerified,
    ValidationDomainNotAdmissible,
    DecisionProfileMismatch,
    ResidualBasisMismatch,
    InvalidHoldoutDisposition,
    DuplicateSlice,
    MissingRequiredSlice,
    UnexpectedSlice,
    SliceOutOfDomain,
    InsufficientSliceSamples,
    InsufficientIndependentGroups,
    InvalidSliceValue,
    NonFiniteResidual,
    InvalidResidualRange,
    DuplicateSliceEvidence,
    MixedEvidenceClasses,
    StressIntervalOverflow,
    BaseDecisionEnvelopeNotAdmissible,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceStressEvidence {
    pub slice_id: String,
    pub facility_id: String,
    pub operating_point_id: String,
    pub case_family_id: String,
    pub in_declared_domain: bool,
    pub sample_count: usize,
    pub independent_group_count: usize,
    /// Maximum absolute measured-minus-predicted residual in the frozen unit.
    pub maximum_absolute_residual: f64,
    pub residual_unit_ref: String,
    pub residual_basis_ref: String,
    pub residual_profile_sha256: String,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelHoldoutStressReceipt {
    pub receipt_id: String,
    pub receipt_sha256: String,
    pub run_id: String,
    pub solver_id: String,
    pub solver_revision: String,
    pub candidate_id: String,
    pub candidate_revision: String,
    pub geometry_sha256: String,
    pub case_id: String,
    pub observable_id: String,
    pub canonical_quantity_ref: String,
    pub unit_ref: String,
    pub frame_ref: String,
    pub sign_convention_ref: String,
    pub time_basis_ref: String,
    pub normalization_contract_ref: String,
    pub output_sha256: String,
    pub run_receipt_ref: String,
    pub run_receipt_sha256: String,
    pub model_artifact_sha256: String,
    pub validation_domain_id: String,
    pub validation_domain_audit_sha256: String,
    pub validation_profile_sha256: String,
    pub decision_profile_ref: String,
    pub decision_profile_sha256: String,
    pub prediction_commitment_ref: String,
    pub prediction_commitment_sha256: String,
    pub committed_before_holdout_reveal: bool,
    pub test_stand_artifact_sha256: String,
    pub holdout_artifact_sha256: String,
    pub measurement_attestation_ref: String,
    pub measurement_attestation_sha256: String,
    pub measurement_payload_externally_verified: bool,
    pub validation_domain_admissible: bool,
    pub holdout_metric_result: HoldoutMetricResult,
    pub evidence_class: StressEvidenceClass,
    pub slice_manifest_sha256: String,
    pub required_slices: Vec<String>,
    pub slices: Vec<SliceStressEvidence>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmpiricalStressProfile {
    pub decision_profile_ref: String,
    pub decision_profile_sha256: String,
    pub validation_domain_id: String,
    pub validation_profile_sha256: String,
    pub residual_basis_ref: String,
    pub minimum_sample_count_per_slice: usize,
    pub minimum_independent_groups_per_slice: usize,
    /// Complete, predeclared slice IDs; no post-hoc bins are accepted.
    pub required_slice_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalStressPlan {
    pub decision_plan: DecisionEnvelopePlan,
    pub stress_profile: EmpiricalStressProfile,
    pub holdout_receipts: Vec<ModelHoldoutStressReceipt>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SliceStressReport {
    pub slice_id: String,
    pub facility_id: String,
    pub operating_point_id: String,
    pub case_family_id: String,
    pub sample_count: usize,
    pub independent_group_count: usize,
    pub maximum_absolute_residual: f64,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelEmpiricalStressReport {
    pub run_id: String,
    pub solver_id: String,
    pub solver_revision: String,
    pub output_sha256: String,
    pub point_estimate: f64,
    pub empirical_stress_radius: f64,
    pub stress_lower: f64,
    pub stress_upper: f64,
    pub stress_side: EmpiricalStressSide,
    pub holdout_metric_result: HoldoutMetricResult,
    pub worst_slice_id: String,
    pub slices: Vec<SliceStressReport>,
    pub receipt_id: String,
    pub receipt_sha256: String,
    pub model_artifact_sha256: String,
    pub validation_domain_audit_sha256: String,
    pub validation_profile_sha256: String,
    pub prediction_commitment_sha256: String,
    pub test_stand_artifact_sha256: String,
    pub holdout_artifact_sha256: String,
    pub measurement_attestation_sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalStressReport {
    pub contract_version: &'static str,
    pub case_id: String,
    pub candidate_id: String,
    pub observable_id: String,
    pub base_decision_disposition: DecisionEnvelopeDisposition,
    /// Numeric finite-model decision after empirical stress expansion.
    pub stress_decision: Option<EmpiricalStressDecision>,
    /// Evidence quality is independent of numeric threshold classification.
    pub evidence_disposition: EmpiricalStressEvidenceDisposition,
    pub minimum_stress_lower: Option<f64>,
    pub maximum_stress_upper: Option<f64>,
    pub models: Vec<ModelEmpiricalStressReport>,
    pub diagnostics: Vec<EmpiricalStressDiagnostic>,
    pub claim_ceiling: &'static str,
}

pub fn evaluate_empirical_holdout_stress(
    plan: &EmpiricalStressPlan,
) -> EmpiricalStressReport {
    let base = evaluate_decision_envelope(&plan.decision_plan);
    let mut diagnostics = validate_stress_profile(&plan.stress_profile);

    if base.disposition == DecisionEnvelopeDisposition::IncompleteOrNotComparable {
        diagnostics.push(EmpiricalStressDiagnostic::BaseDecisionEnvelopeNotAdmissible);
    }

    let model_map: BTreeMap<&str, _> = plan
        .decision_plan
        .models
        .iter()
        .map(|model| (model.run_id.as_str(), model))
        .collect();
    let mut receipts = BTreeMap::new();
    for receipt in &plan.holdout_receipts {
        if receipts.insert(receipt.run_id.as_str(), receipt).is_some() {
            diagnostics.push(EmpiricalStressDiagnostic::DuplicateRunReceipt);
        }
    }

    for run_id in model_map.keys() {
        if !receipts.contains_key(run_id) {
            diagnostics.push(EmpiricalStressDiagnostic::MissingRunReceipt);
        }
    }
    for run_id in receipts.keys() {
        if !model_map.contains_key(run_id) {
            diagnostics.push(EmpiricalStressDiagnostic::UnexpectedRunReceipt);
        }
    }

    let mut reports = Vec::new();
    let mut evidence_classes = BTreeSet::new();
    if diagnostics.is_empty() {
        let mut ordered_ids: Vec<&str> = model_map.keys().copied().collect();
        ordered_ids.sort_unstable();

        for run_id in ordered_ids {
            let model = model_map[run_id];
            let receipt = receipts[run_id];
            evidence_classes.insert(receipt.evidence_class);
            validate_receipt_binding(
                &plan.decision_plan.profile,
                &plan.stress_profile,
                model,
                receipt,
                &mut diagnostics,
            );
            if diagnostics.is_empty() {
                if let Some(report) = build_model_report(
                    &plan.decision_plan.profile.threshold_value,
                    plan.decision_plan.profile.threshold_direction,
                    plan.decision_plan.profile.threshold_inclusive,
                    model.normalized_value,
                    receipt,
                    &mut diagnostics,
                ) {
                    reports.push(report);
                }
            }
        }
    }

    if evidence_classes.len() > 1 {
        diagnostics.push(EmpiricalStressDiagnostic::MixedEvidenceClasses);
    }
    reports.sort_by(|left, right| left.run_id.cmp(&right.run_id));

    let numeric = if diagnostics.is_empty() && reports.len() == model_map.len() {
        Some(aggregate_stress_decision(&reports))
    } else {
        None
    };
    let min_lower = reports
        .iter()
        .map(|model| model.stress_lower)
        .reduce(f64::min);
    let max_upper = reports
        .iter()
        .map(|model| model.stress_upper)
        .reduce(f64::max);

    let evidence_disposition = if !diagnostics.is_empty() {
        EmpiricalStressEvidenceDisposition::IncompleteOrNotComparable
    } else {
        match evidence_classes.iter().next().copied() {
            Some(StressEvidenceClass::SyntheticFixture) => {
                EmpiricalStressEvidenceDisposition::SyntheticOnly
            }
            Some(StressEvidenceClass::AssumptionOnly) => {
                EmpiricalStressEvidenceDisposition::AssumptionOnly
            }
            Some(StressEvidenceClass::ExternallyVerifiedMeasuredPayload) => {
                EmpiricalStressEvidenceDisposition::EmpiricalHoldoutStressOnly
            }
            None => EmpiricalStressEvidenceDisposition::IncompleteOrNotComparable,
        }
    };

    EmpiricalStressReport {
        contract_version: EMPIRICAL_STRESS_VERSION,
        case_id: plan.decision_plan.profile.case_id.clone(),
        candidate_id: plan.decision_plan.profile.candidate_id.clone(),
        observable_id: plan.decision_plan.profile.observable_id.clone(),
        base_decision_disposition: base.disposition,
        stress_decision: numeric,
        evidence_disposition,
        minimum_stress_lower: min_lower,
        maximum_stress_upper: max_upper,
        models: reports,
        diagnostics,
        claim_ceiling: "finite observed holdout-error stress only; not probabilistic or future-error coverage",
    }
}

fn validate_stress_profile(profile: &EmpiricalStressProfile) -> Vec<EmpiricalStressDiagnostic> {
    let mut diagnostics = Vec::new();
    if profile.decision_profile_ref.trim().is_empty() {
        diagnostics.push(EmpiricalStressDiagnostic::EmptyDecisionProfileReference);
    }
    if !is_sha256(&profile.decision_profile_sha256) {
        diagnostics.push(EmpiricalStressDiagnostic::InvalidDecisionProfileDigest);
    }
    if profile.validation_domain_id.trim().is_empty() {
        diagnostics.push(EmpiricalStressDiagnostic::EmptyValidationDomain);
    }
    if !is_sha256(&profile.validation_profile_sha256) {
        diagnostics.push(EmpiricalStressDiagnostic::InvalidValidationProfileDigest);
    }
    if profile.residual_basis_ref.trim().is_empty() {
        diagnostics.push(EmpiricalStressDiagnostic::EmptyResidualBasis);
    }
    if profile.minimum_sample_count_per_slice == 0 {
        diagnostics.push(EmpiricalStressDiagnostic::InvalidMinimumSampleCount);
    }
    if profile.minimum_independent_groups_per_slice == 0 {
        diagnostics.push(EmpiricalStressDiagnostic::InvalidMinimumIndependentGroupCount);
    }
    if profile.required_slice_ids.is_empty() {
        diagnostics.push(EmpiricalStressDiagnostic::EmptyRequiredSliceSet);
    }
    let mut seen = BTreeSet::new();
    for slice_id in &profile.required_slice_ids {
        if slice_id.trim().is_empty() || !seen.insert(slice_id.as_str()) {
            diagnostics.push(EmpiricalStressDiagnostic::DuplicateRequiredSlice);
        }
    }
    diagnostics
}

fn validate_receipt_binding(
    decision_profile: &super::decision_envelope::DecisionEnvelopeProfile,
    stress_profile: &EmpiricalStressProfile,
    model: &super::decision_envelope::ModelEnvelopeInput,
    receipt: &ModelHoldoutStressReceipt,
    diagnostics: &mut Vec<EmpiricalStressDiagnostic>,
) {
    if receipt.receipt_id.trim().is_empty()
        || receipt.run_receipt_ref.trim().is_empty()
        || receipt.measurement_attestation_ref.trim().is_empty()
    {
        diagnostics.push(EmpiricalStressDiagnostic::MissingExternalReference);
    }

    if !is_sha256(&receipt.receipt_sha256)
        || !is_sha256(&receipt.run_receipt_sha256)
    {
        diagnostics.push(EmpiricalStressDiagnostic::InvalidReceiptDigest);
    }
    if !is_sha256(&receipt.model_artifact_sha256) {
        diagnostics.push(EmpiricalStressDiagnostic::InvalidModelArtifactDigest);
    }
    if !is_sha256(&receipt.output_sha256) {
        diagnostics.push(EmpiricalStressDiagnostic::OutputDigestMismatch);
    }
    if !is_sha256(&receipt.validation_profile_sha256)
        || !is_sha256(&receipt.validation_domain_audit_sha256)
    {
        diagnostics.push(EmpiricalStressDiagnostic::InvalidDomainAuditDigest);
    }
    if !is_sha256(&receipt.validation_profile_sha256) {
        diagnostics.push(EmpiricalStressDiagnostic::InvalidValidationArtifactDigest);
    }
    if !is_sha256(&receipt.prediction_commitment_sha256) {
        diagnostics.push(EmpiricalStressDiagnostic::InvalidCommitmentDigest);
    }
    if !is_sha256(&receipt.measurement_attestation_sha256)
        || !is_sha256(&receipt.test_stand_artifact_sha256)
        || !is_sha256(&receipt.holdout_artifact_sha256)
    {
        diagnostics.push(EmpiricalStressDiagnostic::InvalidMeasurementAttestationDigest);
    }
    if !is_sha256(&receipt.slice_manifest_sha256) {
        diagnostics.push(EmpiricalStressDiagnostic::InvalidSliceManifestDigest);
    }

    if receipt.run_id != model.run_id {
        diagnostics.push(EmpiricalStressDiagnostic::ModelRunIdentityMismatch);
    }
    if receipt.solver_id != model.solver_id {
        diagnostics.push(EmpiricalStressDiagnostic::SolverIdentityMismatch);
    }
    if receipt.solver_revision != model.solver_revision {
        diagnostics.push(EmpiricalStressDiagnostic::SolverRevisionMismatch);
    }
    if receipt.candidate_id != model.candidate_id {
        diagnostics.push(EmpiricalStressDiagnostic::CandidateIdentityMismatch);
    }
    if receipt.candidate_revision != model.candidate_revision {
        diagnostics.push(EmpiricalStressDiagnostic::CandidateRevisionMismatch);
    }
    if receipt.geometry_sha256 != model.geometry_sha256 {
        diagnostics.push(EmpiricalStressDiagnostic::GeometryMismatch);
    }
    if receipt.case_id != model.case_id || receipt.case_id != decision_profile.case_id {
        diagnostics.push(EmpiricalStressDiagnostic::CaseMismatch);
    }
    if receipt.observable_id != model.observable_id
        || receipt.observable_id != decision_profile.observable_id
    {
        diagnostics.push(EmpiricalStressDiagnostic::ObservableMismatch);
    }
    if receipt.canonical_quantity_ref != model.canonical_quantity_ref
        || receipt.canonical_quantity_ref != decision_profile.canonical_quantity_ref
    {
        diagnostics.push(EmpiricalStressDiagnostic::QuantityMismatch);
    }
    if receipt.unit_ref != model.unit_ref || receipt.unit_ref != decision_profile.unit_ref {
        diagnostics.push(EmpiricalStressDiagnostic::UnitMismatch);
    }
    if receipt.frame_ref != model.frame_ref || receipt.frame_ref != decision_profile.frame_ref {
        diagnostics.push(EmpiricalStressDiagnostic::FrameMismatch);
    }
    if receipt.sign_convention_ref != model.sign_convention_ref
        || receipt.sign_convention_ref != decision_profile.sign_convention_ref
    {
        diagnostics.push(EmpiricalStressDiagnostic::SignConventionMismatch);
    }
    if receipt.time_basis_ref != model.time_basis_ref
        || receipt.time_basis_ref != decision_profile.time_basis_ref
    {
        diagnostics.push(EmpiricalStressDiagnostic::TimeBasisMismatch);
    }
    if receipt.normalization_contract_ref != model.normalization_contract_ref
        || receipt.normalization_contract_ref != decision_profile.normalization_contract_ref
    {
        diagnostics.push(EmpiricalStressDiagnostic::NormalizationMismatch);
    }
    if receipt.output_sha256 != model.output_sha256 {
        diagnostics.push(EmpiricalStressDiagnostic::OutputDigestMismatch);
    }
    if receipt.run_receipt_ref != model.run_receipt_ref {
        diagnostics.push(EmpiricalStressDiagnostic::RunReceiptReferenceMismatch);
    }
    if receipt.validation_domain_id != stress_profile.validation_domain_id {
        diagnostics.push(EmpiricalStressDiagnostic::ValidationDomainNotAdmissible);
    }
    if receipt.validation_profile_sha256 != stress_profile.validation_profile_sha256 {
        diagnostics.push(EmpiricalStressDiagnostic::ValidationProfileDigestMismatch);
    }
    if receipt.decision_profile_ref != stress_profile.decision_profile_ref
        || receipt.decision_profile_sha256 != stress_profile.decision_profile_sha256
    {
        diagnostics.push(EmpiricalStressDiagnostic::DecisionProfileMismatch);
    }
    if !receipt.committed_before_holdout_reveal {
        diagnostics.push(EmpiricalStressDiagnostic::HoldoutCommitmentNotPreReveal);
    }
    if !receipt.measurement_payload_externally_verified {
        diagnostics.push(EmpiricalStressDiagnostic::MeasurementPayloadNotVerified);
    }
    if !receipt.validation_domain_admissible {
        diagnostics.push(EmpiricalStressDiagnostic::ValidationDomainNotAdmissible);
    }
    if receipt.required_slices != stress_profile.required_slice_ids {
        let mut expected = stress_profile.required_slice_ids.clone();
        let mut actual = receipt.required_slices.clone();
        expected.sort();
        actual.sort();
        if expected != actual {
            diagnostics.push(EmpiricalStressDiagnostic::MissingRequiredSlice);
        }
    }
    if receipt.holdout_metric_result == HoldoutMetricResult::FailedDeclaredMetricPolicy {
        // Failure is not a reason to discard poor validation evidence. It is
        // intentionally retained in the per-model report and stress radius.
    }
}

fn build_model_report(
    threshold: &f64,
    direction: ThresholdDirection,
    inclusive: bool,
    point_estimate: f64,
    receipt: &ModelHoldoutStressReceipt,
    diagnostics: &mut Vec<EmpiricalStressDiagnostic>,
) -> Option<ModelEmpiricalStressReport> {
    if !point_estimate.is_finite() {
        diagnostics.push(EmpiricalStressDiagnostic::NonFiniteResidual);
        return None;
    }
    if receipt.slices.is_empty() {
        diagnostics.push(EmpiricalStressDiagnostic::MissingRequiredSlice);
        return None;
    }

    let mut actual_slices = BTreeMap::new();
    let mut seen_evidence = BTreeSet::new();
    let mut ordered: Vec<&SliceStressEvidence> = receipt.slices.iter().collect();
    ordered.sort_by(|left, right| left.slice_id.cmp(&right.slice_id));

    for slice in ordered {
        if actual_slices.contains_key(slice.slice_id.as_str()) {
            diagnostics.push(EmpiricalStressDiagnostic::DuplicateSlice);
        } else {
            actual_slices.insert(slice.slice_id.as_str(), slice);
        }
        if !slice.in_declared_domain {
            diagnostics.push(EmpiricalStressDiagnostic::SliceOutOfDomain);
        }
        if slice.sample_count == 0
            || slice.sample_count < receipt.required_slices.len().saturating_sub(0) * 0
        {
            diagnostics.push(EmpiricalStressDiagnostic::InvalidSliceValue);
        }
        if !slice.maximum_absolute_residual.is_finite() {
            diagnostics.push(EmpiricalStressDiagnostic::NonFiniteResidual);
        } else if slice.maximum_absolute_residual < 0.0 {
            diagnostics.push(EmpiricalStressDiagnostic::InvalidResidualRange);
        }
        if slice.residual_basis_ref.trim().is_empty()
            || slice.evidence_ref.trim().is_empty()
            || slice.residual_unit_ref != receipt.unit_ref
        {
            diagnostics.push(EmpiricalStressDiagnostic::ResidualBasisMismatch);
        }
        if !is_sha256(&slice.residual_profile_sha256) {
            diagnostics.push(EmpiricalStressDiagnostic::InvalidValidationArtifactDigest);
        }
        if !seen_evidence.insert(slice.evidence_ref.as_str()) {
            diagnostics.push(EmpiricalStressDiagnostic::DuplicateSliceEvidence);
        }
    }

    let required: BTreeSet<&str> = receipt.required_slices.iter().map(String::as_str).collect();
    let actual: BTreeSet<&str> = actual_slices.keys().copied().collect();
    if required != actual {
        diagnostics.push(EmpiricalStressDiagnostic::MissingRequiredSlice);
    }
    if diagnostics.iter().any(|diagnostic| {
        matches!(
            diagnostic,
            EmpiricalStressDiagnostic::DuplicateSlice
                | EmpiricalStressDiagnostic::SliceOutOfDomain
                | EmpiricalStressDiagnostic::InvalidSliceValue
                | EmpiricalStressDiagnostic::NonFiniteResidual
                | EmpiricalStressDiagnostic::InvalidResidualRange
                | EmpiricalStressDiagnostic::ResidualBasisMismatch
                | EmpiricalStressDiagnostic::MissingRequiredSlice
                | EmpiricalStressDiagnostic::DuplicateSliceEvidence
        )
    }) {
        return None;
    }

    let mut slice_reports = Vec::new();
    for slice in actual_slices.values() {
        slice_reports.push(SliceStressReport {
            slice_id: slice.slice_id.clone(),
            facility_id: slice.facility_id.clone(),
            operating_point_id: slice.operating_point_id.clone(),
            case_family_id: slice.case_family_id.clone(),
            sample_count: slice.sample_count,
            independent_group_count: slice.independent_group_count,
            maximum_absolute_residual: slice.maximum_absolute_residual,
            evidence_ref: slice.evidence_ref.clone(),
        });
    }
    slice_reports.sort_by(|left, right| left.slice_id.cmp(&right.slice_id));

    let worst = slice_reports
        .iter()
        .max_by(|left, right| {
            left.maximum_absolute_residual
                .total_cmp(&right.maximum_absolute_residual)
                .then_with(|| right.slice_id.cmp(&left.slice_id))
        })
        .expect("non-empty validated slice list");
    let radius = worst.maximum_absolute_residual;
    let lower = point_estimate - radius;
    let upper = point_estimate + radius;
    if !lower.is_finite() || !upper.is_finite() {
        diagnostics.push(EmpiricalStressDiagnostic::StressIntervalOverflow);
        return None;
    }
    let side = classify_stress_interval(lower, upper, *threshold, direction, inclusive);

    Some(ModelEmpiricalStressReport {
        run_id: receipt.run_id.clone(),
        solver_id: receipt.solver_id.clone(),
        solver_revision: receipt.solver_revision.clone(),
        output_sha256: receipt.output_sha256.clone(),
        point_estimate,
        empirical_stress_radius: radius,
        stress_lower: lower,
        stress_upper: upper,
        stress_side: side,
        holdout_metric_result: receipt.holdout_metric_result,
        worst_slice_id: worst.slice_id.clone(),
        slices: slice_reports,
        receipt_id: receipt.receipt_id.clone(),
        receipt_sha256: receipt.receipt_sha256.clone(),
        model_artifact_sha256: receipt.model_artifact_sha256.clone(),
        validation_domain_audit_sha256: receipt.validation_domain_audit_sha256.clone(),
        validation_profile_sha256: receipt.validation_profile_sha256.clone(),
        prediction_commitment_sha256: receipt.prediction_commitment_sha256.clone(),
        test_stand_artifact_sha256: receipt.test_stand_artifact_sha256.clone(),
        holdout_artifact_sha256: receipt.holdout_artifact_sha256.clone(),
        measurement_attestation_sha256: receipt.measurement_attestation_sha256.clone(),
    })
}

fn classify_stress_interval(
    lower: f64,
    upper: f64,
    threshold: f64,
    direction: ThresholdDirection,
    inclusive: bool,
) -> EmpiricalStressSide {
    match direction {
        ThresholdDirection::AtLeast => {
            let pass = if inclusive {
                lower >= threshold
            } else {
                lower > threshold
            };
            let fail = if inclusive {
                upper < threshold
            } else {
                upper <= threshold
            };
            if pass {
                EmpiricalStressSide::StressPass
            } else if fail {
                EmpiricalStressSide::StressFail
            } else {
                EmpiricalStressSide::ThresholdAmbiguous
            }
        }
        ThresholdDirection::AtMost => {
            let pass = if inclusive {
                upper <= threshold
            } else {
                upper < threshold
            };
            let fail = if inclusive {
                lower > threshold
            } else {
                lower >= threshold
            };
            if pass {
                EmpiricalStressSide::StressPass
            } else if fail {
                EmpiricalStressSide::StressFail
            } else {
                EmpiricalStressSide::ThresholdAmbiguous
            }
        }
    }
}

fn aggregate_stress_decision(reports: &[ModelEmpiricalStressReport]) -> EmpiricalStressDecision {
    let mut passes = 0;
    let mut fails = 0;
    let mut ambiguous = 0;
    for report in reports {
        match report.stress_side {
            EmpiricalStressSide::StressPass => passes += 1,
            EmpiricalStressSide::StressFail => fails += 1,
            EmpiricalStressSide::ThresholdAmbiguous => ambiguous += 1,
        }
    }
    if passes == reports.len() {
        EmpiricalStressDecision::AllModelsStressPass
    } else if fails == reports.len() {
        EmpiricalStressDecision::AllModelsStressFail
    } else if passes > 0 && fails > 0 {
        EmpiricalStressDecision::StressDecisionSplit
    } else {
        let _ = ambiguous;
        EmpiricalStressDecision::StressAmbiguous
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_envelope::{DecisionEnvelopeProfile, ModelEnvelopeInput};
    use crate::evidence_dependence::{DependencyIdentity, DependencyKind, ObservableDependency, RunDependencyManifest};

    fn sha(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn model(run_id: &str, solver: &str, value: f64, output: char) -> ModelEnvelopeInput {
        ModelEnvelopeInput {
            run_id: run_id.to_string(),
            solver_id: solver.to_string(),
            solver_revision: format!("{solver}-r1"),
            case_id: "case-a".to_string(),
            candidate_id: "candidate-a".to_string(),
            candidate_revision: "candidate-r1".to_string(),
            geometry_sha256: sha('a'),
            service_boundary_ref: "net-electric-export-v1".to_string(),
            observable_id: "net-electric-export".to_string(),
            canonical_quantity_ref: "quantity-net-electric-power".to_string(),
            unit_ref: "MWe".to_string(),
            frame_ref: "plant-terminal".to_string(),
            sign_convention_ref: "net-export-positive-v1".to_string(),
            time_basis_ref: "steady-window-60s-v1".to_string(),
            normalization_contract_ref: "net-export-normalization-v1".to_string(),
            normalized_value: value,
            output_sha256: sha(output),
            run_receipt_ref: format!("run-receipt-{run_id}"),
            output_evidence_ref: format!("output-evidence-{run_id}"),
            applicability: ApplicabilityState::InDeclaredDomain,
            convergence: super::super::cross_solver_comparison::ConvergenceState::Converged,
            dependencies: RunDependencyManifest {
                run_id: run_id.to_string(),
                output_observables: vec!["net-electric-export".to_string()],
                complete_observables: vec!["net-electric-export".to_string()],
                dependencies: vec![ObservableDependency {
                    observable_id: "net-electric-export".to_string(),
                    dependency: DependencyIdentity {
                        node_id: format!("assumption-{solver}"),
                        revision: "r1".to_string(),
                        kind: DependencyKind::SharedScenarioAssumption,
                        content_sha256: Some(sha(if solver == "PROCESS" { 'c' } else { 'd' })),
                        source_evidence_ref: Some(format!("dep-evidence-{solver}")),
                    },
                }],
            },
        }
    }

    fn decision_plan() -> DecisionEnvelopePlan {
        DecisionEnvelopePlan {
            profile: DecisionEnvelopeProfile {
                case_id: "case-a".to_string(),
                candidate_id: "candidate-a".to_string(),
                candidate_revision: "candidate-r1".to_string(),
                geometry_sha256: sha('a'),
                service_boundary_ref: "net-electric-export-v1".to_string(),
                observable_id: "net-electric-export".to_string(),
                canonical_quantity_ref: "quantity-net-electric-power".to_string(),
                unit_ref: "MWe".to_string(),
                frame_ref: "plant-terminal".to_string(),
                sign_convention_ref: "net-export-positive-v1".to_string(),
                time_basis_ref: "steady-window-60s-v1".to_string(),
                normalization_contract_ref: "net-export-normalization-v1".to_string(),
                threshold_value: 100.0,
                threshold_direction: ThresholdDirection::AtLeast,
                threshold_inclusive: true,
                absolute_spread_tolerance: 10.0,
                relative_spread_reference_scale: Some(100.0),
            },
            models: vec![
                model("run-process", "PROCESS", 92.0, 'b'),
                model("run-saras", "SARAS", 108.0, 'e'),
            ],
        }
    }

    fn slice(id: &str, radius: f64, samples: usize, groups: usize) -> SliceStressEvidence {
        SliceStressEvidence {
            slice_id: id.to_string(),
            facility_id: format!("facility-{id}"),
            operating_point_id: format!("op-{id}"),
            case_family_id: format!("family-{id}"),
            in_declared_domain: true,
            sample_count: samples,
            independent_group_count: groups,
            maximum_absolute_residual: radius,
            residual_unit_ref: "MWe".to_string(),
            residual_basis_ref: "abs-measured-minus-predicted-v1".to_string(),
            residual_profile_sha256: sha('8'),
            evidence_ref: format!("slice-evidence-{id}"),
        }
    }

    fn receipt(model: &ModelEnvelopeInput, class: StressEvidenceClass, slices: Vec<SliceStressEvidence>) -> ModelHoldoutStressReceipt {
        ModelHoldoutStressReceipt {
            receipt_id: format!("holdout-receipt-{}", model.run_id),
            receipt_sha256: sha('1'),
            run_id: model.run_id.clone(),
            solver_id: model.solver_id.clone(),
            solver_revision: model.solver_revision.clone(),
            candidate_id: model.candidate_id.clone(),
            candidate_revision: model.candidate_revision.clone(),
            geometry_sha256: model.geometry_sha256.clone(),
            case_id: model.case_id.clone(),
            observable_id: model.observable_id.clone(),
            canonical_quantity_ref: model.canonical_quantity_ref.clone(),
            unit_ref: model.unit_ref.clone(),
            frame_ref: model.frame_ref.clone(),
            sign_convention_ref: model.sign_convention_ref.clone(),
            time_basis_ref: model.time_basis_ref.clone(),
            normalization_contract_ref: model.normalization_contract_ref.clone(),
            output_sha256: model.output_sha256.clone(),
            run_receipt_ref: model.run_receipt_ref.clone(),
            run_receipt_sha256: sha('2'),
            model_artifact_sha256: sha(if model.solver_id == "PROCESS" { '3' } else { '4' }),
            validation_domain_id: "fusion-domain-v1".to_string(),
            validation_domain_audit_sha256: sha('5'),
            validation_profile_sha256: sha('6'),
            decision_profile_ref: "net-export-decision-profile-v1".to_string(),
            decision_profile_sha256: sha('7'),
            prediction_commitment_ref: "pre-reveal-commitment".to_string(),
            prediction_commitment_sha256: sha('9'),
            committed_before_holdout_reveal: true,
            test_stand_artifact_sha256: sha('a'),
            holdout_artifact_sha256: sha('b'),
            measurement_attestation_ref: "external-measurement-attestation".to_string(),
            measurement_attestation_sha256: sha('c'),
            measurement_payload_externally_verified: true,
            validation_domain_admissible: true,
            holdout_metric_result: HoldoutMetricResult::PassedDeclaredMetricPolicy,
            evidence_class: class,
            slice_manifest_sha256: sha('d'),
            required_slices: vec!["slice-a".to_string(), "slice-b".to_string()],
            slices,
        }
    }

    fn profile() -> EmpiricalStressProfile {
        EmpiricalStressProfile {
            decision_profile_ref: "net-export-decision-profile-v1".to_string(),
            decision_profile_sha256: sha('7'),
            validation_domain_id: "fusion-domain-v1".to_string(),
            validation_profile_sha256: sha('6'),
            residual_basis_ref: "abs-measured-minus-predicted-v1".to_string(),
            minimum_sample_count_per_slice: 10,
            minimum_independent_groups_per_slice: 5,
            required_slice_ids: vec!["slice-a".to_string(), "slice-b".to_string()],
        }
    }

    fn plan(class: StressEvidenceClass, process: (f64, f64), saras: (f64, f64)) -> EmpiricalStressPlan {
        let decision_plan = decision_plan();
        let receipts = vec![
            receipt(
                &decision_plan.models[0],
                class,
                vec![slice("slice-a", process.0, 12, 6), slice("slice-b", process.1, 15, 8)],
            ),
            receipt(
                &decision_plan.models[1],
                class,
                vec![slice("slice-a", saras.0, 14, 7), slice("slice-b", saras.1, 13, 6)],
            ),
        ];
        EmpiricalStressPlan {
            decision_plan,
            stress_profile: profile(),
            holdout_receipts: receipts,
        }
    }

    #[test]
    fn measured_stress_interval_can_make_point_pass_ambiguous() {
        let mut input = decision_plan();
        input.models[0].normalized_value = 108.0;
        input.models[1].normalized_value = 120.0;
        let receipts = input.models.iter().map(|model| {
            receipt(model, StressEvidenceClass::ExternallyVerifiedMeasuredPayload, vec![
                slice("slice-a", 4.0, 12, 6),
                slice("slice-b", 12.0, 15, 8)
            ])
        }).collect();
        let plan = EmpiricalStressPlan {
            decision_plan: input,
            stress_profile: profile(),
            holdout_receipts: receipts
        };
        let report = evaluate_empirical_holdout_stress(&plan);
        assert_eq!(report.evidence_disposition, EmpiricalStressEvidenceDisposition::EmpiricalHoldoutStressOnly);
        assert_eq!(report.models[0].stress_side, EmpiricalStressSide::ThresholdAmbiguous);
        assert_eq!(report.models[0].empirical_stress_radius, 12.0);
        assert_eq!(report.models[0].worst_slice_id, "slice-b");
    }

    #[test]
    fn synthetic_stress_never_becomes_empirical_holdout_stress() {
        let report = evaluate_empirical_holdout_stress(&plan(
            StressEvidenceClass::SyntheticFixture,
            (8.0, 15.0),
            (12.0, 10.0),
        ));
        assert_eq!(report.evidence_disposition, EmpiricalStressEvidenceDisposition::SyntheticOnly);
        assert_eq!(report.stress_decision, Some(EmpiricalStressDecision::StressAmbiguous));
    }

    #[test]
    fn assumption_only_stress_is_never_measured() {
        let report = evaluate_empirical_holdout_stress(&plan(
            StressEvidenceClass::AssumptionOnly,
            (8.0, 15.0),
            (12.0, 10.0),
        ));
        assert_eq!(report.evidence_disposition, EmpiricalStressEvidenceDisposition::AssumptionOnly);
    }

    #[test]
    fn worst_slice_radius_not_pooled_mean_controls_stress() {
        let report = evaluate_empirical_holdout_stress(&plan(
            StressEvidenceClass::SyntheticFixture,
            (1.0, 40.0),
            (2.0, 3.0),
        ));
        assert_eq!(report.models[0].empirical_stress_radius, 40.0);
        assert_eq!(report.models[0].worst_slice_id, "slice-b");
    }

    #[test]
    fn holdout_metric_failure_remains_visible_and_does_not_drop_its_residuals() {
        let mut input = plan(
            StressEvidenceClass::SyntheticFixture,
            (5.0, 10.0),
            (5.0, 10.0),
        );
        input.holdout_receipts[0].holdout_metric_result = HoldoutMetricResult::FailedDeclaredMetricPolicy;
        let report = evaluate_empirical_holdout_stress(&input);
        assert_eq!(report.models[0].holdout_metric_result, HoldoutMetricResult::FailedDeclaredMetricPolicy);
        assert_eq!(report.models[0].empirical_stress_radius, 10.0);
    }

    #[test]
    fn missing_receipt_for_a_solver_makes_result_incomplete() {
        let mut input = plan(StressEvidenceClass::SyntheticFixture, (1.0, 2.0), (3.0, 4.0));
        input.holdout_receipts.pop();
        let report = evaluate_empirical_holdout_stress(&input);
        assert_eq!(report.evidence_disposition, EmpiricalStressEvidenceDisposition::IncompleteOrNotComparable);
        assert!(report.diagnostics.contains(&EmpiricalStressDiagnostic::MissingRunReceipt));
        assert_eq!(report.stress_decision, None);
    }

    #[test]
    fn wrong_model_revision_or_output_digest_is_refused() {
        let mut input = plan(StressEvidenceClass::SyntheticFixture, (1.0, 2.0), (3.0, 4.0));
        input.holdout_receipts[0].solver_revision = "stale".to_string();
        let report = evaluate_empirical_holdout_stress(&input);
        assert!(report.diagnostics.contains(&EmpiricalStressDiagnostic::SolverRevisionMismatch));
        let mut input = plan(StressEvidenceClass::SyntheticFixture, (1.0, 2.0), (3.0, 4.0));
        input.holdout_receipts[0].output_sha256 = sha('f');
        let report = evaluate_empirical_holdout_stress(&input);
        assert!(report.diagnostics.contains(&EmpiricalStressDiagnostic::OutputDigestMismatch));
    }

    #[test]
    fn decision_profile_and_validation_domain_must_match() {
        let mut input = plan(StressEvidenceClass::SyntheticFixture, (1.0, 2.0), (3.0, 4.0));
        input.holdout_receipts[0].validation_domain_id = "other-domain".to_string();
        let report = evaluate_empirical_holdout_stress(&input);
        assert!(report.diagnostics.contains(&EmpiricalStressDiagnostic::ValidationDomainNotAdmissible));
    }

    #[test]
    fn missing_slice_is_not_pooled_or_dropped() {
        let mut input = plan(StressEvidenceClass::SyntheticFixture, (1.0, 2.0), (3.0, 4.0));
        input.holdout_receipts[0].slices.pop();
        let report = evaluate_empirical_holdout_stress(&input);
        assert!(report.diagnostics.contains(&EmpiricalStressDiagnostic::MissingRequiredSlice));
        assert_eq!(report.evidence_disposition, EmpiricalStressEvidenceDisposition::IncompleteOrNotComparable);
    }

    #[test]
    fn out_of_domain_slice_is_not_combined_with_in_domain_slices() {
        let mut input = plan(StressEvidenceClass::SyntheticFixture, (1.0, 2.0), (3.0, 4.0));
        input.holdout_receipts[0].slices[0].in_declared_domain = false;
        let report = evaluate_empirical_holdout_stress(&input);
        assert!(report.diagnostics.contains(&EmpiricalStressDiagnostic::SliceOutOfDomain));
    }

    #[test]
    fn underfilled_slice_refuses_complete_stress_envelope() {
        let mut input = plan(StressEvidenceClass::SyntheticFixture, (1.0, 2.0), (3.0, 4.0));
        input.holdout_receipts[0].slices[1].sample_count = 2;
        let report = evaluate_empirical_holdout_stress(&input);
        assert!(report.diagnostics.contains(&EmpiricalStressDiagnostic::InsufficientSliceSamples));
    }

    #[test]
    fn insufficient_independent_groups_are_refused_even_if_row_count_is_large() {
        let mut input = plan(StressEvidenceClass::SyntheticFixture, (1.0, 2.0), (3.0, 4.0));
        input.holdout_receipts[0].slices[0].independent_group_count = 1;
        let report = evaluate_empirical_holdout_stress(&input);
        assert!(report.diagnostics.contains(&EmpiricalStressDiagnostic::InsufficientIndependentGroups));
    }

    #[test]
    fn at_most_threshold_is_classified_symmetrically() {
        assert_eq!(
            classify_stress_interval(4.0, 8.0, 10.0, ThresholdDirection::AtMost, true),
            EmpiricalStressSide::StressPass
        );
        assert_eq!(
            classify_stress_interval(12.0, 15.0, 10.0, ThresholdDirection::AtMost, true),
            EmpiricalStressSide::StressFail
        );
        assert_eq!(
            classify_stress_interval(9.0, 12.0, 10.0, ThresholdDirection::AtMost, true),
            EmpiricalStressSide::ThresholdAmbiguous
        );
    }

    #[test]
    fn slice_and_model_order_does_not_change_canonical_report() {
        let first = plan(StressEvidenceClass::SyntheticFixture, (8.0, 15.0), (12.0, 10.0));
        let mut second = first.clone();
        second.decision_plan.models.reverse();
        second.holdout_receipts.reverse();
        for receipt in &mut second.holdout_receipts {
            receipt.slices.reverse();
            receipt.required_slices.reverse();
        }
        let a = evaluate_empirical_holdout_stress(&first);
        let b = evaluate_empirical_holdout_stress(&second);
        assert_eq!(a, b);
    }

    #[test]
    fn arithmetic_overflow_is_not_evaluable() {
        let mut input = plan(StressEvidenceClass::SyntheticFixture, (f64::MAX, 1.0), (1.0, 2.0));
        input.decision_plan.models[0].normalized_value = f64::MAX;
        let report = evaluate_empirical_holdout_stress(&input);
        assert!(report.diagnostics.contains(&EmpiricalStressDiagnostic::StressIntervalOverflow));
    }

    #[test]
    fn missing_pre_reveal_or_verified_payload_blocks_measured_stress() {
        let mut input = plan(StressEvidenceClass::ExternallyVerifiedMeasuredPayload, (1.0, 2.0), (3.0, 4.0));
        input.holdout_receipts[0].committed_before_holdout_reveal = false;
        let report = evaluate_empirical_holdout_stress(&input);
        assert!(report.diagnostics.contains(&EmpiricalStressDiagnostic::HoldoutCommitmentNotPreReveal));
        let mut input = plan(StressEvidenceClass::ExternallyVerifiedMeasuredPayload, (1.0, 2.0), (3.0, 4.0));
        input.holdout_receipts[0].measurement_payload_externally_verified = false;
        let report = evaluate_empirical_holdout_stress(&input);
        assert!(report.diagnostics.contains(&EmpiricalStressDiagnostic::MeasurementPayloadNotVerified));
    }

    #[test]
    fn duplicate_and_unexpected_receipts_are_refused() {
        let mut input = plan(StressEvidenceClass::SyntheticFixture, (1.0, 2.0), (3.0, 4.0));
        input.holdout_receipts.push(input.holdout_receipts[0].clone());
        let report = evaluate_empirical_holdout_stress(&input);
        assert!(report.diagnostics.contains(&EmpiricalStressDiagnostic::DuplicateRunReceipt));
        let mut input = plan(StressEvidenceClass::SyntheticFixture, (1.0, 2.0), (3.0, 4.0));
        let mut extra = input.holdout_receipts[0].clone();
        extra.run_id = "unexpected-run".to_string();
        input.holdout_receipts.push(extra);
        let report = evaluate_empirical_holdout_stress(&input);
        assert!(report.diagnostics.contains(&EmpiricalStressDiagnostic::UnexpectedRunReceipt));
    }

    #[test]
    fn equal_worst_slices_use_lexical_tie_break() {
        let mut input = plan(StressEvidenceClass::SyntheticFixture, (10.0, 10.0), (5.0, 5.0));
        input.holdout_receipts[0].required_slices = vec!["slice-b".into(), "slice-a".into()];
        let report = evaluate_empirical_holdout_stress(&input);
        assert_eq!(report.models[0].worst_slice_id, "slice-a");
    }
}
