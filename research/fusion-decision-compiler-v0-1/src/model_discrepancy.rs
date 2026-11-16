//! Domain-conditioned model-form discrepancy for fusion plant service.
//!
//! This module consumes a frozen model, independent measured calibration
//! residuals, an applicability receipt and a parametric lower-bound receipt.
//! It computes a split-conformal absolute-residual allowance and combines
//! coverage with a Bonferroni/union-bound allocation. It does not fit the
//! model, train the OOD detector, verify payload bytes or establish physics.

use std::collections::BTreeSet;

use super::cross_solver_comparison::ApplicabilityState;
use super::robust_net_export::ModelFormValidity;
use super::{EvidenceClass, ModelUse};
use super::plant_service_gate::{
    PowerBalanceProfile, PowerBalanceRun, PowerObservable,
};

pub const MODEL_DISCREPANCY_VERSION: &str = "fusion-model-discrepancy/0.1";
pub const COVERAGE_BASIS_POINTS: u32 = 10_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscrepancyCalibrationProfile {
    pub profile_id: String,
    pub profile_sha256: String,
    pub solver_id: String,
    pub solver_revision: String,
    pub validation_domain_id: String,
    pub validation_domain_sha256: String,
    pub regime_id: String,
    pub observable_id: String,
    pub quantity_ref: String,
    pub unit_ref: String,
    pub service_boundary_ref: String,
    pub averaging_window_ref: String,
    pub sign_convention_ref: String,
    pub allowed_geometry_sha256: Vec<String>,
    pub calibration_dataset_id: String,
    pub calibration_manifest_sha256: String,
    pub exchangeability_scope_ref: String,
    pub calibration_method_ref: String,
    pub discrepancy_coverage_basis_points: u32,
    pub required_total_coverage_basis_points: u32,
    pub minimum_calibration_samples: usize,
    pub maximum_discrepancy_mwe: f64,
    pub discrepancy_tolerance_inclusive: bool,
    /// Digest of the canonical evaluation request envelope, supplied by the
    /// evidence packager. This module validates its shape, not its payload.
    pub evaluation_envelope_ref: String,
    pub evaluation_envelope_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationExposure {
    pub used_for_model_fit: bool,
    pub used_for_normalization_fit: bool,
    pub used_for_feature_selection: bool,
    pub used_for_hyperparameter_selection: bool,
    pub used_for_model_selection: bool,
    pub used_for_policy_selection: bool,
}

impl CalibrationExposure {
    fn is_unexposed(&self) -> bool {
        !self.used_for_model_fit
            && !self.used_for_normalization_fit
            && !self.used_for_feature_selection
            && !self.used_for_hyperparameter_selection
            && !self.used_for_model_selection
            && !self.used_for_policy_selection
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationResidual {
    pub record_id: String,
    pub dataset_id: String,
    pub independence_cluster_id: String,
    pub episode_id: String,
    pub facility_id: String,
    pub case_id: String,
    pub operating_point_id: String,
    pub solver_id: String,
    pub model_revision: String,
    pub model_run_id: String,
    pub model_output_sha256: String,
    pub observation_id: String,
    pub observation_artifact_sha256: String,
    pub evidence_class: EvidenceClass,
    pub measurement_uncertainty_ref: String,
    pub validation_domain_id: String,
    pub validation_domain_sha256: String,
    pub regime_id: String,
    pub geometry_sha256: String,
    pub observable_id: String,
    pub quantity_ref: String,
    pub unit_ref: String,
    pub service_boundary_ref: String,
    pub averaging_window_ref: String,
    pub sign_convention_ref: String,
    /// Absolute measured prediction residual in the canonical unit.
    pub absolute_residual_mwe: f64,
    pub exposure: CalibrationExposure,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplicabilityReceipt {
    pub receipt_ref: String,
    pub receipt_sha256: String,
    pub evidence_ref: String,
    pub evidence_sha256: String,
    pub solver_id: String,
    pub solver_revision: String,
    pub run_id: String,
    pub input_sha256: String,
    pub output_sha256: String,
    pub candidate_revision: String,
    pub geometry_sha256: String,
    pub regime_id: String,
    pub validation_domain_id: String,
    pub validation_domain_sha256: String,
    pub requested_use: ModelUse,
    pub state: ApplicabilityState,
    pub evidence_class: EvidenceClass,
    pub method_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiscrepancyTarget {
    pub candidate_revision: String,
    pub input_sha256: String,
    pub geometry_sha256: String,
    pub regime_id: String,
    pub observable_id: String,
    pub quantity_ref: String,
    pub requested_use: ModelUse,
    pub point_estimate_mwe: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParametricLowerBoundReceipt {
    pub receipt_ref: String,
    pub receipt_sha256: String,
    pub evidence_ref: String,
    pub evidence_sha256: String,
    pub method_ref: String,
    pub assumption_manifest_sha256: String,
    pub run_id: String,
    pub input_sha256: String,
    pub output_sha256: String,
    pub schema_ref: String,
    pub schema_sha256: String,
    pub service_boundary_ref: String,
    pub unit_ref: String,
    pub averaging_window_ref: String,
    pub sign_convention_ref: String,
    pub lower_bound_mwe: f64,
    pub one_sided_coverage_basis_points: u32,
    pub coverage_scope_ref: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelDiscrepancyDisposition {
    QualifiedWithinDeclaredDomain,
    SyntheticOnly,
    OutOfDomain,
    ApplicabilityUnknown,
    MaterialDiscrepancy,
    InsufficientCalibration,
    CoverageAllocationInsufficient,
    NotEvaluable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelDiscrepancyDiagnostic {
    EmptyProfileIdentity,
    InvalidProfileDigest,
    EmptySolverIdentity,
    EmptyValidationDomain,
    InvalidValidationDomainDigest,
    EmptyRegime,
    EmptyObservable,
    EmptyQuantity,
    UnitMismatch,
    EmptyServiceBoundary,
    EmptyAveragingWindow,
    EmptySignConvention,
    EmptyCalibrationDataset,
    InvalidCalibrationManifestDigest,
    EmptyExchangeabilityScope,
    EmptyCalibrationMethod,
    InvalidCoverageBasisPoints,
    InvalidMinimumSampleCount,
    InvalidDiscrepancyTolerance,
    MissingEvaluationEnvelope,
    InvalidEvaluationEnvelopeDigest,
    UnsupportedSolver,
    SolverRevisionMismatch,
    SchemaReferenceMismatch,
    SchemaDigestMismatch,
    ServiceBoundaryMismatch,
    UnitMismatchWithRun,
    AveragingWindowMismatch,
    SignConventionMismatch,
    RunIdentityMismatch,
    OutputDigestMismatch,
    InvalidOutputDigest,
    CandidateRevisionMismatch,
    InputDigestMismatch,
    GeometryOutsideDomain,
    RegimeMismatch,
    ObservableMismatch,
    QuantityMismatch,
    MissingNetElectricObservable,
    PointEstimateMismatch,
    EmptyCalibrationSet,
    CalibrationSampleCountBelowMinimum,
    DuplicateCalibrationRecord,
    DuplicateEpisode,
    DuplicateCluster,
    DuplicateObservation,
    MissingResidualIdentity,
    InvalidResidualDigest,
    InvalidResidualValue,
    MissingMeasurementUncertainty,
    ResidualDomainMismatch,
    ResidualModelRevisionMismatch,
    ResidualSolverRunIdentityMissing,
    ResidualGeometryOutsideDomain,
    ResidualObservableMismatch,
    CalibrationExposureLeakage,
    SyntheticAndMeasuredResidualMix,
    NonMeasuredResidualEvidence,
    ApplicabilityReceiptIdentityMissing,
    ApplicabilityReceiptDigestInvalid,
    ApplicabilityEvidenceDigestInvalid,
    ApplicabilityRunMismatch,
    ApplicabilityInputMismatch,
    ApplicabilityOutputMismatch,
    ApplicabilityDomainMismatch,
    ApplicabilityModelMismatch,
    ApplicabilityGeometryMismatch,
    ApplicabilityRegimeMismatch,
    ApplicabilityUseMismatch,
    ApplicabilityEvidenceNotIndependent,
    ParametricReceiptMissing,
    ParametricReceiptIdentityMissing,
    ParametricReceiptDigestInvalid,
    ParametricEvidenceDigestInvalid,
    ParametricRunMismatch,
    ParametricInputMismatch,
    ParametricOutputMismatch,
    ParametricSchemaMismatch,
    ParametricBoundaryMismatch,
    ParametricUnitMismatch,
    ParametricTimeWindowMismatch,
    ParametricSignConventionMismatch,
    ParametricCoverageInvalid,
    ParametricCoverageScopeMissing,
    ParametricLowerBoundAbovePointEstimate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelDiscrepancyReport {
    pub contract_version: &'static str,
    pub profile_id: String,
    pub run_id: String,
    pub disposition: ModelDiscrepancyDisposition,
    pub model_form_validity: ModelFormValidity,
    pub calibration_sample_count: usize,
    pub conformal_rank: Option<usize>,
    pub discrepancy_bound_mwe: Option<f64>,
    pub parametric_lower_bound_mwe: Option<f64>,
    pub adjusted_lower_bound_mwe: Option<f64>,
    pub parametric_coverage_basis_points: Option<u32>,
    pub discrepancy_coverage_basis_points: u32,
    /// Conservative union-bound floor; no independence assumption is made.
    pub combined_coverage_floor_basis_points: Option<u32>,
    pub required_total_coverage_basis_points: u32,
    pub scalar_tolerance_mwe: f64,
    pub model_form_evidence_ref: String,
    pub model_form_evidence_sha256: String,
    pub evidence_refs: Vec<String>,
    pub diagnostics: Vec<ModelDiscrepancyDiagnostic>,
    pub claim_ceiling: &'static str,
}

/// Evaluate a domain-conditioned split-conformal residual allowance and
/// conservatively subtract it from a source-produced parametric lower bound.
///
/// The parametric and model-discrepancy coverages are combined using the
/// union bound: C_total >= C_parametric + C_discrepancy - 1. Their statistical
/// independence is neither required nor assumed.
///
/// The finite-sample conformal rank is calculated with integer basis points:
/// k = ceil((n + 1) * coverage_discrepancy). If k > n, no finite bound exists.
pub fn evaluate_model_discrepancy(
    profile: &DiscrepancyCalibrationProfile,
    target: &DiscrepancyTarget,
    power_profile: &PowerBalanceProfile,
    power_run: &PowerBalanceRun,
    residuals: &[CalibrationResidual],
    applicability: &ApplicabilityReceipt,
    parametric: Option<&ParametricLowerBoundReceipt>,
) -> ModelDiscrepancyReport {
    let mut diagnostics = validate_profile(profile);
    validate_target_and_run(
        profile,
        target,
        power_profile,
        power_run,
        &mut diagnostics,
    );
    validate_residuals(profile, target, residuals, &mut diagnostics);
    validate_applicability(profile, target, power_run, applicability, &mut diagnostics);

    let parametric = match parametric {
        Some(value) => {
            validate_parametric(
                profile,
                power_profile,
                power_run,
                target,
                value,
                &mut diagnostics,
            );
            Some(value)
        }
        None => {
            diagnostics.push(ModelDiscrepancyDiagnostic::ParametricReceiptMissing);
            None
        }
    };

    if !diagnostics.is_empty() {
        return report(
            profile,
            power_run,
            ModelDiscrepancyDisposition::NotEvaluable,
            ModelFormValidity::Unknown,
            residuals.len(),
            None,
            None,
            parametric,
            None,
            None,
            diagnostics,
            Vec::new(),
        );
    }

    match applicability.state {
        ApplicabilityState::OutOfDomain => {
            return report(
                profile,
                power_run,
                ModelDiscrepancyDisposition::OutOfDomain,
                ModelFormValidity::OutOfDomain,
                residuals.len(),
                None,
                None,
                parametric,
                None,
                None,
                Vec::new(),
                vec![
                    applicability.receipt_ref.clone(),
                    applicability.evidence_ref.clone(),
                ],
            );
        }
        ApplicabilityState::Unknown => {
            return report(
                profile,
                power_run,
                ModelDiscrepancyDisposition::ApplicabilityUnknown,
                ModelFormValidity::Unknown,
                residuals.len(),
                None,
                None,
                parametric,
                None,
                None,
                Vec::new(),
                vec![
                    applicability.receipt_ref.clone(),
                    applicability.evidence_ref.clone(),
                ],
            );
        }
        ApplicabilityState::InDeclaredDomain => {}
    }

    let rank = conformal_order_statistic_rank(
        residuals.len(),
        profile.discrepancy_coverage_basis_points,
    );
    if residuals.len() < profile.minimum_calibration_samples {
        return report(
            profile,
            power_run,
            ModelDiscrepancyDisposition::InsufficientCalibration,
            ModelFormValidity::Unknown,
            residuals.len(),
            Some(rank),
            None,
            parametric,
            None,
            None,
            Vec::new(),
            vec![
                applicability.receipt_ref.clone(),
                profile.calibration_manifest_sha256.clone(),
            ],
        );
    }
    if rank > residuals.len() {
        return report(
            profile,
            power_run,
            ModelDiscrepancyDisposition::InsufficientCalibration,
            ModelFormValidity::Unknown,
            residuals.len(),
            Some(rank),
            None,
            parametric,
            None,
            None,
            Vec::new(),
            vec![
                applicability.receipt_ref.clone(),
                profile.calibration_manifest_sha256.clone(),
            ],
        );
    }

    let mut sorted_residuals: Vec<f64> = residuals
        .iter()
        .map(|residual| residual.absolute_residual_mwe)
        .collect();
    sorted_residuals.sort_by(f64::total_cmp);
    let discrepancy_bound = sorted_residuals[rank - 1];

    let parametric = parametric.expect("validated parametric receipt is present");
    let combined_coverage = parametric
        .one_sided_coverage_basis_points
        .saturating_add(profile.discrepancy_coverage_basis_points)
        .saturating_sub(COVERAGE_BASIS_POINTS);

    let mut evidence_refs = vec![
        applicability.receipt_ref.clone(),
        applicability.evidence_ref.clone(),
        parametric.receipt_ref.clone(),
        parametric.evidence_ref.clone(),
        profile.calibration_manifest_sha256.clone(),
        profile.evaluation_envelope_ref.clone(),
    ];

    if combined_coverage < profile.required_total_coverage_basis_points {
        return report(
            profile,
            power_run,
            ModelDiscrepancyDisposition::CoverageAllocationInsufficient,
            ModelFormValidity::Unknown,
            residuals.len(),
            Some(rank),
            Some(discrepancy_bound),
            Some(parametric),
            Some(combined_coverage),
            None,
            Vec::new(),
            evidence_refs,
        );
    }

    let discrepancy_is_material = if profile.discrepancy_tolerance_inclusive {
        discrepancy_bound > profile.maximum_discrepancy_mwe
    } else {
        discrepancy_bound >= profile.maximum_discrepancy_mwe
    };
    let adjusted_lower = parametric.lower_bound_mwe - discrepancy_bound;

    if discrepancy_is_material {
        return report(
            profile,
            power_run,
            ModelDiscrepancyDisposition::MaterialDiscrepancy,
            ModelFormValidity::MaterialDiscrepancy,
            residuals.len(),
            Some(rank),
            Some(discrepancy_bound),
            Some(parametric),
            Some(combined_coverage),
            None,
            Vec::new(),
            evidence_refs,
        );
    }

    let disposition = if residuals
        .iter()
        .any(|residual| residual.evidence_class == EvidenceClass::SyntheticFixture)
    {
        ModelDiscrepancyDisposition::SyntheticOnly
    } else {
        ModelDiscrepancyDisposition::QualifiedWithinDeclaredDomain
    };
    let model_form_validity = if disposition == ModelDiscrepancyDisposition::SyntheticOnly {
        ModelFormValidity::Unknown
    } else {
        ModelFormValidity::WithinDeclaredDomain
    };

    evidence_refs.push(profile.evaluation_envelope_sha256.clone());
    report(
        profile,
        power_run,
        disposition,
        model_form_validity,
        residuals.len(),
        Some(rank),
        Some(discrepancy_bound),
        Some(parametric),
        Some(combined_coverage),
        Some(adjusted_lower),
        Vec::new(),
        evidence_refs,
    )
}

pub fn conformal_order_statistic_rank(sample_count: usize, coverage_basis_points: u32) -> usize {
    if coverage_basis_points == 0 || coverage_basis_points > COVERAGE_BASIS_POINTS {
        return usize::MAX;
    }
    let numerator = (sample_count as u128 + 1) * u128::from(coverage_basis_points);
    numerator.div_ceil(u128::from(COVERAGE_BASIS_POINTS)) as usize
}

fn validate_profile(profile: &DiscrepancyCalibrationProfile) -> Vec<ModelDiscrepancyDiagnostic> {
    let mut diagnostics = Vec::new();
    if profile.profile_id.trim().is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::EmptyProfileIdentity);
    }
    if !is_sha256(&profile.profile_sha256) {
        diagnostics.push(ModelDiscrepancyDiagnostic::InvalidProfileDigest);
    }
    if profile.solver_id.trim().is_empty() || profile.solver_revision.trim().is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::EmptySolverIdentity);
    }
    if profile.validation_domain_id.trim().is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::EmptyValidationDomain);
    }
    if !is_sha256(&profile.validation_domain_sha256) {
        diagnostics.push(ModelDiscrepancyDiagnostic::InvalidValidationDomainDigest);
    }
    if profile.regime_id.trim().is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::EmptyRegime);
    }
    if profile.observable_id.trim().is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::EmptyObservable);
    }
    if profile.quantity_ref.trim().is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::EmptyQuantity);
    }
    if profile.unit_ref != "MWe" {
        diagnostics.push(ModelDiscrepancyDiagnostic::UnitMismatch);
    }
    if profile.service_boundary_ref.trim().is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::EmptyServiceBoundary);
    }
    if profile.averaging_window_ref.trim().is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::EmptyAveragingWindow);
    }
    if profile.sign_convention_ref.trim().is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::EmptySignConvention);
    }
    if profile.calibration_dataset_id.trim().is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::EmptyCalibrationDataset);
    }
    if !is_sha256(&profile.calibration_manifest_sha256) {
        diagnostics.push(ModelDiscrepancyDiagnostic::InvalidCalibrationManifestDigest);
    }
    if profile.exchangeability_scope_ref.trim().is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::EmptyExchangeabilityScope);
    }
    if profile.calibration_method_ref != "split-conformal-absolute-residual-v1" {
        diagnostics.push(ModelDiscrepancyDiagnostic::EmptyCalibrationMethod);
    }
    if profile.discrepancy_coverage_basis_points == 0
        || profile.discrepancy_coverage_basis_points > COVERAGE_BASIS_POINTS
        || profile.required_total_coverage_basis_points == 0
        || profile.required_total_coverage_basis_points > COVERAGE_BASIS_POINTS
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::InvalidCoverageBasisPoints);
    }
    if profile.minimum_calibration_samples == 0 {
        diagnostics.push(ModelDiscrepancyDiagnostic::InvalidMinimumSampleCount);
    }
    if !profile.maximum_discrepancy_mwe.is_finite() || profile.maximum_discrepancy_mwe < 0.0 {
        diagnostics.push(ModelDiscrepancyDiagnostic::InvalidDiscrepancyTolerance);
    }
    if profile.evaluation_envelope_ref.trim().is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::MissingEvaluationEnvelope);
    }
    if !is_sha256(&profile.evaluation_envelope_sha256) {
        diagnostics.push(ModelDiscrepancyDiagnostic::InvalidEvaluationEnvelopeDigest);
    }
    validate_digest_list(
        &profile.allowed_geometry_sha256,
        ModelDiscrepancyDiagnostic::InvalidValidationDomainDigest,
        &mut diagnostics,
    );
    if profile.allowed_geometry_sha256.is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::GeometryOutsideDomain);
    }
    diagnostics
}

fn validate_target_and_run(
    profile: &DiscrepancyCalibrationProfile,
    target: &DiscrepancyTarget,
    power_profile: &PowerBalanceProfile,
    run: &PowerBalanceRun,
    diagnostics: &mut Vec<ModelDiscrepancyDiagnostic>,
) {
    if run.solver_id != profile.solver_id {
        diagnostics.push(ModelDiscrepancyDiagnostic::UnsupportedSolver);
    }
    if run.solver_revision != profile.solver_revision {
        diagnostics.push(ModelDiscrepancyDiagnostic::SolverRevisionMismatch);
    }
    if run.schema_ref != power_profile.schema_ref {
        diagnostics.push(ModelDiscrepancyDiagnostic::SchemaReferenceMismatch);
    }
    if run.schema_sha256 != power_profile.schema_sha256 || !is_sha256(&run.schema_sha256) {
        diagnostics.push(ModelDiscrepancyDiagnostic::SchemaDigestMismatch);
    }
    if run.service_boundary_ref != profile.service_boundary_ref
        || run.service_boundary_ref != power_profile.service_boundary_ref
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::ServiceBoundaryMismatch);
    }
    if run.unit_ref != profile.unit_ref || run.unit_ref != power_profile.unit_ref {
        diagnostics.push(ModelDiscrepancyDiagnostic::UnitMismatchWithRun);
    }
    if run.averaging_window_ref != profile.averaging_window_ref
        || run.averaging_window_ref != power_profile.averaging_window_ref
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::AveragingWindowMismatch);
    }
    if run.sign_convention_ref != profile.sign_convention_ref
        || run.sign_convention_ref != power_profile.sign_convention_ref
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::SignConventionMismatch);
    }
    if run.run_id.trim().is_empty() || !is_sha256(&run.output_sha256) {
        diagnostics.push(ModelDiscrepancyDiagnostic::InvalidOutputDigest);
    }
    if target.candidate_revision != run.candidate_revision {
        diagnostics.push(ModelDiscrepancyDiagnostic::CandidateRevisionMismatch);
    }
    if target.input_sha256.trim().is_empty() || !is_sha256(&target.input_sha256) {
        diagnostics.push(ModelDiscrepancyDiagnostic::InputDigestMismatch);
    }
    if target.geometry_sha256.trim().is_empty()
        || !profile.allowed_geometry_sha256.contains(&target.geometry_sha256)
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::GeometryOutsideDomain);
    }
    if target.regime_id != profile.regime_id {
        diagnostics.push(ModelDiscrepancyDiagnostic::RegimeMismatch);
    }
    if target.observable_id != profile.observable_id {
        diagnostics.push(ModelDiscrepancyDiagnostic::ObservableMismatch);
    }
    if target.quantity_ref != profile.quantity_ref {
        diagnostics.push(ModelDiscrepancyDiagnostic::QuantityMismatch);
    }

    let reported_net = run
        .terms
        .iter()
        .find(|term| term.observable == PowerObservable::NetElectric)
        .map(|term| term.value_mw);
    match reported_net {
        None => diagnostics.push(ModelDiscrepancyDiagnostic::MissingNetElectricObservable),
        Some(value) if !value.is_finite() => {
            diagnostics.push(ModelDiscrepancyDiagnostic::PointEstimateMismatch);
        }
        Some(value) if (value - target.point_estimate_mwe).abs() > 1e-9 => {
            diagnostics.push(ModelDiscrepancyDiagnostic::PointEstimateMismatch);
        }
        Some(_) => {}
    }
    if !target.point_estimate_mwe.is_finite() {
        diagnostics.push(ModelDiscrepancyDiagnostic::PointEstimateMismatch);
    }
}

fn validate_residuals(
    profile: &DiscrepancyCalibrationProfile,
    target: &DiscrepancyTarget,
    residuals: &[CalibrationResidual],
    diagnostics: &mut Vec<ModelDiscrepancyDiagnostic>,
) {
    if residuals.is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::EmptyCalibrationSet);
        return;
    }
    let mut record_ids = BTreeSet::new();
    let mut episode_ids = BTreeSet::new();
    let mut cluster_ids = BTreeSet::new();
    let mut observation_ids = BTreeSet::new();
    let mut has_synthetic = false;
    let mut has_measured = false;

    for residual in residuals {
        if residual.record_id.trim().is_empty()
            || residual.dataset_id.trim().is_empty()
            || residual.independence_cluster_id.trim().is_empty()
            || residual.episode_id.trim().is_empty()
            || residual.facility_id.trim().is_empty()
            || residual.case_id.trim().is_empty()
            || residual.operating_point_id.trim().is_empty()
            || residual.observation_id.trim().is_empty()
        {
            diagnostics.push(ModelDiscrepancyDiagnostic::MissingResidualIdentity);
        }
        if !record_ids.insert(residual.record_id.as_str()) {
            diagnostics.push(ModelDiscrepancyDiagnostic::DuplicateCalibrationRecord);
        }
        if !episode_ids.insert(residual.episode_id.as_str()) {
            diagnostics.push(ModelDiscrepancyDiagnostic::DuplicateEpisode);
        }
        if !cluster_ids.insert(residual.independence_cluster_id.as_str()) {
            diagnostics.push(ModelDiscrepancyDiagnostic::DuplicateCluster);
        }
        if !observation_ids.insert(residual.observation_id.as_str()) {
            diagnostics.push(ModelDiscrepancyDiagnostic::DuplicateObservation);
        }

        if !is_sha256(&residual.model_output_sha256)
            || !is_sha256(&residual.observation_artifact_sha256)
            || !is_sha256(&residual.geometry_sha256)
        {
            diagnostics.push(ModelDiscrepancyDiagnostic::InvalidResidualDigest);
        }
        if !residual.absolute_residual_mwe.is_finite() || residual.absolute_residual_mwe < 0.0 {
            diagnostics.push(ModelDiscrepancyDiagnostic::InvalidResidualValue);
        }
        if residual.measurement_uncertainty_ref.trim().is_empty() {
            diagnostics.push(ModelDiscrepancyDiagnostic::MissingMeasurementUncertainty);
        }
        if residual.validation_domain_id != profile.validation_domain_id
            || residual.validation_domain_sha256 != profile.validation_domain_sha256
            || residual.regime_id != profile.regime_id
            || residual.observable_id != profile.observable_id
            || residual.quantity_ref != profile.quantity_ref
            || residual.unit_ref != profile.unit_ref
            || residual.service_boundary_ref != profile.service_boundary_ref
            || residual.averaging_window_ref != profile.averaging_window_ref
            || residual.sign_convention_ref != profile.sign_convention_ref
            || residual.dataset_id != profile.calibration_dataset_id
        {
            diagnostics.push(ModelDiscrepancyDiagnostic::ResidualDomainMismatch);
        }
        if residual.solver_id != profile.solver_id {
            diagnostics.push(ModelDiscrepancyDiagnostic::ResidualModelRevisionMismatch);
        }
        if residual.model_revision != profile.solver_revision {
            diagnostics.push(ModelDiscrepancyDiagnostic::ResidualModelRevisionMismatch);
        }
        if residual.model_run_id.trim().is_empty() {
            diagnostics.push(ModelDiscrepancyDiagnostic::ResidualSolverRunIdentityMissing);
        }
        if !profile
            .allowed_geometry_sha256
            .contains(&residual.geometry_sha256)
        {
            diagnostics.push(ModelDiscrepancyDiagnostic::ResidualGeometryOutsideDomain);
        }
        if target.observable_id != residual.observable_id {
            diagnostics.push(ModelDiscrepancyDiagnostic::ResidualObservableMismatch);
        }
        if !residual.exposure.is_unexposed() {
            diagnostics.push(ModelDiscrepancyDiagnostic::CalibrationExposureLeakage);
        }
        match residual.evidence_class {
            EvidenceClass::SyntheticFixture => has_synthetic = true,
            EvidenceClass::MeasuredObservation | EvidenceClass::IndependentVerification => {
                has_measured = true;
            }
            _ => diagnostics.push(ModelDiscrepancyDiagnostic::NonMeasuredResidualEvidence),
        }
    }

    if has_synthetic && has_measured {
        diagnostics.push(ModelDiscrepancyDiagnostic::SyntheticAndMeasuredResidualMix);
    }
}

fn validate_applicability(
    profile: &DiscrepancyCalibrationProfile,
    target: &DiscrepancyTarget,
    run: &PowerBalanceRun,
    receipt: &ApplicabilityReceipt,
    diagnostics: &mut Vec<ModelDiscrepancyDiagnostic>,
) {
    if receipt.receipt_ref.trim().is_empty()
        || receipt.evidence_ref.trim().is_empty()
        || receipt.method_ref.trim().is_empty()
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::ApplicabilityReceiptIdentityMissing);
    }
    if !is_sha256(&receipt.receipt_sha256) {
        diagnostics.push(ModelDiscrepancyDiagnostic::ApplicabilityReceiptDigestInvalid);
    }
    if !is_sha256(&receipt.evidence_sha256) {
        diagnostics.push(ModelDiscrepancyDiagnostic::ApplicabilityEvidenceDigestInvalid);
    }
    if receipt.solver_id != profile.solver_id
        || receipt.solver_revision != profile.solver_revision
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::ApplicabilityModelMismatch);
    }
    if receipt.run_id != run.run_id {
        diagnostics.push(ModelDiscrepancyDiagnostic::ApplicabilityRunMismatch);
    }
    if receipt.input_sha256 != target.input_sha256 {
        diagnostics.push(ModelDiscrepancyDiagnostic::ApplicabilityInputMismatch);
    }
    if receipt.output_sha256 != run.output_sha256 {
        diagnostics.push(ModelDiscrepancyDiagnostic::ApplicabilityOutputMismatch);
    }
    if receipt.candidate_revision != target.candidate_revision {
        diagnostics.push(ModelDiscrepancyDiagnostic::CandidateRevisionMismatch);
    }
    if receipt.geometry_sha256 != target.geometry_sha256 {
        diagnostics.push(ModelDiscrepancyDiagnostic::ApplicabilityGeometryMismatch);
    }
    if receipt.regime_id != target.regime_id {
        diagnostics.push(ModelDiscrepancyDiagnostic::ApplicabilityRegimeMismatch);
    }
    if receipt.validation_domain_id != profile.validation_domain_id
        || receipt.validation_domain_sha256 != profile.validation_domain_sha256
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::ApplicabilityDomainMismatch);
    }
    if receipt.requested_use != target.requested_use {
        diagnostics.push(ModelDiscrepancyDiagnostic::ApplicabilityUseMismatch);
    }
    if receipt.state == ApplicabilityState::InDeclaredDomain
        && receipt.evidence_class != EvidenceClass::IndependentVerification
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::ApplicabilityEvidenceNotIndependent);
    }
}

fn validate_parametric(
    profile: &DiscrepancyCalibrationProfile,
    power_profile: &PowerBalanceProfile,
    run: &PowerBalanceRun,
    target: &DiscrepancyTarget,
    receipt: &ParametricLowerBoundReceipt,
    diagnostics: &mut Vec<ModelDiscrepancyDiagnostic>,
) {
    if receipt.receipt_ref.trim().is_empty()
        || receipt.evidence_ref.trim().is_empty()
        || receipt.method_ref.trim().is_empty()
        || receipt.coverage_scope_ref.trim().is_empty()
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricReceiptIdentityMissing);
    }
    if !is_sha256(&receipt.receipt_sha256) {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricReceiptDigestInvalid);
    }
    if !is_sha256(&receipt.evidence_sha256)
        || !is_sha256(&receipt.assumption_manifest_sha256)
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricEvidenceDigestInvalid);
    }
    if receipt.run_id != run.run_id {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricRunMismatch);
    }
    if receipt.input_sha256 != target.input_sha256 {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricInputMismatch);
    }
    if receipt.output_sha256 != run.output_sha256 {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricOutputMismatch);
    }
    if receipt.schema_ref != run.schema_ref
        || receipt.schema_ref != power_profile.schema_ref
        || receipt.schema_sha256 != run.schema_sha256
        || receipt.schema_sha256 != power_profile.schema_sha256
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricSchemaMismatch);
    }
    if receipt.service_boundary_ref != profile.service_boundary_ref
        || receipt.service_boundary_ref != run.service_boundary_ref
        || receipt.service_boundary_ref != power_profile.service_boundary_ref
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricBoundaryMismatch);
    }
    if receipt.unit_ref != profile.unit_ref || receipt.unit_ref != run.unit_ref {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricUnitMismatch);
    }
    if receipt.averaging_window_ref != profile.averaging_window_ref
        || receipt.averaging_window_ref != run.averaging_window_ref
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricTimeWindowMismatch);
    }
    if receipt.sign_convention_ref != profile.sign_convention_ref
        || receipt.sign_convention_ref != run.sign_convention_ref
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricSignConventionMismatch);
    }
    if receipt.one_sided_coverage_basis_points == 0
        || receipt.one_sided_coverage_basis_points > COVERAGE_BASIS_POINTS
        || !receipt.lower_bound_mwe.is_finite()
    {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricCoverageInvalid);
    }
    if receipt.coverage_scope_ref.trim().is_empty() {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricCoverageScopeMissing);
    }
    if receipt.lower_bound_mwe > target.point_estimate_mwe + 1e-9 {
        diagnostics.push(ModelDiscrepancyDiagnostic::ParametricLowerBoundAbovePointEstimate);
    }
}

#[allow(clippy::too_many_arguments)]
fn report(
    profile: &DiscrepancyCalibrationProfile,
    run: &PowerBalanceRun,
    disposition: ModelDiscrepancyDisposition,
    model_form_validity: ModelFormValidity,
    calibration_sample_count: usize,
    conformal_rank: Option<usize>,
    discrepancy_bound_mwe: Option<f64>,
    parametric: Option<&ParametricLowerBoundReceipt>,
    combined_coverage_floor_basis_points: Option<u32>,
    adjusted_lower_bound_mwe: Option<f64>,
    diagnostics: Vec<ModelDiscrepancyDiagnostic>,
    evidence_refs: Vec<String>,
) -> ModelDiscrepancyReport {
    ModelDiscrepancyReport {
        contract_version: MODEL_DISCREPANCY_VERSION,
        profile_id: profile.profile_id.clone(),
        run_id: run.run_id.clone(),
        disposition,
        model_form_validity,
        calibration_sample_count,
        conformal_rank,
        discrepancy_bound_mwe,
        parametric_lower_bound_mwe: parametric.map(|receipt| receipt.lower_bound_mwe),
        adjusted_lower_bound_mwe,
        parametric_coverage_basis_points: parametric
            .map(|receipt| receipt.one_sided_coverage_basis_points),
        discrepancy_coverage_basis_points: profile.discrepancy_coverage_basis_points,
        combined_coverage_floor_basis_points,
        required_total_coverage_basis_points: profile.required_total_coverage_basis_points,
        scalar_tolerance_mwe: profile.maximum_discrepancy_mwe,
        model_form_evidence_ref: profile.evaluation_envelope_ref.clone(),
        model_form_evidence_sha256: profile.evaluation_envelope_sha256.clone(),
        evidence_refs,
        diagnostics,
        claim_ceiling: "domain-conditioned residual envelope and conservative coverage accounting only",
    }
}

fn validate_digest_list(
    values: &[String],
    diagnostic: ModelDiscrepancyDiagnostic,
    diagnostics: &mut Vec<ModelDiscrepancyDiagnostic>,
) {
    if values.iter().any(|value| !is_sha256(value)) {
        diagnostics.push(diagnostic);
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plant_service_gate::{
        PowerBalanceProfile, PowerBalanceRun, PowerTermEvidence,
    };

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn power_profile() -> PowerBalanceProfile {
        PowerBalanceProfile {
            schema_ref: "fusion-power-schema-v1".to_string(),
            schema_sha256: sha('a'),
            service_boundary_ref: "plant-net-export-v1".to_string(),
            unit_ref: "MWe".to_string(),
            averaging_window_ref: "steady-state-60s-v1".to_string(),
            sign_convention_ref: "net-export-positive-v1".to_string(),
            absolute_balance_tolerance_mw: 0.001,
            required_net_export_reserve_mw: 50.0,
            reserve_threshold_inclusive: true,
        }
    }

    fn power_run() -> PowerBalanceRun {
        let profile = power_profile();
        PowerBalanceRun {
            run_id: "fusion-net-run-1".to_string(),
            solver_id: "PROCESS".to_string(),
            solver_revision: "process-revision-r1".to_string(),
            candidate_revision: "candidate-r1".to_string(),
            output_sha256: sha('b'),
            schema_ref: profile.schema_ref,
            schema_sha256: profile.schema_sha256,
            service_boundary_ref: profile.service_boundary_ref,
            unit_ref: profile.unit_ref,
            averaging_window_ref: profile.averaging_window_ref,
            sign_convention_ref: profile.sign_convention_ref,
            receipt_ref: "process-run-receipt-1".to_string(),
            terms: vec![PowerTermEvidence {
                observable: PowerObservable::NetElectric,
                value_mw: 150.0,
                source_locator: "outputs.P_elec".to_string(),
                evidence_ref: "net-electric-output-evidence".to_string(),
            }],
        }
    }

    fn discrepancy_profile() -> DiscrepancyCalibrationProfile {
        let profile = power_profile();
        DiscrepancyCalibrationProfile {
            profile_id: "net-power-discrepancy-v1".to_string(),
            profile_sha256: sha('c'),
            solver_id: "PROCESS".to_string(),
            solver_revision: "process-revision-r1".to_string(),
            validation_domain_id: "spherical-tokamak-net-power-v1".to_string(),
            validation_domain_sha256: sha('d'),
            regime_id: "steady-state-spherical-tokamak".to_string(),
            observable_id: "P_elec".to_string(),
            quantity_ref: "net-electric-output".to_string(),
            unit_ref: "MWe".to_string(),
            service_boundary_ref: profile.service_boundary_ref,
            averaging_window_ref: profile.averaging_window_ref,
            sign_convention_ref: profile.sign_convention_ref,
            allowed_geometry_sha256: vec![sha('e'), sha('f')],
            calibration_dataset_id: "process-saras-heldout-calibration-v1".to_string(),
            calibration_manifest_sha256: sha('g'),
            exchangeability_scope_ref: "one-independent-residual-per-shot-in-domain-v1".to_string(),
            calibration_method_ref: "split-conformal-absolute-residual-v1".to_string(),
            discrepancy_coverage_basis_points: 9750,
            required_total_coverage_basis_points: 9500,
            minimum_calibration_samples: 19,
            maximum_discrepancy_mwe: 25.0,
            discrepancy_tolerance_inclusive: true,
            evaluation_envelope_ref: "model-form-evaluation-envelope-1".to_string(),
            evaluation_envelope_sha256: sha('h'),
        }
    }

    fn target() -> DiscrepancyTarget {
        DiscrepancyTarget {
            candidate_revision: "candidate-r1".to_string(),
            input_sha256: sha('i'),
            geometry_sha256: sha('e'),
            regime_id: "steady-state-spherical-tokamak".to_string(),
            observable_id: "P_elec".to_string(),
            quantity_ref: "net-electric-output".to_string(),
            requested_use: ModelUse::EngineeringScreening,
            point_estimate_mwe: 150.0,
        }
    }

    fn residual(index: usize, value: f64, evidence_class: EvidenceClass) -> CalibrationResidual {
        CalibrationResidual {
            record_id: format!("residual-{index}"),
            dataset_id: "process-saras-heldout-calibration-v1".to_string(),
            independence_cluster_id: format!("cluster-{index}"),
            episode_id: format!("shot-{index}"),
            facility_id: "facility-reference".to_string(),
            case_id: format!("case-{index}"),
            operating_point_id: format!("op-{index}"),
            solver_id: "PROCESS".to_string(),
            model_revision: "process-revision-r1".to_string(),
            model_run_id: format!("calibration-run-{index}"),
            model_output_sha256: sha('j'),
            observation_id: format!("measurement-{index}"),
            observation_artifact_sha256: sha('k'),
            evidence_class,
            measurement_uncertainty_ref: format!("metrology-uncertainty-{index}"),
            validation_domain_id: "spherical-tokamak-net-power-v1".to_string(),
            validation_domain_sha256: sha('d'),
            regime_id: "steady-state-spherical-tokamak".to_string(),
            geometry_sha256: if index % 2 == 0 { sha('e') } else { sha('f') },
            observable_id: "P_elec".to_string(),
            quantity_ref: "net-electric-output".to_string(),
            unit_ref: "MWe".to_string(),
            service_boundary_ref: "plant-net-export-v1".to_string(),
            averaging_window_ref: "steady-state-60s-v1".to_string(),
            sign_convention_ref: "net-export-positive-v1".to_string(),
            absolute_residual_mwe: value,
            exposure: CalibrationExposure {
                used_for_model_fit: false,
                used_for_normalization_fit: false,
                used_for_feature_selection: false,
                used_for_hyperparameter_selection: false,
                used_for_model_selection: false,
                used_for_policy_selection: false,
            },
        }
    }

    fn applicability() -> ApplicabilityReceipt {
        ApplicabilityReceipt {
            receipt_ref: "applicability-receipt-1".to_string(),
            receipt_sha256: sha('l'),
            evidence_ref: "independent-applicability-assessment-1".to_string(),
            evidence_sha256: sha('m'),
            solver_id: "PROCESS".to_string(),
            solver_revision: "process-revision-r1".to_string(),
            run_id: "fusion-net-run-1".to_string(),
            input_sha256: sha('i'),
            output_sha256: sha('b'),
            candidate_revision: "candidate-r1".to_string(),
            geometry_sha256: sha('e'),
            regime_id: "steady-state-spherical-tokamak".to_string(),
            validation_domain_id: "spherical-tokamak-net-power-v1".to_string(),
            validation_domain_sha256: sha('d'),
            requested_use: ModelUse::EngineeringScreening,
            state: ApplicabilityState::InDeclaredDomain,
            evidence_class: EvidenceClass::IndependentVerification,
            method_ref: "domain-distance-policy-v1".to_string(),
        }
    }

    fn parametric() -> ParametricLowerBoundReceipt {
        let power = power_profile();
        ParametricLowerBoundReceipt {
            receipt_ref: "parametric-lower-bound-1".to_string(),
            receipt_sha256: sha('n'),
            evidence_ref: "symtropy-net-power-uq-evidence-1".to_string(),
            evidence_sha256: sha('o'),
            method_ref: "joint-covariance-linear-contrast-v1".to_string(),
            assumption_manifest_sha256: sha('p'),
            run_id: "fusion-net-run-1".to_string(),
            input_sha256: sha('i'),
            output_sha256: sha('b'),
            schema_ref: power.schema_ref,
            schema_sha256: power.schema_sha256,
            service_boundary_ref: power.service_boundary_ref,
            unit_ref: power.unit_ref,
            averaging_window_ref: power.averaging_window_ref,
            sign_convention_ref: power.sign_convention_ref,
            lower_bound_mwe: 120.0,
            one_sided_coverage_basis_points: 9750,
            coverage_scope_ref: "single-run-net-power-contrast-v1".to_string(),
        }
    }

    fn evaluate(
        profile: &DiscrepancyCalibrationProfile,
        target: &DiscrepancyTarget,
        residuals: &[CalibrationResidual],
        applicability: &ApplicabilityReceipt,
        parametric: Option<&ParametricLowerBoundReceipt>,
    ) -> ModelDiscrepancyReport {
        evaluate_model_discrepancy(
            profile,
            target,
            &power_profile(),
            &power_run(),
            residuals,
            applicability,
            parametric,
        )
    }

    fn measured_residuals(count: usize) -> Vec<CalibrationResidual> {
        (1..=count)
            .map(|index| residual(index, index as f64 / 2.0, EvidenceClass::MeasuredObservation))
            .collect()
    }

    #[test]
    fn split_conformal_95_percent_rank_for_19_samples_is_19() {
        assert_eq!(conformal_order_statistic_rank(19, 9500), 19);
    }

    #[test]
    fn split_conformal_95_percent_with_18_samples_has_no_finite_rank() {
        assert_eq!(conformal_order_statistic_rank(18, 9500), 19);
        assert!(conformal_order_statistic_rank(18, 9500) > 18);
    }

    #[test]
    fn split_conformal_97_5_percent_rank_for_39_samples_is_39() {
        assert_eq!(conformal_order_statistic_rank(39, 9750), 39);
    }

    #[test]
    fn measured_calibration_residuals_produce_bonferroni_adjusted_lower_bound() {
        let profile = discrepancy_profile();
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(39),
            &applicability(),
            Some(&parametric()),
        );
        assert_eq!(
            report.disposition,
            ModelDiscrepancyDisposition::QualifiedWithinDeclaredDomain
        );
        assert_eq!(report.conformal_rank, Some(39));
        assert_eq!(report.discrepancy_bound_mwe, Some(19.5));
        assert_eq!(report.combined_coverage_floor_basis_points, Some(9500));
        assert_eq!(report.adjusted_lower_bound_mwe, Some(100.5));
        assert_eq!(
            report.model_form_validity,
            ModelFormValidity::WithinDeclaredDomain
        );
    }

    #[test]
    fn synthetic_calibration_fixture_never_qualifies_model_form() {
        let profile = discrepancy_profile();
        let residuals: Vec<_> = (1..=39)
            .map(|index| residual(index, index as f64 / 2.0, EvidenceClass::SyntheticFixture))
            .collect();
        let report = evaluate(
            &profile,
            &target(),
            &residuals,
            &applicability(),
            Some(&parametric()),
        );
        assert_eq!(report.disposition, ModelDiscrepancyDisposition::SyntheticOnly);
        assert_eq!(report.model_form_validity, ModelFormValidity::Unknown);
        assert_eq!(report.adjusted_lower_bound_mwe, Some(100.5));
    }

    #[test]
    fn out_of_domain_applicability_refuses_adjusted_lower_bound() {
        let profile = discrepancy_profile();
        let mut applicability = applicability();
        applicability.state = ApplicabilityState::OutOfDomain;
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(39),
            &applicability,
            Some(&parametric()),
        );
        assert_eq!(report.disposition, ModelDiscrepancyDisposition::OutOfDomain);
        assert_eq!(report.model_form_validity, ModelFormValidity::OutOfDomain);
        assert_eq!(report.adjusted_lower_bound_mwe, None);
    }

    #[test]
    fn unknown_applicability_refuses_adjusted_lower_bound() {
        let profile = discrepancy_profile();
        let mut applicability = applicability();
        applicability.state = ApplicabilityState::Unknown;
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(39),
            &applicability,
            Some(&parametric()),
        );
        assert_eq!(report.disposition, ModelDiscrepancyDisposition::ApplicabilityUnknown);
        assert_eq!(report.adjusted_lower_bound_mwe, None);
    }

    #[test]
    fn in_domain_claim_requires_independent_applicability_evidence() {
        let profile = discrepancy_profile();
        let mut applicability = applicability();
        applicability.evidence_class = EvidenceClass::ModelOutput;
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(39),
            &applicability,
            Some(&parametric()),
        );
        assert_eq!(report.disposition, ModelDiscrepancyDisposition::NotEvaluable);
        assert!(report
            .diagnostics
            .contains(&ModelDiscrepancyDiagnostic::ApplicabilityEvidenceNotIndependent));
    }

    #[test]
    fn inadequate_finite_sample_rank_is_insufficient_calibration() {
        let profile = discrepancy_profile();
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(18),
            &applicability(),
            Some(&parametric()),
        );
        assert_eq!(report.disposition, ModelDiscrepancyDisposition::InsufficientCalibration);
        assert_eq!(report.conformal_rank, Some(19));
        assert_eq!(report.discrepancy_bound_mwe, None);
    }

    #[test]
    fn parametric_95_plus_discrepancy_95_does_not_meet_95_combined_target() {
        let profile = discrepancy_profile();
        let mut parametric = parametric();
        parametric.one_sided_coverage_basis_points = 9500;
        let mut profile = profile;
        profile.discrepancy_coverage_basis_points = 9500;
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(39),
            &applicability(),
            Some(&parametric),
        );
        assert_eq!(
            report.disposition,
            ModelDiscrepancyDisposition::CoverageAllocationInsufficient
        );
        assert_eq!(report.combined_coverage_floor_basis_points, Some(9000));
        assert_eq!(report.adjusted_lower_bound_mwe, None);
    }

    #[test]
    fn material_discrepancy_blocks_even_when_coverage_is_sufficient() {
        let mut profile = discrepancy_profile();
        profile.maximum_discrepancy_mwe = 10.0;
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(39),
            &applicability(),
            Some(&parametric()),
        );
        assert_eq!(report.disposition, ModelDiscrepancyDisposition::MaterialDiscrepancy);
        assert_eq!(report.model_form_validity, ModelFormValidity::MaterialDiscrepancy);
        assert_eq!(report.adjusted_lower_bound_mwe, None);
    }

    #[test]
    fn calibration_episode_reuse_is_rejected() {
        let profile = discrepancy_profile();
        let mut residuals = measured_residuals(39);
        residuals[1].episode_id = residuals[0].episode_id.clone();
        let report = evaluate(
            &profile,
            &target(),
            &residuals,
            &applicability(),
            Some(&parametric()),
        );
        assert_eq!(report.disposition, ModelDiscrepancyDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&ModelDiscrepancyDiagnostic::DuplicateEpisode));
    }

    #[test]
    fn calibration_cluster_reuse_is_rejected() {
        let profile = discrepancy_profile();
        let mut residuals = measured_residuals(39);
        residuals[1].independence_cluster_id = residuals[0].independence_cluster_id.clone();
        let report = evaluate(
            &profile,
            &target(),
            &residuals,
            &applicability(),
            Some(&parametric()),
        );
        assert!(report.diagnostics.contains(&ModelDiscrepancyDiagnostic::DuplicateCluster));
    }

    #[test]
    fn calibration_training_or_selection_leakage_is_rejected() {
        let profile = discrepancy_profile();
        let mut residuals = measured_residuals(39);
        residuals[0].exposure.used_for_model_selection = true;
        let report = evaluate(
            &profile,
            &target(),
            &residuals,
            &applicability(),
            Some(&parametric()),
        );
        assert!(report
            .diagnostics
            .contains(&ModelDiscrepancyDiagnostic::CalibrationExposureLeakage));
    }

    #[test]
    fn synthetic_and_measured_residuals_cannot_be_mixed() {
        let profile = discrepancy_profile();
        let mut residuals = measured_residuals(39);
        residuals[0].evidence_class = EvidenceClass::SyntheticFixture;
        let report = evaluate(
            &profile,
            &target(),
            &residuals,
            &applicability(),
            Some(&parametric()),
        );
        assert!(report
            .diagnostics
            .contains(&ModelDiscrepancyDiagnostic::SyntheticAndMeasuredResidualMix));
    }

    #[test]
    fn solver_revision_mismatch_is_rejected() {
        let mut profile = discrepancy_profile();
        profile.solver_revision = "other-revision".to_string();
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(39),
            &applicability(),
            Some(&parametric()),
        );
        assert!(report
            .diagnostics
            .contains(&ModelDiscrepancyDiagnostic::SolverRevisionMismatch));
    }

    #[test]
    fn applicability_receipt_must_bind_exact_run_input_and_output() {
        let profile = discrepancy_profile();
        let mut applicability = applicability();
        applicability.output_sha256 = sha('z');
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(39),
            &applicability,
            Some(&parametric()),
        );
        assert!(report
            .diagnostics
            .contains(&ModelDiscrepancyDiagnostic::ApplicabilityOutputMismatch));
    }

    #[test]
    fn parametric_lower_bound_must_bind_exact_run_output_and_schema() {
        let profile = discrepancy_profile();
        let mut parametric = parametric();
        parametric.output_sha256 = sha('z');
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(39),
            &applicability(),
            Some(&parametric),
        );
        assert!(report
            .diagnostics
            .contains(&ModelDiscrepancyDiagnostic::ParametricOutputMismatch));
    }

    #[test]
    fn parametric_lower_bound_cannot_exceed_point_estimate() {
        let profile = discrepancy_profile();
        let mut parametric = parametric();
        parametric.lower_bound_mwe = 151.0;
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(39),
            &applicability(),
            Some(&parametric),
        );
        assert!(report
            .diagnostics
            .contains(&ModelDiscrepancyDiagnostic::ParametricLowerBoundAbovePointEstimate));
    }

    #[test]
    fn residual_input_order_does_not_change_conformal_allowance() {
        let profile = discrepancy_profile();
        let residuals = measured_residuals(39);
        let first = evaluate(
            &profile,
            &target(),
            &residuals,
            &applicability(),
            Some(&parametric()),
        );
        let mut reversed = residuals;
        reversed.reverse();
        let second = evaluate(
            &profile,
            &target(),
            &reversed,
            &applicability(),
            Some(&parametric()),
        );
        assert_eq!(first.discrepancy_bound_mwe, second.discrepancy_bound_mwe);
        assert_eq!(first.conformal_rank, second.conformal_rank);
        assert_eq!(first.combined_coverage_floor_basis_points, second.combined_coverage_floor_basis_points);
    }

    #[test]
    fn residual_geometry_must_belong_to_the_declared_domain() {
        let profile = discrepancy_profile();
        let mut residuals = measured_residuals(39);
        residuals[0].geometry_sha256 = sha('z');
        let report = evaluate(
            &profile,
            &target(),
            &residuals,
            &applicability(),
            Some(&parametric()),
        );
        assert!(report
            .diagnostics
            .contains(&ModelDiscrepancyDiagnostic::ResidualGeometryOutsideDomain));
    }

    #[test]
    fn missing_parametric_bound_is_not_replaced_by_zero() {
        let profile = discrepancy_profile();
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(39),
            &applicability(),
            None,
        );
        assert_eq!(report.disposition, ModelDiscrepancyDisposition::NotEvaluable);
        assert_eq!(report.adjusted_lower_bound_mwe, None);
    }

    #[test]
    fn maximum_discrepancy_threshold_equality_obeys_policy() {
        let mut profile = discrepancy_profile();
        profile.maximum_discrepancy_mwe = 19.5;
        profile.discrepancy_tolerance_inclusive = true;
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(39),
            &applicability(),
            Some(&parametric()),
        );
        assert_eq!(report.disposition, ModelDiscrepancyDisposition::QualifiedWithinDeclaredDomain);

        profile.discrepancy_tolerance_inclusive = false;
        let report = evaluate(
            &profile,
            &target(),
            &measured_residuals(39),
            &applicability(),
            Some(&parametric()),
        );
        assert_eq!(report.disposition, ModelDiscrepancyDisposition::MaterialDiscrepancy);
    }
}
