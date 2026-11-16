//! Domain-conditioned model-form discrepancy accounting for fusion service.
//!
//! Consumes independent measured calibration residuals and separately
//! authenticated applicability / parametric-bound receipts. It does not train
//! a model, estimate covariance, or infer an OOD classifier.

use std::collections::BTreeSet;

use super::robust_net_export::{
    ModelFormValidity, NetExportBoundMethod, NetExportEvidenceClass, NetExportLowerBoundReceipt,
};

pub const MODEL_FORM_DISCREPANCY_VERSION: &str =
    "fusion-model-form-discrepancy/0.1";
pub const CONFORMAL_METHOD: &str = "split-conformal-absolute-residual-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFormEvidenceClass {
    SyntheticFixture,
    ModelOutput,
    MeasuredObservation,
    IndependentVerification,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApplicabilityState {
    InDeclaredDomain,
    OutOfDomain,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFormDisposition {
    QualifiedWithinDeclaredDomain,
    OutOfDomain,
    ApplicabilityUnknown,
    MaterialDiscrepancy,
    InsufficientCalibration,
    CoverageAllocationInsufficient,
    SyntheticOnly,
    NotEvaluable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFormDiagnostic {
    EmptyProfileIdentity,
    InvalidProfileDigest,
    EmptyModelFamily,
    EmptySolverRevision,
    EmptyObservable,
    EmptyUnit,
    EmptyServiceBoundary,
    EmptyAveragingWindow,
    EmptyRegime,
    InvalidGeometryDigest,
    EmptyValidityDomain,
    InvalidValidityDomainDigest,
    EmptyRequestedUse,
    InvalidCalibrationManifestDigest,
    EmptyExchangeabilityScope,
    InvalidCoverage,
    InvalidMaterialTolerance,
    InvalidMinimumSampleCount,
    EmptyTargetRunIdentity,
    InvalidTargetInputDigest,
    InvalidTargetOutputDigest,
    InvalidTargetSchemaDigest,
    TargetProfileMismatch,
    EmptyCalibrationSet,
    DuplicateResidualRecord,
    DuplicateEpisode,
    DuplicateMeasurement,
    DuplicateArtifact,
    DuplicateModelRun,
    InvalidResidualIdentity,
    InvalidResidualDigest,
    ResidualProfileMismatch,
    NonFiniteResidual,
    NegativeResidual,
    MissingMetrologyReference,
    CalibrationLeakage,
    InvalidCalibrationEvidenceClass,
    ApplicabilityReceiptMismatch,
    ApplicabilityEvidenceInvalid,
    ApplicabilityProfileMismatch,
    ParametricReceiptMismatch,
    ParametricEvidenceInvalid,
    NonFiniteParametricBound,
    NonFinitePointEstimate,
    ParametricBoundAbovePointEstimate,
    InvalidParametricCoverage,
    EmptyCoverageScope,
    InvalidDecisionEvidence,
    ConformalRankUnavailable,
    CoverageFloorBelowTarget,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelFormDiscrepancyProfile {
    pub profile_id: String,
    pub profile_sha256: String,
    pub model_family: String,
    pub solver_revision: String,
    pub observable_id: String,
    pub unit_ref: String,
    pub service_boundary_ref: String,
    pub averaging_window_ref: String,
    pub regime_ref: String,
    pub geometry_sha256: String,
    pub validity_domain_ref: String,
    pub validity_domain_sha256: String,
    pub requested_model_use_ref: String,
    pub target_total_coverage: f64,
    pub parametric_coverage: f64,
    pub discrepancy_coverage: f64,
    pub material_discrepancy_tolerance: f64,
    /// true: equality is within tolerance; false: equality is material.
    pub tolerance_inclusive: bool,
    pub minimum_calibration_samples: usize,
    pub calibration_dataset_manifest_sha256: String,
    pub exchangeability_scope_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelUseIdentity {
    pub run_id: String,
    pub candidate_revision: String,
    pub solver_revision: String,
    pub input_sha256: String,
    pub output_sha256: String,
    pub schema_ref: String,
    pub schema_sha256: String,
    pub observable_id: String,
    pub point_estimate_mwe: f64,
    pub unit_ref: String,
    pub service_boundary_ref: String,
    pub averaging_window_ref: String,
    pub regime_ref: String,
    pub geometry_sha256: String,
    pub requested_model_use_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationResidual {
    pub record_id: String,
    pub source_case_id: String,
    pub episode_id: String,
    pub facility_id: String,
    pub model_run_id: String,
    pub model_output_sha256: String,
    pub measured_observation_id: String,
    pub measured_artifact_sha256: String,
    pub observable_id: String,
    pub unit_ref: String,
    pub service_boundary_ref: String,
    pub averaging_window_ref: String,
    pub regime_ref: String,
    pub geometry_sha256: String,
    pub absolute_residual_mwe: f64,
    pub evidence_class: ModelFormEvidenceClass,
    pub metrology_reference: String,
    pub evidence_ref: String,
    pub used_for_model_fit: bool,
    pub used_for_normalization: bool,
    pub used_for_feature_selection: bool,
    pub used_for_hyperparameter_selection: bool,
    pub used_for_model_selection: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplicabilityReceipt {
    pub receipt_ref: String,
    pub receipt_sha256: String,
    pub run_id: String,
    pub solver_revision: String,
    pub input_sha256: String,
    pub regime_ref: String,
    pub geometry_sha256: String,
    pub requested_model_use_ref: String,
    pub validity_domain_ref: String,
    pub validity_domain_sha256: String,
    pub method_profile_ref: String,
    pub evidence_ref: String,
    pub evidence_sha256: String,
    pub state: ApplicabilityState,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParametricNetPowerLowerBound {
    pub receipt_ref: String,
    pub receipt_sha256: String,
    pub run_id: String,
    pub output_sha256: String,
    pub schema_ref: String,
    pub schema_sha256: String,
    pub observable_id: String,
    pub lower_bound_mwe: f64,
    pub one_sided_coverage: f64,
    pub coverage_scope_ref: String,
    pub method_ref: String,
    pub assumption_manifest_sha256: String,
    pub unit_ref: String,
    pub service_boundary_ref: String,
    pub averaging_window_ref: String,
    pub evidence_ref: String,
    pub evidence_sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelFormDiscrepancyPlan {
    pub profile: ModelFormDiscrepancyProfile,
    pub target: ModelUseIdentity,
    pub calibration_residuals: Vec<CalibrationResidual>,
    pub applicability: ApplicabilityReceipt,
    pub parametric_bound: ParametricNetPowerLowerBound,
    /// Digest of the canonical decision receipt, supplied by the evidence
    /// publisher. This crate validates its identity format, not the payload bytes.
    pub decision_receipt_ref: String,
    pub decision_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelFormDiscrepancyReport {
    pub contract_version: &'static str,
    pub profile_id: String,
    pub run_id: String,
    pub disposition: ModelFormDisposition,
    pub diagnostics: Vec<ModelFormDiagnostic>,
    pub calibration_sample_count: usize,
    pub conformal_rank_one_based: Option<usize>,
    pub discrepancy_allowance_mwe: Option<f64>,
    pub parametric_lower_bound_mwe: Option<f64>,
    pub combined_lower_bound_mwe: Option<f64>,
    pub parametric_coverage: Option<f64>,
    pub discrepancy_coverage: Option<f64>,
    pub combined_coverage_floor: Option<f64>,
    pub target_total_coverage: f64,
    pub residual_order_statistic_mwe: Option<Vec<f64>>,
    pub model_form_validity: ModelFormValidity,
    pub calibration_evidence_refs: Vec<String>,
    pub applicability_evidence_ref: Option<String>,
    pub parametric_evidence_ref: Option<String>,
    pub decision_receipt_ref: Option<String>,
    pub decision_receipt_sha256: Option<String>,
    pub claim_ceiling: &'static str,
}

impl ModelFormDiscrepancyReport {
    /// Convert only a fully qualified measured/modelled result into the
    /// downstream robust net-export receipt. No receipt is produced for
    /// synthetic, assumption-only, OOD, unknown or insufficient cases.
    pub fn to_net_export_receipt(
        &self,
        plan: &ModelFormDiscrepancyPlan,
    ) -> Option<NetExportLowerBoundReceipt> {
        if self.disposition != ModelFormDisposition::QualifiedWithinDeclaredDomain {
            return None;
        }
        let lower_bound_mwe = self.combined_lower_bound_mwe?;
        let decision_receipt_ref = self.decision_receipt_ref.clone()?;
        let decision_receipt_sha256 = self.decision_receipt_sha256.clone()?;
        let applicability = &plan.applicability;
        let parametric = &plan.parametric_bound;
        let profile = &plan.profile;
        let target = &plan.target;
        let combined_coverage = self.combined_coverage_floor?;

        Some(NetExportLowerBoundReceipt {
            receipt_id: decision_receipt_ref.clone(),
            run_id: target.run_id.clone(),
            output_sha256: target.output_sha256.clone(),
            schema_ref: target.schema_ref.clone(),
            schema_sha256: target.schema_sha256.clone(),
            service_boundary_ref: target.service_boundary_ref.clone(),
            unit_ref: target.unit_ref.clone(),
            averaging_window_ref: target.averaging_window_ref.clone(),
            sign_convention_ref: "net-export-positive-import-negative-v1".to_string(),
            lower_bound_mwe,
            one_sided_coverage_level: combined_coverage,
            evidence_class: NetExportEvidenceClass::ModelledOutput,
            coverage_scope_ref: format!(
                "{}|{}|{}",
                profile.exchangeability_scope_ref,
                parametric.coverage_scope_ref,
                applicability.method_profile_ref
            ),
            method: NetExportBoundMethod::ParametricPlusModelFormDiscrepancy,
            method_ref: format!(
                "parametric-lcb-minus-{}",
                CONFORMAL_METHOD
            ),
            method_evidence_sha256: decision_receipt_sha256.clone(),
            method_run_id: target.run_id.clone(),
            method_output_sha256: target.output_sha256.clone(),
            method_variable_order: vec![target.observable_id.clone()],
            distribution_assumption_ref: format!(
                "parametric-coverage:{}; discrepancy-exchangeability:{}; composition:bonferroni-union-bound",
                parametric.method_ref,
                profile.exchangeability_scope_ref
            ),
            assumption_manifest_sha256: profile.profile_sha256.clone(),
            validity_domain_ref: profile.validity_domain_ref.clone(),
            model_form_validity: ModelFormValidity::WithinDeclaredDomain,
            model_form_evidence_ref: decision_receipt_ref.clone(),
            model_form_evidence_sha256: decision_receipt_sha256.clone(),
            covariance_receipt_ref: None,
            covariance_receipt_sha256: None,
            covariance_run_id: None,
            covariance_output_sha256: None,
            covariance_variable_order: Vec::new(),
            simultaneous_allocation_ref: None,
            direct_net_analysis_ref: None,
            evidence_ref: format!(
                "{}|{}|{}|{}",
                decision_receipt_ref,
                applicability.evidence_ref,
                parametric.evidence_ref,
                profile.calibration_dataset_manifest_sha256
            ),
        })
    }
}

/// Evaluate a split-conformal absolute-residual envelope and conservatively
/// compose it with a same-run parametric lower bound.
///
/// If c_p and c_d are the two marginal coverage levels, the combined coverage
/// floor is max(0, 1 - ((1-c_p) + (1-c_d))). This union bound does not assume
/// independence between parametric uncertainty and model discrepancy.
pub fn evaluate_model_form_discrepancy(
    plan: &ModelFormDiscrepancyPlan,
) -> ModelFormDiscrepancyReport {
    let mut diagnostics = validate_plan(plan);
    if !diagnostics.is_empty() {
        return report(
            plan,
            ModelFormDisposition::NotEvaluable,
            diagnostics,
            None,
            None,
            None,
            None,
            None,
            None,
        );
    }

    match plan.applicability.state {
        ApplicabilityState::OutOfDomain => {
            return report(
                plan,
                ModelFormDisposition::OutOfDomain,
                Vec::new(),
                None,
                None,
                None,
                None,
                None,
                None,
            );
        }
        ApplicabilityState::Unknown => {
            return report(
                plan,
                ModelFormDisposition::ApplicabilityUnknown,
                Vec::new(),
                None,
                None,
                None,
                None,
                None,
                None,
            );
        }
        ApplicabilityState::InDeclaredDomain => {}
    }

    if plan
        .calibration_residuals
        .iter()
        .any(|item| item.evidence_class == ModelFormEvidenceClass::SyntheticFixture)
    {
        return report(
            plan,
            ModelFormDisposition::SyntheticOnly,
            Vec::new(),
            None,
            None,
            None,
            None,
            None,
            None,
        );
    }

    let mut residual_values: Vec<f64> = plan
        .calibration_residuals
        .iter()
        .map(|item| item.absolute_residual_mwe)
        .collect();
    residual_values.sort_by(f64::total_cmp);

    let n = residual_values.len();
    let rank = ((n + 1) as f64 * plan.profile.discrepancy_coverage).ceil() as usize;
    if rank == 0 || rank > n {
        return report(
            plan,
            ModelFormDisposition::InsufficientCalibration,
            vec![ModelFormDiagnostic::ConformalRankUnavailable],
            Some(rank),
            None,
            Some(plan.parametric_bound.lower_bound_mwe),
            None,
            Some(coverage_floor(
                plan.parametric_bound.one_sided_coverage,
                plan.profile.discrepancy_coverage,
            )),
            Some(residual_values),
        );
    }

    let allowance = residual_values[rank - 1];
    let floor = coverage_floor(
        plan.parametric_bound.one_sided_coverage,
        plan.profile.discrepancy_coverage,
    );
    let combined = plan.parametric_bound.lower_bound_mwe - allowance;

    let material = if plan.profile.tolerance_inclusive {
        allowance > plan.profile.material_discrepancy_tolerance
    } else {
        allowance >= plan.profile.material_discrepancy_tolerance
    };
    if material {
        return report(
            plan,
            ModelFormDisposition::MaterialDiscrepancy,
            Vec::new(),
            Some(rank),
            Some(allowance),
            Some(plan.parametric_bound.lower_bound_mwe),
            None,
            Some(floor),
            Some(residual_values),
        );
    }

    if floor + f64::EPSILON < plan.profile.target_total_coverage {
        return report(
            plan,
            ModelFormDisposition::CoverageAllocationInsufficient,
            vec![ModelFormDiagnostic::CoverageFloorBelowTarget],
            Some(rank),
            Some(allowance),
            Some(plan.parametric_bound.lower_bound_mwe),
            None,
            Some(floor),
            Some(residual_values),
        );
    }

    report(
        plan,
        ModelFormDisposition::QualifiedWithinDeclaredDomain,
        Vec::new(),
        Some(rank),
        Some(allowance),
        Some(plan.parametric_bound.lower_bound_mwe),
        Some(combined),
        Some(floor),
        Some(residual_values),
    )
}

fn validate_plan(plan: &ModelFormDiscrepancyPlan) -> Vec<ModelFormDiagnostic> {
    let profile = &plan.profile;
    let target = &plan.target;
    let applicability = &plan.applicability;
    let parametric = &plan.parametric_bound;
    let mut errors = Vec::new();

    if profile.profile_id.trim().is_empty() {
        errors.push(ModelFormDiagnostic::EmptyProfileIdentity);
    }
    if !is_sha256(&profile.profile_sha256) {
        errors.push(ModelFormDiagnostic::InvalidProfileDigest);
    }
    if profile.model_family.trim().is_empty() {
        errors.push(ModelFormDiagnostic::EmptyModelFamily);
    }
    if profile.solver_revision.trim().is_empty() {
        errors.push(ModelFormDiagnostic::EmptySolverRevision);
    }
    if profile.observable_id.trim().is_empty() {
        errors.push(ModelFormDiagnostic::EmptyObservable);
    }
    if profile.unit_ref.trim().is_empty() {
        errors.push(ModelFormDiagnostic::EmptyUnit);
    }
    if profile.service_boundary_ref.trim().is_empty() {
        errors.push(ModelFormDiagnostic::EmptyServiceBoundary);
    }
    if profile.averaging_window_ref.trim().is_empty() {
        errors.push(ModelFormDiagnostic::EmptyAveragingWindow);
    }
    if profile.regime_ref.trim().is_empty() {
        errors.push(ModelFormDiagnostic::EmptyRegime);
    }
    if !is_sha256(&profile.geometry_sha256) {
        errors.push(ModelFormDiagnostic::InvalidGeometryDigest);
    }
    if profile.validity_domain_ref.trim().is_empty() {
        errors.push(ModelFormDiagnostic::EmptyValidityDomain);
    }
    if !is_sha256(&profile.validity_domain_sha256) {
        errors.push(ModelFormDiagnostic::InvalidValidityDomainDigest);
    }
    if profile.requested_model_use_ref.trim().is_empty() {
        errors.push(ModelFormDiagnostic::EmptyRequestedUse);
    }
    if !is_sha256(&profile.calibration_dataset_manifest_sha256) {
        errors.push(ModelFormDiagnostic::InvalidCalibrationManifestDigest);
    }
    if profile.exchangeability_scope_ref.trim().is_empty() {
        errors.push(ModelFormDiagnostic::EmptyExchangeabilityScope);
    }
    for coverage in [
        profile.target_total_coverage,
        profile.parametric_coverage,
        profile.discrepancy_coverage,
    ] {
        if !coverage.is_finite() || coverage <= 0.0 || coverage >= 1.0 {
            errors.push(ModelFormDiagnostic::InvalidCoverage);
        }
    }
    if !profile.material_discrepancy_tolerance.is_finite()
        || profile.material_discrepancy_tolerance < 0.0
    {
        errors.push(ModelFormDiagnostic::InvalidMaterialTolerance);
    }
    if profile.minimum_calibration_samples == 0 {
        errors.push(ModelFormDiagnostic::InvalidMinimumSampleCount);
    }

    if target.run_id.trim().is_empty()
        || target.candidate_revision.trim().is_empty()
        || target.solver_revision.trim().is_empty()
        || target.schema_ref.trim().is_empty()
        || target.observable_id.trim().is_empty()
    {
        errors.push(ModelFormDiagnostic::EmptyTargetRunIdentity);
    }
    if !is_sha256(&target.input_sha256) {
        errors.push(ModelFormDiagnostic::InvalidTargetInputDigest);
    }
    if !is_sha256(&target.output_sha256) {
        errors.push(ModelFormDiagnostic::InvalidTargetOutputDigest);
    }
    if !is_sha256(&target.schema_sha256) {
        errors.push(ModelFormDiagnostic::InvalidTargetSchemaDigest);
    }
    if !target.point_estimate_mwe.is_finite() {
        errors.push(ModelFormDiagnostic::NonFinitePointEstimate);
    }

    if profile.solver_revision != target.solver_revision
        || profile.observable_id != target.observable_id
        || profile.unit_ref != target.unit_ref
        || profile.service_boundary_ref != target.service_boundary_ref
        || profile.averaging_window_ref != target.averaging_window_ref
        || profile.regime_ref != target.regime_ref
        || profile.geometry_sha256 != target.geometry_sha256
        || profile.requested_model_use_ref != target.requested_model_use_ref
    {
        errors.push(ModelFormDiagnostic::TargetProfileMismatch);
    }

    if plan.calibration_residuals.is_empty() {
        errors.push(ModelFormDiagnostic::EmptyCalibrationSet);
    }
    let mut record_ids = BTreeSet::new();
    let mut episode_ids = BTreeSet::new();
    let mut measurement_ids = BTreeSet::new();
    let mut artifact_digests = BTreeSet::new();
    let mut model_run_ids = BTreeSet::new();
    for residual in &plan.calibration_residuals {
        if residual.record_id.trim().is_empty()
            || residual.source_case_id.trim().is_empty()
            || residual.episode_id.trim().is_empty()
            || residual.facility_id.trim().is_empty()
            || residual.model_run_id.trim().is_empty()
            || residual.measured_observation_id.trim().is_empty()
        {
            errors.push(ModelFormDiagnostic::InvalidResidualIdentity);
        }
        if !record_ids.insert(residual.record_id.as_str()) {
            errors.push(ModelFormDiagnostic::DuplicateResidualRecord);
        }
        if !episode_ids.insert(residual.episode_id.as_str()) {
            errors.push(ModelFormDiagnostic::DuplicateEpisode);
        }
        if !measurement_ids.insert(residual.measured_observation_id.as_str()) {
            errors.push(ModelFormDiagnostic::DuplicateMeasurement);
        }
        if !artifact_digests.insert(residual.measured_artifact_sha256.as_str()) {
            errors.push(ModelFormDiagnostic::DuplicateArtifact);
        }
        if !model_run_ids.insert(residual.model_run_id.as_str()) {
            errors.push(ModelFormDiagnostic::DuplicateModelRun);
        }
        if !is_sha256(&residual.model_output_sha256)
            || !is_sha256(&residual.measured_artifact_sha256)
            || !is_sha256(&residual.geometry_sha256)
        {
            errors.push(ModelFormDiagnostic::InvalidResidualDigest);
        }
        if residual.solver_mismatch(profile)
        {
            errors.push(ModelFormDiagnostic::ResidualProfileMismatch);
        }
        if !residual.absolute_residual_mwe.is_finite() {
            errors.push(ModelFormDiagnostic::NonFiniteResidual);
        } else if residual.absolute_residual_mwe < 0.0 {
            errors.push(ModelFormDiagnostic::NegativeResidual);
        }
        if residual.metrology_reference.trim().is_empty() || residual.evidence_ref.trim().is_empty() {
            errors.push(ModelFormDiagnostic::MissingMetrologyReference);
        }
        if residual.used_for_model_fit
            || residual.used_for_normalization
            || residual.used_for_feature_selection
            || residual.used_for_hyperparameter_selection
            || residual.used_for_model_selection
        {
            errors.push(ModelFormDiagnostic::CalibrationLeakage);
        }
        if !matches!(
            residual.evidence_class,
            ModelFormEvidenceClass::MeasuredObservation
                | ModelFormEvidenceClass::IndependentVerification
                | ModelFormEvidenceClass::SyntheticFixture
        ) {
            errors.push(ModelFormDiagnostic::InvalidCalibrationEvidenceClass);
        }
    }

    if applicability.receipt_ref.trim().is_empty()
        || applicability.evidence_ref.trim().is_empty()
    {
        errors.push(ModelFormDiagnostic::ApplicabilityEvidenceInvalid);
    }
    if !is_sha256(&applicability.receipt_sha256)
        || !is_sha256(&applicability.evidence_sha256)
    {
        errors.push(ModelFormDiagnostic::ApplicabilityEvidenceInvalid);
    }
    if applicability.run_id != target.run_id
        || applicability.solver_revision != target.solver_revision
        || applicability.input_sha256 != target.input_sha256
        || applicability.regime_ref != target.regime_ref
        || applicability.geometry_sha256 != target.geometry_sha256
        || applicability.requested_model_use_ref != target.requested_model_use_ref
    {
        errors.push(ModelFormDiagnostic::ApplicabilityReceiptMismatch);
    }
    if applicability.validity_domain_ref != profile.validity_domain_ref
        || applicability.validity_domain_sha256 != profile.validity_domain_sha256
        || applicability.method_profile_ref != profile.profile_id
    {
        errors.push(ModelFormDiagnostic::ApplicabilityProfileMismatch);
    }

    if parametric.receipt_ref.trim().is_empty()
        || parametric.evidence_ref.trim().is_empty()
        || parametric.method_ref.trim().is_empty()
        || parametric.coverage_scope_ref.trim().is_empty()
    {
        errors.push(ModelFormDiagnostic::ParametricEvidenceInvalid);
    }
    if !is_sha256(&parametric.receipt_sha256)
        || !is_sha256(&parametric.evidence_sha256)
        || !is_sha256(&parametric.assumption_manifest_sha256)
    {
        errors.push(ModelFormDiagnostic::ParametricEvidenceInvalid);
    }
    if parametric.run_id != target.run_id
        || parametric.output_sha256 != target.output_sha256
        || parametric.schema_ref != target.schema_ref
        || parametric.schema_sha256 != target.schema_sha256
        || parametric.observable_id != target.observable_id
        || parametric.unit_ref != target.unit_ref
        || parametric.service_boundary_ref != target.service_boundary_ref
        || parametric.averaging_window_ref != target.averaging_window_ref
    {
        errors.push(ModelFormDiagnostic::ParametricReceiptMismatch);
    }
    if !parametric.lower_bound_mwe.is_finite() {
        errors.push(ModelFormDiagnostic::NonFiniteParametricBound);
    } else if parametric.lower_bound_mwe > target.point_estimate_mwe {
        errors.push(ModelFormDiagnostic::ParametricBoundAbovePointEstimate);
    }
    if !parametric.one_sided_coverage.is_finite()
        || parametric.one_sided_coverage <= 0.0
        || parametric.one_sided_coverage >= 1.0
    {
        errors.push(ModelFormDiagnostic::InvalidParametricCoverage);
    }
    if !is_sha256(&plan.decision_receipt_sha256) || plan.decision_receipt_ref.trim().is_empty() {
        errors.push(ModelFormDiagnostic::InvalidDecisionEvidence);
    }

    errors
}

impl CalibrationResidual {
    fn solver_mismatch(&self, profile: &ModelFormDiscrepancyProfile) -> bool {
        self.observable_id != profile.observable_id
            || self.unit_ref != profile.unit_ref
            || self.service_boundary_ref != profile.service_boundary_ref
            || self.averaging_window_ref != profile.averaging_window_ref
            || self.regime_ref != profile.regime_ref
            || self.geometry_sha256 != profile.geometry_sha256
    }
}

fn report(
    plan: &ModelFormDiscrepancyPlan,
    disposition: ModelFormDisposition,
    diagnostics: Vec<ModelFormDiagnostic>,
    rank: Option<usize>,
    allowance: Option<f64>,
    parametric_lower: Option<f64>,
    combined_lower: Option<f64>,
    coverage_floor_value: Option<f64>,
    residuals: Option<Vec<f64>>,
) -> ModelFormDiscrepancyReport {
    let model_form_validity = match disposition {
        ModelFormDisposition::QualifiedWithinDeclaredDomain => {
            ModelFormValidity::WithinDeclaredDomain
        }
        ModelFormDisposition::OutOfDomain => ModelFormValidity::OutOfDomain,
        ModelFormDisposition::ApplicabilityUnknown
        | ModelFormDisposition::InsufficientCalibration
        | ModelFormDisposition::CoverageAllocationInsufficient
        | ModelFormDisposition::SyntheticOnly => ModelFormValidity::Unknown,
        ModelFormDisposition::MaterialDiscrepancy => ModelFormValidity::MaterialDiscrepancy,
        ModelFormDisposition::NotEvaluable => ModelFormValidity::Unknown,
    };
    ModelFormDiscrepancyReport {
        contract_version: MODEL_FORM_DISCREPANCY_VERSION,
        profile_id: plan.profile.profile_id.clone(),
        run_id: plan.target.run_id.clone(),
        disposition,
        diagnostics,
        calibration_sample_count: plan.calibration_residuals.len(),
        conformal_rank_one_based: rank,
        discrepancy_allowance_mwe: allowance,
        parametric_lower_bound_mwe: parametric_lower,
        combined_lower_bound_mwe: combined_lower,
        parametric_coverage: Some(plan.parametric_bound.one_sided_coverage),
        discrepancy_coverage: Some(plan.profile.discrepancy_coverage),
        combined_coverage_floor: coverage_floor_value,
        target_total_coverage: plan.profile.target_total_coverage,
        residual_order_statistic_mwe: residuals,
        model_form_validity,
        calibration_evidence_refs: plan
            .calibration_residuals
            .iter()
            .map(|item| item.evidence_ref.clone())
            .collect(),
        applicability_evidence_ref: Some(plan.applicability.evidence_ref.clone()),
        parametric_evidence_ref: Some(plan.parametric_bound.evidence_ref.clone()),
        decision_receipt_ref: Some(plan.decision_receipt_ref.clone()),
        decision_receipt_sha256: Some(plan.decision_receipt_sha256.clone()),
        claim_ceiling: "domain-conditioned conformal residual accounting under declared exchangeability scope only",
    }
}

fn coverage_floor(parametric: f64, discrepancy: f64) -> f64 {
    (1.0 - ((1.0 - parametric) + (1.0 - discrepancy))).max(0.0)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn profile() -> ModelFormDiscrepancyProfile {
        ModelFormDiscrepancyProfile {
            profile_id: "D0FUS-ITER-net-power-model-form-v1".to_string(),
            profile_sha256: sha('a'),
            model_family: "D0FUS".to_string(),
            solver_revision: "d0fus-3.2.0-a682c351".to_string(),
            observable_id: "P_elec".to_string(),
            unit_ref: "MWe".to_string(),
            service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
            averaging_window_ref: "iter-run-steady-state-window-v1".to_string(),
            regime_ref: "ITER-reference-regime-v1".to_string(),
            geometry_sha256: sha('b'),
            validity_domain_ref: "d0fus-iter-domain-v1".to_string(),
            validity_domain_sha256: sha('c'),
            requested_model_use_ref: "reference-case-model-output-v1".to_string(),
            target_total_coverage: 0.95,
            parametric_coverage: 0.975,
            discrepancy_coverage: 0.975,
            material_discrepancy_tolerance: 25.0,
            tolerance_inclusive: true,
            minimum_calibration_samples: 19,
            calibration_dataset_manifest_sha256: sha('d'),
            exchangeability_scope_ref: "one-independent-shot-per-episode-v1".to_string(),
        }
    }

    fn target() -> ModelUseIdentity {
        let profile = profile();
        ModelUseIdentity {
            run_id: "target-run-900".to_string(),
            candidate_revision: "iter-reference-r1".to_string(),
            solver_revision: profile.solver_revision.clone(),
            input_sha256: sha('e'),
            output_sha256: sha('f'),
            schema_ref: "d0fus-iter-schema-165-v1".to_string(),
            schema_sha256: sha('a'),
            observable_id: profile.observable_id,
            point_estimate_mwe: -42.84920863494321,
            unit_ref: profile.unit_ref,
            service_boundary_ref: profile.service_boundary_ref,
            averaging_window_ref: profile.averaging_window_ref,
            regime_ref: profile.regime_ref,
            geometry_sha256: profile.geometry_sha256,
            requested_model_use_ref: profile.requested_model_use_ref,
        }
    }

    fn residual(index: usize, value: f64) -> CalibrationResidual {
        let profile = profile();
        CalibrationResidual {
            record_id: format!("residual-{index}"),
            source_case_id: format!("case-{index}"),
            episode_id: format!("episode-{index}"),
            facility_id: "facility-a".to_string(),
            model_run_id: format!("calibration-run-{index}"),
            model_output_sha256: sha(char::from_u32('g' as u32 + (index % 10) as u32).unwrap()),
            measured_observation_id: format!("measurement-{index}"),
            measured_artifact_sha256: sha(char::from_u32('q' as u32 + (index % 8) as u32).unwrap()),
            observable_id: profile.observable_id,
            unit_ref: profile.unit_ref,
            service_boundary_ref: profile.service_boundary_ref,
            averaging_window_ref: profile.averaging_window_ref,
            regime_ref: profile.regime_ref,
            geometry_sha256: profile.geometry_sha256,
            absolute_residual_mwe: value,
            evidence_class: ModelFormEvidenceClass::MeasuredObservation,
            metrology_reference: format!("metrology-{index}"),
            evidence_ref: format!("measured-residual-evidence-{index}"),
            used_for_model_fit: false,
            used_for_normalization: false,
            used_for_feature_selection: false,
            used_for_hyperparameter_selection: false,
            used_for_model_selection: false,
        }
    }

    fn plan(count: usize, residual_value: f64) -> ModelFormDiscrepancyPlan {
        let profile = profile();
        let target = target();
        ModelFormDiscrepancyPlan {
            profile: profile.clone(),
            target: target.clone(),
            calibration_residuals: (0..count)
                .map(|index| residual(index, residual_value + index as f64 * 0.01))
                .collect(),
            applicability: ApplicabilityReceipt {
                receipt_ref: "applicability-receipt-900".to_string(),
                receipt_sha256: sha('b'),
                run_id: target.run_id.clone(),
                solver_revision: target.solver_revision.clone(),
                input_sha256: target.input_sha256.clone(),
                regime_ref: target.regime_ref.clone(),
                geometry_sha256: target.geometry_sha256.clone(),
                requested_model_use_ref: target.requested_model_use_ref.clone(),
                validity_domain_ref: profile.validity_domain_ref.clone(),
                validity_domain_sha256: profile.validity_domain_sha256.clone(),
                method_profile_ref: profile.profile_id.clone(),
                evidence_ref: "applicability-evidence-900".to_string(),
                evidence_sha256: sha('c'),
                state: ApplicabilityState::InDeclaredDomain,
            },
            parametric_bound: ParametricNetPowerLowerBound {
                receipt_ref: "parametric-bound-900".to_string(),
                receipt_sha256: sha('d'),
                run_id: target.run_id.clone(),
                output_sha256: target.output_sha256.clone(),
                schema_ref: target.schema_ref.clone(),
                schema_sha256: target.schema_sha256.clone(),
                observable_id: target.observable_id.clone(),
                lower_bound_mwe: -45.0,
                one_sided_coverage: 0.975,
                coverage_scope_ref: "symtropy-joint-contrast-family-v1".to_string(),
                method_ref: "joint-covariance-net-power-contrast-v1".to_string(),
                assumption_manifest_sha256: sha('e'),
                unit_ref: target.unit_ref.clone(),
                service_boundary_ref: target.service_boundary_ref.clone(),
                averaging_window_ref: target.averaging_window_ref.clone(),
                evidence_ref: "parametric-bound-evidence-900".to_string(),
                evidence_sha256: sha('f'),
            },
            decision_receipt_ref: "model-form-decision-900".to_string(),
            decision_receipt_sha256: sha('a'),
        }
    }

    #[test]
    fn nineteen_independent_residuals_at_ninety_five_percent_use_the_maximum() {
        let mut plan = plan(19, 1.0);
        plan.profile.discrepancy_coverage = 0.95;
        plan.profile.parametric_coverage = 0.975;
        let report = evaluate_model_form_discrepancy(&plan);
        assert_eq!(report.disposition, ModelFormDisposition::QualifiedWithinDeclaredDomain);
        assert_eq!(report.conformal_rank_one_based, Some(19));
        assert_eq!(report.discrepancy_allowance_mwe, Some(1.18));
        assert_eq!(report.combined_coverage_floor, Some(0.925));
    }

    #[test]
    fn eighteen_residuals_cannot_support_finite_ninety_five_percent_bound() {
        let mut plan = plan(18, 1.0);
        plan.profile.discrepancy_coverage = 0.95;
        let report = evaluate_model_form_discrepancy(&plan);
        assert_eq!(report.disposition, ModelFormDisposition::InsufficientCalibration);
        assert_eq!(report.conformal_rank_one_based, Some(19));
        assert_eq!(report.discrepancy_allowance_mwe, None);
    }

    #[test]
    fn thirty_nine_residuals_at_ninety_seven_point_five_percent_use_maximum() {
        let report = evaluate_model_form_discrepancy(&plan(39, 1.0));
        assert_eq!(report.disposition, ModelFormDisposition::QualifiedWithinDeclaredDomain);
        assert_eq!(report.conformal_rank_one_based, Some(39));
    }

    #[test]
    fn parametric_and_discrepancy_coverages_use_union_bound_not_product() {
        let report = evaluate_model_form_discrepancy(&plan(39, 1.0));
        assert_eq!(report.combined_coverage_floor, Some(0.95));
    }

    #[test]
    fn two_ninety_five_percent_components_do_not_meet_ninety_five_percent_total() {
        let mut plan = plan(39, 1.0);
        plan.profile.parametric_coverage = 0.95;
        plan.profile.discrepancy_coverage = 0.95;
        let report = evaluate_model_form_discrepancy(&plan);
        assert_eq!(
            report.disposition,
            ModelFormDisposition::CoverageAllocationInsufficient
        );
        assert_eq!(report.combined_coverage_floor, Some(0.8999999999999999));
    }

    #[test]
    fn out_of_domain_applicability_produces_no_lower_bound() {
        let mut plan = plan(39, 1.0);
        plan.applicability.state = ApplicabilityState::OutOfDomain;
        let report = evaluate_model_form_discrepancy(&plan);
        assert_eq!(report.disposition, ModelFormDisposition::OutOfDomain);
        assert_eq!(report.combined_lower_bound_mwe, None);
        assert!(report.to_net_export_receipt(&plan).is_none());
    }

    #[test]
    fn unknown_applicability_produces_no_lower_bound() {
        let mut plan = plan(39, 1.0);
        plan.applicability.state = ApplicabilityState::Unknown;
        let report = evaluate_model_form_discrepancy(&plan);
        assert_eq!(report.disposition, ModelFormDisposition::ApplicabilityUnknown);
        assert_eq!(report.combined_lower_bound_mwe, None);
    }

    #[test]
    fn duplicate_episode_measurement_artifact_or_model_run_is_refused() {
        for mutation in 0..4 {
            let mut plan = plan(19, 1.0);
            match mutation {
                0 => plan.calibration_residuals[1].episode_id =
                    plan.calibration_residuals[0].episode_id.clone(),
                1 => plan.calibration_residuals[1].measured_observation_id =
                    plan.calibration_residuals[0].measured_observation_id.clone(),
                2 => plan.calibration_residuals[1].measured_artifact_sha256 =
                    plan.calibration_residuals[0].measured_artifact_sha256.clone(),
                _ => plan.calibration_residuals[1].model_run_id =
                    plan.calibration_residuals[0].model_run_id.clone(),
            }
            assert_eq!(
                evaluate_model_form_discrepancy(&plan).disposition,
                ModelFormDisposition::NotEvaluable
            );
        }
    }

    #[test]
    fn calibration_used_for_model_or_feature_selection_is_leakage() {
        let mut plan = plan(19, 1.0);
        plan.calibration_residuals[0].used_for_feature_selection = true;
        let report = evaluate_model_form_discrepancy(&plan);
        assert_eq!(report.disposition, ModelFormDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&ModelFormDiagnostic::CalibrationLeakage));
    }

    #[test]
    fn synthetic_residuals_are_not_promoted_to_measured_discrepancy() {
        let mut plan = plan(19, 1.0);
        plan.calibration_residuals[0].evidence_class = ModelFormEvidenceClass::SyntheticFixture;
        let report = evaluate_model_form_discrepancy(&plan);
        assert_eq!(report.disposition, ModelFormDisposition::SyntheticOnly);
        assert!(report.to_net_export_receipt(&plan).is_none());
    }

    #[test]
    fn regime_geometry_boundary_or_observable_mismatch_is_refused() {
        let mutations: [fn(&mut CalibrationResidual); 4] = [
            |residual| residual.regime_ref = "different-regime".to_string(),
            |residual| residual.geometry_sha256 = sha('z'),
            |residual| residual.service_boundary_ref = "gross-electric".to_string(),
            |residual| residual.observable_id = "P_gross".to_string(),
        ];
        for mutation in mutations {
            let mut plan = plan(19, 1.0);
            mutation(&mut plan.calibration_residuals[0]);
            assert_eq!(
                evaluate_model_form_discrepancy(&plan).disposition,
                ModelFormDisposition::NotEvaluable
            );
        }
    }

    #[test]
    fn stale_target_applicability_or_output_identity_is_refused() {
        let mut plan = plan(19, 1.0);
        plan.applicability.input_sha256 = sha('z');
        assert_eq!(
            evaluate_model_form_discrepancy(&plan).disposition,
            ModelFormDisposition::NotEvaluable
        );
        let mut plan = plan(19, 1.0);
        plan.parametric_bound.output_sha256 = sha('z');
        assert_eq!(
            evaluate_model_form_discrepancy(&plan).disposition,
            ModelFormDisposition::NotEvaluable
        );
    }

    #[test]
    fn material_discrepancy_above_tolerance_blocks_combined_bound() {
        let report = evaluate_model_form_discrepancy(&plan(39, 30.0));
        assert_eq!(report.disposition, ModelFormDisposition::MaterialDiscrepancy);
        assert_eq!(report.combined_lower_bound_mwe, None);
    }

    #[test]
    fn material_tolerance_equality_obeys_inclusive_policy() {
        let mut plan = plan(39, 20.0);
        plan.profile.material_discrepancy_tolerance = 20.38;
        plan.profile.tolerance_inclusive = true;
        assert_eq!(
            evaluate_model_form_discrepancy(&plan).disposition,
            ModelFormDisposition::QualifiedWithinDeclaredDomain
        );
        plan.profile.tolerance_inclusive = false;
        assert_eq!(
            evaluate_model_form_discrepancy(&plan).disposition,
            ModelFormDisposition::MaterialDiscrepancy
        );
    }

    #[test]
    fn residual_order_does_not_change_conformal_allowance() {
        let mut first = plan(39, 1.0);
        let mut second = plan(39, 1.0);
        second.calibration_residuals.reverse();
        let a = evaluate_model_form_discrepancy(&first);
        let b = evaluate_model_form_discrepancy(&second);
        assert_eq!(a.conformal_rank_one_based, b.conformal_rank_one_based);
        assert_eq!(a.discrepancy_allowance_mwe, b.discrepancy_allowance_mwe);
    }

    #[test]
    fn combined_lower_bound_never_exceeds_parametric_lower_bound() {
        let report = evaluate_model_form_discrepancy(&plan(39, 1.0));
        assert!(report.combined_lower_bound_mwe.unwrap() <= report.parametric_lower_bound_mwe.unwrap());
    }

    #[test]
    fn missing_method_coverage_or_source_evidence_is_refused() {
        let mut plan = plan(19, 1.0);
        plan.parametric_bound.coverage_scope_ref.clear();
        assert_eq!(
            evaluate_model_form_discrepancy(&plan).disposition,
            ModelFormDisposition::NotEvaluable
        );
        let mut plan = plan(19, 1.0);
        plan.decision_receipt_sha256 = "invalid".to_string();
        assert_eq!(
            evaluate_model_form_discrepancy(&plan).disposition,
            ModelFormDisposition::NotEvaluable
        );
    }

    #[test]
    fn conformal_allowance_preserves_extreme_tail_observation() {
        let mut plan = plan(19, 1.0);
        plan.calibration_residuals[18].absolute_residual_mwe = 100.0;
        let report = evaluate_model_form_discrepancy(&plan);
        assert_eq!(report.discrepancy_allowance_mwe, Some(100.0));
        assert_eq!(report.disposition, ModelFormDisposition::MaterialDiscrepancy);
    }

    #[test]
    fn model_form_decision_produces_downstream_robust_receipt_only_when_qualified() {
        let plan = plan(39, 1.0);
        let report = evaluate_model_form_discrepancy(&plan);
        let receipt = report.to_net_export_receipt(&plan).unwrap();
        assert_eq!(
            receipt.method,
            NetExportBoundMethod::ParametricPlusModelFormDiscrepancy
        );
        assert_eq!(
            receipt.model_form_validity,
            ModelFormValidity::WithinDeclaredDomain
        );
        assert!(receipt.lower_bound_mwe < receipt.one_sided_coverage_level + 100.0);
        assert!(report.to_net_export_receipt(&plan).is_some());
    }

    #[test]
    fn reordered_residuals_produce_same_bound_and_order_statistic() {
        let mut first = plan(39, 1.0);
        let mut second = plan(39, 1.0);
        second.calibration_residuals.reverse();
        let a = evaluate_model_form_discrepancy(&first);
        let b = evaluate_model_form_discrepancy(&second);
        assert_eq!(a.residual_order_statistic_mwe, b.residual_order_statistic_mwe);
    }
}
