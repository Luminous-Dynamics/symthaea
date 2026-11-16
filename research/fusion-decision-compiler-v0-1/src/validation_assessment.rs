//! Quantitative blind-holdout predictive assessment.
//!
//! This module composes with validation_domain::validate_validation_domain.
//! Admission and predictive assessment remain separate: a successful split
//! audit proves declared separation/domain membership only; this module then
//! evaluates exact predictions against source-linked measured holdouts.
//!
//! It reports descriptive empirical errors and interval performance. It does
//! not fit models, calibrate intervals, infer independent samples, compute
//! confidence bounds, authenticate external artifact bytes, or prove
//! universal physical validity.

use std::collections::{BTreeMap, BTreeSet};

use super::ModelUse;
use super::validation_domain::{ValidationPlan, validate_validation_domain};

pub const VALIDATION_ASSESSMENT_VERSION: &str = "fusion-model-validation-assessment/0.1";

#[derive(Clone, Debug, PartialEq)]
pub struct ObservableValidationSpec {
    pub observable_id: String,
    pub unit_ref: String,
    pub service_boundary_ref: String,
    pub averaging_window_ref: String,
    /// Positive, physically meaningful scale fixed before holdout unsealing.
    pub normalization_scale: f64,
    /// Nominal prediction-interval coverage stated by the producer.
    pub nominal_interval_coverage: f64,
    pub interval_method_ref: String,
    pub interval_assumption_ref: String,
    /// Descriptive empirical coverage threshold; not a confidence bound.
    pub minimum_empirical_point_coverage: f64,
    pub maximum_normalized_rmse: f64,
    pub maximum_absolute_normalized_bias: f64,
    /// Prevents arbitrarily wide intervals from satisfying coverage by default.
    pub maximum_mean_normalized_interval_width: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValidationAssessmentProfile {
    pub profile_id: String,
    pub profile_sha256: String,
    pub domain_id: String,
    pub model_revision: String,
    pub regime_id: String,
    pub requested_use: ModelUse,
    pub minimum_holdout_count: usize,
    pub minimum_distinct_facilities: usize,
    pub minimum_distinct_operating_points: usize,
    pub observables: Vec<ObservableValidationSpec>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelRunIdentity {
    pub run_id: String,
    pub model_revision: String,
    pub input_sha256: String,
    pub output_sha256: String,
    pub run_receipt_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PredictedObservable {
    pub observable_id: String,
    pub model_run_id: String,
    pub model_output_sha256: String,
    pub interval_method_ref: String,
    pub interval_assumption_ref: String,
    pub unit_ref: String,
    pub service_boundary_ref: String,
    pub averaging_window_ref: String,
    pub predicted_value: f64,
    pub prediction_interval_lower: f64,
    pub prediction_interval_upper: f64,
    pub source_locator: String,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredObservable {
    pub observable_id: String,
    pub unit_ref: String,
    pub service_boundary_ref: String,
    pub averaging_window_ref: String,
    pub measured_value: f64,
    pub measurement_interval_lower: f64,
    pub measurement_interval_upper: f64,
    pub source_artifact_sha256: String,
    pub source_locator: String,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlindHoldoutAssessmentCase {
    pub plan: ValidationPlan,
    /// Explicit final-evaluation authorization/unseal evidence.
    pub unseal_receipt_ref: String,
    pub unseal_receipt_sha256: String,
    pub model_run: ModelRunIdentity,
    pub predictions: Vec<PredictedObservable>,
    pub measurements: Vec<MeasuredObservable>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValidationAssessmentCampaign {
    pub profile: ValidationAssessmentProfile,
    pub cases: Vec<BlindHoldoutAssessmentCase>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictiveAssessmentDisposition {
    /// Metrics meet the frozen profile on the finite admitted holdout set.
    WithinDeclaredAcceptance,
    /// Metrics violate one or more predeclared acceptance thresholds.
    OutsideDeclaredAcceptance,
    /// The point metrics are computable, but sample/domain diversity targets
    /// have not been met.
    InsufficientHoldoutEvidence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationAssessmentErrorKind {
    EmptyProfileIdentity,
    InvalidProfileDigest,
    EmptyDomainIdentity,
    EmptyModelRevision,
    ZeroMinimumHoldoutCount,
    ZeroMinimumFacilityCount,
    ZeroMinimumOperatingPointCount,
    EmptyObservableSet,
    DuplicateObservableSpec,
    InvalidObservableIdentity,
    InvalidObservableUnit,
    InvalidObservableBoundary,
    InvalidAveragingWindow,
    InvalidNormalizationScale,
    InvalidNominalCoverage,
    EmptyIntervalMethod,
    EmptyIntervalAssumption,
    PredictionRunIdentityMismatch,
    PredictionOutputDigestMismatch,
    PredictionIntervalMethodMismatch,
    PredictionIntervalAssumptionMismatch,
    ReusedUnsealReceipt,
    ReusedModelRunReceipt,
    InvalidMinimumCoverage,
    InvalidMaximumNormalizedRmse,
    InvalidMaximumNormalizedBias,
    InvalidMaximumIntervalWidth,
    NoHoldoutCases,
    HoldoutAdmissionRefused,
    HoldoutModelRevisionMismatch,
    HoldoutDomainMismatch,
    HoldoutUseMismatch,
    EmptyUnsealReceipt,
    InvalidUnsealReceiptDigest,
    EmptyModelRunIdentity,
    ModelRunRevisionMismatch,
    InvalidModelInputDigest,
    InvalidModelOutputDigest,
    EmptyModelRunReceipt,
    DuplicateHoldoutRecord,
    DuplicateHoldoutDataset,
    DuplicateHoldoutArtifact,
    DuplicateHoldoutEpisode,
    DuplicateModelRunId,
    ReusedModelOutputDigest,
    MissingPrediction,
    MissingMeasurement,
    DuplicatePrediction,
    DuplicateMeasurement,
    UnexpectedPredictionObservable,
    UnexpectedMeasurementObservable,
    NonFinitePrediction,
    NonFiniteMeasurement,
    NonFinitePredictionInterval,
    NonFiniteMeasurementInterval,
    InvertedPredictionInterval,
    InvertedMeasurementInterval,
    PredictionOutsideItsInterval,
    MeasurementOutsideItsInterval,
    PredictionUnitMismatch,
    MeasurementUnitMismatch,
    PredictionBoundaryMismatch,
    MeasurementBoundaryMismatch,
    PredictionAveragingWindowMismatch,
    MeasurementAveragingWindowMismatch,
    MeasurementArtifactMismatch,
    InvalidMeasurementArtifactDigest,
    EmptyPredictionLocator,
    EmptyMeasurementLocator,
    EmptyPredictionEvidence,
    EmptyMeasurementEvidence,
    DuplicatePredictionEvidence,
    DuplicateMeasurementEvidence,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationAssessmentError {
    pub kind: ValidationAssessmentErrorKind,
    pub case_id: Option<String>,
    pub observable_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PerCaseValidationResult {
    pub holdout_record_id: String,
    pub facility_id: String,
    pub case_id: String,
    pub operating_point_id: String,
    pub model_run_id: String,
    pub observable_id: String,
    pub measured_value: f64,
    pub predicted_value: f64,
    pub signed_error: f64,
    pub normalized_signed_error: f64,
    pub absolute_error: f64,
    pub normalized_absolute_error: f64,
    pub prediction_interval_lower: f64,
    pub prediction_interval_upper: f64,
    pub point_inside_prediction_interval: bool,
    pub measurement_interval_lower: f64,
    pub measurement_interval_upper: f64,
    pub measurement_interval_overlaps_prediction_interval: bool,
    pub measurement_artifact_sha256: String,
    pub prediction_evidence_ref: String,
    pub measurement_evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PerObservableValidationMetrics {
    pub observable_id: String,
    pub unit_ref: String,
    pub service_boundary_ref: String,
    pub averaging_window_ref: String,
    pub nominal_interval_coverage: f64,
    pub interval_method_ref: String,
    pub interval_assumption_ref: String,
    pub sample_count: usize,
    pub mean_signed_error: f64,
    pub mean_absolute_error: f64,
    pub root_mean_square_error: f64,
    pub normalized_bias: f64,
    pub normalized_rmse: f64,
    pub maximum_absolute_normalized_error: f64,
    pub empirical_point_interval_coverage: f64,
    pub measurement_interval_overlap_coverage: f64,
    pub mean_normalized_prediction_interval_width: f64,
    pub meets_declared_acceptance: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValidationAssessmentReport {
    pub contract_version: &'static str,
    pub profile_id: String,
    pub profile_sha256: String,
    pub domain_id: String,
    pub model_revision: String,
    pub requested_use: ModelUse,
    pub disposition: PredictiveAssessmentDisposition,
    pub holdout_count: usize,
    pub distinct_facility_count: usize,
    pub distinct_case_count: usize,
    pub distinct_operating_point_count: usize,
    pub calibration_record_count: usize,
    pub calibration_episode_count: usize,
    pub admitted_holdout_record_ids: Vec<String>,
    pub per_observable_metrics: Vec<PerObservableValidationMetrics>,
    pub per_case_results: Vec<PerCaseValidationResult>,
    pub unseal_receipt_refs: Vec<String>,
    pub model_run_receipt_refs: Vec<String>,
    pub measurement_evidence_refs: Vec<String>,
    pub prediction_evidence_refs: Vec<String>,
    pub claim_ceiling: &'static str,
}

/// Assess predictions only after every underlying blind-holdout plan passes
/// the independent split-integrity/domain-admission contract.
///
/// A campaign is rejected in full if any holdout is stale, exposed, missing,
/// duplicated or dimensionally incompatible. No missing value is imputed.
pub fn assess_blind_holdout_campaign(
    campaign: &ValidationAssessmentCampaign,
) -> Result<ValidationAssessmentReport, Vec<ValidationAssessmentError>> {
    let mut errors = validate_profile(&campaign.profile);
    if campaign.cases.is_empty() {
        push_error(
            &mut errors,
            ValidationAssessmentErrorKind::NoHoldoutCases,
            None,
            None,
        );
    }

    let mut holdout_records = BTreeSet::new();
    let mut holdout_datasets = BTreeSet::new();
    let mut holdout_artifacts = BTreeSet::new();
    let mut holdout_episodes = BTreeSet::new();
    let mut model_run_ids = BTreeSet::new();
    let mut model_output_digests = BTreeSet::new();
    let mut facility_ids = BTreeSet::new();
    let mut case_ids = BTreeSet::new();
    let mut operating_point_ids = BTreeSet::new();
    let mut admitted_plans = Vec::new();
    let mut case_results = Vec::new();
    let mut unseal_refs = BTreeSet::new();
    let mut run_receipt_refs = BTreeSet::new();
    let mut measurement_refs = BTreeSet::new();
    let mut prediction_refs = BTreeSet::new();

    for case in &campaign.cases {
        let observation = &case.plan.blind_holdout.observation;
        let case_id = observation.record_id.clone();
        let model_revision = &campaign.profile.model_revision;

        match validate_validation_domain(&case.plan) {
            Ok(audit) => admitted_plans.push(audit),
            Err(_) => push_error(
                &mut errors,
                ValidationAssessmentErrorKind::HoldoutAdmissionRefused,
                Some(case_id.clone()),
                None,
            ),
        }

        if case.plan.model_revision != *model_revision
            || case.plan.domain.model_revision != *model_revision
            || case.model_run.model_revision != *model_revision
        {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::HoldoutModelRevisionMismatch,
                Some(case_id.clone()),
                None,
            );
        }
        if case.plan.domain.domain_id != campaign.profile.domain_id
            || case.plan.domain.regime_id != campaign.profile.regime_id
        {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::HoldoutDomainMismatch,
                Some(case_id.clone()),
                None,
            );
        }
        if case.plan.requested_use != campaign.profile.requested_use {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::HoldoutUseMismatch,
                Some(case_id.clone()),
                None,
            );
        }

        if case.unseal_receipt_ref.trim().is_empty() {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::EmptyUnsealReceipt,
                Some(case_id.clone()),
                None,
            );
        }
        if !is_sha256(&case.unseal_receipt_sha256) {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::InvalidUnsealReceiptDigest,
                Some(case_id.clone()),
                None,
            );
        }
        if case.model_run.run_id.trim().is_empty()
            || case.model_run.model_revision.trim().is_empty()
        {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::EmptyModelRunIdentity,
                Some(case_id.clone()),
                None,
            );
        }
        if !is_sha256(&case.model_run.input_sha256) {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::InvalidModelInputDigest,
                Some(case_id.clone()),
                None,
            );
        }
        if !is_sha256(&case.model_run.output_sha256) {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::InvalidModelOutputDigest,
                Some(case_id.clone()),
                None,
            );
        }
        if case.model_run.run_receipt_ref.trim().is_empty() {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::EmptyModelRunReceipt,
                Some(case_id.clone()),
                None,
            );
        }

        if !holdout_records.insert(observation.record_id.clone()) {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::DuplicateHoldoutRecord,
                Some(case_id.clone()),
                None,
            );
        }
        if !holdout_datasets.insert(observation.dataset_id.clone()) {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::DuplicateHoldoutDataset,
                Some(case_id.clone()),
                None,
            );
        }
        if !holdout_artifacts.insert(observation.source_artifact_sha256.clone()) {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::DuplicateHoldoutArtifact,
                Some(case_id.clone()),
                None,
            );
        }
        if !case.model_run.run_id.trim().is_empty()
            && !model_run_ids.insert(case.model_run.run_id.clone())
        {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::DuplicateModelRunId,
                Some(case_id.clone()),
                None,
            );
        }
        if is_sha256(&case.model_run.output_sha256)
            && !model_output_digests.insert(case.model_run.output_sha256.clone())
        {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::ReusedModelOutputDigest,
                Some(case_id.clone()),
                None,
            );
        }
        facility_ids.insert(observation.facility_id.clone());
        case_ids.insert(observation.case_id.clone());
        operating_point_ids.insert(observation.operating_point_id.clone());

        for episode in &observation.episode_ids {
            if !holdout_episodes.insert(episode.clone()) {
                push_error(
                    &mut errors,
                    ValidationAssessmentErrorKind::DuplicateHoldoutEpisode,
                    Some(case_id.clone()),
                    None,
                );
            }
        }

        if !case.unseal_receipt_ref.trim().is_empty()
            && !unseal_refs.insert(case.unseal_receipt_ref.clone())
        {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::ReusedUnsealReceipt,
                Some(case_id.clone()),
                None,
            );
        }
        if !case.model_run.run_receipt_ref.trim().is_empty()
            && !run_receipt_refs.insert(case.model_run.run_receipt_ref.clone())
        {
            push_error(
                &mut errors,
                ValidationAssessmentErrorKind::ReusedModelRunReceipt,
                Some(case_id.clone()),
                None,
            );
        }

        validate_case_observables(
            &campaign.profile,
            case,
            &mut case_results,
            &mut errors,
            &mut measurement_refs,
            &mut prediction_refs,
        );
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    case_results.sort_by(|left, right| {
        (&left.holdout_record_id, &left.observable_id)
            .cmp(&(&right.holdout_record_id, &right.observable_id))
    });

    let mut metrics = Vec::new();
    let mut any_metric_failed = false;
    for spec in &campaign.profile.observables {
        let selected: Vec<&PerCaseValidationResult> = case_results
            .iter()
            .filter(|result| result.observable_id == spec.observable_id)
            .collect();
        let count = selected.len();
        let n = count as f64;
        let signed_sum: f64 = selected.iter().map(|result| result.signed_error).sum();
        let absolute_sum: f64 = selected.iter().map(|result| result.absolute_error).sum();
        let squared_sum: f64 = selected
            .iter()
            .map(|result| result.signed_error * result.signed_error)
            .sum();
        let normalized_width_sum: f64 = selected
            .iter()
            .map(|result| {
                (result.prediction_interval_upper - result.prediction_interval_lower)
                    / spec.normalization_scale
            })
            .sum();
        let point_coverage_count = selected
            .iter()
            .filter(|result| result.point_inside_prediction_interval)
            .count();
        let overlap_count = selected
            .iter()
            .filter(|result| result.measurement_interval_overlaps_prediction_interval)
            .count();
        let normalized_bias = (signed_sum / n) / spec.normalization_scale;
        let normalized_rmse = (squared_sum / n).sqrt() / spec.normalization_scale;
        let point_coverage = point_coverage_count as f64 / n;
        let interval_width = normalized_width_sum / n;
        let maximum_error = selected
            .iter()
            .map(|result| result.normalized_absolute_error)
            .fold(0.0_f64, f64::max);
        let meets = point_coverage >= spec.minimum_empirical_point_coverage
            && normalized_rmse <= spec.maximum_normalized_rmse
            && normalized_bias.abs() <= spec.maximum_absolute_normalized_bias
            && interval_width <= spec.maximum_mean_normalized_interval_width;

        any_metric_failed |= !meets;
        metrics.push(PerObservableValidationMetrics {
            observable_id: spec.observable_id.clone(),
            unit_ref: spec.unit_ref.clone(),
            service_boundary_ref: spec.service_boundary_ref.clone(),
            averaging_window_ref: spec.averaging_window_ref.clone(),
            nominal_interval_coverage: spec.nominal_interval_coverage,
            interval_method_ref: spec.interval_method_ref.clone(),
            interval_assumption_ref: spec.interval_assumption_ref.clone(),
            sample_count: count,
            mean_signed_error: signed_sum / n,
            mean_absolute_error: absolute_sum / n,
            root_mean_square_error: (squared_sum / n).sqrt(),
            normalized_bias,
            normalized_rmse,
            maximum_absolute_normalized_error: maximum_error,
            empirical_point_interval_coverage: point_coverage,
            measurement_interval_overlap_coverage: overlap_count as f64 / n,
            mean_normalized_prediction_interval_width: interval_width,
            meets_declared_acceptance: meets,
        });
    }

    metrics.sort_by(|left, right| left.observable_id.cmp(&right.observable_id));
    let insufficient = campaign.cases.len() < campaign.profile.minimum_holdout_count
        || facility_ids.len() < campaign.profile.minimum_distinct_facilities
        || operating_point_ids.len() < campaign.profile.minimum_distinct_operating_points;
    let disposition = if any_metric_failed {
        PredictiveAssessmentDisposition::OutsideDeclaredAcceptance
    } else if insufficient {
        PredictiveAssessmentDisposition::InsufficientHoldoutEvidence
    } else {
        PredictiveAssessmentDisposition::WithinDeclaredAcceptance
    };

    let mut admitted_holdout_record_ids: Vec<String> =
        campaign.cases.iter().map(|case| case.plan.blind_holdout.observation.record_id.clone()).collect();
    admitted_holdout_record_ids.sort();

    Ok(ValidationAssessmentReport {
        contract_version: VALIDATION_ASSESSMENT_VERSION,
        profile_id: campaign.profile.profile_id.clone(),
        profile_sha256: campaign.profile.profile_sha256.clone(),
        domain_id: campaign.profile.domain_id.clone(),
        model_revision: campaign.profile.model_revision.clone(),
        requested_use: campaign.profile.requested_use,
        disposition,
        holdout_count: campaign.cases.len(),
        distinct_facility_count: facility_ids.len(),
        distinct_case_count: case_ids.len(),
        distinct_operating_point_count: operating_point_ids.len(),
        calibration_record_count: admitted_plans
            .iter()
            .map(|audit| audit.calibration_record_count)
            .sum(),
        calibration_episode_count: admitted_plans
            .iter()
            .map(|audit| audit.calibration_episode_count)
            .sum(),
        admitted_holdout_record_ids,
        per_observable_metrics: metrics,
        per_case_results: case_results,
        unseal_receipt_refs: unseal_refs.into_iter().collect(),
        model_run_receipt_refs: run_receipt_refs.into_iter().collect(),
        measurement_evidence_refs: measurement_refs.into_iter().collect(),
        prediction_evidence_refs: prediction_refs.into_iter().collect(),
        claim_ceiling: "predictive performance on finite admitted blind holdouts only",
    })
}

fn validate_profile(
    profile: &ValidationAssessmentProfile,
) -> Vec<ValidationAssessmentError> {
    let mut errors = Vec::new();
    if profile.profile_id.trim().is_empty() {
        push_error(&mut errors, ValidationAssessmentErrorKind::EmptyProfileIdentity, None, None);
    }
    if !is_sha256(&profile.profile_sha256) {
        push_error(&mut errors, ValidationAssessmentErrorKind::InvalidProfileDigest, None, None);
    }
    if profile.domain_id.trim().is_empty() {
        push_error(&mut errors, ValidationAssessmentErrorKind::EmptyDomainIdentity, None, None);
    }
    if profile.model_revision.trim().is_empty() {
        push_error(&mut errors, ValidationAssessmentErrorKind::EmptyModelRevision, None, None);
    }
    if profile.regime_id.trim().is_empty() {
        push_error(&mut errors, ValidationAssessmentErrorKind::EmptyDomainIdentity, None, None);
    }
    if profile.minimum_holdout_count == 0 {
        push_error(&mut errors, ValidationAssessmentErrorKind::ZeroMinimumHoldoutCount, None, None);
    }
    if profile.minimum_distinct_facilities == 0 {
        push_error(&mut errors, ValidationAssessmentErrorKind::ZeroMinimumFacilityCount, None, None);
    }
    if profile.minimum_distinct_operating_points == 0 {
        push_error(&mut errors, ValidationAssessmentErrorKind::ZeroMinimumOperatingPointCount, None, None);
    }
    if profile.observables.is_empty() {
        push_error(&mut errors, ValidationAssessmentErrorKind::EmptyObservableSet, None, None);
    }

    let mut seen = BTreeSet::new();
    for spec in &profile.observables {
        let observable = Some(spec.observable_id.clone());
        if spec.observable_id.trim().is_empty() {
            push_error(&mut errors, ValidationAssessmentErrorKind::InvalidObservableIdentity, None, observable.clone());
        }
        if !seen.insert(spec.observable_id.clone()) {
            push_error(&mut errors, ValidationAssessmentErrorKind::DuplicateObservableSpec, None, observable.clone());
        }
        if spec.unit_ref.trim().is_empty() {
            push_error(&mut errors, ValidationAssessmentErrorKind::InvalidObservableUnit, None, observable.clone());
        }
        if spec.service_boundary_ref.trim().is_empty() {
            push_error(&mut errors, ValidationAssessmentErrorKind::InvalidObservableBoundary, None, observable.clone());
        }
        if spec.averaging_window_ref.trim().is_empty() {
            push_error(&mut errors, ValidationAssessmentErrorKind::InvalidAveragingWindow, None, observable.clone());
        }
        if !spec.normalization_scale.is_finite() || spec.normalization_scale <= 0.0 {
            push_error(&mut errors, ValidationAssessmentErrorKind::InvalidNormalizationScale, None, observable.clone());
        }
        if !valid_probability(spec.nominal_interval_coverage) {
            push_error(&mut errors, ValidationAssessmentErrorKind::InvalidNominalCoverage, None, observable.clone());
        }
        if spec.interval_method_ref.trim().is_empty() {
            push_error(&mut errors, ValidationAssessmentErrorKind::EmptyIntervalMethod, None, observable.clone());
        }
        if spec.interval_assumption_ref.trim().is_empty() {
            push_error(&mut errors, ValidationAssessmentErrorKind::EmptyIntervalAssumption, None, observable.clone());
        }
        if !valid_probability(spec.minimum_empirical_point_coverage) {
            push_error(&mut errors, ValidationAssessmentErrorKind::InvalidMinimumCoverage, None, observable.clone());
        }
        if !spec.maximum_normalized_rmse.is_finite() || spec.maximum_normalized_rmse < 0.0 {
            push_error(&mut errors, ValidationAssessmentErrorKind::InvalidMaximumNormalizedRmse, None, observable.clone());
        }
        if !spec.maximum_absolute_normalized_bias.is_finite() || spec.maximum_absolute_normalized_bias < 0.0 {
            push_error(&mut errors, ValidationAssessmentErrorKind::InvalidMaximumNormalizedBias, None, observable.clone());
        }
        if !spec.maximum_mean_normalized_interval_width.is_finite()
            || spec.maximum_mean_normalized_interval_width < 0.0
        {
            push_error(&mut errors, ValidationAssessmentErrorKind::InvalidMaximumIntervalWidth, None, observable);
        }
    }
    errors
}

fn validate_case_observables(
    profile: &ValidationAssessmentProfile,
    case: &BlindHoldoutAssessmentCase,
    results: &mut Vec<PerCaseValidationResult>,
    errors: &mut Vec<ValidationAssessmentError>,
    measurement_refs: &mut BTreeSet<String>,
    prediction_refs: &mut BTreeSet<String>,
) {
    let observation = &case.plan.blind_holdout.observation;
    let case_id = observation.record_id.clone();
    let expected: BTreeSet<&str> = profile
        .observables
        .iter()
        .map(|spec| spec.observable_id.as_str())
        .collect();
    let mut predictions = BTreeMap::new();
    let mut measurements = BTreeMap::new();

    for prediction in &case.predictions {
        if !expected.contains(prediction.observable_id.as_str()) {
            push_error(
                errors,
                ValidationAssessmentErrorKind::UnexpectedPredictionObservable,
                Some(case_id.clone()),
                Some(prediction.observable_id.clone()),
            );
        }
        if predictions.insert(prediction.observable_id.as_str(), prediction).is_some() {
            push_error(
                errors,
                ValidationAssessmentErrorKind::DuplicatePrediction,
                Some(case_id.clone()),
                Some(prediction.observable_id.clone()),
            );
        }
    }
    for measurement in &case.measurements {
        if !expected.contains(measurement.observable_id.as_str()) {
            push_error(
                errors,
                ValidationAssessmentErrorKind::UnexpectedMeasurementObservable,
                Some(case_id.clone()),
                Some(measurement.observable_id.clone()),
            );
        }
        if measurements.insert(measurement.observable_id.as_str(), measurement).is_some() {
            push_error(
                errors,
                ValidationAssessmentErrorKind::DuplicateMeasurement,
                Some(case_id.clone()),
                Some(measurement.observable_id.clone()),
            );
        }
    }

    for spec in &profile.observables {
        let predicted = predictions.get(spec.observable_id.as_str()).copied();
        let measured = measurements.get(spec.observable_id.as_str()).copied();
        let Some(predicted) = predicted else {
            push_error(
                errors,
                ValidationAssessmentErrorKind::MissingPrediction,
                Some(case_id.clone()),
                Some(spec.observable_id.clone()),
            );
            continue;
        };
        let Some(measured) = measured else {
            push_error(
                errors,
                ValidationAssessmentErrorKind::MissingMeasurement,
                Some(case_id.clone()),
                Some(spec.observable_id.clone()),
            );
            continue;
        };

        if predicted.model_run_id != case.model_run.run_id {
            push_error(errors, ValidationAssessmentErrorKind::PredictionRunIdentityMismatch, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if predicted.model_output_sha256 != case.model_run.output_sha256
            || !is_sha256(&predicted.model_output_sha256)
        {
            push_error(errors, ValidationAssessmentErrorKind::PredictionOutputDigestMismatch, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if predicted.interval_method_ref != spec.interval_method_ref {
            push_error(errors, ValidationAssessmentErrorKind::PredictionIntervalMethodMismatch, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if predicted.interval_assumption_ref != spec.interval_assumption_ref {
            push_error(errors, ValidationAssessmentErrorKind::PredictionIntervalAssumptionMismatch, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if !predicted.predicted_value.is_finite() {
            push_error(errors, ValidationAssessmentErrorKind::NonFinitePrediction, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if !measured.measured_value.is_finite() {
            push_error(errors, ValidationAssessmentErrorKind::NonFiniteMeasurement, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if !predicted.prediction_interval_lower.is_finite() || !predicted.prediction_interval_upper.is_finite() {
            push_error(errors, ValidationAssessmentErrorKind::NonFinitePredictionInterval, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if !measured.measurement_interval_lower.is_finite() || !measured.measurement_interval_upper.is_finite() {
            push_error(errors, ValidationAssessmentErrorKind::NonFiniteMeasurementInterval, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if predicted.prediction_interval_lower > predicted.prediction_interval_upper {
            push_error(errors, ValidationAssessmentErrorKind::InvertedPredictionInterval, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if measured.measurement_interval_lower > measured.measurement_interval_upper {
            push_error(errors, ValidationAssessmentErrorKind::InvertedMeasurementInterval, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if predicted.predicted_value < predicted.prediction_interval_lower
            || predicted.predicted_value > predicted.prediction_interval_upper
        {
            push_error(errors, ValidationAssessmentErrorKind::PredictionOutsideItsInterval, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if measured.measured_value < measured.measurement_interval_lower
            || measured.measured_value > measured.measurement_interval_upper
        {
            push_error(errors, ValidationAssessmentErrorKind::MeasurementOutsideItsInterval, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if predicted.unit_ref != spec.unit_ref {
            push_error(errors, ValidationAssessmentErrorKind::PredictionUnitMismatch, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if measured.unit_ref != spec.unit_ref {
            push_error(errors, ValidationAssessmentErrorKind::MeasurementUnitMismatch, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if predicted.service_boundary_ref != spec.service_boundary_ref {
            push_error(errors, ValidationAssessmentErrorKind::PredictionBoundaryMismatch, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if measured.service_boundary_ref != spec.service_boundary_ref {
            push_error(errors, ValidationAssessmentErrorKind::MeasurementBoundaryMismatch, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if predicted.averaging_window_ref != spec.averaging_window_ref {
            push_error(errors, ValidationAssessmentErrorKind::PredictionAveragingWindowMismatch, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if measured.averaging_window_ref != spec.averaging_window_ref {
            push_error(errors, ValidationAssessmentErrorKind::MeasurementAveragingWindowMismatch, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if measured.source_artifact_sha256 != observation.source_artifact_sha256 {
            push_error(errors, ValidationAssessmentErrorKind::MeasurementArtifactMismatch, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if !is_sha256(&measured.source_artifact_sha256) {
            push_error(errors, ValidationAssessmentErrorKind::InvalidMeasurementArtifactDigest, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if predicted.source_locator.trim().is_empty() {
            push_error(errors, ValidationAssessmentErrorKind::EmptyPredictionLocator, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if measured.source_locator.trim().is_empty() {
            push_error(errors, ValidationAssessmentErrorKind::EmptyMeasurementLocator, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if predicted.evidence_ref.trim().is_empty() {
            push_error(errors, ValidationAssessmentErrorKind::EmptyPredictionEvidence, Some(case_id.clone()), Some(spec.observable_id.clone()));
        } else if !prediction_refs.insert(predicted.evidence_ref.clone()) {
            push_error(errors, ValidationAssessmentErrorKind::DuplicatePredictionEvidence, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }
        if measured.evidence_ref.trim().is_empty() {
            push_error(errors, ValidationAssessmentErrorKind::EmptyMeasurementEvidence, Some(case_id.clone()), Some(spec.observable_id.clone()));
        } else if !measurement_refs.insert(measured.evidence_ref.clone()) {
            push_error(errors, ValidationAssessmentErrorKind::DuplicateMeasurementEvidence, Some(case_id.clone()), Some(spec.observable_id.clone()));
        }

        if [
            predicted.predicted_value,
            measured.measured_value,
            predicted.prediction_interval_lower,
            predicted.prediction_interval_upper,
            measured.measurement_interval_lower,
            measured.measurement_interval_upper,
        ]
        .iter()
        .all(|value| value.is_finite())
            && spec.normalization_scale.is_finite()
            && spec.normalization_scale > 0.0
            && predicted.unit_ref == spec.unit_ref
            && measured.unit_ref == spec.unit_ref
            && predicted.service_boundary_ref == spec.service_boundary_ref
            && measured.service_boundary_ref == spec.service_boundary_ref
            && predicted.averaging_window_ref == spec.averaging_window_ref
            && measured.averaging_window_ref == spec.averaging_window_ref
            && !predicted.source_locator.trim().is_empty()
            && !measured.source_locator.trim().is_empty()
            && !predicted.evidence_ref.trim().is_empty()
            && !measured.evidence_ref.trim().is_empty()
            && predicted.model_run_id == case.model_run.run_id
            && predicted.model_output_sha256 == case.model_run.output_sha256
            && predicted.interval_method_ref == spec.interval_method_ref
            && predicted.interval_assumption_ref == spec.interval_assumption_ref
            && measured.source_artifact_sha256 == observation.source_artifact_sha256
            && is_sha256(&measured.source_artifact_sha256)
            && predicted.prediction_interval_lower <= predicted.predicted_value
            && predicted.predicted_value <= predicted.prediction_interval_upper
            && measured.measurement_interval_lower <= measured.measured_value
            && measured.measured_value <= measured.measurement_interval_upper
        {
            let signed_error = predicted.predicted_value - measured.measured_value;
            let normalized_signed_error = signed_error / spec.normalization_scale;
            let absolute_error = signed_error.abs();
            let measurement_interval_overlaps_prediction_interval =
                measured.measurement_interval_lower <= predicted.prediction_interval_upper
                && predicted.prediction_interval_lower <= measured.measurement_interval_upper;
            results.push(PerCaseValidationResult {
                holdout_record_id: case_id.clone(),
                facility_id: observation.facility_id.clone(),
                case_id: observation.case_id.clone(),
                operating_point_id: observation.operating_point_id.clone(),
                model_run_id: case.model_run.run_id.clone(),
                observable_id: spec.observable_id.clone(),
                measured_value: measured.measured_value,
                predicted_value: predicted.predicted_value,
                signed_error,
                normalized_signed_error,
                absolute_error,
                normalized_absolute_error: absolute_error / spec.normalization_scale,
                prediction_interval_lower: predicted.prediction_interval_lower,
                prediction_interval_upper: predicted.prediction_interval_upper,
                point_inside_prediction_interval: measured.measured_value >= predicted.prediction_interval_lower
                    && measured.measured_value <= predicted.prediction_interval_upper,
                measurement_interval_lower: measured.measurement_interval_lower,
                measurement_interval_upper: measured.measurement_interval_upper,
                measurement_interval_overlaps_prediction_interval,
                measurement_artifact_sha256: measured.source_artifact_sha256.clone(),
                prediction_evidence_ref: predicted.evidence_ref.clone(),
                measurement_evidence_ref: measured.evidence_ref.clone(),
            });
        }
    }
}

fn push_error(
    errors: &mut Vec<ValidationAssessmentError>,
    kind: ValidationAssessmentErrorKind,
    case_id: Option<String>,
    observable_id: Option<String>,
) {
    errors.push(ValidationAssessmentError {
        kind,
        case_id,
        observable_id,
    });
}

fn valid_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EvidenceClass;
    use crate::validation_domain::{
        BlindHoldoutMeasurement, CalibrationExposure, HoldoutExposure, MeasuredObservation,
        ValidationDomain,
    };

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn profile() -> ValidationAssessmentProfile {
        ValidationAssessmentProfile {
            profile_id: "fusion-validation-profile-v1".to_string(),
            profile_sha256: sha('a'),
            domain_id: "tokamak-hmode-domain-v1".to_string(),
            model_revision: "transport-model-r17".to_string(),
            regime_id: "tokamak-hmode".to_string(),
            requested_use: ModelUse::SingleStepPrediction,
            minimum_holdout_count: 3,
            minimum_distinct_facilities: 2,
            minimum_distinct_operating_points: 3,
            observables: vec![ObservableValidationSpec {
                observable_id: "net-heat-flux".to_string(),
                unit_ref: "MW/m2".to_string(),
                service_boundary_ref: "divertor-target-surface-v1".to_string(),
                averaging_window_ref: "steady-state-1s-v1".to_string(),
                normalization_scale: 100.0,
                nominal_interval_coverage: 0.95,
                minimum_empirical_point_coverage: 0.60,
                maximum_normalized_rmse: 0.20,
                maximum_absolute_normalized_bias: 0.10,
                maximum_mean_normalized_interval_width: 0.25,
                interval_method_ref: "synthetic-empirical-quantile-band-v1".to_string(),
                interval_assumption_ref: "synthetic-known-answer-fixture-v1".to_string(),
            }],
        }
    }

    fn observation(
        id: &str,
        facility: &str,
        case_id: &str,
        operating_point: &str,
        dataset_id: &str,
        artifact_sha: char,
        episode_id: &str,
    ) -> MeasuredObservation {
        MeasuredObservation {
            record_id: id.to_string(),
            evidence_class: EvidenceClass::MeasuredObservation,
            dataset_id: dataset_id.to_string(),
            source_artifact_sha256: sha(artifact_sha),
            regime_id: "tokamak-hmode".to_string(),
            geometry_sha256: sha('b'),
            facility_id: facility.to_string(),
            case_id: case_id.to_string(),
            operating_point_id: operating_point.to_string(),
            episode_ids: vec![episode_id.to_string()],
            diagnostic_revision: "diagnostics-r3".to_string(),
            measurement_procedure_ref: "measurement-procedure-v1".to_string(),
            diagnostic_calibration_ref: "diagnostic-calibration-v1".to_string(),
            measurement_uncertainty_ref: "measurement-uncertainty-v1".to_string(),
        }
    }

    fn campaign_case(index: usize) -> BlindHoldoutAssessmentCase {
        let holdout_id = format!("holdout-{index}");
        let artifact_char = char::from(b'd' + index as u8);
        let calibration = CalibrationExposure {
            record_id: format!("calibration-{index}"),
            dataset_id: format!("calibration-dataset-{index}"),
            source_artifact_sha256: sha('c'),
            regime_id: "tokamak-hmode".to_string(),
            geometry_sha256: sha('b'),
            episode_ids: vec![format!("calibration-episode-{index}")],
            preprocessing_fit_id: "calibration-only-preprocessing-v1".to_string(),
        };
        let test_stand = observation(
            &format!("test-stand-{index}"),
            "test-stand-facility",
            "test-stand-case",
            "test-stand-operating-point",
            &format!("test-stand-dataset-{index}"),
            'z',
            &format!("test-stand-episode-{index}"),
        );
        let holdout_observation = observation(
            &holdout_id,
            if index == 0 { "facility-a" } else { "facility-b" },
            &format!("validation-case-{index}"),
            &format!("operating-point-{index}"),
            &format!("validation-dataset-{index}"),
            artifact_char,
            &format!("validation-episode-{index}"),
        );
        let plan = ValidationPlan {
            model_revision: "transport-model-r17".to_string(),
            requested_use: ModelUse::SingleStepPrediction,
            domain: ValidationDomain {
                domain_id: "tokamak-hmode-domain-v1".to_string(),
                model_revision: "transport-model-r17".to_string(),
                regime_id: "tokamak-hmode".to_string(),
                supported_uses: vec![ModelUse::SingleStepPrediction],
                allowed_geometry_sha256: vec![sha('b')],
                allowed_facility_ids: vec![
                    "test-stand-facility".to_string(),
                    "facility-a".to_string(),
                    "facility-b".to_string(),
                ],
                allowed_case_ids: vec![
                    "test-stand-case".to_string(),
                    format!("validation-case-{index}"),
                ],
                allowed_operating_point_ids: vec![
                    "test-stand-operating-point".to_string(),
                    format!("operating-point-{index}"),
                ],
            },
            calibration: vec![calibration],
            test_stand,
            blind_holdout: BlindHoldoutMeasurement {
                observation: holdout_observation,
                exposure: HoldoutExposure {
                    used_for_training: false,
                    used_for_normalization_fit: false,
                    used_for_feature_selection: false,
                    used_for_hyperparameter_selection: false,
                    used_for_model_or_policy_selection: false,
                    used_for_calibration: false,
                },
                withheld_until_final_evaluation: true,
            },
        };
        let (measured, predicted, lower, upper) = match index {
            0 => (100.0, 105.0, 90.0, 110.0),
            1 => (120.0, 118.0, 110.0, 125.0),
            _ => (80.0, 100.0, 95.0, 105.0),
        };
        BlindHoldoutAssessmentCase {
            plan,
            unseal_receipt_ref: format!("final-unseal-receipt-{index}"),
            unseal_receipt_sha256: sha('e'),
            model_run: ModelRunIdentity {
                run_id: format!("model-run-{index}"),
                model_revision: "transport-model-r17".to_string(),
                input_sha256: sha(char::from(b'g' + index as u8)),
                output_sha256: sha(char::from(b'j' + index as u8)),
                run_receipt_ref: format!("model-run-receipt-{index}"),
            },
            predictions: vec![PredictedObservable {
                observable_id: "net-heat-flux".to_string(),
                model_run_id: format!("model-run-{index}"),
                model_output_sha256: sha(char::from(b'j' + index as u8)),
                interval_method_ref: "synthetic-empirical-quantile-band-v1".to_string(),
                interval_assumption_ref: "synthetic-known-answer-fixture-v1".to_string(),
                unit_ref: "MW/m2".to_string(),
                service_boundary_ref: "divertor-target-surface-v1".to_string(),
                averaging_window_ref: "steady-state-1s-v1".to_string(),
                predicted_value: predicted,
                prediction_interval_lower: lower,
                prediction_interval_upper: upper,
                source_locator: format!("prediction[{index}].net-heat-flux"),
                evidence_ref: format!("prediction-evidence-{index}"),
            }],
            measurements: vec![MeasuredObservable {
                observable_id: "net-heat-flux".to_string(),
                unit_ref: "MW/m2".to_string(),
                service_boundary_ref: "divertor-target-surface-v1".to_string(),
                averaging_window_ref: "steady-state-1s-v1".to_string(),
                measured_value: measured,
                measurement_interval_lower: measured - 2.0,
                measurement_interval_upper: measured + 2.0,
                source_artifact_sha256: sha(artifact_char),
                source_locator: format!("measurement[{index}].net-heat-flux"),
                evidence_ref: format!("measurement-evidence-{index}"),
            }],
        }
    }

    fn campaign() -> ValidationAssessmentCampaign {
        ValidationAssessmentCampaign {
            profile: profile(),
            cases: vec![campaign_case(0), campaign_case(1), campaign_case(2)],
        }
    }

    #[test]
    fn blind_holdout_campaign_reports_error_and_interval_metrics() {
        let report = assess_blind_holdout_campaign(&campaign()).unwrap();
        assert_eq!(
            report.disposition,
            PredictiveAssessmentDisposition::WithinDeclaredAcceptance
        );
        assert_eq!(report.holdout_count, 3);
        assert_eq!(report.distinct_facility_count, 2);
        assert_eq!(report.distinct_operating_point_count, 3);
        let metric = &report.per_observable_metrics[0];
        assert_eq!(metric.sample_count, 3);
        assert_eq!(metric.mean_signed_error, 23.0 / 3.0);
        assert_eq!(metric.mean_absolute_error, 9.0);
        assert!((metric.root_mean_square_error - 143.0_f64.sqrt()).abs() < 1e-12);
        assert!((metric.empirical_point_interval_coverage - 2.0 / 3.0).abs() < 1e-12);
        assert!((metric.mean_normalized_prediction_interval_width - 0.15).abs() < 1e-12);
        assert_eq!(report.per_case_results.len(), 3);
    }

    #[test]
    fn insufficient_holdout_count_does_not_claim_acceptance() {
        let mut source = campaign();
        source.cases.pop();
        let report = assess_blind_holdout_campaign(&source).unwrap();
        assert_eq!(
            report.disposition,
            PredictiveAssessmentDisposition::InsufficientHoldoutEvidence
        );
    }

    #[test]
    fn low_empirical_coverage_fails_declared_acceptance() {
        let mut source = campaign();
        source.profile.observables[0].minimum_empirical_point_coverage = 0.90;
        let report = assess_blind_holdout_campaign(&source).unwrap();
        assert_eq!(
            report.disposition,
            PredictiveAssessmentDisposition::OutsideDeclaredAcceptance
        );
    }

    #[test]
    fn overly_wide_prediction_interval_fails_width_guard() {
        let mut source = campaign();
        source.profile.observables[0].maximum_mean_normalized_interval_width = 0.10;
        let report = assess_blind_holdout_campaign(&source).unwrap();
        assert_eq!(
            report.disposition,
            PredictiveAssessmentDisposition::OutsideDeclaredAcceptance
        );
    }

    #[test]
    fn exposed_holdout_is_refused_before_metric_evaluation() {
        let mut source = campaign();
        source.cases[0].plan.blind_holdout.exposure.used_for_hyperparameter_selection = true;
        let errors = assess_blind_holdout_campaign(&source).unwrap_err();
        assert!(errors.iter().any(|error| {
            error.kind == ValidationAssessmentErrorKind::HoldoutAdmissionRefused
        }));
    }

    #[test]
    fn holdout_dataset_and_artifact_must_be_unique_across_campaign() {
        let mut source = campaign();
        source.cases[1].plan.blind_holdout.observation.dataset_id =
            source.cases[0].plan.blind_holdout.observation.dataset_id.clone();
        let errors = assess_blind_holdout_campaign(&source).unwrap_err();
        assert!(errors.iter().any(|error| {
            error.kind == ValidationAssessmentErrorKind::DuplicateHoldoutDataset
        }));
    }

    #[test]
    fn holdout_model_revision_must_match_profile_and_run() {
        let mut source = campaign();
        source.cases[0].model_run.model_revision = "stale-model-r16".to_string();
        assert!(assess_blind_holdout_campaign(&source).unwrap_err().iter().any(
            |error| error.kind == ValidationAssessmentErrorKind::HoldoutModelRevisionMismatch
        ));
    }

    #[test]
    fn model_run_output_digest_cannot_be_reused_for_another_holdout() {
        let mut source = campaign();
        source.cases[1].model_run.output_sha256 = source.cases[0].model_run.output_sha256.clone();
        assert!(assess_blind_holdout_campaign(&source).unwrap_err().iter().any(
            |error| error.kind == ValidationAssessmentErrorKind::ReusedModelOutputDigest
        ));
    }

    #[test]
    fn missing_prediction_is_refused_not_imputed() {
        let mut source = campaign();
        source.cases[0].predictions.clear();
        assert!(assess_blind_holdout_campaign(&source).unwrap_err().iter().any(
            |error| error.kind == ValidationAssessmentErrorKind::MissingPrediction
        ));
    }

    #[test]
    fn missing_measurement_is_refused_not_imputed() {
        let mut source = campaign();
        source.cases[0].measurements.clear();
        assert!(assess_blind_holdout_campaign(&source).unwrap_err().iter().any(
            |error| error.kind == ValidationAssessmentErrorKind::MissingMeasurement
        ));
    }

    #[test]
    fn observable_unit_boundary_and_averaging_window_must_match() {
        let mut source = campaign();
        source.cases[0].predictions[0].unit_ref = "MW".to_string();
        source.cases[1].measurements[0].service_boundary_ref = "wrong-boundary".to_string();
        source.cases[2].predictions[0].averaging_window_ref = "instantaneous".to_string();
        let errors = assess_blind_holdout_campaign(&source).unwrap_err();
        assert!(errors.iter().any(|error| error.kind == ValidationAssessmentErrorKind::PredictionUnitMismatch));
        assert!(errors.iter().any(|error| error.kind == ValidationAssessmentErrorKind::MeasurementBoundaryMismatch));
        assert!(errors.iter().any(|error| error.kind == ValidationAssessmentErrorKind::PredictionAveragingWindowMismatch));
    }

    #[test]
    fn malformed_prediction_and_measurement_intervals_are_refused() {
        let mut source = campaign();
        source.cases[0].predictions[0].prediction_interval_lower = 120.0;
        source.cases[1].measurements[0].measurement_interval_upper = 100.0;
        let errors = assess_blind_holdout_campaign(&source).unwrap_err();
        assert!(errors.iter().any(|error| error.kind == ValidationAssessmentErrorKind::InvertedPredictionInterval));
        assert!(errors.iter().any(|error| error.kind == ValidationAssessmentErrorKind::MeasurementOutsideItsInterval));
    }

    #[test]
    fn measurement_payload_digest_must_match_admitted_holdout() {
        let mut source = campaign();
        source.cases[0].measurements[0].source_artifact_sha256 = sha('z');
        assert!(assess_blind_holdout_campaign(&source).unwrap_err().iter().any(
            |error| error.kind == ValidationAssessmentErrorKind::MeasurementArtifactMismatch
        ));
    }

    #[test]
    fn too_few_distinct_facilities_is_insufficient() {
        let mut source = campaign();
        for case in &mut source.cases {
            case.plan.blind_holdout.observation.facility_id = "one-facility".to_string();
            if !case.plan.domain.allowed_facility_ids.contains(&"one-facility".to_string()) {
                case.plan.domain.allowed_facility_ids.push("one-facility".to_string());
            }
        }
        source.profile.minimum_distinct_facilities = 2;
        let report = assess_blind_holdout_campaign(&source).unwrap();
        assert_eq!(
            report.disposition,
            PredictiveAssessmentDisposition::InsufficientHoldoutEvidence
        );
    }

    #[test]
    fn point_coverage_and_measurement_interval_overlap_are_distinct() {
        let mut source = campaign();
        source.cases[2].measurements[0].measurement_interval_lower = 100.0;
        source.cases[2].measurements[0].measurement_interval_upper = 120.0;
        source.cases[2].measurements[0].measured_value = 110.0;
        let report = assess_blind_holdout_campaign(&source).unwrap();
        let metric = &report.per_observable_metrics[0];
        assert!(metric.measurement_interval_overlap_coverage > metric.empirical_point_interval_coverage);
    }

    #[test]
    fn reordering_cases_does_not_change_canonical_report() {
        let first = assess_blind_holdout_campaign(&campaign()).unwrap();
        let mut reordered = campaign();
        reordered.cases.reverse();
        let second = assess_blind_holdout_campaign(&reordered).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn final_unseal_authorization_is_required() {
        let mut source = campaign();
        source.cases[0].unseal_receipt_ref.clear();
        assert!(assess_blind_holdout_campaign(&source).unwrap_err().iter().any(
            |error| error.kind == ValidationAssessmentErrorKind::EmptyUnsealReceipt
        ));
    }

    #[test]
    fn no_holdout_values_are_accepted_when_measurement_source_is_unmeasured() {
        let mut source = campaign();
        source.cases[0].plan.blind_holdout.observation.evidence_class = EvidenceClass::ModelOutput;
        let errors = assess_blind_holdout_campaign(&source).unwrap_err();
        assert!(errors.iter().any(|error| {
            error.kind == ValidationAssessmentErrorKind::HoldoutAdmissionRefused
        }));
    }

    #[test]
    fn every_case_preserves_prediction_and_measurement_evidence_references() {
        let report = assess_blind_holdout_campaign(&campaign()).unwrap();
        assert_eq!(report.prediction_evidence_refs.len(), 3);
        assert_eq!(report.measurement_evidence_refs.len(), 3);
        assert_eq!(report.model_run_receipt_refs.len(), 3);
        assert_eq!(report.unseal_receipt_refs.len(), 3);
    }

    #[test]
    fn prediction_must_bind_exact_model_run_and_output_digest() {
        let mut source = campaign();
        source.cases[0].predictions[0].model_run_id = "stale-run".to_string();
        assert!(assess_blind_holdout_campaign(&source).unwrap_err().iter().any(
            |error| error.kind == ValidationAssessmentErrorKind::PredictionRunIdentityMismatch
        ));
        let mut source = campaign();
        source.cases[0].predictions[0].model_output_sha256 = sha('x');
        assert!(assess_blind_holdout_campaign(&source).unwrap_err().iter().any(
            |error| error.kind == ValidationAssessmentErrorKind::PredictionOutputDigestMismatch
        ));
    }

    #[test]
    fn prediction_interval_method_and_assumption_must_match_frozen_profile() {
        let mut source = campaign();
        source.cases[0].predictions[0].interval_method_ref = "post-hoc-tuned-band".to_string();
        source.cases[1].predictions[0].interval_assumption_ref.clear();
        let errors = assess_blind_holdout_campaign(&source).unwrap_err();
        assert!(errors.iter().any(|error| error.kind == ValidationAssessmentErrorKind::PredictionIntervalMethodMismatch));
        assert!(errors.iter().any(|error| error.kind == ValidationAssessmentErrorKind::PredictionIntervalAssumptionMismatch));
    }

    #[test]
    fn unseal_and_model_run_receipts_cannot_be_reused_between_holdouts() {
        let mut source = campaign();
        source.cases[1].unseal_receipt_ref = source.cases[0].unseal_receipt_ref.clone();
        source.cases[2].model_run.run_receipt_ref = source.cases[1].model_run.run_receipt_ref.clone();
        let errors = assess_blind_holdout_campaign(&source).unwrap_err();
        assert!(errors.iter().any(|error| error.kind == ValidationAssessmentErrorKind::ReusedUnsealReceipt));
        assert!(errors.iter().any(|error| error.kind == ValidationAssessmentErrorKind::ReusedModelRunReceipt));
    }

    #[test]
    fn holdout_count_does_not_hide_out_of_domain_observation() {
        let mut source = campaign();
        source.cases[0].plan.blind_holdout.observation.geometry_sha256 = sha('z');
        assert!(assess_blind_holdout_campaign(&source).unwrap_err().iter().any(
            |error| error.kind == ValidationAssessmentErrorKind::HoldoutAdmissionRefused
        ));
    }
}

