//! Paired measured test-stand and blind-holdout evaluation.
//!
//! Composes with validation_domain.rs for split integrity and declared domain
//! checks. This module computes descriptive residual metrics only; it does
//! not infer independence, statistical confidence or universal model validity.

use std::collections::BTreeSet;

use super::validation_domain::{ValidationDomainError, ValidationPlan, validate_validation_domain};

pub const HOLDOUT_METRICS_VERSION: &str = "fusion-vv-holdout-metrics/0.1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitmentEvidenceClass {
    SyntheticFixture,
    DeclaredOnly,
    IndependentlyTimestampedPreReveal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeasurementAttestationClass {
    SyntheticFixture,
    DeclaredOnly,
    ExternallyVerifiedMeasuredPayload,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasurementVerificationReceipt {
    pub observation_record_id: String,
    pub source_artifact_sha256: String,
    pub verifier_ref: String,
    pub attestation_ref: String,
    pub attestation_sha256: String,
    pub evidence_class: MeasurementAttestationClass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredictionCommitment {
    pub commitment_id: String,
    pub commitment_ref: String,
    pub commitment_sha256: String,
    pub model_revision: String,
    pub model_config_sha256: String,
    pub input_manifest_sha256: String,
    pub test_stand_output_sha256: String,
    pub holdout_output_sha256: String,
    pub observable_profile_sha256: String,
    pub test_stand_record_id: String,
    pub holdout_record_id: String,
    pub test_stand_artifact_sha256: String,
    pub holdout_artifact_sha256: String,
    pub expected_test_stand_sample_count: usize,
    pub expected_holdout_sample_count: usize,
    pub committed_before_test_stand_reveal: bool,
    pub committed_before_holdout_reveal: bool,
    pub commitment_evidence_class: CommitmentEvidenceClass,
    pub timestamp_authority_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HoldoutMetricProfile {
    pub profile_id: String,
    pub profile_sha256: String,
    pub observable_id: String,
    pub unit_ref: String,
    pub coordinate_frame_ref: String,
    pub sampling_window_ref: String,
    pub normalized_error_scale: f64,
    pub absolute_error_tolerance: f64,
    pub maximum_mean_absolute_error: f64,
    pub maximum_root_mean_square_error: f64,
    pub maximum_absolute_error: f64,
    pub maximum_failed_predictions_per_set: usize,
    pub minimum_holdout_group_count: usize,
    pub minimum_empirical_interval_coverage: Option<f64>,
    pub maximum_mean_interval_width: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictionInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairedValidationSample {
    pub sample_id: String,
    pub group_id: String,
    pub observation_record_id: String,
    pub observable_id: String,
    pub unit_ref: String,
    pub coordinate_frame_ref: String,
    pub sampling_window_ref: String,
    pub measured_value: f64,
    pub measurement_standard_uncertainty: Option<f64>,
    pub predicted_value: Option<f64>,
    pub prediction_interval: Option<PredictionInterval>,
    pub measurement_artifact_sha256: String,
    pub prediction_output_sha256: String,
    pub measurement_evidence_ref: String,
    /// A failed prediction still needs a run/failure evidence reference.
    pub prediction_evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HoldoutEvaluationCampaign {
    pub validation_plan: ValidationPlan,
    pub metric_profile: HoldoutMetricProfile,
    pub prediction_commitment: PredictionCommitment,
    pub test_stand_measurement_verification: MeasurementVerificationReceipt,
    pub holdout_measurement_verification: MeasurementVerificationReceipt,
    pub test_stand_samples: Vec<PairedValidationSample>,
    pub holdout_samples: Vec<PairedValidationSample>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HoldoutEvaluationDisposition {
    NotEvaluable,
    InsufficientMeasuredEvidence,
    OutOfDomain,
    HoldoutFailed,
    HoldoutPassedWithinDeclaredDomain,
    DescriptiveOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HoldoutEvaluationDiagnostic {
    InvalidMetricProfile,
    InvalidPredictionCommitment,
    PredictionCommitmentModelMismatch,
    PredictionCommitmentProfileMismatch,
    PredictionCommitmentRecordMismatch,
    PredictionCommitmentArtifactMismatch,
    PredictionCommitmentOutputMismatch,
    PredictionCommitmentNotPreReveal,
    MissingTimestampAuthority,
    MeasurementAttestationMismatch,
    MeasurementPayloadNotVerified,
    MeasurementAttestationDigestInvalid,
    MissingTestStandSamples,
    MissingHoldoutSamples,
    SampleCountMismatch,
    DuplicateSampleIdentity,
    EmptySampleIdentity,
    EmptyGroupIdentity,
    ObservableMismatch,
    UnitMismatch,
    CoordinateFrameMismatch,
    SamplingWindowMismatch,
    MeasurementArtifactMismatch,
    PredictionOutputMismatch,
    MissingMeasurementUncertainty,
    InvalidMeasurementUncertainty,
    NonFiniteMeasurement,
    NonFinitePrediction,
    MissingMeasurementEvidence,
    MissingPredictionEvidence,
    InvalidPredictionInterval,
    PartialPredictionIntervalCoverage,
    MissingRequiredPredictionIntervals,
    MissingPredictionsExceedPolicy,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValidationSetMetrics {
    pub sample_count: usize,
    pub prediction_count: usize,
    pub failed_prediction_count: usize,
    pub unique_group_count: usize,
    pub mean_signed_error: Option<f64>,
    pub mean_absolute_error: Option<f64>,
    pub root_mean_square_error: Option<f64>,
    pub maximum_absolute_error: Option<f64>,
    pub normalized_root_mean_square_error: Option<f64>,
    /// Descriptive |error| / stated measurement standard uncertainty.
    /// This is not assumed to follow any probability distribution.
    pub mean_absolute_error_over_measurement_uncertainty: Option<f64>,
    pub within_tolerance_count: usize,
    pub outside_tolerance_count: usize,
    pub empirical_prediction_interval_coverage: Option<f64>,
    pub mean_prediction_interval_width: Option<f64>,
    pub interval_sample_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HoldoutEvaluationReport {
    pub contract_version: &'static str,
    pub campaign_id: String,
    pub model_revision: String,
    pub disposition: HoldoutEvaluationDisposition,
    pub diagnostics: Vec<HoldoutEvaluationDiagnostic>,
    pub test_stand_metrics: Option<ValidationSetMetrics>,
    pub holdout_metrics: Option<ValidationSetMetrics>,
    pub calibration_record_count: usize,
    pub test_stand_record_id: String,
    pub holdout_record_id: String,
    pub commitment_ref: String,
    pub measurement_verification_refs: Vec<String>,
    pub claim_ceiling: &'static str,
}

/// Evaluate predictions only after the existing split/domain audit succeeds.
///
/// The returned interval coverage is empirical descriptive coverage. It is
/// not a confidence statement and does not assume independent samples.
pub fn evaluate_measured_holdout(campaign: &HoldoutEvaluationCampaign) -> HoldoutEvaluationReport {
    let plan = &campaign.validation_plan;
    let commitment = &campaign.prediction_commitment;
    let profile = &campaign.metric_profile;
    let mut diagnostics = validate_metric_profile(profile);
    let domain_audit = match validate_validation_domain(plan) {
        Ok(audit) => Some(audit),
        Err(errors) => {
            return report_for_domain_errors(campaign, errors);
        }
    };

    if !commitment_is_well_formed(commitment) {
        diagnostics.push(HoldoutEvaluationDiagnostic::InvalidPredictionCommitment);
    }
    if commitment.model_revision != plan.model_revision {
        diagnostics.push(HoldoutEvaluationDiagnostic::PredictionCommitmentModelMismatch);
    }
    if commitment.observable_profile_sha256 != profile.profile_sha256 {
        diagnostics.push(HoldoutEvaluationDiagnostic::PredictionCommitmentProfileMismatch);
    }
    if commitment.test_stand_record_id != plan.test_stand.record_id
        || commitment.holdout_record_id != plan.blind_holdout.observation.record_id
    {
        diagnostics.push(HoldoutEvaluationDiagnostic::PredictionCommitmentRecordMismatch);
    }
    if commitment.test_stand_artifact_sha256 != plan.test_stand.source_artifact_sha256
        || commitment.holdout_artifact_sha256
            != plan.blind_holdout.observation.source_artifact_sha256
    {
        diagnostics.push(HoldoutEvaluationDiagnostic::PredictionCommitmentArtifactMismatch);
    }
    if commitment.timestamp_authority_ref.trim().is_empty() {
        diagnostics.push(HoldoutEvaluationDiagnostic::MissingTimestampAuthority);
    }

    validate_measurement_attestation(
        &campaign.test_stand_measurement_verification,
        &plan.test_stand.record_id,
        &plan.test_stand.source_artifact_sha256,
        &mut diagnostics,
    );
    validate_measurement_attestation(
        &campaign.holdout_measurement_verification,
        &plan.blind_holdout.observation.record_id,
        &plan.blind_holdout.observation.source_artifact_sha256,
        &mut diagnostics,
    );

    if campaign.test_stand_samples.is_empty() {
        diagnostics.push(HoldoutEvaluationDiagnostic::MissingTestStandSamples);
    }
    if campaign.holdout_samples.is_empty() {
        diagnostics.push(HoldoutEvaluationDiagnostic::MissingHoldoutSamples);
    }
    if campaign.test_stand_samples.len() != commitment.expected_test_stand_sample_count
        || campaign.holdout_samples.len() != commitment.expected_holdout_sample_count
    {
        diagnostics.push(HoldoutEvaluationDiagnostic::SampleCountMismatch);
    }

    validate_samples(
        &campaign.test_stand_samples,
        &plan.test_stand.record_id,
        &plan.test_stand.source_artifact_sha256,
        &commitment.test_stand_output_sha256,
        profile,
        &mut diagnostics,
    );
    validate_samples(
        &campaign.holdout_samples,
        &plan.blind_holdout.observation.record_id,
        &plan.blind_holdout.observation.source_artifact_sha256,
        &commitment.holdout_output_sha256,
        profile,
        &mut diagnostics,
    );
    let test_ids: BTreeSet<&str> = campaign
        .test_stand_samples
        .iter()
        .map(|sample| sample.sample_id.as_str())
        .collect();
    if campaign
        .holdout_samples
        .iter()
        .any(|sample| test_ids.contains(sample.sample_id.as_str()))
    {
        diagnostics.push(HoldoutEvaluationDiagnostic::DuplicateSampleIdentity);
    }

    if !diagnostics.is_empty() {
        return HoldoutEvaluationReport {
            contract_version: HOLDOUT_METRICS_VERSION,
            campaign_id: plan.domain.domain_id.clone(),
            model_revision: plan.model_revision.clone(),
            disposition: HoldoutEvaluationDisposition::NotEvaluable,
            diagnostics,
            test_stand_metrics: None,
            holdout_metrics: None,
            calibration_record_count: plan.calibration.len(),
            test_stand_record_id: plan.test_stand.record_id.clone(),
            holdout_record_id: plan.blind_holdout.observation.record_id.clone(),
            commitment_ref: commitment.commitment_ref.clone(),
            measurement_verification_refs: vec![
                campaign
                    .test_stand_measurement_verification
                    .attestation_ref
                    .clone(),
                campaign
                    .holdout_measurement_verification
                    .attestation_ref
                    .clone(),
            ],
            claim_ceiling: "descriptive paired metrics on declared validation inputs only",
        };
    }

    let test_metrics = summarize_samples(
        &campaign.test_stand_samples,
        profile.absolute_error_tolerance,
        profile.normalized_error_scale,
    );
    let holdout_metrics = summarize_samples(
        &campaign.holdout_samples,
        profile.absolute_error_tolerance,
        profile.normalized_error_scale,
    );

    let test_passes = meets_thresholds(&test_metrics, profile, false);
    let holdout_passes = meets_thresholds(&holdout_metrics, profile, true);
    let missing_predictions_exceed = test_metrics.failed_prediction_count
        > profile.maximum_failed_predictions_per_set
        || holdout_metrics.failed_prediction_count > profile.maximum_failed_predictions_per_set;
    if missing_predictions_exceed {
        diagnostics.push(HoldoutEvaluationDiagnostic::MissingPredictionsExceedPolicy);
    }

    let commitment_pre_reveal = commitment.committed_before_test_stand_reveal
        && commitment.committed_before_holdout_reveal
        && commitment.commitment_evidence_class
            == CommitmentEvidenceClass::IndependentlyTimestampedPreReveal;
    let measured_payloads_verified = campaign.test_stand_measurement_verification.evidence_class
        == MeasurementAttestationClass::ExternallyVerifiedMeasuredPayload
        && campaign.holdout_measurement_verification.evidence_class
            == MeasurementAttestationClass::ExternallyVerifiedMeasuredPayload;

    let disposition = if !commitment_pre_reveal {
        diagnostics.push(HoldoutEvaluationDiagnostic::PredictionCommitmentNotPreReveal);
        HoldoutEvaluationDisposition::DescriptiveOnly
    } else if !measured_payloads_verified {
        diagnostics.push(HoldoutEvaluationDiagnostic::MeasurementPayloadNotVerified);
        HoldoutEvaluationDisposition::InsufficientMeasuredEvidence
    } else if missing_predictions_exceed || !test_passes || !holdout_passes {
        HoldoutEvaluationDisposition::HoldoutFailed
    } else {
        HoldoutEvaluationDisposition::HoldoutPassedWithinDeclaredDomain
    };

    HoldoutEvaluationReport {
        contract_version: HOLDOUT_METRICS_VERSION,
        campaign_id: plan.domain.domain_id.clone(),
        model_revision: plan.model_revision.clone(),
        disposition,
        diagnostics,
        test_stand_metrics: Some(test_metrics),
        holdout_metrics: Some(holdout_metrics),
        calibration_record_count: plan.calibration.len(),
        test_stand_record_id: plan.test_stand.record_id.clone(),
        holdout_record_id: plan.blind_holdout.observation.record_id.clone(),
        commitment_ref: commitment.commitment_ref.clone(),
        measurement_verification_refs: vec![
            campaign
                .test_stand_measurement_verification
                .attestation_ref
                .clone(),
            campaign
                .holdout_measurement_verification
                .attestation_ref
                .clone(),
        ],
        claim_ceiling: "paired measured metrics on the declared held-out domain only",
    }
}

fn report_for_domain_errors(
    campaign: &HoldoutEvaluationCampaign,
    errors: Vec<ValidationDomainError>,
) -> HoldoutEvaluationReport {
    let out_of_domain = errors
        .iter()
        .any(|error| matches!(error, ValidationDomainError::OutOfDomain { .. }));
    let insufficient_measured = errors
        .iter()
        .any(|error| matches!(error, ValidationDomainError::NonMeasuredObservation { .. }));
    let diagnostics = if out_of_domain {
        vec![HoldoutEvaluationDiagnostic::ObservableMismatch]
    } else if insufficient_measured {
        vec![HoldoutEvaluationDiagnostic::MeasurementPayloadNotVerified]
    } else {
        vec![HoldoutEvaluationDiagnostic::InvalidPredictionCommitment]
    };
    HoldoutEvaluationReport {
        contract_version: HOLDOUT_METRICS_VERSION,
        campaign_id: campaign.validation_plan.domain.domain_id.clone(),
        model_revision: campaign.validation_plan.model_revision.clone(),
        disposition: if out_of_domain {
            HoldoutEvaluationDisposition::OutOfDomain
        } else if insufficient_measured {
            HoldoutEvaluationDisposition::InsufficientMeasuredEvidence
        } else {
            HoldoutEvaluationDisposition::NotEvaluable
        },
        diagnostics,
        test_stand_metrics: None,
        holdout_metrics: None,
        calibration_record_count: campaign.validation_plan.calibration.len(),
        test_stand_record_id: campaign.validation_plan.test_stand.record_id.clone(),
        holdout_record_id: campaign
            .validation_plan
            .blind_holdout
            .observation
            .record_id
            .clone(),
        commitment_ref: campaign.prediction_commitment.commitment_ref.clone(),
        measurement_verification_refs: vec![
            campaign
                .test_stand_measurement_verification
                .attestation_ref
                .clone(),
            campaign
                .holdout_measurement_verification
                .attestation_ref
                .clone(),
        ],
        claim_ceiling: "split and domain audit failed before metric evaluation",
    }
}

fn validate_metric_profile(profile: &HoldoutMetricProfile) -> Vec<HoldoutEvaluationDiagnostic> {
    let mut diagnostics = Vec::new();
    if profile.profile_id.trim().is_empty()
        || !is_sha256(&profile.profile_sha256)
        || profile.observable_id.trim().is_empty()
        || profile.unit_ref.trim().is_empty()
        || profile.coordinate_frame_ref.trim().is_empty()
        || profile.sampling_window_ref.trim().is_empty()
        || !profile.normalized_error_scale.is_finite()
        || profile.normalized_error_scale <= 0.0
        || !nonnegative_finite(profile.absolute_error_tolerance)
        || !nonnegative_finite(profile.maximum_mean_absolute_error)
        || !nonnegative_finite(profile.maximum_root_mean_square_error)
        || !nonnegative_finite(profile.maximum_absolute_error)
        || profile.minimum_holdout_group_count == 0
        || profile
            .minimum_empirical_interval_coverage
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        || profile
            .maximum_mean_interval_width
            .is_some_and(|value| !nonnegative_finite(value))
    {
        diagnostics.push(HoldoutEvaluationDiagnostic::InvalidMetricProfile);
    }
    diagnostics
}

fn commitment_is_well_formed(commitment: &PredictionCommitment) -> bool {
    !commitment.commitment_id.trim().is_empty()
        && !commitment.commitment_ref.trim().is_empty()
        && is_sha256(&commitment.commitment_sha256)
        && !commitment.model_revision.trim().is_empty()
        && is_sha256(&commitment.model_config_sha256)
        && is_sha256(&commitment.input_manifest_sha256)
        && is_sha256(&commitment.test_stand_output_sha256)
        && is_sha256(&commitment.holdout_output_sha256)
        && is_sha256(&commitment.observable_profile_sha256)
        && !commitment.test_stand_record_id.trim().is_empty()
        && !commitment.holdout_record_id.trim().is_empty()
        && is_sha256(&commitment.test_stand_artifact_sha256)
        && is_sha256(&commitment.holdout_artifact_sha256)
        && commitment.expected_test_stand_sample_count > 0
        && commitment.expected_holdout_sample_count > 0
}

fn validate_measurement_attestation(
    attestation: &MeasurementVerificationReceipt,
    expected_record_id: &str,
    expected_artifact_sha256: &str,
    diagnostics: &mut Vec<HoldoutEvaluationDiagnostic>,
) {
    if attestation.observation_record_id != expected_record_id
        || attestation.source_artifact_sha256 != expected_artifact_sha256
        || attestation.verifier_ref.trim().is_empty()
        || attestation.attestation_ref.trim().is_empty()
        || !is_sha256(&attestation.attestation_sha256)
    {
        diagnostics.push(HoldoutEvaluationDiagnostic::MeasurementAttestationMismatch);
    }
}

fn validate_samples(
    samples: &[PairedValidationSample],
    expected_record_id: &str,
    expected_artifact_sha256: &str,
    expected_output_sha256: &str,
    profile: &HoldoutMetricProfile,
    diagnostics: &mut Vec<HoldoutEvaluationDiagnostic>,
) {
    let mut sample_ids = BTreeSet::new();
    for sample in samples {
        if sample.sample_id.trim().is_empty() {
            diagnostics.push(HoldoutEvaluationDiagnostic::EmptySampleIdentity);
        }
        if !sample_ids.insert(sample.sample_id.as_str()) {
            diagnostics.push(HoldoutEvaluationDiagnostic::DuplicateSampleIdentity);
        }
        if sample.group_id.trim().is_empty() {
            diagnostics.push(HoldoutEvaluationDiagnostic::EmptyGroupIdentity);
        }
        if sample.observation_record_id != expected_record_id {
            diagnostics.push(HoldoutEvaluationDiagnostic::PredictionCommitmentRecordMismatch);
        }
        if sample.observable_id != profile.observable_id {
            diagnostics.push(HoldoutEvaluationDiagnostic::ObservableMismatch);
        }
        if sample.unit_ref != profile.unit_ref {
            diagnostics.push(HoldoutEvaluationDiagnostic::UnitMismatch);
        }
        if sample.coordinate_frame_ref != profile.coordinate_frame_ref {
            diagnostics.push(HoldoutEvaluationDiagnostic::CoordinateFrameMismatch);
        }
        if sample.sampling_window_ref != profile.sampling_window_ref {
            diagnostics.push(HoldoutEvaluationDiagnostic::SamplingWindowMismatch);
        }
        if sample.measurement_artifact_sha256 != expected_artifact_sha256
            || !is_sha256(&sample.measurement_artifact_sha256)
        {
            diagnostics.push(HoldoutEvaluationDiagnostic::MeasurementArtifactMismatch);
        }
        if sample.prediction_output_sha256 != expected_output_sha256
            || !is_sha256(&sample.prediction_output_sha256)
        {
            diagnostics.push(HoldoutEvaluationDiagnostic::PredictionOutputMismatch);
        }
        if !sample.measured_value.is_finite() {
            diagnostics.push(HoldoutEvaluationDiagnostic::NonFiniteMeasurement);
        }
        match sample.measurement_standard_uncertainty {
            None => diagnostics.push(HoldoutEvaluationDiagnostic::MissingMeasurementUncertainty),
            Some(value) if !value.is_finite() || value <= 0.0 => {
                diagnostics.push(HoldoutEvaluationDiagnostic::InvalidMeasurementUncertainty);
            }
            Some(_) => {}
        }
        if sample.measurement_evidence_ref.trim().is_empty() {
            diagnostics.push(HoldoutEvaluationDiagnostic::MissingMeasurementEvidence);
        }
        if sample.prediction_evidence_ref.trim().is_empty() {
            diagnostics.push(HoldoutEvaluationDiagnostic::MissingPredictionEvidence);
        }
        if sample
            .predicted_value
            .is_some_and(|value| !value.is_finite())
        {
            diagnostics.push(HoldoutEvaluationDiagnostic::NonFinitePrediction);
        }
        if let Some(interval) = sample.prediction_interval
            && (!interval.lower.is_finite()
                || !interval.upper.is_finite()
                || interval.lower > interval.upper
                || sample.predicted_value.is_none())
        {
            diagnostics.push(HoldoutEvaluationDiagnostic::InvalidPredictionInterval);
        }
    }
    let prediction_count = samples
        .iter()
        .filter(|sample| sample.predicted_value.is_some())
        .count();
    let interval_count = samples
        .iter()
        .filter(|sample| sample.prediction_interval.is_some())
        .count();
    if interval_count > 0 && interval_count < prediction_count {
        diagnostics.push(HoldoutEvaluationDiagnostic::PartialPredictionIntervalCoverage);
    }
}

fn summarize_samples(
    samples: &[PairedValidationSample],
    absolute_error_tolerance: f64,
    normalized_error_scale: f64,
) -> ValidationSetMetrics {
    let mut ordered_samples: Vec<&PairedValidationSample> = samples.iter().collect();
    ordered_samples.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    let sample_count = ordered_samples.len();
    let unique_group_count = ordered_samples
        .iter()
        .map(|sample| sample.group_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let predictions: Vec<(&PairedValidationSample, f64)> = ordered_samples
        .iter()
        .filter_map(|sample| {
            sample
                .predicted_value
                .map(|predicted| (*sample, predicted - sample.measured_value))
        })
        .collect();
    let prediction_count = predictions.len();
    let failed_prediction_count = sample_count - prediction_count;
    let mean_signed_error = average(predictions.iter().map(|(_, error)| *error));
    let mean_absolute_error = average(predictions.iter().map(|(_, error)| error.abs()));
    let mean_square_error = average(predictions.iter().map(|(_, error)| error * error));
    let root_mean_square_error = mean_square_error.map(f64::sqrt);
    let maximum_absolute_error = predictions
        .iter()
        .map(|(_, error)| error.abs())
        .reduce(f64::max);
    let normalized_root_mean_square_error =
        root_mean_square_error.map(|value| value / normalized_error_scale);
    let mean_absolute_error_over_measurement_uncertainty =
        average(predictions.iter().filter_map(|(sample, error)| {
            sample
                .measurement_standard_uncertainty
                .map(|uncertainty| error.abs() / uncertainty)
        }));
    let within_tolerance_count = predictions
        .iter()
        .filter(|(_, error)| error.abs() <= absolute_error_tolerance)
        .count();
    let outside_tolerance_count = prediction_count - within_tolerance_count;

    let intervals: Vec<(bool, f64)> = predictions
        .iter()
        .filter_map(|(sample, _)| {
            sample.prediction_interval.map(|interval| {
                (
                    sample.measured_value >= interval.lower
                        && sample.measured_value <= interval.upper,
                    interval.upper - interval.lower,
                )
            })
        })
        .collect();
    let interval_sample_count = intervals.len();
    let empirical_prediction_interval_coverage =
        if interval_sample_count == prediction_count && prediction_count > 0 {
            Some(
                intervals.iter().filter(|(covered, _)| *covered).count() as f64
                    / prediction_count as f64,
            )
        } else {
            None
        };
    let mean_prediction_interval_width =
        if interval_sample_count == prediction_count && prediction_count > 0 {
            average(intervals.iter().map(|(_, width)| *width))
        } else {
            None
        };

    ValidationSetMetrics {
        sample_count,
        prediction_count,
        failed_prediction_count,
        unique_group_count,
        mean_signed_error,
        mean_absolute_error,
        root_mean_square_error,
        maximum_absolute_error,
        normalized_root_mean_square_error,
        mean_absolute_error_over_measurement_uncertainty,
        within_tolerance_count,
        outside_tolerance_count,
        empirical_prediction_interval_coverage,
        mean_prediction_interval_width,
        interval_sample_count,
    }
}

fn meets_thresholds(
    metrics: &ValidationSetMetrics,
    profile: &HoldoutMetricProfile,
    is_holdout: bool,
) -> bool {
    let enough_groups =
        !is_holdout || metrics.unique_group_count >= profile.minimum_holdout_group_count;
    let interval_coverage_ok = profile
        .minimum_empirical_interval_coverage
        .is_none_or(|minimum| {
            metrics
                .empirical_prediction_interval_coverage
                .is_some_and(|coverage| coverage >= minimum)
        });
    let interval_width_ok = profile.maximum_mean_interval_width.is_none_or(|maximum| {
        metrics
            .mean_prediction_interval_width
            .is_some_and(|width| width <= maximum)
    });

    enough_groups
        && metrics.failed_prediction_count <= profile.maximum_failed_predictions_per_set
        && metrics
            .mean_absolute_error
            .is_some_and(|value| value <= profile.maximum_mean_absolute_error)
        && metrics
            .root_mean_square_error
            .is_some_and(|value| value <= profile.maximum_root_mean_square_error)
        && metrics
            .maximum_absolute_error
            .is_some_and(|value| value <= profile.maximum_absolute_error)
        && interval_coverage_ok
        && interval_width_ok
}

fn average(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0_usize;
    for value in values {
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f64)
}

fn nonnegative_finite(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EvidenceClass;
    use crate::ModelUse;
    use crate::validation_domain::{
        BlindHoldoutMeasurement, CalibrationExposure, HoldoutExposure, MeasuredObservation,
        ValidationDomain,
    };

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn observation(
        record_id: &str,
        dataset_id: &str,
        artifact: char,
        facility: &str,
        case_id: &str,
        operating_point_id: &str,
        episode_id: &str,
    ) -> MeasuredObservation {
        MeasuredObservation {
            record_id: record_id.into(),
            evidence_class: EvidenceClass::MeasuredObservation,
            dataset_id: dataset_id.into(),
            source_artifact_sha256: digest(artifact),
            regime_id: "tokamak-regime".into(),
            geometry_sha256: digest('b'),
            facility_id: facility.into(),
            case_id: case_id.into(),
            operating_point_id: operating_point_id.into(),
            episode_ids: vec![episode_id.into()],
            diagnostic_revision: "diagnostic-r3".into(),
            measurement_procedure_ref: "procedure-v2".into(),
            diagnostic_calibration_ref: "calibration-cert-17".into(),
            measurement_uncertainty_ref: "measurement-uncertainty-u7".into(),
        }
    }

    fn validation_plan() -> ValidationPlan {
        ValidationPlan {
            model_revision: "model-r4".into(),
            requested_use: ModelUse::SingleStepPrediction,
            domain: ValidationDomain {
                domain_id: "tokamak-domain-v1".into(),
                model_revision: "model-r4".into(),
                regime_id: "tokamak-regime".into(),
                supported_uses: vec![ModelUse::SingleStepPrediction],
                allowed_geometry_sha256: vec![digest('b')],
                allowed_facility_ids: vec!["facility-a".into(), "facility-b".into()],
                allowed_case_ids: vec!["case-a".into(), "case-b".into()],
                allowed_operating_point_ids: vec!["op-a".into(), "op-b".into()],
            },
            calibration: vec![CalibrationExposure {
                record_id: "calibration-r1".into(),
                dataset_id: "calibration-dataset".into(),
                source_artifact_sha256: digest('c'),
                regime_id: "tokamak-regime".into(),
                geometry_sha256: digest('b'),
                episode_ids: vec!["calibration-shot-1".into()],
                preprocessing_fit_id: "fit-calibration-only".into(),
            }],
            test_stand: observation(
                "test-stand-1",
                "test-dataset",
                'd',
                "facility-a",
                "case-a",
                "op-a",
                "test-shot-1",
            ),
            blind_holdout: BlindHoldoutMeasurement {
                observation: observation(
                    "holdout-1",
                    "holdout-dataset",
                    'e',
                    "facility-b",
                    "case-b",
                    "op-b",
                    "holdout-shot-1",
                ),
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
        }
    }

    fn profile() -> HoldoutMetricProfile {
        HoldoutMetricProfile {
            profile_id: "plasma-current-measured-holdout-v1".into(),
            profile_sha256: digest('f'),
            observable_id: "plasma-current".into(),
            unit_ref: "MA".into(),
            coordinate_frame_ref: "magnetic-axis-frame-v1".into(),
            sampling_window_ref: "steady-window-1s-v1".into(),
            normalized_error_scale: 1.0,
            absolute_error_tolerance: 0.2,
            maximum_mean_absolute_error: 0.15,
            maximum_root_mean_square_error: 0.2,
            maximum_absolute_error: 0.3,
            maximum_failed_predictions_per_set: 0,
            minimum_holdout_group_count: 2,
            minimum_empirical_interval_coverage: Some(0.8),
            maximum_mean_interval_width: Some(0.5),
        }
    }

    fn commitment() -> PredictionCommitment {
        PredictionCommitment {
            commitment_id: "blind-prediction-commitment-v1".into(),
            commitment_ref: "independent-timestamped-commitment".into(),
            commitment_sha256: digest('1'),
            model_revision: "model-r4".into(),
            model_config_sha256: digest('2'),
            input_manifest_sha256: digest('3'),
            test_stand_output_sha256: digest('4'),
            holdout_output_sha256: digest('5'),
            observable_profile_sha256: profile().profile_sha256,
            test_stand_record_id: "test-stand-1".into(),
            holdout_record_id: "holdout-1".into(),
            test_stand_artifact_sha256: digest('d'),
            holdout_artifact_sha256: digest('e'),
            expected_test_stand_sample_count: 2,
            expected_holdout_sample_count: 2,
            committed_before_test_stand_reveal: true,
            committed_before_holdout_reveal: true,
            commitment_evidence_class: CommitmentEvidenceClass::IndependentlyTimestampedPreReveal,
            timestamp_authority_ref: "trusted-timestamp-authority-v1".into(),
        }
    }

    fn attestation(record_id: &str, artifact: char) -> MeasurementVerificationReceipt {
        MeasurementVerificationReceipt {
            observation_record_id: record_id.into(),
            source_artifact_sha256: digest(artifact),
            verifier_ref: "measurement-payload-verifier-v1".into(),
            attestation_ref: format!("verified-payload-{record_id}"),
            attestation_sha256: digest('6'),
            evidence_class: MeasurementAttestationClass::ExternallyVerifiedMeasuredPayload,
        }
    }

    fn sample(
        id: &str,
        group: &str,
        record_id: &str,
        artifact: char,
        output: char,
        measured: f64,
        predicted: f64,
    ) -> PairedValidationSample {
        PairedValidationSample {
            sample_id: id.into(),
            group_id: group.into(),
            observation_record_id: record_id.into(),
            observable_id: "plasma-current".into(),
            unit_ref: "MA".into(),
            coordinate_frame_ref: "magnetic-axis-frame-v1".into(),
            sampling_window_ref: "steady-window-1s-v1".into(),
            measured_value: measured,
            measurement_standard_uncertainty: Some(0.05),
            predicted_value: Some(predicted),
            prediction_interval: Some(PredictionInterval {
                lower: predicted - 0.15,
                upper: predicted + 0.15,
            }),
            measurement_artifact_sha256: digest(artifact),
            prediction_output_sha256: digest(output),
            measurement_evidence_ref: format!("measurement-{id}"),
            prediction_evidence_ref: format!("prediction-{id}"),
        }
    }

    fn campaign() -> HoldoutEvaluationCampaign {
        HoldoutEvaluationCampaign {
            validation_plan: validation_plan(),
            metric_profile: profile(),
            prediction_commitment: commitment(),
            test_stand_measurement_verification: attestation("test-stand-1", 'd'),
            holdout_measurement_verification: attestation("holdout-1", 'e'),
            test_stand_samples: vec![
                sample(
                    "test-1",
                    "test-group-a",
                    "test-stand-1",
                    'd',
                    '4',
                    10.0,
                    10.05,
                ),
                sample(
                    "test-2",
                    "test-group-b",
                    "test-stand-1",
                    'd',
                    '4',
                    11.0,
                    11.08,
                ),
            ],
            holdout_samples: vec![
                sample(
                    "holdout-a",
                    "holdout-group-a",
                    "holdout-1",
                    'e',
                    '5',
                    12.0,
                    12.04,
                ),
                sample(
                    "holdout-b",
                    "holdout-group-b",
                    "holdout-1",
                    'e',
                    '5',
                    13.0,
                    13.07,
                ),
            ],
        }
    }

    #[test]
    fn valid_pre_registered_measured_holdout_passes_only_with_attested_inputs() {
        let report = evaluate_measured_holdout(&campaign());
        assert_eq!(
            report.disposition,
            HoldoutEvaluationDisposition::HoldoutPassedWithinDeclaredDomain
        );
        assert_eq!(report.test_stand_metrics.as_ref().unwrap().sample_count, 2);
        assert_eq!(
            report.holdout_metrics.as_ref().unwrap().unique_group_count,
            2
        );
        assert_eq!(
            report
                .holdout_metrics
                .as_ref()
                .unwrap()
                .empirical_prediction_interval_coverage,
            Some(1.0)
        );
    }

    #[test]
    fn synthetic_measurement_attestations_cannot_pass() {
        let mut source = campaign();
        source.test_stand_measurement_verification.evidence_class =
            MeasurementAttestationClass::SyntheticFixture;
        let report = evaluate_measured_holdout(&source);
        assert_eq!(
            report.disposition,
            HoldoutEvaluationDisposition::InsufficientMeasuredEvidence
        );
        assert!(report.test_stand_metrics.is_some());
    }

    #[test]
    fn declared_only_or_post_reveal_commitment_is_descriptive_only() {
        let mut source = campaign();
        source.prediction_commitment.commitment_evidence_class =
            CommitmentEvidenceClass::DeclaredOnly;
        let report = evaluate_measured_holdout(&source);
        assert_eq!(
            report.disposition,
            HoldoutEvaluationDisposition::DescriptiveOnly
        );
    }

    #[test]
    fn model_revision_or_profile_digest_mismatch_is_not_evaluable() {
        let mut source = campaign();
        source.prediction_commitment.model_revision = "model-r5".into();
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
        let mut source = campaign();
        source.prediction_commitment.observable_profile_sha256 = digest('9');
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
    }

    #[test]
    fn record_and_artifact_identity_mismatch_is_not_evaluable() {
        let mut source = campaign();
        source.prediction_commitment.holdout_artifact_sha256 = digest('8');
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
        let mut source = campaign();
        source.holdout_samples[0].observation_record_id = "other-record".into();
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
    }

    #[test]
    fn duplicate_sample_ids_are_rejected() {
        let mut source = campaign();
        source.holdout_samples[1].sample_id = source.holdout_samples[0].sample_id.clone();
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
    }

    #[test]
    fn unit_frame_and_sampling_window_mismatch_are_rejected() {
        let mut source = campaign();
        source.holdout_samples[0].unit_ref = "kA".into();
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
        let mut source = campaign();
        source.holdout_samples[0].coordinate_frame_ref = "wrong-frame".into();
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
        let mut source = campaign();
        source.holdout_samples[0].sampling_window_ref = "instantaneous".into();
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
    }

    #[test]
    fn missing_measurement_uncertainty_is_not_defaulted_to_zero() {
        let mut source = campaign();
        source.holdout_samples[0].measurement_standard_uncertainty = None;
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
    }

    #[test]
    fn failed_prediction_remains_in_sample_count_and_fails_policy() {
        let mut source = campaign();
        source.holdout_samples[0].predicted_value = None;
        source.holdout_samples[0].prediction_interval = None;
        source.holdout_samples[0].prediction_evidence_ref = "solver-run-failed-receipt".into();
        let report = evaluate_measured_holdout(&source);
        assert_eq!(
            report.disposition,
            HoldoutEvaluationDisposition::HoldoutFailed
        );
        let metrics = report.holdout_metrics.unwrap();
        assert_eq!(metrics.sample_count, 2);
        assert_eq!(metrics.prediction_count, 1);
        assert_eq!(metrics.failed_prediction_count, 1);
    }

    #[test]
    fn non_finite_values_and_invalid_intervals_are_rejected() {
        let mut source = campaign();
        source.holdout_samples[0].measured_value = f64::NAN;
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
        let mut source = campaign();
        source.holdout_samples[0].prediction_interval = Some(PredictionInterval {
            lower: 14.0,
            upper: 13.0,
        });
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
    }

    #[test]
    fn partial_interval_coverage_is_not_reported_as_full_coverage() {
        let mut source = campaign();
        source.holdout_samples[0].prediction_interval = None;
        let report = evaluate_measured_holdout(&source);
        assert_eq!(
            report.disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
        assert!(
            report
                .diagnostics
                .contains(&HoldoutEvaluationDiagnostic::PartialPredictionIntervalCoverage)
        );
    }

    #[test]
    fn missing_required_intervals_fail_the_predeclared_metric() {
        let mut source = campaign();
        source.holdout_samples.iter_mut().for_each(|sample| {
            sample.prediction_interval = None;
        });
        source.test_stand_samples.iter_mut().for_each(|sample| {
            sample.prediction_interval = None;
        });
        source.metric_profile.minimum_empirical_interval_coverage = Some(0.8);
        let report = evaluate_measured_holdout(&source);
        assert_eq!(
            report.disposition,
            HoldoutEvaluationDisposition::HoldoutFailed
        );
        assert!(
            report
                .holdout_metrics
                .unwrap()
                .empirical_prediction_interval_coverage
                .is_none()
        );
    }

    #[test]
    fn missed_error_threshold_fails_holdout_without_dropping_case() {
        let mut source = campaign();
        source.holdout_samples[0].predicted_value = Some(15.0);
        let report = evaluate_measured_holdout(&source);
        assert_eq!(
            report.disposition,
            HoldoutEvaluationDisposition::HoldoutFailed
        );
        assert_eq!(report.holdout_metrics.as_ref().unwrap().sample_count, 2);
        assert!(
            report
                .holdout_metrics
                .unwrap()
                .maximum_absolute_error
                .unwrap()
                > 1.0
        );
    }

    #[test]
    fn insufficient_holdout_groups_fails_even_when_row_count_is_large() {
        let mut source = campaign();
        source.holdout_samples[1].group_id = source.holdout_samples[0].group_id.clone();
        let report = evaluate_measured_holdout(&source);
        assert_eq!(
            report.disposition,
            HoldoutEvaluationDisposition::HoldoutFailed
        );
        assert_eq!(report.holdout_metrics.unwrap().unique_group_count, 1);
    }

    #[test]
    fn row_reordering_does_not_change_descriptive_metrics() {
        let first = campaign();
        let mut second = campaign();
        second.holdout_samples.reverse();
        second.test_stand_samples.reverse();
        assert_eq!(
            evaluate_measured_holdout(&first),
            evaluate_measured_holdout(&second)
        );
    }

    #[test]
    fn interval_coverage_is_empirical_not_a_confidence_claim() {
        let report = evaluate_measured_holdout(&campaign());
        assert_eq!(
            report
                .holdout_metrics
                .unwrap()
                .empirical_prediction_interval_coverage,
            Some(1.0)
        );
        assert!(report.claim_ceiling.contains("declared held-out domain"));
    }

    #[test]
    fn out_of_domain_holdout_cannot_be_rescued_by_low_residuals() {
        let mut source = campaign();
        source
            .validation_plan
            .blind_holdout
            .observation
            .geometry_sha256 = digest('9');
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::OutOfDomain
        );
    }

    #[test]
    fn failed_prediction_output_digest_mismatch_is_not_evaluable() {
        let mut source = campaign();
        source.holdout_samples[0].prediction_output_sha256 = digest('8');
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
    }

    #[test]
    fn zero_or_negative_measurement_uncertainty_is_rejected() {
        let mut source = campaign();
        source.holdout_samples[0].measurement_standard_uncertainty = Some(0.0);
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
    }

    #[test]
    fn pre_reveal_flags_and_measurement_attestations_are_both_required() {
        let mut source = campaign();
        source.prediction_commitment.committed_before_holdout_reveal = false;
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::DescriptiveOnly
        );
        let mut source = campaign();
        source.holdout_measurement_verification.attestation_sha256 = "bad".into();
        assert_eq!(
            evaluate_measured_holdout(&source).disposition,
            HoldoutEvaluationDisposition::NotEvaluable
        );
    }
}
