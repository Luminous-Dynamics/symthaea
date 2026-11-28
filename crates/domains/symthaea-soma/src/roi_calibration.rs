// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Reproducible calibration-report verification for value-of-information ROI.
//!
//! The verifier recomputes ECE and Brier score from held-out reliability-bin
//! aggregates and binds ROI estimates to a report, model, and task family.
//! Digests establish content identity, not dataset authenticity or provenance.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_foveation::{RoiValueEstimate};

pub const ROI_CALIBRATION_REPORT_SCHEMA_V1: &str = "symthaea.roi-calibration-report.v1";
const ROI_CALIBRATION_DIGEST_CONTEXT: &str = "symthaea.roi-calibration-report.v1";
const MIN_CALIBRATION_SAMPLES: u64 = 100;
const MIN_EVALUATION_SAMPLES: u64 = 100;
const METRIC_TOLERANCE: f32 = 0.000_01;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoiCalibrationBinV1 {
    pub sample_count: u64,
    pub mean_confidence: f32,
    pub observed_accuracy: f32,
    /// Mean squared probability error for samples in this bin.
    pub mean_brier_score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoiCalibrationReportV1 {
    pub schema: String,
    pub report_id: String,
    pub model_digest: String,
    pub task_family: String,
    pub calibration_dataset_digest: String,
    pub evaluation_dataset_digest: String,
    pub calibration_split_digest: String,
    pub evaluation_split_digest: String,
    pub calibration_sample_count: u64,
    pub evaluation_sample_count: u64,
    pub bins: Vec<RoiCalibrationBinV1>,
    pub expected_calibration_error: f32,
    pub brier_score: f32,
    pub report_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoiCalibrationError {
    WrongSchema,
    InvalidIdentifier,
    InvalidDigest,
    DatasetOrSplitLeakage,
    InsufficientSamples,
    InvalidBinCount,
    InvalidBin,
    BinSampleCountMismatch,
    InvalidMetric,
    MetricMismatch,
    ReportDigestMismatch,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoiCalibrationVerification {
    pub report_id: String,
    pub report_digest: String,
    pub model_digest: String,
    pub task_family: String,
    pub calibration_sample_count: u64,
    pub evaluation_sample_count: u64,
    pub recomputed_expected_calibration_error: f32,
    pub recomputed_brier_score: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoiCalibrationBindingError {
    MissingReport,
    AmbiguousReport,
    InvalidReport(RoiCalibrationError),
    ReportIdentityMismatch,
    ModelMismatch,
    TaskFamilyMismatch,
    SampleCountMismatch,
    EceMismatch,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoiCalibrationBindingRejection {
    pub candidate_id: String,
    pub reason: RoiCalibrationBindingError,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoiCalibrationBindingResult {
    /// Only estimates bound to a validated report are eligible for the existing
    /// EVI ranker. This does not authorize sensor capture or inference.
    pub eligible: Vec<RoiValueEstimate>,
    pub rejected: Vec<RoiCalibrationBindingRejection>,
}

pub fn compute_roi_calibration_report_digest(
    report: &RoiCalibrationReportV1,
) -> Result<String, RoiCalibrationError> {
    let mut material = report.clone();
    material.report_digest.clear();
    let bytes = serde_json::to_vec(&material).map_err(|_| RoiCalibrationError::InvalidMetric)?;
    let mut hasher = blake3::Hasher::new_derive_key(ROI_CALIBRATION_DIGEST_CONTEXT);
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

/// Recompute metrics from held-out bins. The report is internally consistent
/// only if its claimed metrics and digest match these recomputations.
pub fn verify_roi_calibration_report(
    report: &RoiCalibrationReportV1,
) -> Result<RoiCalibrationVerification, RoiCalibrationError> {
    if report.schema != ROI_CALIBRATION_REPORT_SCHEMA_V1 {
        return Err(RoiCalibrationError::WrongSchema);
    }
    validate_identifier(&report.report_id)?;
    validate_identifier(&report.task_family)?;
    for digest in [
        &report.model_digest,
        &report.calibration_dataset_digest,
        &report.evaluation_dataset_digest,
        &report.calibration_split_digest,
        &report.evaluation_split_digest,
        &report.report_digest,
    ] {
        validate_digest(digest)?;
    }
    if report.calibration_dataset_digest == report.evaluation_dataset_digest
        || report.calibration_split_digest == report.evaluation_split_digest
    {
        return Err(RoiCalibrationError::DatasetOrSplitLeakage);
    }
    if report.calibration_sample_count < MIN_CALIBRATION_SAMPLES
        || report.evaluation_sample_count < MIN_EVALUATION_SAMPLES
    {
        return Err(RoiCalibrationError::InsufficientSamples);
    }
    if !(5..=100).contains(&report.bins.len()) {
        return Err(RoiCalibrationError::InvalidBinCount);
    }

    let mut count_sum = 0_u64;
    let mut weighted_ece = 0.0_f64;
    let mut weighted_brier = 0.0_f64;
    for bin in &report.bins {
        if bin.sample_count == 0
            || !finite_unit(bin.mean_confidence)
            || !finite_unit(bin.observed_accuracy)
            || !finite_unit(bin.mean_brier_score)
        {
            return Err(RoiCalibrationError::InvalidBin);
        }
        count_sum = count_sum.checked_add(bin.sample_count).ok_or(RoiCalibrationError::BinSampleCountMismatch)?;
        let weight = bin.sample_count as f64;
        weighted_ece += weight * (bin.mean_confidence - bin.observed_accuracy).abs() as f64;
        weighted_brier += weight * bin.mean_brier_score as f64;
    }
    if count_sum != report.evaluation_sample_count {
        return Err(RoiCalibrationError::BinSampleCountMismatch);
    }
    let n = report.evaluation_sample_count as f64;
    let ece = (weighted_ece / n) as f32;
    let brier = (weighted_brier / n) as f32;
    if !finite_unit(report.expected_calibration_error) || !finite_unit(report.brier_score) {
        return Err(RoiCalibrationError::InvalidMetric);
    }
    if (ece - report.expected_calibration_error).abs() > METRIC_TOLERANCE
        || (brier - report.brier_score).abs() > METRIC_TOLERANCE
    {
        return Err(RoiCalibrationError::MetricMismatch);
    }
    let digest = compute_roi_calibration_report_digest(report)?;
    if digest != report.report_digest {
        return Err(RoiCalibrationError::ReportDigestMismatch);
    }

    Ok(RoiCalibrationVerification {
        report_id: report.report_id.clone(),
        report_digest: report.report_digest.clone(),
        model_digest: report.model_digest.clone(),
        task_family: report.task_family.clone(),
        calibration_sample_count: report.calibration_sample_count,
        evaluation_sample_count: report.evaluation_sample_count,
        recomputed_expected_calibration_error: ece,
        recomputed_brier_score: brier,
    })
}

/// Bind each ROI estimate to one unique, internally consistent calibration
/// report. Missing, duplicate, forged, wrong-model, and wrong-family reports
/// are rejected rather than silently treated as calibrated.
pub fn bind_roi_estimates_to_calibration_reports(
    estimates: &[RoiValueEstimate],
    reports: &[RoiCalibrationReportV1],
) -> RoiCalibrationBindingResult {
    let mut by_id: HashMap<&str, Vec<&RoiCalibrationReportV1>> = HashMap::new();
    for report in reports {
        by_id.entry(report.report_id.as_str()).or_default().push(report);
    }

    let mut eligible = Vec::new();
    let mut rejected = Vec::new();
    for estimate in estimates {
        let Some(matches) = by_id.get(estimate.calibration.calibration_id.as_str()) else {
            rejected.push(rejection(estimate, RoiCalibrationBindingError::MissingReport));
            continue;
        };
        if matches.len() != 1 {
            rejected.push(rejection(estimate, RoiCalibrationBindingError::AmbiguousReport));
            continue;
        }
        let report = matches[0];
        if let Err(error) = verify_roi_calibration_report(report) {
            rejected.push(rejection(estimate, RoiCalibrationBindingError::InvalidReport(error)));
            continue;
        }
        let reason = if estimate.calibration.calibration_id != report.report_id
            || estimate.calibration.calibration_report_digest != report.report_digest
        {
            Some(RoiCalibrationBindingError::ReportIdentityMismatch)
        } else if estimate.calibration.model_digest != report.model_digest {
            Some(RoiCalibrationBindingError::ModelMismatch)
        } else if estimate.task_family != report.task_family
            || estimate.calibration.task_family != report.task_family
        {
            Some(RoiCalibrationBindingError::TaskFamilyMismatch)
        } else if estimate.calibration.sample_count != report.calibration_sample_count {
            Some(RoiCalibrationBindingError::SampleCountMismatch)
        } else if (estimate.calibration.expected_calibration_error - report.expected_calibration_error).abs() > METRIC_TOLERANCE {
            Some(RoiCalibrationBindingError::EceMismatch)
        } else {
            None
        };
        if let Some(reason) = reason {
            rejected.push(rejection(estimate, reason));
        } else {
            eligible.push(estimate.clone());
        }
    }
    rejected.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    eligible.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    RoiCalibrationBindingResult { eligible, rejected }
}

fn rejection(estimate: &RoiValueEstimate, reason: RoiCalibrationBindingError) -> RoiCalibrationBindingRejection {
    RoiCalibrationBindingRejection { candidate_id: estimate.candidate_id.clone(), reason }
}

fn finite_unit(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn validate_identifier(value: &str) -> Result<(), RoiCalibrationError> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(RoiCalibrationError::InvalidIdentifier);
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), RoiCalibrationError> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(RoiCalibrationError::InvalidDigest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_foveation::RoiCalibrationEvidence;

    fn digest(ch: u8) -> String { std::iter::repeat_n(ch as char, 64).collect() }

    fn report() -> RoiCalibrationReportV1 {
        let mut value = RoiCalibrationReportV1 {
            schema: ROI_CALIBRATION_REPORT_SCHEMA_V1.to_owned(),
            report_id: "calibration-001".to_owned(),
            model_digest: digest(b'a'),
            task_family: "turn_floor_state".to_owned(),
            calibration_dataset_digest: digest(b'b'),
            evaluation_dataset_digest: digest(b'c'),
            calibration_split_digest: digest(b'd'),
            evaluation_split_digest: digest(b'e'),
            calibration_sample_count: 150,
            evaluation_sample_count: 500,
            bins: (0..10).map(|_| RoiCalibrationBinV1 {
                sample_count: 50,
                mean_confidence: 0.80,
                observed_accuracy: 0.78,
                mean_brier_score: 0.18,
            }).collect(),
            expected_calibration_error: 0.02,
            brier_score: 0.18,
            report_digest: String::new(),
        };
        value.report_digest = compute_roi_calibration_report_digest(&value).unwrap();
        value
    }

    fn estimate(report: &RoiCalibrationReportV1) -> RoiValueEstimate {
        RoiValueEstimate {
            candidate_id: "roi-001".to_owned(),
            task_id: "task-001".to_owned(),
            task_family: report.task_family.clone(),
            source_clock_domain_id: "clock:camera-1".to_owned(),
            source_frame_id: 1,
            source_timestamp_us: 1_000,
            expected_loss_without_roi: 0.8,
            expected_loss_with_roi: 0.5,
            expected_loss_reduction_lcb: 0.2,
            confidence_level: 0.95,
            latency_penalty: 0.01,
            staleness_penalty: 0.01,
            source_pixel_cost: 10_000,
            calibration: RoiCalibrationEvidence {
                calibration_id: report.report_id.clone(),
                model_digest: report.model_digest.clone(),
                calibration_report_digest: report.report_digest.clone(),
                task_family: report.task_family.clone(),
                sample_count: report.calibration_sample_count,
                expected_calibration_error: report.expected_calibration_error,
            },
        }
    }

    #[test]
    fn report_recomputes_ece_and_brier() {
        let checked = verify_roi_calibration_report(&report()).unwrap();
        assert!((checked.recomputed_expected_calibration_error - 0.02).abs() < 0.000_01);
        assert!((checked.recomputed_brier_score - 0.18).abs() < 0.000_01);
    }

    #[test]
    fn claimed_ece_must_match_bin_recomputation() {
        let mut value = report();
        value.expected_calibration_error = 0.0;
        value.report_digest = compute_roi_calibration_report_digest(&value).unwrap();
        assert_eq!(verify_roi_calibration_report(&value), Err(RoiCalibrationError::MetricMismatch));
    }

    #[test]
    fn duplicate_dataset_or_split_is_rejected() {
        let mut value = report();
        value.evaluation_dataset_digest = value.calibration_dataset_digest.clone();
        value.report_digest = compute_roi_calibration_report_digest(&value).unwrap();
        assert_eq!(verify_roi_calibration_report(&value), Err(RoiCalibrationError::DatasetOrSplitLeakage));
    }

    #[test]
    fn report_digest_detects_content_mutation() {
        let mut value = report();
        value.bins[0].observed_accuracy = 0.7;
        assert_eq!(verify_roi_calibration_report(&value), Err(RoiCalibrationError::MetricMismatch));
    }

    #[test]
    fn fewer_than_one_hundred_calibration_samples_is_rejected() {
        let mut value = report();
        value.calibration_sample_count = 99;
        value.report_digest = compute_roi_calibration_report_digest(&value).unwrap();
        assert_eq!(verify_roi_calibration_report(&value), Err(RoiCalibrationError::InsufficientSamples));
    }

    #[test]
    fn bin_counts_must_cover_the_held_out_set() {
        let mut value = report();
        value.bins[0].sample_count = 49;
        value.report_digest = compute_roi_calibration_report_digest(&value).unwrap();
        assert_eq!(verify_roi_calibration_report(&value), Err(RoiCalibrationError::BinSampleCountMismatch));
    }

    #[test]
    fn estimates_bind_to_model_task_family_and_report_digest() {
        let value = report();
        let estimate = estimate(&value);
        let result = bind_roi_estimates_to_calibration_reports(&[estimate], &[value]);
        assert_eq!(result.eligible.len(), 1);
        assert!(result.rejected.is_empty());
    }

    #[test]
    fn estimate_with_wrong_model_is_rejected() {
        let value = report();
        let mut candidate = estimate(&value);
        candidate.calibration.model_digest = digest(b'f');
        let result = bind_roi_estimates_to_calibration_reports(&[candidate], &[value]);
        assert_eq!(result.rejected[0].reason, RoiCalibrationBindingError::ModelMismatch);
    }

    #[test]
    fn ambiguous_report_ids_are_rejected() {
        let value = report();
        let result = bind_roi_estimates_to_calibration_reports(
            &[estimate(&value)],
            &[value.clone(), value],
        );
        assert_eq!(result.rejected[0].reason, RoiCalibrationBindingError::AmbiguousReport);
    }

    #[test]
    fn report_digest_is_deterministic() {
        let value = report();
        assert_eq!(
            compute_roi_calibration_report_digest(&value).unwrap(),
            compute_roi_calibration_report_digest(&value).unwrap()
        );
    }
}
