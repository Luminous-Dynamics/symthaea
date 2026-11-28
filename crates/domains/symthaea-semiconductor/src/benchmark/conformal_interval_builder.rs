// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finite-sample split-conformal interval construction from K1-qualified,
//! chronologically available semiconductor metrology labels.
//!
//! The caller supplies frozen prior prediction residuals. This module checks
//! their K1 evidence, computes the corrected finite-sample order statistic,
//! and emits an immutable prediction-set receipt. It does not claim
//! conditional coverage under adaptive selection or arbitrary drift.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
    LabelEvidenceFrontierV1,
};

pub const CONFORMAL_SCORE_POLICY_SCHEMA_V1: &str =
    "luminous.semi-bench.conformal-score-policy.v1";
pub const CONFORMAL_CALIBRATION_SAMPLE_SCHEMA_V1: &str =
    "luminous.semi-bench.conformal-calibration-sample.v1";
pub const CONFORMAL_INTERVAL_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.conformal-interval-input.v1";
pub const CONFORMAL_SET_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.conformal-set-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConformalScoreKindV1 {
    AbsoluteResidual,
    ScaleNormalizedAbsoluteResidual,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConformalSetBuildStatusV1 {
    Ready,
    FullSetRequired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConformalSetBuildReasonV1 {
    CalibrationWindowEmpty,
    InsufficientCalibrationForFiniteQuantile,
    CalibrationLabelNotCurrentK1,
    CalibrationLabelNotEligible,
    CalibrationReportDigestMismatch,
    CalibrationLabelIdentityMismatch,
    CalibrationPayloadDigestMismatch,
    CalibrationNumericValueMismatch,
    CalibrationSourceNotAvailableAtCutoff,
    CalibrationPredictionChronologyInvalid,
    CalibrationFeatureEventReused,
    CalibrationLabelRevisionDuplicate,
    CalibrationSampleDuplicate,
    CalibrationModelArtifactMismatch,
    CalibrationScorePolicyMismatch,
    CalibrationMeasurementContractMismatch,
    CalibrationEventDuplicate,
    CalibrationSourceAuthenticityAssertionMissing,
    UpstreamAuthorityFlagSet,
    InvalidCalibrationScale,
    InvalidPredictionScale,
    InvalidPredictionInterval,
    CalibrationPolicyDigestMismatch,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalScorePolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub score_kind: ConformalScoreKindV1,
    /// Positive lower bound for per-sample scale in normalized-residual mode.
    pub scale_floor: f64,
    /// Retain the most recently available K1 calibration outcomes.
    pub maximum_calibration_samples: usize,
    pub require_source_authenticity_assertion: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalCalibrationSampleV1 {
    pub schema: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub label_payload_sha256: String,
    pub global_model_artifact_sha256: String,
    pub score_policy_sha256: String,
    pub measurement_contract_sha256: String,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub label_available_at_unix_seconds: i64,
    pub eligibility_report_available_at_unix_seconds: i64,
    pub prediction_value: f64,
    pub prediction_scale: f64,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalIntervalInputV1 {
    pub schema: String,
    pub prediction_id: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub measurement_contract_sha256: String,
    pub point_prediction: f64,
    pub prediction_scale: f64,
    pub target_miscoverage: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalCalibrationScoreV1 {
    pub label_id: String,
    pub label_revision: u64,
    pub report_sha256: String,
    pub label_payload_sha256: String,
    pub label_available_at_unix_seconds: i64,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalSetReceiptV1 {
    pub schema: String,
    pub prediction_id: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub score_policy_sha256: String,
    pub measurement_contract_sha256: String,
    pub score_kind: ConformalScoreKindV1,
    pub target_miscoverage: f64,
    pub finite_sample_quantile_rank: usize,
    pub calibration_sample_count: usize,
    pub calibration_window_label_ids: Vec<String>,
    pub calibration_scores: Vec<ConformalCalibrationScoreV1>,
    pub point_prediction: f64,
    pub prediction_scale: f64,
    pub conformal_quantile: Option<f64>,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub status: ConformalSetBuildStatusV1,
    pub reasons: Vec<ConformalSetBuildReasonV1>,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct ConformalSetBuilderErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), ConformalSetBuilderErrorV1> {
    if value.trim().is_empty() {
        return Err(ConformalSetBuilderErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, ConformalSetBuilderErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        ConformalSetBuilderErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, ConformalSetBuilderErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn validate_policy(policy: &ConformalScorePolicyV1) -> Result<(), ConformalSetBuilderErrorV1> {
    if policy.schema != CONFORMAL_SCORE_POLICY_SCHEMA_V1 {
        return Err(ConformalSetBuilderErrorV1(
            "conformal score policy schema is unsupported".into(),
        ));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if policy.revision == 0
        || !policy.scale_floor.is_finite()
        || policy.scale_floor <= 0.0
        || policy.maximum_calibration_samples == 0
    {
        return Err(ConformalSetBuilderErrorV1(
            "conformal score policy has invalid numeric or window parameters".into(),
        ));
    }
    Ok(())
}

fn calibration_score(
    policy: &ConformalScorePolicyV1,
    sample: &ConformalCalibrationSampleV1,
    prediction: &ConformalIntervalInputV1,
    used_labels: &mut BTreeSet<String>,
    used_revisions: &mut BTreeSet<String>,
    used_events: &mut BTreeSet<String>,
    used_samples: &mut BTreeSet<String>,
    expected_policy_sha256: &str,
) -> Result<ConformalCalibrationScoreV1, ConformalSetBuildReasonV1> {
    let report = &sample.eligibility_report;
    if sample.schema != CONFORMAL_CALIBRATION_SAMPLE_SCHEMA_V1
        || report.schema != "luminous.semi-bench.label-consumption-report.v1"
    {
        return Err(ConformalSetBuildReasonV1::CalibrationReportDigestMismatch);
    }
    if report_digest(report).map_err(|_| ConformalSetBuildReasonV1::CalibrationReportDigestMismatch)?
        != report.report_sha256
    {
        return Err(ConformalSetBuildReasonV1::CalibrationReportDigestMismatch);
    }
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        return Err(ConformalSetBuildReasonV1::CalibrationLabelNotCurrentK1);
    }
    if report.eligibility != LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
        || !report.downstream_consumption_allowed
    {
        return Err(ConformalSetBuildReasonV1::CalibrationLabelNotEligible);
    }
    if report.training_snapshot_mutated || report.equipment_control_authority {
        return Err(ConformalSetBuildReasonV1::UpstreamAuthorityFlagSet);
    }
    if policy.require_source_authenticity_assertion && !report.source_authenticity_proven {
        return Err(ConformalSetBuildReasonV1::CalibrationSourceAuthenticityAssertionMissing);
    }
    if report.label_id != sample.label_id
        || report.label_revision != sample.label_revision
        || report.measurement_event_id != sample.measurement_event_id
    {
        return Err(ConformalSetBuildReasonV1::CalibrationLabelIdentityMismatch);
    }
    if !valid_sha256(&sample.label_payload_sha256)
        || sample.label_payload_sha256 != report.label_sha256
    {
        return Err(ConformalSetBuildReasonV1::CalibrationPayloadDigestMismatch);
    }
    if !valid_sha256(&report.measurement_sha256)
        || !valid_sha256(&report.qualification_report_sha256)
        || !valid_sha256(&report.requalification_report_sha256)
        || !valid_sha256(&report.requalification_frontier_sha256)
    {
        return Err(ConformalSetBuildReasonV1::CalibrationReportDigestMismatch);
    }
    if sample.global_model_artifact_sha256 != prediction.global_model_artifact_sha256
        || !valid_sha256(&sample.global_model_artifact_sha256)
    {
        return Err(ConformalSetBuildReasonV1::CalibrationModelArtifactMismatch);
    }
    if sample.score_policy_sha256 != expected_policy_sha256 {
        return Err(ConformalSetBuildReasonV1::CalibrationScorePolicyMismatch);
    }
    if sample.measurement_contract_sha256 != prediction.measurement_contract_sha256
        || !valid_sha256(&sample.measurement_contract_sha256)
    {
        return Err(ConformalSetBuildReasonV1::CalibrationMeasurementContractMismatch);
    }
    if !report.qualified_corrected_value.is_finite()
        || !report
            .qualified_standard_uncertainty_upper_bound
            .is_some_and(|value| value.is_finite() && value >= 0.0)
        || !sample.prediction_value.is_finite()
        || !sample.prediction_scale.is_finite()
    {
        return Err(ConformalSetBuildReasonV1::InvalidCalibrationScale);
    }
    if sample.prediction_at_unix_seconds < sample.event_time_unix_seconds
        || sample.label_available_at_unix_seconds < sample.prediction_at_unix_seconds
        || sample.eligibility_report_available_at_unix_seconds
            < sample.label_available_at_unix_seconds
    {
        return Err(ConformalSetBuildReasonV1::CalibrationPredictionChronologyInvalid);
    }
    if sample.label_available_at_unix_seconds > prediction.prediction_at_unix_seconds
        || sample.eligibility_report_available_at_unix_seconds
            > prediction.prediction_at_unix_seconds
    {
        return Err(ConformalSetBuildReasonV1::CalibrationSourceNotAvailableAtCutoff);
    }
    if sample.sample_id == prediction.sample_id
        || sample.measurement_event_id == prediction.measurement_event_id
    {
        return Err(ConformalSetBuildReasonV1::CalibrationFeatureEventReused);
    }
    if !used_samples.insert(sample.sample_id.clone()) {
        return Err(ConformalSetBuildReasonV1::CalibrationSampleDuplicate);
    }
    if !used_labels.insert(sample.label_id.clone())
        || !used_events.insert(sample.measurement_event_id.clone())
    {
        return Err(ConformalSetBuildReasonV1::CalibrationEventDuplicate);
    }
    if !used_revisions.insert(format!("{}:{}", sample.label_id, sample.label_revision)) {
        return Err(ConformalSetBuildReasonV1::CalibrationLabelRevisionDuplicate);
    }
    let residual = (report.qualified_corrected_value - sample.prediction_value).abs();
    let score = match policy.score_kind {
        ConformalScoreKindV1::AbsoluteResidual => residual,
        ConformalScoreKindV1::ScaleNormalizedAbsoluteResidual => {
            if sample.prediction_scale <= 0.0 || sample.prediction_scale < policy.scale_floor {
                return Err(ConformalSetBuildReasonV1::InvalidCalibrationScale);
            }
            residual / sample.prediction_scale
        }
    };
    if !score.is_finite() {
        return Err(ConformalSetBuildReasonV1::InvalidCalibrationScale);
    }
    Ok(ConformalCalibrationScoreV1 {
        label_id: sample.label_id.clone(),
        label_revision: sample.label_revision,
        report_sha256: report.report_sha256.clone(),
        label_payload_sha256: sample.label_payload_sha256.clone(),
        label_available_at_unix_seconds: sample.label_available_at_unix_seconds,
        score,
    })
}

/// Builds a finite-sample two-sided conformal interval using the corrected
/// order statistic k=ceil((n+1)(1-alpha)). If k>n, returns FullSetRequired
/// rather than emitting an artificially narrow interval.
pub fn build_conformal_interval_v1(
    policy: &ConformalScorePolicyV1,
    input: &ConformalIntervalInputV1,
    calibration_samples: &[ConformalCalibrationSampleV1],
) -> Result<ConformalSetReceiptV1, ConformalSetBuilderErrorV1> {
    validate_policy(policy)?;
    if input.schema != CONFORMAL_INTERVAL_INPUT_SCHEMA_V1 {
        return Err(ConformalSetBuilderErrorV1(
            "conformal interval input schema is unsupported".into(),
        ));
    }
    require_text(&input.prediction_id, "input.prediction_id")?;
    require_text(&input.sample_id, "input.sample_id")?;
    require_text(&input.measurement_event_id, "input.measurement_event_id")?;
    require_text(&input.clock_domain_id, "input.clock_domain_id")?;
    if !valid_sha256(&input.global_model_artifact_sha256)
        || !valid_sha256(&input.measurement_contract_sha256)
        || !input.point_prediction.is_finite()
        || !input.prediction_scale.is_finite()
        || input.prediction_scale <= 0.0
        || !input.target_miscoverage.is_finite()
        || !(0.0..1.0).contains(&input.target_miscoverage)
    {
        return Err(ConformalSetBuilderErrorV1(
            "conformal interval input contains invalid digest or numeric parameters".into(),
        ));
    }

    let policy_sha256 = json_sha256(policy)?;
    let mut samples = calibration_samples.to_vec();
    samples.sort_by(|a, b| {
        a.label_available_at_unix_seconds
            .cmp(&b.label_available_at_unix_seconds)
            .then_with(|| a.label_id.cmp(&b.label_id))
            .then_with(|| a.label_revision.cmp(&b.label_revision))
    });
    if samples.len() > policy.maximum_calibration_samples {
        samples = samples
            .split_off(samples.len() - policy.maximum_calibration_samples);
    }

    let mut used_labels = BTreeSet::new();
    let mut used_revisions = BTreeSet::new();
    let mut used_events = BTreeSet::new();
    let mut used_samples = BTreeSet::new();
    let mut scores = Vec::with_capacity(samples.len());
    let mut reasons = Vec::new();
    for sample in &samples {
        match calibration_score(
            policy,
            sample,
            input,
            &mut used_labels,
            &mut used_revisions,
            &mut used_events,
            &mut used_samples,
            &policy_sha256,
        ) {
            Ok(score) => scores.push(score),
            Err(reason) => reasons.push(reason),
        }
    }
    reasons.sort();
    reasons.dedup();
    if !reasons.is_empty() {
        return Err(ConformalSetBuilderErrorV1(format!(
            "calibration evidence rejected: {reasons:?}"
        )));
    }

    let n = scores.len();
    if n == 0 {
        reasons.push(ConformalSetBuildReasonV1::CalibrationWindowEmpty);
    }
    let quantile_rank = (((n + 1) as f64) * (1.0 - input.target_miscoverage)).ceil() as usize;
    let mut status = ConformalSetBuildStatusV1::Ready;
    let mut quantile = None;
    let mut lower = None;
    let mut upper = None;
    if quantile_rank == 0 || quantile_rank > n {
        status = ConformalSetBuildStatusV1::FullSetRequired;
        reasons.push(ConformalSetBuildReasonV1::InsufficientCalibrationForFiniteQuantile);
    } else {
        scores.sort_by(|a, b| {
            a.score
                .total_cmp(&b.score)
                .then_with(|| a.label_id.cmp(&b.label_id))
                .then_with(|| a.label_revision.cmp(&b.label_revision))
        });
        let q = scores[quantile_rank - 1].score;
        let radius = match policy.score_kind {
            ConformalScoreKindV1::AbsoluteResidual => q,
            ConformalScoreKindV1::ScaleNormalizedAbsoluteResidual => q * input.prediction_scale,
        };
        let lo = input.point_prediction - radius;
        let hi = input.point_prediction + radius;
        if !q.is_finite() || !radius.is_finite() || !lo.is_finite() || !hi.is_finite() {
            return Err(ConformalSetBuilderErrorV1(
                "conformal quantile or interval bounds overflowed".into(),
            ));
        }
        quantile = Some(q);
        lower = Some(lo);
        upper = Some(hi);
    }

    let mut receipt = ConformalSetReceiptV1 {
        schema: CONFORMAL_SET_RECEIPT_SCHEMA_V1.into(),
        prediction_id: input.prediction_id.clone(),
        sample_id: input.sample_id.clone(),
        measurement_event_id: input.measurement_event_id.clone(),
        prediction_at_unix_seconds: input.prediction_at_unix_seconds,
        clock_domain_id: input.clock_domain_id.clone(),
        global_model_artifact_sha256: input.global_model_artifact_sha256.clone(),
        score_policy_sha256: policy_sha256,
        measurement_contract_sha256: input.measurement_contract_sha256.clone(),
        score_kind: policy.score_kind,
        target_miscoverage: input.target_miscoverage,
        finite_sample_quantile_rank: quantile_rank,
        calibration_sample_count: n,
        calibration_window_label_ids: scores.iter().map(|score| score.label_id.clone()).collect(),
        calibration_scores: scores,
        point_prediction: input.point_prediction,
        prediction_scale: input.prediction_scale,
        conformal_quantile: quantile,
        lower_bound: lower,
        upper_bound: upper,
        status,
        reasons,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = {
        let mut canonical = receipt.clone();
        canonical.receipt_sha256.clear();
        json_sha256(&canonical)?
    };
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy(kind: ConformalScoreKindV1, maximum: usize) -> ConformalScorePolicyV1 {
        ConformalScorePolicyV1 {
            schema: CONFORMAL_SCORE_POLICY_SCHEMA_V1.into(),
            policy_id: "semiconductor-abs-residual.v1".into(),
            revision: 1,
            score_kind: kind,
            scale_floor: 1.0e-6,
            maximum_calibration_samples: maximum,
            require_source_authenticity_assertion: false,
        }
    }

    fn input(alpha: f64) -> ConformalIntervalInputV1 {
        ConformalIntervalInputV1 {
            schema: CONFORMAL_INTERVAL_INPUT_SCHEMA_V1.into(),
            prediction_id: "prediction-new".into(),
            sample_id: "sample-new".into(),
            measurement_event_id: "event-new".into(),
            prediction_at_unix_seconds: 1_800_100_000,
            clock_domain_id: "fab-clock".into(),
            global_model_artifact_sha256: digest('a'),
            measurement_contract_sha256: digest('m'),
            point_prediction: 100.0,
            prediction_scale: 2.0,
            target_miscoverage: alpha,
        }
    }

    fn sample(
        index: u64,
        residual: f64,
        available_at: i64,
        score_policy: &ConformalScorePolicyV1,
    ) -> ConformalCalibrationSampleV1 {
        let label_id = format!("label-{index}");
        let event_id = format!("event-{index}");
        let payload = digest(char::from(b'b' + (index % 20) as u8));
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label_id.clone(),
            label_revision: 1,
            measurement_event_id: event_id.clone(),
            calibration_certificate_id: "cal-1".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            downstream_consumption_allowed: true,
            reasons: Vec::new(),
            label_sha256: payload.clone(),
            measurement_sha256: digest('c'),
            qualified_corrected_value: 100.0 + residual,
            qualified_standard_uncertainty_upper_bound: Some(0.01),
            qualification_report_sha256: digest('d'),
            requalification_report_sha256: digest('e'),
            requalification_frontier_sha256: digest('f'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        let mut canonical = report.clone();
        canonical.report_sha256.clear();
        report.report_sha256 = json_sha256(&canonical).unwrap();
        ConformalCalibrationSampleV1 {
            schema: CONFORMAL_CALIBRATION_SAMPLE_SCHEMA_V1.into(),
            sample_id: format!("sample-{index}"),
            measurement_event_id: event_id,
            label_id,
            label_revision: 1,
            label_payload_sha256: payload,
            global_model_artifact_sha256: digest('a'),
            score_policy_sha256: json_sha256(score_policy).unwrap(),
            measurement_contract_sha256: digest('m'),
            event_time_unix_seconds: available_at - 120,
            prediction_at_unix_seconds: available_at - 60,
            label_available_at_unix_seconds: available_at,
            eligibility_report_available_at_unix_seconds: available_at + 1,
            prediction_value: 100.0,
            prediction_scale: 2.0,
            eligibility_report: report,
        }
    }

    #[test]
    fn finite_sample_quantile_uses_the_corrected_order_statistic() {
        let policy = policy(ConformalScoreKindV1::AbsoluteResidual, 50);
        let samples = (0..19)
            .map(|index| sample(index, index as f64 + 1.0, 1_800_000_000 + index as i64 * 10, &policy))
            .collect::<Vec<_>>();
        let output = build_conformal_interval_v1(&policy, &input(0.05), &samples).unwrap();
        assert_eq!(output.status, ConformalSetBuildStatusV1::Ready);
        assert_eq!(output.finite_sample_quantile_rank, 19);
        assert_eq!(output.conformal_quantile, Some(19.0));
        assert_eq!(output.lower_bound, Some(81.0));
        assert_eq!(output.upper_bound, Some(119.0));
    }

    #[test]
    fn too_few_qualified_calibration_labels_returns_full_set_state() {
        let policy = policy(ConformalScoreKindV1::AbsoluteResidual, 50);
        let samples = (0..18)
            .map(|index| sample(index, index as f64 + 1.0, 1_800_000_000 + index as i64 * 10, &policy))
            .collect::<Vec<_>>();
        let output = build_conformal_interval_v1(&policy, &input(0.05), &samples).unwrap();
        assert_eq!(output.status, ConformalSetBuildStatusV1::FullSetRequired);
        assert!(output.lower_bound.is_none());
        assert!(output.upper_bound.is_none());
        assert!(output.reasons.contains(&ConformalSetBuildReasonV1::InsufficientCalibrationForFiniteQuantile));
    }

    #[test]
    fn labels_not_available_by_prediction_cutoff_are_rejected() {
        let policy = policy(ConformalScoreKindV1::AbsoluteResidual, 50);
        let mut candidate = sample(0, 1.0, 1_800_100_000, &policy);
        candidate.eligibility_report_available_at_unix_seconds = 1_800_100_001;
        assert!(
            build_conformal_interval_v1(&policy, &input(0.05), &[candidate]).is_err()
        );
    }

    #[test]
    fn blocked_or_unknown_k1_label_cannot_enter_calibration_buffer() {
        let policy = policy(ConformalScoreKindV1::AbsoluteResidual, 50);
        let mut candidate = sample(0, 1.0, 1_800_000_100, &policy);
        candidate.eligibility_report.eligibility = LabelConsumptionEligibilityV1::Blocked;
        candidate.eligibility_report.downstream_consumption_allowed = false;
        let mut canonical = candidate.eligibility_report.clone();
        canonical.report_sha256.clear();
        candidate.eligibility_report.report_sha256 = json_sha256(&canonical).unwrap();
        assert!(
            build_conformal_interval_v1(&policy, &input(0.05), &[candidate]).is_err()
        );
    }

    #[test]
    fn normalized_score_uses_sample_scale_and_current_prediction_scale() {
        let policy = policy(ConformalScoreKindV1::ScaleNormalizedAbsoluteResidual, 50);
        let samples = (0..19)
            .map(|index| sample(index, (index as f64 + 1.0) * 2.0, 1_800_000_000 + index as i64 * 10, &policy))
            .collect::<Vec<_>>();
        let output = build_conformal_interval_v1(&policy, &input(0.05), &samples).unwrap();
        assert_eq!(output.conformal_quantile, Some(19.0));
        assert_eq!(output.lower_bound, Some(62.0));
        assert_eq!(output.upper_bound, Some(138.0));
    }

    #[test]
    fn calibration_window_is_capped_by_label_availability_order() {
        let policy = policy(ConformalScoreKindV1::AbsoluteResidual, 19);
        let samples = (0..25)
            .map(|index| sample(index, index as f64 + 1.0, 1_800_000_000 + index as i64 * 10, &policy))
            .collect::<Vec<_>>();
        let output = build_conformal_interval_v1(&policy, &input(0.05), &samples).unwrap();
        assert_eq!(output.calibration_sample_count, 19);
        assert!(output.calibration_window_label_ids.contains(&"label-24".into()));
        assert!(!output.calibration_window_label_ids.contains(&"label-0".into()));
    }

    #[test]
    fn set_receipt_is_deterministic_under_input_order_permutation() {
        let policy = policy(ConformalScoreKindV1::AbsoluteResidual, 50);
        let mut samples = (0..19)
            .map(|index| sample(index, index as f64 + 1.0, 1_800_000_000 + index as i64 * 10, &policy))
            .collect::<Vec<_>>();
        let first = build_conformal_interval_v1(&policy, &input(0.05), &samples).unwrap();
        samples.reverse();
        let second = build_conformal_interval_v1(&policy, &input(0.05), &samples).unwrap();
        assert_eq!(first.receipt_sha256, second.receipt_sha256);
    }
}
