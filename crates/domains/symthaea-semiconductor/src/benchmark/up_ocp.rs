// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Universal-portfolio online conformal prediction (UP-OCP) reference controller.
//!
//! V1 reproduces the author's scalar-radius, immediate ordered-feedback update.
//! The alpha correction found in the pinned author code is explicit policy data;
//! setting it to zero yields a separately labelled unshifted track. Missing or
//! forfeited outcomes invalidate the original complete-feedback guarantee.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};

pub const UP_OCP_ENGINE_SCHEMA_V1: &str = "luminous.semi-bench.up-ocp-engine.v1";
pub const UP_OCP_POLICY_SCHEMA_V1: &str = "luminous.semi-bench.up-ocp-policy.v1";
pub const UP_OCP_PREDICTION_SCHEMA_V1: &str = "luminous.semi-bench.up-ocp-prediction.v1";
pub const UP_OCP_PREDICTION_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.up-ocp-prediction-receipt.v1";
pub const UP_OCP_FEEDBACK_SCHEMA_V1: &str = "luminous.semi-bench.up-ocp-feedback.v1";
pub const UP_OCP_UPDATE_RECEIPT_SCHEMA_V1: &str = "luminous.semi-bench.up-ocp-update-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpOcpMixtureV1 {
    DirichletHalfHalf,
    Uniform,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpOcpGuaranteeStateV1 {
    CompleteOrderedFeedback,
    InvalidatedByForfeit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpOcpFeedbackDispositionV1 {
    Updated,
    HeldUnknown,
    Blocked,
    Rejected,
    Duplicate,
    Forfeited,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpOcpReasonV1 {
    LabelNotCurrentK1,
    LabelNotEligible,
    DownstreamConsumptionNotAllowed,
    LabelIdMismatch,
    LabelRevisionMismatch,
    MeasurementEventMismatch,
    LabelPayloadDigestMismatch,
    EligibilityReportDigestMismatch,
    ReportValueNonFinite,
    ClockDomainMismatch,
    LabelAvailableBeforeMeasurement,
    ReportBeforeLabelAvailability,
    ReportUnavailableAtDelivery,
    DeliveryBeforePrediction,
    DuplicateAcceptedLabelRevision,
    PendingPredictionMissing,
    PredictionRoundMismatch,
    AlphaOutOfRange,
    InvalidNonconformityScore,
    WealthOrRadiusDomainViolation,
    GuaranteeAlreadyInvalidated,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpOcpPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    /// Requested marginal miscoverage level.
    pub target_alpha: f64,
    /// The upstream author's code subtracts 0.015 from alpha. Set to 0.0 for
    /// the unshifted theoretical comparison; never hide this correction.
    pub finite_sample_alpha_correction: f64,
    pub mixture: UpOcpMixtureV1,
    pub initial_wealth: f64,
    pub global_model_artifact_sha256: String,
    pub clock_domain_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpOcpAlgorithmStateV1 {
    /// The source implementation starts its time index at one.
    pub time_index: u64,
    pub wealth: f64,
    pub portfolio_fraction: f64,
    pub betting_fraction: f64,
    pub radius: f64,
    pub covered_count: u64,
    pub accepted_score_count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpOcpPredictionInputV1 {
    pub schema: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub feature_available_at_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub prediction_center: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpOcpPredictionReceiptV1 {
    pub schema: String,
    pub round_index: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub feature_available_at_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub prediction_center: f64,
    pub requested_alpha: f64,
    pub effective_alpha: f64,
    pub alpha_correction: f64,
    pub mixture: UpOcpMixtureV1,
    pub radius_used: f64,
    pub interval_lower: f64,
    pub interval_upper: f64,
    pub algorithm_state_before_sha256: String,
    pub algorithm_state_after_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpOcpEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub policy: UpOcpPolicyV1,
    pub algorithm_state: UpOcpAlgorithmStateV1,
    pub guarantee_state: UpOcpGuaranteeStateV1,
    pub next_round_index: u64,
    pub pending_prediction: Option<UpOcpPredictionReceiptV1>,
    pub seen_measurement_event_ids: BTreeSet<String>,
    pub accepted_label_revisions: BTreeSet<String>,
    pub forfeited_round_count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpOcpLabelFeedbackV1 {
    pub schema: String,
    pub round_index: u64,
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub measurement_event_id: String,
    pub label_payload_sha256: String,
    pub label_available_at_unix_seconds: i64,
    pub eligibility_report_available_at_unix_seconds: i64,
    pub delivered_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpOcpUpdateReceiptV1 {
    pub schema: String,
    pub engine_id: String,
    pub round_index: u64,
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub disposition: UpOcpFeedbackDispositionV1,
    pub reasons: Vec<UpOcpReasonV1>,
    pub prediction_receipt_sha256: Option<String>,
    pub eligibility_report_sha256: Option<String>,
    pub label_payload_sha256: String,
    pub nonconformity_score: Option<f64>,
    pub covered: Option<bool>,
    pub algorithm_state_before_sha256: String,
    pub algorithm_state_after_sha256: String,
    pub next_radius: Option<f64>,
    pub next_wealth: Option<f64>,
    pub next_portfolio_fraction: Option<f64>,
    pub next_betting_fraction: Option<f64>,
    pub guarantee_state: UpOcpGuaranteeStateV1,
    pub algorithm_state_mutated: bool,
    pub equipment_control_authority: bool,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct UpOcpErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), UpOcpErrorV1> {
    if value.trim().is_empty() {
        return Err(UpOcpErrorV1(format!("{field}: expected non-empty text")));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, UpOcpErrorV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| UpOcpErrorV1(format!("canonical JSON serialization failed: {error}")))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn algorithm_state_digest(state: &UpOcpAlgorithmStateV1) -> Result<String, UpOcpErrorV1> {
    json_sha256(state)
}

fn eligibility_report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, UpOcpErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn prediction_receipt_digest(receipt: &UpOcpPredictionReceiptV1) -> Result<String, UpOcpErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn update_receipt_digest(receipt: &UpOcpUpdateReceiptV1) -> Result<String, UpOcpErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn effective_alpha(policy: &UpOcpPolicyV1) -> Result<f64, UpOcpErrorV1> {
    if !policy.target_alpha.is_finite()
        || !policy.finite_sample_alpha_correction.is_finite()
        || policy.target_alpha <= 0.0
        || policy.target_alpha >= 1.0
        || policy.finite_sample_alpha_correction < 0.0
    {
        return Err(UpOcpErrorV1("requested alpha or correction is invalid".into()));
    }
    let effective = policy.target_alpha - policy.finite_sample_alpha_correction;
    if !effective.is_finite() || !(0.0..1.0).contains(&effective) {
        return Err(UpOcpErrorV1(
            "effective alpha is outside the open interval (0,1)".into(),
        ));
    }
    Ok(effective)
}

fn validate_policy(policy: &UpOcpPolicyV1) -> Result<(), UpOcpErrorV1> {
    if policy.schema != UP_OCP_POLICY_SCHEMA_V1 {
        return Err(UpOcpErrorV1("UP-OCP policy schema is unsupported".into()));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    require_text(&policy.clock_domain_id, "policy.clock_domain_id")?;
    if policy.revision == 0
        || !valid_sha256(&policy.global_model_artifact_sha256)
        || !policy.initial_wealth.is_finite()
        || policy.initial_wealth <= 0.0
    {
        return Err(UpOcpErrorV1("UP-OCP policy identity/wealth is invalid".into()));
    }
    let _ = effective_alpha(policy)?;
    Ok(())
}

fn validate_state(state: &UpOcpAlgorithmStateV1) -> Result<(), UpOcpErrorV1> {
    if state.time_index == 0
        || state.covered_count > state.accepted_score_count
        || !state.wealth.is_finite()
        || state.wealth <= 0.0
        || !state.portfolio_fraction.is_finite()
        || !(0.0..=1.0).contains(&state.portfolio_fraction)
        || !state.betting_fraction.is_finite()
        || !state.radius.is_finite()
        || state.radius < 0.0
    {
        return Err(UpOcpErrorV1("UP-OCP algorithm state is outside its domain".into()));
    }
    Ok(())
}

fn initial_state(policy: &UpOcpPolicyV1) -> Result<UpOcpAlgorithmStateV1, UpOcpErrorV1> {
    let alpha = effective_alpha(policy)?;
    let state = UpOcpAlgorithmStateV1 {
        time_index: 1,
        wealth: policy.initial_wealth,
        portfolio_fraction: 0.5,
        betting_fraction: 0.5 * (1.0 / alpha - 1.0 / (1.0 - alpha)),
        radius: 0.0,
        covered_count: 0,
        accepted_score_count: 0,
    };
    validate_state(&state)?;
    Ok(state)
}

pub fn create_up_ocp_engine_v1(
    engine_id: &str,
    policy: UpOcpPolicyV1,
) -> Result<UpOcpEngineV1, UpOcpErrorV1> {
    require_text(engine_id, "engine_id")?;
    validate_policy(&policy)?;
    let algorithm_state = initial_state(&policy)?;
    Ok(UpOcpEngineV1 {
        schema: UP_OCP_ENGINE_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        policy,
        algorithm_state,
        guarantee_state: UpOcpGuaranteeStateV1::CompleteOrderedFeedback,
        next_round_index: 0,
        pending_prediction: None,
        seen_measurement_event_ids: BTreeSet::new(),
        accepted_label_revisions: BTreeSet::new(),
        forfeited_round_count: 0,
    })
}

fn validate_engine(engine: &UpOcpEngineV1) -> Result<(), UpOcpErrorV1> {
    if engine.schema != UP_OCP_ENGINE_SCHEMA_V1 {
        return Err(UpOcpErrorV1("UP-OCP engine schema is unsupported".into()));
    }
    require_text(&engine.engine_id, "engine.engine_id")?;
    validate_policy(&engine.policy)?;
    validate_state(&engine.algorithm_state)
}

/// Issues the current scalar-radius prediction. Only one round may be
/// outstanding in theorem-parity mode; this makes feedback order explicit.
pub fn issue_up_ocp_prediction_v1(
    engine: &mut UpOcpEngineV1,
    input: &UpOcpPredictionInputV1,
) -> Result<UpOcpPredictionReceiptV1, UpOcpErrorV1> {
    validate_engine(engine)?;
    if input.schema != UP_OCP_PREDICTION_SCHEMA_V1 {
        return Err(UpOcpErrorV1("UP-OCP prediction schema is unsupported".into()));
    }
    require_text(&input.sample_id, "input.sample_id")?;
    require_text(&input.measurement_event_id, "input.measurement_event_id")?;
    require_text(&input.clock_domain_id, "input.clock_domain_id")?;
    if input.clock_domain_id != engine.policy.clock_domain_id
        || input.global_model_artifact_sha256 != engine.policy.global_model_artifact_sha256
    {
        return Err(UpOcpErrorV1(
            "UP-OCP prediction clock/model identity mismatch".into(),
        ));
    }
    if input.event_time_unix_seconds > input.feature_available_at_unix_seconds
        || input.feature_available_at_unix_seconds > input.prediction_at_unix_seconds
        || !input.prediction_center.is_finite()
    {
        return Err(UpOcpErrorV1("UP-OCP prediction chronology/value is invalid".into()));
    }
    if engine.pending_prediction.is_some() {
        return Err(UpOcpErrorV1(
            "previous UP-OCP round still awaits complete feedback".into(),
        ));
    }
    if engine.seen_measurement_event_ids.contains(&input.measurement_event_id) {
        return Err(UpOcpErrorV1(
            "measurement event has already been predicted".into(),
        ));
    }
    let next_round_index = engine
        .next_round_index
        .checked_add(1)
        .ok_or_else(|| UpOcpErrorV1("round index overflow".into()))?;
    let before = algorithm_state_digest(&engine.algorithm_state)?;
    let radius = engine.algorithm_state.radius;
    let lower = input.prediction_center - radius;
    let upper = input.prediction_center + radius;
    if !lower.is_finite() || !upper.is_finite() {
        return Err(UpOcpErrorV1("UP-OCP prediction interval is non-finite".into()));
    }
    let mut receipt = UpOcpPredictionReceiptV1 {
        schema: UP_OCP_PREDICTION_RECEIPT_SCHEMA_V1.into(),
        round_index: engine.next_round_index,
        sample_id: input.sample_id.clone(),
        measurement_event_id: input.measurement_event_id.clone(),
        event_time_unix_seconds: input.event_time_unix_seconds,
        feature_available_at_unix_seconds: input.feature_available_at_unix_seconds,
        prediction_at_unix_seconds: input.prediction_at_unix_seconds,
        clock_domain_id: input.clock_domain_id.clone(),
        global_model_artifact_sha256: input.global_model_artifact_sha256.clone(),
        prediction_center: input.prediction_center,
        requested_alpha: engine.policy.target_alpha,
        effective_alpha: effective_alpha(&engine.policy)?,
        alpha_correction: engine.policy.finite_sample_alpha_correction,
        mixture: engine.policy.mixture,
        radius_used: radius,
        interval_lower: lower,
        interval_upper: upper,
        algorithm_state_before_sha256: before.clone(),
        algorithm_state_after_sha256: before,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = prediction_receipt_digest(&receipt)?;
    engine.pending_prediction = Some(receipt.clone());
    engine.seen_measurement_event_ids.insert(input.measurement_event_id.clone());
    engine.next_round_index = next_round_index;
    Ok(receipt)
}

fn feedback_receipt(
    engine: &UpOcpEngineV1,
    feedback: &UpOcpLabelFeedbackV1,
    prediction: Option<&UpOcpPredictionReceiptV1>,
    disposition: UpOcpFeedbackDispositionV1,
    reasons: Vec<UpOcpReasonV1>,
    score: Option<f64>,
    covered: Option<bool>,
    before: &str,
    after: &str,
    next_state: Option<&UpOcpAlgorithmStateV1>,
) -> Result<UpOcpUpdateReceiptV1, UpOcpErrorV1> {
    let mut receipt = UpOcpUpdateReceiptV1 {
        schema: UP_OCP_UPDATE_RECEIPT_SCHEMA_V1.into(),
        engine_id: engine.engine_id.clone(),
        round_index: feedback.round_index,
        sample_id: feedback.sample_id.clone(),
        label_id: feedback.label_id.clone(),
        label_revision: feedback.label_revision,
        disposition,
        reasons,
        prediction_receipt_sha256: prediction.map(|value| value.receipt_sha256.clone()),
        eligibility_report_sha256: Some(feedback.eligibility_report.report_sha256.clone()),
        label_payload_sha256: feedback.label_payload_sha256.clone(),
        nonconformity_score: score,
        covered,
        algorithm_state_before_sha256: before.into(),
        algorithm_state_after_sha256: after.into(),
        next_radius: next_state.map(|state| state.radius),
        next_wealth: next_state.map(|state| state.wealth),
        next_portfolio_fraction: next_state.map(|state| state.portfolio_fraction),
        next_betting_fraction: next_state.map(|state| state.betting_fraction),
        guarantee_state: engine.guarantee_state,
        algorithm_state_mutated: next_state.is_some(),
        equipment_control_authority: false,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = update_receipt_digest(&receipt)?;
    Ok(receipt)
}

/// Applies one current K1-qualified score update. The score is independently
/// derived as |qualified corrected value - prediction center|; callers cannot
/// inject a different nonconformity score.
pub fn deliver_up_ocp_feedback_v1(
    engine: &mut UpOcpEngineV1,
    feedback: &UpOcpLabelFeedbackV1,
) -> Result<UpOcpUpdateReceiptV1, UpOcpErrorV1> {
    validate_engine(engine)?;
    if feedback.schema != UP_OCP_FEEDBACK_SCHEMA_V1 {
        return Err(UpOcpErrorV1("UP-OCP feedback schema is unsupported".into()));
    }
    require_text(&feedback.sample_id, "feedback.sample_id")?;
    require_text(&feedback.label_id, "feedback.label_id")?;
    require_text(&feedback.measurement_event_id, "feedback.measurement_event_id")?;
    require_text(&feedback.clock_domain_id, "feedback.clock_domain_id")?;
    if feedback.label_revision == 0 || !valid_sha256(&feedback.label_payload_sha256) {
        return Err(UpOcpErrorV1("label revision/payload digest is invalid".into()));
    }

    let before = algorithm_state_digest(&engine.algorithm_state)?;
    let report = &feedback.eligibility_report;
    let mut reasons = Vec::new();
    if engine.accepted_label_revisions.contains(&format!(
        "{}:{}",
        feedback.label_id, feedback.label_revision
    )) {
        reasons.push(UpOcpReasonV1::DuplicateAcceptedLabelRevision);
        return feedback_receipt(
            engine,
            feedback,
            engine.pending_prediction.as_ref(),
            UpOcpFeedbackDispositionV1::Duplicate,
            reasons,
            None,
            None,
            &before,
            &before,
            None,
        );
    }

    if let Some(prediction) = engine.pending_prediction.as_ref() {
        if prediction.round_index != feedback.round_index {
            reasons.push(UpOcpReasonV1::PredictionRoundMismatch);
        }
        if prediction.sample_id != feedback.sample_id {
            reasons.push(UpOcpReasonV1::LabelIdMismatch);
        }
        if prediction.measurement_event_id != feedback.measurement_event_id {
            reasons.push(UpOcpReasonV1::MeasurementEventMismatch);
        }
        if prediction.clock_domain_id != feedback.clock_domain_id {
            reasons.push(UpOcpReasonV1::ClockDomainMismatch);
        }
        if feedback.label_available_at_unix_seconds < prediction.event_time_unix_seconds {
            reasons.push(UpOcpReasonV1::LabelAvailableBeforeMeasurement);
        }
        if feedback.eligibility_report_available_at_unix_seconds
            < feedback.label_available_at_unix_seconds
        {
            reasons.push(UpOcpReasonV1::ReportBeforeLabelAvailability);
        }
        if feedback.delivered_at_unix_seconds
            < feedback.eligibility_report_available_at_unix_seconds
        {
            reasons.push(UpOcpReasonV1::ReportUnavailableAtDelivery);
        }
        if feedback.delivered_at_unix_seconds < prediction.prediction_at_unix_seconds {
            reasons.push(UpOcpReasonV1::DeliveryBeforePrediction);
        }
    } else {
        reasons.push(UpOcpReasonV1::PendingPredictionMissing);
    }

    if feedback.clock_domain_id != engine.policy.clock_domain_id {
        reasons.push(UpOcpReasonV1::ClockDomainMismatch);
    }
    if !report.qualified_corrected_value.is_finite() {
        reasons.push(UpOcpReasonV1::ReportValueNonFinite);
    }
    if !valid_sha256(&report.label_sha256)
        || report.label_sha256 != feedback.label_payload_sha256
    {
        reasons.push(UpOcpReasonV1::LabelPayloadDigestMismatch);
    }
    if eligibility_report_digest(report)? != report.report_sha256 {
        reasons.push(UpOcpReasonV1::EligibilityReportDigestMismatch);
    }
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        reasons.push(UpOcpReasonV1::LabelNotCurrentK1);
    }
    match report.eligibility {
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
            if report.downstream_consumption_allowed => {}
        LabelConsumptionEligibilityV1::Unknown => reasons.push(UpOcpReasonV1::LabelNotEligible),
        LabelConsumptionEligibilityV1::Blocked => reasons.push(UpOcpReasonV1::LabelNotEligible),
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => {
            reasons.push(UpOcpReasonV1::DownstreamConsumptionNotAllowed)
        }
    }
    if report.label_id != feedback.label_id {
        reasons.push(UpOcpReasonV1::LabelIdMismatch);
    }
    if report.label_revision != feedback.label_revision {
        reasons.push(UpOcpReasonV1::LabelRevisionMismatch);
    }
    if report.measurement_event_id != feedback.measurement_event_id {
        reasons.push(UpOcpReasonV1::MeasurementEventMismatch);
    }

    reasons.sort();
    reasons.dedup();
    let Some(prediction) = engine.pending_prediction.clone() else {
        let after = before.clone();
        return feedback_receipt(
            engine,
            feedback,
            None,
            UpOcpFeedbackDispositionV1::Rejected,
            reasons,
            None,
            None,
            &before,
            &after,
            None,
        );
    };

    if !reasons.is_empty() {
        let disposition = if report.eligibility == LabelConsumptionEligibilityV1::Unknown {
            UpOcpFeedbackDispositionV1::HeldUnknown
        } else if report.eligibility == LabelConsumptionEligibilityV1::Blocked {
            UpOcpFeedbackDispositionV1::Blocked
        } else {
            UpOcpFeedbackDispositionV1::Rejected
        };
        return feedback_receipt(
            engine,
            feedback,
            Some(&prediction),
            disposition,
            reasons,
            None,
            None,
            &before,
            &before,
            None,
        );
    }

    let score = (report.qualified_corrected_value - prediction.prediction_center).abs();
    let covered = score <= prediction.radius_used;
    let alpha = effective_alpha(&engine.policy)?;
    let gradient = if covered { alpha } else { -(1.0 - alpha) };
    let next_time = engine
        .algorithm_state
        .time_index
        .checked_add(1)
        .ok_or_else(|| UpOcpErrorV1("UP-OCP time index overflow".into()))?;
    let next_covered_count = engine
        .algorithm_state
        .covered_count
        .checked_add(u64::from(covered))
        .ok_or_else(|| UpOcpErrorV1("UP-OCP covered count overflow".into()))?;
    let next_score_count = engine
        .algorithm_state
        .accepted_score_count
        .checked_add(1)
        .ok_or_else(|| UpOcpErrorV1("UP-OCP score count overflow".into()))?;
    let next_wealth =
        engine.algorithm_state.wealth - gradient * engine.algorithm_state.radius;
    let t = engine.algorithm_state.time_index as f64;
    let next_covered = next_covered_count as f64;
    let next_fraction = match engine.policy.mixture {
        UpOcpMixtureV1::DirichletHalfHalf => (t - next_covered + 0.5) / (t + 1.0),
        UpOcpMixtureV1::Uniform => (t - next_covered + 1.0) / (t + 2.0),
    };
    let next_beta =
        -1.0 / (1.0 - alpha) + next_fraction * (1.0 / alpha + 1.0 / (1.0 - alpha));
    let next_radius = next_beta * next_wealth;
    let next_state = UpOcpAlgorithmStateV1 {
        time_index: next_time,
        wealth: next_wealth,
        portfolio_fraction: next_fraction,
        betting_fraction: next_beta,
        radius: next_radius,
        covered_count: next_covered_count,
        accepted_score_count: next_score_count,
    };
    if !score.is_finite() || next_wealth <= 0.0 || !next_radius.is_finite() || next_radius < 0.0 {
        return feedback_receipt(
            engine,
            feedback,
            Some(&prediction),
            UpOcpFeedbackDispositionV1::Rejected,
            vec![UpOcpReasonV1::WealthOrRadiusDomainViolation],
            Some(score),
            Some(covered),
            &before,
            &before,
            None,
        );
    }
    validate_state(&next_state)?;
    let after = algorithm_state_digest(&next_state)?;
    let receipt = feedback_receipt(
        engine,
        feedback,
        Some(&prediction),
        UpOcpFeedbackDispositionV1::Updated,
        Vec::new(),
        Some(score),
        Some(covered),
        &before,
        &after,
        Some(&next_state),
    )?;
    engine.algorithm_state = next_state;
    engine.accepted_label_revisions.insert(format!(
        "{}:{}",
        feedback.label_id, feedback.label_revision
    ));
    engine.pending_prediction = None;
    engine.next_round_index = engine
        .next_round_index
        .checked_add(1)
        .ok_or_else(|| UpOcpErrorV1("round index overflow".into()))?;
    Ok(receipt)
}

/// Explicitly drops a pending outcome. This does not alter the UP-OCP betting
/// state, but irreversibly de-certifies the original complete-feedback claim.
pub fn forfeit_pending_up_ocp_feedback_v1(
    engine: &mut UpOcpEngineV1,
    reason: &str,
) -> Result<(), UpOcpErrorV1> {
    validate_engine(engine)?;
    require_text(reason, "forfeit.reason")?;
    if engine.pending_prediction.take().is_none() {
        return Err(UpOcpErrorV1("no pending UP-OCP prediction to forfeit".into()));
    }
    engine.guarantee_state = UpOcpGuaranteeStateV1::InvalidatedByForfeit;
    engine.forfeited_round_count = engine
        .forfeited_round_count
        .checked_add(1)
        .ok_or_else(|| UpOcpErrorV1("forfeited round count overflow".into()))?;
    engine.next_round_index = engine
        .next_round_index
        .checked_add(1)
        .ok_or_else(|| UpOcpErrorV1("round index overflow".into()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy(mixture: UpOcpMixtureV1, correction: f64) -> UpOcpPolicyV1 {
        UpOcpPolicyV1 {
            schema: UP_OCP_POLICY_SCHEMA_V1.into(),
            policy_id: "up-ocp-reference-v1".into(),
            revision: 1,
            target_alpha: 0.10,
            finite_sample_alpha_correction: correction,
            mixture,
            initial_wealth: 1.0,
            global_model_artifact_sha256: digest('a'),
            clock_domain_id: "fab-clock".into(),
        }
    }

    fn engine(mixture: UpOcpMixtureV1, correction: f64) -> UpOcpEngineV1 {
        create_up_ocp_engine_v1("up-ocp-engine-001", policy(mixture, correction)).unwrap()
    }

    fn prediction(round: u64, sample: &str, event: &str, center: f64) -> UpOcpPredictionInputV1 {
        UpOcpPredictionInputV1 {
            schema: UP_OCP_PREDICTION_SCHEMA_V1.into(),
            sample_id: sample.into(),
            measurement_event_id: event.into(),
            event_time_unix_seconds: 100 + (round as i64) * 10,
            feature_available_at_unix_seconds: 101 + (round as i64) * 10,
            prediction_at_unix_seconds: 102 + (round as i64) * 10,
            clock_domain_id: "fab-clock".into(),
            global_model_artifact_sha256: digest('a'),
            prediction_center: center,
        }
    }

    fn report(
        label_id: &str,
        label_revision: u64,
        event_id: &str,
        payload_sha: &str,
        value: f64,
        eligibility: LabelConsumptionEligibilityV1,
        frontier: LabelEvidenceFrontierV1,
    ) -> LabelConsumptionEligibilityReportV1 {
        let mut value_report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label_id.into(),
            label_revision,
            measurement_event_id: event_id.into(),
            calibration_certificate_id: "cal-001".into(),
            frontier,
            eligibility,
            downstream_consumption_allowed: eligibility
                == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: payload_sha.into(),
            measurement_sha256: digest('b'),
            qualified_corrected_value: value,
            qualified_standard_uncertainty_upper_bound: Some(0.05),
            qualification_report_sha256: digest('c'),
            requalification_report_sha256: digest('d'),
            requalification_frontier_sha256: digest('e'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        value_report.report_sha256 = eligibility_report_digest(&value_report).unwrap();
        value_report
    }

    fn feedback(
        round_index: u64,
        sample: &str,
        label_id: &str,
        event_id: &str,
        value: f64,
        eligibility: LabelConsumptionEligibilityV1,
    ) -> UpOcpLabelFeedbackV1 {
        let payload = digest('f');
        UpOcpLabelFeedbackV1 {
            schema: UP_OCP_FEEDBACK_SCHEMA_V1.into(),
            round_index,
            sample_id: sample.into(),
            label_id: label_id.into(),
            label_revision: 1,
            measurement_event_id: event_id.into(),
            label_payload_sha256: payload.clone(),
            label_available_at_unix_seconds: 110 + (round_index as i64) * 10,
            eligibility_report_available_at_unix_seconds: 111 + (round_index as i64) * 10,
            delivered_at_unix_seconds: 112 + (round_index as i64) * 10,
            clock_domain_id: "fab-clock".into(),
            eligibility_report: report(
                label_id,
                1,
                event_id,
                &payload,
                value,
                eligibility,
                LabelEvidenceFrontierV1::CurrentAtK1,
            ),
        }
    }

    #[test]
    fn initial_state_matches_pinned_author_defaults_and_alpha_correction() {
        let engine = engine(UpOcpMixtureV1::DirichletHalfHalf, 0.015);
        let alpha = 0.085_f64;
        assert_eq!(engine.algorithm_state.time_index, 1);
        assert_eq!(engine.algorithm_state.wealth, 1.0);
        assert_eq!(engine.algorithm_state.portfolio_fraction, 0.5);
        assert_eq!(engine.algorithm_state.radius, 0.0);
        assert!((engine.algorithm_state.betting_fraction
            - 0.5 * (1.0 / alpha - 1.0 / (1.0 - alpha)))
        .abs()
            < 1.0e-12);
    }

    #[test]
    fn dirichlet_and_uniform_mixtures_follow_their_distinct_source_recurrences() {
        let mut dirichlet = engine(UpOcpMixtureV1::DirichletHalfHalf, 0.015);
        let mut uniform = engine(UpOcpMixtureV1::Uniform, 0.015);
        for (round, value) in [(0, 10.5), (1, 10.2)] {
            for current in [&mut dirichlet, &mut uniform] {
                issue_up_ocp_prediction_v1(
                    current,
                    &prediction(round, &format!("s{round}"), &format!("e{round}"), 10.0),
                )
                .unwrap();
            }
            let dir_receipt = deliver_up_ocp_feedback_v1(
                &mut dirichlet,
                &feedback(round, &format!("s{round}"), &format!("l{round}"), &format!("e{round}"), value, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy),
            ).unwrap();
            let uni_receipt = deliver_up_ocp_feedback_v1(
                &mut uniform,
                &feedback(round, &format!("s{round}"), &format!("l{round}"), &format!("e{round}"), value, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy),
            ).unwrap();
            assert_eq!(dir_receipt.disposition, UpOcpFeedbackDispositionV1::Updated);
            assert_eq!(uni_receipt.disposition, UpOcpFeedbackDispositionV1::Updated);
        }
        assert_ne!(dirichlet.algorithm_state.portfolio_fraction, uniform.algorithm_state.portfolio_fraction);
        assert_ne!(dirichlet.algorithm_state.radius, uniform.algorithm_state.radius);
    }

    #[test]
    fn accepted_feedback_matches_independent_scalar_universal_portfolio_oracle() {
        let mut engine = engine(UpOcpMixtureV1::DirichletHalfHalf, 0.015);
        let alpha = 0.085_f64;
        let mut wealth = 1.0_f64;
        let mut radius = 0.0_f64;
        let mut covered_count = 0_u64;
        let mut time_index = 1_u64;
        let mut expected_radii = Vec::new();
        for (round, score) in [0.5_f64, 0.2, 0.7, 1.0].into_iter().enumerate() {
            let centre = 10.0;
            issue_up_ocp_prediction_v1(
                &mut engine,
                &prediction(round as u64, &format!("s{round}"), &format!("e{round}"), centre),
            )
            .unwrap();
            let covered = score <= radius;
            let g = if covered { alpha } else { -(1.0 - alpha) };
            covered_count += u64::from(covered);
            wealth -= g * radius;
            let x = (time_index as f64 - covered_count as f64 + 0.5)
                / (time_index as f64 + 1.0);
            let beta = -1.0 / (1.0 - alpha) + x * (1.0 / alpha + 1.0 / (1.0 - alpha));
            radius = beta * wealth;
            time_index += 1;
            expected_radii.push(radius);
            let receipt = deliver_up_ocp_feedback_v1(
                &mut engine,
                &feedback(
                    round as u64,
                    &format!("s{round}"),
                    &format!("l{round}"),
                    &format!("e{round}"),
                    centre + score,
                    LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                ),
            )
            .unwrap();
            assert_eq!(receipt.disposition, UpOcpFeedbackDispositionV1::Updated);
            assert!((engine.algorithm_state.radius - expected_radii[round]).abs() < 1.0e-12);
            assert!((engine.algorithm_state.wealth - wealth).abs() < 1.0e-12);
        }
        assert_eq!(engine.algorithm_state.accepted_score_count, 4);
    }

    #[test]
    fn author_alpha_correction_and_unshifted_mode_are_explicit_and_distinct() {
        let author = engine(UpOcpMixtureV1::DirichletHalfHalf, 0.015);
        let unshifted = engine(UpOcpMixtureV1::DirichletHalfHalf, 0.0);
        assert!((effective_alpha(&author.policy).unwrap() - 0.085).abs() < 1.0e-12);
        assert!((effective_alpha(&unshifted.policy).unwrap() - 0.10).abs() < 1.0e-12);
        assert_ne!(
            author.algorithm_state.betting_fraction,
            unshifted.algorithm_state.betting_fraction
        );
    }

    #[test]
    fn effective_alpha_outside_open_unit_interval_is_rejected() {
        let mut too_small = policy(UpOcpMixtureV1::DirichletHalfHalf, 0.015);
        too_small.target_alpha = 0.01;
        assert!(create_up_ocp_engine_v1("bad", too_small).is_err());
        let mut negative = policy(UpOcpMixtureV1::DirichletHalfHalf, 0.0);
        negative.target_alpha = -0.1;
        assert!(create_up_ocp_engine_v1("bad", negative).is_err());
    }

    #[test]
    fn prediction_is_read_only_and_freezes_the_threshold_interval() {
        let mut engine = engine(UpOcpMixtureV1::DirichletHalfHalf, 0.015);
        let before = algorithm_state_digest(&engine.algorithm_state).unwrap();
        let receipt = issue_up_ocp_prediction_v1(
            &mut engine,
            &prediction(0, "s0", "e0", 10.0),
        )
        .unwrap();
        assert_eq!(receipt.radius_used, 0.0);
        assert_eq!(receipt.interval_lower, 10.0);
        assert_eq!(receipt.interval_upper, 10.0);
        assert_eq!(receipt.algorithm_state_before_sha256, before);
        assert_eq!(receipt.algorithm_state_after_sha256, before);
        assert_eq!(algorithm_state_digest(&engine.algorithm_state).unwrap(), before);
    }

    #[test]
    fn unknown_feedback_is_held_and_can_be_represented_with_current_k1_receipt() {
        let mut engine = engine(UpOcpMixtureV1::DirichletHalfHalf, 0.015);
        issue_up_ocp_prediction_v1(&mut engine, &prediction(0, "s0", "e0", 10.0)).unwrap();
        let before = algorithm_state_digest(&engine.algorithm_state).unwrap();
        let held = deliver_up_ocp_feedback_v1(
            &mut engine,
            &feedback(0, "s0", "l0", "e0", 10.5, LabelConsumptionEligibilityV1::Unknown),
        ).unwrap();
        assert_eq!(held.disposition, UpOcpFeedbackDispositionV1::HeldUnknown);
        assert_eq!(algorithm_state_digest(&engine.algorithm_state).unwrap(), before);
        assert!(engine.pending_prediction.is_some());

        let accepted = deliver_up_ocp_feedback_v1(
            &mut engine,
            &feedback(0, "s0", "l0", "e0", 10.5, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy),
        ).unwrap();
        assert_eq!(accepted.disposition, UpOcpFeedbackDispositionV1::Updated);
        assert_eq!(engine.algorithm_state.accepted_score_count, 1);
    }

    #[test]
    fn blocked_feedback_does_not_update_or_advance_the_stream() {
        let mut engine = engine(UpOcpMixtureV1::DirichletHalfHalf, 0.015);
        issue_up_ocp_prediction_v1(&mut engine, &prediction(0, "s0", "e0", 10.0)).unwrap();
        let before = algorithm_state_digest(&engine.algorithm_state).unwrap();
        let receipt = deliver_up_ocp_feedback_v1(
            &mut engine,
            &feedback(0, "s0", "l0", "e0", 10.5, LabelConsumptionEligibilityV1::Blocked),
        ).unwrap();
        assert_eq!(receipt.disposition, UpOcpFeedbackDispositionV1::Blocked);
        assert_eq!(algorithm_state_digest(&engine.algorithm_state).unwrap(), before);
        assert!(engine.pending_prediction.is_some());
        assert!(issue_up_ocp_prediction_v1(&mut engine, &prediction(1, "s1", "e1", 10.0)).is_err());
    }

    #[test]
    fn stale_or_revision_mismatched_report_is_rejected_without_state_change() {
        let mut engine = engine(UpOcpMixtureV1::DirichletHalfHalf, 0.015);
        issue_up_ocp_prediction_v1(&mut engine, &prediction(0, "s0", "e0", 10.0)).unwrap();
        let before = algorithm_state_digest(&engine.algorithm_state).unwrap();
        let mut stale = feedback(0, "s0", "l0", "e0", 10.5, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy);
        stale.eligibility_report.frontier = LabelEvidenceFrontierV1::AsKnownAtK0;
        stale.eligibility_report.report_sha256 = eligibility_report_digest(&stale.eligibility_report).unwrap();
        let receipt = deliver_up_ocp_feedback_v1(&mut engine, &stale).unwrap();
        assert_eq!(receipt.disposition, UpOcpFeedbackDispositionV1::Rejected);
        assert_eq!(algorithm_state_digest(&engine.algorithm_state).unwrap(), before);

        let mut wrong_revision = feedback(0, "s0", "l0", "e0", 10.5, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy);
        wrong_revision.label_revision = 2;
        let receipt = deliver_up_ocp_feedback_v1(&mut engine, &wrong_revision).unwrap();
        assert_eq!(receipt.disposition, UpOcpFeedbackDispositionV1::Rejected);
        assert_eq!(algorithm_state_digest(&engine.algorithm_state).unwrap(), before);
    }

    #[test]
    fn duplicate_accepted_revision_is_idempotent() {
        let mut engine = engine(UpOcpMixtureV1::DirichletHalfHalf, 0.015);
        issue_up_ocp_prediction_v1(&mut engine, &prediction(0, "s0", "e0", 10.0)).unwrap();
        let feedback_value = feedback(0, "s0", "l0", "e0", 10.5, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy);
        assert_eq!(
            deliver_up_ocp_feedback_v1(&mut engine, &feedback_value).unwrap().disposition,
            UpOcpFeedbackDispositionV1::Updated
        );
        let before = algorithm_state_digest(&engine.algorithm_state).unwrap();
        let duplicate = deliver_up_ocp_feedback_v1(&mut engine, &feedback_value).unwrap();
        assert_eq!(duplicate.disposition, UpOcpFeedbackDispositionV1::Duplicate);
        assert_eq!(algorithm_state_digest(&engine.algorithm_state).unwrap(), before);
    }

    #[test]
    fn explicit_forfeit_does_not_mutate_betting_state_but_invalidates_theorem_claim() {
        let mut engine = engine(UpOcpMixtureV1::DirichletHalfHalf, 0.015);
        issue_up_ocp_prediction_v1(&mut engine, &prediction(0, "s0", "e0", 10.0)).unwrap();
        let before = algorithm_state_digest(&engine.algorithm_state).unwrap();
        forfeit_pending_up_ocp_feedback_v1(&mut engine, "physical label was not returned").unwrap();
        assert_eq!(algorithm_state_digest(&engine.algorithm_state).unwrap(), before);
        assert_eq!(
            engine.guarantee_state,
            UpOcpGuaranteeStateV1::InvalidatedByForfeit
        );
        assert!(engine.pending_prediction.is_none());
        issue_up_ocp_prediction_v1(&mut engine, &prediction(1, "s1", "e1", 10.0)).unwrap();
    }

    #[test]
    fn same_immediate_score_stream_replays_byte_identically() {
        fn run() -> String {
            let mut engine = engine(UpOcpMixtureV1::DirichletHalfHalf, 0.015);
            for (round, score) in [0.5, 0.2, 0.7].into_iter().enumerate() {
                issue_up_ocp_prediction_v1(
                    &mut engine,
                    &prediction(round as u64, &format!("s{round}"), &format!("e{round}"), 10.0),
                )
                .unwrap();
                deliver_up_ocp_feedback_v1(
                    &mut engine,
                    &feedback(
                        round as u64,
                        &format!("s{round}"),
                        &format!("l{round}"),
                        &format!("e{round}"),
                        10.0 + score,
                        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                    ),
                )
                .unwrap();
            }
            algorithm_state_digest(&engine.algorithm_state).unwrap()
        }
        assert_eq!(run(), run());
    }
}
