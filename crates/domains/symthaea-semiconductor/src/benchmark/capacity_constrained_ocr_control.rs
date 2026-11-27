// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Faithful arbitrary-bounded-delay CA-OCRC reference engine.
//!
//! This module implements the order and equations in Wang & Simeone,
//! arXiv:2611.17059, Sec. IV-V. Proxy-delay samples are explicit inputs so
//! simulations are replayable and their random source/seed can be archived.
//! The theorem is reported as ineligible when an admitted/retained feedback
//! item fails K1 measurement qualification; the original theorem does not
//! automatically cover that additional missingness process.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};

pub const CAPACITY_OCRC_POLICY_SCHEMA_V1: &str = "luminous.semi-bench.capacity-ocrc-policy.v1";
pub const CAPACITY_OCRC_ENGINE_SCHEMA_V1: &str = "luminous.semi-bench.capacity-ocrc-engine.v1";
pub const CAPACITY_OCRC_PREDICTION_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.capacity-ocrc-prediction-input.v1";
pub const CAPACITY_OCRC_PREDICTION_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.capacity-ocrc-prediction-receipt.v1";
pub const CAPACITY_OCRC_ROUND_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.capacity-ocrc-round-input.v1";
pub const CAPACITY_OCRC_FEEDBACK_SCHEMA_V1: &str = "luminous.semi-bench.capacity-ocrc-feedback.v1";
pub const CAPACITY_OCRC_ROUND_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.capacity-ocrc-round-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrcIntervalKindV1 {
    Empty,
    Bounded,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrcTheoremEligibilityV1 {
    EligibleUnderDeclaredAssumptions,
    AssumptionsNotMet,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrcFeedbackDispositionV1 {
    Applied,
    NotTracked,
    RejectedK1,
    RejectedIdentity,
    RejectedTiming,
    RejectedDigest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrcFeedbackReasonV1 {
    LabelNotCurrentK1,
    LabelNotEligible,
    DownstreamConsumptionNotAllowed,
    LabelIdentityMismatch,
    LabelRevisionMismatch,
    MeasurementEventMismatch,
    LabelPayloadDigestMismatch,
    EligibilityReportDigestMismatch,
    CorrectedValueMismatch,
    LabelUnavailableAtDelivery,
    ReportUnavailableAtDelivery,
    ReportPredatesLabelAvailability,
    DelayOutOfRange,
    DuplicateFeedback,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapacityOcrcPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub target_miscoverage: f64,
    pub learning_rate: f64,
    pub score_bound: f64,
    pub capacity: usize,
    pub maximum_feedback_delay: u64,
    /// F(a) = P(proxy_delay >= a), indexed by a=0..=maximum_feedback_delay.
    pub proxy_delay_survival: Vec<f64>,
    pub initial_threshold: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapacityOcrcPredictionInputV1 {
    pub schema: String,
    pub round: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub feature_available_at_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub global_model_artifact_sha256: String,
    pub feature_schema_sha256: String,
    pub scale_artifact_sha256: String,
    pub prediction_value: f64,
    /// Positive scale used to normalize the residual score.
    pub residual_scale: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapacityOcrcPredictionReceiptV1 {
    pub schema: String,
    pub round: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub global_model_artifact_sha256: String,
    pub feature_schema_sha256: String,
    pub scale_artifact_sha256: String,
    pub prediction_value: f64,
    pub residual_scale: f64,
    pub threshold_before: f64,
    pub interval_kind: OcrcIntervalKindV1,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub score_bound: f64,
    pub proxy_delay_sample: i64,
    pub tracking_admitted: bool,
    pub tracking_occupancy_after_admission: usize,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapacityOcrcQualifiedFeedbackV1 {
    pub schema: String,
    pub source_round: u64,
    pub delivered_round: u64,
    pub actual_delay: u64,
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub measurement_event_id: String,
    pub label_payload_sha256: String,
    pub label_value: f64,
    pub label_available_at_unix_seconds: i64,
    pub eligibility_report_available_at_unix_seconds: i64,
    pub delivered_at_unix_seconds: i64,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapacityOcrcTrackedRoundV1 {
    pub prediction: CapacityOcrcPredictionReceiptV1,
    pub proxy_deadline_round: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapacityOcrcFeedbackUpdateV1 {
    pub source_round: u64,
    pub label_id: String,
    pub actual_delay: u64,
    pub observed_miscoverage: f64,
    pub survival_probability: f64,
    pub inverse_probability_weight: f64,
    pub threshold_delta: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapacityOcrcFeedbackOutcomeV1 {
    pub source_round: u64,
    pub label_id: String,
    pub disposition: OcrcFeedbackDispositionV1,
    pub reasons: Vec<OcrcFeedbackReasonV1>,
    pub miscoverage: Option<f64>,
    pub threshold_delta: f64,
    pub feedback_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapacityOcrcEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub policy: CapacityOcrcPolicyV1,
    pub threshold: f64,
    pub current_round: u64,
    pub tracked_rounds: BTreeMap<u64, CapacityOcrcTrackedRoundV1>,
    pub processed_feedback_rounds: BTreeSet<u64>,
    pub seen_sample_ids: BTreeSet<String>,
    pub seen_measurement_event_ids: BTreeSet<String>,
    pub theorem_eligibility: OcrcTheoremEligibilityV1,
    pub theorem_reason_codes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapacityOcrcRoundInputV1 {
    pub schema: String,
    pub prediction: CapacityOcrcPredictionInputV1,
    /// Draw from the policy's proxy-delay distribution. -1 means not tracked.
    /// For trivial empty/full sets this must be -1.
    pub sampled_proxy_delay: i64,
    /// Feedback delivered at the end of this round. Every item must have
    /// delivered_round equal to prediction.round.
    pub qualified_feedback: Vec<CapacityOcrcQualifiedFeedbackV1>,
    /// Provenance for the exogenous proxy-delay sample source/seed.
    pub proxy_randomness_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapacityOcrcRoundReceiptV1 {
    pub schema: String,
    pub engine_id: String,
    pub round: u64,
    pub prediction: CapacityOcrcPredictionReceiptV1,
    pub evicted_rounds: Vec<u64>,
    pub feedback_outcomes: Vec<CapacityOcrcFeedbackOutcomeV1>,
    pub threshold_before: f64,
    pub threshold_after: f64,
    pub trivial_loss: Option<f64>,
    pub feedback_threshold_delta: f64,
    pub capacity: usize,
    pub tracking_occupancy_after_round: usize,
    pub tracking_load: f64,
    pub tracking_load_within_capacity: bool,
    pub theorem_eligibility: OcrcTheoremEligibilityV1,
    pub theorem_reason_codes: Vec<String>,
    pub proxy_randomness_receipt_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapacityOcrcRiskBoundV1 {
    pub horizon: u64,
    pub score_bound: f64,
    pub learning_rate: f64,
    pub capacity: usize,
    pub maximum_feedback_delay: u64,
    pub terminal_survival_probability: f64,
    pub tracking_load: f64,
    pub saturation_term: f64,
    pub risk_gap_upper_bound: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapacityOcrcOptimizedProxyV1 {
    pub capacity: usize,
    pub maximum_feedback_delay: u64,
    pub horizon: u64,
    pub bernoulli_survival_probability: f64,
    pub proxy_delay_survival: Vec<f64>,
    pub objective_value: f64,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct CapacityOcrcErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), CapacityOcrcErrorV1> {
    if value.trim().is_empty() {
        return Err(CapacityOcrcErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, CapacityOcrcErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        CapacityOcrcErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn eligibility_report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, CapacityOcrcErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn prediction_receipt_digest(
    receipt: &CapacityOcrcPredictionReceiptV1,
) -> Result<String, CapacityOcrcErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn feedback_receipt_digest(
    feedback: &CapacityOcrcQualifiedFeedbackV1,
) -> Result<String, CapacityOcrcErrorV1> {
    json_sha256(feedback)
}

fn round_receipt_digest(
    receipt: &CapacityOcrcRoundReceiptV1,
) -> Result<String, CapacityOcrcErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

pub fn validate_capacity_ocrc_policy_v1(
    policy: &CapacityOcrcPolicyV1,
) -> Result<(), CapacityOcrcErrorV1> {
    if policy.schema != CAPACITY_OCRC_POLICY_SCHEMA_V1 {
        return Err(CapacityOcrcErrorV1(
            "CA-OCRC policy schema is unsupported".into(),
        ));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if policy.revision == 0 {
        return Err(CapacityOcrcErrorV1(
            "policy revision must be positive".into(),
        ));
    }
    if !policy.target_miscoverage.is_finite()
        || !(0.0..1.0).contains(&policy.target_miscoverage)
        || !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || !policy.score_bound.is_finite()
        || policy.score_bound <= 0.0
        || policy.capacity == 0
        || !policy.initial_threshold.is_finite()
        || !(0.0..=policy.score_bound).contains(&policy.initial_threshold)
    {
        return Err(CapacityOcrcErrorV1(
            "CA-OCRC policy contains invalid risk, rate, bound, capacity or threshold".into(),
        ));
    }
    let expected_len = usize::try_from(policy.maximum_feedback_delay)
        .map_err(|_| CapacityOcrcErrorV1("maximum delay does not fit usize".into()))?
        + 1;
    if policy.proxy_delay_survival.len() != expected_len {
        return Err(CapacityOcrcErrorV1(
            "proxy survival vector length must equal maximum delay plus one".into(),
        ));
    }
    let mut previous = 1.0_f64;
    for value in &policy.proxy_delay_survival {
        if !value.is_finite() || !(0.0..=1.0).contains(value) || *value > previous {
            return Err(CapacityOcrcErrorV1(
                "proxy survival probabilities must be finite, in [0,1], and non-increasing".into(),
            ));
        }
        previous = *value;
    }
    let terminal = *policy.proxy_delay_survival.last().unwrap_or(&0.0);
    if terminal <= 0.0 {
        return Err(CapacityOcrcErrorV1(
            "F(d_max) must be positive for the declared CA-OCRC bound".into(),
        ));
    }
    let load: f64 = policy.proxy_delay_survival.iter().skip(1).sum();
    if !load.is_finite() || load > policy.capacity as f64 {
        return Err(CapacityOcrcErrorV1(
            "proxy tracking load must not exceed tracking capacity".into(),
        ));
    }
    Ok(())
}

pub fn capacity_ocrc_tracking_load_v1(policy: &CapacityOcrcPolicyV1) -> f64 {
    policy.proxy_delay_survival.iter().skip(1).sum()
}

pub fn create_capacity_ocrc_engine_v1(
    engine_id: &str,
    policy: CapacityOcrcPolicyV1,
) -> Result<CapacityOcrcEngineV1, CapacityOcrcErrorV1> {
    require_text(engine_id, "engine_id")?;
    validate_capacity_ocrc_policy_v1(&policy)?;
    Ok(CapacityOcrcEngineV1 {
        schema: CAPACITY_OCRC_ENGINE_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        threshold: policy.initial_threshold,
        policy,
        current_round: 0,
        tracked_rounds: BTreeMap::new(),
        processed_feedback_rounds: BTreeSet::new(),
        seen_sample_ids: BTreeSet::new(),
        seen_measurement_event_ids: BTreeSet::new(),
        theorem_eligibility: OcrcTheoremEligibilityV1::EligibleUnderDeclaredAssumptions,
        theorem_reason_codes: Vec::new(),
    })
}

fn interval_for_threshold(
    prediction: &CapacityOcrcPredictionInputV1,
    threshold: f64,
    score_bound: f64,
) -> (OcrcIntervalKindV1, Option<f64>, Option<f64>) {
    if threshold < 0.0 {
        return (OcrcIntervalKindV1::Empty, None, None);
    }
    if threshold >= score_bound {
        return (OcrcIntervalKindV1::Full, None, None);
    }
    let radius = threshold * prediction.residual_scale;
    (
        OcrcIntervalKindV1::Bounded,
        Some(prediction.prediction_value - radius),
        Some(prediction.prediction_value + radius),
    )
}

fn proxy_sample_probability(policy: &CapacityOcrcPolicyV1, sample: i64) -> f64 {
    if sample == -1 {
        return 1.0 - policy.proxy_delay_survival[0];
    }
    if sample < -1 || sample as u64 > policy.maximum_feedback_delay {
        return 0.0;
    }
    let index = sample as usize;
    let next = policy
        .proxy_delay_survival
        .get(index + 1)
        .copied()
        .unwrap_or(0.0);
    policy.proxy_delay_survival[index] - next
}

fn validate_prediction_input(
    engine: &CapacityOcrcEngineV1,
    input: &CapacityOcrcPredictionInputV1,
) -> Result<(), CapacityOcrcErrorV1> {
    if input.schema != CAPACITY_OCRC_PREDICTION_INPUT_SCHEMA_V1 {
        return Err(CapacityOcrcErrorV1(
            "prediction input schema is unsupported".into(),
        ));
    }
    require_text(&input.sample_id, "prediction.sample_id")?;
    require_text(
        &input.measurement_event_id,
        "prediction.measurement_event_id",
    )?;
    if !valid_sha256(&input.global_model_artifact_sha256)
        || !valid_sha256(&input.feature_schema_sha256)
        || !valid_sha256(&input.scale_artifact_sha256)
    {
        return Err(CapacityOcrcErrorV1(
            "model, feature-schema or scale digest is malformed".into(),
        ));
    }
    if input.event_time_unix_seconds > input.feature_available_at_unix_seconds
        || input.feature_available_at_unix_seconds > input.prediction_at_unix_seconds
        || !input.prediction_value.is_finite()
        || !input.residual_scale.is_finite()
        || input.residual_scale <= 0.0
    {
        return Err(CapacityOcrcErrorV1(
            "prediction chronology or numeric inputs are invalid".into(),
        ));
    }
    let expected_round = engine
        .current_round
        .checked_add(1)
        .ok_or_else(|| CapacityOcrcErrorV1("round sequence overflow".into()))?;
    if input.round != expected_round {
        return Err(CapacityOcrcErrorV1(
            "CA-OCRC rounds must be consecutive and start at one".into(),
        ));
    }
    Ok(())
}

fn validate_feedback(
    feedback: &CapacityOcrcQualifiedFeedbackV1,
    tracked: &CapacityOcrcTrackedRoundV1,
    current_round: u64,
) -> Result<Vec<OcrcFeedbackReasonV1>, CapacityOcrcErrorV1> {
    let report = &feedback.eligibility_report;
    let mut reasons = Vec::new();
    if feedback.schema != CAPACITY_OCRC_FEEDBACK_SCHEMA_V1 {
        return Err(CapacityOcrcErrorV1(
            "qualified feedback schema is unsupported".into(),
        ));
    }
    if feedback.delivered_round != current_round
        || feedback.source_round != tracked.prediction.round
        || feedback.actual_delay != current_round.saturating_sub(feedback.source_round)
    {
        reasons.push(OcrcFeedbackReasonV1::DelayOutOfRange);
    }
    if feedback.sample_id != tracked.prediction.sample_id
        || feedback.measurement_event_id != tracked.prediction.measurement_event_id
        || report.measurement_event_id != feedback.measurement_event_id
    {
        reasons.push(OcrcFeedbackReasonV1::MeasurementEventMismatch);
    }
    if feedback.label_revision == 0 {
        reasons.push(OcrcFeedbackReasonV1::LabelRevisionMismatch);
    }
    if report.schema != "luminous.semi-bench.label-consumption-report.v1" {
        reasons.push(OcrcFeedbackReasonV1::EligibilityReportDigestMismatch);
    }
    if report.label_id != feedback.label_id {
        reasons.push(OcrcFeedbackReasonV1::LabelIdentityMismatch);
    }
    if report.label_revision != feedback.label_revision {
        reasons.push(OcrcFeedbackReasonV1::LabelRevisionMismatch);
    }
    if !valid_sha256(&feedback.label_payload_sha256)
        || report.label_sha256 != feedback.label_payload_sha256
    {
        reasons.push(OcrcFeedbackReasonV1::LabelPayloadDigestMismatch);
    }
    if eligibility_report_digest(report)? != report.report_sha256 {
        reasons.push(OcrcFeedbackReasonV1::EligibilityReportDigestMismatch);
    }
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        reasons.push(OcrcFeedbackReasonV1::LabelNotCurrentK1);
    }
    if report.eligibility != LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy {
        reasons.push(OcrcFeedbackReasonV1::LabelNotEligible);
    }
    if !report.downstream_consumption_allowed {
        reasons.push(OcrcFeedbackReasonV1::DownstreamConsumptionNotAllowed);
    }
    if feedback.label_value != report.qualified_corrected_value || !feedback.label_value.is_finite()
    {
        reasons.push(OcrcFeedbackReasonV1::CorrectedValueMismatch);
    }
    if feedback.label_available_at_unix_seconds > feedback.delivered_at_unix_seconds
        || feedback.label_available_at_unix_seconds < tracked.prediction.event_time_unix_seconds
    {
        reasons.push(OcrcFeedbackReasonV1::LabelUnavailableAtDelivery);
    }
    if feedback.delivered_at_unix_seconds < tracked.prediction.prediction_at_unix_seconds {
        reasons.push(OcrcFeedbackReasonV1::LabelUnavailableAtDelivery);
    }
    if feedback.eligibility_report_available_at_unix_seconds > feedback.delivered_at_unix_seconds {
        reasons.push(OcrcFeedbackReasonV1::ReportUnavailableAtDelivery);
    }
    if feedback.eligibility_report_available_at_unix_seconds
        < feedback.label_available_at_unix_seconds
    {
        reasons.push(OcrcFeedbackReasonV1::ReportPredatesLabelAvailability);
    }
    reasons.sort();
    reasons.dedup();
    Ok(reasons)
}

fn mark_theorem_not_met(engine: &mut CapacityOcrcEngineV1, reason: &str) {
    engine.theorem_eligibility = OcrcTheoremEligibilityV1::AssumptionsNotMet;
    if !engine
        .theorem_reason_codes
        .iter()
        .any(|item| item == reason)
    {
        engine.theorem_reason_codes.push(reason.to_string());
        engine.theorem_reason_codes.sort();
    }
}

pub fn advance_capacity_ocrc_round_v1(
    engine: &mut CapacityOcrcEngineV1,
    input: &CapacityOcrcRoundInputV1,
) -> Result<CapacityOcrcRoundReceiptV1, CapacityOcrcErrorV1> {
    if engine.schema != CAPACITY_OCRC_ENGINE_SCHEMA_V1 {
        return Err(CapacityOcrcErrorV1(
            "CA-OCRC engine schema is unsupported".into(),
        ));
    }
    if input.schema != CAPACITY_OCRC_ROUND_INPUT_SCHEMA_V1 {
        return Err(CapacityOcrcErrorV1(
            "CA-OCRC round schema is unsupported".into(),
        ));
    }
    require_text(
        &input.proxy_randomness_receipt_sha256,
        "proxy randomness receipt",
    )?;
    if !valid_sha256(&input.proxy_randomness_receipt_sha256) {
        return Err(CapacityOcrcErrorV1(
            "proxy randomness receipt must be a lowercase SHA-256 digest".into(),
        ));
    }
    validate_capacity_ocrc_policy_v1(&engine.policy)?;
    validate_prediction_input(engine, &input.prediction)?;
    if engine.seen_sample_ids.contains(&input.prediction.sample_id)
        || engine
            .seen_measurement_event_ids
            .contains(&input.prediction.measurement_event_id)
    {
        return Err(CapacityOcrcErrorV1(
            "duplicate prediction sample or measurement event".into(),
        ));
    }

    let prediction = &input.prediction;
    let threshold_before = engine.threshold;
    let (interval_kind, lower_bound, upper_bound) =
        interval_for_threshold(prediction, threshold_before, engine.policy.score_bound);
    let trivial_loss = match interval_kind {
        OcrcIntervalKindV1::Empty => Some(1.0),
        OcrcIntervalKindV1::Full => Some(0.0),
        OcrcIntervalKindV1::Bounded => None,
    };
    if trivial_loss.is_some() && input.sampled_proxy_delay != -1 {
        return Err(CapacityOcrcErrorV1(
            "trivial prediction sets must use proxy delay -1 and not consume tracking capacity"
                .into(),
        ));
    }
    if trivial_loss.is_none()
        && proxy_sample_probability(&engine.policy, input.sampled_proxy_delay) <= 0.0
    {
        return Err(CapacityOcrcErrorV1(
            "sampled proxy delay has zero probability under the declared survival distribution"
                .into(),
        ));
    }

    let mut candidate = engine.clone();
    candidate
        .seen_sample_ids
        .insert(input.prediction.sample_id.clone());
    candidate
        .seen_measurement_event_ids
        .insert(input.prediction.measurement_event_id.clone());
    let mut evicted_rounds = Vec::new();
    let expired: Vec<u64> = candidate
        .tracked_rounds
        .iter()
        .filter(|(round, tracked)| tracked.proxy_deadline_round < prediction.round)
        .map(|(round, _)| *round)
        .collect();
    for round in expired {
        candidate.tracked_rounds.remove(&round);
        evicted_rounds.push(round);
    }

    let mut prediction_receipt = CapacityOcrcPredictionReceiptV1 {
        schema: CAPACITY_OCRC_PREDICTION_RECEIPT_SCHEMA_V1.into(),
        round: prediction.round,
        sample_id: prediction.sample_id.clone(),
        measurement_event_id: prediction.measurement_event_id.clone(),
        event_time_unix_seconds: prediction.event_time_unix_seconds,
        prediction_at_unix_seconds: prediction.prediction_at_unix_seconds,
        global_model_artifact_sha256: prediction.global_model_artifact_sha256.clone(),
        feature_schema_sha256: prediction.feature_schema_sha256.clone(),
        scale_artifact_sha256: prediction.scale_artifact_sha256.clone(),
        prediction_value: prediction.prediction_value,
        residual_scale: prediction.residual_scale,
        threshold_before,
        interval_kind,
        lower_bound,
        upper_bound,
        score_bound: candidate.policy.score_bound,
        proxy_delay_sample: input.sampled_proxy_delay,
        tracking_admitted: false,
        tracking_occupancy_after_admission: candidate.tracked_rounds.len(),
        receipt_sha256: String::new(),
    };

    if input.sampled_proxy_delay >= 0 && candidate.tracked_rounds.len() < candidate.policy.capacity
    {
        let proxy_deadline_round = prediction
            .round
            .checked_add(input.sampled_proxy_delay as u64)
            .ok_or_else(|| CapacityOcrcErrorV1("proxy deadline overflow".into()))?;
        candidate.tracked_rounds.insert(
            prediction.round,
            CapacityOcrcTrackedRoundV1 {
                prediction: prediction_receipt.clone(),
                proxy_deadline_round,
            },
        );
        prediction_receipt.tracking_admitted = true;
        candidate
            .tracked_rounds
            .get_mut(&prediction.round)
            .expect("just inserted")
            .prediction = prediction_receipt.clone();
    }
    prediction_receipt.tracking_occupancy_after_admission = candidate.tracked_rounds.len();
    prediction_receipt.receipt_sha256 = prediction_receipt_digest(&prediction_receipt)?;
    if prediction_receipt.tracking_admitted {
        candidate
            .tracked_rounds
            .get_mut(&prediction.round)
            .expect("admitted round is present")
            .prediction = prediction_receipt.clone();
    }

    let mut feedback = input.qualified_feedback.clone();
    feedback.sort_by(|left, right| {
        left.source_round
            .cmp(&right.source_round)
            .then_with(|| left.label_id.cmp(&right.label_id))
    });
    let mut feedback_outcomes = Vec::new();
    let mut feedback_threshold_delta = 0.0_f64;

    for event in feedback {
        if event.delivered_round != prediction.round
            || event.source_round == 0
            || event.source_round > prediction.round
        {
            return Err(CapacityOcrcErrorV1(
                "feedback delivery round/source round is invalid".into(),
            ));
        }
        if !candidate
            .processed_feedback_rounds
            .insert(event.source_round)
        {
            return Err(CapacityOcrcErrorV1(
                "duplicate feedback was presented for a prediction round".into(),
            ));
        }
        let Some(tracked) = candidate.tracked_rounds.get(&event.source_round).cloned() else {
            let outcome = CapacityOcrcFeedbackOutcomeV1 {
                source_round: event.source_round,
                label_id: event.label_id.clone(),
                disposition: OcrcFeedbackDispositionV1::NotTracked,
                reasons: Vec::new(),
                miscoverage: None,
                threshold_delta: 0.0,
                feedback_receipt_sha256: feedback_receipt_digest(&event)?,
            };
            feedback_outcomes.push(outcome);
            continue;
        };

        let reasons = validate_feedback(&event, &tracked, prediction.round)?;
        if !reasons.is_empty() {
            candidate.tracked_rounds.remove(&event.source_round);
            mark_theorem_not_met(&mut candidate, "qualified_feedback_assumptions_not_met");
            let disposition = if reasons.iter().any(|reason| {
                matches!(
                    reason,
                    OcrcFeedbackReasonV1::LabelNotCurrentK1
                        | OcrcFeedbackReasonV1::LabelNotEligible
                        | OcrcFeedbackReasonV1::DownstreamConsumptionNotAllowed
                )
            }) {
                OcrcFeedbackDispositionV1::RejectedK1
            } else if reasons.iter().any(|reason| {
                matches!(
                    reason,
                    OcrcFeedbackReasonV1::LabelUnavailableAtDelivery
                        | OcrcFeedbackReasonV1::ReportUnavailableAtDelivery
                        | OcrcFeedbackReasonV1::ReportPredatesLabelAvailability
                        | OcrcFeedbackReasonV1::DelayOutOfRange
                )
            }) {
                OcrcFeedbackDispositionV1::RejectedTiming
            } else if reasons.iter().any(|reason| {
                matches!(
                    reason,
                    OcrcFeedbackReasonV1::LabelPayloadDigestMismatch
                        | OcrcFeedbackReasonV1::EligibilityReportDigestMismatch
                )
            }) {
                OcrcFeedbackDispositionV1::RejectedDigest
            } else {
                OcrcFeedbackDispositionV1::RejectedIdentity
            };
            feedback_outcomes.push(CapacityOcrcFeedbackOutcomeV1 {
                source_round: event.source_round,
                label_id: event.label_id.clone(),
                disposition,
                reasons,
                miscoverage: None,
                threshold_delta: 0.0,
                feedback_receipt_sha256: feedback_receipt_digest(&event)?,
            });
            continue;
        }

        if event.actual_delay > candidate.policy.maximum_feedback_delay {
            candidate.tracked_rounds.remove(&event.source_round);
            mark_theorem_not_met(&mut candidate, "feedback_delay_exceeded_declared_bound");
            feedback_outcomes.push(CapacityOcrcFeedbackOutcomeV1 {
                source_round: event.source_round,
                label_id: event.label_id.clone(),
                disposition: OcrcFeedbackDispositionV1::RejectedTiming,
                reasons: vec![OcrcFeedbackReasonV1::DelayOutOfRange],
                miscoverage: None,
                threshold_delta: 0.0,
                feedback_receipt_sha256: feedback_receipt_digest(&event)?,
            });
            continue;
        }
        let survival = candidate.policy.proxy_delay_survival[event.actual_delay as usize];
        if survival <= 0.0 {
            candidate.tracked_rounds.remove(&event.source_round);
            mark_theorem_not_met(&mut candidate, "feedback_delay_has_zero_proxy_survival");
            feedback_outcomes.push(CapacityOcrcFeedbackOutcomeV1 {
                source_round: event.source_round,
                label_id: event.label_id.clone(),
                disposition: OcrcFeedbackDispositionV1::RejectedTiming,
                reasons: vec![OcrcFeedbackReasonV1::DelayOutOfRange],
                miscoverage: None,
                threshold_delta: 0.0,
                feedback_receipt_sha256: feedback_receipt_digest(&event)?,
            });
            continue;
        }

        let miscoverage = match tracked.prediction.interval_kind {
            OcrcIntervalKindV1::Empty => 1.0,
            OcrcIntervalKindV1::Full => 0.0,
            OcrcIntervalKindV1::Bounded => {
                let lower = tracked
                    .prediction
                    .lower_bound
                    .expect("bounded interval has a lower bound");
                let upper = tracked
                    .prediction
                    .upper_bound
                    .expect("bounded interval has an upper bound");
                if event.label_value < lower || event.label_value > upper {
                    1.0
                } else {
                    0.0
                }
            }
        };
        let weight = 1.0 / survival;
        let threshold_delta = candidate.policy.learning_rate
            * (miscoverage - candidate.policy.target_miscoverage)
            * weight;
        if !threshold_delta.is_finite() {
            return Err(CapacityOcrcErrorV1(
                "CA-OCRC threshold update is non-finite".into(),
            ));
        }
        feedback_threshold_delta += threshold_delta;
        candidate.tracked_rounds.remove(&event.source_round);
        feedback_outcomes.push(CapacityOcrcFeedbackOutcomeV1 {
            source_round: event.source_round,
            label_id: event.label_id.clone(),
            disposition: OcrcFeedbackDispositionV1::Applied,
            reasons: Vec::new(),
            miscoverage: Some(miscoverage),
            threshold_delta,
            feedback_receipt_sha256: feedback_receipt_digest(&event)?,
        });
    }

    let trivial_delta = trivial_loss
        .map(|loss| candidate.policy.learning_rate * (loss - candidate.policy.target_miscoverage))
        .unwrap_or(0.0);
    let threshold_after = candidate.threshold + trivial_delta + feedback_threshold_delta;
    if !threshold_after.is_finite() {
        return Err(CapacityOcrcErrorV1(
            "candidate threshold is non-finite".into(),
        ));
    }
    candidate.threshold = threshold_after;
    candidate.current_round = prediction.round;
    let load = capacity_ocrc_tracking_load_v1(&candidate.policy);
    let tracking_load_within_capacity = load <= candidate.policy.capacity as f64;
    if !tracking_load_within_capacity {
        mark_theorem_not_met(&mut candidate, "tracking_load_exceeds_capacity");
    }

    let mut round_receipt = CapacityOcrcRoundReceiptV1 {
        schema: CAPACITY_OCRC_ROUND_RECEIPT_SCHEMA_V1.into(),
        engine_id: candidate.engine_id.clone(),
        round: prediction.round,
        prediction: prediction_receipt,
        evicted_rounds,
        feedback_outcomes,
        threshold_before,
        threshold_after,
        trivial_loss,
        feedback_threshold_delta,
        capacity: candidate.policy.capacity,
        tracking_occupancy_after_round: candidate.tracked_rounds.len(),
        tracking_load: load,
        tracking_load_within_capacity,
        theorem_eligibility: candidate.theorem_eligibility,
        theorem_reason_codes: candidate.theorem_reason_codes.clone(),
        proxy_randomness_receipt_sha256: input.proxy_randomness_receipt_sha256.clone(),
        receipt_sha256: String::new(),
    };
    round_receipt.receipt_sha256 = round_receipt_digest(&round_receipt)?;
    *engine = candidate;
    Ok(round_receipt)
}

pub fn capacity_ocrc_risk_bound_v1(
    policy: &CapacityOcrcPolicyV1,
    horizon: u64,
) -> Result<CapacityOcrcRiskBoundV1, CapacityOcrcErrorV1> {
    validate_capacity_ocrc_policy_v1(policy)?;
    if horizon == 0 {
        return Err(CapacityOcrcErrorV1(
            "risk-bound horizon must be positive".into(),
        ));
    }
    let dmax = policy.maximum_feedback_delay as f64;
    if policy.capacity as u64 >= policy.maximum_feedback_delay.saturating_add(1) {
        return Ok(CapacityOcrcRiskBoundV1 {
            horizon,
            score_bound: policy.score_bound,
            learning_rate: policy.learning_rate,
            capacity: policy.capacity,
            maximum_feedback_delay: policy.maximum_feedback_delay,
            terminal_survival_probability: 1.0,
            tracking_load: dmax,
            saturation_term: 0.0,
            risk_gap_upper_bound: policy.score_bound / (policy.learning_rate * horizon as f64)
                + (policy.maximum_feedback_delay as f64 + 1.0) / horizon as f64,
        });
    }
    if dmax == 0.0 {
        return Ok(CapacityOcrcRiskBoundV1 {
            horizon,
            score_bound: policy.score_bound,
            learning_rate: policy.learning_rate,
            capacity: policy.capacity,
            maximum_feedback_delay: 0,
            terminal_survival_probability: 1.0,
            tracking_load: 0.0,
            saturation_term: 0.0,
            risk_gap_upper_bound: policy.score_bound / (policy.learning_rate * horizon as f64)
                + 1.0 / horizon as f64,
        });
    }
    let terminal = *policy
        .proxy_delay_survival
        .last()
        .expect("validated non-empty");
    let load = capacity_ocrc_tracking_load_v1(policy);
    let ratio = load / policy.capacity as f64;
    let h = ratio - 1.0 - ratio.ln();
    let saturation = if ratio == 0.0 {
        0.0
    } else {
        (-((policy.capacity as f64) * h)).exp()
    };
    let bound = policy.score_bound / (policy.learning_rate * horizon as f64)
        + (policy.capacity as f64 + 1.0) / (terminal * horizon as f64)
        + saturation;
    Ok(CapacityOcrcRiskBoundV1 {
        horizon,
        score_bound: policy.score_bound,
        learning_rate: policy.learning_rate,
        capacity: policy.capacity,
        maximum_feedback_delay: policy.maximum_feedback_delay,
        terminal_survival_probability: terminal,
        tracking_load: load,
        saturation_term: saturation,
        risk_gap_upper_bound: bound,
    })
}

fn proxy_objective(p: f64, capacity: usize, maximum_delay: u64, horizon: u64) -> f64 {
    let k = capacity as f64;
    let d = maximum_delay as f64;
    let ratio = d * p / k;
    if ratio <= 0.0 || ratio > 1.0 {
        return f64::INFINITY;
    }
    let h = ratio - 1.0 - ratio.ln();
    (k + 1.0) / (p * horizon as f64) + (-k * h).exp()
}

/// Deterministic grid optimizer for Proposition 1 (Eq. 29).
/// The selected objective is compared against every grid point and is
/// reproducible; callers must retain the returned distribution and receipt.
pub fn optimize_capacity_ocrc_proxy_v1(
    capacity: usize,
    maximum_feedback_delay: u64,
    horizon: u64,
    grid_points: usize,
) -> Result<CapacityOcrcOptimizedProxyV1, CapacityOcrcErrorV1> {
    if capacity == 0 || horizon == 0 || grid_points < 2 {
        return Err(CapacityOcrcErrorV1(
            "optimizer requires positive capacity/horizon and at least two grid points".into(),
        ));
    }
    let d = maximum_feedback_delay as usize;
    if maximum_feedback_delay == 0 {
        return Ok(CapacityOcrcOptimizedProxyV1 {
            capacity,
            maximum_feedback_delay,
            horizon,
            bernoulli_survival_probability: 1.0,
            proxy_delay_survival: vec![1.0],
            objective_value: (capacity as f64 + 1.0) / horizon as f64,
        });
    }
    if capacity > d {
        let survival = vec![1.0; d + 1];
        let policy_load = d as f64;
        let _ = policy_load;
        return Ok(CapacityOcrcOptimizedProxyV1 {
            capacity,
            maximum_feedback_delay,
            horizon,
            bernoulli_survival_probability: 1.0,
            proxy_delay_survival: survival,
            objective_value: proxy_objective(1.0, capacity, maximum_feedback_delay, horizon),
        });
    }
    let upper = (capacity as f64 / maximum_feedback_delay as f64).min(1.0);
    let mut best_p = 0.0_f64;
    let mut best_value = f64::INFINITY;
    for index in 1..=grid_points {
        let p = upper * index as f64 / grid_points as f64;
        let value = proxy_objective(p, capacity, maximum_feedback_delay, horizon);
        if value < best_value {
            best_value = value;
            best_p = p;
        }
    }
    if best_p <= 0.0 || !best_value.is_finite() {
        return Err(CapacityOcrcErrorV1(
            "proxy optimizer did not find a finite interior solution".into(),
        ));
    }
    let mut survival = vec![best_p; d + 1];
    survival[0] = 1.0;
    Ok(CapacityOcrcOptimizedProxyV1 {
        capacity,
        maximum_feedback_delay,
        horizon,
        bernoulli_survival_probability: best_p,
        proxy_delay_survival: survival,
        objective_value: best_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy() -> CapacityOcrcPolicyV1 {
        CapacityOcrcPolicyV1 {
            schema: CAPACITY_OCRC_POLICY_SCHEMA_V1.into(),
            policy_id: "paper-ca-ocrc-arbitrary-v1".into(),
            revision: 1,
            target_miscoverage: 0.10,
            learning_rate: 0.20,
            score_bound: 4.0,
            capacity: 2,
            maximum_feedback_delay: 3,
            proxy_delay_survival: vec![1.0, 0.75, 0.50, 0.25],
            initial_threshold: 1.0,
        }
    }

    fn prediction(round: u64) -> CapacityOcrcPredictionInputV1 {
        CapacityOcrcPredictionInputV1 {
            schema: CAPACITY_OCRC_PREDICTION_INPUT_SCHEMA_V1.into(),
            round,
            sample_id: format!("sample-{round}"),
            measurement_event_id: format!("event-{round}"),
            event_time_unix_seconds: 100 + round as i64,
            feature_available_at_unix_seconds: 101 + round as i64,
            prediction_at_unix_seconds: 102 + round as i64,
            global_model_artifact_sha256: digest('a'),
            feature_schema_sha256: digest('b'),
            scale_artifact_sha256: digest('c'),
            prediction_value: 10.0,
            residual_scale: 1.0,
        }
    }

    fn feedback(
        source_round: u64,
        delivered_round: u64,
        label_value: f64,
        report_status: LabelConsumptionEligibilityV1,
    ) -> CapacityOcrcQualifiedFeedbackV1 {
        let payload = digest('d');
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: format!("label-{source_round}"),
            label_revision: 1,
            measurement_event_id: format!("event-{source_round}"),
            calibration_certificate_id: "calibration-1".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: report_status,
            downstream_consumption_allowed: report_status
                == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: payload.clone(),
            measurement_sha256: digest('e'),
            qualified_corrected_value: label_value,
            qualified_standard_uncertainty_upper_bound: Some(0.1),
            qualification_report_sha256: digest('f'),
            requalification_report_sha256: digest('0'),
            requalification_frontier_sha256: digest('1'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        report.report_sha256 = eligibility_report_digest(&report).unwrap();
        CapacityOcrcQualifiedFeedbackV1 {
            schema: CAPACITY_OCRC_FEEDBACK_SCHEMA_V1.into(),
            source_round,
            delivered_round,
            actual_delay: delivered_round - source_round,
            sample_id: format!("sample-{source_round}"),
            label_id: format!("label-{source_round}"),
            label_revision: 1,
            measurement_event_id: format!("event-{source_round}"),
            label_payload_sha256: payload,
            label_value,
            label_available_at_unix_seconds: 110 + delivered_round as i64,
            eligibility_report_available_at_unix_seconds: 111 + delivered_round as i64,
            delivered_at_unix_seconds: 112 + delivered_round as i64,
            eligibility_report: report,
        }
    }

    fn round_input(
        round: u64,
        proxy: i64,
        feedback: Vec<CapacityOcrcQualifiedFeedbackV1>,
    ) -> CapacityOcrcRoundInputV1 {
        CapacityOcrcRoundInputV1 {
            schema: CAPACITY_OCRC_ROUND_INPUT_SCHEMA_V1.into(),
            prediction: prediction(round),
            sampled_proxy_delay: proxy,
            qualified_feedback: feedback,
            proxy_randomness_receipt_sha256: digest('9'),
        }
    }

    #[test]
    fn strict_equation_20_expiration_exposes_the_paper_figure_one_off_by_one() {
        let mut engine = create_capacity_ocrc_engine_v1("ca-ocrc-figure-1", policy()).unwrap();
        let r1 = advance_capacity_ocrc_round_v1(&mut engine, &round_input(1, 3, vec![])).unwrap();
        assert!(r1.prediction.tracking_admitted);
        assert_eq!(r1.tracking_occupancy_after_round, 1);

        let r2 = advance_capacity_ocrc_round_v1(&mut engine, &round_input(2, 3, vec![])).unwrap();
        assert!(r2.prediction.tracking_admitted);
        assert_eq!(r2.tracking_occupancy_after_round, 2);

        let r3 = advance_capacity_ocrc_round_v1(&mut engine, &round_input(3, 3, vec![])).unwrap();
        assert!(!r3.prediction.tracking_admitted);
        assert_eq!(r3.tracking_occupancy_after_round, 2);

        // Eq. (20) is strict: round 1 deadline is 1+3=4, so it is
        // not evicted at the beginning of round 4. The figure/caption says
        // it is evicted here, an off-by-one discrepancy in the paper.
        let r4 = advance_capacity_ocrc_round_v1(&mut engine, &round_input(4, 0, vec![])).unwrap();
        assert!(r4.evicted_rounds.is_empty());
        assert!(!r4.prediction.tracking_admitted);
        assert_eq!(r4.tracking_occupancy_after_round, 2);

        // If still pending, round 1 is expired at the beginning of round 5.
        let r5 = advance_capacity_ocrc_round_v1(&mut engine, &round_input(5, 0, vec![])).unwrap();
        assert_eq!(r5.evicted_rounds, vec![1]);
        assert!(r5.prediction.tracking_admitted);
        assert_eq!(r5.tracking_occupancy_after_round, 2);
    }

    #[test]
    fn zero_delay_feedback_only_arrives_when_current_round_is_admitted() {
        let mut engine = create_capacity_ocrc_engine_v1("ca-ocrc-zero", policy()).unwrap();
        let receipt = advance_capacity_ocrc_round_v1(
            &mut engine,
            &round_input(
                1,
                0,
                vec![feedback(
                    1,
                    1,
                    10.0,
                    LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                )],
            ),
        )
        .unwrap();
        assert!(receipt.prediction.tracking_admitted);
        assert_eq!(
            receipt.feedback_outcomes[0].disposition,
            OcrcFeedbackDispositionV1::Applied
        );
    }

    #[test]
    fn trivial_full_and_empty_intervals_do_not_consume_capacity() {
        let mut p = policy();
        p.initial_threshold = p.score_bound;
        let mut engine = create_capacity_ocrc_engine_v1("ca-ocrc-full", p).unwrap();
        let full =
            advance_capacity_ocrc_round_v1(&mut engine, &round_input(1, -1, vec![])).unwrap();
        assert_eq!(full.prediction.interval_kind, OcrcIntervalKindV1::Full);
        assert_eq!(full.trivial_loss, Some(0.0));
        assert_eq!(full.tracking_occupancy_after_round, 0);
        assert!(full.threshold_after < full.threshold_before);

        engine.threshold = -0.1;
        let empty =
            advance_capacity_ocrc_round_v1(&mut engine, &round_input(2, -1, vec![])).unwrap();
        assert_eq!(empty.prediction.interval_kind, OcrcIntervalKindV1::Empty);
        assert_eq!(empty.trivial_loss, Some(1.0));
        assert_eq!(empty.tracking_occupancy_after_round, 0);
    }

    #[test]
    fn eviction_uses_strict_less_than_and_keeps_feedback_at_deadline() {
        let mut engine = create_capacity_ocrc_engine_v1("ca-ocrc-deadline", policy()).unwrap();
        advance_capacity_ocrc_round_v1(&mut engine, &round_input(1, 2, vec![])).unwrap();
        advance_capacity_ocrc_round_v1(&mut engine, &round_input(2, 0, vec![])).unwrap();
        let at_deadline = advance_capacity_ocrc_round_v1(
            &mut engine,
            &round_input(
                3,
                0,
                vec![feedback(
                    1,
                    3,
                    10.0,
                    LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                )],
            ),
        )
        .unwrap();
        assert!(at_deadline.evicted_rounds.is_empty());
        assert_eq!(
            at_deadline.feedback_outcomes[0].disposition,
            OcrcFeedbackDispositionV1::Applied
        );
    }

    #[test]
    fn invalid_k1_feedback_does_not_update_threshold_and_disables_theorem_claim() {
        let mut engine = create_capacity_ocrc_engine_v1("ca-ocrc-invalid", policy()).unwrap();
        advance_capacity_ocrc_round_v1(&mut engine, &round_input(1, 2, vec![])).unwrap();
        let before = engine.threshold;
        let receipt = advance_capacity_ocrc_round_v1(
            &mut engine,
            &round_input(
                2,
                0,
                vec![feedback(1, 2, 10.0, LabelConsumptionEligibilityV1::Unknown)],
            ),
        )
        .unwrap();
        assert_eq!(engine.threshold, before);
        assert_eq!(
            receipt.feedback_outcomes[0].disposition,
            OcrcFeedbackDispositionV1::RejectedK1
        );
        assert_eq!(
            receipt.theorem_eligibility,
            OcrcTheoremEligibilityV1::AssumptionsNotMet
        );
    }

    #[test]
    fn label_revision_payload_and_value_are_bound_to_k1_report() {
        let mut engine = create_capacity_ocrc_engine_v1("ca-ocrc-bind", policy()).unwrap();
        advance_capacity_ocrc_round_v1(&mut engine, &round_input(1, 2, vec![])).unwrap();
        let mut wrong = feedback(
            1,
            2,
            10.0,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        );
        wrong.label_revision = 2;
        let receipt =
            advance_capacity_ocrc_round_v1(&mut engine, &round_input(2, 0, vec![wrong])).unwrap();
        assert_eq!(
            receipt.feedback_outcomes[0].disposition,
            OcrcFeedbackDispositionV1::RejectedIdentity
        );
        assert_eq!(
            receipt.theorem_eligibility,
            OcrcTheoremEligibilityV1::AssumptionsNotMet
        );
    }

    #[test]
    fn proxy_survival_weighting_uses_actual_delay_index() {
        let mut p = policy();
        p.proxy_delay_survival = vec![1.0, 0.75, 0.50, 0.25];
        let mut engine = create_capacity_ocrc_engine_v1("ca-ocrc-ipw", p).unwrap();
        advance_capacity_ocrc_round_v1(&mut engine, &round_input(1, 3, vec![])).unwrap();
        let before = engine.threshold;
        let receipt = advance_capacity_ocrc_round_v1(
            &mut engine,
            &round_input(
                2,
                0,
                vec![feedback(
                    1,
                    2,
                    20.0,
                    LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                )],
            ),
        )
        .unwrap();
        let loss = 1.0;
        let expected = 0.2 * (loss - 0.1) / 0.75;
        assert!((engine.threshold - (before + expected)).abs() < 1.0e-12);
        assert_eq!(
            receipt.feedback_outcomes[0].disposition,
            OcrcFeedbackDispositionV1::Applied
        );
    }

    #[test]
    fn empirical_risk_bound_matches_scalar_formula() {
        let p = policy();
        let bound = capacity_ocrc_risk_bound_v1(&p, 10_000).unwrap();
        let load: f64 = 0.75 + 0.50 + 0.25;
        let ratio: f64 = load / 2.0;
        let h = ratio - 1.0 - ratio.ln();
        let expected = 4.0 / (0.2 * 10_000.0) + 3.0 / (0.25 * 10_000.0) + (-2.0 * h).exp();
        assert!((bound.risk_gap_upper_bound - expected).abs() < 1.0e-12);
    }

    #[test]
    fn proxy_optimizer_returns_monotone_underloaded_distribution() {
        let optimized = optimize_capacity_ocrc_proxy_v1(2, 8, 10_000, 8192).unwrap();
        assert!(optimized.bernoulli_survival_probability > 0.0);
        assert!(optimized.bernoulli_survival_probability <= 0.25);
        assert_eq!(optimized.proxy_delay_survival[0], 1.0);
        assert!(
            optimized
                .proxy_delay_survival
                .windows(2)
                .all(|pair| pair[0] >= pair[1])
        );
        let load: f64 = optimized.proxy_delay_survival.iter().skip(1).sum();
        assert!(load <= 2.0 + 1.0e-12);
    }

    #[test]
    fn bernoulli_proxy_optimizer_is_no_worse_than_neighbouring_grid_points() {
        let optimized = optimize_capacity_ocrc_proxy_v1(2, 8, 10_000, 4096).unwrap();
        let p = optimized.bernoulli_survival_probability;
        let objective = optimized.objective_value;
        for neighbour in [p - 0.25 / 4096.0, p + 0.25 / 4096.0] {
            if neighbour > 0.0 && neighbour <= 0.25 {
                assert!(objective <= proxy_objective(neighbour, 2, 8, 10_000) + 1.0e-12);
            }
        }
    }

    #[test]
    fn replay_of_same_proxy_and_feedback_schedule_is_identical() {
        fn run() -> Vec<String> {
            let mut engine =
                create_capacity_ocrc_engine_v1("ca-ocrc-determinism", policy()).unwrap();
            let r1 =
                advance_capacity_ocrc_round_v1(&mut engine, &round_input(1, 2, vec![])).unwrap();
            let r2 = advance_capacity_ocrc_round_v1(
                &mut engine,
                &round_input(
                    2,
                    0,
                    vec![feedback(
                        1,
                        2,
                        10.0,
                        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                    )],
                ),
            )
            .unwrap();
            vec![r1.receipt_sha256, r2.receipt_sha256]
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn feedback_outside_proxy_deadline_is_not_tracked_or_applied() {
        let mut p = policy();
        // Proxy delay zero has positive probability; proxy delay three also
        // has positive terminal probability to retain theorem eligibility.
        p.proxy_delay_survival = vec![1.0, 0.5, 0.5, 0.5];
        let mut engine = create_capacity_ocrc_engine_v1("ca-ocrc-evicted", p).unwrap();
        advance_capacity_ocrc_round_v1(&mut engine, &round_input(1, 0, vec![])).unwrap();
        advance_capacity_ocrc_round_v1(&mut engine, &round_input(2, 0, vec![])).unwrap();
        let receipt = advance_capacity_ocrc_round_v1(
            &mut engine,
            &round_input(
                3,
                0,
                vec![feedback(
                    1,
                    3,
                    10.0,
                    LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                )],
            ),
        )
        .unwrap();
        assert_eq!(
            receipt.feedback_outcomes[0].disposition,
            OcrcFeedbackDispositionV1::NotTracked
        );
    }
}
