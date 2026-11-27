// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Delayed-feedback adaptive miscoverage controller for semiconductor VM.
//!
//! The interval builder is external: it receives the current alpha state and
//! returns the interval. This module freezes that interval, then updates ACI
//! only after the exact K1-qualified numeric label receipt becomes available.
//! The implementation is an arrival-order controller; it does not claim the
//! fixed-delay theorem for arbitrary variable or out-of-order feedback.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
    LabelEvidenceFrontierV1,
};

pub const DELAYED_ACI_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci.v1";
pub const DELAYED_ACI_POLICY_SCHEMA_V1: &str =
    "luminous.semi-bench.delayed-aci-policy.v1";
pub const DELAYED_ACI_PREDICTION_SCHEMA_V1: &str =
    "luminous.semi-bench.delayed-aci-prediction.v1";
pub const DELAYED_ACI_DELIVERY_SCHEMA_V1: &str =
    "luminous.semi-bench.delayed-aci-delivery.v1";
pub const DELAYED_ACI_PREDICTION_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.delayed-aci-prediction-receipt.v1";
pub const DELAYED_ACI_UPDATE_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.delayed-aci-update-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelayedAciDispositionV1 {
    UpdatedCovered,
    UpdatedMissed,
    HeldUnknown,
    Blocked,
    Rejected,
    Duplicate,
    PendingPredictionMissing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelayedAciReasonV1 {
    LabelNotCurrentK1,
    LabelNotEligible,
    DownstreamConsumptionNotAllowed,
    LabelIdentityMismatch,
    LabelRevisionMismatch,
    MeasurementEventMismatch,
    LabelPayloadDigestMismatch,
    EligibilityReportDigestMismatch,
    QualifiedValueMismatch,
    NumericValueNonFinite,
    ClockDomainMismatch,
    LabelUnavailableAtDelivery,
    ReportBeforeLabelAvailability,
    ReportUnavailableAtDelivery,
    DeliveryBeforePrediction,
    PredictionSchemaMismatch,
    IntervalMalformed,
    AlphaStateMismatch,
    DuplicateLabelRevision,
    DeliveryOrderRegression,
    FixedDelayScheduleMismatch,
    FeedbackPhaseOrderMismatch,
    PriorPhaseFeedbackMissing,
    PendingPredictionMissing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelayedAciFeedbackModeV1 {
    /// Exact fixed-delay recursion split into independent modulo-tau phases.
    FixedDelayPhased { delay_steps: u64 },
    /// Variable-delay adaptive heuristic; no fixed-delay theorem is claimed.
    ArrivalOrder,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    /// Target long-run miscoverage rate, e.g. 0.05 for nominal 95% coverage.
    pub target_miscoverage: f64,
    /// ACI adaptation step size gamma.
    pub learning_rate: f64,
    pub feedback_mode: DelayedAciFeedbackModeV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciPredictionInputV1 {
    pub schema: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub feature_available_at_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub calibration_model_sha256: String,
    /// Alpha state that the external interval builder used.
    pub alpha_state_used: f64,
    pub feedback_phase: Option<u64>,
    pub policy_sha256: String,
    pub point_prediction: f64,
    pub interval_lower: f64,
    pub interval_upper: f64,
    /// Optional scale used by the interval builder; retained for diagnostics.
    pub residual_scale: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciPredictionReceiptV1 {
    pub schema: String,
    pub sequence: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub feature_available_at_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub calibration_model_sha256: String,
    pub alpha_state_used: f64,
    pub feedback_phase: Option<u64>,
    pub policy_sha256: String,
    pub point_prediction: f64,
    pub interval_lower: f64,
    pub interval_upper: f64,
    pub residual_scale: Option<f64>,
    pub prediction_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciLabelDeliveryV1 {
    pub schema: String,
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub measurement_event_id: String,
    pub label_payload_sha256: String,
    pub label_value: f64,
    pub label_available_at_unix_seconds: i64,
    pub eligibility_report_available_at_unix_seconds: i64,
    pub delivered_at_unix_seconds: i64,
    /// Zero-based stream round at which this label delivery is processed.
    pub delivery_round_sequence: u64,
    pub clock_domain_id: String,
    pub idempotency_key: String,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub calibration_model_sha256: String,
    pub policy: DelayedAciPolicyV1,
    pub policy_sha256: String,
    pub prediction_sequence: u64,
    /// Used only by ArrivalOrder mode.
    pub current_alpha: f64,
    /// One ACI state per sequence phase when fixed-delay mode is active.
    pub phase_alpha_states: Vec<f64>,
    /// Next prediction sequence whose feedback is required for each phase.
    pub next_feedback_sequence_by_phase: Vec<u64>,
    pub accepted_feedback_count: u64,
    pub last_delivery_time_unix_seconds: Option<i64>,
    pub pending_predictions: BTreeMap<String, DelayedAciPredictionReceiptV1>,
    pub seen_measurement_event_ids: BTreeSet<String>,
    pub accepted_label_revisions: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciUpdateReceiptV1 {
    pub schema: String,
    pub engine_id: String,
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub measurement_event_id: String,
    pub disposition: DelayedAciDispositionV1,
    pub reasons: Vec<DelayedAciReasonV1>,
    pub prediction_receipt_sha256: Option<String>,
    pub eligibility_report_sha256: String,
    pub label_payload_sha256: String,
    pub policy_sha256: String,
    pub feedback_mode: DelayedAciFeedbackModeV1,
    pub feedback_phase: Option<u64>,
    pub delivered_at_unix_seconds: i64,
    pub delivery_round_sequence: u64,
    pub interval_lower: Option<f64>,
    pub interval_upper: Option<f64>,
    pub prediction_alpha_state_used: Option<f64>,
    pub miss_indicator: Option<bool>,
    pub target_miscoverage: f64,
    pub alpha_before: f64,
    pub alpha_after: f64,
    pub update_sequence: Option<u64>,
    pub alpha_state_mutated: bool,
    pub capacity_tracking_used: bool,
    pub source_authenticity_proven: bool,
    pub equipment_control_authority: bool,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct DelayedAciErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), DelayedAciErrorV1> {
    if value.trim().is_empty() {
        return Err(DelayedAciErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, DelayedAciErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        DelayedAciErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, DelayedAciErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn prediction_digest(
    prediction: &DelayedAciPredictionReceiptV1,
) -> Result<String, DelayedAciErrorV1> {
    let mut canonical = prediction.clone();
    canonical.prediction_receipt_sha256.clear();
    json_sha256(&canonical)
}

fn update_digest(receipt: &DelayedAciUpdateReceiptV1) -> Result<String, DelayedAciErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn validate_policy(policy: &DelayedAciPolicyV1) -> Result<(), DelayedAciErrorV1> {
    if policy.schema != DELAYED_ACI_POLICY_SCHEMA_V1 {
        return Err(DelayedAciErrorV1("delayed ACI policy schema is unsupported".into()));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    let invalid_mode = match policy.feedback_mode {
        DelayedAciFeedbackModeV1::ArrivalOrder => false,
        DelayedAciFeedbackModeV1::FixedDelayPhased { delay_steps } => {
            delay_steps == 0 || delay_steps > 4096
        }
    };
    if policy.revision == 0
        || !policy.target_miscoverage.is_finite()
        || !(0.0..1.0).contains(&policy.target_miscoverage)
        || !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || invalid_mode
    {
        return Err(DelayedAciErrorV1("delayed ACI policy parameters are invalid".into()));
    }
    Ok(())
}

fn policy_digest(policy: &DelayedAciPolicyV1) -> Result<String, DelayedAciErrorV1> {
    json_sha256(policy)
}

#[cfg(test)]
fn alpha_state_digest(engine: &DelayedAciEngineV1) -> Result<String, DelayedAciErrorV1> {
    json_sha256(&(
        engine.current_alpha,
        engine.phase_alpha_states.clone(),
        engine.next_feedback_sequence_by_phase.clone(),
        engine.accepted_feedback_count,
        engine.policy.clone(),
    ))
}

fn validate_engine(engine: &DelayedAciEngineV1) -> Result<(), DelayedAciErrorV1> {
    if engine.schema != DELAYED_ACI_SCHEMA_V1 {
        return Err(DelayedAciErrorV1("delayed ACI engine schema is unsupported".into()));
    }
    require_text(&engine.engine_id, "engine.engine_id")?;
    require_text(&engine.clock_domain_id, "engine.clock_domain_id")?;
    if !valid_sha256(&engine.global_model_artifact_sha256)
        || !valid_sha256(&engine.calibration_model_sha256)
    {
        return Err(DelayedAciErrorV1("model/calibration digest is malformed".into()));
    }
    validate_policy(&engine.policy)?;
    if !valid_sha256(&engine.policy_sha256) || policy_digest(&engine.policy)? != engine.policy_sha256 {
        return Err(DelayedAciErrorV1("ACI policy digest is missing or inconsistent".into()));
    }
    if !engine.current_alpha.is_finite() || !(0.0..=1.0).contains(&engine.current_alpha)
        || engine
            .phase_alpha_states
            .iter()
            .any(|alpha| !alpha.is_finite() || !(0.0..=1.0).contains(alpha))
    {
        return Err(DelayedAciErrorV1("current/phase alpha is outside [0,1]".into()));
    }
    match engine.policy.feedback_mode {
        DelayedAciFeedbackModeV1::ArrivalOrder => {
            if !engine.phase_alpha_states.is_empty()
                || !engine.next_feedback_sequence_by_phase.is_empty()
            {
                return Err(DelayedAciErrorV1(
                    "arrival-order mode must not carry fixed-delay phases".into(),
                ));
            }
        }
        DelayedAciFeedbackModeV1::FixedDelayPhased { delay_steps } => {
            if engine.phase_alpha_states.len() != delay_steps as usize
                || engine.next_feedback_sequence_by_phase.len() != delay_steps as usize
            {
                return Err(DelayedAciErrorV1(
                    "fixed-delay phase state dimensions do not match delay".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Creates the ACI controller. The caller constructs each interval using the
/// current alpha returned in the engine and records the interval in the next
/// prediction event.
pub fn create_delayed_aci_engine_v1(
    engine_id: &str,
    clock_domain_id: &str,
    global_model_artifact_sha256: &str,
    calibration_model_sha256: &str,
    policy: DelayedAciPolicyV1,
    initial_alpha: f64,
) -> Result<DelayedAciEngineV1, DelayedAciErrorV1> {
    require_text(engine_id, "engine_id")?;
    require_text(clock_domain_id, "clock_domain_id")?;
    if !valid_sha256(global_model_artifact_sha256) || !valid_sha256(calibration_model_sha256) {
        return Err(DelayedAciErrorV1("model/calibration digest is malformed".into()));
    }
    validate_policy(&policy)?;
    if !initial_alpha.is_finite() || !(0.0..=1.0).contains(&initial_alpha) {
        return Err(DelayedAciErrorV1("initial alpha must be in [0,1]".into()));
    }
    let policy_sha256 = policy_digest(&policy)?;
    let (phase_alpha_states, next_feedback_sequence_by_phase) = match policy.feedback_mode {
        DelayedAciFeedbackModeV1::ArrivalOrder => (Vec::new(), Vec::new()),
        DelayedAciFeedbackModeV1::FixedDelayPhased { delay_steps } => (
            vec![initial_alpha; delay_steps as usize],
            (0..delay_steps).collect(),
        ),
    };
    Ok(DelayedAciEngineV1 {
        schema: DELAYED_ACI_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        clock_domain_id: clock_domain_id.into(),
        global_model_artifact_sha256: global_model_artifact_sha256.into(),
        calibration_model_sha256: calibration_model_sha256.into(),
        policy,
        policy_sha256,
        current_alpha: initial_alpha,
        phase_alpha_states,
        next_feedback_sequence_by_phase,
        prediction_sequence: 0,
        accepted_feedback_count: 0,
        last_delivery_time_unix_seconds: None,
        pending_predictions: BTreeMap::new(),
        seen_measurement_event_ids: BTreeSet::new(),
        accepted_label_revisions: BTreeSet::new(),
    })
}

pub fn current_delayed_aci_alpha_v1(
    engine: &DelayedAciEngineV1,
) -> Result<f64, DelayedAciErrorV1> {
    validate_engine(engine)?;
    match engine.policy.feedback_mode {
        DelayedAciFeedbackModeV1::ArrivalOrder => Ok(engine.current_alpha),
        DelayedAciFeedbackModeV1::FixedDelayPhased { delay_steps } => {
            let phase = (engine.prediction_sequence % delay_steps) as usize;
            Ok(engine.phase_alpha_states[phase])
        }
    }
}

/// Freezes the interval and alpha used by the predictor. This is intentionally
/// separate from alpha updates; later labels never rewrite issued intervals.
pub fn register_delayed_aci_prediction_v1(
    engine: &mut DelayedAciEngineV1,
    input: &DelayedAciPredictionInputV1,
) -> Result<DelayedAciPredictionReceiptV1, DelayedAciErrorV1> {
    validate_engine(engine)?;
    if input.schema != DELAYED_ACI_PREDICTION_SCHEMA_V1 {
        return Err(DelayedAciErrorV1("prediction input schema is unsupported".into()));
    }
    require_text(&input.sample_id, "input.sample_id")?;
    require_text(&input.measurement_event_id, "input.measurement_event_id")?;
    if input.clock_domain_id != engine.clock_domain_id {
        return Err(DelayedAciErrorV1("prediction clock domain mismatch".into()));
    }
    if input.global_model_artifact_sha256 != engine.global_model_artifact_sha256
        || input.calibration_model_sha256 != engine.calibration_model_sha256
    {
        return Err(DelayedAciErrorV1("prediction model/calibration identity mismatch".into()));
    }
    if input.event_time_unix_seconds > input.feature_available_at_unix_seconds
        || input.feature_available_at_unix_seconds > input.prediction_at_unix_seconds
    {
        return Err(DelayedAciErrorV1("prediction chronology is invalid".into()));
    }
    let (expected_alpha, feedback_phase) = match engine.policy.feedback_mode {
        DelayedAciFeedbackModeV1::ArrivalOrder => (engine.current_alpha, None),
        DelayedAciFeedbackModeV1::FixedDelayPhased { delay_steps } => {
            let phase = engine.prediction_sequence % delay_steps;
            if engine.prediction_sequence >= delay_steps
                && engine.next_feedback_sequence_by_phase[phase as usize]
                    != engine.prediction_sequence
            {
                return Err(DelayedAciErrorV1(
                    "prior fixed-delay phase feedback has not been qualified and consumed".into(),
                ));
            }
            (engine.phase_alpha_states[phase as usize], Some(phase))
        }
    };
    if !input.alpha_state_used.is_finite()
        || input.alpha_state_used != expected_alpha
        || !input.point_prediction.is_finite()
        || !input.interval_lower.is_finite()
        || !input.interval_upper.is_finite()
        || input.interval_lower > input.interval_upper
        || input
            .residual_scale
            .is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
    {
        return Err(DelayedAciErrorV1("prediction interval/alpha is invalid".into()));
    }
    if engine.pending_predictions.contains_key(&input.sample_id)
        || engine.seen_measurement_event_ids.contains(&input.measurement_event_id)
    {
        return Err(DelayedAciErrorV1("duplicate pending sample or measurement event".into()));
    }
    let next_sequence = engine
        .prediction_sequence
        .checked_add(1)
        .ok_or_else(|| DelayedAciErrorV1("prediction sequence overflow".into()))?;
    let mut receipt = DelayedAciPredictionReceiptV1 {
        schema: DELAYED_ACI_PREDICTION_RECEIPT_SCHEMA_V1.into(),
        sequence: engine.prediction_sequence,
        sample_id: input.sample_id.clone(),
        measurement_event_id: input.measurement_event_id.clone(),
        event_time_unix_seconds: input.event_time_unix_seconds,
        feature_available_at_unix_seconds: input.feature_available_at_unix_seconds,
        prediction_at_unix_seconds: input.prediction_at_unix_seconds,
        clock_domain_id: input.clock_domain_id.clone(),
        global_model_artifact_sha256: input.global_model_artifact_sha256.clone(),
        calibration_model_sha256: input.calibration_model_sha256.clone(),
        alpha_state_used: input.alpha_state_used,
        feedback_phase,
        policy_sha256: engine.policy_sha256.clone(),
        point_prediction: input.point_prediction,
        interval_lower: input.interval_lower,
        interval_upper: input.interval_upper,
        residual_scale: input.residual_scale,
        prediction_receipt_sha256: String::new(),
    };
    receipt.prediction_receipt_sha256 = prediction_digest(&receipt)?;
    engine
        .seen_measurement_event_ids
        .insert(input.measurement_event_id.clone());
    engine
        .pending_predictions
        .insert(input.sample_id.clone(), receipt.clone());
    engine.prediction_sequence = next_sequence;
    Ok(receipt)
}

fn build_update_receipt(
    engine: &DelayedAciEngineV1,
    delivery: &DelayedAciLabelDeliveryV1,
    prediction: Option<&DelayedAciPredictionReceiptV1>,
    disposition: DelayedAciDispositionV1,
    reasons: Vec<DelayedAciReasonV1>,
    alpha_before: f64,
    alpha_after: f64,
    miss: Option<bool>,
    feedback_phase: Option<u64>,
    update_sequence: Option<u64>,
) -> Result<DelayedAciUpdateReceiptV1, DelayedAciErrorV1> {
    let mut receipt = DelayedAciUpdateReceiptV1 {
        schema: DELAYED_ACI_UPDATE_RECEIPT_SCHEMA_V1.into(),
        engine_id: engine.engine_id.clone(),
        sample_id: delivery.sample_id.clone(),
        label_id: delivery.label_id.clone(),
        label_revision: delivery.label_revision,
        measurement_event_id: delivery.measurement_event_id.clone(),
        disposition,
        reasons,
        prediction_receipt_sha256: prediction.map(|item| item.prediction_receipt_sha256.clone()),
        eligibility_report_sha256: delivery.eligibility_report.report_sha256.clone(),
        label_payload_sha256: delivery.label_payload_sha256.clone(),
        policy_sha256: engine.policy_sha256.clone(),
        feedback_mode: engine.policy.feedback_mode,
        feedback_phase,
        delivered_at_unix_seconds: delivery.delivered_at_unix_seconds,
        delivery_round_sequence: delivery.delivery_round_sequence,
        interval_lower: prediction.map(|item| item.interval_lower),
        interval_upper: prediction.map(|item| item.interval_upper),
        prediction_alpha_state_used: prediction.map(|item| item.alpha_state_used),
        miss_indicator: miss,
        target_miscoverage: engine.policy.target_miscoverage,
        alpha_before,
        alpha_after,
        update_sequence,
        alpha_state_mutated: alpha_before != alpha_after,
        capacity_tracking_used: false,
        source_authenticity_proven: false,
        equipment_control_authority: false,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = update_digest(&receipt)?;
    Ok(receipt)
}

/// Accepts one delivered physical label in monotonic arrival-time order.
/// In FixedDelayPhased mode, this implements the delayed ACI recursion
/// alpha_{t+tau} = alpha_t + gamma(target_alpha - miss_t) using tau independent
/// phase states. In ArrivalOrder mode it is a variable-delay adaptive heuristic;
/// the fixed-delay interleaving theorem is not claimed.
pub fn deliver_delayed_aci_feedback_v1(
    engine: &mut DelayedAciEngineV1,
    delivery: &DelayedAciLabelDeliveryV1,
) -> Result<DelayedAciUpdateReceiptV1, DelayedAciErrorV1> {
    validate_engine(engine)?;
    if delivery.schema != DELAYED_ACI_DELIVERY_SCHEMA_V1 {
        return Err(DelayedAciErrorV1("label delivery schema is unsupported".into()));
    }
    require_text(&delivery.sample_id, "delivery.sample_id")?;
    require_text(&delivery.label_id, "delivery.label_id")?;
    require_text(&delivery.measurement_event_id, "delivery.measurement_event_id")?;
    require_text(&delivery.clock_domain_id, "delivery.clock_domain_id")?;
    require_text(&delivery.idempotency_key, "delivery.idempotency_key")?;
    if delivery.label_revision == 0 || !valid_sha256(&delivery.label_payload_sha256) {
        return Err(DelayedAciErrorV1("label revision or payload digest is invalid".into()));
    }

    let report = &delivery.eligibility_report;
    let prediction = engine.pending_predictions.get(&delivery.sample_id);
    let alpha_before = match engine.policy.feedback_mode {
        DelayedAciFeedbackModeV1::ArrivalOrder => engine.current_alpha,
        DelayedAciFeedbackModeV1::FixedDelayPhased { .. } => prediction
            .map(|item| item.alpha_state_used)
            .unwrap_or(current_delayed_aci_alpha_v1(engine)?),
    };

    if engine
        .accepted_label_revisions
        .contains(&format!("{}:{}", delivery.label_id, delivery.label_revision))
    {
        return build_update_receipt(
            engine,
            delivery,
            prediction,
            DelayedAciDispositionV1::Duplicate,
            vec![DelayedAciReasonV1::DuplicateLabelRevision],
            alpha_before,
            alpha_before,
            None,
            prediction.and_then(|item| item.feedback_phase),
            None,
        );
    }

    let mut reasons = Vec::new();
    if delivery.clock_domain_id != engine.clock_domain_id {
        reasons.push(DelayedAciReasonV1::ClockDomainMismatch);
    }
    if !delivery.label_value.is_finite()
        || !report.qualified_corrected_value.is_finite()
        || report.qualified_standard_uncertainty_upper_bound.is_none()
        || report
            .qualified_standard_uncertainty_upper_bound
            .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        reasons.push(DelayedAciReasonV1::NumericValueNonFinite);
    }
    if !valid_sha256(&report.label_sha256)
        || report.label_sha256 != delivery.label_payload_sha256
    {
        reasons.push(DelayedAciReasonV1::LabelPayloadDigestMismatch);
    }
    if report_digest(report)? != report.report_sha256 {
        reasons.push(DelayedAciReasonV1::EligibilityReportDigestMismatch);
    }
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        reasons.push(DelayedAciReasonV1::LabelNotCurrentK1);
    }
    match report.eligibility {
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
            if report.downstream_consumption_allowed => {}
        LabelConsumptionEligibilityV1::Unknown => {
            reasons.push(DelayedAciReasonV1::LabelNotEligible);
        }
        LabelConsumptionEligibilityV1::Blocked => {
            reasons.push(DelayedAciReasonV1::LabelNotEligible);
        }
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => {
            reasons.push(DelayedAciReasonV1::DownstreamConsumptionNotAllowed);
        }
    }
    if report.training_snapshot_mutated || report.equipment_control_authority {
        reasons.push(DelayedAciReasonV1::DownstreamConsumptionNotAllowed);
    }
    if report.label_id != delivery.label_id {
        reasons.push(DelayedAciReasonV1::LabelIdentityMismatch);
    }
    if report.label_revision != delivery.label_revision {
        reasons.push(DelayedAciReasonV1::LabelRevisionMismatch);
    }
    if report.measurement_event_id != delivery.measurement_event_id {
        reasons.push(DelayedAciReasonV1::MeasurementEventMismatch);
    }
    if report.qualified_corrected_value != delivery.label_value {
        reasons.push(DelayedAciReasonV1::QualifiedValueMismatch);
    }
    if delivery.idempotency_key != format!("{}:{}", delivery.label_id, delivery.label_revision) {
        reasons.push(DelayedAciReasonV1::LabelIdentityMismatch);
    }

    if let Some(prediction) = prediction {
        if prediction_digest(prediction)? != prediction.prediction_receipt_sha256 {
            reasons.push(DelayedAciReasonV1::PredictionSchemaMismatch);
        }
        if prediction.global_model_artifact_sha256 != engine.global_model_artifact_sha256
            || prediction.calibration_model_sha256 != engine.calibration_model_sha256
        {
            reasons.push(DelayedAciReasonV1::PredictionSchemaMismatch);
        }
        if prediction.measurement_event_id != delivery.measurement_event_id {
            reasons.push(DelayedAciReasonV1::MeasurementEventMismatch);
        }
        if prediction.clock_domain_id != delivery.clock_domain_id {
            reasons.push(DelayedAciReasonV1::ClockDomainMismatch);
        }
        if delivery.label_available_at_unix_seconds < prediction.event_time_unix_seconds {
            reasons.push(DelayedAciReasonV1::LabelUnavailableAtDelivery);
        }
        if delivery.delivered_at_unix_seconds < prediction.prediction_at_unix_seconds {
            reasons.push(DelayedAciReasonV1::DeliveryBeforePrediction);
        }
    }
    if delivery.delivered_at_unix_seconds < delivery.label_available_at_unix_seconds {
        reasons.push(DelayedAciReasonV1::LabelUnavailableAtDelivery);
    }
    if delivery.eligibility_report_available_at_unix_seconds
        < delivery.label_available_at_unix_seconds
    {
        reasons.push(DelayedAciReasonV1::ReportBeforeLabelAvailability);
    }
    if delivery.delivered_at_unix_seconds
        < delivery.eligibility_report_available_at_unix_seconds
    {
        reasons.push(DelayedAciReasonV1::ReportUnavailableAtDelivery);
    }
    if engine
        .last_delivery_time_unix_seconds
        .is_some_and(|last| delivery.delivered_at_unix_seconds < last)
    {
        reasons.push(DelayedAciReasonV1::DeliveryOrderRegression);
    }
    if let Some(prediction) = prediction {
        if let DelayedAciFeedbackModeV1::FixedDelayPhased { delay_steps } =
            engine.policy.feedback_mode
        {
            let expected_delivery_round = prediction
                .sequence
                .checked_add(delay_steps)
                .ok_or_else(|| DelayedAciErrorV1("fixed-delay sequence overflow".into()))?;
            if delivery.delivery_round_sequence != expected_delivery_round {
                reasons.push(DelayedAciReasonV1::FixedDelayScheduleMismatch);
            }
            if let Some(phase) = prediction.feedback_phase {
                let phase_index = phase as usize;
                if phase != prediction.sequence % delay_steps
                    || engine.next_feedback_sequence_by_phase[phase_index] != prediction.sequence
                    || engine.phase_alpha_states[phase_index] != prediction.alpha_state_used
                {
                    reasons.push(DelayedAciReasonV1::FeedbackPhaseOrderMismatch);
                }
            }
        }
    }

    reasons.sort();
    reasons.dedup();
    let Some(prediction_receipt) = prediction.cloned() else {
        reasons.push(DelayedAciReasonV1::PendingPredictionMissing);
        reasons.sort();
        reasons.dedup();
        return build_update_receipt(
            engine,
            delivery,
            None,
            DelayedAciDispositionV1::PendingPredictionMissing,
            reasons,
            alpha_before,
            alpha_before,
            None,
            prediction.and_then(|item| item.feedback_phase),
            None,
        );
    };

    if !reasons.is_empty() {
        let only_unknown = report.eligibility == LabelConsumptionEligibilityV1::Unknown
            && reasons
                .iter()
                .all(|reason| *reason == DelayedAciReasonV1::LabelNotEligible);
        let only_blocked = report.eligibility == LabelConsumptionEligibilityV1::Blocked
            && reasons
                .iter()
                .all(|reason| *reason == DelayedAciReasonV1::LabelNotEligible);
        let disposition = if only_unknown {
            DelayedAciDispositionV1::HeldUnknown
        } else if only_blocked {
            DelayedAciDispositionV1::Blocked
        } else {
            DelayedAciDispositionV1::Rejected
        };
        return build_update_receipt(
            engine,
            delivery,
            Some(&prediction_receipt),
            disposition,
            reasons,
            alpha_before,
            alpha_before,
            None,
            prediction.and_then(|item| item.feedback_phase),
            None,
        );
    }

    let miss = delivery.label_value < prediction_receipt.interval_lower
        || delivery.label_value > prediction_receipt.interval_upper;
    let direction = engine.policy.target_miscoverage - if miss { 1.0 } else { 0.0 };
    let alpha_base = match engine.policy.feedback_mode {
        DelayedAciFeedbackModeV1::ArrivalOrder => engine.current_alpha,
        DelayedAciFeedbackModeV1::FixedDelayPhased { .. } => {
            prediction_receipt.alpha_state_used
        }
    };
    let candidate_alpha =
        (alpha_base + engine.policy.learning_rate * direction).clamp(0.0, 1.0);
    if !candidate_alpha.is_finite() {
        return Err(DelayedAciErrorV1("candidate alpha is non-finite".into()));
    }
    let next_count = engine
        .accepted_feedback_count
        .checked_add(1)
        .ok_or_else(|| DelayedAciErrorV1("accepted feedback count overflow".into()))?;
    let receipt = build_update_receipt(
        engine,
        delivery,
        Some(&prediction_receipt),
        if miss {
            DelayedAciDispositionV1::UpdatedMissed
        } else {
            DelayedAciDispositionV1::UpdatedCovered
        },
        Vec::new(),
        alpha_before,
        candidate_alpha,
        Some(miss),
        prediction_receipt.feedback_phase,
        Some(next_count),
    )?;

    match engine.policy.feedback_mode {
        DelayedAciFeedbackModeV1::ArrivalOrder => {
            engine.current_alpha = candidate_alpha;
        }
        DelayedAciFeedbackModeV1::FixedDelayPhased { delay_steps } => {
            let phase = prediction_receipt
                .feedback_phase
                .expect("fixed-delay prediction has a phase") as usize;
            engine.phase_alpha_states[phase] = candidate_alpha;
            engine.next_feedback_sequence_by_phase[phase] = prediction_receipt
                .sequence
                .checked_add(delay_steps)
                .ok_or_else(|| DelayedAciErrorV1("next feedback sequence overflow".into()))?;
        }
    }
    engine.accepted_feedback_count = next_count;
    engine.last_delivery_time_unix_seconds = Some(delivery.delivered_at_unix_seconds);
    engine.accepted_label_revisions.insert(format!(
        "{}:{}",
        delivery.label_id, delivery.label_revision
    ));
    engine.pending_predictions.remove(&delivery.sample_id);
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy() -> DelayedAciPolicyV1 {
        DelayedAciPolicyV1 {
            schema: DELAYED_ACI_POLICY_SCHEMA_V1.into(),
            policy_id: "delayed-aci-v1".into(),
            revision: 1,
            target_miscoverage: 0.05,
            learning_rate: 0.10,
            feedback_mode: DelayedAciFeedbackModeV1::ArrivalOrder,
        }
    }

    fn engine() -> DelayedAciEngineV1 {
        create_delayed_aci_engine_v1(
            "aci-engine-001",
            "fab-clock",
            &digest('a'),
            &digest('b'),
            policy(),
            0.10,
        )
        .unwrap()
    }

    fn prediction(sample: &str, event: &str, event_time: i64, prediction_time: i64, alpha: f64) -> DelayedAciPredictionInputV1 {
        DelayedAciPredictionInputV1 {
            schema: DELAYED_ACI_PREDICTION_SCHEMA_V1.into(),
            sample_id: sample.into(),
            measurement_event_id: event.into(),
            event_time_unix_seconds: event_time,
            feature_available_at_unix_seconds: event_time + 1,
            prediction_at_unix_seconds: prediction_time,
            clock_domain_id: "fab-clock".into(),
            global_model_artifact_sha256: digest('a'),
            calibration_model_sha256: digest('b'),
            alpha_state_used: alpha,
            point_prediction: 10.0,
            interval_lower: 9.0,
            interval_upper: 11.0,
            residual_scale: Some(1.0),
        }
    }

    fn report(
        label_id: &str,
        revision: u64,
        event_id: &str,
        payload: &str,
        eligibility: LabelConsumptionEligibilityV1,
        frontier: LabelEvidenceFrontierV1,
    ) -> LabelConsumptionEligibilityReportV1 {
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label_id.into(),
            label_revision: revision,
            measurement_event_id: event_id.into(),
            calibration_certificate_id: "cal-001".into(),
            frontier,
            eligibility,
            downstream_consumption_allowed:
                eligibility == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: payload.into(),
            measurement_sha256: digest('c'),
            qualified_corrected_value: 10.0,
            qualified_standard_uncertainty_upper_bound: Some(0.1),
            qualification_report_sha256: digest('d'),
            requalification_report_sha256: digest('e'),
            requalification_frontier_sha256: digest('f'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        report.report_sha256 = report_digest(&report).unwrap();
        report
    }

    fn delivery(
        sample: &str,
        event: &str,
        label_id: &str,
        revision: u64,
        status: LabelConsumptionEligibilityV1,
        frontier: LabelEvidenceFrontierV1,
        label_value: f64,
        label_available: i64,
        report_available: i64,
        delivered: i64,
    ) -> DelayedAciLabelDeliveryV1 {
        let payload = digest('c');
        DelayedAciLabelDeliveryV1 {
            schema: DELAYED_ACI_DELIVERY_SCHEMA_V1.into(),
            sample_id: sample.into(),
            label_id: label_id.into(),
            label_revision: revision,
            measurement_event_id: event.into(),
            label_payload_sha256: payload.clone(),
            label_value,
            label_available_at_unix_seconds: label_available,
            eligibility_report_available_at_unix_seconds: report_available,
            delivered_at_unix_seconds: delivered,
            delivery_round_sequence: delivered as u64,
            clock_domain_id: "fab-clock".into(),
            idempotency_key: format!("{label_id}:{revision}"),
            eligibility_report: {
            let mut eligibility = report(
                label_id,
                revision,
                event,
                &payload,
                status,
                frontier,
            );
            eligibility.qualified_corrected_value = label_value;
            eligibility.report_sha256 = report_digest(&eligibility).unwrap();
            eligibility
        },
        }
    }

    #[test]
    fn prediction_freezes_alpha_and_interval_without_updating_alpha() {
        let mut engine = engine();
        let before = alpha_state_digest(&engine).unwrap();
        let receipt = register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s1", "e1", 100, 102, 0.10),
        )
        .unwrap();
        assert_eq!(receipt.alpha_state_used, 0.10);
        assert_eq!(receipt.interval_lower, 9.0);
        assert_eq!(receipt.interval_upper, 11.0);
        assert_eq!(alpha_state_digest(&engine).unwrap(), before);
    }

    #[test]
    fn covered_label_updates_alpha_by_aci_recursion() {
        let mut engine = engine();
        register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s1", "e1", 100, 102, 0.10),
        )
        .unwrap();
        let receipt = deliver_delayed_aci_feedback_v1(
            &mut engine,
            &delivery("s1", "e1", "l1", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, LabelEvidenceFrontierV1::CurrentAtK1, 10.5, 110, 111, 112),
        )
        .unwrap();
        assert_eq!(receipt.disposition, DelayedAciDispositionV1::UpdatedCovered);
        assert!((receipt.alpha_after - 0.105).abs() < 1.0e-12);
        assert!(receipt.alpha_state_mutated);
        assert_eq!(engine.accepted_feedback_count, 1);
    }

    #[test]
    fn miss_decreases_alpha_and_makes_future_intervals_more_conservative() {
        let mut engine = engine();
        register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s1", "e1", 100, 102, 0.10),
        )
        .unwrap();
        let receipt = deliver_delayed_aci_feedback_v1(
            &mut engine,
            &delivery("s1", "e1", "l1", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, LabelEvidenceFrontierV1::CurrentAtK1, 12.0, 110, 111, 112),
        )
        .unwrap();
        assert_eq!(receipt.disposition, DelayedAciDispositionV1::UpdatedMissed);
        assert!((receipt.alpha_after - 0.005).abs() < 1.0e-12);
    }

    #[test]
    fn unknown_feedback_is_held_without_calibration_mutation_and_can_be_represented() {
        let mut engine = engine();
        register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s1", "e1", 100, 102, 0.10),
        )
        .unwrap();
        let before = alpha_state_digest(&engine).unwrap();
        let held = delivery("s1", "e1", "l1", 1, LabelConsumptionEligibilityV1::Unknown, LabelEvidenceFrontierV1::CurrentAtK1, 10.5, 110, 111, 112);
        let receipt = deliver_delayed_aci_feedback_v1(&mut engine, &held).unwrap();
        assert_eq!(receipt.disposition, DelayedAciDispositionV1::HeldUnknown);
        assert_eq!(alpha_state_digest(&engine).unwrap(), before);
        assert!(engine.pending_predictions.contains_key("s1"));

        let qualified = delivery("s1", "e1", "l1", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, LabelEvidenceFrontierV1::CurrentAtK1, 10.5, 110, 113, 114);
        assert_eq!(
            deliver_delayed_aci_feedback_v1(&mut engine, &qualified).unwrap().disposition,
            DelayedAciDispositionV1::UpdatedCovered
        );
    }

    #[test]
    fn blocked_feedback_never_changes_alpha() {
        let mut engine = engine();
        register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s1", "e1", 100, 102, 0.10),
        )
        .unwrap();
        let before = alpha_state_digest(&engine).unwrap();
        let receipt = deliver_delayed_aci_feedback_v1(
            &mut engine,
            &delivery("s1", "e1", "l1", 1, LabelConsumptionEligibilityV1::Blocked, LabelEvidenceFrontierV1::CurrentAtK1, 10.5, 110, 111, 112),
        )
        .unwrap();
        assert_eq!(receipt.disposition, DelayedAciDispositionV1::Blocked);
        assert_eq!(alpha_state_digest(&engine).unwrap(), before);
    }

    #[test]
    fn duplicate_revision_cannot_update_twice() {
        let mut engine = engine();
        register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s1", "e1", 100, 102, 0.10),
        )
        .unwrap();
        let d = delivery("s1", "e1", "l1", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, LabelEvidenceFrontierV1::CurrentAtK1, 10.5, 110, 111, 112);
        assert_eq!(deliver_delayed_aci_feedback_v1(&mut engine, &d).unwrap().disposition, DelayedAciDispositionV1::UpdatedCovered);
        let before = engine.current_alpha;
        assert_eq!(deliver_delayed_aci_feedback_v1(&mut engine, &d).unwrap().disposition, DelayedAciDispositionV1::Duplicate);
        assert_eq!(engine.current_alpha, before);
    }

    #[test]
    fn report_revision_identity_value_payload_and_clock_mismatches_are_rejected() {
        let mut engine = engine();
        register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s1", "e1", 100, 102, 0.10),
        )
        .unwrap();
        let before = engine.current_alpha;
        let mut d = delivery("s1", "e1", "l1", 2, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, LabelEvidenceFrontierV1::CurrentAtK1, 10.5, 110, 111, 112);
        d.eligibility_report.label_revision = 1;
        d.eligibility_report.report_sha256 = report_digest(&d.eligibility_report).unwrap();
        let receipt = deliver_delayed_aci_feedback_v1(&mut engine, &d).unwrap();
        assert_eq!(receipt.disposition, DelayedAciDispositionV1::Rejected);
        assert_eq!(engine.current_alpha, before);
    }

    #[test]
    fn non_k1_and_tampered_report_are_rejected() {
        let mut engine = engine();
        register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s1", "e1", 100, 102, 0.10),
        )
        .unwrap();
        let before = engine.current_alpha;
        let stale = delivery("s1", "e1", "l1", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, LabelEvidenceFrontierV1::AsKnownAtK0, 10.5, 110, 111, 112);
        assert_eq!(deliver_delayed_aci_feedback_v1(&mut engine, &stale).unwrap().disposition, DelayedAciDispositionV1::Rejected);
        let mut tampered = delivery("s1", "e1", "l2", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, LabelEvidenceFrontierV1::CurrentAtK1, 10.5, 110, 111, 112);
        tampered.eligibility_report.report_sha256 = digest('0');
        assert_eq!(deliver_delayed_aci_feedback_v1(&mut engine, &tampered).unwrap().disposition, DelayedAciDispositionV1::Rejected);
        assert_eq!(engine.current_alpha, before);
    }

    #[test]
    fn label_and_report_availability_must_precede_delivery() {
        let mut engine = engine();
        register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s1", "e1", 100, 102, 0.10),
        )
        .unwrap();
        let before = engine.current_alpha;
        let early = delivery("s1", "e1", "l1", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, LabelEvidenceFrontierV1::CurrentAtK1, 10.5, 110, 111, 109);
        assert_eq!(deliver_delayed_aci_feedback_v1(&mut engine, &early).unwrap().disposition, DelayedAciDispositionV1::Rejected);
        assert_eq!(engine.current_alpha, before);
    }

    #[test]
    fn delivery_time_must_be_monotonic_even_when_sample_order_is_out_of_order() {
        let mut engine = engine();
        for (sample, event, event_time) in [("s1", "e1", 100), ("s2", "e2", 103)] {
            register_delayed_aci_prediction_v1(
                &mut engine,
                &prediction(sample, event, event_time, event_time + 2, 0.10),
            )
            .unwrap();
        }
        let later = delivery("s2", "e2", "l2", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, LabelEvidenceFrontierV1::CurrentAtK1, 10.5, 112, 113, 114);
        assert_eq!(deliver_delayed_aci_feedback_v1(&mut engine, &later).unwrap().disposition, DelayedAciDispositionV1::UpdatedCovered);
        let earlier = delivery("s1", "e1", "l1", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, LabelEvidenceFrontierV1::CurrentAtK1, 10.5, 110, 111, 112);
        assert_eq!(deliver_delayed_aci_feedback_v1(&mut engine, &earlier).unwrap().disposition, DelayedAciDispositionV1::Rejected);
    }

    #[test]
    fn fixed_delay_uses_independent_interleaved_alpha_phases() {
        let mut configured = policy();
        configured.feedback_mode = DelayedAciFeedbackModeV1::FixedDelayPhased { delay_steps: 3 };
        let mut engine = create_delayed_aci_engine_v1(
            "aci-fixed-001",
            "fab-clock",
            &digest('a'),
            &digest('b'),
            configured,
            0.10,
        )
        .unwrap();

        for index in 0..3 {
            let sample = format!("s{index}");
            let event = format!("e{index}");
            let alpha = current_delayed_aci_alpha_v1(&engine).unwrap();
            assert!((alpha - 0.10).abs() < 1.0e-12);
            register_delayed_aci_prediction_v1(
                &mut engine,
                &prediction(&sample, &event, 100 + index, 102 + index, alpha),
            )
            .unwrap();
        }

        let first = delivery(
            "s0",
            "e0",
            "l0",
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            10.5,
            110,
            111,
            112,
        );
        let mut first = first;
        first.delivery_round_sequence = 3;
        let receipt = deliver_delayed_aci_feedback_v1(&mut engine, &first).unwrap();
        assert_eq!(receipt.disposition, DelayedAciDispositionV1::UpdatedCovered);
        assert!((engine.phase_alpha_states[0] - 0.105).abs() < 1.0e-12);
        assert!((engine.phase_alpha_states[1] - 0.10).abs() < 1.0e-12);
        assert!((engine.phase_alpha_states[2] - 0.10).abs() < 1.0e-12);

        let alpha_for_sequence_three = current_delayed_aci_alpha_v1(&engine).unwrap();
        assert!((alpha_for_sequence_three - 0.105).abs() < 1.0e-12);
        register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s3", "e3", 103, 105, alpha_for_sequence_three),
        )
        .unwrap();
        assert!((engine.phase_alpha_states[0] - 0.105).abs() < 1.0e-12);
    }

    #[test]
    fn fixed_delay_blocks_next_same_phase_prediction_until_prior_feedback_is_consumed() {
        let mut configured = policy();
        configured.feedback_mode = DelayedAciFeedbackModeV1::FixedDelayPhased { delay_steps: 2 };
        let mut engine = create_delayed_aci_engine_v1(
            "aci-fixed-002",
            "fab-clock",
            &digest('a'),
            &digest('b'),
            configured,
            0.10,
        )
        .unwrap();
        register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s0", "e0", 100, 102, 0.10),
        )
        .unwrap();
        register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s1", "e1", 101, 103, 0.10),
        )
        .unwrap();
        assert!(
            register_delayed_aci_prediction_v1(
                &mut engine,
                &prediction("s2", "e2", 104, 106, 0.10),
            )
            .is_err()
        );
    }

    #[test]
    fn fixed_delay_rejects_feedback_arriving_on_wrong_stream_round() {
        let mut configured = policy();
        configured.feedback_mode = DelayedAciFeedbackModeV1::FixedDelayPhased { delay_steps: 2 };
        let mut engine = create_delayed_aci_engine_v1(
            "aci-fixed-003",
            "fab-clock",
            &digest('a'),
            &digest('b'),
            configured,
            0.10,
        )
        .unwrap();
        register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s0", "e0", 100, 102, 0.10),
        )
        .unwrap();
        register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s1", "e1", 101, 103, 0.10),
        )
        .unwrap();
        let mut d = delivery(
            "s0",
            "e0",
            "l0",
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            10.5,
            110,
            111,
            112,
        );
        d.delivery_round_sequence = 99;
        let receipt = deliver_delayed_aci_feedback_v1(&mut engine, &d).unwrap();
        assert_eq!(receipt.disposition, DelayedAciDispositionV1::Rejected);
        assert_eq!(engine.phase_alpha_states, vec![0.10, 0.10]);
    }

    #[test]
    fn alpha_is_projected_to_unit_interval() {
        let mut engine = engine();
        engine.current_alpha = 0.99;
        engine.policy.learning_rate = 1.0;
        engine.policy_sha256 = policy_digest(&engine.policy).unwrap();
        let input = prediction("s1", "e1", 100, 102, 0.99);
        register_delayed_aci_prediction_v1(&mut engine, &input).unwrap();
        let d = delivery("s1", "e1", "l1", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, LabelEvidenceFrontierV1::CurrentAtK1, 10.5, 110, 111, 112);
        let receipt = deliver_delayed_aci_feedback_v1(&mut engine, &d).unwrap();
        assert_eq!(receipt.alpha_after, 1.0);
    }

    #[test]
    fn frozen_prediction_interval_is_used_after_alpha_has_changed() {
        let mut engine = engine();
        let prediction_one = register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s1", "e1", 100, 102, 0.10),
        )
        .unwrap();
        let prediction_two = register_delayed_aci_prediction_v1(
            &mut engine,
            &prediction("s2", "e2", 103, 105, 0.10),
        )
        .unwrap();
        let missed = delivery("s1", "e1", "l1", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, LabelEvidenceFrontierV1::CurrentAtK1, 12.0, 110, 111, 112);
        deliver_delayed_aci_feedback_v1(&mut engine, &missed).unwrap();
        assert!((engine.current_alpha - 0.005).abs() < 1.0e-12);
        assert_eq!(prediction_one.interval_upper, 11.0);
        assert_eq!(prediction_two.alpha_state_used, 0.10);
        assert_eq!(prediction_two.interval_upper, 11.0);
    }

    #[test]
    fn deterministic_replay_for_fixed_feedback_arrival_schedule() {
        fn replay() -> (f64, u64) {
            let mut engine = engine();
            for index in 0..3 {
                let sample = format!("s{index}");
                let event = format!("e{index}");
                register_delayed_aci_prediction_v1(
                    &mut engine,
                    &prediction(&sample, &event, 100 + index * 2, 101 + index * 2, 0.10),
                )
                .unwrap();
            }
            for index in 0..3 {
                let sample = format!("s{index}");
                let event = format!("e{index}");
                let value = if index == 1 { 12.0 } else { 10.5 };
                let d = delivery(&sample, &event, &format!("l{index}"), 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, LabelEvidenceFrontierV1::CurrentAtK1, value, 110 + index * 2, 111 + index * 2, 112 + index * 2);
                deliver_delayed_aci_feedback_v1(&mut engine, &d).unwrap();
            }
            (engine.current_alpha, engine.accepted_feedback_count)
        }
        assert_eq!(replay(), replay());
    }
}
