// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-gated, delayed-label streaming adapters for semiconductor VM.
//!
//! The global predictor is supplied as a frozen prediction. This module owns
//! only the online residual adapters and their event/receipt protocol. It
//! separates prediction time from label arrival, stores prediction-time
//! features, and updates state only from a current, numerically qualified K1
//! label receipt. It never fits the global predictor, executes equipment
//! control, or mutates historical prediction/update receipts.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};

pub const GATED_STREAMING_ADAPTER_SCHEMA_V1: &str =
    "luminous.semi-bench.gated-streaming-adapter.v1";
pub const STREAMING_ADAPTER_POLICY_SCHEMA_V1: &str =
    "luminous.semi-bench.streaming-adapter-policy.v1";
pub const STREAMING_PREDICTION_SCHEMA_V1: &str = "luminous.semi-bench.streaming-prediction.v1";
pub const STREAMING_DELIVERY_SCHEMA_V1: &str = "luminous.semi-bench.streaming-label-delivery.v1";
pub const STREAMING_UPDATE_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.streaming-update-receipt.v1";
pub const STREAMING_PREDICTION_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.streaming-prediction-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingUpdateDispositionV1 {
    Updated,
    HeldUnknown,
    Blocked,
    Rejected,
    Duplicate,
    PendingPredictionMissing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingUpdateReasonV1 {
    LabelNotCurrentK1,
    LabelNotEligible,
    DownstreamConsumptionNotAllowed,
    LabelIdentityMismatch,
    LabelRevisionMismatch,
    MeasurementEventMismatch,
    LabelPayloadDigestMismatch,
    EligibilityReportDigestMismatch,
    ReportNumericValueNonFinite,
    DeliveredValueMismatch,
    LabelRevisionInvalid,
    IdempotencyKeyMismatch,
    ClockDomainMismatch,
    DeliveryBeforeLabelAvailability,
    EligibilityReportUnavailableAtDelivery,
    EligibilityReportBeforeLabelAvailability,
    DeliveryBeforePrediction,
    LabelAvailabilityBeforeMeasurement,
    FeatureSchemaMismatch,
    PendingPredictionMissing,
    DuplicateAcceptedLabelRevision,
    InvalidStreamingNumericInput,
    FeatureDimensionMismatch,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamingAdapterPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    /// Normalized LMS learning rate.
    pub feature_learning_rate: f64,
    /// Added to ||x||² to avoid division by zero.
    pub feature_norm_floor: f64,
    /// Per-coordinate maximum absolute weight change.
    pub maximum_abs_weight_delta: f64,
    pub maximum_abs_feature_value: f64,
    pub maximum_abs_prediction_error: f64,
    pub global_context_ema_alpha: f64,
    pub local_context_ema_alpha: f64,
    /// Local context residuals are shrunk toward the global residual until this support is reached.
    pub minimum_context_support: u64,
    pub temporal_ema_alpha: f64,
    pub temporal_activation_threshold: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamingContextStateV1 {
    pub residual_ema: f64,
    pub support_count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamingAdapterStateV1 {
    pub feature_weights: Vec<f64>,
    pub global_context_residual_ema: f64,
    pub global_temporal_residual_ema: f64,
    pub contexts: BTreeMap<String, StreamingContextStateV1>,
    pub accepted_update_count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GatedStreamingAdapterEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub feature_schema_sha256: String,
    pub policy_sha256: String,
    pub policy: StreamingAdapterPolicyV1,
    pub adapter_state: StreamingAdapterStateV1,
    pub next_prediction_sequence: u64,
    pub pending_predictions: BTreeMap<String, StreamingPredictionReceiptV1>,
    pub seen_measurement_event_ids: BTreeSet<String>,
    pub accepted_label_revisions: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamingPredictionInputV1 {
    pub schema: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub feature_available_at_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub feature_schema_sha256: String,
    pub feature_vector: Vec<f64>,
    /// Indices are in the input feature vector's declared schema order.
    pub selected_feature_indices: Vec<usize>,
    /// Prediction from the immutable/global VM model.
    pub global_baseline_prediction: f64,
    pub context_key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamingPredictionReceiptV1 {
    pub schema: String,
    pub sequence: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub feature_available_at_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub feature_schema_sha256: String,
    pub policy_sha256: String,
    pub context_key: Option<String>,
    pub feature_vector_sha256: String,
    pub selected_feature_indices: Vec<usize>,
    /// The selected feature values are frozen at prediction time for delayed updates.
    pub selected_feature_vector: Vec<f64>,
    pub global_baseline_prediction: f64,
    pub feature_adapter_prediction: f64,
    pub context_correction: f64,
    pub temporal_correction: f64,
    pub base_prediction: f64,
    pub prediction_value: f64,
    pub adapter_state_before_sha256: String,
    pub adapter_state_after_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamingLabelDeliveryV1 {
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
    pub clock_domain_id: String,
    pub idempotency_key: String,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamingUpdateReceiptV1 {
    pub schema: String,
    pub engine_id: String,
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub measurement_event_id: String,
    pub disposition: StreamingUpdateDispositionV1,
    pub reasons: Vec<StreamingUpdateReasonV1>,
    pub prediction_receipt_sha256: Option<String>,
    pub eligibility_report_sha256: Option<String>,
    pub policy_sha256: String,
    pub label_payload_sha256: String,
    pub adapter_state_before_sha256: String,
    pub adapter_state_after_sha256: String,
    pub update_sequence: Option<u64>,
    pub adapter_state_mutated: bool,
    pub retraining_executed: bool,
    pub equipment_control_authority: bool,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct GatedStreamingAdapterErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), GatedStreamingAdapterErrorV1> {
    if value.trim().is_empty() {
        return Err(GatedStreamingAdapterErrorV1(format!(
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

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, GatedStreamingAdapterErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        GatedStreamingAdapterErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn adapter_state_digest(
    state: &StreamingAdapterStateV1,
) -> Result<String, GatedStreamingAdapterErrorV1> {
    json_sha256(state)
}

fn eligibility_report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, GatedStreamingAdapterErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn prediction_receipt_digest(
    receipt: &StreamingPredictionReceiptV1,
) -> Result<String, GatedStreamingAdapterErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn update_receipt_digest(
    receipt: &StreamingUpdateReceiptV1,
) -> Result<String, GatedStreamingAdapterErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn validate_policy(policy: &StreamingAdapterPolicyV1) -> Result<(), GatedStreamingAdapterErrorV1> {
    if policy.schema != STREAMING_ADAPTER_POLICY_SCHEMA_V1 {
        return Err(GatedStreamingAdapterErrorV1(
            "streaming adapter policy schema is unsupported".into(),
        ));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if !policy.feature_learning_rate.is_finite()
        || policy.feature_learning_rate < 0.0
        || !policy.feature_norm_floor.is_finite()
        || policy.feature_norm_floor <= 0.0
        || !policy.maximum_abs_weight_delta.is_finite()
        || policy.maximum_abs_weight_delta <= 0.0
        || !policy.maximum_abs_feature_value.is_finite()
        || policy.maximum_abs_feature_value <= 0.0
        || !policy.maximum_abs_prediction_error.is_finite()
        || policy.maximum_abs_prediction_error <= 0.0
        || !policy.global_context_ema_alpha.is_finite()
        || !(0.0..=1.0).contains(&policy.global_context_ema_alpha)
        || !policy.local_context_ema_alpha.is_finite()
        || !(0.0..=1.0).contains(&policy.local_context_ema_alpha)
        || !policy.temporal_ema_alpha.is_finite()
        || !(0.0..=1.0).contains(&policy.temporal_ema_alpha)
        || !policy.temporal_activation_threshold.is_finite()
        || policy.temporal_activation_threshold < 0.0
    {
        return Err(GatedStreamingAdapterErrorV1(
            "streaming adapter policy contains invalid numeric parameters".into(),
        ));
    }
    Ok(())
}

fn validate_engine(
    engine: &GatedStreamingAdapterEngineV1,
) -> Result<(), GatedStreamingAdapterErrorV1> {
    if engine.schema != GATED_STREAMING_ADAPTER_SCHEMA_V1 {
        return Err(GatedStreamingAdapterErrorV1(
            "streaming adapter engine schema is unsupported".into(),
        ));
    }
    require_text(&engine.engine_id, "engine.engine_id")?;
    require_text(&engine.clock_domain_id, "engine.clock_domain_id")?;
    if !valid_sha256(&engine.global_model_artifact_sha256)
        || !valid_sha256(&engine.feature_schema_sha256)
        || !valid_sha256(&engine.policy_sha256)
    {
        return Err(GatedStreamingAdapterErrorV1(
            "engine model, feature-schema or policy digest is malformed".into(),
        ));
    }
    validate_policy(&engine.policy)?;
    if json_sha256(&engine.policy)? != engine.policy_sha256 {
        return Err(GatedStreamingAdapterErrorV1(
            "streaming adapter policy content differs from its sealed digest".into(),
        ));
    }
    if engine.adapter_state.feature_weights.is_empty()
        || engine
            .adapter_state
            .feature_weights
            .iter()
            .any(|value| !value.is_finite())
        || !engine.adapter_state.global_context_residual_ema.is_finite()
        || !engine
            .adapter_state
            .global_temporal_residual_ema
            .is_finite()
        || engine
            .adapter_state
            .contexts
            .values()
            .any(|context| !context.residual_ema.is_finite())
    {
        return Err(GatedStreamingAdapterErrorV1(
            "streaming adapter state contains an invalid numeric value".into(),
        ));
    }
    Ok(())
}

/// Creates an adapter engine. The global VM predictor and its weights are
/// intentionally external and immutable in v0.1.
pub fn create_gated_streaming_adapter_engine_v1(
    engine_id: &str,
    clock_domain_id: &str,
    global_model_artifact_sha256: &str,
    feature_schema_sha256: &str,
    initial_feature_weights: Vec<f64>,
    policy: StreamingAdapterPolicyV1,
) -> Result<GatedStreamingAdapterEngineV1, GatedStreamingAdapterErrorV1> {
    require_text(engine_id, "engine_id")?;
    require_text(clock_domain_id, "clock_domain_id")?;
    if !valid_sha256(global_model_artifact_sha256) || !valid_sha256(feature_schema_sha256) {
        return Err(GatedStreamingAdapterErrorV1(
            "global model or feature-schema digest is malformed".into(),
        ));
    }
    validate_policy(&policy)?;
    let policy_sha256 = json_sha256(&policy)?;
    if initial_feature_weights.is_empty()
        || initial_feature_weights
            .iter()
            .any(|weight| !weight.is_finite())
    {
        return Err(GatedStreamingAdapterErrorV1(
            "initial feature adapter weights must be non-empty and finite".into(),
        ));
    }
    Ok(GatedStreamingAdapterEngineV1 {
        schema: GATED_STREAMING_ADAPTER_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        clock_domain_id: clock_domain_id.into(),
        global_model_artifact_sha256: global_model_artifact_sha256.into(),
        feature_schema_sha256: feature_schema_sha256.into(),
        policy_sha256,
        policy,
        adapter_state: StreamingAdapterStateV1 {
            feature_weights: initial_feature_weights,
            global_context_residual_ema: 0.0,
            global_temporal_residual_ema: 0.0,
            contexts: BTreeMap::new(),
            accepted_update_count: 0,
        },
        next_prediction_sequence: 0,
        pending_predictions: BTreeMap::new(),
        seen_measurement_event_ids: BTreeSet::new(),
        accepted_label_revisions: BTreeSet::new(),
    })
}

fn context_correction(engine: &GatedStreamingAdapterEngineV1, context_key: Option<&str>) -> f64 {
    let Some(key) = context_key else {
        return engine.adapter_state.global_context_residual_ema;
    };
    let Some(local) = engine.adapter_state.contexts.get(key) else {
        return engine.adapter_state.global_context_residual_ema;
    };
    let minimum = engine.policy.minimum_context_support as f64;
    let support = local.support_count as f64;
    let local_weight = if minimum <= 0.0 {
        1.0
    } else {
        support / (support + minimum)
    };
    (1.0 - local_weight) * engine.adapter_state.global_context_residual_ema
        + local_weight * local.residual_ema
}

/// Creates an immutable prediction receipt and registers it for a future label.
///
/// Prediction is read-only with respect to adapter parameters. The engine
/// stores the exact selected features and prediction decomposition so delayed
/// labels never recompute an old residual against a newer model state.
pub fn predict_and_register_stream_sample_v1(
    engine: &mut GatedStreamingAdapterEngineV1,
    input: &StreamingPredictionInputV1,
) -> Result<StreamingPredictionReceiptV1, GatedStreamingAdapterErrorV1> {
    validate_engine(engine)?;
    if input.schema != STREAMING_PREDICTION_SCHEMA_V1 {
        return Err(GatedStreamingAdapterErrorV1(
            "streaming prediction input schema is unsupported".into(),
        ));
    }
    require_text(&input.sample_id, "input.sample_id")?;
    require_text(&input.measurement_event_id, "input.measurement_event_id")?;
    require_text(&input.clock_domain_id, "input.clock_domain_id")?;
    if input.clock_domain_id != engine.clock_domain_id {
        return Err(GatedStreamingAdapterErrorV1(
            "prediction clock domain does not match the engine".into(),
        ));
    }
    if input.global_model_artifact_sha256 != engine.global_model_artifact_sha256 {
        return Err(GatedStreamingAdapterErrorV1(
            "prediction global-model artifact does not match the engine".into(),
        ));
    }
    if input.feature_schema_sha256 != engine.feature_schema_sha256 {
        return Err(GatedStreamingAdapterErrorV1(
            "prediction feature schema does not match the engine".into(),
        ));
    }
    if input.event_time_unix_seconds > input.feature_available_at_unix_seconds
        || input.feature_available_at_unix_seconds > input.prediction_at_unix_seconds
    {
        return Err(GatedStreamingAdapterErrorV1(
            "prediction event/feature/prediction chronology is invalid".into(),
        ));
    }
    if !input.global_baseline_prediction.is_finite()
        || input.feature_vector.iter().any(|value| {
            !value.is_finite() || value.abs() > engine.policy.maximum_abs_feature_value
        })
    {
        return Err(GatedStreamingAdapterErrorV1(
            "prediction contains a non-finite or out-of-policy feature/baseline value".into(),
        ));
    }
    if input
        .context_key
        .as_deref()
        .is_some_and(|key| key.trim().is_empty())
    {
        return Err(GatedStreamingAdapterErrorV1(
            "context_key cannot be empty when supplied".into(),
        ));
    }
    if input.selected_feature_indices.is_empty()
        || input.selected_feature_indices.len() != engine.adapter_state.feature_weights.len()
    {
        return Err(GatedStreamingAdapterErrorV1(
            "selected feature count does not match adapter weight count".into(),
        ));
    }
    let mut selected_seen = BTreeSet::new();
    let mut selected_feature_vector = Vec::with_capacity(input.selected_feature_indices.len());
    for index in &input.selected_feature_indices {
        if *index >= input.feature_vector.len() || !selected_seen.insert(*index) {
            return Err(GatedStreamingAdapterErrorV1(
                "selected feature indices are out of range or duplicated".into(),
            ));
        }
        selected_feature_vector.push(input.feature_vector[*index]);
    }
    if engine.pending_predictions.contains_key(&input.sample_id)
        || engine
            .seen_measurement_event_ids
            .contains(&input.measurement_event_id)
    {
        return Err(GatedStreamingAdapterErrorV1(
            "duplicate sample or measurement event prediction".into(),
        ));
    }

    let state_before = adapter_state_digest(&engine.adapter_state)?;
    let feature_adapter_prediction = engine
        .adapter_state
        .feature_weights
        .iter()
        .zip(&selected_feature_vector)
        .map(|(weight, feature)| weight * feature)
        .sum::<f64>();
    let context = context_correction(engine, input.context_key.as_deref());
    let base_prediction = input.global_baseline_prediction + feature_adapter_prediction + context;
    let temporal = if engine.adapter_state.global_temporal_residual_ema.abs()
        >= engine.policy.temporal_activation_threshold
    {
        engine.adapter_state.global_temporal_residual_ema
    } else {
        0.0
    };
    let prediction_value = base_prediction + temporal;
    if !base_prediction.is_finite() || !prediction_value.is_finite() {
        return Err(GatedStreamingAdapterErrorV1(
            "streaming prediction output is non-finite".into(),
        ));
    }
    let feature_vector_sha256 = json_sha256(&input.feature_vector)?;
    let next_prediction_sequence = engine
        .next_prediction_sequence
        .checked_add(1)
        .ok_or_else(|| GatedStreamingAdapterErrorV1("prediction sequence overflow".into()))?;
    let mut receipt = StreamingPredictionReceiptV1 {
        schema: STREAMING_PREDICTION_RECEIPT_SCHEMA_V1.into(),
        sequence: engine.next_prediction_sequence,
        sample_id: input.sample_id.clone(),
        measurement_event_id: input.measurement_event_id.clone(),
        event_time_unix_seconds: input.event_time_unix_seconds,
        feature_available_at_unix_seconds: input.feature_available_at_unix_seconds,
        prediction_at_unix_seconds: input.prediction_at_unix_seconds,
        clock_domain_id: input.clock_domain_id.clone(),
        global_model_artifact_sha256: input.global_model_artifact_sha256.clone(),
        feature_schema_sha256: input.feature_schema_sha256.clone(),
        policy_sha256: engine.policy_sha256.clone(),
        context_key: input.context_key.clone(),
        feature_vector_sha256,
        selected_feature_indices: input.selected_feature_indices.clone(),
        selected_feature_vector,
        global_baseline_prediction: input.global_baseline_prediction,
        feature_adapter_prediction,
        context_correction: context,
        temporal_correction: temporal,
        base_prediction,
        prediction_value,
        adapter_state_before_sha256: state_before.clone(),
        adapter_state_after_sha256: state_before,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = prediction_receipt_digest(&receipt)?;

    engine
        .seen_measurement_event_ids
        .insert(input.measurement_event_id.clone());
    engine
        .pending_predictions
        .insert(input.sample_id.clone(), receipt.clone());
    engine.next_prediction_sequence = next_prediction_sequence;
    Ok(receipt)
}

fn update_receipt(
    engine: &GatedStreamingAdapterEngineV1,
    delivery: &StreamingLabelDeliveryV1,
    prediction: Option<&StreamingPredictionReceiptV1>,
    disposition: StreamingUpdateDispositionV1,
    reasons: Vec<StreamingUpdateReasonV1>,
    state_digests: (&str, &str),
    update_sequence: Option<u64>,
) -> Result<StreamingUpdateReceiptV1, GatedStreamingAdapterErrorV1> {
    let mut receipt = StreamingUpdateReceiptV1 {
        schema: STREAMING_UPDATE_RECEIPT_SCHEMA_V1.into(),
        engine_id: engine.engine_id.clone(),
        sample_id: delivery.sample_id.clone(),
        label_id: delivery.label_id.clone(),
        label_revision: delivery.label_revision,
        measurement_event_id: delivery.measurement_event_id.clone(),
        disposition,
        reasons,
        prediction_receipt_sha256: prediction.map(|value| value.receipt_sha256.clone()),
        eligibility_report_sha256: Some(delivery.eligibility_report.report_sha256.clone()),
        policy_sha256: engine.policy_sha256.clone(),
        label_payload_sha256: delivery.label_payload_sha256.clone(),
        adapter_state_before_sha256: state_digests.0.into(),
        adapter_state_after_sha256: state_digests.1.into(),
        update_sequence,
        adapter_state_mutated: update_sequence.is_some(),
        retraining_executed: false,
        equipment_control_authority: false,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = update_receipt_digest(&receipt)?;
    Ok(receipt)
}

/// Delivers one physical label to the stream. Adapter state changes only when
/// the K1 eligibility receipt is current, digest-valid, identity/value-bound,
/// and available by the declared delivery time.
pub fn deliver_qualified_stream_label_v1(
    engine: &mut GatedStreamingAdapterEngineV1,
    delivery: &StreamingLabelDeliveryV1,
) -> Result<StreamingUpdateReceiptV1, GatedStreamingAdapterErrorV1> {
    validate_engine(engine)?;
    if delivery.schema != STREAMING_DELIVERY_SCHEMA_V1 {
        return Err(GatedStreamingAdapterErrorV1(
            "streaming label delivery schema is unsupported".into(),
        ));
    }
    require_text(&delivery.sample_id, "delivery.sample_id")?;
    require_text(&delivery.label_id, "delivery.label_id")?;
    require_text(
        &delivery.measurement_event_id,
        "delivery.measurement_event_id",
    )?;
    require_text(&delivery.clock_domain_id, "delivery.clock_domain_id")?;
    require_text(&delivery.idempotency_key, "delivery.idempotency_key")?;
    if delivery.label_revision == 0 || !valid_sha256(&delivery.label_payload_sha256) {
        return Err(GatedStreamingAdapterErrorV1(
            "label revision or payload digest is invalid".into(),
        ));
    }

    let before = adapter_state_digest(&engine.adapter_state)?;
    let report = &delivery.eligibility_report;
    let mut reasons = Vec::new();
    let prediction = engine.pending_predictions.get(&delivery.sample_id);

    if engine.accepted_label_revisions.contains(&format!(
        "{}:{}",
        delivery.label_id, delivery.label_revision
    )) {
        reasons.push(StreamingUpdateReasonV1::DuplicateAcceptedLabelRevision);
        return update_receipt(
            engine,
            delivery,
            prediction,
            StreamingUpdateDispositionV1::Duplicate,
            reasons,
            (&before, &before),
            None,
        );
    }

    if delivery.clock_domain_id != engine.clock_domain_id {
        reasons.push(StreamingUpdateReasonV1::ClockDomainMismatch);
    }
    if !delivery.label_value.is_finite()
        || !report.qualified_corrected_value.is_finite()
        || report.qualified_standard_uncertainty_upper_bound.is_none()
        || report
            .qualified_standard_uncertainty_upper_bound
            .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        reasons.push(StreamingUpdateReasonV1::ReportNumericValueNonFinite);
    }
    if delivery.eligibility_report_available_at_unix_seconds
        < delivery.label_available_at_unix_seconds
    {
        reasons.push(StreamingUpdateReasonV1::EligibilityReportBeforeLabelAvailability);
    }
    if delivery.delivered_at_unix_seconds < delivery.eligibility_report_available_at_unix_seconds {
        reasons.push(StreamingUpdateReasonV1::EligibilityReportUnavailableAtDelivery);
    }
    if !valid_sha256(&report.label_sha256) || report.label_sha256 != delivery.label_payload_sha256 {
        reasons.push(StreamingUpdateReasonV1::LabelPayloadDigestMismatch);
    }
    if eligibility_report_digest(report)? != report.report_sha256 {
        reasons.push(StreamingUpdateReasonV1::EligibilityReportDigestMismatch);
    }
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        reasons.push(StreamingUpdateReasonV1::LabelNotCurrentK1);
    }
    match report.eligibility {
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
            if report.downstream_consumption_allowed => {}
        LabelConsumptionEligibilityV1::Unknown => {
            reasons.push(StreamingUpdateReasonV1::LabelNotEligible);
        }
        LabelConsumptionEligibilityV1::Blocked => {
            reasons.push(StreamingUpdateReasonV1::LabelNotEligible);
        }
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => {
            reasons.push(StreamingUpdateReasonV1::DownstreamConsumptionNotAllowed);
        }
    }
    if report.label_id != delivery.label_id {
        reasons.push(StreamingUpdateReasonV1::LabelIdentityMismatch);
    }
    if report.label_revision != delivery.label_revision {
        reasons.push(StreamingUpdateReasonV1::LabelRevisionMismatch);
    }
    if report.measurement_event_id != delivery.measurement_event_id {
        reasons.push(StreamingUpdateReasonV1::MeasurementEventMismatch);
    }
    if report.qualified_corrected_value != delivery.label_value {
        reasons.push(StreamingUpdateReasonV1::DeliveredValueMismatch);
    }
    if delivery.idempotency_key != format!("{}:{}", delivery.label_id, delivery.label_revision) {
        reasons.push(StreamingUpdateReasonV1::IdempotencyKeyMismatch);
    }
    if let Some(prediction) = prediction {
        if prediction.measurement_event_id != delivery.measurement_event_id {
            reasons.push(StreamingUpdateReasonV1::MeasurementEventMismatch);
        }
        if prediction.clock_domain_id != delivery.clock_domain_id {
            reasons.push(StreamingUpdateReasonV1::ClockDomainMismatch);
        }
        if delivery.label_available_at_unix_seconds < prediction.event_time_unix_seconds {
            reasons.push(StreamingUpdateReasonV1::LabelAvailabilityBeforeMeasurement);
        }
        if delivery.delivered_at_unix_seconds < delivery.label_available_at_unix_seconds {
            reasons.push(StreamingUpdateReasonV1::DeliveryBeforeLabelAvailability);
        }
        if delivery.delivered_at_unix_seconds
            < delivery.eligibility_report_available_at_unix_seconds
        {
            reasons.push(StreamingUpdateReasonV1::EligibilityReportUnavailableAtDelivery);
        }
        if delivery.delivered_at_unix_seconds < prediction.prediction_at_unix_seconds {
            reasons.push(StreamingUpdateReasonV1::DeliveryBeforePrediction);
        }
    }

    reasons.sort();
    reasons.dedup();

    let Some(prediction_receipt) = prediction.cloned() else {
        reasons.push(StreamingUpdateReasonV1::PendingPredictionMissing);
        let mut sorted = reasons.clone();
        sorted.sort();
        sorted.dedup();
        return update_receipt(
            engine,
            delivery,
            None,
            StreamingUpdateDispositionV1::PendingPredictionMissing,
            sorted,
            (&before, &before),
            None,
        );
    };

    if !reasons.is_empty() {
        let only_eligibility_unknown = report.eligibility == LabelConsumptionEligibilityV1::Unknown
            && reasons
                .iter()
                .all(|reason| *reason == StreamingUpdateReasonV1::LabelNotEligible);
        let only_eligibility_blocked = report.eligibility == LabelConsumptionEligibilityV1::Blocked
            && reasons
                .iter()
                .all(|reason| *reason == StreamingUpdateReasonV1::LabelNotEligible);
        let disposition = if only_eligibility_unknown {
            StreamingUpdateDispositionV1::HeldUnknown
        } else if only_eligibility_blocked {
            StreamingUpdateDispositionV1::Blocked
        } else {
            StreamingUpdateDispositionV1::Rejected
        };
        return update_receipt(
            engine,
            delivery,
            Some(&prediction_receipt),
            disposition,
            reasons,
            (&before, &before),
            None,
        );
    }

    // All candidate mutations happen on a private clone. Any invalid delta,
    // overflow, non-finite result or receipt-serialization error leaves the
    // live model state and pending prediction unchanged.
    let error = delivery.label_value - prediction_receipt.prediction_value;
    if !error.is_finite() || error.abs() > engine.policy.maximum_abs_prediction_error {
        return update_receipt(
            engine,
            delivery,
            Some(&prediction_receipt),
            StreamingUpdateDispositionV1::Rejected,
            vec![StreamingUpdateReasonV1::InvalidStreamingNumericInput],
            (&before, &before),
            None,
        );
    }

    let denominator = prediction_receipt
        .selected_feature_vector
        .iter()
        .map(|feature| feature * feature)
        .sum::<f64>()
        + engine.policy.feature_norm_floor;
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(GatedStreamingAdapterErrorV1(
            "NLMS denominator is invalid".into(),
        ));
    }

    let mut next_state = engine.adapter_state.clone();
    for (weight, feature) in next_state
        .feature_weights
        .iter_mut()
        .zip(&prediction_receipt.selected_feature_vector)
    {
        let raw_delta = engine.policy.feature_learning_rate * error * feature / denominator;
        if !raw_delta.is_finite() {
            return Err(GatedStreamingAdapterErrorV1(
                "NLMS update delta is non-finite".into(),
            ));
        }
        *weight += raw_delta.clamp(
            -engine.policy.maximum_abs_weight_delta,
            engine.policy.maximum_abs_weight_delta,
        );
        if !weight.is_finite() {
            return Err(GatedStreamingAdapterErrorV1(
                "NLMS candidate weight is non-finite".into(),
            ));
        }
    }

    next_state.global_temporal_residual_ema = (1.0 - engine.policy.temporal_ema_alpha)
        * next_state.global_temporal_residual_ema
        + engine.policy.temporal_ema_alpha * error;

    let updated_feature_correction = next_state
        .feature_weights
        .iter()
        .zip(&prediction_receipt.selected_feature_vector)
        .map(|(weight, feature)| weight * feature)
        .sum::<f64>();
    if !updated_feature_correction.is_finite()
        || !next_state.global_temporal_residual_ema.is_finite()
    {
        return Err(GatedStreamingAdapterErrorV1(
            "streaming adapter candidate state is non-finite".into(),
        ));
    }

    // Residual left after the feature adapter is updated, preserving the
    // prediction-time context/temporal corrections as the local target.
    let context_residual = delivery.label_value
        - (prediction_receipt.global_baseline_prediction
            + updated_feature_correction
            + prediction_receipt.temporal_correction);
    if !context_residual.is_finite() {
        return Err(GatedStreamingAdapterErrorV1(
            "context residual is non-finite".into(),
        ));
    }
    next_state.global_context_residual_ema = (1.0 - engine.policy.global_context_ema_alpha)
        * next_state.global_context_residual_ema
        + engine.policy.global_context_ema_alpha * context_residual;

    if let Some(context_key) = prediction_receipt.context_key.clone() {
        let entry = next_state
            .contexts
            .entry(context_key)
            .or_insert(StreamingContextStateV1 {
                residual_ema: next_state.global_context_residual_ema,
                support_count: 0,
            });
        let next_support = entry
            .support_count
            .checked_add(1)
            .ok_or_else(|| GatedStreamingAdapterErrorV1("context support count overflow".into()))?;
        entry.residual_ema = (1.0 - engine.policy.local_context_ema_alpha) * entry.residual_ema
            + engine.policy.local_context_ema_alpha * context_residual;
        entry.support_count = next_support;
        if !entry.residual_ema.is_finite() {
            return Err(GatedStreamingAdapterErrorV1(
                "local context candidate state is non-finite".into(),
            ));
        }
    }

    next_state.accepted_update_count = next_state
        .accepted_update_count
        .checked_add(1)
        .ok_or_else(|| GatedStreamingAdapterErrorV1("accepted update count overflow".into()))?;
    let after = adapter_state_digest(&next_state)?;
    let receipt = update_receipt(
        engine,
        delivery,
        Some(&prediction_receipt),
        StreamingUpdateDispositionV1::Updated,
        Vec::new(),
        (&before, &after),
        Some(next_state.accepted_update_count),
    )?;

    engine.adapter_state = next_state;
    engine
        .accepted_label_revisions
        .insert(format!("{}:{}", delivery.label_id, delivery.label_revision));
    engine.pending_predictions.remove(&delivery.sample_id);
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy() -> StreamingAdapterPolicyV1 {
        StreamingAdapterPolicyV1 {
            schema: STREAMING_ADAPTER_POLICY_SCHEMA_V1.into(),
            policy_id: "grape-evidence-gated-adapter.v1".into(),
            revision: 1,
            feature_learning_rate: 0.20,
            feature_norm_floor: 1.0e-6,
            maximum_abs_weight_delta: 0.25,
            maximum_abs_feature_value: 100.0,
            maximum_abs_prediction_error: 100.0,
            global_context_ema_alpha: 0.25,
            local_context_ema_alpha: 0.50,
            minimum_context_support: 2,
            temporal_ema_alpha: 0.20,
            temporal_activation_threshold: 0.05,
        }
    }

    fn engine() -> GatedStreamingAdapterEngineV1 {
        create_gated_streaming_adapter_engine_v1(
            "stream-engine-001",
            "fab-clock",
            &digest('c'),
            &digest('a'),
            vec![0.0, 0.0],
            policy(),
        )
        .unwrap()
    }

    fn prediction(
        sample_id: &str,
        event_id: &str,
        event_time: i64,
        prediction_time: i64,
        context_key: Option<&str>,
        features: Vec<f64>,
    ) -> StreamingPredictionInputV1 {
        StreamingPredictionInputV1 {
            schema: STREAMING_PREDICTION_SCHEMA_V1.into(),
            sample_id: sample_id.into(),
            measurement_event_id: event_id.into(),
            event_time_unix_seconds: event_time,
            feature_available_at_unix_seconds: event_time + 1,
            prediction_at_unix_seconds: prediction_time,
            clock_domain_id: "fab-clock".into(),
            global_model_artifact_sha256: digest('c'),
            feature_schema_sha256: digest('a'),
            selected_feature_indices: vec![0, 1],
            feature_vector: features,
            global_baseline_prediction: 10.0,
            context_key: context_key.map(str::to_string),
        }
    }

    fn eligibility_report(
        label_id: &str,
        event_id: &str,
        payload_digest: &str,
        label_revision: u64,
        eligibility: LabelConsumptionEligibilityV1,
        frontier: LabelEvidenceFrontierV1,
    ) -> LabelConsumptionEligibilityReportV1 {
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label_id.into(),
            label_revision,
            measurement_event_id: event_id.into(),
            calibration_certificate_id: "cal-cert-001".into(),
            frontier,
            eligibility,
            downstream_consumption_allowed: eligibility
                == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: payload_digest.into(),
            measurement_sha256: digest('c'),
            qualified_corrected_value: 11.0,
            qualified_standard_uncertainty_upper_bound: Some(0.10),
            qualification_report_sha256: digest('d'),
            requalification_report_sha256: digest('e'),
            requalification_frontier_sha256: digest('f'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        report.report_sha256 = eligibility_report_digest(&report).unwrap();
        report
    }

    fn delivery(
        identity: (&str, &str, &str),
        revision: u64,
        status: LabelConsumptionEligibilityV1,
        frontier: LabelEvidenceFrontierV1,
        times: (i64, i64),
    ) -> StreamingLabelDeliveryV1 {
        let (sample_id, event_id, label_id) = identity;
        let (available_at, delivered_at) = times;
        let payload = digest('b');
        StreamingLabelDeliveryV1 {
            schema: STREAMING_DELIVERY_SCHEMA_V1.into(),
            sample_id: sample_id.into(),
            label_id: label_id.into(),
            label_revision: revision,
            measurement_event_id: event_id.into(),
            label_payload_sha256: payload.clone(),
            label_value: 11.0,
            label_available_at_unix_seconds: available_at,
            eligibility_report_available_at_unix_seconds: delivered_at.saturating_sub(1),
            delivered_at_unix_seconds: delivered_at,
            clock_domain_id: "fab-clock".into(),
            idempotency_key: format!("{label_id}:{revision}"),
            eligibility_report: eligibility_report(
                label_id, event_id, &payload, revision, status, frontier,
            ),
        }
    }

    #[test]
    fn prediction_is_read_only_and_freezes_features_and_state_digest() {
        let mut engine = engine();
        let before = adapter_state_digest(&engine.adapter_state).unwrap();
        let receipt = predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction(
                "sample-1",
                "event-1",
                100,
                102,
                Some("eqp-1/ch-1"),
                vec![1.0, 2.0],
            ),
        )
        .unwrap();

        assert_eq!(receipt.adapter_state_before_sha256, before);
        assert_eq!(receipt.adapter_state_after_sha256, before);
        assert_eq!(adapter_state_digest(&engine.adapter_state).unwrap(), before);
        assert_eq!(receipt.selected_feature_vector, vec![1.0, 2.0]);
        assert_eq!(receipt.context_key.as_deref(), Some("eqp-1/ch-1"));
        assert_eq!(receipt.prediction_value, 10.0);
    }

    #[test]
    fn qualified_zero_delay_label_updates_adapter_once() {
        let mut engine = engine();
        let prediction = predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction("sample-1", "event-1", 100, 102, None, vec![1.0, 2.0]),
        )
        .unwrap();
        let before = adapter_state_digest(&engine.adapter_state).unwrap();
        let report = deliver_qualified_stream_label_v1(
            &mut engine,
            &delivery(
                ("sample-1", "event-1", "label-1"),
                1,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                LabelEvidenceFrontierV1::CurrentAtK1,
                (102, 102),
            ),
        )
        .unwrap();

        assert_eq!(
            report.disposition,
            StreamingUpdateDispositionV1::Updated,
            "rejection reasons: {:?}",
            report.reasons
        );
        assert!(report.adapter_state_mutated);
        assert_ne!(report.adapter_state_after_sha256, before);
        assert_eq!(
            report.prediction_receipt_sha256,
            Some(prediction.receipt_sha256)
        );
        assert_eq!(engine.adapter_state.accepted_update_count, 1);
        assert!(!report.retraining_executed);
        assert!(!report.equipment_control_authority);
    }

    #[test]
    fn delayed_label_uses_the_original_prediction_record_after_intervening_update() {
        let mut engine = engine();
        let old_prediction = predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction("old-sample", "old-event", 100, 102, None, vec![1.0, 0.0]),
        )
        .unwrap();
        let new_prediction = predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction("new-sample", "new-event", 103, 104, None, vec![0.0, 1.0]),
        )
        .unwrap();

        let newer_delivery = delivery(
            ("new-sample", "new-event", "new-label"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (104, 104),
        );
        assert_eq!(
            deliver_qualified_stream_label_v1(&mut engine, &newer_delivery)
                .unwrap()
                .disposition,
            StreamingUpdateDispositionV1::Updated
        );
        let weights_after_newer = engine.adapter_state.feature_weights.clone();
        let old_delivery = delivery(
            ("old-sample", "old-event", "old-label"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 112),
        );
        let old_update = deliver_qualified_stream_label_v1(&mut engine, &old_delivery).unwrap();

        assert_eq!(
            old_update.disposition,
            StreamingUpdateDispositionV1::Updated,
            "rejection reasons: {:?}",
            old_update.reasons
        );
        assert_eq!(
            old_update.prediction_receipt_sha256,
            Some(old_prediction.receipt_sha256)
        );
        assert_ne!(engine.adapter_state.feature_weights, weights_after_newer);
        assert_eq!(new_prediction.base_prediction, 10.0);
        assert_eq!(old_prediction.base_prediction, 10.0);
        assert_eq!(engine.adapter_state.accepted_update_count, 2);
    }

    #[test]
    fn unknown_label_is_held_without_state_mutation_and_can_be_requalified_later() {
        let mut engine = engine();
        predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction("sample-1", "event-1", 100, 102, None, vec![1.0, 1.0]),
        )
        .unwrap();
        let before = adapter_state_digest(&engine.adapter_state).unwrap();
        let unknown = delivery(
            ("sample-1", "event-1", "label-1"),
            1,
            LabelConsumptionEligibilityV1::Unknown,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 111),
        );
        let held = deliver_qualified_stream_label_v1(&mut engine, &unknown).unwrap();
        assert_eq!(held.disposition, StreamingUpdateDispositionV1::HeldUnknown);
        assert!(!held.adapter_state_mutated);
        assert_eq!(adapter_state_digest(&engine.adapter_state).unwrap(), before);
        assert!(engine.pending_predictions.contains_key("sample-1"));

        let qualified = delivery(
            ("sample-1", "event-1", "label-1"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 112),
        );
        let accepted = deliver_qualified_stream_label_v1(&mut engine, &qualified).unwrap();
        assert_eq!(accepted.disposition, StreamingUpdateDispositionV1::Updated);
    }

    #[test]
    fn blocked_label_never_updates_adapter_state() {
        let mut engine = engine();
        predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction("sample-1", "event-1", 100, 102, None, vec![1.0, 1.0]),
        )
        .unwrap();
        let before = adapter_state_digest(&engine.adapter_state).unwrap();
        let blocked = delivery(
            ("sample-1", "event-1", "label-1"),
            1,
            LabelConsumptionEligibilityV1::Blocked,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 112),
        );
        let receipt = deliver_qualified_stream_label_v1(&mut engine, &blocked).unwrap();
        assert_eq!(receipt.disposition, StreamingUpdateDispositionV1::Blocked);
        assert!(!receipt.adapter_state_mutated);
        assert_eq!(adapter_state_digest(&engine.adapter_state).unwrap(), before);
    }

    #[test]
    fn accepted_label_revision_is_idempotent() {
        let mut engine = engine();
        predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction("sample-1", "event-1", 100, 102, None, vec![1.0, 1.0]),
        )
        .unwrap();
        let accepted = delivery(
            ("sample-1", "event-1", "label-1"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 112),
        );
        assert_eq!(
            deliver_qualified_stream_label_v1(&mut engine, &accepted)
                .unwrap()
                .disposition,
            StreamingUpdateDispositionV1::Updated
        );
        let after = adapter_state_digest(&engine.adapter_state).unwrap();
        let duplicate = deliver_qualified_stream_label_v1(&mut engine, &accepted).unwrap();
        assert_eq!(
            duplicate.disposition,
            StreamingUpdateDispositionV1::Duplicate
        );
        assert!(!duplicate.adapter_state_mutated);
        assert_eq!(adapter_state_digest(&engine.adapter_state).unwrap(), after);
    }

    #[test]
    fn label_id_event_value_and_payload_mismatches_are_rejected_without_update() {
        let mut cases = Vec::new();
        cases.push(delivery(
            ("sample-1", "other-event", "label-1"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 112),
        ));
        let mut wrong_label = delivery(
            ("sample-1", "event-1", "other-label"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 112),
        );
        wrong_label.eligibility_report.label_id = "label-1".into();
        wrong_label.eligibility_report.report_sha256 =
            eligibility_report_digest(&wrong_label.eligibility_report).unwrap();
        cases.push(wrong_label);
        let mut wrong_value = delivery(
            ("sample-1", "event-1", "label-1"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 112),
        );
        wrong_value.label_value = 12.0;
        cases.push(wrong_value);
        let mut wrong_payload = delivery(
            ("sample-1", "event-1", "label-1"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 112),
        );
        wrong_payload.label_payload_sha256 = digest('d');
        cases.push(wrong_payload);

        for (index, candidate) in cases.into_iter().enumerate() {
            let mut engine = engine();
            predict_and_register_stream_sample_v1(
                &mut engine,
                &prediction("sample-1", "event-1", 100, 102, None, vec![1.0, 1.0]),
            )
            .unwrap();
            let before = adapter_state_digest(&engine.adapter_state).unwrap();
            let receipt = deliver_qualified_stream_label_v1(&mut engine, &candidate).unwrap();
            assert_eq!(
                receipt.disposition,
                StreamingUpdateDispositionV1::Rejected,
                "case {index}"
            );
            assert!(!receipt.adapter_state_mutated);
            assert_eq!(adapter_state_digest(&engine.adapter_state).unwrap(), before);
        }
    }

    #[test]
    fn eligibility_report_must_be_available_before_label_delivery() {
        let mut engine = engine();
        predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction("sample-1", "event-1", 100, 102, None, vec![1.0, 1.0]),
        )
        .unwrap();
        let before = adapter_state_digest(&engine.adapter_state).unwrap();
        let mut candidate = delivery(
            ("sample-1", "event-1", "label-1"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 112),
        );
        candidate.eligibility_report_available_at_unix_seconds = 113;
        let receipt = deliver_qualified_stream_label_v1(&mut engine, &candidate).unwrap();
        assert_eq!(receipt.disposition, StreamingUpdateDispositionV1::Rejected);
        assert!(
            receipt
                .reasons
                .contains(&StreamingUpdateReasonV1::EligibilityReportUnavailableAtDelivery)
        );
        assert_eq!(adapter_state_digest(&engine.adapter_state).unwrap(), before);
    }

    #[test]
    fn eligibility_report_cannot_predate_label_availability() {
        let mut engine = engine();
        predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction("sample-1", "event-1", 100, 102, None, vec![1.0, 1.0]),
        )
        .unwrap();
        let before = adapter_state_digest(&engine.adapter_state).unwrap();
        let mut candidate = delivery(
            ("sample-1", "event-1", "label-1"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 112),
        );
        candidate.eligibility_report_available_at_unix_seconds = 109;
        let receipt = deliver_qualified_stream_label_v1(&mut engine, &candidate).unwrap();
        assert_eq!(receipt.disposition, StreamingUpdateDispositionV1::Rejected);
        assert!(
            receipt
                .reasons
                .contains(&StreamingUpdateReasonV1::EligibilityReportBeforeLabelAvailability)
        );
        assert_eq!(adapter_state_digest(&engine.adapter_state).unwrap(), before);
    }

    #[test]
    fn prediction_sequence_overflow_does_not_register_or_consume_sample() {
        let mut engine = engine();
        engine.next_prediction_sequence = u64::MAX;
        let before = adapter_state_digest(&engine.adapter_state).unwrap();
        assert!(
            predict_and_register_stream_sample_v1(
                &mut engine,
                &prediction("sample-1", "event-1", 100, 102, None, vec![1.0, 1.0]),
            )
            .is_err()
        );
        assert!(engine.pending_predictions.is_empty());
        assert!(engine.seen_measurement_event_ids.is_empty());
        assert_eq!(adapter_state_digest(&engine.adapter_state).unwrap(), before);
    }

    #[test]
    fn failed_context_support_increment_is_atomic() {
        let mut engine = engine();
        predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction(
                "sample-1",
                "event-1",
                100,
                102,
                Some("eqp-1/ch-1"),
                vec![1.0, 1.0],
            ),
        )
        .unwrap();
        engine.adapter_state.contexts.insert(
            "eqp-1/ch-1".into(),
            StreamingContextStateV1 {
                residual_ema: 0.25,
                support_count: u64::MAX,
            },
        );
        let before = adapter_state_digest(&engine.adapter_state).unwrap();
        assert!(
            deliver_qualified_stream_label_v1(
                &mut engine,
                &delivery(
                    ("sample-1", "event-1", "label-1"),
                    1,
                    LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                    LabelEvidenceFrontierV1::CurrentAtK1,
                    (110, 112),
                ),
            )
            .is_err()
        );
        assert_eq!(adapter_state_digest(&engine.adapter_state).unwrap(), before);
        assert!(engine.pending_predictions.contains_key("sample-1"));
        assert!(engine.accepted_label_revisions.is_empty());
    }

    #[test]
    fn delivery_revision_must_match_revision_attested_by_eligibility_report() {
        let mut engine = engine();
        predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction("sample-1", "event-1", 100, 102, None, vec![1.0, 1.0]),
        )
        .unwrap();
        let before = adapter_state_digest(&engine.adapter_state).unwrap();
        let mut wrong_revision = delivery(
            ("sample-1", "event-1", "label-1"),
            2,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 112),
        );
        wrong_revision.eligibility_report.label_revision = 1;
        wrong_revision.eligibility_report.report_sha256 =
            eligibility_report_digest(&wrong_revision.eligibility_report).unwrap();

        let receipt = deliver_qualified_stream_label_v1(&mut engine, &wrong_revision).unwrap();
        assert_eq!(receipt.disposition, StreamingUpdateDispositionV1::Rejected);
        assert!(
            receipt
                .reasons
                .contains(&StreamingUpdateReasonV1::LabelRevisionMismatch)
        );
        assert_eq!(adapter_state_digest(&engine.adapter_state).unwrap(), before);
    }

    #[test]
    fn availability_and_clock_failures_never_update() {
        let mut engine = engine();
        predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction("sample-1", "event-1", 100, 102, None, vec![1.0, 1.0]),
        )
        .unwrap();
        let before = adapter_state_digest(&engine.adapter_state).unwrap();

        let too_early = delivery(
            ("sample-1", "event-1", "label-1"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 109),
        );
        let receipt = deliver_qualified_stream_label_v1(&mut engine, &too_early).unwrap();
        assert_eq!(receipt.disposition, StreamingUpdateDispositionV1::Rejected);
        assert!(
            receipt
                .reasons
                .contains(&StreamingUpdateReasonV1::DeliveryBeforeLabelAvailability)
        );

        let mut wrong_clock = delivery(
            ("sample-1", "event-1", "label-1"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 112),
        );
        wrong_clock.clock_domain_id = "other-clock".into();
        let receipt = deliver_qualified_stream_label_v1(&mut engine, &wrong_clock).unwrap();
        assert_eq!(receipt.disposition, StreamingUpdateDispositionV1::Rejected);
        assert!(
            receipt
                .reasons
                .contains(&StreamingUpdateReasonV1::ClockDomainMismatch)
        );
        assert_eq!(adapter_state_digest(&engine.adapter_state).unwrap(), before);
    }

    #[test]
    fn stale_or_non_k1_report_is_not_consumed() {
        let mut engine = engine();
        predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction("sample-1", "event-1", 100, 102, None, vec![1.0, 1.0]),
        )
        .unwrap();
        let before = adapter_state_digest(&engine.adapter_state).unwrap();
        let stale = delivery(
            ("sample-1", "event-1", "label-1"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::AsKnownAtK0,
            (110, 112),
        );
        let receipt = deliver_qualified_stream_label_v1(&mut engine, &stale).unwrap();
        assert_eq!(receipt.disposition, StreamingUpdateDispositionV1::Rejected);
        assert!(
            receipt
                .reasons
                .contains(&StreamingUpdateReasonV1::LabelNotCurrentK1)
        );
        assert_eq!(adapter_state_digest(&engine.adapter_state).unwrap(), before);

        let mut tampered = delivery(
            ("sample-1", "event-1", "label-2"),
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
            (110, 112),
        );
        tampered.eligibility_report.report_sha256 = digest('0');
        let receipt = deliver_qualified_stream_label_v1(&mut engine, &tampered).unwrap();
        assert_eq!(receipt.disposition, StreamingUpdateDispositionV1::Rejected);
        assert!(
            receipt
                .reasons
                .contains(&StreamingUpdateReasonV1::EligibilityReportDigestMismatch)
        );
        assert_eq!(adapter_state_digest(&engine.adapter_state).unwrap(), before);
    }

    #[test]
    fn zero_feature_vector_is_safe_and_does_not_change_feature_weights() {
        let mut engine = engine();
        predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction("sample-1", "event-1", 100, 102, None, vec![0.0, 0.0]),
        )
        .unwrap();
        let before_weights = engine.adapter_state.feature_weights.clone();
        let receipt = deliver_qualified_stream_label_v1(
            &mut engine,
            &delivery(
                ("sample-1", "event-1", "label-1"),
                1,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                LabelEvidenceFrontierV1::CurrentAtK1,
                (110, 112),
            ),
        )
        .unwrap();
        assert_eq!(receipt.disposition, StreamingUpdateDispositionV1::Updated);
        assert_eq!(engine.adapter_state.feature_weights, before_weights);
    }

    #[test]
    fn context_correction_is_local_and_shrunk_toward_global_support() {
        let mut engine = engine();
        predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction(
                "sample-1",
                "event-1",
                100,
                102,
                Some("eqp-1/ch-1"),
                vec![1.0, 0.0],
            ),
        )
        .unwrap();
        deliver_qualified_stream_label_v1(
            &mut engine,
            &delivery(
                ("sample-1", "event-1", "label-1"),
                1,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                LabelEvidenceFrontierV1::CurrentAtK1,
                (110, 112),
            ),
        )
        .unwrap();

        let unseen_context = predict_and_register_stream_sample_v1(
            &mut engine,
            &prediction(
                "sample-2",
                "event-2",
                113,
                115,
                Some("eqp-2/ch-9"),
                vec![1.0, 0.0],
            ),
        )
        .unwrap();
        assert_eq!(
            unseen_context.context_correction,
            engine.adapter_state.global_context_residual_ema
        );
        assert!(engine.adapter_state.contexts.contains_key("eqp-1/ch-1"));
        assert!(!engine.adapter_state.contexts.contains_key("eqp-2/ch-9"));
    }

    #[test]
    fn wrong_feature_schema_and_duplicate_measurement_events_reject_prediction() {
        let mut engine = engine();
        let mut wrong_schema = prediction("sample-1", "event-1", 100, 102, None, vec![1.0, 1.0]);
        wrong_schema.feature_schema_sha256 = digest('f');
        assert!(predict_and_register_stream_sample_v1(&mut engine, &wrong_schema).is_err());

        let good = prediction("sample-1", "event-1", 100, 102, None, vec![1.0, 1.0]);
        predict_and_register_stream_sample_v1(&mut engine, &good).unwrap();
        let duplicate = prediction("sample-2", "event-1", 103, 105, None, vec![1.0, 1.0]);
        assert!(predict_and_register_stream_sample_v1(&mut engine, &duplicate).is_err());
    }

    #[test]
    fn same_fixed_arrival_schedule_replays_identically() {
        fn run() -> String {
            let mut engine = engine();
            for index in 0..2 {
                let sample_id = format!("sample-{index}");
                let event_id = format!("event-{index}");
                predict_and_register_stream_sample_v1(
                    &mut engine,
                    &prediction(
                        &sample_id,
                        &event_id,
                        100 + index * 3,
                        102 + index * 3,
                        Some("eqp-1/ch-1"),
                        vec![1.0 + index as f64, 2.0],
                    ),
                )
                .unwrap();
            }
            for index in 0..2 {
                let sample_id = format!("sample-{index}");
                let event_id = format!("event-{index}");
                let label_id = format!("label-{index}");
                deliver_qualified_stream_label_v1(
                    &mut engine,
                    &delivery(
                        (&sample_id, &event_id, &label_id),
                        1,
                        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                        LabelEvidenceFrontierV1::CurrentAtK1,
                        (110 + index * 3, 112 + index * 3),
                    ),
                )
                .unwrap();
            }
            adapter_state_digest(&engine.adapter_state).unwrap()
        }
        assert_eq!(run(), run());
    }
}
