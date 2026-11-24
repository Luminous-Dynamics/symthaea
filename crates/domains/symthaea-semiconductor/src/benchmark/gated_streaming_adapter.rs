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
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
    LabelEvidenceFrontierV1,
};

pub const GATED_STREAMING_ADAPTER_SCHEMA_V1: &str =
    "luminous.semi-bench.gated-streaming-adapter.v1";
pub const STREAMING_ADAPTER_POLICY_SCHEMA_V1: &str =
    "luminous.semi-bench.streaming-adapter-policy.v1";
pub const STREAMING_PREDICTION_SCHEMA_V1: &str =
    "luminous.semi-bench.streaming-prediction.v1";
pub const STREAMING_DELIVERY_SCHEMA_V1: &str =
    "luminous.semi-bench.streaming-label-delivery.v1";
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
    MeasurementEventMismatch,
    LabelPayloadDigestMismatch,
    EligibilityReportDigestMismatch,
    ReportNumericValueNonFinite,
    DeliveredValueMismatch,
    LabelRevisionInvalid,
    IdempotencyKeyMismatch,
    ClockDomainMismatch,
    DeliveryBeforeLabelAvailability,
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
    pub feature_schema_sha256: String,
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
    pub feature_schema_sha256: String,
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

fn json_sha256<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, GatedStreamingAdapterErrorV1> {
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

fn validate_engine(engine: &GatedStreamingAdapterEngineV1) -> Result<(), GatedStreamingAdapterErrorV1> {
    if engine.schema != GATED_STREAMING_ADAPTER_SCHEMA_V1 {
        return Err(GatedStreamingAdapterErrorV1(
            "streaming adapter engine schema is unsupported".into(),
        ));
    }
    require_text(&engine.engine_id, "engine.engine_id")?;
    require_text(&engine.clock_domain_id, "engine.clock_domain_id")?;
    if !valid_sha256(&engine.feature_schema_sha256) {
        return Err(GatedStreamingAdapterErrorV1(
            "engine feature-schema digest is malformed".into(),
        ));
    }
    validate_policy(&engine.policy)?;
    if engine.adapter_state.feature_weights.is_empty()
        || engine
            .adapter_state
            .feature_weights
            .iter()
            .any(|value| !value.is_finite())
        || !engine.adapter_state.global_context_residual_ema.is_finite()
        || !engine.adapter_state.global_temporal_residual_ema.is_finite()
        || engine.adapter_state.contexts.values().any(|context| {
            !context.residual_ema.is_finite()
        })
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
    feature_schema_sha256: &str,
    initial_feature_weights: Vec<f64>,
    policy: StreamingAdapterPolicyV1,
) -> Result<GatedStreamingAdapterEngineV1, GatedStreamingAdapterErrorV1> {
    require_text(engine_id, "engine_id")?;
    require_text(clock_domain_id, "clock_domain_id")?;
    if !valid_sha256(feature_schema_sha256) {
        return Err(GatedStreamingAdapterErrorV1(
            "feature_schema_sha256: malformed lowercase SHA-256".into(),
        ));
    }
    validate_policy(&policy)?;
    if initial_feature_weights.is_empty()
        || initial_feature_weights.iter().any(|weight| !weight.is_finite())
    {
        return Err(GatedStreamingAdapterErrorV1(
            "initial feature adapter weights must be non-empty and finite".into(),
        ));
    }
    Ok(GatedStreamingAdapterEngineV1 {
        schema: GATED_STREAMING_ADAPTER_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        clock_domain_id: clock_domain_id.into(),
        feature_schema_sha256: feature_schema_sha256.into(),
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

fn context_correction(
    engine: &GatedStreamingAdapterEngineV1,
    context_key: Option<&str>,
) -> f64 {
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
        || input
            .feature_vector
            .iter()
            .any(|value| !value.is_finite() || value.abs() > engine.policy.maximum_abs_feature_value)
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
        || engine.seen_measurement_event_ids.contains(&input.measurement_event_id)
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
    let mut receipt = StreamingPredictionReceiptV1 {
        schema: STREAMING_PREDICTION_RECEIPT_SCHEMA_V1.into(),
        sequence: engine.next_prediction_sequence,
        sample_id: input.sample_id.clone(),
        measurement_event_id: input.measurement_event_id.clone(),
        event_time_unix_seconds: input.event_time_unix_seconds,
        feature_available_at_unix_seconds: input.feature_available_at_unix_seconds,
        prediction_at_unix_seconds: input.prediction_at_unix_seconds,
        clock_domain_id: input.clock_domain_id.clone(),
        feature_schema_sha256: input.feature_schema_sha256.clone(),
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

    engine.seen_measurement_event_ids.insert(input.measurement_event_id.clone());
    engine.pending_predictions.insert(input.sample_id.clone(), receipt.clone());
    engine.next_prediction_sequence = engine
        .next_prediction_sequence
        .checked_add(1)
        .ok_or_else(|| GatedStreamingAdapterErrorV1("prediction sequence overflow".into()))?;
    Ok(receipt)
}

fn update_receipt(
    engine: &GatedStreamingAdapterEngineV1,
    delivery: &StreamingLabelDeliveryV1,
    prediction: Option<&StreamingPredictionReceiptV1>,
    disposition: StreamingUpdateDispositionV1,
    reasons: Vec<StreamingUpdateReasonV1>,
    before: &str,
    after: &str,
    update_sequence: Option<u64>,
    mutated: bool,
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
        label_payload_sha256: delivery.label_payload_sha256.clone(),
        adapter_state_before_sha256: before.into(),
        adapter_state_after_sha256: after.into(),
        update_sequence,
        adapter_state_mutated: mutated,
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
    require_text(&delivery.measurement_event_id, "delivery.measurement_event_id")?;
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

    if engine
        .accepted_label_revisions
        .contains(&format!("{}:{}", delivery.label_id, delivery.label_revision))
    {
        reasons.push(StreamingUpdateReasonV1::DuplicateAcceptedLabelRevision);
        return update_receipt(
            engine,
            delivery,
            prediction,
            StreamingUpdateDispositionV1::Duplicate,
            reasons,
            &before,
            &before,
            None,
            false,
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
    if !valid_sha256(&report.label_sha256)
        || report.label_sha256 != delivery.label_payload_sha256
    {
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
            &before,
            &before,
            None,
            false,
        );
    };

    if !reasons.is_empty() {
        let disposition = match report.eligibility {
            LabelConsumptionEligibilityV1::Unknown => StreamingUpdateDispositionV1::HeldUnknown,
            LabelConsumptionEligibilityV1::Blocked => StreamingUpdateDispositionV1::Blocked,
            _ => StreamingUpdateDispositionV1::Rejected,
        };
        return update_receipt(
            engine,
            delivery,
            Some(&prediction_receipt),
            disposition,
            reasons,
            &before,
            &before,
            None,
            false,
        );
    }

    let error = delivery.label_value - prediction_receipt.base_prediction;
    if !error.is_finite() || error.abs() > engine.policy.maximum_abs_prediction_error {
        return update_receipt(
            engine,
            delivery,
            Some(&prediction_receipt),
            StreamingUpdateDispositionV1::Rejected,
            vec![StreamingUpdateReasonV1::InvalidStreamingNumericInput],
            &before,
            &before,
            None,
            false,
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

    for (weight, feature) in engine
        .adapter_state
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
    }

    engine.adapter_state.global_temporal_residual_ema =
        (1.0 - engine.policy.temporal_ema_alpha)
            * engine.adapter_state.global_temporal_residual_ema
            + engine.policy.temporal_ema_alpha * error;

    let updated_feature_correction = engine
        .adapter_state
        .feature_weights
        .iter()
        .zip(&prediction_receipt.selected_feature_vector)
        .map(|(weight, feature)| weight * feature)
        .sum::<f64>();
    // Like the GRAPE context residual, remove the frozen global prediction and
    // newly updated feature correction, but not the local context correction.
    let context_residual =
        delivery.label_value - (prediction_receipt.global_baseline_prediction + updated_feature_correction);
    engine.adapter_state.global_context_residual_ema =
        (1.0 - engine.policy.global_context_ema_alpha)
            * engine.adapter_state.global_context_residual_ema
            + engine.policy.global_context_ema_alpha * context_residual;

    if let Some(context_key) = prediction_receipt.context_key.clone() {
        let entry = engine
            .adapter_state
            .contexts
            .entry(context_key)
            .or_insert(StreamingContextStateV1 {
                residual_ema: engine.adapter_state.global_context_residual_ema,
                support_count: 0,
            });
        entry.residual_ema = (1.0 - engine.policy.local_context_ema_alpha) * entry.residual_ema
            + engine.policy.local_context_ema_alpha * context_residual;
        entry.support_count = entry
            .support_count
            .checked_add(1)
            .ok_or_else(|| GatedStreamingAdapterErrorV1("context support count overflow".into()))?;
    }

    engine.adapter_state.accepted_update_count = engine
        .adapter_state
        .accepted_update_count
        .checked_add(1)
        .ok_or_else(|| GatedStreamingAdapterErrorV1("accepted update count overflow".into()))?;
    let after = adapter_state_digest(&engine.adapter_state)?;
    engine.accepted_label_revisions.insert(format!(
        "{}:{}",
        delivery.label_id, delivery.label_revision
    ));
    engine.pending_predictions.remove(&delivery.sample_id);
    update_receipt(
        engine,
        delivery,
        Some(&prediction_receipt),
        StreamingUpdateDispositionV1::Updated,
        Vec::new(),
        &before,
        &after,
        Some(engine.adapter_state.accepted_update_count),
        true,
    )
}

