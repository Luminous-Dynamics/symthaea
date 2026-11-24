// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-gated adaptive conformal miscoverage control with delayed feedback.
//!
//! Prediction intervals are immutable once issued. Labels enter the controller
//! only after their exact current-K1 numeric eligibility report is replayed.
//! Fixed-horizon delayed ACI and irregular arrival-order ACI are distinct modes.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::{
    LABEL_CONSUMPTION_REPORT_SCHEMA_V1, LabelConsumptionEligibilityReportV1,
    LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};

pub const DELAYED_ACI_POLICY_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci-policy.v1";
pub const DELAYED_ACI_ENGINE_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci-engine.v1";
pub const ACI_PREDICTION_INPUT_SCHEMA_V1: &str = "luminous.semi-bench.aci-prediction-input.v1";
pub const ACI_PREDICTION_RECEIPT_SCHEMA_V1: &str = "luminous.semi-bench.aci-prediction-receipt.v1";
pub const ACI_LABEL_DELIVERY_SCHEMA_V1: &str = "luminous.semi-bench.aci-label-delivery.v1";
pub const ACI_UPDATE_RECEIPT_SCHEMA_V1: &str = "luminous.semi-bench.aci-update-receipt.v1";
pub const ACI_DELAY_MEMORY_DIAGNOSTIC_SCHEMA_V1: &str =
    "luminous.semi-bench.aci-delay-memory-diagnostic.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AciUpdateModeV1 {
    /// Delayed ACI recursion: alpha[t + horizon] = alpha[t] + gamma * (target - miss[t]).
    FixedHorizonDelayed { horizon_steps: u64 },
    /// Updates the current alpha state when a qualified label arrives.
    /// This supports irregular delays but is not the fixed-horizon recursion.
    ArrivalOrder,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AciUpdateDispositionV1 {
    Scheduled,
    UpdatedArrivalOrder,
    HeldUnknown,
    Blocked,
    Rejected,
    Duplicate,
    PendingPredictionMissing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AciUpdateReasonV1 {
    LabelNotCurrentK1,
    LabelNotEligible,
    DownstreamConsumptionNotAllowed,
    LabelIdentityMismatch,
    LabelRevisionMismatch,
    MeasurementEventMismatch,
    LabelPayloadDigestMismatch,
    EligibilityReportDigestMismatch,
    DeliveredValueNonFinite,
    DeliveredValueOutOfPolicy,
    DeliveredValueMismatch,
    PendingPredictionMissing,
    ClockDomainMismatch,
    LabelAvailableBeforeMeasurement,
    DeliveryBeforePrediction,
    DeliveryBeforeLabelAvailability,
    DeliveryDelayExceeded,
    ReportBeforeLabelAvailability,
    ReportUnavailableAtDelivery,
    DuplicateAcceptedLabelRevision,
    FixedHorizonSlotAlreadyPassed,
    FixedHorizonScheduleConflict,
    InvalidAciMode,
    InvalidPrediction,
    PredictionAlreadyRegistered,
    AlphaInputMismatch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualDependenceFamilyV1 {
    Ar1,
    Garch11,
    MarkovSwitching,
    Unspecified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelayMemoryRegimeV1 {
    DelayShorterThanMemory,
    DelayComparableToMemory,
    DelayLongerThanMemory,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub target_miscoverage: f64,
    pub learning_rate: f64,
    pub minimum_alpha: f64,
    pub maximum_alpha: f64,
    pub maximum_abs_target: f64,
    pub maximum_delivery_delay_seconds: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciPredictionInputV1 {
    pub schema: String,
    pub prediction_id: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub interval_policy_sha256: String,
    /// The alpha state used by the upstream interval constructor.
    pub alpha_used: f64,
    /// Bounds are frozen here; the controller never reconstructs an old interval.
    pub interval_lower: f64,
    pub interval_upper: f64,
    pub upstream_prediction_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciPredictionReceiptV1 {
    pub schema: String,
    pub sequence: u64,
    pub prediction_id: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub interval_policy_sha256: String,
    pub alpha_at_prediction: f64,
    pub interval_lower: f64,
    pub interval_upper: f64,
    pub upstream_prediction_receipt_sha256: String,
    pub calibrator_state_before_sha256: String,
    pub calibrator_state_after_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciLabelDeliveryV1 {
    pub schema: String,
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub measurement_event_id: String,
    pub label_payload_sha256: String,
    pub target_value: f64,
    pub label_available_at_unix_seconds: i64,
    pub eligibility_report_available_at_unix_seconds: i64,
    pub delivered_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub idempotency_key: String,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciUpdateReceiptV1 {
    pub schema: String,
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub prediction_receipt_sha256: Option<String>,
    pub eligibility_report_sha256: Option<String>,
    pub label_payload_sha256: String,
    pub update_mode: AciUpdateModeV1,
    pub disposition: AciUpdateDispositionV1,
    pub reason: Option<AciUpdateReasonV1>,
    pub interval_missed: Option<bool>,
    pub alpha_before: f64,
    /// Arrival-order: alpha after update. Fixed-horizon: alpha scheduled for a future sequence.
    pub alpha_after_or_scheduled: f64,
    pub current_alpha_after: f64,
    pub scheduled_sequence: Option<u64>,
    pub effective_delay_seconds: Option<i64>,
    pub state_before_sha256: String,
    pub state_after_sha256: String,
    pub calibrator_state_mutated: bool,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciCalibratorStateV1 {
    pub alpha_state: f64,
    pub next_prediction_sequence: u64,
    pub scheduled_alpha_by_sequence: BTreeMap<u64, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub clock_domain_id: String,
    pub mode: AciUpdateModeV1,
    pub policy: DelayedAciPolicyV1,
    pub policy_sha256: String,
    pub state: AciCalibratorStateV1,
    pub pending_predictions: BTreeMap<String, AciPredictionReceiptV1>,
    pub accepted_label_revisions: BTreeSet<String>,
    pub seen_measurement_event_ids: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciDelayMemoryDiagnosticV1 {
    pub schema: String,
    pub residual_dependence_family: ResidualDependenceFamilyV1,
    pub delay_steps: u64,
    pub memory_steps: Option<f64>,
    pub delay_to_memory_ratio: Option<f64>,
    pub regime: DelayMemoryRegimeV1,
    pub interpretation: String,
    pub diagnostic_sha256: String,
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

fn state_digest(state: &AciCalibratorStateV1) -> Result<String, DelayedAciErrorV1> {
    json_sha256(state)
}

fn policy_digest(policy: &DelayedAciPolicyV1) -> Result<String, DelayedAciErrorV1> {
    json_sha256(policy)
}

fn prediction_receipt_digest(
    receipt: &AciPredictionReceiptV1,
) -> Result<String, DelayedAciErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn update_receipt_digest(receipt: &AciUpdateReceiptV1) -> Result<String, DelayedAciErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn eligibility_report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, DelayedAciErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn validate_policy(policy: &DelayedAciPolicyV1) -> Result<(), DelayedAciErrorV1> {
    if policy.schema != DELAYED_ACI_POLICY_SCHEMA_V1 {
        return Err(DelayedAciErrorV1("ACI policy schema is unsupported".into()));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if !policy.target_miscoverage.is_finite()
        || !(0.0..1.0).contains(&policy.target_miscoverage)
        || !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || !policy.minimum_alpha.is_finite()
        || !policy.maximum_alpha.is_finite()
        || policy.minimum_alpha <= 0.0
        || policy.maximum_alpha >= 1.0
        || policy.minimum_alpha >= policy.maximum_alpha
        || policy.target_miscoverage < policy.minimum_alpha
        || policy.target_miscoverage > policy.maximum_alpha
        || !policy.maximum_abs_target.is_finite()
        || policy.maximum_abs_target <= 0.0
        || policy.maximum_delivery_delay_seconds < 0
    {
        return Err(DelayedAciErrorV1(
            "ACI policy contains invalid parameters".into(),
        ));
    }
    Ok(())
}

fn validate_engine(engine: &DelayedAciEngineV1) -> Result<(), DelayedAciErrorV1> {
    if engine.schema != DELAYED_ACI_ENGINE_SCHEMA_V1 {
        return Err(DelayedAciErrorV1("ACI engine schema is unsupported".into()));
    }
    validate_policy(&engine.policy)?;
    if policy_digest(&engine.policy)? != engine.policy_sha256
        || !engine.state.alpha_state.is_finite()
        || engine.state.alpha_state < engine.policy.minimum_alpha
        || engine.state.alpha_state > engine.policy.maximum_alpha
        || engine
            .state
            .scheduled_alpha_by_sequence
            .values()
            .any(|alpha| {
                !alpha.is_finite()
                    || *alpha < engine.policy.minimum_alpha
                    || *alpha > engine.policy.maximum_alpha
            })
    {
        return Err(DelayedAciErrorV1(
            "ACI engine or scheduled state is invalid".into(),
        ));
    }
    if let AciUpdateModeV1::FixedHorizonDelayed { horizon_steps: 0 } = engine.mode {
        return Err(DelayedAciErrorV1(
            "fixed ACI horizon must be at least one step".into(),
        ));
    }
    Ok(())
}

/// Returns the alpha that must be used to construct the next prediction interval.
/// A scheduled fixed-delay update is visible to the interval constructor exactly
/// at its effective sequence, before that prediction is issued.
pub fn current_aci_alpha_v1(engine: &DelayedAciEngineV1) -> Result<f64, DelayedAciErrorV1> {
    validate_engine(engine)?;
    Ok(engine
        .state
        .scheduled_alpha_by_sequence
        .get(&engine.state.next_prediction_sequence)
        .copied()
        .unwrap_or(engine.state.alpha_state))
}

pub fn create_delayed_aci_engine_v1(
    engine_id: &str,
    clock_domain_id: &str,
    mode: AciUpdateModeV1,
    policy: DelayedAciPolicyV1,
) -> Result<DelayedAciEngineV1, DelayedAciErrorV1> {
    require_text(engine_id, "engine_id")?;
    require_text(clock_domain_id, "clock_domain_id")?;
    validate_policy(&policy)?;
    if matches!(
        mode,
        AciUpdateModeV1::FixedHorizonDelayed { horizon_steps: 0 }
    ) {
        return Err(DelayedAciErrorV1(
            "fixed ACI horizon must be at least one step".into(),
        ));
    }
    let initial_alpha = policy.target_miscoverage;
    let policy_sha256 = policy_digest(&policy)?;
    Ok(DelayedAciEngineV1 {
        schema: DELAYED_ACI_ENGINE_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        clock_domain_id: clock_domain_id.into(),
        mode,
        policy,
        policy_sha256,
        state: AciCalibratorStateV1 {
            alpha_state: initial_alpha,
            next_prediction_sequence: 0,
            scheduled_alpha_by_sequence: BTreeMap::new(),
        },
        pending_predictions: BTreeMap::new(),
        accepted_label_revisions: BTreeSet::new(),
        seen_measurement_event_ids: BTreeSet::new(),
    })
}

/// Registers one immutable interval after the caller has constructed it using
/// current_aci_alpha_v1. The ACI engine never recomputes a historical interval.
pub fn register_aci_prediction_v1(
    engine: &mut DelayedAciEngineV1,
    input: &AciPredictionInputV1,
) -> Result<AciPredictionReceiptV1, DelayedAciErrorV1> {
    validate_engine(engine)?;
    if input.schema != ACI_PREDICTION_INPUT_SCHEMA_V1 {
        return Err(DelayedAciErrorV1(
            "ACI prediction input schema is unsupported".into(),
        ));
    }
    require_text(&input.prediction_id, "prediction_id")?;
    require_text(&input.sample_id, "sample_id")?;
    require_text(&input.measurement_event_id, "measurement_event_id")?;
    if input.clock_domain_id != engine.clock_domain_id
        || input.event_time_unix_seconds > input.prediction_at_unix_seconds
        || !valid_sha256(&input.global_model_artifact_sha256)
        || !valid_sha256(&input.interval_policy_sha256)
        || !valid_sha256(&input.upstream_prediction_receipt_sha256)
        || !input.alpha_used.is_finite()
        || !input.interval_lower.is_finite()
        || !input.interval_upper.is_finite()
        || input.interval_lower > input.interval_upper
        || input.interval_lower.abs() > engine.policy.maximum_abs_target
        || input.interval_upper.abs() > engine.policy.maximum_abs_target
    {
        return Err(DelayedAciErrorV1(
            "ACI prediction identity, time, digest or bounds invalid".into(),
        ));
    }

    let sequence = engine.state.next_prediction_sequence;
    let next_sequence = sequence
        .checked_add(1)
        .ok_or_else(|| DelayedAciErrorV1("ACI prediction sequence overflow".into()))?;
    let effective_alpha = current_aci_alpha_v1(engine)?;
    if input.alpha_used != effective_alpha {
        return Err(DelayedAciErrorV1(
            "interval alpha does not equal the current scheduled ACI alpha".into(),
        ));
    }
    if engine.pending_predictions.contains_key(&input.sample_id)
        || engine
            .seen_measurement_event_ids
            .contains(&input.measurement_event_id)
    {
        return Err(DelayedAciErrorV1(
            "duplicate ACI sample or measurement event".into(),
        ));
    }

    let before = state_digest(&engine.state)?;
    let mut next_state = engine.state.clone();
    next_state.alpha_state = effective_alpha;
    next_state.scheduled_alpha_by_sequence.remove(&sequence);
    next_state.next_prediction_sequence = next_sequence;
    let after = state_digest(&next_state)?;

    let mut receipt = AciPredictionReceiptV1 {
        schema: ACI_PREDICTION_RECEIPT_SCHEMA_V1.into(),
        sequence,
        prediction_id: input.prediction_id.clone(),
        sample_id: input.sample_id.clone(),
        measurement_event_id: input.measurement_event_id.clone(),
        event_time_unix_seconds: input.event_time_unix_seconds,
        prediction_at_unix_seconds: input.prediction_at_unix_seconds,
        clock_domain_id: input.clock_domain_id.clone(),
        global_model_artifact_sha256: input.global_model_artifact_sha256.clone(),
        interval_policy_sha256: input.interval_policy_sha256.clone(),
        alpha_at_prediction: effective_alpha,
        interval_lower: input.interval_lower,
        interval_upper: input.interval_upper,
        upstream_prediction_receipt_sha256: input.upstream_prediction_receipt_sha256.clone(),
        calibrator_state_before_sha256: before,
        calibrator_state_after_sha256: after,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = prediction_receipt_digest(&receipt)?;

    engine.state = next_state;
    engine
        .seen_measurement_event_ids
        .insert(input.measurement_event_id.clone());
    engine
        .pending_predictions
        .insert(input.sample_id.clone(), receipt.clone());
    Ok(receipt)
}

#[derive(Clone, Debug)]
struct AciUpdateOutcomeV1 {
    disposition: AciUpdateDispositionV1,
    reason: Option<AciUpdateReasonV1>,
    miss: Option<bool>,
    alpha_before: f64,
    alpha_after_or_scheduled: f64,
    current_alpha_after: f64,
    scheduled_sequence: Option<u64>,
    state_before_sha256: String,
    state_after_sha256: String,
    mutated: bool,
}

fn make_update_receipt(
    engine: &DelayedAciEngineV1,
    delivery: &AciLabelDeliveryV1,
    prediction: Option<&AciPredictionReceiptV1>,
    outcome: AciUpdateOutcomeV1,
) -> Result<AciUpdateReceiptV1, DelayedAciErrorV1> {
    let mut receipt = AciUpdateReceiptV1 {
        schema: ACI_UPDATE_RECEIPT_SCHEMA_V1.into(),
        sample_id: delivery.sample_id.clone(),
        label_id: delivery.label_id.clone(),
        label_revision: delivery.label_revision,
        prediction_receipt_sha256: prediction.map(|value| value.receipt_sha256.clone()),
        eligibility_report_sha256: Some(delivery.eligibility_report.report_sha256.clone()),
        label_payload_sha256: delivery.label_payload_sha256.clone(),
        update_mode: engine.mode,
        disposition: outcome.disposition,
        reason: outcome.reason,
        interval_missed: outcome.miss,
        alpha_before: outcome.alpha_before,
        alpha_after_or_scheduled: outcome.alpha_after_or_scheduled,
        current_alpha_after: outcome.current_alpha_after,
        scheduled_sequence: outcome.scheduled_sequence,
        effective_delay_seconds: prediction
            .map(|value| delivery.delivered_at_unix_seconds - value.prediction_at_unix_seconds),
        state_before_sha256: outcome.state_before_sha256,
        state_after_sha256: outcome.state_after_sha256,
        calibrator_state_mutated: outcome.mutated,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = update_receipt_digest(&receipt)?;
    Ok(receipt)
}

/// Delivers one qualified label. In fixed-horizon mode the alpha update is
/// scheduled at prediction_sequence + horizon_steps; feedback arriving after
/// that sequence has already been issued is rejected instead of being applied
/// as an arrival-order update. In ArrivalOrder mode it updates current alpha.
pub fn deliver_aci_label_v1(
    engine: &mut DelayedAciEngineV1,
    delivery: &AciLabelDeliveryV1,
) -> Result<AciUpdateReceiptV1, DelayedAciErrorV1> {
    validate_engine(engine)?;
    if delivery.schema != ACI_LABEL_DELIVERY_SCHEMA_V1 {
        return Err(DelayedAciErrorV1(
            "ACI label delivery schema is unsupported".into(),
        ));
    }
    require_text(&delivery.sample_id, "delivery.sample_id")?;
    require_text(&delivery.label_id, "delivery.label_id")?;
    require_text(
        &delivery.measurement_event_id,
        "delivery.measurement_event_id",
    )?;
    require_text(&delivery.idempotency_key, "delivery.idempotency_key")?;
    let report = &delivery.eligibility_report;
    let prediction = engine.pending_predictions.get(&delivery.sample_id);
    let before = state_digest(&engine.state)?;
    let revision_key = format!("{}:{}", delivery.label_id, delivery.label_revision);

    if engine.accepted_label_revisions.contains(&revision_key) {
        return make_update_receipt(
            engine,
            delivery,
            prediction,
            AciUpdateOutcomeV1 {
                disposition: AciUpdateDispositionV1::Duplicate,
                reason: Some(AciUpdateReasonV1::DuplicateAcceptedLabelRevision),
                miss: None,
                alpha_before: engine.state.alpha_state,
                alpha_after_or_scheduled: engine.state.alpha_state,
                current_alpha_after: engine.state.alpha_state,
                scheduled_sequence: None,
                state_before_sha256: before.clone(),
                state_after_sha256: before.clone(),
                mutated: false,
            },
        );
    }

    let Some(prediction_receipt) = prediction.cloned() else {
        return make_update_receipt(
            engine,
            delivery,
            None,
            AciUpdateOutcomeV1 {
                disposition: AciUpdateDispositionV1::PendingPredictionMissing,
                reason: Some(AciUpdateReasonV1::PendingPredictionMissing),
                miss: None,
                alpha_before: engine.state.alpha_state,
                alpha_after_or_scheduled: engine.state.alpha_state,
                current_alpha_after: engine.state.alpha_state,
                scheduled_sequence: None,
                state_before_sha256: before.clone(),
                state_after_sha256: before.to_string(),
                mutated: false,
            },
        );
    };

    let mut reason = None;
    if delivery.label_revision == 0
        || !valid_sha256(&delivery.label_payload_sha256)
        || report.schema != LABEL_CONSUMPTION_REPORT_SCHEMA_V1
        || eligibility_report_digest(report)? != report.report_sha256
        || report.label_sha256 != delivery.label_payload_sha256
    {
        reason = Some(AciUpdateReasonV1::EligibilityReportDigestMismatch);
    } else if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        reason = Some(AciUpdateReasonV1::LabelNotCurrentK1);
    } else if report.eligibility == LabelConsumptionEligibilityV1::Unknown {
        return make_update_receipt(
            engine,
            delivery,
            Some(&prediction_receipt),
            AciUpdateOutcomeV1 {
                disposition: AciUpdateDispositionV1::HeldUnknown,
                reason: Some(AciUpdateReasonV1::LabelNotEligible),
                miss: None,
                alpha_before: engine.state.alpha_state,
                alpha_after_or_scheduled: engine.state.alpha_state,
                current_alpha_after: engine.state.alpha_state,
                scheduled_sequence: None,
                state_before_sha256: before.clone(),
                state_after_sha256: &before.into(),
                mutated: false,
            },
        );
    } else if report.eligibility == LabelConsumptionEligibilityV1::Blocked {
        return make_update_receipt(
            engine,
            delivery,
            Some(&prediction_receipt),
            AciUpdateOutcomeV1 {
                disposition: AciUpdateDispositionV1::Blocked,
                reason: Some(AciUpdateReasonV1::LabelNotEligible),
                miss: None,
                alpha_before: engine.state.alpha_state,
                alpha_after_or_scheduled: engine.state.alpha_state,
                current_alpha_after: engine.state.alpha_state,
                scheduled_sequence: None,
                state_before_sha256: before.clone(),
                state_after_sha256: &before.into(),
                mutated: false,
            },
        );
    } else if !report.downstream_consumption_allowed {
        reason = Some(AciUpdateReasonV1::DownstreamConsumptionNotAllowed);
    } else if report.label_id != delivery.label_id {
        reason = Some(AciUpdateReasonV1::LabelIdentityMismatch);
    } else if report.label_revision != delivery.label_revision {
        reason = Some(AciUpdateReasonV1::LabelRevisionMismatch);
    } else if report.measurement_event_id != delivery.measurement_event_id
        || prediction_receipt.measurement_event_id != delivery.measurement_event_id
    {
        reason = Some(AciUpdateReasonV1::MeasurementEventMismatch);
    } else if !delivery.target_value.is_finite() {
        reason = Some(AciUpdateReasonV1::DeliveredValueNonFinite);
    } else if delivery.target_value.abs() > engine.policy.maximum_abs_target {
        reason = Some(AciUpdateReasonV1::DeliveredValueOutOfPolicy);
    } else if report.qualified_corrected_value != delivery.target_value {
        reason = Some(AciUpdateReasonV1::DeliveredValueMismatch);
    } else if delivery.label_payload_sha256 != report.label_sha256 {
        reason = Some(AciUpdateReasonV1::LabelPayloadDigestMismatch);
    } else if delivery.clock_domain_id != engine.clock_domain_id
        || prediction_receipt.clock_domain_id != delivery.clock_domain_id
    {
        reason = Some(AciUpdateReasonV1::ClockDomainMismatch);
    } else if delivery.label_available_at_unix_seconds < prediction_receipt.event_time_unix_seconds
    {
        reason = Some(AciUpdateReasonV1::LabelAvailableBeforeMeasurement);
    } else if delivery.delivered_at_unix_seconds < delivery.label_available_at_unix_seconds {
        reason = Some(AciUpdateReasonV1::DeliveryBeforeLabelAvailability);
    } else if delivery.eligibility_report_available_at_unix_seconds
        < delivery.label_available_at_unix_seconds
    {
        reason = Some(AciUpdateReasonV1::ReportBeforeLabelAvailability);
    } else if delivery.delivered_at_unix_seconds
        < delivery.eligibility_report_available_at_unix_seconds
    {
        reason = Some(AciUpdateReasonV1::ReportUnavailableAtDelivery);
    } else if delivery.delivered_at_unix_seconds < prediction_receipt.prediction_at_unix_seconds {
        reason = Some(AciUpdateReasonV1::DeliveryBeforePrediction);
    } else if delivery.delivered_at_unix_seconds - prediction_receipt.prediction_at_unix_seconds
        > engine.policy.maximum_delivery_delay_seconds
    {
        reason = Some(AciUpdateReasonV1::DeliveryDelayExceeded);
    } else if delivery.idempotency_key != revision_key {
        reason = Some(AciUpdateReasonV1::LabelIdentityMismatch);
    }

    if let Some(reason) = reason {
        return make_update_receipt(
            engine,
            delivery,
            Some(&prediction_receipt),
            AciUpdateOutcomeV1 {
                disposition: AciUpdateDispositionV1::Rejected,
                reason: Some(reason),
                miss: None,
                alpha_before: engine.state.alpha_state,
                alpha_after_or_scheduled: engine.state.alpha_state,
                current_alpha_after: engine.state.alpha_state,
                scheduled_sequence: None,
                state_before_sha256: before.clone(),
                state_after_sha256: &before.into(),
                mutated: false,
            },
        );
    }

    let miss = delivery.target_value < prediction_receipt.interval_lower
        || delivery.target_value > prediction_receipt.interval_upper;
    let error = if miss { 1.0 } else { 0.0 };
    let alpha_before = match engine.mode {
        AciUpdateModeV1::FixedHorizonDelayed { .. } => prediction_receipt.alpha_at_prediction,
        AciUpdateModeV1::ArrivalOrder => engine.state.alpha_state,
    };
    let candidate_alpha =
        alpha_before + engine.policy.learning_rate * (engine.policy.target_miscoverage - error);
    let candidate_alpha =
        candidate_alpha.clamp(engine.policy.minimum_alpha, engine.policy.maximum_alpha);

    let (disposition, current_alpha_after, scheduled_sequence, alpha_after_or_scheduled) =
        match engine.mode {
            AciUpdateModeV1::ArrivalOrder => (
                AciUpdateDispositionV1::UpdatedArrivalOrder,
                candidate_alpha,
                None,
                candidate_alpha,
            ),
            AciUpdateModeV1::FixedHorizonDelayed { horizon_steps } => {
                let target_sequence = prediction_receipt
                    .sequence
                    .checked_add(horizon_steps)
                    .ok_or_else(|| {
                        DelayedAciErrorV1("fixed ACI target sequence overflow".into())
                    })?;
                if target_sequence < engine.state.next_prediction_sequence {
                    return make_update_receipt(
                        engine,
                        delivery,
                        Some(&prediction_receipt),
                        AciUpdateOutcomeV1 {
                            disposition: AciUpdateDispositionV1::Rejected,
                            reason: Some(AciUpdateReasonV1::FixedHorizonSlotAlreadyPassed),
                            miss: Some(miss),
                            alpha_before,
                            alpha_after_or_scheduled: alpha_before,
                            current_alpha_after: engine.state.alpha_state,
                            scheduled_sequence: None,
                            state_before_sha256: before.clone(),
                            state_after_sha256: &before.into(),
                            mutated: false,
                        },
                    );
                }
                if engine
                    .state
                    .scheduled_alpha_by_sequence
                    .contains_key(&target_sequence)
                {
                    return make_update_receipt(
                        engine,
                        delivery,
                        Some(&prediction_receipt),
                        AciUpdateOutcomeV1 {
                            disposition: AciUpdateDispositionV1::Rejected,
                            reason: Some(AciUpdateReasonV1::FixedHorizonScheduleConflict),
                            miss: Some(miss),
                            alpha_before,
                            alpha_after_or_scheduled: alpha_before,
                            current_alpha_after: engine.state.alpha_state,
                            scheduled_sequence: None,
                            state_before_sha256: before.clone(),
                            state_after_sha256: &before.into(),
                            mutated: false,
                        },
                    );
                }
                (
                    AciUpdateDispositionV1::Scheduled,
                    engine.state.alpha_state,
                    Some(target_sequence),
                    candidate_alpha,
                )
            }
        };

    let mut next_state = engine.state.clone();
    if let Some(sequence) = scheduled_sequence {
        next_state
            .scheduled_alpha_by_sequence
            .insert(sequence, alpha_after_or_scheduled);
    } else {
        next_state.alpha_state = current_alpha_after;
    }
    let after = state_digest(&next_state)?;
    let receipt = make_update_receipt(
        engine,
        delivery,
        Some(&prediction_receipt),
        AciUpdateOutcomeV1 {
            disposition,
            reason: None,
            miss: Some(miss),
            alpha_before,
            alpha_after_or_scheduled,
            current_alpha_after,
            scheduled_sequence,
            state_before_sha256: before.clone(),
            state_after_sha256: after.clone(),
            mutated: true,
        },
    )?;

    engine.state = next_state;
    engine.accepted_label_revisions.insert(revision_key);
    engine.pending_predictions.remove(&delivery.sample_id);
    Ok(receipt)
}

pub fn compute_delay_memory_diagnostic_v1(
    delay_steps: u64,
    memory_steps: Option<f64>,
    residual_dependence_family: ResidualDependenceFamilyV1,
) -> Result<AciDelayMemoryDiagnosticV1, DelayedAciErrorV1> {
    let (ratio, regime, interpretation) = match memory_steps {
        Some(memory) if memory.is_finite() && memory > 0.0 => {
            let ratio = delay_steps as f64 / memory;
            if !ratio.is_finite() {
                return Err(DelayedAciErrorV1(
                    "delay-to-memory ratio overflowed; supply a better-scaled memory estimate"
                        .into(),
                ));
            }
            let regime = if ratio < 0.5 {
                DelayMemoryRegimeV1::DelayShorterThanMemory
            } else if ratio <= 2.0 {
                DelayMemoryRegimeV1::DelayComparableToMemory
            } else {
                DelayMemoryRegimeV1::DelayLongerThanMemory
            };
            let interpretation = match residual_dependence_family {
                ResidualDependenceFamilyV1::Ar1 => {
                    "AR(1): delay-to-memory ratio is a useful persistence diagnostic under the cited delayed-ACI study."
                }
                ResidualDependenceFamilyV1::Garch11 => {
                    "GARCH(1,1): use volatility-memory interpretation; the ratio is less predictive of overall performance."
                }
                ResidualDependenceFamilyV1::MarkovSwitching => {
                    "Markov switching: use regime-memory interpretation; the ratio does not summarize all regime effects."
                }
                ResidualDependenceFamilyV1::Unspecified => {
                    "Unspecified residual process: ratio is descriptive and carries no method-specific guarantee."
                }
            };
            (Some(ratio), regime, interpretation)
        }
        _ => (
            None,
            DelayMemoryRegimeV1::Unknown,
            "Residual memory is unavailable or invalid; no delay-to-memory conclusion is reported.",
        ),
    };
    let mut diagnostic = AciDelayMemoryDiagnosticV1 {
        schema: ACI_DELAY_MEMORY_DIAGNOSTIC_SCHEMA_V1.into(),
        residual_dependence_family,
        delay_steps,
        memory_steps,
        delay_to_memory_ratio: ratio,
        regime,
        interpretation: interpretation.into(),
        diagnostic_sha256: String::new(),
    };
    let mut canonical = diagnostic.clone();
    canonical.diagnostic_sha256.clear();
    diagnostic.diagnostic_sha256 = json_sha256(&canonical)?;
    Ok(diagnostic)
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
            policy_id: "semi-delayed-aci-v1".into(),
            revision: 1,
            target_miscoverage: 0.10,
            learning_rate: 0.05,
            minimum_alpha: 0.01,
            maximum_alpha: 0.49,
            maximum_abs_target: 1.0e6,
            maximum_delivery_delay_seconds: 10_000,
        }
    }

    fn engine(mode: AciUpdateModeV1) -> DelayedAciEngineV1 {
        create_delayed_aci_engine_v1("aci-001", "fab-clock", mode, policy()).unwrap()
    }

    fn prediction_input(
        sample: &str,
        sequence_alpha: f64,
        event_id: &str,
        event_time: i64,
        prediction_time: i64,
        low: f64,
        high: f64,
    ) -> AciPredictionInputV1 {
        AciPredictionInputV1 {
            schema: ACI_PREDICTION_INPUT_SCHEMA_V1.into(),
            prediction_id: format!("prediction-{sample}"),
            sample_id: sample.into(),
            measurement_event_id: event_id.into(),
            event_time_unix_seconds: event_time,
            prediction_at_unix_seconds: prediction_time,
            clock_domain_id: "fab-clock".into(),
            global_model_artifact_sha256: digest('a'),
            interval_policy_sha256: digest('b'),
            alpha_used: sequence_alpha,
            interval_lower: low,
            interval_upper: high,
            upstream_prediction_receipt_sha256: digest('c'),
        }
    }

    fn eligible_report(
        label: &str,
        event_id: &str,
        revision: u64,
        payload: &str,
        value: f64,
    ) -> LabelConsumptionEligibilityReportV1 {
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: LABEL_CONSUMPTION_REPORT_SCHEMA_V1.into(),
            label_id: label.into(),
            label_revision: revision,
            measurement_event_id: event_id.into(),
            calibration_certificate_id: "cal-cert-1".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            downstream_consumption_allowed: true,
            reasons: Vec::new(),
            label_sha256: payload.into(),
            measurement_sha256: digest('d'),
            qualified_corrected_value: value,
            qualified_standard_uncertainty_upper_bound: Some(0.1),
            qualification_report_sha256: digest('e'),
            requalification_report_sha256: digest('f'),
            requalification_frontier_sha256: digest('1'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        report.report_sha256 = eligibility_report_digest(&report).unwrap();
        report
    }

    fn delivery(
        sample: &str,
        event_id: &str,
        label: &str,
        revision: u64,
        value: f64,
        availability_times: (i64, i64, i64),
    ) -> AciLabelDeliveryV1 {
        let (available, report_available, delivered) = availability_times;
        let payload = digest('2');
        AciLabelDeliveryV1 {
            schema: ACI_LABEL_DELIVERY_SCHEMA_V1.into(),
            sample_id: sample.into(),
            label_id: label.into(),
            label_revision: revision,
            measurement_event_id: event_id.into(),
            label_payload_sha256: payload.clone(),
            target_value: value,
            label_available_at_unix_seconds: available,
            eligibility_report_available_at_unix_seconds: report_available,
            delivered_at_unix_seconds: delivered,
            clock_domain_id: "fab-clock".into(),
            idempotency_key: format!("{label}:{revision}"),
            eligibility_report: eligible_report(label, event_id, revision, &payload, value),
        }
    }

    fn register(
        engine: &mut DelayedAciEngineV1,
        sample: &str,
        event_id: &str,
        time: i64,
        lower: f64,
        upper: f64,
    ) -> AciPredictionReceiptV1 {
        let alpha = current_aci_alpha_v1(engine).unwrap();
        register_aci_prediction_v1(
            engine,
            &prediction_input(sample, alpha, event_id, time, time + 1, lower, upper),
        )
        .unwrap()
    }

    #[test]
    fn prediction_interval_and_alpha_are_immutable_after_outcome() {
        let mut engine = engine(AciUpdateModeV1::ArrivalOrder);
        let prediction = register(&mut engine, "s1", "e1", 100, -1.0, 1.0);
        let delivery = delivery("s1", "e1", "l1", 1, 2.0, (110, 111, 112));
        let receipt = deliver_aci_label_v1(&mut engine, &delivery).unwrap();
        assert_eq!(receipt.interval_missed, Some(true));
        assert_eq!(prediction.alpha_at_prediction, 0.10);
        assert_eq!(prediction.interval_lower, -1.0);
        assert_eq!(prediction.interval_upper, 1.0);
        assert!(engine.state.alpha_state < prediction.alpha_at_prediction);
    }

    #[test]
    fn covered_outcome_increases_alpha_for_future_predictions() {
        let mut engine = engine(AciUpdateModeV1::ArrivalOrder);
        register(&mut engine, "s1", "e1", 100, -1.0, 1.0);
        let receipt = deliver_aci_label_v1(
            &mut engine,
            &delivery("s1", "e1", "l1", 1, 0.0, (110, 111, 112)),
        )
        .unwrap();
        assert_eq!(receipt.interval_missed, Some(false));
        assert!(engine.state.alpha_state > 0.10);
    }

    #[test]
    fn fixed_horizon_delayed_update_is_scheduled_at_the_correct_future_index() {
        let mut engine = engine(AciUpdateModeV1::FixedHorizonDelayed { horizon_steps: 2 });
        let first = register(&mut engine, "s1", "e1", 100, -1.0, 1.0);
        let receipt = deliver_aci_label_v1(
            &mut engine,
            &delivery("s1", "e1", "l1", 1, 2.0, (102, 103, 103)),
        )
        .unwrap();
        assert_eq!(receipt.disposition, AciUpdateDispositionV1::Scheduled);
        assert_eq!(receipt.scheduled_sequence, Some(first.sequence + 2));
        assert_eq!(current_aci_alpha_v1(&engine).unwrap(), 0.10);

        let second = register(&mut engine, "s2", "e2", 104, -1.0, 1.0);
        assert_eq!(second.alpha_at_prediction, 0.10);
        let third_alpha = current_aci_alpha_v1(&engine).unwrap();
        let third = register(&mut engine, "s3", "e3", 105, -1.0, 1.0);
        assert!((third.alpha_at_prediction - third_alpha).abs() < f64::EPSILON);
        assert!(third.alpha_at_prediction < 0.10);
    }

    #[test]
    fn fixed_horizon_label_arriving_after_its_slot_is_rejected_without_update() {
        let mut engine = engine(AciUpdateModeV1::FixedHorizonDelayed { horizon_steps: 1 });
        register(&mut engine, "s1", "e1", 100, -1.0, 1.0);
        register(&mut engine, "s2", "e2", 102, -1.0, 1.0);
        let before = state_digest(&engine.state).unwrap();
        let receipt = deliver_aci_label_v1(
            &mut engine,
            &delivery("s1", "e1", "l1", 1, 2.0, (103, 104, 104)),
        )
        .unwrap();
        assert_eq!(receipt.disposition, AciUpdateDispositionV1::Rejected);
        assert_eq!(
            receipt.reason,
            Some(AciUpdateReasonV1::FixedHorizonSlotAlreadyPassed)
        );
        assert_eq!(state_digest(&engine.state).unwrap(), before);
    }

    #[test]
    fn unknown_label_is_held_and_can_be_represented_as_current_k1() {
        let mut engine = engine(AciUpdateModeV1::ArrivalOrder);
        register(&mut engine, "s1", "e1", 100, -1.0, 1.0);
        let before = state_digest(&engine.state).unwrap();
        let mut unknown = delivery("s1", "e1", "l1", 1, 0.0, (110, 111, 112));
        unknown.eligibility_report.eligibility = LabelConsumptionEligibilityV1::Unknown;
        unknown.eligibility_report.report_sha256 =
            eligibility_report_digest(&unknown.eligibility_report).unwrap();
        let held = deliver_aci_label_v1(&mut engine, &unknown).unwrap();
        assert_eq!(held.disposition, AciUpdateDispositionV1::HeldUnknown);
        assert_eq!(state_digest(&engine.state).unwrap(), before);

        let qualified = delivery("s1", "e1", "l1", 1, 0.0, (110, 111, 113));
        let accepted = deliver_aci_label_v1(&mut engine, &qualified).unwrap();
        assert_eq!(
            accepted.disposition,
            AciUpdateDispositionV1::UpdatedArrivalOrder
        );
    }

    #[test]
    fn blocked_label_never_updates_alpha() {
        let mut engine = engine(AciUpdateModeV1::ArrivalOrder);
        register(&mut engine, "s1", "e1", 100, -1.0, 1.0);
        let before = state_digest(&engine.state).unwrap();
        let mut blocked = delivery("s1", "e1", "l1", 1, 2.0, (110, 111, 112));
        blocked.eligibility_report.eligibility = LabelConsumptionEligibilityV1::Blocked;
        blocked.eligibility_report.report_sha256 =
            eligibility_report_digest(&blocked.eligibility_report).unwrap();
        let receipt = deliver_aci_label_v1(&mut engine, &blocked).unwrap();
        assert_eq!(receipt.disposition, AciUpdateDispositionV1::Blocked);
        assert_eq!(state_digest(&engine.state).unwrap(), before);
    }

    #[test]
    fn duplicate_label_revision_cannot_update_twice() {
        let mut engine = engine(AciUpdateModeV1::ArrivalOrder);
        register(&mut engine, "s1", "e1", 100, -1.0, 1.0);
        let delivered = delivery("s1", "e1", "l1", 1, 2.0, (110, 111, 112));
        assert_eq!(
            deliver_aci_label_v1(&mut engine, &delivered)
                .unwrap()
                .disposition,
            AciUpdateDispositionV1::UpdatedArrivalOrder
        );
        let before = state_digest(&engine.state).unwrap();
        let duplicate = deliver_aci_label_v1(&mut engine, &delivered).unwrap();
        assert_eq!(duplicate.disposition, AciUpdateDispositionV1::Duplicate);
        assert_eq!(state_digest(&engine.state).unwrap(), before);
    }

    #[test]
    fn mismatched_identity_revision_value_or_payload_is_rejected() {
        let mut engine = engine(AciUpdateModeV1::ArrivalOrder);
        register(&mut engine, "s1", "e1", 100, -1.0, 1.0);
        let mut wrong_revision = delivery("s1", "e1", "l1", 2, 0.0, (110, 111, 112));
        wrong_revision.eligibility_report.label_revision = 1;
        wrong_revision.eligibility_report.report_sha256 =
            eligibility_report_digest(&wrong_revision.eligibility_report).unwrap();
        let before = state_digest(&engine.state).unwrap();
        let receipt = deliver_aci_label_v1(&mut engine, &wrong_revision).unwrap();
        assert_eq!(receipt.disposition, AciUpdateDispositionV1::Rejected);
        assert_eq!(state_digest(&engine.state).unwrap(), before);
    }

    #[test]
    fn report_must_be_available_after_label_and_before_delivery() {
        let mut engine = engine(AciUpdateModeV1::ArrivalOrder);
        register(&mut engine, "s1", "e1", 100, -1.0, 1.0);
        let mut candidate = delivery("s1", "e1", "l1", 1, 0.0, (110, 111, 112));
        candidate.eligibility_report_available_at_unix_seconds = 109;
        let before = state_digest(&engine.state).unwrap();
        let receipt = deliver_aci_label_v1(&mut engine, &candidate).unwrap();
        assert_eq!(receipt.disposition, AciUpdateDispositionV1::Rejected);
        assert_eq!(state_digest(&engine.state).unwrap(), before);
    }

    #[test]
    fn out_of_order_label_arrivals_update_only_current_arrival_order_state() {
        let mut engine = engine(AciUpdateModeV1::ArrivalOrder);
        register(&mut engine, "s1", "e1", 100, -1.0, 1.0);
        register(&mut engine, "s2", "e2", 103, -1.0, 1.0);
        let before = engine.state.alpha_state;

        let later_sample_first = delivery("s2", "e2", "l2", 1, 2.0, (120, 121, 122));
        let later_receipt = deliver_aci_label_v1(&mut engine, &later_sample_first).unwrap();
        assert_eq!(
            later_receipt.disposition,
            AciUpdateDispositionV1::UpdatedArrivalOrder
        );
        let after_first = engine.state.alpha_state;
        assert!(after_first < before);

        let earlier_sample_late = delivery("s1", "e1", "l1", 1, 0.0, (110, 111, 130));
        let earlier_receipt = deliver_aci_label_v1(&mut engine, &earlier_sample_late).unwrap();
        assert_eq!(
            earlier_receipt.disposition,
            AciUpdateDispositionV1::UpdatedArrivalOrder
        );
        assert!(engine.state.alpha_state > after_first);
        assert_eq!(earlier_receipt.effective_delay_seconds, Some(29));
    }

    #[test]
    fn fixed_horizon_schedule_is_deterministic_with_multiple_interleaved_phases() {
        let mut engine = engine(AciUpdateModeV1::FixedHorizonDelayed { horizon_steps: 3 });
        let p0 = register(&mut engine, "s0", "e0", 100, -1.0, 1.0);
        let p1 = register(&mut engine, "s1", "e1", 103, -1.0, 1.0);
        let p2 = register(&mut engine, "s2", "e2", 106, -1.0, 1.0);
        let d0 = delivery("s0", "e0", "l0", 1, 2.0, (110, 111, 112));
        let d1 = delivery("s1", "e1", "l1", 1, 0.0, (113, 114, 115));
        assert_eq!(
            deliver_aci_label_v1(&mut engine, &d0)
                .unwrap()
                .scheduled_sequence,
            Some(p0.sequence + 3)
        );
        assert_eq!(
            deliver_aci_label_v1(&mut engine, &d1)
                .unwrap()
                .scheduled_sequence,
            Some(p1.sequence + 3)
        );
        let p3 = register(&mut engine, "s3", "e3", 109, -1.0, 1.0);
        let p4 = register(&mut engine, "s4", "e4", 112, -1.0, 1.0);
        let p5 = register(&mut engine, "s5", "e5", 115, -1.0, 1.0);
        assert!(p3.alpha_at_prediction < 0.10);
        assert!(p4.alpha_at_prediction > 0.10);
        assert_eq!(p5.alpha_at_prediction, p4.alpha_at_prediction);
        assert_eq!(p2.alpha_at_prediction, 0.10);
    }

    #[test]
    fn delay_memory_diagnostic_is_explicit_and_process_specific() {
        let ar =
            compute_delay_memory_diagnostic_v1(10, Some(20.0), ResidualDependenceFamilyV1::Ar1)
                .unwrap();
        assert_eq!(ar.delay_to_memory_ratio, Some(0.5));
        assert_eq!(ar.regime, DelayMemoryRegimeV1::DelayComparableToMemory);
        assert!(ar.interpretation.contains("AR(1)"));

        let garch = compute_delay_memory_diagnostic_v1(
            100,
            Some(10.0),
            ResidualDependenceFamilyV1::Garch11,
        )
        .unwrap();
        assert_eq!(garch.regime, DelayMemoryRegimeV1::DelayLongerThanMemory);
        assert!(garch.interpretation.contains("GARCH"));

        let unknown =
            compute_delay_memory_diagnostic_v1(5, None, ResidualDependenceFamilyV1::Unspecified)
                .unwrap();
        assert_eq!(unknown.regime, DelayMemoryRegimeV1::Unknown);
    }

    #[test]
    fn deterministic_replay_preserves_arrival_order_alpha_path() {
        fn run() -> (f64, String) {
            let mut engine = engine(AciUpdateModeV1::ArrivalOrder);
            for index in 0..3 {
                register(
                    &mut engine,
                    &format!("s{index}"),
                    &format!("e{index}"),
                    100 + index * 3,
                    -1.0,
                    1.0,
                );
                deliver_aci_label_v1(
                    &mut engine,
                    &delivery(
                        &format!("s{index}"),
                        &format!("e{index}"),
                        &format!("l{index}"),
                        1,
                        if index == 1 { 2.0 } else { 0.0 },
                        (110 + index * 3, 111 + index * 3, 112 + index * 3),
                    ),
                )
                .unwrap();
            }
            (
                engine.state.alpha_state,
                state_digest(&engine.state).unwrap(),
            )
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn prediction_sequence_overflow_does_not_mutate_calibrator() {
        let mut engine = engine(AciUpdateModeV1::ArrivalOrder);
        engine.state.next_prediction_sequence = u64::MAX;
        let before = state_digest(&engine.state).unwrap();
        let input = prediction_input("s1", 0.10, "e1", 100, 101, -1.0, 1.0);
        assert!(register_aci_prediction_v1(&mut engine, &input).is_err());
        assert_eq!(state_digest(&engine.state).unwrap(), before);
        assert!(engine.pending_predictions.is_empty());
    }
}
