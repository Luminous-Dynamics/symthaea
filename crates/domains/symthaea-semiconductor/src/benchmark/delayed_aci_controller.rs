// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fixed-delay, interleaved Adaptive Conformal Inference (ACI) controller.
//!
//! The caller supplies an interval constructed at the controller's issued
//! miscoverage level. This module freezes that interval and adapts the
//! miscoverage state only after an exactly-on-schedule, K1-qualified outcome
//! arrives. It does not construct intervals or claim conditional per-wafer
//! coverage. Variable-delay and capacity-constrained tracking are separate
//! algorithms and are intentionally not approximated here.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};

pub const DELAYED_ACI_POLICY_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci-policy.v1";
pub const DELAYED_ACI_ENGINE_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci-engine.v1";
pub const DELAYED_ACI_PREDICTION_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci-prediction.v1";
pub const DELAYED_ACI_DELIVERY_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci-delivery.v1";
pub const DELAYED_ACI_PREDICTION_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.delayed-aci-prediction-receipt.v1";
pub const DELAYED_ACI_UPDATE_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.delayed-aci-update-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelayedAciDispositionV1 {
    Updated,
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
    DeliveredValueMismatch,
    LabelNotYetAvailable,
    LabelAvailableBeforeMeasurement,
    DeliveryBeforePrediction,
    EligibilityReportBeforeLabel,
    EligibilityReportUnavailableAtDelivery,
    FeedbackStepMismatch,
    ClockDomainMismatch,
    DuplicateAcceptedRevision,
    IntervalMethodMismatch,
    InvalidNumericInput,
    PendingPredictionMissing,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub target_miscoverage: f64,
    pub adaptation_rate: f64,
    pub minimum_miscoverage: f64,
    pub maximum_miscoverage: f64,
    /// Fixed feedback delay in prediction steps. Zero-delay is a separate
    /// immediate-feedback baseline; v1 requires at least one step.
    pub feedback_delay_steps: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciStateV1 {
    /// One ACI state per residue class modulo the fixed feedback delay.
    pub alpha_lanes: Vec<f64>,
    pub lane_update_counts: Vec<u64>,
    pub accepted_feedback_count: u64,
    /// Becomes false after missing, late, unqualified, or out-of-order feedback.
    /// Alpha state is still retained, but theorem assumptions are no longer
    /// represented as intact by this run.
    pub fixed_delay_assumptions_intact: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub clock_domain_id: String,
    pub interval_method_sha256: String,
    pub policy_sha256: String,
    pub policy: DelayedAciPolicyV1,
    pub state: DelayedAciStateV1,
    pub next_prediction_step: u64,
    pub pending_predictions: BTreeMap<String, DelayedAciPredictionReceiptV1>,
    pub accepted_label_revisions: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciPredictionInputV1 {
    pub schema: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub interval_method_sha256: String,
    /// The caller's interval constructor must declare which alpha it used.
    pub constructed_for_alpha: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciPredictionReceiptV1 {
    pub schema: String,
    pub prediction_step: u64,
    pub lane: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub interval_method_sha256: String,
    pub policy_sha256: String,
    pub alpha_used: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub fixed_delay_assumptions_intact_at_issue: bool,
    pub receipt_sha256: String,
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
    /// Logical feedback step in the wafer/prediction stream.
    pub feedback_step: u64,
    pub clock_domain_id: String,
    pub idempotency_key: String,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
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
    pub eligibility_report_sha256: Option<String>,
    pub label_payload_sha256: String,
    pub lane: Option<u64>,
    pub alpha_before: Option<f64>,
    pub alpha_after: Option<f64>,
    pub missed_interval: Option<bool>,
    pub feedback_step: u64,
    pub fixed_delay_assumptions_intact: bool,
    pub alpha_state_mutated: bool,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct DelayedAciErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), DelayedAciErrorV1> {
    if value.trim().is_empty() {
        return Err(DelayedAciErrorV1(format!("{field}: expected non-empty text")));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, DelayedAciErrorV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| DelayedAciErrorV1(format!("JSON serialization failed: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn policy_digest(policy: &DelayedAciPolicyV1) -> Result<String, DelayedAciErrorV1> {
    json_sha256(policy)
}

fn eligibility_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, DelayedAciErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn prediction_digest(
    receipt: &DelayedAciPredictionReceiptV1,
) -> Result<String, DelayedAciErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
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
    if policy.revision == 0
        || policy.feedback_delay_steps == 0
        || !policy.target_miscoverage.is_finite()
        || !policy.adaptation_rate.is_finite()
        || policy.adaptation_rate <= 0.0
        || !policy.minimum_miscoverage.is_finite()
        || !policy.maximum_miscoverage.is_finite()
        || policy.minimum_miscoverage < 0.0
        || policy.maximum_miscoverage > 1.0
        || policy.minimum_miscoverage >= policy.maximum_miscoverage
        || policy.target_miscoverage < policy.minimum_miscoverage
        || policy.target_miscoverage > policy.maximum_miscoverage
    {
        return Err(DelayedAciErrorV1("delayed ACI policy parameters are invalid".into()));
    }
    Ok(())
}

fn validate_engine(engine: &DelayedAciEngineV1) -> Result<(), DelayedAciErrorV1> {
    if engine.schema != DELAYED_ACI_ENGINE_SCHEMA_V1 {
        return Err(DelayedAciErrorV1("delayed ACI engine schema is unsupported".into()));
    }
    require_text(&engine.engine_id, "engine.engine_id")?;
    require_text(&engine.clock_domain_id, "engine.clock_domain_id")?;
    if !valid_sha256(&engine.interval_method_sha256)
        || !valid_sha256(&engine.policy_sha256)
        || policy_digest(&engine.policy)? != engine.policy_sha256
    {
        return Err(DelayedAciErrorV1("interval or policy digest is invalid".into()));
    }
    validate_policy(&engine.policy)?;
    let delay = usize::try_from(engine.policy.feedback_delay_steps)
        .map_err(|_| DelayedAciErrorV1("feedback delay exceeds addressable memory".into()))?;
    if engine.state.alpha_lanes.len() != delay
        || engine.state.lane_update_counts.len() != delay
        || engine.state.alpha_lanes.iter().any(|alpha| {
            !alpha.is_finite()
                || *alpha < engine.policy.minimum_miscoverage
                || *alpha > engine.policy.maximum_miscoverage
        })
    {
        return Err(DelayedAciErrorV1("delayed ACI lane state is invalid".into()));
    }
    for (sample_id, receipt) in &engine.pending_predictions {
        if sample_id != &receipt.sample_id
            || receipt.schema != DELAYED_ACI_PREDICTION_RECEIPT_SCHEMA_V1
            || !valid_sha256(&receipt.receipt_sha256)
            || prediction_digest(receipt)? != receipt.receipt_sha256
            || receipt.policy_sha256 != engine.policy_sha256
            || receipt.interval_method_sha256 != engine.interval_method_sha256
            || receipt.lane >= engine.policy.feedback_delay_steps
            || receipt.lane != receipt.prediction_step % engine.policy.feedback_delay_steps
        {
            return Err(DelayedAciErrorV1(
                "pending prediction receipt is malformed or tampered".into(),
            ));
        }
    }
    Ok(())
}

/// Create a fixed-delay interleaved ACI controller. Variable feedback delays
/// and capacity-limited feedback tracking are deliberately not accepted here.
pub fn create_delayed_aci_engine_v1(
    engine_id: &str,
    clock_domain_id: &str,
    interval_method_sha256: &str,
    policy: DelayedAciPolicyV1,
) -> Result<DelayedAciEngineV1, DelayedAciErrorV1> {
    require_text(engine_id, "engine_id")?;
    require_text(clock_domain_id, "clock_domain_id")?;
    if !valid_sha256(interval_method_sha256) {
        return Err(DelayedAciErrorV1("interval method digest is malformed".into()));
    }
    validate_policy(&policy)?;
    let delay = usize::try_from(policy.feedback_delay_steps)
        .map_err(|_| DelayedAciErrorV1("feedback delay exceeds addressable memory".into()))?;
    let policy_sha256 = policy_digest(&policy)?;
    Ok(DelayedAciEngineV1 {
        schema: DELAYED_ACI_ENGINE_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        clock_domain_id: clock_domain_id.into(),
        interval_method_sha256: interval_method_sha256.into(),
        policy_sha256,
        state: DelayedAciStateV1 {
            alpha_lanes: vec![policy.target_miscoverage; delay],
            lane_update_counts: vec![0; delay],
            accepted_feedback_count: 0,
            fixed_delay_assumptions_intact: true,
        },
        policy,
        next_prediction_step: 0,
        pending_predictions: BTreeMap::new(),
        accepted_label_revisions: BTreeSet::new(),
    })
}

/// Freeze an interval and the exact alpha lane used to construct it.
/// The interval method remains external and is bound by its content digest.
pub fn issue_delayed_aci_prediction_v1(
    engine: &mut DelayedAciEngineV1,
    input: &DelayedAciPredictionInputV1,
) -> Result<DelayedAciPredictionReceiptV1, DelayedAciErrorV1> {
    validate_engine(engine)?;
    if input.schema != DELAYED_ACI_PREDICTION_SCHEMA_V1 {
        return Err(DelayedAciErrorV1("prediction schema is unsupported".into()));
    }
    require_text(&input.sample_id, "prediction.sample_id")?;
    require_text(&input.measurement_event_id, "prediction.measurement_event_id")?;
    require_text(&input.clock_domain_id, "prediction.clock_domain_id")?;
    if input.clock_domain_id != engine.clock_domain_id
        || input.interval_method_sha256 != engine.interval_method_sha256
    {
        return Err(DelayedAciErrorV1("clock domain or interval method mismatch".into()));
    }
    if input.event_time_unix_seconds > input.prediction_at_unix_seconds
        || !input.constructed_for_alpha.is_finite()
        || !input.lower_bound.is_finite()
        || !input.upper_bound.is_finite()
        || input.lower_bound > input.upper_bound
    {
        return Err(DelayedAciErrorV1("prediction interval or chronology is invalid".into()));
    }
    if engine.pending_predictions.contains_key(&input.sample_id) {
        return Err(DelayedAciErrorV1("sample already has a pending interval".into()));
    }
    let step = engine.next_prediction_step;
    let delay = engine.policy.feedback_delay_steps;
    if engine.pending_predictions.values().any(|pending| {
        pending.prediction_step.saturating_add(delay) <= step
    }) {
        engine.state.fixed_delay_assumptions_intact = false;
    }
    let lane = step % delay;
    let lane_index = usize::try_from(lane)
        .map_err(|_| DelayedAciErrorV1("ACI lane exceeds addressable memory".into()))?;
    let alpha_used = engine.state.alpha_lanes[lane_index];
    if input.constructed_for_alpha != alpha_used {
        return Err(DelayedAciErrorV1(
            "interval was not constructed for the currently active alpha lane".into(),
        ));
    }
    let next_step = step
        .checked_add(1)
        .ok_or_else(|| DelayedAciErrorV1("prediction-step overflow".into()))?;
    let mut receipt = DelayedAciPredictionReceiptV1 {
        schema: DELAYED_ACI_PREDICTION_RECEIPT_SCHEMA_V1.into(),
        prediction_step: step,
        lane,
        sample_id: input.sample_id.clone(),
        measurement_event_id: input.measurement_event_id.clone(),
        event_time_unix_seconds: input.event_time_unix_seconds,
        prediction_at_unix_seconds: input.prediction_at_unix_seconds,
        clock_domain_id: input.clock_domain_id.clone(),
        interval_method_sha256: input.interval_method_sha256.clone(),
        policy_sha256: engine.policy_sha256.clone(),
        alpha_used,
        lower_bound: input.lower_bound,
        upper_bound: input.upper_bound,
        fixed_delay_assumptions_intact_at_issue: engine.state.fixed_delay_assumptions_intact,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = prediction_digest(&receipt)?;
    engine
        .pending_predictions
        .insert(input.sample_id.clone(), receipt.clone());
    engine.next_prediction_step = next_step;
    Ok(receipt)
}

fn make_update_receipt(
    engine: &DelayedAciEngineV1,
    delivery: &DelayedAciLabelDeliveryV1,
    prediction: Option<&DelayedAciPredictionReceiptV1>,
    disposition: DelayedAciDispositionV1,
    reasons: Vec<DelayedAciReasonV1>,
    alpha_before: Option<f64>,
    alpha_after: Option<f64>,
    missed_interval: Option<bool>,
    mutated: bool,
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
        prediction_receipt_sha256: prediction.map(|value| value.receipt_sha256.clone()),
        eligibility_report_sha256: Some(delivery.eligibility_report.report_sha256.clone()),
        label_payload_sha256: delivery.label_payload_sha256.clone(),
        lane: prediction.map(|value| value.lane),
        alpha_before,
        alpha_after,
        missed_interval,
        feedback_step: delivery.feedback_step,
        fixed_delay_assumptions_intact: engine.state.fixed_delay_assumptions_intact,
        alpha_state_mutated: mutated,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = update_digest(&receipt)?;
    Ok(receipt)
}

/// Apply feedback for a prediction exactly one fixed-delay window after issue.
/// Any unqualified, late, early or mismatched outcome invalidates the run's
/// fixed-delay guarantee flag and does not update alpha.
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
    if delivery.label_revision == 0
        || !valid_sha256(&delivery.label_payload_sha256)
        || !delivery.label_value.is_finite()
    {
        return Err(DelayedAciErrorV1("delivery revision, digest or numeric value is invalid".into()));
    }

    let prediction = engine.pending_predictions.get(&delivery.sample_id).cloned();
    let report = &delivery.eligibility_report;
    let mut reasons = Vec::new();

    if engine.accepted_label_revisions.contains(&format!(
        "{}:{}",
        delivery.label_id, delivery.label_revision
    )) {
        reasons.push(DelayedAciReasonV1::DuplicateAcceptedRevision);
        return make_update_receipt(
            engine,
            delivery,
            prediction.as_ref(),
            DelayedAciDispositionV1::Duplicate,
            reasons,
            None,
            None,
            None,
            false,
        );
    }

    let Some(prediction) = prediction else {
        engine.state.fixed_delay_assumptions_intact = false;
        reasons.push(DelayedAciReasonV1::PendingPredictionMissing);
        return make_update_receipt(
            engine,
            delivery,
            None,
            DelayedAciDispositionV1::PendingPredictionMissing,
            reasons,
            None,
            None,
            None,
            false,
        );
    };

    let expected_step = prediction
        .prediction_step
        .checked_add(engine.policy.feedback_delay_steps)
        .ok_or_else(|| DelayedAciErrorV1("expected feedback step overflow".into()))?;

    if delivery.clock_domain_id != engine.clock_domain_id
        || prediction.clock_domain_id != delivery.clock_domain_id
    {
        reasons.push(DelayedAciReasonV1::ClockDomainMismatch);
    }
    if delivery.feedback_step != expected_step
        || delivery.feedback_step != engine.next_prediction_step
    {
        reasons.push(DelayedAciReasonV1::FeedbackStepMismatch);
    }
    if delivery.delivered_at_unix_seconds < delivery.label_available_at_unix_seconds {
        reasons.push(DelayedAciReasonV1::LabelNotYetAvailable);
    }
    if delivery.label_available_at_unix_seconds < prediction.event_time_unix_seconds {
        reasons.push(DelayedAciReasonV1::LabelAvailableBeforeMeasurement);
    }
    if delivery.delivered_at_unix_seconds < prediction.prediction_at_unix_seconds {
        reasons.push(DelayedAciReasonV1::DeliveryBeforePrediction);
    }
    if delivery.eligibility_report_available_at_unix_seconds
        < delivery.label_available_at_unix_seconds
    {
        reasons.push(DelayedAciReasonV1::EligibilityReportBeforeLabel);
    }
    if delivery.delivered_at_unix_seconds
        < delivery.eligibility_report_available_at_unix_seconds
    {
        reasons.push(DelayedAciReasonV1::EligibilityReportUnavailableAtDelivery);
    }
    if prediction.measurement_event_id != delivery.measurement_event_id
        || report.measurement_event_id != delivery.measurement_event_id
    {
        reasons.push(DelayedAciReasonV1::MeasurementEventMismatch);
    }
    if report.label_id != delivery.label_id {
        reasons.push(DelayedAciReasonV1::LabelIdentityMismatch);
    }
    if report.label_revision != delivery.label_revision {
        reasons.push(DelayedAciReasonV1::LabelRevisionMismatch);
    }
    if report.label_sha256 != delivery.label_payload_sha256
        || !valid_sha256(&report.label_sha256)
    {
        reasons.push(DelayedAciReasonV1::LabelPayloadDigestMismatch);
    }
    if eligibility_digest(report)? != report.report_sha256 {
        reasons.push(DelayedAciReasonV1::EligibilityReportDigestMismatch);
    }
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        reasons.push(DelayedAciReasonV1::LabelNotCurrentK1);
    }
    if report.qualified_corrected_value != delivery.label_value {
        reasons.push(DelayedAciReasonV1::DeliveredValueMismatch);
    }
    if !report.qualified_corrected_value.is_finite()
        || report
            .qualified_standard_uncertainty_upper_bound
            .is_none_or(|bound| !bound.is_finite() || bound < 0.0)
    {
        reasons.push(DelayedAciReasonV1::InvalidNumericInput);
    }
    if delivery.idempotency_key
        != format!("{}:{}", delivery.label_id, delivery.label_revision)
    {
        reasons.push(DelayedAciReasonV1::LabelIdentityMismatch);
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

    reasons.sort();
    reasons.dedup();
    if !reasons.is_empty() {
        engine.state.fixed_delay_assumptions_intact = false;
        let only_eligibility_unknown =
            report.eligibility == LabelConsumptionEligibilityV1::Unknown
                && reasons
                    .iter()
                    .all(|reason| *reason == DelayedAciReasonV1::LabelNotEligible);
        let only_eligibility_blocked =
            report.eligibility == LabelConsumptionEligibilityV1::Blocked
                && reasons
                    .iter()
                    .all(|reason| *reason == DelayedAciReasonV1::LabelNotEligible);
        let disposition = if only_eligibility_unknown {
            DelayedAciDispositionV1::HeldUnknown
        } else if only_eligibility_blocked {
            DelayedAciDispositionV1::Blocked
        } else {
            DelayedAciDispositionV1::Rejected
        };
        return make_update_receipt(
            engine,
            delivery,
            Some(&prediction),
            disposition,
            reasons,
            Some(prediction.alpha_used),
            Some(engine.state.alpha_lanes[prediction.lane as usize]),
            None,
            false,
        );
    }

    let lane = prediction.lane as usize;
    let alpha_before = engine.state.alpha_lanes[lane];
    if alpha_before != prediction.alpha_used {
        return Err(DelayedAciErrorV1(
            "ACI lane alpha changed before the matching fixed-delay feedback arrived".into(),
        ));
    }
    let missed = delivery.label_value < prediction.lower_bound
        || delivery.label_value > prediction.upper_bound;
    let candidate_alpha = alpha_before
        + engine.policy.adaptation_rate
            * (engine.policy.target_miscoverage - if missed { 1.0 } else { 0.0 });
    let alpha_after = candidate_alpha.clamp(
        engine.policy.minimum_miscoverage,
        engine.policy.maximum_miscoverage,
    );
    if !alpha_after.is_finite() {
        return Err(DelayedAciErrorV1("candidate alpha is non-finite".into()));
    }

    let mut candidate_state = engine.state.clone();
    candidate_state.alpha_lanes[lane] = alpha_after;
    candidate_state.lane_update_counts[lane] = candidate_state.lane_update_counts[lane]
        .checked_add(1)
        .ok_or_else(|| DelayedAciErrorV1("ACI lane update count overflow".into()))?;
    candidate_state.accepted_feedback_count = candidate_state
        .accepted_feedback_count
        .checked_add(1)
        .ok_or_else(|| DelayedAciErrorV1("accepted ACI feedback count overflow".into()))?;

    let mut candidate_engine = engine.clone();
    candidate_engine.state = candidate_state.clone();
    let receipt = make_update_receipt(
        &candidate_engine,
        delivery,
        Some(&prediction),
        DelayedAciDispositionV1::Updated,
        Vec::new(),
        Some(alpha_before),
        Some(alpha_after),
        Some(missed),
        true,
    )?;

    engine.state = candidate_state;
    engine.pending_predictions.remove(&delivery.sample_id);
    engine.accepted_label_revisions.insert(format!(
        "{}:{}",
        delivery.label_id, delivery.label_revision
    ));
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy(delay: u64) -> DelayedAciPolicyV1 {
        DelayedAciPolicyV1 {
            schema: DELAYED_ACI_POLICY_SCHEMA_V1.into(),
            policy_id: "delayed-aci-interleaved-v1".into(),
            revision: 1,
            target_miscoverage: 0.10,
            adaptation_rate: 0.10,
            minimum_miscoverage: 0.01,
            maximum_miscoverage: 0.40,
            feedback_delay_steps: delay,
        }
    }

    fn engine(delay: u64) -> DelayedAciEngineV1 {
        create_delayed_aci_engine_v1(
            "aci-engine-001",
            "fab-clock",
            &digest('a'),
            policy(delay),
        )
        .unwrap()
    }

    fn prediction(sample: &str, event: &str, alpha: f64) -> DelayedAciPredictionInputV1 {
        DelayedAciPredictionInputV1 {
            schema: DELAYED_ACI_PREDICTION_SCHEMA_V1.into(),
            sample_id: sample.into(),
            measurement_event_id: event.into(),
            event_time_unix_seconds: 1_000,
            prediction_at_unix_seconds: 1_001,
            clock_domain_id: "fab-clock".into(),
            interval_method_sha256: digest('a'),
            constructed_for_alpha: alpha,
            lower_bound: 0.0,
            upper_bound: 1.0,
        }
    }

    fn report(
        label: &str,
        event: &str,
        revision: u64,
        status: LabelConsumptionEligibilityV1,
        payload: &str,
    ) -> LabelConsumptionEligibilityReportV1 {
        let mut value = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label.into(),
            label_revision: revision,
            measurement_event_id: event.into(),
            calibration_certificate_id: "cert-1".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: status,
            downstream_consumption_allowed:
                status == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: payload.into(),
            measurement_sha256: digest('b'),
            qualified_corrected_value: 0.5,
            qualified_standard_uncertainty_upper_bound: Some(0.01),
            qualification_report_sha256: digest('c'),
            requalification_report_sha256: digest('d'),
            requalification_frontier_sha256: digest('e'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        let mut canonical = value.clone();
        canonical.report_sha256.clear();
        value.report_sha256 = json_sha256(&canonical).unwrap();
        value
    }

    fn delivery(
        sample: &str,
        event: &str,
        label: &str,
        revision: u64,
        value: f64,
        step: u64,
        status: LabelConsumptionEligibilityV1,
    ) -> DelayedAciLabelDeliveryV1 {
        let payload = digest('f');
        DelayedAciLabelDeliveryV1 {
            schema: DELAYED_ACI_DELIVERY_SCHEMA_V1.into(),
            sample_id: sample.into(),
            label_id: label.into(),
            label_revision: revision,
            measurement_event_id: event.into(),
            label_payload_sha256: payload.clone(),
            label_value: value,
            label_available_at_unix_seconds: 1_010,
            eligibility_report_available_at_unix_seconds: 1_011,
            delivered_at_unix_seconds: 1_012,
            feedback_step: step,
            clock_domain_id: "fab-clock".into(),
            idempotency_key: format!("{label}:{revision}"),
            eligibility_report: report(label, event, revision, status, &payload),
        }
    }

    fn issue(
        engine: &mut DelayedAciEngineV1,
        sample: &str,
        event: &str,
    ) -> DelayedAciPredictionReceiptV1 {
        let lane = (engine.next_prediction_step % engine.policy.feedback_delay_steps) as usize;
        let alpha = engine.state.alpha_lanes[lane];
        issue_delayed_aci_prediction_v1(engine, &prediction(sample, event, alpha)).unwrap()
    }

    #[test]
    fn fixed_delay_interleaves_independent_aci_lanes() {
        let mut engine = engine(2);
        let first = issue(&mut engine, "s0", "e0");
        let second = issue(&mut engine, "s1", "e1");
        assert_eq!(first.lane, 0);
        assert_eq!(second.lane, 1);
        assert_eq!(first.alpha_used, 0.10);
        assert_eq!(second.alpha_used, 0.10);

        let first_update = deliver_delayed_aci_feedback_v1(
            &mut engine,
            &delivery("s0", "e0", "l0", 1, 0.5, 2, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy),
        ).unwrap();
        assert_eq!(first_update.disposition, DelayedAciDispositionV1::Updated);
        assert_eq!(first_update.missed_interval, Some(false));
        assert!((engine.state.alpha_lanes[0] - 0.11).abs() < 1e-12);
        assert_eq!(engine.state.alpha_lanes[1], 0.10);

        let third = issue(&mut engine, "s2", "e2");
        assert_eq!(third.lane, 0);
        assert_eq!(third.alpha_used, 0.11);
    }

    #[test]
    fn interval_miss_reduces_alpha_to_widen_future_intervals() {
        let mut engine = engine(1);
        let receipt = issue(&mut engine, "s0", "e0");
        assert_eq!(receipt.alpha_used, 0.10);
        let update = deliver_delayed_aci_feedback_v1(
            &mut engine,
            &delivery("s0", "e0", "l0", 1, 1.5, 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy),
        ).unwrap();
        assert_eq!(update.missed_interval, Some(true));
        assert!((engine.state.alpha_lanes[0] - 0.01).abs() < 1e-12);
    }

    #[test]
    fn exact_label_revision_and_payload_are_required() {
        let mut engine = engine(1);
        issue(&mut engine, "s0", "e0");
        let mut candidate = delivery("s0", "e0", "l0", 2, 0.5, 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy);
        candidate.eligibility_report.label_revision = 1;
        let mut canonical = candidate.eligibility_report.clone();
        canonical.report_sha256.clear();
        candidate.eligibility_report.report_sha256 = json_sha256(&canonical).unwrap();
        let update = deliver_delayed_aci_feedback_v1(&mut engine, &candidate).unwrap();
        assert_eq!(update.disposition, DelayedAciDispositionV1::Rejected);
        assert!(!update.alpha_state_mutated);
    }

    #[test]
    fn unknown_feedback_does_not_update_alpha_and_invalidates_bound_flag() {
        let mut engine = engine(1);
        issue(&mut engine, "s0", "e0");
        let before = engine.state.alpha_lanes.clone();
        let update = deliver_delayed_aci_feedback_v1(
            &mut engine,
            &delivery("s0", "e0", "l0", 1, 0.5, 1, LabelConsumptionEligibilityV1::Unknown),
        ).unwrap();
        assert_eq!(update.disposition, DelayedAciDispositionV1::HeldUnknown);
        assert_eq!(engine.state.alpha_lanes, before);
        assert!(!engine.state.fixed_delay_assumptions_intact);
    }

    #[test]
    fn blocked_feedback_never_updates_alpha() {
        let mut engine = engine(1);
        issue(&mut engine, "s0", "e0");
        let before = engine.state.alpha_lanes.clone();
        let update = deliver_delayed_aci_feedback_v1(
            &mut engine,
            &delivery("s0", "e0", "l0", 1, 0.5, 1, LabelConsumptionEligibilityV1::Blocked),
        ).unwrap();
        assert_eq!(update.disposition, DelayedAciDispositionV1::Blocked);
        assert_eq!(engine.state.alpha_lanes, before);
        assert!(!engine.state.fixed_delay_assumptions_intact);
    }

    #[test]
    fn out_of_schedule_feedback_is_rejected_without_alpha_change() {
        let mut engine = engine(2);
        issue(&mut engine, "s0", "e0");
        issue(&mut engine, "s1", "e1");
        let before = engine.state.alpha_lanes.clone();
        let update = deliver_delayed_aci_feedback_v1(
            &mut engine,
            &delivery("s0", "e0", "l0", 1, 0.5, 3, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy),
        ).unwrap();
        assert_eq!(update.disposition, DelayedAciDispositionV1::Rejected);
        assert_eq!(engine.state.alpha_lanes, before);
        assert!(!engine.state.fixed_delay_assumptions_intact);
    }

    #[test]
    fn report_availability_must_follow_label_and_precede_delivery() {
        let mut engine = engine(1);
        issue(&mut engine, "s0", "e0");
        let mut candidate = delivery(
            "s0",
            "e0",
            "l0",
            1,
            0.5,
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        );
        candidate.eligibility_report_available_at_unix_seconds = 1_009;
        let before = engine.state.alpha_lanes.clone();
        let update = deliver_delayed_aci_feedback_v1(&mut engine, &candidate).unwrap();
        assert_eq!(update.disposition, DelayedAciDispositionV1::Rejected);
        assert!(update.reasons.contains(&DelayedAciReasonV1::EligibilityReportBeforeLabel));
        assert_eq!(engine.state.alpha_lanes, before);
        assert!(!engine.state.fixed_delay_assumptions_intact);
    }

    #[test]
    fn tampered_pending_prediction_receipt_is_rejected() {
        let mut engine = engine(1);
        issue(&mut engine, "s0", "e0");
        engine.pending_predictions.get_mut("s0").unwrap().upper_bound = 3.0;
        assert!(validate_engine(&engine).is_err());
    }

    #[test]
    fn missing_feedback_deadline_invalidates_assumptions_before_next_prediction() {
        let mut engine = engine(2);
        issue(&mut engine, "s0", "e0");
        issue(&mut engine, "s1", "e1");
        assert!(engine.state.fixed_delay_assumptions_intact);
        let third = issue(&mut engine, "s2", "e2");
        assert_eq!(third.prediction_step, 2);
        assert!(!third.fixed_delay_assumptions_intact_at_issue);
        assert!(!engine.state.fixed_delay_assumptions_intact);
    }

    #[test]
    fn label_cannot_be_available_before_its_measurement_event() {
        let mut engine = engine(1);
        issue(&mut engine, "s0", "e0");
        let mut candidate = delivery(
            "s0",
            "e0",
            "l0",
            1,
            0.5,
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        );
        candidate.label_available_at_unix_seconds = 999;
        candidate.eligibility_report_available_at_unix_seconds = 1_011;
        let update = deliver_delayed_aci_feedback_v1(&mut engine, &candidate).unwrap();
        assert_eq!(update.disposition, DelayedAciDispositionV1::Rejected);
        assert!(update.reasons.contains(&DelayedAciReasonV1::LabelAvailableBeforeMeasurement));
        assert!(!update.alpha_state_mutated);
    }

    #[test]
    fn historical_interval_receipt_never_changes_after_feedback() {
        let mut engine = engine(1);
        let prediction = issue(&mut engine, "s0", "e0");
        let frozen = prediction.clone();
        deliver_delayed_aci_feedback_v1(
            &mut engine,
            &delivery(
                "s0",
                "e0",
                "l0",
                1,
                0.5,
                1,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            ),
        )
        .unwrap();
        assert_eq!(prediction, frozen);
        assert_eq!(prediction.lower_bound, 0.0);
        assert_eq!(prediction.upper_bound, 1.0);
        assert_eq!(prediction.alpha_used, 0.10);
    }

    #[test]
    fn interval_method_digest_mismatch_blocks_prediction() {
        let mut engine = engine(1);
        let mut candidate = prediction("s0", "e0", 0.10);
        candidate.interval_method_sha256 = digest('f');
        assert!(issue_delayed_aci_prediction_v1(&mut engine, &candidate).is_err());
        assert_eq!(engine.next_prediction_step, 0);
        assert!(engine.pending_predictions.is_empty());
    }

    #[test]
    fn invalid_policy_delay_and_alpha_bounds_are_rejected() {
        let bad = policy(0);
        assert!(create_delayed_aci_engine_v1("e", "c", &digest('a'), bad.clone()).is_err());
        let mut bad = bad;
        bad.feedback_delay_steps = 2;
        bad.target_miscoverage = 0.9;
        assert!(create_delayed_aci_engine_v1("e", "c", &digest('a'), bad).is_err());
    }

    #[test]
    fn same_event_schedule_replays_byte_identically() {
        fn run() -> (String, String) {
            let mut engine = engine(2);
            issue(&mut engine, "s0", "e0");
            issue(&mut engine, "s1", "e1");
            deliver_delayed_aci_feedback_v1(
                &mut engine,
                &delivery("s0", "e0", "l0", 1, 1.5, 2, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy),
            ).unwrap();
            deliver_delayed_aci_feedback_v1(
                &mut engine,
                &delivery("s1", "e1", "l1", 1, 0.5, 3, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy),
            ).unwrap();
            (json_sha256(&engine.state).unwrap(), policy_digest(&engine.policy).unwrap())
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn accepted_revision_is_idempotent() {
        let mut engine = engine(1);
        issue(&mut engine, "s0", "e0");
        let accepted = delivery("s0", "e0", "l0", 1, 0.5, 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy);
        assert_eq!(
            deliver_delayed_aci_feedback_v1(&mut engine, &accepted).unwrap().disposition,
            DelayedAciDispositionV1::Updated
        );
        let before = engine.state.alpha_lanes.clone();
        let duplicate = deliver_delayed_aci_feedback_v1(&mut engine, &accepted).unwrap();
        assert_eq!(duplicate.disposition, DelayedAciDispositionV1::Duplicate);
        assert_eq!(engine.state.alpha_lanes, before);
    }
}
