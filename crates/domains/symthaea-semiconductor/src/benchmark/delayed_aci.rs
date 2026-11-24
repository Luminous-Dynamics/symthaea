// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fixed-delay adaptive conformal inference (tau-DACI) with K1-qualified feedback.
//!
//! This module implements the phase-interleaved delayed ACI recursion for a
//! declared deterministic delay. It does not implement capacity-constrained
//! CA-OCRC, variable-delay corrections, or a general conformal quantile model.
//! Prediction intervals are supplied by the caller and frozen in receipts.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};

pub const DELAYED_ACI_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci.v1";
pub const DELAYED_ACI_POLICY_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci-policy.v1";
pub const DELAYED_ACI_PREDICTION_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci-prediction.v1";
pub const DELAYED_ACI_DELIVERY_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci-delivery.v1";
pub const DELAYED_ACI_PREDICTION_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.delayed-aci-prediction-receipt.v1";
pub const DELAYED_ACI_UPDATE_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.delayed-aci-update-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AciIntervalKindV1 {
    Bounded,
    FullRealLine,
    Empty,
}

/// Infinite endpoints are explicit variants, not non-finite serialized floats.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciPredictionIntervalV1 {
    pub kind: AciIntervalKindV1,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

impl AciPredictionIntervalV1 {
    pub fn bounded(lower: f64, upper: f64) -> Self {
        Self { kind: AciIntervalKindV1::Bounded, lower: Some(lower), upper: Some(upper) }
    }

    pub fn full_real_line() -> Self {
        Self { kind: AciIntervalKindV1::FullRealLine, lower: None, upper: None }
    }

    pub fn empty() -> Self {
        Self { kind: AciIntervalKindV1::Empty, lower: None, upper: None }
    }

    fn validate_for_alpha(&self, alpha: f64) -> Result<(), DelayedAciErrorV1> {
        match self.kind {
            AciIntervalKindV1::Bounded => {
                let (Some(lower), Some(upper)) = (self.lower, self.upper) else {
                    return Err(DelayedAciErrorV1("bounded interval requires both endpoints".into()));
                };
                if !lower.is_finite() || !upper.is_finite() || lower > upper || alpha <= 0.0 || alpha > 1.0 {
                    return Err(DelayedAciErrorV1("bounded interval or alpha is invalid".into()));
                }
            }
            AciIntervalKindV1::FullRealLine => {
                if self.lower.is_some() || self.upper.is_some() || alpha > 0.0 {
                    return Err(DelayedAciErrorV1("full-real-line interval requires alpha <= 0".into()));
                }
            }
            AciIntervalKindV1::Empty => {
                if self.lower.is_some() || self.upper.is_some() || alpha <= 1.0 {
                    return Err(DelayedAciErrorV1("empty interval requires alpha > 1".into()));
                }
            }
        }
        Ok(())
    }

    fn misses(&self, value: f64) -> bool {
        match self.kind {
            AciIntervalKindV1::FullRealLine => false,
            AciIntervalKindV1::Empty => true,
            AciIntervalKindV1::Bounded => {
                let lower = self.lower.expect("validated interval");
                let upper = self.upper.expect("validated interval");
                value < lower || value > upper
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub target_miscoverage: f64,
    pub learning_rate: f64,
    pub delay_rounds: u64,
    pub initial_phase_alpha: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AciGuaranteeStateV1 {
    UnderDeclaredFixedDelayAssumptions,
    UncertifiedMissingOrInvalidFeedback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AciUpdateDispositionV1 {
    Updated,
    HeldUnknown,
    Blocked,
    Rejected,
    Duplicate,
    PendingPredictionMissing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
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
    ReportNumericValueInvalid,
    DeliveredValueMismatch,
    InvalidLabelRevision,
    IdempotencyKeyMismatch,
    ClockDomainMismatch,
    LabelNotAvailableByDelivery,
    ReportBeforeLabelAvailability,
    ReportNotAvailableByDelivery,
    DeliveryRoundMismatch,
    DeliveryBeforePredictionTime,
    LabelAvailableBeforeMeasurement,
    DuplicateAcceptedLabelRevision,
    PendingPredictionMissing,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciPredictionInputV1 {
    pub schema: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub round: u64,
    pub event_time_unix_seconds: i64,
    pub prediction_time_unix_seconds: i64,
    pub clock_domain_id: String,
    pub model_artifact_sha256: String,
    pub point_prediction: f64,
    pub interval: AciPredictionIntervalV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciPredictionReceiptV1 {
    pub schema: String,
    pub sequence: u64,
    pub phase: u64,
    pub expected_delivery_round: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub round: u64,
    pub event_time_unix_seconds: i64,
    pub prediction_time_unix_seconds: i64,
    pub clock_domain_id: String,
    pub model_artifact_sha256: String,
    pub point_prediction: f64,
    pub interval: AciPredictionIntervalV1,
    pub alpha_at_issue: f64,
    pub target_miscoverage: f64,
    pub learning_rate: f64,
    pub delay_rounds: u64,
    pub policy_sha256: String,
    pub calibration_state_before_sha256: String,
    pub calibration_state_after_sha256: String,
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
    pub delivered_round: u64,
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
    pub prediction_receipt_sha256: Option<String>,
    pub eligibility_report_sha256: Option<String>,
    pub label_payload_sha256: String,
    pub phase: Option<u64>,
    pub disposition: AciUpdateDispositionV1,
    pub reasons: Vec<AciUpdateReasonV1>,
    pub miscoverage_observed: Option<bool>,
    pub alpha_before: Option<f64>,
    pub alpha_after: Option<f64>,
    pub update_sequence: Option<u64>,
    pub guarantee_state: AciGuaranteeStateV1,
    pub calibration_state_before_sha256: String,
    pub calibration_state_after_sha256: String,
    pub phase_state_mutated: bool,
    pub equipment_control_authority: bool,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciMissingFeedbackRecordV1 {
    pub sample_id: String,
    pub measurement_event_id: String,
    pub issued_round: u64,
    pub expected_delivery_round: u64,
}

pub struct AciDegenerateResolutionReceiptV1 {
    pub schema: String,
    pub engine_id: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub issued_round: u64,
    pub resolution_round: u64,
    pub phase: u64,
    pub interval_kind: AciIntervalKindV1,
    /// Full-real-line gives false; empty gives true. No physical label is used.
    pub miscoverage_observed: bool,
    pub alpha_before: f64,
    pub alpha_after: f64,
    pub update_sequence: u64,
    pub calibration_state_before_sha256: String,
    pub calibration_state_after_sha256: String,
    pub receipt_sha256: String,
}


#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub clock_domain_id: String,
    pub policy: DelayedAciPolicyV1,
    pub policy_sha256: String,
    pub phase_alphas: Vec<f64>,
    pub current_round: Option<u64>,
    pub prediction_issued_this_round: bool,
    pub pending_predictions: BTreeMap<u64, DelayedAciPredictionReceiptV1>,
    pub accepted_label_revisions: BTreeSet<String>,
    pub accepted_update_count: u64,
    pub missing_feedback: Vec<AciMissingFeedbackRecordV1>,
    pub guarantee_state: AciGuaranteeStateV1,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct DelayedAciErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), DelayedAciErrorV1> {
    if value.trim().is_empty() {
        return Err(DelayedAciErrorV1(format!("{field}: expected non-empty text")));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, DelayedAciErrorV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| DelayedAciErrorV1(format!("canonical JSON serialization failed: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn policy_digest(policy: &DelayedAciPolicyV1) -> Result<String, DelayedAciErrorV1> {
    json_sha256(policy)
}

fn prediction_digest(value: &DelayedAciPredictionReceiptV1) -> Result<String, DelayedAciErrorV1> {
    let mut canonical = value.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn update_digest(value: &DelayedAciUpdateReceiptV1) -> Result<String, DelayedAciErrorV1> {
    let mut canonical = value.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn eligibility_report_digest(value: &LabelConsumptionEligibilityReportV1) -> Result<String, DelayedAciErrorV1> {
    let mut canonical = value.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn phase_state_digest(engine: &DelayedAciEngineV1) -> Result<String, DelayedAciErrorV1> {
    #[derive(Serialize)]
    struct PhaseState<'a> {
        phase_alphas: &'a [f64],
        accepted_update_count: u64,
        guarantee_state: AciGuaranteeStateV1,
        missing_feedback: &'a [AciMissingFeedbackRecordV1],
    }
    json_sha256(&PhaseState {
        phase_alphas: &engine.phase_alphas,
        accepted_update_count: engine.accepted_update_count,
        guarantee_state: engine.guarantee_state,
        missing_feedback: &engine.missing_feedback,
    })
}

fn validate_policy(policy: &DelayedAciPolicyV1) -> Result<(), DelayedAciErrorV1> {
    if policy.schema != DELAYED_ACI_POLICY_SCHEMA_V1 {
        return Err(DelayedAciErrorV1("delayed ACI policy schema is unsupported".into()));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if !policy.target_miscoverage.is_finite()
        || !(0.0..=1.0).contains(&policy.target_miscoverage)
        || !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || policy.delay_rounds == 0
        || policy.delay_rounds > 10_000
        || !policy.initial_phase_alpha.is_finite()
        || !(0.0..=1.0).contains(&policy.initial_phase_alpha)
    {
        return Err(DelayedAciErrorV1("delayed ACI policy has invalid parameters".into()));
    }
    Ok(())
}

fn validate_engine(engine: &DelayedAciEngineV1) -> Result<(), DelayedAciErrorV1> {
    if engine.schema != DELAYED_ACI_SCHEMA_V1 {
        return Err(DelayedAciErrorV1("delayed ACI engine schema is unsupported".into()));
    }
    require_text(&engine.engine_id, "engine.engine_id")?;
    require_text(&engine.clock_domain_id, "engine.clock_domain_id")?;
    validate_policy(&engine.policy)?;
    if engine.phase_alphas.len() != engine.policy.delay_rounds as usize
        || engine.phase_alphas.iter().any(|alpha| {
            !alpha.is_finite()
                || *alpha < -engine.policy.learning_rate
                || *alpha > 1.0 + engine.policy.learning_rate
        })
        || policy_digest(&engine.policy)? != engine.policy_sha256
    {
        return Err(DelayedAciErrorV1("delayed ACI phase/policy state is invalid".into()));
    }
    Ok(())
}

pub fn create_delayed_aci_engine_v1(
    engine_id: &str,
    clock_domain_id: &str,
    policy: DelayedAciPolicyV1,
) -> Result<DelayedAciEngineV1, DelayedAciErrorV1> {
    require_text(engine_id, "engine_id")?;
    require_text(clock_domain_id, "clock_domain_id")?;
    validate_policy(&policy)?;
    let policy_sha256 = policy_digest(&policy)?;
    Ok(DelayedAciEngineV1 {
        schema: DELAYED_ACI_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        clock_domain_id: clock_domain_id.into(),
        phase_alphas: vec![policy.initial_phase_alpha; policy.delay_rounds as usize],
        policy,
        policy_sha256,
        current_round: None,
        prediction_issued_this_round: false,
        pending_predictions: BTreeMap::new(),
        accepted_label_revisions: BTreeSet::new(),
        accepted_update_count: 0,
        missing_feedback: Vec::new(),
        guarantee_state: AciGuaranteeStateV1::UnderDeclaredFixedDelayAssumptions,
    })
}

/// Opens the next round. Outcomes whose due round has passed are marked
/// missing; they are never fabricated as covered or uncovered.
pub fn begin_delayed_aci_round_v1(
    engine: &mut DelayedAciEngineV1,
    round: u64,
) -> Result<Vec<AciMissingFeedbackRecordV1>, DelayedAciErrorV1> {
    validate_engine(engine)?;
    let expected = engine.current_round.map_or(Some(0), |previous| previous.checked_add(1));
    if expected != Some(round) {
        return Err(DelayedAciErrorV1("ACI rounds must begin at zero and advance sequentially".into()));
    }
    let overdue: Vec<u64> = engine
        .pending_predictions
        .iter()
        .filter(|(_, prediction)| prediction.expected_delivery_round < round)
        .map(|(issued_round, _)| *issued_round)
        .collect();
    let mut newly_missing = Vec::with_capacity(overdue.len());
    for issued_round in overdue {
        if let Some(prediction) = engine.pending_predictions.remove(&issued_round) {
            let record = AciMissingFeedbackRecordV1 {
                sample_id: prediction.sample_id,
                measurement_event_id: prediction.measurement_event_id,
                issued_round,
                expected_delivery_round: prediction.expected_delivery_round,
            };
            engine.missing_feedback.push(record.clone());
            newly_missing.push(record);
        }
    }
    if !newly_missing.is_empty() {
        engine.guarantee_state = AciGuaranteeStateV1::UncertifiedMissingOrInvalidFeedback;
    }
    engine.current_round = Some(round);
    engine.prediction_issued_this_round = false;
    Ok(newly_missing)
}

pub fn issue_delayed_aci_prediction_v1(
    engine: &mut DelayedAciEngineV1,
    input: &DelayedAciPredictionInputV1,
) -> Result<DelayedAciPredictionReceiptV1, DelayedAciErrorV1> {
    validate_engine(engine)?;
    if input.schema != DELAYED_ACI_PREDICTION_SCHEMA_V1 {
        return Err(DelayedAciErrorV1("ACI prediction input schema is unsupported".into()));
    }
    require_text(&input.sample_id, "prediction.sample_id")?;
    require_text(&input.measurement_event_id, "prediction.measurement_event_id")?;
    require_text(&input.clock_domain_id, "prediction.clock_domain_id")?;
    if input.clock_domain_id != engine.clock_domain_id {
        return Err(DelayedAciErrorV1("prediction clock domain mismatch".into()));
    }
    if !valid_sha256(&input.model_artifact_sha256) || !input.point_prediction.is_finite() {
        return Err(DelayedAciErrorV1("prediction model digest or point value is invalid".into()));
    }
    if input.event_time_unix_seconds > input.prediction_time_unix_seconds {
        return Err(DelayedAciErrorV1("prediction occurs before its event time".into()));
    }
    let round = engine.current_round.ok_or_else(|| {
        DelayedAciErrorV1("begin_delayed_aci_round_v1 must precede prediction".into())
    })?;
    if input.round != round || engine.prediction_issued_this_round {
        return Err(DelayedAciErrorV1("prediction round is not current or already used".into()));
    }
    let phase = round % engine.policy.delay_rounds;
    let alpha_at_issue = engine.phase_alphas[phase as usize];
    input.interval.validate_for_alpha(alpha_at_issue)?;
    let expected_delivery_round = round
        .checked_add(engine.policy.delay_rounds)
        .ok_or_else(|| DelayedAciErrorV1("expected delivery round overflow".into()))?;
    let state_before = phase_state_digest(engine)?;
    let mut receipt = DelayedAciPredictionReceiptV1 {
        schema: DELAYED_ACI_PREDICTION_RECEIPT_SCHEMA_V1.into(),
        sequence: round,
        phase,
        expected_delivery_round,
        sample_id: input.sample_id.clone(),
        measurement_event_id: input.measurement_event_id.clone(),
        round,
        event_time_unix_seconds: input.event_time_unix_seconds,
        prediction_time_unix_seconds: input.prediction_time_unix_seconds,
        clock_domain_id: input.clock_domain_id.clone(),
        model_artifact_sha256: input.model_artifact_sha256.clone(),
        point_prediction: input.point_prediction,
        interval: input.interval,
        alpha_at_issue,
        target_miscoverage: engine.policy.target_miscoverage,
        learning_rate: engine.policy.learning_rate,
        delay_rounds: engine.policy.delay_rounds,
        policy_sha256: engine.policy_sha256.clone(),
        calibration_state_before_sha256: state_before.clone(),
        calibration_state_after_sha256: state_before,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = prediction_digest(&receipt)?;
    engine.pending_predictions.insert(round, receipt.clone());
    engine.prediction_issued_this_round = true;
    Ok(receipt)
}

fn make_update_receipt(
    engine: &DelayedAciEngineV1,
    delivery: &DelayedAciLabelDeliveryV1,
    prediction: Option<&DelayedAciPredictionReceiptV1>,
    disposition: AciUpdateDispositionV1,
    reasons: Vec<AciUpdateReasonV1>,
    phase: Option<u64>,
    miscoverage_observed: Option<bool>,
    alphas: (Option<f64>, Option<f64>),
    update_sequence: Option<u64>,
    state_digests: (&str, &str),
) -> Result<DelayedAciUpdateReceiptV1, DelayedAciErrorV1> {
    let mut receipt = DelayedAciUpdateReceiptV1 {
        schema: DELAYED_ACI_UPDATE_RECEIPT_SCHEMA_V1.into(),
        engine_id: engine.engine_id.clone(),
        sample_id: delivery.sample_id.clone(),
        label_id: delivery.label_id.clone(),
        label_revision: delivery.label_revision,
        measurement_event_id: delivery.measurement_event_id.clone(),
        prediction_receipt_sha256: prediction.map(|item| item.receipt_sha256.clone()),
        eligibility_report_sha256: Some(delivery.eligibility_report.report_sha256.clone()),
        label_payload_sha256: delivery.label_payload_sha256.clone(),
        phase,
        disposition,
        reasons,
        miscoverage_observed,
        alpha_before: alphas.0,
        alpha_after: alphas.1,
        update_sequence,
        guarantee_state: engine.guarantee_state,
        calibration_state_before_sha256: state_digests.0.into(),
        calibration_state_after_sha256: state_digests.1.into(),
        phase_state_mutated: update_sequence.is_some(),
        equipment_control_authority: false,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = update_digest(&receipt)?;
    Ok(receipt)
}

fn degenerate_resolution_digest(
    receipt: &AciDegenerateResolutionReceiptV1,
) -> Result<String, DelayedAciErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

/// Resolves a full-real-line or empty prediction set at its due round.
/// The miss bit is mathematically determined by the set itself, so no physical
/// label or K1 receipt is needed. This outcome source remains distinct from a
/// metrology-observed coverage result.
pub fn resolve_degenerate_aci_interval_v1(
    engine: &mut DelayedAciEngineV1,
    sample_id: &str,
) -> Result<AciDegenerateResolutionReceiptV1, DelayedAciErrorV1> {
    validate_engine(engine)?;
    require_text(sample_id, "sample_id")?;
    let current_round = engine
        .current_round
        .ok_or_else(|| DelayedAciErrorV1("ACI round has not begun".into()))?;
    let prediction = engine
        .pending_predictions
        .values()
        .find(|item| item.sample_id == sample_id)
        .cloned()
        .ok_or_else(|| DelayedAciErrorV1("pending ACI prediction not found".into()))?;
    if prediction.expected_delivery_round != current_round {
        return Err(DelayedAciErrorV1(
            "degenerate interval can only be resolved in its exact due round".into(),
        ));
    }
    let misses = match prediction.interval.kind {
        AciIntervalKindV1::FullRealLine => false,
        AciIntervalKindV1::Empty => true,
        AciIntervalKindV1::Bounded => {
            return Err(DelayedAciErrorV1(
                "bounded intervals require a qualified physical label".into(),
            ));
        }
    };
    let state_before = phase_state_digest(engine)?;
    let alpha_before = engine.phase_alphas[prediction.phase as usize];
    if alpha_before.to_bits() != prediction.alpha_at_issue.to_bits() {
        return Err(DelayedAciErrorV1(
            "phase alpha changed before its fixed-delay feedback".into(),
        ));
    }
    let error_value = if misses { 1.0 } else { 0.0 };
    let alpha_after = alpha_before
        + engine.policy.learning_rate * (engine.policy.target_miscoverage - error_value);
    if !alpha_after.is_finite()
        || alpha_after < -engine.policy.learning_rate - 1.0e-12
        || alpha_after > 1.0 + engine.policy.learning_rate + 1.0e-12
    {
        return Err(DelayedAciErrorV1(
            "degenerate ACI update escaped the theoretical phase bound".into(),
        ));
    }
    let update_sequence = engine
        .accepted_update_count
        .checked_add(1)
        .ok_or_else(|| DelayedAciErrorV1("ACI update count overflow".into()))?;
    let mut next_engine = engine.clone();
    next_engine.phase_alphas[prediction.phase as usize] = alpha_after;
    next_engine.accepted_update_count = update_sequence;
    next_engine.pending_predictions.remove(&prediction.round);
    let state_after = phase_state_digest(&next_engine)?;
    let mut receipt = AciDegenerateResolutionReceiptV1 {
        schema: "luminous.semi-bench.delayed-aci-degenerate-resolution.v1".into(),
        engine_id: engine.engine_id.clone(),
        sample_id: prediction.sample_id,
        measurement_event_id: prediction.measurement_event_id,
        issued_round: prediction.round,
        resolution_round: current_round,
        phase: prediction.phase,
        interval_kind: prediction.interval.kind,
        miscoverage_observed: misses,
        alpha_before,
        alpha_after,
        update_sequence,
        calibration_state_before_sha256: state_before,
        calibration_state_after_sha256: state_after,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = degenerate_resolution_digest(&receipt)?;
    engine.phase_alphas = next_engine.phase_alphas;
    engine.accepted_update_count = next_engine.accepted_update_count;
    engine.pending_predictions = next_engine.pending_predictions;
    Ok(receipt)
}

/// Delivers one outcome in the round at which its fixed-delay feedback is due.
/// Unknown/blocked/mismatched evidence cannot update the alpha phase.
pub fn deliver_delayed_aci_label_v1(
    engine: &mut DelayedAciEngineV1,
    delivery: &DelayedAciLabelDeliveryV1,
) -> Result<DelayedAciUpdateReceiptV1, DelayedAciErrorV1> {
    validate_engine(engine)?;
    if delivery.schema != DELAYED_ACI_DELIVERY_SCHEMA_V1 {
        return Err(DelayedAciErrorV1("ACI label delivery schema is unsupported".into()));
    }
    require_text(&delivery.sample_id, "delivery.sample_id")?;
    require_text(&delivery.label_id, "delivery.label_id")?;
    require_text(&delivery.measurement_event_id, "delivery.measurement_event_id")?;
    require_text(&delivery.clock_domain_id, "delivery.clock_domain_id")?;
    require_text(&delivery.idempotency_key, "delivery.idempotency_key")?;
    if delivery.label_revision == 0 || !valid_sha256(&delivery.label_payload_sha256) {
        return Err(DelayedAciErrorV1("label revision or payload digest is invalid".into()));
    }

    let state_before = phase_state_digest(engine)?;
    let report = &delivery.eligibility_report;
    let prediction = engine
        .pending_predictions
        .values()
        .find(|item| item.sample_id == delivery.sample_id);

    if engine.accepted_label_revisions.contains(&format!(
        "{}:{}",
        delivery.label_id, delivery.label_revision
    )) {
        return make_update_receipt(
            engine,
            delivery,
            prediction,
            AciUpdateDispositionV1::Duplicate,
            vec![AciUpdateReasonV1::DuplicateAcceptedLabelRevision],
            prediction.map(|item| item.phase),
            None,
            (None, None),
            None,
            (&state_before, &state_before),
        );
    }

    let mut reasons = Vec::new();
    if delivery.delivered_round != engine.current_round.unwrap_or(u64::MAX) {
        reasons.push(AciUpdateReasonV1::DeliveryRoundMismatch);
    }
    if delivery.clock_domain_id != engine.clock_domain_id {
        reasons.push(AciUpdateReasonV1::ClockDomainMismatch);
    }
    if !delivery.label_value.is_finite()
        || !report.qualified_corrected_value.is_finite()
        || report.qualified_standard_uncertainty_upper_bound.is_none()
        || report
            .qualified_standard_uncertainty_upper_bound
            .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        reasons.push(AciUpdateReasonV1::ReportNumericValueInvalid);
    }
    if delivery.eligibility_report_available_at_unix_seconds
        < delivery.label_available_at_unix_seconds
    {
        reasons.push(AciUpdateReasonV1::ReportBeforeLabelAvailability);
    }
    if delivery.label_available_at_unix_seconds > delivery.delivered_at_unix_seconds {
        reasons.push(AciUpdateReasonV1::LabelNotAvailableByDelivery);
    }
    if delivery.eligibility_report_available_at_unix_seconds
        > delivery.delivered_at_unix_seconds
    {
        reasons.push(AciUpdateReasonV1::ReportNotAvailableByDelivery);
    }
    if !valid_sha256(&report.label_sha256) || report.label_sha256 != delivery.label_payload_sha256 {
        reasons.push(AciUpdateReasonV1::LabelPayloadDigestMismatch);
    }
    if eligibility_report_digest(report)? != report.report_sha256 {
        reasons.push(AciUpdateReasonV1::EligibilityReportDigestMismatch);
    }
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        reasons.push(AciUpdateReasonV1::LabelNotCurrentK1);
    }
    match report.eligibility {
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
            if report.downstream_consumption_allowed => {}
        LabelConsumptionEligibilityV1::Unknown | LabelConsumptionEligibilityV1::Blocked => {
            reasons.push(AciUpdateReasonV1::LabelNotEligible);
        }
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => {
            reasons.push(AciUpdateReasonV1::DownstreamConsumptionNotAllowed);
        }
    }
    if report.label_id != delivery.label_id {
        reasons.push(AciUpdateReasonV1::LabelIdentityMismatch);
    }
    if report.label_revision != delivery.label_revision {
        reasons.push(AciUpdateReasonV1::LabelRevisionMismatch);
    }
    if report.measurement_event_id != delivery.measurement_event_id {
        reasons.push(AciUpdateReasonV1::MeasurementEventMismatch);
    }
    if report.qualified_corrected_value != delivery.label_value {
        reasons.push(AciUpdateReasonV1::DeliveredValueMismatch);
    }
    if delivery.idempotency_key != format!("{}:{}", delivery.label_id, delivery.label_revision) {
        reasons.push(AciUpdateReasonV1::IdempotencyKeyMismatch);
    }

    let prediction = prediction.cloned();
    if let Some(prediction) = &prediction {
        if prediction.measurement_event_id != delivery.measurement_event_id {
            reasons.push(AciUpdateReasonV1::MeasurementEventMismatch);
        }
        if prediction.clock_domain_id != delivery.clock_domain_id {
            reasons.push(AciUpdateReasonV1::ClockDomainMismatch);
        }
        if delivery.label_available_at_unix_seconds < prediction.event_time_unix_seconds {
            reasons.push(AciUpdateReasonV1::LabelAvailableBeforeMeasurement);
        }
        if delivery.delivered_at_unix_seconds < prediction.prediction_time_unix_seconds {
            reasons.push(AciUpdateReasonV1::DeliveryBeforePredictionTime);
        }
        if delivery.delivered_round != prediction.expected_delivery_round {
            reasons.push(AciUpdateReasonV1::DeliveryRoundMismatch);
        }
    } else {
        reasons.push(AciUpdateReasonV1::PendingPredictionMissing);
    }
    reasons.sort();
    reasons.dedup();

    let Some(prediction) = prediction else {
        return make_update_receipt(
            engine,
            delivery,
            None,
            AciUpdateDispositionV1::PendingPredictionMissing,
            reasons,
            None,
            None,
            (None, None),
            None,
            (&state_before, &state_before),
        );
    };

    if !reasons.is_empty() {
        let only_unknown = report.eligibility == LabelConsumptionEligibilityV1::Unknown
            && reasons.iter().all(|reason| *reason == AciUpdateReasonV1::LabelNotEligible);
        let only_blocked = report.eligibility == LabelConsumptionEligibilityV1::Blocked
            && reasons.iter().all(|reason| *reason == AciUpdateReasonV1::LabelNotEligible);
        let disposition = if only_unknown {
            AciUpdateDispositionV1::HeldUnknown
        } else if only_blocked {
            AciUpdateDispositionV1::Blocked
        } else {
            AciUpdateDispositionV1::Rejected
        };
        return make_update_receipt(
            engine,
            delivery,
            Some(&prediction),
            disposition,
            reasons,
            Some(prediction.phase),
            None,
            (None, None),
            None,
            (&state_before, &state_before),
        );
    }

    let misses = prediction.interval.misses(delivery.label_value);
    let alpha_before = engine.phase_alphas[prediction.phase as usize];
    if alpha_before.to_bits() != prediction.alpha_at_issue.to_bits() {
        return Err(DelayedAciErrorV1("phase alpha changed before its fixed-delay feedback".into()));
    }
    let error_value = if misses { 1.0 } else { 0.0 };
    let alpha_after = alpha_before
        + engine.policy.learning_rate * (engine.policy.target_miscoverage - error_value);
    if !alpha_after.is_finite()
        || alpha_after < -engine.policy.learning_rate - 1.0e-12
        || alpha_after > 1.0 + engine.policy.learning_rate + 1.0e-12
    {
        return Err(DelayedAciErrorV1("delayed ACI update escaped the theoretical phase bound".into()));
    }
    let next_update_count = engine
        .accepted_update_count
        .checked_add(1)
        .ok_or_else(|| DelayedAciErrorV1("ACI update count overflow".into()))?;

    let mut next_engine = engine.clone();
    next_engine.phase_alphas[prediction.phase as usize] = alpha_after;
    next_engine.accepted_update_count = next_update_count;
    next_engine.pending_predictions.remove(&prediction.round);
    next_engine.accepted_label_revisions.insert(format!(
        "{}:{}",
        delivery.label_id, delivery.label_revision
    ));
    let state_after = phase_state_digest(&next_engine)?;
    let receipt = make_update_receipt(
        &next_engine,
        delivery,
        Some(&prediction),
        AciUpdateDispositionV1::Updated,
        Vec::new(),
        Some(prediction.phase),
        Some(misses),
        (Some(alpha_before), Some(alpha_after)),
        Some(next_update_count),
        (&state_before, &state_after),
    )?;
    engine.phase_alphas = next_engine.phase_alphas;
    engine.accepted_update_count = next_engine.accepted_update_count;
    engine.pending_predictions = next_engine.pending_predictions;
    engine.accepted_label_revisions = next_engine.accepted_label_revisions;
    Ok(receipt)
}

pub fn close_delayed_aci_stream_v1(
    engine: &mut DelayedAciEngineV1,
) -> Result<Vec<AciMissingFeedbackRecordV1>, DelayedAciErrorV1> {
    validate_engine(engine)?;
    let mut missing = Vec::with_capacity(engine.pending_predictions.len());
    for (round, prediction) in std::mem::take(&mut engine.pending_predictions) {
        let record = AciMissingFeedbackRecordV1 {
            sample_id: prediction.sample_id,
            measurement_event_id: prediction.measurement_event_id,
            issued_round: round,
            expected_delivery_round: prediction.expected_delivery_round,
        };
        engine.missing_feedback.push(record.clone());
        missing.push(record);
    }
    if !missing.is_empty() {
        engine.guarantee_state = AciGuaranteeStateV1::UncertifiedMissingOrInvalidFeedback;
    }
    Ok(missing)
}

/// Calculates tau/L; the residual-memory scale L must be estimated and
/// validated by the caller. This helper does not estimate temporal memory.
pub fn delay_to_memory_ratio_v1(
    delay_rounds: f64,
    memory_scale_rounds: f64,
) -> Result<f64, DelayedAciErrorV1> {
    if !delay_rounds.is_finite()
        || delay_rounds < 0.0
        || !memory_scale_rounds.is_finite()
        || memory_scale_rounds <= 0.0
    {
        return Err(DelayedAciErrorV1("delay-to-memory ratio has invalid inputs".into()));
    }
    Ok(delay_rounds / memory_scale_rounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy(delay_rounds: u64) -> DelayedAciPolicyV1 {
        DelayedAciPolicyV1 {
            schema: DELAYED_ACI_POLICY_SCHEMA_V1.into(),
            policy_id: "semi-vm-fixed-delay-aci.v1".into(),
            target_miscoverage: 0.10,
            learning_rate: 0.20,
            delay_rounds,
            initial_phase_alpha: 0.10,
        }
    }

    fn engine(delay_rounds: u64) -> DelayedAciEngineV1 {
        create_delayed_aci_engine_v1("aci-engine-1", "fab-clock", policy(delay_rounds)).unwrap()
    }

    fn prediction_input(
        sample: &str,
        event: &str,
        round: u64,
        interval: AciPredictionIntervalV1,
    ) -> DelayedAciPredictionInputV1 {
        DelayedAciPredictionInputV1 {
            schema: DELAYED_ACI_PREDICTION_SCHEMA_V1.into(),
            sample_id: sample.into(),
            measurement_event_id: event.into(),
            round,
            event_time_unix_seconds: 100 + round as i64 * 10,
            prediction_time_unix_seconds: 101 + round as i64 * 10,
            clock_domain_id: "fab-clock".into(),
            model_artifact_sha256: digest('a'),
            point_prediction: 10.0,
            interval,
        }
    }

    fn eligibility_report(
        label_id: &str,
        event_id: &str,
        revision: u64,
        payload: &str,
        eligibility: LabelConsumptionEligibilityV1,
        frontier: LabelEvidenceFrontierV1,
        value: f64,
    ) -> LabelConsumptionEligibilityReportV1 {
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label_id.into(),
            label_revision: revision,
            measurement_event_id: event_id.into(),
            calibration_certificate_id: "cal-cert-aci".into(),
            frontier,
            eligibility,
            downstream_consumption_allowed: eligibility == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: payload.into(),
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
        let mut canonical = report.clone();
        canonical.report_sha256.clear();
        report.report_sha256 = json_sha256(&canonical).unwrap();
        report
    }

    fn delivery(
        identity: (&str, &str, &str),
        revision: u64,
        round: u64,
        value: f64,
        eligibility: LabelConsumptionEligibilityV1,
        frontier: LabelEvidenceFrontierV1,
    ) -> DelayedAciLabelDeliveryV1 {
        let (sample, event, label) = identity;
        let payload = digest('f');
        let available = 100 + round as i64 * 10;
        DelayedAciLabelDeliveryV1 {
            schema: DELAYED_ACI_DELIVERY_SCHEMA_V1.into(),
            sample_id: sample.into(),
            label_id: label.into(),
            label_revision: revision,
            measurement_event_id: event.into(),
            label_payload_sha256: payload.clone(),
            label_value: value,
            label_available_at_unix_seconds: available,
            eligibility_report_available_at_unix_seconds: available,
            delivered_at_unix_seconds: available + 1,
            delivered_round: round,
            clock_domain_id: "fab-clock".into(),
            idempotency_key: format!("{label}:{revision}"),
            eligibility_report: eligibility_report(
                label, event, revision, &payload, eligibility, frontier, value,
            ),
        }
    }

    fn begin_and_issue(
        engine: &mut DelayedAciEngineV1,
        round: u64,
        interval: AciPredictionIntervalV1,
    ) {
        begin_delayed_aci_round_v1(engine, round).unwrap();
        issue_delayed_aci_prediction_v1(
            engine,
            &prediction_input(&format!("sample-{round}"), &format!("event-{round}"), round, interval),
        )
        .unwrap();
    }

    #[test]
    fn tau_phases_start_independently_and_freeze_prediction_alpha() {
        let mut engine = engine(3);
        for round in 0..3 {
            begin_and_issue(&mut engine, round, AciPredictionIntervalV1::bounded(9.0, 11.0));
        }
        assert_eq!(engine.phase_alphas, vec![0.10, 0.10, 0.10]);
        assert_eq!(
            engine.pending_predictions.values().map(|item| item.phase).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(engine.pending_predictions.values().all(|item| item.alpha_at_issue == 0.10));
    }

    #[test]
    fn delayed_miss_updates_only_its_phase_then_affects_next_same_phase_prediction() {
        let mut engine = engine(3);
        for round in 0..3 {
            begin_and_issue(&mut engine, round, AciPredictionIntervalV1::bounded(9.0, 11.0));
        }
        begin_delayed_aci_round_v1(&mut engine, 3).unwrap();
        let receipt = deliver_delayed_aci_label_v1(
            &mut engine,
            &delivery(
                ("sample-0", "event-0", "label-0"),
                1,
                3,
                12.0,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                LabelEvidenceFrontierV1::CurrentAtK1,
            ),
        )
        .unwrap();
        assert_eq!(receipt.disposition, AciUpdateDispositionV1::Updated);
        assert_eq!(receipt.miscoverage_observed, Some(true));
        assert_eq!(receipt.alpha_before, Some(0.10));
        assert_eq!(receipt.alpha_after, Some(-0.08));
        assert_eq!(engine.phase_alphas, vec![-0.08, 0.10, 0.10]);

        let next = issue_delayed_aci_prediction_v1(
            &mut engine,
            &prediction_input("sample-3", "event-3", 3, AciPredictionIntervalV1::full_real_line()),
        )
        .unwrap();
        assert_eq!(next.phase, 0);
        assert_eq!(next.alpha_at_issue, -0.08);
    }

    #[test]
    fn covered_outcome_increases_alpha() {
        let mut engine = engine(1);
        begin_and_issue(&mut engine, 0, AciPredictionIntervalV1::bounded(9.0, 11.0));
        begin_delayed_aci_round_v1(&mut engine, 1).unwrap();
        let receipt = deliver_delayed_aci_label_v1(
            &mut engine,
            &delivery(
                ("sample-0", "event-0", "label-0"),
                1,
                1,
                10.5,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                LabelEvidenceFrontierV1::CurrentAtK1,
            ),
        )
        .unwrap();
        assert_eq!(receipt.miscoverage_observed, Some(false));
        assert_eq!(receipt.alpha_after, Some(0.12));
    }

    #[test]
    fn unknown_label_is_held_and_can_be_requalified_in_same_due_round() {
        let mut engine = engine(1);
        begin_and_issue(&mut engine, 0, AciPredictionIntervalV1::bounded(9.0, 11.0));
        begin_delayed_aci_round_v1(&mut engine, 1).unwrap();
        let unknown = delivery(
            ("sample-0", "event-0", "label-0"),
            1,
            1,
            10.5,
            LabelConsumptionEligibilityV1::Unknown,
            LabelEvidenceFrontierV1::CurrentAtK1,
        );
        assert_eq!(
            deliver_delayed_aci_label_v1(&mut engine, &unknown).unwrap().disposition,
            AciUpdateDispositionV1::HeldUnknown
        );
        assert_eq!(engine.phase_alphas, vec![0.10]);
        assert!(engine.pending_predictions.contains_key(&0));

        let qualified = delivery(
            ("sample-0", "event-0", "label-0"),
            1,
            1,
            10.5,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
        );
        assert_eq!(
            deliver_delayed_aci_label_v1(&mut engine, &qualified).unwrap().disposition,
            AciUpdateDispositionV1::Updated
        );
        assert_eq!(engine.phase_alphas, vec![0.12]);
    }

    #[test]
    fn blocked_label_never_updates_alpha() {
        let mut engine = engine(1);
        begin_and_issue(&mut engine, 0, AciPredictionIntervalV1::bounded(9.0, 11.0));
        begin_delayed_aci_round_v1(&mut engine, 1).unwrap();
        let blocked = delivery(
            ("sample-0", "event-0", "label-0"),
            1,
            1,
            10.5,
            LabelConsumptionEligibilityV1::Blocked,
            LabelEvidenceFrontierV1::CurrentAtK1,
        );
        assert_eq!(
            deliver_delayed_aci_label_v1(&mut engine, &blocked).unwrap().disposition,
            AciUpdateDispositionV1::Blocked
        );
        assert_eq!(engine.phase_alphas, vec![0.10]);
    }

    #[test]
    fn missing_due_feedback_degrades_guarantee_without_fabricating_miss() {
        let mut engine = engine(2);
        begin_and_issue(&mut engine, 0, AciPredictionIntervalV1::bounded(9.0, 11.0));
        begin_and_issue(&mut engine, 1, AciPredictionIntervalV1::bounded(9.0, 11.0));
        assert!(begin_delayed_aci_round_v1(&mut engine, 2).unwrap().is_empty());
        let missing = begin_delayed_aci_round_v1(&mut engine, 3).unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(engine.phase_alphas, vec![0.10, 0.10]);
        assert_eq!(
            engine.guarantee_state,
            AciGuaranteeStateV1::UncertifiedMissingOrInvalidFeedback
        );
    }

    #[test]
    fn close_stream_marks_pending_feedback_missing() {
        let mut engine = engine(2);
        begin_and_issue(&mut engine, 0, AciPredictionIntervalV1::bounded(9.0, 11.0));
        let missing = close_delayed_aci_stream_v1(&mut engine).unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(engine.phase_alphas, vec![0.10, 0.10]);
        assert_eq!(
            engine.guarantee_state,
            AciGuaranteeStateV1::UncertifiedMissingOrInvalidFeedback
        );
    }

    #[test]
    fn accepted_label_revision_is_idempotent() {
        let mut engine = engine(1);
        begin_and_issue(&mut engine, 0, AciPredictionIntervalV1::bounded(9.0, 11.0));
        begin_delayed_aci_round_v1(&mut engine, 1).unwrap();
        let accepted = delivery(
            ("sample-0", "event-0", "label-0"),
            1,
            1,
            10.5,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
        );
        assert_eq!(
            deliver_delayed_aci_label_v1(&mut engine, &accepted).unwrap().disposition,
            AciUpdateDispositionV1::Updated
        );
        let alpha = engine.phase_alphas.clone();
        assert_eq!(
            deliver_delayed_aci_label_v1(&mut engine, &accepted).unwrap().disposition,
            AciUpdateDispositionV1::Duplicate
        );
        assert_eq!(engine.phase_alphas, alpha);
    }

    #[test]
    fn revision_mismatch_is_rejected_without_alpha_change() {
        let mut engine = engine(1);
        begin_and_issue(&mut engine, 0, AciPredictionIntervalV1::bounded(9.0, 11.0));
        begin_delayed_aci_round_v1(&mut engine, 1).unwrap();
        let mut candidate = delivery(
            ("sample-0", "event-0", "label-0"),
            2,
            1,
            10.5,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
        );
        candidate.eligibility_report.label_revision = 1;
        let mut canonical = candidate.eligibility_report.clone();
        canonical.report_sha256.clear();
        candidate.eligibility_report.report_sha256 = json_sha256(&canonical).unwrap();
        let receipt = deliver_delayed_aci_label_v1(&mut engine, &candidate).unwrap();
        assert_eq!(receipt.disposition, AciUpdateDispositionV1::Rejected);
        assert!(receipt.reasons.contains(&AciUpdateReasonV1::LabelRevisionMismatch));
        assert_eq!(engine.phase_alphas, vec![0.10]);
    }

    #[test]
    fn wrong_delivery_round_is_rejected_without_state_mutation() {
        let mut engine = engine(2);
        begin_and_issue(&mut engine, 0, AciPredictionIntervalV1::bounded(9.0, 11.0));
        begin_and_issue(&mut engine, 1, AciPredictionIntervalV1::bounded(9.0, 11.0));
        begin_delayed_aci_round_v1(&mut engine, 2).unwrap();
        let wrong = delivery(
            ("sample-0", "event-0", "label-0"),
            1,
            2,
            10.5,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
        );
        let receipt = deliver_delayed_aci_label_v1(&mut engine, &wrong).unwrap();
        assert_eq!(receipt.disposition, AciUpdateDispositionV1::Rejected);
        assert_eq!(engine.phase_alphas, vec![0.10, 0.10]);
    }

    #[test]
    fn stale_k0_report_never_updates() {
        let mut engine = engine(1);
        begin_and_issue(&mut engine, 0, AciPredictionIntervalV1::bounded(9.0, 11.0));
        begin_delayed_aci_round_v1(&mut engine, 1).unwrap();
        let stale = delivery(
            ("sample-0", "event-0", "label-0"),
            1,
            1,
            10.5,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::AsKnownAtK0,
        );
        let receipt = deliver_delayed_aci_label_v1(&mut engine, &stale).unwrap();
        assert_eq!(receipt.disposition, AciUpdateDispositionV1::Rejected);
        assert_eq!(engine.phase_alphas, vec![0.10]);
    }

    #[test]
    fn full_and_empty_intervals_resolve_without_physical_labels() {
        let mut full_engine = engine(1);
        full_engine.phase_alphas[0] = -0.05;
        begin_delayed_aci_round_v1(&mut full_engine, 0).unwrap();
        issue_delayed_aci_prediction_v1(
            &mut full_engine,
            &prediction_input("sample-0", "event-0", 0, AciPredictionIntervalV1::full_real_line()),
        )
        .unwrap();
        begin_delayed_aci_round_v1(&mut full_engine, 1).unwrap();
        let full = resolve_degenerate_aci_interval_v1(&mut full_engine, "sample-0").unwrap();
        assert!(!full.miscoverage_observed);
        assert_eq!(full.interval_kind, AciIntervalKindV1::FullRealLine);
        assert!(full_engine.pending_predictions.is_empty());

        let mut empty_engine = engine(1);
        empty_engine.phase_alphas[0] = 1.05;
        begin_delayed_aci_round_v1(&mut empty_engine, 0).unwrap();
        issue_delayed_aci_prediction_v1(
            &mut empty_engine,
            &prediction_input("sample-0", "event-0", 0, AciPredictionIntervalV1::empty()),
        )
        .unwrap();
        begin_delayed_aci_round_v1(&mut empty_engine, 1).unwrap();
        let empty = resolve_degenerate_aci_interval_v1(&mut empty_engine, "sample-0").unwrap();
        assert!(empty.miscoverage_observed);
        assert_eq!(empty.interval_kind, AciIntervalKindV1::Empty);
        assert!(empty_engine.pending_predictions.is_empty());
    }

    #[test]
    fn bounded_interval_cannot_use_label_free_resolution() {
        let mut engine = engine(1);
        begin_and_issue(&mut engine, 0, AciPredictionIntervalV1::bounded(9.0, 11.0));
        begin_delayed_aci_round_v1(&mut engine, 1).unwrap();
        assert!(resolve_degenerate_aci_interval_v1(&mut engine, "sample-0").is_err());
        assert_eq!(engine.phase_alphas, vec![0.10]);
        assert!(engine.pending_predictions.contains_key(&0));
    }

    #[test]
    fn deterministic_fixed_delay_replay_produces_identical_phase_state() {
        fn run() -> Vec<f64> {
            let mut engine = engine(2);
            for round in 0..6 {
                begin_delayed_aci_round_v1(&mut engine, round).unwrap();
                if round >= 2 {
                    let issued = round - 2;
                    let sample = format!("sample-{issued}");
                    let event = format!("event-{issued}");
                    let label = format!("label-{issued}");
                    let delivery = delivery(
                        (&sample, &event, &label),
                        1,
                        round,
                        10.5,
                        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                        LabelEvidenceFrontierV1::CurrentAtK1,
                    );
                    deliver_delayed_aci_label_v1(&mut engine, &delivery).unwrap();
                }
                issue_delayed_aci_prediction_v1(
                    &mut engine,
                    &prediction_input(
                        &format!("sample-{round}"),
                        &format!("event-{round}"),
                        round,
                        AciPredictionIntervalV1::bounded(9.0, 11.0),
                    ),
                )
                .unwrap();
            }
            engine.phase_alphas
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn delay_to_memory_ratio_validates_inputs() {
        assert_eq!(delay_to_memory_ratio_v1(5.0, 20.0).unwrap(), 0.25);
        assert!(delay_to_memory_ratio_v1(1.0, 0.0).is_err());
        assert!(delay_to_memory_ratio_v1(-1.0, 2.0).is_err());
    }

    #[test]
    fn round_sequence_and_single_prediction_per_round_are_enforced() {
        let mut engine = engine(2);
        begin_delayed_aci_round_v1(&mut engine, 0).unwrap();
        let input = prediction_input("sample-0", "event-0", 0, AciPredictionIntervalV1::bounded(9.0, 11.0));
        issue_delayed_aci_prediction_v1(&mut engine, &input).unwrap();
        assert!(issue_delayed_aci_prediction_v1(&mut engine, &input).is_err());
        assert!(begin_delayed_aci_round_v1(&mut engine, 2).is_err());
    }

    #[test]
    fn interval_alpha_semantics_are_fail_closed() {
        let mut engine = engine(1);
        begin_delayed_aci_round_v1(&mut engine, 0).unwrap();
        let invalid = prediction_input("sample-0", "event-0", 0, AciPredictionIntervalV1::full_real_line());
        assert!(issue_delayed_aci_prediction_v1(&mut engine, &invalid).is_err());
    }
}
