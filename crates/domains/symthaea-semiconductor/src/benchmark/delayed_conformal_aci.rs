// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fixed-horizon delayed Adaptive Conformal Inference (tau-DACI) with
//! evidence-qualified feedback.
//!
//! This follows the fixed-delay recursion of El Halabi & Brandt (2026).
//! Each delay phase has its own alpha state. Prediction intervals are frozen
//! before outcomes are delivered and are never rewritten. Missing or
//! unqualified feedback suspends the declared fixed-delay guarantee.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
    LabelEvidenceFrontierV1,
};

pub const DELAYED_ACI_ENGINE_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci-engine.v1";
pub const DELAYED_ACI_POLICY_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci-policy.v1";
pub const DELAYED_ACI_PREDICTION_SCHEMA_V1: &str =
    "luminous.semi-bench.delayed-aci-prediction.v1";
pub const DELAYED_ACI_DELIVERY_SCHEMA_V1: &str = "luminous.semi-bench.delayed-aci-delivery.v1";
pub const DELAYED_ACI_PREDICTION_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.delayed-aci-prediction-receipt.v1";
pub const DELAYED_ACI_UPDATE_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.delayed-aci-update-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelayedAciGuaranteeStateV1 {
    ActiveUnderDeclaredFixedDelay,
    SuspendedByUnqualifiedFeedback,
    SuspendedByProtocolViolation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelayedAciDispositionV1 {
    UpdatedCovered,
    UpdatedMissed,
    HeldUnknown,
    Blocked,
    Rejected,
    Duplicate,
    MissingPrediction,
    GuaranteeSuspended,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelayedAciReasonV1 {
    LabelNotCurrentK1,
    LabelNotEligible,
    DownstreamConsumptionNotAllowed,
    SourceAuthenticityAssertionMissing,
    LabelIdentityMismatch,
    LabelRevisionMismatch,
    MeasurementEventMismatch,
    LabelPayloadDigestMismatch,
    EligibilityReportDigestMismatch,
    ReportNumericValueInvalid,
    DeliveredValueMismatch,
    ClockDomainMismatch,
    LabelAvailabilityBeforeMeasurement,
    ReportBeforeLabelAvailability,
    DeliveryBeforeLabelAvailability,
    ReportUnavailableAtDelivery,
    FixedDelayRoundMismatch,
    FeedbackOrderMismatch,
    PredictionReceiptDigestMismatch,
    DuplicateAcceptedRevision,
    PendingPredictionMissing,
    PredictionIdMismatch,
    UpstreamAuthorityFlagSet,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub target_miscoverage: f64,
    pub learning_rate: f64,
    pub fixed_delay_rounds: u64,
    /// This boolean is only an upstream assertion requirement; it is not
    /// signature verification or issuer authentication.
    pub require_source_authenticity_assertion: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciPredictionInputV1 {
    pub schema: String,
    pub prediction_id: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub round_index: u64,
    pub event_time_unix_seconds: i64,
    pub feature_available_at_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciPredictionReceiptV1 {
    pub schema: String,
    pub prediction_id: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub round_index: u64,
    pub delay_phase: u64,
    pub alpha_used: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub expected_delivery_round: u64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub guarantee_state_at_prediction: DelayedAciGuaranteeStateV1,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciLabelDeliveryV1 {
    pub schema: String,
    pub prediction_id: String,
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub measurement_event_id: String,
    pub label_payload_sha256: String,
    pub label_value: f64,
    pub prediction_round: u64,
    pub delivery_round: u64,
    pub label_available_at_unix_seconds: i64,
    pub eligibility_report_available_at_unix_seconds: i64,
    pub delivered_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciUpdateReceiptV1 {
    pub schema: String,
    pub engine_id: String,
    pub prediction_id: String,
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub prediction_receipt_sha256: Option<String>,
    pub eligibility_report_sha256: Option<String>,
    pub disposition: DelayedAciDispositionV1,
    pub reasons: Vec<DelayedAciReasonV1>,
    pub coverage_error: Option<bool>,
    pub delay_phase: Option<u64>,
    pub alpha_before: Option<f64>,
    pub alpha_after: Option<f64>,
    pub guarantee_state_before: DelayedAciGuaranteeStateV1,
    pub guarantee_state_after: DelayedAciGuaranteeStateV1,
    pub alpha_state_mutated: bool,
    pub physical_source_authenticity_proven: bool,
    pub equipment_control_authority: bool,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayedAciEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub clock_domain_id: String,
    pub policy: DelayedAciPolicyV1,
    pub policy_sha256: String,
    /// One independent alpha state for each fixed-delay phase.
    pub phase_alpha: Vec<f64>,
    pub next_prediction_round: u64,
    pub next_feedback_prediction_round: u64,
    pub pending_predictions: BTreeMap<String, DelayedAciPredictionReceiptV1>,
    pub accepted_label_revisions: BTreeSet<String>,
    pub processed_prediction_ids: BTreeSet<String>,
    pub guarantee_state: DelayedAciGuaranteeStateV1,
    pub suspension_reasons: Vec<String>,
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

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, DelayedAciErrorV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| DelayedAciErrorV1(format!("canonical JSON serialization failed: {error}")))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn report_digest(report: &LabelConsumptionEligibilityReportV1) -> Result<String, DelayedAciErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn prediction_digest(receipt: &DelayedAciPredictionReceiptV1) -> Result<String, DelayedAciErrorV1> {
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
        || !policy.target_miscoverage.is_finite()
        || !(0.0..1.0).contains(&policy.target_miscoverage)
        || !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || policy.fixed_delay_rounds == u64::MAX
    {
        return Err(DelayedAciErrorV1("delayed ACI policy has invalid parameters".into()));
    }
    Ok(())
}

/// Creates tau-DACI state. Every delay phase starts at the same initial alpha.
pub fn create_delayed_aci_engine_v1(
    engine_id: &str,
    clock_domain_id: &str,
    initial_miscoverage: f64,
    policy: DelayedAciPolicyV1,
) -> Result<DelayedAciEngineV1, DelayedAciErrorV1> {
    require_text(engine_id, "engine_id")?;
    require_text(clock_domain_id, "clock_domain_id")?;
    validate_policy(&policy)?;
    if !initial_miscoverage.is_finite() || !(0.0..=1.0).contains(&initial_miscoverage) {
        return Err(DelayedAciErrorV1("initial miscoverage must be in [0, 1]".into()));
    }
    let phase_count = usize::try_from(policy.fixed_delay_rounds.max(1))
        .map_err(|_| DelayedAciErrorV1("delay phase count exceeds platform capacity".into()))?;
    let policy_sha256 = json_sha256(&policy)?;
    Ok(DelayedAciEngineV1 {
        schema: DELAYED_ACI_ENGINE_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        clock_domain_id: clock_domain_id.into(),
        policy,
        policy_sha256,
        phase_alpha: vec![initial_miscoverage; phase_count],
        next_prediction_round: 0,
        next_feedback_prediction_round: 0,
        pending_predictions: BTreeMap::new(),
        accepted_label_revisions: BTreeSet::new(),
        processed_prediction_ids: BTreeSet::new(),
        guarantee_state: DelayedAciGuaranteeStateV1::ActiveUnderDeclaredFixedDelay,
        suspension_reasons: Vec::new(),
    })
}

/// Returns the phase-specific alpha to use for the next prediction round.
pub fn delayed_aci_alpha_for_round_v1(
    engine: &DelayedAciEngineV1,
    round_index: u64,
) -> Result<f64, DelayedAciErrorV1> {
    if round_index != engine.next_prediction_round {
        return Err(DelayedAciErrorV1("prediction round is not the next expected round".into()));
    }
    let phase = (round_index % engine.phase_alpha.len() as u64) as usize;
    Ok(engine.phase_alpha[phase])
}

/// Registers the interval before its target label becomes available.
pub fn register_delayed_aci_prediction_v1(
    engine: &mut DelayedAciEngineV1,
    input: &DelayedAciPredictionInputV1,
) -> Result<DelayedAciPredictionReceiptV1, DelayedAciErrorV1> {
    if engine.schema != DELAYED_ACI_ENGINE_SCHEMA_V1 || input.schema != DELAYED_ACI_PREDICTION_SCHEMA_V1 {
        return Err(DelayedAciErrorV1("delayed ACI engine or prediction schema is unsupported".into()));
    }
    if json_sha256(&engine.policy)? != engine.policy_sha256 {
        return Err(DelayedAciErrorV1("delayed ACI policy digest mismatch".into()));
    }
    require_text(&input.prediction_id, "prediction_id")?;
    require_text(&input.sample_id, "sample_id")?;
    require_text(&input.measurement_event_id, "measurement_event_id")?;
    require_text(&input.clock_domain_id, "clock_domain_id")?;
    if input.clock_domain_id != engine.clock_domain_id
        || !valid_sha256(&input.global_model_artifact_sha256)
        || input.event_time_unix_seconds > input.feature_available_at_unix_seconds
        || input.feature_available_at_unix_seconds > input.prediction_at_unix_seconds
        || !input.lower_bound.is_finite()
        || !input.upper_bound.is_finite()
        || input.lower_bound > input.upper_bound
        || input.round_index != engine.next_prediction_round
        || engine.pending_predictions.contains_key(&input.prediction_id)
        || engine.processed_prediction_ids.contains(&input.prediction_id)
    {
        return Err(DelayedAciErrorV1("prediction identity, digest, chronology or bounds are invalid".into()));
    }
    let expected_delivery_round = input.round_index
        .checked_add(engine.policy.fixed_delay_rounds)
        .ok_or_else(|| DelayedAciErrorV1("expected delivery round overflow".into()))?;
    let next_round = engine.next_prediction_round
        .checked_add(1)
        .ok_or_else(|| DelayedAciErrorV1("prediction round overflow".into()))?;
    let phase_count = engine.phase_alpha.len() as u64;
    let delay_phase = input.round_index % phase_count;
    let alpha_used = engine.phase_alpha[delay_phase as usize];
    let mut receipt = DelayedAciPredictionReceiptV1 {
        schema: DELAYED_ACI_PREDICTION_RECEIPT_SCHEMA_V1.into(),
        prediction_id: input.prediction_id.clone(),
        sample_id: input.sample_id.clone(),
        measurement_event_id: input.measurement_event_id.clone(),
        round_index: input.round_index,
        delay_phase,
        alpha_used,
        lower_bound: input.lower_bound,
        upper_bound: input.upper_bound,
        event_time_unix_seconds: input.event_time_unix_seconds,
        prediction_at_unix_seconds: input.prediction_at_unix_seconds,
        expected_delivery_round,
        clock_domain_id: input.clock_domain_id.clone(),
        global_model_artifact_sha256: input.global_model_artifact_sha256.clone(),
        guarantee_state_at_prediction: engine.guarantee_state,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = prediction_digest(&receipt)?;
    engine.pending_predictions.insert(input.prediction_id.clone(), receipt.clone());
    engine.next_prediction_round = next_round;
    Ok(receipt)
}

fn build_update_receipt(
    engine: &DelayedAciEngineV1,
    delivery: &DelayedAciLabelDeliveryV1,
    prediction: Option<&DelayedAciPredictionReceiptV1>,
    disposition: DelayedAciDispositionV1,
    reasons: Vec<DelayedAciReasonV1>,
    coverage_error: Option<bool>,
    alpha: Option<(f64, f64)>,
    guarantee_before: DelayedAciGuaranteeStateV1,
    guarantee_after: DelayedAciGuaranteeStateV1,
    mutated: bool,
) -> Result<DelayedAciUpdateReceiptV1, DelayedAciErrorV1> {
    let mut receipt = DelayedAciUpdateReceiptV1 {
        schema: DELAYED_ACI_UPDATE_RECEIPT_SCHEMA_V1.into(),
        engine_id: engine.engine_id.clone(),
        prediction_id: delivery.prediction_id.clone(),
        sample_id: delivery.sample_id.clone(),
        label_id: delivery.label_id.clone(),
        label_revision: delivery.label_revision,
        prediction_receipt_sha256: prediction.map(|value| value.receipt_sha256.clone()),
        eligibility_report_sha256: Some(delivery.eligibility_report.report_sha256.clone()),
        disposition,
        reasons,
        coverage_error,
        delay_phase: prediction.map(|value| value.delay_phase),
        alpha_before: alpha.map(|value| value.0),
        alpha_after: alpha.map(|value| value.1),
        guarantee_state_before: guarantee_before,
        guarantee_state_after: guarantee_after,
        alpha_state_mutated: mutated,
        physical_source_authenticity_proven: false,
        equipment_control_authority: false,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = update_digest(&receipt)?;
    Ok(receipt)
}

fn suspend(engine: &mut DelayedAciEngineV1, reason: &str) {
    engine.guarantee_state = DelayedAciGuaranteeStateV1::SuspendedByUnqualifiedFeedback;
    engine.suspension_reasons.push(reason.to_owned());
}

/// Accepts feedback only in the declared fixed-delay, chronological order.
/// Any ineligible/missing/late/mismatched outcome suspends the guarantee.
pub fn deliver_delayed_aci_label_v1(
    engine: &mut DelayedAciEngineV1,
    delivery: &DelayedAciLabelDeliveryV1,
) -> Result<DelayedAciUpdateReceiptV1, DelayedAciErrorV1> {
    require_text(&delivery.prediction_id, "delivery.prediction_id")?;
    require_text(&delivery.sample_id, "delivery.sample_id")?;
    require_text(&delivery.label_id, "delivery.label_id")?;
    require_text(&delivery.measurement_event_id, "delivery.measurement_event_id")?;
    if delivery.label_revision == 0 || !valid_sha256(&delivery.label_payload_sha256) {
        return Err(DelayedAciErrorV1("label revision or payload digest is invalid".into()));
    }

    let guarantee_before = engine.guarantee_state;
    let prediction = engine.pending_predictions.get(&delivery.prediction_id).cloned();
    let revision_key = format!("{}:{}", delivery.label_id, delivery.label_revision);
    if engine.accepted_label_revisions.contains(&revision_key) {
        return build_update_receipt(
            engine, delivery, prediction.as_ref(), DelayedAciDispositionV1::Duplicate,
            vec![DelayedAciReasonV1::DuplicateAcceptedRevision], None, None,
            guarantee_before, engine.guarantee_state, false,
        );
    }
    let Some(prediction) = prediction else {
        suspend(engine, "pending_prediction_missing");
        return build_update_receipt(
            engine, delivery, None, DelayedAciDispositionV1::MissingPrediction,
            vec![DelayedAciReasonV1::PendingPredictionMissing], None, None,
            guarantee_before, engine.guarantee_state, false,
        );
    };

    let report = &delivery.eligibility_report;
    let mut reasons = Vec::new();
    if prediction.receipt_sha256 != prediction_digest(&prediction)? {
        reasons.push(DelayedAciReasonV1::PredictionReceiptDigestMismatch);
    }
    if delivery.sample_id != prediction.sample_id || delivery.prediction_id != prediction.prediction_id {
        reasons.push(DelayedAciReasonV1::PredictionIdMismatch);
    }
    if delivery.measurement_event_id != prediction.measurement_event_id {
        reasons.push(DelayedAciReasonV1::MeasurementEventMismatch);
    }
    if delivery.prediction_round != prediction.round_index
        || delivery.delivery_round != prediction.expected_delivery_round
        || delivery.delivery_round != engine.next_feedback_prediction_round
            .checked_add(engine.policy.fixed_delay_rounds).unwrap_or(u64::MAX)
    {
        reasons.push(DelayedAciReasonV1::FixedDelayRoundMismatch);
    }
    if prediction.round_index != engine.next_feedback_prediction_round {
        reasons.push(DelayedAciReasonV1::FeedbackOrderMismatch);
    }
    if delivery.clock_domain_id != prediction.clock_domain_id || delivery.clock_domain_id != engine.clock_domain_id {
        reasons.push(DelayedAciReasonV1::ClockDomainMismatch);
    }
    if delivery.label_available_at_unix_seconds < prediction.event_time_unix_seconds {
        reasons.push(DelayedAciReasonV1::LabelAvailabilityBeforeMeasurement);
    }
    if delivery.eligibility_report_available_at_unix_seconds < delivery.label_available_at_unix_seconds {
        reasons.push(DelayedAciReasonV1::ReportBeforeLabelAvailability);
    }
    if delivery.delivered_at_unix_seconds < delivery.label_available_at_unix_seconds {
        reasons.push(DelayedAciReasonV1::DeliveryBeforeLabelAvailability);
    }
    if delivery.delivered_at_unix_seconds < delivery.eligibility_report_available_at_unix_seconds {
        reasons.push(DelayedAciReasonV1::ReportUnavailableAtDelivery);
    }
    if !valid_sha256(&report.label_sha256) || report.label_sha256 != delivery.label_payload_sha256 {
        reasons.push(DelayedAciReasonV1::LabelPayloadDigestMismatch);
    }
    if report_digest(report)? != report.report_sha256 {
        reasons.push(DelayedAciReasonV1::EligibilityReportDigestMismatch);
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
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        reasons.push(DelayedAciReasonV1::LabelNotCurrentK1);
    }
    if !delivery.label_value.is_finite()
        || !report.qualified_corrected_value.is_finite()
        || report.qualified_standard_uncertainty_upper_bound.is_none()
    {
        reasons.push(DelayedAciReasonV1::ReportNumericValueInvalid);
    }
    if report.qualified_corrected_value != delivery.label_value {
        reasons.push(DelayedAciReasonV1::DeliveredValueMismatch);
    }
    if engine.policy.require_source_authenticity_assertion && !report.source_authenticity_proven {
        reasons.push(DelayedAciReasonV1::SourceAuthenticityAssertionMissing);
    }
    if report.training_snapshot_mutated || report.equipment_control_authority {
        reasons.push(DelayedAciReasonV1::UpstreamAuthorityFlagSet);
    }
    match report.eligibility {
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy if report.downstream_consumption_allowed => {}
        LabelConsumptionEligibilityV1::Unknown | LabelConsumptionEligibilityV1::Blocked => {
            reasons.push(DelayedAciReasonV1::LabelNotEligible);
        }
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => {
            reasons.push(DelayedAciReasonV1::DownstreamConsumptionNotAllowed);
        }
    }
    reasons.sort();
    reasons.dedup();

    if !reasons.is_empty() {
        let disposition = match report.eligibility {
            LabelConsumptionEligibilityV1::Unknown => DelayedAciDispositionV1::HeldUnknown,
            LabelConsumptionEligibilityV1::Blocked => DelayedAciDispositionV1::Blocked,
            _ => DelayedAciDispositionV1::Rejected,
        };
        suspend(engine, "feedback_not_usable_under_fixed_delay_contract");
        return build_update_receipt(
            engine, delivery, Some(&prediction), disposition, reasons, None, None,
            guarantee_before, engine.guarantee_state, false,
        );
    }
    if engine.guarantee_state != DelayedAciGuaranteeStateV1::ActiveUnderDeclaredFixedDelay {
        return build_update_receipt(
            engine, delivery, Some(&prediction), DelayedAciDispositionV1::GuaranteeSuspended,
            vec![DelayedAciReasonV1::FeedbackOrderMismatch], None, None,
            guarantee_before, engine.guarantee_state, false,
        );
    }

    let coverage_error = delivery.label_value < prediction.lower_bound
        || delivery.label_value > prediction.upper_bound;
    let alpha_before = prediction.alpha_used;
    let loss = if coverage_error { 1.0 } else { 0.0 };
    let alpha_after = (alpha_before + engine.policy.learning_rate * (engine.policy.target_miscoverage - loss))
        .clamp(0.0, 1.0);
    let receipt = build_update_receipt(
        engine, delivery, Some(&prediction),
        if coverage_error { DelayedAciDispositionV1::UpdatedMissed } else { DelayedAciDispositionV1::UpdatedCovered },
        Vec::new(), Some(coverage_error), Some((alpha_before, alpha_after)),
        guarantee_before, guarantee_before, true,
    )?;

    let phase = prediction.delay_phase as usize;
    engine.phase_alpha[phase] = alpha_after;
    engine.accepted_label_revisions.insert(revision_key);
    engine.processed_prediction_ids.insert(prediction.prediction_id.clone());
    engine.pending_predictions.remove(&delivery.prediction_id);
    engine.next_feedback_prediction_round = engine.next_feedback_prediction_round
        .checked_add(1).ok_or_else(|| DelayedAciErrorV1("feedback round overflow".into()))?;
    Ok(receipt)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualMemoryBasisV1 {
    ResidualLevel,
    SquaredResidual,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelayToMemoryDiagnosticV1 {
    pub basis: ResidualMemoryBasisV1,
    pub sample_count: usize,
    pub estimated_ar1_phi: Option<f64>,
    pub memory_length_rounds: Option<f64>,
    pub delay_rounds: u64,
    pub delay_to_memory_ratio: Option<f64>,
    pub remaining_ar1_correlation_at_delay: Option<f64>,
}

/// Estimates a positive-AR(1) e-folding memory diagnostic. It does not infer a
/// meaningful memory length for non-positive or non-decaying estimates.
pub fn estimate_delay_to_memory_v1(
    residuals: &[f64],
    basis: ResidualMemoryBasisV1,
    delay_rounds: u64,
) -> Result<DelayToMemoryDiagnosticV1, DelayedAciErrorV1> {
    if residuals.len() < 3 || residuals.iter().any(|value| !value.is_finite()) {
        return Err(DelayedAciErrorV1("at least three finite residuals are required".into()));
    }
    let values: Vec<f64> = match basis {
        ResidualMemoryBasisV1::ResidualLevel => residuals.to_vec(),
        ResidualMemoryBasisV1::SquaredResidual => residuals.iter().map(|value| value * value).collect(),
    };
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let denominator = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>();
    let numerator = values.windows(2)
        .map(|pair| (pair[0] - mean) * (pair[1] - mean))
        .sum::<f64>();
    let phi = (denominator > f64::EPSILON).then(|| (numerator / denominator).clamp(-1.0, 1.0));
    let memory = phi.filter(|value| *value > 0.0 && *value < 1.0).map(|value| -1.0 / value.ln());
    let ratio = memory.map(|length| delay_rounds as f64 / length);
    let remaining = phi.map(|value| value.powf(delay_rounds as f64));
    Ok(DelayToMemoryDiagnosticV1 {
        basis, sample_count: residuals.len(), estimated_ar1_phi: phi,
        memory_length_rounds: memory, delay_rounds,
        delay_to_memory_ratio: ratio, remaining_ar1_correlation_at_delay: remaining,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String { std::iter::repeat_n(ch, 64).collect() }

    fn policy(delay: u64) -> DelayedAciPolicyV1 {
        DelayedAciPolicyV1 {
            schema: DELAYED_ACI_POLICY_SCHEMA_V1.into(),
            policy_id: "semi-delayed-aci.v1".into(),
            revision: 1, target_miscoverage: 0.05, learning_rate: 0.10,
            fixed_delay_rounds: delay, require_source_authenticity_assertion: false,
        }
    }

    fn engine(delay: u64) -> DelayedAciEngineV1 {
        create_delayed_aci_engine_v1("aci-engine", "fab-clock", 0.10, policy(delay)).unwrap()
    }

    fn prediction(round: u64, lower: f64, upper: f64) -> DelayedAciPredictionInputV1 {
        DelayedAciPredictionInputV1 {
            schema: DELAYED_ACI_PREDICTION_SCHEMA_V1.into(),
            prediction_id: format!("prediction-{round}"), sample_id: format!("sample-{round}"),
            measurement_event_id: format!("event-{round}"), round_index: round,
            event_time_unix_seconds: 1_800_000_000 + round as i64 * 60,
            feature_available_at_unix_seconds: 1_800_000_001 + round as i64 * 60,
            prediction_at_unix_seconds: 1_800_000_002 + round as i64 * 60,
            clock_domain_id: "fab-clock".into(), global_model_artifact_sha256: digest('a'),
            lower_bound: lower, upper_bound: upper,
        }
    }

    fn report(
        label: &str, revision: u64, event: &str, value: f64, payload: &str,
        eligibility: LabelConsumptionEligibilityV1,
    ) -> LabelConsumptionEligibilityReportV1 {
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label.into(), label_revision: revision, measurement_event_id: event.into(),
            calibration_certificate_id: "cal-1".into(), frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility, downstream_consumption_allowed: eligibility == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(), label_sha256: payload.into(), measurement_sha256: digest('b'),
            qualified_corrected_value: value, qualified_standard_uncertainty_upper_bound: Some(0.01),
            qualification_report_sha256: digest('c'), requalification_report_sha256: digest('d'),
            requalification_frontier_sha256: digest('e'), source_authenticity_proven: false,
            training_snapshot_mutated: false, equipment_control_authority: false, report_sha256: String::new(),
        };
        let mut canonical = report.clone();
        canonical.report_sha256.clear();
        report.report_sha256 = json_sha256(&canonical).unwrap();
        report
    }

    fn delivery(round: u64, value: f64, eligibility: LabelConsumptionEligibilityV1, delivery_round: u64) -> DelayedAciLabelDeliveryV1 {
        let payload = digest('f');
        DelayedAciLabelDeliveryV1 {
            schema: DELAYED_ACI_DELIVERY_SCHEMA_V1.into(), prediction_id: format!("prediction-{round}"),
            sample_id: format!("sample-{round}"), label_id: format!("label-{round}"), label_revision: 1,
            measurement_event_id: format!("event-{round}"), label_payload_sha256: payload.clone(),
            label_value: value, prediction_round: round, delivery_round,
            label_available_at_unix_seconds: 1_800_000_120 + round as i64 * 60,
            eligibility_report_available_at_unix_seconds: 1_800_000_121 + round as i64 * 60,
            delivered_at_unix_seconds: 1_800_000_122 + round as i64 * 60, clock_domain_id: "fab-clock".into(),
            eligibility_report: report(&format!("label-{round}"), 1, &format!("event-{round}"), value, &payload, eligibility),
        }
    }

    #[test]
    fn fixed_delay_uses_independent_interleaved_alpha_phases() {
        let mut engine = engine(2);
        for round in 0..2 {
            register_delayed_aci_prediction_v1(&mut engine, &prediction(round, 0.0, 1.0)).unwrap();
        }
        let first = deliver_delayed_aci_label_v1(&mut engine, &delivery(0, 2.0, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 2)).unwrap();
        assert_eq!(first.disposition, DelayedAciDispositionV1::UpdatedMissed);
        assert!((engine.phase_alpha[0] - 0.005).abs() < 1e-12);
        assert!((engine.phase_alpha[1] - 0.10).abs() < 1e-12);
        let second = deliver_delayed_aci_label_v1(&mut engine, &delivery(1, 0.5, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 3)).unwrap();
        assert_eq!(second.disposition, DelayedAciDispositionV1::UpdatedCovered);
        assert!((engine.phase_alpha[1] - 0.105).abs() < 1e-12);
    }

    #[test]
    fn undercoverage_reduces_alpha_and_overcoverage_increases_it() {
        let mut engine = engine(0);
        register_delayed_aci_prediction_v1(&mut engine, &prediction(0, 0.0, 1.0)).unwrap();
        let missed = deliver_delayed_aci_label_v1(&mut engine, &delivery(0, 2.0, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 0)).unwrap();
        assert_eq!(missed.alpha_after, Some(0.005));

        register_delayed_aci_prediction_v1(&mut engine, &prediction(1, 0.0, 1.0)).unwrap();
        let covered = deliver_delayed_aci_label_v1(&mut engine, &delivery(1, 0.5, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 1)).unwrap();
        assert_eq!(covered.alpha_after, Some(0.105));
    }

    #[test]
    fn unknown_feedback_suspends_fixed_delay_guarantee_without_alpha_mutation() {
        let mut engine = engine(1);
        register_delayed_aci_prediction_v1(&mut engine, &prediction(0, 0.0, 1.0)).unwrap();
        let before = engine.phase_alpha.clone();
        let result = deliver_delayed_aci_label_v1(&mut engine, &delivery(0, 0.5, LabelConsumptionEligibilityV1::Unknown, 1)).unwrap();
        assert_eq!(result.disposition, DelayedAciDispositionV1::HeldUnknown);
        assert_eq!(engine.phase_alpha, before);
        assert_eq!(engine.guarantee_state, DelayedAciGuaranteeStateV1::SuspendedByUnqualifiedFeedback);
    }

    #[test]
    fn blocked_feedback_never_updates() {
        let mut engine = engine(1);
        register_delayed_aci_prediction_v1(&mut engine, &prediction(0, 0.0, 1.0)).unwrap();
        let before = engine.phase_alpha.clone();
        let result = deliver_delayed_aci_label_v1(&mut engine, &delivery(0, 0.5, LabelConsumptionEligibilityV1::Blocked, 1)).unwrap();
        assert_eq!(result.disposition, DelayedAciDispositionV1::Blocked);
        assert_eq!(engine.phase_alpha, before);
    }

    #[test]
    fn wrong_delivery_round_suspends_fixed_delay_assumption() {
        let mut engine = engine(2);
        register_delayed_aci_prediction_v1(&mut engine, &prediction(0, 0.0, 1.0)).unwrap();
        let result = deliver_delayed_aci_label_v1(&mut engine, &delivery(0, 0.5, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 3)).unwrap();
        assert_eq!(result.disposition, DelayedAciDispositionV1::Rejected);
        assert_eq!(engine.phase_alpha[0], 0.10);
        assert_eq!(engine.guarantee_state, DelayedAciGuaranteeStateV1::SuspendedByUnqualifiedFeedback);
    }

    #[test]
    fn duplicate_label_revision_is_idempotent() {
        let mut engine = engine(0);
        register_delayed_aci_prediction_v1(&mut engine, &prediction(0, 0.0, 1.0)).unwrap();
        let input = delivery(0, 0.5, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 0);
        assert_eq!(deliver_delayed_aci_label_v1(&mut engine, &input).unwrap().disposition, DelayedAciDispositionV1::UpdatedCovered);
        assert_eq!(deliver_delayed_aci_label_v1(&mut engine, &input).unwrap().disposition, DelayedAciDispositionV1::Duplicate);
    }

    #[test]
    fn revision_payload_and_identity_mismatches_are_rejected() {
        let mut engine = engine(0);
        register_delayed_aci_prediction_v1(&mut engine, &prediction(0, 0.0, 1.0)).unwrap();
        let mut input = delivery(0, 0.5, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 0);
        input.label_revision = 2;
        let result = deliver_delayed_aci_label_v1(&mut engine, &input).unwrap();
        assert_eq!(result.disposition, DelayedAciDispositionV1::Rejected);
        assert_eq!(engine.phase_alpha[0], 0.10);
    }

    #[test]
    fn prediction_receipt_digest_is_immutable() {
        let mut engine = engine(0);
        let receipt = register_delayed_aci_prediction_v1(&mut engine, &prediction(0, 0.0, 1.0)).unwrap();
        let mut altered = receipt.clone();
        altered.upper_bound = 0.9;
        assert_ne!(receipt.receipt_sha256, prediction_digest(&altered).unwrap());
    }

    #[test]
    fn delay_to_memory_estimates_positive_ar1_persistence() {
        let mut residuals = Vec::new();
        let mut value = 1.0;
        for _ in 0..200 {
            residuals.push(value);
            value *= 0.90;
        }
        let diagnostic = estimate_delay_to_memory_v1(&residuals, ResidualMemoryBasisV1::ResidualLevel, 5).unwrap();
        let phi = diagnostic.estimated_ar1_phi.unwrap();
        assert!(phi > 0.89 && phi < 0.91);
        assert!(diagnostic.memory_length_rounds.unwrap() > 9.0);
        assert!(diagnostic.delay_to_memory_ratio.unwrap() > 0.4);
    }
}
