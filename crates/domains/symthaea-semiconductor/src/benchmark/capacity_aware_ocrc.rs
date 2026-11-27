// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capacity-aware online conformal risk control for arbitrary bounded delays.
//!
//! This is the deterministic-delay CA-OCRC protocol. It separates physical
//! metrology acquisition from the finite capacity to retain pending feedback.
//! All feedback updates require a current K1 numeric-bound label receipt.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::{
    LABEL_CONSUMPTION_REPORT_SCHEMA_V1, LabelConsumptionEligibilityReportV1,
    LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};

pub const CA_OCRC_POLICY_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-policy.v1";
pub const CA_OCRC_ENGINE_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-engine.v1";
pub const CA_OCRC_ROUND_INPUT_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-round-input.v1";
pub const CA_OCRC_FEEDBACK_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-feedback.v1";
pub const CA_OCRC_ROUND_RECEIPT_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-round-receipt.v1";
pub const CA_OCRC_FEEDBACK_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.ca-ocrc-feedback-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaoOcrcTrivialSetV1 {
    Empty,
    Full,
}

impl CaoOcrcTrivialSetV1 {
    fn known_loss(self) -> f64 {
        match self {
            Self::Empty => 1.0,
            Self::Full => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaoOcrcAdmissionV1 {
    NotNeededTrivialSet,
    ProxyNotTracked,
    CapacityAdmitted,
    CapacitySaturated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaoOcrcFeedbackDispositionV1 {
    ReceivedAndApplied,
    HeldUnknown,
    Blocked,
    Rejected,
    Duplicate,
    NotTracked,
    AlreadyExpired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaoOcrcFeedbackReasonV1 {
    LabelNotCurrentK1,
    LabelNotEligible,
    DownstreamConsumptionNotAllowed,
    LabelIdentityMismatch,
    LabelRevisionMismatch,
    MeasurementEventMismatch,
    LabelPayloadDigestMismatch,
    EligibilityReportDigestMismatch,
    DeliveredValueNonFinite,
    DeliveredValueMismatch,
    ClockDomainMismatch,
    LabelAvailableBeforeMeasurement,
    ReportBeforeLabelAvailability,
    ReportUnavailableAtDelivery,
    DeliveryRoundMismatch,
    FeedbackDelayOutOfPolicy,
    ProxySurvivalZero,
    DuplicateFeedback,
    PredictionReceiptMismatch,
    ScoreRangeViolation,
    PredictionIntervalInvalid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaoOcrcPopulationScopeV1 {
    QualifiedMeasuredStreamOnly,
    FullPredictionPopulation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaoOcrcGuaranteeStatusV1 {
    ConditionalOnDeclaredPaperAssumptions,
    InvalidatedByK1QualificationFiltering,
    InvalidatedByFeedbackDelayBound,
    InvalidatedByMalformedFeedbackEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoOcrcPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub target_risk: f64,
    pub learning_rate: f64,
    pub score_upper_bound: f64,
    pub initial_threshold: f64,
    pub tracking_capacity: usize,
    pub maximum_feedback_delay_rounds: u64,
    /// Declared estimand population. This is metadata, not proof that the
    /// selected sequence represents all production wafers.
    pub population_scope: CaoOcrcPopulationScopeV1,
    /// Survival values F(a)=P[proxy_delay >= a] for a=0..d_max.
    /// F(-1)=1 is implicit.
    pub proxy_survival: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoOcrcRoundInputV1 {
    pub schema: String,
    pub round_index: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub interval_policy_sha256: String,
    pub upstream_prediction_receipt_sha256: String,
    pub threshold_used: f64,
    /// Regression score center for the bounded residual score |y - center|.
    pub score_location: f64,
    /// None means a nontrivial interval that requires physical feedback.
    /// Empty/full sets have known loss and consume no tracking slot.
    pub trivial_set: Option<CaoOcrcTrivialSetV1>,
    pub interval_lower: Option<f64>,
    pub interval_upper: Option<f64>,
    /// Recorded U[0,1) variate from the caller's declared seeded PRNG.
    /// Required only for nontrivial prediction sets.
    pub proxy_uniform_draw: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoOcrcRoundRecordV1 {
    pub round_index: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub interval_policy_sha256: String,
    pub upstream_prediction_receipt_sha256: String,
    pub threshold_at_prediction: f64,
    pub score_location: f64,
    pub trivial_set: Option<CaoOcrcTrivialSetV1>,
    pub interval_lower: Option<f64>,
    pub interval_upper: Option<f64>,
    pub prediction_receipt_sha256: String,
    pub round_record_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoOcrcTrackedRoundV1 {
    pub round_index: u64,
    pub proxy_delay_rounds: u64,
    pub proxy_expiry_round: u64,
    pub proxy_uniform_draw: f64,
    pub prediction_receipt_sha256: String,
    pub round_record_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoOcrcFeedbackInputV1 {
    pub schema: String,
    pub delivery_round: u64,
    pub originating_round: u64,
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub measurement_event_id: String,
    pub label_payload_sha256: String,
    pub corrected_measurement_value: f64,
    pub label_available_at_unix_seconds: i64,
    pub eligibility_report_available_at_unix_seconds: i64,
    pub delivered_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub idempotency_key: String,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoOcrcFeedbackReceiptV1 {
    pub schema: String,
    pub delivery_round: u64,
    pub originating_round: u64,
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub disposition: CaoOcrcFeedbackDispositionV1,
    pub reason: Option<CaoOcrcFeedbackReasonV1>,
    pub actual_delay_rounds: Option<u64>,
    pub proxy_delay_rounds: Option<u64>,
    pub inverse_proxy_survival_weight: Option<f64>,
    pub computed_loss: Option<f64>,
    pub prediction_receipt_sha256: Option<String>,
    pub eligibility_report_sha256: String,
    pub label_payload_sha256: String,
    pub threshold_before_sha256: String,
    pub threshold_after_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoOcrcRoundReceiptV1 {
    pub schema: String,
    pub round_index: u64,
    pub sample_id: String,
    pub threshold_before: f64,
    pub threshold_after: f64,
    pub population_scope: CaoOcrcPopulationScopeV1,
    pub guarantee_status: CaoOcrcGuaranteeStatusV1,
    pub trivial_loss: Option<f64>,
    pub proxy_uniform_draw: Option<f64>,
    pub selected_proxy_delay_rounds: i64,
    pub admission: CaoOcrcAdmissionV1,
    pub tracking_count_after_admission: usize,
    pub evicted_round_ids: Vec<u64>,
    pub expired_feedback_round_ids: Vec<u64>,
    pub received_feedback_receipts: Vec<CaoOcrcFeedbackReceiptV1>,
    pub policy_sha256: String,
    pub tracking_state_before_sha256: String,
    pub tracking_state_after_sha256: String,
    pub round_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoOcrcEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub clock_domain_id: String,
    pub policy: CaoOcrcPolicyV1,
    pub policy_sha256: String,
    pub proxy_survival_sha256: String,
    pub threshold: f64,
    pub guarantee_status: CaoOcrcGuaranteeStatusV1,
    pub next_round_index: u64,
    pub tracked_rounds: BTreeMap<u64, CaoOcrcTrackedRoundV1>,
    /// Retained immutable prediction evidence, including non-admitted rounds,
    /// so late outcomes can be classified as lost rather than misapplied.
    pub round_ledger: BTreeMap<u64, CaoOcrcRoundRecordV1>,
    pub accepted_label_revisions: BTreeSet<String>,
    pub seen_feedback_idempotency_keys: BTreeSet<String>,
    pub seen_sample_ids: BTreeSet<String>,
    pub seen_measurement_event_ids: BTreeSet<String>,
    pub lost_feedback_round_ids: BTreeSet<u64>,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct CaoOcrcErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), CaoOcrcErrorV1> {
    if value.trim().is_empty() {
        return Err(CaoOcrcErrorV1(format!("{field}: expected non-empty text")));
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

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, CaoOcrcErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        CaoOcrcErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn policy_digest(policy: &CaoOcrcPolicyV1) -> Result<String, CaoOcrcErrorV1> {
    json_sha256(policy)
}

fn engine_tracking_digest(
    tracked: &BTreeMap<u64, CaoOcrcTrackedRoundV1>,
    threshold: f64,
) -> Result<String, CaoOcrcErrorV1> {
    json_sha256(&(tracked, threshold))
}

fn eligibility_report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, CaoOcrcErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn round_record_digest(
    record: &CaoOcrcRoundRecordV1,
) -> Result<String, CaoOcrcErrorV1> {
    let mut canonical = record.clone();
    canonical.round_record_sha256.clear();
    json_sha256(&canonical)
}

fn feedback_receipt_digest(
    receipt: &CaoOcrcFeedbackReceiptV1,
) -> Result<String, CaoOcrcErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn round_receipt_digest(
    receipt: &CaoOcrcRoundReceiptV1,
) -> Result<String, CaoOcrcErrorV1> {
    let mut canonical = receipt.clone();
    canonical.round_receipt_sha256.clear();
    json_sha256(&canonical)
}

fn validate_proxy_survival(values: &[f64], d_max: u64) -> Result<(), CaoOcrcErrorV1> {
    if values.len() != d_max as usize + 1 {
        return Err(CaoOcrcErrorV1(
            "proxy survival vector length must equal d_max + 1".into(),
        ));
    }
    if values.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value)) {
        return Err(CaoOcrcErrorV1(
            "proxy survival values must be finite and in [0,1]".into(),
        ));
    }
    if values.windows(2).any(|window| window[0] < window[1]) {
        return Err(CaoOcrcErrorV1(
            "proxy survival vector must be monotonically non-increasing".into(),
        ));
    }
    if values.last().copied().unwrap_or(0.0) <= 0.0 {
        return Err(CaoOcrcErrorV1(
            "F(d_max) must be positive to support every bounded actual delay".into(),
        ));
    }
    Ok(())
}

fn validate_policy(policy: &CaoOcrcPolicyV1) -> Result<(), CaoOcrcErrorV1> {
    if policy.schema != CA_OCRC_POLICY_SCHEMA_V1 {
        return Err(CaoOcrcErrorV1("CA-OCRC policy schema is unsupported".into()));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if !policy.target_risk.is_finite()
        || !(0.0..1.0).contains(&policy.target_risk)
        || !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || !policy.score_upper_bound.is_finite()
        || policy.score_upper_bound <= 0.0
        || !policy.initial_threshold.is_finite()
        || !(0.0..=policy.score_upper_bound).contains(&policy.initial_threshold)
    {
        return Err(CaoOcrcErrorV1("CA-OCRC policy parameters are invalid".into()));
    }
    validate_proxy_survival(&policy.proxy_survival, policy.maximum_feedback_delay_rounds)
}

fn sample_proxy_delay(survival: &[f64], uniform: f64) -> Result<i64, CaoOcrcErrorV1> {
    if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
        return Err(CaoOcrcErrorV1("proxy uniform draw must be in [0,1)".into()));
    }
    let d_max = survival.len() - 1;
    let mut cumulative = 1.0 - survival[0];
    if uniform < cumulative {
        return Ok(-1);
    }
    for delay in 0..=d_max {
        let probability = if delay == d_max {
            survival[delay]
        } else {
            survival[delay] - survival[delay + 1]
        };
        cumulative += probability;
        if uniform < cumulative || (delay == d_max && uniform <= 1.0) {
            return Ok(delay as i64);
        }
    }
    Err(CaoOcrcErrorV1("proxy delay distribution did not cover unit interval".into()))
}

fn validate_engine(engine: &CaoOcrcEngineV1) -> Result<(), CaoOcrcErrorV1> {
    if engine.schema != CA_OCRC_ENGINE_SCHEMA_V1 {
        return Err(CaoOcrcErrorV1("CA-OCRC engine schema is unsupported".into()));
    }
    validate_policy(&engine.policy)?;
    if policy_digest(&engine.policy)? != engine.policy_sha256
        || json_sha256(&engine.policy.proxy_survival)? != engine.proxy_survival_sha256
        || !engine.threshold.is_finite()
        || engine.tracked_rounds.len() > engine.policy.tracking_capacity
    {
        return Err(CaoOcrcErrorV1("CA-OCRC engine state or policy digest is invalid".into()));
    }
    Ok(())
}

pub fn create_ca_ocrc_engine_v1(
    engine_id: &str,
    clock_domain_id: &str,
    policy: CaoOcrcPolicyV1,
) -> Result<CaoOcrcEngineV1, CaoOcrcErrorV1> {
    require_text(engine_id, "engine_id")?;
    require_text(clock_domain_id, "clock_domain_id")?;
    validate_policy(&policy)?;
    let policy_sha256 = policy_digest(&policy)?;
    let proxy_survival_sha256 = json_sha256(&policy.proxy_survival)?;
    Ok(CaoOcrcEngineV1 {
        schema: CA_OCRC_ENGINE_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        clock_domain_id: clock_domain_id.into(),
        threshold: policy.initial_threshold,
        guarantee_status: CaoOcrcGuaranteeStatusV1::ConditionalOnDeclaredPaperAssumptions,
        policy,
        policy_sha256,
        proxy_survival_sha256,
        next_round_index: 1,
        tracked_rounds: BTreeMap::new(),
        round_ledger: BTreeMap::new(),
        accepted_label_revisions: BTreeSet::new(),
        seen_feedback_idempotency_keys: BTreeSet::new(),
        seen_sample_ids: BTreeSet::new(),
        seen_measurement_event_ids: BTreeSet::new(),
        lost_feedback_round_ids: BTreeSet::new(),
    })
}

fn make_feedback_receipt(
    feedback: &CaoOcrcFeedbackInputV1,
    disposition: CaoOcrcFeedbackDispositionV1,
    reason: Option<CaoOcrcFeedbackReasonV1>,
    actual_delay: Option<u64>,
    proxy_delay: Option<u64>,
    inverse_weight: Option<f64>,
    computed_loss: Option<f64>,
    prediction_receipt_sha256: Option<String>,
    threshold_before: f64,
    threshold_after: f64,
) -> Result<CaoOcrcFeedbackReceiptV1, CaoOcrcErrorV1> {
    let before_sha256 = json_sha256(&threshold_before)?;
    let after_sha256 = json_sha256(&threshold_after)?;
    let mut receipt = CaoOcrcFeedbackReceiptV1 {
        schema: CA_OCRC_FEEDBACK_RECEIPT_SCHEMA_V1.into(),
        delivery_round: feedback.delivery_round,
        originating_round: feedback.originating_round,
        sample_id: feedback.sample_id.clone(),
        label_id: feedback.label_id.clone(),
        label_revision: feedback.label_revision,
        disposition,
        reason,
        actual_delay_rounds: actual_delay,
        proxy_delay_rounds: proxy_delay,
        inverse_proxy_survival_weight: inverse_weight,
        computed_loss,
        prediction_receipt_sha256,
        eligibility_report_sha256: feedback.eligibility_report.report_sha256.clone(),
        label_payload_sha256: feedback.label_payload_sha256.clone(),
        threshold_before_sha256: before_sha256,
        threshold_after_sha256: after_sha256,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = feedback_receipt_digest(&receipt)?;
    Ok(receipt)
}

/// Advance exactly one CA-OCRC round. Feedback events are those that become
/// K1-qualified and available at the end of this round. All state changes are
/// evaluated on a cloned engine and committed only after the round receipt is
/// sealed.
pub fn advance_ca_ocrc_round_v1(
    engine: &mut CaoOcrcEngineV1,
    input: &CaoOcrcRoundInputV1,
    feedback_events: &[CaoOcrcFeedbackInputV1],
) -> Result<CaoOcrcRoundReceiptV1, CaoOcrcErrorV1> {
    validate_engine(engine)?;
    if input.schema != CA_OCRC_ROUND_INPUT_SCHEMA_V1 {
        return Err(CaoOcrcErrorV1("CA-OCRC round input schema is unsupported".into()));
    }
    if input.round_index != engine.next_round_index {
        return Err(CaoOcrcErrorV1(
            "CA-OCRC rounds must be supplied in strictly increasing contiguous order".into(),
        ));
    }
    require_text(&input.sample_id, "input.sample_id")?;
    require_text(&input.measurement_event_id, "input.measurement_event_id")?;
    if input.clock_domain_id != engine.clock_domain_id
        || input.event_time_unix_seconds > input.prediction_at_unix_seconds
        || !input.threshold_used.is_finite()
        || input.threshold_used != engine.threshold
        || !input.score_location.is_finite()
        || !valid_sha256(&input.global_model_artifact_sha256)
        || !valid_sha256(&input.interval_policy_sha256)
        || !valid_sha256(&input.upstream_prediction_receipt_sha256)
    {
        return Err(CaoOcrcErrorV1(
            "CA-OCRC round identity, chronology, clock or digest is invalid".into(),
        ));
    }
    match input.trivial_set {
        Some(CaoOcrcTrivialSetV1::Empty) => {
            if input.threshold_used >= 0.0
                || input.interval_lower.is_some()
                || input.interval_upper.is_some()
                || input.proxy_uniform_draw.is_some()
            {
                return Err(CaoOcrcErrorV1(
                    "empty prediction set requires threshold < 0 and no interval/proxy draw".into(),
                ));
            }
        }
        Some(CaoOcrcTrivialSetV1::Full) => {
            if input.threshold_used < engine.policy.score_upper_bound
                || input.interval_lower.is_some()
                || input.interval_upper.is_some()
                || input.proxy_uniform_draw.is_some()
            {
                return Err(CaoOcrcErrorV1(
                    "full prediction set requires threshold >= score bound and no interval/proxy draw".into(),
                ));
            }
        }
        None => {
            let (Some(lower), Some(upper), Some(draw)) = (
                input.interval_lower,
                input.interval_upper,
                input.proxy_uniform_draw,
            ) else {
                return Err(CaoOcrcErrorV1(
                    "nontrivial prediction set requires finite interval bounds and a proxy draw".into(),
                ));
            };
            if input.threshold_used < 0.0
                || input.threshold_used >= engine.policy.score_upper_bound
                || !lower.is_finite()
                || !upper.is_finite()
                || lower > upper
                || input.score_location < lower
                || input.score_location > upper
                || lower != input.score_location - input.threshold_used
                || upper != input.score_location + input.threshold_used
                || !draw.is_finite()
                || !(0.0..1.0).contains(&draw)
            {
                return Err(CaoOcrcErrorV1(
                    "nontrivial prediction interval, threshold or proxy draw is invalid".into(),
                ));
            }
        }
    }
    if engine.seen_sample_ids.contains(&input.sample_id)
        || engine.seen_measurement_event_ids.contains(&input.measurement_event_id)
    {
        return Err(CaoOcrcErrorV1(
            "duplicate sample or measurement event in CA-OCRC stream".into(),
        ));
    }
    let mut next = engine.clone();
    let tracking_before = engine_tracking_digest(&engine.tracked_rounds, engine.threshold)?;
    let threshold_before = next.threshold;
    let mut guarantee_status = next.guarantee_status;

    // 1. Evict only if the proxy horizon is strictly before the current round.
    let expired: Vec<u64> = next
        .tracked_rounds
        .iter()
        .filter(|(_, tracked)| tracked.proxy_expiry_round < input.round_index)
        .map(|(round, _)| *round)
        .collect();
    for round in &expired {
        next.tracked_rounds.remove(round);
        next.lost_feedback_round_ids.insert(*round);
    }

    // 2. Determine the current round's trivial loss or proxy retention horizon.
    let (trivial_loss, selected_proxy, proxy_draw, admission) = if let Some(trivial) = input.trivial_set {
        (
            Some(trivial.known_loss()),
            -1_i64,
            None,
            CaoOcrcAdmissionV1::NotNeededTrivialSet,
        )
    } else {
        let draw = input.proxy_uniform_draw.expect("validated nontrivial proxy draw");
        let proxy_delay = sample_proxy_delay(&next.policy.proxy_survival, draw)?;
        if proxy_delay < 0 {
            (None, proxy_delay, Some(draw), CaoOcrcAdmissionV1::ProxyNotTracked)
        } else if next.tracked_rounds.len() >= next.policy.tracking_capacity {
            (None, proxy_delay, Some(draw), CaoOcrcAdmissionV1::CapacitySaturated)
        } else {
            let proxy_delay_u64 = proxy_delay as u64;
            let expiry = input
                .round_index
                .checked_add(proxy_delay_u64)
                .ok_or_else(|| CaoOcrcErrorV1("proxy expiry round overflow".into()))?;
            next.tracked_rounds.insert(
                input.round_index,
                CaoOcrcTrackedRoundV1 {
                    round_index: input.round_index,
                    proxy_delay_rounds: proxy_delay_u64,
                    proxy_expiry_round: expiry,
                    proxy_uniform_draw: draw,
                    prediction_receipt_sha256: input.upstream_prediction_receipt_sha256.clone(),
                    round_record_sha256: String::new(),
                },
            );
            (None, proxy_delay, Some(draw), CaoOcrcAdmissionV1::CapacityAdmitted)
        }
    };

    let round_record = CaoOcrcRoundRecordV1 {
        round_index: input.round_index,
        sample_id: input.sample_id.clone(),
        measurement_event_id: input.measurement_event_id.clone(),
        event_time_unix_seconds: input.event_time_unix_seconds,
        prediction_at_unix_seconds: input.prediction_at_unix_seconds,
        clock_domain_id: input.clock_domain_id.clone(),
        global_model_artifact_sha256: input.global_model_artifact_sha256.clone(),
        interval_policy_sha256: input.interval_policy_sha256.clone(),
        upstream_prediction_receipt_sha256: input.upstream_prediction_receipt_sha256.clone(),
        threshold_at_prediction: input.threshold_used,
        score_location: input.score_location,
        trivial_set: input.trivial_set,
        interval_lower: input.interval_lower,
        interval_upper: input.interval_upper,
        prediction_receipt_sha256: input.upstream_prediction_receipt_sha256.clone(),
        round_record_sha256: String::new(),
    };
    let mut round_record = round_record;
    round_record.round_record_sha256 = round_record_digest(&round_record)?;
    if let Some(tracked) = next.tracked_rounds.get_mut(&input.round_index) {
        tracked.round_record_sha256 = round_record.round_record_sha256.clone();
    }
    if next.round_ledger.insert(input.round_index, round_record).is_some() {
        return Err(CaoOcrcErrorV1("duplicate CA-OCRC round ledger entry".into()));
    }
    next.seen_sample_ids.insert(input.sample_id.clone());
    next.seen_measurement_event_ids
        .insert(input.measurement_event_id.clone());
    let tracking_count_after_admission = next.tracked_rounds.len();

    // 3. Receive all now-available feedback; process in a stable order.
    let mut feedback_sorted = feedback_events.to_vec();
    feedback_sorted.sort_by(|a, b| {
        (
            a.originating_round,
            &a.label_id,
            a.label_revision,
            &a.idempotency_key,
            &a.label_payload_sha256,
            &a.measurement_event_id,
        )
            .cmp(&(
                b.originating_round,
                &b.label_id,
                b.label_revision,
                &b.idempotency_key,
                &b.label_payload_sha256,
                &b.measurement_event_id,
            ))
    });
    let mut feedback_receipts = Vec::new();
    let mut feedback_loss_weighted_sum = 0.0_f64;
    let mut feedback_seen_in_round = BTreeSet::new();

    for feedback in &feedback_sorted {
        require_text(&feedback.sample_id, "feedback.sample_id")?;
        require_text(&feedback.label_id, "feedback.label_id")?;
        if feedback.delivery_round != input.round_index {
            return Err(CaoOcrcErrorV1("feedback delivery round differs from current round".into()));
        }
        let idempotency = feedback.idempotency_key.clone();
        if next.seen_feedback_idempotency_keys.contains(&idempotency)
            || !feedback_seen_in_round.insert(idempotency.clone())
        {
            feedback_receipts.push(make_feedback_receipt(
                feedback,
                CaoOcrcFeedbackDispositionV1::Duplicate,
                Some(CaoOcrcFeedbackReasonV1::DuplicateFeedback),
                None,
                None,
                None,
                None,
                None,
                threshold_before,
                threshold_before,
            )?);
            continue;
        }

        let tracked = next.tracked_rounds.get(&feedback.originating_round).cloned();
        let revision_key = format!("{}:{}", feedback.label_id, feedback.label_revision);
        if next.accepted_label_revisions.contains(&revision_key) {
            feedback_receipts.push(make_feedback_receipt(
                feedback,
                CaoOcrcFeedbackDispositionV1::Duplicate,
                Some(CaoOcrcFeedbackReasonV1::DuplicateFeedback),
                None,
                None,
                None,
                None,
                None,
                threshold_before,
                threshold_before,
            )?);
            continue;
        }

        let round = next.round_ledger.get(&feedback.originating_round);
        let Some(round) = round else {
            feedback_receipts.push(make_feedback_receipt(
                feedback,
                CaoOcrcFeedbackDispositionV1::NotTracked,
                Some(CaoOcrcFeedbackReasonV1::PredictionReceiptMismatch),
                None,
                None,
                None,
                None,
                None,
                threshold_before,
                threshold_before,
            )?);
            next.seen_feedback_idempotency_keys.insert(idempotency);
            continue;
        };
        let actual_delay = input.round_index.saturating_sub(feedback.originating_round);
        if let Some(reason) = validate_feedback(feedback, round, &next, actual_delay)? {
            guarantee_status = match reason {
                CaoOcrcFeedbackReasonV1::FeedbackDelayOutOfPolicy => {
                    CaoOcrcGuaranteeStatusV1::InvalidatedByFeedbackDelayBound
                }
                CaoOcrcFeedbackReasonV1::LabelNotCurrentK1
                | CaoOcrcFeedbackReasonV1::LabelNotEligible
                | CaoOcrcFeedbackReasonV1::DownstreamConsumptionNotAllowed => {
                    CaoOcrcGuaranteeStatusV1::InvalidatedByK1QualificationFiltering
                }
                _ => CaoOcrcGuaranteeStatusV1::InvalidatedByMalformedFeedbackEvidence,
            };
            let disposition = match (
                feedback.eligibility_report.eligibility,
                reason,
            ) {
                (
                    LabelConsumptionEligibilityV1::Unknown,
                    CaoOcrcFeedbackReasonV1::LabelNotEligible,
                ) => CaoOcrcFeedbackDispositionV1::HeldUnknown,
                (
                    LabelConsumptionEligibilityV1::Blocked,
                    CaoOcrcFeedbackReasonV1::LabelNotEligible,
                ) => CaoOcrcFeedbackDispositionV1::Blocked,
                _ => CaoOcrcFeedbackDispositionV1::Rejected,
            };
            feedback_receipts.push(make_feedback_receipt(
                feedback,
                disposition,
                Some(reason),
                Some(actual_delay),
                tracked.as_ref().map(|item| item.proxy_delay_rounds),
                None,
                None,
                Some(round.prediction_receipt_sha256.clone()),
                threshold_before,
                threshold_before,
            )?);
            continue;
        }

        let Some(tracked_round) = tracked else {
            let disposition = if next.lost_feedback_round_ids.contains(&feedback.originating_round) {
                CaoOcrcFeedbackDispositionV1::AlreadyExpired
            } else {
                CaoOcrcFeedbackDispositionV1::NotTracked
            };
            feedback_receipts.push(make_feedback_receipt(
                feedback,
                disposition,
                Some(CaoOcrcFeedbackReasonV1::PredictionReceiptMismatch),
                Some(actual_delay),
                None,
                None,
                None,
                Some(round.prediction_receipt_sha256.clone()),
                threshold_before,
                threshold_before,
            )?);
            next.seen_feedback_idempotency_keys.insert(idempotency);
            continue;
        };
        if round.prediction_receipt_sha256 != tracked_round.prediction_receipt_sha256
            || round.round_record_sha256 != tracked_round.round_record_sha256
        {
            feedback_receipts.push(make_feedback_receipt(
                feedback,
                CaoOcrcFeedbackDispositionV1::Rejected,
                Some(CaoOcrcFeedbackReasonV1::PredictionReceiptMismatch),
                Some(actual_delay),
                Some(tracked_round.proxy_delay_rounds),
                None,
                None,
                Some(round.prediction_receipt_sha256.clone()),
                threshold_before,
                threshold_before,
            )?);
            continue;
        }

        let value = feedback.corrected_measurement_value;
        let loss = if value < round.interval_lower.unwrap_or(f64::NEG_INFINITY)
            || value > round.interval_upper.unwrap_or(f64::INFINITY)
        {
            1.0
        } else {
            0.0
        };
        let survival = next.policy.proxy_survival[actual_delay as usize];
        if survival <= 0.0 {
            feedback_receipts.push(make_feedback_receipt(
                feedback,
                CaoOcrcFeedbackDispositionV1::Rejected,
                Some(CaoOcrcFeedbackReasonV1::ProxySurvivalZero),
                Some(actual_delay),
                Some(tracked_round.proxy_delay_rounds),
                None,
                Some(loss),
                Some(round.prediction_receipt_sha256.clone()),
                threshold_before,
                threshold_before,
            )?);
            continue;
        }
        let inverse_weight = 1.0 / survival;
        feedback_loss_weighted_sum += (loss - next.policy.target_risk) * inverse_weight;
        next.tracked_rounds.remove(&feedback.originating_round);
        next.accepted_label_revisions
            .insert(format!("{}:{}", feedback.label_id, feedback.label_revision));
        next.seen_feedback_idempotency_keys.insert(idempotency);
        feedback_receipts.push(make_feedback_receipt(
            feedback,
            CaoOcrcFeedbackDispositionV1::ReceivedAndApplied,
            None,
            Some(actual_delay),
            Some(tracked_round.proxy_delay_rounds),
            Some(inverse_weight),
            Some(loss),
            Some(round.prediction_receipt_sha256.clone()),
            threshold_before,
            threshold_before + next.policy.learning_rate * feedback_loss_weighted_sum,
        )?);
    }

    // 4. Add the trivial-set immediate loss and all received delayed feedback.
    // phi_t is one only for trivial sets; nontrivial current rounds contribute
    // only when their K1-qualified feedback is received later.
    let immediate_term = trivial_loss
        .map(|loss| loss - next.policy.target_risk)
        .unwrap_or(0.0);
    let threshold_after = threshold_before
        + next.policy.learning_rate * (immediate_term + feedback_loss_weighted_sum);
    if !threshold_after.is_finite() {
        return Err(CaoOcrcErrorV1("CA-OCRC threshold update is non-finite".into()));
    }
    next.threshold = threshold_after;
    next.guarantee_status = guarantee_status;
    next.next_round_index = next
        .next_round_index
        .checked_add(1)
        .ok_or_else(|| CaoOcrcErrorV1("CA-OCRC round sequence overflow".into()))?;

    let tracking_after = engine_tracking_digest(&next.tracked_rounds, next.threshold)?;
    let mut receipt = CaoOcrcRoundReceiptV1 {
        schema: CA_OCRC_ROUND_RECEIPT_SCHEMA_V1.into(),
        round_index: input.round_index,
        sample_id: input.sample_id.clone(),
        threshold_before,
        threshold_after,
        trivial_loss,
        proxy_uniform_draw: proxy_draw,
        selected_proxy_delay_rounds: selected_proxy,
        admission,
        tracking_count_after_admission,
        evicted_round_ids: expired.clone(),
        expired_feedback_round_ids: expired,
        received_feedback_receipts: feedback_receipts,
        population_scope: next.policy.population_scope,
        guarantee_status,
        policy_sha256: next.policy_sha256.clone(),
        tracking_state_before_sha256: tracking_before,
        tracking_state_after_sha256: tracking_after,
        round_receipt_sha256: String::new(),
    };
    receipt.round_receipt_sha256 = round_receipt_digest(&receipt)?;
    *engine = next;
    Ok(receipt)
}

fn validate_feedback(
    feedback: &CaoOcrcFeedbackInputV1,
    round: &CaoOcrcRoundRecordV1,
    engine: &CaoOcrcEngineV1,
    actual_delay: u64,
) -> Result<Option<CaoOcrcFeedbackReasonV1>, CaoOcrcErrorV1> {
    let report = &feedback.eligibility_report;
    if feedback.delivery_round < feedback.originating_round
        || actual_delay > engine.policy.maximum_feedback_delay_rounds
    {
        return Ok(Some(CaoOcrcFeedbackReasonV1::FeedbackDelayOutOfPolicy));
    }
    if report.schema != LABEL_CONSUMPTION_REPORT_SCHEMA_V1
        || eligibility_report_digest(report)? != report.report_sha256
    {
        return Ok(Some(CaoOcrcFeedbackReasonV1::EligibilityReportDigestMismatch));
    }
    if round_record_digest(round)? != round.round_record_sha256 {
        return Ok(Some(CaoOcrcFeedbackReasonV1::PredictionIntervalInvalid));
    }
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        return Ok(Some(CaoOcrcFeedbackReasonV1::LabelNotCurrentK1));
    }
    match report.eligibility {
        LabelConsumptionEligibilityV1::Unknown => {
            return Ok(Some(CaoOcrcFeedbackReasonV1::LabelNotEligible));
        }
        LabelConsumptionEligibilityV1::Blocked => {
            return Ok(Some(CaoOcrcFeedbackReasonV1::LabelNotEligible));
        }
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => {}
    }
    if !report.downstream_consumption_allowed {
        return Ok(Some(CaoOcrcFeedbackReasonV1::DownstreamConsumptionNotAllowed));
    }
    if report.label_id != feedback.label_id {
        return Ok(Some(CaoOcrcFeedbackReasonV1::LabelIdentityMismatch));
    }
    if report.label_revision != feedback.label_revision
        || feedback.label_revision == 0
        || feedback.idempotency_key != format!("{}:{}", feedback.label_id, feedback.label_revision)
    {
        return Ok(Some(CaoOcrcFeedbackReasonV1::LabelRevisionMismatch));
    }
    if report.measurement_event_id != feedback.measurement_event_id
        || feedback.measurement_event_id != round.measurement_event_id
        || feedback.sample_id != round.sample_id
    {
        return Ok(Some(CaoOcrcFeedbackReasonV1::MeasurementEventMismatch));
    }
    if !valid_sha256(&feedback.label_payload_sha256)
        || feedback.label_payload_sha256 != report.label_sha256
    {
        return Ok(Some(CaoOcrcFeedbackReasonV1::LabelPayloadDigestMismatch));
    }
    if !feedback.corrected_measurement_value.is_finite() {
        return Ok(Some(CaoOcrcFeedbackReasonV1::DeliveredValueNonFinite));
    }
    if (feedback.corrected_measurement_value - round.score_location).abs()
        > engine.policy.score_upper_bound
    {
        return Ok(Some(CaoOcrcFeedbackReasonV1::ScoreRangeViolation));
    }
    if feedback.corrected_measurement_value != report.qualified_corrected_value {
        return Ok(Some(CaoOcrcFeedbackReasonV1::DeliveredValueMismatch));
    }
    if feedback.clock_domain_id != engine.clock_domain_id
        || feedback.delivered_at_unix_seconds < round.prediction_at_unix_seconds
    {
        return Ok(Some(CaoOcrcFeedbackReasonV1::ClockDomainMismatch));
    }
    if feedback.label_available_at_unix_seconds < round.event_time_unix_seconds {
        return Ok(Some(CaoOcrcFeedbackReasonV1::LabelAvailableBeforeMeasurement));
    }
    if feedback.eligibility_report_available_at_unix_seconds
        < feedback.label_available_at_unix_seconds
    {
        return Ok(Some(CaoOcrcFeedbackReasonV1::ReportBeforeLabelAvailability));
    }
    if feedback.delivered_at_unix_seconds < feedback.eligibility_report_available_at_unix_seconds
    {
        return Ok(Some(CaoOcrcFeedbackReasonV1::ReportUnavailableAtDelivery));
    }
    Ok(None)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoOcrcBernoulliOptimizerResultV1 {
    pub capacity: usize,
    pub maximum_delay_rounds: u64,
    pub horizon_rounds: u64,
    pub optimal_track_probability: f64,
    pub expected_tracking_load: f64,
    pub risk_bound: f64,
    pub optimizer_iterations: u32,
    pub corollary_one_regime_applicable: bool,
}

pub fn full_tracking_proxy_survival_v1(
    maximum_delay_rounds: u64,
) -> Vec<f64> {
    vec![1.0; maximum_delay_rounds as usize + 1]
}

pub fn optimized_bernoulli_proxy_survival_v1(
    result: &CaoOcrcBernoulliOptimizerResultV1,
) -> Vec<f64> {
    vec![result.optimal_track_probability; result.maximum_delay_rounds as usize + 1]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaoOcrcBoundRegimeV1 {
    CapacityConstrainedTheoremOne,
    UnlimitedTrackingOcrcLemmaTwo,
    NoFeedbackCapacityZero,
    ImmediateFeedbackSpecialCase,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoOcrcRiskBoundV1 {
    pub regime: CaoOcrcBoundRegimeV1,
    pub horizon_rounds: u64,
    pub score_upper_bound: f64,
    pub learning_rate: f64,
    pub tracking_capacity: usize,
    pub maximum_delay_rounds: u64,
    pub population_scope: CaoOcrcPopulationScopeV1,
    pub tracking_load: Option<f64>,
    pub minimum_proxy_survival: Option<f64>,
    pub bound: Option<f64>,
    pub corollary_one_regime_applicable: bool,
}

/// Computes the exact displayed risk bound for the regime covered by
/// Theorem 1 (capacity-constrained) or Lemma 2 (unlimited tracking).
/// It refuses to silently apply either expression outside its assumptions.
pub fn compute_ca_ocrc_risk_bound_v1(
    policy: &CaoOcrcPolicyV1,
    horizon_rounds: u64,
) -> Result<CaoOcrcRiskBoundV1, CaoOcrcErrorV1> {
    validate_policy(policy)?;
    if horizon_rounds == 0 {
        return Err(CaoOcrcErrorV1("risk-bound horizon must be positive".into()));
    }
    let d_max = policy.maximum_feedback_delay_rounds;
    let k = policy.tracking_capacity;
    let common = policy.score_upper_bound / (policy.learning_rate * horizon_rounds as f64);

    if k == 0 {
        return Ok(CaoOcrcRiskBoundV1 {
            regime: CaoOcrcBoundRegimeV1::NoFeedbackCapacityZero,
            horizon_rounds,
            score_upper_bound: policy.score_upper_bound,
            learning_rate: policy.learning_rate,
            tracking_capacity: k,
            maximum_delay_rounds: d_max,
            tracking_load: None,
            minimum_proxy_survival: None,
            bound: None,
            corollary_one_regime_applicable: false,
        });
    }
    if d_max == 0 {
        return Ok(CaoOcrcRiskBoundV1 {
            regime: CaoOcrcBoundRegimeV1::ImmediateFeedbackSpecialCase,
            horizon_rounds,
            score_upper_bound: policy.score_upper_bound,
            learning_rate: policy.learning_rate,
            tracking_capacity: k,
            maximum_delay_rounds: d_max,
            tracking_load: Some(0.0),
            minimum_proxy_survival: Some(policy.proxy_survival[0]),
            bound: if policy.proxy_survival[0] == 1.0 { Some(common) } else { None },
            corollary_one_regime_applicable: false,
        });
    }
    let tracking_load: f64 = policy.proxy_survival.iter().skip(1).sum();
    if tracking_load > k as f64 + 1.0e-12 {
        return Err(CaoOcrcErrorV1(
            "CA-OCRC Theorem 1 requires proxy tracking load <= capacity".into(),
        ));
    }
    let minimum_survival = *policy.proxy_survival.last().unwrap();
    if k as u64 >= d_max + 1 {
        let bound = common + (d_max + 1) as f64 / horizon_rounds as f64;
        return Ok(CaoOcrcRiskBoundV1 {
            regime: CaoOcrcBoundRegimeV1::UnlimitedTrackingOcrcLemmaTwo,
            horizon_rounds,
            score_upper_bound: policy.score_upper_bound,
            learning_rate: policy.learning_rate,
            tracking_capacity: k,
            maximum_delay_rounds: d_max,
            tracking_load: Some(tracking_load),
            minimum_proxy_survival: Some(minimum_survival),
            bound: Some(bound),
            corollary_one_regime_applicable: false,
        });
    }
    if k as u64 > d_max {
        return Err(CaoOcrcErrorV1(
            "capacity regime is inconsistent with maximum-delay boundary".into(),
        ));
    }
    if minimum_survival <= 0.0 || tracking_load <= 0.0 {
        return Err(CaoOcrcErrorV1(
            "Theorem 1 needs positive F(d_max) and positive tracking load".into(),
        ));
    }
    let x = tracking_load / k as f64;
    let h = x - 1.0 - x.ln();
    let bound = common
        + (k as f64 + 1.0) / (minimum_survival * horizon_rounds as f64)
        + (-k as f64 * h).exp();
    Ok(CaoOcrcRiskBoundV1 {
        regime: CaoOcrcBoundRegimeV1::CapacityConstrainedTheoremOne,
        horizon_rounds,
        score_upper_bound: policy.score_upper_bound,
        learning_rate: policy.learning_rate,
        tracking_capacity: k,
        maximum_delay_rounds: d_max,
        tracking_load: Some(tracking_load),
        minimum_proxy_survival: Some(minimum_survival),
        bound: Some(bound),
        corollary_one_regime_applicable: (1..=d_max).contains(&(k as u64))
            && d_max <= horizon_rounds,
    })
}

pub fn optimize_ca_ocrc_bernoulli_proxy_v1(
    capacity: usize,
    maximum_delay_rounds: u64,
    horizon_rounds: u64,
    score_upper_bound: f64,
    learning_rate: f64,
) -> Result<CaoOcrcBernoulliOptimizerResultV1, CaoOcrcErrorV1> {
    if capacity == 0
        || maximum_delay_rounds == 0
        || horizon_rounds == 0
        || !score_upper_bound.is_finite()
        || score_upper_bound <= 0.0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
    {
        return Err(CaoOcrcErrorV1("Bernoulli proxy optimizer inputs are invalid".into()));
    }
    if capacity as u64 > maximum_delay_rounds {
        return Err(CaoOcrcErrorV1(
            "Bernoulli capacity optimizer requires 1 <= K <= d_max; use full tracking otherwise".into(),
        ));
    }
    let upper = (capacity as f64 / maximum_delay_rounds as f64).min(1.0);
    let objective = |p: f64| {
        let x = maximum_delay_rounds as f64 * p / capacity as f64;
        let h = x - 1.0 - x.ln();
        (capacity as f64 + 1.0) / (p * horizon_rounds as f64)
            + (-capacity as f64 * h).exp()
    };
    let mut low = 1.0e-12_f64.min(upper / 1000.0);
    let mut high = upper;
    let golden = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut x1 = high - golden * (high - low);
    let mut x2 = low + golden * (high - low);
    let mut f1 = objective(x1);
    let mut f2 = objective(x2);
    let mut iterations = 0;
    while iterations < 160 && (high - low) > 1.0e-12 * upper.max(1.0) {
        if f1 > f2 {
            low = x1;
            x1 = x2;
            f1 = f2;
            x2 = low + golden * (high - low);
            f2 = objective(x2);
        } else {
            high = x2;
            x2 = x1;
            f2 = f1;
            x1 = high - golden * (high - low);
            f1 = objective(x1);
        }
        iterations += 1;
    }
    let p = 0.5 * (low + high);
    let x = maximum_delay_rounds as f64 * p / capacity as f64;
    let h = x - 1.0 - x.ln();
    let risk_bound = score_upper_bound / (learning_rate * horizon_rounds as f64)
        + (capacity as f64 + 1.0) / (p * horizon_rounds as f64)
        + (-capacity as f64 * h).exp();
    Ok(CaoOcrcBernoulliOptimizerResultV1 {
        capacity,
        maximum_delay_rounds,
        horizon_rounds,
        optimal_track_probability: p,
        expected_tracking_load: maximum_delay_rounds as f64 * p,
        risk_bound,
        optimizer_iterations: iterations,
        corollary_one_regime_applicable: maximum_delay_rounds <= horizon_rounds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy(capacity: usize, d_max: u64, survival: Vec<f64>) -> CaoOcrcPolicyV1 {
        CaoOcrcPolicyV1 {
            schema: CA_OCRC_POLICY_SCHEMA_V1.into(),
            policy_id: "ca-ocrc-deterministic-v1".into(),
            revision: 1,
            target_risk: 0.10,
            learning_rate: 0.05,
            score_upper_bound: 100.0,
            initial_threshold: 0.5,
            tracking_capacity: capacity,
            maximum_feedback_delay_rounds: d_max,
            proxy_survival: survival,
        }
    }

    fn engine(capacity: usize, d_max: u64, survival: Vec<f64>) -> CaoOcrcEngineV1 {
        create_ca_ocrc_engine_v1("ca-ocrc-001", "fab-clock", policy(capacity, d_max, survival))
            .unwrap()
    }

    fn round(
        index: u64,
        trivial: Option<CaoOcrcTrivialSetV1>,
        interval: Option<(f64, f64)>,
        draw: Option<f64>,
    ) -> CaoOcrcRoundInputV1 {
        CaoOcrcRoundInputV1 {
            schema: CA_OCRC_ROUND_INPUT_SCHEMA_V1.into(),
            round_index: index,
            sample_id: format!("sample-{index}"),
            measurement_event_id: format!("event-{index}"),
            event_time_unix_seconds: index as i64 * 10,
            prediction_at_unix_seconds: index as i64 * 10 + 1,
            clock_domain_id: "fab-clock".into(),
            global_model_artifact_sha256: digest('a'),
            interval_policy_sha256: digest('b'),
            upstream_prediction_receipt_sha256: digest('c'),
            trivial_set: trivial,
            threshold_used: 0.5,
            score_location: interval
                .map(|bounds| (bounds.0 + bounds.1) / 2.0)
                .unwrap_or(0.0),
            interval_lower: interval.map(|bounds| bounds.0),
            interval_upper: interval.map(|bounds| bounds.1),
            proxy_uniform_draw: draw,
        }
    }

    fn advance(
        engine: &mut CaoOcrcEngineV1,
        mut input: CaoOcrcRoundInputV1,
        feedback_events: &[CaoOcrcFeedbackInputV1],
    ) -> Result<CaoOcrcRoundReceiptV1, CaoOcrcErrorV1> {
        input.threshold_used = engine.threshold;
        if input.trivial_set.is_none() {
            input.interval_lower = Some(input.score_location - engine.threshold);
            input.interval_upper = Some(input.score_location + engine.threshold);
        }
        advance_ca_ocrc_round_v1(engine, &input, feedback_events)
    }

    fn feedback(
        origin: u64,
        delivery_round: u64,
        value: f64,
        label: &str,
        revision: u64,
        status: LabelConsumptionEligibilityV1,
    ) -> CaoOcrcFeedbackInputV1 {
        let payload = digest('d');
        let event_id = format!("event-{origin}");
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: LABEL_CONSUMPTION_REPORT_SCHEMA_V1.into(),
            label_id: label.into(),
            label_revision: revision,
            measurement_event_id: event_id.clone(),
            calibration_certificate_id: "cal-cert-001".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: status,
            downstream_consumption_allowed: status
                == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: payload.clone(),
            measurement_sha256: digest('e'),
            qualified_corrected_value: value,
            qualified_standard_uncertainty_upper_bound: Some(0.10),
            qualification_report_sha256: digest('f'),
            requalification_report_sha256: digest('1'),
            requalification_frontier_sha256: digest('2'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        let mut canonical = report.clone();
        canonical.report_sha256.clear();
        report.report_sha256 = json_sha256(&canonical).unwrap();
        CaoOcrcFeedbackInputV1 {
            schema: CA_OCRC_FEEDBACK_SCHEMA_V1.into(),
            delivery_round,
            originating_round: origin,
            sample_id: format!("sample-{origin}"),
            label_id: label.into(),
            label_revision: revision,
            measurement_event_id: event_id,
            label_payload_sha256: payload,
            corrected_measurement_value: value,
            label_available_at_unix_seconds: origin as i64 * 10 + 2,
            eligibility_report_available_at_unix_seconds: delivery_round as i64 * 10 + 2,
            delivered_at_unix_seconds: delivery_round as i64 * 10 + 3,
            clock_domain_id: "fab-clock".into(),
            idempotency_key: format!("{label}:{revision}"),
            eligibility_report: report,
        }
    }

    #[test]
    fn k_one_saturates_when_a_long_proxy_round_is_still_tracked() {
        let mut engine = engine(1, 2, vec![1.0, 1.0, 1.0]);
        let first = advance(
            &mut engine,
            &round(1, None, Some((0.0, 1.0)), Some(0.5)),
            &[],
        )
        .unwrap();
        assert_eq!(first.admission, CaoOcrcAdmissionV1::CapacityAdmitted);
        let second = advance(
            &mut engine,
            &round(2, None, Some((0.0, 1.0)), Some(0.5)),
            &[],
        )
        .unwrap();
        assert_eq!(second.admission, CaoOcrcAdmissionV1::CapacitySaturated);
        assert_eq!(engine.tracked_rounds.len(), 1);
    }

    #[test]
    fn feedback_on_proxy_expiry_round_is_received_before_eviction() {
        let mut engine = engine(1, 2, vec![1.0, 1.0, 1.0]);
        advance(
            &mut engine,
            &round(1, None, Some((0.0, 1.0)), Some(0.5)),
            &[],
        )
        .unwrap();
        advance(
            &mut engine,
            &round(2, None, Some((0.0, 1.0)), Some(0.5)),
            &[],
        )
        .unwrap();
        let exact = advance(
            &mut engine,
            &round(3, None, Some((0.0, 1.0)), Some(0.5)),
            &[feedback(
                1,
                3,
                2.0,
                "label-1",
                1,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            )],
        )
        .unwrap();
        assert!(exact.evicted_round_ids.is_empty());
        assert_eq!(
            exact.received_feedback_receipts[0].disposition,
            CaoOcrcFeedbackDispositionV1::ReceivedAndApplied
        );
        assert!(!engine.tracked_rounds.contains_key(&1));
    }

    #[test]
    fn feedback_after_proxy_expiry_is_permanently_lost_to_online_update() {
        let mut engine = engine(1, 2, vec![1.0, 1.0, 1.0]);
        advance(
            &mut engine,
            &round(1, None, Some((0.0, 1.0)), Some(0.5)),
            &[],
        )
        .unwrap();
        for round_index in 2..=3 {
            advance(
                &mut engine,
                &round(round_index, None, Some((0.0, 1.0)), Some(0.5)),
                &[],
            )
            .unwrap();
        }
        let late = advance(
            &mut engine,
            &round(4, None, Some((0.0, 1.0)), Some(0.5)),
            &[feedback(
                1,
                4,
                2.0,
                "label-late",
                1,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            )],
        )
        .unwrap();
        assert_eq!(late.evicted_round_ids, vec![1]);
        assert_eq!(
            late.received_feedback_receipts[0].disposition,
            CaoOcrcFeedbackDispositionV1::AlreadyExpired
        );
    }

    #[test]
    fn immediate_feedback_is_received_only_when_current_round_was_admitted() {
        let mut engine = engine(1, 0, vec![1.0]);
        let receipt = advance(
            &mut engine,
            &round(1, None, Some((0.0, 1.0)), Some(0.5)),
            &[feedback(
                1,
                1,
                2.0,
                "label-1",
                1,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            )],
        )
        .unwrap();
        assert_eq!(receipt.admission, CaoOcrcAdmissionV1::CapacityAdmitted);
        assert_eq!(
            receipt.received_feedback_receipts[0].disposition,
            CaoOcrcFeedbackDispositionV1::ReceivedAndApplied
        );
        assert_eq!(
            receipt.received_feedback_receipts[0].actual_delay_rounds,
            Some(0)
        );
        assert!(receipt.threshold_after > receipt.threshold_before);
        assert!(engine.tracked_rounds.is_empty());
    }

    #[test]
    fn proxy_minus_one_is_not_admitted_and_feedback_is_reported_lost() {
        let mut engine = engine(1, 2, vec![0.5, 0.5, 0.5]);
        let receipt = advance(
            &mut engine,
            &round(1, None, Some((0.0, 1.0)), Some(0.1)),
            &[],
        )
        .unwrap();
        assert_eq!(receipt.selected_proxy_delay_rounds, -1);
        assert_eq!(receipt.admission, CaoOcrcAdmissionV1::ProxyNotTracked);
        let later = advance(
            &mut engine,
            &round(2, None, Some((0.0, 1.0)), Some(0.8)),
            &[feedback(
                1,
                2,
                2.0,
                "label-1",
                1,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            )],
        )
        .unwrap();
        assert_eq!(
            later.received_feedback_receipts[0].disposition,
            CaoOcrcFeedbackDispositionV1::NotTracked
        );
        assert!(!engine.tracked_rounds.contains_key(&1));
        assert!((engine.threshold - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn trivial_empty_and_full_sets_update_immediately_without_tracking() {
        let mut empty_engine = engine(1, 2, vec![1.0, 1.0, 1.0]);
        empty_engine.threshold = -0.5;
        let mut empty_round = round(1, Some(CaoOcrcTrivialSetV1::Empty), None, None);
        empty_round.threshold_used = -0.5;
        let empty = advance(
            &mut empty_engine,
            empty_round,
            &[],
        )
        .unwrap();
        assert_eq!(empty.trivial_loss, Some(1.0));
        assert_eq!(empty_engine.tracked_rounds.len(), 0);
        assert!(empty.threshold_after > empty.threshold_before);

        let mut full_engine = engine(1, 2, vec![1.0, 1.0, 1.0]);
        full_engine.threshold = 100.0;
        let mut full_round = round(1, Some(CaoOcrcTrivialSetV1::Full), None, None);
        full_round.threshold_used = 100.0;
        let full = advance(
            &mut full_engine,
            full_round,
            &[],
        )
        .unwrap();
        assert_eq!(full.trivial_loss, Some(0.0));
        assert!(full.threshold_after < full.threshold_before);
    }

    #[test]
    fn k1_unknown_feedback_does_not_update_threshold_or_consume_tracking() {
        let mut engine = engine(1, 2, vec![1.0, 1.0, 1.0]);
        advance(
            &mut engine,
            &round(1, None, Some((0.0, 1.0)), Some(0.5)),
            &[],
        )
        .unwrap();
        let before = engine.threshold;
        let unknown = feedback(1, 2, 2.0, "label-1", 1, LabelConsumptionEligibilityV1::Unknown);
        let receipt = advance(
            &mut engine,
            &round(2, None, Some((0.0, 1.0)), Some(0.5)),
            &[unknown],
        )
        .unwrap();
        assert_eq!(receipt.received_feedback_receipts[0].disposition, CaoOcrcFeedbackDispositionV1::HeldUnknown);
        assert_eq!(engine.threshold, before);
        assert!(engine.tracked_rounds.contains_key(&1));
    }

    #[test]
    fn qualified_feedback_is_weighted_by_inverse_proxy_survival() {
        let mut engine = engine(1, 2, vec![1.0, 0.5, 0.5]);
        advance(
            &mut engine,
            &round(1, None, Some((0.0, 1.0)), Some(0.5)),
            &[],
        )
        .unwrap();
        // A delay-one feedback is observable only if the proxy delay is at least one.
        let result = advance(
            &mut engine,
            &round(2, None, Some((0.0, 1.0)), Some(0.5)),
            &[feedback(1, 2, 2.0, "label-1", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy)],
        )
        .unwrap();
        let event = &result.received_feedback_receipts[0];
        assert_eq!(event.disposition, CaoOcrcFeedbackDispositionV1::ReceivedAndApplied);
        assert_eq!(event.actual_delay_rounds, Some(1));
        assert_eq!(event.inverse_proxy_survival_weight, Some(2.0));
    }

    #[test]
    fn simultaneous_feedback_is_sorted_and_deterministic() {
        fn run(reverse: bool) -> (f64, String) {
            let mut engine = engine(2, 2, vec![1.0, 1.0, 1.0]);
            for index in 1..=2 {
                advance(
                    &mut engine,
                    &round(index, None, Some((0.0, 1.0)), Some(0.5)),
                    &[],
                )
                .unwrap();
            }
            let mut feedbacks = vec![
                feedback(1, 3, 2.0, "label-a", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy),
                feedback(2, 3, 0.0, "label-b", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy),
            ];
            if reverse {
                feedbacks.reverse();
            }
            let _ = advance(
                &mut engine,
                &round(3, None, Some((0.0, 1.0)), Some(0.5)),
                &feedbacks,
            )
            .unwrap();
            let mut canonical = engine.clone();
            canonical.round_ledger.clear();
            canonical.accepted_label_revisions.clear();
            canonical.seen_feedback_idempotency_keys.clear();
            (engine.threshold, json_sha256(&canonical).unwrap())
        }
        assert_eq!(run(false), run(true));
    }

    #[test]
    fn survival_vector_must_be_monotonic_and_positive_at_maximum_delay() {
        assert!(create_ca_ocrc_engine_v1("bad", "clock", policy(1, 2, vec![0.5, 0.8, 0.2])).is_err());
        assert!(create_ca_ocrc_engine_v1("bad", "clock", policy(1, 2, vec![1.0, 0.5, 0.0])).is_err());
    }

    #[test]
    fn zero_capacity_is_an_explicit_no_feedback_baseline() {
        let mut engine = engine(0, 2, vec![1.0, 1.0, 1.0]);
        let receipt = advance(
            &mut engine,
            &round(1, None, Some((0.0, 1.0)), Some(0.5)),
            &[feedback(
                1,
                1,
                2.0,
                "label-zero-capacity",
                1,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            )],
        )
        .unwrap();
        assert_eq!(receipt.admission, CaoOcrcAdmissionV1::CapacitySaturated);
        assert_eq!(
            receipt.received_feedback_receipts[0].disposition,
            CaoOcrcFeedbackDispositionV1::NotTracked
        );
        assert_eq!(engine.threshold, 0.5);
        assert!(engine.tracked_rounds.is_empty());
    }

    #[test]
    fn capacity_at_least_dmax_plus_one_can_track_all_pending_rounds() {
        let mut engine = engine(3, 2, vec![1.0, 1.0, 1.0]);
        for index in 1..=3 {
            let receipt = advance(
                &mut engine,
                &round(index, None, Some((0.0, 1.0)), Some(0.5)),
                &[],
            )
            .unwrap();
            assert_eq!(receipt.admission, CaoOcrcAdmissionV1::CapacityAdmitted);
            assert!(engine.tracked_rounds.len() <= 3);
        }
        assert_eq!(engine.tracked_rounds.len(), 3);
        assert!(full_tracking_proxy_survival_v1(2).iter().all(|value| *value == 1.0));
    }

    #[test]
    fn tampering_with_prediction_interval_invalidates_delayed_feedback() {
        let mut engine = engine(1, 2, vec![1.0, 1.0, 1.0]);
        advance(
            &mut engine,
            &round(1, None, Some((0.0, 1.0)), Some(0.5)),
            &[],
        )
        .unwrap();
        engine.round_ledger.get_mut(&1).unwrap().interval_upper = Some(2.0);
        let receipt = advance(
            &mut engine,
            &round(2, None, Some((0.0, 1.0)), Some(0.5)),
            &[feedback(
                1,
                2,
                2.0,
                "label-tamper",
                1,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            )],
        )
        .unwrap();
        assert_eq!(
            receipt.received_feedback_receipts[0].disposition,
            CaoOcrcFeedbackDispositionV1::Rejected
        );
        assert_eq!(
            receipt.received_feedback_receipts[0].reason,
            Some(CaoOcrcFeedbackReasonV1::PredictionIntervalInvalid)
        );
        assert_eq!(engine.threshold, 0.5);
        assert!(engine.tracked_rounds.contains_key(&1));
    }

    #[test]
    fn mismatched_label_revision_and_payload_do_not_update_threshold() {
        let mut engine = engine(1, 2, vec![1.0, 1.0, 1.0]);
        advance(
            &mut engine,
            &round(1, None, Some((0.0, 1.0)), Some(0.5)),
            &[],
        )
        .unwrap();
        let mut wrong = feedback(
            1,
            2,
            2.0,
            "label-wrong",
            2,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        );
        wrong.eligibility_report.label_revision = 1;
        let mut canonical = wrong.eligibility_report.clone();
        canonical.report_sha256.clear();
        wrong.eligibility_report.report_sha256 = json_sha256(&canonical).unwrap();
        wrong.label_payload_sha256 = digest('z');
        let receipt = advance(
            &mut engine,
            &round(2, None, Some((0.0, 1.0)), Some(0.5)),
            &[wrong],
        )
        .unwrap();
        assert_eq!(
            receipt.received_feedback_receipts[0].disposition,
            CaoOcrcFeedbackDispositionV1::Rejected
        );
        assert_eq!(engine.threshold, 0.5);
    }

    #[test]
    fn malformed_non_monotone_or_zero_maximum_survival_is_rejected() {
        assert!(engine(1, 2, vec![0.5, 0.8, 0.2]).tracked_rounds.is_empty());
        assert!(create_ca_ocrc_engine_v1("bad", "clock", policy(1, 2, vec![1.0, 0.5, 0.0])).is_err());
    }

    #[test]
    fn optimizer_survival_vector_tracks_maximum_delay_with_probability_p() {
        let result = optimize_ca_ocrc_bernoulli_proxy_v1(2, 5, 1000, 1.0, 0.1).unwrap();
        let survival = optimized_bernoulli_proxy_survival_v1(&result);
        assert_eq!(survival.len(), 6);
        assert!(survival.iter().all(|value| (*value - result.optimal_track_probability).abs() < f64::EPSILON));
        assert!(result.expected_tracking_load <= 2.0);
    }

    #[test]
    fn general_arbitrary_delay_bound_matches_theorem_one_formula() {
        let policy = policy(2, 3, vec![1.0, 0.8, 0.4, 0.2]);
        let result = compute_ca_ocrc_risk_bound_v1(&policy, 100).unwrap();
        let load = 0.8 + 0.4 + 0.2;
        let x = load / 2.0;
        let h = x - 1.0 - x.ln();
        let expected = policy.score_upper_bound / (policy.learning_rate * 100.0)
            + 3.0 / (0.2 * 100.0)
            + (-2.0 * h).exp();
        assert_eq!(result.regime, CaoOcrcBoundRegimeV1::CapacityConstrainedTheoremOne);
        assert!((result.tracking_load.unwrap() - load).abs() < 1.0e-12);
        assert!((result.bound.unwrap() - expected).abs() < 1.0e-12);
        assert!(result.corollary_one_regime_applicable);
    }

    #[test]
    fn dmax_exceeding_horizon_keeps_theorem_one_but_not_corollary_one_claim() {
        let policy = policy(2, 5, vec![1.0, 0.5, 0.4, 0.3, 0.2, 0.1]);
        let result = compute_ca_ocrc_risk_bound_v1(&policy, 4).unwrap();
        assert_eq!(result.regime, CaoOcrcBoundRegimeV1::CapacityConstrainedTheoremOne);
        assert!(result.bound.is_some());
        assert!(!result.corollary_one_regime_applicable);
    }

    #[test]
    fn full_tracking_uses_the_unlimited_ocrc_lemma_two_bound() {
        let policy = policy(4, 3, full_tracking_proxy_survival_v1(3));
        let result = compute_ca_ocrc_risk_bound_v1(&policy, 100).unwrap();
        let expected = policy.score_upper_bound / (policy.learning_rate * 100.0)
            + 4.0 / 100.0;
        assert_eq!(result.regime, CaoOcrcBoundRegimeV1::UnlimitedTrackingOcrcLemmaTwo);
        assert!((result.bound.unwrap() - expected).abs() < 1.0e-12);
    }

    #[test]
    fn zero_capacity_and_dmax_zero_are_reported_as_special_regimes() {
        let no_capacity = compute_ca_ocrc_risk_bound_v1(
            &policy(0, 2, vec![1.0, 1.0, 1.0]),
            100,
        )
        .unwrap();
        assert_eq!(no_capacity.regime, CaoOcrcBoundRegimeV1::NoFeedbackCapacityZero);
        assert!(no_capacity.bound.is_none());

        let immediate = compute_ca_ocrc_risk_bound_v1(
            &policy(1, 0, vec![1.0]),
            100,
        )
        .unwrap();
        assert_eq!(immediate.regime, CaoOcrcBoundRegimeV1::ImmediateFeedbackSpecialCase);
        assert!(immediate.bound.is_some());
    }

    #[test]
    fn bernoulli_optimizer_allows_dmax_over_horizon_but_disclaims_corollary() {
        let result = optimize_ca_ocrc_bernoulli_proxy_v1(2, 6, 4, 100.0, 0.05).unwrap();
        assert!(result.optimal_track_probability > 0.0);
        assert!(!result.corollary_one_regime_applicable);
    }

    #[test]
    fn bernoulli_proxy_optimizer_returns_valid_probability_and_bound() {
        let result = optimize_ca_ocrc_bernoulli_proxy_v1(2, 5, 1000, 1.0, 0.1).unwrap();
        assert!(result.optimal_track_probability > 0.0 && result.optimal_track_probability <= 0.4);
        assert!(result.expected_tracking_load <= 2.0);
        assert!(result.risk_bound.is_finite());
    }
}
