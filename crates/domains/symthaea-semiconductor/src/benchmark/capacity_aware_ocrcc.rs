// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capacity-aware online conformal risk control (CA-OCRC), arbitrary bounded
//! delay path, following Wang & Simeone (arXiv:2611.17059).
//!
//! This v1 implementation supports the paper's arbitrary bounded-delay model
//! only. It implements proxy-delay admission/eviction, a finite tracking set,
//! the exact F(d)^-1 inverse-probability update, and K1-gated physical labels.
//! The stochastic-delay q_G correction and optimized stochastic proxy design
//! are intentionally not implemented here.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
    LabelEvidenceFrontierV1,
};

pub const CA_OCRC_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc.v1";
pub const CA_OCRC_POLICY_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-policy.v1";
pub const CA_OCRC_ROUND_INPUT_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-round-input.v1";
pub const CA_OCRC_DELIVERY_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-label-delivery.v1";
pub const CA_OCRC_ROUND_RECEIPT_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-round-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcSetKindV1 {
    FullRealLine,
    Empty,
    BoundedInterval,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcGuaranteeStateV1 {
    UnderDeclaredArbitraryBoundedDelayAssumptions,
    UncertifiedInvalidOrOutOfModelFeedback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcAdmissionV1 {
    TrivialSetNoTracking,
    ProxyRejected,
    Admitted,
    CapacitySaturated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcFeedbackDispositionV1 {
    Updated,
    HeldUnknown,
    Blocked,
    Rejected,
    Duplicate,
    Untracked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcFeedbackReasonV1 {
    LabelNotCurrentK1,
    LabelNotEligible,
    DownstreamConsumptionNotAllowed,
    LabelIdentityMismatch,
    LabelRevisionMismatch,
    MeasurementEventMismatch,
    LabelPayloadDigestMismatch,
    EligibilityReportDigestMismatch,
    DeliveredValueMismatch,
    InvalidLabelRevision,
    InvalidIdempotencyKey,
    InvalidNumericReport,
    ClockDomainMismatch,
    LabelNotAvailableByDelivery,
    DeliveryBeforePredictionTime,
    LabelAvailableBeforeMeasurement,
    ReportBeforeLabelAvailability,
    ReportNotAvailableByDelivery,
    WrongDeliveryRound,
    FeedbackDelayOutOfDeclaredBound,
    FeedbackNotCoveredByProxyDelay,
    DuplicateAcceptedRevision,
    NoTrackedPrediction,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcProxyDelayDistributionV1 {
    pub schema: String,
    pub distribution_id: String,
    pub maximum_proxy_delay: u64,
    /// PMF indexed by delay + 1: index 0 is delay -1; index 1 is delay 0.
    /// The support is exactly {-1, 0, ..., maximum_proxy_delay}.
    pub probability_mass: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub target_risk: f64,
    pub learning_rate: f64,
    pub score_bound: f64,
    pub tracking_capacity: usize,
    pub maximum_real_delay: u64,
    pub proxy_distribution: CaOcrcProxyDelayDistributionV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcRoundInputV1 {
    pub schema: String,
    pub round: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub clock_domain_id: String,
    pub model_artifact_sha256: String,
    pub event_time_unix_seconds: i64,
    pub prediction_time_unix_seconds: i64,
    pub point_prediction: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcPredictionReceiptV1 {
    pub schema: String,
    pub round: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub clock_domain_id: String,
    pub model_artifact_sha256: String,
    pub event_time_unix_seconds: i64,
    pub prediction_time_unix_seconds: i64,
    pub point_prediction: f64,
    pub set_kind: CaOcrcSetKindV1,
    pub interval_lower: Option<f64>,
    pub interval_upper: Option<f64>,
    pub threshold_at_issue: f64,
    pub target_risk: f64,
    pub learning_rate: f64,
    pub policy_sha256: String,
    pub proxy_delay: i64,
    pub proxy_survival_at_issue: Option<f64>,
    pub calibration_state_before_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcLabelDeliveryV1 {
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
pub struct CaOcrcFeedbackReceiptV1 {
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub measurement_event_id: String,
    pub disposition: CaOcrcFeedbackDispositionV1,
    pub reasons: Vec<CaOcrcFeedbackReasonV1>,
    pub issue_round: Option<u64>,
    pub delivered_round: u64,
    pub actual_delay: Option<u64>,
    pub proxy_delay: Option<i64>,
    pub proxy_survival_probability: Option<f64>,
    pub observed_loss: Option<f64>,
    pub inverse_probability_weight: Option<f64>,
    pub threshold_update: f64,
    pub eligibility_report_sha256: Option<String>,
    pub label_payload_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcRoundReceiptV1 {
    pub schema: String,
    pub engine_id: String,
    pub round: u64,
    pub prediction: CaOcrcPredictionReceiptV1,
    pub admission: CaOcrcAdmissionV1,
    pub evicted_rounds: Vec<u64>,
    pub feedback: Vec<CaOcrcFeedbackReceiptV1>,
    pub trivial_loss: Option<f64>,
    pub feedback_update_sum: f64,
    pub threshold_before: f64,
    pub threshold_after: f64,
    pub tracking_count_before: usize,
    pub tracking_count_after: usize,
    pub accepted_feedback_count: u64,
    pub rejected_feedback_count: u64,
    pub guarantee_state: CaOcrcGuaranteeStateV1,
    pub tracking_set_sha256: String,
    pub state_before_sha256: String,
    pub state_after_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcTrackedPredictionV1 {
    pub prediction: CaOcrcPredictionReceiptV1,
    pub proxy_delay: u64,
    pub saw_unknown_feedback: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub clock_domain_id: String,
    pub policy: CaOcrcPolicyV1,
    pub policy_sha256: String,
    pub threshold: f64,
    pub current_round: Option<u64>,
    pub tracking_set: BTreeMap<u64, CaOcrcTrackedPredictionV1>,
    pub accepted_label_revisions: BTreeSet<String>,
    pub accepted_feedback_count: u64,
    pub rejected_feedback_count: u64,
    pub evicted_feedback_count: u64,
    pub untracked_feedback_count: u64,
    pub prng_state: u64,
    pub guarantee_state: CaOcrcGuaranteeStateV1,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct CaOcrcErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), CaOcrcErrorV1> {
    if value.trim().is_empty() {
        return Err(CaOcrcErrorV1(format!("{field}: expected non-empty text")));
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

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, CaOcrcErrorV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| CaOcrcErrorV1(format!("canonical JSON serialization failed: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn policy_digest(policy: &CaOcrcPolicyV1) -> Result<String, CaOcrcErrorV1> {
    json_sha256(policy)
}

fn report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, CaOcrcErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn state_digest(engine: &CaOcrcEngineV1) -> Result<String, CaOcrcErrorV1> {
    #[derive(Serialize)]
    struct State<'a> {
        threshold: f64,
        current_round: Option<u64>,
        tracking_set: &'a BTreeMap<u64, CaOcrcTrackedPredictionV1>,
        accepted_label_revisions: &'a BTreeSet<String>,
        accepted_feedback_count: u64,
        rejected_feedback_count: u64,
        evicted_feedback_count: u64,
        untracked_feedback_count: u64,
        guarantee_state: CaOcrcGuaranteeStateV1,
        prng_state: u64,
    }
    json_sha256(&State {
        threshold: engine.threshold,
        current_round: engine.current_round,
        tracking_set: &engine.tracking_set,
        accepted_label_revisions: &engine.accepted_label_revisions,
        accepted_feedback_count: engine.accepted_feedback_count,
        rejected_feedback_count: engine.rejected_feedback_count,
        evicted_feedback_count: engine.evicted_feedback_count,
        untracked_feedback_count: engine.untracked_feedback_count,
        guarantee_state: engine.guarantee_state,
        prng_state: engine.prng_state,
    })
}

fn prediction_digest(
    receipt: &CaOcrcPredictionReceiptV1,
) -> Result<String, CaOcrcErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn round_receipt_digest(
    receipt: &CaOcrcRoundReceiptV1,
) -> Result<String, CaOcrcErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn validate_distribution(
    distribution: &CaOcrcProxyDelayDistributionV1,
    maximum_real_delay: u64,
) -> Result<(), CaOcrcErrorV1> {
    if distribution.schema != "luminous.semi-bench.ca-ocrc-proxy-delay.v1" {
        return Err(CaOcrcErrorV1(
            "proxy delay distribution schema is unsupported".into(),
        ));
    }
    require_text(&distribution.distribution_id, "distribution.distribution_id")?;
    if distribution.maximum_proxy_delay != maximum_real_delay {
        return Err(CaOcrcErrorV1(
            "arbitrary bounded-delay v1 requires maximum proxy delay = maximum real delay".into(),
        ));
    }
    let support_len = maximum_real_delay
        .checked_add(2)
        .ok_or_else(|| CaOcrcErrorV1("proxy delay support length overflow".into()))?;
    let expected_len = usize::try_from(support_len)
        .map_err(|_| CaOcrcErrorV1("proxy delay support length overflow".into()))?;
    if distribution.probability_mass.len() != expected_len {
        return Err(CaOcrcErrorV1(
            "proxy delay PMF must cover delays {-1,0,...,d_max}".into(),
        ));
    }
    let mut total = 0.0;
    for (index, probability) in distribution.probability_mass.iter().enumerate() {
        if !probability.is_finite() || *probability < 0.0 || *probability > 1.0 {
            return Err(CaOcrcErrorV1(format!(
                "proxy PMF probability at index {index} is invalid"
            )));
        }
        total += probability;
    }
    if (total - 1.0).abs() > 1.0e-12 {
        return Err(CaOcrcErrorV1(
            "proxy PMF probabilities must sum to one".into(),
        ));
    }
    let load = proxy_tracking_load_v1(distribution)?;
    Ok(load)
}

pub fn proxy_survival_probability_v1(
    distribution: &CaOcrcProxyDelayDistributionV1,
    delay: u64,
) -> Result<f64, CaOcrcErrorV1> {
    if delay > distribution.maximum_proxy_delay {
        return Ok(0.0);
    }
    let start = usize::try_from(delay + 1)
        .map_err(|_| CaOcrcErrorV1("proxy delay index overflow".into()))?;
    Ok(distribution.probability_mass[start..].iter().sum())
}

pub fn proxy_tracking_load_v1(
    distribution: &CaOcrcProxyDelayDistributionV1,
) -> Result<f64, CaOcrcErrorV1> {
    let mut load = 0.0;
    for delay in 1..=distribution.maximum_proxy_delay {
        load += proxy_survival_probability_v1(distribution, delay)?;
    }
    Ok(load)
}

fn validate_policy(policy: &CaOcrcPolicyV1) -> Result<(), CaOcrcErrorV1> {
    if policy.schema != CA_OCRC_POLICY_SCHEMA_V1 {
        return Err(CaOcrcErrorV1("CA-OCRC policy schema is unsupported".into()));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if policy.revision == 0
        || !policy.target_risk.is_finite()
        || !(0.0..1.0).contains(&policy.target_risk)
        || !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || !policy.score_bound.is_finite()
        || policy.score_bound <= 0.0
        || policy.tracking_capacity == 0
    {
        return Err(CaOcrcErrorV1("CA-OCRC policy parameters are invalid".into()));
    }
    validate_distribution(&policy.proxy_distribution, policy.maximum_real_delay)?;
    let load = proxy_tracking_load_v1(&policy.proxy_distribution)?;
    if load > policy.tracking_capacity as f64 + 1.0e-12 {
        return Err(CaOcrcErrorV1(
            "proxy tracking load exceeds the declared capacity under Assumption 4".into(),
        ));
    }
    Ok(())
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn sample_proxy_delay(
    engine: &mut CaOcrcEngineV1,
) -> Result<i64, CaOcrcErrorV1> {
    let draw = splitmix64_next(&mut engine.prng_state);
    let uniform = (draw >> 11) as f64 / ((1_u64 << 53) as f64);
    let mut cumulative = 0.0;
    for (index, probability) in engine
        .policy
        .proxy_distribution
        .probability_mass
        .iter()
        .enumerate()
    {
        cumulative += probability;
        if uniform < cumulative || index + 1 == engine.policy.proxy_distribution.probability_mass.len() {
            return Ok(index as i64 - 1);
        }
    }
    Err(CaOcrcErrorV1("proxy delay sampler failed".into()))
}

fn classify_set(
    threshold: f64,
    score_bound: f64,
    point_prediction: f64,
) -> Result<(CaOcrcSetKindV1, Option<f64>, Option<f64>, Option<f64>), CaOcrcErrorV1> {
    if threshold.is_finite() && threshold < 0.0 {
        return Ok((CaOcrcSetKindV1::Empty, None, None, Some(1.0)));
    }
    if threshold >= score_bound {
        return Ok((CaOcrcSetKindV1::FullRealLine, None, None, Some(0.0)));
    }
    let lower = point_prediction - threshold;
    let upper = point_prediction + threshold;
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(CaOcrcErrorV1(
            "bounded prediction interval endpoints are invalid".into(),
        ));
    }
    Ok((CaOcrcSetKindV1::BoundedInterval, Some(lower), Some(upper), None))
}

fn validate_engine(engine: &CaOcrcEngineV1) -> Result<(), CaOcrcErrorV1> {
    if engine.schema != CA_OCRC_SCHEMA_V1 {
        return Err(CaOcrcErrorV1("CA-OCRC engine schema is unsupported".into()));
    }
    require_text(&engine.engine_id, "engine.engine_id")?;
    require_text(&engine.clock_domain_id, "engine.clock_domain_id")?;
    validate_policy(&engine.policy)?;
    if policy_digest(&engine.policy)? != engine.policy_sha256 {
        return Err(CaOcrcErrorV1("CA-OCRC policy digest mismatch".into()));
    }
    if !engine.threshold.is_finite()
        || engine.tracking_set.len() > engine.policy.tracking_capacity
        || engine.tracking_set.iter().any(|(round, tracked)| {
            *round != tracked.prediction.round
                || tracked.proxy_delay > engine.policy.maximum_real_delay
                || tracked.prediction.set_kind != CaOcrcSetKindV1::BoundedInterval
        })
    {
        return Err(CaOcrcErrorV1("CA-OCRC engine state is invalid".into()));
    }
    Ok(())
}

pub fn create_ca_ocrc_engine_v1(
    engine_id: &str,
    clock_domain_id: &str,
    initial_threshold: f64,
    policy: CaOcrcPolicyV1,
    seed: u64,
) -> Result<CaOcrcEngineV1, CaOcrcErrorV1> {
    require_text(engine_id, "engine_id")?;
    require_text(clock_domain_id, "clock_domain_id")?;
    if !initial_threshold.is_finite() || initial_threshold < 0.0 || initial_threshold > policy.score_bound {
        return Err(CaOcrcErrorV1(
            "initial threshold must be in [0, score_bound]".into(),
        ));
    }
    validate_policy(&policy)?;
    let policy_sha256 = policy_digest(&policy)?;
    Ok(CaOcrcEngineV1 {
        schema: CA_OCRC_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        clock_domain_id: clock_domain_id.into(),
        policy,
        policy_sha256,
        threshold: initial_threshold,
        current_round: None,
        tracking_set: BTreeMap::new(),
        accepted_label_revisions: BTreeSet::new(),
        accepted_feedback_count: 0,
        rejected_feedback_count: 0,
        evicted_feedback_count: 0,
        untracked_feedback_count: 0,
        prng_state: seed,
        guarantee_state: CaOcrcGuaranteeStateV1::UnderDeclaredArbitraryBoundedDelayAssumptions,
    })
}

fn build_prediction_receipt(
    engine: &CaOcrcEngineV1,
    input: &CaOcrcRoundInputV1,
    proxy_delay: i64,
    before: &str,
) -> Result<CaOcrcPredictionReceiptV1, CaOcrcErrorV1> {
    let (set_kind, interval_lower, interval_upper, _) = classify_set(
        engine.threshold,
        engine.policy.score_bound,
        input.point_prediction,
    )?;
    let mut receipt = CaOcrcPredictionReceiptV1 {
        schema: "luminous.semi-bench.ca-ocrc-prediction-receipt.v1".into(),
        round: input.round,
        sample_id: input.sample_id.clone(),
        measurement_event_id: input.measurement_event_id.clone(),
        clock_domain_id: input.clock_domain_id.clone(),
        model_artifact_sha256: input.model_artifact_sha256.clone(),
        event_time_unix_seconds: input.event_time_unix_seconds,
        prediction_time_unix_seconds: input.prediction_time_unix_seconds,
        point_prediction: input.point_prediction,
        set_kind,
        interval_lower,
        interval_upper,
        threshold_at_issue: engine.threshold,
        target_risk: engine.policy.target_risk,
        learning_rate: engine.policy.learning_rate,
        policy_sha256: engine.policy_sha256.clone(),
        proxy_delay,
        proxy_survival_at_issue: if proxy_delay >= 0 {
            Some(proxy_survival_probability_v1(
                &engine.policy.proxy_distribution,
                proxy_delay as u64,
            )?)
        } else {
            None
        },
        calibration_state_before_sha256: before.into(),
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = prediction_digest(&receipt)?;
    Ok(receipt)
}

fn feedback_receipt(
    delivery: &CaOcrcLabelDeliveryV1,
    disposition: CaOcrcFeedbackDispositionV1,
    reasons: Vec<CaOcrcFeedbackReasonV1>,
    issue_round: Option<u64>,
    actual_delay: Option<u64>,
    proxy_delay: Option<i64>,
    survival: Option<f64>,
    loss: Option<f64>,
    weight: Option<f64>,
    threshold_update: f64,
) -> CaOcrcFeedbackReceiptV1 {
    CaOcrcFeedbackReceiptV1 {
        sample_id: delivery.sample_id.clone(),
        label_id: delivery.label_id.clone(),
        label_revision: delivery.label_revision,
        measurement_event_id: delivery.measurement_event_id.clone(),
        disposition,
        reasons,
        issue_round,
        delivered_round: delivery.delivered_round,
        actual_delay,
        proxy_delay,
        proxy_survival_probability: survival,
        observed_loss: loss,
        inverse_probability_weight: weight,
        threshold_update,
        eligibility_report_sha256: Some(delivery.eligibility_report.report_sha256.clone()),
        label_payload_sha256: delivery.label_payload_sha256.clone(),
    }
}

fn validate_delivery(
    engine: &CaOcrcEngineV1,
    tracked: &CaOcrcTrackedPredictionV1,
    delivery: &CaOcrcLabelDeliveryV1,
    current_round: u64,
) -> Vec<CaOcrcFeedbackReasonV1> {
    let mut reasons = Vec::new();
    let report = &delivery.eligibility_report;
    if delivery.delivered_round != current_round {
        reasons.push(CaOcrcFeedbackReasonV1::WrongDeliveryRound);
    }
    if delivery.delivered_round < tracked.prediction.round {
        reasons.push(CaOcrcFeedbackReasonV1::WrongDeliveryRound);
    }
    let actual_delay = delivery.delivered_round.saturating_sub(tracked.prediction.round);
    if actual_delay > engine.policy.maximum_real_delay {
        reasons.push(CaOcrcFeedbackReasonV1::FeedbackDelayOutOfDeclaredBound);
    }
    if tracked.proxy_delay < 0 || actual_delay > tracked.proxy_delay as u64 {
        reasons.push(CaOcrcFeedbackReasonV1::FeedbackNotCoveredByProxyDelay);
    }
    if delivery.clock_domain_id != engine.clock_domain_id
        || tracked.prediction.clock_domain_id != delivery.clock_domain_id
    {
        reasons.push(CaOcrcFeedbackReasonV1::ClockDomainMismatch);
    }
    if delivery.delivered_at_unix_seconds < delivery.label_available_at_unix_seconds {
        reasons.push(CaOcrcFeedbackReasonV1::LabelNotAvailableByDelivery);
    }
    if delivery.eligibility_report_available_at_unix_seconds
        < tracked.prediction.event_time_unix_seconds
    {
        reasons.push(CaOcrcFeedbackReasonV1::LabelAvailableBeforeMeasurement);
    }
    if delivery.delivered_at_unix_seconds < tracked.prediction.prediction_time_unix_seconds {
        reasons.push(CaOcrcFeedbackReasonV1::DeliveryBeforePredictionTime);
    }
    if delivery.eligibility_report_available_at_unix_seconds
        < delivery.label_available_at_unix_seconds
    {
        reasons.push(CaOcrcFeedbackReasonV1::ReportBeforeLabelAvailability);
    }
    if delivery.delivered_at_unix_seconds
        < delivery.eligibility_report_available_at_unix_seconds
    {
        reasons.push(CaOcrcFeedbackReasonV1::ReportNotAvailableByDelivery);
    }
    if !delivery.label_value.is_finite()
        || !report.qualified_corrected_value.is_finite()
        || report
            .qualified_standard_uncertainty_upper_bound
            .is_none_or(|value| !value.is_finite() || value < 0.0)
    {
        reasons.push(CaOcrcFeedbackReasonV1::InvalidNumericReport);
    }
    if delivery.label_revision == 0 || report.label_revision == 0 {
        reasons.push(CaOcrcFeedbackReasonV1::InvalidLabelRevision);
    }
    if delivery.label_revision != report.label_revision {
        reasons.push(CaOcrcFeedbackReasonV1::LabelRevisionMismatch);
    }
    if delivery.label_id != report.label_id {
        reasons.push(CaOcrcFeedbackReasonV1::LabelIdentityMismatch);
    }
    if delivery.measurement_event_id != report.measurement_event_id
        || delivery.measurement_event_id != tracked.prediction.measurement_event_id
    {
        reasons.push(CaOcrcFeedbackReasonV1::MeasurementEventMismatch);
    }
    if !valid_sha256(&delivery.label_payload_sha256)
        || !valid_sha256(&report.label_sha256)
        || delivery.label_payload_sha256 != report.label_sha256
    {
        reasons.push(CaOcrcFeedbackReasonV1::LabelPayloadDigestMismatch);
    }
    if !report
        .qualification_report_sha256
        .as_str()
        .bytes()
        .all(|byte| byte.is_ascii_hexdigit())
        || report.calibration_certificate_id.trim().is_empty()
        || !valid_sha256(&report.measurement_sha256)
        || !valid_sha256(&report.qualification_report_sha256)
        || !valid_sha256(&report.requalification_report_sha256)
        || !valid_sha256(&report.requalification_frontier_sha256)
        || report.training_snapshot_mutated
        || report.equipment_control_authority
    {
        reasons.push(CaOcrcFeedbackReasonV1::InvalidNumericReport);
    }
    if !matches!(report_digest(report), Ok(digest) if digest == report.report_sha256) {
        reasons.push(CaOcrcFeedbackReasonV1::EligibilityReportDigestMismatch);
    }
    if report.qualified_corrected_value != delivery.label_value {
        reasons.push(CaOcrcFeedbackReasonV1::DeliveredValueMismatch);
    }
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        reasons.push(CaOcrcFeedbackReasonV1::LabelNotCurrentK1);
    }
    if report.eligibility != LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy {
        reasons.push(CaOcrcFeedbackReasonV1::LabelNotEligible);
    }
    if !report.downstream_consumption_allowed {
        reasons.push(CaOcrcFeedbackReasonV1::DownstreamConsumptionNotAllowed);
    }
    if delivery.idempotency_key != format!("{}:{}", delivery.label_id, delivery.label_revision) {
        reasons.push(CaOcrcFeedbackReasonV1::InvalidIdempotencyKey);
    }
    reasons.sort();
    reasons.dedup();
    reasons
}

fn label_loss(prediction: &CaOcrcPredictionReceiptV1, label_value: f64) -> f64 {
    match prediction.set_kind {
        CaOcrcSetKindV1::FullRealLine => 0.0,
        CaOcrcSetKindV1::Empty => 1.0,
        CaOcrcSetKindV1::BoundedInterval => {
            let lower = prediction.interval_lower.expect("bounded interval has lower endpoint");
            let upper = prediction.interval_upper.expect("bounded interval has upper endpoint");
            if label_value < lower || label_value > upper { 1.0 } else { 0.0 }
        }
    }
}

fn tracking_set_digest(
    tracking: &BTreeMap<u64, CaOcrcTrackedPredictionV1>,
) -> Result<String, CaOcrcErrorV1> {
    json_sha256(tracking)
}

fn process_round_inner(
    engine: &mut CaOcrcEngineV1,
    input: &CaOcrcRoundInputV1,
    deliveries: &[CaOcrcLabelDeliveryV1],
) -> Result<CaOcrcRoundReceiptV1, CaOcrcErrorV1> {
    validate_engine(engine)?;
    if input.schema != CA_OCRC_ROUND_INPUT_SCHEMA_V1 {
        return Err(CaOcrcErrorV1("CA-OCRC round input schema is unsupported".into()));
    }
    require_text(&input.sample_id, "input.sample_id")?;
    require_text(&input.measurement_event_id, "input.measurement_event_id")?;
    require_text(&input.clock_domain_id, "input.clock_domain_id")?;
    if input.clock_domain_id != engine.clock_domain_id
        || !valid_sha256(&input.model_artifact_sha256)
        || !input.point_prediction.is_finite()
        || input.event_time_unix_seconds > input.prediction_time_unix_seconds
    {
        return Err(CaOcrcErrorV1("CA-OCRC round input identity, clock or numeric fields are invalid".into()));
    }
    let expected_round = match engine.current_round {
        Some(previous) => previous
            .checked_add(1)
            .ok_or_else(|| CaOcrcErrorV1("round counter overflow".into()))?,
        None => 0,
    };
    if input.round != expected_round {
        return Err(CaOcrcErrorV1(format!(
            "round sequence mismatch: expected {expected_round}, got {}",
            input.round
        )));
    }
    if engine.tracking_set.values().any(|tracked| tracked.prediction.sample_id == input.sample_id)
        || engine.tracking_set.values().any(|tracked| tracked.prediction.measurement_event_id == input.measurement_event_id)
    {
        return Err(CaOcrcErrorV1("duplicate prediction sample/event identity".into()));
    }

    let state_before = state_digest(engine)?;
    let threshold_before = engine.threshold;
    let tracking_count_before = engine.tracking_set.len();

    let expired: Vec<u64> = engine
        .tracking_set
        .iter()
        .filter_map(|(issue_round, tracked)| {
            (issue_round.saturating_add(tracked.proxy_delay) < input.round)
                .then_some(*issue_round)
        })
        .collect();
    for issue_round in &expired {
        if let Some(evicted) = engine.tracking_set.remove(issue_round)
            && evicted.saw_unknown_feedback
        {
            engine.guarantee_state =
                CaOcrcGuaranteeStateV1::UncertifiedInvalidOrOutOfModelFeedback;
        }
        engine.evicted_feedback_count = engine.evicted_feedback_count.saturating_add(1);
    }

    let (_, _, _, trivial_loss) = classify_set(
        engine.threshold,
        engine.policy.score_bound,
        input.point_prediction,
    )?;
    let proxy_delay = if trivial_loss.is_some() {
        -1
    } else {
        sample_proxy_delay(engine)?
    };
    let mut prediction = build_prediction_receipt(engine, input, proxy_delay, &state_before)?;

    let admission = if trivial_loss.is_some() {
        CaOcrcAdmissionV1::TrivialSetNoTracking
    } else if proxy_delay < 0 {
        CaOcrcAdmissionV1::ProxyRejected
    } else if engine.tracking_set.len() >= engine.policy.tracking_capacity {
        CaOcrcAdmissionV1::CapacitySaturated
    } else {
        engine.tracking_set.insert(
            input.round,
            CaOcrcTrackedPredictionV1 {
                prediction: prediction.clone(),
                proxy_delay: proxy_delay as u64,
                saw_unknown_feedback: false,
            },
        );
        CaOcrcAdmissionV1::Admitted
    };

    let mut feedback_receipts = Vec::with_capacity(deliveries.len());
    let mut feedback_update_sum = 0.0;
    let mut received_issue_rounds = BTreeSet::new();
    for delivery in deliveries {
        require_text(&delivery.sample_id, "delivery.sample_id")?;
        require_text(&delivery.label_id, "delivery.label_id")?;
        require_text(&delivery.measurement_event_id, "delivery.measurement_event_id")?;
        let revision_key = format!("{}:{}", delivery.label_id, delivery.label_revision);
        if engine.accepted_label_revisions.contains(&revision_key) {
            feedback_receipts.push(feedback_receipt(
                delivery,
                CaOcrcFeedbackDispositionV1::Duplicate,
                vec![CaOcrcFeedbackReasonV1::DuplicateAcceptedRevision],
                None,
                None,
                None,
                None,
                None,
                None,
                0.0,
            ));
            continue;
        }
        let tracked_match = engine
            .tracking_set
            .iter()
            .find(|(_, tracked)| tracked.prediction.sample_id == delivery.sample_id)
            .map(|(round, tracked)| (*round, tracked.clone()));
        let Some((issue_round, tracked)) = tracked_match else {
            engine.untracked_feedback_count = engine.untracked_feedback_count.saturating_add(1);
            feedback_receipts.push(feedback_receipt(
                delivery,
                CaOcrcFeedbackDispositionV1::Untracked,
                vec![CaOcrcFeedbackReasonV1::NoTrackedPrediction],
                None,
                None,
                None,
                None,
                None,
                None,
                0.0,
            ));
            continue;
        };
        if !received_issue_rounds.insert(issue_round) {
            return Err(CaOcrcErrorV1(
                "two feedback records target the same tracked prediction in one round".into(),
            ));
        }
        let reasons = validate_delivery(engine, &tracked, delivery, input.round);
        if !reasons.is_empty() {
            let status = delivery.eligibility_report.eligibility;
            let is_eligibility_only = |reason: &CaOcrcFeedbackReasonV1| {
                matches!(
                    reason,
                    CaOcrcFeedbackReasonV1::LabelNotEligible
                        | CaOcrcFeedbackReasonV1::DownstreamConsumptionNotAllowed
                )
            };
            let only_unknown = status == LabelConsumptionEligibilityV1::Unknown
                && reasons.iter().all(is_eligibility_only);
            let only_blocked = status == LabelConsumptionEligibilityV1::Blocked
                && reasons.iter().all(is_eligibility_only);
            let disposition = if only_unknown {
                if let Some(pending) = engine.tracking_set.get_mut(&issue_round) {
                    pending.saw_unknown_feedback = true;
                }
                CaOcrcFeedbackDispositionV1::HeldUnknown
            } else if only_blocked {
                engine.guarantee_state =
                    CaOcrcGuaranteeStateV1::UncertifiedInvalidOrOutOfModelFeedback;
                engine.rejected_feedback_count =
                    engine.rejected_feedback_count.saturating_add(1);
                CaOcrcFeedbackDispositionV1::Blocked
            } else {
                engine.guarantee_state =
                    CaOcrcGuaranteeStateV1::UncertifiedInvalidOrOutOfModelFeedback;
                engine.rejected_feedback_count =
                    engine.rejected_feedback_count.saturating_add(1);
                CaOcrcFeedbackDispositionV1::Rejected
            };
            feedback_receipts.push(feedback_receipt(
                delivery,
                disposition,
                reasons,
                Some(issue_round),
                Some(input.round - issue_round),
                Some(tracked.proxy_delay as i64),
                Some(proxy_survival_probability_v1(
                    &engine.policy.proxy_distribution,
                    input.round - issue_round,
                )?),
                None,
                None,
                0.0,
            ));
            continue;
        }

        let actual_delay = input.round - issue_round;
        let survival = proxy_survival_probability_v1(
            &engine.policy.proxy_distribution,
            actual_delay,
        )?;
        if survival <= 0.0 || actual_delay > tracked.proxy_delay {
            engine.guarantee_state =
                CaOcrcGuaranteeStateV1::UncertifiedInvalidOrOutOfModelFeedback;
            engine.rejected_feedback_count =
                engine.rejected_feedback_count.saturating_add(1);
            feedback_receipts.push(feedback_receipt(
                delivery,
                CaOcrcFeedbackDispositionV1::Rejected,
                vec![CaOcrcFeedbackReasonV1::FeedbackNotCoveredByProxyDelay],
                Some(issue_round),
                Some(actual_delay),
                Some(tracked.proxy_delay as i64),
                Some(survival),
                None,
                None,
                0.0,
            ));
            continue;
        }
        let loss = label_loss(&tracked.prediction, delivery.label_value);
        let weight = 1.0 / survival;
        let threshold_update = engine.policy.learning_rate
            * (loss - engine.policy.target_risk)
            * weight;
        if !threshold_update.is_finite() {
            return Err(CaOcrcErrorV1("CA-OCRC IPW threshold update is non-finite".into()));
        }
        feedback_update_sum += threshold_update;
        engine.tracking_set.remove(&issue_round);
        engine
            .accepted_label_revisions
            .insert(revision_key);
        engine.accepted_feedback_count = engine.accepted_feedback_count.saturating_add(1);
        feedback_receipts.push(feedback_receipt(
            delivery,
            CaOcrcFeedbackDispositionV1::Updated,
            Vec::new(),
            Some(issue_round),
            Some(actual_delay),
            Some(tracked.proxy_delay as i64),
            Some(survival),
            Some(loss),
            Some(weight),
            threshold_update,
        ));
    }

    let trivial_update = trivial_loss
        .map(|loss| engine.policy.learning_rate * (loss - engine.policy.target_risk))
        .unwrap_or(0.0);
    let threshold_after = threshold_before + trivial_update + feedback_update_sum;
    if !threshold_after.is_finite() {
        return Err(CaOcrcErrorV1("CA-OCRC threshold update overflows".into()));
    }
    engine.threshold = threshold_after;
    engine.current_round = Some(input.round);

    let tracking_digest = tracking_set_digest(&engine.tracking_set)?;
    let state_after = state_digest(engine)?;
    prediction.receipt_sha256 = prediction_digest(&prediction)?;
    let mut receipt = CaOcrcRoundReceiptV1 {
        schema: CA_OCRC_ROUND_RECEIPT_SCHEMA_V1.into(),
        engine_id: engine.engine_id.clone(),
        round: input.round,
        prediction,
        admission,
        evicted_rounds: expired,
        feedback: feedback_receipts,
        trivial_loss,
        feedback_update_sum,
        threshold_before,
        threshold_after,
        tracking_count_before,
        tracking_count_after: engine.tracking_set.len(),
        accepted_feedback_count: engine.accepted_feedback_count,
        rejected_feedback_count: engine.rejected_feedback_count,
        guarantee_state: engine.guarantee_state,
        tracking_set_sha256: tracking_digest,
        state_before_sha256: state_before,
        state_after_sha256: state_after,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = round_receipt_digest(&receipt)?;
    Ok(receipt)
}

/// Processes one complete CA-OCRC round: prediction, proxy eviction, admission,
/// qualified feedback reception, and the exact threshold update. State changes
/// are atomic: the live engine is replaced only after the round receipt seals.
pub fn process_ca_ocrc_round_v1(
    engine: &mut CaOcrcEngineV1,
    input: &CaOcrcRoundInputV1,
    deliveries: &[CaOcrcLabelDeliveryV1],
) -> Result<CaOcrcRoundReceiptV1, CaOcrcErrorV1> {
    let mut next_engine = engine.clone();
    let receipt = process_round_inner(&mut next_engine, input, deliveries)?;
    *engine = next_engine;
    Ok(receipt)
}

/// One-dimensional optimizer for Proposition 1 of arXiv:2611.17059.
/// Returns p* for the Bernoulli-type survival F(a)=p for a=1..d_max.
pub fn optimal_arbitrary_proxy_survival_v1(
    capacity: usize,
    maximum_delay: u64,
    horizon: u64,
) -> Result<f64, CaOcrcErrorV1> {
    if capacity == 0 || horizon == 0 {
        return Err(CaOcrcErrorV1("capacity and horizon must be positive".into()));
    }
    if maximum_delay == 0 {
        return Ok(1.0);
    }
    let upper = (capacity as f64 / maximum_delay as f64).min(1.0);
    if upper <= 0.0 {
        return Err(CaOcrcErrorV1("proxy survival search domain is empty".into()));
    }
    let objective = |p: f64| {
        let x = maximum_delay as f64 * p / capacity as f64;
        let saturation = if x <= 0.0 {
            f64::INFINITY
        } else {
            (-capacity as f64 * (x - 1.0 - x.ln())).exp()
        };
        (capacity as f64 + 1.0) / (p * horizon as f64) + saturation
    };
    let mut left = upper * 1.0e-12;
    let mut right = upper;
    let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut x1 = right - ratio * (right - left);
    let mut x2 = left + ratio * (right - left);
    let mut f1 = objective(x1);
    let mut f2 = objective(x2);
    for _ in 0..160 {
        if f1 <= f2 {
            right = x2;
            x2 = x1;
            f2 = f1;
            x1 = right - ratio * (right - left);
            f1 = objective(x1);
        } else {
            left = x1;
            x1 = x2;
            f1 = f2;
            x2 = left + ratio * (right - left);
            f2 = objective(x2);
        }
    }
    let result = (left + right) / 2.0;
    if !result.is_finite() || result <= 0.0 || result > upper + 1.0e-12 {
        return Err(CaOcrcErrorV1("proxy survival optimizer failed".into()));
    }
    Ok(result)
}

pub fn bernoulli_proxy_distribution_v1(
    distribution_id: &str,
    maximum_delay: u64,
    survival_probability: f64,
) -> Result<CaOcrcProxyDelayDistributionV1, CaOcrcErrorV1> {
    require_text(distribution_id, "distribution_id")?;
    if !survival_probability.is_finite()
        || survival_probability <= 0.0
        || survival_probability > 1.0
    {
        return Err(CaOcrcErrorV1(
            "Bernoulli proxy survival probability must lie in (0,1]".into(),
        ));
    }
    let support_len = maximum_delay
        .checked_add(2)
        .ok_or_else(|| CaOcrcErrorV1("proxy PMF support length overflow".into()))?;
    let length = usize::try_from(support_len)
        .map_err(|_| CaOcrcErrorV1("proxy PMF support length overflow".into()))?;
    let mut probability_mass = vec![0.0; length];
    probability_mass[0] = 1.0 - survival_probability;
    probability_mass[length - 1] = survival_probability;
    Ok(CaOcrcProxyDelayDistributionV1 {
        schema: "luminous.semi-bench.ca-ocrc-proxy-delay.v1".into(),
        distribution_id: distribution_id.into(),
        maximum_proxy_delay: maximum_delay,
        probability_mass,
    })
}

/// Theorem 1 finite-horizon bound for arbitrary delays, assuming the input
/// policy satisfies Assumption 4 (tracking load <= capacity).
pub fn arbitrary_delay_ca_ocrc_risk_bound_v1(
    score_bound: f64,
    learning_rate: f64,
    horizon: u64,
    capacity: usize,
    distribution: &CaOcrcProxyDelayDistributionV1,
) -> Result<f64, CaOcrcErrorV1> {
    if !score_bound.is_finite()
        || score_bound <= 0.0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
        || horizon == 0
        || capacity == 0
    {
        return Err(CaOcrcErrorV1("risk-bound parameters are invalid".into()));
    }
    let load = proxy_tracking_load_v1(distribution)?;
    if load > capacity as f64 + 1.0e-12 {
        return Err(CaOcrcErrorV1("proxy tracking load exceeds capacity".into()));
    }
    let last_survival =
        proxy_survival_probability_v1(distribution, distribution.maximum_proxy_delay)?;
    if last_survival <= 0.0 {
        return Err(CaOcrcErrorV1(
            "F(d_max) must be positive for the arbitrary-delay bound".into(),
        ));
    }
    let saturation = if load <= 0.0 {
        0.0
    } else {
        let x = load / capacity as f64;
        if x <= 0.0 {
            0.0
        } else {
            (-(capacity as f64) * (x - 1.0 - x.ln())).exp()
        }
    };
    Ok(score_bound / (learning_rate * horizon as f64)
        + (capacity as f64 + 1.0) / (last_survival * horizon as f64)
        + saturation)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy(proxy: CaOcrcProxyDelayDistributionV1, capacity: usize) -> CaOcrcPolicyV1 {
        CaOcrcPolicyV1 {
            schema: CA_OCRC_POLICY_SCHEMA_V1.into(),
            policy_id: "ca-ocrc-arbitrary-delay.v1".into(),
            revision: 1,
            target_risk: 0.10,
            learning_rate: 0.20,
            score_bound: 10.0,
            tracking_capacity: capacity,
            maximum_real_delay: proxy.maximum_proxy_delay,
            proxy_distribution: proxy,
        }
    }

    fn engine(proxy: CaOcrcProxyDelayDistributionV1, capacity: usize, seed: u64) -> CaOcrcEngineV1 {
        create_ca_ocrc_engine_v1(
            "ca-ocrc-engine-1",
            "fab-clock",
            0.0,
            policy(proxy, capacity),
            seed,
        )
        .unwrap()
    }

    fn pmf_always_delay(maximum_delay: u64) -> CaOcrcProxyDelayDistributionV1 {
        let length = maximum_delay as usize + 2;
        let mut pmf = vec![0.0; length];
        pmf[length - 1] = 1.0;
        CaOcrcProxyDelayDistributionV1 {
            schema: "luminous.semi-bench.ca-ocrc-proxy-delay.v1".into(),
            distribution_id: "always-maximum-delay".into(),
            maximum_proxy_delay: maximum_delay,
            probability_mass: pmf,
        }
    }

    fn round(round: u64, sample: &str, event: &str, point: f64) -> CaOcrcRoundInputV1 {
        CaOcrcRoundInputV1 {
            schema: CA_OCRC_ROUND_INPUT_SCHEMA_V1.into(),
            round,
            sample_id: sample.into(),
            measurement_event_id: event.into(),
            clock_domain_id: "fab-clock".into(),
            model_artifact_sha256: SHA_A.into(),
            event_time_unix_seconds: 100 + round as i64 * 10,
            prediction_time_unix_seconds: 101 + round as i64 * 10,
            point_prediction: point,
        }
    }

    fn label_report(
        label: &str,
        revision: u64,
        event: &str,
        payload: &str,
        value: f64,
        status: LabelConsumptionEligibilityV1,
        frontier: LabelEvidenceFrontierV1,
    ) -> LabelConsumptionEligibilityReportV1 {
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label.into(),
            label_revision: revision,
            measurement_event_id: event.into(),
            calibration_certificate_id: "cal-cert-ca-ocrc".into(),
            frontier,
            eligibility: status,
            downstream_consumption_allowed:
                status == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: payload.into(),
            measurement_sha256: SHA_B.into(),
            qualified_corrected_value: value,
            qualified_standard_uncertainty_upper_bound: Some(0.05),
            qualification_report_sha256: SHA_A.into(),
            requalification_report_sha256: SHA_B.into(),
            requalification_frontier_sha256: SHA_C.into(),
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
        label: &str,
        revision: u64,
        event: &str,
        value: f64,
        issue_round: u64,
        delivered_round: u64,
        status: LabelConsumptionEligibilityV1,
    ) -> CaOcrcLabelDeliveryV1 {
        let payload = digest('d');
        let available = 100 + delivered_round as i64 * 10;
        CaOcrcLabelDeliveryV1 {
            schema: CA_OCRC_DELIVERY_SCHEMA_V1.into(),
            sample_id: sample.into(),
            label_id: label.into(),
            label_revision: revision,
            measurement_event_id: event.into(),
            label_payload_sha256: payload.clone(),
            label_value: value,
            label_available_at_unix_seconds: available,
            eligibility_report_available_at_unix_seconds: available,
            delivered_at_unix_seconds: available + 1,
            delivered_round,
            clock_domain_id: "fab-clock".into(),
            idempotency_key: format!("{label}:{revision}"),
            eligibility_report: label_report(
                label,
                revision,
                event,
                &payload,
                value,
                status,
                LabelEvidenceFrontierV1::CurrentAtK1,
            ),
        }
    }

    #[test]
    fn capacity_limit_proxy_eviction_and_paper_order_are_exact() {
        let mut engine = engine(pmf_always_delay(2), 2, 7);
        let r0 = process_ca_ocrc_round_v1(&mut engine, &round(0, "s0", "e0", 100.0), &[]).unwrap();
        assert_eq!(r0.admission, CaOcrcAdmissionV1::Admitted);
        assert_eq!(r0.tracking_count_after, 1);

        let r1 = process_ca_ocrc_round_v1(&mut engine, &round(1, "s1", "e1", 100.0), &[]).unwrap();
        assert_eq!(r1.admission, CaOcrcAdmissionV1::Admitted);
        assert_eq!(r1.tracking_count_after, 2);

        let r2 = process_ca_ocrc_round_v1(&mut engine, &round(2, "s2", "e2", 100.0), &[]).unwrap();
        assert_eq!(r2.admission, CaOcrcAdmissionV1::CapacitySaturated);
        assert!(r2.evicted_rounds.is_empty());
        assert_eq!(r2.tracking_count_after, 2);

        let r3 = process_ca_ocrc_round_v1(&mut engine, &round(3, "s3", "e3", 100.0), &[]).unwrap();
        assert_eq!(r3.evicted_rounds, vec![0]);
        assert_eq!(r3.admission, CaOcrcAdmissionV1::Admitted);
        assert_eq!(r3.tracking_count_after, 2);
        assert!(r3.tracking_count_after <= engine.policy.tracking_capacity);
    }

    #[test]
    fn zero_delay_feedback_is_received_at_end_of_same_round() {
        let mut engine = engine(pmf_always_delay(0), 1, 1);
        let input = round(0, "s0", "e0", 10.0);
        let label = delivery(
            "s0",
            "l0",
            1,
            "e0",
            12.0,
            0,
            0,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        );
        let receipt = process_ca_ocrc_round_v1(&mut engine, &input, &[label]).unwrap();
        assert_eq!(receipt.admission, CaOcrcAdmissionV1::Admitted);
        assert_eq!(receipt.feedback.len(), 1);
        assert_eq!(receipt.feedback[0].disposition, CaOcrcFeedbackDispositionV1::Updated);
        assert_eq!(receipt.feedback[0].actual_delay, Some(0));
        assert_eq!(receipt.feedback[0].proxy_survival_probability, Some(1.0));
        assert_eq!(receipt.feedback[0].inverse_probability_weight, Some(1.0));
        assert!((engine.threshold - 0.18).abs() < 1.0e-12);
        assert!(engine.tracking_set.is_empty());
    }

    #[test]
    fn delayed_feedback_uses_inverse_proxy_survival_weight() {
        let mut distribution = pmf_always_delay(2);
        distribution.probability_mass = vec![0.5, 0.0, 0.0, 0.5];
        let mut engine = engine(distribution, 2, 4);
        process_ca_ocrc_round_v1(&mut engine, &round(0, "s0", "e0", 10.0), &[]).unwrap();
        process_ca_ocrc_round_v1(&mut engine, &round(1, "s1", "e1", 10.0), &[]).unwrap();
        let label = delivery(
            "s0",
            "l0",
            1,
            "e0",
            12.0,
            0,
            2,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        );
        let receipt = process_ca_ocrc_round_v1(&mut engine, &round(2, "s2", "e2", 10.0), &[label]).unwrap();
        let item = receipt
            .feedback
            .iter()
            .find(|feedback| feedback.sample_id == "s0")
            .unwrap();
        assert_eq!(item.disposition, CaOcrcFeedbackDispositionV1::Updated);
        assert_eq!(item.actual_delay, Some(2));
        assert_eq!(item.proxy_survival_probability, Some(0.5));
        assert_eq!(item.inverse_probability_weight, Some(2.0));
        assert!((item.threshold_update - 0.36).abs() < 1.0e-12);
    }

    #[test]
    fn full_and_empty_sets_update_from_known_losses_without_tracking() {
        let mut full_engine = engine(pmf_always_delay(2), 1, 1);
        full_engine.threshold = 10.0;
        let full = process_ca_ocrc_round_v1(
            &mut full_engine,
            &round(0, "full", "full-event", 10.0),
            &[],
        )
        .unwrap();
        assert_eq!(full.prediction.set_kind, CaOcrcSetKindV1::FullRealLine);
        assert_eq!(full.admission, CaOcrcAdmissionV1::TrivialSetNoTracking);
        assert_eq!(full.trivial_loss, Some(0.0));
        assert!((full.threshold_after - 9.98).abs() < 1.0e-12);
        assert!(full_engine.tracking_set.is_empty());

        let mut empty_engine = engine(pmf_always_delay(2), 1, 1);
        empty_engine.threshold = -0.1;
        let empty = process_ca_ocrc_round_v1(
            &mut empty_engine,
            &round(0, "empty", "empty-event", 10.0),
            &[],
        )
        .unwrap();
        assert_eq!(empty.prediction.set_kind, CaOcrcSetKindV1::Empty);
        assert_eq!(empty.trivial_loss, Some(1.0));
        assert!((empty.threshold_after - 0.08).abs() < 1.0e-12);
        assert!(empty_engine.tracking_set.is_empty());
    }

    #[test]
    fn unknown_feedback_is_held_and_can_be_represented_as_qualified_before_expiry() {
        let mut engine = engine(pmf_always_delay(2), 2, 1);
        process_ca_ocrc_round_v1(&mut engine, &round(0, "s0", "e0", 10.0), &[]).unwrap();
        let unknown = delivery(
            "s0",
            "l0",
            1,
            "e0",
            10.5,
            0,
            1,
            LabelConsumptionEligibilityV1::Unknown,
        );
        let held = process_ca_ocrc_round_v1(
            &mut engine,
            &round(1, "s1", "e1", 10.0),
            &[unknown],
        )
        .unwrap();
        assert_eq!(
            held.feedback[0].disposition,
            CaOcrcFeedbackDispositionV1::HeldUnknown
        );
        assert!((engine.threshold - 0.0).abs() < 1.0e-12);
        assert!(engine.tracking_set.contains_key(&0));

        let qualified = delivery(
            "s0",
            "l0",
            1,
            "e0",
            10.5,
            0,
            2,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        );
        let accepted = process_ca_ocrc_round_v1(
            &mut engine,
            &round(2, "s2", "e2", 10.0),
            &[qualified],
        )
        .unwrap();
        assert_eq!(
            accepted.feedback[0].disposition,
            CaOcrcFeedbackDispositionV1::Updated
        );
        assert!(engine.tracking_set.is_empty());
    }

    #[test]
    fn blocked_or_mismatched_feedback_never_updates_threshold() {
        let mut engine = engine(pmf_always_delay(2), 1, 3);
        process_ca_ocrc_round_v1(&mut engine, &round(0, "s0", "e0", 10.0), &[]).unwrap();
        let blocked = delivery(
            "s0",
            "l0",
            1,
            "e0",
            12.0,
            0,
            1,
            LabelConsumptionEligibilityV1::Blocked,
        );
        let receipt = process_ca_ocrc_round_v1(
            &mut engine,
            &round(1, "s1", "e1", 10.0),
            &[blocked],
        )
        .unwrap();
        assert_eq!(receipt.feedback[0].disposition, CaOcrcFeedbackDispositionV1::Blocked);
        assert_eq!(receipt.threshold_after, 0.0);
        assert_eq!(
            engine.guarantee_state,
            CaOcrcGuaranteeStateV1::UncertifiedInvalidOrOutOfModelFeedback
        );
    }

    #[test]
    fn expired_or_not_admitted_feedback_is_untracked_and_does_not_update() {
        let mut engine = engine(pmf_always_delay(1), 1, 7);
        process_ca_ocrc_round_v1(&mut engine, &round(0, "s0", "e0", 10.0), &[]).unwrap();
        process_ca_ocrc_round_v1(&mut engine, &round(1, "s1", "e1", 10.0), &[]).unwrap();
        let late = delivery(
            "s0",
            "l0",
            1,
            "e0",
            12.0,
            0,
            2,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        );
        let receipt = process_ca_ocrc_round_v1(
            &mut engine,
            &round(2, "s2", "e2", 10.0),
            &[late],
        )
        .unwrap();
        let feedback = receipt.feedback.iter().find(|x| x.sample_id == "s0").unwrap();
        assert_eq!(feedback.disposition, CaOcrcFeedbackDispositionV1::Untracked);
        assert_eq!(receipt.threshold_after, 0.0);
    }

    #[test]
    fn duplicate_label_revision_is_idempotent() {
        let mut engine = engine(pmf_always_delay(0), 1, 7);
        let accepted = delivery(
            "s0",
            "l0",
            1,
            "e0",
            12.0,
            0,
            0,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        );
        process_ca_ocrc_round_v1(
            &mut engine,
            &round(0, "s0", "e0", 10.0),
            std::slice::from_ref(&accepted),
        )
        .unwrap();
        let before = engine.threshold;
        let duplicate = delivery(
            "s1",
            "l0",
            1,
            "e1",
            12.0,
            1,
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        );
        let receipt = process_ca_ocrc_round_v1(
            &mut engine,
            &round(1, "s1", "e1", 10.0),
            &[duplicate],
        )
        .unwrap();
        assert_eq!(
            receipt.feedback[0].disposition,
            CaOcrcFeedbackDispositionV1::Duplicate
        );
        assert_eq!(engine.threshold, before);
    }

    #[test]
    fn proxy_pmf_survival_load_and_capacity_validation_are_exact() {
        let bernoulli = bernoulli_proxy_distribution_v1("p", 3, 0.25).unwrap();
        assert!((proxy_survival_probability_v1(&bernoulli, 0).unwrap() - 0.25).abs() < 1.0e-12);
        assert!((proxy_survival_probability_v1(&bernoulli, 3).unwrap() - 0.25).abs() < 1.0e-12);
        assert!((proxy_tracking_load_v1(&bernoulli).unwrap() - 0.75).abs() < 1.0e-12);
        assert!(validate_policy(&policy(bernoulli, 1)).is_ok());
        assert!(validate_policy(&policy(pmf_always_delay(3), 2)).is_err());
    }

    #[test]
    fn optimized_arbitrary_proxy_probability_matches_grid_oracle() {
        let capacity = 2;
        let dmax = 5;
        let horizon = 500;
        let optimum = optimal_arbitrary_proxy_survival_v1(capacity, dmax, horizon).unwrap();
        let objective = |p: f64| {
            let x = dmax as f64 * p / capacity as f64;
            (capacity as f64 + 1.0) / (p * horizon as f64)
                + (-capacity as f64 * (x - 1.0 - x.ln())).exp()
        };
        let grid_best = (1..=200_000)
            .map(|i| i as f64 / 200_000.0 * (capacity as f64 / dmax as f64))
            .min_by(|left, right| objective(*left).total_cmp(&objective(*right)))
            .unwrap();
        assert!((objective(optimum) - objective(grid_best)).abs() < 1.0e-5);
        let distribution = bernoulli_proxy_distribution_v1("optimal", dmax, optimum).unwrap();
        assert!(proxy_tracking_load_v1(&distribution).unwrap() <= capacity as f64 + 1.0e-12);
    }

    #[test]
    fn arbitrary_delay_risk_bound_matches_known_formula() {
        let distribution = bernoulli_proxy_distribution_v1("p", 2, 0.5).unwrap();
        let bound = arbitrary_delay_ca_ocrc_risk_bound_v1(10.0, 0.2, 100, 2, &distribution).unwrap();
        let load = 1.0;
        let h = load / 2.0 - 1.0 - (load / 2.0_f64).ln();
        let expected = 10.0 / (0.2 * 100.0) + 3.0 / (0.5 * 100.0) + (-2.0 * h).exp();
        assert!((bound - expected).abs() < 1.0e-12);
    }

    #[test]
    fn fixed_schedule_replay_is_byte_deterministic_and_capacity_never_exceeded() {
        fn run() -> (String, Vec<f64>) {
            let proxy = bernoulli_proxy_distribution_v1("p", 3, 0.5).unwrap();
            let mut engine = engine(proxy, 2, 0x1234);
            let mut receipts = Vec::new();
            for round_index in 0..50 {
                let receipt = process_ca_ocrc_round_v1(
                    &mut engine,
                    &round(round_index, &format!("s{round_index}"), &format!("e{round_index}"), 10.0),
                    &[],
                )
                .unwrap();
                assert!(receipt.tracking_count_after <= engine.policy.tracking_capacity);
                receipts.push(receipt.receipt_sha256);
            }
            (json_sha256(&receipts).unwrap(), engine.threshold_history_for_test())
        }
        let first = run();
        let second = run();
        assert_eq!(first, second);
    }
}

#[cfg(test)]
trait TestThresholdHistory {
    fn threshold_history_for_test(&self) -> Vec<f64>;
}

#[cfg(test)]
impl TestThresholdHistory for CaOcrcEngineV1 {
    fn threshold_history_for_test(&self) -> Vec<f64> {
        vec![self.threshold]
    }
}
