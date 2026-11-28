// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capacity-aware online conformal risk control (CA-OCRC) for bounded delays.
//!
//! Implements the arbitrary-bounded-delay policy of Wang & Simeone (2026):
//! proxy-delay sampling, expiry/eviction, capacity-limited admission and the
//! inverse-probability-weighted threshold update. K1 evidence qualification
//! is an additional application gate; an unqualified outcome suspends the
//! published CA-OCRC guarantee because it is not ordinary random delay loss.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
    LabelEvidenceFrontierV1,
};

pub const CA_OCRC_ENGINE_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-engine.v1";
pub const CA_OCRC_POLICY_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-policy.v1";
pub const CA_OCRC_ROUND_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-round.v1";
pub const CA_OCRC_FEEDBACK_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-feedback.v1";
pub const CA_OCRC_UPDATE_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.ca-ocrc-update-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcPredictionKindV1 {
    KnownEmpty,
    Nontrivial,
    KnownFull,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcGuaranteeStateV1 {
    ActiveUnderBoundedDelayAndUnderload,
    SuspendedByUnqualifiedFeedback,
    SuspendedByDelayAssumptionViolation,
    SuspendedByNumericOrProtocolViolation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcTrackingDispositionV1 {
    Admitted,
    NotTrackedByRandomizedPolicy,
    NotAdmittedCapacityFull,
    ImmediateKnownLossApplied,
    FeedbackUpdated,
    Duplicate,
    FeedbackNotTracked,
    HeldUnknown,
    Blocked,
    Rejected,
    GuaranteeSuspended,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcReasonV1 {
    InvalidRandomDraw,
    PredictionRoundOutOfOrder,
    InvalidPredictionIdentity,
    IntervalOrPredictionKindMismatch,
    FeedbackDelayExceedsBound,
    FeedbackRoundMismatch,
    FeedbackNotInTrackingSet,
    DuplicateAcceptedRevision,
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
    ClockDomainMismatch,
    LabelAvailabilityBeforePrediction,
    ReportBeforeLabelAvailability,
    DeliveryBeforeLabelAvailability,
    ReportUnavailableAtDelivery,
    SourceAuthenticityAssertionMissing,
    UpstreamAuthorityFlagSet,
    ZeroOrInvalidInclusionProbability,
    InverseProbabilityWeightTooLarge,
    NonFiniteThresholdUpdate,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub target_risk: f64,
    pub learning_rate: f64,
    pub score_bound_b: f64,
    pub score_contract_sha256: String,
    pub initial_threshold: f64,
    pub tracking_capacity: usize,
    pub maximum_feedback_delay: u64,
    /// PMF indexed by proxy_delay + 1: index 0 is delay -1, index 1 is 0, etc.
    pub proxy_delay_probabilities: Vec<f64>,
    pub require_source_authenticity_assertion: bool,
    /// Defensive ceiling. The theorem uses the exact inverse propensity;
    /// if this ceiling is exceeded, the controller suspends rather than clips.
    pub maximum_inverse_probability_weight: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcRoundInputV1 {
    pub schema: String,
    pub round_index: u64,
    pub prediction_id: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub prediction_kind: CaOcrcPredictionKindV1,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
    /// Caller-supplied U[0,1) draw, making randomized tracking replayable.
    pub uniform_draw: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcTrackedRoundV1 {
    pub round_index: u64,
    pub prediction_id: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub proxy_delay: u64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub clock_domain_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcTrackingDecisionV1 {
    pub schema: String,
    pub round_index: u64,
    pub prediction_id: String,
    pub prediction_kind: CaOcrcPredictionKindV1,
    pub threshold_used: f64,
    pub sampled_proxy_delay: Option<i64>,
    pub admitted: bool,
    pub expired_rounds: Vec<u64>,
    pub active_tracking_count: usize,
    pub tracking_capacity: usize,
    pub tracking_load_ell_f: f64,
    pub guarantee_state: CaOcrcGuaranteeStateV1,
    pub immediate_loss: Option<f64>,
    pub threshold_after_round: f64,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcFeedbackV1 {
    pub schema: String,
    pub current_round: u64,
    pub origin_round: u64,
    pub actual_delay: u64,
    pub prediction_id: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub label_payload_sha256: String,
    pub label_value: f64,
    pub label_available_at_unix_seconds: i64,
    pub eligibility_report_available_at_unix_seconds: i64,
    pub delivered_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcFeedbackReceiptV1 {
    pub schema: String,
    pub engine_id: String,
    pub origin_round: u64,
    pub current_round: u64,
    pub label_id: String,
    pub label_revision: u64,
    pub disposition: CaOcrcTrackingDispositionV1,
    pub reasons: Vec<CaOcrcReasonV1>,
    pub observed_loss: Option<f64>,
    pub proxy_survival_probability: Option<f64>,
    pub inverse_probability_weight: Option<f64>,
    pub threshold_before: f64,
    pub threshold_after: f64,
    pub active_tracking_count: usize,
    pub guarantee_state_before: CaOcrcGuaranteeStateV1,
    pub guarantee_state_after: CaOcrcGuaranteeStateV1,
    pub threshold_mutated: bool,
    pub physical_source_authenticity_proven: bool,
    pub equipment_control_authority: bool,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub clock_domain_id: String,
    pub policy: CaOcrcPolicyV1,
    pub policy_sha256: String,
    pub threshold: f64,
    pub next_round: u64,
    pub current_round: Option<u64>,
    pub tracked_rounds: BTreeMap<u64, CaOcrcTrackedRoundV1>,
    pub accepted_label_revisions: std::collections::BTreeSet<String>,
    pub guarantee_state: CaOcrcGuaranteeStateV1,
    pub suspension_reasons: Vec<String>,
    pub capacity_saturation_count: u64,
    pub randomized_nontracking_count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcRiskGapBoundV1 {
    pub horizon: u64,
    pub boundary_term: f64,
    pub feedback_term: f64,
    pub capacity_saturation_term: f64,
    pub total_upper_bound: f64,
    pub tracking_load_ell_f: f64,
    pub proxy_survival_at_max_delay: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcOptimizedProxyPolicyV1 {
    pub maximum_delay: u64,
    pub capacity: usize,
    pub horizon: u64,
    pub optimal_tracking_probability: f64,
    /// PMF indexed by proxy_delay + 1, with mass at -1 and d_max.
    pub proxy_delay_probabilities: Vec<f64>,
    pub theoretical_objective: f64,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct CaOcrcErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), CaOcrcErrorV1> {
    if value.trim().is_empty() {
        return Err(CaOcrcErrorV1(format!("{field}: expected non-empty text")));
    }
    Ok(())
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, CaOcrcErrorV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| CaOcrcErrorV1(format!("canonical JSON serialization failed: {error}")))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn report_digest(report: &LabelConsumptionEligibilityReportV1) -> Result<String, CaOcrcErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn survival(policy: &CaOcrcPolicyV1, delay: u64) -> f64 {
    let index = (delay as usize).saturating_add(1);
    policy.proxy_delay_probabilities.iter().skip(index).sum()
}

fn tracking_load(policy: &CaOcrcPolicyV1) -> f64 {
    (1..=policy.maximum_feedback_delay)
        .map(|delay| survival(policy, delay))
        .sum()
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
        || !policy.score_bound_b.is_finite()
        || policy.score_bound_b <= 0.0
        || !valid_sha256(&policy.score_contract_sha256)
        || !policy.initial_threshold.is_finite()
        || !(0.0..=policy.score_bound_b).contains(&policy.initial_threshold)
        || policy.tracking_capacity == 0
        || policy.maximum_feedback_delay == 0
        || !policy.maximum_inverse_probability_weight.is_finite()
        || policy.maximum_inverse_probability_weight < 1.0
    {
        return Err(CaOcrcErrorV1("CA-OCRC policy has invalid numeric parameters".into()));
    }
    let expected = usize::try_from(policy.maximum_feedback_delay + 2)
        .map_err(|_| CaOcrcErrorV1("proxy delay support exceeds platform capacity".into()))?;
    if policy.proxy_delay_probabilities.len() != expected
        || policy.proxy_delay_probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
    {
        return Err(CaOcrcErrorV1("proxy delay PMF has invalid support or probabilities".into()));
    }
    let total: f64 = policy.proxy_delay_probabilities.iter().sum();
    if (total - 1.0).abs() > 1e-10 {
        return Err(CaOcrcErrorV1("proxy delay PMF must sum to one".into()));
    }
    let ell = tracking_load(policy);
    if ell > policy.tracking_capacity as f64 + 1e-10 {
        return Err(CaOcrcErrorV1(format!(
            "CA-OCRC underload condition failed: ell_F={ell} exceeds capacity {}",
            policy.tracking_capacity
        )));
    }
    if survival(policy, policy.maximum_feedback_delay) <= 0.0 {
        return Err(CaOcrcErrorV1("proxy survival at maximum feedback delay must be positive".into()));
    }
    Ok(())
}

/// Computes the paper's proxy survival F(d)=P(proxy_delay>=d).
pub fn ca_ocrc_proxy_survival_v1(
    policy: &CaOcrcPolicyV1,
    delay: u64,
) -> Result<f64, CaOcrcErrorV1> {
    validate_policy(policy)?;
    if delay > policy.maximum_feedback_delay {
        return Ok(0.0);
    }
    Ok(survival(policy, delay))
}

/// Creates a bounded-delay CA-OCRC engine. The delay assumption is an
/// externally declared deterministic upper bound, not an estimated latency.
pub fn create_ca_ocrc_engine_v1(
    engine_id: &str,
    clock_domain_id: &str,
    policy: CaOcrcPolicyV1,
) -> Result<CaOcrcEngineV1, CaOcrcErrorV1> {
    require_text(engine_id, "engine_id")?;
    require_text(clock_domain_id, "clock_domain_id")?;
    validate_policy(&policy)?;
    let policy_sha256 = json_sha256(&policy)?;
    Ok(CaOcrcEngineV1 {
        schema: CA_OCRC_ENGINE_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        clock_domain_id: clock_domain_id.into(),
        threshold: policy.initial_threshold,
        policy,
        policy_sha256,
        next_round: 0,
        current_round: None,
        tracked_rounds: BTreeMap::new(),
        accepted_label_revisions: std::collections::BTreeSet::new(),
        guarantee_state: CaOcrcGuaranteeStateV1::ActiveUnderBoundedDelayAndUnderload,
        suspension_reasons: Vec::new(),
        capacity_saturation_count: 0,
        randomized_nontracking_count: 0,
    })
}

fn sample_proxy_delay(policy: &CaOcrcPolicyV1, uniform: f64) -> Result<i64, CaOcrcErrorV1> {
    if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
        return Err(CaOcrcErrorV1("proxy-delay uniform draw must be in [0,1)".into()));
    }
    let mut cumulative = 0.0;
    for (index, probability) in policy.proxy_delay_probabilities.iter().enumerate() {
        cumulative += probability;
        if uniform < cumulative || index + 1 == policy.proxy_delay_probabilities.len() {
            return Ok(index as i64 - 1);
        }
    }
    Err(CaOcrcErrorV1("proxy-delay sampling failed".into()))
}

fn decision_digest(decision: &CaOcrcTrackingDecisionV1) -> Result<String, CaOcrcErrorV1> {
    let mut canonical = decision.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn feedback_digest(receipt: &CaOcrcFeedbackReceiptV1) -> Result<String, CaOcrcErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn suspend(engine: &mut CaOcrcEngineV1, reason: &str, state: CaOcrcGuaranteeStateV1) {
    engine.guarantee_state = state;
    engine.suspension_reasons.push(reason.to_owned());
}

/// Executes one CA-OCRC round: expire proxy leases, apply a known trivial-set
/// loss, sample a proxy delay for nontrivial sets, and admit subject to K.
pub fn start_ca_ocrc_round_v1(
    engine: &mut CaOcrcEngineV1,
    input: &CaOcrcRoundInputV1,
) -> Result<CaOcrcTrackingDecisionV1, CaOcrcErrorV1> {
    if engine.schema != CA_OCRC_ENGINE_SCHEMA_V1 || input.schema != CA_OCRC_ROUND_SCHEMA_V1 {
        return Err(CaOcrcErrorV1("CA-OCRC engine or round schema is unsupported".into()));
    }
    if json_sha256(&engine.policy)? != engine.policy_sha256 {
        return Err(CaOcrcErrorV1("CA-OCRC policy digest mismatch".into()));
    }
    require_text(&input.prediction_id, "prediction_id")?;
    require_text(&input.sample_id, "sample_id")?;
    require_text(&input.measurement_event_id, "measurement_event_id")?;
    require_text(&input.clock_domain_id, "clock_domain_id")?;
    if input.clock_domain_id != engine.clock_domain_id
        || input.round_index != engine.next_round
        || input.event_time_unix_seconds > input.prediction_at_unix_seconds
    {
        return Err(CaOcrcErrorV1(
            "round order, clock domain or prediction chronology mismatch".into(),
        ));
    }
    if engine.guarantee_state != CaOcrcGuaranteeStateV1::ActiveUnderBoundedDelayAndUnderload {
        return Err(CaOcrcErrorV1("CA-OCRC guarantee is suspended; explicit reset/restart required".into()));
    }
    let next_round = engine.next_round.checked_add(1)
        .ok_or_else(|| CaOcrcErrorV1("round index overflow".into()))?;
    let mut next = engine.clone();
    let threshold_used = engine.threshold;
    let mut expired = Vec::new();
    next.tracked_rounds.retain(|origin, tracked| {
        let active = origin.saturating_add(tracked.proxy_delay) >= input.round_index;
        if !active { expired.push(*origin); }
        active
    });

    let mut immediate_loss = None;
    let mut sampled_proxy_delay = None;
    let mut admitted = false;
    match input.prediction_kind {
        CaOcrcPredictionKindV1::KnownEmpty => {
            if threshold_used >= 0.0 || input.lower_bound.is_some() || input.upper_bound.is_some() {
                return Err(CaOcrcErrorV1("empty-set declaration conflicts with threshold/bounds".into()));
            }
            let loss = 1.0;
            next.threshold += next.policy.learning_rate * (loss - next.policy.target_risk);
            immediate_loss = Some(loss);
        }
        CaOcrcPredictionKindV1::KnownFull => {
            if threshold_used < next.policy.score_bound_b || input.lower_bound.is_some() || input.upper_bound.is_some() {
                return Err(CaOcrcErrorV1("full-set declaration conflicts with threshold/bounds".into()));
            }
            let loss = 0.0;
            next.threshold += next.policy.learning_rate * (loss - next.policy.target_risk);
            immediate_loss = Some(loss);
        }
        CaOcrcPredictionKindV1::Nontrivial => {
            if threshold_used < 0.0 || threshold_used >= next.policy.score_bound_b
                || input.lower_bound.is_none() || input.upper_bound.is_none()
                || input.lower_bound.is_some_and(|v| !v.is_finite())
                || input.upper_bound.is_some_and(|v| !v.is_finite())
                || input.lower_bound.zip(input.upper_bound).is_some_and(|(a,b)| a>b)
            {
                return Err(CaOcrcErrorV1("nontrivial set declaration conflicts with threshold/bounds".into()));
            }
            let proxy = sample_proxy_delay(&next.policy, input.uniform_draw)?;
            sampled_proxy_delay = Some(proxy);
            if proxy < 0 {
                next.randomized_nontracking_count = next.randomized_nontracking_count.saturating_add(1);
            } else if next.tracked_rounds.len() < next.policy.tracking_capacity {
                next.tracked_rounds.insert(input.round_index, CaOcrcTrackedRoundV1 {
                    round_index: input.round_index,
                    prediction_id: input.prediction_id.clone(),
                    sample_id: input.sample_id.clone(),
                    measurement_event_id: input.measurement_event_id.clone(),
                    proxy_delay: proxy as u64,
                    lower_bound: input.lower_bound.unwrap(),
                    upper_bound: input.upper_bound.unwrap(),
                    event_time_unix_seconds: input.event_time_unix_seconds,
                    prediction_at_unix_seconds: input.prediction_at_unix_seconds,
                    clock_domain_id: input.clock_domain_id.clone(),
                });
                admitted = true;
            } else {
                next.capacity_saturation_count = next.capacity_saturation_count.saturating_add(1);
            }
        }
    }
    if !next.threshold.is_finite() {
        return Err(CaOcrcErrorV1("threshold update became non-finite".into()));
    }
    next.current_round = Some(input.round_index);
    next.next_round = next_round;
    let mut decision = CaOcrcTrackingDecisionV1 {
        schema: CA_OCRC_ROUND_SCHEMA_V1.into(),
        round_index: input.round_index,
        prediction_id: input.prediction_id.clone(),
        prediction_kind: input.prediction_kind,
        threshold_used,
        sampled_proxy_delay,
        admitted,
        expired_rounds: expired,
        active_tracking_count: next.tracked_rounds.len(),
        tracking_capacity: next.policy.tracking_capacity,
        tracking_load_ell_f: tracking_load(&next.policy),
        guarantee_state: next.guarantee_state,
        immediate_loss,
        threshold_after_round: next.threshold,
        receipt_sha256: String::new(),
    };
    decision.receipt_sha256 = decision_digest(&decision)?;
    *engine = next;
    Ok(decision)
}

/// Accepts a delayed outcome only if its round remains in the tracking set and
/// its K1 numeric label receipt is valid. An unqualified result invalidates the
/// theorem assumptions; it is never converted into a zero/one loss.
pub fn receive_ca_ocrc_feedback_v1(
    engine: &mut CaOcrcEngineV1,
    feedback: &CaOcrcFeedbackV1,
) -> Result<CaOcrcFeedbackReceiptV1, CaOcrcErrorV1> {
    if feedback.schema != CA_OCRC_FEEDBACK_SCHEMA_V1 {
        return Err(CaOcrcErrorV1("CA-OCRC feedback schema is unsupported".into()));
    }
    if json_sha256(&engine.policy)? != engine.policy_sha256 {
        return Err(CaOcrcErrorV1("CA-OCRC policy digest mismatch".into()));
    }
    let current = engine.current_round.ok_or_else(|| CaOcrcErrorV1("no CA-OCRC round is active".into()))?;
    let before = engine.threshold;
    let guarantee_before = engine.guarantee_state;
    let revision_key = format!("{}:{}", feedback.label_id, feedback.label_revision);
    if engine.accepted_label_revisions.contains(&revision_key) {
        let mut receipt = CaOcrcFeedbackReceiptV1 {
            schema: CA_OCRC_UPDATE_RECEIPT_SCHEMA_V1.into(),
            engine_id: engine.engine_id.clone(),
            origin_round: feedback.origin_round,
            current_round: feedback.current_round,
            label_id: feedback.label_id.clone(),
            label_revision: feedback.label_revision,
            disposition: CaOcrcTrackingDispositionV1::Duplicate,
            reasons: vec![CaOcrcReasonV1::DuplicateAcceptedRevision],
            observed_loss: None,
            proxy_survival_probability: None,
            inverse_probability_weight: None,
            threshold_before: before,
            threshold_after: before,
            active_tracking_count: engine.tracked_rounds.len(),
            guarantee_state_before: engine.guarantee_state,
            guarantee_state_after: engine.guarantee_state,
            threshold_mutated: false,
            physical_source_authenticity_proven: false,
            equipment_control_authority: false,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = feedback_digest(&receipt)?;
        return Ok(receipt);
    }
    let tracked = engine.tracked_rounds.get(&feedback.origin_round).cloned();
    let mut reasons = Vec::new();

    if feedback.current_round != current
        || feedback.origin_round.checked_add(feedback.actual_delay) != Some(feedback.current_round)
    {
        reasons.push(CaOcrcReasonV1::FeedbackRoundMismatch);
    }
    if feedback.actual_delay > engine.policy.maximum_feedback_delay {
        reasons.push(CaOcrcReasonV1::FeedbackDelayExceedsBound);
    }
    require_text(&feedback.prediction_id, "feedback.prediction_id")?;
    require_text(&feedback.sample_id, "feedback.sample_id")?;
    require_text(&feedback.measurement_event_id, "feedback.measurement_event_id")?;
    let Some(tracked) = tracked else {
        let guarantee_after = if reasons.contains(&CaOcrcReasonV1::FeedbackDelayExceedsBound)
            || reasons.contains(&CaOcrcReasonV1::FeedbackRoundMismatch)
        {
            suspend(
                engine,
                "observed feedback violated the declared delay/round contract",
                CaOcrcGuaranteeStateV1::SuspendedByDelayAssumptionViolation,
            );
            engine.guarantee_state
        } else {
            engine.guarantee_state
        };
        let disposition = if guarantee_after != guarantee_before {
            CaOcrcTrackingDispositionV1::GuaranteeSuspended
        } else {
            CaOcrcTrackingDispositionV1::FeedbackNotTracked
        };
        let mut receipt = CaOcrcFeedbackReceiptV1 {
            schema: CA_OCRC_UPDATE_RECEIPT_SCHEMA_V1.into(),
            engine_id: engine.engine_id.clone(),
            origin_round: feedback.origin_round,
            current_round: feedback.current_round,
            label_id: feedback.label_id.clone(),
            label_revision: feedback.label_revision,
            disposition,
            reasons: if reasons.is_empty() { vec![CaOcrcReasonV1::FeedbackNotInTrackingSet] } else { reasons },
            observed_loss: None,
            proxy_survival_probability: None,
            inverse_probability_weight: None,
            threshold_before: before,
            threshold_after: before,
            active_tracking_count: engine.tracked_rounds.len(),
            guarantee_state_before: guarantee_before,
            guarantee_state_after: guarantee_after,
            threshold_mutated: false,
            physical_source_authenticity_proven: false,
            equipment_control_authority: false,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = feedback_digest(&receipt)?;
        return Ok(receipt);
    };

    let report = &feedback.eligibility_report;
    if feedback.actual_delay > tracked.proxy_delay {
        reasons.push(CaOcrcReasonV1::FeedbackNotInTrackingSet);
    }
    if feedback.prediction_id != tracked.prediction_id
        || feedback.sample_id != tracked.sample_id
        || feedback.measurement_event_id != tracked.measurement_event_id
    {
        reasons.push(CaOcrcReasonV1::InvalidPredictionIdentity);
    }
    if feedback.clock_domain_id != tracked.clock_domain_id || feedback.clock_domain_id != engine.clock_domain_id {
        reasons.push(CaOcrcReasonV1::ClockDomainMismatch);
    }
    if report.schema != "luminous.semi-bench.label-consumption-report.v1" {
        reasons.push(CaOcrcReasonV1::EligibilityReportDigestMismatch);
    }
    if feedback.label_available_at_unix_seconds < tracked.prediction_at_unix_seconds {
        reasons.push(CaOcrcReasonV1::LabelAvailabilityBeforePrediction);
    }
    if feedback.eligibility_report_available_at_unix_seconds < feedback.label_available_at_unix_seconds {
        reasons.push(CaOcrcReasonV1::ReportBeforeLabelAvailability);
    }
    if feedback.delivered_at_unix_seconds < feedback.label_available_at_unix_seconds {
        reasons.push(CaOcrcReasonV1::DeliveryBeforeLabelAvailability);
    }
    if feedback.delivered_at_unix_seconds < feedback.eligibility_report_available_at_unix_seconds {
        reasons.push(CaOcrcReasonV1::ReportUnavailableAtDelivery);
    }
    if report.schema != "luminous.semi-bench.label-consumption-report.v1" {
        reasons.push(CaOcrcReasonV1::EligibilityReportDigestMismatch);
    }
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        reasons.push(CaOcrcReasonV1::LabelNotCurrentK1);
    }
    if report.label_id != feedback.label_id {
        reasons.push(CaOcrcReasonV1::LabelIdentityMismatch);
    }
    if report.label_revision != feedback.label_revision {
        reasons.push(CaOcrcReasonV1::LabelRevisionMismatch);
    }
    if report.measurement_event_id != tracked.measurement_event_id {
        reasons.push(CaOcrcReasonV1::MeasurementEventMismatch);
    }
    if report.label_sha256 != feedback.label_payload_sha256 || !valid_sha256(&report.label_sha256) {
        reasons.push(CaOcrcReasonV1::LabelPayloadDigestMismatch);
    }
    if report_digest(report)? != report.report_sha256 {
        reasons.push(CaOcrcReasonV1::EligibilityReportDigestMismatch);
    }
    if !feedback.label_value.is_finite()
        || !report.qualified_corrected_value.is_finite()
        || report.qualified_standard_uncertainty_upper_bound
            .is_none_or(|value| !value.is_finite() || value < 0.0)
        || feedback.label_revision == 0
    {
        reasons.push(CaOcrcReasonV1::ReportNumericValueInvalid);
    }
    if report.qualified_corrected_value != feedback.label_value {
        reasons.push(CaOcrcReasonV1::DeliveredValueMismatch);
    }
    if engine.policy.require_source_authenticity_assertion && !report.source_authenticity_proven {
        reasons.push(CaOcrcReasonV1::SourceAuthenticityAssertionMissing);
    }
    if report.training_snapshot_mutated || report.equipment_control_authority {
        reasons.push(CaOcrcReasonV1::UpstreamAuthorityFlagSet);
    }
    match report.eligibility {
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy if report.downstream_consumption_allowed => {}
        LabelConsumptionEligibilityV1::Unknown | LabelConsumptionEligibilityV1::Blocked => {
            reasons.push(CaOcrcReasonV1::LabelNotEligible);
        }
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => {
            reasons.push(CaOcrcReasonV1::DownstreamConsumptionNotAllowed);
        }
    }
    reasons.sort();
    reasons.dedup();

    if !reasons.is_empty() {
        let disposition = match report.eligibility {
            LabelConsumptionEligibilityV1::Unknown => CaOcrcTrackingDispositionV1::HeldUnknown,
            LabelConsumptionEligibilityV1::Blocked => CaOcrcTrackingDispositionV1::Blocked,
            _ => CaOcrcTrackingDispositionV1::Rejected,
        };
        suspend(engine, "K1 feedback was not eligible for loss observation", CaOcrcGuaranteeStateV1::SuspendedByUnqualifiedFeedback);
        engine.tracked_rounds.remove(&feedback.origin_round);
        let mut receipt = CaOcrcFeedbackReceiptV1 {
            schema: CA_OCRC_UPDATE_RECEIPT_SCHEMA_V1.into(),
            engine_id: engine.engine_id.clone(),
            origin_round: feedback.origin_round,
            current_round: feedback.current_round,
            label_id: feedback.label_id.clone(),
            label_revision: feedback.label_revision,
            disposition,
            reasons,
            observed_loss: None,
            proxy_survival_probability: None,
            inverse_probability_weight: None,
            threshold_before: before,
            threshold_after: before,
            active_tracking_count: engine.tracked_rounds.len(),
            guarantee_state_before: guarantee_before,
            guarantee_state_after: engine.guarantee_state,
            threshold_mutated: false,
            physical_source_authenticity_proven: false,
            equipment_control_authority: false,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = feedback_digest(&receipt)?;
        return Ok(receipt);
    }

    if engine.guarantee_state != CaOcrcGuaranteeStateV1::ActiveUnderBoundedDelayAndUnderload {
        let mut receipt = CaOcrcFeedbackReceiptV1 {
            schema: CA_OCRC_UPDATE_RECEIPT_SCHEMA_V1.into(),
            engine_id: engine.engine_id.clone(),
            origin_round: feedback.origin_round,
            current_round: feedback.current_round,
            label_id: feedback.label_id.clone(),
            label_revision: feedback.label_revision,
            disposition: CaOcrcTrackingDispositionV1::GuaranteeSuspended,
            reasons: Vec::new(),
            observed_loss: None,
            proxy_survival_probability: None,
            inverse_probability_weight: None,
            threshold_before: before,
            threshold_after: before,
            active_tracking_count: engine.tracked_rounds.len(),
            guarantee_state_before: guarantee_before,
            guarantee_state_after: engine.guarantee_state,
            threshold_mutated: false,
            physical_source_authenticity_proven: false,
            equipment_control_authority: false,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = feedback_digest(&receipt)?;
        return Ok(receipt);
    }

    let probability = survival(&engine.policy, feedback.actual_delay);
    if probability <= 0.0 || !probability.is_finite() {
        suspend(engine, "feedback inverse propensity is undefined", CaOcrcGuaranteeStateV1::SuspendedByDelayAssumptionViolation);
        return Err(CaOcrcErrorV1("feedback proxy survival probability is zero".into()));
    }
    let inverse_weight = 1.0 / probability;
    if !inverse_weight.is_finite() || inverse_weight > engine.policy.maximum_inverse_probability_weight {
        suspend(engine, "inverse propensity exceeds configured numeric ceiling", CaOcrcGuaranteeStateV1::SuspendedByNumericOrProtocolViolation);
        return Err(CaOcrcErrorV1("inverse propensity exceeds configured ceiling; no clipping applied".into()));
    }

    let loss = if feedback.label_value < tracked.lower_bound || feedback.label_value > tracked.upper_bound {
        1.0
    } else {
        0.0
    };
    let after = before + engine.policy.learning_rate * inverse_weight * (loss - engine.policy.target_risk);
    if !after.is_finite() {
        suspend(engine, "threshold update became non-finite", CaOcrcGuaranteeStateV1::SuspendedByNumericOrProtocolViolation);
        return Err(CaOcrcErrorV1("CA-OCRC threshold update is non-finite".into()));
    }

    let mut receipt = CaOcrcFeedbackReceiptV1 {
        schema: CA_OCRC_UPDATE_RECEIPT_SCHEMA_V1.into(),
        engine_id: engine.engine_id.clone(),
        origin_round: feedback.origin_round,
        current_round: feedback.current_round,
        label_id: feedback.label_id.clone(),
        label_revision: feedback.label_revision,
        disposition: CaOcrcTrackingDispositionV1::FeedbackUpdated,
        reasons: Vec::new(),
        observed_loss: Some(loss),
        proxy_survival_probability: Some(probability),
        inverse_probability_weight: Some(inverse_weight),
        threshold_before: before,
        threshold_after: after,
        active_tracking_count: engine.tracked_rounds.len().saturating_sub(1),
        guarantee_state_before: guarantee_before,
        guarantee_state_after: guarantee_before,
        threshold_mutated: true,
        physical_source_authenticity_proven: false,
        equipment_control_authority: false,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = feedback_digest(&receipt)?;
    engine.threshold = after;
    engine.accepted_label_revisions.insert(revision_key);
    engine.tracked_rounds.remove(&feedback.origin_round);
    Ok(receipt)
}

fn h_function(x: f64) -> f64 {
    x - 1.0 - x.ln()
}

/// Computes the paper's arbitrary-bounded-delay risk-gap upper bound under its
/// score-boundedness, empty/full loss, underload and fixed-delay assumptions.
pub fn ca_ocrc_risk_gap_bound_v1(
    policy: &CaOcrcPolicyV1,
    horizon: u64,
) -> Result<CaOcrcRiskGapBoundV1, CaOcrcErrorV1> {
    validate_policy(policy)?;
    if horizon == 0 {
        return Err(CaOcrcErrorV1("horizon must be positive".into()));
    }
    let ell = tracking_load(policy);
    let f_max = survival(policy, policy.maximum_feedback_delay);
    let boundary = policy.score_bound_b / (policy.learning_rate * horizon as f64);
    let feedback = (policy.tracking_capacity as f64 + 1.0) / (f_max * horizon as f64);
    let x = ell / policy.tracking_capacity as f64;
    let capacity = if x > 0.0 { (-policy.tracking_capacity as f64 * h_function(x)).exp() } else { 0.0 };
    Ok(CaOcrcRiskGapBoundV1 {
        horizon,
        boundary_term: boundary,
        feedback_term: feedback,
        capacity_saturation_term: capacity,
        total_upper_bound: boundary + feedback + capacity,
        tracking_load_ell_f: ell,
        proxy_survival_at_max_delay: f_max,
    })
}

/// Solves the one-dimensional worst-case proxy-delay design from Proposition 1
/// by deterministic bounded grid refinement. The resulting PMF places mass
/// 1-p at proxy delay -1 and mass p at d_max.
pub fn optimize_ca_ocrc_proxy_policy_v1(
    capacity: usize,
    maximum_delay: u64,
    horizon: u64,
) -> Result<CaOcrcOptimizedProxyPolicyV1, CaOcrcErrorV1> {
    if capacity == 0 || maximum_delay == 0 || horizon == 0 {
        return Err(CaOcrcErrorV1("capacity, maximum delay and horizon must be positive".into()));
    }
    let upper = (capacity as f64 / maximum_delay as f64).min(1.0);
    let objective = |p: f64| {
        let x = maximum_delay as f64 * p / capacity as f64;
        let saturation = if x > 0.0 { (-capacity as f64 * h_function(x)).exp() } else { 0.0 };
        (capacity as f64 + 1.0) / (p * horizon as f64) + saturation
    };
    let mut best_p = upper;
    let mut best = objective(upper);
    for index in 1..=20_000 {
        let p = upper * index as f64 / 20_000.0;
        let value = objective(p);
        if value < best {
            best = value;
            best_p = p;
        }
    }
    let mut probabilities = vec![0.0; usize::try_from(maximum_delay + 2)
        .map_err(|_| CaOcrcErrorV1("proxy support exceeds platform capacity".into()))?];
    probabilities[0] = 1.0 - best_p;
    probabilities[maximum_delay as usize + 1] = best_p;
    Ok(CaOcrcOptimizedProxyPolicyV1 {
        maximum_delay,
        capacity,
        horizon,
        optimal_tracking_probability: best_p,
        proxy_delay_probabilities: probabilities,
        theoretical_objective: best,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String { std::iter::repeat_n(ch, 64).collect() }

    fn policy(capacity: usize, delay: u64, p_track_max: f64) -> CaOcrcPolicyV1 {
        let mut pmf = vec![0.0; delay as usize + 2];
        pmf[0] = 1.0 - p_track_max;
        pmf[delay as usize + 1] = p_track_max;
        CaOcrcPolicyV1 {
            schema: CA_OCRC_POLICY_SCHEMA_V1.into(),
            policy_id: "ca-ocrc-bounded-delay.v1".into(),
            revision: 1,
            target_risk: 0.05,
            learning_rate: 0.10,
            score_bound_b: 1.0,
            score_contract_sha256: digest('f'),
            initial_threshold: 0.5,
            tracking_capacity: capacity,
            maximum_feedback_delay: delay,
            proxy_delay_probabilities: pmf,
            require_source_authenticity_assertion: false,
            maximum_inverse_probability_weight: 1000.0,
        }
    }

    fn report(label: &str, rev: u64, event: &str, value: f64, payload: &str) -> LabelConsumptionEligibilityReportV1 {
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label.into(), label_revision: rev, measurement_event_id: event.into(),
            calibration_certificate_id: "cal-1".into(), frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            downstream_consumption_allowed: true, reasons: Vec::new(),
            label_sha256: payload.into(), measurement_sha256: digest('a'),
            qualified_corrected_value: value, qualified_standard_uncertainty_upper_bound: Some(0.01),
            qualification_report_sha256: digest('b'), requalification_report_sha256: digest('c'),
            requalification_frontier_sha256: digest('d'), source_authenticity_proven: false,
            training_snapshot_mutated: false, equipment_control_authority: false, report_sha256: String::new(),
        };
        let mut canonical = report.clone(); canonical.report_sha256.clear();
        report.report_sha256 = json_sha256(&canonical).unwrap();
        report
    }

    fn round(index: u64, draw: f64) -> CaOcrcRoundInputV1 {
        CaOcrcRoundInputV1 {
            schema: CA_OCRC_ROUND_SCHEMA_V1.into(), round_index: index,
            prediction_id: format!("pred-{index}"), sample_id: format!("sample-{index}"),
            measurement_event_id: format!("event-{index}"),
            prediction_kind: CaOcrcPredictionKindV1::Nontrivial,
            lower_bound: Some(0.0), upper_bound: Some(1.0),
            event_time_unix_seconds: 1_800_000_000 + index as i64 * 60,
            prediction_at_unix_seconds: 1_800_000_002 + index as i64 * 60,
            clock_domain_id: "fab-clock".into(), uniform_draw: draw,
        }
    }

    fn feedback(origin: u64, current: u64, value: f64) -> CaOcrcFeedbackV1 {
        let payload = digest('e');
        CaOcrcFeedbackV1 {
            schema: CA_OCRC_FEEDBACK_SCHEMA_V1.into(), current_round: current,
            origin_round: origin, actual_delay: current - origin,
            prediction_id: format!("pred-{origin}"),
            sample_id: format!("sample-{origin}"),
            measurement_event_id: format!("event-{origin}"),
            label_id: format!("label-{origin}"), label_revision: 1,
            label_payload_sha256: payload.clone(), label_value: value,
            label_available_at_unix_seconds: 1_800_000_120 + origin as i64 * 60,
            eligibility_report_available_at_unix_seconds: 1_800_000_121 + origin as i64 * 60,
            delivered_at_unix_seconds: 1_800_000_122 + current as i64 * 60,
            clock_domain_id: "fab-clock".into(),
            eligibility_report: report(&format!("label-{origin}"), 1, &format!("event-{origin}"), value, &payload),
        }
    }

    #[test]
    fn policy_rejects_proxy_tracking_load_above_capacity() {
        assert!(create_ca_ocrc_engine_v1("e", "fab-clock", policy(1, 4, 0.50)).is_err());
    }

    #[test]
    fn proxy_sampling_is_replayable_and_capacity_bounded() {
        let mut engine = create_ca_ocrc_engine_v1("e", "fab-clock", policy(2, 4, 0.50)).unwrap();
        let first = start_ca_ocrc_round_v1(&mut engine, &round(0, 0.1)).unwrap();
        assert_eq!(first.sampled_proxy_delay, Some(-1));
        assert!(!first.admitted);
        let second = start_ca_ocrc_round_v1(&mut engine, &round(1, 0.9)).unwrap();
        assert_eq!(second.sampled_proxy_delay, Some(4));
        assert!(second.admitted);
        assert_eq!(second.active_tracking_count, 1);
    }

    #[test]
    fn known_empty_and_full_sets_apply_immediate_known_loss_without_tracking() {
        let mut engine = create_ca_ocrc_engine_v1("e", "fab-clock", policy(2, 4, 0.50)).unwrap();
        engine.threshold = -0.01;
        let mut empty = round(0, 0.0);
        empty.prediction_kind = CaOcrcPredictionKindV1::KnownEmpty;
        empty.lower_bound = None; empty.upper_bound = None;
        let decision = start_ca_ocrc_round_v1(&mut engine, &empty).unwrap();
        assert_eq!(decision.immediate_loss, Some(1.0));
        assert!((decision.threshold_after_round - 0.085).abs() < 1e-12);

        engine.threshold = 1.01;
        let mut full = round(1, 0.0);
        full.prediction_kind = CaOcrcPredictionKindV1::KnownFull;
        full.lower_bound = None; full.upper_bound = None;
        let decision = start_ca_ocrc_round_v1(&mut engine, &full).unwrap();
        assert_eq!(decision.immediate_loss, Some(0.0));
        assert!((decision.threshold_after_round - 1.005).abs() < 1e-12);
    }

    #[test]
    fn valid_feedback_uses_inverse_proxy_survival_weight() {
        let mut engine = create_ca_ocrc_engine_v1("e", "fab-clock", policy(2, 4, 0.50)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(0, 0.9)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(1, 0.1)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(2, 0.1)).unwrap();
        let result = receive_ca_ocrc_feedback_v1(&mut engine, &feedback(0, 2, 2.0)).unwrap();
        assert_eq!(result.disposition, CaOcrcTrackingDispositionV1::FeedbackUpdated);
        assert_eq!(result.proxy_survival_probability, Some(0.50));
        assert_eq!(result.inverse_probability_weight, Some(2.0));
        assert_eq!(result.observed_loss, Some(1.0));
        assert!((result.threshold_after - (0.5 + 0.1 * 2.0 * 0.95)).abs() < 1e-12);
    }

    #[test]
    fn capacity_full_round_is_not_admitted_and_feedback_is_not_used() {
        let mut engine = create_ca_ocrc_engine_v1("e", "fab-clock", policy(1, 4, 0.25)).unwrap();
        let first = start_ca_ocrc_round_v1(&mut engine, &round(0, 0.9)).unwrap();
        assert!(first.admitted);
        let second = start_ca_ocrc_round_v1(&mut engine, &round(1, 0.9)).unwrap();
        assert!(!second.admitted);
        assert_eq!(engine.capacity_saturation_count, 1);
        let result = receive_ca_ocrc_feedback_v1(&mut engine, &feedback(1, 1, 0.5)).unwrap();
        assert_eq!(result.disposition, CaOcrcTrackingDispositionV1::FeedbackNotTracked);
        assert_eq!(engine.threshold, 0.5);
    }

    #[test]
    fn unqualified_feedback_suspends_guarantee_and_does_not_update_threshold() {
        let mut engine = create_ca_ocrc_engine_v1("e", "fab-clock", policy(2, 4, 0.50)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(0, 0.9)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(1, 0.1)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(2, 0.1)).unwrap();
        let before = engine.threshold;
        let mut input = feedback(0, 2, 2.0);
        input.eligibility_report.eligibility = LabelConsumptionEligibilityV1::Blocked;
        input.eligibility_report.downstream_consumption_allowed = false;
        let mut canonical = input.eligibility_report.clone(); canonical.report_sha256.clear();
        input.eligibility_report.report_sha256 = json_sha256(&canonical).unwrap();
        let result = receive_ca_ocrc_feedback_v1(&mut engine, &input).unwrap();
        assert_eq!(result.disposition, CaOcrcTrackingDispositionV1::Blocked);
        assert_eq!(engine.threshold, before);
        assert_eq!(engine.guarantee_state, CaOcrcGuaranteeStateV1::SuspendedByUnqualifiedFeedback);
    }

    #[test]
    fn accepted_revision_replay_is_explicitly_duplicate() {
        let mut engine = create_ca_ocrc_engine_v1("e", "fab-clock", policy(2, 4, 0.50)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(0, 0.9)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(1, 0.1)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(2, 0.1)).unwrap();
        let input = feedback(0, 2, 0.5);
        assert_eq!(
            receive_ca_ocrc_feedback_v1(&mut engine, &input).unwrap().disposition,
            CaOcrcTrackingDispositionV1::FeedbackUpdated
        );
        let before = engine.threshold;
        let duplicate = receive_ca_ocrc_feedback_v1(&mut engine, &input).unwrap();
        assert_eq!(duplicate.disposition, CaOcrcTrackingDispositionV1::Duplicate);
        assert!(!duplicate.threshold_mutated);
        assert_eq!(engine.threshold, before);
    }

    #[test]
    fn prediction_identity_mismatch_is_rejected_without_threshold_update() {
        let mut engine = create_ca_ocrc_engine_v1("e", "fab-clock", policy(2, 4, 0.50)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(0, 0.9)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(1, 0.1)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(2, 0.1)).unwrap();
        let mut input = feedback(0, 2, 0.5);
        input.measurement_event_id = "another-event".into();
        let before = engine.threshold;
        let receipt = receive_ca_ocrc_feedback_v1(&mut engine, &input).unwrap();
        assert_eq!(receipt.disposition, CaOcrcTrackingDispositionV1::Rejected);
        assert!(!receipt.threshold_mutated);
        assert_eq!(engine.threshold, before);
        assert_eq!(
            engine.guarantee_state,
            CaOcrcGuaranteeStateV1::SuspendedByUnqualifiedFeedback
        );
    }

    #[test]
    fn observed_delay_beyond_declared_bound_suspends_even_when_outcome_was_not_tracked() {
        let mut engine = create_ca_ocrc_engine_v1("e", "fab-clock", policy(2, 4, 0.50)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(0, 0.1)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(1, 0.1)).unwrap();
        start_ca_ocrc_round_v1(&mut engine, &round(2, 0.1)).unwrap();
        let mut input = feedback(0, 6, 0.5);
        input.current_round = 6;
        input.actual_delay = 6;
        let receipt = receive_ca_ocrc_feedback_v1(&mut engine, &input).unwrap();
        assert_eq!(receipt.disposition, CaOcrcTrackingDispositionV1::GuaranteeSuspended);
        assert_eq!(
            engine.guarantee_state,
            CaOcrcGuaranteeStateV1::SuspendedByDelayAssumptionViolation
        );
    }

    #[test]
    fn theorem_bound_is_finite_under_declared_underload_policy() {
        let policy = policy(2, 4, 0.50);
        let bound = ca_ocrc_risk_gap_bound_v1(&policy, 10_000).unwrap();
        assert!(bound.total_upper_bound.is_finite());
        assert!(bound.tracking_load_ell_f <= policy.tracking_capacity as f64);
        assert!(bound.proxy_survival_at_max_delay > 0.0);
    }

    #[test]
    fn optimized_proxy_distribution_satisfies_underload_condition() {
        let optimized = optimize_ca_ocrc_proxy_policy_v1(2, 8, 10_000).unwrap();
        let policy = CaOcrcPolicyV1 {
            schema: CA_OCRC_POLICY_SCHEMA_V1.into(), policy_id: "optimized".into(), revision: 1,
            target_risk: 0.05, learning_rate: 0.1, score_bound_b: 1.0,
            score_contract_sha256: digest('f'), initial_threshold: 0.5,
            tracking_capacity: 2, maximum_feedback_delay: 8,
            proxy_delay_probabilities: optimized.proxy_delay_probabilities,
            require_source_authenticity_assertion: false, maximum_inverse_probability_weight: 10_000.0,
        };
        validate_policy(&policy).unwrap();
    }
}
