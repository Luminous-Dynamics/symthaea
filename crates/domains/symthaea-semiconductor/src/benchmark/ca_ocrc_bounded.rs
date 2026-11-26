// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Faithful CA-OCRC tracking for arbitrary, known-bounded feedback delays.
//!
//! This v1 implements q_G = 1 (actual delay <= declared d_max), an injected
//! proxy-delay uniform draw, exact eviction/admission/reception order and
//! inverse-survival weighting. Physical measurement selection is represented
//! separately and can invalidate theorem applicability; tracking is not
//! measurement acquisition. No equipment or recipe authority is provided.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
    LabelEvidenceFrontierV1,
};

pub const CA_OCRC_POLICY_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-policy.v1";
pub const CA_OCRC_ENGINE_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-engine.v1";
pub const CA_OCRC_ROUND_INPUT_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-round-input.v1";
pub const CA_OCRC_PREDICTION_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-prediction.v1";
pub const CA_OCRC_PREDICTION_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.ca-ocrc-prediction-receipt.v1";
pub const CA_OCRC_FEEDBACK_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-feedback.v1";
pub const CA_OCRC_ROUND_REPORT_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-round-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcSetKindV1 {
    Empty,
    Full,
    Interval,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcPredictionSetV1 {
    pub kind: CaOcrcSetKindV1,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub measurand_id: String,
    pub unit: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcFeedbackDispositionV1 {
    Updated,
    HeldUnknown,
    Blocked,
    Rejected,
    NotTracked,
    Duplicate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcReasonV1 {
    LabelNotCurrentK1,
    LabelNotEligible,
    DownstreamConsumptionNotAllowed,
    LabelIdentityMismatch,
    LabelRevisionMismatch,
    MeasurementEventMismatch,
    PayloadDigestMismatch,
    EligibilityReportDigestMismatch,
    CorrectedValueMismatch,
    MeasurandMismatch,
    UnitMismatch,
    ClockDomainMismatch,
    LabelNotYetAvailable,
    ReportNotYetAvailable,
    LabelAvailabilityBeforeMeasurement,
    ActualDelayOutOfBounds,
    FeedbackRoundMismatch,
    PredictionNotFound,
    PredictionNotTracked,
    ProxyWindowExpired,
    DuplicateAcceptedLabelRevision,
    DuplicateFeedbackAttempt,
    NonFiniteOutcome,
    InvalidLoss,
    FeedbackForUnselectedMeasurement,
    ReportAuthorityViolation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub target_risk_alpha: f64,
    pub learning_rate_eta: f64,
    pub bounded_score_upper: f64,
    pub maximum_actual_delay_steps: u64,
    pub tracking_capacity: usize,
    /// PMF indexed by proxy delay + 1: index 0 means -1, index d+1 means d.
    pub proxy_delay_pmf: Vec<f64>,
    /// External declaration/receipt supporting the arbitrary bounded-delay assumption.
    pub actual_delay_contract_sha256: String,
    pub bounded_actual_delay_assumption_attested: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcPredictionInputV1 {
    pub schema: String,
    pub round: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub round_time_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub interval_method_sha256: String,
    pub threshold_used: f64,
    pub prediction_set: CaOcrcPredictionSetV1,
    /// False means no physical metrology outcome is expected for this wafer.
    /// That is a separate selection mechanism and invalidates the paper's
    /// complete bounded-feedback assumption for population-risk claims.
    pub physical_measurement_selected: bool,
    /// Must be None for trivial full/empty sets and Some([0,1)) otherwise.
    pub proxy_uniform_draw: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcPredictionReceiptV1 {
    pub schema: String,
    pub round: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub round_time_unix_seconds: i64,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub interval_method_sha256: String,
    pub threshold_used: f64,
    pub prediction_set: CaOcrcPredictionSetV1,
    pub physical_measurement_selected: bool,
    pub trivial_loss: Option<f64>,
    pub proxy_uniform_draw: Option<f64>,
    pub sampled_proxy_delay: i64,
    pub admitted_to_tracking: bool,
    pub occupancy_after_admission: usize,
    pub policy_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcLabelFeedbackV1 {
    pub schema: String,
    pub feedback_round: u64,
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub measurement_event_id: String,
    pub measurand_id: String,
    pub unit: String,
    pub label_payload_sha256: String,
    pub label_value: f64,
    pub label_available_at_unix_seconds: i64,
    pub eligibility_report_available_at_unix_seconds: i64,
    pub delivered_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcFeedbackResultV1 {
    pub sample_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub originating_round: Option<u64>,
    pub actual_delay_steps: Option<u64>,
    pub proxy_delay: Option<i64>,
    pub disposition: CaOcrcFeedbackDispositionV1,
    pub reasons: Vec<CaOcrcReasonV1>,
    pub bounded_loss: Option<f64>,
    pub proxy_survival_probability: Option<f64>,
    pub weighted_update_contribution: Option<f64>,
    pub feedback_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcTrackedEntryV1 {
    pub prediction_receipt: CaOcrcPredictionReceiptV1,
    pub proxy_expiry_round: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcEngineStateV1 {
    pub threshold: f64,
    pub next_round: u64,
    pub tracked: BTreeMap<u64, CaOcrcTrackedEntryV1>,
    pub accepted_label_revisions: BTreeSet<String>,
    pub seen_feedback_attempt_digests: BTreeSet<String>,
    pub theorem_assumptions_intact: bool,
    pub guarantee_invalid_reasons: BTreeSet<String>,
    pub max_observed_occupancy: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub clock_domain_id: String,
    pub global_model_artifact_sha256: String,
    pub interval_method_sha256: String,
    pub policy: CaOcrcPolicyV1,
    pub policy_sha256: String,
    pub state: CaOcrcEngineStateV1,
    /// Prediction receipts are immutable after insertion.
    pub predictions_by_round: BTreeMap<u64, CaOcrcPredictionReceiptV1>,
    pub sample_to_round: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcRoundInputV1 {
    pub schema: String,
    pub prediction: CaOcrcPredictionInputV1,
    /// Every feedback item must have feedback_round equal to the current round.
    pub feedbacks: Vec<CaOcrcLabelFeedbackV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcRoundReportV1 {
    pub schema: String,
    pub engine_id: String,
    pub round: u64,
    pub prediction_receipt: CaOcrcPredictionReceiptV1,
    pub evicted_rounds: Vec<u64>,
    pub feedback_results: Vec<CaOcrcFeedbackResultV1>,
    pub immediate_trivial_loss: Option<f64>,
    pub weighted_feedback_sum: f64,
    pub threshold_before: f64,
    pub threshold_after: f64,
    pub tracking_occupancy_after: usize,
    pub tracking_capacity: usize,
    pub tracking_load_ell_f: f64,
    pub proxy_survival_at_max_delay: f64,
    pub theorem_assumptions_intact: bool,
    /// Evaluates the arbitrary bounded-delay paper expression only when the
    /// declared assumptions and underload condition remain intact.
    pub paper_risk_gap_bound: Option<f64>,
    pub input_sha256: String,
    pub state_before_sha256: String,
    pub state_after_sha256: String,
    pub report_sha256: String,
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

fn state_digest(state: &CaOcrcEngineStateV1) -> Result<String, CaOcrcErrorV1> {
    json_sha256(state)
}

fn label_report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, CaOcrcErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn prediction_receipt_digest(
    receipt: &CaOcrcPredictionReceiptV1,
) -> Result<String, CaOcrcErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn round_input_digest(input: &CaOcrcRoundInputV1) -> Result<String, CaOcrcErrorV1> {
    let mut canonical = input.clone();
    canonical.feedbacks.sort_by(|a, b| {
        (a.sample_id.as_str(), a.label_id.as_str(), a.label_revision)
            .cmp(&(b.sample_id.as_str(), b.label_id.as_str(), b.label_revision))
    });
    json_sha256(&canonical)
}

fn round_report_digest(report: &CaOcrcRoundReportV1) -> Result<String, CaOcrcErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn feedback_result_digest(result: &CaOcrcFeedbackResultV1) -> Result<String, CaOcrcErrorV1> {
    let mut canonical = result.clone();
    canonical.feedback_receipt_sha256.clear();
    json_sha256(&canonical)
}

fn validate_pmf(policy: &CaOcrcPolicyV1) -> Result<(), CaOcrcErrorV1> {
    let expected_len = usize::try_from(policy.maximum_actual_delay_steps)
        .map_err(|_| CaOcrcErrorV1("maximum delay exceeds addressable memory".into()))?
        .checked_add(2)
        .ok_or_else(|| CaOcrcErrorV1("proxy PMF length overflow".into()))?;
    if policy.maximum_actual_delay_steps >= i64::MAX as u64 {
        return Err(CaOcrcErrorV1("maximum delay exceeds signed proxy-delay range".into()));
    }
    if policy.proxy_delay_pmf.len() != expected_len {
        return Err(CaOcrcErrorV1(
            "proxy-delay PMF must cover -1 through d_max inclusive".into(),
        ));
    }
    if policy
        .proxy_delay_pmf
        .iter()
        .any(|probability| !probability.is_finite() || *probability < 0.0 || *probability > 1.0)
    {
        return Err(CaOcrcErrorV1(
            "proxy-delay PMF contains an invalid probability".into(),
        ));
    }
    let sum: f64 = policy.proxy_delay_pmf.iter().sum();
    if (sum - 1.0).abs() > 1.0e-12 {
        return Err(CaOcrcErrorV1(
            "proxy-delay PMF must sum to one within 1e-12".into(),
        ));
    }
    let survival = proxy_survival_vector(policy)?;
    let tracking_load: f64 = survival.iter().skip(1).sum();
    if tracking_load > policy.tracking_capacity as f64 + 1.0e-12 {
        return Err(CaOcrcErrorV1(
            "proxy-delay distribution violates the CA-OCRC underload condition ell_F <= K".into(),
        ));
    }
    if survival.iter().any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0) {
        return Err(CaOcrcErrorV1("proxy survival probability is invalid".into()));
    }
    if survival[policy.maximum_actual_delay_steps as usize] <= 0.0 {
        return Err(CaOcrcErrorV1(
            "F(d_max) must be positive for the declared risk bound".into(),
        ));
    }
    Ok(())
}

/// F(a) = P(proxy_delay >= a), for a=0..d_max.
pub fn proxy_survival_vector(policy: &CaOcrcPolicyV1) -> Result<Vec<f64>, CaOcrcErrorV1> {
    let d_max = usize::try_from(policy.maximum_actual_delay_steps)
        .map_err(|_| CaOcrcErrorV1("maximum delay exceeds addressable memory".into()))?;
    if policy.proxy_delay_pmf.len() != d_max + 2 {
        return Err(CaOcrcErrorV1("proxy-delay PMF length is invalid".into()));
    }
    let mut survival = vec![0.0; d_max + 1];
    for delay in 0..=d_max {
        survival[delay] = policy.proxy_delay_pmf[delay + 1..].iter().sum();
    }
    Ok(survival)
}

pub fn proxy_tracking_load(policy: &CaOcrcPolicyV1) -> Result<f64, CaOcrcErrorV1> {
    let survival = proxy_survival_vector(policy)?;
    Ok(survival.iter().skip(1).sum())
}

fn sample_proxy_delay(policy: &CaOcrcPolicyV1, draw: f64) -> Result<i64, CaOcrcErrorV1> {
    if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
        return Err(CaOcrcErrorV1("proxy uniform draw must be finite in [0,1)".into()));
    }
    let mut cumulative = 0.0;
    for (index, probability) in policy.proxy_delay_pmf.iter().enumerate() {
        cumulative += probability;
        if draw < cumulative {
            return Ok(index as i64 - 1);
        }
    }
    // Floating point summation may leave a sub-ulp gap after the validated sum.
    Ok(policy.proxy_delay_pmf.len() as i64 - 2)
}

fn prediction_loss(
    set: &CaOcrcPredictionSetV1,
    label_value: f64,
) -> Result<f64, CaOcrcErrorV1> {
    if !label_value.is_finite() {
        return Err(CaOcrcErrorV1("label value must be finite".into()));
    }
    match set.kind {
        CaOcrcSetKindV1::Empty => Ok(1.0),
        CaOcrcSetKindV1::Full => Ok(0.0),
        CaOcrcSetKindV1::Interval => {
            let lower = set
                .lower
                .ok_or_else(|| CaOcrcErrorV1("interval lower bound is missing".into()))?;
            let upper = set
                .upper
                .ok_or_else(|| CaOcrcErrorV1("interval upper bound is missing".into()))?;
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(CaOcrcErrorV1("prediction interval is invalid".into()));
            }
            Ok(if (lower..=upper).contains(&label_value) { 0.0 } else { 1.0 })
        }
    }
}

fn validate_prediction_set(set: &CaOcrcPredictionSetV1) -> Result<(), CaOcrcErrorV1> {
    require_text(&set.measurand_id, "prediction_set.measurand_id")?;
    require_text(&set.unit, "prediction_set.unit")?;
    match set.kind {
        CaOcrcSetKindV1::Empty | CaOcrcSetKindV1::Full => {
            if set.lower.is_some() || set.upper.is_some() {
                return Err(CaOcrcErrorV1(
                    "trivial prediction set must not supply numeric interval bounds".into(),
                ));
            }
        }
        CaOcrcSetKindV1::Interval => {
            let lower = set
                .lower
                .ok_or_else(|| CaOcrcErrorV1("interval lower bound is missing".into()))?;
            let upper = set
                .upper
                .ok_or_else(|| CaOcrcErrorV1("interval upper bound is missing".into()))?;
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(CaOcrcErrorV1("prediction interval is invalid".into()));
            }
        }
    }
    Ok(())
}

fn validate_policy(policy: &CaOcrcPolicyV1) -> Result<(), CaOcrcErrorV1> {
    if policy.schema != CA_OCRC_POLICY_SCHEMA_V1 {
        return Err(CaOcrcErrorV1("CA-OCRC policy schema is unsupported".into()));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if policy.revision == 0
        || !valid_sha256(&policy.actual_delay_contract_sha256)
        || !policy.target_risk_alpha.is_finite()
        || !(0.0..1.0).contains(&policy.target_risk_alpha)
        || !policy.learning_rate_eta.is_finite()
        || policy.learning_rate_eta <= 0.0
        || !policy.bounded_score_upper.is_finite()
        || policy.bounded_score_upper <= 0.0
        || policy.maximum_actual_delay_steps == 0
        || policy.tracking_capacity == 0
    {
        return Err(CaOcrcErrorV1("CA-OCRC policy numeric parameter is invalid".into()));
    }
    validate_pmf(policy)
}

fn h_function(value: f64) -> f64 {
    if value <= 0.0 {
        f64::INFINITY
    } else {
        value - 1.0 - value.ln()
    }
}

fn calculate_paper_bound(
    policy: &CaOcrcPolicyV1,
    survival: &[f64],
    tracking_load_ell: f64,
    horizon: u64,
) -> Option<f64> {
    if horizon == 0 || tracking_load_ell > policy.tracking_capacity as f64 {
        return None;
    }
    let f_max = survival[policy.maximum_actual_delay_steps as usize];
    if f_max <= 0.0 {
        return None;
    }
    let ratio = tracking_load_ell / policy.tracking_capacity as f64;
    let exponential = if ratio == 0.0 {
        0.0
    } else {
        (-((policy.tracking_capacity as f64) * h_function(ratio))).exp()
    };
    Some(
        policy.bounded_score_upper / (policy.learning_rate_eta * horizon as f64)
            + (policy.tracking_capacity as f64 + 1.0) / (f_max * horizon as f64)
            + exponential,
    )
}

fn validate_engine(engine: &CaOcrcEngineV1) -> Result<(), CaOcrcErrorV1> {
    if engine.schema != CA_OCRC_ENGINE_SCHEMA_V1 {
        return Err(CaOcrcErrorV1("CA-OCRC engine schema is unsupported".into()));
    }
    require_text(&engine.engine_id, "engine.engine_id")?;
    require_text(&engine.clock_domain_id, "engine.clock_domain_id")?;
    if !valid_sha256(&engine.global_model_artifact_sha256)
        || !valid_sha256(&engine.interval_method_sha256)
        || !valid_sha256(&engine.policy_sha256)
    {
        return Err(CaOcrcErrorV1("CA-OCRC engine digest is malformed".into()));
    }
    validate_policy(&engine.policy)?;
    if policy_digest(&engine.policy)? != engine.policy_sha256 {
        return Err(CaOcrcErrorV1("CA-OCRC policy digest mismatch".into()));
    }
    if !engine.state.threshold.is_finite() {
        return Err(CaOcrcErrorV1("CA-OCRC threshold is non-finite".into()));
    }
    if engine.state.tracked.len() > engine.policy.tracking_capacity {
        return Err(CaOcrcErrorV1("CA-OCRC tracking set exceeds capacity".into()));
    }
    for (round, tracked) in &engine.state.tracked {
        if *round != tracked.prediction_receipt.round
            || tracked.proxy_expiry_round
                != round
                    .checked_add(tracked.prediction_receipt.sampled_proxy_delay as u64)
                    .ok_or_else(|| CaOcrcErrorV1("proxy expiry round overflow".into()))?
        {
            return Err(CaOcrcErrorV1("CA-OCRC tracked entry is inconsistent".into()));
        }
    }
    Ok(())
}

/// Seals a caller policy using the canonical policy JSON representation.
pub fn seal_ca_ocrc_policy_v1(policy: &CaOcrcPolicyV1) -> Result<String, CaOcrcErrorV1> {
    validate_policy(policy)?;
    policy_digest(policy)
}

pub fn create_ca_ocrc_engine_v1(
    engine_id: &str,
    clock_domain_id: &str,
    global_model_artifact_sha256: &str,
    interval_method_sha256: &str,
    initial_threshold: f64,
    policy: CaOcrcPolicyV1,
) -> Result<CaOcrcEngineV1, CaOcrcErrorV1> {
    require_text(engine_id, "engine_id")?;
    require_text(clock_domain_id, "clock_domain_id")?;
    if !valid_sha256(global_model_artifact_sha256)
        || !valid_sha256(interval_method_sha256)
        || !initial_threshold.is_finite()
        || initial_threshold < 0.0
        || initial_threshold > policy.bounded_score_upper
    {
        return Err(CaOcrcErrorV1("engine identity or initial threshold is invalid".into()));
    }
    validate_policy(&policy)?;
    let policy_sha256 = policy_digest(&policy)?;
    let delay_assumption_attested = policy.bounded_actual_delay_assumption_attested;
    let mut guarantee_invalid_reasons = BTreeSet::new();
    if !delay_assumption_attested {
        guarantee_invalid_reasons
            .insert("bounded_actual_delay_assumption_not_attested".to_string());
    }
    Ok(CaOcrcEngineV1 {
        schema: CA_OCRC_ENGINE_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        clock_domain_id: clock_domain_id.into(),
        global_model_artifact_sha256: global_model_artifact_sha256.into(),
        interval_method_sha256: interval_method_sha256.into(),
        policy,
        policy_sha256,
        state: CaOcrcEngineStateV1 {
            threshold: initial_threshold,
            next_round: 0,
            tracked: BTreeMap::new(),
            accepted_label_revisions: BTreeSet::new(),
            seen_feedback_attempt_digests: BTreeSet::new(),
            theorem_assumptions_intact: delay_assumption_attested,
            guarantee_invalid_reasons,
            max_observed_occupancy: 0,
        },
        predictions_by_round: BTreeMap::new(),
        sample_to_round: BTreeMap::new(),
    })
}

fn mark_assumption_invalid(engine: &mut CaOcrcEngineV1, reason: &str) {
    engine.state.theorem_assumptions_intact = false;
    engine.state.guarantee_invalid_reasons.insert(reason.into());
}

fn make_feedback_result(
    feedback: &CaOcrcLabelFeedbackV1,
    originating_round: Option<u64>,
    actual_delay_steps: Option<u64>,
    proxy_delay: Option<i64>,
    disposition: CaOcrcFeedbackDispositionV1,
    mut reasons: Vec<CaOcrcReasonV1>,
    loss: Option<f64>,
    survival: Option<f64>,
    contribution: Option<f64>,
) -> Result<CaOcrcFeedbackResultV1, CaOcrcErrorV1> {
    reasons.sort();
    reasons.dedup();
    let mut result = CaOcrcFeedbackResultV1 {
        sample_id: feedback.sample_id.clone(),
        label_id: feedback.label_id.clone(),
        label_revision: feedback.label_revision,
        originating_round,
        actual_delay_steps,
        proxy_delay,
        disposition,
        reasons,
        bounded_loss: loss,
        proxy_survival_probability: survival,
        weighted_update_contribution: contribution,
        feedback_receipt_sha256: String::new(),
    };
    result.feedback_receipt_sha256 = feedback_result_digest(&result)?;
    Ok(result)
}

fn validate_feedback_against_prediction(
    engine: &CaOcrcEngineV1,
    feedback: &CaOcrcLabelFeedbackV1,
    prediction: &CaOcrcPredictionReceiptV1,
    current_round: u64,
    current_time_unix_seconds: i64,
) -> Result<(f64, u64, f64), Vec<CaOcrcReasonV1>> {
    let mut reasons = Vec::new();
    let actual_delay = current_round.saturating_sub(prediction.round);
    let report = &feedback.eligibility_report;

    if feedback.feedback_round != current_round {
        reasons.push(CaOcrcReasonV1::FeedbackRoundMismatch);
    }
    if actual_delay > engine.policy.maximum_actual_delay_steps {
        reasons.push(CaOcrcReasonV1::ActualDelayOutOfBounds);
    }
    if feedback.measurement_event_id != prediction.measurement_event_id {
        reasons.push(CaOcrcReasonV1::MeasurementEventMismatch);
    }
    if feedback.clock_domain_id != engine.clock_domain_id
        || feedback.clock_domain_id != prediction.clock_domain_id
    {
        reasons.push(CaOcrcReasonV1::ClockDomainMismatch);
    }
    if feedback.label_available_at_unix_seconds < prediction.event_time_unix_seconds {
        reasons.push(CaOcrcReasonV1::LabelAvailabilityBeforeMeasurement);
    }
    if feedback.delivered_at_unix_seconds < feedback.label_available_at_unix_seconds
        || feedback.delivered_at_unix_seconds > current_time_unix_seconds
    {
        reasons.push(CaOcrcReasonV1::LabelNotYetAvailable);
    }
    if feedback.eligibility_report_available_at_unix_seconds
        < feedback.label_available_at_unix_seconds
        || feedback.eligibility_report_available_at_unix_seconds
            > feedback.delivered_at_unix_seconds
    {
        reasons.push(CaOcrcReasonV1::ReportNotYetAvailable);
    }
    if report.training_snapshot_mutated || report.equipment_control_authority {
        reasons.push(CaOcrcReasonV1::ReportAuthorityViolation);
    }
    if !prediction.physical_measurement_selected {
        reasons.push(CaOcrcReasonV1::FeedbackForUnselectedMeasurement);
    }
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        reasons.push(CaOcrcReasonV1::LabelNotCurrentK1);
    }
    if report.eligibility != LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy {
        reasons.push(CaOcrcReasonV1::LabelNotEligible);
    }
    if !report.downstream_consumption_allowed {
        reasons.push(CaOcrcReasonV1::DownstreamConsumptionNotAllowed);
    }
    if report.label_id != feedback.label_id {
        reasons.push(CaOcrcReasonV1::LabelIdentityMismatch);
    }
    if report.label_revision != feedback.label_revision {
        reasons.push(CaOcrcReasonV1::LabelRevisionMismatch);
    }
    if report.measurement_event_id != feedback.measurement_event_id {
        reasons.push(CaOcrcReasonV1::MeasurementEventMismatch);
    }
    if !valid_sha256(&feedback.label_payload_sha256)
        || feedback.label_payload_sha256 != report.label_sha256
        || !valid_sha256(&report.label_sha256)
    {
        reasons.push(CaOcrcReasonV1::PayloadDigestMismatch);
    }
    if label_report_digest(report).ok().as_deref() != Some(report.report_sha256.as_str()) {
        reasons.push(CaOcrcReasonV1::EligibilityReportDigestMismatch);
    }
    if !feedback.label_value.is_finite() || !report.qualified_corrected_value.is_finite() {
        reasons.push(CaOcrcReasonV1::NonFiniteOutcome);
    } else if feedback.label_value != report.qualified_corrected_value {
        reasons.push(CaOcrcReasonV1::CorrectedValueMismatch);
    }
    if feedback.measurand_id != prediction.prediction_set.measurand_id {
        reasons.push(CaOcrcReasonV1::MeasurandMismatch);
    }
    if feedback.unit != prediction.prediction_set.unit {
        reasons.push(CaOcrcReasonV1::UnitMismatch);
    }
    if engine
        .state
        .accepted_label_revisions
        .contains(&format!("{}:{}", feedback.label_id, feedback.label_revision))
    {
        reasons.push(CaOcrcReasonV1::DuplicateAcceptedLabelRevision);
    }

    if !reasons.is_empty() {
        return Err(reasons);
    }
    let loss = prediction_loss(&prediction.prediction_set, feedback.label_value)
        .map_err(|_| vec![CaOcrcReasonV1::InvalidLoss])?;
    let survival = proxy_survival_vector(&engine.policy)
        .map_err(|_| vec![CaOcrcReasonV1::InvalidLoss])?[actual_delay as usize];
    if survival <= 0.0 || !survival.is_finite() {
        return Err(vec![CaOcrcReasonV1::InvalidLoss]);
    }
    let contribution = (loss - engine.policy.target_risk_alpha) / survival;
    if !contribution.is_finite() {
        return Err(vec![CaOcrcReasonV1::InvalidLoss]);
    }
    Ok((loss, actual_delay, contribution))
}

fn evict_expired(
    engine: &mut CaOcrcEngineV1,
    round: u64,
) -> Vec<u64> {
    let expired: Vec<u64> = engine
        .state
        .tracked
        .iter()
        .filter(|(_, entry)| entry.proxy_expiry_round < round)
        .map(|(origin_round, _)| *origin_round)
        .collect();
    for origin_round in &expired {
        engine.state.tracked.remove(origin_round);
    }
    expired
}

/// Processes one CA-OCRC round in paper order:
/// prediction -> proxy expiry/eviction -> admission -> feedback reception ->
/// threshold update. The uniform draw is injected by the caller and recorded.
pub fn step_ca_ocrc_round_v1(
    engine: &mut CaOcrcEngineV1,
    input: &CaOcrcRoundInputV1,
) -> Result<CaOcrcRoundReportV1, CaOcrcErrorV1> {
    validate_engine(engine)?;
    if input.schema != CA_OCRC_ROUND_INPUT_SCHEMA_V1 {
        return Err(CaOcrcErrorV1("CA-OCRC round input schema is unsupported".into()));
    }
    let prediction = &input.prediction;
    if prediction.schema != CA_OCRC_PREDICTION_SCHEMA_V1 {
        return Err(CaOcrcErrorV1("CA-OCRC prediction schema is unsupported".into()));
    }
    if prediction.round != engine.state.next_round {
        return Err(CaOcrcErrorV1("CA-OCRC rounds must be contiguous and monotonic".into()));
    }
    require_text(&prediction.sample_id, "prediction.sample_id")?;
    require_text(&prediction.measurement_event_id, "prediction.measurement_event_id")?;
    require_text(&prediction.clock_domain_id, "prediction.clock_domain_id")?;
    if prediction.clock_domain_id != engine.clock_domain_id
        || prediction.global_model_artifact_sha256 != engine.global_model_artifact_sha256
        || prediction.interval_method_sha256 != engine.interval_method_sha256
    {
        return Err(CaOcrcErrorV1("prediction clock/model/interval identity mismatch".into()));
    }
    if prediction.event_time_unix_seconds > prediction.prediction_at_unix_seconds
        || prediction.prediction_at_unix_seconds > prediction.round_time_unix_seconds
        || !prediction.threshold_used.is_finite()
        || prediction.threshold_used != engine.state.threshold
    {
        return Err(CaOcrcErrorV1("prediction chronology or threshold mismatch".into()));
    }
    validate_prediction_set(&prediction.prediction_set)?;
    if engine.sample_to_round.contains_key(&prediction.sample_id)
        || engine
            .predictions_by_round
            .contains_key(&prediction.round)
    {
        return Err(CaOcrcErrorV1("duplicate sample or prediction round".into()));
    }
    if prediction.prediction_set.kind == CaOcrcSetKindV1::Interval
        && prediction.proxy_uniform_draw.is_none()
    {
        return Err(CaOcrcErrorV1(
            "nontrivial interval requires an injected proxy uniform draw".into(),
        ));
    }
    if prediction.prediction_set.kind != CaOcrcSetKindV1::Interval
        && prediction.proxy_uniform_draw.is_some()
    {
        return Err(CaOcrcErrorV1(
            "trivial prediction set must not consume a proxy uniform draw".into(),
        ));
    }

    let state_before = state_digest(&engine.state)?;
    let threshold_before = engine.state.threshold;
    let mut candidate = engine.clone();
    let evicted_rounds = evict_expired(&mut candidate, prediction.round);

    let trivial_loss = match prediction.prediction_set.kind {
        CaOcrcSetKindV1::Empty => Some(1.0),
        CaOcrcSetKindV1::Full => Some(0.0),
        CaOcrcSetKindV1::Interval => None,
    };
    let sampled_proxy_delay = if trivial_loss.is_some() {
        -1
    } else {
        sample_proxy_delay(
            &candidate.policy,
            prediction.proxy_uniform_draw.expect("validated nontrivial draw"),
        )?
    };
    let admitted = sampled_proxy_delay >= 0
        && prediction.physical_measurement_selected
        && candidate.state.tracked.len() < candidate.policy.tracking_capacity;
    if !prediction.physical_measurement_selected && trivial_loss.is_none() {
        mark_assumption_invalid(&mut candidate, "physical_measurement_not_selected");
    }

    let mut receipt = CaOcrcPredictionReceiptV1 {
        schema: CA_OCRC_PREDICTION_RECEIPT_SCHEMA_V1.into(),
        round: prediction.round,
        sample_id: prediction.sample_id.clone(),
        measurement_event_id: prediction.measurement_event_id.clone(),
        event_time_unix_seconds: prediction.event_time_unix_seconds,
        prediction_at_unix_seconds: prediction.prediction_at_unix_seconds,
        round_time_unix_seconds: prediction.round_time_unix_seconds,
        clock_domain_id: prediction.clock_domain_id.clone(),
        global_model_artifact_sha256: prediction.global_model_artifact_sha256.clone(),
        interval_method_sha256: prediction.interval_method_sha256.clone(),
        threshold_used: prediction.threshold_used,
        prediction_set: prediction.prediction_set.clone(),
        physical_measurement_selected: prediction.physical_measurement_selected,
        trivial_loss,
        proxy_uniform_draw: prediction.proxy_uniform_draw,
        sampled_proxy_delay,
        admitted_to_tracking: admitted,
        occupancy_after_admission: candidate.state.tracked.len() + usize::from(admitted),
        policy_sha256: candidate.policy_sha256.clone(),
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = prediction_receipt_digest(&receipt)?;
    candidate
        .predictions_by_round
        .insert(prediction.round, receipt.clone());
    candidate
        .sample_to_round
        .insert(prediction.sample_id.clone(), prediction.round);

    if admitted {
        let expiry = prediction
            .round
            .checked_add(sampled_proxy_delay as u64)
            .ok_or_else(|| CaOcrcErrorV1("proxy expiry round overflow".into()))?;
        candidate.state.tracked.insert(
            prediction.round,
            CaOcrcTrackedEntryV1 {
                prediction_receipt: receipt.clone(),
                proxy_expiry_round: expiry,
            },
        );
    }
    candidate.state.max_observed_occupancy = candidate
        .state
        .max_observed_occupancy
        .max(candidate.state.tracked.len());

    let mut feedbacks = input.feedbacks.clone();
    feedbacks.sort_by(|a, b| {
        (a.sample_id.as_str(), a.label_id.as_str(), a.label_revision)
            .cmp(&(b.sample_id.as_str(), b.label_id.as_str(), b.label_revision))
    });
    let mut feedback_results = Vec::with_capacity(feedbacks.len());
    let mut contribution_sum = 0.0_f64;

    for feedback in &feedbacks {
        if feedback.schema != CA_OCRC_FEEDBACK_SCHEMA_V1 {
            return Err(CaOcrcErrorV1("CA-OCRC feedback schema is unsupported".into()));
        }
        require_text(&feedback.sample_id, "feedback.sample_id")?;
        require_text(&feedback.label_id, "feedback.label_id")?;
        if feedback.label_revision == 0 {
            return Err(CaOcrcErrorV1("feedback label revision must be positive".into()));
        }
        let feedback_attempt_digest = json_sha256(feedback)?;
        if !candidate
            .state
            .seen_feedback_attempt_digests
            .insert(feedback_attempt_digest)
        {
            let result = make_feedback_result(
                feedback,
                None,
                None,
                None,
                CaOcrcFeedbackDispositionV1::Duplicate,
                vec![CaOcrcReasonV1::DuplicateFeedbackAttempt],
                None,
                None,
                None,
            )?;
            feedback_results.push(result);
            continue;
        }
        let Some(origin_round) = candidate.sample_to_round.get(&feedback.sample_id).copied() else {
            let result = make_feedback_result(
                feedback,
                None,
                None,
                None,
                CaOcrcFeedbackDispositionV1::Rejected,
                vec![CaOcrcReasonV1::PredictionNotFound],
                None,
                None,
                None,
            )?;
            feedback_results.push(result);
            continue;
        };
        let Some(prediction_receipt) = candidate.predictions_by_round.get(&origin_round) else {
            let result = make_feedback_result(
                feedback,
                Some(origin_round),
                Some(prediction.round.saturating_sub(origin_round)),
                None,
                CaOcrcFeedbackDispositionV1::Rejected,
                vec![CaOcrcReasonV1::PredictionNotFound],
                None,
                None,
                None,
            )?;
            feedback_results.push(result);
            continue;
        };
        let tracked = candidate.state.tracked.get(&origin_round).cloned();
        let Some(tracked_entry) = tracked else {
            let actual_delay = prediction.round.saturating_sub(origin_round);
            let mut reasons = vec![CaOcrcReasonV1::PredictionNotTracked];
            if actual_delay > candidate.policy.maximum_actual_delay_steps {
                reasons.push(CaOcrcReasonV1::ActualDelayOutOfBounds);
                mark_assumption_invalid(&mut candidate, "actual_delay_exceeded_declared_bound");
            }
            let result = make_feedback_result(
                feedback,
                Some(origin_round),
                Some(actual_delay),
                Some(prediction_receipt.sampled_proxy_delay),
                CaOcrcFeedbackDispositionV1::NotTracked,
                reasons,
                None,
                None,
                None,
            )?;
            feedback_results.push(result);
            continue;
        };
        let actual_delay = prediction.round.saturating_sub(origin_round);
        if actual_delay > candidate.policy.maximum_actual_delay_steps {
            let result = make_feedback_result(
                feedback,
                Some(origin_round),
                Some(actual_delay),
                Some(tracked_entry.prediction_receipt.sampled_proxy_delay),
                CaOcrcFeedbackDispositionV1::Rejected,
                vec![CaOcrcReasonV1::ActualDelayOutOfBounds],
                None,
                None,
                None,
            )?;
            mark_assumption_invalid(&mut candidate, "actual_delay_out_of_bounds");
            feedback_results.push(result);
            continue;
        }
        let validated = validate_feedback_against_prediction(
            &candidate,
            feedback,
            &tracked_entry.prediction_receipt,
            prediction.round,
            prediction.round_time_unix_seconds,
        );
        let (loss, delay, contribution) = match validated {
            Ok(value) => value,
            Err(reasons) => {
                let blocked = feedback.eligibility_report.eligibility
                    == LabelConsumptionEligibilityV1::Blocked;
                let unknown = feedback.eligibility_report.eligibility
                    == LabelConsumptionEligibilityV1::Unknown;
                if blocked {
                    candidate.state.tracked.remove(&origin_round);
                    mark_assumption_invalid(&mut candidate, "k1_feedback_blocked");
                } else if unknown {
                    mark_assumption_invalid(&mut candidate, "k1_feedback_unknown_or_unqualified");
                } else {
                    mark_assumption_invalid(&mut candidate, "k1_feedback_rejected_or_malformed");
                }
                let disposition = if unknown {
                    CaOcrcFeedbackDispositionV1::HeldUnknown
                } else if blocked {
                    CaOcrcFeedbackDispositionV1::Blocked
                } else {
                    CaOcrcFeedbackDispositionV1::Rejected
                };
                let result = make_feedback_result(
                    feedback,
                    Some(origin_round),
                    Some(actual_delay),
                    Some(tracked_entry.prediction_receipt.sampled_proxy_delay),
                    disposition,
                    reasons,
                    None,
                    None,
                    None,
                )?;
                feedback_results.push(result);
                continue;
            }
        };
        if actual_delay > tracked_entry.prediction_receipt.sampled_proxy_delay as u64 {
            let result = make_feedback_result(
                feedback,
                Some(origin_round),
                Some(delay),
                Some(tracked_entry.prediction_receipt.sampled_proxy_delay),
                CaOcrcFeedbackDispositionV1::NotTracked,
                vec![CaOcrcReasonV1::ProxyWindowExpired],
                Some(loss),
                None,
                None,
            )?;
            feedback_results.push(result);
            continue;
        }
        let survival = proxy_survival_vector(&candidate.policy)?[delay as usize];
        contribution_sum += contribution;
        candidate.state.accepted_label_revisions.insert(format!(
            "{}:{}",
            feedback.label_id, feedback.label_revision
        ));
        candidate.state.tracked.remove(&origin_round);
        let result = make_feedback_result(
            feedback,
            Some(origin_round),
            Some(delay),
            Some(tracked_entry.prediction_receipt.sampled_proxy_delay),
            CaOcrcFeedbackDispositionV1::Updated,
            Vec::new(),
            Some(loss),
            Some(survival),
            Some(contribution),
        )?;
        feedback_results.push(result);
    }

    let weighted_feedback_sum = contribution_sum;
    let update_term = trivial_loss
        .map(|loss| loss - candidate.policy.target_risk_alpha)
        .unwrap_or(0.0)
        + weighted_feedback_sum;
    let next_threshold = threshold_before + candidate.policy.learning_rate_eta * update_term;
    if !update_term.is_finite() || !next_threshold.is_finite() {
        return Err(CaOcrcErrorV1("CA-OCRC threshold update is non-finite".into()));
    }
    candidate.state.threshold = next_threshold;
    candidate.state.next_round = candidate
        .state
        .next_round
        .checked_add(1)
        .ok_or_else(|| CaOcrcErrorV1("CA-OCRC round counter overflow".into()))?;
    let state_after = state_digest(&candidate.state)?;
    let survival = proxy_survival_vector(&candidate.policy)?;
    let load_ell = proxy_tracking_load(&candidate.policy)?;
    let bound = if candidate.state.theorem_assumptions_intact
        && load_ell <= candidate.policy.tracking_capacity as f64
    {
        calculate_paper_bound(
            &candidate.policy,
            &survival,
            load_ell,
            candidate.state.next_round,
        )
    } else {
        None
    };

    let mut report = CaOcrcRoundReportV1 {
        schema: CA_OCRC_ROUND_REPORT_SCHEMA_V1.into(),
        engine_id: candidate.engine_id.clone(),
        round: prediction.round,
        prediction_receipt: receipt,
        evicted_rounds,
        feedback_results,
        immediate_trivial_loss: trivial_loss,
        weighted_feedback_sum,
        threshold_before,
        threshold_after: candidate.state.threshold,
        tracking_occupancy_after: candidate.state.tracked.len(),
        tracking_capacity: candidate.policy.tracking_capacity,
        tracking_load_ell_f: load_ell,
        proxy_survival_at_max_delay: survival[candidate.policy.maximum_actual_delay_steps as usize],
        theorem_assumptions_intact: candidate.state.theorem_assumptions_intact,
        paper_risk_gap_bound: bound,
        input_sha256: round_input_digest(input)?,
        state_before_sha256: state_before,
        state_after_sha256: state_after,
        report_sha256: String::new(),
    };
    report.report_sha256 = round_report_digest(&report)?;

    *engine = candidate;
    Ok(report)
}

/// Marks feedback that was still absent at the end of an online horizon.
/// This is audit-only; it does not impute a zero loss or change past thresholds.
pub fn close_ca_ocrc_feedback_horizon_v1(
    engine: &mut CaOcrcEngineV1,
) -> Result<Vec<u64>, CaOcrcErrorV1> {
    validate_engine(engine)?;
    let unresolved: Vec<u64> = engine
        .state
        .tracked
        .keys()
        .copied()
        .collect();
    let unresolved_selected: BTreeSet<u64> = engine
        .state
        .tracked
        .iter()
        .filter(|(_, entry)| entry.prediction_receipt.physical_measurement_selected)
        .map(|(round, _)| *round)
        .collect();
    if !unresolved_selected.is_empty() {
        mark_assumption_invalid(engine, "tracked_feedback_unresolved_at_horizon_end");
    }
    Ok(unresolved)
}

/// Computes the paper's arbitrary-delay optimized Bernoulli proxy distribution.
/// It tracks to d_max with probability p and otherwise draws -1.
pub fn optimal_arbitrary_delay_proxy_policy_v1(
    maximum_actual_delay_steps: u64,
    tracking_capacity: usize,
    learning_horizon: u64,
) -> Result<(Vec<f64>, f64, f64), CaOcrcErrorV1> {
    if maximum_actual_delay_steps == 0 || tracking_capacity == 0 || learning_horizon == 0 {
        return Err(CaOcrcErrorV1("optimizer requires positive d_max, K and horizon".into()));
    }
    let d_max = maximum_actual_delay_steps as f64;
    let upper = (tracking_capacity as f64 / d_max).min(1.0);
    if upper <= 0.0 || !upper.is_finite() {
        return Err(CaOcrcErrorV1("optimizer proxy-probability range is empty".into()));
    }
    let objective = |p: f64| {
        let ratio = d_max * p / tracking_capacity as f64;
        let exp_term = if ratio == 0.0 {
            0.0
        } else {
            (-((tracking_capacity as f64) * h_function(ratio))).exp()
        };
        (tracking_capacity as f64 + 1.0) / (p * learning_horizon as f64) + exp_term
    };
    let mut left = (upper * 1.0e-10).max(f64::MIN_POSITIVE);
    let mut right = upper;
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut x1 = right - (right - left) / phi;
    let mut x2 = left + (right - left) / phi;
    for _ in 0..160 {
        if objective(x1) <= objective(x2) {
            right = x2;
            x2 = x1;
            x1 = right - (right - left) / phi;
        } else {
            left = x1;
            x1 = x2;
            x2 = left + (right - left) / phi;
        }
    }
    let p = (left + right) / 2.0;
    let mut pmf = vec![0.0; maximum_actual_delay_steps as usize + 2];
    pmf[0] = 1.0 - p;
    pmf[maximum_actual_delay_steps as usize + 1] = p;
    Ok((pmf, p, objective(p)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy(k: usize, d_max: u64) -> CaOcrcPolicyV1 {
        let mut pmf = vec![0.0; d_max as usize + 2];
        pmf[0] = 0.5;
        pmf[d_max as usize + 1] = 0.5;
        CaOcrcPolicyV1 {
            schema: CA_OCRC_POLICY_SCHEMA_V1.into(),
            policy_id: "ca-ocrc-arbitrary-delay.v1".into(),
            revision: 1,
            target_risk_alpha: 0.10,
            learning_rate_eta: 0.05,
            bounded_score_upper: 1.0,
            maximum_actual_delay_steps: d_max,
            tracking_capacity: k,
            proxy_delay_pmf: pmf,
            actual_delay_contract_sha256: digest('f'),
            bounded_actual_delay_assumption_attested: true,
        }
    }

    fn engine(k: usize, d_max: u64) -> CaOcrcEngineV1 {
        create_ca_ocrc_engine_v1(
            "ca-ocrc-engine-001",
            "fab-clock",
            &digest('a'),
            &digest('b'),
            0.5,
            policy(k, d_max),
        )
        .unwrap()
    }

    fn prediction(
        round: u64,
        sample_id: &str,
        kind: CaOcrcSetKindV1,
        selected: bool,
        draw: Option<f64>,
    ) -> CaOcrcPredictionInputV1 {
        CaOcrcPredictionInputV1 {
            schema: CA_OCRC_PREDICTION_SCHEMA_V1.into(),
            round,
            sample_id: sample_id.into(),
            measurement_event_id: format!("event-{sample_id}"),
            event_time_unix_seconds: 1_800_000_000 + round as i64 * 60,
            prediction_at_unix_seconds: 1_800_000_005 + round as i64 * 60,
            round_time_unix_seconds: 1_800_000_010 + round as i64 * 60,
            clock_domain_id: "fab-clock".into(),
            global_model_artifact_sha256: digest('a'),
            interval_method_sha256: digest('b'),
            threshold_used: 0.5,
            prediction_set: match kind {
                CaOcrcSetKindV1::Empty => CaOcrcPredictionSetV1 {
                    kind,
                    lower: None,
                    upper: None,
                    measurand_id: "cd".into(),
                    unit: "nm".into(),
                },
                CaOcrcSetKindV1::Full => CaOcrcPredictionSetV1 {
                    kind,
                    lower: None,
                    upper: None,
                    measurand_id: "cd".into(),
                    unit: "nm".into(),
                },
                CaOcrcSetKindV1::Interval => CaOcrcPredictionSetV1 {
                    kind,
                    lower: Some(90.0),
                    upper: Some(110.0),
                    measurand_id: "cd".into(),
                    unit: "nm".into(),
                },
            },
            physical_measurement_selected: selected,
            proxy_uniform_draw: draw,
        }
    }

    fn feedback(
        round: u64,
        sample_id: &str,
        label_id: &str,
        revision: u64,
        status: LabelConsumptionEligibilityV1,
    ) -> CaOcrcLabelFeedbackV1 {
        let payload = digest('c');
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label_id.into(),
            label_revision: revision,
            measurement_event_id: format!("event-{sample_id}"),
            calibration_certificate_id: "cert-1".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: status,
            downstream_consumption_allowed:
                status == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: payload.clone(),
            measurement_sha256: digest('d'),
            qualified_corrected_value: 100.0,
            qualified_standard_uncertainty_upper_bound: Some(0.1),
            qualification_report_sha256: digest('e'),
            requalification_report_sha256: digest('f'),
            requalification_frontier_sha256: digest('0'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        let mut canonical = report.clone();
        canonical.report_sha256.clear();
        report.report_sha256 = json_sha256(&canonical).unwrap();
        let round_time = 1_800_000_010 + round as i64 * 60;
        CaOcrcLabelFeedbackV1 {
            schema: CA_OCRC_FEEDBACK_SCHEMA_V1.into(),
            feedback_round: round,
            sample_id: sample_id.into(),
            label_id: label_id.into(),
            label_revision: revision,
            measurement_event_id: format!("event-{sample_id}"),
            measurand_id: "cd".into(),
            unit: "nm".into(),
            label_payload_sha256: payload,
            label_value: 100.0,
            label_available_at_unix_seconds: 1_800_000_010 + round as i64 * 60,
            eligibility_report_available_at_unix_seconds: round_time,
            delivered_at_unix_seconds: round_time,
            clock_domain_id: "fab-clock".into(),
            eligibility_report: report,
        }
    }

    fn input(
        prediction: CaOcrcPredictionInputV1,
        feedbacks: Vec<CaOcrcLabelFeedbackV1>,
    ) -> CaOcrcRoundInputV1 {
        CaOcrcRoundInputV1 {
            schema: CA_OCRC_ROUND_INPUT_SCHEMA_V1.into(),
            prediction,
            feedbacks,
        }
    }

    #[test]
    fn pmf_maps_draws_to_proxy_delay_and_survival() {
        let policy = policy(2, 2);
        let survival = proxy_survival_vector(&policy).unwrap();
        assert_eq!(sample_proxy_delay(&policy, 0.0).unwrap(), -1);
        assert_eq!(sample_proxy_delay(&policy, 0.4999).unwrap(), -1);
        assert_eq!(sample_proxy_delay(&policy, 0.5).unwrap(), 2);
        assert_eq!(survival, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn capacity_is_never_exceeded_and_admission_happens_after_eviction() {
        let mut engine = engine(1, 2);
        let first = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(0, "s0", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                Vec::new(),
            ),
        )
        .unwrap();
        assert!(first.prediction_receipt.admitted_to_tracking);
        assert_eq!(first.tracking_occupancy_after, 1);

        let second = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(1, "s1", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                Vec::new(),
            ),
        )
        .unwrap();
        assert!(!second.prediction_receipt.admitted_to_tracking);
        assert_eq!(second.tracking_occupancy_after, 1);

        let third = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(2, "s2", CaOcrcSetKindV1::Interval, true, Some(0.0)),
                Vec::new(),
            ),
        )
        .unwrap();
        assert_eq!(third.tracking_occupancy_after, 1);
        assert!(third.evicted_rounds.is_empty());
        let fourth = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(3, "s3", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                Vec::new(),
            ),
        )
        .unwrap();
        assert_eq!(fourth.evicted_rounds, vec![0]);
        assert_eq!(fourth.tracking_occupancy_after, 1);
        assert!(fourth.prediction_receipt.admitted_to_tracking);
    }

    #[test]
    fn proxy_delay_minus_one_is_never_admitted() {
        let mut engine = engine(2, 2);
        let report = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(0, "s0", CaOcrcSetKindV1::Interval, true, Some(0.1)),
                Vec::new(),
            ),
        )
        .unwrap();
        assert_eq!(report.prediction_receipt.sampled_proxy_delay, -1);
        assert!(!report.prediction_receipt.admitted_to_tracking);
        assert_eq!(engine.state.tracked.len(), 0);
    }

    #[test]
    fn trivial_empty_and_full_sets_update_immediately_without_tracking() {
        let mut empty_engine = engine(1, 2);
        let empty = step_ca_ocrc_round_v1(
            &mut empty_engine,
            &input(
                prediction(0, "empty", CaOcrcSetKindV1::Empty, true, None),
                Vec::new(),
            ),
        )
        .unwrap();
        assert_eq!(empty.immediate_trivial_loss, Some(1.0));
        assert!((empty.threshold_after - (0.5 + 0.05 * (1.0 - 0.1))).abs() < 1e-12);
        assert_eq!(empty.tracking_occupancy_after, 0);

        let mut full_engine = engine(1, 2);
        let full = step_ca_ocrc_round_v1(
            &mut full_engine,
            &input(
                prediction(0, "full", CaOcrcSetKindV1::Full, true, None),
                Vec::new(),
            ),
        )
        .unwrap();
        assert_eq!(full.immediate_trivial_loss, Some(0.0));
        assert!((full.threshold_after - (0.5 + 0.05 * (0.0 - 0.1))).abs() < 1e-12);
        assert_eq!(full.tracking_occupancy_after, 0);
    }

    #[test]
    fn qualified_feedback_uses_prediction_time_interval_and_inverse_survival() {
        let mut engine = engine(2, 2);
        step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(0, "s0", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                Vec::new(),
            ),
        )
        .unwrap();
        let feedback = feedback(1, "s0", "l0", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy);
        // 0.9 maps to proxy delay 2; actual delay 1; F(1) = 0.5.
        let report = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(1, "s1", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                vec![feedback],
            ),
        )
        .unwrap();
        assert_eq!(report.feedback_results[0].disposition, CaOcrcFeedbackDispositionV1::Updated);
        assert_eq!(report.feedback_results[0].bounded_loss, Some(0.0));
        assert_eq!(report.feedback_results[0].proxy_survival_probability, Some(0.5));
        assert_eq!(report.feedback_results[0].weighted_update_contribution, Some(-0.2));
        assert!((report.threshold_after - (0.5 + 0.05 * -0.2)).abs() < 1e-12);
        assert_eq!(report.tracking_occupancy_after, 1);
    }

    #[test]
    fn zero_delay_feedback_is_processed_in_the_same_round_after_admission() {
        let mut configured = policy(1, 2);
        configured.proxy_delay_pmf = vec![0.0, 0.5, 0.0, 0.5];
        let mut engine = create_ca_ocrc_engine_v1(
            "ca-ocrc-zero-delay",
            "fab-clock",
            &digest('a'),
            &digest('b'),
            0.5,
            configured,
        )
        .unwrap();

        let report = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(0, "same-round", CaOcrcSetKindV1::Interval, true, Some(0.1)),
                vec![feedback(
                    0,
                    "same-round",
                    "label-same-round",
                    1,
                    LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                )],
            ),
        )
        .unwrap();

        assert_eq!(report.prediction_receipt.sampled_proxy_delay, 0);
        assert!(report.prediction_receipt.admitted_to_tracking);
        assert_eq!(report.feedback_results.len(), 1);
        assert_eq!(
            report.feedback_results[0].disposition,
            CaOcrcFeedbackDispositionV1::Updated
        );
        assert_eq!(report.feedback_results[0].actual_delay_steps, Some(0));
        assert_eq!(report.feedback_results[0].proxy_survival_probability, Some(1.0));
        assert_eq!(report.tracking_occupancy_after, 0);
        assert!((report.threshold_after - 0.495).abs() < 1.0e-12);
    }

    #[test]
    fn simultaneous_feedback_updates_are_summed_once_and_order_independently() {
        fn run(reverse: bool) -> String {
            let mut engine = engine(2, 2);
            for round in 0..2 {
                let id = format!("s{round}");
                step_ca_ocrc_round_v1(
                    &mut engine,
                    &input(
                        prediction(round, &id, CaOcrcSetKindV1::Interval, true, Some(0.9)),
                        Vec::new(),
                    ),
                )
                .unwrap();
            }
            let mut feedbacks = vec![
                feedback(2, "s0", "l0", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy),
                feedback(2, "s1", "l1", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy),
            ];
            if reverse {
                feedbacks.reverse();
            }
            let report = step_ca_ocrc_round_v1(
                &mut engine,
                &input(
                    prediction(2, "s2", CaOcrcSetKindV1::Interval, true, Some(0.1)),
                    feedbacks,
                ),
            )
            .unwrap();
            assert_eq!(report.feedback_results.len(), 2);
            assert!(
                report
                    .feedback_results
                    .iter()
                    .all(|result| result.disposition == CaOcrcFeedbackDispositionV1::Updated)
            );
            assert!((report.weighted_feedback_sum - (-0.4)).abs() < 1.0e-12);
            assert!((report.threshold_after - 0.48).abs() < 1.0e-12);
            report.report_sha256
        }
        assert_eq!(run(false), run(true));
    }

    #[test]
    fn feedback_on_exact_proxy_expiry_is_received_before_next_round_eviction() {
        let mut engine = engine(2, 2);
        step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(0, "s0", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                Vec::new(),
            ),
        )
        .unwrap();
        let feedback = feedback(2, "s0", "l0", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy);
        let report = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(2, "s2", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                vec![feedback],
            ),
        )
        .unwrap();
        assert_eq!(report.feedback_results[0].disposition, CaOcrcFeedbackDispositionV1::Updated);
        assert_eq!(report.feedback_results[0].actual_delay_steps, Some(2));
        assert_eq!(report.feedback_results[0].proxy_survival_probability, Some(0.5));
    }

    #[test]
    fn feedback_after_proxy_expiry_is_not_used_online() {
        let mut engine = engine(2, 2);
        step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(0, "s0", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                Vec::new(),
            ),
        )
        .unwrap();
        step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(1, "s1", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                Vec::new(),
            ),
        )
        .unwrap();
        step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(2, "s2", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                Vec::new(),
            ),
        )
        .unwrap();
        let late = feedback(3, "s0", "l0", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy);
        let report = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(3, "s3", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                vec![late],
            ),
        )
        .unwrap();
        assert_eq!(report.feedback_results[0].disposition, CaOcrcFeedbackDispositionV1::NotTracked);
        assert_eq!(report.feedback_results[0].reasons, vec![CaOcrcReasonV1::PredictionNotTracked]);
    }

    #[test]
    fn unknown_feedback_is_held_without_threshold_mutation() {
        let mut engine = engine(2, 2);
        step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(0, "s0", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                Vec::new(),
            ),
        )
        .unwrap();
        let before = engine.state.threshold;
        let report = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(1, "s1", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                vec![feedback(1, "s0", "l0", 1, LabelConsumptionEligibilityV1::Unknown)],
            ),
        )
        .unwrap();
        assert_eq!(report.feedback_results[0].disposition, CaOcrcFeedbackDispositionV1::HeldUnknown);
        assert_eq!(engine.state.threshold, before);
        assert!(engine.state.tracked.contains_key(&0));
        assert!(!report.theorem_assumptions_intact);
    }

    #[test]
    fn blocked_feedback_is_never_consumed_and_invalidates_guarantee_flag() {
        let mut engine = engine(2, 2);
        step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(0, "s0", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                Vec::new(),
            ),
        )
        .unwrap();
        let before = engine.state.threshold;
        let report = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(1, "s1", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                vec![feedback(1, "s0", "l0", 1, LabelConsumptionEligibilityV1::Blocked)],
            ),
        )
        .unwrap();
        assert_eq!(report.feedback_results[0].disposition, CaOcrcFeedbackDispositionV1::Blocked);
        assert_eq!(engine.state.threshold, before);
        assert!(!report.theorem_assumptions_intact);
        assert!(!engine.state.tracked.contains_key(&0));
    }

    #[test]
    fn physical_measurement_selection_is_not_conflated_with_tracking() {
        let mut engine = engine(2, 2);
        let report = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(0, "not-measured", CaOcrcSetKindV1::Interval, false, Some(0.9)),
                Vec::new(),
            ),
        )
        .unwrap();
        assert!(!report.prediction_receipt.admitted_to_tracking);
        assert_eq!(report.tracking_occupancy_after, 0);
        assert!(!report.theorem_assumptions_intact);
        assert!(engine.state.guarantee_invalid_reasons.contains("physical_measurement_not_selected"));
    }

    #[test]
    fn feedback_revision_event_value_and_payload_are_bound() {
        let mut engine = engine(2, 2);
        step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(0, "s0", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                Vec::new(),
            ),
        )
        .unwrap();
        let mut bad = feedback(1, "s0", "l0", 2, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy);
        bad.eligibility_report.label_revision = 1;
        let mut canonical = bad.eligibility_report.clone();
        canonical.report_sha256.clear();
        bad.eligibility_report.report_sha256 = json_sha256(&canonical).unwrap();
        let before = engine.state.threshold;
        let report = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(1, "s1", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                vec![bad],
            ),
        )
        .unwrap();
        assert_eq!(report.feedback_results[0].disposition, CaOcrcFeedbackDispositionV1::Rejected);
        assert_eq!(engine.state.threshold, before);
    }

    #[test]
    fn arbitrary_delay_underload_and_paper_bound_are_explicit() {
        let mut engine = engine(1, 2);
        // This PMF has F(1)=F(2)=0.5, so ell_F=1=K.
        let report = step_ca_ocrc_round_v1(
            &mut engine,
            &input(
                prediction(0, "s0", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                Vec::new(),
            ),
        )
        .unwrap();
        assert!((report.tracking_load_ell_f - 1.0).abs() < 1e-12);
        assert_eq!(report.paper_risk_gap_bound.is_some(), true);
    }

    #[test]
    fn proxy_policy_overload_is_rejected_before_engine_creation() {
        let mut policy = policy(1, 2);
        policy.proxy_delay_pmf = vec![0.0, 0.0, 0.0, 1.0];
        assert!(validate_policy(&policy).is_err());
    }

    #[test]
    fn seeded_draw_and_feedback_order_replay_identically() {
        fn run() -> String {
            let mut engine = engine(2, 2);
            let first = step_ca_ocrc_round_v1(
                &mut engine,
                &input(
                    prediction(0, "s0", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                    Vec::new(),
                ),
            )
            .unwrap();
            let second = step_ca_ocrc_round_v1(
                &mut engine,
                &input(
                    prediction(1, "s1", CaOcrcSetKindV1::Interval, true, Some(0.9)),
                    vec![feedback(1, "s0", "l0", 1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy)],
                ),
            )
            .unwrap();
            format!("{}{}", first.report_sha256, second.report_sha256)
        }
        assert_eq!(run(), run());
    }
}
