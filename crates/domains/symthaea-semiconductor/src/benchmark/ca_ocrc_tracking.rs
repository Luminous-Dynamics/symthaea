// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Faithful CA-OCRC tracking state machine for arbitrary bounded delays.
//!
//! Implements proxy-delay sampling, inclusive proxy deadlines, capacity
//! admission/eviction, degenerate-set direct losses and the q_G=1 IPW update.
//! The stochastic IID-delay variant is deliberately out of scope for v1.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::delayed_aci::{AciIntervalKindV1, AciPredictionIntervalV1};
use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};

pub const CA_OCRC_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc.v1";
pub const CA_OCRC_POLICY_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-policy.v1";
pub const CA_OCRC_ROUND_INPUT_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-round-input.v1";
pub const CA_OCRC_FEEDBACK_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-feedback.v1";
pub const CA_OCRC_ROUND_RECEIPT_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-round-receipt.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoCrcPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub target_risk: f64,
    pub learning_rate: f64,
    pub score_bound: f64,
    pub capacity: usize,
    pub maximum_real_delay: u64,
    /// PMF indexed by proxy delay + 1: index 0 is -1 (not admitted),
    /// index 1 is 0, ..., index dmax+1 is dmax.
    pub proxy_delay_probabilities: Vec<f64>,
    pub initial_threshold: f64,
    pub clock_domain_id: String,
    pub random_seed: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaoCrcGuaranteeStateV1 {
    UnderDeclaredArbitraryBoundedDelayAssumptions,
    InvalidDelayAssumptionObserved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaoCrcFeedbackDispositionV1 {
    Updated,
    HeldUnknown,
    Blocked,
    Rejected,
    Duplicate,
    Untracked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaoCrcFeedbackReasonV1 {
    NotTrackedAtArrival,
    DelayExceedsDeclaredBound,
    DeliveryRoundMismatch,
    ClockDomainMismatch,
    LabelNotCurrentK1,
    LabelNotEligible,
    DownstreamConsumptionNotAllowed,
    LabelIdentityInvalid,
    LabelRevisionMismatch,
    MeasurementEventMismatch,
    LabelPayloadDigestMismatch,
    EligibilityReportDigestMismatch,
    LabelValueMismatch,
    UncertaintyMissingOrInvalid,
    LabelUnavailableAtDelivery,
    ReportBeforeLabelAvailability,
    ReportUnavailableAtDelivery,
    DeliveryBeforePrediction,
    ProxyDeadlineViolated,
    FutureSourceRound,
    DuplicateAcceptedLabelRevision,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoCrcRoundInputV1 {
    pub schema: String,
    pub round: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub prediction_time_unix_seconds: i64,
    pub clock_domain_id: String,
    pub model_artifact_sha256: String,
    /// Must equal the engine threshold used to construct the supplied interval.
    pub threshold_used: f64,
    pub interval: AciPredictionIntervalV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoCrcPredictionReceiptV1 {
    pub schema: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub round: u64,
    pub event_time_unix_seconds: i64,
    pub prediction_time_unix_seconds: i64,
    pub clock_domain_id: String,
    pub model_artifact_sha256: String,
    pub threshold_used: f64,
    pub interval: AciPredictionIntervalV1,
    /// -1 means no proxy admission; non-negative values are sampled delays.
    pub sampled_proxy_delay: i64,
    pub admitted_to_tracking: bool,
    /// Full/empty intervals have a loss known by construction.
    pub known_loss: Option<f64>,
    pub policy_sha256: String,
    pub tracking_state_before_sha256: String,
    pub tracking_state_after_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoCrcFeedbackV1 {
    pub schema: String,
    pub source_round: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub label_value: f64,
    pub label_payload_sha256: String,
    pub label_available_at_unix_seconds: i64,
    pub eligibility_report_available_at_unix_seconds: i64,
    pub delivered_at_unix_seconds: i64,
    pub delivered_round: u64,
    pub clock_domain_id: String,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoCrcFeedbackResultV1 {
    pub source_round: u64,
    pub sample_id: String,
    pub label_id: Option<String>,
    pub label_revision: Option<u64>,
    pub disposition: CaoCrcFeedbackDispositionV1,
    pub reasons: Vec<CaoCrcFeedbackReasonV1>,
    pub actual_delay: Option<u64>,
    pub proxy_delay: Option<u64>,
    /// The arbitrary-bounded-delay version has q_G(dbar_max)=1.
    pub inverse_probability_weight: Option<f64>,
    pub loss: Option<f64>,
    pub threshold_contribution: f64,
    pub tracking_entry_retained: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoCrcTrackedEntryV1 {
    pub sample_id: String,
    pub measurement_event_id: String,
    pub round: u64,
    pub event_time_unix_seconds: i64,
    pub prediction_time_unix_seconds: i64,
    pub clock_domain_id: String,
    pub interval: AciPredictionIntervalV1,
    pub sampled_proxy_delay: u64,
    pub prediction_input_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoCrcRoundReceiptV1 {
    pub schema: String,
    pub engine_id: String,
    pub round: u64,
    pub prediction: CaoCrcPredictionReceiptV1,
    pub evicted_source_rounds: Vec<u64>,
    pub feedback_results: Vec<CaoCrcFeedbackResultV1>,
    pub threshold_before: f64,
    pub threshold_after: f64,
    pub known_loss_contribution: f64,
    pub received_feedback_contribution: f64,
    pub tracked_count_after_round: usize,
    pub tracking_capacity: usize,
    pub guarantee_state: CaoCrcGuaranteeStateV1,
    pub state_before_sha256: String,
    pub state_after_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaoCrcTrackingEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub policy: CaoCrcPolicyV1,
    pub policy_sha256: String,
    pub threshold: f64,
    pub current_round: Option<u64>,
    pub tracked: BTreeMap<u64, CaoCrcTrackedEntryV1>,
    pub accepted_label_revisions: BTreeSet<String>,
    pub seen_sample_ids: BTreeSet<String>,
    pub random_state: u64,
    pub total_rounds: u64,
    pub total_admitted: u64,
    pub total_not_admitted: u64,
    pub total_evicted: u64,
    pub total_received: u64,
    pub total_untracked_feedback: u64,
    pub guarantee_state: CaoCrcGuaranteeStateV1,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct CaoCrcErrorV1(pub String);

#[derive(Clone, Debug, PartialEq)]
pub struct CaoCrcArbitraryProxyDesignV1 {
    pub maximum_delay: u64,
    pub optimal_long_delay_probability: f64,
    pub probabilities: Vec<f64>,
    pub tracking_load: f64,
    pub risk_gap_bound: f64,
}

fn require_text(value: &str, field: &str) -> Result<(), CaoCrcErrorV1> {
    if value.trim().is_empty() {
        return Err(CaoCrcErrorV1(format!("{field}: expected non-empty text")));
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

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, CaoCrcErrorV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| CaoCrcErrorV1(format!("canonical JSON serialization failed: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn policy_digest(policy: &CaoCrcPolicyV1) -> Result<String, CaoCrcErrorV1> {
    json_sha256(policy)
}

fn prediction_digest(prediction: &CaoCrcPredictionReceiptV1) -> Result<String, CaoCrcErrorV1> {
    let mut canonical = prediction.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn round_receipt_digest(receipt: &CaoCrcRoundReceiptV1) -> Result<String, CaoCrcErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn feedback_report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, CaoCrcErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn valid_interval(interval: &AciPredictionIntervalV1) -> bool {
    match interval.kind {
        AciIntervalKindV1::Bounded => match (interval.lower, interval.upper) {
            (Some(lower), Some(upper)) => lower.is_finite() && upper.is_finite() && lower <= upper,
            _ => false,
        },
        AciIntervalKindV1::FullRealLine | AciIntervalKindV1::Empty => {
            interval.lower.is_none() && interval.upper.is_none()
        }
    }
}

fn interval_loss(interval: &AciPredictionIntervalV1, value: f64) -> f64 {
    match interval.kind {
        AciIntervalKindV1::FullRealLine => 0.0,
        AciIntervalKindV1::Empty => 1.0,
        AciIntervalKindV1::Bounded => {
            if value < interval.lower.expect("validated interval")
                || value > interval.upper.expect("validated interval")
            {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// F(a)=Pr(proxy_delay >= a); the PMF index is delay + 1.
fn proxy_probability(policy: &CaoCrcPolicyV1, proxy_delay: u64) -> f64 {
    policy.proxy_delay_probabilities[(proxy_delay + 1) as usize..]
        .iter()
        .sum()
}

fn tracking_load(policy: &CaoCrcPolicyV1) -> f64 {
    (1..=policy.maximum_real_delay)
        .map(|delay| proxy_probability(policy, delay))
        .sum()
}

fn validate_policy(policy: &CaoCrcPolicyV1) -> Result<(), CaoCrcErrorV1> {
    if policy.schema != CA_OCRC_POLICY_SCHEMA_V1 {
        return Err(CaoCrcErrorV1("CA-OCRC policy schema is unsupported".into()));
    }
    require_text(&policy.policy_id, "policy_id")?;
    require_text(&policy.clock_domain_id, "clock_domain_id")?;
    if !policy.target_risk.is_finite()
        || !(0.0..=1.0).contains(&policy.target_risk)
        || !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || !policy.score_bound.is_finite()
        || policy.score_bound < 1.0
        || policy.capacity == 0
        || policy.maximum_real_delay == 0
        || policy.capacity > policy.maximum_real_delay as usize
        || !policy.initial_threshold.is_finite()
        || !(0.0..=policy.score_bound).contains(&policy.initial_threshold)
    {
        return Err(CaoCrcErrorV1("CA-OCRC policy has invalid scalar parameters".into()));
    }
    if policy.proxy_delay_probabilities.len() != policy.maximum_real_delay as usize + 2 {
        return Err(CaoCrcErrorV1(
            "proxy delay PMF must cover -1 through maximum_real_delay".into(),
        ));
    }
    let mut total = 0.0;
    for probability in &policy.proxy_delay_probabilities {
        if !probability.is_finite() || *probability < 0.0 {
            return Err(CaoCrcErrorV1("proxy delay PMF contains invalid probability".into()));
        }
        total += probability;
    }
    if (total - 1.0).abs() > 1.0e-10 {
        return Err(CaoCrcErrorV1("proxy delay PMF probabilities must sum to one".into()));
    }
    if proxy_probability(policy, policy.maximum_real_delay) <= 0.0 {
        return Err(CaoCrcErrorV1(
            "F(d_max) must be positive for the arbitrary-delay risk bound".into(),
        ));
    }
    let load = tracking_load(policy);
    if !load.is_finite() || load > policy.capacity as f64 + 1.0e-10 {
        return Err(CaoCrcErrorV1("proxy tracking load exceeds capacity K".into()));
    }
    Ok(())
}

fn tracking_state_digest(engine: &CaoCrcTrackingEngineV1) -> Result<String, CaoCrcErrorV1> {
    #[derive(Serialize)]
    struct State<'a> {
        threshold: f64,
        current_round: Option<u64>,
        tracked: &'a BTreeMap<u64, CaoCrcTrackedEntryV1>,
        accepted_label_revisions: &'a BTreeSet<String>,
        seen_sample_ids: &'a BTreeSet<String>,
        random_state: u64,
        total_rounds: u64,
        total_admitted: u64,
        total_not_admitted: u64,
        total_evicted: u64,
        total_received: u64,
        total_untracked_feedback: u64,
        guarantee_state: CaoCrcGuaranteeStateV1,
    }
    json_sha256(&State {
        threshold: engine.threshold,
        current_round: engine.current_round,
        tracked: &engine.tracked,
        accepted_label_revisions: &engine.accepted_label_revisions,
        seen_sample_ids: &engine.seen_sample_ids,
        random_state: engine.random_state,
        total_rounds: engine.total_rounds,
        total_admitted: engine.total_admitted,
        total_not_admitted: engine.total_not_admitted,
        total_evicted: engine.total_evicted,
        total_received: engine.total_received,
        total_untracked_feedback: engine.total_untracked_feedback,
        guarantee_state: engine.guarantee_state,
    })
}

/// SplitMix64 is for deterministic replay/tests only, not production entropy.
fn next_uniform(random_state: &mut u64) -> f64 {
    *random_state = random_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *random_state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn sample_proxy_delay(policy: &CaoCrcPolicyV1, random_state: &mut u64) -> i64 {
    let uniform = next_uniform(random_state);
    let mut cumulative = 0.0;
    for (index, probability) in policy.proxy_delay_probabilities.iter().enumerate() {
        cumulative += probability;
        if uniform < cumulative || index + 1 == policy.proxy_delay_probabilities.len() {
            return index as i64 - 1;
        }
    }
    -1
}

pub fn create_cao_crc_engine_v1(
    engine_id: &str,
    policy: CaoCrcPolicyV1,
) -> Result<CaoCrcTrackingEngineV1, CaoCrcErrorV1> {
    require_text(engine_id, "engine_id")?;
    validate_policy(&policy)?;
    let policy_sha256 = policy_digest(&policy)?;
    Ok(CaoCrcTrackingEngineV1 {
        schema: CA_OCRC_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        threshold: policy.initial_threshold,
        random_state: policy.random_seed,
        policy,
        policy_sha256,
        current_round: None,
        tracked: BTreeMap::new(),
        accepted_label_revisions: BTreeSet::new(),
        seen_sample_ids: BTreeSet::new(),
        total_rounds: 0,
        total_admitted: 0,
        total_not_admitted: 0,
        total_evicted: 0,
        total_received: 0,
        total_untracked_feedback: 0,
        guarantee_state: CaoCrcGuaranteeStateV1::UnderDeclaredArbitraryBoundedDelayAssumptions,
    })
}

fn validate_feedback(
    engine: &CaoCrcTrackingEngineV1,
    tracked: &CaoCrcTrackedEntryV1,
    feedback: &CaoCrcFeedbackV1,
    current_round: u64,
) -> Result<Vec<CaoCrcFeedbackReasonV1>, CaoCrcErrorV1> {
    let prediction = tracked;
    let report = &feedback.eligibility_report;
    let mut reasons = Vec::new();

    if feedback.delivered_round != current_round {
        reasons.push(CaoCrcFeedbackReasonV1::DeliveryRoundMismatch);
    }
    if feedback.source_round != prediction.round {
        reasons.push(CaoCrcFeedbackReasonV1::MeasurementEventMismatch);
    }
    if feedback.clock_domain_id != engine.policy.clock_domain_id
        || feedback.clock_domain_id != prediction.clock_domain_id
    {
        reasons.push(CaoCrcFeedbackReasonV1::ClockDomainMismatch);
    }
    if feedback.sample_id != prediction.sample_id
        || feedback.measurement_event_id != prediction.measurement_event_id
        || report.measurement_event_id != prediction.measurement_event_id
    {
        reasons.push(CaoCrcFeedbackReasonV1::MeasurementEventMismatch);
    }
    if !feedback.label_value.is_finite()
        || !report.qualified_corrected_value.is_finite()
        || feedback.label_payload_sha256 != report.label_sha256
        || !valid_sha256(&feedback.label_payload_sha256)
    {
        reasons.push(CaoCrcFeedbackReasonV1::LabelPayloadDigestMismatch);
    }
    if feedback.label_value.to_bits() != report.qualified_corrected_value.to_bits() {
        reasons.push(CaoCrcFeedbackReasonV1::LabelValueMismatch);
    }
    if report.qualified_standard_uncertainty_upper_bound.is_none()
        || report.qualified_standard_uncertainty_upper_bound
            .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        reasons.push(CaoCrcFeedbackReasonV1::UncertaintyMissingOrInvalid);
    }
    if report.label_id.trim().is_empty() || report.label_revision == 0 {
        reasons.push(CaoCrcFeedbackReasonV1::LabelIdentityInvalid);
    }
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        reasons.push(CaoCrcFeedbackReasonV1::LabelNotCurrentK1);
    }
    match report.eligibility {
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
            if report.downstream_consumption_allowed => {}
        LabelConsumptionEligibilityV1::Unknown | LabelConsumptionEligibilityV1::Blocked => {
            reasons.push(CaoCrcFeedbackReasonV1::LabelNotEligible);
        }
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => {
            reasons.push(CaoCrcFeedbackReasonV1::DownstreamConsumptionNotAllowed);
        }
    }
    if feedback.label_available_at_unix_seconds < prediction.event_time_unix_seconds {
        reasons.push(CaoCrcFeedbackReasonV1::LabelUnavailableAtDelivery);
    }
    if feedback.eligibility_report_available_at_unix_seconds
        < feedback.label_available_at_unix_seconds
    {
        reasons.push(CaoCrcFeedbackReasonV1::ReportBeforeLabelAvailability);
    }
    if feedback.label_available_at_unix_seconds > feedback.delivered_at_unix_seconds
        || feedback.eligibility_report_available_at_unix_seconds
            > feedback.delivered_at_unix_seconds
    {
        reasons.push(CaoCrcFeedbackReasonV1::ReportUnavailableAtDelivery);
    }
    if feedback.delivered_at_unix_seconds < prediction.prediction_time_unix_seconds {
        reasons.push(CaoCrcFeedbackReasonV1::DeliveryBeforePrediction);
    }
    if !valid_sha256(&report.label_sha256)
        || feedback_report_digest(report)? != report.report_sha256
    {
        reasons.push(CaoCrcFeedbackReasonV1::EligibilityReportDigestMismatch);
    }
    reasons.sort();
    reasons.dedup();
    Ok(reasons)
}

pub fn process_cao_crc_round_v1(
    engine: &mut CaoCrcTrackingEngineV1,
    input: &CaoCrcRoundInputV1,
    feedbacks: &[CaoCrcFeedbackV1],
) -> Result<CaoCrcRoundReceiptV1, CaoCrcErrorV1> {
    if engine.schema != CA_OCRC_SCHEMA_V1 || policy_digest(&engine.policy)? != engine.policy_sha256 {
        return Err(CaoCrcErrorV1("CA-OCRC engine schema/policy integrity check failed".into()));
    }
    if input.schema != CA_OCRC_ROUND_INPUT_SCHEMA_V1 {
        return Err(CaoCrcErrorV1("CA-OCRC round input schema is unsupported".into()));
    }
    require_text(&input.sample_id, "round.sample_id")?;
    require_text(&input.measurement_event_id, "round.measurement_event_id")?;
    require_text(&input.clock_domain_id, "round.clock_domain_id")?;
    if input.clock_domain_id != engine.policy.clock_domain_id
        || !valid_sha256(&input.model_artifact_sha256)
        || !input.threshold_used.is_finite()
        || input.threshold_used.to_bits() != engine.threshold.to_bits()
        || !valid_interval(&input.interval)
        || input.event_time_unix_seconds > input.prediction_time_unix_seconds
    {
        return Err(CaoCrcErrorV1("round input is not bound to current model, threshold or clock".into()));
    }
    let expected_round = engine.current_round.map_or(Some(0), |round| round.checked_add(1));
    if expected_round != Some(input.round) {
        return Err(CaoCrcErrorV1("CA-OCRC rounds must begin at zero and advance sequentially".into()));
    }
    if engine.seen_sample_ids.contains(&input.sample_id) {
        return Err(CaoCrcErrorV1("sample ID has already been issued".into()));
    }

    // Work on a clone so an error never leaves a partially-applied round.
    let mut next_engine = engine.clone();
    let state_before = tracking_state_digest(&next_engine)?;
    let threshold_before = next_engine.threshold;

    // Step 1: evict after each proxy deadline has expired; deadline is inclusive.
    let expired: Vec<u64> = next_engine
        .tracked
        .iter()
        .filter(|(_, entry)| {
            entry.round.saturating_add(entry.sampled_proxy_delay) < input.round
        })
        .map(|(round, _)| *round)
        .collect();
    for round in &expired {
        next_engine.tracked.remove(round);
        next_engine.total_evicted = next_engine
            .total_evicted
            .checked_add(1)
            .ok_or_else(|| CaoCrcErrorV1("eviction counter overflow".into()))?;
    }

    // Step 2: direct loss for degenerate sets; otherwise draw proxy delay.
    let (sampled_proxy_delay, admitted, known_loss, known_contribution) =
        match input.interval.kind {
            AciIntervalKindV1::FullRealLine => {
                let loss = 0.0;
                (-1_i64, false, Some(loss), next_engine.policy.learning_rate * (loss - next_engine.policy.target_risk))
            }
            AciIntervalKindV1::Empty => {
                let loss = 1.0;
                (-1_i64, false, Some(loss), next_engine.policy.learning_rate * (loss - next_engine.policy.target_risk))
            }
            AciIntervalKindV1::Bounded => {
                let delay = sample_proxy_delay(&next_engine.policy, &mut next_engine.random_state);
                let can_track = delay >= 0 && next_engine.tracked.len() < next_engine.policy.capacity;
                if can_track {
                    let entry = CaoCrcTrackedEntryV1 {
                        sample_id: input.sample_id.clone(),
                        measurement_event_id: input.measurement_event_id.clone(),
                        round: input.round,
                        event_time_unix_seconds: input.event_time_unix_seconds,
                        prediction_time_unix_seconds: input.prediction_time_unix_seconds,
                        clock_domain_id: input.clock_domain_id.clone(),
                        interval: input.interval,
                        sampled_proxy_delay: delay as u64,
                        prediction_input_sha256: json_sha256(input)?,
                    };
                    next_engine.tracked.insert(input.round, entry);
                    next_engine.total_admitted = next_engine
                        .total_admitted
                        .checked_add(1)
                        .ok_or_else(|| CaoCrcErrorV1("admission counter overflow".into()))?;
                } else {
                    next_engine.total_not_admitted = next_engine
                        .total_not_admitted
                        .checked_add(1)
                        .ok_or_else(|| CaoCrcErrorV1("non-admission counter overflow".into()))?;
                }
                (delay, can_track, None, 0.0)
            }
        };

    // Step 3: receive feedback for tracked outcomes, in deterministic order.
    let mut sorted_feedbacks = feedbacks.to_vec();
    sorted_feedbacks.sort_by(|left, right| {
        (
            left.source_round,
            left.eligibility_report.label_id.as_str(),
            left.eligibility_report.label_revision,
        )
            .cmp(&(
                right.source_round,
                right.eligibility_report.label_id.as_str(),
                right.eligibility_report.label_revision,
            ))
    });
    let mut feedback_results = Vec::with_capacity(sorted_feedbacks.len());
    let mut received_contribution = 0.0;

    for feedback in &sorted_feedbacks {
        if feedback.schema != CA_OCRC_FEEDBACK_SCHEMA_V1 {
            return Err(CaoCrcErrorV1("CA-OCRC feedback schema is unsupported".into()));
        }
        let label_key = format!(
            "{}:{}",
            feedback.eligibility_report.label_id,
            feedback.eligibility_report.label_revision
        );
        if next_engine.accepted_label_revisions.contains(&label_key) {
            feedback_results.push(CaoCrcFeedbackResultV1 {
                source_round: feedback.source_round,
                sample_id: feedback.sample_id.clone(),
                label_id: Some(feedback.eligibility_report.label_id.clone()),
                label_revision: Some(feedback.eligibility_report.label_revision),
                disposition: CaoCrcFeedbackDispositionV1::Duplicate,
                reasons: vec![CaoCrcFeedbackReasonV1::DuplicateAcceptedLabelRevision],
                actual_delay: input.round.checked_sub(feedback.source_round),
                proxy_delay: None,
                inverse_probability_weight: None,
                loss: None,
                threshold_contribution: 0.0,
                tracking_entry_retained: false,
            });
            continue;
        }

        let Some(entry) = next_engine.tracked.get(&feedback.source_round).cloned() else {
            let actual_delay = input.round.checked_sub(feedback.source_round);
            if actual_delay.is_none() {
                feedback_results.push(CaoCrcFeedbackResultV1 {
                    source_round: feedback.source_round,
                    sample_id: feedback.sample_id.clone(),
                    label_id: Some(feedback.eligibility_report.label_id.clone()),
                    label_revision: Some(feedback.eligibility_report.label_revision),
                    disposition: CaoCrcFeedbackDispositionV1::Rejected,
                    reasons: vec![CaoCrcFeedbackReasonV1::FutureSourceRound],
                    actual_delay: None,
                    proxy_delay: None,
                    inverse_probability_weight: None,
                    loss: None,
                    threshold_contribution: 0.0,
                    tracking_entry_retained: false,
                });
                continue;
            }
            if actual_delay.is_some_and(|delay| delay > next_engine.policy.maximum_real_delay) {
                next_engine.guarantee_state = CaoCrcGuaranteeStateV1::InvalidDelayAssumptionObserved;
                feedback_results.push(CaoCrcFeedbackResultV1 {
                    source_round: feedback.source_round,
                    sample_id: feedback.sample_id.clone(),
                    label_id: Some(feedback.eligibility_report.label_id.clone()),
                    label_revision: Some(feedback.eligibility_report.label_revision),
                    disposition: CaoCrcFeedbackDispositionV1::Untracked,
                    reasons: vec![CaoCrcFeedbackReasonV1::DelayExceedsDeclaredBound],
                    actual_delay,
                    proxy_delay: None,
                    inverse_probability_weight: None,
                    loss: None,
                    threshold_contribution: 0.0,
                    tracking_entry_retained: false,
                });
                next_engine.total_untracked_feedback += 1;
                continue;
            }
            next_engine.total_untracked_feedback = next_engine
                .total_untracked_feedback
                .checked_add(1)
                .ok_or_else(|| CaoCrcErrorV1("untracked feedback counter overflow".into()))?;
            feedback_results.push(CaoCrcFeedbackResultV1 {
                source_round: feedback.source_round,
                sample_id: feedback.sample_id.clone(),
                label_id: Some(feedback.eligibility_report.label_id.clone()),
                label_revision: Some(feedback.eligibility_report.label_revision),
                disposition: CaoCrcFeedbackDispositionV1::Untracked,
                reasons: vec![CaoCrcFeedbackReasonV1::NotTrackedAtArrival],
                actual_delay,
                proxy_delay: None,
                inverse_probability_weight: None,
                loss: None,
                threshold_contribution: 0.0,
                tracking_entry_retained: false,
            });
            continue;
        };

        let actual_delay = input
            .round
            .checked_sub(feedback.source_round)
            .ok_or_else(|| CaoCrcErrorV1("feedback source round is in the future".into()))?;
        let mut reasons = validate_feedback(&next_engine, &entry, feedback, input.round)?;
        if actual_delay > entry.sampled_proxy_delay {
            reasons.push(CaoCrcFeedbackReasonV1::ProxyDeadlineViolated);
        }
        reasons.sort();
        reasons.dedup();

        if !reasons.is_empty() {
            let only_unknown = feedback.eligibility_report.eligibility == LabelConsumptionEligibilityV1::Unknown
                && reasons.iter().all(|reason| *reason == CaoCrcFeedbackReasonV1::LabelNotEligible);
            let only_blocked = feedback.eligibility_report.eligibility == LabelConsumptionEligibilityV1::Blocked
                && reasons.iter().all(|reason| *reason == CaoCrcFeedbackReasonV1::LabelNotEligible);
            let disposition = if only_unknown {
                CaoCrcFeedbackDispositionV1::HeldUnknown
            } else if only_blocked {
                CaoCrcFeedbackDispositionV1::Blocked
            } else {
                CaoCrcFeedbackDispositionV1::Rejected
            };
            feedback_results.push(CaoCrcFeedbackResultV1 {
                source_round: feedback.source_round,
                sample_id: feedback.sample_id.clone(),
                label_id: Some(feedback.eligibility_report.label_id.clone()),
                label_revision: Some(feedback.eligibility_report.label_revision),
                disposition,
                reasons,
                actual_delay: Some(actual_delay),
                proxy_delay: Some(entry.sampled_proxy_delay),
                inverse_probability_weight: None,
                loss: None,
                threshold_contribution: 0.0,
                tracking_entry_retained: true,
            });
            continue;
        }

        let survival = proxy_probability(&next_engine.policy, actual_delay);
        if !survival.is_finite() || survival <= 0.0 {
            return Err(CaoCrcErrorV1("F(actual_delay) is zero or invalid for a tracked outcome".into()));
        }
        // Arbitrary bounded delays have q_G(dbar_max)=1. The stochastic
        // two-level q correction is explicitly deferred to v2.
        let weight = 1.0 / survival;
        let loss = interval_loss(&entry.interval, feedback.label_value);
        let contribution = next_engine.policy.learning_rate
            * (loss - next_engine.policy.target_risk)
            * weight;
        if !weight.is_finite() || !contribution.is_finite() || loss < 0.0 || loss > next_engine.policy.score_bound {
            return Err(CaoCrcErrorV1("IPW weight or loss is invalid".into()));
        }

        received_contribution += contribution;
        next_engine.tracked.remove(&feedback.source_round);
        next_engine.accepted_label_revisions.insert(label_key);
        next_engine.total_received = next_engine
            .total_received
            .checked_add(1)
            .ok_or_else(|| CaoCrcErrorV1("received feedback counter overflow".into()))?;
        feedback_results.push(CaoCrcFeedbackResultV1 {
            source_round: feedback.source_round,
            sample_id: feedback.sample_id.clone(),
            label_id: Some(feedback.eligibility_report.label_id.clone()),
            label_revision: Some(feedback.eligibility_report.label_revision),
            disposition: CaoCrcFeedbackDispositionV1::Updated,
            reasons: Vec::new(),
            actual_delay: Some(actual_delay),
            proxy_delay: Some(entry.sampled_proxy_delay),
            inverse_probability_weight: Some(weight),
            loss: Some(loss),
            threshold_contribution: contribution,
            tracking_entry_retained: false,
        });
    }

    // Step 4: exact threshold update (22), with q=1 for this arbitrary-delay v1.
    let threshold_after = next_engine.threshold + known_contribution + received_contribution;
    if !threshold_after.is_finite() {
        return Err(CaoCrcErrorV1("CA-OCRC threshold update became non-finite".into()));
    }
    next_engine.threshold = threshold_after;
    next_engine.current_round = Some(input.round);
    next_engine.seen_sample_ids.insert(input.sample_id.clone());
    next_engine.total_rounds = next_engine
        .total_rounds
        .checked_add(1)
        .ok_or_else(|| CaoCrcErrorV1("round counter overflow".into()))?;

    let state_after = tracking_state_digest(&next_engine)?;
    let mut prediction_receipt = CaoCrcPredictionReceiptV1 {
        schema: "luminous.semi-bench.ca-ocrc-prediction-receipt.v1".into(),
        sample_id: input.sample_id.clone(),
        measurement_event_id: input.measurement_event_id.clone(),
        round: input.round,
        event_time_unix_seconds: input.event_time_unix_seconds,
        prediction_time_unix_seconds: input.prediction_time_unix_seconds,
        clock_domain_id: input.clock_domain_id.clone(),
        model_artifact_sha256: input.model_artifact_sha256.clone(),
        threshold_used: input.threshold_used,
        interval: input.interval,
        sampled_proxy_delay,
        admitted_to_tracking: admitted,
        known_loss,
        policy_sha256: next_engine.policy_sha256.clone(),
        tracking_state_before_sha256: state_before.clone(),
        tracking_state_after_sha256: state_after.clone(),
        receipt_sha256: String::new(),
    };
    prediction_receipt.receipt_sha256 = prediction_digest(&prediction_receipt)?;
    let mut receipt = CaoCrcRoundReceiptV1 {
        schema: CA_OCRC_ROUND_RECEIPT_SCHEMA_V1.into(),
        engine_id: next_engine.engine_id.clone(),
        round: input.round,
        prediction: prediction_receipt,
        evicted_source_rounds: expired,
        feedback_results,
        threshold_before,
        threshold_after,
        known_loss_contribution: known_contribution,
        received_feedback_contribution: received_contribution,
        tracked_count_after_round: next_engine.tracked.len(),
        tracking_capacity: next_engine.policy.capacity,
        guarantee_state: next_engine.guarantee_state,
        state_before_sha256: state_before,
        state_after_sha256: state_after,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = round_receipt_digest(&receipt)?;
    *engine = next_engine;
    Ok(receipt)
}

/// Theorem 1 bound for arbitrary bounded delays:
/// B/(eta*T) + (K+1)/(F(dmax)*T) + exp(-K*h(load/K)).
pub fn cao_crc_arbitrary_delay_risk_bound_v1(
    policy: &CaoCrcPolicyV1,
    horizon: u64,
) -> Result<f64, CaoCrcErrorV1> {
    validate_policy(policy)?;
    if horizon == 0 {
        return Err(CaoCrcErrorV1("risk-bound horizon must be positive".into()));
    }
    let f_dmax = proxy_probability(policy, policy.maximum_real_delay);
    let load = tracking_load(policy);
    let x = load / policy.capacity as f64;
    let h = x - 1.0 - x.ln();
    let saturation_term = (-((policy.capacity as f64) * h)).exp();
    let bound = policy.score_bound / (policy.learning_rate * horizon as f64)
        + (policy.capacity as f64 + 1.0) / (f_dmax * horizon as f64)
        + saturation_term;
    if !bound.is_finite() {
        return Err(CaoCrcErrorV1("arbitrary-delay risk bound is non-finite".into()));
    }
    Ok(bound)
}

/// Theorem/proposition 1 proxy survival design for arbitrary bounded delays:
/// F*(a)=p* for a=1..dmax, with F*(0)=1 and F*(-1)=1. The PMF therefore
/// admits a round with proxy delay 0 with probability 1-p* and dmax with p*.
pub fn optimal_arbitrary_cao_crc_proxy_design_v1(
    maximum_delay: u64,
    capacity: usize,
    score_bound: f64,
    learning_rate: f64,
    horizon: u64,
) -> Result<CaoCrcArbitraryProxyDesignV1, CaoCrcErrorV1> {
    if maximum_delay == 0
        || capacity == 0
        || capacity > maximum_delay as usize
        || !score_bound.is_finite()
        || score_bound < 1.0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
        || horizon == 0
    {
        return Err(CaoCrcErrorV1("optimal proxy design inputs are invalid".into()));
    }
    let upper = (capacity as f64 / maximum_delay as f64).min(1.0);
    let mut lower = (upper * 1.0e-12).max(1.0e-15);
    let mut right = upper;
    let golden = (5.0_f64.sqrt() - 1.0) / 2.0;
    let objective = |p: f64| {
        let x = maximum_delay as f64 * p / capacity as f64;
        let h = x - 1.0 - x.ln();
        (capacity as f64 + 1.0) / (p * horizon as f64)
            + (-((capacity as f64) * h)).exp()
    };
    for _ in 0..192 {
        let left_third = right - golden * (right - lower);
        let right_third = lower + golden * (right - lower);
        if objective(left_third) <= objective(right_third) {
            right = right_third;
        } else {
            lower = left_third;
        }
    }
    let p_star = (lower + right) * 0.5;
    let mut probabilities = vec![0.0; maximum_delay as usize + 2];
    probabilities[1] = 1.0 - p_star;
    probabilities[maximum_delay as usize + 1] = p_star;
    let load = maximum_delay as f64 * p_star;
    let design_policy = CaoCrcPolicyV1 {
        schema: CA_OCRC_POLICY_SCHEMA_V1.into(),
        policy_id: "cao-crc-arbitrary-optimal-proxy.v1".into(),
        target_risk: 0.1,
        learning_rate,
        score_bound,
        capacity,
        maximum_real_delay: maximum_delay,
        proxy_delay_probabilities: probabilities.clone(),
        initial_threshold: 0.5 * score_bound,
        clock_domain_id: "design-only".into(),
        random_seed: 1,
    };
    let risk_gap_bound = cao_crc_arbitrary_delay_risk_bound_v1(&design_policy, horizon)?;
    Ok(CaoCrcArbitraryProxyDesignV1 {
        maximum_delay,
        optimal_long_delay_probability: p_star,
        probabilities,
        tracking_load: load,
        risk_gap_bound,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn base_policy() -> CaoCrcPolicyV1 {
        CaoCrcPolicyV1 {
            schema: CA_OCRC_POLICY_SCHEMA_V1.into(),
            policy_id: "cao-crc-fixture.v1".into(),
            target_risk: 0.10,
            learning_rate: 0.05,
            score_bound: 1.0,
            capacity: 2,
            maximum_real_delay: 2,
            // Proxy d=-1:0; d=0:0.5; d=1:0; d=2:0.5.
            proxy_delay_probabilities: vec![0.0, 0.5, 0.0, 0.5],
            initial_threshold: 0.5,
            clock_domain_id: "fab-clock".into(),
            random_seed: 42,
        }
    }

    fn engine() -> CaoCrcTrackingEngineV1 {
        create_cao_crc_engine_v1("cao-crc-engine-1", base_policy()).unwrap()
    }

    fn round_input(
        round: u64,
        sample_id: &str,
        event_id: &str,
        interval: AciPredictionIntervalV1,
        threshold: f64,
    ) -> CaoCrcRoundInputV1 {
        CaoCrcRoundInputV1 {
            schema: CA_OCRC_ROUND_INPUT_SCHEMA_V1.into(),
            round,
            sample_id: sample_id.into(),
            measurement_event_id: event_id.into(),
            event_time_unix_seconds: 100 + round as i64 * 10,
            prediction_time_unix_seconds: 101 + round as i64 * 10,
            clock_domain_id: "fab-clock".into(),
            model_artifact_sha256: digest('a'),
            threshold_used: threshold,
            interval,
        }
    }

    fn feedback(
        source_round: u64,
        delivered_round: u64,
        sample_id: &str,
        event_id: &str,
        label_id: &str,
        value: f64,
    ) -> CaoCrcFeedbackV1 {
        let payload = digest('f');
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label_id.into(),
            label_revision: 1,
            measurement_event_id: event_id.into(),
            calibration_certificate_id: "cal-cert-cao-crc".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            downstream_consumption_allowed: true,
            reasons: Vec::new(),
            label_sha256: payload.clone(),
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
        CaoCrcFeedbackV1 {
            schema: CA_OCRC_FEEDBACK_SCHEMA_V1.into(),
            source_round,
            sample_id: sample_id.into(),
            measurement_event_id: event_id.into(),
            label_value: value,
            label_payload_sha256: payload,
            label_available_at_unix_seconds: 100 + delivered_round as i64 * 10,
            eligibility_report_available_at_unix_seconds: 100 + delivered_round as i64 * 10,
            delivered_at_unix_seconds: 101 + delivered_round as i64 * 10,
            delivered_round,
            clock_domain_id: "fab-clock".into(),
            eligibility_report: report,
        }
    }

    #[test]
    fn tracking_capacity_never_exceeds_k() {
        let mut engine = engine();
        for round in 0..10 {
            let input = round_input(
                round,
                &format!("sample-{round}"),
                &format!("event-{round}"),
                AciPredictionIntervalV1::bounded(9.0, 11.0),
                engine.threshold,
            );
            let receipt = process_cao_crc_round_v1(&mut engine, &input, &[]).unwrap();
            assert!(receipt.tracked_count_after_round <= 2);
            assert_eq!(receipt.tracking_capacity, 2);
        }
        assert!(engine.total_admitted + engine.total_not_admitted >= 10);
    }

    #[test]
    fn full_and_empty_sets_update_known_loss_without_tracking() {
        let mut full_engine = engine();
        let full_input = round_input(
            0, "full", "event-full", AciPredictionIntervalV1::full_real_line(), 0.5,
        );
        let full = process_cao_crc_round_v1(&mut full_engine, &full_input, &[]).unwrap();
        assert_eq!(full.prediction.sampled_proxy_delay, -1);
        assert!(!full.prediction.admitted_to_tracking);
        assert!((full.known_loss_contribution + 0.005).abs() < 1.0e-12);
        assert!(full_engine.tracked.is_empty());

        let mut empty_engine = engine();
        let empty_input = round_input(
            0, "empty", "event-empty", AciPredictionIntervalV1::empty(), 0.5,
        );
        let empty = process_cao_crc_round_v1(&mut empty_engine, &empty_input, &[]).unwrap();
        assert!((empty.known_loss_contribution - 0.045).abs() < 1.0e-12);
        assert!(empty_engine.tracked.is_empty());
    }

    #[test]
    fn proxy_deadline_is_inclusive_and_eviction_precedes_admission() {
        let mut policy = base_policy();
        policy.proxy_delay_probabilities = vec![0.0, 0.0, 0.0, 1.0];
        let mut engine = create_cao_crc_engine_v1("cao-crc-engine-1", policy).unwrap();
        let first = round_input(0, "s0", "e0", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        assert!(process_cao_crc_round_v1(&mut engine, &first, &[]).unwrap().prediction.admitted_to_tracking);
        let second = round_input(1, "s1", "e1", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        process_cao_crc_round_v1(&mut engine, &second, &[]).unwrap();
        let third = round_input(2, "s2", "e2", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        let r2 = process_cao_crc_round_v1(&mut engine, &third, &[]).unwrap();
        assert!(r2.evicted_source_rounds.is_empty());
        let fourth = round_input(3, "s3", "e3", AciPredictionIntervalV1::bounded(0.0, 1.0), engine.threshold);
        let r3 = process_cao_crc_round_v1(&mut engine, &fourth, &[]).unwrap();
        assert_eq!(r3.evicted_source_rounds, vec![0]);
    }

    #[test]
    fn immediate_feedback_uses_inverse_survival_weight_and_consumes_tracking() {
        let mut policy = base_policy();
        policy.proxy_delay_probabilities = vec![0.0, 0.0, 0.0, 1.0];
        let mut engine = create_cao_crc_engine_v1("cao-crc-engine-1", policy).unwrap();
        let first = round_input(0, "s0", "e0", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        process_cao_crc_round_v1(&mut engine, &first, &[]).unwrap();
        let second = round_input(1, "s1", "e1", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        let receipt = process_cao_crc_round_v1(
            &mut engine,
            &second,
            &[feedback(0, 1, "s0", "e0", "label-0", 2.0)],
        )
        .unwrap();
        assert_eq!(receipt.feedback_results[0].disposition, CaoCrcFeedbackDispositionV1::Updated);
        assert_eq!(receipt.feedback_results[0].inverse_probability_weight, Some(1.0));
        assert_eq!(receipt.feedback_results[0].loss, Some(1.0));
        assert!(!engine.tracked.contains_key(&0));
    }

    #[test]
    fn delayed_feedback_after_proxy_expiry_is_untracked() {
        let mut policy = base_policy();
        policy.proxy_delay_probabilities = vec![0.0, 0.0, 1.0, 0.0]; // proxy delay 1
        let mut engine = create_cao_crc_engine_v1("cao-crc-engine-1", policy).unwrap();
        let first = round_input(0, "s0", "e0", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        process_cao_crc_round_v1(&mut engine, &first, &[]).unwrap();
        let second = round_input(1, "s1", "e1", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        process_cao_crc_round_v1(&mut engine, &second, &[]).unwrap();
        let threshold = engine.threshold;
        let third = round_input(2, "s2", "e2", AciPredictionIntervalV1::bounded(0.0, 1.0), threshold);
        let receipt = process_cao_crc_round_v1(
            &mut engine,
            &third,
            &[feedback(0, 2, "s0", "e0", "label-0", 2.0)],
        )
        .unwrap();
        assert_eq!(receipt.feedback_results[0].disposition, CaoCrcFeedbackDispositionV1::Untracked);
        assert_eq!(engine.threshold, threshold);
    }

    #[test]
    fn unknown_k1_outcome_is_held_without_consuming_tracking_entry() {
        let mut engine = engine();
        let first = round_input(0, "s0", "e0", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        process_cao_crc_round_v1(&mut engine, &first, &[]).unwrap();
        let mut unknown = feedback(0, 1, "s0", "e0", "label-0", 2.0);
        unknown.eligibility_report.eligibility = LabelConsumptionEligibilityV1::Unknown;
        unknown.eligibility_report.downstream_consumption_allowed = false;
        let mut canonical = unknown.eligibility_report.clone();
        canonical.report_sha256.clear();
        unknown.eligibility_report.report_sha256 = json_sha256(&canonical).unwrap();
        let second = round_input(1, "s1", "e1", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        let receipt = process_cao_crc_round_v1(&mut engine, &second, &[unknown]).unwrap();
        assert_eq!(receipt.feedback_results[0].disposition, CaoCrcFeedbackDispositionV1::HeldUnknown);
        assert!(engine.tracked.contains_key(&0));
    }

    #[test]
    fn corrected_k1_report_can_be_represented_before_proxy_deadline() {
        let mut engine = engine();
        let first = round_input(0, "s0", "e0", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        process_cao_crc_round_v1(&mut engine, &first, &[]).unwrap();
        let mut unknown = feedback(0, 1, "s0", "e0", "label-0", 2.0);
        unknown.eligibility_report.eligibility = LabelConsumptionEligibilityV1::Unknown;
        unknown.eligibility_report.downstream_consumption_allowed = false;
        let mut canonical = unknown.eligibility_report.clone();
        canonical.report_sha256.clear();
        unknown.eligibility_report.report_sha256 = json_sha256(&canonical).unwrap();
        let second = round_input(1, "s1", "e1", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        process_cao_crc_round_v1(&mut engine, &second, &[unknown]).unwrap();
        let third = round_input(2, "s2", "e2", AciPredictionIntervalV1::bounded(0.0, 1.0), engine.threshold);
        let receipt = process_cao_crc_round_v1(
            &mut engine,
            &third,
            &[feedback(0, 2, "s0", "e0", "label-0", 2.0)],
        )
        .unwrap();
        assert_eq!(receipt.feedback_results[0].disposition, CaoCrcFeedbackDispositionV1::Updated);
    }

    #[test]
    fn mismatched_event_or_digest_is_rejected_without_threshold_update() {
        let mut policy = base_policy();
        policy.proxy_delay_probabilities = vec![0.0, 0.0, 0.0, 1.0];
        let mut engine = create_cao_crc_engine_v1("cao-crc-engine-1", policy).unwrap();
        let first = round_input(0, "s0", "e0", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        process_cao_crc_round_v1(&mut engine, &first, &[]).unwrap();
        let second = round_input(1, "s1", "e1", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        let mut wrong = feedback(0, 1, "s0", "other-event", "label-0", 2.0);
        wrong.eligibility_report.report_sha256.clear();
        let mut canonical = wrong.eligibility_report.clone();
        canonical.report_sha256.clear();
        wrong.eligibility_report.report_sha256 = json_sha256(&canonical).unwrap();
        let threshold = engine.threshold;
        let receipt = process_cao_crc_round_v1(&mut engine, &second, &[wrong]).unwrap();
        assert_eq!(receipt.feedback_results[0].disposition, CaoCrcFeedbackDispositionV1::Rejected);
        assert_eq!(engine.threshold, threshold);
        assert!(engine.tracked.contains_key(&0));
    }

    #[test]
    fn duplicate_accepted_label_revision_is_a_no_update() {
        let mut policy = base_policy();
        policy.proxy_delay_probabilities = vec![0.0, 0.0, 0.0, 1.0];
        let mut engine = create_cao_crc_engine_v1("cao-crc-engine-1", policy).unwrap();
        let first = round_input(0, "s0", "e0", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        process_cao_crc_round_v1(&mut engine, &first, &[]).unwrap();
        let accepted = feedback(0, 1, "s0", "e0", "label-0", 2.0);
        let second = round_input(1, "s1", "e1", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        process_cao_crc_round_v1(&mut engine, &second, &[accepted.clone()]).unwrap();
        let threshold = engine.threshold;
        let third = round_input(2, "s2", "e2", AciPredictionIntervalV1::bounded(0.0, 1.0), threshold);
        let receipt = process_cao_crc_round_v1(&mut engine, &third, &[accepted]).unwrap();
        assert_eq!(receipt.feedback_results[0].disposition, CaoCrcFeedbackDispositionV1::Duplicate);
        assert_eq!(engine.threshold, threshold);
    }

    #[test]
    fn full_capacity_denies_admission_after_sampling_proxy_delay() {
        let mut policy = base_policy();
        policy.capacity = 1;
        policy.proxy_delay_probabilities = vec![0.0, 0.0, 0.0, 1.0];
        let mut engine = create_cao_crc_engine_v1("cao-crc-engine-1", policy).unwrap();
        let first = round_input(0, "s0", "e0", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        assert!(process_cao_crc_round_v1(&mut engine, &first, &[]).unwrap().prediction.admitted_to_tracking);
        let second = round_input(1, "s1", "e1", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        let receipt = process_cao_crc_round_v1(&mut engine, &second, &[]).unwrap();
        assert!(!receipt.prediction.admitted_to_tracking);
        assert_eq!(receipt.prediction.sampled_proxy_delay, 2);
        assert_eq!(engine.tracked.len(), 1);
    }

    #[test]
    fn fixed_seed_round_stream_replays_identically() {
        fn run() -> Vec<(u64, f64, usize)> {
            let mut engine = engine();
            let mut output = Vec::new();
            for round in 0..40 {
                let input = round_input(
                    round,
                    &format!("sample-{round}"),
                    &format!("event-{round}"),
                    AciPredictionIntervalV1::bounded(0.0, 1.0),
                    engine.threshold,
                );
                let receipt = process_cao_crc_round_v1(&mut engine, &input, &[]).unwrap();
                output.push((round, receipt.threshold_after, receipt.tracked_count_after_round));
            }
            output
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn proxy_load_constraint_is_fail_closed() {
        let mut policy = base_policy();
        policy.proxy_delay_probabilities = vec![0.0, 0.0, 1.0, 0.0]; // load F(1)+F(2)=2
        policy.capacity = 1;
        assert!(create_cao_crc_engine_v1("invalid", policy).is_err());
    }

    #[test]
    fn optimal_arbitrary_proxy_design_satisfies_load_and_bound() {
        let design = optimal_arbitrary_cao_crc_proxy_design_v1(10, 3, 1.0, 0.1, 10_000).unwrap();
        assert!(design.optimal_long_delay_probability > 0.0);
        assert!(design.optimal_long_delay_probability <= 0.3);
        assert!(design.tracking_load <= 3.0 + 1.0e-8);
        assert!(design.risk_gap_bound.is_finite());
        assert!((design.probabilities.iter().sum::<f64>() - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn proxy_survival_values_and_tracking_load_match_the_pmf() {
        let policy = base_policy();
        assert!((proxy_probability(&policy, 0) - 0.5).abs() < 1.0e-12);
        assert!((proxy_probability(&policy, 1) - 0.5).abs() < 1.0e-12);
        assert!((proxy_probability(&policy, 2) - 0.5).abs() < 1.0e-12);
        assert!((tracking_load(&policy) - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn input_threshold_must_match_interval_threshold() {
        let mut engine = engine();
        let wrong = round_input(
            0,
            "s0",
            "e0",
            AciPredictionIntervalV1::bounded(0.0, 1.0),
            0.55,
        );
        assert!(process_cao_crc_round_v1(&mut engine, &wrong, &[]).is_err());
        assert_eq!(engine.current_round, None);
    }

    #[test]
    fn round_sequence_and_sample_identity_are_enforced() {
        let mut engine = engine();
        let first = round_input(0, "s0", "e0", AciPredictionIntervalV1::bounded(0.0, 1.0), 0.5);
        process_cao_crc_round_v1(&mut engine, &first, &[]).unwrap();
        let skipped = round_input(2, "s2", "e2", AciPredictionIntervalV1::bounded(0.0, 1.0), engine.threshold);
        assert!(process_cao_crc_round_v1(&mut engine, &skipped, &[]).is_err());
    }
}
