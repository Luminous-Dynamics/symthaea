// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reference implementation of bounded-arbitrary-delay CA-OCRC.
//! Based on Wang & Simeone, arXiv:2611.17059.
//! Qualification failures are recorded separately and invalidate the inherited
//! theorem claim; UNKNOWN/BLOCKED feedback is never turned into zero loss.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const CA_OCRC_POLICY_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-policy.v1";
pub const CA_OCRC_ENGINE_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-engine.v1";
pub const CA_OCRC_ROUND_INPUT_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-round-input.v1";
pub const CA_OCRC_FEEDBACK_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-feedback.v1";
pub const CA_OCRC_ROUND_RECEIPT_SCHEMA_V1: &str = "luminous.semi-bench.ca-ocrc-round-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcPredictionSetKindV1 { NonTrivial, Empty, Full }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcEvidenceDispositionV1 { Qualified, Unknown, Blocked }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaOcrcFeedbackDispositionV1 { Applied, ExcludedUnknown, ExcludedBlocked, NotTracked }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub capacity: usize,
    pub maximum_feedback_delay: u64,
    pub target_risk: f64,
    pub learning_rate: f64,
    pub score_bound: f64,
    /// F(a)=P(proxy_delay >= a), indexed by age a=0..maximum_feedback_delay.
    pub proxy_delay_survival: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcTrackingTicketV1 {
    pub round_id: u64,
    pub proxy_delay: u64,
    pub admission_round: u64,
    pub policy_sha256: String,
    pub model_artifact_sha256: String,
    pub score_policy_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcFeedbackArrivalV1 {
    pub schema: String,
    pub origin_round_id: u64,
    pub delivered_round_id: u64,
    pub actual_delay: u64,
    /// Only a tracked and K1-qualified outcome may expose a loss to the learner.
    pub loss: Option<f64>,
    /// None means the evidence state is not visible to the learner.
    pub evidence_disposition: Option<CaOcrcEvidenceDispositionV1>,
    pub evidence_receipt_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcRoundInputV1 {
    pub schema: String,
    pub round_id: u64,
    pub prediction_set_kind: CaOcrcPredictionSetKindV1,
    /// Required for NonTrivial and supplied by caller for deterministic replay.
    pub proxy_delay_uniform_draw: Option<f64>,
    pub model_artifact_sha256: String,
    pub score_policy_sha256: String,
    pub feedback_arrivals: Vec<CaOcrcFeedbackArrivalV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcFeedbackReceiptV1 {
    pub origin_round_id: u64,
    pub actual_delay: u64,
    pub disposition: CaOcrcFeedbackDispositionV1,
    pub observed_loss: Option<f64>,
    pub inverse_proxy_probability: Option<f64>,
    pub threshold_contribution: f64,
    pub evidence_receipt_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcRoundReceiptV1 {
    pub schema: String,
    pub policy_id: String,
    pub policy_sha256: String,
    pub round_id: u64,
    pub threshold_before: f64,
    pub threshold_after: f64,
    pub prediction_set_kind: CaOcrcPredictionSetKindV1,
    pub known_trivial_loss: Option<f64>,
    pub proxy_delay_uniform_draw: Option<f64>,
    /// -1 means do not track; otherwise the sampled proxy delay.
    pub sampled_proxy_delay: i64,
    pub admitted_current_round: bool,
    pub evicted_round_ids: Vec<u64>,
    pub capacity_occupancy_after: usize,
    pub feedback: Vec<CaOcrcFeedbackReceiptV1>,
    pub trivial_loss_threshold_contribution: f64,
    pub feedback_threshold_contribution: f64,
    pub tracking_load: f64,
    pub theoretical_risk_bound: Option<f64>,
    pub theorem_assumptions_intact: bool,
    pub input_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcEngineV1 {
    pub schema: String,
    pub policy: CaOcrcPolicyV1,
    pub policy_sha256: String,
    pub round_index: u64,
    pub threshold: f64,
    pub tracking_set: BTreeMap<u64, CaOcrcTrackingTicketV1>,
    pub seen_feedback_round_ids: BTreeSet<u64>,
    pub accepted_feedback_count: u64,
    pub lost_untracked_feedback_count: u64,
    pub excluded_unknown_feedback_count: u64,
    pub excluded_blocked_feedback_count: u64,
    pub proxy_expired_count: u64,
    pub capacity_denied_count: u64,
    pub theorem_assumptions_intact: bool,
    pub last_receipt_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaOcrcRiskBoundV1 {
    pub horizon: u64,
    pub first_term: f64,
    pub feedback_term: f64,
    pub capacity_term: f64,
    pub total: f64,
    pub tracking_load: f64,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct CaOcrcErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
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

fn receipt_digest(receipt: &CaOcrcRoundReceiptV1) -> Result<String, CaOcrcErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn tracking_load(policy: &CaOcrcPolicyV1) -> f64 {
    policy.proxy_delay_survival.iter().skip(1).sum()
}

fn validate_policy(policy: &CaOcrcPolicyV1) -> Result<(), CaOcrcErrorV1> {
    if policy.schema != CA_OCRC_POLICY_SCHEMA_V1 || policy.policy_id.trim().is_empty() {
        return Err(CaOcrcErrorV1("policy schema or identity is invalid".into()));
    }
    if policy.capacity == 0 {
        return Err(CaOcrcErrorV1("capacity must be at least one".into()));
    }
    if !policy.target_risk.is_finite() || !(0.0..1.0).contains(&policy.target_risk) {
        return Err(CaOcrcErrorV1("target risk must be finite and in (0,1)".into()));
    }
    if !policy.learning_rate.is_finite() || policy.learning_rate <= 0.0 {
        return Err(CaOcrcErrorV1("learning rate must be finite and positive".into()));
    }
    if !policy.score_bound.is_finite() || policy.score_bound <= 0.0 {
        return Err(CaOcrcErrorV1("score bound must be finite and positive".into()));
    }
    let expected = usize::try_from(policy.maximum_feedback_delay)
        .ok()
        .and_then(|x| x.checked_add(1))
        .ok_or_else(|| CaOcrcErrorV1("maximum delay does not fit usize".into()))?;
    if policy.proxy_delay_survival.len() != expected {
        return Err(CaOcrcErrorV1("survival table length must equal d_max + 1".into()));
    }
    for (index, value) in policy.proxy_delay_survival.iter().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(value) {
            return Err(CaOcrcErrorV1(format!("F({index}) must be in [0,1]")));
        }
        if index > 0 && *value > policy.proxy_delay_survival[index - 1] {
            return Err(CaOcrcErrorV1("survival table must be non-increasing".into()));
        }
    }
    let load = tracking_load(policy);
    if load > policy.capacity as f64 + 1.0e-12 {
        return Err(CaOcrcErrorV1("tracking-load underload condition ell_F <= K is violated".into()));
    }
    if policy.proxy_delay_survival.last().copied().unwrap_or(0.0) <= 0.0 {
        return Err(CaOcrcErrorV1("F(d_max) must be positive for the risk bound".into()));
    }
    Ok(())
}

/// Samples proxy delay from the exact discrete distribution implied by F.
/// P(dbar=-1)=1-F(0); P(dbar=a)=F(a)-F(a+1); P(dbar=d_max)=F(d_max).
pub fn sample_ca_ocrc_proxy_delay_v1(
    policy: &CaOcrcPolicyV1,
    uniform_draw: f64,
) -> Result<i64, CaOcrcErrorV1> {
    validate_policy(policy)?;
    if !uniform_draw.is_finite() || !(0.0..1.0).contains(&uniform_draw) {
        return Err(CaOcrcErrorV1("uniform draw must be finite and in [0,1)".into()));
    }
    let mut cumulative = 1.0 - policy.proxy_delay_survival[0];
    if uniform_draw < cumulative {
        return Ok(-1);
    }
    for age in 0..policy.maximum_feedback_delay {
        cumulative += policy.proxy_delay_survival[age as usize]
            - policy.proxy_delay_survival[(age + 1) as usize];
        if uniform_draw < cumulative {
            return Ok(age as i64);
        }
    }
    Ok(policy.maximum_feedback_delay as i64)
}

/// Equation (25) from the arbitrary-bounded-delay theorem.
pub fn ca_ocrc_risk_bound_v1(
    policy: &CaOcrcPolicyV1,
    horizon: u64,
) -> Result<Option<CaOcrcRiskBoundV1>, CaOcrcErrorV1> {
    validate_policy(policy)?;
    if horizon == 0 {
        return Ok(None);
    }
    let load = tracking_load(policy);
    let ratio = load / policy.capacity as f64;
    let capacity_term = if load == 0.0 {
        0.0
    } else {
        let h = ratio - 1.0 - ratio.ln();
        (-((policy.capacity as f64) * h)).exp()
    };
    let f_max = *policy.proxy_delay_survival.last().unwrap();
    let first = policy.score_bound / (policy.learning_rate * horizon as f64);
    let feedback = (policy.capacity as f64 + 1.0) / (f_max * horizon as f64);
    let total = first + feedback + capacity_term;
    if !first.is_finite() || !feedback.is_finite() || !capacity_term.is_finite() || !total.is_finite() {
        return Err(CaOcrcErrorV1("risk bound contains a non-finite term".into()));
    }
    Ok(Some(CaOcrcRiskBoundV1 {
        horizon, first_term: first, feedback_term: feedback,
        capacity_term, total, tracking_load: load,
    }))
}

pub fn create_ca_ocrc_engine_v1(
    policy: CaOcrcPolicyV1,
    initial_threshold: f64,
) -> Result<CaOcrcEngineV1, CaOcrcErrorV1> {
    validate_policy(&policy)?;
    if !initial_threshold.is_finite() || !(0.0..=policy.score_bound).contains(&initial_threshold) {
        return Err(CaOcrcErrorV1("initial threshold must lie in [0,B]".into()));
    }
    let digest = policy_digest(&policy)?;
    Ok(CaOcrcEngineV1 {
        schema: CA_OCRC_ENGINE_SCHEMA_V1.into(),
        policy, policy_sha256: digest, round_index: 0, threshold: initial_threshold,
        tracking_set: BTreeMap::new(), seen_feedback_round_ids: BTreeSet::new(),
        accepted_feedback_count: 0, lost_untracked_feedback_count: 0,
        excluded_unknown_feedback_count: 0, excluded_blocked_feedback_count: 0,
        proxy_expired_count: 0, capacity_denied_count: 0,
        theorem_assumptions_intact: true, last_receipt_sha256: None,
    })
}

fn validate_engine_state(engine: &CaOcrcEngineV1) -> Result<(), CaOcrcErrorV1> {
    if !engine.threshold.is_finite() {
        return Err(CaOcrcErrorV1("engine threshold is non-finite".into()));
    }
    if engine.tracking_set.len() > engine.policy.capacity {
        return Err(CaOcrcErrorV1("tracking set exceeds configured capacity".into()));
    }
    for (round_id, ticket) in &engine.tracking_set {
        if *round_id != ticket.round_id
            || ticket.round_id == 0
            || ticket.round_id > engine.round_index
            || ticket.proxy_delay > engine.policy.maximum_feedback_delay
            || ticket.policy_sha256 != engine.policy_sha256
            || engine.policy.proxy_delay_survival[ticket.proxy_delay as usize] <= 0.0
            || !valid_sha256(&ticket.model_artifact_sha256)
            || !valid_sha256(&ticket.score_policy_sha256)
        {
            return Err(CaOcrcErrorV1("tracking ticket violates engine invariants".into()));
        }
    }
    Ok(())
}

fn set_kind_for_threshold(threshold: f64, bound: f64) -> CaOcrcPredictionSetKindV1 {
    if threshold < 0.0 { CaOcrcPredictionSetKindV1::Empty }
    else if threshold >= bound { CaOcrcPredictionSetKindV1::Full }
    else { CaOcrcPredictionSetKindV1::NonTrivial }
}

fn make_receipt(
    engine: &CaOcrcEngineV1,
    input: &CaOcrcRoundInputV1,
    input_sha256: String,
    threshold_before: f64,
    known_loss: Option<f64>,
    draw: Option<f64>,
    proxy_delay: i64,
    admitted: bool,
    evicted: Vec<u64>,
    feedback: Vec<CaOcrcFeedbackReceiptV1>,
    trivial_term: f64,
    feedback_term: f64,
) -> Result<CaOcrcRoundReceiptV1, CaOcrcErrorV1> {
    let mut receipt = CaOcrcRoundReceiptV1 {
        schema: CA_OCRC_ROUND_RECEIPT_SCHEMA_V1.into(),
        policy_id: engine.policy.policy_id.clone(),
        policy_sha256: engine.policy_sha256.clone(),
        round_id: input.round_id,
        threshold_before,
        threshold_after: engine.threshold,
        prediction_set_kind: input.prediction_set_kind,
        known_trivial_loss: known_loss,
        proxy_delay_uniform_draw: draw,
        sampled_proxy_delay: proxy_delay,
        admitted_current_round: admitted,
        evicted_round_ids: evicted,
        capacity_occupancy_after: engine.tracking_set.len(),
        feedback,
        trivial_loss_threshold_contribution: trivial_term,
        feedback_threshold_contribution: feedback_term,
        tracking_load: tracking_load(&engine.policy),
        theoretical_risk_bound: if engine.theorem_assumptions_intact {
            ca_ocrc_risk_bound_v1(&engine.policy, input.round_id)?.map(|bound| bound.total)
        } else {
            None
        },
        theorem_assumptions_intact: engine.theorem_assumptions_intact,
        input_sha256,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_digest(&receipt)?;
    Ok(receipt)
}

/// One global round, with the same order as Algorithm 1: expire, sample/admit,
/// observe the currently tracked arrivals, then update the threshold.
pub fn advance_ca_ocrc_round_v1(
    engine: &mut CaOcrcEngineV1,
    input: &CaOcrcRoundInputV1,
) -> Result<CaOcrcRoundReceiptV1, CaOcrcErrorV1> {
    if engine.schema != CA_OCRC_ENGINE_SCHEMA_V1
        || policy_digest(&engine.policy)? != engine.policy_sha256
    {
        return Err(CaOcrcErrorV1("engine or policy digest is invalid".into()));
    }
    validate_policy(&engine.policy)?;
    validate_engine_state(engine)?;
    if input.schema != CA_OCRC_ROUND_INPUT_SCHEMA_V1 {
        return Err(CaOcrcErrorV1("round input schema is unsupported".into()));
    }
    let expected_round = engine.round_index.checked_add(1)
        .ok_or_else(|| CaOcrcErrorV1("round index overflow".into()))?;
    if input.round_id != expected_round {
        return Err(CaOcrcErrorV1(format!("expected round {expected_round}, got {}", input.round_id)));
    }
    if !valid_sha256(&input.model_artifact_sha256) || !valid_sha256(&input.score_policy_sha256) {
        return Err(CaOcrcErrorV1("model or score-policy digest is malformed".into()));
    }
    let expected_kind = set_kind_for_threshold(engine.threshold, engine.policy.score_bound);
    if input.prediction_set_kind != expected_kind {
        return Err(CaOcrcErrorV1("prediction-set kind disagrees with current threshold".into()));
    }
    if expected_kind == CaOcrcPredictionSetKindV1::NonTrivial {
        let draw = input.proxy_delay_uniform_draw.ok_or_else(|| CaOcrcErrorV1(
            "nontrivial set requires an explicit uniform proxy-delay draw".into()
        ))?;
        if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
            return Err(CaOcrcErrorV1("proxy-delay draw must be in [0,1)".into()));
        }
    } else if input.proxy_delay_uniform_draw.is_some() {
        return Err(CaOcrcErrorV1("trivial set must not consume a proxy-delay draw".into()));
    }

    for arrival in &input.feedback_arrivals {
        if arrival.schema != CA_OCRC_FEEDBACK_SCHEMA_V1
            || arrival.origin_round_id == 0
            || arrival.delivered_round_id != input.round_id
            || arrival.origin_round_id > input.round_id
            || arrival.actual_delay != input.round_id - arrival.origin_round_id
            || arrival.actual_delay > engine.policy.maximum_feedback_delay
        {
            return Err(CaOcrcErrorV1("feedback schema, round, or bounded delay is invalid".into()));
        }
        if let Some(digest) = &arrival.evidence_receipt_sha256
            && !valid_sha256(digest)
        {
            return Err(CaOcrcErrorV1("evidence receipt digest is malformed".into()));
        }
    }

    // Work on a clone so every failure leaves the live capacity/threshold state unchanged.
    let mut next = engine.clone();
    let threshold_before = next.threshold;
    let mut canonical_input = input.clone();
    canonical_input.feedback_arrivals.sort_by_key(|arrival| arrival.origin_round_id);
    let input_sha256 = json_sha256(&canonical_input)?;

    // Eq. (20): evict iff i + proxy_delay_i < t; equality remains trackable this round.
    let evicted: Vec<u64> = next.tracking_set.iter()
        .filter_map(|(id, ticket)| (id.saturating_add(ticket.proxy_delay) < input.round_id).then_some(*id))
        .collect();
    for id in &evicted { next.tracking_set.remove(id); }
    next.proxy_expired_count = next.proxy_expired_count.checked_add(evicted.len() as u64)
        .ok_or_else(|| CaOcrcErrorV1("proxy-expired counter overflow".into()))?;

    let (draw, proxy_delay, admitted, known_loss, trivial_term) = match expected_kind {
        CaOcrcPredictionSetKindV1::Empty => {
            let loss = 1.0;
            (None, -1, false, Some(loss), next.policy.learning_rate * (loss - next.policy.target_risk))
        }
        CaOcrcPredictionSetKindV1::Full => {
            let loss = 0.0;
            (None, -1, false, Some(loss), next.policy.learning_rate * (loss - next.policy.target_risk))
        }
        CaOcrcPredictionSetKindV1::NonTrivial => {
            let uniform = input.proxy_delay_uniform_draw.unwrap();
            let sampled = sample_ca_ocrc_proxy_delay_v1(&next.policy, uniform)?;
            let mut admitted = false;
            if sampled >= 0 {
                if next.tracking_set.len() < next.policy.capacity {
                    next.tracking_set.insert(input.round_id, CaOcrcTrackingTicketV1 {
                        round_id: input.round_id,
                        proxy_delay: sampled as u64,
                        admission_round: input.round_id,
                        policy_sha256: next.policy_sha256.clone(),
                        model_artifact_sha256: input.model_artifact_sha256.clone(),
                        score_policy_sha256: input.score_policy_sha256.clone(),
                    });
                    admitted = true;
                } else {
                    next.capacity_denied_count = next.capacity_denied_count.checked_add(1)
                        .ok_or_else(|| CaOcrcErrorV1("capacity-denied counter overflow".into()))?;
                }
            }
            (Some(uniform), sampled, admitted, None, 0.0)
        }
    };

    let arrivals = canonical_input.feedback_arrivals;
    let mut feedback_receipts = Vec::with_capacity(arrivals.len());
    let mut feedback_term = 0.0;
    let mut seen_this_round = BTreeSet::new();

    for arrival in arrivals {
        if !seen_this_round.insert(arrival.origin_round_id)
            || next.seen_feedback_round_ids.contains(&arrival.origin_round_id)
        {
            return Err(CaOcrcErrorV1("duplicate feedback arrival".into()));
        }

        let Some(ticket) = next.tracking_set.get(&arrival.origin_round_id).cloned() else {
            if arrival.loss.is_some()
                || arrival.evidence_disposition.is_some()
                || arrival.evidence_receipt_sha256.is_some()
            {
                return Err(CaOcrcErrorV1(
                    "loss/evidence must not be exposed for a round that was not tracked".into()
                ));
            }
            next.lost_untracked_feedback_count = next.lost_untracked_feedback_count.checked_add(1)
                .ok_or_else(|| CaOcrcErrorV1("untracked feedback counter overflow".into()))?;
            next.seen_feedback_round_ids.insert(arrival.origin_round_id);
            feedback_receipts.push(CaOcrcFeedbackReceiptV1 {
                origin_round_id: arrival.origin_round_id,
                actual_delay: arrival.actual_delay,
                disposition: CaOcrcFeedbackDispositionV1::NotTracked,
                observed_loss: None,
                inverse_proxy_probability: None,
                threshold_contribution: 0.0,
                evidence_receipt_sha256: None,
            });
            continue;
        };

        if arrival.actual_delay > ticket.proxy_delay {
            return Err(CaOcrcErrorV1("tracked feedback exceeded its proxy retention horizon".into()));
        }
        next.tracking_set.remove(&arrival.origin_round_id);
        next.seen_feedback_round_ids.insert(arrival.origin_round_id);

        let evidence = arrival.evidence_disposition.ok_or_else(|| CaOcrcErrorV1(
            "tracked feedback requires an explicit evidence disposition".into()
        ))?;
        let evidence_digest = arrival.evidence_receipt_sha256.as_ref().ok_or_else(|| {
            CaOcrcErrorV1("tracked feedback requires a K1 evidence receipt digest".into())
        })?;
        if !valid_sha256(evidence_digest) {
            return Err(CaOcrcErrorV1("tracked K1 evidence receipt digest is malformed".into()));
        }
        let (disposition, observed_loss, inverse_weight, contribution) = match evidence {
            CaOcrcEvidenceDispositionV1::Qualified => {
                let loss = arrival.loss.ok_or_else(|| CaOcrcErrorV1(
                    "qualified feedback must carry a loss observation".into()
                ))?;
                if !loss.is_finite() || !(0.0..=1.0).contains(&loss) {
                    return Err(CaOcrcErrorV1("loss must be finite and in [0,1]".into()));
                }
                let survival = next.policy.proxy_delay_survival[arrival.actual_delay as usize];
                if survival <= 0.0 {
                    return Err(CaOcrcErrorV1("F(actual_delay) must be positive for tracked feedback".into()));
                }
                let weight = 1.0 / survival;
                let contribution = next.policy.learning_rate
                    * (loss - next.policy.target_risk) * weight;
                if !weight.is_finite() || !contribution.is_finite() {
                    return Err(CaOcrcErrorV1("inverse-probability feedback term is non-finite".into()));
                }
                next.accepted_feedback_count = next.accepted_feedback_count.checked_add(1)
                    .ok_or_else(|| CaOcrcErrorV1("accepted feedback counter overflow".into()))?;
                (CaOcrcFeedbackDispositionV1::Applied, Some(loss), Some(weight), contribution)
            }
            CaOcrcEvidenceDispositionV1::Unknown => {
                if arrival.loss.is_some() {
                    return Err(CaOcrcErrorV1("UNKNOWN feedback must not reveal a loss".into()));
                }
                next.excluded_unknown_feedback_count = next.excluded_unknown_feedback_count.checked_add(1)
                    .ok_or_else(|| CaOcrcErrorV1("UNKNOWN counter overflow".into()))?;
                next.theorem_assumptions_intact = false;
                (CaOcrcFeedbackDispositionV1::ExcludedUnknown, None, None, 0.0)
            }
            CaOcrcEvidenceDispositionV1::Blocked => {
                if arrival.loss.is_some() {
                    return Err(CaOcrcErrorV1("BLOCKED feedback must not reveal a loss".into()));
                }
                next.excluded_blocked_feedback_count = next.excluded_blocked_feedback_count.checked_add(1)
                    .ok_or_else(|| CaOcrcErrorV1("BLOCKED counter overflow".into()))?;
                next.theorem_assumptions_intact = false;
                (CaOcrcFeedbackDispositionV1::ExcludedBlocked, None, None, 0.0)
            }
        };

        feedback_term += contribution;
        if !feedback_term.is_finite() {
            return Err(CaOcrcErrorV1("sum of feedback contributions is non-finite".into()));
        }
        feedback_receipts.push(CaOcrcFeedbackReceiptV1 {
            origin_round_id: arrival.origin_round_id,
            actual_delay: arrival.actual_delay,
            disposition,
            observed_loss,
            inverse_proxy_probability: inverse_weight,
            threshold_contribution: contribution,
            evidence_receipt_sha256: arrival.evidence_receipt_sha256,
        });
    }

    let next_threshold = next.threshold + trivial_term + feedback_term;
    if !next_threshold.is_finite() {
        return Err(CaOcrcErrorV1("candidate threshold is non-finite".into()));
    }
    next.threshold = next_threshold;
    next.round_index = input.round_id;

    let mut receipt = make_receipt(
        &next, input, input_sha256, threshold_before, known_loss, draw, proxy_delay,
        admitted, evicted, feedback_receipts, trivial_term, feedback_term,
    )?;
    receipt.theorem_assumptions_intact = next.theorem_assumptions_intact;
    receipt.receipt_sha256 = receipt_digest(&receipt)?;
    next.last_receipt_sha256 = Some(receipt.receipt_sha256.clone());
    *engine = next;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String { std::iter::repeat_n(ch, 64).collect() }

    fn policy(k: usize, survival: Vec<f64>) -> CaOcrcPolicyV1 {
        CaOcrcPolicyV1 {
            schema: CA_OCRC_POLICY_SCHEMA_V1.into(),
            policy_id: "ca-ocrc-bounded-arbitrary-v1".into(),
            capacity: k,
            maximum_feedback_delay: (survival.len() - 1) as u64,
            target_risk: 0.10,
            learning_rate: 0.05,
            score_bound: 1.0,
            proxy_delay_survival: survival,
        }
    }

    fn engine(k: usize, survival: Vec<f64>, threshold: f64) -> CaOcrcEngineV1 {
        create_ca_ocrc_engine_v1(policy(k, survival), threshold).unwrap()
    }

    fn input(round: u64, kind: CaOcrcPredictionSetKindV1, draw: Option<f64>,
        arrivals: Vec<CaOcrcFeedbackArrivalV1>) -> CaOcrcRoundInputV1 {
        CaOcrcRoundInputV1 {
            schema: CA_OCRC_ROUND_INPUT_SCHEMA_V1.into(),
            round_id: round,
            prediction_set_kind: kind,
            proxy_delay_uniform_draw: draw,
            model_artifact_sha256: digest('a'),
            score_policy_sha256: digest('b'),
            feedback_arrivals: arrivals,
        }
    }

    fn feedback(origin: u64, delivered: u64, evidence: Option<CaOcrcEvidenceDispositionV1>,
        loss: Option<f64>) -> CaOcrcFeedbackArrivalV1 {
        CaOcrcFeedbackArrivalV1 {
            schema: CA_OCRC_FEEDBACK_SCHEMA_V1.into(),
            origin_round_id: origin,
            delivered_round_id: delivered,
            actual_delay: delivered - origin,
            loss,
            evidence_disposition: evidence,
            evidence_receipt_sha256: evidence.map(|_| digest('c')),
        }
    }

    #[test]
    fn uniform_draw_matches_survival_distribution_support() {
        let p = policy(1, vec![0.5, 0.5, 0.5]);
        assert_eq!(sample_ca_ocrc_proxy_delay_v1(&p, 0.1).unwrap(), -1);
        assert_eq!(sample_ca_ocrc_proxy_delay_v1(&p, 0.6).unwrap(), 2);
    }

    #[test]
    fn capacity_is_never_exceeded_and_denied_admission_is_recorded() {
        let mut e = engine(1, vec![0.5, 0.5, 0.5], 0.5);
        let a = advance_ca_ocrc_round_v1(&mut e,
            &input(1, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.75), vec![])).unwrap();
        assert!(a.admitted_current_round);
        let b = advance_ca_ocrc_round_v1(&mut e,
            &input(2, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.75), vec![])).unwrap();
        assert!(!b.admitted_current_round);
        assert_eq!(b.capacity_occupancy_after, 1);
        assert_eq!(e.capacity_denied_count, 1);
    }

    #[test]
    fn eviction_precedes_admission_and_frees_the_slot_in_same_round() {
        let mut e = engine(1, vec![1.0, 0.5, 0.0], 0.5);
        advance_ca_ocrc_round_v1(&mut e,
            &input(1, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.75), vec![])).unwrap();
        advance_ca_ocrc_round_v1(&mut e,
            &input(2, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.75), vec![])).unwrap();
        let r3 = advance_ca_ocrc_round_v1(&mut e,
            &input(3, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.75), vec![])).unwrap();
        assert_eq!(r3.evicted_round_ids, vec![1]);
        assert!(r3.admitted_current_round);
        assert_eq!(r3.capacity_occupancy_after, 1);
    }

    #[test]
    fn feedback_at_exact_proxy_horizon_is_received_before_next_round_eviction() {
        let mut e = engine(1, vec![1.0, 0.5, 0.25], 0.5);
        advance_ca_ocrc_round_v1(&mut e,
            &input(1, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.9), vec![])).unwrap();
        advance_ca_ocrc_round_v1(&mut e,
            &input(2, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.9), vec![])).unwrap();
        let r3 = advance_ca_ocrc_round_v1(&mut e,
            &input(3, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.9),
                vec![feedback(1, 3, Some(CaOcrcEvidenceDispositionV1::Qualified), Some(1.0))])).unwrap();
        assert_eq!(r3.feedback[0].disposition, CaOcrcFeedbackDispositionV1::Applied);
    }

    #[test]
    fn capacity_at_least_dmax_plus_one_reproduces_unlimited_delayed_ocrc() {
        let mut e = engine(3, vec![1.0, 1.0, 1.0], 0.5);
        let one = advance_ca_ocrc_round_v1(
            &mut e,
            &input(1, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.9), vec![]),
        )
        .unwrap();
        assert!(one.admitted_current_round);
        let two = advance_ca_ocrc_round_v1(
            &mut e,
            &input(2, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.9), vec![]),
        )
        .unwrap();
        assert!(two.admitted_current_round);
        let three = advance_ca_ocrc_round_v1(
            &mut e,
            &input(
                3,
                CaOcrcPredictionSetKindV1::NonTrivial,
                Some(0.9),
                vec![
                    feedback(1, 3, Some(CaOcrcEvidenceDispositionV1::Qualified), Some(1.0)),
                    feedback(2, 3, Some(CaOcrcEvidenceDispositionV1::Qualified), Some(0.0)),
                ],
            ),
        )
        .unwrap();
        assert!(three.admitted_current_round);
        assert!((three.feedback_threshold_contribution - 0.04).abs() < 1.0e-12);
        assert!((three.threshold_after - 0.54).abs() < 1.0e-12);
        assert_eq!(three.capacity_occupancy_after, 1);
    }

    #[test]
    fn inverse_probability_weight_uses_f_of_actual_delay() {
        let mut e = engine(2, vec![1.0, 0.5, 0.25], 0.5);
        advance_ca_ocrc_round_v1(&mut e,
            &input(1, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.9), vec![])).unwrap();
        let r2 = advance_ca_ocrc_round_v1(&mut e,
            &input(2, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.9),
                vec![feedback(1, 2, Some(CaOcrcEvidenceDispositionV1::Qualified), Some(1.0))])).unwrap();
        assert_eq!(r2.feedback[0].inverse_proxy_probability, Some(2.0));
        assert!((r2.feedback_threshold_contribution - 0.05 * 0.9 * 2.0).abs() < 1e-12);
    }

    #[test]
    fn unknown_and_blocked_evidence_never_enter_threshold_update() {
        for disposition in [CaOcrcEvidenceDispositionV1::Unknown, CaOcrcEvidenceDispositionV1::Blocked] {
            let mut e = engine(1, vec![1.0, 1.0], 0.5);
            advance_ca_ocrc_round_v1(&mut e,
                &input(1, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.5), vec![])).unwrap();
            let before = e.threshold;
            let r = advance_ca_ocrc_round_v1(&mut e,
                &input(2, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.5),
                    vec![feedback(1, 2, Some(disposition), None)])).unwrap();
            assert_eq!(r.feedback[0].threshold_contribution, 0.0);
            assert_eq!(e.threshold, before);
            assert!(!e.theorem_assumptions_intact);
        }
    }

    #[test]
    fn trivial_empty_and_full_sets_use_known_losses_without_tracking() {
        let mut empty = engine(1, vec![1.0, 1.0], 0.0);
        empty.threshold = -0.1;
        let r = advance_ca_ocrc_round_v1(&mut empty,
            &input(1, CaOcrcPredictionSetKindV1::Empty, None, vec![])).unwrap();
        assert_eq!(r.known_trivial_loss, Some(1.0));
        assert_eq!(r.capacity_occupancy_after, 0);

        let mut full = engine(1, vec![1.0, 1.0], 0.0);
        full.threshold = 1.0;
        let r = advance_ca_ocrc_round_v1(&mut full,
            &input(1, CaOcrcPredictionSetKindV1::Full, None, vec![])).unwrap();
        assert_eq!(r.known_trivial_loss, Some(0.0));
        assert_eq!(r.capacity_occupancy_after, 0);
    }

    #[test]
    fn feedback_after_proxy_expiry_is_not_tracked() {
        let mut e = engine(1, vec![1.0, 0.5, 0.0], 0.5);
        advance_ca_ocrc_round_v1(&mut e,
            &input(1, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.75), vec![])).unwrap();
        advance_ca_ocrc_round_v1(&mut e,
            &input(2, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.75), vec![])).unwrap();
        let r3 = advance_ca_ocrc_round_v1(&mut e,
            &input(3, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.75),
                vec![feedback(1, 3, None, None)])).unwrap();
        assert_eq!(r3.feedback[0].disposition, CaOcrcFeedbackDispositionV1::NotTracked);
        assert_eq!(r3.feedback_threshold_contribution, 0.0);
    }

    #[test]
    fn invalid_survival_functions_and_underload_are_rejected() {
        assert!(create_ca_ocrc_engine_v1(policy(1, vec![0.5, 0.7]), 0.5).is_err());
        assert!(create_ca_ocrc_engine_v1(policy(1, vec![0.0, 0.0]), 0.5).is_err());
        assert!(create_ca_ocrc_engine_v1(policy(1, vec![1.0, 1.0, 1.0]), 0.5).is_err());
    }

    #[test]
    fn weighted_update_overflow_is_transactional() {
        let mut p = policy(1, vec![1.0, 0.01]);
        p.learning_rate = 1.0e308;
        let mut e = create_ca_ocrc_engine_v1(p, 0.5).unwrap();
        let before = e.clone();
        // Proxy delay 1; a loss of 1 generates a non-finite weighted correction.
        let first = advance_ca_ocrc_round_v1(&mut e,
            &input(1, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.999), vec![])).unwrap();
        assert!(first.admitted_current_round);
        let after_first = e.clone();
        let result = advance_ca_ocrc_round_v1(&mut e,
            &input(2, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.0),
                vec![feedback(1, 2, Some(CaOcrcEvidenceDispositionV1::Qualified), Some(1.0))]));
        assert!(result.is_err());
        assert_eq!(e, after_first);
        assert_ne!(before, e);
    }

    #[test]
    fn fixed_draw_replay_is_byte_deterministic() {
        fn run() -> (String, String) {
            let mut e = engine(2, vec![1.0, 0.5, 0.25], 0.5);
            let r1 = advance_ca_ocrc_round_v1(&mut e,
                &input(1, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.9), vec![])).unwrap();
            let r2 = advance_ca_ocrc_round_v1(&mut e,
                &input(2, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.9),
                    vec![feedback(1, 2, Some(CaOcrcEvidenceDispositionV1::Qualified), Some(0.0))])).unwrap();
            (json_sha256(&e).unwrap(), format!("{}:{}", r1.receipt_sha256, r2.receipt_sha256))
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn risk_bound_reports_each_term_for_valid_underloaded_policy() {
        let p = policy(2, vec![1.0, 0.5, 0.25]);
        let bound = ca_ocrc_risk_bound_v1(&p, 100).unwrap().unwrap();
        assert_eq!(bound.tracking_load, 0.75);
        assert!(bound.total >= bound.first_term);
        assert!(bound.total >= bound.feedback_term);
        assert!(bound.capacity_term.is_finite());
    }

    #[test]
    fn reordered_feedback_input_has_same_canonical_receipt() {
        fn run(reverse: bool) -> String {
            let mut e = engine(2, vec![1.0, 0.5, 0.25], 0.5);
            advance_ca_ocrc_round_v1(&mut e,
                &input(1, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.9), vec![])).unwrap();
            advance_ca_ocrc_round_v1(&mut e,
                &input(2, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.9), vec![])).unwrap();
            let mut events = vec![
                feedback(1, 3, Some(CaOcrcEvidenceDispositionV1::Qualified), Some(0.0)),
                feedback(2, 3, Some(CaOcrcEvidenceDispositionV1::Qualified), Some(1.0)),
            ];
            if reverse { events.reverse(); }
            advance_ca_ocrc_round_v1(&mut e,
                &input(3, CaOcrcPredictionSetKindV1::NonTrivial, Some(0.9), events)).unwrap()
                .receipt_sha256
        }
        assert_eq!(run(false), run(true));
    }
}
