// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Decision-value-aware retrieval selection.
//!
//! Two scoring modes are intentionally separate:
//! 1. EVSI uses a consumer-pinned, externally verified calibrated prior and
//!    per-state observation likelihoods.
//! 2. Robust value uses a probability-free support set and minimax regret.
//!
//! This module evaluates a consumer-owned finite decision model. It does not
//! infer probabilities, utilities, risk tolerances or decision requirements
//! from the candidate memory system. It never promotes a retrieval strategy;
//! promotion still requires the independent decision/outcome and held-out
//! counterfactual qualification gates in retrieval_policy_memory.rs.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

const PROBABILITY_SCALE: i128 = 1_000_000;
const MAX_STATES: usize = 128;
const MAX_ACTIONS: usize = 128;
const MAX_OBSERVATIONS: usize = 128;
const MAX_QUERY_OPTIONS: usize = 256;
const MAX_ID_BYTES: usize = 512;
const MAX_UTILITY_ABS: u64 = 1_000_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionValueRejectV1 {
    InvalidPolicy,
    InvalidOption,
    InvalidIdentifier,
    InvalidDigest,
    InvalidDimensions,
    InvalidProbability,
    InvalidObservationSupport,
    UtilityOutOfRange,
    BudgetExceeded,
    TooManyOptions,
    CalibrationReceiptRejected,
    ArithmeticOverflow,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionValuePolicyV1 {
    /// Digest of the consumer-owned DecisionNeed, never supplied by candidate memory.
    pub decision_need_sha256: String,
    /// Digest of the consumer-owned utility/loss profile.
    pub utility_profile_sha256: String,
    /// State IDs must be strictly sorted and unique.
    pub state_ids: Vec<String>,
    /// Action IDs must be strictly sorted and unique.
    pub action_ids: Vec<String>,
    /// Rows follow state_ids; columns follow action_ids.
    /// Larger values are better. Safety penalties belong in this utility table.
    pub utility_by_state_action: Vec<Vec<i64>>,
    pub maximum_tool_calls: u32,
    pub maximum_latency_ns: u64,
    pub maximum_context_tokens: u64,
    pub maximum_inspected_bytes: u64,
    /// Must be expressed in the same utility units as the decision table.
    pub maximum_acquisition_cost_utility_units: u64,
}

impl DecisionValuePolicyV1 {
    pub fn validate(&self) -> Result<(), DecisionValueRejectV1> {
        if !valid_sha256(&self.decision_need_sha256)
            || !valid_sha256(&self.utility_profile_sha256)
            || self.state_ids.is_empty()
            || self.state_ids.len() > MAX_STATES
            || self.action_ids.is_empty()
            || self.action_ids.len() > MAX_ACTIONS
            || self.maximum_tool_calls == 0
        {
            return Err(DecisionValueRejectV1::InvalidPolicy);
        }
        if !strict_sorted_unique(&self.state_ids)
            || !strict_sorted_unique(&self.action_ids)
            || self.state_ids.iter().chain(self.action_ids.iter())
                .any(|value| !valid_atom(value, MAX_ID_BYTES))
        {
            return Err(DecisionValueRejectV1::InvalidIdentifier);
        }
        if self.utility_by_state_action.len() != self.state_ids.len()
            || self.utility_by_state_action.iter()
                .any(|row| row.len() != self.action_ids.len())
        {
            return Err(DecisionValueRejectV1::InvalidDimensions);
        }
        if self.utility_by_state_action.iter().flatten()
            .any(|value| value.unsigned_abs() > MAX_UTILITY_ABS)
        {
            return Err(DecisionValueRejectV1::UtilityOutOfRange);
        }
        Ok(())
    }
}

/// Fixed-point calibrated prior. Each probability is in parts per million and
/// the complete prior must sum to 1,000,000 exactly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibratedDecisionPriorV1 {
    pub calibration_profile_sha256: String,
    pub calibration_receipt_sha256: String,
    pub valid_until_unix: u64,
    pub probability_ppm_by_state: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationReceiptRejectV1 {
    InvalidSignature,
    ProfileMismatch,
    DecisionNeedMismatch,
    ProbabilityModelMismatch,
    Expired,
    Revoked,
}

pub trait DecisionCalibrationReceiptVerifierV1 {
    /// Implementations must verify the signed calibration receipt, profile
    /// pin, DecisionNeed binding, exact probability-model digest and validity
    /// window using the caller's trusted current time. A digest-shaped string
    /// alone is not verification.
    fn verify(
        &self,
        decision_need_sha256: &str,
        calibration_profile_sha256: &str,
        calibration_receipt_sha256: &str,
        probability_model_sha256: &str,
        now_unix: u64,
    ) -> Result<(), CalibrationReceiptRejectV1>;
}

/// One candidate retrieval/verification operation, not a retrieved fact.
/// Likelihood probabilities are used only by EVSI. The separate Boolean
/// support matrix drives robust minimax evaluation and must be derived from
/// a conservative support analysis, not from a calibrated probability cutoff.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionRetrievalOptionV1 {
    pub query_option_id: String,
    pub action_template_sha256: String,
    /// Strictly sorted observation/result classes. Include an explicit
    /// "unknown"/"unavailable" class whenever that result is possible.
    pub observation_ids: Vec<String>,
    /// P(observation | state, this query) in parts per million.
    /// One row per state, one column per observation; each row sums to scale.
    pub likelihood_ppm_by_state: Vec<Vec<u32>>,
    /// Conservative possibility support, independent of probability values.
    /// One row per state, one column per observation. Each state needs at
    /// least one possible observation.
    pub possible_observation_by_state: Vec<Vec<bool>>,
    pub tool_calls: u32,
    pub estimated_latency_ns: u64,
    pub estimated_context_tokens: u64,
    pub estimated_inspected_bytes: u64,
    /// Explicit decision-utility cost, in the policy's utility units.
    pub acquisition_cost_utility_units: u64,
}

impl DecisionRetrievalOptionV1 {
    pub fn validate(
        &self,
        policy: &DecisionValuePolicyV1,
        require_calibrated_likelihoods: bool,
    ) -> Result<(), DecisionValueRejectV1> {
        if !valid_atom(&self.query_option_id, MAX_ID_BYTES)
            || !valid_sha256(&self.action_template_sha256)
        {
            return Err(DecisionValueRejectV1::InvalidIdentifier);
        }
        if self.observation_ids.is_empty()
            || self.observation_ids.len() > MAX_OBSERVATIONS
            || !strict_sorted_unique(&self.observation_ids)
            || self.observation_ids.iter().any(|value| !valid_atom(value, MAX_ID_BYTES))
        {
            return Err(DecisionValueRejectV1::InvalidIdentifier);
        }
        let states = policy.state_ids.len();
        let observations = self.observation_ids.len();
        if self.possible_observation_by_state.len() != states
            || self.possible_observation_by_state.iter().any(|row| row.len() != observations)
            || (require_calibrated_likelihoods
                && (self.likelihood_ppm_by_state.len() != states
                    || self.likelihood_ppm_by_state.iter().any(|row| row.len() != observations)))
            || (!self.likelihood_ppm_by_state.is_empty()
                && (self.likelihood_ppm_by_state.len() != states
                    || self.likelihood_ppm_by_state.iter().any(|row| row.len() != observations)))
        {
            return Err(DecisionValueRejectV1::InvalidDimensions);
        }
        for support in &self.possible_observation_by_state {
            if !support.iter().any(|possible| *possible) {
                return Err(DecisionValueRejectV1::InvalidObservationSupport);
            }
        }
        if require_calibrated_likelihoods || !self.likelihood_ppm_by_state.is_empty() {
            for likelihoods in &self.likelihood_ppm_by_state {
                let sum = likelihoods.iter().try_fold(0_u64, |total, value| {
                    if *value > 1_000_000 { return None; }
                    total.checked_add(u64::from(*value))
                }).ok_or(DecisionValueRejectV1::InvalidProbability)?;
                if sum != 1_000_000 {
                    return Err(DecisionValueRejectV1::InvalidProbability);
                }
            }
        }
        if self.tool_calls == 0 {
            return Err(DecisionValueRejectV1::InvalidOption);
        }
        if self.tool_calls > policy.maximum_tool_calls
            || self.estimated_latency_ns > policy.maximum_latency_ns
            || self.estimated_context_tokens > policy.maximum_context_tokens
            || self.estimated_inspected_bytes > policy.maximum_inspected_bytes
            || self.acquisition_cost_utility_units > policy.maximum_acquisition_cost_utility_units
        {
            return Err(DecisionValueRejectV1::BudgetExceeded);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedValueObservationChoiceV1 {
    pub observation_id: String,
    /// None means the observation has zero probability under the supplied model.
    pub selected_action_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedValueAssessmentV1 {
    pub query_option_id: String,
    pub baseline_action_id: String,
    pub selected_actions_by_observation: Vec<ExpectedValueObservationChoiceV1>,
    /// EVSI in utility units multiplied by 1,000,000 squared.
    pub gross_evsi_numerator_ppm2: i128,
    /// Acquisition cost in the same fixed-point units.
    pub acquisition_cost_numerator_ppm2: i128,
    /// Positive values support retrieval; zero/negative values say stop.
    pub expected_net_value_numerator_ppm2: i128,
    pub worth_retrieving: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RobustValueObservationChoiceV1 {
    pub observation_id: String,
    pub compatible_state_ids: Vec<String>,
    pub selected_action_id: Option<String>,
    pub worst_case_regret_utility_units: Option<i128>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RobustValueAssessmentV1 {
    pub query_option_id: String,
    pub baseline_minimax_regret_utility_units: i128,
    pub post_query_worst_case_regret_utility_units: i128,
    pub gross_regret_reduction_utility_units: i128,
    pub acquisition_cost_utility_units: i128,
    pub robust_net_value_utility_units: i128,
    pub selected_actions_by_observation: Vec<RobustValueObservationChoiceV1>,
    pub worth_retrieving: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedValueSelectionV1 {
    pub query_option_id: String,
    pub assessment: ExpectedValueAssessmentV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RobustValueSelectionV1 {
    pub query_option_id: String,
    pub assessment: RobustValueAssessmentV1,
}

/// Evaluate expected value of sample information (EVSI). The likelihood model
/// and prior are not inferred here: the caller must supply a current,
/// independently verified calibration receipt for this exact DecisionNeed.
pub fn evaluate_expected_value_v1(
    policy: &DecisionValuePolicyV1,
    prior: &CalibratedDecisionPriorV1,
    option: &DecisionRetrievalOptionV1,
    now_unix: u64,
    calibration_verifier: &impl DecisionCalibrationReceiptVerifierV1,
) -> Result<ExpectedValueAssessmentV1, DecisionValueRejectV1> {
    policy.validate()?;
    option.validate(policy, true)?;
    if !valid_sha256(&prior.calibration_profile_sha256)
        || !valid_sha256(&prior.calibration_receipt_sha256)
        || prior.valid_until_unix <= now_unix
        || prior.probability_ppm_by_state.len() != policy.state_ids.len()
    {
        return Err(DecisionValueRejectV1::InvalidCalibration);
    }
    let prior_sum = prior.probability_ppm_by_state.iter()
        .try_fold(0_u64, |total, value| {
            if *value > 1_000_000 { return None; }
            total.checked_add(u64::from(*value))
        }).ok_or(DecisionValueRejectV1::InvalidProbability)?;
    if prior_sum != 1_000_000 {
        return Err(DecisionValueRejectV1::InvalidProbability);
    }
    let probability_model_sha256 = probability_model_sha256(policy, prior, option);
    calibration_verifier.verify(
        &policy.decision_need_sha256,
        &prior.calibration_profile_sha256,
        &prior.calibration_receipt_sha256,
        &probability_model_sha256,
        now_unix,
    ).map_err(|_| DecisionValueRejectV1::CalibrationReceiptRejected)?;

    let prior_weights = prior.probability_ppm_by_state.iter()
        .map(|value| i128::from(*value)).collect::<Vec<_>>();
    let (baseline_action_index, baseline_weighted_utility) =
        best_expected_action(policy, &prior_weights)?;
    let mut after_utility_numerator = 0_i128;
    let mut selected_actions_by_observation = Vec::with_capacity(option.observation_ids.len());

    for observation_index in 0..option.observation_ids.len() {
        let posterior_joint_weights = prior.probability_ppm_by_state.iter()
            .zip(option.likelihood_ppm_by_state.iter())
            .map(|(prior, likelihoods)| {
                i128::from(*prior) * i128::from(likelihoods[observation_index])
            }).collect::<Vec<_>>();
        if posterior_joint_weights.iter().all(|weight| *weight == 0) {
            selected_actions_by_observation.push(ExpectedValueObservationChoiceV1 {
                observation_id: option.observation_ids[observation_index].clone(),
                selected_action_id: None,
            });
            continue;
        }
        let (action_index, weighted_utility) = best_expected_action(policy, &posterior_joint_weights)?;
        after_utility_numerator = after_utility_numerator.checked_add(weighted_utility)
            .ok_or(DecisionValueRejectV1::ArithmeticOverflow)?;
        selected_actions_by_observation.push(ExpectedValueObservationChoiceV1 {
            observation_id: option.observation_ids[observation_index].clone(),
            selected_action_id: Some(policy.action_ids[action_index].clone()),
        });
    }

    let baseline_scaled = baseline_weighted_utility.checked_mul(PROBABILITY_SCALE)
        .ok_or(DecisionValueRejectV1::ArithmeticOverflow)?;
    let gross_evsi_numerator_ppm2 = after_utility_numerator - baseline_scaled;
    // A negative EVSI signals an invalid/inconsistent observation model or
    // arithmetic/implementation defect. Never silently clamp it to zero.
    if gross_evsi_numerator_ppm2 < 0 {
        return Err(DecisionValueRejectV1::InvalidProbability);
    }
    let acquisition_cost_numerator_ppm2 =
        i128::from(option.acquisition_cost_utility_units)
            .checked_mul(PROBABILITY_SCALE * PROBABILITY_SCALE)
            .ok_or(DecisionValueRejectV1::ArithmeticOverflow)?;
    let expected_net_value_numerator_ppm2 =
        gross_evsi_numerator_ppm2 - acquisition_cost_numerator_ppm2;

    Ok(ExpectedValueAssessmentV1 {
        query_option_id: option.query_option_id.clone(),
        baseline_action_id: policy.action_ids[baseline_action_index].clone(),
        selected_actions_by_observation,
        gross_evsi_numerator_ppm2,
        acquisition_cost_numerator_ppm2,
        expected_net_value_numerator_ppm2,
        worth_retrieving: expected_net_value_numerator_ppm2 > 0,
    })
}

/// Choose the candidate query with the highest positive calibrated EVSI after
/// acquisition cost. No positive candidate means stop; no zero-value fallback.
pub fn choose_expected_value_query_v1(
    policy: &DecisionValuePolicyV1,
    prior: &CalibratedDecisionPriorV1,
    options: &[DecisionRetrievalOptionV1],
    now_unix: u64,
    calibration_verifier: &impl DecisionCalibrationReceiptVerifierV1,
) -> Result<Option<ExpectedValueSelectionV1>, DecisionValueRejectV1> {
    if options.len() > MAX_QUERY_OPTIONS {
        return Err(DecisionValueRejectV1::TooManyOptions);
    }
    let mut best: Option<ExpectedValueSelectionV1> = None;
    for option in options {
        let assessment = match evaluate_expected_value_v1(
            policy, prior, option, now_unix, calibration_verifier,
        ) {
            Ok(assessment) => assessment,
            Err(DecisionValueRejectV1::BudgetExceeded) => continue,
            Err(error) => return Err(error),
        };
        if !assessment.worth_retrieving {
            continue;
        }
        let replace = best.as_ref().is_none_or(|current| {
            assessment.expected_net_value_numerator_ppm2
                > current.assessment.expected_net_value_numerator_ppm2
        });
        if replace {
            best = Some(ExpectedValueSelectionV1 {
                query_option_id: option.query_option_id.clone(),
                assessment,
            });
        }
    }
    Ok(best)
}

/// Probability-free robust value of information using a conservative support
/// set. For each possible observation, choose the action minimizing maximum
/// regret over every state compatible with that observation; the query is
/// scored by the worst post-query regret. This does not use calibrated priors.
pub fn evaluate_robust_value_v1(
    policy: &DecisionValuePolicyV1,
    option: &DecisionRetrievalOptionV1,
) -> Result<RobustValueAssessmentV1, DecisionValueRejectV1> {
    policy.validate()?;
    option.validate(policy, false)?;

    let best_utility_by_state = policy.utility_by_state_action.iter()
        .map(|row| *row.iter().max().ok_or(DecisionValueRejectV1::InvalidDimensions))
        .collect::<Result<Vec<_>, _>>()?;

    let mut baseline_minimax_regret = i128::MAX;
    let mut baseline_action_index = 0usize;
    for action_index in 0..policy.action_ids.len() {
        let worst_regret = (0..policy.state_ids.len())
            .map(|state_index| {
                i128::from(best_utility_by_state[state_index])
                    - i128::from(policy.utility_by_state_action[state_index][action_index])
            })
            .max().ok_or(DecisionValueRejectV1::InvalidDimensions)?;
        if worst_regret < baseline_minimax_regret {
            baseline_minimax_regret = worst_regret;
            baseline_action_index = action_index;
        }
    }

    let mut post_query_worst_case_regret = 0_i128;
    let mut selected_actions_by_observation = Vec::with_capacity(option.observation_ids.len());
    for observation_index in 0..option.observation_ids.len() {
        let compatible_states = (0..policy.state_ids.len())
            .filter(|state_index| option.possible_observation_by_state[*state_index][observation_index])
            .collect::<Vec<_>>();
        if compatible_states.is_empty() {
            selected_actions_by_observation.push(RobustValueObservationChoiceV1 {
                observation_id: option.observation_ids[observation_index].clone(),
                compatible_state_ids: Vec::new(),
                selected_action_id: None,
                worst_case_regret_utility_units: None,
            });
            continue;
        }

        let mut best_action_regret = i128::MAX;
        let mut best_action_index = 0usize;
        for action_index in 0..policy.action_ids.len() {
            let worst_regret = compatible_states.iter()
                .map(|state_index| {
                    i128::from(best_utility_by_state[*state_index])
                        - i128::from(policy.utility_by_state_action[*state_index][action_index])
                })
                .max().ok_or(DecisionValueRejectV1::InvalidDimensions)?;
            if worst_regret < best_action_regret {
                best_action_regret = worst_regret;
                best_action_index = action_index;
            }
        }
        post_query_worst_case_regret = post_query_worst_case_regret.max(best_action_regret);
        selected_actions_by_observation.push(RobustValueObservationChoiceV1 {
            observation_id: option.observation_ids[observation_index].clone(),
            compatible_state_ids: compatible_states.iter()
                .map(|index| policy.state_ids[*index].clone()).collect(),
            selected_action_id: Some(policy.action_ids[best_action_index].clone()),
            worst_case_regret_utility_units: Some(best_action_regret),
        });
    }

    let gross_regret_reduction_utility_units =
        baseline_minimax_regret - post_query_worst_case_regret;
    let acquisition_cost_utility_units = i128::from(option.acquisition_cost_utility_units);
    let robust_net_value_utility_units =
        gross_regret_reduction_utility_units - acquisition_cost_utility_units;

    Ok(RobustValueAssessmentV1 {
        query_option_id: option.query_option_id.clone(),
        baseline_minimax_regret_utility_units: baseline_minimax_regret,
        post_query_worst_case_regret_utility_units: post_query_worst_case_regret,
        gross_regret_reduction_utility_units,
        acquisition_cost_utility_units,
        robust_net_value_utility_units,
        selected_actions_by_observation,
        worth_retrieving: robust_net_value_utility_units > 0,
    })
}

/// Select the query with the strongest positive robust regret reduction after
/// cost. This is the probability-free fallback when no defensible calibrated
/// probability model is available.
pub fn choose_robust_value_query_v1(
    policy: &DecisionValuePolicyV1,
    options: &[DecisionRetrievalOptionV1],
) -> Result<Option<RobustValueSelectionV1>, DecisionValueRejectV1> {
    if options.len() > MAX_QUERY_OPTIONS {
        return Err(DecisionValueRejectV1::TooManyOptions);
    }
    let mut best: Option<RobustValueSelectionV1> = None;
    for option in options {
        let assessment = match evaluate_robust_value_v1(policy, option) {
            Ok(assessment) => assessment,
            Err(DecisionValueRejectV1::BudgetExceeded) => continue,
            Err(error) => return Err(error),
        };
        if !assessment.worth_retrieving {
            continue;
        }
        let replace = best.as_ref().is_none_or(|current| {
            assessment.robust_net_value_utility_units
                > current.assessment.robust_net_value_utility_units
        });
        if replace {
            best = Some(RobustValueSelectionV1 {
                query_option_id: option.query_option_id.clone(),
                assessment,
            });
        }
    }
    Ok(best)
}

fn probability_model_sha256(
    policy: &DecisionValuePolicyV1,
    prior: &CalibratedDecisionPriorV1,
    option: &DecisionRetrievalOptionV1,
) -> String {
    let mut h = Sha256::new();
    h.update(b"luminous:mace:retrieval-probability-model:v1\\0");
    hash_field(&mut h, &policy.decision_need_sha256);
    hash_field(&mut h, &policy.utility_profile_sha256);
    hash_field(&mut h, &prior.calibration_profile_sha256);
    h.update(prior.valid_until_unix.to_be_bytes());
    h.update((policy.state_ids.len() as u64).to_be_bytes());
    for state in &policy.state_ids { hash_field(&mut h, state); }
    h.update((policy.action_ids.len() as u64).to_be_bytes());
    for action in &policy.action_ids { hash_field(&mut h, action); }
    h.update((policy.utility_by_state_action.len() as u64).to_be_bytes());
    for row in &policy.utility_by_state_action {
        h.update((row.len() as u64).to_be_bytes());
        for value in row { h.update(value.to_be_bytes()); }
    }
    h.update((prior.probability_ppm_by_state.len() as u64).to_be_bytes());
    for probability in &prior.probability_ppm_by_state {
        h.update(probability.to_be_bytes());
    }
    hash_field(&mut h, &option.query_option_id);
    hash_field(&mut h, &option.action_template_sha256);
    h.update((option.observation_ids.len() as u64).to_be_bytes());
    for observation in &option.observation_ids { hash_field(&mut h, observation); }
    h.update((option.likelihood_ppm_by_state.len() as u64).to_be_bytes());
    for row in &option.likelihood_ppm_by_state {
        h.update((row.len() as u64).to_be_bytes());
        for probability in row { h.update(probability.to_be_bytes()); }
    }
    hex_digest(&h.finalize())
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn best_expected_action(
    policy: &DecisionValuePolicyV1,
    state_weights: &[i128],
) -> Result<(usize, i128), DecisionValueRejectV1> {
    if state_weights.len() != policy.state_ids.len() {
        return Err(DecisionValueRejectV1::InvalidDimensions);
    }
    let mut best_action = 0usize;
    let mut best_value: Option<i128> = None;
    for action_index in 0..policy.action_ids.len() {
        let mut value = 0_i128;
        for (state_index, weight) in state_weights.iter().enumerate() {
            let term = weight.checked_mul(
                i128::from(policy.utility_by_state_action[state_index][action_index]),
            ).ok_or(DecisionValueRejectV1::ArithmeticOverflow)?;
            value = value.checked_add(term).ok_or(DecisionValueRejectV1::ArithmeticOverflow)?;
        }
        if best_value.is_none_or(|current| value > current) {
            best_action = action_index;
            best_value = Some(value);
        }
    }
    Ok((best_action, best_value.ok_or(DecisionValueRejectV1::InvalidDimensions)?))
}

fn strict_sorted_unique(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn valid_atom(value: &str, maximum: usize) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= maximum
        && !value.chars().any(char::is_control)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AcceptPinnedCalibration;
    impl DecisionCalibrationReceiptVerifierV1 for AcceptPinnedCalibration {
        fn verify(
            &self,
            _decision_need_sha256: &str,
            _calibration_profile_sha256: &str,
            _calibration_receipt_sha256: &str,
            _now_unix: u64,
        ) -> Result<(), CalibrationReceiptRejectV1> {
            Ok(())
        }
    }

    struct RejectCalibration;
    impl DecisionCalibrationReceiptVerifierV1 for RejectCalibration {
        fn verify(
            &self,
            _decision_need_sha256: &str,
            _calibration_profile_sha256: &str,
            _calibration_receipt_sha256: &str,
            _now_unix: u64,
        ) -> Result<(), CalibrationReceiptRejectV1> {
            Err(CalibrationReceiptRejectV1::InvalidSignature)
        }
    }

    fn policy() -> DecisionValuePolicyV1 {
        DecisionValuePolicyV1 {
            decision_need_sha256: "a".repeat(64),
            utility_profile_sha256: "b".repeat(64),
            state_ids: vec!["risky".into(), "safe".into()],
            action_ids: vec!["abstain".into(), "proceed".into()],
            utility_by_state_action: vec![
                vec![0, -100],
                vec![0, 100],
            ],
            maximum_tool_calls: 2,
            maximum_latency_ns: 10_000,
            maximum_context_tokens: 1_000,
            maximum_inspected_bytes: 10_000,
            maximum_acquisition_cost_utility_units: 100,
        }
    }

    fn informative_option() -> DecisionRetrievalOptionV1 {
        DecisionRetrievalOptionV1 {
            query_option_id: "verify-source".into(),
            action_template_sha256: "c".repeat(64),
            observation_ids: vec!["risky".into(), "safe".into()],
            likelihood_ppm_by_state: vec![
                vec![1_000_000, 0],
                vec![0, 1_000_000],
            ],
            possible_observation_by_state: vec![
                vec![true, false],
                vec![false, true],
            ],
            tool_calls: 1,
            estimated_latency_ns: 100,
            estimated_context_tokens: 40,
            estimated_inspected_bytes: 500,
            acquisition_cost_utility_units: 5,
        }
    }

    fn prior() -> CalibratedDecisionPriorV1 {
        CalibratedDecisionPriorV1 {
            calibration_profile_sha256: "d".repeat(64),
            calibration_receipt_sha256: "e".repeat(64),
            valid_until_unix: 100,
            probability_ppm_by_state: vec![500_000, 500_000],
        }
    }

    #[test]
    fn informative_query_has_positive_evsi_after_cost() {
        let assessment = evaluate_expected_value_v1(
            &policy(), &prior(), &informative_option(), 10, &AcceptPinnedCalibration,
        ).unwrap();
        assert_eq!(assessment.baseline_action_id, "abstain");
        assert_eq!(assessment.gross_evsi_numerator_ppm2, 50_000_000_000_000);
        assert_eq!(assessment.acquisition_cost_numerator_ppm2, 5_000_000_000_000);
        assert_eq!(assessment.expected_net_value_numerator_ppm2, 45_000_000_000_000);
        assert!(assessment.worth_retrieving);
        assert_eq!(
            assessment.selected_actions_by_observation.iter()
                .map(|item| item.selected_action_id.as_deref().unwrap()).collect::<Vec<_>>(),
            vec!["abstain", "proceed"]
        );
    }

    struct PinnedModelCalibration {
        expected_model_sha256: String,
    }
    impl DecisionCalibrationReceiptVerifierV1 for PinnedModelCalibration {
        fn verify(
            &self,
            _decision_need_sha256: &str,
            _calibration_profile_sha256: &str,
            _calibration_receipt_sha256: &str,
            probability_model_sha256: &str,
            _now_unix: u64,
        ) -> Result<(), CalibrationReceiptRejectV1> {
            if probability_model_sha256 == self.expected_model_sha256 {
                Ok(())
            } else {
                Err(CalibrationReceiptRejectV1::ProbabilityModelMismatch)
            }
        }
    }

    #[test]
    fn calibration_receipt_is_bound_to_the_exact_prior_and_likelihood_model() {
        let policy = policy();
        let prior = prior();
        let option = informative_option();
        let model_sha256 = probability_model_sha256(&policy, &prior, &option);
        let verifier = PinnedModelCalibration { expected_model_sha256: model_sha256 };
        assert!(evaluate_expected_value_v1(
            &policy, &prior, &option, 10, &verifier,
        ).is_ok());
        let mut altered_option = option.clone();
        altered_option.likelihood_ppm_by_state = vec![
            vec![500_000, 500_000],
            vec![500_000, 500_000],
        ];
        assert_eq!(
            evaluate_expected_value_v1(
                &policy, &prior, &altered_option, 10, &verifier,
            ),
            Err(DecisionValueRejectV1::CalibrationReceiptRejected)
        );
    }

    #[test]
    fn an_uninformative_query_with_positive_cost_is_not_selected() {
        let mut option = informative_option();
        option.likelihood_ppm_by_state = vec![
            vec![500_000, 500_000],
            vec![500_000, 500_000],
        ];
        option.possible_observation_by_state = vec![
            vec![true, true],
            vec![true, true],
        ];
        let assessment = evaluate_expected_value_v1(
            &policy(), &prior(), &option, 10, &AcceptPinnedCalibration,
        ).unwrap();
        assert_eq!(assessment.gross_evsi_numerator_ppm2, 0);
        assert!(!assessment.worth_retrieving);
    }

    #[test]
    fn expected_value_never_uses_an_unverified_calibration_receipt() {
        assert_eq!(
            evaluate_expected_value_v1(
                &policy(), &prior(), &informative_option(), 10, &RejectCalibration,
            ),
            Err(DecisionValueRejectV1::CalibrationReceiptRejected)
        );
    }

    #[test]
    fn expected_value_rejects_malformed_prior_probability_mass() {
        let mut prior = prior();
        prior.probability_ppm_by_state = vec![500_000, 400_000];
        assert_eq!(
            evaluate_expected_value_v1(
                &policy(), &prior, &informative_option(), 10, &AcceptPinnedCalibration,
            ),
            Err(DecisionValueRejectV1::InvalidProbability)
        );
    }

    #[test]
    fn robust_minimax_value_is_positive_without_any_probability_model() {
        let assessment = evaluate_robust_value_v1(&policy(), &informative_option()).unwrap();
        assert_eq!(assessment.baseline_minimax_regret_utility_units, 100);
        assert_eq!(assessment.post_query_worst_case_regret_utility_units, 0);
        assert_eq!(assessment.gross_regret_reduction_utility_units, 100);
        assert_eq!(assessment.acquisition_cost_utility_units, 5);
        assert_eq!(assessment.robust_net_value_utility_units, 95);
        assert!(assessment.worth_retrieving);
    }

    #[test]
    fn robust_evaluation_needs_no_calibrated_probability_matrix() {
        let mut option = informative_option();
        option.likelihood_ppm_by_state.clear();
        let assessment = evaluate_robust_value_v1(&policy(), &option).unwrap();
        assert_eq!(assessment.robust_net_value_utility_units, 95);
    }

    #[test]
    fn an_uninformative_robust_query_does_not_reduce_worst_case_regret() {
        let mut option = informative_option();
        option.possible_observation_by_state = vec![
            vec![true, true],
            vec![true, true],
        ];
        let assessment = evaluate_robust_value_v1(&policy(), &option).unwrap();
        assert_eq!(assessment.gross_regret_reduction_utility_units, 0);
        assert!(!assessment.worth_retrieving);
    }

    #[test]
    fn query_budget_exhaustion_blocks_selection_in_both_modes() {
        let mut option = informative_option();
        option.estimated_latency_ns = 20_000;
        assert_eq!(
            evaluate_expected_value_v1(
                &policy(), &prior(), &option, 10, &AcceptPinnedCalibration,
            ),
            Err(DecisionValueRejectV1::BudgetExceeded)
        );
        assert_eq!(
            evaluate_robust_value_v1(&policy(), &option),
            Err(DecisionValueRejectV1::BudgetExceeded)
        );
    }

    #[test]
    fn stale_calibration_receipt_is_rejected() {
        let mut prior = prior();
        prior.valid_until_unix = 10;
        assert_eq!(
            evaluate_expected_value_v1(
                &policy(), &prior, &informative_option(), 10, &AcceptPinnedCalibration,
            ),
            Err(DecisionValueRejectV1::InvalidCalibration)
        );
    }
}
