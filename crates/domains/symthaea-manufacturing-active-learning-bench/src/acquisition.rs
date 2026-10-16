// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finite-hypothesis expected-information-gain (EIG) and decision-value kernel.
//!
//! Unlike the baseline harness's spatial novelty proxy, this module computes
//! expected entropy reduction from an explicit prior and observation likelihood.
//! It is a deterministic mathematical fixture, not a calibrated model of any
//! physical manufacturing process.
use crate::Feasibility;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const PROBABILITY_SCALE_PPM: u32 = 1_000_000;
const NUMERIC_TOLERANCE: f64 = 1e-10;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HypothesisPrior {
    pub hypothesis_id: String,
    /// Prior/posterior probability in parts per million; all weights sum to 1,000,000.
    /// Zero is valid after conditioning when an observation rules a hypothesis out.
    pub probability_ppm: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutcomeProbability {
    pub outcome_id: String,
    /// Conditional probability in parts per million; each hypothesis row sums to 1,000,000.
    pub probability_ppm: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HypothesisLikelihood {
    pub hypothesis_id: String,
    pub outcomes: Vec<OutcomeProbability>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperimentActionModel {
    pub action_id: String,
    pub feasibility: Feasibility,
    /// Declared relative benchmark cost units; not currency or physical process cost.
    pub cost_units: u32,
    pub measurement_model_id: String,
    pub likelihood_by_hypothesis: Vec<HypothesisLikelihood>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HypothesisUtility {
    pub hypothesis_id: String,
    /// Synthetic utility in milli-units for this decision under this hypothesis.
    pub utility_milli: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionAlternative {
    pub decision_id: String,
    pub utility_by_hypothesis: Vec<HypothesisUtility>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiniteHypothesisModel {
    pub schema_version: u16,
    pub model_id: String,
    pub model_version: String,
    pub prior: Vec<HypothesisPrior>,
    pub actions: Vec<ExperimentActionModel>,
    pub decisions: Vec<DecisionAlternative>,
}

impl FiniteHypothesisModel {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err("unsupported finite-hypothesis model schema".into());
        }
        if self.model_id.trim().is_empty() || self.model_version.trim().is_empty() {
            return Err("model ID and version must be non-empty".into());
        }
        if self.prior.is_empty() {
            return Err("hypothesis prior cannot be empty".into());
        }
        if self.actions.is_empty() {
            return Err("at least one candidate action is required".into());
        }
        if self.decisions.is_empty() {
            return Err("at least one decision alternative is required".into());
        }

        let mut hypothesis_ids = BTreeSet::new();
        let mut prior_total = 0u64;
        for hypothesis in &self.prior {
            if hypothesis.hypothesis_id.trim().is_empty()
                || !hypothesis_ids.insert(hypothesis.hypothesis_id.as_str())
            {
                return Err("hypothesis IDs must be non-empty and unique".into());
            }
            if hypothesis.probability_ppm > PROBABILITY_SCALE_PPM {
                return Err("prior probability exceeds one million ppm".into());
            }
            prior_total += u64::from(hypothesis.probability_ppm);
        }
        if prior_total != u64::from(PROBABILITY_SCALE_PPM) {
            return Err("prior probabilities must sum exactly to one million ppm".into());
        }

        let expected_hypotheses: BTreeSet<&str> =
            self.prior.iter().map(|row| row.hypothesis_id.as_str()).collect();
        let mut action_ids = BTreeSet::new();
        for action in &self.actions {
            if action.action_id.trim().is_empty()
                || !action_ids.insert(action.action_id.as_str())
            {
                return Err("action IDs must be non-empty and unique".into());
            }
            if action.cost_units == 0 {
                return Err(format!("action {} has zero declared cost", action.action_id));
            }
            if action.measurement_model_id.trim().is_empty() {
                return Err(format!(
                    "action {} has no measurement-model identity",
                    action.action_id
                ));
            }
            match &action.feasibility {
                Feasibility::Blocked { reason } | Feasibility::Unknown { reason }
                    if reason.trim().is_empty() =>
                {
                    return Err(format!(
                        "action {} has a non-feasible disposition without a reason",
                        action.action_id
                    ));
                }
                _ => {}
            }
            if action.likelihood_by_hypothesis.len() != self.prior.len() {
                return Err(format!(
                    "action {} does not provide one likelihood row per hypothesis",
                    action.action_id
                ));
            }

            let mut seen_rows = BTreeSet::new();
            let mut common_outcomes: Option<BTreeSet<&str>> = None;
            for row in &action.likelihood_by_hypothesis {
                if !expected_hypotheses.contains(row.hypothesis_id.as_str())
                    || !seen_rows.insert(row.hypothesis_id.as_str())
                {
                    return Err(format!(
                        "action {} has an unknown or duplicate hypothesis likelihood",
                        action.action_id
                    ));
                }
                if row.outcomes.is_empty() {
                    return Err(format!(
                        "action {} has an empty outcome distribution",
                        action.action_id
                    ));
                }
                let mut outcome_ids = BTreeSet::new();
                let mut row_total = 0u64;
                for outcome in &row.outcomes {
                    if outcome.outcome_id.trim().is_empty()
                        || !outcome_ids.insert(outcome.outcome_id.as_str())
                    {
                        return Err(format!(
                            "action {} has empty or duplicate outcome IDs",
                            action.action_id
                        ));
                    }
                    if outcome.probability_ppm > PROBABILITY_SCALE_PPM {
                        return Err(format!(
                            "action {} has an outcome probability above one",
                            action.action_id
                        ));
                    }
                    row_total += u64::from(outcome.probability_ppm);
                }
                if row_total != u64::from(PROBABILITY_SCALE_PPM) {
                    return Err(format!(
                        "action {} outcome probabilities do not sum to one million ppm",
                        action.action_id
                    ));
                }
                if let Some(expected) = &common_outcomes {
                    if expected != &outcome_ids {
                        return Err(format!(
                            "action {} uses inconsistent outcome sets across hypotheses",
                            action.action_id
                        ));
                    }
                } else {
                    common_outcomes = Some(outcome_ids);
                }
            }
            if seen_rows != expected_hypotheses {
                return Err(format!(
                    "action {} is missing one or more hypothesis rows",
                    action.action_id
                ));
            }
        }

        let mut decision_ids = BTreeSet::new();
        for decision in &self.decisions {
            if decision.decision_id.trim().is_empty()
                || !decision_ids.insert(decision.decision_id.as_str())
            {
                return Err("decision IDs must be non-empty and unique".into());
            }
            if decision.utility_by_hypothesis.len() != self.prior.len() {
                return Err(format!(
                    "decision {} needs one utility per hypothesis",
                    decision.decision_id
                ));
            }
            let mut seen_utility_hypotheses = BTreeSet::new();
            for utility in &decision.utility_by_hypothesis {
                if !expected_hypotheses.contains(utility.hypothesis_id.as_str())
                    || !seen_utility_hypotheses.insert(utility.hypothesis_id.as_str())
                {
                    return Err(format!(
                        "decision {} has an unknown or duplicate utility row",
                        decision.decision_id
                    ));
                }
            }
            if seen_utility_hypotheses != expected_hypotheses {
                return Err(format!(
                    "decision {} is missing a hypothesis utility",
                    decision.decision_id
                ));
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("model serialization failed: {error}"))?;
        Ok(hex::encode(Sha256::digest(bytes)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PosteriorEvidenceReceipt {
    pub evidence_id: String,
    pub action_id: String,
    pub observed_outcome_id: String,
    pub measurement_model_id: String,
    pub parent_state_digest: String,
    pub posterior_after: Vec<HypothesisPrior>,
    pub posterior_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PosteriorState {
    pub root_model_digest: String,
    pub current_prior: Vec<HypothesisPrior>,
    pub evidence_receipts: Vec<PosteriorEvidenceReceipt>,
    pub state_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PosteriorUpdate {
    pub parent_state_digest: String,
    pub evidence_id: String,
    pub action_id: String,
    pub observed_outcome_id: String,
    pub posterior: Vec<HypothesisPrior>,
    pub posterior_model_digest: String,
    pub next_state: PosteriorState,
}

/// Condition the finite prior on one admitted action's observed categorical outcome.
///
/// Posterior weights are renormalized to integer parts-per-million using the
/// largest-remainder method; ties are broken by hypothesis ID. An outcome with
/// zero probability under every hypothesis is rejected rather than assigned a
/// fabricated posterior.
/// Create the immutable empty evidence-history state for a validated model.
pub fn initial_posterior_state(model: &FiniteHypothesisModel) -> Result<PosteriorState, String> {
    model.validate()?;
    let root_model_digest = model.digest()?;
    let current_prior = model.prior.clone();
    let evidence_receipts = Vec::new();
    let state_digest = posterior_state_digest(
        &root_model_digest,
        &current_prior,
        &evidence_receipts,
    )?;
    Ok(PosteriorState {
        root_model_digest,
        current_prior,
        evidence_receipts,
        state_digest,
    })
}

/// Condition a posterior on one observation with a unique evidence ID.
///
/// Every transition is recorded with its parent-state digest. State validation
/// replays the complete history and rejects duplicate IDs or mutated transitions.
/// Digests establish deterministic identity, not signature-based source authenticity.
pub fn condition_posterior(
    model: &FiniteHypothesisModel,
    state: &PosteriorState,
    evidence_id: &str,
    action_id: &str,
    observed_outcome_id: &str,
) -> Result<PosteriorUpdate, String> {
    validate_posterior_state(model, state)?;
    if evidence_id.trim().is_empty() {
        return Err("evidence ID must be non-empty".into());
    }
    if state
        .evidence_receipts
        .iter()
        .any(|receipt| receipt.evidence_id == evidence_id)
    {
        return Err(format!("duplicate evidence ID: {evidence_id}"));
    }

    let parent_state_digest = state.state_digest.clone();
    let action = model
        .actions
        .iter()
        .find(|action| action.action_id == action_id)
        .ok_or_else(|| format!("unknown acquisition action: {action_id}"))?;
    if action.feasibility != Feasibility::Feasible {
        return Err(format!(
            "action {} is not feasible and cannot condition the posterior",
            action.action_id
        ));
    }

    let posterior = posterior_from_observation(
        model,
        &state.current_prior,
        action,
        observed_outcome_id,
    )?;
    let posterior_digest = digest_serializable(&posterior)?;
    let receipt = PosteriorEvidenceReceipt {
        evidence_id: evidence_id.into(),
        action_id: action_id.into(),
        observed_outcome_id: observed_outcome_id.into(),
        measurement_model_id: action.measurement_model_id.clone(),
        parent_state_digest: parent_state_digest.clone(),
        posterior_after: posterior.clone(),
        posterior_digest,
    };

    let mut next_state = state.clone();
    next_state.current_prior = posterior.clone();
    next_state.evidence_receipts.push(receipt);
    next_state.state_digest = posterior_state_digest(
        &next_state.root_model_digest,
        &next_state.current_prior,
        &next_state.evidence_receipts,
    )?;

    let mut posterior_model = model.clone();
    posterior_model.prior = posterior.clone();
    let posterior_model_digest = posterior_model.digest()?;
    Ok(PosteriorUpdate {
        parent_state_digest,
        evidence_id: evidence_id.into(),
        action_id: action_id.into(),
        observed_outcome_id: observed_outcome_id.into(),
        posterior,
        posterior_model_digest,
        next_state,
    })
}

fn posterior_state_digest(
    root_model_digest: &str,
    current_prior: &[HypothesisPrior],
    evidence_receipts: &[PosteriorEvidenceReceipt],
) -> Result<String, String> {
    digest_serializable(&(root_model_digest, current_prior, evidence_receipts))
}

fn validate_posterior_state(
    model: &FiniteHypothesisModel,
    state: &PosteriorState,
) -> Result<(), String> {
    model.validate()?;
    let root_model_digest = model.digest()?;
    if state.root_model_digest != root_model_digest {
        return Err("posterior state belongs to a different root model".into());
    }

    let mut expected_prior = model.prior.clone();
    let mut expected_receipts: Vec<PosteriorEvidenceReceipt> = Vec::new();
    let mut expected_state_digest =
        posterior_state_digest(&root_model_digest, &expected_prior, &expected_receipts)?;
    let mut seen_evidence_ids = BTreeSet::new();

    for receipt in &state.evidence_receipts {
        if receipt.evidence_id.trim().is_empty()
            || !seen_evidence_ids.insert(receipt.evidence_id.as_str())
        {
            return Err("posterior history has an empty or duplicate evidence ID".into());
        }
        if receipt.parent_state_digest != expected_state_digest {
            return Err(format!(
                "posterior history parent-state digest mismatch at evidence {}",
                receipt.evidence_id
            ));
        }
        let action = model
            .actions
            .iter()
            .find(|action| action.action_id == receipt.action_id)
            .ok_or_else(|| {
                format!(
                    "posterior history references unknown action {}",
                    receipt.action_id
                )
            })?;
        if action.measurement_model_id != receipt.measurement_model_id {
            return Err(format!(
                "measurement model identity mismatch at evidence {}",
                receipt.evidence_id
            ));
        }
        if action.feasibility != Feasibility::Feasible {
            return Err(format!(
                "posterior history contains non-feasible action {}",
                receipt.action_id
            ));
        }
        let expected_posterior = posterior_from_observation(
            model,
            &expected_prior,
            action,
            &receipt.observed_outcome_id,
        )?;
        let expected_posterior_digest = digest_serializable(&expected_posterior)?;
        if receipt.posterior_after != expected_posterior
            || receipt.posterior_digest != expected_posterior_digest
        {
            return Err(format!(
                "posterior history transition mismatch at evidence {}",
                receipt.evidence_id
            ));
        }

        expected_prior = expected_posterior;
        expected_receipts.push(receipt.clone());
        expected_state_digest = posterior_state_digest(
            &root_model_digest,
            &expected_prior,
            &expected_receipts,
        )?;
    }

    if state.current_prior != expected_prior {
        return Err("posterior state current prior does not match replayed history".into());
    }
    if state.state_digest != expected_state_digest {
        return Err("posterior state digest mismatch".into());
    }
    Ok(())
}

fn posterior_from_observation(
    model: &FiniteHypothesisModel,
    prior: &[HypothesisPrior],
    action: &ExperimentActionModel,
    observed_outcome_id: &str,
) -> Result<Vec<HypothesisPrior>, String> {
    let expected_ids: BTreeSet<&str> = model
        .prior
        .iter()
        .map(|item| item.hypothesis_id.as_str())
        .collect();
    if prior.len() != model.prior.len() {
        return Err("current posterior has the wrong hypothesis count".into());
    }
    let mut prior_total = 0u64;
    let mut seen_ids = BTreeSet::new();
    for item in prior {
        if !expected_ids.contains(item.hypothesis_id.as_str())
            || !seen_ids.insert(item.hypothesis_id.as_str())
        {
            return Err("current posterior has unknown or duplicate hypothesis IDs".into());
        }
        prior_total += u64::from(item.probability_ppm);
    }
    if prior_total != u64::from(PROBABILITY_SCALE_PPM) || seen_ids != expected_ids {
        return Err("current posterior probabilities must sum to one million ppm".into());
    }

    let mut raw_weights = Vec::with_capacity(prior.len());
    for item in prior {
        let row = action
            .likelihood_by_hypothesis
            .iter()
            .find(|row| row.hypothesis_id == item.hypothesis_id)
            .ok_or_else(|| format!("missing likelihood row for {}", item.hypothesis_id))?;
        let likelihood = row
            .outcomes
            .iter()
            .find(|outcome| outcome.outcome_id == observed_outcome_id)
            .ok_or_else(|| format!("unknown outcome ID: {observed_outcome_id}"))?;
        raw_weights.push(
            u128::from(item.probability_ppm) * u128::from(likelihood.probability_ppm),
        );
    }

    let total_weight: u128 = raw_weights.iter().sum();
    if total_weight == 0 {
        return Err("observed outcome has zero probability under every hypothesis".into());
    }

    let scale = u128::from(PROBABILITY_SCALE_PPM);
    let mut quantized: Vec<(u32, u128)> = raw_weights
        .iter()
        .map(|weight| {
            let scaled = *weight * scale;
            ((scaled / total_weight) as u32, scaled % total_weight)
        })
        .collect();
    let assigned: u64 = quantized.iter().map(|(floor, _)| u64::from(*floor)).sum();
    let remainder = u64::from(PROBABILITY_SCALE_PPM)
        .checked_sub(assigned)
        .ok_or_else(|| "posterior normalization overflow".to_string())?;
    let mut order: Vec<usize> = (0..quantized.len()).collect();
    order.sort_by(|left, right| {
        quantized[*right]
            .1
            .cmp(&quantized[*left].1)
            .then_with(|| {
                prior[*left]
                    .hypothesis_id
                    .cmp(&prior[*right].hypothesis_id)
            })
    });
    if remainder as usize > order.len() {
        return Err("posterior normalization remainder is out of range".into());
    }
    for index in order.into_iter().take(remainder as usize) {
        quantized[index].0 += 1;
    }

    Ok(prior
        .iter()
        .zip(quantized.iter())
        .map(|(item, (probability_ppm, _))| HypothesisPrior {
            hypothesis_id: item.hypothesis_id.clone(),
            probability_ppm: *probability_ppm,
        })
        .collect())
}

pub struct OutcomeMarginal {
    pub outcome_id: String,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcquisitionMetrics {
    pub model_digest: String,
    pub action_id: String,
    pub action_cost_units: u32,
    pub prior_entropy_nats: f64,
    pub expected_posterior_entropy_nats: f64,
    pub expected_information_gain_nats: f64,
    pub expected_sample_information_value_milli: f64,
    pub information_gain_per_cost: f64,
    pub decision_value_per_cost: f64,
    pub outcome_marginals: Vec<OutcomeMarginal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcquisitionObjective {
    ExpectedInformationGain,
    InformationGainPerCost,
    ExpectedDecisionValue,
    DecisionValuePerCost,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedAction {
    pub action_id: String,
    pub objective_score: f64,
    pub metrics: AcquisitionMetrics,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedAction {
    pub action_id: String,
    pub disposition: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcquisitionRanking {
    pub model_digest: String,
    pub objective: AcquisitionObjective,
    pub ranked_actions: Vec<RankedAction>,
    pub rejected_actions: Vec<RejectedAction>,
}

pub fn expected_information_gain(
    model: &FiniteHypothesisModel,
    action_id: &str,
) -> Result<AcquisitionMetrics, String> {
    model.validate()?;
    let digest = model.digest()?;
    let action = model
        .actions
        .iter()
        .find(|action| action.action_id == action_id)
        .ok_or_else(|| format!("unknown acquisition action: {action_id}"))?;
    calculate_acquisition(model, action, digest)
}

pub fn rank_feasible_actions(
    model: &FiniteHypothesisModel,
    objective: AcquisitionObjective,
) -> Result<AcquisitionRanking, String> {
    model.validate()?;
    let model_digest = model.digest()?;
    let mut ranked_actions = Vec::new();
    let mut rejected_actions = Vec::new();

    for action in &model.actions {
        match &action.feasibility {
            Feasibility::Feasible => {
                let metrics =
                    calculate_acquisition(model, action, model_digest.clone())?;
                let objective_score = match objective {
                    AcquisitionObjective::ExpectedInformationGain => {
                        metrics.expected_information_gain_nats
                    }
                    AcquisitionObjective::InformationGainPerCost => {
                        metrics.information_gain_per_cost
                    }
                    AcquisitionObjective::ExpectedDecisionValue => {
                        metrics.expected_sample_information_value_milli
                    }
                    AcquisitionObjective::DecisionValuePerCost => metrics.decision_value_per_cost,
                };
                ranked_actions.push(RankedAction {
                    action_id: action.action_id.clone(),
                    objective_score,
                    metrics,
                });
            }
            Feasibility::Blocked { reason } => rejected_actions.push(RejectedAction {
                action_id: action.action_id.clone(),
                disposition: "Blocked".into(),
                reason: reason.clone(),
            }),
            Feasibility::Unknown { reason } => rejected_actions.push(RejectedAction {
                action_id: action.action_id.clone(),
                disposition: "Unknown".into(),
                reason: reason.clone(),
            }),
        }
    }

    ranked_actions.sort_by(|left, right| {
        right
            .objective_score
            .total_cmp(&left.objective_score)
            .then_with(|| left.action_id.cmp(&right.action_id))
    });

    Ok(AcquisitionRanking {
        model_digest,
        objective,
        ranked_actions,
        rejected_actions,
    })
}

fn calculate_acquisition(
    model: &FiniteHypothesisModel,
    action: &ExperimentActionModel,
    model_digest: String,
) -> Result<AcquisitionMetrics, String> {
    match &action.feasibility {
        Feasibility::Feasible => {}
        Feasibility::Blocked { reason } => {
            return Err(format!("action {} is blocked: {reason}", action.action_id));
        }
        Feasibility::Unknown { reason } => {
            return Err(format!("action {} feasibility is unknown: {reason}", action.action_id));
        }
    }

    let hypothesis_ids: Vec<&str> = model
        .prior
        .iter()
        .map(|row| row.hypothesis_id.as_str())
        .collect();
    let prior_probabilities: Vec<f64> = model
        .prior
        .iter()
        .map(|row| f64::from(row.probability_ppm) / f64::from(PROBABILITY_SCALE_PPM))
        .collect();
    let prior_entropy = entropy(&prior_probabilities);

    let likelihood_rows: BTreeMap<&str, BTreeMap<&str, f64>> = action
        .likelihood_by_hypothesis
        .iter()
        .map(|row| {
            let outcome_map = row
                .outcomes
                .iter()
                .map(|outcome| {
                    (
                        outcome.outcome_id.as_str(),
                        f64::from(outcome.probability_ppm)
                            / f64::from(PROBABILITY_SCALE_PPM),
                    )
                })
                .collect();
            (row.hypothesis_id.as_str(), outcome_map)
        })
        .collect();

    let outcome_ids: BTreeSet<&str> = action
        .likelihood_by_hypothesis
        .iter()
        .flat_map(|row| row.outcomes.iter().map(|outcome| outcome.outcome_id.as_str()))
        .collect();
    let prior_by_hypothesis: BTreeMap<&str, f64> = hypothesis_ids
        .iter()
        .copied()
        .zip(prior_probabilities.iter().copied())
        .collect();

    let current_best_utility = model
        .decisions
        .iter()
        .map(|decision| expected_decision_utility(decision, &prior_by_hypothesis))
        .fold(f64::NEG_INFINITY, f64::max);

    let mut outcome_marginals = Vec::new();
    let mut expected_posterior_entropy = 0.0;
    let mut expected_post_observation_best_utility = 0.0;

    for outcome_id in outcome_ids {
        let joint_probabilities: Vec<f64> = hypothesis_ids
            .iter()
            .zip(prior_probabilities.iter())
            .map(|(hypothesis_id, prior)| {
                let likelihood = likelihood_rows
                    .get(*hypothesis_id)
                    .and_then(|row| row.get(outcome_id))
                    .copied()
                    .ok_or_else(|| {
                        format!(
                            "validated likelihood is missing outcome {} for hypothesis {}",
                            outcome_id, hypothesis_id
                        )
                    })?;
                Ok::<f64, String>(*prior * likelihood)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let marginal = joint_probabilities.iter().sum::<f64>();
        outcome_marginals.push(OutcomeMarginal {
            outcome_id: outcome_id.to_string(),
            probability: marginal,
        });
        if marginal <= 0.0 {
            continue;
        }

        let posterior: Vec<f64> = joint_probabilities
            .iter()
            .map(|joint| joint / marginal)
            .collect();
        expected_posterior_entropy += marginal * entropy(&posterior);

        let posterior_by_hypothesis: BTreeMap<&str, f64> = hypothesis_ids
            .iter()
            .copied()
            .zip(posterior.iter().copied())
            .collect();
        let best_posterior_decision_utility = model
            .decisions
            .iter()
            .map(|decision| expected_decision_utility(decision, &posterior_by_hypothesis))
            .fold(f64::NEG_INFINITY, f64::max);
        expected_post_observation_best_utility += marginal * best_posterior_decision_utility;
    }

    outcome_marginals.sort_by(|left, right| left.outcome_id.cmp(&right.outcome_id));
    let mut eig = prior_entropy - expected_posterior_entropy;
    if eig < 0.0 && eig > -NUMERIC_TOLERANCE {
        eig = 0.0;
    }
    if !eig.is_finite() || eig < 0.0 {
        return Err("computed expected information gain is non-finite or negative".into());
    }

    let mut decision_value = expected_post_observation_best_utility - current_best_utility;
    let decision_tolerance = NUMERIC_TOLERANCE
        * expected_post_observation_best_utility
            .abs()
            .max(current_best_utility.abs())
            .max(1.0);
    if decision_value < 0.0 && decision_value > -decision_tolerance {
        decision_value = 0.0;
    }
    if !decision_value.is_finite() || decision_value < 0.0 {
        return Err("computed expected decision value is non-finite or negative".into());
    }

    let cost = f64::from(action.cost_units);
    Ok(AcquisitionMetrics {
        model_digest,
        action_id: action.action_id.clone(),
        action_cost_units: action.cost_units,
        prior_entropy_nats: prior_entropy,
        expected_posterior_entropy_nats: expected_posterior_entropy,
        expected_information_gain_nats: eig,
        expected_sample_information_value_milli: decision_value,
        information_gain_per_cost: eig / cost,
        decision_value_per_cost: decision_value / cost,
        outcome_marginals,
    })
}

fn expected_decision_utility(
    decision: &DecisionAlternative,
    probabilities_by_hypothesis: &BTreeMap<&str, f64>,
) -> f64 {
    decision
        .utility_by_hypothesis
        .iter()
        .map(|utility| {
            probabilities_by_hypothesis[utility.hypothesis_id.as_str()]
                * utility.utility_milli as f64
        })
        .sum()
}

fn entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(
        action_id: &str,
        feasibility: Feasibility,
        h1_outcomes: [u32; 2],
        h2_outcomes: [u32; 2],
        decisions: Vec<DecisionAlternative>,
    ) -> FiniteHypothesisModel {
        FiniteHypothesisModel {
            schema_version: 1,
            model_id: "finite-test-model".into(),
            model_version: "1.0.0".into(),
            prior: vec![
                HypothesisPrior {
                    hypothesis_id: "H1".into(),
                    probability_ppm: 500_000,
                },
                HypothesisPrior {
                    hypothesis_id: "H2".into(),
                    probability_ppm: 500_000,
                },
            ],
            actions: vec![ExperimentActionModel {
                action_id: action_id.into(),
                feasibility,
                cost_units: 1,
                measurement_model_id: "synthetic-likelihood-v1".into(),
                likelihood_by_hypothesis: vec![
                    HypothesisLikelihood {
                        hypothesis_id: "H1".into(),
                        outcomes: vec![
                            OutcomeProbability {
                                outcome_id: "O1".into(),
                                probability_ppm: h1_outcomes[0],
                            },
                            OutcomeProbability {
                                outcome_id: "O2".into(),
                                probability_ppm: h1_outcomes[1],
                            },
                        ],
                    },
                    HypothesisLikelihood {
                        hypothesis_id: "H2".into(),
                        outcomes: vec![
                            OutcomeProbability {
                                outcome_id: "O1".into(),
                                probability_ppm: h2_outcomes[0],
                            },
                            OutcomeProbability {
                                outcome_id: "O2".into(),
                                probability_ppm: h2_outcomes[1],
                            },
                        ],
                    },
                ],
            }],
            decisions,
        }
    }

    fn opposing_decisions() -> Vec<DecisionAlternative> {
        vec![
            DecisionAlternative {
                decision_id: "choose-H1".into(),
                utility_by_hypothesis: vec![
                    HypothesisUtility {
                        hypothesis_id: "H1".into(),
                        utility_milli: 1_000,
                    },
                    HypothesisUtility {
                        hypothesis_id: "H2".into(),
                        utility_milli: 0,
                    },
                ],
            },
            DecisionAlternative {
                decision_id: "choose-H2".into(),
                utility_by_hypothesis: vec![
                    HypothesisUtility {
                        hypothesis_id: "H1".into(),
                        utility_milli: 0,
                    },
                    HypothesisUtility {
                        hypothesis_id: "H2".into(),
                        utility_milli: 1_000,
                    },
                ],
            },
        ]
    }

    #[test]
    fn perfect_observation_has_ln_two_information_gain_and_half_thousand_value() {
        let model = model(
            "perfect",
            Feasibility::Feasible,
            [1_000_000, 0],
            [0, 1_000_000],
            opposing_decisions(),
        );
        let metrics = expected_information_gain(&model, "perfect").unwrap();
        assert!((metrics.expected_information_gain_nats - std::f64::consts::LN_2).abs() < 1e-12);
        assert!(metrics.expected_posterior_entropy_nats.abs() < 1e-12);
        assert!((metrics.expected_sample_information_value_milli - 500.0).abs() < 1e-9);
        assert_eq!(metrics.outcome_marginals.len(), 2);
        assert!((metrics.outcome_marginals[0].probability - 0.5).abs() < 1e-12);
    }

    #[test]
    fn uninformative_observation_has_zero_information_and_zero_decision_value() {
        let model = model(
            "uninformative",
            Feasibility::Feasible,
            [500_000, 500_000],
            [500_000, 500_000],
            opposing_decisions(),
        );
        let metrics = expected_information_gain(&model, "uninformative").unwrap();
        assert!(metrics.expected_information_gain_nats.abs() < 1e-12);
        assert!(metrics.expected_sample_information_value_milli.abs() < 1e-9);
    }

    #[test]
    fn information_gain_does_not_imply_decision_value() {
        let decisions = vec![
            DecisionAlternative {
                decision_id: "always-safe".into(),
                utility_by_hypothesis: vec![
                    HypothesisUtility {
                        hypothesis_id: "H1".into(),
                        utility_milli: 1_000,
                    },
                    HypothesisUtility {
                        hypothesis_id: "H2".into(),
                        utility_milli: 1_000,
                    },
                ],
            },
            DecisionAlternative {
                decision_id: "riskier".into(),
                utility_by_hypothesis: vec![
                    HypothesisUtility {
                        hypothesis_id: "H1".into(),
                        utility_milli: 900,
                    },
                    HypothesisUtility {
                        hypothesis_id: "H2".into(),
                        utility_milli: 900,
                    },
                ],
            },
        ];
        let model = model(
            "perfect",
            Feasibility::Feasible,
            [1_000_000, 0],
            [0, 1_000_000],
            decisions,
        );
        let metrics = expected_information_gain(&model, "perfect").unwrap();
        assert!((metrics.expected_information_gain_nats - std::f64::consts::LN_2).abs() < 1e-12);
        assert!(metrics.expected_sample_information_value_milli.abs() < 1e-9);
    }

    #[test]
    fn noisy_binary_fixture_matches_hand_calculable_information_gain() {
        let model = model(
            "noisy",
            Feasibility::Feasible,
            [900_000, 100_000],
            [100_000, 900_000],
            opposing_decisions(),
        );
        let expected = std::f64::consts::LN_2
            + 0.9_f64 * 0.9_f64.ln()
            + 0.1_f64 * 0.1_f64.ln();
        let metrics = expected_information_gain(&model, "noisy").unwrap();
        assert!((metrics.expected_information_gain_nats - expected).abs() < 1e-12);
        assert!(metrics.expected_sample_information_value_milli > 0.0);
    }

    #[test]
    fn infeasible_high_information_action_is_not_ranked() {
        let decisions = opposing_decisions();
        let mut model = model(
            "feasible",
            Feasibility::Feasible,
            [500_000, 500_000],
            [500_000, 500_000],
            decisions.clone(),
        );
        let mut blocked = model.actions[0].clone();
        blocked.action_id = "blocked-perfect".into();
        blocked.feasibility = Feasibility::Blocked {
            reason: "outside synthetic capability envelope".into(),
        };
        blocked.likelihood_by_hypothesis = vec![
            HypothesisLikelihood {
                hypothesis_id: "H1".into(),
                outcomes: vec![
                    OutcomeProbability {
                        outcome_id: "O1".into(),
                        probability_ppm: 1_000_000,
                    },
                    OutcomeProbability {
                        outcome_id: "O2".into(),
                        probability_ppm: 0,
                    },
                ],
            },
            HypothesisLikelihood {
                hypothesis_id: "H2".into(),
                outcomes: vec![
                    OutcomeProbability {
                        outcome_id: "O1".into(),
                        probability_ppm: 0,
                    },
                    OutcomeProbability {
                        outcome_id: "O2".into(),
                        probability_ppm: 1_000_000,
                    },
                ],
            },
        ];
        model.actions.push(blocked);
        let ranking =
            rank_feasible_actions(&model, AcquisitionObjective::ExpectedInformationGain).unwrap();
        assert_eq!(ranking.ranked_actions.len(), 1);
        assert_eq!(ranking.ranked_actions[0].action_id, "feasible");
        assert_eq!(ranking.rejected_actions.len(), 1);
        assert_eq!(ranking.rejected_actions[0].action_id, "blocked-perfect");
        assert_eq!(ranking.rejected_actions[0].disposition, "Blocked");
    }

    #[test]
    fn missing_likelihood_or_bad_probability_mass_fails_closed() {
        let mut model = model(
            "bad",
            Feasibility::Feasible,
            [900_000, 100_000],
            [100_000, 900_000],
            opposing_decisions(),
        );
        model.actions[0].likelihood_by_hypothesis[0].outcomes[0].probability_ppm = 800_000;
        assert!(model.validate().unwrap_err().contains("do not sum"));
    }

    #[test]
    fn missing_hypothesis_likelihood_produces_no_numeric_acquisition_score() {
        let mut model = model(
            "incomplete",
            Feasibility::Feasible,
            [900_000, 100_000],
            [100_000, 900_000],
            opposing_decisions(),
        );
        model.actions[0].likelihood_by_hypothesis.pop();
        let error = expected_information_gain(&model, "incomplete").unwrap_err();
        assert!(error.contains("one likelihood row per hypothesis"));
    }

    #[test]
    fn conditioning_perfect_observation_sets_posterior_to_one_hypothesis() {
        let model = model(
            "perfect",
            Feasibility::Feasible,
            [1_000_000, 0],
            [0, 1_000_000],
            opposing_decisions(),
        );
        let state = initial_posterior_state(&model).unwrap();
        let update =
            condition_posterior(&model, &state, "evidence-1", "perfect", "O1").unwrap();
        assert_eq!(update.posterior[0].probability_ppm, 1_000_000);
        assert_eq!(update.posterior[1].probability_ppm, 0);
        assert_eq!(
            update.posterior.iter().map(|row| row.probability_ppm).sum::<u32>(),
            1_000_000
        );
        assert_ne!(update.parent_state_digest, update.next_state.state_digest);
        assert_ne!(model.digest().unwrap(), update.posterior_model_digest);
        assert_eq!(update.next_state.evidence_receipts.len(), 1);
    }

    #[test]
    fn conditioning_noisy_observation_preserves_exact_ppm_normalization() {
        let model = model(
            "noisy",
            Feasibility::Feasible,
            [900_000, 100_000],
            [100_000, 900_000],
            opposing_decisions(),
        );
        let state = initial_posterior_state(&model).unwrap();
        let update =
            condition_posterior(&model, &state, "evidence-noisy-1", "noisy", "O1").unwrap();
        assert_eq!(update.posterior[0].probability_ppm, 900_000);
        assert_eq!(update.posterior[1].probability_ppm, 100_000);
    }

    #[test]
    fn impossible_observation_does_not_invent_a_posterior() {
        let model = model(
            "deterministic",
            Feasibility::Feasible,
            [1_000_000, 0],
            [1_000_000, 0],
            opposing_decisions(),
        );
        let state = initial_posterior_state(&model).unwrap();
        assert!(
            condition_posterior(&model, &state, "impossible-1", "deterministic", "O2")
                .unwrap_err()
                .contains("zero probability")
        );
    }

    #[test]
    fn duplicate_evidence_id_cannot_update_posterior_twice() {
        let model = model(
            "noisy",
            Feasibility::Feasible,
            [900_000, 100_000],
            [100_000, 900_000],
            opposing_decisions(),
        );
        let initial = initial_posterior_state(&model).unwrap();
        let first =
            condition_posterior(&model, &initial, "same-measurement", "noisy", "O1").unwrap();
        let error = condition_posterior(
            &model,
            &first.next_state,
            "same-measurement",
            "noisy",
            "O1",
        )
        .unwrap_err();
        assert!(error.contains("duplicate evidence ID"));
    }

    #[test]
    fn altered_posterior_state_is_rejected_before_conditioning() {
        let model = model(
            "noisy",
            Feasibility::Feasible,
            [900_000, 100_000],
            [100_000, 900_000],
            opposing_decisions(),
        );
        let mut state = initial_posterior_state(&model).unwrap();
        state.current_prior[0].probability_ppm -= 1;
        state.current_prior[1].probability_ppm += 1;
        assert!(condition_posterior(&model, &state, "evidence-1", "noisy", "O1")
            .unwrap_err()
            .contains("current prior"));
    }

    #[test]
    fn model_digest_changes_when_likelihood_changes() {
        let mut model = model(
            "noisy",
            Feasibility::Feasible,
            [900_000, 100_000],
            [100_000, 900_000],
            opposing_decisions(),
        );
        let first = model.digest().unwrap();
        model.actions[0].likelihood_by_hypothesis[0].outcomes[0].probability_ppm = 800_000;
        model.actions[0].likelihood_by_hypothesis[0].outcomes[1].probability_ppm = 200_000;
        let second = model.digest().unwrap();
        assert_ne!(first, second);
    }
}
