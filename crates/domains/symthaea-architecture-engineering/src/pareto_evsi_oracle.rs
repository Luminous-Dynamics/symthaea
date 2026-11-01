//! Exact finite-state value-of-sample-information (EVSI) oracle.
//!
//! This module is a reference policy for small, explicitly modelled decision problems.
//! It computes expected Bayes-risk reduction for candidate/objective measurement actions
//! under a finite joint world distribution and a declared observation likelihood model.
//! The world distribution may encode correlations across candidates and objectives.
//!
//! EVSI is model- and loss-dependent. This oracle does not claim that its supplied model is
//! correct, and it is not a scalable Bayesian optimizer. It exists to provide exact answers
//! against which approximate replication controllers can be calibrated.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

const PROBLEM_SCHEMA: &str = "symthaea.architecture-engineering.exact-discrete-evsi-problem";
const PROBLEM_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.exact-discrete-evsi-report";
const REPORT_VERSION: &str = "0.1.0";
const PROBABILITY_TOLERANCE: f64 = 1e-10;
const RISK_TOLERANCE: f64 = 1e-9;
const MAX_WORLDS: usize = 512;
const MAX_DECISIONS: usize = 256;
const MAX_ACTIONS: usize = 512;
const MAX_OUTCOMES_PER_ACTION: usize = 128;
const MAX_LOSS_CELLS: usize = 131_072;
const MAX_LIKELIHOOD_CELLS: usize = 2_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactEvsiError {
    pub path: String,
    pub message: String,
}

impl ExactEvsiError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ExactEvsiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for ExactEvsiError {}

/// Current posterior probability over one latent joint world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvsiWorldProbability {
    pub world_id: String,
    pub posterior_probability: f64,
}

/// Loss incurred when a terminal decision is taken in one latent world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvsiWorldLoss {
    pub world_id: String,
    pub loss: f64,
}

/// An available terminal decision and its explicitly declared loss in every world.
///
/// Decision IDs may encode a Pareto-front classification, a target-attainment decision,
/// or a safety/feasibility outcome. The oracle does not infer the loss function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvsiDecision {
    pub decision_id: String,
    pub losses_by_world: Vec<EvsiWorldLoss>,
}

/// Likelihood of one observed outcome under one latent world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvsiWorldLikelihood {
    pub world_id: String,
    pub probability: f64,
}

/// One possible outcome of a proposed replicate action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvsiActionOutcome {
    pub outcome_id: String,
    pub likelihoods_by_world: Vec<EvsiWorldLikelihood>,
}

/// A candidate/objective measurement that can be purchased for an all-in incremental cost.
///
/// The outcome likelihood table must define a normalized distribution for every latent world.
/// Correlation is represented by the joint world states, not by assuming coordinates are
/// independent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvsiReplicateAction {
    pub action_id: String,
    pub candidate_id: String,
    pub objective_id: String,
    pub incremental_cost_units: u64,
    pub outcomes: Vec<EvsiActionOutcome>,
}

/// Finite decision problem used as an exact EVSI oracle.
///
/// The posterior probabilities are the current beliefs after all already observed evidence.
/// Model IDs and digests freeze the prior/posterior construction, observation likelihoods,
/// decision losses, and all-in action-cost convention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactEvsiProblem {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub posterior_model_id: String,
    pub posterior_model_sha256: String,
    pub observation_model_id: String,
    pub observation_model_sha256: String,
    pub decision_loss_model_id: String,
    pub decision_loss_model_sha256: String,
    pub cost_model_id: String,
    pub cost_model_sha256: String,
    pub worlds: Vec<EvsiWorldProbability>,
    pub decisions: Vec<EvsiDecision>,
    pub actions: Vec<EvsiReplicateAction>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvsiOutcomeValue {
    pub outcome_id: String,
    pub predictive_probability: f64,
    pub posterior_bayes_risk: f64,
    pub posterior_optimal_decision_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvsiActionValue {
    pub action_id: String,
    pub candidate_id: String,
    pub objective_id: String,
    pub incremental_cost_units: u64,
    pub current_bayes_risk: f64,
    pub expected_posterior_bayes_risk: f64,
    pub expected_value_of_sample_information: f64,
    pub value_per_cost_unit: f64,
    pub outcome_values: Vec<EvsiOutcomeValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactEvsiReport {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub problem_sha256: String,
    pub current_bayes_risk: f64,
    pub current_optimal_decision_id: String,
    /// None means every available action has zero decision value under this model/loss.
    pub selected_action_id: Option<String>,
    /// Descending EVSI-per-cost order, then action ID for deterministic ties.
    pub ranked_actions: Vec<EvsiActionValue>,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportDigestPayload<'a> {
    schema: &'a str,
    version: &'a str,
    campaign_id: &'a str,
    problem_sha256: &'a str,
    current_bayes_risk: f64,
    current_optimal_decision_id: &'a str,
    selected_action_id: &'a Option<String>,
    ranked_actions: &'a [EvsiActionValue],
}

impl ExactEvsiProblem {
    /// Canonicalize all ID-indexed tables, then validate the finite decision model.
    ///
    /// Canonical ordering makes identical models hash identically regardless of the order
    /// in which worlds, decisions, actions, outcomes, and likelihood rows were assembled.
    pub fn canonicalize(mut self) -> Result<Self, ExactEvsiError> {
        self.worlds.sort_by(|left, right| left.world_id.cmp(&right.world_id));
        self.decisions
            .sort_by(|left, right| left.decision_id.cmp(&right.decision_id));
        for decision in &mut self.decisions {
            decision
                .losses_by_world
                .sort_by(|left, right| left.world_id.cmp(&right.world_id));
        }
        self.actions
            .sort_by(|left, right| left.action_id.cmp(&right.action_id));
        for action in &mut self.actions {
            action.outcomes.sort_by(|left, right| left.outcome_id.cmp(&right.outcome_id));
            for outcome in &mut action.outcomes {
                outcome
                    .likelihoods_by_world
                    .sort_by(|left, right| left.world_id.cmp(&right.world_id));
            }
        }
        self.validate()?;
        Ok(self)
    }

    /// Validate schema, canonical ordering, probability measures, and bounded exact-work size.
    pub fn validate(&self) -> Result<(), ExactEvsiError> {
        if self.schema != PROBLEM_SCHEMA || self.version != PROBLEM_VERSION {
            return Err(ExactEvsiError::new(
                "problem.schema",
                "unsupported exact-discrete-EVSI schema/version",
            ));
        }
        if self.campaign_id.trim().is_empty() {
            return Err(ExactEvsiError::new("problem.campaign_id", "campaign ID must be non-empty"));
        }
        for (path, id, digest) in [
            ("posterior_model", self.posterior_model_id.as_str(), self.posterior_model_sha256.as_str()),
            ("observation_model", self.observation_model_id.as_str(), self.observation_model_sha256.as_str()),
            ("decision_loss_model", self.decision_loss_model_id.as_str(), self.decision_loss_model_sha256.as_str()),
            ("cost_model", self.cost_model_id.as_str(), self.cost_model_sha256.as_str()),
        ] {
            if id.trim().is_empty() {
                return Err(ExactEvsiError::new(format!("problem.{path}_id"), "model ID must be non-empty"));
            }
            validate_sha256(digest, &format!("problem.{path}_sha256"))?;
        }
        if !(2..=MAX_WORLDS).contains(&self.worlds.len()) {
            return Err(ExactEvsiError::new(
                "problem.worlds",
                format!("exact oracle requires 2..={MAX_WORLDS} finite worlds"),
            ));
        }
        if self.decisions.is_empty() || self.decisions.len() > MAX_DECISIONS {
            return Err(ExactEvsiError::new(
                "problem.decisions",
                format!("decision count must be in 1..={MAX_DECISIONS}"),
            ));
        }
        if self.actions.is_empty() || self.actions.len() > MAX_ACTIONS {
            return Err(ExactEvsiError::new(
                "problem.actions",
                format!("action count must be in 1..={MAX_ACTIONS}"),
            ));
        }
        let loss_cells = self.worlds.len().checked_mul(self.decisions.len()).ok_or_else(|| {
            ExactEvsiError::new("problem.decisions", "loss table size overflow")
        })?;
        if loss_cells > MAX_LOSS_CELLS {
            return Err(ExactEvsiError::new(
                "problem.decisions",
                "loss table exceeds the exact oracle cell ceiling",
            ));
        }

        let mut posterior_total = 0.0;
        let mut posterior_compensation = 0.0;
        let mut previous_world: Option<&str> = None;
        for (index, world) in self.worlds.iter().enumerate() {
            if world.world_id.trim().is_empty() {
                return Err(ExactEvsiError::new(
                    format!("problem.worlds[{index}].world_id"),
                    "world ID must be non-empty",
                ));
            }
            if previous_world.is_some_and(|prior| prior >= world.world_id.as_str()) {
                return Err(ExactEvsiError::new(
                    format!("problem.worlds[{index}].world_id"),
                    "world IDs must be unique and sorted",
                ));
            }
            previous_world = Some(&world.world_id);
            if !world.posterior_probability.is_finite()
                || !(0.0..=1.0).contains(&world.posterior_probability)
            {
                return Err(ExactEvsiError::new(
                    format!("problem.worlds[{index}].posterior_probability"),
                    "posterior probabilities must be finite and lie in [0, 1]",
                ));
            }
            kahan_add(
                &mut posterior_total,
                &mut posterior_compensation,
                world.posterior_probability,
            );
        }
        if (posterior_total - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(ExactEvsiError::new(
                "problem.worlds.posterior_probability",
                format!("posterior probabilities must sum to one; found {posterior_total}"),
            ));
        }

        let mut previous_decision: Option<&str> = None;
        for (decision_index, decision) in self.decisions.iter().enumerate() {
            if decision.decision_id.trim().is_empty()
                || previous_decision.is_some_and(|prior| prior >= decision.decision_id.as_str())
            {
                return Err(ExactEvsiError::new(
                    format!("problem.decisions[{decision_index}].decision_id"),
                    "decision IDs must be non-empty, unique, and sorted",
                ));
            }
            previous_decision = Some(&decision.decision_id);
            self.validate_world_rows(
                &decision.losses_by_world,
                |row| row.world_id.as_str(),
                "loss",
                decision_index,
                "decisions",
            )?;
            for (world_index, loss) in decision.losses_by_world.iter().enumerate() {
                if !loss.loss.is_finite() || loss.loss < 0.0 {
                    return Err(ExactEvsiError::new(
                        format!("problem.decisions[{decision_index}].losses_by_world[{world_index}].loss"),
                        "decision losses must be finite and non-negative",
                    ));
                }
            }
        }

        let mut previous_action: Option<&str> = None;
        let mut likelihood_cells = 0usize;
        for (action_index, action) in self.actions.iter().enumerate() {
            if action.action_id.trim().is_empty()
                || previous_action.is_some_and(|prior| prior >= action.action_id.as_str())
            {
                return Err(ExactEvsiError::new(
                    format!("problem.actions[{action_index}].action_id"),
                    "action IDs must be non-empty, unique, and sorted",
                ));
            }
            previous_action = Some(&action.action_id);
            if action.candidate_id.trim().is_empty() || action.objective_id.trim().is_empty() {
                return Err(ExactEvsiError::new(
                    format!("problem.actions[{action_index}]"),
                    "candidate and objective IDs must be non-empty",
                ));
            }
            if action.incremental_cost_units == 0 {
                return Err(ExactEvsiError::new(
                    format!("problem.actions[{action_index}].incremental_cost_units"),
                    "incremental all-in action cost must be positive",
                ));
            }
            if action.outcomes.is_empty() || action.outcomes.len() > MAX_OUTCOMES_PER_ACTION {
                return Err(ExactEvsiError::new(
                    format!("problem.actions[{action_index}].outcomes"),
                    format!("outcome count must be in 1..={MAX_OUTCOMES_PER_ACTION}"),
                ));
            }
            let action_cells = self.worlds.len().checked_mul(action.outcomes.len()).ok_or_else(|| {
                ExactEvsiError::new(
                    format!("problem.actions[{action_index}].outcomes"),
                    "likelihood table size overflow",
                )
            })?;
            likelihood_cells = likelihood_cells.checked_add(action_cells).ok_or_else(|| {
                ExactEvsiError::new("problem.actions", "total likelihood table size overflow")
            })?;
            if likelihood_cells > MAX_LIKELIHOOD_CELLS {
                return Err(ExactEvsiError::new(
                    "problem.actions",
                    "likelihood tables exceed the exact oracle cell ceiling",
                ));
            }

            let mut previous_outcome: Option<&str> = None;
            for (outcome_index, outcome) in action.outcomes.iter().enumerate() {
                if outcome.outcome_id.trim().is_empty()
                    || previous_outcome.is_some_and(|prior| prior >= outcome.outcome_id.as_str())
                {
                    return Err(ExactEvsiError::new(
                        format!("problem.actions[{action_index}].outcomes[{outcome_index}].outcome_id"),
                        "outcome IDs must be non-empty, unique, and sorted",
                    ));
                }
                previous_outcome = Some(&outcome.outcome_id);
                self.validate_world_rows(
                    &outcome.likelihoods_by_world,
                    |row| row.world_id.as_str(),
                    "likelihood",
                    action_index * MAX_OUTCOMES_PER_ACTION + outcome_index,
                    "actions",
                )?;
                for (world_index, likelihood) in outcome.likelihoods_by_world.iter().enumerate() {
                    if !likelihood.probability.is_finite()
                        || !(0.0..=1.0).contains(&likelihood.probability)
                    {
                        return Err(ExactEvsiError::new(
                            format!("problem.actions[{action_index}].outcomes[{outcome_index}].likelihoods_by_world[{world_index}].probability"),
                            "likelihood probabilities must be finite and lie in [0, 1]",
                        ));
                    }
                }
            }
            for world_index in 0..self.worlds.len() {
                let mut total = 0.0;
                let mut compensation = 0.0;
                for outcome in &action.outcomes {
                    kahan_add(
                        &mut total,
                        &mut compensation,
                        outcome.likelihoods_by_world[world_index].probability,
                    );
                }
                if (total - 1.0).abs() > PROBABILITY_TOLERANCE {
                    return Err(ExactEvsiError::new(
                        format!("problem.actions[{action_index}].outcomes"),
                        format!(
                            "outcome likelihoods must sum to one for world {}; found {total}",
                            self.worlds[world_index].world_id
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_world_rows<T>(
        &self,
        rows: &[T],
        world_id: impl for<'a> Fn(&'a T) -> &'a str,
        row_kind: &str,
        parent_index: usize,
        parent_name: &str,
    ) -> Result<(), ExactEvsiError> {
        if rows.len() != self.worlds.len() {
            return Err(ExactEvsiError::new(
                format!("problem.{parent_name}[{parent_index}]"),
                format!("{row_kind} table must contain exactly one row for every world"),
            ));
        }
        for (index, world) in self.worlds.iter().enumerate() {
            if world_id(&rows[index]) != world.world_id {
                return Err(ExactEvsiError::new(
                    format!("problem.{parent_name}[{parent_index}].{row_kind}s_by_world[{index}]"),
                    "rows must exactly match canonical world ID order",
                ));
            }
        }
        Ok(())
    }

    pub fn problem_sha256(&self) -> Result<String, ExactEvsiError> {
        self.validate()?;
        digest_serializable(self)
    }
}

/// Compute exact expected value of sample information for every available action.
///
/// For each action, enumerate every possible observation, form the exact finite posterior,
/// calculate its minimum expected declared decision loss, and average those risks under the
/// predictive outcome distribution. The action value is current Bayes risk minus expected
/// posterior Bayes risk. Selection ranks by EVSI per all-in incremental cost unit.
pub fn analyze_exact_evsi(
    problem: &ExactEvsiProblem,
) -> Result<ExactEvsiReport, ExactEvsiError> {
    problem.validate()?;
    let world_index: BTreeMap<&str, usize> = problem
        .worlds
        .iter()
        .enumerate()
        .map(|(index, world)| (world.world_id.as_str(), index))
        .collect();
    let posterior: Vec<f64> = problem
        .worlds
        .iter()
        .map(|world| world.posterior_probability)
        .collect();
    let (current_bayes_risk, current_optimal_decision_id) =
        minimum_bayes_risk(problem, &posterior)?;

    let mut ranked_actions = Vec::with_capacity(problem.actions.len());
    for action in &problem.actions {
        let mut expected_posterior_bayes_risk = 0.0;
        let mut risk_compensation = 0.0;
        let mut outcome_values = Vec::with_capacity(action.outcomes.len());

        for outcome in &action.outcomes {
            let mut predictive_probability = 0.0;
            let mut predictive_compensation = 0.0;
            let mut likelihood_vector = vec![0.0; problem.worlds.len()];
            for row in &outcome.likelihoods_by_world {
                let index = *world_index.get(row.world_id.as_str()).ok_or_else(|| {
                    ExactEvsiError::new(
                        "problem.actions.outcomes.likelihoods_by_world",
                        "validated likelihood row references an unknown world",
                    )
                })?;
                likelihood_vector[index] = row.probability;
                kahan_add(
                    &mut predictive_probability,
                    &mut predictive_compensation,
                    posterior[index] * row.probability,
                );
            }

            let (posterior_bayes_risk, posterior_optimal_decision_id) =
                if predictive_probability <= 0.0 {
                    (0.0, "zero-probability-outcome".to_owned())
                } else {
                    let posterior_after_outcome: Vec<f64> = posterior
                        .iter()
                        .zip(&likelihood_vector)
                        .map(|(prior, likelihood)| prior * likelihood / predictive_probability)
                        .collect();
                    minimum_bayes_risk(problem, &posterior_after_outcome)?
                };
            kahan_add(
                &mut expected_posterior_bayes_risk,
                &mut risk_compensation,
                predictive_probability * posterior_bayes_risk,
            );
            outcome_values.push(EvsiOutcomeValue {
                outcome_id: outcome.outcome_id.clone(),
                predictive_probability,
                posterior_bayes_risk,
                posterior_optimal_decision_id,
            });
        }

        if expected_posterior_bayes_risk > current_bayes_risk + RISK_TOLERANCE {
            return Err(ExactEvsiError::new(
                format!("problem.actions[{}]", action.action_id),
                "expected posterior Bayes risk exceeds current Bayes risk; likelihood or arithmetic contract is invalid",
            ));
        }
        let expected_value_of_sample_information =
            (current_bayes_risk - expected_posterior_bayes_risk).max(0.0);
        let value_per_cost_unit =
            expected_value_of_sample_information / action.incremental_cost_units as f64;
        if !expected_value_of_sample_information.is_finite() || !value_per_cost_unit.is_finite() {
            return Err(ExactEvsiError::new(
                format!("problem.actions[{}]", action.action_id),
                "EVSI or value-per-cost calculation is non-finite",
            ));
        }
        ranked_actions.push(EvsiActionValue {
            action_id: action.action_id.clone(),
            candidate_id: action.candidate_id.clone(),
            objective_id: action.objective_id.clone(),
            incremental_cost_units: action.incremental_cost_units,
            current_bayes_risk,
            expected_posterior_bayes_risk,
            expected_value_of_sample_information,
            value_per_cost_unit,
            outcome_values,
        });
    }

    ranked_actions.sort_by(|left, right| {
        right
            .value_per_cost_unit
            .total_cmp(&left.value_per_cost_unit)
            .then_with(|| left.action_id.cmp(&right.action_id))
    });
    let selected_action_id = ranked_actions
        .first()
        .filter(|action| action.expected_value_of_sample_information > 0.0)
        .map(|action| action.action_id.clone());
    let problem_sha256 = problem.problem_sha256()?;
    let payload = ReportDigestPayload {
        schema: REPORT_SCHEMA,
        version: REPORT_VERSION,
        campaign_id: &problem.campaign_id,
        problem_sha256: &problem_sha256,
        current_bayes_risk,
        current_optimal_decision_id: &current_optimal_decision_id,
        selected_action_id: &selected_action_id,
        ranked_actions: &ranked_actions,
    };
    let report_sha256 = digest_serializable(&payload)?;
    Ok(ExactEvsiReport {
        schema: REPORT_SCHEMA.to_owned(),
        version: REPORT_VERSION.to_owned(),
        campaign_id: problem.campaign_id.clone(),
        problem_sha256,
        current_bayes_risk,
        current_optimal_decision_id,
        selected_action_id,
        ranked_actions,
        report_sha256,
    })
}

/// Recompute an exact EVSI report from its finite model; altered reports fail closed.
pub fn verify_exact_evsi_report(
    problem: &ExactEvsiProblem,
    report: &ExactEvsiReport,
) -> Result<(), ExactEvsiError> {
    let reconstructed = analyze_exact_evsi(problem)?;
    if &reconstructed != report {
        return Err(ExactEvsiError::new(
            "report",
            "stored EVSI report differs from exact reconstruction from the frozen finite model",
        ));
    }
    Ok(())
}

fn minimum_bayes_risk(
    problem: &ExactEvsiProblem,
    posterior: &[f64],
) -> Result<(f64, String), ExactEvsiError> {
    if posterior.len() != problem.worlds.len() {
        return Err(ExactEvsiError::new(
            "posterior",
            "posterior vector length differs from the finite world space",
        ));
    }
    let mut best: Option<(f64, String)> = None;
    for decision in &problem.decisions {
        let mut risk = 0.0;
        let mut compensation = 0.0;
        for (world_index, loss) in decision.losses_by_world.iter().enumerate() {
            kahan_add(
                &mut risk,
                &mut compensation,
                posterior[world_index] * loss.loss,
            );
        }
        match &best {
            None => best = Some((risk, decision.decision_id.clone())),
            Some((best_risk, best_id))
                if risk < *best_risk
                    || (risk == *best_risk && decision.decision_id < *best_id) =>
            {
                best = Some((risk, decision.decision_id.clone()));
            }
            _ => {}
        }
    }
    best.ok_or_else(|| ExactEvsiError::new("decisions", "no terminal decisions are available"))
}

fn kahan_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let adjusted = value - *compensation;
    let next = *sum + adjusted;
    *compensation = (next - *sum) - adjusted;
    *sum = next;
}

fn validate_sha256(value: &str, path: &str) -> Result<(), ExactEvsiError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ExactEvsiError::new(
            path,
            "expected lowercase 64-character SHA-256 digest",
        ));
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, ExactEvsiError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        ExactEvsiError::new(
            "serialization",
            format!("cannot serialize exact EVSI receipt: {error}"),
        )
    })?;
    Ok(sha256_bytes(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H4: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn world_rows(values: [f64; 2]) -> Vec<EvsiWorldLoss> {
        vec![
            EvsiWorldLoss { world_id: "world-a".to_owned(), loss: values[0] },
            EvsiWorldLoss { world_id: "world-b".to_owned(), loss: values[1] },
        ]
    }

    fn likelihoods(a: f64, b: f64) -> Vec<EvsiWorldLikelihood> {
        vec![
            EvsiWorldLikelihood { world_id: "world-a".to_owned(), probability: a },
            EvsiWorldLikelihood { world_id: "world-b".to_owned(), probability: b },
        ]
    }

    fn deterministic_action(action_id: &str, candidate_id: &str, objective_id: &str, cost: u64) -> EvsiReplicateAction {
        EvsiReplicateAction {
            action_id: action_id.to_owned(),
            candidate_id: candidate_id.to_owned(),
            objective_id: objective_id.to_owned(),
            incremental_cost_units: cost,
            outcomes: vec![
                EvsiActionOutcome {
                    outcome_id: "observed-a".to_owned(),
                    likelihoods_by_world: likelihoods(1.0, 0.0),
                },
                EvsiActionOutcome {
                    outcome_id: "observed-b".to_owned(),
                    likelihoods_by_world: likelihoods(0.0, 1.0),
                },
            ],
        }
    }

    fn problem() -> ExactEvsiProblem {
        ExactEvsiProblem {
            schema: PROBLEM_SCHEMA.to_owned(),
            version: PROBLEM_VERSION.to_owned(),
            campaign_id: "evsi-two-world-test".to_owned(),
            posterior_model_id: "finite-posterior-v1".to_owned(),
            posterior_model_sha256: H1.to_owned(),
            observation_model_id: "deterministic-and-uninformative-observations-v1".to_owned(),
            observation_model_sha256: H2.to_owned(),
            decision_loss_model_id: "binary-classification-loss-v1".to_owned(),
            decision_loss_model_sha256: H3.to_owned(),
            cost_model_id: "integer-all-in-cost-units-v1".to_owned(),
            cost_model_sha256: H4.to_owned(),
            worlds: vec![
                EvsiWorldProbability { world_id: "world-b".to_owned(), posterior_probability: 0.5 },
                EvsiWorldProbability { world_id: "world-a".to_owned(), posterior_probability: 0.5 },
            ],
            decisions: vec![
                EvsiDecision {
                    decision_id: "retain-a".to_owned(),
                    losses_by_world: world_rows([0.0, 10.0]),
                },
                EvsiDecision {
                    decision_id: "retain-b".to_owned(),
                    losses_by_world: vec![
                        EvsiWorldLoss { world_id: "world-b".to_owned(), loss: 0.0 },
                        EvsiWorldLoss { world_id: "world-a".to_owned(), loss: 10.0 },
                    ],
                },
            ],
            actions: vec![
                EvsiReplicateAction {
                    action_id: "uninformative".to_owned(),
                    candidate_id: "candidate-b".to_owned(),
                    objective_id: "cost".to_owned(),
                    incremental_cost_units: 1,
                    outcomes: vec![
                        EvsiActionOutcome { outcome_id: "noise-0".to_owned(), likelihoods_by_world: likelihoods(0.5, 0.5) },
                        EvsiActionOutcome { outcome_id: "noise-1".to_owned(), likelihoods_by_world: likelihoods(0.5, 0.5) },
                    ],
                },
                deterministic_action("measure-a-quality", "candidate-a", "quality", 1),
                deterministic_action("measure-a-quality-expensive", "candidate-a", "quality", 2),
            ],
        }
        .canonicalize()
        .unwrap()
    }

    #[test]
    fn exact_evsi_selects_the_measurement_with_highest_value_per_all_in_cost() {
        let problem = problem();
        let report = analyze_exact_evsi(&problem).unwrap();
        assert_eq!(report.current_optimal_decision_id, "retain-a");
        assert!((report.current_bayes_risk - 5.0).abs() < 1e-12);

        let best = &report.ranked_actions[0];
        assert_eq!(best.action_id, "measure-a-quality");
        assert!((best.expected_value_of_sample_information - 5.0).abs() < 1e-12);
        assert!((best.expected_posterior_bayes_risk - 0.0).abs() < 1e-12);
        assert!((best.value_per_cost_unit - 5.0).abs() < 1e-12);
        assert_eq!(report.selected_action_id.as_deref(), Some("measure-a-quality"));
        verify_exact_evsi_report(&problem, &report).unwrap();
    }

    #[test]
    fn uninformative_observation_has_zero_evsi_and_does_not_displace_informative_action() {
        let report = analyze_exact_evsi(&problem()).unwrap();
        let action = report.ranked_actions.iter().find(|item| item.action_id == "uninformative").unwrap();
        assert!((action.expected_value_of_sample_information - 0.0).abs() < 1e-12);
        assert!((action.expected_posterior_bayes_risk - action.current_bayes_risk).abs() < 1e-12);
        assert_ne!(report.selected_action_id.as_deref(), Some("uninformative"));
    }

    #[test]
    fn all_in_cost_changes_the_action_ranking() {
        let report = analyze_exact_evsi(&problem()).unwrap();
        let cheap = report.ranked_actions.iter().find(|item| item.action_id == "measure-a-quality").unwrap();
        let expensive = report.ranked_actions.iter().find(|item| item.action_id == "measure-a-quality-expensive").unwrap();
        assert!((cheap.expected_value_of_sample_information - expensive.expected_value_of_sample_information).abs() < 1e-12);
        assert!(cheap.value_per_cost_unit > expensive.value_per_cost_unit);
    }

    #[test]
    fn correlated_worlds_are_supported_by_joint_likelihood_tables() {
        let mut model = problem();
        let action = model.actions.iter_mut().find(|item| item.action_id == "uninformative").unwrap();
        action.outcomes = vec![
            EvsiActionOutcome { outcome_id: "shared-low".to_owned(), likelihoods_by_world: likelihoods(0.9, 0.1) },
            EvsiActionOutcome { outcome_id: "shared-high".to_owned(), likelihoods_by_world: likelihoods(0.1, 0.9) },
        ];
        model = model.canonicalize().unwrap();
        let report = analyze_exact_evsi(&model).unwrap();
        assert!(report.ranked_actions.iter().any(|item| item.action_id == "uninformative"
            && item.expected_value_of_sample_information > 0.0));
    }

    #[test]
    fn malformed_probability_measures_fail_closed() {
        let mut model = problem();
        let action = model.actions.iter_mut().find(|item| item.action_id == "uninformative").unwrap();
        action.outcomes[0].likelihoods_by_world[0].probability = 0.7;
        assert!(model.validate().is_err());

        let mut negative = problem();
        negative.worlds[0].posterior_probability = -0.1;
        assert!(negative.validate().is_err());

        let mut nonfinite = problem();
        nonfinite.actions[0].outcomes[0].likelihoods_by_world[0].probability = f64::NAN;
        assert!(nonfinite.validate().is_err());
    }

    #[test]
    fn decision_loss_and_world_tables_must_cover_exact_canonical_world_set() {
        let mut missing = problem();
        missing.decisions[0].losses_by_world.pop();
        assert!(missing.validate().is_err());

        let mut wrong_world = problem();
        wrong_world.decisions[0].losses_by_world[0].world_id = "unknown-world".to_owned();
        assert!(wrong_world.validate().is_err());
    }

    #[test]
    fn canonical_problem_hash_is_independent_of_input_row_order() {
        let baseline = problem();
        let mut reordered = baseline.clone();
        reordered.worlds.reverse();
        reordered.decisions.reverse();
        reordered.actions.reverse();
        for decision in &mut reordered.decisions {
            decision.losses_by_world.reverse();
        }
        for action in &mut reordered.actions {
            action.outcomes.reverse();
            for outcome in &mut action.outcomes {
                outcome.likelihoods_by_world.reverse();
            }
        }
        let canonical = reordered.canonicalize().unwrap();
        assert_eq!(baseline.problem_sha256().unwrap(), canonical.problem_sha256().unwrap());
    }

    #[test]
    fn report_tampering_is_detected_and_zero_value_policy_abstains() {
        let problem = problem();
        let mut report = analyze_exact_evsi(&problem).unwrap();
        report.ranked_actions[0].expected_value_of_sample_information += 1.0;
        assert!(verify_exact_evsi_report(&problem, &report).is_err());

        let mut no_information = problem.clone();
        for action in &mut no_information.actions {
            action.outcomes = vec![
                EvsiActionOutcome { outcome_id: "constant".to_owned(), likelihoods_by_world: likelihoods(1.0, 1.0) },
            ];
        }
        let no_information = no_information.canonicalize().unwrap();
        let report = analyze_exact_evsi(&no_information).unwrap();
        assert_eq!(report.selected_action_id, None);
    }

    #[test]
    fn cost_and_model_identity_are_bound_to_the_problem_digest() {
        let baseline = problem().problem_sha256().unwrap();
        let mut changed_cost = problem();
        changed_cost.actions[0].incremental_cost_units += 1;
        assert_ne!(baseline, changed_cost.problem_sha256().unwrap());

        let mut changed_model = problem();
        changed_model.cost_model_sha256 = "9999999999999999999999999999999999999999999999999999999999999999".to_owned();
        assert_ne!(baseline, changed_model.problem_sha256().unwrap());
    }
}
