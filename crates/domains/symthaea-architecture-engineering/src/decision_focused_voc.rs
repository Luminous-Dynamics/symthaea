//! Decision-focused value-of-computation planning for multi-fidelity NAS.
//!
//! Unlike interval-width heuristics, this module evaluates actions by their expected reduction
//! in Bayes top-K simple regret under an explicitly supplied, task-qualified discrete posterior
//! model. Each posterior particle contains latent final-fidelity utility for every candidate.
//! Each legal action supplies a normalized observation-likelihood model over those particles.
//!
//! The planner computes the exact expected posterior Bayes regret for each action, subtracts a
//! predeclared regret-equivalent resource cost, enforces hard budget/reserve constraints, and
//! returns either one action or an explicit STOP decision.
//!
//! This is decision-theoretic planning under the supplied posterior model, not a frequentist
//! certificate. Its quality depends on the model's task-disjoint calibration and likelihood
//! adequacy. The run-wide pruning ledger remains a separate certificate and must gate pruning.
//! SentinelAudit likelihood outcomes must be aggregate dispositions only; raw sentinel identities
//! and scores must remain inside the evaluator.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use sha2::{Digest, Sha256};

use super::tabular_nas_firewall::MetricDirection;

const VOC_DOMAIN: &[u8] = b"symthaea.nas.decision-focused-voc.v0.1";

#[derive(Debug, Clone, PartialEq)]
pub struct FinalPerformanceParticle {
    pub particle_id: String,
    /// Prior posterior mass. The complete vector must sum to one.
    pub probability: f64,
    /// Final-fidelity metric values in PosteriorModel.candidate_ids order.
    pub final_values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PosteriorModel {
    pub candidate_ids: Vec<String>,
    pub particles: Vec<FinalPerformanceParticle>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionFocusedVoCModelManifest {
    pub benchmark_id: String,
    pub target_metric_id: String,
    pub metric_direction: MetricDirection,
    pub evaluation_task_id: String,
    pub calibration_task_ids: Vec<String>,
    pub source_artifact_sha256: String,
    pub posterior_model_sha256: String,
    pub observation_likelihood_model_sha256: String,
    /// Set only after the external task-transfer/calibration gate has qualified this exact model.
    /// The planner verifies all associated IDs and digests but does not independently reproduce
    /// the upstream calibration experiment.
    pub task_calibration_qualified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VoCActionKind {
    FidelityQuery,
    DormantReactivation,
    SentinelAudit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationLikelihood {
    pub outcome_id: String,
    /// P(outcome | posterior particle j), one value for every particle in the model.
    pub likelihood_by_particle: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoCAction {
    pub action_id: String,
    pub kind: VoCActionKind,
    /// Required for FidelityQuery and DormantReactivation; absent for aggregate SentinelAudit.
    pub candidate_id: Option<String>,
    pub resource_unit: String,
    pub incremental_cost_units: u64,
    pub source_artifact_sha256: String,
    pub posterior_model_sha256: String,
    pub observation_likelihood_model_sha256: String,
    /// Mutually exclusive aggregate outcomes; likelihoods sum to one for every particle.
    pub observation_outcomes: Vec<ObservationLikelihood>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoCActionReserve {
    pub kind: VoCActionKind,
    /// Minimum resource units to preserve for this action class until spent by that class.
    pub reserve_units: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoCSpendByKind {
    pub kind: VoCActionKind,
    pub spent_units: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionFocusedVoCConfig {
    pub target_k: usize,
    pub resource_unit: String,
    pub total_budget_units: u64,
    pub already_spent_units: u64,
    /// Regret-equivalent penalty per resource unit. This is an explicit decision preference,
    /// not a statistically estimated parameter. Report Pareto frontiers when no defensible
    /// common resource valuation exists.
    pub cost_penalty_per_unit: f64,
    pub reserves: Vec<VoCActionReserve>,
    pub spend_by_kind: Vec<VoCSpendByKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoCActionStatus {
    Feasible,
    BlockedByTotalBudget,
    BlockedByProtectedReserve,
    NotWorthItsCost,
    Selected,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoCActionEvaluation {
    pub action_id: String,
    pub kind: VoCActionKind,
    pub candidate_id: Option<String>,
    pub incremental_cost_units: u64,
    pub expected_posterior_regret: f64,
    pub expected_regret_reduction: f64,
    pub regret_reduction_per_cost_unit: f64,
    pub regret_equivalent_cost: f64,
    pub net_expected_utility: f64,
    pub status: VoCActionStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoCStopReason {
    NoLegalActions,
    NoActionFitsResourceAndReserveBudgets,
    NoPositiveNetExpectedUtility,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionFocusedVoCPlan {
    pub benchmark_id: String,
    pub target_metric_id: String,
    pub evaluation_task_id: String,
    pub target_k: usize,
    pub resource_unit: String,
    pub total_budget_units: u64,
    pub already_spent_units: u64,
    pub current_expected_regret: f64,
    pub cost_penalty_per_unit: f64,
    pub selected_action_id: Option<String>,
    pub stop_reason: Option<VoCStopReason>,
    pub action_evaluations: Vec<VoCActionEvaluation>,
    pub source_artifact_sha256: String,
    pub posterior_model_sha256: String,
    pub observation_likelihood_model_sha256: String,
    pub plan_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionFocusedVoCError {
    pub field: String,
    pub message: String,
}

impl DecisionFocusedVoCError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for DecisionFocusedVoCError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for DecisionFocusedVoCError {}

/// Select the legal experiment with the highest positive expected net reduction in top-K regret.
///
/// The posterior particle set is joint across candidates, so correlations between architectures
/// are preserved. This matters: evaluating one candidate can change the relative ranking of
/// several candidates even when their marginal predictive intervals barely change.
pub fn plan_decision_focused_voc(
    manifest: &DecisionFocusedVoCModelManifest,
    posterior: &PosteriorModel,
    actions: &[VoCAction],
    config: &DecisionFocusedVoCConfig,
) -> Result<DecisionFocusedVoCPlan, DecisionFocusedVoCError> {
    validate_manifest(manifest)?;
    validate_posterior(posterior)?;
    validate_config(config, posterior.candidate_ids.len())?;

    if posterior.candidate_ids.len() < config.target_k {
        return Err(DecisionFocusedVoCError::new(
            "target_k",
            "target K exceeds the number of candidates in the posterior",
        ));
    }

    let particle_probabilities: Vec<f64> =
        posterior.particles.iter().map(|particle| particle.probability).collect();
    let utility_particles = utility_particles(posterior, manifest.metric_direction);
    let current_expected_regret = bayes_top_k_regret(
        &utility_particles,
        &particle_probabilities,
        config.target_k,
    )?;

    let mut seen_action_ids = HashSet::with_capacity(actions.len());
    let mut evaluations = Vec::with_capacity(actions.len());
    for (index, action) in actions.iter().enumerate() {
        validate_action(
            action,
            index,
            manifest,
            posterior,
            config,
            &mut seen_action_ids,
        )?;

        let expected_posterior_regret = expected_regret_after_action(
            &utility_particles,
            &particle_probabilities,
            &action.observation_outcomes,
            config.target_k,
        )?;
        let raw_reduction = current_expected_regret - expected_posterior_regret;
        if raw_reduction < -1e-9 {
            return Err(DecisionFocusedVoCError::new(
                format!("actions[{index}].observation_outcomes"),
                "the supplied likelihood model makes expected Bayes regret increase after observing information",
            ));
        }
        let expected_regret_reduction = raw_reduction.max(0.0);
        let cost = action.incremental_cost_units as f64;
        let regret_equivalent_cost = config.cost_penalty_per_unit * cost;
        let net_expected_utility = expected_regret_reduction - regret_equivalent_cost;
        let status = if !fits_total_budget(action, config) {
            VoCActionStatus::BlockedByTotalBudget
        } else if !fits_reserve_policy(action, config) {
            VoCActionStatus::BlockedByProtectedReserve
        } else if net_expected_utility <= 0.0 {
            VoCActionStatus::NotWorthItsCost
        } else {
            VoCActionStatus::Feasible
        };

        evaluations.push(VoCActionEvaluation {
            action_id: action.action_id.clone(),
            kind: action.kind,
            candidate_id: action.candidate_id.clone(),
            incremental_cost_units: action.incremental_cost_units,
            expected_posterior_regret,
            expected_regret_reduction,
            regret_reduction_per_cost_unit: expected_regret_reduction / cost,
            regret_equivalent_cost,
            net_expected_utility,
            status,
        });
    }

    // Deterministic tie-breaking: highest net utility, then highest regret reduction per cost,
    // then lexicographically smallest action ID.
    let selected_index = evaluations
        .iter()
        .enumerate()
        .filter(|(_, evaluation)| evaluation.status == VoCActionStatus::Feasible)
        .max_by(|(_, left), (_, right)| {
            left.net_expected_utility
                .total_cmp(&right.net_expected_utility)
                .then_with(|| {
                    left.regret_reduction_per_cost_unit
                        .total_cmp(&right.regret_reduction_per_cost_unit)
                })
                // max_by chooses Greater; reverse IDs so lexical smaller wins.
                .then_with(|| right.action_id.cmp(&left.action_id))
        })
        .map(|(index, _)| index);

    let (selected_action_id, stop_reason) = if let Some(index) = selected_index {
        evaluations[index].status = VoCActionStatus::Selected;
        (Some(evaluations[index].action_id.clone()), None)
    } else {
        let reason = if actions.is_empty() {
            VoCStopReason::NoLegalActions
        } else if evaluations.iter().all(|evaluation| {
            matches!(
                evaluation.status,
                VoCActionStatus::BlockedByTotalBudget | VoCActionStatus::BlockedByProtectedReserve
            )
        }) {
            VoCStopReason::NoActionFitsResourceAndReserveBudgets
        } else {
            VoCStopReason::NoPositiveNetExpectedUtility
        };
        (None, Some(reason))
    };

    let plan_sha256 = digest_plan(
        manifest,
        posterior,
        config,
        current_expected_regret,
        &evaluations,
        selected_action_id.as_deref(),
        stop_reason,
    );

    Ok(DecisionFocusedVoCPlan {
        benchmark_id: manifest.benchmark_id.clone(),
        target_metric_id: manifest.target_metric_id.clone(),
        evaluation_task_id: manifest.evaluation_task_id.clone(),
        target_k: config.target_k,
        resource_unit: config.resource_unit.clone(),
        total_budget_units: config.total_budget_units,
        already_spent_units: config.already_spent_units,
        current_expected_regret,
        cost_penalty_per_unit: config.cost_penalty_per_unit,
        selected_action_id,
        stop_reason,
        action_evaluations: evaluations,
        source_artifact_sha256: manifest.source_artifact_sha256.clone(),
        posterior_model_sha256: manifest.posterior_model_sha256.clone(),
        observation_likelihood_model_sha256: manifest.observation_likelihood_model_sha256.clone(),
        plan_sha256,
    })
}

fn validate_manifest(
    manifest: &DecisionFocusedVoCModelManifest,
) -> Result<(), DecisionFocusedVoCError> {
    for (field, value) in [
        ("benchmark_id", manifest.benchmark_id.as_str()),
        ("target_metric_id", manifest.target_metric_id.as_str()),
        ("evaluation_task_id", manifest.evaluation_task_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(DecisionFocusedVoCError::new(field, "must be non-empty"));
        }
    }
    validate_sha256(&manifest.source_artifact_sha256, "source_artifact_sha256")?;
    validate_sha256(&manifest.posterior_model_sha256, "posterior_model_sha256")?;
    validate_sha256(
        &manifest.observation_likelihood_model_sha256,
        "observation_likelihood_model_sha256",
    )?;
    if !manifest.task_calibration_qualified {
        return Err(DecisionFocusedVoCError::new(
            "task_calibration_qualified",
            "the likelihood/posterior model has not passed its external task-calibration gate",
        ));
    }
    if manifest.calibration_task_ids.is_empty() {
        return Err(DecisionFocusedVoCError::new(
            "calibration_task_ids",
            "at least one disjoint calibration task is required",
        ));
    }
    let mut task_ids = HashSet::with_capacity(manifest.calibration_task_ids.len());
    for (index, task_id) in manifest.calibration_task_ids.iter().enumerate() {
        if task_id.trim().is_empty() || !task_ids.insert(task_id.as_str()) {
            return Err(DecisionFocusedVoCError::new(
                format!("calibration_task_ids[{index}]"),
                "calibration task IDs must be non-empty and unique",
            ));
        }
        if task_id == &manifest.evaluation_task_id {
            return Err(DecisionFocusedVoCError::new(
                format!("calibration_task_ids[{index}]"),
                "evaluation task must be disjoint from all calibration tasks",
            ));
        }
    }
    Ok(())
}

fn validate_posterior(posterior: &PosteriorModel) -> Result<(), DecisionFocusedVoCError> {
    if posterior.candidate_ids.is_empty() {
        return Err(DecisionFocusedVoCError::new(
            "candidate_ids",
            "at least one candidate is required",
        ));
    }
    let mut candidate_ids = HashSet::with_capacity(posterior.candidate_ids.len());
    for (index, candidate_id) in posterior.candidate_ids.iter().enumerate() {
        if candidate_id.trim().is_empty() || !candidate_ids.insert(candidate_id.as_str()) {
            return Err(DecisionFocusedVoCError::new(
                format!("candidate_ids[{index}]"),
                "candidate IDs must be non-empty and unique",
            ));
        }
    }
    if posterior.particles.is_empty() {
        return Err(DecisionFocusedVoCError::new(
            "particles",
            "at least one posterior particle is required",
        ));
    }
    let mut particle_ids = HashSet::with_capacity(posterior.particles.len());
    let mut total_probability = 0.0;
    for (index, particle) in posterior.particles.iter().enumerate() {
        if particle.particle_id.trim().is_empty() || !particle_ids.insert(particle.particle_id.as_str()) {
            return Err(DecisionFocusedVoCError::new(
                format!("particles[{index}].particle_id"),
                "particle IDs must be non-empty and unique",
            ));
        }
        if !particle.probability.is_finite() || particle.probability < 0.0 {
            return Err(DecisionFocusedVoCError::new(
                format!("particles[{index}].probability"),
                "probability must be finite and non-negative",
            ));
        }
        if particle.final_values.len() != posterior.candidate_ids.len()
            || particle.final_values.iter().any(|value| !value.is_finite())
        {
            return Err(DecisionFocusedVoCError::new(
                format!("particles[{index}].final_values"),
                "final values must be finite and cover every candidate exactly once in canonical order",
            ));
        }
        total_probability += particle.probability;
    }
    if !approximately_equal(total_probability, 1.0) {
        return Err(DecisionFocusedVoCError::new(
            "particles.probability",
            "posterior particle probabilities must sum to one",
        ));
    }
    Ok(())
}

fn validate_config(
    config: &DecisionFocusedVoCConfig,
    candidate_count: usize,
) -> Result<(), DecisionFocusedVoCError> {
    if config.target_k == 0 || config.target_k > candidate_count {
        return Err(DecisionFocusedVoCError::new(
            "target_k",
            "must be in 1..=candidate_count",
        ));
    }
    if config.resource_unit.trim().is_empty() {
        return Err(DecisionFocusedVoCError::new("resource_unit", "must be non-empty"));
    }
    if config.total_budget_units == 0 || config.already_spent_units > config.total_budget_units {
        return Err(DecisionFocusedVoCError::new(
            "total_budget_units",
            "total budget must be positive and already-spent cost cannot exceed it",
        ));
    }
    if !config.cost_penalty_per_unit.is_finite() || config.cost_penalty_per_unit < 0.0 {
        return Err(DecisionFocusedVoCError::new(
            "cost_penalty_per_unit",
            "must be finite and non-negative",
        ));
    }
    let mut reserve_kinds = HashSet::new();
    for (index, reserve) in config.reserves.iter().enumerate() {
        if !reserve_kinds.insert(reserve.kind) {
            return Err(DecisionFocusedVoCError::new(
                format!("reserves[{index}].kind"),
                "each action kind may have only one protected reserve",
            ));
        }
    }
    let mut spend_kinds = HashSet::new();
    for (index, spend) in config.spend_by_kind.iter().enumerate() {
        if !spend_kinds.insert(spend.kind) {
            return Err(DecisionFocusedVoCError::new(
                format!("spend_by_kind[{index}].kind"),
                "each action kind may have only one accumulated spend record",
            ));
        }
    }
    let spent_by_kind: u64 = config.spend_by_kind.iter()
        .try_fold(0u64, |total, spend| total.checked_add(spend.spent_units))
        .ok_or_else(|| DecisionFocusedVoCError::new("spend_by_kind", "resource cost overflow"))?;
    if spent_by_kind > config.already_spent_units {
        return Err(DecisionFocusedVoCError::new(
            "spend_by_kind",
            "classified action spend cannot exceed total already-spent units",
        ));
    }
    Ok(())
}

fn validate_action(
    action: &VoCAction,
    index: usize,
    manifest: &DecisionFocusedVoCModelManifest,
    posterior: &PosteriorModel,
    config: &DecisionFocusedVoCConfig,
    seen_action_ids: &mut HashSet<String>,
) -> Result<(), DecisionFocusedVoCError> {
    if action.action_id.trim().is_empty() || !seen_action_ids.insert(action.action_id.clone()) {
        return Err(DecisionFocusedVoCError::new(
            format!("actions[{index}].action_id"),
            "action IDs must be non-empty and unique",
        ));
    }
    if action.resource_unit != config.resource_unit {
        return Err(DecisionFocusedVoCError::new(
            format!("actions[{index}].resource_unit"),
            "all candidate actions must use the declared resource unit",
        ));
    }
    if action.incremental_cost_units == 0 {
        return Err(DecisionFocusedVoCError::new(
            format!("actions[{index}].incremental_cost_units"),
            "action cost must be positive",
        ));
    }
    if action.source_artifact_sha256 != manifest.source_artifact_sha256
        || action.posterior_model_sha256 != manifest.posterior_model_sha256
        || action.observation_likelihood_model_sha256
            != manifest.observation_likelihood_model_sha256
    {
        return Err(DecisionFocusedVoCError::new(
            format!("actions[{index}].provenance"),
            "action likelihoods must bind to the exact source, posterior and likelihood-model digests",
        ));
    }
    match (action.kind, action.candidate_id.as_deref()) {
        (VoCActionKind::FidelityQuery | VoCActionKind::DormantReactivation, Some(candidate_id))
            if posterior.candidate_ids.iter().any(|known| known == candidate_id) => {}
        (VoCActionKind::SentinelAudit, None) => {}
        (VoCActionKind::SentinelAudit, Some(_)) => {
            return Err(DecisionFocusedVoCError::new(
                format!("actions[{index}].candidate_id"),
                "sentinel audit actions must expose no individual candidate ID",
            ));
        }
        _ => {
            return Err(DecisionFocusedVoCError::new(
                format!("actions[{index}].candidate_id"),
                "query/reactivation actions require a known candidate ID; sentinel audits require none",
            ));
        }
    }
    if action.observation_outcomes.is_empty() {
        return Err(DecisionFocusedVoCError::new(
            format!("actions[{index}].observation_outcomes"),
            "at least one possible observation outcome is required",
        ));
    }
    let particle_count = posterior.particles.len();
    let mut outcome_ids = HashSet::with_capacity(action.observation_outcomes.len());
    let mut sums = vec![0.0; particle_count];
    for (outcome_index, outcome) in action.observation_outcomes.iter().enumerate() {
        if outcome.outcome_id.trim().is_empty() || !outcome_ids.insert(outcome.outcome_id.as_str()) {
            return Err(DecisionFocusedVoCError::new(
                format!("actions[{index}].observation_outcomes[{outcome_index}].outcome_id"),
                "observation outcome IDs must be non-empty and unique",
            ));
        }
        if outcome.likelihood_by_particle.len() != particle_count {
            return Err(DecisionFocusedVoCError::new(
                format!("actions[{index}].observation_outcomes[{outcome_index}]"),
                "likelihood vector must contain one value per posterior particle",
            ));
        }
        for (particle_index, likelihood) in outcome.likelihood_by_particle.iter().enumerate() {
            if !likelihood.is_finite() || !(0.0..=1.0).contains(likelihood) {
                return Err(DecisionFocusedVoCError::new(
                    format!("actions[{index}].observation_outcomes[{outcome_index}].likelihood_by_particle[{particle_index}]"),
                    "likelihoods must be finite and in [0, 1]",
                ));
            }
            sums[particle_index] += likelihood;
        }
    }
    if sums.iter().any(|sum| !approximately_equal(*sum, 1.0)) {
        return Err(DecisionFocusedVoCError::new(
            format!("actions[{index}].observation_outcomes"),
            "outcome likelihoods must sum to one for every posterior particle",
        ));
    }
    Ok(())
}

fn utility_particles(
    posterior: &PosteriorModel,
    direction: MetricDirection,
) -> Vec<Vec<f64>> {
    posterior.particles.iter().map(|particle| {
        particle.final_values.iter().map(|value| match direction {
            MetricDirection::Maximize => *value,
            MetricDirection::Minimize => -*value,
        }).collect()
    }).collect()
}

fn bayes_top_k_regret(
    utility_particles: &[Vec<f64>],
    probabilities: &[f64],
    target_k: usize,
) -> Result<f64, DecisionFocusedVoCError> {
    if utility_particles.is_empty()
        || utility_particles.len() != probabilities.len()
        || utility_particles[0].is_empty()
        || target_k == 0
        || target_k > utility_particles[0].len()
    {
        return Err(DecisionFocusedVoCError::new(
            "bayes_top_k_regret",
            "particle dimensions or target K are invalid",
        ));
    }
    let candidate_count = utility_particles[0].len();
    let mut expected_oracle_top_k = 0.0;
    let mut posterior_means = vec![0.0; candidate_count];
    for (particle_index, utilities) in utility_particles.iter().enumerate() {
        if utilities.len() != candidate_count {
            return Err(DecisionFocusedVoCError::new(
                "bayes_top_k_regret",
                "all particles must cover the same candidate set",
            ));
        }
        let mut sorted = utilities.clone();
        sorted.sort_by(|left, right| right.total_cmp(left));
        expected_oracle_top_k += probabilities[particle_index]
            * sorted.iter().take(target_k).sum::<f64>();
        for (candidate_index, utility) in utilities.iter().enumerate() {
            posterior_means[candidate_index] += probabilities[particle_index] * utility;
        }
    }
    posterior_means.sort_by(|left, right| right.total_cmp(left));
    let selected_expected_top_k = posterior_means.iter().take(target_k).sum::<f64>();
    let regret = expected_oracle_top_k - selected_expected_top_k;
    if !regret.is_finite() {
        return Err(DecisionFocusedVoCError::new(
            "bayes_top_k_regret",
            "calculated regret is non-finite",
        ));
    }
    if regret < -1e-8 {
        return Err(DecisionFocusedVoCError::new(
            "bayes_top_k_regret",
            "expected oracle top-K utility cannot be below the posterior-mean selected utility",
        ));
    }
    Ok(regret.max(0.0))
}

fn expected_regret_after_action(
    utility_particles: &[Vec<f64>],
    prior_probabilities: &[f64],
    outcomes: &[ObservationLikelihood],
    target_k: usize,
) -> Result<f64, DecisionFocusedVoCError> {
    let mut expected_regret = 0.0;
    let mut total_outcome_probability = 0.0;
    for (outcome_index, outcome) in outcomes.iter().enumerate() {
        let mut joint = Vec::with_capacity(prior_probabilities.len());
        let mut outcome_probability = 0.0;
        for (particle_index, prior) in prior_probabilities.iter().enumerate() {
            let mass = prior * outcome.likelihood_by_particle[particle_index];
            joint.push(mass);
            outcome_probability += mass;
        }
        if outcome_probability <= 0.0 {
            continue;
        }
        total_outcome_probability += outcome_probability;
        let posterior_probabilities: Vec<f64> =
            joint.iter().map(|mass| mass / outcome_probability).collect();
        let regret = bayes_top_k_regret(
            utility_particles,
            &posterior_probabilities,
            target_k,
        )?;
        expected_regret += outcome_probability * regret;
        if !expected_regret.is_finite() {
            return Err(DecisionFocusedVoCError::new(
                format!("observation_outcomes[{outcome_index}]"),
                "expected posterior regret is non-finite",
            ));
        }
    }
    if !approximately_equal(total_outcome_probability, 1.0) {
        return Err(DecisionFocusedVoCError::new(
            "observation_outcomes",
            "marginal outcome probabilities must sum to one",
        ));
    }
    Ok(expected_regret)
}

fn fits_total_budget(
    action: &VoCAction,
    config: &DecisionFocusedVoCConfig,
) -> bool {
    config.already_spent_units
        .checked_add(action.incremental_cost_units)
        .is_some_and(|next_spent| next_spent <= config.total_budget_units)
}

fn fits_reserve_policy(
    action: &VoCAction,
    config: &DecisionFocusedVoCConfig,
) -> bool {
    let Some(next_spent) = config.already_spent_units.checked_add(action.incremental_cost_units)
    else {
        return false;
    };
    let spent_for_kind = |kind| {
        config.spend_by_kind.iter()
            .find(|record| record.kind == kind)
            .map(|record| record.spent_units)
            .unwrap_or(0)
            .saturating_add(if action.kind == kind { action.incremental_cost_units } else { 0 })
    };
    let outstanding_reserves = config.reserves.iter().fold(0u64, |total, reserve| {
        total.saturating_add(reserve.reserve_units.saturating_sub(spent_for_kind(reserve.kind)))
    });
    let remaining = config.total_budget_units.saturating_sub(next_spent);
    remaining >= outstanding_reserves
}

fn validate_sha256(
    value: &str,
    field: &str,
) -> Result<(), DecisionFocusedVoCError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DecisionFocusedVoCError::new(
            field,
            "must be exactly 64 hexadecimal characters",
        ));
    }
    Ok(())
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-10 * left.abs().max(right.abs()).max(1.0)
}

fn digest_plan(
    manifest: &DecisionFocusedVoCModelManifest,
    posterior: &PosteriorModel,
    config: &DecisionFocusedVoCConfig,
    current_expected_regret: f64,
    evaluations: &[VoCActionEvaluation],
    selected_action_id: Option<&str>,
    stop_reason: Option<VoCStopReason>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(VOC_DOMAIN);
    update_string(&mut hasher, &manifest.benchmark_id);
    update_string(&mut hasher, &manifest.target_metric_id);
    update_string(&mut hasher, &manifest.evaluation_task_id);
    update_string(&mut hasher, &manifest.source_artifact_sha256);
    update_string(&mut hasher, &manifest.posterior_model_sha256);
    update_string(&mut hasher, &manifest.observation_likelihood_model_sha256);
    update_string(&mut hasher, &config.resource_unit);
    hasher.update((config.target_k as u64).to_be_bytes());
    hasher.update(config.total_budget_units.to_be_bytes());
    hasher.update(config.already_spent_units.to_be_bytes());
    hasher.update(config.cost_penalty_per_unit.to_bits().to_be_bytes());
    hasher.update(current_expected_regret.to_bits().to_be_bytes());
    for reserve in &config.reserves {
        hasher.update([action_kind_tag(reserve.kind)]);
        hasher.update(reserve.reserve_units.to_be_bytes());
    }
    for spend in &config.spend_by_kind {
        hasher.update([action_kind_tag(spend.kind)]);
        hasher.update(spend.spent_units.to_be_bytes());
    }
    let mut candidates: Vec<&String> = posterior.candidate_ids.iter().collect();
    candidates.sort();
    for candidate in candidates {
        update_string(&mut hasher, candidate);
    }
    let mut particles: Vec<&FinalPerformanceParticle> = posterior.particles.iter().collect();
    particles.sort_by(|left, right| left.particle_id.cmp(&right.particle_id));
    for particle in particles {
        update_string(&mut hasher, &particle.particle_id);
        hasher.update(particle.probability.to_bits().to_be_bytes());
        for value in &particle.final_values {
            hasher.update(value.to_bits().to_be_bytes());
        }
    }
    for evaluation in evaluations {
        update_string(&mut hasher, &evaluation.action_id);
        hasher.update([action_kind_tag(evaluation.kind)]);
        update_optional_string(&mut hasher, evaluation.candidate_id.as_deref());
        hasher.update(evaluation.incremental_cost_units.to_be_bytes());
        hasher.update(evaluation.expected_posterior_regret.to_bits().to_be_bytes());
        hasher.update(evaluation.expected_regret_reduction.to_bits().to_be_bytes());
        hasher.update(evaluation.regret_reduction_per_cost_unit.to_bits().to_be_bytes());
        hasher.update(evaluation.regret_equivalent_cost.to_bits().to_be_bytes());
        hasher.update(evaluation.net_expected_utility.to_bits().to_be_bytes());
        hasher.update([action_status_tag(evaluation.status)]);
    }
    update_optional_string(&mut hasher, selected_action_id);
    hasher.update([stop_reason.map(stop_reason_tag).unwrap_or(0)]);
    hex_digest(hasher.finalize().as_slice())
}

fn action_kind_tag(kind: VoCActionKind) -> u8 {
    match kind {
        VoCActionKind::FidelityQuery => 0,
        VoCActionKind::DormantReactivation => 1,
        VoCActionKind::SentinelAudit => 2,
    }
}

fn action_status_tag(status: VoCActionStatus) -> u8 {
    match status {
        VoCActionStatus::Feasible => 0,
        VoCActionStatus::BlockedByTotalBudget => 1,
        VoCActionStatus::BlockedByProtectedReserve => 2,
        VoCActionStatus::NotWorthItsCost => 3,
        VoCActionStatus::Selected => 4,
    }
}

fn stop_reason_tag(reason: VoCStopReason) -> u8 {
    match reason {
        VoCStopReason::NoLegalActions => 1,
        VoCStopReason::NoActionFitsResourceAndReserveBudgets => 2,
        VoCStopReason::NoPositiveNetExpectedUtility => 3,
    }
}

fn update_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn update_optional_string(hasher: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            update_string(hasher, value);
        }
        None => hasher.update([0]),
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    let alphabet = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes.iter().copied() {
        encoded.push(alphabet[(byte >> 4) as usize] as char);
        encoded.push(alphabet[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const POSTERIOR_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const LIKELIHOOD_SHA: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn manifest() -> DecisionFocusedVoCModelManifest {
        DecisionFocusedVoCModelManifest {
            benchmark_id: "NB201-CIFAR10".to_owned(),
            target_metric_id: "validation_accuracy".to_owned(),
            metric_direction: MetricDirection::Maximize,
            evaluation_task_id: "target-task".to_owned(),
            calibration_task_ids: vec!["cal-task-1".to_owned(), "cal-task-2".to_owned()],
            source_artifact_sha256: SOURCE_SHA.to_owned(),
            posterior_model_sha256: POSTERIOR_SHA.to_owned(),
            observation_likelihood_model_sha256: LIKELIHOOD_SHA.to_owned(),
            task_calibration_qualified: true,
        }
    }

    fn posterior() -> PosteriorModel {
        PosteriorModel {
            candidate_ids: vec!["A".to_owned(), "B".to_owned()],
            particles: vec![
                FinalPerformanceParticle {
                    particle_id: "p1".to_owned(),
                    probability: 0.5,
                    final_values: vec![1.0, 0.0],
                },
                FinalPerformanceParticle {
                    particle_id: "p2".to_owned(),
                    probability: 0.5,
                    final_values: vec![0.0, 1.0],
                },
            ],
        }
    }

    fn config() -> DecisionFocusedVoCConfig {
        DecisionFocusedVoCConfig {
            target_k: 1,
            resource_unit: "gpu_seconds".to_owned(),
            total_budget_units: 100,
            already_spent_units: 0,
            cost_penalty_per_unit: 0.1,
            reserves: Vec::new(),
            spend_by_kind: Vec::new(),
        }
    }

    fn query_action(action_id: &str, candidate_id: &str, cost: u64) -> VoCAction {
        VoCAction {
            action_id: action_id.to_owned(),
            kind: VoCActionKind::FidelityQuery,
            candidate_id: Some(candidate_id.to_owned()),
            resource_unit: "gpu_seconds".to_owned(),
            incremental_cost_units: cost,
            source_artifact_sha256: SOURCE_SHA.to_owned(),
            posterior_model_sha256: POSTERIOR_SHA.to_owned(),
            observation_likelihood_model_sha256: LIKELIHOOD_SHA.to_owned(),
            observation_outcomes: vec![
                ObservationLikelihood {
                    outcome_id: "high".to_owned(),
                    likelihood_by_particle: vec![0.9, 0.1],
                },
                ObservationLikelihood {
                    outcome_id: "low".to_owned(),
                    likelihood_by_particle: vec![0.1, 0.9],
                },
            ],
        }
    }

    #[test]
    fn decision_focused_voi_beats_an_interval_width_surrogate() {
        let action = query_action("query-A", "A", 1);
        let plan = plan_decision_focused_voc(&manifest(), &posterior(), &[action], &config()).unwrap();
        assert!((plan.current_expected_regret - 0.5).abs() < 1e-9);
        let evaluation = &plan.action_evaluations[0];
        assert!((evaluation.expected_posterior_regret - 0.1).abs() < 1e-9);
        assert!((evaluation.expected_regret_reduction - 0.4).abs() < 1e-9);
        assert!((evaluation.net_expected_utility - 0.3).abs() < 1e-9);
        assert_eq!(plan.selected_action_id.as_deref(), Some("query-A"));
        assert!(plan.plan_sha256.len() == 64);
    }

    #[test]
    fn uninformative_action_is_rejected_and_planner_stops() {
        let mut action = query_action("noop", "A", 1);
        action.observation_outcomes = vec![ObservationLikelihood {
            outcome_id: "same".to_owned(),
            likelihood_by_particle: vec![1.0, 1.0],
        }];
        let plan = plan_decision_focused_voc(&manifest(), &posterior(), &[action], &config()).unwrap();
        assert_eq!(plan.selected_action_id, None);
        assert_eq!(plan.stop_reason, Some(VoCStopReason::NoPositiveNetExpectedUtility));
        assert_eq!(plan.action_evaluations[0].expected_regret_reduction, 0.0);
        assert_eq!(plan.action_evaluations[0].status, VoCActionStatus::NotWorthItsCost);
    }

    #[test]
    fn top_k_objective_reduces_to_zero_when_all_candidates_are_selected() {
        let mut cfg = config();
        cfg.target_k = 2;
        let plan = plan_decision_focused_voc(&manifest(), &posterior(), &[], &cfg).unwrap();
        assert_eq!(plan.current_expected_regret, 0.0);
        assert_eq!(plan.stop_reason, Some(VoCStopReason::NoLegalActions));
    }

    #[test]
    fn lower_is_better_metrics_are_transformed_to_utility() {
        let mut m = manifest();
        m.metric_direction = MetricDirection::Minimize;
        let mut p = posterior();
        p.particles[0].final_values = vec![0.0, 1.0];
        p.particles[1].final_values = vec![1.0, 0.0];
        let plan = plan_decision_focused_voc(&m, &p, &[], &config()).unwrap();
        assert!((plan.current_expected_regret - 0.5).abs() < 1e-9);
    }

    #[test]
    fn protected_reserve_is_preserved_for_a_dormant_candidate() {
        let mut cfg = config();
        cfg.total_budget_units = 10;
        cfg.reserves = vec![VoCActionReserve {
            kind: VoCActionKind::DormantReactivation,
            reserve_units: 3,
        }];
        let too_expensive = query_action("big-query", "A", 8);
        let affordable = query_action("small-query", "A", 2);
        let plan = plan_decision_focused_voc(
            &manifest(),
            &posterior(),
            &[too_expensive, affordable],
            &cfg,
        ).unwrap();
        assert_eq!(plan.selected_action_id.as_deref(), Some("small-query"));
        assert_eq!(plan.action_evaluations[0].status, VoCActionStatus::BlockedByProtectedReserve);
    }

    #[test]
    fn hard_total_budget_blocks_unaffordable_actions() {
        let mut cfg = config();
        cfg.total_budget_units = 1;
        let action = query_action("query", "A", 2);
        let plan = plan_decision_focused_voc(&manifest(), &posterior(), &[action], &cfg).unwrap();
        assert_eq!(plan.stop_reason, Some(VoCStopReason::NoActionFitsResourceAndReserveBudgets));
        assert_eq!(plan.action_evaluations[0].status, VoCActionStatus::BlockedByTotalBudget);
    }

    #[test]
    fn rejects_calibration_task_leakage() {
        let mut m = manifest();
        m.evaluation_task_id = "cal-task-1".to_owned();
        assert!(plan_decision_focused_voc(&m, &posterior(), &[], &config()).is_err());
    }

    #[test]
    fn rejects_unqualified_likelihood_model() {
        let mut m = manifest();
        m.task_calibration_qualified = false;
        assert!(plan_decision_focused_voc(&m, &posterior(), &[], &config()).is_err());
    }

    #[test]
    fn rejects_action_digest_mismatch() {
        let mut action = query_action("query", "A", 1);
        action.posterior_model_sha256 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_owned();
        assert!(plan_decision_focused_voc(&manifest(), &posterior(), &[action], &config()).is_err());
    }

    #[test]
    fn rejects_non_normalized_observation_likelihoods() {
        let mut action = query_action("query", "A", 1);
        action.observation_outcomes[0].likelihood_by_particle = vec![0.9, 0.9];
        assert!(plan_decision_focused_voc(&manifest(), &posterior(), &[action], &config()).is_err());
    }

    #[test]
    fn sentinel_audits_expose_no_individual_candidate_id() {
        let mut action = query_action("audit", "A", 1);
        action.kind = VoCActionKind::SentinelAudit;
        action.candidate_id = None;
        let plan = plan_decision_focused_voc(&manifest(), &posterior(), &[action], &config()).unwrap();
        assert_eq!(plan.selected_action_id.as_deref(), Some("audit"));
    }

    #[test]
    fn deterministic_tie_break_uses_lexical_action_id() {
        let a = query_action("a-query", "A", 1);
        let b = query_action("b-query", "B", 1);
        let plan = plan_decision_focused_voc(&manifest(), &posterior(), &[b, a], &config()).unwrap();
        assert_eq!(plan.selected_action_id.as_deref(), Some("a-query"));
    }
}
