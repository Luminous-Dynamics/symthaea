//! Portfolio-class Pareto regret and objective-decoupled measurement value of computation.
//!
//! A bounded family of candidate portfolios is supplied by an external search policy (e.g.
//! NSGA-II, a preference sweep, qEHVI/HVKG or a local Pareto archive). This module evaluates
//! those portfolios under a finite joint posterior, selects the best posterior-expected
//! hypervolume portfolio, and measures its regret against the all-feasible-candidate oracle.
//!
//! The reported regret is relative to the supplied portfolio family. It is exact only when that
//! family contains every allowed candidate subset; otherwise it is explicitly portfolio-class
//! regret. This avoids enumerating all subsets of a large NAS search space while retaining a
//! clear comparator and claim ceiling.
//!
//! Portfolio selection uses one half of an immutable randomized-QMC design. The other half is
//! held out to evaluate the selected portfolio and oracle, avoiding same-sample winner scoring.
//! Every posterior outcome reuses the same selection and held-out scrambles. VOI uncertainty is
//! computed from replicate-level regret differences, preserving common-random-number covariance.
//!
//! The module is a diagnostic optimizer primitive, not an operational pruning certificate.
//! Action likelihoods require separate empirical qualification and mandatory receipt enforcement.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::hardware_objectives::{
    validate_objective_registry, ObjectiveRegistry, ObjectiveRegistryError,
};
use crate::pareto_hypervolume_qmc::{
    estimate_hypervolume_qmc_with_replicates, HypervolumeQmcDesign, HypervolumeQmcError,
};

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const ACTION_DOMAIN: &[u8] = b"symthaea.nas.portfolio-voc-action.v0.1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoPortfolioParticle {
    pub particle_id: String,
    pub probability: f64,
    /// One optional objective vector per candidate, in posterior.candidate_ids order.
    /// Feasible candidates require a complete vector; infeasible candidates may be None.
    pub objective_values: Vec<Option<Vec<f64>>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoPortfolioPosterior {
    /// Strictly sorted, unique candidate IDs bind every candidate-vector column.
    pub candidate_ids: Vec<String>,
    /// Strictly sorted objective IDs; this is the exact coordinate order for every objective vector.
    pub objective_ids: Vec<String>,
    /// Digest of the complete objective registry defining units, directions, protocols and costs.
    pub objective_registry_sha256: String,
    /// Exact architecture configuration SHA-256 for each candidate, in candidate_ids order.
    pub candidate_config_sha256: Vec<String>,
    /// Device/profile feasibility is a hard state and is not encoded as a numeric objective.
    pub candidate_feasible: Vec<bool>,
    pub particles: Vec<ParetoPortfolioParticle>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidatePortfolio {
    pub portfolio_id: String,
    /// Candidate indices in posterior.candidate_ids. Order is canonicalized internally.
    pub candidate_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoPortfolioConfig {
    pub target_k: usize,
    /// Hard minimum normalized utility for each objective. A candidate below any floor is
    /// infeasible for this deployment profile, even if its weighted/average score is high.
    pub minimum_feasible_utility: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioSelectionScore {
    pub portfolio_id: String,
    pub candidate_ids: Vec<String>,
    /// Used for selecting the portfolio; estimated on the selection half of QMC scrambles.
    pub selection_expected_hypervolume: f64,
    pub selection_standard_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParetoPortfolioDecision {
    pub oracle_expected_hypervolume: f64,
    pub oracle_validation_standard_error: f64,
    pub selected_portfolio_id: String,
    pub selected_candidate_ids: Vec<String>,
    /// Held-out expected hypervolume of the selected portfolio.
    pub selected_portfolio_expected_hypervolume: f64,
    pub selected_portfolio_validation_standard_error: f64,
    /// Oracle minus selected portfolio; measured only on held-out scrambles.
    pub portfolio_class_regret: f64,
    pub portfolio_class_regret_standard_error: f64,
    pub portfolio_class_regret_interval_95: (f64, f64),
    /// Held-out replicate regret values are retained so action outcomes can be aggregated
    /// using common random numbers rather than adding independent standard errors.
    pub validation_regret_replicates: Vec<f64>,
    pub selection_replicate_indices: Vec<usize>,
    pub validation_replicate_indices: Vec<usize>,
    pub portfolio_selection_scores: Vec<PortfolioSelectionScore>,
    pub posterior_particle_count: usize,
    /// Identity of the exact joint posterior, including candidate configuration hashes.
    pub posterior_sha256: String,
    /// Identity of the exact frozen family of candidate portfolios and feasibility config.
    pub portfolio_family_sha256: String,
    /// Immutable full QMC design plus disjoint selection/validation scramble subsets.
    pub qmc_sampling_design_sha256: String,
    pub selection_replicates_sha256: String,
    pub validation_replicates_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortfolioActionOutcome {
    pub outcome_id: String,
    /// P(outcome | posterior particle j), in posterior.particles order.
    pub likelihood_by_particle: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortfolioMeasurementAction {
    pub action_id: String,
    pub candidate_id: String,
    pub candidate_config_sha256: String,
    pub objective_registry_sha256: String,
    pub objective_id: String,
    pub hardware_profile_sha256: Option<String>,
    pub measurement_protocol_sha256: String,
    pub resource_unit: String,
    pub incremental_cost_units: u64,
    /// SHA-256 over this action and its exact outcome likelihood arrays.
    pub observation_likelihood_model_sha256: String,
    pub outcomes: Vec<PortfolioActionOutcome>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioParetoVoC {
    pub action_id: String,
    pub current_decision: ParetoPortfolioDecision,
    pub expected_posterior_regret: f64,
    pub expected_posterior_regret_standard_error: f64,
    /// May be negative due to finite numerical estimation; use its interval for abstention.
    pub estimated_regret_reduction: f64,
    pub estimated_regret_reduction_standard_error: f64,
    pub estimated_regret_reduction_interval_95: (f64, f64),
    pub regret_reduction_per_cost_unit: f64,
    pub incremental_cost_units: u64,
    pub resource_unit: String,
    pub expected_observation_probabilities: Vec<(String, f64)>,
    pub observation_likelihood_model_sha256: String,
    pub objective_registry_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortfolioVoCError {
    pub field: String,
    pub message: String,
}

impl PortfolioVoCError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for PortfolioVoCError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for PortfolioVoCError {}

impl From<HypervolumeQmcError> for PortfolioVoCError {
    fn from(error: HypervolumeQmcError) -> Self {
        Self::new(error.field, error.message)
    }
}

impl From<ObjectiveRegistryError> for PortfolioVoCError {
    fn from(error: ObjectiveRegistryError) -> Self {
        Self::new(error.field, error.message)
    }
}

/// Compute a canonical SHA-256 over action identity, cost, device/protocol and labelled
/// observation likelihoods. Outcome vector order is non-semantic and canonicalized by label.
pub fn compute_portfolio_action_likelihood_sha256(
    action: &PortfolioMeasurementAction,
) -> Result<String, PortfolioVoCError> {
    let mut canonical = action.clone();
    canonical
        .outcomes
        .sort_by(|left, right| left.outcome_id.cmp(&right.outcome_id));
    canonical.observation_likelihood_model_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| PortfolioVoCError::new("action", error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update((ACTION_DOMAIN.len() as u64).to_be_bytes());
    hasher.update(ACTION_DOMAIN);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    Ok(encoded)
}

/// Evaluate the best portfolio in a frozen portfolio family under a joint finite posterior.
///
/// The first half of the immutable QMC design selects a portfolio. The second half estimates
/// the chosen portfolio and the all-feasible-candidate oracle. Therefore the reported regret
/// is not evaluated on the same numerical samples used to choose the winner.
pub fn evaluate_portfolio_class_regret(
    posterior: &ParetoPortfolioPosterior,
    portfolios: &[CandidatePortfolio],
    config: &ParetoPortfolioConfig,
    design: &HypervolumeQmcDesign,
) -> Result<ParetoPortfolioDecision, PortfolioVoCError> {
    validate_posterior(posterior, config, design)?;
    let canonical_portfolios = validate_and_canonicalize_portfolios(
        portfolios,
        posterior.candidate_ids.len(),
        config.target_k,
    )?;

    let replicate_count = design.scramble_replicates();
    if replicate_count < 16 || replicate_count % 2 != 0 {
        return Err(PortfolioVoCError::new(
            "qmc_design",
            "portfolio selection requires an even number of at least sixteen scrambles (eight per split)",
        ));
    }
    let split = replicate_count / 2;
    let selection_replicate_indices: Vec<usize> = (0..split).collect();
    let validation_replicate_indices: Vec<usize> = (split..replicate_count).collect();

    let mut portfolio_selection_scores = Vec::with_capacity(canonical_portfolios.len());
    for portfolio in &canonical_portfolios {
        let replicate_values = expected_hypervolume_replicates(
            posterior,
            &portfolio.candidate_indices,
            config,
            design,
            &selection_replicate_indices,
        )?;
        portfolio_selection_scores.push(PortfolioSelectionScore {
            portfolio_id: portfolio.portfolio_id.clone(),
            candidate_ids: portfolio
                .candidate_indices
                .iter()
                .map(|index| posterior.candidate_ids[*index].clone())
                .collect(),
            selection_expected_hypervolume: mean(&replicate_values),
            selection_standard_error: standard_error(&replicate_values),
        });
    }
    portfolio_selection_scores.sort_by(|left, right| left.portfolio_id.cmp(&right.portfolio_id));

    let selected = portfolio_selection_scores
        .iter()
        .max_by(|left, right| {
            left.selection_expected_hypervolume
                .total_cmp(&right.selection_expected_hypervolume)
                .then_with(|| right.portfolio_id.cmp(&left.portfolio_id))
        })
        .ok_or_else(|| {
            PortfolioVoCError::new("portfolios", "at least one valid portfolio is required")
        })?;
    let selected_portfolio = canonical_portfolios
        .iter()
        .find(|portfolio| portfolio.portfolio_id == selected.portfolio_id)
        .ok_or_else(|| {
            PortfolioVoCError::new(
                "selected_portfolio_id",
                "selected portfolio score has no matching source portfolio",
            )
        })?;

    let all_candidate_indices: Vec<usize> = (0..posterior.candidate_ids.len()).collect();
    let oracle_replicates = expected_hypervolume_replicates(
        posterior,
        &all_candidate_indices,
        config,
        design,
        &validation_replicate_indices,
    )?;
    let selected_replicates = expected_hypervolume_replicates(
        posterior,
        &selected_portfolio.candidate_indices,
        config,
        design,
        &validation_replicate_indices,
    )?;
    let regret_replicates: Vec<f64> = oracle_replicates
        .iter()
        .zip(&selected_replicates)
        .map(|(oracle, selected)| (oracle - selected).max(0.0))
        .collect();

    let portfolio_class_regret = mean(&regret_replicates);
    let portfolio_class_regret_standard_error = standard_error(&regret_replicates);
    let margin = 1.96 * portfolio_class_regret_standard_error;
    let maximum_possible_regret = design
        .reference()
        .iter()
        .map(|lower| 1.0 - lower)
        .product::<f64>();
    let regret_interval = (
        (portfolio_class_regret - margin).max(0.0),
        (portfolio_class_regret + margin).min(maximum_possible_regret),
    );

    let posterior_sha256 = digest_serializable(
        b"posterior",
        posterior,
    )?;
    let portfolio_family_sha256 = digest_portfolio_family(
        &canonical_portfolios,
        config,
    )?;
    let qmc_sampling_design_sha256 = design.sampling_design_sha256().to_owned();
    let selection_replicates_sha256 =
        digest_replicate_indices(&qmc_sampling_design_sha256, &selection_replicate_indices);
    let validation_replicates_sha256 =
        digest_replicate_indices(&qmc_sampling_design_sha256, &validation_replicate_indices);

    Ok(ParetoPortfolioDecision {
        oracle_expected_hypervolume: mean(&oracle_replicates),
        oracle_validation_standard_error: standard_error(&oracle_replicates),
        selected_portfolio_id: selected.portfolio_id.clone(),
        selected_candidate_ids: selected.candidate_ids.clone(),
        selected_portfolio_expected_hypervolume: mean(&selected_replicates),
        selected_portfolio_validation_standard_error: standard_error(&selected_replicates),
        portfolio_class_regret,
        portfolio_class_regret_standard_error,
        portfolio_class_regret_interval_95: regret_interval,
        validation_regret_replicates: regret_replicates,
        selection_replicate_indices,
        validation_replicate_indices,
        portfolio_selection_scores,
        posterior_particle_count: posterior.particles.len(),
        posterior_sha256,
        portfolio_family_sha256,
        qmc_sampling_design_sha256,
        selection_replicates_sha256,
        validation_replicates_sha256,
    })
}

/// Evaluate one objective-decoupled measurement action by its expected reduction in
/// portfolio-class Pareto hypervolume regret per unit of objective acquisition cost.
///
/// The action must match an objective in the registry. Its outcome likelihood model is
/// identity-bound to the exact posterior particle order and must sum to one per particle.
pub fn evaluate_portfolio_measurement_voc(
    registry: &ObjectiveRegistry,
    posterior: &ParetoPortfolioPosterior,
    portfolios: &[CandidatePortfolio],
    config: &ParetoPortfolioConfig,
    design: &HypervolumeQmcDesign,
    action: &PortfolioMeasurementAction,
) -> Result<PortfolioParetoVoC, PortfolioVoCError> {
    validate_posterior(posterior, config, design)?;
    validate_objective_registry(registry)?;
    if posterior.objective_registry_sha256 != registry.registry_sha256 {
        return Err(PortfolioVoCError::new(
            "objective_registry_sha256",
            "posterior objective vectors were normalized under a different objective registry",
        ));
    }
    let mut registry_objective_ids: Vec<String> = registry
        .objectives
        .iter()
        .map(|objective| objective.objective_id.clone())
        .collect();
    registry_objective_ids.sort();
    if posterior.objective_ids != registry_objective_ids {
        return Err(PortfolioVoCError::new(
            "objective_ids",
            "posterior objective coordinate order does not match the objective registry",
        ));
    }
    if posterior.objective_ids.len() != design.reference().len() {
        return Err(PortfolioVoCError::new(
            "objective_ids",
            "posterior objective count does not match the QMC design",
        ));
    }
    validate_action(registry, posterior, action)?;

    let current_decision =
        evaluate_portfolio_class_regret(posterior, portfolios, config, design)?;
    let mut expected_observation_probabilities =
        Vec::with_capacity(action.outcomes.len());
    let mut expected_posterior_regret_by_replicate =
        vec![0.0; current_decision.validation_regret_replicates.len()];

    for outcome in &action.outcomes {
        let outcome_probability: f64 = posterior
            .particles
            .iter()
            .zip(&outcome.likelihood_by_particle)
            .map(|(particle, likelihood)| particle.probability * likelihood)
            .sum();
        expected_observation_probabilities
            .push((outcome.outcome_id.clone(), outcome_probability));
        if outcome_probability <= 0.0 {
            continue;
        }

        let mut updated = posterior.clone();
        for (particle, likelihood) in updated
            .particles
            .iter_mut()
            .zip(&outcome.likelihood_by_particle)
        {
            particle.probability = particle.probability * likelihood / outcome_probability;
        }
        updated
            .particles
            .retain(|particle| particle.probability > 0.0);

        let posterior_decision =
            evaluate_portfolio_class_regret(&updated, portfolios, config, design)?;
        if posterior_decision.validation_regret_replicates.len()
            != expected_posterior_regret_by_replicate.len()
        {
            return Err(PortfolioVoCError::new(
                "qmc_validation",
                "prior and posterior decisions used different validation replicate counts",
            ));
        }
        for (expected, observed) in expected_posterior_regret_by_replicate
            .iter_mut()
            .zip(&posterior_decision.validation_regret_replicates)
        {
            *expected += outcome_probability * observed;
        }
    }

    let regret_reduction_by_replicate: Vec<f64> = current_decision
        .validation_regret_replicates
        .iter()
        .zip(&expected_posterior_regret_by_replicate)
        .map(|(current, expected_posterior)| current - expected_posterior)
        .collect();
    let expected_posterior_regret = mean(&expected_posterior_regret_by_replicate);
    let expected_posterior_regret_standard_error =
        standard_error(&expected_posterior_regret_by_replicate);
    let estimated_regret_reduction = mean(&regret_reduction_by_replicate);
    let estimated_regret_reduction_standard_error =
        standard_error(&regret_reduction_by_replicate);
    let margin = 1.96 * estimated_regret_reduction_standard_error;

    let objective = registry
        .objectives
        .iter()
        .find(|definition| definition.objective_id == action.objective_id)
        .ok_or_else(|| PortfolioVoCError::new("objective_id", "objective disappeared"))?;

    Ok(PortfolioParetoVoC {
        action_id: action.action_id.clone(),
        current_decision,
        expected_posterior_regret,
        expected_posterior_regret_standard_error,
        estimated_regret_reduction,
        estimated_regret_reduction_standard_error,
        estimated_regret_reduction_interval_95: (
            estimated_regret_reduction - margin,
            estimated_regret_reduction + margin,
        ),
        regret_reduction_per_cost_unit: estimated_regret_reduction
            / action.incremental_cost_units as f64,
        incremental_cost_units: action.incremental_cost_units,
        resource_unit: objective.measurement_cost_unit.clone(),
        expected_observation_probabilities,
        observation_likelihood_model_sha256: action
            .observation_likelihood_model_sha256
            .clone(),
        objective_registry_sha256: registry.registry_sha256.clone(),
    })
}

fn validate_action(
    registry: &ObjectiveRegistry,
    posterior: &ParetoPortfolioPosterior,
    action: &PortfolioMeasurementAction,
) -> Result<(), PortfolioVoCError> {
    if action.action_id.trim().is_empty() || action.candidate_id.trim().is_empty() {
        return Err(PortfolioVoCError::new(
            "action",
            "action and candidate IDs must be non-empty",
        ));
    }
    let candidate_index = posterior
        .candidate_ids
        .binary_search(&action.candidate_id)
        .map_err(|_| PortfolioVoCError::new("candidate_id", "action candidate is not in posterior"))?;
    if !posterior.candidate_feasible[candidate_index] {
        return Err(PortfolioVoCError::new(
            "candidate_id",
            "cannot measure a candidate already marked hard-infeasible for this deployment profile",
        ));
    }
    if action.candidate_config_sha256 != posterior.candidate_config_sha256[candidate_index] {
        return Err(PortfolioVoCError::new(
            "candidate_config_sha256",
            "action configuration digest does not match the exact posterior candidate",
        ));
    }
    if action.objective_registry_sha256 != registry.registry_sha256 {
        return Err(PortfolioVoCError::new(
            "objective_registry_sha256",
            "action was created against a different objective registry or acquisition-cost model",
        ));
    }
    if action.incremental_cost_units == 0 {
        return Err(PortfolioVoCError::new(
            "incremental_cost_units",
            "must be positive",
        ));
    }
    if action.outcomes.is_empty() {
        return Err(PortfolioVoCError::new(
            "outcomes",
            "at least one outcome is required",
        ));
    }
    if action.observation_likelihood_model_sha256.len() != 64
        || !action
            .observation_likelihood_model_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PortfolioVoCError::new(
            "observation_likelihood_model_sha256",
            "must be a lowercase hexadecimal SHA-256 digest",
        ));
    }
    if compute_portfolio_action_likelihood_sha256(action)?
        != action.observation_likelihood_model_sha256
    {
        return Err(PortfolioVoCError::new(
            "observation_likelihood_model_sha256",
            "does not match exact action outcome likelihoods",
        ));
    }

    let objective = registry
        .objectives
        .iter()
        .find(|definition| definition.objective_id == action.objective_id)
        .ok_or_else(|| {
            PortfolioVoCError::new(
                "objective_id",
                format!("unknown objective {:?}", action.objective_id),
            )
        })?;
    if action.resource_unit != objective.measurement_cost_unit {
        return Err(PortfolioVoCError::new(
            "resource_unit",
            "action acquisition-cost unit does not match the objective registry",
        ));
    }
    if action.measurement_protocol_sha256 != objective.measurement_protocol_sha256 {
        return Err(PortfolioVoCError::new(
            "measurement_protocol_sha256",
            "action protocol does not match the frozen objective definition",
        ));
    }
    if action.hardware_profile_sha256 != objective.hardware_profile_sha256 {
        return Err(PortfolioVoCError::new(
            "hardware_profile_sha256",
            "action device profile does not match the frozen objective definition",
        ));
    }

    let mut outcome_ids = HashSet::with_capacity(action.outcomes.len());
    let mut particle_sums = vec![0.0; posterior.particles.len()];
    for (outcome_index, outcome) in action.outcomes.iter().enumerate() {
        if outcome.outcome_id.trim().is_empty()
            || !outcome_ids.insert(outcome.outcome_id.as_str())
        {
            return Err(PortfolioVoCError::new(
                format!("outcomes[{outcome_index}].outcome_id"),
                "outcome IDs must be non-empty and unique",
            ));
        }
        if outcome.likelihood_by_particle.len() != posterior.particles.len() {
            return Err(PortfolioVoCError::new(
                format!("outcomes[{outcome_index}].likelihood_by_particle"),
                "requires exactly one likelihood per posterior particle",
            ));
        }
        for (particle_index, likelihood) in
            outcome.likelihood_by_particle.iter().copied().enumerate()
        {
            if !likelihood.is_finite() || !(0.0..=1.0).contains(&likelihood) {
                return Err(PortfolioVoCError::new(
                    format!("outcomes[{outcome_index}].likelihood_by_particle[{particle_index}]"),
                    "likelihoods must be finite and in [0,1]",
                ));
            }
            particle_sums[particle_index] += likelihood;
        }
    }
    for (particle_index, sum) in particle_sums.into_iter().enumerate() {
        if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(PortfolioVoCError::new(
                format!("outcomes[*].likelihood_by_particle[{particle_index}]"),
                "mutually exclusive outcome likelihoods must sum to one for every particle",
            ));
        }
    }
    Ok(())
}

fn validate_posterior(
    posterior: &ParetoPortfolioPosterior,
    config: &ParetoPortfolioConfig,
    design: &HypervolumeQmcDesign,
) -> Result<(), PortfolioVoCError> {
    let candidate_count = posterior.candidate_ids.len();
    if candidate_count == 0
        || posterior.candidate_feasible.len() != candidate_count
        || posterior.candidate_config_sha256.len() != candidate_count
    {
        return Err(PortfolioVoCError::new(
            "candidate_ids",
            "candidate IDs, configuration digests and feasibility flags must have the same non-zero length",
        ));
    }
    if posterior.objective_ids.len() != design.reference().len()
        || posterior.objective_ids.iter().any(|id| id.trim().is_empty())
        || posterior.objective_ids.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(PortfolioVoCError::new(
            "objective_ids",
            "objective IDs must be non-empty, strictly sorted and match the QMC design dimension",
        ));
    }
    if !is_sha256(&posterior.objective_registry_sha256) {
        return Err(PortfolioVoCError::new(
            "objective_registry_sha256",
            "must be a lowercase hexadecimal SHA-256 digest",
        ));
    }
    for (index, digest) in posterior.candidate_config_sha256.iter().enumerate() {
        if !is_sha256(digest) {
            return Err(PortfolioVoCError::new(
                format!("candidate_config_sha256[{index}]"),
                "must be a lowercase hexadecimal SHA-256 digest",
            ));
        }
    }
    if posterior
        .candidate_ids
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(PortfolioVoCError::new(
            "candidate_ids",
            "candidate IDs must be strictly sorted and unique",
        ));
    }
    if posterior.particles.is_empty() {
        return Err(PortfolioVoCError::new(
            "particles",
            "at least one posterior particle is required",
        ));
    }
    if config.target_k == 0 || config.target_k > candidate_count {
        return Err(PortfolioVoCError::new(
            "target_k",
            "must be positive and no greater than the number of candidates",
        ));
    }
    if config.minimum_feasible_utility.len() != design.reference().len() {
        return Err(PortfolioVoCError::new(
            "minimum_feasible_utility",
            "requires one hard feasibility floor per objective",
        ));
    }
    for (index, floor) in config.minimum_feasible_utility.iter().copied().enumerate() {
        if !floor.is_finite()
            || !(0.0..=1.0).contains(&floor)
            || floor < design.reference()[index]
        {
            return Err(PortfolioVoCError::new(
                format!("minimum_feasible_utility[{index}]"),
                "must be finite in [reference,1]",
            ));
        }
    }

    let dimension = design.reference().len();
    let mut particle_ids = HashSet::with_capacity(posterior.particles.len());
    let mut probability_sum = 0.0;
    for (particle_index, particle) in posterior.particles.iter().enumerate() {
        if particle.particle_id.trim().is_empty()
            || !particle_ids.insert(particle.particle_id.as_str())
        {
            return Err(PortfolioVoCError::new(
                format!("particles[{particle_index}].particle_id"),
                "particle IDs must be non-empty and unique",
            ));
        }
        if !particle.probability.is_finite() || particle.probability < 0.0 {
            return Err(PortfolioVoCError::new(
                format!("particles[{particle_index}].probability"),
                "posterior probabilities must be finite and non-negative",
            ));
        }
        if particle.objective_values.len() != candidate_count {
            return Err(PortfolioVoCError::new(
                format!("particles[{particle_index}].objective_values"),
                "requires one optional objective vector per candidate",
            ));
        }
        for (candidate_index, values) in particle.objective_values.iter().enumerate() {
            match (posterior.candidate_feasible[candidate_index], values) {
                (true, None) => {
                    return Err(PortfolioVoCError::new(
                        format!("particles[{particle_index}].objective_values[{candidate_index}]"),
                        "feasible candidates require a complete posterior objective vector",
                    ));
                }
                (_, Some(vector)) => {
                    if vector.len() != dimension {
                        return Err(PortfolioVoCError::new(
                            format!("particles[{particle_index}].objective_values[{candidate_index}]"),
                            "objective vector dimension does not match the QMC design",
                        ));
                    }
                    for (objective_index, value) in vector.iter().copied().enumerate() {
                        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                            return Err(PortfolioVoCError::new(
                                format!("particles[{particle_index}].objective_values[{candidate_index}][{objective_index}]"),
                                "normalized objective values must be finite and in [0,1]",
                            ));
                        }
                    }
                }
                (false, None) => {}
            }
        }
        probability_sum += particle.probability;
    }
    if !probability_sum.is_finite()
        || (probability_sum - 1.0).abs() > PROBABILITY_TOLERANCE
    {
        return Err(PortfolioVoCError::new(
            "particles",
            "posterior probabilities must sum to one",
        ));
    }
    Ok(())
}

fn validate_and_canonicalize_portfolios(
    portfolios: &[CandidatePortfolio],
    candidate_count: usize,
    target_k: usize,
) -> Result<Vec<CandidatePortfolio>, PortfolioVoCError> {
    if portfolios.is_empty() {
        return Err(PortfolioVoCError::new(
            "portfolios",
            "at least one candidate portfolio is required",
        ));
    }
    let mut portfolio_ids = HashSet::with_capacity(portfolios.len());
    let mut output = Vec::with_capacity(portfolios.len());
    for (portfolio_index, portfolio) in portfolios.iter().enumerate() {
        if portfolio.portfolio_id.trim().is_empty()
            || !portfolio_ids.insert(portfolio.portfolio_id.as_str())
        {
            return Err(PortfolioVoCError::new(
                format!("portfolios[{portfolio_index}].portfolio_id"),
                "portfolio IDs must be non-empty and unique",
            ));
        }
        if portfolio.candidate_indices.is_empty()
            || portfolio.candidate_indices.len() > target_k
        {
            return Err(PortfolioVoCError::new(
                format!("portfolios[{portfolio_index}].candidate_indices"),
                "portfolio size must be in 1..=target_k",
            ));
        }
        let mut seen = HashSet::with_capacity(portfolio.candidate_indices.len());
        let mut candidate_indices = portfolio.candidate_indices.clone();
        for index in &candidate_indices {
            if *index >= candidate_count || !seen.insert(*index) {
                return Err(PortfolioVoCError::new(
                    format!("portfolios[{portfolio_index}].candidate_indices"),
                    "candidate indices must be unique and within the frozen candidate list",
                ));
            }
        }
        candidate_indices.sort_unstable();
        output.push(CandidatePortfolio {
            portfolio_id: portfolio.portfolio_id.clone(),
            candidate_indices,
        });
    }
    output.sort_by(|left, right| left.portfolio_id.cmp(&right.portfolio_id));
    Ok(output)
}

fn expected_hypervolume_replicates(
    posterior: &ParetoPortfolioPosterior,
    candidate_indices: &[usize],
    config: &ParetoPortfolioConfig,
    design: &HypervolumeQmcDesign,
    replicate_indices: &[usize],
) -> Result<Vec<f64>, PortfolioVoCError> {
    let mut expected_replicates = vec![0.0; replicate_indices.len()];
    for particle in &posterior.particles {
        if particle.probability == 0.0 {
            continue;
        }
        let points = feasible_points_for_indices(
            particle,
            &posterior.candidate_feasible,
            candidate_indices,
            &config.minimum_feasible_utility,
        );
        let estimate =
            estimate_hypervolume_qmc_with_replicates(&points, design, replicate_indices)?;
        let scale = estimate.box_volume / estimate.sample_count_per_replicate as f64;
        for (expected, hits) in expected_replicates
            .iter_mut()
            .zip(estimate.hit_counts_per_replicate)
        {
            *expected += particle.probability * hits as f64 * scale;
        }
    }
    Ok(expected_replicates)
}

fn feasible_points_for_indices(
    particle: &ParetoPortfolioParticle,
    candidate_feasible: &[bool],
    candidate_indices: &[usize],
    minimum_utility: &[f64],
) -> Vec<Vec<f64>> {
    candidate_indices
        .iter()
        .filter_map(|index| {
            if !candidate_feasible[*index] {
                return None;
            }
            let values = particle.objective_values[*index].as_ref()?;
            if values
                .iter()
                .zip(minimum_utility)
                .all(|(value, minimum)| value >= minimum)
            {
                Some(values.clone())
            } else {
                None
            }
        })
        .collect()
}

fn digest_serializable<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, PortfolioVoCError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| PortfolioVoCError::new("serialization", error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    Ok(encode_digest(hasher.finalize().as_slice()))
}

fn digest_portfolio_family(
    portfolios: &[CandidatePortfolio],
    config: &ParetoPortfolioConfig,
) -> Result<String, PortfolioVoCError> {
    let mut canonical = portfolios.to_vec();
    canonical.sort_by(|left, right| left.portfolio_id.cmp(&right.portfolio_id));
    digest_serializable(
        b"portfolio-family",
        &(canonical, config),
    )
}

fn digest_replicate_indices(base_digest: &str, indices: &[usize]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((base_digest.len() as u64).to_be_bytes());
    hasher.update(base_digest.as_bytes());
    for index in indices {
        hasher.update((*index as u64).to_be_bytes());
    }
    encode_digest(hasher.finalize().as_slice())
}

fn encode_digest(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_standard_deviation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let center = mean(values);
    let sum_squared = values.iter().map(|value| (value - center).powi(2)).sum::<f64>();
    (sum_squared / (values.len() - 1) as f64).sqrt()
}

fn standard_error(values: &[f64]) -> f64 {
    sample_standard_deviation(values) / (values.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware_objectives::{
        compute_objective_registry_sha256, ObjectiveDefinition,
        ObjectiveDirection, ObjectiveNormalization, ObjectiveSource,
    };

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn design() -> HypervolumeQmcDesign {
        HypervolumeQmcDesign::new(
            &[0.0, 0.0],
            &crate::pareto_hypervolume_qmc::HypervolumeQmcConfig {
                samples_per_replicate: 4096,
                scramble_replicates: 16,
                seed: 7,
            },
        )
        .unwrap()
    }

    fn posterior() -> ParetoPortfolioPosterior {
        ParetoPortfolioPosterior {
            candidate_ids: vec!["A".into(), "B".into(), "C".into()],
            objective_ids: vec!["accuracy".into(), "latency".into()],
            objective_registry_sha256: registry().registry_sha256,
            candidate_config_sha256: vec![sha('a'), sha('b'), sha('c')],
            candidate_feasible: vec![true, true, true],
            particles: vec![ParetoPortfolioParticle {
                particle_id: "p1".into(),
                probability: 1.0,
                objective_values: vec![
                    Some(vec![0.8, 0.3]),
                    Some(vec![0.4, 0.7]),
                    Some(vec![0.2, 0.2]),
                ],
            }],
        }
    }

    fn portfolios() -> Vec<CandidatePortfolio> {
        vec![
            CandidatePortfolio {
                portfolio_id: "A-only".into(),
                candidate_indices: vec![0],
            },
            CandidatePortfolio {
                portfolio_id: "B-only".into(),
                candidate_indices: vec![1],
            },
        ]
    }

    fn registry() -> ObjectiveRegistry {
        let mut value = ObjectiveRegistry {
            schema: "symthaea.nas.objective-registry".into(),
            version: "0.1.0".into(),
            benchmark_id: "portfolio-voc-test".into(),
            objectives: vec![
                ObjectiveDefinition {
                    objective_id: "accuracy".into(),
                    value_unit: "fraction".into(),
                    direction: ObjectiveDirection::Maximize,
                    source: ObjectiveSource::TrainedEvaluation,
                    normalization: ObjectiveNormalization {
                        raw_min: 0.0,
                        raw_max: 1.0,
                    },
                    measurement_protocol_sha256: sha('a'),
                    hardware_profile_sha256: None,
                    measurement_cost_unit: "objective-query".into(),
                    measurement_cost_model_sha256: sha('b'),
                },
                ObjectiveDefinition {
                    objective_id: "latency".into(),
                    value_unit: "milliseconds".into(),
                    direction: ObjectiveDirection::Minimize,
                    source: ObjectiveSource::TrainedEvaluation,
                    normalization: ObjectiveNormalization {
                        raw_min: 0.0,
                        raw_max: 100.0,
                    },
                    measurement_protocol_sha256: sha('c'),
                    hardware_profile_sha256: None,
                    measurement_cost_unit: "objective-query".into(),
                    measurement_cost_model_sha256: sha('d'),
                },
            ],
            hardware_profiles: vec![],
            registry_sha256: String::new(),
        };
        value.registry_sha256 = compute_objective_registry_sha256(&value).unwrap();
        value
    }

    fn action(posterior: &ParetoPortfolioPosterior) -> PortfolioMeasurementAction {
        let mut value = PortfolioMeasurementAction {
            action_id: "measure-A-accuracy".into(),
            candidate_id: "A".into(),
            candidate_config_sha256: posterior.candidate_config_sha256[0].clone(),
            objective_registry_sha256: registry().registry_sha256,
            objective_id: "accuracy".into(),
            hardware_profile_sha256: None,
            measurement_protocol_sha256: sha('a'),
            resource_unit: "objective-query".into(),
            incremental_cost_units: 2,
            observation_likelihood_model_sha256: String::new(),
            outcomes: vec![
                PortfolioActionOutcome {
                    outcome_id: "p1".into(),
                    likelihood_by_particle: vec![1.0, 0.0],
                },
                PortfolioActionOutcome {
                    outcome_id: "p2".into(),
                    likelihood_by_particle: vec![0.0, 1.0],
                },
            ],
        };
        value.observation_likelihood_model_sha256 =
            compute_portfolio_action_likelihood_sha256(&value).unwrap();
        value
    }

    fn two_particle_posterior() -> ParetoPortfolioPosterior {
        ParetoPortfolioPosterior {
            candidate_ids: vec!["A".into(), "B".into(), "C".into()],
            objective_ids: vec!["accuracy".into(), "latency".into()],
            objective_registry_sha256: registry().registry_sha256,
            candidate_config_sha256: vec![sha('a'), sha('b'), sha('c')],
            candidate_feasible: vec![true, true, true],
            particles: vec![
                ParetoPortfolioParticle {
                    particle_id: "p1".into(),
                    probability: 0.5,
                    objective_values: vec![
                        Some(vec![0.95, 0.95]),
                        Some(vec![0.2, 0.2]),
                        Some(vec![0.4, 0.4]),
                    ],
                },
                ParetoPortfolioParticle {
                    particle_id: "p2".into(),
                    probability: 0.5,
                    objective_values: vec![
                        Some(vec![0.2, 0.2]),
                        Some(vec![0.95, 0.95]),
                        Some(vec![0.4, 0.4]),
                    ],
                },
            ],
        }
    }

    fn two_portfolios() -> Vec<CandidatePortfolio> {
        vec![
            CandidatePortfolio {
                portfolio_id: "A-only".into(),
                candidate_indices: vec![0],
            },
            CandidatePortfolio {
                portfolio_id: "B-only".into(),
                candidate_indices: vec![1],
            },
        ]
    }

    #[test]
    fn evaluates_portfolio_regret_against_all_feasible_candidate_oracle() {
        let result = evaluate_portfolio_class_regret(
            &posterior(),
            &portfolios(),
            &ParetoPortfolioConfig {
                target_k: 1,
                minimum_feasible_utility: vec![0.0, 0.0],
            },
            &design(),
        )
        .unwrap();
        assert!(result.oracle_expected_hypervolume > 0.39);
        assert!(result.selected_portfolio_expected_hypervolume < 0.30);
        assert!(result.portfolio_class_regret > 0.10);
        assert_eq!(result.selected_portfolio_id, "B-only");
    }

    #[test]
    fn rejects_underpowered_selection_validation_scramble_split() {
        let small_design = HypervolumeQmcDesign::new(
            &[0.0, 0.0],
            &crate::pareto_hypervolume_qmc::HypervolumeQmcConfig {
                samples_per_replicate: 4096,
                scramble_replicates: 8,
                seed: 7,
            },
        )
        .unwrap();
        assert!(evaluate_portfolio_class_regret(
            &posterior(),
            &portfolios(),
            &ParetoPortfolioConfig {
                target_k: 1,
                minimum_feasible_utility: vec![0.0, 0.0],
            },
            &small_design,
        )
        .is_err());
    }

    #[test]
    fn decision_receipt_binds_posterior_portfolio_family_and_qmc_design() {
        let result = evaluate_portfolio_class_regret(
            &posterior(),
            &portfolios(),
            &ParetoPortfolioConfig {
                target_k: 1,
                minimum_feasible_utility: vec![0.0, 0.0],
            },
            &design(),
        )
        .unwrap();
        assert_eq!(result.posterior_sha256.len(), 64);
        assert_eq!(result.portfolio_family_sha256.len(), 64);
        assert_eq!(result.qmc_sampling_design_sha256.len(), 64);
        assert_eq!(result.selection_replicates_sha256.len(), 64);
        assert_eq!(result.validation_replicates_sha256.len(), 64);
        assert_ne!(result.selection_replicates_sha256, result.validation_replicates_sha256);
    }

    #[test]
    fn infeasible_candidates_do_not_receive_fabricated_objective_vectors() {
        let mut posterior = posterior();
        posterior.candidate_feasible[2] = false;
        posterior.particles[0].objective_values[2] = None;
        let result = evaluate_portfolio_class_regret(
            &posterior,
            &portfolios(),
            &ParetoPortfolioConfig {
                target_k: 1,
                minimum_feasible_utility: vec![0.0, 0.0],
            },
            &design(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn hard_minimum_utility_filters_candidates_from_the_frontier() {
        let result = evaluate_portfolio_class_regret(
            &posterior(),
            &portfolios(),
            &ParetoPortfolioConfig {
                target_k: 1,
                minimum_feasible_utility: vec![0.5, 0.5],
            },
            &design(),
        )
        .unwrap();
        assert_eq!(result.oracle_expected_hypervolume, 0.0);
        assert_eq!(result.portfolio_class_regret, 0.0);
    }

    #[test]
    fn selection_and_validation_scrambles_are_disjoint() {
        let result = evaluate_portfolio_class_regret(
            &posterior(),
            &portfolios(),
            &ParetoPortfolioConfig {
                target_k: 1,
                minimum_feasible_utility: vec![0.0, 0.0],
            },
            &design(),
        )
        .unwrap();
        assert!(result
            .selection_replicate_indices
            .iter()
            .all(|index| !result.validation_replicate_indices.contains(index)));
    }

    #[test]
    fn portfolio_order_and_candidate_index_order_are_canonicalized() {
        let mut reverse = portfolios();
        reverse.reverse();
        reverse[0].candidate_indices.reverse();
        let cfg = ParetoPortfolioConfig {
            target_k: 1,
            minimum_feasible_utility: vec![0.0, 0.0],
        };
        let first = evaluate_portfolio_class_regret(&posterior(), &portfolios(), &cfg, &design())
            .unwrap();
        let second =
            evaluate_portfolio_class_regret(&posterior(), &reverse, &cfg, &design()).unwrap();
        assert_eq!(first.selected_portfolio_id, second.selected_portfolio_id);
        assert_eq!(first.portfolio_class_regret, second.portfolio_class_regret);
    }

    #[test]
    fn perfectly_informative_objective_measurement_reduces_portfolio_regret() {
        let posterior = two_particle_posterior();
        let portfolios = two_portfolios();
        let cfg = ParetoPortfolioConfig {
            target_k: 1,
            minimum_feasible_utility: vec![0.0, 0.0],
        };
        let action = action(&posterior);
        let result = evaluate_portfolio_measurement_voc(
            &registry(),
            &posterior,
            &portfolios,
            &cfg,
            &design(),
            &action,
        )
        .unwrap();
        assert!(result.current_decision.portfolio_class_regret > 0.15);
        assert!(result.expected_posterior_regret.abs() < 0.01);
        assert!(result.estimated_regret_reduction > 0.15);
        assert_eq!(result.incremental_cost_units, 2);
        assert!((result.regret_reduction_per_cost_unit
            - result.estimated_regret_reduction / 2.0)
            .abs()
            < 1e-12);
    }

    #[test]
    fn uninformative_objective_measurement_has_zero_regret_reduction() {
        let posterior = two_particle_posterior();
        let portfolios = two_portfolios();
        let cfg = ParetoPortfolioConfig {
            target_k: 1,
            minimum_feasible_utility: vec![0.0, 0.0],
        };
        let mut action = action(&posterior);
        action.outcomes = vec![PortfolioActionOutcome {
            outcome_id: "same".into(),
            likelihood_by_particle: vec![1.0, 1.0],
        }];
        action.observation_likelihood_model_sha256 =
            compute_portfolio_action_likelihood_sha256(&action).unwrap();
        let result = evaluate_portfolio_measurement_voc(
            &registry(),
            &posterior,
            &portfolios,
            &cfg,
            &design(),
            &action,
        )
        .unwrap();
        assert!(result.estimated_regret_reduction.abs() < 0.01);
    }

    #[test]
    fn action_likelihood_digest_tampering_fails_closed() {
        let posterior = two_particle_posterior();
        let mut action = action(&posterior);
        action.outcomes[0].likelihood_by_particle = vec![0.5, 0.5];
        assert!(evaluate_portfolio_measurement_voc(
            &registry(),
            &posterior,
            &two_portfolios(),
            &ParetoPortfolioConfig {
                target_k: 1,
                minimum_feasible_utility: vec![0.0, 0.0],
            },
            &design(),
            &action,
        )
        .is_err());
    }

    #[test]
    fn action_cannot_be_replayed_with_a_different_candidate_configuration() {
        let posterior = two_particle_posterior();
        let mut action = action(&posterior);
        action.candidate_config_sha256 = sha('f');
        action.observation_likelihood_model_sha256 =
            compute_portfolio_action_likelihood_sha256(&action).unwrap();
        assert!(evaluate_portfolio_measurement_voc(
            &registry(),
            &posterior,
            &two_portfolios(),
            &ParetoPortfolioConfig {
                target_k: 1,
                minimum_feasible_utility: vec![0.0, 0.0],
            },
            &design(),
            &action,
        )
        .is_err());
    }

    #[test]
    fn action_cannot_be_replayed_against_a_different_objective_registry() {
        let posterior = two_particle_posterior();
        let mut action = action(&posterior);
        action.objective_registry_sha256 = sha('f');
        action.observation_likelihood_model_sha256 =
            compute_portfolio_action_likelihood_sha256(&action).unwrap();
        assert!(evaluate_portfolio_measurement_voc(
            &registry(),
            &posterior,
            &two_portfolios(),
            &ParetoPortfolioConfig {
                target_k: 1,
                minimum_feasible_utility: vec![0.0, 0.0],
            },
            &design(),
            &action,
        )
        .is_err());
    }

    #[test]
    fn rejects_posterior_with_a_different_objective_registry_digest() {
        let posterior = two_particle_posterior();
        let mut changed_registry = registry();
        changed_registry.benchmark_id = "another-benchmark".into();
        changed_registry.registry_sha256 =
            compute_objective_registry_sha256(&changed_registry).unwrap();
        let mut action = action(&posterior);
        action.objective_registry_sha256 = changed_registry.registry_sha256.clone();
        action.observation_likelihood_model_sha256 =
            compute_portfolio_action_likelihood_sha256(&action).unwrap();
        assert!(evaluate_portfolio_measurement_voc(
            &changed_registry,
            &posterior,
            &two_portfolios(),
            &ParetoPortfolioConfig {
                target_k: 1,
                minimum_feasible_utility: vec![0.0, 0.0],
            },
            &design(),
            &action,
        )
        .is_err());
    }

    #[test]
    fn rejects_posterior_objective_column_order_mismatch() {
        let mut posterior = two_particle_posterior();
        posterior.objective_ids.reverse();
        let action = action(&posterior);
        assert!(evaluate_portfolio_measurement_voc(
            &registry(),
            &posterior,
            &two_portfolios(),
            &ParetoPortfolioConfig {
                target_k: 1,
                minimum_feasible_utility: vec![0.0, 0.0],
            },
            &design(),
            &action,
        )
        .is_err());
    }

    #[test]
    fn invalid_action_cost_unit_fails_closed() {
        let posterior = two_particle_posterior();
        let mut action = action(&posterior);
        action.resource_unit = "milliseconds".into();
        action.observation_likelihood_model_sha256 =
            compute_portfolio_action_likelihood_sha256(&action).unwrap();
        assert!(evaluate_portfolio_measurement_voc(
            &registry(),
            &posterior,
            &two_portfolios(),
            &ParetoPortfolioConfig {
                target_k: 1,
                minimum_feasible_utility: vec![0.0, 0.0],
            },
            &design(),
            &action,
        )
        .is_err());
    }

    #[test]
    fn invalid_portfolios_and_objective_dimensions_are_rejected() {
        let duplicate = vec![CandidatePortfolio {
            portfolio_id: "bad".into(),
            candidate_indices: vec![0, 0],
        }];
        assert!(evaluate_portfolio_class_regret(
            &posterior(),
            &duplicate,
            &ParetoPortfolioConfig {
                target_k: 2,
                minimum_feasible_utility: vec![0.0, 0.0],
            },
            &design(),
        )
        .is_err());

        let mut malformed = posterior();
        malformed.particles[0].objective_values[0] = Some(vec![0.5]);
        assert!(evaluate_portfolio_class_regret(
            &malformed,
            &portfolios(),
            &ParetoPortfolioConfig {
                target_k: 1,
                minimum_feasible_utility: vec![0.0, 0.0],
            },
            &design(),
        )
        .is_err());
    }
}
