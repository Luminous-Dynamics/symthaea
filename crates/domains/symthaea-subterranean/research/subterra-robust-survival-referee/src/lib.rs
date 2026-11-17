//! Independent multi-model referee for Subterra survival-conditioned policy trees.
//!
//! The referee evaluates one supplied policy unchanged across every model.
//! It independently recomputes probabilities and posteriors; planner-cached
//! branch probabilities and posterior vectors are diagnostics only.
//!
//! This crate is an offline evaluator, not an optimizer, controller, safety
//! case, evidence authenticator, dispatch authority, or physical qualification.

use std::collections::{BTreeMap, BTreeSet};

use subterra_survival_reference::{
    CostPrices, MeasurementAction, MissionDecision, OutcomeKind, PlanningProblem, PolicyBranch,
    PolicyNode, ResourceBudget, RiskLimits,
};

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const NUMERICAL_TOLERANCE: f64 = 1.0e-8;
const MAX_MODELS: usize = 32;
const MAX_ACTIONS: usize = 32;
const MAX_OUTCOMES_PER_ACTION: usize = 16;
const MAX_POLICY_DEPTH: usize = 16;
const MAX_POLICY_NODES: usize = 10_000;
const UNKNOWN_HYPOTHESIS_ID: &str = "unknown";

#[derive(Debug, Clone, PartialEq)]
pub struct ModelScenario {
    pub model_id: String,
    /// Priors and likelihoods may differ; semantic/action/loss schemas may not.
    pub problem: PlanningProblem,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ResourceUse {
    pub energy_j: f64,
    pub duration_s: f64,
    pub data_bytes: f64,
    pub exposure_units: f64,
}

impl ResourceUse {
    fn from_budget(value: ResourceBudget) -> Self {
        Self {
            energy_j: value.energy_j,
            duration_s: value.duration_s,
            data_bytes: value.data_bytes,
            exposure_units: value.exposure_units,
        }
    }

    fn add_cost(
        self,
        cost: subterra_survival_reference::ResourceCost,
    ) -> Result<Self, RefereeError> {
        let result = Self {
            energy_j: self.energy_j + cost.energy_j,
            duration_s: self.duration_s + cost.duration_s,
            data_bytes: self.data_bytes + cost.data_bytes,
            exposure_units: self.exposure_units + cost.exposure_units,
        };
        if result.is_finite_nonnegative() {
            Ok(result)
        } else {
            Err(RefereeError::NumericalOverflow(
                "path resource accumulation",
            ))
        }
    }

    fn plus_weighted(self, future: Self, probability: f64) -> Result<Self, RefereeError> {
        let result = Self {
            energy_j: self.energy_j + future.energy_j * probability,
            duration_s: self.duration_s + future.duration_s * probability,
            data_bytes: self.data_bytes + future.data_bytes * probability,
            exposure_units: self.exposure_units + future.exposure_units * probability,
        };
        if result.is_finite_nonnegative() {
            Ok(result)
        } else {
            Err(RefereeError::NumericalOverflow(
                "expected resource accumulation",
            ))
        }
    }

    fn component_max(self, other: Self) -> Self {
        Self {
            energy_j: self.energy_j.max(other.energy_j),
            duration_s: self.duration_s.max(other.duration_s),
            data_bytes: self.data_bytes.max(other.data_bytes),
            exposure_units: self.exposure_units.max(other.exposure_units),
        }
    }

    fn within(self, budget: ResourceBudget) -> bool {
        self.energy_j <= budget.energy_j + NUMERICAL_TOLERANCE
            && self.duration_s <= budget.duration_s + NUMERICAL_TOLERANCE
            && self.data_bytes <= budget.data_bytes + NUMERICAL_TOLERANCE
            && self.exposure_units <= budget.exposure_units + NUMERICAL_TOLERANCE
    }

    fn fits_cost(cost: subterra_survival_reference::ResourceCost, budget: ResourceBudget) -> bool {
        cost.energy_j <= budget.energy_j + NUMERICAL_TOLERANCE
            && cost.duration_s <= budget.duration_s + NUMERICAL_TOLERANCE
            && cost.data_bytes <= budget.data_bytes + NUMERICAL_TOLERANCE
            && cost.exposure_units <= budget.exposure_units + NUMERICAL_TOLERANCE
    }

    fn subtract_cost(
        budget: ResourceBudget,
        cost: subterra_survival_reference::ResourceCost,
    ) -> ResourceBudget {
        ResourceBudget {
            energy_j: (budget.energy_j - cost.energy_j).max(0.0),
            duration_s: (budget.duration_s - cost.duration_s).max(0.0),
            data_bytes: (budget.data_bytes - cost.data_bytes).max(0.0),
            exposure_units: (budget.exposure_units - cost.exposure_units).max(0.0),
        }
    }

    fn weighted(
        cost: subterra_survival_reference::ResourceCost,
        prices: CostPrices,
    ) -> Result<f64, RefereeError> {
        let value = cost.energy_j * prices.energy_per_j
            + cost.duration_s * prices.time_per_s
            + cost.data_bytes * prices.data_per_byte
            + cost.exposure_units * prices.exposure_per_unit;
        if value.is_finite() && value >= 0.0 {
            Ok(value)
        } else {
            Err(RefereeError::NumericalOverflow("weighted resource cost"))
        }
    }

    fn is_finite_nonnegative(self) -> bool {
        [
            self.energy_j,
            self.duration_s,
            self.data_bytes,
            self.exposure_units,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefereeConfig {
    pub resource_budget: ResourceBudget,
    pub cost_prices: CostPrices,
    pub risk_limits: RiskLimits,
    pub maximum_models: usize,
    pub maximum_policy_nodes: usize,
    pub maximum_policy_depth: usize,
}

impl Default for RefereeConfig {
    fn default() -> Self {
        Self {
            resource_budget: ResourceBudget {
                energy_j: 10_000.0,
                duration_s: 3_600.0,
                data_bytes: 10_000_000.0,
                exposure_units: 1.0,
            },
            cost_prices: CostPrices::default(),
            risk_limits: RiskLimits {
                maximum_mission_interruption_probability: 0.10,
                maximum_robot_loss_probability: 0.05,
            },
            maximum_models: MAX_MODELS,
            maximum_policy_nodes: MAX_POLICY_NODES,
            maximum_policy_depth: MAX_POLICY_DEPTH,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelPolicyMetrics {
    pub model_id: String,
    pub expected_decision_loss: f64,
    pub expected_resource_cost_units: f64,
    pub expected_total_value: f64,
    pub expected_resources: ResourceUse,
    /// Component-wise maxima across supported root-to-leaf policy paths.
    /// The maxima for different components need not occur on the same path.
    pub maximum_path_resources: ResourceUse,
    pub mission_interruption_probability: f64,
    pub robot_loss_probability: f64,
    pub external_gates_passed: bool,
    pub all_supported_paths_within_budget: bool,
    pub within_risk_limits: bool,
    /// Number of cached branch probability/posterior records that disagree
    /// with this model's independently recomputed values.
    pub cached_branch_mismatches: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyDisposition {
    AdmissibleAcrossSuppliedModels,
    InadmissibleAcrossSuppliedModels,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustPolicyReport {
    pub model_count: usize,
    pub per_model: Vec<ModelPolicyMetrics>,
    pub worst_case_expected_total_value: f64,
    pub best_case_expected_total_value: f64,
    /// Descriptive unweighted arithmetic mean, not a model-weighted expectation.
    pub mean_expected_total_value: f64,
    pub maximum_mission_interruption_probability: f64,
    pub maximum_robot_loss_probability: f64,
    pub all_external_gates_passed: bool,
    pub all_supported_paths_within_budget: bool,
    pub all_models_within_risk_limits: bool,
    pub disposition: PolicyDisposition,
    pub violations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefereeError {
    EmptyModelSet,
    TooManyModels,
    EmptyModelId,
    DuplicateModelId(String),
    InvalidConfig(&'static str),
    InvalidModel(&'static str),
    DuplicateIdentifier(String),
    MissingUnknownHypothesis,
    MultipleUnknownHypotheses,
    UnknownHypothesisHasZeroPrior,
    ProbabilityNotNormalized {
        context: String,
        sum_bits: u64,
    },
    ModelSchemaMismatch(String),
    PolicyNodeLimitExceeded,
    PolicyDepthExceeded,
    PolicyHasBothActionAndStop,
    PolicyHasNeitherActionNorStop,
    UnknownAction(String),
    UnknownDecision(String),
    UnknownOutcome {
        action_id: String,
        outcome_id: String,
    },
    DuplicatePolicyOutcome {
        action_id: String,
        outcome_id: String,
    },
    MissingPolicyOutcome {
        action_id: String,
        outcome_id: String,
    },
    PolicyOutcomeKindMismatch {
        action_id: String,
        outcome_id: String,
    },
    ObservationMissingContinuation(String),
    TerminalHasContinuation(String),
    StopNodeHasBranches(String),
    PolicyOutcomeMassMismatch {
        action_id: String,
        sum_bits: u64,
    },
    PosteriorUndefined(String),
    NumericalOverflow(&'static str),
}

impl std::fmt::Display for RefereeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RefereeError {}

#[derive(Debug, Clone, Copy)]
struct PartialMetrics {
    expected_decision_loss: f64,
    expected_resource_cost_units: f64,
    expected_resources: ResourceUse,
    maximum_path_resources: ResourceUse,
    mission_interruption_probability: f64,
    robot_loss_probability: f64,
    external_gates_passed: bool,
    all_supported_paths_within_budget: bool,
    cached_branch_mismatches: usize,
}

fn validate_config(config: RefereeConfig) -> Result<(), RefereeError> {
    let b = config.resource_budget;
    if ![
        b.energy_j,
        b.duration_s,
        b.data_bytes,
        b.exposure_units,
        config.cost_prices.energy_per_j,
        config.cost_prices.time_per_s,
        config.cost_prices.data_per_byte,
        config.cost_prices.exposure_per_unit,
    ]
    .into_iter()
    .all(|value| value.is_finite() && value >= 0.0)
    {
        return Err(RefereeError::InvalidConfig(
            "budget and resource prices must be finite and non-negative",
        ));
    }
    if !config
        .risk_limits
        .maximum_mission_interruption_probability
        .is_finite()
        || !(0.0..=1.0).contains(&config.risk_limits.maximum_mission_interruption_probability)
    {
        return Err(RefereeError::InvalidConfig("mission interruption limit"));
    }
    if !config
        .risk_limits
        .maximum_robot_loss_probability
        .is_finite()
        || !(0.0..=1.0).contains(&config.risk_limits.maximum_robot_loss_probability)
    {
        return Err(RefereeError::InvalidConfig("robot loss limit"));
    }
    if config.maximum_models == 0
        || config.maximum_models > MAX_MODELS
        || config.maximum_policy_nodes == 0
        || config.maximum_policy_nodes > MAX_POLICY_NODES
        || config.maximum_policy_depth == 0
        || config.maximum_policy_depth > MAX_POLICY_DEPTH
    {
        return Err(RefereeError::InvalidConfig(
            "limits exceed the bounded v0.1 envelope",
        ));
    }
    Ok(())
}

fn validate_problem(problem: &PlanningProblem) -> Result<(), RefereeError> {
    if problem.hypotheses.is_empty() || problem.decisions.is_empty() {
        return Err(RefereeError::InvalidModel(
            "hypotheses and decisions must be non-empty",
        ));
    }
    if problem.actions.len() > MAX_ACTIONS {
        return Err(RefereeError::InvalidModel(
            "action count exceeds v0.1 bound",
        ));
    }

    let mut hypothesis_ids = BTreeSet::new();
    let mut prior_sum = 0.0;
    let mut unknown_count = 0usize;
    let mut unknown_prior = 0.0;
    for hypothesis in &problem.hypotheses {
        if hypothesis.id.trim().is_empty() || !hypothesis_ids.insert(hypothesis.id.as_str()) {
            return Err(RefereeError::DuplicateIdentifier(hypothesis.id.clone()));
        }
        if !hypothesis.prior_probability.is_finite()
            || !(0.0..=1.0).contains(&hypothesis.prior_probability)
        {
            return Err(RefereeError::InvalidModel("invalid hypothesis prior"));
        }
        prior_sum += hypothesis.prior_probability;
        if !prior_sum.is_finite() {
            return Err(RefereeError::NumericalOverflow("prior sum"));
        }
        if hypothesis.id == UNKNOWN_HYPOTHESIS_ID {
            unknown_count += 1;
            unknown_prior = hypothesis.prior_probability;
        }
    }
    if unknown_count == 0 {
        return Err(RefereeError::MissingUnknownHypothesis);
    }
    if unknown_count > 1 {
        return Err(RefereeError::MultipleUnknownHypotheses);
    }
    if unknown_prior <= 0.0 {
        return Err(RefereeError::UnknownHypothesisHasZeroPrior);
    }
    if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(RefereeError::ProbabilityNotNormalized {
            context: "prior".into(),
            sum_bits: prior_sum.to_bits(),
        });
    }

    let hypothesis_count = problem.hypotheses.len();
    let mut decisions = BTreeSet::new();
    for decision in &problem.decisions {
        if decision.id.trim().is_empty() || !decisions.insert(decision.id.as_str()) {
            return Err(RefereeError::DuplicateIdentifier(decision.id.clone()));
        }
        if decision.loss_by_hypothesis.len() != hypothesis_count
            || decision
                .loss_by_hypothesis
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(RefereeError::InvalidModel("invalid decision loss vector"));
        }
    }

    let mut actions = BTreeSet::new();
    for action in &problem.actions {
        validate_action(action, hypothesis_count)?;
        if !actions.insert(action.id.as_str()) {
            return Err(RefereeError::DuplicateIdentifier(action.id.clone()));
        }
    }
    Ok(())
}

fn validate_action(
    action: &MeasurementAction,
    hypothesis_count: usize,
) -> Result<(), RefereeError> {
    if action.id.trim().is_empty() || action.modality.trim().is_empty() {
        return Err(RefereeError::InvalidModel("empty action ID or modality"));
    }
    let cost = action.cost;
    if ![
        cost.energy_j,
        cost.duration_s,
        cost.data_bytes,
        cost.exposure_units,
    ]
    .into_iter()
    .all(|value| value.is_finite() && value >= 0.0)
    {
        return Err(RefereeError::InvalidModel("invalid action resource cost"));
    }
    if action.outcomes.is_empty() || action.outcomes.len() > MAX_OUTCOMES_PER_ACTION {
        return Err(RefereeError::InvalidModel(
            "outcome count outside v0.1 bound",
        ));
    }

    let mut outcome_ids = BTreeSet::new();
    for outcome in &action.outcomes {
        if outcome.id.trim().is_empty() || !outcome_ids.insert(outcome.id.as_str()) {
            return Err(RefereeError::DuplicateIdentifier(outcome.id.clone()));
        }
        if outcome.likelihood_by_hypothesis.len() != hypothesis_count
            || outcome
                .likelihood_by_hypothesis
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(RefereeError::InvalidModel(
                "invalid outcome likelihood vector",
            ));
        }

        match (outcome.kind, outcome.terminal_loss_by_hypothesis.as_ref()) {
            (OutcomeKind::Observation, None) => {}
            (OutcomeKind::Observation, Some(_)) => {
                return Err(RefereeError::InvalidModel("observation has terminal loss"));
            }
            (_, None) => {
                return Err(RefereeError::InvalidModel(
                    "terminal outcome has no terminal loss",
                ));
            }
            (_, Some(losses))
                if losses.len() != hypothesis_count
                    || losses
                        .iter()
                        .any(|value| !value.is_finite() || *value < 0.0) =>
            {
                return Err(RefereeError::InvalidModel("invalid terminal loss vector"));
            }
            _ => {}
        }
    }

    let mut ordered_outcomes = action.outcomes.iter().collect::<Vec<_>>();
    ordered_outcomes.sort_by(|a, b| a.id.cmp(&b.id));
    for hypothesis_index in 0..hypothesis_count {
        let sum = ordered_outcomes
            .iter()
            .map(|outcome| outcome.likelihood_by_hypothesis[hypothesis_index])
            .sum::<f64>();
        if !sum.is_finite() {
            return Err(RefereeError::NumericalOverflow("likelihood sum"));
        }
        if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(RefereeError::ProbabilityNotNormalized {
                context: format!("action {} hypothesis {hypothesis_index}", action.id),
                sum_bits: sum.to_bits(),
            });
        }
    }
    Ok(())
}

fn validate_model_schema(models: &[ModelScenario]) -> Result<(), RefereeError> {
    let first = &models[0].problem;
    let first_hypotheses = first
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.id.as_str())
        .collect::<Vec<_>>();
    let first_decisions = first
        .decisions
        .iter()
        .map(|decision| (decision.id.as_str(), decision.loss_by_hypothesis.as_slice()))
        .collect::<Vec<_>>();
    let mut first_actions = BTreeMap::new();
    for action in &first.actions {
        let mut outcomes = action.outcomes.iter().collect::<Vec<_>>();
        outcomes.sort_by(|a, b| a.id.cmp(&b.id));
        first_actions.insert(
            action.id.as_str(),
            (
                action.modality.as_str(),
                action.cost,
                outcomes
                    .into_iter()
                    .map(|outcome| {
                        (
                            outcome.id.as_str(),
                            outcome.kind,
                            outcome.terminal_loss_by_hypothesis.as_deref(),
                        )
                    })
                    .collect::<Vec<_>>(),
            ),
        );
    }

    for model in models.iter().skip(1) {
        let problem = &model.problem;
        let hypothesis_ids = problem
            .hypotheses
            .iter()
            .map(|hypothesis| hypothesis.id.as_str())
            .collect::<Vec<_>>();
        if hypothesis_ids != first_hypotheses {
            return Err(RefereeError::ModelSchemaMismatch(format!(
                "model {} changes hypothesis identity/order",
                model.model_id
            )));
        }
        let decisions = problem
            .decisions
            .iter()
            .map(|decision| (decision.id.as_str(), decision.loss_by_hypothesis.as_slice()))
            .collect::<Vec<_>>();
        if decisions != first_decisions {
            return Err(RefereeError::ModelSchemaMismatch(format!(
                "model {} changes decision identity/order/loss",
                model.model_id
            )));
        }
        if problem.actions.len() != first_actions.len() {
            return Err(RefereeError::ModelSchemaMismatch(format!(
                "model {} changes action count",
                model.model_id
            )));
        }
        for action in &problem.actions {
            let Some((modality, cost, expected_outcomes)) = first_actions.get(action.id.as_str())
            else {
                return Err(RefereeError::ModelSchemaMismatch(format!(
                    "model {} adds/removes action {}",
                    model.model_id, action.id
                )));
            };
            let mut actual_outcomes = action.outcomes.iter().collect::<Vec<_>>();
            actual_outcomes.sort_by(|a, b| a.id.cmp(&b.id));
            let actual = actual_outcomes
                .into_iter()
                .map(|outcome| {
                    (
                        outcome.id.as_str(),
                        outcome.kind,
                        outcome.terminal_loss_by_hypothesis.as_deref(),
                    )
                })
                .collect::<Vec<_>>();
            if action.modality.as_str() != *modality
                || action.cost != *cost
                || actual.as_slice() != expected_outcomes.as_slice()
            {
                return Err(RefereeError::ModelSchemaMismatch(format!(
                    "model {} changes action/outcome semantics for {}",
                    model.model_id, action.id
                )));
            }
        }
    }
    Ok(())
}

fn find_action<'a>(
    problem: &'a PlanningProblem,
    action_id: &str,
) -> Result<&'a MeasurementAction, RefereeError> {
    problem
        .actions
        .iter()
        .find(|action| action.id == action_id)
        .ok_or_else(|| RefereeError::UnknownAction(action_id.to_owned()))
}

fn validate_policy(
    node: &PolicyNode,
    models: &[ModelScenario],
    depth: usize,
    node_count: &mut usize,
    config: RefereeConfig,
) -> Result<(), RefereeError> {
    *node_count = (*node_count).saturating_add(1);
    if *node_count > config.maximum_policy_nodes {
        return Err(RefereeError::PolicyNodeLimitExceeded);
    }
    if depth > config.maximum_policy_depth {
        return Err(RefereeError::PolicyDepthExceeded);
    }

    match (&node.measurement_action_id, &node.stop_decision_id) {
        (Some(_), Some(_)) => return Err(RefereeError::PolicyHasBothActionAndStop),
        (None, None) => return Err(RefereeError::PolicyHasNeitherActionNorStop),
        (None, Some(decision_id)) => {
            if node.branches.is_empty() {
                if models[0]
                    .problem
                    .decisions
                    .iter()
                    .any(|decision| decision.id.as_str() == decision_id)
                {
                    return Ok(());
                }
                return Err(RefereeError::UnknownDecision(decision_id.clone()));
            }
            return Err(RefereeError::StopNodeHasBranches(decision_id.clone()));
        }
        (Some(action_id), None) => {
            let first_action = find_action(&models[0].problem, action_id)?;
            let mut branch_by_id = BTreeMap::new();
            for branch in &node.branches {
                if branch_by_id
                    .insert(branch.outcome_id.as_str(), branch)
                    .is_some()
                {
                    return Err(RefereeError::DuplicatePolicyOutcome {
                        action_id: action_id.clone(),
                        outcome_id: branch.outcome_id.clone(),
                    });
                }
                let Some(outcome) = first_action
                    .outcomes
                    .iter()
                    .find(|outcome| outcome.id == branch.outcome_id)
                else {
                    return Err(RefereeError::UnknownOutcome {
                        action_id: action_id.clone(),
                        outcome_id: branch.outcome_id.clone(),
                    });
                };
                if branch.kind != outcome.kind {
                    return Err(RefereeError::PolicyOutcomeKindMismatch {
                        action_id: action_id.clone(),
                        outcome_id: branch.outcome_id.clone(),
                    });
                }
                match (outcome.kind, branch.continuation.as_deref()) {
                    (OutcomeKind::Observation, None) => {
                        return Err(RefereeError::ObservationMissingContinuation(
                            branch.outcome_id.clone(),
                        ));
                    }
                    (OutcomeKind::Observation, Some(child)) => {
                        validate_policy(child, models, depth + 1, node_count, config)?;
                    }
                    (_, Some(_)) => {
                        return Err(RefereeError::TerminalHasContinuation(
                            branch.outcome_id.clone(),
                        ));
                    }
                    (_, None) => {}
                }
            }

            // Require a branch for every outcome that is possible in at least
            // one supplied model, so model-specific support cannot disappear.
            for outcome in &first_action.outcomes {
                let possible_in_any_model = models.iter().any(|model| {
                    model
                        .problem
                        .actions
                        .iter()
                        .find(|candidate| candidate.id.as_str() == action_id)
                        .and_then(|action| {
                            action
                                .outcomes
                                .iter()
                                .find(|candidate| candidate.id == outcome.id)
                        })
                        .is_some_and(|candidate| {
                            candidate
                                .likelihood_by_hypothesis
                                .iter()
                                .any(|value| *value > 0.0)
                        })
                });
                if possible_in_any_model && !branch_by_id.contains_key(outcome.id.as_str()) {
                    return Err(RefereeError::MissingPolicyOutcome {
                        action_id: action_id.clone(),
                        outcome_id: outcome.id.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn expected_loss(posterior: &[f64], losses: &[f64]) -> Result<f64, RefereeError> {
    if posterior.len() != losses.len() {
        return Err(RefereeError::InvalidModel(
            "loss/posterior dimension mismatch",
        ));
    }
    let value = posterior
        .iter()
        .zip(losses)
        .map(|(p, loss)| p * loss)
        .sum::<f64>();
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(RefereeError::NumericalOverflow("expected terminal loss"))
    }
}

fn outcome_probability(posterior: &[f64], likelihood: &[f64]) -> Result<f64, RefereeError> {
    let value = posterior
        .iter()
        .zip(likelihood)
        .map(|(p, l)| p * l)
        .sum::<f64>();
    if value.is_finite() && value >= 0.0 && value <= 1.0 + PROBABILITY_TOLERANCE {
        Ok(value.min(1.0))
    } else {
        Err(RefereeError::NumericalOverflow("outcome probability"))
    }
}

fn update_posterior(
    posterior: &[f64],
    likelihood: &[f64],
    probability: f64,
    outcome_id: &str,
) -> Result<Vec<f64>, RefereeError> {
    if probability <= 0.0 {
        return Err(RefereeError::PosteriorUndefined(outcome_id.to_owned()));
    }
    let updated = posterior
        .iter()
        .zip(likelihood)
        .map(|(prior, conditional)| prior * conditional / probability)
        .collect::<Vec<_>>();
    let sum = updated.iter().sum::<f64>();
    if !sum.is_finite() || (sum - 1.0).abs() > NUMERICAL_TOLERANCE {
        return Err(RefereeError::NumericalOverflow("posterior normalization"));
    }
    Ok(updated)
}

fn cache_mismatch(branch: &PolicyBranch, probability: f64, posterior: Option<&[f64]>) -> bool {
    if !branch.probability.is_finite()
        || (branch.probability - probability).abs() > NUMERICAL_TOLERANCE
    {
        return true;
    }
    let Some(expected) = posterior else {
        return false;
    };
    branch.posterior.len() != expected.len()
        || branch
            .posterior
            .iter()
            .zip(expected)
            .any(|(actual, expected)| {
                !actual.is_finite() || (actual - expected).abs() > NUMERICAL_TOLERANCE
            })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_node(
    problem: &PlanningProblem,
    node: &PolicyNode,
    posterior: &[f64],
    remaining_budget: ResourceBudget,
    spent: ResourceUse,
    root_budget: ResourceBudget,
    prices: CostPrices,
) -> Result<PartialMetrics, RefereeError> {
    if let Some(decision_id) = node.stop_decision_id.as_deref() {
        let decision = problem
            .decisions
            .iter()
            .find(|candidate| candidate.id == decision_id)
            .ok_or_else(|| RefereeError::UnknownDecision(decision_id.to_owned()))?;
        return Ok(PartialMetrics {
            expected_decision_loss: expected_loss(posterior, &decision.loss_by_hypothesis)?,
            expected_resource_cost_units: 0.0,
            expected_resources: ResourceUse::default(),
            maximum_path_resources: spent,
            mission_interruption_probability: 0.0,
            robot_loss_probability: 0.0,
            external_gates_passed: true,
            all_supported_paths_within_budget: spent.within(root_budget),
            cached_branch_mismatches: 0,
        });
    }

    let action_id = node
        .measurement_action_id
        .as_deref()
        .ok_or(RefereeError::PolicyHasNeitherActionNorStop)?;
    let action = find_action(problem, action_id)?;
    let action_cost_value = ResourceUse::weighted(action.cost, prices)?;
    let action_fits = ResourceUse::fits_cost(action.cost, remaining_budget);
    let budget_after = ResourceUse::subtract_cost(remaining_budget, action.cost);
    let spent_after = spent.add_cost(action.cost)?;

    let mut expected_decision_loss = 0.0;
    let mut expected_future_cost = 0.0;
    let mut expected_resources = ResourceUse::from_budget(ResourceBudget {
        energy_j: action.cost.energy_j,
        duration_s: action.cost.duration_s,
        data_bytes: action.cost.data_bytes,
        exposure_units: action.cost.exposure_units,
    });
    let mut maximum_path_resources = spent_after;
    let mut interruption_probability = 0.0;
    let mut robot_loss_probability = 0.0;
    let mut external_gates_passed = action.external_gate_passed;
    let mut all_paths_within_budget = action_fits && spent_after.within(root_budget);
    let mut cached_branch_mismatches = 0usize;
    let mut total_branch_probability = 0.0;

    let mut ordered_branches = node.branches.iter().collect::<Vec<_>>();
    ordered_branches.sort_by(|a, b| a.outcome_id.cmp(&b.outcome_id));

    for branch in ordered_branches {
        let outcome = action
            .outcomes
            .iter()
            .find(|candidate| candidate.id == branch.outcome_id)
            .ok_or_else(|| RefereeError::UnknownOutcome {
                action_id: action_id.to_owned(),
                outcome_id: branch.outcome_id.clone(),
            })?;
        let probability = outcome_probability(posterior, &outcome.likelihood_by_hypothesis)?;
        total_branch_probability += probability;

        if probability <= 0.0 {
            if cache_mismatch(branch, probability, None) {
                cached_branch_mismatches += 1;
            }
            continue;
        }

        let next_posterior = update_posterior(
            posterior,
            &outcome.likelihood_by_hypothesis,
            probability,
            &outcome.id,
        )?;
        if cache_mismatch(branch, probability, Some(&next_posterior)) {
            cached_branch_mismatches += 1;
        }

        if outcome.kind == OutcomeKind::Observation {
            let child = branch
                .continuation
                .as_deref()
                .ok_or_else(|| RefereeError::ObservationMissingContinuation(outcome.id.clone()))?;
            let future = evaluate_node(
                problem,
                child,
                &next_posterior,
                budget_after,
                spent_after,
                root_budget,
                prices,
            )?;
            expected_decision_loss += probability * future.expected_decision_loss;
            expected_future_cost += probability * future.expected_resource_cost_units;
            expected_resources =
                expected_resources.plus_weighted(future.expected_resources, probability)?;
            interruption_probability += probability * future.mission_interruption_probability;
            robot_loss_probability += probability * future.robot_loss_probability;
            maximum_path_resources =
                maximum_path_resources.component_max(future.maximum_path_resources);
            external_gates_passed &= future.external_gates_passed;
            all_paths_within_budget &= future.all_supported_paths_within_budget;
            cached_branch_mismatches =
                cached_branch_mismatches.saturating_add(future.cached_branch_mismatches);
        } else {
            let losses = outcome.terminal_loss_by_hypothesis.as_deref().ok_or(
                RefereeError::InvalidModel("terminal outcome missing loss vector"),
            )?;
            expected_decision_loss += probability * expected_loss(&next_posterior, losses)?;
            if matches!(
                outcome.kind,
                OutcomeKind::MissionInterrupted | OutcomeKind::RobotLost
            ) {
                interruption_probability += probability;
            }
            if outcome.kind == OutcomeKind::RobotLost {
                robot_loss_probability += probability;
            }
        }
    }

    if (total_branch_probability - 1.0).abs() > NUMERICAL_TOLERANCE {
        return Err(RefereeError::PolicyOutcomeMassMismatch {
            action_id: action_id.to_owned(),
            sum_bits: total_branch_probability.to_bits(),
        });
    }

    let expected_resource_cost_units = action_cost_value + expected_future_cost;
    let expected_total_value = expected_decision_loss + expected_resource_cost_units;
    if !expected_decision_loss.is_finite()
        || !expected_resource_cost_units.is_finite()
        || !expected_total_value.is_finite()
        || !interruption_probability.is_finite()
        || !robot_loss_probability.is_finite()
    {
        return Err(RefereeError::NumericalOverflow("policy value or risk"));
    }

    Ok(PartialMetrics {
        expected_decision_loss,
        expected_resource_cost_units,
        expected_resources,
        maximum_path_resources,
        mission_interruption_probability: interruption_probability,
        robot_loss_probability,
        external_gates_passed,
        all_supported_paths_within_budget: all_paths_within_budget,
        cached_branch_mismatches,
    })
}

/// Independently evaluate the *same* candidate policy tree against every model.
///
/// Model-specific priors and likelihoods may differ; hypothesis order, decision
/// losses, action semantics, costs and outcome semantics must match. Branch
/// probabilities/posteriors inside the candidate tree are never trusted.
pub fn evaluate_policy_robustly(
    models: &[ModelScenario],
    policy: &PolicyNode,
    config: RefereeConfig,
) -> Result<RobustPolicyReport, RefereeError> {
    validate_config(config)?;
    if models.is_empty() {
        return Err(RefereeError::EmptyModelSet);
    }
    if models.len() > config.maximum_models {
        return Err(RefereeError::TooManyModels);
    }

    let mut ordered_models = models.iter().collect::<Vec<_>>();
    ordered_models.sort_by(|a, b| a.model_id.cmp(&b.model_id));
    let mut model_ids = BTreeSet::new();
    for model in &ordered_models {
        if model.model_id.trim().is_empty() {
            return Err(RefereeError::EmptyModelId);
        }
        if !model_ids.insert(model.model_id.as_str()) {
            return Err(RefereeError::DuplicateModelId(model.model_id.clone()));
        }
        validate_problem(&model.problem)?;
    }
    validate_model_schema(models)?;

    let mut policy_nodes = 0usize;
    validate_policy(policy, models, 1, &mut policy_nodes, config)?;

    let mut per_model = Vec::with_capacity(models.len());
    for model in ordered_models {
        let posterior = model
            .problem
            .hypotheses
            .iter()
            .map(|hypothesis| hypothesis.prior_probability)
            .collect::<Vec<_>>();
        let metrics = evaluate_node(
            &model.problem,
            policy,
            &posterior,
            config.resource_budget,
            ResourceUse::default(),
            config.resource_budget,
            config.cost_prices,
        )?;
        let within_risk_limits = metrics.mission_interruption_probability
            <= config.risk_limits.maximum_mission_interruption_probability + NUMERICAL_TOLERANCE
            && metrics.robot_loss_probability
                <= config.risk_limits.maximum_robot_loss_probability + NUMERICAL_TOLERANCE;
        per_model.push(ModelPolicyMetrics {
            model_id: model.model_id.clone(),
            expected_decision_loss: metrics.expected_decision_loss,
            expected_resource_cost_units: metrics.expected_resource_cost_units,
            expected_total_value: metrics.expected_decision_loss
                + metrics.expected_resource_cost_units,
            expected_resources: metrics.expected_resources,
            maximum_path_resources: metrics.maximum_path_resources,
            mission_interruption_probability: metrics.mission_interruption_probability,
            robot_loss_probability: metrics.robot_loss_probability,
            external_gates_passed: metrics.external_gates_passed,
            all_supported_paths_within_budget: metrics.all_supported_paths_within_budget,
            within_risk_limits,
            cached_branch_mismatches: metrics.cached_branch_mismatches,
        });
    }

    let values = per_model
        .iter()
        .map(|metric| metric.expected_total_value)
        .collect::<Vec<_>>();
    let worst_case_expected_total_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let best_case_expected_total_value = values.iter().copied().fold(f64::INFINITY, f64::min);
    let total_value_sum = values.iter().sum::<f64>();
    if !total_value_sum.is_finite() {
        return Err(RefereeError::NumericalOverflow("mean model value"));
    }
    let mean_expected_total_value = total_value_sum / values.len() as f64;
    let maximum_mission_interruption_probability = per_model
        .iter()
        .map(|metric| metric.mission_interruption_probability)
        .fold(0.0, f64::max);
    let maximum_robot_loss_probability = per_model
        .iter()
        .map(|metric| metric.robot_loss_probability)
        .fold(0.0, f64::max);
    let all_external_gates_passed = per_model.iter().all(|metric| metric.external_gates_passed);
    let all_supported_paths_within_budget = per_model
        .iter()
        .all(|metric| metric.all_supported_paths_within_budget);
    let all_models_within_risk_limits = per_model.iter().all(|metric| metric.within_risk_limits);

    let mut violations = Vec::new();
    for metric in &per_model {
        if !metric.external_gates_passed {
            violations.push(format!(
                "{}: at least one selected action failed an external gate",
                metric.model_id
            ));
        }
        if !metric.all_supported_paths_within_budget {
            violations.push(format!(
                "{}: at least one supported policy path exceeded a resource budget",
                metric.model_id
            ));
        }
        if !metric.within_risk_limits {
            violations.push(format!(
                "{}: interruption or robot-loss risk exceeded its hard limit",
                metric.model_id
            ));
        }
    }

    let disposition = if all_external_gates_passed
        && all_supported_paths_within_budget
        && all_models_within_risk_limits
    {
        PolicyDisposition::AdmissibleAcrossSuppliedModels
    } else {
        PolicyDisposition::InadmissibleAcrossSuppliedModels
    };

    Ok(RobustPolicyReport {
        model_count: per_model.len(),
        per_model,
        worst_case_expected_total_value,
        best_case_expected_total_value,
        mean_expected_total_value,
        maximum_mission_interruption_probability,
        maximum_robot_loss_probability,
        all_external_gates_passed,
        all_supported_paths_within_budget,
        all_models_within_risk_limits,
        disposition,
        violations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use subterra_survival_reference::{
        Hypothesis, MeasurementOutcome, PolicyBranch, PolicyNode, ResourceCost,
    };

    fn hypotheses() -> Vec<Hypothesis> {
        vec![
            Hypothesis {
                id: "open".into(),
                prior_probability: 0.6,
            },
            Hypothesis {
                id: "blocked".into(),
                prior_probability: 0.3,
            },
            Hypothesis {
                id: "unknown".into(),
                prior_probability: 0.1,
            },
        ]
    }

    fn decisions() -> Vec<MissionDecision> {
        vec![
            MissionDecision {
                id: "enter".into(),
                loss_by_hypothesis: vec![0.0, 100.0, 50.0],
            },
            MissionDecision {
                id: "hold".into(),
                loss_by_hypothesis: vec![40.0, 0.0, 10.0],
            },
        ]
    }

    fn action(robot_loss_probability: f64, gate: bool) -> MeasurementAction {
        MeasurementAction {
            id: "scan".into(),
            modality: "geophysical_scan".into(),
            outcomes: vec![
                MeasurementOutcome {
                    id: "observed".into(),
                    kind: OutcomeKind::Observation,
                    likelihood_by_hypothesis: vec![1.0 - robot_loss_probability; 3],
                    terminal_loss_by_hypothesis: None,
                },
                MeasurementOutcome {
                    id: "robot_lost".into(),
                    kind: OutcomeKind::RobotLost,
                    likelihood_by_hypothesis: vec![robot_loss_probability; 3],
                    terminal_loss_by_hypothesis: Some(vec![100.0; 3]),
                },
            ],
            cost: ResourceCost {
                energy_j: 10.0,
                duration_s: 2.0,
                data_bytes: 100.0,
                exposure_units: 0.1,
            },
            external_gate_passed: gate,
        }
    }

    fn problem(robot_loss_probability: f64, gate: bool) -> PlanningProblem {
        PlanningProblem {
            hypotheses: hypotheses(),
            decisions: decisions(),
            actions: vec![action(robot_loss_probability, gate)],
        }
    }

    fn policy() -> PolicyNode {
        PolicyNode {
            measurement_action_id: Some("scan".into()),
            stop_decision_id: None,
            branches: vec![
                PolicyBranch {
                    outcome_id: "observed".into(),
                    kind: OutcomeKind::Observation,
                    probability: 0.8,
                    posterior: vec![0.6, 0.3, 0.1],
                    continuation: Some(Box::new(PolicyNode {
                        measurement_action_id: None,
                        stop_decision_id: Some("enter".into()),
                        branches: vec![],
                    })),
                },
                PolicyBranch {
                    outcome_id: "robot_lost".into(),
                    kind: OutcomeKind::RobotLost,
                    probability: 0.2,
                    posterior: vec![0.6, 0.3, 0.1],
                    continuation: None,
                },
            ],
        }
    }

    fn config() -> RefereeConfig {
        RefereeConfig {
            resource_budget: ResourceBudget {
                energy_j: 100.0,
                duration_s: 10.0,
                data_bytes: 1_000.0,
                exposure_units: 1.0,
            },
            cost_prices: CostPrices {
                energy_per_j: 2.0,
                time_per_s: 0.0,
                data_per_byte: 0.0,
                exposure_per_unit: 0.0,
            },
            risk_limits: RiskLimits {
                maximum_mission_interruption_probability: 0.1,
                maximum_robot_loss_probability: 0.1,
            },
            maximum_models: 8,
            maximum_policy_nodes: 32,
            maximum_policy_depth: 8,
        }
    }

    #[test]
    fn hand_computable_policy_value_and_risk_match() {
        let report = evaluate_policy_robustly(
            &[ModelScenario {
                model_id: "nominal".into(),
                problem: problem(0.02, true),
            }],
            &policy(),
            config(),
        )
        .unwrap();
        let model = &report.per_model[0];
        // Enter loss under the prior is 0.6*0 + 0.3*100 + 0.1*50 = 35.
        // Robot-loss terminal loss is 100; expected loss = 0.98*35 + 0.02*100 = 36.3.
        assert!((model.expected_decision_loss - 36.3).abs() < 1.0e-8);
        assert!((model.expected_resource_cost_units - 20.0).abs() < 1.0e-8);
        assert!((model.expected_total_value - 56.3).abs() < 1.0e-8);
        assert!((model.robot_loss_probability - 0.02).abs() < 1.0e-8);
        assert!((model.mission_interruption_probability - 0.02).abs() < 1.0e-8);
        assert!(model.all_supported_paths_within_budget);
    }

    #[test]
    fn one_shared_policy_is_scored_under_each_model_not_reoptimized() {
        let models = vec![
            ModelScenario {
                model_id: "optimistic".into(),
                problem: problem(0.02, true),
            },
            ModelScenario {
                model_id: "pessimistic".into(),
                problem: problem(0.15, true),
            },
        ];
        let report = evaluate_policy_robustly(&models, &policy(), config()).unwrap();
        assert_eq!(report.model_count, 2);
        assert!((report.maximum_robot_loss_probability - 0.15).abs() < 1.0e-8);
        assert_eq!(
            report.disposition,
            PolicyDisposition::InadmissibleAcrossSuppliedModels
        );
        assert!(
            report.per_model[0].expected_total_value < report.per_model[1].expected_total_value
        );
        assert_eq!(report.per_model[0].model_id, "optimistic");
        assert_eq!(report.per_model[1].model_id, "pessimistic");
    }

    #[test]
    fn opposing_model_likelihoods_score_the_same_observable_policy_tree() {
        use subterra_survival_reference::{MeasurementAction, MeasurementOutcome};

        let make_problem = |open_likelihoods: [f64; 3]| {
            let blocked_likelihoods = open_likelihoods.map(|value| 1.0 - value);
            PlanningProblem {
                hypotheses: hypotheses(),
                decisions: decisions(),
                actions: vec![MeasurementAction {
                    id: "scan".into(),
                    modality: "geophysical_scan".into(),
                    outcomes: vec![
                        MeasurementOutcome {
                            id: "signal_open".into(),
                            kind: OutcomeKind::Observation,
                            likelihood_by_hypothesis: open_likelihoods.to_vec(),
                            terminal_loss_by_hypothesis: None,
                        },
                        MeasurementOutcome {
                            id: "signal_blocked".into(),
                            kind: OutcomeKind::Observation,
                            likelihood_by_hypothesis: blocked_likelihoods.to_vec(),
                            terminal_loss_by_hypothesis: None,
                        },
                    ],
                    cost: ResourceCost {
                        energy_j: 10.0,
                        duration_s: 2.0,
                        data_bytes: 100.0,
                        exposure_units: 0.1,
                    },
                    external_gate_passed: true,
                }],
            }
        };

        let shared_policy = PolicyNode {
            measurement_action_id: Some("scan".into()),
            stop_decision_id: None,
            branches: vec![
                PolicyBranch {
                    outcome_id: "signal_open".into(),
                    kind: OutcomeKind::Observation,
                    probability: 0.5,
                    posterior: vec![0.6, 0.3, 0.1],
                    continuation: Some(Box::new(PolicyNode {
                        measurement_action_id: None,
                        stop_decision_id: Some("enter".into()),
                        branches: vec![],
                    })),
                },
                PolicyBranch {
                    outcome_id: "signal_blocked".into(),
                    kind: OutcomeKind::Observation,
                    probability: 0.5,
                    posterior: vec![0.6, 0.3, 0.1],
                    continuation: Some(Box::new(PolicyNode {
                        measurement_action_id: None,
                        stop_decision_id: Some("hold".into()),
                        branches: vec![],
                    })),
                },
            ],
        };

        let models = vec![
            ModelScenario {
                model_id: "model-a".into(),
                problem: make_problem([0.9, 0.1, 0.2]),
            },
            ModelScenario {
                model_id: "model-b".into(),
                problem: make_problem([0.1, 0.9, 0.8]),
            },
        ];

        let report = evaluate_policy_robustly(&models, &shared_policy, config()).unwrap();
        assert_eq!(report.model_count, 2);
        assert_eq!(report.per_model[0].model_id, "model-a");
        assert_eq!(report.per_model[1].model_id, "model-b");
        assert!(
            report.per_model[1].expected_decision_loss
                > report.per_model[0].expected_decision_loss + 30.0
        );
        assert!(
            report
                .per_model
                .iter()
                .all(|metric| metric.cached_branch_mismatches > 0)
        );
        assert!(report.all_external_gates_passed);
        assert!(report.all_supported_paths_within_budget);
    }
    #[test]
    fn untrusted_policy_probability_and_posterior_cache_cannot_change_score() {
        let models = vec![
            ModelScenario {
                model_id: "a".into(),
                problem: problem(0.02, true),
            },
            ModelScenario {
                model_id: "b".into(),
                problem: problem(0.05, true),
            },
        ];
        let original = evaluate_policy_robustly(&models, &policy(), config()).unwrap();
        let mut tampered = policy();
        tampered.branches[0].probability = f64::NAN;
        tampered.branches[0].posterior = vec![999.0, -10.0];
        let changed = evaluate_policy_robustly(&models, &tampered, config()).unwrap();
        assert_eq!(
            original.worst_case_expected_total_value,
            changed.worst_case_expected_total_value
        );
        assert_eq!(
            original.maximum_robot_loss_probability,
            changed.maximum_robot_loss_probability
        );
        assert!(
            changed
                .per_model
                .iter()
                .all(|metric| metric.cached_branch_mismatches > 0)
        );
    }

    #[test]
    fn external_gate_denial_in_any_model_blocks_admissibility() {
        let models = vec![
            ModelScenario {
                model_id: "admitted".into(),
                problem: problem(0.02, true),
            },
            ModelScenario {
                model_id: "denied".into(),
                problem: problem(0.02, false),
            },
        ];
        let report = evaluate_policy_robustly(&models, &policy(), config()).unwrap();
        assert!(!report.all_external_gates_passed);
        assert_eq!(
            report.disposition,
            PolicyDisposition::InadmissibleAcrossSuppliedModels
        );
    }

    #[test]
    fn missing_possible_policy_outcome_is_rejected() {
        let mut incomplete = policy();
        incomplete.branches.pop();
        let result = evaluate_policy_robustly(
            &[ModelScenario {
                model_id: "only".into(),
                problem: problem(0.02, true),
            }],
            &incomplete,
            config(),
        );
        assert!(matches!(
            result,
            Err(RefereeError::MissingPolicyOutcome { .. })
        ));
    }

    #[test]
    fn model_cost_schema_mismatch_is_rejected() {
        let mut changed = problem(0.02, true);
        changed.actions[0].cost.energy_j += 1.0;
        let result = evaluate_policy_robustly(
            &[
                ModelScenario {
                    model_id: "a".into(),
                    problem: problem(0.02, true),
                },
                ModelScenario {
                    model_id: "b".into(),
                    problem: changed,
                },
            ],
            &policy(),
            config(),
        );
        assert!(matches!(result, Err(RefereeError::ModelSchemaMismatch(_))));
    }

    fn deterministic_observation_action(id: &str, energy_j: f64) -> MeasurementAction {
        MeasurementAction {
            id: id.into(),
            modality: "survey".into(),
            outcomes: vec![MeasurementOutcome {
                id: "continue".into(),
                kind: OutcomeKind::Observation,
                likelihood_by_hypothesis: vec![1.0; 3],
                terminal_loss_by_hypothesis: None,
            }],
            cost: ResourceCost {
                energy_j,
                duration_s: 0.0,
                data_bytes: 0.0,
                exposure_units: 0.0,
            },
            external_gate_passed: true,
        }
    }

    #[test]
    fn pathwise_budget_violation_is_not_hidden_by_expected_resource_cost() {
        let mut p = problem(0.0, true);
        p.actions = vec![
            deterministic_observation_action("first", 6.0),
            deterministic_observation_action("second", 6.0),
        ];
        let tree = PolicyNode {
            measurement_action_id: Some("first".into()),
            stop_decision_id: None,
            branches: vec![PolicyBranch {
                outcome_id: "continue".into(),
                kind: OutcomeKind::Observation,
                probability: 1.0,
                posterior: vec![0.6, 0.3, 0.1],
                continuation: Some(Box::new(PolicyNode {
                    measurement_action_id: Some("second".into()),
                    stop_decision_id: None,
                    branches: vec![PolicyBranch {
                        outcome_id: "continue".into(),
                        kind: OutcomeKind::Observation,
                        probability: 1.0,
                        posterior: vec![0.6, 0.3, 0.1],
                        continuation: Some(Box::new(PolicyNode {
                            measurement_action_id: None,
                            stop_decision_id: Some("enter".into()),
                            branches: vec![],
                        })),
                    }],
                })),
            }],
        };
        let mut cfg = config();
        cfg.resource_budget.energy_j = 10.0;
        let report = evaluate_policy_robustly(
            &[ModelScenario {
                model_id: "path".into(),
                problem: p,
            }],
            &tree,
            cfg,
        )
        .unwrap();
        assert!(!report.all_supported_paths_within_budget);
        assert!((report.per_model[0].maximum_path_resources.energy_j - 12.0).abs() < 1.0e-8);
        assert_eq!(
            report.disposition,
            PolicyDisposition::InadmissibleAcrossSuppliedModels
        );
    }

    #[test]
    fn policy_is_deterministic_under_model_and_branch_order_permutation() {
        let mut models = vec![
            ModelScenario {
                model_id: "z".into(),
                problem: problem(0.05, true),
            },
            ModelScenario {
                model_id: "a".into(),
                problem: problem(0.02, true),
            },
        ];
        let left = evaluate_policy_robustly(&models, &policy(), config()).unwrap();
        models.reverse();
        let mut reversed = policy();
        reversed.branches.reverse();
        let right = evaluate_policy_robustly(&models, &reversed, config()).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn outcome_possible_in_one_model_may_have_zero_mass_in_another() {
        let mut zero = problem(0.0, true);
        zero.actions[0].outcomes[1].likelihood_by_hypothesis = vec![0.0; 3];
        zero.actions[0].outcomes[0].likelihood_by_hypothesis = vec![1.0; 3];
        let positive = problem(0.02, true);
        // The shared policy contains both branches because robot loss is possible
        // in at least one model. The zero-mass model must skip that branch.
        let report = evaluate_policy_robustly(
            &[
                ModelScenario {
                    model_id: "zero".into(),
                    problem: zero,
                },
                ModelScenario {
                    model_id: "positive".into(),
                    problem: positive,
                },
            ],
            &policy(),
            config(),
        )
        .unwrap();
        assert!((report.per_model[0].robot_loss_probability).abs() < 1.0e-12);
        assert!((report.per_model[1].robot_loss_probability - 0.02).abs() < 1.0e-8);
    }

    #[test]
    fn robot_loss_limit_fails_even_when_mean_model_risk_is_lower() {
        let models = vec![
            ModelScenario {
                model_id: "low".into(),
                problem: problem(0.01, true),
            },
            ModelScenario {
                model_id: "high".into(),
                problem: problem(0.12, true),
            },
        ];
        let mut cfg = config();
        cfg.risk_limits.maximum_robot_loss_probability = 0.1;
        let report = evaluate_policy_robustly(&models, &policy(), cfg).unwrap();
        let mean = report
            .per_model
            .iter()
            .map(|metric| metric.robot_loss_probability)
            .sum::<f64>()
            / 2.0;
        assert!(mean < 0.1);
        assert!(report.maximum_robot_loss_probability > 0.1);
        assert!(!report.all_models_within_risk_limits);
    }
}
