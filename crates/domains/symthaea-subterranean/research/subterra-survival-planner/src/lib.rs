//! Bounded survival-conditioned scientific measurement planning.
//!
//! A finite-horizon extension of the single-step EVSI reference kernel.
//! This planner models observation branches and absorbing mission outcomes so
//! the expected value of future measurements disappears when a robot must
//! retreat, a mission is interrupted, or a robot is lost.
//!
//! It is an offline research planner, not a robot controller, route planner,
//! sensor model, safety case, or authorization system. Candidate actions must
//! be pre-admitted by external reviewed gates. Probabilities, decision losses,
//! terminal losses, and resource costs are supplied by the caller and require
//! independent calibration / review.

use std::collections::BTreeSet;

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const COMPARISON_TOLERANCE: f64 = 1.0e-10;
const UNKNOWN_HYPOTHESIS_ID: &str = "unknown";
const MAX_HORIZON_STEPS: usize = 16;
const MAX_ACTIONS: usize = 32;
const MAX_OUTCOMES_PER_ACTION: usize = 16;

#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    pub id: String,
    pub prior_probability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionDecision {
    pub id: String,
    /// Terminal decision loss under each hypothesis, in fixed hypothesis order.
    pub loss_by_hypothesis: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeKind {
    /// An observation outcome is admitted by the caller; evidence / receipt verification is external.
    Observation,
    /// The robot safely retreats; no further observation in this mission branch.
    SafeRetreat,
    /// The planned scientific mission is interrupted.
    MissionInterrupted,
    /// The robot is lost; no future observations or data return are assumed.
    RobotLost,
    /// Communications are lost. Use this only when an external record verifies
    /// local evidence retention for store-and-forward; otherwise encode the data
    /// loss in a different terminal outcome and its supplied loss vector.
    RelayLostStoreAndForward,
}

impl OutcomeKind {
    fn is_mission_interruption(self) -> bool {
        matches!(self, Self::MissionInterrupted | Self::RobotLost)
    }

    fn is_robot_loss(self) -> bool {
        self == Self::RobotLost
    }

    fn is_terminal(self) -> bool {
        self != Self::Observation
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementOutcome {
    pub id: String,
    pub kind: OutcomeKind,
    /// P(outcome | hypothesis, action), in the same order as the problem's hypotheses.
    pub likelihood_by_hypothesis: Vec<f64>,
    /// Required for terminal outcomes, forbidden for continuing observations.
    /// It should account for truncated science value and any declared terminal
    /// mission consequences under each hypothesis.
    pub terminal_loss_by_hypothesis: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ResourceCost {
    pub energy_j: f64,
    pub duration_s: f64,
    pub data_bytes: f64,
    pub exposure_units: f64,
}

impl ResourceCost {
    fn validate(self) -> Result<(), PlanningError> {
        for (value, field) in [
            (self.energy_j, "energy_j"),
            (self.duration_s, "duration_s"),
            (self.data_bytes, "data_bytes"),
            (self.exposure_units, "exposure_units"),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(PlanningError::InvalidResourceValue(field));
            }
        }
        Ok(())
    }

    fn fits(self, budget: ResourceBudget) -> bool {
        self.energy_j <= budget.energy_j
            && self.duration_s <= budget.duration_s
            && self.data_bytes <= budget.data_bytes
            && self.exposure_units <= budget.exposure_units
    }

    fn subtract_from(self, budget: ResourceBudget) -> ResourceBudget {
        ResourceBudget {
            energy_j: (budget.energy_j - self.energy_j).max(0.0),
            duration_s: (budget.duration_s - self.duration_s).max(0.0),
            data_bytes: (budget.data_bytes - self.data_bytes).max(0.0),
            exposure_units: (budget.exposure_units - self.exposure_units).max(0.0),
        }
    }

    fn weighted(self, prices: CostPrices) -> Result<f64, PlanningError> {
        let value = self.energy_j * prices.energy_per_j
            + self.duration_s * prices.time_per_s
            + self.data_bytes * prices.data_per_byte
            + self.exposure_units * prices.exposure_per_unit;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(PlanningError::NumericalOverflow("weighted resource cost"))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ResourceBudget {
    pub energy_j: f64,
    pub duration_s: f64,
    pub data_bytes: f64,
    pub exposure_units: f64,
}

impl ResourceBudget {
    fn validate(self) -> Result<(), PlanningError> {
        ResourceCost {
            energy_j: self.energy_j,
            duration_s: self.duration_s,
            data_bytes: self.data_bytes,
            exposure_units: self.exposure_units,
        }
        .validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CostPrices {
    pub energy_per_j: f64,
    pub time_per_s: f64,
    pub data_per_byte: f64,
    pub exposure_per_unit: f64,
}

impl CostPrices {
    fn validate(self) -> Result<(), PlanningError> {
        for (value, field) in [
            (self.energy_per_j, "energy_per_j"),
            (self.time_per_s, "time_per_s"),
            (self.data_per_byte, "data_per_byte"),
            (self.exposure_per_unit, "exposure_per_unit"),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(PlanningError::InvalidCostPrice(field));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementAction {
    pub id: String,
    pub modality: String,
    pub outcomes: Vec<MeasurementOutcome>,
    pub cost: ResourceCost,
    /// Externally supplied result of a reviewed admissibility/ODD/preservation gate.
    /// This planner never derives or widens this permission.
    pub external_gate_passed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanningProblem {
    pub hypotheses: Vec<Hypothesis>,
    pub decisions: Vec<MissionDecision>,
    pub actions: Vec<MeasurementAction>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiskLimits {
    pub maximum_mission_interruption_probability: f64,
    pub maximum_robot_loss_probability: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlannerConfig {
    /// Maximum number of measurement actions along any observation branch.
    pub horizon_steps: usize,
    /// Maximum recursive belief states expanded across this planning run.
    pub maximum_search_nodes: usize,
    pub maximum_actions: usize,
    pub maximum_outcomes_per_action: usize,
    pub risk_limits: RiskLimits,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanningError {
    EmptyHypotheses,
    MissingUnknownHypothesis,
    MultipleUnknownHypotheses,
    UnknownHypothesisHasZeroPrior,
    DuplicateHypothesisId(String),
    DuplicateDecisionId(String),
    DuplicateActionId(String),
    DuplicateOutcomeId { action_id: String, outcome_id: String },
    EmptyIdentifier(&'static str),
    InvalidProbability(String),
    PriorNotNormalized(f64),
    NoMissionDecisions,
    InvalidDecisionLoss(String),
    DecisionDimensionMismatch(String),
    NoObservationOutcomes(String),
    InvalidLikelihood { action_id: String, outcome_id: String },
    LikelihoodDimensionMismatch { action_id: String, outcome_id: String },
    LikelihoodsNotNormalized { action_id: String, hypothesis_id: String, sum: f64 },
    UnexpectedTerminalLoss(String),
    MissingTerminalLoss(String),
    InvalidTerminalLoss(String),
    InvalidResourceValue(&'static str),
    InvalidCostPrice(&'static str),
    InvalidRiskLimit(&'static str),
    InvalidPlannerLimit(&'static str),
    NumericalOverflow(&'static str),
    SearchBudgetExceeded { maximum: usize },
}

impl std::fmt::Display for PlanningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PlanningError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionStatus {
    Selected,
    FeasibleNotSelected,
    ExternalGateDenied,
    OverBudget,
    RiskLimitExceeded,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionAssessment {
    pub action_id: String,
    pub status: ActionStatus,
    pub expected_decision_loss: Option<f64>,
    pub expected_resource_cost_units: Option<f64>,
    pub expected_total_value: Option<f64>,
    pub mission_interruption_probability: Option<f64>,
    pub robot_loss_probability: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyBranch {
    pub outcome_id: String,
    pub kind: OutcomeKind,
    /// Conditional probability at this policy node.
    pub probability: f64,
    /// Posterior probability for each hypothesis, in declared order.
    pub posterior: Vec<f64>,
    /// Present only for an Observation outcome. Terminal branches have none.
    pub continuation: Option<Box<PolicyNode>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyNode {
    /// Exactly one of measurement_action_id / stop_decision_id is present.
    pub measurement_action_id: Option<String>,
    pub stop_decision_id: Option<String>,
    pub branches: Vec<PolicyBranch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ExpectedResources {
    pub energy_j: f64,
    pub duration_s: f64,
    pub data_bytes: f64,
    pub exposure_units: f64,
}

impl ExpectedResources {
    fn add_action_and_future(
        action: ResourceCost,
        future: Self,
    ) -> Result<Self, PlanningError> {
        let result = Self {
            energy_j: action.energy_j + future.energy_j,
            duration_s: action.duration_s + future.duration_s,
            data_bytes: action.data_bytes + future.data_bytes,
            exposure_units: action.exposure_units + future.exposure_units,
        };
        ResourceCost {
            energy_j: result.energy_j,
            duration_s: result.duration_s,
            data_bytes: result.data_bytes,
            exposure_units: result.exposure_units,
        }
        .validate()?;
        Ok(result)
    }

    fn plus_weighted(
        current: Self,
        future: Self,
        probability: f64,
    ) -> Result<Self, PlanningError> {
        let result = Self {
            energy_j: current.energy_j + future.energy_j * probability,
            duration_s: current.duration_s + future.duration_s * probability,
            data_bytes: current.data_bytes + future.data_bytes * probability,
            exposure_units: current.exposure_units + future.exposure_units * probability,
        };
        ResourceCost {
            energy_j: result.energy_j,
            duration_s: result.duration_s,
            data_bytes: result.data_bytes,
            exposure_units: result.exposure_units,
        }
        .validate()?;
        Ok(result)
    }

}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanMetrics {
    pub expected_decision_loss: f64,
    pub expected_resource_cost_units: f64,
    pub expected_total_value: f64,
    pub expected_resources: ExpectedResources,
    pub mission_interruption_probability: f64,
    pub robot_loss_probability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanningReport {
    pub policy: PolicyNode,
    pub metrics: PlanMetrics,
    pub root_action_assessments: Vec<ActionAssessment>,
    pub expanded_search_nodes: usize,
}

#[derive(Debug, Clone)]
struct Evaluation {
    policy: PolicyNode,
    expected_decision_loss: f64,
    expected_resource_cost_units: f64,
    expected_resources: ExpectedResources,
    mission_interruption_probability: f64,
    robot_loss_probability: f64,
}

impl Evaluation {
    fn total_value(&self) -> f64 {
        self.expected_decision_loss + self.expected_resource_cost_units
    }

    fn metrics(&self) -> PlanMetrics {
        PlanMetrics {
            expected_decision_loss: self.expected_decision_loss,
            expected_resource_cost_units: self.expected_resource_cost_units,
            expected_total_value: self.total_value(),
            expected_resources: self.expected_resources,
            mission_interruption_probability: self.mission_interruption_probability,
            robot_loss_probability: self.robot_loss_probability,
        }
    }
}

struct SearchCounter {
    expanded: usize,
    maximum: usize,
}

impl SearchCounter {
    fn visit(&mut self) -> Result<(), PlanningError> {
        self.expanded = self.expanded.saturating_add(1);
        if self.expanded > self.maximum {
            Err(PlanningError::SearchBudgetExceeded {
                maximum: self.maximum,
            })
        } else {
            Ok(())
        }
    }
}

fn require_identifier(value: &str, field: &'static str) -> Result<(), PlanningError> {
    if value.trim().is_empty() {
        Err(PlanningError::EmptyIdentifier(field))
    } else {
        Ok(())
    }
}

fn normalize_posterior(
    belief: &[f64],
    likelihood: &[f64],
    outcome_probability: f64,
) -> Result<Vec<f64>, PlanningError> {
    let posterior = belief
        .iter()
        .zip(likelihood)
        .map(|(prior, conditional)| prior * conditional / outcome_probability)
        .collect::<Vec<_>>();
    let sum = posterior.iter().sum::<f64>();
    if !sum.is_finite() || (sum - 1.0).abs() > 1.0e-8 {
        return Err(PlanningError::NumericalOverflow("posterior normalization"));
    }
    Ok(posterior)
}

fn outcome_probability(belief: &[f64], likelihood: &[f64]) -> Result<f64, PlanningError> {
    let probability = belief
        .iter()
        .zip(likelihood)
        .map(|(prior, conditional)| prior * conditional)
        .sum::<f64>();
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(PlanningError::NumericalOverflow("outcome probability"));
    }
    Ok(probability)
}

fn expected_loss(
    belief: &[f64],
    losses: &[f64],
) -> Result<f64, PlanningError> {
    let value = belief
        .iter()
        .zip(losses)
        .map(|(probability, loss)| probability * loss)
        .sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PlanningError::NumericalOverflow("expected decision loss"))
    }
}

fn best_stop_decision(
    belief: &[f64],
    decisions: &[MissionDecision],
) -> Result<Evaluation, PlanningError> {
    let mut ordered = decisions.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.id.cmp(&b.id));

    let mut best_id: Option<&str> = None;
    let mut best_loss = f64::INFINITY;
    for decision in ordered {
        let value = expected_loss(belief, &decision.loss_by_hypothesis)?;
        if value < best_loss - COMPARISON_TOLERANCE {
            best_loss = value;
            best_id = Some(&decision.id);
        }
    }

    let Some(decision_id) = best_id else {
        return Err(PlanningError::NoMissionDecisions);
    };

    Ok(Evaluation {
        policy: PolicyNode {
            measurement_action_id: None,
            stop_decision_id: Some(decision_id.to_string()),
            branches: Vec::new(),
        },
        expected_decision_loss: best_loss,
        expected_resource_cost_units: 0.0,
        expected_resources: ExpectedResources::default(),
        mission_interruption_probability: 0.0,
        robot_loss_probability: 0.0,
    })
}

fn validate_problem(
    problem: &PlanningProblem,
    config: PlannerConfig,
) -> Result<(), PlanningError> {
    if problem.hypotheses.is_empty() {
        return Err(PlanningError::EmptyHypotheses);
    }
    if problem.decisions.is_empty() {
        return Err(PlanningError::NoMissionDecisions);
    }
    if config.horizon_steps == 0
        || config.horizon_steps > MAX_HORIZON_STEPS
        || config.maximum_search_nodes == 0
        || config.maximum_actions == 0
        || config.maximum_actions > MAX_ACTIONS
        || config.maximum_outcomes_per_action == 0
        || config.maximum_outcomes_per_action > MAX_OUTCOMES_PER_ACTION
    {
        return Err(PlanningError::InvalidPlannerLimit(
            "planner limits are outside the supported bounded envelope",
        ));
    }
    if problem.actions.len() > config.maximum_actions {
        return Err(PlanningError::InvalidPlannerLimit("maximum action count exceeded"));
    }
    if !config.risk_limits.maximum_mission_interruption_probability.is_finite()
        || !(0.0..=1.0).contains(
            &config.risk_limits.maximum_mission_interruption_probability,
        )
    {
        return Err(PlanningError::InvalidRiskLimit("mission interruption"));
    }
    if !config.risk_limits.maximum_robot_loss_probability.is_finite()
        || !(0.0..=1.0).contains(&config.risk_limits.maximum_robot_loss_probability)
    {
        return Err(PlanningError::InvalidRiskLimit("robot loss"));
    }

    let mut hypothesis_ids = BTreeSet::new();
    let mut prior_sum = 0.0;
    let mut unknown_count = 0usize;
    let mut unknown_prior = 0.0;

    for hypothesis in &problem.hypotheses {
        require_identifier(&hypothesis.id, "hypothesis.id")?;
        if !hypothesis_ids.insert(hypothesis.id.as_str()) {
            return Err(PlanningError::DuplicateHypothesisId(hypothesis.id.clone()));
        }
        if !hypothesis.prior_probability.is_finite()
            || !(0.0..=1.0).contains(&hypothesis.prior_probability)
        {
            return Err(PlanningError::InvalidProbability(hypothesis.id.clone()));
        }
        prior_sum += hypothesis.prior_probability;
        if !prior_sum.is_finite() {
            return Err(PlanningError::NumericalOverflow("prior probability sum"));
        }
        if hypothesis.id == UNKNOWN_HYPOTHESIS_ID {
            unknown_count += 1;
            unknown_prior = hypothesis.prior_probability;
        }
    }

    if unknown_count == 0 {
        return Err(PlanningError::MissingUnknownHypothesis);
    }
    if unknown_count > 1 {
        return Err(PlanningError::MultipleUnknownHypotheses);
    }
    if unknown_prior <= 0.0 {
        return Err(PlanningError::UnknownHypothesisHasZeroPrior);
    }
    if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(PlanningError::PriorNotNormalized(prior_sum));
    }

    let hypothesis_count = problem.hypotheses.len();
    let mut decision_ids = BTreeSet::new();
    for decision in &problem.decisions {
        require_identifier(&decision.id, "decision.id")?;
        if !decision_ids.insert(decision.id.as_str()) {
            return Err(PlanningError::DuplicateDecisionId(decision.id.clone()));
        }
        if decision.loss_by_hypothesis.len() != hypothesis_count {
            return Err(PlanningError::DecisionDimensionMismatch(decision.id.clone()));
        }
        if decision
            .loss_by_hypothesis
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(PlanningError::InvalidDecisionLoss(decision.id.clone()));
        }
    }

    let mut action_ids = BTreeSet::new();
    for action in &problem.actions {
        require_identifier(&action.id, "action.id")?;
        require_identifier(&action.modality, "action.modality")?;
        if !action_ids.insert(action.id.as_str()) {
            return Err(PlanningError::DuplicateActionId(action.id.clone()));
        }
        action.cost.validate()?;
        if action.outcomes.is_empty() {
            return Err(PlanningError::NoObservationOutcomes(action.id.clone()));
        }
        if action.outcomes.len() > config.maximum_outcomes_per_action {
            return Err(PlanningError::InvalidPlannerLimit("maximum outcome count exceeded"));
        }

        let mut outcome_ids = BTreeSet::new();
        for outcome in &action.outcomes {
            require_identifier(&outcome.id, "outcome.id")?;
            if !outcome_ids.insert(outcome.id.as_str()) {
                return Err(PlanningError::DuplicateOutcomeId {
                    action_id: action.id.clone(),
                    outcome_id: outcome.id.clone(),
                });
            }
            if outcome.likelihood_by_hypothesis.len() != hypothesis_count {
                return Err(PlanningError::LikelihoodDimensionMismatch {
                    action_id: action.id.clone(),
                    outcome_id: outcome.id.clone(),
                });
            }
            if outcome
                .likelihood_by_hypothesis
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            {
                return Err(PlanningError::InvalidLikelihood {
                    action_id: action.id.clone(),
                    outcome_id: outcome.id.clone(),
                });
            }

            match (outcome.kind, outcome.terminal_loss_by_hypothesis.as_ref()) {
                (OutcomeKind::Observation, None) => {}
                (OutcomeKind::Observation, Some(_)) => {
                    return Err(PlanningError::UnexpectedTerminalLoss(outcome.id.clone()));
                }
                (kind, None) if kind.is_terminal() => {
                    return Err(PlanningError::MissingTerminalLoss(outcome.id.clone()));
                }
                (_, Some(losses)) if losses.len() != hypothesis_count => {
                    return Err(PlanningError::DecisionDimensionMismatch(outcome.id.clone()));
                }
                (_, Some(losses))
                    if losses.iter().any(|value| !value.is_finite() || *value < 0.0) =>
                {
                    return Err(PlanningError::InvalidTerminalLoss(outcome.id.clone()));
                }
                _ => {}
            }
        }

        for (hypothesis_index, hypothesis) in problem.hypotheses.iter().enumerate() {
            let sum = action
                .outcomes
                .iter()
                .map(|outcome| outcome.likelihood_by_hypothesis[hypothesis_index])
                .sum::<f64>();
            if !sum.is_finite() {
                return Err(PlanningError::NumericalOverflow("outcome likelihood sum"));
            }
            if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(PlanningError::LikelihoodsNotNormalized {
                    action_id: action.id.clone(),
                    hypothesis_id: hypothesis.id.clone(),
                    sum,
                });
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_action(
    action: &MeasurementAction,
    belief: &[f64],
    remaining_budget: ResourceBudget,
    remaining_steps: usize,
    problem: &PlanningProblem,
    prices: CostPrices,
    config: PlannerConfig,
    counter: &mut SearchCounter,
) -> Result<Evaluation, PlanningError> {
    let budget_after_action = action.cost.subtract_from(remaining_budget);
    let action_cost_units = action.cost.weighted(prices)?;
    let mut expected_loss = 0.0;
    let mut expected_future_cost_units = 0.0;
    let mut expected_resources = ExpectedResources::default();
    let mut interruption_probability = 0.0;
    let mut robot_loss_probability = 0.0;
    let mut branches = Vec::new();

    for outcome in &action.outcomes {
        let branch_probability =
            outcome_probability(belief, &outcome.likelihood_by_hypothesis)?;
        if branch_probability <= 0.0 {
            continue;
        }
        let posterior = normalize_posterior(
            belief,
            &outcome.likelihood_by_hypothesis,
            branch_probability,
        )?;

        let (branch_loss, branch_cost, branch_resources, branch_interruption, branch_robot_loss, continuation) =
            if outcome.kind == OutcomeKind::Observation {
                let child = solve_state(
                    &posterior,
                    budget_after_action,
                    remaining_steps - 1,
                    problem,
                    prices,
                    config,
                    counter,
                )?;
                (
                    child.expected_decision_loss,
                    child.expected_resource_cost_units,
                    child.expected_resources,
                    child.mission_interruption_probability,
                    child.robot_loss_probability,
                    Some(Box::new(child.policy)),
                )
            } else {
                let terminal_losses = outcome
                    .terminal_loss_by_hypothesis
                    .as_ref()
                    .ok_or_else(|| PlanningError::MissingTerminalLoss(outcome.id.clone()))?;
                (
                    expected_loss(&posterior, terminal_losses)?,
                    0.0,
                    ExpectedResources::default(),
                    if outcome.kind.is_mission_interruption() { 1.0 } else { 0.0 },
                    if outcome.kind.is_robot_loss() { 1.0 } else { 0.0 },
                    None,
                )
            };

        expected_loss += branch_probability * branch_loss;
        expected_future_cost_units += branch_probability * branch_cost;
        expected_resources =
            ExpectedResources::plus_weighted(expected_resources, branch_resources, branch_probability)?;
        interruption_probability += branch_probability * branch_interruption;
        robot_loss_probability += branch_probability * branch_robot_loss;

        branches.push(PolicyBranch {
            outcome_id: outcome.id.clone(),
            kind: outcome.kind,
            probability: branch_probability,
            posterior,
            continuation,
        });
    }

    let expected_resource_cost_units = action_cost_units + expected_future_cost_units;
    let expected_resources =
        ExpectedResources::add_action_and_future(action.cost, expected_resources)?;
    let expected_total_value = expected_loss + expected_resource_cost_units;

    if !expected_loss.is_finite()
        || !expected_resource_cost_units.is_finite()
        || !expected_total_value.is_finite()
        || !interruption_probability.is_finite()
        || !robot_loss_probability.is_finite()
    {
        return Err(PlanningError::NumericalOverflow("policy value or risk"));
    }

    branches.sort_by(|a, b| a.outcome_id.cmp(&b.outcome_id));

    Ok(Evaluation {
        policy: PolicyNode {
            measurement_action_id: Some(action.id.clone()),
            stop_decision_id: None,
            branches,
        },
        expected_decision_loss: expected_loss,
        expected_resource_cost_units,
        expected_resources,
        mission_interruption_probability: interruption_probability,
        robot_loss_probability,
    })
}

#[allow(clippy::too_many_arguments)]
fn solve_state(
    belief: &[f64],
    budget: ResourceBudget,
    remaining_steps: usize,
    problem: &PlanningProblem,
    prices: CostPrices,
    config: PlannerConfig,
    counter: &mut SearchCounter,
) -> Result<Evaluation, PlanningError> {
    counter.visit()?;
    let mut best = best_stop_decision(belief, &problem.decisions)?;
    if remaining_steps == 0 {
        return Ok(best);
    }

    let mut actions = problem.actions.iter().collect::<Vec<_>>();
    actions.sort_by(|a, b| a.id.cmp(&b.id));

    for action in actions {
        if !action.external_gate_passed || !action.cost.fits(budget) {
            continue;
        }
        let candidate = evaluate_action(
            action,
            belief,
            budget,
            remaining_steps,
            problem,
            prices,
            config,
            counter,
        )?;
        if candidate.mission_interruption_probability
            > config.risk_limits.maximum_mission_interruption_probability
            || candidate.robot_loss_probability
                > config.risk_limits.maximum_robot_loss_probability
        {
            continue;
        }
        if candidate.total_value() < best.total_value() - COMPARISON_TOLERANCE {
            best = candidate;
        }
    }

    Ok(best)
}

/// Plan a bounded, contingent measurement policy.
///
/// The optimizer may stop and choose a terminal mission decision at any node.
/// A measurement action is considered only if the external gate passed and
/// its immediate resources fit the current path budget. Each continuing
/// observation branch is planned separately with its updated posterior and
/// remaining resources. Terminal branches receive no continuation value:
/// future measurements cannot be counted after retreat, mission interruption,
/// robot loss, or the declared relay-loss/store-forward terminal event.
pub fn plan_survival_conditioned(
    problem: &PlanningProblem,
    budget: ResourceBudget,
    prices: CostPrices,
    config: PlannerConfig,
) -> Result<PlanningReport, PlanningError> {
    budget.validate()?;
    prices.validate()?;
    validate_problem(problem, config)?;

    let belief = problem
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .collect::<Vec<_>>();

    let mut counter = SearchCounter {
        expanded: 0,
        maximum: config.maximum_search_nodes,
    };
    counter.visit()?;
    let mut root = best_stop_decision(&belief, &problem.decisions)?;
    let mut root_action_assessments = Vec::new();

    let mut actions = problem.actions.iter().collect::<Vec<_>>();
    actions.sort_by(|a, b| a.id.cmp(&b.id));

    for action in actions {
        if !action.external_gate_passed {
            root_action_assessments.push(ActionAssessment {
                action_id: action.id.clone(),
                status: ActionStatus::ExternalGateDenied,
                expected_decision_loss: None,
                expected_resource_cost_units: None,
                expected_total_value: None,
                mission_interruption_probability: None,
                robot_loss_probability: None,
            });
            continue;
        }
        if !action.cost.fits(budget) {
            root_action_assessments.push(ActionAssessment {
                action_id: action.id.clone(),
                status: ActionStatus::OverBudget,
                expected_decision_loss: None,
                expected_resource_cost_units: None,
                expected_total_value: None,
                mission_interruption_probability: None,
                robot_loss_probability: None,
            });
            continue;
        }

        let candidate = evaluate_action(
            action,
            &belief,
            budget,
            config.horizon_steps,
            problem,
            prices,
            config,
            &mut counter,
        )?;
        let exceeds_risk = candidate.mission_interruption_probability
            > config.risk_limits.maximum_mission_interruption_probability
            || candidate.robot_loss_probability
                > config.risk_limits.maximum_robot_loss_probability;

        root_action_assessments.push(ActionAssessment {
            action_id: action.id.clone(),
            status: if exceeds_risk {
                ActionStatus::RiskLimitExceeded
            } else {
                ActionStatus::FeasibleNotSelected
            },
            expected_decision_loss: Some(candidate.expected_decision_loss),
            expected_resource_cost_units: Some(candidate.expected_resource_cost_units),
            expected_total_value: Some(candidate.total_value()),
            mission_interruption_probability: Some(candidate.mission_interruption_probability),
            robot_loss_probability: Some(candidate.robot_loss_probability),
        });

        if !exceeds_risk
            && candidate.total_value() < root.total_value() - COMPARISON_TOLERANCE
        {
            root = candidate;
        }
    }

    if let Some(selected_id) = root.policy.measurement_action_id.as_deref() {
        if let Some(assessment) = root_action_assessments
            .iter_mut()
            .find(|assessment| assessment.action_id == selected_id)
        {
            assessment.status = ActionStatus::Selected;
        }
    }

    Ok(PlanningReport {
        policy: root.policy.clone(),
        metrics: root.metrics(),
        root_action_assessments,
        expanded_search_nodes: counter.expanded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hypotheses() -> Vec<Hypothesis> {
        vec![
            Hypothesis {
                id: "open_continuation".into(),
                prior_probability: 0.45,
            },
            Hypothesis {
                id: "blocked_dead_end".into(),
                prior_probability: 0.45,
            },
            Hypothesis {
                id: UNKNOWN_HYPOTHESIS_ID.into(),
                prior_probability: 0.10,
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
                loss_by_hypothesis: vec![30.0, 0.0, 10.0],
            },
        ]
    }

    fn scan(id: &str) -> MeasurementAction {
        MeasurementAction {
            id: id.into(),
            modality: "geophysical_scan".into(),
            outcomes: vec![
                MeasurementOutcome {
                    id: "continuation_signal".into(),
                    kind: OutcomeKind::Observation,
                    likelihood_by_hypothesis: vec![0.891, 0.099, 0.495],
                    terminal_loss_by_hypothesis: None,
                },
                MeasurementOutcome {
                    id: "no_continuation_signal".into(),
                    kind: OutcomeKind::Observation,
                    likelihood_by_hypothesis: vec![0.099, 0.891, 0.495],
                    terminal_loss_by_hypothesis: None,
                },
                MeasurementOutcome {
                    id: "robot_lost".into(),
                    kind: OutcomeKind::RobotLost,
                    likelihood_by_hypothesis: vec![0.01, 0.01, 0.01],
                    terminal_loss_by_hypothesis: Some(vec![50.0, 50.0, 50.0]),
                },
            ],
            cost: ResourceCost {
                energy_j: 100.0,
                duration_s: 5.0,
                data_bytes: 1_000.0,
                exposure_units: 0.1,
            },
            external_gate_passed: true,
        }
    }

    fn problem(actions: Vec<MeasurementAction>) -> PlanningProblem {
        PlanningProblem {
            hypotheses: hypotheses(),
            decisions: decisions(),
            actions,
        }
    }

    fn budget() -> ResourceBudget {
        ResourceBudget {
            energy_j: 1_000.0,
            duration_s: 60.0,
            data_bytes: 10_000.0,
            exposure_units: 1.0,
        }
    }

    fn prices() -> CostPrices {
        CostPrices {
            energy_per_j: 0.0,
            time_per_s: 0.0,
            data_per_byte: 0.0,
            exposure_per_unit: 0.0,
        }
    }

    fn config(horizon_steps: usize) -> PlannerConfig {
        PlannerConfig {
            horizon_steps,
            maximum_search_nodes: 10_000,
            maximum_actions: 8,
            maximum_outcomes_per_action: 8,
            risk_limits: RiskLimits {
                maximum_mission_interruption_probability: 0.10,
                maximum_robot_loss_probability: 0.05,
            },
        }
    }

    #[test]
    fn terminal_robot_loss_branch_has_no_future_measurements() {
        let report = plan_survival_conditioned(
            &problem(vec![scan("survey")]),
            budget(),
            prices(),
            config(2),
        )
        .unwrap();

        assert_eq!(
            report.policy.measurement_action_id.as_deref(),
            Some("survey")
        );
        let lost = report
            .policy
            .branches
            .iter()
            .find(|branch| branch.kind == OutcomeKind::RobotLost)
            .expect("robot-loss branch");
        assert!(lost.continuation.is_none());

        let observed = report
            .policy
            .branches
            .iter()
            .find(|branch| branch.kind == OutcomeKind::Observation)
            .expect("observation branch");
        assert!(observed.continuation.is_some());
    }

    #[test]
    fn unsafe_interruption_probability_causes_stop_instead_of_measurement() {
        let mut dangerous = scan("dangerous-survey");
        dangerous.outcomes = vec![
            MeasurementOutcome {
                id: "observation".into(),
                kind: OutcomeKind::Observation,
                likelihood_by_hypothesis: vec![0.8, 0.8, 0.8],
                terminal_loss_by_hypothesis: None,
            },
            MeasurementOutcome {
                id: "interrupted".into(),
                kind: OutcomeKind::MissionInterrupted,
                likelihood_by_hypothesis: vec![0.2, 0.2, 0.2],
                terminal_loss_by_hypothesis: Some(vec![80.0, 80.0, 80.0]),
            },
        ];

        let report = plan_survival_conditioned(
            &problem(vec![dangerous]),
            budget(),
            prices(),
            config(2),
        )
        .unwrap();

        assert!(report.policy.measurement_action_id.is_none());
        assert_eq!(
            report.root_action_assessments[0].status,
            ActionStatus::RiskLimitExceeded
        );
    }

    #[test]
    fn resource_budget_is_enforced_across_observation_branches() {
        let mut action = scan("survey");
        action.cost.energy_j = 600.0;
        let small_budget = ResourceBudget {
            energy_j: 1_000.0,
            duration_s: 60.0,
            data_bytes: 10_000.0,
            exposure_units: 1.0,
        };

        let report = plan_survival_conditioned(
            &problem(vec![action]),
            small_budget,
            prices(),
            config(3),
        )
        .unwrap();

        assert_eq!(
            report.policy.measurement_action_id.as_deref(),
            Some("survey")
        );
        let observed = report
            .policy
            .branches
            .iter()
            .find(|branch| branch.kind == OutcomeKind::Observation)
            .unwrap();
        let child = observed.continuation.as_ref().unwrap();
        assert!(child.measurement_action_id.is_none());
    }

    #[test]
    fn safe_retreat_and_relay_store_forward_are_terminal_branches() {
        let mut action = scan("survey");
        action.outcomes = vec![
            MeasurementOutcome {
                id: "continue".into(),
                kind: OutcomeKind::Observation,
                likelihood_by_hypothesis: vec![0.7, 0.7, 0.7],
                terminal_loss_by_hypothesis: None,
            },
            MeasurementOutcome {
                id: "safe_retreat".into(),
                kind: OutcomeKind::SafeRetreat,
                likelihood_by_hypothesis: vec![0.2, 0.2, 0.2],
                terminal_loss_by_hypothesis: Some(vec![10.0, 10.0, 10.0]),
            },
            MeasurementOutcome {
                id: "relay_lost".into(),
                kind: OutcomeKind::RelayLostStoreAndForward,
                likelihood_by_hypothesis: vec![0.1, 0.1, 0.1],
                terminal_loss_by_hypothesis: Some(vec![15.0, 15.0, 15.0]),
            },
        ];

        let report = plan_survival_conditioned(
            &problem(vec![action]),
            budget(),
            prices(),
            config(2),
        )
        .unwrap();

        assert_eq!(report.policy.measurement_action_id.as_deref(), Some("survey"));
        assert!(report.metrics.mission_interruption_probability.abs() < 1e-9);
        assert!(report.policy.branches.iter().all(|branch| {
            if branch.kind.is_terminal() {
                branch.continuation.is_none()
            } else {
                branch.continuation.is_some()
            }
        }));
    }

    #[test]
    fn mandatory_open_world_unknown_is_validated() {
        let mut p = problem(vec![scan("survey")]);
        p.hypotheses.pop();
        assert_eq!(
            validate_problem(&p, config(2)),
            Err(PlanningError::MissingUnknownHypothesis)
        );
    }

    #[test]
    fn malformed_outcome_distribution_is_rejected() {
        let mut action = scan("survey");
        action.outcomes[0].likelihood_by_hypothesis[0] = 0.8;
        let p = problem(vec![action]);
        assert!(matches!(
            validate_problem(&p, config(2)),
            Err(PlanningError::LikelihoodsNotNormalized { .. })
        ));
    }

    #[test]
    fn action_order_does_not_change_selected_policy() {
        let left = plan_survival_conditioned(
            &problem(vec![scan("z-survey"), scan("a-survey")]),
            budget(),
            prices(),
            config(2),
        )
        .unwrap();
        let right = plan_survival_conditioned(
            &problem(vec![scan("a-survey"), scan("z-survey")]),
            budget(),
            prices(),
            config(2),
        )
        .unwrap();

        assert_eq!(left.policy, right.policy);
        assert_eq!(left.metrics, right.metrics);
    }

    #[test]
    fn horizon_limit_is_hard_capped() {
        assert!(matches!(
            plan_survival_conditioned(
                &problem(vec![scan("survey")]),
                budget(),
                prices(),
                config(MAX_HORIZON_STEPS + 1),
            ),
            Err(PlanningError::InvalidPlannerLimit(_))
        ));
    }

    #[test]
    fn search_expansion_is_bounded() {
        let mut limited = config(4);
        limited.maximum_search_nodes = 1;
        assert_eq!(
            plan_survival_conditioned(
                &problem(vec![scan("survey")]),
                budget(),
                prices(),
                limited
            ),
            Err(PlanningError::SearchBudgetExceeded { maximum: 1 })
        );
    }

    #[test]
    fn external_gate_denial_is_not_overridden_by_information_value() {
        let mut action = scan("denied");
        action.external_gate_passed = false;
        let report = plan_survival_conditioned(
            &problem(vec![action]),
            budget(),
            prices(),
            config(2),
        )
        .unwrap();
        assert!(report.policy.measurement_action_id.is_none());
        assert_eq!(
            report.root_action_assessments[0].status,
            ActionStatus::ExternalGateDenied
        );
    }

    #[test]
    fn repeated_measurements_are_only_planned_on_observation_branches() {
        let report = plan_survival_conditioned(
            &problem(vec![scan("survey")]),
            budget(),
            prices(),
            config(3),
        )
        .unwrap();

        assert_eq!(report.policy.measurement_action_id.as_deref(), Some("survey"));
        let observation = report.policy.branches.iter()
            .find(|branch| branch.kind == OutcomeKind::Observation)
            .unwrap();
        let child = observation.continuation.as_ref().unwrap();
        assert_eq!(child.measurement_action_id.as_deref(), Some("survey"));

        let lost = report.policy.branches.iter()
            .find(|branch| branch.kind == OutcomeKind::RobotLost)
            .unwrap();
        assert!(lost.continuation.is_none());
    }
}
