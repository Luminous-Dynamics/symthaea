//! Exact prior-free finite-horizon sequential experiment-policy reference.
//!
//! This evaluator is intentionally separate from the Bayesian EVSI evaluator.
//! It uses set-valued outcome supports and worst-case terminal task loss; it
//! assigns no probabilities to outcomes and never authorizes experiment
//! execution. The bounded dynamic programme is intended for tiny known-answer
//! worlds. Its exhaustive policy-tree enumerator is a separately written
//! cross-check, not a production planner.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{self, Display, Formatter};

const MAX_STATES: usize = 64;
const MAX_DECISIONS: usize = 32;
const MAX_ACTIONS: usize = 32;
const MAX_OUTCOMES_PER_ACTION: usize = 64;
const MAX_PROTOCOL_STATES: usize = 128;
const MAX_RESOURCE_DIMENSIONS: usize = 8;
const MAX_HORIZON_ACTIONS: u8 = 12;
const MAX_SEARCH_NODES: usize = 100_000;
const MAX_ENUMERATED_CHOICES: usize = 100_000;
const VALUE_TOLERANCE: f64 = 1.0e-12;

/// Frozen finite problem definition for prior-free sequential policy planning.
///
/// Resource dimensions are integer, application-declared units. Each action
/// must declare every budget dimension exactly once. Different dimensions are
/// enforced component-wise and are never converted into task-loss units.
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialRobustProblem {
    pub state_ids: Vec<String>,
    pub decision_ids: Vec<String>,
    /// decision_loss[decision_id][state_id] is a bounded terminal task loss.
    pub decision_loss: BTreeMap<String, BTreeMap<String, f64>>,
    pub maximum_worst_case_loss: f64,
    pub initial_protocol_state_id: String,
    pub protocol_states: Vec<SequentialProtocolState>,
    pub actions: Vec<SequentialAction>,
    pub resource_budget: BTreeMap<String, u64>,
    pub current_step: u64,
    pub decision_deadline_step: u64,
    pub maximum_actions: u8,
}

/// Publicly observable controller state that affects legal future actions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequentialProtocolState {
    pub protocol_state_id: String,
    pub legal_action_ids: Vec<String>,
}

/// Experiment/action contract. Outcome supports are evaluator-side model
/// semantics; candidate agents receive only the actual public outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequentialAction {
    pub action_id: String,
    pub resource_costs: BTreeMap<String, u64>,
    pub result_delay_steps: u64,
    pub outcomes: Vec<SequentialOutcome>,
}

/// A set-valued outcome. Overlapping supports are permitted and represent
/// observations compatible with multiple hidden states.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequentialOutcome {
    pub outcome_id: String,
    pub possible_state_ids: Vec<String>,
    pub next_protocol_state_id: String,
}

/// One branch of an adaptive policy tree.
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialPolicyBranch {
    pub outcome_id: String,
    pub possible_state_ids: Vec<String>,
    pub next_protocol_state_id: String,
    pub continuation: Box<SequentialPolicyNode>,
}

/// Complete contingent policy. A stop node selects one caller-declared
/// terminal decision; a probe node selects an action and a continuation for
/// every possible public outcome.
#[derive(Clone, Debug, PartialEq)]
pub enum SequentialPolicyNode {
    Stop {
        decision_id: String,
        worst_case_terminal_loss: f64,
    },
    Probe {
        action_id: String,
        branches: Vec<SequentialPolicyBranch>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct SequentialPolicyValue {
    pub worst_case_terminal_loss: f64,
    pub root_risk_gate_passed: bool,
    pub explored_nodes: usize,
    pub policy: SequentialPolicyNode,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SequentialOracleError {
    EmptyStateSet,
    TooManyStates(usize),
    EmptyDecisionSet,
    TooManyDecisions(usize),
    TooManyActions(usize),
    TooManyProtocolStates(usize),
    TooManyOutcomes(usize),
    TooManyResourceDimensions(usize),
    TooManyHorizonActions(u8),
    EmptyIdentifier(&'static str),
    DuplicateIdentifier {
        field: &'static str,
        id: String,
    },
    UnknownState(String),
    UnknownDecision(String),
    UnknownAction(String),
    UnknownProtocolState(String),
    InvalidRiskThreshold(f64),
    InvalidLoss {
        decision_id: String,
        state_id: String,
        value_bits: u64,
    },
    MissingDecisionLoss {
        decision_id: String,
        state_id: String,
    },
    EmptyResourceBudget,
    ResourceDimensionMismatch {
        action_id: String,
        expected: Vec<String>,
        actual: Vec<String>,
    },
    InvalidOutcomeSupport(String),
    UncoveredState {
        action_id: String,
        state_id: String,
    },
    InvalidDeadline,
    TimeOverflow,
    SearchLimitExceeded(usize),
    EnumerationLimitExceeded(usize),
    NoPolicy,
}

impl Display for SequentialOracleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for SequentialOracleError {}

fn validate(problem: &SequentialRobustProblem) -> Result<(), SequentialOracleError> {
    if problem.state_ids.is_empty() {
        return Err(SequentialOracleError::EmptyStateSet);
    }
    if problem.state_ids.len() > MAX_STATES {
        return Err(SequentialOracleError::TooManyStates(problem.state_ids.len()));
    }
    if problem.decision_ids.is_empty() {
        return Err(SequentialOracleError::EmptyDecisionSet);
    }
    if problem.decision_ids.len() > MAX_DECISIONS {
        return Err(SequentialOracleError::TooManyDecisions(
            problem.decision_ids.len(),
        ));
    }
    if problem.actions.len() > MAX_ACTIONS {
        return Err(SequentialOracleError::TooManyActions(problem.actions.len()));
    }
    if problem.protocol_states.len() > MAX_PROTOCOL_STATES {
        return Err(SequentialOracleError::TooManyProtocolStates(
            problem.protocol_states.len(),
        ));
    }
    if problem.maximum_actions > MAX_HORIZON_ACTIONS {
        return Err(SequentialOracleError::TooManyHorizonActions(
            problem.maximum_actions,
        ));
    }
    if !problem.maximum_worst_case_loss.is_finite()
        || !(0.0..=1.0).contains(&problem.maximum_worst_case_loss)
    {
        return Err(SequentialOracleError::InvalidRiskThreshold(
            problem.maximum_worst_case_loss,
        ));
    }
    if problem.decision_deadline_step < problem.current_step {
        return Err(SequentialOracleError::InvalidDeadline);
    }
    if problem.resource_budget.is_empty() {
        return Err(SequentialOracleError::EmptyResourceBudget);
    }
    if problem.resource_budget.len() > MAX_RESOURCE_DIMENSIONS {
        return Err(SequentialOracleError::TooManyResourceDimensions(
            problem.resource_budget.len(),
        ));
    }
    if problem
        .resource_budget
        .keys()
        .any(|dimension| dimension.trim().is_empty())
    {
        return Err(SequentialOracleError::EmptyIdentifier("resource dimension"));
    }

    let mut states = BTreeSet::new();
    for state_id in &problem.state_ids {
        if state_id.trim().is_empty() {
            return Err(SequentialOracleError::EmptyIdentifier("state"));
        }
        if !states.insert(state_id.as_str()) {
            return Err(SequentialOracleError::DuplicateIdentifier {
                field: "state",
                id: state_id.clone(),
            });
        }
    }

    let mut decisions = BTreeSet::new();
    for decision_id in &problem.decision_ids {
        if decision_id.trim().is_empty() {
            return Err(SequentialOracleError::EmptyIdentifier("decision"));
        }
        if !decisions.insert(decision_id.as_str()) {
            return Err(SequentialOracleError::DuplicateIdentifier {
                field: "decision",
                id: decision_id.clone(),
            });
        }
        let losses = problem.decision_loss.get(decision_id).ok_or_else(|| {
            SequentialOracleError::MissingDecisionLoss {
                decision_id: decision_id.clone(),
                state_id: problem.state_ids[0].clone(),
            }
        })?;
        for state_id in &problem.state_ids {
            let loss = losses.get(state_id).copied().ok_or_else(|| {
                SequentialOracleError::MissingDecisionLoss {
                    decision_id: decision_id.clone(),
                    state_id: state_id.clone(),
                }
            })?;
            if !loss.is_finite() || !(0.0..=1.0).contains(&loss) {
                return Err(SequentialOracleError::InvalidLoss {
                    decision_id: decision_id.clone(),
                    state_id: state_id.clone(),
                    value_bits: loss.to_bits(),
                });
            }
        }
    }

    for decision_id in problem.decision_loss.keys() {
        if !decisions.contains(decision_id.as_str()) {
            return Err(SequentialOracleError::UnknownDecision(decision_id.clone()));
        }
    }
    for (decision_id, losses) in &problem.decision_loss {
        for state_id in losses.keys() {
            if !states.contains(state_id.as_str()) {
                return Err(SequentialOracleError::UnknownState(state_id.clone()));
            }
        }
        if !decisions.contains(decision_id.as_str()) {
            return Err(SequentialOracleError::UnknownDecision(decision_id.clone()));
        }
    }

    let resource_dimensions: Vec<String> = problem.resource_budget.keys().cloned().collect();
    let protocol_ids: BTreeSet<&str> = problem
        .protocol_states
        .iter()
        .map(|state| state.protocol_state_id.as_str())
        .collect();
    if !protocol_ids.contains(problem.initial_protocol_state_id.as_str()) {
        return Err(SequentialOracleError::UnknownProtocolState(
            problem.initial_protocol_state_id.clone(),
        ));
    }

    let mut protocol_seen = BTreeSet::new();
    for protocol_state in &problem.protocol_states {
        if protocol_state.protocol_state_id.trim().is_empty() {
            return Err(SequentialOracleError::EmptyIdentifier("protocol state"));
        }
        if !protocol_seen.insert(protocol_state.protocol_state_id.as_str()) {
            return Err(SequentialOracleError::DuplicateIdentifier {
                field: "protocol state",
                id: protocol_state.protocol_state_id.clone(),
            });
        }
    }

    let mut action_seen = BTreeSet::new();
    for action in &problem.actions {
        if action.action_id.trim().is_empty() {
            return Err(SequentialOracleError::EmptyIdentifier("action"));
        }
        if !action_seen.insert(action.action_id.as_str()) {
            return Err(SequentialOracleError::DuplicateIdentifier {
                field: "action",
                id: action.action_id.clone(),
            });
        }
        if action.outcomes.is_empty() || action.outcomes.len() > MAX_OUTCOMES_PER_ACTION {
            return Err(SequentialOracleError::TooManyOutcomes(
                action.outcomes.len(),
            ));
        }
        let actual_dimensions: Vec<String> = action.resource_costs.keys().cloned().collect();
        if actual_dimensions != resource_dimensions {
            return Err(SequentialOracleError::ResourceDimensionMismatch {
                action_id: action.action_id.clone(),
                expected: resource_dimensions.clone(),
                actual: actual_dimensions,
            });
        }
        let mut outcome_seen = BTreeSet::new();
        for outcome in &action.outcomes {
            if outcome.outcome_id.trim().is_empty() {
                return Err(SequentialOracleError::EmptyIdentifier("outcome"));
            }
            if !outcome_seen.insert(outcome.outcome_id.as_str()) {
                return Err(SequentialOracleError::DuplicateIdentifier {
                    field: "outcome",
                    id: outcome.outcome_id.clone(),
                });
            }
            if outcome.possible_state_ids.is_empty() {
                return Err(SequentialOracleError::InvalidOutcomeSupport(
                    outcome.outcome_id.clone(),
                ));
            }
            if !protocol_ids.contains(outcome.next_protocol_state_id.as_str()) {
                return Err(SequentialOracleError::UnknownProtocolState(
                    outcome.next_protocol_state_id.clone(),
                ));
            }
            let mut local_seen = BTreeSet::new();
            for state_id in &outcome.possible_state_ids {
                if !states.contains(state_id.as_str()) {
                    return Err(SequentialOracleError::UnknownState(state_id.clone()));
                }
                if !local_seen.insert(state_id.as_str()) {
                    return Err(SequentialOracleError::DuplicateIdentifier {
                        field: "state in outcome support",
                        id: state_id.clone(),
                    });
                }
            }
        }
    }

    let action_ids: BTreeSet<&str> = action_seen;
    for protocol_state in &problem.protocol_states {
        let mut legal_seen = BTreeSet::new();
        for action_id in &protocol_state.legal_action_ids {
            if !action_ids.contains(action_id.as_str()) {
                return Err(SequentialOracleError::UnknownAction(action_id.clone()));
            }
            if !legal_seen.insert(action_id.as_str()) {
                return Err(SequentialOracleError::DuplicateIdentifier {
                    field: "legal action",
                    id: action_id.clone(),
                });
            }
        }
    }
    Ok(())
}

fn terminal_minimax(
    problem: &SequentialRobustProblem,
    possible_states: &[String],
) -> Result<(f64, String), SequentialOracleError> {
    let mut ordered_decisions = problem.decision_ids.clone();
    ordered_decisions.sort();
    let mut best: Option<(f64, String)> = None;

    for decision_id in ordered_decisions {
        let losses = problem.decision_loss.get(&decision_id).ok_or_else(|| {
            SequentialOracleError::UnknownDecision(decision_id.clone())
        })?;
        let mut worst_loss = 0.0_f64;
        for state_id in possible_states {
            let loss = losses.get(state_id).copied().ok_or_else(|| {
                SequentialOracleError::MissingDecisionLoss {
                    decision_id: decision_id.clone(),
                    state_id: state_id.clone(),
                }
            })?;
            worst_loss = worst_loss.max(loss);
        }
        let replace = match &best {
            None => true,
            Some((best_loss, best_decision)) => {
                worst_loss < *best_loss
                    || (worst_loss == *best_loss
                        && decision_id.as_str() < best_decision.as_str())
            }
        };
        if replace {
            best = Some((worst_loss, decision_id));
        }
    }
    best.ok_or(SequentialOracleError::EmptyDecisionSet)
}

fn feasible_action(
    problem: &SequentialRobustProblem,
    action: &SequentialAction,
    remaining_budget: &BTreeMap<String, u64>,
    current_step: u64,
) -> Result<Option<(BTreeMap<String, u64>, u64)>, SequentialOracleError> {
    let next_step = current_step
        .checked_add(action.result_delay_steps)
        .ok_or(SequentialOracleError::TimeOverflow)?;
    if next_step > problem.decision_deadline_step {
        return Ok(None);
    }

    let mut next_budget = BTreeMap::new();
    for (dimension, remaining) in remaining_budget {
        let cost = action.resource_costs.get(dimension).copied().ok_or_else(|| {
            SequentialOracleError::ResourceDimensionMismatch {
                action_id: action.action_id.clone(),
                expected: remaining_budget.keys().cloned().collect(),
                actual: action.resource_costs.keys().cloned().collect(),
            }
        })?;
        if cost > *remaining {
            return Ok(None);
        }
        next_budget.insert(dimension.clone(), remaining - cost);
    }
    Ok(Some((next_budget, next_step)))
}

/// Return the non-empty intersections of outcome supports with the current
/// possible-state set, requiring the outcomes to cover that entire set.
fn active_branches(
    action: &SequentialAction,
    possible_states: &[String],
) -> Result<Vec<(String, Vec<String>, String)>, SequentialOracleError> {
    let current: BTreeSet<&str> = possible_states.iter().map(String::as_str).collect();
    let mut covered: BTreeSet<String> = BTreeSet::new();
    let mut branches = Vec::new();

    for outcome in &action.outcomes {
        let mut support: Vec<String> = outcome
            .possible_state_ids
            .iter()
            .filter(|state_id| current.contains(state_id.as_str()))
            .cloned()
            .collect();
        support.sort();
        if support.is_empty() {
            continue;
        }
        for state_id in &support {
            covered.insert(state_id.clone());
        }
        branches.push((
            outcome.outcome_id.clone(),
            support,
            outcome.next_protocol_state_id.clone(),
        ));
    }

    for state_id in possible_states {
        if !covered.contains(state_id) {
            return Err(SequentialOracleError::UncoveredState {
                action_id: action.action_id.clone(),
                state_id: state_id.clone(),
            });
        }
    }
    Ok(branches)
}

struct SolvedNode {
    loss: f64,
    policy: SequentialPolicyNode,
}

fn bump_search(nodes: &mut usize) -> Result<(), SequentialOracleError> {
    *nodes = (*nodes).saturating_add(1);
    if *nodes > MAX_SEARCH_NODES {
        return Err(SequentialOracleError::SearchLimitExceeded(*nodes));
    }
    Ok(())
}

fn solve_node(
    problem: &SequentialRobustProblem,
    possible_states: &[String],
    protocol_state_id: &str,
    remaining_budget: &BTreeMap<String, u64>,
    current_step: u64,
    actions_left: u8,
    nodes: &mut usize,
) -> Result<SolvedNode, SequentialOracleError> {
    bump_search(nodes)?;
    let (stop_loss, stop_decision) = terminal_minimax(problem, possible_states)?;
    let mut best = SolvedNode {
        loss: stop_loss,
        policy: SequentialPolicyNode::Stop {
            decision_id: stop_decision,
            worst_case_terminal_loss: stop_loss,
        },
    };
    if actions_left == 0 {
        return Ok(best);
    }

    let protocol_state = problem
        .protocol_states
        .iter()
        .find(|state| state.protocol_state_id == protocol_state_id)
        .ok_or_else(|| {
            SequentialOracleError::UnknownProtocolState(protocol_state_id.to_owned())
        })?;
    let mut legal_action_ids = protocol_state.legal_action_ids.clone();
    legal_action_ids.sort();

    for action_id in legal_action_ids {
        let action = problem
            .actions
            .iter()
            .find(|candidate| candidate.action_id == action_id)
            .ok_or_else(|| SequentialOracleError::UnknownAction(action_id.clone()))?;
        let Some((next_budget, next_step)) =
            feasible_action(problem, action, remaining_budget, current_step)?
        else {
            continue;
        };
        let branches = active_branches(action, possible_states)?;
        let mut policy_branches = Vec::with_capacity(branches.len());
        let mut action_worst_case_loss = 0.0_f64;

        for (outcome_id, support, next_protocol_state_id) in branches {
            let continuation = solve_node(
                problem,
                &support,
                &next_protocol_state_id,
                &next_budget,
                next_step,
                actions_left - 1,
                nodes,
            )?;
            action_worst_case_loss = action_worst_case_loss.max(continuation.loss);
            policy_branches.push(SequentialPolicyBranch {
                outcome_id,
                possible_state_ids: support,
                next_protocol_state_id,
                continuation: Box::new(continuation.policy),
            });
        }

        // Stop wins exact ties: the oracle does not buy an experiment that
        // cannot strictly improve the frozen terminal objective.
        if action_worst_case_loss + VALUE_TOLERANCE < best.loss {
            best = SolvedNode {
                loss: action_worst_case_loss,
                policy: SequentialPolicyNode::Probe {
                    action_id: action.action_id.clone(),
                    branches: policy_branches,
                },
            };
        }
    }
    Ok(best)
}

/// Solve the prior-free finite-horizon policy problem using backward dynamic
/// programming over set-valued supports and public protocol state.
///
/// The returned root risk-gate flag is a status measurement, not an experiment
/// authorization. Action permissions in this model are research-policy
/// constraints only; physical execution must use its existing authority path.
pub fn evaluate_sequential_robust(
    problem: &SequentialRobustProblem,
) -> Result<SequentialPolicyValue, SequentialOracleError> {
    validate(problem)?;
    let mut possible_states = problem.state_ids.clone();
    possible_states.sort();
    let mut nodes = 0;
    let solved = solve_node(
        problem,
        &possible_states,
        &problem.initial_protocol_state_id,
        &problem.resource_budget,
        problem.current_step,
        problem.maximum_actions,
        &mut nodes,
    )?;
    Ok(SequentialPolicyValue {
        worst_case_terminal_loss: solved.loss,
        root_risk_gate_passed: solved.loss <= problem.maximum_worst_case_loss + VALUE_TOLERANCE,
        explored_nodes: nodes,
        policy: solved.policy,
    })
}

#[derive(Clone)]
struct EnumeratedPolicy {
    loss: f64,
    policy: SequentialPolicyNode,
}

fn bump_enumeration(work: &mut usize) -> Result<(), SequentialOracleError> {
    *work = (*work).saturating_add(1);
    if *work > MAX_ENUMERATED_CHOICES {
        return Err(SequentialOracleError::EnumerationLimitExceeded(*work));
    }
    Ok(())
}

/// Independently enumerate every legal contingent policy tree for a tiny
/// world. This intentionally does not call the dynamic-programming solver or
/// its terminal-value helper. It is computationally bounded and exists only
/// to cross-check known-answer worlds.
fn enumerate_node(
    problem: &SequentialRobustProblem,
    possible_states: &[String],
    protocol_state_id: &str,
    remaining_budget: &BTreeMap<String, u64>,
    current_step: u64,
    actions_left: u8,
    work: &mut usize,
) -> Result<Vec<EnumeratedPolicy>, SequentialOracleError> {
    bump_enumeration(work)?;

    let mut stop_value = f64::INFINITY;
    let mut stop_decision: Option<String> = None;
    for decision_id in &problem.decision_ids {
        let mut worst = 0.0_f64;
        for state_id in possible_states {
            let loss = problem
                .decision_loss
                .get(decision_id)
                .and_then(|by_state| by_state.get(state_id))
                .copied()
                .ok_or_else(|| SequentialOracleError::MissingDecisionLoss {
                    decision_id: decision_id.clone(),
                    state_id: state_id.clone(),
                })?;
            worst = if loss > worst { loss } else { worst };
        }
        let is_better = match &stop_decision {
            None => true,
            Some(current_id) => {
                worst < stop_value
                    || (worst == stop_value && decision_id < current_id)
            }
        };
        if is_better {
            stop_value = worst;
            stop_decision = Some(decision_id.clone());
        }
    }
    let stop_decision = stop_decision.ok_or(SequentialOracleError::EmptyDecisionSet)?;
    let mut all_policies = vec![EnumeratedPolicy {
        loss: stop_value,
        policy: SequentialPolicyNode::Stop {
            decision_id: stop_decision,
            worst_case_terminal_loss: stop_value,
        },
    }];
    if actions_left == 0 {
        return Ok(all_policies);
    }

    let protocol = problem
        .protocol_states
        .iter()
        .find(|state| state.protocol_state_id == protocol_state_id)
        .ok_or_else(|| {
            SequentialOracleError::UnknownProtocolState(protocol_state_id.to_owned())
        })?;
    let mut legal = protocol.legal_action_ids.clone();
    legal.sort();

    for action_id in legal {
        let action = problem
            .actions
            .iter()
            .find(|candidate| candidate.action_id == action_id)
            .ok_or_else(|| SequentialOracleError::UnknownAction(action_id.clone()))?;
        let Some((budget_after, step_after)) =
            feasible_action(problem, action, remaining_budget, current_step)?
        else {
            continue;
        };
        let active = active_branches(action, possible_states)?;

        // Cartesian product of all branch-continuation choices constructs
        // complete policies. Each branch sees only its own public outcome.
        let mut combinations: Vec<(Vec<SequentialPolicyBranch>, f64)> =
            vec![(Vec::new(), 0.0)];
        for (outcome_id, support, next_protocol_state_id) in active {
            let child_policies = enumerate_node(
                problem,
                &support,
                &next_protocol_state_id,
                &budget_after,
                step_after,
                actions_left - 1,
                work,
            )?;
            let mut next_combinations = Vec::new();
            for (partial_branches, partial_worst_loss) in &combinations {
                for child in &child_policies {
                    bump_enumeration(work)?;
                    let mut branches = partial_branches.clone();
                    branches.push(SequentialPolicyBranch {
                        outcome_id: outcome_id.clone(),
                        possible_state_ids: support.clone(),
                        next_protocol_state_id: next_protocol_state_id.clone(),
                        continuation: Box::new(child.policy.clone()),
                    });
                    next_combinations.push((
                        branches,
                        partial_worst_loss.max(child.loss),
                    ));
                    if next_combinations.len() > MAX_ENUMERATED_CHOICES {
                        return Err(SequentialOracleError::EnumerationLimitExceeded(
                            next_combinations.len(),
                        ));
                    }
                }
            }
            combinations = next_combinations;
        }

        for (branches, loss) in combinations {
            bump_enumeration(work)?;
            all_policies.push(EnumeratedPolicy {
                loss,
                policy: SequentialPolicyNode::Probe {
                    action_id: action.action_id.clone(),
                    branches,
                },
            });
            if all_policies.len() > MAX_ENUMERATED_CHOICES {
                return Err(SequentialOracleError::EnumerationLimitExceeded(
                    all_policies.len(),
                ));
            }
        }
    }
    Ok(all_policies)
}

/// Exhaustive policy-tree oracle for tiny finite worlds. It is intentionally
/// independent of the dynamic-programming recurrence and should only be used
/// on bounded fixtures.
pub fn enumerate_sequential_robust(
    problem: &SequentialRobustProblem,
) -> Result<SequentialPolicyValue, SequentialOracleError> {
    validate(problem)?;
    let mut possible_states = problem.state_ids.clone();
    possible_states.sort();
    let mut work = 0;
    let policies = enumerate_node(
        problem,
        &possible_states,
        &problem.initial_protocol_state_id,
        &problem.resource_budget,
        problem.current_step,
        problem.maximum_actions,
        &mut work,
    )?;
    let mut best: Option<EnumeratedPolicy> = None;
    for candidate in policies {
        let replace = match &best {
            None => true,
            Some(current) => candidate.loss + VALUE_TOLERANCE < current.loss,
        };
        if replace {
            best = Some(candidate);
        }
    }
    let best = best.ok_or(SequentialOracleError::NoPolicy)?;
    Ok(SequentialPolicyValue {
        worst_case_terminal_loss: best.loss,
        root_risk_gate_passed: best.loss <= problem.maximum_worst_case_loss + VALUE_TOLERANCE,
        explored_nodes: work,
        policy: best.policy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(units: u64) -> BTreeMap<String, u64> {
        BTreeMap::from([("experiment_units".to_owned(), units)])
    }

    fn support(states: &[String], predicate: impl Fn(&str) -> bool) -> Vec<String> {
        states
            .iter()
            .filter(|state_id| predicate(state_id.as_str()))
            .cloned()
            .collect()
    }

    fn branch_unlock_world() -> SequentialRobustProblem {
        let mut state_ids = Vec::new();
        let mut loss_d0 = BTreeMap::new();
        let mut loss_d1 = BTreeMap::new();
        for x in 0..=1 {
            for z in 0..=1 {
                for n in 0..=3 {
                    let id = format!("x{x}_z{z}_n{n}");
                    state_ids.push(id.clone());
                    loss_d0.insert(id.clone(), if x == 0 { 0.0 } else { 1.0 });
                    loss_d1.insert(id, if x == 1 { 0.0 } else { 1.0 });
                }
            }
        }

        let outcomes_a = (0..=3)
            .map(|n| SequentialOutcome {
                outcome_id: format!("n{n}"),
                possible_state_ids: support(&state_ids, |id| {
                    id.ends_with(format!("n{n}").as_str())
                }),
                next_protocol_state_id: "after_A".to_owned(),
            })
            .collect();
        let outcomes_b = (0..=1)
            .map(|z| SequentialOutcome {
                outcome_id: format!("z{z}"),
                possible_state_ids: support(&state_ids, |id| {
                    id.contains(format!("_z{z}_").as_str())
                }),
                next_protocol_state_id: format!("after_B_z{z}"),
            })
            .collect();

        let (action_c0, action_c1) = {
            let action_c = |z: u8| SequentialAction {
                action_id: format!("C{z}"),
                resource_costs: cost(1),
                result_delay_steps: 1,
                outcomes: (0..=1)
                    .map(|x| SequentialOutcome {
                        outcome_id: format!("x{x}"),
                        possible_state_ids: support(&state_ids, |id| {
                            id.starts_with(format!("x{x}_z{z}_").as_str())
                        }),
                        next_protocol_state_id: format!("terminal_x{x}"),
                    })
                    .collect(),
            };
            (action_c(0), action_c(1))
        };

        SequentialRobustProblem {
            state_ids,
            decision_ids: vec!["d0".to_owned(), "d1".to_owned()],
            decision_loss: BTreeMap::from([
                ("d0".to_owned(), loss_d0),
                ("d1".to_owned(), loss_d1),
            ]),
            maximum_worst_case_loss: 0.0,
            initial_protocol_state_id: "root".to_owned(),
            protocol_states: vec![
                SequentialProtocolState {
                    protocol_state_id: "root".to_owned(),
                    legal_action_ids: vec!["A".to_owned(), "B".to_owned()],
                },
                SequentialProtocolState {
                    protocol_state_id: "after_A".to_owned(),
                    legal_action_ids: vec![],
                },
                SequentialProtocolState {
                    protocol_state_id: "after_B_z0".to_owned(),
                    legal_action_ids: vec!["C0".to_owned()],
                },
                SequentialProtocolState {
                    protocol_state_id: "after_B_z1".to_owned(),
                    legal_action_ids: vec!["C1".to_owned()],
                },
                SequentialProtocolState {
                    protocol_state_id: "terminal_x0".to_owned(),
                    legal_action_ids: vec![],
                },
                SequentialProtocolState {
                    protocol_state_id: "terminal_x1".to_owned(),
                    legal_action_ids: vec![],
                },
            ],
            actions: vec![
                SequentialAction {
                    action_id: "A".to_owned(),
                    resource_costs: cost(1),
                    result_delay_steps: 1,
                    outcomes: outcomes_a,
                },
                SequentialAction {
                    action_id: "B".to_owned(),
                    resource_costs: cost(1),
                    result_delay_steps: 1,
                    outcomes: outcomes_b,
                },
                action_c0,
                action_c1,
            ],
            resource_budget: cost(2),
            current_step: 0,
            decision_deadline_step: 2,
            maximum_actions: 2,
        }
    }

    fn root_action(policy: &SequentialPolicyNode) -> Option<&str> {
        match policy {
            SequentialPolicyNode::Stop { .. } => None,
            SequentialPolicyNode::Probe { action_id, .. } => Some(action_id),
        }
    }

    #[test]
    fn dynamic_programme_finds_non_myopic_branch_unlock_policy() {
        let problem = branch_unlock_world();
        let value = evaluate_sequential_robust(&problem).unwrap();
        assert_eq!(value.worst_case_terminal_loss, 0.0);
        assert!(value.root_risk_gate_passed);
        assert_eq!(root_action(&value.policy), Some("B"));

        let SequentialPolicyNode::Probe { branches, .. } = value.policy else {
            panic!("expected a probing root policy");
        };
        assert_eq!(branches.len(), 2);
        for branch in branches {
            let expected = if branch.outcome_id == "z0" { "C0" } else { "C1" };
            assert_eq!(root_action(&branch.continuation), Some(expected));
        }
    }

    #[test]
    fn dynamic_programme_matches_independent_exhaustive_policy_tree_oracle() {
        let problem = branch_unlock_world();
        let dynamic = evaluate_sequential_robust(&problem).unwrap();
        let exhaustive = enumerate_sequential_robust(&problem).unwrap();
        assert_eq!(
            dynamic.worst_case_terminal_loss,
            exhaustive.worst_case_terminal_loss
        );
        assert_eq!(dynamic.worst_case_terminal_loss, 0.0);
        assert_eq!(root_action(&dynamic.policy), Some("B"));
        assert_eq!(root_action(&exhaustive.policy), Some("B"));
    }

    #[test]
    fn dead_end_nuisance_action_is_not_purchased_without_decision_value() {
        let problem = branch_unlock_world();
        let value = evaluate_sequential_robust(&problem).unwrap();
        // A is considered, but the exact policy selects B because only B
        // unlocks a branch-specific diagnostic that changes terminal loss.
        assert_eq!(root_action(&value.policy), Some("B"));
    }

    #[test]
    fn overlapping_outcome_supports_preserve_every_compatible_state() {
        let states = vec!["s0".to_owned(), "s1".to_owned()];
        let problem = SequentialRobustProblem {
            state_ids: states.clone(),
            decision_ids: vec!["d0".to_owned(), "d1".to_owned()],
            decision_loss: BTreeMap::from([
                ("d0".to_owned(), BTreeMap::from([
                    ("s0".to_owned(), 0.0), ("s1".to_owned(), 1.0),
                ])),
                ("d1".to_owned(), BTreeMap::from([
                    ("s0".to_owned(), 1.0), ("s1".to_owned(), 0.0),
                ])),
            ]),
            maximum_worst_case_loss: 0.0,
            initial_protocol_state_id: "root".to_owned(),
            protocol_states: vec![
                SequentialProtocolState {
                    protocol_state_id: "root".to_owned(),
                    legal_action_ids: vec!["observe".to_owned()],
                },
                SequentialProtocolState {
                    protocol_state_id: "after".to_owned(),
                    legal_action_ids: vec![],
                },
            ],
            actions: vec![SequentialAction {
                action_id: "observe".to_owned(),
                resource_costs: cost(1),
                result_delay_steps: 1,
                outcomes: vec![
                    SequentialOutcome {
                        outcome_id: "ambiguous_a".to_owned(),
                        possible_state_ids: states.clone(),
                        next_protocol_state_id: "after".to_owned(),
                    },
                    SequentialOutcome {
                        outcome_id: "ambiguous_b".to_owned(),
                        possible_state_ids: states.clone(),
                        next_protocol_state_id: "after".to_owned(),
                    },
                ],
            }],
            resource_budget: cost(1),
            current_step: 0,
            decision_deadline_step: 1,
            maximum_actions: 1,
        };
        let value = evaluate_sequential_robust(&problem).unwrap();
        assert_eq!(value.worst_case_terminal_loss, 1.0);
        assert!(matches!(value.policy, SequentialPolicyNode::Stop { .. }));
    }

    #[test]
    fn an_action_that_misses_the_deadline_is_not_selected() {
        let mut problem = branch_unlock_world();
        problem.decision_deadline_step = 0;
        let value = evaluate_sequential_robust(&problem).unwrap();
        assert_eq!(value.worst_case_terminal_loss, 1.0);
        assert!(matches!(value.policy, SequentialPolicyNode::Stop { .. }));
    }

    #[test]
    fn every_resource_dimension_is_enforced_component_wise() {
        let mut problem = branch_unlock_world();
        problem.resource_budget.insert("compute_units".to_owned(), 1);
        for action in &mut problem.actions {
            action.resource_costs.insert("compute_units".to_owned(), 0);
        }
        problem.actions[1].resource_costs.insert("compute_units".to_owned(), 2);
        let value = evaluate_sequential_robust(&problem).unwrap();
        assert_eq!(value.worst_case_terminal_loss, 1.0);
        assert!(matches!(value.policy, SequentialPolicyNode::Stop { .. }));
    }
}
