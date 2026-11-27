//! Exact prior-free finite-horizon sequential experiment-policy reference.
//!
//! This evaluator is intentionally separate from the Bayesian EVSI evaluator.
//! It uses set-valued outcome supports and worst-case terminal task loss; it
//! assigns no probabilities to outcomes and never authorizes experiment
//! execution. The bounded dynamic programme is intended for tiny known-answer
//! worlds. Its exhaustive policy-tree enumerator is a separately written
//! cross-check, not a production planner.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
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

/// One branch of an evaluator-side adaptive policy tree.
///
/// The possible-state support is hidden model information. Never serialize
/// this branch or its support into a candidate-facing AgentView.
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

/// Evaluator-only oracle output. Neither the policy tree nor its
/// possible-state support sets may be disclosed to a candidate during a blind
/// episode. This value is a reference for scoring, not an agent action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequentialResourceStep {
    pub action_id: String,
    pub outcome_id: String,
    /// Evaluator-side support remaining after this public outcome.
    pub possible_state_ids: Vec<String>,
}

/// Exact resource and terminal-decision receipt for one policy branch.
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialTerminalResourcePath {
    /// Action/outcome sequence from the root to this terminal node.
    pub steps: Vec<SequentialResourceStep>,
    /// Compatible hidden states remaining at the terminal decision.
    pub terminal_possible_state_ids: Vec<String>,
    /// Decision selected at this terminal leaf.
    pub terminal_decision_id: String,
    /// Worst-case task loss of the selected decision on this terminal support.
    pub terminal_worst_case_loss: f64,
    /// Total resource units along this path, kept separate by dimension.
    pub resource_usage: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SequentialPolicyValue {
    /// Best terminal decision loss available immediately, without another action.
    pub current_minimax_loss: f64,
    /// Whether the current information set admits a declared terminal decision.
    pub current_risk_gate_passed: bool,
    /// Worst-case terminal decision loss under the selected complete policy.
    pub worst_case_terminal_loss: f64,
    /// Whether every terminal branch of the selected policy meets the risk limit.
    /// This does not imply that a decision is admissible before the policy runs.
    pub policy_terminal_risk_gate_passed: bool,
    /// Exact resource vector for every terminal branch of the selected policy.
    pub terminal_resource_paths: Vec<SequentialTerminalResourcePath>,
    /// Bounded search work; DP-node and exhaustive-enumerator counts are distinct.
    pub explored_nodes: usize,
    /// Evaluator-only policy tree. Never expose state supports to a candidate.
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
    ResourceOverflow {
        action_id: String,
        dimension: String,
    },
    TooManyTerminalResourcePaths(usize),
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
        return Err(SequentialOracleError::TooManyStates(
            problem.state_ids.len(),
        ));
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

    // Validate outcome exhaustiveness against the state support actually
    // reachable at each public protocol node. Branch-specific actions need
    // not cover the original universe, only states that can reach them.
    // Each (protocol node, possible state) pair is processed at most once.
    let mut reachable: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut worklist = VecDeque::new();
    let initial_support = reachable
        .entry(problem.initial_protocol_state_id.clone())
        .or_default();
    for state_id in &problem.state_ids {
        if initial_support.insert(state_id.clone()) {
            worklist.push_back((problem.initial_protocol_state_id.clone(), state_id.clone()));
        }
    }

    while let Some((protocol_state_id, state_id)) = worklist.pop_front() {
        let protocol_state = problem
            .protocol_states
            .iter()
            .find(|state| state.protocol_state_id == protocol_state_id)
            .ok_or_else(|| {
                SequentialOracleError::UnknownProtocolState(protocol_state_id.clone())
            })?;

        for action_id in &protocol_state.legal_action_ids {
            let action = problem
                .actions
                .iter()
                .find(|candidate| candidate.action_id == *action_id)
                .ok_or_else(|| SequentialOracleError::UnknownAction(action_id.clone()))?;
            let mut covered = false;

            for outcome in &action.outcomes {
                if !outcome.possible_state_ids.contains(&state_id) {
                    continue;
                }
                covered = true;
                let next_protocol_state_id = &outcome.next_protocol_state_id;
                let next_support = reachable.entry(next_protocol_state_id.clone()).or_default();
                if next_support.insert(state_id.clone()) {
                    worklist.push_back((next_protocol_state_id.clone(), state_id.clone()));
                }
            }

            if !covered {
                return Err(SequentialOracleError::UncoveredState {
                    action_id: action.action_id.clone(),
                    state_id: state_id.clone(),
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
        let losses = problem
            .decision_loss
            .get(&decision_id)
            .ok_or_else(|| SequentialOracleError::UnknownDecision(decision_id.clone()))?;
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
                    || (worst_loss == *best_loss && decision_id.as_str() < best_decision.as_str())
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
        let cost = action
            .resource_costs
            .get(dimension)
            .copied()
            .ok_or_else(|| SequentialOracleError::ResourceDimensionMismatch {
                action_id: action.action_id.clone(),
                expected: remaining_budget.keys().cloned().collect(),
                actual: action.resource_costs.keys().cloned().collect(),
            })?;
        if cost > *remaining {
            return Ok(None);
        }
        next_budget.insert(dimension.clone(), *remaining - cost);
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
        .ok_or_else(|| SequentialOracleError::UnknownProtocolState(protocol_state_id.to_owned()))?;
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
        if action_worst_case_loss < best.loss {
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
    let current_minimax_loss = terminal_minimax(problem, &possible_states)?.0;
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
    let terminal_resource_paths =
        terminal_resource_paths(problem, &solved.policy, &possible_states)?;
    Ok(SequentialPolicyValue {
        current_minimax_loss,
        current_risk_gate_passed: current_minimax_loss
            <= problem.maximum_worst_case_loss + VALUE_TOLERANCE,
        worst_case_terminal_loss: solved.loss,
        policy_terminal_risk_gate_passed: solved.loss
            <= problem.maximum_worst_case_loss + VALUE_TOLERANCE,
        terminal_resource_paths,
        explored_nodes: nodes,
        policy: solved.policy,
    })
}

fn terminal_resource_paths(
    problem: &SequentialRobustProblem,
    policy: &SequentialPolicyNode,
    possible_states: &[String],
) -> Result<Vec<SequentialTerminalResourcePath>, SequentialOracleError> {
    match policy {
        SequentialPolicyNode::Stop {
            decision_id,
            worst_case_terminal_loss,
        } => Ok(vec![SequentialTerminalResourcePath {
            steps: Vec::new(),
            terminal_possible_state_ids: possible_states.to_vec(),
            terminal_decision_id: decision_id.clone(),
            terminal_worst_case_loss: *worst_case_terminal_loss,
            resource_usage: problem
                .resource_budget
                .keys()
                .map(|dimension| (dimension.clone(), 0))
                .collect(),
        }]),
        SequentialPolicyNode::Probe {
            action_id,
            branches,
        } => {
            let action = problem
                .actions
                .iter()
                .find(|candidate| candidate.action_id == *action_id)
                .ok_or_else(|| SequentialOracleError::UnknownAction(action_id.clone()))?;
            let mut paths = Vec::new();

            for branch in branches {
                let child_paths = terminal_resource_paths(
                    problem,
                    &branch.continuation,
                    &branch.possible_state_ids,
                )?;
                for mut child in child_paths {
                    let mut steps = Vec::with_capacity(child.steps.len() + 1);
                    steps.push(SequentialResourceStep {
                        action_id: action_id.clone(),
                        outcome_id: branch.outcome_id.clone(),
                        possible_state_ids: branch.possible_state_ids.clone(),
                    });
                    steps.append(&mut child.steps);
                    child.steps = steps;

                    for (dimension, cost) in &action.resource_costs {
                        let used = child.resource_usage.get(dimension).copied().unwrap_or(0);
                        let total = used.checked_add(*cost).ok_or_else(|| {
                            SequentialOracleError::ResourceOverflow {
                                action_id: action_id.clone(),
                                dimension: dimension.clone(),
                            }
                        })?;
                        child.resource_usage.insert(dimension.clone(), total);
                    }
                    paths.push(child);
                    if paths.len() > MAX_SEARCH_NODES {
                        return Err(SequentialOracleError::TooManyTerminalResourcePaths(
                            paths.len(),
                        ));
                    }
                }
            }
            Ok(paths)
        }
    }
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

// These two transition helpers deliberately duplicate the small
// feasibility/support calculations used by the dynamic programme. The
// exhaustive oracle shares the validated problem schema, but not the dynamic
// programme's resource or outcome-intersection implementation.

fn enumeration_feasible_action(
    problem: &SequentialRobustProblem,
    action: &SequentialAction,
    remaining_budget: &BTreeMap<String, u64>,
    current_step: u64,
) -> Result<Option<(BTreeMap<String, u64>, u64)>, SequentialOracleError> {
    let completed_at = current_step
        .checked_add(action.result_delay_steps)
        .ok_or(SequentialOracleError::TimeOverflow)?;
    if completed_at > problem.decision_deadline_step {
        return Ok(None);
    }

    let mut budget_after = BTreeMap::new();
    for (dimension, available) in remaining_budget {
        let Some(required) = action.resource_costs.get(dimension) else {
            return Err(SequentialOracleError::ResourceDimensionMismatch {
                action_id: action.action_id.clone(),
                expected: remaining_budget.keys().cloned().collect(),
                actual: action.resource_costs.keys().cloned().collect(),
            });
        };
        if required > available {
            return Ok(None);
        }
        budget_after.insert(dimension.clone(), *available - *required);
    }
    Ok(Some((budget_after, completed_at)))
}

fn enumeration_active_branches(
    action: &SequentialAction,
    possible_states: &[String],
) -> Result<Vec<(String, Vec<String>, String)>, SequentialOracleError> {
    let mut branches = Vec::new();
    let mut states_accounted_for = BTreeSet::new();

    for outcome in &action.outcomes {
        let mut consistent = Vec::new();
        for current_state in possible_states {
            if outcome
                .possible_state_ids
                .iter()
                .any(|id| id == current_state)
            {
                consistent.push(current_state.clone());
                states_accounted_for.insert(current_state.clone());
            }
        }
        if !consistent.is_empty() {
            consistent.sort();
            branches.push((
                outcome.outcome_id.clone(),
                consistent,
                outcome.next_protocol_state_id.clone(),
            ));
        }
    }

    for current_state in possible_states {
        if !states_accounted_for.contains(current_state) {
            return Err(SequentialOracleError::UncoveredState {
                action_id: action.action_id.clone(),
                state_id: current_state.clone(),
            });
        }
    }
    Ok(branches)
}

/// Independently enumerate every legal contingent policy tree for a tiny
/// world. This does not call the dynamic-programming solver, its terminal
/// value helper, or its resource/support transition helpers. It is bounded
/// and exists only to cross-check known-answer worlds.
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
    let mut ordered_decisions = problem.decision_ids.clone();
    ordered_decisions.sort();
    for decision_id in &ordered_decisions {
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
            worst = worst.max(loss);
        }
        let is_better = match &stop_decision {
            None => true,
            Some(current_id) => {
                worst < stop_value || (worst == stop_value && decision_id < current_id)
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
        .ok_or_else(|| SequentialOracleError::UnknownProtocolState(protocol_state_id.to_owned()))?;
    let mut legal = protocol.legal_action_ids.clone();
    legal.sort();

    for action_id in legal {
        let action = problem
            .actions
            .iter()
            .find(|candidate| candidate.action_id == action_id)
            .ok_or_else(|| SequentialOracleError::UnknownAction(action_id.clone()))?;
        let Some((budget_after, step_after)) =
            enumeration_feasible_action(problem, action, remaining_budget, current_step)?
        else {
            continue;
        };
        let active = enumeration_active_branches(action, possible_states)?;

        // Cartesian product of all branch-continuation choices constructs
        // complete policies. Each branch sees only its own public outcome.
        let mut combinations: Vec<(Vec<SequentialPolicyBranch>, f64)> = vec![(Vec::new(), 0.0)];
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
                    next_combinations.push((branches, partial_worst_loss.max(child.loss)));
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
    let current_minimax_loss = {
        let mut best = f64::INFINITY;
        for decision_id in &problem.decision_ids {
            let mut worst = 0.0_f64;
            for state_id in &possible_states {
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
            if worst < best {
                best = worst;
            }
        }
        best
    };
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
            Some(current) => candidate.loss < current.loss,
        };
        if replace {
            best = Some(candidate);
        }
    }
    let best = best.ok_or(SequentialOracleError::NoPolicy)?;
    let terminal_resource_paths = terminal_resource_paths(problem, &best.policy, &possible_states)?;
    Ok(SequentialPolicyValue {
        current_minimax_loss,
        current_risk_gate_passed: current_minimax_loss
            <= problem.maximum_worst_case_loss + VALUE_TOLERANCE,
        worst_case_terminal_loss: best.loss,
        policy_terminal_risk_gate_passed: best.loss
            <= problem.maximum_worst_case_loss + VALUE_TOLERANCE,
        terminal_resource_paths,
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
            decision_loss: BTreeMap::from([("d0".to_owned(), loss_d0), ("d1".to_owned(), loss_d1)]),
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

    struct TinyRng(u64);

    impl TinyRng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        fn range(&mut self, upper: usize) -> usize {
            (self.next() % upper as u64) as usize
        }

        fn unit(&mut self) -> f64 {
            (self.next() % 1_001) as f64 / 1_000.0
        }
    }

    fn random_support_partition(
        states: &[String],
        max_groups: usize,
        rng: &mut TinyRng,
    ) -> Vec<Vec<String>> {
        let group_count = 1 + rng.range(max_groups.min(states.len()));
        let mut groups = vec![Vec::new(); group_count];

        // Seed each group with one state so no outcome is empty.
        for (index, state_id) in states.iter().enumerate() {
            groups[index % group_count].push(state_id.clone());
            if rng.range(3) == 0 {
                let other = rng.range(group_count);
                if !groups[other].contains(state_id) {
                    groups[other].push(state_id.clone());
                }
            }
        }

        // Add further overlaps without ever removing coverage.
        for state_id in states {
            for group in &mut groups {
                if rng.range(5) == 0 && !group.contains(state_id) {
                    group.push(state_id.clone());
                }
            }
        }
        groups
    }

    fn generated_tiny_world(rng: &mut TinyRng) -> SequentialRobustProblem {
        let state_count = 2 + rng.range(3);
        let state_ids: Vec<String> = (0..state_count).map(|i| format!("s{i}")).collect();
        let decision_count = 2 + rng.range(2);
        let decision_ids: Vec<String> = (0..decision_count).map(|i| format!("d{i}")).collect();

        let mut decision_loss = BTreeMap::new();
        for decision_id in &decision_ids {
            let losses = state_ids
                .iter()
                .map(|state_id| (state_id.clone(), rng.unit()))
                .collect();
            decision_loss.insert(decision_id.clone(), losses);
        }

        let resource_budget = BTreeMap::from([
            ("compute_units".to_owned(), 1),
            ("experiment_units".to_owned(), 2),
        ]);
        let action_costs = |rng: &mut TinyRng| {
            BTreeMap::from([
                ("compute_units".to_owned(), rng.range(2) as u64),
                ("experiment_units".to_owned(), 1),
            ])
        };

        let mut protocol_states = Vec::new();
        let mut actions = Vec::new();
        let mut root_action_ids = Vec::new();

        for root_index in 0..2 {
            let root_action_id = format!("A{root_index}");
            root_action_ids.push(root_action_id.clone());
            let root_groups = random_support_partition(&state_ids, 3, rng);
            let mut root_outcomes = Vec::new();

            for (branch_index, branch_support) in root_groups.iter().enumerate() {
                let branch_protocol = format!("after_{root_index}_{branch_index}");
                let child_action_id = format!("C{root_index}_{branch_index}");
                protocol_states.push(SequentialProtocolState {
                    protocol_state_id: branch_protocol.clone(),
                    legal_action_ids: vec![child_action_id.clone()],
                });

                let child_groups = random_support_partition(branch_support, 3, rng);
                let mut child_outcomes = Vec::new();
                for (child_index, child_support) in child_groups.into_iter().enumerate() {
                    let terminal_protocol =
                        format!("terminal_{root_index}_{branch_index}_{child_index}");
                    protocol_states.push(SequentialProtocolState {
                        protocol_state_id: terminal_protocol.clone(),
                        legal_action_ids: Vec::new(),
                    });
                    child_outcomes.push(SequentialOutcome {
                        outcome_id: format!("c{child_index}"),
                        possible_state_ids: child_support,
                        next_protocol_state_id: terminal_protocol,
                    });
                }

                actions.push(SequentialAction {
                    action_id: child_action_id,
                    resource_costs: action_costs(rng),
                    result_delay_steps: 1,
                    outcomes: child_outcomes,
                });
                root_outcomes.push(SequentialOutcome {
                    outcome_id: format!("o{branch_index}"),
                    possible_state_ids: branch_support.clone(),
                    next_protocol_state_id: branch_protocol,
                });
            }

            actions.push(SequentialAction {
                action_id: root_action_id,
                resource_costs: action_costs(rng),
                result_delay_steps: 1,
                outcomes: root_outcomes,
            });
        }

        protocol_states.push(SequentialProtocolState {
            protocol_state_id: "root".to_owned(),
            legal_action_ids: root_action_ids,
        });

        SequentialRobustProblem {
            state_ids,
            decision_ids,
            decision_loss,
            maximum_worst_case_loss: rng.unit(),
            initial_protocol_state_id: "root".to_owned(),
            protocol_states,
            actions,
            resource_budget,
            current_step: 0,
            decision_deadline_step: 1 + rng.range(2) as u64,
            maximum_actions: 1 + rng.range(2) as u8,
        }
    }

    #[test]
    fn sub_tolerance_policy_improvement_is_not_discarded() {
        let delta = 1.0e-13;
        let problem = SequentialRobustProblem {
            state_ids: vec!["s0".to_owned(), "s1".to_owned()],
            decision_ids: vec!["d0".to_owned(), "d1".to_owned()],
            decision_loss: BTreeMap::from([
                (
                    "d0".to_owned(),
                    BTreeMap::from([
                        ("s0".to_owned(), 0.5 - delta),
                        ("s1".to_owned(), 0.5 + delta),
                    ]),
                ),
                (
                    "d1".to_owned(),
                    BTreeMap::from([
                        ("s0".to_owned(), 0.5 + delta),
                        ("s1".to_owned(), 0.5 - delta),
                    ]),
                ),
            ]),
            maximum_worst_case_loss: 0.0,
            initial_protocol_state_id: "root".to_owned(),
            protocol_states: vec![
                SequentialProtocolState {
                    protocol_state_id: "root".to_owned(),
                    legal_action_ids: vec!["observe".to_owned()],
                },
                SequentialProtocolState {
                    protocol_state_id: "terminal_s0".to_owned(),
                    legal_action_ids: Vec::new(),
                },
                SequentialProtocolState {
                    protocol_state_id: "terminal_s1".to_owned(),
                    legal_action_ids: Vec::new(),
                },
            ],
            actions: vec![SequentialAction {
                action_id: "observe".to_owned(),
                resource_costs: cost(1),
                result_delay_steps: 1,
                outcomes: vec![
                    SequentialOutcome {
                        outcome_id: "s0".to_owned(),
                        possible_state_ids: vec!["s0".to_owned()],
                        next_protocol_state_id: "terminal_s0".to_owned(),
                    },
                    SequentialOutcome {
                        outcome_id: "s1".to_owned(),
                        possible_state_ids: vec!["s1".to_owned()],
                        next_protocol_state_id: "terminal_s1".to_owned(),
                    },
                ],
            }],
            resource_budget: cost(1),
            current_step: 0,
            decision_deadline_step: 1,
            maximum_actions: 1,
        };

        let dynamic = evaluate_sequential_robust(&problem).unwrap();
        let exhaustive = enumerate_sequential_robust(&problem).unwrap();
        assert!(dynamic.worst_case_terminal_loss < dynamic.current_minimax_loss);
        assert!(dynamic.current_minimax_loss - dynamic.worst_case_terminal_loss < VALUE_TOLERANCE);
        assert_eq!(
            dynamic.worst_case_terminal_loss,
            exhaustive.worst_case_terminal_loss
        );
        assert!(matches!(dynamic.policy, SequentialPolicyNode::Probe { .. }));
        assert!(matches!(
            exhaustive.policy,
            SequentialPolicyNode::Probe { .. }
        ));
    }

    #[test]
    fn generated_tiny_worlds_match_independent_exhaustive_policy_tree_oracle() {
        let mut rng = TinyRng(0x5a17_c3d1);
        for case_index in 0..96 {
            let problem = generated_tiny_world(&mut rng);
            let dynamic = evaluate_sequential_robust(&problem).unwrap_or_else(|error| {
                panic!("dynamic oracle failed on generated case {case_index}: {error}")
            });
            let exhaustive = enumerate_sequential_robust(&problem).unwrap_or_else(|error| {
                panic!("exhaustive oracle failed on generated case {case_index}: {error}")
            });

            assert_eq!(
                dynamic.current_minimax_loss, exhaustive.current_minimax_loss,
                "current minimax mismatch on generated case {case_index}"
            );
            assert_eq!(
                dynamic.current_risk_gate_passed, exhaustive.current_risk_gate_passed,
                "current risk-gate mismatch on generated case {case_index}"
            );
            assert_eq!(
                dynamic.worst_case_terminal_loss, exhaustive.worst_case_terminal_loss,
                "terminal value mismatch on generated case {case_index}"
            );
            assert_eq!(
                dynamic.policy_terminal_risk_gate_passed,
                exhaustive.policy_terminal_risk_gate_passed,
                "policy-terminal risk-gate mismatch on generated case {case_index}"
            );
        }
    }

    #[test]
    fn dynamic_and_exhaustive_oracles_use_same_lexicographic_decision_tie_break() {
        let problem = SequentialRobustProblem {
            state_ids: vec!["s0".to_owned()],
            // Deliberately reverse the lexical order.
            decision_ids: vec!["z_last".to_owned(), "a_first".to_owned()],
            decision_loss: BTreeMap::from([
                (
                    "z_last".to_owned(),
                    BTreeMap::from([("s0".to_owned(), 0.25)]),
                ),
                (
                    "a_first".to_owned(),
                    BTreeMap::from([("s0".to_owned(), 0.25)]),
                ),
            ]),
            maximum_worst_case_loss: 0.5,
            initial_protocol_state_id: "root".to_owned(),
            protocol_states: vec![SequentialProtocolState {
                protocol_state_id: "root".to_owned(),
                legal_action_ids: Vec::new(),
            }],
            actions: Vec::new(),
            resource_budget: cost(0),
            current_step: 0,
            decision_deadline_step: 0,
            maximum_actions: 0,
        };
        let dynamic = evaluate_sequential_robust(&problem).unwrap();
        let exhaustive = enumerate_sequential_robust(&problem).unwrap();

        fn selected(policy: &SequentialPolicyNode) -> &str {
            match policy {
                SequentialPolicyNode::Stop { decision_id, .. } => decision_id.as_str(),
                SequentialPolicyNode::Probe { .. } => panic!("unexpected probe"),
            }
        }
        assert_eq!(selected(&dynamic.policy), "a_first");
        assert_eq!(selected(&exhaustive.policy), "a_first");
    }

    #[test]
    fn dynamic_programme_finds_non_myopic_branch_unlock_policy() {
        let problem = branch_unlock_world();
        let value = evaluate_sequential_robust(&problem).unwrap();
        assert_eq!(value.worst_case_terminal_loss, 0.0);
        assert!((value.current_minimax_loss - 1.0).abs() < VALUE_TOLERANCE);
        assert!(!value.current_risk_gate_passed);
        assert!(value.policy_terminal_risk_gate_passed);
        assert_eq!(root_action(&value.policy), Some("B"));

        let SequentialPolicyNode::Probe { branches, .. } = value.policy else {
            panic!("expected a probing root policy");
        };
        assert_eq!(branches.len(), 2);
        for branch in branches {
            let expected = if branch.outcome_id == "z0" {
                "C0"
            } else {
                "C1"
            };
            assert_eq!(root_action(&branch.continuation), Some(expected));
        }
    }

    #[test]
    fn terminal_resource_paths_report_exact_branch_costs() {
        let mut problem = branch_unlock_world();
        problem
            .resource_budget
            .insert("compute_units".to_owned(), 1);
        for action in &mut problem.actions {
            action.resource_costs.insert("compute_units".to_owned(), 0);
        }
        problem
            .actions
            .iter_mut()
            .find(|action| action.action_id == "C1")
            .unwrap()
            .resource_costs
            .insert("compute_units".to_owned(), 1);

        let dynamic = evaluate_sequential_robust(&problem).unwrap();
        let exhaustive = enumerate_sequential_robust(&problem).unwrap();

        assert_eq!(dynamic.terminal_resource_paths.len(), 4);
        assert_eq!(
            dynamic.terminal_resource_paths,
            exhaustive.terminal_resource_paths
        );
        for path in &dynamic.terminal_resource_paths {
            assert_eq!(path.steps.len(), 2);
            assert_eq!(path.steps[0].action_id, "B");
            assert_eq!(path.steps[0].possible_state_ids.len(), 8);
            assert!(path.steps[1].action_id == "C0" || path.steps[1].action_id == "C1");
            assert_eq!(path.steps[1].possible_state_ids.len(), 4);
            assert_eq!(path.terminal_possible_state_ids.len(), 4);
            assert_eq!(path.resource_usage["experiment_units"], 2);
            let expected_compute = if path.steps[1].action_id == "C0" {
                0
            } else {
                1
            };
            assert_eq!(path.resource_usage["compute_units"], expected_compute);

            let terminal_x = path.terminal_possible_state_ids[0]
                .strip_prefix('x')
                .unwrap()
                .chars()
                .next()
                .unwrap();
            assert!(
                path.terminal_possible_state_ids
                    .iter()
                    .all(|state| { state.starts_with(&format!("x{terminal_x}_")) })
            );
            assert_eq!(path.terminal_decision_id, format!("d{terminal_x}"));
            assert_eq!(path.terminal_worst_case_loss, 0.0);
        }
    }

    #[test]
    fn dynamic_programme_matches_independent_exhaustive_policy_tree_oracle() {
        let problem = branch_unlock_world();
        let dynamic = evaluate_sequential_robust(&problem).unwrap();
        let exhaustive = enumerate_sequential_robust(&problem).unwrap();
        assert_eq!(
            dynamic.current_minimax_loss,
            exhaustive.current_minimax_loss
        );
        assert_eq!(
            dynamic.current_risk_gate_passed,
            exhaustive.current_risk_gate_passed
        );
        assert_eq!(
            dynamic.worst_case_terminal_loss,
            exhaustive.worst_case_terminal_loss
        );
        assert!(!dynamic.current_risk_gate_passed);
        assert!(dynamic.policy_terminal_risk_gate_passed);
        assert!(exhaustive.policy_terminal_risk_gate_passed);
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
                (
                    "d0".to_owned(),
                    BTreeMap::from([("s0".to_owned(), 0.0), ("s1".to_owned(), 1.0)]),
                ),
                (
                    "d1".to_owned(),
                    BTreeMap::from([("s0".to_owned(), 1.0), ("s1".to_owned(), 0.0)]),
                ),
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
        let exhaustive = enumerate_sequential_robust(&problem).unwrap();
        assert_eq!(exhaustive.worst_case_terminal_loss, 1.0);
        assert!(matches!(
            exhaustive.policy,
            SequentialPolicyNode::Stop { .. }
        ));
    }

    #[test]
    fn an_action_that_misses_the_deadline_is_not_selected() {
        let mut problem = branch_unlock_world();
        problem.decision_deadline_step = 0;
        let value = evaluate_sequential_robust(&problem).unwrap();
        assert_eq!(value.worst_case_terminal_loss, 1.0);
        assert!(matches!(value.policy, SequentialPolicyNode::Stop { .. }));
        let exhaustive = enumerate_sequential_robust(&problem).unwrap();
        assert_eq!(exhaustive.worst_case_terminal_loss, 1.0);
        assert!(matches!(
            exhaustive.policy,
            SequentialPolicyNode::Stop { .. }
        ));
    }

    #[test]
    fn an_action_with_incomplete_outcome_support_fails_closed() {
        let mut problem = branch_unlock_world();
        let action = problem
            .actions
            .iter_mut()
            .find(|action| action.action_id == "B")
            .unwrap();
        action.outcomes[0].possible_state_ids.pop();
        assert!(matches!(
            evaluate_sequential_robust(&problem),
            Err(SequentialOracleError::UncoveredState { .. })
        ));
        assert!(matches!(
            enumerate_sequential_robust(&problem),
            Err(SequentialOracleError::UncoveredState { .. })
        ));
    }

    #[test]
    fn protocol_permissions_exclude_otherwise_useful_actions() {
        let mut problem = branch_unlock_world();
        let root = problem
            .protocol_states
            .iter_mut()
            .find(|state| state.protocol_state_id == "root")
            .unwrap();
        root.legal_action_ids = vec!["A".to_owned()];
        let value = evaluate_sequential_robust(&problem).unwrap();
        assert_eq!(value.worst_case_terminal_loss, 1.0);
        assert!(matches!(value.policy, SequentialPolicyNode::Stop { .. }));
        assert!(!value.current_risk_gate_passed);
        assert!(!value.policy_terminal_risk_gate_passed);
        let exhaustive = enumerate_sequential_robust(&problem).unwrap();
        assert_eq!(exhaustive.worst_case_terminal_loss, 1.0);
    }

    #[test]
    fn cumulative_sequence_cost_is_enforced_across_branches() {
        let mut problem = branch_unlock_world();
        // The first branch probe fits, but its required follow-up does not.
        problem.resource_budget = cost(1);
        let dynamic = evaluate_sequential_robust(&problem).unwrap();
        let exhaustive = enumerate_sequential_robust(&problem).unwrap();
        assert_eq!(dynamic.worst_case_terminal_loss, 1.0);
        assert_eq!(exhaustive.worst_case_terminal_loss, 1.0);
        assert!(matches!(dynamic.policy, SequentialPolicyNode::Stop { .. }));
        assert!(matches!(
            exhaustive.policy,
            SequentialPolicyNode::Stop { .. }
        ));
    }

    #[test]
    fn follow_up_that_arrives_after_deadline_is_not_selected() {
        let mut problem = branch_unlock_world();
        // B arrives at step 1. Its C follow-up would arrive at step 2.
        problem.decision_deadline_step = 1;
        let dynamic = evaluate_sequential_robust(&problem).unwrap();
        let exhaustive = enumerate_sequential_robust(&problem).unwrap();
        assert_eq!(dynamic.worst_case_terminal_loss, 1.0);
        assert_eq!(exhaustive.worst_case_terminal_loss, 1.0);
        assert!(matches!(dynamic.policy, SequentialPolicyNode::Stop { .. }));
        assert!(matches!(
            exhaustive.policy,
            SequentialPolicyNode::Stop { .. }
        ));
    }

    #[test]
    fn every_resource_dimension_is_enforced_component_wise() {
        let mut problem = branch_unlock_world();
        problem
            .resource_budget
            .insert("compute_units".to_owned(), 1);
        for action in &mut problem.actions {
            action.resource_costs.insert("compute_units".to_owned(), 0);
        }
        problem.actions[1]
            .resource_costs
            .insert("compute_units".to_owned(), 2);
        let value = evaluate_sequential_robust(&problem).unwrap();
        assert_eq!(value.worst_case_terminal_loss, 1.0);
        assert!(matches!(value.policy, SequentialPolicyNode::Stop { .. }));
        let exhaustive = enumerate_sequential_robust(&problem).unwrap();
        assert_eq!(exhaustive.worst_case_terminal_loss, 1.0);
        assert!(matches!(
            exhaustive.policy,
            SequentialPolicyNode::Stop { .. }
        ));
    }
}
