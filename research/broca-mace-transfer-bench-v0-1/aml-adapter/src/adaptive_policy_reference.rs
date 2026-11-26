// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent exhaustive reference evaluator for small adaptive evidence-acquisition worlds.
//!
//! This module intentionally uses a finite joint-world table instead of the
//! production candidate's marginal likelihood calculations. A world fixes the
//! outcomes of every query, so repeated queries over a shared source root are
//! correlated by construction. The evaluator is an oracle for tiny fixtures,
//! not a production policy executor or a learned retrieval controller.

use std::collections::{BTreeMap, BTreeSet};

const MAX_STATES: usize = 8;
const MAX_WORLDS: usize = 64;
const MAX_ACTIONS: usize = 6;
const MAX_QUERIES: usize = 8;
const MAX_HORIZON: usize = 4;
const MAX_ID_BYTES: usize = 128;
const MAX_VISITED_HISTORIES: usize = 50_000;
const MAX_UTILITY_ABS: i64 = 1_000_000_000_000;
const PROBABILITY_SCALE: u64 = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReferenceRejectV1 {
    EmptyScenario,
    InvalidIdentifier,
    DuplicateIdentifier,
    InvalidDimensions,
    InvalidProbabilityMass,
    InvalidQueryObservation,
    InvalidSourceRoots,
    InvalidActionTable,
    NoPossibleWorlds,
    NoPermittedAction,
    BudgetExceeded,
    ArithmeticOverflow,
    SearchLimitExceeded,
    PolicyTooLarge,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceWorldV1 {
    pub world_id: String,
    pub state_id: String,
    /// Expected-value mass in parts per million. Robust mode ignores this field.
    /// Robust-only fixtures may set it to None.
    pub probability_ppm: Option<u32>,
    /// Whether this world belongs to the conservative support set.
    pub possible: bool,
    /// One deterministic observation for each query in this latent world.
    pub observation_by_query: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceActionV1 {
    pub action_id: String,
    /// Consumer-owned utility for each state.
    pub utility_by_state: BTreeMap<String, i64>,
    /// Consumer-owned hard admissibility for each state.
    pub allowed_by_state: BTreeMap<String, bool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceQueryV1 {
    pub query_id: String,
    /// Queries with overlapping roots are not presumed independent.
    pub source_root_ids: Vec<String>,
    pub tool_calls: u32,
    pub latency_ns: u64,
    pub context_tokens: u64,
    pub inspected_bytes: u64,
    pub cost_utility_units: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceBudgetsV1 {
    pub maximum_queries: u32,
    pub maximum_tool_calls: u32,
    pub maximum_latency_ns: u64,
    pub maximum_context_tokens: u64,
    pub maximum_inspected_bytes: u64,
    pub maximum_cost_utility_units: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceScenarioV1 {
    pub state_ids: Vec<String>,
    /// World IDs are unique; each row is a joint outcome assignment.
    pub worlds: Vec<ReferenceWorldV1>,
    /// Action IDs are unique; input enumeration order does not affect ties.
    pub actions: Vec<ReferenceActionV1>,
    /// Query IDs are unique; input enumeration order does not affect ties.
    pub queries: Vec<ReferenceQueryV1>,
    pub maximum_horizon: u32,
    pub maximum_policy_nodes: usize,
    pub budgets: ReferenceBudgetsV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferencePolicyBranchV1 {
    pub observation_id: String,
    pub next: Box<ReferencePolicyNodeV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReferencePolicyNodeV1 {
    Terminal {
        action_id: String,
    },
    Query {
        query_id: String,
        branches: Vec<ReferencePolicyBranchV1>,
    },
}

impl ReferencePolicyNodeV1 {
    pub fn node_count(&self) -> usize {
        match self {
            Self::Terminal { .. } => 1,
            Self::Query { branches, .. } => {
                1 + branches.iter().map(|branch| branch.next.node_count()).sum::<usize>()
            }
        }
    }

    fn canonical_key(&self) -> String {
        match self {
            Self::Terminal { action_id } => format!("0:T:{action_id}"),
            Self::Query { query_id, branches } => {
                let encoded = branches.iter()
                    .map(|branch| format!("{}=>{}", branch.observation_id, branch.next.canonical_key()))
                    .collect::<Vec<_>>()
                    .join("|");
                format!("1:Q:{query_id}[{encoded}]")
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedReferenceResultV1 {
    pub policy: ReferencePolicyNodeV1,
    /// Numerator over PROBABILITY_SCALE; do not round before comparisons.
    pub expected_utility_numerator_ppm: i128,
    /// Expected acquisition cost, numerator over PROBABILITY_SCALE.
    pub expected_cost_numerator_ppm: i128,
    /// Expected net utility, numerator over PROBABILITY_SCALE.
    pub expected_net_utility_numerator_ppm: i128,
    pub explored_histories: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RobustReferenceResultV1 {
    pub policy: ReferencePolicyNodeV1,
    /// Maximum state-wise regret plus query cost over all possible worlds.
    pub worst_case_regret_utility_units: i128,
    pub worst_case_acquisition_cost_utility_units: u64,
    pub explored_histories: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SpentV1 {
    queries: u32,
    tool_calls: u32,
    latency_ns: u64,
    context_tokens: u64,
    inspected_bytes: u64,
    cost_utility_units: u64,
}

impl SpentV1 {
    fn add_query(
        self,
        query: &ReferenceQueryV1,
        budgets: &ReferenceBudgetsV1,
    ) -> Result<Option<Self>, ReferenceRejectV1> {
        let next = Self {
            queries: self.queries.checked_add(1).ok_or(ReferenceRejectV1::ArithmeticOverflow)?,
            tool_calls: self.tool_calls.checked_add(query.tool_calls)
                .ok_or(ReferenceRejectV1::ArithmeticOverflow)?,
            latency_ns: self.latency_ns.checked_add(query.latency_ns)
                .ok_or(ReferenceRejectV1::ArithmeticOverflow)?,
            context_tokens: self.context_tokens.checked_add(query.context_tokens)
                .ok_or(ReferenceRejectV1::ArithmeticOverflow)?,
            inspected_bytes: self.inspected_bytes.checked_add(query.inspected_bytes)
                .ok_or(ReferenceRejectV1::ArithmeticOverflow)?,
            cost_utility_units: self.cost_utility_units.checked_add(query.cost_utility_units)
                .ok_or(ReferenceRejectV1::ArithmeticOverflow)?,
        };
        let within = next.queries <= budgets.maximum_queries
            && next.tool_calls <= budgets.maximum_tool_calls
            && next.latency_ns <= budgets.maximum_latency_ns
            && next.context_tokens <= budgets.maximum_context_tokens
            && next.inspected_bytes <= budgets.maximum_inspected_bytes
            && next.cost_utility_units <= budgets.maximum_cost_utility_units;
        Ok(within.then_some(next))
    }
}

#[derive(Clone, Debug)]
struct ExpectedCandidateV1 {
    policy: ReferencePolicyNodeV1,
    utility_numerator_ppm: i128,
    cost_numerator_ppm: i128,
}

impl ExpectedCandidateV1 {
    fn net(&self) -> i128 {
        self.utility_numerator_ppm - self.cost_numerator_ppm
    }
}

#[derive(Clone, Debug)]
struct RobustCandidateV1 {
    policy: ReferencePolicyNodeV1,
    worst_regret: i128,
    worst_cost: u64,
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= MAX_ID_BYTES
        && !value.chars().any(char::is_control)
}

fn sorted_unique(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn unique_ids<'a>(values: impl IntoIterator<Item = &'a str>) -> bool {
    let mut seen = BTreeSet::new();
    values.into_iter().all(|value| valid_id(value) && seen.insert(value))
}

impl ReferenceScenarioV1 {
    pub fn validate(&self, expected_value_mode: bool) -> Result<(), ReferenceRejectV1> {
        if self.state_ids.is_empty()
            || self.state_ids.len() > MAX_STATES
            || self.worlds.is_empty()
            || self.worlds.len() > MAX_WORLDS
            || self.actions.is_empty()
            || self.actions.len() > MAX_ACTIONS
            || self.queries.is_empty()
            || self.queries.len() > MAX_QUERIES
            || self.maximum_horizon == 0
            || self.maximum_horizon > MAX_HORIZON as u32
            || self.maximum_policy_nodes < 1
            || self.budgets.maximum_queries > MAX_QUERIES as u32
            || self.budgets.maximum_tool_calls > (MAX_QUERIES * 4) as u32
        {
            return Err(ReferenceRejectV1::EmptyScenario);
        }
        if !sorted_unique(&self.state_ids)
            || self.state_ids.iter().any(|value| !valid_id(value))
            || !unique_ids(self.worlds.iter().map(|world| world.world_id.as_str()))
            || self.worlds.iter().any(|world| !valid_id(&world.state_id))
            || !unique_ids(self.actions.iter().map(|action| action.action_id.as_str()))
            || !unique_ids(self.queries.iter().map(|query| query.query_id.as_str()))
        {
            return Err(ReferenceRejectV1::InvalidIdentifier);
        }
        if !self.worlds.iter().any(|world| world.possible) {
            return Err(ReferenceRejectV1::NoPossibleWorlds);
        }
        for world in &self.worlds {
            if self.state_ids.binary_search(&world.state_id).is_err() {
                return Err(ReferenceRejectV1::InvalidDimensions);
            }
            if world.observation_by_query.len() != self.queries.len()
                || self.queries.iter().any(|query| {
                    world.observation_by_query.get(&query.query_id).is_none_or(|value| !valid_id(value))
                })
            {
                return Err(ReferenceRejectV1::InvalidQueryObservation);
            }
            if expected_value_mode {
                match (world.possible, world.probability_ppm) {
                    (true, Some(value)) if value <= PROBABILITY_SCALE as u32 => {}
                    (false, None | Some(0)) => {}
                    _ => return Err(ReferenceRejectV1::InvalidProbabilityMass),
                }
            }
        }
        for query in &self.queries {
            if query.source_root_ids.is_empty()
                || !sorted_unique(&query.source_root_ids)
                || query.source_root_ids.iter().any(|root| !valid_id(root))
                || query.tool_calls == 0
            {
                return Err(ReferenceRejectV1::InvalidSourceRoots);
            }
        }
        for action in &self.actions {
            if action.utility_by_state.len() != self.state_ids.len()
                || action.allowed_by_state.len() != self.state_ids.len()
                || self.state_ids.iter().any(|state| {
                    !action.utility_by_state.contains_key(state)
                        || !action.allowed_by_state.contains_key(state)
                })
                || action.utility_by_state.values().any(|value| value.unsigned_abs() > MAX_UTILITY_ABS as u64)
            {
                return Err(ReferenceRejectV1::InvalidActionTable);
            }
        }
        if expected_value_mode {
            let mass = self.worlds.iter()
                .filter(|world| world.possible)
                .try_fold(0_u64, |sum, world| {
                    sum.checked_add(u64::from(world.probability_ppm.unwrap_or(0)))
                })
                .ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
            if mass != PROBABILITY_SCALE {
                return Err(ReferenceRejectV1::InvalidProbabilityMass);
            }
        }
        // Every possible state needs at least one action that is admissible in it.
        for state in &self.state_ids {
            if self.worlds.iter().any(|world| world.possible && world.state_id == *state)
                && !self.actions.iter().any(|action| action.allowed_by_state[state])
            {
                return Err(ReferenceRejectV1::NoPermittedAction);
            }
        }
        Ok(())
    }
}

pub fn evaluate_expected_reference_v1(
    scenario: &ReferenceScenarioV1,
) -> Result<ExpectedReferenceResultV1, ReferenceRejectV1> {
    scenario.validate(true)?;
    let worlds = scenario.worlds.iter().enumerate()
        .filter(|(_, world)| world.possible && world.probability_ppm.unwrap_or(0) > 0)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut visited = 0usize;
    let best = search_expected(
        scenario,
        &worlds,
        SpentV1::default(),
        0,
        &BTreeSet::new(),
        &mut visited,
    )?.ok_or(ReferenceRejectV1::NoPermittedAction)?;
    if best.policy.node_count() > scenario.maximum_policy_nodes {
        return Err(ReferenceRejectV1::PolicyTooLarge);
    }
    Ok(ExpectedReferenceResultV1 {
        expected_utility_numerator_ppm: best.utility_numerator_ppm,
        expected_cost_numerator_ppm: best.cost_numerator_ppm,
        expected_net_utility_numerator_ppm: best.net(),
        policy: best.policy,
        explored_histories: visited,
    })
}

fn search_expected(
    scenario: &ReferenceScenarioV1,
    world_indices: &[usize],
    spent: SpentV1,
    depth: u32,
    queried: &BTreeSet<String>,
    visited: &mut usize,
) -> Result<Option<ExpectedCandidateV1>, ReferenceRejectV1> {
    *visited = visited.checked_add(1).ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
    if *visited > MAX_VISITED_HISTORIES {
        return Err(ReferenceRejectV1::SearchLimitExceeded);
    }
    let mut best: Option<ExpectedCandidateV1> = None;

    for action in &scenario.actions {
        if !action_allowed_for_worlds(action, scenario, world_indices) {
            continue;
        }
        let mut utility_numerator = 0_i128;
        let mut cost_numerator = 0_i128;
        for index in world_indices {
            let world = &scenario.worlds[*index];
            let mass = i128::from(world.probability_ppm.unwrap_or(0));
            utility_numerator = utility_numerator.checked_add(
                mass.checked_mul(i128::from(action.utility_by_state[&world.state_id]))
                    .ok_or(ReferenceRejectV1::ArithmeticOverflow)?,
            ).ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
            cost_numerator = cost_numerator.checked_add(
                mass.checked_mul(i128::from(spent.cost_utility_units))
                    .ok_or(ReferenceRejectV1::ArithmeticOverflow)?,
            ).ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
        }
        consider_expected(
            &mut best,
            ExpectedCandidateV1 {
                policy: ReferencePolicyNodeV1::Terminal { action_id: action.action_id.clone() },
                utility_numerator_ppm: utility_numerator,
                cost_numerator_ppm: cost_numerator,
            },
        );
    }

    if depth < scenario.maximum_horizon {
        for query in &scenario.queries {
            if queried.contains(&query.query_id) {
                continue;
            }
            let Some(next_spent) = spent.add_query(query, &scenario.budgets)? else {
                continue;
            };
            let next_queried = queried_with(queried, &query.query_id);
            let groups = partition_worlds(scenario, world_indices, &query.query_id);
            let mut branches = Vec::with_capacity(groups.len());
            let mut utility_numerator = 0_i128;
            let mut cost_numerator = 0_i128;
            let mut complete = true;
            for (observation_id, child_worlds) in groups {
                let child = search_expected(
                    scenario,
                    &child_worlds,
                    next_spent,
                    depth + 1,
                    &next_queried,
                    visited,
                )?;
                let Some(child) = child else {
                    complete = false;
                    break;
                };
                utility_numerator = utility_numerator.checked_add(child.utility_numerator_ppm)
                    .ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
                cost_numerator = cost_numerator.checked_add(child.cost_numerator_ppm)
                    .ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
                branches.push(ReferencePolicyBranchV1 {
                    observation_id,
                    next: Box::new(child.policy),
                });
            }
            if complete && !branches.is_empty() {
                let candidate = ExpectedCandidateV1 {
                    policy: ReferencePolicyNodeV1::Query {
                        query_id: query.query_id.clone(),
                        branches,
                    },
                    utility_numerator_ppm: utility_numerator,
                    cost_numerator_ppm: cost_numerator,
                };
                consider_expected(&mut best, candidate);
            }
        }
    }
    Ok(best)
}

fn consider_expected(best: &mut Option<ExpectedCandidateV1>, candidate: ExpectedCandidateV1) {
    let replace = best.as_ref().is_none_or(|current| {
        candidate.net() > current.net()
            || (candidate.net() == current.net()
                && (candidate.policy.node_count() < current.policy.node_count()
                    || (candidate.policy.node_count() == current.policy.node_count()
                        && candidate.policy.canonical_key() < current.policy.canonical_key())))
    });
    if replace {
        *best = Some(candidate);
    }
}

pub fn evaluate_robust_reference_v1(
    scenario: &ReferenceScenarioV1,
) -> Result<RobustReferenceResultV1, ReferenceRejectV1> {
    scenario.validate(false)?;
    let worlds = scenario.worlds.iter().enumerate()
        .filter(|(_, world)| world.possible)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut visited = 0usize;
    let best = search_robust(
        scenario,
        &worlds,
        SpentV1::default(),
        0,
        &BTreeSet::new(),
        &mut visited,
    )?.ok_or(ReferenceRejectV1::NoPermittedAction)?;
    if best.policy.node_count() > scenario.maximum_policy_nodes {
        return Err(ReferenceRejectV1::PolicyTooLarge);
    }
    Ok(RobustReferenceResultV1 {
        policy: best.policy,
        worst_case_regret_utility_units: best.worst_regret,
        worst_case_acquisition_cost_utility_units: best.worst_cost,
        explored_histories: visited,
    })
}

fn search_robust(
    scenario: &ReferenceScenarioV1,
    world_indices: &[usize],
    spent: SpentV1,
    depth: u32,
    queried: &BTreeSet<String>,
    visited: &mut usize,
) -> Result<Option<RobustCandidateV1>, ReferenceRejectV1> {
    *visited = visited.checked_add(1).ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
    if *visited > MAX_VISITED_HISTORIES {
        return Err(ReferenceRejectV1::SearchLimitExceeded);
    }
    let mut best: Option<RobustCandidateV1> = None;

    for action in &scenario.actions {
        if !action_allowed_for_worlds(action, scenario, world_indices) {
            continue;
        }
        let mut worst_regret = 0_i128;
        for index in world_indices {
            let world = &scenario.worlds[*index];
            let oracle = scenario.actions.iter()
                .filter(|candidate| candidate.allowed_by_state[&world.state_id])
                .map(|candidate| candidate.utility_by_state[&world.state_id])
                .max()
                .ok_or(ReferenceRejectV1::NoPermittedAction)?;
            let realized = action.utility_by_state[&world.state_id];
            let regret = i128::from(oracle)
                .checked_sub(i128::from(realized))
                .and_then(|value| value.checked_add(i128::from(spent.cost_utility_units)))
                .ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
            worst_regret = worst_regret.max(regret);
        }
        consider_robust(
            &mut best,
            RobustCandidateV1 {
                policy: ReferencePolicyNodeV1::Terminal { action_id: action.action_id.clone() },
                worst_regret,
                worst_cost: spent.cost_utility_units,
            },
        );
    }

    if depth < scenario.maximum_horizon {
        for query in &scenario.queries {
            if queried.contains(&query.query_id) {
                continue;
            }
            let Some(next_spent) = spent.add_query(query, &scenario.budgets)? else {
                continue;
            };
            let next_queried = queried_with(queried, &query.query_id);
            let groups = partition_worlds(scenario, world_indices, &query.query_id);
            let mut branches = Vec::with_capacity(groups.len());
            let mut worst_regret = 0_i128;
            let mut worst_cost = 0_u64;
            let mut complete = true;
            for (observation_id, child_worlds) in groups {
                let child = search_robust(
                    scenario,
                    &child_worlds,
                    next_spent,
                    depth + 1,
                    &next_queried,
                    visited,
                )?;
                let Some(child) = child else {
                    complete = false;
                    break;
                };
                worst_regret = worst_regret.max(child.worst_regret);
                worst_cost = worst_cost.max(child.worst_cost);
                branches.push(ReferencePolicyBranchV1 {
                    observation_id,
                    next: Box::new(child.policy),
                });
            }
            if complete && !branches.is_empty() {
                let candidate = RobustCandidateV1 {
                    policy: ReferencePolicyNodeV1::Query {
                        query_id: query.query_id.clone(),
                        branches,
                    },
                    worst_regret,
                    worst_cost,
                };
                consider_robust(&mut best, candidate);
            }
        }
    }
    Ok(best)
}

fn consider_robust(best: &mut Option<RobustCandidateV1>, candidate: RobustCandidateV1) {
    let replace = best.as_ref().is_none_or(|current| {
        candidate.worst_regret < current.worst_regret
            || (candidate.worst_regret == current.worst_regret
                && (candidate.policy.node_count() < current.policy.node_count()
                    || (candidate.policy.node_count() == current.policy.node_count()
                        && candidate.policy.canonical_key() < current.policy.canonical_key())))
    });
    if replace {
        *best = Some(candidate);
    }
}

fn action_allowed_for_worlds(
    action: &ReferenceActionV1,
    scenario: &ReferenceScenarioV1,
    world_indices: &[usize],
) -> bool {
    !world_indices.is_empty()
        && world_indices.iter().all(|index| {
            action.allowed_by_state[&scenario.worlds[*index].state_id]
        })
}

fn queried_with(queried: &BTreeSet<String>, query_id: &str) -> BTreeSet<String> {
    let mut next = queried.clone();
    next.insert(query_id.to_owned());
    next
}

fn partition_worlds(
    scenario: &ReferenceScenarioV1,
    world_indices: &[usize],
    query_id: &str,
) -> Vec<(String, Vec<usize>)> {
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    for index in world_indices {
        let observation = scenario.worlds[*index].observation_by_query[query_id].clone();
        groups.entry(observation).or_default().push(*index);
    }
    groups.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(
        query_id: &str,
        roots: &[&str],
        cost: u64,
    ) -> ReferenceQueryV1 {
        ReferenceQueryV1 {
            query_id: query_id.into(),
            source_root_ids: roots.iter().map(|root| (*root).to_owned()).collect(),
            tool_calls: 1,
            latency_ns: 10,
            context_tokens: 10,
            inspected_bytes: 10,
            cost_utility_units: cost,
        }
    }

    fn action(
        action_id: &str,
        utilities: &[(&str, i64)],
        allowed: &[(&str, bool)],
    ) -> ReferenceActionV1 {
        ReferenceActionV1 {
            action_id: action_id.into(),
            utility_by_state: utilities.iter().map(|(state, value)| ((*state).into(), *value)).collect(),
            allowed_by_state: allowed.iter().map(|(state, value)| ((*state).into(), *value)).collect(),
        }
    }

    fn world(
        id: &str,
        state: &str,
        probability: Option<u32>,
        observations: &[(&str, &str)],
    ) -> ReferenceWorldV1 {
        ReferenceWorldV1 {
            world_id: id.into(),
            state_id: state.into(),
            probability_ppm: probability,
            possible: true,
            observation_by_query: observations.iter()
                .map(|(query, observation)| ((*query).into(), (*observation).into()))
                .collect(),
        }
    }

    fn budgets() -> ReferenceBudgetsV1 {
        ReferenceBudgetsV1 {
            maximum_queries: 2,
            maximum_tool_calls: 2,
            maximum_latency_ns: 100,
            maximum_context_tokens: 100,
            maximum_inspected_bytes: 100,
            maximum_cost_utility_units: 10,
        }
    }

    fn adaptive_scenario() -> ReferenceScenarioV1 {
        ReferenceScenarioV1 {
            state_ids: vec!["bad".into(), "good".into()],
            worlds: vec![
                world("a-bad", "bad", Some(250_000), &[
                    ("route", "A"), ("check-a", "bad"), ("check-b", "unknown"),
                ]),
                world("a-good", "good", Some(250_000), &[
                    ("route", "A"), ("check-a", "good"), ("check-b", "unknown"),
                ]),
                world("b-bad", "bad", Some(250_000), &[
                    ("route", "B"), ("check-a", "unknown"), ("check-b", "bad"),
                ]),
                world("b-good", "good", Some(250_000), &[
                    ("route", "B"), ("check-a", "unknown"), ("check-b", "good"),
                ]),
            ],
            actions: vec![
                action("abstain", &[("bad", 0), ("good", 0)], &[("bad", true), ("good", true)]),
                action("proceed", &[("bad", -20), ("good", 20)], &[("bad", true), ("good", true)]),
            ],
            queries: vec![
                query("check-a", &["root-a"], 1),
                query("check-b", &["root-b"], 1),
                query("route", &["root-route"], 0),
            ],
            maximum_horizon: 2,
            maximum_policy_nodes: 16,
            budgets: budgets(),
        }
    }

    fn contains_query(node: &ReferencePolicyNodeV1, query_id: &str) -> bool {
        match node {
            ReferencePolicyNodeV1::Terminal { .. } => false,
            ReferencePolicyNodeV1::Query { query_id: selected, branches } => {
                selected == query_id || branches.iter().any(|branch| contains_query(&branch.next, query_id))
            }
        }
    }

    #[test]
    fn adaptive_evsi_uses_the_joint_world_to_choose_conditional_followups() {
        let result = evaluate_expected_reference_v1(&adaptive_scenario()).unwrap();
        assert_eq!(result.expected_utility_numerator_ppm, 10_000_000);
        assert_eq!(result.expected_cost_numerator_ppm, 1_000_000);
        assert_eq!(result.expected_net_utility_numerator_ppm, 9_000_000);
        assert!(matches!(result.policy, ReferencePolicyNodeV1::Query { .. }));
        assert!(result.explored_histories > 1);
    }

    #[test]
    fn adaptive_policy_routes_to_the_query_with_highest_conditional_value() {
        let result = evaluate_expected_reference_v1(&adaptive_scenario()).unwrap();
        let ReferencePolicyNodeV1::Query { query_id, branches } = result.policy else {
            panic!("positive conditional information value should select a query");
        };
        assert_eq!(query_id, "route");
        let route_a = branches.iter().find(|branch| branch.observation_id == "A").unwrap();
        let route_b = branches.iter().find(|branch| branch.observation_id == "B").unwrap();
        assert!(matches!(&*route_a.next, ReferencePolicyNodeV1::Query { query_id, .. } if query_id == "check-a"));
        assert!(matches!(&*route_b.next, ReferencePolicyNodeV1::Query { query_id, .. } if query_id == "check-b"));
    }

    #[test]
    fn policy_node_cap_rejects_an_oversized_oracle_result() {
        let mut scenario = adaptive_scenario();
        scenario.maximum_policy_nodes = 1;
        assert_eq!(
            evaluate_expected_reference_v1(&scenario),
            Err(ReferenceRejectV1::PolicyTooLarge),
        );
    }

    #[test]
    fn robust_mode_does_not_require_or_use_probability_masses() {
        let mut scenario = adaptive_scenario();
        for world in &mut scenario.worlds {
            world.probability_ppm = None;
        }
        let result = evaluate_robust_reference_v1(&scenario).unwrap();
        assert_eq!(result.worst_case_regret_utility_units, 1);
        assert_eq!(result.worst_case_acquisition_cost_utility_units, 1);
        assert!(matches!(result.policy, ReferencePolicyNodeV1::Query { .. }));
    }

    #[test]
    fn uninformative_positive_cost_query_is_not_selected() {
        let mut scenario = adaptive_scenario();
        scenario.queries = vec![query("noop", &["same-root"], 1)];
        for world in &mut scenario.worlds {
            world.observation_by_query = BTreeMap::from([("noop".into(), "same-observation".into())]);
        }
        let result = evaluate_expected_reference_v1(&scenario).unwrap();
        assert_eq!(result.expected_net_utility_numerator_ppm, 0);
        assert!(matches!(result.policy, ReferencePolicyNodeV1::Terminal { action_id } if action_id == "abstain"));
    }

    #[test]
    fn contradictory_observation_classes_remain_separate_policy_branches() {
        let mut scenario = adaptive_scenario();
        scenario.queries = vec![query("resolve-conflict", &["source-family"], 1)];
        for world in &mut scenario.worlds {
            let observation = if world.state_id == "good" { "supports-proceed" } else { "contradicts-proceed" };
            world.observation_by_query = BTreeMap::from([("resolve-conflict".into(), observation.into())]);
        }
        scenario.maximum_horizon = 1;
        let result = evaluate_expected_reference_v1(&scenario).unwrap();
        let ReferencePolicyNodeV1::Query { query_id, branches } = result.policy else {
            panic!("decision-relevant contradictory outcomes must remain explicit");
        };
        assert_eq!(query_id, "resolve-conflict");
        assert!(branches.iter().any(|branch| branch.observation_id == "supports-proceed"));
        assert!(branches.iter().any(|branch| branch.observation_id == "contradicts-proceed"));
    }

    #[test]
    fn duplicate_source_roots_do_not_create_independent_information() {
        let mut scenario = adaptive_scenario();
        scenario.queries[1].source_root_ids = vec!["root-a".into()];
        // The second path is an exact mirror of the first across every latent world.
        for world in &mut scenario.worlds {
            let mirrored = world.observation_by_query["check-a"].clone();
            world.observation_by_query.insert("check-b".into(), mirrored);
        }
        let result = evaluate_expected_reference_v1(&scenario).unwrap();
        let count = (if contains_query(&result.policy, "check-a") { 1 } else { 0 })
            + (if contains_query(&result.policy, "check-b") { 1 } else { 0 });
        assert!(count <= 1);
    }

    fn action_for_world<'a>(
        node: &'a ReferencePolicyNodeV1,
        world: &ReferenceWorldV1,
    ) -> &'a str {
        match node {
            ReferencePolicyNodeV1::Terminal { action_id } => action_id,
            ReferencePolicyNodeV1::Query { query_id, branches } => {
                let observation = &world.observation_by_query[query_id];
                let branch = branches.iter().find(|branch| &branch.observation_id == observation)
                    .expect("the selected policy has a branch for every possible observation");
                action_for_world(&branch.next, world)
            }
        }
    }

    #[test]
    fn mid_run_revocation_requires_revalidation_before_commit() {
        let scenario = ReferenceScenarioV1 {
            state_ids: vec!["active".into(), "revoked".into()],
            worlds: vec![
                world("active-world", "active", Some(500_000), &[
                    ("initial", "valid-at-first-check"), ("revalidate", "still-current"),
                ]),
                world("revoked-world", "revoked", Some(500_000), &[
                    ("initial", "valid-at-first-check"), ("revalidate", "revoked"),
                ]),
            ],
            actions: vec![
                action("abstain", &[("active", 0), ("revoked", 0)], &[("active", true), ("revoked", true)]),
                action("commit", &[("active", 10), ("revoked", -10)], &[("active", true), ("revoked", false)]),
            ],
            queries: vec![
                query("initial", &["origin-registry"], 1),
                query("revalidate", &["revocation-registry"], 1),
            ],
            maximum_horizon: 2,
            maximum_policy_nodes: 12,
            budgets: budgets(),
        };
        let result = evaluate_expected_reference_v1(&scenario).unwrap();
        let active = scenario.worlds.iter().find(|world| world.world_id == "active-world").unwrap();
        let revoked = scenario.worlds.iter().find(|world| world.world_id == "revoked-world").unwrap();
        assert_eq!(action_for_world(&result.policy, active), "commit");
        assert_eq!(action_for_world(&result.policy, revoked), "abstain");
    }

    #[test]
    fn hard_action_constraints_prevent_commit_when_any_world_forbids_it() {
        let mut scenario = adaptive_scenario();
        scenario.budgets.maximum_queries = 0;
        scenario.actions = vec![
            action("abstain", &[("bad", 0), ("good", 0)], &[("bad", true), ("good", true)]),
            action("commit", &[("bad", 100), ("good", 100)], &[("bad", false), ("good", true)]),
        ];
        let result = evaluate_expected_reference_v1(&scenario).unwrap();
        let ReferencePolicyNodeV1::Terminal { action_id } = result.policy else {
            panic!("no query budget means the evaluator must choose a universally permitted terminal action");
        };
        assert_eq!(action_id, "abstain");
    }

    #[test]
    fn query_budget_and_cost_caps_exclude_ineligible_acquisitions() {
        let mut scenario = adaptive_scenario();
        scenario.budgets.maximum_queries = 2;
        scenario.budgets.maximum_cost_utility_units = 0;
        scenario.queries.iter_mut().for_each(|query| query.cost_utility_units = 1);
        let result = evaluate_expected_reference_v1(&scenario).unwrap();
        assert!(matches!(result.policy, ReferencePolicyNodeV1::Terminal { .. }));
    }

    #[test]
    fn malformed_world_probability_mass_is_rejected() {
        let mut scenario = adaptive_scenario();
        scenario.worlds[0].probability_ppm = Some(200_000);
        assert_eq!(
            evaluate_expected_reference_v1(&scenario),
            Err(ReferenceRejectV1::InvalidProbabilityMass),
        );
    }

    #[test]
    fn equal_value_policy_selection_is_independent_of_query_order() {
        let first = adaptive_scenario();
        let mut second = first.clone();
        second.queries.reverse();
        let left = evaluate_expected_reference_v1(&first).unwrap();
        let right = evaluate_expected_reference_v1(&second).unwrap();
        assert_eq!(left.expected_net_utility_numerator_ppm, right.expected_net_utility_numerator_ppm);
        assert_eq!(left.policy.canonical_key(), right.policy.canonical_key());
    }

    #[test]
    fn robust_policy_selection_is_invariant_to_query_enumeration_order() {
        let first = adaptive_scenario();
        let mut second = first.clone();
        second.queries.reverse();
        let left = evaluate_robust_reference_v1(&first).unwrap();
        let right = evaluate_robust_reference_v1(&second).unwrap();
        assert_eq!(left.worst_case_regret_utility_units, right.worst_case_regret_utility_units);
        assert_eq!(left.policy.canonical_key(), right.policy.canonical_key());
    }

    #[test]
    fn search_is_bounded_and_reports_explored_histories() {
        let result = evaluate_expected_reference_v1(&adaptive_scenario()).unwrap();
        assert!(result.explored_histories <= MAX_VISITED_HISTORIES);
        assert!(result.policy.node_count() <= 16);
    }

    #[test]
    fn only_possible_worlds_contribute_to_the_expected_value_denominator() {
        let mut scenario = adaptive_scenario();
        let mut impossible = scenario.worlds[0].clone();
        impossible.world_id = "z-impossible".into();
        impossible.possible = false;
        impossible.probability_ppm = Some(0);
        scenario.worlds[0].probability_ppm = Some(250_000);
        scenario.worlds.push(impossible);
        let result = evaluate_expected_reference_v1(&scenario).unwrap();
        assert!(result.expected_net_utility_numerator_ppm > 0);
    }
}
