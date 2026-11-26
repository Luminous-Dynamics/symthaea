// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact memoized adaptive-policy solvers for the V1 finite-world contract.
//!
//! The exhaustive evaluator remains the correctness oracle. This module
//! changes only the search strategy: it memoizes equivalent subproblems using
//! the possible-world bitset and the set of already-used query indices.
//!
//! This key is sufficient only because V1 queries are single-use and have
//! fixed, additive resource costs. If a later contract adds history-dependent
//! costs, permissions, expiry, or rights state, that state must be committed
//! into the cache key before reusing this solver.

use std::collections::{BTreeMap, HashMap};

use crate::adaptive_policy_reference::{
    ExpectedReferenceResultV1, ReferenceActionV1, ReferencePolicyBranchV1,
    ReferencePolicyNodeV1, ReferenceQueryV1, ReferenceRejectV1,
    ReferenceScenarioV1, RobustReferenceResultV1,
};

const MAX_MEMOIZED_STATES: usize = 50_000;
const MAX_MEMOIZED_CALLS: usize = 250_000;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct StateKeyV1 {
    possible_worlds: u64,
    queried: u16,
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
        scenario: &ReferenceScenarioV1,
    ) -> Result<Option<Self>, ReferenceRejectV1> {
        let next = Self {
            queries: self.queries.checked_add(1)
                .ok_or(ReferenceRejectV1::ArithmeticOverflow)?,
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
        let budget = &scenario.budgets;
        let within = next.queries <= budget.maximum_queries
            && next.tool_calls <= budget.maximum_tool_calls
            && next.latency_ns <= budget.maximum_latency_ns
            && next.context_tokens <= budget.maximum_context_tokens
            && next.inspected_bytes <= budget.maximum_inspected_bytes
            && next.cost_utility_units <= budget.maximum_cost_utility_units;
        Ok(within.then_some(next))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoizedSearchStatsV1 {
    /// Number of distinct subproblems actually expanded.
    pub recursive_calls: usize,
    /// Number of distinct subproblems actually expanded.
    pub expanded_states: usize,
    /// Number of recursive calls satisfied from the memo table.
    pub cache_hits: usize,
    /// Number of distinct subproblems inserted into the memo table.
    pub cache_misses: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoizedExpectedResultV1 {
    pub reference_result: ExpectedReferenceResultV1,
    pub stats: MemoizedSearchStatsV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoizedRobustResultV1 {
    pub reference_result: RobustReferenceResultV1,
    pub stats: MemoizedSearchStatsV1,
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

#[derive(Default)]
struct SearchStatsV1 {
    recursive_calls: usize,
    expanded_states: usize,
    cache_hits: usize,
    cache_misses: usize,
}

impl SearchStatsV1 {
    fn snapshot(&self) -> MemoizedSearchStatsV1 {
        MemoizedSearchStatsV1 {
            recursive_calls: self.recursive_calls,
            expanded_states: self.expanded_states,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
        }
    }

    fn call(&mut self) -> Result<(), ReferenceRejectV1> {
        self.recursive_calls = self.recursive_calls.checked_add(1)
            .ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
        if self.recursive_calls > MAX_MEMOIZED_CALLS {
            return Err(ReferenceRejectV1::SearchLimitExceeded);
        }
        Ok(())
    }

    fn expand(&mut self) -> Result<(), ReferenceRejectV1> {
        self.expanded_states = self.expanded_states.checked_add(1)
            .ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
        if self.expanded_states > MAX_MEMOIZED_STATES {
            return Err(ReferenceRejectV1::SearchLimitExceeded);
        }
        Ok(())
    }
}

fn possible_world_mask(scenario: &ReferenceScenarioV1) -> u64 {
    scenario.worlds.iter().enumerate().fold(0_u64, |mask, (index, world)| {
        if world.possible { mask | (1_u64 << index) } else { mask }
    })
}

fn world_indices(mask: u64, scenario: &ReferenceScenarioV1) -> impl Iterator<Item = usize> + '_ {
    (0..scenario.worlds.len()).filter(move |index| (mask & (1_u64 << index)) != 0)
}

fn spent_for_mask(
    scenario: &ReferenceScenarioV1,
    queried: u16,
) -> Result<Option<SpentV1>, ReferenceRejectV1> {
    let mut spent = SpentV1::default();
    for (index, query) in scenario.queries.iter().enumerate() {
        if queried & (1_u16 << index) == 0 {
            continue;
        }
        let Some(next) = spent.add_query(query, scenario)? else {
            return Ok(None);
        };
        spent = next;
    }
    Ok(Some(spent))
}

fn action_allowed_for_mask(
    action: &ReferenceActionV1,
    scenario: &ReferenceScenarioV1,
    worlds: u64,
) -> bool {
    worlds != 0 && world_indices(worlds, scenario)
        .all(|index| action.allowed_by_state[&scenario.worlds[index].state_id])
}

fn partition_world_mask(
    scenario: &ReferenceScenarioV1,
    worlds: u64,
    query_id: &str,
) -> Vec<(String, u64)> {
    let mut groups = BTreeMap::<String, u64>::new();
    for index in world_indices(worlds, scenario) {
        let observation = scenario.worlds[index].observation_by_query[query_id].clone();
        *groups.entry(observation).or_default() |= 1_u64 << index;
    }
    groups.into_iter().collect()
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

fn search_expected_state(
    scenario: &ReferenceScenarioV1,
    memo: &mut HashMap<StateKeyV1, Option<ExpectedCandidateV1>>,
    stats: &mut SearchStatsV1,
    worlds: u64,
    queried: u16,
) -> Result<Option<ExpectedCandidateV1>, ReferenceRejectV1> {
    stats.call()?;
    let key = StateKeyV1 { possible_worlds: worlds, queried };
    if let Some(cached) = memo.get(&key) {
        stats.cache_hits += 1;
        return Ok(cached.clone());
    }
    stats.expand()?;
    stats.cache_misses += 1;

    let spent = spent_for_mask(scenario, queried)?
        .ok_or(ReferenceRejectV1::BudgetExceeded)?;
    let indices = world_indices(worlds, scenario).collect::<Vec<_>>();
    if indices.is_empty() {
        return Err(ReferenceRejectV1::NoPossibleWorlds);
    }

    let mut best = None;
    let mut mass_sum = 0_i128;
    for index in &indices {
        mass_sum = mass_sum.checked_add(i128::from(
            scenario.worlds[*index].probability_ppm.unwrap_or(0),
        )).ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
    }

    for action in &scenario.actions {
        if !action_allowed_for_mask(action, scenario, worlds) {
            continue;
        }
        let mut utility_numerator_ppm = 0_i128;
        for index in &indices {
            let world = &scenario.worlds[*index];
            let mass = i128::from(world.probability_ppm.unwrap_or(0));
            let contribution = mass.checked_mul(i128::from(
                action.utility_by_state[&world.state_id],
            )).ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
            utility_numerator_ppm = utility_numerator_ppm.checked_add(contribution)
                .ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
        }
        let cost_numerator_ppm = mass_sum.checked_mul(i128::from(spent.cost_utility_units))
            .ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
        consider_expected(&mut best, ExpectedCandidateV1 {
            policy: ReferencePolicyNodeV1::Terminal {
                action_id: action.action_id.clone(),
            },
            utility_numerator_ppm,
            cost_numerator_ppm,
        });
    }

    if queried.count_ones() < scenario.maximum_horizon {
        for (query_index, query) in scenario.queries.iter().enumerate() {
            let query_bit = 1_u16 << query_index;
            if queried & query_bit != 0 {
                continue;
            }
            if spent.add_query(query, scenario)?.is_none() {
                continue;
            }
            let next_queried = queried | query_bit;
            let mut branches = Vec::new();
            let mut utility_numerator_ppm = 0_i128;
            let mut cost_numerator_ppm = 0_i128;
            let mut complete = true;
            for (observation_id, child_worlds) in
                partition_world_mask(scenario, worlds, &query.query_id)
            {
                let Some(child) = search_expected_state(
                    scenario, memo, stats, child_worlds, next_queried,
                )? else {
                    complete = false;
                    break;
                };
                utility_numerator_ppm = utility_numerator_ppm
                    .checked_add(child.utility_numerator_ppm)
                    .ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
                cost_numerator_ppm = cost_numerator_ppm
                    .checked_add(child.cost_numerator_ppm)
                    .ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
                branches.push(ReferencePolicyBranchV1 {
                    observation_id,
                    next: Box::new(child.policy),
                });
            }
            if complete && !branches.is_empty() {
                consider_expected(&mut best, ExpectedCandidateV1 {
                    policy: ReferencePolicyNodeV1::Query {
                        query_id: query.query_id.clone(),
                        branches,
                    },
                    utility_numerator_ppm,
                    cost_numerator_ppm,
                });
            }
        }
    }

    memo.insert(key, best.clone());
    Ok(best)
}

fn search_robust_state(
    scenario: &ReferenceScenarioV1,
    memo: &mut HashMap<StateKeyV1, Option<RobustCandidateV1>>,
    stats: &mut SearchStatsV1,
    worlds: u64,
    queried: u16,
) -> Result<Option<RobustCandidateV1>, ReferenceRejectV1> {
    let key = StateKeyV1 { possible_worlds: worlds, queried };
    if let Some(cached) = memo.get(&key) {
        stats.cache_hits += 1;
        return Ok(cached.clone());
    }
    stats.expand()?;
    stats.cache_misses += 1;

    let spent = spent_for_mask(scenario, queried)?
        .ok_or(ReferenceRejectV1::BudgetExceeded)?;
    let indices = world_indices(worlds, scenario).collect::<Vec<_>>();
    if indices.is_empty() {
        return Err(ReferenceRejectV1::NoPossibleWorlds);
    }

    let mut best = None;
    for action in &scenario.actions {
        if !action_allowed_for_mask(action, scenario, worlds) {
            continue;
        }
        let mut worst_regret = 0_i128;
        for index in &indices {
            let world = &scenario.worlds[*index];
            let oracle = scenario.actions.iter()
                .filter(|candidate| candidate.allowed_by_state[&world.state_id])
                .map(|candidate| candidate.utility_by_state[&world.state_id])
                .max()
                .ok_or(ReferenceRejectV1::NoPermittedAction)?;
            let regret = i128::from(oracle)
                .checked_sub(i128::from(action.utility_by_state[&world.state_id]))
                .and_then(|value| value.checked_add(i128::from(spent.cost_utility_units)))
                .ok_or(ReferenceRejectV1::ArithmeticOverflow)?;
            worst_regret = worst_regret.max(regret);
        }
        consider_robust(&mut best, RobustCandidateV1 {
            policy: ReferencePolicyNodeV1::Terminal {
                action_id: action.action_id.clone(),
            },
            worst_regret,
            worst_cost: spent.cost_utility_units,
        });
    }

    if queried.count_ones() < scenario.maximum_horizon {
        for (query_index, query) in scenario.queries.iter().enumerate() {
            let query_bit = 1_u16 << query_index;
            if queried & query_bit != 0 {
                continue;
            }
            if spent.add_query(query, scenario)?.is_none() {
                continue;
            }
            let next_queried = queried | query_bit;
            let mut branches = Vec::new();
            let mut worst_regret = 0_i128;
            let mut worst_cost = 0_u64;
            let mut complete = true;
            for (observation_id, child_worlds) in
                partition_world_mask(scenario, worlds, &query.query_id)
            {
                let Some(child) = search_robust_state(
                    scenario, memo, stats, child_worlds, next_queried,
                )? else {
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
                consider_robust(&mut best, RobustCandidateV1 {
                    policy: ReferencePolicyNodeV1::Query {
                        query_id: query.query_id.clone(),
                        branches,
                    },
                    worst_regret,
                    worst_cost,
                });
            }
        }
    }

    memo.insert(key, best.clone());
    Ok(best)
}

/// Evaluate expected utility with exact memoization.
pub fn evaluate_expected_memoized_v1(
    scenario: &ReferenceScenarioV1,
) -> Result<MemoizedExpectedResultV1, ReferenceRejectV1> {
    scenario.validate(true)?;
    let mut memo = HashMap::new();
    let mut stats = SearchStatsV1::default();
    let best = search_expected_state(
        scenario, &mut memo, &mut stats, possible_world_mask(scenario), 0,
    )?.ok_or(ReferenceRejectV1::NoPermittedAction)?;
    if best.policy.node_count() > scenario.maximum_policy_nodes {
        return Err(ReferenceRejectV1::PolicyTooLarge);
    }
    Ok(MemoizedExpectedResultV1 {
        reference_result: ExpectedReferenceResultV1 {
            expected_utility_numerator_ppm: best.utility_numerator_ppm,
            expected_cost_numerator_ppm: best.cost_numerator_ppm,
            expected_net_utility_numerator_ppm: best.net(),
            policy: best.policy,
            explored_histories: stats.recursive_calls,
        },
        stats: stats.snapshot(),
    })
}

/// Evaluate worst-case regret plus acquisition cost with exact memoization.
pub fn evaluate_robust_memoized_v1(
    scenario: &ReferenceScenarioV1,
) -> Result<MemoizedRobustResultV1, ReferenceRejectV1> {
    scenario.validate(false)?;
    let mut memo = HashMap::new();
    let mut stats = SearchStatsV1::default();
    let best = search_robust_state(
        scenario, &mut memo, &mut stats, possible_world_mask(scenario), 0,
    )?.ok_or(ReferenceRejectV1::NoPermittedAction)?;
    if best.policy.node_count() > scenario.maximum_policy_nodes {
        return Err(ReferenceRejectV1::PolicyTooLarge);
    }
    Ok(MemoizedRobustResultV1 {
        reference_result: RobustReferenceResultV1 {
            policy: best.policy,
            worst_case_regret_utility_units: best.worst_regret,
            worst_case_acquisition_cost_utility_units: best.worst_cost,
            explored_histories: stats.expanded_states,
        },
        stats: stats.snapshot(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_policy_reference::{
        evaluate_expected_reference_v1, evaluate_robust_reference_v1, ReferenceWorldV1,
    };

    fn action(
        id: &str,
        utility_bad: i64,
        utility_good: i64,
        allowed_bad: bool,
        allowed_good: bool,
    ) -> ReferenceActionV1 {
        ReferenceActionV1 {
            action_id: id.to_owned(),
            utility_by_state: BTreeMap::from([
                ("bad".to_owned(), utility_bad),
                ("good".to_owned(), utility_good),
            ]),
            allowed_by_state: BTreeMap::from([
                ("bad".to_owned(), allowed_bad),
                ("good".to_owned(), allowed_good),
            ]),
        }
    }

    fn scenario(seed: usize) -> ReferenceScenarioV1 {
        let weights = [200_000, 0, 150_000, 150_000, 100_000, 150_000, 100_000, 150_000];
        let worlds = (0..8).map(|index| {
            let state_id = if (index + seed) % 2 == 0 { "bad" } else { "good" };
            ReferenceWorldV1 {
                world_id: format!("world-{index}"),
                state_id: state_id.to_owned(),
                probability_ppm: Some(weights[index]),
                possible: true,
                observation_by_query: BTreeMap::from([
                    ("q-a".to_owned(), if index / 4 == 0 { "left" } else { "right" }.to_owned()),
                    ("q-b".to_owned(), if (index / 2) % 2 == 0 { "low" } else { "high" }.to_owned()),
                    ("q-c".to_owned(), if index % 2 == 0 { "even" } else { "odd" }.to_owned()),
                ]),
            }
        }).collect();
        ReferenceScenarioV1 {
            state_ids: vec!["bad".to_owned(), "good".to_owned()],
            worlds,
            actions: vec![
                action("abstain", 0, 0, true, true),
                action("defer", -1, 2, true, true),
                action("proceed", -20, 20, seed % 2 == 0, true),
            ],
            queries: vec![
                ReferenceQueryV1 {
                    query_id: "q-a".to_owned(),
                    source_root_ids: vec!["root-a".to_owned()],
                    tool_calls: 1,
                    latency_ns: 10,
                    context_tokens: 8,
                    inspected_bytes: 32,
                    cost_utility_units: 1,
                },
                ReferenceQueryV1 {
                    query_id: "q-b".to_owned(),
                    source_root_ids: vec!["root-b".to_owned()],
                    tool_calls: 1,
                    latency_ns: 12,
                    context_tokens: 9,
                    inspected_bytes: 36,
                    cost_utility_units: 2,
                },
                ReferenceQueryV1 {
                    query_id: "q-c".to_owned(),
                    source_root_ids: vec!["root-c".to_owned()],
                    tool_calls: 1,
                    latency_ns: 14,
                    context_tokens: 10,
                    inspected_bytes: 40,
                    cost_utility_units: 1,
                },
            ],
            maximum_horizon: 3,
            maximum_policy_nodes: 128,
            budgets: crate::adaptive_policy_reference::ReferenceBudgetsV1 {
                maximum_queries: 3,
                maximum_tool_calls: 3,
                maximum_latency_ns: 100,
                maximum_context_tokens: 100,
                maximum_inspected_bytes: 200,
                maximum_cost_utility_units: 10,
            },
        }
    }

    fn assert_expected_equal(
        left: &ExpectedReferenceResultV1,
        right: &ExpectedReferenceResultV1,
    ) {
        assert_eq!(left.expected_utility_numerator_ppm, right.expected_utility_numerator_ppm);
        assert_eq!(left.expected_cost_numerator_ppm, right.expected_cost_numerator_ppm);
        assert_eq!(left.expected_net_utility_numerator_ppm, right.expected_net_utility_numerator_ppm);
        assert_eq!(left.policy, right.policy);
    }

    fn assert_robust_equal(
        left: &RobustReferenceResultV1,
        right: &RobustReferenceResultV1,
    ) {
        assert_eq!(left.worst_case_regret_utility_units, right.worst_case_regret_utility_units);
        assert_eq!(left.worst_case_acquisition_cost_utility_units, right.worst_case_acquisition_cost_utility_units);
        assert_eq!(left.policy, right.policy);
    }

    #[test]
    fn expected_solver_matches_exhaustive_reference_over_deterministic_world_families() {
        for seed in 0..16 {
            let input = scenario(seed);
            let reference = evaluate_expected_reference_v1(&input).unwrap();
            let optimized = evaluate_expected_memoized_v1(&input).unwrap();
            assert_expected_equal(&reference, &optimized.reference_result);
            assert_eq!(optimized.stats.cache_misses, optimized.stats.expanded_states);
            assert_eq!(optimized.stats.recursive_calls, optimized.stats.expanded_states + optimized.stats.cache_hits);
        }
    }

    #[test]
    fn robust_solver_matches_exhaustive_reference_without_probabilities() {
        for seed in 0..16 {
            let mut input = scenario(seed);
            for world in &mut input.worlds {
                world.probability_ppm = None;
            }
            let reference = evaluate_robust_reference_v1(&input).unwrap();
            let optimized = evaluate_robust_memoized_v1(&input).unwrap();
            assert_robust_equal(&reference, &optimized.reference_result);
        }
    }

    #[test]
    fn zero_mass_possible_world_still_constrains_memoized_expected_action() {
        let mut input = scenario(0);
        let mut zero_mass_forbidden = input.worlds[0].clone();
        zero_mass_forbidden.world_id = "zero-mass-forbidden".to_owned();
        zero_mass_forbidden.state_id = "bad".to_owned();
        zero_mass_forbidden.probability_ppm = Some(0);
        let mut positive_mass_allowed = input.worlds[1].clone();
        positive_mass_allowed.world_id = "positive-mass-allowed".to_owned();
        positive_mass_allowed.state_id = "good".to_owned();
        positive_mass_allowed.probability_ppm = Some(1_000_000);
        input.worlds = vec![zero_mass_forbidden, positive_mass_allowed];
        input.budgets.maximum_queries = 0;
        input.actions = vec![
            action("abstain", 0, 0, true, true),
            action("commit", 100, 100, false, true),
        ];
        let reference = evaluate_expected_reference_v1(&input).unwrap();
        let optimized = evaluate_expected_memoized_v1(&input).unwrap();
        assert_expected_equal(&reference, &optimized.reference_result);
        assert!(matches!(
            optimized.reference_result.policy,
            ReferencePolicyNodeV1::Terminal { action_id } if action_id == "abstain"
        ));
    }

    #[test]
    fn exact_cache_reuses_order_convergent_joint_observation_subproblems() {
        let input = scenario(0);
        let optimized = evaluate_expected_memoized_v1(&input).unwrap();
        assert!(optimized.stats.cache_hits > 0);
        assert!(optimized.stats.expanded_states < 20_000);
    }

    #[test]
    fn memoized_solver_is_deterministic_under_query_enumeration_permutation() {
        let first = scenario(4);
        let mut second = first.clone();
        second.queries.reverse();

        let left = evaluate_expected_memoized_v1(&first).unwrap();
        let right = evaluate_expected_memoized_v1(&second).unwrap();
        assert_expected_equal(&left.reference_result, &right.reference_result);

        let left_robust = evaluate_robust_memoized_v1(&first).unwrap();
        let right_robust = evaluate_robust_memoized_v1(&second).unwrap();
        assert_robust_equal(&left_robust.reference_result, &right_robust.reference_result);
    }
}
