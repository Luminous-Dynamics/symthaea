//! Exhaustive finite-lattice check for adaptive/fixed support-expansion monotonicity.
//!
//! Two opposing-decision worlds, two binary observation actions and four
//! possible joint tuples per world induce 4,225 ordered support-inclusion
//! pairs. This executable enumerates all of them and fails closed on any
//! monotonicity violation.

use clap::Parser;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const WORLD_SUPPORT_MASKS: u8 = 0b1111;
const TUPLE_COUNT: usize = 4;
const ACTION_COUNT: usize = 2;
const ACTION_COSTS: [u64; ACTION_COUNT] = [1, 2];
const MAX_RETAINED_COUNTEREXAMPLES: usize = 16;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-support-expansion-lattice-audit",
    version,
    about = "Exhaustively check exact adaptive/fixed monotonicity over a bounded joint-support lattice"
)]
struct Args {
    /// Optional deterministic JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Eq, PartialEq)]
struct Counterexample {
    base_world_0_support_mask: u8,
    expanded_world_0_support_mask: u8,
    base_world_1_support_mask: u8,
    expanded_world_1_support_mask: u8,
    adaptive_cost_before: Option<u64>,
    adaptive_cost_after: Option<u64>,
    fixed_cost_before: Option<u64>,
    fixed_cost_after: Option<u64>,
    adaptive_monotonicity_violated: bool,
    fixed_monotonicity_violated: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    tool: &'static str,
    world_count: usize,
    action_count: usize,
    joint_tuple_universe_per_world: usize,
    action_costs: [u64; ACTION_COUNT],
    nonempty_base_support_configurations_per_world: usize,
    ordered_support_inclusion_pairs_checked: u64,
    adaptive_feasible_base_cases: u64,
    adaptive_feasible_expanded_cases: u64,
    adaptive_cost_unchanged_cases: u64,
    adaptive_cost_increased_cases: u64,
    adaptive_policy_lost_cases: u64,
    adaptive_monotonicity_violations: u64,
    adaptive_policy_resurrections: u64,
    fixed_feasible_base_cases: u64,
    fixed_feasible_expanded_cases: u64,
    fixed_cost_unchanged_cases: u64,
    fixed_cost_increased_cases: u64,
    fixed_plan_lost_cases: u64,
    fixed_monotonicity_violations: u64,
    fixed_plan_resurrections: u64,
    counterexamples_retained: usize,
    counterexamples: Vec<Counterexample>,
    monotonicity_passed: bool,
    exhaustive: bool,
    status: &'static str,
    scope_note: &'static str,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
struct State {
    remaining_actions: u8,
    world_support_masks: [u8; 2],
}

fn tuple_outcome(tuple_index: usize, action_index: usize) -> u8 {
    ((tuple_index >> action_index) & 1) as u8
}

fn filter_support(mask: u8, action_index: usize, outcome: u8) -> u8 {
    let mut filtered = 0_u8;
    for tuple_index in 0..TUPLE_COUNT {
        let bit = 1_u8 << tuple_index;
        if mask & bit != 0 && tuple_outcome(tuple_index, action_index) == outcome {
            filtered |= bit;
        }
    }
    filtered
}

fn query_is_unanimous(supports: [u8; 2]) -> bool {
    // World 0 supports the frozen query; world 1 refutes it.
    (supports[0] != 0) ^ (supports[1] != 0)
}

fn solve_adaptive(
    state: State,
    memo: &mut BTreeMap<State, Option<u64>>,
) -> Option<u64> {
    if let Some(cached) = memo.get(&state) {
        return *cached;
    }
    if query_is_unanimous(state.world_support_masks) {
        memo.insert(state, Some(0));
        return Some(0);
    }
    if state.remaining_actions == 0 {
        memo.insert(state, None);
        return None;
    }

    let mut best: Option<u64> = None;
    for (action_index, action_cost) in ACTION_COSTS.iter().copied().enumerate() {
        let action_bit = 1_u8 << action_index;
        if state.remaining_actions & action_bit == 0 {
            continue;
        }
        let mut possible_outcomes = 0_u8;
        for world_index in 0..2 {
            for tuple_index in 0..TUPLE_COUNT {
                if state.world_support_masks[world_index] & (1_u8 << tuple_index) != 0 {
                    possible_outcomes |= 1_u8 << tuple_outcome(tuple_index, action_index);
                }
            }
        }
        let mut worst_child = 0_u64;
        let mut action_valid = true;
        for outcome in 0..2 {
            if possible_outcomes & (1_u8 << outcome) == 0 {
                continue;
            }
            let child = State {
                remaining_actions: state.remaining_actions & !action_bit,
                world_support_masks: [
                    filter_support(state.world_support_masks[0], action_index, outcome),
                    filter_support(state.world_support_masks[1], action_index, outcome),
                ],
            };
            let Some(child_cost) = solve_adaptive(child, memo) else {
                action_valid = false;
                break;
            };
            worst_child = worst_child.max(child_cost);
        }
        if action_valid {
            let cost = action_cost
                .checked_add(worst_child)
                .expect("bounded lattice action cost cannot overflow");
            if best.is_none_or(|known| cost < known) {
                best = Some(cost);
            }
        }
    }
    memo.insert(state, best);
    best
}

fn adaptive_cost(world_support_masks: [u8; 2]) -> Option<u64> {
    let initial = State {
        remaining_actions: (1_u8 << ACTION_COUNT) - 1,
        world_support_masks,
    };
    solve_adaptive(initial, &mut BTreeMap::new())
}

fn projection(mask: u8, selected_actions: u8) -> Vec<u8> {
    let mut values = Vec::new();
    for tuple_index in 0..TUPLE_COUNT {
        if mask & (1_u8 << tuple_index) == 0 {
            continue;
        }
        let mut projected = 0_u8;
        for action_index in 0..ACTION_COUNT {
            if selected_actions & (1_u8 << action_index) != 0 {
                projected |= tuple_outcome(tuple_index, action_index) << action_index;
            }
        }
        values.push(projected);
    }
    values.sort_unstable();
    values.dedup();
    values
}

fn fixed_cost(world_support_masks: [u8; 2]) -> Option<u64> {
    let mut best = None;
    for selected_actions in 1_u8..(1_u8 << ACTION_COUNT) {
        let left = projection(world_support_masks[0], selected_actions);
        let right = projection(world_support_masks[1], selected_actions);
        if !left.iter().any(|value| right.binary_search(value).is_ok()) {
            let mut effort = 0_u64;
            for (action_index, action_cost) in ACTION_COSTS.iter().copied().enumerate() {
                if selected_actions & (1_u8 << action_index) != 0 {
                    effort += action_cost;
                }
            }
            if best.is_none_or(|known| effort < known) {
                best = Some(effort);
            }
        }
    }
    best
}

fn support_supersets(base_mask: u8) -> Vec<u8> {
    (1..=WORLD_SUPPORT_MASKS)
        .filter(|expanded| base_mask & !expanded == 0)
        .collect()
}

fn cost_non_decreasing(before: Option<u64>, after: Option<u64>) -> bool {
    match (before, after) {
        (Some(before), Some(after)) => after >= before,
        (Some(_), None) | (None, None) => true,
        (None, Some(_)) => false,
    }
}

fn evaluate() -> Report {
    let masks_per_world = WORLD_SUPPORT_MASKS as usize;
    let mut report = Report {
        schema_version: 1,
        tool: "connectome-support-expansion-lattice-audit",
        world_count: 2,
        action_count: ACTION_COUNT,
        joint_tuple_universe_per_world: TUPLE_COUNT,
        action_costs: ACTION_COSTS,
        nonempty_base_support_configurations_per_world: masks_per_world,
        ordered_support_inclusion_pairs_checked: 0,
        adaptive_feasible_base_cases: 0,
        adaptive_feasible_expanded_cases: 0,
        adaptive_cost_unchanged_cases: 0,
        adaptive_cost_increased_cases: 0,
        adaptive_policy_lost_cases: 0,
        adaptive_monotonicity_violations: 0,
        adaptive_policy_resurrections: 0,
        fixed_feasible_base_cases: 0,
        fixed_feasible_expanded_cases: 0,
        fixed_cost_unchanged_cases: 0,
        fixed_cost_increased_cases: 0,
        fixed_plan_lost_cases: 0,
        fixed_monotonicity_violations: 0,
        fixed_plan_resurrections: 0,
        counterexamples_retained: 0,
        counterexamples: Vec::new(),
        monotonicity_passed: false,
        exhaustive: true,
        status: "NOT_EVALUATED",
        scope_note: "Exhaustive over all non-empty support masks for two opposing-decision worlds, two binary observation actions and four possible joint tuples per world. It verifies only finite-set monotonicity; it does not validate biological support completeness, evidence provenance or measurement-model validity.",
    };

    for base_0 in 1..=WORLD_SUPPORT_MASKS {
        let expanded_0_options = support_supersets(base_0);
        for expanded_0 in expanded_0_options {
            for base_1 in 1..=WORLD_SUPPORT_MASKS {
                let expanded_1_options = support_supersets(base_1);
                for expanded_1 in expanded_1_options {
                    let base = [base_0, base_1];
                    let expanded = [expanded_0, expanded_1];
                    let adaptive_before = adaptive_cost(base);
                    let adaptive_after = adaptive_cost(expanded);
                    let fixed_before = fixed_cost(base);
                    let fixed_after = fixed_cost(expanded);
                    report.ordered_support_inclusion_pairs_checked += 1;

                    if adaptive_before.is_some() {
                        report.adaptive_feasible_base_cases += 1;
                    }
                    if adaptive_after.is_some() {
                        report.adaptive_feasible_expanded_cases += 1;
                    }
                    match (adaptive_before, adaptive_after) {
                        (Some(before), Some(after)) if after == before => {
                            report.adaptive_cost_unchanged_cases += 1;
                        }
                        (Some(before), Some(after)) if after > before => {
                            report.adaptive_cost_increased_cases += 1;
                        }
                        (Some(_), None) => report.adaptive_policy_lost_cases += 1,
                        _ => {}
                    }
                    let adaptive_violated = !cost_non_decreasing(adaptive_before, adaptive_after);
                    if adaptive_violated {
                        report.adaptive_monotonicity_violations += 1;
                    }
                    if adaptive_before.is_none() && adaptive_after.is_some() {
                        report.adaptive_policy_resurrections += 1;
                    }

                    if fixed_before.is_some() {
                        report.fixed_feasible_base_cases += 1;
                    }
                    if fixed_after.is_some() {
                        report.fixed_feasible_expanded_cases += 1;
                    }
                    match (fixed_before, fixed_after) {
                        (Some(before), Some(after)) if after == before => {
                            report.fixed_cost_unchanged_cases += 1;
                        }
                        (Some(before), Some(after)) if after > before => {
                            report.fixed_cost_increased_cases += 1;
                        }
                        (Some(_), None) => report.fixed_plan_lost_cases += 1,
                        _ => {}
                    }
                    let fixed_violated = !cost_non_decreasing(fixed_before, fixed_after);
                    if fixed_violated {
                        report.fixed_monotonicity_violations += 1;
                    }
                    if fixed_before.is_none() && fixed_after.is_some() {
                        report.fixed_plan_resurrections += 1;
                    }

                    if (adaptive_violated || fixed_violated)
                        && report.counterexamples.len() < MAX_RETAINED_COUNTEREXAMPLES
                    {
                        report.counterexamples.push(Counterexample {
                            base_world_0_support_mask: base_0,
                            expanded_world_0_support_mask: expanded_0,
                            base_world_1_support_mask: base_1,
                            expanded_world_1_support_mask: expanded_1,
                            adaptive_cost_before: adaptive_before,
                            adaptive_cost_after: adaptive_after,
                            fixed_cost_before: fixed_before,
                            fixed_cost_after: fixed_after,
                            adaptive_monotonicity_violated: adaptive_violated,
                            fixed_monotonicity_violated: fixed_violated,
                        });
                    }
                }
            }
        }
    }

    report.monotonicity_passed = report.ordered_support_inclusion_pairs_checked == 4_225
        && report.adaptive_monotonicity_violations == 0
        && report.fixed_monotonicity_violations == 0
        && report.adaptive_policy_resurrections == 0
        && report.fixed_plan_resurrections == 0;
    report.status = if report.monotonicity_passed {
        "PASS_EXHAUSTIVE_SUPPORT_INCLUSION_LATTICE"
    } else {
        "FAIL_SUPPORT_EXPANSION_MONOTONICITY"
    };
    report.counterexamples_retained = report.counterexamples.len();
    report
}

fn main() -> ExitCode {
    let args = Args::parse();
    let report = evaluate();
    let rendered = match serde_json::to_string_pretty(&report) {
        Ok(json) => json + "\n",
        Err(error) => {
            eprintln!("connectome-support-expansion-lattice-audit: report serialization failed: {error}");
            return ExitCode::from(2);
        }
    };
    if let Some(path) = args.output {
        if let Err(error) = fs::write(path, rendered.as_bytes()) {
            eprintln!("connectome-support-expansion-lattice-audit: report write failed: {error}");
            return ExitCode::from(2);
        }
    } else {
        print!("{rendered}");
    }
    if report.monotonicity_passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_lattice_checks_every_ordered_inclusion_pair() {
        let report = evaluate();
        assert_eq!(report.ordered_support_inclusion_pairs_checked, 4_225);
        assert!(report.monotonicity_passed);
        assert_eq!(report.adaptive_monotonicity_violations, 0);
        assert_eq!(report.fixed_monotonicity_violations, 0);
        assert_eq!(report.adaptive_policy_resurrections, 0);
        assert_eq!(report.fixed_plan_resurrections, 0);
        assert!(report.counterexamples.is_empty());
    }

    #[test]
    fn each_nonempty_support_has_at_least_one_superset_and_full_is_unique() {
        assert_eq!(support_supersets(WORLD_SUPPORT_MASKS), vec![WORLD_SUPPORT_MASKS]);
        for mask in 1..=WORLD_SUPPORT_MASKS {
            assert!(support_supersets(mask).contains(&mask));
        }
    }

    #[test]
    fn adding_a_tuple_is_represented_as_a_strict_mask_superset() {
        for base in 1..=WORLD_SUPPORT_MASKS {
            for expanded in support_supersets(base) {
                assert_eq!(base & !expanded, 0);
            }
        }
    }
}
