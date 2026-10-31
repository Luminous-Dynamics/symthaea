// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preflight Pareto frontier for exact cutset choices on square binary grids.
//!
//! Run with:
//! cargo run --release --locked -p symthaea-iit --example cutset_frontier_sweep
//!
//! This ranks a declared family of exact separator candidates before any
//! conditioned branch is evaluated. It reports the non-dominated frontier over
//! branch count, residual induced width, and predicted peak factor-table entries.
//! It is not an optimal-cutset solver and does not report inference latency.

use std::collections::{BTreeMap, BTreeSet};

use symthaea_iit::{
    plan_min_fill_cutset_elimination, BinaryLogFactor, CutsetFactorEliminationPlan,
};

const FACTOR_ENTRY_BUDGET: usize = 1 << 16;
const CUTSET_ASSIGNMENT_BUDGET: usize = 1 << 16;
const GRID_SIDES: [usize; 4] = [4, 5, 6, 8];

#[derive(Debug, Clone)]
struct PlannedCutset {
    label: String,
    variables: Vec<usize>,
    branch_count: usize,
    residual_induced_width: usize,
    predicted_peak_factor_entries: usize,
}

fn grid_factors(side: usize) -> Result<Vec<BinaryLogFactor>, String> {
    const PAIR_POTENTIAL: [f64; 4] = [0.83, 0.17, 0.29, 0.71];
    let log_values = PAIR_POTENTIAL.iter().map(|value| value.ln()).collect::<Vec<_>>();
    let mut factors = Vec::new();
    for row in 0..side {
        for column in 0..side {
            let here = row * side + column;
            if column + 1 < side {
                factors.push(
                    BinaryLogFactor::new(vec![here, here + 1], log_values.clone())
                        .map_err(|error| error.to_string())?,
                );
            }
            if row + 1 < side {
                factors.push(
                    BinaryLogFactor::new(vec![here, here + side], log_values.clone())
                        .map_err(|error| error.to_string())?,
                );
            }
        }
    }
    Ok(factors)
}

fn insert_candidate(
    candidates: &mut BTreeMap<Vec<usize>, String>,
    label: String,
    mut variables: Vec<usize>,
    retained: usize,
) {
    variables.sort_unstable();
    variables.dedup();
    if variables.is_empty() || variables.contains(&retained) {
        return;
    }
    candidates.entry(variables).or_insert(label);
}

fn candidate_cutsets(side: usize, retained: usize) -> Vec<(String, Vec<usize>)> {
    let mut candidates = BTreeMap::<Vec<usize>, String>::new();

    // Every single horizontal and vertical separator.
    for row in 0..side {
        insert_candidate(
            &mut candidates,
            format!("row-{row}"),
            (0..side).map(|column| row * side + column).collect(),
            retained,
        );
    }
    for column in 0..side {
        insert_candidate(
            &mut candidates,
            format!("column-{column}"),
            (0..side).map(|row| row * side + column).collect(),
            retained,
        );
    }

    // Width-two bands test whether more conditioning can substantially lower
    // the residual elimination width.
    for first_row in 0..side.saturating_sub(1) {
        let variables = (first_row..first_row + 2)
            .flat_map(|row| (0..side).map(move |column| row * side + column))
            .collect();
        insert_candidate(
            &mut candidates,
            format!("row-band-{first_row}-{}", first_row + 1),
            variables,
            retained,
        );
    }
    for first_column in 0..side.saturating_sub(1) {
        let variables = (0..side)
            .flat_map(|row| (first_column..first_column + 2).map(move |column| row * side + column))
            .collect();
        insert_candidate(
            &mut candidates,
            format!("column-band-{first_column}-{}", first_column + 1),
            variables,
            retained,
        );
    }

    // Cross separators (one full row union one full column).
    for row in 0..side {
        for column in 0..side {
            let mut variables = (0..side)
                .map(|cell_column| row * side + cell_column)
                .collect::<Vec<_>>();
            variables.extend((0..side).map(|cell_row| cell_row * side + column));
            insert_candidate(
                &mut candidates,
                format!("cross-{row}-{column}"),
                variables,
                retained,
            );
        }
    }

    candidates
        .into_iter()
        .map(|(variables, label)| (label, variables))
        .collect()
}

fn preflight_candidate(
    factor_scopes: &[Vec<usize>],
    eliminate: &[usize],
    label: String,
    variables: Vec<usize>,
) -> Result<PlannedCutset, String> {
    let plan: CutsetFactorEliminationPlan = plan_min_fill_cutset_elimination(
        factor_scopes,
        eliminate,
        &variables,
        FACTOR_ENTRY_BUDGET,
        CUTSET_ASSIGNMENT_BUDGET,
    )
    .map_err(|error| error.to_string())?;

    Ok(PlannedCutset {
        label,
        variables,
        branch_count: plan.cutset_assignments(),
        residual_induced_width: plan.remaining_elimination_plan().induced_width,
        predicted_peak_factor_entries: plan.predicted_peak_factor_entries(),
    })
}

// A dominates B only when it is no worse on all three resources and strictly
// better on at least one. Exact metric ties use the lexical label as a stable
// representative so symmetric separators do not flood the frontier.
fn dominates(left: &PlannedCutset, right: &PlannedCutset) -> bool {
    let no_worse = left.branch_count <= right.branch_count
        && left.residual_induced_width <= right.residual_induced_width
        && left.predicted_peak_factor_entries <= right.predicted_peak_factor_entries;
    let strictly_better = left.branch_count < right.branch_count
        || left.residual_induced_width < right.residual_induced_width
        || left.predicted_peak_factor_entries < right.predicted_peak_factor_entries;
    let exact_tie_precedes = left.branch_count == right.branch_count
        && left.residual_induced_width == right.residual_induced_width
        && left.predicted_peak_factor_entries == right.predicted_peak_factor_entries
        && left.label < right.label;
    no_worse && (strictly_better || exact_tie_precedes)
}

fn pareto_frontier(plans: &[PlannedCutset]) -> Vec<usize> {
    plans
        .iter()
        .enumerate()
        .filter(|(index, candidate)| {
            !plans
                .iter()
                .enumerate()
                .any(|(other_index, other)| other_index != *index && dominates(other, candidate))
        })
        .map(|(index, _)| index)
        .collect()
}

fn validate_frontier(plans: &[PlannedCutset], frontier: &[usize]) -> Result<(), String> {
    let frontier_set = frontier.iter().copied().collect::<BTreeSet<_>>();
    for (index, candidate) in plans.iter().enumerate() {
        let represented = if frontier_set.contains(&index) {
            true
        } else {
            frontier.iter().any(|frontier_index| {
                dominates(&plans[*frontier_index], candidate)
            })
        };
        if !represented {
            return Err(format!(
                "candidate {} is neither on the frontier nor dominated by it",
                candidate.label
            ));
        }
    }
    for index in frontier {
        if plans.iter().enumerate().any(|(other_index, other)| {
            other_index != *index && dominates(other, &plans[*index])
        }) {
            return Err(format!(
                "frontier candidate {} is dominated",
                plans[*index].label
            ));
        }
    }
    Ok(())
}

fn run() -> Result<(), String> {
    println!(
        "side,candidate_count,feasible_count,rejected_count,frontier_count,label,cutset_size,cutset_variables,branch_count,residual_induced_width,predicted_peak_factor_entries,on_pareto_frontier"
    );

    for side in GRID_SIDES {
        let factors = grid_factors(side)?;
        let scopes = factors
            .iter()
            .map(|factor| factor.variables().to_vec())
            .collect::<Vec<_>>();
        let variable_count = side * side;
        let retained = variable_count - 1;
        let eliminate = (0..variable_count)
            .filter(|variable| *variable != retained)
            .collect::<Vec<_>>();

        let candidates = candidate_cutsets(side, retained);
        let candidate_count = candidates.len();
        let mut plans = Vec::new();
        let mut rejected_count = 0usize;
        for (label, variables) in candidates {
            match preflight_candidate(&scopes, &eliminate, label, variables) {
                Ok(plan) => plans.push(plan),
                Err(_) => rejected_count += 1,
            }
        }
        if plans.is_empty() {
            return Err(format!("no feasible cutset candidates for {side}x{side}"));
        }

        plans.sort_by(|left, right| {
            left.branch_count
                .cmp(&right.branch_count)
                .then(left.predicted_peak_factor_entries.cmp(&right.predicted_peak_factor_entries))
                .then(left.residual_induced_width.cmp(&right.residual_induced_width))
                .then(left.label.cmp(&right.label))
        });
        let frontier = pareto_frontier(&plans);
        validate_frontier(&plans, &frontier)?;
        let frontier_set = frontier.into_iter().collect::<BTreeSet<_>>();

        for (index, plan) in plans.iter().enumerate() {
            let variables = plan
                .variables
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(":");
            println!(
                "{},{},{},{},{},{},{},\"{}\",{},{},{},{}",
                side,
                candidate_count,
                plans.len(),
                rejected_count,
                frontier_set.len(),
                plan.label,
                plan.variables.len(),
                variables,
                plan.branch_count,
                plan.residual_induced_width,
                plan.predicted_peak_factor_entries,
                frontier_set.contains(&index),
            );
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("cutset frontier sweep failed: {error}");
        std::process::exit(1);
    }
}
