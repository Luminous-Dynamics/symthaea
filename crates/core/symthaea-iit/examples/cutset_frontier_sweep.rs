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
    plan_min_fill_cutset_frontier, BinaryLogFactor,
};

const FACTOR_ENTRY_BUDGET: usize = 1 << 16;
const CUTSET_ASSIGNMENT_BUDGET: usize = 1 << 16;

#[derive(Debug, Clone, Copy)]
struct ExpectedFrontier {
    label: &'static str,
    branches: usize,
    residual_width: usize,
    peak_entries: usize,
}

struct GridFixture {
    side: usize,
    expected_candidates: usize,
    expected_frontier: &'static [ExpectedFrontier],
}

// These fixture expectations are independently derived from the grid primal
// graphs and the deterministic min-fill objective. They pin both the candidate
// family size and its three-resource nondominated frontier.
const GRID_FIXTURES: [GridFixture; 4] = [
    GridFixture {
        side: 4,
        expected_candidates: 19,
        expected_frontier: &[
            ExpectedFrontier { label: "column-1", branches: 16, residual_width: 2, peak_entries: 8 },
            ExpectedFrontier { label: "column-band-1-2", branches: 256, residual_width: 1, peak_entries: 4 },
        ],
    },
    GridFixture {
        side: 5,
        expected_candidates: 30,
        expected_frontier: &[
            ExpectedFrontier { label: "column-2", branches: 32, residual_width: 2, peak_entries: 8 },
        ],
    },
    GridFixture {
        side: 6,
        expected_candidates: 43,
        expected_frontier: &[
            ExpectedFrontier { label: "column-2", branches: 64, residual_width: 3, peak_entries: 16 },
            ExpectedFrontier { label: "column-band-2-3", branches: 4_096, residual_width: 2, peak_entries: 8 },
        ],
    },
    GridFixture {
        side: 8,
        expected_candidates: 75,
        expected_frontier: &[
            ExpectedFrontier { label: "column-3", branches: 256, residual_width: 4, peak_entries: 32 },
            ExpectedFrontier { label: "column-band-3-4", branches: 65_536, residual_width: 3, peak_entries: 16 },
        ],
    },
];

#[derive(Debug, Clone)]
struct PlannedCutset {
    candidate_index: usize,
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

fn run() -> Result<(), String> {
    println!(
        "side,candidate_count,feasible_count,rejected_count,frontier_count,label,cutset_size,cutset_variables,branch_count,residual_induced_width,predicted_peak_factor_entries,on_pareto_frontier"
    );

    for fixture in GRID_FIXTURES {
        let side = fixture.side;
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
        if candidate_count != fixture.expected_candidates {
            return Err(format!(
                "candidate-family count changed for {side}x{side}: expected {}, got {candidate_count}",
                fixture.expected_candidates
            ));
        }
        let candidate_cutsets = candidates
            .iter()
            .map(|(_, variables)| variables.clone())
            .collect::<Vec<_>>();
        let report = plan_min_fill_cutset_frontier(
            &scopes,
            &eliminate,
            &candidate_cutsets,
            FACTOR_ENTRY_BUDGET,
            CUTSET_ASSIGNMENT_BUDGET,
        ).map_err(|error| format!("frontier preflight failed for {side}x{side}: {error}"))?;
        let rejected_count = report.rejected_candidates.len();
        let mut plans = report.candidates.iter().map(|candidate| {
            let (label, _) = &candidates[candidate.candidate_index];
            PlannedCutset {
                candidate_index: candidate.candidate_index,
                label: label.clone(),
                variables: candidate.plan.cutset_variables().to_vec(),
                branch_count: candidate.plan.cutset_assignments(),
                residual_induced_width: candidate.plan.remaining_elimination_plan().induced_width,
                predicted_peak_factor_entries: candidate.plan.predicted_peak_factor_entries(),
            }
        }).collect::<Vec<_>>();
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
        let frontier_set = report.frontier_candidate_indices
            .into_iter()
            .collect::<BTreeSet<_>>();
        if frontier_set.len() != fixture.expected_frontier.len() {
            return Err(format!(
                "Pareto frontier count changed for {side}x{side}: expected {}, got {}",
                fixture.expected_frontier.len(),
                frontier_set.len()
            ));
        }
        let actual_frontier = plans.iter()
            .filter(|plan| frontier_set.contains(&plan.candidate_index))
            .collect::<Vec<_>>();
        for (actual, expected) in actual_frontier.iter().zip(fixture.expected_frontier) {
            if actual.label != expected.label
                || actual.branch_count != expected.branches
                || actual.residual_induced_width != expected.residual_width
                || actual.predicted_peak_factor_entries != expected.peak_entries
            {
                return Err(format!(
                    "Pareto frontier regression for {side}x{side}: expected {} ({}, {}, {}), got {} ({}, {}, {})",
                    expected.label,
                    expected.branches,
                    expected.residual_width,
                    expected.peak_entries,
                    actual.label,
                    actual.branch_count,
                    actual.residual_induced_width,
                    actual.predicted_peak_factor_entries
                ));
            }
        }

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
                frontier_set.contains(&plan.candidate_index),
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
