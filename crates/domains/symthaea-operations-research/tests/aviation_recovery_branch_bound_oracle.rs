// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-checks for the anytime branch-and-bound recovery search.
//!
//! Expected costs are independently fixed by the tiny reference scenarios.
//! Every returned incumbent is also run through the stable-ID plan verifier.

use symthaea_operations_research::aviation_enterprise::{
    reference_scenario, solve_branch_and_bound, solve_exact, verify_plan,
    BranchAndBoundOptions, BranchAndBoundStatus, PartSource, VerificationStatus,
};

fn exact_options() -> BranchAndBoundOptions {
    BranchAndBoundOptions {
        max_expanded_nodes: 100_000,
        max_absolute_optimality_gap: 0,
    }
}

#[test]
fn branch_and_bound_matches_reference_optimum_and_verifier() {
    let scenario = reference_scenario();
    let exact = solve_exact(&scenario).expect("exact reference solver is feasible");
    let report = solve_branch_and_bound(&scenario, exact_options()).unwrap();

    assert_eq!(report.status, BranchAndBoundStatus::OptimalWithinTolerance);
    assert_eq!(report.best_objective_cost, Some(exact.total_cost));
    assert_eq!(report.best_objective_cost, Some(40));
    assert_eq!(report.global_lower_bound_cost, Some(40));
    assert_eq!(report.absolute_optimality_gap, Some(0));

    let plan = report.best_plan.as_ref().expect("a feasible incumbent is reported");
    assert_eq!(verify_plan(&scenario, plan).status, VerificationStatus::Verified);
    assert_eq!(verify_plan(&scenario, plan).recomputed_cost, Some(40));

    // There are two eligible source choices and 3^2 assignment/cancellation
    // combinations per source in this tiny scenario. The bounded search should
    // prove the reference optimum without visiting all 18 complete assignments.
    assert!(report.stats.complete_assignments_examined < 18);
    assert!(report.stats.generated_nodes > 0);
    assert!(report.stats.expanded_nodes > 0);
}

#[test]
fn branch_and_bound_respects_stock_and_production_evidence() {
    let mut scenario = reference_scenario();
    scenario.qualified_stock_count = 1;
    let stock_report = solve_branch_and_bound(&scenario, exact_options()).unwrap();
    assert_eq!(stock_report.best_objective_cost, Some(5));
    assert_eq!(
        stock_report.best_plan.as_ref().unwrap().part_source,
        PartSource::QualifiedStock
    );

    let mut stale_production = reference_scenario();
    stale_production.production.evidence_current = false;
    let stale_report = solve_branch_and_bound(&stale_production, exact_options()).unwrap();
    assert_eq!(stale_report.best_objective_cost, Some(100));
    assert_ne!(
        stale_report.best_plan.as_ref().unwrap().part_source,
        PartSource::QualifiedProduction
    );
    assert_eq!(
        verify_plan(&stale_production, stale_report.best_plan.as_ref().unwrap()).status,
        VerificationStatus::Verified
    );
}

#[test]
fn one_node_budget_does_not_claim_a_solution_or_optimality() {
    let scenario = reference_scenario();
    let report = solve_branch_and_bound(
        &scenario,
        BranchAndBoundOptions {
            max_expanded_nodes: 1,
            max_absolute_optimality_gap: 0,
        },
    )
    .unwrap();

    assert_eq!(
        report.status,
        BranchAndBoundStatus::NoFeasiblePlanWithinBudget
    );
    assert_eq!(report.best_plan, None);
    assert_eq!(report.best_objective_cost, None);
    assert_eq!(report.stats.expanded_nodes, 1);
    assert!(report.global_lower_bound_cost.is_some());
}

#[test]
fn deterministic_inputs_and_options_produce_identical_reports() {
    let scenario = reference_scenario();
    let first = solve_branch_and_bound(&scenario, exact_options()).unwrap();
    let second = solve_branch_and_bound(&scenario, exact_options()).unwrap();
    assert_eq!(first, second);
}

#[test]
fn zero_expansion_budget_is_rejected() {
    let scenario = reference_scenario();
    assert!(solve_branch_and_bound(
        &scenario,
        BranchAndBoundOptions {
            max_expanded_nodes: 0,
            max_absolute_optimality_gap: 0,
        }
    )
    .is_err());
}
