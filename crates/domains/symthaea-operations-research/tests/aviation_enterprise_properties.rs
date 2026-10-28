// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Metamorphic/property tests for resource, evidence, and cost monotonicity.
//!
//! These tests exercise a bounded synthetic model, not real operational safety.

use symthaea_operations_research::aviation_enterprise::{
    reference_scenario, solve_exact, PartSource, VerificationStatus, verify_plan,
};

#[test]
fn loss_of_any_production_prerequisite_never_improves_optimal_cost() {
    let baseline_scenario = reference_scenario();
    let baseline = solve_exact(&baseline_scenario).expect("baseline remains feasible");
    assert_eq!(baseline.total_cost, 40);

    for case in 0..5 {
        let mut degraded = baseline_scenario.clone();
        match case {
            0 => degraded.production.available = false,
            1 => degraded.production.qualified_worker = false,
            2 => degraded.production.capable_tool = false,
            3 => degraded.production.matches_configuration = false,
            4 => degraded.production.evidence_current = false,
            _ => unreachable!(),
        }
        let plan = solve_exact(&degraded).expect("cancellation keeps this case feasible");
        assert!(
            plan.total_cost >= baseline.total_cost,
            "degrading production prerequisite {case} improved cost from {} to {}",
            baseline.total_cost,
            plan.total_cost
        );
        assert_eq!(verify_plan(&degraded, &plan).status, VerificationStatus::Verified);
    }
}

#[test]
fn loss_of_stock_qualification_never_improves_optimal_cost() {
    let mut available = reference_scenario();
    available.qualified_stock_count = 1;
    let baseline = solve_exact(&available).expect("qualified stock case is feasible");
    assert_eq!(baseline.part_source, PartSource::QualifiedStock);
    assert_eq!(baseline.total_cost, 5);

    let mut degraded_cases = Vec::new();
    let mut no_stock = available.clone();
    no_stock.qualified_stock_count = 0;
    degraded_cases.push(no_stock);
    let mut stale = available.clone();
    stale.stock_evidence_current = false;
    degraded_cases.push(stale);
    let mut mismatched = available.clone();
    mismatched.stock_matches_configuration = false;
    degraded_cases.push(mismatched);

    for degraded in degraded_cases {
        let plan = solve_exact(&degraded).expect("other choices keep this case feasible");
        assert!(
            plan.total_cost >= baseline.total_cost,
            "loss of qualified stock must not lower objective cost"
        );
        assert_eq!(verify_plan(&degraded, &plan).status, VerificationStatus::Verified);
    }
}

#[test]
fn increasing_production_cost_never_lowers_optimal_cost_across_grid() {
    let mut previous = 0_u64;
    for production_cost in (0_u64..=200).step_by(5) {
        let mut scenario = reference_scenario();
        scenario.production.cost = production_cost;
        let plan = solve_exact(&scenario).expect("there is always a no-production plan");
        assert!(
            plan.total_cost >= previous,
            "optimal cost decreased when production cost rose to {production_cost}: {previous} -> {}",
            plan.total_cost
        );
        assert_eq!(verify_plan(&scenario, &plan).status, VerificationStatus::Verified);
        previous = plan.total_cost;
    }
}

#[test]
fn increasing_repair_duration_never_reduces_optimal_cost() {
    let mut previous = 0_u64;
    for maintenance_duration in (0_u32..=100).step_by(5) {
        let mut scenario = reference_scenario();
        scenario.maintenance_duration = maintenance_duration;
        let plan = solve_exact(&scenario).expect("no-part/cancel plan remains feasible");
        assert!(
            plan.total_cost >= previous,
            "optimal cost decreased when repair duration rose to {maintenance_duration}: {previous} -> {}",
            plan.total_cost
        );
        assert_eq!(verify_plan(&scenario, &plan).status, VerificationStatus::Verified);
        previous = plan.total_cost;
    }
}

#[test]
fn evidence_degradation_cannot_turn_rejection_into_success() {
    let scenario = reference_scenario();
    let mut plan = solve_exact(&scenario).expect("baseline feasible");
    plan.part_source = PartSource::QualifiedProduction;
    let mut degraded = scenario.clone();
    degraded.production.evidence_current = false;
    let report = verify_plan(&degraded, &plan);
    assert_eq!(report.status, VerificationStatus::Rejected);
    assert!(report.violations.iter().any(|item| item.contains("part source")));
}
