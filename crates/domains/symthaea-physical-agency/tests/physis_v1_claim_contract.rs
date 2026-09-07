// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public PHYSIS v1 claim-preflight contract.
//!
//! These tests exercise normalized claim identity/satisfiability only. They do
//! not execute a solver and grant no physical authority.

use symthaea_physical_agency::{
    ConfirmatoryContractError, MetricCriterion, MetricPredicate, MetricUncertaintyPolicy,
    SimulationOutcomeClaim, canonical_claim_transcript,
};

fn criterion(metric: &str, predicate: MetricPredicate) -> MetricCriterion {
    MetricCriterion {
        metric_name: metric.into(),
        unit: "1".into(),
        predicate,
        uncertainty_policy: MetricUncertaintyPolicy::RequireInterval,
    }
}

fn claim(criteria: Vec<MetricCriterion>) -> SimulationOutcomeClaim {
    SimulationOutcomeClaim::all_criteria(
        "physis-v1-preflight",
        "physis-v1-transition",
        "physis-v1-acoustic",
        criteria,
    )
}

#[test]
fn contradictory_bounds_fail_before_any_solver_run() {
    let impossible = claim(vec![
        criterion("diagnostic_quality", MetricPredicate::AtLeast(0.9)),
        criterion("diagnostic_quality", MetricPredicate::AtMost(0.8)),
    ]);

    assert!(matches!(
        canonical_claim_transcript(&impossible),
        Err(ConfirmatoryContractError::UnsatisfiableClaim { metric_name, unit })
            if metric_name == "diagnostic_quality" && unit == "1"
    ));
}

#[test]
fn disjunctive_outside_constraint_can_make_claim_impossible() {
    let impossible = claim(vec![
        criterion(
            "diagnostic_quality",
            MetricPredicate::InsideClosedInterval {
                lower: 0.3,
                upper: 0.7,
            },
        ),
        criterion(
            "diagnostic_quality",
            MetricPredicate::OutsideOpenInterval {
                lower: 0.2,
                upper: 0.8,
            },
        ),
    ]);

    assert!(matches!(
        canonical_claim_transcript(&impossible),
        Err(ConfirmatoryContractError::UnsatisfiableClaim { .. })
    ));
}

#[test]
fn closed_boundary_points_remain_valid_outside_open_interval() {
    let boundary_feasible = claim(vec![
        criterion(
            "diagnostic_quality",
            MetricPredicate::InsideClosedInterval {
                lower: 0.2,
                upper: 0.8,
            },
        ),
        criterion(
            "diagnostic_quality",
            MetricPredicate::OutsideOpenInterval {
                lower: 0.2,
                upper: 0.8,
            },
        ),
    ]);

    assert!(canonical_claim_transcript(&boundary_feasible).is_ok());
}
