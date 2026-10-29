// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent raw-input oracle for the delay-aware recovery search.

use symthaea_operations_research::aviation_retiming_recovery::{
    solve_retiming_branch_and_bound, verify_retiming_plan, RetimingAircraft,
    RetimingAssignment, RetimingFlight, RetimingPlan, RetimingScenario,
    RetimingSearchOptions, RetimingSearchStatus, RetimingVerificationStatus,
};

fn scenario() -> RetimingScenario {
    RetimingScenario {
        id: "retiming-fixture".into(),
        configuration_revision: "r1".into(),
        input_digest: "synthetic-retiming-input-v1".into(),
        flights: vec![
            RetimingFlight {
                id: "F1".into(),
                scheduled_start_minutes: 10,
                duration_minutes: 10,
                max_delay_minutes: 20,
                delay_cost_per_minute: 1,
                cancellation_cost: 100,
            },
            RetimingFlight {
                id: "F2".into(),
                scheduled_start_minutes: 10,
                duration_minutes: 10,
                max_delay_minutes: 20,
                delay_cost_per_minute: 1,
                cancellation_cost: 100,
            },
        ],
        aircraft: vec![RetimingAircraft {
            id: "A1".into(),
            available_at_minutes: 0,
            turnaround_minutes: 5,
        }],
    }
}

fn options() -> RetimingSearchOptions {
    RetimingSearchOptions {
        max_expanded_nodes: 100_000,
        max_generated_nodes: 100_000,
        max_absolute_optimality_gap: 0,
    }
}

/// Independent brute-force evaluator. It calculates retimed intervals and
/// costs directly from raw scenario fields, without calling production helpers.
fn oracle_best_cost(scenario: &RetimingScenario) -> Option<u64> {
    let mut choices = vec![None; scenario.flights.len()];
    let mut best = None;
    oracle_enumerate(scenario, 0, &mut choices, &mut best);
    best
}

fn oracle_enumerate(
    scenario: &RetimingScenario,
    flight_index: usize,
    choices: &mut [Option<(usize, u32)>],
    best: &mut Option<u64>,
) {
    if flight_index == scenario.flights.len() {
        let Some(cost) = oracle_evaluate(scenario, choices) else {
            return;
        };
        if best.as_ref().is_none_or(|current| cost < *current) {
            *best = Some(cost);
        }
        return;
    }

    let flight = &scenario.flights[flight_index];
    for aircraft_index in 0..scenario.aircraft.len() {
        for delay in 0..=flight.max_delay_minutes {
            choices[flight_index] = Some((aircraft_index, delay));
            oracle_enumerate(scenario, flight_index + 1, choices, best);
        }
    }
    choices[flight_index] = None;
    oracle_enumerate(scenario, flight_index + 1, choices, best);
}

fn oracle_evaluate(
    scenario: &RetimingScenario,
    choices: &[Option<(usize, u32)>],
) -> Option<u64> {
    if choices.len() != scenario.flights.len() {
        return None;
    }
    let mut intervals: Vec<Vec<(u32, u32)>> = vec![Vec::new(); scenario.aircraft.len()];
    let mut cost = 0_u64;
    for (index, choice) in choices.iter().enumerate() {
        let flight = &scenario.flights[index];
        match choice {
            None => {
                cost = cost.checked_add(flight.cancellation_cost)?;
            }
            Some((aircraft_index, delay)) => {
                if *delay > flight.max_delay_minutes {
                    return None;
                }
                let aircraft = scenario.aircraft.get(*aircraft_index)?;
                let start = flight.scheduled_start_minutes.checked_add(*delay)?;
                let end = start.checked_add(flight.duration_minutes)?;
                if start < aircraft.available_at_minutes {
                    return None;
                }
                intervals[*aircraft_index].push((start, end));
                cost = cost.checked_add(
                    flight.delay_cost_per_minute.checked_mul(u64::from(*delay))?,
                )?;
            }
        }
    }

    for (aircraft_index, assigned) in intervals.iter_mut().enumerate() {
        assigned.sort_unstable();
        for pair in assigned.windows(2) {
            if pair[0].1.checked_add(scenario.aircraft[aircraft_index].turnaround_minutes)? > pair[1].0 {
                return None;
            }
        }
    }
    Some(cost)
}

fn varied_scenarios() -> Vec<RetimingScenario> {
    let mut cases = Vec::new();
    for seed in 0..12_u32 {
        let mut s = scenario();
        s.id = format!("retiming-seed-{seed}");
        s.configuration_revision = format!("r{seed}");
        s.input_digest = format!("input-{seed}");
        s.flights[0].scheduled_start_minutes = 10 + seed % 3;
        s.flights[0].duration_minutes = 6 + seed % 4;
        s.flights[0].max_delay_minutes = seed % 4;
        s.flights[0].delay_cost_per_minute = 1 + u64::from(seed % 5);
        s.flights[0].cancellation_cost = 20 + u64::from(seed) * 3;
        s.flights[1].scheduled_start_minutes = 11 + (seed % 4);
        s.flights[1].duration_minutes = 7 + seed % 3;
        s.flights[1].max_delay_minutes = (seed + 1) % 4;
        s.flights[1].delay_cost_per_minute = 2 + u64::from(seed % 3);
        s.flights[1].cancellation_cost = 25 + u64::from(seed) * 2;
        s.aircraft[0].available_at_minutes = seed % 4;
        s.aircraft[0].turnaround_minutes = 1 + seed % 5;
        if seed % 2 == 0 {
            s.aircraft.push(RetimingAircraft {
                id: "A2".into(),
                available_at_minutes: seed % 5,
                turnaround_minutes: 2 + seed % 4,
            });
            s.flights.push(RetimingFlight {
                id: "F3".into(),
                scheduled_start_minutes: 20 + seed % 6,
                duration_minutes: 5 + seed % 5,
                max_delay_minutes: seed % 3,
                delay_cost_per_minute: 1 + u64::from(seed % 4),
                cancellation_cost: 35 + u64::from(seed) * 3,
            });
        }
        cases.push(s);
    }
    cases
}

#[test]
fn retiming_search_finds_the_minimum_delay_schedule() {
    let s = scenario();
    let expected = oracle_best_cost(&s);
    assert_eq!(expected, Some(15));
    let report = solve_retiming_branch_and_bound(&s, options()).unwrap();
    assert_eq!(report.status, RetimingSearchStatus::OptimalWithinTolerance);
    assert_eq!(report.best_objective_cost, expected);
    assert_eq!(report.global_lower_bound_cost, expected);
    assert_eq!(report.absolute_optimality_gap, Some(0));

    let plan = report.best_plan.as_ref().expect("a verified plan exists");
    let verification = verify_retiming_plan(&s, plan);
    assert_eq!(verification.status, RetimingVerificationStatus::Verified);
    assert_eq!(verification.recomputed_cost, expected);
    assert_eq!(plan.assignments.iter().filter(|a| a.aircraft_id.is_none()).count(), 0);
    assert!(plan.assignments.iter().any(|a| a.delay_minutes >= 15));
}

#[test]
fn branch_and_bound_matches_independent_raw_oracle_across_varied_scenarios() {
    for s in varied_scenarios() {
        let expected = oracle_best_cost(&s).expect("cancelling all flights is always feasible");
        let report = solve_retiming_branch_and_bound(&s, options())
            .unwrap_or_else(|error| panic!("{}: {error}", s.id));
        assert_eq!(report.best_objective_cost, Some(expected), "scenario {}", s.id);
        assert_eq!(
            report.status,
            RetimingSearchStatus::OptimalWithinTolerance,
            "scenario {}",
            s.id
        );
        let plan = report.best_plan.as_ref().expect("feasible oracle result");
        let verified = verify_retiming_plan(&s, plan);
        assert_eq!(verified.status, RetimingVerificationStatus::Verified, "{}", s.id);
        assert_eq!(verified.recomputed_cost, Some(expected), "{}", s.id);
        assert_eq!(report.absolute_optimality_gap, Some(0), "{}", s.id);
    }
}

#[test]
fn generation_budget_preserves_the_unexpanded_subtree_and_makes_no_claim() {
    let s = scenario();
    let report = solve_retiming_branch_and_bound(
        &s,
        RetimingSearchOptions {
            max_expanded_nodes: 100,
            max_generated_nodes: 1,
            max_absolute_optimality_gap: 0,
        },
    )
    .unwrap();
    assert_eq!(report.status, RetimingSearchStatus::NoFeasiblePlanWithinBudget);
    assert_eq!(report.best_plan, None);
    assert_eq!(report.stats.generated_nodes, 1);
    assert_eq!(report.stats.expanded_nodes, 0);
    assert_eq!(report.global_lower_bound_cost, Some(0));
}

#[test]
fn one_expansion_budget_does_not_claim_optimality() {
    let s = scenario();
    let report = solve_retiming_branch_and_bound(
        &s,
        RetimingSearchOptions {
            max_expanded_nodes: 1,
            max_generated_nodes: 10_000,
            max_absolute_optimality_gap: 0,
        },
    )
    .unwrap();
    assert_ne!(report.status, RetimingSearchStatus::OptimalWithinTolerance);
    assert!(report.stats.expanded_nodes <= 1);
    if let Some(plan) = &report.best_plan {
        assert_eq!(verify_retiming_plan(&s, plan).status, RetimingVerificationStatus::Verified);
    }
}

#[test]
fn verifier_rejects_over_delay_and_tampered_cost() {
    let s = scenario();
    let invalid_delay = RetimingPlan {
        scenario_id: s.id.clone(),
        configuration_revision: s.configuration_revision.clone(),
        input_digest: s.input_digest.clone(),
        assignments: vec![
            RetimingAssignment {
                flight_id: "F1".into(),
                aircraft_id: Some("A1".into()),
                delay_minutes: 21,
            },
            RetimingAssignment {
                flight_id: "F2".into(),
                aircraft_id: None,
                delay_minutes: 0,
            },
        ],
        total_cost: 100,
    };
    assert_eq!(
        verify_retiming_plan(&s, &invalid_delay).status,
        RetimingVerificationStatus::Rejected
    );

    let mut valid = solve_retiming_branch_and_bound(&s, options())
        .unwrap()
        .best_plan
        .unwrap();
    valid.total_cost = valid.total_cost.saturating_add(1);
    assert_eq!(
        verify_retiming_plan(&s, &valid).status,
        RetimingVerificationStatus::Rejected
    );
}

#[test]
fn verifier_rejects_overlapping_assignments_and_cancelled_delay() {
    let mut s = scenario();
    let mut invalid = RetimingPlan {
        scenario_id: s.id.clone(),
        configuration_revision: s.configuration_revision.clone(),
        input_digest: s.input_digest.clone(),
        assignments: vec![
            RetimingAssignment {
                flight_id: "F1".into(),
                aircraft_id: Some("A1".into()),
                delay_minutes: 0,
            },
            RetimingAssignment {
                flight_id: "F2".into(),
                aircraft_id: Some("A1".into()),
                delay_minutes: 0,
            },
        ],
        total_cost: 0,
    };
    assert_eq!(
        verify_retiming_plan(&s, &invalid).status,
        RetimingVerificationStatus::Rejected
    );

    invalid.assignments[1].aircraft_id = None;
    invalid.assignments[1].delay_minutes = 1;
    invalid.total_cost = 100;
    assert_eq!(
        verify_retiming_plan(&s, &invalid).status,
        RetimingVerificationStatus::Rejected
    );

    s.configuration_revision = "r2".into();
    assert_eq!(
        verify_retiming_plan(&s, &invalid).status,
        RetimingVerificationStatus::Rejected
    );
}

#[test]
fn scenario_validation_rejects_duplicate_ids_and_time_overflow() {
    let mut duplicate = scenario();
    duplicate.flights[1].id = duplicate.flights[0].id.clone();
    assert!(duplicate.validate().is_err());

    let mut overflow = scenario();
    overflow.flights[0].scheduled_start_minutes = u32::MAX - 3;
    overflow.flights[0].duration_minutes = 10;
    assert!(overflow.validate().is_err());
}

#[test]
fn deterministic_search_and_bad_budgets() {
    let s = scenario();
    assert_eq!(
        solve_retiming_branch_and_bound(&s, options()).unwrap(),
        solve_retiming_branch_and_bound(&s, options()).unwrap()
    );
    assert!(solve_retiming_branch_and_bound(
        &s,
        RetimingSearchOptions {
            max_expanded_nodes: 0,
            max_generated_nodes: 10,
            max_absolute_optimality_gap: 0,
        }
    )
    .is_err());
}
