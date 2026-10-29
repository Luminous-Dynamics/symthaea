// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-checks for the anytime branch-and-bound recovery search.
//!
//! Expected costs are independently fixed by the tiny reference scenarios.
//! Every returned incumbent is also run through the stable-ID plan verifier.

use symthaea_operations_research::aviation_enterprise::{
    reference_scenario, solve_branch_and_bound, solve_exact, verify_plan, Flight,
    PartSource, Scenario, BranchAndBoundOptions,
    BranchAndBoundStatus, VerificationStatus,
};


/// Independent small-instance oracle. This deliberately enumerates every source
/// and assignment from raw scenario fields instead of calling either solver's
/// feasibility/cost helpers.
fn oracle_optimum(scenario: &Scenario) -> Option<u64> {
    let eligible_sources = [PartSource::None, PartSource::QualifiedStock, PartSource::QualifiedProduction]
        .into_iter()
        .filter(|source| oracle_source_is_eligible(scenario, *source));
    let mut best = None;
    for source in eligible_sources {
        let mut choices = vec![None; scenario.flights.len()];
        oracle_enumerate(scenario, source, 0, &mut choices, &mut best);
    }
    best
}

fn oracle_source_is_eligible(scenario: &Scenario, source: PartSource) -> bool {
    match source {
        PartSource::None => true,
        PartSource::QualifiedStock => {
            scenario.qualified_stock_count > 0
                && scenario.stock_matches_configuration
                && scenario.stock_evidence_current
        }
        PartSource::QualifiedProduction => {
            scenario.production.available
                && scenario.production.qualified_worker
                && scenario.production.capable_tool
                && scenario.production.matches_configuration
                && scenario.production.evidence_current
                && scenario.production.duration.checked_add(scenario.maintenance_duration).is_some()
        }
    }
}

fn oracle_enumerate(
    scenario: &Scenario,
    source: PartSource,
    flight_index: usize,
    choices: &mut [Option<usize>],
    best: &mut Option<u64>,
) {
    if flight_index == scenario.flights.len() {
        let Some(cost) = oracle_evaluate(scenario, source, choices) else {
            return;
        };
        if best.is_none_or(|current| cost < current) {
            *best = Some(cost);
        }
        return;
    }

    for aircraft_index in 0..scenario.aircraft.len() {
        choices[flight_index] = Some(aircraft_index);
        oracle_enumerate(scenario, source, flight_index + 1, choices, best);
    }
    choices[flight_index] = None;
    oracle_enumerate(scenario, source, flight_index + 1, choices, best);
}

fn oracle_evaluate(scenario: &Scenario, source: PartSource, choices: &[Option<usize>]) -> Option<u64> {
    if !oracle_source_is_eligible(scenario, source) || choices.len() != scenario.flights.len() {
        return None;
    }

    let repair_ready = match source {
        PartSource::None => None,
        PartSource::QualifiedStock => Some(scenario.maintenance_duration),
        PartSource::QualifiedProduction => Some(
            scenario.production.duration.checked_add(scenario.maintenance_duration)?,
        ),
    };

    let mut by_aircraft: Vec<Vec<usize>> = vec![Vec::new(); scenario.aircraft.len()];
    for (flight_index, choice) in choices.iter().enumerate() {
        let Some(aircraft_index) = *choice else {
            continue;
        };
        let flight = &scenario.flights[flight_index];
        let aircraft = scenario.aircraft.get(aircraft_index)?;
        let release_time = if aircraft.id == scenario.repair_aircraft_id {
            aircraft.available_at.max(repair_ready?)
        } else {
            aircraft.available_at
        };
        if flight.start < release_time {
            return None;
        }
        by_aircraft[aircraft_index].push(flight_index);
    }

    for (aircraft_index, assigned) in by_aircraft.iter_mut().enumerate() {
        assigned.sort_by_key(|&index| {
            (scenario.flights[index].start, scenario.flights[index].end, scenario.flights[index].id.as_str())
        });
        let turnaround = scenario.aircraft[aircraft_index].turnaround;
        for pair in assigned.windows(2) {
            let earliest_next = scenario.flights[pair[0]].end.checked_add(turnaround)?;
            if scenario.flights[pair[1]].start < earliest_next {
                return None;
            }
        }
    }

    let mut cost = match source {
        PartSource::None => 0,
        PartSource::QualifiedStock => scenario.maintenance_cost,
        PartSource::QualifiedProduction => scenario.production.cost.checked_add(scenario.maintenance_cost)?,
    };
    for (flight_index, choice) in choices.iter().enumerate() {
        if choice.is_none() {
            cost = cost.checked_add(scenario.flights[flight_index].cancellation_cost)?;
        }
    }
    Some(cost)
}

fn varied_scenarios() -> Vec<Scenario> {
    let mut scenarios = Vec::new();
    for seed in 0..16_u32 {
        let mut scenario = reference_scenario();
        scenario.id = format!("independent-oracle-{seed}");
        scenario.configuration_revision = format!("config-{seed}");
        scenario.input_digest = format!("fixture-{seed}");
        scenario.flights[0].start = 20 + seed * 5;
        scenario.flights[0].end = scenario.flights[0].start + 25;
        scenario.flights[0].cancellation_cost = 40 + u64::from(seed) * 3;
        scenario.flights[1].start = 35 + (seed % 5) * 7;
        scenario.flights[1].end = scenario.flights[1].start + 25;
        scenario.flights[1].cancellation_cost = 55 + u64::from(seed) * 2;
        scenario.aircraft[0].available_at = seed % 4 * 8;
        scenario.aircraft[0].turnaround = 5 + seed % 7;
        scenario.aircraft[1].turnaround = 3 + seed % 9;
        scenario.production.duration = 3 + seed % 13;
        scenario.production.cost = 20 + u64::from(seed) * 2;
        if seed % 3 == 0 {
            scenario.qualified_stock_count = 1;
        }
        if seed % 4 == 0 {
            scenario.production.evidence_current = false;
        }
        if seed % 2 == 0 {
            scenario.flights.push(Flight {
                id: "FLIGHT-3".into(),
                start: 60 + seed * 2,
                end: 78 + seed * 2,
                cancellation_cost: 70 + u64::from(seed),
            });
        }
        scenarios.push(scenario);
    }
    scenarios
}

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
fn branch_and_bound_matches_independent_exhaustive_oracle_across_varied_scenarios() {
    for scenario in varied_scenarios() {
        let expected = oracle_optimum(&scenario)
            .expect("cancel-all assignment makes each valid scenario feasible");
        let report = solve_branch_and_bound(&scenario, exact_options()).unwrap_or_else(|error| {
            panic!("scenario {} failed: {error}", scenario.id)
        });
        assert_eq!(report.best_objective_cost, Some(expected), "scenario {}", scenario.id);
        assert_eq!(
            report.status,
            BranchAndBoundStatus::OptimalWithinTolerance,
            "scenario {}",
            scenario.id
        );
        let plan = report.best_plan.as_ref().expect("oracle says feasible");
        assert_eq!(verify_plan(&scenario, plan).status, VerificationStatus::Verified);
        assert_eq!(report.absolute_optimality_gap, Some(0));
    }
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
