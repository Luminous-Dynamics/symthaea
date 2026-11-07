// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Separately compiled reference checks for the tiny aviation enterprise model.
//!
//! Keep this evaluator structurally independent from the production module:
//! it consumes only public scenario/plan types, resolves IDs itself, and
//! enumerates every source/assignment combination for the tiny fixtures.
//! This is test-only research machinery, not a production authority.

use symthaea_operations_research::aviation_enterprise::{
    reference_scenario, solve_exact, Flight, FlightAssignment, PartSource, Plan, Scenario,
};

fn evaluate_independently(scenario: &Scenario, plan: &Plan) -> Option<u64> {
    if scenario.id.is_empty()
        || scenario.configuration_revision.is_empty()
        || scenario.input_digest.is_empty()
        || plan.scenario_id != scenario.id
        || plan.configuration_revision != scenario.configuration_revision
        || plan.input_digest != scenario.input_digest
        || scenario.flights.is_empty()
        || scenario.aircraft.is_empty()
        || plan.assignments.len() != scenario.flights.len()
    {
        return None;
    }

    // Validate identities and intervals here rather than delegating to the
    // implementation under test.
    for (i, flight) in scenario.flights.iter().enumerate() {
        if flight.id.is_empty() || flight.start >= flight.end {
            return None;
        }
        if scenario.flights[..i].iter().any(|prior| prior.id == flight.id) {
            return None;
        }
    }
    for (i, aircraft) in scenario.aircraft.iter().enumerate() {
        if aircraft.id.is_empty()
            || scenario.aircraft[..i].iter().any(|prior| prior.id == aircraft.id)
        {
            return None;
        }
    }
    if !scenario.aircraft.iter().any(|a| a.id == scenario.repair_aircraft_id) {
        return None;
    }

    match plan.part_source {
        PartSource::None => {}
        PartSource::QualifiedStock => {
            if scenario.qualified_stock_count == 0
                || !scenario.stock_matches_configuration
                || !scenario.stock_evidence_current
            {
                return None;
            }
        }
        PartSource::QualifiedProduction => {
            if !scenario.production.available
                || !scenario.production.qualified_worker
                || !scenario.production.capable_tool
                || !scenario.production.matches_configuration
                || !scenario.production.evidence_current
            {
                return None;
            }
        }
    }

    let repair_ready = match plan.part_source {
        PartSource::None => None,
        PartSource::QualifiedStock => Some(scenario.maintenance_duration),
        PartSource::QualifiedProduction => Some(
            scenario
                .production
                .duration
                .checked_add(scenario.maintenance_duration)?,
        ),
    };

    let mut seen = vec![false; scenario.flights.len()];
    let mut assigned_aircraft = vec![None; scenario.flights.len()];
    for assignment in &plan.assignments {
        let flight_index = scenario
            .flights
            .iter()
            .position(|f| f.id == assignment.flight_id)?;
        if seen[flight_index] {
            return None;
        }
        seen[flight_index] = true;
        let aircraft_index = match assignment.aircraft_id.as_deref() {
            None => None,
            Some(id) => Some(scenario.aircraft.iter().position(|a| a.id == id)?),
        };
        assigned_aircraft[flight_index] = aircraft_index;
    }
    if seen.iter().any(|was_seen| !was_seen) {
        return None;
    }

    for (flight_index, aircraft_index) in assigned_aircraft.iter().enumerate() {
        let Some(aircraft_index) = aircraft_index else {
            continue;
        };
        let flight = &scenario.flights[flight_index];
        let aircraft = &scenario.aircraft[*aircraft_index];
        let release_time = if aircraft.id == scenario.repair_aircraft_id {
            aircraft.available_at.max(repair_ready?)
        } else {
            aircraft.available_at
        };
        if flight.start < release_time {
            return None;
        }
    }

    // Build independent per-aircraft time sequences, then check their
    // turnaround constraints. Sort by start time; IDs are only used as a
    // deterministic secondary key.
    for (aircraft_index, aircraft) in scenario.aircraft.iter().enumerate() {
        let mut assigned: Vec<&Flight> = assigned_aircraft
            .iter()
            .enumerate()
            .filter_map(|(flight_index, chosen)| {
                (*chosen == Some(aircraft_index)).then_some(&scenario.flights[flight_index])
            })
            .collect();
        assigned.sort_by(|a, b| (a.start, a.end, a.id.as_str()).cmp(&(b.start, b.end, b.id.as_str())));
        for consecutive in assigned.windows(2) {
            if consecutive[0].end.checked_add(aircraft.turnaround)? > consecutive[1].start {
                return None;
            }
        }
    }

    let mut cost = match plan.part_source {
        PartSource::None => 0,
        PartSource::QualifiedStock => scenario.maintenance_cost,
        PartSource::QualifiedProduction => scenario
            .production
            .cost
            .checked_add(scenario.maintenance_cost)?,
    };
    for (i, aircraft_index) in assigned_aircraft.iter().enumerate() {
        if aircraft_index.is_none() {
            cost = cost.checked_add(scenario.flights[i].cancellation_cost)?;
        }
    }

    if cost == plan.total_cost {
        Some(cost)
    } else {
        None
    }
}

fn source_is_eligible(scenario: &Scenario, source: PartSource) -> bool {
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
                && scenario
                    .production
                    .duration
                    .checked_add(scenario.maintenance_duration)
                    .is_some()
        }
    }
}

fn enumerate_for_source(
    scenario: &Scenario,
    source: PartSource,
    flight_index: usize,
    assignments: &mut Vec<FlightAssignment>,
    best: &mut Option<Plan>,
) {
    if flight_index == scenario.flights.len() {
        let mut candidate = Plan {
            scenario_id: scenario.id.clone(),
            configuration_revision: scenario.configuration_revision.clone(),
            input_digest: scenario.input_digest.clone(),
            part_source: source,
            assignments: assignments.clone(),
            total_cost: 0,
        };

        // Evaluate with the independently written constraints/cost, then bind
        // the independently computed cost and re-check the serialized claim.
        let cost_without_declared_check = evaluate_for_oracle(scenario, &candidate);
        let Some(cost) = cost_without_declared_check else {
            return;
        };
        candidate.total_cost = cost;
        if evaluate_independently(scenario, &candidate).is_none() {
            return;
        }

        if best.as_ref().is_none_or(|current| cost < current.total_cost) {
            *best = Some(candidate);
        }
        return;
    }

    let flight = &scenario.flights[flight_index];
    for aircraft in &scenario.aircraft {
        assignments.push(FlightAssignment {
            flight_id: flight.id.clone(),
            aircraft_id: Some(aircraft.id.clone()),
        });
        enumerate_for_source(scenario, source, flight_index + 1, assignments, best);
        assignments.pop();
    }

    assignments.push(FlightAssignment {
        flight_id: flight.id.clone(),
        aircraft_id: None,
    });
    enumerate_for_source(scenario, source, flight_index + 1, assignments, best);
    assignments.pop();
}

// This oracle helper duplicates the feasibility and objective semantics
// intentionally, but does not inspect or call the production solver/verifier.
fn evaluate_for_oracle(scenario: &Scenario, plan: &Plan) -> Option<u64> {
    if !source_is_eligible(scenario, plan.part_source)
        || plan.assignments.len() != scenario.flights.len()
    {
        return None;
    }

    let mut by_flight: Vec<Option<Option<usize>>> = vec![None; scenario.flights.len()];
    for assignment in &plan.assignments {
        let index = scenario.flights.iter().position(|f| f.id == assignment.flight_id)?;
        if by_flight[index].is_some() {
            return None;
        }
        let chosen = match assignment.aircraft_id.as_deref() {
            None => None,
            Some(id) => Some(scenario.aircraft.iter().position(|a| a.id == id)?),
        };
        by_flight[index] = Some(chosen);
    }
    if by_flight.iter().any(Option::is_none) {
        return None;
    }

    let repair_ready = match plan.part_source {
        PartSource::None => None,
        PartSource::QualifiedStock => Some(scenario.maintenance_duration),
        PartSource::QualifiedProduction => Some(
            scenario
                .production
                .duration
                .checked_add(scenario.maintenance_duration)?,
        ),
    };

    let mut total = match plan.part_source {
        PartSource::None => 0,
        PartSource::QualifiedStock => scenario.maintenance_cost,
        PartSource::QualifiedProduction => {
            scenario.production.cost.checked_add(scenario.maintenance_cost)?
        }
    };

    for (i, option) in by_flight.iter().enumerate() {
        let chosen = option.as_ref()?;
        if let Some(aircraft_index) = chosen {
            let flight = &scenario.flights[i];
            let aircraft = &scenario.aircraft[*aircraft_index];
            if flight.start < aircraft.available_at {
                return None;
            }
            if aircraft.id == scenario.repair_aircraft_id && flight.start < repair_ready? {
                return None;
            }
        } else {
            total = total.checked_add(scenario.flights[i].cancellation_cost)?;
        }
    }

    for (aircraft_index, aircraft) in scenario.aircraft.iter().enumerate() {
        let mut times: Vec<(u32, u32)> = by_flight
            .iter()
            .enumerate()
            .filter_map(|(i, chosen)| {
                (chosen.as_ref().and_then(|v| *v) == Some(aircraft_index))
                    .then_some((scenario.flights[i].start, scenario.flights[i].end))
            })
            .collect();
        times.sort_unstable();
        for pair in times.windows(2) {
            if pair[0].1.checked_add(aircraft.turnaround)? > pair[1].0 {
                return None;
            }
        }
    }
    Some(total)
}

fn exact_oracle(scenario: &Scenario) -> Option<Plan> {
    let mut best = None;
    for source in [
        PartSource::None,
        PartSource::QualifiedStock,
        PartSource::QualifiedProduction,
    ] {
        if !source_is_eligible(scenario, source) {
            continue;
        }
        enumerate_for_source(scenario, source, 0, &mut Vec::new(), &mut best);
    }
    best
}

fn variants() -> Vec<Scenario> {
    let base = reference_scenario();

    let mut stock = base.clone();
    stock.qualified_stock_count = 1;

    let mut stale_stock = base.clone();
    stale_stock.qualified_stock_count = 1;
    stale_stock.stock_evidence_current = false;

    let mut no_tool = base.clone();
    no_tool.production.capable_tool = false;

    let mut production_too_late = base.clone();
    production_too_late.production.duration = 100;

    let mut expensive_production = base.clone();
    expensive_production.production.cost = 200;

    let mut non_overlapping_flights = base.clone();
    non_overlapping_flights.flights[1].start = 90;
    non_overlapping_flights.flights[1].end = 120;

    vec![
        base,
        stock,
        stale_stock,
        no_tool,
        production_too_late,
        expensive_production,
        non_overlapping_flights,
    ]
}

#[test]
fn production_solver_matches_separate_exact_reference_on_frozen_variants() {
    for scenario in variants() {
        let plan = solve_exact(&scenario).expect("valid small scenario should be solvable");
        let reference = exact_oracle(&scenario).expect("reference oracle should find a plan");
        assert_eq!(plan.total_cost, reference.total_cost, "scenario {}", scenario.id);
        assert_eq!(
            evaluate_independently(&scenario, &plan),
            Some(plan.total_cost),
            "independent plan check failed for {}",
            scenario.id
        );
    }
}

#[test]
fn independent_checker_rejects_tampering_and_stale_configuration() {
    let scenario = reference_scenario();
    let baseline = solve_exact(&scenario).unwrap();
    assert_eq!(evaluate_independently(&scenario, &baseline), Some(40));

    let mut stale = baseline.clone();
    stale.configuration_revision = "old-revision".into();
    assert_eq!(evaluate_independently(&scenario, &stale), None);

    let mut wrong_digest = baseline.clone();
    wrong_digest.input_digest = "wrong-bundle".into();
    assert_eq!(evaluate_independently(&scenario, &wrong_digest), None);

    let mut false_cost = baseline.clone();
    false_cost.total_cost = 0;
    assert_eq!(evaluate_independently(&scenario, &false_cost), None);

    let mut duplicate = baseline.clone();
    duplicate.assignments[1].flight_id = duplicate.assignments[0].flight_id.clone();
    assert_eq!(evaluate_independently(&scenario, &duplicate), None);

    let mut unknown_aircraft = baseline;
    unknown_aircraft.assignments[0].aircraft_id = Some("unknown".into());
    assert_eq!(evaluate_independently(&scenario, &unknown_aircraft), None);
}

#[test]
fn independent_checker_rejects_a_source_without_qualifications() {
    let mut scenario = reference_scenario();
    scenario.production.qualified_worker = false;
    let mut plan = solve_exact(&scenario).unwrap();
    plan.part_source = PartSource::QualifiedProduction;
    plan.total_cost = 40;
    assert_eq!(evaluate_independently(&scenario, &plan), None);
}

#[test]
fn invalid_duration_overflow_is_not_assessed_as_feasible() {
    let mut scenario = reference_scenario();
    scenario.production.duration = u32::MAX;
    assert!(scenario.validate().is_err());
    assert!(solve_exact(&scenario).is_err());
}
