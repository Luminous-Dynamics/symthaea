// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tiny deterministic research benchmark for cross-lifecycle aviation planning.
//!
//! This synthetic operations-research fixture is not an airline dispatch,
//! maintenance-release, aircraft-airworthiness, or production authority. The
//! exact solver is bounded to small instances so it can serve as an oracle for
//! later heuristic/coordinator implementations.
//!
//! A verified plan establishes only feasibility under this scenario's declared
//! synthetic constraints. It does not establish aircraft safety, regulatory
//! compliance, or commercial savings. The planner and verifier have separate
//! feasibility/cost paths; the verifier resolves stable IDs and independently
//! recomputes constraints and objective cost.

/// Upper bound keeps exact enumeration appropriate for unit tests and tiny oracles.
pub const MAX_EXACT_FLIGHTS: usize = 6;
/// Upper bound on aircraft in the exhaustive reference instance.
pub const MAX_EXACT_AIRCRAFT: usize = 4;

/// Source used for the one synthetic repairable component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PartSource {
    /// Do not repair the designated aircraft during this planning horizon.
    None,
    /// Use one qualified, configuration-matched stock component.
    QualifiedStock,
    /// Manufacture one qualified component and then perform maintenance.
    QualifiedProduction,
}

/// One synthetic flight operating in a common time unit (for example minutes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Flight {
    pub id: String,
    pub start: u32,
    pub end: u32,
    /// Cost assigned to cancellation in this synthetic scenario.
    pub cancellation_cost: u64,
}

/// One aircraft in the tiny scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aircraft {
    pub id: String,
    /// Earliest time the aircraft is otherwise available.
    pub available_at: u32,
    /// Required spacing after any assigned flight ends.
    pub turnaround: u32,
}

/// Factory capability and evidence for producing the one repair component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionOption {
    pub available: bool,
    pub qualified_worker: bool,
    pub capable_tool: bool,
    pub matches_configuration: bool,
    pub evidence_current: bool,
    /// Production duration from time zero in the scenario's time unit.
    pub duration: u32,
    pub cost: u64,
}

/// Fully specified synthetic benchmark scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scenario {
    pub id: String,
    pub configuration_revision: String,
    /// Caller-supplied digest identifying the frozen input bundle.
    ///
    /// This crate does not implement cryptographic hashing. The harness must
    /// bind this digest to canonical scenario bytes.
    pub input_digest: String,
    pub flights: Vec<Flight>,
    pub aircraft: Vec<Aircraft>,
    /// The single aircraft that is unavailable until repaired.
    pub repair_aircraft_id: String,
    pub qualified_stock_count: u32,
    pub stock_matches_configuration: bool,
    pub stock_evidence_current: bool,
    pub production: ProductionOption,
    pub maintenance_duration: u32,
    pub maintenance_cost: u64,
}

/// Assignment for one flight. None means cancelled in this synthetic plan;
/// every scenario flight must still appear exactly once in assignments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlightAssignment {
    pub flight_id: String,
    pub aircraft_id: Option<String>,
}

/// Plan emitted by the tiny exact solver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub scenario_id: String,
    pub configuration_revision: String,
    pub input_digest: String,
    pub part_source: PartSource,
    pub assignments: Vec<FlightAssignment>,
    pub total_cost: u64,
}

/// Independent verification outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStatus {
    /// Every declared constraint passed and the recomputed cost matches.
    Verified,
    /// The plan violates a known constraint or is bound to the wrong revision.
    Rejected,
    /// The scenario is malformed or the verifier cannot assess it.
    Indeterminate,
}

/// Result of re-evaluating a proposed plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationReport {
    pub status: VerificationStatus,
    pub violations: Vec<String>,
    pub recomputed_cost: Option<u64>,
}

impl VerificationReport {
    fn rejected(violations: Vec<String>, recomputed_cost: Option<u64>) -> Self {
        Self {
            status: VerificationStatus::Rejected,
            violations,
            recomputed_cost,
        }
    }

    fn indeterminate(reason: String) -> Self {
        Self {
            status: VerificationStatus::Indeterminate,
            violations: vec![reason],
            recomputed_cost: None,
        }
    }
}

impl Scenario {
    /// Validate schema and arithmetic bounds. False capability/evidence flags
    /// mean a source is unavailable; they do not by themselves malform a case.
    pub fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty()
            || self.configuration_revision.trim().is_empty()
            || self.input_digest.trim().is_empty()
        {
            return Err("scenario identity, revision, and input digest are required".into());
        }
        if self.flights.is_empty() {
            return Err("scenario must contain at least one flight".into());
        }
        if self.flights.len() > MAX_EXACT_FLIGHTS {
            return Err(format!("exact oracle supports at most {MAX_EXACT_FLIGHTS} flights"));
        }
        if self.aircraft.is_empty() {
            return Err("scenario must contain at least one aircraft".into());
        }
        if self.aircraft.len() > MAX_EXACT_AIRCRAFT {
            return Err(format!("exact oracle supports at most {MAX_EXACT_AIRCRAFT} aircraft"));
        }

        for (i, flight) in self.flights.iter().enumerate() {
            if flight.id.trim().is_empty() || flight.start >= flight.end {
                return Err(format!("flight {i} has an empty ID or invalid time interval"));
            }
            if self.flights[..i].iter().any(|prior| prior.id == flight.id) {
                return Err(format!("duplicate flight ID: {}", flight.id));
            }
        }

        for (i, aircraft) in self.aircraft.iter().enumerate() {
            if aircraft.id.trim().is_empty() {
                return Err(format!("aircraft {i} has an empty ID"));
            }
            if self.aircraft[..i].iter().any(|prior| prior.id == aircraft.id) {
                return Err(format!("duplicate aircraft ID: {}", aircraft.id));
            }
        }

        if !self.aircraft.iter().any(|a| a.id == self.repair_aircraft_id) {
            return Err("repair aircraft ID does not identify a declared aircraft".into());
        }

        let base_cost = self
            .maintenance_cost
            .checked_add(self.production.cost)
            .ok_or_else(|| "maintenance plus production cost overflows u64".to_string())?;
        let _maximum_scenario_cost = self
            .flights
            .iter()
            .try_fold(base_cost, |total, flight| {
                total.checked_add(flight.cancellation_cost)
            })
            .ok_or_else(|| "maximum scenario cost overflows u64".to_string())?;

        if self.production.available
            && self
                .production
                .duration
                .checked_add(self.maintenance_duration)
                .is_none()
        {
            return Err("production plus maintenance duration overflows u32".into());
        }
        Ok(())
    }

    fn source_is_usable(&self, source: PartSource) -> bool {
        match source {
            PartSource::None => true,
            PartSource::QualifiedStock => {
                self.qualified_stock_count > 0
                    && self.stock_matches_configuration
                    && self.stock_evidence_current
            }
            PartSource::QualifiedProduction => {
                self.production.available
                    && self.production.qualified_worker
                    && self.production.capable_tool
                    && self.production.matches_configuration
                    && self.production.evidence_current
            }
        }
    }

    fn source_options(&self) -> Vec<PartSource> {
        let mut sources = vec![PartSource::None];
        if self.source_is_usable(PartSource::QualifiedStock) {
            sources.push(PartSource::QualifiedStock);
        }
        if self.source_is_usable(PartSource::QualifiedProduction) {
            sources.push(PartSource::QualifiedProduction);
        }
        sources
    }
}

/// Exhaustively enumerate bounded assignments and part-source choices.
///
/// This is an exact oracle for this declared small model, not a scalable airline
/// optimizer. Ties keep the first encountered plan, making fixed inputs stable.
pub fn solve_exact(scenario: &Scenario) -> Result<Plan, String> {
    scenario.validate()?;
    let mut best: Option<Plan> = None;

    for source in scenario.source_options() {
        let mut choices = vec![None; scenario.flights.len()];
        enumerate_assignments(scenario, source, 0, &mut choices, &mut best);
    }

    best.ok_or_else(|| "no feasible plan exists for the declared scenario".to_string())
}

fn enumerate_assignments(
    scenario: &Scenario,
    source: PartSource,
    flight_index: usize,
    choices: &mut [Option<usize>],
    best: &mut Option<Plan>,
) {
    if flight_index < scenario.flights.len() {
        for aircraft_index in 0..scenario.aircraft.len() {
            choices[flight_index] = Some(aircraft_index);
            enumerate_assignments(scenario, source, flight_index + 1, choices, best);
        }
        choices[flight_index] = None;
        enumerate_assignments(scenario, source, flight_index + 1, choices, best);
        return;
    }

    if !candidate_is_feasible(scenario, source, choices) {
        return;
    }
    let Some(cost) = candidate_cost(scenario, source, choices) else {
        return;
    };
    if best.as_ref().is_some_and(|current| current.total_cost <= cost) {
        return;
    }

    let assignments = scenario
        .flights
        .iter()
        .zip(choices.iter())
        .map(|(flight, aircraft_index)| FlightAssignment {
            flight_id: flight.id.clone(),
            aircraft_id: aircraft_index.map(|index| scenario.aircraft[index].id.clone()),
        })
        .collect();

    *best = Some(Plan {
        scenario_id: scenario.id.clone(),
        configuration_revision: scenario.configuration_revision.clone(),
        input_digest: scenario.input_digest.clone(),
        part_source: source,
        assignments,
        total_cost: cost,
    });
}

/// Planner-side feasibility path. The public verifier independently checks the
/// emitted stable-ID plan and recomputes constraints and cost.
fn candidate_is_feasible(
    scenario: &Scenario,
    source: PartSource,
    choices: &[Option<usize>],
) -> bool {
    if !scenario.source_is_usable(source) || choices.len() != scenario.flights.len() {
        return false;
    }

    let repaired_ready_at = match source {
        PartSource::None => None,
        PartSource::QualifiedStock => Some(scenario.maintenance_duration),
        PartSource::QualifiedProduction => {
            scenario.production.duration.checked_add(scenario.maintenance_duration)
        }
    };

    let mut by_aircraft: Vec<Vec<usize>> = vec![Vec::new(); scenario.aircraft.len()];
    for (flight_index, choice) in choices.iter().enumerate() {
        let Some(aircraft_index) = *choice else { continue };
        let Some(aircraft) = scenario.aircraft.get(aircraft_index) else {
            return false;
        };
        let flight = &scenario.flights[flight_index];
        let ready_at = if aircraft.id == scenario.repair_aircraft_id {
            let Some(repair_ready) = repaired_ready_at else { return false };
            aircraft.available_at.max(repair_ready)
        } else {
            aircraft.available_at
        };
        if flight.start < ready_at {
            return false;
        }
        by_aircraft[aircraft_index].push(flight_index);
    }

    for (aircraft_index, assigned) in by_aircraft.iter_mut().enumerate() {
        assigned.sort_by_key(|&index| (scenario.flights[index].start, scenario.flights[index].end));
        let turnaround = scenario.aircraft[aircraft_index].turnaround;
        for pair in assigned.windows(2) {
            let previous = &scenario.flights[pair[0]];
            let next = &scenario.flights[pair[1]];
            let Some(earliest_next) = previous.end.checked_add(turnaround) else {
                return false;
            };
            if next.start < earliest_next {
                return false;
            }
        }
    }
    true
}

fn candidate_cost(
    scenario: &Scenario,
    source: PartSource,
    choices: &[Option<usize>],
) -> Option<u64> {
    let mut total = match source {
        PartSource::None => 0,
        PartSource::QualifiedStock => scenario.maintenance_cost,
        PartSource::QualifiedProduction => scenario.production.cost.checked_add(scenario.maintenance_cost)?,
    };
    for (index, choice) in choices.iter().enumerate() {
        if choice.is_none() {
            total = total.checked_add(scenario.flights[index].cancellation_cost)?;
        }
    }
    Some(total)
}

/// Independently check a plan against its scenario. No result from this function
/// authorizes a real-world operation.
pub fn verify_plan(scenario: &Scenario, plan: &Plan) -> VerificationReport {
    if let Err(reason) = scenario.validate() {
        return VerificationReport::indeterminate(reason);
    }

    let mut violations = Vec::new();
    if plan.scenario_id != scenario.id {
        violations.push("plan belongs to a different scenario".to_string());
    }
    if plan.configuration_revision != scenario.configuration_revision {
        violations.push("plan configuration revision is stale or mismatched".to_string());
    }
    if plan.input_digest != scenario.input_digest {
        violations.push("plan input digest does not match the frozen scenario".to_string());
    }

    let source_usable = match plan.part_source {
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
        }
    };
    if !source_usable {
        violations.push("selected part source lacks stock, capability, configuration match, or current evidence".into());
    }

    if plan.assignments.len() != scenario.flights.len() {
        violations.push("plan must contain exactly one assignment record per scenario flight".into());
    }

    // Resolve by stable IDs, independently of the planner's vector indexes.
    let mut seen_flights: Vec<&str> = Vec::new();
    let mut resolved: Vec<(usize, Option<usize>)> = Vec::new();
    for assignment in &plan.assignments {
        if seen_flights.contains(&assignment.flight_id.as_str()) {
            violations.push(format!("duplicate assignment record for flight {}", assignment.flight_id));
            continue;
        }
        seen_flights.push(&assignment.flight_id);
        let Some(flight_index) = scenario.flights.iter().position(|f| f.id.as_str() == assignment.flight_id.as_str()) else {
            violations.push(format!("unknown flight ID in plan: {}", assignment.flight_id));
            continue;
        };
        let aircraft_index = match assignment.aircraft_id.as_ref() {
            None => None,
            Some(id) => match scenario.aircraft.iter().position(|a| &a.id == id) {
                Some(index) => Some(index),
                None => {
                    violations.push(format!("unknown aircraft ID in plan: {id}"));
                    continue;
                }
            },
        };
        resolved.push((flight_index, aircraft_index));
    }

    for flight in &scenario.flights {
        if !seen_flights.contains(&flight.id.as_str()) {
            violations.push(format!("flight {} is missing from plan", flight.id));
        }
    }

    let repair_ready_at = match plan.part_source {
        PartSource::None => None,
        PartSource::QualifiedStock => Some(scenario.maintenance_duration),
        PartSource::QualifiedProduction => {
            scenario.production.duration.checked_add(scenario.maintenance_duration)
        }
    };

    for &(flight_index, aircraft_index) in &resolved {
        let Some(aircraft_index) = aircraft_index else { continue };
        let aircraft = &scenario.aircraft[aircraft_index];
        let flight = &scenario.flights[flight_index];
        let ready_at = if aircraft.id == scenario.repair_aircraft_id {
            match repair_ready_at {
                Some(time) => aircraft.available_at.max(time),
                None => {
                    violations.push(format!("repaired aircraft {} is assigned without a part source", aircraft.id));
                    continue;
                }
            }
        } else {
            aircraft.available_at
        };
        if flight.start < ready_at {
            violations.push(format!("aircraft {} is not available for flight {}", aircraft.id, flight.id));
        }
    }

    for (aircraft_index, aircraft) in scenario.aircraft.iter().enumerate() {
        let mut assigned: Vec<&Flight> = resolved
            .iter()
            .filter_map(|(flight_index, choice)| {
                (*choice == Some(aircraft_index)).then_some(&scenario.flights[*flight_index])
            })
            .collect();
        assigned.sort_by_key(|flight| (flight.start, flight.end));
        for pair in assigned.windows(2) {
            let Some(earliest_next) = pair[0].end.checked_add(aircraft.turnaround) else {
                violations.push(format!("turnaround arithmetic overflows for aircraft {}", aircraft.id));
                continue;
            };
            if pair[1].start < earliest_next {
                violations.push(format!("overlapping or insufficient-turnaround assignments for aircraft {}", aircraft.id));
            }
        }
    }

    let recomputed_cost = recompute_cost_independently(scenario, plan, &resolved);
    if recomputed_cost.is_none() {
        violations.push("independent objective recomputation overflowed or lacked a complete plan".into());
    }
    if let Some(cost) = recomputed_cost {
        if cost != plan.total_cost {
            violations.push(format!(
                "declared plan cost {} differs from independently recomputed cost {cost}",
                plan.total_cost
            ));
        }
    }

    if violations.is_empty() {
        VerificationReport {
            status: VerificationStatus::Verified,
            violations,
            recomputed_cost,
        }
    } else {
        VerificationReport::rejected(violations, recomputed_cost)
    }
}

fn recompute_cost_independently(
    scenario: &Scenario,
    plan: &Plan,
    resolved: &[(usize, Option<usize>)],
) -> Option<u64> {
    if plan.assignments.len() != scenario.flights.len() || resolved.len() != scenario.flights.len() {
        return None;
    }

    let mut total = match plan.part_source {
        PartSource::None => 0,
        PartSource::QualifiedStock => scenario.maintenance_cost,
        PartSource::QualifiedProduction => scenario.production.cost.checked_add(scenario.maintenance_cost)?,
    };

    for (flight_index, flight) in scenario.flights.iter().enumerate() {
        let mut assigned_to_aircraft = false;
        for (resolved_index, resolved_aircraft) in resolved {
            if *resolved_index == flight_index && resolved_aircraft.is_some() {
                assigned_to_aircraft = true;
                break;
            }
        }
        if !assigned_to_aircraft {
            total = total.checked_add(flight.cancellation_cost)?;
        }
    }
    Some(total)
}

/// Canonical tiny fixture. A healthy aircraft can cover one of two concurrent
/// flights. A second needs a component; factory production plus maintenance costs
/// less than cancelling the uncovered flight. This is not an airline cost model.
pub fn reference_scenario() -> Scenario {
    Scenario {
        id: "AECB-001-TINY-001".into(),
        configuration_revision: "aircraft-config-r1".into(),
        input_digest: "synthetic-fixture-digest-v1".into(),
        flights: vec![
            Flight { id: "FLIGHT-1".into(), start: 50, end: 80, cancellation_cost: 100 },
            Flight { id: "FLIGHT-2".into(), start: 50, end: 80, cancellation_cost: 100 },
        ],
        aircraft: vec![
            Aircraft { id: "AIRCRAFT-REPAIRABLE".into(), available_at: 0, turnaround: 10 },
            Aircraft { id: "AIRCRAFT-HEALTHY".into(), available_at: 0, turnaround: 10 },
        ],
        repair_aircraft_id: "AIRCRAFT-REPAIRABLE".into(),
        qualified_stock_count: 0,
        stock_matches_configuration: true,
        stock_evidence_current: true,
        production: ProductionOption {
            available: true,
            qualified_worker: true,
            capable_tool: true,
            matches_configuration: true,
            evidence_current: true,
            duration: 10,
            cost: 35,
        },
        maintenance_duration: 10,
        maintenance_cost: 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_oracle_uses_production_to_avoid_cancellation() {
        let scenario = reference_scenario();
        let plan = solve_exact(&scenario).expect("reference scenario is feasible");
        assert_eq!(plan.part_source, PartSource::QualifiedProduction);
        assert_eq!(plan.total_cost, 40);
        assert!(plan.assignments.iter().all(|a| a.aircraft_id.is_some()));
        let report = verify_plan(&scenario, &plan);
        assert_eq!(report.status, VerificationStatus::Verified);
        assert_eq!(report.recomputed_cost, Some(40));
    }

    #[test]
    fn exact_oracle_is_deterministic() {
        let scenario = reference_scenario();
        assert_eq!(solve_exact(&scenario), solve_exact(&scenario));
    }

    #[test]
    fn stale_revision_is_rejected() {
        let scenario = reference_scenario();
        let mut plan = solve_exact(&scenario).unwrap();
        plan.configuration_revision = "aircraft-config-old".into();
        let report = verify_plan(&scenario, &plan);
        assert_eq!(report.status, VerificationStatus::Rejected);
        assert!(report.violations.iter().any(|v| v.contains("revision")));
    }

    #[test]
    fn production_without_current_evidence_is_rejected() {
        let mut scenario = reference_scenario();
        let plan = solve_exact(&scenario).unwrap();
        scenario.production.evidence_current = false;
        let report = verify_plan(&scenario, &plan);
        assert_eq!(report.status, VerificationStatus::Rejected);
        assert!(report.violations.iter().any(|v| v.contains("part source")));
    }

    #[test]
    fn production_without_qualified_tool_is_not_selected() {
        let mut scenario = reference_scenario();
        scenario.production.capable_tool = false;
        let plan = solve_exact(&scenario).unwrap();
        assert_ne!(plan.part_source, PartSource::QualifiedProduction);
        assert_eq!(verify_plan(&scenario, &plan).status, VerificationStatus::Verified);
    }

    #[test]
    fn duplicate_assignment_record_is_rejected() {
        let scenario = reference_scenario();
        let mut plan = solve_exact(&scenario).unwrap();
        plan.assignments[1].flight_id = plan.assignments[0].flight_id.clone();
        let report = verify_plan(&scenario, &plan);
        assert_eq!(report.status, VerificationStatus::Rejected);
        assert!(report.violations.iter().any(|v| v.contains("duplicate assignment")));
    }

    #[test]
    fn unknown_aircraft_id_is_rejected() {
        let scenario = reference_scenario();
        let mut plan = solve_exact(&scenario).unwrap();
        plan.assignments[0].aircraft_id = Some("UNQUALIFIED-AIRCRAFT".into());
        let report = verify_plan(&scenario, &plan);
        assert_eq!(report.status, VerificationStatus::Rejected);
        assert!(report.violations.iter().any(|v| v.contains("unknown aircraft")));
    }

    #[test]
    fn tampered_objective_is_rejected() {
        let scenario = reference_scenario();
        let mut plan = solve_exact(&scenario).unwrap();
        plan.total_cost = 0;
        let report = verify_plan(&scenario, &plan);
        assert_eq!(report.status, VerificationStatus::Rejected);
        assert!(report.violations.iter().any(|v| v.contains("differs")));
    }

    #[test]
    fn malformed_time_interval_is_indeterminate() {
        let mut scenario = reference_scenario();
        scenario.flights[0].end = scenario.flights[0].start;
        let plan = solve_exact(&reference_scenario()).unwrap();
        let report = verify_plan(&scenario, &plan);
        assert_eq!(report.status, VerificationStatus::Indeterminate);
    }

    #[test]
    fn stale_stock_cannot_be_used_as_a_part_source() {
        let mut scenario = reference_scenario();
        scenario.qualified_stock_count = 1;
        scenario.stock_evidence_current = false;
        let mut plan = solve_exact(&scenario).unwrap();
        plan.part_source = PartSource::QualifiedStock;
        plan.total_cost = 5;
        let report = verify_plan(&scenario, &plan);
        assert_eq!(report.status, VerificationStatus::Rejected);
        assert!(report.violations.iter().any(|v| v.contains("part source")));
    }

    #[test]
    fn overlapping_assignments_to_one_aircraft_are_rejected() {
        let mut scenario = reference_scenario();
        scenario.flights[1].start = 60;
        scenario.flights[1].end = 90;
        let mut plan = solve_exact(&scenario).unwrap();
        plan.assignments[0].aircraft_id = Some("AIRCRAFT-HEALTHY".into());
        plan.assignments[1].aircraft_id = Some("AIRCRAFT-HEALTHY".into());
        plan.total_cost = 40;
        let report = verify_plan(&scenario, &plan);
        assert_eq!(report.status, VerificationStatus::Rejected);
        assert!(report.violations.iter().any(|v| v.contains("overlapping")));
    }

    #[test]
    fn qualified_stock_beats_more_expensive_production() {
        let mut scenario = reference_scenario();
        scenario.qualified_stock_count = 1;
        let plan = solve_exact(&scenario).unwrap();
        assert_eq!(plan.part_source, PartSource::QualifiedStock);
        assert_eq!(plan.total_cost, 5);
        assert_eq!(verify_plan(&scenario, &plan).status, VerificationStatus::Verified);
    }

    #[test]
    fn production_that_finishes_after_departure_is_not_used() {
        let mut scenario = reference_scenario();
        scenario.production.duration = 100;
        let plan = solve_exact(&scenario).unwrap();
        assert_eq!(plan.part_source, PartSource::None);
        assert_eq!(plan.total_cost, 100);
        assert_eq!(verify_plan(&scenario, &plan).status, VerificationStatus::Verified);
    }

    #[test]
    fn production_and_repair_time_overflow_is_indeterminate() {
        let mut scenario = reference_scenario();
        let plan = solve_exact(&scenario).unwrap();
        scenario.production.duration = u32::MAX;
        let report = verify_plan(&scenario, &plan);
        assert_eq!(report.status, VerificationStatus::Indeterminate);
    }

    #[test]
    fn input_digest_mismatch_is_rejected() {
        let scenario = reference_scenario();
        let mut plan = solve_exact(&scenario).unwrap();
        plan.input_digest = "another-input-bundle".into();
        let report = verify_plan(&scenario, &plan);
        assert_eq!(report.status, VerificationStatus::Rejected);
        assert!(report.violations.iter().any(|v| v.contains("input digest")));
    }
}
