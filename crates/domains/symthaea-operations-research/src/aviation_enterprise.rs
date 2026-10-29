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
//! feasibility/cost paths; the verifier resolves stable IDs and recomputes
//! constraints and objective cost. This is a separate code path in the same
//! crate, not yet an independently maintained checker; that stronger check
//! remains required before a benchmark release.

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

/// Verification outcome from the separate stable-ID checking path.
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

/// Check a plan against its scenario using a separate stable-ID code path. No result
/// from this function authorizes a real-world operation.
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

/// Machine-readable run record for a synthetic benchmark execution.
///
/// The input digest is supplied by the outer harness; this struct does not hash,
/// sign, or authenticate the run. Its JSON is an audit artifact, not an authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunManifest {
    pub schema_version: String,
    pub benchmark_id: String,
    pub scenario_id: String,
    pub configuration_revision: String,
    pub input_digest: String,
    pub method_id: String,
    pub part_source: PartSource,
    pub seed: u64,
    pub declared_plan_cost: Option<u64>,
    pub recomputed_cost: Option<u64>,
    pub verifier_status: VerificationStatus,
    pub violation_count: usize,
    pub violations: Vec<String>,
    pub assigned_flight_count: usize,
    pub cancelled_flight_count: usize,
    pub synthetic: bool,
}

impl RunManifest {
    /// Record plan and verifier outputs without upgrading their claim level.
    pub fn from_run(
        scenario: &Scenario,
        seed: u64,
        plan: &Plan,
        verification: &VerificationReport,
    ) -> Self {
        Self {
            schema_version: "AECB-001-run-v1".into(),
            benchmark_id: "AECB-001".into(),
            scenario_id: scenario.id.clone(),
            configuration_revision: scenario.configuration_revision.clone(),
            input_digest: scenario.input_digest.clone(),
            method_id: "bounded-exact-enumerator-v1".into(),
            part_source: plan.part_source,
            seed,
            declared_plan_cost: Some(plan.total_cost),
            recomputed_cost: verification.recomputed_cost,
            verifier_status: verification.status,
            violation_count: verification.violations.len(),
            violations: verification.violations.clone(),
            assigned_flight_count: plan.assignments.iter().filter(|a| a.aircraft_id.is_some()).count(),
            cancelled_flight_count: plan.assignments.iter().filter(|a| a.aircraft_id.is_none()).count(),
            synthetic: true,
        }
    }

    /// Emit compact JSON using a local string escaper and no serialization dependency.
    pub fn to_json(&self) -> String {
        let violations_json = self
            .violations
            .iter()
            .map(|v| json_quote(v))
            .collect::<Vec<_>>()
            .join(",");
        let cost_json = self
            .declared_plan_cost
            .map(|cost| cost.to_string())
            .unwrap_or_else(|| "null".into());
        let recomputed_cost_json = self
            .recomputed_cost
            .map(|cost| cost.to_string())
            .unwrap_or_else(|| "null".into());

        format!(
            "{{\"schema_version\":{},\"benchmark_id\":{},\"scenario_id\":{},\"configuration_revision\":{},\"input_digest\":{},\"method_id\":{},\"part_source\":{},\"seed\":{},\"declared_plan_cost\":{},\"recomputed_cost\":{},\"verifier_status\":{},\"violation_count\":{},\"violations\":[{}],\"assigned_flight_count\":{},\"cancelled_flight_count\":{},\"synthetic\":{}}}",
            json_quote(&self.schema_version),
            json_quote(&self.benchmark_id),
            json_quote(&self.scenario_id),
            json_quote(&self.configuration_revision),
            json_quote(&self.input_digest),
            json_quote(&self.method_id),
            json_quote(part_source_name(self.part_source)),
            self.seed,
            cost_json,
            recomputed_cost_json,
            json_quote(verification_status_name(self.verifier_status)),
            self.violation_count,
            violations_json,
            self.assigned_flight_count,
            self.cancelled_flight_count,
            self.synthetic,
        )
    }
}

fn part_source_name(source: PartSource) -> &'static str {
    match source {
        PartSource::None => "none",
        PartSource::QualifiedStock => "qualified_stock",
        PartSource::QualifiedProduction => "qualified_production",
    }
}

fn verification_status_name(status: VerificationStatus) -> &'static str {
    match status {
        VerificationStatus::Verified => "verified",
        VerificationStatus::Rejected => "rejected",
        VerificationStatus::Indeterminate => "indeterminate",
    }
}

fn json_quote(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            control if control <= '\u{001f}' => {
                escaped.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => escaped.push(other),
        }
    }
    escaped.push('"');
    escaped
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
    fn factory_coupling_ablation_exposes_cross_domain_value() {
        let integrated_scenario = reference_scenario();
        let integrated = solve_exact(&integrated_scenario).unwrap();

        // Ablation: hide factory production from the exact same scenario.
        // This is a toy structural ablation, not a comparison to industry SOTA.
        let mut production_blind = integrated_scenario.clone();
        production_blind.production.available = false;
        let siloed = solve_exact(&production_blind).unwrap();

        assert_eq!(integrated.total_cost, 40);
        assert_eq!(siloed.total_cost, 100);
        assert_eq!(siloed.total_cost - integrated.total_cost, 60);
        assert_eq!(
            verify_plan(&integrated_scenario, &integrated).status,
            VerificationStatus::Verified
        );
        assert_eq!(
            verify_plan(&production_blind, &siloed).status,
            VerificationStatus::Verified
        );
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

    #[test]
    fn run_manifest_records_verified_result_and_objective() {
        let scenario = reference_scenario();
        let plan = solve_exact(&scenario).unwrap();
        let report = verify_plan(&scenario, &plan);
        let manifest = RunManifest::from_run(&scenario, 7, &plan, &report);
        let json = manifest.to_json();
        assert!(json.contains("\"verifier_status\":\"verified\""));
        assert!(json.contains("\"declared_plan_cost\":40"));
        assert!(json.contains("\"recomputed_cost\":40"));
        assert!(json.contains("\"part_source\":\"qualified_production\""));
        assert!(json.contains("\"cancelled_flight_count\":0"));
        assert!(json.contains("\"seed\":7"));
    }

    #[test]
    fn increasing_cancellation_penalty_never_increases_cancellations() {
        let mut previous_cancelled = usize::MAX;
        for penalty in (0_u64..=200).step_by(5) {
            let mut scenario = reference_scenario();
            for flight in &mut scenario.flights {
                flight.cancellation_cost = penalty;
            }
            let plan = solve_exact(&scenario).expect("tiny scenario has a feasible plan");
            let cancelled = plan.assignments.iter().filter(|a| a.aircraft_id.is_none()).count();
            assert!(
                cancelled <= previous_cancelled,
                "cancellation count increased at penalty {penalty}: {previous_cancelled} -> {cancelled}"
            );
            assert_eq!(verify_plan(&scenario, &plan).status, VerificationStatus::Verified);
            previous_cancelled = cancelled;
        }
    }

    #[test]
    fn increasing_production_lead_time_never_improves_optimal_cost() {
        let mut previous_cost = 0_u64;
        for duration in (0_u32..=100).step_by(5) {
            let mut scenario = reference_scenario();
            scenario.production.duration = duration;
            let plan = solve_exact(&scenario).expect("no-part/cancel plan remains feasible");
            assert!(
                plan.total_cost >= previous_cost,
                "cost decreased when production lead time increased to {duration}: {previous_cost} -> {}",
                plan.total_cost
            );
            assert_eq!(verify_plan(&scenario, &plan).status, VerificationStatus::Verified);
            previous_cost = plan.total_cost;
        }
    }

    #[test]
    fn manifest_json_escapes_control_characters() {
        let scenario = reference_scenario();
        let plan = solve_exact(&scenario).unwrap();
        let report = VerificationReport {
            status: VerificationStatus::Rejected,
            violations: vec!["stale \"revision\"\nblocked".into()],
            recomputed_cost: None,
        };
        let manifest = RunManifest::from_run(&scenario, 0, &plan, &report);
        let json = manifest.to_json();
        assert!(json.contains("\\\"revision\\\""));
        assert!(json.contains("\\n"));
    }
}


/// Deterministic node budget and accepted absolute objective gap for the
/// best-first branch-and-bound solver. Objective costs and the gap use the
/// scenario's integer cost units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchAndBoundOptions {
    pub max_expanded_nodes: u64,
    pub max_absolute_optimality_gap: u64,
}

/// Honest terminal status of a bounded branch-and-bound run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchAndBoundStatus {
    /// The incumbent is within the declared absolute gap of the frontier bound.
    OptimalWithinTolerance,
    /// A verified feasible incumbent exists, but the requested gap is not proved.
    FeasibleUncertified,
    /// The complete search space was exhausted without a feasible plan.
    NoFeasiblePlan,
    /// The node budget ended before a feasible plan was established.
    NoFeasiblePlanWithinBudget,
}

/// Counts useful for matched, machine-readable comparisons against the exhaustive
/// baseline. Node counts are deterministic for frozen input and options.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BranchAndBoundStats {
    pub expanded_nodes: u64,
    pub generated_nodes: u64,
    pub complete_assignments_examined: u64,
    pub infeasible_complete_assignments: u64,
    pub partial_feasibility_prunes: u64,
    pub objective_bound_prunes: u64,
    pub peak_frontier_nodes: usize,
}

/// Report from an exact/anytime bounded search. A reported lower bound is over
/// the remaining frontier together with the incumbent. Without a feasible
/// incumbent, the lower bound alone does not imply existence of a feasible plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchAndBoundReport {
    pub scenario_id: String,
    pub configuration_revision: String,
    pub input_digest: String,
    pub status: BranchAndBoundStatus,
    pub best_plan: Option<Plan>,
    pub best_objective_cost: Option<u64>,
    pub global_lower_bound_cost: Option<u64>,
    pub absolute_optimality_gap: Option<u64>,
    pub accepted_absolute_gap: u64,
    pub stats: BranchAndBoundStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RecoverySearchNode {
    source: PartSource,
    depth: usize,
    /// None = not yet decided; Some(None) = cancelled; Some(Some(i)) = aircraft i.
    choices: Vec<Option<Option<usize>>>,
    /// Admissible bound: fixed source cost + cancellation costs already committed.
    lower_bound: u64,
    serial: u64,
}

impl Ord for RecoverySearchNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap; reverse bound and serial to pop the smallest
        // lower bound first, with stable FIFO order for equal bounds.
        other
            .lower_bound
            .cmp(&self.lower_bound)
            .then_with(|| other.serial.cmp(&self.serial))
    }
}

impl PartialOrd for RecoverySearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Search the complete finite assignment space with best-first branch and bound.
///
/// Each flight is assigned to one declared aircraft or explicitly cancelled;
/// each eligible component source gets its own root. The objective lower bound
/// includes fixed repair/source costs and cancellation costs already committed.
/// Unassigned flights contribute zero, so the bound is admissible for this
/// synthetic model's non-negative objective. Partial constraint checks prune
/// impossible prefixes. Candidate leaves are checked with the planner path, and
/// every new incumbent is independently rechecked through `verify_plan`.
///
/// This is a bounded research solver for `Scenario`, not a production airline
/// optimizer. The node budget is deterministic work accounting, not wall-clock
/// time. A feasible incumbent is never labeled optimal unless the minimum
/// frontier bound proves the requested gap. The model's bounded exact limits
/// are intentionally retained by `Scenario::validate`.
pub fn solve_branch_and_bound(
    scenario: &Scenario,
    options: BranchAndBoundOptions,
) -> Result<BranchAndBoundReport, String> {
    scenario.validate()?;
    if options.max_expanded_nodes == 0 {
        return Err("max_expanded_nodes must be greater than zero".into());
    }

    let mut frontier = std::collections::BinaryHeap::<RecoverySearchNode>::new();
    let mut serial = 0_u64;
    let mut stats = BranchAndBoundStats::default();

    for source in scenario.source_options() {
        let lower_bound = source_base_cost(scenario, source)
            .ok_or_else(|| "source base cost overflows u64".to_string())?;
        frontier.push(RecoverySearchNode {
            source,
            depth: 0,
            choices: vec![None; scenario.flights.len()],
            lower_bound,
            serial,
        });
        serial = serial.saturating_add(1);
        stats.generated_nodes = stats.generated_nodes.saturating_add(1);
    }
    stats.peak_frontier_nodes = frontier.len();

    let mut best_plan: Option<Plan> = None;

    while let Some(frontier_best) = frontier.peek() {
        if let Some(incumbent) = &best_plan {
            let bound = frontier_best.lower_bound.min(incumbent.total_cost);
            let gap = incumbent.total_cost - bound;
            if gap <= options.max_absolute_optimality_gap {
                break;
            }
        }
        if stats.expanded_nodes >= options.max_expanded_nodes {
            break;
        }

        let Some(node) = frontier.pop() else {
            break;
        };
        stats.expanded_nodes = stats.expanded_nodes.saturating_add(1);

        if node.depth == scenario.flights.len() {
            stats.complete_assignments_examined =
                stats.complete_assignments_examined.saturating_add(1);
            let mut complete_choices = Vec::with_capacity(node.choices.len());
            for choice in &node.choices {
                match choice {
                    Some(decision) => complete_choices.push(*decision),
                    None => return Err("leaf reached with an undecided flight".into()),
                }
            }

            if !candidate_is_feasible(scenario, node.source, &complete_choices) {
                stats.infeasible_complete_assignments =
                    stats.infeasible_complete_assignments.saturating_add(1);
                continue;
            }
            let Some(cost) = candidate_cost(scenario, node.source, &complete_choices) else {
                return Err("objective cost overflowed on a feasible assignment".into());
            };
            if best_plan
                .as_ref()
                .is_some_and(|incumbent| incumbent.total_cost <= cost)
            {
                continue;
            }

            let plan = Plan {
                scenario_id: scenario.id.clone(),
                configuration_revision: scenario.configuration_revision.clone(),
                input_digest: scenario.input_digest.clone(),
                part_source: node.source,
                assignments: scenario
                    .flights
                    .iter()
                    .zip(complete_choices.iter())
                    .map(|(flight, aircraft_index)| FlightAssignment {
                        flight_id: flight.id.clone(),
                        aircraft_id: aircraft_index.map(|index| scenario.aircraft[index].id.clone()),
                    })
                    .collect(),
                total_cost: cost,
            };
            let verification = verify_plan(scenario, &plan);
            if verification.status != VerificationStatus::Verified
                || verification.recomputed_cost != Some(cost)
            {
                return Err(format!(
                    "planner/verifier disagreement for branch-and-bound incumbent: {:?}",
                    verification.violations
                ));
            }
            best_plan = Some(plan);
            continue;
        }

        let flight_index = node.depth;
        for aircraft_index in 0..scenario.aircraft.len() {
            let mut choices = node.choices.clone();
            choices[flight_index] = Some(Some(aircraft_index));
            if !partial_assignment_is_feasible(
                scenario,
                node.source,
                &choices,
                flight_index + 1,
            ) {
                stats.partial_feasibility_prunes =
                    stats.partial_feasibility_prunes.saturating_add(1);
                continue;
            }
            push_recovery_child(
                &mut frontier,
                &mut stats,
                &mut serial,
                &mut best_plan,
                RecoverySearchNode {
                    source: node.source,
                    depth: flight_index + 1,
                    choices,
                    lower_bound: node.lower_bound,
                    serial: 0,
                },
            );
            serial = serial.saturating_add(1);
        }

        let mut cancelled_choices = node.choices.clone();
        cancelled_choices[flight_index] = Some(None);
        let cancellation_bound = node
            .lower_bound
            .checked_add(scenario.flights[flight_index].cancellation_cost)
            .ok_or_else(|| "partial cancellation bound overflows u64".to_string())?;
        push_recovery_child(
            &mut frontier,
            &mut stats,
            &mut serial,
            &mut best_plan,
            RecoverySearchNode {
                source: node.source,
                depth: flight_index + 1,
                choices: cancelled_choices,
                lower_bound: cancellation_bound,
                serial: 0,
            },
        );
        serial = serial.saturating_add(1);
        stats.peak_frontier_nodes = stats.peak_frontier_nodes.max(frontier.len());
    }

    let frontier_bound = frontier.peek().map(|node| node.lower_bound);
    let best_cost = best_plan.as_ref().map(|plan| plan.total_cost);
    let global_lower_bound_cost = match (best_cost, frontier_bound) {
        (Some(best), Some(bound)) => Some(best.min(bound)),
        (Some(best), None) => Some(best),
        (None, Some(bound)) => Some(bound),
        (None, None) => None,
    };
    let absolute_optimality_gap = best_cost.zip(global_lower_bound_cost).map(|(best, lower)| best - lower);

    let status = match (best_cost, absolute_optimality_gap, frontier.is_empty()) {
        (Some(_), Some(gap), _) if gap <= options.max_absolute_optimality_gap => {
            BranchAndBoundStatus::OptimalWithinTolerance
        }
        (Some(_), _, _) => BranchAndBoundStatus::FeasibleUncertified,
        (None, _, true) => BranchAndBoundStatus::NoFeasiblePlan,
        (None, _, false) => BranchAndBoundStatus::NoFeasiblePlanWithinBudget,
    };

    Ok(BranchAndBoundReport {
        scenario_id: scenario.id.clone(),
        configuration_revision: scenario.configuration_revision.clone(),
        input_digest: scenario.input_digest.clone(),
        status,
        best_plan,
        best_objective_cost: best_cost,
        global_lower_bound_cost,
        absolute_optimality_gap,
        accepted_absolute_gap: options.max_absolute_optimality_gap,
        stats,
    })
}

fn push_recovery_child(
    frontier: &mut std::collections::BinaryHeap<RecoverySearchNode>,
    stats: &mut BranchAndBoundStats,
    serial: &mut u64,
    best_plan: &mut Option<Plan>,
    mut node: RecoverySearchNode,
) {
    if best_plan
        .as_ref()
        .is_some_and(|incumbent| node.lower_bound >= incumbent.total_cost)
    {
        stats.objective_bound_prunes = stats.objective_bound_prunes.saturating_add(1);
        return;
    }
    node.serial = *serial;
    frontier.push(node);
    *serial = (*serial).saturating_add(1);
    stats.generated_nodes = stats.generated_nodes.saturating_add(1);
    stats.peak_frontier_nodes = stats.peak_frontier_nodes.max(frontier.len());
}

fn source_base_cost(scenario: &Scenario, source: PartSource) -> Option<u64> {
    match source {
        PartSource::None => Some(0),
        PartSource::QualifiedStock => Some(scenario.maintenance_cost),
        PartSource::QualifiedProduction => scenario
            .production
            .cost
            .checked_add(scenario.maintenance_cost),
    }
}

/// Check only already-decided assignments. A false result proves no completion
/// of this prefix can satisfy the declared aircraft availability/turnaround
/// constraints; undecided flights are deliberately ignored.
fn partial_assignment_is_feasible(
    scenario: &Scenario,
    source: PartSource,
    choices: &[Option<Option<usize>>],
    decided_count: usize,
) -> bool {
    if choices.len() != scenario.flights.len()
        || decided_count > scenario.flights.len()
        || !scenario.source_is_usable(source)
    {
        return false;
    }

    let repair_ready_at = match source {
        PartSource::None => None,
        PartSource::QualifiedStock => Some(scenario.maintenance_duration),
        PartSource::QualifiedProduction => scenario
            .production
            .duration
            .checked_add(scenario.maintenance_duration),
    };

    let mut by_aircraft: Vec<Vec<usize>> = vec![Vec::new(); scenario.aircraft.len()];
    for flight_index in 0..decided_count {
        let decision = match choices[flight_index] {
            Some(decision) => decision,
            None => return false,
        };
        let Some(aircraft_index) = decision else {
            continue;
        };
        let Some(aircraft) = scenario.aircraft.get(aircraft_index) else {
            return false;
        };
        let flight = &scenario.flights[flight_index];
        let ready_at = if aircraft.id == scenario.repair_aircraft_id {
            let Some(repair_ready) = repair_ready_at else {
                return false;
            };
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
        assigned.sort_by_key(|&index| {
            (
                scenario.flights[index].start,
                scenario.flights[index].end,
                scenario.flights[index].id.as_str(),
            )
        });
        let turnaround = scenario.aircraft[aircraft_index].turnaround;
        for pair in assigned.windows(2) {
            let Some(earliest_next) = scenario.flights[pair[0]]
                .end
                .checked_add(turnaround)
            else {
                return false;
            };
            if scenario.flights[pair[1]].start < earliest_next {
                return false;
            }
        }
    }
    true
}
