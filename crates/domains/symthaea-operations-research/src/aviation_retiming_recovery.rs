// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tiny delay-aware aircraft-recovery research model.
//!
//! Unlike AECB-001, this model makes schedule retiming a decision: each flight
//! can be assigned to an aircraft at an integer delay or cancelled. The
//! objective is delay cost plus cancellation cost. Its best-first branch-and-
//! bound solver uses committed decision costs as a lower bound and checks
//! retimed availability/turnaround constraints on every partial assignment.
//!
//! This is still a small synthetic model: no crew, passengers, airports,
//! airspace, weather, operational data, or dispatch authority is represented.

pub const MAX_RETIMING_FLIGHTS: usize = 6;
pub const MAX_RETIMING_AIRCRAFT: usize = 4;
pub const MAX_FLIGHT_DELAY_MINUTES: u32 = 60;

/// A flight whose departure may be retimed within a frozen integer-minute window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetimingFlight {
    pub id: String,
    pub scheduled_start_minutes: u32,
    pub duration_minutes: u32,
    pub max_delay_minutes: u32,
    pub delay_cost_per_minute: u64,
    pub cancellation_cost: u64,
}

/// An aircraft with an availability boundary and required post-flight turnaround.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetimingAircraft {
    pub id: String,
    pub available_at_minutes: u32,
    pub turnaround_minutes: u32,
}

/// Frozen synthetic recovery scenario. `input_digest` is caller-supplied
/// provenance; this module checks presence but does not hash or authenticate it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetimingScenario {
    pub id: String,
    pub configuration_revision: String,
    pub input_digest: String,
    pub flights: Vec<RetimingFlight>,
    pub aircraft: Vec<RetimingAircraft>,
}

impl RetimingScenario {
    pub fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty()
            || self.configuration_revision.trim().is_empty()
            || self.input_digest.trim().is_empty()
        {
            return Err("scenario identity, configuration revision and input digest are required".into());
        }
        if self.flights.is_empty() || self.flights.len() > MAX_RETIMING_FLIGHTS {
            return Err(format!("scenario must contain 1..={MAX_RETIMING_FLIGHTS} flights"));
        }
        if self.aircraft.is_empty() || self.aircraft.len() > MAX_RETIMING_AIRCRAFT {
            return Err(format!("scenario must contain 1..={MAX_RETIMING_AIRCRAFT} aircraft"));
        }

        let mut maximum_cost = 0_u64;
        for (index, flight) in self.flights.iter().enumerate() {
            if flight.id.trim().is_empty() {
                return Err(format!("flight {index} has an empty ID"));
            }
            if self.flights[..index].iter().any(|prior| prior.id == flight.id) {
                return Err(format!("duplicate flight ID: {}", flight.id));
            }
            if flight.duration_minutes == 0 {
                return Err(format!("flight {} has zero duration", flight.id));
            }
            if flight.max_delay_minutes > MAX_FLIGHT_DELAY_MINUTES {
                return Err(format!(
                    "flight {} delay window exceeds {MAX_FLIGHT_DELAY_MINUTES} minutes",
                    flight.id
                ));
            }
            flight
                .scheduled_start_minutes
                .checked_add(flight.max_delay_minutes)
                .and_then(|start| start.checked_add(flight.duration_minutes))
                .ok_or_else(|| format!("flight {} time arithmetic overflows u32", flight.id))?;

            let maximum_delay_cost = flight
                .delay_cost_per_minute
                .checked_mul(u64::from(flight.max_delay_minutes))
                .ok_or_else(|| format!("flight {} delay cost overflows u64", flight.id))?;
            maximum_cost = maximum_cost
                .checked_add(maximum_delay_cost.max(flight.cancellation_cost))
                .ok_or_else(|| "maximum total recovery cost overflows u64".to_string())?;
        }
        for (index, aircraft) in self.aircraft.iter().enumerate() {
            if aircraft.id.trim().is_empty() {
                return Err(format!("aircraft {index} has an empty ID"));
            }
            if self.aircraft[..index].iter().any(|prior| prior.id == aircraft.id) {
                return Err(format!("duplicate aircraft ID: {}", aircraft.id));
            }
        }
        let _maximum_scenario_cost = maximum_cost;
        Ok(())
    }
}

/// One decision per flight. A cancellation must use `aircraft_id = None` and
/// `delay_minutes = 0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetimingAssignment {
    pub flight_id: String,
    pub aircraft_id: Option<String>,
    pub delay_minutes: u32,
}

/// Complete plan emitted by the delay-aware search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetimingPlan {
    pub scenario_id: String,
    pub configuration_revision: String,
    pub input_digest: String,
    pub assignments: Vec<RetimingAssignment>,
    pub total_cost: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetimingVerificationStatus {
    Verified,
    Rejected,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetimingVerificationReport {
    pub status: RetimingVerificationStatus,
    pub violations: Vec<String>,
    pub recomputed_cost: Option<u64>,
}

/// Independent stable-ID plan checker. It resolves the submitted plan rather
/// than trusting vector indexes or the solver's internal cost.
pub fn verify_retiming_plan(
    scenario: &RetimingScenario,
    plan: &RetimingPlan,
) -> RetimingVerificationReport {
    if let Err(reason) = scenario.validate() {
        return RetimingVerificationReport {
            status: RetimingVerificationStatus::Indeterminate,
            violations: vec![reason],
            recomputed_cost: None,
        };
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
    if plan.assignments.len() != scenario.flights.len() {
        violations.push("plan must contain exactly one assignment per scenario flight".to_string());
    }

    #[derive(Clone, Copy)]
    struct ResolvedAssignment {
        aircraft_index: Option<usize>,
        delay_minutes: u32,
    }

    let mut seen = vec![false; scenario.flights.len()];
    let mut resolved: Vec<Option<ResolvedAssignment>> = vec![None; scenario.flights.len()];
    for assignment in &plan.assignments {
        let Some(flight_index) = scenario
            .flights
            .iter()
            .position(|flight| flight.id == assignment.flight_id)
        else {
            violations.push(format!("unknown flight ID in plan: {}", assignment.flight_id));
            continue;
        };
        if seen[flight_index] {
            violations.push(format!(
                "duplicate assignment record for flight {}",
                assignment.flight_id
            ));
            continue;
        }
        seen[flight_index] = true;
        let aircraft_index = match assignment.aircraft_id.as_deref() {
            None => None,
            Some(aircraft_id) => match scenario
                .aircraft
                .iter()
                .position(|aircraft| aircraft.id == aircraft_id)
            {
                Some(index) => Some(index),
                None => {
                    violations.push(format!("unknown aircraft ID in plan: {aircraft_id}"));
                    continue;
                }
            },
        };
        resolved[flight_index] = Some(ResolvedAssignment {
            aircraft_index,
            delay_minutes: assignment.delay_minutes,
        });
    }
    for (index, flight) in scenario.flights.iter().enumerate() {
        if !seen[index] {
            violations.push(format!("flight {} is missing from plan", flight.id));
        }
    }

    let mut by_aircraft: Vec<Vec<(u32, u32, String)>> =
        vec![Vec::new(); scenario.aircraft.len()];
    let mut recomputed_cost = Some(0_u64);
    for (flight_index, flight) in scenario.flights.iter().enumerate() {
        let Some(assignment) = resolved[flight_index] else {
            recomputed_cost = None;
            continue;
        };
        match assignment.aircraft_index {
            None => {
                if assignment.delay_minutes != 0 {
                    violations.push(format!(
                        "cancelled flight {} must not carry a delay",
                        flight.id
                    ));
                }
                recomputed_cost = recomputed_cost.and_then(|cost| cost.checked_add(flight.cancellation_cost));
            }
            Some(aircraft_index) => {
                if assignment.delay_minutes > flight.max_delay_minutes {
                    violations.push(format!(
                        "flight {} delay exceeds its declared maximum",
                        flight.id
                    ));
                }
                let Some(actual_start) = flight
                    .scheduled_start_minutes
                    .checked_add(assignment.delay_minutes)
                else {
                    violations.push(format!("flight {} actual start overflows", flight.id));
                    recomputed_cost = None;
                    continue;
                };
                let Some(actual_end) = actual_start.checked_add(flight.duration_minutes) else {
                    violations.push(format!("flight {} actual end overflows", flight.id));
                    recomputed_cost = None;
                    continue;
                };
                let aircraft = &scenario.aircraft[aircraft_index];
                if actual_start < aircraft.available_at_minutes {
                    violations.push(format!(
                        "aircraft {} is not available for flight {}",
                        aircraft.id, flight.id
                    ));
                }
                by_aircraft[aircraft_index].push((actual_start, actual_end, flight.id.clone()));
                let delay_cost = flight
                    .delay_cost_per_minute
                    .checked_mul(u64::from(assignment.delay_minutes));
                recomputed_cost = match (recomputed_cost, delay_cost) {
                    (Some(cost), Some(delay_cost)) => cost.checked_add(delay_cost),
                    _ => None,
                };
            }
        }
    }

    for (aircraft_index, assigned) in by_aircraft.iter_mut().enumerate() {
        assigned.sort_by(|left, right| {
            (left.0, left.1, left.2.as_str()).cmp(&(right.0, right.1, right.2.as_str()))
        });
        let turnaround = scenario.aircraft[aircraft_index].turnaround_minutes;
        for pair in assigned.windows(2) {
            match pair[0].1.checked_add(turnaround) {
                Some(earliest_next) if earliest_next <= pair[1].0 => {}
                Some(_) => violations.push(format!(
                    "overlap or insufficient turnaround for aircraft {} between {} and {}",
                    scenario.aircraft[aircraft_index].id, pair[0].2, pair[1].2
                )),
                None => violations.push(format!(
                    "turnaround arithmetic overflows for aircraft {}",
                    scenario.aircraft[aircraft_index].id
                )),
            }
        }
    }

    if let Some(cost) = recomputed_cost {
        if cost != plan.total_cost {
            violations.push(format!(
                "declared objective {} differs from independently recomputed cost {cost}",
                plan.total_cost
            ));
        }
    } else {
        violations.push("independent objective recomputation overflowed or lacked a complete plan".into());
    }

    if violations.is_empty() {
        RetimingVerificationReport {
            status: RetimingVerificationStatus::Verified,
            violations,
            recomputed_cost,
        }
    } else {
        RetimingVerificationReport {
            status: RetimingVerificationStatus::Rejected,
            violations,
            recomputed_cost,
        }
    }
}

/// Diagnostics from the deterministic greedy baseline.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetimingGreedyStats {
    pub processed_flights: u64,
    pub candidate_actions_considered: u64,
    pub feasible_candidate_actions_considered: u64,
    pub assigned_flight_count: u64,
    pub cancelled_flight_count: u64,
    pub total_delay_minutes: u64,
}

/// A verified feasible plan and transparent greedy work counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetimingGreedyReport {
    pub plan: RetimingPlan,
    pub stats: RetimingGreedyStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetimingSearchOptions {
    pub max_expanded_nodes: u64,
    /// Caps all enqueued nodes (including the root) and bounds frontier growth.
    pub max_generated_nodes: u64,
    pub max_absolute_optimality_gap: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetimingSearchStatus {
    OptimalWithinTolerance,
    FeasibleUncertified,
    NoFeasiblePlan,
    NoFeasiblePlanWithinBudget,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetimingSearchStats {
    pub expanded_nodes: u64,
    pub generated_nodes: u64,
    pub complete_assignments_examined: u64,
    pub verified_complete_assignments: u64,
    pub partial_feasibility_prunes: u64,
    pub objective_bound_prunes: u64,
    pub peak_frontier_nodes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetimingSearchReport {
    pub scenario_id: String,
    pub configuration_revision: String,
    pub input_digest: String,
    pub status: RetimingSearchStatus,
    pub best_plan: Option<RetimingPlan>,
    pub best_objective_cost: Option<u64>,
    pub global_lower_bound_cost: Option<u64>,
    pub absolute_optimality_gap: Option<u64>,
    pub accepted_absolute_gap: u64,
    pub stats: RetimingSearchStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetimingChoice {
    Cancel,
    Assign {
        aircraft_index: usize,
        delay_minutes: u32,
    },
}

#[derive(Debug, Clone)]
struct RetimingSearchNode {
    depth: usize,
    choices: Vec<Option<RetimingChoice>>,
    /// Exact cost of the actions already committed in this search prefix.
    committed_cost: u64,
    /// Admissible lower bound: committed cost plus an availability-only
    /// relaxation for each unresolved flight.
    lower_bound: u64,
    serial: u64,
}

impl PartialEq for RetimingSearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.lower_bound == other.lower_bound && self.serial == other.serial
    }
}

impl Eq for RetimingSearchNode {}

impl Ord for RetimingSearchNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .lower_bound
            .cmp(&self.lower_bound)
            .then_with(|| other.serial.cmp(&self.serial))
    }
}

impl PartialOrd for RetimingSearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Best-first branch-and-bound over every allowed aircraft/delay decision and
/// the explicit cancellation decision for every flight.
///
/// The admissible lower bound is the cost of already-fixed actions plus, for
/// every unresolved flight, the cheapest of (a) its minimum delay needed to meet
/// any aircraft's individual availability or (b) cancellation. The relaxation
/// ignores conflicts among unresolved flights and between unresolved and
/// committed flights, so it can underestimate but cannot overestimate the true
/// completion cost. A finite generated-node cap prevents frontier growth without
/// bound. If an expansion would exceed that cap, its parent remains on the
/// frontier so the reported global bound still covers its unexplored subtree.
pub fn solve_retiming_branch_and_bound(
    scenario: &RetimingScenario,
    options: RetimingSearchOptions,
) -> Result<RetimingSearchReport, String> {
    scenario.validate()?;
    if options.max_expanded_nodes == 0 || options.max_generated_nodes == 0 {
        return Err("node budgets must be greater than zero".into());
    }

    let mut frontier = std::collections::BinaryHeap::<RetimingSearchNode>::new();
    let root_lower_bound = retiming_optimistic_lower_bound(scenario, 0, 0)
        .ok_or_else(|| "root recovery lower bound overflows u64".to_string())?;
    frontier.push(RetimingSearchNode {
        depth: 0,
        choices: vec![None; scenario.flights.len()],
        committed_cost: 0,
        lower_bound: root_lower_bound,
        serial: 0,
    });
    let mut serial = 1_u64;
    let mut stats = RetimingSearchStats {
        generated_nodes: 1,
        peak_frontier_nodes: 1,
        ..RetimingSearchStats::default()
    };
    let mut best_plan: Option<RetimingPlan> = None;
    let mut stopped_by_generation_budget = false;

    while let Some(frontier_best) = frontier.peek() {
        if let Some(incumbent) = &best_plan {
            let bound = frontier_best.lower_bound.min(incumbent.total_cost);
            if incumbent.total_cost - bound <= options.max_absolute_optimality_gap {
                break;
            }
        }
        if stats.expanded_nodes >= options.max_expanded_nodes {
            break;
        }
        let Some(node) = frontier.pop() else {
            break;
        };

        if node.depth == scenario.flights.len() {
            stats.expanded_nodes = stats.expanded_nodes.saturating_add(1);
            stats.complete_assignments_examined =
                stats.complete_assignments_examined.saturating_add(1);
            let mut assignments = Vec::with_capacity(node.choices.len());
            for (index, choice) in node.choices.iter().enumerate() {
                let Some(choice) = choice else {
                    return Err(format!("complete node has undecided flight at index {index}"));
                };
                let flight = &scenario.flights[index];
                assignments.push(match choice {
                    RetimingChoice::Cancel => RetimingAssignment {
                        flight_id: flight.id.clone(),
                        aircraft_id: None,
                        delay_minutes: 0,
                    },
                    RetimingChoice::Assign {
                        aircraft_index,
                        delay_minutes,
                    } => RetimingAssignment {
                        flight_id: flight.id.clone(),
                        aircraft_id: Some(scenario.aircraft[*aircraft_index].id.clone()),
                        delay_minutes: *delay_minutes,
                    },
                });
            }
            let plan = RetimingPlan {
                scenario_id: scenario.id.clone(),
                configuration_revision: scenario.configuration_revision.clone(),
                input_digest: scenario.input_digest.clone(),
                assignments,
                total_cost: node.committed_cost,
            };
            let verification = verify_retiming_plan(scenario, &plan);
            if verification.status != RetimingVerificationStatus::Verified
                || verification.recomputed_cost != Some(node.committed_cost)
            {
                return Err(format!(
                    "search/verifier disagreement on complete assignment: {:?}",
                    verification.violations
                ));
            }
            stats.verified_complete_assignments =
                stats.verified_complete_assignments.saturating_add(1);
            if best_plan
                .as_ref()
                .is_none_or(|incumbent| plan.total_cost < incumbent.total_cost)
            {
                best_plan = Some(plan);
            }
            continue;
        }

        let flight_index = node.depth;
        let flight = &scenario.flights[flight_index];
        let mut children = Vec::<RetimingSearchNode>::new();
        let mut partial_prunes = 0_u64;

        for aircraft_index in 0..scenario.aircraft.len() {
            for delay_minutes in 0..=flight.max_delay_minutes {
                let delay_cost = flight
                    .delay_cost_per_minute
                    .checked_mul(u64::from(delay_minutes))
                    .ok_or_else(|| format!("flight {} delay cost overflows", flight.id))?;
                let committed_cost = node
                    .committed_cost
                    .checked_add(delay_cost)
                    .ok_or_else(|| "partial delay objective overflows u64".to_string())?;
                let mut choices = node.choices.clone();
                choices[flight_index] = Some(RetimingChoice::Assign {
                    aircraft_index,
                    delay_minutes,
                });
                if !partial_retiming_is_feasible(
                    scenario,
                    &choices,
                    flight_index + 1,
                ) {
                    partial_prunes = partial_prunes.saturating_add(1);
                    continue;
                }
                let lower_bound = retiming_optimistic_lower_bound(
                    scenario,
                    flight_index + 1,
                    committed_cost,
                )
                .ok_or_else(|| "partial recovery lower bound overflows u64".to_string())?;
                children.push(RetimingSearchNode {
                    depth: flight_index + 1,
                    choices,
                    committed_cost,
                    lower_bound,
                    serial: 0,
                });
            }
        }

        let cancellation_cost = node
            .committed_cost
            .checked_add(flight.cancellation_cost)
            .ok_or_else(|| "partial cancellation objective overflows u64".to_string())?;
        let mut cancelled_choices = node.choices.clone();
        cancelled_choices[flight_index] = Some(RetimingChoice::Cancel);
        let cancellation_bound = retiming_optimistic_lower_bound(
            scenario,
            flight_index + 1,
            cancellation_cost,
        )
        .ok_or_else(|| "partial cancellation lower bound overflows u64".to_string())?;
        children.push(RetimingSearchNode {
            depth: flight_index + 1,
            choices: cancelled_choices,
            committed_cost: cancellation_cost,
            lower_bound: cancellation_bound,
            serial: 0,
        });

        let viable_child_count = children
            .iter()
            .filter(|child| {
                best_plan
                    .as_ref()
                    .is_none_or(|incumbent| child.lower_bound < incumbent.total_cost)
            })
            .count() as u64;
        if stats.generated_nodes.saturating_add(viable_child_count)
            > options.max_generated_nodes
        {
            frontier.push(node);
            stopped_by_generation_budget = true;
            break;
        }

        stats.expanded_nodes = stats.expanded_nodes.saturating_add(1);
        stats.partial_feasibility_prunes = stats
            .partial_feasibility_prunes
            .saturating_add(partial_prunes);
        for mut child in children {
            if best_plan
                .as_ref()
                .is_some_and(|incumbent| child.lower_bound >= incumbent.total_cost)
            {
                stats.objective_bound_prunes = stats.objective_bound_prunes.saturating_add(1);
                continue;
            }
            child.serial = serial;
            serial = serial.saturating_add(1);
            frontier.push(child);
            stats.generated_nodes = stats.generated_nodes.saturating_add(1);
        }
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
    let absolute_optimality_gap = best_cost
        .zip(global_lower_bound_cost)
        .map(|(best, lower)| best - lower);

    let status = match (best_cost, absolute_optimality_gap, frontier.is_empty()) {
        (Some(_), Some(gap), _) if gap <= options.max_absolute_optimality_gap => {
            RetimingSearchStatus::OptimalWithinTolerance
        }
        (Some(_), _, _) => RetimingSearchStatus::FeasibleUncertified,
        (None, _, true) if !stopped_by_generation_budget => RetimingSearchStatus::NoFeasiblePlan,
        (None, _, _) => RetimingSearchStatus::NoFeasiblePlanWithinBudget,
    };

    Ok(RetimingSearchReport {
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

/// Admissible lower bound for a fixed decision prefix. For every unresolved
/// flight, relax conflicts and choose the cheapest action that independently
/// satisfies some aircraft's availability, or cancellation if cheaper. Every
/// feasible completion pays at least that much for each unresolved flight.
fn retiming_optimistic_lower_bound(
    scenario: &RetimingScenario,
    first_unresolved_flight: usize,
    committed_cost: u64,
) -> Option<u64> {
    if first_unresolved_flight > scenario.flights.len() {
        return None;
    }
    let mut lower_bound = committed_cost;
    for flight in &scenario.flights[first_unresolved_flight..] {
        let mut minimum_action_cost = flight.cancellation_cost;
        for aircraft in &scenario.aircraft {
            let minimum_delay = aircraft
                .available_at_minutes
                .saturating_sub(flight.scheduled_start_minutes);
            if minimum_delay > flight.max_delay_minutes {
                continue;
            }
            let delay_cost = flight
                .delay_cost_per_minute
                .checked_mul(u64::from(minimum_delay))?;
            minimum_action_cost = minimum_action_cost.min(delay_cost);
        }
        lower_bound = lower_bound.checked_add(minimum_action_cost)?;
    }
    Some(lower_bound)
}

/// Deterministic, feasibility-preserving greedy baseline. Flights are considered
/// by scheduled start and stable ID. For each flight, it chooses the lowest-cost
/// currently feasible aircraft/delay action, preferring assignment over
/// cancellation on exact cost ties. It never revisits earlier assignments, so
/// it is a baseline for the anytime search, not a claim of global optimality.
pub fn solve_retiming_greedy(
    scenario: &RetimingScenario,
) -> Result<RetimingGreedyReport, String> {
    scenario.validate()?;
    let mut order: Vec<usize> = (0..scenario.flights.len()).collect();
    order.sort_by(|left, right| {
        let a = &scenario.flights[*left];
        let b = &scenario.flights[*right];
        (a.scheduled_start_minutes, a.id.as_str())
            .cmp(&(b.scheduled_start_minutes, b.id.as_str()))
    });

    let mut choices: Vec<Option<RetimingChoice>> = vec![None; scenario.flights.len()];
    let mut stats = RetimingGreedyStats::default();

    for flight_index in order {
        let flight = &scenario.flights[flight_index];
        // The tuple sorts by cost first; then assignment (0) wins over
        // cancellation (1) on equal cost; delay and aircraft ID settle ties.
        let mut best_key = (
            flight.cancellation_cost,
            1_u8,
            0_u32,
            String::new(),
        );
        let mut best_choice = RetimingChoice::Cancel;

        for aircraft_index in 0..scenario.aircraft.len() {
            let aircraft = &scenario.aircraft[aircraft_index];
            for delay_minutes in 0..=flight.max_delay_minutes {
                stats.candidate_actions_considered =
                    stats.candidate_actions_considered.saturating_add(1);
                let action_cost = flight
                    .delay_cost_per_minute
                    .checked_mul(u64::from(delay_minutes))
                    .ok_or_else(|| format!("flight {} delay cost overflows", flight.id))?;
                let key = (action_cost, 0_u8, delay_minutes, aircraft.id.clone());
                choices[flight_index] = Some(RetimingChoice::Assign {
                    aircraft_index,
                    delay_minutes,
                });
                if partial_retiming_is_feasible(scenario, &choices, scenario.flights.len()) {
                    stats.feasible_candidate_actions_considered =
                        stats.feasible_candidate_actions_considered.saturating_add(1);
                    if key < best_key {
                        best_key = key;
                        best_choice = RetimingChoice::Assign {
                            aircraft_index,
                            delay_minutes,
                        };
                    }
                }
                choices[flight_index] = None;
            }
        }

        choices[flight_index] = Some(best_choice);
        stats.processed_flights = stats.processed_flights.saturating_add(1);
    }

    let mut assignments = Vec::with_capacity(scenario.flights.len());
    let mut total_delay_minutes = 0_u64;
    let mut assigned_count = 0_u64;
    let mut cancellation_count = 0_u64;
    let mut total_cost = 0_u64;

    for (index, choice) in choices.iter().enumerate() {
        let flight = &scenario.flights[index];
        let Some(choice) = choice else {
            return Err(format!("greedy baseline left flight {} undecided", flight.id));
        };
        match choice {
            RetimingChoice::Cancel => {
                assignments.push(RetimingAssignment {
                    flight_id: flight.id.clone(),
                    aircraft_id: None,
                    delay_minutes: 0,
                });
                cancellation_count = cancellation_count.saturating_add(1);
                total_cost = total_cost
                    .checked_add(flight.cancellation_cost)
                    .ok_or_else(|| "greedy objective overflows u64".to_string())?;
            }
            RetimingChoice::Assign {
                aircraft_index,
                delay_minutes,
            } => {
                assignments.push(RetimingAssignment {
                    flight_id: flight.id.clone(),
                    aircraft_id: Some(scenario.aircraft[*aircraft_index].id.clone()),
                    delay_minutes: *delay_minutes,
                });
                assigned_count = assigned_count.saturating_add(1);
                total_delay_minutes = total_delay_minutes
                    .checked_add(u64::from(*delay_minutes))
                    .ok_or_else(|| "greedy delay total overflows u64".to_string())?;
                let delay_cost = flight
                    .delay_cost_per_minute
                    .checked_mul(u64::from(*delay_minutes))
                    .ok_or_else(|| "greedy delay cost overflows u64".to_string())?;
                total_cost = total_cost
                    .checked_add(delay_cost)
                    .ok_or_else(|| "greedy objective overflows u64".to_string())?;
            }
        }
    }

    let plan = RetimingPlan {
        scenario_id: scenario.id.clone(),
        configuration_revision: scenario.configuration_revision.clone(),
        input_digest: scenario.input_digest.clone(),
        assignments,
        total_cost,
    };
    let verification = verify_retiming_plan(scenario, &plan);
    if verification.status != RetimingVerificationStatus::Verified
        || verification.recomputed_cost != Some(total_cost)
    {
        return Err(format!(
            "greedy baseline/verifier disagreement: {:?}",
            verification.violations
        ));
    }

    stats.total_delay_minutes = total_delay_minutes;
    stats.assigned_flight_count = assigned_count;
    stats.cancelled_flight_count = cancellation_count;
    Ok(RetimingGreedyReport { plan, stats })
}

/// Validate chosen actions among the first `considered_count` flight indices,
/// skipping undecided entries. Reject unavailable flights and any pair of
/// committed intervals that violate an aircraft's turnaround. Branch-and-bound
/// passes its fully decided prefix; the greedy baseline passes all flight slots
/// while some later slots are still undecided.
fn partial_retiming_is_feasible(
    scenario: &RetimingScenario,
    choices: &[Option<RetimingChoice>],
    considered_count: usize,
) -> bool {
    if choices.len() != scenario.flights.len() || considered_count > scenario.flights.len() {
        return false;
    }
    let mut by_aircraft: Vec<Vec<(u32, u32, usize)>> =
        vec![Vec::new(); scenario.aircraft.len()];
    for flight_index in 0..considered_count {
        match choices[flight_index] {
            None => continue,
            Some(RetimingChoice::Cancel) => {}
            Some(RetimingChoice::Assign {
                aircraft_index,
                delay_minutes,
            }) => {
                let Some(flight) = scenario.flights.get(flight_index) else {
                    return false;
                };
                let Some(aircraft) = scenario.aircraft.get(aircraft_index) else {
                    return false;
                };
                if delay_minutes > flight.max_delay_minutes {
                    return false;
                }
                let Some(actual_start) = flight.scheduled_start_minutes.checked_add(delay_minutes) else {
                    return false;
                };
                let Some(actual_end) = actual_start.checked_add(flight.duration_minutes) else {
                    return false;
                };
                if actual_start < aircraft.available_at_minutes {
                    return false;
                }
                by_aircraft[aircraft_index].push((actual_start, actual_end, flight_index));
            }
        }
    }

    for (aircraft_index, assigned) in by_aircraft.iter_mut().enumerate() {
        assigned.sort_by_key(|(start, end, flight_index)| {
            (*start, *end, scenario.flights[*flight_index].id.as_str())
        });
        let turnaround = scenario.aircraft[aircraft_index].turnaround_minutes;
        for pair in assigned.windows(2) {
            let Some(earliest_next) = pair[0].1.checked_add(turnaround) else {
                return false;
            };
            if earliest_next > pair[1].0 {
                return false;
            }
        }
    }
    true
}
