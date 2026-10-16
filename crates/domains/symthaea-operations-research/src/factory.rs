// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic capability- and qualification-constrained factory-line scheduling.
//!
//! This is a small earliest-finish list-scheduling baseline, not an optimal general
//! job-shop solver or a production authority. It fails closed when process order,
//! worker skills, resource capabilities, or availability cannot support a schedule.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// A single process step with immutable identity, duration, dependencies and required capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactoryOperation {
    pub id: String,
    pub duration_seconds: u64,
    pub predecessors: Vec<String>,
    pub required_skills: Vec<String>,
    pub required_resource_capabilities: Vec<String>,
}

/// A qualified worker available during one contiguous time window for this planning run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactoryWorker {
    pub id: String,
    pub skills: Vec<String>,
    pub available_from_seconds: u64,
    pub available_until_seconds: u64,
}

/// A tool, workstation, inspection station or machine with declared capabilities and availability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactoryResource {
    pub id: String,
    pub capabilities: Vec<String>,
    pub available_from_seconds: u64,
    pub available_until_seconds: u64,
}

/// Assignment of an operation to one worker and one resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledOperation {
    pub operation_id: String,
    pub worker_id: String,
    pub resource_id: String,
    pub start_seconds: u64,
    pub finish_seconds: u64,
}

/// A deterministic feasible schedule plus a resource-independent critical-path lower bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorySchedule {
    /// Sorted by start time, then finish time, then operation ID.
    pub operations: Vec<ScheduledOperation>,
    pub makespan_seconds: u64,
    pub critical_path_lower_bound_seconds: u64,
    /// Difference between this heuristic schedule and the precedence-only lower bound.
    pub heuristic_gap_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactoryScheduleError {
    InvalidInput(String),
    DependencyCycle(Vec<String>),
    NoQualifiedAssignment(String),
    NoFeasibleAvailability(String),
    TimeOverflow(String),
}

impl fmt::Display for FactoryScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid factory schedule input: {message}"),
            Self::DependencyCycle(operations) => {
                write!(f, "factory process graph contains a dependency cycle: {operations:?}")
            }
            Self::NoQualifiedAssignment(operation) => write!(
                f,
                "operation {operation:?} has no worker/resource with all required skills and capabilities"
            ),
            Self::NoFeasibleAvailability(operation) => write!(
                f,
                "operation {operation:?} cannot fit a qualified worker/resource availability window"
            ),
            Self::TimeOverflow(operation) => {
                write!(f, "time arithmetic overflow while scheduling {operation:?}")
            }
        }
    }
}

impl std::error::Error for FactoryScheduleError {}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    // Field order defines deterministic tie-breaking: earliest finish, earliest start,
    // operation ID, worker ID, resource ID.
    finish_seconds: u64,
    start_seconds: u64,
    operation_id: String,
    worker_id: String,
    resource_id: String,
}

/// Build a deterministic feasible schedule using earliest-finish list scheduling.
///
/// Each operation reserves exactly one worker and one resource for its full duration.
/// Every predecessor must finish before an operation starts. Required skill/capability
/// IDs are exact strings and are never inferred from job titles or resource names.
/// Each worker/resource has one availability window for this scheduling run.
///
/// This baseline does not model setup/changeover, shifts that repeat, rework loops,
/// multi-worker tasks, stochastic downtime, or optimal job-shop search. Those features
/// require separately specified semantics and benchmarks.
pub fn plan_factory_line(
    operations: &[FactoryOperation],
    workers: &[FactoryWorker],
    resources: &[FactoryResource],
) -> Result<FactorySchedule, FactoryScheduleError> {
    if operations.is_empty() {
        return Err(FactoryScheduleError::InvalidInput(
            "operations must not be empty".into(),
        ));
    }
    if workers.is_empty() {
        return Err(FactoryScheduleError::InvalidInput(
            "workers must not be empty".into(),
        ));
    }
    if resources.is_empty() {
        return Err(FactoryScheduleError::InvalidInput(
            "resources must not be empty".into(),
        ));
    }

    let operation_map = index_operations(operations)?;
    validate_predecessors(operations, &operation_map)?;
    validate_workers(workers)?;
    validate_resources(resources)?;

    let critical_path_lower_bound_seconds = critical_path_lower_bound(operations, &operation_map)?;

    // Reject permanently unqualified operations before attempting any schedule.
    for operation in operations {
        let has_worker = workers
            .iter()
            .any(|worker| contains_all(&worker.skills, &operation.required_skills));
        let has_resource = resources.iter().any(|resource| {
            contains_all(
                &resource.capabilities,
                &operation.required_resource_capabilities,
            )
        });
        if !has_worker || !has_resource {
            return Err(FactoryScheduleError::NoQualifiedAssignment(
                operation.id.clone(),
            ));
        }
    }

    let worker_map: BTreeMap<&str, &FactoryWorker> = workers
        .iter()
        .map(|worker| (worker.id.as_str(), worker))
        .collect();
    let resource_map: BTreeMap<&str, &FactoryResource> = resources
        .iter()
        .map(|resource| (resource.id.as_str(), resource))
        .collect();

    let mut worker_free_at: BTreeMap<String, u64> = workers
        .iter()
        .map(|worker| (worker.id.clone(), worker.available_from_seconds))
        .collect();
    let mut resource_free_at: BTreeMap<String, u64> = resources
        .iter()
        .map(|resource| (resource.id.clone(), resource.available_from_seconds))
        .collect();

    let mut pending: BTreeSet<String> = operations
        .iter()
        .map(|operation| operation.id.clone())
        .collect();
    let mut finish_by_operation: BTreeMap<String, u64> = BTreeMap::new();
    let mut schedule = Vec::with_capacity(operations.len());

    while !pending.is_empty() {
        let mut best_candidate: Option<Candidate> = None;
        let mut ready_without_assignment = Vec::new();

        for operation_id in &pending {
            let operation = operation_map
                .get(operation_id.as_str())
                .expect("pending operation was validated and indexed");
            if !operation
                .predecessors
                .iter()
                .all(|predecessor| finish_by_operation.contains_key(predecessor))
            {
                continue;
            }
            ready_without_assignment.push(operation_id.clone());

            let dependency_ready_at = operation
                .predecessors
                .iter()
                .filter_map(|predecessor| finish_by_operation.get(predecessor).copied())
                .max()
                .unwrap_or(0);

            for worker in workers {
                if !contains_all(&worker.skills, &operation.required_skills) {
                    continue;
                }
                let worker_ready_at = worker_free_at
                    .get(&worker.id)
                    .copied()
                    .unwrap_or(worker.available_from_seconds)
                    .max(worker.available_from_seconds);

                for resource in resources {
                    if !contains_all(
                        &resource.capabilities,
                        &operation.required_resource_capabilities,
                    ) {
                        continue;
                    }
                    let resource_ready_at = resource_free_at
                        .get(&resource.id)
                        .copied()
                        .unwrap_or(resource.available_from_seconds)
                        .max(resource.available_from_seconds);
                    let start_seconds = dependency_ready_at
                        .max(worker_ready_at)
                        .max(resource_ready_at);
                    let finish_seconds = start_seconds
                        .checked_add(operation.duration_seconds)
                        .ok_or_else(|| {
                            FactoryScheduleError::TimeOverflow(operation.id.clone())
                        })?;

                    if finish_seconds > worker.available_until_seconds
                        || finish_seconds > resource.available_until_seconds
                    {
                        continue;
                    }

                    let candidate = Candidate {
                        finish_seconds,
                        start_seconds,
                        operation_id: operation.id.clone(),
                        worker_id: worker.id.clone(),
                        resource_id: resource.id.clone(),
                    };
                    if best_candidate
                        .as_ref()
                        .is_none_or(|current| candidate < *current)
                    {
                        best_candidate = Some(candidate);
                    }
                }
            }
        }

        let Some(candidate) = best_candidate else {
            if ready_without_assignment.is_empty() {
                return Err(FactoryScheduleError::DependencyCycle(
                    pending.into_iter().collect(),
                ));
            }
            return Err(FactoryScheduleError::NoFeasibleAvailability(
                ready_without_assignment[0].clone(),
            ));
        };

        // The candidate references came from validated stable IDs. Update only the
        // selected worker/resource; the others remain available for parallel work.
        worker_free_at.insert(candidate.worker_id.clone(), candidate.finish_seconds);
        resource_free_at.insert(candidate.resource_id.clone(), candidate.finish_seconds);
        finish_by_operation.insert(candidate.operation_id.clone(), candidate.finish_seconds);
        schedule.push(ScheduledOperation {
            operation_id: candidate.operation_id.clone(),
            worker_id: candidate.worker_id,
            resource_id: candidate.resource_id,
            start_seconds: candidate.start_seconds,
            finish_seconds: candidate.finish_seconds,
        });
        pending.remove(&candidate.operation_id);

        // These maps are intentionally built from the validated input. Touching them
        // here documents that all assignments correspond to known workers/resources.
        debug_assert!(worker_map.contains_key(schedule.last().unwrap().worker_id.as_str()));
        debug_assert!(resource_map.contains_key(schedule.last().unwrap().resource_id.as_str()));
    }

    schedule.sort_by(|a, b| {
        (a.start_seconds, a.finish_seconds, &a.operation_id)
            .cmp(&(b.start_seconds, b.finish_seconds, &b.operation_id))
    });
    let makespan_seconds = schedule
        .iter()
        .map(|operation| operation.finish_seconds)
        .max()
        .unwrap_or(0);
    let heuristic_gap_seconds =
        makespan_seconds.saturating_sub(critical_path_lower_bound_seconds);

    Ok(FactorySchedule {
        operations: schedule,
        makespan_seconds,
        critical_path_lower_bound_seconds,
        heuristic_gap_seconds,
    })
}

fn index_operations<'a>(
    operations: &'a [FactoryOperation],
) -> Result<BTreeMap<&'a str, &'a FactoryOperation>, FactoryScheduleError> {
    let mut indexed = BTreeMap::new();
    for operation in operations {
        validate_non_empty("operation ID", &operation.id)?;
        if operation.duration_seconds == 0 {
            return Err(FactoryScheduleError::InvalidInput(format!(
                "operation {:?} duration must be greater than zero",
                operation.id
            )));
        }
        validate_unique_non_empty_ids(
            &format!("required skills for operation {:?}", operation.id),
            &operation.required_skills,
        )?;
        validate_unique_non_empty_ids(
            &format!("required resource capabilities for operation {:?}", operation.id),
            &operation.required_resource_capabilities,
        )?;
        validate_unique_non_empty_ids(
            &format!("predecessors for operation {:?}", operation.id),
            &operation.predecessors,
        )?;
        if indexed.insert(operation.id.as_str(), operation).is_some() {
            return Err(FactoryScheduleError::InvalidInput(format!(
                "duplicate operation ID {:?}",
                operation.id
            )));
        }
    }
    Ok(indexed)
}

fn validate_predecessors(
    operations: &[FactoryOperation],
    indexed: &BTreeMap<&str, &FactoryOperation>,
) -> Result<(), FactoryScheduleError> {
    for operation in operations {
        for predecessor in &operation.predecessors {
            if !indexed.contains_key(predecessor.as_str()) {
                return Err(FactoryScheduleError::InvalidInput(format!(
                    "operation {:?} references unknown predecessor {:?}",
                    operation.id, predecessor
                )));
            }
            if predecessor == &operation.id {
                return Err(FactoryScheduleError::InvalidInput(format!(
                    "operation {:?} cannot depend on itself",
                    operation.id
                )));
            }
        }
    }
    Ok(())
}

fn validate_workers(workers: &[FactoryWorker]) -> Result<(), FactoryScheduleError> {
    let mut ids = BTreeSet::new();
    for worker in workers {
        validate_non_empty("worker ID", &worker.id)?;
        validate_unique_non_empty_ids(&format!("skills for worker {:?}", worker.id), &worker.skills)?;
        if worker.available_from_seconds > worker.available_until_seconds {
            return Err(FactoryScheduleError::InvalidInput(format!(
                "worker {:?} availability starts after it ends",
                worker.id
            )));
        }
        if !ids.insert(worker.id.as_str()) {
            return Err(FactoryScheduleError::InvalidInput(format!(
                "duplicate worker ID {:?}",
                worker.id
            )));
        }
    }
    Ok(())
}

fn validate_resources(resources: &[FactoryResource]) -> Result<(), FactoryScheduleError> {
    let mut ids = BTreeSet::new();
    for resource in resources {
        validate_non_empty("resource ID", &resource.id)?;
        validate_unique_non_empty_ids(
            &format!("capabilities for resource {:?}", resource.id),
            &resource.capabilities,
        )?;
        if resource.available_from_seconds > resource.available_until_seconds {
            return Err(FactoryScheduleError::InvalidInput(format!(
                "resource {:?} availability starts after it ends",
                resource.id
            )));
        }
        if !ids.insert(resource.id.as_str()) {
            return Err(FactoryScheduleError::InvalidInput(format!(
                "duplicate resource ID {:?}",
                resource.id
            )));
        }
    }
    Ok(())
}

fn validate_non_empty(field: &str, value: &str) -> Result<(), FactoryScheduleError> {
    if value.trim().is_empty() {
        return Err(FactoryScheduleError::InvalidInput(format!(
            "{field} must not be empty"
        )));
    }
    Ok(())
}

fn validate_unique_non_empty_ids(
    field: &str,
    ids: &[String],
) -> Result<(), FactoryScheduleError> {
    let mut seen = BTreeSet::new();
    for id in ids {
        validate_non_empty(field, id)?;
        if !seen.insert(id.as_str()) {
            return Err(FactoryScheduleError::InvalidInput(format!(
                "{field} contains duplicate ID {id:?}"
            )));
        }
    }
    Ok(())
}

fn contains_all(declared: &[String], required: &[String]) -> bool {
    required.iter().all(|requirement| declared.contains(requirement))
}

/// Precedence-only lower bound. Ignores resource/worker contention, so it is a lower
/// bound for any feasible schedule, not a promised achievable makespan.
fn critical_path_lower_bound(
    operations: &[FactoryOperation],
    indexed: &BTreeMap<&str, &FactoryOperation>,
) -> Result<u64, FactoryScheduleError> {
    let mut pending: BTreeSet<String> = operations
        .iter()
        .map(|operation| operation.id.clone())
        .collect();
    let mut earliest_finish: BTreeMap<String, u64> = BTreeMap::new();

    while !pending.is_empty() {
        let ready: Vec<String> = pending
            .iter()
            .filter(|operation_id| {
                indexed[*operation_id]
                    .predecessors
                    .iter()
                    .all(|predecessor| earliest_finish.contains_key(predecessor))
            })
            .cloned()
            .collect();
        if ready.is_empty() {
            return Err(FactoryScheduleError::DependencyCycle(
                pending.into_iter().collect(),
            ));
        }

        for operation_id in ready {
            let operation = indexed[operation_id.as_str()];
            let predecessor_finish = operation
                .predecessors
                .iter()
                .filter_map(|predecessor| earliest_finish.get(predecessor).copied())
                .max()
                .unwrap_or(0);
            let finish = predecessor_finish
                .checked_add(operation.duration_seconds)
                .ok_or_else(|| FactoryScheduleError::TimeOverflow(operation.id.clone()))?;
            earliest_finish.insert(operation_id.clone(), finish);
            pending.remove(&operation_id);
        }
    }

    Ok(earliest_finish.values().copied().max().unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn operation(
        id: &str,
        duration_seconds: u64,
        predecessors: &[&str],
        skills: &[&str],
        capabilities: &[&str],
    ) -> FactoryOperation {
        FactoryOperation {
            id: id.into(),
            duration_seconds,
            predecessors: predecessors.iter().map(|value| (*value).into()).collect(),
            required_skills: skills.iter().map(|value| (*value).into()).collect(),
            required_resource_capabilities: capabilities
                .iter()
                .map(|value| (*value).into())
                .collect(),
        }
    }

    fn worker(id: &str, skills: &[&str], from: u64, until: u64) -> FactoryWorker {
        FactoryWorker {
            id: id.into(),
            skills: skills.iter().map(|value| (*value).into()).collect(),
            available_from_seconds: from,
            available_until_seconds: until,
        }
    }

    fn resource(id: &str, capabilities: &[&str], from: u64, until: u64) -> FactoryResource {
        FactoryResource {
            id: id.into(),
            capabilities: capabilities.iter().map(|value| (*value).into()).collect(),
            available_from_seconds: from,
            available_until_seconds: until,
        }
    }

    fn aircraft_line() -> (Vec<FactoryOperation>, Vec<FactoryWorker>, Vec<FactoryResource>) {
        let operations = vec![
            operation("incoming_inspection", 10, &[], &["inspection"], &["metrology"]),
            operation("cut_members", 15, &["incoming_inspection"], &["machining"], &["cnc"]),
            operation("build_harness", 10, &["incoming_inspection"], &["electrical"], &["electrical_test"]),
            operation("structural_subassembly", 20, &["cut_members"], &["assembly"], &["assembly_fixture"]),
            operation("final_assembly", 30, &["structural_subassembly", "build_harness"], &["assembly"], &["assembly_fixture"]),
            operation("final_inspection", 15, &["final_assembly"], &["inspection"], &["metrology"]),
        ];
        let workers = vec![
            worker("inspector", &["inspection"], 0, 200),
            worker("machinist", &["machining"], 0, 200),
            worker("electrician", &["electrical"], 0, 200),
            worker("assembler", &["assembly"], 0, 200),
        ];
        let resources = vec![
            resource("metrology_station", &["metrology"], 0, 200),
            resource("cnc_machine", &["cnc"], 0, 200),
            resource("harness_bench", &["electrical_test"], 0, 200),
            resource("assembly_fixture", &["assembly_fixture"], 0, 200),
        ];
        (operations, workers, resources)
    }

    fn assert_schedule_invariants(
        operations: &[FactoryOperation],
        workers: &[FactoryWorker],
        resources: &[FactoryResource],
        schedule: &FactorySchedule,
    ) {
        assert_eq!(schedule.operations.len(), operations.len());
        let operation_map: BTreeMap<&str, &FactoryOperation> = operations
            .iter()
            .map(|operation| (operation.id.as_str(), operation))
            .collect();
        let worker_map: BTreeMap<&str, &FactoryWorker> = workers
            .iter()
            .map(|worker| (worker.id.as_str(), worker))
            .collect();
        let resource_map: BTreeMap<&str, &FactoryResource> = resources
            .iter()
            .map(|resource| (resource.id.as_str(), resource))
            .collect();

        let mut seen = BTreeSet::new();
        for assignment in &schedule.operations {
            assert!(seen.insert(assignment.operation_id.as_str()));
            let operation = operation_map[assignment.operation_id.as_str()];
            let worker = worker_map[assignment.worker_id.as_str()];
            let resource = resource_map[assignment.resource_id.as_str()];
            assert_eq!(
                assignment.finish_seconds - assignment.start_seconds,
                operation.duration_seconds
            );
            assert!(assignment.start_seconds >= worker.available_from_seconds);
            assert!(assignment.finish_seconds <= worker.available_until_seconds);
            assert!(assignment.start_seconds >= resource.available_from_seconds);
            assert!(assignment.finish_seconds <= resource.available_until_seconds);
            assert!(contains_all(&worker.skills, &operation.required_skills));
            assert!(contains_all(
                &resource.capabilities,
                &operation.required_resource_capabilities
            ));
            for predecessor in &operation.predecessors {
                let predecessor_assignment = schedule
                    .operations
                    .iter()
                    .find(|other| other.operation_id == *predecessor)
                    .unwrap();
                assert!(predecessor_assignment.finish_seconds <= assignment.start_seconds);
            }
        }
        assert_eq!(seen.len(), operations.len());

        for left in 0..schedule.operations.len() {
            for right in left + 1..schedule.operations.len() {
                let a = &schedule.operations[left];
                let b = &schedule.operations[right];
                let overlaps = a.start_seconds < b.finish_seconds && b.start_seconds < a.finish_seconds;
                if a.worker_id == b.worker_id || a.resource_id == b.resource_id {
                    assert!(!overlaps, "overlap in resource/worker assignment: {a:?}, {b:?}");
                }
            }
        }
    }

    #[test]
    fn aircraft_line_schedule_respects_dependencies_and_parallel_preparation() {
        let (operations, workers, resources) = aircraft_line();
        let schedule = plan_factory_line(&operations, &workers, &resources).unwrap();
        assert_schedule_invariants(&operations, &workers, &resources, &schedule);
        assert_eq!(schedule.makespan_seconds, 90);
        assert_eq!(schedule.critical_path_lower_bound_seconds, 90);
        assert_eq!(schedule.heuristic_gap_seconds, 0);

        let cut = schedule.operations.iter().find(|op| op.operation_id == "cut_members").unwrap();
        let harness = schedule.operations.iter().find(|op| op.operation_id == "build_harness").unwrap();
        assert_eq!(cut.start_seconds, harness.start_seconds);
        assert!(cut.finish_seconds >= harness.finish_seconds);
    }

    #[test]
    fn identical_input_produces_identical_schedule() {
        let (operations, workers, resources) = aircraft_line();
        let first = plan_factory_line(&operations, &workers, &resources).unwrap();
        let second = plan_factory_line(&operations, &workers, &resources).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn missing_inspection_skill_or_metrology_resource_fails_closed() {
        let (operations, mut workers, resources) = aircraft_line();
        workers.retain(|worker| worker.id != "inspector");
        assert_eq!(
            plan_factory_line(&operations, &workers, &resources),
            Err(FactoryScheduleError::NoQualifiedAssignment(
                "incoming_inspection".into()
            ))
        );

        let (operations, workers, mut resources) = aircraft_line();
        resources.retain(|resource| resource.id != "metrology_station");
        assert_eq!(
            plan_factory_line(&operations, &workers, &resources),
            Err(FactoryScheduleError::NoQualifiedAssignment(
                "incoming_inspection".into()
            ))
        );
    }

    #[test]
    fn cycles_and_unknown_predecessors_are_rejected() {
        let cyclic = vec![
            operation("a", 1, &["b"], &[], &[]),
            operation("b", 1, &["a"], &[], &[]),
        ];
        let basic_worker = vec![worker("worker", &[], 0, 10)];
        let basic_resource = vec![resource("station", &[], 0, 10)];
        assert!(matches!(
            plan_factory_line(&cyclic, &basic_worker, &basic_resource),
            Err(FactoryScheduleError::DependencyCycle(_))
        ));

        let unknown = vec![operation("a", 1, &["not-there"], &[], &[])];
        assert!(matches!(
            plan_factory_line(&unknown, &basic_worker, &basic_resource),
            Err(FactoryScheduleError::InvalidInput(message)) if message.contains("unknown predecessor")
        ));
    }

    #[test]
    fn unavailable_window_duplicate_identity_and_invalid_duration_fail() {
        let operations = vec![operation("inspect", 10, &[], &["inspection"], &["metrology"])];
        let workers = vec![worker("inspector", &["inspection"], 0, 9)];
        let resources = vec![resource("station", &["metrology"], 0, 100)];
        assert_eq!(
            plan_factory_line(&operations, &workers, &resources),
            Err(FactoryScheduleError::NoFeasibleAvailability("inspect".into()))
        );

        let duplicates = vec![
            operation("inspect", 10, &[], &[], &[]),
            operation("inspect", 5, &[], &[], &[]),
        ];
        assert!(matches!(
            plan_factory_line(&duplicates, &workers, &resources),
            Err(FactoryScheduleError::InvalidInput(message)) if message.contains("duplicate operation ID")
        ));

        let zero = vec![operation("inspect", 0, &[], &[], &[])];
        assert!(matches!(
            plan_factory_line(&zero, &workers, &resources),
            Err(FactoryScheduleError::InvalidInput(message)) if message.contains("duration")
        ));
    }

    #[test]
    fn impossible_time_arithmetic_fails_closed() {
        let operations = vec![operation("overflow", u64::MAX, &[], &[], &[])];
        let workers = vec![worker("worker", &[], 1, u64::MAX)];
        let resources = vec![resource("station", &[], 1, u64::MAX)];
        assert_eq!(
            plan_factory_line(&operations, &workers, &resources),
            Err(FactoryScheduleError::TimeOverflow("overflow".into()))
        );
    }
}
