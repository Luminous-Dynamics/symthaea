//! Deterministic diagnostic event simulator for synchronous barriers vs.
//! asynchronous steady-state worker assignment.
//!
//! This is a test/analysis fixture generator, not a production scheduler.
//! It consumes a fixed task sequence and worker count, emits a WorkerTrace,
//! and delegates all measurement to the existing trace analyzer.

use crate::{AttemptDisposition, ExecutionRegime, WorkerAttempt, WorkerTrace};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationTask {
    pub attempt_id: String,
    pub candidate_id: String,
    pub duration_ms: u64,
    pub disposition: AttemptDisposition,
    pub resource_cost_units: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerSimulationPlan {
    pub worker_count: u32,
    pub wall_clock_budget_ms: u64,
    pub tasks: Vec<SimulationTask>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct WorkerRegimeComparisonReport {
    pub task_count: usize,
    pub worker_count: u32,
    pub wall_clock_budget_ms: u64,
    pub synchronous_makespan_ms: u64,
    pub asynchronous_makespan_ms: u64,
    pub async_makespan_speedup: f64,
    pub same_attempt_count: bool,
    pub same_total_busy_ms: bool,
    pub same_unique_valid_candidates: bool,
    pub same_resource_cost_units: bool,
    pub synchronous: crate::WorkerTraceReport,
    pub asynchronous: crate::WorkerTraceReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimulationError {
    InvalidWorkerCount,
    InvalidWallClockBudget,
    EmptyTasks,
    EmptyIdentifier(&'static str),
    DuplicateAttemptId(String),
    InvalidDuration(String),
    InvalidResourceCost(String),
    ScheduleExceedsWallClockBudget(String),
    CapacityOverflow,
}

impl std::fmt::Display for SimulationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for SimulationError {}

fn validate_tasks(tasks: &[SimulationTask]) -> Result<(), SimulationError> {
    if tasks.is_empty() {
        return Err(SimulationError::EmptyTasks);
    }
    let mut attempt_ids = BTreeSet::new();
    for task in tasks {
        if task.attempt_id.trim().is_empty() {
            return Err(SimulationError::EmptyIdentifier("attempt_id"));
        }
        if task.candidate_id.trim().is_empty() {
            return Err(SimulationError::EmptyIdentifier("candidate_id"));
        }
        if !attempt_ids.insert(task.attempt_id.as_str()) {
            return Err(SimulationError::DuplicateAttemptId(task.attempt_id.clone()));
        }
        if task.duration_ms == 0 {
            return Err(SimulationError::InvalidDuration(task.attempt_id.clone()));
        }
        if !task.resource_cost_units.is_finite() || task.resource_cost_units < 0.0 {
            return Err(SimulationError::InvalidResourceCost(task.attempt_id.clone()));
        }
    }
    Ok(())
}

/// Schedule a fixed task sequence under a synchronous barrier or
/// asynchronous earliest-available-worker policy.
///
/// For synchronous barriers, tasks are dispatched in waves of at most N; the
/// next wave waits for every task in the current wave to finish.
/// For asynchronous steady state, each next task is dispatched to the worker
/// with the earliest available finish time (ties choose the lowest worker ID).
pub fn simulate_worker_regime(
    tasks: &[SimulationTask],
    worker_count: u32,
    wall_clock_budget_ms: u64,
    regime: ExecutionRegime,
) -> Result<WorkerTrace, SimulationError> {
    validate_tasks(tasks)?;
    if worker_count == 0 {
        return Err(SimulationError::InvalidWorkerCount);
    }
    if wall_clock_budget_ms == 0 {
        return Err(SimulationError::InvalidWallClockBudget);
    }

    let mut attempts = Vec::with_capacity(tasks.len());
    match regime {
        ExecutionRegime::SynchronousBarrier => {
            let mut wave_start = 0u64;
            for wave in tasks.chunks(worker_count as usize) {
                let wave_duration = wave.iter().map(|task| task.duration_ms).max().unwrap_or(0);
                for (worker_index, task) in wave.iter().enumerate() {
                    let end_ms = wave_start
                        .checked_add(task.duration_ms)
                        .ok_or(SimulationError::CapacityOverflow)?;
                    if end_ms > wall_clock_budget_ms {
                        return Err(SimulationError::ScheduleExceedsWallClockBudget(
                            task.attempt_id.clone(),
                        ));
                    }
                    attempts.push(WorkerAttempt {
                        attempt_id: task.attempt_id.clone(),
                        candidate_id: task.candidate_id.clone(),
                        worker_id: worker_index as u32,
                        start_ms: wave_start,
                        end_ms,
                        disposition: task.disposition,
                        resource_cost_units: task.resource_cost_units,
                    });
                }
                wave_start = wave_start
                    .checked_add(wave_duration)
                    .ok_or(SimulationError::CapacityOverflow)?;
            }
        }
        ExecutionRegime::AsynchronousSteadyState => {
            let mut available_at = vec![0u64; worker_count as usize];
            for task in tasks {
                let worker_id = available_at
                    .iter()
                    .enumerate()
                    .min_by_key(|(worker_id, available)| (**available, *worker_id))
                    .map(|(worker_id, _)| worker_id)
                    .expect("worker_count is non-zero");
                let start_ms = available_at[worker_id];
                let end_ms = start_ms
                    .checked_add(task.duration_ms)
                    .ok_or(SimulationError::CapacityOverflow)?;
                if end_ms > wall_clock_budget_ms {
                    return Err(SimulationError::ScheduleExceedsWallClockBudget(
                        task.attempt_id.clone(),
                    ));
                }
                available_at[worker_id] = end_ms;
                attempts.push(WorkerAttempt {
                    attempt_id: task.attempt_id.clone(),
                    candidate_id: task.candidate_id.clone(),
                    worker_id: worker_id as u32,
                    start_ms,
                    end_ms,
                    disposition: task.disposition,
                    resource_cost_units: task.resource_cost_units,
                });
            }
        }
    }

    Ok(WorkerTrace {
        trace_id: format!(
            "simulated-{}-{}workers",
            match regime {
                ExecutionRegime::SynchronousBarrier => "synchronous",
                ExecutionRegime::AsynchronousSteadyState => "asynchronous",
            },
            worker_count
        ),
        regime,
        worker_count,
        wall_clock_budget_ms,
        attempts,
    })
}

/// Simulate and analyze both execution regimes over exactly the same ordered task set.
pub fn compare_worker_regimes(
    plan: &WorkerSimulationPlan,
) -> Result<WorkerRegimeComparisonReport, Box<dyn std::error::Error>> {
    let synchronous_trace = simulate_worker_regime(
        &plan.tasks,
        plan.worker_count,
        plan.wall_clock_budget_ms,
        ExecutionRegime::SynchronousBarrier,
    )?;
    let asynchronous_trace = simulate_worker_regime(
        &plan.tasks,
        plan.worker_count,
        plan.wall_clock_budget_ms,
        ExecutionRegime::AsynchronousSteadyState,
    )?;

    let synchronous = crate::analyze_worker_trace(&synchronous_trace)?;
    let asynchronous = crate::analyze_worker_trace(&asynchronous_trace)?;
    let synchronous_makespan_ms = synchronous.observed_elapsed_ms;
    let asynchronous_makespan_ms = asynchronous.observed_elapsed_ms;
    let async_makespan_speedup = synchronous_makespan_ms as f64
        / asynchronous_makespan_ms as f64;

    Ok(WorkerRegimeComparisonReport {
        task_count: plan.tasks.len(),
        worker_count: plan.worker_count,
        wall_clock_budget_ms: plan.wall_clock_budget_ms,
        synchronous_makespan_ms,
        asynchronous_makespan_ms,
        async_makespan_speedup,
        same_attempt_count: synchronous.total_attempts == asynchronous.total_attempts,
        same_total_busy_ms: synchronous.total_worker_busy_ms == asynchronous.total_worker_busy_ms,
        same_unique_valid_candidates:
            synchronous.unique_valid_candidates == asynchronous.unique_valid_candidates,
        same_resource_cost_units:
            (synchronous.total_resource_cost_units - asynchronous.total_resource_cost_units).abs()
                <= 1e-12,
        synchronous,
        asynchronous,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze_worker_trace;

    fn tasks() -> Vec<SimulationTask> {
        [100, 1000, 200, 200, 200, 200]
            .into_iter()
            .enumerate()
            .map(|(index, duration_ms)| SimulationTask {
                attempt_id: format!("attempt-{:02}", index + 1),
                candidate_id: format!("candidate-{:02}", index + 1),
                duration_ms,
                disposition: AttemptDisposition::Valid,
                resource_cost_units: 1.0,
            })
            .collect()
    }

    #[test]
    fn synchronous_barriers_wait_for_the_slowest_worker_each_wave() {
        let trace = simulate_worker_regime(
            &tasks(),
            2,
            1_500,
            ExecutionRegime::SynchronousBarrier,
        )
        .expect("synchronous simulation fits budget");
        let report = analyze_worker_trace(&trace).expect("simulated trace validates");
        assert_eq!(trace.attempts.iter().map(|attempt| attempt.end_ms).max(), Some(1_400));
        assert_eq!(report.unique_valid_candidates, 6);
    }

    #[test]
    fn asynchronous_steady_state_dispatches_to_the_earliest_free_worker() {
        let trace = simulate_worker_regime(
            &tasks(),
            2,
            1_500,
            ExecutionRegime::AsynchronousSteadyState,
        )
        .expect("asynchronous simulation fits budget");
        let report = analyze_worker_trace(&trace).expect("simulated trace validates");
        assert_eq!(trace.attempts.iter().map(|attempt| attempt.end_ms).max(), Some(1_000));
        assert_eq!(report.unique_valid_candidates, 6);
    }

    #[test]
    fn async_reduces_synthetic_makespan_without_changing_attempts_or_work() {
        let task_set = tasks();
        let sync = simulate_worker_regime(
            &task_set,
            2,
            1_500,
            ExecutionRegime::SynchronousBarrier,
        )
        .expect("sync simulation");
        let async_trace = simulate_worker_regime(
            &task_set,
            2,
            1_500,
            ExecutionRegime::AsynchronousSteadyState,
        )
        .expect("async simulation");
        let sync_report = analyze_worker_trace(&sync).expect("sync trace");
        let async_report = analyze_worker_trace(&async_trace).expect("async trace");

        assert_eq!(sync_report.total_attempts, async_report.total_attempts);
        assert_eq!(sync_report.total_worker_busy_ms, async_report.total_worker_busy_ms);
        assert_eq!(sync_report.total_resource_cost_units, async_report.total_resource_cost_units);
        assert_eq!(sync_report.unique_valid_candidates, async_report.unique_valid_candidates);
        assert!((sync_report.worker_utilization - async_report.worker_utilization).abs() < 1e-12);
        assert!((sync_report.valid_candidates_per_hour - async_report.valid_candidates_per_hour).abs() < 1e-12);
        let sync_makespan = sync.attempts.iter().map(|attempt| attempt.end_ms).max().unwrap();
        let async_makespan = async_trace.attempts.iter().map(|attempt| attempt.end_ms).max().unwrap();
        assert_eq!(sync_makespan, 1_400);
        assert_eq!(async_makespan, 1_000);
        assert!(async_makespan < sync_makespan);
        assert_eq!(sync_report.observed_worker_idle_ms, 900);
        assert_eq!(async_report.observed_worker_idle_ms, 100);
        assert!((sync_report.observed_worker_utilization - (1900.0 / 2800.0)).abs() < 1e-12);
        assert!((async_report.observed_worker_utilization - 0.95).abs() < 1e-12);
        assert!((async_report.valid_candidates_per_observed_hour
            / sync_report.valid_candidates_per_observed_hour - 1.4).abs() < 1e-12);
    }

    #[test]
    fn rejects_schedule_that_exceeds_fixed_wall_clock_budget() {
        assert_eq!(
            simulate_worker_regime(
                &tasks(),
                2,
                999,
                ExecutionRegime::AsynchronousSteadyState
            ).expect_err("final worker completion exceeds budget"),
            SimulationError::ScheduleExceedsWallClockBudget("attempt-02".to_owned())
        );
    }

    #[test]
    fn rejects_duplicate_attempt_ids() {
        let mut input = tasks();
        input[1].attempt_id = input[0].attempt_id.clone();
        assert_eq!(
            simulate_worker_regime(
                &input,
                2,
                1_500,
                ExecutionRegime::AsynchronousSteadyState
            ).expect_err("attempt IDs unique"),
            SimulationError::DuplicateAttemptId("attempt-01".to_owned())
        );
    }
}
