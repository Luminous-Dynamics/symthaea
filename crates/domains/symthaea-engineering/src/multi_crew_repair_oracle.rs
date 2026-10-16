// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exhaustive exact oracle for tiny multi-crew repair scheduling.
//!
//! Scope: at most six jobs and three crews; all jobs' outage clocks and crew
//! availability begin at step zero; jobs have deterministic positive integer
//! durations, positive finite service weights and acyclic precedence constraints;
//! each repair needs one skill; no crew travel, preemption or spare-part constraints.
//! Every dispatch recursively chooses a precedence-ready job and qualified crew,
//! placing it at the earliest time allowed by that crew and its prerequisites.
//!
//! The objective is the sum of service_weight * completion_step. This oracle is
//! intentionally small and explicit: it supplies exact comparison cases for skill
//! assignment and precedence, not a production optimizer for general restoration.

use crate::repair_crew_scheduling::{
    schedule_repairs, RepairCrew, RepairJob, RepairSchedulingConfig, RepairSchedulingError,
    RepairSchedulingPolicy,
};
use std::collections::{HashMap, HashSet};

pub const MAX_MULTI_CREW_ORACLE_JOBS: usize = 6;
pub const MAX_MULTI_CREW_ORACLE_CREWS: usize = 3;
const EPSILON: f64 = 1.0e-9;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactMultiCrewRepair {
    pub job_id: String,
    pub crew_id: String,
    pub start_step: usize,
    pub finish_step: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactMultiCrewRepairResult {
    pub optimal_schedule: Vec<ExactMultiCrewRepair>,
    pub minimum_weighted_completion_loss: f64,
    /// Counts explored complete job/crew dispatch sequences; identical schedules
    /// reached by different serial-generation sequences may be counted more than once.
    pub complete_dispatch_sequences_evaluated: usize,
    pub jobs_evaluated: usize,
    pub crews_evaluated: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MultiCrewRepairOracleError {
    EmptyJobSet,
    EmptyCrewSet,
    TooManyJobs { jobs: usize, maximum: usize },
    TooManyCrews { crews: usize, maximum: usize },
    EmptyIdentifier,
    DuplicateJob(String),
    DuplicateCrew(String),
    InvalidDuration(String),
    InvalidServiceValue(String),
    NonzeroOutageStart(String),
    LateCrewAvailability(String),
    HasSpareParts(String),
    UnknownDependency { job_id: String, dependency_id: String },
    DuplicateDependency { job_id: String, dependency_id: String },
    DependencyCycle(String),
    NoQualifiedCrew(String),
    TimeOverflow(String),
    ObjectiveOverflow,
    SchedulerFailed(String),
    ExactOracleMismatch,
}

impl std::fmt::Display for MultiCrewRepairOracleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyJobSet => write!(f, "at least one repair job is required"),
            Self::EmptyCrewSet => write!(f, "at least one repair crew is required"),
            Self::TooManyJobs { jobs, maximum } => write!(f, "multi-crew oracle supports at most {maximum} jobs, got {jobs}"),
            Self::TooManyCrews { crews, maximum } => write!(f, "multi-crew oracle supports at most {maximum} crews, got {crews}"),
            Self::EmptyIdentifier => write!(f, "job IDs, crew IDs and required skills must not be empty"),
            Self::DuplicateJob(id) => write!(f, "duplicate repair job ID: {id}"),
            Self::DuplicateCrew(id) => write!(f, "duplicate crew ID: {id}"),
            Self::InvalidDuration(id) => write!(f, "repair duration must be positive for {id}"),
            Self::InvalidServiceValue(id) => write!(f, "service weight must be finite and positive for {id}"),
            Self::NonzeroOutageStart(id) => write!(f, "job {id} must have outage_start_step == 0"),
            Self::LateCrewAvailability(id) => write!(f, "crew {id} must have available_from_step == 0"),
            Self::HasSpareParts(id) => write!(f, "job {id} has spare-part constraints outside oracle scope"),
            Self::UnknownDependency { job_id, dependency_id } => write!(f, "job {job_id} depends on unknown job {dependency_id}"),
            Self::DuplicateDependency { job_id, dependency_id } => write!(f, "job {job_id} repeats dependency {dependency_id}"),
            Self::DependencyCycle(id) => write!(f, "precedence cycle includes job {id}"),
            Self::NoQualifiedCrew(id) => write!(f, "no crew has the required skill for job {id}"),
            Self::TimeOverflow(id) => write!(f, "finish time overflows usize for job {id}"),
            Self::ObjectiveOverflow => write!(f, "weighted completion objective is non-finite"),
            Self::SchedulerFailed(message) => write!(f, "heuristic scheduler failed on validated oracle fixture: {message}"),
            Self::ExactOracleMismatch => write!(f, "heuristic objective is below exhaustive oracle optimum"),
        }
    }
}
impl std::error::Error for MultiCrewRepairOracleError {}

fn has_skill(crew: &RepairCrew, job: &RepairJob) -> bool {
    crew.skills.iter().any(|skill| skill == &job.required_skill)
}

fn validate_inputs(
    jobs: &[RepairJob],
    crews: &[RepairCrew],
) -> Result<HashMap<String, usize>, MultiCrewRepairOracleError> {
    if jobs.is_empty() { return Err(MultiCrewRepairOracleError::EmptyJobSet); }
    if crews.is_empty() { return Err(MultiCrewRepairOracleError::EmptyCrewSet); }
    if jobs.len() > MAX_MULTI_CREW_ORACLE_JOBS {
        return Err(MultiCrewRepairOracleError::TooManyJobs {
            jobs: jobs.len(), maximum: MAX_MULTI_CREW_ORACLE_JOBS,
        });
    }
    if crews.len() > MAX_MULTI_CREW_ORACLE_CREWS {
        return Err(MultiCrewRepairOracleError::TooManyCrews {
            crews: crews.len(), maximum: MAX_MULTI_CREW_ORACLE_CREWS,
        });
    }

    let mut job_index = HashMap::with_capacity(jobs.len());
    for (index, job) in jobs.iter().enumerate() {
        if job.id.trim().is_empty() || job.required_skill.trim().is_empty() {
            return Err(MultiCrewRepairOracleError::EmptyIdentifier);
        }
        if job_index.insert(job.id.clone(), index).is_some() {
            return Err(MultiCrewRepairOracleError::DuplicateJob(job.id.clone()));
        }
        if job.duration_steps == 0 {
            return Err(MultiCrewRepairOracleError::InvalidDuration(job.id.clone()));
        }
        if !job.recovered_service_value_per_step.is_finite()
            || job.recovered_service_value_per_step <= 0.0
        {
            return Err(MultiCrewRepairOracleError::InvalidServiceValue(job.id.clone()));
        }
        if job.outage_start_step != 0 {
            return Err(MultiCrewRepairOracleError::NonzeroOutageStart(job.id.clone()));
        }
        if !job.required_parts.is_empty() {
            return Err(MultiCrewRepairOracleError::HasSpareParts(job.id.clone()));
        }
    }

    let mut crew_ids = HashSet::with_capacity(crews.len());
    for crew in crews {
        if crew.id.trim().is_empty() || crew.skills.iter().any(|skill| skill.trim().is_empty()) {
            return Err(MultiCrewRepairOracleError::EmptyIdentifier);
        }
        if !crew_ids.insert(crew.id.as_str()) {
            return Err(MultiCrewRepairOracleError::DuplicateCrew(crew.id.clone()));
        }
        if crew.available_from_step != 0 {
            return Err(MultiCrewRepairOracleError::LateCrewAvailability(crew.id.clone()));
        }
    }

    for job in jobs {
        if !crews.iter().any(|crew| has_skill(crew, job)) {
            return Err(MultiCrewRepairOracleError::NoQualifiedCrew(job.id.clone()));
        }
        let mut deps = HashSet::new();
        for dep in &job.depends_on {
            if !job_index.contains_key(dep) {
                return Err(MultiCrewRepairOracleError::UnknownDependency {
                    job_id: job.id.clone(), dependency_id: dep.clone(),
                });
            }
            if !deps.insert(dep.as_str()) {
                return Err(MultiCrewRepairOracleError::DuplicateDependency {
                    job_id: job.id.clone(), dependency_id: dep.clone(),
                });
            }
        }
    }

    fn visit(
        index: usize,
        jobs: &[RepairJob],
        job_index: &HashMap<String, usize>,
        state: &mut [u8],
    ) -> Result<(), MultiCrewRepairOracleError> {
        match state[index] {
            2 => return Ok(()),
            1 => return Err(MultiCrewRepairOracleError::DependencyCycle(jobs[index].id.clone())),
            _ => {}
        }
        state[index] = 1;
        let mut deps: Vec<usize> = jobs[index].depends_on.iter().map(|dep| job_index[dep]).collect();
        deps.sort_by(|a, b| jobs[*a].id.cmp(&jobs[*b].id));
        for dependency in deps { visit(dependency, jobs, job_index, state)?; }
        state[index] = 2;
        Ok(())
    }
    let mut state = vec![0; jobs.len()];
    for index in 0..jobs.len() { visit(index, jobs, &job_index, &mut state)?; }
    Ok(job_index)
}

struct Search<'a> {
    jobs: &'a [RepairJob],
    crews: &'a [RepairCrew],
    job_index: HashMap<String, usize>,
    scheduled: Vec<bool>,
    finish_by_job: Vec<Option<usize>>,
    crew_free: Vec<usize>,
    assigned_crew: Vec<Option<usize>>,
    start_by_job: Vec<Option<usize>>,
    complete_sequences: usize,
    best_loss: f64,
    best_schedule: Vec<ExactMultiCrewRepair>,
}

fn schedule_signature(schedule: &[ExactMultiCrewRepair]) -> Vec<(usize, usize, &str, &str)> {
    schedule.iter().map(|repair| (
        repair.start_step, repair.finish_step, repair.job_id.as_str(), repair.crew_id.as_str(),
    )).collect()
}

impl Search<'_> {
    fn consider_leaf(&mut self) -> Result<(), MultiCrewRepairOracleError> {
        self.complete_sequences += 1;
        let mut loss = 0.0_f64;
        let mut schedule = Vec::with_capacity(self.jobs.len());
        for (index, job) in self.jobs.iter().enumerate() {
            let finish = self.finish_by_job[index].expect("all jobs scheduled at leaf");
            loss += finish as f64 * job.recovered_service_value_per_step;
            if !loss.is_finite() { return Err(MultiCrewRepairOracleError::ObjectiveOverflow); }
            let crew_index = self.assigned_crew[index].expect("each job has a crew at leaf");
            schedule.push(ExactMultiCrewRepair {
                job_id: job.id.clone(),
                crew_id: self.crews[crew_index].id.clone(),
                start_step: self.start_by_job[index].expect("each job has a start at leaf"),
                finish_step: finish,
            });
        }
        schedule.sort_by(|a, b| (a.start_step, a.finish_step, &a.job_id, &a.crew_id)
            .cmp(&(b.start_step, b.finish_step, &b.job_id, &b.crew_id)));
        let replace = if loss < self.best_loss - EPSILON {
            true
        } else if (loss - self.best_loss).abs() <= EPSILON {
            self.best_schedule.is_empty()
                || schedule_signature(&schedule) < schedule_signature(&self.best_schedule)
        } else {
            false
        };
        if replace {
            self.best_loss = loss;
            self.best_schedule = schedule;
        }
        Ok(())
    }

    fn visit(&mut self, count: usize) -> Result<(), MultiCrewRepairOracleError> {
        if count == self.jobs.len() { return self.consider_leaf(); }

        let mut ready_jobs: Vec<usize> = (0..self.jobs.len())
            .filter(|index| {
                !self.scheduled[*index] && self.jobs[*index].depends_on.iter()
                    .all(|dep| self.finish_by_job[self.job_index[dep]].is_some())
            }).collect();
        ready_jobs.sort_by(|a, b| self.jobs[*a].id.cmp(&self.jobs[*b].id));

        for job_index in ready_jobs {
            let job = &self.jobs[job_index];
            let dependency_ready = job.depends_on.iter()
                .filter_map(|dep| self.finish_by_job[self.job_index[dep]])
                .max().unwrap_or(0);
            let mut crew_candidates: Vec<usize> = (0..self.crews.len())
                .filter(|crew_index| has_skill(&self.crews[*crew_index], job)).collect();
            crew_candidates.sort_by(|a, b| self.crews[*a].id.cmp(&self.crews[*b].id));

            for crew_index in crew_candidates {
                let start = self.crew_free[crew_index].max(dependency_ready);
                let finish = start.checked_add(job.duration_steps)
                    .ok_or_else(|| MultiCrewRepairOracleError::TimeOverflow(job.id.clone()))?;
                let prior_crew_free = self.crew_free[crew_index];
                self.scheduled[job_index] = true;
                self.finish_by_job[job_index] = Some(finish);
                self.assigned_crew[job_index] = Some(crew_index);
                self.start_by_job[job_index] = Some(start);
                self.crew_free[crew_index] = finish;

                self.visit(count + 1)?;

                self.crew_free[crew_index] = prior_crew_free;
                self.start_by_job[job_index] = None;
                self.assigned_crew[job_index] = None;
                self.finish_by_job[job_index] = None;
                self.scheduled[job_index] = false;
            }
        }
        Ok(())
    }
}

/// Exhaustively compare dispatch sequences and skill-qualified crew assignments.
///
/// The serial schedule generator places each selected job at its earliest start
/// consistent with already scheduled prerequisite jobs and the selected crew.
/// All input restrictions are validated before enumeration begins.
pub fn exhaustive_multi_crew_repair_optimum(
    jobs: &[RepairJob],
    crews: &[RepairCrew],
) -> Result<ExactMultiCrewRepairResult, MultiCrewRepairOracleError> {
    let job_index = validate_inputs(jobs, crews)?;
    let mut search = Search {
        jobs,
        crews,
        job_index,
        scheduled: vec![false; jobs.len()],
        finish_by_job: vec![None; jobs.len()],
        crew_free: vec![0; crews.len()],
        assigned_crew: vec![None; jobs.len()],
        start_by_job: vec![None; jobs.len()],
        complete_sequences: 0,
        best_loss: f64::INFINITY,
        best_schedule: Vec::new(),
    };
    search.visit(0)?;
    if search.best_schedule.is_empty() {
        // Validation guarantees a qualified crew per job and an acyclic dependency
        // graph, so this is an invariant check rather than a silent fallback.
        return Err(MultiCrewRepairOracleError::ExactOracleMismatch);
    }
    Ok(ExactMultiCrewRepairResult {
        optimal_schedule: search.best_schedule,
        minimum_weighted_completion_loss: search.best_loss,
        complete_dispatch_sequences_evaluated: search.complete_sequences,
        jobs_evaluated: jobs.len(),
        crews_evaluated: crews.len(),
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiCrewPolicyGap {
    pub policy: RepairSchedulingPolicy,
    pub heuristic_loss: f64,
    pub absolute_gap: f64,
    pub relative_gap: f64,
    pub schedule: Vec<ExactMultiCrewRepair>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiCrewPolicyEvaluation {
    pub exact: ExactMultiCrewRepairResult,
    pub policy_gaps: Vec<MultiCrewPolicyGap>,
}

/// Evaluate all three current list-scheduling policies against the exact multi-crew
/// oracle. This objective comparison requires all outage clocks to start at zero.
pub fn evaluate_multi_crew_policies_against_exact_optimum(
    jobs: &[RepairJob],
    crews: &[RepairCrew],
) -> Result<MultiCrewPolicyEvaluation, MultiCrewRepairOracleError> {
    validate_inputs(jobs, crews)?;
    let exact = exhaustive_multi_crew_repair_optimum(jobs, crews)?;
    let horizon = jobs.iter().try_fold(1usize, |sum, job| {
        sum.checked_add(job.duration_steps).ok_or(MultiCrewRepairOracleError::ObjectiveOverflow)
    })?;
    let policies = [
        RepairSchedulingPolicy::PriorityFirst,
        RepairSchedulingPolicy::ShortestJobFirst,
        RepairSchedulingPolicy::WeightedRecoveryPerStep,
    ];
    let mut policy_gaps = Vec::with_capacity(policies.len());
    for policy in policies {
        let report = schedule_repairs(
            jobs, crews, &[],
            &RepairSchedulingConfig::new(horizon, policy),
        ).map_err(|error: RepairSchedulingError| {
            MultiCrewRepairOracleError::SchedulerFailed(error.to_string())
        })?;
        let raw_gap = report.total_weighted_service_unit_steps_lost
            - exact.minimum_weighted_completion_loss;
        if raw_gap < -EPSILON { return Err(MultiCrewRepairOracleError::ExactOracleMismatch); }
        let gap = raw_gap.max(0.0);
        let schedule = report.schedule.iter().map(|repair| ExactMultiCrewRepair {
            job_id: repair.job_id.clone(), crew_id: repair.crew_id.clone(),
            start_step: repair.start_step, finish_step: repair.finish_step,
        }).collect();
        policy_gaps.push(MultiCrewPolicyGap {
            policy,
            heuristic_loss: report.total_weighted_service_unit_steps_lost,
            absolute_gap: gap,
            relative_gap: gap / exact.minimum_weighted_completion_loss,
            schedule,
        });
    }
    Ok(MultiCrewPolicyEvaluation { exact, policy_gaps })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repair_crew_scheduling::RepairPartRequirement;

    fn job(id: &str, skill: &str, duration: usize, priority: u32, weight: f64, deps: &[&str]) -> RepairJob {
        RepairJob::new(
            id, skill, duration, priority, weight, 0, deps.iter().copied(),
            Vec::<RepairPartRequirement>::new(),
        )
    }
    fn crew(id: &str, skills: &[&str]) -> RepairCrew {
        RepairCrew::new(id, skills.iter().copied(), 0)
    }

    #[test]
    fn exact_oracle_finds_parallel_schedule_with_skill_bottleneck() {
        let jobs = [
            job("a", "x", 4, 0, 1.0, &[]),
            job("b", "y", 3, 1, 5.0, &[]),
            job("c", "x", 2, 2, 10.0, &[]),
        ];
        let crews = [crew("crew-x", &["x"]), crew("crew-y", &["y"])];
        let result = exhaustive_multi_crew_repair_optimum(&jobs, &crews).unwrap();
        assert!((result.minimum_weighted_completion_loss - 41.0).abs() < EPSILON);
        assert_eq!(result.complete_dispatch_sequences_evaluated, 6);
        let c = result.optimal_schedule.iter().find(|repair| repair.job_id == "c").unwrap();
        let a = result.optimal_schedule.iter().find(|repair| repair.job_id == "a").unwrap();
        let b = result.optimal_schedule.iter().find(|repair| repair.job_id == "b").unwrap();
        assert_eq!((c.start_step, c.finish_step), (0, 2));
        assert_eq!((a.start_step, a.finish_step), (2, 6));
        assert_eq!((b.start_step, b.finish_step), (0, 3));
        assert_eq!(c.crew_id, "crew-x");
        assert_eq!(b.crew_id, "crew-y");
    }

    #[test]
    fn exact_oracle_honors_cross_crew_precedence() {
        let jobs = [job("power", "electric", 2, 0, 1.0, &[]),
            job("pump", "mechanical", 3, 1, 10.0, &["power"])];
        let crews = [crew("electrician", &["electric"]), crew("mechanic", &["mechanical"])];
        let result = exhaustive_multi_crew_repair_optimum(&jobs, &crews).unwrap();
        assert!((result.minimum_weighted_completion_loss - 52.0).abs() < EPSILON);
        assert_eq!(result.complete_dispatch_sequences_evaluated, 1);
        let power = result.optimal_schedule.iter().find(|r| r.job_id == "power").unwrap();
        let pump = result.optimal_schedule.iter().find(|r| r.job_id == "pump").unwrap();
        assert!(pump.start_step >= power.finish_step);
        assert_eq!(pump.start_step, 2);
    }

    #[test]
    fn exact_policy_gap_evaluator_reports_all_three_policies() {
        let jobs = [job("a", "x", 4, 0, 1.0, &[]), job("b", "y", 3, 1, 5.0, &[]),
            job("c", "x", 2, 2, 10.0, &[])];
        let crews = [crew("crew-x", &["x"]), crew("crew-y", &["y"])];
        let result = evaluate_multi_crew_policies_against_exact_optimum(&jobs, &crews).unwrap();
        assert_eq!(result.policy_gaps.len(), 3);
        assert!(result.policy_gaps.iter().all(|gap| gap.absolute_gap >= 0.0 && gap.relative_gap >= 0.0));
    }

    #[test]
    fn rejects_dependency_cycles_and_unknown_dependencies() {
        let crews = [crew("crew", &["x"])];
        let cycle = [job("a", "x", 1, 0, 1.0, &["b"]), job("b", "x", 1, 1, 1.0, &["a"])];
        assert!(matches!(exhaustive_multi_crew_repair_optimum(&cycle, &crews),
            Err(MultiCrewRepairOracleError::DependencyCycle(_))));
        let unknown = [job("a", "x", 1, 0, 1.0, &["missing"])];
        assert!(matches!(exhaustive_multi_crew_repair_optimum(&unknown, &crews),
            Err(MultiCrewRepairOracleError::UnknownDependency { .. })));
    }

    #[test]
    fn rejects_parts_late_crew_start_and_unqualified_work() {
        let late_crew = [RepairCrew::new("late", ["x"], 1)];
        let jobs = [job("a", "x", 1, 0, 1.0, &[])];
        assert_eq!(exhaustive_multi_crew_repair_optimum(&jobs, &late_crew).unwrap_err(),
            MultiCrewRepairOracleError::LateCrewAvailability("late".into()));

        let part_job = RepairJob::new("part", "x", 1, 0, 1.0, 0, Vec::<String>::new(),
            [RepairPartRequirement::new("relay", 1.0)]);
        assert_eq!(exhaustive_multi_crew_repair_optimum(&[part_job], &[crew("crew", &["x"])]).unwrap_err(),
            MultiCrewRepairOracleError::HasSpareParts("part".into()));

        let no_qualified = [job("a", "x", 1, 0, 1.0, &[])];
        assert_eq!(exhaustive_multi_crew_repair_optimum(&no_qualified, &[crew("crew", &["y"])]).unwrap_err(),
            MultiCrewRepairOracleError::NoQualifiedCrew("a".into()));
    }

    #[test]
    fn enforces_tiny_search_bounds() {
        let too_many_jobs: Vec<RepairJob> = (0..7).map(|i| job(&format!("j{i}"), "x", 1, i as u32, 1.0, &[])).collect();
        assert!(matches!(exhaustive_multi_crew_repair_optimum(&too_many_jobs, &[crew("c", &["x"])]),
            Err(MultiCrewRepairOracleError::TooManyJobs { jobs: 7, maximum: 6 })));
        let crews = [crew("c1", &["x"]), crew("c2", &["x"]), crew("c3", &["x"]), crew("c4", &["x"])];
        assert!(matches!(exhaustive_multi_crew_repair_optimum(&[job("a", "x", 1, 0, 1.0, &[])], &crews),
            Err(MultiCrewRepairOracleError::TooManyCrews { crews: 4, maximum: 3 })));
    }

    #[test]
    fn flexible_crew_is_reserved_for_unique_skill_bottleneck() {
        let jobs = [
            job("relay", "electrical", 3, 0, 2.0, &[]),
            job("sensor", "electrical", 2, 1, 7.0, &[]),
            job("valve", "plumbing", 2, 2, 9.0, &[]),
        ];
        let crews = [
            crew("electrician", &["electrical"]),
            crew("versatile", &["electrical", "plumbing"]),
        ];
        let result = exhaustive_multi_crew_repair_optimum(&jobs, &crews).unwrap();
        assert!((result.minimum_weighted_completion_loss - 42.0).abs() < EPSILON);
        assert_eq!(result.complete_dispatch_sequences_evaluated, 24);
        let sensor = result.optimal_schedule.iter().find(|r| r.job_id == "sensor").unwrap();
        let valve = result.optimal_schedule.iter().find(|r| r.job_id == "valve").unwrap();
        let relay = result.optimal_schedule.iter().find(|r| r.job_id == "relay").unwrap();
        assert_eq!((sensor.start_step, sensor.finish_step, sensor.crew_id.as_str()), (0, 2, "electrician"));
        assert_eq!((valve.start_step, valve.finish_step, valve.crew_id.as_str()), (0, 2, "versatile"));
        assert_eq!((relay.start_step, relay.finish_step, relay.crew_id.as_str()), (2, 5, "electrician"));
    }

    #[test]
    fn identical_fixture_and_reordered_inputs_return_same_canonical_optimum() {
        let jobs = [job("a", "x", 4, 0, 1.0, &[]), job("b", "y", 3, 1, 5.0, &[]), job("c", "x", 2, 2, 10.0, &[])];
        let crews = [crew("crew-x", &["x"]), crew("crew-y", &["y"])];
        let reversed_jobs = [jobs[2].clone(), jobs[1].clone(), jobs[0].clone()];
        let reversed_crews = [crews[1].clone(), crews[0].clone()];
        let first = exhaustive_multi_crew_repair_optimum(&jobs, &crews).unwrap();
        let second = exhaustive_multi_crew_repair_optimum(&reversed_jobs, &reversed_crews).unwrap();
        assert_eq!(first.optimal_schedule, second.optimal_schedule);
        assert_eq!(first.minimum_weighted_completion_loss, second.minimum_weighted_completion_loss);
    }
}
