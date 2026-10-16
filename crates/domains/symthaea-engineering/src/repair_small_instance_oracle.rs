// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exhaustive exact oracle for tiny, single-crew repair-scheduling instances.
//!
//! This is intentionally a narrow oracle for heuristic measurement, not a general
//! repair optimizer. It supports up to eight jobs, all released at step zero, with
//! no precedence edges or spare-part requirements. The modeled objective is
//! weighted completion time: sum(duration-weight * completion step). This equals
//! weighted lost service through a horizon that exceeds every completion, provided
//! all jobs start disrupted at step zero.
//!
//! Enumerating all permutations is exponential (n!) and must stay restricted to
//! small fixtures. The oracle is independent of the production list-scheduler code.

use crate::repair_crew_scheduling::RepairJob;
use std::collections::HashSet;

pub const MAX_EXACT_ORACLE_JOBS: usize = 8;
const EPSILON: f64 = 1.0e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct ExactRepairOracleResult {
    pub optimal_order: Vec<String>,
    pub minimum_weighted_completion_loss: f64,
    pub feasible_permutations: usize,
    pub jobs_evaluated: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExactRepairOracleError {
    EmptyJobSet,
    TooManyJobs { jobs: usize, maximum: usize },
    EmptyIdentifier,
    DuplicateJob(String),
    InvalidDuration(String),
    InvalidServiceValue(String),
    NonzeroOutageStart(String),
    HasDependencies(String),
    HasSpareParts(String),
    ObjectiveOverflow,
}

impl std::fmt::Display for ExactRepairOracleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyJobSet => write!(f, "at least one job is required"),
            Self::TooManyJobs { jobs, maximum } => write!(f, "exact oracle supports at most {maximum} jobs, got {jobs}"),
            Self::EmptyIdentifier => write!(f, "job identifiers and required skills must not be empty"),
            Self::DuplicateJob(id) => write!(f, "duplicate job id: {id}"),
            Self::InvalidDuration(id) => write!(f, "duration must be positive for job {id}"),
            Self::InvalidServiceValue(id) => write!(f, "service value must be finite and positive for job {id}"),
            Self::NonzeroOutageStart(id) => write!(f, "job {id} must have outage_start_step == 0"),
            Self::HasDependencies(id) => write!(f, "job {id} has precedence dependencies outside oracle scope"),
            Self::HasSpareParts(id) => write!(f, "job {id} has spare-part requirements outside oracle scope"),
            Self::ObjectiveOverflow => write!(f, "weighted completion objective is non-finite or time arithmetic overflowed"),
        }
    }
}
impl std::error::Error for ExactRepairOracleError {}

fn validate(jobs: &[RepairJob]) -> Result<(), ExactRepairOracleError> {
    if jobs.is_empty() {
        return Err(ExactRepairOracleError::EmptyJobSet);
    }
    if jobs.len() > MAX_EXACT_ORACLE_JOBS {
        return Err(ExactRepairOracleError::TooManyJobs {
            jobs: jobs.len(),
            maximum: MAX_EXACT_ORACLE_JOBS,
        });
    }
    let mut ids = HashSet::with_capacity(jobs.len());
    for job in jobs {
        if job.id.trim().is_empty() || job.required_skill.trim().is_empty() {
            return Err(ExactRepairOracleError::EmptyIdentifier);
        }
        if !ids.insert(job.id.as_str()) {
            return Err(ExactRepairOracleError::DuplicateJob(job.id.clone()));
        }
        if job.duration_steps == 0 {
            return Err(ExactRepairOracleError::InvalidDuration(job.id.clone()));
        }
        if !job.recovered_service_value_per_step.is_finite()
            || job.recovered_service_value_per_step <= 0.0
        {
            return Err(ExactRepairOracleError::InvalidServiceValue(job.id.clone()));
        }
        if job.outage_start_step != 0 {
            return Err(ExactRepairOracleError::NonzeroOutageStart(job.id.clone()));
        }
        if !job.depends_on.is_empty() {
            return Err(ExactRepairOracleError::HasDependencies(job.id.clone()));
        }
        if !job.required_parts.is_empty() {
            return Err(ExactRepairOracleError::HasSpareParts(job.id.clone()));
        }
    }
    Ok(())
}

fn lexicographically_precedes(candidate: &[usize], incumbent: &[usize], jobs: &[RepairJob]) -> bool {
    candidate.iter().map(|i| jobs[*i].id.as_str())
        .cmp(incumbent.iter().map(|i| jobs[*i].id.as_str()))
        .is_lt()
}

fn evaluate_order(order: &[usize], jobs: &[RepairJob]) -> Result<f64, ExactRepairOracleError> {
    let mut clock = 0usize;
    let mut loss = 0.0_f64;
    for index in order {
        clock = clock.checked_add(jobs[*index].duration_steps)
            .ok_or(ExactRepairOracleError::ObjectiveOverflow)?;
        loss += clock as f64 * jobs[*index].recovered_service_value_per_step;
        if !loss.is_finite() {
            return Err(ExactRepairOracleError::ObjectiveOverflow);
        }
    }
    Ok(loss)
}

fn enumerate(
    jobs: &[RepairJob],
    used: &mut [bool],
    current: &mut Vec<usize>,
    best_order: &mut Option<Vec<usize>>,
    best_loss: &mut f64,
    count: &mut usize,
) -> Result<(), ExactRepairOracleError> {
    if current.len() == jobs.len() {
        let loss = evaluate_order(current, jobs)?;
        *count += 1;
        let replace = match best_order.as_ref() {
            None => true,
            Some(best) if loss < *best_loss - EPSILON => true,
            Some(best) if (loss - *best_loss).abs() <= EPSILON => {
                lexicographically_precedes(current, best, jobs)
            }
            _ => false,
        };
        if replace {
            *best_loss = loss;
            *best_order = Some(current.clone());
        }
        return Ok(());
    }

    // Sort candidate indices by stable ID so traversal itself is reproducible.
    let mut candidates: Vec<usize> = (0..jobs.len()).filter(|i| !used[*i]).collect();
    candidates.sort_by(|a, b| jobs[*a].id.cmp(&jobs[*b].id));
    for index in candidates {
        used[index] = true;
        current.push(index);
        enumerate(jobs, used, current, best_order, best_loss, count)?;
        current.pop();
        used[index] = false;
    }
    Ok(())
}

/// Compute the exact optimum by enumerating all job permutations.
///
/// Every repair is assumed to be service-restoring on completion, and all jobs
/// have the same release time (zero). A compatible caller must compare this result
/// against a scheduler with one crew qualified for every job and a horizon large
/// enough to include all completions.
pub fn exhaustive_single_crew_repair_optimum(
    jobs: &[RepairJob],
) -> Result<ExactRepairOracleResult, ExactRepairOracleError> {
    validate(jobs)?;
    let mut used = vec![false; jobs.len()];
    let mut current = Vec::with_capacity(jobs.len());
    let mut best_order = None;
    let mut best_loss = f64::INFINITY;
    let mut count = 0usize;
    enumerate(
        jobs, &mut used, &mut current, &mut best_order, &mut best_loss, &mut count,
    )?;
    let order = best_order.expect("non-empty validated jobs have at least one permutation");
    Ok(ExactRepairOracleResult {
        optimal_order: order.iter().map(|i| jobs[*i].id.clone()).collect(),
        minimum_weighted_completion_loss: best_loss,
        feasible_permutations: count,
        jobs_evaluated: jobs.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repair_crew_scheduling::{
        schedule_repairs, RepairCrew, RepairPartRequirement, RepairSchedulingConfig,
        RepairSchedulingPolicy,
    };

    fn job(id: &str, duration: usize, priority: u32, value: f64) -> RepairJob {
        RepairJob::new(
            id, "general", duration, priority, value, 0,
            Vec::<String>::new(), Vec::<RepairPartRequirement>::new(),
        )
    }

    fn schedule_loss(jobs: &[RepairJob], policy: RepairSchedulingPolicy) -> f64 {
        let crews = [RepairCrew::new("omnicompetent", ["general"], 0)];
        let horizon = jobs.iter().map(|job| job.duration_steps).sum::<usize>() + 1;
        schedule_repairs(jobs, &crews, &[], &RepairSchedulingConfig::new(horizon, policy))
            .unwrap().total_weighted_service_unit_steps_lost
    }

    #[test]
    fn exact_oracle_matches_known_two_job_optimum() {
        let jobs = [job("slow_low", 10, 0, 1.0), job("fast_high", 1, 1, 100.0)];
        let result = exhaustive_single_crew_repair_optimum(&jobs).unwrap();
        assert_eq!(result.optimal_order, vec!["fast_high", "slow_low"]);
        assert!((result.minimum_weighted_completion_loss - 111.0).abs() < EPSILON);
        assert_eq!(result.feasible_permutations, 2);
        assert!((schedule_loss(&jobs, RepairSchedulingPolicy::WeightedRecoveryPerStep)
            - result.minimum_weighted_completion_loss).abs() < EPSILON);
    }

    #[test]
    fn exact_oracle_finds_priority_first_counterexample() {
        let jobs = [job("slow_low", 10, 0, 1.0), job("fast_high", 1, 1, 100.0)];
        let exact = exhaustive_single_crew_repair_optimum(&jobs).unwrap();
        assert!((schedule_loss(&jobs, RepairSchedulingPolicy::PriorityFirst) - 1110.0).abs() < EPSILON);
        assert!((schedule_loss(&jobs, RepairSchedulingPolicy::PriorityFirst)
            / exact.minimum_weighted_completion_loss - 1.0 - 9.0).abs() < EPSILON);
    }

    #[test]
    fn exact_oracle_enumerates_the_full_eight_job_space() {
        let jobs = [
            job("a", 8, 0, 1.0), job("b", 1, 1, 40.0), job("c", 3, 2, 8.0),
            job("d", 2, 3, 13.0), job("e", 5, 4, 3.0), job("f", 1, 5, 9.0),
            job("g", 4, 6, 6.0), job("h", 2, 7, 11.0),
        ];
        let result = exhaustive_single_crew_repair_optimum(&jobs).unwrap();
        assert_eq!(result.feasible_permutations, 40_320);
        assert_eq!(result.jobs_evaluated, 8);
        assert!(result.minimum_weighted_completion_loss.is_finite());
    }

    #[test]
    fn equal_objectives_use_stable_identifier_tiebreak() {
        let jobs = [job("z", 2, 0, 1.0), job("a", 4, 1, 2.0)];
        let result = exhaustive_single_crew_repair_optimum(&jobs).unwrap();
        assert_eq!(result.optimal_order, vec!["a", "z"]);
        assert!((result.minimum_weighted_completion_loss - 14.0).abs() < EPSILON);
    }

    #[test]
    fn matches_three_job_reference_fixture() {
        let jobs = [job("A", 3, 2, 1.0), job("B", 1, 0, 10.0), job("C", 2, 1, 4.0)];
        let result = exhaustive_single_crew_repair_optimum(&jobs).unwrap();
        assert_eq!(result.optimal_order, vec!["B", "C", "A"]);
        assert!((result.minimum_weighted_completion_loss - 28.0).abs() < EPSILON);
        assert_eq!(result.feasible_permutations, 6);
    }

    #[test]
    fn matches_four_job_reference_fixture() {
        let jobs = [job("A", 2, 3, 8.0), job("B", 3, 2, 9.0),
            job("C", 1, 1, 2.0), job("D", 4, 0, 4.0)];
        let result = exhaustive_single_crew_repair_optimum(&jobs).unwrap();
        assert_eq!(result.optimal_order, vec!["A", "B", "C", "D"]);
        assert!((result.minimum_weighted_completion_loss - 113.0).abs() < EPSILON);
        assert_eq!(result.feasible_permutations, 24);
    }

    #[test]
    fn one_job_fixture_has_one_feasible_permutation() {
        let jobs = [job("only", 3, 0, 5.0)];
        let result = exhaustive_single_crew_repair_optimum(&jobs).unwrap();
        assert_eq!(result.optimal_order, vec!["only"]);
        assert!((result.minimum_weighted_completion_loss - 15.0).abs() < EPSILON);
        assert_eq!(result.feasible_permutations, 1);
    }

    #[test]
    fn rejects_scope_expansion_instead_of_silently_ignoring_constraints() {
        let jobs = [RepairJob::new(
            "dependency", "general", 1, 0, 1.0, 0, ["other"],
            Vec::<RepairPartRequirement>::new(),
        )];
        assert_eq!(exhaustive_single_crew_repair_optimum(&jobs).unwrap_err(),
            ExactRepairOracleError::HasDependencies("dependency".into()));
    }

    #[test]
    fn rejects_more_than_eight_jobs() {
        let jobs: Vec<RepairJob> = (0..9).map(|i| job(&format!("j{i}"), 1, i as u32, 1.0)).collect();
        assert!(matches!(
            exhaustive_single_crew_repair_optimum(&jobs),
            Err(ExactRepairOracleError::TooManyJobs { jobs: 9, maximum: 8 })
        ));
    }
}
