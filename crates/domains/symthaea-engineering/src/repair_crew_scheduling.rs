// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic repair-crew scheduling with skill constraints, precedence, and finite spares.
//!
//! This is a transparent list-scheduling heuristic for comparing policies, not an
//! optimal solver. All jobs are assumed disrupted at their declared outage step;
//! repair durations are deterministic; crews do not travel; parts are reserved
//! when a plan is constructed; and service value is a caller-supplied weight.
//! Compare policies on the same scenario and inspect every job schedule. Do not
//! treat the weighted objective as an ethically complete value function.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Debug, PartialEq)]
pub struct RepairPartRequirement {
    pub part_id: String,
    pub quantity: f64,
}

impl RepairPartRequirement {
    pub fn new(part_id: impl Into<String>, quantity: f64) -> Self {
        Self { part_id: part_id.into(), quantity }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepairPartStock {
    pub id: String,
    pub quantity: f64,
}

impl RepairPartStock {
    pub fn new(id: impl Into<String>, quantity: f64) -> Self {
        Self { id: id.into(), quantity }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepairJob {
    pub id: String,
    /// A job can only be assigned to a crew carrying this skill label.
    pub required_skill: String,
    pub duration_steps: usize,
    /// Lower numeric values are scheduled first by PriorityFirst.
    pub priority: u32,
    /// Weighted essential-service units restored per step once repair completes.
    /// This must be a positive finite caller-defined weight.
    pub recovered_service_value_per_step: f64,
    /// Step at which the associated service first became degraded.
    pub outage_start_step: usize,
    /// Jobs that must finish before this job may begin.
    pub depends_on: Vec<String>,
    /// Finite spare-part quantities reserved by this repair.
    pub required_parts: Vec<RepairPartRequirement>,
}

impl RepairJob {
    pub fn new(
        id: impl Into<String>,
        required_skill: impl Into<String>,
        duration_steps: usize,
        priority: u32,
        recovered_service_value_per_step: f64,
        outage_start_step: usize,
        depends_on: impl IntoIterator<Item = impl Into<String>>,
        required_parts: impl IntoIterator<Item = RepairPartRequirement>,
    ) -> Self {
        Self {
            id: id.into(),
            required_skill: required_skill.into(),
            duration_steps,
            priority,
            recovered_service_value_per_step,
            outage_start_step,
            depends_on: depends_on.into_iter().map(Into::into).collect(),
            required_parts: required_parts.into_iter().collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepairCrew {
    pub id: String,
    pub skills: Vec<String>,
    pub available_from_step: usize,
}

impl RepairCrew {
    pub fn new(
        id: impl Into<String>,
        skills: impl IntoIterator<Item = impl Into<String>>,
        available_from_step: usize,
    ) -> Self {
        Self {
            id: id.into(),
            skills: skills.into_iter().map(Into::into).collect(),
            available_from_step,
        }
    }

    fn can_repair(&self, job: &RepairJob) -> bool {
        self.skills.iter().any(|skill| skill == &job.required_skill)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepairSchedulingPolicy {
    /// Lower numeric priority first; duration and stable identifiers break ties.
    PriorityFirst,
    /// Shortest duration first; priority and stable identifiers break ties.
    ShortestJobFirst,
    /// Highest recovered-service-value / repair-duration first.
    /// This is a greedy heuristic, not a global optimum or ethical objective.
    WeightedRecoveryPerStep,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepairSchedulingConfig {
    pub horizon_steps: usize,
    pub policy: RepairSchedulingPolicy,
}

impl RepairSchedulingConfig {
    pub fn new(horizon_steps: usize, policy: RepairSchedulingPolicy) -> Self {
        Self { horizon_steps, policy }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScheduledRepair {
    pub job_id: String,
    pub crew_id: String,
    pub start_step: usize,
    pub finish_step: usize,
    pub outage_start_step: usize,
    pub recovered_service_value_per_step: f64,
    pub weighted_service_unit_steps_lost_within_horizon: f64,
    pub parts_reserved: Vec<RepairPartRequirement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RemainingRepairPartStock {
    pub part_id: String,
    pub initial_quantity: f64,
    pub reserved_quantity: f64,
    pub remaining_quantity: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepairSchedulingReport {
    pub horizon_steps: usize,
    pub policy: RepairSchedulingPolicy,
    pub total_weighted_service_unit_steps_lost: f64,
    pub jobs_completed_within_horizon: usize,
    pub jobs_finishing_after_horizon: usize,
    pub schedule: Vec<ScheduledRepair>,
    pub remaining_parts: Vec<RemainingRepairPartStock>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepairSchedulingError {
    EmptyJobSet,
    EmptyCrewSet,
    ZeroHorizon,
    EmptyIdentifier,
    DuplicateJob(String),
    DuplicateCrew(String),
    DuplicatePart(String),
    InvalidJobDuration(String),
    InvalidRecoveredServiceValue(String),
    OutageStartsOutsideHorizon(String),
    UnknownJobDependency { job_id: String, dependency_id: String },
    DuplicateJobDependency { job_id: String, dependency_id: String },
    DependencyCycle(String),
    InvalidPartQuantity { job_id: String, part_id: String },
    UnknownPartRequirement { job_id: String, part_id: String },
    DuplicatePartRequirement { job_id: String, part_id: String },
    InvalidPartStock(String),
    NoQualifiedCrew(String),
    InsufficientPartStock { job_id: String, part_id: String },
    NoSchedulableJob,
}

impl std::fmt::Display for RepairSchedulingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyJobSet => write!(f, "at least one repair job is required"),
            Self::EmptyCrewSet => write!(f, "at least one repair crew is required"),
            Self::ZeroHorizon => write!(f, "horizon_steps must be greater than zero"),
            Self::EmptyIdentifier => write!(f, "identifiers and required skills must not be empty"),
            Self::DuplicateJob(id) => write!(f, "duplicate job id: {id}"),
            Self::DuplicateCrew(id) => write!(f, "duplicate crew id: {id}"),
            Self::DuplicatePart(id) => write!(f, "duplicate part id: {id}"),
            Self::InvalidJobDuration(id) => write!(f, "repair duration must be positive for {id}"),
            Self::InvalidRecoveredServiceValue(id) => {
                write!(f, "recovered service value must be finite and positive for {id}")
            }
            Self::OutageStartsOutsideHorizon(id) => {
                write!(f, "outage for {id} starts outside the horizon")
            }
            Self::UnknownJobDependency { job_id, dependency_id } => {
                write!(f, "job {job_id} depends on unknown job {dependency_id}")
            }
            Self::DuplicateJobDependency { job_id, dependency_id } => {
                write!(f, "job {job_id} repeats dependency {dependency_id}")
            }
            Self::DependencyCycle(id) => write!(f, "repair dependency cycle includes job {id}"),
            Self::InvalidPartQuantity { job_id, part_id } => {
                write!(f, "part quantity must be finite and positive for {job_id}/{part_id}")
            }
            Self::UnknownPartRequirement { job_id, part_id } => {
                write!(f, "job {job_id} requires unknown part {part_id}")
            }
            Self::DuplicatePartRequirement { job_id, part_id } => {
                write!(f, "job {job_id} repeats part requirement {part_id}")
            }
            Self::InvalidPartStock(id) => write!(f, "part stock must be finite and non-negative for {id}"),
            Self::NoQualifiedCrew(id) => write!(f, "no crew has the required skill for job {id}"),
            Self::InsufficientPartStock { job_id, part_id } => {
                write!(f, "insufficient stock of {part_id} for repair job {job_id}")
            }
            Self::NoSchedulableJob => write!(f, "no precedence-ready, crew-eligible, parts-feasible job can be scheduled"),
        }
    }
}

impl std::error::Error for RepairSchedulingError {}

#[derive(Clone, Debug)]
struct Candidate {
    job_index: usize,
    crew_index: usize,
    start_step: usize,
    finish_step: usize,
}

fn policy_order(
    a: &Candidate,
    b: &Candidate,
    jobs: &[RepairJob],
    crews: &[RepairCrew],
    policy: RepairSchedulingPolicy,
) -> Ordering {
    // Earlier feasible starts keep eligible crews from sitting idle. Policy only
    // chooses among candidate assignments at the same earliest start time.
    a.start_step
        .cmp(&b.start_step)
        .then_with(|| {
            let ja = &jobs[a.job_index];
            let jb = &jobs[b.job_index];
            match policy {
                RepairSchedulingPolicy::PriorityFirst => ja
                    .priority
                    .cmp(&jb.priority)
                    .then_with(|| ja.duration_steps.cmp(&jb.duration_steps)),
                RepairSchedulingPolicy::ShortestJobFirst => ja
                    .duration_steps
                    .cmp(&jb.duration_steps)
                    .then_with(|| ja.priority.cmp(&jb.priority)),
                RepairSchedulingPolicy::WeightedRecoveryPerStep => {
                    let score_a =
                        ja.recovered_service_value_per_step / ja.duration_steps as f64;
                    let score_b =
                        jb.recovered_service_value_per_step / jb.duration_steps as f64;
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| ja.priority.cmp(&jb.priority))
                }
            }
        })
        .then_with(|| jobs[a.job_index].id.cmp(&jobs[b.job_index].id))
        .then_with(|| {
            // Prefer a less-flexible crew when multiple crews can perform the same
            // job, preserving versatile crews for work with fewer alternatives.
            let flexibility_a = jobs
                .iter()
                .filter(|job| crews[a.crew_index].can_repair(job))
                .count();
            let flexibility_b = jobs
                .iter()
                .filter(|job| crews[b.crew_index].can_repair(job))
                .count();
            flexibility_a.cmp(&flexibility_b)
        })
        .then_with(|| crews[a.crew_index].id.cmp(&crews[b.crew_index].id))
}

fn validate_repair_inputs(
    jobs: &[RepairJob],
    crews: &[RepairCrew],
    parts: &[RepairPartStock],
    config: &RepairSchedulingConfig,
) -> Result<(HashMap<String, usize>, HashMap<String, usize>, HashMap<String, usize>), RepairSchedulingError> {
    if config.horizon_steps == 0 {
        return Err(RepairSchedulingError::ZeroHorizon);
    }
    if jobs.is_empty() {
        return Err(RepairSchedulingError::EmptyJobSet);
    }
    if crews.is_empty() {
        return Err(RepairSchedulingError::EmptyCrewSet);
    }

    let mut job_index = HashMap::with_capacity(jobs.len());
    for (i, job) in jobs.iter().enumerate() {
        if job.id.trim().is_empty() || job.required_skill.trim().is_empty() {
            return Err(RepairSchedulingError::EmptyIdentifier);
        }
        if job_index.insert(job.id.clone(), i).is_some() {
            return Err(RepairSchedulingError::DuplicateJob(job.id.clone()));
        }
        if job.duration_steps == 0 {
            return Err(RepairSchedulingError::InvalidJobDuration(job.id.clone()));
        }
        if !job.recovered_service_value_per_step.is_finite()
            || job.recovered_service_value_per_step <= 0.0
        {
            return Err(RepairSchedulingError::InvalidRecoveredServiceValue(job.id.clone()));
        }
        if job.outage_start_step >= config.horizon_steps {
            return Err(RepairSchedulingError::OutageStartsOutsideHorizon(job.id.clone()));
        }
    }

    let mut crew_index = HashMap::with_capacity(crews.len());
    for (i, crew) in crews.iter().enumerate() {
        if crew.id.trim().is_empty() || crew.skills.iter().any(|s| s.trim().is_empty()) {
            return Err(RepairSchedulingError::EmptyIdentifier);
        }
        if crew_index.insert(crew.id.clone(), i).is_some() {
            return Err(RepairSchedulingError::DuplicateCrew(crew.id.clone()));
        }
    }

    let mut part_index = HashMap::with_capacity(parts.len());
    for (i, part) in parts.iter().enumerate() {
        if part.id.trim().is_empty() {
            return Err(RepairSchedulingError::EmptyIdentifier);
        }
        if part_index.insert(part.id.clone(), i).is_some() {
            return Err(RepairSchedulingError::DuplicatePart(part.id.clone()));
        }
        if !part.quantity.is_finite() || part.quantity < 0.0 {
            return Err(RepairSchedulingError::InvalidPartStock(part.id.clone()));
        }
    }

    for job in jobs {
        if !crews.iter().any(|crew| crew.can_repair(job)) {
            return Err(RepairSchedulingError::NoQualifiedCrew(job.id.clone()));
        }
        let mut dependencies = HashSet::new();
        for dependency_id in &job.depends_on {
            if !job_index.contains_key(dependency_id) {
                return Err(RepairSchedulingError::UnknownJobDependency {
                    job_id: job.id.clone(),
                    dependency_id: dependency_id.clone(),
                });
            }
            if !dependencies.insert(dependency_id) {
                return Err(RepairSchedulingError::DuplicateJobDependency {
                    job_id: job.id.clone(),
                    dependency_id: dependency_id.clone(),
                });
            }
        }
        let mut required_part_ids = HashSet::new();
        for requirement in &job.required_parts {
            if !part_index.contains_key(&requirement.part_id) {
                return Err(RepairSchedulingError::UnknownPartRequirement {
                    job_id: job.id.clone(),
                    part_id: requirement.part_id.clone(),
                });
            }
            if !required_part_ids.insert(&requirement.part_id) {
                return Err(RepairSchedulingError::DuplicatePartRequirement {
                    job_id: job.id.clone(),
                    part_id: requirement.part_id.clone(),
                });
            }
            if !requirement.quantity.is_finite() || requirement.quantity <= 0.0 {
                return Err(RepairSchedulingError::InvalidPartQuantity {
                    job_id: job.id.clone(),
                    part_id: requirement.part_id.clone(),
                });
            }
        }
    }

    fn visit(
        index: usize,
        jobs: &[RepairJob],
        job_index: &HashMap<String, usize>,
        state: &mut [u8],
    ) -> Result<(), RepairSchedulingError> {
        match state[index] {
            2 => return Ok(()),
            1 => return Err(RepairSchedulingError::DependencyCycle(jobs[index].id.clone())),
            _ => {}
        }
        state[index] = 1;
        let mut deps: Vec<usize> = jobs[index].depends_on.iter().map(|id| job_index[id]).collect();
        deps.sort_by(|a, b| jobs[*a].id.cmp(&jobs[*b].id));
        for dep in deps {
            visit(dep, jobs, job_index, state)?;
        }
        state[index] = 2;
        Ok(())
    }

    let mut state = vec![0; jobs.len()];
    for i in 0..jobs.len() {
        visit(i, jobs, &job_index, &mut state)?;
    }

    Ok((job_index, crew_index, part_index))
}

/// Construct a feasible deterministic repair schedule using a list-scheduling heuristic.
///
/// Each repair is assumed to be needed from outage_start_step until finish_step.
/// Precedence dependencies must finish before a repair may start. Crews must have
/// the required skill and are single-job-at-a-time. Parts are reserved when the
/// schedule is planned, in policy order, to prevent two jobs consuming the same stock.
/// This function fails closed when no feasible schedule can be constructed.
pub fn schedule_repairs(
    jobs: &[RepairJob],
    crews: &[RepairCrew],
    parts: &[RepairPartStock],
    config: &RepairSchedulingConfig,
) -> Result<RepairSchedulingReport, RepairSchedulingError> {
    let (job_index, _crew_index, part_index) =
        validate_repair_inputs(jobs, crews, parts, config)?;

    let mut remaining_parts: Vec<f64> = parts.iter().map(|part| part.quantity).collect();
    let mut reserved_parts = vec![0.0_f64; parts.len()];
    let mut crew_available: Vec<usize> =
        crews.iter().map(|crew| crew.available_from_step).collect();
    let mut finish_steps: Vec<Option<usize>> = vec![None; jobs.len()];
    let mut schedule = Vec::with_capacity(jobs.len());

    while schedule.len() < jobs.len() {
        let mut candidates = Vec::new();

        for (job_i, job) in jobs.iter().enumerate() {
            if finish_steps[job_i].is_some() {
                continue;
            }
            let dependencies_scheduled = job.depends_on.iter().all(|dep| {
                finish_steps[job_index[dep]].is_some()
            });
            if !dependencies_scheduled {
                continue;
            }

            let dependency_ready = job
                .depends_on
                .iter()
                .filter_map(|dep| finish_steps[job_index[dep]])
                .max()
                .unwrap_or(0);
            let parts_available = job.required_parts.iter().all(|req| {
                remaining_parts[part_index[&req.part_id]] + EPSILON >= req.quantity
            });
            if !parts_available {
                continue;
            }

            for (crew_i, crew) in crews.iter().enumerate() {
                if !crew.can_repair(job) {
                    continue;
                }
                let start = crew_available[crew_i]
                    .max(dependency_ready)
                    .max(job.outage_start_step);
                let finish = start.saturating_add(job.duration_steps);
                candidates.push(Candidate {
                    job_index: job_i,
                    crew_index: crew_i,
                    start_step: start,
                    finish_step: finish,
                });
            }
        }

        if candidates.is_empty() {
            // Distinguish exhausted/overcommitted stock from an impossible plan.
            for (job_i, job) in jobs.iter().enumerate() {
                if finish_steps[job_i].is_some()
                    || !job.depends_on.iter().all(|dep| finish_steps[job_index[dep]].is_some())
                {
                    continue;
                }
                for requirement in &job.required_parts {
                    let available = remaining_parts[part_index[&requirement.part_id]];
                    if available + EPSILON < requirement.quantity {
                        return Err(RepairSchedulingError::InsufficientPartStock {
                            job_id: job.id.clone(),
                            part_id: requirement.part_id.clone(),
                        });
                    }
                }
            }
            return Err(RepairSchedulingError::NoSchedulableJob);
        }

        candidates.sort_by(|a, b| policy_order(a, b, jobs, crews, config.policy));
        let chosen = candidates.remove(0);
        let job = &jobs[chosen.job_index];
        let crew = &crews[chosen.crew_index];

        for requirement in &job.required_parts {
            let pi = part_index[&requirement.part_id];
            remaining_parts[pi] = (remaining_parts[pi] - requirement.quantity).max(0.0);
            reserved_parts[pi] += requirement.quantity;
        }

        finish_steps[chosen.job_index] = Some(chosen.finish_step);
        crew_available[chosen.crew_index] = chosen.finish_step;

        let lost_steps = chosen
            .finish_step
            .min(config.horizon_steps)
            .saturating_sub(job.outage_start_step.min(config.horizon_steps));
        schedule.push(ScheduledRepair {
            job_id: job.id.clone(),
            crew_id: crew.id.clone(),
            start_step: chosen.start_step,
            finish_step: chosen.finish_step,
            outage_start_step: job.outage_start_step,
            recovered_service_value_per_step: job.recovered_service_value_per_step,
            weighted_service_unit_steps_lost_within_horizon: lost_steps as f64
                * job.recovered_service_value_per_step,
            parts_reserved: job.required_parts.clone(),
        });
    }

    schedule.sort_by(|a, b| {
        a.start_step
            .cmp(&b.start_step)
            .then_with(|| a.finish_step.cmp(&b.finish_step))
            .then_with(|| a.job_id.cmp(&b.job_id))
            .then_with(|| a.crew_id.cmp(&b.crew_id))
    });

    let total_weighted_service_unit_steps_lost = schedule
        .iter()
        .map(|item| item.weighted_service_unit_steps_lost_within_horizon)
        .sum();
    let jobs_completed_within_horizon = schedule
        .iter()
        .filter(|item| item.finish_step <= config.horizon_steps)
        .count();
    let jobs_finishing_after_horizon = schedule.len() - jobs_completed_within_horizon;
    let remaining_parts = parts
        .iter()
        .enumerate()
        .map(|(i, part)| RemainingRepairPartStock {
            part_id: part.id.clone(),
            initial_quantity: part.quantity,
            reserved_quantity: reserved_parts[i],
            remaining_quantity: remaining_parts[i],
        })
        .collect();

    Ok(RepairSchedulingReport {
        horizon_steps: config.horizon_steps,
        policy: config.policy,
        total_weighted_service_unit_steps_lost,
        jobs_completed_within_horizon,
        jobs_finishing_after_horizon,
        schedule,
        remaining_parts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(policy: RepairSchedulingPolicy) -> RepairSchedulingConfig {
        RepairSchedulingConfig::new(20, policy)
    }

    fn crew(id: &str, skill: &str) -> RepairCrew {
        RepairCrew::new(id, [skill], 0)
    }

    fn job(
        id: &str,
        skill: &str,
        duration: usize,
        priority: u32,
        value: f64,
        depends_on: impl IntoIterator<Item = impl Into<String>>,
    ) -> RepairJob {
        RepairJob::new(id, skill, duration, priority, value, 0, depends_on, Vec::<RepairPartRequirement>::new())
    }

    /// Independent exact reference for tiny, single-crew, no-dependency fixtures.
    /// This enumerates permutations instead of calling the production scheduler.
    fn brute_force_single_crew_best_loss(
        jobs: &[RepairJob],
        horizon_steps: usize,
    ) -> (f64, Vec<String>) {
        fn visit(
            jobs: &[RepairJob],
            horizon_steps: usize,
            used: &mut [bool],
            order: &mut Vec<usize>,
            best: &mut Option<(f64, Vec<String>)>,
        ) {
            if order.len() == jobs.len() {
                let mut time = 0_usize;
                let mut loss = 0.0_f64;
                let mut ids = Vec::with_capacity(order.len());
                for &job_index in order.iter() {
                    let job = &jobs[job_index];
                    time = time.max(job.outage_start_step);
                    time = time.saturating_add(job.duration_steps);
                    let lost_steps = time
                        .min(horizon_steps)
                        .saturating_sub(job.outage_start_step.min(horizon_steps));
                    loss += lost_steps as f64 * job.recovered_service_value_per_step;
                    ids.push(job.id.clone());
                }
                let replace = match best {
                    None => true,
                    Some((best_loss, best_ids)) => {
                        loss < *best_loss - EPSILON
                            || ((loss - *best_loss).abs() <= EPSILON && ids < *best_ids)
                    }
                };
                if replace {
                    *best = Some((loss, ids));
                }
                return;
            }
            for index in 0..jobs.len() {
                if !used[index] {
                    used[index] = true;
                    order.push(index);
                    visit(jobs, horizon_steps, used, order, best);
                    order.pop();
                    used[index] = false;
                }
            }
        }

        let mut best = None;
        visit(
            jobs,
            horizon_steps,
            &mut vec![false; jobs.len()],
            &mut Vec::with_capacity(jobs.len()),
            &mut best,
        );
        best.expect("non-empty fixture should have a permutation")
    }

    #[test]
    fn weighted_policy_matches_exhaustive_oracle_on_tiny_reference_case() {
        let jobs = [
            job("long-low-impact", "mechanical", 10, 0, 1.0, Vec::<String>::new()),
            job("short-critical", "mechanical", 1, 1, 100.0, Vec::<String>::new()),
        ];
        let crew = [crew("crew-a", "mechanical")];
        let (exact_loss, exact_order) = brute_force_single_crew_best_loss(&jobs, 20);
        let weighted = schedule_repairs(
            &jobs,
            &crew,
            &[],
            &config(RepairSchedulingPolicy::WeightedRecoveryPerStep),
        )
        .unwrap();
        assert_eq!(exact_loss, 111.0);
        assert_eq!(weighted.total_weighted_service_unit_steps_lost, exact_loss);
        assert_eq!(exact_order, vec!["short-critical", "long-low-impact"]);
    }

    #[test]
    fn weighted_recovery_beats_priority_first_on_disclosed_fixture() {
        let jobs = [
            job("long-low-impact", "mechanical", 10, 0, 1.0, Vec::<String>::new()),
            job("short-critical", "electrical", 1, 1, 100.0, Vec::<String>::new()),
        ];
        // Give one crew both skills so policy choice is actually tested.
        let crews = [RepairCrew::new("crew-a", ["mechanical", "electrical"], 0)];
        let priority = schedule_repairs(&jobs, &crews, &[], &config(RepairSchedulingPolicy::PriorityFirst)).unwrap();
        let weighted = schedule_repairs(&jobs, &crews, &[], &config(RepairSchedulingPolicy::WeightedRecoveryPerStep)).unwrap();
        assert!(weighted.total_weighted_service_unit_steps_lost < priority.total_weighted_service_unit_steps_lost);
        assert_eq!(weighted.schedule[0].job_id, "short-critical");
    }

    #[test]
    fn shortest_job_first_is_deterministic_and_short_job_starts_first() {
        let jobs = [
            job("long", "mechanical", 5, 0, 1.0, Vec::<String>::new()),
            job("short", "mechanical", 1, 9, 1.0, Vec::<String>::new()),
        ];
        let crews = [crew("crew-a", "mechanical")];
        let report = schedule_repairs(&jobs, &crews, &[], &config(RepairSchedulingPolicy::ShortestJobFirst)).unwrap();
        assert_eq!(report.schedule[0].job_id, "short");
        assert_eq!(report.schedule[0].finish_step, 1);
    }

    #[test]
    fn heterogeneous_skills_are_respected() {
        let jobs = [
            job("water", "plumbing", 2, 0, 1.0, Vec::<String>::new()),
            job("grid", "electrical", 3, 0, 1.0, Vec::<String>::new()),
        ];
        let crews = [crew("plumber", "plumbing"), crew("electrician", "electrical")];
        let report = schedule_repairs(&jobs, &crews, &[], &config(RepairSchedulingPolicy::PriorityFirst)).unwrap();
        assert_eq!(report.jobs_completed_within_horizon, 2);
        assert!(report.schedule.iter().any(|s| s.job_id == "water" && s.crew_id == "plumber"));
        assert!(report.schedule.iter().any(|s| s.job_id == "grid" && s.crew_id == "electrician"));
    }

    #[test]
    fn flexible_crew_is_preserved_for_skill_constrained_work() {
        let jobs = [
            job("mechanical", "mechanical", 4, 0, 1.0, Vec::<String>::new()),
            job("electrical", "electrical", 2, 1, 1.0, Vec::<String>::new()),
        ];
        let crews = [
            RepairCrew::new("crew-a-flexible", ["mechanical", "electrical"], 0),
            RepairCrew::new("crew-z-specialist", ["mechanical"], 0),
        ];
        let report = schedule_repairs(
            &jobs,
            &crews,
            &[],
            &config(RepairSchedulingPolicy::PriorityFirst),
        )
        .unwrap();
        let mechanical = report.schedule.iter().find(|s| s.job_id == "mechanical").unwrap();
        let electrical = report.schedule.iter().find(|s| s.job_id == "electrical").unwrap();
        assert_eq!(mechanical.crew_id, "crew-z-specialist");
        assert_eq!(electrical.crew_id, "crew-a-flexible");
        assert_eq!(mechanical.start_step, 0);
        assert_eq!(electrical.start_step, 0);
    }

    #[test]
    fn one_crew_never_has_overlapping_jobs() {
        let jobs = [
            job("a", "mechanical", 3, 0, 1.0, Vec::<String>::new()),
            job("b", "mechanical", 2, 1, 1.0, Vec::<String>::new()),
        ];
        let report = schedule_repairs(&jobs, &[crew("crew", "mechanical")], &[], &config(RepairSchedulingPolicy::PriorityFirst)).unwrap();
        let a = report.schedule.iter().find(|s| s.job_id == "a").unwrap();
        let b = report.schedule.iter().find(|s| s.job_id == "b").unwrap();
        assert!(a.finish_step <= b.start_step || b.finish_step <= a.start_step);
    }

    #[test]
    fn dependency_must_finish_before_dependent_repair_starts() {
        let jobs = [
            job("power", "electrical", 3, 0, 10.0, Vec::<String>::new()),
            job("water", "plumbing", 2, 0, 5.0, ["power"]),
        ];
        let crews = [crew("electrician", "electrical"), crew("plumber", "plumbing")];
        let report = schedule_repairs(&jobs, &crews, &[], &config(RepairSchedulingPolicy::PriorityFirst)).unwrap();
        let power = report.schedule.iter().find(|s| s.job_id == "power").unwrap();
        let water = report.schedule.iter().find(|s| s.job_id == "water").unwrap();
        assert!(water.start_step >= power.finish_step);
    }

    #[test]
    fn spare_parts_are_reserved_once_and_stock_never_goes_negative() {
        let jobs = [
            RepairJob::new("replace-pump", "mechanical", 2, 0, 3.0, 0, Vec::<String>::new(),
                [RepairPartRequirement::new("pump-kit", 1.0)]),
        ];
        let parts = [RepairPartStock::new("pump-kit", 2.0)];
        let report = schedule_repairs(&jobs, &[crew("mechanic", "mechanical")], &parts, &config(RepairSchedulingPolicy::PriorityFirst)).unwrap();
        assert_eq!(report.remaining_parts[0].reserved_quantity, 1.0);
        assert_eq!(report.remaining_parts[0].remaining_quantity, 1.0);
    }

    #[test]
    fn insufficient_parts_fail_closed() {
        let jobs = [RepairJob::new("replace-pump", "mechanical", 2, 0, 3.0, 0, Vec::<String>::new(),
            [RepairPartRequirement::new("pump-kit", 1.0)])];
        let parts = [RepairPartStock::new("pump-kit", 0.0)];
        assert!(matches!(
            schedule_repairs(&jobs, &[crew("mechanic", "mechanical")], &parts, &config(RepairSchedulingPolicy::PriorityFirst)),
            Err(RepairSchedulingError::InsufficientPartStock { .. })
        ));
    }

    #[test]
    fn out_of_horizon_repair_is_not_reported_completed() {
        let jobs = [RepairJob::new("late", "mechanical", 5, 0, 1.0, 0, Vec::<String>::new(),
            Vec::<RepairPartRequirement>::new())];
        let crews = [RepairCrew::new("late-crew", ["mechanical"], 30)];
        let report = schedule_repairs(&jobs, &crews, &[], &RepairSchedulingConfig::new(10, RepairSchedulingPolicy::PriorityFirst)).unwrap();
        assert_eq!(report.jobs_completed_within_horizon, 0);
        assert_eq!(report.jobs_finishing_after_horizon, 1);
        assert_eq!(report.total_weighted_service_unit_steps_lost, 1.0 * 10.0);
    }

    #[test]
    fn dependency_cycle_fails_closed() {
        let jobs = [
            job("a", "mechanical", 1, 0, 1.0, ["b"]),
            job("b", "mechanical", 1, 0, 1.0, ["a"]),
        ];
        assert!(matches!(
            schedule_repairs(&jobs, &[crew("crew", "mechanical")], &[], &config(RepairSchedulingPolicy::PriorityFirst)),
            Err(RepairSchedulingError::DependencyCycle(_))
        ));
    }

    #[test]
    fn unknown_job_dependency_and_unknown_crew_skill_fail_closed() {
        let unknown_dependency = [job("a", "mechanical", 1, 0, 1.0, ["missing"])];
        assert!(matches!(
            schedule_repairs(&unknown_dependency, &[crew("crew", "mechanical")], &[], &config(RepairSchedulingPolicy::PriorityFirst)),
            Err(RepairSchedulingError::UnknownJobDependency { .. })
        ));
        let unknown_skill = [job("a", "chemical", 1, 0, 1.0, Vec::<String>::new())];
        assert!(matches!(
            schedule_repairs(&unknown_skill, &[crew("crew", "mechanical")], &[], &config(RepairSchedulingPolicy::PriorityFirst)),
            Err(RepairSchedulingError::NoQualifiedCrew(_))
        ));
    }

    #[test]
    fn invalid_numeric_inputs_fail_closed() {
        let bad_value = [job("a", "mechanical", 1, 0, f64::NAN, Vec::<String>::new())];
        assert!(matches!(
            schedule_repairs(&bad_value, &[crew("crew", "mechanical")], &[], &config(RepairSchedulingPolicy::PriorityFirst)),
            Err(RepairSchedulingError::InvalidRecoveredServiceValue(_))
        ));
        let bad_stock = [RepairPartStock::new("part", -1.0)];
        assert!(matches!(
            schedule_repairs(
                &[RepairJob::new("a", "mechanical", 1, 0, 1.0, 0, Vec::<String>::new(), [RepairPartRequirement::new("part", 1.0)])],
                &[crew("crew", "mechanical")],
                &bad_stock,
                &config(RepairSchedulingPolicy::PriorityFirst),
            ),
            Err(RepairSchedulingError::InvalidPartStock(_))
        ));
    }

    #[test]
    fn equal_priority_schedules_are_order_invariant() {
        let a = job("alpha", "mechanical", 2, 4, 1.0, Vec::<String>::new());
        let z = job("zeta", "mechanical", 1, 4, 1.0, Vec::<String>::new());
        let crews = [crew("crew", "mechanical")];
        let one = schedule_repairs(&[a.clone(), z.clone()], &crews, &[], &config(RepairSchedulingPolicy::ShortestJobFirst)).unwrap();
        let two = schedule_repairs(&[z, a], &crews, &[], &config(RepairSchedulingPolicy::ShortestJobFirst)).unwrap();
        assert_eq!(one.schedule, two.schedule);
    }

    #[test]
    fn same_input_produces_identical_schedule() {
        let jobs = [
            job("a", "mechanical", 3, 2, 2.0, Vec::<String>::new()),
            job("b", "mechanical", 1, 1, 5.0, Vec::<String>::new()),
        ];
        let crews = [crew("crew", "mechanical")];
        let one = schedule_repairs(&jobs, &crews, &[], &config(RepairSchedulingPolicy::WeightedRecoveryPerStep)).unwrap();
        let two = schedule_repairs(&jobs, &crews, &[], &config(RepairSchedulingPolicy::WeightedRecoveryPerStep)).unwrap();
        assert_eq!(one, two);
    }

    #[test]
    fn out_of_horizon_outage_and_zero_duration_fail_closed() {
        let no_duration = [job("a", "mechanical", 0, 0, 1.0, Vec::<String>::new())];
        assert!(matches!(
            schedule_repairs(&no_duration, &[crew("crew", "mechanical")], &[], &config(RepairSchedulingPolicy::PriorityFirst)),
            Err(RepairSchedulingError::InvalidJobDuration(_))
        ));
        let outage_after_horizon = [RepairJob::new("a", "mechanical", 1, 0, 1.0, 20, Vec::<String>::new(),
            Vec::<RepairPartRequirement>::new())];
        assert!(matches!(
            schedule_repairs(&outage_after_horizon, &[crew("crew", "mechanical")], &[], &config(RepairSchedulingPolicy::PriorityFirst)),
            Err(RepairSchedulingError::OutageStartsOutsideHorizon(_))
        ));
    }

    #[test]
    fn all_tasks_require_skills_and_declared_finite_parts() {
        let jobs = [RepairJob::new("a", "mechanical", 1, 0, 1.0, 0, Vec::<String>::new(),
            [RepairPartRequirement::new("bearing", 1.0)])];
        let bad_part = [RepairPartStock::new("bearing", f64::INFINITY)];
        assert!(matches!(
            schedule_repairs(&jobs, &[crew("crew", "mechanical")], &bad_part, &config(RepairSchedulingPolicy::PriorityFirst)),
            Err(RepairSchedulingError::InvalidPartStock(_))
        ));
        let valid_parts = [RepairPartStock::new("bearing", 0.5)];
        assert!(matches!(
            schedule_repairs(&jobs, &[crew("crew", "mechanical")], &valid_parts, &config(RepairSchedulingPolicy::PriorityFirst)),
            Err(RepairSchedulingError::InsufficientPartStock { .. })
        ));
    }
}
