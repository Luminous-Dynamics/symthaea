// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Event-triggered rolling-horizon repair dispatch under partial information.
//!
//! Decisions use only observable state: completed jobs, available crews, released
//! outages, dependencies, and remaining parts. A realized duration is consulted
//! only after its job is selected, to schedule its completion event. Pending jobs
//! are ranked with nominal durations. This is a transparent deterministic research
//! abstraction, not a calibrated field model.
//!
//! Limitations: no travel, replenishment, preemption, partial repair, changing
//! demand, or probabilistic duration model. Scenario outcomes are illustrative
//! unless their distributions and weights are justified by observations.

use crate::repair_crew_scheduling::{
    RepairCrew, RepairJob, RepairPartRequirement, RepairPartStock, RepairSchedulingPolicy,
    RemainingRepairPartStock,
};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RealizedRepairDuration {
    pub job_id: String,
    pub duration_steps: usize,
}

impl RealizedRepairDuration {
    pub fn new(job_id: impl Into<String>, duration_steps: usize) -> Self {
        Self { job_id: job_id.into(), duration_steps }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RollingHorizonConfig {
    pub horizon_steps: usize,
    pub policy: RepairSchedulingPolicy,
}

impl RollingHorizonConfig {
    pub fn new(horizon_steps: usize, policy: RepairSchedulingPolicy) -> Self {
        Self { horizon_steps, policy }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepairDispatchDecision {
    pub decision_step: usize,
    pub job_id: String,
    pub crew_id: String,
    pub completed_jobs_observed: Vec<String>,
    pub eligible_jobs_observed: Vec<String>,
    pub available_crews_observed: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RollingHorizonScheduledRepair {
    pub job_id: String,
    pub crew_id: String,
    pub start_step: usize,
    pub finish_step: usize,
    pub outage_start_step: usize,
    pub realized_duration_steps: usize,
    pub weighted_service_unit_steps_lost_within_horizon: f64,
    pub parts_consumed: Vec<RepairPartRequirement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RollingHorizonRepairReport {
    pub horizon_steps: usize,
    pub policy: RepairSchedulingPolicy,
    pub total_weighted_service_unit_steps_lost: f64,
    pub jobs_started: usize,
    pub jobs_completed_within_horizon: usize,
    pub jobs_in_progress_at_horizon: usize,
    pub jobs_not_started: Vec<String>,
    /// Dispatch order, not finish-time order.
    pub schedule: Vec<RollingHorizonScheduledRepair>,
    /// Audit trail of observable facts available to every dispatch decision.
    pub decisions: Vec<RepairDispatchDecision>,
    pub remaining_parts: Vec<RemainingRepairPartStock>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RollingHorizonRepairError {
    EmptyJobSet,
    EmptyCrewSet,
    ZeroHorizon,
    EmptyIdentifier,
    DuplicateJob(String),
    DuplicateCrew(String),
    DuplicatePart(String),
    InvalidNominalDuration(String),
    InvalidRealizedDuration(String),
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
    MissingRealizedDuration(String),
    UnknownRealizedDuration(String),
    DuplicateRealizedDuration(String),
    TimeOverflow(String),
}

impl std::fmt::Display for RollingHorizonRepairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyJobSet => write!(f, "at least one repair job is required"),
            Self::EmptyCrewSet => write!(f, "at least one repair crew is required"),
            Self::ZeroHorizon => write!(f, "horizon_steps must be greater than zero"),
            Self::EmptyIdentifier => write!(f, "identifiers and required skills must not be empty"),
            Self::DuplicateJob(id) => write!(f, "duplicate repair job id: {id}"),
            Self::DuplicateCrew(id) => write!(f, "duplicate repair crew id: {id}"),
            Self::DuplicatePart(id) => write!(f, "duplicate spare-part id: {id}"),
            Self::InvalidNominalDuration(id) => write!(f, "nominal duration must be positive for {id}"),
            Self::InvalidRealizedDuration(id) => write!(f, "realized duration must be positive for {id}"),
            Self::InvalidRecoveredServiceValue(id) => write!(f, "service recovery value must be finite and positive for {id}"),
            Self::OutageStartsOutsideHorizon(id) => write!(f, "outage for {id} starts outside the horizon"),
            Self::UnknownJobDependency { job_id, dependency_id } => write!(f, "job {job_id} depends on unknown job {dependency_id}"),
            Self::DuplicateJobDependency { job_id, dependency_id } => write!(f, "job {job_id} repeats dependency {dependency_id}"),
            Self::DependencyCycle(id) => write!(f, "repair dependency cycle includes job {id}"),
            Self::InvalidPartQuantity { job_id, part_id } => write!(f, "part quantity must be finite and positive for {job_id}/{part_id}"),
            Self::UnknownPartRequirement { job_id, part_id } => write!(f, "job {job_id} requires unknown part {part_id}"),
            Self::DuplicatePartRequirement { job_id, part_id } => write!(f, "job {job_id} repeats part requirement {part_id}"),
            Self::InvalidPartStock(id) => write!(f, "part stock must be finite and non-negative for {id}"),
            Self::NoQualifiedCrew(id) => write!(f, "no crew has the required skill for job {id}"),
            Self::MissingRealizedDuration(id) => write!(f, "missing realized duration for job {id}"),
            Self::UnknownRealizedDuration(id) => write!(f, "realized duration refers to unknown job {id}"),
            Self::DuplicateRealizedDuration(id) => write!(f, "duplicate realized duration for job {id}"),
            Self::TimeOverflow(id) => write!(f, "repair finish time overflows usize for job {id}"),
        }
    }
}
impl std::error::Error for RollingHorizonRepairError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JobState { Pending, Active, Complete }

#[derive(Clone, Debug)]
struct Candidate { job_index: usize, crew_index: usize }

fn policy_order(
    a: &Candidate,
    b: &Candidate,
    jobs: &[RepairJob],
    crews: &[RepairCrew],
    crew_flexibility: &[usize],
    policy: RepairSchedulingPolicy,
) -> Ordering {
    let ja = &jobs[a.job_index];
    let jb = &jobs[b.job_index];
    let order = match policy {
        RepairSchedulingPolicy::PriorityFirst => ja.priority.cmp(&jb.priority)
            .then_with(|| ja.duration_steps.cmp(&jb.duration_steps)),
        RepairSchedulingPolicy::ShortestJobFirst => ja.duration_steps.cmp(&jb.duration_steps)
            .then_with(|| ja.priority.cmp(&jb.priority)),
        RepairSchedulingPolicy::WeightedRecoveryPerStep => {
            let score_a = ja.recovered_service_value_per_step / ja.duration_steps as f64;
            let score_b = jb.recovered_service_value_per_step / jb.duration_steps as f64;
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
                .then_with(|| ja.priority.cmp(&jb.priority))
        }
    }.then_with(|| ja.id.cmp(&jb.id));

    // On a tied job choice, preserve flexibility by assigning a crew qualified
    // for fewer jobs before consuming a crew with broader skill coverage.
    order.then_with(|| crew_flexibility[a.crew_index].cmp(&crew_flexibility[b.crew_index]))
        .then_with(|| crews[a.crew_index].id.cmp(&crews[b.crew_index].id))
}

fn validate_inputs(
    jobs: &[RepairJob],
    crews: &[RepairCrew],
    parts: &[RepairPartStock],
    outcomes: &[RealizedRepairDuration],
    config: &RollingHorizonConfig,
) -> Result<(HashMap<String, usize>, HashMap<String, usize>), RollingHorizonRepairError> {
    if config.horizon_steps == 0 { return Err(RollingHorizonRepairError::ZeroHorizon); }
    if jobs.is_empty() { return Err(RollingHorizonRepairError::EmptyJobSet); }
    if crews.is_empty() { return Err(RollingHorizonRepairError::EmptyCrewSet); }

    let mut job_index = HashMap::with_capacity(jobs.len());
    for (i, job) in jobs.iter().enumerate() {
        if job.id.trim().is_empty() || job.required_skill.trim().is_empty() {
            return Err(RollingHorizonRepairError::EmptyIdentifier);
        }
        if job_index.insert(job.id.clone(), i).is_some() {
            return Err(RollingHorizonRepairError::DuplicateJob(job.id.clone()));
        }
        if job.duration_steps == 0 {
            return Err(RollingHorizonRepairError::InvalidNominalDuration(job.id.clone()));
        }
        if !job.recovered_service_value_per_step.is_finite() || job.recovered_service_value_per_step <= 0.0 {
            return Err(RollingHorizonRepairError::InvalidRecoveredServiceValue(job.id.clone()));
        }
        if job.outage_start_step >= config.horizon_steps {
            return Err(RollingHorizonRepairError::OutageStartsOutsideHorizon(job.id.clone()));
        }
    }

    let mut crew_ids = HashSet::new();
    for crew in crews {
        if crew.id.trim().is_empty() || crew.skills.iter().any(|skill| skill.trim().is_empty()) {
            return Err(RollingHorizonRepairError::EmptyIdentifier);
        }
        if !crew_ids.insert(&crew.id) {
            return Err(RollingHorizonRepairError::DuplicateCrew(crew.id.clone()));
        }
    }

    let mut part_index = HashMap::with_capacity(parts.len());
    for (i, part) in parts.iter().enumerate() {
        if part.id.trim().is_empty() { return Err(RollingHorizonRepairError::EmptyIdentifier); }
        if part_index.insert(part.id.clone(), i).is_some() {
            return Err(RollingHorizonRepairError::DuplicatePart(part.id.clone()));
        }
        if !part.quantity.is_finite() || part.quantity < 0.0 {
            return Err(RollingHorizonRepairError::InvalidPartStock(part.id.clone()));
        }
    }

    for job in jobs {
        if !crews.iter().any(|crew| crew.skills.iter().any(|skill| skill == &job.required_skill)) {
            return Err(RollingHorizonRepairError::NoQualifiedCrew(job.id.clone()));
        }
        let mut dependencies = HashSet::new();
        for dep in &job.depends_on {
            if !job_index.contains_key(dep) {
                return Err(RollingHorizonRepairError::UnknownJobDependency {
                    job_id: job.id.clone(), dependency_id: dep.clone(),
                });
            }
            if !dependencies.insert(dep) {
                return Err(RollingHorizonRepairError::DuplicateJobDependency {
                    job_id: job.id.clone(), dependency_id: dep.clone(),
                });
            }
        }
        let mut requirements = HashSet::new();
        for req in &job.required_parts {
            if !part_index.contains_key(&req.part_id) {
                return Err(RollingHorizonRepairError::UnknownPartRequirement {
                    job_id: job.id.clone(), part_id: req.part_id.clone(),
                });
            }
            if !requirements.insert(&req.part_id) {
                return Err(RollingHorizonRepairError::DuplicatePartRequirement {
                    job_id: job.id.clone(), part_id: req.part_id.clone(),
                });
            }
            if !req.quantity.is_finite() || req.quantity <= 0.0 {
                return Err(RollingHorizonRepairError::InvalidPartQuantity {
                    job_id: job.id.clone(), part_id: req.part_id.clone(),
                });
            }
        }
    }

    fn visit(
        i: usize, jobs: &[RepairJob], indices: &HashMap<String, usize>, state: &mut [u8],
    ) -> Result<(), RollingHorizonRepairError> {
        match state[i] {
            2 => return Ok(()),
            1 => return Err(RollingHorizonRepairError::DependencyCycle(jobs[i].id.clone())),
            _ => {}
        }
        state[i] = 1;
        let mut deps: Vec<usize> = jobs[i].depends_on.iter().map(|id| indices[id]).collect();
        deps.sort_by(|a, b| jobs[*a].id.cmp(&jobs[*b].id));
        for dep in deps { visit(dep, jobs, indices, state)?; }
        state[i] = 2;
        Ok(())
    }
    let mut visit_state = vec![0; jobs.len()];
    for i in 0..jobs.len() { visit(i, jobs, &job_index, &mut visit_state)?; }

    let mut duration_by_job = HashMap::with_capacity(outcomes.len());
    for outcome in outcomes {
        if !job_index.contains_key(&outcome.job_id) {
            return Err(RollingHorizonRepairError::UnknownRealizedDuration(outcome.job_id.clone()));
        }
        if outcome.duration_steps == 0 {
            return Err(RollingHorizonRepairError::InvalidRealizedDuration(outcome.job_id.clone()));
        }
        if duration_by_job.insert(outcome.job_id.clone(), outcome.duration_steps).is_some() {
            return Err(RollingHorizonRepairError::DuplicateRealizedDuration(outcome.job_id.clone()));
        }
    }
    for job in jobs {
        if !duration_by_job.contains_key(&job.id) {
            return Err(RollingHorizonRepairError::MissingRealizedDuration(job.id.clone()));
        }
    }
    Ok((job_index, duration_by_job))
}

/// Dispatch from the current observable state rather than freezing a nominal plan.
///
/// A job's realized duration is indexed only after the job has been selected. The
/// scenario is known to the simulation harness, but its future outcomes are hidden
/// from policy ranking. All completed repairs are observed at each event time.
/// Spare parts are consumed at dispatch; infeasible jobs remain explicitly
/// unstarted and incur unmet-service loss through the configured horizon.
pub fn dispatch_repairs_rolling_horizon(
    jobs: &[RepairJob],
    crews: &[RepairCrew],
    parts: &[RepairPartStock],
    realized_durations: &[RealizedRepairDuration],
    config: &RollingHorizonConfig,
) -> Result<RollingHorizonRepairReport, RollingHorizonRepairError> {
    let (job_index, duration_by_job) = validate_inputs(jobs, crews, parts, realized_durations, config)?;
    let part_index: HashMap<String, usize> = parts.iter().enumerate()
        .map(|(i, part)| (part.id.clone(), i)).collect();
    let crew_flexibility: Vec<usize> = crews.iter().map(|crew| jobs.iter()
        .filter(|job| crew.skills.iter().any(|skill| skill == &job.required_skill))
        .count()).collect();
    let mut remaining_parts: Vec<f64> = parts.iter().map(|part| part.quantity).collect();
    let mut consumed_parts = vec![0.0_f64; parts.len()];
    let mut crew_available: Vec<usize> = crews.iter().map(|crew| crew.available_from_step).collect();
    let mut states = vec![JobState::Pending; jobs.len()];
    let mut finishes: Vec<Option<usize>> = vec![None; jobs.len()];
    let mut schedule: Vec<RollingHorizonScheduledRepair> = Vec::new();
    let mut decisions = Vec::new();
    let mut now = 0usize;

    loop {
        // Completion feedback is exposed only when the event time reaches finish.
        for i in 0..jobs.len() {
            if states[i] == JobState::Active && finishes[i].is_some_and(|finish| finish <= now) {
                states[i] = JobState::Complete;
            }
        }
        if now >= config.horizon_steps { break; }

        loop {
            let available_crews: Vec<usize> = crews.iter().enumerate()
                .filter(|(i, _)| crew_available[*i] <= now)
                .map(|(i, _)| i).collect();
            if available_crews.is_empty() { break; }

            let mut candidates = Vec::new();
            for (job_i, job) in jobs.iter().enumerate() {
                if states[job_i] != JobState::Pending || job.outage_start_step > now { continue; }
                if !job.depends_on.iter().all(|dep| states[job_index[dep]] == JobState::Complete) {
                    continue;
                }
                if !job.required_parts.iter().all(|req| {
                    remaining_parts[part_index[&req.part_id]] + EPSILON >= req.quantity
                }) { continue; }
                for crew_i in &available_crews {
                    if crews[*crew_i].skills.iter().any(|skill| skill == &job.required_skill) {
                        candidates.push(Candidate { job_index: job_i, crew_index: *crew_i });
                    }
                }
            }
            if candidates.is_empty() { break; }
            candidates.sort_by(|a, b| policy_order(a, b, jobs, crews, &crew_flexibility, config.policy));
            let chosen = candidates[0].clone();
            let job = &jobs[chosen.job_index];
            let crew = &crews[chosen.crew_index];

            let mut completed_observed: Vec<String> = jobs.iter().enumerate()
                .filter(|(i, _)| states[*i] == JobState::Complete)
                .map(|(_, job)| job.id.clone()).collect();
            completed_observed.sort();
            let mut eligible_observed: Vec<String> = candidates.iter()
                .map(|item| jobs[item.job_index].id.clone()).collect();
            eligible_observed.sort();
            eligible_observed.dedup();
            let mut crews_observed: Vec<String> = available_crews.iter()
                .map(|i| crews[*i].id.clone()).collect();
            crews_observed.sort();
            decisions.push(RepairDispatchDecision {
                decision_step: now,
                job_id: job.id.clone(),
                crew_id: crew.id.clone(),
                completed_jobs_observed: completed_observed,
                eligible_jobs_observed: eligible_observed,
                available_crews_observed: crews_observed,
            });

            // Hidden outcome is consulted only after selection; it changes the
            // future event clock, not the ordering of currently pending jobs.
            let duration = duration_by_job[&job.id];
            let finish = now.checked_add(duration)
                .ok_or_else(|| RollingHorizonRepairError::TimeOverflow(job.id.clone()))?;
            let mut parts_for_job = Vec::with_capacity(job.required_parts.len());
            for requirement in &job.required_parts {
                let index = part_index[&requirement.part_id];
                remaining_parts[index] = (remaining_parts[index] - requirement.quantity).max(0.0);
                consumed_parts[index] += requirement.quantity;
                parts_for_job.push(requirement.clone());
            }
            states[chosen.job_index] = JobState::Active;
            finishes[chosen.job_index] = Some(finish);
            crew_available[chosen.crew_index] = finish;
            schedule.push(RollingHorizonScheduledRepair {
                job_id: job.id.clone(),
                crew_id: crew.id.clone(),
                start_step: now,
                finish_step: finish,
                outage_start_step: job.outage_start_step,
                realized_duration_steps: duration,
                weighted_service_unit_steps_lost_within_horizon: 0.0,
                parts_consumed: parts_for_job,
            });
        }

        let mut next_event: Option<usize> = None;
        let mut consider = |candidate: usize| {
            if candidate > now {
                next_event = Some(next_event.map_or(candidate, |existing| existing.min(candidate)));
            }
        };
        for finish in finishes.iter().flatten() { consider(*finish); }
        for available in &crew_available {
            if *available > now { consider(*available); }
        }
        for (i, job) in jobs.iter().enumerate() {
            if states[i] == JobState::Pending && job.outage_start_step > now {
                consider(job.outage_start_step);
            }
        }
        match next_event {
            Some(next) => now = next.min(config.horizon_steps),
            None => break,
        }
    }

    for repair in &mut schedule {
        let effective_finish = repair.finish_step.min(config.horizon_steps);
        let outage = repair.outage_start_step.min(config.horizon_steps);
        let lost_steps = effective_finish.saturating_sub(outage);
        repair.weighted_service_unit_steps_lost_within_horizon =
            lost_steps as f64 * jobs[job_index[&repair.job_id]].recovered_service_value_per_step;
    }
    let mut total_loss = 0.0;
    for (i, job) in jobs.iter().enumerate() {
        let finish = finishes[i].unwrap_or(config.horizon_steps).min(config.horizon_steps);
        let outage = job.outage_start_step.min(config.horizon_steps);
        total_loss += finish.saturating_sub(outage) as f64 * job.recovered_service_value_per_step;
    }
    let completed_count = finishes.iter().filter(|f| f.is_some_and(|step| step <= config.horizon_steps)).count();
    let in_progress_count = states.iter().enumerate()
        .filter(|(i, state)| **state == JobState::Active && finishes[*i].is_some_and(|f| f > config.horizon_steps))
        .count();
    let mut not_started: Vec<String> = jobs.iter().enumerate()
        .filter(|(i, _)| states[*i] == JobState::Pending)
        .map(|(_, job)| job.id.clone()).collect();
    not_started.sort();
    let remaining_parts = parts.iter().enumerate().map(|(i, part)| RemainingRepairPartStock {
        part_id: part.id.clone(),
        initial_quantity: part.quantity,
        reserved_quantity: consumed_parts[i],
        remaining_quantity: remaining_parts[i],
    }).collect();

    Ok(RollingHorizonRepairReport {
        horizon_steps: config.horizon_steps,
        policy: config.policy,
        total_weighted_service_unit_steps_lost: total_loss,
        jobs_started: schedule.len(),
        jobs_completed_within_horizon: completed_count,
        jobs_in_progress_at_horizon: in_progress_count,
        jobs_not_started: not_started,
        schedule,
        decisions,
        remaining_parts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repair_crew_scheduling::{schedule_repairs, RepairSchedulingConfig};

    fn job(id: &str, skill: &str, nominal: usize, priority: u32, value: f64,
        outage: usize, deps: &[&str], parts: Vec<RepairPartRequirement>) -> RepairJob {
        RepairJob::new(id, skill, nominal, priority, value, outage, deps.iter().copied(), parts)
    }
    fn crew(id: &str, skills: &[&str], available: usize) -> RepairCrew {
        RepairCrew::new(id, skills.iter().copied(), available)
    }
    fn durations(items: &[(&str, usize)]) -> Vec<RealizedRepairDuration> {
        items.iter().map(|(id, d)| RealizedRepairDuration::new(*id, *d)).collect()
    }
    fn run(jobs: &[RepairJob], crews: &[RepairCrew], parts: &[RepairPartStock],
        durations: &[RealizedRepairDuration], horizon: usize, policy: RepairSchedulingPolicy)
        -> RollingHorizonRepairReport {
        dispatch_repairs_rolling_horizon(jobs, crews, parts, durations,
            &RollingHorizonConfig::new(horizon, policy)).expect("valid test fixture")
    }

    #[test]
    fn adapts_order_after_observed_long_repair() {
        let jobs = vec![
            job("A", "general", 1, 0, 1.0, 0, &[], vec![]),
            job("B", "general", 2, 1, 100.0, 5, &[], vec![]),
            job("C", "general", 2, 20, 1.0, 0, &[], vec![]),
        ];
        let crews = vec![crew("crew", &["general"], 0)];
        let fixed = schedule_repairs(&jobs, &crews, &[],
            &RepairSchedulingConfig::new(20, RepairSchedulingPolicy::PriorityFirst)).unwrap();
        assert_eq!(fixed.schedule.iter().map(|r| r.job_id.as_str()).collect::<Vec<_>>(), vec!["A", "C", "B"]);
        // Replay only the fixed dispatch order, retiming each repair with the same
        // realized durations. This is a paired comparison, not a clairvoyant oracle.
        let mut crew_ready = 0usize;
        let mut fixed_loss = 0.0;
        for planned in &fixed.schedule {
            let j = jobs.iter().find(|j| j.id == planned.job_id).unwrap();
            let actual_duration = match planned.job_id.as_str() { "A" => 10, "B" => 2, "C" => 2, _ => unreachable!() };
            let start = crew_ready.max(j.outage_start_step);
            let finish = start + actual_duration;
            crew_ready = finish;
            fixed_loss += finish.min(20).saturating_sub(j.outage_start_step) as f64 * j.recovered_service_value_per_step;
        }
        assert!((fixed_loss - 922.0).abs() < 1.0e-9);
        let adaptive = run(&jobs, &crews, &[], &durations(&[("A", 10), ("B", 2), ("C", 2)]),
            20, RepairSchedulingPolicy::PriorityFirst);
        assert_eq!(adaptive.schedule.iter().map(|r| r.job_id.as_str()).collect::<Vec<_>>(), vec!["A", "B", "C"]);
        assert!((adaptive.total_weighted_service_unit_steps_lost - 724.0).abs() < 1.0e-9);
    }

    #[test]
    fn future_duration_does_not_change_the_first_decision() {
        let jobs = vec![job("A", "g", 2, 0, 1.0, 0, &[], vec![]), job("B", "g", 2, 1, 1.0, 0, &[], vec![])];
        let crews = vec![crew("crew", &["g"], 0)];
        let a = run(&jobs, &crews, &[], &durations(&[("A", 2), ("B", 100)]), 20, RepairSchedulingPolicy::PriorityFirst);
        let b = run(&jobs, &crews, &[], &durations(&[("A", 2), ("B", 1)]), 20, RepairSchedulingPolicy::PriorityFirst);
        assert_eq!(a.decisions[0].job_id, b.decisions[0].job_id);
        assert_eq!(a.decisions[0].eligible_jobs_observed, b.decisions[0].eligible_jobs_observed);
    }

    #[test]
    fn dependent_job_waits_for_observed_completion() {
        let jobs = vec![job("upstream", "civil", 1, 0, 1.0, 0, &[], vec![]),
            job("downstream", "electric", 1, 1, 1.0, 0, &["upstream"], vec![])];
        let crews = vec![crew("c1", &["civil"], 0), crew("c2", &["electric"], 0)];
        let report = run(&jobs, &crews, &[], &durations(&[("upstream", 5), ("downstream", 2)]), 10, RepairSchedulingPolicy::PriorityFirst);
        let downstream = report.schedule.iter().find(|r| r.job_id == "downstream").unwrap();
        assert_eq!(downstream.start_step, 5);
        assert!(report.decisions.iter().find(|d| d.job_id == "downstream").unwrap().completed_jobs_observed.contains(&"upstream".to_string()));
    }

    #[test]
    fn respects_crew_availability_and_outage_release() {
        let jobs = vec![job("repair", "e", 1, 0, 1.0, 3, &[], vec![])];
        let crews = vec![crew("late", &["e"], 4)];
        let report = run(&jobs, &crews, &[], &durations(&[("repair", 2)]), 10, RepairSchedulingPolicy::PriorityFirst);
        assert_eq!(report.schedule[0].start_step, 4);
    }

    #[test]
    fn consumes_finite_parts_only_when_started() {
        let req = RepairPartRequirement::new("relay", 1.0);
        let jobs = vec![job("first", "e", 1, 0, 1.0, 0, &[], vec![req.clone()]),
            job("second", "e", 1, 1, 1.0, 0, &[], vec![req])];
        let crews = vec![crew("crew", &["e"], 0)];
        let report = run(&jobs, &crews, &[RepairPartStock::new("relay", 1.0)],
            &durations(&[("first", 1), ("second", 1)]), 10, RepairSchedulingPolicy::PriorityFirst);
        assert_eq!(report.jobs_started, 1);
        assert_eq!(report.jobs_not_started, vec!["second".to_string()]);
        assert_eq!(report.remaining_parts[0].remaining_quantity, 0.0);
        assert_eq!(report.remaining_parts[0].reserved_quantity, 1.0);
    }

    #[test]
    fn every_assignment_has_skill_and_exclusive_crew() {
        let jobs = vec![job("e", "electric", 1, 0, 1.0, 0, &[], vec![]), job("m", "mechanical", 1, 1, 1.0, 0, &[], vec![])];
        let crews = vec![crew("ce", &["electric"], 0), crew("cm", &["mechanical"], 0)];
        let report = run(&jobs, &crews, &[], &durations(&[("e", 3), ("m", 2)]), 10, RepairSchedulingPolicy::PriorityFirst);
        for repair in &report.schedule {
            let j = jobs.iter().find(|j| j.id == repair.job_id).unwrap();
            let c = crews.iter().find(|c| c.id == repair.crew_id).unwrap();
            assert!(c.skills.contains(&j.required_skill));
        }
        for c in &crews {
            let mut intervals: Vec<(usize, usize)> = report.schedule.iter().filter(|r| r.crew_id == c.id).map(|r| (r.start_step, r.finish_step)).collect();
            intervals.sort();
            assert!(intervals.windows(2).all(|w| w[0].1 <= w[1].0));
        }
    }

    #[test]
    fn exact_horizon_completion_and_overrun_are_distinguished() {
        let jobs = vec![job("done", "g", 1, 0, 1.0, 0, &[], vec![]), job("long", "g", 1, 1, 2.0, 0, &[], vec![])];
        let crews = vec![crew("crew1", &["g"], 0), crew("crew2", &["g"], 0)];
        let report = run(&jobs, &crews, &[], &durations(&[("done", 2), ("long", 8)]), 2, RepairSchedulingPolicy::PriorityFirst);
        assert_eq!(report.jobs_completed_within_horizon, 1);
        assert_eq!(report.jobs_in_progress_at_horizon, 1);
    }

    #[test]
    fn decisions_and_reports_are_deterministic() {
        let jobs = vec![job("a", "g", 2, 1, 2.0, 0, &[], vec![]), job("b", "g", 1, 0, 1.0, 0, &[], vec![])];
        let crews = vec![crew("crew", &["g"], 0)];
        let observed = durations(&[("a", 3), ("b", 2)]);
        assert_eq!(run(&jobs, &crews, &[], &observed, 10, RepairSchedulingPolicy::PriorityFirst),
            run(&jobs, &crews, &[], &observed, 10, RepairSchedulingPolicy::PriorityFirst));
    }

    #[test]
    fn rejects_missing_unknown_duplicate_and_zero_durations() {
        let jobs = vec![job("a", "g", 1, 0, 1.0, 0, &[], vec![])];
        let crews = vec![crew("crew", &["g"], 0)];
        let cfg = RollingHorizonConfig::new(10, RepairSchedulingPolicy::PriorityFirst);
        assert_eq!(dispatch_repairs_rolling_horizon(&jobs, &crews, &[], &[], &cfg).unwrap_err(), RollingHorizonRepairError::MissingRealizedDuration("a".into()));
        assert_eq!(dispatch_repairs_rolling_horizon(&jobs, &crews, &[], &durations(&[("a", 1), ("a", 2)]), &cfg).unwrap_err(), RollingHorizonRepairError::DuplicateRealizedDuration("a".into()));
        assert_eq!(dispatch_repairs_rolling_horizon(&jobs, &crews, &[], &durations(&[("a", 0)]), &cfg).unwrap_err(), RollingHorizonRepairError::InvalidRealizedDuration("a".into()));
        assert!(matches!(dispatch_repairs_rolling_horizon(&jobs, &crews, &[], &durations(&[("a", 1), ("missing", 1)]), &cfg), Err(RollingHorizonRepairError::UnknownRealizedDuration(id)) if id == "missing"));
    }

    #[test]
    fn rejects_dependency_cycles_and_unknown_dependencies() {
        let crews = vec![crew("crew", &["g"], 0)];
        let cycle = vec![job("a", "g", 1, 0, 1.0, 0, &["b"], vec![]), job("b", "g", 1, 1, 1.0, 0, &["a"], vec![])];
        assert!(matches!(dispatch_repairs_rolling_horizon(&cycle, &crews, &[], &durations(&[("a", 1), ("b", 1)]), &RollingHorizonConfig::new(10, RepairSchedulingPolicy::PriorityFirst)), Err(RollingHorizonRepairError::DependencyCycle(_))));
        let unknown = vec![job("a", "g", 1, 0, 1.0, 0, &["missing"], vec![])];
        assert!(matches!(dispatch_repairs_rolling_horizon(&unknown, &crews, &[], &durations(&[("a", 1)]), &RollingHorizonConfig::new(10, RepairSchedulingPolicy::PriorityFirst)), Err(RollingHorizonRepairError::UnknownJobDependency { .. })));
    }

    #[test]
    fn chosen_job_is_ranked_on_nominal_duration() {
        let jobs = vec![job("long_nominal", "g", 8, 0, 1.0, 0, &[], vec![]), job("short_nominal", "g", 1, 1, 1.0, 0, &[], vec![])];
        let crews = vec![crew("crew", &["g"], 0)];
        let report = run(&jobs, &crews, &[], &durations(&[("long_nominal", 1), ("short_nominal", 20)]), 30, RepairSchedulingPolicy::ShortestJobFirst);
        assert_eq!(report.decisions[0].job_id, "short_nominal");
    }
}
