// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Out-of-sample repair-policy evaluation under uncertain repair durations.
//!
//! A schedule is chosen using nominal durations only. Its dispatch order and
//! crew assignments are then held fixed while alternate duration scenarios are
//! replayed. Actual durations may delay later jobs through crew contention and
//! prerequisite dependencies, but cannot reorder decisions using future duration
//! information. This avoids a common "perfect information" comparison error.
//!
//! Scenario weights are caller-supplied and must sum to one. They are not
//! probabilities unless supported by evidence. CVaR summarizes the weighted upper
//! tail of lost-service outcomes; it does not replace per-scenario results or
//! distributional/equity constraints.

use std::collections::{HashMap, HashSet};

use super::repair_crew_scheduling::{
    schedule_repairs, RepairCrew, RepairJob, RepairPartStock, RepairSchedulingConfig,
    RepairSchedulingError, RepairSchedulingPolicy,
};

const EPSILON: f64 = 1.0e-9;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobDuration {
    pub job_id: String,
    pub duration_steps: usize,
}

impl JobDuration {
    pub fn new(job_id: impl Into<String>, duration_steps: usize) -> Self {
        Self { job_id: job_id.into(), duration_steps }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepairDurationScenario {
    pub id: String,
    /// Externally supplied scenario weight; all scenario weights must sum to 1.
    /// This is a probability only when empirical/statistical evidence justifies it.
    pub weight: f64,
    /// Must contain exactly one duration for each nominal job.
    pub durations: Vec<JobDuration>,
}

impl RepairDurationScenario {
    pub fn new(
        id: impl Into<String>,
        weight: f64,
        durations: impl IntoIterator<Item = JobDuration>,
    ) -> Self {
        Self {
            id: id.into(),
            weight,
            durations: durations.into_iter().collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobustRepairConfig {
    pub horizon_steps: usize,
    /// CVaR confidence level alpha in [0, 1). CVaR averages the worst 1-alpha
    /// weighted fraction of scenario losses.
    pub cvar_alpha: f64,
}

impl RobustRepairConfig {
    pub fn new(horizon_steps: usize, cvar_alpha: f64) -> Self {
        Self { horizon_steps, cvar_alpha }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayedRepair {
    pub job_id: String,
    /// Crew assignment selected using nominal information only.
    pub crew_id: String,
    pub start_step: usize,
    pub finish_step: usize,
    pub realized_duration_steps: usize,
    pub outage_start_step: usize,
    pub weighted_service_unit_steps_lost_within_horizon: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DurationScenarioOutcome {
    pub scenario_id: String,
    pub scenario_weight: f64,
    pub total_weighted_service_unit_steps_lost: f64,
    pub jobs_completed_within_horizon: usize,
    pub jobs_finishing_after_horizon: usize,
    /// The nominal plan's job/crew decisions, retimed under the realized durations.
    pub schedule: Vec<ReplayedRepair>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepairPolicyRiskSummary {
    pub policy: RepairSchedulingPolicy,
    pub expected_weighted_service_unit_steps_lost: f64,
    pub worst_case_weighted_service_unit_steps_lost: f64,
    pub cvar_weighted_service_unit_steps_lost: f64,
    pub cvar_alpha: f64,
    pub scenarios: Vec<DurationScenarioOutcome>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobustRepairEvaluation {
    pub horizon_steps: usize,
    pub cvar_alpha: f64,
    pub policy_summaries: Vec<RepairPolicyRiskSummary>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RobustRepairError {
    ZeroHorizon,
    InvalidCvarAlpha,
    EmptyScenarioSet,
    EmptyScenarioId,
    DuplicateScenarioId(String),
    InvalidScenarioWeight(String),
    ScenarioWeightsDoNotSumToOne,
    DuplicateNominalJob(String),
    EmptyDurationJobId,
    UnknownScenarioJob { scenario_id: String, job_id: String },
    DuplicateScenarioJob { scenario_id: String, job_id: String },
    MissingScenarioJob { scenario_id: String, job_id: String },
    InvalidScenarioDuration { scenario_id: String, job_id: String },
    NominalSchedule(RepairSchedulingError),
}

impl std::fmt::Display for RobustRepairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroHorizon => write!(f, "horizon_steps must be greater than zero"),
            Self::InvalidCvarAlpha => write!(f, "cvar_alpha must be finite and in [0, 1)"),
            Self::EmptyScenarioSet => write!(f, "at least one repair-duration scenario is required"),
            Self::EmptyScenarioId => write!(f, "scenario id must not be empty"),
            Self::DuplicateScenarioId(id) => write!(f, "duplicate scenario id: {id}"),
            Self::InvalidScenarioWeight(id) => {
                write!(f, "scenario weight must be finite and positive for {id}")
            }
            Self::ScenarioWeightsDoNotSumToOne => {
                write!(f, "scenario weights must sum to one within tolerance")
            }
            Self::DuplicateNominalJob(id) => write!(f, "duplicate nominal job id: {id}"),
            Self::EmptyDurationJobId => write!(f, "duration override job id must not be empty"),
            Self::UnknownScenarioJob { scenario_id, job_id } => {
                write!(f, "scenario {scenario_id} has duration for unknown job {job_id}")
            }
            Self::DuplicateScenarioJob { scenario_id, job_id } => {
                write!(f, "scenario {scenario_id} repeats duration for job {job_id}")
            }
            Self::MissingScenarioJob { scenario_id, job_id } => {
                write!(f, "scenario {scenario_id} lacks a duration for job {job_id}")
            }
            Self::InvalidScenarioDuration { scenario_id, job_id } => {
                write!(f, "scenario {scenario_id} has zero duration for job {job_id}")
            }
            Self::NominalSchedule(error) => write!(f, "nominal schedule infeasible: {error}"),
        }
    }
}

impl std::error::Error for RobustRepairError {}

fn validate_duration_scenarios(
    nominal_jobs: &[RepairJob],
    scenarios: &[RepairDurationScenario],
    config: &RobustRepairConfig,
) -> Result<(), RobustRepairError> {
    if config.horizon_steps == 0 {
        return Err(RobustRepairError::ZeroHorizon);
    }
    if !config.cvar_alpha.is_finite() || !(0.0..1.0).contains(&config.cvar_alpha) {
        return Err(RobustRepairError::InvalidCvarAlpha);
    }
    if scenarios.is_empty() {
        return Err(RobustRepairError::EmptyScenarioSet);
    }

    let mut nominal_ids = HashSet::with_capacity(nominal_jobs.len());
    for job in nominal_jobs {
        if !nominal_ids.insert(job.id.as_str()) {
            return Err(RobustRepairError::DuplicateNominalJob(job.id.clone()));
        }
    }

    let mut scenario_ids = HashSet::with_capacity(scenarios.len());
    let mut total_weight = 0.0_f64;
    for scenario in scenarios {
        if scenario.id.trim().is_empty() {
            return Err(RobustRepairError::EmptyScenarioId);
        }
        if !scenario_ids.insert(scenario.id.as_str()) {
            return Err(RobustRepairError::DuplicateScenarioId(scenario.id.clone()));
        }
        if !scenario.weight.is_finite() || scenario.weight <= 0.0 {
            return Err(RobustRepairError::InvalidScenarioWeight(scenario.id.clone()));
        }
        total_weight += scenario.weight;

        let mut duration_ids = HashSet::with_capacity(scenario.durations.len());
        for duration in &scenario.durations {
            if duration.job_id.trim().is_empty() {
                return Err(RobustRepairError::EmptyDurationJobId);
            }
            if !nominal_ids.contains(duration.job_id.as_str()) {
                return Err(RobustRepairError::UnknownScenarioJob {
                    scenario_id: scenario.id.clone(),
                    job_id: duration.job_id.clone(),
                });
            }
            if !duration_ids.insert(duration.job_id.as_str()) {
                return Err(RobustRepairError::DuplicateScenarioJob {
                    scenario_id: scenario.id.clone(),
                    job_id: duration.job_id.clone(),
                });
            }
            if duration.duration_steps == 0 {
                return Err(RobustRepairError::InvalidScenarioDuration {
                    scenario_id: scenario.id.clone(),
                    job_id: duration.job_id.clone(),
                });
            }
        }
        for job in nominal_jobs {
            if !duration_ids.contains(job.id.as_str()) {
                return Err(RobustRepairError::MissingScenarioJob {
                    scenario_id: scenario.id.clone(),
                    job_id: job.id.clone(),
                });
            }
        }
    }

    if (total_weight - 1.0).abs() > EPSILON {
        return Err(RobustRepairError::ScenarioWeightsDoNotSumToOne);
    }
    Ok(())
}

fn replay_nominal_plan(
    nominal_jobs: &[RepairJob],
    crews: &[RepairCrew],
    nominal_schedule: &[super::repair_crew_scheduling::ScheduledRepair],
    scenario: &RepairDurationScenario,
    horizon_steps: usize,
) -> DurationScenarioOutcome {
    let job_by_id: HashMap<&str, &RepairJob> =
        nominal_jobs.iter().map(|job| (job.id.as_str(), job)).collect();
    let durations_by_id: HashMap<&str, usize> = scenario
        .durations
        .iter()
        .map(|duration| (duration.job_id.as_str(), duration.duration_steps))
        .collect();

    let mut crew_available: HashMap<&str, usize> = crews
        .iter()
        .map(|crew| (crew.id.as_str(), crew.available_from_step))
        .collect();
    let mut finish_by_job: HashMap<&str, usize> = HashMap::new();
    let mut schedule = Vec::with_capacity(nominal_schedule.len());
    let mut total_loss = 0.0_f64;
    let mut jobs_completed_within_horizon = 0;

    for planned in nominal_schedule {
        let job = job_by_id[planned.job_id.as_str()];
        let duration = durations_by_id[job.id.as_str()];
        let crew_ready = *crew_available.get(planned.crew_id.as_str()).unwrap_or(&0);
        let dependency_ready = job
            .depends_on
            .iter()
            .filter_map(|dependency| finish_by_job.get(dependency.as_str()).copied())
            .max()
            .unwrap_or(0);
        let start_step = crew_ready
            .max(dependency_ready)
            .max(job.outage_start_step);
        let finish_step = start_step.saturating_add(duration);
        let lost_steps = finish_step
            .min(horizon_steps)
            .saturating_sub(job.outage_start_step.min(horizon_steps));
        let weighted_loss = lost_steps as f64 * job.recovered_service_value_per_step;

        if finish_step <= horizon_steps {
            jobs_completed_within_horizon += 1;
        }
        total_loss += weighted_loss;
        crew_available.insert(planned.crew_id.as_str(), finish_step);
        finish_by_job.insert(job.id.as_str(), finish_step);
        schedule.push(ReplayedRepair {
            job_id: job.id.clone(),
            crew_id: planned.crew_id.clone(),
            start_step,
            finish_step,
            realized_duration_steps: duration,
            outage_start_step: job.outage_start_step,
            weighted_service_unit_steps_lost_within_horizon: weighted_loss,
        });
    }

    DurationScenarioOutcome {
        scenario_id: scenario.id.clone(),
        scenario_weight: scenario.weight,
        total_weighted_service_unit_steps_lost: total_loss,
        jobs_completed_within_horizon,
        jobs_finishing_after_horizon: schedule.len() - jobs_completed_within_horizon,
        schedule,
    }
}

fn cvar_upper_tail(outcomes: &[DurationScenarioOutcome], alpha: f64) -> f64 {
    let tail_mass = 1.0 - alpha;
    let mut ranked: Vec<(f64, f64)> = outcomes
        .iter()
        .map(|outcome| (
            outcome.total_weighted_service_unit_steps_lost,
            outcome.scenario_weight,
        ))
        .collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut remaining_mass = tail_mass;
    let mut weighted_loss = 0.0_f64;
    for (loss, weight) in ranked {
        if remaining_mass <= EPSILON {
            break;
        }
        let included = weight.min(remaining_mass);
        weighted_loss += loss * included;
        remaining_mass -= included;
    }
    weighted_loss / tail_mass
}

/// Evaluate three repair policies using nominal durations, then replay each
/// policy's same job/crew decisions against all realized-duration scenarios.
///
/// The duration scenarios do not affect the initial plan. At replay, a longer
/// actual repair delays later work assigned to the same crew and all dependent
/// jobs. Scenario weights must sum to one. Report expected, worst-case and CVaR
/// losses together with per-scenario traces; no one summary substitutes for them.
pub fn evaluate_repair_policies_under_duration_scenarios(
    nominal_jobs: &[RepairJob],
    crews: &[RepairCrew],
    parts: &[RepairPartStock],
    scenarios: &[RepairDurationScenario],
    config: &RobustRepairConfig,
) -> Result<RobustRepairEvaluation, RobustRepairError> {
    validate_duration_scenarios(nominal_jobs, scenarios, config)?;

    let policies = [
        RepairSchedulingPolicy::PriorityFirst,
        RepairSchedulingPolicy::ShortestJobFirst,
        RepairSchedulingPolicy::WeightedRecoveryPerStep,
    ];
    let mut policy_summaries = Vec::with_capacity(policies.len());

    for policy in policies {
        // The schedule is chosen from nominal inputs only. No scenario outcome is
        // visible until the decision order and crew assignment have been produced.
        let nominal = schedule_repairs(
            nominal_jobs,
            crews,
            parts,
            &RepairSchedulingConfig::new(config.horizon_steps, policy),
        )
        .map_err(RobustRepairError::NominalSchedule)?;

        let outcomes: Vec<DurationScenarioOutcome> = scenarios
            .iter()
            .map(|scenario| {
                replay_nominal_plan(
                    nominal_jobs,
                    crews,
                    &nominal.schedule,
                    scenario,
                    config.horizon_steps,
                )
            })
            .collect();

        let expected = outcomes
            .iter()
            .map(|outcome| {
                outcome.scenario_weight * outcome.total_weighted_service_unit_steps_lost
            })
            .sum::<f64>();
        let worst = outcomes
            .iter()
            .map(|outcome| outcome.total_weighted_service_unit_steps_lost)
            .fold(0.0_f64, f64::max);
        let cvar = cvar_upper_tail(&outcomes, config.cvar_alpha);

        policy_summaries.push(RepairPolicyRiskSummary {
            policy,
            expected_weighted_service_unit_steps_lost: expected,
            worst_case_weighted_service_unit_steps_lost: worst,
            cvar_weighted_service_unit_steps_lost: cvar,
            cvar_alpha: config.cvar_alpha,
            scenarios: outcomes,
        });
    }

    Ok(RobustRepairEvaluation {
        horizon_steps: config.horizon_steps,
        cvar_alpha: config.cvar_alpha,
        policy_summaries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crew(id: &str, skill: &str) -> RepairCrew {
        RepairCrew::new(id, [skill], 0)
    }

    fn config(alpha: f64) -> RobustRepairConfig {
        RobustRepairConfig::new(20, alpha)
    }

    fn job(id: &str, duration: usize, priority: u32, value: f64) -> RepairJob {
        RepairJob::new(
            id,
            "mechanical",
            duration,
            priority,
            value,
            0,
            Vec::<String>::new(),
            Vec::<super::super::repair_crew_scheduling::RepairPartRequirement>::new(),
        )
    }

    fn scenario(id: &str, weight: f64, durations: &[(&str, usize)]) -> RepairDurationScenario {
        RepairDurationScenario::new(
            id,
            weight,
            durations.iter().map(|(job_id, duration)| JobDuration::new(*job_id, *duration)),
        )
    }

    #[test]
    fn cvar_exposes_a_rare_slow_repair_hidden_by_expected_loss() {
        let jobs = [job("pump", 1, 0, 1.0)];
        let scenarios = [
            scenario("ordinary", 0.9, &[("pump", 1)]),
            scenario("severe", 0.1, &[("pump", 9)]),
        ];
        let result = evaluate_repair_policies_under_duration_scenarios(
            &jobs,
            &[crew("crew", "mechanical")],
            &[],
            &scenarios,
            &config(0.9),
        )
        .unwrap();
        let summary = result
            .policy_summaries
            .iter()
            .find(|summary| summary.policy == RepairSchedulingPolicy::PriorityFirst)
            .unwrap();
        assert!((summary.expected_weighted_service_unit_steps_lost - 1.8).abs() < 1.0e-9);
        assert!((summary.worst_case_weighted_service_unit_steps_lost - 9.0).abs() < 1.0e-9);
        assert!((summary.cvar_weighted_service_unit_steps_lost - 9.0).abs() < 1.0e-9);
    }

    #[test]
    fn realized_durations_retime_but_do_not_reorder_the_nominal_plan() {
        let jobs = [
            job("a-priority", 2, 0, 1.0),
            job("b-later", 1, 1, 1.0),
        ];
        let scenarios = [
            scenario("fast-a", 0.5, &[("a-priority", 1), ("b-later", 9)]),
            scenario("slow-a", 0.5, &[("a-priority", 8), ("b-later", 1)]),
        ];
        let result = evaluate_repair_policies_under_duration_scenarios(
            &jobs,
            &[crew("crew", "mechanical")],
            &[],
            &scenarios,
            &config(0.9),
        )
        .unwrap();
        let summary = result.policy_summaries
            .iter()
            .find(|summary| summary.policy == RepairSchedulingPolicy::PriorityFirst)
            .unwrap();
        for outcome in &summary.scenarios {
            assert_eq!(outcome.schedule[0].job_id, "a-priority");
            assert_eq!(outcome.schedule[1].job_id, "b-later");
            assert_eq!(outcome.schedule[1].start_step, outcome.schedule[0].finish_step);
        }
    }

    #[test]
    fn longer_realized_repair_does_not_reduce_its_service_loss() {
        let jobs = [job("pump", 2, 0, 2.0)];
        let scenarios = [
            scenario("short", 0.5, &[("pump", 1)]),
            scenario("long", 0.5, &[("pump", 7)]),
        ];
        let result = evaluate_repair_policies_under_duration_scenarios(
            &jobs,
            &[crew("crew", "mechanical")],
            &[],
            &scenarios,
            &config(0.9),
        )
        .unwrap();
        let summary = &result.policy_summaries[0];
        assert!(
            summary.scenarios[1].total_weighted_service_unit_steps_lost
                >= summary.scenarios[0].total_weighted_service_unit_steps_lost
        );
    }

    #[test]
    fn dependency_delays_propagate_during_scenario_replay() {
        let jobs = [
            RepairJob::new("power", "electrical", 2, 0, 5.0, 0, Vec::<String>::new(),
                Vec::<super::super::repair_crew_scheduling::RepairPartRequirement>::new()),
            RepairJob::new("water", "plumbing", 2, 0, 4.0, 0, ["power"],
                Vec::<super::super::repair_crew_scheduling::RepairPartRequirement>::new()),
        ];
        let crews = [RepairCrew::new("electrician", ["electrical"], 0), RepairCrew::new("plumber", ["plumbing"], 0)];
        let scenarios = [
            scenario("nominal", 0.5, &[("power", 2), ("water", 2)]),
            scenario("slow-power", 0.5, &[("power", 7), ("water", 2)]),
        ];
        let result = evaluate_repair_policies_under_duration_scenarios(
            &jobs, &crews, &[], &scenarios, &config(0.9),
        ).unwrap();
        let summary = &result.policy_summaries[0];
        let nominal = &summary.scenarios[0].schedule;
        let slow = &summary.scenarios[1].schedule;
        let nominal_water = nominal.iter().find(|job| job.job_id == "water").unwrap();
        let slow_water = slow.iter().find(|job| job.job_id == "water").unwrap();
        assert!(slow_water.start_step > nominal_water.start_step);
        assert_eq!(slow_water.start_step, 7);
    }

    #[test]
    fn scenario_weights_must_sum_to_one() {
        let jobs = [job("pump", 1, 0, 1.0)];
        let scenarios = [scenario("only", 0.9, &[("pump", 1)])];
        assert!(matches!(
            evaluate_repair_policies_under_duration_scenarios(
                &jobs,
                &[crew("crew", "mechanical")],
                &[],
                &scenarios,
                &config(0.9),
            ),
            Err(RobustRepairError::ScenarioWeightsDoNotSumToOne)
        ));
    }

    #[test]
    fn scenario_must_cover_every_nominal_job_exactly_once() {
        let jobs = [job("a", 1, 0, 1.0), job("b", 2, 1, 1.0)];
        let scenarios = [scenario("missing-b", 1.0, &[("a", 1)])];
        assert!(matches!(
            evaluate_repair_policies_under_duration_scenarios(
                &jobs,
                &[crew("crew", "mechanical")],
                &[],
                &scenarios,
                &config(0.9),
            ),
            Err(RobustRepairError::MissingScenarioJob { .. })
        ));
    }

    #[test]
    fn repeated_or_unknown_duration_job_fails_closed() {
        let jobs = [job("pump", 1, 0, 1.0)];
        let repeated = [scenario("repeat", 1.0, &[("pump", 1), ("pump", 2)])];
        assert!(matches!(
            evaluate_repair_policies_under_duration_scenarios(
                &jobs, &[crew("crew", "mechanical")], &[], &repeated, &config(0.9),
            ),
            Err(RobustRepairError::DuplicateScenarioJob { .. })
        ));
        let unknown = [scenario("unknown", 1.0, &[("missing", 1)])];
        assert!(matches!(
            evaluate_repair_policies_under_duration_scenarios(
                &jobs, &[crew("crew", "mechanical")], &[], &unknown, &config(0.9),
            ),
            Err(RobustRepairError::UnknownScenarioJob { .. })
        ));
    }

    #[test]
    fn invalid_tail_alpha_and_zero_duration_fail_closed() {
        let jobs = [job("pump", 1, 0, 1.0)];
        let scenarios = [scenario("one", 1.0, &[("pump", 0)])];
        assert!(matches!(
            evaluate_repair_policies_under_duration_scenarios(
                &jobs, &[crew("crew", "mechanical")], &[], &scenarios, &config(0.9),
            ),
            Err(RobustRepairError::InvalidScenarioDuration { .. })
        ));
        let valid = [scenario("one", 1.0, &[("pump", 1)])];
        for alpha in [-0.1, 1.0, f64::NAN] {
            assert!(matches!(
                evaluate_repair_policies_under_duration_scenarios(
                    &jobs, &[crew("crew", "mechanical")], &[], &valid, &config(alpha),
                ),
                Err(RobustRepairError::InvalidCvarAlpha)
            ));
        }
    }

    #[test]
    fn late_crew_availability_is_preserved_in_scenario_replay() {
        let jobs = [job("pump", 2, 0, 1.0)];
        let crews = [RepairCrew::new("late-crew", ["mechanical"], 5)];
        let scenarios = [scenario("short", 1.0, &[("pump", 1)])];
        let result = evaluate_repair_policies_under_duration_scenarios(
            &jobs, &crews, &[], &scenarios, &config(0.9),
        ).unwrap();
        for summary in result.policy_summaries {
            let repair = &summary.scenarios[0].schedule[0];
            assert_eq!(repair.start_step, 5);
            assert_eq!(repair.finish_step, 6);
        }
    }

    #[test]
    fn identical_scenarios_produce_identical_policy_risk_reports() {
        let jobs = [job("pump", 2, 0, 1.0)];
        let scenarios = [scenario("nominal", 1.0, &[("pump", 2)])];
        let crews = [crew("crew", "mechanical")];
        let first = evaluate_repair_policies_under_duration_scenarios(
            &jobs, &crews, &[], &scenarios, &config(0.9),
        ).unwrap();
        let second = evaluate_repair_policies_under_duration_scenarios(
            &jobs, &crews, &[], &scenarios, &config(0.9),
        ).unwrap();
        assert_eq!(first, second);
    }
}
