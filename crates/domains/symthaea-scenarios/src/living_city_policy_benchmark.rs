//! Deterministic, synthetic-only policy benchmark harness for Living City.
//!
//! This module deliberately models one abstract essential service, not an urban
//! physical system. It exists to qualify experiment plumbing, policy information
//! boundaries, delayed interventions, service-floor accounting and paired
//! comparisons before any domain model is integrated.
//!
//! Policies receive only the current observation and benchmark config. They do
//! not receive the scenario's hidden actual supply/demand timeline. Results are
//! synthetic engineering fixtures, not calibrated predictions or evidence of
//! real-world impact.

use std::collections::BTreeSet;

pub const LIVING_CITY_POLICY_BENCHMARK_VERSION: u32 = 1;
pub const POLICY_BENCHMARK_SCHEMA_VERSION: u32 = 1;
pub const PER_MILLE: u16 = 1_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolicyKind {
    /// Control baseline: never deploy backup.
    NoIntervention,
    /// Deploy when fresh reported supply is below fresh reported demand.
    ReactiveGreedy,
    /// Deploy only when the reported unmet-service cost exceeds the deployment cost.
    CostFirst,
    /// Protect the configured service floor; stale/missing observations fail closed.
    FloorFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Observation {
    /// Reported supply. None means unavailable, not zero.
    pub reported_supply_units: Option<u32>,
    /// Reported demand. None means unavailable, not zero.
    pub reported_demand_units: Option<u32>,
    /// Age of the observation in simulation steps.
    pub age_steps: usize,
    /// Public forecast from the observation channel; not hidden scenario truth.
    pub forecast_service_ratio_per_mille: Option<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScenarioStep {
    /// Simulator-only ground truth; never passed to the policy function.
    pub actual_supply_units: u32,
    /// Simulator-only ground truth; never passed to the policy function.
    pub actual_demand_units: u32,
    pub observation: Observation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyScenario {
    pub scenario_id: String,
    pub steps: Vec<ScenarioStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkConfig {
    /// A backup event adds this much supply per active step.
    pub backup_capacity_units: u32,
    /// Number of steps for which a backup event remains active.
    pub backup_duration_steps: usize,
    /// Delay between requesting backup and its first active step.
    pub deployment_delay_steps: usize,
    /// Maximum number of backup events a policy may request.
    pub max_deployments: usize,
    /// Minimum delivered-service ratio, in thousandths, to avoid a floor breach.
    pub minimum_service_ratio_per_mille: u16,
    /// Synthetic resource/cost units paid once per deployment.
    pub deployment_cost_units: u64,
    /// Synthetic cost per unit of service left unmet per step.
    pub unmet_service_cost_per_unit: u64,
    /// An observation older than this is stale.
    pub maximum_observation_age_steps: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl std::fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid policy benchmark {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for BenchmarkError {}

fn invalid(field: &'static str, reason: &'static str) -> BenchmarkError {
    BenchmarkError { field, reason }
}

impl BenchmarkConfig {
    pub fn validate(self) -> Result<(), BenchmarkError> {
        if self.backup_capacity_units == 0 {
            return Err(invalid("backup_capacity_units", "must be greater than zero"));
        }
        if self.backup_duration_steps == 0 {
            return Err(invalid("backup_duration_steps", "must be greater than zero"));
        }
        if !(1..=PER_MILLE).contains(&self.minimum_service_ratio_per_mille) {
            return Err(invalid(
                "minimum_service_ratio_per_mille",
                "must be in 1..=1000",
            ));
        }
        Ok(())
    }
}

fn validate_scenario(scenario: &PolicyScenario) -> Result<(), BenchmarkError> {
    if scenario.scenario_id.trim().is_empty() {
        return Err(invalid("scenario_id", "must be non-empty"));
    }
    if scenario.steps.is_empty() {
        return Err(invalid("steps", "scenario must contain at least one step"));
    }
    for step in &scenario.steps {
        if step
            .observation
            .forecast_service_ratio_per_mille
            .is_some_and(|ratio| ratio > PER_MILLE)
        {
            return Err(invalid(
                "forecast_service_ratio_per_mille",
                "forecast ratio must be in 0..=1000",
            ));
        }
        // Deliberately do not validate observations against ground truth: a
        // biased/stale sensor is a valid adversarial test input.
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentRecord {
    pub requested_at_step: usize,
    pub active_from_step: usize,
    /// Exclusive endpoint: active when active_from <= step < active_until.
    pub active_until_step: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyRun {
    pub schema_version: u32,
    pub suite_version: u32,
    pub scenario_id: String,
    pub policy: PolicyKind,
    pub config: BenchmarkConfig,
    pub step_count: usize,
    /// Sum of abstract service units not delivered across all steps.
    pub unmet_service_units: u64,
    /// Number of steps where actual delivered service violated the hard floor.
    pub floor_breach_steps: usize,
    pub floor_breach_step_indices: Vec<usize>,
    pub longest_consecutive_floor_breach_steps: usize,
    pub first_floor_breach_step: Option<usize>,
    pub deployments: Vec<DeploymentRecord>,
    pub deployment_cost_units: u64,
    pub unmet_service_cost_units: u64,
    pub total_cost_units: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyComparison {
    pub scenario_id: String,
    pub baseline_policy: PolicyKind,
    pub candidate_policy: PolicyKind,
    pub baseline_unmet_service_units: u64,
    pub candidate_unmet_service_units: u64,
    /// Candidate minus baseline; negative is an improvement.
    pub unmet_service_change_units: i128,
    /// Positive means service loss was reduced. None if baseline loss is zero.
    pub service_loss_reduction_per_mille: Option<i128>,
}

fn observation_is_fresh(observation: Observation, config: BenchmarkConfig) -> bool {
    observation.age_steps <= config.maximum_observation_age_steps
        && observation.reported_supply_units.is_some()
        && observation.reported_demand_units.is_some()
}

fn floor_is_breached(supply: u32, demand: u32, floor_per_mille: u16) -> bool {
    if demand == 0 {
        return false;
    }
    (supply as u128) * (PER_MILLE as u128)
        < (demand as u128) * (floor_per_mille as u128)
}

fn should_deploy(
    policy: PolicyKind,
    observation: Observation,
    config: BenchmarkConfig,
) -> bool {
    let fresh = observation_is_fresh(observation, config);
    let current_deficit = match (
        observation.reported_supply_units,
        observation.reported_demand_units,
    ) {
        (Some(supply), Some(demand)) if fresh => Some(demand.saturating_sub(supply)),
        _ => None,
    };

    match policy {
        PolicyKind::NoIntervention => false,
        PolicyKind::ReactiveGreedy => current_deficit.is_some_and(|deficit| deficit > 0),
        PolicyKind::CostFirst => current_deficit.is_some_and(|deficit| {
            (deficit as u64).saturating_mul(config.unmet_service_cost_per_unit)
                > config.deployment_cost_units
        }),
        PolicyKind::FloorFirst => {
            // Unknown/stale telemetry is not interpreted as a healthy service.
            if !fresh {
                return true;
            }
            let (supply, demand) = (
                observation.reported_supply_units.unwrap_or(0),
                observation.reported_demand_units.unwrap_or(0),
            );
            floor_is_breached(
                supply,
                demand,
                config.minimum_service_ratio_per_mille,
            ) || observation
                .forecast_service_ratio_per_mille
                .is_some_and(|ratio| ratio < config.minimum_service_ratio_per_mille)
        }
    }
}

/// Run a single policy. The policy decision function receives only the current
/// Observation, not PolicyScenario or its hidden future truth.
pub fn evaluate_policy(
    scenario: &PolicyScenario,
    policy: PolicyKind,
    config: BenchmarkConfig,
) -> Result<PolicyRun, BenchmarkError> {
    config.validate()?;
    validate_scenario(scenario)?;

    let mut deployments: Vec<DeploymentRecord> = Vec::new();
    let mut unmet_service_units = 0_u64;
    let mut floor_breach_step_indices = Vec::new();
    let mut current_breach_run = 0_usize;
    let mut longest_consecutive_floor_breach_steps = 0_usize;
    let mut unmet_service_cost_units = 0_u64;

    for (step_index, step) in scenario.steps.iter().enumerate() {
        let event_covers_or_is_pending = deployments.iter().any(|event| {
            event.active_from_step > step_index || event.active_until_step > step_index
        });
        let can_deploy = !event_covers_or_is_pending
            && deployments.len() < config.max_deployments;

        if can_deploy && should_deploy(policy, step.observation, config) {
            let active_from_step = step_index.saturating_add(config.deployment_delay_steps);
            let active_until_step =
                active_from_step.saturating_add(config.backup_duration_steps);
            deployments.push(DeploymentRecord {
                requested_at_step: step_index,
                active_from_step,
                active_until_step,
            });
        }

        let backup_active = deployments.iter().any(|event| {
            event.active_from_step <= step_index && step_index < event.active_until_step
        });
        let available_supply = if backup_active {
            step.actual_supply_units
                .saturating_add(config.backup_capacity_units)
        } else {
            step.actual_supply_units
        };
        let delivered = available_supply.min(step.actual_demand_units);
        let unmet = step.actual_demand_units - delivered;
        unmet_service_units = unmet_service_units.saturating_add(unmet as u64);
        unmet_service_cost_units = unmet_service_cost_units.saturating_add(
            (unmet as u64).saturating_mul(config.unmet_service_cost_per_unit),
        );

        if floor_is_breached(
            delivered,
            step.actual_demand_units,
            config.minimum_service_ratio_per_mille,
        ) {
            floor_breach_step_indices.push(step_index);
            current_breach_run = current_breach_run.saturating_add(1);
            longest_consecutive_floor_breach_steps =
                longest_consecutive_floor_breach_steps.max(current_breach_run);
        } else {
            current_breach_run = 0;
        }
    }

    let deployment_cost_units =
        (deployments.len() as u64).saturating_mul(config.deployment_cost_units);
    let total_cost_units = deployment_cost_units.saturating_add(unmet_service_cost_units);
    Ok(PolicyRun {
        schema_version: POLICY_BENCHMARK_SCHEMA_VERSION,
        suite_version: LIVING_CITY_POLICY_BENCHMARK_VERSION,
        scenario_id: scenario.scenario_id.clone(),
        policy,
        config,
        step_count: scenario.steps.len(),
        unmet_service_units,
        floor_breach_steps: floor_breach_step_indices.len(),
        first_floor_breach_step: floor_breach_step_indices.first().copied(),
        floor_breach_step_indices,
        longest_consecutive_floor_breach_steps,
        deployments,
        deployment_cost_units,
        unmet_service_cost_units,
        total_cost_units,
    })
}

/// Run every requested policy on every scenario. Duplicate IDs/policies are
/// rejected so accidental omission/duplication cannot masquerade as coverage.
pub fn evaluate_suite(
    scenarios: &[PolicyScenario],
    policies: &[PolicyKind],
    config: BenchmarkConfig,
) -> Result<Vec<PolicyRun>, BenchmarkError> {
    config.validate()?;
    if scenarios.is_empty() {
        return Err(invalid("scenarios", "suite cannot be empty"));
    }
    if policies.is_empty() {
        return Err(invalid("policies", "suite must include at least one policy"));
    }

    let mut scenario_ids = BTreeSet::new();
    for scenario in scenarios {
        validate_scenario(scenario)?;
        if !scenario_ids.insert(scenario.scenario_id.as_str()) {
            return Err(invalid("scenario_id", "scenario IDs must be unique"));
        }
    }
    let mut unique_policies = BTreeSet::new();
    for policy in policies {
        if !unique_policies.insert(*policy) {
            return Err(invalid("policies", "policy list contains a duplicate"));
        }
    }

    let mut results = Vec::with_capacity(scenarios.len().saturating_mul(policies.len()));
    for scenario in scenarios {
        for policy in policies {
            results.push(evaluate_policy(scenario, *policy, config)?);
        }
    }
    Ok(results)
}

/// Compare two policies only when their scenarios, timestep counts and full
/// configurations match. This function reports differences; it does not assert
/// that any policy is better or qualifies for deployment.
pub fn compare_service_loss(
    candidate: &PolicyRun,
    baseline: &PolicyRun,
) -> Result<PolicyComparison, BenchmarkError> {
    if candidate.scenario_id != baseline.scenario_id {
        return Err(invalid(
            "scenario_id",
            "paired comparisons require the same scenario ID",
        ));
    }
    if candidate.step_count != baseline.step_count || candidate.config != baseline.config {
        return Err(invalid(
            "paired_run",
            "step count and full benchmark config must match",
        ));
    }
    let change = (candidate.unmet_service_units as i128)
        - (baseline.unmet_service_units as i128);
    let reduction = if baseline.unmet_service_units == 0 {
        None
    } else {
        Some(
            ((baseline.unmet_service_units as i128
                - candidate.unmet_service_units as i128)
                * (PER_MILLE as i128))
                / (baseline.unmet_service_units as i128),
        )
    };
    Ok(PolicyComparison {
        scenario_id: candidate.scenario_id.clone(),
        baseline_policy: baseline.policy,
        candidate_policy: candidate.policy,
        baseline_unmet_service_units: baseline.unmet_service_units,
        candidate_unmet_service_units: candidate.unmet_service_units,
        unmet_service_change_units: change,
        service_loss_reduction_per_mille: reduction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BenchmarkConfig {
        BenchmarkConfig {
            backup_capacity_units: 5,
            backup_duration_steps: 3,
            deployment_delay_steps: 0,
            max_deployments: 2,
            minimum_service_ratio_per_mille: 950,
            deployment_cost_units: 2,
            unmet_service_cost_per_unit: 1,
            maximum_observation_age_steps: 0,
        }
    }

    fn observed(supply: u32, demand: u32) -> Observation {
        Observation {
            reported_supply_units: Some(supply),
            reported_demand_units: Some(demand),
            age_steps: 0,
            forecast_service_ratio_per_mille: None,
        }
    }

    fn step(actual_supply: u32, demand: u32, observation: Observation) -> ScenarioStep {
        ScenarioStep {
            actual_supply_units: actual_supply,
            actual_demand_units: demand,
            observation,
        }
    }

    fn scenario(id: &str, steps: Vec<ScenarioStep>) -> PolicyScenario {
        PolicyScenario {
            scenario_id: id.into(),
            steps,
        }
    }

    #[test]
    fn reactive_dispatch_covers_a_synthetic_immediate_deficit() {
        let case = scenario("immediate-deficit", vec![step(5, 10, observed(5, 10))]);
        let no_intervention =
            evaluate_policy(&case, PolicyKind::NoIntervention, config()).unwrap();
        let reactive =
            evaluate_policy(&case, PolicyKind::ReactiveGreedy, config()).unwrap();

        assert_eq!(no_intervention.unmet_service_units, 5);
        assert_eq!(no_intervention.floor_breach_steps, 1);
        assert_eq!(reactive.unmet_service_units, 0);
        assert_eq!(reactive.floor_breach_steps, 0);
        assert_eq!(reactive.deployments.len(), 1);
        assert_eq!(reactive.deployments[0].requested_at_step, 0);
        assert_eq!(reactive.deployments[0].active_from_step, 0);
    }

    #[test]
    fn floor_first_fails_closed_on_missing_or_stale_observations() {
        let mut cfg = config();
        cfg.maximum_observation_age_steps = 0;
        let stale = Observation {
            reported_supply_units: Some(20),
            reported_demand_units: Some(10),
            age_steps: 7,
            forecast_service_ratio_per_mille: Some(1_000),
        };
        let case = scenario("stale-sensor", vec![step(5, 10, stale)]);
        let floor_first =
            evaluate_policy(&case, PolicyKind::FloorFirst, cfg).unwrap();
        let reactive =
            evaluate_policy(&case, PolicyKind::ReactiveGreedy, cfg).unwrap();

        assert_eq!(floor_first.deployments.len(), 1);
        assert_eq!(floor_first.floor_breach_steps, 0);
        assert!(reactive.deployments.is_empty());
        assert_eq!(reactive.floor_breach_steps, 1);
    }

    #[test]
    fn missing_values_are_not_silently_coerced_to_zero_or_healthy() {
        let missing = Observation {
            reported_supply_units: None,
            reported_demand_units: None,
            age_steps: 0,
            forecast_service_ratio_per_mille: None,
        };
        let case = scenario("missing-sensor", vec![step(5, 10, missing)]);
        let result = evaluate_policy(&case, PolicyKind::FloorFirst, config()).unwrap();
        assert_eq!(result.deployments.len(), 1);
        assert_eq!(result.floor_breach_steps, 0);
    }

    #[test]
    fn a_deployment_delay_is_reflected_in_actual_service_metrics() {
        let mut cfg = config();
        cfg.deployment_delay_steps = 1;
        cfg.backup_duration_steps = 2;
        let case = scenario(
            "delayed-backup",
            vec![
                step(5, 10, observed(5, 10)),
                step(5, 10, observed(5, 10)),
                step(5, 10, observed(5, 10)),
            ],
        );
        let run = evaluate_policy(&case, PolicyKind::ReactiveGreedy, cfg).unwrap();

        assert_eq!(run.deployments.len(), 1);
        assert_eq!(run.deployments[0].active_from_step, 1);
        assert_eq!(run.deployments[0].active_until_step, 3);
        assert_eq!(run.floor_breach_step_indices, vec![0]);
        assert_eq!(run.longest_consecutive_floor_breach_steps, 1);
        assert_eq!(run.unmet_service_units, 5);
    }

    #[test]
    fn decisions_do_not_peek_at_hidden_future_truth() {
        let observations = vec![observed(5, 10), observed(10, 10), observed(5, 10)];
        let first = scenario(
            "world-a",
            observations
                .iter()
                .enumerate()
                .map(|(index, observation)| {
                    step(if index == 0 { 5 } else { 10 }, 10, *observation)
                })
                .collect(),
        );
        let second = scenario(
            "world-b",
            observations
                .iter()
                .enumerate()
                .map(|(index, observation)| {
                    step(if index == 0 { 5 } else { 0 }, 10, *observation)
                })
                .collect(),
        );
        let mut cfg = config();
        cfg.backup_duration_steps = 1;
        let first_run = evaluate_policy(&first, PolicyKind::FloorFirst, cfg).unwrap();
        let second_run = evaluate_policy(&second, PolicyKind::FloorFirst, cfg).unwrap();

        assert_ne!(first.steps[1].actual_supply_units, second.steps[1].actual_supply_units);
        assert_eq!(first_run.deployments, second_run.deployments);
    }

    #[test]
    fn paired_comparison_uses_identical_scenario_and_config() {
        let case = scenario("paired", vec![step(5, 10, observed(5, 10))]);
        let baseline =
            evaluate_policy(&case, PolicyKind::NoIntervention, config()).unwrap();
        let candidate =
            evaluate_policy(&case, PolicyKind::ReactiveGreedy, config()).unwrap();
        let comparison = compare_service_loss(&candidate, &baseline).unwrap();

        assert_eq!(comparison.unmet_service_change_units, -5);
        assert_eq!(comparison.service_loss_reduction_per_mille, Some(1_000));
    }

    #[test]
    fn percentage_reduction_is_undefined_when_baseline_has_no_loss() {
        let case = scenario("healthy", vec![step(10, 10, observed(10, 10))]);
        let baseline =
            evaluate_policy(&case, PolicyKind::NoIntervention, config()).unwrap();
        let candidate =
            evaluate_policy(&case, PolicyKind::FloorFirst, config()).unwrap();
        let comparison = compare_service_loss(&candidate, &baseline).unwrap();

        assert_eq!(comparison.service_loss_reduction_per_mille, None);
    }

    #[test]
    fn a_delivered_ratio_exactly_at_the_floor_is_not_a_breach() {
        let mut cfg = config();
        cfg.minimum_service_ratio_per_mille = 950;
        let case = scenario("floor-boundary", vec![step(19, 20, observed(19, 20))]);
        let run = evaluate_policy(&case, PolicyKind::NoIntervention, cfg).unwrap();
        assert_eq!(run.floor_breach_steps, 0);
    }

    #[test]
    fn zero_demand_is_not_a_service_floor_breach() {
        let case = scenario("zero-demand", vec![step(0, 0, observed(0, 0))]);
        let run = evaluate_policy(&case, PolicyKind::NoIntervention, config()).unwrap();
        assert_eq!(run.floor_breach_steps, 0);
        assert_eq!(run.unmet_service_units, 0);
    }

    #[test]
    fn suite_rejects_duplicate_ids_and_duplicate_policies() {
        let case = scenario("duplicate", vec![step(1, 1, observed(1, 1))]);
        assert_eq!(
            evaluate_suite(&[case.clone(), case.clone()], &[PolicyKind::NoIntervention], config())
                .unwrap_err()
                .field,
            "scenario_id"
        );
        assert_eq!(
            evaluate_suite(
                &[case],
                &[PolicyKind::NoIntervention, PolicyKind::NoIntervention],
                config(),
            )
                .unwrap_err()
                .field,
            "policies"
        );
    }

    #[test]
    fn invalid_forecast_ratio_is_rejected() {
        let mut observation = observed(10, 10);
        observation.forecast_service_ratio_per_mille = Some(1_001);
        let case = scenario("invalid-forecast", vec![step(10, 10, observation)]);
        assert_eq!(
            evaluate_policy(&case, PolicyKind::FloorFirst, config())
                .unwrap_err()
                .field,
            "forecast_service_ratio_per_mille"
        );
    }

    #[test]
    fn repeated_runs_produce_identical_results() {
        let case = scenario(
            "repeatable",
            vec![
                step(5, 10, observed(5, 10)),
                step(10, 10, observed(10, 10)),
                step(4, 10, observed(4, 10)),
                step(10, 10, observed(10, 10)),
            ],
        );
        let first = evaluate_suite(
            &[case.clone()],
            &[
                PolicyKind::NoIntervention,
                PolicyKind::ReactiveGreedy,
                PolicyKind::CostFirst,
                PolicyKind::FloorFirst,
            ],
            config(),
        )
        .unwrap();
        let second = evaluate_suite(
            &[case],
            &[
                PolicyKind::NoIntervention,
                PolicyKind::ReactiveGreedy,
                PolicyKind::CostFirst,
                PolicyKind::FloorFirst,
            ],
            config(),
        )
        .unwrap();
        assert_eq!(first, second);
    }
}
