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

use serde::{Deserialize, Serialize};

use crate::living_city_evidence::sha256_hex as hash_sha256_hex;

pub const LIVING_CITY_POLICY_BENCHMARK_VERSION: u32 = 1;
pub const POLICY_BENCHMARK_SCHEMA_VERSION: u32 = 1;
pub const PER_MILLE: u16 = 1_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

/// Baselines used by the built-in synthetic smoke suite. A Symthaea policy is
/// intentionally not represented here until one is integrated through the same
/// information/action contract and can be compared without privileged inputs.
pub const STANDARD_BASELINE_POLICIES: [PolicyKind; 4] = [
    PolicyKind::NoIntervention,
    PolicyKind::ReactiveGreedy,
    PolicyKind::CostFirst,
    PolicyKind::FloorFirst,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioStep {
    /// Simulator-only ground truth; never passed to the policy function.
    pub actual_supply_units: u32,
    /// Simulator-only ground truth; never passed to the policy function.
    pub actual_demand_units: u32,
    pub observation: Observation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyScenario {
    pub scenario_id: String,
    pub steps: Vec<ScenarioStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

/// Fixed configuration for the built-in synthetic-only test suite. These values
/// are fixture parameters, not recommended capacities or real-world targets.
pub const SYNTHETIC_SMOKE_CONFIG: BenchmarkConfig = BenchmarkConfig {
    backup_capacity_units: 5,
    backup_duration_steps: 2,
    deployment_delay_steps: 1,
    max_deployments: 2,
    minimum_service_ratio_per_mille: 950,
    deployment_cost_units: 2,
    unmet_service_cost_per_unit: 1,
    maximum_observation_age_steps: 0,
};

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

fn canonical_json_bytes<T: Serialize>(
    value: &T,
    field: &'static str,
) -> Result<Vec<u8>, BenchmarkError> {
    serde_json::to_vec(value)
        .map_err(|_| invalid(field, "canonical JSON serialization failed"))
}

impl PolicyScenario {
    /// Compact JSON in declared field order; the scenario has no unordered maps.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BenchmarkError> {
        canonical_json_bytes(self, "scenario")
    }

    pub fn sha256(&self) -> Result<String, BenchmarkError> {
        Ok(hash_sha256_hex(&self.canonical_bytes()?))
    }
}

impl BenchmarkConfig {
    pub fn canonical_bytes(self) -> Result<Vec<u8>, BenchmarkError> {
        canonical_json_bytes(&self, "config")
    }

    pub fn sha256(self) -> Result<String, BenchmarkError> {
        Ok(hash_sha256_hex(&self.canonical_bytes()?))
    }

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentRecord {
    pub requested_at_step: usize,
    pub active_from_step: usize,
    /// Exclusive endpoint: active when active_from <= step < active_until.
    pub active_until_step: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyRun {
    pub schema_version: u32,
    pub suite_version: u32,
    pub scenario_id: String,
    pub policy: PolicyKind,
    pub config: BenchmarkConfig,
    /// SHA-256 of the canonical scenario input and benchmark config.
    pub scenario_sha256: String,
    pub config_sha256: String,
    pub step_count: usize,
    /// Sum of abstract service units not delivered across all steps.
    pub unmet_service_units: u64,
    /// Number of steps where actual delivered service violated the hard floor.
    pub floor_breach_steps: usize,
    pub floor_breach_step_indices: Vec<usize>,
    pub longest_consecutive_floor_breach_steps: usize,
    pub first_floor_breach_step: Option<usize>,
    /// Steps from the first floor breach to the first later restored-service
    /// step. This does not prove stable recovery; later breaches remain visible.
    pub steps_to_first_restored_service: Option<usize>,
    pub deployments: Vec<DeploymentRecord>,
    pub deployment_cost_units: u64,
    pub unmet_service_cost_units: u64,
    pub total_cost_units: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

impl PolicyRun {
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BenchmarkError> {
        canonical_json_bytes(self, "run")
    }

    pub fn sha256(&self) -> Result<String, BenchmarkError> {
        Ok(hash_sha256_hex(&self.canonical_bytes()?))
    }

    pub fn receipt(&self) -> Result<PolicyRunReceipt, BenchmarkError> {
        Ok(PolicyRunReceipt {
            schema_version: POLICY_BENCHMARK_SCHEMA_VERSION,
            suite_version: LIVING_CITY_POLICY_BENCHMARK_VERSION,
            scenario_id: self.scenario_id.clone(),
            policy: self.policy,
            scenario_sha256: self.scenario_sha256.clone(),
            config_sha256: self.config_sha256.clone(),
            run_sha256: self.sha256()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyRunReceipt {
    pub schema_version: u32,
    pub suite_version: u32,
    pub scenario_id: String,
    pub policy: PolicyKind,
    pub scenario_sha256: String,
    pub config_sha256: String,
    pub run_sha256: String,
}

impl PolicyRunReceipt {
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BenchmarkError> {
        canonical_json_bytes(self, "receipt")
    }

    pub fn sha256(&self) -> Result<String, BenchmarkError> {
        Ok(hash_sha256_hex(&self.canonical_bytes()?))
    }
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
    let scenario_sha256 = scenario.sha256()?;
    let config_sha256 = config.sha256()?;

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

    let first_floor_breach_step = floor_breach_step_indices.first().copied();
    let steps_to_first_restored_service = first_floor_breach_step.and_then(|first_breach| {
        (first_breach.saturating_add(1)..scenario.steps.len())
            .find(|step_index| floor_breach_step_indices.binary_search(step_index).is_err())
            .map(|restored_step| restored_step - first_breach)
    });
    let deployment_cost_units =
        (deployments.len() as u64).saturating_mul(config.deployment_cost_units);
    let total_cost_units = deployment_cost_units.saturating_add(unmet_service_cost_units);
    Ok(PolicyRun {
        schema_version: POLICY_BENCHMARK_SCHEMA_VERSION,
        suite_version: LIVING_CITY_POLICY_BENCHMARK_VERSION,
        scenario_id: scenario.scenario_id.clone(),
        policy,
        config,
        scenario_sha256,
        config_sha256,
        step_count: scenario.steps.len(),
        unmet_service_units,
        floor_breach_steps: floor_breach_step_indices.len(),
        first_floor_breach_step,
        steps_to_first_restored_service,
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

/// Recompute the run from the exact scenario/config, then verify every digest
/// in the receipt. This protects determinism and artifact linkage; it does not
/// independently validate the physical truth of the synthetic model.
pub fn verify_policy_run_receipt(
    scenario: &PolicyScenario,
    config: BenchmarkConfig,
    run: &PolicyRun,
    receipt: &PolicyRunReceipt,
) -> Result<(), BenchmarkError> {
    config.validate()?;
    validate_scenario(scenario)?;
    if receipt.schema_version != POLICY_BENCHMARK_SCHEMA_VERSION
        || receipt.suite_version != LIVING_CITY_POLICY_BENCHMARK_VERSION
    {
        return Err(invalid("receipt.version", "unsupported receipt or suite version"));
    }
    if receipt.scenario_id != scenario.scenario_id
        || receipt.scenario_id != run.scenario_id
        || receipt.policy != run.policy
    {
        return Err(invalid("receipt.identity", "receipt identity does not match the run"));
    }

    let scenario_sha256 = scenario.sha256()?;
    let config_sha256 = config.sha256()?;
    if run.config != config
        || run.scenario_sha256 != scenario_sha256
        || run.config_sha256 != config_sha256
        || receipt.scenario_sha256 != scenario_sha256
        || receipt.config_sha256 != config_sha256
    {
        return Err(invalid("receipt.inputs", "scenario/config digest linkage failed"));
    }

    let replay = evaluate_policy(scenario, run.policy, config)?;
    if &replay != run {
        return Err(invalid("run", "recorded metrics/actions differ from deterministic replay"));
    }
    if receipt.run_sha256 != run.sha256()? {
        return Err(invalid("receipt.run_sha256", "run digest does not match canonical run"));
    }
    Ok(())
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
    if candidate.scenario_sha256 != baseline.scenario_sha256
        || candidate.config_sha256 != baseline.config_sha256
    {
        return Err(invalid(
            "paired_run",
            "paired runs must bind identical scenario and configuration digests",
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

fn synthetic_observation(
    supply: Option<u32>,
    demand: Option<u32>,
    age_steps: usize,
    forecast_service_ratio_per_mille: Option<u16>,
) -> Observation {
    Observation {
        reported_supply_units: supply,
        reported_demand_units: demand,
        age_steps,
        forecast_service_ratio_per_mille,
    }
}

fn synthetic_step(
    actual_supply_units: u32,
    actual_demand_units: u32,
    observation: Observation,
) -> ScenarioStep {
    ScenarioStep {
        actual_supply_units,
        actual_demand_units,
        observation,
    }
}

/// Four fixed deterministic cases for harness smoke/negative-control testing:
/// healthy baseline, an observed compound-time shortfall forecast, stale
/// telemetry during an outage, and a repeated outage before full recovery.
/// These are abstract unitless fixtures, not a calibrated urban model.
pub fn built_in_synthetic_smoke_scenarios() -> Vec<PolicyScenario> {
    vec![
        PolicyScenario {
            scenario_id: "synthetic-healthy-baseline".into(),
            steps: vec![
                synthetic_step(10, 10, synthetic_observation(Some(10), Some(10), 0, Some(1_000))),
                synthetic_step(10, 10, synthetic_observation(Some(10), Some(10), 0, Some(1_000))),
                synthetic_step(10, 10, synthetic_observation(Some(10), Some(10), 0, Some(1_000))),
            ],
        },
        PolicyScenario {
            scenario_id: "synthetic-forecasted-shortfall".into(),
            steps: vec![
                synthetic_step(10, 10, synthetic_observation(Some(10), Some(10), 0, Some(500))),
                synthetic_step(5, 10, synthetic_observation(Some(5), Some(10), 0, None)),
                synthetic_step(5, 10, synthetic_observation(Some(5), Some(10), 0, None)),
            ],
        },
        PolicyScenario {
            scenario_id: "synthetic-stale-telemetry-outage".into(),
            steps: vec![
                synthetic_step(5, 10, synthetic_observation(Some(20), Some(10), 3, Some(1_000))),
                synthetic_step(5, 10, synthetic_observation(Some(20), Some(10), 3, Some(1_000))),
                synthetic_step(5, 10, synthetic_observation(Some(20), Some(10), 3, Some(1_000))),
            ],
        },
        PolicyScenario {
            scenario_id: "synthetic-recurrent-outage".into(),
            steps: vec![
                synthetic_step(5, 10, synthetic_observation(Some(5), Some(10), 0, None)),
                synthetic_step(5, 10, synthetic_observation(Some(5), Some(10), 0, None)),
                synthetic_step(10, 10, synthetic_observation(Some(10), Some(10), 0, None)),
                synthetic_step(5, 10, synthetic_observation(Some(5), Some(10), 0, None)),
                synthetic_step(5, 10, synthetic_observation(Some(5), Some(10), 0, None)),
                synthetic_step(10, 10, synthetic_observation(Some(10), Some(10), 0, None)),
            ],
        },
    ]
}


/// Status of one policy on a frozen set of required synthetic scenarios.
/// Qualified means only that the policy stayed within configured floors on
/// this declared suite; it is not real-world or physical qualification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicySuiteState {
    Qualified,
    Failed,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicySuiteAssessment {
    pub schema_version: u32,
    pub suite_version: u32,
    pub policy: PolicyKind,
    pub state: PolicySuiteState,
    pub required_case_ids: Vec<String>,
    pub observed_case_ids: Vec<String>,
    pub missing_case_ids: Vec<String>,
    pub unexpected_case_ids: Vec<String>,
    pub duplicate_case_ids: Vec<String>,
    pub complete_coverage: bool,
    pub integrity_verified: bool,
    pub floor_breach_steps: Option<usize>,
    pub total_unmet_service_units: Option<u64>,
    pub worst_case_unmet_service_units: Option<u64>,
    pub total_cost_units: Option<u64>,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyCaseComparison {
    pub scenario_id: String,
    pub candidate_unmet_service_units: u64,
    pub baseline_unmet_service_units: u64,
    /// Candidate minus baseline. Negative means candidate lost less service.
    pub unmet_service_change_units: i128,
    pub candidate_floor_breach_steps: usize,
    pub baseline_floor_breach_steps: usize,
    pub candidate_run_sha256: String,
    pub baseline_run_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyRobustnessAssessment {
    pub schema_version: u32,
    pub suite_version: u32,
    pub candidate: PolicySuiteAssessment,
    pub baseline: PolicySuiteAssessment,
    pub comparison_ready: bool,
    pub matched_case_count: usize,
    pub candidate_worst_case_loss_change_units: Option<i128>,
    /// Positive is a reduction; 1000 means 100%. Undefined if baseline worst
    /// case has zero loss. This is a stress-suite metric, not a probability.
    pub worst_case_loss_reduction_per_mille: Option<i128>,
    /// Equal-weight arithmetic sum across the declared cases, not an expected
    /// value unless a separate justified weighting model is supplied.
    pub total_suite_loss_reduction_per_mille: Option<i128>,
    pub per_case: Vec<PolicyCaseComparison>,
}

/// Assess coverage, replay integrity and hard-floor performance for one policy.
///
/// Missing scenarios or run records, duplicate IDs, extra runs, wrong policy
/// identities, mismatched configs and replay/digest failures all BLOCK the
/// assessment. A fully verified case with a floor breach is FAILED, not blocked.
pub fn assess_policy_suite(
    required_case_ids: &[String],
    scenarios: &[PolicyScenario],
    policy: PolicyKind,
    runs: &[PolicyRun],
    config: BenchmarkConfig,
) -> Result<PolicySuiteAssessment, BenchmarkError> {
    config.validate()?;
    if required_case_ids.is_empty() {
        return Err(invalid("required_case_ids", "must not be empty"));
    }

    let mut required_set = BTreeSet::new();
    for id in required_case_ids {
        if id.trim().is_empty() {
            return Err(invalid("required_case_ids", "case IDs must be non-empty"));
        }
        if !required_set.insert(id.as_str()) {
            return Err(invalid("required_case_ids", "required case IDs must be unique"));
        }
    }

    let mut blockers = BTreeSet::new();
    let mut scenario_ids = BTreeSet::new();
    let mut duplicate_scenario_ids = BTreeSet::new();
    for scenario in scenarios {
        if scenario.scenario_id.trim().is_empty() || scenario.steps.is_empty() {
            blockers.insert(format!(
                "scenario {} has invalid identity or empty steps",
                scenario.scenario_id
            ));
        }
        if !scenario_ids.insert(scenario.scenario_id.as_str()) {
            duplicate_scenario_ids.insert(scenario.scenario_id.clone());
        }
    }
    for id in &duplicate_scenario_ids {
        blockers.insert(format!("duplicate scenario input: {id}"));
    }

    let missing_scenario_inputs: Vec<String> = required_case_ids
        .iter()
        .filter(|id| !scenario_ids.contains(id.as_str()))
        .cloned()
        .collect();
    let unexpected_scenario_inputs: Vec<String> = scenario_ids
        .iter()
        .filter(|id| !required_set.contains(**id))
        .map(|id| (*id).to_string())
        .collect();
    if !missing_scenario_inputs.is_empty() {
        blockers.insert(format!("missing scenario inputs: {}", missing_scenario_inputs.join(",")));
    }
    if !unexpected_scenario_inputs.is_empty() {
        blockers.insert(format!(
            "unexpected scenario inputs: {}",
            unexpected_scenario_inputs.join(",")
        ));
    }

    let mut run_ids = BTreeSet::new();
    let mut duplicate_case_ids = BTreeSet::new();
    let mut observed_case_ids = Vec::with_capacity(runs.len());
    let mut run_total_unmet = 0_u64;
    let mut run_worst_unmet = 0_u64;
    let mut run_floor_breaches = 0_usize;
    let mut run_total_cost = 0_u64;
    let mut all_run_artifacts_valid = true;

    for run in runs {
        observed_case_ids.push(run.scenario_id.clone());
        if !run_ids.insert(run.scenario_id.as_str()) {
            duplicate_case_ids.insert(run.scenario_id.clone());
        }
        if run.policy != policy {
            all_run_artifacts_valid = false;
            blockers.insert(format!(
                "run {} has policy {:?}, expected {:?}",
                run.scenario_id, run.policy, policy
            ));
        }
        if run.config != config {
            all_run_artifacts_valid = false;
            blockers.insert(format!("run {} has an unexpected benchmark config", run.scenario_id));
        }
        if !required_set.contains(run.scenario_id.as_str()) {
            blockers.insert(format!("unexpected run case: {}", run.scenario_id));
        }

        if let Some(scenario) = scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == run.scenario_id)
        {
            match run.receipt().and_then(|receipt| {
                verify_policy_run_receipt(scenario, config, run, &receipt)
            }) {
                Ok(()) => {}
                Err(error) => {
                    all_run_artifacts_valid = false;
                    blockers.insert(format!(
                        "run {} failed deterministic replay/digest verification: {}",
                        run.scenario_id, error
                    ));
                }
            }
        } else {
            all_run_artifacts_valid = false;
            blockers.insert(format!("run {} has no source scenario", run.scenario_id));
        }

        run_total_unmet = run_total_unmet.saturating_add(run.unmet_service_units);
        run_worst_unmet = run_worst_unmet.max(run.unmet_service_units);
        run_floor_breaches = run_floor_breaches.saturating_add(run.floor_breach_steps);
        run_total_cost = run_total_cost.saturating_add(run.total_cost_units);
    }

    for id in &duplicate_case_ids {
        blockers.insert(format!("duplicate run case: {id}"));
    }
    let missing_case_ids: Vec<String> = required_case_ids
        .iter()
        .filter(|id| !run_ids.contains(id.as_str()))
        .cloned()
        .collect();
    let unexpected_case_ids: Vec<String> = run_ids
        .iter()
        .filter(|id| !required_set.contains(**id))
        .map(|id| (*id).to_string())
        .collect();
    if !missing_case_ids.is_empty() {
        blockers.insert(format!("missing policy run cases: {}", missing_case_ids.join(",")));
    }
    if !unexpected_case_ids.is_empty() {
        blockers.insert(format!("unexpected policy run cases: {}", unexpected_case_ids.join(",")));
    }

    let complete_coverage = missing_case_ids.is_empty()
        && unexpected_case_ids.is_empty()
        && duplicate_case_ids.is_empty()
        && missing_scenario_inputs.is_empty()
        && unexpected_scenario_inputs.is_empty()
        && duplicate_scenario_ids.is_empty()
        && runs.len() == required_case_ids.len();
    let integrity_verified = all_run_artifacts_valid && blockers.is_empty();
    let state = if !complete_coverage || !integrity_verified {
        PolicySuiteState::Blocked
    } else if run_floor_breaches > 0 {
        PolicySuiteState::Failed
    } else {
        PolicySuiteState::Qualified
    };
    let usable_metrics = complete_coverage && integrity_verified;

    let mut observed_case_ids_sorted = observed_case_ids.clone();
    observed_case_ids_sorted.sort();
    let blockers = blockers.into_iter().collect();

    Ok(PolicySuiteAssessment {
        schema_version: POLICY_BENCHMARK_SCHEMA_VERSION,
        suite_version: LIVING_CITY_POLICY_BENCHMARK_VERSION,
        policy,
        state,
        required_case_ids: required_case_ids.to_vec(),
        observed_case_ids: observed_case_ids_sorted,
        missing_case_ids,
        unexpected_case_ids,
        duplicate_case_ids: duplicate_case_ids.into_iter().collect(),
        complete_coverage,
        integrity_verified,
        floor_breach_steps: usable_metrics.then_some(run_floor_breaches),
        total_unmet_service_units: usable_metrics.then_some(run_total_unmet),
        worst_case_unmet_service_units: usable_metrics.then_some(run_worst_unmet),
        total_cost_units: usable_metrics.then_some(run_total_cost),
        blockers,
    })
}

impl PolicyRobustnessAssessment {
    /// Canonical compact JSON bytes in declared field order. Case entries follow
    /// the caller's frozen required-case order; no unordered maps are included.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BenchmarkError> {
        canonical_json_bytes(self, "robustness_assessment")
    }

    pub fn sha256(&self) -> Result<String, BenchmarkError> {
        Ok(hash_sha256_hex(&self.canonical_bytes()?))
    }
}

/// Compare two policy suites only after every required case and run is present,
/// input receipts replay, and paired policy runs bind the exact same scenario
/// and configuration digests. This deliberately assigns no probabilities to
/// the stress cases.
pub fn assess_policy_robustness(
    required_case_ids: &[String],
    scenarios: &[PolicyScenario],
    candidate_policy: PolicyKind,
    candidate_runs: &[PolicyRun],
    baseline_policy: PolicyKind,
    baseline_runs: &[PolicyRun],
    config: BenchmarkConfig,
) -> Result<PolicyRobustnessAssessment, BenchmarkError> {
    if candidate_policy == baseline_policy {
        return Err(invalid("policy_pair", "candidate and baseline policies must differ"));
    }
    let candidate = assess_policy_suite(
        required_case_ids,
        scenarios,
        candidate_policy,
        candidate_runs,
        config,
    )?;
    let baseline = assess_policy_suite(
        required_case_ids,
        scenarios,
        baseline_policy,
        baseline_runs,
        config,
    )?;

    let comparison_ready = candidate.complete_coverage
        && baseline.complete_coverage
        && candidate.integrity_verified
        && baseline.integrity_verified;
    let mut per_case = Vec::new();
    let mut candidate_worst_case = 0_u64;
    let mut baseline_worst_case = 0_u64;

    if comparison_ready {
        for case_id in required_case_ids {
            let candidate_run = candidate_runs
                .iter()
                .find(|run| run.scenario_id == *case_id)
                .ok_or_else(|| invalid("candidate_runs", "validated candidate run disappeared"))?;
            let baseline_run = baseline_runs
                .iter()
                .find(|run| run.scenario_id == *case_id)
                .ok_or_else(|| invalid("baseline_runs", "validated baseline run disappeared"))?;
            if candidate_run.scenario_sha256 != baseline_run.scenario_sha256
                || candidate_run.config_sha256 != baseline_run.config_sha256
            {
                return Err(invalid(
                    "paired_inputs",
                    "candidate and baseline must bind identical scenario/config digests",
                ));
            }
            candidate_worst_case = candidate_worst_case.max(candidate_run.unmet_service_units);
            baseline_worst_case = baseline_worst_case.max(baseline_run.unmet_service_units);
            per_case.push(PolicyCaseComparison {
                scenario_id: case_id.clone(),
                candidate_unmet_service_units: candidate_run.unmet_service_units,
                baseline_unmet_service_units: baseline_run.unmet_service_units,
                unmet_service_change_units: candidate_run.unmet_service_units as i128
                    - baseline_run.unmet_service_units as i128,
                candidate_floor_breach_steps: candidate_run.floor_breach_steps,
                baseline_floor_breach_steps: baseline_run.floor_breach_steps,
                candidate_run_sha256: candidate_run.sha256()?,
                baseline_run_sha256: baseline_run.sha256()?,
            });
        }
    }

    let worst_case_loss_reduction_per_mille = if comparison_ready && baseline_worst_case > 0 {
        Some(
            ((baseline_worst_case as i128 - candidate_worst_case as i128)
                * (PER_MILLE as i128))
                / baseline_worst_case as i128,
        )
    } else {
        None
    };
    let total_suite_loss_reduction_per_mille = match (
        comparison_ready,
        candidate.total_unmet_service_units,
        baseline.total_unmet_service_units,
    ) {
        (true, Some(candidate_total), Some(baseline_total)) if baseline_total > 0 => Some(
            ((baseline_total as i128 - candidate_total as i128) * (PER_MILLE as i128))
                / baseline_total as i128,
        ),
        _ => None,
    };

    Ok(PolicyRobustnessAssessment {
        schema_version: POLICY_BENCHMARK_SCHEMA_VERSION,
        suite_version: LIVING_CITY_POLICY_BENCHMARK_VERSION,
        candidate,
        baseline,
        comparison_ready,
        matched_case_count: per_case.len(),
        candidate_worst_case_loss_change_units: comparison_ready.then_some(
            candidate_worst_case as i128 - baseline_worst_case as i128,
        ),
        worst_case_loss_reduction_per_mille,
        total_suite_loss_reduction_per_mille,
        per_case,
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
    fn built_in_smoke_suite_is_paired_deterministic_and_receipt_verifiable() {
        let scenarios = built_in_synthetic_smoke_scenarios();
        let first = evaluate_suite(
            &scenarios,
            &STANDARD_BASELINE_POLICIES,
            SYNTHETIC_SMOKE_CONFIG,
        )
        .unwrap();
        let second = evaluate_suite(
            &scenarios,
            &STANDARD_BASELINE_POLICIES,
            SYNTHETIC_SMOKE_CONFIG,
        )
        .unwrap();

        assert_eq!(scenarios.len(), 4);
        assert_eq!(first.len(), 16);
        assert_eq!(first, second);

        for run in &first {
            let scenario = scenarios
                .iter()
                .find(|scenario| scenario.scenario_id == run.scenario_id)
                .unwrap();
            let receipt = run.receipt().unwrap();
            assert!(verify_policy_run_receipt(
                scenario,
                SYNTHETIC_SMOKE_CONFIG,
                run,
                &receipt,
            )
            .is_ok());
        }

        let healthy = first
            .iter()
            .filter(|run| run.scenario_id == "synthetic-healthy-baseline");
        assert!(healthy.all(|run| run.unmet_service_units == 0));
    }

    #[test]
    fn floor_first_forecast_reduces_service_loss_vs_reactive_on_frozen_fixture() {
        let scenarios = built_in_synthetic_smoke_scenarios();
        let case = scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "synthetic-forecasted-shortfall")
            .unwrap();
        let floor_first = evaluate_policy(
            case,
            PolicyKind::FloorFirst,
            SYNTHETIC_SMOKE_CONFIG,
        )
        .unwrap();
        let reactive = evaluate_policy(
            case,
            PolicyKind::ReactiveGreedy,
            SYNTHETIC_SMOKE_CONFIG,
        )
        .unwrap();

        assert_eq!(floor_first.unmet_service_units, 0);
        assert!(reactive.unmet_service_units > floor_first.unmet_service_units);
        let comparison = compare_service_loss(&floor_first, &reactive).unwrap();
        assert_eq!(comparison.service_loss_reduction_per_mille, Some(1_000));
    }

    #[test]
    fn robustness_assessment_requires_complete_replay_verified_paired_coverage() {
        let scenarios = vec![scenario(
            "forecast-case",
            vec![
                step(10, 10, Observation {
                    reported_supply_units: Some(10),
                    reported_demand_units: Some(10),
                    age_steps: 0,
                    forecast_service_ratio_per_mille: Some(500),
                }),
                step(5, 10, observed(5, 10)),
                step(5, 10, observed(5, 10)),
            ],
        )];
        let required = vec!["forecast-case".to_string()];
        let mut cfg = config();
        cfg.deployment_delay_steps = 1;
        cfg.backup_duration_steps = 2;
        let candidate =
            evaluate_suite(&scenarios, &[PolicyKind::FloorFirst], cfg).unwrap();
        let baseline =
            evaluate_suite(&scenarios, &[PolicyKind::ReactiveGreedy], cfg).unwrap();
        let assessment = assess_policy_robustness(
            &required,
            &scenarios,
            PolicyKind::FloorFirst,
            &candidate,
            PolicyKind::ReactiveGreedy,
            &baseline,
            cfg,
        )
        .unwrap();

        assert!(assessment.comparison_ready);
        assert_eq!(assessment.matched_case_count, 1);
        assert_eq!(assessment.candidate.state, PolicySuiteState::Qualified);
        assert_eq!(assessment.baseline.state, PolicySuiteState::Failed);
        assert_eq!(assessment.candidate.total_unmet_service_units, Some(0));
        assert_eq!(assessment.baseline.total_unmet_service_units, Some(5));
        assert_eq!(assessment.worst_case_loss_reduction_per_mille, Some(1_000));
        assert_eq!(assessment.total_suite_loss_reduction_per_mille, Some(1_000));
        assert_eq!(assessment.per_case[0].candidate_run_sha256, candidate[0].sha256().unwrap());
        assert_eq!(assessment.per_case[0].baseline_run_sha256, baseline[0].sha256().unwrap());
        let canonical = assessment.canonical_bytes().unwrap();
        let decoded = serde_json::from_slice::<PolicyRobustnessAssessment>(&canonical).unwrap();
        assert_eq!(decoded, assessment);
        assert_eq!(assessment.sha256().unwrap(), decoded.sha256().unwrap());
    }

    #[test]
    fn missing_required_run_blocks_and_withholds_all_aggregate_metrics() {
        let scenarios = vec![
            scenario("case-a", vec![step(10, 10, observed(10, 10))]),
            scenario("case-b", vec![step(5, 10, observed(5, 10))]),
        ];
        let required = vec!["case-a".to_string(), "case-b".to_string()];
        let candidate = evaluate_suite(
            &scenarios[..1],
            &[PolicyKind::FloorFirst],
            config(),
        )
        .unwrap();
        let baseline = evaluate_suite(
            &scenarios,
            &[PolicyKind::ReactiveGreedy],
            config(),
        )
        .unwrap();
        let assessment = assess_policy_robustness(
            &required,
            &scenarios,
            PolicyKind::FloorFirst,
            &candidate,
            PolicyKind::ReactiveGreedy,
            &baseline,
            config(),
        )
        .unwrap();

        assert_eq!(assessment.candidate.state, PolicySuiteState::Blocked);
        assert!(!assessment.candidate.complete_coverage);
        assert_eq!(assessment.candidate.missing_case_ids, vec!["case-b"]);
        assert_eq!(assessment.candidate.total_unmet_service_units, None);
        assert!(!assessment.comparison_ready);
        assert!(assessment.per_case.is_empty());
    }

    #[test]
    fn duplicate_extra_or_wrong_policy_run_blocks_suite() {
        let case = scenario("one-case", vec![step(10, 10, observed(10, 10))]);
        let required = vec!["one-case".to_string()];
        let good = evaluate_suite(
            std::slice::from_ref(&case),
            &[PolicyKind::FloorFirst],
            config(),
        )
        .unwrap();
        let extra_scenarios = vec![case.clone(), extra_case_fixture()];
        let extra = evaluate_suite(
            &extra_scenarios,
            &[PolicyKind::FloorFirst],
            config(),
        )
        .unwrap();

        let duplicate_result = assess_policy_suite(
            &required,
            std::slice::from_ref(&case),
            PolicyKind::FloorFirst,
            &[good[0].clone(), good[0].clone()],
            config(),
        )
        .unwrap();
        assert_eq!(duplicate_result.state, PolicySuiteState::Blocked);
        assert_eq!(duplicate_result.duplicate_case_ids, vec!["one-case"]);

        let extra_result = assess_policy_suite(
            &required,
            &extra_scenarios,
            PolicyKind::FloorFirst,
            &extra,
            config(),
        )
        .unwrap();
        assert_eq!(extra_result.state, PolicySuiteState::Blocked);
        assert_eq!(extra_result.unexpected_case_ids, vec!["extra-case"]);

        let wrong_policy = assess_policy_suite(
            &required,
            std::slice::from_ref(&case),
            PolicyKind::ReactiveGreedy,
            &good,
            config(),
        )
        .unwrap();
        assert_eq!(wrong_policy.state, PolicySuiteState::Blocked);
    }

    fn extra_case_fixture() -> PolicyScenario {
        scenario("extra-case", vec![step(10, 10, observed(10, 10))])
    }

    #[test]
    fn a_valid_floor_breach_fails_policy_instead_of_becoming_a_blocker() {
        let case = scenario("unsafe-case", vec![step(5, 10, observed(5, 10))]);
        let required = vec!["unsafe-case".to_string()];
        let runs = evaluate_suite(
            std::slice::from_ref(&case),
            &[PolicyKind::NoIntervention],
            config(),
        )
        .unwrap();
        let assessment = assess_policy_suite(
            &required,
            std::slice::from_ref(&case),
            PolicyKind::NoIntervention,
            &runs,
            config(),
        )
        .unwrap();

        assert_eq!(assessment.state, PolicySuiteState::Failed);
        assert!(assessment.integrity_verified);
        assert_eq!(assessment.floor_breach_steps, Some(1));
    }

    #[test]
    fn suite_assessor_blocks_altered_result_even_if_run_id_and_config_match() {
        let case = scenario("tampered-result", vec![step(5, 10, observed(5, 10))]);
        let required = vec!["tampered-result".to_string()];
        let mut runs = evaluate_suite(
            std::slice::from_ref(&case),
            &[PolicyKind::ReactiveGreedy],
            config(),
        )
        .unwrap();
        runs[0].unmet_service_units = 999;
        let assessment = assess_policy_suite(
            &required,
            std::slice::from_ref(&case),
            PolicyKind::ReactiveGreedy,
            &runs,
            config(),
        )
        .unwrap();

        assert_eq!(assessment.state, PolicySuiteState::Blocked);
        assert!(!assessment.integrity_verified);
        assert_eq!(assessment.total_unmet_service_units, None);
    }

    #[test]
    fn paired_run_comparison_rejects_same_id_with_different_hidden_truth() {
        let candidate_case = scenario("same-id", vec![step(5, 10, observed(5, 10))]);
        let baseline_case = scenario("same-id", vec![step(4, 10, observed(5, 10))]);
        let candidate =
            evaluate_policy(&candidate_case, PolicyKind::FloorFirst, config()).unwrap();
        let baseline =
            evaluate_policy(&baseline_case, PolicyKind::ReactiveGreedy, config()).unwrap();
        assert_eq!(
            compare_service_loss(&candidate, &baseline).unwrap_err().field,
            "paired_run"
        );
    }

    #[test]
    fn zero_baseline_loss_does_not_invent_a_percentage_improvement() {
        let case = scenario("zero-loss", vec![step(10, 10, observed(10, 10))]);
        let required = vec!["zero-loss".to_string()];
        let candidate = evaluate_suite(
            std::slice::from_ref(&case),
            &[PolicyKind::FloorFirst],
            config(),
        )
        .unwrap();
        let baseline = evaluate_suite(
            std::slice::from_ref(&case),
            &[PolicyKind::ReactiveGreedy],
            config(),
        )
        .unwrap();
        let assessment = assess_policy_robustness(
            &required,
            std::slice::from_ref(&case),
            PolicyKind::FloorFirst,
            &candidate,
            PolicyKind::ReactiveGreedy,
            &baseline,
            config(),
        )
        .unwrap();
        assert_eq!(assessment.worst_case_loss_reduction_per_mille, None);
        assert_eq!(assessment.total_suite_loss_reduction_per_mille, None);
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
    fn first_restoration_delay_is_reported_even_if_a_later_breach_occurs() {
        let case = scenario(
            "restoration-not-stability",
            vec![
                step(5, 10, observed(5, 10)),
                step(10, 10, observed(10, 10)),
                step(5, 10, observed(5, 10)),
            ],
        );
        let run = evaluate_policy(&case, PolicyKind::NoIntervention, config()).unwrap();

        assert_eq!(run.first_floor_breach_step, Some(0));
        assert_eq!(run.floor_breach_step_indices, vec![0, 2]);
        assert_eq!(run.steps_to_first_restored_service, Some(1));
    }

    #[test]
    fn unrecovered_floor_breach_has_no_restoration_delay() {
        let case = scenario(
            "persistent-shortfall",
            vec![
                step(5, 10, observed(5, 10)),
                step(5, 10, observed(5, 10)),
            ],
        );
        let run = evaluate_policy(&case, PolicyKind::NoIntervention, config()).unwrap();

        assert_eq!(run.first_floor_breach_step, Some(0));
        assert_eq!(run.steps_to_first_restored_service, None);
        assert_eq!(run.longest_consecutive_floor_breach_steps, 2);
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
    fn receipt_binds_scenario_config_and_replayed_run() {
        let case = scenario(
            "receipt-fixture",
            vec![
                step(5, 10, observed(5, 10)),
                step(10, 10, observed(10, 10)),
            ],
        );
        let cfg = config();
        let run = evaluate_policy(&case, PolicyKind::ReactiveGreedy, cfg).unwrap();
        let receipt = run.receipt().unwrap();

        assert!(verify_policy_run_receipt(&case, cfg, &run, &receipt).is_ok());
        assert_eq!(
            receipt.scenario_sha256,
            case.sha256().unwrap()
        );
        assert_eq!(receipt.config_sha256, cfg.sha256().unwrap());
        assert_eq!(receipt.run_sha256, run.sha256().unwrap());
        let receipt_bytes = receipt.canonical_bytes().unwrap();
        let decoded = serde_json::from_slice::<PolicyRunReceipt>(&receipt_bytes).unwrap();
        assert_eq!(decoded, receipt);
    }

    #[test]
    fn receipt_verifier_rejects_tampered_run_metrics_and_stale_digest() {
        let case = scenario("tamper-fixture", vec![step(5, 10, observed(5, 10))]);
        let cfg = config();
        let run = evaluate_policy(&case, PolicyKind::ReactiveGreedy, cfg).unwrap();
        let receipt = run.receipt().unwrap();

        let mut tampered_run = run.clone();
        tampered_run.unmet_service_units = tampered_run.unmet_service_units.saturating_add(1);
        assert_eq!(
            verify_policy_run_receipt(&case, cfg, &tampered_run, &receipt)
                .unwrap_err()
                .field,
            "run"
        );

        let mut tampered_receipt = receipt;
        tampered_receipt.run_sha256 = "0".repeat(64);
        assert_eq!(
            verify_policy_run_receipt(&case, cfg, &run, &tampered_receipt)
                .unwrap_err()
                .field,
            "receipt.run_sha256"
        );
    }

    #[test]
    fn receipt_verifier_rejects_same_id_with_changed_scenario_or_config() {
        let case = scenario("bound-inputs", vec![step(5, 10, observed(5, 10))]);
        let cfg = config();
        let run = evaluate_policy(&case, PolicyKind::ReactiveGreedy, cfg).unwrap();
        let receipt = run.receipt().unwrap();

        let mut altered_case = case.clone();
        altered_case.steps[0].actual_supply_units = 4;
        assert_eq!(
            verify_policy_run_receipt(&altered_case, cfg, &run, &receipt)
                .unwrap_err()
                .field,
            "receipt.inputs"
        );

        let mut altered_config = cfg;
        altered_config.backup_capacity_units += 1;
        assert_eq!(
            verify_policy_run_receipt(&case, altered_config, &run, &receipt)
                .unwrap_err()
                .field,
            "receipt.inputs"
        );
    }

    #[test]
    fn scenario_and_config_digests_change_when_bound_inputs_change() {
        let case = scenario("digest-fixture", vec![step(5, 10, observed(5, 10))]);
        let original_scenario_digest = case.sha256().unwrap();
        let mut changed_case = case.clone();
        changed_case.steps[0].actual_supply_units = 4;
        assert_ne!(original_scenario_digest, changed_case.sha256().unwrap());

        let cfg = config();
        let mut changed_config = cfg;
        changed_config.backup_capacity_units += 1;
        assert_ne!(cfg.sha256().unwrap(), changed_config.sha256().unwrap());
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
