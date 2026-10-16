// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic, synthetic experiment-selection benchmark for marine freight.
//!
//! The policy receives only interval bounds and candidate experiment metadata.
//! The simulator retains synthetic truth to generate idealized nested intervals.
//! This is a policy/software benchmark, not a measurement model or physical result.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

pub const EXPERIMENT_CATALOG_VERSION: &str = "marine-experiment-catalog-v1";
pub const SCENARIO_GENERATOR_VERSION: &str = "marine-synthetic-scenarios-v1";
pub const TOY_ORACLE_VERSION: &str = "exhaustive-subset-oracle-v1";
pub const BENCHMARK_SCHEMA_VERSION: &str = "1.0.0";
pub const DEFAULT_SCENARIOS: usize = 1_000;
pub const DEFAULT_PAIRED_SEEDS: usize = 30;
pub const DEFAULT_BUDGET_UNITS: u32 = 6;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

impl Interval {
    pub fn width(self) -> f64 {
        self.upper - self.lower
    }

    fn contains(self, value: f64) -> bool {
        self.lower <= value && value <= self.upper
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Parameter {
    PropulsionEnergyDemand,
    UsableBatteryEnergy,
    RequiredWaterDepth,
    AvailableWaterDepth,
    RequiredChargePower,
    AvailableChargePower,
}

impl Parameter {
    pub const ALL: [Self; 6] = [
        Self::PropulsionEnergyDemand,
        Self::UsableBatteryEnergy,
        Self::RequiredWaterDepth,
        Self::AvailableWaterDepth,
        Self::RequiredChargePower,
        Self::AvailableChargePower,
    ];

    pub fn index(self) -> usize {
        match self {
            Self::PropulsionEnergyDemand => 0,
            Self::UsableBatteryEnergy => 1,
            Self::RequiredWaterDepth => 2,
            Self::AvailableWaterDepth => 3,
            Self::RequiredChargePower => 4,
            Self::AvailableChargePower => 5,
        }
    }

    fn scale(self) -> f64 {
        match self {
            Self::PropulsionEnergyDemand | Self::UsableBatteryEnergy => 5_000.0,
            Self::RequiredWaterDepth | Self::AvailableWaterDepth => 5.0,
            Self::RequiredChargePower | Self::AvailableChargePower => 1_000.0,
        }
    }

    fn constraint_index(self) -> usize {
        match self {
            Self::PropulsionEnergyDemand | Self::UsableBatteryEnergy => 0,
            Self::RequiredWaterDepth | Self::AvailableWaterDepth => 1,
            Self::RequiredChargePower | Self::AvailableChargePower => 2,
        }
    }

    fn constraint_weight(self) -> f64 {
        match self.constraint_index() {
            0 => 4.0,
            1 => 5.0,
            _ => 3.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Experiment {
    pub id: String,
    pub target: Parameter,
    pub cost_units: u32,
    /// Fraction of the current interval width retained after the simulated test.
    /// This is a synthetic oracle setting, not an empirical test characteristic.
    pub retained_width_fraction: f64,
}

pub fn experiment_catalog() -> Vec<Experiment> {
    vec![
        Experiment {
            id: "propulsion_log_review".to_owned(),
            target: Parameter::PropulsionEnergyDemand,
            cost_units: 1,
            retained_width_fraction: 0.65,
        },
        Experiment {
            id: "instrumented_propulsion_trial".to_owned(),
            target: Parameter::PropulsionEnergyDemand,
            cost_units: 3,
            retained_width_fraction: 0.25,
        },
        Experiment {
            id: "battery_data_audit".to_owned(),
            target: Parameter::UsableBatteryEnergy,
            cost_units: 1,
            retained_width_fraction: 0.60,
        },
        Experiment {
            id: "battery_capacity_test".to_owned(),
            target: Parameter::UsableBatteryEnergy,
            cost_units: 2,
            retained_width_fraction: 0.30,
        },
        Experiment {
            id: "waterway_chart_review".to_owned(),
            target: Parameter::RequiredWaterDepth,
            cost_units: 1,
            retained_width_fraction: 0.60,
        },
        Experiment {
            id: "bathymetric_survey".to_owned(),
            target: Parameter::RequiredWaterDepth,
            cost_units: 3,
            retained_width_fraction: 0.25,
        },
        Experiment {
            id: "available_depth_reference_audit".to_owned(),
            target: Parameter::AvailableWaterDepth,
            cost_units: 1,
            retained_width_fraction: 0.60,
        },
        Experiment {
            id: "route_depth_survey".to_owned(),
            target: Parameter::AvailableWaterDepth,
            cost_units: 2,
            retained_width_fraction: 0.30,
        },
        Experiment {
            id: "charge_schedule_audit".to_owned(),
            target: Parameter::RequiredChargePower,
            cost_units: 1,
            retained_width_fraction: 0.60,
        },
        Experiment {
            id: "instrumented_charge_cycle".to_owned(),
            target: Parameter::RequiredChargePower,
            cost_units: 2,
            retained_width_fraction: 0.30,
        },
        Experiment {
            id: "shore_power_availability_audit".to_owned(),
            target: Parameter::AvailableChargePower,
            cost_units: 1,
            retained_width_fraction: 0.60,
        },
        Experiment {
            id: "terminal_load_bank_test".to_owned(),
            target: Parameter::AvailableChargePower,
            cost_units: 2,
            retained_width_fraction: 0.30,
        },
    ]
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintStatus {
    RobustFeasible,
    RobustInfeasible,
    Indeterminate,
}

impl ConstraintStatus {
    fn is_resolved(self) -> bool {
        self != Self::Indeterminate
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConstraintDisposition {
    pub id: String,
    pub status: ConstraintStatus,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioRun {
    pub scenario_id: u64,
    pub constraints: Vec<ConstraintDisposition>,
    pub resolved_constraints: u8,
    pub robust_feasible_constraints: u8,
    pub robust_infeasible_constraints: u8,
    pub fully_resolved: bool,
    pub budget_spent_units: u32,
    pub cost_to_full_resolution_units: Option<u32>,
    pub normalized_interval_width_reduction: f64,
    pub oracle_regret_constraints: u8,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    CheapestFirst,
    RoundRobin,
    SeededRandom,
    ConstraintCriticality,
    ExhaustiveToyOracle,
}

impl Policy {
    fn all_heuristics() -> [Self; 4] {
        [
            Self::CheapestFirst,
            Self::RoundRobin,
            Self::SeededRandom,
            Self::ConstraintCriticality,
        ]
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PolicySummary {
    pub policy: Policy,
    /// None for the exhaustive toy oracle because its result is deterministic per scenario
    /// and uses hidden simulator truth as an explicitly non-deployable comparison bound.
    pub seed: Option<u64>,
    pub evaluated_scenarios: u64,
    pub mean_resolved_constraints: f64,
    pub mean_robust_feasible_constraints: f64,
    pub mean_robust_infeasible_constraints: f64,
    pub full_resolution_fraction: f64,
    pub mean_budget_spent_units: f64,
    pub mean_cost_to_full_resolution_units: Option<f64>,
    pub mean_normalized_interval_width_reduction: f64,
    pub mean_oracle_regret_constraints: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkBody {
    pub schema_version: String,
    pub suite_id: String,
    pub experiment_catalog_version: String,
    pub scenario_generator_version: String,
    pub toy_oracle_version: String,
    pub scenario_count: usize,
    pub paired_seeds: Vec<u64>,
    pub budget_units_per_scenario: u32,
    pub experiment_catalog: Vec<Experiment>,
    pub summaries: Vec<PolicySummary>,
    pub limitations: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkReceipt {
    pub body: BenchmarkBody,
    pub body_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BenchmarkError {
    EmptyScenarioSet,
    EmptySeeds,
    DuplicateSeed(u64),
    Serialization(String),
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyScenarioSet => write!(f, "scenario_count must be greater than zero"),
            Self::EmptySeeds => write!(f, "at least one paired seed is required"),
            Self::DuplicateSeed(seed) => write!(f, "paired seed is duplicated: {seed}"),
            Self::Serialization(message) => {
                write!(f, "could not serialize benchmark receipt: {message}")
            }
        }
    }
}

impl std::error::Error for BenchmarkError {}

#[derive(Clone, Copy, Debug)]
struct SyntheticScenario {
    id: u64,
    truth: [f64; 6],
    initial: [Interval; 6],
}

#[derive(Clone, Copy, Debug)]
struct ObservedState {
    intervals: [Interval; 6],
    initial: [Interval; 6],
}

#[derive(Clone, Copy, Debug)]
struct RunMetrics {
    scenario_id: u64,
    statuses: [ConstraintStatus; 3],
    resolved_constraints: u8,
    robust_feasible_constraints: u8,
    robust_infeasible_constraints: u8,
    fully_resolved: bool,
    budget_spent_units: u32,
    cost_to_full_resolution_units: Option<u32>,
    normalized_interval_width_reduction: f64,
}

impl RunMetrics {
    fn to_public(self, oracle_resolved: u8) -> ScenarioRun {
        ScenarioRun {
            scenario_id: self.scenario_id,
            constraints: vec![
                ConstraintDisposition {
                    id: "battery_energy_sufficiency".to_owned(),
                    status: self.statuses[0],
                },
                ConstraintDisposition {
                    id: "under_keel_clearance".to_owned(),
                    status: self.statuses[1],
                },
                ConstraintDisposition {
                    id: "charge_power_sufficiency".to_owned(),
                    status: self.statuses[2],
                },
            ],
            resolved_constraints: self.resolved_constraints,
            robust_feasible_constraints: self.robust_feasible_constraints,
            robust_infeasible_constraints: self.robust_infeasible_constraints,
            fully_resolved: self.fully_resolved,
            budget_spent_units: self.budget_spent_units,
            cost_to_full_resolution_units: self.cost_to_full_resolution_units,
            normalized_interval_width_reduction: self.normalized_interval_width_reduction,
            oracle_regret_constraints: oracle_resolved.saturating_sub(self.resolved_constraints),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64)
    }
}

fn interval_around_truth(truth: f64, half_width: f64) -> Interval {
    Interval {
        lower: (truth - half_width).max(0.0),
        upper: truth + half_width,
    }
}

fn generate_scenario(id: u64) -> SyntheticScenario {
    let mut rng = SplitMix64::new(0x4D4152494E455F31_u64 ^ id.wrapping_mul(0x9E3779B97F4A7C15));
    let truth = [
        2_200.0 + rng.unit() * 1_800.0,
        2_600.0 + rng.unit() * 1_900.0,
        2.30 + rng.unit() * 1.10,
        2.40 + rng.unit() * 1.20,
        300.0 + rng.unit() * 600.0,
        350.0 + rng.unit() * 650.0,
    ];
    let half_widths = [
        500.0 + rng.unit() * 500.0,
        450.0 + rng.unit() * 450.0,
        0.20 + rng.unit() * 0.30,
        0.20 + rng.unit() * 0.30,
        100.0 + rng.unit() * 180.0,
        100.0 + rng.unit() * 180.0,
    ];
    let initial: [Interval; 6] = std::array::from_fn(|index| {
        interval_around_truth(truth[index], half_widths[index])
    });
    SyntheticScenario { id, truth, initial }
}

fn initial_state(scenario: &SyntheticScenario) -> ObservedState {
    ObservedState {
        intervals: scenario.initial,
        initial: scenario.initial,
    }
}

fn classify_pair(demand: Interval, capacity: Interval) -> ConstraintStatus {
    if demand.upper <= capacity.lower {
        ConstraintStatus::RobustFeasible
    } else if demand.lower > capacity.upper {
        ConstraintStatus::RobustInfeasible
    } else {
        ConstraintStatus::Indeterminate
    }
}

fn statuses(state: &ObservedState) -> [ConstraintStatus; 3] {
    [
        classify_pair(
            state.intervals[Parameter::PropulsionEnergyDemand.index()],
            state.intervals[Parameter::UsableBatteryEnergy.index()],
        ),
        classify_pair(
            state.intervals[Parameter::RequiredWaterDepth.index()],
            state.intervals[Parameter::AvailableWaterDepth.index()],
        ),
        classify_pair(
            state.intervals[Parameter::RequiredChargePower.index()],
            state.intervals[Parameter::AvailableChargePower.index()],
        ),
    ]
}

fn total_normalized_width(state: &ObservedState) -> f64 {
    Parameter::ALL
        .iter()
        .map(|parameter| {
            state.intervals[parameter.index()].width() / parameter.scale()
        })
        .sum()
}

fn width_reduction(initial: &ObservedState, current: &ObservedState) -> f64 {
    let initial_width = total_normalized_width(initial);
    if initial_width <= 0.0 {
        return 0.0;
    }
    (1.0 - total_normalized_width(current) / initial_width).clamp(0.0, 1.0)
}

fn summarize_state(scenario_id: u64, initial: &ObservedState, current: &ObservedState,
                   spent: u32, cost_to_full_resolution_units: Option<u32>) -> RunMetrics {
    let disposition = statuses(current);
    let feasible = disposition
        .iter()
        .filter(|s| **s == ConstraintStatus::RobustFeasible)
        .count() as u8;
    let infeasible = disposition
        .iter()
        .filter(|s| **s == ConstraintStatus::RobustInfeasible)
        .count() as u8;
    let resolved = feasible + infeasible;
    RunMetrics {
        scenario_id,
        statuses: disposition,
        resolved_constraints: resolved,
        robust_feasible_constraints: feasible,
        robust_infeasible_constraints: infeasible,
        fully_resolved: resolved == 3,
        budget_spent_units: spent,
        cost_to_full_resolution_units,
        normalized_interval_width_reduction: width_reduction(initial, current),
    }
}

fn simulate_experiment(
    scenario: &SyntheticScenario,
    state: &mut ObservedState,
    experiment: &Experiment,
) {
    let index = experiment.target.index();
    let current = state.intervals[index];
    let truth = scenario.truth[index];
    debug_assert!(current.contains(truth));
    let target_width = current.width() * experiment.retained_width_fraction;
    let narrowed = Interval {
        lower: (truth - target_width / 2.0).max(current.lower),
        upper: (truth + target_width / 2.0).min(current.upper),
    };
    debug_assert!(narrowed.contains(truth));
    debug_assert!(narrowed.width() <= current.width());
    state.intervals[index] = narrowed;
}

fn is_fully_resolved(state: &ObservedState) -> bool {
    statuses(state).iter().all(|status| status.is_resolved())
}

fn criticality_score(experiment: &Experiment, state: &ObservedState) -> f64 {
    let constraint_status = statuses(state)[experiment.target.constraint_index()];
    if constraint_status != ConstraintStatus::Indeterminate {
        return 0.0;
    }
    let width = state.intervals[experiment.target.index()].width();
    let expected_fraction_reduced = 1.0 - experiment.retained_width_fraction;
    experiment.target.constraint_weight() * (width / experiment.target.scale())
        * expected_fraction_reduced / experiment.cost_units as f64
}

fn choose_experiment(
    policy: Policy,
    state: &ObservedState,
    available: &[bool],
    spent: u32,
    budget: u32,
    selection_count: usize,
    rng: &mut SplitMix64,
    catalog: &[Experiment],
) -> Option<usize> {
    let affordable: Vec<usize> = catalog.iter().enumerate()
        .filter(|(index, item)| {
            available[*index] && item.cost_units <= budget.saturating_sub(spent)
        })
        .map(|(index, _)| index)
        .collect();
    if affordable.is_empty() {
        return None;
    }

    match policy {
        Policy::CheapestFirst => affordable
            .into_iter()
            .min_by_key(|index| (catalog[*index].cost_units, *index)),
        Policy::RoundRobin => {
            let desired = Parameter::ALL[selection_count % Parameter::ALL.len()];
            affordable.iter()
                .copied()
                .filter(|index| catalog[*index].target == desired)
                .min_by_key(|index| (catalog[*index].cost_units, *index))
                .or_else(|| {
                    affordable
                        .iter()
                        .copied()
                        .min_by_key(|index| (catalog[*index].cost_units, *index))
                })
        }
        Policy::SeededRandom => Some(affordable[(rng.next_u64() as usize) % affordable.len()]),
        Policy::ConstraintCriticality => {
            let mut best: Option<(usize, f64)> = None;
            for index in affordable.iter().copied() {
                let score = criticality_score(&catalog[index], state);
                match best {
                    None => best = Some((index, score)),
                    Some((best_index, best_score))
                        if score > best_score
                            || (score == best_score
                                && (catalog[index].cost_units, index)
                                    < (catalog[best_index].cost_units, best_index)) =>
                    {
                        best = Some((index, score));
                    }
                    _ => {}
                }
            }
            best.map(|(index, _)| index)
        }
        Policy::ExhaustiveToyOracle => None,
    }
}

fn run_policy(
    scenario: &SyntheticScenario,
    policy: Policy,
    seed: u64,
    budget: u32,
    catalog: &[Experiment],
) -> RunMetrics {
    let initial = initial_state(scenario);
    let mut current = initial;
    let mut available = vec![true; catalog.len()];
    let mut spent = 0_u32;
    let mut selected = 0_usize;
    let mut rng = SplitMix64::new(seed ^ scenario.id.wrapping_mul(0xD6E8FEB86659FD93));
    let mut cost_to_resolution = if is_fully_resolved(&current) { Some(0) } else { None };

    loop {
        if is_fully_resolved(&current) {
            break;
        }
        let Some(index) = choose_experiment(
            policy, &current, &available, spent, budget, selected, &mut rng, catalog,
        ) else { break };
        let experiment = &catalog[index];
        simulate_experiment(scenario, &mut current, experiment);
        available[index] = false;
        spent += experiment.cost_units;
        selected += 1;
        if cost_to_resolution.is_none() && is_fully_resolved(&current) {
            cost_to_resolution = Some(spent);
        }
    }

    summarize_state(scenario.id, &initial, &current, spent, cost_to_resolution)
}

fn better_oracle_candidate(candidate: &RunMetrics, candidate_cost: u32,
                           best: &RunMetrics, best_cost: u32) -> bool {
    if candidate.resolved_constraints != best.resolved_constraints {
        return candidate.resolved_constraints > best.resolved_constraints;
    }
    if (candidate.normalized_interval_width_reduction
        - best.normalized_interval_width_reduction)
        .abs()
        > 1e-12
    {
        return candidate.normalized_interval_width_reduction
            > best.normalized_interval_width_reduction;
    }
    candidate_cost < best_cost
}

/// Enumerates every experiment subset within the declared budget. It uses the
/// simulator's hidden truth to model the best outcome of that subset, so this is
/// a clairvoyant upper bound for this toy measurement model, not a deployable policy.
fn exhaustive_toy_oracle(
    scenario: &SyntheticScenario,
    budget: u32,
    catalog: &[Experiment],
) -> RunMetrics {
    assert!(catalog.len() < usize::BITS as usize);
    let initial = initial_state(scenario);
    let mut best = summarize_state(scenario.id, &initial, &initial, 0,
        if is_fully_resolved(&initial) { Some(0) } else { None });
    let mut best_cost = 0_u32;
    let subset_count = 1_usize << catalog.len();

    for mask in 1..subset_count {
        let mut cost = 0_u32;
        let mut valid = true;
        for (index, experiment) in catalog.iter().enumerate() {
            if mask & (1 << index) != 0 {
                cost = cost.saturating_add(experiment.cost_units);
                if cost > budget {
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            continue;
        }

        let mut current = initial;
        let mut cumulative_cost = 0_u32;
        let mut cost_to_resolution = if is_fully_resolved(&current) { Some(0) } else { None };
        for (index, experiment) in catalog.iter().enumerate() {
            if mask & (1 << index) != 0 {
                simulate_experiment(scenario, &mut current, experiment);
                cumulative_cost += experiment.cost_units;
                if cost_to_resolution.is_none() && is_fully_resolved(&current) {
                    cost_to_resolution = Some(cumulative_cost);
                }
            }
        }
        let candidate = summarize_state(
            scenario.id,
            &initial,
            &current,
            cost,
            cost_to_resolution,
        );
        if better_oracle_candidate(&candidate, cost, &best, best_cost) {
            best = candidate;
            best_cost = cost;
        }
    }
    best
}

fn summarize_policy(
    policy: Policy,
    seed: Option<u64>,
    results: &[RunMetrics],
    oracle_results: &[RunMetrics],
) -> PolicySummary {
    let n = results.len() as f64;
    let mut resolved = 0.0;
    let mut feasible = 0.0;
    let mut infeasible = 0.0;
    let mut fully = 0.0;
    let mut spent = 0.0;
    let mut width_reduction = 0.0;
    let mut regret = 0.0;
    let mut resolution_cost_sum = 0.0;
    let mut resolution_cost_count = 0.0;

    for (index, result) in results.iter().enumerate() {
        resolved += result.resolved_constraints as f64;
        feasible += result.robust_feasible_constraints as f64;
        infeasible += result.robust_infeasible_constraints as f64;
        fully += if result.fully_resolved { 1.0 } else { 0.0 };
        spent += result.budget_spent_units as f64;
        width_reduction += result.normalized_interval_width_reduction;
        regret += oracle_results[index]
            .resolved_constraints
            .saturating_sub(result.resolved_constraints) as f64;
        if let Some(cost) = result.cost_to_full_resolution_units {
            resolution_cost_sum += cost as f64;
            resolution_cost_count += 1.0;
        }
    }

    PolicySummary {
        policy,
        seed,
        evaluated_scenarios: results.len() as u64,
        mean_resolved_constraints: resolved / n,
        mean_robust_feasible_constraints: feasible / n,
        mean_robust_infeasible_constraints: infeasible / n,
        full_resolution_fraction: fully / n,
        mean_budget_spent_units: spent / n,
        mean_cost_to_full_resolution_units: if resolution_cost_count > 0.0 {
            Some(resolution_cost_sum / resolution_cost_count)
        } else { None },
        mean_normalized_interval_width_reduction: width_reduction / n,
        mean_oracle_regret_constraints: regret / n,
    }
}

/// Runs paired synthetic scenarios against four executable policies plus an
/// exhaustive, clairvoyant toy-model subset oracle. Inputs, catalog, seed list,
/// budget, generator/oracle revisions, summary metrics and serialized report hash
/// are bound into the returned receipt.
pub fn run_reference_benchmark(
    scenario_count: usize,
    paired_seeds: &[u64],
    budget_units_per_scenario: u32,
) -> Result<BenchmarkReceipt, BenchmarkError> {
    if scenario_count == 0 {
        return Err(BenchmarkError::EmptyScenarioSet);
    }
    if paired_seeds.is_empty() {
        return Err(BenchmarkError::EmptySeeds);
    }
    let mut seen_seeds = std::collections::BTreeSet::new();
    for seed in paired_seeds {
        if !seen_seeds.insert(*seed) {
            return Err(BenchmarkError::DuplicateSeed(*seed));
        }
    }

    let catalog = experiment_catalog();
    let scenarios: Vec<SyntheticScenario> = (0..scenario_count)
        .map(|id| generate_scenario(id as u64))
        .collect();
    let oracle_results: Vec<RunMetrics> = scenarios.iter()
        .map(|scenario| exhaustive_toy_oracle(scenario, budget_units_per_scenario, &catalog))
        .collect();

    let mut summaries = Vec::new();
    for policy in Policy::all_heuristics() {
        for seed in paired_seeds.iter().copied() {
            let results: Vec<RunMetrics> = scenarios.iter()
                .map(|scenario| {
                    run_policy(scenario, policy, seed, budget_units_per_scenario, &catalog)
                })
                .collect();
            summaries.push(summarize_policy(policy, Some(seed), &results, &oracle_results));
        }
    }
    summaries.push(summarize_policy(
        Policy::ExhaustiveToyOracle, None, &oracle_results, &oracle_results,
    ));

    let body = BenchmarkBody {
        schema_version: BENCHMARK_SCHEMA_VERSION.to_owned(),
        suite_id: "marine-experiment-selection-synthetic-v1".to_owned(),
        experiment_catalog_version: EXPERIMENT_CATALOG_VERSION.to_owned(),
        scenario_generator_version: SCENARIO_GENERATOR_VERSION.to_owned(),
        toy_oracle_version: TOY_ORACLE_VERSION.to_owned(),
        scenario_count,
        paired_seeds: paired_seeds.to_vec(),
        budget_units_per_scenario,
        experiment_catalog: catalog,
        summaries,
        limitations: vec![
            concat!(
                "Every scenario and simulated measurement is synthetic; no real route, ship, ",
                "battery, charger, or hydrodynamic solver is represented.",
            )
            .to_owned(),
            concat!(
                "Experiment outcomes are idealized nested bounds centered on hidden synthetic ",
                "truth; real tests can be noisy, biased, correlated, invalid, or inconclusive.",
            )
            .to_owned(),
            concat!(
                "The exhaustive subset oracle sees hidden truth and is a clairvoyant upper bound ",
                "only for this finite toy measurement model; it is not deployable and is not a ",
                "general global bound for real engineering.",
            )
            .to_owned(),
            concat!(
                "Interval-width reduction is not calibrated probability, expected information ",
                "gain, model-bias correction, or physical evidence.",
            )
            .to_owned(),
            concat!(
                "Budget units are dimensionless synthetic comparison units, not monetary amounts, ",
                "lab-hours, or resource estimates.",
            )
            .to_owned(),
        ],
    };
    let serialized = serde_json::to_vec(&body)
        .map_err(|error| BenchmarkError::Serialization(error.to_string()))?;
    let body_sha256 = format!("{:x}", Sha256::digest(serialized));
    Ok(BenchmarkReceipt { body, body_sha256 })
}

/// Recomputes the digest over the complete benchmark body. It verifies receipt
/// integrity only, not the scientific validity or authenticity of the inputs.
pub fn verify_receipt(receipt: &BenchmarkReceipt) -> Result<bool, BenchmarkError> {
    let serialized = serde_json::to_vec(&receipt.body)
        .map_err(|error| BenchmarkError::Serialization(error.to_string()))?;
    let expected = format!("{:x}", Sha256::digest(serialized));
    Ok(expected == receipt.body_sha256)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_unique_ids_valid_costs_and_contraction_factors() {
        let catalog = experiment_catalog();
        let ids: std::collections::BTreeSet<&str> = catalog.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids.len(), catalog.len());
        assert!(catalog.iter().all(|e| e.cost_units > 0));
        assert!(catalog
            .iter()
            .all(|e| e.retained_width_fraction > 0.0 && e.retained_width_fraction < 1.0));
    }

    #[test]
    fn generated_initial_intervals_contain_truth() {
        for id in 0..10_000 {
            let scenario = generate_scenario(id);
            for index in 0..scenario.truth.len() {
                assert!(scenario.initial[index].contains(scenario.truth[index]));
                assert!(scenario.initial[index].width() > 0.0);
            }
        }
    }

    #[test]
    fn experiment_shrinks_only_its_declared_parameter_and_keeps_truth_inside() {
        let scenario = generate_scenario(7);
        let mut state = initial_state(&scenario);
        let before = state.intervals;
        let experiment = experiment_catalog().remove(0);
        simulate_experiment(&scenario, &mut state, &experiment);
        assert!(
            state.intervals[experiment.target.index()].width()
                < before[experiment.target.index()].width()
        );
        assert!(
            state.intervals[experiment.target.index()]
                .contains(scenario.truth[experiment.target.index()])
        );
        for parameter in Parameter::ALL {
            if parameter != experiment.target {
                assert_eq!(state.intervals[parameter.index()], before[parameter.index()]);
            }
        }
    }

    #[test]
    fn interval_constraint_status_never_treats_overlap_as_resolved() {
        let demand = Interval { lower: 100.0, upper: 200.0 };
        let capacity = Interval { lower: 150.0, upper: 250.0 };
        assert_eq!(classify_pair(demand, capacity), ConstraintStatus::Indeterminate);
        assert_eq!(
            classify_pair(
                Interval { lower: 1.0, upper: 2.0 },
                Interval { lower: 2.0, upper: 3.0 },
            ),
            ConstraintStatus::RobustFeasible
        );
        assert_eq!(
            classify_pair(
                Interval { lower: 3.1, upper: 4.0 },
                Interval { lower: 1.0, upper: 3.0 },
            ),
            ConstraintStatus::RobustInfeasible
        );
    }

    #[test]
    fn heuristics_never_exceed_budget() {
        let catalog = experiment_catalog();
        for id in 0..100 {
            let scenario = generate_scenario(id);
            for policy in Policy::all_heuristics() {
                let run = run_policy(&scenario, policy, 23, 6, &catalog);
                assert!(run.budget_spent_units <= 6);
            }
        }
    }

    #[test]
    fn seeded_random_is_reproducible() {
        let scenario = generate_scenario(92);
        let catalog = experiment_catalog();
        let left = run_policy(&scenario, Policy::SeededRandom, 77, 6, &catalog);
        let right = run_policy(&scenario, Policy::SeededRandom, 77, 6, &catalog);
        assert_eq!(left.resolved_constraints, right.resolved_constraints);
        assert_eq!(left.budget_spent_units, right.budget_spent_units);
        assert_eq!(
            left.normalized_interval_width_reduction,
            right.normalized_interval_width_reduction
        );
    }

    #[test]
    fn exhaustive_toy_oracle_is_at_least_as_good_at_resolving_as_any_policy() {
        let catalog = experiment_catalog();
        for id in 0..100 {
            let scenario = generate_scenario(id);
            let oracle = exhaustive_toy_oracle(&scenario, 6, &catalog);
            for policy in Policy::all_heuristics() {
                let run = run_policy(&scenario, policy, 1234, 6, &catalog);
                assert!(oracle.resolved_constraints >= run.resolved_constraints);
                assert!(oracle.budget_spent_units <= 6);
            }
        }
    }

    #[test]
    fn receipts_are_deterministic_and_content_addressed() {
        let seeds = [0, 1, 2];
        let first = run_reference_benchmark(20, &seeds, 6).expect("benchmark runs");
        let second = run_reference_benchmark(20, &seeds, 6).expect("repeat benchmark runs");
        assert_eq!(first.body_sha256, second.body_sha256);
        assert!(verify_receipt(&first).expect("receipt serializes"));
    }

    #[test]
    fn receipt_verifier_detects_body_mutation() {
        let mut receipt = run_reference_benchmark(5, &[8], 4).expect("benchmark runs");
        receipt.body.scenario_count += 1;
        assert!(!verify_receipt(&receipt).expect("tampered receipt still serializes"));
    }

    #[test]
    fn duplicate_seeds_are_rejected() {
        assert_eq!(
            run_reference_benchmark(3, &[1, 1], 4).unwrap_err(),
            BenchmarkError::DuplicateSeed(1)
        );
    }

    #[test]
    fn zero_scenarios_and_empty_seed_list_are_rejected() {
        assert_eq!(
            run_reference_benchmark(0, &[1], 4).unwrap_err(),
            BenchmarkError::EmptyScenarioSet
        );
        assert_eq!(run_reference_benchmark(1, &[], 4).unwrap_err(), BenchmarkError::EmptySeeds);
    }

    #[test]
    fn oracle_regret_is_nonnegative() {
        let receipt = run_reference_benchmark(25, &[1, 2], 6).expect("benchmark runs");
        for summary in receipt
            .body
            .summaries
            .iter()
            .filter(|s| s.policy != Policy::ExhaustiveToyOracle)
        {
            assert!(summary.mean_oracle_regret_constraints >= 0.0);
        }
    }

    #[test]
    fn summary_keeps_feasible_and_infeasible_separate() {
        let receipt = run_reference_benchmark(15, &[2], 6).expect("benchmark runs");
        assert!(receipt.body.summaries.iter().all(|s| s.mean_robust_feasible_constraints >= 0.0));
        assert!(receipt.body.summaries.iter().all(|s| s.mean_robust_infeasible_constraints >= 0.0));
    }
}
