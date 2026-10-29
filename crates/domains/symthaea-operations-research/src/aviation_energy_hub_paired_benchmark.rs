// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paired policy comparison for static plans, greedy dispatch, nonanticipative rolling
//! horizon, and a perfect-information exact reference. Scenario weights are design
//! weights unless independently calibrated. The exact reference is an optimistic
//! comparator, not an implementable online policy.

use std::collections::BTreeSet;
use std::fmt;

use sha2::{Digest, Sha256};

use crate::aviation_energy_hub::{
    canonical_energy_hub_digest, solve_exact_energy_hub_pareto, solve_renewable_first_greedy,
    verify_energy_hub_plan, EnergyHubError, EnergyHubPlan, EvaluatedEnergyHubPlan,
    TimeIndexedEnergyHubScenario,
};
use crate::aviation_energy_hub_rolling::{
    run_rolling_horizon, RollingHorizonEpisode, RollingHorizonError, RollingHorizonRun,
    RollingHorizonStatus,
};

const PPM: u64 = 1_000_000;
const MAX_PAIRED_SCENARIOS: usize = 32;
const MAX_SEARCH_NODES: u64 = 250_000;
const REGRET_EPSILON: f64 = 1e-9;

/// One realized scenario and its as-of forecast history. Weights are integer design
/// weights, not real-world probabilities unless separately calibrated.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedRollingEpisode {
    pub weight_ppm: u32,
    pub episode: RollingHorizonEpisode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedScenarioOutcome {
    pub scenario_id: String,
    pub weight_ppm: u32,
    pub static_plan: Option<EvaluatedEnergyHubPlan>,
    pub static_plan_error: Option<EnergyHubError>,
    pub greedy_plan: Option<EvaluatedEnergyHubPlan>,
    pub greedy_error: Option<EnergyHubError>,
    pub rolling_run: RollingHorizonRun,
    pub perfect_information_plan: Option<EvaluatedEnergyHubPlan>,
    pub perfect_information_error: Option<EnergyHubError>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyCoverageSummary {
    pub feasible_weight_ppm: u64,
    pub infeasible_weight_ppm: u64,
    pub unresolved_weight_ppm: u64,
    pub mean_cost_given_feasible: Option<f64>,
    pub mean_accounted_co2e_kg_given_feasible: Option<f64>,
    pub worst_cost_among_feasible_cases: Option<f64>,
    pub worst_accounted_co2e_kg_among_feasible_cases: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedDeltaSummary {
    pub comparable_weight_ppm: u64,
    /// Positive means the first-named policy has higher cost than the second.
    pub mean_cost_delta: Option<f64>,
    pub worst_cost_delta: Option<f64>,
    pub best_cost_delta: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedRollingBenchmarkReport {
    pub benchmark_digest: String,
    pub baseline_scenario_digest: String,
    pub outcomes: Vec<PairedScenarioOutcome>,
    pub static_plan_summary: PolicyCoverageSummary,
    pub greedy_summary: PolicyCoverageSummary,
    pub rolling_horizon_summary: PolicyCoverageSummary,
    pub perfect_information_summary: PolicyCoverageSummary,
    /// Rolling cost minus perfect-information cost, only where both completed feasibly.
    pub rolling_regret_vs_perfect_information: PairedDeltaSummary,
    /// Static cost minus rolling cost; positive means rolling was cheaper.
    pub static_minus_rolling_cost: PairedDeltaSummary,
    /// Greedy cost minus rolling cost; positive means rolling was cheaper.
    pub greedy_minus_rolling_cost: PairedDeltaSummary,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PairedBenchmarkError {
    NoEpisodes,
    TooManyEpisodes,
    ZeroWeight { scenario_id: String },
    DuplicateScenarioId { scenario_id: String },
    WeightMassMismatch { actual_ppm: u64 },
    InvalidSearchBudget,
    InvalidScenarioDigest { scenario_id: String },
    InvalidEpisode { scenario_id: String, source: RollingHorizonError },
    IncompatibleScenario(&'static str),
    InvalidStaticBaseline(EnergyHubError),
    ComparatorDominanceViolation { scenario_id: String, regret: f64 },
    ArithmeticOverflow,
}

impl fmt::Display for PairedBenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoEpisodes => write!(f, "at least one paired rolling episode is required"),
            Self::TooManyEpisodes => write!(f, "paired scenario count exceeds the declared limit"),
            Self::ZeroWeight { scenario_id } => write!(f, "scenario {scenario_id} has zero weight"),
            Self::DuplicateScenarioId { scenario_id } => write!(f, "duplicate scenario ID {scenario_id}"),
            Self::WeightMassMismatch { actual_ppm } => {
                write!(f, "scenario weights sum to {actual_ppm} ppm, expected 1000000")
            }
            Self::InvalidSearchBudget => write!(f, "search budget must be within the hard limit"),
            Self::InvalidScenarioDigest { scenario_id } => {
                write!(f, "scenario digest is invalid for {scenario_id}")
            }
            Self::InvalidEpisode { scenario_id, source } => {
                write!(f, "rolling episode is invalid for {scenario_id}: {source}")
            }
            Self::IncompatibleScenario(field) => write!(f, "paired scenarios differ in {field}"),
            Self::InvalidStaticBaseline(error) => write!(f, "static baseline failed verification: {error}"),
            Self::ComparatorDominanceViolation { scenario_id, regret } => write!(
                f,
                "rolling policy cost is below the perfect-information reference for {scenario_id} by {}",
                -regret
            ),
            Self::ArithmeticOverflow => write!(f, "paired benchmark aggregation overflowed"),
        }
    }
}

impl std::error::Error for PairedBenchmarkError {}

/// Run every policy on the same weighted realized-scenario panel. The unchanged
/// baseline actions are independently replayed against each realized scenario.
pub fn evaluate_paired_rolling_benchmark(
    static_baseline_scenario: &TimeIndexedEnergyHubScenario,
    static_baseline_plan: &EnergyHubPlan,
    episodes: &[WeightedRollingEpisode],
    max_nodes_per_policy: u64,
) -> Result<PairedRollingBenchmarkReport, PairedBenchmarkError> {
    if episodes.is_empty() {
        return Err(PairedBenchmarkError::NoEpisodes);
    }
    if episodes.len() > MAX_PAIRED_SCENARIOS {
        return Err(PairedBenchmarkError::TooManyEpisodes);
    }
    if max_nodes_per_policy == 0 || max_nodes_per_policy > MAX_SEARCH_NODES {
        return Err(PairedBenchmarkError::InvalidSearchBudget);
    }

    let baseline_digest = canonical_energy_hub_digest(static_baseline_scenario)
        .map_err(|_| PairedBenchmarkError::InvalidScenarioDigest {
            scenario_id: static_baseline_scenario.scenario_id.clone(),
        })?;
    if static_baseline_scenario.input_digest != baseline_digest {
        return Err(PairedBenchmarkError::InvalidScenarioDigest {
            scenario_id: static_baseline_scenario.scenario_id.clone(),
        });
    }
    verify_energy_hub_plan(static_baseline_scenario, static_baseline_plan)
        .map_err(PairedBenchmarkError::InvalidStaticBaseline)?;

    let mut seen_ids = BTreeSet::new();
    let mut total_weight = 0_u64;
    for weighted in episodes {
        let scenario = &weighted.episode.realized_scenario;
        if weighted.weight_ppm == 0 {
            return Err(PairedBenchmarkError::ZeroWeight {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        if !seen_ids.insert(scenario.scenario_id.as_str()) {
            return Err(PairedBenchmarkError::DuplicateScenarioId {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        let digest = canonical_energy_hub_digest(scenario)
            .map_err(|_| PairedBenchmarkError::InvalidScenarioDigest {
                scenario_id: scenario.scenario_id.clone(),
            })?;
        if scenario.input_digest != digest {
            return Err(PairedBenchmarkError::InvalidScenarioDigest {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        ensure_comparable(static_baseline_scenario, scenario)?;
        total_weight = total_weight
            .checked_add(u64::from(weighted.weight_ppm))
            .ok_or(PairedBenchmarkError::ArithmeticOverflow)?;
    }
    if total_weight != PPM {
        return Err(PairedBenchmarkError::WeightMassMismatch {
            actual_ppm: total_weight,
        });
    }

    let mut ordered = episodes.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        left.episode.realized_scenario.scenario_id.cmp(
            &right.episode.realized_scenario.scenario_id
        )
    });

    let mut outcomes = Vec::with_capacity(ordered.len());
    for weighted in ordered {
        let scenario = &weighted.episode.realized_scenario;
        let static_replay_plan = EnergyHubPlan {
            scenario_id: scenario.scenario_id.clone(),
            input_digest: scenario.input_digest.clone(),
            actions: static_baseline_plan.actions.clone(),
        };
        let (static_plan, static_plan_error) =
            match verify_energy_hub_plan(scenario, &static_replay_plan) {
                Ok(evaluated) => (Some(evaluated), None),
                Err(error) => (None, Some(error)),
            };

        let (greedy_plan, greedy_error) = match solve_renewable_first_greedy(scenario) {
            Ok(evaluated) => (Some(evaluated), None),
            Err(error) => (None, Some(error)),
        };

        let rolling_run = run_rolling_horizon(&weighted.episode, max_nodes_per_policy)
            .map_err(|source| PairedBenchmarkError::InvalidEpisode {
                scenario_id: scenario.scenario_id.clone(),
                source,
            })?;

        let (perfect_information_plan, perfect_information_error) =
            match solve_exact_pareto(scenario, max_nodes_per_policy) {
                Ok(result) => (result.frontier.into_iter().next(), None),
                Err(error) => (None, Some(error)),
            };

        outcomes.push(PairedScenarioOutcome {
            scenario_id: scenario.scenario_id.clone(),
            weight_ppm: weighted.weight_ppm,
            static_plan,
            static_plan_error,
            greedy_plan,
            greedy_error,
            rolling_run,
            perfect_information_plan,
            perfect_information_error,
        });
    }

    let static_plan_summary = summarize_policy(&outcomes, PolicySelector::Static)?;
    let greedy_summary = summarize_policy(&outcomes, PolicySelector::Greedy)?;
    let rolling_horizon_summary = summarize_policy(&outcomes, PolicySelector::Rolling)?;
    let perfect_information_summary = summarize_policy(&outcomes, PolicySelector::PerfectInformation)?;

    let rolling_regret_vs_perfect_information =
        summarize_delta(&outcomes, PolicySelector::Rolling, PolicySelector::PerfectInformation, true)?;
    let static_minus_rolling_cost =
        summarize_delta(&outcomes, PolicySelector::Static, PolicySelector::Rolling, false)?;
    let greedy_minus_rolling_cost =
        summarize_delta(&outcomes, PolicySelector::Greedy, PolicySelector::Rolling, false)?;

    let benchmark_digest = canonical_benchmark_digest(&baseline_digest, static_baseline_plan, &outcomes)?;
    Ok(PairedRollingBenchmarkReport {
        benchmark_digest,
        baseline_scenario_digest: baseline_digest,
        outcomes,
        static_plan_summary,
        greedy_summary,
        rolling_horizon_summary,
        perfect_information_summary,
        rolling_regret_vs_perfect_information,
        static_minus_rolling_cost,
        greedy_minus_rolling_cost,
    })
}

#[derive(Debug, Clone, Copy)]
enum PolicySelector {
    Static,
    Greedy,
    Rolling,
    PerfectInformation,
}

fn selected_plan(
    outcome: &PairedScenarioOutcome,
    selector: PolicySelector,
) -> Option<&EvaluatedEnergyHubPlan> {
    match selector {
        PolicySelector::Static => outcome.static_plan.as_ref(),
        PolicySelector::Greedy => outcome.greedy_plan.as_ref(),
        PolicySelector::Rolling => outcome.rolling_run.evaluated_realized_plan.as_ref(),
        PolicySelector::PerfectInformation => outcome.perfect_information_plan.as_ref(),
    }
}

/// Return (is_infeasible, is_unresolved) when no verified plan exists.
fn failure_status(outcome: &PairedScenarioOutcome, selector: PolicySelector) -> (bool, bool) {
    if selected_plan(outcome, selector).is_some() {
        return (false, false);
    }
    match selector {
        PolicySelector::Static => (true, false),
        PolicySelector::Greedy => match outcome.greedy_error.as_ref() {
            Some(EnergyHubError::NoFeasiblePlan) => (true, false),
            Some(_) | None => (false, true),
        },
        PolicySelector::Rolling => match &outcome.rolling_run.status {
            RollingHorizonStatus::NoFeasiblePlan { .. } => (true, false),
            RollingHorizonStatus::Completed
            | RollingHorizonStatus::SearchBudgetExceeded { .. }
            | RollingHorizonStatus::SolverFailure { .. } => (false, true),
        },
        PolicySelector::PerfectInformation => match outcome.perfect_information_error.as_ref() {
            Some(EnergyHubError::NoFeasiblePlan) => (true, false),
            Some(_) | None => (false, true),
        },
    }
}

fn summarize_policy(
    outcomes: &[PairedScenarioOutcome],
    selector: PolicySelector,
) -> Result<PolicyCoverageSummary, PairedBenchmarkError> {
    let mut feasible_weight = 0_u64;
    let mut infeasible_weight = 0_u64;
    let mut unresolved_weight = 0_u64;
    let mut weighted_cost = 0.0_f64;
    let mut weighted_co2e = 0.0_f64;
    let mut worst_cost: Option<f64> = None;
    let mut worst_co2e: Option<f64> = None;

    for outcome in outcomes {
        if let Some(plan) = selected_plan(outcome, selector) {
            let weight = f64::from(outcome.weight_ppm);
            weighted_cost += plan.metrics.source_ledger.total_cost * weight;
            weighted_co2e += plan.metrics.source_ledger.accounted_lifecycle_co2e_kg * weight;
            if !weighted_cost.is_finite() || !weighted_co2e.is_finite() {
                return Err(PairedBenchmarkError::ArithmeticOverflow);
            }
            feasible_weight = feasible_weight
                .checked_add(u64::from(outcome.weight_ppm))
                .ok_or(PairedBenchmarkError::ArithmeticOverflow)?;
            worst_cost = Some(worst_cost.map_or(plan.metrics.source_ledger.total_cost, |old| {
                old.max(plan.metrics.source_ledger.total_cost)
            }));
            worst_co2e = Some(worst_co2e.map_or(
                plan.metrics.source_ledger.accounted_lifecycle_co2e_kg,
                |old| old.max(plan.metrics.source_ledger.accounted_lifecycle_co2e_kg),
            ));
        } else {
            let (infeasible, unresolved) = failure_status(outcome, selector);
            if unresolved {
                unresolved_weight = unresolved_weight
                    .checked_add(u64::from(outcome.weight_ppm))
                    .ok_or(PairedBenchmarkError::ArithmeticOverflow)?;
            } else if infeasible {
                infeasible_weight = infeasible_weight
                    .checked_add(u64::from(outcome.weight_ppm))
                    .ok_or(PairedBenchmarkError::ArithmeticOverflow)?;
            } else {
                return Err(PairedBenchmarkError::ArithmeticOverflow);
            }
        }
    }

    let total = feasible_weight
        .checked_add(infeasible_weight)
        .and_then(|value| value.checked_add(unresolved_weight))
        .ok_or(PairedBenchmarkError::ArithmeticOverflow)?;
    if total != PPM {
        return Err(PairedBenchmarkError::ArithmeticOverflow);
    }

    let (mean_cost_given_feasible, mean_accounted_co2e_kg_given_feasible) =
        if feasible_weight > 0 {
            let denominator = feasible_weight as f64;
            (Some(weighted_cost / denominator), Some(weighted_co2e / denominator))
        } else {
            (None, None)
        };

    Ok(PolicyCoverageSummary {
        feasible_weight_ppm: feasible_weight,
        infeasible_weight_ppm: infeasible_weight,
        unresolved_weight_ppm: unresolved_weight,
        mean_cost_given_feasible,
        mean_accounted_co2e_kg_given_feasible,
        worst_cost_among_feasible_cases: worst_cost,
        worst_accounted_co2e_kg_among_feasible_cases: worst_co2e,
    })
}

fn summarize_delta(
    outcomes: &[PairedScenarioOutcome],
    first: PolicySelector,
    second: PolicySelector,
    check_nonnegative: bool,
) -> Result<PairedDeltaSummary, PairedBenchmarkError> {
    let mut comparable_weight = 0_u64;
    let mut weighted_delta = 0.0_f64;
    let mut worst_delta: Option<f64> = None;
    let mut best_delta: Option<f64> = None;
    for outcome in outcomes {
        let (Some(first_plan), Some(second_plan)) =
            (selected_plan(outcome, first), selected_plan(outcome, second))
        else {
            continue;
        };
        let delta = first_plan.metrics.source_ledger.total_cost
            - second_plan.metrics.source_ledger.total_cost;
        if !delta.is_finite() {
            return Err(PairedBenchmarkError::ArithmeticOverflow);
        }
        if check_nonnegative && delta < -REGRET_EPSILON {
            return Err(PairedBenchmarkError::ComparatorDominanceViolation {
                scenario_id: outcome.scenario_id.clone(),
                regret: delta,
            });
        }
        weighted_delta += delta * f64::from(outcome.weight_ppm);
        if !weighted_delta.is_finite() {
            return Err(PairedBenchmarkError::ArithmeticOverflow);
        }
        comparable_weight = comparable_weight
            .checked_add(u64::from(outcome.weight_ppm))
            .ok_or(PairedBenchmarkError::ArithmeticOverflow)?;
        worst_delta = Some(worst_delta.map_or(delta, |old| old.max(delta)));
        best_delta = Some(best_delta.map_or(delta, |old| old.min(delta)));
    }

    Ok(PairedDeltaSummary {
        comparable_weight_ppm: comparable_weight,
        mean_cost_delta: if comparable_weight > 0 {
            Some(weighted_delta / comparable_weight as f64)
        } else {
            None
        },
        worst_cost_delta: worst_delta,
        best_cost_delta: best_delta,
    })
}

fn ensure_comparable(
    baseline: &TimeIndexedEnergyHubScenario,
    candidate: &TimeIndexedEnergyHubScenario,
) -> Result<(), PairedBenchmarkError> {
    if baseline.currency_code != candidate.currency_code {
        return Err(PairedBenchmarkError::IncompatibleScenario("currency"));
    }
    if baseline.quantum_kwh != candidate.quantum_kwh {
        return Err(PairedBenchmarkError::IncompatibleScenario("energy quantum"));
    }
    if baseline.service.period_id != candidate.service.period_id
        || baseline.service.lifecycle_boundary_id != candidate.service.lifecycle_boundary_id
        || baseline.service.completed_passenger_km != candidate.service.completed_passenger_km
        || baseline.service.available_seat_km != candidate.service.available_seat_km
    {
        return Err(PairedBenchmarkError::IncompatibleScenario("service/accounting boundary"));
    }
    if baseline.storage != candidate.storage {
        return Err(PairedBenchmarkError::IncompatibleScenario("storage physics/opening inventory"));
    }
    let baseline_ids = baseline.intervals.iter().map(|item| item.id.as_str()).collect::<Vec<_>>();
    let candidate_ids = candidate.intervals.iter().map(|item| item.id.as_str()).collect::<Vec<_>>();
    if baseline_ids != candidate_ids {
        return Err(PairedBenchmarkError::IncompatibleScenario("ordered interval grid"));
    }
    Ok(())
}

fn canonical_benchmark_digest(
    baseline_digest: &str,
    static_plan: &EnergyHubPlan,
    outcomes: &[PairedScenarioOutcome],
) -> Result<String, PairedBenchmarkError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"symthaea.clean-airline.paired-rolling-benchmark.v1\0");
    put_string(&mut bytes, baseline_digest);
    put_string(&mut bytes, &static_plan.scenario_id);
    put_string(&mut bytes, &static_plan.input_digest);
    put_u32(&mut bytes, usize_to_u32(static_plan.actions.len())?);
    for action in &static_plan.actions {
        encode_action(&mut bytes, action);
    }
    put_u32(&mut bytes, usize_to_u32(outcomes.len())?);
    for outcome in outcomes {
        put_string(&mut bytes, &outcome.scenario_id);
        put_u32(&mut bytes, outcome.weight_ppm);
        put_string(&mut bytes, &outcome.rolling_run.episode_digest);
        let status = format!("{:?}", outcome.rolling_run.status);
        put_string(&mut bytes, &status);
    }
    Ok(sha256_hex(&bytes))
}

fn encode_action(
    bytes: &mut Vec<u8>,
    action: &crate::aviation_energy_hub::EnergyHubStepAction,
) {
    put_string(bytes, &action.interval_id);
    for value in [
        action.renewable_direct_quanta,
        action.grid_import_quanta,
        action.battery_discharge_output_quanta,
        action.renewable_charge_input_quanta,
        action.renewable_curtailed_quanta,
        action.soc_before_quanta,
        action.soc_after_quanta,
    ] {
        put_u32(bytes, value);
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(7 + digest.len() * 2);
    result.push_str("sha256:");
    for byte in digest.iter() {
        use std::fmt::Write as _;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn usize_to_u32(value: usize) -> Result<u32, PairedBenchmarkError> {
    u32::try_from(value).map_err(|_| PairedBenchmarkError::ArithmeticOverflow)
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}
