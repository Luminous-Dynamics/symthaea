// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Stress-replay and perfect-information recourse reference for the synthetic airport
//! energy hub. Per-scenario recourse sees the entire scenario and is an optimistic
//! reference, not an online policy. Nonanticipative/receding-horizon control and
//! empirical probability calibration are separate future work.

use std::collections::BTreeSet;
use std::fmt;

use sha2::{Digest, Sha256};

use crate::aviation_energy_hub::{
    canonical_energy_hub_digest, solve_exact_energy_hub_pareto, verify_energy_hub_plan,
    EnergyHubError, EnergyHubPlan, EvaluatedEnergyHubPlan, TimeIndexedEnergyHubScenario,
};

const PPM: u64 = 1_000_000;
const MAX_STRESS_CASES: usize = 32;
const MAX_SEARCH_NODES: u64 = 250_000;

/// Weights are scenario-design weights, not calibrated real-world probabilities unless
/// an independent calibration procedure establishes that interpretation.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedEnergyHubScenario {
    pub scenario: TimeIndexedEnergyHubScenario,
    pub weight_ppm: u32,
}

/// Failed plans remain explicit outcomes; infeasible scenarios are not dropped or
/// assigned an invented zero cost.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyHubStressOutcome {
    pub scenario_id: String,
    pub weight_ppm: u32,
    pub static_replay: Option<EvaluatedEnergyHubPlan>,
    pub static_replay_error: Option<EnergyHubError>,
    pub perfect_information_recourse: Option<EvaluatedEnergyHubPlan>,
    pub recourse_error: Option<EnergyHubError>,
}

/// Conditional statistics always travel with feasible probability mass. Conditional
/// means must not be interpreted as unconditional expected cost when cases are infeasible.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalMetricSummary {
    pub feasible_weight_ppm: u64,
    pub infeasible_weight_ppm: u64,
    pub unresolved_weight_ppm: u64,
    pub mean_cost_given_feasible: Option<f64>,
    pub mean_accounted_co2e_kg_given_feasible: Option<f64>,
    pub worst_cost_among_feasible_cases: Option<f64>,
    pub worst_accounted_co2e_kg_among_feasible_cases: Option<f64>,
    pub worst_final_soc_quanta: Option<u32>,
    pub best_final_soc_quanta: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnergyHubRobustnessReport {
    pub stress_set_digest: String,
    pub nominal_plan: EvaluatedEnergyHubPlan,
    pub outcomes: Vec<EnergyHubStressOutcome>,
    pub static_replay_summary: ConditionalMetricSummary,
    pub perfect_information_recourse_summary: ConditionalMetricSummary,
    pub static_replay_covers_all_cases: bool,
    /// Perfect-information reference coverage is not deployable policy coverage.
    pub perfect_information_recourse_covers_all_cases: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnergyHubRobustnessError {
    NoStressCases,
    TooManyStressCases,
    MissingIdentifier(&'static str),
    DuplicateScenarioId,
    ZeroScenarioWeight,
    WeightMassMismatch { actual_ppm: u64 },
    InvalidSearchBudget,
    IncompatibleScenario(&'static str),
    InvalidScenarioDigest,
    NominalPlan(EnergyHubError),
    ArithmeticOverflow,
}

impl fmt::Display for EnergyHubRobustnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoStressCases => write!(f, "at least one stress case is required"),
            Self::TooManyStressCases => write!(f, "stress case count exceeds the declared limit"),
            Self::MissingIdentifier(field) => write!(f, "missing identifier: {field}"),
            Self::DuplicateScenarioId => write!(f, "duplicate stress scenario ID"),
            Self::ZeroScenarioWeight => write!(f, "stress scenario weight must be positive"),
            Self::WeightMassMismatch { actual_ppm } => {
                write!(f, "stress scenario weights sum to {actual_ppm} ppm, expected 1000000")
            }
            Self::InvalidSearchBudget => write!(f, "search budget must be within the hard limit"),
            Self::IncompatibleScenario(field) => {
                write!(f, "stress scenarios do not share comparable {field}")
            }
            Self::InvalidScenarioDigest => write!(f, "one or more stress scenario digests are invalid"),
            Self::NominalPlan(error) => write!(f, "nominal plan verification failed: {error}"),
            Self::ArithmeticOverflow => write!(f, "stress summary aggregate overflowed"),
        }
    }
}

impl std::error::Error for EnergyHubRobustnessError {}

/// Canonical SHA-256 identity for the weighted scenario set. Input order is normalized
/// by scenario ID; each scenario digest is verified before inclusion.
pub fn canonical_stress_set_digest(
    cases: &[WeightedEnergyHubScenario],
) -> Result<String, EnergyHubRobustnessError> {
    validate_stress_cases(cases)?;
    let mut sorted = cases.iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| left.scenario.scenario_id.cmp(&right.scenario.scenario_id));

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"symthaea.clean-airline.energy-hub-stress-set.v1\0");
    put_u32(&mut bytes, sorted.len() as u32);
    for case in sorted {
        put_string(&mut bytes, &case.scenario.scenario_id);
        put_u32(&mut bytes, case.weight_ppm);
        put_string(&mut bytes, &case.scenario.input_digest);
    }
    Ok(sha256_hex(&bytes))
}

/// Evaluate a nominal plan against weighted stress scenarios.
///
/// Static replay applies the same actions to each scenario, rebinding only the scenario
/// identity. Independent verification determines whether those actions remain feasible.
/// Perfect-information recourse solves each stress case separately and is an optimistic
/// reference because it knows the complete realized scenario.
pub fn evaluate_energy_hub_stress_set(
    nominal: &TimeIndexedEnergyHubScenario,
    nominal_plan: &EnergyHubPlan,
    cases: &[WeightedEnergyHubScenario],
    max_nodes_per_case: u64,
) -> Result<EnergyHubRobustnessReport, EnergyHubRobustnessError> {
    validate_stress_cases(cases)?;
    if max_nodes_per_case == 0 || max_nodes_per_case > MAX_SEARCH_NODES {
        return Err(EnergyHubRobustnessError::InvalidSearchBudget);
    }
    ensure_comparable(nominal, cases)?;

    let nominal_plan = verify_energy_hub_plan(nominal, nominal_plan)
        .map_err(EnergyHubRobustnessError::NominalPlan)?;
    let stress_set_digest = canonical_stress_set_digest(cases)?;
    let mut ordered_cases = cases.iter().collect::<Vec<_>>();
    ordered_cases.sort_by(|left, right| left.scenario.scenario_id.cmp(&right.scenario.scenario_id));

    let mut outcomes = Vec::with_capacity(ordered_cases.len());
    for weighted in ordered_cases {
        let scenario = &weighted.scenario;
        let static_plan = crate::aviation_energy_hub::EnergyHubPlan {
            scenario_id: scenario.scenario_id.clone(),
            input_digest: scenario.input_digest.clone(),
            actions: nominal_plan.plan.actions.clone(),
        };
        let (static_replay, static_replay_error) =
            match verify_energy_hub_plan(scenario, &static_plan) {
                Ok(evaluated) => (Some(evaluated), None),
                Err(error) => (None, Some(error)),
            };

        let (perfect_information_recourse, recourse_error) =
            match solve_exact_energy_hub_pareto(scenario, max_nodes_per_case) {
                Ok(result) => (result.frontier.into_iter().next(), None),
                Err(error) => (None, Some(error)),
            };

        outcomes.push(EnergyHubStressOutcome {
            scenario_id: scenario.scenario_id.clone(),
            weight_ppm: weighted.weight_ppm,
            static_replay,
            static_replay_error,
            perfect_information_recourse,
            recourse_error,
        });
    }

    let static_replay_summary = summarize_outcomes(&outcomes, true)?;
    let perfect_information_recourse_summary = summarize_outcomes(&outcomes, false)?;
    Ok(EnergyHubRobustnessReport {
        stress_set_digest,
        nominal_plan,
        static_replay_covers_all_cases: static_replay_summary.feasible_weight_ppm == PPM,
        perfect_information_recourse_covers_all_cases:
            perfect_information_recourse_summary.feasible_weight_ppm == PPM,
        static_replay_summary,
        perfect_information_recourse_summary,
        outcomes,
    })
}

fn validate_stress_cases(
    cases: &[WeightedEnergyHubScenario],
) -> Result<(), EnergyHubRobustnessError> {
    if cases.is_empty() {
        return Err(EnergyHubRobustnessError::NoStressCases);
    }
    if cases.len() > MAX_STRESS_CASES {
        return Err(EnergyHubRobustnessError::TooManyStressCases);
    }
    let mut ids = BTreeSet::new();
    let mut weight_sum = 0_u64;
    for case in cases {
        if case.scenario.scenario_id.trim().is_empty() {
            return Err(EnergyHubRobustnessError::MissingIdentifier("stress scenario ID"));
        }
        if case.weight_ppm == 0 {
            return Err(EnergyHubRobustnessError::ZeroScenarioWeight);
        }
        if !ids.insert(case.scenario.scenario_id.as_str()) {
            return Err(EnergyHubRobustnessError::DuplicateScenarioId);
        }
        let canonical = canonical_energy_hub_digest(&case.scenario)
            .map_err(|_| EnergyHubRobustnessError::InvalidScenarioDigest)?;
        if case.scenario.input_digest != canonical {
            return Err(EnergyHubRobustnessError::InvalidScenarioDigest);
        }
        weight_sum = weight_sum
            .checked_add(u64::from(case.weight_ppm))
            .ok_or(EnergyHubRobustnessError::ArithmeticOverflow)?;
    }
    if weight_sum != PPM {
        return Err(EnergyHubRobustnessError::WeightMassMismatch { actual_ppm: weight_sum });
    }
    Ok(())
}

fn ensure_comparable(
    nominal: &TimeIndexedEnergyHubScenario,
    cases: &[WeightedEnergyHubScenario],
) -> Result<(), EnergyHubRobustnessError> {
    let nominal_digest = canonical_energy_hub_digest(nominal)
        .map_err(|_| EnergyHubRobustnessError::InvalidScenarioDigest)?;
    if nominal.input_digest != nominal_digest {
        return Err(EnergyHubRobustnessError::InvalidScenarioDigest);
    }
    let reference_ids = nominal.intervals.iter().map(|interval| interval.id.as_str()).collect::<Vec<_>>();
    for case in cases {
        let scenario = &case.scenario;
        if scenario.currency_code != nominal.currency_code {
            return Err(EnergyHubRobustnessError::IncompatibleScenario("currency"));
        }
        if scenario.quantum_kwh != nominal.quantum_kwh {
            return Err(EnergyHubRobustnessError::IncompatibleScenario("energy quantum"));
        }
        if scenario.service.period_id != nominal.service.period_id
            || scenario.service.lifecycle_boundary_id != nominal.service.lifecycle_boundary_id
            || scenario.service.completed_passenger_km != nominal.service.completed_passenger_km
            || scenario.service.available_seat_km != nominal.service.available_seat_km
        {
            return Err(EnergyHubRobustnessError::IncompatibleScenario("service/accounting boundary"));
        }
        let candidate_ids = scenario.intervals.iter().map(|interval| interval.id.as_str()).collect::<Vec<_>>();
        if candidate_ids != reference_ids {
            return Err(EnergyHubRobustnessError::IncompatibleScenario("ordered interval grid"));
        }
    }
    Ok(())
}

fn summarize_outcomes(
    outcomes: &[EnergyHubStressOutcome],
    use_static_replay: bool,
) -> Result<ConditionalMetricSummary, EnergyHubRobustnessError> {
    let mut feasible_weight_ppm = 0_u64;
    let mut weighted_cost = 0.0_f64;
    let mut weighted_co2e = 0.0_f64;
    let mut worst_cost: Option<f64> = None;
    let mut worst_co2e: Option<f64> = None;
    let mut worst_soc: Option<u32> = None;
    let mut best_soc: Option<u32> = None;
    let mut infeasible_weight_ppm = 0_u64;
    let mut unresolved_weight_ppm = 0_u64;

    for outcome in outcomes {
        let evaluated = if use_static_replay {
            outcome.static_replay.as_ref()
        } else {
            outcome.perfect_information_recourse.as_ref()
        };
        let Some(evaluated) = evaluated else {
            if use_static_replay {
                infeasible_weight_ppm = infeasible_weight_ppm
                    .checked_add(u64::from(outcome.weight_ppm))
                    .ok_or(EnergyHubRobustnessError::ArithmeticOverflow)?;
            } else if matches!(
                outcome.recourse_error.as_ref(),
                Some(EnergyHubError::NoFeasiblePlan)
            ) {
                infeasible_weight_ppm = infeasible_weight_ppm
                    .checked_add(u64::from(outcome.weight_ppm))
                    .ok_or(EnergyHubRobustnessError::ArithmeticOverflow)?;
            } else {
                // Budget exhaustion and other solver failures are unresolved, not
                // proof that the scenario has no feasible plan.
                unresolved_weight_ppm = unresolved_weight_ppm
                    .checked_add(u64::from(outcome.weight_ppm))
                    .ok_or(EnergyHubRobustnessError::ArithmeticOverflow)?;
            }
            continue;
        };
        let weight = f64::from(outcome.weight_ppm);
        weighted_cost += evaluated.metrics.source_ledger.total_cost * weight;
        weighted_co2e += evaluated.metrics.source_ledger.accounted_lifecycle_co2e_kg * weight;
        if !weighted_cost.is_finite() || !weighted_co2e.is_finite() {
            return Err(EnergyHubRobustnessError::ArithmeticOverflow);
        }
        feasible_weight_ppm = feasible_weight_ppm
            .checked_add(u64::from(outcome.weight_ppm))
            .ok_or(EnergyHubRobustnessError::ArithmeticOverflow)?;
        worst_cost = Some(worst_cost.map_or(evaluated.metrics.source_ledger.total_cost, |old| {
            old.max(evaluated.metrics.source_ledger.total_cost)
        }));
        worst_co2e = Some(worst_co2e.map_or(
            evaluated.metrics.source_ledger.accounted_lifecycle_co2e_kg,
            |old| old.max(evaluated.metrics.source_ledger.accounted_lifecycle_co2e_kg),
        ));
        worst_soc = Some(worst_soc.map_or(evaluated.metrics.final_soc_quanta, |old| {
            old.min(evaluated.metrics.final_soc_quanta)
        }));
        best_soc = Some(best_soc.map_or(evaluated.metrics.final_soc_quanta, |old| {
            old.max(evaluated.metrics.final_soc_quanta)
        }));
    }

    let accounted_weight_ppm = feasible_weight_ppm
        .checked_add(infeasible_weight_ppm)
        .and_then(|value| value.checked_add(unresolved_weight_ppm))
        .ok_or(EnergyHubRobustnessError::ArithmeticOverflow)?;
    if accounted_weight_ppm != PPM {
        return Err(EnergyHubRobustnessError::ArithmeticOverflow);
    }
    let (mean_cost_given_feasible, mean_accounted_co2e_kg_given_feasible) =
        if feasible_weight_ppm > 0 {
            let denominator = feasible_weight_ppm as f64;
            (
                Some(weighted_cost / denominator),
                Some(weighted_co2e / denominator),
            )
        } else {
            (None, None)
        };
    Ok(ConditionalMetricSummary {
        feasible_weight_ppm,
        infeasible_weight_ppm,
        unresolved_weight_ppm,
        mean_cost_given_feasible,
        mean_accounted_co2e_kg_given_feasible,
        worst_cost_among_feasible_cases: worst_cost,
        worst_accounted_co2e_kg_among_feasible_cases: worst_co2e,
        worst_final_soc_quanta: worst_soc,
        best_final_soc_quanta: best_soc,
    })
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

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_string(bytes: &mut Vec<u8>, value: &str) {
    put_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}
