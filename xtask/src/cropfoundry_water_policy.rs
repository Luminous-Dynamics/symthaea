use anyhow::{Result, anyhow};
use serde::Serialize;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;

const DAYS: usize = 42;
const ZONES: usize = 3;
const STORAGE_CAP_MM: f64 = 10.0;
const MAX_IRRIGATION_PER_ZONE_MM: f64 = 3.5;
const ALLOCATION_STEP_MM: f64 = 0.05;
const BASE_DEMAND_MM: [f64; ZONES] = [2.2, 2.7, 1.8];
pub(crate) const WATER_MATCHED_POLICY_NAME: &str = "water_matched_rotating_equal_share";

#[derive(Clone, Copy)]
pub(crate) struct Scenario {
    pub(crate) id: &'static str,
    pub(crate) daily_water_budget_plot_mm: f64,
    pub(crate) drought_start_day: Option<usize>,
    pub(crate) heat_start_day: Option<usize>,
    pub(crate) forecast_bias: f64,
    pub(crate) sensor_dropout_period: Option<usize>,
}

pub(crate) /// Fixed, deterministic extension of the CF-022 base suite for robustness sensitivity.
pub(crate) fn robustness_scenarios() -> Vec<Scenario> {
    let mut suite = scenarios().to_vec();
    suite.extend([
        Scenario {
            id: "water_cap_only_3_5",
            daily_water_budget_plot_mm: 3.5,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "water_cap_only_4_0",
            daily_water_budget_plot_mm: 4.0,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "water_cap_only_5_5",
            daily_water_budget_plot_mm: 5.5,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "water_cap_only_6_5",
            daily_water_budget_plot_mm: 6.5,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "forecast_bias_only_negative_0_30",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: -0.30,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "forecast_bias_only_negative_0_20",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: -0.20,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "forecast_bias_only_positive_0_10",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.10,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "forecast_bias_only_positive_0_20",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.20,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "sensor_dropout_only_period_3",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: Some(3),
        },
        Scenario {
            id: "sensor_dropout_only_period_5",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: Some(5),
        },
        Scenario {
            id: "sensor_dropout_only_period_11",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: Some(11),
        },
        Scenario {
            id: "drought_only_start_day_8",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: Some(8),
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "drought_only_start_day_16",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: Some(16),
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "drought_only_start_day_24",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: Some(24),
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "heat_only_start_day_16",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: None,
            heat_start_day: Some(16),
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "heat_only_start_day_24",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: None,
            heat_start_day: Some(24),
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "heat_only_start_day_32",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: None,
            heat_start_day: Some(32),
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "heat_drought_low_water",
            daily_water_budget_plot_mm: 4.0,
            drought_start_day: Some(8),
            heat_start_day: Some(16),
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "heat_drought_severe_underforecast",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: Some(8),
            heat_start_day: Some(16),
            forecast_bias: -0.30,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "heat_drought_sensor_dropout_3",
            daily_water_budget_plot_mm: 5.0,
            drought_start_day: Some(8),
            heat_start_day: Some(16),
            forecast_bias: 0.0,
            sensor_dropout_period: Some(3),
        },
        Scenario {
            id: "low_water_severe_underforecast",
            daily_water_budget_plot_mm: 3.5,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: -0.30,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "low_water_sensor_dropout_3",
            daily_water_budget_plot_mm: 3.5,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: Some(3),
        },
        Scenario {
            id: "compound_heat_drought_underforecast_dropout",
            daily_water_budget_plot_mm: 4.0,
            drought_start_day: Some(8),
            heat_start_day: Some(16),
            forecast_bias: -0.30,
            sensor_dropout_period: Some(3),
        },
        Scenario {
            id: "compound_late_heat_late_drought_bias_dropout",
            daily_water_budget_plot_mm: 4.5,
            drought_start_day: Some(24),
            heat_start_day: Some(32),
            forecast_bias: -0.20,
            sensor_dropout_period: Some(5),
        },
    ]);
    suite
}

/// Re-simulate each fixed scenario for the same five policies and the dose-matched comparator.
pub(crate) fn simulate_policy_suite(all_scenarios: &[Scenario]) -> Result<Vec<ScenarioResult>> {
    if all_scenarios.is_empty() {
        return Err(anyhow!("scenario suite must contain at least one scenario"));
    }
    let mut ids = std::collections::BTreeSet::new();
    if all_scenarios.iter().any(|scenario| !ids.insert(scenario.id)) {
        return Err(anyhow!("scenario suite contains duplicate scenario IDs"));
    }
    let mut results = Vec::with_capacity(all_scenarios.len() * 5);
    for scenario in all_scenarios.iter().copied() {
        let (equal_share_result, _) = simulate_one(scenario, Policy::EqualShareCalendar, None);
        let (soil_threshold_result, _) = simulate_one(scenario, Policy::SoilThreshold, None);
        let (risk_result, risk_trace) = simulate_one(scenario, Policy::UncertaintyAware, None);
        let (minimax_result, _) = simulate_one(scenario, Policy::GreedyMinimaxStress, None);
        let (matched_result, matched_trace) = simulate_one(
            scenario,
            Policy::EqualShareCalendar,
            Some(&risk_trace),
        );
        if (risk_result.total_irrigation_applied_plot_mm
            - matched_result.total_irrigation_applied_plot_mm)
            .abs()
            > 1e-3
            || risk_trace
                .iter()
                .zip(matched_trace)
                .any(|(risk, matched)| (risk - matched).abs() > 1e-8)
        {
            return Err(anyhow!(
                "dose-matched counterfactual mismatch for scenario {}",
                scenario.id
            ));
        }
        results.extend([
            equal_share_result,
            soil_threshold_result,
            risk_result,
            minimax_result,
            matched_result,
        ]);
    }
    let water_violations = results
        .iter()
        .map(|result| result.water_budget_violation_days)
        .sum::<usize>();
    let plot_violations = results
        .iter()
        .map(|result| result.per_plot_limit_violation_days)
        .sum::<usize>();
    if water_violations != 0 || plot_violations != 0 {
        return Err(anyhow!(
            "scenario suite has {water_violations} water-budget violations and {plot_violations} per-plot limit violations"
        ));
    }
    Ok(results)
}

fn scenarios() -> [Scenario; 6] {
    [
        Scenario {
            id: "reference",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "water_rationing_only",
            daily_water_budget_plot_mm: 4.5,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "heat_and_drought",
            daily_water_budget_plot_mm: 5.0,
            drought_start_day: Some(8),
            heat_start_day: Some(18),
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "forecast_bias_only",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: -0.15,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "sensor_dropout_only",
            daily_water_budget_plot_mm: 6.0,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: Some(7),
        },
        Scenario {
            id: "combined_heat_drought_forecast_bias_and_sensor_dropout",
            daily_water_budget_plot_mm: 5.5,
            drought_start_day: Some(12),
            heat_start_day: Some(24),
            forecast_bias: -0.15,
            sensor_dropout_period: Some(7),
        },
    ]
}
#[derive(Clone, Copy)]
pub(crate) enum Policy {
    EqualShareCalendar,
    SoilThreshold,
    UncertaintyAware,
    GreedyMinimaxStress,
}

impl Policy {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::EqualShareCalendar => "equal_share_calendar",
            Self::SoilThreshold => "soil_threshold",
            Self::UncertaintyAware => "uncertainty_aware_marginal_risk",
            Self::GreedyMinimaxStress => "greedy_minimax_stress",
        }
    }
}

#[derive(Debug, Serialize, Clone)]
pub(crate) struct ScenarioResult {
    pub(crate) scenario_id: String,
    pub(crate) policy: String,
    drought_start_day: Option<usize>,
    heat_start_day: Option<usize>,
    forecast_bias_fraction: f64,
    sensor_dropout_period_days: Option<usize>,
    simulated_days: usize,
    daily_water_budget_plot_mm: f64,
    pub(crate) total_irrigation_applied_plot_mm: f64,
    zone_irrigation_applied_plot_mm: [f64; ZONES],
    pub(crate) plot_allocation_dispersion_plot_mm: f64,
    total_overflow_proxy_plot_mm: f64,
    pub(crate) stress_exposure_proxy: f64,
    zone_stress_exposure_proxy: [f64; ZONES],
    pub(crate) max_zone_stress_exposure_proxy: f64,
    min_zone_stress_exposure_proxy: f64,
    worst_to_best_zone_stress_ratio: Option<f64>,
    stress_delta_vs_reference_proxy: Option<f64>,
    relative_stress_change_vs_reference_pct: Option<f64>,
    relative_stress_change_vs_equal_share_pct: Option<f64>,
    relative_stress_change_vs_matched_water_equal_share_pct: Option<f64>,
    sensor_dropout_fraction: f64,
    pub(crate) water_budget_violation_days: usize,
    pub(crate) per_plot_limit_violation_days: usize,
}

#[derive(Debug, Serialize)]
struct PolicyAggregate {
    policy: String,
    scenario_count: usize,
    worst_to_best_ratio_defined_scenario_count: usize,
    mean_stress_exposure_proxy: f64,
    median_stress_exposure_proxy: f64,
    total_irrigation_applied_plot_mm: f64,
    mean_plot_allocation_dispersion_plot_mm: f64,
    median_worst_zone_stress_exposure_proxy: f64,
    max_worst_zone_stress_exposure_proxy: f64,
    mean_worst_to_best_zone_stress_ratio: Option<f64>,
    mean_relative_stress_change_vs_equal_share_pct: Option<f64>,
    mean_relative_stress_change_vs_matched_water_equal_share_pct: Option<f64>,
    total_water_budget_violation_days: usize,
    total_per_plot_limit_violation_days: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct WaterPolicyReceipt {
    schema_version: String,
    benchmark_id: String,
    execution_mode: String,
    budget_measurement_unit: String,
    synthetic_only: bool,
    biological_performance_claim: bool,
    training_authorized: bool,
    pub(crate) scenario_count: usize,
    pub(crate) policy_count: usize,
    invariant_summary: String,
    aggregates: Vec<PolicyAggregate>,
    pub(crate) results: Vec<ScenarioResult>,
    caveats: Vec<String>,
}

fn actual_demand_mm(day: usize, zone: usize, scenario: Scenario) -> f64 {
    let phase = (day as f64 / DAYS as f64) * PI + zone as f64 * 0.7;
    let phenology_proxy = 0.88 + 0.22 * (phase.sin() + 1.0) / 2.0;
    let heat_multiplier = if scenario.heat_start_day.is_some_and(|start| day >= start) {
        1.30
    } else {
        1.0
    };
    BASE_DEMAND_MM[zone] * phenology_proxy * heat_multiplier
}

fn forecast_demand_mm(day: usize, zone: usize, scenario: Scenario) -> f64 {
    let deterministic_error = 0.04 * ((day * 3 + zone * 5) as f64).sin();
    (actual_demand_mm(day, zone, scenario) * (1.0 + scenario.forecast_bias + deterministic_error))
        .max(0.1)
}

fn rain_mm(day: usize, scenario: Scenario) -> f64 {
    let is_rain_day = day % 7 == 4;
    let in_drought = scenario.drought_start_day.is_some_and(|start| day >= start);
    if is_rain_day && !in_drought { 1.1 } else { 0.0 }
}

fn criticality(day: usize, zone: usize) -> f64 {
    let critical_window = match zone {
        0 => 10..=25,
        1 => 17..=34,
        _ => 4..=19,
    };
    if critical_window.contains(&day) {
        1.5
    } else {
        0.8
    }
}

fn minimum_storage_mm(day: usize, zone: usize) -> f64 {
    2.6 + criticality(day, zone) * 0.9
}

fn exposure(storage_mm: f64, demand_mm: f64, rain_mm: f64, day: usize, zone: usize) -> f64 {
    // Preserve a negative end-of-step water margin as unmet demand for scoring.
    // The simulator clamps physical storage separately before the next step.
    let end_storage_margin = storage_mm.min(STORAGE_CAP_MM) - demand_mm + rain_mm;
    let deficit = (minimum_storage_mm(day, zone) - end_storage_margin).max(0.0);
    criticality(day, zone) * deficit * deficit
}

fn is_sensor_missing(day: usize, zone: usize, scenario: Scenario) -> bool {
    scenario
        .sensor_dropout_period
        .is_some_and(|period| period > 0 && (day + zone) % period == 0)
}

fn sensor_reading(storage_mm: f64, day: usize, zone: usize) -> f64 {
    let repeatable_error = 0.18 * ((day * 11 + zone * 7) as f64).sin();
    (storage_mm + repeatable_error).clamp(0.0, STORAGE_CAP_MM)
}

fn predicted_zone_risks(
    observed_storage: [f64; ZONES],
    allocation: [f64; ZONES],
    stale_age: [usize; ZONES],
    day: usize,
    scenario: Scenario,
) -> [f64; ZONES] {
    let forecast_rain = rain_mm(day, scenario);
    std::array::from_fn(|zone| {
        let stale = stale_age[zone].min(4) as f64;
        let conservative_demand = forecast_demand_mm(day, zone, scenario) * (1.10 + 0.05 * stale);
        let risk_weight = criticality(day, zone) * (1.0 + 0.12 * stale);
        risk_weight
            * exposure(
                observed_storage[zone] + allocation[zone],
                conservative_demand,
                forecast_rain,
                day,
                zone,
            )
    })
}

fn allocate_water(
    policy: Policy,
    observed_storage: [f64; ZONES],
    stale_age: [usize; ZONES],
    day: usize,
    scenario: Scenario,
) -> [f64; ZONES] {
    let mut allocation = [0.0; ZONES];
    match policy {
        Policy::EqualShareCalendar => {
            let share = (scenario.daily_water_budget_plot_mm / ZONES as f64)
                .min(MAX_IRRIGATION_PER_ZONE_MM);
            allocation.fill(share);
        }
        Policy::SoilThreshold => {
            let units = (scenario.daily_water_budget_plot_mm / ALLOCATION_STEP_MM).floor() as usize;
            for _ in 0..units {
                let next = (0..ZONES)
                    .filter(|zone| {
                        allocation[*zone] + ALLOCATION_STEP_MM <= MAX_IRRIGATION_PER_ZONE_MM + 1e-9
                    })
                    .max_by(|left, right| {
                        let left_gap = (7.2 - observed_storage[*left] - allocation[*left]).max(0.0);
                        let right_gap =
                            (7.2 - observed_storage[*right] - allocation[*right]).max(0.0);
                        left_gap.total_cmp(&right_gap)
                    });
                let Some(zone) = next else { break };
                let gap = (7.2 - observed_storage[zone] - allocation[zone]).max(0.0);
                if gap <= 0.0 {
                    break;
                }
                allocation[zone] += ALLOCATION_STEP_MM;
            }
        }
        Policy::UncertaintyAware => {
            let units = (scenario.daily_water_budget_plot_mm / ALLOCATION_STEP_MM).floor() as usize;
            for _ in 0..units {
                let mut best: Option<(usize, f64)> = None;
                for zone in 0..ZONES {
                    if allocation[zone] + ALLOCATION_STEP_MM > MAX_IRRIGATION_PER_ZONE_MM + 1e-9 {
                        continue;
                    }
                    let forecast_demand = forecast_demand_mm(day, zone, scenario);
                    let forecast_rain = rain_mm(day, scenario);
                    let current_storage = observed_storage[zone] + allocation[zone];
                    let stale = stale_age[zone].min(4) as f64;
                    let risk_weight = criticality(day, zone) * (1.0 + 0.12 * stale);
                    let conservative_demand = forecast_demand * (1.10 + 0.05 * stale);
                    let before = risk_weight
                        * exposure(
                            current_storage,
                            conservative_demand,
                            forecast_rain,
                            day,
                            zone,
                        );
                    let after = risk_weight
                        * exposure(
                            current_storage + ALLOCATION_STEP_MM,
                            conservative_demand,
                            forecast_rain,
                            day,
                            zone,
                        );
                    let marginal_gain = before - after;
                    if best.is_none_or(|(_, best_gain)| marginal_gain > best_gain) {
                        best = Some((zone, marginal_gain));
                    }
                }
                let Some((zone, gain)) = best else { break };
                if gain <= 1e-9 {
                    break;
                }
                allocation[zone] += ALLOCATION_STEP_MM;
            }
        }
        Policy::GreedyMinimaxStress => {
            let units = (scenario.daily_water_budget_plot_mm / ALLOCATION_STEP_MM).floor() as usize;
            for _ in 0..units {
                let current_risks =
                    predicted_zone_risks(observed_storage, allocation, stale_age, day, scenario);
                let current_max = current_risks
                    .iter()
                    .copied()
                    .max_by(f64::total_cmp)
                    .unwrap_or(0.0);
                let current_sum = current_risks.iter().sum::<f64>();
                let mut best: Option<(usize, f64, f64)> = None;

                for zone in 0..ZONES {
                    if allocation[zone] + ALLOCATION_STEP_MM > MAX_IRRIGATION_PER_ZONE_MM + 1e-9 {
                        continue;
                    }
                    let mut candidate = allocation;
                    candidate[zone] += ALLOCATION_STEP_MM;
                    let risks =
                        predicted_zone_risks(observed_storage, candidate, stale_age, day, scenario);
                    let candidate_max = risks.iter().copied().max_by(f64::total_cmp).unwrap_or(0.0);
                    let candidate_sum = risks.iter().sum::<f64>();
                    let better = best.is_none_or(|(_, best_max, best_sum)| {
                        candidate_max < best_max - 1e-9
                            || ((candidate_max - best_max).abs() <= 1e-9
                                && candidate_sum < best_sum - 1e-9)
                    });
                    if better {
                        best = Some((zone, candidate_max, candidate_sum));
                    }
                }

                let Some((zone, best_max, best_sum)) = best else {
                    break;
                };
                let improves_max = best_max < current_max - 1e-9;
                let improves_sum =
                    (best_max - current_max).abs() <= 1e-9 && best_sum < current_sum - 1e-9;
                if !improves_max && !improves_sum {
                    break;
                }
                allocation[zone] += ALLOCATION_STEP_MM;
            }
        }
    }
    allocation
}

fn matched_uniform_allocation(daily_total_mm: f64, day: usize) -> [f64; ZONES] {
    let units = (daily_total_mm / ALLOCATION_STEP_MM).round() as usize;
    let base_units = units / ZONES;
    let remainder = units % ZONES;
    let mut allocation = [base_units as f64 * ALLOCATION_STEP_MM; ZONES];
    for offset in 0..remainder {
        allocation[(day + offset) % ZONES] += ALLOCATION_STEP_MM;
    }
    allocation
}

pub(crate) fn simulate_one(
    scenario: Scenario,
    policy: Policy,
    matched_daily_totals: Option<&[f64; DAYS]>,
) -> (ScenarioResult, [f64; DAYS]) {
    let mut storage = [6.0; ZONES];
    let mut last_observation = storage;
    let mut stale_age = [0_usize; ZONES];
    let mut irrigation_total = 0.0;
    let mut irrigation_by_zone = [0.0; ZONES];
    let mut daily_irrigation_trace = [0.0; DAYS];
    let mut overflow_total = 0.0;
    let mut stress_exposure = 0.0;
    let mut zone_stress_exposure = [0.0; ZONES];
    let mut sensor_reads = 0_usize;
    let mut missing_reads = 0_usize;
    let mut violation_days = 0_usize;
    let mut per_plot_limit_violation_days = 0_usize;

    for day in 0..DAYS {
        let mut observed = last_observation;
        for zone in 0..ZONES {
            if is_sensor_missing(day, zone, scenario) {
                missing_reads += 1;
                stale_age[zone] += 1;
            } else {
                observed[zone] = sensor_reading(storage[zone], day, zone);
                last_observation[zone] = observed[zone];
                stale_age[zone] = 0;
                sensor_reads += 1;
            }
        }

        let allocation = if let Some(matched_totals) = matched_daily_totals {
            matched_uniform_allocation(matched_totals[day], day)
        } else {
            allocate_water(policy, observed, stale_age, day, scenario)
        };
        let applied: f64 = allocation.iter().sum();
        daily_irrigation_trace[day] = applied;
        irrigation_total += applied;
        if applied > scenario.daily_water_budget_plot_mm + 1e-8 {
            violation_days += 1;
        }
        if allocation
            .iter()
            .any(|amount| *amount > MAX_IRRIGATION_PER_ZONE_MM + 1e-8)
        {
            per_plot_limit_violation_days += 1;
        }

        for zone in 0..ZONES {
            irrigation_by_zone[zone] += allocation[zone];
            let irrigated_storage = storage[zone] + allocation[zone];
            overflow_total += (irrigated_storage - STORAGE_CAP_MM).max(0.0);
            let after_use = (irrigated_storage.min(STORAGE_CAP_MM)
                - actual_demand_mm(day, zone, scenario))
            .max(0.0);
            let after_rain = (after_use + rain_mm(day, scenario)).clamp(0.0, STORAGE_CAP_MM);
            storage[zone] = after_rain;
            let zone_exposure = exposure(
                irrigated_storage,
                actual_demand_mm(day, zone, scenario),
                rain_mm(day, scenario),
                day,
                zone,
            );
            stress_exposure += zone_exposure;
            zone_stress_exposure[zone] += zone_exposure;
        }
    }

    let total_reads = sensor_reads + missing_reads;
    let mean_zone_irrigation = irrigation_by_zone.iter().sum::<f64>() / ZONES as f64;
    let variance_zone_irrigation = irrigation_by_zone
        .iter()
        .map(|value| (value - mean_zone_irrigation).powi(2))
        .sum::<f64>()
        / ZONES as f64;
    let max_zone_stress = zone_stress_exposure
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    let min_zone_stress = zone_stress_exposure
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let worst_to_best_zone_stress_ratio =
        (min_zone_stress > 1e-9).then_some(max_zone_stress / min_zone_stress);
    let policy_name = if matched_daily_totals.is_some() {
        WATER_MATCHED_POLICY_NAME
    } else {
        policy.name()
    };
    (
        ScenarioResult {
            scenario_id: scenario.id.to_owned(),
            policy: policy_name.to_owned(),
            drought_start_day: scenario.drought_start_day,
            heat_start_day: scenario.heat_start_day,
            forecast_bias_fraction: scenario.forecast_bias,
            sensor_dropout_period_days: scenario.sensor_dropout_period,
            simulated_days: DAYS,
            daily_water_budget_plot_mm: scenario.daily_water_budget_plot_mm,
            total_irrigation_applied_plot_mm: round3(irrigation_total),
            zone_irrigation_applied_plot_mm: irrigation_by_zone.map(round3),
            plot_allocation_dispersion_plot_mm: round3(variance_zone_irrigation.sqrt()),
            total_overflow_proxy_plot_mm: round3(overflow_total),
            stress_exposure_proxy: round3(stress_exposure),
            zone_stress_exposure_proxy: zone_stress_exposure.map(round3),
            max_zone_stress_exposure_proxy: round3(max_zone_stress),
            min_zone_stress_exposure_proxy: round3(min_zone_stress),
            worst_to_best_zone_stress_ratio: worst_to_best_zone_stress_ratio.map(round3),
            stress_delta_vs_reference_proxy: None,
            relative_stress_change_vs_reference_pct: None,
            relative_stress_change_vs_equal_share_pct: None,
            relative_stress_change_vs_matched_water_equal_share_pct: None,
            sensor_dropout_fraction: if total_reads == 0 {
                0.0
            } else {
                missing_reads as f64 / total_reads as f64
            },
            water_budget_violation_days: violation_days,
            per_plot_limit_violation_days,
        },
        daily_irrigation_trace,
    )
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

pub(crate) fn build_receipt() -> Result<WaterPolicyReceipt> {
    let all_scenarios = scenarios();
    let mut results = Vec::new();
    for scenario in all_scenarios {
        let (equal_share_result, _) = simulate_one(scenario, Policy::EqualShareCalendar, None);
        let (soil_threshold_result, _) = simulate_one(scenario, Policy::SoilThreshold, None);
        let (risk_aware_result, risk_daily_totals) =
            simulate_one(scenario, Policy::UncertaintyAware, None);
        let (minimax_fair_result, _) = simulate_one(scenario, Policy::GreedyMinimaxStress, None);
        let (matched_uniform_result, matched_daily_totals) = simulate_one(
            scenario,
            Policy::EqualShareCalendar,
            Some(&risk_daily_totals),
        );
        let dose_diff = risk_aware_result.total_irrigation_applied_plot_mm
            - matched_uniform_result.total_irrigation_applied_plot_mm;
        if dose_diff.abs() > 1e-3 {
            return Err(anyhow!(
                "matched-water counterfactual did not preserve daily dose trace for {}",
                scenario.id
            ));
        }
        if risk_daily_totals
            .iter()
            .zip(matched_daily_totals)
            .any(|(risk, matched)| (risk - matched).abs() > 1e-8)
        {
            return Err(anyhow!(
                "matched-water counterfactual daily dose mismatch for {}",
                scenario.id
            ));
        }
        results.push(equal_share_result);
        results.push(soil_threshold_result);
        results.push(risk_aware_result);
        results.push(minimax_fair_result);
        results.push(matched_uniform_result);
    }

    let baseline_scores: BTreeMap<String, f64> = results
        .iter()
        .filter(|result| result.policy == Policy::EqualShareCalendar.name())
        .map(|result| (result.scenario_id.clone(), result.stress_exposure_proxy))
        .collect();
    for result in &mut results {
        if result.policy != Policy::EqualShareCalendar.name() {
            if let Some(baseline) = baseline_scores.get(&result.scenario_id).copied() {
                if baseline > 1e-9 {
                    result.relative_stress_change_vs_equal_share_pct = Some(round3(
                        (result.stress_exposure_proxy - baseline) / baseline * 100.0,
                    ));
                }
            }
        }
    }
    let matched_uniform_scores: BTreeMap<String, f64> = results
        .iter()
        .filter(|result| result.policy == WATER_MATCHED_POLICY_NAME)
        .map(|result| (result.scenario_id.clone(), result.stress_exposure_proxy))
        .collect();
    for result in &mut results {
        if result.policy == Policy::UncertaintyAware.name() {
            if let Some(matched) = matched_uniform_scores.get(&result.scenario_id).copied() {
                if matched > 1e-9 {
                    result.relative_stress_change_vs_matched_water_equal_share_pct = Some(round3(
                        (result.stress_exposure_proxy - matched) / matched * 100.0,
                    ));
                }
            }
        }
    }

    let reference_scores: BTreeMap<String, f64> = results
        .iter()
        .filter(|result| result.scenario_id == "reference")
        .map(|result| (result.policy.clone(), result.stress_exposure_proxy))
        .collect();
    for result in &mut results {
        if result.scenario_id != "reference" {
            if let Some(reference) = reference_scores.get(&result.policy).copied() {
                let delta = result.stress_exposure_proxy - reference;
                result.stress_delta_vs_reference_proxy = Some(round3(delta));
                if reference > 1e-9 {
                    result.relative_stress_change_vs_reference_pct =
                        Some(round3(delta / reference * 100.0));
                }
            }
        }
    }

    let policy_names = [
        Policy::EqualShareCalendar.name(),
        Policy::SoilThreshold.name(),
        Policy::UncertaintyAware.name(),
        Policy::GreedyMinimaxStress.name(),
        WATER_MATCHED_POLICY_NAME,
    ];
    let mut aggregates = Vec::new();
    for policy_name in policy_names {
        let policy_results: Vec<&ScenarioResult> = results
            .iter()
            .filter(|result| result.policy == policy_name)
            .collect();
        let mut scores: Vec<f64> = policy_results
            .iter()
            .map(|result| result.stress_exposure_proxy)
            .collect();
        scores.sort_by(f64::total_cmp);
        let mean_stress = scores.iter().sum::<f64>() / scores.len() as f64;
        let median_stress = (scores[(scores.len() - 1) / 2] + scores[scores.len() / 2]) / 2.0;
        let mut worst_zone_scores: Vec<f64> = policy_results
            .iter()
            .map(|result| result.max_zone_stress_exposure_proxy)
            .collect();
        worst_zone_scores.sort_by(f64::total_cmp);
        let median_worst_zone = (worst_zone_scores[(worst_zone_scores.len() - 1) / 2]
            + worst_zone_scores[worst_zone_scores.len() / 2])
            / 2.0;
        let defined_worst_to_best_ratios: Vec<f64> = policy_results
            .iter()
            .filter_map(|result| result.worst_to_best_zone_stress_ratio)
            .collect();
        let worst_to_best_ratio_defined_scenario_count = defined_worst_to_best_ratios.len();
        let mean_worst_to_best_ratio = if defined_worst_to_best_ratios.is_empty() {
            None
        } else {
            Some(round3(
                defined_worst_to_best_ratios.iter().sum::<f64>()
                    / defined_worst_to_best_ratios.len() as f64,
            ))
        };
        let relative_changes: Vec<f64> = policy_results
            .iter()
            .filter_map(|result| result.relative_stress_change_vs_equal_share_pct)
            .collect();
        let mean_relative_change = if relative_changes.is_empty() {
            None
        } else {
            Some(round3(
                relative_changes.iter().sum::<f64>() / relative_changes.len() as f64,
            ))
        };
        let relative_matched_changes: Vec<f64> = policy_results
            .iter()
            .filter_map(|result| result.relative_stress_change_vs_matched_water_equal_share_pct)
            .collect();
        let mean_relative_matched_change = if relative_matched_changes.is_empty() {
            None
        } else {
            Some(round3(
                relative_matched_changes.iter().sum::<f64>()
                    / relative_matched_changes.len() as f64,
            ))
        };
        aggregates.push(PolicyAggregate {
            policy: policy_name.to_owned(),
            scenario_count: policy_results.len(),
            worst_to_best_ratio_defined_scenario_count,
            mean_stress_exposure_proxy: round3(mean_stress),
            median_stress_exposure_proxy: round3(median_stress),
            total_irrigation_applied_plot_mm: round3(
                policy_results
                    .iter()
                    .map(|result| result.total_irrigation_applied_plot_mm)
                    .sum(),
            ),
            mean_plot_allocation_dispersion_plot_mm: round3(
                policy_results
                    .iter()
                    .map(|result| result.plot_allocation_dispersion_plot_mm)
                    .sum::<f64>()
                    / policy_results.len() as f64,
            ),
            median_worst_zone_stress_exposure_proxy: round3(median_worst_zone),
            max_worst_zone_stress_exposure_proxy: worst_zone_scores
                .iter()
                .copied()
                .max_by(f64::total_cmp)
                .map(round3)
                .unwrap_or(0.0),
            mean_worst_to_best_zone_stress_ratio: mean_worst_to_best_ratio,
            mean_relative_stress_change_vs_equal_share_pct: mean_relative_change,
            mean_relative_stress_change_vs_matched_water_equal_share_pct:
                mean_relative_matched_change,
            total_water_budget_violation_days: policy_results
                .iter()
                .map(|result| result.water_budget_violation_days)
                .sum(),
            total_per_plot_limit_violation_days: policy_results
                .iter()
                .map(|result| result.per_plot_limit_violation_days)
                .sum(),
        });
    }

    let budget_violations = results
        .iter()
        .map(|result| result.water_budget_violation_days)
        .sum::<usize>();
    let per_plot_violations = results
        .iter()
        .map(|result| result.per_plot_limit_violation_days)
        .sum::<usize>();
    if budget_violations != 0 || per_plot_violations != 0 {
        return Err(anyhow!(
            "synthetic water-policy invariant failed: {budget_violations} daily budget violations, {per_plot_violations} per-plot limit violations"
        ));
    }
    if results.iter().any(|result| {
        !result.stress_exposure_proxy.is_finite()
            || !result.total_irrigation_applied_plot_mm.is_finite()
            || !result.plot_allocation_dispersion_plot_mm.is_finite()
            || !result
                .zone_stress_exposure_proxy
                .iter()
                .all(|value| value.is_finite())
            || !result
                .zone_irrigation_applied_plot_mm
                .iter()
                .all(|value| value.is_finite())
            || !result.max_zone_stress_exposure_proxy.is_finite()
            || !result.min_zone_stress_exposure_proxy.is_finite()
            || !result.sensor_dropout_fraction.is_finite()
    }) {
        return Err(anyhow!(
            "synthetic water-policy output contains non-finite metrics"
        ));
    }

    Ok(WaterPolicyReceipt {
        schema_version: "cropfoundry-water-policy-receipt-v2".to_owned(),
        benchmark_id: "CF-022".to_owned(),
        execution_mode: "deterministic_synthetic_algorithm_harness".to_owned(),
        budget_measurement_unit: "plot-mm sum; not physical volume".to_owned(),
        synthetic_only: true,
        biological_performance_claim: false,
        training_authorized: false,
        scenario_count: all_scenarios.len(),
        policy_count: policy_names.len(),
        invariant_summary:
            "Zero daily-budget and per-plot-limit violations; unmet-demand margin is scored below empty storage."
                .to_owned(),
        aggregates,
        results,
        caveats: vec![
            "Scenario traces and parameters are illustrative synthetic assumptions.".to_owned(),
            "The simulator separately fails on daily-total water-budget and per-plot actuator-limit violations.".to_owned(),
            "The stress proxy is not calibrated to a crop or farm economics.".to_owned(),
            "Aggregate scores hide uneven burdens; inspect per-plot stress and irrigation."
                .to_owned(),
            "The water-matched uniform policy holds the risk-aware daily dose trace constant and rotates discrete units across plots.".to_owned(),
            "Lower scores do not establish field water savings or yield gains.".to_owned(),
            "This tests deterministic execution and hard water accounting only.".to_owned(),
            "Field use requires calibration, farmer review, and replicated trials.".to_owned(),
        ],
    })
}

/// Run a deterministic policy harness; never accesses external datasets or authorizes training.
pub fn run(out: Option<PathBuf>) -> Result<()> {
    let receipt = build_receipt()?;
    let rendered = serde_json::to_string_pretty(&receipt)?;
    println!("{rendered}");
    if let Some(path) = out {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| {
                anyhow!(
                    "cannot create output directory {}: {error}",
                    parent.display()
                )
            })?;
        }
        fs::write(&path, format!("{rendered}\n"))
            .map_err(|error| anyhow!("cannot write receipt {}: {error}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cropfoundry_water_policy_harness_is_deterministic() {
        let first = serde_json::to_string(&build_receipt().unwrap()).unwrap();
        let second = serde_json::to_string(&build_receipt().unwrap()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn cropfoundry_water_policy_has_all_expected_scenarios_and_baselines() {
        let receipt = build_receipt().unwrap();
        assert_eq!(
            receipt.schema_version,
            "cropfoundry-water-policy-receipt-v2"
        );
        assert_eq!(receipt.benchmark_id, "CF-022");
        assert_eq!(receipt.scenario_count, 6);
        assert_eq!(receipt.policy_count, 5);
        assert_eq!(receipt.results.len(), 30);
        assert!(receipt.results.iter().any(|result| {
            result.scenario_id == "forecast_bias_only"
                && result.forecast_bias_fraction < 0.0
                && result.sensor_dropout_period_days.is_none()
                && result.drought_start_day.is_none()
                && result.heat_start_day.is_none()
        }));
        assert!(receipt.results.iter().any(|result| {
            result.scenario_id == "sensor_dropout_only"
                && result.sensor_dropout_period_days == Some(7)
                && result.forecast_bias_fraction == 0.0
                && result.sensor_dropout_fraction > 0.0
        }));
        assert_eq!(receipt.aggregates.len(), 5);
        assert!(
            receipt
                .aggregates
                .iter()
                .all(|aggregate| aggregate.scenario_count == 6)
        );
        assert!(
            receipt
                .results
                .iter()
                .all(|result| result.water_budget_violation_days == 0)
        );
        assert!(
            receipt
                .results
                .iter()
                .all(|result| result.per_plot_limit_violation_days == 0)
        );
    }

    #[test]
    fn cropfoundry_failure_ablation_scenarios_change_only_the_declared_factor() {
        let receipt = build_receipt().unwrap();
        for policy_result in receipt
            .results
            .iter()
            .filter(|result| result.scenario_id == "water_rationing_only")
        {
            let reference = receipt
                .results
                .iter()
                .find(|result| {
                    result.scenario_id == "reference" && result.policy == policy_result.policy
                })
                .unwrap();
            assert_ne!(
                policy_result.daily_water_budget_plot_mm,
                reference.daily_water_budget_plot_mm
            );
            assert_eq!(policy_result.drought_start_day, reference.drought_start_day);
            assert_eq!(policy_result.heat_start_day, reference.heat_start_day);
            assert_eq!(
                policy_result.forecast_bias_fraction,
                reference.forecast_bias_fraction
            );
            assert_eq!(
                policy_result.sensor_dropout_period_days,
                reference.sensor_dropout_period_days
            );
        }
        for policy_result in receipt
            .results
            .iter()
            .filter(|result| result.scenario_id == "forecast_bias_only")
        {
            let reference = receipt
                .results
                .iter()
                .find(|result| {
                    result.scenario_id == "reference" && result.policy == policy_result.policy
                })
                .unwrap();
            assert_eq!(
                policy_result.daily_water_budget_plot_mm,
                reference.daily_water_budget_plot_mm
            );
            assert_eq!(policy_result.drought_start_day, reference.drought_start_day);
            assert_eq!(policy_result.heat_start_day, reference.heat_start_day);
            assert_eq!(
                policy_result.sensor_dropout_period_days,
                reference.sensor_dropout_period_days
            );
            assert_ne!(
                policy_result.forecast_bias_fraction,
                reference.forecast_bias_fraction
            );
        }
        for policy_result in receipt
            .results
            .iter()
            .filter(|result| result.scenario_id == "sensor_dropout_only")
        {
            let reference = receipt
                .results
                .iter()
                .find(|result| {
                    result.scenario_id == "reference" && result.policy == policy_result.policy
                })
                .unwrap();
            assert_eq!(
                policy_result.daily_water_budget_plot_mm,
                reference.daily_water_budget_plot_mm
            );
            assert_eq!(policy_result.drought_start_day, reference.drought_start_day);
            assert_eq!(policy_result.heat_start_day, reference.heat_start_day);
            assert_eq!(
                policy_result.forecast_bias_fraction,
                reference.forecast_bias_fraction
            );
            assert_ne!(
                policy_result.sensor_dropout_period_days,
                reference.sensor_dropout_period_days
            );
        }
    }

    #[test]
    fn cropfoundry_stress_proxy_preserves_signal_below_empty_storage() {
        let day = 22;
        let zone = 1;
        let fully_starved = exposure(0.0, 4.0, 0.0, day, zone);
        let slightly_more_water = exposure(ALLOCATION_STEP_MM, 4.0, 0.0, day, zone);
        let demand_covered = exposure(4.0, 4.0, 0.0, day, zone);
        assert!(slightly_more_water < fully_starved);
        assert!(demand_covered < slightly_more_water);
    }

    #[test]
    fn cropfoundry_minimax_policy_reports_all_zone_constraints_and_metrics() {
        let receipt = build_receipt().unwrap();
        for scenario in scenarios() {
            let fair = receipt
                .results
                .iter()
                .find(|result| {
                    result.scenario_id == scenario.id
                        && result.policy == Policy::GreedyMinimaxStress.name()
                })
                .unwrap();
            assert_eq!(fair.water_budget_violation_days, 0);
            assert_eq!(fair.per_plot_limit_violation_days, 0);
            assert_eq!(fair.zone_stress_exposure_proxy.len(), ZONES);
            assert_eq!(fair.zone_irrigation_applied_plot_mm.len(), ZONES);
            assert!(fair.max_zone_stress_exposure_proxy.is_finite());
            assert!(fair.relative_stress_change_vs_equal_share_pct.is_some());
        }
    }

    #[test]
    fn cropfoundry_matched_water_baseline_preserves_each_daily_dose() {
        let receipt = build_receipt().unwrap();
        for scenario in scenarios() {
            let risk = receipt
                .results
                .iter()
                .find(|result| {
                    result.scenario_id == scenario.id
                        && result.policy == Policy::UncertaintyAware.name()
                })
                .unwrap();
            let matched = receipt
                .results
                .iter()
                .find(|result| {
                    result.scenario_id == scenario.id && result.policy == WATER_MATCHED_POLICY_NAME
                })
                .unwrap();
            assert_eq!(
                risk.total_irrigation_applied_plot_mm,
                matched.total_irrigation_applied_plot_mm
            );
            assert_eq!(matched.water_budget_violation_days, 0);
            assert_eq!(matched.per_plot_limit_violation_days, 0);
            assert!(
                matched
                    .relative_stress_change_vs_matched_water_equal_share_pct
                    .is_none()
            );
            assert!(
                risk.relative_stress_change_vs_matched_water_equal_share_pct
                    .is_some()
            );
            assert!(matched.plot_allocation_dispersion_plot_mm <= 0.2);
        }
    }

    #[test]
    fn cropfoundry_policy_results_include_paired_reference_deltas() {
        let receipt = build_receipt().unwrap();
        assert!(receipt.results.iter().all(|result| {
            if result.scenario_id == "reference" {
                result.stress_delta_vs_reference_proxy.is_none()
                    && result.relative_stress_change_vs_reference_pct.is_none()
            } else {
                result.stress_delta_vs_reference_proxy.is_some()
            }
        }));
    }

    #[test]
    fn cropfoundry_equal_share_baseline_has_zero_allocation_dispersion() {
        let receipt = build_receipt().unwrap();
        assert!(
            receipt
                .results
                .iter()
                .filter(|result| result.policy == Policy::EqualShareCalendar.name())
                .all(|result| result.plot_allocation_dispersion_plot_mm <= 1e-6)
        );
    }

    #[test]
    fn cropfoundry_disparity_ratio_denominator_is_explicit() {
        let receipt = build_receipt().unwrap();
        for aggregate in &receipt.aggregates {
            assert!(
                aggregate.worst_to_best_ratio_defined_scenario_count <= aggregate.scenario_count
            );
            assert_eq!(
                aggregate.mean_worst_to_best_zone_stress_ratio.is_some(),
                aggregate.worst_to_best_ratio_defined_scenario_count > 0
            );
            let independently_counted = receipt
                .results
                .iter()
                .filter(|result| {
                    result.policy == aggregate.policy
                        && result.worst_to_best_zone_stress_ratio.is_some()
                })
                .count();
            assert_eq!(
                aggregate.worst_to_best_ratio_defined_scenario_count,
                independently_counted
            );
        }
        for result in &receipt.results {
            assert_eq!(
                result.worst_to_best_zone_stress_ratio.is_some(),
                result.min_zone_stress_exposure_proxy > 1e-9
            );
        }
    }

    #[test]
    fn cropfoundry_water_policy_reports_comparable_baseline_deltas() {
        let receipt = build_receipt().unwrap();
        let baseline = receipt
            .results
            .iter()
            .filter(|result| result.policy == Policy::EqualShareCalendar.name())
            .collect::<Vec<_>>();
        assert_eq!(baseline.len(), 6);
        assert!(
            baseline
                .iter()
                .all(|result| result.relative_stress_change_vs_equal_share_pct.is_none())
        );
        assert!(receipt.results.iter().all(|result| {
            result
                .relative_stress_change_vs_equal_share_pct
                .is_none_or(f64::is_finite)
        }));
    }

    #[test]
    fn cropfoundry_policy_receipt_exposes_worst_plot_stress_and_allocation() {
        let receipt = build_receipt().unwrap();
        for result in &receipt.results {
            let zone_stress_sum = result.zone_stress_exposure_proxy.iter().sum::<f64>();
            let zone_irrigation_sum = result.zone_irrigation_applied_plot_mm.iter().sum::<f64>();
            assert!((zone_stress_sum - result.stress_exposure_proxy).abs() <= 0.01);
            assert!((zone_irrigation_sum - result.total_irrigation_applied_plot_mm).abs() <= 0.01);
            assert!(
                result
                    .zone_stress_exposure_proxy
                    .iter()
                    .all(|value| *value >= 0.0)
            );
            assert!(
                result
                    .zone_irrigation_applied_plot_mm
                    .iter()
                    .all(|value| *value >= 0.0)
            );
            assert!(result.max_zone_stress_exposure_proxy >= result.min_zone_stress_exposure_proxy);
            assert_eq!(
                result.max_zone_stress_exposure_proxy,
                result
                    .zone_stress_exposure_proxy
                    .iter()
                    .copied()
                    .max_by(f64::total_cmp)
                    .unwrap()
            );
        }
        assert!(receipt.aggregates.iter().all(|aggregate| {
            aggregate.max_worst_zone_stress_exposure_proxy
                >= aggregate.median_worst_zone_stress_exposure_proxy
                && aggregate.total_per_plot_limit_violation_days == 0
                && aggregate.worst_to_best_ratio_defined_scenario_count <= aggregate.scenario_count
                && (aggregate.mean_worst_to_best_zone_stress_ratio.is_some()
                    == (aggregate.worst_to_best_ratio_defined_scenario_count > 0))
        }));
    }

    #[test]
    fn cropfoundry_water_policy_is_never_a_biological_claim() {
        let receipt = build_receipt().unwrap();
        assert!(receipt.synthetic_only);
        assert!(!receipt.biological_performance_claim);
        assert!(!receipt.training_authorized);
        assert!(
            receipt
                .caveats
                .iter()
                .any(|note| note.contains("not calibrated to a crop"))
        );
    }

    #[test]
    fn cropfoundry_water_policy_metrics_are_finite_and_bounded() {
        let receipt = build_receipt().unwrap();
        for result in receipt.results {
            assert!(result.stress_exposure_proxy.is_finite());
            assert!(result.total_irrigation_applied_plot_mm.is_finite());
            assert!(result.plot_allocation_dispersion_plot_mm.is_finite());
            assert!(result.plot_allocation_dispersion_plot_mm >= 0.0);
            assert!((0.0..=1.0).contains(&result.sensor_dropout_fraction));
            assert!(
                result.total_irrigation_applied_plot_mm
                    <= result.simulated_days as f64 * result.daily_water_budget_plot_mm + 1e-6
            );
        }
    }
}
