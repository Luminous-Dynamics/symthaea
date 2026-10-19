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

#[derive(Clone, Copy)]
struct Scenario {
    id: &'static str,
    daily_water_cap_mm: f64,
    drought_start_day: Option<usize>,
    heat_start_day: Option<usize>,
    forecast_bias: f64,
    sensor_dropout_period: Option<usize>,
}

fn scenarios() -> [Scenario; 4] {
    [
        Scenario {
            id: "reference",
            daily_water_cap_mm: 6.0,
            drought_start_day: None,
            heat_start_day: None,
            forecast_bias: 0.0,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "water_rationing",
            daily_water_cap_mm: 4.5,
            drought_start_day: Some(10),
            heat_start_day: None,
            forecast_bias: -0.10,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "heat_and_drought",
            daily_water_cap_mm: 5.0,
            drought_start_day: Some(8),
            heat_start_day: Some(18),
            forecast_bias: -0.20,
            sensor_dropout_period: None,
        },
        Scenario {
            id: "forecast_bias_and_sensor_dropout",
            daily_water_cap_mm: 5.5,
            drought_start_day: Some(12),
            heat_start_day: Some(24),
            forecast_bias: -0.15,
            sensor_dropout_period: Some(7),
        },
    ]
}

#[derive(Clone, Copy)]
enum Policy {
    EqualShareCalendar,
    SoilThreshold,
    UncertaintyAware,
}

impl Policy {
    fn name(self) -> &'static str {
        match self {
            Self::EqualShareCalendar => "equal_share_calendar",
            Self::SoilThreshold => "soil_threshold",
            Self::UncertaintyAware => "uncertainty_aware_marginal_risk",
        }
    }
}

#[derive(Debug, Serialize, Clone)]
struct ScenarioResult {
    scenario_id: String,
    policy: String,
    simulated_days: usize,
    daily_water_cap_mm: f64,
    total_irrigation_applied_mm: f64,
    total_overflow_proxy_mm: f64,
    stress_exposure_proxy: f64,
    relative_stress_change_vs_equal_share_pct: Option<f64>,
    sensor_dropout_fraction: f64,
    water_budget_violation_days: usize,
}

#[derive(Debug, Serialize)]
struct PolicyAggregate {
    policy: String,
    scenario_count: usize,
    mean_stress_exposure_proxy: f64,
    median_stress_exposure_proxy: f64,
    total_irrigation_applied_mm: f64,
    mean_relative_stress_change_vs_equal_share_pct: Option<f64>,
    total_water_budget_violation_days: usize,
}

#[derive(Debug, Serialize)]
struct WaterPolicyReceipt {
    schema_version: String,
    benchmark_id: String,
    execution_mode: String,
    synthetic_only: bool,
    biological_performance_claim: bool,
    training_authorized: bool,
    scenario_count: usize,
    policy_count: usize,
    invariant_summary: String,
    aggregates: Vec<PolicyAggregate>,
    results: Vec<ScenarioResult>,
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
    let end_storage = (storage_mm - demand_mm + rain_mm).clamp(0.0, STORAGE_CAP_MM);
    let deficit = (minimum_storage_mm(day, zone) - end_storage).max(0.0);
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
            let share =
                (scenario.daily_water_cap_mm / ZONES as f64).min(MAX_IRRIGATION_PER_ZONE_MM);
            allocation.fill(share);
        }
        Policy::SoilThreshold => {
            let units = (scenario.daily_water_cap_mm / ALLOCATION_STEP_MM).floor() as usize;
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
            let units = (scenario.daily_water_cap_mm / ALLOCATION_STEP_MM).floor() as usize;
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
    }
    allocation
}

fn simulate_one(scenario: Scenario, policy: Policy) -> ScenarioResult {
    let mut storage = [6.0; ZONES];
    let mut last_observation = storage;
    let mut stale_age = [0_usize; ZONES];
    let mut irrigation_total = 0.0;
    let mut overflow_total = 0.0;
    let mut stress_exposure = 0.0;
    let mut sensor_reads = 0_usize;
    let mut missing_reads = 0_usize;
    let mut violation_days = 0_usize;

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

        let allocation = allocate_water(policy, observed, stale_age, day, scenario);
        let applied: f64 = allocation.iter().sum();
        irrigation_total += applied;
        if applied > scenario.daily_water_cap_mm + 1e-8 {
            violation_days += 1;
        }

        for zone in 0..ZONES {
            let irrigated_storage = storage[zone] + allocation[zone];
            overflow_total += (irrigated_storage - STORAGE_CAP_MM).max(0.0);
            let after_use = (irrigated_storage.min(STORAGE_CAP_MM)
                - actual_demand_mm(day, zone, scenario))
            .max(0.0);
            let after_rain = (after_use + rain_mm(day, scenario)).clamp(0.0, STORAGE_CAP_MM);
            storage[zone] = after_rain;
            stress_exposure += exposure(after_rain, 0.0, 0.0, day, zone);
        }
    }

    let total_reads = sensor_reads + missing_reads;
    ScenarioResult {
        scenario_id: scenario.id.to_owned(),
        policy: policy.name().to_owned(),
        simulated_days: DAYS,
        daily_water_cap_mm: scenario.daily_water_cap_mm,
        total_irrigation_applied_mm: round3(irrigation_total),
        total_overflow_proxy_mm: round3(overflow_total),
        stress_exposure_proxy: round3(stress_exposure),
        relative_stress_change_vs_equal_share_pct: None,
        sensor_dropout_fraction: if total_reads == 0 {
            0.0
        } else {
            missing_reads as f64 / total_reads as f64
        },
        water_budget_violation_days: violation_days,
    }
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

fn build_receipt() -> Result<WaterPolicyReceipt> {
    let all_scenarios = scenarios();
    let policies = [
        Policy::EqualShareCalendar,
        Policy::SoilThreshold,
        Policy::UncertaintyAware,
    ];
    let mut results = Vec::new();
    for scenario in all_scenarios {
        for policy in policies {
            results.push(simulate_one(scenario, policy));
        }
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
                    result.relative_stress_change_vs_equal_share_pct =
                        Some(round3((result.stress_exposure_proxy - baseline) / baseline * 100.0));
                }
            }
        }
    }
    let mut aggregates = Vec::new();
    for policy in [
        Policy::EqualShareCalendar,
        Policy::SoilThreshold,
        Policy::UncertaintyAware,
    ] {
        let policy_results: Vec<&ScenarioResult> = results
            .iter()
            .filter(|result| result.policy == policy.name())
            .collect();
        let mut scores: Vec<f64> = policy_results
            .iter()
            .map(|result| result.stress_exposure_proxy)
            .collect();
        scores.sort_by(f64::total_cmp);
        let mean_stress = scores.iter().sum::<f64>() / scores.len() as f64;
        let median_stress = (scores[(scores.len() - 1) / 2] + scores[scores.len() / 2]) / 2.0;
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
        aggregates.push(PolicyAggregate {
            policy: policy.name().to_owned(),
            scenario_count: policy_results.len(),
            mean_stress_exposure_proxy: round3(mean_stress),
            median_stress_exposure_proxy: round3(median_stress),
            total_irrigation_applied_mm: round3(
                policy_results
                    .iter()
                    .map(|result| result.total_irrigation_applied_mm)
                    .sum(),
            ),
            mean_relative_stress_change_vs_equal_share_pct: mean_relative_change,
            total_water_budget_violation_days: policy_results
                .iter()
                .map(|result| result.water_budget_violation_days)
                .sum(),
        });
    }
    let violations = results
        .iter()
        .map(|result| result.water_budget_violation_days)
        .sum::<usize>();
    if violations != 0 {
        return Err(anyhow!(
            "synthetic water-policy invariant failed: {violations} daily budget violations"
        ));
    }
    if results.iter().any(|result| {
        !result.stress_exposure_proxy.is_finite()
            || !result.total_irrigation_applied_mm.is_finite()
            || !result.sensor_dropout_fraction.is_finite()
    }) {
        return Err(anyhow!(
            "synthetic water-policy output contains non-finite metrics"
        ));
    }

    Ok(WaterPolicyReceipt {
        schema_version: "cropfoundry-water-policy-receipt-v0".to_owned(),
        benchmark_id: "CF-022".to_owned(),
        execution_mode: "deterministic_synthetic_algorithm_harness".to_owned(),
        synthetic_only: true,
        biological_performance_claim: false,
        training_authorized: false,
        scenario_count: all_scenarios.len(),
        policy_count: policies.len(),
        invariant_summary: "Zero cap violations; synthetic proxy, not a plant model.".to_owned(),
        aggregates,
        results,
        caveats: vec![
            "Scenario traces and parameters are illustrative synthetic assumptions.".to_owned(),
            "The stress proxy is not calibrated to a crop or farm economics.".to_owned(),
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
        assert_eq!(receipt.benchmark_id, "CF-022");
        assert_eq!(receipt.scenario_count, 4);
        assert_eq!(receipt.policy_count, 3);
        assert_eq!(receipt.results.len(), 12);
        assert_eq!(receipt.aggregates.len(), 3);
        assert!(receipt.aggregates.iter().all(|aggregate| aggregate.scenario_count == 4));
        assert!(
            receipt
                .results
                .iter()
                .all(|result| result.water_budget_violation_days == 0)
        );
    }

    #[test]
    fn cropfoundry_water_policy_reports_comparable_baseline_deltas() {
        let receipt = build_receipt().unwrap();
        let baseline = receipt
            .results
            .iter()
            .filter(|result| result.policy == Policy::EqualShareCalendar.name())
            .collect::<Vec<_>>();
        assert_eq!(baseline.len(), 4);
        assert!(baseline
            .iter()
            .all(|result| result.relative_stress_change_vs_equal_share_pct.is_none()));
        assert!(receipt.results.iter().all(|result| {
            result
                .relative_stress_change_vs_equal_share_pct
                .is_none_or(f64::is_finite)
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
            assert!(result.total_irrigation_applied_mm.is_finite());
            assert!((0.0..=1.0).contains(&result.sensor_dropout_fraction));
            assert!(
                result.total_irrigation_applied_mm
                    <= result.simulated_days as f64 * result.daily_water_cap_mm + 1e-6
            );
        }
    }
}
