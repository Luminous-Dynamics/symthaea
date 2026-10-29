use symthaea_operations_research::aviation_energy_accounting::{EvidenceClass, TransportService};
use symthaea_operations_research::aviation_energy_hub::{
    seal_energy_hub_scenario, solve_exact_energy_hub_pareto, EnergyHubError, EnergyHubInterval,
    EnergyStorageSpec, TimeIndexedEnergyHubScenario,
};
use symthaea_operations_research::aviation_energy_hub_paired_benchmark::{
    evaluate_paired_rolling_benchmark, PairedBenchmarkError, WeightedRollingEpisode,
};
use symthaea_operations_research::aviation_energy_hub_rolling::{
    ForecastWindow, RollingHorizonEpisode,
};

fn storage() -> EnergyStorageSpec {
    EnergyStorageSpec {
        capacity_quanta: 1,
        initial_soc_quanta: 1,
        initial_soc_ledger_ref: "synthetic:external-opening-inventory-ledger-v1".to_owned(),
        max_charge_input_quanta_per_interval: 1,
        max_discharge_output_quanta_per_interval: 1,
        charge_efficiency_ppm: 1_000_000,
        discharge_efficiency_ppm: 1_000_000,
        minimum_terminal_soc_quanta: 0,
    }
}

fn interval(
    id: &str,
    demand: u32,
    grid_capacity: u32,
    grid_cost: f64,
) -> EnergyHubInterval {
    EnergyHubInterval {
        id: id.to_owned(),
        demand_quanta: demand,
        charger_capacity_quanta: demand,
        renewable_generation_quanta: 0,
        grid_import_capacity_quanta: grid_capacity,
        renewable_cost_per_quantum: 0.0,
        renewable_lifecycle_co2e_kg_per_quantum: 0.0,
        renewable_evidence_ref: format!("synthetic:renewable:{id}"),
        renewable_evidence_class: EvidenceClass::Synthetic,
        grid_cost_per_quantum: grid_cost,
        grid_lifecycle_co2e_kg_per_quantum: 10.0 * grid_cost,
        grid_evidence_ref: format!("synthetic:grid:{id}"),
        grid_evidence_class: EvidenceClass::Synthetic,
    }
}

fn scenario(
    scenario_id: &str,
    first_grid_capacity: u32,
    second_grid_cost: f64,
) -> TimeIndexedEnergyHubScenario {
    let mut value = TimeIndexedEnergyHubScenario {
        scenario_id: scenario_id.to_owned(),
        input_digest: "pending".to_owned(),
        currency_code: "USD".to_owned(),
        quantum_kwh: 1.0,
        service: TransportService {
            period_id: "paired-panel-period-v1".to_owned(),
            lifecycle_boundary_id: "source-to-load-v1".to_owned(),
            completed_passenger_km: 100.0,
            available_seat_km: 120.0,
        },
        storage: storage(),
        intervals: vec![
            interval("t0", 1, first_grid_capacity, 1.0),
            interval("t1", 1, 1, second_grid_cost),
        ],
    };
    seal_energy_hub_scenario(&mut value).unwrap();
    value
}

fn episode(realized: TimeIndexedEnergyHubScenario) -> RollingHorizonEpisode {
    RollingHorizonEpisode {
        realized_scenario: realized,
        decision_times_epoch: vec![100, 200],
        observation_available_at_epoch: vec![90, 190],
        forecasts_by_stage: vec![
            ForecastWindow {
                issued_at_epoch: 70,
                available_at_epoch: 80,
                future_intervals: vec![interval("t1", 1, 1, 10.0)],
            },
            ForecastWindow {
                issued_at_epoch: 150,
                available_at_epoch: 160,
                future_intervals: vec![],
            },
        ],
    }
}

fn panel() -> (TimeIndexedEnergyHubScenario, Vec<WeightedRollingEpisode>) {
    let baseline = scenario("nominal-forecast-plan-v1", 1, 10.0);
    let regular = episode(scenario("actual-cheap-future-v1", 1, 0.1));
    let outage = episode(scenario("actual-first-slot-grid-outage-v1", 0, 0.1));

    (
        baseline,
        vec![
            WeightedRollingEpisode {
                weight_ppm: 500_000,
                episode: regular,
            },
            WeightedRollingEpisode {
                weight_ppm: 500_000,
                episode: outage,
            },
        ],
    )
}

#[test]
fn paired_panel_reports_coverage_regret_and_static_failure_separately() {
    let (baseline, episodes) = panel();
    let plan = solve_exact_energy_hub_pareto(&baseline, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let report =
        evaluate_paired_rolling_benchmark(&baseline, &plan, &episodes, 100_000).unwrap();

    assert_eq!(report.outcomes.len(), 2);
    assert_eq!(report.static_plan_summary.feasible_weight_ppm, 500_000);
    assert_eq!(report.static_plan_summary.infeasible_weight_ppm, 500_000);
    assert_eq!(report.static_plan_summary.unresolved_weight_ppm, 0);

    assert_eq!(report.rolling_horizon_summary.feasible_weight_ppm, 1_000_000);
    assert_eq!(report.rolling_horizon_summary.infeasible_weight_ppm, 0);
    assert_eq!(report.rolling_horizon_summary.unresolved_weight_ppm, 0);

    assert_eq!(report.perfect_information_summary.feasible_weight_ppm, 1_000_000);
    assert_eq!(report.rolling_regret_vs_perfect_information.comparable_weight_ppm, 1_000_000);
    assert_eq!(
        report.rolling_regret_vs_perfect_information.mean_cost_delta,
        Some(0.45)
    );
    assert_eq!(
        report.static_minus_rolling_cost.comparable_weight_ppm,
        500_000
    );

    let outage = report
        .outcomes
        .iter()
        .find(|outcome| outcome.scenario_id == "actual-first-slot-grid-outage-v1")
        .unwrap();
    assert_eq!(
        outage.static_plan_error,
        Some(EnergyHubError::ResourceBalanceViolation)
    );
    assert!(outage.rolling_run.evaluated_realized_plan.is_some());
}

#[test]
fn paired_report_is_deterministic_under_scenario_input_order() {
    let (baseline, mut episodes) = panel();
    let plan = solve_exact_energy_hub_pareto(&baseline, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let forward = evaluate_paired_rolling_benchmark(&baseline, &plan, &episodes, 100_000).unwrap();
    episodes.reverse();
    let reverse = evaluate_paired_rolling_benchmark(&baseline, &plan, &episodes, 100_000).unwrap();
    assert_eq!(forward, reverse);
    assert!(forward.benchmark_digest.starts_with("sha256:"));
}

#[test]
fn weights_must_sum_to_one_million_and_ids_must_be_unique() {
    let (baseline, mut episodes) = panel();
    let plan = solve_exact_energy_hub_pareto(&baseline, 100_000).unwrap().frontier[0]
        .plan
        .clone();

    episodes[0].weight_ppm = 400_000;
    episodes[1].weight_ppm = 400_000;
    assert_eq!(
        evaluate_paired_rolling_benchmark(&baseline, &plan, &episodes, 100_000).unwrap_err(),
        PairedBenchmarkError::WeightMassMismatch { actual_ppm: 800_000 }
    );

    episodes = panel().1;
    episodes[1].episode.realized_scenario.scenario_id =
        episodes[0].episode.realized_scenario.scenario_id.clone();
    seal_energy_hub_scenario(&mut episodes[1].episode.realized_scenario).unwrap();
    assert_eq!(
        evaluate_paired_rolling_benchmark(&baseline, &plan, &episodes, 100_000).unwrap_err(),
        PairedBenchmarkError::DuplicateScenarioId {
            scenario_id: "actual-cheap-future-v1".to_owned()
        }
    );
}

#[test]
fn exact_perfect_information_result_is_a_lower_cost_bound() {
    let (baseline, episodes) = panel();
    let plan = solve_exact_energy_hub_pareto(&baseline, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let report =
        evaluate_paired_rolling_benchmark(&baseline, &plan, &episodes, 100_000).unwrap();

    for outcome in &report.outcomes {
        if let (Some(rolling), Some(perfect)) = (
            outcome.rolling_run.evaluated_realized_plan.as_ref(),
            outcome.perfect_information_plan.as_ref(),
        ) {
            assert!(
                rolling.metrics.source_ledger.total_cost
                    + 1e-9
                    >= perfect.metrics.source_ledger.total_cost
            );
        }
    }
}

#[test]
fn exhausted_policy_search_remains_unresolved_not_infeasible() {
    let (baseline, episodes) = panel();
    let plan = solve_exact_energy_hub_pareto(&baseline, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let report = evaluate_paired_rolling_benchmark(&baseline, &plan, &episodes, 1).unwrap();

    assert_eq!(report.rolling_horizon_summary.feasible_weight_ppm, 0);
    assert_eq!(report.rolling_horizon_summary.infeasible_weight_ppm, 0);
    assert_eq!(report.rolling_horizon_summary.unresolved_weight_ppm, 1_000_000);
    assert_eq!(report.perfect_information_summary.feasible_weight_ppm, 0);
    assert_eq!(report.perfect_information_summary.unresolved_weight_ppm, 1_000_000);
}
