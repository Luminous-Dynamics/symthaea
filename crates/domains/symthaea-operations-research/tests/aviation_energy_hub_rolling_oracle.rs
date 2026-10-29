use symthaea_operations_research::aviation_energy_accounting::{EvidenceClass, TransportService};
use symthaea_operations_research::aviation_energy_hub::{
    seal_energy_hub_scenario, solve_exact_energy_hub_pareto, EnergyHubInterval,
    EnergyStorageSpec, TimeIndexedEnergyHubScenario,
};
use symthaea_operations_research::aviation_energy_hub_rolling::{
    canonical_forecast_digest, run_rolling_horizon, ForecastWindow, RollingHorizonEpisode,
    RollingHorizonError, RollingHorizonStatus,
};

fn interval(
    id: &str,
    demand: u32,
    renewable: u32,
    grid_capacity: u32,
    renewable_cost: f64,
    renewable_co2e: f64,
    grid_cost: f64,
    grid_co2e: f64,
) -> EnergyHubInterval {
    EnergyHubInterval {
        id: id.to_owned(),
        demand_quanta: demand,
        charger_capacity_quanta: demand,
        renewable_generation_quanta: renewable,
        grid_import_capacity_quanta: grid_capacity,
        renewable_cost_per_quantum: renewable_cost,
        renewable_lifecycle_co2e_kg_per_quantum: renewable_co2e,
        renewable_evidence_ref: format!("synthetic:renewable:{id}"),
        renewable_evidence_class: EvidenceClass::Synthetic,
        grid_cost_per_quantum: grid_cost,
        grid_lifecycle_co2e_kg_per_quantum: grid_co2e,
        grid_evidence_ref: format!("synthetic:grid:{id}"),
        grid_evidence_class: EvidenceClass::Synthetic,
    }
}

fn storage(initial_soc: u32) -> EnergyStorageSpec {
    EnergyStorageSpec {
        capacity_quanta: 1,
        initial_soc_quanta: initial_soc,
        max_charge_input_quanta_per_interval: 1,
        max_discharge_output_quanta_per_interval: 1,
        charge_efficiency_ppm: 1_000_000,
        discharge_efficiency_ppm: 1_000_000,
        minimum_terminal_soc_quanta: 0,
    }
}

fn realized_scenario() -> TimeIndexedEnergyHubScenario {
    let mut scenario = TimeIndexedEnergyHubScenario {
        scenario_id: "rolling-horizon-future-price-trap-v1".to_owned(),
        input_digest: "pending".to_owned(),
        currency_code: "USD".to_owned(),
        quantum_kwh: 1.0,
        service: TransportService {
            period_id: "rolling-test-period".to_owned(),
            lifecycle_boundary_id: "source-to-load-v1".to_owned(),
            completed_passenger_km: 100.0,
            available_seat_km: 120.0,
        },
        storage: storage(1),
        intervals: vec![
            interval("t0", 1, 0, 1, 0.0, 0.0, 1.0, 10.0),
            interval("t1", 1, 0, 1, 0.0, 0.0, 0.1, 1.0),
        ],
    };
    seal_energy_hub_scenario(&mut scenario).unwrap();
    scenario
}

fn forecast_window(future_intervals: Vec<EnergyHubInterval>, issued: i64, available: i64) -> ForecastWindow {
    ForecastWindow {
        issued_at_epoch: issued,
        available_at_epoch: available,
        future_intervals,
    }
}

fn episode(realized: TimeIndexedEnergyHubScenario) -> RollingHorizonEpisode {
    let forecast_future_t1 = interval("t1", 1, 0, 1, 0.0, 0.0, 10.0, 50.0);
    RollingHorizonEpisode {
        realized_scenario: realized,
        decision_times_epoch: vec![100, 200],
        observation_available_at_epoch: vec![90, 190],
        forecasts_by_stage: vec![
            forecast_window(vec![forecast_future_t1], 70, 80),
            forecast_window(vec![], 150, 160),
        ],
    }
}

#[test]
fn rolling_horizon_does_not_use_unobserved_future_realization() {
    let realized = realized_scenario();
    let clairvoyant = solve_exact_energy_hub_pareto(&realized, 100_000).unwrap().frontier[0]
        .plan;
    assert_eq!(clairvoyant.actions[0].battery_discharge_output_quanta, 1);
    assert_eq!(clairvoyant.actions[0].grid_import_quanta, 0);

    let run = run_rolling_horizon(&episode(realized), 100_000).unwrap();
    assert_eq!(run.status, RollingHorizonStatus::Completed);
    assert_eq!(run.steps.len(), 2);
    assert_eq!(run.steps[0].action.grid_import_quanta, 1);
    assert_eq!(run.steps[0].action.battery_discharge_output_quanta, 0);
    assert_eq!(run.steps[1].action.battery_discharge_output_quanta, 1);
    assert_eq!(run.final_soc_quanta, 0);

    let evaluated = run.evaluated_realized_plan.unwrap();
    assert_eq!(evaluated.metrics.source_ledger.total_cost, 1.0);
    assert_eq!(
        evaluated.metrics.source_ledger.accounted_lifecycle_co2e_kg,
        10.0
    );
}

#[test]
fn rolling_horizon_matches_full_information_when_forecasts_are_correct() {
    let realized = realized_scenario();
    let mut case = episode(realized.clone());
    case.forecasts_by_stage[0].future_intervals[0] = realized.intervals[1].clone();

    let rolling = run_rolling_horizon(&case, 100_000).unwrap();
    let exact = solve_exact_energy_hub_pareto(&realized, 100_000).unwrap().frontier[0].clone();

    assert_eq!(rolling.status, RollingHorizonStatus::Completed);
    assert_eq!(
        rolling.evaluated_realized_plan.as_ref().unwrap().metrics.source_ledger.total_cost,
        exact.metrics.source_ledger.total_cost
    );
    assert_eq!(
        rolling.evaluated_realized_plan.as_ref().unwrap().metrics.source_ledger.accounted_lifecycle_co2e_kg,
        exact.metrics.source_ledger.accounted_lifecycle_co2e_kg
    );
}

#[test]
fn forecast_not_available_by_decision_is_rejected() {
    let realized = realized_scenario();
    let mut case = episode(realized);
    case.forecasts_by_stage[0].available_at_epoch = 101;
    assert_eq!(
        run_rolling_horizon(&case, 100_000).unwrap_err(),
        RollingHorizonError::ForecastUnavailable { stage_index: 0 }
    );
}

#[test]
fn observation_not_available_by_decision_is_rejected() {
    let realized = realized_scenario();
    let mut case = episode(realized);
    case.observation_available_at_epoch[0] = 101;
    assert_eq!(
        run_rolling_horizon(&case, 100_000).unwrap_err(),
        RollingHorizonError::ObservationUnavailable { stage_index: 0 }
    );
}

#[test]
fn mismatched_future_interval_id_is_rejected() {
    let realized = realized_scenario();
    let mut case = episode(realized);
    case.forecasts_by_stage[0].future_intervals[0].id = "different-interval".to_owned();
    assert_eq!(
        run_rolling_horizon(&case, 100_000).unwrap_err(),
        RollingHorizonError::ForecastIntervalIdentityMismatch {
            stage_index: 0,
            interval_index: 1,
        }
    );
}

#[test]
fn search_timeout_is_not_reported_as_infeasibility_or_success() {
    let realized = realized_scenario();
    let run = run_rolling_horizon(&episode(realized), 1).unwrap();
    assert_eq!(
        run.status,
        RollingHorizonStatus::SearchBudgetExceeded {
            stage_index: 0,
            nodes_visited: 1,
        }
    );
    assert!(run.evaluated_realized_plan.is_none());
    assert!(run.steps.is_empty());
}

#[test]
fn forecast_digest_binds_time_metadata_and_future_values() {
    let forecast = episode(realized_scenario()).forecasts_by_stage[0].clone();
    let digest = canonical_forecast_digest(&forecast).unwrap();
    assert!(digest.starts_with("sha256:"));

    let mut changed_time = forecast.clone();
    changed_time.issued_at_epoch += 1;
    assert_ne!(canonical_forecast_digest(&changed_time).unwrap(), digest);

    let mut changed_value = forecast;
    changed_value.future_intervals[0].grid_cost_per_quantum = 11.0;
    assert_ne!(canonical_forecast_digest(&changed_value).unwrap(), digest);
}

#[test]
fn initial_soc_above_storage_capacity_is_rejected_by_energy_hub() {
    let mut scenario = realized_scenario();
    scenario.storage.initial_soc_quanta = 2;
    assert_eq!(
        solve_exact_energy_hub_pareto(&scenario, 100_000).unwrap_err(),
        symthaea_operations_research::aviation_energy_hub::EnergyHubError::InvalidStorage
    );
}
