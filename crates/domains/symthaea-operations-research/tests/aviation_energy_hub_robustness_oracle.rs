use symthaea_operations_research::aviation_energy_accounting::{EvidenceClass, TransportService};
use symthaea_operations_research::aviation_energy_hub::{
    seal_energy_hub_scenario, solve_exact_energy_hub_pareto, EnergyHubError, EnergyHubInterval,
    EnergyStorageSpec, TimeIndexedEnergyHubScenario,
};
use symthaea_operations_research::aviation_energy_hub_robustness::{
    canonical_stress_set_digest, evaluate_energy_hub_stress_set, EnergyHubRobustnessError,
    WeightedEnergyHubScenario,
};

fn storage() -> EnergyStorageSpec {
    EnergyStorageSpec {
        capacity_quanta: 2,
        max_charge_input_quanta_per_interval: 2,
        max_discharge_output_quanta_per_interval: 2,
        charge_efficiency_ppm: 1_000_000,
        discharge_efficiency_ppm: 1_000_000,
        minimum_terminal_soc_quanta: 0,
    }
}

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

fn seal(mut scenario: TimeIndexedEnergyHubScenario) -> TimeIndexedEnergyHubScenario {
    seal_energy_hub_scenario(&mut scenario).unwrap();
    scenario
}

fn nominal() -> TimeIndexedEnergyHubScenario {
    seal(TimeIndexedEnergyHubScenario {
        scenario_id: "nominal-energy-recourse-v1".to_owned(),
        input_digest: "pending".to_owned(),
        currency_code: "USD".to_owned(),
        quantum_kwh: 1.0,
        service: TransportService {
            period_id: "stress-period-v1".to_owned(),
            lifecycle_boundary_id: "source-to-airport-bus-v1".to_owned(),
            completed_passenger_km: 1_000.0,
            available_seat_km: 1_200.0,
        },
        storage: storage(),
        intervals: vec![
            interval("t0", 2, 2, 2, 0.5, 1.0, 1.0, 10.0),
            interval("t1", 2, 0, 2, 0.0, 0.0, 5.0, 1.0),
        ],
    })
}

fn stress_price_reversal(base: &TimeIndexedEnergyHubScenario) -> TimeIndexedEnergyHubScenario {
    let mut case = base.clone();
    case.scenario_id = "stress-price-reversal-v1".to_owned();
    case.intervals[1].grid_cost_per_quantum = 0.5;
    seal_energy_hub_scenario(&mut case).unwrap();
    case
}

fn stress_renewable_shortfall(base: &TimeIndexedEnergyHubScenario) -> TimeIndexedEnergyHubScenario {
    let mut case = base.clone();
    case.scenario_id = "stress-renewable-shortfall-v1".to_owned();
    case.intervals[0].renewable_generation_quanta = 1;
    seal_energy_hub_scenario(&mut case).unwrap();
    case
}

#[test]
fn recourse_restores_feasibility_when_nominal_plan_fails_under_shortfall() {
    let base = nominal();
    let nominal_plan = solve_exact_energy_hub_pareto(&base, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let cases = vec![
        WeightedEnergyHubScenario {
            scenario: stress_price_reversal(&base),
            weight_ppm: 500_000,
        },
        WeightedEnergyHubScenario {
            scenario: stress_renewable_shortfall(&base),
            weight_ppm: 500_000,
        },
    ];

    let report = evaluate_energy_hub_stress_set(&base, &nominal_plan, &cases, 100_000).unwrap();
    assert_eq!(report.outcomes.len(), 2);
    assert_eq!(report.static_replay_summary.feasible_weight_ppm, 500_000);
    assert_eq!(report.static_replay_summary.infeasible_weight_ppm, 500_000);
    assert_eq!(report.perfect_information_recourse_summary.feasible_weight_ppm, 1_000_000);
    assert_eq!(report.perfect_information_recourse_summary.infeasible_weight_ppm, 0);
    assert!(report.static_replay_covers_all_cases == false);
    assert!(report.perfect_information_recourse_covers_all_cases);

    let shortage = report
        .outcomes
        .iter()
        .find(|outcome| outcome.scenario_id == "stress-renewable-shortfall-v1")
        .unwrap();
    assert!(shortage.static_replay.is_none());
    assert_eq!(
        shortage.static_replay_error,
        Some(EnergyHubError::ResourceBalanceViolation)
    );
    assert!(shortage.perfect_information_recourse.is_some());
    assert_eq!(shortage.recourse_error, None);

    let price_shift = report
        .outcomes
        .iter()
        .find(|outcome| outcome.scenario_id == "stress-price-reversal-v1")
        .unwrap();
    assert!(price_shift.static_replay.is_some());
    assert!(price_shift.perfect_information_recourse.is_some());
    assert!(
        price_shift.perfect_information_recourse.as_ref().unwrap().metrics.source_ledger.total_cost
            < price_shift.static_replay.as_ref().unwrap().metrics.source_ledger.total_cost
    );
}

#[test]
fn stress_set_digest_and_outcomes_ignore_case_input_order() {
    let base = nominal();
    let nominal_plan = solve_exact_energy_hub_pareto(&base, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let first = WeightedEnergyHubScenario {
        scenario: stress_price_reversal(&base),
        weight_ppm: 400_000,
    };
    let second = WeightedEnergyHubScenario {
        scenario: stress_renewable_shortfall(&base),
        weight_ppm: 600_000,
    };
    let forward = vec![first.clone(), second.clone()];
    let mut reverse = forward.clone();
    reverse.reverse();
    assert_eq!(
        canonical_stress_set_digest(&forward).unwrap(),
        canonical_stress_set_digest(&reverse).unwrap()
    );
    assert_eq!(
        evaluate_energy_hub_stress_set(&base, &nominal_plan, &forward, 100_000).unwrap(),
        evaluate_energy_hub_stress_set(&base, &nominal_plan, &reverse, 100_000).unwrap()
    );
}

#[test]
fn scenario_weights_must_sum_to_one_million_ppm() {
    let base = nominal();
    let nominal_plan = solve_exact_energy_hub_pareto(&base, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let cases = vec![
        WeightedEnergyHubScenario {
            scenario: stress_price_reversal(&base),
            weight_ppm: 400_000,
        },
        WeightedEnergyHubScenario {
            scenario: stress_renewable_shortfall(&base),
            weight_ppm: 400_000,
        },
    ];
    assert_eq!(
        evaluate_energy_hub_stress_set(&base, &nominal_plan, &cases, 100_000).unwrap_err(),
        EnergyHubRobustnessError::WeightMassMismatch { actual_ppm: 800_000 }
    );
}

#[test]
fn zero_weight_and_duplicate_scenario_ids_are_rejected() {
    let base = nominal();
    let nominal_plan = solve_exact_energy_hub_pareto(&base, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let one = stress_price_reversal(&base);
    let cases = vec![WeightedEnergyHubScenario {
        scenario: one.clone(),
        weight_ppm: 0,
    }];
    assert_eq!(
        evaluate_energy_hub_stress_set(&base, &nominal_plan, &cases, 100_000).unwrap_err(),
        EnergyHubRobustnessError::ZeroScenarioWeight
    );

    let cases = vec![
        WeightedEnergyHubScenario {
            scenario: one.clone(),
            weight_ppm: 500_000,
        },
        WeightedEnergyHubScenario {
            scenario: one,
            weight_ppm: 500_000,
        },
    ];
    assert_eq!(
        evaluate_energy_hub_stress_set(&base, &nominal_plan, &cases, 100_000).unwrap_err(),
        EnergyHubRobustnessError::DuplicateScenarioId
    );
}

#[test]
fn incompatible_time_grid_and_stale_scenario_identity_fail_closed() {
    let base = nominal();
    let nominal_plan = solve_exact_energy_hub_pareto(&base, 100_000).unwrap().frontier[0]
        .plan
        .clone();

    let mut incompatible = stress_price_reversal(&base);
    incompatible.intervals.swap(0, 1);
    seal_energy_hub_scenario(&mut incompatible).unwrap();
    let cases = vec![WeightedEnergyHubScenario {
        scenario: incompatible,
        weight_ppm: 1_000_000,
    }];
    assert_eq!(
        evaluate_energy_hub_stress_set(&base, &nominal_plan, &cases, 100_000).unwrap_err(),
        EnergyHubRobustnessError::IncompatibleScenario("ordered interval grid")
    );

    let mut stale = stress_price_reversal(&base);
    stale.intervals[0].grid_cost_per_quantum += 1.0;
    let cases = vec![WeightedEnergyHubScenario {
        scenario: stale,
        weight_ppm: 1_000_000,
    }];
    assert_eq!(
        evaluate_energy_hub_stress_set(&base, &nominal_plan, &cases, 100_000).unwrap_err(),
        EnergyHubRobustnessError::InvalidScenarioDigest
    );
}

#[test]
fn search_budget_failure_remains_unresolved_not_a_free_success() {
    let base = nominal();
    let nominal_plan = solve_exact_energy_hub_pareto(&base, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let cases = vec![WeightedEnergyHubScenario {
        scenario: base.clone(),
        weight_ppm: 1_000_000,
    }];
    let report = evaluate_energy_hub_stress_set(&base, &nominal_plan, &cases, 1).unwrap();
    assert_eq!(
        report.outcomes[0].recourse_error,
        Some(EnergyHubError::SearchBudgetExceeded { nodes_visited: 1 })
    );
    assert!(!report.perfect_information_recourse_covers_all_cases);
}
