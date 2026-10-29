use symthaea_operations_research::aviation_energy_accounting::{EvidenceClass, TransportService};
use symthaea_operations_research::aviation_energy_hub::{
    canonical_energy_hub_digest, seal_energy_hub_scenario, solve_exact_energy_hub_pareto,
    solve_renewable_first_greedy, verify_energy_hub_plan, verify_energy_hub_digest,
    EnergyHubError, EnergyHubInterval, EnergyHubMetrics, EnergyStorageSpec,
    TimeIndexedEnergyHubScenario,
};

fn storage(capacity: u32, charge: u32, discharge: u32, terminal: u32) -> EnergyStorageSpec {
    EnergyStorageSpec {
        capacity_quanta: capacity,
        initial_soc_quanta: 0,
        max_charge_input_quanta_per_interval: charge,
        max_discharge_output_quanta_per_interval: discharge,
        charge_efficiency_ppm: 1_000_000,
        discharge_efficiency_ppm: 1_000_000,
        minimum_terminal_soc_quanta: terminal,
    }
}

fn interval(
    id: &str,
    demand: u32,
    charger_cap: u32,
    renewable: u32,
    grid_cap: u32,
    renewable_cost: f64,
    renewable_co2e: f64,
    grid_cost: f64,
    grid_co2e: f64,
) -> EnergyHubInterval {
    EnergyHubInterval {
        id: id.to_owned(),
        demand_quanta: demand,
        charger_capacity_quanta: charger_cap,
        renewable_generation_quanta: renewable,
        grid_import_capacity_quanta: grid_cap,
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

fn frontier_scenario() -> TimeIndexedEnergyHubScenario {
    seal(TimeIndexedEnergyHubScenario {
        scenario_id: "hub-frontier-synthetic-v1".to_owned(),
        input_digest: "pending".to_owned(),
        currency_code: "USD".to_owned(),
        quantum_kwh: 1.0,
        service: TransportService {
            period_id: "synthetic-period-01".to_owned(),
            lifecycle_boundary_id: "source-to-airport-bus-v1".to_owned(),
            completed_passenger_km: 1_000.0,
            available_seat_km: 1_200.0,
        },
        storage: storage(2, 2, 2, 0),
        intervals: vec![
            interval("t0", 2, 2, 2, 2, 0.5, 1.0, 1.0, 10.0),
            interval("t1", 2, 2, 0, 2, 0.0, 0.0, 5.0, 1.0),
        ],
    })
}

fn storage_bridge_scenario() -> TimeIndexedEnergyHubScenario {
    seal(TimeIndexedEnergyHubScenario {
        scenario_id: "hub-storage-bridge-synthetic-v1".to_owned(),
        input_digest: "pending".to_owned(),
        currency_code: "USD".to_owned(),
        quantum_kwh: 1.0,
        service: TransportService {
            period_id: "synthetic-period-02".to_owned(),
            lifecycle_boundary_id: "source-to-airport-bus-v1".to_owned(),
            completed_passenger_km: 100.0,
            available_seat_km: 120.0,
        },
        storage: storage(2, 2, 2, 1),
        intervals: vec![
            interval("t0", 2, 2, 2, 2, 0.5, 1.0, 1.0, 10.0),
            interval("t1", 3, 3, 0, 2, 0.0, 0.0, 5.0, 1.0),
        ],
    })
}

fn has_point(
    metrics: &EnergyHubMetrics,
    cost: f64,
    emissions: f64,
    final_soc: u32,
) -> bool {
    metrics.source_ledger.total_cost == cost
        && metrics.source_ledger.accounted_lifecycle_co2e_kg == emissions
        && metrics.final_soc_quanta == final_soc
}

#[test]
fn exact_frontier_contains_known_cost_emissions_tradeoffs() {
    let result = solve_exact_energy_hub_pareto(&frontier_scenario(), 100_000).unwrap();
    assert!(result.nodes_visited > 0);
    assert!(result.feasible_plans_seen >= 3);

    assert!(result.frontier.iter().any(|plan| has_point(&plan.metrics, 11.0, 4.0, 0)));
    assert!(result.frontier.iter().any(|plan| has_point(&plan.metrics, 7.0, 13.0, 0)));
    assert!(result.frontier.iter().any(|plan| has_point(&plan.metrics, 3.0, 22.0, 0)));

    for left in &result.frontier {
        for right in &result.frontier {
            if left.plan == right.plan {
                continue;
            }
            let dominates = left.metrics.source_ledger.total_cost
                <= right.metrics.source_ledger.total_cost
                && left.metrics.source_ledger.accounted_lifecycle_co2e_kg
                    <= right.metrics.source_ledger.accounted_lifecycle_co2e_kg
                && left.metrics.final_soc_quanta >= right.metrics.final_soc_quanta
                && (left.metrics.source_ledger.total_cost < right.metrics.source_ledger.total_cost
                    || left.metrics.source_ledger.accounted_lifecycle_co2e_kg
                        < right.metrics.source_ledger.accounted_lifecycle_co2e_kg
                    || left.metrics.final_soc_quanta > right.metrics.final_soc_quanta);
            assert!(!dominates, "frontier contains a dominated point");
        }
    }
}

#[test]
fn storage_aware_exact_plan_succeeds_when_myopic_renewable_first_fails() {
    let case = storage_bridge_scenario();
    let result = solve_exact_energy_hub_pareto(&case, 100_000).unwrap();
    assert_eq!(result.frontier.len(), 1);
    let best = &result.frontier[0];
    assert_eq!(best.metrics.final_soc_quanta, 1);
    assert_eq!(best.metrics.renewable_charge_input_quanta, 2);
    assert_eq!(best.metrics.grid_import_quanta, 4);
    assert_eq!(best.metrics.battery_discharge_output_quanta, 1);
    assert_eq!(best.metrics.renewable_generation_quanta, 2);
    assert_eq!(best.metrics.source_ledger.total_cost, 13.0);
    assert_eq!(best.metrics.source_ledger.accounted_lifecycle_co2e_kg, 24.0);
    assert_eq!(verify_energy_hub_plan(&case, &best.plan).unwrap(), best.clone());

    assert_eq!(
        solve_renewable_first_greedy(&case).unwrap_err(),
        EnergyHubError::NoFeasiblePlan
    );
}

#[test]
fn insufficient_grid_and_no_storage_is_infeasible() {
    let mut case = storage_bridge_scenario();
    case.storage = storage(0, 0, 0, 0);
    seal_energy_hub_scenario(&mut case).unwrap();
    assert_eq!(
        solve_exact_energy_hub_pareto(&case, 100_000).unwrap_err(),
        EnergyHubError::NoFeasiblePlan
    );
}

#[test]
fn canonical_digest_matches_independent_known_answer_vector() {
    assert_eq!(
        canonical_energy_hub_digest(&frontier_scenario()).unwrap(),
        "sha256:21048e86350e423468d5fcac5a137c6cfea74922df14894ad5331c05b70ac2e7"
    );
}

#[test]
fn renewable_curtailment_is_not_counted_as_delivered_energy() {
    let case = seal(TimeIndexedEnergyHubScenario {
        scenario_id: "hub-curtailment-synthetic-v1".to_owned(),
        input_digest: "pending".to_owned(),
        currency_code: "USD".to_owned(),
        quantum_kwh: 1.0,
        service: TransportService {
            period_id: "synthetic-period-curtailment".to_owned(),
            lifecycle_boundary_id: "source-to-airport-bus-v1".to_owned(),
            completed_passenger_km: 100.0,
            available_seat_km: 120.0,
        },
        storage: storage(1, 1, 1, 1),
        intervals: vec![interval("t0", 1, 1, 4, 0, 1.0, 2.0, 1.0, 10.0)],
    });
    let result = solve_exact_energy_hub_pareto(&case, 10_000).unwrap();
    let plan = result.frontier.iter().find(|candidate| {
        candidate.plan.actions[0].renewable_direct_quanta == 1
            && candidate.plan.actions[0].renewable_charge_input_quanta == 1
            && candidate.plan.actions[0].renewable_curtailed_quanta == 2
    }).expect("renewable surplus should serve demand, charge storage, and curtail remainder");

    assert_eq!(plan.metrics.source_ledger.total_source_energy_kwh, 4.0);
    assert_eq!(plan.metrics.source_ledger.total_delivered_energy_kwh, 2.0);
    assert_eq!(plan.metrics.renewable_generation_quanta, 4);
    assert_eq!(plan.metrics.renewable_curtailed_quanta, 2);
    assert_eq!(plan.metrics.final_soc_quanta, 1);
}

#[test]
fn canonical_digest_is_stable_and_binds_ordered_interval_content() {
    let original = frontier_scenario();
    assert_eq!(
        canonical_energy_hub_digest(&original).unwrap(),
        original.input_digest
    );
    verify_energy_hub_digest(&original).unwrap();

    let mut reversed = original.clone();
    reversed.intervals.reverse();
    assert_ne!(
        canonical_energy_hub_digest(&reversed).unwrap(),
        original.input_digest
    );
}

#[test]
fn stale_digest_after_price_change_is_rejected() {
    let mut case = frontier_scenario();
    case.intervals[0].grid_cost_per_quantum = 1.5;
    assert_eq!(
        solve_exact_energy_hub_pareto(&case, 100_000).unwrap_err(),
        EnergyHubError::ScenarioDigestMismatch
    );
}

#[test]
fn bad_plan_digest_is_rejected() {
    let case = frontier_scenario();
    let mut plan = solve_exact_energy_hub_pareto(&case, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    plan.input_digest = "sha256:wrong".to_owned();
    assert_eq!(
        verify_energy_hub_plan(&case, &plan).unwrap_err(),
        EnergyHubError::PlanIdentityMismatch
    );
}

#[test]
fn input_order_is_time_semantic_not_silently_normalized() {
    let mut case = frontier_scenario();
    case.intervals.swap(0, 1);
    assert_eq!(
        solve_exact_energy_hub_pareto(&case, 100_000).unwrap_err(),
        EnergyHubError::ScenarioDigestMismatch
    );
}

#[test]
fn demand_above_charger_capacity_is_infeasible() {
    let mut case = frontier_scenario();
    case.intervals[0].charger_capacity_quanta = 1;
    seal_energy_hub_scenario(&mut case).unwrap();
    assert_eq!(
        solve_exact_energy_hub_pareto(&case, 100_000).unwrap_err(),
        EnergyHubError::NoFeasiblePlan
    );
}

#[test]
fn exact_search_budget_exhaustion_is_not_reported_as_exact() {
    assert_eq!(
        solve_exact_energy_hub_pareto(&frontier_scenario(), 1).unwrap_err(),
        EnergyHubError::SearchBudgetExceeded { nodes_visited: 1 }
    );
}

#[test]
fn invalid_search_budget_is_rejected() {
    assert_eq!(
        solve_exact_energy_hub_pareto(&frontier_scenario(), 0).unwrap_err(),
        EnergyHubError::InvalidSearchBudget
    );
}

#[test]
fn duplicate_interval_identity_is_rejected() {
    let mut case = frontier_scenario();
    case.intervals[1].id = case.intervals[0].id.clone();
    assert_eq!(
        solve_exact_energy_hub_pareto(&case, 100_000).unwrap_err(),
        EnergyHubError::DuplicateIdentifier("interval ID")
    );
}

#[test]
fn nonfinite_factors_and_invalid_efficiency_fail_closed() {
    let mut case = frontier_scenario();
    case.intervals[0].grid_cost_per_quantum = f64::INFINITY;
    assert_eq!(
        solve_exact_energy_hub_pareto(&case, 100_000).unwrap_err(),
        EnergyHubError::InvalidNumericValue("grid cost factor")
    );

    let mut case = frontier_scenario();
    case.storage.charge_efficiency_ppm = 0;
    assert_eq!(
        solve_exact_energy_hub_pareto(&case, 100_000).unwrap_err(),
        EnergyHubError::InvalidEfficiency
    );
}

#[test]
fn storage_efficiency_uses_conservative_integer_rounding() {
    let mut case = frontier_scenario();
    case.storage = EnergyStorageSpec {
        capacity_quanta: 2,
        initial_soc_quanta: 0,
        max_charge_input_quanta_per_interval: 4,
        max_discharge_output_quanta_per_interval: 1,
        charge_efficiency_ppm: 500_000,
        discharge_efficiency_ppm: 500_000,
        minimum_terminal_soc_quanta: 0,
    };
    case.intervals[0] = interval("t0", 1, 1, 5, 1, 0.5, 1.0, 1.0, 10.0);
    case.intervals[1] = interval("t1", 2, 2, 0, 1, 0.0, 0.0, 5.0, 1.0);
    seal_energy_hub_scenario(&mut case).unwrap();

    let result = solve_exact_energy_hub_pareto(&case, 100_000).unwrap();
    let plan = result.frontier.iter().find(|plan| {
        plan.plan.actions[0].renewable_charge_input_quanta == 4
            && plan.plan.actions[0].soc_after_quanta == 2
            && plan.plan.actions[1].battery_discharge_output_quanta == 1
    }).expect("the known-answer feasible charging/discharging path exists");
    assert_eq!(plan.metrics.battery_charge_loss_quanta, 2);
    assert_eq!(plan.metrics.battery_discharge_loss_quanta, 1);
    assert_eq!(plan.metrics.final_soc_quanta, 0);
}

#[test]
fn action_count_mismatch_is_rejected() {
    let case = frontier_scenario();
    let mut plan = solve_exact_energy_hub_pareto(&case, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    plan.actions.pop();
    assert_eq!(
        verify_energy_hub_plan(&case, &plan).unwrap_err(),
        EnergyHubError::ActionCountMismatch
    );
}

#[test]
fn tampered_storage_state_is_rejected() {
    let case = storage_bridge_scenario();
    let mut plan = solve_exact_energy_hub_pareto(&case, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    plan.actions[0].soc_after_quanta += 1;
    assert_eq!(
        verify_energy_hub_plan(&case, &plan).unwrap_err(),
        EnergyHubError::StorageStateMismatch
    );
}

#[test]
fn tampered_renewable_curtailment_is_rejected() {
    let case = frontier_scenario();
    let mut plan = solve_exact_energy_hub_pareto(&case, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    plan.actions[0].renewable_curtailed_quanta = 99;
    assert_eq!(
        verify_energy_hub_plan(&case, &plan).unwrap_err(),
        EnergyHubError::ResourceBalanceViolation
    );
}

#[test]
fn terminal_reserve_is_a_hard_constraint() {
    let mut case = frontier_scenario();
    case.storage.minimum_terminal_soc_quanta = 1;
    seal_energy_hub_scenario(&mut case).unwrap();

    let no_reserve_case = frontier_scenario();
    let mut plan = solve_exact_energy_hub_pareto(&no_reserve_case, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    plan.input_digest = case.input_digest.clone();
    assert_eq!(
        verify_energy_hub_plan(&case, &plan).unwrap_err(),
        EnergyHubError::TerminalReserveNotMet
    );
}

#[test]
fn interval_sequence_is_recomputed_and_not_trusted_from_plan() {
    let case = frontier_scenario();
    let mut plan = solve_exact_energy_hub_pareto(&case, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    plan.actions[0].interval_id = "wrong-interval".to_owned();
    assert_eq!(
        verify_energy_hub_plan(&case, &plan).unwrap_err(),
        EnergyHubError::IntervalOrderMismatch
    );
}
