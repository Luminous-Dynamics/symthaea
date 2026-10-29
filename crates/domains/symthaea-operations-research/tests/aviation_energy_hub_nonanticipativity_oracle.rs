const HASH_A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const HASH_B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const HASH_C: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const HASH_D: &str = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

use symthaea_operations_research::aviation_energy_accounting::{EvidenceClass, TransportService};
use symthaea_operations_research::aviation_energy_hub::{
    seal_energy_hub_scenario, solve_exact_energy_hub_pareto, EnergyHubInterval, EnergyHubPlan,
    EnergyStorageSpec, TimeIndexedEnergyHubScenario,
};
use symthaea_operations_research::aviation_energy_hub_nonanticipativity::{
    verify_nonanticipative_policy, NonanticipativityError, ScenarioPolicyPlan,
};

fn storage() -> EnergyStorageSpec {
    EnergyStorageSpec {
        capacity_quanta: 2,
        initial_soc_quanta: 0,
        initial_soc_ledger_ref: "none:opening-soc".to_owned(),
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

fn nominal() -> TimeIndexedEnergyHubScenario {
    let mut scenario = TimeIndexedEnergyHubScenario {
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
    };
    seal_energy_hub_scenario(&mut scenario).unwrap();
    scenario
}

fn price_reversal(base: &TimeIndexedEnergyHubScenario) -> TimeIndexedEnergyHubScenario {
    let mut scenario = base.clone();
    scenario.scenario_id = "stress-price-reversal-v1".to_owned();
    scenario.intervals[1].grid_cost_per_quantum = 0.5;
    seal_energy_hub_scenario(&mut scenario).unwrap();
    scenario
}

fn bind_plan(source: &EnergyHubPlan, scenario: &TimeIndexedEnergyHubScenario) -> EnergyHubPlan {
    EnergyHubPlan {
        scenario_id: scenario.scenario_id.clone(),
        input_digest: scenario.input_digest.clone(),
        actions: source.actions.clone(),
    }
}

fn policy(
    scenario: &TimeIndexedEnergyHubScenario,
    plan: EnergyHubPlan,
    information_sets: &[&str],
    observed_history_digests: &[&str],
) -> ScenarioPolicyPlan {
    ScenarioPolicyPlan {
        scenario_id: scenario.scenario_id.clone(),
        plan,
        information_contract_ref: "synthetic-observation-contract-v1".to_owned(),
        information_set_ids_by_stage: information_sets.iter().map(|id| (*id).to_owned()).collect(),
        observed_history_digest_by_stage: observed_history_digests.iter().map(|digest| (*digest).to_owned()).collect(),
    }
}

#[test]
fn perfect_information_optima_can_violate_nonanticipativity() {
    let base = nominal();
    let price = price_reversal(&base);
    let nominal_plan = solve_exact_energy_hub_pareto(&base, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let price_plan = solve_exact_energy_hub_pareto(&price, 100_000).unwrap().frontier[0]
        .plan
        .clone();

    assert_ne!(nominal_plan.actions[0], price_plan.actions[0]);

    let report = verify_nonanticipative_policy(
        &[base.clone(), price.clone()],
        &[
            policy(&base, nominal_plan, &["root", "normal-future"], &[HASH_A, HASH_B]),
            policy(&price, price_plan, &["root", "cheap-future"], &[HASH_A, HASH_C]),
        ],
    )
    .unwrap();

    assert!(!report.is_nonanticipative());
    assert_eq!(report.scenarios_verified, 2);
    assert_eq!(report.pairwise_action_constraints_checked, 1);
    assert_eq!(report.violations.len(), 1);
    assert_eq!(report.violations[0].stage_index, 0);
    assert_eq!(report.violations[0].information_set_id, "root");
    assert_eq!(
        report.violations[0].conflicting_scenario_ids,
        vec![
            "nominal-energy-recourse-v1".to_owned(),
            "stress-price-reversal-v1".to_owned()
        ]
    );
    assert!(report.policy_digest.starts_with("sha256:"));
}

#[test]
fn the_same_verified_action_sequence_passes_shared_information_constraints() {
    let base = nominal();
    let price = price_reversal(&base);
    let nominal_plan = solve_exact_energy_hub_pareto(&base, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let price_bound_plan = bind_plan(&nominal_plan, &price);

    let report = verify_nonanticipative_policy(
        &[base.clone(), price.clone()],
        &[
            policy(&base, nominal_plan.clone(), &["root", "normal-future"], &[HASH_A, HASH_B]),
            policy(&price, price_bound_plan.clone(), &["root", "cheap-future"], &[HASH_A, HASH_C]),
        ],
    )
    .unwrap();
    assert!(report.is_nonanticipative());
    assert!(report.violations.is_empty());
    assert_eq!(report.pairwise_action_constraints_checked, 1);

    let reversed = verify_nonanticipative_policy(
        &[price.clone(), base.clone()],
        &[
            policy(&price, price_bound_plan, &["root", "cheap-future"], &[HASH_A, HASH_C]),
            policy(&base, nominal_plan, &["root", "normal-future"], &[HASH_A, HASH_B]),
        ],
    )
    .unwrap();
    assert_eq!(report, reversed);
}

#[test]
fn information_sets_cannot_merge_inconsistent_prior_histories() {
    let base = nominal();
    let price = price_reversal(&base);
    let base_plan = solve_exact_energy_hub_pareto(&base, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let price_plan = bind_plan(&base_plan, &price);

    assert_eq!(
        verify_nonanticipative_policy(
            &[base.clone(), price.clone()],
            &[
                policy(&base, base_plan, &["path-a", "merged-late"], &[HASH_A, HASH_B]),
                policy(&price, price_plan, &["path-b", "merged-late"], &[HASH_C, HASH_D]),
            ],
        )
        .unwrap_err(),
        NonanticipativityError::InformationSetHistoryMismatch {
            stage_index: 1,
            information_set_id: "merged-late".to_owned(),
        }
    );
}

#[test]
fn identical_observed_histories_cannot_be_split_across_information_sets() {
    let base = nominal();
    let price = price_reversal(&base);
    let base_plan = solve_exact_energy_hub_pareto(&base, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let price_plan = bind_plan(&base_plan, &price);

    assert_eq!(
        verify_nonanticipative_policy(
            &[base.clone(), price.clone()],
            &[
                policy(&base, base_plan, &["root-a", "normal-future"], &[HASH_A, HASH_B]),
                policy(&price, price_plan, &["root-b", "cheap-future"], &[HASH_A, HASH_C]),
            ],
        ).unwrap_err(),
        NonanticipativityError::IdenticalHistorySplitAcrossInformationSets { stage_index: 0 }
    );
}

#[test]
fn incompatible_time_grids_and_missing_policy_plans_are_rejected() {
    let base = nominal();
    let mut incompatible = price_reversal(&base);
    incompatible.intervals.swap(0, 1);
    seal_energy_hub_scenario(&mut incompatible).unwrap();
    let base_plan = solve_exact_energy_hub_pareto(&base, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    let incompatible_plan = bind_plan(&base_plan, &incompatible);

    assert_eq!(
        verify_nonanticipative_policy(
            &[base.clone(), incompatible.clone()],
            &[
                policy(&base, base_plan.clone(), &["root", "future-a"], &[HASH_A, HASH_B]),
                policy(&incompatible, incompatible_plan, &["root", "future-b"], &[HASH_A, HASH_C]),
            ],
        )
        .unwrap_err(),
        NonanticipativityError::IncompatibleScenario("ordered interval grid")
    );

    assert_eq!(
        verify_nonanticipative_policy(
            &[base.clone(), price_reversal(&base)],
            &[policy(&base, base_plan, &["root", "future-a"], &[HASH_A, HASH_B])],
        )
        .unwrap_err(),
        NonanticipativityError::MissingIdentifier("one plan per scenario")
    );
}

#[test]
fn plan_verification_is_required_before_information_consistency() {
    let base = nominal();
    let mut invalid_plan = solve_exact_energy_hub_pareto(&base, 100_000).unwrap().frontier[0]
        .plan
        .clone();
    invalid_plan.actions[0].renewable_curtailed_quanta += 1;

    assert_eq!(
        verify_nonanticipative_policy(
            &[base.clone()],
            &[policy(&base, invalid_plan, &["root", "future"], &[HASH_A, HASH_B])],
        )
        .unwrap_err(),
        NonanticipativityError::PlanVerification {
            scenario_id: "nominal-energy-recourse-v1".to_owned(),
            source: symthaea_operations_research::aviation_energy_hub::EnergyHubError::ResourceBalanceViolation,
        }
    );
}
