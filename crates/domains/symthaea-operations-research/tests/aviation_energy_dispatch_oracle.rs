use symthaea_operations_research::aviation_energy_accounting::{EnergyCarrier, EvidenceClass, TransportService};
use symthaea_operations_research::aviation_energy_dispatch::{
    canonical_scenario_digest, seal_scenario, solve_cost_greedy, solve_exact_pareto,
    verify_energy_plan, verify_scenario_digest, CleanAirlineEnergyScenario, EnergyAllocation,
    EnergyDemand, EnergyDispatchError, EnergyPlan, EnergySourceOffer,
};

fn scenario() -> CleanAirlineEnergyScenario {
    let mut scenario = CleanAirlineEnergyScenario {
        scenario_id: "airport-synthetic-v1".to_owned(),
        input_digest: "external-manifest-digest-v1".to_owned(),
        currency_code: "USD".to_owned(),
        quantum_kwh: 1.0,
        service: TransportService {
            period_id: "synthetic-day-01".to_owned(),
            lifecycle_boundary_id: "well-to-service-v1".to_owned(),
            completed_passenger_km: 100.0,
            available_seat_km: 120.0,
        },
        demands: vec![EnergyDemand {
            id: "regional-charging-demand".to_owned(),
            carrier: EnergyCarrier::Electricity,
            required_quanta: 5,
        }],
        offers: vec![
            EnergySourceOffer {
                id: "grid-offer".to_owned(),
                source_id: "grid-electricity".to_owned(),
                carrier: EnergyCarrier::Electricity,
                capacity_quanta: 5,
                cost_per_quantum: 1.0,
                upstream_co2e_kg_per_quantum: 5.0,
                direct_operational_co2e_kg_per_quantum: 0.0,
                evidence_ref: "synthetic:grid-factor-v1".to_owned(),
                evidence_class: EvidenceClass::Synthetic,
            },
            EnergySourceOffer {
                id: "low-carbon-offer".to_owned(),
                source_id: "low-carbon-electricity".to_owned(),
                carrier: EnergyCarrier::Electricity,
                capacity_quanta: 5,
                cost_per_quantum: 3.0,
                upstream_co2e_kg_per_quantum: 0.1,
                direct_operational_co2e_kg_per_quantum: 0.0,
                evidence_ref: "synthetic:low-carbon-factor-v1".to_owned(),
                evidence_class: EvidenceClass::Synthetic,
            },
        ],
    };
    seal_scenario(&mut scenario).unwrap();
    scenario
}

fn plan(allocations: Vec<EnergyAllocation>) -> EnergyPlan {
    let scenario = scenario();
    EnergyPlan {
        scenario_id: scenario.scenario_id,
        input_digest: scenario.input_digest,
        allocations,
    }
}

#[test]
fn exact_oracle_returns_the_six_point_cost_emissions_pareto_frontier() {
    let result = solve_exact_pareto(&scenario(), 100_000).unwrap();
    assert_eq!(result.frontier.len(), 6);
    assert_eq!(result.feasible_plans_seen, 6);
    assert!(result.nodes_visited > 0);

    let costs: Vec<f64> = result.frontier.iter().map(|item| item.ledger.total_cost).collect();
    let emissions: Vec<f64> = result
        .frontier
        .iter()
        .map(|item| item.ledger.accounted_lifecycle_co2e_kg)
        .collect();
    assert_eq!(costs, vec![5.0, 7.0, 9.0, 11.0, 13.0, 15.0]);
    assert_eq!(emissions, vec![25.0, 20.1, 15.2, 10.3, 5.4, 0.5]);

    // The oracle returns a vector of trade-offs, not an arbitrary weighted winner.
    for pair in result.frontier.windows(2) {
        assert!(pair[0].ledger.total_cost < pair[1].ledger.total_cost);
        assert!(pair[0].ledger.accounted_lifecycle_co2e_kg
            > pair[1].ledger.accounted_lifecycle_co2e_kg);
    }
}

#[test]
fn cost_greedy_baseline_selects_the_cheapest_source_and_is_verified() {
    let scenario = scenario();
    let greedy = solve_cost_greedy(&scenario).unwrap();
    assert_eq!(
        greedy.plan.allocations,
        vec![EnergyAllocation {
            offer_id: "grid-offer".to_owned(),
            quanta: 5,
        }]
    );
    assert_eq!(greedy.ledger.total_cost, 5.0);
    assert_eq!(greedy.ledger.accounted_lifecycle_co2e_kg, 25.0);
    assert_eq!(verify_energy_plan(&scenario, &greedy.plan).unwrap(), greedy);
}

#[test]
fn exact_frontier_is_invariant_to_offer_input_order() {
    let mut reversed = scenario();
    reversed.offers.reverse();
    let forward_result = solve_exact_pareto(&scenario(), 100_000).unwrap();
    let reverse_result = solve_exact_pareto(&reversed, 100_000).unwrap();
    assert_eq!(forward_result.frontier, reverse_result.frontier);
}

#[test]
fn exact_search_budget_exhaustion_is_not_reported_as_exact() {
    assert_eq!(
        solve_exact_pareto(&scenario(), 1).unwrap_err(),
        EnergyDispatchError::SearchBudgetExceeded { nodes_visited: 1 }
    );
}

#[test]
fn invalid_search_budgets_are_rejected() {
    assert_eq!(
        solve_exact_pareto(&scenario(), 0).unwrap_err(),
        EnergyDispatchError::InvalidSearchBudget
    );
    assert_eq!(
        solve_exact_pareto(&scenario(), 1_000_001).unwrap_err(),
        EnergyDispatchError::InvalidSearchBudget
    );
}

#[test]
fn underprovisioned_supply_has_no_feasible_plan() {
    let mut case = scenario();
    case.offers[0].capacity_quanta = 2;
    case.offers[1].capacity_quanta = 2;
    seal_scenario(&mut case).unwrap();
    assert_eq!(
        solve_exact_pareto(&case, 100_000).unwrap_err(),
        EnergyDispatchError::NoFeasiblePlan
    );
    assert_eq!(
        solve_cost_greedy(&case).unwrap_err(),
        EnergyDispatchError::NoFeasiblePlan
    );
}

#[test]
fn plan_with_wrong_manifest_identity_is_rejected() {
    let case = scenario();
    let mut candidate = plan(vec![EnergyAllocation {
        offer_id: "grid-offer".to_owned(),
        quanta: 5,
    }]);
    candidate.input_digest = "different-digest".to_owned();
    assert_eq!(
        verify_energy_plan(&case, &candidate).unwrap_err(),
        EnergyDispatchError::PlanIdentityMismatch
    );
}

#[test]
fn plan_unknown_offer_is_rejected() {
    let candidate = plan(vec![EnergyAllocation {
        offer_id: "unknown".to_owned(),
        quanta: 5,
    }]);
    assert_eq!(
        verify_energy_plan(&scenario(), &candidate).unwrap_err(),
        EnergyDispatchError::UnknownOffer
    );
}

#[test]
fn duplicate_allocation_for_one_offer_is_rejected() {
    let candidate = plan(vec![
        EnergyAllocation {
            offer_id: "grid-offer".to_owned(),
            quanta: 2,
        },
        EnergyAllocation {
            offer_id: "grid-offer".to_owned(),
            quanta: 3,
        },
    ]);
    assert_eq!(
        verify_energy_plan(&scenario(), &candidate).unwrap_err(),
        EnergyDispatchError::DuplicateAllocation
    );
}

#[test]
fn source_capacity_limit_is_independently_enforced() {
    let candidate = plan(vec![EnergyAllocation {
        offer_id: "grid-offer".to_owned(),
        quanta: 6,
    }]);
    assert_eq!(
        verify_energy_plan(&scenario(), &candidate).unwrap_err(),
        EnergyDispatchError::AllocationExceedsCapacity
    );
}

#[test]
fn verifier_rejects_excess_and_missing_carrier_demand() {
    let excess = plan(vec![
        EnergyAllocation {
            offer_id: "grid-offer".to_owned(),
            quanta: 5,
        },
        EnergyAllocation {
            offer_id: "low-carbon-offer".to_owned(),
            quanta: 1,
        },
    ]);
    assert_eq!(
        verify_energy_plan(&scenario(), &excess).unwrap_err(),
        EnergyDispatchError::DemandExceeded
    );

    let missing = plan(vec![EnergyAllocation {
        offer_id: "grid-offer".to_owned(),
        quanta: 4,
    }]);
    assert_eq!(
        verify_energy_plan(&scenario(), &missing).unwrap_err(),
        EnergyDispatchError::DemandNotMet
    );
}

#[test]
fn zero_quanta_in_plan_is_rejected() {
    let candidate = plan(vec![EnergyAllocation {
        offer_id: "grid-offer".to_owned(),
        quanta: 0,
    }]);
    assert_eq!(
        verify_energy_plan(&scenario(), &candidate).unwrap_err(),
        EnergyDispatchError::ZeroAllocation
    );
}

#[test]
fn scenario_rejects_duplicate_source_identity() {
    let mut case = scenario();
    case.offers[1].source_id = case.offers[0].source_id.clone();
    assert_eq!(
        solve_exact_pareto(&case, 100_000).unwrap_err(),
        EnergyDispatchError::DuplicateIdentifier("source ID")
    );
}

#[test]
fn scenario_rejects_nonfinite_energy_or_emission_inputs() {
    let mut case = scenario();
    case.quantum_kwh = f64::NAN;
    assert_eq!(
        solve_exact_pareto(&case, 100_000).unwrap_err(),
        EnergyDispatchError::InvalidQuantum
    );

    let mut case = scenario();
    case.offers[0].upstream_co2e_kg_per_quantum = f64::INFINITY;
    assert_eq!(
        solve_exact_pareto(&case, 100_000).unwrap_err(),
        EnergyDispatchError::InvalidNumericValue("upstream CO2e factor")
    );

    let mut case = scenario();
    case.offers[0].upstream_co2e_kg_per_quantum = -1.0;
    assert_eq!(
        solve_exact_pareto(&case, 100_000).unwrap_err(),
        EnergyDispatchError::NegativeValue("upstream CO2e factor")
    );
}

#[test]
fn scenario_rejects_invalid_currency_and_negative_cost() {
    let mut case = scenario();
    case.currency_code = "usd".to_owned();
    assert_eq!(
        solve_exact_pareto(&case, 100_000).unwrap_err(),
        EnergyDispatchError::InvalidCurrencyCode
    );

    let mut case = scenario();
    case.offers[0].cost_per_quantum = -1.0;
    assert_eq!(
        solve_cost_greedy(&case).unwrap_err(),
        EnergyDispatchError::NegativeValue("unit cost")
    );
}

#[test]
fn empty_manifest_digest_is_rejected() {
    let mut case = scenario();
    case.input_digest.clear();
    assert_eq!(
        solve_exact_pareto(&case, 100_000).unwrap_err(),
        EnergyDispatchError::MissingIdentifier("input digest")
    );
}

#[test]
fn canonical_digest_is_stable_and_normalizes_input_vector_order() {
    let original = scenario();
    let first = canonical_scenario_digest(&original).unwrap();
    assert_eq!(original.input_digest, first);
    verify_scenario_digest(&original).unwrap();

    let mut reordered = original.clone();
    reordered.offers.reverse();
    reordered.demands.reverse();
    assert_eq!(canonical_scenario_digest(&reordered).unwrap(), first);
}

#[test]
fn canonical_digest_changes_when_a_material_field_changes() {
    let original = scenario();
    let original_digest = original.input_digest.clone();

    let mut changed = original.clone();
    changed.offers[0].cost_per_quantum = 1.25;
    assert_ne!(canonical_scenario_digest(&changed).unwrap(), original_digest);
    assert_eq!(
        solve_exact_pareto(&changed, 100_000).unwrap_err(),
        EnergyDispatchError::ScenarioDigestMismatch
    );
}

#[test]
fn plan_is_bound_to_verified_canonical_scenario_digest() {
    let mut case = scenario();
    case.offers[0].cost_per_quantum = 1.25;
    let candidate = EnergyPlan {
        scenario_id: case.scenario_id.clone(),
        input_digest: case.input_digest.clone(),
        allocations: vec![EnergyAllocation {
            offer_id: "grid-offer".to_owned(),
            quanta: 5,
        }],
    };
    assert_eq!(
        verify_energy_plan(&case, &candidate).unwrap_err(),
        EnergyDispatchError::ScenarioDigestMismatch
    );
}
