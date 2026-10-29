// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Raw-input scenario-ensemble checks for paired greedy/search recovery.

use symthaea_operations_research::aviation_retiming_ensemble::{
    evaluate_retiming_ensemble, RetimingEnsembleCase, RetimingEnsembleError,
    RetimingEnsembleOptions,
};
use symthaea_operations_research::aviation_retiming_recovery::{
    RetimingAircraft, RetimingFlight, RetimingScenario,
};

fn greedy_trap() -> RetimingScenario {
    RetimingScenario {
        id: "high-penalty-protected".into(),
        configuration_revision: "r1".into(),
        input_digest: "synthetic-digest-high-penalty".into(),
        flights: vec![
            RetimingFlight {
                id: "low-penalty-long".into(),
                scheduled_start_minutes: 0,
                duration_minutes: 20,
                max_delay_minutes: 0,
                delay_cost_per_minute: 1,
                cancellation_cost: 1,
            },
            RetimingFlight {
                id: "high-penalty-short".into(),
                scheduled_start_minutes: 10,
                duration_minutes: 10,
                max_delay_minutes: 0,
                delay_cost_per_minute: 1,
                cancellation_cost: 100,
            },
        ],
        aircraft: vec![RetimingAircraft {
            id: "A1".into(),
            available_at_minutes: 0,
            turnaround_minutes: 0,
        }],
    }
}

fn unconstrained_scenario() -> RetimingScenario {
    let mut s = greedy_trap();
    s.id = "no-overlap-control".into();
    s.input_digest = "synthetic-digest-no-overlap".into();
    s.flights[1].scheduled_start_minutes = 30;
    s
}

fn options() -> RetimingEnsembleOptions {
    RetimingEnsembleOptions {
        max_expanded_nodes_per_scenario: 100_000,
        max_generated_nodes_per_scenario: 100_000,
        max_absolute_optimality_gap: 0,
        tv_radius_ppm: 200_000,
    }
}

fn cases() -> Vec<RetimingEnsembleCase> {
    vec![
        RetimingEnsembleCase {
            scenario: greedy_trap(),
            nominal_probability_ppm: 600_000,
        },
        RetimingEnsembleCase {
            scenario: unconstrained_scenario(),
            nominal_probability_ppm: 400_000,
        },
    ]
}

#[test]
fn ensemble_reveals_greedy_regret_and_tv_robust_cost_delta() {
    let report = evaluate_retiming_ensemble(&cases(), options()).unwrap();
    assert_eq!(report.scenario_count, 2);
    assert_eq!(report.scenarios[0].scenario_id, "high-penalty-protected");
    assert_eq!(report.scenarios[1].scenario_id, "no-overlap-control");

    let trap = &report.scenarios[0];
    assert_eq!(trap.greedy_cost_units, 100);
    assert_eq!(trap.search_cost_units, 1);
    assert_eq!(trap.candidate_minus_greedy_cost_units, -99);
    assert_eq!(trap.search_absolute_gap_units, Some(0));

    let control = &report.scenarios[1];
    assert_eq!(control.greedy_cost_units, 0);
    assert_eq!(control.search_cost_units, 0);
    assert_eq!(control.candidate_minus_greedy_cost_units, 0);

    let robust = &report.distributionally_robust_cost_delta;
    assert_eq!(robust.nominal_expected_delta_numerator_ppm(), -59_400_000);
    assert_eq!(robust.worst_case_expected_delta_numerator_ppm(), -39_600_000);
    assert_eq!(robust.worst_single_scenario_delta_units(), 0);
    assert_eq!(robust.scenarios_improved(), 1);
    assert_eq!(robust.scenarios_unchanged(), 1);
}

#[test]
fn ensemble_result_is_invariant_to_input_order() {
    let forward = cases();
    let reverse = vec![forward[1].clone(), forward[0].clone()];
    assert_eq!(
        evaluate_retiming_ensemble(&forward, options()).unwrap(),
        evaluate_retiming_ensemble(&reverse, options()).unwrap()
    );
}

#[test]
fn ensemble_fails_closed_on_duplicate_ids_or_bad_probability_mass() {
    let one = greedy_trap();
    let duplicate = vec![
        RetimingEnsembleCase {
            scenario: one.clone(),
            nominal_probability_ppm: 500_000,
        },
        RetimingEnsembleCase {
            scenario: one,
            nominal_probability_ppm: 500_000,
        },
    ];
    assert!(matches!(
        evaluate_retiming_ensemble(&duplicate, options()),
        Err(RetimingEnsembleError::DuplicateScenarioId { .. })
    ));

    let invalid_mass = vec![RetimingEnsembleCase {
        scenario: greedy_trap(),
        nominal_probability_ppm: 999_999,
    }];
    assert_eq!(
        evaluate_retiming_ensemble(&invalid_mass, options()),
        Err(RetimingEnsembleError::ProbabilityMassMismatch {
            actual_ppm: 999_999,
        })
    );
}

#[test]
fn ensemble_rejects_invalid_budgets_and_tv_radius_before_evaluation() {
    let mut bad_budget = options();
    bad_budget.max_expanded_nodes_per_scenario = 0;
    assert_eq!(
        evaluate_retiming_ensemble(&cases(), bad_budget),
        Err(RetimingEnsembleError::InvalidSearchBudget)
    );

    let mut bad_radius = options();
    bad_radius.tv_radius_ppm = 1_000_001;
    assert_eq!(
        evaluate_retiming_ensemble(&cases(), bad_radius),
        Err(RetimingEnsembleError::InvalidTvRadius {
            radius_ppm: 1_000_001,
        })
    );
}

#[test]
fn ensemble_refuses_to_score_when_search_budget_finds_no_feasible_plan() {
    let mut constrained = options();
    constrained.max_generated_nodes_per_scenario = 1;
    assert!(matches!(
        evaluate_retiming_ensemble(&cases(), constrained),
        Err(RetimingEnsembleError::NoSearchPlanWithinBudget {
            scenario_id: _,
        })
    ));
}
