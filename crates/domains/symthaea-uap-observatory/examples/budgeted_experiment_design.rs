// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_uap_observatory::{
    BudgetedDesignConfig, DesignScenario, ExperimentOption, exhaustive_budgeted_design,
    greedy_budgeted_design,
};

fn main() -> Result<(), Box<dyn Error>> {
    // Synthetic whitened linearized Jacobians for target XYZ plus a sensor
    // clock-offset nuisance parameter. These rows are a benchmark fixture,
    // not measurements from a real observatory.
    let scenarios = vec![
        DesignScenario {
            scenario_id: "nominal-target-region".to_string(),
            weight: 0.65,
            prior_precision: vec![
                vec![0.01, 0.0, 0.0, 0.0],
                vec![0.0, 0.01, 0.0, 0.0],
                vec![0.0, 0.0, 0.01, 0.0],
                vec![0.0, 0.0, 0.0, 0.001],
            ],
        },
        DesignScenario {
            scenario_id: "high-altitude-target-region".to_string(),
            weight: 0.35,
            prior_precision: vec![
                vec![0.01, 0.0, 0.0, 0.0],
                vec![0.0, 0.01, 0.0, 0.0],
                vec![0.0, 0.0, 0.01, 0.0],
                vec![0.0, 0.0, 0.0, 0.001],
            ],
        },
    ];

    let options = vec![
        ExperimentOption {
            option_id: "station-a-low-fidelity".to_string(),
            cost: 1.0,
            exclusive_group: Some("station-a".to_string()),
            jacobian_rows_by_scenario: vec![
                vec![vec![0.15, 1.20, 0.10, 0.02], vec![1.00, 0.10, 0.20, 0.00]],
                vec![vec![0.10, 0.75, 0.55, 0.03], vec![0.85, 0.10, 0.25, 0.00]],
            ],
        },
        ExperimentOption {
            option_id: "station-a-high-fidelity".to_string(),
            cost: 2.2,
            exclusive_group: Some("station-a".to_string()),
            jacobian_rows_by_scenario: vec![
                vec![vec![0.22, 1.80, 0.15, 0.03], vec![1.50, 0.15, 0.30, 0.00]],
                vec![vec![0.15, 1.10, 0.82, 0.04], vec![1.25, 0.15, 0.38, 0.00]],
            ],
        },
        ExperimentOption {
            option_id: "station-b-baseline".to_string(),
            cost: 1.25,
            exclusive_group: None,
            jacobian_rows_by_scenario: vec![
                vec![vec![0.20, 0.10, 1.35, 0.08], vec![0.90, 0.12, 0.10, 0.00]],
                vec![vec![0.25, 0.18, 0.92, 0.12], vec![0.75, 0.20, 0.12, 0.02]],
            ],
        },
        ExperimentOption {
            option_id: "clock-calibration-pulse".to_string(),
            cost: 0.65,
            exclusive_group: Some("clock-reference".to_string()),
            jacobian_rows_by_scenario: vec![
                vec![vec![0.01, 0.00, 0.00, 1.80]],
                vec![vec![0.02, 0.00, 0.00, 1.60]],
            ],
        },
        ExperimentOption {
            option_id: "platform-excitation-maneuver".to_string(),
            cost: 1.1,
            exclusive_group: None,
            jacobian_rows_by_scenario: vec![
                vec![vec![0.25, 0.20, 0.15, 0.90], vec![0.45, 0.10, 0.20, 0.75]],
                vec![vec![0.18, 0.25, 0.22, 1.10], vec![0.36, 0.15, 0.28, 0.88]],
            ],
        },
    ];

    let config = BudgetedDesignConfig {
        budget: 3.0,
        // Metres for XYZ and milliseconds for the clock-offset state.
        parameter_scales: vec![100.0, 100.0, 100.0, 1.0],
    };

    let greedy = greedy_budgeted_design(&scenarios, &options, &config)?;
    let exact = exhaustive_budgeted_design(&scenarios, &options, &config)?;

    println!("epistemic_status=SYNTHETIC_LINEARIZED_BUDGETED_DESIGN_NOT_DEPLOYMENT_ADVICE");
    println!("state=target_x,target_y,target_z,clock_offset");
    println!("budget={:.6}", config.budget);
    println!("candidate_options={}", options.len());
    println!("exclusive_fidelity_groups=station-a,clock-reference");
    println!("greedy_selected={}", greedy.selected_option_ids.join(";"));
    println!("greedy_cost={:.6}", greedy.total_cost);
    println!(
        "greedy_expected_eig_nats={:.9}",
        greedy.expected_information_gain_nats
    );
    println!(
        "greedy_worst_case_eig_nats={:.9}",
        greedy.worst_case_information_gain_nats
    );
    println!("exact_selected={}", exact.selected_option_ids.join(";"));
    println!("exact_cost={:.6}", exact.total_cost);
    println!(
        "exact_expected_eig_nats={:.9}",
        exact.expected_information_gain_nats
    );
    println!(
        "greedy_optimality_gap_nats={:.9}",
        exact.expected_information_gain_nats - greedy.expected_information_gain_nats
    );
    println!("selection_step,option_id,cost,marginal_expected_eig,marginal_eig_per_cost");
    for (index, step) in greedy.steps.iter().enumerate() {
        println!(
            "{index},{},{:.6},{:.9},{:.9}",
            step.option_id,
            step.cost,
            step.marginal_expected_gain_nats,
            step.marginal_gain_per_cost,
        );
    }
    println!("scenario,information_gain_nats,data_only_rank,data_only_nullity,condition_number");
    for score in &greedy.scenario_scores {
        println!(
            "{},{:.9},{},{},{}",
            score.scenario_id,
            score.information_gain_nats,
            score.data_only_rank,
            score.data_only_nullity,
            score
                .data_only_condition_number
                .map(|value| format!("{value:.9}"))
                .unwrap_or_else(|| "unobservable".to_string()),
        );
    }
    println!(
        "claim_boundary=linearized_whitened_independent_action_rows; shared_cross_sensor_covariance_not_included"
    );
    Ok(())
}
