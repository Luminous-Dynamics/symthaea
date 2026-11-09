// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_uap_observatory::{
    BudgetedDesignConfig, CorrelatedDesignScenario, CorrelatedExperimentOption, DesignScenario,
    ExperimentOption, exhaustive_correlated_budgeted_design, greedy_budgeted_design,
    greedy_correlated_budgeted_design,
};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Symthaea selected-set correlated observation design reference");
    println!(
        "epistemic_status=SYNTHETIC_GAUSSIAN_COMMON_MODE_CONTROL_NOT_FIELD_COVARIANCE_VALIDATION"
    );
    println!("model=two_sensors_measure_one_scalar_target");
    println!("prior_precision=1.0");
    println!("jacobian_rows=1.0,1.0");
    println!(
        "common_mode_correlation_rho,correlated_target_eig_nats,independent_assumption_target_eig_nats,correlated_selected_cost,correlated_fisher_information"
    );

    let parameter_scales = vec![1.0];
    let config = BudgetedDesignConfig {
        budget: 2.0,
        parameter_scales: parameter_scales.clone(),
    };

    let independent_scenarios = vec![DesignScenario {
        scenario_id: "single-target".to_string(),
        target_parameter_count: 1,
        weight: 1.0,
        prior_precision: vec![vec![1.0]],
    }];
    let independent_options = vec![
        ExperimentOption {
            option_id: "sensor-a".to_string(),
            cost: 1.0,
            exclusive_group: None,
            jacobian_rows_by_scenario: vec![vec![vec![1.0]]],
        },
        ExperimentOption {
            option_id: "sensor-b".to_string(),
            cost: 1.0,
            exclusive_group: None,
            jacobian_rows_by_scenario: vec![vec![vec![1.0]]],
        },
    ];
    let independent =
        greedy_budgeted_design(&independent_scenarios, &independent_options, &config)?;

    for rho in [0.0_f64, 0.5, 0.9, 0.99] {
        let correlated_scenarios = vec![CorrelatedDesignScenario {
            scenario_id: "single-target".to_string(),
            target_parameter_count: 1,
            weight: 1.0,
            prior_precision: vec![vec![1.0]],
            measurement_jacobian_rows: vec![vec![1.0], vec![1.0]],
            measurement_covariance: vec![vec![1.0, rho], vec![rho, 1.0]],
        }];
        let correlated_options = vec![CorrelatedExperimentOption {
            option_id: "two-sensor-pair".to_string(),
            cost: 2.0,
            exclusive_group: None,
            observation_indices_by_scenario: vec![vec![0, 1]],
        }];
        let correlated =
            greedy_correlated_budgeted_design(&correlated_scenarios, &correlated_options, &config)?;
        let exact = exhaustive_correlated_budgeted_design(
            &correlated_scenarios,
            &correlated_options,
            &config,
        )?;
        let expected_information = (1.0 + 2.0 / (1.0 + rho)).ln();
        let correlated_fisher_information = 2.0 / (1.0 + rho);

        assert!((correlated.expected_information_gain_nats - expected_information).abs() < 1.0e-10);
        assert!(
            (correlated.expected_information_gain_nats - exact.expected_information_gain_nats)
                .abs()
                < 1.0e-12
        );
        if rho == 0.0 {
            assert!(
                (correlated.expected_information_gain_nats
                    - independent.expected_information_gain_nats)
                    .abs()
                    < 1.0e-12
            );
        }
        if rho > 0.0 {
            assert!(
                correlated.expected_information_gain_nats
                    < independent.expected_information_gain_nats
            );
        }

        println!(
            "{rho:.2},{:.12},{:.12},{:.6},{correlated_fisher_information:.12}",
            correlated.expected_information_gain_nats,
            independent.expected_information_gain_nats,
            correlated.total_cost,
        );
    }

    println!("diagonal_limit_equivalence=true");
    println!("common_mode_information_saturation=true");
    println!("correlation_is_selected_set_dependent=true");
    println!("The independent-noise column is an intentionally misspecified control for rho > 0.");
    Ok(())
}
