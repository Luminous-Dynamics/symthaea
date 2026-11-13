// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Plasma-specific composition example for the existing UAP-OED selector.
//!
//! This example does not implement an optimizer. It compares:
//! 1. the existing independent-row greedy selector (misspecified for the
//!    declared shared preparation residuals);
//! 2. the existing selected-set correlated-covariance greedy selector; and
//! 3. each existing small-set exhaustive oracle.
//!
//! All Jacobians, costs, covariance values and scenarios are synthetic.

use std::collections::BTreeSet;
use std::error::Error;

use symthaea_uap_observatory::{
    BudgetedDesignConfig, CorrelatedDesignScenario, CorrelatedExperimentOption, DesignScenario,
    ExperimentOption, exhaustive_budgeted_design, exhaustive_correlated_budgeted_design,
    greedy_budgeted_design, greedy_correlated_budgeted_design,
};

const ACTIONS: [(&str, &str, f64, usize); 5] = [
    ("WCA_SCREEN", "F0", 1.0, 0),
    ("FTIR_CHEMISTRY", "F1", 3.0, 1),
    ("MOISTURE_CONTENT", "F1", 2.0, 2),
    ("LAP_SHEAR", "F2", 8.0, 3),
    ("AGED_LAP_SHEAR", "F3", 14.0, 4),
];

const DRY_ROWS: [[f64; 4]; 5] = [
    [1.10, 0.50, 0.10, 1.50],
    [1.00, 0.50, 0.10, 1.30],
    [0.00, 0.00, 1.50, 0.00],
    [1.00, 0.20, 0.10, 0.20],
    [0.30, 0.80, 0.10, 0.20],
];

const CONDITIONED_ROWS: [[f64; 4]; 5] = [
    [1.10, 0.50, 0.10, 1.50],
    [1.00, 0.50, 0.10, 1.30],
    [0.00, 0.00, 1.80, 0.00],
    [0.80, 0.15, 0.75, 0.20],
    [0.25, 0.75, 0.70, 0.20],
];

fn prior_precision() -> Vec<Vec<f64>> {
    // Target state: initial bond, aged-bond state, substrate moisture.
    // Nuisance state: common sample-preparation offset.
    vec![
        vec![0.25, 0.0, 0.0, 0.0],
        vec![0.0, 0.25, 0.0, 0.0],
        vec![0.0, 0.0, 0.25, 0.0],
        vec![0.0, 0.0, 0.0, 4.0],
    ]
}

fn residual_covariance() -> Vec<Vec<f64>> {
    // WCA/FTIR share a preparation-induced residual (rho=0.95).
    // Initial/aged lap-shear share a smaller preparation/test residual (rho=0.30).
    // The matrix is block diagonal and positive definite by construction.
    vec![
        vec![1.0, 0.95, 0.0, 0.0, 0.0],
        vec![0.95, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.30],
        vec![0.0, 0.0, 0.0, 0.30, 1.0],
    ]
}

fn scenarios_and_options() -> (
    Vec<DesignScenario>,
    Vec<ExperimentOption>,
    Vec<CorrelatedDesignScenario>,
    Vec<CorrelatedExperimentOption>,
    BudgetedDesignConfig,
) {
    let prior = prior_precision();
    let independent_scenarios = vec![
        DesignScenario {
            scenario_id: "PA6_DRY_CONDITION".to_owned(),
            target_parameter_count: 3,
            weight: 0.5,
            prior_precision: prior.clone(),
        },
        DesignScenario {
            scenario_id: "PA6_HYGROSCOPIC_CONDITION".to_owned(),
            target_parameter_count: 3,
            weight: 0.5,
            prior_precision: prior.clone(),
        },
    ];
    let correlated_scenarios = vec![
        CorrelatedDesignScenario {
            scenario_id: "PA6_DRY_CONDITION".to_owned(),
            target_parameter_count: 3,
            weight: 0.5,
            prior_precision: prior.clone(),
            measurement_jacobian_rows: DRY_ROWS.iter().map(|row| row.to_vec()).collect(),
            measurement_covariance: residual_covariance(),
        },
        CorrelatedDesignScenario {
            scenario_id: "PA6_HYGROSCOPIC_CONDITION".to_owned(),
            target_parameter_count: 3,
            weight: 0.5,
            prior_precision: prior,
            measurement_jacobian_rows: CONDITIONED_ROWS
                .iter()
                .map(|row| row.to_vec())
                .collect(),
            measurement_covariance: residual_covariance(),
        },
    ];

    let mut independent_options = Vec::with_capacity(ACTIONS.len());
    let mut correlated_options = Vec::with_capacity(ACTIONS.len());
    for (action_id, _fidelity, cost, observation_index) in ACTIONS {
        independent_options.push(ExperimentOption {
            option_id: action_id.to_owned(),
            cost,
            exclusive_group: None,
            jacobian_rows_by_scenario: vec![
                vec![DRY_ROWS[observation_index].to_vec()],
                vec![CONDITIONED_ROWS[observation_index].to_vec()],
            ],
        });
        correlated_options.push(CorrelatedExperimentOption {
            option_id: action_id.to_owned(),
            cost,
            exclusive_group: None,
            observation_indices_by_scenario: vec![
                vec![observation_index],
                vec![observation_index],
            ],
        });
    }

    let config = BudgetedDesignConfig {
        budget: 26.0,
        parameter_scales: vec![1.0, 1.0, 1.0, 0.5],
    };
    (
        independent_scenarios,
        independent_options,
        correlated_scenarios,
        correlated_options,
        config,
    )
}

fn missing_mandatory_actions(selected: &[String]) -> Vec<&'static str> {
    const REQUIRED: [&str; 3] = ["MOISTURE_CONTENT", "LAP_SHEAR", "AGED_LAP_SHEAR"];
    let selected: BTreeSet<_> = selected.iter().map(String::as_str).collect();
    REQUIRED
        .into_iter()
        .filter(|action| !selected.contains(*action))
        .collect()
}

fn print_selection(label: &str, selected: &[String], cost: f64, gain: f64) {
    println!("{label}_selected={}", selected.join(";"));
    println!("{label}_cost={cost:.6}");
    println!("{label}_target_marginal_eig_nats={gain:.12}");
    println!(
        "{label}_mandatory_missing={}",
        missing_mandatory_actions(selected).join(";")
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let (ind_scenarios, ind_options, cor_scenarios, cor_options, config) =
        scenarios_and_options();

    let independent_greedy =
        greedy_budgeted_design(&ind_scenarios, &ind_options, &config)?;
    let independent_exact =
        exhaustive_budgeted_design(&ind_scenarios, &ind_options, &config)?;
    let correlated_greedy =
        greedy_correlated_budgeted_design(&cor_scenarios, &cor_options, &config)?;
    let correlated_exact =
        exhaustive_correlated_budgeted_design(&cor_scenarios, &cor_options, &config)?;

    println!("benchmark=PLASMA_UAP_OED_CORRELATED_PREPARATION_V0");
    println!("evidence_class=SYNTHETIC");
    println!("optimizer_owner=UAP-OED-001");
    println!("covariance_owner=UAP-OED-003");
    println!("target_utility_owner=UAP-OED-002");
    println!("target_state=initial_bond,aged_bond,substrate_moisture");
    println!("nuisance_state=shared_sample_preparation_offset");
    println!("surface_activation_method_reference=ASTM_D6105-04(2019)");
    println!("rigid_plastic_lap_shear_reference=ASTM_D3163-01(2023)");
    println!("plastic_surface_preparation_control=ASTM_D2093-03(2025)");
    println!("moisture_temperature_aging_reference=ASTM_D1151-00(2022)");
    println!("cyclic_aging_reference=ASTM_D1183-03(2019)");
    println!("single_lap_interpretation_reference=ASTM_D4896-01(2024)");
    println!("standard_compliance_claimed=false");
    println!("scenario_count={}", cor_scenarios.len());
    println!("candidate_action_count={}", cor_options.len());
    println!("budget={:.6}", config.budget);
    println!("correlation_wca_ftir=0.95");
    println!("correlation_lap_aged=0.30");
    println!();

    print_selection(
        "independent_greedy",
        &independent_greedy.selected_option_ids,
        independent_greedy.total_cost,
        independent_greedy.expected_information_gain_nats,
    );
    print_selection(
        "correlated_greedy",
        &correlated_greedy.selected_option_ids,
        correlated_greedy.total_cost,
        correlated_greedy.expected_information_gain_nats,
    );
    print_selection(
        "independent_exact",
        &independent_exact.selected_option_ids,
        independent_exact.total_cost,
        independent_exact.expected_information_gain_nats,
    );
    print_selection(
        "correlated_exact",
        &correlated_exact.selected_option_ids,
        correlated_exact.total_cost,
        correlated_exact.expected_information_gain_nats,
    );
    println!(
        "independent_greedy_exact_gap_nats={:.12}",
        independent_exact.expected_information_gain_nats
            - independent_greedy.expected_information_gain_nats
    );
    println!(
        "correlated_greedy_exact_gap_nats={:.12}",
        correlated_exact.expected_information_gain_nats
            - correlated_greedy.expected_information_gain_nats
    );

    let independent_missing = missing_mandatory_actions(&independent_greedy.selected_option_ids);
    let correlated_missing = missing_mandatory_actions(&correlated_greedy.selected_option_ids);

    let independent_ids: BTreeSet<_> = independent_greedy
        .selected_option_ids
        .iter()
        .map(String::as_str)
        .collect();
    let correlated_ids: BTreeSet<_> = correlated_greedy
        .selected_option_ids
        .iter()
        .map(String::as_str)
        .collect();
    let correlated_exact_ids: BTreeSet<_> = correlated_exact
        .selected_option_ids
        .iter()
        .map(String::as_str)
        .collect();
    let independent_exact_missing =
        missing_mandatory_actions(&independent_exact.selected_option_ids);

    // Known-answer controls for this frozen synthetic profile:
    // independent-row greedy over-credits the correlated proxy pair, while
    // selected-set covariance retains both destructive product endpoints.
    assert!(independent_ids.contains("FTIR_CHEMISTRY"));
    assert!(!independent_ids.contains("LAP_SHEAR"));
    assert!(correlated_ids.contains("LAP_SHEAR"));
    assert!(correlated_ids.contains("AGED_LAP_SHEAR"));
    assert!(correlated_ids.contains("MOISTURE_CONTENT"));
    assert!(!correlated_ids.contains("FTIR_CHEMISTRY"));
    assert_eq!(correlated_ids, correlated_exact_ids);
    assert!(independent_exact_missing.is_empty());
    assert!(!independent_missing.is_empty());
    assert!(
        correlated_greedy.expected_information_gain_nats
            > independent_greedy.expected_information_gain_nats + 0.1
    );
    assert!(correlated_missing.is_empty());
    assert!(
        independent_exact.expected_information_gain_nats
            > independent_greedy.expected_information_gain_nats + 0.1
    );
    assert!(
        (correlated_exact.expected_information_gain_nats
            - correlated_greedy.expected_information_gain_nats)
            .abs()
            < 1.0e-10
    );
    println!(
        "independent_product_evidence_gate={}",
        if independent_missing.is_empty() {
            "PORTFOLIO_READY_FOR_PHYSICAL_REVIEW"
        } else {
            "ABSTAIN_MANDATORY_ENDPOINTS_MISSING"
        }
    );
    println!(
        "correlated_product_evidence_gate={}",
        if correlated_missing.is_empty() {
            "PORTFOLIO_READY_FOR_PHYSICAL_REVIEW"
        } else {
            "ABSTAIN_MANDATORY_ENDPOINTS_MISSING"
        }
    );

    // Deliberately do not convert a design selection into a product acceptance.
    println!("automatic_experiment_execution=false");
    println!("automatic_quality_release=false");
    println!("physical_measurements=false");
    println!("claim_ceiling=synthetic selector composition only");
    Ok(())
}
