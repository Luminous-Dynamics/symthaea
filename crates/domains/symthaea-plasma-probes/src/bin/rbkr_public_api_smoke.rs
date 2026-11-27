// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Compile-time and runtime smoke test for the public RBKR integration API.
//!
//! This binary imports the API as an external crate consumer. It intentionally
//! cannot access private items from rbkr.rs, so it protects the public
//! integration boundary rather than only testing module-internal functions.

use symthaea_plasma_probes::{
    velocity_kick_remap::{
        VelocityKickRemapJacobianConfig, linearize_velocity_kick_remap,
        remap_distribution_after_velocity_kick, remap_distribution_after_velocity_kick_weighted,
    },
    pulse_map::{PhaseSpaceTangent, SinusoidalVelocityKick, check_pulse_jacobians_central_difference},
    EmpiricalGramian,
    rbkr::{
        FiniteHorizonGramianConfig, RbkrConfig, ResponseBalancedBasis,
        ResponseGramianAggregationConfig, ResponseGramianScenario, ResponseScenarioRole,
        aggregate_response_gramians, build_finite_horizon_gramians,
        evaluate_response_scenario_residuals,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pulse = SinusoidalVelocityKick::new(
        -std::f64::consts::PI,
        2.0 * std::f64::consts::PI,
        2,
        0.012,
        0.37,
    )?;
    let pulse_check =
        check_pulse_jacobians_central_difference(&pulse, 0.23, -0.4, 1.0e-5)?;
    assert!(pulse_check.state_jacobian_max_abs_error < 1.0e-8);
    assert!(pulse_check.parameter_jacobian_max_abs_error < 1.0e-8);
    assert!(pulse_check.symplectic_defect < 1.0e-14);

    let event_time = pulse.application_time_sensitivity(
        0.23,
        -0.4,
        0.0,
        0.0,
        PhaseSpaceTangent {
            dx_dt: 0.0,
            dv_dt: 0.0,
        },
    )?;
    assert!(event_time.x.is_finite() && event_time.v.is_finite());

    let velocity_edges = (0..=16)
        .map(|index| -4.0 + 0.5 * index as f64)
        .collect::<Vec<_>>();
    let cell_averages = vec![vec![
        0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
    ]];
    let remap = remap_distribution_after_velocity_kick(
        &pulse,
        &[std::f64::consts::PI / 2.0],
        &velocity_edges,
        &cell_averages,
    )?;
    assert!(remap.distribution[0].iter().all(|value| *value >= 0.0));
    assert!(remap.total_boundary_outflow < 1.0e-12);
    assert!(remap.total_mass_balance_residual.abs() < 1.0e-12);

    let spatial_weighted_remap = remap_distribution_after_velocity_kick_weighted(
        &pulse,
        &[std::f64::consts::PI / 2.0],
        &[0.25],
        &velocity_edges,
        &cell_averages,
    )?;
    assert!((spatial_weighted_remap.total_mass_before - 0.25 * remap.total_mass_before).abs() < 1.0e-12);
    assert!((spatial_weighted_remap.total_mass_after - 0.25 * remap.total_mass_after).abs() < 1.0e-12);
    let remap_jacobian = linearize_velocity_kick_remap(
        &pulse,
        &[std::f64::consts::PI / 2.0],
        &[0.25],
        &velocity_edges,
        &cell_averages,
        VelocityKickRemapJacobianConfig::new(1.0e-6, 10_000),
    )?;
    assert!(remap_jacobian.minimum_topology_margin > 1.0e-6);
    assert!(remap_jacobian.rows[0].distribution_wrt_amplitude.iter().all(|v| v.is_finite()));
    assert!(remap_jacobian.rows[0].boundary_outflow.amplitude.is_finite());

    let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let input = vec![vec![1.0], vec![0.0]];
    let output = vec![vec![1.0, 0.0]];

    let (reachability, observability) = build_finite_horizon_gramians(
        std::slice::from_ref(&identity),
        std::slice::from_ref(&input),
        &[vec![1.0]],
        &[],
        std::slice::from_ref(&output),
        &[vec![1.0]],
        FiniteHorizonGramianConfig::default(),
    )?;

    let scenario = ResponseGramianScenario {
        scenario_id: "public-api-smoke",
        role: ResponseScenarioRole::Training,
        trajectory_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        task_contract_id: "public-api-smoke-v1",
        checkpoint_id: "checkpoint-smoke",
        checkpoint_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        state_coordinate_id: "vlasov-1d1v-layout-smoke",
        state_metric_id: "nondimensional-smoke",
        reachability: &reachability,
        observability: &observability,
        scenario_weight: 1.0,
    };

    let (family_reachability, family_observability) =
        aggregate_response_gramians(&[scenario], ResponseGramianAggregationConfig::default())?;
    let mut basis = ResponseBalancedBasis::from_empirical_gramians(
        &family_reachability,
        &family_observability,
        RbkrConfig::new(1),
    )?;
    let checkpoint_residual = basis.linearized_response_residual(&reachability, &observability)?;
    let pulse_residual =
        basis.linearized_pulse_response_residual(&[vec![1.0, 0.0]], &observability)?;
    let report = evaluate_response_scenario_residuals(&basis, &[scenario], 0.9)?;

    assert!(checkpoint_residual <= 1.0e-12);
    assert!(pulse_residual <= 1.0e-12);
    assert!(report.worst_case <= 1.0e-12);

    // A separate rare-response scenario exercises the public robust greedy
    // enrichment API from an external consumer crate.
    let mut rare_reachability = EmpiricalGramian::new(2)?;
    rare_reachability.add_direction(&[0.0, 1.0], 1.0)?;
    let mut rare_observability = EmpiricalGramian::new(2)?;
    rare_observability.add_direction(&[0.0, 1.0], 1.0)?;
    let rare_scenario = ResponseGramianScenario {
        scenario_id: "public-api-smoke-rare",
        role: ResponseScenarioRole::DevelopmentChallenge,
        trajectory_sha256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        task_contract_id: "public-api-smoke-v1",
        checkpoint_id: "checkpoint-smoke",
        checkpoint_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        state_coordinate_id: "vlasov-1d1v-layout-smoke",
        state_metric_id: "nondimensional-smoke",
        reachability: &rare_reachability,
        observability: &rare_observability,
        scenario_weight: 0.49,
    };
    let development_scenario = ResponseGramianScenario {
        role: ResponseScenarioRole::DevelopmentChallenge,
        ..scenario
    };
    let challenge = [development_scenario, rare_scenario];
    let receipt = basis.enrich_from_worst_scenario(&challenge, RbkrConfig::new(2))?;
    assert_eq!(receipt.scenario_id, "public-api-smoke-rare");
    assert_eq!(
        receipt.checkpoint_sha256,
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    );
    assert_eq!(receipt.state_coordinate_id, "vlasov-1d1v-layout-smoke");
    assert_eq!(receipt.new_order, 2);
    assert!(receipt.post_response_residual < 1.0e-12);

    println!("rbkr_external_public_api_smoke=PASS");
    Ok(())
}
