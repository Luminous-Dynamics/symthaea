// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Compile-time and runtime smoke test for the public RBKR integration API.
//!
//! This binary imports the API as an external crate consumer. It intentionally
//! cannot access private items from rbkr.rs, so it protects the public
//! integration boundary rather than only testing module-internal functions.

use symthaea_plasma_probes::{
    EmpiricalGramian,
    rbkr::{
        FiniteHorizonGramianConfig, RbkrConfig, ResponseBalancedBasis,
        ResponseGramianAggregationConfig, ResponseGramianScenario,
        aggregate_response_gramians, build_finite_horizon_gramians,
        evaluate_response_scenario_residuals,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let (family_reachability, family_observability) = aggregate_response_gramians(
        &[scenario],
        ResponseGramianAggregationConfig::default(),
    )?;
    let basis = ResponseBalancedBasis::from_empirical_gramians(
        &family_reachability,
        &family_observability,
        RbkrConfig::new(1),
    )?;
    let checkpoint_residual =
        basis.linearized_response_residual(&reachability, &observability)?;
    let pulse_residual =
        basis.linearized_pulse_response_residual(&[vec![1.0, 0.0]], &observability)?;
    let report = evaluate_response_scenario_residuals(&basis, &[scenario], 0.9)?;

    assert!(checkpoint_residual <= 1.0e-12);
    assert!(pulse_residual <= 1.0e-12);
    assert!(report.worst_case <= 1.0e-12);
    println!("rbkr_external_public_api_smoke=PASS");
    Ok(())
}
