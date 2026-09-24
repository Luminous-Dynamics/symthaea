// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! External qualification fixtures. These tests are ignored in ordinary CI
//! because they require a real `ngspice` executable; a qualification runner can
//! opt in with `cargo test -p symthaea-ngspice-bridge --test external_ngspice -- --ignored`.

use std::path::PathBuf;
use symthaea_ngspice_bridge::NgspiceBridge;
use symthaea_sim_bridge::{
    EngineeringDomain, ExecutionMode, SimulationBackend, SimulationRequest, SolverKind,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
#[ignore = "requires a real ngspice executable"]
fn resistive_divider_produces_five_volts_with_external_evidence() {
    let bridge = NgspiceBridge::default()
        .with_netlist_path(fixture("resistive_divider.cir"))
        .with_metric_unit("vout_final", "V");
    let mut request = SimulationRequest::new(
        "divider-external",
        EngineeringDomain::Electrical,
        SolverKind::Circuit,
        "qualify a 10 V 1k/1k resistive divider",
    );
    request.requested_metrics = vec!["vout_final".into()];

    let result = bridge.run(&request).unwrap();
    assert!(result.converged);
    assert!(result.is_engineering_evidence());
    assert_eq!(result.evidence.mode, ExecutionMode::ExternalSolver);
    assert!((result.metrics[0].value - 5.0).abs() < 1e-9);
}

#[test]
#[ignore = "requires a real ngspice executable"]
fn rc_one_time_constant_matches_closed_form_reference() {
    let bridge = NgspiceBridge::default()
        .with_netlist_path(fixture("rc_step.cir"))
        .with_metric_unit("vout_1ms", "V");
    let mut request = SimulationRequest::new(
        "rc-external",
        EngineeringDomain::Electrical,
        SolverKind::Circuit,
        "qualify a 1 kOhm / 1 uF step response at one time constant",
    );
    request.requested_metrics = vec!["vout_1ms".into()];

    let result = bridge.run(&request).unwrap();
    let expected = 1.0 - (-1.0_f64).exp();
    assert!(result.is_engineering_evidence());
    assert!((result.metrics[0].value - expected).abs() < 2e-3);
}
