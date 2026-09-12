// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::path::PathBuf;
use symthaea_extension_simulation_host::{
    GuestSimulationFailure, SimulationComponentHost, SimulationHostError,
};
use symthaea_sim_bridge::{
    EngineeringDomain, ExecutionMode, SimulationRequest, SolverKind,
};

fn fixture_paths() -> (PathBuf, PathBuf) {
    let component = std::env::var_os("SYMTHAEA_SIM_COMPONENT")
        .expect("SYMTHAEA_SIM_COMPONENT must point to the exact release Component");
    let manifest = std::env::var_os("SYMTHAEA_SIM_MANIFEST")
        .expect("SYMTHAEA_SIM_MANIFEST must point to the exact fixture manifest");
    (component.into(), manifest.into())
}

#[test]
#[ignore = "requires the release-built standalone hello-simulation Component"]
fn release_guest_executes_through_production_simulation_host() {
    let (component_path, manifest_path) = fixture_paths();
    let component = fs::read(&component_path).expect("read exact Component bytes");
    let manifest = fs::read(&manifest_path).expect("read exact manifest bytes");
    assert!(!component.is_empty(), "release Component must not be empty");
    assert!(!manifest.is_empty(), "manifest must not be empty");

    let host = SimulationComponentHost::default();
    let request = SimulationRequest::new(
        "production-host-fixture-1",
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "qualify typed production simulation host",
    )
    .with_parameter("a", 2.5, "fixture-unit", "production-host-qualification")
    .with_parameter("b", 1.5, "fixture-unit", "production-host-qualification");

    let invocation = host
        .invoke(&manifest, &component, &request)
        .expect("valid custom request must execute through the production host");
    let result = invocation.result();

    assert_eq!(invocation.extension_id(), "org.example.hello-simulation");
    assert_eq!(invocation.extension_version(), "0.1.0");
    assert_eq!(result.request_id, request.id);
    assert!(result.converged);
    assert_eq!(result.confidence, 0.0);
    assert_eq!(result.uncertainty.epistemic, 1.0);
    assert_eq!(result.uncertainty.aleatoric, 0.0);
    assert_eq!(result.metrics.len(), 1);
    assert_eq!(result.metrics[0].name, "fixture.parameter-sum");
    assert_eq!(result.metrics[0].value, 4.0);
    assert_eq!(result.metrics[0].unit, "fixture-unit");
    assert!(result
        .warnings
        .iter()
        .any(|warning| warning.contains("not engineering evidence")));

    // Technical execution must not mint provenance or engineering authority.
    assert_eq!(result.evidence.mode, ExecutionMode::Unknown);
    assert!(result.evidence.backend.is_none());
    assert!(!result.is_engineering_evidence());

    let unsupported = SimulationRequest::new(
        "production-host-fixture-unsupported",
        EngineeringDomain::Electrical,
        SolverKind::Circuit,
        "qualify typed unsupported-solver transport",
    );
    let error = host
        .invoke(&manifest, &component, &unsupported)
        .expect_err("fixture supports only custom solver requests");
    assert!(matches!(
        error,
        SimulationHostError::Guest(GuestSimulationFailure::UnsupportedSolver)
    ));
}
