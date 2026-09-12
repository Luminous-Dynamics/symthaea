// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Compile-time contract test for the public `simulation-provider-v1` world.
//!
//! `wasmtime::component::bindgen!` parses the exact multi-file WIT package from
//! `symthaea-extension-core`. If the world becomes syntactically invalid, its
//! package references stop resolving, exported type ownership changes, or the
//! generated call surface drifts, this target fails to compile before runtime
//! invocation code can silently diverge from the public ABI.

wasmtime::component::bindgen!({
    world: "simulation-provider-v1",
    path: "../../core/symthaea-extension-core/wit",
});

use exports::luminous::symthaea_extension::control::{ExtensionIdentity, HealthReport};
use exports::luminous::symthaea_extension::simulation_types::{
    SimulationOutput, SimulationRequest,
};

#[test]
fn simulation_provider_world_generates_typed_host_bindings() {
    assert!(
        std::any::type_name::<SimulationProviderV1>().ends_with("SimulationProviderV1")
    );
    assert!(std::any::type_name::<ExtensionIdentity>().contains("ExtensionIdentity"));
    assert!(std::any::type_name::<HealthReport>().contains("HealthReport"));
    assert!(std::any::type_name::<SimulationRequest>().contains("SimulationRequest"));
    assert!(std::any::type_name::<SimulationOutput>().contains("SimulationOutput"));
}

/// This function is intentionally never executed. Its body is a compile-time
/// assertion over the exact generated Wasmtime API used by the conformance and
/// eventual production hosts.
#[allow(dead_code)]
fn generated_call_surface_compiles(
    bindings: &SimulationProviderV1,
    store: &mut wasmtime::Store<()>,
    request: &SimulationRequest,
) {
    let _ = bindings
        .luminous_symthaea_extension_control()
        .call_identity(&mut *store);
    let _ = bindings
        .luminous_symthaea_extension_control()
        .call_health(&mut *store);
    let _ = bindings
        .luminous_symthaea_extension_simulation_provider()
        .call_simulate(&mut *store, request);
}
