// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Compile-time contract test for the public `simulation-provider-v1` world.
//!
//! `wasmtime::component::bindgen!` parses the exact multi-file WIT package from
//! `symthaea-extension-core`. If the world becomes syntactically invalid or its
//! package references stop resolving, this test target fails to compile before a
//! runtime host can silently drift from the public ABI.

wasmtime::component::bindgen!({
    world: "simulation-provider-v1",
    path: "../../core/symthaea-extension-core/wit",
});

#[test]
fn simulation_provider_world_generates_typed_host_bindings() {
    let generated = std::any::type_name::<SimulationProviderV1>();
    assert!(generated.ends_with("SimulationProviderV1"));
}
