// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end qualification fixture for a release-built third-party simulation
//! component. This test is ignored by default because it requires the standalone
//! `hello-simulation` artifact to be built first by the focused workflow.

use std::{env, fs, path::PathBuf};
use symthaea_extension_host::{
    ControlPlaneHost, GuestHealthState, CONTROL_WASM_PROFILE_V1,
};

fn required_path(name: &str) -> PathBuf {
    env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must point to the exact qualification artifact"))
}

#[test]
#[ignore = "requires the release-built hello-simulation Component fixture"]
fn hello_simulation_release_component_has_zero_ambient_imports() {
    let manifest_path = required_path("SYMTHAEA_HELLO_SIMULATION_MANIFEST");
    let component_path = required_path("SYMTHAEA_HELLO_SIMULATION_COMPONENT");
    let manifest_bytes = fs::read(&manifest_path).expect("read exact hello-simulation manifest");
    let component_bytes = fs::read(&component_path).expect("read release-built hello-simulation component");

    let inspection = ControlPlaneHost::default()
        .inspect(&manifest_bytes, &component_bytes)
        .expect("release-built simulation fixture must satisfy the empty-linker control host");

    assert_eq!(inspection.identity.id, "org.example.hello-simulation");
    assert_eq!(inspection.identity.version, "0.1.0");
    assert_eq!(inspection.identity.abi_major, 1);
    assert_eq!(inspection.identity.abi_minor, 0);
    assert_eq!(inspection.health.state, GuestHealthState::Ready);
    assert_eq!(inspection.wasm_profile, CONTROL_WASM_PROFILE_V1);

    // `ControlPlaneHost::inspect` pre-instantiates the *whole component* with an
    // empty Component linker before calling control. Success therefore proves
    // this exact release artifact has no unsatisfied WASI or Symthaea host
    // imports. This is technical compatibility only, not signer/admission,
    // scientific-validity, or invocation-authority evidence.
}
