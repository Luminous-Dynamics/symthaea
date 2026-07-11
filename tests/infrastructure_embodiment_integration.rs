// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regression test for robotics-plan bug 0.5: platforms constructible but
//! excluded from the Phase 2.5 drive gate, Gate 5 safety-override gate, or
//! the telemetry gate are silently never driven — `step()`, the moral gate,
//! and Red overrides never fire even though the bridge exists. `Infrastructure`
//! was one of six platforms named in that finding; the four gates were fixed
//! 2026-07-06 (see SYMTHAEA_ROBOTICS_IMPROVEMENT_PLAN_2026-07-06.md and
//! PLANETARY_ENERGY_COORDINATION_PLAN_2026-07-06.md Phase 0).
//!
//! IMPORTANT (discovered writing this test): `CognitiveLoopConfig.embodiment_platform`
//! is NOT read by `CognitiveLoopService::new()` — `constructor.rs` has no logic
//! that builds a bridge from it. Setting the field (directly, or via
//! `CognitiveLoopConfig::for_platform`/`for_platform_domain`) is decorative;
//! the ONLY path that actually constructs and wires an `EmbodimentBridge` is
//! calling `switch_embodiment()` on an already-constructed service. This
//! appears to be a fleet-wide gap, not Infrastructure-specific — other
//! platforms' scenario tests (e.g. `robotics_scenarios.rs`'s `make_service()`
//! helper) use the same config-only pattern and never assert
//! `telemetry.platform` equality, so the gap has been silently masked there
//! too. Flagged for the robotics plan; out of scope to fix fleet-wide here.
//!
//! This test proves constructible ⇔ driven end-to-end through the real
//! cognitive loop via the one construction path that actually works
//! (`switch_embodiment`), not just via direct crate-level unit tests on
//! `InfrastructureEmbodiment` (which only prove the bridge itself is
//! correct, not that the loop actually calls it).
//!
//! Run: `cargo test --features infrastructure --test infrastructure_embodiment_integration`

use symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn make_service_with_infrastructure() -> CognitiveLoopService {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        embodiment_blend_weight: 0.3,
        embodiment_step_interval: 1,
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .expect("CognitiveLoopService");
    // The only construction path that actually wires a bridge — see module
    // doc comment. `embodiment_platform` in config is not read by `new()`.
    service.switch_embodiment(EmbodimentPlatform::Infrastructure);
    service
}

#[cfg(feature = "infrastructure")]
#[test]
fn test_infrastructure_bridge_is_actually_driven() {
    let mut service = make_service_with_infrastructure();

    // Bug 0.5 regression: before the drive-gate fix, `total_steps` would
    // stay at 0 forever because the Phase 2.5 block excluded Infrastructure
    // even though `switch_embodiment` had already built the bridge.
    for _ in 0..10 {
        service.cycle("microgrid node monitoring nominal load");
    }

    let telem = service.embodiment_telemetry();
    assert_eq!(telem.platform, "infrastructure");
    assert!(
        telem.total_steps > 0,
        "Infrastructure bridge was constructed but never stepped — bug 0.5 regressed"
    );
}

#[cfg(feature = "infrastructure")]
#[test]
fn test_infrastructure_platform_specific_telemetry_reaches_loop() {
    let mut service = make_service_with_infrastructure();
    for _ in 0..5 {
        service.cycle("microgrid node monitoring nominal load");
    }
    let telem = service.embodiment_telemetry();
    assert!(
        !telem.platform_specific.is_empty(),
        "platform_specific telemetry should be populated once the bridge has stepped"
    );
    let parsed: serde_json::Value = serde_json::from_slice(&telem.platform_specific)
        .expect("platform_specific telemetry must be valid JSON");
    assert!(parsed.get("operating_mode").is_some());
}

#[cfg(feature = "infrastructure")]
#[test]
fn test_infrastructure_switch_embodiment_also_drives() {
    // Construct with no embodiment, then switch at runtime — exercises the
    // same working path as make_service_with_infrastructure() but starting
    // from an explicit EmbodimentPlatform::None to demonstrate the switch
    // itself (not just construction-time helper wiring) drives the bridge.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        embodiment_blend_weight: 0.3,
        embodiment_step_interval: 1,
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .expect("CognitiveLoopService");

    service.switch_embodiment(EmbodimentPlatform::Infrastructure);

    for _ in 0..10 {
        service.cycle("microgrid node monitoring nominal load");
    }

    let telem = service.embodiment_telemetry();
    assert_eq!(telem.platform, "infrastructure");
    assert!(
        telem.total_steps > 0,
        "Infrastructure bridge constructed via switch_embodiment was never stepped — bug 0.5 regressed"
    );
}
