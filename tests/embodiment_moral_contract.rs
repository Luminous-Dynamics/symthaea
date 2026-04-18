// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-platform EmbodimentBridge safety contract.
//!
//! Verifies that EVERY concrete `EmbodimentBridge` implementation satisfies
//! the `apply_moral_gate` contract. Per-platform tests already check this
//! in isolation, but a new platform can silently fall back to the trait
//! default no-op (`fn apply_moral_gate(&mut self, _: MoralGateInput) {}`)
//! and pass its own tests. This test catches that regression class by
//! running the same checks against every platform uniformly.
//!
//! The contract:
//!   1. `ahimsa_violated = true` → next step's safety_level is Red
//!   2. `consent_violation = true` → next step's safety_level is Orange
//!   3. After `reset()`, moral state is cleared → Green at high phi
//!
//! Run: cargo test --features \
//!   humanoid,flight,vehicle,auv,helicopter,manipulator,surgical,orbital,quadruped,exoskeleton \
//!   --test embodiment_moral_contract
//!
//! Any platform whose feature is NOT enabled is simply skipped.

#![allow(unused_imports)]

use symthaea_core::embodiment::{EmbodimentBridge, MoralGateInput, MotorSafetyLevel};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

/// Run the three-check moral contract against any `EmbodimentBridge`.
///
/// Parameterized on `platform_name` for clear assertion messages and on
/// `dt` because different platforms have different physics time steps.
fn assert_moral_contract<B: EmbodimentBridge>(mut bridge: B, platform_name: &str, dt: f32) {
    let hv = ContinuousHV::random(16384, 42);

    // ── 1. Ahimsa forces Red regardless of phi ──────────────────────
    bridge.apply_moral_gate(MoralGateInput {
        verdict: MoralGateInput::VERDICT_SAFE,
        consent_violation: false,
        ahimsa_violated: true,
    });
    let r = bridge.step(&hv, dt, 0.9);
    assert_eq!(
        r.safety_level,
        MotorSafetyLevel::Red,
        "{platform_name}: ahimsa_violated must force Red at any phi \
         (got {:?}). If this fires, the platform's EmbodimentBridge impl \
         is using the trait default no-op — add an inherent \
         `apply_moral_gate` and forward to it.",
        r.safety_level,
    );

    // ── 2. Consent violation → Orange (fresh bridge to avoid sticky state) ──
    bridge.reset();
    bridge.apply_moral_gate(MoralGateInput {
        verdict: MoralGateInput::VERDICT_SAFE,
        consent_violation: true,
        ahimsa_violated: false,
    });
    let r = bridge.step(&hv, dt, 0.9);
    assert_eq!(
        r.safety_level,
        MotorSafetyLevel::Orange,
        "{platform_name}: consent_violation must force Orange at any phi \
         (got {:?})",
        r.safety_level,
    );

    // ── 3. reset() clears moral state ───────────────────────────────
    bridge.reset();
    let r = bridge.step(&hv, dt, 0.9);
    assert_eq!(
        r.safety_level,
        MotorSafetyLevel::Green,
        "{platform_name}: reset() must clear moral_safety — next step at \
         phi=0.9 should return Green (got {:?}). If stuck on Orange/Red, \
         the reset() impl is forgetting to null `moral_safety`.",
        r.safety_level,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// One test per platform, each gated on its feature flag.
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "humanoid")]
#[test]
fn moral_contract_humanoid() {
    let bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(
        &GenesisSeed::from_phrase("contract-humanoid"),
    );
    assert_moral_contract(bridge, "humanoid", 0.025);
}

#[cfg(feature = "flight")]
#[test]
fn moral_contract_flight() {
    let bridge = symthaea_flight::embodiment::FlightEmbodiment::new(&GenesisSeed::from_phrase(
        "contract-flight",
    ));
    assert_moral_contract(bridge, "flight", 0.002);
}

#[cfg(feature = "vehicle")]
#[test]
fn moral_contract_vehicle() {
    let bridge = symthaea_vehicle::embodiment::VehicleEmbodiment::new(&GenesisSeed::from_phrase(
        "contract-vehicle",
    ));
    assert_moral_contract(bridge, "vehicle", 0.005);
}

#[cfg(feature = "auv")]
#[test]
fn moral_contract_auv() {
    let bridge =
        symthaea_auv::embodiment::AuvEmbodiment::new(&GenesisSeed::from_phrase("contract-auv"));
    assert_moral_contract(bridge, "auv", 0.01);
}

#[cfg(feature = "helicopter")]
#[test]
fn moral_contract_helicopter() {
    let bridge = symthaea_helicopter::embodiment::HelicopterEmbodiment::new(
        &GenesisSeed::from_phrase("contract-helicopter"),
    );
    assert_moral_contract(bridge, "helicopter", 1.0 / 300.0);
}

#[cfg(feature = "manipulator")]
#[test]
fn moral_contract_manipulator() {
    let bridge = symthaea_manipulator::embodiment::ManipulatorEmbodiment::new(
        &GenesisSeed::from_phrase("contract-manipulator"),
    );
    assert_moral_contract(bridge, "manipulator", 0.001);
}

#[cfg(feature = "surgical")]
#[test]
fn moral_contract_surgical() {
    let bridge = symthaea_surgical::embodiment::SurgicalEmbodiment::new(&GenesisSeed::from_phrase(
        "contract-surgical",
    ));
    assert_moral_contract(bridge, "surgical", 0.001);
}

#[cfg(feature = "orbital")]
#[test]
fn moral_contract_orbital() {
    let bridge = symthaea_orbital::embodiment::OrbitalEmbodiment::new(&GenesisSeed::from_phrase(
        "contract-orbital",
    ));
    assert_moral_contract(bridge, "orbital", 0.01);
}

#[cfg(feature = "quadruped")]
#[test]
fn moral_contract_quadruped() {
    let bridge = symthaea_quadruped::embodiment::QuadrupedEmbodiment::new(
        &GenesisSeed::from_phrase("contract-quadruped"),
    );
    assert_moral_contract(bridge, "quadruped", 0.005);
}

#[cfg(feature = "exoskeleton")]
#[test]
fn moral_contract_exoskeleton() {
    let bridge = symthaea_exoskeleton::embodiment::ExoskeletonEmbodiment::new(
        &GenesisSeed::from_phrase("contract-exoskeleton"),
    );
    assert_moral_contract(bridge, "exoskeleton", 0.001);
}
