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

    // ── Baseline: what this bridge reports with NO moral input ──────
    //
    // REWRITTEN 2026-07-31 to be floor-based and baseline-relative. The original
    // asserted EXACT tiers (consent => exactly Orange; after reset => exactly Green),
    // which silently assumed every bridge is Phi-driven and lands on Green at phi=0.9.
    // `helios` violates that legitimately: its safety level comes from battery
    // state-of-charge, and a default-constructed bridge has soc=0.0, so it is Red before
    // any ethics input exists. Composing a moral veto most-restrictive-wins can then
    // never yield exactly Orange.
    //
    // Demanding an exact tier forbids a platform from being MORE conservative than the
    // contract for its own domain reasons, which is backwards for a safety contract. The
    // assertions below are floors ("at least this restrictive"), which still catch the
    // real defect class -- detritivore returned Green under an ahimsa violation, and
    // Green < Red fails a floor check just as it failed an equality check.
    let baseline = bridge.step(&hv, dt, 0.9).safety_level;
    bridge.reset();

    // ── 1. Ahimsa forces Red regardless of phi ──────────────────────
    bridge.apply_moral_gate(MoralGateInput {
        verdict: MoralGateInput::VERDICT_SAFE,
        consent_violation: false,
        ahimsa_violated: true,
    });
    let r = bridge.step(&hv, dt, 0.9);
    assert!(
        r.safety_level >= MotorSafetyLevel::Red,
        "{platform_name}: ahimsa_violated must force Red at any phi (got {:?}). \
         If this fires, the platform's EmbodimentBridge impl is ignoring the moral gate \
         -- either inheriting the trait default no-op, or overriding it with an EMPTY \
         body (which a grep audit counts as compliant).",
        r.safety_level,
    );

    // ── 2. Consent violation → at least Orange ──────────────────────
    bridge.reset();
    bridge.apply_moral_gate(MoralGateInput {
        verdict: MoralGateInput::VERDICT_SAFE,
        consent_violation: true,
        ahimsa_violated: false,
    });
    let r = bridge.step(&hv, dt, 0.9);
    assert!(
        r.safety_level >= MotorSafetyLevel::Orange,
        "{platform_name}: consent_violation must force AT LEAST Orange at any phi \
         (got {:?}). More restrictive is allowed; less is not.",
        r.safety_level,
    );

    // ── 3. reset() clears moral state, restoring the baseline ───────
    bridge.reset();
    let r = bridge.step(&hv, dt, 0.9);
    assert_eq!(
        r.safety_level, baseline,
        "{platform_name}: reset() must clear the moral veto and restore this bridge's \
         own baseline ({baseline:?}), got {:?}. If it is stuck at Orange/Red, the reset() \
         impl is forgetting to null `moral_safety`.",
        r.safety_level,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// One test per platform, each gated on its feature flag.
// ═══════════════════════════════════════════════════════════════════════════

// RENAMED 2026-07-31. This was called `moral_contract_humanoid`, but it constructs
// `MotorBridge` -- the main-crate bridge the LIVE COGNITIVE LOOP drives -- not the
// `symthaea-humanoid` platform type. The name asserted coverage the test did not
// provide, and it hid a real gap: `HumanoidEmbodiment` was untested here despite a
// test appearing to name it. A genuine humanoid case now follows below.
//
// Found by simulating the regression this file exists to catch (neutering
// MotorBridge's `apply_moral_gate` override): BOTH this test and the added
// motor-bridge one failed together, which is what revealed they exercise the same
// type. Validating a guard against the broken state, not just the fixed one.
#[cfg(feature = "humanoid")]
#[test]
fn moral_contract_motor_bridge() {
    let bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(
        &GenesisSeed::from_phrase("contract-motor-bridge"),
    );
    assert_moral_contract(bridge, "motor_bridge (LIVE cognitive-loop path)", 0.025);
}

/// The actual `symthaea-humanoid` platform type, which nothing covered until now.
#[cfg(feature = "humanoid")]
#[test]
fn moral_contract_humanoid() {
    let bridge = symthaea_humanoid::embodiment::HumanoidEmbodiment::new(&GenesisSeed::from_phrase(
        "contract-humanoid",
    ));
    assert_moral_contract(bridge, "humanoid", 0.025);
}

#[cfg(feature = "flight")]
#[test]
fn moral_contract_flight() {
    let bridge = symthaea_multirotor::embodiment::FlightEmbodiment::new(&GenesisSeed::from_phrase(
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

// ═══════════════════════════════════════════════════════════════════════════
// The nine implementors this contract did not cover until 2026-07-31.
//
// A static audit (grep for `fn apply_moral_gate`) showed eight of the nine
// override it and one -- PhoneBridge -- did NOT, inheriting the trait's no-op
// default while executing taps, swipes and OpenUrl on a real connected device.
// That gap is now fixed in symthaea-phone-embodiment.
//
// These tests exist because static presence of an override is NOT behavioural
// compliance: an override can forward to a stub, set the wrong tier, or be
// skipped by the step path that actually computes safety_level. Only running the
// contract establishes that. They also convert the one-time audit into a standing
// guard, which is the whole point of this file.
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "agribot")]
#[test]
fn moral_contract_agribot() {
    let bridge = symthaea_agribot::embodiment::AgribotEmbodiment::new(&GenesisSeed::from_phrase(
        "contract-agribot",
    ));
    assert_moral_contract(bridge, "agribot", 0.01);
}

#[cfg(feature = "biota")]
#[test]
fn moral_contract_biota() {
    let bridge = symthaea_biota::embodiment::BiotaEmbodiment::new(&GenesisSeed::from_phrase(
        "contract-biota",
    ));
    assert_moral_contract(bridge, "biota", 0.01);
}

#[cfg(feature = "clime")]
#[test]
fn moral_contract_clime() {
    let bridge = symthaea_clime::embodiment::ClimeEmbodiment::new(&GenesisSeed::from_phrase(
        "contract-clime",
    ));
    assert_moral_contract(bridge, "clime", 0.01);
}

#[cfg(feature = "infrastructure")]
#[test]
fn moral_contract_infrastructure() {
    let bridge = symthaea_infrastructure::embodiment::InfrastructureEmbodiment::new(
        &GenesisSeed::from_phrase("contract-infrastructure"),
    );
    assert_moral_contract(bridge, "infrastructure", 0.01);
}

#[cfg(feature = "scavenger")]
#[test]
fn moral_contract_scavenger() {
    let bridge = symthaea_scavenger::embodiment::ScavengerEmbodiment::new(
        &GenesisSeed::from_phrase("contract-scavenger"),
    );
    assert_moral_contract(bridge, "scavenger", 0.005);
}

#[cfg(feature = "subterranean")]
#[test]
fn moral_contract_subterranean() {
    let bridge = symthaea_subterranean::embodiment::SubterraneanEmbodiment::new(
        &GenesisSeed::from_phrase("contract-subterranean"),
    );
    assert_moral_contract(bridge, "subterranean", 0.005);
}

// NOT feature-gated: `pub mod embodiment;` is unconditional in lib.rs, so these two
// always exist. An earlier draft wrote #[cfg(feature = "detritivore")] / "helios" --
// neither feature exists, so both tests would have silently never compiled. That is
// precisely the dead-test failure class this file was just rescued from, nearly
// reintroduced while extending it. Verified against Cargo.toml's feature list.
#[test]
fn moral_contract_detritivore() {
    let bridge = symthaea::embodiment::DetritivoreEmbodiment::new(&GenesisSeed::from_phrase(
        "contract-detritivore",
    ));
    assert_moral_contract(bridge, "detritivore", 0.01);
}

#[test]
fn moral_contract_helios() {
    let bridge =
        symthaea::embodiment::HeliosEmbodiment::new(&GenesisSeed::from_phrase("contract-helios"));
    assert_moral_contract(bridge, "helios", 0.01);
}

/// PhoneBridge is the one that was actually broken. It executes real actions on a
/// connected Android device (taps, swipes, OpenUrl) and, until 2026-07-31, silently
/// discarded every moral veto via the trait's no-op default.
///
/// Runs WITHOUT a device attached: `step` computes `safety_level` from
/// Phi/override/moral before any ADB capture is required, so a failed capture only
/// clears `success`. The contract asserts on `safety_level`, so this is a real check
/// rather than one that vacuously passes when no phone is plugged in.
#[cfg(feature = "phone")]
#[test]
fn moral_contract_phone() {
    let bridge =
        symthaea_phone_embodiment::bridge::PhoneBridge::new("contract-test-serial", 1080, 2400);
    assert_moral_contract(bridge, "phone", 0.05);
}

// ═══════════════════════════════════════════════════════════════════════════
// SELF-TESTS: does this contract still catch the defects it was written for?
//
// Added 2026-07-31 after the assertions were relaxed from exact-tier equality to
// floors (see `assert_moral_contract`'s header). Relaxing a safety test to make it
// green is the classic way to destroy one, so the relaxation must be shown NOT to
// have cost the contract its detection power -- and shown permanently, not once by
// hand. These reproduce, in-process, the two real defects found on 2026-07-31:
//
//   * detritivore returned Green under an ahimsa violation (fail-OPEN).
//   * detritivore/helios kept the moral veto across reset() -- a fix's own bug,
//     caught by assertion 3.
//
// If a future edit weakens the contract, these turn red instead of the whole file
// silently going permissive.
// ═══════════════════════════════════════════════════════════════════════════

/// Bridge that never honours a moral gate. It does not override `apply_moral_gate`,
/// so it inherits the trait's no-op default — the exact shape of the PhoneBridge
/// defect, and behaviourally identical to detritivore/helios's explicitly-empty
/// overrides.
#[derive(Default)]
struct NoOpMoralGateBridge;

impl EmbodimentBridge for NoOpMoralGateBridge {
    fn step(
        &mut self,
        _hv: &ContinuousHV,
        _dt: f32,
        phi: f64,
    ) -> symthaea_core::embodiment::EmbodimentResult {
        symthaea_core::embodiment::EmbodimentResult {
            num_actuators: 1,
            control_effort: 0.0,
            success: true,
            prediction_error: 0.0,
            safety_level: MotorSafetyLevel::from_phi(phi),
            epistemic_grounding: 0,
            observation_confidence: 1.0,
        }
    }
    fn encode_perception(&mut self) -> ContinuousHV {
        ContinuousHV::zero(16384)
    }
    fn reset(&mut self) {}
    fn safety_level(&self) -> MotorSafetyLevel {
        MotorSafetyLevel::Green
    }
    fn set_safety_override(&mut self, _level: MotorSafetyLevel) {}
    fn clear_safety_override(&mut self) {}
    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform {
        symthaea_core::embodiment::EmbodimentPlatform::None
    }
    fn num_actuators(&self) -> usize {
        1
    }
    fn total_steps(&self) -> usize {
        0
    }
    fn telemetry(&self) -> symthaea_core::embodiment::EmbodimentTelemetry {
        Default::default()
    }
}

#[test]
#[should_panic(expected = "ahimsa_violated must force Red")]
fn contract_catches_a_noop_moral_gate() {
    assert_moral_contract(NoOpMoralGateBridge, "deliberately-noncompliant", 0.01);
}

/// Honours the gate but never clears it on `reset()` — the bug I introduced into
/// detritivore and helios while fixing them, which assertion 3 caught.
#[derive(Default)]
struct StickyMoralGateBridge {
    moral: Option<MotorSafetyLevel>,
}

impl EmbodimentBridge for StickyMoralGateBridge {
    fn step(
        &mut self,
        _hv: &ContinuousHV,
        _dt: f32,
        phi: f64,
    ) -> symthaea_core::embodiment::EmbodimentResult {
        let base = MotorSafetyLevel::from_phi(phi);
        symthaea_core::embodiment::EmbodimentResult {
            num_actuators: 1,
            control_effort: 0.0,
            success: true,
            prediction_error: 0.0,
            safety_level: match self.moral {
                Some(m) => base.max(m),
                None => base,
            },
            epistemic_grounding: 0,
            observation_confidence: 1.0,
        }
    }
    fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        self.moral = if gate.ahimsa_violated {
            Some(MotorSafetyLevel::Red)
        } else if gate.consent_violation {
            Some(MotorSafetyLevel::Orange)
        } else {
            None
        };
    }
    fn encode_perception(&mut self) -> ContinuousHV {
        ContinuousHV::zero(16384)
    }
    fn reset(&mut self) {
        // DELIBERATELY does not clear `moral` — that is the bug under test.
    }
    fn safety_level(&self) -> MotorSafetyLevel {
        MotorSafetyLevel::Green
    }
    fn set_safety_override(&mut self, _level: MotorSafetyLevel) {}
    fn clear_safety_override(&mut self) {}
    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform {
        symthaea_core::embodiment::EmbodimentPlatform::None
    }
    fn num_actuators(&self) -> usize {
        1
    }
    fn total_steps(&self) -> usize {
        0
    }
    fn telemetry(&self) -> symthaea_core::embodiment::EmbodimentTelemetry {
        Default::default()
    }
}

#[test]
#[should_panic(expected = "must clear the moral veto")]
fn contract_catches_a_moral_veto_that_survives_reset() {
    assert_moral_contract(
        StickyMoralGateBridge::default(),
        "deliberately-sticky",
        0.01,
    );
}
