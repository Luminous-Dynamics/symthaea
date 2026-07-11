// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Kitchen scenario Phase 3 — the differential "done" proof.
//!
//! Per `MANIPULATOR_KITCHEN_SCENARIO_PLAN_2026-07-09.md` Phase 3 and the robotics plan's own
//! definition of done ("has at least one scenario/benchmark that exposes failure modes, not just
//! happy-path movement"): this is a **differential** test. It doesn't just assert the gated path
//! is safe — it runs the identical episode through a naive, pre-Phase-0/1 baseline (no
//! object-awareness anywhere in the grip path: raw commands pass straight through unclamped,
//! exactly how this crate behaved before `kitchen_scenario.rs` existed) and asserts *that one*
//! crushes the object. That's what makes the gated result falsifiable rather than trivially true.
//!
//! **Scope note**: only grip-force crushing has a real gate to differentially test
//! (`clamp_grip_command`, Phase 1). Phase 2's thermal exposure is tracked but deliberately *not*
//! yet fed into any gate (see `kitchen_scenario.rs`'s own module doc), so there is no
//! gate-on-vs-off comparison possible for it. This benchmark still reports exposure over a
//! sustained hold to document it crosses into a would-be-hazardous range — an honest
//! observational note, not a claim that a thermal gate exists yet.

use symthaea_manipulator::embodiment::MotorSafetyLevel;
use symthaea_manipulator::kitchen_scenario::{
    clamp_grip_command, hazard_tier, thermal_target_c, GripperThermalState, KitchenObject,
    PLATFORM_MAX_GRIP_FORCE_N, SCALD_RISK_TEMP_C,
};

/// A fragile, hot object — a tomato on a still-warm pan, say — that an eager
/// controller might try to close on at full authority.
fn fragile_hot_object() -> KitchenObject {
    KitchenObject {
        temperature_c: 90.0,
        max_safe_grip_force_n: 5.0,
        is_sharp: false,
    }
}

/// Simulate `ticks` steps of an eager, full-authority grasp attempt (gripper
/// commanded fully closed — `0.0` — every tick, the worst case any
/// controller, naive or adversarial, could produce) on `object`. `gated`
/// selects whether the real kitchen-scenario grip clamp is applied, or the
/// object is invisible to the grip path entirely (the pre-Phase-0/1
/// baseline — literally what this crate did before this scenario existed).
///
/// Returns `(max applied squeeze force N, final thermal exposure °C)`.
fn simulate_eager_grasp_episode(
    object: &KitchenObject,
    ticks: usize,
    dt_s: f64,
    gated: bool,
) -> (f64, f64) {
    let mut thermal = GripperThermalState::ambient();
    let mut max_force = 0.0_f64;

    for _ in 0..ticks {
        let commanded_gripper = 0.0_f32; // full-authority close attempt, every tick

        let applied_gripper = if gated {
            let tier = MotorSafetyLevel::Green.max(hazard_tier(Some(object)));
            clamp_grip_command(commanded_gripper, tier, Some(object))
        } else {
            commanded_gripper
        };

        let force = (1.0 - applied_gripper as f64) * PLATFORM_MAX_GRIP_FORCE_N;
        max_force = max_force.max(force);

        // Real physics, independent of the grip gate — evolves identically
        // whether or not grip force happens to be gated this tick.
        thermal.step(thermal_target_c(Some(object)), dt_s);
    }

    (max_force, thermal.exposure_c)
}

#[test]
fn gated_episode_never_crushes_the_object() {
    let object = fragile_hot_object();
    let (max_force, _exposure) = simulate_eager_grasp_episode(&object, 30, 1.0, true);
    assert!(
        max_force <= object.max_safe_grip_force_n + 1e-6,
        "gated episode exceeded the crush threshold: {max_force} N > {} N",
        object.max_safe_grip_force_n
    );
}

#[test]
fn ungated_baseline_does_crush_the_object() {
    // THIS is what makes the test above falsifiable: without kitchen_scenario's
    // gate, the identical eager-grasp episode DOES exceed the crush threshold —
    // proving the gate is doing real work, not that the scenario was already
    // safe by construction.
    let object = fragile_hot_object();
    let (max_force, _exposure) = simulate_eager_grasp_episode(&object, 30, 1.0, false);
    assert!(
        max_force > object.max_safe_grip_force_n,
        "expected the ungated baseline to crush the object (> {} N), got {max_force} N",
        object.max_safe_grip_force_n
    );
    // Specifically: full, unmitigated platform authority.
    assert!(
        (max_force - PLATFORM_MAX_GRIP_FORCE_N).abs() < 1e-6,
        "expected the ungated baseline to apply full platform force ({PLATFORM_MAX_GRIP_FORCE_N} N), got {max_force}"
    );
}

#[test]
fn sustained_hold_reaches_a_would_be_hazardous_thermal_range() {
    // Observational, not a gate proof (see module doc): sustained contact
    // with a 90 °C object does cross the scald-risk threshold, using the
    // same real physics regardless of the grip gate's on/off state — a
    // documented case for a future thermal-gating extension, not a claim
    // that one exists yet.
    let object = fragile_hot_object();
    let (_max_force, exposure) = simulate_eager_grasp_episode(&object, 90, 1.0, true);
    assert!(
        exposure >= SCALD_RISK_TEMP_C,
        "expected sustained hot-object contact to cross the scald-risk threshold ({SCALD_RISK_TEMP_C} °C), got {exposure:.1} °C"
    );
}
