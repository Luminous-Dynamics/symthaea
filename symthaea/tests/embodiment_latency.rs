// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-platform `EmbodimentBridge::step()` latency measurement.
//!
//! We claim 10/10 robotics platforms are moral-gate wired. That claim is
//! worthless without knowing how long each step takes — a 500ms gate is a
//! bug regardless of correctness. This file measures:
//!
//!   1. `step(thought_hv, dt, phi=0.9)` on a fresh bridge — the common path
//!   2. `step(thought_hv, dt, phi=0.9)` after `apply_moral_gate(ahimsa=true)`
//!      — the safety-critical path
//!
//! Output is p50 / p95 / max per platform, printed to stderr (use
//! `-- --nocapture` to see). A loose 100 ms p95 guard catches catastrophic
//! regressions; the real value is the recorded numbers.
//!
//! **Run in release mode for numbers that mean anything:**
//! ```
//! cargo test --release --features \
//!   humanoid,flight,vehicle,auv,helicopter,manipulator,surgical,orbital,quadruped,exoskeleton \
//!   --test embodiment_latency -- --nocapture
//! ```
//!
//! Any platform whose feature is NOT enabled is simply skipped.

#![allow(unused_imports)]

use std::time::{Duration, Instant};

use symthaea_core::embodiment::{EmbodimentBridge, MoralGateInput, MotorSafetyLevel};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

const WARMUP_ITERS: usize = 10;
const MEASURE_ITERS: usize = 50;
/// Loose catastrophic-regression guard — real step() should be well under this.
const P95_BUDGET: Duration = Duration::from_millis(100);

/// Compute p50 / p95 / max from a slice of durations. `samples` must be non-empty.
fn percentiles(samples: &mut [Duration]) -> (Duration, Duration, Duration) {
    samples.sort_unstable();
    let p50 = samples[samples.len() / 2];
    let p95_idx = ((samples.len() as f64) * 0.95).ceil() as usize - 1;
    let p95 = samples[p95_idx.min(samples.len() - 1)];
    let max = *samples.last().unwrap();
    (p50, p95, max)
}

/// Run the measurement against any `EmbodimentBridge`.
///
/// Prints a one-line report for the baseline path and the post-Red path,
/// then asserts the post-Red p95 stays under `P95_BUDGET` (the safety path
/// is what we most care about — it must not be slower than baseline by so
/// much that a Red trigger itself becomes a hazard).
fn measure_latency<B: EmbodimentBridge>(mut bridge: B, platform_name: &str, dt: f32) {
    let hv = ContinuousHV::random(16384, 42);

    // ── Baseline: plain step() on a fresh-reset bridge ──────────────────
    bridge.reset();
    for _ in 0..WARMUP_ITERS {
        let _ = bridge.step(&hv, dt, 0.9);
    }
    let mut baseline = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let t0 = Instant::now();
        let _ = bridge.step(&hv, dt, 0.9);
        baseline.push(t0.elapsed());
    }
    let (b50, b95, bmax) = percentiles(&mut baseline);

    // ── Post-Red: apply Red moral gate once, then measure step() cost ───
    //
    // Note: `apply_moral_gate` only toggles a field; the cost we're measuring
    // is whatever extra work the Red-tier branch inside `step()` does (motor
    // clamping, emergency-stop kinematics, etc.).
    bridge.reset();
    bridge.apply_moral_gate(MoralGateInput {
        verdict: MoralGateInput::VERDICT_SAFE,
        consent_violation: false,
        ahimsa_violated: true,
    });
    for _ in 0..WARMUP_ITERS {
        let _ = bridge.step(&hv, dt, 0.9);
    }
    let mut red_path = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let t0 = Instant::now();
        let r = bridge.step(&hv, dt, 0.9);
        red_path.push(t0.elapsed());
        // Correctness sanity: Red must stick every cycle on the ahimsa path.
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Red,
            "{platform_name}: Red path decayed to {:?} mid-measurement — \
             apply_moral_gate state should persist across step() calls",
            r.safety_level,
        );
    }
    let (r50, r95, rmax) = percentiles(&mut red_path);

    eprintln!(
        "{:<12} baseline p50={:>7.1?} p95={:>7.1?} max={:>7.1?}  |  \
         red p50={:>7.1?} p95={:>7.1?} max={:>7.1?}",
        platform_name, b50, b95, bmax, r50, r95, rmax,
    );

    assert!(
        r95 < P95_BUDGET,
        "{platform_name}: Red-path step() p95 {:?} exceeds {:?} budget — \
         catastrophic regression. If this fires, a Red-tier branch inside \
         step() has quadratic behavior or a per-call allocation.",
        r95,
        P95_BUDGET,
    );
    assert!(
        b95 < P95_BUDGET,
        "{platform_name}: baseline step() p95 {:?} exceeds {:?} budget.",
        b95,
        P95_BUDGET,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// One test per platform, each gated on its feature flag.
// Ordering mirrors `embodiment_moral_contract.rs` for easy side-by-side read.
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "humanoid")]
#[test]
fn latency_humanoid() {
    let bridge = symthaea::cognitive_loop::motor_bridge::MotorBridge::new(
        &GenesisSeed::from_phrase("latency-humanoid"),
    );
    measure_latency(bridge, "humanoid", 0.025);
}

#[cfg(feature = "flight")]
#[test]
fn latency_flight() {
    let bridge = symthaea_flight::embodiment::FlightEmbodiment::new(&GenesisSeed::from_phrase(
        "latency-flight",
    ));
    measure_latency(bridge, "flight", 0.002);
}

#[cfg(feature = "vehicle")]
#[test]
fn latency_vehicle() {
    let bridge = symthaea_vehicle::embodiment::VehicleEmbodiment::new(&GenesisSeed::from_phrase(
        "latency-vehicle",
    ));
    measure_latency(bridge, "vehicle", 0.005);
}

#[cfg(feature = "auv")]
#[test]
fn latency_auv() {
    let bridge =
        symthaea_auv::embodiment::AuvEmbodiment::new(&GenesisSeed::from_phrase("latency-auv"));
    measure_latency(bridge, "auv", 0.01);
}

#[cfg(feature = "helicopter")]
#[test]
fn latency_helicopter() {
    let bridge = symthaea_helicopter::embodiment::HelicopterEmbodiment::new(
        &GenesisSeed::from_phrase("latency-helicopter"),
    );
    measure_latency(bridge, "helicopter", 1.0 / 300.0);
}

#[cfg(feature = "manipulator")]
#[test]
fn latency_manipulator() {
    let bridge = symthaea_manipulator::embodiment::ManipulatorEmbodiment::new(
        &GenesisSeed::from_phrase("latency-manipulator"),
    );
    measure_latency(bridge, "manipulator", 0.001);
}

#[cfg(feature = "surgical")]
#[test]
fn latency_surgical() {
    let bridge = symthaea_surgical::embodiment::SurgicalEmbodiment::new(&GenesisSeed::from_phrase(
        "latency-surgical",
    ));
    measure_latency(bridge, "surgical", 0.001);
}

#[cfg(feature = "orbital")]
#[test]
fn latency_orbital() {
    let bridge = symthaea_orbital::embodiment::OrbitalEmbodiment::new(&GenesisSeed::from_phrase(
        "latency-orbital",
    ));
    measure_latency(bridge, "orbital", 0.01);
}

#[cfg(feature = "quadruped")]
#[test]
fn latency_quadruped() {
    let bridge = symthaea_quadruped::embodiment::QuadrupedEmbodiment::new(
        &GenesisSeed::from_phrase("latency-quadruped"),
    );
    measure_latency(bridge, "quadruped", 0.005);
}

#[cfg(feature = "exoskeleton")]
#[test]
fn latency_exoskeleton() {
    let bridge = symthaea_exoskeleton::embodiment::ExoskeletonEmbodiment::new(
        &GenesisSeed::from_phrase("latency-exoskeleton"),
    );
    measure_latency(bridge, "exoskeleton", 0.001);
}
