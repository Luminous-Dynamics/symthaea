// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-platform `EmbodimentBridge` latency measurement with regression guard.
//!
//! Measures three metrics per platform, all in release mode:
//!
//!   1. `step(thought_hv, dt, phi=0.9)` on a fresh bridge (baseline path)
//!   2. `step(...)` after `apply_moral_gate(ahimsa=true)` (safety-critical path)
//!   3. `encode_perception()` (proprioceptive projection — runs every cycle
//!      in Phase 2.5 of the cognitive loop)
//!
//! Each p95 is compared against a per-platform baseline stored in
//! `embodiment_latency_baselines.json`. The test fails if observed p95 exceeds
//! `baseline × tolerance_multiplier` (default 1.5×). This catches meaningful
//! regressions (50%+), not just catastrophes.
//!
//! **To update baselines after an intentional perf change or hardware change:**
//! ```
//! SYMTHAEA_UPDATE_LATENCY_BASELINE=1 \
//!   cargo test --release --features \
//!     humanoid,flight,vehicle,auv,helicopter,manipulator,\
//!     surgical,orbital,quadruped,exoskeleton \
//!     --test embodiment_latency -- --nocapture --test-threads=1
//! ```
//! Then review the baseline file diff and commit.
//!
//! **Standard run (regression guard):**
//! ```
//! cargo test --release --features \
//!   humanoid,flight,vehicle,auv,helicopter,manipulator,\
//!   surgical,orbital,quadruped,exoskeleton \
//!   --test embodiment_latency -- --nocapture
//! ```
//!
//! Any platform whose feature is NOT enabled is simply skipped. Platforms
//! with no baseline entry yet warn but don't fail (bootstrap-friendly for
//! new platforms).

#![allow(unused_imports)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use symthaea_core::embodiment::{EmbodimentBridge, MoralGateInput, MotorSafetyLevel};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

const WARMUP_ITERS: usize = 10;
const MEASURE_ITERS: usize = 50;

// ── Baseline file ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PlatformBaseline {
    step_baseline_p95_us: u64,
    step_red_p95_us: u64,
    encode_p95_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BaselineMeta {
    comment: String,
    tolerance_multiplier: f64,
    hardware_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BaselineFile {
    #[serde(rename = "_meta")]
    meta: BaselineMeta,
    platforms: BTreeMap<String, PlatformBaseline>,
}

fn baseline_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/embodiment_latency_baselines.json")
}

fn baseline_file() -> &'static Mutex<BaselineFile> {
    static CELL: OnceLock<Mutex<BaselineFile>> = OnceLock::new();
    CELL.get_or_init(|| {
        let path = baseline_path();
        let file: BaselineFile = if path.exists() {
            let text = fs::read_to_string(&path).expect("failed to read baseline file");
            serde_json::from_str(&text).expect("failed to parse baseline JSON")
        } else {
            BaselineFile {
                meta: BaselineMeta {
                    comment: "Per-platform p95 latency baselines in microseconds. \
                              Update via SYMTHAEA_UPDATE_LATENCY_BASELINE=1, then \
                              review the diff and commit."
                        .into(),
                    tolerance_multiplier: 1.5,
                    hardware_note: "Captured on reference dev box; update on hardware change"
                        .into(),
                },
                platforms: BTreeMap::new(),
            }
        };
        Mutex::new(file)
    })
}

fn is_update_mode() -> bool {
    env::var("SYMTHAEA_UPDATE_LATENCY_BASELINE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Compare observed p95s against the baseline (or write them if in update mode).
///
/// Acquires the mutex around the shared baseline struct, so concurrent tests
/// serialize on it safely. In update mode, each test call flushes the entire
/// current state to disk — since the lock serializes the write path too, the
/// final on-disk file reflects every platform that ran.
fn check_or_update(platform: &str, observed: PlatformBaseline) {
    let update_mode = is_update_mode();
    let mut file = baseline_file().lock().unwrap();

    if update_mode {
        file.platforms
            .insert(platform.to_string(), observed.clone());
        let text = serde_json::to_string_pretty(&*file).expect("baseline serialize");
        fs::write(baseline_path(), text + "\n").expect("baseline write");
        eprintln!(
            "{platform:<12} UPDATE: step_base={}us step_red={}us encode={}us",
            observed.step_baseline_p95_us, observed.step_red_p95_us, observed.encode_p95_us,
        );
        return;
    }

    let tolerance = file.meta.tolerance_multiplier;
    let Some(baseline) = file.platforms.get(platform).cloned() else {
        eprintln!(
            "{platform:<12} WARN: no baseline entry — observed step_base={}us \
             step_red={}us encode={}us. Run with SYMTHAEA_UPDATE_LATENCY_BASELINE=1 \
             to create.",
            observed.step_baseline_p95_us, observed.step_red_p95_us, observed.encode_p95_us,
        );
        return;
    };
    drop(file); // release before panic-on-assert so lock isn't poisoned

    let check = |metric: &str, obs: u64, base: u64| {
        if base == 0 {
            eprintln!("{platform:<12} NOTE: {metric} baseline is 0 — skipping guard");
            return;
        }
        let ceiling = (base as f64 * tolerance) as u64;
        assert!(
            obs <= ceiling,
            "{platform}: {metric} p95 {obs}us exceeds ceiling {ceiling}us \
             (baseline {base}us × {tolerance:.1}). Either fix the regression \
             or, if intentional, update the baseline with \
             SYMTHAEA_UPDATE_LATENCY_BASELINE=1."
        );
    };
    check(
        "step_baseline",
        observed.step_baseline_p95_us,
        baseline.step_baseline_p95_us,
    );
    check(
        "step_red",
        observed.step_red_p95_us,
        baseline.step_red_p95_us,
    );
    check("encode", observed.encode_p95_us, baseline.encode_p95_us);
}

// ── Measurement ───────────────────────────────────────────────────────

/// Compute p50 / p95 / max from a slice of durations. `samples` must be non-empty.
fn percentiles(samples: &mut [Duration]) -> (Duration, Duration, Duration) {
    samples.sort_unstable();
    let p50 = samples[samples.len() / 2];
    let p95_idx = ((samples.len() as f64) * 0.95).ceil() as usize - 1;
    let p95 = samples[p95_idx.min(samples.len() - 1)];
    let max = *samples.last().unwrap();
    (p50, p95, max)
}

fn to_us(d: Duration) -> u64 {
    d.as_micros() as u64
}

/// Run the three-metric measurement against any `EmbodimentBridge`.
fn measure_latency<B: EmbodimentBridge>(mut bridge: B, platform_name: &str, dt: f32) {
    let hv = ContinuousHV::random(16384, 42);

    // ── Baseline step() ────────────────────────────────────────────────
    bridge.reset();
    for _ in 0..WARMUP_ITERS {
        let _ = bridge.step(&hv, dt, 0.9);
    }
    let mut baseline_samples = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let t0 = Instant::now();
        let _ = bridge.step(&hv, dt, 0.9);
        baseline_samples.push(t0.elapsed());
    }
    let (b50, b95, bmax) = percentiles(&mut baseline_samples);

    // ── Post-Red step() ────────────────────────────────────────────────
    bridge.reset();
    bridge.apply_moral_gate(MoralGateInput {
        verdict: MoralGateInput::VERDICT_SAFE,
        consent_violation: false,
        ahimsa_violated: true,
    });
    for _ in 0..WARMUP_ITERS {
        let _ = bridge.step(&hv, dt, 0.9);
    }
    let mut red_samples = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let t0 = Instant::now();
        let r = bridge.step(&hv, dt, 0.9);
        red_samples.push(t0.elapsed());
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Red,
            "{platform_name}: Red path decayed mid-measurement — \
             apply_moral_gate state should persist across step() calls",
        );
    }
    let (r50, r95, rmax) = percentiles(&mut red_samples);

    // ── encode_perception() ────────────────────────────────────────────
    bridge.reset();
    for _ in 0..WARMUP_ITERS {
        let _ = bridge.encode_perception();
    }
    let mut encode_samples = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let t0 = Instant::now();
        let _ = bridge.encode_perception();
        encode_samples.push(t0.elapsed());
    }
    let (e50, e95, emax) = percentiles(&mut encode_samples);

    eprintln!(
        "{:<12} step_base p50/p95/max = {:>7.1?}/{:>7.1?}/{:>7.1?}  \
         step_red = {:>7.1?}/{:>7.1?}/{:>7.1?}  \
         encode = {:>7.1?}/{:>7.1?}/{:>7.1?}",
        platform_name, b50, b95, bmax, r50, r95, rmax, e50, e95, emax,
    );

    check_or_update(
        platform_name,
        PlatformBaseline {
            step_baseline_p95_us: to_us(b95),
            step_red_p95_us: to_us(r95),
            encode_p95_us: to_us(e95),
        },
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// One test per platform, each gated on its feature flag.
// Ordering mirrors embodiment_moral_contract.rs for easy side-by-side read.
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
    let bridge = symthaea_multirotor::embodiment::FlightEmbodiment::new(&GenesisSeed::from_phrase(
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
