// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Live integration test for observer-ΔΨ (feature `art-observer`, visual-art
//! plan "Option 2"): proves the whole viewing pipeline actually fires inside
//! a running `CognitiveLoopService` — artwork generated → rasterized →
//! injected as vision frames → ψ accumulated → verdict recorded — rather
//! than merely compiling. Companion to `creative_mesh_sharing.rs`, which
//! plays the same role for the culture-sharing wiring.

#![cfg(feature = "art-observer")]

use super::super::{CognitiveLoopConfig, CognitiveLoopService};

const INPUTS: [&str; 5] = [
    "consciousness emerges from integration",
    "the garden grows in silence",
    "hello world, what do you see",
    "a memory of light and water",
    "we are learning to trust each other",
];

/// Drive a live service until at least `want` observer verdicts land, or
/// `max_cycles` elapse. Returns the verdict count reached.
fn run_until_verdicts(service: &mut CognitiveLoopService, want: u64, max_cycles: usize) -> u64 {
    let mut verdicts = 0;
    for i in 0..max_cycles {
        let _ = service.cycle(INPUTS[i % INPUTS.len()]);
        verdicts = service
            .sensorimotor
            .motor_rendering
            .creative_manager
            .as_ref()
            .expect("creative_manager present under `creative`")
            .last_telemetry()
            .observer_verdicts;
        if verdicts >= want {
            break;
        }
    }
    verdicts
}

#[test]
fn observer_verdict_fires_in_live_loop() {
    let mut config = CognitiveLoopConfig::default();
    // Ultra auto-dilation OOMs camera-less loops (~17-22 GB at 65,536 dims);
    // see enable_vision_auto_dilation docs.
    config.enable_vision_auto_dilation = false;
    assert!(
        config.enable_vision_manifold,
        "test precondition: vision manifold must be on by default under \
         the vision-manifold feature (art-observer implies it)"
    );
    let mut service = CognitiveLoopService::new(config).unwrap();

    let verdicts = run_until_verdicts(&mut service, 1, 400);
    assert!(
        verdicts >= 1,
        "no observer verdict across 400 cycles — the viewing window never \
         opened or never completed (artwork generation happens well within \
         this budget per creative_mesh_sharing.rs)"
    );

    let telemetry = service
        .sensorimotor
        .motor_rendering
        .creative_manager
        .as_ref()
        .unwrap()
        .last_telemetry()
        .clone();
    assert!(
        telemetry.observer_delta_psi.is_finite(),
        "Δψ must be a finite measurement, got {}",
        telemetry.observer_delta_psi
    );
    assert!(
        !telemetry.observer_was_control,
        "A/B mode is off by default — the first verdict must be the real \
         artwork, not a control frame"
    );
}

#[test]
fn ab_mode_alternates_real_and_control() {
    let mut config = CognitiveLoopConfig::default();
    config.art_observer_ab_mode = true;
    config.enable_vision_auto_dilation = false;
    let mut service = CognitiveLoopService::new(config).unwrap();

    // First verdict: real artwork (window counter parity 0).
    let verdicts = run_until_verdicts(&mut service, 1, 400);
    assert!(verdicts >= 1, "no first verdict across 400 cycles");
    let first_was_control = service
        .sensorimotor
        .motor_rendering
        .creative_manager
        .as_ref()
        .unwrap()
        .last_telemetry()
        .observer_was_control;
    assert!(!first_was_control, "first A/B window must be the real arm");

    // Second verdict: scrambled control (parity 1).
    let verdicts = run_until_verdicts(&mut service, 2, 800);
    assert!(verdicts >= 2, "no second verdict across the cycle budget");
    let second_was_control = service
        .sensorimotor
        .motor_rendering
        .creative_manager
        .as_ref()
        .unwrap()
        .last_telemetry()
        .observer_was_control;
    assert!(
        second_was_control,
        "second A/B window must be the scrambled control arm"
    );
}

/// The actual A/B experiment probe, run inside the optimized test binary
/// (the `art_observer_ab` example in the dev profile is ~100× slower —
/// 16,384-dim HDC cycles are brutal unoptimized, found 2026-07-11 when
/// three example runs crawled at ~10s/cycle and got reaped).
///
/// `#[ignore]`d: it asserts almost nothing — it MEASURES. Run with:
/// ```bash
/// cargo test -p symthaea --lib --no-default-features \
///   --features art-observer,gallery,reasoning_engine \
///   ab_experiment_probe -- --ignored --nocapture --test-threads=1
/// ```
/// Prints per-arm Δψ values and means. One process = one observation;
/// repeat separated runs before believing a direction (the loop is
/// time-seeded, not deterministic across processes).
#[test]
#[ignore = "research probe, not a regression test — run with --ignored --nocapture"]
fn ab_experiment_probe() {
    const VERDICTS_PER_ARM: usize = 6;
    const MAX_CYCLES: usize = 3_000;

    let mut config = CognitiveLoopConfig::default();
    config.art_observer_ab_mode = true;
    // Counterbalancing across battery runs: AB_CONTROL_FIRST=1 makes the
    // scrambled control open the session (breaks the arm↔order confound
    // when combined with art-first runs).
    config.art_observer_ab_control_first = std::env::var("AB_CONTROL_FIRST")
        .map(|v| v == "1")
        .unwrap_or(false);
    // Default dilation config. (A prior version of this comment described
    // working around "the vision-manifold OOM defect" — that balloon was
    // root-caused 2026-07-16 to creative_bridge's unbounded motif-replay
    // drain, fixed in `6bc375d033`; full-length sessions run fine now and
    // the old many-short-processes battery workaround is obsolete.)
    let mut service = CognitiveLoopService::new(config).unwrap();

    let mut art: Vec<f32> = Vec::new();
    let mut control: Vec<f32> = Vec::new();
    let mut seen = 0u64;

    // Own-process RSS in MB from /proc/self/status — the OOM hunt needs
    // cycle-number ↔ memory correlation (an external sampler sees wall
    // time only, and cycle duration varies 100× as subsystems activate).
    fn rss_mb() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines().find(|l| l.starts_with("VmRSS")).and_then(|l| {
                    l.split_whitespace()
                        .nth(1)
                        .and_then(|kb| kb.parse::<u64>().ok())
                })
            })
            .unwrap_or(0)
            / 1024
    }

    // OOM-hunt instrumentation (opt-in via AB_TRACE=1 — slows cycles ~5×
    // and floods the log): debug-level tracing to stdout so the LAST log
    // line before the balloon names the active subsystem (gdb crashes on
    // this binary's DWARF, so logs are the stack trace we get).
    if std::env::var("AB_TRACE").map(|v| v == "1").unwrap_or(false) {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();
    }

    println!("observer-ΔΨ A/B probe: real artwork vs pixel-scrambled control");
    for i in 0..MAX_CYCLES {
        let _ = service.cycle(INPUTS[i % INPUTS.len()]);
        {
            use std::io::Write as _;
            println!("  [progress] cycle {i:4}  rss_mb={}", rss_mb());
            let _ = std::io::stdout().flush();
        }
        let t = service
            .creative_telemetry()
            .expect("creative manager present");
        if t.observer_verdicts > seen {
            seen = t.observer_verdicts;
            let arm = if t.observer_was_control {
                control.push(t.observer_delta_psi);
                "control"
            } else {
                art.push(t.observer_delta_psi);
                "art    "
            };
            println!(
                "  verdict {seen:2} [{arm}] dpsi = {:+.5}  (surprise {:.4}, cycle {i})",
                t.observer_delta_psi, t.observer_viewing_surprise
            );
        }
        if art.len() >= VERDICTS_PER_ARM && control.len() >= VERDICTS_PER_ARM {
            break;
        }
    }

    let mean = |v: &[f32]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };
    println!("\nart:     N={} mean dpsi {:+.5}", art.len(), mean(&art));
    println!(
        "control: N={} mean dpsi {:+.5}",
        control.len(),
        mean(&control)
    );
    println!(
        "dpsi(art) - dpsi(control) = {:+.5}   (one observation; repeat runs)",
        mean(&art) - mean(&control)
    );

    // Only sanity assertions — the probe measures, it doesn't judge.
    assert!(
        !art.is_empty() && !control.is_empty(),
        "both arms must collect at least one verdict within {MAX_CYCLES} cycles \
         (art {}, control {})",
        art.len(),
        control.len()
    );
    for v in art.iter().chain(control.iter()) {
        assert!(v.is_finite(), "Δψ must be finite");
    }
}
