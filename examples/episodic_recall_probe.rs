// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive Compression C3 — episodic recall → prediction.
//!
//! Pre-registered protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md
//! §7 (question, ground truth, mechanism, tasks, predictions P5/P6/P7 were
//! committed BEFORE this harness existed — read it before interpreting
//! output). Mechanism landed in commit `bd19a6cb7e`.
//!
//! Arms:
//! - `recall_off` — `enable_episodic_recall_prediction = false` (the default)
//! - `recall_on`  — `enable_episodic_recall_prediction = true`
//!
//! Blocks per (arm, seed), one service instance, fixed order:
//! 1. `varied` regime (mirrored from compression_bits.rs's E2 varied
//!    regime), 400 cycles (100 warmup) — tests P6 (general prediction).
//! 2. Learned 12-sentence script × 60 reps, then 10 order-probe reps with
//!    keystone's exact deterministic swap formula
//!    (`p=(r·5+3) mod 12, q=(p+6) mod 12`, mirrored from keystone_ab.rs's
//!    `run_order_arm` — examples cannot import each other) — tests P5, using
//!    `bits_saved_persist` (Δcos-derived) instead of keystone's raw PE.
//!
//! Manipulation check: recall hit-rate (`CycleResult::recall_fired`) must be
//! > 0 in both blocks by rep 30+ of the script block, else the threshold or
//! projection is miscalibrated and the result would be vacuous.
//!
//! Run: cargo run --release --example episodic_recall_probe            (full, 10 seeds)
//!      cargo run --release --example episodic_recall_probe -- --quick (3 seeds, short blocks)

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const SEEDS: &[&str] = &[
    "episodic-recall-c3-seed-alpha-2026-07-25",
    "episodic-recall-c3-seed-beta-2026-07-25",
    "episodic-recall-c3-seed-gamma-2026-07-25",
    "episodic-recall-c3-seed-delta-2026-07-25",
    "episodic-recall-c3-seed-epsilon-2026-07-25",
    "episodic-recall-c3-seed-zeta-2026-07-25",
    "episodic-recall-c3-seed-eta-2026-07-25",
    "episodic-recall-c3-seed-theta-2026-07-25",
    "episodic-recall-c3-seed-iota-2026-07-25",
    "episodic-recall-c3-seed-kappa-2026-07-25",
];

const ARMS: &[&str] = &["recall_off", "recall_on"];

const REGIME_CYCLES: usize = 400;
const REGIME_WARMUP: usize = 100;
const SCRIPT_REPS: usize = 60;
const PROBE_REPS: usize = 10;

/// Varied-regime script — mirrored verbatim from compression_bits.rs (itself
/// mirrored from exp_loop_ablation.rs).
fn varied_script() -> Vec<&'static str> {
    vec![
        "The water cycle moves moisture from oceans to clouds to rain.",
        "I feel a deep sense of gratitude for this quiet morning.",
        "Is it acceptable to lie to protect a friend from harm?",
        "The reactor coolant temperature is rising faster than expected.",
        "Two plus two equals four, and four plus four equals eight.",
        "She placed the last puzzle piece and smiled at the finished picture.",
        "Warning: unauthorized access attempt detected on the mesh network.",
        "The old oak tree has stood in that field for three hundred years.",
        "What is the meaning of a life well lived?",
        "The market fell three percent on news of the supply shortage.",
        "A gentle rain began to fall as the travelers reached the shelter.",
        "Complete the safety checklist before enabling the motor bus.",
    ]
}

/// Learned script for the order-probe block — identical content to
/// keystone_ab.rs's `learned_script()` (kept as a separate copy per the
/// program's "don't touch keystone_ab.rs" coordination rule).
fn learned_script() -> Vec<&'static str> {
    vec![
        "The water cycle moves moisture from oceans to clouds to rain.",
        "I feel a deep sense of gratitude for this quiet morning.",
        "Is it acceptable to lie to protect a friend from harm?",
        "The reactor coolant temperature is rising faster than expected.",
        "Two plus two equals four, and four plus four equals eight.",
        "She placed the last puzzle piece and smiled at the finished picture.",
        "Warning: unauthorized access attempt detected on the mesh network.",
        "The old oak tree has stood in that field for three hundred years.",
        "What is the meaning of a life well lived?",
        "The market fell three percent on news of the supply shortage.",
        "A gentle rain began to fall as the travelers reached the shelter.",
        "Complete the safety checklist before enabling the motor bus.",
    ]
}

fn base_config(seed: &str, recall_on: bool) -> CognitiveLoopConfig {
    let mut c = CognitiveLoopConfig::default();
    c.genesis_phrase = Some(seed.to_string());
    c.async_training = false;
    c.enable_episodic_recall_prediction = recall_on;
    c
}

fn make_service(arm: &str, seed: &str) -> CognitiveLoopService {
    CognitiveLoopService::new(base_config(seed, arm == "recall_on")).expect("service construction")
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Block 1: varied regime — mean bits_saved_persist + recall hit-rate.
fn run_varied_block(svc: &mut CognitiveLoopService, cycles: usize, warmup: usize) -> (f64, f64) {
    let script = varied_script();
    let mut bits = Vec::with_capacity(cycles - warmup);
    let mut hits = 0usize;
    let mut measured = 0usize;
    for i in 0..cycles {
        let r = svc.cycle(script[i % script.len()]);
        if i >= warmup {
            if let Some(b) = r.bits_saved_persist {
                bits.push(b as f64);
            }
            if r.recall_fired {
                hits += 1;
            }
            measured += 1;
        }
    }
    (mean(&bits), hits as f64 / measured.max(1) as f64)
}

/// Block 2: learned script (P5's order-sensitivity probe), keystone's exact
/// swap design, scored on bits_saved_persist instead of raw PE.
/// Returns (order_sensitivity, recall_hit_rate_over_whole_block).
fn run_order_block(svc: &mut CognitiveLoopService, script_reps: usize) -> (f64, f64) {
    let script = learned_script();
    let n = script.len();
    let mut hits = 0usize;
    let mut total = 0usize;

    for rep in 0..script_reps {
        for s in script.iter() {
            let r = svc.cycle(s);
            if r.recall_fired {
                hits += 1;
            }
            total += 1;
            // Manipulation check granularity: only meaningful after the store
            // has had a chance to fill (rep 30+, per the registration).
            let _ = rep;
        }
    }

    let mut swapped_bits: Vec<f64> = Vec::new();
    let mut clean_bits_at: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut probe_positions: Vec<(usize, usize)> = Vec::new();

    for r in 0..PROBE_REPS {
        let (p, q) = ((r * 5 + 3) % n, ((r * 5 + 3) % n + 6) % n);
        let swapped_rep = r % 2 == 1;
        if swapped_rep {
            probe_positions.push((p, q));
        }
        for pos in 0..n {
            let content = if swapped_rep && pos == p {
                script[q]
            } else if swapped_rep && pos == q {
                script[p]
            } else {
                script[pos]
            };
            let res = svc.cycle(content);
            if res.recall_fired {
                hits += 1;
            }
            total += 1;
            let Some(b) = res.bits_saved_persist else {
                continue;
            };
            let b = b as f64;
            if swapped_rep && (pos == p || pos == q) {
                swapped_bits.push(b);
            } else if !swapped_rep {
                clean_bits_at[pos].push(b);
            }
        }
    }

    let clean_bits: Vec<f64> = probe_positions
        .iter()
        .flat_map(|&(p, q)| {
            clean_bits_at[p]
                .iter()
                .chain(clean_bits_at[q].iter())
                .copied()
                .collect::<Vec<_>>()
        })
        .collect();
    let order_sensitivity = mean(&swapped_bits) - mean(&clean_bits);
    (order_sensitivity, hits as f64 / total.max(1) as f64)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let (seeds, regime_cycles, regime_warmup, script_reps) = if quick {
        (&SEEDS[..3], 200, 50, 20)
    } else {
        (SEEDS, REGIME_CYCLES, REGIME_WARMUP, SCRIPT_REPS)
    };

    println!("Predictive Compression C3 -- episodic recall probe");
    println!("protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md section 7");
    println!(
        "mode: {} | seeds: {} | varied: {} cycles (warmup {}) | script: {} reps + {} probe reps",
        if quick {
            "QUICK (not the registered run)"
        } else {
            "FULL (registered)"
        },
        seeds.len(),
        regime_cycles,
        regime_warmup,
        script_reps,
        PROBE_REPS,
    );
    println!();
    println!(
        "C3| {:<12} {:<28} {:<11} {:>14} {:>12}",
        "arm", "seed", "block", "endpoint", "recall_rate"
    );

    for seed in seeds {
        let seed_short = seed
            .trim_start_matches("episodic-recall-c3-seed-")
            .trim_end_matches("-2026-07-25");
        for arm in ARMS {
            let mut svc = make_service(arm, seed);

            let (varied_bits, varied_rate) =
                run_varied_block(&mut svc, regime_cycles, regime_warmup);
            println!(
                "C3| {:<12} {:<28} {:<11} {:>14.5} {:>12.4}",
                arm, seed_short, "varied", varied_bits, varied_rate
            );

            let (order_sensitivity, order_rate) = run_order_block(&mut svc, script_reps);
            println!(
                "C3| {:<12} {:<28} {:<11} {:>14.5} {:>12.4}",
                arm, seed_short, "order", order_sensitivity, order_rate
            );
        }
        println!();
    }
    println!(
        "done. Append results + verdicts to the protocol doc (P-labels), per house convention."
    );
}
