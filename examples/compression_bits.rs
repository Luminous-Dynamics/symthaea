// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive Compression C1 — does the liquid state compress?
//!
//! Pre-registered protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md
//! (§5 + Amendment 1 — arms, metric, seeds, and falsifiable predictions were
//! committed BEFORE this harness existed; read it before interpreting output).
//!
//! Arms:
//! - `live`       — default config (HdcLtcUnified backend), async_training off
//! - `frozen`     — identical + `freeze_cfc_training()` (weights never update;
//!                  state still evolves)
//! - `memoryless` — identical + `reset_temporal_state()` before every cycle
//!                  (no state carryover; inject is a documented reset on the
//!                  HdcLtc backend)
//!
//! Blocks per (arm, seed), one service instance, fixed order:
//! four E2 regimes (repetitive/varied/alarming/empty — mirrored from
//! examples/exp_loop_ablation.rs §E2) × 500 cycles each (WARMUP 100 /
//! MEASURE 400), then the keystone 12-sentence learned script × 60 reps
//! (mirrored from examples/keystone_ab.rs; examples cannot import each other).
//!
//! Primary endpoint: mean CycleResult::bits_saved_persist over MEASURE cycles
//! per regime. Secondary: learning growth on the learned-script block
//! (late reps 55–60 minus early reps 2–4 — positive = saving grows with
//! exposure). Manipulation check: bits_saved_zero > 0 in non-empty regimes.
//! Cost is deliberately NOT measured (pre-registered exclusion; ambient load).
//!
//! Rows print as they complete so partial output survives an external kill.
//!
//! Run: cargo run --release --example compression_bits            (full, 10 seeds)
//!      cargo run --release --example compression_bits -- --quick (3 seeds, short blocks)

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const SEEDS: &[&str] = &[
    "compression-c1-seed-alpha-2026-07-17",
    "compression-c1-seed-beta-2026-07-17",
    "compression-c1-seed-gamma-2026-07-17",
    "compression-c1-seed-delta-2026-07-17",
    "compression-c1-seed-epsilon-2026-07-17",
    "compression-c1-seed-zeta-2026-07-17",
    "compression-c1-seed-eta-2026-07-17",
    "compression-c1-seed-theta-2026-07-17",
    "compression-c1-seed-iota-2026-07-17",
    "compression-c1-seed-kappa-2026-07-17",
];

const ARMS: &[&str] = &["live", "frozen", "memoryless"];

const REGIME_CYCLES: usize = 500;
const REGIME_WARMUP: usize = 100;
const SCRIPT_REPS: usize = 60;

/// Varied-regime script — mirrored verbatim from exp_loop_ablation.rs.
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

fn regime_input(regime: &str, i: usize) -> &'static str {
    match regime {
        "repetitive" => "the system hums quietly in the background",
        "varied" => varied_script()[i % 12],
        "alarming" => [
            "URGENT: fire detected in the server room, evacuate immediately!",
            "Critical failure: coolant pressure dropping, meltdown risk rising!",
            "Intruder alert: perimeter breach at the north gate right now!",
        ][i % 3],
        "empty" => "",
        _ => unreachable!(),
    }
}

fn base_config(seed: &str) -> CognitiveLoopConfig {
    let mut c = CognitiveLoopConfig::default();
    c.genesis_phrase = Some(seed.to_string());
    c.async_training = false;
    c
}

fn make_service(arm: &str, seed: &str) -> CognitiveLoopService {
    let mut svc = CognitiveLoopService::new(base_config(seed)).expect("service construction");
    if arm == "frozen" {
        svc.freeze_cfc_training();
    }
    svc
}

#[derive(Clone, Copy)]
struct CycleRow {
    bits_persist: Option<f32>,
    bits_zero: Option<f32>,
    kappa: Option<f32>,
    pe: f32,
}

struct BlockStats {
    mean_bits_persist: f64,
    mean_bits_zero: f64,
    /// Mean Δcos vs persistence — the verdict quantity (Amendment 3).
    mean_dcos_persist: f64,
    mean_pe: f64,
    coverage: usize,
    total: usize,
}

fn summarize(rows: &[CycleRow]) -> BlockStats {
    let ln2 = std::f64::consts::LN_2;
    let persist: Vec<f64> = rows
        .iter()
        .filter_map(|r| r.bits_persist.map(|v| v as f64))
        .collect();
    let zero: Vec<f64> = rows
        .iter()
        .filter_map(|r| r.bits_zero.map(|v| v as f64))
        .collect();
    // Δcos = bits·ln2/κ — dimensionless, bounded; the quantity C1 verdicts
    // are evaluated on (Amendment 3).
    let dcos: Vec<f64> = rows
        .iter()
        .filter_map(|r| match (r.bits_persist, r.kappa) {
            (Some(b), Some(k)) if k > 0.0 => Some(b as f64 * ln2 / k as f64),
            _ => None,
        })
        .collect();
    let pe: f64 = rows.iter().map(|r| r.pe as f64).sum::<f64>() / rows.len().max(1) as f64;
    let mean = |v: &[f64]| {
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    BlockStats {
        mean_bits_persist: mean(&persist),
        mean_bits_zero: mean(&zero),
        mean_dcos_persist: mean(&dcos),
        mean_pe: pe,
        coverage: persist.len(),
        total: rows.len(),
    }
}

fn run_cycle(svc: &mut CognitiveLoopService, arm: &str, input: &str) -> CycleRow {
    if arm == "memoryless" {
        svc.reset_temporal_state();
    }
    let r = svc.cycle(input);
    CycleRow {
        bits_persist: r.bits_saved_persist,
        bits_zero: r.bits_saved_zero,
        kappa: r.bits_kappa,
        pe: r.prediction_error,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");

    // Coverage diagnostic (Amendment 3 item 3): per-cycle detail on the
    // alarming + varied regimes to pin which inputs produce bits=None.
    if args.iter().any(|a| a == "--probe") {
        let mut svc = make_service("live", SEEDS[0]);
        println!("C1| probe: alarming regime, per-cycle detail (live arm, seed alpha)");
        for i in 0..30 {
            let input = regime_input("alarming", i);
            let r = run_cycle(&mut svc, "live", input);
            println!(
                "C1| probe alarm i={:02} msg={} pe={:.4} bits={} kappa={}",
                i,
                i % 3,
                r.pe,
                r.bits_persist.map_or("None".into(), |v| format!("{v:.4}")),
                r.kappa.map_or("None".into(), |v| format!("{v:.1}")),
            );
        }
        println!("C1| probe: varied regime, per-cycle detail");
        for i in 0..36 {
            let input = regime_input("varied", i);
            let r = run_cycle(&mut svc, "live", input);
            println!(
                "C1| probe varied i={:02} sent={:02} pe={:.4} bits={}",
                i,
                i % 12,
                r.pe,
                r.bits_persist.map_or("None".into(), |v| format!("{v:.4}")),
            );
        }
        return;
    }
    let (seeds, regime_cycles, regime_warmup, script_reps) = if quick {
        (&SEEDS[..3], 200, 50, 20)
    } else {
        (SEEDS, REGIME_CYCLES, REGIME_WARMUP, SCRIPT_REPS)
    };

    println!("Predictive Compression C1 — bits-saved A/B");
    println!("protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md (§5 + Amendments 1-3)");
    println!(
        "mode: {} | seeds: {} | regimes: 4×{} (warmup {}) | script reps: {}",
        if quick {
            "QUICK (not the registered run)"
        } else {
            "FULL (registered)"
        },
        seeds.len(),
        regime_cycles,
        regime_warmup,
        script_reps,
    );
    println!();
    // Rows carry a "C1|" sigil: loop subsystems print their own study banners
    // to stdout mid-run, so the report is recovered with `grep '^C1|'`.
    println!(
        "C1| {:<12} {:<28} {:<11} {:>10} {:>12} {:>10} {:>8} {:>9}",
        "arm", "seed", "block", "dcos", "bits/persist", "bits/zero", "meanPE", "coverage"
    );

    let script = varied_script();
    for seed in seeds {
        let seed_short = seed
            .trim_start_matches("compression-c1-seed-")
            .trim_end_matches("-2026-07-17");
        for arm in ARMS {
            let mut svc = make_service(arm, seed);

            // Blocks 1–4: the E2 regimes, fixed order.
            for regime in ["repetitive", "varied", "alarming", "empty"] {
                let mut measured = Vec::with_capacity(regime_cycles - regime_warmup);
                for i in 0..regime_cycles {
                    let row = run_cycle(&mut svc, arm, regime_input(regime, i));
                    if i >= regime_warmup {
                        measured.push(row);
                    }
                }
                let s = summarize(&measured);
                println!(
                    "C1| {:<12} {:<28} {:<11} {:>10.5} {:>12.4} {:>10.4} {:>8.4} {:>6}/{}",
                    arm,
                    seed_short,
                    regime,
                    s.mean_dcos_persist,
                    s.mean_bits_persist,
                    s.mean_bits_zero,
                    s.mean_pe,
                    s.coverage,
                    s.total,
                );
            }

            // Block 5: learned script — 60 reps × 12 sentences.
            let mut per_rep: Vec<Vec<CycleRow>> = Vec::new();
            for _rep in 0..script_reps {
                let mut rep_rows = Vec::with_capacity(12);
                for line in &script {
                    rep_rows.push(run_cycle(&mut svc, arm, line));
                }
                per_rep.push(rep_rows);
            }
            let early: Vec<_> = per_rep[1.min(per_rep.len() - 1)..4.min(per_rep.len())]
                .iter()
                .flatten()
                .cloned()
                .collect();
            let late_start = per_rep.len().saturating_sub(6);
            let late: Vec<_> = per_rep[late_start..].iter().flatten().cloned().collect();
            let se = summarize(&early);
            let sl = summarize(&late);
            // learning_growth on Δcos (Amendment 2 sign convention, Amendment 3
            // quantity): positive = the prediction tracks the stream better
            // with exposure.
            let growth = sl.mean_dcos_persist - se.mean_dcos_persist;
            println!(
                "C1| {:<12} {:<28} {:<11} {:>10.5} {:>12.4} {:>10.4} {:>8.4} {:>6}/{}  (dcos early {:.5} → late {:.5}, growth {:+.5})",
                arm,
                seed_short,
                "script",
                sl.mean_dcos_persist,
                sl.mean_bits_persist,
                sl.mean_bits_zero,
                sl.mean_pe,
                sl.coverage,
                sl.total,
                se.mean_dcos_persist,
                sl.mean_dcos_persist,
                growth,
            );
        }
        println!();
    }
    println!(
        "done. Append results + verdicts to the protocol doc (P-labels), per house convention."
    );
}
