// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive Compression C3f — homogeneity x difficulty factorial.
//!
//! Pre-registered protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md
//! "Experiment C3f" (P20/P21/P22/P23 -- committed BEFORE this harness existed).
//!
//! The homogeneity diagnostic confirmed C3e's EASY tier clusters ~470x more tightly than
//! HARD in the compressed space, but EASY/HARD conflate homogeneity with difficulty by
//! construction. C3f is the genuine causal test: a 2x2 -- homogeneous-easy (= C3e's EASY),
//! homogeneous-hard (new), heterogeneous-easy (new), heterogeneous-hard (= C3e's HARD) --
//! round-robin 1:1:1:1, same on/off confound-control methodology as C3d/C3e.
//!
//! Run: cargo run --release --example episodic_recall_factorial_probe            (full, 10 seeds, 400 cycles)
//!      cargo run --release --example episodic_recall_factorial_probe -- --quick (3 seeds, 160 cycles)

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

/// Cell A: Homogeneous + Easy (= C3e's EASY, reused as-is).
const CELL_A_HOMOG_EASY: [&str; 4] = [
    "The cat sat on the mat.",
    "The dog sat on the rug.",
    "The bird sat on the branch.",
    "The fish swam in the bowl.",
];

/// Cell B: Homogeneous + Hard (new -- one shared rigid template, information-dense content).
const CELL_B_HOMOG_HARD: [&str; 4] = [
    "The catalyst exhibits nonlinear behavior under high-pressure conditions.",
    "The alloy exhibits fatigue behavior under cyclic-loading conditions.",
    "The organism exhibits adaptive behavior under resource-scarce conditions.",
    "The algorithm exhibits divergent behavior under adversarial conditions.",
];

/// Cell C: Heterogeneous + Easy (new -- structurally distinct, individually predictable).
const CELL_C_HETEROG_EASY: [&str; 4] = [
    "It is raining outside today.",
    "She likes hot tea in the morning.",
    "The store closes at nine tonight.",
    "He walked his dog around the block.",
];

/// Cell D: Heterogeneous + Hard (= C3e's HARD, reused as-is).
const CELL_D_HETEROG_HARD: [&str; 4] = [
    "Quantum entanglement links particles across arbitrary distances instantaneously.",
    "The committee postponed its decision pending further budgetary review.",
    "Despite forecasts, unexpected turbulence delayed the connecting flight significantly.",
    "Her handwriting, illegible at first glance, revealed a hidden apology.",
];

const CELLS: [(&str, [&str; 4]); 4] = [
    ("homog_easy", CELL_A_HOMOG_EASY),
    ("homog_hard", CELL_B_HOMOG_HARD),
    ("heterog_easy", CELL_C_HETEROG_EASY),
    ("heterog_hard", CELL_D_HETEROG_HARD),
];

/// Round-robin 1:1:1:1 across all four cells -- equal recurrence frequency for every tier.
fn build_schedule(total_cycles: usize) -> Vec<(&'static str, &'static str)> {
    let mut schedule = Vec::with_capacity(total_cycles);
    let mut idx = [0usize; 4];
    for i in 0..total_cycles {
        let cell_i = i % 4;
        let (tier, sentences) = CELLS[cell_i];
        schedule.push((sentences[idx[cell_i] % 4], tier));
        idx[cell_i] += 1;
    }
    schedule
}

fn base_config(seed: &str, recall_on: bool) -> CognitiveLoopConfig {
    let mut c = CognitiveLoopConfig::default();
    c.genesis_phrase = Some(seed.to_string());
    c.async_training = false;
    c.enable_episodic_recall_prediction = recall_on;
    c
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let (seeds, cycles) = if quick {
        (&SEEDS[..3], 160)
    } else {
        (SEEDS, 400)
    };

    println!("Predictive Compression C3f -- homogeneity x difficulty factorial");
    println!("protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md (Experiment C3f)");
    println!(
        "mode: {} | seeds: {} | cycles: {} | 4 cells round-robin 1:1:1:1",
        if quick {
            "QUICK (not the registered run)"
        } else {
            "FULL (registered)"
        },
        seeds.len(),
        cycles,
    );
    println!();
    println!(
        "C3f| {:<28} {:<13} {:>10} {:>10} {:>10} {:>8}",
        "seed", "cell", "on_mean", "off_mean", "diff", "n_each"
    );

    for seed in seeds {
        let seed_short = seed
            .trim_start_matches("episodic-recall-c3-seed-")
            .trim_end_matches("-2026-07-25");
        let schedule = build_schedule(cycles);

        let mut svc_on =
            CognitiveLoopService::new(base_config(seed, true)).expect("service construction");
        let mut on_bits: Vec<(&str, f64)> = Vec::new();
        for (content, tier) in &schedule {
            let r = svc_on.cycle(content);
            if let Some(b) = r.bits_saved_persist {
                on_bits.push((tier, b as f64));
            }
        }

        let mut svc_off =
            CognitiveLoopService::new(base_config(seed, false)).expect("service construction");
        let mut off_bits: Vec<(&str, f64)> = Vec::new();
        for (content, tier) in &schedule {
            let r = svc_off.cycle(content);
            if let Some(b) = r.bits_saved_persist {
                off_bits.push((tier, b as f64));
            }
        }

        for (tier, _) in CELLS {
            let on_vals: Vec<f64> = on_bits
                .iter()
                .filter(|(t, _)| *t == tier)
                .map(|(_, b)| *b)
                .collect();
            let off_vals: Vec<f64> = off_bits
                .iter()
                .filter(|(t, _)| *t == tier)
                .map(|(_, b)| *b)
                .collect();
            let on_mean = if on_vals.is_empty() {
                f64::NAN
            } else {
                on_vals.iter().sum::<f64>() / on_vals.len() as f64
            };
            let off_mean = if off_vals.is_empty() {
                f64::NAN
            } else {
                off_vals.iter().sum::<f64>() / off_vals.len() as f64
            };
            println!(
                "C3f| {:<28} {:<13} {:>10.5} {:>10.5} {:>+10.5} {:>8}",
                seed_short,
                tier,
                on_mean,
                off_mean,
                on_mean - off_mean,
                on_vals.len(),
            );
        }
    }

    println!();
    println!(
        "done. Append results + verdict (P20/P21/P22/P23) to the protocol doc (P-labels), per house convention."
    );
}
