// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive Compression C3e — content-difficulty test.
//!
//! Pre-registered protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md
//! "Experiment C3e" (P16/P17 — committed BEFORE this harness existed).
//!
//! C3c refuted category confusion; C3d refuted recurrence-frequency. Both left one
//! observation unexplained: paraphrase content's off-arm (recall-disabled) bits-saved
//! baseline was measurably lower than prototype's in every C3d seed -- paraphrases are
//! inherently harder to predict, independent of recall. C3e tests this directly with a
//! fresh easy/hard axis, deliberately decoupled from the prototype/paraphrase category
//! label, using the exact same confound-control methodology as C3d (equal 1:1 frequency,
//! both arms, stratified by tier, same 10 seeds, 400 cycles).
//!
//! Run: cargo run --release --example episodic_recall_difficulty_probe            (full, 10 seeds, 400 cycles)
//!      cargo run --release --example episodic_recall_difficulty_probe -- --quick (3 seeds, 150 cycles)

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

/// Low-surprise: one shared subject-verb-location template, only the nouns vary.
const EASY: [&str; 4] = [
    "The cat sat on the mat.",
    "The dog sat on the rug.",
    "The bird sat on the branch.",
    "The fish swam in the bowl.",
];

/// High-surprise: four structurally unrelated, information-dense sentences, no shared template.
const HARD: [&str; 4] = [
    "Quantum entanglement links particles across arbitrary distances instantaneously.",
    "The committee postponed its decision pending further budgetary review.",
    "Despite forecasts, unexpected turbulence delayed the connecting flight significantly.",
    "Her handwriting, illegible at first glance, revealed a hidden apology.",
];

/// Strict 1:1 alternation, mirroring C3d exactly: equal recurrence frequency for both
/// tiers by construction (~200 cycles each at 400 total).
fn build_schedule(total_cycles: usize) -> Vec<(&'static str, &'static str)> {
    let mut schedule = Vec::with_capacity(total_cycles);
    let mut easy_i = 0usize;
    let mut hard_i = 0usize;
    for i in 0..total_cycles {
        if i % 2 == 0 {
            schedule.push((EASY[easy_i % 4], "easy"));
            easy_i += 1;
        } else {
            schedule.push((HARD[hard_i % 4], "hard"));
            hard_i += 1;
        }
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
        (&SEEDS[..3], 150)
    } else {
        (SEEDS, 400)
    };

    println!("Predictive Compression C3e -- content-difficulty test");
    println!("protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md (Experiment C3e)");
    println!(
        "mode: {} | seeds: {} | cycles: {} | easy:hard = 1:1 (equal frequency, fresh difficulty axis)",
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
        "C3e| {:<28} {:<7} {:>10} {:>10} {:>10} {:>8}",
        "seed", "tier", "on_mean", "off_mean", "diff", "n_each"
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

        for tier in ["easy", "hard"] {
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
                "C3e| {:<28} {:<7} {:>10.5} {:>10.5} {:>+10.5} {:>8}",
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
        "done. Append results + verdict (P16/P17) to the protocol doc (P-labels), per house convention."
    );
}
