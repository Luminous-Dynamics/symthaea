// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive Compression C3d — frequency-equalization test.
//!
//! Pre-registered protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md
//! "Experiment C3d" (P14/P15 — committed BEFORE this harness existed).
//!
//! C3c refuted category confusion (paraphrase recalls cleanly match prior
//! paraphrases, never prototypes) and left recurrence-frequency as the
//! leading hypothesis for C3b's paraphrase-specific harm (paraphrases
//! recurred ~5x less often than prototypes there). This probe removes that
//! confound: prototype and paraphrase content alternate 1:1, giving them
//! equal recurrence frequency, then reruns C3b's exact confound-control
//! methodology (same schedule, both arms, stratified by tier).
//!
//! Novel content is dropped from this test — it fired zero recalls in C3c
//! by construction (each appears once) and contributes nothing to this
//! specific frequency question.
//!
//! Run: cargo run --release --example episodic_recall_frequency_probe            (full, 10 seeds, 400 cycles)
//!      cargo run --release --example episodic_recall_frequency_probe -- --quick (3 seeds, 150 cycles)

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

const PROTOTYPES: [&str; 4] = [
    "The water cycle moves moisture from oceans to clouds to rain.",
    "The reactor coolant temperature is rising faster than expected.",
    "She placed the last puzzle piece and smiled at the finished picture.",
    "A gentle rain began to fall as the travelers reached the shelter.",
];

const PARAPHRASES: [&str; 4] = [
    "Water evaporates from the sea, forms clouds, and falls again as rain.",
    "Coolant temperature in the reactor is climbing more quickly than anticipated.",
    "She fit the final piece into the puzzle and grinned at the completed image.",
    "As the travelers arrived at the shelter, a soft rain started falling.",
];

/// Strict 1:1 alternation: prototype, paraphrase, prototype, paraphrase...
/// Equal recurrence frequency for both tiers by construction (~200 cycles
/// each at 400 total, vs C3b/C3c's ~312 prototype / ~62 paraphrase split).
fn build_schedule(total_cycles: usize) -> Vec<(&'static str, &'static str)> {
    let mut schedule = Vec::with_capacity(total_cycles);
    let mut proto_i = 0usize;
    let mut para_i = 0usize;
    for i in 0..total_cycles {
        if i % 2 == 0 {
            schedule.push((PROTOTYPES[proto_i % 4], "prototype"));
            proto_i += 1;
        } else {
            schedule.push((PARAPHRASES[para_i % 4], "paraphrase"));
            para_i += 1;
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

    println!("Predictive Compression C3d -- frequency-equalization test");
    println!("protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md (Experiment C3d)");
    println!(
        "mode: {} | seeds: {} | cycles: {} | prototype:paraphrase = 1:1 (equal frequency)",
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
        "C3d| {:<28} {:<11} {:>10} {:>10} {:>10} {:>8}",
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

        for tier in ["prototype", "paraphrase"] {
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
                "C3d| {:<28} {:<11} {:>10.5} {:>10.5} {:>+10.5} {:>8}",
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
        "done. Append results + verdict (P14/P15) to the protocol doc (P-labels), per house convention."
    );
}
