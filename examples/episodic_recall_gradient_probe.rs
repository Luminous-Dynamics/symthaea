// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive Compression C3b — similarity-gradient probe.
//!
//! Pre-registered protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md
//! "Experiment C3b" (P8/P9 — committed BEFORE this harness existed).
//!
//! C3's main run (`episodic_recall_probe.rs`) saturated at a 100%/0% recall
//! hit-rate because both its blocks used deliberately repeated content. This
//! probe uses a 3-tier content design to produce a genuine similarity
//! SPREAD, then asks whether `bits_saved_persist` shows a dose-response
//! relationship with `recall_similarity` within the `recall_on` arm.
//!
//! Single arm only (this is a within-run stratification question, not an
//! on/off comparison) — same 10 seeds reused from C3.
//!
//! Run: cargo run --release --example episodic_recall_gradient_probe            (full, 10 seeds, 400 cycles)
//!      cargo run --release --example episodic_recall_gradient_probe -- --quick (3 seeds, 150 cycles)

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

/// Repeated many times — expected to produce high-similarity recalls
/// (~0.9+) once the store has a few reps of history.
const PROTOTYPES: [&str; 4] = [
    "The water cycle moves moisture from oceans to clouds to rain.",
    "The reactor coolant temperature is rising faster than expected.",
    "She placed the last puzzle piece and smiled at the finished picture.",
    "A gentle rain began to fall as the travelers reached the shelter.",
];

/// Same topic/meaning as the prototypes above (same index), different
/// wording — expected medium similarity against the matching prototype's
/// episodes, but this is NOT assumed: HDC similarity doesn't guarantee
/// tracking semantic paraphrase similarity, and that's part of what this
/// probe observes.
const PARAPHRASES: [&str; 4] = [
    "Water evaporates from the sea, forms clouds, and falls again as rain.",
    "Coolant temperature in the reactor is climbing more quickly than anticipated.",
    "She fit the final piece into the puzzle and grinned at the completed image.",
    "As the travelers arrived at the shelter, a soft rain started falling.",
];

/// Each appears exactly once — expected low similarity or no recall at all
/// on first (and only) appearance.
const NOVELS: [&str; 12] = [
    "The blacksmith hammered the horseshoe until it rang true.",
    "Quantum tunneling lets particles cross barriers classical physics forbids.",
    "The committee postponed the vote until further evidence arrived.",
    "Frost crept up the window in delicate fern-like patterns overnight.",
    "The violinist tuned each string before the orchestra began.",
    "A stray cat wandered into the bakery and refused to leave.",
    "The treaty was signed after eleven hours of tense negotiation.",
    "Moss grows thickest on the north side of old stone walls.",
    "The engineer traced the fault to a single corroded relay.",
    "Grandfather told the same story every year at the harvest table.",
    "The tide pool held a universe of tiny darting creatures.",
    "Static electricity crackled as she pulled off her wool sweater.",
];

/// Build a 400-cycle schedule: prototypes appear ~4x as often as
/// paraphrases+novel combined (320 : 80, exactly 4x). Round = 4 prototype
/// cycles + 1 "X" cycle alternating novel/paraphrase; novel exhausts after
/// its 12 uses and the X slot falls back to paraphrase thereafter.
fn build_schedule(total_cycles: usize) -> Vec<(&'static str, &'static str)> {
    let mut schedule = Vec::with_capacity(total_cycles);
    let mut proto_i = 0usize;
    let mut para_i = 0usize;
    let mut novel_i = 0usize;
    let mut round = 0usize;
    while schedule.len() < total_cycles {
        for _ in 0..4 {
            if schedule.len() >= total_cycles {
                break;
            }
            schedule.push((PROTOTYPES[proto_i % 4], "prototype"));
            proto_i += 1;
        }
        if schedule.len() >= total_cycles {
            break;
        }
        if round % 2 == 0 && novel_i < NOVELS.len() {
            schedule.push((NOVELS[novel_i], "novel"));
            novel_i += 1;
        } else {
            schedule.push((PARAPHRASES[para_i % 4], "paraphrase"));
            para_i += 1;
        }
        round += 1;
    }
    schedule
}

fn base_config(seed: &str) -> CognitiveLoopConfig {
    let mut c = CognitiveLoopConfig::default();
    c.genesis_phrase = Some(seed.to_string());
    c.async_training = false;
    c.enable_episodic_recall_prediction = true;
    c
}

/// Similarity bins, floor at the registered recall threshold (0.5).
const BIN_EDGES: [(f32, f32); 3] = [(0.5, 0.7), (0.7, 0.9), (0.9, 1.000001)];
const BIN_NAMES: [&str; 3] = ["[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]"];
const MIN_SAMPLES_PER_BIN: usize = 20;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let (seeds, cycles) = if quick {
        (&SEEDS[..3], 150)
    } else {
        (SEEDS, 400)
    };

    println!("Predictive Compression C3b -- similarity-gradient probe");
    println!("protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md (Experiment C3b)");
    println!(
        "mode: {} | seeds: {} | cycles: {} | single arm: recall_on",
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
        "C3b| {:<28} {:<11} {:>8} {:>14}",
        "seed", "bin", "n", "mean_bits"
    );

    for seed in seeds {
        let seed_short = seed
            .trim_start_matches("episodic-recall-c3-seed-")
            .trim_end_matches("-2026-07-25");
        let mut svc = CognitiveLoopService::new(base_config(seed)).expect("service construction");
        let schedule = build_schedule(cycles);

        // (recall_similarity, bits_saved_persist) for every cycle where
        // recall fired.
        let mut samples: Vec<(f32, f64)> = Vec::new();
        for (content, _tier) in &schedule {
            let r = svc.cycle(content);
            if r.recall_fired
                && let Some(sim) = r.recall_similarity
                && let Some(bits) = r.bits_saved_persist
            {
                samples.push((sim, bits as f64));
            }
        }

        for (bin_idx, &(lo, hi)) in BIN_EDGES.iter().enumerate() {
            let in_bin: Vec<f64> = samples
                .iter()
                .filter(|(sim, _)| *sim >= lo && *sim < hi)
                .map(|(_, b)| *b)
                .collect();
            let n = in_bin.len();
            let mean = if n > 0 {
                in_bin.iter().sum::<f64>() / n as f64
            } else {
                f64::NAN
            };
            let flag = if n < MIN_SAMPLES_PER_BIN {
                " (UNDERPOWERED)"
            } else {
                ""
            };
            println!(
                "C3b| {:<28} {:<11} {:>8} {:>14.5}{}",
                seed_short, BIN_NAMES[bin_idx], n, mean, flag
            );
        }
    }
    println!();
    println!(
        "done. Append results + verdicts to the protocol doc (P-labels), per house convention."
    );
}
