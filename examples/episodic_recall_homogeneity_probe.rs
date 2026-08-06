// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive Compression -- within-tier homogeneity diagnostic.
//!
//! Pre-registered protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md
//! "Diagnostic -- within-tier homogeneity probe" (P18/P19 -- committed BEFORE this
//! harness existed).
//!
//! C3e disclosed a self-introduced confound: EASY and HARD differ in raw predictability
//! AND (plausibly) in within-tier surface homogeneity, conflated by construction. This
//! probe checks the homogeneity half directly and cheaply: `recall_similarity` is already
//! populated whenever a nearest-neighbor candidate exists, whether or not it clears the
//! firing threshold -- so storing one sentence and probing with another yields a genuine
//! continuous similarity score with zero new production code.
//!
//! Design: fresh service per ordered pair (store i, probe j), both cross-pairs (i != j)
//! and self-pairs (i == j, a calibration ceiling), across 3 seeds -- a descriptive
//! diagnostic (C3c precedent), not a 10-seed A/B.
//!
//! Run: cargo run --release --example episodic_recall_homogeneity_probe

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const SEEDS: &[&str] = &[
    "episodic-recall-c3-seed-alpha-2026-07-25",
    "episodic-recall-c3-seed-beta-2026-07-25",
    "episodic-recall-c3-seed-gamma-2026-07-25",
];

const EASY: [&str; 4] = [
    "The cat sat on the mat.",
    "The dog sat on the rug.",
    "The bird sat on the branch.",
    "The fish swam in the bowl.",
];

const HARD: [&str; 4] = [
    "Quantum entanglement links particles across arbitrary distances instantaneously.",
    "The committee postponed its decision pending further budgetary review.",
    "Despite forecasts, unexpected turbulence delayed the connecting flight significantly.",
    "Her handwriting, illegible at first glance, revealed a hidden apology.",
];

/// C3f Cell B: Homogeneous + (intended) Hard.
const HOMOG_HARD: [&str; 4] = [
    "The catalyst exhibits nonlinear behavior under high-pressure conditions.",
    "The alloy exhibits fatigue behavior under cyclic-loading conditions.",
    "The organism exhibits adaptive behavior under resource-scarce conditions.",
    "The algorithm exhibits divergent behavior under adversarial conditions.",
];

/// C3f Cell C: Heterogeneous + (intended) Easy.
const HETEROG_EASY: [&str; 4] = [
    "It is raining outside today.",
    "She likes hot tea in the morning.",
    "The store closes at nine tonight.",
    "He walked his dog around the block.",
];

fn base_config(seed: &str) -> CognitiveLoopConfig {
    let mut c = CognitiveLoopConfig::default();
    c.genesis_phrase = Some(seed.to_string());
    c.async_training = false;
    c.enable_episodic_recall_prediction = true;
    c
}

/// Store `store_text` (one cycle, empty prior store), then probe with `probe_text`
/// (one cycle) and return the resulting `recall_similarity` (populated unconditionally
/// once the store is non-empty, per `planning.rs`).
fn pairwise_similarity(seed: &str, store_text: &str, probe_text: &str) -> Option<f32> {
    let mut svc = CognitiveLoopService::new(base_config(seed)).expect("service construction");
    let _ = svc.cycle(store_text);
    let r = svc.cycle(probe_text);
    r.recall_similarity
}

fn main() {
    println!("Predictive Compression -- within-tier homogeneity diagnostic");
    println!("protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md (Diagnostic)");
    println!();
    println!(
        "HOMOG| {:<8} {:<6} {:>4} {:>4} {:>10}",
        "seed", "tier", "i", "j", "similarity"
    );

    for seed in SEEDS {
        let seed_short = seed
            .trim_start_matches("episodic-recall-c3-seed-")
            .trim_end_matches("-2026-07-25");

        for (tier_name, tier) in [
            ("easy", EASY),
            ("hard", HARD),
            ("homog_hard", HOMOG_HARD),
            ("heterog_easy", HETEROG_EASY),
        ] {
            let mut cross_sims: Vec<f32> = Vec::new();
            let mut self_sims: Vec<f32> = Vec::new();

            for i in 0..4 {
                for j in 0..4 {
                    if let Some(sim) = pairwise_similarity(seed, tier[i], tier[j]) {
                        println!(
                            "HOMOG| {:<8} {:<6} {:>4} {:>4} {:>10.5}",
                            seed_short, tier_name, i, j, sim
                        );
                        if i == j {
                            self_sims.push(sim);
                        } else {
                            cross_sims.push(sim);
                        }
                    }
                }
            }

            let cross_mean = cross_sims.iter().sum::<f32>() / cross_sims.len() as f32;
            let self_mean = self_sims.iter().sum::<f32>() / self_sims.len() as f32;
            println!(
                "HOMOG-SUMMARY| {:<8} {:<6} cross_mean={:>8.5} self_mean={:>8.5} n_cross={} n_self={}",
                seed_short,
                tier_name,
                cross_mean,
                self_mean,
                cross_sims.len(),
                self_sims.len(),
            );
        }
    }

    println!();
    println!(
        "done. Append results + verdict (P18/P19) to the protocol doc (P-labels), per house convention."
    );
}
