// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive Compression C3c — root-cause diagnostic for the C3b paraphrase harm.
//!
//! Pre-registered protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md
//! "Experiment C3c" (P12/P13 — committed BEFORE this harness existed).
//!
//! Tests C3b Follow-up 1's leading hypothesis directly: does recall on a
//! paraphrase cycle typically match against a PROTOTYPE episode (not another
//! paraphrase occurrence), blending in a specific-but-wrong target? Uses the
//! identical content/schedule design as `episodic_recall_gradient_probe.rs`
//! (separate copy — examples cannot import each other), plus the new
//! `recall_matched_timestamp` telemetry to look up the matched episode's
//! own tier via the same deterministic schedule.
//!
//! Single representative seed (this is a descriptive diagnostic, not a new
//! A/B needing the 10-seed sign test).
//!
//! Run: cargo run --release --example episodic_recall_provenance_probe

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const SEED: &str = "episodic-recall-c3-seed-alpha-2026-07-25";
const CYCLES: usize = 400;

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

/// Identical to episodic_recall_gradient_probe.rs's build_schedule (separate
/// copy per the coordination rule).
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

fn main() {
    println!("Predictive Compression C3c -- recall provenance diagnostic");
    println!("protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md (Experiment C3c)");
    println!("seed: {SEED} | cycles: {CYCLES} | single representative seed, recall_on");
    println!();

    let mut config = CognitiveLoopConfig::default();
    config.genesis_phrase = Some(SEED.to_string());
    config.async_training = false;
    config.enable_episodic_recall_prediction = true;
    let mut svc = CognitiveLoopService::new(config).expect("service construction");

    let schedule = build_schedule(CYCLES);

    // (current_tier, matched_tier) for every cycle where recall fired.
    let mut matches: Vec<(&str, &str)> = Vec::new();
    // Also track raw similarity for context.
    let mut unmatched_lookups = 0usize;

    for (i, (content, current_tier)) in schedule.iter().enumerate() {
        let r = svc.cycle(content);
        if r.recall_fired
            && let Some(ts) = r.recall_matched_timestamp
        {
            // Episode.timestamp is written from self.stats.total_cycles AFTER
            // it's incremented at the top of cycle() -- so cycle index i
            // (0-based, this loop) corresponds to timestamp i+1.
            let matched_idx = ts.saturating_sub(1) as usize;
            if matched_idx < schedule.len() && matched_idx <= i {
                matches.push((current_tier, schedule[matched_idx].1));
            } else {
                unmatched_lookups += 1;
            }
        }
    }

    println!(
        "total cycles: {} | recalls fired+matched: {} | unresolvable timestamp lookups: {}",
        CYCLES,
        matches.len(),
        unmatched_lookups
    );
    println!();

    for current_tier in ["prototype", "paraphrase", "novel"] {
        let this_tier: Vec<&str> = matches
            .iter()
            .filter(|(cur, _)| *cur == current_tier)
            .map(|(_, matched)| *matched)
            .collect();
        let n = this_tier.len();
        if n == 0 {
            println!("C3c| current_tier={current_tier:<10} n=0 (no fired recalls)");
            continue;
        }
        let proto_matches = this_tier.iter().filter(|t| **t == "prototype").count();
        let para_matches = this_tier.iter().filter(|t| **t == "paraphrase").count();
        let novel_matches = this_tier.iter().filter(|t| **t == "novel").count();
        println!(
            "C3c| current_tier={:<10} n={:<4} matched_prototype={:.1}% matched_paraphrase={:.1}% matched_novel={:.1}%",
            current_tier,
            n,
            100.0 * proto_matches as f64 / n as f64,
            100.0 * para_matches as f64 / n as f64,
            100.0 * novel_matches as f64 / n as f64,
        );
    }

    println!();
    println!("done. Append results + verdict (P12/P13) to the protocol doc, per house convention.");
}
