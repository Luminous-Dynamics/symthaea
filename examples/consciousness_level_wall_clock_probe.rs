// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Direct counterfactual test for the load-dependent `consciousness_level` finding
//! (memory/symthaea_e1_consciousness_level_load_dependent_jul28.md).
//!
//! The clean `exp_loop_ablation --section e1` rerun found `consciousness_level` split into
//! two sharp regimes across 15 sequential arms (~0.06 during heavy host contention / slow
//! wall-clock cycles vs ~0.645 once load eased), correlated with cycle wall-time, not with
//! which subsystem was ablated. Static tracing ruled out one candidate (a dead, never-cleared
//! `ENTROPY_CACHE` -- confirmed not on the live Phi path) and found no process-global state in
//! the `consciousness_level` computation chain (`master_equation`/`narrative_coherence`
//! modules), nor any wall-clock dependency in `accumulate_allostatic_load` (purely
//! per-call/cycle-count based, no `Instant::now()`).
//!
//! This is a genuine on/off counterfactual instead of more static reading: run the SAME
//! baseline config, same fixed input script, same cycle count, TWICE in one process --
//! once at natural (fast) pace, once with an artificial `std::thread::sleep` injected between
//! cycles to simulate the wall-clock slowdown observed under heavy load. If consciousness_level
//! diverges between the two conditions despite identical cycle counts and identical config,
//! that confirms real wall-clock time-per-cycle (not cycle count, not which subsystem is
//! ablated) is a causal input somewhere in the chain. If both converge to the same value, the
//! wall-clock hypothesis is refuted and the true mechanism remains open.
//!
//! Run: cargo run --example consciousness_level_wall_clock_probe

use std::time::Duration;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const GENESIS: &str = "cl-wall-clock-probe-2026-07-28";
const CYCLES: usize = 150;
/// Injected delay per cycle in the SLOW condition, chosen to be comparable in order of
/// magnitude to the e1 rerun's heaviest-contention per-cycle wall time (8.5-11.4s), while
/// keeping total probe wall time bounded (150 cycles x 3s = 450s = 7.5min for the slow arm).
const SLEEP_PER_CYCLE: Duration = Duration::from_secs(3);

fn input_script() -> Vec<&'static str> {
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

fn base_config() -> CognitiveLoopConfig {
    let mut c = CognitiveLoopConfig::default();
    c.genesis_phrase = Some(GENESIS.to_string());
    c.async_training = false;
    c
}

/// Run `CYCLES` cycles, optionally sleeping `sleep_per_cycle` after each one. Returns
/// (final consciousness_level, per-cycle trace of consciousness_level, mean measured
/// wall-clock micros/cycle, final attention_budget_exceeded_count if accessible).
fn run_condition(label: &str, sleep_per_cycle: Option<Duration>) -> (f64, Vec<f64>) {
    let config = base_config();
    let mut svc = CognitiveLoopService::new(config).expect("service construction");
    let script = input_script();
    let mut trace = Vec::with_capacity(CYCLES);

    let t_start = std::time::Instant::now();
    for (i, input) in script.iter().cycle().take(CYCLES).enumerate() {
        let r = svc.cycle(input);
        let cl = r.metadata.consciousness.consciousness_level;
        trace.push(cl);
        if i % 25 == 0 || i == CYCLES - 1 {
            println!("  [{label}] cycle {i:>3}: consciousness_level={cl:.4}");
        }
        if let Some(d) = sleep_per_cycle {
            std::thread::sleep(d);
        }
    }
    let elapsed = t_start.elapsed();
    let mean_us_per_cycle = elapsed.as_micros() as f64 / CYCLES as f64;
    println!(
        "  [{label}] done: {CYCLES} cycles in {:.1}s ({mean_us_per_cycle:.0} us/cycle mean)",
        elapsed.as_secs_f64()
    );

    (*trace.last().unwrap(), trace)
}

fn main() {
    println!("consciousness_level wall-clock counterfactual probe");
    println!("protocol: memory/symthaea_e1_consciousness_level_load_dependent_jul28.md follow-up");
    println!(
        "CYCLES={CYCLES}, SLEEP_PER_CYCLE={:.1}s (SLOW condition only)",
        SLEEP_PER_CYCLE.as_secs_f64()
    );
    println!();

    println!("-- FAST condition (no artificial delay, natural ambient pace) --");
    let (fast_final, fast_trace) = run_condition("FAST", None);
    println!();

    println!("-- SLOW condition (artificial sleep injected between cycles) --");
    let (slow_final, slow_trace) = run_condition("SLOW", Some(SLEEP_PER_CYCLE));
    println!();

    let last_n = 20.min(fast_trace.len()).min(slow_trace.len());
    let fast_tail_mean: f64 =
        fast_trace[fast_trace.len() - last_n..].iter().sum::<f64>() / last_n as f64;
    let slow_tail_mean: f64 =
        slow_trace[slow_trace.len() - last_n..].iter().sum::<f64>() / last_n as f64;

    println!("=== RESULT ===");
    println!(
        "FAST final consciousness_level: {fast_final:.4} (last-{last_n} mean: {fast_tail_mean:.4})"
    );
    println!(
        "SLOW final consciousness_level: {slow_final:.4} (last-{last_n} mean: {slow_tail_mean:.4})"
    );
    println!(
        "delta (SLOW - FAST), last-{last_n} mean: {:+.4}",
        slow_tail_mean - fast_tail_mean
    );
    println!();
    if (slow_tail_mean - fast_tail_mean).abs() > 0.1 {
        println!(
            "VERDICT: large divergence despite identical cycle count and config -- wall-clock \
             time-per-cycle IS a causal input somewhere in the consciousness_level chain. \
             Hypothesis CONFIRMED."
        );
    } else {
        println!(
            "VERDICT: FAST and SLOW converge to similar values despite very different wall-clock \
             pacing -- the wall-clock-time hypothesis is REFUTED for this specific mechanism. \
             The e1 rerun's load-correlated pattern needs a different explanation (candidates: \
             a still-unidentified process-global state elsewhere, or a coincidental confound \
             in the e1 run itself, e.g. contention affecting something other than cycle pacing)."
        );
    }
}
