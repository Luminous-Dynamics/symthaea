// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Keystone A/B — does the consciousness machinery help?
//!
//! Pre-registered protocol: docs/KEYSTONE_AB_PROTOCOL_2026-07-17.md
//! (arms, tasks, metrics, and falsifiable predictions were committed BEFORE
//! this harness first ran — read it before interpreting output).
//!
//! Arms: `full` (default), `min2` (13 NULL-verdict subsystems off, the two
//! load-bearing ones kept), `off15` (all 15 flag-gated subsystems off).
//! Tasks: predictive learning on a repeated sequence, surprise contrast on
//! novel input, regime-separation manipulation check, and compute cost.
//!
//! Deterministic per (arm, seed): fixed genesis phrase, async_training off,
//! fixed input schedule. Rows print as they complete so partial output
//! survives an external kill.
//!
//! Run: cargo run --release --example keystone_ab

use std::time::Instant;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const SEEDS: &[&str] = &[
    "keystone-ab-seed-alpha-2026-07-17",
    "keystone-ab-seed-beta-2026-07-17",
    "keystone-ab-seed-gamma-2026-07-17",
    "keystone-ab-seed-delta-2026-07-17",
    "keystone-ab-seed-epsilon-2026-07-17",
    "keystone-ab-seed-zeta-2026-07-17",
    "keystone-ab-seed-eta-2026-07-17",
    "keystone-ab-seed-theta-2026-07-17",
    "keystone-ab-seed-iota-2026-07-17",
    "keystone-ab-seed-kappa-2026-07-17",
];

const REPS: usize = 60; // learned-sequence repetitions

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

fn novel_script() -> Vec<&'static str> {
    vec![
        "Volcanic ash grounded flights across the northern corridor.",
        "He whittled a small boat from driftwood while the kettle sang.",
        "Should an heir refuse a fortune built on stolen land?",
        "Neutron flux in channel seven exceeds the amber threshold.",
        "Seven times eight is fifty-six, and nine squared is eighty-one.",
        "The choir's final chord hung in the rafters like light.",
        "Alert: checksum mismatch detected in the boot partition image.",
        "Glaciers carved this valley long before any road crossed it.",
        "Can forgiveness be genuine if the wound is never named?",
        "Copper futures spiked after the port strike entered week two.",
        "Fog rolled off the marsh and swallowed the harbor bells.",
        "Verify the harness anchors before descending into the shaft.",
        "The library's card catalog still smells of cedar and dust.",
        "A single bee traced circles above the clover at noon.",
        "Is loyalty owed to an institution that betrays its charter?",
        "Turbine bearing vibration is trending upward at two hertz.",
        "The integral of one over x is the natural logarithm of x.",
        "Grandmother's recipe called for patience more than flour.",
        "Notice: certificate authority root expires at midnight UTC.",
        "River silt renewed these fields every spring for millennia.",
        "What do we owe to people we will never meet?",
        "Grain shipments resumed after the canal locks reopened.",
        "Thunder rolled twice, then the valley held its breath.",
        "Confirm lockout-tagout before servicing the conveyor motor.",
    ]
}

fn coda_input() -> &'static str {
    "the system hums quietly in the background"
}

fn base_config(seed: &str) -> CognitiveLoopConfig {
    let mut c = CognitiveLoopConfig::default();
    c.genesis_phrase = Some(seed.to_string());
    c.async_training = false;
    c
}

fn arm_config(arm: &str, seed: &str) -> CognitiveLoopConfig {
    let mut c = base_config(seed);
    let null13 = |c: &mut CognitiveLoopConfig| {
        c.enable_gwt = false;
        c.enable_prefrontal = false;
        c.enable_surprise_exploration = false;
        c.enable_consciousness_thermodynamics = false;
        c.enable_hierarchical_free_energy = false;
        c.enable_phi_attention = false;
        c.enable_predictive_processing = false;
        c.enable_dream_replay = false;
        c.enable_quantum_coherence = false;
        c.enable_resonance = false;
        c.enable_narrative_self = false;
        c.enable_temporal_consciousness = false;
        c.enable_phenomenal_binding = false;
    };
    match arm {
        "full" => {}
        "min2" => null13(&mut c),
        "off15" => {
            null13(&mut c);
            c.enable_meta_cognition = false;
            c.enable_embodied_cognition = false;
        }
        // Phase 2 bisection of the 13 (protocol amendment 2026-07-17):
        // bisectA = first 7 of the null-13 off; bisectB = the other 6 off.
        "bisectA" => {
            c.enable_gwt = false;
            c.enable_prefrontal = false;
            c.enable_surprise_exploration = false;
            c.enable_consciousness_thermodynamics = false;
            c.enable_hierarchical_free_energy = false;
            c.enable_phi_attention = false;
            c.enable_predictive_processing = false;
        }
        "bisectB" => {
            c.enable_dream_replay = false;
            c.enable_quantum_coherence = false;
            c.enable_resonance = false;
            c.enable_narrative_self = false;
            c.enable_temporal_consciousness = false;
            c.enable_phenomenal_binding = false;
        }
        // Round 2: split the B-6 (round 1 localized the learning carrier there)
        "bisectB1" => {
            c.enable_dream_replay = false;
            c.enable_quantum_coherence = false;
            c.enable_resonance = false;
        }
        "bisectB2" => {
            c.enable_narrative_self = false;
            c.enable_temporal_consciousness = false;
            c.enable_phenomenal_binding = false;
        }
        // Round 3: single-subsystem-off arms for the B2 finalists
        "no_dream" => c.enable_dream_replay = false,
        "no_temporal" => c.enable_temporal_consciousness = false,
        "no_narrative" => c.enable_narrative_self = false,
        "no_binding" => c.enable_phenomenal_binding = false,
        // Engine kill-switch arm (P5 follow-up): ablates the measurement
        // spine itself — the 15 enable_* flags never covered it.
        "no_engine" => c.enable_consciousness_engine = false,
        other => panic!("unknown arm {other}"),
    }
    c
}

struct ArmResult {
    arm: String,
    seed_idx: usize,
    pe_early: f64,
    pe_late: f64,
    learning_delta: f64,
    pe_novel: f64,
    surprise_contrast: f64,
    cl_body: f64,
    cl_coda: f64,
    regime_separation: f64,
    mean_cycle_us: f64,
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Phase 3: order-anticipation probe (protocol amendment 2026-07-17).
///
/// 60 learning reps, then 10 probe reps: odd probe reps swap one
/// deterministic position pair (p, q); even reps are clean controls.
/// Endpoint: order_sensitivity = mean PE at swapped slots − mean PE at the
/// same slots in clean reps. Both slot contents are fully familiar — only
/// ORDER distinguishes them, so a positive value is evidence of genuine
/// sequence anticipation rather than familiarity adaptation.
fn run_order_arm(arm: &str, seed_idx: usize) {
    let seed = SEEDS[seed_idx];
    let mut svc = CognitiveLoopService::new(arm_config(arm, seed)).expect("construct");
    let script = learned_script();
    let n = script.len();

    for _ in 0..REPS {
        for s in script.iter() {
            let _ = svc.cycle(s);
        }
    }

    const PROBE_REPS: usize = 10;
    let mut swapped_pes: Vec<f64> = Vec::new();
    // clean-rep PE per position, for the control means
    let mut clean_pes_at: Vec<Vec<f64>> = vec![Vec::new(); n];
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
            let pe = res.prediction_error as f64;
            if swapped_rep && (pos == p || pos == q) {
                swapped_pes.push(pe);
            } else if !swapped_rep {
                clean_pes_at[pos].push(pe);
            }
        }
    }

    let control_pes: Vec<f64> = probe_positions
        .iter()
        .flat_map(|&(p, q)| {
            clean_pes_at[p]
                .iter()
                .chain(clean_pes_at[q].iter())
                .copied()
                .collect::<Vec<_>>()
        })
        .collect();
    let sensitivity = mean(&swapped_pes) - mean(&control_pes);
    println!(
        "{:11} seed{}  PE swapped {:.4} vs in-order {:.4} → order_sensitivity {:+.4}",
        arm,
        seed_idx,
        mean(&swapped_pes),
        mean(&control_pes),
        sensitivity,
    );
}

fn run_arm(arm: &str, seed_idx: usize) -> ArmResult {
    let seed = SEEDS[seed_idx];
    let mut svc = CognitiveLoopService::new(arm_config(arm, seed)).expect("construct");
    let script = learned_script();
    let n = script.len();

    let mut pes: Vec<f64> = Vec::with_capacity(REPS * n);
    let mut cls_body: Vec<f64> = Vec::new();
    let mut cycle_us: Vec<f64> = Vec::new();

    // Task 1: predictive learning on the repeated sequence
    for rep in 0..REPS {
        for s in script.iter() {
            let t = Instant::now();
            let r = svc.cycle(s);
            cycle_us.push(t.elapsed().as_micros() as f64);
            pes.push(r.prediction_error as f64);
            // "body" CL sample from the last 20 repetitions (varied, steady state)
            if rep >= REPS - 20 {
                let cl = r.metadata.consciousness.consciousness_level;
                if cl > 0.0 {
                    cls_body.push(cl);
                }
            }
        }
    }
    // Early window: repetitions 3-4 (skip cold start); late: last 2 repetitions
    let pe_early = mean(&pes[2 * n..4 * n]);
    let pe_late = mean(&pes[(REPS - 2) * n..]);

    // Task 2: surprise contrast on novel input
    let mut pes_novel: Vec<f64> = Vec::new();
    for s in novel_script() {
        let t = Instant::now();
        let r = svc.cycle(s);
        cycle_us.push(t.elapsed().as_micros() as f64);
        pes_novel.push(r.prediction_error as f64);
    }
    let pe_novel = mean(&pes_novel);

    // Task 3 (manipulation check): repetitive coda, CL separation vs body
    let mut cls_coda: Vec<f64> = Vec::new();
    for _ in 0..60 {
        let t = Instant::now();
        let r = svc.cycle(coda_input());
        cycle_us.push(t.elapsed().as_micros() as f64);
        let cl = r.metadata.consciousness.consciousness_level;
        if cl > 0.0 {
            cls_coda.push(cl);
        }
    }

    let cl_body = mean(&cls_body);
    let cl_coda = mean(&cls_coda);
    let res = ArmResult {
        arm: arm.to_string(),
        seed_idx,
        pe_early,
        pe_late,
        learning_delta: pe_early - pe_late,
        pe_novel,
        surprise_contrast: pe_novel - pe_late,
        cl_body,
        cl_coda,
        regime_separation: cl_body - cl_coda,
        mean_cycle_us: mean(&cycle_us),
    };
    println!(
        "{:6} seed{}  PE early {:.4} → late {:.4} (Δlearn {:+.4}) | novel {:.4} (surprise {:+.4}) | CL body {:.3} coda {:.3} (sep {:+.3}) | {:.0} µs/cyc",
        res.arm,
        res.seed_idx,
        res.pe_early,
        res.pe_late,
        res.learning_delta,
        res.pe_novel,
        res.surprise_contrast,
        res.cl_body,
        res.cl_coda,
        res.regime_separation,
        res.mean_cycle_us,
    );
    res
}

fn main() {
    // Modes (protocol Phase 2 amendment):
    //   --phase1     original 3 arms x seeds 0-2
    //   --replicate  full + off15 across ALL 10 seeds, sign-test summary
    //   --bisect     bisectA + bisectB on seeds 0-2
    let mode = std::env::args().nth(1).unwrap_or_default();
    println!("=== KEYSTONE A/B (protocol: docs/KEYSTONE_AB_PROTOCOL_2026-07-17.md) ===");

    let (arms, seed_count): (Vec<&str>, usize) = match mode.as_str() {
        "--phase1" => (vec!["full", "min2", "off15"], 3),
        "--bisect" => (vec!["bisectA", "bisectB"], 3),
        "--bisect2" => (vec!["bisectB1", "bisectB2"], 2),
        "--bisect3" => (vec!["no_narrative", "no_temporal", "no_binding"], 2),
        // Phase 3: order-anticipation probes (separate flow, returns early)
        "--order" => {
            println!(
                "mode --order | arms [full, off15, no_temporal] | seeds 3 | 60 learn reps + 10 probe reps\n"
            );
            for seed_idx in 0..3 {
                for arm in ["full", "off15", "no_temporal"] {
                    run_order_arm(arm, seed_idx);
                }
                println!();
            }
            println!("Interpretation gates: protocol Phase 3 amendment (Q1-Q3).");
            return;
        }
        _ => (vec!["full", "off15"], SEEDS.len()), // --replicate is the default
    };
    println!(
        "mode {} | arms {:?} | seeds {} | learned {}x{} + novel 24 + coda 60\n",
        if mode.is_empty() {
            "--replicate"
        } else {
            &mode
        },
        arms,
        seed_count,
        REPS,
        learned_script().len()
    );

    let mut results: Vec<ArmResult> = Vec::new();
    for seed_idx in 0..seed_count {
        for arm in &arms {
            results.push(run_arm(arm, seed_idx));
        }
        println!();
    }

    println!("=== AGGREGATE (mean over {seed_count} seeds) ===");
    println!(
        "{:8} {:>10} {:>10} {:>12} {:>10} {:>10}",
        "arm", "Δlearn", "surprise", "regime-sep", "µs/cyc", "PE late"
    );
    for arm in &arms {
        let rows: Vec<&ArmResult> = results.iter().filter(|r| &r.arm == arm).collect();
        let agg = |f: fn(&ArmResult) -> f64| mean(&rows.iter().map(|r| f(r)).collect::<Vec<_>>());
        println!(
            "{:8} {:>10.4} {:>10.4} {:>12.4} {:>10.0} {:>10.4}",
            arm,
            agg(|r| r.learning_delta),
            agg(|r| r.surprise_contrast),
            agg(|r| r.regime_separation),
            agg(|r| r.mean_cycle_us),
            agg(|r| r.pe_late),
        );
    }

    // Sign-test summary for the pre-registered replication endpoint
    if arms.contains(&"full") && arms.contains(&"off15") {
        let mut positive = 0usize;
        let mut total = 0usize;
        println!("\n=== SIGN TEST: Δlearn(full) − Δlearn(off15) per seed ===");
        for seed_idx in 0..seed_count {
            let f = results
                .iter()
                .find(|r| r.arm == "full" && r.seed_idx == seed_idx);
            let o = results
                .iter()
                .find(|r| r.arm == "off15" && r.seed_idx == seed_idx);
            if let (Some(f), Some(o)) = (f, o) {
                let d = f.learning_delta - o.learning_delta;
                println!("  seed{seed_idx}: {d:+.4}");
                total += 1;
                if d > 0.0 {
                    positive += 1;
                }
            }
        }
        println!(
            "  positive in {positive}/{total} seeds — pre-registered gates: ≥9/10 CONFIRMED, 7-8/10 suggestive, ≤6/10 not supported"
        );
    }
    println!("\nInterpretation gates are pre-registered in the protocol doc.");
}
