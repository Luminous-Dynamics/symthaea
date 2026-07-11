// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CLS Threshold Phenotype Promotion Gate
//!
//! Step 2 of 3 in the Tier 1.2 promotion path
//! (`DISCOVERY_AND_SELF_IMPROVEMENT_PLAN_2026-07-06.md`): re-evaluates a
//! candidate `ThresholdPhenotype` written by `examples/evolve_cls.rs` on
//! FRESH seeds — never the ones used during evolution — using the SAME real
//! `CognitiveLoopService` harness (never a cheap proxy). This guards against
//! a candidate that overfit to the specific evolution seeds rather than
//! genuinely improving the cognitive loop.
//!
//! Cheap-before-expensive: checks the phenotype's internal consistency
//! (`evaluate_threshold_fitness`, no CLS cycles) FIRST and fails fast if it's
//! degenerate, before paying for a full re-evaluation run.
//!
//! Only if fresh fitness clears the recorded (evolution-time) fitness within
//! a tolerance does this write `<candidate-dir>/PROMOTION_READY.json` — the
//! marker `scripts/cls_promote_candidate.sh` requires before it will touch
//! anything. This binary NEVER writes to the path the live system reads.
//!
//! Usage:
//!   cargo run --release --features neuroevolution --example cls_promotion_gate -- <candidate-dir>
//!
//! Env overrides:
//!   CLS_GATE_TOLERANCE   fraction below recorded composite fitness still
//!                        acceptable (default 0.10 = candidate must reach at
//!                        least 90% of its evolution-time composite score on
//!                        fresh seeds)
//!   CLS_GATE_EVAL_CYCLES override eval_cycles from provenance.json

#[cfg(not(feature = "neuroevolution"))]
fn main() {
    eprintln!("Requires `neuroevolution` feature.");
    std::process::exit(2);
}

#[cfg(feature = "neuroevolution")]
fn main() {
    use symthaea::cognitive_loop::cls_evolution_harness::{
        FRESH_INPUTS, PromotionReady, current_git_sha, evaluate_with_cls, load_candidate_phenotype,
        load_provenance,
    };
    use symthaea_neuroevolution::threshold_genome::evaluate_threshold_fitness;

    let candidate_dir = match std::env::args().nth(1) {
        Some(d) => std::path::PathBuf::from(d),
        None => {
            eprintln!("Usage: cls_promotion_gate <candidate-dir>");
            std::process::exit(2);
        }
    };

    let phenotype_path = candidate_dir.join("candidate-phenotype.json");
    let provenance_path = candidate_dir.join("provenance.json");

    println!("[cls-gate] candidate dir: {}", candidate_dir.display());

    let phenotype = match load_candidate_phenotype(&phenotype_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[cls-gate] REFUSED: {e:#}");
            std::process::exit(1);
        }
    };
    let provenance = match load_provenance(&provenance_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[cls-gate] REFUSED: {e:#}");
            std::process::exit(1);
        }
    };

    // ── Cheap check first: internal consistency, no CLS cycles ─────────────
    let consistency = evaluate_threshold_fitness(&phenotype);
    println!("[cls-gate] internal consistency score: {consistency:.4}");
    const MIN_CONSISTENCY: f64 = 0.5;
    if consistency < MIN_CONSISTENCY {
        eprintln!(
            "[cls-gate] FAIL: internal consistency {consistency:.4} < {MIN_CONSISTENCY} — \
             refusing to spend a full CLS re-evaluation on a degenerate phenotype"
        );
        std::process::exit(1);
    }

    // ── Expensive check: full CLS re-evaluation on FRESH, disjoint seeds ───
    let eval_cycles: usize = std::env::var("CLS_GATE_EVAL_CYCLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(provenance.eval_cycles);

    println!(
        "[cls-gate] re-evaluating candidate on {} FRESH seeds ({} cycles) — never the evolution seeds...",
        FRESH_INPUTS.len(),
        eval_cycles
    );

    let mut fresh_fitness = evaluate_with_cls(&phenotype, FRESH_INPUTS, eval_cycles);
    fresh_fitness.threshold_consistency = consistency;

    let recorded_composite = provenance.final_fitness.composite();
    let fresh_composite = fresh_fitness.composite();

    let tolerance: f64 = std::env::var("CLS_GATE_TOLERANCE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.10);

    let required = recorded_composite * (1.0 - tolerance);
    let passed = fresh_composite >= required;

    println!(
        "[cls-gate] recorded (evolution-seed) composite: {recorded_composite:.4}\n\
         [cls-gate] fresh (held-out-seed) composite:      {fresh_composite:.4}\n\
         [cls-gate] required (>= {:.0}% of recorded):      {required:.4}",
        (1.0 - tolerance) * 100.0
    );

    let promotion_ready = PromotionReady {
        candidate_phenotype_path: phenotype_path
            .canonicalize()
            .unwrap_or(phenotype_path.clone())
            .to_string_lossy()
            .to_string(),
        created_at_utc: chrono::Utc::now().to_rfc3339(),
        gate_git_sha: current_git_sha(),
        recorded_fitness: provenance.final_fitness.clone(),
        fresh_fitness: fresh_fitness.clone(),
        fresh_input_count: FRESH_INPUTS.len(),
        eval_cycles,
        tolerance,
        passed,
    };

    if !passed {
        eprintln!(
            "[cls-gate] FAIL: candidate did not clear fresh-seed re-evaluation \
             (overfit to evolution seeds, or genuinely worse) — refusing to write PROMOTION_READY.json"
        );
        std::process::exit(1);
    }

    let ready_path = candidate_dir.join("PROMOTION_READY.json");
    std::fs::write(
        &ready_path,
        serde_json::to_string_pretty(&promotion_ready).expect("PromotionReady serializes"),
    )
    .unwrap_or_else(|e| panic!("failed to write {ready_path:?}: {e}"));

    println!("[cls-gate] PASSED — wrote {}", ready_path.display());
    println!(
        "[cls-gate] to promote: scripts/cls_promote_candidate.sh {} --i-understand-this-is-live",
        candidate_dir.display()
    );
}
