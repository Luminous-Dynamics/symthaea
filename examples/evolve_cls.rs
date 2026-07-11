// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CLS-in-the-Loop Threshold Evolution
//!
//! Evolves cognitive thresholds by evaluating each organism inside a REAL
//! CognitiveLoopService — not a lightweight CfC proxy. This is expensive
//! (~1 sec/cycle) but is the only way to optimize thresholds that actually
//! improve the full consciousness pipeline.
//!
//! Each organism:
//! 1. Decodes ThresholdPhenotype from its genome (bits 400-634)
//! 2. Creates a fresh CognitiveLoopService
//! 3. Applies overrides via threshold_overrides_mut()
//! 4. Runs N cognitive cycles with natural language input
//! 5. Measures 5 objectives: Phi, FE-reduction, pred-acc, stability, consistency
//!
//! Default: 10 pop × 8 r#gen × 30 cycles = 2,400 CLS cycles (~40 min)
//! Scale up with env vars: POP_SIZE, GENERATIONS, EVAL_CYCLES
//!
//! # Promotion path (Tier 1.2, `DISCOVERY_AND_SELF_IMPROVEMENT_PLAN_2026-07-06.md`)
//!
//! This binary is step 1 of 3 in getting an evolved phenotype into the live
//! system, mirroring the Broca curriculum bridge's shape:
//!
//! 1. **This binary** writes the winning phenotype to
//!    `<candidate-dir>/candidate-phenotype.json` + a provenance sidecar at
//!    `<candidate-dir>/provenance.json`. It NEVER writes to any path the live
//!    system reads.
//! 2. `cargo run --release --features neuroevolution --example cls_promotion_gate -- <candidate-dir>`
//!    re-evaluates the candidate on fresh seeds (never the ones used here) and,
//!    if it clears the recorded fitness within tolerance, writes
//!    `<candidate-dir>/PROMOTION_READY.json`.
//! 3. `scripts/cls_promote_candidate.sh <candidate-dir> --i-understand-this-is-live`
//!    (human-run only) copies the candidate to the path
//!    `SYMTHAEA_THRESHOLD_OVERRIDES_PATH` points at.
//!
//! Candidate directory: `CLS_CANDIDATE_DIR` env var, default
//! `target/cls-evolution/candidates/<UTC-timestamp>`.
//!
//! Usage: cargo run --release --features neuroevolution --example evolve_cls

#[cfg(not(feature = "neuroevolution"))]
fn main() {
    eprintln!("Requires `neuroevolution` feature.");
}

#[cfg(feature = "neuroevolution")]
fn main() {
    use symthaea::cognitive_loop::cls_evolution_harness::{
        CandidateProvenance, ClsFitness, EVOLUTION_INPUTS, current_git_sha, evaluate_with_cls,
    };
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_neuroevolution::{
        NeuralGenome, ThresholdPhenotype,
        threshold_genome::{decode_thresholds, evaluate_threshold_fitness},
    };

    let pop_size: usize = std::env::var("POP_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let generations: usize = std::env::var("GENERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let eval_cycles: usize = std::env::var("EVAL_CYCLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let total_cls_cycles = pop_size * generations * eval_cycles;
    let est_minutes = total_cls_cycles / 60; // ~1 sec/cycle

    println!("═══════════════════════════════════════════════════════════════");
    println!("  CLS-in-the-Loop Threshold Evolution");
    println!(
        "  {} pop × {} r#gen × {} cycles = {} total (~{} min)",
        pop_size, generations, eval_cycles, total_cls_cycles, est_minutes
    );
    println!("  Override: POP_SIZE, GENERATIONS, EVAL_CYCLES env vars");
    println!("═══════════════════════════════════════════════════════════════\n");

    let inputs: &[&str] = EVOLUTION_INPUTS;
    let genesis_seed_phrase = "cls-in-the-loop-evolution";
    let genesis = GenesisSeed::from_phrase(genesis_seed_phrase);

    // Initialize population
    let mut population: Vec<(NeuralGenome, ClsFitness)> = (0..pop_size)
        .map(|i| {
            let genome = NeuralGenome::from_genesis(&genesis, &format!("org-{i}"));
            (genome, ClsFitness::default())
        })
        .collect();

    let defaults = ThresholdPhenotype::default();
    let default_fitness = evaluate_with_cls(&defaults, inputs, eval_cycles);
    println!(
        "  Default baseline: Φ={:.4}  FE-red={:.4}  pred={:.4}  stab={:.4}\n",
        default_fitness.mean_phi,
        default_fitness.fe_reduction,
        default_fitness.pred_accuracy,
        default_fitness.phi_stability
    );

    println!(
        "{:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "Gen", "Best Φ", "Mean Φ", "FE-red", "Pred", "Stab", "Thresh"
    );
    println!("{}", "-".repeat(62));

    let mut rng_state: u64 = 42;

    for r#gen in 0..generations {
        // Evaluate each organism via CLS
        for (genome, fitness) in population.iter_mut() {
            let phenotype = decode_thresholds(&genome.hv);
            *fitness = evaluate_with_cls(&phenotype, inputs, eval_cycles);
            fitness.threshold_consistency = evaluate_threshold_fitness(&phenotype);
        }

        // Sort by composite
        population.sort_by(|a, b| {
            b.1.composite()
                .partial_cmp(&a.1.composite())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = &population[0].1;
        let mean_phi: f64 = population.iter().map(|p| p.1.mean_phi).sum::<f64>() / pop_size as f64;

        println!(
            "{:>4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
            r#gen,
            best.mean_phi,
            mean_phi,
            best.fe_reduction,
            best.pred_accuracy,
            best.phi_stability,
            best.threshold_consistency
        );

        // Elitism: keep top 30%
        let elite_count = (pop_size * 3 / 10).max(2);
        let mut next_gen: Vec<(NeuralGenome, ClsFitness)> =
            population[..elite_count].iter().cloned().collect();

        // Fill with mutated offspring from tournament selection
        while next_gen.len() < pop_size {
            // Tournament select (size 3)
            rng_state = xorshift(rng_state);
            let i1 = rng_state as usize % population.len();
            rng_state = xorshift(rng_state);
            let i2 = rng_state as usize % population.len();
            rng_state = xorshift(rng_state);
            let i3 = rng_state as usize % population.len();
            let parent_idx = [i1, i2, i3]
                .into_iter()
                .max_by(|&a, &b| {
                    population[a]
                        .1
                        .composite()
                        .partial_cmp(&population[b].1.composite())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            // Mutate threshold region only (bytes 50-80)
            let mut child = population[parent_idx].0.clone();
            rng_state = xorshift(rng_state);
            for byte_idx in 50..80 {
                rng_state = xorshift(rng_state);
                if (rng_state >> 40) as f64 / (1u64 << 24) as f64 > 0.99 {
                    child.hv.0[byte_idx] ^= rng_state as u8;
                }
            }
            next_gen.push((child, ClsFitness::default()));
        }

        population = next_gen;
    }

    // Final evaluation of best
    let best_pheno = decode_thresholds(&population[0].0.hv);
    let mut best_fitness = evaluate_with_cls(&best_pheno, inputs, eval_cycles);
    best_fitness.threshold_consistency = evaluate_threshold_fitness(&best_pheno);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Final Best vs Defaults");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!(
        "  {:>20}  {:>10}  {:>10}  {:>8}",
        "Metric", "Default", "Evolved", "Delta"
    );
    println!("  {}", "-".repeat(52));
    cmp_row("Mean Φ", default_fitness.mean_phi, best_fitness.mean_phi);
    cmp_row(
        "FE reduction",
        default_fitness.fe_reduction,
        best_fitness.fe_reduction,
    );
    cmp_row(
        "Pred accuracy",
        default_fitness.pred_accuracy,
        best_fitness.pred_accuracy,
    );
    cmp_row(
        "Phi stability",
        default_fitness.phi_stability,
        best_fitness.phi_stability,
    );

    println!("\n  Evolved thresholds:");
    let d = ThresholdPhenotype::default();
    thresh_row(
        "fep_surprise_scale",
        best_pheno.fep_surprise_scale,
        d.fep_surprise_scale,
    );
    thresh_row("fep_lr_decay", best_pheno.fep_lr_decay, d.fep_lr_decay);
    thresh_row(
        "neuromod_d2_baseline",
        best_pheno.neuromod_d2_baseline as f32,
        d.neuromod_d2_baseline as f32,
    );
    thresh_row(
        "ne_phasic_threshold",
        best_pheno.neuromod_ne_phasic_threshold,
        d.neuromod_ne_phasic_threshold,
    );
    thresh_row(
        "arousal_ema_decay",
        best_pheno.neuromod_arousal_ema_decay,
        d.neuromod_arousal_ema_decay,
    );
    thresh_row(
        "homeostasis_high",
        best_pheno.homeostasis_recalibrate_high,
        d.homeostasis_recalibrate_high,
    );
    thresh_row(
        "homeostasis_low",
        best_pheno.homeostasis_recalibrate_low,
        d.homeostasis_recalibrate_low,
    );
    thresh_row(
        "self_model_weight_high",
        best_pheno.self_model_weight_high,
        d.self_model_weight_high,
    );
    thresh_row(
        "homeostasis_pull_cruise",
        best_pheno.homeostasis_pull_cruise,
        d.homeostasis_pull_cruise,
    );

    // ── Write candidate + provenance (Tier 1.2 promotion path, step 1) ─────
    // Never writes to any path the live system reads — see module doc.
    let candidate_dir = write_candidate(
        &best_pheno,
        &CandidateProvenance {
            created_at_utc: chrono::Utc::now().to_rfc3339(),
            git_sha: current_git_sha(),
            pop_size,
            generations,
            eval_cycles,
            genesis_seed_phrase: genesis_seed_phrase.to_string(),
            evolution_input_count: inputs.len(),
            default_fitness,
            final_fitness: best_fitness,
        },
    );

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Candidate written: {}", candidate_dir.display());
    println!(
        "  Next: cargo run --release --features neuroevolution --example cls_promotion_gate -- {}",
        candidate_dir.display()
    );
    println!("═══════════════════════════════════════════════════════════════");
}

#[cfg(feature = "neuroevolution")]
fn write_candidate(
    phenotype: &symthaea_neuroevolution::ThresholdPhenotype,
    provenance: &symthaea::cognitive_loop::cls_evolution_harness::CandidateProvenance,
) -> std::path::PathBuf {
    let base_dir = std::env::var("CLS_CANDIDATE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
            std::path::PathBuf::from("target/cls-evolution/candidates").join(timestamp)
        });

    std::fs::create_dir_all(&base_dir)
        .unwrap_or_else(|e| panic!("failed to create candidate dir {base_dir:?}: {e}"));

    let phenotype_path = base_dir.join("candidate-phenotype.json");
    std::fs::write(
        &phenotype_path,
        serde_json::to_string_pretty(phenotype).expect("phenotype serializes"),
    )
    .unwrap_or_else(|e| panic!("failed to write {phenotype_path:?}: {e}"));

    let provenance_path = base_dir.join("provenance.json");
    std::fs::write(
        &provenance_path,
        serde_json::to_string_pretty(provenance).expect("provenance serializes"),
    )
    .unwrap_or_else(|e| panic!("failed to write {provenance_path:?}: {e}"));

    base_dir
}

#[cfg(feature = "neuroevolution")]
fn xorshift(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

#[cfg(feature = "neuroevolution")]
fn cmp_row(name: &str, default: f64, evolved: f64) {
    let delta = if default.abs() > 1e-6 {
        (evolved - default) / default * 100.0
    } else {
        0.0
    };
    println!(
        "  {:>20}  {:>10.4}  {:>10.4}  {:>+7.1}%",
        name, default, evolved, delta
    );
}

#[cfg(feature = "neuroevolution")]
fn thresh_row(name: &str, evolved: f32, default: f32) {
    let delta = if default.abs() > 1e-6 {
        ((evolved - default) / default * 100.0) as i32
    } else {
        0
    };
    println!(
        "    {:>25}  {:>8.3}  (default: {:.3}, {:>+4}%)",
        name, evolved, default, delta
    );
}
