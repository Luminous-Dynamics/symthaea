// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dream Feedback Loop Experiment
//!
//! Tests whether closing the dream-to-behavior feedback loop improves
//! prediction calibration compared to a control system.
//!
//! ## Hypothesis (H3 from research plan)
//!
//! Closing the dream-to-behavior feedback loop (connecting counterfactual
//! insights to MAGI Loop priors) improves calibration and reduces
//! prediction error compared to a control system.
//!
//! ## Experimental Design
//!
//! 1. Generate synthetic prediction tasks with known outcomes
//! 2. Run predictions with and without dream feedback
//! 3. Compare Brier scores between conditions
//! 4. Statistical analysis (Mann-Whitney U test)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example dream_feedback_experiment --release
//! ```

use anyhow::Result;
use std::collections::HashMap;

// Import dream feedback types
use symthaea::consciousness::recursive_improvement::dream_feedback::{
    DreamFeedbackBridge, DreamInsight,
};

fn main() -> Result<()> {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   DREAM FEEDBACK LOOP EXPERIMENT                             ║");
    println!("║   Testing H3: Dream Learning Improves Calibration            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let n_contexts = 50;
    let n_cycles = 500;
    let dream_frequency = 0.3;
    let true_prior_strength = 0.15;

    println!("Configuration:");
    println!("  Contexts: {}", n_contexts);
    println!("  Cycles: {}", n_cycles);
    println!("  Dream frequency: {:.0}%", dream_frequency * 100.0);
    println!();

    // Run control condition
    println!("════════════════════════════════════════════════════════════════");
    println!("   CONDITION 1: CONTROL (No Dream Feedback)");
    println!("════════════════════════════════════════════════════════════════\n");

    let control = run_simulation(n_contexts, n_cycles, 0.0, true_prior_strength, false);
    println!(
        "Control: Mean Brier = {:.4} ± {:.4}",
        control.mean_brier, control.brier_std
    );

    // Run treatment condition
    println!("\n════════════════════════════════════════════════════════════════");
    println!("   CONDITION 2: TREATMENT (With Dream Feedback)");
    println!("════════════════════════════════════════════════════════════════\n");

    let treatment = run_simulation(
        n_contexts,
        n_cycles,
        dream_frequency,
        true_prior_strength,
        true,
    );
    println!(
        "Treatment: Mean Brier = {:.4} ± {:.4}",
        treatment.mean_brier, treatment.brier_std
    );
    println!(
        "Dreams: {}, Priors learned: {}",
        treatment.dreams_generated, treatment.priors_learned
    );

    // Statistical analysis
    println!("\n════════════════════════════════════════════════════════════════");
    println!("   RESULTS");
    println!("════════════════════════════════════════════════════════════════\n");

    let improvement = control.mean_brier - treatment.mean_brier;
    let improvement_pct = (improvement / control.mean_brier) * 100.0;

    let pooled_std = ((control.brier_std.powi(2) + treatment.brier_std.powi(2)) / 2.0).sqrt();
    let cohens_d = if pooled_std > 0.0 {
        improvement / pooled_std
    } else {
        0.0
    };

    let (_, p_value) = mann_whitney_u(&control.brier_scores, &treatment.brier_scores);

    println!(
        "Brier Score Improvement: {:+.4} ({:+.1}%)",
        improvement, improvement_pct
    );
    println!("Cohen's d: {:+.3}", cohens_d);
    println!(
        "p-value: {:.4} {}",
        p_value,
        if p_value < 0.05 { "*" } else { "" }
    );

    println!("\n════════════════════════════════════════════════════════════════");
    if improvement > 0.0 && p_value < 0.05 {
        println!("✓ H3 SUPPORTED: Dream feedback improves calibration");
    } else if improvement > 0.0 {
        println!("◐ WEAK SUPPORT: Direction correct but not significant");
    } else {
        println!("✗ H3 NOT SUPPORTED: No improvement from dream feedback");
    }
    println!("════════════════════════════════════════════════════════════════\n");

    Ok(())
}

struct SimulationResults {
    mean_brier: f64,
    brier_std: f64,
    brier_scores: Vec<f64>,
    dreams_generated: usize,
    priors_learned: usize,
}

fn run_simulation(
    n_contexts: usize,
    n_cycles: usize,
    dream_frequency: f64,
    true_prior_strength: f64,
    apply_priors: bool,
) -> SimulationResults {
    let mut bridge = DreamFeedbackBridge::new();
    let mut rng_seed: u64 = 12345;
    let mut brier_scores = Vec::new();
    let mut dreams_generated = 0;
    let mut priors_learned = 0;

    let mut context_difficulty: HashMap<u64, f64> = HashMap::new();
    for i in 0..n_contexts {
        context_difficulty.insert(i as u64, pseudo_random(&mut rng_seed) * 0.5 + 0.25);
    }

    for _ in 0..n_cycles {
        let context_hash = (pseudo_random(&mut rng_seed) * n_contexts as f64) as u64;
        let true_success_prob = context_difficulty
            .get(&context_hash)
            .copied()
            .unwrap_or(0.5);

        let base_confidence = 0.5 + (pseudo_random(&mut rng_seed) - 0.5) * 0.3;

        let (adjusted_confidence, was_dream_informed) = if apply_priors {
            bridge.adjust_confidence(base_confidence, context_hash)
        } else {
            (base_confidence, false)
        };

        let final_confidence = if was_dream_informed {
            if let Some(p) = bridge.get_prior(context_hash) {
                (adjusted_confidence + p.strength * true_prior_strength).clamp(0.01, 0.99)
            } else {
                adjusted_confidence.clamp(0.01, 0.99)
            }
        } else {
            adjusted_confidence.clamp(0.01, 0.99)
        };

        let outcome = pseudo_random(&mut rng_seed) < true_success_prob;
        let brier = (final_confidence - if outcome { 1.0 } else { 0.0 }).powi(2);
        brier_scores.push(brier);
        bridge.record_outcome(context_hash, brier);

        if pseudo_random(&mut rng_seed) < dream_frequency {
            let phi_improvement = if outcome {
                pseudo_random(&mut rng_seed) * 0.1
            } else {
                pseudo_random(&mut rng_seed) * 0.3
            };

            let insight =
                DreamInsight::new(context_hash, vec![0.0; 10], vec![1.0; 10], phi_improvement);

            if bridge.process_insight(insight) {
                dreams_generated += 1;
                priors_learned += 1;
            }
        }
    }

    let mean_brier = brier_scores.iter().sum::<f64>() / brier_scores.len() as f64;
    let variance = brier_scores
        .iter()
        .map(|b| (b - mean_brier).powi(2))
        .sum::<f64>()
        / brier_scores.len() as f64;

    SimulationResults {
        mean_brier,
        brier_std: variance.sqrt(),
        brier_scores,
        dreams_generated,
        priors_learned,
    }
}

fn pseudo_random(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*seed as f64) / (u64::MAX as f64)
}

fn mann_whitney_u(a: &[f64], b: &[f64]) -> (f64, f64) {
    let n1 = a.len();
    let n2 = b.len();

    let mut combined: Vec<(f64, bool)> = a.iter().map(|&x| (x, true)).collect();
    combined.extend(b.iter().map(|&x| (x, false)));
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut ranks: Vec<(f64, bool)> = Vec::new();
    let mut i = 0;
    while i < combined.len() {
        let mut j = i;
        while j < combined.len() && combined[j].0 == combined[i].0 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            ranks.push((avg_rank, combined[k].1));
        }
        i = j;
    }

    let r1: f64 = ranks.iter().filter(|(_, is_a)| *is_a).map(|(r, _)| r).sum();
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u = u1.min((n1 * n2) as f64 - u1);

    let mu = (n1 * n2) as f64 / 2.0;
    let sigma = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();
    let z = (u - mu) / sigma;
    let p = 2.0 * (1.0 - normal_cdf(z.abs()));

    (u, p)
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp());
    if x < 0.0 { -y } else { y }
}
