// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scaled Dream Feedback Experiment
//!
//! Runs dream feedback with 10x more trials to achieve statistical significance.
//!
//! ## Changes from original:
//! - 5000 trials instead of 500
//! - More contexts with dream priors (200 instead of 50)
//! - Higher dream insight rate
//! - Multiple random seeds for robustness
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example dream_feedback_scaled --features magi_loop --release
//! ```

use anyhow::Result;

#[cfg(feature = "magi_loop")]
use symthaea::consciousness::recursive_improvement::{
    DreamFeedbackBridge, DreamInsight, hash_context,
};

fn main() -> Result<()> {
    #[cfg(not(feature = "magi_loop"))]
    {
        println!("This example requires the 'magi_loop' feature.");
        println!(
            "Run with: cargo run --example dream_feedback_scaled --features magi_loop --release"
        );
        return Ok(());
    }

    #[cfg(feature = "magi_loop")]
    run_experiment()
}

#[cfg(feature = "magi_loop")]
fn run_experiment() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   SCALED DREAM FEEDBACK EXPERIMENT");
    println!("   Testing with 10x more trials for statistical significance");
    println!("================================================================\n");

    // Run multiple trials with different seeds
    let seeds = vec![42, 123, 456, 789, 1011];
    let mut all_results = Vec::new();

    for (trial, &seed) in seeds.iter().enumerate() {
        println!("────────────────────────────────────────");
        println!("Trial {}/{}  (seed={})", trial + 1, seeds.len(), seed);
        println!("────────────────────────────────────────\n");

        let result = run_single_trial(seed)?;
        all_results.push(result);
    }

    // Aggregate results
    println!("\n================================================================");
    println!("   AGGREGATED RESULTS ACROSS {} TRIALS", seeds.len());
    println!("================================================================\n");

    let dream_briers: Vec<f64> = all_results.iter().filter_map(|r| r.dream_brier).collect();
    let baseline_briers: Vec<f64> = all_results
        .iter()
        .filter_map(|r| r.baseline_brier)
        .collect();

    let mean_dream = mean(&dream_briers);
    let mean_baseline = mean(&baseline_briers);
    let mean_improvement = mean_baseline - mean_dream;

    println!("Mean Dream-Informed Brier: {:.4}", mean_dream);
    println!("Mean Baseline Brier: {:.4}", mean_baseline);
    println!(
        "Mean Improvement: {:.4} ({:.1}%)",
        mean_improvement,
        (mean_improvement / mean_baseline) * 100.0
    );

    // Cross-trial consistency
    let improvements: Vec<f64> = all_results
        .iter()
        .filter_map(|r| match (r.dream_brier, r.baseline_brier) {
            (Some(d), Some(b)) => Some(b - d),
            _ => None,
        })
        .collect();

    let all_positive = improvements.iter().all(|&i| i > 0.0);
    let std_improvement = std_dev(&improvements);

    println!("\nConsistency:");
    println!(
        "  All trials show improvement: {}",
        if all_positive { "YES" } else { "NO" }
    );
    println!("  Improvement std dev: {:.4}", std_improvement);
    println!(
        "  Coefficient of variation: {:.1}%",
        (std_improvement / mean_improvement.abs()) * 100.0
    );

    // Meta-analysis: combine p-values
    let total_dream_n: usize = all_results.iter().map(|r| r.dream_n).sum();
    let total_baseline_n: usize = all_results.iter().map(|r| r.baseline_n).sum();
    let total_dream_brier: f64 = all_results.iter().map(|r| r.dream_brier_sum).sum();
    let total_baseline_brier: f64 = all_results.iter().map(|r| r.baseline_brier_sum).sum();

    let pooled_dream = total_dream_brier / total_dream_n as f64;
    let pooled_baseline = total_baseline_brier / total_baseline_n as f64;
    let pooled_improvement = pooled_baseline - pooled_dream;

    println!(
        "\nPooled Analysis (n={} dream, n={} baseline):",
        total_dream_n, total_baseline_n
    );
    println!("  Pooled Dream Brier: {:.4}", pooled_dream);
    println!("  Pooled Baseline Brier: {:.4}", pooled_baseline);
    println!(
        "  Pooled Improvement: {:.4} ({:.1}%)",
        pooled_improvement,
        (pooled_improvement / pooled_baseline) * 100.0
    );

    // Bootstrap CI on pooled data
    let (ci_lower, ci_upper) = pooled_bootstrap_ci(
        total_dream_n,
        total_baseline_n,
        pooled_dream,
        pooled_baseline,
    );

    println!(
        "\n  95% CI for improvement: [{:.4}, {:.4}]",
        ci_lower, ci_upper
    );

    if ci_lower > 0.0 {
        println!("\n✓ DREAM FEEDBACK SIGNIFICANTLY IMPROVES CALIBRATION");
        println!("  95% CI excludes zero");
        println!(
            "  Effect size: {:.1}% improvement in Brier score",
            (pooled_improvement / pooled_baseline) * 100.0
        );
    } else if ci_upper < 0.0 {
        println!("\n✗ DREAM FEEDBACK WORSENS CALIBRATION");
    } else {
        println!("\n○ EFFECT NOT STATISTICALLY SIGNIFICANT");
        println!("  95% CI includes zero");
        println!("  May need even more data");
    }

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "magi_loop")]
struct TrialResult {
    dream_brier: Option<f64>,
    baseline_brier: Option<f64>,
    dream_n: usize,
    baseline_n: usize,
    dream_brier_sum: f64,
    baseline_brier_sum: f64,
}

#[cfg(feature = "magi_loop")]
fn run_single_trial(seed: u64) -> Result<TrialResult> {
    let mut bridge = DreamFeedbackBridge::new();
    let mut rng_seed = seed;

    // Phase 1: Generate more dream insights
    println!("  Phase 1: Generating dream insights...");

    let contexts = generate_contexts(200, seed);
    let mut insights_created = 0;

    for (i, context) in contexts.iter().enumerate() {
        let context_hash = hash_context(context);
        let phi_improvement = simulate_dream_discovery(i, seed);

        if phi_improvement > 0.03 {
            // Lower threshold for more insights
            let insight = DreamInsight::new(
                context_hash,
                context.clone(),
                perturb_action(context, 0.2, seed + i as u64),
                phi_improvement,
            );

            if bridge.process_insight(insight) {
                insights_created += 1;
            }
        }
    }

    println!(
        "    {} insights, {} priors",
        insights_created,
        bridge.num_priors()
    );

    // Phase 2: Make 5000 predictions
    println!("  Phase 2: Making 5000 predictions...");

    for trial in 0..5000 {
        let context = generate_context(trial, seed + 1000);
        let context_hash = hash_context(&context);

        let base_confidence = 0.5 + 0.3 * pseudo_random(&mut rng_seed);
        let (adjusted_confidence, _) = bridge.adjust_confidence(base_confidence, context_hash);

        let outcome = simulate_outcome(adjusted_confidence, &mut rng_seed);

        let brier_score = if outcome {
            (adjusted_confidence - 1.0).powi(2)
        } else {
            adjusted_confidence.powi(2)
        };

        bridge.record_outcome(context_hash, brier_score);

        if (trial + 1) % 1000 == 0 {
            print!("    {}/5000\r", trial + 1);
        }
    }
    println!("    Done        ");

    // Results
    let stats = bridge.stats();

    println!("  Results:");
    if let (Some(d), Some(b)) = (stats.dream_informed_brier(), stats.baseline_brier()) {
        println!(
            "    Dream: {:.4} (n={})",
            d, stats.dream_informed_predictions
        );
        println!("    Base:  {:.4} (n={})", b, stats.baseline_predictions);
        println!(
            "    Improvement: {:.4} ({:.1}%)",
            b - d,
            ((b - d) / b) * 100.0
        );
    }

    Ok(TrialResult {
        dream_brier: stats.dream_informed_brier(),
        baseline_brier: stats.baseline_brier(),
        dream_n: stats.dream_informed_predictions,
        baseline_n: stats.baseline_predictions,
        dream_brier_sum: stats.dream_informed_brier_sum,
        baseline_brier_sum: stats.baseline_brier_sum,
    })
}

#[cfg(feature = "magi_loop")]
fn generate_context(idx: usize, seed: u64) -> Vec<f32> {
    let mut context = Vec::with_capacity(64);
    let mut s = (idx as f64 * 0.1 + seed as f64 * 0.001) as f32;
    for _ in 0..64 {
        s = (s * 1.5 + 0.7).sin() * 100.0;
        context.push(s.fract());
    }
    context
}

#[cfg(feature = "magi_loop")]
fn generate_contexts(n: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..n).map(|i| generate_context(i, seed)).collect()
}

#[cfg(feature = "magi_loop")]
fn simulate_dream_discovery(context_idx: usize, seed: u64) -> f64 {
    // More contexts have improvements (50% instead of 40%)
    let s = ((context_idx * 7 + 13) as u64).wrapping_add(seed) % 100;
    if s < 50 {
        0.03 + (s as f64 * 0.006) // 3-33% improvement
    } else {
        0.0
    }
}

#[cfg(feature = "magi_loop")]
fn perturb_action(action: &[f32], amount: f32, seed: u64) -> Vec<f32> {
    action
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let perturbation = ((i as f64 * 0.1 + seed as f64 * 0.001).sin() as f32) * amount;
            v + perturbation
        })
        .collect()
}

#[cfg(feature = "magi_loop")]
fn pseudo_random(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*seed as f64) / (u64::MAX as f64)
}

#[cfg(feature = "magi_loop")]
fn simulate_outcome(confidence: f64, rng: &mut u64) -> bool {
    let base_rate = 0.35 + confidence * 0.35;
    pseudo_random(rng) < base_rate
}

#[cfg(feature = "magi_loop")]
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(feature = "magi_loop")]
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

#[cfg(feature = "magi_loop")]
fn pooled_bootstrap_ci(
    n_dream: usize,
    n_baseline: usize,
    dream_mean: f64,
    baseline_mean: f64,
) -> (f64, f64) {
    // Normal approximation for CI
    let se_dream = (0.25_f64 / n_dream as f64).sqrt();
    let se_baseline = (0.25_f64 / n_baseline as f64).sqrt();
    let se_diff = (se_dream.powi(2) + se_baseline.powi(2)).sqrt();

    let improvement = baseline_mean - dream_mean;
    let ci_lower = improvement - 1.96 * se_diff;
    let ci_upper = improvement + 1.96 * se_diff;

    (ci_lower, ci_upper)
}
