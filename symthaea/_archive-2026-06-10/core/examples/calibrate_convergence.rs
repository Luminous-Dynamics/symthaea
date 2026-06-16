// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Calibration harness for the topological immune system's convergence detection.
//!
//! Generates labeled scenario families (benign-diverse, benign-repetitive,
//! adversarial-converging, adversarial-evasive) and produces a report with
//! ROC curves and recommended thresholds.
//!
//! ```bash
//! cargo run --example calibrate_convergence
//! ```

use symthaea::hdc::moral_topology::{
    CalibrationScenario, MoralAnomalyConfig, calibrate_convergence_threshold,
};
use symthaea_core::hdc::ContinuousHV;

const DIM: usize = 2048;

fn main() {
    println!("=== Topological Immune System: Convergence Calibration ===\n");

    let base_config = MoralAnomalyConfig::default();

    // ── Generate scenario families ──────────────────────────────────────

    let mut scenarios = Vec::new();

    // Family 1: Benign diverse (random topics)
    for family_seed in 0..5 {
        let hvs: Vec<_> = (0..8)
            .map(|i| ContinuousHV::random(DIM, 1000 + family_seed * 100 + i))
            .collect();
        scenarios.push(CalibrationScenario {
            scenarios: hvs,
            is_adversarial: false,
            label: format!("benign_diverse_{family_seed}"),
        });
    }

    // Family 2: Benign repetitive (same topic, no malice)
    for family_seed in 0..3 {
        let base = ContinuousHV::random(DIM, 2000 + family_seed);
        let hvs: Vec<_> = (0..8).map(|_| base.perturb(0.05)).collect();
        scenarios.push(CalibrationScenario {
            scenarios: hvs,
            is_adversarial: false,
            label: format!("benign_repetitive_{family_seed}"),
        });
    }

    // Family 3: Adversarial converging (tight cluster, minimal noise)
    for family_seed in 0..5 {
        let target = ContinuousHV::random(DIM, 3000 + family_seed);
        let hvs: Vec<_> = (0..8).map(|_| target.perturb(0.01)).collect();
        scenarios.push(CalibrationScenario {
            scenarios: hvs,
            is_adversarial: true,
            label: format!("adversarial_converging_{family_seed}"),
        });
    }

    // Family 4: Adversarial evasive (converging with decoy interleaving)
    for family_seed in 0..3 {
        let target = ContinuousHV::random(DIM, 4000 + family_seed);
        let mut hvs = Vec::new();
        for round in 0..4 {
            // Decoy
            hvs.push(ContinuousHV::random(DIM, 5000 + family_seed * 10 + round));
            // Adversarial
            hvs.push(target.perturb(0.02));
        }
        scenarios.push(CalibrationScenario {
            scenarios: hvs,
            is_adversarial: true,
            label: format!("adversarial_evasive_{family_seed}"),
        });
    }

    let total_pos = scenarios.iter().filter(|s| s.is_adversarial).count();
    let total_neg = scenarios.len() - total_pos;
    println!(
        "Generated {} scenario families ({} adversarial, {} benign)\n",
        scenarios.len(),
        total_pos,
        total_neg,
    );

    // ── Sweep similarity thresholds ─────────────────────────────────────

    let thresholds: Vec<f64> = (0..25).map(|i| i as f64 * 0.02).collect();
    let result = calibrate_convergence_threshold(&scenarios, &thresholds, DIM, &base_config);

    println!("ROC Curve (similarity_threshold sweep):");
    println!(
        "{:<12} {:<8} {:<8} {:<10} {:<8}",
        "Threshold", "TPR", "FPR", "Precision", "F1"
    );
    println!("{}", "-".repeat(50));
    for pt in &result.roc_curve {
        println!(
            "{:<12.3} {:<8.3} {:<8.3} {:<10.3} {:<8.3}",
            pt.threshold, pt.true_positive_rate, pt.false_positive_rate, pt.precision, pt.f1,
        );
    }

    println!("\n--- Summary ---");
    println!("AUC:              {:.4}", result.auc);
    println!("Best F1:          {:.4}", result.best_f1);
    println!("Best threshold:   {:.4}", result.best_f1_threshold);
    println!(
        "\nRecommendation: Set `convergence_similarity_threshold = {:.3}`",
        result.best_f1_threshold
    );
    println!(
        "  (current default: {:.3})",
        base_config.convergence_similarity_threshold
    );
}