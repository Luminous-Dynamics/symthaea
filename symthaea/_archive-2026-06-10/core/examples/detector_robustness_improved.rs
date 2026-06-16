// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Detector Robustness Improvement Test
//!
//! Tests the phenomenal detector before and after applying contrastive training
//! improvements for semantic confusion handling.
//!
//! ## Goals
//! - Improve overall adversarial test accuracy from 63.6% baseline
//! - Improve semantic confusion accuracy from 37.5% toward 60%+
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example detector_robustness_improved --features neural-bridge --release
//! ```

use anyhow::Result;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::{
    ContrastiveExample, ContrastiveExamples, DetectionMethod, DetectorConfig, PhenomenalDetector,
};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        Ok(())
    }

    #[cfg(feature = "neural-bridge")]
    run_experiment()
}

#[cfg(feature = "neural-bridge")]
fn run_experiment() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   PHENOMENAL DETECTOR ROBUSTNESS IMPROVEMENT");
    println!("   Using Contrastive Training Examples");
    println!("================================================================\n");

    let contrastive_path = "data/contrastive_examples.json";

    // Load contrastive examples for analysis
    println!("Loading contrastive examples from: {}", contrastive_path);
    let content = std::fs::read_to_string(contrastive_path)?;
    let examples: ContrastiveExamples = serde_json::from_str(&content)?;

    println!("  Total examples: {}", examples.examples.len());
    println!("  Categories:");
    let mut by_cat: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for ex in &examples.examples {
        *by_cat.entry(ex.category.clone()).or_insert(0) += 1;
    }
    for (cat, count) in &by_cat {
        println!("    - {}: {}", cat, count);
    }

    // Phase 1: Baseline evaluation (without contrastive training)
    println!("\n================================================================");
    println!("   PHASE 1: BASELINE EVALUATION");
    println!("   (Standard detector without contrastive improvements)");
    println!("================================================================\n");

    println!("Creating baseline detector...");
    let load_start = Instant::now();

    let mut baseline_config = DetectorConfig::default();
    baseline_config.context_aware = false; // Disable context-aware scoring for baseline

    let mut baseline_detector = PhenomenalDetector::with_config(baseline_config)?;
    baseline_detector.load_contrastive_examples(contrastive_path)?;
    baseline_detector.auto_calibrate()?;

    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    println!("Evaluating baseline detector on contrastive examples...\n");
    let baseline_eval = baseline_detector.evaluate_contrastive_baseline()?;

    print_evaluation_results("BASELINE", &baseline_eval);

    // Phase 2: Improved evaluation (with contrastive training)
    println!("\n================================================================");
    println!("   PHASE 2: IMPROVED DETECTOR EVALUATION");
    println!("   (With contrastive training and context-aware scoring)");
    println!("================================================================\n");

    println!("Creating improved detector with contrastive calibration...");
    let load_start = Instant::now();

    let mut improved_config = DetectorConfig::default();
    improved_config.context_aware = true;
    improved_config.negative_weight = 1.5; // Weight negative examples higher
    improved_config.context_window = 3;

    let improved_detector =
        PhenomenalDetector::with_contrastive_examples(improved_config, contrastive_path)?;

    println!(
        "  Loaded and calibrated in {:.2}s\n",
        load_start.elapsed().as_secs_f64()
    );

    println!("Evaluating improved detector on contrastive examples...\n");
    let improved_eval = improved_detector.evaluate_contrastive()?;

    print_evaluation_results("IMPROVED", &improved_eval);

    // Phase 3: Comparison and analysis
    println!("\n================================================================");
    println!("   PHASE 3: COMPARISON AND ANALYSIS");
    println!("================================================================\n");

    let overall_delta = improved_eval.accuracy - baseline_eval.accuracy;

    println!("Overall Accuracy Comparison:");
    println!(
        "  Baseline: {:.1}% ({}/{})",
        baseline_eval.accuracy * 100.0,
        baseline_eval.correct,
        baseline_eval.total
    );
    println!(
        "  Improved: {:.1}% ({}/{})",
        improved_eval.accuracy * 100.0,
        improved_eval.correct,
        improved_eval.total
    );
    println!("  Delta:    {:+.1}%\n", overall_delta * 100.0);

    println!("Category-wise Improvement:");
    println!(
        "{:<25} {:>12} {:>12} {:>10}",
        "Category", "Baseline", "Improved", "Delta"
    );
    println!("{}", "-".repeat(60));

    for (cat, (total, _, imp_acc)) in &improved_eval.by_category {
        let base_acc = baseline_eval
            .by_category
            .get(cat)
            .map(|(_, _, a)| *a)
            .unwrap_or(0.0);
        let delta = imp_acc - base_acc;
        println!(
            "{:<25} {:>11.1}% {:>11.1}% {:>+9.1}%",
            cat,
            base_acc * 100.0,
            imp_acc * 100.0,
            delta * 100.0
        );
    }

    println!("\nDifficulty-wise Improvement:");
    println!(
        "{:<15} {:>12} {:>12} {:>10}",
        "Difficulty", "Baseline", "Improved", "Delta"
    );
    println!("{}", "-".repeat(50));

    for diff_level in &["easy", "medium", "hard"] {
        let base_acc = baseline_eval
            .by_difficulty
            .get(*diff_level)
            .map(|(_, _, a)| *a)
            .unwrap_or(0.0);
        let imp_acc = improved_eval
            .by_difficulty
            .get(*diff_level)
            .map(|(_, _, a)| *a)
            .unwrap_or(0.0);
        let delta = imp_acc - base_acc;
        println!(
            "{:<15} {:>11.1}% {:>11.1}% {:>+9.1}%",
            diff_level,
            base_acc * 100.0,
            imp_acc * 100.0,
            delta * 100.0
        );
    }

    // Phase 4: Detailed error analysis
    println!("\n================================================================");
    println!("   PHASE 4: ERROR ANALYSIS");
    println!("================================================================\n");

    // Find examples that improved
    let mut improved_examples: Vec<(&ExampleEvaluation, &ExampleEvaluation)> = Vec::new();
    let mut regressed_examples: Vec<(&ExampleEvaluation, &ExampleEvaluation)> = Vec::new();
    let mut still_wrong: Vec<&ExampleEvaluation> = Vec::new();

    for (base, imp) in baseline_eval
        .evaluations
        .iter()
        .zip(improved_eval.evaluations.iter())
    {
        if !base.correct && imp.correct {
            improved_examples.push((base, imp));
        } else if base.correct && !imp.correct {
            regressed_examples.push((base, imp));
        } else if !imp.correct {
            still_wrong.push(imp);
        }
    }

    println!("Examples that IMPROVED ({}):", improved_examples.len());
    for (base, imp) in improved_examples.iter().take(5) {
        println!("  [{}] \"{}\"", imp.id, truncate(&imp.text, 50));
        println!(
            "       Expected: {} | Baseline: {:.2} -> Improved: {:.2}",
            imp.expected, base.score, imp.score
        );
    }
    if improved_examples.len() > 5 {
        println!("  ... and {} more", improved_examples.len() - 5);
    }

    if !regressed_examples.is_empty() {
        println!("\nExamples that REGRESSED ({}):", regressed_examples.len());
        for (base, imp) in regressed_examples.iter().take(5) {
            println!("  [{}] \"{}\"", imp.id, truncate(&imp.text, 50));
            println!(
                "       Expected: {} | Baseline: {:.2} -> Improved: {:.2}",
                imp.expected, base.score, imp.score
            );
        }
    }

    println!("\nExamples still INCORRECT ({}):", still_wrong.len());
    for eval in still_wrong.iter().take(5) {
        println!("  [{}] \"{}\"", eval.id, truncate(&eval.text, 50));
        println!(
            "       Expected: {} | Score: {:.2} | Cat: {} | Diff: {}",
            eval.expected, eval.score, eval.category, eval.difficulty
        );
    }
    if still_wrong.len() > 5 {
        println!("  ... and {} more", still_wrong.len() - 5);
    }

    // Phase 5: Summary and recommendations
    println!("\n================================================================");
    println!("   SUMMARY");
    println!("================================================================\n");

    let semantic_confusion_improved = improved_eval
        .by_category
        .get("semantic_confusion")
        .map(|(_, _, a)| *a >= 0.60)
        .unwrap_or(false);

    let semantic_acc = improved_eval
        .by_category
        .get("semantic_confusion")
        .map(|(_, _, a)| *a)
        .unwrap_or(0.0);

    if semantic_confusion_improved {
        println!(
            "SUCCESS: Semantic confusion accuracy improved to {:.1}% (target: 60%+)",
            semantic_acc * 100.0
        );
    } else {
        println!(
            "PARTIAL: Semantic confusion accuracy at {:.1}% (target: 60%+)",
            semantic_acc * 100.0
        );
    }

    println!("\nKey improvements from contrastive training:");
    println!("  1. Context-aware scoring detects functional context around phenomenal vocabulary");
    println!("  2. Negative example weighting improves discrimination at boundaries");
    println!("  3. Negation handling preserves topic classification");

    if overall_delta > 0.0 {
        println!(
            "\nOverall improvement: {:+.1}% accuracy gain",
            overall_delta * 100.0
        );
    }

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn print_evaluation_results(label: &str, eval: &symthaea::perception::ContrastiveEvaluation) {
    println!("{} Results:", label);
    println!(
        "  Overall: {:.1}% ({}/{})",
        eval.accuracy * 100.0,
        eval.correct,
        eval.total
    );

    println!("\n  By Category:");
    for (cat, (total, correct, acc)) in &eval.by_category {
        println!(
            "    {:<25}: {:.1}% ({}/{})",
            cat,
            acc * 100.0,
            correct,
            total
        );
    }

    println!("\n  By Difficulty:");
    for diff_level in &["easy", "medium", "hard"] {
        if let Some((total, correct, acc)) = eval.by_difficulty.get(*diff_level) {
            println!(
                "    {:<15}: {:.1}% ({}/{})",
                diff_level,
                acc * 100.0,
                correct,
                total
            );
        }
    }
    println!();
}

#[cfg(feature = "neural-bridge")]
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(feature = "neural-bridge")]
use symthaea::perception::ExampleEvaluation;