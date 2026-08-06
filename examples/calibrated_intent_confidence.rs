// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive Compression Program C4 -- calibrated surprise.
//!
//! Pre-registered protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md §9
//! (registered BEFORE this harness existed).
//!
//! Does `SemanticIntentClassifier::confidence_calibrated` (softmax(beta*sim), a genuine
//! probability) reduce Expected Calibration Error relative to `classify()`'s existing raw
//! affine `confidence` (`similarity + keyword_boost`, unbounded, not a probability), on a
//! held-out labeled set the classifier's own prototype-building examples never saw?
//!
//! Method: build a labeled validation set (deliberately DIFFERENT sentences from
//! `SemanticIntentClassifier::new()`'s own prototype-building examples -- a genuine held-out
//! test, not testing on training data), classify each example, form `(confidence,
//! was_correct)` pairs for the raw baseline and for a grid of `beta` values, score each with
//! `analyze_pairs` (already exists, unused until now), and report the beta minimizing ECE
//! against the raw baseline's own ECE.

use symthaea::consciousness::recursive_improvement::calibration_analytics::analyze_pairs;
use symthaea::language::semantic_intent::{IntentCategory, SemanticIntentClassifier};

/// Held-out labeled validation set: ~8 examples per category, deliberately distinct wording
/// from `SemanticIntentClassifier::new()`'s own prototype-building examples (verified by
/// inspection, not just intent) -- a genuine test of generalization, not training-set recall.
/// Hand-authored and disclosed as such; not a gold-standard corpus.
fn labeled_examples() -> Vec<(&'static str, IntentCategory)> {
    use IntentCategory::*;
    vec![
        // NixOS
        (
            "how do I add a new package to my nixos configuration",
            NixOS,
        ),
        ("what does nix flake check actually verify", NixOS),
        ("my home-manager module won't build, what's wrong", NixOS),
        ("explain nix derivations and how they're built", NixOS),
        ("how to pin nixpkgs to a specific commit", NixOS),
        ("nixos generation rollback after a broken update", NixOS),
        ("overlay a package version in my nix flake", NixOS),
        ("why is my nix-shell missing a library at runtime", NixOS),
        // Programming
        (
            "what's the difference between a list and a tuple in python",
            Programming,
        ),
        (
            "help me fix this null pointer exception in java",
            Programming,
        ),
        ("write a recursive fibonacci function in rust", Programming),
        (
            "how do closures capture variables in javascript",
            Programming,
        ),
        (
            "optimize this nested loop for better performance",
            Programming,
        ),
        (
            "explain the difference between == and === in js",
            Programming,
        ),
        ("how to implement a binary search tree", Programming),
        (
            "what causes a stack overflow in recursive code",
            Programming,
        ),
        // Math
        ("what is the derivative of sin(x) times x squared", Math),
        ("explain eigenvalues and eigenvectors simply", Math),
        ("how do you compute a determinant of a 3x3 matrix", Math),
        ("prove that the square root of two is irrational", Math),
        ("what is the fundamental theorem of calculus", Math),
        ("solve this system of linear equations", Math),
        ("explain bayes theorem with an example", Math),
        (
            "what's the difference between permutations and combinations",
            Math,
        ),
        // General
        ("what's a good recipe for dinner tonight", General),
        ("recommend a book about ancient rome", General),
        ("what causes the northern lights", General),
        ("how far away is the moon from earth", General),
        ("tell me an interesting fact about octopuses", General),
        ("what's the tallest mountain in the world", General),
        ("explain why the sky is blue", General),
        ("what year did the berlin wall fall", General),
        // SystemAdmin
        (
            "how do I find which process is using port 8080",
            SystemAdmin,
        ),
        ("set up a systemd timer for a backup script", SystemAdmin),
        ("check memory usage on a remote linux server", SystemAdmin),
        ("configure a reverse proxy with nginx", SystemAdmin),
        ("rotate log files automatically with logrotate", SystemAdmin),
        ("how to add a user to the sudoers file", SystemAdmin),
        (
            "diagnose high cpu usage from a runaway process",
            SystemAdmin,
        ),
        (
            "set up a cron job that runs every five minutes",
            SystemAdmin,
        ),
    ]
}

/// Clamp the raw affine confidence into [0,1] for a fair `analyze_pairs` comparison -- the
/// unbounded raw score (similarity + keyword_boost, boost up to 0.05) isn't naturally a
/// probability, but this is the closest honest baseline reading of "how confident does the
/// existing production code already claim to be."
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

fn main() {
    println!("Predictive Compression C4 -- calibrated surprise (intent classifier confidence)");
    println!("protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md (Experiment C4)");
    println!();

    let mut classifier = SemanticIntentClassifier::new();
    let examples = labeled_examples();
    println!(
        "Held-out labeled set: {} examples across 5 categories",
        examples.len()
    );

    // Classify every example once, keep the full per-category scores for each (needed to
    // recompute confidence_calibrated at every candidate beta without re-running the
    // classifier).
    let classified: Vec<_> = examples
        .iter()
        .map(|(query, expected)| {
            let result = classifier.classify(query);
            let correct = result.category == *expected;
            (result, correct)
        })
        .collect();

    let n_correct = classified.iter().filter(|(_, c)| *c).count();
    println!(
        "Raw top-1 accuracy on held-out set: {}/{} ({:.1}%)",
        n_correct,
        classified.len(),
        100.0 * n_correct as f64 / classified.len() as f64
    );
    println!();

    // Baseline: raw affine confidence, clamped to [0,1].
    let baseline_pairs: Vec<(f64, bool)> = classified
        .iter()
        .map(|(result, correct)| (clamp01(result.confidence) as f64, *correct))
        .collect();

    // Diagnostic: a `resolution=0.0000` result means every prediction landed in the same bin
    // (no discriminative spread) -- ECE can look artificially good on a degenerate,
    // uninformative predictor. Print the raw distribution before trusting the ECE numbers.
    {
        let vals: Vec<f64> = baseline_pairs.iter().map(|(p, _)| *p).collect();
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        println!(
            "  [diagnostic] baseline confidence range: min={min:.4} max={max:.4} mean={mean:.4} \
             (narrow range -> single-bin collapse -> resolution=0 -> ECE looks better than it is)"
        );
    }
    let baseline_report = analyze_pairs(&baseline_pairs, 10);
    match &baseline_report {
        Some(r) => println!(
            "BASELINE (raw affine confidence, clamped): ECE={:.4} brier={:.4} \
             reliability={:.4} resolution={:.4} accuracy={:.4}",
            r.ece, r.brier, r.reliability, r.resolution, r.accuracy
        ),
        None => println!("BASELINE: analyze_pairs returned None (empty set?)"),
    }
    println!();

    // Grid search over beta, scoring each with analyze_pairs.
    let betas = [
        0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0,
        250.0, 500.0, 1000.0,
    ];
    println!(
        "{:>6} | {:>8} {:>8} {:>10} {:>10}",
        "beta", "ECE", "brier", "reliability", "resolution"
    );
    let mut best: Option<(f32, f64)> = None; // (beta, ece)
    for &beta in &betas {
        let pairs: Vec<(f64, bool)> = classified
            .iter()
            .map(|(result, correct)| {
                let p = SemanticIntentClassifier::confidence_calibrated(&result.scores, beta);
                (p as f64, *correct)
            })
            .collect();
        if let Some(report) = analyze_pairs(&pairs, 10) {
            println!(
                "{beta:>6.1} | {:>8.4} {:>8.4} {:>10.4} {:>10.4}",
                report.ece, report.brier, report.reliability, report.resolution
            );
            if best.is_none_or(|(_, best_ece)| report.ece < best_ece) {
                best = Some((beta, report.ece));
            }
        }
    }
    println!();

    if let (Some((best_beta, best_ece)), Some(baseline)) = (best, &baseline_report) {
        println!(
            "Best beta={best_beta:.1} gives ECE={best_ece:.4} vs baseline ECE={:.4} \
             (delta={:+.4}, negative = calibrated confidence is better calibrated)",
            baseline.ece,
            best_ece - baseline.ece
        );
    }

    println!();
    println!(
        "done. Append results + verdict to the protocol doc (§9, C4 Results), per house \
         convention."
    );
}
