// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Error Analysis for Causal Discovery
//!
//! Analyzes which pairs Majority Voting gets wrong and identifies patterns.
//! Uses only the imports that work from the existing benchmark system.

use symthaea::benchmarks::{
    AnmDiscovery, CausalDirection, IgciDiscovery, ReciDiscovery, TuebingenAdapter,
    discover_information_theoretic, discover_majority_voting,
};

/// Meta-features extracted from a pair
#[derive(Debug, Clone)]
struct MetaFeatures {
    pair_id: String,
    n_samples: usize,
    x_mean: f64,
    x_std: f64,
    x_skewness: f64,
    y_mean: f64,
    y_std: f64,
    y_skewness: f64,
    correlation: f64,
    nonlinearity_score: f64,
    noise_ratio: f64,
}

fn compute_mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn compute_std(v: &[f64], mean: f64) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    variance.sqrt()
}

fn compute_skewness(v: &[f64], mean: f64, std: f64) -> f64 {
    if v.len() < 3 || std == 0.0 {
        return 0.0;
    }
    let n = v.len() as f64;
    v.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n
}

fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let x_mean = compute_mean(x);
    let y_mean = compute_mean(y);
    let x_std = compute_std(x, x_mean);
    let y_std = compute_std(y, y_mean);

    if x_std == 0.0 || y_std == 0.0 {
        return 0.0;
    }

    let n = x.len() as f64;
    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
        .sum::<f64>()
        / (n - 1.0);

    cov / (x_std * y_std)
}

fn compute_nonlinearity(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 5 {
        return 0.0;
    }

    let x_mean = compute_mean(x);
    let y_mean = compute_mean(y);

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        ss_xy += (xi - x_mean) * (yi - y_mean);
        ss_xx += (xi - x_mean).powi(2);
    }

    if ss_xx == 0.0 {
        return 0.0;
    }

    let slope = ss_xy / ss_xx;
    let intercept = y_mean - slope * x_mean;

    let residuals: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| yi - (slope * xi + intercept))
        .collect();

    let mut sorted_pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(residuals.iter())
        .map(|(xi, ri)| (*xi, *ri))
        .collect();
    sorted_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let sorted_residuals: Vec<f64> = sorted_pairs.iter().map(|(_, r)| *r).collect();

    if sorted_residuals.len() < 3 {
        return 0.0;
    }

    let r_mean = compute_mean(&sorted_residuals);
    let mut num = 0.0;
    let mut den = 0.0;

    for i in 1..sorted_residuals.len() {
        num += (sorted_residuals[i] - r_mean) * (sorted_residuals[i - 1] - r_mean);
    }
    for r in &sorted_residuals {
        den += (r - r_mean).powi(2);
    }

    if den == 0.0 {
        return 0.0;
    }

    (num / den).abs()
}

fn compute_noise_ratio(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 3 {
        return 1.0;
    }

    let x_mean = compute_mean(x);
    let y_mean = compute_mean(y);
    let y_var = compute_std(y, y_mean).powi(2);

    if y_var == 0.0 {
        return 0.0;
    }

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        ss_xy += (xi - x_mean) * (yi - y_mean);
        ss_xx += (xi - x_mean).powi(2);
    }

    if ss_xx == 0.0 {
        return 1.0;
    }

    let slope = ss_xy / ss_xx;
    let intercept = y_mean - slope * x_mean;

    let residual_var: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
        .sum::<f64>()
        / x.len() as f64;

    residual_var / y_var
}

fn extract_meta_features(pair_id: &str, x: &[f64], y: &[f64]) -> MetaFeatures {
    let x_mean = compute_mean(x);
    let x_std = compute_std(x, x_mean);
    let y_mean = compute_mean(y);
    let y_std = compute_std(y, y_mean);

    MetaFeatures {
        pair_id: pair_id.to_string(),
        n_samples: x.len(),
        x_mean,
        x_std,
        x_skewness: compute_skewness(x, x_mean, x_std),
        y_mean,
        y_std,
        y_skewness: compute_skewness(y, y_mean, y_std),
        correlation: compute_correlation(x, y),
        nonlinearity_score: compute_nonlinearity(x, y),
        noise_ratio: compute_noise_ratio(x, y),
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║           ERROR ANALYSIS - CAUSAL DISCOVERY METHODS                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let tuebingen_path = "benchmarks/external/tuebingen";
    let adapter = match TuebingenAdapter::load(tuebingen_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to load dataset: {}", e);
            return;
        }
    };

    let pairs = adapter.get_pairs();
    println!("Loaded {} cause-effect pairs\n", pairs.len());

    let reci = ReciDiscovery::new();
    let igci = IgciDiscovery::new();
    let anm = AnmDiscovery::new();

    // Track results
    let mut reci_correct = 0;
    let mut igci_correct = 0;
    let mut anm_correct = 0;
    let mut info_correct = 0;
    let mut majority_correct = 0;

    let mut error_cases: Vec<(String, MetaFeatures, [bool; 5])> = Vec::new();
    let mut correct_cases: Vec<(String, MetaFeatures, [bool; 5])> = Vec::new();

    println!("Analyzing all pairs...\n");

    for pair in pairs {
        let meta = extract_meta_features(&pair.id, &pair.x, &pair.y);
        let gt = pair.ground_truth;

        // Run each method
        let reci_pred = reci.discover(&pair.x, &pair.y).direction;
        let igci_pred = igci.discover(&pair.x, &pair.y).direction;
        let anm_pred = anm.discover(&pair.x, &pair.y).direction;
        let info_pred = discover_information_theoretic(&pair.x, &pair.y);
        let majority_pred = discover_majority_voting(&pair.x, &pair.y);

        let results = [
            reci_pred == gt,
            igci_pred == gt,
            anm_pred == gt,
            info_pred == gt,
            majority_pred == gt,
        ];

        if results[0] {
            reci_correct += 1;
        }
        if results[1] {
            igci_correct += 1;
        }
        if results[2] {
            anm_correct += 1;
        }
        if results[3] {
            info_correct += 1;
        }
        if results[4] {
            majority_correct += 1;
        }

        if results[4] {
            correct_cases.push((pair.id.clone(), meta, results));
        } else {
            error_cases.push((pair.id.clone(), meta, results));
        }
    }

    // SUMMARY
    println!("┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ METHOD ACCURACY SUMMARY                                                 │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    let n = pairs.len();
    println!(
        "  {:15} {:3}/{} ({:.1}%)",
        "RECI",
        reci_correct,
        n,
        reci_correct as f64 / n as f64 * 100.0
    );
    println!(
        "  {:15} {:3}/{} ({:.1}%)",
        "IGCI",
        igci_correct,
        n,
        igci_correct as f64 / n as f64 * 100.0
    );
    println!(
        "  {:15} {:3}/{} ({:.1}%)",
        "ANM",
        anm_correct,
        n,
        anm_correct as f64 / n as f64 * 100.0
    );
    println!(
        "  {:15} {:3}/{} ({:.1}%)",
        "Info-Theoretic",
        info_correct,
        n,
        info_correct as f64 / n as f64 * 100.0
    );
    println!(
        "  {:15} {:3}/{} ({:.1}%)",
        "Majority Voting",
        majority_correct,
        n,
        majority_correct as f64 / n as f64 * 100.0
    );

    // ERROR CASE ANALYSIS
    println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
    println!(
        "│ ERROR CASES ({} pairs where Majority Voting fails)                       │",
        error_cases.len()
    );
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    let mut rescue_counts = [0usize; 4]; // RECI, IGCI, ANM, Info

    for (pair_id, meta, results) in &error_cases {
        let mut rescuers = Vec::new();
        if results[0] {
            rescuers.push("RECI");
            rescue_counts[0] += 1;
        }
        if results[1] {
            rescuers.push("IGCI");
            rescue_counts[1] += 1;
        }
        if results[2] {
            rescuers.push("ANM");
            rescue_counts[2] += 1;
        }
        if results[3] {
            rescuers.push("Info");
            rescue_counts[3] += 1;
        }

        println!(
            "  Pair {:>4}: n={:>5}, corr={:>6.2}, nonlin={:.3}, noise={:.3}  Rescuers: {:?}",
            pair_id,
            meta.n_samples,
            meta.correlation,
            meta.nonlinearity_score,
            meta.noise_ratio,
            rescuers
        );
    }

    println!("\n  RESCUE POTENTIAL:");
    println!(
        "  {:15} rescues {}/{} error cases",
        "RECI",
        rescue_counts[0],
        error_cases.len()
    );
    println!(
        "  {:15} rescues {}/{} error cases",
        "IGCI",
        rescue_counts[1],
        error_cases.len()
    );
    println!(
        "  {:15} rescues {}/{} error cases",
        "ANM",
        rescue_counts[2],
        error_cases.len()
    );
    println!(
        "  {:15} rescues {}/{} error cases",
        "Info-Theoretic",
        rescue_counts[3],
        error_cases.len()
    );

    // Recoverable analysis
    let mut recoverable = 0;
    let mut unrecoverable_pairs = Vec::new();

    for (pair_id, meta, results) in &error_cases {
        if results[0] || results[1] || results[2] || results[3] {
            recoverable += 1;
        } else {
            unrecoverable_pairs.push((pair_id.clone(), meta.clone()));
        }
    }

    println!("\n  Recoverable (some method is right): {}", recoverable);
    println!(
        "  Unrecoverable (all methods wrong):  {}",
        unrecoverable_pairs.len()
    );

    // META-FEATURE PATTERNS
    println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ META-FEATURE PATTERNS                                                   │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    let avg_correct_n: f64 = correct_cases
        .iter()
        .map(|(_, m, _)| m.n_samples as f64)
        .sum::<f64>()
        / correct_cases.len().max(1) as f64;
    let avg_error_n: f64 = error_cases
        .iter()
        .map(|(_, m, _)| m.n_samples as f64)
        .sum::<f64>()
        / error_cases.len().max(1) as f64;

    let avg_correct_corr: f64 = correct_cases
        .iter()
        .map(|(_, m, _)| m.correlation.abs())
        .sum::<f64>()
        / correct_cases.len().max(1) as f64;
    let avg_error_corr: f64 = error_cases
        .iter()
        .map(|(_, m, _)| m.correlation.abs())
        .sum::<f64>()
        / error_cases.len().max(1) as f64;

    let avg_correct_nonlin: f64 = correct_cases
        .iter()
        .map(|(_, m, _)| m.nonlinearity_score)
        .sum::<f64>()
        / correct_cases.len().max(1) as f64;
    let avg_error_nonlin: f64 = error_cases
        .iter()
        .map(|(_, m, _)| m.nonlinearity_score)
        .sum::<f64>()
        / error_cases.len().max(1) as f64;

    let avg_correct_noise: f64 = correct_cases
        .iter()
        .map(|(_, m, _)| m.noise_ratio)
        .sum::<f64>()
        / correct_cases.len().max(1) as f64;
    let avg_error_noise: f64 = error_cases
        .iter()
        .map(|(_, m, _)| m.noise_ratio)
        .sum::<f64>()
        / error_cases.len().max(1) as f64;

    println!("  Feature            Correct Cases    Error Cases    Difference");
    println!("  ─────────────────────────────────────────────────────────────────");
    println!(
        "  Sample size        {:8.1}         {:8.1}       {:+.1}",
        avg_correct_n,
        avg_error_n,
        avg_error_n - avg_correct_n
    );
    println!(
        "  |Correlation|      {:8.3}         {:8.3}       {:+.3}",
        avg_correct_corr,
        avg_error_corr,
        avg_error_corr - avg_correct_corr
    );
    println!(
        "  Nonlinearity       {:8.3}         {:8.3}       {:+.3}",
        avg_correct_nonlin,
        avg_error_nonlin,
        avg_error_nonlin - avg_correct_nonlin
    );
    println!(
        "  Noise ratio        {:8.3}         {:8.3}       {:+.3}",
        avg_correct_noise,
        avg_error_noise,
        avg_error_noise - avg_correct_noise
    );

    // ORACLE ANALYSIS
    println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ ORACLE ANALYSIS (Best possible with perfect method selection)           │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    let oracle_correct = correct_cases.len() + recoverable;

    println!(
        "  Oracle accuracy (perfect method selection): {}/{} ({:.1}%)",
        oracle_correct,
        n,
        oracle_correct as f64 / n as f64 * 100.0
    );
    println!(
        "  Current best (Majority Voting):             {}/{} ({:.1}%)",
        majority_correct,
        n,
        majority_correct as f64 / n as f64 * 100.0
    );
    println!(
        "  Potential improvement:                      +{:.1}%",
        (oracle_correct - majority_correct) as f64 / n as f64 * 100.0
    );

    // UNRECOVERABLE CASES
    if !unrecoverable_pairs.is_empty() {
        println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
        println!("│ UNRECOVERABLE CASES (All methods fail)                                  │");
        println!("└──────────────────────────────────────────────────────────────────────────┘\n");

        for (pair_id, meta) in &unrecoverable_pairs {
            println!(
                "  Pair {:>4}: n={:>5}, corr={:>6.2}, nonlin={:.3}, noise={:.3}",
                pair_id,
                meta.n_samples,
                meta.correlation,
                meta.nonlinearity_score,
                meta.noise_ratio
            );
        }
    }

    // METHOD AGREEMENT ANALYSIS
    println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ METHOD DISAGREEMENT ON ERROR CASES                                      │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    // For each error case, how many methods agree with majority?
    for (pair_id, meta, results) in &error_cases {
        let votes_for_majority: usize = results[..4].iter().filter(|&&r| !r).count();
        let votes_against_majority: usize = results[..4].iter().filter(|&&r| r).count();

        if votes_against_majority >= 2 {
            println!(
                "  Pair {:>4}: {}/{} methods DISAGREE with majority (rescuable)",
                pair_id, votes_against_majority, 4
            );
        }
    }

    println!("\n  Done!");
}
