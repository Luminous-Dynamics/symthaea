// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal Prediction O(1) Proof Benchmark
//!
//! Proves that the CfC closed-form solution gives O(1) temporal prediction:
//! predicting plasma state 10s ahead costs the same wall-clock time as 1ms ahead.
//!
//! Run: `cargo run -p symthaea-physics --example temporal_benchmark --release`

#![deny(unsafe_code)]

use serde::Serialize;
use std::time::Instant;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use symthaea_physics::fusion_twin::{
    PLASMA_HORIZON_LABELS, PLASMA_HORIZONS, PlasmaMultiScalePredictor,
};

const WARMUP_ITERATIONS: usize = 100;
const MEASUREMENT_ITERATIONS: usize = 1000;
/// O(1) pass threshold: max_mean / min_mean must be below this ratio
const O1_RATIO_THRESHOLD: f64 = 2.0;

// Extended horizons to stress-test O(1) property across 7 orders of magnitude
const EXTENDED_HORIZONS: &[f32] = &[100.0, 1000.0, 10_000.0];
const EXTENDED_LABELS: &[&str] = &[
    "100 s (extended-1)",
    "1000 s (extended-2)",
    "10000 s (extended-3)",
];

#[derive(Debug, Clone, Serialize)]
struct HorizonResult {
    label: String,
    horizon_seconds: f64,
    mean_us: f64,
    std_dev_us: f64,
    min_us: f64,
    max_us: f64,
    ci_95_lower_us: f64,
    ci_95_upper_us: f64,
    prediction_similarity: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkReport {
    hdc_dimension: usize,
    warmup_iterations: usize,
    measurement_iterations: usize,
    horizon_results: Vec<HorizonResult>,
    o1_ratio: f64,
    o1_threshold: f64,
    o1_pass: bool,
    total_wall_clock_seconds: f64,
}

fn measure_horizon(
    predictor: &PlasmaMultiScalePredictor,
    input: &ContinuousHV,
    horizon: f32,
    label: &str,
) -> HorizonResult {
    // Warmup: discard these timings
    for _ in 0..WARMUP_ITERATIONS {
        let _ = predictor.predict_at_horizon(input, horizon);
    }

    // Measurement
    let mut times_us = Vec::with_capacity(MEASUREMENT_ITERATIONS);
    for _ in 0..MEASUREMENT_ITERATIONS {
        let start = Instant::now();
        let _ = predictor.predict_at_horizon(input, horizon);
        let elapsed = start.elapsed();
        times_us.push(elapsed.as_nanos() as f64 / 1000.0);
    }

    let n = times_us.len() as f64;
    let mean = times_us.iter().sum::<f64>() / n;
    let variance = times_us.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    let min = times_us.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times_us.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ci_margin = 1.96 * std_dev / n.sqrt();

    // Measure prediction quality: similarity of predicted state to input
    let predicted = predictor.predict_at_horizon(input, horizon);
    let similarity = predicted.similarity(input) as f64;

    HorizonResult {
        label: label.to_string(),
        horizon_seconds: horizon as f64,
        mean_us: mean,
        std_dev_us: std_dev,
        min_us: min,
        max_us: max,
        ci_95_lower_us: mean - ci_margin,
        ci_95_upper_us: mean + ci_margin,
        prediction_similarity: similarity,
    }
}

fn print_table(results: &[HorizonResult]) {
    println!();
    println!(
        "{:<28} {:>12} {:>10} {:>10} {:>10} {:>18} {:>12}",
        "Horizon", "Mean (us)", "Std Dev", "Min (us)", "Max (us)", "95% CI (us)", "Similarity"
    );
    println!("{}", "-".repeat(112));
    for r in results {
        println!(
            "{:<28} {:>12.2} {:>10.2} {:>10.2} {:>10.2} {:>8.2} - {:<8.2} {:>10.4}",
            r.label,
            r.mean_us,
            r.std_dev_us,
            r.min_us,
            r.max_us,
            r.ci_95_lower_us,
            r.ci_95_upper_us,
            r.prediction_similarity,
        );
    }
    println!();
}

fn main() {
    let total_start = Instant::now();

    println!("==========================================================");
    println!("  Temporal Prediction O(1) Proof Benchmark");
    println!("  HDC Dimension: {}", HDC_DIMENSION);
    println!("  Warmup: {} iterations (discarded)", WARMUP_ITERATIONS);
    println!(
        "  Measurement: {} iterations per horizon",
        MEASUREMENT_ITERATIONS
    );
    println!("==========================================================");

    let predictor = PlasmaMultiScalePredictor::new();
    let input = ContinuousHV::random(HDC_DIMENSION, 0xBE_0001);

    // Feed some observations to warm up the predictor's internal state
    {
        let mut warmup_pred = PlasmaMultiScalePredictor::new();
        for i in 0..50 {
            let obs = ContinuousHV::random(HDC_DIMENSION, 0x0B_0000 + i);
            warmup_pred.observe(&obs, 0.001);
        }
    }

    // ── Phase 1: Standard plasma horizons (1ms - 10s) ──────────────────
    println!("\n--- Phase 1: Standard Plasma Horizons ---");

    let mut all_results = Vec::new();
    for (i, (&horizon, &label)) in PLASMA_HORIZONS
        .iter()
        .zip(PLASMA_HORIZON_LABELS)
        .enumerate()
    {
        let result = measure_horizon(&predictor, &input, horizon, label);
        println!(
            "  [{}/{}] {}: {:.2} us (std: {:.2})",
            i + 1,
            PLASMA_HORIZONS.len(),
            label,
            result.mean_us,
            result.std_dev_us,
        );
        all_results.push(result);
    }

    print_table(&all_results);

    // ── Phase 2: Extended horizons (100s - 10000s) ─────────────────────
    println!("--- Phase 2: Extended Horizons (stress test) ---");

    let mut extended_results = Vec::new();
    for (i, (&horizon, &label)) in EXTENDED_HORIZONS.iter().zip(EXTENDED_LABELS).enumerate() {
        let result = measure_horizon(&predictor, &input, horizon, label);
        println!(
            "  [{}/{}] {}: {:.2} us (std: {:.2})",
            i + 1,
            EXTENDED_HORIZONS.len(),
            label,
            result.mean_us,
            result.std_dev_us,
        );
        extended_results.push(result);
    }

    print_table(&extended_results);

    // ── O(1) Verdict ───────────────────────────────────────────────────
    let combined: Vec<&HorizonResult> = all_results.iter().chain(extended_results.iter()).collect();

    let min_mean = combined
        .iter()
        .map(|r| r.mean_us)
        .fold(f64::INFINITY, f64::min);
    let max_mean = combined
        .iter()
        .map(|r| r.mean_us)
        .fold(f64::NEG_INFINITY, f64::max);
    let o1_ratio = if min_mean > 0.0 {
        max_mean / min_mean
    } else {
        f64::INFINITY
    };
    let o1_pass = o1_ratio < O1_RATIO_THRESHOLD;

    println!("==========================================================");
    println!("  O(1) TEMPORAL PREDICTION VERDICT");
    println!("==========================================================");
    println!("  Fastest horizon mean:  {:.2} us", min_mean);
    println!("  Slowest horizon mean:  {:.2} us", max_mean);
    println!(
        "  Ratio (slowest/fastest): {:.3}x (threshold: <{:.1}x)",
        o1_ratio, O1_RATIO_THRESHOLD
    );
    println!(
        "  Verdict: {} (ratio {:.3} {} {:.1})",
        if o1_pass { "PASS" } else { "FAIL" },
        o1_ratio,
        if o1_pass { "<" } else { ">=" },
        O1_RATIO_THRESHOLD,
    );
    println!("==========================================================");

    // ── Similarity analysis ────────────────────────────────────────────
    println!("\n--- Prediction Similarity Analysis ---");
    println!("  (How similar is the predicted state to the input?)");
    for r in combined.iter() {
        let bar_len = (r.prediction_similarity.abs() * 40.0).min(40.0) as usize;
        let bar: String = "#".repeat(bar_len);
        println!(
            "  {:<28} sim={:>7.4}  [{}]",
            r.label, r.prediction_similarity, bar
        );
    }

    let total_elapsed = total_start.elapsed();

    // ── JSON output ────────────────────────────────────────────────────
    let mut all_horizon_results: Vec<HorizonResult> = all_results;
    all_horizon_results.extend(extended_results);

    let report = BenchmarkReport {
        hdc_dimension: HDC_DIMENSION,
        warmup_iterations: WARMUP_ITERATIONS,
        measurement_iterations: MEASUREMENT_ITERATIONS,
        horizon_results: all_horizon_results,
        o1_ratio,
        o1_threshold: O1_RATIO_THRESHOLD,
        o1_pass,
        total_wall_clock_seconds: total_elapsed.as_secs_f64(),
    };

    println!("\n--- JSON Output ---");
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("JSON serialization failed")
    );

    println!(
        "\nTotal benchmark wall-clock time: {:.2}s",
        total_elapsed.as_secs_f64()
    );

    if !o1_pass {
        std::process::exit(1);
    }
}
