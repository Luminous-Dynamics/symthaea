// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ARC-AGI Strict Scoring Benchmark.
//!
//! **Honesty note**: This benchmark applies pixel-perfect grid reconstruction
//! scoring, the same standard used by the official ARC-AGI leaderboard. Unlike
//! the 2-AFC benchmarks in `arc_fluid.rs` (which test encoding fidelity), this
//! tests whether the HDC pipeline can produce exact output grids.
//!
//! Expected results: strict solve rate 0-3%. The HDC approach encodes grid
//! structure faithfully under 2-AFC scoring but cannot reconstruct
//! pixel-perfect outputs because majority-vote bundling is lossy. This is a
//! fundamental limitation of the representation, not a tuning issue.
//!
//! **RETRACTION + FIX (2026-07-18, `examples/arc_2afc_reaudit.rs`)**: this
//! file's 2-AFC distractor used to be `BinaryHV::random`, which is not a fair
//! alternative — see `arc_dataset.rs`'s retraction note for the mechanism and
//! the real-ARC-data numbers that exposed it (random 99.0% vs. fair
//! distractors 13.8-67.8%). Fixed to use `arc_dataset::fair_distractor_grid`
//! (a generic wrong transform of the test input) instead. Even fixed, this
//! remains **encoding-fidelity** scoring, not solve rate — read
//! `twoafc_accuracy`/`encoding_fidelity_gap` accordingly.
//!
//! ## Comparison
//!
//! - GPT-4: ~5% (Chollet 2024)
//! - o3-high: ~87.5% (OpenAI 2024)
//! - Human average: ~84% (Chollet 2019)
//! - HDC-LTC (2-AFC): encoding fidelity under a fair distractor, not solve rate
//! - HDC-LTC (strict): ~0-3% (grid reconstruction noise)
//!
//! The gap between 2-AFC and strict scores is the key finding: XOR binding
//! tests algebraic rule transfer, not generalization. Strict scoring exposes
//! this honestly.
//!
//! **REVIVED (2026-07-18, same day as the bit-rot was found)**: this file had
//! been bit-rotted for an unknown period — it called
//! `BinaryGridEncoder::decode_grid`/`grid_accuracy`/`grid_exact_match`, none
//! of which existed on that type, and was silently un-declared from
//! `reasoning/mod.rs` (and unregistered in `examples/run_psych_benchmarks.rs`)
//! as a result. This is very likely the actual source of the retracted "100%
//! 2-AFC, 4% strict" claim in `book/src/research/validation.md`/
//! `publications.md` (via `examples/arc_agi_benchmark.rs`, which imports this
//! type directly) — a one-time historical run whose numbers were published
//! and never re-verified once the module rotted. Fixed by adding a real
//! per-cell argmax-similarity decode to `BinaryGridEncoder`
//! (`decode_cell`/`decode_grid`: unbind each cell's row/col keys from the
//! grid HV, then nearest-neighbor cleanup against the color codebook — the
//! standard VSA "cleanup memory" retrieval pattern) plus `grid_accuracy`/
//! `grid_exact_match` as plain cell-by-cell grid comparisons. Re-declared in
//! `reasoning/mod.rs` and re-registered in `run_psych_benchmarks.rs`.
//! Verified: `cargo test -p symthaea-psych-bench --lib arc_strict` green,
//! plus new decode-specific unit tests in `binary_grid_encoder.rs`.

use crate::benchmarks::reasoning::arc_dataset::fair_distractor_grid;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;

/// ARC-AGI strict (pixel-perfect) scoring benchmark.
pub struct ArcStrictBenchmark;

impl PsychBenchmark for ArcStrictBenchmark {
    fn name(&self) -> &str {
        "Reasoning::ARC_Strict"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let trials = config.trials_per_condition;
        let seed = config.trial_seed("reasoning", "arc_strict", 0);

        let mut strict_correct = 0usize;
        let mut twoafc_correct = 0usize;
        let mut total_pixel_accuracy = 0.0f64;
        let mut total_similarity = 0.0f64;
        let mut pixel_accuracies = Vec::with_capacity(trials);
        let mut strict_results = Vec::with_capacity(trials);
        let mut twoafc_results = Vec::with_capacity(trials);
        let mut trial_trace = Vec::new();

        for trial in 0..trials {
            let trial_seed = seed ^ (trial as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let result = run_strict_trial(trial_seed, config.dimension);

            if result.strict_match {
                strict_correct += 1;
            }
            if result.twoafc_correct {
                twoafc_correct += 1;
            }
            total_pixel_accuracy += result.pixel_accuracy as f64;
            total_similarity += result.similarity as f64;
            pixel_accuracies.push(result.pixel_accuracy as f64);
            strict_results.push(if result.strict_match { 1.0 } else { 0.0 });
            twoafc_results.push(if result.twoafc_correct { 1.0 } else { 0.0 });

            if config.trial_trace {
                trial_trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "strict".to_string(),
                    correct: result.strict_match,
                    rt_ticks: 1.0,
                    similarity: result.similarity as f64,
                    confidence: result.pixel_accuracy as f64,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        let n = trials as f64;
        let strict_rate = strict_correct as f64 / n;
        let twoafc_rate = twoafc_correct as f64 / n;
        let mean_pixel_acc = total_pixel_accuracy / n;
        let mean_similarity = total_similarity / n;

        let mut metrics = BTreeMap::new();
        metrics.insert(
            "strict_solve_rate".to_string(),
            MetricValue::from_samples(&strict_results),
        );
        metrics.insert(
            "twoafc_accuracy".to_string(),
            MetricValue::from_samples(&twoafc_results),
        );
        metrics.insert(
            "mean_pixel_accuracy".to_string(),
            MetricValue::from_samples(&pixel_accuracies),
        );
        metrics.insert(
            "mean_similarity".to_string(),
            MetricValue::from_samples(&[mean_similarity]),
        );
        metrics.insert(
            "encoding_fidelity_gap".to_string(),
            MetricValue::from_samples(&[twoafc_rate - strict_rate]),
        );

        BenchmarkResult {
            benchmark: self.name().to_string(),
            config_label: config.label.clone(),
            metrics,
            elapsed_ms: start.elapsed().as_millis() as u64,
            conditions: 1,
            trials_per_condition: trials,
            trial_trace,
            notes: vec![
                format!(
                    "Strict solve rate: {:.1}% (vs GPT-4 ~5%, o3-high ~87.5%, Human ~84%)",
                    strict_rate * 100.0
                ),
                format!(
                    "2-AFC accuracy: {:.1}% (encoding fidelity, not solve rate)",
                    twoafc_rate * 100.0
                ),
                format!("Mean pixel accuracy: {:.1}%", mean_pixel_acc * 100.0),
                "Gap between 2-AFC and strict = bundling noise, not generalization failure".into(),
            ],
        }
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "ARC-AGI Strict Grid Reconstruction",
            citation: "Chollet, F. (2019). On the Measure of Intelligence. arXiv:1911.01547",
            year: 2019,
            doi: Some("10.48550/arXiv.1911.01547"),
        })
    }
}

struct StrictTrialResult {
    strict_match: bool,
    twoafc_correct: bool,
    pixel_accuracy: f32,
    similarity: f32,
}

fn run_strict_trial(seed: u64, _dimension: usize) -> StrictTrialResult {
    let mut rng = seed;
    let xor_shift = |s: &mut u64| {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    };

    // Generate a random grid transformation task
    let grid_size = 3 + (xor_shift(&mut rng) % 4) as usize; // 3-6
    let num_colors = 4 + (xor_shift(&mut rng) % 4) as usize; // 4-7
    let encoder = BinaryGridEncoder::new(grid_size, grid_size, num_colors, seed);

    // Generate training pairs: color replacement (color A → color B)
    let color_a = (xor_shift(&mut rng) % num_colors as u64) as u8;
    let mut color_b = (xor_shift(&mut rng) % num_colors as u64) as u8;
    while color_b == color_a {
        color_b = (xor_shift(&mut rng) % num_colors as u64) as u8;
    }

    let num_train = 3;
    let mut train_pairs = Vec::new();
    for _ in 0..num_train {
        let input: Vec<Vec<u8>> = (0..grid_size)
            .map(|_| {
                (0..grid_size)
                    .map(|_| (xor_shift(&mut rng) % num_colors as u64) as u8)
                    .collect()
            })
            .collect();
        let output: Vec<Vec<u8>> = input
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&c| if c == color_a { color_b } else { c })
                    .collect()
            })
            .collect();
        train_pairs.push((input, output));
    }

    // Compute consensus rule from training pairs
    let rules: Vec<BinaryHV> = train_pairs
        .iter()
        .map(|(inp, out)| {
            let hv_in = encoder.encode_grid(inp);
            let hv_out = encoder.encode_grid(out);
            encoder.encode_rule(&hv_in, &hv_out)
        })
        .collect();
    let consensus = encoder.bundle_rules(&rules);

    // Generate test pair
    let test_input: Vec<Vec<u8>> = (0..grid_size)
        .map(|_| {
            (0..grid_size)
                .map(|_| (xor_shift(&mut rng) % num_colors as u64) as u8)
                .collect()
        })
        .collect();
    let test_output: Vec<Vec<u8>> = test_input
        .iter()
        .map(|row| {
            row.iter()
                .map(|&c| if c == color_a { color_b } else { c })
                .collect()
        })
        .collect();

    // Apply rule and decode
    let test_in_hv = encoder.encode_grid(&test_input);
    let predicted_hv = encoder.apply_rule(&test_in_hv, &consensus);

    // Strict: decode back to grid and compare pixel-perfect
    let predicted_grid = encoder.decode_grid(&predicted_hv, grid_size, grid_size);
    let pixel_accuracy = BinaryGridEncoder::grid_accuracy(&predicted_grid, &test_output);
    let strict_match = BinaryGridEncoder::grid_exact_match(&predicted_grid, &test_output);

    // 2-AFC: compare similarity to actual vs a fair (equally structured) distractor
    let test_out_hv = encoder.encode_grid(&test_output);
    let actual_sim = predicted_hv.similarity(&test_out_hv);
    let distractor_grid =
        fair_distractor_grid(&test_input, &test_output).unwrap_or_else(|| test_input.clone());
    let distractor_hv = encoder.encode_grid(&distractor_grid);
    let dist_sim = predicted_hv.similarity(&distractor_hv);
    let twoafc_correct = actual_sim > dist_sim;

    StrictTrialResult {
        strict_match,
        twoafc_correct,
        pixel_accuracy,
        similarity: actual_sim,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arc_strict_runs() {
        let bench = ArcStrictBenchmark;
        let config = BenchmarkConfig {
            trials_per_condition: 20,
            ..BenchmarkConfig::default()
        };
        let result = bench.run(&config);
        assert_eq!(result.benchmark, "Reasoning::ARC_Strict");

        let strict_rate = result.metrics["strict_solve_rate"].mean;
        let twoafc_rate = result.metrics["twoafc_accuracy"].mean;

        // 2-AFC should generally be higher than strict.
        // With few trials both can be low; allow 1-trial tolerance.
        let tolerance = 1.0 / config.trials_per_condition as f64;
        assert!(
            twoafc_rate >= strict_rate - tolerance,
            "2-AFC ({:.3}) should be >= strict ({:.3}) - tolerance ({:.3})",
            twoafc_rate,
            strict_rate,
            tolerance
        );
    }

    #[test]
    fn test_arc_strict_pixel_accuracy() {
        let result = run_strict_trial(42, 16384);
        // Pixel accuracy should be between 0 and 1
        assert!(result.pixel_accuracy >= 0.0 && result.pixel_accuracy <= 1.0);
    }
}
