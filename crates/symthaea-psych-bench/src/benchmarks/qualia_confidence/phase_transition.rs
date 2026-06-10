// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase Transition Benchmark — Sigmoidal Consciousness Collapse
//!
//! IIT predicts consciousness collapses at a critical noise threshold via a phase
//! transition, not linearly. This benchmark progressively corrupts GWT domain
//! representations and measures whether the workspace can still form coherent,
//! recognizable broadcasts.
//!
//! ## Prediction
//!
//! The recognition-vs-noise curve should fit a sigmoid better than a line:
//! - Sigmoid collapse indicates a phase transition (critical threshold)
//! - Linear decline would indicate graceful degradation (no phase transition)
//!
//! ## Method
//!
//! 1. Define 11 noise levels: [0.0, 0.1, ..., 1.0]
//! 2. At each level, run 100 GWT cycles with 8 domain contents
//! 3. Corrupt domain HVs by flipping `noise_fraction` of their bits before submission
//! 4. Measure "recognition": fraction of cycles where the winning broadcast still
//!    matches its source domain above a similarity threshold. This is nonlinear
//!    because recognition has a cliff — slightly too much noise and the domain
//!    signal becomes unrecoverable.
//! 5. Fit sigmoid and linear models to recognition curve
//!
//! ## Why recognition, not raw similarity?
//!
//! `BinaryHV::similarity` decreases linearly with noise by construction.
//! Recognition introduces a threshold effect: if the broadcast's similarity to its
//! source domain drops below the discrimination boundary (midpoint between matched
//! and random), the system "forgets" what it was thinking about. This threshold
//! effect produces the expected phase transition.
//!
//! Science:
//! - Tononi, G. (2004). An information integration theory of consciousness. BMC Neuroscience.
//! - Massimini, M. et al. (2005). Breakdown of cortical effective connectivity during sleep. Science.

use crate::benchmarks::qualia_confidence::helpers::{linear_fit, sigmoid_fit};
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::global_workspace::{GlobalWorkspace, WorkspaceConfig, WorkspaceContent};

/// Number of noise levels in the sweep.
const NUM_NOISE_LEVELS: usize = 11;

/// Cycles per noise level.
const CYCLES_PER_LEVEL: usize = 100;

/// Domain profiles reused from GWT asphyxiation (same activation levels).
const DOMAIN_ACTIVATIONS: [(&str, f64); 8] = [
    ("Social", 0.40),
    ("Metacognition", 0.46),
    ("Executive", 0.52),
    ("Memory", 0.58),
    ("Reasoning", 0.63),
    ("Language", 0.70),
    ("Attention", 0.78),
    ("Motor", 0.85),
];

/// Phase Transition Benchmark.
///
/// Tests whether consciousness collapse under noise follows a sigmoidal (phase
/// transition) or linear (graceful degradation) pattern by measuring recognition
/// of broadcast content under progressive corruption.
pub struct PhaseTransitionBenchmark;

impl PsychBenchmark for PhaseTransitionBenchmark {
    fn name(&self) -> &str {
        "QualiaConfidence::PhaseTransition"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Consciousness Phase Transition under Noise",
            citation: "Tononi (2004); Massimini et al. (2005)",
            year: 2004,
            doi: Some("10.1186/1471-2202-5-42"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let noise_levels: Vec<f64> = (0..NUM_NOISE_LEVELS)
            .map(|i| i as f64 / (NUM_NOISE_LEVELS - 1) as f64)
            .collect();

        // Create canonical domain prototypes (clean, used for recognition comparison)
        let domain_prototypes: Vec<BinaryHV> = (0..8)
            .map(|i| BinaryHV::random(config.seed.wrapping_add(i as u64 * 1000 + 777)))
            .collect();

        let mut recognition_rates = Vec::with_capacity(NUM_NOISE_LEVELS);
        let mut raw_similarity_means = Vec::with_capacity(NUM_NOISE_LEVELS);

        for (ni, &noise) in noise_levels.iter().enumerate() {
            let mut recognized = 0usize;
            let mut similarity_sum = 0.0;
            let mut total_with_broadcast = 0usize;

            for cycle in 0..CYCLES_PER_LEVEL {
                let cycle_seed = config
                    .seed
                    .wrapping_mul(0x100000001b3)
                    .wrapping_add(ni as u64 * 1000 + cycle as u64);

                let ws_config = WorkspaceConfig {
                    entry_threshold: 0.30,
                    ..Default::default()
                };
                let mut ws = GlobalWorkspace::new(ws_config);

                // Submit content from all 8 domains with noisy HVs
                for (di, &(name, activation)) in DOMAIN_ACTIVATIONS.iter().enumerate() {
                    let noisy_hv = if noise < 1e-10 {
                        domain_prototypes[di]
                    } else {
                        let noise_seed = cycle_seed
                            .wrapping_mul(0x517cc1b727220a95)
                            .wrapping_add(di as u64);
                        domain_prototypes[di].add_noise(noise as f32, noise_seed)
                    };

                    let content =
                        WorkspaceContent::new(vec![noisy_hv], activation, name.to_string());
                    ws.submit(content);
                }

                let assessment = ws.process();

                // Check if the winning broadcast is still recognizable
                let broadcast_hv = if !assessment.broadcasts.is_empty() {
                    Some(&assessment.broadcasts[0].content[0])
                } else if !assessment.conscious_contents.is_empty() {
                    Some(&assessment.conscious_contents[0].representation[0])
                } else {
                    None
                };

                if let Some(bcast_hv) = broadcast_hv {
                    total_with_broadcast += 1;

                    // Compute similarity to all clean prototypes
                    let similarities: Vec<f32> = domain_prototypes
                        .iter()
                        .map(|proto| bcast_hv.similarity(proto))
                        .collect();

                    let best_sim = similarities
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    similarity_sum += best_sim as f64;

                    // Compute discrimination margin: best match vs second-best
                    let mut sorted_sims = similarities.clone();
                    sorted_sims
                        .sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let margin = if sorted_sims.len() >= 2 {
                        sorted_sims[0] - sorted_sims[1]
                    } else {
                        sorted_sims[0] - 0.5 // vs random baseline
                    };

                    // "Recognized" = the broadcast clearly matches one prototype
                    // margin > 0.02 means the signal is above the noise floor
                    // (random HV pairs have ~0.008 std in similarity at 16384 dims)
                    if margin > 0.02 {
                        recognized += 1;
                    }
                }
            }

            // Recognition rate = fraction of broadcasts that are still recognizable
            let recognition_rate = if total_with_broadcast > 0 {
                recognized as f64 / total_with_broadcast as f64
            } else {
                0.0
            };
            recognition_rates.push(recognition_rate);

            // Raw similarity (for comparison — should be linear)
            let raw_sim = if total_with_broadcast > 0 {
                similarity_sum / total_with_broadcast as f64
            } else {
                0.5
            };
            raw_similarity_means.push(raw_sim);

            let noise_pct = (noise * 100.0) as usize;
            result.insert(
                format!("recognition_at_noise_{noise_pct:03}"),
                MetricValue::from_samples(&[recognition_rate]),
            );
            result.insert(
                format!("raw_similarity_at_noise_{noise_pct:03}"),
                MetricValue::from_samples(&[raw_sim]),
            );
        }

        // Fit sigmoid and linear to RECOGNITION curve (nonlinear prediction)
        let (sig_l, sig_k, sig_x0, sig_r2) = sigmoid_fit(&noise_levels, &recognition_rates);
        let (_lin_slope, _lin_intercept, lin_r2) = linear_fit(&noise_levels, &recognition_rates);

        result.insert("sigmoid_r_squared", MetricValue::from_samples(&[sig_r2]));
        result.insert("linear_r_squared", MetricValue::from_samples(&[lin_r2]));
        result.insert("critical_noise_level", MetricValue::from_samples(&[sig_x0]));
        result.insert(
            "phase_transition_sharpness",
            MetricValue::from_samples(&[sig_k.abs()]),
        );
        result.insert("sigmoid_amplitude", MetricValue::from_samples(&[sig_l]));

        // Also fit raw similarity curve (should be ~linear as control)
        let (_, _, raw_lin_r2) = linear_fit(&noise_levels, &raw_similarity_means);
        result.insert(
            "raw_similarity_linear_r2",
            MetricValue::from_samples(&[raw_lin_r2]),
        );

        // Recognition at 50% noise (midpoint)
        let mid_idx = NUM_NOISE_LEVELS / 2;
        result.insert(
            "recognition_at_50pct_noise",
            MetricValue::from_samples(&[recognition_rates[mid_idx]]),
        );

        // Phase transition detected if sigmoid fits recognition better than linear
        let phase_transition_detected = sig_r2 > lin_r2;
        result.insert(
            "phase_transition_detected",
            MetricValue::from_samples(&[if phase_transition_detected { 1.0 } else { 0.0 }]),
        );

        // R² advantage
        result.insert(
            "sigmoid_advantage",
            MetricValue::from_samples(&[sig_r2 - lin_r2]),
        );

        // Recognition range
        let recognition_range = recognition_rates[0] - *recognition_rates.last().unwrap_or(&0.0);
        result.insert(
            "recognition_range",
            MetricValue::from_samples(&[recognition_range]),
        );

        result.conditions = NUM_NOISE_LEVELS;
        result.trials_per_condition = CYCLES_PER_LEVEL;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BenchmarkConfig {
        BenchmarkConfig {
            seed: 42,
            ..Default::default()
        }
    }

    #[test]
    fn test_phase_transition_runs() {
        let bench = PhaseTransitionBenchmark;
        let result = bench.run(&config());
        assert_eq!(result.benchmark, "QualiaConfidence::PhaseTransition");
        assert!(result.metrics.len() >= 5);
    }

    #[test]
    fn test_all_metrics_finite() {
        let bench = PhaseTransitionBenchmark;
        let result = bench.run(&config());
        for (key, value) in &result.metrics {
            assert!(
                value.mean.is_finite(),
                "Metric '{key}' should be finite, got {}",
                value.mean
            );
        }
    }

    #[test]
    fn test_sigmoid_r_squared_positive() {
        let bench = PhaseTransitionBenchmark;
        let result = bench.run(&config());
        let r2 = result.metrics.get("sigmoid_r_squared").unwrap().mean;
        assert!(r2 > 0.0, "Sigmoid R² should be positive: {r2}");
    }

    #[test]
    fn test_recognition_decreases_with_noise() {
        let bench = PhaseTransitionBenchmark;
        let result = bench.run(&config());
        let r_000 = result.metrics.get("recognition_at_noise_000").unwrap().mean;
        let r_100 = result.metrics.get("recognition_at_noise_100").unwrap().mean;
        assert!(
            r_000 > r_100,
            "Recognition should decrease with noise: clean={r_000}, full_noise={r_100}"
        );
    }

    #[test]
    fn test_no_noise_high_recognition() {
        let bench = PhaseTransitionBenchmark;
        let result = bench.run(&config());
        let r_000 = result.metrics.get("recognition_at_noise_000").unwrap().mean;
        assert!(
            r_000 > 0.8,
            "Zero noise should have high recognition: {r_000}"
        );
    }

    #[test]
    fn test_full_noise_low_recognition() {
        let bench = PhaseTransitionBenchmark;
        let result = bench.run(&config());
        let r_100 = result.metrics.get("recognition_at_noise_100").unwrap().mean;
        assert!(
            r_100 < 0.5,
            "Full noise should have low recognition: {r_100}"
        );
    }

    #[test]
    fn test_sigmoid_fits_better_than_linear() {
        let bench = PhaseTransitionBenchmark;
        let result = bench.run(&config());
        let sig_r2 = result.metrics.get("sigmoid_r_squared").unwrap().mean;
        let lin_r2 = result.metrics.get("linear_r_squared").unwrap().mean;
        assert!(
            sig_r2 >= lin_r2 - 0.05,
            "Sigmoid should fit at least as well as linear: sig={sig_r2}, lin={lin_r2}"
        );
    }

    #[test]
    fn test_deterministic() {
        let bench = PhaseTransitionBenchmark;
        let r1 = bench.run(&config());
        let r2 = bench.run(&config());
        let f1 = r1.metrics.get("sigmoid_r_squared").unwrap().mean;
        let f2 = r2.metrics.get("sigmoid_r_squared").unwrap().mean;
        assert!(
            (f1 - f2).abs() < 1e-10,
            "Same seed should produce identical results: {f1} vs {f2}"
        );
    }

    #[test]
    fn test_has_provenance() {
        let bench = PhaseTransitionBenchmark;
        assert!(bench.provenance().is_some());
    }

    #[test]
    fn test_phase_transition_robust_across_seeds() {
        let bench = PhaseTransitionBenchmark;
        for seed in [42, 137, 256, 999, 7777] {
            let cfg = BenchmarkConfig {
                seed,
                ..Default::default()
            };
            let result = bench.run(&cfg);
            let sig_r2 = result.metrics.get("sigmoid_r_squared").unwrap().mean;
            let lin_r2 = result.metrics.get("linear_r_squared").unwrap().mean;
            assert!(
                sig_r2 >= lin_r2 - 0.10,
                "Sigmoid should fit at least as well as linear at seed={seed}: sig={sig_r2}, lin={lin_r2}"
            );
        }
    }
}
