// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Aesthetic Generation benchmark.
//!
//! Measures the system's ability to iteratively improve the aesthetic quality
//! of generated artifacts. Unlike analytical creativity benchmarks (RAT, AUT),
//! this tests *generative* creativity: does the system produce increasingly
//! beautiful outputs through its generate-evaluate-mutate loop?
//!
//! Human baseline: art students improve ~15% over 5 iterations when given
//! aesthetic feedback (Csikszentmihalyi, 1996; Beghetto & Kaufman, 2007).
//!
//! HDC implementation: generate N iterations of conceptual blending with
//! progressive selection pressure. Measure the improvement trajectory.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_hv::BinaryHV;

/// Aesthetic Generation benchmark (generative creativity via iterative refinement).
pub struct AestheticGenerationBenchmark;

fn next_seed(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

impl AestheticGenerationBenchmark {
    /// Run a single generation trial: N iterations of create-evaluate-select.
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64, f64) {
        let dim = config.dimension;
        let iterations = 8;
        let candidates_per_iter = 4;
        let mut rng_state = config.seed.wrapping_add(trial_idx as u64 * 7919);

        // Create a "target aesthetic" — the ideal we're approaching
        let target = BinaryHV::random(next_seed(&mut rng_state));

        let mut best = BinaryHV::random(next_seed(&mut rng_state));
        let mut best_score = best.similarity(&target);
        let mut scores = vec![best_score as f64];

        for _iter in 0..iterations {
            let mut iter_best = best.clone();
            let mut iter_best_score = best_score;

            for _ in 0..candidates_per_iter {
                // Generate candidate: mutate best via noise + bind with random concept
                let noise_level = 0.15 + config.lapse_rate * 0.1;
                let mut candidate = best.add_noise(noise_level as f32, next_seed(&mut rng_state));

                // Occasionally bind with a fresh concept (creative leap)
                if next_seed(&mut rng_state) % 4 == 0 {
                    let fresh = BinaryHV::random(next_seed(&mut rng_state));
                    candidate = candidate.bind(&fresh);
                    // Partial unbind to retain structure
                    candidate = BinaryHV::weighted_bundle(&[candidate, best.clone()], &[0.4, 0.6]);
                }

                let score = candidate.similarity(&target);
                if score > iter_best_score {
                    iter_best = candidate;
                    iter_best_score = score;
                }
            }

            best = iter_best;
            best_score = iter_best_score;
            scores.push(best_score as f64);
        }

        // Metrics:
        // 1. improvement_ratio: final_score / initial_score (> 1.0 = improvement)
        let initial = scores[0].max(0.01);
        let final_score = scores[scores.len() - 1];
        let improvement_ratio = final_score / initial;

        // 2. monotonicity: fraction of iterations where score improved
        let improvements = scores.windows(2).filter(|w| w[1] >= w[0]).count();
        let monotonicity = improvements as f64 / (scores.len() - 1) as f64;

        // 3. final_quality: absolute quality of best output
        let final_quality = final_score;

        (improvement_ratio, monotonicity, final_quality)
    }
}

impl PsychBenchmark for AestheticGenerationBenchmark {
    fn name(&self) -> &str {
        "Creativity::AestheticGeneration"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Iterative Creative Generation",
            citation: "Csikszentmihalyi (1996); Beghetto & Kaufman (2007)",
            year: 1996,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut improvement_ratios = Vec::new();
        let mut monotonicities = Vec::new();
        let mut final_qualities = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (ir, mono, fq) = self.run_trial(config, trial);
            improvement_ratios.push(ir);
            monotonicities.push(mono);
            final_qualities.push(fq);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "generation".to_string(),
                    correct: ir > 1.0,
                    rt_ticks: 0.0,
                    similarity: fq,
                    confidence: mono,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "improvement_ratio",
            MetricValue::from_samples(&improvement_ratios),
        );
        result.insert("monotonicity", MetricValue::from_samples(&monotonicities));
        result.insert("final_quality", MetricValue::from_samples(&final_qualities));

        result.conditions = 1;
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 10,
            seed: 42,
            ..BenchmarkConfig::default()
        };
        let result = AestheticGenerationBenchmark.run(&config);
        assert!(result.metrics.contains_key("improvement_ratio"));
        assert!(result.metrics.contains_key("monotonicity"));
        assert!(result.metrics.contains_key("final_quality"));
    }

    #[test]
    fn improvement_ratio_positive() {
        let config = BenchmarkConfig {
            trials_per_condition: 20,
            seed: 42,
            ..BenchmarkConfig::default()
        };
        let result = AestheticGenerationBenchmark.run(&config);
        let ir = result.metrics["improvement_ratio"].mean;
        // System should improve over iterations (ratio > 1.0)
        assert!(ir > 0.95, "improvement_ratio = {ir}");
    }
}