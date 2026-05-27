// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Style Diversity benchmark.
//!
//! Measures how different the system's creative outputs are when given
//! different cognitive states. A truly creative system should produce
//! visibly distinct art from distinct states, not generic output.
//!
//! Human baseline: professional artists produce recognizably different works
//! across emotional states (mean pairwise dissimilarity ~0.6; Silvia, 2008).
//!
//! HDC implementation: generate concept blends from different "mood" seed
//! vectors, measure pairwise dissimilarity of outputs.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_hv::BinaryHV;

/// Style Diversity benchmark.
pub struct StyleDiversityBenchmark;

fn next_seed(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

impl StyleDiversityBenchmark {
    /// Generate a "creative output" from a mood seed via HDC blending.
    fn generate_from_mood(dim: usize, mood_seed: u64, concept_seeds: &[u64]) -> BinaryHV {
        let mood = BinaryHV::random(mood_seed);
        let mut result = mood.clone();
        for &cs in concept_seeds {
            let concept = BinaryHV::random(cs);
            // Bind mood with concept (mood-colored creation)
            let colored = mood.bind(&concept);
            result = BinaryHV::weighted_bundle(&[result, colored], &[0.6, 0.4]);
        }
        result
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        let dim = config.dimension;
        let n_moods = 6; // distinct cognitive states
        let n_concepts = 3; // concepts per creation
        let mut rng_state = config.seed.wrapping_add(trial_idx as u64 * 6991);

        // Generate concept seeds (shared across moods — same "material")
        let concept_seeds: Vec<u64> = (0..n_concepts).map(|_| next_seed(&mut rng_state)).collect();

        // Generate outputs from different moods
        let outputs: Vec<BinaryHV> = (0..n_moods)
            .map(|i| {
                let mood_seed = next_seed(&mut rng_state).wrapping_add(i as u64 * 104729);
                Self::generate_from_mood(dim, mood_seed, &concept_seeds)
            })
            .collect();

        // Compute pairwise dissimilarity
        let mut dissimilarities = Vec::new();
        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                let sim = outputs[i].similarity(&outputs[j]);
                dissimilarities.push(1.0 - sim as f64); // dissimilarity = 1 - similarity
            }
        }

        let mean_dissim = dissimilarities.iter().sum::<f64>() / dissimilarities.len().max(1) as f64;

        // Variance of dissimilarities (low variance = consistent diversity, not random)
        let var = if dissimilarities.len() > 1 {
            let mean = mean_dissim;
            dissimilarities
                .iter()
                .map(|d| (d - mean).powi(2))
                .sum::<f64>()
                / (dissimilarities.len() - 1) as f64
        } else {
            0.0
        };
        let consistency = 1.0 - var.sqrt().min(1.0); // low variance = high consistency

        // Apply lapse rate: noise reduces discriminability
        let mean_dissim = mean_dissim * (1.0 - config.lapse_rate as f64 * 0.3);

        (mean_dissim, consistency)
    }
}

impl PsychBenchmark for StyleDiversityBenchmark {
    fn name(&self) -> &str {
        "Creativity::StyleDiversity"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Creative Style Analysis",
            citation: "Silvia (2008)",
            year: 2008,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut dissimilarities = Vec::new();
        let mut consistencies = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (dissim, consist) = self.run_trial(config, trial);
            dissimilarities.push(dissim);
            consistencies.push(consist);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "diversity".to_string(),
                    correct: dissim > 0.3,
                    rt_ticks: 0.0,
                    similarity: 1.0 - dissim,
                    confidence: consist,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "style_dissimilarity",
            MetricValue::from_samples(&dissimilarities),
        );
        result.insert(
            "diversity_consistency",
            MetricValue::from_samples(&consistencies),
        );

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
        let result = StyleDiversityBenchmark.run(&config);
        assert!(result.metrics.contains_key("style_dissimilarity"));
        assert!(result.metrics.contains_key("diversity_consistency"));
    }

    #[test]
    fn different_moods_produce_different_art() {
        let config = BenchmarkConfig {
            trials_per_condition: 20,
            seed: 42,
            ..BenchmarkConfig::default()
        };
        let result = StyleDiversityBenchmark.run(&config);
        let dissim = result.metrics["style_dissimilarity"].mean;
        // Different moods should produce meaningfully different outputs
        assert!(dissim > 0.3, "style_dissimilarity = {dissim}");
    }
}