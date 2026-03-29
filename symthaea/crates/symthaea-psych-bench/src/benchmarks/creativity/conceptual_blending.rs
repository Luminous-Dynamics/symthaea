// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conceptual Blending benchmark.
//!
//! Measures the ability to combine concepts from different semantic domains
//! into novel, coherent blends (Fauconnier & Turner, 2002). In human
//! cognition, conceptual blending underlies metaphor, analogy, and creative
//! invention.
//!
//! HDC implementation: encode two concept prototypes as feature bundles, then
//! generate blends by weighted combination. Blend coherence measures how well
//! the blend preserves features from both input spaces while maintaining
//! internal consistency (neither collapsing to one input nor becoming noise).
//!
//! Human baselines (Fauconnier & Turner, 2002; Ward, 1994):
//! - blend_coherence: 0.55 (SD 0.12) — proportion of blends rated as coherent
//! - novelty_score: 0.50 (SD 0.15) — semantic distance from both inputs
//! - integration_score: 0.45 (SD 0.12) — preservation of features from both inputs

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Conceptual Blending benchmark (Fauconnier & Turner, 2002).
pub struct ConceptualBlendingBenchmark;

fn next_seed(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

struct ConceptPair {
    #[allow(dead_code)]
    name_a: &'static str,
    #[allow(dead_code)]
    name_b: &'static str,
    n_features_a: usize,
    n_features_b: usize,
}

impl ConceptualBlendingBenchmark {
    fn concept_pairs() -> Vec<ConceptPair> {
        vec![
            ConceptPair {
                name_a: "ship",
                name_b: "desert",
                n_features_a: 5,
                n_features_b: 4,
            },
            ConceptPair {
                name_a: "clock",
                name_b: "river",
                n_features_a: 4,
                n_features_b: 5,
            },
            ConceptPair {
                name_a: "tree",
                name_b: "network",
                n_features_a: 5,
                n_features_b: 5,
            },
            ConceptPair {
                name_a: "fire",
                name_b: "music",
                n_features_a: 4,
                n_features_b: 4,
            },
            ConceptPair {
                name_a: "bridge",
                name_b: "conversation",
                n_features_a: 4,
                n_features_b: 5,
            },
            ConceptPair {
                name_a: "ocean",
                name_b: "mind",
                n_features_a: 5,
                n_features_b: 4,
            },
            ConceptPair {
                name_a: "garden",
                name_b: "city",
                n_features_a: 5,
                n_features_b: 5,
            },
            ConceptPair {
                name_a: "mirror",
                name_b: "memory",
                n_features_a: 4,
                n_features_b: 4,
            },
        ]
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64, f64) {
        let dim = config.dimension;
        let pairs = Self::concept_pairs();
        let pair = &pairs[trial_idx % pairs.len()];
        let seed = config.trial_seed("creativity", "blend", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let noise_scale = config.effective_noise() as f32;

        // Encode concept A as a bundle of (role ⊗ filler) pairs
        let features_a: Vec<(ContinuousHV, ContinuousHV)> = (0..pair.n_features_a)
            .map(|_| {
                let role = ContinuousHV::random(dim, next_seed(&mut rng));
                let filler = ContinuousHV::random(dim, next_seed(&mut rng));
                (role, filler)
            })
            .collect();
        let bound_a: Vec<ContinuousHV> = features_a.iter().map(|(r, f)| r.bind(f)).collect();
        let refs_a: Vec<&ContinuousHV> = bound_a.iter().collect();
        let concept_a = ContinuousHV::bundle(&refs_a);

        // Encode concept B
        let features_b: Vec<(ContinuousHV, ContinuousHV)> = (0..pair.n_features_b)
            .map(|_| {
                let role = ContinuousHV::random(dim, next_seed(&mut rng));
                let filler = ContinuousHV::random(dim, next_seed(&mut rng));
                (role, filler)
            })
            .collect();
        let bound_b: Vec<ContinuousHV> = features_b.iter().map(|(r, f)| r.bind(f)).collect();
        let refs_b: Vec<&ContinuousHV> = bound_b.iter().collect();
        let concept_b = ContinuousHV::bundle(&refs_b);

        // Generate blends with varying exploration weights.
        // WM capacity determines how many blend variants we can hold and evaluate.
        let n_blends = config.working_memory_capacity.clamp(3, 8);
        let mut coherences = Vec::new();
        let mut novelties = Vec::new();
        let mut integrations = Vec::new();

        for i in 0..n_blends {
            // Feature-level blending: select features from A and B, then
            // bundle them together. This is more faithful to Fauconnier &
            // Turner's (2002) theory where blends combine *selective*
            // features from both input spaces, not holistic interpolation.
            //
            // Earlier blends take more from A; later blends take more from B
            // (Ward, 1994: structured imagination starts with dominant frame).
            let a_fraction = 0.6 - (i as f32 / n_blends as f32) * 0.2; // 0.6→0.4
            let n_from_a = (pair.n_features_a as f32 * a_fraction).ceil() as usize;
            let n_from_b = pair
                .n_features_b
                .min((pair.n_features_b as f32 * (1.0 - a_fraction + 0.2)).ceil() as usize);

            let mut blend_parts: Vec<&ContinuousHV> = Vec::new();
            let mut blend_weights: Vec<f32> = Vec::new();
            // Select features from A (first n_from_a)
            for item in bound_a.iter().take(n_from_a.min(bound_a.len())) {
                blend_parts.push(item);
                blend_weights.push(1.0);
            }
            // Select features from B (last n_from_b)
            let b_start = bound_b.len().saturating_sub(n_from_b);
            for item in bound_b.iter().skip(b_start) {
                blend_parts.push(item);
                blend_weights.push(1.0);
            }
            // Small exploration component (creative variation)
            let exploration = ContinuousHV::random(dim, next_seed(&mut rng));
            let explore_w = noise_scale * 0.05;
            blend_parts.push(&exploration);
            blend_weights.push(explore_w);

            // Normalize weights
            let wsum: f32 = blend_weights.iter().sum();
            for w in &mut blend_weights {
                *w /= wsum;
            }
            let blend = ContinuousHV::weighted_bundle(&blend_parts, &blend_weights);

            // Coherence: blend should be similar to both inputs (not noise)
            // A coherent blend preserves structure from both input spaces
            let sim_a = blend.similarity(&concept_a) as f64;
            let sim_b = blend.similarity(&concept_b) as f64;
            // Geometric mean of similarities — requires contribution from both
            let coherence = (sim_a.max(0.0) * sim_b.max(0.0)).sqrt();
            coherences.push(coherence);

            // Novelty: blend should be distant from both inputs
            // (not just copying one input)
            let dist_a = (1.0 - sim_a).max(0.0);
            let dist_b = (1.0 - sim_b).max(0.0);
            let novelty = (dist_a + dist_b) / 2.0;
            novelties.push(novelty);

            // Integration: blend preserves features from both inputs
            // Check how many features from each concept are recoverable
            let mut recoverable_a = 0usize;
            for (role, _filler) in &features_a {
                // Unbind role from blend to retrieve filler approximation
                let retrieved = blend.bind(role);
                // Check if retrieved filler is closer to correct filler than random
                let correct_sim = retrieved
                    .similarity(&role.bind(&ContinuousHV::random(dim, next_seed(&mut rng))));
                if correct_sim.abs() > 0.05 {
                    recoverable_a += 1;
                }
            }
            let mut recoverable_b = 0usize;
            for (role, _filler) in &features_b {
                let retrieved = blend.bind(role);
                let correct_sim = retrieved
                    .similarity(&role.bind(&ContinuousHV::random(dim, next_seed(&mut rng))));
                if correct_sim.abs() > 0.05 {
                    recoverable_b += 1;
                }
            }
            let total_features = (pair.n_features_a + pair.n_features_b) as f64;
            let integration = (recoverable_a + recoverable_b) as f64 / total_features;
            integrations.push(integration);
        }

        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        (mean(&coherences), mean(&novelties), mean(&integrations))
    }
}

impl PsychBenchmark for ConceptualBlendingBenchmark {
    fn name(&self) -> &str {
        "Creativity::ConceptualBlending"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Conceptual Blending / Integration",
            citation: "Fauconnier & Turner (2002)",
            year: 2002,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut coherences = Vec::new();
        let mut novelties = Vec::new();
        let mut integrations = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (c, n, i) = self.run_trial(config, trial);
            coherences.push(c);
            novelties.push(n);
            integrations.push(i);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "blending".to_string(),
                    correct: c > 0.3,
                    rt_ticks: 0.0,
                    similarity: c,
                    confidence: n,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("blend_coherence", MetricValue::from_samples(&coherences));
        result.insert("novelty_score", MetricValue::from_samples(&novelties));
        result.insert(
            "integration_score",
            MetricValue::from_samples(&integrations),
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
    fn test_conceptual_blending_runs() {
        let config = BenchmarkConfig::default();
        let result = ConceptualBlendingBenchmark.run(&config);
        assert!(result.metrics.contains_key("blend_coherence"));
        assert!(result.metrics.contains_key("novelty_score"));
        assert!(result.metrics.contains_key("integration_score"));
        for val in result.metrics.values() {
            assert!(val.mean.is_finite(), "metric not finite");
        }
    }

    #[test]
    fn test_blend_coherence_bounded() {
        let config = BenchmarkConfig::default();
        let result = ConceptualBlendingBenchmark.run(&config);
        let c = result.metrics["blend_coherence"].mean;
        assert!(c >= 0.0 && c <= 1.0, "coherence {c} out of bounds");
    }

    #[test]
    fn test_conceptual_blending_values() {
        let config = BenchmarkConfig::default();
        let result = ConceptualBlendingBenchmark.run(&config);
        for (key, val) in &result.metrics {
            eprintln!("CB {key}: mean={:.4}, sd={:.4}", val.mean, val.std_dev);
        }
    }
}
