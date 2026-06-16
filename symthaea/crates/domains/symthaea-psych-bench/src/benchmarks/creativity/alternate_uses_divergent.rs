// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Divergent Thinking (Alternative Uses variant) benchmark.
//!
//! Extends the standard AUT by measuring semantic distance between generated
//! uses. Guilford (1967) identified four components of divergent thinking:
//! fluency, flexibility, originality, and elaboration. This benchmark focuses
//! on originality — how semantically distant are the generated uses from the
//! object's typical use?
//!
//! HDC implementation: generates candidate "uses" by exploring HDC space around
//! an object prototype. Semantic distance is measured via cosine dissimilarity
//! between each generated use-vector and the object prototype. Higher distance
//! indicates more original/creative thinking.
//!
//! Human baselines (Guilford, 1967; Silvia et al., 2008):
//! - originality_score: 0.45 (SD 0.15) — mean semantic distance of uses
//! - flexibility_score: 0.60 (SD 0.12) — proportion of uses in distinct categories
//! - elaboration_score: 0.35 (SD 0.12) — detail richness of uses

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Divergent Thinking benchmark (Alternative Uses originality variant).
pub struct DivergentThinkingBenchmark;

struct TrialResult {
    originality_score: f64,
    flexibility_score: f64,
    elaboration_score: f64,
}

impl DivergentThinkingBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("creativity", "divergent_thinking", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Object prototype (the thing we generate uses for)
        let object_proto = ContinuousHV::random(dim, seed.wrapping_add(100));

        // Category prototypes (8 semantic categories for flexibility)
        let n_categories = 8usize;
        let category_protos: Vec<ContinuousHV> = (0..n_categories)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(200 + i as u64)))
            .collect();

        // WM capacity influences how many uses can be generated
        let n_uses = config.working_memory_capacity.clamp(3, 12);

        let noise_scale = 0.3 + config.effective_noise() as f32 * 0.2;

        // Generate uses by exploring HDC space
        let mut use_hvs: Vec<ContinuousHV> = Vec::new();
        let mut distances = Vec::new();
        let mut categories_used = vec![false; n_categories];

        for i in 0..n_uses {
            xor_shift(&mut rng);

            // Generate a "use" by blending the object with a random exploration vector
            // and a category anchor. More creative exploration = higher blend with random.
            // Exploration increases across uses (warming up / incubation effect).
            let exploration = ContinuousHV::random(dim, rng);
            xor_shift(&mut rng);
            let cat_idx = rng as usize % n_categories;

            // Blend: object (familiar) + exploration (novel) + category (structure)
            // Higher exploration weights produce uses more distant from the object
            // prototype, matching human divergent thinking where later responses
            // are more original (serial order effect; Silvia et al. 2008).
            let explore_weight = 0.52 + (i as f32 / n_uses as f32) * 0.22;
            let cat_weight = 0.12;
            let obj_weight = 1.0 - explore_weight - cat_weight;

            let use_hv = ContinuousHV::weighted_bundle(
                &[&object_proto, &exploration, &category_protos[cat_idx]],
                &[obj_weight, explore_weight, cat_weight],
            );

            // Add encoding noise
            xor_shift(&mut rng);
            let noise_hv = ContinuousHV::random(dim, rng);
            let noisy_use = ContinuousHV::weighted_bundle(
                &[&use_hv, &noise_hv],
                &[1.0 - noise_scale * 0.1, noise_scale * 0.1],
            );

            // Measure originality: distance from object prototype
            let sim = noisy_use.similarity(&object_proto) as f64;
            let distance = (1.0 - sim).max(0.0);
            distances.push(distance);

            // Assign to nearest category for flexibility scoring
            let mut best_cat = 0;
            let mut best_cat_sim = f32::NEG_INFINITY;
            for (j, cat) in category_protos.iter().enumerate() {
                let s = noisy_use.similarity(cat);
                if s > best_cat_sim {
                    best_cat_sim = s;
                    best_cat = j;
                }
            }
            categories_used[best_cat] = true;

            use_hvs.push(noisy_use);
        }

        // Originality: mean semantic distance from object
        let originality_score = if !distances.is_empty() {
            distances.iter().sum::<f64>() / distances.len() as f64
        } else {
            0.0
        };

        // Flexibility: proportion of distinct categories used
        let categories_hit = categories_used.iter().filter(|&&x| x).count();
        let flexibility_score = categories_hit as f64 / n_categories as f64;

        // Elaboration: measured as inter-use distinctiveness
        // (how different are the uses from each other — higher = more elaborated)
        let mut pairwise_distances = Vec::new();
        for i in 0..use_hvs.len() {
            for j in (i + 1)..use_hvs.len() {
                let sim = use_hvs[i].similarity(&use_hvs[j]) as f64;
                pairwise_distances.push((1.0 - sim).max(0.0));
            }
        }
        let elaboration_score = if !pairwise_distances.is_empty() {
            pairwise_distances.iter().sum::<f64>() / pairwise_distances.len() as f64
        } else {
            0.0
        };

        TrialResult {
            originality_score,
            flexibility_score,
            elaboration_score,
        }
    }
}

impl PsychBenchmark for DivergentThinkingBenchmark {
    fn name(&self) -> &str {
        "Creativity::DivergentThinking"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Divergent Thinking / Alternative Uses",
            citation: "Guilford (1967); Silvia et al. (2008)",
            year: 1967,
            doi: Some("10.1037/h0046572"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut originalities = Vec::new();
        let mut flexibilities = Vec::new();
        let mut elaborations = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            originalities.push(r.originality_score);
            flexibilities.push(r.flexibility_score);
            elaborations.push(r.elaboration_score);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "divergent".to_string(),
                    correct: r.originality_score > 0.3,
                    rt_ticks: 0.0,
                    similarity: r.originality_score,
                    confidence: r.flexibility_score,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "originality_score",
            MetricValue::from_samples(&originalities),
        );
        result.insert(
            "flexibility_score",
            MetricValue::from_samples(&flexibilities),
        );
        result.insert(
            "elaboration_score",
            MetricValue::from_samples(&elaborations),
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
    fn test_divergent_thinking_runs() {
        let config = BenchmarkConfig::default();
        let result = DivergentThinkingBenchmark.run(&config);
        assert!(result.metrics.contains_key("originality_score"));
        assert!(result.metrics.contains_key("flexibility_score"));
        assert!(result.metrics.contains_key("elaboration_score"));
    }

    #[test]
    fn test_divergent_thinking_values() {
        let config = BenchmarkConfig::default();
        let result = DivergentThinkingBenchmark.run(&config);
        for (key, val) in &result.metrics {
            eprintln!("DT {key}: mean={:.4}, sd={:.4}", val.mean, val.std_dev);
        }
    }

    #[test]
    fn test_divergent_thinking_finite() {
        let config = BenchmarkConfig::default();
        let result = DivergentThinkingBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_originality_bounded() {
        let config = BenchmarkConfig::default();
        let result = DivergentThinkingBenchmark.run(&config);
        let orig = result.metrics["originality_score"].mean;
        assert!(
            orig >= 0.0 && orig <= 1.0,
            "originality {orig} out of bounds"
        );
    }
}
