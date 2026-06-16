// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Feature Conjunction Search benchmark.
//!
//! Tests the ability to detect targets defined by a conjunction of features
//! (e.g., red AND vertical) versus targets defined by a single feature.
//! Feature Integration Theory (Treisman & Gelade, 1980) predicts that
//! conjunction search requires serial attention-mediated binding, producing
//! set-size-dependent RT slopes, whereas feature search is parallel/pop-out.
//!
//! HDC implementation: encodes stimuli as role-filler bindings (color x shape).
//! Conjunction targets require matching BOTH bindings simultaneously, while
//! feature targets match a single dimension. The conjunction cost measures
//! the accuracy gap between feature and conjunction conditions.
//!
//! Human baselines (Treisman & Gelade, 1980; Wolfe, 1998):
//! - conjunction_accuracy: 0.82 (SD 0.10) — conjunction target detection
//! - feature_accuracy: 0.95 (SD 0.04) — single-feature target detection
//! - conjunction_cost: 0.13 (SD 0.06) — feature_accuracy minus conjunction_accuracy

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Feature Conjunction Search benchmark.
pub struct FeatureConjunctionBenchmark;

struct TrialResult {
    conjunction_accuracy: f64,
    feature_accuracy: f64,
    conjunction_cost: f64,
}

impl FeatureConjunctionBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("binding", "feature_conjunction", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Feature prototypes: 6 colors, 6 shapes
        // Larger feature vocabularies produce more orthogonal HDC role-filler
        // bindings, reducing cross-talk in the bundled scene representation
        // (Kanerva, 2009). With 4 features per dimension, distractors sharing
        // one feature have only 3 alternatives for the other dimension, creating
        // high overlap. With 6, there are 5 alternatives, spreading the bundle
        // energy more uniformly and improving target discriminability.
        let n_colors = 6usize;
        let n_shapes = 6usize;
        let color_hvs: Vec<ContinuousHV> = (0..n_colors)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(100 + i as u64)))
            .collect();
        let shape_hvs: Vec<ContinuousHV> = (0..n_shapes)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(200 + i as u64)))
            .collect();

        // Role binders for color and shape dimensions
        let role_color = ContinuousHV::random(dim, seed.wrapping_add(1000));
        let role_shape = ContinuousHV::random(dim, seed.wrapping_add(1001));

        // Base noise 0.012: with 6 feature prototypes (up from 4), the bundle
        // representation is cleaner due to more orthogonal role-filler bindings.
        // The reduced base noise compensates for the inherent difficulty of
        // conjunction search while maintaining the set-size effect.
        let noise_frac =
            0.012 + config.time_pressure as f32 * 0.08 + config.effective_noise() as f32 * 0.05;

        let set_sizes = [4usize, 8, 12];
        let trials_per_size = 20;

        // --- Conjunction search: find the target defined by BOTH color AND shape ---
        // Target: color_hvs[0] + shape_hvs[0]
        let target_color = 0usize;
        let target_shape = 0usize;

        let mut conj_correct = 0u32;
        let mut conj_total = 0u32;

        for &set_size in &set_sizes {
            for _trial in 0..trials_per_size {
                // Build display: one target + (set_size-1) distractors
                // Conjunction distractors share one feature with target
                let mut display_hvs: Vec<ContinuousHV> = Vec::new();

                // Target item: bind both features
                let target_hv = color_hvs[target_color]
                    .bind(&role_color)
                    .bind(&shape_hvs[target_shape].bind(&role_shape));
                display_hvs.push(target_hv.clone());

                // Distractors: share color OR shape but not both
                for d in 0..(set_size - 1) {
                    xor_shift(&mut rng);
                    let share_color = d % 2 == 0;
                    let d_color = if share_color {
                        target_color
                    } else {
                        1 + (rng as usize % (n_colors - 1))
                    };
                    let d_shape = if !share_color {
                        target_shape
                    } else {
                        1 + (rng as usize % (n_shapes - 1))
                    };
                    let d_hv = color_hvs[d_color]
                        .bind(&role_color)
                        .bind(&shape_hvs[d_shape].bind(&role_shape));
                    display_hvs.push(d_hv);
                }

                // Add noise to the display memory
                let display_refs: Vec<&ContinuousHV> = display_hvs.iter().collect();
                let scene = ContinuousHV::bundle(&display_refs);
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                let noisy_scene = ContinuousHV::weighted_bundle(
                    &[&scene, &noise_hv],
                    &[1.0 - noise_frac, noise_frac],
                );

                // Query: does target exist in scene? Compare target to scene
                let target_sim = noisy_scene.similarity(&target_hv);
                // Compare random non-target to scene as foil
                xor_shift(&mut rng);
                let foil_color = 1 + (rng as usize % (n_colors - 1));
                let foil_shape = 1 + (rng as usize % (n_shapes - 1));
                let foil_hv = color_hvs[foil_color]
                    .bind(&role_color)
                    .bind(&shape_hvs[foil_shape].bind(&role_shape));
                let foil_sim = noisy_scene.similarity(&foil_hv);

                conj_total += 1;
                if target_sim > foil_sim {
                    conj_correct += 1;
                }
            }
        }

        // --- Feature search: find target by single feature (pop-out) ---
        let mut feat_correct = 0u32;
        let mut feat_total = 0u32;

        for &set_size in &set_sizes {
            for _trial in 0..trials_per_size {
                // Target: unique color among distractors of different color
                let target_feat_hv = color_hvs[target_color].bind(&role_color);

                let mut display_hvs: Vec<ContinuousHV> = Vec::new();
                display_hvs.push(target_feat_hv.clone());

                // Distractors: all same non-target color, various shapes
                for d in 0..(set_size - 1) {
                    xor_shift(&mut rng);
                    let d_color = 1 + (d % (n_colors - 1));
                    let d_hv = color_hvs[d_color].bind(&role_color);
                    display_hvs.push(d_hv);
                }

                let display_refs: Vec<&ContinuousHV> = display_hvs.iter().collect();
                let scene = ContinuousHV::bundle(&display_refs);
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                let noisy_scene = ContinuousHV::weighted_bundle(
                    &[&scene, &noise_hv],
                    &[1.0 - noise_frac, noise_frac],
                );

                let target_sim = noisy_scene.similarity(&target_feat_hv);
                xor_shift(&mut rng);
                let foil_color = 1 + (rng as usize % (n_colors - 1));
                let foil_hv = color_hvs[foil_color].bind(&role_color);
                let foil_sim = noisy_scene.similarity(&foil_hv);

                feat_total += 1;
                if target_sim > foil_sim {
                    feat_correct += 1;
                }
            }
        }

        let conjunction_accuracy = conj_correct as f64 / conj_total.max(1) as f64;
        let feature_accuracy = feat_correct as f64 / feat_total.max(1) as f64;
        let conjunction_cost = (feature_accuracy - conjunction_accuracy).max(0.0);

        TrialResult {
            conjunction_accuracy,
            feature_accuracy,
            conjunction_cost,
        }
    }
}

impl PsychBenchmark for FeatureConjunctionBenchmark {
    fn name(&self) -> &str {
        "Binding::FeatureConjunction"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Feature Conjunction Search",
            citation: "Treisman & Gelade (1980)",
            year: 1980,
            doi: Some("10.1016/0010-0285(80)90005-5"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut conj_accs = Vec::new();
        let mut feat_accs = Vec::new();
        let mut costs = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            conj_accs.push(r.conjunction_accuracy);
            feat_accs.push(r.feature_accuracy);
            costs.push(r.conjunction_cost);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "conjunction".to_string(),
                    correct: r.conjunction_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.conjunction_accuracy,
                    confidence: 1.0 - r.conjunction_cost,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "conjunction_accuracy",
            MetricValue::from_samples(&conj_accs),
        );
        result.insert("feature_accuracy", MetricValue::from_samples(&feat_accs));
        result.insert("conjunction_cost", MetricValue::from_samples(&costs));

        result.conditions = 6; // 3 set sizes x 2 conditions
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
    fn test_feature_conjunction_runs() {
        let config = BenchmarkConfig::default();
        let result = FeatureConjunctionBenchmark.run(&config);
        assert!(result.metrics.contains_key("conjunction_accuracy"));
        assert!(result.metrics.contains_key("feature_accuracy"));
        assert!(result.metrics.contains_key("conjunction_cost"));
    }

    #[test]
    fn test_feature_conjunction_finite() {
        let config = BenchmarkConfig::default();
        let result = FeatureConjunctionBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_accuracy_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = FeatureConjunctionBenchmark.run(&config);
        let conj = result.metrics["conjunction_accuracy"].mean;
        let feat = result.metrics["feature_accuracy"].mean;
        assert!(
            conj >= 0.0 && conj <= 1.0,
            "conjunction accuracy {conj} out of bounds"
        );
        assert!(
            feat >= 0.0 && feat <= 1.0,
            "feature accuracy {feat} out of bounds"
        );
    }
}
