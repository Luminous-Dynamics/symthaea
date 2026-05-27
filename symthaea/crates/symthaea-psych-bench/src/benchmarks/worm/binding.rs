// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Feature binding task.
//!
//! Tests whether WM correctly binds features (color+shape) together,
//! vs remembering features independently. Compares binding accuracy
//! against feature-only accuracy (partial-feature lure detection).

use crate::adapter::StimulusAdapter;
use crate::adapter::spatial::{VisualObject, VisualObjectAdapter};
use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::{WmConfig, WorkingMemory};
use std::collections::BTreeMap;

use symthaea_core::hdc::ContinuousHV;

/// Binding benchmark testing feature-conjunction WM.
pub struct BindingBenchmark;

impl BindingBenchmark {
    /// Run a single trial.
    /// Returns (binding_correct, feature_only_correct, rt_ticks).
    fn run_trial(
        &self,
        set_size: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64, f64) {
        let dim = config.dimension;
        let seed = config.trial_seed("worm", &format!("binding_{}", set_size), trial_idx);
        let adapter = VisualObjectAdapter::default();

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;

        // Generate study objects (unique color-shape combos)
        let mut objects = Vec::with_capacity(set_size);
        for _ in 0..set_size {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            objects.push(VisualObject::new(
                (rng_state % 6) as u32,
                ((rng_state >> 8) % 5) as u32,
                ((rng_state >> 16) % 3) as u32,
            ));
        }

        // Time pressure: 0.15/unit encoding noise models degraded feature integration under
        // speed emphasis (Treisman, 1996 FIT); rushed binding produces illusory conjunctions.
        let diff_model = difficulty_model_for("WorM::Binding");
        let tp_noise_frac = (config.time_pressure as f32 * 0.15 + config.difficulty as f32 * 0.10)
            * diff_model.interference_multiplier(config.difficulty) as f32;
        for (pos, obj) in objects.iter().enumerate() {
            let hv = adapter.encode(obj, dim);
            let hv = if tp_noise_frac > 0.01 {
                let noise = ContinuousHV::random(dim, seed.wrapping_add(800 + pos as u64));
                ContinuousHV::weighted_bundle(&[&hv, &noise], &[1.0 - tp_noise_frac, tp_noise_frac])
            } else {
                hv
            };
            wm.perceive(hv);
            wm.tick();
        }

        // Delay
        for _ in 0..2 {
            wm.tick();
        }

        let contents = wm.contents();

        // Test 1: Binding probe (exact object from study set)
        let target = &objects[0];
        let target_hv = adapter.encode(target, dim);
        let binding_sim = contents
            .iter()
            .map(|item| item.similarity(&target_hv))
            .fold(0.0f32, f32::max);

        // Test 2: Feature-swap lure (swap color from one object with shape from another)
        let lure = if set_size >= 2 {
            VisualObject::new(objects[0].color, objects[1].shape, objects[0].size)
        } else {
            VisualObject::new(
                (objects[0].color + 3) % 6,
                objects[0].shape,
                objects[0].size,
            )
        };
        let lure_hv = adapter.encode(&lure, dim);
        let lure_sim = contents
            .iter()
            .map(|item| item.similarity(&lure_hv))
            .fold(0.0f32, f32::max);

        // Test 3: Novel object (not from study set)
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let novel = VisualObject::new(
            ((rng_state % 6) as u32 + 4) % 6,
            ((rng_state >> 8) % 5 + 3) as u32 % 5,
            ((rng_state >> 16) % 3 + 2) as u32 % 3,
        );
        let novel_hv = adapter.encode(&novel, dim);
        let novel_sim = contents
            .iter()
            .map(|item| item.similarity(&novel_hv))
            .fold(0.0f32, f32::max);

        // Binding accuracy: correctly accept target AND reject lure
        let binding_correct = if binding_sim > lure_sim { 1.0 } else { 0.0 };

        // Feature-only accuracy: correctly distinguish target from novel
        let feature_correct = if binding_sim > novel_sim { 1.0 } else { 0.0 };

        // RT proxy: deliberation ticks based on binding decision difficulty.
        // Smaller margin between target and lure → harder discrimination → longer RT
        // (Luck & Vogel, 1997; Wheeler & Treisman, 2002).
        let decision_margin = ((binding_sim - lure_sim).abs() as f64).min(1.0);
        let rt_ticks = 4.0 + (1.0 - decision_margin) * 6.0;

        (binding_correct, feature_correct, rt_ticks)
    }
}

impl PsychBenchmark for BindingBenchmark {
    fn name(&self) -> &str {
        "WorM::Binding"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Visual Working Memory Binding",
            citation: "Luck & Vogel (1997)",
            year: 1997,
            doi: Some("10.1038/36846"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        for k in [2, 4, 6] {
            let mut binding_accs = Vec::new();
            let mut feature_accs = Vec::new();
            let mut rts = Vec::new();

            for trial in 0..config.trials_per_condition {
                let (bind, feat, rt) = self.run_trial(k, config, trial);
                binding_accs.push(bind);
                feature_accs.push(feat);
                rts.push(rt);
                if config.trial_trace {
                    trace.push(TrialOutcome {
                        trial_idx: trace.len(),
                        condition: format!("binding_{}", k),
                        correct: bind > 0.5,
                        rt_ticks: rt,
                        similarity: 0.0,
                        confidence: 0.0,
                        response_idx: 0,
                        extra: BTreeMap::new(),
                    });
                }
            }

            result.insert(
                format!("set_{}::binding_accuracy", k),
                MetricValue::from_samples(&binding_accs),
            );
            result.insert(
                format!("set_{}::feature_accuracy", k),
                MetricValue::from_samples(&feature_accs),
            );

            // Binding deficit: feature_accuracy - binding_accuracy
            let deficits: Vec<f64> = feature_accs
                .iter()
                .zip(binding_accs.iter())
                .map(|(f, b)| f - b)
                .collect();
            result.insert(
                format!("set_{}::binding_deficit", k),
                MetricValue::from_samples(&deficits),
            );
            result.insert(
                format!("set_{}::rt_ticks", k),
                MetricValue::from_samples(&rts),
            );
        }

        // Aggregate: mean binding accuracy across all set sizes
        let all_binding: Vec<f64> = [2, 4, 6]
            .iter()
            .filter_map(|k| result.metrics.get(&format!("set_{}::binding_accuracy", k)))
            .map(|m| m.mean)
            .collect();
        if !all_binding.is_empty() {
            result.insert(
                "overall_binding_accuracy",
                MetricValue::from_samples(&all_binding),
            );
        }

        result.conditions = 3;
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
    fn test_binding_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = BindingBenchmark.run(&config);
        assert_eq!(result.conditions, 3);
        assert!(result.metrics.contains_key("set_2::binding_accuracy"));
        assert!(result.metrics.contains_key("set_2::feature_accuracy"));
    }
}
