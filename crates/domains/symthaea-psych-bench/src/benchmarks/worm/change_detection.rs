// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Change detection working memory task.
//!
//! Present an array of K visual objects, brief delay, then probe with
//! the original or a modified array. Tests maintenance capacity.

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

/// Change detection benchmark testing WM maintenance capacity.
pub struct ChangeDetectionBenchmark;

impl ChangeDetectionBenchmark {
    /// Run a single trial: present K objects, optionally change one, probe.
    /// Returns (accuracy, rt_ticks).
    fn run_trial(&self, set_size: usize, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        let dim = config.dimension;
        let seed = config.trial_seed("worm", &format!("cd_{}", set_size), trial_idx);
        let adapter = VisualObjectAdapter::default();

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        // Generate K unique visual objects
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        let mut objects = Vec::with_capacity(set_size);
        for i in 0..set_size {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            objects.push(VisualObject::new(
                (rng_state % 6) as u32,
                (i as u64 % 4) as u32,
                (rng_state % 3) as u32,
            ));
        }

        // Encode and present each object to WM.
        // Encoding noise scales with set size: WM precision decreases as
        // more items compete for limited resources (Bays & Husain, 2008;
        // Zhang & Luck, 2008 — variable precision model). This causes
        // detection failures even within nominal capacity.
        let mut original_hvs = Vec::with_capacity(set_size);
        for (pos, obj) in objects.iter().enumerate() {
            let hv = adapter.encode(obj, dim);
            // Encoding noise: 5% per item models variable-precision encoding (Bays & Husain, 2008).
            // Time pressure: +0.10/unit adds encoding degradation, reflecting reduced dwell time
            // in visual WM consolidation under speed emphasis (Vogel et al., 2006).
            let diff_model = difficulty_model_for("WorM::ChangeDetection");
            let noise_frac = (0.03 * set_size as f32 + config.time_pressure as f32 * 0.10)
                * diff_model.interference_multiplier(config.difficulty) as f32;
            let encoding_noise_seed = rng_state.wrapping_add(700 + pos as u64);
            let noisy_hv = if noise_frac > 0.01 {
                let noise = ContinuousHV::random(dim, encoding_noise_seed);
                ContinuousHV::weighted_bundle(&[&hv, &noise], &[1.0 - noise_frac, noise_frac])
            } else {
                hv.clone()
            };
            original_hvs.push(hv);
            wm.perceive(noisy_hv);
            wm.tick();
        }

        // Bundle original array
        let original_bundle = ContinuousHV::bundle_owned(&original_hvs);

        // Retention interval: ticks + noise injection models the degradation
        // of visual WM representations over time (Zhang & Luck, 2009 — slot+noise
        // model). Even items within capacity degrade, causing detection failures
        // that scale with set size (more items → more interference during delay).
        for delay_tick in 0..4 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            // Inject mild distractor noise every other tick
            if delay_tick % 2 == 1 {
                let noise = ContinuousHV::random(dim, rng_state);
                wm.perceive(noise);
            }
            wm.tick();
        }

        // Decide whether to change an item (50% probability)
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let is_changed = rng_state % 2 == 0;

        // Create probe array
        let mut probe_objects = objects.clone();
        if is_changed {
            let change_idx = (rng_state as usize / 2) % set_size;
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            probe_objects[change_idx] = VisualObject::new(
                ((rng_state % 6) as u32 + 3) % 6, // different color
                probe_objects[change_idx].shape,
                probe_objects[change_idx].size,
            );
        }

        let probe_hvs: Vec<ContinuousHV> = probe_objects
            .iter()
            .map(|obj| adapter.encode(obj, dim))
            .collect();
        let probe_bundle = ContinuousHV::bundle_owned(&probe_hvs);

        // Compare original bundle (as remembered by WM) with probe
        let contents: Vec<_> = wm.contents().iter().cloned().collect();
        let wm_bundle = if contents.is_empty() {
            ContinuousHV::zero(dim)
        } else {
            ContinuousHV::bundle_owned(&contents)
        };

        // Encoding noise degrades WM representation fidelity
        let noise_degrade = config.effective_noise() as f32 * 0.4;
        let sim_to_original = wm_bundle.similarity(&original_bundle) * (1.0 - noise_degrade);
        let sim_to_probe = wm_bundle.similarity(&probe_bundle) * (1.0 - noise_degrade);

        // System responds "same" if probe similarity ≈ original similarity.
        // Wider margin (0.06) models the decision noise in human change
        // detection: small encoding-noise differences between bundles are
        // sometimes undetectable (Wilken & Ma, 2004 — signal detection model).
        // Time pressure: base margin 0.06 models signal detection noise (Wilken & Ma, 2004);
        // +0.04/unit widens the "same" bias zone, reflecting hasty criterion under SAT (Wickelgren, 1977).
        let margin = 0.06 + config.time_pressure * 0.04;
        let responded_same = sim_to_probe >= sim_to_original - margin as f32;

        let correct = if is_changed {
            !responded_same // Should detect the change
        } else {
            responded_same // Should confirm same
        };

        // RT proxy: deliberation ticks based on same/different decision difficulty.
        // Smaller margin between original and probe similarity → harder to discriminate
        // → longer RT (Wilken & Ma, 2004; Luck & Vogel, 1997).
        let decision_margin = ((sim_to_original - sim_to_probe).abs() as f64).min(1.0);
        let rt_ticks = 4.0 + (1.0 - decision_margin) * 7.0;

        let acc = if correct { 1.0 } else { 0.0 };
        (acc, rt_ticks)
    }
}

impl PsychBenchmark for ChangeDetectionBenchmark {
    fn name(&self) -> &str {
        "WorM::ChangeDetection"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Visual Change Detection",
            citation: "Luck & Vogel (1997)",
            year: 1997,
            doi: Some("10.1038/36846"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        for k in [2, 4, 6, 8] {
            let mut accuracies = Vec::new();
            let mut rts = Vec::new();

            for trial in 0..config.trials_per_condition {
                let (acc, rt) = self.run_trial(k, config, trial);
                accuracies.push(acc);
                rts.push(rt);
                if config.trial_trace {
                    trace.push(TrialOutcome {
                        trial_idx: trace.len(),
                        condition: format!("change_{}", k),
                        correct: acc > 0.5,
                        rt_ticks: rt,
                        similarity: 0.0,
                        confidence: 0.0,
                        response_idx: 0,
                        extra: BTreeMap::new(),
                    });
                }
            }

            result.insert(
                format!("set_size_{}::accuracy", k),
                MetricValue::from_samples(&accuracies),
            );
            result.insert(
                format!("set_size_{}::rt_ticks", k),
                MetricValue::from_samples(&rts),
            );
        }

        result.conditions = 4;
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
    fn test_change_detection_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = ChangeDetectionBenchmark.run(&config);
        assert_eq!(result.conditions, 4);
        assert!(result.metrics.contains_key("set_size_2::accuracy"));
        assert!(result.metrics.contains_key("set_size_8::accuracy"));
    }
}
