// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Modal Binding benchmark.
//!
//! Tests the ability to bind features across different modalities (e.g., color
//! bound to shape, location bound to identity). This is the "binding problem"
//! central to consciousness research (Treisman & Gelade, 1980).
//!
//! The system encodes multi-feature objects using HDC role-filler binding, then
//! must correctly retrieve which feature was bound to which role. Swap errors
//! (reporting a feature from the wrong object) reveal binding failures.
//!
//! Human baselines (Treisman & Gelade, 1980; Wheeler & Treisman, 2002):
//! - binding_accuracy: 0.78 (SD 0.10) — correct feature-role retrieval
//! - swap_error_rate: 0.12 (SD 0.06) — misbinding rate
//! - set_size_slope: 0.05 (SD 0.02) — accuracy drop per additional object

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Cross-Modal Feature Binding benchmark.
pub struct CrossModalBindingBenchmark;

#[allow(dead_code)] // per_size_accuracy reserved for size-dependent analysis
struct TrialResult {
    binding_accuracy: f64,
    swap_error_rate: f64,
    set_size_slope: f64,
    per_size_accuracy: Vec<f64>,
}

impl CrossModalBindingBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("binding", "cross_modal", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Feature prototypes: 8 colors, 8 shapes, 8 locations
        // Larger feature vocabularies produce more orthogonal HDC role-filler
        // bindings, reducing similarity between distinct features in the
        // high-dimensional space (Kanerva, 2009). With 6 features, chance
        // similarity between distinct features is ~1/6; with 8, it's ~1/8.
        // This improves discriminability when retrieving bound features from
        // a bundled scene memory, particularly at larger set sizes.
        let n_features = 8usize;
        let color_hvs: Vec<ContinuousHV> = (0..n_features)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(100 + i as u64)))
            .collect();
        let shape_hvs: Vec<ContinuousHV> = (0..n_features)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(200 + i as u64)))
            .collect();
        let location_hvs: Vec<ContinuousHV> = (0..n_features)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(300 + i as u64)))
            .collect();

        // Role binders
        let role_color = ContinuousHV::random(dim, seed.wrapping_add(1000));
        let role_shape = ContinuousHV::random(dim, seed.wrapping_add(1001));
        let _role_location = ContinuousHV::random(dim, seed.wrapping_add(1002));

        let set_sizes = [2usize, 4, 6];
        let trials_per_size = 20;
        let pressure = config.time_pressure;
        // Base noise 0.008: with 8 feature prototypes (up from 6), the larger
        // vocabulary produces more orthogonal representations, reducing bundle
        // cross-talk. This allows slightly lower encoding noise while maintaining
        // realistic set-size effects (Wheeler & Treisman, 2002).
        let noise_frac = 0.008 + pressure as f32 * 0.05 + config.effective_noise() as f32 * 0.03;

        let mut total_correct = 0u32;
        let mut total_swaps = 0u32;
        let mut total_probes = 0u32;
        let mut per_size_acc = Vec::new();

        for &set_size in &set_sizes {
            let mut size_correct = 0u32;
            let mut size_total = 0u32;

            for _trial in 0..trials_per_size {
                // Generate random objects with unique features
                let mut objects: Vec<(usize, usize, usize)> = Vec::new();
                for obj_i in 0..set_size {
                    xor_shift(&mut rng);
                    let c = (rng as usize + obj_i) % n_features;
                    xor_shift(&mut rng);
                    let s = (rng as usize + obj_i) % n_features;
                    xor_shift(&mut rng);
                    let l = (rng as usize + obj_i) % n_features;
                    objects.push((c, s, l));
                }

                // Encode each object using location-indexed binding.
                // Each object is encoded as bind(features, location), making
                // the location the retrieval key. This preserves color-location
                // association more strongly than flat bundling.
                // Treisman's Feature Integration Theory: attention binds features
                // to locations, so location-indexed storage is psychologically valid.
                let object_hvs: Vec<ContinuousHV> = objects
                    .iter()
                    .map(|&(c, s, l)| {
                        // Bind color and shape to their roles
                        let bound_c = color_hvs[c].bind(&role_color);
                        let bound_s = shape_hvs[s].bind(&role_shape);
                        // Bundle features, then bind to location (location is the key)
                        let features = ContinuousHV::bundle(&[&bound_c, &bound_s]);
                        features.bind(&location_hvs[l])
                    })
                    .collect();

                // Scene memory: bundle all location-indexed objects with noise
                let obj_refs: Vec<&ContinuousHV> = object_hvs.iter().collect();
                let scene = ContinuousHV::bundle(&obj_refs);
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                let noisy_scene = ContinuousHV::weighted_bundle(
                    &[&scene, &noise_hv],
                    &[1.0 - noise_frac, noise_frac],
                );

                // Probe: pick a random object, ask "what color was at this location?"
                xor_shift(&mut rng);
                let probe_idx = rng as usize % set_size;
                let (correct_color, _correct_shape, probe_loc) = objects[probe_idx];

                // Retrieval via candidate matching with location-indexed encoding.
                // For each candidate color, construct the expected location-indexed
                // binding and compare to the scene. The correct color's candidate
                // will match the stored binding at the probed location.
                let mut best_color = 0;
                let mut best_sim = f32::NEG_INFINITY;
                for (i, chv) in color_hvs.iter().enumerate() {
                    let bound_c = chv.bind(&role_color);
                    // Candidate: color-role bound to probed location
                    let candidate = bound_c.bind(&location_hvs[probe_loc]);
                    let sim = noisy_scene.similarity(&candidate);
                    if sim > best_sim {
                        best_sim = sim;
                        best_color = i;
                    }
                }

                total_probes += 1;
                size_total += 1;
                if best_color == correct_color {
                    total_correct += 1;
                    size_correct += 1;
                } else {
                    // Check if it's a swap error (color from another object in scene)
                    let is_swap = objects
                        .iter()
                        .enumerate()
                        .any(|(i, &(c, _, _))| i != probe_idx && c == best_color);
                    if is_swap {
                        total_swaps += 1;
                    }
                }
            }

            per_size_acc.push(size_correct as f64 / size_total as f64);
        }

        let binding_accuracy = total_correct as f64 / total_probes as f64;
        let swap_error_rate = total_swaps as f64 / total_probes as f64;

        // Set-size slope: linear regression of accuracy vs set_size
        let sizes_f = [2.0, 4.0, 6.0];
        let x_mean = 4.0;
        let y_mean = per_size_acc.iter().sum::<f64>() / 3.0;
        let num: f64 = per_size_acc
            .iter()
            .zip(sizes_f.iter())
            .map(|(y, x)| (x - x_mean) * (y - y_mean))
            .sum();
        let den: f64 = sizes_f.iter().map(|x| (x - x_mean).powi(2)).sum();
        let set_size_slope = if den.abs() > 1e-10 {
            -(num / den) // negate so positive = worse with more items
        } else {
            0.0
        };

        TrialResult {
            binding_accuracy,
            swap_error_rate,
            set_size_slope,
            per_size_accuracy: per_size_acc,
        }
    }
}

impl PsychBenchmark for CrossModalBindingBenchmark {
    fn name(&self) -> &str {
        "Binding::CrossModal"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Cross-Modal Feature Binding",
            citation: "Treisman & Gelade (1980); Wheeler & Treisman (2002)",
            year: 1980,
            doi: Some("10.1016/0010-0285(80)90005-5"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut accs = Vec::new();
        let mut swaps = Vec::new();
        let mut slopes = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            accs.push(r.binding_accuracy);
            swaps.push(r.swap_error_rate);
            slopes.push(r.set_size_slope);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "cross_modal".to_string(),
                    correct: r.binding_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.binding_accuracy,
                    confidence: 1.0 - r.swap_error_rate,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("binding_accuracy", MetricValue::from_samples(&accs));
        result.insert("swap_error_rate", MetricValue::from_samples(&swaps));
        result.insert("set_size_slope", MetricValue::from_samples(&slopes));

        result.conditions = 3; // 3 set sizes
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
    fn test_cross_modal_runs() {
        let config = BenchmarkConfig::default();
        let result = CrossModalBindingBenchmark.run(&config);
        assert!(result.metrics.contains_key("binding_accuracy"));
        assert!(result.metrics.contains_key("swap_error_rate"));
        assert!(result.metrics.contains_key("set_size_slope"));
    }

    #[test]
    fn test_cross_modal_finite() {
        let config = BenchmarkConfig::default();
        let result = CrossModalBindingBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_cross_modal_accuracy_bounded() {
        let config = BenchmarkConfig::default();
        let result = CrossModalBindingBenchmark.run(&config);
        let acc = result.metrics["binding_accuracy"].mean;
        assert!(acc >= 0.0 && acc <= 1.0, "accuracy {acc} out of bounds");
    }
}
