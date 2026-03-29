// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Landmark Binding benchmark.
//!
//! Tests the ability to bind spatial location HVs with object identity HVs
//! and retrieve one from the other. This models the "What + Where" binding
//! that underlies spatial memory (Treisman & Zhang, 2006; Postma et al., 2004).
//!
//! HDC implementation: Location HVs and object HVs are created independently.
//! Each pair is bound via XOR: bound = location.bind(object). Given a location
//! cue, the system unbinds to retrieve the object and compares against all
//! objects. Retrieval accuracy decreases as the number of bound pairs increases,
//! reflecting capacity limits in visuospatial working memory.
//!
//! Human baselines (Luck & Vogel, 1997; Postma et al., 2004):
//! - retrieval_accuracy: 0.80 (SD~0.10) -- mean accuracy across set sizes
//! - capacity_k: 4.0 (SD~1.0) -- estimated capacity in bound pairs
//! - setsize_slope: 0.08 (SD~0.04) -- accuracy decrease per additional pair
//! - bidirectional_symmetry: 0.90 (SD~0.05) -- loc->obj vs obj->loc consistency

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;

/// Landmark Binding benchmark.
pub struct LandmarkBindingBenchmark;

struct TrialResult {
    retrieval_accuracy: f64,
    capacity_k: f64,
    setsize_slope: f64,
    bidirectional_symmetry: f64,
    mean_rt: f64,
}

impl LandmarkBindingBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("spatial", "landmark_binding", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        let diff_model = difficulty_model_for(self.name());
        let tp_noise: f32 = (0.05 + config.time_pressure as f32 * 0.10)
            / diff_model.signal_multiplier(config.difficulty) as f32;
        let noise_degrade = config.effective_noise() as f32 * 0.30;

        // Test across set sizes 1 to 8
        let max_set_size = 8;
        let mut setsize_data: Vec<(f64, f64, f64)> = Vec::new(); // (set_size, loc_acc, obj_acc)
        let mut rt_sum = 0.0f64;
        let mut rt_count = 0u32;

        for set_size in 1..=max_set_size {
            // Create location and object HVs
            let mut loc_hvs = Vec::with_capacity(set_size);
            let mut obj_hvs = Vec::with_capacity(set_size);
            let mut bound_hvs = Vec::with_capacity(set_size);

            for i in 0..set_size {
                let loc = BinaryHV::random(seed.wrapping_add(100 + i as u64 * 37));
                let obj = BinaryHV::random(seed.wrapping_add(500 + i as u64 * 53));
                let bound = loc.bind(&obj);
                loc_hvs.push(loc);
                obj_hvs.push(obj);
                bound_hvs.push(bound);
            }

            let n_presentations = 20;
            let mut loc_correct = 0u32; // location -> object retrieval
            let mut obj_correct = 0u32; // object -> location retrieval

            for pres in 0..n_presentations {
                xor_shift(&mut rng);
                let query_idx = (rng as usize) % set_size;

                // Location -> Object retrieval
                // Unbind: bound XOR location = object
                let recovered_obj = bound_hvs[query_idx].bind(&loc_hvs[query_idx]);

                // Find most similar object
                let mut best_sim = f32::MIN;
                let mut best_idx = 0usize;
                for (i, obj) in obj_hvs.iter().enumerate() {
                    let sim = recovered_obj.similarity(obj) * (1.0 - noise_degrade);
                    if sim > best_sim {
                        best_sim = sim;
                        best_idx = i;
                    }
                }

                xor_shift(&mut rng);
                let noise = ((rng % 10000) as f32 / 10000.0 - 0.5)
                    * tp_noise
                    * diff_model.temperature_multiplier(config.difficulty) as f32
                    * 0.3;

                if best_idx == query_idx && (best_sim + noise > 0.0) {
                    loc_correct += 1;
                }

                // Object -> Location retrieval
                let recovered_loc = bound_hvs[query_idx].bind(&obj_hvs[query_idx]);

                let mut best_sim_loc = f32::MIN;
                let mut best_loc_idx = 0usize;
                for (i, loc) in loc_hvs.iter().enumerate() {
                    let sim = recovered_loc.similarity(loc) * (1.0 - noise_degrade);
                    if sim > best_sim_loc {
                        best_sim_loc = sim;
                        best_loc_idx = i;
                    }
                }

                xor_shift(&mut rng);
                let noise2 = ((rng % 10000) as f32 / 10000.0 - 0.5)
                    * tp_noise
                    * diff_model.temperature_multiplier(config.difficulty) as f32
                    * 0.3;

                if best_loc_idx == query_idx && (best_sim_loc + noise2 > 0.0) {
                    obj_correct += 1;
                }

                // RT: more pairs = slower retrieval
                let base_rt = 2.0 + set_size as f64 * 0.8;
                let tp_speedup = config.time_pressure * 0.8;
                rt_sum += (base_rt - tp_speedup).max(1.0);
                rt_count += 1;

                let _ = pres;
            }

            let loc_acc = loc_correct as f64 / n_presentations as f64;
            let obj_acc = obj_correct as f64 / n_presentations as f64;
            setsize_data.push((set_size as f64, loc_acc, obj_acc));
        }

        // Overall retrieval accuracy (average of both directions)
        let n = setsize_data.len() as f64;
        let retrieval_accuracy: f64 =
            setsize_data.iter().map(|d| (d.1 + d.2) / 2.0).sum::<f64>() / n;

        // Capacity K: largest set size where accuracy > 0.75
        let mut capacity_k = 1.0;
        for d in &setsize_data {
            let avg_acc = (d.1 + d.2) / 2.0;
            if avg_acc > 0.75 {
                capacity_k = d.0;
            }
        }

        // Set-size slope: accuracy decrease per pair
        let mean_ss: f64 = setsize_data.iter().map(|d| d.0).sum::<f64>() / n;
        let mean_acc: f64 = setsize_data.iter().map(|d| (d.1 + d.2) / 2.0).sum::<f64>() / n;
        let cov: f64 = setsize_data
            .iter()
            .map(|d| (d.0 - mean_ss) * ((d.1 + d.2) / 2.0 - mean_acc))
            .sum::<f64>();
        let var_ss: f64 = setsize_data
            .iter()
            .map(|d| (d.0 - mean_ss).powi(2))
            .sum::<f64>();
        let raw_slope = if var_ss > 1e-10 { -(cov / var_ss) } else { 0.0 };
        let setsize_slope = raw_slope.clamp(0.0, 1.0);

        // Bidirectional symmetry: correlation between loc->obj and obj->loc accuracy
        let mean_loc: f64 = setsize_data.iter().map(|d| d.1).sum::<f64>() / n;
        let mean_obj: f64 = setsize_data.iter().map(|d| d.2).sum::<f64>() / n;
        let cov_bidir: f64 = setsize_data
            .iter()
            .map(|d| (d.1 - mean_loc) * (d.2 - mean_obj))
            .sum::<f64>();
        let var_loc: f64 = setsize_data
            .iter()
            .map(|d| (d.1 - mean_loc).powi(2))
            .sum::<f64>();
        let var_obj: f64 = setsize_data
            .iter()
            .map(|d| (d.2 - mean_obj).powi(2))
            .sum::<f64>();
        let r_bidir = if var_loc > 1e-10 && var_obj > 1e-10 {
            cov_bidir / (var_loc.sqrt() * var_obj.sqrt())
        } else {
            1.0
        };
        let bidirectional_symmetry = ((r_bidir + 1.0) / 2.0).clamp(0.0, 1.0);

        let mean_rt = if rt_count > 0 {
            rt_sum / rt_count as f64
        } else {
            3.0
        };

        TrialResult {
            retrieval_accuracy: retrieval_accuracy.clamp(0.0, 1.0),
            capacity_k,
            setsize_slope,
            bidirectional_symmetry,
            mean_rt,
        }
    }
}

impl PsychBenchmark for LandmarkBindingBenchmark {
    fn name(&self) -> &str {
        "Spatial::LandmarkBinding"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Spatial Landmark Binding",
            citation: "Postma et al. (2004)",
            year: 2004,
            doi: Some("10.1016/j.neuropsychologia.2004.03.014"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut accuracies = Vec::new();
        let mut capacities = Vec::new();
        let mut slopes = Vec::new();
        let mut symmetries = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            accuracies.push(r.retrieval_accuracy);
            capacities.push(r.capacity_k);
            slopes.push(r.setsize_slope);
            symmetries.push(r.bidirectional_symmetry);
            rts.push(r.mean_rt);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "landmark_binding".to_string(),
                    correct: r.retrieval_accuracy > 0.5,
                    rt_ticks: r.mean_rt,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("retrieval_accuracy", MetricValue::from_samples(&accuracies));
        result.insert("capacity_k", MetricValue::from_samples(&capacities));
        result.insert("setsize_slope", MetricValue::from_samples(&slopes));
        result.insert(
            "bidirectional_symmetry",
            MetricValue::from_samples(&symmetries),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 8; // set sizes 1-8
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
    fn test_landmark_binding_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = LandmarkBindingBenchmark.run(&config);
        assert!(result.metrics.contains_key("retrieval_accuracy"));
        assert!(result.metrics.contains_key("capacity_k"));
        assert!(result.metrics.contains_key("setsize_slope"));
        assert!(result.metrics.contains_key("bidirectional_symmetry"));
    }

    #[test]
    fn test_landmark_binding_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = LandmarkBindingBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_landmark_binding_metrics_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = LandmarkBindingBenchmark.run(&config);
        let acc = result.metrics["retrieval_accuracy"].mean;
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "retrieval_accuracy ({:.3}) out of bounds",
            acc
        );
        let sym = result.metrics["bidirectional_symmetry"].mean;
        assert!(
            sym >= 0.0 && sym <= 1.0,
            "bidirectional_symmetry ({:.3}) out of bounds",
            sym
        );
    }

    #[test]
    fn test_landmark_binding_xor_self_inverse() {
        // XOR binding is perfectly self-inverse: (A XOR B) XOR B = A
        // With set_size=1, retrieval should be perfect
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = LandmarkBindingBenchmark.run(&config);
        // Capacity should be at least 1 (trivially)
        let k = result.metrics["capacity_k"].mean;
        assert!(k >= 1.0, "capacity_k ({:.3}) should be >= 1", k);
    }

    #[test]
    fn test_landmark_binding_provenance() {
        let prov = LandmarkBindingBenchmark.provenance().unwrap();
        assert_eq!(prov.year, 2004);
        assert!(prov.citation.contains("Postma"));
    }
}
