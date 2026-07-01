// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Visual Search Task.
//!
//! Tests attentional efficiency: feature search (pop-out, parallel) vs
//! conjunction search (serial attention required). In feature search, the
//! target has a unique feature making it instantly detectable regardless
//! of set size. In conjunction search, distractors share features with
//! the target, requiring serial inspection.
//!
//! Human baselines (Treisman & Gelade 1980; Wolfe 1994):
//! - feature_search_accuracy: 0.98 (SD≈0.02)
//! - conjunction_search_accuracy: 0.88 (SD≈0.06)
//! - feature_search_slope: 0.0 (SD≈0.5) ticks/item
//! - conjunction_search_slope: 2.0 (SD≈0.8) ticks/item
//! - search_asymmetry: 2.0 (SD≈0.5)

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Visual Search benchmark testing parallel vs serial attentional processing.
pub struct VisualSearchBenchmark;

impl VisualSearchBenchmark {
    /// Run a single visual search trial.
    /// Returns (feature_accs_by_size, conjunction_accs_by_size, feature_rts_by_size, conjunction_rts_by_size).
    #[allow(clippy::type_complexity)]
    fn run_trial(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> ([Vec<f64>; 4], [Vec<f64>; 4], [Vec<f64>; 4], [Vec<f64>; 4]) {
        let dim = config.dimension;
        let seed = config.trial_seed("attention", "visual_search", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Feature dimensions: color and shape
        let red = ContinuousHV::random(dim, seed.wrapping_add(100));
        let blue = ContinuousHV::random(dim, seed.wrapping_add(101));
        let circle = ContinuousHV::random(dim, seed.wrapping_add(200));
        let square = ContinuousHV::random(dim, seed.wrapping_add(201));

        // Target: red circle (bound via HDC binding)
        let target_template = red.bind(&circle);

        // Feature search distractors: blue circles (differ on one feature — pop-out)
        // Conjunction search distractors: red squares AND blue circles
        //   (each shares one feature with target — requires serial inspection)

        // Time pressure: threshold reduction models liberal response criterion
        // under speed emphasis (Wickelgren, 1977 SAT); temperature increase models
        // noisier evidence accumulation (Heitz, 2014).
        let pressure = config.time_pressure;
        let diff_model = difficulty_model_for(self.name());
        let sig_mult = diff_model.signal_multiplier(config.difficulty);
        let threshold: f64 = (0.35 - pressure * 0.10) * sig_mult;
        let temperature: f64 =
            (0.25 + pressure * 0.15) * diff_model.temperature_multiplier(config.difficulty);

        let set_sizes = [4usize, 8, 16, 24];
        let trials_per_size = 20;

        let mut feat_accs: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut conj_accs: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut feat_rts: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut conj_rts: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        for (si, &set_size) in set_sizes.iter().enumerate() {
            for _t in 0..trials_per_size {
                xor_shift(&mut rng);
                let target_present = rng % 2 == 0;

                // === Feature search ===
                // All distractors are blue circles — target pops out by color
                let mut max_feat_sim = f64::NEG_INFINITY;
                for item_idx in 0..set_size {
                    let is_target = target_present && item_idx == 0;
                    xor_shift(&mut rng);
                    let noise = ContinuousHV::random(dim, rng);

                    let item = if is_target {
                        // Target: red circle + noise
                        ContinuousHV::weighted_bundle(&[&target_template, &noise], &[0.85, 0.15])
                    } else {
                        // Distractor: blue circle (shares shape but not color)
                        let dist = blue.bind(&circle);
                        ContinuousHV::weighted_bundle(&[&dist, &noise], &[0.85, 0.15])
                    };

                    // Encoding noise degrades perceptual discriminability
                    let noise_degrade = config.effective_noise() as f32 * 0.4;
                    let sim =
                        item.similarity(&target_template) as f64 * (1.0 - noise_degrade as f64);
                    if sim > max_feat_sim {
                        max_feat_sim = sim;
                    }
                }

                // Feature search RT: constant across set sizes (parallel processing;
                // Treisman & Gelade, 1980). Base 3 ticks + small noise.
                let feat_rt = 3.0 + (1.0 - max_feat_sim.clamp(0.0, 1.0)) * 2.0;
                feat_rts[si].push(feat_rt);

                let feat_ev = (max_feat_sim - threshold) / temperature;
                let feat_no_ev = 0.0_f64;
                let max_ev = feat_ev.max(feat_no_ev);
                let p_present = (feat_ev - max_ev).exp()
                    / ((feat_ev - max_ev).exp() + (feat_no_ev - max_ev).exp());

                xor_shift(&mut rng);
                let r = (rng % 10000) as f64 / 10000.0;
                let responded_present = r < p_present;
                let correct = responded_present == target_present;
                feat_accs[si].push(if correct { 1.0 } else { 0.0 });

                // === Conjunction search ===
                // Distractors: mix of red squares and blue circles
                // Each shares one feature with target → serial search needed
                let mut max_conj_sim = f64::NEG_INFINITY;
                // Serial inspection adds RT per item (Treisman & Gelade, 1980).
                let mut cumulative_rt = 3.0;
                for item_idx in 0..set_size {
                    let is_target = target_present && item_idx == 0;
                    xor_shift(&mut rng);
                    let noise = ContinuousHV::random(dim, rng);

                    let item = if is_target {
                        ContinuousHV::weighted_bundle(&[&target_template, &noise], &[0.85, 0.15])
                    } else {
                        // Alternate between red-square and blue-circle distractors
                        let dist = if item_idx % 2 == 0 {
                            red.bind(&square) // shares color with target
                        } else {
                            blue.bind(&circle) // shares shape with target
                        };
                        ContinuousHV::weighted_bundle(&[&dist, &noise], &[0.85, 0.15])
                    };

                    let noise_degrade = config.effective_noise() as f32 * 0.4;
                    let sim =
                        item.similarity(&target_template) as f64 * (1.0 - noise_degrade as f64);
                    if sim > max_conj_sim {
                        max_conj_sim = sim;
                    }

                    // Serial inspection cost: ~0.5 ticks per item for conjunction
                    // (Wolfe, 1994 guided search: slope ~25ms/item ≈ 0.5 ticks/item).
                    cumulative_rt += 0.75;

                    // Early termination if strong match found
                    if sim > threshold + 0.15 {
                        break;
                    }
                }
                conj_rts[si].push(cumulative_rt);

                let conj_ev = (max_conj_sim - threshold) / temperature;
                let conj_no_ev = 0.0_f64;
                let max_ev2 = conj_ev.max(conj_no_ev);
                let p_present2 = (conj_ev - max_ev2).exp()
                    / ((conj_ev - max_ev2).exp() + (conj_no_ev - max_ev2).exp());

                xor_shift(&mut rng);
                let r2 = (rng % 10000) as f64 / 10000.0;
                let resp2 = r2 < p_present2;
                let correct2 = resp2 == target_present;
                conj_accs[si].push(if correct2 { 1.0 } else { 0.0 });
            }
        }

        (feat_accs, conj_accs, feat_rts, conj_rts)
    }

    /// Compute search slope via least-squares regression of RT on set size.
    fn compute_slope(set_sizes: &[f64], mean_rts: &[f64]) -> f64 {
        let n = set_sizes.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let sx: f64 = set_sizes.iter().sum();
        let sy: f64 = mean_rts.iter().sum();
        let sxx: f64 = set_sizes.iter().map(|x| x * x).sum();
        let sxy: f64 = set_sizes.iter().zip(mean_rts).map(|(x, y)| x * y).sum();
        let denom = n * sxx - sx * sx;
        if denom.abs() < 1e-15 {
            return 0.0;
        }
        (n * sxy - sx * sy) / denom
    }
}

impl PsychBenchmark for VisualSearchBenchmark {
    fn name(&self) -> &str {
        "Attention::VisualSearch"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Visual Search",
            citation: "Treisman & Gelade (1980)",
            year: 1980,
            doi: Some("10.1016/0010-0285(80)90005-5"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let set_sizes_f = [4.0, 8.0, 16.0, 24.0];
        let mut feat_accs_all = Vec::new();
        let mut conj_accs_all = Vec::new();
        let mut feat_slopes = Vec::new();
        let mut conj_slopes = Vec::new();
        let mut all_rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (fa, ca, fr, cr) = self.run_trial(config, trial);

            // Mean accuracy across all set sizes
            let feat_acc: f64 = fa.iter().flat_map(|v| v.iter()).sum::<f64>()
                / fa.iter().map(|v| v.len()).sum::<usize>().max(1) as f64;
            let conj_acc: f64 = ca.iter().flat_map(|v| v.iter()).sum::<f64>()
                / ca.iter().map(|v| v.len()).sum::<usize>().max(1) as f64;
            feat_accs_all.push(feat_acc);
            conj_accs_all.push(conj_acc);

            // Mean RT per set size
            let feat_mean_rts: Vec<f64> = fr
                .iter()
                .map(|v| {
                    if v.is_empty() {
                        0.0
                    } else {
                        v.iter().sum::<f64>() / v.len() as f64
                    }
                })
                .collect();
            let conj_mean_rts: Vec<f64> = cr
                .iter()
                .map(|v| {
                    if v.is_empty() {
                        0.0
                    } else {
                        v.iter().sum::<f64>() / v.len() as f64
                    }
                })
                .collect();

            feat_slopes.push(Self::compute_slope(&set_sizes_f, &feat_mean_rts));
            conj_slopes.push(Self::compute_slope(&set_sizes_f, &conj_mean_rts));

            // Collect all RTs
            for v in fr.iter().chain(cr.iter()) {
                all_rts.extend_from_slice(v);
            }
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "visual_search".to_string(),
                    correct: feat_acc > 0.5 && conj_acc > 0.5,
                    rt_ticks: all_rts.last().copied().unwrap_or(0.0),
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        let asymmetries: Vec<f64> = conj_slopes
            .iter()
            .zip(&feat_slopes)
            .map(|(c, f)| c - f)
            .collect();

        result.insert(
            "feature_search_accuracy",
            MetricValue::from_samples(&feat_accs_all),
        );
        result.insert(
            "conjunction_search_accuracy",
            MetricValue::from_samples(&conj_accs_all),
        );
        result.insert(
            "feature_search_slope",
            MetricValue::from_samples(&feat_slopes),
        );
        result.insert(
            "conjunction_search_slope",
            MetricValue::from_samples(&conj_slopes),
        );
        result.insert("search_asymmetry", MetricValue::from_samples(&asymmetries));
        result.insert("rt_ticks", MetricValue::from_samples(&all_rts));

        result.conditions = 8; // 2 search types × 4 set sizes
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
    fn test_visual_search_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = VisualSearchBenchmark.run(&config);
        assert!(result.metrics.contains_key("feature_search_accuracy"));
        assert!(result.metrics.contains_key("conjunction_search_accuracy"));
        assert!(result.metrics.contains_key("feature_search_slope"));
        assert!(result.metrics.contains_key("conjunction_search_slope"));
        assert!(result.metrics.contains_key("search_asymmetry"));
    }

    #[test]
    fn test_visual_search_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = VisualSearchBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_conjunction_harder_than_feature() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = VisualSearchBenchmark.run(&config);
        let feat = result.metrics["feature_search_accuracy"].mean;
        let conj = result.metrics["conjunction_search_accuracy"].mean;
        // Feature search should be at least as accurate as conjunction (with tolerance)
        assert!(
            feat >= conj - 0.10,
            "Feature ({:.3}) should be >= Conjunction ({:.3}) - 0.10",
            feat,
            conj
        );
    }
}
