// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Perceptual Crowding benchmark.
//!
//! Tests identification accuracy for peripheral targets when flanked by nearby
//! distractors. Crowding is a fundamental limit of conscious perception: flanker
//! proximity degrades target recognition in peripheral vision (Whitney & Levi, 2011).
//!
//! HDC implementation: encodes target and flanker stimuli as hypervectors, then
//! bundles them with proximity-weighted interference. Closer flankers contribute
//! more noise to the target representation. Identification requires matching
//! the degraded representation against clean prototypes.
//!
//! Human baselines (Pelli et al., 2004; Whitney & Levi, 2011):
//! - unflanked_accuracy: 0.95 (SD 0.04) — isolated target identification
//! - flanked_accuracy: 0.62 (SD 0.12) — target identification with close flankers
//! - crowding_magnitude: 0.33 (SD 0.10) — unflanked minus flanked accuracy

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Perceptual Crowding benchmark.
pub struct PerceptualCrowdingBenchmark;

struct TrialResult {
    unflanked_accuracy: f64,
    flanked_accuracy: f64,
    crowding_magnitude: f64,
}

impl PerceptualCrowdingBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("consciousness", "perceptual_crowding", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Letter/stimulus prototypes (8 distinct items)
        let n_stimuli = 8usize;
        let prototypes: Vec<ContinuousHV> = (0..n_stimuli)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(100 + i as u64)))
            .collect();

        let noise_base = 0.12 + config.effective_noise() as f32 * 0.08;
        let trials_per_condition = 40;

        // --- Unflanked condition: target alone ---
        let mut unflanked_correct = 0u32;
        let mut unflanked_total = 0u32;

        for _trial in 0..trials_per_condition {
            xor_shift(&mut rng);
            let target_idx = rng as usize % n_stimuli;

            // Target with baseline noise only
            xor_shift(&mut rng);
            let noise_hv = ContinuousHV::random(dim, rng);
            let stimulus = ContinuousHV::weighted_bundle(
                &[&prototypes[target_idx], &noise_hv],
                &[1.0 - noise_base, noise_base],
            );

            // Identify: find closest prototype
            let mut best_idx = 0;
            let mut best_sim = f32::NEG_INFINITY;
            for (i, proto) in prototypes.iter().enumerate() {
                let sim = stimulus.similarity(proto);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }

            unflanked_total += 1;
            if best_idx == target_idx {
                unflanked_correct += 1;
            }
        }

        // --- Flanked condition: target + 2 flankers at close spacing ---
        let mut flanked_correct = 0u32;
        let mut flanked_total = 0u32;

        // Spacing levels (closer = more crowding)
        // Bouma's law: critical spacing ~ 0.5 * eccentricity
        let spacings = [0.15f32, 0.25, 0.40];

        for &spacing in &spacings {
            for _trial in 0..trials_per_condition / spacings.len() {
                xor_shift(&mut rng);
                let target_idx = rng as usize % n_stimuli;

                // Two flankers: random different stimuli
                xor_shift(&mut rng);
                let flanker1_idx = (target_idx + 1 + rng as usize % (n_stimuli - 1)) % n_stimuli;
                xor_shift(&mut rng);
                let flanker2_idx = (target_idx + 2 + rng as usize % (n_stimuli - 1)) % n_stimuli;

                // Crowding interference: flankers bleed into target representation
                // proportional to 1/spacing (Bouma's law). In peripheral vision,
                // flanker features are mandatorily pooled with the target, creating
                // a texture-like percept rather than individual objects (Parkes et al., 2001).
                let crowding_strength = (0.3 / spacing).min(1.0);

                // Compulsory averaging: flankers pool with target in peripheral
                // vision. HDC weighted bundles in high dimensions preserve target
                // identity too well, so we model crowding as feature-level corruption:
                // a fraction of the target's dimensions are replaced by flanker values.
                let corruption_rate = crowding_strength * 0.65;

                // Build the crowded representation by element-wise dimension masking:
                // for each dimension, with probability corruption_rate, replace target
                // value with a flanker value. This models the feature-pooling mechanism
                // more faithfully than weighted bundle (which preserves direction).
                let target_hv = &prototypes[target_idx];
                let flanker1_hv = &prototypes[flanker1_idx];
                let flanker2_hv = &prototypes[flanker2_idx];

                let target_data = target_hv.as_slice();
                let f1_data = flanker1_hv.as_slice();
                let f2_data = flanker2_hv.as_slice();

                let mut crowded_data = target_data.to_vec();
                let mut local_rng = rng;
                for d in 0..dim {
                    // Simple hash-based per-dimension decision
                    local_rng ^= local_rng << 13;
                    local_rng ^= local_rng >> 7;
                    local_rng ^= local_rng << 17;
                    let r = (local_rng as f32) / (u64::MAX as f32);
                    if r < corruption_rate {
                        // Replace with flanker feature (alternating flankers)
                        crowded_data[d] = if d % 2 == 0 { f1_data[d] } else { f2_data[d] };
                    }
                }
                rng = local_rng;
                let crowded = ContinuousHV::from_slice(&crowded_data);

                // Peripheral encoding noise (higher than foveal baseline)
                let peripheral_noise = noise_base + crowding_strength * 0.05;
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                let stimulus = ContinuousHV::weighted_bundle(
                    &[&crowded, &noise_hv],
                    &[1.0 - peripheral_noise, peripheral_noise],
                );

                // Identify
                let mut best_idx = 0;
                let mut best_sim = f32::NEG_INFINITY;
                for (i, proto) in prototypes.iter().enumerate() {
                    let sim = stimulus.similarity(proto);
                    if sim > best_sim {
                        best_sim = sim;
                        best_idx = i;
                    }
                }

                flanked_total += 1;
                if best_idx == target_idx {
                    flanked_correct += 1;
                }
            }
        }

        let unflanked_accuracy = unflanked_correct as f64 / unflanked_total.max(1) as f64;
        let flanked_accuracy = flanked_correct as f64 / flanked_total.max(1) as f64;
        let crowding_magnitude = (unflanked_accuracy - flanked_accuracy).max(0.0);

        TrialResult {
            unflanked_accuracy,
            flanked_accuracy,
            crowding_magnitude,
        }
    }
}

impl PsychBenchmark for PerceptualCrowdingBenchmark {
    fn name(&self) -> &str {
        "Consciousness::PerceptualCrowding"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Perceptual Crowding",
            citation: "Whitney & Levi (2011); Pelli et al. (2004)",
            year: 2011,
            doi: Some("10.1016/j.tics.2011.02.002"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut unflanked = Vec::new();
        let mut flanked = Vec::new();
        let mut magnitudes = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            unflanked.push(r.unflanked_accuracy);
            flanked.push(r.flanked_accuracy);
            magnitudes.push(r.crowding_magnitude);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "crowding".to_string(),
                    correct: r.flanked_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.flanked_accuracy,
                    confidence: 1.0 - r.crowding_magnitude,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("unflanked_accuracy", MetricValue::from_samples(&unflanked));
        result.insert("flanked_accuracy", MetricValue::from_samples(&flanked));
        result.insert("crowding_magnitude", MetricValue::from_samples(&magnitudes));

        result.conditions = 4; // 1 unflanked + 3 spacing levels
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
    fn test_perceptual_crowding_runs() {
        let config = BenchmarkConfig::default();
        let result = PerceptualCrowdingBenchmark.run(&config);
        assert!(result.metrics.contains_key("unflanked_accuracy"));
        assert!(result.metrics.contains_key("flanked_accuracy"));
        assert!(result.metrics.contains_key("crowding_magnitude"));
    }

    #[test]
    fn test_perceptual_crowding_finite() {
        let config = BenchmarkConfig::default();
        let result = PerceptualCrowdingBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_unflanked_better_than_flanked() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = PerceptualCrowdingBenchmark.run(&config);
        let unflanked = result.metrics["unflanked_accuracy"].mean;
        let flanked = result.metrics["flanked_accuracy"].mean;
        assert!(
            unflanked >= flanked - 0.05,
            "Unflanked ({unflanked:.3}) should be >= Flanked ({flanked:.3})"
        );
    }
}
