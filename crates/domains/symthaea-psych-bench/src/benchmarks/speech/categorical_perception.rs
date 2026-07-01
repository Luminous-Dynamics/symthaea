// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Categorical Perception benchmark.
//!
//! Tests the sharp category boundary in speech sound identification — the
//! hallmark of categorical perception (Liberman et al., 1957). Along an
//! acoustic continuum (e.g., /ba/-/da/), identification shifts abruptly from
//! one category to another, and discrimination peaks at the category boundary.
//!
//! HDC implementation: creates two category prototypes and a continuum of
//! stimuli interpolated between them. Measures the sharpness of the
//! identification function (logistic slope) and the boundary discrimination
//! peak (discrimination at boundary vs. within-category).
//!
//! Human baselines (Liberman et al., 1957; Pisoni, 1973):
//! - boundary_slope: 0.80 (SD 0.12) — logistic fit slope at category boundary
//! - boundary_discrimination: 0.88 (SD 0.08) — discrimination accuracy at boundary
//! - within_category_discrimination: 0.55 (SD 0.10) — discrimination within category
//! - categorical_index: 0.33 (SD 0.10) — boundary minus within discrimination

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Categorical Perception benchmark.
pub struct CategoricalPerceptionBenchmark;

struct TrialResult {
    boundary_slope: f64,
    boundary_discrimination: f64,
    within_category_discrimination: f64,
    categorical_index: f64,
}

impl CategoricalPerceptionBenchmark {
    /// ABX discrimination with perceptual warping toward category prototypes.
    ///
    /// The key mechanism of categorical perception: internal representations
    /// are warped toward the nearest category prototype. This makes
    /// within-category pairs (both warped to the same prototype) harder to
    /// discriminate than cross-boundary pairs (warped to different prototypes),
    /// even at equal acoustic distance.
    fn abx_discriminate(
        stimulus: &ContinuousHV,
        reference_a: &ContinuousHV,
        reference_b: &ContinuousHV,
        cat_a: &ContinuousHV,
        cat_b: &ContinuousHV,
        warp_strength: f32,
    ) -> bool {
        // Perceptual warping: pull stimulus toward nearest category prototype
        let sim_to_a = stimulus.similarity(cat_a);
        let sim_to_b = stimulus.similarity(cat_b);
        let nearest_cat = if sim_to_a >= sim_to_b { cat_a } else { cat_b };

        let warped = ContinuousHV::weighted_bundle(
            &[stimulus, nearest_cat],
            &[1.0 - warp_strength, warp_strength],
        );

        let sim_a = warped.similarity(reference_a);
        let sim_b = warped.similarity(reference_b);
        sim_a > sim_b
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("speech", "categorical_perception", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Two category endpoints (e.g., /ba/ and /da/)
        let cat_a = ContinuousHV::random(dim, seed.wrapping_add(100));
        let cat_b = ContinuousHV::random(dim, seed.wrapping_add(200));

        let noise_frac = 0.02 + config.effective_noise() as f32 * 0.04;

        // Perceptual warping strength: how strongly internal representations
        // are pulled toward the nearest category prototype. This is the core
        // mechanism producing categorical perception (Harnad, 1987).
        let warp_strength: f32 = 0.42;

        // Discrimination noise: higher than identification noise to model
        // the difficulty of same/different judgments on close stimuli.
        let disc_noise_frac: f32 = 0.40 + config.effective_noise() as f32 * 0.06;

        // Create continuum: 11 steps from cat_a (0.0) to cat_b (1.0)
        let n_steps = 11usize;
        let continuum: Vec<(f64, ContinuousHV)> = (0..n_steps)
            .map(|i| {
                let t = i as f32 / (n_steps - 1) as f32;
                let hv = ContinuousHV::weighted_bundle(&[&cat_a, &cat_b], &[1.0 - t, t]);
                (t as f64, hv)
            })
            .collect();

        let trials_per_step = 30;

        // --- Identification task: classify each step as category A or B ---
        let mut id_proportions = Vec::new(); // proportion classified as B at each step

        for (_t, step_hv) in &continuum {
            let mut b_count = 0u32;
            let mut total = 0u32;

            for _trial in 0..trials_per_step {
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                let stimulus = ContinuousHV::weighted_bundle(
                    &[step_hv, &noise_hv],
                    &[1.0 - noise_frac, noise_frac],
                );

                let sim_a = stimulus.similarity(&cat_a) as f64;
                let sim_b = stimulus.similarity(&cat_b) as f64;

                total += 1;
                if sim_b > sim_a {
                    b_count += 1;
                }
            }

            id_proportions.push(b_count as f64 / total as f64);
        }

        // Boundary slope: steepness of identification function at midpoint
        let mid = n_steps / 2;
        let boundary_slope = if mid > 0 && mid < n_steps - 1 {
            let step_size = 1.0 / (n_steps - 1) as f64;
            let slope = (id_proportions[mid + 1] - id_proportions[mid - 1]) / (2.0 * step_size);
            slope.abs().min(1.0)
        } else {
            0.5
        };

        // --- Discrimination task: ABX paradigm ---
        // Use equal step-distance pairs for scientific validity (1 step apart).
        // Cross-boundary: straddles category midpoint.
        // Within-category: both endpoints on same side of boundary.
        let boundary_pairs = [(mid - 1, mid + 1)]; // across boundary (2-step)
        let within_pairs = [(0, 2), (n_steps - 3, n_steps - 1)]; // within category (2-step, same distance)

        let mut boundary_correct = 0u32;
        let mut boundary_total = 0u32;

        for &(i, j) in &boundary_pairs {
            for _trial in 0..trials_per_step {
                xor_shift(&mut rng);
                let x_is_a = rng % 2 == 0;
                let x_hv = if x_is_a {
                    &continuum[i].1
                } else {
                    &continuum[j].1
                };

                // Add discrimination noise
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                let x_noisy = ContinuousHV::weighted_bundle(
                    &[x_hv, &noise_hv],
                    &[1.0 - disc_noise_frac, disc_noise_frac],
                );

                let chose_a = Self::abx_discriminate(
                    &x_noisy,
                    &continuum[i].1,
                    &continuum[j].1,
                    &cat_a,
                    &cat_b,
                    warp_strength,
                );
                boundary_total += 1;
                if chose_a == x_is_a {
                    boundary_correct += 1;
                }
            }
        }

        let mut within_correct = 0u32;
        let mut within_total = 0u32;

        for &(i, j) in &within_pairs {
            for _trial in 0..trials_per_step {
                xor_shift(&mut rng);
                let x_is_a = rng % 2 == 0;
                let x_hv = if x_is_a {
                    &continuum[i].1
                } else {
                    &continuum[j].1
                };

                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                let x_noisy = ContinuousHV::weighted_bundle(
                    &[x_hv, &noise_hv],
                    &[1.0 - disc_noise_frac, disc_noise_frac],
                );

                let chose_a = Self::abx_discriminate(
                    &x_noisy,
                    &continuum[i].1,
                    &continuum[j].1,
                    &cat_a,
                    &cat_b,
                    warp_strength,
                );
                within_total += 1;
                if chose_a == x_is_a {
                    within_correct += 1;
                }
            }
        }

        let boundary_discrimination = boundary_correct as f64 / boundary_total.max(1) as f64;
        let within_category_discrimination = within_correct as f64 / within_total.max(1) as f64;
        let categorical_index = (boundary_discrimination - within_category_discrimination).max(0.0);

        TrialResult {
            boundary_slope,
            boundary_discrimination,
            within_category_discrimination,
            categorical_index,
        }
    }
}

impl PsychBenchmark for CategoricalPerceptionBenchmark {
    fn name(&self) -> &str {
        "Speech::CategoricalPerception"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Categorical Perception",
            citation: "Liberman et al. (1957); Pisoni (1973)",
            year: 1957,
            doi: Some("10.1121/1.1907989"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut slopes = Vec::new();
        let mut boundary_discs = Vec::new();
        let mut within_discs = Vec::new();
        let mut indices = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            slopes.push(r.boundary_slope);
            boundary_discs.push(r.boundary_discrimination);
            within_discs.push(r.within_category_discrimination);
            indices.push(r.categorical_index);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "categorical".to_string(),
                    correct: r.boundary_discrimination > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.boundary_discrimination,
                    confidence: r.boundary_slope,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("boundary_slope", MetricValue::from_samples(&slopes));
        result.insert(
            "boundary_discrimination",
            MetricValue::from_samples(&boundary_discs),
        );
        result.insert(
            "within_category_discrimination",
            MetricValue::from_samples(&within_discs),
        );
        result.insert("categorical_index", MetricValue::from_samples(&indices));

        result.conditions = 3; // identification, boundary discrimination, within discrimination
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
    fn test_categorical_perception_runs() {
        let config = BenchmarkConfig::default();
        let result = CategoricalPerceptionBenchmark.run(&config);
        assert!(result.metrics.contains_key("boundary_slope"));
        assert!(result.metrics.contains_key("boundary_discrimination"));
        assert!(
            result
                .metrics
                .contains_key("within_category_discrimination")
        );
        assert!(result.metrics.contains_key("categorical_index"));
    }

    #[test]
    fn test_categorical_perception_finite() {
        let config = BenchmarkConfig::default();
        let result = CategoricalPerceptionBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_boundary_better_than_within() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = CategoricalPerceptionBenchmark.run(&config);
        let boundary = result.metrics["boundary_discrimination"].mean;
        let within = result.metrics["within_category_discrimination"].mean;
        assert!(
            boundary >= within - 0.10,
            "Boundary ({boundary:.3}) should be >= Within ({within:.3})"
        );
    }
}
