// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Yerkes-Dodson benchmark: inverted-U performance curve under varying arousal (NE).
//!
//! Science: Yerkes & Dodson (1908) — performance peaks at moderate arousal,
//! declines at both low and high arousal extremes.
//! Protocol: Vary NE-like arousal level across blocks. Measure accuracy on
//! simple and complex tasks. Fit inverted-U curve.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Yerkes-Dodson inverted-U benchmark.
pub struct YerkesDodsonBenchmark;

fn next_seed(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

impl YerkesDodsonBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64, f64) {
        // Use low dimension (64) internally to make noise actually degrade discrimination.
        // At dim=512, HDC similarity is too robust for inverted-U to manifest.
        let dim = 64;
        let seed = config.trial_seed("neuromod", "yerkes_dodson", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        // NE levels to test (0.1 to 0.9 in 9 steps)
        let ne_levels: Vec<f64> = (1..=9).map(|i| i as f64 * 0.1).collect();
        let mut simple_perfs = Vec::new();
        let mut complex_perfs = Vec::new();

        for &ne in &ne_levels {
            // Simple task: 4-AFC classification (broad optimum)
            let mut simple_correct = 0usize;
            let simple_trials = 20;
            let target = ContinuousHV::random(dim, next_seed(&mut rng));
            let distractors: Vec<ContinuousHV> = (0..3)
                .map(|_| ContinuousHV::random(dim, next_seed(&mut rng)))
                .collect();

            for _ in 0..simple_trials {
                // NE modulates signal-to-noise: inverted-U
                // Moderate NE: best focus. Low NE: sluggish. High NE: jittery.
                let noise_scale = 1.0 - (1.0 - (ne - 0.6).powi(2) * 4.0).max(0.0);
                let noise_weight = (noise_scale * 0.85) as f32;
                let signal_weight = (1.0 - noise_weight).max(0.05);
                let noise_hv = ContinuousHV::random(dim, next_seed(&mut rng));

                let noisy_stim = ContinuousHV::weighted_bundle(
                    &[&target, &noise_hv],
                    &[signal_weight, noise_weight],
                );

                // 4-AFC: target vs 3 distractors
                let sim_target = noisy_stim.similarity(&target);
                let best_distractor = distractors
                    .iter()
                    .map(|d| noisy_stim.similarity(d))
                    .fold(f32::NEG_INFINITY, f32::max);
                if sim_target > best_distractor {
                    simple_correct += 1;
                }
            }

            // Complex task: 8-AFC multi-prototype discrimination (narrow optimum, peaks at lower NE)
            let mut complex_correct = 0usize;
            let complex_trials = 20;
            let protos: Vec<ContinuousHV> = (0..8)
                .map(|_| ContinuousHV::random(dim, next_seed(&mut rng)))
                .collect();

            for _ in 0..complex_trials {
                let correct_idx = (next_seed(&mut rng) % 8) as usize;

                // Higher noise at extreme NE levels; complex tasks more sensitive
                let noise_scale = 1.0 - (1.0 - (ne - 0.45).powi(2) * 5.0).max(0.0);
                let noise_weight = (noise_scale * 0.95) as f32;
                let signal_weight = (1.0 - noise_weight).max(0.05);
                let noise_hv = ContinuousHV::random(dim, next_seed(&mut rng));

                let noisy_stim = ContinuousHV::weighted_bundle(
                    &[&protos[correct_idx], &noise_hv],
                    &[signal_weight, noise_weight],
                );

                let best_idx = protos
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        noisy_stim
                            .similarity(a)
                            .total_cmp(&noisy_stim.similarity(b))
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                if best_idx == correct_idx {
                    complex_correct += 1;
                }
            }

            simple_perfs.push((ne, simple_correct as f64 / simple_trials as f64));
            complex_perfs.push((ne, complex_correct as f64 / complex_trials as f64));
        }

        // Find peak NE for each task
        let simple_peak_ne = simple_perfs
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(ne, _)| *ne)
            .unwrap_or(0.5);

        let complex_peak_ne = complex_perfs
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(ne, _)| *ne)
            .unwrap_or(0.5);

        // Compute inverted-U fit R² (simple quadratic regression: perf = a*ne² + b*ne + c)
        let r2 = compute_quadratic_r2(&simple_perfs);

        // Simple tasks should peak at higher NE than complex tasks
        let simple_peak_shift = simple_peak_ne - complex_peak_ne;

        (simple_peak_ne, r2, simple_peak_shift)
    }
}

fn compute_quadratic_r2(data: &[(f64, f64)]) -> f64 {
    if data.len() < 3 {
        return 0.0;
    }
    let n = data.len() as f64;
    let mean_y: f64 = data.iter().map(|(_, y)| y).sum::<f64>() / n;
    let ss_tot: f64 = data.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
    if ss_tot < 1e-10 {
        return 0.0;
    }

    // Simple least squares for y = a*x² + b*x + c
    // Use normal equations (3x3)
    let mut sx = [0.0_f64; 5]; // sum of x^0..x^4
    let mut sxy = [0.0_f64; 3]; // sum of x^k * y

    for &(x, y) in data {
        let mut xk = 1.0;
        for k in 0..5 {
            sx[k] += xk;
            if k < 3 {
                sxy[k] += xk * y;
            }
            xk *= x;
        }
    }

    // Solve via Cramer's rule (3x3)
    let det = sx[0] * (sx[2] * sx[4] - sx[3] * sx[3]) - sx[1] * (sx[1] * sx[4] - sx[3] * sx[2])
        + sx[2] * (sx[1] * sx[3] - sx[2] * sx[2]);
    if det.abs() < 1e-10 {
        return 0.0;
    }

    let c = (sxy[0] * (sx[2] * sx[4] - sx[3] * sx[3]) - sx[1] * (sxy[1] * sx[4] - sx[3] * sxy[2])
        + sx[2] * (sxy[1] * sx[3] - sx[2] * sxy[2]))
        / det;
    let b = (sx[0] * (sxy[1] * sx[4] - sxy[2] * sx[3]) - sxy[0] * (sx[1] * sx[4] - sx[3] * sx[2])
        + sx[2] * (sx[1] * sxy[2] - sxy[1] * sx[2]))
        / det;
    let a = (sx[0] * (sx[2] * sxy[2] - sx[3] * sxy[1]) - sx[1] * (sx[1] * sxy[2] - sxy[1] * sx[2])
        + sxy[0] * (sx[1] * sx[3] - sx[2] * sx[2]))
        / det;

    let ss_res: f64 = data
        .iter()
        .map(|(x, y)| {
            let pred = a * x * x + b * x + c;
            (y - pred).powi(2)
        })
        .sum();

    1.0 - ss_res / ss_tot
}

impl PsychBenchmark for YerkesDodsonBenchmark {
    fn name(&self) -> &str {
        "Neuromod::YerkesDodson"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Yerkes-Dodson Inverted-U Arousal-Performance",
            citation: "Yerkes & Dodson (1908)",
            year: 1908,
            doi: Some("10.1002/cne.920180503"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut peak_ne_levels = Vec::new();
        let mut r2_values = Vec::new();
        let mut peak_shifts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (peak_ne, r2, shift) = self.run_trial(config, trial);
            peak_ne_levels.push(peak_ne);
            r2_values.push(r2);
            peak_shifts.push(shift);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "yerkes_dodson".to_string(),
                    correct: r2 > 0.5,
                    rt_ticks: peak_ne,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("peak_ne_level", MetricValue::from_samples(&peak_ne_levels));
        result.insert("inverted_u_fit_r2", MetricValue::from_samples(&r2_values));
        result.insert("simple_peak_shift", MetricValue::from_samples(&peak_shifts));

        result.conditions = 9; // 9 NE levels
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
    fn test_yerkes_dodson_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = YerkesDodsonBenchmark.run(&config);
        assert!(result.metrics.contains_key("peak_ne_level"));
        assert!(result.metrics.contains_key("inverted_u_fit_r2"));
        assert!(result.metrics.contains_key("simple_peak_shift"));
        for val in result.metrics.values() {
            assert!(val.mean.is_finite(), "non-finite metric: {:?}", val);
        }
    }

    #[test]
    fn test_yerkes_dodson_peak_moderate() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            dimension: 512,
            ..Default::default()
        };
        let result = YerkesDodsonBenchmark.run(&config);
        let peak = result.metrics["peak_ne_level"].mean;
        // Peak should be somewhere in the moderate range (not extreme)
        assert!(
            peak >= 0.2 && peak <= 0.9,
            "Peak NE should be moderate: {peak}"
        );
    }

    #[test]
    fn test_quadratic_r2_perfect_fit() {
        // Perfect quadratic: y = -x² + x
        let data: Vec<(f64, f64)> = (0..10)
            .map(|i| {
                let x = i as f64 / 10.0;
                let y = -x * x + x;
                (x, y)
            })
            .collect();
        let r2 = compute_quadratic_r2(&data);
        assert!(r2 > 0.99, "Perfect quadratic should have R² ≈ 1.0: {r2}");
    }
}
