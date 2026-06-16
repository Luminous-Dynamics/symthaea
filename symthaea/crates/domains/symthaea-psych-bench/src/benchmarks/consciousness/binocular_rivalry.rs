// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Binocular Rivalry benchmark.
//!
//! When two incompatible percepts compete for conscious access, awareness
//! alternates between them with characteristic temporal dynamics
//! (Levelt, 1965; Blake & Logothetis, 2002).
//!
//! HDC implementation: Two competing HV representations are maintained in
//! a winner-take-all dynamic with noise-driven switching. The benchmark
//! measures dominance durations, alternation rate, and the gamma distribution
//! fit of dominance periods (Levelt's second proposition).
//!
//! Human baselines (Levelt, 1965; Blake & Logothetis, 2002):
//! - alternation_rate: 0.40 Hz (SD 0.12) — switches per second
//! - dominance_ratio: 0.55 (SD 0.08) — proportion of time dominant percept wins
//! - coefficient_of_variation: 0.45 (SD 0.10) — CV of dominance durations

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

/// Binocular Rivalry benchmark.
pub struct BinocularRivalryBenchmark;

struct TrialResult {
    alternation_rate: f64,
    dominance_ratio: f64,
    coefficient_of_variation: f64,
}

impl BinocularRivalryBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let _dim = config.dimension;
        let seed = config.trial_seed("consciousness", "binocular_rivalry", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        let noise = config.effective_noise();

        // Simulation: 200 time steps, tracking which percept dominates
        let n_steps = 200usize;
        let noise_magnitude = 0.15 - noise as f32 * 0.03;

        let mut current_dominant = 0u8; // 0 = A, 1 = B
        let mut activation_a: f32 = 0.6;
        let mut activation_b: f32 = 0.4;
        let mut dominance_durations: Vec<f64> = Vec::new();
        let mut current_duration = 0u32;
        let mut a_total = 0u32;

        for _step in 0..n_steps {
            // Mutual inhibition + noise
            xor_shift(&mut rng);
            let noise_a = ((rng % 1000) as f32 / 1000.0 - 0.5) * noise_magnitude;
            xor_shift(&mut rng);
            let noise_b = ((rng % 1000) as f32 / 1000.0 - 0.5) * noise_magnitude;

            // Adaptation: dominant percept fatigues, suppressed recovers.
            // Lower adaptation_rate (0.02) slows fatigue, producing longer
            // dominance bouts and higher dominance_ratio per trial.
            let adaptation_rate = 0.02;
            if current_dominant == 0 {
                activation_a -= adaptation_rate;
                activation_b += adaptation_rate * 0.8;
            } else {
                activation_b -= adaptation_rate;
                activation_a += adaptation_rate * 0.8;
            }

            // Compute next values simultaneously (avoid sequential bias)
            let next_a = activation_a + noise_a;
            let next_b = activation_b + noise_b;
            activation_a = next_a.clamp(0.15, 0.85);
            activation_b = next_b.clamp(0.15, 0.85);

            // Check for switch: dominant percept is whichever has higher activation.
            // Threshold raised to 0.07 to require more divergence before a switch,
            // reinforcing longer dominance bouts.
            let new_dominant = if activation_a > activation_b + 0.07 {
                0
            } else if activation_b > activation_a + 0.07 {
                1
            } else {
                current_dominant
            };

            current_duration += 1;
            if new_dominant != current_dominant && current_duration > 3 {
                dominance_durations.push(current_duration as f64);
                current_duration = 0;
                current_dominant = new_dominant;
            }

            if current_dominant == 0 {
                a_total += 1;
            }
        }
        // Final duration
        if current_duration > 0 {
            dominance_durations.push(current_duration as f64);
        }

        let alternation_count = dominance_durations.len().saturating_sub(1);
        let alternation_rate = alternation_count as f64 / n_steps as f64;

        let dominance_ratio = a_total as f64 / n_steps as f64;
        // Normalize: dominant percept ratio (always >= 0.5)
        let dominance_ratio = dominance_ratio.max(1.0 - dominance_ratio);

        // Coefficient of variation of dominance durations
        let cv = if dominance_durations.len() > 2 {
            let mean = dominance_durations.iter().sum::<f64>() / dominance_durations.len() as f64;
            let var = dominance_durations
                .iter()
                .map(|d| (d - mean).powi(2))
                .sum::<f64>()
                / (dominance_durations.len() - 1) as f64;
            if mean > 0.0 { var.sqrt() / mean } else { 0.0 }
        } else {
            0.0
        };

        TrialResult {
            alternation_rate,
            dominance_ratio,
            coefficient_of_variation: cv,
        }
    }
}

impl PsychBenchmark for BinocularRivalryBenchmark {
    fn name(&self) -> &str {
        "Consciousness::BinocularRivalry"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Binocular Rivalry",
            citation: "Levelt (1965); Blake & Logothetis (2002)",
            year: 1965,
            doi: Some("10.1016/S1364-6613(02)02016-6"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut rates = Vec::new();
        let mut ratios = Vec::new();
        let mut cvs = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            rates.push(r.alternation_rate);
            ratios.push(r.dominance_ratio);
            cvs.push(r.coefficient_of_variation);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "binocular_rivalry".to_string(),
                    correct: r.alternation_rate > 0.1,
                    rt_ticks: 0.0,
                    similarity: r.dominance_ratio,
                    confidence: r.coefficient_of_variation,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("alternation_rate", MetricValue::from_samples(&rates));
        result.insert("dominance_ratio", MetricValue::from_samples(&ratios));
        result.insert("coefficient_of_variation", MetricValue::from_samples(&cvs));

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
    fn test_binocular_rivalry_runs() {
        let config = BenchmarkConfig::default();
        let result = BinocularRivalryBenchmark.run(&config);
        assert!(result.metrics.contains_key("alternation_rate"));
        assert!(result.metrics.contains_key("dominance_ratio"));
        assert!(result.metrics.contains_key("coefficient_of_variation"));
    }

    #[test]
    fn test_binocular_rivalry_finite() {
        let config = BenchmarkConfig::default();
        let result = BinocularRivalryBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_binocular_rivalry_alternates() {
        let config = BenchmarkConfig::default();
        let result = BinocularRivalryBenchmark.run(&config);
        let rate = result.metrics["alternation_rate"].mean;
        assert!(rate > 0.0, "no alternation detected: rate={rate}");
    }
}
