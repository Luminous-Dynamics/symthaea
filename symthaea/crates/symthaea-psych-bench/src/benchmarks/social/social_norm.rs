// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Social Norm Violation Detection.
//!
//! Participants detect when an actor's behavior violates social norms in a
//! given context. Norm-congruent scenarios have matching actor-action-context;
//! violations swap one element, creating a mismatch.
//!
//! HDC implementation: Actor, action, and context are each ContinuousHVs.
//! Norm-congruent scenarios bind all three into a composite; violations
//! swap one component with a mismatched HV. Signal detection (d') measures
//! the ability to discriminate violations from congruent scenarios.
//!
//! Human baselines (Bicchieri, 2006; Krueger et al., 2012):
//! - d_prime: 2.10 (SD~0.50) — violation detection sensitivity
//! - detection_accuracy: 0.82 (SD~0.08)
//! - false_alarm_rate: 0.12 (SD~0.06)
//! - violation_rt_cost: 0.15 (SD~0.05) — RT increase for violations

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Social Norm Violation Detection benchmark.
pub struct SocialNormBenchmark;

struct TrialResult {
    d_prime: f64,
    detection_accuracy: f64,
    false_alarm_rate: f64,
    violation_rt_cost: f64,
    rt_ticks: f64,
}

impl SocialNormBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let diff_model = difficulty_model_for(self.name());
        let dim = config.dimension;
        let seed = config.trial_seed("social", "social_norm", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // 8 norm scenarios: each has actor, action, context prototypes
        let n_scenarios = 8;
        let actors: Vec<ContinuousHV> = (0..n_scenarios)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(i as u64 * 100 + 1)))
            .collect();
        let actions: Vec<ContinuousHV> = (0..n_scenarios)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(i as u64 * 100 + 50)))
            .collect();
        let contexts: Vec<ContinuousHV> = (0..n_scenarios)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(i as u64 * 100 + 99)))
            .collect();

        // Build norm prototypes: bind(actor, action, context)
        let norms: Vec<ContinuousHV> = (0..n_scenarios)
            .map(|i| {
                let ab = actors[i].bind(&actions[i]);
                ab.bind(&contexts[i])
            })
            .collect();

        // Time pressure and social config
        let noise_level: f32 = (0.20 + config.time_pressure as f32 * 0.15)
            * diff_model.interference_multiplier(config.difficulty) as f32;
        let social_boost: f32 = if config.enable_social { 0.25 } else { 0.0 };

        let n_items = 40; // 20 congruent + 20 violations
        let mut hits = 0u32; // correct violation detection
        let mut violation_total = 0u32;
        let mut false_alarms = 0u32; // congruent wrongly called violation
        let mut congruent_total = 0u32;
        let mut congruent_rt_sum = 0.0f64;
        let mut violation_rt_sum = 0.0f64;
        let mut total_rt_sum = 0.0f64;

        for item in 0..n_items {
            let is_violation = item >= n_items / 2;

            xor_shift(&mut rng);
            let scenario_idx = (rng % n_scenarios as u64) as usize;

            // Build stimulus
            let stimulus = if is_violation {
                // Swap one element: replace action with mismatched action
                xor_shift(&mut rng);
                let swap_idx =
                    ((scenario_idx + 1) + (rng % (n_scenarios as u64 - 1)) as usize) % n_scenarios;
                let ab = actors[scenario_idx].bind(&actions[swap_idx]);
                ab.bind(&contexts[scenario_idx])
            } else {
                norms[scenario_idx].clone()
            };

            // Compare stimulus to norm prototype
            // Encoding noise degrades norm representation fidelity
            let noise_degrade = config.effective_noise() as f32 * 0.4;
            let norm_sim =
                stimulus.similarity(&norms[scenario_idx]) * (1.0 - noise_degrade) + social_boost;
            xor_shift(&mut rng);
            let noise = (rng % 10000) as f32 / 10000.0 * noise_level;

            // Detection: low similarity to norm = violation detected
            let detection_threshold = 0.50;
            let detected_violation = (norm_sim + noise) < detection_threshold;

            if is_violation {
                violation_total += 1;
                if detected_violation {
                    hits += 1;
                }
                // Violations take longer (surprise processing)
                let base_rt = 7.0;
                let tp_speedup = config.time_pressure * 1.5;
                let rt = (base_rt - tp_speedup).max(1.0);
                violation_rt_sum += rt;
                total_rt_sum += rt;
            } else {
                congruent_total += 1;
                if detected_violation {
                    false_alarms += 1;
                }
                let base_rt = 5.0;
                let tp_speedup = config.time_pressure * 1.5;
                let rt = (base_rt - tp_speedup).max(1.0);
                congruent_rt_sum += rt;
                total_rt_sum += rt;
            }
        }

        let hit_rate = if violation_total > 0 {
            (hits as f64 / violation_total as f64).clamp(0.01, 0.99)
        } else {
            0.5
        };
        let fa_rate = if congruent_total > 0 {
            (false_alarms as f64 / congruent_total as f64).clamp(0.01, 0.99)
        } else {
            0.5
        };

        // d' = z(hit_rate) - z(false_alarm_rate)
        let d_prime = probit(hit_rate) - probit(fa_rate);

        let detection_acc = if (violation_total + congruent_total) > 0 {
            let correct = hits + (congruent_total - false_alarms);
            correct as f64 / (violation_total + congruent_total) as f64
        } else {
            0.0
        };

        let mean_congruent_rt = if congruent_total > 0 {
            congruent_rt_sum / congruent_total as f64
        } else {
            5.0
        };
        let mean_violation_rt = if violation_total > 0 {
            violation_rt_sum / violation_total as f64
        } else {
            7.0
        };
        let violation_rt_cost = if mean_congruent_rt > 0.0 {
            ((mean_violation_rt - mean_congruent_rt) / mean_congruent_rt).max(0.0)
        } else {
            0.0
        };

        TrialResult {
            d_prime,
            detection_accuracy: detection_acc,
            false_alarm_rate: fa_rate,
            violation_rt_cost,
            rt_ticks: total_rt_sum / n_items as f64,
        }
    }
}

/// Probit function (inverse normal CDF approximation).
fn probit(p: f64) -> f64 {
    let p = p.clamp(0.001, 0.999);
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 { -z } else { z }
}

impl PsychBenchmark for SocialNormBenchmark {
    fn name(&self) -> &str {
        "Social::SocialNorm"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Social Norm Violation Detection",
            citation: "Bicchieri (2006)",
            year: 2006,
            doi: Some("10.1017/CBO9780511616037"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut dprimes = Vec::new();
        let mut det_accs = Vec::new();
        let mut fa_rates = Vec::new();
        let mut rt_costs = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            dprimes.push(r.d_prime);
            det_accs.push(r.detection_accuracy);
            fa_rates.push(r.false_alarm_rate);
            rt_costs.push(r.violation_rt_cost);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "social_norm".to_string(),
                    correct: r.detection_accuracy > 0.5,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("d_prime", MetricValue::from_samples(&dprimes));
        result.insert("detection_accuracy", MetricValue::from_samples(&det_accs));
        result.insert("false_alarm_rate", MetricValue::from_samples(&fa_rates));
        result.insert("violation_rt_cost", MetricValue::from_samples(&rt_costs));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 2; // congruent vs violation
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
    fn test_social_norm_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = SocialNormBenchmark.run(&config);
        assert!(result.metrics.contains_key("d_prime"));
        assert!(result.metrics.contains_key("detection_accuracy"));
        assert!(result.metrics.contains_key("false_alarm_rate"));
        assert!(result.metrics.contains_key("violation_rt_cost"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_social_norm_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = SocialNormBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_social_norm_accuracy_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = SocialNormBenchmark.run(&config);
        let acc = result.metrics["detection_accuracy"].mean;
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "detection accuracy ({:.3}) out of bounds",
            acc
        );
    }

    #[test]
    fn test_social_norm_d_prime_positive() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = SocialNormBenchmark.run(&config);
        let dp = result.metrics["d_prime"].mean;
        assert!(
            dp > 0.0,
            "d' ({:.3}) should be positive for above-chance detection",
            dp
        );
    }

    #[test]
    fn test_social_norm_time_pressure() {
        let base = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            time_pressure: 0.0,
            ..Default::default()
        };
        let pressed = BenchmarkConfig {
            time_pressure: 1.0,
            ..base.clone()
        };
        let r_base = SocialNormBenchmark.run(&base);
        let r_press = SocialNormBenchmark.run(&pressed);
        let rt_base = r_base.metrics["rt_ticks"].mean;
        let rt_press = r_press.metrics["rt_ticks"].mean;
        assert!(
            rt_press <= rt_base + 0.5,
            "time pressure should reduce RT: base={:.2}, pressed={:.2}",
            rt_base,
            rt_press
        );
    }
}
