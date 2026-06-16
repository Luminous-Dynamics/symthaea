// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sustained Attention to Response Task (SART).
//!
//! Participants respond to every digit except one target (e.g., "3").
//! Measures sustained attention via commission errors (responding to
//! the no-go target) and omission errors (failing to respond to go stimuli).
//!
//! HDC implementation: Each digit encoded as ContinuousHV. Target "3" is
//! a specific HV that must be inhibited. Automatic response tendency builds
//! via repetition; target requires overriding this prepotent response.
//!
//! Human baselines (Robertson et al., 1997; Manly et al., 1999):
//! - commission_errors: 0.30 (SD≈0.12) — false alarms to no-go target
//! - omission_errors: 0.04 (SD≈0.03) — misses to go stimuli
//! - d_prime: 2.50 (SD≈0.60) — signal detection sensitivity

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// SART benchmark.
pub struct SartBenchmark;

struct TrialResult {
    commission_errors: f64,
    omission_errors: f64,
    d_prime: f64,
    rt_ticks: f64,
}

impl SartBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("sustained", "sart", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Encode digits 1-9 as HVs
        let digits: Vec<ContinuousHV> = (1..=9)
            .map(|d| ContinuousHV::random(dim, seed.wrapping_add(d as u64 * 111)))
            .collect();

        // Target digit (no-go): digit "3" (index 2)
        let target_idx = 2;
        let target_hv = &digits[target_idx];

        let total_stimuli = 225; // Standard SART: 225 trials (25 per digit)

        // Time pressure: raises response criterion noise, modeling impulsive responding
        // under deadline (Robertson et al., 1997).
        let diff_model = difficulty_model_for(self.name());
        let tp_noise: f32 = (0.10 + config.time_pressure as f32 * 0.15)
            / diff_model.signal_multiplier(config.difficulty) as f32;

        // Automatic response tendency builds over go trials.
        // Starts moderate — the system begins cautious and becomes more
        // impulsive with repeated go trials (Robertson et al., 1997).
        // Higher difficulty → weaker top-down inhibitory control → higher starting
        // tendency and faster habituation (more commission errors).
        let signal_mult = diff_model.signal_multiplier(config.difficulty) as f32;
        let difficulty_tendency_boost: f32 = config.difficulty as f32 * 0.12;
        let mut response_tendency: f32 = 0.40 + difficulty_tendency_boost;
        let tendency_growth: f32 =
            0.003 * diff_model.temperature_multiplier(config.difficulty) as f32;

        let mut commission = 0u32; // false alarm to target
        let mut commission_total = 0u32;
        let mut omission = 0u32; // miss on go trial
        let mut go_total = 0u32;
        let mut rt_sum = 0.0f64;
        let mut rt_count = 0u32;

        for trial in 0..total_stimuli {
            xor_shift(&mut rng);
            let digit_idx = (rng % 9) as usize;
            let is_target = digit_idx == target_idx;
            let stimulus = &digits[digit_idx];

            // Similarity to target (inhibition signal)
            // Encoding noise degrades digit discrimination
            let noise_degrade = config.effective_noise() as f32 * 0.4;
            let target_sim = stimulus.similarity(target_hv) * (1.0 - noise_degrade);

            // Response decision: high response_tendency → likely to respond
            // Must inhibit when target_sim is high
            xor_shift(&mut rng);
            let noise = (rng % 10000) as f32 / 10000.0 * tp_noise;

            // Difficulty degrades target salience → weaker inhibition cue.
            let inhibit_threshold = 0.45 + target_sim * 0.30 * signal_mult;
            let respond = response_tendency + noise > inhibit_threshold;

            if is_target {
                commission_total += 1;
                if respond {
                    commission += 1; // commission error
                // Failed inhibition: response tendency remains high
                // (error-related negativity effect — Hester et al., 2005)
                } else {
                    // Successful inhibition: temporarily suppresses automatic
                    // response tendency (post-inhibition slowing, Robertson et al., 1997).
                    // This models the "attention reset" after catching a target.
                    response_tendency = (response_tendency - 0.08).max(0.3);
                }
            } else {
                go_total += 1;
                if !respond {
                    omission += 1; // omission error
                } else {
                    // RT: faster with higher response tendency
                    let base_rt = 5.0 - response_tendency as f64 * 2.0;
                    let tp_speedup = config.time_pressure * 1.0;
                    rt_sum += (base_rt - tp_speedup).max(1.0);
                    rt_count += 1;
                }
                // Go trials increase automatic response tendency
                response_tendency = (response_tendency + tendency_growth).min(0.95);
            }

            // Vigilance decrement: every 50 trials, attention lapses slightly
            if trial > 0 && trial % 50 == 0 {
                response_tendency = (response_tendency + 0.015).min(0.95);
            }
        }

        let comm_rate = if commission_total > 0 {
            commission as f64 / commission_total as f64
        } else {
            0.0
        };
        let omit_rate = if go_total > 0 {
            omission as f64 / go_total as f64
        } else {
            0.0
        };

        // d' = z(hit_rate) - z(false_alarm_rate)
        // hit_rate = 1 - omission_rate (correct go responses)
        // false_alarm_rate = commission_rate
        let hit_rate = (1.0 - omit_rate).clamp(0.01, 0.99);
        let fa_rate = comm_rate.clamp(0.01, 0.99);
        let z_hit = probit(hit_rate);
        let z_fa = probit(fa_rate);
        let d_prime = z_hit - z_fa;

        let mean_rt = if rt_count > 0 {
            rt_sum / rt_count as f64
        } else {
            5.0
        };

        TrialResult {
            commission_errors: comm_rate,
            omission_errors: omit_rate,
            d_prime,
            rt_ticks: mean_rt,
        }
    }
}

/// Probit function (inverse normal CDF approximation).
fn probit(p: f64) -> f64 {
    // Rational approximation (Abramowitz & Stegun 26.2.23)
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

impl PsychBenchmark for SartBenchmark {
    fn name(&self) -> &str {
        "SustainedAttention::SART"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Sustained Attention to Response Task",
            citation: "Robertson et al. (1997)",
            year: 1997,
            doi: Some("10.1016/S0028-3932(97)00015-2"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut commissions = Vec::new();
        let mut omissions = Vec::new();
        let mut dprimes = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            commissions.push(r.commission_errors);
            omissions.push(r.omission_errors);
            dprimes.push(r.d_prime);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "sart".to_string(),
                    correct: r.d_prime > 0.0,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("commission_errors", MetricValue::from_samples(&commissions));
        result.insert("omission_errors", MetricValue::from_samples(&omissions));
        result.insert("d_prime", MetricValue::from_samples(&dprimes));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 2; // go vs no-go
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
    fn test_sart_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = SartBenchmark.run(&config);
        assert!(result.metrics.contains_key("commission_errors"));
        assert!(result.metrics.contains_key("omission_errors"));
        assert!(result.metrics.contains_key("d_prime"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_sart_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = SartBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_sart_commission_rate_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = SartBenchmark.run(&config);
        let comm = result.metrics["commission_errors"].mean;
        assert!(
            comm >= 0.0 && comm <= 1.0,
            "commission rate ({:.3}) out of bounds",
            comm
        );
    }

    #[test]
    fn test_sart_time_pressure() {
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
        let r_base = SartBenchmark.run(&base);
        let r_press = SartBenchmark.run(&pressed);
        // Under time pressure, commission errors should increase or RT should decrease
        let rt_base = r_base.metrics["rt_ticks"].mean;
        let rt_press = r_press.metrics["rt_ticks"].mean;
        assert!(
            rt_press <= rt_base + 0.5,
            "time pressure should reduce or maintain RT: base={:.2}, pressed={:.2}",
            rt_base,
            rt_press
        );
    }
}
