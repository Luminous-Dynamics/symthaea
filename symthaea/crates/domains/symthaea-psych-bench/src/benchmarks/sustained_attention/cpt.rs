// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Continuous Performance Task (CPT).
//!
//! Participants monitor a stream of digits and respond when the current digit
//! matches the digit presented 2 positions back (2-back variant). Measures
//! sustained attention and working memory load via signal detection (d').
//!
//! HDC implementation: Digit HVs are encoded as ContinuousHVs. A 2-back buffer
//! tracks recent stimuli. Target detection requires similarity between current
//! and 2-back HV to exceed a threshold. Vigilance decrement modeled as rising
//! threshold over blocks.
//!
//! Human baselines (Rosvold et al., 1956; Riccio et al., 2002):
//! - d_prime: 2.80 (SD~0.60) — detection sensitivity
//! - hit_rate: 0.85 (SD~0.08) — correct target detection
//! - false_alarm_rate: 0.08 (SD~0.05) — incorrect responses to non-targets
//! - vigilance_decrement: 0.03 (SD~0.02) — d' decrease per block

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// CPT benchmark.
pub struct CptBenchmark;

struct TrialResult {
    d_prime: f64,
    hit_rate: f64,
    false_alarm_rate: f64,
    vigilance_decrement: f64,
    rt_ticks: f64,
}

impl CptBenchmark {
    fn run_trial(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
        trace: &mut Vec<TrialOutcome>,
        global_trial_idx: &mut usize,
    ) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("sustained", "cpt", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Digit HVs (0-9)
        let n_digits = 10;
        let digits: Vec<ContinuousHV> = (0..n_digits)
            .map(|d| ContinuousHV::random(dim, seed.wrapping_add(d as u64 * 111 + 1)))
            .collect();

        let total_trials = 300;
        let target_rate = 0.15; // 15% targets
        let n_blocks = 6;
        let trials_per_block = total_trials / n_blocks;

        // Time pressure noise
        let diff_model = difficulty_model_for(self.name());
        let noise_level: f32 = (0.15 + config.time_pressure as f32 * 0.15)
            / diff_model.signal_multiplier(config.difficulty) as f32;

        // Vigilance decrement: threshold rises per block
        let threshold_increment: f32 = 0.005;

        // 2-back buffer
        let mut buffer: Vec<usize> = Vec::with_capacity(total_trials);
        let mut hits = 0u32;
        let mut target_total = 0u32;
        let mut false_alarms = 0u32;
        let mut nontarget_total = 0u32;
        let mut rt_sum = 0.0f64;
        let mut rt_count = 0u32;

        // Per-block d' for vigilance decrement
        let mut block_hits = vec![0u32; n_blocks];
        let mut block_targets = vec![0u32; n_blocks];
        let mut block_fa = vec![0u32; n_blocks];
        let mut block_nontargets = vec![0u32; n_blocks];

        for trial in 0..total_trials {
            let block = trial / trials_per_block;
            let block = block.min(n_blocks - 1);

            // Decide if this should be a target (match 2-back)
            xor_shift(&mut rng);
            let want_target = (rng % 100) as f64 / 100.0 < target_rate && buffer.len() >= 2;

            let digit_idx = if want_target {
                // Present same digit as 2 positions back
                buffer[buffer.len() - 2]
            } else {
                // Random digit (avoid matching 2-back)
                xor_shift(&mut rng);
                let mut d = (rng % n_digits as u64) as usize;
                if buffer.len() >= 2 && d == buffer[buffer.len() - 2] {
                    d = (d + 1) % n_digits;
                }
                d
            };

            let is_target = buffer.len() >= 2 && digit_idx == buffer[buffer.len() - 2];
            buffer.push(digit_idx);

            // Detection: compare current to 2-back
            if buffer.len() >= 3 {
                let current = &digits[digit_idx];
                let two_back = &digits[buffer[buffer.len() - 3]];
                // Encoding noise degrades digit discrimination
                let noise_degrade = config.effective_noise() as f32 * 0.4;
                let similarity = current.similarity(two_back) * (1.0 - noise_degrade);

                xor_shift(&mut rng);
                let noise = (rng % 10000) as f32 / 10000.0 * noise_level;

                // Threshold rises with vigilance decrement
                let base_threshold: f32 = 0.50;
                let threshold = base_threshold
                    + threshold_increment * block as f32
                    + config.difficulty as f32 * 0.15;

                let responded = (similarity + noise) > threshold;

                let correct;
                let trial_rt;
                if is_target {
                    target_total += 1;
                    block_targets[block] += 1;
                    correct = responded;
                    if responded {
                        hits += 1;
                        block_hits[block] += 1;
                        // RT for hits
                        let base_rt = 6.0 + block as f64 * 0.2; // fatigue
                        let tp_speedup = config.time_pressure * 1.0;
                        trial_rt = (base_rt - tp_speedup).max(1.0);
                        rt_sum += trial_rt;
                        rt_count += 1;
                    } else {
                        trial_rt = 0.0;
                    }
                } else {
                    nontarget_total += 1;
                    block_nontargets[block] += 1;
                    correct = !responded;
                    trial_rt = 0.0;
                    if responded {
                        false_alarms += 1;
                        block_fa[block] += 1;
                    }
                }

                if config.trial_trace {
                    trace.push(TrialOutcome {
                        trial_idx: *global_trial_idx,
                        condition: if is_target {
                            "target".to_string()
                        } else {
                            "nontarget".to_string()
                        },
                        correct,
                        rt_ticks: trial_rt,
                        similarity: similarity as f64,
                        confidence: 0.0,
                        response_idx: if responded { 1 } else { 0 },
                        extra: BTreeMap::new(),
                    });
                    *global_trial_idx += 1;
                }
            }
        }

        let hit_rate_val = if target_total > 0 {
            (hits as f64 / target_total as f64).clamp(0.01, 0.99)
        } else {
            0.5
        };
        let fa_rate_val = if nontarget_total > 0 {
            (false_alarms as f64 / nontarget_total as f64).clamp(0.01, 0.99)
        } else {
            0.01
        };

        let d_prime = probit(hit_rate_val) - probit(fa_rate_val);

        // Vigilance decrement: slope of per-block d' via linear regression
        let mut block_dprimes = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let bhr = if block_targets[b] > 0 {
                (block_hits[b] as f64 / block_targets[b] as f64).clamp(0.01, 0.99)
            } else {
                0.5
            };
            let bfar = if block_nontargets[b] > 0 {
                (block_fa[b] as f64 / block_nontargets[b] as f64).clamp(0.01, 0.99)
            } else {
                0.01
            };
            block_dprimes.push(probit(bhr) - probit(bfar));
        }

        let n = n_blocks as f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut sum_xx = 0.0f64;
        for (i, &dp) in block_dprimes.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += dp;
            sum_xy += x * dp;
            sum_xx += x * x;
        }
        let denom = n * sum_xx - sum_x * sum_x;
        let vigilance_decrement = if denom.abs() > 1e-10 {
            // Negative slope = decrement (we report absolute value)
            ((n * sum_xy - sum_x * sum_y) / denom).abs()
        } else {
            0.0
        };

        let mean_rt = if rt_count > 0 {
            rt_sum / rt_count as f64
        } else {
            6.0
        };

        TrialResult {
            d_prime,
            hit_rate: hit_rate_val,
            false_alarm_rate: fa_rate_val,
            vigilance_decrement,
            rt_ticks: mean_rt,
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

impl PsychBenchmark for CptBenchmark {
    fn name(&self) -> &str {
        "SustainedAttention::CPT"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Continuous Performance Task",
            citation: "Rosvold et al. (1956)",
            year: 1956,
            doi: Some("10.1037/t02624-000"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut dprimes = Vec::new();
        let mut hit_rates = Vec::new();
        let mut fa_rates = Vec::new();
        let mut decrements = Vec::new();
        let mut rts = Vec::new();
        let mut trace = Vec::new();
        let mut global_trial_idx = 0usize;

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial, &mut trace, &mut global_trial_idx);
            dprimes.push(r.d_prime);
            hit_rates.push(r.hit_rate);
            fa_rates.push(r.false_alarm_rate);
            decrements.push(r.vigilance_decrement);
            rts.push(r.rt_ticks);
        }

        result.insert("d_prime", MetricValue::from_samples(&dprimes));
        result.insert("hit_rate", MetricValue::from_samples(&hit_rates));
        result.insert("false_alarm_rate", MetricValue::from_samples(&fa_rates));
        result.insert(
            "vigilance_decrement",
            MetricValue::from_samples(&decrements),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 2; // target vs non-target
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpt_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = CptBenchmark.run(&config);
        assert!(result.metrics.contains_key("d_prime"));
        assert!(result.metrics.contains_key("hit_rate"));
        assert!(result.metrics.contains_key("false_alarm_rate"));
        assert!(result.metrics.contains_key("vigilance_decrement"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_cpt_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = CptBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_cpt_d_prime_positive() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = CptBenchmark.run(&config);
        let dp = result.metrics["d_prime"].mean;
        assert!(
            dp > 0.0,
            "d' ({:.3}) should be positive for above-chance detection",
            dp
        );
    }

    #[test]
    fn test_cpt_hit_rate_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = CptBenchmark.run(&config);
        let hr = result.metrics["hit_rate"].mean;
        assert!(hr >= 0.0 && hr <= 1.0, "hit rate ({:.3}) out of bounds", hr);
    }

    #[test]
    fn test_cpt_time_pressure() {
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
        let r_base = CptBenchmark.run(&base);
        let r_press = CptBenchmark.run(&pressed);
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
