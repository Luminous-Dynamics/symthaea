// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Attentional Blink Task.
//!
//! Tests temporal attention limits in rapid serial visual presentation (RSVP).
//! Two targets (T1, T2) embedded in a stream of distractors. T2 detection
//! drops when it follows T1 by 200-500ms (lag 2-5).
//!
//! HDC implementation: T1 encoding consumes WM attention capacity, degrading
//! T2 representation at short lags. At longer lags, attention recovers.
//!
//! Human baselines (Raymond et al. 1992; Shapiro et al. 1997):
//! - t1_accuracy: 0.92 (SD≈0.05)
//! - lag3_t2_accuracy: 0.55 (SD≈0.15)
//! - lag8_t2_accuracy: 0.85 (SD≈0.10)
//! - blink_magnitude: 0.30 (SD≈0.12)

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::ssm_temporal::SsmTemporalBackend;
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Attentional Blink benchmark.
pub struct AttentionalBlinkBenchmark;

struct TrialResult {
    t1_accuracy: f64,
    lag3_t2_accuracy: f64,
    lag8_t2_accuracy: f64,
    blink_magnitude: f64,
    t1_rt_ticks: Vec<f64>,
    t2_rt_ticks_by_lag: [Vec<f64>; 4],
}

impl AttentionalBlinkBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("attention", "blink", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // SSM temporal backend for attention recovery (A=-0.15 for moderate recovery rate)
        let mut ssm = SsmTemporalBackend::new(-0.15, 4);

        // Create target and distractor category prototypes with shared features
        // (visual similarity makes depleted-attention detection harder)
        let shared_features = ContinuousHV::random(dim, seed.wrapping_add(50));
        let target_unique = ContinuousHV::random(dim, seed.wrapping_add(100));
        let distractor_unique = ContinuousHV::random(dim, seed.wrapping_add(200));
        let category_overlap: f32 = 0.35;
        let target_category = ContinuousHV::weighted_bundle(
            &[&target_unique, &shared_features],
            &[1.0 - category_overlap, category_overlap],
        );
        let distractor_category = ContinuousHV::weighted_bundle(
            &[&distractor_unique, &shared_features],
            &[1.0 - category_overlap, category_overlap],
        );

        // Attention depletion model parameters
        // Working memory capacity modulates attention pool — higher WM capacity
        // supports greater attentional resource availability (Vogel et al., 2005).
        // Lapse_rate reduces effective capacity and increases T1 cost, modeling
        // attention-gate efficiency (Dux & Marois, 2009; individual differences
        // in attentional control capacity).
        let wm_bonus = (config.working_memory_capacity as f64 - 4.0) * 0.04;
        let lapse_capacity_penalty = config.lapse_rate * 0.35; // up to -8.75% capacity
        let attention_capacity: f64 = (1.0 + wm_bonus - lapse_capacity_penalty).clamp(0.7, 1.3);
        let t1_cost: f64 = 0.75 + config.lapse_rate * 0.30; // lapse increases T1 consolidation cost
        let recovery_rate: f64 = 0.12; // Per-lag recovery
        // Encoding noise degrades target discrimination
        let enc_noise = config.effective_noise() as f32;

        // Time pressure: base 0.30 matches ~50% T2|T1 accuracy at lag-2 (Raymond et al., 1992 AB);
        // +0.15/unit degrades target discrimination, modeling attention-gate narrowing under SAT (Heitz, 2014).
        let diff_model = difficulty_model_for(self.name());
        let temperature: f64 = (0.30 + config.time_pressure * 0.15)
            * diff_model.temperature_multiplier(config.difficulty);
        let trials_per_lag = 25;
        let lags = [2, 3, 5, 8];

        let mut t1_correct_total = 0u32;
        let mut t1_total = 0u32;
        let mut lag_correct: [u32; 4] = [0; 4]; // indexed by lag position in array
        let mut t1_rt_ticks = Vec::new();
        let mut t2_rt_ticks_by_lag: [Vec<f64>; 4] =
            [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        for (lag_idx, &lag) in lags.iter().enumerate() {
            for inner_trial in 0..trials_per_lag {
                // Build RSVP stream of 15 items
                let stream_len = 15;
                let t1_pos = 3; // T1 always at position 3
                let t2_pos = t1_pos + lag;

                if t2_pos >= stream_len {
                    continue;
                }

                // Generate T1 as noisy target
                xor_shift(&mut rng);
                let t1_noise = ContinuousHV::random(dim, rng);
                let t1 =
                    ContinuousHV::weighted_bundle(&[&target_category, &t1_noise], &[0.80, 0.20]);

                // Detect T1 (similarity to target category)
                // Encoding noise adds per-comparison noise (individual perceptual noise)
                let t1_noise_a = {
                    let ns = rng.wrapping_add(8000 + lag as u64 * 3);
                    ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                };
                let t1_noise_b = {
                    let ns = rng.wrapping_add(8001 + lag as u64 * 3);
                    ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                };
                let t1_sim =
                    (t1.similarity(&target_category) + t1_noise_a * enc_noise * 0.12) as f64;
                let t1_dsim =
                    (t1.similarity(&distractor_category) + t1_noise_b * enc_noise * 0.12) as f64;

                let t1_ev = t1_sim / temperature;
                let t1_dev = t1_dsim / temperature;
                let max_t1 = t1_ev.max(t1_dev);
                let p_t1 =
                    (t1_ev - max_t1).exp() / ((t1_ev - max_t1).exp() + (t1_dev - max_t1).exp());

                xor_shift(&mut rng);
                let r1 = (rng % 10000) as f64 / 10000.0;
                // Lapse model applied to T1 detection.
                let sub_trial = trial_idx * lags.len() * trials_per_lag
                    + lag_idx * trials_per_lag
                    + inner_trial;
                let t1_detected =
                    config.check_correct(r1 < p_t1, "attentional_blink", sub_trial * 2);

                // T1 RT: based on decision margin
                let t1_margin = (t1_sim - t1_dsim).abs();
                let t1_rt = 3.0 + (1.0 - t1_margin.min(1.0)) * 5.0;
                t1_rt_ticks.push(t1_rt);

                t1_total += 1;
                if t1_detected {
                    t1_correct_total += 1;
                }

                // T2 detection — attention depleted by T1 processing
                // Lapse_rate degrades attention recovery: higher lapse → slower recovery
                // from the attentional blink, producing larger blink magnitude.
                // This models attention-gate sluggishness (Di Lollo et al., 2005).
                let lapse_recovery_penalty = config.lapse_rate * 0.7;
                let effective_recovery = recovery_rate * (1.0 - lapse_recovery_penalty);
                let attention_available = if t1_detected {
                    if config.ssm_backend {
                        // SSM path: T1 depletes attention (negative pulse),
                        // then recovery emerges from state-space dynamics over lags
                        ssm.reset();
                        ssm.step(-1.0); // T1 depletion
                        // Step through intervening lags (lag-1 steps of 0-input recovery)
                        let mut ssm_out = 0.0_f32;
                        for _ in 0..(lag - 1) {
                            ssm_out = ssm.step(0.0);
                        }
                        // SSM output is negative (decaying from -1 input); map to [0, 1]
                        // attention = 1.0 + ssm_out  (ssm_out in ~[-1, 0] range)
                        let base = (1.0 + ssm_out as f64).clamp(0.0, attention_capacity);
                        // Lapse penalty slows SSM recovery at short lags
                        (base - lapse_recovery_penalty * 0.3).clamp(0.0, attention_capacity)
                    } else {
                        let depleted = attention_capacity - t1_cost;
                        let recovered = effective_recovery * (lag as f64 - 1.0);
                        (depleted + recovered).clamp(0.0, attention_capacity)
                    }
                } else {
                    attention_capacity // No T1 cost if T1 missed
                };

                // Generate T2 with attention-dependent encoding quality
                xor_shift(&mut rng);
                let t2_noise = ContinuousHV::random(dim, rng);
                // Lower attention → noisier encoding + distractor intrusion
                let signal_weight = 0.30 + 0.50 * attention_available;
                // At low attention, distractors "leak" into T2 representation
                let distractor_leak = (1.0 - attention_available) * 0.50;
                let noise_weight = (1.0 - signal_weight - distractor_leak).max(0.05);
                let t2 = ContinuousHV::weighted_bundle(
                    &[&target_category, &t2_noise, &distractor_category],
                    &[
                        signal_weight as f32,
                        noise_weight as f32,
                        distractor_leak as f32,
                    ],
                );

                let t2_noise_a = {
                    let ns = rng.wrapping_add(9000 + lag as u64 * 5);
                    ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                };
                let t2_noise_b = {
                    let ns = rng.wrapping_add(9001 + lag as u64 * 5);
                    ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                };
                // T2 noise scales with attention depletion: at short lags (low attention),
                // encoding noise has stronger impact (temporal masking amplifies noise)
                let t2_noise_coeff = 0.12 + (1.0 - attention_available as f32) * 0.18;
                let t2_sim = (t2.similarity(&target_category)
                    + t2_noise_a * enc_noise * t2_noise_coeff) as f64;
                let t2_dsim = (t2.similarity(&distractor_category)
                    + t2_noise_b * enc_noise * t2_noise_coeff) as f64;

                let t2_ev = t2_sim / temperature;
                let t2_dev = t2_dsim / temperature;
                let max_t2 = t2_ev.max(t2_dev);
                let p_t2 =
                    (t2_ev - max_t2).exp() / ((t2_ev - max_t2).exp() + (t2_dev - max_t2).exp());

                // T2 RT: harder when attention depleted (wider margin at high attention)
                let t2_margin = (t2_sim - t2_dsim).abs();
                let t2_rt =
                    3.0 + (1.0 - t2_margin.min(1.0)) * 5.0 + (1.0 - attention_available) * 3.0;
                t2_rt_ticks_by_lag[lag_idx].push(t2_rt);

                xor_shift(&mut rng);
                let r2 = (rng % 10000) as f64 / 10000.0;
                if config.check_correct(r2 < p_t2, "attentional_blink", sub_trial * 2 + 1) {
                    lag_correct[lag_idx] += 1;
                }
            }
        }

        let t1_acc = if t1_total > 0 {
            t1_correct_total as f64 / t1_total as f64
        } else {
            0.0
        };

        // lags = [2, 3, 5, 8], so lag3 is index 1, lag8 is index 3
        let lag3_acc = lag_correct[1] as f64 / trials_per_lag as f64;
        let lag8_acc = lag_correct[3] as f64 / trials_per_lag as f64;

        TrialResult {
            t1_accuracy: t1_acc,
            lag3_t2_accuracy: lag3_acc,
            lag8_t2_accuracy: lag8_acc,
            blink_magnitude: lag8_acc - lag3_acc,
            t1_rt_ticks,
            t2_rt_ticks_by_lag,
        }
    }
}

impl PsychBenchmark for AttentionalBlinkBenchmark {
    fn name(&self) -> &str {
        "Attention::AttentionalBlink"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Attentional Blink (RSVP)",
            citation: "Raymond et al. (1992)",
            year: 1992,
            doi: Some("10.1037/0096-1523.18.3.849"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut t1_accs = Vec::new();
        let mut lag3_accs = Vec::new();
        let mut lag8_accs = Vec::new();
        let mut blink_mags = Vec::new();
        let mut all_t1_rts = Vec::new();
        let mut all_lag3_rts = Vec::new();
        let mut all_lag8_rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            t1_accs.push(r.t1_accuracy);
            lag3_accs.push(r.lag3_t2_accuracy);
            lag8_accs.push(r.lag8_t2_accuracy);
            blink_mags.push(r.blink_magnitude);
            all_t1_rts.extend_from_slice(&r.t1_rt_ticks);
            all_lag3_rts.extend_from_slice(&r.t2_rt_ticks_by_lag[1]); // lag 3
            all_lag8_rts.extend_from_slice(&r.t2_rt_ticks_by_lag[3]); // lag 8
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "attentional_blink".to_string(),
                    correct: r.t1_accuracy > 0.5,
                    rt_ticks: if r.t1_rt_ticks.is_empty() {
                        0.0
                    } else {
                        r.t1_rt_ticks.iter().sum::<f64>() / r.t1_rt_ticks.len() as f64
                    },
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("t1_accuracy", MetricValue::from_samples(&t1_accs));
        result.insert("lag3_t2_accuracy", MetricValue::from_samples(&lag3_accs));
        result.insert("lag8_t2_accuracy", MetricValue::from_samples(&lag8_accs));
        result.insert("blink_magnitude", MetricValue::from_samples(&blink_mags));
        result.insert("t1::rt_ticks", MetricValue::from_samples(&all_t1_rts));
        result.insert("lag3::rt_ticks", MetricValue::from_samples(&all_lag3_rts));
        result.insert("lag8::rt_ticks", MetricValue::from_samples(&all_lag8_rts));

        result.conditions = 4; // 4 lag conditions
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
    fn test_attentional_blink_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = AttentionalBlinkBenchmark.run(&config);
        assert!(result.metrics.contains_key("t1_accuracy"));
        assert!(result.metrics.contains_key("lag3_t2_accuracy"));
        assert!(result.metrics.contains_key("lag8_t2_accuracy"));
        assert!(result.metrics.contains_key("blink_magnitude"));
    }

    #[test]
    fn test_attentional_blink_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = AttentionalBlinkBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_blink_direction() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = AttentionalBlinkBenchmark.run(&config);
        let lag3 = result.metrics["lag3_t2_accuracy"].mean;
        let lag8 = result.metrics["lag8_t2_accuracy"].mean;
        // Lag 8 should generally be better than lag 3 (blink recovery)
        assert!(
            lag8 >= lag3 - 0.10,
            "lag8 ({:.3}) should be >= lag3 ({:.3}) - 0.10",
            lag8,
            lag3
        );
    }

    #[test]
    fn test_attblink_ssm_matches_baseline() {
        let base_config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 20,
            ..Default::default()
        };
        let ssm_config = BenchmarkConfig {
            ssm_backend: true,
            ..base_config.clone()
        };
        let base_result = AttentionalBlinkBenchmark.run(&base_config);
        let ssm_result = AttentionalBlinkBenchmark.run(&ssm_config);
        // Compare lag-3 accuracy (blink window)
        let base_lag3 = base_result.metrics["lag3_t2_accuracy"].mean;
        let ssm_lag3 = ssm_result.metrics["lag3_t2_accuracy"].mean;
        let diff = (base_lag3 - ssm_lag3).abs();
        assert!(
            diff < 0.20,
            "SSM lag3 ({:.3}) too far from baseline ({:.3}), diff={:.3}",
            ssm_lag3,
            base_lag3,
            diff
        );
    }
}
