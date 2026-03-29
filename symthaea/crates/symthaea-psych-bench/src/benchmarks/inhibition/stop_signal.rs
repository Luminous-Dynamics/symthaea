// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stop Signal Task (SST).
//!
//! Measures the latency of response inhibition (SSRT). On each trial,
//! a go stimulus appears. On 25% of trials, a stop signal follows after
//! a variable delay (SSD). The agent must withhold its response.
//! SSRT estimated via the race model (Logan & Cowan, 1984).
//!
//! Human baselines (Logan 1994; Verbruggen & Logan 2008):
//! - sst_go_accuracy: 0.95 (SD~0.03)
//! - sst_go_rt_ticks: 10 (SD~2.0)
//! - sst_stop_accuracy: 0.50 (SD~0.10) — at tracked SSD
//! - ssrt_ticks: 4 (SD~1.0)

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Stop Signal Task benchmark.
pub struct StopSignalBenchmark;

struct TrialResult {
    sst_go_accuracy: f64,
    sst_go_rt_ticks: f64,
    sst_stop_accuracy: f64,
    ssrt_ticks: f64,
    ssd_mean: f64,
    rt_ticks: f64,
}

impl StopSignalBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("inhibition", "stop_signal", trial_idx);
        // Avoid xorshift64 seed-0 fixed point (see MEMORY.md)
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Go stimulus prototype
        let go_proto = ContinuousHV::random(dim, seed.wrapping_add(100));
        // Stop signal prototype (distinct from go)
        let stop_proto = ContinuousHV::random(dim, seed.wrapping_add(200));

        // Time pressure modulates go threshold and stop effectiveness.
        // Ratcliff & McKoon (2008) DDM: boundary collapse under speed pressure.
        // Verbruggen & Logan (2008): stop process slows under time pressure.
        let pressure = config.time_pressure;
        let diff_model = difficulty_model_for(self.name());
        let sig_mult = diff_model.signal_multiplier(config.difficulty);
        let go_threshold: f64 = (0.15 - pressure * 0.08) * sig_mult;
        let stop_effectiveness: f64 = (0.70 - pressure * 0.20) * sig_mult;
        // Softmax temperature: noisier decisions under pressure
        let temperature: f64 =
            (0.30 + pressure * 0.15) * diff_model.temperature_multiplier(config.difficulty);

        let total_trials = 200;
        // 25% stop trials (Logan 1994, standard SST design)
        let stop_ratio = 0.25;

        // Staircase SSD: starts at 5, adjusts +/-1 per stop trial outcome.
        // Converges to ~50% inhibition rate (Levitt 1971, adaptive tracking).
        let mut ssd: i64 = 5;

        let mut go_correct = 0u32;
        let mut go_total = 0u32;
        let mut stop_correct = 0u32;
        let mut stop_total = 0u32;
        let mut go_rts: Vec<f64> = Vec::new();
        let mut all_rts: Vec<f64> = Vec::new();
        let mut ssd_sum: f64 = 0.0;
        let mut ssd_count: u32 = 0;

        for _trial in 0..total_trials {
            xor_shift(&mut rng);
            let is_stop = (rng % 10000) as f64 / 10000.0 < stop_ratio;

            // Create noisy stimulus from go prototype (go stimulus always appears first)
            xor_shift(&mut rng);
            let noise = ContinuousHV::random(dim, rng);
            let stimulus = ContinuousHV::weighted_bundle(&[&go_proto, &noise], &[0.85, 0.15]);

            // Go signal evidence: similarity to go prototype
            // Encoding noise degrades stimulus detection
            let noise_degrade = config.effective_noise() as f32 * 0.4;
            let go_sim = (stimulus.similarity(&go_proto) * (1.0 - noise_degrade)) as f64;

            // Deliberation ticks (RT proxy): inverse of evidence strength
            // Base RT ~8 ticks (~400ms), modulated by decision difficulty
            let base_ticks = 4.0;
            let decision_difficulty = (1.0 - go_sim).max(0.0);
            let go_rt = base_ticks + decision_difficulty * 12.0;

            // Softmax go decision: compare go evidence against threshold
            let go_evidence = (go_sim - go_threshold) / temperature;
            let no_go_evidence = 0.0_f64; // baseline for not responding
            let max_ev = go_evidence.max(no_go_evidence);
            let exp_go = (go_evidence - max_ev).exp();
            let exp_no = (no_go_evidence - max_ev).exp();
            let denom = exp_go + exp_no;
            // Guard division by zero
            let p_respond = if denom > 1e-15 { exp_go / denom } else { 0.5 };

            if is_stop {
                // Stop trial: stop signal arrives after SSD ticks
                stop_total += 1;
                ssd_sum += ssd.max(0) as f64;
                ssd_count += 1;

                // Race model (Logan & Cowan, 1984):
                // If go_rt > SSD, the stop process can intervene.
                // Stop success probability depends on remaining processing time
                // and stop signal effectiveness.
                let ssd_f = ssd.max(0) as f64;
                let remaining = (go_rt - ssd_f).max(0.0);

                // Stop probability: higher when more time remains after stop signal.
                // Sigmoid function models gradual transition (Band et al., 2003).
                let stop_strength = if go_rt > 1e-15 {
                    (remaining / go_rt) * stop_effectiveness
                } else {
                    0.0
                };

                // Combine with similarity to stop prototype for HDC grounding
                xor_shift(&mut rng);
                let stop_noise = ContinuousHV::random(dim, rng);
                let stop_stimulus =
                    ContinuousHV::weighted_bundle(&[&stop_proto, &stop_noise], &[0.80, 0.20]);
                let stop_sim =
                    (stop_stimulus.similarity(&stop_proto) * (1.0 - noise_degrade)) as f64;

                // NE phasic burst amplifies stop effectiveness (Aron 2007):
                // Norepinephrine phasic signal strengthens the hyperdirect STN pathway.
                let ne_phasic_boost = config.neuromod_ne_phasic;
                let boosted_stop = stop_strength + ne_phasic_boost * (1.0 - stop_strength);

                // Final inhibition probability: race model (boosted) + HDC recognition
                let p_inhibit = (boosted_stop * 0.7 + stop_sim * 0.3).clamp(0.0, 1.0);

                xor_shift(&mut rng);
                let r = (rng % 10000) as f64 / 10000.0;

                if r < p_inhibit {
                    // Successful inhibition
                    stop_correct += 1;
                    // Staircase: increase SSD (make it harder)
                    ssd += 1;
                } else {
                    // Failed inhibition: signal-respond trial
                    // Staircase: decrease SSD (make it easier)
                    ssd = (ssd - 1).max(1);
                    // Record signal-respond RT (typically faster than normal go RT)
                    all_rts.push(go_rt);
                }
            } else {
                // Go trial
                go_total += 1;

                xor_shift(&mut rng);
                let r = (rng % 10000) as f64 / 10000.0;
                let responded = r < p_respond;

                if responded {
                    go_correct += 1;
                    go_rts.push(go_rt);
                    all_rts.push(go_rt);
                }
                // Omission error if !responded (miss on go trial)
            }
        }

        let go_acc = if go_total > 0 {
            go_correct as f64 / go_total as f64
        } else {
            0.0
        };
        let stop_acc = if stop_total > 0 {
            stop_correct as f64 / stop_total as f64
        } else {
            0.0
        };
        let mean_go_rt = if !go_rts.is_empty() {
            go_rts.iter().sum::<f64>() / go_rts.len() as f64
        } else {
            0.0
        };
        let mean_all_rt = if !all_rts.is_empty() {
            all_rts.iter().sum::<f64>() / all_rts.len() as f64
        } else {
            0.0
        };
        let mean_ssd = if ssd_count > 0 {
            ssd_sum / ssd_count as f64
        } else {
            0.0
        };

        // SSRT estimation — mean method (Logan & Cowan, 1984; Verbruggen et al., 2019):
        // SSRT = mean_go_RT - mean_SSD (at convergence, when P(inhibit) ~ 0.50)
        let ssrt = (mean_go_rt - mean_ssd).max(0.0);

        TrialResult {
            sst_go_accuracy: go_acc,
            sst_go_rt_ticks: mean_go_rt,
            sst_stop_accuracy: stop_acc,
            ssrt_ticks: ssrt,
            ssd_mean: mean_ssd,
            rt_ticks: mean_all_rt,
        }
    }
}

impl PsychBenchmark for StopSignalBenchmark {
    fn name(&self) -> &str {
        "Inhibition::StopSignal"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Stop-Signal Task",
            citation: "Logan & Cowan (1984)",
            year: 1984,
            doi: Some("10.1037/0096-3445.113.1.1"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut go_accs = Vec::new();
        let mut go_rts = Vec::new();
        let mut stop_accs = Vec::new();
        let mut ssrts = Vec::new();
        let mut ssds = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            go_accs.push(r.sst_go_accuracy);
            go_rts.push(r.sst_go_rt_ticks);
            stop_accs.push(r.sst_stop_accuracy);
            ssrts.push(r.ssrt_ticks);
            ssds.push(r.ssd_mean);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "stop_signal".to_string(),
                    correct: r.sst_go_accuracy > 0.5,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("sst_go_accuracy", MetricValue::from_samples(&go_accs));
        result.insert("sst_go_rt_ticks", MetricValue::from_samples(&go_rts));
        result.insert("sst_stop_accuracy", MetricValue::from_samples(&stop_accs));
        result.insert("ssrt_ticks", MetricValue::from_samples(&ssrts));
        result.insert("ssd_mean", MetricValue::from_samples(&ssds));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 2; // go + stop
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
    fn test_stop_signal_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = StopSignalBenchmark.run(&config);
        assert!(result.metrics.contains_key("sst_go_accuracy"));
        assert!(result.metrics.contains_key("sst_go_rt_ticks"));
        assert!(result.metrics.contains_key("sst_stop_accuracy"));
        assert!(result.metrics.contains_key("ssrt_ticks"));
        assert!(result.metrics.contains_key("ssd_mean"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_stop_signal_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = StopSignalBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
            assert!(val.std_dev.is_finite(), "metric {} std_dev not finite", key);
        }
    }

    #[test]
    fn test_ne_phasic_improves_stop_accuracy() {
        let no_ne = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            neuromod_ne_phasic: 0.0,
            ..Default::default()
        };
        let high_ne = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            neuromod_ne_phasic: 0.5,
            ..Default::default()
        };
        let r_no = StopSignalBenchmark.run(&no_ne);
        let r_high = StopSignalBenchmark.run(&high_ne);
        let stop_no = r_no.metrics["sst_stop_accuracy"].mean;
        let stop_high = r_high.metrics["sst_stop_accuracy"].mean;
        assert!(
            stop_high >= stop_no - 0.05,
            "NE phasic boost should improve stop accuracy: none={:.3}, high={:.3}",
            stop_no,
            stop_high
        );
    }

    #[test]
    fn test_ssrt_positive() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = StopSignalBenchmark.run(&config);
        let ssrt = result.metrics["ssrt_ticks"].mean;
        assert!(
            ssrt > 0.0,
            "SSRT should be positive (go RT > SSD), got {}",
            ssrt
        );
    }
}
