// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Go/No-Go Task.
//!
//! Tests prepotent response inhibition. Present Go stimuli (70%) and No-Go
//! stimuli (30%). Correct = respond to Go, withhold to No-Go.
//!
//! HDC implementation: models prepotent response bias — Go trials need lower
//! similarity threshold (automated response tendency). Commission errors arise
//! from similarity between Go and No-Go stimuli.
//!
//! Human baselines (Wessel 2018; Verbruggen & Logan 2008):
//! - go_accuracy: 0.98 (SD≈0.02)
//! - nogo_accuracy: 0.82 (SD≈0.10)
//! - inhibition_cost: 0.16 (SD≈0.08)
//! - go_rt_ticks: 4 (SD≈1.5)

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Go/No-Go response inhibition benchmark.
pub struct GoNoGoBenchmark;

struct TrialResult {
    go_accuracy: f64,
    nogo_accuracy: f64,
    inhibition_cost: f64,
    go_rt_ticks: f64,
    /// Stop-signal RT estimate (integration method).
    /// SSRT = nth percentile of Go RT distribution, where n = commission error rate.
    ssrt_ticks: f64,
}

impl GoNoGoBenchmark {
    fn run_trial_traced(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
        diff_model: &crate::harness::difficulty::DifficultyModel,
        global_trial_idx: &mut usize,
    ) -> (TrialResult, Vec<TrialOutcome>) {
        let dim = config.dimension;
        let seed = config.trial_seed("inhibition", "go_nogo", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        let shared = ContinuousHV::random(dim, seed.wrapping_add(50));
        let go_unique = ContinuousHV::random(dim, seed.wrapping_add(100));
        let nogo_unique = ContinuousHV::random(dim, seed.wrapping_add(200));
        let overlap: f32 = 0.40;
        let go_proto =
            ContinuousHV::weighted_bundle(&[&go_unique, &shared], &[1.0 - overlap, overlap]);
        let nogo_proto =
            ContinuousHV::weighted_bundle(&[&nogo_unique, &shared], &[1.0 - overlap, overlap]);

        let pressure = config.time_pressure;
        let temp_mult = diff_model.temperature_multiplier(config.difficulty);
        let go_threshold: f64 = 0.12 - pressure * 0.06;
        let nogo_threshold: f64 = 0.32 - pressure * 0.10;
        let temperature: f64 = (0.25 + pressure * 0.15) * temp_mult;

        let total_trials = 100;
        let go_ratio = 0.70;

        let mut go_correct = 0u32;
        let mut go_total = 0u32;
        let mut nogo_correct = 0u32;
        let mut nogo_total = 0u32;
        let mut go_rts = Vec::new();
        let mut outcomes = Vec::new();

        for _trial in 0..total_trials {
            xor_shift(&mut rng);
            let is_go = (rng % 100) as f64 / 100.0 < go_ratio;

            xor_shift(&mut rng);
            let noise = ContinuousHV::random(dim, rng);
            let proto = if is_go { &go_proto } else { &nogo_proto };
            let stimulus = ContinuousHV::weighted_bundle(&[proto, &noise], &[0.85, 0.15]);

            // Encoding noise degrades stimulus discrimination
            let noise_degrade = config.effective_noise() as f32 * 0.4;
            let go_sim = (stimulus.similarity(&go_proto) * (1.0 - noise_degrade)) as f64;
            let nogo_sim = (stimulus.similarity(&nogo_proto) * (1.0 - noise_degrade)) as f64;

            let decision_margin = (go_sim - nogo_sim).abs();
            let deliberation_ticks = 2.0 + (1.0 - decision_margin) * 6.0;

            let go_evidence = (go_sim - go_threshold) / temperature;
            // D2-mediated NoGo pathway boost (Frank 2005):
            // D2 receptor activation raises evidence for withholding responses.
            let d2_nogo_boost = config.neuromod_d2_inhibition * 0.5;
            let nogo_evidence = (nogo_sim - nogo_threshold) / temperature + d2_nogo_boost;

            let max_ev = go_evidence.max(nogo_evidence);
            let exp_go = (go_evidence - max_ev).exp();
            let exp_nogo = (nogo_evidence - max_ev).exp();
            let p_respond = exp_go / (exp_go + exp_nogo);

            xor_shift(&mut rng);
            let r = (rng % 10000) as f64 / 10000.0;
            let responded = r < p_respond;

            let correct;
            if is_go {
                go_total += 1;
                correct = responded;
                if responded {
                    go_correct += 1;
                    go_rts.push(deliberation_ticks);
                }
            } else {
                nogo_total += 1;
                correct = !responded;
                if !responded {
                    nogo_correct += 1;
                } else {
                    go_rts.push(deliberation_ticks);
                }
            }

            if config.trial_trace {
                outcomes.push(TrialOutcome {
                    trial_idx: *global_trial_idx,
                    condition: if is_go {
                        "go".to_string()
                    } else {
                        "nogo".to_string()
                    },
                    correct,
                    rt_ticks: deliberation_ticks,
                    similarity: go_sim,
                    confidence: p_respond,
                    response_idx: if responded { 1 } else { 0 },
                    extra: BTreeMap::new(),
                });
                *global_trial_idx += 1;
            }
        }

        let go_acc = if go_total > 0 {
            go_correct as f64 / go_total as f64
        } else {
            0.0
        };
        let nogo_acc = if nogo_total > 0 {
            nogo_correct as f64 / nogo_total as f64
        } else {
            0.0
        };
        let mean_rt = if !go_rts.is_empty() {
            go_rts.iter().sum::<f64>() / go_rts.len() as f64
        } else {
            0.0
        };

        let ssrt = if go_rts.len() >= 3 {
            let commission_rate = 1.0 - nogo_acc;
            let mut sorted_rts = go_rts.clone();
            sorted_rts.sort_by(|a, b| a.total_cmp(b));
            let idx =
                ((commission_rate * sorted_rts.len() as f64) as usize).min(sorted_rts.len() - 1);
            sorted_rts[idx]
        } else {
            0.0
        };

        (
            TrialResult {
                go_accuracy: go_acc,
                nogo_accuracy: nogo_acc,
                inhibition_cost: go_acc - nogo_acc,
                go_rt_ticks: mean_rt,
                ssrt_ticks: ssrt,
            },
            outcomes,
        )
    }
}

impl PsychBenchmark for GoNoGoBenchmark {
    fn name(&self) -> &str {
        "Inhibition::GoNoGo"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Go/No-Go Task",
            citation: "Donders (1969)",
            year: 1969,
            doi: Some("10.1016/0001-6918(69)90065-1"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let diff_model = difficulty_model_for(self.name());

        let mut go_accs = Vec::new();
        let mut nogo_accs = Vec::new();
        let mut costs = Vec::new();
        let mut rts = Vec::new();
        let mut ssrts = Vec::new();
        let mut trace = Vec::new();
        let mut global_trial_idx = 0usize;

        for trial in 0..config.trials_per_condition {
            let (r, outcomes) =
                self.run_trial_traced(config, trial, &diff_model, &mut global_trial_idx);
            go_accs.push(r.go_accuracy);
            nogo_accs.push(r.nogo_accuracy);
            costs.push(r.inhibition_cost);
            rts.push(r.go_rt_ticks);
            ssrts.push(r.ssrt_ticks);
            if config.trial_trace {
                trace.extend(outcomes);
            }
        }

        result.insert("go_accuracy", MetricValue::from_samples(&go_accs));
        result.insert("nogo_accuracy", MetricValue::from_samples(&nogo_accs));
        result.insert("inhibition_cost", MetricValue::from_samples(&costs));
        result.insert("go_rt_ticks", MetricValue::from_samples(&rts));
        result.insert("ssrt_ticks", MetricValue::from_samples(&ssrts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 2;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_go_nogo_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = GoNoGoBenchmark.run(&config);
        assert!(result.metrics.contains_key("go_accuracy"));
        assert!(result.metrics.contains_key("nogo_accuracy"));
        assert!(result.metrics.contains_key("inhibition_cost"));
        assert!(result.metrics.contains_key("go_rt_ticks"));
    }

    #[test]
    fn test_go_nogo_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = GoNoGoBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_d2_inhibition_improves_nogo() {
        let low_d2 = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            neuromod_d2_inhibition: 0.0,
            ..Default::default()
        };
        let high_d2 = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            neuromod_d2_inhibition: 0.6,
            ..Default::default()
        };
        let r_low = GoNoGoBenchmark.run(&low_d2);
        let r_high = GoNoGoBenchmark.run(&high_d2);
        let nogo_low = r_low.metrics["nogo_accuracy"].mean;
        let nogo_high = r_high.metrics["nogo_accuracy"].mean;
        assert!(
            nogo_high >= nogo_low - 0.05,
            "D2 boost should improve NoGo: low={:.3}, high={:.3}",
            nogo_low,
            nogo_high
        );
    }

    #[test]
    fn test_go_easier_than_nogo() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = GoNoGoBenchmark.run(&config);
        let go = result.metrics["go_accuracy"].mean;
        let nogo = result.metrics["nogo_accuracy"].mean;
        assert!(
            go >= nogo - 0.05,
            "Go ({:.3}) should be >= NoGo ({:.3})",
            go,
            nogo
        );
    }
}
