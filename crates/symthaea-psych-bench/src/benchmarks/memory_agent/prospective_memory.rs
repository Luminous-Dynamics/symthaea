// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prospective Memory Task.
//!
//! Tests the ability to remember to perform an action when a cue appears
//! during an ongoing task. Event-based PM: detect a target word while
//! doing category classification.
//!
//! HDC implementation: PM intention stored in WM at start, occupies 1 slot.
//! PM cue detection requires matching against intention HV while performing
//! ongoing classification. Cost = reduced ongoing capacity from monitoring.
//!
//! Human baselines (Einstein & McDaniel 2005; Kliegel et al. 2008):
//! - pm_hit_rate: 0.75 (SD≈0.15)
//! - pm_ongoing_accuracy: 0.90 (SD≈0.06)
//! - pm_cost: 0.05 (SD≈0.03)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::ssm_temporal::SsmTemporalBackend;
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Prospective Memory benchmark.
pub struct ProspectiveMemoryBenchmark;

struct TrialResult {
    pm_hit_rate: f64,
    ongoing_accuracy: f64,
    pm_cost: f64,
    rt_ticks: f64,
}

impl ProspectiveMemoryBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("memory", "prospective", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Two categories for ongoing task
        let category_a = ContinuousHV::random(dim, seed.wrapping_add(100));
        let category_b = ContinuousHV::random(dim, seed.wrapping_add(200));

        // PM target — a special stimulus to watch for
        let pm_target = ContinuousHV::random(dim, seed.wrapping_add(300));

        // PM intention stored in WM (reduces effective capacity)
        let pm_intention = ContinuousHV::weighted_bundle(
            &[
                &pm_target,
                &ContinuousHV::random(dim, seed.wrapping_add(400)),
            ],
            &[0.65, 0.35],
        );

        let temperature: f64 = 0.30;
        let total_trials = 100;
        let pm_cue_rate = 0.20; // 20% of trials contain PM cue

        // PM intention decay: activation decreases over time.
        // Time pressure: base decay 0.015 models ~70% PM hit rate (Einstein & McDaniel, 2005);
        // +0.010/unit accelerates decay, reflecting reduced rehearsal under deadline (Wickelgren, 1977).
        let pm_decay_rate: f64 = 0.015 + config.time_pressure * 0.010;

        // WM capacity cost from PM monitoring.
        // Time pressure: base noise 0.35 models ongoing-task interference; +0.10/unit raises
        // monitoring noise, reflecting divided attention under speed emphasis (Marsh et al., 2003).
        let monitoring_noise: f32 = 0.35 + config.time_pressure as f32 * 0.10;

        let mut pm_hits = 0u32;
        let mut pm_total = 0u32;
        let mut ongoing_correct = 0u32;
        let mut ongoing_total = 0u32;

        // Baseline accuracy (without PM monitoring) — run short control
        let mut baseline_correct = 0u32;
        let baseline_trials = 30;
        for _trial in 0..baseline_trials {
            xor_shift(&mut rng);
            let is_cat_a = rng % 2 == 0;
            let proto = if is_cat_a { &category_a } else { &category_b };

            xor_shift(&mut rng);
            let noise = ContinuousHV::random(dim, rng);
            let stimulus = ContinuousHV::weighted_bundle(&[proto, &noise], &[0.75, 0.25]);

            let sim_a = stimulus.similarity(&category_a) as f64;
            let sim_b = stimulus.similarity(&category_b) as f64;

            let ev_a = sim_a / temperature;
            let ev_b = sim_b / temperature;
            let max_ev = ev_a.max(ev_b);
            let p_a = (ev_a - max_ev).exp() / ((ev_a - max_ev).exp() + (ev_b - max_ev).exp());

            xor_shift(&mut rng);
            let r = (rng % 10000) as f64 / 10000.0;
            let chose_a = r < p_a;

            if (is_cat_a && chose_a) || (!is_cat_a && !chose_a) {
                baseline_correct += 1;
            }
        }
        let baseline_acc = baseline_correct as f64 / baseline_trials as f64;

        // Main task with PM monitoring
        let mut pm_activation: f64 = 1.0;
        // SSM temporal backend (A=-0.03 for slow PM decay)
        let mut ssm = SsmTemporalBackend::new(-0.03, 4);
        // Initialize SSM with full activation
        ssm.step(1.0);

        for _trial in 0..total_trials {
            xor_shift(&mut rng);
            let is_pm_cue = (rng % 100) as f64 / 100.0 < pm_cue_rate;

            // PM intention decays over time
            if config.ssm_backend {
                pm_activation = ssm.step(0.0) as f64;
            } else {
                pm_activation = (pm_activation - pm_decay_rate).max(0.0);
            }

            if is_pm_cue {
                pm_total += 1;
                // PM stimulus looks like the target
                xor_shift(&mut rng);
                let noise = ContinuousHV::random(dim, rng);
                // PM cue mixed with category noise (distractor-like context)
                xor_shift(&mut rng);
                let context_noise = ContinuousHV::random(dim, rng);
                let stimulus = ContinuousHV::weighted_bundle(
                    &[&pm_target, &noise, &context_noise],
                    &[0.35, 0.35, 0.30],
                );

                // Check if PM intention is detected
                let pm_sim = stimulus.similarity(&pm_intention) as f64;
                let cue_threshold = 0.27 + (1.0 - pm_activation) * 0.25;

                if pm_sim > cue_threshold {
                    pm_hits += 1;
                    // Refresh PM intention on successful detection
                    if config.ssm_backend {
                        ssm.step(1.0);
                    }
                }
            } else {
                ongoing_total += 1;
                // Regular ongoing classification
                xor_shift(&mut rng);
                let is_cat_a = rng % 2 == 0;
                let proto = if is_cat_a { &category_a } else { &category_b };

                xor_shift(&mut rng);
                let noise = ContinuousHV::random(dim, rng);
                // Add monitoring noise (PM cost)
                xor_shift(&mut rng);
                let monitor_noise = ContinuousHV::random(dim, rng);
                let stimulus = ContinuousHV::weighted_bundle(
                    &[proto, &noise, &monitor_noise],
                    &[0.75 - monitoring_noise, 0.25, monitoring_noise],
                );

                let sim_a = stimulus.similarity(&category_a) as f64;
                let sim_b = stimulus.similarity(&category_b) as f64;

                let ev_a = sim_a / temperature;
                let ev_b = sim_b / temperature;
                let max_ev = ev_a.max(ev_b);
                let p_a = (ev_a - max_ev).exp() / ((ev_a - max_ev).exp() + (ev_b - max_ev).exp());

                xor_shift(&mut rng);
                let r = (rng % 10000) as f64 / 10000.0;
                let chose_a = r < p_a;

                if (is_cat_a && chose_a) || (!is_cat_a && !chose_a) {
                    ongoing_correct += 1;
                }
            }
        }

        let pm_hr = if pm_total > 0 {
            pm_hits as f64 / pm_total as f64
        } else {
            0.0
        };
        let ongoing_acc = if ongoing_total > 0 {
            ongoing_correct as f64 / ongoing_total as f64
        } else {
            0.0
        };
        let cost = (baseline_acc - ongoing_acc).max(0.0);

        // RT proxy: PM monitoring adds overhead to every trial.
        // Base 5 ticks (ongoing classification), +2 ticks for PM monitoring
        // cost (dual-task), +scaling by intention decay (weaker intention =
        // slower detection; Einstein & McDaniel, 2005).
        let monitoring_cost = 2.0;
        let decay_penalty = (1.0 - pm_activation) * 3.0;
        let rt_ticks = 5.0 + monitoring_cost + decay_penalty;

        TrialResult {
            pm_hit_rate: pm_hr,
            ongoing_accuracy: ongoing_acc,
            pm_cost: cost,
            rt_ticks,
        }
    }
}

impl PsychBenchmark for ProspectiveMemoryBenchmark {
    fn name(&self) -> &str {
        "MemoryAgent::ProspectiveMemory"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Prospective Memory",
            citation: "Einstein & McDaniel (1990)",
            year: 1990,
            doi: Some("10.1037/0278-7393.16.4.717"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut hit_rates = Vec::new();
        let mut ongoing_accs = Vec::new();
        let mut costs = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            hit_rates.push(r.pm_hit_rate);
            ongoing_accs.push(r.ongoing_accuracy);
            costs.push(r.pm_cost);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "prospective".to_string(),
                    correct: r.pm_hit_rate > 0.5,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("pm_hit_rate", MetricValue::from_samples(&hit_rates));
        result.insert(
            "pm_ongoing_accuracy",
            MetricValue::from_samples(&ongoing_accs),
        );
        result.insert("pm_cost", MetricValue::from_samples(&costs));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 2; // PM cue vs ongoing
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
    fn test_prospective_memory_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = ProspectiveMemoryBenchmark.run(&config);
        assert!(result.metrics.contains_key("pm_hit_rate"));
        assert!(result.metrics.contains_key("pm_ongoing_accuracy"));
        assert!(result.metrics.contains_key("pm_cost"));
    }

    #[test]
    fn test_prospective_memory_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ProspectiveMemoryBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_pm_cost_non_negative() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = ProspectiveMemoryBenchmark.run(&config);
        let cost = result.metrics["pm_cost"].mean;
        assert!(cost >= 0.0, "PM cost ({:.3}) should be >= 0", cost);
    }

    #[test]
    fn test_prospective_ssm_matches_baseline() {
        let base_config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 20,
            ..Default::default()
        };
        let ssm_config = BenchmarkConfig {
            ssm_backend: true,
            ..base_config.clone()
        };
        let base_result = ProspectiveMemoryBenchmark.run(&base_config);
        let ssm_result = ProspectiveMemoryBenchmark.run(&ssm_config);
        let base_hr = base_result.metrics["pm_hit_rate"].mean;
        let ssm_hr = ssm_result.metrics["pm_hit_rate"].mean;
        let diff = (base_hr - ssm_hr).abs();
        // SSM should produce similar results (within 0.20 absolute)
        assert!(
            diff < 0.20,
            "SSM pm_hit_rate ({:.3}) too far from baseline ({:.3}), diff={:.3}",
            ssm_hr,
            base_hr,
            diff
        );
    }

    #[test]
    fn test_ssm_backend_flag_default_off() {
        let config = BenchmarkConfig::default();
        assert!(!config.ssm_backend, "ssm_backend should default to false");
    }
}
