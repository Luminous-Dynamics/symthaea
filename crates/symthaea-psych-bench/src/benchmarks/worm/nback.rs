// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! N-back working memory task.
//!
//! The system sees a stream of items and must identify when the current item
//! matches the item N positions back. Tests the updating component of WM.

use crate::adapter::StimulusAdapter;
use crate::adapter::sequence::{SequenceAdapter, SequenceItem};
use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::{WmConfig, WorkingMemory};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// N-back benchmark testing the updating function of working memory.
pub struct NBackBenchmark;

impl NBackBenchmark {
    /// Run a single N-back trial with difficulty scaling and optional trace collection.
    fn run_trial_traced(
        &self,
        n: usize,
        sequence_len: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
        temp_mult: f64,
        global_trial_idx: &mut usize,
    ) -> (f64, f64, Vec<f64>, Vec<TrialOutcome>) {
        let dim = config.dimension;
        let seed = config.trial_seed("worm", &format!("nback_{}", n), trial_idx);
        let adapter = SequenceAdapter;

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        let mut rng_state = seed;
        let vocab_size = 8u64;
        let mut sequence = Vec::with_capacity(sequence_len);
        for i in 0..sequence_len {
            let is_target = i >= n && {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                (rng_state % 100) < 30
            };
            if is_target {
                sequence.push(sequence[i - n]);
            } else {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                sequence.push(SequenceItem(rng_state % vocab_size));
            }
        }

        let mut hits = 0u64;
        let mut misses = 0u64;
        let mut false_alarms = 0u64;
        let mut correct_rejections = 0u64;
        let mut rt_ticks = Vec::new();
        let mut outcomes = Vec::new();

        let mut perceived_history: Vec<ContinuousHV> = Vec::with_capacity(sequence_len);

        // Difficulty: scale match threshold via temperature multiplier.
        // Time pressure widens the acceptance window (lower threshold → more false alarms).
        // Encoding noise degrades similarity signal, further reducing discriminability.
        let noise = config.effective_noise();
        let base_threshold = 0.50 - config.time_pressure * 0.25 - noise * 0.15;
        let match_threshold = (base_threshold / temp_mult).max(0.05) as f32;

        for (i, &item) in sequence.iter().enumerate() {
            let hv = adapter.encode(&item, dim);
            wm.perceive(hv.clone());
            wm.tick();
            perceived_history.push(hv.clone());

            if i >= n {
                let is_target = sequence[i] == sequence[i - n];
                let nback_retained = n < config.working_memory_capacity;
                let nback_hv = &perceived_history[i - n];
                // Encoding noise degrades similarity signal (top-down refinement absent)
                let match_sim = hv.similarity(nback_hv) * (1.0 - noise as f32 * 0.4);

                let mut lure_match = false;
                if !is_target && nback_retained {
                    for offset in [1i32, -1] {
                        let lure_pos = i as i32 - n as i32 + offset;
                        if lure_pos >= 0 && (lure_pos as usize) < i {
                            let lure_sim = hv.similarity(&perceived_history[lure_pos as usize]);
                            if lure_sim > match_threshold {
                                lure_match = true;
                                break;
                            }
                        }
                    }
                }

                let responded_match = nback_retained && (match_sim > match_threshold || lure_match);
                let margin = (match_sim - match_threshold).abs() as f64;
                // Time pressure reduces deliberation ticks
                let rt_scale = 1.0 - config.time_pressure * 0.4;
                let ticks = (3.0 + (1.0 - margin.min(1.0)) * 5.0) * rt_scale;
                rt_ticks.push(ticks);

                let correct = match (is_target, responded_match) {
                    (true, true) => {
                        hits += 1;
                        true
                    }
                    (true, false) => {
                        misses += 1;
                        false
                    }
                    (false, true) => {
                        false_alarms += 1;
                        false
                    }
                    (false, false) => {
                        correct_rejections += 1;
                        true
                    }
                };

                if config.trial_trace {
                    outcomes.push(TrialOutcome {
                        trial_idx: *global_trial_idx,
                        condition: format!("nback_{}", n),
                        correct,
                        rt_ticks: ticks,
                        similarity: match_sim as f64,
                        confidence: margin.min(1.0),
                        response_idx: if responded_match { 1 } else { 0 },
                        extra: BTreeMap::new(),
                    });
                    *global_trial_idx += 1;
                }
            }
        }

        let hit_rate = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };
        let fa_rate = if false_alarms + correct_rejections > 0 {
            false_alarms as f64 / (false_alarms + correct_rejections) as f64
        } else {
            0.0
        };

        (hit_rate, fa_rate, rt_ticks, outcomes)
    }
}

impl PsychBenchmark for NBackBenchmark {
    fn name(&self) -> &str {
        "WorM::N-back"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "N-back Working Memory",
            citation: "Kirchner (1958)",
            year: 1958,
            doi: Some("10.1037/h0043688"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let sequence_len = 30;
        let diff_model = difficulty_model_for(self.name());
        let temp_mult = diff_model.temperature_multiplier(config.difficulty);
        let mut trace = Vec::new();
        let mut global_trial_idx = 0usize;

        for n in [1, 2, 3] {
            let mut hit_rates = Vec::new();
            let mut fa_rates = Vec::new();
            let mut accuracies = Vec::new();
            let mut all_rts = Vec::new();

            for trial in 0..config.trials_per_condition {
                let (hr, fa, rts, trial_outcomes) = self.run_trial_traced(
                    n,
                    sequence_len,
                    config,
                    trial,
                    temp_mult,
                    &mut global_trial_idx,
                );
                hit_rates.push(hr);
                fa_rates.push(fa);
                accuracies.push(hr - fa);
                all_rts.extend_from_slice(&rts);
                if config.trial_trace {
                    trace.extend(trial_outcomes);
                }
            }

            result.insert(
                format!("nback_{}::hit_rate", n),
                MetricValue::from_samples(&hit_rates),
            );
            result.insert(
                format!("nback_{}::false_alarm_rate", n),
                MetricValue::from_samples(&fa_rates),
            );
            result.insert(
                format!("nback_{}::accuracy", n),
                MetricValue::from_samples(&accuracies),
            );
            result.insert(
                format!("nback_{}::rt_ticks", n),
                MetricValue::from_samples(&all_rts),
            );
        }

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 3;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nback_runs_without_panic() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let bench = NBackBenchmark;
        let result = bench.run(&config);
        assert_eq!(result.conditions, 3);
        assert!(result.metrics.contains_key("nback_1::hit_rate"));
        assert!(result.metrics.contains_key("nback_2::hit_rate"));
        assert!(result.metrics.contains_key("nback_3::hit_rate"));
    }

    #[test]
    fn test_nback_metrics_finite() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            dimension: 256,
            ..Default::default()
        };
        let result = NBackBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
            assert!(val.std_dev.is_finite(), "metric {} std_dev not finite", key);
        }
    }
}
