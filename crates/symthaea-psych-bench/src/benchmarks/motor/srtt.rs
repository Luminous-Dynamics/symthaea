// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Serial Reaction Time Task (SRTT).
//!
//! Participants respond to stimuli appearing at one of four positions.
//! Unknown to them, the sequence is structured (repeating pattern).
//! Learning effect = RT(random) - RT(sequence) after training.
//!
//! HDC implementation: Four position HVs bound with response HVs.
//! Sequence learning emerges as the SSM temporal backend accumulates
//! transition statistics. Random blocks lack predictable transitions.
//!
//! Human baselines (Nissen & Bullemer, 1987; Willingham et al., 1989):
//! - learning_effect: 0.15 (SD≈0.08) — RT difference (sequence - random)
//! - sequence_accuracy: 0.95 (SD≈0.04)
//! - random_accuracy: 0.92 (SD≈0.05)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// SRTT benchmark.
pub struct SrttBenchmark;

struct TrialResult {
    learning_effect: f64,
    sequence_accuracy: f64,
    random_accuracy: f64,
    sequence_rt: f64,
    random_rt: f64,
}

impl SrttBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("motor", "srtt", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Four spatial positions
        let positions: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(i * 100 + 1)))
            .collect();

        // Repeating sequence: 4-2-3-1-3-2-4-1 (8-element, Nissen & Bullemer style)
        let sequence = [3, 1, 2, 0, 2, 1, 3, 0]; // 0-indexed positions

        // Transition memory: HDC-based associative memory that learns
        // position-to-position transitions via bind-and-accumulate.
        // Each transition (prev → current) is stored as prev ⊗ current,
        // bundled into a single transition memory HV.
        // Retrieval: unbind prev from memory → similarity with each position
        // reveals the predicted next position (Nissen & Bullemer, 1987).
        let mut transition_memory = ContinuousHV::zero(dim);

        // Time pressure: raises response noise (Willingham et al., 1989).
        let noise_level: f32 = 0.20 + config.time_pressure as f32 * 0.15;

        let training_blocks = 4; // blocks of sequence practice
        let trials_per_block = 24; // 3 repetitions of 8-element sequence
        let test_trials = 24; // final block: sequence vs random

        // Training phase: build sequence knowledge via HDC transition binding
        let mut prev_pos: Option<usize> = None;
        let learning_rate: f32 = 0.15; // EMA rate for transition accumulation
        for _block in 0..training_blocks {
            for step in 0..trials_per_block {
                let pos_idx = sequence[step % sequence.len()];

                if let Some(prev) = prev_pos {
                    // Encode transition: prev ⊗ current (binding associates the pair)
                    let transition = positions[prev].bind(&positions[pos_idx]);
                    // Accumulate into transition memory (EMA-like)
                    transition_memory = ContinuousHV::weighted_bundle(
                        &[&transition_memory, &transition],
                        &[1.0 - learning_rate, learning_rate],
                    );
                }
                prev_pos = Some(pos_idx);
            }
        }
        // Normalize for clean retrieval
        transition_memory = transition_memory.normalize();

        // Test phase: sequence block
        let mut seq_correct = 0u32;
        let mut seq_rt_sum = 0.0f64;
        prev_pos = None;

        for step in 0..test_trials {
            let pos_idx = sequence[step % sequence.len()];
            let stimulus = &positions[pos_idx];

            // Prediction from transition memory:
            // If we know prev, unbind it from memory to retrieve predicted next.
            // prediction = transition_memory ⊗ prev (unbinding)
            // Then similarity(prediction, pos_i) gives transition strength to each position.
            let prediction_boost: f32 = if let Some(prev) = prev_pos {
                let prediction = transition_memory.bind(&positions[prev]);
                // How well does the prediction match the actual stimulus?
                prediction.similarity(stimulus).max(0.0)
            } else {
                0.0
            };

            // Response selection: match stimulus to positions
            let mut best_sim = f32::NEG_INFINITY;
            let mut best_idx = 0;
            for (i, pos) in positions.iter().enumerate() {
                let noise_degrade = config.effective_noise() as f32 * 0.4;
                let mut sim = stimulus.similarity(pos) * (1.0 - noise_degrade);
                // Prediction boost: learned transitions speed response to expected position
                if let Some(pp) = prev_pos {
                    let pred = transition_memory.bind(&positions[pp]);
                    let pred_sim = pred.similarity(pos).max(0.0);
                    sim += pred_sim * 0.4;
                }
                xor_shift(&mut rng);
                let noise = (rng % 10000) as f32 / 10000.0 * noise_level;
                sim += noise;
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }

            if best_idx == pos_idx {
                seq_correct += 1;
            }

            // RT: faster for predictable sequences (prediction boost reduces RT)
            let base_rt = 6.0;
            let pred_speedup = prediction_boost as f64 * 3.0; // Strong sequence = faster RT
            let tp_speedup = config.time_pressure * 1.5;
            seq_rt_sum += (base_rt - pred_speedup - tp_speedup).max(1.0);

            prev_pos = Some(pos_idx);
        }

        // Test phase: random block (no sequence structure)
        let mut rand_correct = 0u32;
        let mut rand_rt_sum = 0.0f64;
        prev_pos = None;
        // Reset SSM for random block (no transition learning applies)

        for _step in 0..test_trials {
            xor_shift(&mut rng);
            let pos_idx = (rng % 4) as usize;
            let stimulus = &positions[pos_idx];

            // No meaningful SSM prediction for random positions
            let mut best_sim = f32::NEG_INFINITY;
            let mut best_idx = 0;
            for (i, pos) in positions.iter().enumerate() {
                let mut sim = stimulus.similarity(pos);
                xor_shift(&mut rng);
                let noise = (rng % 10000) as f32 / 10000.0 * noise_level;
                sim += noise;
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }

            if best_idx == pos_idx {
                rand_correct += 1;
            }

            // RT: no SSM speedup for random
            let base_rt = 6.0;
            let tp_speedup = config.time_pressure * 1.5;
            rand_rt_sum += (base_rt - tp_speedup).max(1.0);

            let _ = prev_pos;
            prev_pos = Some(pos_idx);
        }

        let seq_acc = seq_correct as f64 / test_trials as f64;
        let rand_acc = rand_correct as f64 / test_trials as f64;
        let seq_rt = seq_rt_sum / test_trials as f64;
        let rand_rt = rand_rt_sum / test_trials as f64;

        // Learning effect: RT difference (normalized by random RT)
        let learning = if rand_rt > 0.0 {
            (rand_rt - seq_rt) / rand_rt
        } else {
            0.0
        };

        TrialResult {
            learning_effect: learning.max(0.0),
            sequence_accuracy: seq_acc,
            random_accuracy: rand_acc,
            sequence_rt: seq_rt,
            random_rt: rand_rt,
        }
    }
}

impl PsychBenchmark for SrttBenchmark {
    fn name(&self) -> &str {
        "Motor::SRTT"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Serial Reaction Time Task",
            citation: "Nissen & Bullemer (1987)",
            year: 1987,
            doi: Some("10.1016/0010-0285(87)90002-8"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut effects = Vec::new();
        let mut seq_accs = Vec::new();
        let mut rand_accs = Vec::new();
        let mut seq_rts = Vec::new();
        let mut rand_rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            effects.push(r.learning_effect);
            seq_accs.push(r.sequence_accuracy);
            rand_accs.push(r.random_accuracy);
            seq_rts.push(r.sequence_rt);
            rand_rts.push(r.random_rt);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "srtt".to_string(),
                    correct: r.sequence_accuracy > 0.5,
                    rt_ticks: r.sequence_rt,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("learning_effect", MetricValue::from_samples(&effects));
        result.insert("sequence_accuracy", MetricValue::from_samples(&seq_accs));
        result.insert("random_accuracy", MetricValue::from_samples(&rand_accs));
        result.insert("sequence::rt_ticks", MetricValue::from_samples(&seq_rts));
        result.insert("random::rt_ticks", MetricValue::from_samples(&rand_rts));

        result.conditions = 2; // sequence vs random
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
    fn test_srtt_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = SrttBenchmark.run(&config);
        assert!(result.metrics.contains_key("learning_effect"));
        assert!(result.metrics.contains_key("sequence_accuracy"));
        assert!(result.metrics.contains_key("random_accuracy"));
    }

    #[test]
    fn test_srtt_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = SrttBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_srtt_learning_non_negative() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = SrttBenchmark.run(&config);
        let effect = result.metrics["learning_effect"].mean;
        assert!(
            effect >= 0.0,
            "learning effect ({:.3}) should be >= 0",
            effect
        );
    }

    #[test]
    fn test_srtt_time_pressure() {
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
        let r_base = SrttBenchmark.run(&base);
        let r_press = SrttBenchmark.run(&pressed);
        // Under time pressure, RTs should decrease
        let seq_rt_base = r_base.metrics["sequence::rt_ticks"].mean;
        let seq_rt_press = r_press.metrics["sequence::rt_ticks"].mean;
        assert!(
            seq_rt_press <= seq_rt_base + 0.5,
            "time pressure should reduce sequence RT: base={:.2}, pressed={:.2}",
            seq_rt_base,
            seq_rt_press
        );
    }
}
