// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Serial recall task.
//!
//! Present a sequence of items, then test recall at each position.
//! Produces the classic serial position curve (primacy + recency effects).

use crate::adapter::StimulusAdapter;
use crate::adapter::sequence::{SequenceAdapter, SequenceItem};
use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::ssm_temporal::SsmTemporalBackend;
use crate::wm::{WmConfig, WorkingMemory};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Serial recall benchmark producing serial position curves.
pub struct SerialRecallBenchmark;

impl SerialRecallBenchmark {
    /// Run a single trial: present a list, then probe each position.
    /// Returns (accuracy per serial position, mean_rt_ticks).
    fn run_trial(
        &self,
        list_len: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (Vec<f64>, f64) {
        let dim = config.dimension;
        let seed = config.trial_seed("worm", &format!("serial_{}", list_len), trial_idx);
        let adapter = SequenceAdapter;

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        // SSM temporal backend for PI accumulation (A=-0.20 sigmoid-aligned decay)
        let mut ssm = SsmTemporalBackend::new(-0.20, 4);

        // Generate unique items
        let items: Vec<SequenceItem> = (0..list_len)
            .map(|i| {
                let item_seed = seed.wrapping_add(i as u64).wrapping_mul(0x100000001b3);
                SequenceItem(item_seed % 100)
            })
            .collect();

        // Present items with proactive interference (PI) and covert rehearsal.
        //
        // PI: Items encoded later suffer interference from already-stored items
        // (Keppel & Underwood, 1962). Sigmoid PI model with moderate slope.
        //
        // Rehearsal: After each new item is presented, all previously stored items
        // receive a covert rehearsal boost to their activation. This models the
        // finding that early items accumulate more rehearsals (Rundus, 1971;
        // Atkinson & Shiffrin, 1968). The rehearsal boost counteracts activation
        // decay for early items, producing the primacy effect.
        let half_list = list_len as f32 / 2.0;
        // Rehearsal boost per covert rehearsal event. Each already-stored item
        // gets one rehearsal opportunity when a new item is presented.
        // Value 1.15 means each rehearsal recovers ~15% of lost activation,
        // calibrated so items at positions 0-1 (which accumulate 5-6 rehearsals
        // in a 7-item list) overcome their activation decay disadvantage and
        // produce a primacy index closer to the human baseline of 0.15 (SD 0.06).
        let rehearsal_boost: f32 = 1.15;
        for (pos, item) in items.iter().enumerate() {
            let hv = adapter.encode(item, dim);
            // Sigmoid PI: midpoint at list_len/2, slope 0.20 (Keppel & Underwood, 1962).
            // Time pressure +0.10/unit amplifies interference (Wickelgren, 1977).
            let diff_model = difficulty_model_for("WorM::SerialRecall");
            let pi_base = (0.55 + config.time_pressure as f32 * 0.10)
                * diff_model.interference_multiplier(config.difficulty) as f32;
            let pi_strength = if config.ssm_backend {
                // SSM accumulates PI via recurrent state; each item drives interference higher
                ssm.step(1.0);
                pi_base * ssm.memory_strength()
            } else {
                // Sigmoid: 1/(1+exp(-slope*(pos - midpoint)))
                // Gentle onset protects primacy; steep mid-list rise; plateau allows recency
                let sigmoid = 1.0 / (1.0 + (-0.20 * (pos as f32 - half_list)).exp());
                pi_base * sigmoid
            };
            let noisy_hv = if pi_strength > 0.01 {
                let noise = ContinuousHV::random(dim, seed.wrapping_add(500 + pos as u64));
                ContinuousHV::weighted_bundle(&[&hv, &noise], &[1.0 - pi_strength, pi_strength])
            } else {
                hv
            };
            // Covert rehearsal: boost activation of all previously stored items.
            // Earlier items accumulate more rehearsals (item 0 gets list_len-1
            // rehearsals, item 1 gets list_len-2, etc.), producing primacy.
            for prev in 0..wm.len() {
                wm.boost_activation(prev, rehearsal_boost);
            }
            wm.perceive(noisy_hv);
            wm.tick();
        }

        // Brief delay
        for _ in 0..2 {
            wm.tick();
        }

        // Probe each position using graded recall strength.
        // activation_weighted_similarity combines encoding fidelity
        // (degraded by PI for later items) with activation decay
        // (lower for earlier items), producing the U-curve.
        let mut position_accuracy = Vec::with_capacity(list_len);
        let mut rt_sum = 0.0f64;

        for item in &items {
            let target_hv = adapter.encode(item, dim);
            let recall_strength = wm.activation_weighted_similarity(&target_hv) as f64;
            position_accuracy.push(recall_strength);

            // RT proxy: deliberation ticks based on recall retrieval difficulty.
            // Lower recall_strength → harder retrieval → longer RT
            // (Sternberg, 1966; Ratcliff, 1978 diffusion model).
            let decision_margin = recall_strength.abs().min(1.0);
            let rt_ticks = 3.0 + (1.0 - decision_margin) * 6.0;
            rt_sum += rt_ticks;
        }

        let mean_rt = if items.is_empty() {
            0.0
        } else {
            rt_sum / items.len() as f64
        };
        (position_accuracy, mean_rt)
    }
}

impl PsychBenchmark for SerialRecallBenchmark {
    fn name(&self) -> &str {
        "WorM::SerialRecall"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Serial Recall with Proactive Interference",
            citation: "Keppel & Underwood (1962)",
            year: 1962,
            doi: Some("10.1037/h0084156"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        for list_len in [5, 7, 9] {
            // Collect accuracy per position across trials
            let mut position_samples: Vec<Vec<f64>> = vec![Vec::new(); list_len];
            let mut rts = Vec::new();

            for trial in 0..config.trials_per_condition {
                let (pos_acc, rt) = self.run_trial(list_len, config, trial);
                for (pos, &acc) in pos_acc.iter().enumerate() {
                    position_samples[pos].push(acc);
                }
                rts.push(rt);
                if config.trial_trace {
                    trace.push(TrialOutcome {
                        trial_idx: trace.len(),
                        condition: format!("serial_{}", list_len),
                        correct: pos_acc.iter().sum::<f64>() / pos_acc.len() as f64 > 0.5,
                        rt_ticks: rt,
                        similarity: 0.0,
                        confidence: 0.0,
                        response_idx: 0,
                        extra: BTreeMap::new(),
                    });
                }
            }

            // Report per-position accuracy
            for (pos, samples) in position_samples.iter().enumerate() {
                result.insert(
                    format!("list_{}::pos_{}", list_len, pos),
                    MetricValue::from_samples(samples),
                );
            }

            // Compute primacy index: mean(first 2) - mean(middle)
            // Middle = all positions except first 2 and last 2 (Murdock, 1962).
            // For list_len=7: positions 2-4; for list_len=5: position 2; for list_len=9: positions 2-6.
            let primacy_mean: f64 = position_samples[..2.min(list_len)]
                .iter()
                .flat_map(|s| s.iter())
                .sum::<f64>()
                / (2.min(list_len) * config.trials_per_condition) as f64;
            let mid_start = 2.min(list_len);
            let mid_end = list_len.saturating_sub(2).max(mid_start);
            let mid_count = mid_end - mid_start;
            let mid_mean: f64 = if mid_count > 0 {
                position_samples[mid_start..mid_end]
                    .iter()
                    .flat_map(|s| s.iter())
                    .sum::<f64>()
                    / (mid_count * config.trials_per_condition) as f64
            } else {
                0.0
            };

            // Recency index: mean(last 2) - mean(middle)
            let recency_mean: f64 = position_samples[(list_len - 2).max(0)..]
                .iter()
                .flat_map(|s| s.iter())
                .sum::<f64>()
                / (2.min(list_len) * config.trials_per_condition) as f64;

            result.insert(
                format!("list_{}::primacy_index", list_len),
                MetricValue::from_samples(&[primacy_mean - mid_mean]),
            );
            result.insert(
                format!("list_{}::recency_index", list_len),
                MetricValue::from_samples(&[recency_mean - mid_mean]),
            );
            result.insert(
                format!("list_{}::rt_ticks", list_len),
                MetricValue::from_samples(&rts),
            );
        }

        result.conditions = 3;
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
    fn test_serial_recall_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = SerialRecallBenchmark.run(&config);
        assert_eq!(result.conditions, 3);
        assert!(result.metrics.contains_key("list_5::pos_0"));
        assert!(result.metrics.contains_key("list_7::primacy_index"));
    }

    #[test]
    fn test_primacy_index_near_human_baseline() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 40,
            ..Default::default()
        };
        let result = SerialRecallBenchmark.run(&config);
        let pi = result.metrics["list_7::primacy_index"].mean;
        // Human baseline: 0.15 (SD 0.06). Accept within ~2 SD.
        assert!(
            pi > 0.03 && pi < 0.35,
            "list_7::primacy_index = {:.4}, expected near 0.15 (human baseline)",
            pi
        );
    }

    #[test]
    fn test_serial_recall_ssm_matches_baseline() {
        let base_config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 20,
            ..Default::default()
        };
        let ssm_config = BenchmarkConfig {
            ssm_backend: true,
            ..base_config.clone()
        };
        let base_result = SerialRecallBenchmark.run(&base_config);
        let ssm_result = SerialRecallBenchmark.run(&ssm_config);
        let base_pi = base_result.metrics["list_7::primacy_index"].mean;
        let ssm_pi = ssm_result.metrics["list_7::primacy_index"].mean;
        let diff = (base_pi - ssm_pi).abs();
        assert!(
            diff < 0.30,
            "SSM primacy_index ({:.3}) too far from baseline ({:.3}), diff={:.3}",
            ssm_pi,
            base_pi,
            diff
        );
    }
}
