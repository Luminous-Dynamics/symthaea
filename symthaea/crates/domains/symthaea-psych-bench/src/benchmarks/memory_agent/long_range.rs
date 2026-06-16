// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Long-range understanding task.
//!
//! Present a fact early (cycle 10), then query much later (cycle 190).
//! Tests episodic Phi-weighting: significant items should survive long delays.

use crate::adapter::StimulusAdapter;
use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::{WmConfig, WorkingMemory};
use std::collections::BTreeMap;

/// Long-range understanding benchmark.
pub struct LongRangeBenchmark;

impl LongRangeBenchmark {
    fn run_trial(
        &self,
        delay_cycles: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64) {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let seed = config.trial_seed("memory_agent", &format!("long_{}", delay_cycles), trial_idx);

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        // Warm-up ticks
        for _ in 0..5 {
            wm.tick();
        }

        // Present the key fact
        let fact = "the treasure is hidden under the old oak tree";
        let fact_hv = adapter.encode(&Scenario::new(fact), dim);
        wm.perceive(fact_hv.clone());
        wm.tick();

        // Fill WM with distractors during the delay
        let distractors = [
            "the weather is sunny today",
            "birds are singing in the garden",
            "a car passes on the street",
            "the clock strikes twelve",
            "leaves fall from the trees",
        ];

        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        for _ in 0..delay_cycles {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let distractor = distractors[(rng_state as usize) % distractors.len()];
            let hv = adapter.encode(&Scenario::new(distractor), dim);
            wm.perceive(hv);
            wm.tick();
        }

        // Query: is the key fact still accessible?
        let contents = wm.contents();
        let max_sim = contents
            .iter()
            .map(|item| item.similarity(&fact_hv))
            .fold(0.0f32, f32::max);

        // For long delays, the fact may have been evicted but should be
        // in episodic memory; here we test WM persistence directly
        // Time pressure: base 0.2 threshold for long-range retrieval; +0.10/unit raises criterion,
        // modeling truncated memory search under deadline (Ratcliff & McKoon, 2008 DDM).
        let diff_model = difficulty_model_for("MemoryAgent::LongRange");
        let threshold = 0.2 * diff_model.interference_multiplier(config.difficulty) as f32
            + config.time_pressure as f32 * 0.10;
        let accuracy = if max_sim > threshold { 1.0 } else { 0.0 };

        // RT proxy: delay ticks add retention interval cost, retrieval
        // decision margin modulates deliberation. Base 4 ticks (warmup +
        // encoding + motor), +0.15/delay cycle, weaker match = longer search.
        let retrieval_margin = (max_sim as f64).clamp(0.0, 1.0);
        let rt_ticks = 4.0 + delay_cycles as f64 * 0.15 + (1.0 - retrieval_margin) * 5.0;

        (accuracy, rt_ticks)
    }
}

impl PsychBenchmark for LongRangeBenchmark {
    fn name(&self) -> &str {
        "MemoryAgent::LongRange"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Long-Range Memory Dependency",
            citation: "Ebbinghaus (1885)",
            year: 1885,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        for delay in [5, 20, 50, 100] {
            let mut accuracies = Vec::new();
            let mut rts = Vec::new();
            for trial in 0..config.trials_per_condition {
                let (acc, rt) = self.run_trial(delay, config, trial);
                accuracies.push(acc);
                rts.push(rt);
                if config.trial_trace {
                    trace.push(TrialOutcome {
                        trial_idx: trace.len(),
                        condition: format!("delay_{}", delay),
                        correct: acc > 0.5,
                        rt_ticks: rt,
                        similarity: 0.0,
                        confidence: 0.0,
                        response_idx: 0,
                        extra: BTreeMap::new(),
                    });
                }
            }

            result.insert(
                format!("delay_{}::retention", delay),
                MetricValue::from_samples(&accuracies),
            );
            result.insert(
                format!("delay_{}::rt_ticks", delay),
                MetricValue::from_samples(&rts),
            );
        }

        result.conditions = 4;
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
    fn test_long_range_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = LongRangeBenchmark.run(&config);
        assert!(result.metrics.contains_key("delay_5::retention"));
        assert!(result.metrics.contains_key("delay_100::retention"));
    }
}
