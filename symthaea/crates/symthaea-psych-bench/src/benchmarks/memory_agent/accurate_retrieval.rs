// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Accurate retrieval task.
//!
//! Store N facts as HDC vectors, run delay ticks, then query
//! working memory to see if the facts are retrievable.

use crate::adapter::StimulusAdapter;
use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::{WmConfig, WorkingMemory};
use std::collections::BTreeMap;

/// Accurate retrieval benchmark.
pub struct AccurateRetrievalBenchmark;

impl AccurateRetrievalBenchmark {
    fn run_trial(
        &self,
        num_facts: usize,
        delay_ticks: usize,
        config: &BenchmarkConfig,
        _trial_idx: usize,
    ) -> (f64, f64) {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        let facts = [
            "the capital of France is Paris",
            "water boils at one hundred degrees",
            "the sun is a star",
            "cats are mammals",
            "the earth orbits the sun",
            "humans have five senses",
            "iron is a metal element",
            "the moon orbits the earth",
        ];

        let active_facts: Vec<&str> = facts.iter().take(num_facts).copied().collect();

        // Store facts
        for fact in &active_facts {
            let hv = adapter.encode(&Scenario::new(*fact), dim);
            wm.perceive(hv);
            wm.tick();
        }

        // Delay
        for _ in 0..delay_ticks {
            wm.tick();
        }

        // Query: how many facts are still retrievable?
        let contents = wm.contents();
        let mut retrieved = 0;

        for fact in &active_facts {
            let query_hv = adapter.encode(&Scenario::new(*fact), dim);
            let max_sim = contents
                .iter()
                .map(|item| item.similarity(&query_hv))
                .fold(0.0f32, f32::max);

            // Time pressure: base 0.3 yields ~85% retrieval; +0.10/unit raises criterion,
            // modeling truncated memory search under deadline (Ratcliff & McKoon, 2008 DDM).
            let threshold = 0.3 + config.time_pressure as f32 * 0.10;
            if max_sim > threshold {
                retrieved += 1;
            }
        }

        // RT proxy: encoding ticks scale with num_facts, delay adds retention
        // interval cost, and retrieval decision margin modulates deliberation.
        // Base 3 ticks (motor + perceptual), +0.8/fact encoding, +0.3/delay tick.
        let retrieval_margin = if !contents.is_empty() {
            let query_hv = adapter.encode(&Scenario::new(active_facts[0]), dim);
            let max_sim = contents
                .iter()
                .map(|item| item.similarity(&query_hv))
                .fold(0.0f32, f32::max) as f64;
            max_sim.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let rt_ticks = 3.0
            + num_facts as f64 * 0.8
            + delay_ticks as f64 * 0.3
            + (1.0 - retrieval_margin) * 4.0;

        let accuracy = retrieved as f64 / active_facts.len() as f64;
        (accuracy, rt_ticks)
    }
}

impl PsychBenchmark for AccurateRetrievalBenchmark {
    fn name(&self) -> &str {
        "MemoryAgent::AccurateRetrieval"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Memory Retrieval Accuracy",
            citation: "Anderson (1983)",
            year: 1983,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut all_rts = Vec::new();

        for (num_facts, delay) in [(3, 2), (5, 5), (7, 10)] {
            let mut accuracies = Vec::new();
            let mut rts = Vec::new();
            for trial in 0..config.trials_per_condition {
                let (acc, rt) = self.run_trial(num_facts, delay, config, trial);
                accuracies.push(acc);
                rts.push(rt);
                if config.trial_trace {
                    trace.push(TrialOutcome {
                        trial_idx: trace.len(),
                        condition: format!("facts_{}_delay_{}", num_facts, delay),
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
                format!("facts_{}_delay_{}::accuracy", num_facts, delay),
                MetricValue::from_samples(&accuracies),
            );
            result.insert(
                format!("facts_{}_delay_{}::rt_ticks", num_facts, delay),
                MetricValue::from_samples(&rts),
            );
            all_rts.extend_from_slice(&rts);
        }

        // Overall RT across all conditions
        result.insert("rt_ticks", MetricValue::from_samples(&all_rts));

        // Compute overall retrieval accuracy (mean of all condition accuracies)
        let all_accuracies: Vec<f64> = result
            .metrics
            .iter()
            .filter(|(k, _)| k.ends_with("::accuracy"))
            .map(|(_, m)| m.mean)
            .collect();
        if !all_accuracies.is_empty() {
            let overall = all_accuracies.iter().sum::<f64>() / all_accuracies.len() as f64;
            result.insert(
                "overall_retrieval_accuracy",
                MetricValue::from_samples(&[overall]),
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
    fn test_accurate_retrieval_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = AccurateRetrievalBenchmark.run(&config);
        assert!(result.metrics.contains_key("facts_3_delay_2::accuracy"));
    }
}
