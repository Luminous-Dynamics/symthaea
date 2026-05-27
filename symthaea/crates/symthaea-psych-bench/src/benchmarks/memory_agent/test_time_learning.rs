// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Test-time learning (correction) task.
//!
//! Present a fact (A), then a correction (B), and verify that
//! querying retrieves B (the correction) rather than A.

use crate::adapter::StimulusAdapter;
use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::{WmConfig, WorkingMemory};
use std::collections::BTreeMap;

/// Test-time learning benchmark.
pub struct TestTimeLearningBenchmark;

struct CorrectionPair {
    original: &'static str,
    correction: &'static str,
    #[allow(dead_code)]
    query: &'static str,
}

impl TestTimeLearningBenchmark {
    fn pairs() -> Vec<CorrectionPair> {
        vec![
            CorrectionPair {
                original: "the meeting is on Tuesday",
                correction: "the meeting has been moved to Wednesday",
                query: "when is the meeting",
            },
            CorrectionPair {
                original: "the password is alpha one two three",
                correction: "the password has been changed to beta four five six",
                query: "what is the password",
            },
            CorrectionPair {
                original: "the project deadline is March",
                correction: "the deadline has been extended to April",
                query: "when is the project deadline",
            },
        ]
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let pairs = Self::pairs();
        let pair = &pairs[trial_idx % pairs.len()];

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        // Present original fact
        let orig_hv = adapter.encode(&Scenario::new(pair.original), dim);
        wm.perceive(orig_hv);
        wm.tick();

        // Filler ticks
        for _ in 0..3 {
            wm.tick();
        }

        // Present correction
        let corr_hv = adapter.encode(&Scenario::new(pair.correction), dim);
        wm.perceive(corr_hv);
        wm.tick();

        // Post-correction retention interval: ticks without new perceives.
        // Both original and correction remain in WM (only 2 perceives total,
        // well under capacity=7). The correction has higher activation (more
        // recent), creating a recency advantage.
        for _ in 0..5 {
            wm.tick();
        }

        // Query
        let original_hv = adapter.encode(&Scenario::new(pair.original), dim);
        let correction_hv = adapter.encode(&Scenario::new(pair.correction), dim);

        let orig_sim = wm.activation_weighted_similarity(&original_hv) as f64;
        let corr_sim = wm.activation_weighted_similarity(&correction_hv) as f64;

        // Softmax response selection: retrieval is probabilistic, not
        // deterministic. The correction has a recency advantage but retrieval
        // competition, source confusion, and decision noise prevent perfect
        // updating (Karpicke & Roediger 2008 — testing effect).
        // beta controls discriminability: lower beta = more noise = more errors.
        // Time pressure: base beta=4.0 yields ~85% correction retrieval (Karpicke & Roediger, 2008);
        // -1.5/unit reduces discriminability, modeling source confusion under SAT (Luce, 1986).
        let beta = 4.0 - config.time_pressure * 1.5;
        let p_correction =
            (beta * corr_sim).exp() / ((beta * corr_sim).exp() + (beta * orig_sim).exp());

        let roll_seed = config.trial_seed("memory", "ttl_decision", trial_idx);
        let mut ns = roll_seed ^ 0x9E3779B97F4A7C15;
        ns ^= ns << 13;
        ns ^= ns >> 7;
        ns ^= ns << 17;
        let roll = (ns % 10000) as f64 / 10000.0;

        let accuracy = if roll < p_correction { 1.0 } else { 0.0 };

        // RT proxy: retrieval difficulty drives deliberation.
        // Base 4 ticks (encoding + filler + correction presentation),
        // closer competition between original and correction = longer
        // deliberation (Karpicke & Roediger, 2008 — testing effect).
        let decision_margin = (corr_sim - orig_sim).abs().clamp(0.0, 1.0);
        let rt_ticks = 4.0 + (1.0 - decision_margin) * 6.0;

        (accuracy, rt_ticks)
    }
}

impl PsychBenchmark for TestTimeLearningBenchmark {
    fn name(&self) -> &str {
        "MemoryAgent::TestTimeLearning"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Test-Enhanced Learning",
            citation: "Roediger & Karpicke (2006)",
            year: 2006,
            doi: Some("10.1111/j.1467-9280.2006.01693.x"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut accuracies = Vec::new();
        let mut rts = Vec::new();
        for trial in 0..config.trials_per_condition {
            let (acc, rt) = self.run_trial(config, trial);
            accuracies.push(acc);
            rts.push(rt);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "ttl".to_string(),
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
            "correction_accuracy",
            MetricValue::from_samples(&accuracies),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 1;
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
    fn test_ttl_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = TestTimeLearningBenchmark.run(&config);
        assert!(result.metrics.contains_key("correction_accuracy"));
    }
}
