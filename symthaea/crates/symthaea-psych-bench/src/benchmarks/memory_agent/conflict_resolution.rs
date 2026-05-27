// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conflict resolution task.
//!
//! Present fact A, then contradicting fact not-A. Query which version
//! is retrieved. Tests whether the system has a consistent strategy
//! for handling conflicting information (recency, frequency, etc.).

use crate::adapter::StimulusAdapter;
use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::{WmConfig, WorkingMemory};
use std::collections::BTreeMap;

/// Conflict resolution benchmark.
pub struct ConflictResolutionBenchmark;

struct ConflictPair {
    fact_a: &'static str,
    fact_b: &'static str, // contradicts A
    #[allow(dead_code)]
    query: &'static str,
}

impl ConflictResolutionBenchmark {
    fn pairs() -> Vec<ConflictPair> {
        vec![
            ConflictPair {
                fact_a: "the store opens at nine in the morning",
                fact_b: "the store now opens at ten in the morning",
                query: "when does the store open",
            },
            ConflictPair {
                fact_a: "the team lead is Alice",
                fact_b: "the team lead has changed to Bob",
                query: "who is the team lead",
            },
            ConflictPair {
                fact_a: "the product costs fifty dollars",
                fact_b: "the product price has increased to sixty dollars",
                query: "how much does the product cost",
            },
        ]
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64, f64) {
        let dim = config.dimension;
        let seed = config.trial_seed("memory_agent", "conflict", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;
        let adapter = ScenarioAdapter;
        let pairs = Self::pairs();
        let pair = &pairs[trial_idx % pairs.len()];

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        // Present fact A
        let a_hv = adapter.encode(&Scenario::new(pair.fact_a), dim);
        wm.perceive(a_hv.clone());
        wm.tick();

        // Filler
        for _ in 0..3 {
            wm.tick();
        }

        // Present contradicting fact B
        let b_hv = adapter.encode(&Scenario::new(pair.fact_b), dim);
        wm.perceive(b_hv.clone());
        wm.tick();

        // Delay
        for _ in 0..2 {
            wm.tick();
        }

        // Query: softmax retrieval competition (not deterministic max).
        // Humans show ~65% recency preference, not 100%, because older
        // memories sometimes intrude (Oberauer, 2002).
        let a_sim = wm.activation_weighted_similarity(&a_hv) as f64;
        let b_sim = wm.activation_weighted_similarity(&b_hv) as f64;

        // Time pressure: base 0.25 yields ~65% recency preference (Oberauer, 2002 WM updating);
        // +0.15/unit adds retrieval noise, modeling noisier competition under SAT (Heitz, 2014).
        let diff_model = difficulty_model_for("MemoryAgent::ConflictResolution");
        let temperature = (0.25 + config.time_pressure * 0.15)
            * diff_model.temperature_multiplier(config.difficulty);
        let max_sim = a_sim.max(b_sim);
        let a_weight = ((a_sim - max_sim) / temperature).exp();
        let b_weight = ((b_sim - max_sim) / temperature).exp();
        let b_prob = b_weight / (a_weight + b_weight);

        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let roll = (rng % 10000) as f64 / 10000.0;
        let recency_correct = if roll < b_prob { 1.0 } else { 0.0 };

        // Consistency: the margin between the two
        let consistency = (b_sim - a_sim).abs();

        // RT proxy: conflict difficulty drives deliberation time.
        // Base 4 ticks (encoding + filler + retrieval), closer competing
        // memories (smaller margin) require more deliberation (Oberauer, 2002).
        let margin = consistency.clamp(0.0, 1.0);
        let rt_ticks = 4.0 + (1.0 - margin) * 6.0;

        (recency_correct, consistency, rt_ticks)
    }
}

impl PsychBenchmark for ConflictResolutionBenchmark {
    fn name(&self) -> &str {
        "MemoryAgent::ConflictResolution"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Memory Conflict Resolution",
            citation: "Anderson & Neely (1996)",
            year: 1996,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut recency_accs = Vec::new();
        let mut consistencies = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (rec, con, rt) = self.run_trial(config, trial);
            recency_accs.push(rec);
            consistencies.push(con);
            rts.push(rt);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "conflict".to_string(),
                    correct: rec > 0.5,
                    rt_ticks: rt,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "recency_preference",
            MetricValue::from_samples(&recency_accs),
        );
        result.insert(
            "resolution_consistency",
            MetricValue::from_samples(&consistencies),
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
    fn test_conflict_resolution_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = ConflictResolutionBenchmark.run(&config);
        assert!(result.metrics.contains_key("recency_preference"));
        assert!(result.metrics.contains_key("resolution_consistency"));
    }
}
