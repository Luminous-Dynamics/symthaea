//! Conflict resolution task.
//!
//! Present fact A, then contradicting fact not-A. Query which version
//! is retrieved. Tests whether the system has a consistent strategy
//! for handling conflicting information (recency, frequency, etc.).

use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};

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

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let pairs = Self::pairs();
        let pair = &pairs[trial_idx % pairs.len()];

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
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

        // Query
        let contents = wm.contents();
        let a_sim: f32 = contents
            .iter()
            .map(|item| item.similarity(&a_hv))
            .fold(0.0f32, f32::max);
        let b_sim: f32 = contents
            .iter()
            .map(|item| item.similarity(&b_hv))
            .fold(0.0f32, f32::max);

        // "Recency correct" if B (most recent) is retrieved
        let recency_correct = if b_sim > a_sim { 1.0 } else { 0.0 };

        // Consistency: the margin between the two
        let consistency = (b_sim - a_sim).abs() as f64;

        (recency_correct, consistency)
    }
}

impl PsychBenchmark for ConflictResolutionBenchmark {
    fn name(&self) -> &str {
        "MemoryAgent::ConflictResolution"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut recency_accs = Vec::new();
        let mut consistencies = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (rec, con) = self.run_trial(config, trial);
            recency_accs.push(rec);
            consistencies.push(con);
        }

        result.insert("recency_preference", MetricValue::from_samples(&recency_accs));
        result.insert("resolution_consistency", MetricValue::from_samples(&consistencies));

        result.conditions = 1;
        result.trials_per_condition = config.trials_per_condition;
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
