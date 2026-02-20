//! Test-time learning (correction) task.
//!
//! Present a fact (A), then a correction (B), and verify that
//! querying retrieves B (the correction) rather than A.

use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};

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

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
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

        // Delay
        for _ in 0..2 {
            wm.tick();
        }

        // Query
        let original_hv = adapter.encode(&Scenario::new(pair.original), dim);
        let correction_hv = adapter.encode(&Scenario::new(pair.correction), dim);

        let orig_sim = wm.activation_weighted_similarity(&original_hv);
        let corr_sim = wm.activation_weighted_similarity(&correction_hv);

        // Correct if correction is more accessible than original
        if corr_sim > orig_sim { 1.0 } else { 0.0 }
    }
}

impl PsychBenchmark for TestTimeLearningBenchmark {
    fn name(&self) -> &str {
        "MemoryAgent::TestTimeLearning"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut accuracies = Vec::new();
        for trial in 0..config.trials_per_condition {
            accuracies.push(self.run_trial(config, trial));
        }

        result.insert("correction_accuracy", MetricValue::from_samples(&accuracies));

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
