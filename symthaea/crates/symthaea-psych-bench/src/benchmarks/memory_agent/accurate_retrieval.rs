//! Accurate retrieval task.
//!
//! Store N facts as HDC vectors, run delay ticks, then query
//! working memory to see if the facts are retrievable.

use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};

/// Accurate retrieval benchmark.
pub struct AccurateRetrievalBenchmark;

impl AccurateRetrievalBenchmark {
    fn run_trial(
        &self,
        num_facts: usize,
        delay_ticks: usize,
        config: &BenchmarkConfig,
        _trial_idx: usize,
    ) -> f64 {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
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

            if max_sim > 0.3 {
                retrieved += 1;
            }
        }

        retrieved as f64 / active_facts.len() as f64
    }
}

impl PsychBenchmark for AccurateRetrievalBenchmark {
    fn name(&self) -> &str {
        "MemoryAgent::AccurateRetrieval"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        for (num_facts, delay) in [(3, 2), (5, 5), (7, 10)] {
            let mut accuracies = Vec::new();
            for trial in 0..config.trials_per_condition {
                let acc = self.run_trial(num_facts, delay, config, trial);
                accuracies.push(acc);
            }

            result.insert(
                format!("facts_{}_delay_{}::accuracy", num_facts, delay),
                MetricValue::from_samples(&accuracies),
            );
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
