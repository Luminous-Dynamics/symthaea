//! Long-range understanding task.
//!
//! Present a fact early (cycle 10), then query much later (cycle 190).
//! Tests episodic Phi-weighting: significant items should survive long delays.

use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};

/// Long-range understanding benchmark.
pub struct LongRangeBenchmark;

impl LongRangeBenchmark {
    fn run_trial(
        &self,
        delay_cycles: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> f64 {
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
        if max_sim > 0.2 { 1.0 } else { 0.0 }
    }
}

impl PsychBenchmark for LongRangeBenchmark {
    fn name(&self) -> &str {
        "MemoryAgent::LongRange"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        for delay in [5, 20, 50, 100] {
            let mut accuracies = Vec::new();
            for trial in 0..config.trials_per_condition {
                let acc = self.run_trial(delay, config, trial);
                accuracies.push(acc);
            }

            result.insert(
                format!("delay_{}::retention", delay),
                MetricValue::from_samples(&accuracies),
            );
        }

        result.conditions = 4;
        result.trials_per_condition = config.trials_per_condition;
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
