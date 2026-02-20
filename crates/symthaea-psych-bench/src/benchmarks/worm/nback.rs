//! N-back working memory task.
//!
//! The system sees a stream of items and must identify when the current item
//! matches the item N positions back. Tests the updating component of WM.

use crate::adapter::sequence::{SequenceAdapter, SequenceItem};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};

/// N-back benchmark testing the updating function of working memory.
pub struct NBackBenchmark;

impl NBackBenchmark {
    /// Run a single N-back trial and return (hit_rate, false_alarm_rate).
    fn run_trial(
        &self,
        n: usize,
        sequence_len: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64) {
        let dim = config.dimension;
        let seed = config.trial_seed("worm", &format!("nback_{}", n), trial_idx);
        let adapter = SequenceAdapter;

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        // Generate sequence with ~30% match targets
        let mut rng_state = seed;
        let vocab_size = 8u64;
        let mut sequence = Vec::with_capacity(sequence_len);
        for i in 0..sequence_len {
            let is_target = i >= n && {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                (rng_state % 100) < 30
            };
            if is_target {
                sequence.push(sequence[i - n]);
            } else {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                sequence.push(SequenceItem(rng_state % vocab_size));
            }
        }

        let mut hits = 0u64;
        let mut misses = 0u64;
        let mut false_alarms = 0u64;
        let mut correct_rejections = 0u64;

        // Present sequence to working memory
        for (i, &item) in sequence.iter().enumerate() {
            let hv = adapter.encode(&item, dim);
            wm.perceive(hv.clone());
            wm.tick();

            if i >= n {
                let is_target = sequence[i] == sequence[i - n];

                // Check if WM contains the n-back item by scanning WM contents
                let nback_hv = adapter.encode(&sequence[i - n], dim);
                let contents = wm.contents();

                // Find the best similarity to the n-back item in current WM
                let max_sim = contents
                    .iter()
                    .map(|wm_item| wm_item.similarity(&nback_hv))
                    .fold(0.0f32, f32::max);

                // System "responds match" if similarity exceeds threshold
                let threshold = 0.4;
                let responded_match = max_sim > threshold;

                match (is_target, responded_match) {
                    (true, true) => hits += 1,
                    (true, false) => misses += 1,
                    (false, true) => false_alarms += 1,
                    (false, false) => correct_rejections += 1,
                }
            }
        }

        let hit_rate = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };
        let fa_rate = if false_alarms + correct_rejections > 0 {
            false_alarms as f64 / (false_alarms + correct_rejections) as f64
        } else {
            0.0
        };

        (hit_rate, fa_rate)
    }
}

impl PsychBenchmark for NBackBenchmark {
    fn name(&self) -> &str {
        "WorM::N-back"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let sequence_len = 30;

        for n in [1, 2, 3] {
            let mut hit_rates = Vec::new();
            let mut fa_rates = Vec::new();
            let mut accuracies = Vec::new();

            for trial in 0..config.trials_per_condition {
                let (hr, fa) = self.run_trial(n, sequence_len, config, trial);
                hit_rates.push(hr);
                fa_rates.push(fa);
                // d'-like accuracy: hit_rate - false_alarm_rate
                accuracies.push(hr - fa);
            }

            result.insert(
                format!("nback_{}::hit_rate", n),
                MetricValue::from_samples(&hit_rates),
            );
            result.insert(
                format!("nback_{}::false_alarm_rate", n),
                MetricValue::from_samples(&fa_rates),
            );
            result.insert(
                format!("nback_{}::accuracy", n),
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
    fn test_nback_runs_without_panic() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let bench = NBackBenchmark;
        let result = bench.run(&config);
        assert_eq!(result.conditions, 3);
        assert!(result.metrics.contains_key("nback_1::hit_rate"));
        assert!(result.metrics.contains_key("nback_2::hit_rate"));
        assert!(result.metrics.contains_key("nback_3::hit_rate"));
    }

    #[test]
    fn test_nback_metrics_finite() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            dimension: 256,
            ..Default::default()
        };
        let result = NBackBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
            assert!(val.std_dev.is_finite(), "metric {} std_dev not finite", key);
        }
    }
}
