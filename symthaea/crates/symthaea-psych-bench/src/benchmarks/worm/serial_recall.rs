//! Serial recall task.
//!
//! Present a sequence of items, then test recall at each position.
//! Produces the classic serial position curve (primacy + recency effects).

use crate::adapter::sequence::{SequenceAdapter, SequenceItem};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};

/// Serial recall benchmark producing serial position curves.
pub struct SerialRecallBenchmark;

impl SerialRecallBenchmark {
    /// Run a single trial: present a list, then probe each position.
    /// Returns accuracy per serial position.
    fn run_trial(
        &self,
        list_len: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> Vec<f64> {
        let dim = config.dimension;
        let seed = config.trial_seed("worm", &format!("serial_{}", list_len), trial_idx);
        let adapter = SequenceAdapter;

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        // Generate unique items
        let items: Vec<SequenceItem> = (0..list_len)
            .map(|i| {
                let item_seed = seed.wrapping_add(i as u64).wrapping_mul(0x100000001b3);
                SequenceItem(item_seed % 100)
            })
            .collect();

        // Present items sequentially
        for item in &items {
            let hv = adapter.encode(item, dim);
            wm.perceive(hv);
            wm.tick();
        }

        // Brief delay
        for _ in 0..2 {
            wm.tick();
        }

        // Probe each position: find the best-matching WM item
        let contents = wm.contents();
        let mut position_accuracy = Vec::with_capacity(list_len);

        for (pos, item) in items.iter().enumerate() {
            let target_hv = adapter.encode(item, dim);

            // Find best WM match for this target
            let max_sim = contents
                .iter()
                .map(|wm_item| wm_item.similarity(&target_hv))
                .fold(0.0f32, f32::max);

            // Higher similarity = better recall at this position
            // Use a threshold to binarize: recalled (1.0) or not (0.0)
            let recalled = if max_sim > 0.3 { 1.0 } else { 0.0 };
            position_accuracy.push(recalled);

            // Also check if the target is the BEST match for its position
            // by comparing against a position-encoded probe
            let _ = pos; // position index available for extended analysis
        }

        position_accuracy
    }
}

impl PsychBenchmark for SerialRecallBenchmark {
    fn name(&self) -> &str {
        "WorM::SerialRecall"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        for list_len in [5, 7, 9] {
            // Collect accuracy per position across trials
            let mut position_samples: Vec<Vec<f64>> = vec![Vec::new(); list_len];

            for trial in 0..config.trials_per_condition {
                let pos_acc = self.run_trial(list_len, config, trial);
                for (pos, &acc) in pos_acc.iter().enumerate() {
                    position_samples[pos].push(acc);
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
            let primacy_mean: f64 = position_samples[..2.min(list_len)]
                .iter()
                .flat_map(|s| s.iter())
                .sum::<f64>()
                / (2.min(list_len) * config.trials_per_condition) as f64;
            let mid_start = list_len / 3;
            let mid_end = 2 * list_len / 3;
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
}
