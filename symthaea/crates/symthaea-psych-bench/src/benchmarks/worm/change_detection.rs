//! Change detection working memory task.
//!
//! Present an array of K visual objects, brief delay, then probe with
//! the original or a modified array. Tests maintenance capacity.

use crate::adapter::spatial::{VisualObject, VisualObjectAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};

use symthaea_core::hdc::ContinuousHV;

/// Change detection benchmark testing WM maintenance capacity.
pub struct ChangeDetectionBenchmark;

impl ChangeDetectionBenchmark {
    /// Run a single trial: present K objects, optionally change one, probe.
    /// Returns 1.0 for correct, 0.0 for incorrect.
    fn run_trial(
        &self,
        set_size: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> f64 {
        let dim = config.dimension;
        let seed = config.trial_seed("worm", &format!("cd_{}", set_size), trial_idx);
        let adapter = VisualObjectAdapter::default();

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
        });

        // Generate K unique visual objects
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        let mut objects = Vec::with_capacity(set_size);
        for i in 0..set_size {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            objects.push(VisualObject::new(
                (rng_state % 6) as u32,
                (i as u64 % 4) as u32,
                (rng_state % 3) as u32,
            ));
        }

        // Encode and present each object to WM
        let mut original_hvs = Vec::with_capacity(set_size);
        for obj in &objects {
            let hv = adapter.encode(obj, dim);
            original_hvs.push(hv.clone());
            wm.perceive(hv);
            wm.tick();
        }

        // Bundle original array
        let original_bundle = ContinuousHV::bundle_owned(&original_hvs);

        // Delay ticks (retention interval)
        for _ in 0..3 {
            wm.tick();
        }

        // Decide whether to change an item (50% probability)
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let is_changed = rng_state % 2 == 0;

        // Create probe array
        let mut probe_objects = objects.clone();
        if is_changed {
            let change_idx = (rng_state as usize / 2) % set_size;
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            probe_objects[change_idx] = VisualObject::new(
                ((rng_state % 6) as u32 + 3) % 6, // different color
                probe_objects[change_idx].shape,
                probe_objects[change_idx].size,
            );
        }

        let probe_hvs: Vec<ContinuousHV> = probe_objects
            .iter()
            .map(|obj| adapter.encode(obj, dim))
            .collect();
        let probe_bundle = ContinuousHV::bundle_owned(&probe_hvs);

        // Compare original bundle (as remembered by WM) with probe
        let contents = wm.contents();
        let wm_bundle = if contents.is_empty() {
            ContinuousHV::zero(dim)
        } else {
            ContinuousHV::bundle_owned(contents)
        };

        let sim_to_original = wm_bundle.similarity(&original_bundle);
        let sim_to_probe = wm_bundle.similarity(&probe_bundle);

        // System responds "same" if probe similarity >= original similarity
        let responded_same = sim_to_probe >= sim_to_original - 0.01;

        let correct = if is_changed {
            !responded_same // Should detect the change
        } else {
            responded_same // Should confirm same
        };

        if correct { 1.0 } else { 0.0 }
    }
}

impl PsychBenchmark for ChangeDetectionBenchmark {
    fn name(&self) -> &str {
        "WorM::ChangeDetection"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        for k in [2, 4, 6, 8] {
            let mut accuracies = Vec::new();

            for trial in 0..config.trials_per_condition {
                let acc = self.run_trial(k, config, trial);
                accuracies.push(acc);
            }

            result.insert(
                format!("set_size_{}::accuracy", k),
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
    fn test_change_detection_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = ChangeDetectionBenchmark.run(&config);
        assert_eq!(result.conditions, 4);
        assert!(result.metrics.contains_key("set_size_2::accuracy"));
        assert!(result.metrics.contains_key("set_size_8::accuracy"));
    }
}
