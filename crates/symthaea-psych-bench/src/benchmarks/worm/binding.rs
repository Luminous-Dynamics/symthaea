//! Feature binding task.
//!
//! Tests whether WM correctly binds features (color+shape) together,
//! vs remembering features independently. Compares binding accuracy
//! against feature-only accuracy (partial-feature lure detection).

use crate::adapter::spatial::{VisualObject, VisualObjectAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};

/// Binding benchmark testing feature-conjunction WM.
pub struct BindingBenchmark;

impl BindingBenchmark {
    /// Run a single trial.
    /// Returns (binding_correct, feature_only_correct).
    fn run_trial(
        &self,
        set_size: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64) {
        let dim = config.dimension;
        let seed = config.trial_seed("worm", &format!("binding_{}", set_size), trial_idx);
        let adapter = VisualObjectAdapter::default();

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;

        // Generate study objects (unique color-shape combos)
        let mut objects = Vec::with_capacity(set_size);
        for _ in 0..set_size {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            objects.push(VisualObject::new(
                (rng_state % 6) as u32,
                ((rng_state >> 8) % 5) as u32,
                ((rng_state >> 16) % 3) as u32,
            ));
        }

        // Present study objects
        for obj in &objects {
            let hv = adapter.encode(obj, dim);
            wm.perceive(hv);
            wm.tick();
        }

        // Delay
        for _ in 0..2 {
            wm.tick();
        }

        let contents = wm.contents();

        // Test 1: Binding probe (exact object from study set)
        let target = &objects[0];
        let target_hv = adapter.encode(target, dim);
        let binding_sim = contents
            .iter()
            .map(|item| item.similarity(&target_hv))
            .fold(0.0f32, f32::max);

        // Test 2: Feature-swap lure (swap color from one object with shape from another)
        let lure = if set_size >= 2 {
            VisualObject::new(objects[0].color, objects[1].shape, objects[0].size)
        } else {
            VisualObject::new(
                (objects[0].color + 3) % 6,
                objects[0].shape,
                objects[0].size,
            )
        };
        let lure_hv = adapter.encode(&lure, dim);
        let lure_sim = contents
            .iter()
            .map(|item| item.similarity(&lure_hv))
            .fold(0.0f32, f32::max);

        // Test 3: Novel object (not from study set)
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let novel = VisualObject::new(
            ((rng_state % 6) as u32 + 4) % 6,
            ((rng_state >> 8) % 5 + 3) as u32 % 5,
            ((rng_state >> 16) % 3 + 2) as u32 % 3,
        );
        let novel_hv = adapter.encode(&novel, dim);
        let novel_sim = contents
            .iter()
            .map(|item| item.similarity(&novel_hv))
            .fold(0.0f32, f32::max);

        // Binding accuracy: correctly accept target AND reject lure
        let binding_correct = if binding_sim > lure_sim { 1.0 } else { 0.0 };

        // Feature-only accuracy: correctly distinguish target from novel
        let feature_correct = if binding_sim > novel_sim { 1.0 } else { 0.0 };

        (binding_correct, feature_correct)
    }
}

impl PsychBenchmark for BindingBenchmark {
    fn name(&self) -> &str {
        "WorM::Binding"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        for k in [2, 4, 6] {
            let mut binding_accs = Vec::new();
            let mut feature_accs = Vec::new();

            for trial in 0..config.trials_per_condition {
                let (bind, feat) = self.run_trial(k, config, trial);
                binding_accs.push(bind);
                feature_accs.push(feat);
            }

            result.insert(
                format!("set_{}::binding_accuracy", k),
                MetricValue::from_samples(&binding_accs),
            );
            result.insert(
                format!("set_{}::feature_accuracy", k),
                MetricValue::from_samples(&feature_accs),
            );

            // Binding deficit: feature_accuracy - binding_accuracy
            let deficits: Vec<f64> = feature_accs
                .iter()
                .zip(binding_accs.iter())
                .map(|(f, b)| f - b)
                .collect();
            result.insert(
                format!("set_{}::binding_deficit", k),
                MetricValue::from_samples(&deficits),
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
    fn test_binding_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = BindingBenchmark.run(&config);
        assert_eq!(result.conditions, 3);
        assert!(result.metrics.contains_key("set_2::binding_accuracy"));
        assert!(result.metrics.contains_key("set_2::feature_accuracy"));
    }
}
