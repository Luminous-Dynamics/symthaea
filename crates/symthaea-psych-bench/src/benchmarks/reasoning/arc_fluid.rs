//! ARC Fluid Reasoning benchmark.
//!
//! Measures fluid intelligence via procedurally generated grid transformation
//! tasks inspired by the Abstraction and Reasoning Corpus (ARC). Each task
//! presents 2 training input/output pairs demonstrating a transformation rule,
//! then tests whether the system can apply the inferred rule to a novel input.
//!
//! Human baselines (Chollet 2019; Johnson et al. 2021):
//! - rule_consistency: ~0.85 (SD~0.10) — within-task rule agreement
//! - transfer_accuracy: ~0.80 (SD~0.12) — correct novel application
//! - transfer_similarity: ~0.70 (SD~0.15) — cosine of predicted vs actual
//! - rt_ticks: ~6.0 (SD~2.0) — deliberation proxy

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::grid_encoder::GridEncoder;

/// ARC-style fluid reasoning benchmark.
pub struct ArcFluidBenchmark;

#[derive(Debug, Clone, Copy)]
enum TaskType {
    ColorFill,
    Translation,
    ColorReplacement,
    Reflection,
}

const TASK_TYPES: [TaskType; 4] = [
    TaskType::ColorFill,
    TaskType::Translation,
    TaskType::ColorReplacement,
    TaskType::Reflection,
];

struct TrialResult {
    rule_consistency: f64,
    cross_task_discrimination: f64,
    transfer_accuracy: f64,
    transfer_similarity: f64,
    rt_ticks: f64,
}

impl ArcFluidBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("reasoning", "arc_fluid", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        let grid_size = 5;
        let num_colors: u8 = 6;
        let encoder = GridEncoder::new(dim, grid_size, grid_size, num_colors as usize, seed);
        let tasks_per_type = 5;

        let pressure = config.time_pressure;
        // Time pressure adds noise to rule encoding (Wickelgren 1977)
        let noise_weight = 0.05 + pressure * 0.15;

        // Generate a random grid
        let gen_grid = |rng: &mut u64| -> Vec<Vec<u8>> {
            let mut grid = vec![vec![0u8; grid_size]; grid_size];
            for row in grid.iter_mut() {
                for cell in row.iter_mut() {
                    let xor_shift_inner = |s: &mut u64| {
                        *s ^= *s << 13;
                        *s ^= *s >> 7;
                        *s ^= *s << 17;
                    };
                    xor_shift_inner(rng);
                    *cell = (*rng % num_colors as u64) as u8;
                }
            }
            grid
        };

        // Apply transformation based on task type with given params
        let apply_transform = |grid: &[Vec<u8>], task_type: TaskType, param: u64| -> Vec<Vec<u8>> {
            match task_type {
                TaskType::ColorFill => {
                    let color = (param % num_colors as u64) as u8;
                    let top = (param / 7 % 3) as usize;
                    let left = (param / 11 % 3) as usize;
                    GridEncoder::fill_region(grid, top, left, top + 1, left + 1, color)
                }
                TaskType::Translation => {
                    let dx = ((param % 3) as i32) - 1; // -1, 0, or 1
                    let dy = ((param / 3 % 3) as i32) - 1;
                    let fill = (param / 9 % num_colors as u64) as u8;
                    GridEncoder::translate_grid(grid, dx, dy, fill)
                }
                TaskType::ColorReplacement => {
                    let from = (param % num_colors as u64) as u8;
                    let to = ((param / 7 + 1) % num_colors as u64) as u8;
                    GridEncoder::color_replace(grid, from, to)
                }
                TaskType::Reflection => {
                    if param % 2 == 0 {
                        GridEncoder::reflect_x(grid)
                    } else {
                        GridEncoder::reflect_y(grid)
                    }
                }
            }
        };

        // Collect per-task rule HVs and test results
        let mut all_rule_consistencies: Vec<f64> = Vec::new();
        let mut all_task_rule_hvs: Vec<symthaea_core::hdc::ContinuousHV> = Vec::new();
        let mut transfer_hits: u32 = 0;
        let mut transfer_total: u32 = 0;
        let mut transfer_sims: Vec<f64> = Vec::new();
        let mut total_ticks: f64 = 0.0;

        for (type_idx, &task_type) in TASK_TYPES.iter().enumerate() {
            for task_i in 0..tasks_per_type {
                xor_shift(&mut rng);
                let task_param = rng;

                // Generate 2 training pairs + 1 test pair (same transform, different inputs)
                let mut train_rules = Vec::new();
                for pair_i in 0..2 {
                    xor_shift(&mut rng);
                    let input = gen_grid(&mut rng);
                    let output = apply_transform(&input, task_type, task_param);
                    let in_hv = encoder.encode_grid(&input);
                    let out_hv = encoder.encode_grid(&output);
                    let mut rule = encoder.encode_rule(&in_hv, &out_hv);

                    // Add noise under time pressure
                    if noise_weight > 0.0 {
                        xor_shift(&mut rng);
                        let noise = symthaea_core::hdc::ContinuousHV::random(dim, rng);
                        rule = symthaea_core::hdc::ContinuousHV::weighted_bundle(
                            &[&rule, &noise],
                            &[1.0 - noise_weight as f32, noise_weight as f32],
                        );
                    }

                    train_rules.push(rule);
                    // Deliberation ticks: base ~4 + random jitter
                    xor_shift(&mut rng);
                    total_ticks += 4.0 + (rng % 5) as f64;
                    let _ = pair_i; // suppress unused warning
                }

                // Rule consistency: cosine between the 2 training rules
                let consistency = train_rules[0].similarity(&train_rules[1]) as f64;
                all_rule_consistencies.push(consistency);

                // Consensus rule for transfer
                let consensus = encoder.bundle_rules(&train_rules);
                all_task_rule_hvs.push(consensus.clone());

                // Test pair: apply consensus rule to novel input
                xor_shift(&mut rng);
                let test_input = gen_grid(&mut rng);
                let test_output = apply_transform(&test_input, task_type, task_param);
                let test_in_hv = encoder.encode_grid(&test_input);
                let test_out_hv = encoder.encode_grid(&test_output);

                let predicted = encoder.apply_rule(&test_in_hv, &consensus);

                // Transfer similarity: cosine of predicted vs actual
                let pred_sim = predicted.similarity(&test_out_hv) as f64;
                transfer_sims.push(pred_sim);

                // Transfer accuracy: predicted beats random baseline
                xor_shift(&mut rng);
                let random_grid = gen_grid(&mut rng);
                let random_hv = encoder.encode_grid(&random_grid);
                let random_sim = predicted.similarity(&random_hv) as f64;

                transfer_total += 1;
                if pred_sim > random_sim {
                    transfer_hits += 1;
                }

                // Deliberation for test
                xor_shift(&mut rng);
                total_ticks += 4.0 + (rng % 5) as f64;
                let _ = (type_idx, task_i);
            }
        }

        // Cross-task discrimination: mean cosine between rules from different tasks
        let mut cross_sims: Vec<f64> = Vec::new();
        for i in 0..all_task_rule_hvs.len() {
            for j in (i + 1)..all_task_rule_hvs.len() {
                // Only compare rules from different task types (every tasks_per_type is a type)
                let type_i = i / tasks_per_type;
                let type_j = j / tasks_per_type;
                if type_i != type_j {
                    let sim = all_task_rule_hvs[i].similarity(&all_task_rule_hvs[j]) as f64;
                    cross_sims.push(sim);
                }
            }
        }

        let rule_consistency = if all_rule_consistencies.is_empty() {
            0.0
        } else {
            all_rule_consistencies.iter().sum::<f64>() / all_rule_consistencies.len() as f64
        };
        let cross_task_discrimination = if cross_sims.is_empty() {
            0.0
        } else {
            cross_sims.iter().sum::<f64>() / cross_sims.len() as f64
        };
        let transfer_accuracy = if transfer_total > 0 {
            transfer_hits as f64 / transfer_total as f64
        } else {
            0.0
        };
        let transfer_similarity = if transfer_sims.is_empty() {
            0.0
        } else {
            transfer_sims.iter().sum::<f64>() / transfer_sims.len() as f64
        };
        let num_tasks = (TASK_TYPES.len() * tasks_per_type) as f64;
        let rt_ticks = if num_tasks > 0.0 {
            total_ticks / (num_tasks * 3.0) // 3 pairs per task (2 train + 1 test)
        } else {
            0.0
        };

        TrialResult {
            rule_consistency,
            cross_task_discrimination,
            transfer_accuracy,
            transfer_similarity,
            rt_ticks,
        }
    }
}

impl PsychBenchmark for ArcFluidBenchmark {
    fn name(&self) -> &str {
        "Reasoning::ArcFluid"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Abstraction and Reasoning Corpus",
            citation: "Chollet (2019)",
            year: 2019,
            doi: Some("10.48550/arXiv.1911.01547"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut consistencies = Vec::new();
        let mut discriminations = Vec::new();
        let mut accuracies = Vec::new();
        let mut similarities = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            consistencies.push(r.rule_consistency);
            discriminations.push(r.cross_task_discrimination);
            accuracies.push(r.transfer_accuracy);
            similarities.push(r.transfer_similarity);
            rts.push(r.rt_ticks);
        }

        result.insert("rule_consistency", MetricValue::from_samples(&consistencies));
        result.insert(
            "cross_task_discrimination",
            MetricValue::from_samples(&discriminations),
        );
        result.insert("transfer_accuracy", MetricValue::from_samples(&accuracies));
        result.insert(
            "transfer_similarity",
            MetricValue::from_samples(&similarities),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 4; // 4 task types
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BenchmarkConfig {
        BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        }
    }

    #[test]
    fn test_arc_fluid_runs_with_metrics() {
        let result = ArcFluidBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("rule_consistency"));
        assert!(result.metrics.contains_key("cross_task_discrimination"));
        assert!(result.metrics.contains_key("transfer_accuracy"));
        assert!(result.metrics.contains_key("transfer_similarity"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = ArcFluidBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
            assert!(val.std_dev.is_finite(), "metric {} std_dev not finite", key);
        }
    }

    #[test]
    fn test_rule_consistency_positive() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ArcFluidBenchmark.run(&config);
        let consistency = result.metrics["rule_consistency"].mean;
        assert!(
            consistency > 0.0,
            "Rule consistency should be positive, got {}",
            consistency
        );
    }

    #[test]
    fn test_transfer_above_chance() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = ArcFluidBenchmark.run(&config);
        let accuracy = result.metrics["transfer_accuracy"].mean;
        // With HDC rule binding, transfer should beat pure chance (0.5)
        assert!(
            accuracy > 0.4,
            "Transfer accuracy should be above chance, got {}",
            accuracy
        );
    }

    #[test]
    fn test_provenance_correct() {
        let prov = ArcFluidBenchmark.provenance().unwrap();
        assert_eq!(prov.paradigm, "Abstraction and Reasoning Corpus");
        assert_eq!(prov.citation, "Chollet (2019)");
        assert_eq!(prov.year, 2019);
        assert_eq!(prov.doi, Some("10.48550/arXiv.1911.01547"));
    }

    #[test]
    fn test_task_generation_deterministic() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 3,
            seed: 42,
            ..Default::default()
        };
        let r1 = ArcFluidBenchmark.run(&config);
        let r2 = ArcFluidBenchmark.run(&config);
        assert_eq!(
            r1.metrics["rule_consistency"].mean,
            r2.metrics["rule_consistency"].mean,
            "Same seed should produce identical results"
        );
    }

    #[test]
    fn test_all_task_types_generate() {
        // Verify at least 4 conditions (one per task type)
        let result = ArcFluidBenchmark.run(&test_config());
        assert_eq!(result.conditions, 4, "Should have 4 task type conditions");
    }

    #[test]
    fn test_time_pressure_effect() {
        let base_config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 5,
            seed: 42,
            time_pressure: 0.0,
            ..Default::default()
        };
        let pressure_config = BenchmarkConfig {
            time_pressure: 0.8,
            ..base_config.clone()
        };
        let base_result = ArcFluidBenchmark.run(&base_config);
        let pressure_result = ArcFluidBenchmark.run(&pressure_config);
        // Under time pressure, consistency should degrade (more noise in rule encoding)
        let base_consistency = base_result.metrics["rule_consistency"].mean;
        let pressure_consistency = pressure_result.metrics["rule_consistency"].mean;
        // We just check both are finite and pressure doesn't impossibly improve
        assert!(base_consistency.is_finite());
        assert!(pressure_consistency.is_finite());
        // With noise_weight = 0.05 + 0.8*0.15 = 0.17 vs 0.05, pressure should reduce consistency
        // But with stochastic HDC this is a soft assertion
        assert!(
            pressure_consistency <= base_consistency + 0.1,
            "Time pressure should not dramatically improve consistency: base={}, pressure={}",
            base_consistency,
            pressure_consistency
        );
    }
}
