// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ARC Abductive Reasoning benchmark (SYNTHETIC, not real ARC).
//!
//! **Honesty note:** This benchmark uses procedurally generated grid tasks.
//! It tests HDC XOR-based unbinding (backward algebraic recovery), not genuine
//! abductive reasoning. With BinaryHV, unbind is XOR — the exact inverse of
//! bind — so "backward inference" is algebraically identical to forward
//! application. The z-scores reflect encoding fidelity, not reasoning ability.
//!
//! Tests backward causal inference: given the output of a transformation and
//! the learned rule, infer what the input must have been. This is the inverse
//! of forward reasoning (arc_fluid) and tests a fundamentally different
//! cognitive operation — abduction (Peirce 1903; Harman 1965).
//!
//! Paradigm:
//!   1. Present 2 training pairs (input → output) to learn a transform rule.
//!   2. Present a novel TEST OUTPUT (the effect).
//!   3. Ask: "Which of 4 candidate inputs produced this output?"
//!   4. 4-AFC: 1 correct input + 3 random distractor inputs.
//!
//! HDC mechanism: unbind(output_hv, rule_hv) ≈ input_hv (approximate inverse).
//!
//! Human baselines (estimated from Johnson et al. 2021; Peirce 1903 tradition):
//! - abduction_accuracy: ~0.70 (SD~0.15) — backward inference is harder than forward
//! - unbinding_similarity: ~0.30 (SD~0.10) — cosine(estimated_input, true_input)
//! - abduction_rt_ticks: ~7.0 (SD~2.5) — backward inference takes longer

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;
use symthaea_core::hdc::grid_encoder::GridEncoder;

/// ARC-style abductive reasoning benchmark (backward inference).
pub struct ArcAbductiveBenchmark;

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

fn xor_shift(s: &mut u64) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
}

fn gen_grid(rng: &mut u64, size: usize, num_colors: u8) -> Vec<Vec<u8>> {
    let mut grid = vec![vec![0u8; size]; size];
    for row in grid.iter_mut() {
        for cell in row.iter_mut() {
            xor_shift(rng);
            *cell = (*rng % num_colors as u64) as u8;
        }
    }
    grid
}

fn apply_transform(
    grid: &[Vec<u8>],
    task_type: TaskType,
    param: u64,
    num_colors: u8,
) -> Vec<Vec<u8>> {
    match task_type {
        TaskType::ColorFill => {
            let color = (param % num_colors as u64) as u8;
            let top = (param / 7 % 3) as usize;
            let left = (param / 11 % 3) as usize;
            GridEncoder::fill_region(grid, top, left, top + 1, left + 1, color)
        }
        TaskType::Translation => {
            let dx = ((param % 3) as i32) - 1;
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
}

struct AbductiveTrialResult {
    abduction_accuracy: f64,
    unbinding_similarity: f64,
    rt_ticks: f64,
    /// Per-type abduction accuracy: [ColorFill, Translation, ColorReplacement, Reflection]
    per_type_accuracy: [f64; 4],
}

impl ArcAbductiveBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> AbductiveTrialResult {
        let seed = config.trial_seed("reasoning", "arc_abductive", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;
        let grid_size = 5;
        let num_colors: u8 = 6;
        let pressure = config.time_pressure;
        let noise_weight = 0.05 + pressure * 0.15;
        let tick_scale = 1.0 - pressure * 0.4;

        let encoder = BinaryGridEncoder::new(grid_size, grid_size, num_colors as usize, seed);
        let tasks_per_type = 5;
        let mut total_ticks: f64 = 0.0;
        let mut abduction_hits: u32 = 0;
        let mut abduction_total: u32 = 0;
        let mut unbinding_sims: Vec<f64> = Vec::new();
        let mut per_type_hits: [u32; 4] = [0; 4];
        let mut per_type_total: [u32; 4] = [0; 4];

        for (type_idx, &task_type) in TASK_TYPES.iter().enumerate() {
            for _task_i in 0..tasks_per_type {
                xor_shift(&mut rng);
                let task_param = rng;

                // ── Learn the rule from 2 training pairs ──
                let mut train_rules = Vec::new();
                for _ in 0..2 {
                    xor_shift(&mut rng);
                    let input = gen_grid(&mut rng, grid_size, num_colors);
                    let output = apply_transform(&input, task_type, task_param, num_colors);
                    let in_hv = encoder.encode_grid(&input);
                    let out_hv = encoder.encode_grid(&output);
                    let mut rule = encoder.encode_rule(&in_hv, &out_hv);

                    if noise_weight > 0.0 {
                        xor_shift(&mut rng);
                        rule = rule.add_noise(noise_weight as f32, rng);
                    }
                    train_rules.push(rule);
                    xor_shift(&mut rng);
                    total_ticks += (4.0 + (rng % 5) as f64) * tick_scale;
                }
                let consensus_rule = encoder.bundle_rules(&train_rules);

                // ── Abductive test: given output, infer input ──
                // Generate the true test input and its output
                xor_shift(&mut rng);
                let true_input = gen_grid(&mut rng, grid_size, num_colors);
                let test_output = apply_transform(&true_input, task_type, task_param, num_colors);
                let true_input_hv = encoder.encode_grid(&true_input);
                let test_output_hv = encoder.encode_grid(&test_output);

                // Forward candidate matching: for each candidate input, compute
                // candidate.bind(rule) → predicted output, compare to test_output.
                // Correct input produces input² * output (positive bias with output),
                // while wrong inputs produce random correlation.

                // Unbinding similarity — with BinaryHV XOR, this is exact inversion
                let estimated_input = test_output_hv.bind(&consensus_rule);
                let unbind_sim = estimated_input.similarity(&true_input_hv) as f64;
                unbinding_sims.push(unbind_sim);

                // 4-AFC via forward matching: apply rule to each candidate, compare to output
                let correct_predicted = true_input_hv.bind(&consensus_rule);
                let correct_sim = correct_predicted.similarity(&test_output_hv) as f64;
                let mut best_distractor_sim = f64::NEG_INFINITY;
                for _ in 0..3 {
                    xor_shift(&mut rng);
                    let distractor_input = gen_grid(&mut rng, grid_size, num_colors);
                    let distractor_hv = encoder.encode_grid(&distractor_input);
                    let dist_predicted = distractor_hv.bind(&consensus_rule);
                    let dist_sim = dist_predicted.similarity(&test_output_hv) as f64;
                    if dist_sim > best_distractor_sim {
                        best_distractor_sim = dist_sim;
                    }
                }

                abduction_total += 1;
                per_type_total[type_idx] += 1;
                if correct_sim > best_distractor_sim {
                    abduction_hits += 1;
                    per_type_hits[type_idx] += 1;
                }

                // Backward inference takes longer (Harman 1965)
                xor_shift(&mut rng);
                total_ticks += (5.0 + (rng % 6) as f64) * tick_scale;
            }
        }

        let abduction_accuracy = if abduction_total > 0 {
            abduction_hits as f64 / abduction_total as f64
        } else {
            0.0
        };
        let unbinding_similarity = if unbinding_sims.is_empty() {
            0.0
        } else {
            unbinding_sims.iter().sum::<f64>() / unbinding_sims.len() as f64
        };
        let num_tasks = (TASK_TYPES.len() * tasks_per_type) as f64;
        let rt_ticks = if num_tasks > 0.0 {
            total_ticks / (num_tasks * 3.0) // 2 train + 1 test per task
        } else {
            0.0
        };
        let per_type_accuracy = [
            if per_type_total[0] > 0 {
                per_type_hits[0] as f64 / per_type_total[0] as f64
            } else {
                0.0
            },
            if per_type_total[1] > 0 {
                per_type_hits[1] as f64 / per_type_total[1] as f64
            } else {
                0.0
            },
            if per_type_total[2] > 0 {
                per_type_hits[2] as f64 / per_type_total[2] as f64
            } else {
                0.0
            },
            if per_type_total[3] > 0 {
                per_type_hits[3] as f64 / per_type_total[3] as f64
            } else {
                0.0
            },
        ];

        AbductiveTrialResult {
            abduction_accuracy,
            unbinding_similarity,
            rt_ticks,
            per_type_accuracy,
        }
    }
}

impl PsychBenchmark for ArcAbductiveBenchmark {
    fn name(&self) -> &str {
        "Reasoning::ArcAbductive"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Synthetic abductive inference via HDC grid unbinding",
            citation: "Chollet (2019); Harman (1965)",
            year: 2019,
            doi: Some("10.48550/arXiv.1911.01547"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut accuracies = Vec::new();
        let mut similarities = Vec::new();
        let mut rts = Vec::new();
        let type_names = ["color_fill", "translation", "color_replace", "reflection"];
        let mut per_type: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            accuracies.push(r.abduction_accuracy);
            similarities.push(r.unbinding_similarity);
            rts.push(r.rt_ticks);
            for (i, acc) in r.per_type_accuracy.iter().enumerate() {
                per_type[i].push(*acc);
            }
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "arc_abductive".to_string(),
                    correct: r.abduction_accuracy > 0.5,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("abduction_accuracy", MetricValue::from_samples(&accuracies));
        result.insert(
            "unbinding_similarity",
            MetricValue::from_samples(&similarities),
        );
        result.insert("abduction_rt_ticks", MetricValue::from_samples(&rts));

        // Per-task-type breakdowns
        for (i, name) in type_names.iter().enumerate() {
            result.insert(
                format!("abduction_{}", name),
                MetricValue::from_samples(&per_type[i]),
            );
        }

        result.conditions = 4; // 4 task types
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.notes.push(
            "SYNTHETIC: Uses procedural grid transforms, not Chollet's real ARC. \
             Tests HDC XOR unbinding (algebraically identical to forward bind). \
             Z-scores reflect encoding fidelity, not abductive reasoning."
                .to_string(),
        );
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
    fn test_abductive_runs_with_metrics() {
        let result = ArcAbductiveBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("abduction_accuracy"));
        assert!(result.metrics.contains_key("unbinding_similarity"));
        assert!(result.metrics.contains_key("abduction_rt_ticks"));
        // Per-type
        assert!(result.metrics.contains_key("abduction_color_fill"));
        assert!(result.metrics.contains_key("abduction_translation"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = ArcAbductiveBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
            assert!(val.std_dev.is_finite(), "metric {} std_dev not finite", key);
        }
    }

    #[test]
    fn test_abduction_above_chance() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ArcAbductiveBenchmark.run(&config);
        let accuracy = result.metrics["abduction_accuracy"].mean;
        // 4-AFC chance = 0.25; abduction should beat it
        assert!(
            accuracy > 0.25,
            "Abduction accuracy should beat 4-AFC chance (0.25), got {}",
            accuracy
        );
    }

    #[test]
    fn test_unbinding_positive_signal() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ArcAbductiveBenchmark.run(&config);
        let sim = result.metrics["unbinding_similarity"].mean;
        // Unbinding should produce positive similarity (above 0.0)
        assert!(
            sim > 0.0,
            "Unbinding similarity should be positive, got {}",
            sim
        );
    }

    #[test]
    fn test_provenance_correct() {
        let prov = ArcAbductiveBenchmark.provenance().unwrap();
        assert!(prov.paradigm.contains("abductive"));
        assert!(prov.citation.contains("Harman"));
        assert_eq!(prov.year, 2019);
    }

    #[test]
    fn test_deterministic() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 2,
            seed: 42,
            ..Default::default()
        };
        let r1 = ArcAbductiveBenchmark.run(&config);
        let r2 = ArcAbductiveBenchmark.run(&config);
        assert_eq!(
            r1.metrics["abduction_accuracy"].mean, r2.metrics["abduction_accuracy"].mean,
            "Same seed should produce identical results"
        );
    }

    #[test]
    fn test_time_pressure_reduces_rt() {
        let base = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            seed: 42,
            time_pressure: 0.0,
            ..Default::default()
        };
        let pressure = BenchmarkConfig {
            time_pressure: 0.8,
            ..base.clone()
        };
        let base_rt = ArcAbductiveBenchmark.run(&base).metrics["abduction_rt_ticks"].mean;
        let pressure_rt = ArcAbductiveBenchmark.run(&pressure).metrics["abduction_rt_ticks"].mean;
        assert!(
            pressure_rt < base_rt,
            "Time pressure should reduce RT: base={}, pressure={}",
            base_rt,
            pressure_rt
        );
    }

    #[test]
    fn test_per_type_breakdowns_finite() {
        let result = ArcAbductiveBenchmark.run(&test_config());
        for name in &[
            "abduction_color_fill",
            "abduction_translation",
            "abduction_color_replace",
            "abduction_reflection",
        ] {
            let val = &result.metrics[*name];
            assert!(
                val.mean.is_finite(),
                "Per-type metric {} is not finite",
                name
            );
            assert!(
                val.mean >= 0.0 && val.mean <= 1.0,
                "Per-type metric {} out of range: {}",
                name,
                val.mean
            );
        }
    }
}
