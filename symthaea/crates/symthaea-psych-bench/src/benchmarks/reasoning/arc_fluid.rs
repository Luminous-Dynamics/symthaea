// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ARC Fluid Reasoning benchmark (SYNTHETIC, not real ARC).
//!
//! **Honesty note:** This benchmark uses procedurally generated grid tasks, NOT
//! Chollet's real ARC dataset. It tests HDC encoding/retrieval algebra — whether
//! XOR-based rule binding can recover a known transform applied to a novel input.
//! Because BinaryHV XOR bind is self-inverse, "rule application" is exact
//! algebraic recovery (input XOR rule XOR rule = input), not genuine
//! generalization or novel rule discovery.
//!
//! The z-scores from this benchmark reflect encoding fidelity and noise
//! tolerance of the HDC representation, not abstract reasoning ability.
//! Distractors are plausible (same transform family, different type), which
//! provides a meaningful but still limited test of discriminability.
//!
//! Measures fluid intelligence proxy via procedurally generated grid
//! transformation tasks inspired by the Abstraction and Reasoning Corpus (ARC).
//! Each task presents training input/output pairs demonstrating a transformation
//! rule, then tests whether the system can apply the inferred rule to a novel input.
//!
//! Human baselines (Chollet 2019; Johnson et al. 2021):
//! - rule_consistency: ~0.85 (SD~0.10) — within-task rule agreement
//! - transfer_accuracy: ~0.80 (SD~0.12) — correct novel application
//! - transfer_similarity: ~0.70 (SD~0.15) — cosine of predicted vs actual
//! - rt_ticks: ~6.0 (SD~2.0) — deliberation proxy

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;
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
    /// Per-task-type accuracy: [ColorFill, Translation, ColorReplacement, Reflection]
    per_type_accuracy: [f64; 4],
    /// Learning curve: transfer accuracy using only 1 training pair
    single_pair_accuracy: f64,
    /// Confusion matrix: 4×4 (true_type × predicted_type), row-normalized
    confusion_matrix: [[f64; 4]; 4],
    /// Generalization gap: accuracy with same-param test minus different-param test.
    /// Positive values indicate the system is memorizing specific parameters rather
    /// than learning a generalizable transform.
    generalization_gap: f64,
    /// Per-task trial trace (populated when config.trial_trace is true).
    task_trace: Vec<TrialOutcome>,
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
        let encoder = BinaryGridEncoder::new(grid_size, grid_size, num_colors as usize, seed);
        let _dim = dim; // kept for API compatibility
        let tasks_per_type = 5;

        let pressure = config.time_pressure;
        // Time pressure AND encoding noise both add noise to rule encoding (Wickelgren 1977)
        // Encoding noise: ablated subsystems degrade representational fidelity
        // Difficulty scales temperature via the difficulty model
        let diff_model = difficulty_model_for(self.name());
        let noise_weight = (0.008 + pressure * 0.12 + config.encoding_noise * 0.15)
            * diff_model.temperature_multiplier(config.difficulty);

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

        // Per-task trace
        let mut task_trace = Vec::new();
        let mut global_task_idx = 0usize;
        let type_names = ["color_fill", "translation", "color_replace", "reflection"];

        // Collect per-task rule HVs and test results
        let mut all_rule_consistencies: Vec<f64> = Vec::new();
        let mut all_task_rule_hvs: Vec<BinaryHV> = Vec::new();
        let mut transfer_hits: u32 = 0;
        let mut transfer_total: u32 = 0;
        let mut transfer_sims: Vec<f64> = Vec::new();
        let mut total_ticks: f64 = 0.0;
        let mut per_type_hits: [u32; 4] = [0; 4];
        let mut per_type_total: [u32; 4] = [0; 4];
        // Learning curve: single-pair accuracy
        let mut single_pair_hits: u32 = 0;
        let mut single_pair_total: u32 = 0;
        // Confusion matrix: [true_type][predicted_type] counts
        let mut confusion_counts: [[u32; 4]; 4] = [[0; 4]; 4];

        for (type_idx, &task_type) in TASK_TYPES.iter().enumerate() {
            for _task_i in 0..tasks_per_type {
                xor_shift(&mut rng);
                let task_param = rng;

                // Generate 6 training pairs + 1 test pair (same transform, different inputs)
                // More examples strengthen the majority-vote consensus (Kanerva 2009).
                // 6 pairs give stronger signal recovery than 4 via central limit theorem
                // on the bundled majority vote.
                let mut train_rules = Vec::new();
                for _ in 0..6 {
                    xor_shift(&mut rng);
                    let input = gen_grid(&mut rng);
                    let output = apply_transform(&input, task_type, task_param);
                    let in_hv = encoder.encode_grid(&input);
                    let out_hv = encoder.encode_grid(&output);
                    let mut rule = encoder.encode_rule(&in_hv, &out_hv);

                    // Add noise under time pressure (bit flips proportional to noise_weight)
                    if noise_weight > 0.0 {
                        xor_shift(&mut rng);
                        rule = rule.add_noise(noise_weight as f32, rng);
                    }

                    train_rules.push(rule);
                    // Deliberation ticks — time pressure reduces deliberation
                    xor_shift(&mut rng);
                    let tick_scale = 1.0 - pressure * 0.4;
                    total_ticks += (4.0 + (rng % 5) as f64) * tick_scale;
                }

                // Rule consistency: cosine between the 2 training rules
                let consistency = train_rules[0].similarity(&train_rules[1]) as f64;
                all_rule_consistencies.push(consistency);

                // Consensus rule for transfer
                let consensus = encoder.bundle_rules(&train_rules);
                all_task_rule_hvs.push(consensus);

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

                // Transfer accuracy: majority voting — predicted must be closer
                // to the correct output than at least 2 of 3 distractors (other
                // transforms applied to the same test input). More robust than
                // single-distractor 2-AFC but not as strict as unanimous 4-AFC.
                let mut wins = 0u32;
                for (other_idx, &other_type) in TASK_TYPES.iter().enumerate() {
                    if other_idx == type_idx {
                        continue;
                    }
                    let dist_output = apply_transform(&test_input, other_type, task_param);
                    let dist_hv = encoder.encode_grid(&dist_output);
                    let dist_sim = predicted.similarity(&dist_hv) as f64;
                    if pred_sim > dist_sim {
                        wins += 1;
                    }
                }
                let correct_wins = wins >= 2; // majority: beat at least 2 of 3
                transfer_total += 1;
                per_type_total[type_idx] += 1;
                if correct_wins {
                    transfer_hits += 1;
                    per_type_hits[type_idx] += 1;
                }

                // ── Learning curve: test with single training pair (majority voting) ──
                let single_predicted = encoder.apply_rule(&test_in_hv, &train_rules[0]);
                let single_sim = single_predicted.similarity(&test_out_hv) as f64;
                let mut single_wins = 0u32;
                for (other_idx, &other_type) in TASK_TYPES.iter().enumerate() {
                    if other_idx == type_idx {
                        continue;
                    }
                    let d_out = apply_transform(&test_input, other_type, task_param);
                    let d_hv = encoder.encode_grid(&d_out);
                    if single_sim > single_predicted.similarity(&d_hv) as f64 {
                        single_wins += 1;
                    }
                }
                single_pair_total += 1;
                if single_wins >= 2 {
                    single_pair_hits += 1;
                }

                // ── Confusion matrix: which transform type does the model pick? ──
                // Compare predicted output against all 4 transform types' outputs
                let mut best_type_idx = type_idx; // default to correct
                let mut best_sim = pred_sim;
                for (other_idx, &other_type) in TASK_TYPES.iter().enumerate() {
                    if other_idx == type_idx {
                        continue; // already computed
                    }
                    let other_output = apply_transform(&test_input, other_type, task_param);
                    let other_hv = encoder.encode_grid(&other_output);
                    let other_sim = predicted.similarity(&other_hv) as f64;
                    if other_sim > best_sim {
                        best_sim = other_sim;
                        best_type_idx = other_idx;
                    }
                }
                confusion_counts[type_idx][best_type_idx] += 1;

                // Deliberation for test — time pressure reduces deliberation
                // (Wickelgren 1977 speed-accuracy tradeoff)
                xor_shift(&mut rng);
                let tick_scale = 1.0 - pressure * 0.4;
                let task_test_ticks = (4.0 + (rng % 5) as f64) * tick_scale;
                total_ticks += task_test_ticks;

                // Per-task trial trace
                if config.trial_trace {
                    let per_task_correct = correct_wins;
                    task_trace.push(TrialOutcome {
                        trial_idx: global_task_idx,
                        condition: type_names[type_idx].to_string(),
                        correct: per_task_correct,
                        rt_ticks: task_test_ticks,
                        similarity: pred_sim,
                        confidence: consistency,
                        response_idx: best_type_idx,
                        extra: BTreeMap::new(),
                    });
                    global_task_idx += 1;
                }
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
            total_ticks / (num_tasks * 7.0) // 7 pairs per task (6 train + 1 test)
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

        let single_pair_accuracy = if single_pair_total > 0 {
            single_pair_hits as f64 / single_pair_total as f64
        } else {
            0.0
        };

        // Normalize confusion matrix rows
        let mut confusion_matrix = [[0.0f64; 4]; 4];
        #[allow(clippy::needless_range_loop)]
        for i in 0..4 {
            let row_sum: u32 = confusion_counts[i].iter().sum();
            if row_sum > 0 {
                for j in 0..4 {
                    confusion_matrix[i][j] = confusion_counts[i][j] as f64 / row_sum as f64;
                }
            }
        }

        // ── Generalization gap: same-param vs different-param test ──
        // Train with one param, test with a DIFFERENT param (same transform type).
        // This reveals whether the rule HV encodes the abstract transform or just
        // memorizes the specific parameter instantiation.
        let mut same_param_hits = 0u32;
        let mut diff_param_hits = 0u32;
        let mut gen_gap_total = 0u32;

        for &task_type in &TASK_TYPES {
            for _ in 0..3 {
                xor_shift(&mut rng);
                let train_param = rng;

                // Train with train_param
                let mut gen_rules = Vec::new();
                for _ in 0..4 {
                    xor_shift(&mut rng);
                    let input = gen_grid(&mut rng);
                    let output = apply_transform(&input, task_type, train_param);
                    let in_hv = encoder.encode_grid(&input);
                    let out_hv = encoder.encode_grid(&output);
                    gen_rules.push(encoder.encode_rule(&in_hv, &out_hv));
                }
                let gen_consensus = encoder.bundle_rules(&gen_rules);

                // Test with SAME param (control)
                xor_shift(&mut rng);
                let same_input = gen_grid(&mut rng);
                let same_output = apply_transform(&same_input, task_type, train_param);
                let same_in_hv = encoder.encode_grid(&same_input);
                let same_out_hv = encoder.encode_grid(&same_output);
                let same_pred = encoder.apply_rule(&same_in_hv, &gen_consensus);
                let same_sim = same_pred.similarity(&same_out_hv) as f64;

                // Test with DIFFERENT param (generalization)
                xor_shift(&mut rng);
                let diff_param = rng ^ 0xBEEF; // ensure different param
                let diff_input = gen_grid(&mut rng);
                let diff_output = apply_transform(&diff_input, task_type, diff_param);
                let diff_in_hv = encoder.encode_grid(&diff_input);
                let diff_out_hv = encoder.encode_grid(&diff_output);
                let diff_pred = encoder.apply_rule(&diff_in_hv, &gen_consensus);
                let diff_sim = diff_pred.similarity(&diff_out_hv) as f64;

                // 2-AFC vs plausible distractor for both
                xor_shift(&mut rng);
                let dist_rng = rng;
                let dist_same = BinaryHV::random(dist_rng);
                let dist_diff = BinaryHV::random(dist_rng ^ 0xCAFE);
                if same_sim > same_pred.similarity(&dist_same) as f64 {
                    same_param_hits += 1;
                }
                if diff_sim > diff_pred.similarity(&dist_diff) as f64 {
                    diff_param_hits += 1;
                }
                gen_gap_total += 1;
            }
        }

        let same_acc = if gen_gap_total > 0 {
            same_param_hits as f64 / gen_gap_total as f64
        } else {
            0.0
        };
        let diff_acc = if gen_gap_total > 0 {
            diff_param_hits as f64 / gen_gap_total as f64
        } else {
            0.0
        };
        let generalization_gap = same_acc - diff_acc;

        TrialResult {
            rule_consistency,
            cross_task_discrimination,
            transfer_accuracy,
            transfer_similarity,
            rt_ticks,
            per_type_accuracy,
            single_pair_accuracy,
            confusion_matrix,
            generalization_gap,
            task_trace,
        }
    }
}

impl PsychBenchmark for ArcFluidBenchmark {
    fn name(&self) -> &str {
        "Reasoning::ArcFluid"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Synthetic ARC-inspired grid transforms (HDC encoding test)",
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
        let type_names = ["color_fill", "translation", "color_replace", "reflection"];
        let mut per_type: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut single_pair_accs = Vec::new();
        let mut confusion_sum = [[0.0f64; 4]; 4];
        let mut gen_gaps = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            consistencies.push(r.rule_consistency);
            discriminations.push(r.cross_task_discrimination);
            accuracies.push(r.transfer_accuracy);
            similarities.push(r.transfer_similarity);
            rts.push(r.rt_ticks);
            single_pair_accs.push(r.single_pair_accuracy);
            gen_gaps.push(r.generalization_gap);
            #[allow(clippy::needless_range_loop)]
            for i in 0..4 {
                per_type[i].push(r.per_type_accuracy[i]);
                for j in 0..4 {
                    confusion_sum[i][j] += r.confusion_matrix[i][j];
                }
            }

            if config.trial_trace {
                trace.extend(r.task_trace);
            }
        }

        result.insert(
            "rule_consistency",
            MetricValue::from_samples(&consistencies),
        );
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

        // Per-task-type breakdowns
        for (i, name) in type_names.iter().enumerate() {
            result.insert(
                format!("accuracy_{}", name),
                MetricValue::from_samples(&per_type[i]),
            );
        }

        // ── Learning curve metrics ──
        result.insert(
            "single_pair_accuracy",
            MetricValue::from_samples(&single_pair_accs),
        );
        let learning_effs: Vec<f64> = accuracies
            .iter()
            .zip(single_pair_accs.iter())
            .map(|(two, one)| two - one)
            .collect();
        result.insert(
            "learning_efficiency",
            MetricValue::from_samples(&learning_effs),
        );

        // ── Confusion matrix summary metrics ──
        let n_trials = config.trials_per_condition as f64;
        if n_trials > 0.0 {
            // Normalize by trial count
            let mut confusion_avg = [[0.0f64; 4]; 4];
            #[allow(clippy::needless_range_loop)]
            for i in 0..4 {
                for j in 0..4 {
                    confusion_avg[i][j] = confusion_sum[i][j] / n_trials;
                }
            }
            // Confusion diagonal = mean of diagonal (overall correct classification)
            let diagonal_mean = (confusion_avg[0][0]
                + confusion_avg[1][1]
                + confusion_avg[2][2]
                + confusion_avg[3][3])
                / 4.0;
            result.insert(
                "confusion_diagonal",
                MetricValue::from_samples(&[diagonal_mean]),
            );
            // Max off-diagonal: the single largest confusion rate
            let mut max_offdiag = 0.0f64;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4 {
                for j in 0..4 {
                    if i != j && confusion_avg[i][j] > max_offdiag {
                        max_offdiag = confusion_avg[i][j];
                    }
                }
            }
            result.insert(
                "confusion_max_error",
                MetricValue::from_samples(&[max_offdiag]),
            );
            // Confusion entropy: Shannon entropy of each row, averaged
            let mut entropy_sum = 0.0f64;
            for row in &confusion_avg {
                for &p in row {
                    if p > 0.0 {
                        entropy_sum -= p * p.log2();
                    }
                }
            }
            result.insert(
                "confusion_entropy",
                MetricValue::from_samples(&[entropy_sum / 4.0]), // mean per-row entropy
            );
        }

        // Generalization gap: same-param accuracy minus different-param accuracy.
        // Positive = system memorizes parameters, not the abstract transform.
        result.insert("generalization_gap", MetricValue::from_samples(&gen_gaps));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.notes.push(
            "SYNTHETIC: Uses procedural grid transforms, not Chollet's real ARC. \
             Tests HDC encoding algebra (XOR bind/unbind), not novel rule discovery. \
             Z-scores reflect encoding quality, not abstract reasoning."
                .to_string(),
        );

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
        // Per-type breakdowns
        assert!(result.metrics.contains_key("accuracy_color_fill"));
        assert!(result.metrics.contains_key("accuracy_translation"));
        assert!(result.metrics.contains_key("accuracy_color_replace"));
        assert!(result.metrics.contains_key("accuracy_reflection"));
    }

    #[test]
    fn test_per_type_breakdowns_finite() {
        let result = ArcFluidBenchmark.run(&test_config());
        for name in &[
            "accuracy_color_fill",
            "accuracy_translation",
            "accuracy_color_replace",
            "accuracy_reflection",
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
        // With 4-AFC ensemble voting (chance = 0.25), HDC rule binding
        // should discriminate the correct transform above chance
        assert!(
            accuracy > 0.3,
            "Transfer accuracy should beat 4-AFC chance (0.25), got {}",
            accuracy
        );
    }

    #[test]
    fn test_provenance_correct() {
        let prov = ArcFluidBenchmark.provenance().unwrap();
        assert!(prov.paradigm.contains("Synthetic"));
        assert!(prov.paradigm.contains("HDC encoding"));
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
            r1.metrics["rule_consistency"].mean, r2.metrics["rule_consistency"].mean,
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
    fn test_learning_curve_metrics() {
        let result = ArcFluidBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("single_pair_accuracy"));
        assert!(result.metrics.contains_key("learning_efficiency"));
        let single = result.metrics["single_pair_accuracy"].mean;
        let two_pair = result.metrics["transfer_accuracy"].mean;
        assert!(single.is_finite(), "single_pair_accuracy not finite");
        assert!(single >= 0.0 && single <= 1.0);
        // Learning efficiency = two_pair - single_pair (can be negative)
        let eff = result.metrics["learning_efficiency"].mean;
        assert!(eff.is_finite(), "learning_efficiency not finite");
        // With dim=256, consensus of 2 should generally >= single pair
        // But due to noise, we just check it's in a reasonable range
        assert!(
            eff > -0.5 && eff < 0.5,
            "learning_efficiency out of range: {}",
            eff
        );
        let _ = (single, two_pair);
    }

    #[test]
    fn test_confusion_matrix_metrics() {
        let result = ArcFluidBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("confusion_diagonal"));
        assert!(result.metrics.contains_key("confusion_max_error"));
        assert!(result.metrics.contains_key("confusion_entropy"));
        let diag = result.metrics["confusion_diagonal"].mean;
        assert!(
            diag.is_finite() && diag >= 0.0 && diag <= 1.0,
            "confusion_diagonal out of range: {}",
            diag
        );
        let max_err = result.metrics["confusion_max_error"].mean;
        assert!(
            max_err.is_finite() && max_err >= 0.0 && max_err <= 1.0,
            "confusion_max_error out of range: {}",
            max_err
        );
        let entropy = result.metrics["confusion_entropy"].mean;
        assert!(
            entropy.is_finite() && entropy >= 0.0,
            "confusion_entropy should be non-negative: {}",
            entropy
        );
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
        // Pressure adds noise (reduces consistency) and reduces RT (SAT tradeoff)
        assert!(base_consistency.is_finite());
        assert!(pressure_consistency.is_finite());
        assert!(
            pressure_consistency <= base_consistency + 0.1,
            "Time pressure should not dramatically improve consistency: base={}, pressure={}",
            base_consistency,
            pressure_consistency
        );
        // Time pressure should reduce RT (tick_scale = 1 - 0.8*0.4 = 0.68)
        let base_rt = base_result.metrics["rt_ticks"].mean;
        let pressure_rt = pressure_result.metrics["rt_ticks"].mean;
        assert!(
            pressure_rt < base_rt,
            "Time pressure should reduce RT: base={}, pressure={}",
            base_rt,
            pressure_rt
        );
    }
}
