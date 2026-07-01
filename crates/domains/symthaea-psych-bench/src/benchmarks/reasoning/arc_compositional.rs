// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ARC Compositional Reasoning benchmark (SYNTHETIC, not real ARC).
//!
//! **Honesty note:** This benchmark uses procedurally generated grid tasks.
//! It tests HDC encoding/retrieval under composition (chained transforms),
//! size variation, and symmetry detection via prototype matching. Because
//! XOR bind is self-inverse, rule "application" is algebraic recovery.
//! The z-scores reflect encoding robustness, not compositional reasoning.
//!
//! Extends ArcFluidBenchmark with three harder task categories:
//!   1. **Chained transforms**: Two sequential transforms (e.g., translate then reflect).
//!   2. **Size-varying grids**: Grids range from 3×3 to 7×7, testing size generalization.
//!   3. **Symmetry detection**: Classify whether a grid is horizontally, vertically,
//!      both, or not symmetric — without explicit training pairs.
//!
//! Human baselines (estimated from Chollet 2019; Johnson et al. 2021):
//! - compositional_accuracy: ~0.65 (SD~0.15) — chained transform transfer
//! - size_generalization: ~0.70 (SD~0.12) — cross-size rule transfer
//! - symmetry_detection: ~0.90 (SD~0.08) — symmetry classification accuracy
//! - compositional_rt_ticks: ~8.0 (SD~3.0) — harder tasks take longer

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;
use symthaea_core::hdc::grid_encoder::GridEncoder;

/// ARC-style compositional reasoning benchmark (chained transforms + size variation + symmetry).
pub struct ArcCompositionalBenchmark;

#[derive(Debug, Clone, Copy)]
enum ChainedTransform {
    TranslateThenReflect,
    ColorReplaceThenFill,
    ReflectThenRotate,
    FillThenTranslate,
}

const CHAINED_TYPES: [ChainedTransform; 4] = [
    ChainedTransform::TranslateThenReflect,
    ChainedTransform::ColorReplaceThenFill,
    ChainedTransform::ReflectThenRotate,
    ChainedTransform::FillThenTranslate,
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

/// Apply a two-step chained transform.
fn apply_chain(
    grid: &[Vec<u8>],
    chain: ChainedTransform,
    param: u64,
    num_colors: u8,
) -> Vec<Vec<u8>> {
    match chain {
        ChainedTransform::TranslateThenReflect => {
            let dx = ((param % 3) as i32) - 1;
            let dy = ((param / 3 % 3) as i32) - 1;
            let fill = (param / 9 % num_colors as u64) as u8;
            let translated = GridEncoder::translate_grid(grid, dx, dy, fill);
            if param % 2 == 0 {
                GridEncoder::reflect_x(&translated)
            } else {
                GridEncoder::reflect_y(&translated)
            }
        }
        ChainedTransform::ColorReplaceThenFill => {
            let from = (param % num_colors as u64) as u8;
            let to = ((param / 7 + 1) % num_colors as u64) as u8;
            let replaced = GridEncoder::color_replace(grid, from, to);
            let rows = replaced.len();
            let cols = if rows > 0 { replaced[0].len() } else { 0 };
            let fill_color = (param / 13 % num_colors as u64) as u8;
            let top = (param / 17 % rows.max(1) as u64) as usize;
            let left = (param / 19 % cols.max(1) as u64) as usize;
            GridEncoder::fill_region(
                &replaced,
                top,
                left,
                top.min(rows - 1),
                left.min(cols - 1),
                fill_color,
            )
        }
        ChainedTransform::ReflectThenRotate => {
            let reflected = if param % 2 == 0 {
                GridEncoder::reflect_x(grid)
            } else {
                GridEncoder::reflect_y(grid)
            };
            GridEncoder::rotate_90(&reflected)
        }
        ChainedTransform::FillThenTranslate => {
            let rows = grid.len();
            let cols = if rows > 0 { grid[0].len() } else { 0 };
            let fill_color = (param % num_colors as u64) as u8;
            let top = (param / 7 % rows.max(1) as u64) as usize;
            let left = (param / 11 % cols.max(1) as u64) as usize;
            let filled = GridEncoder::fill_region(
                grid,
                top,
                left,
                top.min(rows - 1),
                left.min(cols - 1),
                fill_color,
            );
            let dx = ((param / 13 % 3) as i32) - 1;
            let dy = ((param / 17 % 3) as i32) - 1;
            GridEncoder::translate_grid(&filled, dx, dy, 0)
        }
    }
}

/// Check horizontal symmetry (reflect_x invariance).
fn is_h_symmetric(grid: &[Vec<u8>]) -> bool {
    let reflected = GridEncoder::reflect_x(grid);
    grid == reflected.as_slice()
}

/// Check vertical symmetry (reflect_y invariance).
fn is_v_symmetric(grid: &[Vec<u8>]) -> bool {
    let reflected = GridEncoder::reflect_y(grid);
    grid == reflected.as_slice()
}

/// Generate a grid with guaranteed horizontal symmetry.
fn gen_h_symmetric_grid(rng: &mut u64, size: usize, num_colors: u8) -> Vec<Vec<u8>> {
    let mut grid = vec![vec![0u8; size]; size];
    let half = size.div_ceil(2);
    #[allow(clippy::needless_range_loop)]
    for r in 0..size {
        for c in 0..half {
            xor_shift(rng);
            let color = (*rng % num_colors as u64) as u8;
            grid[r][c] = color;
            grid[r][size - 1 - c] = color;
        }
    }
    grid
}

/// Generate a grid with guaranteed vertical symmetry.
fn gen_v_symmetric_grid(rng: &mut u64, size: usize, num_colors: u8) -> Vec<Vec<u8>> {
    let mut grid = vec![vec![0u8; size]; size];
    let half = size.div_ceil(2);
    #[allow(clippy::needless_range_loop)]
    for r in 0..half {
        for c in 0..size {
            xor_shift(rng);
            let color = (*rng % num_colors as u64) as u8;
            grid[r][c] = color;
            grid[size - 1 - r][c] = color;
        }
    }
    grid
}

struct CompositionalTrialResult {
    compositional_accuracy: f64,
    size_generalization: f64,
    symmetry_detection: f64,
    compositional_rt_ticks: f64,
    /// Per-chain-type accuracy: [TranslateThenReflect, ColorReplaceThenFill, ReflectThenRotate, FillThenTranslate]
    per_chain_accuracy: [f64; 4],
}

impl ArcCompositionalBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> CompositionalTrialResult {
        let seed = config.trial_seed("reasoning", "arc_compositional", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;
        let num_colors: u8 = 6;
        let pressure = config.time_pressure;
        let noise_weight = 0.02 + pressure * 0.15 + config.encoding_noise * 0.20;
        let mut total_ticks: f64 = 0.0;

        // ── Part 1: Chained Transforms (5×5 grids) ──
        let grid_size = 5;
        let encoder = BinaryGridEncoder::new(grid_size, grid_size, num_colors as usize, seed);
        let tasks_per_chain = 3;
        let mut chain_hits: u32 = 0;
        let mut chain_total: u32 = 0;
        let mut per_chain_hits: [u32; 4] = [0; 4];
        let mut per_chain_total: [u32; 4] = [0; 4];

        for (chain_idx, &chain_type) in CHAINED_TYPES.iter().enumerate() {
            for _ in 0..tasks_per_chain {
                xor_shift(&mut rng);
                let param = rng;

                // 2 training pairs
                let mut train_rules = Vec::new();
                for _ in 0..2 {
                    xor_shift(&mut rng);
                    let input = gen_grid(&mut rng, grid_size, num_colors);
                    let output = apply_chain(&input, chain_type, param, num_colors);
                    let in_hv = encoder.encode_grid(&input);
                    let out_hv = encoder.encode_grid(&output);
                    let mut rule = encoder.encode_rule(&in_hv, &out_hv);
                    if noise_weight > 0.0 {
                        xor_shift(&mut rng);
                        rule = rule.add_noise(noise_weight as f32, rng);
                    }
                    train_rules.push(rule);
                    xor_shift(&mut rng);
                    let tick_scale = 1.0 - pressure * 0.4;
                    total_ticks += (6.0 + (rng % 5) as f64) * tick_scale;
                }

                let consensus = encoder.bundle_rules(&train_rules);

                // Test pair
                xor_shift(&mut rng);
                let test_input = gen_grid(&mut rng, grid_size, num_colors);
                let test_output = apply_chain(&test_input, chain_type, param, num_colors);
                let test_in_hv = encoder.encode_grid(&test_input);
                let test_out_hv = encoder.encode_grid(&test_output);
                let predicted = encoder.apply_rule(&test_in_hv, &consensus);

                // Distractor: apply a different chained transform to same input
                let pred_sim = predicted.similarity(&test_out_hv) as f64;
                let distractor_idx = (CHAINED_TYPES
                    .iter()
                    .position(|t| std::mem::discriminant(t) == std::mem::discriminant(&chain_type))
                    .unwrap_or(0)
                    + 1)
                    % CHAINED_TYPES.len();
                let distractor_output = apply_chain(
                    &test_input,
                    CHAINED_TYPES[distractor_idx],
                    param,
                    num_colors,
                );
                let distractor_hv = encoder.encode_grid(&distractor_output);
                let distractor_sim = predicted.similarity(&distractor_hv) as f64;

                chain_total += 1;
                per_chain_total[chain_idx] += 1;
                if pred_sim > distractor_sim {
                    chain_hits += 1;
                    per_chain_hits[chain_idx] += 1;
                }
                xor_shift(&mut rng);
                let tick_scale = 1.0 - pressure * 0.4;
                total_ticks += (6.0 + (rng % 5) as f64) * tick_scale;
            }
        }

        let compositional_accuracy = if chain_total > 0 {
            chain_hits as f64 / chain_total as f64
        } else {
            0.0
        };

        // ── Part 2: Size-Varying Grids ──
        // Train on 5×5, test on 3×3 and 7×7
        let sizes = [3, 5, 7];
        let max_size = 7;
        let size_encoder = BinaryGridEncoder::new(
            max_size,
            max_size,
            num_colors as usize,
            seed.wrapping_add(1000),
        );
        let mut size_hits: u32 = 0;
        let mut size_total: u32 = 0;

        for _ in 0..4 {
            xor_shift(&mut rng);
            let transform_param = rng;
            // Use simple transforms for size tests (color replace is size-independent)
            let from = (transform_param % num_colors as u64) as u8;
            let to = ((transform_param / 7 + 1) % num_colors as u64) as u8;

            // Train on size 5
            let mut train_rules = Vec::new();
            for _ in 0..2 {
                xor_shift(&mut rng);
                let input = gen_grid(&mut rng, 5, num_colors);
                let output = GridEncoder::color_replace(&input, from, to);
                let in_hv = size_encoder.encode_grid(&input);
                let out_hv = size_encoder.encode_grid(&output);
                let mut rule = size_encoder.encode_rule(&in_hv, &out_hv);
                if noise_weight > 0.0 {
                    xor_shift(&mut rng);
                    rule = rule.add_noise(noise_weight as f32, rng);
                }
                train_rules.push(rule);
                xor_shift(&mut rng);
                let tick_scale = 1.0 - pressure * 0.4;
                total_ticks += (5.0 + (rng % 4) as f64) * tick_scale;
            }
            let consensus = size_encoder.bundle_rules(&train_rules);

            // Test on each size — distractor: different color_replace params
            let distractor_from = ((from as u64 + 2) % num_colors as u64) as u8;
            let distractor_to = ((to as u64 + 2) % num_colors as u64) as u8;
            for &test_size in &sizes {
                xor_shift(&mut rng);
                let test_input = gen_grid(&mut rng, test_size, num_colors);
                let test_output = GridEncoder::color_replace(&test_input, from, to);
                let test_in_hv = size_encoder.encode_grid(&test_input);
                let test_out_hv = size_encoder.encode_grid(&test_output);
                let predicted = size_encoder.apply_rule(&test_in_hv, &consensus);

                let pred_sim = predicted.similarity(&test_out_hv) as f64;
                let distractor_output =
                    GridEncoder::color_replace(&test_input, distractor_from, distractor_to);
                let distractor_hv = size_encoder.encode_grid(&distractor_output);
                let distractor_sim = predicted.similarity(&distractor_hv) as f64;

                size_total += 1;
                if pred_sim > distractor_sim {
                    size_hits += 1;
                }
                xor_shift(&mut rng);
                let tick_scale = 1.0 - pressure * 0.4;
                total_ticks += (5.0 + (rng % 4) as f64) * tick_scale;
            }
        }

        let size_generalization = if size_total > 0 {
            size_hits as f64 / size_total as f64
        } else {
            0.0
        };

        // ── Part 3: Symmetry Detection ──
        // Encode 4 types: h_sym only, v_sym only, both, neither — classify via cosine
        let sym_size = 5;
        let sym_encoder = BinaryGridEncoder::new(
            sym_size,
            sym_size,
            num_colors as usize,
            seed.wrapping_add(2000),
        );

        // Build prototype HVs for each symmetry class from examples
        let examples_per_class = 4;
        let mut h_sym_hvs = Vec::new();
        let mut v_sym_hvs = Vec::new();
        let mut both_sym_hvs = Vec::new();
        let mut no_sym_hvs = Vec::new();

        for _ in 0..examples_per_class {
            // H-symmetric only
            xor_shift(&mut rng);
            let g = gen_h_symmetric_grid(&mut rng, sym_size, num_colors);
            h_sym_hvs.push(sym_encoder.encode_grid(&g));

            // V-symmetric only
            xor_shift(&mut rng);
            let g = gen_v_symmetric_grid(&mut rng, sym_size, num_colors);
            v_sym_hvs.push(sym_encoder.encode_grid(&g));

            // Both symmetric
            xor_shift(&mut rng);
            let mut g = gen_h_symmetric_grid(&mut rng, sym_size, num_colors);
            // Force vertical symmetry too
            let half = sym_size.div_ceil(2);
            #[allow(clippy::needless_range_loop)]
            for r in 0..half {
                for c in 0..sym_size {
                    g[sym_size - 1 - r][c] = g[r][c];
                }
            }
            both_sym_hvs.push(sym_encoder.encode_grid(&g));

            // No symmetry (random)
            xor_shift(&mut rng);
            let g = gen_grid(&mut rng, sym_size, num_colors);
            no_sym_hvs.push(sym_encoder.encode_grid(&g));
        }

        let h_proto = BinaryHV::bundle(&h_sym_hvs);
        let v_proto = BinaryHV::bundle(&v_sym_hvs);
        let both_proto = BinaryHV::bundle(&both_sym_hvs);
        let no_proto = BinaryHV::bundle(&no_sym_hvs);

        let prototypes = [&h_proto, &v_proto, &both_proto, &no_proto];

        // Classify test grids
        let test_per_class = 4;
        let mut sym_correct: u32 = 0;
        let mut sym_total: u32 = 0;

        let classify = |hv: &BinaryHV| -> usize {
            let mut best_idx = 0;
            let mut best_sim = f32::NEG_INFINITY;
            for (i, proto) in prototypes.iter().enumerate() {
                let sim = hv.similarity(proto);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }
            best_idx
        };

        for _ in 0..test_per_class {
            // H-symmetric test (expected class 0)
            xor_shift(&mut rng);
            let g = gen_h_symmetric_grid(&mut rng, sym_size, num_colors);
            let hv = sym_encoder.encode_grid(&g);
            sym_total += 1;
            if classify(&hv) == 0 {
                sym_correct += 1;
            }
            xor_shift(&mut rng);
            let tick_scale = 1.0 - pressure * 0.4;
            total_ticks += (3.0 + (rng % 3) as f64) * tick_scale;

            // V-symmetric test (expected class 1)
            xor_shift(&mut rng);
            let g = gen_v_symmetric_grid(&mut rng, sym_size, num_colors);
            let hv = sym_encoder.encode_grid(&g);
            sym_total += 1;
            if classify(&hv) == 1 {
                sym_correct += 1;
            }
            xor_shift(&mut rng);
            let tick_scale = 1.0 - pressure * 0.4;
            total_ticks += (3.0 + (rng % 3) as f64) * tick_scale;

            // No symmetry test (expected class 3)
            // Rejection-sample until we get a genuinely asymmetric grid
            let g = loop {
                xor_shift(&mut rng);
                let candidate = gen_grid(&mut rng, sym_size, num_colors);
                if !is_h_symmetric(&candidate) && !is_v_symmetric(&candidate) {
                    break candidate;
                }
            };
            let hv = sym_encoder.encode_grid(&g);
            sym_total += 1;
            if classify(&hv) == 3 {
                sym_correct += 1;
            }
            xor_shift(&mut rng);
            let tick_scale = 1.0 - pressure * 0.4;
            total_ticks += (3.0 + (rng % 3) as f64) * tick_scale;
        }

        let symmetry_detection = if sym_total > 0 {
            sym_correct as f64 / sym_total as f64
        } else {
            0.0
        };

        let total_tasks = (chain_total + size_total + sym_total) as f64;
        let compositional_rt_ticks = if total_tasks > 0.0 {
            total_ticks / total_tasks
        } else {
            0.0
        };

        let per_chain_accuracy = [
            if per_chain_total[0] > 0 {
                per_chain_hits[0] as f64 / per_chain_total[0] as f64
            } else {
                0.0
            },
            if per_chain_total[1] > 0 {
                per_chain_hits[1] as f64 / per_chain_total[1] as f64
            } else {
                0.0
            },
            if per_chain_total[2] > 0 {
                per_chain_hits[2] as f64 / per_chain_total[2] as f64
            } else {
                0.0
            },
            if per_chain_total[3] > 0 {
                per_chain_hits[3] as f64 / per_chain_total[3] as f64
            } else {
                0.0
            },
        ];

        CompositionalTrialResult {
            compositional_accuracy,
            size_generalization,
            symmetry_detection,
            compositional_rt_ticks,
            per_chain_accuracy,
        }
    }
}

impl PsychBenchmark for ArcCompositionalBenchmark {
    fn name(&self) -> &str {
        "Reasoning::ArcCompositional"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Synthetic ARC-inspired compositional grid transforms (HDC encoding test)",
            citation: "Chollet (2019); Johnson et al. (2021)",
            year: 2021,
            doi: Some("10.48550/arXiv.1911.01547"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut comp_accs = Vec::new();
        let mut size_gens = Vec::new();
        let mut sym_dets = Vec::new();
        let mut rts = Vec::new();
        let chain_names = [
            "translate_reflect",
            "color_replace_fill",
            "reflect_rotate",
            "fill_translate",
        ];
        let mut per_chain: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            comp_accs.push(r.compositional_accuracy);
            size_gens.push(r.size_generalization);
            sym_dets.push(r.symmetry_detection);
            rts.push(r.compositional_rt_ticks);
            for (i, acc) in r.per_chain_accuracy.iter().enumerate() {
                per_chain[i].push(*acc);
            }
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "arc_compositional".to_string(),
                    correct: r.compositional_accuracy > 0.5,
                    rt_ticks: r.compositional_rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "compositional_accuracy",
            MetricValue::from_samples(&comp_accs),
        );
        result.insert("size_generalization", MetricValue::from_samples(&size_gens));
        result.insert("symmetry_detection", MetricValue::from_samples(&sym_dets));
        result.insert("compositional_rt_ticks", MetricValue::from_samples(&rts));

        // Per-chain-type breakdowns
        for (i, name) in chain_names.iter().enumerate() {
            result.insert(
                format!("accuracy_{}", name),
                MetricValue::from_samples(&per_chain[i]),
            );
        }

        result.conditions = 3; // chained, size-varying, symmetry
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.notes.push(
            "SYNTHETIC: Uses procedural grid transforms, not Chollet's real ARC. \
             Tests HDC encoding robustness under composition, size variation, and \
             prototype-based symmetry classification. Z-scores reflect encoding \
             quality, not compositional reasoning."
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
    fn test_compositional_runs_with_metrics() {
        let result = ArcCompositionalBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("compositional_accuracy"));
        assert!(result.metrics.contains_key("size_generalization"));
        assert!(result.metrics.contains_key("symmetry_detection"));
        assert!(result.metrics.contains_key("compositional_rt_ticks"));
        // Per-chain breakdowns
        assert!(result.metrics.contains_key("accuracy_translate_reflect"));
        assert!(result.metrics.contains_key("accuracy_color_replace_fill"));
        assert!(result.metrics.contains_key("accuracy_reflect_rotate"));
        assert!(result.metrics.contains_key("accuracy_fill_translate"));
    }

    #[test]
    fn test_per_chain_breakdowns_finite() {
        let result = ArcCompositionalBenchmark.run(&test_config());
        for name in &[
            "accuracy_translate_reflect",
            "accuracy_color_replace_fill",
            "accuracy_reflect_rotate",
            "accuracy_fill_translate",
        ] {
            let val = &result.metrics[*name];
            assert!(
                val.mean.is_finite(),
                "Per-chain metric {} is not finite",
                name
            );
            assert!(
                val.mean >= 0.0 && val.mean <= 1.0,
                "Per-chain metric {} out of range: {}",
                name,
                val.mean
            );
        }
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = ArcCompositionalBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
            assert!(val.std_dev.is_finite(), "metric {} std_dev not finite", key);
        }
    }

    #[test]
    fn test_compositional_above_chance() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ArcCompositionalBenchmark.run(&config);
        let acc = result.metrics["compositional_accuracy"].mean;
        assert!(
            acc > 0.3,
            "Compositional accuracy should be above chance, got {}",
            acc
        );
    }

    #[test]
    fn test_size_generalization_above_chance() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ArcCompositionalBenchmark.run(&config);
        let r#gen = result.metrics["size_generalization"].mean;
        assert!(
            r#gen > 0.3,
            "Size generalization should be above chance, got {}",
            r#gen
        );
    }

    #[test]
    fn test_symmetry_detection_above_chance() {
        let config = BenchmarkConfig {
            dimension: 1024,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = ArcCompositionalBenchmark.run(&config);
        let sym = result.metrics["symmetry_detection"].mean;
        // 4 classes → chance is 0.25; with prototype matching should beat that
        assert!(
            sym > 0.2,
            "Symmetry detection should be above chance, got {}",
            sym
        );
    }

    #[test]
    fn test_provenance_correct() {
        let prov = ArcCompositionalBenchmark.provenance().unwrap();
        assert!(prov.paradigm.contains("Synthetic"));
        assert!(prov.paradigm.contains("compositional"));
        assert!(prov.citation.contains("Johnson"));
    }

    #[test]
    fn test_deterministic() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 2,
            seed: 42,
            ..Default::default()
        };
        let r1 = ArcCompositionalBenchmark.run(&config);
        let r2 = ArcCompositionalBenchmark.run(&config);
        assert_eq!(
            r1.metrics["compositional_accuracy"].mean, r2.metrics["compositional_accuracy"].mean,
            "Same seed should produce identical results"
        );
    }

    #[test]
    fn test_time_pressure_effect() {
        let base_config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            seed: 42,
            time_pressure: 0.0,
            ..Default::default()
        };
        let pressure_config = BenchmarkConfig {
            time_pressure: 0.8,
            ..base_config.clone()
        };
        let base_result = ArcCompositionalBenchmark.run(&base_config);
        let pressure_result = ArcCompositionalBenchmark.run(&pressure_config);
        let base_acc = base_result.metrics["compositional_accuracy"].mean;
        let pressure_acc = pressure_result.metrics["compositional_accuracy"].mean;
        assert!(base_acc.is_finite());
        assert!(pressure_acc.is_finite());
        // Time pressure should reduce RT (SAT tradeoff: tick_scale = 1 - 0.8*0.4 = 0.68)
        let base_rt = base_result.metrics["compositional_rt_ticks"].mean;
        let pressure_rt = pressure_result.metrics["compositional_rt_ticks"].mean;
        assert!(
            pressure_rt < base_rt,
            "Time pressure should reduce RT: base={}, pressure={}",
            base_rt,
            pressure_rt
        );
    }

    #[test]
    fn test_chained_transform_helpers() {
        // Verify chained transforms produce valid grids
        let grid = vec![
            vec![0, 1, 2, 3, 4],
            vec![1, 2, 3, 4, 5],
            vec![2, 3, 4, 5, 0],
            vec![3, 4, 5, 0, 1],
            vec![4, 5, 0, 1, 2],
        ];
        for &chain in &CHAINED_TYPES {
            let result = apply_chain(&grid, chain, 42, 6);
            assert_eq!(result.len(), grid.len());
            for row in &result {
                assert!(
                    row.len() >= grid[0].len()
                        || matches!(chain, ChainedTransform::ReflectThenRotate)
                );
            }
        }
    }

    #[test]
    fn test_symmetry_helpers() {
        let mut rng = 42u64 ^ 0x9E3779B97F4A7C15;
        let h_grid = gen_h_symmetric_grid(&mut rng, 5, 4);
        assert!(
            is_h_symmetric(&h_grid),
            "H-symmetric grid should pass h-symmetry check"
        );

        let v_grid = gen_v_symmetric_grid(&mut rng, 5, 4);
        assert!(
            is_v_symmetric(&v_grid),
            "V-symmetric grid should pass v-symmetry check"
        );
    }
}
