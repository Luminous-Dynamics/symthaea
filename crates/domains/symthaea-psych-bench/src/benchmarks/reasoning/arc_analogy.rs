// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ARC Analogy benchmark (SYNTHETIC, not real ARC).
//!
//! **Honesty note:** This benchmark uses procedurally generated grid tasks.
//! It tests HDC XOR-based rule transfer — whether a rule HV extracted from
//! one input/output pair can be applied to a different input. Because XOR
//! bind is self-inverse, "analogical transfer" is exact algebraic recovery,
//! not genuine analogical reasoning. The z-scores reflect encoding fidelity
//! across different inputs, not abstract analogical ability.
//!
//! Tests analogical reasoning: given A→B via transform T, infer C→D.
//! This is the classic A:B :: C:? paradigm (Cattell/Horn fluid intelligence).
//!
//! Three analogy types of increasing difficulty:
//!   1. **Single-transform analogy**: One transform, same grid size.
//!   2. **Cross-domain analogy**: Transform learned on one color palette,
//!      applied to a different palette.
//!   3. **Multi-example analogy**: 3 source pairs → consensus rule → novel input.
//!
//! Human baselines (estimated from Chollet 2019; Lovett & Forbus 2017):
//! - analogy_accuracy: ~0.75 (SD~0.12) — single transform analogies
//! - cross_domain_accuracy: ~0.60 (SD~0.15) — cross-palette transfer
//! - multi_example_accuracy: ~0.70 (SD~0.13) — 3-pair consensus analogies
//! - analogy_rt_ticks: ~5.0 (SD~2.0) — deliberation proxy

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;
use symthaea_core::hdc::grid_encoder::GridEncoder;

/// ARC-style analogical reasoning benchmark.
pub struct ArcAnalogyBenchmark;

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

/// Available single transforms for analogy pairs.
#[derive(Debug, Clone, Copy)]
enum AnalogyTransform {
    ColorReplace,
    Translate,
    ReflectX,
    ReflectY,
    Rotate90,
    FillRegion,
}

const ANALOGY_TRANSFORMS: [AnalogyTransform; 6] = [
    AnalogyTransform::ColorReplace,
    AnalogyTransform::Translate,
    AnalogyTransform::ReflectX,
    AnalogyTransform::ReflectY,
    AnalogyTransform::Rotate90,
    AnalogyTransform::FillRegion,
];

fn apply_analogy_transform(
    grid: &[Vec<u8>],
    transform: AnalogyTransform,
    param: u64,
    num_colors: u8,
) -> Vec<Vec<u8>> {
    match transform {
        AnalogyTransform::ColorReplace => {
            let from = (param % num_colors as u64) as u8;
            let to = ((param / 7 + 1) % num_colors as u64) as u8;
            GridEncoder::color_replace(grid, from, to)
        }
        AnalogyTransform::Translate => {
            let dx = ((param % 3) as i32) - 1;
            let dy = ((param / 3 % 3) as i32) - 1;
            let fill = (param / 9 % num_colors as u64) as u8;
            GridEncoder::translate_grid(grid, dx, dy, fill)
        }
        AnalogyTransform::ReflectX => GridEncoder::reflect_x(grid),
        AnalogyTransform::ReflectY => GridEncoder::reflect_y(grid),
        AnalogyTransform::Rotate90 => GridEncoder::rotate_90(grid),
        AnalogyTransform::FillRegion => {
            let rows = grid.len();
            let cols = if rows > 0 { grid[0].len() } else { 0 };
            let color = (param % num_colors as u64) as u8;
            let top = (param / 7 % rows.max(1) as u64) as usize;
            let left = (param / 11 % cols.max(1) as u64) as usize;
            GridEncoder::fill_region(
                grid,
                top,
                left,
                top.min(rows - 1),
                left.min(cols - 1),
                color,
            )
        }
    }
}

struct AnalogyTrialResult {
    analogy_accuracy: f64,
    cross_domain_accuracy: f64,
    multi_example_accuracy: f64,
    analogy_rt_ticks: f64,
}

impl ArcAnalogyBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> AnalogyTrialResult {
        let seed = config.trial_seed("reasoning", "arc_analogy", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;
        let grid_size = 5;
        let num_colors: u8 = 6;
        let pressure = config.time_pressure;
        let noise_weight = 0.04 + pressure * 0.10;
        let tick_scale = 1.0 - pressure * 0.4;
        let mut total_ticks: f64 = 0.0;
        let mut total_tasks: u32 = 0;

        let encoder = BinaryGridEncoder::new(grid_size, grid_size, num_colors as usize, seed);

        // ── Part 1: Single-Transform Analogy (A:B :: C:?) ──
        // For each transform type: generate A→B pair, infer rule, apply to novel C
        let mut analogy_hits: u32 = 0;
        let mut analogy_total: u32 = 0;

        for (t_idx, &transform) in ANALOGY_TRANSFORMS.iter().enumerate() {
            for _ in 0..3 {
                xor_shift(&mut rng);
                let param = rng;

                // Source pair: A → B
                xor_shift(&mut rng);
                let a_grid = gen_grid(&mut rng, grid_size, num_colors);
                let b_grid = apply_analogy_transform(&a_grid, transform, param, num_colors);
                let a_hv = encoder.encode_grid(&a_grid);
                let b_hv = encoder.encode_grid(&b_grid);
                let mut rule = encoder.encode_rule(&a_hv, &b_hv);

                if noise_weight > 0.0 {
                    xor_shift(&mut rng);
                    rule = rule.add_noise(noise_weight as f32, rng);
                }

                // Target: C → ? (predict D)
                xor_shift(&mut rng);
                let c_grid = gen_grid(&mut rng, grid_size, num_colors);
                let d_grid = apply_analogy_transform(&c_grid, transform, param, num_colors);
                let c_hv = encoder.encode_grid(&c_grid);
                let d_hv = encoder.encode_grid(&d_grid);
                let predicted = encoder.apply_rule(&c_hv, &rule);

                // 2-AFC: predicted vs distractor (wrong transform on C)
                let pred_sim = predicted.similarity(&d_hv) as f64;
                let dist_transform = ANALOGY_TRANSFORMS[(t_idx + 1) % ANALOGY_TRANSFORMS.len()];
                let dist_grid = apply_analogy_transform(&c_grid, dist_transform, param, num_colors);
                let dist_hv = encoder.encode_grid(&dist_grid);
                let dist_sim = predicted.similarity(&dist_hv) as f64;

                analogy_total += 1;
                if pred_sim > dist_sim {
                    analogy_hits += 1;
                }

                xor_shift(&mut rng);
                total_ticks += (5.0 + (rng % 4) as f64) * tick_scale;
                total_tasks += 1;
            }
        }

        let analogy_accuracy = if analogy_total > 0 {
            analogy_hits as f64 / analogy_total as f64
        } else {
            0.0
        };

        // ── Part 2: Cross-Domain Analogy ──
        // Learn rule from one color palette, apply to different palette
        let mut cross_hits: u32 = 0;
        let mut cross_total: u32 = 0;

        for _ in 0..6 {
            xor_shift(&mut rng);
            let t_idx = (rng % ANALOGY_TRANSFORMS.len() as u64) as usize;
            let transform = ANALOGY_TRANSFORMS[t_idx];
            xor_shift(&mut rng);
            let param = rng;

            // Source pair with palette A (colors 0-2)
            xor_shift(&mut rng);
            let a_grid = gen_grid(&mut rng, grid_size, 3); // only 3 colors
            let b_grid = apply_analogy_transform(&a_grid, transform, param, 3);
            let a_hv = encoder.encode_grid(&a_grid);
            let b_hv = encoder.encode_grid(&b_grid);
            let mut rule = encoder.encode_rule(&a_hv, &b_hv);

            if noise_weight > 0.0 {
                xor_shift(&mut rng);
                rule = rule.add_noise(noise_weight as f32, rng);
            }

            // Target with palette B (colors 3-5) — shift grid colors
            xor_shift(&mut rng);
            let mut c_grid = gen_grid(&mut rng, grid_size, 3);
            for row in c_grid.iter_mut() {
                for cell in row.iter_mut() {
                    *cell += 3; // shift to colors 3-5
                }
            }
            let mut d_grid = apply_analogy_transform(&c_grid, transform, param, num_colors);
            // Ensure fill colors are also shifted for translation
            for row in d_grid.iter_mut() {
                for cell in row.iter_mut() {
                    if *cell < 3 {
                        *cell += 3;
                    }
                }
            }

            let c_hv = encoder.encode_grid(&c_grid);
            let d_hv = encoder.encode_grid(&d_grid);
            let predicted = encoder.apply_rule(&c_hv, &rule);

            // 2-AFC vs wrong transform distractor
            let pred_sim = predicted.similarity(&d_hv) as f64;
            let dist_t = ANALOGY_TRANSFORMS[(t_idx + 1) % ANALOGY_TRANSFORMS.len()];
            let dist_grid = apply_analogy_transform(&c_grid, dist_t, param, num_colors);
            let dist_hv = encoder.encode_grid(&dist_grid);
            let dist_sim = predicted.similarity(&dist_hv) as f64;

            cross_total += 1;
            if pred_sim > dist_sim {
                cross_hits += 1;
            }

            xor_shift(&mut rng);
            total_ticks += (6.0 + (rng % 5) as f64) * tick_scale;
            total_tasks += 1;
        }

        let cross_domain_accuracy = if cross_total > 0 {
            cross_hits as f64 / cross_total as f64
        } else {
            0.0
        };

        // ── Part 3: Multi-Example Analogy ──
        // 3 source pairs → consensus rule → novel target
        let mut multi_hits: u32 = 0;
        let mut multi_total: u32 = 0;

        for _ in 0..6 {
            xor_shift(&mut rng);
            let t_idx = (rng % ANALOGY_TRANSFORMS.len() as u64) as usize;
            let transform = ANALOGY_TRANSFORMS[t_idx];
            xor_shift(&mut rng);
            let param = rng;

            // 3 source pairs
            let mut rules = Vec::new();
            for _ in 0..3 {
                xor_shift(&mut rng);
                let src = gen_grid(&mut rng, grid_size, num_colors);
                let dst = apply_analogy_transform(&src, transform, param, num_colors);
                let src_hv = encoder.encode_grid(&src);
                let dst_hv = encoder.encode_grid(&dst);
                let mut rule = encoder.encode_rule(&src_hv, &dst_hv);
                if noise_weight > 0.0 {
                    xor_shift(&mut rng);
                    rule = rule.add_noise(noise_weight as f32, rng);
                }
                rules.push(rule);
                xor_shift(&mut rng);
                total_ticks += (4.0 + (rng % 4) as f64) * tick_scale;
            }

            let consensus = encoder.bundle_rules(&rules);

            // Novel target
            xor_shift(&mut rng);
            let c_grid = gen_grid(&mut rng, grid_size, num_colors);
            let d_grid = apply_analogy_transform(&c_grid, transform, param, num_colors);
            let c_hv = encoder.encode_grid(&c_grid);
            let d_hv = encoder.encode_grid(&d_grid);
            let predicted = encoder.apply_rule(&c_hv, &consensus);

            // 2-AFC vs distractor
            let pred_sim = predicted.similarity(&d_hv) as f64;
            let dist_t = ANALOGY_TRANSFORMS[(t_idx + 1) % ANALOGY_TRANSFORMS.len()];
            let dist_grid = apply_analogy_transform(&c_grid, dist_t, param, num_colors);
            let dist_hv = encoder.encode_grid(&dist_grid);
            let dist_sim = predicted.similarity(&dist_hv) as f64;

            multi_total += 1;
            if pred_sim > dist_sim {
                multi_hits += 1;
            }

            xor_shift(&mut rng);
            total_ticks += (5.0 + (rng % 5) as f64) * tick_scale;
            total_tasks += 1;
        }

        let multi_example_accuracy = if multi_total > 0 {
            multi_hits as f64 / multi_total as f64
        } else {
            0.0
        };

        let analogy_rt_ticks = if total_tasks > 0 {
            total_ticks / total_tasks as f64
        } else {
            0.0
        };

        AnalogyTrialResult {
            analogy_accuracy,
            cross_domain_accuracy,
            multi_example_accuracy,
            analogy_rt_ticks,
        }
    }
}

impl PsychBenchmark for ArcAnalogyBenchmark {
    fn name(&self) -> &str {
        "Reasoning::ArcAnalogy"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Synthetic analogical transfer via HDC rule binding",
            citation: "Chollet (2019); Lovett & Forbus (2017)",
            year: 2019,
            doi: Some("10.48550/arXiv.1911.01547"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut analogy_accs = Vec::new();
        let mut cross_accs = Vec::new();
        let mut multi_accs = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            analogy_accs.push(r.analogy_accuracy);
            cross_accs.push(r.cross_domain_accuracy);
            multi_accs.push(r.multi_example_accuracy);
            rts.push(r.analogy_rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "arc_analogy".to_string(),
                    correct: r.analogy_accuracy > 0.5,
                    rt_ticks: r.analogy_rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("analogy_accuracy", MetricValue::from_samples(&analogy_accs));
        result.insert(
            "cross_domain_accuracy",
            MetricValue::from_samples(&cross_accs),
        );
        result.insert(
            "multi_example_accuracy",
            MetricValue::from_samples(&multi_accs),
        );
        result.insert("analogy_rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 3; // single, cross-domain, multi-example
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.notes.push(
            "SYNTHETIC: Uses procedural grid transforms, not Chollet's real ARC. \
             Tests HDC rule transfer (XOR bind/unbind), not genuine analogical \
             reasoning. Z-scores reflect encoding fidelity across inputs."
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
    fn test_analogy_runs_with_metrics() {
        let result = ArcAnalogyBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("analogy_accuracy"));
        assert!(result.metrics.contains_key("cross_domain_accuracy"));
        assert!(result.metrics.contains_key("multi_example_accuracy"));
        assert!(result.metrics.contains_key("analogy_rt_ticks"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = ArcAnalogyBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
            assert!(val.std_dev.is_finite(), "metric {} std_dev not finite", key);
        }
    }

    #[test]
    fn test_analogy_accuracy_above_chance() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ArcAnalogyBenchmark.run(&config);
        let acc = result.metrics["analogy_accuracy"].mean;
        assert!(
            acc > 0.4,
            "Analogy accuracy should beat distractor baseline, got {}",
            acc
        );
    }

    #[test]
    fn test_multi_example_above_chance() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ArcAnalogyBenchmark.run(&config);
        let acc = result.metrics["multi_example_accuracy"].mean;
        assert!(
            acc > 0.3,
            "Multi-example analogy should beat chance, got {}",
            acc
        );
    }

    #[test]
    fn test_provenance_correct() {
        let prov = ArcAnalogyBenchmark.provenance().unwrap();
        assert!(prov.paradigm.contains("analogical"));
        assert!(prov.citation.contains("Lovett"));
    }

    #[test]
    fn test_deterministic() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 2,
            seed: 42,
            ..Default::default()
        };
        let r1 = ArcAnalogyBenchmark.run(&config);
        let r2 = ArcAnalogyBenchmark.run(&config);
        assert_eq!(
            r1.metrics["analogy_accuracy"].mean, r2.metrics["analogy_accuracy"].mean,
            "Same seed should produce identical results"
        );
    }

    #[test]
    fn test_time_pressure_reduces_rt() {
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
        let base_result = ArcAnalogyBenchmark.run(&base_config);
        let pressure_result = ArcAnalogyBenchmark.run(&pressure_config);
        let base_rt = base_result.metrics["analogy_rt_ticks"].mean;
        let pressure_rt = pressure_result.metrics["analogy_rt_ticks"].mean;
        assert!(
            pressure_rt < base_rt,
            "Time pressure should reduce RT: base={}, pressure={}",
            base_rt,
            pressure_rt
        );
    }

    #[test]
    fn test_three_conditions() {
        let result = ArcAnalogyBenchmark.run(&test_config());
        assert_eq!(result.conditions, 3, "Should have 3 analogy conditions");
    }
}
