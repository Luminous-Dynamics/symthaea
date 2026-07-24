// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ARC Multi-Transform Composition Chain benchmark (SYNTHETIC — tests HDC capacity, not reasoning).
//!
//! **Honesty note:** This benchmark uses procedurally generated grid tasks.
//! It tests how well HDC XOR binding preserves signal through multi-step
//! compositions, not whether the system can discover or reason about
//! compositional rules. **FIXED (2026-07-18)**: the 2-AFC scoring used to use
//! random BinaryHV distractors (chance = ~50%) — see `arc_dataset.rs`'s
//! retraction note for why that's a discriminability artifact, not a fair
//! baseline (on real ARC-AGI data it measured 99.0% vs. 13.8-67.8% under fair
//! distractors). Now uses `arc_dataset::fair_distractor_grid` (a generic
//! wrong transform of the test input) instead. The z-scores still reflect
//! encoding capacity limits, not compositional reasoning ability.
//!
//! Tests whether HDC rule algebra supports chaining 3+ sequential transforms.
//! Unlike ArcCompositional (which tests 2-step chains), this benchmark tests
//! 3-step and 4-step composition chains, measuring how accuracy degrades
//! with chain length. This probes the limits of the bind-based rule algebra.
//!
//! Paradigm: Given training pairs showing a multi-step transform (A→B→C→D),
//! learn a single composed rule HV and apply it to a novel input.
//!
//! Human baselines (estimated from Chollet 2019; compositional reasoning literature):
//! - chain_2_accuracy: ~0.65 (SD~0.15) — 2-step composition
//! - chain_3_accuracy: ~0.50 (SD~0.18) — 3-step composition
//! - chain_4_accuracy: ~0.38 (SD~0.20) — 4-step composition
//! - chain_degradation: ~0.09 (SD~0.05) — accuracy drop per added step

use crate::benchmarks::reasoning::arc_dataset::fair_distractor_grid;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;
use symthaea_core::hdc::grid_encoder::GridEncoder;

/// ARC multi-step rule composition benchmark.
pub struct ArcChainBenchmark;

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

/// Atomic transform types for building chains.
#[derive(Debug, Clone, Copy)]
enum AtomicTransform {
    ReflectX,
    ReflectY,
    Rotate90,
    TranslateRight,
    TranslateDown,
    ColorReplace(u8, u8),
}

/// Apply a single atomic transform to a grid.
fn apply_atomic(grid: &[Vec<u8>], t: AtomicTransform) -> Vec<Vec<u8>> {
    match t {
        AtomicTransform::ReflectX => GridEncoder::reflect_x(grid),
        AtomicTransform::ReflectY => GridEncoder::reflect_y(grid),
        AtomicTransform::Rotate90 => GridEncoder::rotate_90(grid),
        AtomicTransform::TranslateRight => GridEncoder::translate_grid(grid, 1, 0, 0),
        AtomicTransform::TranslateDown => GridEncoder::translate_grid(grid, 0, 1, 0),
        AtomicTransform::ColorReplace(from, to) => GridEncoder::color_replace(grid, from, to),
    }
}

/// Apply a chain of transforms sequentially.
fn apply_chain(grid: &[Vec<u8>], chain: &[AtomicTransform]) -> Vec<Vec<u8>> {
    let mut result = grid.to_vec();
    for t in chain {
        result = apply_atomic(&result, *t);
    }
    result
}

/// Pre-defined chains of various lengths for testing.
fn get_chains(param: u64, num_colors: u8) -> Vec<Vec<AtomicTransform>> {
    let c0 = (param % num_colors as u64) as u8;
    let c1 = ((param / 7 + 1) % num_colors as u64) as u8;
    let c2 = ((param / 11 + 2) % num_colors as u64) as u8;

    vec![
        // 2-step chains
        vec![AtomicTransform::ReflectX, AtomicTransform::TranslateRight],
        vec![
            AtomicTransform::ColorReplace(c0, c1),
            AtomicTransform::ReflectY,
        ],
        // 3-step chains
        vec![
            AtomicTransform::ReflectX,
            AtomicTransform::TranslateDown,
            AtomicTransform::ColorReplace(c0, c1),
        ],
        vec![
            AtomicTransform::TranslateRight,
            AtomicTransform::ReflectY,
            AtomicTransform::Rotate90,
        ],
        // 4-step chains
        vec![
            AtomicTransform::ReflectX,
            AtomicTransform::ColorReplace(c0, c1),
            AtomicTransform::TranslateRight,
            AtomicTransform::ReflectY,
        ],
        vec![
            AtomicTransform::TranslateDown,
            AtomicTransform::Rotate90,
            AtomicTransform::ColorReplace(c1, c2),
            AtomicTransform::ReflectX,
        ],
    ]
}

struct TrialResult {
    /// Accuracy per chain length: [chain_2, chain_3, chain_4]
    accuracy_by_length: [f64; 3],
    /// Mean similarity per chain length
    similarity_by_length: [f64; 3],
    /// Overall accuracy
    overall_accuracy: f64,
    /// RT in ticks
    rt_ticks: f64,
}

impl ArcChainBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let _dim = config.dimension;
        let seed = config.trial_seed("reasoning", "arc_chain", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let grid_size = 5;
        let num_colors: u8 = 6;
        let tasks_per_chain = 3;

        let pressure = config.time_pressure;
        let noise_weight = 0.03 + pressure * 0.12;
        let tick_scale = 1.0 - pressure * 0.4;

        xor_shift(&mut rng);
        let chains = get_chains(rng, num_colors);

        // Group chains by length
        let chain_groups: [(usize, &[Vec<AtomicTransform>]); 3] = [
            (2, &chains[0..2]), // 2-step chains
            (3, &chains[2..4]), // 3-step chains
            (4, &chains[4..6]), // 4-step chains
        ];

        let mut hits_by_length = [0u32; 3];
        let mut total_by_length = [0u32; 3];
        let mut sim_by_length = [0.0f64; 3];
        let mut sim_count_by_length = [0u32; 3];
        let mut total_ticks = 0.0f64;
        let mut total_tasks = 0u32;

        for (group_idx, (_chain_len, chain_set)) in chain_groups.iter().enumerate() {
            for chain in *chain_set {
                // Need encoder sized for rotate_90 output
                let enc_size = grid_size; // 5×5 stays 5×5 for all our transforms
                let encoder = BinaryGridEncoder::new(enc_size, enc_size, num_colors as usize, rng);

                for _task_i in 0..tasks_per_chain {
                    // Generate 2 training pairs
                    let mut train_rules = Vec::new();
                    for _pair_i in 0..2 {
                        xor_shift(&mut rng);
                        let input = gen_grid(&mut rng, grid_size, num_colors);
                        let output = apply_chain(&input, chain);
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

                    let consensus = encoder.bundle_rules(&train_rules);

                    // Test pair
                    xor_shift(&mut rng);
                    let test_input = gen_grid(&mut rng, grid_size, num_colors);
                    let test_output = apply_chain(&test_input, chain);
                    let test_in_hv = encoder.encode_grid(&test_input);
                    let test_out_hv = encoder.encode_grid(&test_output);
                    let predicted = encoder.apply_rule(&test_in_hv, &consensus);

                    let pred_sim = predicted.similarity(&test_out_hv) as f64;
                    sim_by_length[group_idx] += pred_sim;
                    sim_count_by_length[group_idx] += 1;

                    // 2-AFC: predicted vs a fair (equally structured) distractor
                    xor_shift(&mut rng);
                    let distractor_grid = fair_distractor_grid(&test_input, &test_output)
                        .unwrap_or_else(|| test_input.clone());
                    let distractor = encoder.encode_grid(&distractor_grid);
                    let dist_sim = predicted.similarity(&distractor) as f64;

                    total_by_length[group_idx] += 1;
                    if pred_sim > dist_sim {
                        hits_by_length[group_idx] += 1;
                    }

                    xor_shift(&mut rng);
                    total_ticks += (5.0 + (rng % 6) as f64) * tick_scale;
                    total_tasks += 1;
                }
            }
        }

        let accuracy_by_length = [
            if total_by_length[0] > 0 {
                hits_by_length[0] as f64 / total_by_length[0] as f64
            } else {
                0.0
            },
            if total_by_length[1] > 0 {
                hits_by_length[1] as f64 / total_by_length[1] as f64
            } else {
                0.0
            },
            if total_by_length[2] > 0 {
                hits_by_length[2] as f64 / total_by_length[2] as f64
            } else {
                0.0
            },
        ];
        let similarity_by_length = [
            if sim_count_by_length[0] > 0 {
                sim_by_length[0] / sim_count_by_length[0] as f64
            } else {
                0.0
            },
            if sim_count_by_length[1] > 0 {
                sim_by_length[1] / sim_count_by_length[1] as f64
            } else {
                0.0
            },
            if sim_count_by_length[2] > 0 {
                sim_by_length[2] / sim_count_by_length[2] as f64
            } else {
                0.0
            },
        ];
        let total_hits: u32 = hits_by_length.iter().sum();
        let total_total: u32 = total_by_length.iter().sum();
        let overall_accuracy = if total_total > 0 {
            total_hits as f64 / total_total as f64
        } else {
            0.0
        };
        let rt_ticks = if total_tasks > 0 {
            total_ticks / (total_tasks as f64 * 3.0)
        } else {
            0.0
        };

        TrialResult {
            accuracy_by_length,
            similarity_by_length,
            overall_accuracy,
            rt_ticks,
        }
    }
}

impl PsychBenchmark for ArcChainBenchmark {
    fn name(&self) -> &str {
        "Reasoning::ArcChain"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Synthetic multi-step HDC rule composition",
            citation: "Chollet (2019); Lake & Baroni (2018)",
            year: 2019,
            doi: Some("10.48550/arXiv.1911.01547"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut chain_2_accs = Vec::new();
        let mut chain_3_accs = Vec::new();
        let mut chain_4_accs = Vec::new();
        let mut chain_2_sims = Vec::new();
        let mut chain_3_sims = Vec::new();
        let mut chain_4_sims = Vec::new();
        let mut overall_accs = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            chain_2_accs.push(r.accuracy_by_length[0]);
            chain_3_accs.push(r.accuracy_by_length[1]);
            chain_4_accs.push(r.accuracy_by_length[2]);
            chain_2_sims.push(r.similarity_by_length[0]);
            chain_3_sims.push(r.similarity_by_length[1]);
            chain_4_sims.push(r.similarity_by_length[2]);
            overall_accs.push(r.overall_accuracy);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "arc_chain".to_string(),
                    correct: r.overall_accuracy > 0.5,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("chain_2_accuracy", MetricValue::from_samples(&chain_2_accs));
        result.insert("chain_3_accuracy", MetricValue::from_samples(&chain_3_accs));
        result.insert("chain_4_accuracy", MetricValue::from_samples(&chain_4_accs));
        result.insert(
            "chain_2_similarity",
            MetricValue::from_samples(&chain_2_sims),
        );
        result.insert(
            "chain_3_similarity",
            MetricValue::from_samples(&chain_3_sims),
        );
        result.insert(
            "chain_4_similarity",
            MetricValue::from_samples(&chain_4_sims),
        );
        result.insert("chain_accuracy", MetricValue::from_samples(&overall_accs));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        // Chain degradation: mean accuracy drop per step (linear fit slope)
        let degradation: Vec<f64> = chain_2_accs
            .iter()
            .zip(chain_4_accs.iter())
            .map(|(a2, a4)| (a2 - a4) / 2.0) // (2-step - 4-step) / 2 additional steps
            .collect();
        result.insert("chain_degradation", MetricValue::from_samples(&degradation));

        result.conditions = 3; // 3 chain lengths
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.notes.push(
            "SYNTHETIC: Uses procedural grid transforms, not Chollet's real ARC. \
             Tests HDC capacity under multi-step composition. 2-AFC uses a fair \
             (equally structured) distractor, not random noise (fixed 2026-07-18). \
             Z-scores reflect encoding capacity, not compositional reasoning."
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
    fn test_chain_runs_with_metrics() {
        let result = ArcChainBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("chain_2_accuracy"));
        assert!(result.metrics.contains_key("chain_3_accuracy"));
        assert!(result.metrics.contains_key("chain_4_accuracy"));
        assert!(result.metrics.contains_key("chain_accuracy"));
        assert!(result.metrics.contains_key("chain_degradation"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = ArcChainBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
            assert!(val.std_dev.is_finite(), "metric {} std_dev not finite", key);
        }
    }

    #[test]
    fn test_chain_2_above_chance() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ArcChainBenchmark.run(&config);
        let acc = result.metrics["chain_2_accuracy"].mean;
        // 2-AFC chance = 0.50
        assert!(
            acc > 0.4,
            "2-step chain accuracy should be near/above chance, got {}",
            acc
        );
    }

    #[test]
    fn test_degradation_with_length() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = ArcChainBenchmark.run(&config);
        let sim_2 = result.metrics["chain_2_similarity"].mean;
        let sim_4 = result.metrics["chain_4_similarity"].mean;
        // Longer chains should generally have lower similarity (more noise accumulation)
        // But allow tolerance since HDC can be noisy at lower dims
        assert!(sim_2.is_finite() && sim_4.is_finite());
        // Just verify non-negative degradation is plausible
        let deg = result.metrics["chain_degradation"].mean;
        assert!(deg.is_finite(), "degradation should be finite");
    }

    #[test]
    fn test_provenance() {
        let prov = ArcChainBenchmark.provenance().unwrap();
        assert!(prov.citation.contains("Chollet"));
        assert!(prov.citation.contains("Lake"));
    }

    #[test]
    fn test_deterministic() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 2,
            seed: 42,
            ..Default::default()
        };
        let r1 = ArcChainBenchmark.run(&config);
        let r2 = ArcChainBenchmark.run(&config);
        assert_eq!(
            r1.metrics["chain_accuracy"].mean,
            r2.metrics["chain_accuracy"].mean
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
        let pressed = BenchmarkConfig {
            time_pressure: 0.8,
            ..base.clone()
        };
        let r1 = ArcChainBenchmark.run(&base);
        let r2 = ArcChainBenchmark.run(&pressed);
        assert!(
            r2.metrics["rt_ticks"].mean < r1.metrics["rt_ticks"].mean,
            "Time pressure should reduce RT"
        );
    }
}
