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
//!
//! ## ⚠️ `chain_degradation` and the per-length ordering are NOT interpretable as
//! "degradation with chain length"
//!
//! Measured 2026-07-31, per chain (2-AFC, chance 0.50):
//!
//! | chain | transforms | accuracy |
//! |-------|------------|----------|
//! | c0 (len 2) | ReflectX, TranslateRight | 1.000 |
//! | c1 (len 2) | ColorReplace, ReflectY | 1.000 |
//! | c2 (len 3) | ReflectX, TranslateDown, ColorReplace | 0.700 |
//! | c3 (len 3) | TranslateRight, ReflectY, **Rotate90** | 0.533 |
//! | c4 (len 4) | ReflectX, ColorReplace, TranslateRight, ReflectY | 0.867 |
//! | c5 (len 4) | TranslateDown, **Rotate90**, ColorReplace, ReflectX | 0.567 |
//!
//! The two weakest chains are exactly the two containing `Rotate90`, independent of length,
//! and a 4-step chain (c4, 0.867) outscores both 3-step chains. So the length groups are
//! **confounded by transform difficulty**: they are not matched on anything except length, and
//! `get_chains` does not nest them (a 3-step chain is not its 2-step counterpart plus a step).
//!
//! Consequence: a per-length ordering that is not monotone is *expected* here and is not
//! evidence of a scoring defect. To actually measure degradation-with-length the chains would
//! have to be nested — chain_3 = chain_2 + 1 step, chain_4 = chain_3 + 1 step — so that length
//! is the only variable. That change would alter every recorded value and has not been made.
//! Until it is, treat `chain_degradation` as descriptive of *this specific chain set*, not as a
//! measurement of compositional depth.

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
    /// Mean similarity between the prediction and the *unchanged test input*, per chain.
    ///
    /// This is an honesty metric: a rule that fails to generalise leaves `apply_rule` close to
    /// the identity, so the prediction stays near the input. If this is high while
    /// `similarity_by_length` is low, the model is echoing the input rather than composing the
    /// rule -- and any 2-AFC whose distractor is also near the input will then be scored
    /// systematically *below* chance rather than at it.
    identity_similarity_by_chain: [f64; 6],
    /// Accuracy for each individual chain, in `get_chains` order.
    ///
    /// Length-group aggregates hide *which* chain fails, and that mattered: a 2026-07-30
    /// investigation into `chain_3` scoring below `chain_4` could not be finished because the
    /// benchmark only reported per-length means. Attribution needs per-chain resolution.
    accuracy_by_chain: [f64; 6],
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
        let mut hits_by_chain = [0u32; 6];
        let mut total_by_chain = [0u32; 6];
        let mut identity_sim_by_chain = [0.0f64; 6];
        let mut sim_by_length = [0.0f64; 3];
        let mut sim_count_by_length = [0u32; 3];
        let mut total_ticks = 0.0f64;
        let mut total_tasks = 0u32;

        for (group_idx, (_chain_len, chain_set)) in chain_groups.iter().enumerate() {
            for (within_group, chain) in chain_set.iter().enumerate() {
                // chain_groups slices `chains` as 0..2, 2..4, 4..6, so this recovers the
                // original index into `get_chains`.
                let chain_idx = group_idx * 2 + within_group;
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
                    identity_sim_by_chain[chain_idx] += predicted.similarity(&test_in_hv) as f64;
                    sim_by_length[group_idx] += pred_sim;
                    sim_count_by_length[group_idx] += 1;

                    // 2-AFC: predicted vs a distractor that is the output of a DIFFERENT chain
                    // of the SAME length, applied to the same input.
                    //
                    // Why not `fair_distractor_grid` (used here until 2026-07-31): it returns a
                    // single-transform variation of the input, so its distance from the input is
                    // CONSTANT while the true output moves further from the input with every
                    // additional chain step. Measured on this benchmark, `apply_rule` sits ~0.78
                    // similar to the input but only ~0.55 to a 3-4 step target, so the
                    // input-adjacent distractor won almost every long-chain trial. That produced
                    // scores far BELOW 2-AFC chance (chain_3 = 0.0167, i.e. 1/60) and an
                    // impossible ordering in which 3-step chains scored below 4-step ones.
                    //
                    // Using the sibling chain from the same length group makes both options
                    // equidistant from the input by construction, so the comparison isolates
                    // "did the model learn THIS rule" instead of "did the prediction move away
                    // from the input at all". Each group holds exactly 2 chains, so the sibling
                    // is the other one.
                    xor_shift(&mut rng);
                    let sibling = &chain_set[1 - within_group];
                    let sibling_output = apply_chain(&test_input, sibling);
                    let distractor_grid = if sibling_output == test_output {
                        // The two chains agreed on this input; fall back to the generic
                        // distractor rather than scoring against the correct answer.
                        fair_distractor_grid(&test_input, &test_output)
                            .unwrap_or_else(|| test_input.clone())
                    } else {
                        sibling_output
                    };
                    let distractor = encoder.encode_grid(&distractor_grid);
                    let dist_sim = predicted.similarity(&distractor) as f64;

                    total_by_length[group_idx] += 1;
                    total_by_chain[chain_idx] += 1;
                    if pred_sim > dist_sim {
                        hits_by_length[group_idx] += 1;
                        hits_by_chain[chain_idx] += 1;
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

        let mut accuracy_by_chain = [0.0f64; 6];
        let mut identity_similarity_by_chain = [0.0f64; 6];
        for i in 0..6 {
            if total_by_chain[i] > 0 {
                accuracy_by_chain[i] = hits_by_chain[i] as f64 / total_by_chain[i] as f64;
                identity_similarity_by_chain[i] =
                    identity_sim_by_chain[i] / total_by_chain[i] as f64;
            }
        }

        TrialResult {
            accuracy_by_length,
            accuracy_by_chain,
            identity_similarity_by_chain,
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
        let mut per_chain_accs: [Vec<f64>; 6] = Default::default();
        let mut per_chain_ident: [Vec<f64>; 6] = Default::default();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            chain_2_accs.push(r.accuracy_by_length[0]);
            chain_3_accs.push(r.accuracy_by_length[1]);
            chain_4_accs.push(r.accuracy_by_length[2]);
            for (i, acc) in r.accuracy_by_chain.iter().enumerate() {
                per_chain_accs[i].push(*acc);
                per_chain_ident[i].push(r.identity_similarity_by_chain[i]);
            }
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
        // Per-chain accuracy, named for its position in `get_chains`. These exist so a future
        // anomaly can be attributed to a specific transform sequence rather than only to a
        // length group -- see the `accuracy_by_chain` doc comment.
        const CHAIN_METRIC_NAMES: [&str; 6] = [
            "chain_c0_accuracy",
            "chain_c1_accuracy",
            "chain_c2_accuracy",
            "chain_c3_accuracy",
            "chain_c4_accuracy",
            "chain_c5_accuracy",
        ];
        const IDENT_METRIC_NAMES: [&str; 6] = [
            "chain_c0_identity_similarity",
            "chain_c1_identity_similarity",
            "chain_c2_identity_similarity",
            "chain_c3_identity_similarity",
            "chain_c4_identity_similarity",
            "chain_c5_identity_similarity",
        ];
        for (name, samples) in CHAIN_METRIC_NAMES.iter().zip(per_chain_accs.iter()) {
            result.insert(*name, MetricValue::from_samples(samples));
        }
        for (name, samples) in IDENT_METRIC_NAMES.iter().zip(per_chain_ident.iter()) {
            result.insert(*name, MetricValue::from_samples(samples));
        }
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

        // This test used to read chain_2/chain_4 similarity and assert only that both were
        // finite, never comparing them. It therefore passed while chain_3 accuracy sat at
        // 0.0167 against chain_4's 0.1500 — a full inversion — and is the reason that defect
        // went unnoticed. It cannot assert monotonicity instead, because the chain set is
        // confounded by transform difficulty (see this module's docs). What it CAN assert, and
        // now does, is that the derived metric is consistent with the values it is derived from.
        let acc_2 = result.metrics["chain_2_accuracy"].mean;
        let acc_4 = result.metrics["chain_4_accuracy"].mean;
        let deg = result.metrics["chain_degradation"].mean;

        // chain_degradation is defined as (chain_2 - chain_4) / 2 per trial; means are linear,
        // so the identity must hold exactly up to float error.
        let expected = (acc_2 - acc_4) / 2.0;
        assert!(
            (deg - expected).abs() < 1e-9,
            "chain_degradation ({deg:.6}) does not match its own definition \
             (chain_2 {acc_2:.6} - chain_4 {acc_4:.6}) / 2 = {expected:.6}. Either the metric \
             changed definition or the samples it aggregates are not the ones reported."
        );
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
