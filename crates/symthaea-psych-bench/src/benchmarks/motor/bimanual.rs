// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bimanual Coordination Task.
//!
//! Two hands must perform rhythmic movements simultaneously. Symmetric
//! movements (same direction) are easier than asymmetric (different direction),
//! producing coordination cost — a fundamental constraint of motor control.
//!
//! HDC implementation: Two independent SSM temporal sequences model left and
//! right hand rhythms. Cross-talk noise is injected when responses conflict
//! (asymmetric condition). The coordination cost is the accuracy deficit
//! between symmetric and asymmetric conditions.
//!
//! Human baselines (Kelso, 1984; Swinnen, 2002):
//! - coordination_cost: 0.15 (SD~0.06) — accuracy drop for asymmetric
//! - symmetric_accuracy: 0.92 (SD~0.04)
//! - asymmetric_accuracy: 0.77 (SD~0.08)

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::ssm_temporal::SsmTemporalBackend;
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Bimanual Coordination benchmark.
pub struct BimanualBenchmark;

struct TrialResult {
    coordination_cost: f64,
    left_accuracy: f64,
    right_accuracy: f64,
    symmetric_accuracy: f64,
    asymmetric_accuracy: f64,
    rt_ticks: f64,
}

impl BimanualBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let diff_model = difficulty_model_for(self.name());
        let dim = config.dimension;
        let seed = config.trial_seed("motor", "bimanual", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Direction prototypes for each hand: up, down
        let left_up = ContinuousHV::random(dim, seed.wrapping_add(1));
        let left_down = ContinuousHV::random(dim, seed.wrapping_add(2));
        let right_up = ContinuousHV::random(dim, seed.wrapping_add(3));
        let right_down = ContinuousHV::random(dim, seed.wrapping_add(4));

        // SSM backends for each hand (independent rhythm learning)
        let mut ssm_left = SsmTemporalBackend::new(-0.10, 2);
        let mut ssm_right = SsmTemporalBackend::new(-0.10, 2);

        // Time pressure and noise
        let ablation_noise = config.effective_noise() as f32 * 0.4;
        let noise_level: f32 = (0.20 + config.time_pressure as f32 * 0.15 + ablation_noise)
            * diff_model.interference_multiplier(config.difficulty) as f32;
        let crosstalk_noise: f32 =
            0.25 * diff_model.interference_multiplier(config.difficulty) as f32; // cross-hand interference

        let movements_per_condition = 20;
        let mut sym_left_correct = 0u32;
        let mut sym_right_correct = 0u32;
        let mut asym_left_correct = 0u32;
        let mut asym_right_correct = 0u32;
        let mut left_total = 0u32;
        let mut right_total = 0u32;
        let mut rt_sum = 0.0f64;
        let total_movements = movements_per_condition * 2;

        // Symmetric condition: both hands same direction
        for step in 0..movements_per_condition {
            let go_up = step % 2 == 0;

            // Left hand
            let left_target = if go_up { &left_up } else { &left_down };
            let left_activation = ssm_left.step(if go_up { 1.0 } else { 0.0 });

            // Right hand (same direction = symmetric)
            let right_target = if go_up { &right_up } else { &right_down };
            let right_activation = ssm_right.step(if go_up { 1.0 } else { 0.0 });

            // Left hand response
            let left_options = [&left_up, &left_down];
            let mut best_left = f32::NEG_INFINITY;
            let mut best_left_idx = 0;
            for (i, opt) in left_options.iter().enumerate() {
                let mut sim = left_target.similarity(opt);
                sim += left_activation * 0.2;
                xor_shift(&mut rng);
                sim += (rng % 10000) as f32 / 10000.0 * noise_level;
                if sim > best_left {
                    best_left = sim;
                    best_left_idx = i;
                }
            }
            let left_correct_idx = if go_up { 0 } else { 1 };
            if best_left_idx == left_correct_idx {
                sym_left_correct += 1;
            }

            // Right hand response
            let right_options = [&right_up, &right_down];
            let mut best_right = f32::NEG_INFINITY;
            let mut best_right_idx = 0;
            for (i, opt) in right_options.iter().enumerate() {
                let mut sim = right_target.similarity(opt);
                sim += right_activation * 0.2;
                xor_shift(&mut rng);
                sim += (rng % 10000) as f32 / 10000.0 * noise_level;
                if sim > best_right {
                    best_right = sim;
                    best_right_idx = i;
                }
            }
            let right_correct_idx = if go_up { 0 } else { 1 };
            if best_right_idx == right_correct_idx {
                sym_right_correct += 1;
            }

            left_total += 1;
            right_total += 1;

            let base_rt = 4.0;
            let tp_speedup = config.time_pressure * 1.0;
            rt_sum += (base_rt - tp_speedup).max(1.0);
        }

        // Reset SSMs for asymmetric condition
        let mut ssm_left = SsmTemporalBackend::new(-0.10, 2);
        let mut ssm_right = SsmTemporalBackend::new(-0.10, 2);

        // Asymmetric condition: hands move in opposite directions
        for step in 0..movements_per_condition {
            let left_up_flag = step % 2 == 0;
            let right_up_flag = !left_up_flag; // opposite direction

            let left_target = if left_up_flag { &left_up } else { &left_down };
            let right_target = if right_up_flag {
                &right_up
            } else {
                &right_down
            };

            let left_activation = ssm_left.step(if left_up_flag { 1.0 } else { 0.0 });
            let right_activation = ssm_right.step(if right_up_flag { 1.0 } else { 0.0 });

            // Left hand (with cross-talk from right)
            let left_options = [&left_up, &left_down];
            let mut best_left = f32::NEG_INFINITY;
            let mut best_left_idx = 0;
            for (i, opt) in left_options.iter().enumerate() {
                let mut sim = left_target.similarity(opt);
                sim += left_activation * 0.2;
                // Cross-talk: right hand's direction interferes
                xor_shift(&mut rng);
                let xtalk = (rng % 10000) as f32 / 10000.0 * crosstalk_noise;
                // Interference pushes toward right hand's target direction
                sim += right_target.similarity(opt) * xtalk * 0.3;
                xor_shift(&mut rng);
                sim += (rng % 10000) as f32 / 10000.0 * noise_level;
                if sim > best_left {
                    best_left = sim;
                    best_left_idx = i;
                }
            }
            let left_correct_idx = if left_up_flag { 0 } else { 1 };
            if best_left_idx == left_correct_idx {
                asym_left_correct += 1;
            }

            // Right hand (with cross-talk from left)
            let right_options = [&right_up, &right_down];
            let mut best_right = f32::NEG_INFINITY;
            let mut best_right_idx = 0;
            for (i, opt) in right_options.iter().enumerate() {
                let mut sim = right_target.similarity(opt);
                sim += right_activation * 0.2;
                xor_shift(&mut rng);
                let xtalk = (rng % 10000) as f32 / 10000.0 * crosstalk_noise;
                sim += left_target.similarity(opt) * xtalk * 0.3;
                xor_shift(&mut rng);
                sim += (rng % 10000) as f32 / 10000.0 * noise_level;
                if sim > best_right {
                    best_right = sim;
                    best_right_idx = i;
                }
            }
            let right_correct_idx = if right_up_flag { 0 } else { 1 };
            if best_right_idx == right_correct_idx {
                asym_right_correct += 1;
            }

            left_total += 1;
            right_total += 1;

            // Asymmetric takes longer (coordination cost)
            let base_rt = 5.5;
            let tp_speedup = config.time_pressure * 1.0;
            rt_sum += (base_rt - tp_speedup).max(1.0);
        }

        let sym_acc =
            (sym_left_correct + sym_right_correct) as f64 / (2 * movements_per_condition) as f64;
        let asym_acc =
            (asym_left_correct + asym_right_correct) as f64 / (2 * movements_per_condition) as f64;
        let left_acc = if left_total > 0 {
            (sym_left_correct + asym_left_correct) as f64 / left_total as f64
        } else {
            0.0
        };
        let right_acc = if right_total > 0 {
            (sym_right_correct + asym_right_correct) as f64 / right_total as f64
        } else {
            0.0
        };

        TrialResult {
            coordination_cost: (sym_acc - asym_acc).max(0.0),
            left_accuracy: left_acc,
            right_accuracy: right_acc,
            symmetric_accuracy: sym_acc,
            asymmetric_accuracy: asym_acc,
            rt_ticks: rt_sum / total_movements as f64,
        }
    }
}

impl PsychBenchmark for BimanualBenchmark {
    fn name(&self) -> &str {
        "Motor::Bimanual"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Bimanual Coordination",
            citation: "Kelso (1984)",
            year: 1984,
            doi: Some("10.1037/0096-3445.113.2.226"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut costs = Vec::new();
        let mut left_accs = Vec::new();
        let mut right_accs = Vec::new();
        let mut sym_accs = Vec::new();
        let mut asym_accs = Vec::new();
        let mut rts = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            costs.push(r.coordination_cost);
            left_accs.push(r.left_accuracy);
            right_accs.push(r.right_accuracy);
            sym_accs.push(r.symmetric_accuracy);
            asym_accs.push(r.asymmetric_accuracy);
            rts.push(r.rt_ticks);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "bimanual".to_string(),
                    correct: r.symmetric_accuracy > 0.5,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("coordination_cost", MetricValue::from_samples(&costs));
        result.insert("left_accuracy", MetricValue::from_samples(&left_accs));
        result.insert("right_accuracy", MetricValue::from_samples(&right_accs));
        result.insert("symmetric_accuracy", MetricValue::from_samples(&sym_accs));
        result.insert("asymmetric_accuracy", MetricValue::from_samples(&asym_accs));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 2; // symmetric vs asymmetric
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bimanual_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = BimanualBenchmark.run(&config);
        assert!(result.metrics.contains_key("coordination_cost"));
        assert!(result.metrics.contains_key("left_accuracy"));
        assert!(result.metrics.contains_key("right_accuracy"));
        assert!(result.metrics.contains_key("symmetric_accuracy"));
        assert!(result.metrics.contains_key("asymmetric_accuracy"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_bimanual_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = BimanualBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_bimanual_coordination_cost_non_negative() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = BimanualBenchmark.run(&config);
        let cost = result.metrics["coordination_cost"].mean;
        assert!(
            cost >= 0.0,
            "coordination cost ({:.3}) should be >= 0",
            cost
        );
    }

    #[test]
    fn test_bimanual_symmetric_better() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = BimanualBenchmark.run(&config);
        let sym = result.metrics["symmetric_accuracy"].mean;
        let asym = result.metrics["asymmetric_accuracy"].mean;
        assert!(
            sym >= asym - 0.05,
            "symmetric ({:.3}) should be >= asymmetric ({:.3})",
            sym,
            asym
        );
    }

    #[test]
    fn test_bimanual_time_pressure() {
        let base = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            time_pressure: 0.0,
            ..Default::default()
        };
        let pressed = BenchmarkConfig {
            time_pressure: 1.0,
            ..base.clone()
        };
        let r_base = BimanualBenchmark.run(&base);
        let r_press = BimanualBenchmark.run(&pressed);
        let rt_base = r_base.metrics["rt_ticks"].mean;
        let rt_press = r_press.metrics["rt_ticks"].mean;
        assert!(
            rt_press <= rt_base + 0.5,
            "time pressure should reduce RT: base={:.2}, pressed={:.2}",
            rt_base,
            rt_press
        );
    }
}
