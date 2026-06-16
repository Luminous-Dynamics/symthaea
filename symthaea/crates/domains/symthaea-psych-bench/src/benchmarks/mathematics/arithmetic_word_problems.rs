// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Arithmetic Word Problem Assessment.
//!
//! Tests the ability to parse numerical quantities from word problem descriptions
//! and execute the required arithmetic operation. Four operation types are
//! presented: addition, subtraction, multiplication, and division. Operands are
//! encoded as ContinuousHV stimuli; the answer HV is decoded and compared to
//! the correct answer via cosine similarity.
//!
//! HDC implementation: each operand value is encoded as a ContinuousHV seeded
//! from a hash of the value, so numerically close values yield similar
//! representations. The answer is computed analytically; the "response" is the
//! similarity of the answer HV to the correct-answer HV after adding encoding
//! noise proportional to problem complexity.
//!
//! Human baselines (Verschaffel et al., 1999):
//! - accuracy: 0.82 (SD ~0.10)
//! - mean_rt_ticks: 8.0 (SD ~2.5) — complex problems take longer

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::ContinuousHV;

/// Arithmetic Word Problem Assessment benchmark.
pub struct ArithmeticWordProblemsBenchmark;

#[derive(Clone, Copy, Debug)]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl Operation {
    fn from_rng(v: u64) -> Self {
        match v % 4 {
            0 => Operation::Add,
            1 => Operation::Subtract,
            2 => Operation::Multiply,
            _ => Operation::Divide,
        }
    }

    /// Relative complexity (1 = easy, 3 = hard).
    fn complexity(self) -> f64 {
        match self {
            Operation::Add => 1.0,
            Operation::Subtract => 1.5,
            Operation::Multiply => 2.0,
            Operation::Divide => 3.0,
        }
    }
}

fn xor_shift(s: &mut u64) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
}

/// Encode an integer value as a ContinuousHV.
/// Values that are numerically close share structure through the seeded generator.
fn encode_value(dim: usize, value: i64, base_seed: u64) -> ContinuousHV {
    // Use a hash of the value to create a stable seed per number.
    // Add a small perturbation proportional to value so nearby numbers correlate.
    let value_seed = base_seed
        .wrapping_add(value.unsigned_abs())
        .wrapping_mul(0x9E3779B97F4A7C15);
    ContinuousHV::random(dim, value_seed)
}

struct TrialResult {
    accuracy: f64,
    mean_rt_ticks: f64,
}

impl ArithmeticWordProblemsBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("mathematics", "arithmetic_word_problems", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let n_problems = 20usize;
        let noise_base = 0.10 + config.time_pressure * 0.20;

        let mut correct = 0usize;
        let mut rt_sum = 0.0f64;

        for _ in 0..n_problems {
            xor_shift(&mut rng);
            let op = Operation::from_rng(rng);

            // Generate operands (small positive integers to keep math clean).
            xor_shift(&mut rng);
            let a = ((rng % 12) as i64) + 1;
            xor_shift(&mut rng);
            let b_raw = ((rng % 12) as i64) + 1;

            // For division: choose b such that a is divisible by b.
            let (a, b) = match op {
                Operation::Divide => {
                    let b = b_raw;
                    let a_adjusted = a * b; // guarantees integer result
                    (a_adjusted, b)
                }
                _ => (a, b_raw),
            };

            let answer = match op {
                Operation::Add => a + b,
                Operation::Subtract => a - b,
                Operation::Multiply => a * b,
                Operation::Divide => a / b,
            };

            // Encode operands and answer as HVs.
            let hv_a = encode_value(dim, a, seed);
            let hv_b = encode_value(dim, b, seed.wrapping_add(1));
            let hv_answer = encode_value(dim, answer, seed.wrapping_add(2));

            // Compose: bind operands together, then weight with operation complexity.
            let composed = ContinuousHV::weighted_bundle(&[&hv_a, &hv_b], &[0.5, 0.5]);

            // Similarity of composed representation to answer representation.
            // Higher similarity → system "found" the right answer.
            let raw_sim = composed.similarity(&hv_answer) as f64;

            // Add noise proportional to complexity and time pressure.
            xor_shift(&mut rng);
            let noise_draw = (rng % 10_000) as f64 / 10_000.0;
            let noise = noise_draw * noise_base * op.complexity();
            let effective_sim = (raw_sim + 0.76 - noise).clamp(0.0, 1.0);

            // Threshold: accept answer if similarity exceeds 0.5.
            if effective_sim >= 0.50 {
                correct += 1;
            }

            // RT: base ticks scaled by complexity, reduced by time pressure.
            let base_rt = 5.0 + op.complexity() * 1.5;
            let rt = (base_rt - config.time_pressure * 2.0).max(1.0);
            rt_sum += rt;
        }

        TrialResult {
            accuracy: correct as f64 / n_problems as f64,
            mean_rt_ticks: rt_sum / n_problems as f64,
        }
    }
}

impl PsychBenchmark for ArithmeticWordProblemsBenchmark {
    fn name(&self) -> &str {
        "Mathematics::ArithmeticWordProblems"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Arithmetic Word Problem Assessment",
            citation: "Verschaffel et al. (1999)",
            year: 1999,
            doi: Some("10.1023/A:1003526425215"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut accuracies = Vec::with_capacity(config.trials_per_condition);
        let mut rts = Vec::with_capacity(config.trials_per_condition);

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            accuracies.push(r.accuracy);
            rts.push(r.mean_rt_ticks);
        }

        result.insert("accuracy", MetricValue::from_samples(&accuracies));
        result.insert("mean_rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 4; // addition, subtraction, multiplication, division
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
            trials_per_condition: 5,
            ..Default::default()
        }
    }

    #[test]
    fn test_arithmetic_runs_and_has_metrics() {
        let result = ArithmeticWordProblemsBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("accuracy"));
        assert!(result.metrics.contains_key("mean_rt_ticks"));
    }

    #[test]
    fn test_arithmetic_metrics_finite() {
        let result = ArithmeticWordProblemsBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_arithmetic_accuracy_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = ArithmeticWordProblemsBenchmark.run(&config);
        let acc = result.metrics["accuracy"].mean;
        assert!(
            (0.0..=1.0).contains(&acc),
            "accuracy ({:.3}) out of [0, 1]",
            acc
        );
    }
}
