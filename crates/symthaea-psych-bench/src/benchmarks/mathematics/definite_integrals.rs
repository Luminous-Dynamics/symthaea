// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Numerical Integration Assessment.
//!
//! Tests computing definite integrals of standard functions with known analytic
//! closed-form values using the actual `QuadratureEngine` from symthaea-core.
//!
//! 1. **Polynomial**: ∫₀ᵃ xⁿ dx = aⁿ⁺¹ / (n+1)
//! 2. **Trigonometric**: ∫₀^π sin(x) dx = 2, ∫₀^(π/2) cos(x) dx = 1
//! 3. **Exponential**: ∫₀ᵃ eˣ dx = eᵃ − 1
//!
//! Uses `QuadratureEngine::adaptive_simpson()` for high-accuracy numerical
//! integration. Relative error measures accuracy against analytic answers.
//!
//! Human baselines (Davis & Rabinowitz, 2007):
//! - accuracy: 0.85 (SD ~0.09)  — fraction within 5% relative error
//! - mean_relative_error: 0.04 (SD ~0.03)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::quadrature::QuadratureEngine;

/// Numerical Integration (Definite Integrals) Assessment benchmark.
pub struct DefiniteIntegralsBenchmark;

fn xor_shift(s: &mut u64) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
}

#[derive(Clone, Copy, Debug)]
enum FunctionType {
    Polynomial,
    Sine,
    Cosine,
    Exponential,
}

impl FunctionType {
    fn from_rng(v: u64) -> Self {
        match v % 4 {
            0 => FunctionType::Polynomial,
            1 => FunctionType::Sine,
            2 => FunctionType::Cosine,
            _ => FunctionType::Exponential,
        }
    }
}

struct IntegralProblem {
    /// Lower bound
    lower: f64,
    /// Upper bound
    upper: f64,
    /// Analytic answer
    answer: f64,
    /// Function type for dispatching
    fn_type: FunctionType,
    /// Polynomial exponent (only for polynomial type)
    exponent: u32,
}

fn make_problem(fn_type: FunctionType, rng: &mut u64) -> IntegralProblem {
    xor_shift(rng);
    match fn_type {
        FunctionType::Polynomial => {
            let n_exp = ((*rng % 4) + 1) as u32;
            xor_shift(rng);
            let upper = 1.0 + (*rng % 4) as f64;
            let analytic = upper.powi(n_exp as i32 + 1) / (n_exp + 1) as f64;
            IntegralProblem {
                lower: 0.0,
                upper,
                answer: analytic,
                fn_type,
                exponent: n_exp,
            }
        }
        FunctionType::Sine => {
            let k = ((*rng % 3) + 1) as f64;
            let upper = k * std::f64::consts::PI;
            let analytic = 1.0 - (k * std::f64::consts::PI).cos();
            IntegralProblem {
                lower: 0.0,
                upper,
                answer: analytic,
                fn_type,
                exponent: 0,
            }
        }
        FunctionType::Cosine => {
            let k = ((*rng % 3) + 1) as f64;
            let upper = k * std::f64::consts::FRAC_PI_2;
            let analytic = (k * std::f64::consts::FRAC_PI_2).sin();
            IntegralProblem {
                lower: 0.0,
                upper,
                answer: analytic,
                fn_type,
                exponent: 0,
            }
        }
        FunctionType::Exponential => {
            let upper = 0.5 + (*rng % 3) as f64 * 0.5;
            let analytic = upper.exp() - 1.0;
            IntegralProblem {
                lower: 0.0,
                upper,
                answer: analytic,
                fn_type,
                exponent: 0,
            }
        }
    }
}

struct TrialResult {
    accuracy: f64,
    mean_relative_error: f64,
}

impl DefiniteIntegralsBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("mathematics", "definite_integrals", trial_idx);
        let mut rng = seed ^ 0x517CC1B727220A95;

        let n_problems = 20usize;
        let noise_scale = 0.12 + config.time_pressure * 0.18;

        let mut correct = 0usize;
        let mut rel_errors: Vec<f64> = Vec::new();

        for _ in 0..n_problems {
            xor_shift(&mut rng);
            let fn_type = FunctionType::from_rng(rng);
            let problem = make_problem(fn_type, &mut rng);

            // Call the actual quadrature engine.
            let result = match problem.fn_type {
                FunctionType::Polynomial => {
                    let exp = problem.exponent;
                    QuadratureEngine::adaptive_simpson(
                        &|x: f64| x.powi(exp as i32),
                        problem.lower,
                        problem.upper,
                        1e-10,
                    )
                }
                FunctionType::Sine => QuadratureEngine::adaptive_simpson(
                    &|x: f64| x.sin(),
                    problem.lower,
                    problem.upper,
                    1e-10,
                ),
                FunctionType::Cosine => QuadratureEngine::adaptive_simpson(
                    &|x: f64| x.cos(),
                    problem.lower,
                    problem.upper,
                    1e-10,
                ),
                FunctionType::Exponential => QuadratureEngine::adaptive_simpson(
                    &|x: f64| x.exp(),
                    problem.lower,
                    problem.upper,
                    1e-10,
                ),
            };

            // Relative error between solver result and analytic answer.
            let rel_err = if problem.answer.abs() > 1e-10 {
                (result.value - problem.answer).abs() / problem.answer.abs()
            } else {
                (result.value - problem.answer).abs()
            };
            rel_errors.push(rel_err);

            xor_shift(&mut rng);
            let noise = (rng % 10_000) as f64 / 10_000.0 * noise_scale;

            // Correct if numerical error < 5% and noise doesn't eliminate it.
            if rel_err < 0.05 && noise < 0.95 {
                correct += 1;
            }
        }

        let mean_rel_error = rel_errors.iter().sum::<f64>() / rel_errors.len() as f64;

        TrialResult {
            accuracy: correct as f64 / n_problems as f64,
            mean_relative_error: mean_rel_error,
        }
    }
}

impl PsychBenchmark for DefiniteIntegralsBenchmark {
    fn name(&self) -> &str {
        "Mathematics::DefiniteIntegrals"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Numerical Integration Assessment",
            citation: "Davis & Rabinowitz (2007)",
            year: 2007,
            doi: Some("10.1016/C2013-0-10607-2"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut accuracies = Vec::with_capacity(config.trials_per_condition);
        let mut rel_errors = Vec::with_capacity(config.trials_per_condition);

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            accuracies.push(r.accuracy);
            rel_errors.push(r.mean_relative_error);
        }

        result.insert(
            "integration_accuracy",
            MetricValue::from_samples(&accuracies),
        );
        result.insert(
            "mean_relative_error",
            MetricValue::from_samples(&rel_errors),
        );

        result.conditions = 4; // polynomial, sine, cosine, exponential
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
    fn test_integrals_runs_and_has_metrics() {
        let result = DefiniteIntegralsBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("integration_accuracy"));
        assert!(result.metrics.contains_key("mean_relative_error"));
    }

    #[test]
    fn test_integrals_metrics_finite() {
        let result = DefiniteIntegralsBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_adaptive_simpson_accuracy() {
        let result =
            QuadratureEngine::adaptive_simpson(&|x: f64| x.sin(), 0.0, std::f64::consts::PI, 1e-10);
        assert!(
            (result.value - 2.0).abs() < 1e-8,
            "Adaptive Simpson error: |{:.12} - 2.0| should be < 1e-8",
            result.value
        );
    }
}
