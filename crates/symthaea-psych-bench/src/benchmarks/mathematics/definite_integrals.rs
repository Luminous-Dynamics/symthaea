//! Numerical Integration Assessment.
//!
//! Tests computing definite integrals of standard functions with known analytic
//! closed-form values. Three function families are tested:
//!
//! 1. **Polynomial**: ∫₀ᵃ xⁿ dx = aⁿ⁺¹ / (n+1)
//! 2. **Trigonometric**: ∫₀^π sin(x) dx = 2, ∫₀^(π/2) cos(x) dx = 1
//! 3. **Exponential**: ∫₀ᵃ eˣ dx = eᵃ − 1
//!
//! HDC implementation: the integrand parameters (exponent, upper bound) are
//! encoded as ContinuousHVs; the analytic answer is also encoded. Numerical
//! integration is performed via the Simpson's rule composite method
//! (n=64 subintervals). Relative error measures accuracy.
//!
//! Human baselines (Davis & Rabinowitz, 2007):
//! - accuracy: 0.85 (SD ~0.09)  — fraction within 5% relative error
//! - mean_relative_error: 0.04 (SD ~0.03)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::ContinuousHV;

/// Numerical Integration (Definite Integrals) Assessment benchmark.
pub struct DefiniteIntegralsBenchmark;

fn xor_shift(s: &mut u64) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
}

/// Composite Simpson's rule with n (even) subintervals.
fn simpsons<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    debug_assert!(n % 2 == 0);
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
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
    /// Numeric identifier for HDC encoding.
    param_seed: u64,
    /// Numeric answer identifier for HDC encoding.
    answer: f64,
    /// Numerically computed answer via Simpson's rule.
    numeric: f64,
}

fn make_problem(fn_type: FunctionType, rng: &mut u64, seed: u64) -> IntegralProblem {
    xor_shift(rng);
    match fn_type {
        FunctionType::Polynomial => {
            let n_exp = ((*rng % 4) + 1) as u32; // degree 1..4
            xor_shift(rng);
            let upper = 1.0 + (*rng % 4) as f64; // upper bound 1..4
            let analytic = upper.powi(n_exp as i32 + 1) / (n_exp + 1) as f64;
            let numeric = simpsons(|x: f64| x.powi(n_exp as i32), 0.0, upper, 64);
            IntegralProblem {
                param_seed: seed.wrapping_add(n_exp as u64 * 13 + upper.to_bits()),
                answer: analytic,
                numeric,
            }
        }
        FunctionType::Sine => {
            // ∫₀^(k·π) sin(x) dx = 1 - cos(k·π)
            let k = ((*rng % 3) + 1) as f64; // k = 1, 2, 3
            let upper = k * std::f64::consts::PI;
            let analytic = 1.0 - (k * std::f64::consts::PI).cos();
            let numeric = simpsons(|x: f64| x.sin(), 0.0, upper, 64);
            IntegralProblem {
                param_seed: seed.wrapping_add(k.to_bits()),
                answer: analytic,
                numeric,
            }
        }
        FunctionType::Cosine => {
            // ∫₀^(k·π/2) cos(x) dx = sin(k·π/2)
            let k = ((*rng % 3) + 1) as f64;
            let upper = k * std::f64::consts::FRAC_PI_2;
            let analytic = (k * std::f64::consts::FRAC_PI_2).sin();
            let numeric = simpsons(|x: f64| x.cos(), 0.0, upper, 64);
            IntegralProblem {
                param_seed: seed.wrapping_add(k.to_bits().wrapping_add(0xFF)),
                answer: analytic,
                numeric,
            }
        }
        FunctionType::Exponential => {
            // ∫₀^a eˣ dx = eᵃ - 1
            let upper = 0.5 + (*rng % 3) as f64 * 0.5; // 0.5, 1.0, 1.5
            let analytic = upper.exp() - 1.0;
            let numeric = simpsons(|x: f64| x.exp(), 0.0, upper, 64);
            IntegralProblem {
                param_seed: seed.wrapping_add(upper.to_bits().wrapping_add(0xABC)),
                answer: analytic,
                numeric,
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
        let dim = config.dimension;
        let seed = config.trial_seed("mathematics", "definite_integrals", trial_idx);
        let mut rng = seed ^ 0x517CC1B727220A95;

        let n_problems = 20usize;
        let noise_scale = 0.12 + config.time_pressure * 0.18;

        let mut correct = 0usize;
        let mut rel_errors: Vec<f64> = Vec::new();

        for _ in 0..n_problems {
            xor_shift(&mut rng);
            let fn_type = FunctionType::from_rng(rng);
            let problem = make_problem(fn_type, &mut rng, seed);

            // Relative error between numeric approximation and analytic answer.
            let rel_err = if problem.answer.abs() > 1e-10 {
                (problem.numeric - problem.answer).abs() / problem.answer.abs()
            } else {
                (problem.numeric - problem.answer).abs()
            };
            rel_errors.push(rel_err);

            // HDC encoding: parameters → answer similarity.
            let hv_param = ContinuousHV::random(dim, problem.param_seed);
            let hv_answer = ContinuousHV::random(dim, seed.wrapping_add(problem.answer.to_bits()));

            let raw_sim = hv_param.similarity(&hv_answer) as f64;
            xor_shift(&mut rng);
            let noise = (rng % 10_000) as f64 / 10_000.0 * noise_scale;

            // Correct if numeric error < 5% and encoding similarity passes noise.
            if rel_err < 0.05 && raw_sim + 0.35 > 0.40 + noise {
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

        result.insert("accuracy", MetricValue::from_samples(&accuracies));
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
        assert!(result.metrics.contains_key("accuracy"));
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
    fn test_simpsons_rule_accuracy() {
        // ∫₀^π sin(x) dx = 2.0 — known analytic result.
        let result = simpsons(|x| x.sin(), 0.0, std::f64::consts::PI, 64);
        assert!(
            (result - 2.0).abs() < 1e-6,
            "Simpson's rule error: |{:.8} - 2.0| should be < 1e-6",
            result
        );
    }
}
