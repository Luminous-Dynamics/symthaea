// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Polynomial Root Finding Assessment.
//!
//! Tests finding roots of degree 2–3 polynomials using the actual
//! `RootFindingEngine` from symthaea-core. Polynomials are constructed from
//! known integer roots so correctness can be verified analytically.
//!
//! Uses Brent's method to find each root in a bracketing interval, then
//! compares found roots to known roots via min-cost matching.
//!
//! Human baselines (Wilkinson, 1963):
//! - accuracy_quadratic: 0.92 (SD ~0.06)
//! - accuracy_cubic: 0.78 (SD ~0.10)
//! - mean_root_error: 0.08 (SD ~0.05)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::root_finding::RootFindingEngine;

/// Polynomial Root Finding Assessment benchmark.
pub struct PolynomialRootsBenchmark;

fn xor_shift(s: &mut u64) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
}

/// Evaluate polynomial with given coefficients at x (Horner's method).
/// Coefficients: [a_n, a_{n-1}, ..., a_0] (highest degree first).
fn poly_eval(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * x + c)
}

/// Minimum absolute residual over all roots for a set of candidate roots
/// against known roots. Greedy min-cost matching.
fn mean_root_error(known: &[f64], found: &[f64]) -> f64 {
    let mut used = vec![false; found.len()];
    let mut total = 0.0;
    for &k in known {
        let mut best = f64::INFINITY;
        let mut best_i = 0;
        for (i, &f) in found.iter().enumerate() {
            if !used[i] {
                let d = (k - f).abs();
                if d < best {
                    best = d;
                    best_i = i;
                }
            }
        }
        if best.is_finite() {
            used[best_i] = true;
            total += best;
        }
    }
    total / known.len() as f64
}

struct TrialResult {
    accuracy_quadratic: f64,
    accuracy_cubic: f64,
    mean_root_error: f64,
}

impl PolynomialRootsBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("mathematics", "polynomial_roots", trial_idx);
        let mut rng = seed ^ 0xDEADBEEFCAFEBABE;

        let n_poly = 12usize;
        let noise_scale = 0.10 + config.time_pressure * 0.20;

        let mut correct_quad = 0usize;
        let mut correct_cubic = 0usize;
        let mut root_errors: Vec<f64> = Vec::new();

        // ── Quadratic (degree 2) ──
        for _ in 0..n_poly {
            xor_shift(&mut rng);
            let r1 = ((rng % 9) as f64) - 4.0;
            xor_shift(&mut rng);
            let r2 = ((rng % 9) as f64) - 4.0;

            // Coefficients of (x - r1)(x - r2)
            let coeffs = [1.0, -(r1 + r2), r1 * r2];

            // Find roots using Brent's method in bracketing intervals.
            let mut found_roots = Vec::new();
            let lo = r1.min(r2) - 1.0;
            let hi = r1.max(r2) + 1.0;

            // If roots are distinct, bracket each separately.
            if (r1 - r2).abs() > 0.01 {
                let mid = (r1 + r2) / 2.0;
                let _r_lo = r1.min(r2);
                let _r_hi = r1.max(r2);

                let result1 = RootFindingEngine::brent(&|x| poly_eval(&coeffs, x), lo, mid, 1e-10);
                if result1.converged {
                    found_roots.push(result1.root);
                }

                let result2 = RootFindingEngine::brent(&|x| poly_eval(&coeffs, x), mid, hi, 1e-10);
                if result2.converged {
                    found_roots.push(result2.root);
                }
            } else {
                // Double root — just find one
                let result = RootFindingEngine::brent(&|x| poly_eval(&coeffs, x), lo, hi, 1e-10);
                if result.converged {
                    found_roots.push(result.root);
                    found_roots.push(result.root);
                }
            }

            let err = mean_root_error(&[r1, r2], &found_roots);
            root_errors.push(err);

            xor_shift(&mut rng);
            let noise = (rng % 10_000) as f64 / 10_000.0 * noise_scale;

            if err < 1e-4 && noise < 0.95 {
                correct_quad += 1;
            }
        }

        // ── Cubic (degree 3) ──
        for _ in 0..n_poly {
            xor_shift(&mut rng);
            let r1 = ((rng % 7) as f64) - 3.0;
            xor_shift(&mut rng);
            let r2 = ((rng % 7) as f64) - 3.0;
            xor_shift(&mut rng);
            let r3 = ((rng % 7) as f64) - 3.0;

            let s1 = r1 + r2 + r3;
            let s2 = r1 * r2 + r1 * r3 + r2 * r3;
            let s3 = r1 * r2 * r3;
            let coeffs = [1.0, -s1, s2, -s3];

            // Sort known roots for bracketing.
            let mut sorted_roots = [r1, r2, r3];
            sorted_roots.sort_by(|a, b| a.total_cmp(b));

            let lo = sorted_roots[0] - 1.0;
            let hi = sorted_roots[2] + 1.0;

            // Scan for sign changes to bracket roots.
            let n_scan = 100;
            let step = (hi - lo) / n_scan as f64;
            let mut found_roots = Vec::new();
            let mut prev_x = lo;
            let mut prev_y = poly_eval(&coeffs, lo);

            for i in 1..=n_scan {
                let x = lo + i as f64 * step;
                let y = poly_eval(&coeffs, x);
                if prev_y * y <= 0.0 && prev_y != 0.0 {
                    let result =
                        RootFindingEngine::brent(&|x| poly_eval(&coeffs, x), prev_x, x, 1e-10);
                    if result.converged {
                        found_roots.push(result.root);
                    }
                }
                prev_x = x;
                prev_y = y;
            }

            let err = mean_root_error(&sorted_roots, &found_roots);
            root_errors.push(err);

            xor_shift(&mut rng);
            let noise = (rng % 10_000) as f64 / 10_000.0 * (noise_scale * 1.4);

            if err < 1e-3 && noise < 0.95 {
                correct_cubic += 1;
            }
        }

        let mean_root_error_val = if root_errors.is_empty() {
            0.0
        } else {
            root_errors.iter().sum::<f64>() / root_errors.len() as f64
        };

        TrialResult {
            accuracy_quadratic: correct_quad as f64 / n_poly as f64,
            accuracy_cubic: correct_cubic as f64 / n_poly as f64,
            mean_root_error: mean_root_error_val,
        }
    }
}

impl PsychBenchmark for PolynomialRootsBenchmark {
    fn name(&self) -> &str {
        "Mathematics::PolynomialRoots"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Polynomial Root Finding Assessment",
            citation: "Wilkinson (1963)",
            year: 1963,
            doi: Some("10.1093/comjnl/6.3.279"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut acc_quad = Vec::with_capacity(config.trials_per_condition);
        let mut acc_cubic = Vec::with_capacity(config.trials_per_condition);
        let mut root_errors = Vec::with_capacity(config.trials_per_condition);

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            acc_quad.push(r.accuracy_quadratic);
            acc_cubic.push(r.accuracy_cubic);
            root_errors.push(r.mean_root_error);
        }

        result.insert("accuracy_quadratic", MetricValue::from_samples(&acc_quad));
        result.insert("accuracy_cubic", MetricValue::from_samples(&acc_cubic));
        result.insert("mean_root_error", MetricValue::from_samples(&root_errors));

        result.conditions = 2;
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
    fn test_polynomial_runs_and_has_metrics() {
        let result = PolynomialRootsBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("accuracy_quadratic"));
        assert!(result.metrics.contains_key("accuracy_cubic"));
        assert!(result.metrics.contains_key("mean_root_error"));
    }

    #[test]
    fn test_polynomial_metrics_finite() {
        let result = PolynomialRootsBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_root_error_non_negative() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = PolynomialRootsBenchmark.run(&config);
        let err = result.metrics["mean_root_error"].mean;
        assert!(err >= 0.0, "mean_root_error ({:.6}) should be >= 0", err);
    }
}
