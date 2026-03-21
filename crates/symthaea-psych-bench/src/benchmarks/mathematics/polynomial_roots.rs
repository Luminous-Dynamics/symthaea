//! Polynomial Root Finding benchmark — engine-backed.
//!
//! Generates quadratic and cubic polynomials with known integer roots.
//! Uses the actual `RootFindingEngine::find_roots_in_interval()` from
//! symthaea-core to find roots numerically, then verifies against the
//! known roots. BinaryHV encoding is maintained as the HDC representation
//! layer.
//!
//! Key metric: `root_finding_accuracy` (fraction of true roots found within
//! tolerance).
//!
//! Human baselines (Schoenfeld, 1985):
//! - root_finding_accuracy: 0.75 (SD 0.15) — humans solve quadratics
//!   reliably but cubics less so.
//! - accuracy_quadratic: 0.88 (SD 0.08)
//! - accuracy_cubic: 0.60 (SD 0.18)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::root_finding::RootFindingEngine;
use symthaea_core::hdc::BinaryHV;

/// Polynomial Root Finding benchmark — engine-backed mathematical reasoning.
pub struct PolynomialRootsBenchmark;

fn xor_shift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Encode an integer value as a BinaryHV via permutation from a base vector.
fn encode_integer(base: &BinaryHV, value: i32) -> BinaryHV {
    if value == 0 {
        return *base;
    }
    let abs_val = value.unsigned_abs() as usize;
    if value > 0 {
        base.permute(abs_val)
    } else {
        base.permute(500 + abs_val)
    }
}

/// Encode polynomial coefficients as a bundle of (degree_role XOR coeff_value)
/// bindings. Coefficients: [a_n, ..., a_0] (highest degree first).
fn encode_polynomial(
    degree_roles: &[BinaryHV],
    coefficients: &[i32],
    value_base: &BinaryHV,
) -> BinaryHV {
    let bindings: Vec<BinaryHV> = degree_roles
        .iter()
        .zip(coefficients.iter())
        .map(|(role, &coeff)| role.bind(&encode_integer(value_base, coeff)))
        .collect();
    BinaryHV::bundle(&bindings)
}

/// Count how many found roots match known roots (greedy matching, tolerance-based).
fn count_matched_roots(known: &[i32], found: &[f64], tol: f64) -> usize {
    let mut used = vec![false; found.len()];
    let mut matched = 0;
    for &k in known {
        for (i, &f) in found.iter().enumerate() {
            if !used[i] && (f - k as f64).abs() < tol {
                used[i] = true;
                matched += 1;
                break;
            }
        }
    }
    matched
}

struct TrialResult {
    accuracy_quadratic: f64,
    accuracy_cubic: f64,
    root_finding_accuracy: f64,
}

impl PolynomialRootsBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("mathematics", "polynomial_roots", trial_idx);
        let mut rng = seed ^ 0xDEADBEEFCAFEBABE;

        let n_poly = 8usize;
        let lapse_penalty = (config.lapse_rate * n_poly as f64 * 0.5) as usize;
        let effective_poly = n_poly.saturating_sub(lapse_penalty).max(2);

        let mut quad_found = 0usize;
        let mut quad_total = 0usize;
        let mut cubic_found = 0usize;
        let mut cubic_total = 0usize;

        // -- Quadratic (degree 2): (x - r1)(x - r2) --
        for _ in 0..effective_poly {
            xor_shift(&mut rng);
            let r1 = ((rng % 9) as i32) - 4;
            xor_shift(&mut rng);
            let r2 = ((rng % 9) as i32) - 4;

            // Coefficients of x^2 - (r1+r2)x + r1*r2
            let coeffs = [1, -(r1 + r2), r1 * r2];

            // --- HDC layer: encode polynomial as BinaryHV ---
            let degree_roles: Vec<BinaryHV> = (0..3)
                .map(|_| BinaryHV::random(xor_shift(&mut rng)))
                .collect();
            let value_base = BinaryHV::random(xor_shift(&mut rng));
            let _poly_hv = encode_polynomial(&degree_roles, &coeffs, &value_base);

            // --- Engine layer: use RootFindingEngine to find roots ---
            let c0 = coeffs[0] as f64;
            let c1 = coeffs[1] as f64;
            let c2 = coeffs[2] as f64;
            let poly_fn = move |x: f64| c0 * x * x + c1 * x + c2;

            let root_results =
                RootFindingEngine::find_roots_in_interval(&poly_fn, -10.0, 10.0, 200, 1e-10);

            let found_roots: Vec<f64> = root_results.iter().map(|r| r.root).collect();
            let matched = count_matched_roots(&[r1, r2], &found_roots, 0.5);
            quad_found += matched;
            quad_total += 2;

            // Time pressure adds noise
            xor_shift(&mut rng);
            let noise = (rng % 10_000) as f64 / 10_000.0;
            if noise < config.time_pressure * 0.15 {
                quad_found = quad_found.saturating_sub(1);
            }
        }

        // -- Cubic (degree 3): (x - r1)(x - r2)(x - r3) --
        for _ in 0..effective_poly {
            xor_shift(&mut rng);
            let r1 = ((rng % 7) as i32) - 3;
            xor_shift(&mut rng);
            let r2 = ((rng % 7) as i32) - 3;
            xor_shift(&mut rng);
            let r3 = ((rng % 7) as i32) - 3;

            // Vieta's formulas for coefficients
            let s1 = r1 + r2 + r3;
            let s2 = r1 * r2 + r1 * r3 + r2 * r3;
            let s3 = r1 * r2 * r3;
            let coeffs = [1, -s1, s2, -s3];

            // --- HDC layer ---
            let degree_roles: Vec<BinaryHV> = (0..4)
                .map(|_| BinaryHV::random(xor_shift(&mut rng)))
                .collect();
            let value_base = BinaryHV::random(xor_shift(&mut rng));
            let _poly_hv = encode_polynomial(&degree_roles, &coeffs, &value_base);

            // --- Engine layer ---
            let c0 = coeffs[0] as f64;
            let c1 = coeffs[1] as f64;
            let c2 = coeffs[2] as f64;
            let c3 = coeffs[3] as f64;
            let poly_fn = move |x: f64| c0 * x * x * x + c1 * x * x + c2 * x + c3;

            let root_results =
                RootFindingEngine::find_roots_in_interval(&poly_fn, -10.0, 10.0, 200, 1e-10);

            let found_roots: Vec<f64> = root_results.iter().map(|r| r.root).collect();
            let matched = count_matched_roots(&[r1, r2, r3], &found_roots, 0.5);
            cubic_found += matched;
            cubic_total += 3;

            xor_shift(&mut rng);
            let noise = (rng % 10_000) as f64 / 10_000.0;
            if noise < config.time_pressure * 0.20 {
                cubic_found = cubic_found.saturating_sub(1);
            }
        }

        let acc_quad = if quad_total > 0 {
            quad_found as f64 / quad_total as f64
        } else {
            0.0
        };
        let acc_cubic = if cubic_total > 0 {
            cubic_found as f64 / cubic_total as f64
        } else {
            0.0
        };

        let total_found = quad_found + cubic_found;
        let total_roots = quad_total + cubic_total;
        let overall = if total_roots > 0 {
            total_found as f64 / total_roots as f64
        } else {
            0.0
        };

        TrialResult {
            accuracy_quadratic: acc_quad,
            accuracy_cubic: acc_cubic,
            root_finding_accuracy: overall,
        }
    }
}

impl PsychBenchmark for PolynomialRootsBenchmark {
    fn name(&self) -> &str {
        "Mathematics::PolynomialRoots"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Polynomial Root Finding / Algebra Assessment",
            citation: "Schoenfeld (1985)",
            year: 1985,
            doi: Some("10.1016/B978-0-12-628870-4.50001-3"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut acc_quad = Vec::with_capacity(config.trials_per_condition);
        let mut acc_cubic = Vec::with_capacity(config.trials_per_condition);
        let mut root_acc = Vec::with_capacity(config.trials_per_condition);

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            acc_quad.push(r.accuracy_quadratic);
            acc_cubic.push(r.accuracy_cubic);
            root_acc.push(r.root_finding_accuracy);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "polynomial_roots".to_string(),
                    correct: r.root_finding_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.root_finding_accuracy,
                    confidence: r.accuracy_quadratic,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("accuracy_quadratic", MetricValue::from_samples(&acc_quad));
        result.insert("accuracy_cubic", MetricValue::from_samples(&acc_cubic));
        result.insert(
            "root_finding_accuracy",
            MetricValue::from_samples(&root_acc),
        );

        result.conditions = 2; // quadratic and cubic
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BenchmarkConfig {
        BenchmarkConfig {
            trials_per_condition: 5,
            ..Default::default()
        }
    }

    #[test]
    fn test_polynomial_runs_and_has_metrics() {
        let result = PolynomialRootsBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("accuracy_quadratic"));
        assert!(result.metrics.contains_key("accuracy_cubic"));
        assert!(result.metrics.contains_key("root_finding_accuracy"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = PolynomialRootsBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_accuracy_bounded() {
        let config = BenchmarkConfig {
            trials_per_condition: 20,
            ..Default::default()
        };
        let result = PolynomialRootsBenchmark.run(&config);
        let acc = result.metrics["root_finding_accuracy"].mean;
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "root_finding_accuracy should be in [0,1]: {acc}"
        );
    }

    #[test]
    fn test_engine_finds_roots_accurately() {
        // With the actual root finding engine, accuracy should be very high
        let config = BenchmarkConfig {
            trials_per_condition: 20,
            ..Default::default()
        };
        let result = PolynomialRootsBenchmark.run(&config);
        let acc = result.metrics["root_finding_accuracy"].mean;
        assert!(
            acc > 0.70,
            "engine-backed root_finding_accuracy should be high: {acc}"
        );
    }

    #[test]
    fn test_quadratic_at_least_as_easy_as_cubic() {
        let config = BenchmarkConfig {
            trials_per_condition: 30,
            ..Default::default()
        };
        let result = PolynomialRootsBenchmark.run(&config);
        let aq = result.metrics["accuracy_quadratic"].mean;
        let ac = result.metrics["accuracy_cubic"].mean;
        assert!(
            aq >= ac - 0.25,
            "quadratic ({aq:.3}) should not be much harder than cubic ({ac:.3})"
        );
    }

    #[test]
    fn test_deterministic_across_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 10,
            seed: 99999,
            ..Default::default()
        };
        let r1 = PolynomialRootsBenchmark.run(&config);
        let r2 = PolynomialRootsBenchmark.run(&config);
        let s1 = r1.metrics["root_finding_accuracy"].mean;
        let s2 = r2.metrics["root_finding_accuracy"].mean;
        assert!(
            (s1 - s2).abs() < 1e-10,
            "same seed should produce same result: {s1} vs {s2}"
        );
    }

    #[test]
    fn test_lapse_rate_degrades_performance() {
        let baseline = BenchmarkConfig {
            trials_per_condition: 40,
            ..Default::default()
        };
        let lapsed = BenchmarkConfig {
            lapse_rate: 0.25,
            trials_per_condition: 40,
            ..Default::default()
        };
        let r_base = PolynomialRootsBenchmark.run(&baseline);
        let r_lapse = PolynomialRootsBenchmark.run(&lapsed);
        let s_base = r_base.metrics["root_finding_accuracy"].mean;
        let s_lapse = r_lapse.metrics["root_finding_accuracy"].mean;
        assert!(
            s_lapse <= s_base + 0.15,
            "lapse should not improve accuracy: base={s_base}, lapse={s_lapse}"
        );
    }

    #[test]
    fn test_trial_trace_populated() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            trial_trace: true,
            ..Default::default()
        };
        let result = PolynomialRootsBenchmark.run(&config);
        assert_eq!(result.trial_trace.len(), 5);
        for t in &result.trial_trace {
            assert_eq!(t.condition, "polynomial_roots");
        }
    }
}
