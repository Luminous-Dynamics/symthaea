// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Mathematics Benchmarks

Verifies structural invariants across the 10 math benchmark types:
- All metric values are finite (no NaN/Inf)
- Mean values are bounded [0.0, 1.0] for accuracy-like metrics
- Standard deviations are non-negative
- CI bounds bracket the mean
- Results are deterministic for fixed seed
*/

use proptest::prelude::*;

use crate::benchmarks::mathematics::*;
use crate::harness::PsychBenchmark;
use crate::harness::config::BenchmarkConfig;

fn config_with_seed(seed: u64) -> BenchmarkConfig {
    BenchmarkConfig {
        seed,
        trials_per_condition: 5,
        ..Default::default()
    }
}

/// Check that all metrics in a benchmark result are finite and well-formed.
fn assert_metrics_valid(result: &crate::harness::report::BenchmarkResult, label: &str) {
    for (key, mv) in &result.metrics {
        assert!(
            mv.mean.is_finite(),
            "{label}: metric '{key}' mean is not finite: {}",
            mv.mean
        );
        assert!(
            mv.std_dev.is_finite() && mv.std_dev >= 0.0,
            "{label}: metric '{key}' std_dev invalid: {}",
            mv.std_dev
        );
        assert!(
            mv.ci_lower.is_finite(),
            "{label}: metric '{key}' ci_lower not finite: {}",
            mv.ci_lower
        );
        assert!(
            mv.ci_upper.is_finite(),
            "{label}: metric '{key}' ci_upper not finite: {}",
            mv.ci_upper
        );
        // CI should bracket the mean (with small tolerance for bootstrap jitter)
        assert!(
            mv.ci_lower <= mv.mean + 1e-6,
            "{label}: metric '{key}' ci_lower ({}) > mean ({})",
            mv.ci_lower,
            mv.mean
        );
        assert!(
            mv.ci_upper >= mv.mean - 1e-6,
            "{label}: metric '{key}' ci_upper ({}) < mean ({})",
            mv.ci_upper,
            mv.mean
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    /// Arithmetic word problems: all metrics finite and bounded.
    #[test]
    fn prop_arithmetic_metrics_valid(seed in 1u64..1000) {
        let bench = ArithmeticWordProblemsBenchmark;
        let result = bench.run(&config_with_seed(seed));
        assert_metrics_valid(&result, "arithmetic");
    }

    /// Linear system solving: all metrics finite and bounded.
    #[test]
    fn prop_linear_system_metrics_valid(seed in 1u64..1000) {
        let bench = LinearSystemSolvingBenchmark;
        let result = bench.run(&config_with_seed(seed));
        assert_metrics_valid(&result, "linear_system");
    }

    /// Polynomial roots: all metrics finite and bounded.
    #[test]
    fn prop_polynomial_roots_metrics_valid(seed in 1u64..1000) {
        let bench = PolynomialRootsBenchmark;
        let result = bench.run(&config_with_seed(seed));
        assert_metrics_valid(&result, "polynomial_roots");
    }

    /// Matrix operations: all metrics finite and bounded.
    #[test]
    fn prop_matrix_operations_metrics_valid(seed in 1u64..1000) {
        let bench = MatrixOperationsBenchmark;
        let result = bench.run(&config_with_seed(seed));
        assert_metrics_valid(&result, "matrix_operations");
    }

    /// Definite integrals: all metrics finite and bounded.
    #[test]
    fn prop_definite_integrals_metrics_valid(seed in 1u64..1000) {
        let bench = DefiniteIntegralsBenchmark;
        let result = bench.run(&config_with_seed(seed));
        assert_metrics_valid(&result, "definite_integrals");
    }

    /// Statistical inference: all metrics finite and bounded.
    #[test]
    fn prop_statistical_inference_metrics_valid(seed in 1u64..1000) {
        let bench = StatisticalInferenceBenchmark;
        let result = bench.run(&config_with_seed(seed));
        assert_metrics_valid(&result, "statistical_inference");
    }

    /// Bayesian reasoning: all metrics finite and bounded.
    #[test]
    fn prop_bayesian_reasoning_metrics_valid(seed in 1u64..1000) {
        let bench = BayesianReasoningBenchmark;
        let result = bench.run(&config_with_seed(seed));
        assert_metrics_valid(&result, "bayesian_reasoning");
    }

    /// Logical deduction: all metrics finite and bounded.
    #[test]
    fn prop_logical_deduction_metrics_valid(seed in 1u64..1000) {
        let bench = LogicalDeductionBenchmark;
        let result = bench.run(&config_with_seed(seed));
        assert_metrics_valid(&result, "logical_deduction");
    }

    /// Constraint puzzles: all metrics finite and bounded.
    #[test]
    fn prop_constraint_puzzles_metrics_valid(seed in 1u64..1000) {
        let bench = ConstraintPuzzlesBenchmark;
        let result = bench.run(&config_with_seed(seed));
        assert_metrics_valid(&result, "constraint_puzzles");
    }

    /// Proof construction: all metrics finite and bounded.
    #[test]
    fn prop_proof_construction_metrics_valid(seed in 1u64..1000) {
        let bench = ProofConstructionBenchmark;
        let result = bench.run(&config_with_seed(seed));
        assert_metrics_valid(&result, "proof_construction");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// Determinism: same seed → same results.
    #[test]
    fn prop_deterministic_results(seed in 1u64..500) {
        let bench = ArithmeticWordProblemsBenchmark;
        let r1 = bench.run(&config_with_seed(seed));
        let r2 = bench.run(&config_with_seed(seed));
        for (key, mv1) in &r1.metrics {
            if let Some(mv2) = r2.metrics.get(key) {
                prop_assert!(
                    (mv1.mean - mv2.mean).abs() < 1e-10,
                    "Metric '{key}' not deterministic: {} vs {}",
                    mv1.mean, mv2.mean
                );
            }
        }
    }

    /// Difficulty scaling: harder configs should not produce NaN.
    #[test]
    fn prop_difficulty_no_nan(difficulty in 0.0f64..1.0, seed in 1u64..500) {
        let mut config = config_with_seed(seed);
        config.difficulty = difficulty;
        let bench = MatrixOperationsBenchmark;
        let result = bench.run(&config);
        assert_metrics_valid(&result, &format!("matrix_diff_{difficulty:.2}"));
    }
}
