//! Independent, hand-authored oracle cases for the engineering curriculum's
//! numeric benchmark-acceptance boundary.
//!
//! Expected values and decisions below are fixed fixture data: they are not
//! computed by BenchmarkValidation's production helpers.

#![cfg(all(feature = "engineering-foundations", feature = "school_learning"))]

use symthaea::school::engineering_curriculum::BenchmarkValidation;

struct OracleCase {
    name: &'static str,
    sample_id: &'static str,
    metric_name: &'static str,
    ground_truth: f64,
    generated_value: f64,
    reported_relative_error: f64,
    converged: bool,
    tolerance: f64,
    expected_relative_error: Option<f64>,
    expected_acceptable: bool,
}

const CASES: &[OracleCase] = &[
    OracleCase {
        name: "exact-match",
        sample_id: "oracle-exact",
        metric_name: "length_m",
        ground_truth: 2.0,
        generated_value: 2.0,
        reported_relative_error: 0.0,
        converged: true,
        tolerance: 0.0,
        expected_relative_error: Some(0.0),
        expected_acceptable: true,
    },
    OracleCase {
        name: "upper-bound-tolerance",
        sample_id: "oracle-boundary",
        metric_name: "drag_coefficient",
        ground_truth: 100.0,
        generated_value: 105.0,
        reported_relative_error: 0.05,
        converged: true,
        tolerance: 0.05,
        expected_relative_error: Some(0.05),
        expected_acceptable: true,
    },
    OracleCase {
        name: "outside-tolerance",
        sample_id: "oracle-outside",
        metric_name: "drag_coefficient",
        ground_truth: 100.0,
        generated_value: 105.01,
        reported_relative_error: 0.0501,
        converged: true,
        tolerance: 0.05,
        expected_relative_error: Some(0.0501),
        expected_acceptable: false,
    },
    OracleCase {
        name: "reported-error-lie",
        sample_id: "oracle-lie",
        metric_name: "temperature_k",
        ground_truth: 1.0,
        generated_value: 2.0,
        reported_relative_error: 0.0,
        converged: true,
        tolerance: 1.0,
        expected_relative_error: Some(1.0),
        expected_acceptable: false,
    },
    OracleCase {
        name: "negative-reported-error",
        sample_id: "oracle-negative",
        metric_name: "temperature_k",
        ground_truth: 1.0,
        generated_value: 1.0,
        reported_relative_error: -0.1,
        converged: true,
        tolerance: 0.0,
        expected_relative_error: Some(0.0),
        expected_acceptable: false,
    },
    OracleCase {
        name: "zero-reference-exact",
        sample_id: "oracle-zero-exact",
        metric_name: "residual",
        ground_truth: 0.0,
        generated_value: 0.0,
        reported_relative_error: 0.0,
        converged: true,
        tolerance: 0.0,
        expected_relative_error: Some(0.0),
        expected_acceptable: true,
    },
    OracleCase {
        name: "zero-reference-nonzero-output",
        sample_id: "oracle-zero-mismatch",
        metric_name: "residual",
        ground_truth: 0.0,
        generated_value: 1.0,
        reported_relative_error: 0.0,
        converged: true,
        tolerance: 1.0,
        expected_relative_error: None,
        expected_acceptable: false,
    },
    OracleCase {
        name: "negative-reference",
        sample_id: "oracle-negative-reference",
        metric_name: "signed_power_w",
        ground_truth: -100.0,
        generated_value: -95.0,
        reported_relative_error: 0.05,
        converged: true,
        tolerance: 0.05,
        expected_relative_error: Some(0.05),
        expected_acceptable: true,
    },
    OracleCase {
        name: "unconverged",
        sample_id: "oracle-no-convergence",
        metric_name: "residual",
        ground_truth: 1.0,
        generated_value: 1.0,
        reported_relative_error: 0.0,
        converged: false,
        tolerance: 0.0,
        expected_relative_error: Some(0.0),
        expected_acceptable: false,
    },
    OracleCase {
        name: "missing-sample-identity",
        sample_id: " ",
        metric_name: "residual",
        ground_truth: 1.0,
        generated_value: 1.0,
        reported_relative_error: 0.0,
        converged: true,
        tolerance: 0.0,
        expected_relative_error: Some(0.0),
        expected_acceptable: false,
    },
    OracleCase {
        name: "non-finite-reference",
        sample_id: "oracle-nan-reference",
        metric_name: "residual",
        ground_truth: f64::NAN,
        generated_value: 1.0,
        reported_relative_error: 0.0,
        converged: true,
        tolerance: 1.0,
        expected_relative_error: None,
        expected_acceptable: false,
    },
    OracleCase {
        name: "non-finite-output",
        sample_id: "oracle-infinite-output",
        metric_name: "residual",
        ground_truth: 1.0,
        generated_value: f64::INFINITY,
        reported_relative_error: 0.0,
        converged: true,
        tolerance: 1.0,
        expected_relative_error: None,
        expected_acceptable: false,
    },
    OracleCase {
        name: "subtraction-overflow",
        sample_id: "oracle-overflow",
        metric_name: "signed_value",
        ground_truth: -f64::MAX,
        generated_value: f64::MAX,
        reported_relative_error: 0.0,
        converged: true,
        tolerance: 1.0,
        expected_relative_error: None,
        expected_acceptable: false,
    },
    OracleCase {
        name: "invalid-negative-tolerance",
        sample_id: "oracle-negative-tolerance",
        metric_name: "residual",
        ground_truth: 1.0,
        generated_value: 1.0,
        reported_relative_error: 0.0,
        converged: true,
        tolerance: -0.01,
        expected_relative_error: Some(0.0),
        expected_acceptable: false,
    },
];

#[test]
fn hand_authored_oracle_cases_match_error_and_acceptance() {
    for case in CASES {
        let result = BenchmarkValidation {
            sample_id: case.sample_id.to_string(),
            metric_name: case.metric_name.to_string(),
            ground_truth: case.ground_truth,
            generated_value: case.generated_value,
            relative_error: case.reported_relative_error,
            converged: case.converged,
        };

        let actual_error = result.computed_relative_error();
        match (case.expected_relative_error, actual_error) {
            (Some(expected), Some(actual)) => {
                let scale = expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= 2.0e-14 * scale,
                    "{}: expected relative error {:?}, got {:?}",
                    case.name,
                    expected,
                    actual
                );
            }
            (None, None) => {}
            (expected, actual) => panic!(
                "{}: expected relative error {:?}, got {:?}",
                case.name, expected, actual
            ),
        }

        assert_eq!(
            result.is_acceptable(case.tolerance),
            case.expected_acceptable,
            "{}: acceptance mismatch",
            case.name
        );
    }
}
