#![forbid(unsafe_code)]
#![deny(warnings)]

//! Exact-version analytic validation of the Rust SALib 0.3.0 implementation.
//!
//! The reference cases are analytic and independent of the estimator.
//! These tests validate a narrowly selected Saltelli/Sobol path, not every
//! method advertised by the crate.

use std::f64::consts::PI;

use salib::estimators::estimate_saltelli2010;
use salib::samplers::{SobolSampler, build_saltelli_matrix};
use salib::{Distribution, ProblemBuilder, RngState};

const BASE_SAMPLES: usize = 8_192;

fn ishigami_problem() -> salib::Problem {
    ProblemBuilder::new()
        .factor(
            "x1",
            Distribution::Uniform {
                lo: -PI,
                hi: PI,
            },
        )
        .factor(
            "x2",
            Distribution::Uniform {
                lo: -PI,
                hi: PI,
            },
        )
        .factor(
            "x3",
            Distribution::Uniform {
                lo: -PI,
                hi: PI,
            },
        )
        .build()
        .expect("valid three-factor Ishigami problem")
}

fn ishigami_indices(seed: [u8; 32]) -> salib::estimators::SobolIndices {
    let problem = ishigami_problem();
    let sampler = SobolSampler::minimal(2 * problem.dim());
    let mut rng = RngState::from_seed(seed);
    let design = build_saltelli_matrix(&sampler, BASE_SAMPLES, false, &mut rng)
        .expect("valid Saltelli design");

    estimate_saltelli2010(&design, |unit_cube| {
        let x: Vec<f64> = problem
            .factors()
            .iter()
            .zip(unit_cube)
            .map(|(factor, &u)| factor.distribution.quantile(u))
            .collect();

        x[0].sin() + 7.0 * x[1].sin().powi(2) + 0.1 * x[2].powi(4) * x[0].sin()
    })
}

#[test]
fn saltelli_design_has_expected_evaluation_budget() {
    let problem = ishigami_problem();
    let sampler = SobolSampler::minimal(2 * problem.dim());
    let mut rng = RngState::from_seed([0; 32]);
    let design = build_saltelli_matrix(&sampler, BASE_SAMPLES, false, &mut rng)
        .expect("valid Saltelli design");

    // First/total-order Saltelli design: N * (d + 2), no second-order blocks.
    assert_eq!(design.total_evaluations(), BASE_SAMPLES * (problem.dim() + 2));
}

#[test]
fn ishigami_sobol_indices_match_analytic_reference_with_sampling_tolerance() {
    let indices = ishigami_indices([0; 32]);

    // Published analytic Ishigami references:
    // S1 ~= [0.3139, 0.4424, 0.0000]
    // ST ~= [0.5576, 0.4424, 0.2437]
    let expected_first = [0.3139, 0.4424, 0.0];
    let expected_total = [0.5576, 0.4424, 0.2437];
    let tolerance = 0.03;

    for i in 0..3 {
        assert!(
            (indices.first_order[i] - expected_first[i]).abs() <= tolerance,
            "S1[{}]={}, expected {}, tolerance {}",
            i,
            indices.first_order[i],
            expected_first[i],
            tolerance
        );
        assert!(
            (indices.total_order[i] - expected_total[i]).abs() <= tolerance,
            "ST[{}]={}, expected {}, tolerance {}",
            i,
            indices.total_order[i],
            expected_total[i],
            tolerance
        );
    }

    // x3 contributes through an x1-x3 interaction: its main effect is near
    // zero while its total-order effect is materially positive.
    assert!(indices.first_order[2].abs() <= tolerance);
    assert!(indices.total_order[2] > 0.15);
}

#[test]
fn fixed_rng_state_reproduces_identical_indices() {
    let first = ishigami_indices([0x5a; 32]);
    let second = ishigami_indices([0x5a; 32]);

    assert_eq!(first.first_order, second.first_order);
    assert_eq!(first.total_order, second.total_order);
}
