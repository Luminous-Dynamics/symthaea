// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Regression qualification for configurable PC significance thresholds.
//!
//! The fixture constructs two zero-mean, equal-variance signals with a sample
//! correlation chosen so that Fisher's z statistic is approximately 2.2 for
//! n=100. That statistic lies between the two-tailed standard-normal critical
//! values for alpha=0.05 (~1.96) and alpha=0.01 (~2.576).
//!
//! Therefore a correctly wired configurable alpha must:
//! - retain the edge at alpha=0.05 (reject independence), and
//! - remove the edge at alpha=0.01 (fail to reject independence).

use std::f64::consts::TAU;

use symthaea_causal_reasoning::counterfactual::{ObservationalData, PCAlgorithm};

fn boundary_fixture() -> ObservationalData {
    const N: usize = 100;
    // tanh(2.2 / sqrt(97)); Fisher-z ~= 2.2 for n=100, k=0.
    const TARGET_CORRELATION: f64 = 0.219_733_580_616_176_57;

    let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
    let orthogonal_weight = (1.0 - TARGET_CORRELATION * TARGET_CORRELATION).sqrt();

    // On an evenly sampled full period, sin(theta) and cos(theta) are
    // zero-mean, orthogonal, and have equal variance. The linear combination
    // below therefore has correlation TARGET_CORRELATION with X up to floating
    // point roundoff, without random-number or fixture-seed dependence.
    for i in 0..N {
        let theta = TAU * i as f64 / N as f64;
        let x = theta.sin();
        let orthogonal = theta.cos();
        let y = TARGET_CORRELATION * x + orthogonal_weight * orthogonal;
        data.add_observation(vec![x, y]);
    }

    data
}

#[test]
fn configured_alpha_changes_pc_independence_decision() {
    let data = boundary_fixture();

    let alpha_005 = PCAlgorithm::with_alpha(0.05).discover(&data);
    let alpha_001 = PCAlgorithm::with_alpha(0.01).discover(&data);

    assert!(
        alpha_005.skeleton.adjacent(0, 1),
        "z≈2.2 should reject independence at alpha=0.05 and retain the edge"
    );
    assert!(
        !alpha_001.skeleton.adjacent(0, 1),
        "z≈2.2 should fail to reject independence at alpha=0.01 and remove the edge"
    );
}
