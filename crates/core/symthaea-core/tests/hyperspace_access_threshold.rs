// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact closed-boundary control for HYPERSPACE-001 fourth-coordinate access.
//!
//! The shell obstacle is closed in `w`: `|w| <= W`. Therefore a motion bound
//! exactly equal to `W` remains analytically blocked. A binary-exact positive
//! margin beyond `W` admits a constructive R4 escape while xyz shell geometry,
//! start and goal remain unchanged.

use symthaea_core::continuous_reachability::{
    ContinuousPathReplay, ContinuousValidityOracle, validate_euclidean_path,
};
use symthaea_core::hyperspace_benchmark::{
    FiniteWShellOracle, HyperspaceBenchmarkError, HyperspaceDimension,
    canonical_shell_problem, certify_radial_separation, evaluator_reference_escape_path,
    qualify_projection_trap,
};

const W_SHELL: f64 = 0.25; // 2^-2
const ACCESS_INCREMENT: f64 = 0.000_976_562_5; // 2^-10
const CLEARANCE_MARGIN: f64 = 0.000_488_281_25; // 2^-11

#[test]
fn equality_is_blocked_but_binary_exact_positive_margin_is_feasible() {
    let blocked_bound = W_SHELL;
    let feasible_bound = W_SHELL + ACCESS_INCREMENT;
    let witness_w = W_SHELL + CLEARANCE_MARGIN;

    assert!(witness_w > W_SHELL);
    assert!(witness_w < feasible_bound);

    let blocked_oracle = FiniteWShellOracle::new(
        HyperspaceDimension::R4,
        1.0,
        2.0,
        W_SHELL,
        4.0,
        blocked_bound,
    )
    .unwrap();
    let blocked_problem = canonical_shell_problem(&blocked_oracle, 3.0).unwrap();
    let blocked_certificate =
        certify_radial_separation(&blocked_problem, &blocked_oracle).unwrap();

    let feasible_oracle = FiniteWShellOracle::new(
        HyperspaceDimension::R4,
        1.0,
        2.0,
        W_SHELL,
        4.0,
        feasible_bound,
    )
    .unwrap();
    let feasible_problem = canonical_shell_problem(&feasible_oracle, 3.0).unwrap();

    assert!(matches!(
        certify_radial_separation(&feasible_problem, &feasible_oracle),
        Err(HyperspaceBenchmarkError::SeparationNotApplicable { .. })
    ));

    // The intervention changes only accessible w extent, not shell geometry.
    assert_eq!(
        blocked_oracle.inner_radius().to_bits(),
        feasible_oracle.inner_radius().to_bits()
    );
    assert_eq!(
        blocked_oracle.outer_radius().to_bits(),
        feasible_oracle.outer_radius().to_bits()
    );
    assert_eq!(
        blocked_oracle.w_obstacle_half_thickness().to_bits(),
        feasible_oracle.w_obstacle_half_thickness().to_bits()
    );
    assert_eq!(
        blocked_oracle.xyz_bound().to_bits(),
        feasible_oracle.xyz_bound().to_bits()
    );
    assert_ne!(
        blocked_oracle.profile().identity(),
        feasible_oracle.profile().identity()
    );
    assert_ne!(blocked_problem.identity(), feasible_problem.identity());
    assert_ne!(blocked_certificate.identity(), [0; 32]);

    let path = evaluator_reference_escape_path(
        &feasible_problem,
        &feasible_oracle,
        CLEARANCE_MARGIN,
    )
    .unwrap();

    let validation = match validate_euclidean_path(&feasible_problem, &feasible_oracle, &path)
        .unwrap()
    {
        ContinuousPathReplay::Valid(validation) => validation,
        other => panic!("above-threshold reference path must replay valid, got {other:?}"),
    };
    assert!(validation.total_cost().is_finite());

    let max_abs_w = path
        .iter()
        .map(|state| state[3].abs())
        .fold(0.0_f64, f64::max);
    assert_eq!(max_abs_w.to_bits(), witness_w.to_bits());

    // The same valid R4 path must still collide after projection to xyz.
    let projection = qualify_projection_trap(&feasible_problem, &feasible_oracle, &path).unwrap();
    assert_eq!(projection.full_path_identity(), validation.path_identity());
}
