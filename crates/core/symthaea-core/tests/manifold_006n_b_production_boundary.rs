// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MANIFOLD-006N-B production-boundary regressions.
//!
//! These fixtures are intentionally outside the frozen 006C1 vector corpus. They
//! prove that production robust-reachability proof arithmetic now consumes the
//! hardened certified-interval kernel at the exact normal/subnormal boundary that
//! defeated the original N-A extraction.

use symthaea_core::kinodynamic_reachability::{
    BoundedSingleIntegrator1D, ControlInterval1D, PlantTimeProfile,
};
use symthaea_core::reachability::UnknownReason;
use symthaea_core::robust_reachability::{
    BoundedDisturbance1D, RobustReachabilityPolicy1D, RobustReachabilityResult1D,
    RobustSingleIntegratorQuery1D, TerminalInterval1D, fixed_control_terminal_envelope,
    robust_control_requirement, solve_robust_single_integrator_analytic,
};

fn fixed_model(control: f64) -> BoundedSingleIntegrator1D {
    BoundedSingleIntegrator1D::new(ControlInterval1D::new(control, control).unwrap())
}

fn zero_disturbance() -> BoundedDisturbance1D {
    BoundedDisturbance1D::new(0.0, 0.0).unwrap()
}

#[test]
fn production_multiplication_widens_original_normal_subnormal_counterexample() {
    let one_down = f64::from_bits(1.0_f64.to_bits() - 1);
    let model = fixed_model(one_down);
    let query = RobustSingleIntegratorQuery1D::new(
        &model,
        0.0,
        TerminalInterval1D::new(-1.0, 1.0).unwrap(),
        zero_disturbance(),
        PlantTimeProfile::new(f64::MIN_POSITIVE).unwrap(),
        RobustReachabilityPolicy1D::new(0.0).unwrap(),
    )
    .unwrap();

    let envelope = fixed_control_terminal_envelope(&model, &query, one_down).unwrap();

    // Exact product:
    //   next_down(1.0) * MIN_POSITIVE
    // lies below MIN_POSITIVE, although nearest binary64 rounds upward to it.
    // Production proof authority must therefore widen around that rounded value.
    assert_eq!(
        envelope.lower().to_bits(),
        f64::MIN_POSITIVE.to_bits() - 1,
        "lower enclosure must reach max subnormal"
    );
    assert_eq!(
        envelope.upper().to_bits(),
        f64::MIN_POSITIVE.to_bits() + 1,
        "upper enclosure must reach next_up(MIN_POSITIVE)"
    );
    assert_ne!(envelope.lower().to_bits(), envelope.upper().to_bits());
}

#[test]
fn production_division_boundary_blocks_false_robust_feasible_promotion() {
    let model = fixed_model(f64::MIN_POSITIVE);
    let target_lower = f64::from_bits((2.0 * f64::MIN_POSITIVE).to_bits() - 1);
    let query = RobustSingleIntegratorQuery1D::new(
        &model,
        0.0,
        TerminalInterval1D::new(target_lower, 1.0).unwrap(),
        zero_disturbance(),
        PlantTimeProfile::new(2.0).unwrap(),
        RobustReachabilityPolicy1D::new(0.0).unwrap(),
    )
    .unwrap();

    // The nearest-rounded diagnostic intentionally remains unchanged: the exact
    // lower target displacement divided by two rounds to MIN_POSITIVE, making the
    // diagnostic interval appear actuator-feasible.
    let diagnostic = robust_control_requirement(&model, &query).unwrap();
    assert_eq!(
        diagnostic.required_lower().to_bits(),
        f64::MIN_POSITIVE.to_bits()
    );
    assert!(diagnostic.admissible_nonempty());

    // Proof authority is different. Exact-rational A1 qualification established
    // that the quotient lies below MIN_POSITIVE, so the certified requirement must
    // widen. With the actuator fixed exactly at MIN_POSITIVE there is then no
    // universally certified control. The solver must refuse the old false-positive
    // promotion rather than treating the rounded normal quotient as exact.
    match solve_robust_single_integrator_analytic(&model, &query).unwrap() {
        RobustReachabilityResult1D::Unknown { reason, .. } => {
            assert_eq!(reason, UnknownReason::NumericalFailure);
        }
        other => panic!(
            "normal/subnormal division boundary must not promote RobustFeasible; got {other:?}"
        ),
    }
}
