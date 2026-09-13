// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_core::kinodynamic_reachability::{
    BoundedSingleIntegrator1D, ControlInterval1D, PlantTimeProfile,
};
use symthaea_core::reachability::UnknownReason;
use symthaea_core::robust_reachability::{
    BoundedDisturbance1D, DisturbanceQuantifier1D, NotRobustReason1D,
    RobustReachabilityError, RobustReachabilityPolicy1D, RobustReachabilityResult1D,
    RobustSingleIntegratorQuery1D, TerminalInterval1D, existential_terminal_interval,
    fixed_control_terminal_envelope, robust_control_requirement,
    solve_robust_single_integrator_analytic, verify_not_robust_certificate,
    verify_robust_witness,
};

fn model() -> BoundedSingleIntegrator1D {
    BoundedSingleIntegrator1D::new(ControlInterval1D::new(-1.0, 1.0).unwrap())
}

fn query(
    target_lower: f64,
    target_upper: f64,
    disturbance_lower: f64,
    disturbance_upper: f64,
    horizon: f64,
) -> RobustSingleIntegratorQuery1D {
    let model = model();
    RobustSingleIntegratorQuery1D::new(
        &model,
        0.0,
        TerminalInterval1D::new(target_lower, target_upper).unwrap(),
        BoundedDisturbance1D::new(disturbance_lower, disturbance_upper).unwrap(),
        PlantTimeProfile::new(horizon).unwrap(),
        RobustReachabilityPolicy1D::reference_v1(),
    )
    .unwrap()
}

#[test]
fn fixed_control_envelope_matches_exact_disturbance_extrema() {
    let model = model();
    let query = query(-0.5, 0.5, -0.25, 0.25, 2.0);
    let envelope = fixed_control_terminal_envelope(&model, &query, 0.0).unwrap();
    assert_eq!(envelope.lower(), -0.5);
    assert_eq!(envelope.upper(), 0.5);
    assert_eq!(query.disturbance().quantifier(), DisturbanceQuantifier1D::UniversalAdversarial);
}

#[test]
fn wide_target_gets_independently_verified_robust_witness() {
    let model = model();
    let query = query(-0.5, 0.5, -0.25, 0.25, 2.0);

    match solve_robust_single_integrator_analytic(&model, &query).unwrap() {
        RobustReachabilityResult1D::RobustFeasible { witness, verification } => {
            assert_eq!(witness.control(), 0.0);
            assert_eq!(witness.terminal_envelope().lower(), -0.5);
            assert_eq!(witness.terminal_envelope().upper(), 0.5);
            assert_eq!(verification.witness_identity(), witness.identity());
            let replayed = verify_robust_witness(&model, &query, &witness).unwrap();
            assert_eq!(replayed.witness_identity(), witness.identity());
        }
        other => panic!("expected robust feasible result, got {other:?}"),
    }
}

#[test]
fn existential_reachability_does_not_imply_robust_reachability() {
    let model = model();
    let query = query(-0.1, 0.1, -0.25, 0.25, 2.0);
    let possible = existential_terminal_interval(&model, &query).unwrap();
    assert!(possible.intersection(query.target()).is_some());

    match solve_robust_single_integrator_analytic(&model, &query).unwrap() {
        RobustReachabilityResult1D::CertifiedNotRobust { certificate, verification } => {
            assert_eq!(certificate.reason(), NotRobustReason1D::TargetTooNarrow);
            assert_eq!(verification.certificate_identity(), certificate.identity());
        }
        other => panic!("expected exact not-robust result, got {other:?}"),
    }
}

#[test]
fn target_too_narrow_certificate_independently_replays() {
    let model = model();
    let query = query(-0.1, 0.1, -0.25, 0.25, 2.0);

    match solve_robust_single_integrator_analytic(&model, &query).unwrap() {
        RobustReachabilityResult1D::CertifiedNotRobust { certificate, .. } => {
            assert!(certificate.requirement().required_lower() > certificate.requirement().required_upper());
            let replayed = verify_not_robust_certificate(&model, &query, &certificate).unwrap();
            assert_eq!(replayed.certificate_identity(), certificate.identity());
        }
        other => panic!("expected target-width separator, got {other:?}"),
    }
}

#[test]
fn required_robust_control_above_actuator_bound_gets_verified_certificate() {
    let model = model();
    let query = query(2.0, 3.0, 0.0, 0.0, 1.0);

    match solve_robust_single_integrator_analytic(&model, &query).unwrap() {
        RobustReachabilityResult1D::CertifiedNotRobust { certificate, .. } => {
            assert_eq!(certificate.reason(), NotRobustReason1D::RequiresControlAboveMaximum);
            verify_not_robust_certificate(&model, &query, &certificate).unwrap();
        }
        other => panic!("expected above-actuator separator, got {other:?}"),
    }
}

#[test]
fn required_robust_control_below_actuator_bound_gets_verified_certificate() {
    let model = model();
    let query = query(-3.0, -2.0, 0.0, 0.0, 1.0);

    match solve_robust_single_integrator_analytic(&model, &query).unwrap() {
        RobustReachabilityResult1D::CertifiedNotRobust { certificate, .. } => {
            assert_eq!(certificate.reason(), NotRobustReason1D::RequiresControlBelowMinimum);
            verify_not_robust_certificate(&model, &query, &certificate).unwrap();
        }
        other => panic!("expected below-actuator separator, got {other:?}"),
    }
}

#[test]
fn near_actuator_boundary_returns_unknown_instead_of_impossibility() {
    let model = model();
    let policy = RobustReachabilityPolicy1D::reference_v1();
    let epsilon = 0.5 * policy.absolute_control_tolerance();
    let query = RobustSingleIntegratorQuery1D::new(
        &model,
        0.0,
        TerminalInterval1D::new(1.0 + epsilon, 2.0).unwrap(),
        BoundedDisturbance1D::new(0.0, 0.0).unwrap(),
        PlantTimeProfile::new(1.0).unwrap(),
        policy,
    )
    .unwrap();

    assert!(matches!(
        solve_robust_single_integrator_analytic(&model, &query).unwrap(),
        RobustReachabilityResult1D::Unknown {
            reason: UnknownReason::NumericalFailure,
            ..
        }
    ));
}

#[test]
fn zero_width_disturbance_reduces_to_deterministic_point_target() {
    let model = model();
    let query = query(1.0, 1.0, 0.0, 0.0, 2.0);
    let requirement = robust_control_requirement(&model, &query).unwrap();
    assert_eq!(requirement.required_lower(), 0.5);
    assert_eq!(requirement.required_upper(), 0.5);

    match solve_robust_single_integrator_analytic(&model, &query).unwrap() {
        RobustReachabilityResult1D::RobustFeasible { witness, .. } => {
            assert_eq!(witness.control(), 0.5);
            assert_eq!(witness.terminal_envelope().lower(), 1.0);
            assert_eq!(witness.terminal_envelope().upper(), 1.0);
        }
        other => panic!("expected deterministic reduction to remain robust feasible, got {other:?}"),
    }
}

#[test]
fn widening_adversarial_disturbance_monotonically_shrinks_robust_controls() {
    let model = model();
    let narrow = query(-0.5, 0.5, -0.1, 0.1, 2.0);
    let medium = query(-0.5, 0.5, -0.25, 0.25, 2.0);
    let wide = query(-0.5, 0.5, -0.3, 0.3, 2.0);

    let a = robust_control_requirement(&model, &narrow).unwrap();
    let b = robust_control_requirement(&model, &medium).unwrap();
    let c = robust_control_requirement(&model, &wide).unwrap();

    assert_eq!(a.admissible_lower(), -0.15);
    assert_eq!(a.admissible_upper(), 0.15);
    assert_eq!(b.admissible_lower(), 0.0);
    assert_eq!(b.admissible_upper(), 0.0);
    assert!(c.required_lower() > c.required_upper());

    assert!(b.admissible_lower() >= a.admissible_lower());
    assert!(b.admissible_upper() <= a.admissible_upper());
    assert!(matches!(
        solve_robust_single_integrator_analytic(&model, &narrow).unwrap(),
        RobustReachabilityResult1D::RobustFeasible { .. }
    ));
    assert!(matches!(
        solve_robust_single_integrator_analytic(&model, &medium).unwrap(),
        RobustReachabilityResult1D::RobustFeasible { .. }
    ));
    assert!(matches!(
        solve_robust_single_integrator_analytic(&model, &wide).unwrap(),
        RobustReachabilityResult1D::CertifiedNotRobust { .. }
    ));
}

#[test]
fn signed_zero_does_not_split_disturbance_target_or_query_identity() {
    let model = model();
    let da = BoundedDisturbance1D::new(-0.0, 0.25).unwrap();
    let db = BoundedDisturbance1D::new(0.0, 0.25).unwrap();
    assert_eq!(da.identity(), db.identity());

    let ta = TerminalInterval1D::new(-0.0, 1.0).unwrap();
    let tb = TerminalInterval1D::new(0.0, 1.0).unwrap();
    assert_eq!(ta.identity(), tb.identity());

    let time = PlantTimeProfile::new(1.0).unwrap();
    let policy = RobustReachabilityPolicy1D::reference_v1();
    let qa = RobustSingleIntegratorQuery1D::new(&model, -0.0, ta, da, time, policy).unwrap();
    let qb = RobustSingleIntegratorQuery1D::new(&model, 0.0, tb, db, time, policy).unwrap();
    assert_eq!(qa.identity(), qb.identity());
}

#[test]
fn malformed_nonfinite_profiles_fail_closed() {
    assert!(matches!(
        BoundedDisturbance1D::new(f64::NAN, 1.0),
        Err(RobustReachabilityError::InvalidDisturbance { .. })
    ));
    assert!(matches!(
        TerminalInterval1D::new(0.0, f64::INFINITY),
        Err(RobustReachabilityError::InvalidTarget { .. })
    ));
}

#[test]
fn theorem_uses_plant_time_and_has_no_cognitive_cadence_input() {
    let disturbance = BoundedDisturbance1D::new(-0.1, 0.1).unwrap();
    assert_eq!(disturbance.time_domain(), symthaea_core::kinodynamic_reachability::DynamicsTimeDomain::PlantModelSeconds);
    let short = PlantTimeProfile::new(1.0).unwrap();
    let long = PlantTimeProfile::new(2.0).unwrap();
    assert_ne!(short.identity(), long.identity());
}

#[test]
fn universal_boundary_roundoff_returns_unknown_instead_of_minting_robust_feasibility() {
    let model = model();
    let query = query(0.0, 0.03, 0.0, 0.1, 0.3);

    // The input floats are exact binary rationals. Their exact-real product
    // d_max * T is strictly greater than the exact-real value represented by
    // the f64 target upper endpoint 0.03, even though nearest-rounded FMA lands
    // on that same f64. A universal containment verifier must therefore abstain.
    assert!(matches!(
        solve_robust_single_integrator_analytic(&model, &query).unwrap(),
        RobustReachabilityResult1D::Unknown {
            reason: UnknownReason::NumericalFailure,
            ..
        }
    ));
}
