// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_core::double_integrator_reachability::{
    BoundedDoubleIntegrator1D, DoubleIntegratorQuery1D,
    DoubleIntegratorReachabilityResult1D, DoubleIntegratorSeparator1D,
    DoubleIntegratorState1D, TimedAccelerationSegment1D, TimedAccelerationTrajectory1D,
    double_integrator_terminal_slice, replay_double_integrator_trajectory,
    solve_double_integrator_analytic, verify_double_integrator_infeasibility,
};
use symthaea_core::kinodynamic_reachability::{
    ControlInterval1D, DynamicsReplayPolicy1D, DynamicsSemantics, DynamicsTimeDomain,
    KinodynamicError, PlantTimeProfile,
};
use symthaea_core::reachability::UnknownReason;

fn state(position: f64, velocity: f64) -> DoubleIntegratorState1D {
    DoubleIntegratorState1D::new(position, velocity).unwrap()
}

fn symmetric_model() -> BoundedDoubleIntegrator1D {
    BoundedDoubleIntegrator1D::new(ControlInterval1D::new(-1.0, 1.0).unwrap())
}

fn symmetric_query(position: f64, velocity: f64) -> DoubleIntegratorQuery1D {
    let model = symmetric_model();
    DoubleIntegratorQuery1D::new(
        &model,
        state(0.0, 0.0),
        state(position, velocity),
        PlantTimeProfile::new(2.0).unwrap(),
        DynamicsReplayPolicy1D::reference_v1(),
    )
}

#[test]
fn symmetric_rest_to_rest_slice_matches_closed_form_reference() {
    let model = symmetric_model();
    let query = symmetric_query(0.0, 0.0);
    let slice = double_integrator_terminal_slice(&model, &query).unwrap();
    assert_eq!(slice.velocity_lower(), -2.0);
    assert_eq!(slice.velocity_upper(), 2.0);
    assert_eq!(slice.high_acceleration_duration(), 1.0);
    assert_eq!(slice.position_lower(), -1.0);
    assert_eq!(slice.position_upper(), 1.0);
}

#[test]
fn reachable_target_gets_replay_valid_bang_bang_witness() {
    let model = symmetric_model();
    let query = symmetric_query(0.5, 0.0);
    match solve_double_integrator_analytic(&model, &query).unwrap() {
        DoubleIntegratorReachabilityResult1D::Feasible { trajectory, replay } => {
            assert_eq!(trajectory.segments().len(), 3);
            assert_eq!(replay.trajectory_identity(), trajectory.identity());
            assert_eq!(replay.segment_count(), 3);
            let tolerance = query.replay_policy().absolute_state_tolerance();
            assert!((replay.final_state().position() - 0.5).abs() <= tolerance);
            assert!(replay.final_state().velocity().abs() <= tolerance);
            assert!(replay.maximum_position_residual() <= tolerance);
            assert!(replay.maximum_velocity_residual() <= tolerance);
        }
        other => panic!("expected feasible result, got {other:?}"),
    }
}

#[test]
fn independent_replay_accepts_valid_noncanonical_multi_switch_witness() {
    let model = symmetric_model();
    let query = symmetric_query(0.0, 0.0);

    // This +1/-1/+1 schedule is not the solver's canonical single-high-interval
    // construction for the same target. It independently returns to rest at x=0.
    let trajectory = TimedAccelerationTrajectory1D::new(
        &model,
        &query,
        vec![
            TimedAccelerationSegment1D::new(
                0.0,
                0.5,
                1.0,
                state(0.0, 0.0),
                state(0.125, 0.5),
            ),
            TimedAccelerationSegment1D::new(
                0.5,
                1.5,
                -1.0,
                state(0.125, 0.5),
                state(0.125, -0.5),
            ),
            TimedAccelerationSegment1D::new(
                1.5,
                2.0,
                1.0,
                state(0.125, -0.5),
                state(0.0, 0.0),
            ),
        ],
    )
    .unwrap();

    let replay = replay_double_integrator_trajectory(&model, &query, &trajectory).unwrap();
    assert_eq!(replay.segment_count(), 3);
    assert_eq!(replay.final_state(), state(0.0, 0.0));
    assert_eq!(replay.maximum_position_residual(), 0.0);
    assert_eq!(replay.maximum_velocity_residual(), 0.0);
}

#[test]
fn velocity_outside_impulse_bound_gets_verified_certificate() {
    let model = symmetric_model();
    let query = symmetric_query(0.0, 3.0);
    match solve_double_integrator_analytic(&model, &query).unwrap() {
        DoubleIntegratorReachabilityResult1D::CertifiedInfeasible {
            certificate,
            verification,
        } => {
            assert_eq!(certificate.separator(), DoubleIntegratorSeparator1D::VelocityAbove);
            assert_eq!(certificate.terminal_slice().velocity_upper(), 2.0);
            assert_eq!(verification.certificate_identity(), certificate.identity());
            let replayed =
                verify_double_integrator_infeasibility(&model, &query, &certificate).unwrap();
            assert_eq!(replayed.certificate_identity(), certificate.identity());
        }
        other => panic!("expected certified velocity infeasibility, got {other:?}"),
    }
}

#[test]
fn feasible_velocity_but_impossible_position_gets_verified_moment_certificate() {
    let model = symmetric_model();
    let query = symmetric_query(2.0, 0.0);
    match solve_double_integrator_analytic(&model, &query).unwrap() {
        DoubleIntegratorReachabilityResult1D::CertifiedInfeasible {
            certificate,
            verification,
        } => {
            assert_eq!(certificate.separator(), DoubleIntegratorSeparator1D::PositionAbove);
            assert_eq!(certificate.terminal_slice().position_upper(), 1.0);
            assert_eq!(verification.certificate_identity(), certificate.identity());
        }
        other => panic!("expected certified position infeasibility, got {other:?}"),
    }
}

#[test]
fn bang_bang_degenerate_cases_p_zero_p_horizon_and_fixed_control_are_explicit() {
    let model = symmetric_model();

    let p_zero = symmetric_query(-2.0, -2.0);
    assert_eq!(
        double_integrator_terminal_slice(&model, &p_zero)
            .unwrap()
            .high_acceleration_duration(),
        0.0
    );
    match solve_double_integrator_analytic(&model, &p_zero).unwrap() {
        DoubleIntegratorReachabilityResult1D::Feasible { trajectory, .. } => {
            assert_eq!(trajectory.segments().len(), 1);
            assert_eq!(trajectory.segments()[0].acceleration(), -1.0);
        }
        other => panic!("expected p=0 feasible constant-control witness, got {other:?}"),
    }

    let p_horizon = symmetric_query(2.0, 2.0);
    assert_eq!(
        double_integrator_terminal_slice(&model, &p_horizon)
            .unwrap()
            .high_acceleration_duration(),
        2.0
    );
    match solve_double_integrator_analytic(&model, &p_horizon).unwrap() {
        DoubleIntegratorReachabilityResult1D::Feasible { trajectory, .. } => {
            assert_eq!(trajectory.segments().len(), 1);
            assert_eq!(trajectory.segments()[0].acceleration(), 1.0);
        }
        other => panic!("expected p=T feasible constant-control witness, got {other:?}"),
    }

    let fixed = BoundedDoubleIntegrator1D::new(ControlInterval1D::new(0.5, 0.5).unwrap());
    let fixed_query = DoubleIntegratorQuery1D::new(
        &fixed,
        state(0.0, 0.0),
        state(1.0, 1.0),
        PlantTimeProfile::new(2.0).unwrap(),
        DynamicsReplayPolicy1D::reference_v1(),
    );
    match solve_double_integrator_analytic(&fixed, &fixed_query).unwrap() {
        DoubleIntegratorReachabilityResult1D::Feasible { trajectory, .. } => {
            assert_eq!(trajectory.segments().len(), 1);
            assert_eq!(trajectory.segments()[0].acceleration(), 0.5);
        }
        other => panic!("expected fixed-control feasible witness, got {other:?}"),
    }
}

#[test]
fn velocity_boundary_guard_returns_unknown_instead_of_infeasibility() {
    let model = symmetric_model();
    let policy = DynamicsReplayPolicy1D::reference_v1();
    let query = DoubleIntegratorQuery1D::new(
        &model,
        state(0.0, 0.0),
        state(2.0, 2.0 + 0.5 * policy.absolute_state_tolerance()),
        PlantTimeProfile::new(2.0).unwrap(),
        policy,
    );
    assert!(matches!(
        solve_double_integrator_analytic(&model, &query).unwrap(),
        DoubleIntegratorReachabilityResult1D::Unknown {
            reason: UnknownReason::NumericalFailure,
            ..
        }
    ));
}

#[test]
fn position_moment_boundary_guard_returns_unknown_instead_of_infeasibility() {
    let model = symmetric_model();
    let policy = DynamicsReplayPolicy1D::reference_v1();
    let query = DoubleIntegratorQuery1D::new(
        &model,
        state(0.0, 0.0),
        state(1.0 + 0.5 * policy.absolute_state_tolerance(), 0.0),
        PlantTimeProfile::new(2.0).unwrap(),
        policy,
    );
    assert!(matches!(
        solve_double_integrator_analytic(&model, &query).unwrap(),
        DoubleIntegratorReachabilityResult1D::Unknown {
            reason: UnknownReason::NumericalFailure,
            ..
        }
    ));
}

#[test]
fn replay_rejects_acceleration_outside_exact_control_interval() {
    let model = symmetric_model();
    let query = symmetric_query(0.5, 0.0);
    let bad = TimedAccelerationTrajectory1D::new(
        &model,
        &query,
        vec![TimedAccelerationSegment1D::new(
            0.0,
            2.0,
            2.0,
            state(0.0, 0.0),
            state(0.5, 0.0),
        )],
    )
    .unwrap();
    assert!(matches!(
        replay_double_integrator_trajectory(&model, &query, &bad),
        Err(KinodynamicError::InvalidTrajectory { .. })
    ));
}

#[test]
fn replay_rejects_declared_phase_state_discontinuity() {
    let model = symmetric_model();
    let query = symmetric_query(0.0, 0.0);
    let bad = TimedAccelerationTrajectory1D::new(
        &model,
        &query,
        vec![
            TimedAccelerationSegment1D::new(
                0.0,
                1.0,
                -1.0,
                state(0.0, 0.0),
                state(-0.5, -1.0),
            ),
            TimedAccelerationSegment1D::new(
                1.0,
                2.0,
                1.0,
                state(-0.4, -1.0),
                state(0.0, 0.0),
            ),
        ],
    )
    .unwrap();
    assert!(matches!(
        replay_double_integrator_trajectory(&model, &query, &bad),
        Err(KinodynamicError::InvalidTrajectory { .. })
    ));
}

#[test]
fn wrong_final_plant_horizon_fails_even_when_endpoint_state_is_otherwise_valid() {
    let model = symmetric_model();
    let query = symmetric_query(0.5, 1.0);
    let short = TimedAccelerationTrajectory1D::new(
        &model,
        &query,
        vec![TimedAccelerationSegment1D::new(
            0.0,
            1.0,
            1.0,
            state(0.0, 0.0),
            state(0.5, 1.0),
        )],
    )
    .unwrap();
    assert!(matches!(
        replay_double_integrator_trajectory(&model, &query, &short),
        Err(KinodynamicError::InvalidTrajectory { .. })
    ));
}

#[test]
fn plant_time_theorem_has_no_cognitive_cadence_input() {
    let controls = ControlInterval1D::new(-1.0, 1.0).unwrap();
    let a = BoundedDoubleIntegrator1D::new(controls);
    let b = BoundedDoubleIntegrator1D::new(controls);
    assert_eq!(a.identity(), b.identity());
    assert_eq!(a.semantics(), DynamicsSemantics::ContinuousTime);
    assert_eq!(a.time_domain(), DynamicsTimeDomain::PlantModelSeconds);
    let time = PlantTimeProfile::new(2.0).unwrap();
    assert_eq!(time.domain(), DynamicsTimeDomain::PlantModelSeconds);
    assert_eq!(time.horizon_seconds(), 2.0);
}

#[test]
fn geometric_straight_line_in_phase_space_is_not_promoted_to_dynamic_trajectory() {
    let model = symmetric_model();
    let query = symmetric_query(1.0, 0.0);
    let geometric_only = TimedAccelerationTrajectory1D::new(
        &model,
        &query,
        vec![TimedAccelerationSegment1D::new(
            0.0,
            2.0,
            0.0,
            state(0.0, 0.0),
            state(1.0, 0.0),
        )],
    )
    .unwrap();
    assert!(matches!(
        replay_double_integrator_trajectory(&model, &query, &geometric_only),
        Err(KinodynamicError::InvalidTrajectory { .. })
    ));
}

#[test]
fn signed_zero_does_not_split_state_or_query_identity() {
    let model = symmetric_model();
    let time = PlantTimeProfile::new(2.0).unwrap();
    let policy = DynamicsReplayPolicy1D::reference_v1();
    let qa = DoubleIntegratorQuery1D::new(
        &model,
        state(-0.0, 0.0),
        state(0.0, -0.0),
        time,
        policy,
    );
    let qb = DoubleIntegratorQuery1D::new(
        &model,
        state(0.0, -0.0),
        state(-0.0, 0.0),
        time,
        policy,
    );
    assert_eq!(qa.identity(), qb.identity());
}

#[test]
fn non_finite_phase_state_fails_closed() {
    assert!(matches!(
        DoubleIntegratorState1D::new(f64::NAN, 0.0),
        Err(KinodynamicError::InvalidQuery { .. })
    ));
    assert!(matches!(
        DoubleIntegratorState1D::new(0.0, f64::INFINITY),
        Err(KinodynamicError::InvalidQuery { .. })
    ));
}
