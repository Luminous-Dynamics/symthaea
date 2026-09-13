// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_core::kinodynamic_reachability::{
    BoundedSingleIntegrator1D, ControlInterval1D, DynamicsReplayPolicy1D, InfeasibleSide1D,
    KinodynamicError, PlantTimeProfile, SingleIntegratorQuery1D,
    SingleIntegratorReachabilityResult1D, TimedControlSegment1D, TimedControlledTrajectory1D,
    replay_single_integrator_trajectory, single_integrator_reachable_interval,
    solve_single_integrator_analytic, verify_single_integrator_infeasibility,
};
use symthaea_core::reachability::UnknownReason;

fn model() -> BoundedSingleIntegrator1D {
    BoundedSingleIntegrator1D::new(ControlInterval1D::new(-2.0, 3.0).unwrap())
}

fn query(target: f64) -> SingleIntegratorQuery1D {
    let model = model();
    SingleIntegratorQuery1D::new(
        &model,
        1.0,
        target,
        PlantTimeProfile::new(2.0).unwrap(),
        DynamicsReplayPolicy1D::reference_v1(),
    )
    .unwrap()
}

#[test]
fn analytic_reachable_interval_matches_closed_form_reference() {
    let model = model();
    let query = query(5.0);
    let interval = single_integrator_reachable_interval(&model, &query).unwrap();
    assert_eq!(interval.lower(), -3.0);
    assert_eq!(interval.upper(), 7.0);
    assert!(interval.contains(5.0));
}

#[test]
fn reachable_target_gets_independently_replayed_control_witness() {
    let model = model();
    let query = query(5.0);
    match solve_single_integrator_analytic(&model, &query).unwrap() {
        SingleIntegratorReachabilityResult1D::Feasible { trajectory, replay } => {
            assert_eq!(trajectory.segments().len(), 1);
            assert_eq!(trajectory.segments()[0].control(), 2.0);
            assert_eq!(replay.trajectory_identity(), trajectory.identity());
            assert_eq!(replay.final_state(), 5.0);
            assert!(
                replay.maximum_state_residual()
                    <= query.replay_policy().absolute_state_tolerance()
            );
        }
        other => panic!("expected feasible result, got {other:?}"),
    }
}

#[test]
fn exact_upper_reachable_boundary_is_feasible() {
    let model = model();
    let query = query(7.0);
    assert!(matches!(
        solve_single_integrator_analytic(&model, &query).unwrap(),
        SingleIntegratorReachabilityResult1D::Feasible { .. }
    ));
}

#[test]
fn target_well_outside_horizon_gets_independently_verified_certificate() {
    let model = model();
    let query = query(8.0);
    match solve_single_integrator_analytic(&model, &query).unwrap() {
        SingleIntegratorReachabilityResult1D::CertifiedInfeasible {
            certificate,
            verification,
        } => {
            assert_eq!(certificate.side(), InfeasibleSide1D::Above);
            assert_eq!(certificate.reachable_interval().upper(), 7.0);
            assert_eq!(verification.certificate_identity(), certificate.identity());
            let replayed =
                verify_single_integrator_infeasibility(&model, &query, &certificate).unwrap();
            assert_eq!(replayed.certificate_identity(), certificate.identity());
        }
        other => panic!("expected certified infeasible result, got {other:?}"),
    }
}

#[test]
fn numerical_guard_prevents_boundary_roundoff_from_minting_infeasibility() {
    let model = model();
    let base = query(7.0);
    let target = 7.0 + base.replay_policy().absolute_state_tolerance() * 0.5;
    let guarded = SingleIntegratorQuery1D::new(
        &model,
        1.0,
        target,
        base.plant_time(),
        base.replay_policy(),
    )
    .unwrap();
    assert!(matches!(
        solve_single_integrator_analytic(&model, &guarded).unwrap(),
        SingleIntegratorReachabilityResult1D::Unknown {
            reason: UnknownReason::NumericalFailure,
            ..
        }
    ));
}

#[test]
fn replay_rejects_control_bound_violation() {
    let model = model();
    let query = query(5.0);
    let bad = TimedControlledTrajectory1D::new(
        &model,
        &query,
        vec![TimedControlSegment1D::new(0.0, 2.0, 4.0, 1.0, 5.0)],
    )
    .unwrap();
    assert!(matches!(
        replay_single_integrator_trajectory(&model, &query, &bad),
        Err(KinodynamicError::InvalidTrajectory { .. })
    ));
}

#[test]
fn replay_rejects_declared_state_discontinuity() {
    let model = model();
    let query = query(5.0);
    let bad = TimedControlledTrajectory1D::new(
        &model,
        &query,
        vec![
            TimedControlSegment1D::new(0.0, 1.0, 1.0, 1.0, 2.0),
            TimedControlSegment1D::new(1.0, 2.0, 3.0, 2.25, 5.0),
        ],
    )
    .unwrap();
    assert!(matches!(
        replay_single_integrator_trajectory(&model, &query, &bad),
        Err(KinodynamicError::InvalidTrajectory { .. })
    ));
}

#[test]
fn wrong_plant_horizon_is_rejected_even_when_control_and_endpoint_are_valid() {
    let model = model();
    let query = SingleIntegratorQuery1D::new(
        &model,
        1.0,
        3.0,
        PlantTimeProfile::new(2.0).unwrap(),
        DynamicsReplayPolicy1D::reference_v1(),
    )
    .unwrap();

    // x(1) = 1 + 2*1 = 3 with an admissible control. The only defect is that
    // the witness ends at plant time 1 instead of the exact query horizon 2.
    let trajectory = TimedControlledTrajectory1D::new(
        &model,
        &query,
        vec![TimedControlSegment1D::new(0.0, 1.0, 2.0, 1.0, 3.0)],
    )
    .unwrap();

    assert!(matches!(
        replay_single_integrator_trajectory(&model, &query, &trajectory),
        Err(KinodynamicError::InvalidTrajectory { .. })
    ));
}

#[test]
fn independently_replayed_piecewise_control_witness_can_span_full_plant_horizon() {
    let model = model();
    let query = query(5.0);
    let trajectory = TimedControlledTrajectory1D::new(
        &model,
        &query,
        vec![
            TimedControlSegment1D::new(0.0, 1.0, 1.0, 1.0, 2.0),
            TimedControlSegment1D::new(1.0, 2.0, 3.0, 2.0, 5.0),
        ],
    )
    .unwrap();

    let replay = replay_single_integrator_trajectory(&model, &query, &trajectory).unwrap();
    assert_eq!(replay.segment_count(), 2);
    assert_eq!(replay.final_state(), 5.0);
    assert_eq!(replay.maximum_state_residual(), 0.0);
}

#[test]
fn plant_time_identity_binds_exact_physical_horizon() {
    let short = PlantTimeProfile::new(1.0).unwrap();
    let long = PlantTimeProfile::new(2.0).unwrap();
    assert_ne!(short.identity(), long.identity());
    assert_ne!(short.horizon_seconds(), long.horizon_seconds());
}

#[test]
fn signed_zero_does_not_split_control_or_query_identity() {
    let controls_a = ControlInterval1D::new(-0.0, 1.0).unwrap();
    let controls_b = ControlInterval1D::new(0.0, 1.0).unwrap();
    assert_eq!(controls_a.identity(), controls_b.identity());

    let model_a = BoundedSingleIntegrator1D::new(controls_a);
    let model_b = BoundedSingleIntegrator1D::new(controls_b);
    assert_eq!(model_a.identity(), model_b.identity());

    let time = PlantTimeProfile::new(1.0).unwrap();
    let policy = DynamicsReplayPolicy1D::reference_v1();
    let qa = SingleIntegratorQuery1D::new(&model_a, -0.0, 1.0, time, policy).unwrap();
    let qb = SingleIntegratorQuery1D::new(&model_b, 0.0, 1.0, time, policy).unwrap();
    assert_eq!(qa.identity(), qb.identity());
}

#[test]
fn non_finite_query_fails_closed() {
    let model = BoundedSingleIntegrator1D::new(ControlInterval1D::new(-1.0, 1.0).unwrap());
    assert!(matches!(
        SingleIntegratorQuery1D::new(
            &model,
            0.0,
            f64::NAN,
            PlantTimeProfile::new(1.0).unwrap(),
            DynamicsReplayPolicy1D::reference_v1(),
        ),
        Err(KinodynamicError::InvalidQuery { .. })
    ));
}
