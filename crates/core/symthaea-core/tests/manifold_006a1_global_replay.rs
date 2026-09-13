// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_core::kinodynamic_reachability::{
    BoundedSingleIntegrator1D, ControlInterval1D, DynamicsReplayPolicy1D, KinodynamicError,
    MAX_SINGLE_INTEGRATOR_REPLAY_SEGMENTS, PlantTimeProfile, SingleIntegratorQuery1D,
    TimedControlSegment1D, TimedControlledTrajectory1D, replay_single_integrator_trajectory,
};

#[test]
fn replay_rejects_sub_tolerance_drift_accumulation_across_segments() {
    let model = BoundedSingleIntegrator1D::new(ControlInterval1D::new(-1.0, 1.0).unwrap());
    let policy = DynamicsReplayPolicy1D::reference_v1();
    let step = 0.5 * policy.absolute_state_tolerance();
    let segment_count = 128_usize;
    let horizon = 2.0_f64;
    let dt = horizon / segment_count as f64;

    let mut declared_target = 0.0_f64;
    for _ in 0..segment_count {
        declared_target += step;
    }
    let query = SingleIntegratorQuery1D::new(
        &model,
        0.0,
        declared_target,
        PlantTimeProfile::new(horizon).unwrap(),
        policy,
    )
    .unwrap();

    let mut segments = Vec::with_capacity(segment_count);
    let mut declared_state = 0.0_f64;
    for index in 0..segment_count {
        let next_state = declared_state + step;
        segments.push(TimedControlSegment1D::new(
            index as f64 * dt,
            (index + 1) as f64 * dt,
            0.0,
            declared_state,
            next_state,
        ));
        declared_state = next_state;
    }
    let trajectory = TimedControlledTrajectory1D::new(&model, &query, segments).unwrap();

    // A verifier that restarts integration from every declared segment start accepts
    // each local half-tolerance jump and eventually accepts the fabricated target.
    // Independent global replay must keep its own state at x=0 and reject the chain.
    assert!(matches!(
        replay_single_integrator_trajectory(&model, &query, &trajectory),
        Err(KinodynamicError::InvalidTrajectory { .. })
    ));
}

#[test]
fn replay_accepts_bounded_local_declaration_error_without_promoting_it_to_state_authority() {
    let model = BoundedSingleIntegrator1D::new(ControlInterval1D::new(-2.0, 2.0).unwrap());
    let policy = DynamicsReplayPolicy1D::reference_v1();
    let epsilon = 0.5 * policy.absolute_state_tolerance();
    let query = SingleIntegratorQuery1D::new(
        &model,
        0.0,
        1.0,
        PlantTimeProfile::new(1.0).unwrap(),
        policy,
    )
    .unwrap();

    let trajectory = TimedControlledTrajectory1D::new(
        &model,
        &query,
        vec![
            TimedControlSegment1D::new(0.0, 0.5, 1.0, 0.0, 0.5 + epsilon),
            TimedControlSegment1D::new(0.5, 1.0, 1.0, 0.5 + epsilon, 1.0 + epsilon),
        ],
    )
    .unwrap();

    let receipt = replay_single_integrator_trajectory(&model, &query, &trajectory).unwrap();
    assert_eq!(receipt.segment_count(), 2);
    assert_eq!(receipt.final_state().to_bits(), 1.0_f64.to_bits());
    assert!(receipt.maximum_state_residual() <= policy.absolute_state_tolerance());
}

#[test]
fn replay_rejects_trajectory_above_explicit_segment_budget() {
    let model = BoundedSingleIntegrator1D::new(ControlInterval1D::new(-1.0, 1.0).unwrap());
    let query = SingleIntegratorQuery1D::new(
        &model,
        0.0,
        0.0,
        PlantTimeProfile::new(1.0).unwrap(),
        DynamicsReplayPolicy1D::reference_v1(),
    )
    .unwrap();

    let segments = vec![
        TimedControlSegment1D::new(0.0, 1.0, 0.0, 0.0, 0.0);
        MAX_SINGLE_INTEGRATOR_REPLAY_SEGMENTS + 1
    ];
    let trajectory = TimedControlledTrajectory1D::new(&model, &query, segments).unwrap();

    assert!(matches!(
        replay_single_integrator_trajectory(&model, &query, &trajectory),
        Err(KinodynamicError::InvalidTrajectory { .. })
    ));
}
