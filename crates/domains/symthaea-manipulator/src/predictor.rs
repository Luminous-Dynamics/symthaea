// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit one-step state prediction error for sensorimotor grounding.

use crate::types::{ManipulatorState, NUM_JOINTS};

/// Normalized RMS distance between a predicted and observed manipulator state.
///
/// Each channel family is scaled by a physically meaningful range before
/// aggregation. Unlike consecutive-perception novelty, this remains low during
/// correctly predicted motion and rises when observations disagree with the
/// dynamics model.
pub fn normalized_state_prediction_error(
    predicted: &ManipulatorState,
    observed: &ManipulatorState,
) -> f32 {
    if !predicted.is_finite() || !observed.is_finite() {
        return 1.0;
    }

    let mut squared_error = 0.0f64;
    let mut channels = 0usize;

    for joint in 0..NUM_JOINTS {
        accumulate(
            &mut squared_error,
            &mut channels,
            predicted.joint_angles[joint] - observed.joint_angles[joint],
            std::f64::consts::PI,
        );
        accumulate(
            &mut squared_error,
            &mut channels,
            predicted.joint_velocities[joint] - observed.joint_velocities[joint],
            5.0,
        );
    }
    for axis in 0..3 {
        accumulate(
            &mut squared_error,
            &mut channels,
            predicted.end_effector_position[axis] - observed.end_effector_position[axis],
            1.0,
        );
        accumulate(
            &mut squared_error,
            &mut channels,
            predicted.end_effector_force[axis] - observed.end_effector_force[axis],
            87.0,
        );
    }
    accumulate(
        &mut squared_error,
        &mut channels,
        predicted.gripper_opening - observed.gripper_opening,
        1.0,
    );

    ((squared_error / channels as f64).sqrt().clamp(0.0, 1.0)) as f32
}

fn accumulate(sum: &mut f64, channels: &mut usize, delta: f64, scale: f64) {
    let normalized = delta / scale;
    *sum += normalized * normalized;
    *channels += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_states_have_zero_prediction_error() {
        let state = ManipulatorState::home();
        assert_eq!(normalized_state_prediction_error(&state, &state), 0.0);
    }

    #[test]
    fn correctly_moving_state_is_not_penalized_for_motion_itself() {
        let mut predicted = ManipulatorState::home();
        predicted.joint_angles[0] += 0.5;
        predicted.joint_velocities[0] = 1.0;
        let observed = predicted.clone();
        assert_eq!(
            normalized_state_prediction_error(&predicted, &observed),
            0.0
        );
    }

    #[test]
    fn model_mismatch_produces_positive_error() {
        let predicted = ManipulatorState::home();
        let mut observed = predicted.clone();
        observed.end_effector_position[0] += 0.2;
        observed.end_effector_force[2] = 20.0;
        assert!(normalized_state_prediction_error(&predicted, &observed) > 0.0);
    }

    #[test]
    fn non_finite_observation_fails_closed() {
        let predicted = ManipulatorState::home();
        let mut observed = predicted.clone();
        observed.joint_velocities[0] = f64::NAN;
        assert_eq!(
            normalized_state_prediction_error(&predicted, &observed),
            1.0
        );
    }
}
