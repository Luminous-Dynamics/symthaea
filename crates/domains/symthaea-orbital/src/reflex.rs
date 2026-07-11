// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hand-designed reflex policy: the imitation target for `Trainer::run_episode`.
//!
//! Real-trainer follow-up to `SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md`:
//! the learned controller only drives the arm's `joint_torques` (translational
//! burns and reaction-wheel desaturation are explicitly not yet wired to any
//! policy -- see `controller.rs`), so the reflex only concerns the arm: a
//! mild PD hold toward the stowed pose (`OrbitalState::stowed()`'s
//! all-zero joint angles), matching the default arm posture during ordinary
//! station-keeping when no task is manipulating anything.
use crate::types::{NUM_JOINTS, OrbitalState};

pub fn reflex_torques(state: &OrbitalState) -> [f32; NUM_JOINTS] {
    let mut t = [0.0f32; NUM_JOINTS];
    for i in 0..NUM_JOINTS {
        let err = -state.joint_angles[i]; // stowed pose = all zero
        let pd = 1.5 * err - 0.2 * state.joint_velocities[i];
        t[i] = (pd as f32).clamp(-1.0, 1.0);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflex_is_finite() {
        let t = reflex_torques(&OrbitalState::stowed());
        assert!(t.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_reflex_zero_at_stowed() {
        let t = reflex_torques(&OrbitalState::stowed());
        assert!(t.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn test_reflex_pulls_back_toward_stowed() {
        let mut state = OrbitalState::stowed();
        state.joint_angles[0] = 1.0;
        let t = reflex_torques(&state);
        assert!(t[0] < 0.0, "reflex must pull joint 0 back toward stowed");
    }
}
