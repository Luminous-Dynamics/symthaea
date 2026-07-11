// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hand-designed reflex policy: the imitation target for `Trainer::run_episode`.
//!
//! Real-trainer follow-up to `SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md`:
//! a mild joint-space PD hold toward the home pose. The passive RCM spring
//! (already applied by the simulator's own physics, not commanded here) does
//! the actual remote-center-of-motion constraint; this reflex just resists
//! drift away from home -- matching the "RCM neutral at home" behavior
//! already validated by `simulator::tests::test_rcm_neutral_at_home`.
use crate::types::{NUM_JOINTS, SurgicalState};

/// Home-pose joint angles (from `SurgicalState::home()`), the reflex's
/// attractor.
const HOME_JOINT_ANGLES: [f64; NUM_JOINTS] = [0.0, 0.3, 0.0, -0.5, 0.0, 0.0];

pub fn reflex_torques(state: &SurgicalState) -> [f32; NUM_JOINTS] {
    let mut t = [0.0f32; NUM_JOINTS];
    for i in 0..NUM_JOINTS {
        let err = HOME_JOINT_ANGLES[i] - state.joint_angles[i];
        let pd = 2.0 * err - 0.3 * state.joint_velocities[i];
        t[i] = (pd as f32).clamp(-1.0, 1.0);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflex_is_finite() {
        let t = reflex_torques(&SurgicalState::home());
        assert!(t.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_reflex_zero_at_home() {
        // At exactly the home pose with zero velocity, the PD error is zero.
        let t = reflex_torques(&SurgicalState::home());
        assert!(t.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn test_reflex_pulls_back_toward_home() {
        let mut state = SurgicalState::home();
        state.joint_angles[1] += 1.0; // displace joint 1 away from home
        let t = reflex_torques(&state);
        assert!(t[1] < 0.0, "reflex must pull joint 1 back toward home");
    }
}
