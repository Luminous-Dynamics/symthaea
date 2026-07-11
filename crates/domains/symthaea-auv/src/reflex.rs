// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hand-designed reflex policy: the imitation target for `Trainer::run_episode`.
//!
//! Real-trainer follow-up to `SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md`:
//! a simple depth-hold PD on the vertical thrusters (indices 4/5, matching
//! `AuvCommand::ascend`/`descend`), the most basic real behavior an AUV must
//! exhibit -- hold commanded depth. All other thrusters are left at zero;
//! this reflex makes no claim about heading or forward motion.
use crate::types::{AuvState, NUM_ACTUATORS};

pub fn reflex_thrusters(state: &AuvState, target_depth: f64) -> [f32; NUM_ACTUATORS] {
    let mut t = [0.0f32; NUM_ACTUATORS];
    let depth_error = target_depth - state.depth; // positive => need to descend
    let vertical = (depth_error * 0.1).clamp(-1.0, 1.0) as f32;
    t[4] = vertical;
    t[5] = vertical;
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflex_is_finite() {
        let t = reflex_thrusters(&AuvState::neutral_buoyancy(50.0), 50.0);
        assert!(t.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_reflex_descends_when_too_shallow() {
        let state = AuvState::neutral_buoyancy(10.0);
        let t = reflex_thrusters(&state, 50.0);
        assert!(
            t[4] > 0.0,
            "must command descend thrust when shallower than target"
        );
    }

    #[test]
    fn test_reflex_ascends_when_too_deep() {
        let state = AuvState::neutral_buoyancy(90.0);
        let t = reflex_thrusters(&state, 50.0);
        assert!(
            t[4] < 0.0,
            "must command ascend thrust when deeper than target"
        );
    }
}
