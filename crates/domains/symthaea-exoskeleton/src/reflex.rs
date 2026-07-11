// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hand-designed reflex policy: the imitation target for `Trainer::run_episode`.
//!
//! Real-trainer follow-up to `SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md`:
//! basic assistive co-actuation -- amplify whatever torque the human is
//! already applying, the fundamental behavior of a powered assistive
//! exoskeleton, rather than fighting or ignoring it.
use crate::types::{ExoskeletonState, NUM_ACTUATORS};

/// Fraction of the human's own torque the exoskeleton adds as assistance.
const ASSIST_GAIN: f64 = 0.5;

pub fn reflex_torques(state: &ExoskeletonState) -> [f32; NUM_ACTUATORS] {
    let mut t = [0.0f32; NUM_ACTUATORS];
    for i in 0..NUM_ACTUATORS {
        t[i] = (state.human_torques[i] * ASSIST_GAIN).clamp(-1.0, 1.0) as f32;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflex_is_finite() {
        let t = reflex_torques(&ExoskeletonState::standing());
        assert!(t.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_reflex_amplifies_human_effort() {
        let mut state = ExoskeletonState::standing();
        state.human_torques[0] = 1.0;
        let t = reflex_torques(&state);
        assert!(
            t[0] > 0.0,
            "assistance must be in the same direction as human effort"
        );
    }
}
