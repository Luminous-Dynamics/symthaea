// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hand-designed reflex policy: the imitation target for `Trainer::run_episode`.
//!
//! Tier 2 of `SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md`. See
//! symthaea-subterranean's reflex.rs for the full rationale -- this mirrors
//! that reference fix's pattern.
use crate::types::{BiotaCommand, BiotaOperatingMode, BiotaState};

pub fn reflex_command(state: &BiotaState) -> BiotaCommand {
    let mut cmd = BiotaCommand::zero();
    match state.inferred_mode() {
        BiotaOperatingMode::Observe => {
            cmd.torques[2] = 0.3; // gaze_beacon: keep watch
        }
        BiotaOperatingMode::Escort => {
            cmd.torques[0] = 0.3; // left_drive
            cmd.torques[1] = 0.3; // right_drive
            cmd.torques[2] = 0.4; // gaze_beacon
        }
        BiotaOperatingMode::CrossingGuard => {
            cmd.torques[5] = 0.6; // sanctuary_projector: signal right-of-way
        }
        BiotaOperatingMode::DistressResponse => {
            cmd.torques[3] = 0.5; // acoustic_chime
            cmd.torques[4] = 0.5; // thermal_beacon
            cmd.torques[5] = 0.7; // sanctuary_projector
        }
        BiotaOperatingMode::SanctuaryHold => {
            cmd.torques[5] = 1.0; // sanctuary_projector: full authority
        }
        BiotaOperatingMode::QuietMode => {
            // Minimize acoustic/motion disturbance; stay still and silent.
        }
        BiotaOperatingMode::BlackoutAutonomy => {
            cmd.torques[4] = 0.4; // thermal_beacon: stay visible without comms
        }
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflex_is_finite() {
        let cmd = reflex_command(&BiotaState::home());
        assert!(cmd.torques.iter().all(|t| t.is_finite()));
    }

    #[test]
    fn test_distress_response_signals() {
        let mut state = BiotaState::home();
        state.channels[crate::types::DISTRESS_SIGNAL] = 0.85;
        let cmd = reflex_command(&state);
        assert!(cmd.sanctuary_projector() > 0.0);
    }
}
