// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hand-designed reflex policy: the imitation target for `Trainer::run_episode`.
//!
//! Tier 2 of `SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md`: no trainer
//! in any of the six unaudited platforms ever updated a controller weight.
//! This mirrors the pattern already landed in symthaea-quadruped/helicopter
//! -- a simple, mode-conditioned baseline policy the HDC-LTC controller can
//! be trained (via delta rule) to imitate, since this platform has no
//! existing CPG/PD spinal reflex to reuse as a target.
use crate::types::{SubterraneanCommand, SubterraneanOperatingMode, SubterraneanState};

pub fn reflex_command(state: &SubterraneanState) -> SubterraneanCommand {
    let mut cmd = SubterraneanCommand::zero();
    match state.inferred_mode() {
        SubterraneanOperatingMode::Dig => {
            cmd.torques[0] = 0.6; // cutter_head
            cmd.torques[1] = 0.5; // auger_feed
            cmd.torques[2] = 0.5; // left_track
            cmd.torques[3] = 0.5; // right_track
            cmd.torques[5] = if state.cutter_temp_c() > 60.0 {
                0.5
            } else {
                0.1
            }; // thermal_pump
        }
        SubterraneanOperatingMode::Probe => {
            // Lower confidence in localization -- slow down and look around.
            cmd.torques[0] = 0.2;
            cmd.torques[1] = 0.2;
            cmd.torques[2] = 0.2;
            cmd.torques[3] = 0.2;
            cmd.torques[5] = 0.2;
        }
        SubterraneanOperatingMode::Stabilize => {
            // Overheating or elevated abort risk -- stop cutting, cool down.
            cmd.torques[0] = 0.1;
            cmd.torques[5] = 0.7;
        }
        SubterraneanOperatingMode::Retreat | SubterraneanOperatingMode::Surface => {
            // Head back: reverse tracks, stop cutting, keep the thermal
            // pump at full authority (matches the SafeFallback intent).
            cmd.torques[2] = -0.6;
            cmd.torques[3] = -0.6;
            cmd.torques[5] = 1.0;
        }
        SubterraneanOperatingMode::BlackoutAutonomy => {
            cmd.torques[2] = -0.3;
            cmd.torques[3] = -0.3;
            cmd.torques[5] = 0.3;
        }
        SubterraneanOperatingMode::FloodResponse => {
            cmd.torques[4] = 0.5; // ballast_trim: rise
            cmd.torques[5] = 1.0; // thermal_pump: full authority
        }
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflex_is_finite() {
        let cmd = reflex_command(&SubterraneanState::home());
        assert!(cmd.torques.iter().all(|t| t.is_finite()));
    }

    #[test]
    fn test_dig_mode_cuts_forward() {
        let cmd = reflex_command(&SubterraneanState::home());
        assert!(cmd.cutter_head() > 0.0);
        assert!(cmd.left_track() > 0.0);
    }

    #[test]
    fn test_retreat_mode_reverses_tracks_and_holds_thermal_pump() {
        let mut state = SubterraneanState::home();
        state.channels[crate::types::ABORT_RECOMMENDATION] = 0.95;
        let cmd = reflex_command(&state);
        assert!(cmd.left_track() < 0.0);
        assert!(cmd.right_track() < 0.0);
    }
}
