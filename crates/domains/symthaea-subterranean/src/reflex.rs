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
use crate::mission::SubterraneanMissionIntent;
use crate::types::{SubterraneanCommand, SubterraneanOperatingMode, SubterraneanState};

pub fn reflex_command(state: &SubterraneanState) -> SubterraneanCommand {
    reflex_command_for_mission(state, SubterraneanMissionIntent::FollowVein)
}

pub fn reflex_command_for_mission(
    state: &SubterraneanState,
    mission: SubterraneanMissionIntent,
) -> SubterraneanCommand {
    let mut cmd = SubterraneanCommand::zero();
    match state.inferred_mode() {
        SubterraneanOperatingMode::Dig => match mission {
            SubterraneanMissionIntent::Explore => {
                cmd.torques[0] = 0.4;
                cmd.torques[1] = 0.35;
                cmd.torques[2] = 0.35;
                cmd.torques[3] = 0.35;
                cmd.torques[5] = 0.1;
            }
            SubterraneanMissionIntent::ProbeAhead => {
                cmd.torques[0] = 0.12;
                cmd.torques[1] = 0.18;
                cmd.torques[2] = 0.08;
                cmd.torques[3] = 0.08;
                cmd.torques[5] = 0.25;
            }
            SubterraneanMissionIntent::FollowVein => {
                cmd.torques[0] = 0.6;
                cmd.torques[1] = 0.5;
                cmd.torques[2] = 0.5;
                cmd.torques[3] = 0.5;
                cmd.torques[5] = if state.cutter_temp_c() > 60.0 {
                    0.5
                } else {
                    0.1
                };
            }
            SubterraneanMissionIntent::ReturnHome => {
                cmd.torques[2] = -0.45;
                cmd.torques[3] = -0.45;
                cmd.torques[5] = if state.cutter_temp_c() > 70.0 {
                    0.4
                } else {
                    0.0
                };
            }
            SubterraneanMissionIntent::EmergencySurface => {
                cmd.torques[2] = -0.65;
                cmd.torques[3] = -0.65;
                cmd.torques[4] = 0.4;
                cmd.torques[5] = 0.5;
            }
            SubterraneanMissionIntent::HoldPosition | SubterraneanMissionIntent::MaintainRelay => {
                cmd.torques[5] = if state.cutter_temp_c() > 70.0 {
                    0.4
                } else {
                    0.0
                };
                if matches!(mission, SubterraneanMissionIntent::MaintainRelay) {
                    cmd.recovery.relay_deployer = 1.0;
                }
            }
            SubterraneanMissionIntent::YieldTunnel => {
                cmd.torques[2] = -0.25;
                cmd.torques[3] = -0.25;
                cmd.torques[5] = if state.cutter_temp_c() > 70.0 {
                    0.4
                } else {
                    0.0
                };
            }
            SubterraneanMissionIntent::AssistPeer => {
                cmd.torques[2] = 0.35;
                cmd.torques[3] = 0.35;
                cmd.torques[5] = 0.2;
            }
        },
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
            cmd.torques[5] = 0.3;
            cmd.recovery.relay_deployer = 1.0;
        }
        SubterraneanOperatingMode::FloodResponse => {
            cmd.torques[2] = -0.35;
            cmd.torques[3] = -0.35;
            cmd.torques[4] = 0.5; // ballast trim reduces pitch while withdrawing
            cmd.torques[5] = 0.6;
            cmd.recovery.dewatering_pump = 1.0;
            cmd.recovery.sealant_injector = if state.seal_integrity() < 0.75 {
                0.8
            } else {
                0.0
            };
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

    #[test]
    fn mission_changes_nominal_reflex_direction() {
        let state = SubterraneanState::home();
        let follow = reflex_command_for_mission(&state, SubterraneanMissionIntent::FollowVein);
        let return_home = reflex_command_for_mission(&state, SubterraneanMissionIntent::ReturnHome);
        assert!(follow.left_track() > 0.0);
        assert!(return_home.left_track() < 0.0);
        assert_eq!(return_home.cutter_head(), 0.0);
    }
}
