// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hand-designed reflex policy: the imitation target for `Trainer::run_episode`.
//!
//! Tier 2 of `SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md`. See
//! symthaea-subterranean's reflex.rs for the full rationale -- this mirrors
//! that reference fix's pattern.
use crate::types::{AgribotCommand, AgribotOperatingMode, AgribotState};

pub fn reflex_command(state: &AgribotState) -> AgribotCommand {
    let mut cmd = AgribotCommand::zero();
    match state.inferred_mode() {
        AgribotOperatingMode::Stewardship => {
            cmd.torques[0] = 0.4; // left_drive
            cmd.torques[1] = 0.4; // right_drive
            cmd.torques[3] = 0.3; // tool_head: routine tending
            cmd.torques[6] = 0.3; // canopy_sensor_mast: keep sensing
        }
        AgribotOperatingMode::IrrigationRecovery => {
            cmd.torques[4] = 0.8; // water_pump
        }
        AgribotOperatingMode::DiseaseControl => {
            cmd.torques[3] = 0.6; // tool_head: treatment
            cmd.torques[0] = 0.2;
            cmd.torques[1] = 0.2;
        }
        AgribotOperatingMode::SoilProtection => {
            // Stop driving to avoid further compaction.
        }
        AgribotOperatingMode::PollinatorSafe => {
            // Stop the tool head; a pollinator is active nearby.
        }
        AgribotOperatingMode::HumanSafe => {
            // Stop tool and drive; a human is nearby.
        }
        AgribotOperatingMode::RefillReturn => {
            cmd.torques[0] = -0.5; // reverse toward base
            cmd.torques[1] = -0.5;
        }
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflex_is_finite() {
        let cmd = reflex_command(&AgribotState::home());
        assert!(cmd.torques.iter().all(|t| t.is_finite()));
    }

    #[test]
    fn test_low_reserve_reverses_toward_base() {
        let mut state = AgribotState::home();
        state.channels[crate::types::WATER_TANK_RATIO] = 0.05;
        state.channels[crate::types::BATTERY_RATIO] = 0.05;
        // inferred_mode() reads RESERVE_MARGIN as its own channel (only the
        // simulator's step() recomputes it from water/battery); set it
        // directly here since this is a pure reflex_command() unit test
        // with no simulator step in between.
        state.channels[crate::types::RESERVE_MARGIN] = 0.05;
        let cmd = reflex_command(&state);
        assert!(cmd.torques[0] < 0.0);
    }
}
