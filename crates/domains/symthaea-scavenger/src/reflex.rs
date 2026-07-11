// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hand-designed reflex policy: the imitation target for `Trainer::run_episode`.
//!
//! Tier 2 of `SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md`. See
//! symthaea-subterranean's reflex.rs for the full rationale -- this mirrors
//! that reference fix's pattern.
use crate::types::{ScavengerCommand, ScavengerOperatingMode, ScavengerState};

pub fn reflex_command(state: &ScavengerState) -> ScavengerCommand {
    let mut cmd = ScavengerCommand::zero();
    match state.inferred_mode() {
        ScavengerOperatingMode::Recovery => {
            cmd.torques[0] = 0.4; // left_track
            cmd.torques[1] = 0.4; // right_track
            cmd.torques[5] = 0.5; // cutter
            cmd.torques[7] = 0.4; // hopper_feed
            cmd.torques[9] = 0.3; // dust_suppression baseline
        }
        ScavengerOperatingMode::DustControl => {
            cmd.torques[5] = 0.2; // ease off cutting
            cmd.torques[9] = 1.0; // dust_suppression full
        }
        ScavengerOperatingMode::JamRecovery => {
            cmd.torques[7] = -0.3; // reverse hopper_feed to clear the jam
            cmd.torques[8] = -0.3; // reverse compactor
        }
        ScavengerOperatingMode::Quarantine => {
            cmd.torques[6] = 0.3; // sorter isolates hazardous material
        }
        ScavengerOperatingMode::HumanSafe => {
            // Stop cutting and driving; a human is nearby.
        }
        ScavengerOperatingMode::EmergencyStop => {
            cmd.torques[9] = 1.0; // dust_suppression: hold authority
        }
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflex_is_finite() {
        let cmd = reflex_command(&ScavengerState::home());
        assert!(cmd.torques.iter().all(|t| t.is_finite()));
    }

    #[test]
    fn test_recovery_mode_recovers_material() {
        let cmd = reflex_command(&ScavengerState::home());
        assert!(cmd.cutter() > 0.0);
    }

    #[test]
    fn test_human_safe_mode_stops_cutting() {
        let mut state = ScavengerState::home();
        state.channels[crate::types::HUMAN_PROXIMITY] = 0.7;
        let cmd = reflex_command(&state);
        assert_eq!(cmd.cutter(), 0.0);
    }
}
