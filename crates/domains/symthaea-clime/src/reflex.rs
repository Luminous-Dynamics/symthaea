// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hand-designed reflex policy: the imitation target for `Trainer::run_episode`.
//!
//! Tier 2 of `SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md`. See
//! symthaea-subterranean's reflex.rs for the full rationale -- this mirrors
//! that reference fix's pattern.
use crate::types::{COLD_STRESS, ClimeCommand, ClimeOperatingMode, ClimeState, THERMAL_STRESS};

pub fn reflex_command(state: &ClimeState) -> ClimeCommand {
    let mut cmd = ClimeCommand::zero();
    match state.inferred_mode() {
        ClimeOperatingMode::BalancedHabitat => {
            cmd.torques[0] = 0.3; // ventilation_fan: baseline
            cmd.torques[1] = 0.3; // filtration_loop: baseline
            cmd.torques[6] = 0.5; // light_brightness: comfortable level
        }
        ClimeOperatingMode::AirRecovery => {
            cmd.torques[0] = 0.9; // ventilation_fan
            cmd.torques[1] = 0.8; // filtration_loop
        }
        ClimeOperatingMode::ThermalRecovery => {
            if state.channels[THERMAL_STRESS] >= state.channels[COLD_STRESS] {
                cmd.torques[2] = 0.8; // cooling_loop
            } else {
                cmd.torques[3] = 0.8; // heating_loop
            }
        }
        ClimeOperatingMode::CircadianSupport => {
            cmd.torques[7] = 0.6; // light_circadian_shift
        }
        ClimeOperatingMode::QuietNight => {
            cmd.torques[6] = 0.1; // light_brightness: dim
        }
        ClimeOperatingMode::UtilityConstrained => {
            cmd.torques[0] = 0.2; // ventilation_fan: minimal, conserve reserve
        }
        ClimeOperatingMode::IsolationMode => {
            cmd.torques[0] = 1.0; // ventilation_fan: full
            cmd.torques[1] = 1.0; // filtration_loop: full
        }
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflex_is_finite() {
        let cmd = reflex_command(&ClimeState::home());
        assert!(cmd.torques.iter().all(|t| t.is_finite()));
    }

    #[test]
    fn test_isolation_mode_maximizes_ventilation() {
        let mut state = ClimeState::home();
        state.channels[crate::types::SMOKE_RISK] = 0.9;
        let cmd = reflex_command(&state);
        assert_eq!(cmd.ventilation_fan(), 1.0);
        assert_eq!(cmd.filtration_loop(), 1.0);
    }
}
