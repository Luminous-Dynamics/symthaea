// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hand-designed reflex policy: the imitation target for `Trainer::run_episode`.
//!
//! Tier 2 of `SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md`. See
//! symthaea-subterranean's reflex.rs for the full rationale -- this mirrors
//! that reference fix's pattern.
use crate::types::{InfrastructureCommand, InfrastructureOperatingMode, InfrastructureState};

pub fn reflex_command(state: &InfrastructureState) -> InfrastructureCommand {
    let mut cmd = InfrastructureCommand::zero();
    match state.inferred_mode() {
        InfrastructureOperatingMode::Balanced => {
            cmd.torques[0] = 0.3; // charge_bus: routine charging
            cmd.torques[4] = 0.3; // routing_north: baseline
            cmd.torques[5] = 0.3; // routing_south: baseline
        }
        InfrastructureOperatingMode::LoadShedding => {
            cmd.torques[1] = 0.6; // discharge_bus: supply from storage
        }
        InfrastructureOperatingMode::CoolingRecovery => {
            cmd.torques[2] = 0.9; // cooling_loop
        }
        InfrastructureOperatingMode::Islanding => {
            cmd.torques[0] = 0.2; // charge_bus: maintain local charge only
        }
        InfrastructureOperatingMode::DeadlockRecovery => {
            cmd.torques[4] = -0.3; // routing_north: reroute away
            cmd.torques[6] = 0.3; // routing_east: alternate path
        }
        InfrastructureOperatingMode::Emergency => {
            cmd.torques[1] = 1.0; // discharge_bus: full authority
            cmd.torques[2] = 1.0; // cooling_loop: full authority
        }
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflex_is_finite() {
        let cmd = reflex_command(&InfrastructureState::home());
        assert!(cmd.torques.iter().all(|t| t.is_finite()));
    }

    #[test]
    fn test_emergency_mode_maximizes_cooling_and_discharge() {
        let mut state = InfrastructureState::home();
        state.channels[crate::types::THERMAL_RUNAWAY_RISK] = 0.9;
        let cmd = reflex_command(&state);
        assert_eq!(cmd.discharge_bus(), 1.0);
        assert_eq!(cmd.cooling_loop(), 1.0);
    }
}
