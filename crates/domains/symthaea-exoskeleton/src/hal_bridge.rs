// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Zero-torque HAL command-shape stub for the full-frame research model.
//!
//! This module currently constructs a 21-channel all-zero command and checks
//! its shape in software. It does not instantiate `HalRuntime`, open I2C,
//! drive servos, implement an emergency stop, or authorize physical actuation.
//! It is not a hardware adapter. A separately qualified HAL and independent
//! hardware safety path are required before any physical device connection.

#![cfg(feature = "hal")]

use symthaea_humanoid::types::HumanoidCommand;

use crate::full_frame::{FullFrameSimulator, NUM_FULL_FRAME_JOINTS};

/// HAL channel count (2× PCA9685 × 16 channels = 32, we use 21 for humanoid).
pub const HAL_CHANNELS: usize = 21;

/// Maps 20 full-frame joints to 21 HAL channels.
///
/// HAL channel layout (PCA9685 boards 0x40 and 0x41):
/// - Channel 0-1:   Spine (neck, lumbar)
/// - Channel 2-7:   Left arm (6 joints)
/// - Channel 8-13:  Right arm (6 joints)
/// - Channel 14-16: Left leg (3 joints)
/// - Channel 17-19: Right leg (3 joints)
/// - Channel 20:    Reserved (head/gripper)
/// Produce the current zero-torque placeholder in the declared channel layout.
///
/// The simulator is intentionally not an actuator-command source. The input
/// is retained for call-site compatibility but cannot affect the emitted values.
pub fn full_frame_to_humanoid_command(_sim: &FullFrameSimulator) -> HumanoidCommand {
    HumanoidCommand {
        torques: vec![0.0f32; HAL_CHANNELS],
    }
}

/// Bridge configuration.
#[derive(Debug, Clone)]
pub struct HalBridgeConfig {
    /// Maximum ticks to run (None = until e-stop).
    pub max_ticks: Option<u64>,
    /// Legacy configuration placeholder. No I2C transport is wired or consulted.
    pub use_mock_i2c: bool,
}

impl Default for HalBridgeConfig {
    fn default() -> Self {
        Self {
            max_ticks: Some(50), // Matches current full-frame stability envelope
            use_mock_i2c: true,
        }
    }
}

/// Summary of a HAL bridge session.
#[derive(Debug, Clone, Default)]
pub struct HalBridgeReport {
    pub ticks_executed: u64,
    /// True only if every command generated in this software-only loop was zero torque.
    pub all_commands_zero_torque: bool,
    pub simulator_finite: bool,
}

/// Run a software-only loop that steps the research plant and checks the
/// zero-torque command shape. No HAL runtime or I2C device is used.
pub fn run_bridged_session(
    sim: &mut FullFrameSimulator,
    config: HalBridgeConfig,
) -> HalBridgeReport {
    let mut report = HalBridgeReport {
        all_commands_zero_torque: true,
        ..HalBridgeReport::default()
    };

    let max = config.max_ticks.unwrap_or(50);
    for _ in 0..max {
        // Step the physics simulation
        sim.step(0.001);

        // Generate HAL command from simulator state
        let cmd = full_frame_to_humanoid_command(sim);
        report.all_commands_zero_torque &= cmd.torques.len() == HAL_CHANNELS
            && cmd.torques.iter().all(|torque| *torque == 0.0);
        report.ticks_executed += 1;
    }

    // Check that at least one chain tip is still finite
    let spine_finite = sim
        .spine_chain
        .links
        .first()
        .and_then(|&h| sim.world.body(h))
        .map(|b| b.transform.translation.0.iter().all(|v| v.is_finite()))
        .unwrap_or(false);
    report.simulator_finite = spine_finite;

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_generation_matches_hal_layout() {
        let sim = FullFrameSimulator::new();
        let cmd = full_frame_to_humanoid_command(&sim);
        assert_eq!(cmd.torques.len(), HAL_CHANNELS);
        assert_eq!(HAL_CHANNELS, 21);
    }

    #[test]
    fn test_command_is_zero_independent_of_simulator_state() {
        let sim = FullFrameSimulator::new();
        let cmd = full_frame_to_humanoid_command(&sim);
        assert_eq!(cmd.torques.len(), HAL_CHANNELS);
        assert!(cmd.torques.iter().all(|torque| *torque == 0.0));
    }

    #[test]
    fn test_bridged_session_runs_to_completion() {
        let mut sim = FullFrameSimulator::new();
        let config = HalBridgeConfig::default();
        let report = run_bridged_session(&mut sim, config);

        assert_eq!(report.ticks_executed, 50);
        assert!(report.simulator_finite, "Simulator should remain finite");
    }

    #[test]
    fn test_software_loop_reports_zero_torque_commands() {
        let mut sim = FullFrameSimulator::new();
        let config = HalBridgeConfig {
            max_ticks: Some(20),
            use_mock_i2c: true,
        };
        let report = run_bridged_session(&mut sim, config);
        assert_eq!(report.ticks_executed, 20);
        assert!(report.all_commands_zero_torque);
    }

    #[test]
    fn test_full_frame_joint_count_matches_expectation() {
        // Sanity: 20 full-frame joints, 21 HAL channels (1 reserved)
        assert_eq!(NUM_FULL_FRAME_JOINTS, 20);
        assert_eq!(HAL_CHANNELS, 21);
    }
}
