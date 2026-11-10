// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Software-only command-shape stub for the full-frame research model.
//!
//! The historical module name is retained for source compatibility, but this
//! file does not instantiate a HAL runtime, open I2C, drive servos, implement
//! an emergency stop, or authorize physical actuation. It only emits an
//! all-zero command in a proposed 21-channel logical layout and checks that
//! the software loop remains finite.
//!
//! A real hardware adapter requires a separately reviewed transport, calibrated
//! actuators and sensors, independent hardware safety controls, watchdogs,
//! acknowledgements, fault injection and hardware-in-the-loop qualification.

#![cfg(feature = "hal")]

use symthaea_humanoid::types::HumanoidCommand;

use crate::full_frame::{FullFrameSimulator, NUM_FULL_FRAME_JOINTS};

/// Proposed logical command width only; no physical pinout or board mapping is
/// implemented or validated by this research stub.
pub const HAL_CHANNELS: usize = 21;

/// Emit the current all-zero placeholder command.
///
/// This function intentionally takes no simulator or cognitive input: the
/// research plant is not an actuator-command source.
pub fn zero_torque_humanoid_command() -> HumanoidCommand {
    HumanoidCommand {
        torques: vec![0.0f32; HAL_CHANNELS],
    }
}

/// Configuration for the software-only research smoke loop.
#[derive(Debug, Clone)]
pub struct SoftwareStubConfig {
    /// Number of simulation ticks to execute.
    pub max_ticks: u64,
}

impl Default for SoftwareStubConfig {
    fn default() -> Self {
        Self { max_ticks: 50 }
    }
}

/// Report from the software-only zero-command loop.
#[derive(Debug, Clone, Default)]
pub struct SoftwareStubReport {
    pub ticks_executed: u64,
    pub commands_checked: u64,
    /// True only if every checked command had the declared width and all-zero torque.
    pub all_commands_zero_torque: bool,
    pub simulator_finite: bool,
}

/// Step the simplified research plant and check the zero-torque command shape.
///
/// No HAL runtime or I2C device is used. The result is software smoke evidence
/// only, not hardware, safety, biomechanics or human-subject qualification.
pub fn run_software_stub_session(
    sim: &mut FullFrameSimulator,
    config: SoftwareStubConfig,
) -> SoftwareStubReport {
    let mut report = SoftwareStubReport {
        all_commands_zero_torque: true,
        ..SoftwareStubReport::default()
    };

    for _ in 0..config.max_ticks {
        sim.step(0.001);
        let command = zero_torque_humanoid_command();
        report.all_commands_zero_torque &= command.torques.len() == HAL_CHANNELS
            && command.torques.iter().all(|torque| *torque == 0.0);
        report.commands_checked += 1;
        report.ticks_executed += 1;
    }

    let spine_finite = sim
        .spine_chain
        .links
        .first()
        .and_then(|&handle| sim.world.body(handle))
        .map(|body| body.transform.translation.0.iter().all(|value| value.is_finite()))
        .unwrap_or(false);
    report.simulator_finite = spine_finite;
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_command_has_declared_logical_width() {
        let command = zero_torque_humanoid_command();
        assert_eq!(command.torques.len(), HAL_CHANNELS);
        assert_eq!(HAL_CHANNELS, 21);
        assert!(command.torques.iter().all(|torque| *torque == 0.0));
    }

    #[test]
    fn software_stub_loop_reports_only_zero_commands() {
        let mut sim = FullFrameSimulator::new();
        let report = run_software_stub_session(
            &mut sim,
            SoftwareStubConfig { max_ticks: 20 },
        );
        assert_eq!(report.ticks_executed, 20);
        assert_eq!(report.commands_checked, 20);
        assert!(report.all_commands_zero_torque);
        assert!(report.simulator_finite);
    }

    #[test]
    fn full_frame_joint_count_matches_research_model() {
        assert_eq!(NUM_FULL_FRAME_JOINTS, 20);
        assert_eq!(HAL_CHANNELS, 21);
    }
}
