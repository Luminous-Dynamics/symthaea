// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-pass scavenger / disassembly simulator.
use crate::types::{
    BATTERY_RATIO, BLADE_BIND_RISK, CONTAINMENT_BREACH_RISK, CONTAMINATION_RISK, HOPPER_FILL,
    HUMAN_PROXIMITY, INCIDENT_RISK, JAM_RISK, QUARANTINE_LOAD_RATIO, SALVAGE_PURITY,
    SALVAGE_VALUE_RATE, SORTER_CLOG_RISK, ScavengerCommand, ScavengerOperatingMode, ScavengerState,
    THERMAL_LOAD, TOOL_WEAR, TOXIC_DUST_RISK,
};

pub trait ScavengerPhysicsSimulator {
    fn step(&mut self, cmd: &ScavengerCommand, dt: f64);
    fn state(&self) -> &ScavengerState;
    fn reset(&mut self);
}

/// A simple recovery robot model with sorting, cutting, hopper, and hazard load.
pub struct SimpleScavengerSimulator {
    state: ScavengerState,
}

impl SimpleScavengerSimulator {
    pub fn new() -> Self {
        Self {
            state: ScavengerState::home(),
        }
    }
}

impl ScavengerPhysicsSimulator for SimpleScavengerSimulator {
    fn step(&mut self, cmd: &ScavengerCommand, dt: f64) {
        let human_guard = (1.0 - self.state.channels[HUMAN_PROXIMITY] * 0.7).clamp(0.1, 1.0);
        let contamination_guard =
            (1.0 - self.state.channels[CONTAMINATION_RISK] * 0.5).clamp(0.2, 1.0);
        let jam_guard = (1.0 - self.state.channels[JAM_RISK] * 0.45).clamp(0.15, 1.0);
        let effective_cutter = cmd.cutter() as f64 * human_guard * contamination_guard * jam_guard;
        let effective_feed = cmd.hopper_feed().max(0.0) as f64 * contamination_guard * jam_guard;
        let effective_sorter = cmd.sorter().abs() as f64 * contamination_guard;
        let effective_compactor = cmd.compactor().abs() as f64 * jam_guard;
        let dust_suppression = cmd.dust_suppression().max(0.0) as f64;

        self.state.channels[0] = cmd.torques[0] as f64;
        self.state.channels[1] = cmd.torques[1] as f64;
        self.state.channels[2] =
            (self.state.channels[2] + cmd.torques[2] as f64 * dt * 0.5).clamp(0.0, 1.0);
        self.state.channels[3] =
            (self.state.channels[3] + cmd.torques[3] as f64 * dt * 0.5).clamp(0.0, 1.0);
        self.state.channels[4] = (0.5 + cmd.torques[4] as f64 * 0.5).clamp(0.0, 1.0);
        self.state.channels[5] =
            (self.state.channels[5] + effective_cutter * dt * 120.0).clamp(0.0, 1_500.0);
        self.state.channels[6] = cmd.sorter() as f64;
        self.state.channels[HOPPER_FILL] = (self.state.channels[HOPPER_FILL]
            + effective_feed * dt * 0.35
            - effective_compactor.max(0.0) * dt * 0.2
            + self.state.channels[QUARANTINE_LOAD_RATIO] * dt * 0.03)
            .clamp(0.0, 1.0);
        self.state.channels[8] =
            (self.state.channels[8] + effective_compactor * dt * 0.25).clamp(0.0, 1.0);
        self.state.channels[9] = (self.state.channels[9] + effective_cutter.abs() * dt * 0.2
            - dust_suppression * dt * 0.25)
            .clamp(0.0, 1.0);
        self.state.channels[BATTERY_RATIO] = (self.state.channels[BATTERY_RATIO]
            - (effective_cutter.abs() * 0.01
                + effective_compactor.abs() * 0.007
                + effective_feed.abs() * 0.004
                + dust_suppression * 0.002)
                * dt)
            .clamp(0.0, 1.0);
        self.state.channels[THERMAL_LOAD] = (self.state.channels[THERMAL_LOAD]
            + (effective_cutter.abs() + effective_compactor.abs()) * dt * 0.08
            - dust_suppression * dt * 0.03)
            .clamp(0.0, 1.0);
        self.state.channels[12] = (0.4 + effective_sorter * 0.2).clamp(0.0, 1.0);
        self.state.channels[13] = (0.12
            + self.state.channels[HUMAN_PROXIMITY] * 0.15
            + self.state.channels[QUARANTINE_LOAD_RATIO] * 0.2)
            .clamp(0.0, 1.0);
        self.state.channels[14] = (0.3 + effective_cutter.abs() * 0.15).clamp(0.0, 1.0);
        self.state.channels[15] = (0.25 + effective_sorter * 0.2).clamp(0.0, 1.0);
        self.state.channels[16] = (0.18 + effective_sorter * 0.1).clamp(0.0, 1.0);
        self.state.channels[17] = (0.15 + self.state.channels[HOPPER_FILL] * 0.15).clamp(0.0, 1.0);
        self.state.channels[TOOL_WEAR] = (self.state.channels[TOOL_WEAR]
            + effective_cutter.abs() as f64 * dt * 0.002)
            .clamp(0.0, 1.0);
        self.state.channels[JAM_RISK] = (self.state.channels[HOPPER_FILL] * 0.5
            + self.state.channels[8] * 0.3
            + self.state.channels[9] * 0.1
            + self.state.channels[SORTER_CLOG_RISK] * 0.08)
            .clamp(0.0, 1.0);
        self.state.channels[SALVAGE_VALUE_RATE] = (self.state.channels[12] * 0.5
            + self.state.channels[14] * 0.2
            + self.state.channels[15] * 0.15
            + self.state.channels[SALVAGE_PURITY] * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[21] = (self.state.channels[21]
            + (cmd.torques[0].abs() as f64 + cmd.torques[1].abs() as f64) * dt * 0.03)
            .clamp(0.0, 1.0);
        self.state.channels[22] = (self.state.channels[22] * 0.9
            + (1.0 - self.state.channels[JAM_RISK]) * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[HUMAN_PROXIMITY] = (self.state.channels[HUMAN_PROXIMITY] * 0.94
            + effective_cutter.abs() as f64 * 0.03)
            .clamp(0.0, 1.0);
        self.state.channels[CONTAINMENT_BREACH_RISK] = (self.state.channels[13] * 0.25
            + self.state.channels[9] * 0.18
            + (1.0 - dust_suppression) * 0.08)
            .clamp(0.0, 1.0);
        self.state.channels[SALVAGE_PURITY] = (1.0
            - self.state.channels[13] * 0.32
            - self.state.channels[CONTAMINATION_RISK] * 0.28
            - self.state.channels[QUARANTINE_LOAD_RATIO] * 0.12)
            .clamp(0.0, 1.0);
        self.state.channels[INCIDENT_RISK] = (self.state.channels[HUMAN_PROXIMITY] * 0.4
            + self.state.channels[JAM_RISK] * 0.3
            + self.state.channels[THERMAL_LOAD] * 0.15
            + self.state.channels[CONTAMINATION_RISK] * 0.12)
            .clamp(0.0, 1.0);
        self.state.channels[CONTAMINATION_RISK] = (self.state.channels[13] * 0.32
            + (1.0 - self.state.channels[22]) * 0.18
            + self.state.channels[HOPPER_FILL] * 0.12)
            .clamp(0.0, 1.0);
        self.state.channels[TOXIC_DUST_RISK] = (self.state.channels[9] * 0.5
            + self.state.channels[13] * 0.18
            + (1.0 - dust_suppression) * 0.18)
            .clamp(0.0, 1.0);
        self.state.channels[QUARANTINE_LOAD_RATIO] = (self.state.channels[13] * 0.45
            + self.state.channels[CONTAMINATION_RISK] * 0.28
            - effective_sorter * 0.06)
            .clamp(0.0, 1.0);
        self.state.channels[BLADE_BIND_RISK] = (self.state.channels[JAM_RISK] * 0.38
            + self.state.channels[TOOL_WEAR] * 0.18
            + effective_cutter.abs() * 0.12)
            .clamp(0.0, 1.0);
        self.state.channels[SORTER_CLOG_RISK] = (self.state.channels[HOPPER_FILL] * 0.22
            + self.state.channels[13] * 0.2
            + (1.0 - self.state.channels[22]) * 0.16)
            .clamp(0.0, 1.0);
        self.state.channels[21] = (self.state.channels[21]
            + self.state.channels[SALVAGE_VALUE_RATE] * dt * 0.02)
            .clamp(0.0, 1.0);

        match self.state.inferred_mode() {
            ScavengerOperatingMode::DustControl => {
                self.state.channels[9] =
                    (self.state.channels[9] - dust_suppression * dt * 0.08).clamp(0.0, 1.0);
            }
            ScavengerOperatingMode::JamRecovery => {
                self.state.channels[HOPPER_FILL] = (self.state.channels[HOPPER_FILL]
                    - effective_compactor * dt * 0.06)
                    .clamp(0.0, 1.0);
            }
            ScavengerOperatingMode::Quarantine => {
                self.state.channels[QUARANTINE_LOAD_RATIO] =
                    (self.state.channels[QUARANTINE_LOAD_RATIO] + dt * 0.02).clamp(0.0, 1.0);
            }
            ScavengerOperatingMode::HumanSafe => {
                self.state.channels[INCIDENT_RISK] =
                    (self.state.channels[INCIDENT_RISK] - dt * 0.02).clamp(0.0, 1.0);
            }
            ScavengerOperatingMode::EmergencyStop => {
                self.state.channels[5] *= 0.9;
            }
            ScavengerOperatingMode::Recovery => {}
        }
    }
    fn state(&self) -> &ScavengerState {
        &self.state
    }
    fn reset(&mut self) {
        self.state = ScavengerState::home();
    }
}

impl Default for SimpleScavengerSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stable() {
        let mut sim = SimpleScavengerSimulator::new();
        for _ in 0..1000 {
            sim.step(&ScavengerCommand::zero(), 0.005);
        }
        assert!(sim.state().is_finite());
    }
    #[test]
    fn test_torque_moves() {
        let mut sim = SimpleScavengerSimulator::new();
        let mut cmd = ScavengerCommand::zero();
        cmd.torques[0] = 1.0;
        let before = sim.state().channels[0];
        sim.step(&cmd, 0.01);
        assert_ne!(sim.state().channels[0], before);
    }

    #[test]
    fn test_aggressive_cutting_near_humans_raises_incident_risk() {
        let mut sim = SimpleScavengerSimulator::new();
        let mut cmd = ScavengerCommand::zero();
        cmd.torques[5] = 1.0;
        cmd.torques[8] = 0.8;
        for _ in 0..200 {
            sim.step(&cmd, 0.02);
        }
        assert!(sim.state().incident_risk() > 0.2);
        assert!(sim.state().hopper_fill() >= 0.0);
    }

    #[test]
    fn test_hazardous_feed_pushes_quarantine_mode() {
        let mut sim = SimpleScavengerSimulator::new();
        sim.state.channels[13] = 0.85;
        sim.state.channels[27] = 0.6;
        sim.state.channels[23] = 0.0;
        let quarantine_before = sim.state.channels[29];
        let mut cmd = ScavengerCommand::zero();
        cmd.torques[6] = 0.8;
        cmd.torques[7] = 0.8;
        for _ in 0..80 {
            sim.step(&cmd, 0.02);
        }
        assert!(sim.state().quarantine_load_ratio() >= quarantine_before);
    }

    #[test]
    fn test_unsuppressed_dust_raises_toxic_dust_risk() {
        let mut sim = SimpleScavengerSimulator::new();
        let mut cmd = ScavengerCommand::zero();
        cmd.torques[5] = 1.0;
        for _ in 0..100 {
            sim.step(&cmd, 0.02);
        }
        assert!(sim.state().toxic_dust_risk() > 0.3);
    }

    #[test]
    fn test_jam_conditions_force_jam_recovery() {
        let mut sim = SimpleScavengerSimulator::new();
        sim.state.channels[7] = 0.95;
        sim.state.channels[8] = 0.9;
        sim.step(&ScavengerCommand::zero(), 0.02);
        assert!(sim.state().jam_risk() > 0.5);
        assert!(matches!(
            sim.state().inferred_mode(),
            ScavengerOperatingMode::JamRecovery | ScavengerOperatingMode::EmergencyStop
        ));
    }

    #[test]
    fn test_nearby_humans_push_human_safe_mode() {
        let mut sim = SimpleScavengerSimulator::new();
        sim.state.channels[23] = 0.7;
        sim.step(&ScavengerCommand::zero(), 0.02);
        assert_eq!(
            sim.state().inferred_mode(),
            ScavengerOperatingMode::HumanSafe
        );
    }
}
