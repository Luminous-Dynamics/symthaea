// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-pass infrastructure / microgrid simulator.
use crate::types::{
    InfrastructureCommand, InfrastructureOperatingMode, InfrastructureState, BROWNOUT_RISK,
    COMMUNITY_DEMAND, COOLANT_TEMP_C, CRITICAL_LOAD_FRACTION, DEADLOCK_RISK, GRID_STRESS,
    ISLANDING_CAPABILITY, ISLANDING_RISK, QUEUE_DEPTH, RECOVERY_MARGIN, RELAY_HEALTH,
    SERVICE_INTEGRITY, SHED_LOAD_RATIO, STORAGE_RATIO, SWITCHGEAR_WEAR, THERMAL_LOAD,
    THERMAL_RUNAWAY_RISK, UNSERVED_DEMAND_RATIO, VOLTAGE_STABILITY,
};

pub trait InfrastructurePhysicsSimulator {
    fn step(&mut self, cmd: &InfrastructureCommand, dt: f64);
    fn state(&self) -> &InfrastructureState;
    fn reset(&mut self);
}

/// A simple stationary node model with storage, thermal, routing, and health.
pub struct SimpleInfrastructureSimulator {
    state: InfrastructureState,
}

impl SimpleInfrastructureSimulator {
    pub fn new() -> Self {
        Self {
            state: InfrastructureState::home(),
        }
    }
}

impl InfrastructurePhysicsSimulator for SimpleInfrastructureSimulator {
    fn step(&mut self, cmd: &InfrastructureCommand, dt: f64) {
        let charge = cmd.charge_bus().max(0.0) as f64;
        let discharge = cmd.discharge_bus().max(0.0) as f64;
        let cooling = cmd.cooling_loop().max(0.0) as f64;
        let heating = cmd.heating_loop().max(0.0) as f64;
        let north = cmd.torques[4].abs() as f64;
        let south = cmd.torques[5].abs() as f64;
        let east = cmd.torques[6].abs() as f64;
        let west = cmd.torques[7].abs() as f64;
        let routing_total = north + south + east + west;
        let routing_imbalance = ((north - south).abs() + (east - west).abs()) * 0.5;
        let overload_pressure = (self.state.channels[QUEUE_DEPTH] * 0.26
            + self.state.channels[GRID_STRESS] * 0.22
            + self.state.channels[THERMAL_LOAD] * 0.18
            + (1.0 - self.state.channels[STORAGE_RATIO]) * 0.16
            + (1.0 - self.state.channels[RELAY_HEALTH]) * 0.1)
            .clamp(0.0, 1.0);
        let discharge_effective = discharge * (1.0 - self.state.channels[SHED_LOAD_RATIO] * 0.3);
        let routing_effective =
            (routing_total * (1.0 - self.state.channels[DEADLOCK_RISK] * 0.35)).clamp(0.0, 4.0);

        self.state.channels[2] = charge * 120.0;
        self.state.channels[3] = discharge_effective * 120.0 + routing_effective * 20.0;
        self.state.channels[STORAGE_RATIO] = (self.state.channels[STORAGE_RATIO]
            + (charge - discharge_effective) * dt * 0.08
            - self.state.channels[COMMUNITY_DEMAND] * dt * 0.015)
            .clamp(0.0, 1.0);
        self.state.channels[THERMAL_LOAD] = (self.state.channels[THERMAL_LOAD]
            + routing_effective * dt * 0.12
            + heating * dt * 0.08
            - cooling * dt * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[QUEUE_DEPTH] = (self.state.channels[QUEUE_DEPTH]
            + self.state.channels[COMMUNITY_DEMAND] * dt * 0.15
            - routing_effective * dt * 0.12
            + self.state.channels[DEADLOCK_RISK] * dt * 0.03)
            .clamp(0.0, 1.0);
        self.state.channels[RELAY_HEALTH] = (self.state.channels[RELAY_HEALTH]
            - self.state.channels[THERMAL_LOAD] * dt * 0.002
            - self.state.channels[QUEUE_DEPTH] * dt * 0.001
            - self.state.channels[SWITCHGEAR_WEAR] * dt * 0.001)
            .clamp(0.0, 1.0);
        self.state.channels[VOLTAGE_STABILITY] = (1.0
            - (self.state.channels[QUEUE_DEPTH] * 0.34
                + self.state.channels[GRID_STRESS] * 0.32
                + self.state.channels[BROWNOUT_RISK] * 0.18))
            .clamp(0.0, 1.0);
        self.state.channels[COOLANT_TEMP_C] = (self.state.channels[COOLANT_TEMP_C]
            + self.state.channels[THERMAL_LOAD] * dt * 18.0
            - cooling * dt * 12.0
            + heating * dt * 4.0)
            .clamp(-10.0, 120.0);
        self.state.channels[8] = (0.2 + heating * 0.6).clamp(0.0, 1.0);
        self.state.channels[9] = (self.state.channels[9] + dt * 0.0005).clamp(0.0, 1.0);
        self.state.channels[10] = north.clamp(0.0, 1.0);
        self.state.channels[11] = south.clamp(0.0, 1.0);
        self.state.channels[12] = east.clamp(0.0, 1.0);
        self.state.channels[13] = west.clamp(0.0, 1.0);
        self.state.channels[14] = (self.state.channels[14]
            + (1.0 - self.state.channels[RELAY_HEALTH]) * dt * 0.02)
            .clamp(0.0, 1.0);
        self.state.channels[ISLANDING_CAPABILITY] = (self.state.channels[ISLANDING_CAPABILITY]
            * 0.995
            + self.state.channels[STORAGE_RATIO] * dt * 0.02
            - self.state.channels[ISLANDING_RISK] * dt * 0.02)
            .clamp(0.0, 1.0);
        self.state.channels[SWITCHGEAR_WEAR] = (self.state.channels[SWITCHGEAR_WEAR]
            + routing_total * dt * 0.006
            + self.state.channels[SHED_LOAD_RATIO] * dt * 0.003)
            .clamp(0.0, 1.0);
        self.state.channels[16] = (self.state.channels[16] * 0.9
            + self.state.channels[VOLTAGE_STABILITY] * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[GRID_STRESS] = (self.state.channels[COMMUNITY_DEMAND] * 0.45
            + self.state.channels[QUEUE_DEPTH] * 0.35
            + (1.0 - self.state.channels[STORAGE_RATIO]) * 0.2)
            .clamp(0.0, 1.0);
        self.state.channels[COMMUNITY_DEMAND] =
            (0.45 + routing_total * 0.05 + heating * 0.1).clamp(0.0, 1.0);
        self.state.channels[BROWNOUT_RISK] = (self.state.channels[QUEUE_DEPTH] * 0.28
            + self.state.channels[GRID_STRESS] * 0.25
            + (1.0 - self.state.channels[VOLTAGE_STABILITY]) * 0.22
            + (1.0 - self.state.channels[STORAGE_RATIO]) * 0.15)
            .clamp(0.0, 1.0);
        self.state.channels[SHED_LOAD_RATIO] = (overload_pressure * 0.55
            + self.state.channels[BROWNOUT_RISK] * 0.2
            - self.state.channels[ISLANDING_CAPABILITY] * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[CRITICAL_LOAD_FRACTION] =
            (0.3 + heating * 0.1 + self.state.channels[QUEUE_DEPTH] * 0.05).clamp(0.0, 1.0);
        self.state.channels[UNSERVED_DEMAND_RATIO] = (self.state.channels[COMMUNITY_DEMAND] * 0.45
            + self.state.channels[SHED_LOAD_RATIO] * 0.35
            + self.state.channels[QUEUE_DEPTH] * 0.15
            - discharge_effective * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[DEADLOCK_RISK] = (routing_imbalance * 0.18
            + self.state.channels[QUEUE_DEPTH] * 0.3
            + self.state.channels[GRID_STRESS] * 0.15
            - routing_effective * 0.04)
            .clamp(0.0, 1.0);
        self.state.channels[ISLANDING_RISK] = ((1.0 - self.state.channels[RELAY_HEALTH]) * 0.4
            + (1.0 - self.state.channels[ISLANDING_CAPABILITY]) * 0.2
            + self.state.channels[GRID_STRESS] * 0.18
            + self.state.channels[QUEUE_DEPTH] * 0.08)
            .clamp(0.0, 1.0);
        self.state.channels[SERVICE_INTEGRITY] = (1.0
            - self.state.channels[UNSERVED_DEMAND_RATIO] * 0.4
            - self.state.channels[BROWNOUT_RISK] * 0.25
            - self.state.channels[ISLANDING_RISK] * 0.15
            - self.state.channels[DEADLOCK_RISK] * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[RECOVERY_MARGIN] = (self.state.channels[STORAGE_RATIO] * 0.34
            + self.state.channels[VOLTAGE_STABILITY] * 0.26
            + (1.0 - self.state.channels[THERMAL_LOAD]) * 0.22
            + self.state.channels[RELAY_HEALTH] * 0.18)
            .clamp(0.0, 1.0);
        self.state.channels[THERMAL_RUNAWAY_RISK] = (self.state.channels[THERMAL_LOAD] * 0.35
            + (self.state.channels[COOLANT_TEMP_C] / 120.0) * 0.3
            + heating * 0.08
            - cooling * 0.06)
            .clamp(0.0, 1.0);

        match self.state.inferred_mode() {
            InfrastructureOperatingMode::CoolingRecovery => {
                self.state.channels[THERMAL_LOAD] =
                    (self.state.channels[THERMAL_LOAD] - cooling * dt * 0.05).clamp(0.0, 1.0);
            }
            InfrastructureOperatingMode::LoadShedding => {
                self.state.channels[QUEUE_DEPTH] = (self.state.channels[QUEUE_DEPTH]
                    - self.state.channels[SHED_LOAD_RATIO] * dt * 0.06)
                    .clamp(0.0, 1.0);
            }
            InfrastructureOperatingMode::DeadlockRecovery => {
                self.state.channels[DEADLOCK_RISK] = (self.state.channels[DEADLOCK_RISK]
                    - routing_effective * dt * 0.03)
                    .clamp(0.0, 1.0);
            }
            InfrastructureOperatingMode::Islanding => {
                self.state.channels[SERVICE_INTEGRITY] = (self.state.channels[SERVICE_INTEGRITY]
                    + self.state.channels[ISLANDING_CAPABILITY] * dt * 0.02)
                    .clamp(0.0, 1.0);
            }
            InfrastructureOperatingMode::Emergency => {
                self.state.channels[SHED_LOAD_RATIO] =
                    (self.state.channels[SHED_LOAD_RATIO] + dt * 0.05).clamp(0.0, 1.0);
            }
            InfrastructureOperatingMode::Balanced => {}
        }

        self.state.channels[15] = (self.state.channels[QUEUE_DEPTH] * 0.5
            + self.state.channels[GRID_STRESS] * 0.3
            + self.state.channels[BROWNOUT_RISK] * 0.15)
            .clamp(0.0, 1.0);
    }
    fn state(&self) -> &InfrastructureState {
        &self.state
    }
    fn reset(&mut self) {
        self.state = InfrastructureState::home();
    }
}

impl Default for SimpleInfrastructureSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stable() {
        let mut sim = SimpleInfrastructureSimulator::new();
        for _ in 0..1000 {
            sim.step(&InfrastructureCommand::zero(), 0.005);
        }
        assert!(sim.state().is_finite());
    }
    #[test]
    fn test_torque_moves() {
        let mut sim = SimpleInfrastructureSimulator::new();
        let mut cmd = InfrastructureCommand::zero();
        cmd.torques[0] = 1.0;
        let before = sim.state().channels[0];
        sim.step(&cmd, 0.01);
        assert_ne!(sim.state().channels[0], before);
    }

    #[test]
    fn test_overloaded_node_loses_voltage_stability() {
        let mut sim = SimpleInfrastructureSimulator::new();
        let mut cmd = InfrastructureCommand::zero();
        cmd.torques[4] = 1.0;
        cmd.torques[5] = 1.0;
        cmd.torques[6] = 1.0;
        cmd.torques[7] = 1.0;
        for _ in 0..200 {
            sim.step(&cmd, 0.02);
        }
        assert!(sim.state().thermal_load() > 0.2);
        assert!(sim.state().voltage_stability() < 0.95);
    }

    #[test]
    fn test_overload_triggers_load_shedding() {
        let mut sim = SimpleInfrastructureSimulator::new();
        sim.state.channels[18] = 0.95;
        sim.state.channels[4] = 0.8;
        let mut cmd = InfrastructureCommand::zero();
        cmd.torques[1] = 1.0;
        cmd.torques[4] = 1.0;
        cmd.torques[5] = 1.0;
        cmd.torques[6] = 1.0;
        cmd.torques[7] = 1.0;
        for _ in 0..100 {
            sim.step(&cmd, 0.02);
        }
        assert!(sim.state().shed_load_ratio() > 0.2);
        assert!(matches!(
            sim.state().inferred_mode(),
            InfrastructureOperatingMode::LoadShedding | InfrastructureOperatingMode::Emergency
        ));
    }

    #[test]
    fn test_cooling_failure_raises_thermal_runaway_risk() {
        let mut sim = SimpleInfrastructureSimulator::new();
        sim.state.channels[1] = 0.7;
        sim.state.channels[7] = 80.0;
        let mut cmd = InfrastructureCommand::zero();
        cmd.torques[3] = 1.0;
        for _ in 0..80 {
            sim.step(&cmd, 0.02);
        }
        assert!(sim.state().thermal_runaway_risk() > 0.5);
    }

    #[test]
    fn test_routing_contention_raises_deadlock_risk() {
        let mut sim = SimpleInfrastructureSimulator::new();
        sim.state.channels[4] = 0.7;
        let mut cmd = InfrastructureCommand::zero();
        cmd.torques[4] = 1.0;
        cmd.torques[6] = 1.0;
        for _ in 0..100 {
            sim.step(&cmd, 0.02);
        }
        assert!(sim.state().deadlock_risk() > 0.25);
    }

    #[test]
    fn test_islanding_preserves_some_service_while_links_degrade() {
        let mut sim = SimpleInfrastructureSimulator::new();
        sim.state.channels[5] = 0.25;
        sim.state.channels[15] = 0.9;
        sim.state.channels[18] = 0.7;
        sim.step(&InfrastructureCommand::zero(), 0.05);
        assert!(sim.state().islanding_risk() > 0.3);
        assert!(sim.state().service_integrity() > 0.2);
    }
}
