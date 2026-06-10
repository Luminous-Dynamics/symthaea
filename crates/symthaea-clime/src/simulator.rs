// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-pass habitat homeostasis simulator.
use crate::types::{
    AIR_QUALITY, CIRCADIAN_MISMATCH, CO2_LOAD, COLD_STRESS, COMFORT_INTEGRITY, CONTAMINATION_RISK,
    COOLING_MARGIN, ClimeCommand, ClimeState, FILTRATION_EFFECTIVENESS, HABITAT_CONTINUITY,
    HEATING_MARGIN, HUMIDITY, HUMIDITY_STRESS, LIGHT_LEVEL, MISSION_PROGRESS, NIGHT_MODE_PRESSURE,
    NOISE_STRESS, OCCUPANCY_CONFIDENCE, OCCUPANCY_PRESSURE, PUBLIC_HEALTH_RISK, QUIET_COMPLIANCE,
    SENSOR_CONFIDENCE, SMOKE_RISK, THERMAL_STRESS, UTILITY_RESERVE, VENTILATION_AUTHORITY,
    ZONE_ISOLATION_CONFIDENCE,
};

pub trait ClimePhysicsSimulator {
    fn step(&mut self, cmd: &ClimeCommand, dt: f64);
    fn state(&self) -> &ClimeState;
    fn reset(&mut self);
}

pub struct SimpleClimeSimulator {
    state: ClimeState,
}

impl SimpleClimeSimulator {
    pub fn new() -> Self {
        Self {
            state: ClimeState::home(),
        }
    }
}

impl ClimePhysicsSimulator for SimpleClimeSimulator {
    fn step(&mut self, cmd: &ClimeCommand, dt: f64) {
        let vent = cmd.ventilation_fan().max(0.0) as f64;
        let filter = cmd.filtration_loop().max(0.0) as f64;
        let cool = cmd.cooling_loop().max(0.0) as f64;
        let heat = cmd.heating_loop().max(0.0) as f64;
        let humid = cmd.torques[4].max(0.0) as f64;
        let dehumid = cmd.torques[5].max(0.0) as f64;
        let light = cmd.light_brightness().max(0.0) as f64;
        let circadian = cmd.light_circadian_shift().max(0.0) as f64;

        let utility_gate = if self.state.channels[UTILITY_RESERVE] <= 0.2 {
            0.45
        } else {
            1.0
        };
        let quiet_gate = if self.state.channels[NIGHT_MODE_PRESSURE] >= 0.65 {
            0.35
        } else {
            1.0
        };

        let effective_vent = vent * utility_gate;
        let effective_filter = filter * utility_gate;
        let effective_cool = cool * utility_gate;
        let effective_heat = heat * utility_gate;
        let effective_light = light * utility_gate * quiet_gate;

        self.state.channels[AIR_QUALITY] = (self.state.channels[AIR_QUALITY]
            + effective_vent * dt * 0.08
            + effective_filter * dt * 0.07
            - self.state.channels[CO2_LOAD] * dt * 0.05
            - self.state.channels[SMOKE_RISK] * dt * 0.07
            - self.state.channels[CONTAMINATION_RISK] * dt * 0.06)
            .clamp(0.0, 1.0);
        self.state.channels[CO2_LOAD] = (self.state.channels[CO2_LOAD]
            + self.state.channels[OCCUPANCY_PRESSURE] * dt * 0.06
            - effective_vent * dt * 0.09)
            .clamp(0.0, 1.0);
        self.state.channels[HUMIDITY] = (self.state.channels[HUMIDITY] + humid * dt * 0.06
            - dehumid * dt * 0.06)
            .clamp(0.0, 1.0);
        self.state.channels[HUMIDITY_STRESS] =
            ((self.state.channels[HUMIDITY] - 0.5).abs() * 1.8).clamp(0.0, 1.0);
        self.state.channels[THERMAL_STRESS] = (self.state.channels[THERMAL_STRESS]
            + self.state.channels[OCCUPANCY_PRESSURE] * dt * 0.05
            - effective_cool * dt * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[COLD_STRESS] = (self.state.channels[COLD_STRESS]
            + (1.0 - self.state.channels[OCCUPANCY_PRESSURE]) * dt * 0.015
            - effective_heat * dt * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[OCCUPANCY_PRESSURE] =
            (self.state.channels[OCCUPANCY_PRESSURE] * 0.99 + 0.01).clamp(0.0, 1.0);
        self.state.channels[OCCUPANCY_CONFIDENCE] =
            (self.state.channels[OCCUPANCY_CONFIDENCE] * 0.997 + effective_vent * 0.001)
                .clamp(0.0, 1.0);
        self.state.channels[LIGHT_LEVEL] =
            (self.state.channels[LIGHT_LEVEL] + effective_light * dt * 0.08).clamp(0.0, 1.0);
        self.state.channels[CIRCADIAN_MISMATCH] = (self.state.channels[CIRCADIAN_MISMATCH]
            + self.state.channels[NIGHT_MODE_PRESSURE] * dt * 0.04
            - circadian * dt * 0.08)
            .clamp(0.0, 1.0);
        self.state.channels[NOISE_STRESS] = (self.state.channels[NOISE_STRESS]
            + (vent + cool + heat) * dt * 0.04
            - quiet_gate * dt * 0.015)
            .clamp(0.0, 1.0);
        self.state.channels[UTILITY_RESERVE] = (self.state.channels[UTILITY_RESERVE]
            - (effective_vent * 0.01
                + effective_filter * 0.012
                + effective_cool * 0.016
                + effective_heat * 0.016
                + effective_light * 0.008)
                * dt)
            .clamp(0.0, 1.0);
        self.state.channels[VENTILATION_AUTHORITY] =
            (self.state.channels[VENTILATION_AUTHORITY] * 0.998 + effective_vent * 0.001)
                .clamp(0.0, 1.0);
        self.state.channels[FILTRATION_EFFECTIVENESS] =
            (self.state.channels[FILTRATION_EFFECTIVENESS] * 0.998 + effective_filter * 0.001)
                .clamp(0.0, 1.0);
        self.state.channels[COOLING_MARGIN] =
            (self.state.channels[COOLING_MARGIN] * 0.999 + effective_cool * 0.001).clamp(0.0, 1.0);
        self.state.channels[HEATING_MARGIN] =
            (self.state.channels[HEATING_MARGIN] * 0.999 + effective_heat * 0.001).clamp(0.0, 1.0);
        self.state.channels[ZONE_ISOLATION_CONFIDENCE] =
            (self.state.channels[ZONE_ISOLATION_CONFIDENCE] + effective_filter * dt * 0.03
                - self.state.channels[SMOKE_RISK] * dt * 0.02)
                .clamp(0.0, 1.0);
        self.state.channels[SMOKE_RISK] = (self.state.channels[SMOKE_RISK] * 0.995
            + self.state.channels[OCCUPANCY_PRESSURE] * 0.001)
            .clamp(0.0, 1.0);
        self.state.channels[CONTAMINATION_RISK] = (self.state.channels[CONTAMINATION_RISK] * 0.996
            + self.state.channels[CO2_LOAD] * 0.002
            - effective_filter * dt * 0.04)
            .clamp(0.0, 1.0);
        self.state.channels[SENSOR_CONFIDENCE] =
            (self.state.channels[SENSOR_CONFIDENCE] * 0.999 + 0.0005).clamp(0.0, 1.0);
        self.state.channels[NIGHT_MODE_PRESSURE] =
            (self.state.channels[NIGHT_MODE_PRESSURE] * 0.997 + 0.001).clamp(0.0, 1.0);
        self.state.channels[PUBLIC_HEALTH_RISK] = (self.state.channels[SMOKE_RISK] * 0.4
            + self.state.channels[CONTAMINATION_RISK] * 0.35
            + self.state.channels[CO2_LOAD] * 0.2)
            .clamp(0.0, 1.0);
        self.state.channels[COMFORT_INTEGRITY] = (1.0
            - self.state.channels[THERMAL_STRESS] * 0.25
            - self.state.channels[COLD_STRESS] * 0.2
            - self.state.channels[HUMIDITY_STRESS] * 0.18
            - self.state.channels[CIRCADIAN_MISMATCH] * 0.15
            - self.state.channels[NOISE_STRESS] * 0.12)
            .clamp(0.0, 1.0);
        self.state.channels[23] = (self.state.channels[COMFORT_INTEGRITY] * 0.7
            + self.state.channels[UTILITY_RESERVE] * 0.3)
            .clamp(0.0, 1.0);
        self.state.channels[MISSION_PROGRESS] =
            (self.state.channels[MISSION_PROGRESS] + effective_vent * dt * 0.015).clamp(0.0, 1.0);
        self.state.channels[25] = (1.0
            - (self.state.channels[THERMAL_STRESS] - self.state.channels[COLD_STRESS]).abs())
        .clamp(0.0, 1.0);
        self.state.channels[QUIET_COMPLIANCE] = (1.0
            - self.state.channels[NOISE_STRESS] * 0.6
            - effective_light * 0.1 * self.state.channels[NIGHT_MODE_PRESSURE])
            .clamp(0.0, 1.0);
        self.state.channels[HABITAT_CONTINUITY] = (self.state.channels[COMFORT_INTEGRITY] * 0.45
            + self.state.channels[AIR_QUALITY] * 0.25
            + self.state.channels[ZONE_ISOLATION_CONFIDENCE] * 0.15
            + self.state.channels[UTILITY_RESERVE] * 0.15)
            .clamp(0.0, 1.0);
    }

    fn state(&self) -> &ClimeState {
        &self.state
    }
    fn reset(&mut self) {
        self.state = ClimeState::home();
    }
}

impl Default for SimpleClimeSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ClimeOperatingMode;

    #[test]
    fn test_stable() {
        let mut sim = SimpleClimeSimulator::new();
        for _ in 0..1000 {
            sim.step(&ClimeCommand::zero(), 0.005);
        }
        assert!(sim.state().is_finite());
    }

    #[test]
    fn test_air_recovery_improves_air_quality() {
        let mut sim = SimpleClimeSimulator::new();
        sim.state.channels[AIR_QUALITY] = 0.3;
        sim.state.channels[CO2_LOAD] = 0.8;
        let mut cmd = ClimeCommand::zero();
        cmd.torques[0] = 1.0;
        cmd.torques[1] = 1.0;
        let before = sim.state().air_quality();
        sim.step(&cmd, 0.05);
        assert!(sim.state().air_quality() > before);
    }

    #[test]
    fn test_co2_spike_enters_air_recovery() {
        let mut sim = SimpleClimeSimulator::new();
        sim.state.channels[CO2_LOAD] = 0.85;
        assert_eq!(sim.state().inferred_mode(), ClimeOperatingMode::AirRecovery);
    }

    #[test]
    fn test_smoke_enters_isolation_mode() {
        let mut sim = SimpleClimeSimulator::new();
        sim.state.channels[SMOKE_RISK] = 0.8;
        assert_eq!(
            sim.state().inferred_mode(),
            ClimeOperatingMode::IsolationMode
        );
    }

    #[test]
    fn test_low_utility_enters_constrained_mode() {
        let mut sim = SimpleClimeSimulator::new();
        sim.state.channels[UTILITY_RESERVE] = 0.15;
        assert_eq!(
            sim.state().inferred_mode(),
            ClimeOperatingMode::UtilityConstrained
        );
    }
}
