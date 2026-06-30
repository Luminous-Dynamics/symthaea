// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-pass subterranean scout simulator.
use crate::types::{
    ABORT_RECOMMENDATION, AQUIFER_RISK, BATTERY_RATIO, COMM_SIGNAL, CUTTER_TEMP_C, DEPTH_M,
    ESCAPE_CONFIDENCE, GAS_RISK, HULL_STRESS, LOCALIZATION_CONFIDENCE, MAPPING_CONFIDENCE,
    MISSION_PROGRESS, OBSTACLE_PROXIMITY, RELAY_DISTANCE_NORM, RELAY_LINK_QUALITY,
    RETURN_PATH_CONFIDENCE, ROOF_STABILITY, SEAL_INTEGRITY, SLIP_RATIO, SLURRY_LOAD,
    SPOIL_BUFFER_FILL, SubterraneanCommand, SubterraneanOperatingMode, SubterraneanState,
    THERMAL_MARGIN, TOOL_WEAR, VEIN_SIGNAL, WATER_INGRESS_RATIO,
};

pub trait SubterraneanPhysicsSimulator {
    fn step(&mut self, cmd: &SubterraneanCommand, dt: f64);
    fn state(&self) -> &SubterraneanState;
    fn reset(&mut self);
}

/// A simple boring-scout model with depth, spoil, heat, battery, and comms.
pub struct SimpleSubterraneanSimulator {
    state: SubterraneanState,
}

impl SimpleSubterraneanSimulator {
    pub fn new() -> Self {
        Self {
            state: SubterraneanState::home(),
        }
    }

    pub fn state_mut(&mut self) -> &mut SubterraneanState {
        &mut self.state
    }
}

impl SubterraneanPhysicsSimulator for SimpleSubterraneanSimulator {
    fn step(&mut self, cmd: &SubterraneanCommand, dt: f64) {
        let traction = ((cmd.left_track() + cmd.right_track()) as f64 * 0.5).clamp(-1.0, 1.0);
        let reverse_bias = (-traction).max(0.0);
        let boring_cmd = cmd.cutter_head().max(0.0) as f64;
        let auger = cmd.auger_feed().max(0.0) as f64;
        let cooling = cmd.thermal_pump().max(0.0) as f64;
        let ballast = cmd.ballast_trim() as f64;

        let hazard_pressure = (self.state.channels[WATER_INGRESS_RATIO] * 0.28
            + self.state.channels[AQUIFER_RISK] * 0.18
            + self.state.channels[GAS_RISK] * 0.16
            + (1.0 - self.state.channels[ROOF_STABILITY]) * 0.18
            + self.state.channels[SPOIL_BUFFER_FILL] * 0.08
            + self.state.channels[SLIP_RATIO] * 0.07
            + self.state.channels[HULL_STRESS] * 0.05)
            .clamp(0.0, 1.0);
        let effective_boring = boring_cmd * (1.0 - hazard_pressure).clamp(0.0, 1.0);
        let effective_auger = (auger + reverse_bias * 0.2).clamp(0.0, 1.0);

        self.state.channels[DEPTH_M] = (self.state.channels[DEPTH_M]
            + effective_boring * dt * 0.45
            - reverse_bias * dt * 0.18)
            .max(0.0);
        self.state.channels[1] = traction * (1.0 - self.state.channels[SLIP_RATIO]).max(0.05);
        self.state.channels[2] = (self.state.channels[2] + ballast * dt * 0.2).clamp(-0.6, 0.6);
        self.state.channels[3] =
            (self.state.channels[3] + (traction * 0.1 - ballast * 0.05) * dt).clamp(-0.5, 0.5);

        self.state.channels[CUTTER_TEMP_C] = (self.state.channels[CUTTER_TEMP_C]
            + effective_boring * dt * 16.0
            - cooling * dt * 10.0)
            .clamp(0.0, 180.0);
        self.state.channels[5] = (self.state.channels[5]
            + (effective_boring + traction.abs()) * dt * 8.0
            - cooling * dt * 5.0)
            .clamp(0.0, 160.0);
        self.state.channels[SPOIL_BUFFER_FILL] = (self.state.channels[SPOIL_BUFFER_FILL]
            + effective_boring * dt * 0.35
            - effective_auger * dt * 0.45)
            .clamp(0.0, 1.0);
        self.state.channels[BATTERY_RATIO] = (self.state.channels[BATTERY_RATIO]
            - (effective_boring * 0.012
                + effective_auger * 0.004
                + traction.abs() * 0.006
                + cooling * 0.004)
                * dt)
            .clamp(0.0, 1.0);
        self.state.channels[COMM_SIGNAL] = (1.0
            - self.state.channels[DEPTH_M] / 120.0
            - self.state.channels[WATER_INGRESS_RATIO] * 0.25)
            .clamp(0.0, 1.0);
        self.state.channels[SLIP_RATIO] = (0.15
            + self.state.channels[SPOIL_BUFFER_FILL] * 0.35
            + self.state.channels[SLURRY_LOAD] * 0.22
            + self.state.channels[OBSTACLE_PROXIMITY] * 0.25
            - traction.abs() * 0.05)
            .clamp(0.0, 1.0);
        self.state.channels[10] = (0.6 + self.state.channels[DEPTH_M] / 200.0).clamp(0.0, 2.0);
        self.state.channels[11] =
            (self.state.channels[11] * 0.9 + effective_boring * 0.1).clamp(0.0, 1.0);
        self.state.channels[12] = (0.3 + self.state.channels[DEPTH_M] / 300.0).clamp(0.0, 1.0);
        self.state.channels[MAPPING_CONFIDENCE] = (self.state.channels[MAPPING_CONFIDENCE] * 0.97
            + (1.0 - self.state.channels[OBSTACLE_PROXIMITY]) * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[THERMAL_MARGIN] =
            (1.0 - self.state.channels[CUTTER_TEMP_C] / 180.0).clamp(0.0, 1.0);
        self.state.channels[VEIN_SIGNAL] = (0.2 + self.state.channels[DEPTH_M] / 250.0
            - self.state.channels[SLIP_RATIO] * 0.15)
            .clamp(0.0, 1.0);
        self.state.channels[TOOL_WEAR] =
            (self.state.channels[TOOL_WEAR] + effective_boring * dt * 0.002).clamp(0.0, 1.0);
        self.state.channels[HULL_STRESS] = (self.state.channels[HULL_STRESS] * 0.9
            + self.state.channels[SLIP_RATIO] * 0.05
            + self.state.channels[SPOIL_BUFFER_FILL] * 0.04
            + self.state.channels[SLURRY_LOAD] * 0.03)
            .clamp(0.0, 1.0);
        self.state.channels[RETURN_PATH_CONFIDENCE] = (self.state.channels[RETURN_PATH_CONFIDENCE]
            * 0.92
            + self.state.channels[COMM_SIGNAL] * 0.08)
            .clamp(0.0, 1.0);
        self.state.channels[OBSTACLE_PROXIMITY] = (self.state.channels[OBSTACLE_PROXIMITY] * 0.9
            + (self.state.channels[DEPTH_M] / 400.0) * 0.1
            + self.state.channels[SPOIL_BUFFER_FILL] * 0.03)
            .clamp(0.0, 1.0);

        self.state.channels[WATER_INGRESS_RATIO] = (self.state.channels[WATER_INGRESS_RATIO]
            * 0.96
            + self.state.channels[AQUIFER_RISK] * effective_boring * dt * 1.2
            + self.state.channels[12] * dt * 0.04
            - reverse_bias * dt * 0.08)
            .clamp(0.0, 1.0);
        self.state.channels[AQUIFER_RISK] = (0.08
            + self.state.channels[12] * 0.28
            + self.state.channels[DEPTH_M] / 280.0
            + self.state.channels[VEIN_SIGNAL] * 0.12
            + effective_boring * 0.12)
            .clamp(0.0, 1.0);
        self.state.channels[GAS_RISK] = (self.state.channels[GAS_RISK] * 0.95
            + self.state.channels[12] * 0.1
            + self.state.channels[DEPTH_M] / 500.0 * 0.08
            + effective_boring * 0.08
            - reverse_bias * 0.03)
            .clamp(0.0, 1.0);
        self.state.channels[ROOF_STABILITY] = (0.92
            - self.state.channels[OBSTACLE_PROXIMITY] * 0.28
            - self.state.channels[HULL_STRESS] * 0.22
            - self.state.channels[SLIP_RATIO] * 0.2
            - effective_boring * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[LOCALIZATION_CONFIDENCE] =
            (self.state.channels[LOCALIZATION_CONFIDENCE] * 0.94
                + self.state.channels[MAPPING_CONFIDENCE] * 0.03
                + self.state.channels[COMM_SIGNAL] * 0.02
                - self.state.channels[SLIP_RATIO] * 0.04
                - reverse_bias * 0.01)
                .clamp(0.0, 1.0);
        self.state.channels[RELAY_DISTANCE_NORM] =
            (self.state.channels[DEPTH_M] / 200.0).clamp(0.0, 1.0);
        self.state.channels[RELAY_LINK_QUALITY] = (self.state.channels[COMM_SIGNAL]
            * (1.0 - self.state.channels[RELAY_DISTANCE_NORM] * 0.3)
            * (1.0 - self.state.channels[WATER_INGRESS_RATIO] * 0.35))
            .clamp(0.0, 1.0);
        self.state.channels[SEAL_INTEGRITY] = (self.state.channels[SEAL_INTEGRITY] * 0.995
            - self.state.channels[WATER_INGRESS_RATIO] * dt * 0.05
            - self.state.channels[HULL_STRESS] * dt * 0.02
            + cooling * dt * 0.01)
            .clamp(0.0, 1.0);
        self.state.channels[SLURRY_LOAD] = (self.state.channels[SLURRY_LOAD] * 0.96
            + self.state.channels[WATER_INGRESS_RATIO] * 0.07
            + self.state.channels[SPOIL_BUFFER_FILL] * 0.05
            - effective_auger * dt * 0.08
            - reverse_bias * dt * 0.05)
            .clamp(0.0, 1.0);
        self.state.channels[ESCAPE_CONFIDENCE] = (1.0
            - self.state.channels[SLIP_RATIO] * 0.18
            - self.state.channels[SPOIL_BUFFER_FILL] * 0.16
            - self.state.channels[SLURRY_LOAD] * 0.18
            - self.state.channels[WATER_INGRESS_RATIO] * 0.2
            + self.state.channels[RETURN_PATH_CONFIDENCE] * 0.08
            + reverse_bias * 0.06)
            .clamp(0.0, 1.0);
        self.state.channels[ABORT_RECOMMENDATION] = (self.state.channels[WATER_INGRESS_RATIO]
            * 0.2
            + self.state.channels[AQUIFER_RISK] * 0.15
            + self.state.channels[GAS_RISK] * 0.16
            + (1.0 - self.state.channels[ROOF_STABILITY]) * 0.16
            + (1.0 - self.state.channels[ESCAPE_CONFIDENCE]) * 0.14
            + (1.0 - self.state.channels[SEAL_INTEGRITY]) * 0.08
            + self.state.channels[HULL_STRESS] * 0.06
            + (self.state.channels[CUTTER_TEMP_C] / 180.0) * 0.05)
            .clamp(0.0, 1.0);
        self.state.channels[MISSION_PROGRESS] = (self.state.channels[MISSION_PROGRESS]
            + effective_boring * (1.0 - self.state.channels[ABORT_RECOMMENDATION]) * dt * 0.01)
            .clamp(0.0, 1.0);

        match self.state.inferred_mode() {
            SubterraneanOperatingMode::FloodResponse => {
                self.state.channels[MISSION_PROGRESS] *= 0.999;
                self.state.channels[SEAL_INTEGRITY] =
                    (self.state.channels[SEAL_INTEGRITY] + dt * 0.015).clamp(0.0, 1.0);
            }
            SubterraneanOperatingMode::BlackoutAutonomy => {
                self.state.channels[LOCALIZATION_CONFIDENCE] =
                    (self.state.channels[LOCALIZATION_CONFIDENCE] + dt * 0.01).clamp(0.0, 1.0);
            }
            SubterraneanOperatingMode::Retreat | SubterraneanOperatingMode::Surface => {
                self.state.channels[ESCAPE_CONFIDENCE] = (self.state.channels[ESCAPE_CONFIDENCE]
                    + reverse_bias * dt * 0.08)
                    .clamp(0.0, 1.0);
            }
            SubterraneanOperatingMode::Probe
            | SubterraneanOperatingMode::Stabilize
            | SubterraneanOperatingMode::Dig => {}
        }
    }
    fn state(&self) -> &SubterraneanState {
        &self.state
    }
    fn reset(&mut self) {
        self.state = SubterraneanState::home();
    }
}

impl Default for SimpleSubterraneanSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stable() {
        let mut sim = SimpleSubterraneanSimulator::new();
        for _ in 0..1000 {
            sim.step(&SubterraneanCommand::zero(), 0.005);
        }
        assert!(sim.state().is_finite());
    }
    #[test]
    fn test_torque_moves() {
        let mut sim = SimpleSubterraneanSimulator::new();
        let mut cmd = SubterraneanCommand::zero();
        cmd.torques[0] = 1.0;
        let before = sim.state().channels[0];
        sim.step(&cmd, 0.01);
        assert_ne!(sim.state().channels[0], before);
    }

    #[test]
    fn test_boring_without_auger_fills_spoil_buffer_and_heats_cutter() {
        let mut sim = SimpleSubterraneanSimulator::new();
        let mut cmd = SubterraneanCommand::zero();
        cmd.torques[0] = 1.0;
        for _ in 0..200 {
            sim.step(&cmd, 0.02);
        }
        assert!(sim.state().spoil_buffer_fill() > 0.5);
        assert!(sim.state().cutter_temp_c() > 20.0);
    }

    #[test]
    fn test_aquifer_breach_triggers_flood_response_mode() {
        let mut sim = SimpleSubterraneanSimulator::new();
        sim.state.channels[DEPTH_M] = 150.0;
        sim.state.channels[12] = 0.95;
        sim.state.channels[AQUIFER_RISK] = 0.9;
        sim.state.channels[WATER_INGRESS_RATIO] = 0.2;
        let mut cmd = SubterraneanCommand::zero();
        cmd.torques[0] = 1.0;
        for _ in 0..80 {
            sim.step(&cmd, 0.02);
        }
        assert!(sim.state().water_ingress_ratio() > 0.25);
        assert!(matches!(
            sim.state().inferred_mode(),
            SubterraneanOperatingMode::FloodResponse
                | SubterraneanOperatingMode::Retreat
                | SubterraneanOperatingMode::Surface
        ));
    }

    #[test]
    fn test_gas_risk_suppresses_effective_digging() {
        let mut safe = SimpleSubterraneanSimulator::new();
        let mut hazardous = SimpleSubterraneanSimulator::new();
        hazardous.state.channels[GAS_RISK] = 0.95;
        hazardous.state.channels[ROOF_STABILITY] = 0.4;
        hazardous.state.channels[WATER_INGRESS_RATIO] = 0.35;

        let mut cmd = SubterraneanCommand::zero();
        cmd.torques[0] = 1.0;
        safe.step(&cmd, 0.2);
        hazardous.step(&cmd, 0.2);

        assert!(hazardous.state().depth_m() < safe.state().depth_m());
        assert!(hazardous.state().abort_recommendation() > safe.state().abort_recommendation());
    }

    #[test]
    fn test_roof_instability_raises_abort_recommendation() {
        let mut sim = SimpleSubterraneanSimulator::new();
        sim.state.channels[OBSTACLE_PROXIMITY] = 0.95;
        sim.state.channels[SLIP_RATIO] = 0.8;
        sim.state.channels[HULL_STRESS] = 0.8;
        let abort_before = sim.state().abort_recommendation();
        let mut cmd = SubterraneanCommand::zero();
        cmd.torques[0] = 0.7;
        sim.step(&cmd, 0.05);
        assert!(sim.state().roof_stability() < 0.5);
        assert!(sim.state().abort_recommendation() > abort_before);
    }

    #[test]
    fn test_comms_blackout_switches_to_blackout_autonomy() {
        let mut sim = SimpleSubterraneanSimulator::new();
        sim.state.channels[DEPTH_M] = 100.0;
        sim.state.channels[WATER_INGRESS_RATIO] = 0.0;
        sim.step(&SubterraneanCommand::zero(), 0.02);
        assert!(sim.state().relay_link_quality() < 0.2);
        assert_eq!(
            sim.state().inferred_mode(),
            SubterraneanOperatingMode::BlackoutAutonomy
        );
    }

    #[test]
    fn test_spoil_jam_reduces_escape_confidence() {
        let mut sim = SimpleSubterraneanSimulator::new();
        sim.state.channels[SPOIL_BUFFER_FILL] = 0.95;
        sim.state.channels[SLURRY_LOAD] = 0.8;
        sim.state.channels[SLIP_RATIO] = 0.7;
        sim.step(&SubterraneanCommand::zero(), 0.02);
        assert!(sim.state().escape_confidence() < 0.75);
    }
}
