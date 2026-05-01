// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-pass agribot / stewardship simulator.
use crate::types::{
    AgribotCommand, AgribotState, BATTERY_RATIO, CROP_HEALTH, DISEASE_RISK, DROUGHT_RISK,
    FORECAST_CONFIDENCE, HUMAN_PROXIMITY, MISSION_PROGRESS, POLLINATOR_DISTURBANCE_RISK,
    REFILL_RECOMMENDATION, RESERVE_MARGIN, RUNOFF_RISK, SOIL_EXHAUSTION, SOIL_MOISTURE,
    SOIL_NUTRIENTS, TREATMENT_CONFIDENCE, WATERLOGGING_RISK, WATER_TANK_RATIO, WEED_PRESSURE,
    YIELD_FORECAST,
};

pub trait AgribotPhysicsSimulator {
    fn step(&mut self, cmd: &AgribotCommand, dt: f64);
    fn state(&self) -> &AgribotState;
    fn reset(&mut self);
}

/// A simple field-tending model with water, seeding, weed pressure, and crop health.
pub struct SimpleAgribotSimulator {
    state: AgribotState,
}

impl SimpleAgribotSimulator {
    pub fn new() -> Self {
        Self {
            state: AgribotState::home(),
        }
    }
}

impl AgribotPhysicsSimulator for SimpleAgribotSimulator {
    fn step(&mut self, cmd: &AgribotCommand, dt: f64) {
        let drive_request = (cmd.torques[0].abs() + cmd.torques[1].abs()) as f64 * 0.5;
        let arm = cmd.torques[2].abs() as f64;
        let tool_request = cmd.tool_head().max(0.0) as f64;
        let watering_request = cmd.water_pump().max(0.0) as f64;
        let seeding_request = cmd.seed_dispenser().max(0.0) as f64;
        let mast = cmd.torques[6].max(0.0) as f64;

        let human_gate = if self.state.channels[HUMAN_PROXIMITY] >= 0.55 {
            0.2
        } else {
            1.0
        };
        let pollinator_gate = if self.state.channels[16] >= 0.72
            || self.state.channels[POLLINATOR_DISTURBANCE_RISK] >= 0.55
        {
            0.35
        } else {
            1.0
        };
        let soil_gate =
            if self.state.channels[9] >= 0.6 || self.state.channels[SOIL_EXHAUSTION] >= 0.65 {
                0.45
            } else {
                1.0
            };
        let reserve_gate = if self.state.channels[RESERVE_MARGIN] <= 0.18 {
            0.25
        } else {
            1.0
        };

        let drive = drive_request * soil_gate * reserve_gate;
        let tool = tool_request * human_gate * pollinator_gate;
        let seeding = seeding_request * human_gate * pollinator_gate * reserve_gate;
        let mut watering = watering_request * reserve_gate;
        if self.state.channels[WATERLOGGING_RISK] >= 0.6 || self.state.channels[RUNOFF_RISK] >= 0.55
        {
            watering *= 0.3;
        }

        self.state.channels[SOIL_MOISTURE] = (self.state.channels[SOIL_MOISTURE]
            + watering * dt * 0.2
            - self.state.channels[13] * dt * (0.05 + self.state.channels[19] * 0.03))
            .clamp(0.0, 1.0);
        self.state.channels[SOIL_NUTRIENTS] =
            (self.state.channels[SOIL_NUTRIENTS] - seeding * dt * 0.01 + tool * dt * 0.015
                - self.state.channels[SOIL_EXHAUSTION] * dt * 0.015)
                .clamp(0.0, 1.0);
        self.state.channels[2] = (self.state.channels[2] + self.state.channels[13] * dt * 8.0
            - watering * dt * 5.0)
            .clamp(-10.0, 60.0);
        self.state.channels[3] = (0.55 + mast * 0.35).clamp(0.0, 1.0);
        self.state.channels[CROP_HEALTH] = (self.state.channels[CROP_HEALTH]
            + self.state.channels[SOIL_MOISTURE] * dt * 0.04
            + self.state.channels[SOIL_NUTRIENTS] * dt * 0.03
            - self.state.channels[WEED_PRESSURE] * dt * 0.05
            - self.state.channels[DISEASE_RISK] * dt * 0.04
            - self.state.channels[SOIL_EXHAUSTION] * dt * 0.03
            - self.state.channels[WATERLOGGING_RISK] * dt * 0.025)
            .clamp(0.0, 1.0);
        self.state.channels[WEED_PRESSURE] = (self.state.channels[WEED_PRESSURE]
            - tool * dt * 0.08
            + drive * dt * 0.01
            + self.state.channels[DROUGHT_RISK] * dt * 0.015)
            .clamp(0.0, 1.0);
        self.state.channels[WATER_TANK_RATIO] =
            (self.state.channels[WATER_TANK_RATIO] - watering * dt * 0.03).clamp(0.0, 1.0);
        self.state.channels[BATTERY_RATIO] = (self.state.channels[BATTERY_RATIO]
            - (drive * 0.005 + arm * 0.003 + tool * 0.006 + watering * 0.004) * dt)
            .clamp(0.0, 1.0);
        self.state.channels[8] = (self.state.channels[8] + tool * dt * 0.002).clamp(0.0, 1.0);
        self.state.channels[9] =
            (self.state.channels[9] + drive_request * dt * 0.02 - tool * dt * 0.01).clamp(0.0, 1.0);
        self.state.channels[10] = (self.state.channels[10] + drive * dt * 0.015).clamp(0.0, 1.0);
        self.state.channels[11] = (self.state.channels[11] + seeding * dt * 0.03).clamp(0.0, 1.0);
        self.state.channels[12] = watering;
        self.state.channels[13] =
            (0.25 + (1.0 - self.state.channels[SOIL_MOISTURE]) * 0.35).clamp(0.0, 1.0);
        self.state.channels[YIELD_FORECAST] = (self.state.channels[CROP_HEALTH] * 0.5
            + self.state.channels[SOIL_NUTRIENTS] * 0.2
            + self.state.channels[SOIL_MOISTURE] * 0.15)
            .clamp(0.0, 1.0);
        self.state.channels[DISEASE_RISK] = (self.state.channels[13] * 0.35
            + self.state.channels[2] / 60.0 * 0.2
            + self.state.channels[WEED_PRESSURE] * 0.2
            + self.state.channels[WATERLOGGING_RISK] * 0.25)
            .clamp(0.0, 1.0);
        self.state.channels[16] =
            (self.state.channels[16] * 0.97 + self.state.channels[3] * 0.03).clamp(0.0, 1.0);
        self.state.channels[17] = (0.2 + drive_request * 0.15).clamp(0.0, 1.0);
        self.state.channels[18] = (drive_request * 0.2).clamp(0.0, 1.0);
        self.state.channels[19] = (0.15 + self.state.channels[13] * 0.2).clamp(0.0, 1.0);
        self.state.channels[FORECAST_CONFIDENCE] = (self.state.channels[FORECAST_CONFIDENCE] * 0.9
            + self.state.channels[YIELD_FORECAST] * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[MISSION_PROGRESS] = (self.state.channels[MISSION_PROGRESS]
            + self.state.channels[10] * dt * 0.02)
            .clamp(0.0, 1.0);
        self.state.channels[DROUGHT_RISK] = ((1.0 - self.state.channels[SOIL_MOISTURE]) * 0.55
            + self.state.channels[19] * 0.2
            + (1.0 - self.state.channels[WATER_TANK_RATIO]) * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[WATERLOGGING_RISK] = (self.state.channels[SOIL_MOISTURE] * 0.45
            + watering * 0.25
            + self.state.channels[RUNOFF_RISK] * 0.15)
            .clamp(0.0, 1.0);
        self.state.channels[SOIL_EXHAUSTION] = (self.state.channels[SOIL_EXHAUSTION]
            + drive_request * dt * 0.018
            + seeding * dt * 0.008
            - tool * dt * 0.01)
            .clamp(0.0, 1.0);
        self.state.channels[POLLINATOR_DISTURBANCE_RISK] =
            (self.state.channels[16] * 0.45 + tool_request * 0.2 + drive_request * 0.1)
                .clamp(0.0, 1.0);
        self.state.channels[RUNOFF_RISK] = (watering_request * 0.35
            + self.state.channels[WATERLOGGING_RISK] * 0.3
            + self.state.channels[17] * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[RESERVE_MARGIN] =
            self.state.channels[WATER_TANK_RATIO].min(self.state.channels[BATTERY_RATIO]);
        self.state.channels[TREATMENT_CONFIDENCE] = (self.state.channels[3] * 0.35
            + (1.0 - self.state.channels[17]) * 0.1
            + self.state.channels[FORECAST_CONFIDENCE] * 0.25
            + (1.0 - self.state.channels[DISEASE_RISK]) * 0.2)
            .clamp(0.0, 1.0);
        self.state.channels[REFILL_RECOMMENDATION] = ((1.0 - self.state.channels[RESERVE_MARGIN])
            * 0.6
            + self.state.channels[DROUGHT_RISK] * 0.15
            + self.state.channels[MISSION_PROGRESS] * 0.1)
            .clamp(0.0, 1.0);
    }
    fn state(&self) -> &AgribotState {
        &self.state
    }
    fn reset(&mut self) {
        self.state = AgribotState::home();
    }
}

impl Default for SimpleAgribotSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stable() {
        let mut sim = SimpleAgribotSimulator::new();
        for _ in 0..1000 {
            sim.step(&AgribotCommand::zero(), 0.005);
        }
        assert!(sim.state().is_finite());
    }
    #[test]
    fn test_torque_moves() {
        let mut sim = SimpleAgribotSimulator::new();
        let mut cmd = AgribotCommand::zero();
        cmd.torques[0] = 1.0;
        let coverage_before = sim.state().channels[10];
        let battery_before = sim.state().channels[BATTERY_RATIO];
        sim.step(&cmd, 0.01);
        assert!(sim.state().channels[10] > coverage_before);
        assert!(sim.state().channels[BATTERY_RATIO] < battery_before);
    }

    #[test]
    fn test_watering_recovers_soil_moisture_under_drought() {
        let mut sim = SimpleAgribotSimulator::new();
        sim.state.channels[0] = 0.1;
        let mut cmd = AgribotCommand::zero();
        cmd.torques[4] = 1.0;
        for _ in 0..120 {
            sim.step(&cmd, 0.02);
        }
        assert!(sim.state().soil_moisture() > 0.1);
        assert!(sim.state().water_tank_ratio() < 1.0);
    }

    #[test]
    fn test_disease_pressure_pushes_mode_into_disease_control() {
        let mut sim = SimpleAgribotSimulator::new();
        sim.state.channels[DISEASE_RISK] = 0.75;
        sim.state.channels[15] = 0.75;
        assert_eq!(
            sim.state().inferred_mode(),
            crate::types::AgribotOperatingMode::DiseaseControl
        );
    }

    #[test]
    fn test_human_proximity_suppresses_tool_aggression() {
        let mut sim = SimpleAgribotSimulator::new();
        sim.state.channels[HUMAN_PROXIMITY] = 0.8;
        sim.state.channels[5] = 0.7;
        assert_eq!(
            sim.state().inferred_mode(),
            crate::types::AgribotOperatingMode::HumanSafe
        );
        let mut cmd = AgribotCommand::zero();
        cmd.torques[3] = 1.0;
        sim.step(&cmd, 0.05);
        assert!(sim.state().channels[5] > 0.68);
    }

    #[test]
    fn test_overwatering_drives_waterlogging_and_runoff_risk() {
        let mut sim = SimpleAgribotSimulator::new();
        sim.state.channels[SOIL_MOISTURE] = 0.92;
        let mut cmd = AgribotCommand::zero();
        cmd.torques[4] = 1.0;
        for _ in 0..80 {
            sim.step(&cmd, 0.03);
        }
        assert!(sim.state().waterlogging_risk() > 0.45);
        assert!(sim.state().runoff_risk() > 0.3);
    }

    #[test]
    fn test_compaction_pushes_soil_protection_mode() {
        let mut sim = SimpleAgribotSimulator::new();
        sim.state.channels[SOIL_MOISTURE] = 0.9;
        sim.state.channels[13] = 0.05;
        let mut cmd = AgribotCommand::zero();
        cmd.torques[0] = 1.0;
        cmd.torques[1] = 1.0;
        for _ in 0..800 {
            sim.step(&cmd, 0.02);
        }
        assert!(sim.state().soil_exhaustion() > 0.12);
        sim.state.channels[9] = 0.7;
        assert_eq!(
            sim.state().inferred_mode(),
            crate::types::AgribotOperatingMode::SoilProtection
        );
    }

    #[test]
    fn test_low_reserve_recommends_refill_return() {
        let mut sim = SimpleAgribotSimulator::new();
        sim.state.channels[WATER_TANK_RATIO] = 0.08;
        sim.state.channels[BATTERY_RATIO] = 0.12;
        let mut cmd = AgribotCommand::zero();
        cmd.torques[4] = 1.0;
        sim.step(&cmd, 0.05);
        assert!(sim.state().reserve_margin() < 0.2);
        assert!(sim.state().refill_recommendation() > 0.5);
        assert_eq!(
            sim.state().inferred_mode(),
            crate::types::AgribotOperatingMode::RefillReturn
        );
    }
}
