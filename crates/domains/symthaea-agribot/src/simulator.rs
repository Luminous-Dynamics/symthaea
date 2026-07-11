// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-pass agribot / stewardship simulator.
use crate::environment::FieldEnvironment;
use crate::types::{
    AgribotCommand, AgribotState, BATTERY_RATIO, CROP_HEALTH, DISEASE_RISK, DROUGHT_RISK,
    FORECAST_CONFIDENCE, HUMAN_PROXIMITY, MISSION_PROGRESS, POLLINATOR_DISTURBANCE_RISK,
    REFILL_RECOMMENDATION, RESERVE_MARGIN, RUNOFF_RISK, SOIL_EXHAUSTION, SOIL_MOISTURE,
    SOIL_NUTRIENTS, TREATMENT_CONFIDENCE, WATER_TANK_RATIO, WATERLOGGING_RISK, WEED_PRESSURE,
    YIELD_FORECAST,
};
use symthaea_core::genesis::GenesisSeed;

pub trait AgribotPhysicsSimulator {
    fn step(&mut self, cmd: &AgribotCommand, dt: f64);
    fn state(&self) -> &AgribotState;
    fn reset(&mut self);
}

/// How fast each sensed channel tracks its ground-truth source (per
/// second). Higher = faster/less-lagged sensing.
const LIGHT_SENSOR_GAIN_DEPLOYED: f64 = 2.0;
const LIGHT_SENSOR_GAIN_STOWED: f64 = 0.3;
const TERRAIN_SENSOR_GAIN: f64 = 1.0;
const HUMAN_SENSOR_GAIN: f64 = 1.5;

/// A simple field-tending model with water, seeding, weed pressure, and crop health.
pub struct SimpleAgribotSimulator {
    state: AgribotState,
    environment: FieldEnvironment,
    genesis: GenesisSeed,
}

impl SimpleAgribotSimulator {
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self {
            state: AgribotState::home(),
            environment: FieldEnvironment::new(genesis),
            genesis: genesis.clone(),
        }
    }

    /// Mutable state access, for tests that need to set up a degraded
    /// starting condition (e.g. low soil moisture) before stepping.
    pub fn state_mut(&mut self) -> &mut AgribotState {
        &mut self.state
    }

    /// Mutable access to the ground-truth field world, for tests that need
    /// to script a real human-proximity or terrain scenario independent of
    /// the robot's own actuator history.
    pub fn environment_mut(&mut self) -> &mut FieldEnvironment {
        &mut self.environment
    }

    pub fn environment(&self) -> &FieldEnvironment {
        &self.environment
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

        // Ground-truth world evolves independently of the robot's own
        // actuation.
        self.environment.step(dt);

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
        // light_level: sensed via a lag chasing the ground-truth ambient
        // light, not manufactured from the mast's own torque. A deployed
        // mast (canopy_sensor_mast) senses faster/more accurately -- it
        // provides a better reading, it does not change the actual light.
        let light_gain = if mast >= 0.3 {
            LIGHT_SENSOR_GAIN_DEPLOYED
        } else {
            LIGHT_SENSOR_GAIN_STOWED
        };
        self.state.channels[3] = (self.state.channels[3]
            + (self.environment.ambient_light - self.state.channels[3]) * light_gain * dt)
            .clamp(0.0, 1.0);
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
        // terrain_roughness: sensed via a lag chasing the ground-truth
        // terrain, not manufactured from the robot's own drive effort.
        self.state.channels[17] = (self.state.channels[17]
            + (self.environment.terrain_roughness - self.state.channels[17])
                * TERRAIN_SENSOR_GAIN
                * dt)
            .clamp(0.0, 1.0);
        // human_proximity: sensed via a lag chasing a real ground-truth
        // human encounter, not the robot's own drive request. This is the
        // safety-critical channel that gates human_gate above -- previously
        // a stationary robot always read zero proximity regardless of
        // whether a human was actually standing next to it.
        let human_truth = if self.environment.human_present {
            self.environment.human_intensity
        } else {
            0.0
        };
        self.state.channels[18] = (self.state.channels[18]
            + (human_truth - self.state.channels[18]) * HUMAN_SENSOR_GAIN * dt)
            .clamp(0.0, 1.0);
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
        self.environment = FieldEnvironment::new(&self.genesis);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sim() -> SimpleAgribotSimulator {
        SimpleAgribotSimulator::new(&GenesisSeed::from_phrase("test"))
    }

    #[test]
    fn test_stable() {
        let mut sim = test_sim();
        for _ in 0..1000 {
            sim.step(&AgribotCommand::zero(), 0.005);
        }
        assert!(sim.state().is_finite());
    }
    #[test]
    fn test_torque_moves() {
        let mut sim = test_sim();
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
        let mut sim = test_sim();
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
        let mut sim = test_sim();
        sim.state.channels[DISEASE_RISK] = 0.75;
        sim.state.channels[15] = 0.75;
        assert_eq!(
            sim.state().inferred_mode(),
            crate::types::AgribotOperatingMode::DiseaseControl
        );
    }

    #[test]
    fn test_human_proximity_suppresses_tool_aggression() {
        let mut sim = test_sim();
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
        let mut sim = test_sim();
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
        let mut sim = test_sim();
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
        let mut sim = test_sim();
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

    #[test]
    fn test_human_proximity_is_falsifiable_without_encounter() {
        // Regression: human_proximity used to be `drive_request * 0.2` --
        // pure heavy driving with no human ever nearby now must NOT read a
        // high proximity, unlike the old actuator-only formula. Ground
        // truth is re-pinned absent every step so this doesn't depend on
        // the environment's own stochastic arrival roll never firing.
        let mut sim = test_sim();
        let mut cmd = AgribotCommand::zero();
        cmd.torques[0] = 1.0;
        cmd.torques[1] = 1.0;
        for _ in 0..1000 {
            sim.environment_mut().human_present = false;
            sim.environment_mut().human_intensity = 0.0;
            sim.step(&cmd, 0.01);
        }
        assert!(
            sim.state().channels[HUMAN_PROXIMITY] < 0.2,
            "human_proximity must stay low with no ground-truth encounter even under heavy sustained driving, got {}",
            sim.state().channels[HUMAN_PROXIMITY]
        );
    }

    #[test]
    fn test_human_proximity_tracks_ground_truth_encounter() {
        // The flip side: a scripted real human nearby, with the robot
        // stationary, must still be sensed -- unlike the old formula which
        // required drive_request > 0 to read anything at all.
        let mut sim = test_sim();
        for _ in 0..300 {
            sim.environment_mut().human_present = true;
            sim.environment_mut().human_intensity = 0.9;
            sim.step(&AgribotCommand::zero(), 0.05);
        }
        assert!(
            sim.state().channels[HUMAN_PROXIMITY] > 0.6,
            "human_proximity must rise once the ground truth has a human present, even with the robot stationary, got {}",
            sim.state().channels[HUMAN_PROXIMITY]
        );
    }

    #[test]
    fn test_terrain_roughness_is_not_purely_drive_derived() {
        // Regression: terrain_roughness used to be `0.2 + drive_request *
        // 0.15` -- heavy sustained driving must not inflate it beyond the
        // ground-truth terrain, which is pinned flat here.
        let mut sim = test_sim();
        let mut cmd = AgribotCommand::zero();
        cmd.torques[0] = 1.0;
        cmd.torques[1] = 1.0;
        for _ in 0..500 {
            sim.environment_mut().terrain_roughness = 0.1;
            sim.step(&cmd, 0.01);
        }
        assert!(
            sim.state().channels[17] < 0.3,
            "terrain_roughness must track ground truth, not the robot's own drive effort, got {}",
            sim.state().channels[17]
        );
    }
}
