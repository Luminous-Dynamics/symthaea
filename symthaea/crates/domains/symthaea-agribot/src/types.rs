// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

pub const NUM_ACTUATORS: usize = 7;
pub const NUM_STATE_CHANNELS: usize = 30;
pub const ACTUATOR_LABELS: [&str; NUM_ACTUATORS] = [
    "left_drive",
    "right_drive",
    "arm_lift",
    "tool_head",
    "water_pump",
    "seed_dispenser",
    "canopy_sensor_mast",
];
pub const CHANNEL_LABELS: [&str; NUM_STATE_CHANNELS] = [
    "soil_moisture",
    "soil_nutrients",
    "canopy_temp_c",
    "light_level",
    "crop_health",
    "weed_pressure",
    "water_tank_ratio",
    "battery_ratio",
    "tool_wear",
    "ground_compaction",
    "coverage_ratio",
    "seeding_progress",
    "irrigation_rate",
    "evaporation_stress",
    "yield_forecast",
    "disease_risk",
    "pollinator_activity",
    "terrain_roughness",
    "human_proximity",
    "wind_stress",
    "forecast_confidence",
    "mission_progress",
    "drought_risk",
    "waterlogging_risk",
    "soil_exhaustion",
    "pollinator_disturbance_risk",
    "runoff_risk",
    "reserve_margin",
    "treatment_confidence",
    "refill_recommendation",
];

pub const SOIL_MOISTURE: usize = 0;
pub const SOIL_NUTRIENTS: usize = 1;
pub const CROP_HEALTH: usize = 4;
pub const WATER_TANK_RATIO: usize = 6;
pub const BATTERY_RATIO: usize = 7;
pub const WEED_PRESSURE: usize = 5;
pub const YIELD_FORECAST: usize = 14;
pub const DISEASE_RISK: usize = 15;
pub const HUMAN_PROXIMITY: usize = 18;
pub const FORECAST_CONFIDENCE: usize = 20;
pub const MISSION_PROGRESS: usize = 21;
pub const DROUGHT_RISK: usize = 22;
pub const WATERLOGGING_RISK: usize = 23;
pub const SOIL_EXHAUSTION: usize = 24;
pub const POLLINATOR_DISTURBANCE_RISK: usize = 25;
pub const RUNOFF_RISK: usize = 26;
pub const RESERVE_MARGIN: usize = 27;
pub const TREATMENT_CONFIDENCE: usize = 28;
pub const REFILL_RECOMMENDATION: usize = 29;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgribotOperatingMode {
    Stewardship,
    IrrigationRecovery,
    DiseaseControl,
    SoilProtection,
    PollinatorSafe,
    HumanSafe,
    RefillReturn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgribotState {
    pub channels: [f64; NUM_STATE_CHANNELS],
}

impl AgribotState {
    pub fn home() -> Self {
        let mut channels = [0.0; NUM_STATE_CHANNELS];
        channels[SOIL_MOISTURE] = 0.5;
        channels[SOIL_NUTRIENTS] = 0.6;
        channels[2] = 24.0;
        channels[3] = 0.7;
        channels[CROP_HEALTH] = 0.65;
        channels[WATER_TANK_RATIO] = 1.0;
        channels[BATTERY_RATIO] = 1.0;
        channels[16] = 0.5;
        channels[FORECAST_CONFIDENCE] = 0.7;
        channels[TREATMENT_CONFIDENCE] = 0.75;
        channels[RESERVE_MARGIN] = 1.0;
        Self { channels }
    }
    pub fn to_channels(&self) -> [f32; NUM_STATE_CHANNELS] {
        let mut c = [0.0f32; NUM_STATE_CHANNELS];
        for i in 0..NUM_STATE_CHANNELS {
            c[i] = self.channels[i] as f32;
        }
        c
    }
    pub fn is_finite(&self) -> bool {
        self.channels.iter().all(|v| v.is_finite())
    }

    pub fn soil_moisture(&self) -> f64 {
        self.channels[SOIL_MOISTURE]
    }
    pub fn soil_nutrients(&self) -> f64 {
        self.channels[SOIL_NUTRIENTS]
    }
    pub fn crop_health(&self) -> f64 {
        self.channels[CROP_HEALTH]
    }
    pub fn water_tank_ratio(&self) -> f64 {
        self.channels[WATER_TANK_RATIO]
    }
    pub fn battery_ratio(&self) -> f64 {
        self.channels[BATTERY_RATIO]
    }
    pub fn weed_pressure(&self) -> f64 {
        self.channels[WEED_PRESSURE]
    }
    pub fn yield_forecast(&self) -> f64 {
        self.channels[YIELD_FORECAST]
    }
    pub fn disease_risk(&self) -> f64 {
        self.channels[DISEASE_RISK]
    }
    pub fn drought_risk(&self) -> f64 {
        self.channels[DROUGHT_RISK]
    }
    pub fn waterlogging_risk(&self) -> f64 {
        self.channels[WATERLOGGING_RISK]
    }
    pub fn soil_exhaustion(&self) -> f64 {
        self.channels[SOIL_EXHAUSTION]
    }
    pub fn pollinator_disturbance_risk(&self) -> f64 {
        self.channels[POLLINATOR_DISTURBANCE_RISK]
    }
    pub fn runoff_risk(&self) -> f64 {
        self.channels[RUNOFF_RISK]
    }
    pub fn reserve_margin(&self) -> f64 {
        self.channels[RESERVE_MARGIN]
    }
    pub fn treatment_confidence(&self) -> f64 {
        self.channels[TREATMENT_CONFIDENCE]
    }
    pub fn refill_recommendation(&self) -> f64 {
        self.channels[REFILL_RECOMMENDATION]
    }

    pub fn inferred_mode(&self) -> AgribotOperatingMode {
        if self.refill_recommendation() >= 0.7 || self.reserve_margin() <= 0.2 {
            AgribotOperatingMode::RefillReturn
        } else if self.channels[HUMAN_PROXIMITY] >= 0.55 {
            AgribotOperatingMode::HumanSafe
        } else if self.pollinator_disturbance_risk() >= 0.55 {
            AgribotOperatingMode::PollinatorSafe
        } else if self.soil_exhaustion() >= 0.65 || self.channels[9] >= 0.6 {
            AgribotOperatingMode::SoilProtection
        } else if self.disease_risk() >= 0.55 {
            AgribotOperatingMode::DiseaseControl
        } else if self.drought_risk() >= 0.5 {
            AgribotOperatingMode::IrrigationRecovery
        } else {
            AgribotOperatingMode::Stewardship
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AgribotCommand {
    pub torques: [f32; NUM_ACTUATORS],
}

impl AgribotCommand {
    pub fn zero() -> Self {
        Self {
            torques: [0.0; NUM_ACTUATORS],
        }
    }
    pub fn control_effort(&self) -> f32 {
        self.torques.iter().map(|t| t.abs()).sum::<f32>() / NUM_ACTUATORS as f32
    }
    pub fn water_pump(&self) -> f32 {
        self.torques[4]
    }
    pub fn seed_dispenser(&self) -> f32 {
        self.torques[5]
    }
    pub fn tool_head(&self) -> f32 {
        self.torques[3]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgribotConfig {
    pub genesis_phrase: String,
    pub learning_rate: f32,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub physics_hz: f64,
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
}

impl AgribotConfig {
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }
}

impl Default for AgribotConfig {
    fn default() -> Self {
        Self {
            genesis_phrase: "symthaea-agribot".to_string(),
            learning_rate: 0.001,
            network_layers: 3,
            neurons_per_layer: 8,
            physics_hz: 200.0,
            cognitive_interval: 10,
            steps_per_episode: 2000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_home() {
        assert!(AgribotState::home().is_finite());
    }
    #[test]
    fn test_channels() {
        assert_eq!(AgribotState::home().to_channels().len(), NUM_STATE_CHANNELS);
    }
    #[test]
    fn test_zero_cmd() {
        assert_eq!(AgribotCommand::zero().control_effort(), 0.0);
    }
}
