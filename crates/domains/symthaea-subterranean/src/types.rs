// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

pub const NUM_ACTUATORS: usize = 6;
pub const NUM_STATE_CHANNELS: usize = 32;
pub const ACTUATOR_LABELS: [&str; NUM_ACTUATORS] = [
    "cutter_head",
    "auger_feed",
    "left_track",
    "right_track",
    "ballast_trim",
    "thermal_pump",
];
pub const CHANNEL_LABELS: [&str; NUM_STATE_CHANNELS] = [
    "depth_m",
    "forward_velocity_mps",
    "pitch_rad",
    "roll_rad",
    "cutter_temp_c",
    "motor_temp_c",
    "spoil_buffer_fill",
    "battery_ratio",
    "comm_signal",
    "slip_ratio",
    "soil_density",
    "vibration_level",
    "humidity",
    "mapping_confidence",
    "vein_signal",
    "thermal_margin",
    "tool_wear",
    "hull_stress",
    "return_path_confidence",
    "obstacle_proximity",
    "water_ingress_ratio",
    "aquifer_risk",
    "gas_risk",
    "roof_stability",
    "escape_confidence",
    "localization_confidence",
    "relay_link_quality",
    "seal_integrity",
    "slurry_load",
    "abort_recommendation",
    "relay_distance_norm",
    "mission_progress",
];

pub const DEPTH_M: usize = 0;
pub const CUTTER_TEMP_C: usize = 4;
pub const SPOIL_BUFFER_FILL: usize = 6;
pub const BATTERY_RATIO: usize = 7;
pub const COMM_SIGNAL: usize = 8;
pub const SLIP_RATIO: usize = 9;
pub const VEIN_SIGNAL: usize = 14;
pub const TOOL_WEAR: usize = 16;
pub const HULL_STRESS: usize = 17;
pub const OBSTACLE_PROXIMITY: usize = 19;
pub const WATER_INGRESS_RATIO: usize = 20;
pub const AQUIFER_RISK: usize = 21;
pub const GAS_RISK: usize = 22;
pub const ROOF_STABILITY: usize = 23;
pub const ESCAPE_CONFIDENCE: usize = 24;
pub const LOCALIZATION_CONFIDENCE: usize = 25;
pub const RELAY_LINK_QUALITY: usize = 26;
pub const SEAL_INTEGRITY: usize = 27;
pub const SLURRY_LOAD: usize = 28;
pub const ABORT_RECOMMENDATION: usize = 29;
pub const RELAY_DISTANCE_NORM: usize = 30;
pub const MISSION_PROGRESS: usize = 31;
pub const MAPPING_CONFIDENCE: usize = 13;
pub const THERMAL_MARGIN: usize = 15;
pub const RETURN_PATH_CONFIDENCE: usize = 18;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubterraneanOperatingMode {
    Dig,
    Probe,
    Stabilize,
    Retreat,
    Surface,
    BlackoutAutonomy,
    FloodResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubterraneanState {
    pub channels: [f64; NUM_STATE_CHANNELS],
}

impl SubterraneanState {
    pub fn home() -> Self {
        let mut channels = [0.0; NUM_STATE_CHANNELS];
        channels[BATTERY_RATIO] = 1.0;
        channels[COMM_SIGNAL] = 1.0;
        channels[VEIN_SIGNAL] = 0.2;
        channels[TOOL_WEAR] = 0.05;
        channels[MAPPING_CONFIDENCE] = 0.85;
        channels[THERMAL_MARGIN] = 0.95;
        channels[RETURN_PATH_CONFIDENCE] = 0.9;
        channels[ROOF_STABILITY] = 0.92;
        channels[ESCAPE_CONFIDENCE] = 0.95;
        channels[LOCALIZATION_CONFIDENCE] = 0.9;
        channels[RELAY_LINK_QUALITY] = 0.95;
        channels[SEAL_INTEGRITY] = 1.0;
        channels[MISSION_PROGRESS] = 0.0;
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

    pub fn depth_m(&self) -> f64 {
        self.channels[DEPTH_M]
    }
    pub fn cutter_temp_c(&self) -> f64 {
        self.channels[CUTTER_TEMP_C]
    }
    pub fn spoil_buffer_fill(&self) -> f64 {
        self.channels[SPOIL_BUFFER_FILL]
    }
    pub fn battery_ratio(&self) -> f64 {
        self.channels[BATTERY_RATIO]
    }
    pub fn comm_signal(&self) -> f64 {
        self.channels[COMM_SIGNAL]
    }
    pub fn vein_signal(&self) -> f64 {
        self.channels[VEIN_SIGNAL]
    }
    pub fn water_ingress_ratio(&self) -> f64 {
        self.channels[WATER_INGRESS_RATIO]
    }
    pub fn aquifer_risk(&self) -> f64 {
        self.channels[AQUIFER_RISK]
    }
    pub fn gas_risk(&self) -> f64 {
        self.channels[GAS_RISK]
    }
    pub fn roof_stability(&self) -> f64 {
        self.channels[ROOF_STABILITY]
    }
    pub fn escape_confidence(&self) -> f64 {
        self.channels[ESCAPE_CONFIDENCE]
    }
    pub fn localization_confidence(&self) -> f64 {
        self.channels[LOCALIZATION_CONFIDENCE]
    }
    pub fn relay_link_quality(&self) -> f64 {
        self.channels[RELAY_LINK_QUALITY]
    }
    pub fn seal_integrity(&self) -> f64 {
        self.channels[SEAL_INTEGRITY]
    }
    pub fn slurry_load(&self) -> f64 {
        self.channels[SLURRY_LOAD]
    }
    pub fn abort_recommendation(&self) -> f64 {
        self.channels[ABORT_RECOMMENDATION]
    }

    pub fn inferred_mode(&self) -> SubterraneanOperatingMode {
        if self.abort_recommendation() >= 0.92 || self.escape_confidence() <= 0.15 {
            SubterraneanOperatingMode::Surface
        } else if self.water_ingress_ratio() >= 0.55 || self.aquifer_risk() >= 0.75 {
            SubterraneanOperatingMode::FloodResponse
        } else if self.relay_link_quality() <= 0.2 || self.comm_signal() <= 0.15 {
            SubterraneanOperatingMode::BlackoutAutonomy
        } else if self.abort_recommendation() >= 0.7 || self.roof_stability() <= 0.35 {
            SubterraneanOperatingMode::Retreat
        } else if self.abort_recommendation() >= 0.5 || self.cutter_temp_c() >= 120.0 {
            SubterraneanOperatingMode::Stabilize
        } else if self.localization_confidence() <= 0.45 {
            SubterraneanOperatingMode::Probe
        } else {
            SubterraneanOperatingMode::Dig
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SubterraneanCommand {
    pub torques: [f32; NUM_ACTUATORS],
}

impl SubterraneanCommand {
    pub fn zero() -> Self {
        Self {
            torques: [0.0; NUM_ACTUATORS],
        }
    }
    pub fn control_effort(&self) -> f32 {
        self.torques.iter().map(|t| t.abs()).sum::<f32>() / NUM_ACTUATORS as f32
    }
    pub fn cutter_head(&self) -> f32 {
        self.torques[0]
    }
    pub fn auger_feed(&self) -> f32 {
        self.torques[1]
    }
    pub fn left_track(&self) -> f32 {
        self.torques[2]
    }
    pub fn right_track(&self) -> f32 {
        self.torques[3]
    }
    pub fn ballast_trim(&self) -> f32 {
        self.torques[4]
    }
    pub fn thermal_pump(&self) -> f32 {
        self.torques[5]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubterraneanConfig {
    pub genesis_phrase: String,
    pub learning_rate: f32,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub physics_hz: f64,
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
}

impl SubterraneanConfig {
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }
}

impl Default for SubterraneanConfig {
    fn default() -> Self {
        Self {
            genesis_phrase: "symthaea-subterranean".to_string(),
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
        assert!(SubterraneanState::home().is_finite());
    }
    #[test]
    fn test_channels() {
        assert_eq!(
            SubterraneanState::home().to_channels().len(),
            NUM_STATE_CHANNELS
        );
    }
    #[test]
    fn test_zero_cmd() {
        assert_eq!(SubterraneanCommand::zero().control_effort(), 0.0);
    }
}
