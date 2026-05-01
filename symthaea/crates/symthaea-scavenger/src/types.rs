// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

pub const NUM_ACTUATORS: usize = 10;
pub const NUM_STATE_CHANNELS: usize = 32;
pub const ACTUATOR_LABELS: [&str; NUM_ACTUATORS] = [
    "left_track",
    "right_track",
    "arm_lift",
    "arm_extend",
    "gripper",
    "cutter",
    "sorter",
    "hopper_feed",
    "compactor",
    "dust_suppression",
];
pub const CHANNEL_LABELS: [&str; NUM_STATE_CHANNELS] = [
    "left_track_vel",
    "right_track_vel",
    "arm_height",
    "arm_extension",
    "gripper_aperture",
    "cutter_rpm",
    "sorter_bias",
    "hopper_fill",
    "compactor_load",
    "dust_level",
    "battery_ratio",
    "thermal_load",
    "recyclable_fraction",
    "hazard_fraction",
    "ferrous_signal",
    "plastic_signal",
    "glass_signal",
    "organic_signal",
    "tool_wear",
    "jam_risk",
    "salvage_value_rate",
    "scene_coverage",
    "classification_confidence",
    "human_proximity",
    "containment_breach_risk",
    "salvage_purity",
    "incident_risk",
    "contamination_risk",
    "toxic_dust_risk",
    "quarantine_load_ratio",
    "blade_bind_risk",
    "sorter_clog_risk",
];

pub const HOPPER_FILL: usize = 7;
pub const BATTERY_RATIO: usize = 10;
pub const THERMAL_LOAD: usize = 11;
pub const TOOL_WEAR: usize = 18;
pub const JAM_RISK: usize = 19;
pub const SALVAGE_VALUE_RATE: usize = 20;
pub const HUMAN_PROXIMITY: usize = 23;
pub const INCIDENT_RISK: usize = 26;
pub const CONTAMINATION_RISK: usize = 27;
pub const TOXIC_DUST_RISK: usize = 28;
pub const QUARANTINE_LOAD_RATIO: usize = 29;
pub const BLADE_BIND_RISK: usize = 30;
pub const SORTER_CLOG_RISK: usize = 31;
pub const CONTAINMENT_BREACH_RISK: usize = 24;
pub const SALVAGE_PURITY: usize = 25;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScavengerOperatingMode {
    Recovery,
    DustControl,
    JamRecovery,
    Quarantine,
    HumanSafe,
    EmergencyStop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScavengerState {
    pub channels: [f64; NUM_STATE_CHANNELS],
}

impl ScavengerState {
    pub fn home() -> Self {
        let mut channels = [0.0; NUM_STATE_CHANNELS];
        channels[BATTERY_RATIO] = 1.0;
        channels[12] = 0.45;
        channels[14] = 0.35;
        channels[15] = 0.30;
        channels[16] = 0.20;
        channels[17] = 0.15;
        channels[22] = 0.6;
        channels[SALVAGE_PURITY] = 0.8;
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

    pub fn hopper_fill(&self) -> f64 {
        self.channels[HOPPER_FILL]
    }
    pub fn battery_ratio(&self) -> f64 {
        self.channels[BATTERY_RATIO]
    }
    pub fn salvage_value_rate(&self) -> f64 {
        self.channels[SALVAGE_VALUE_RATE]
    }
    pub fn incident_risk(&self) -> f64 {
        self.channels[INCIDENT_RISK]
    }
    pub fn jam_risk(&self) -> f64 {
        self.channels[JAM_RISK]
    }
    pub fn contamination_risk(&self) -> f64 {
        self.channels[CONTAMINATION_RISK]
    }
    pub fn toxic_dust_risk(&self) -> f64 {
        self.channels[TOXIC_DUST_RISK]
    }
    pub fn quarantine_load_ratio(&self) -> f64 {
        self.channels[QUARANTINE_LOAD_RATIO]
    }
    pub fn salvage_purity(&self) -> f64 {
        self.channels[SALVAGE_PURITY]
    }

    pub fn inferred_mode(&self) -> ScavengerOperatingMode {
        if self.incident_risk() >= 0.85 || self.channels[CONTAINMENT_BREACH_RISK] >= 0.8 {
            ScavengerOperatingMode::EmergencyStop
        } else if self.contamination_risk() >= 0.55 || self.quarantine_load_ratio() >= 0.3 {
            ScavengerOperatingMode::Quarantine
        } else if self.human_proximity() >= 0.55 {
            ScavengerOperatingMode::HumanSafe
        } else if self.jam_risk() >= 0.55 || self.channels[BLADE_BIND_RISK] >= 0.5 {
            ScavengerOperatingMode::JamRecovery
        } else if self.toxic_dust_risk() >= 0.45 {
            ScavengerOperatingMode::DustControl
        } else {
            ScavengerOperatingMode::Recovery
        }
    }

    pub fn human_proximity(&self) -> f64 {
        self.channels[HUMAN_PROXIMITY]
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScavengerCommand {
    pub torques: [f32; NUM_ACTUATORS],
}

impl ScavengerCommand {
    pub fn zero() -> Self {
        Self {
            torques: [0.0; NUM_ACTUATORS],
        }
    }
    pub fn control_effort(&self) -> f32 {
        self.torques.iter().map(|t| t.abs()).sum::<f32>() / NUM_ACTUATORS as f32
    }
    pub fn cutter(&self) -> f32 {
        self.torques[5]
    }
    pub fn sorter(&self) -> f32 {
        self.torques[6]
    }
    pub fn hopper_feed(&self) -> f32 {
        self.torques[7]
    }
    pub fn compactor(&self) -> f32 {
        self.torques[8]
    }
    pub fn dust_suppression(&self) -> f32 {
        self.torques[9]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScavengerConfig {
    pub genesis_phrase: String,
    pub learning_rate: f32,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub physics_hz: f64,
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
}

impl ScavengerConfig {
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }
}

impl Default for ScavengerConfig {
    fn default() -> Self {
        Self {
            genesis_phrase: "symthaea-scavenger".to_string(),
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
        assert!(ScavengerState::home().is_finite());
    }
    #[test]
    fn test_channels() {
        assert_eq!(
            ScavengerState::home().to_channels().len(),
            NUM_STATE_CHANNELS
        );
    }
    #[test]
    fn test_zero_cmd() {
        assert_eq!(ScavengerCommand::zero().control_effort(), 0.0);
    }
}
