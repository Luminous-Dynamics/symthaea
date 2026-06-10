// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

pub const NUM_ACTUATORS: usize = 8;
pub const NUM_STATE_CHANNELS: usize = 28;
pub const ACTUATOR_LABELS: [&str; NUM_ACTUATORS] = [
    "ventilation_fan",
    "filtration_loop",
    "cooling_loop",
    "heating_loop",
    "humidifier",
    "dehumidifier",
    "light_brightness",
    "light_circadian_shift",
];
pub const CHANNEL_LABELS: [&str; NUM_STATE_CHANNELS] = [
    "air_quality",
    "co2_load",
    "humidity",
    "humidity_stress",
    "thermal_stress",
    "cold_stress",
    "occupancy_pressure",
    "occupancy_confidence",
    "light_level",
    "circadian_mismatch",
    "noise_stress",
    "utility_reserve",
    "ventilation_authority",
    "filtration_effectiveness",
    "cooling_margin",
    "heating_margin",
    "zone_isolation_confidence",
    "smoke_risk",
    "contamination_risk",
    "sensor_confidence",
    "night_mode_pressure",
    "public_health_risk",
    "comfort_integrity",
    "energy_efficiency",
    "mission_progress",
    "thermal_balance",
    "quiet_compliance",
    "habitat_continuity",
];

pub const AIR_QUALITY: usize = 0;
pub const CO2_LOAD: usize = 1;
pub const HUMIDITY: usize = 2;
pub const HUMIDITY_STRESS: usize = 3;
pub const THERMAL_STRESS: usize = 4;
pub const COLD_STRESS: usize = 5;
pub const OCCUPANCY_PRESSURE: usize = 6;
pub const OCCUPANCY_CONFIDENCE: usize = 7;
pub const LIGHT_LEVEL: usize = 8;
pub const CIRCADIAN_MISMATCH: usize = 9;
pub const NOISE_STRESS: usize = 10;
pub const UTILITY_RESERVE: usize = 11;
pub const VENTILATION_AUTHORITY: usize = 12;
pub const FILTRATION_EFFECTIVENESS: usize = 13;
pub const COOLING_MARGIN: usize = 14;
pub const HEATING_MARGIN: usize = 15;
pub const ZONE_ISOLATION_CONFIDENCE: usize = 16;
pub const SMOKE_RISK: usize = 17;
pub const CONTAMINATION_RISK: usize = 18;
pub const SENSOR_CONFIDENCE: usize = 19;
pub const NIGHT_MODE_PRESSURE: usize = 20;
pub const PUBLIC_HEALTH_RISK: usize = 21;
pub const COMFORT_INTEGRITY: usize = 22;
pub const MISSION_PROGRESS: usize = 24;
pub const QUIET_COMPLIANCE: usize = 26;
pub const HABITAT_CONTINUITY: usize = 27;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClimeOperatingMode {
    BalancedHabitat,
    AirRecovery,
    ThermalRecovery,
    CircadianSupport,
    QuietNight,
    UtilityConstrained,
    IsolationMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimeState {
    pub channels: [f64; NUM_STATE_CHANNELS],
}

impl ClimeState {
    pub fn home() -> Self {
        let mut channels = [0.0; NUM_STATE_CHANNELS];
        channels[AIR_QUALITY] = 0.88;
        channels[HUMIDITY] = 0.5;
        channels[OCCUPANCY_CONFIDENCE] = 0.82;
        channels[UTILITY_RESERVE] = 0.8;
        channels[VENTILATION_AUTHORITY] = 0.82;
        channels[FILTRATION_EFFECTIVENESS] = 0.84;
        channels[COOLING_MARGIN] = 0.78;
        channels[HEATING_MARGIN] = 0.76;
        channels[ZONE_ISOLATION_CONFIDENCE] = 0.72;
        channels[SENSOR_CONFIDENCE] = 0.85;
        channels[COMFORT_INTEGRITY] = 0.88;
        channels[QUIET_COMPLIANCE] = 0.82;
        channels[HABITAT_CONTINUITY] = 0.9;
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

    pub fn air_quality(&self) -> f64 {
        self.channels[AIR_QUALITY]
    }
    pub fn thermal_stress(&self) -> f64 {
        self.channels[THERMAL_STRESS]
    }
    pub fn utility_reserve(&self) -> f64 {
        self.channels[UTILITY_RESERVE]
    }
    pub fn comfort_integrity(&self) -> f64 {
        self.channels[COMFORT_INTEGRITY]
    }
    pub fn habitat_continuity(&self) -> f64 {
        self.channels[HABITAT_CONTINUITY]
    }

    pub fn inferred_mode(&self) -> ClimeOperatingMode {
        if self.channels[SMOKE_RISK] >= 0.6
            || self.channels[CONTAMINATION_RISK] >= 0.6
            || self.channels[PUBLIC_HEALTH_RISK] >= 0.7
        {
            ClimeOperatingMode::IsolationMode
        } else if self.channels[UTILITY_RESERVE] <= 0.25 {
            ClimeOperatingMode::UtilityConstrained
        } else if self.channels[CO2_LOAD] >= 0.6 || self.channels[AIR_QUALITY] <= 0.5 {
            ClimeOperatingMode::AirRecovery
        } else if self.channels[THERMAL_STRESS] >= 0.55 || self.channels[COLD_STRESS] >= 0.55 {
            ClimeOperatingMode::ThermalRecovery
        } else if self.channels[NIGHT_MODE_PRESSURE] >= 0.6 && self.channels[NOISE_STRESS] >= 0.45 {
            ClimeOperatingMode::QuietNight
        } else if self.channels[CIRCADIAN_MISMATCH] >= 0.5 {
            ClimeOperatingMode::CircadianSupport
        } else {
            ClimeOperatingMode::BalancedHabitat
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClimeCommand {
    pub torques: [f32; NUM_ACTUATORS],
}

impl ClimeCommand {
    pub fn zero() -> Self {
        Self {
            torques: [0.0; NUM_ACTUATORS],
        }
    }
    pub fn control_effort(&self) -> f32 {
        self.torques.iter().map(|t| t.abs()).sum::<f32>() / NUM_ACTUATORS as f32
    }
    pub fn ventilation_fan(&self) -> f32 {
        self.torques[0]
    }
    pub fn filtration_loop(&self) -> f32 {
        self.torques[1]
    }
    pub fn cooling_loop(&self) -> f32 {
        self.torques[2]
    }
    pub fn heating_loop(&self) -> f32 {
        self.torques[3]
    }
    pub fn light_brightness(&self) -> f32 {
        self.torques[6]
    }
    pub fn light_circadian_shift(&self) -> f32 {
        self.torques[7]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimeConfig {
    pub genesis_phrase: String,
    pub learning_rate: f32,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub physics_hz: f64,
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
}

impl ClimeConfig {
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }
}

impl Default for ClimeConfig {
    fn default() -> Self {
        Self {
            genesis_phrase: "symthaea-clime".to_string(),
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
        assert!(ClimeState::home().is_finite());
    }
    #[test]
    fn test_channels() {
        assert_eq!(ClimeState::home().to_channels().len(), NUM_STATE_CHANNELS);
    }
    #[test]
    fn test_zero_cmd() {
        assert_eq!(ClimeCommand::zero().control_effort(), 0.0);
    }
}
