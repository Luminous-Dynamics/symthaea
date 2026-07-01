// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

pub const NUM_ACTUATORS: usize = 6;
pub const NUM_STATE_CHANNELS: usize = 24;
pub const ACTUATOR_LABELS: [&str; NUM_ACTUATORS] = [
    "left_drive",
    "right_drive",
    "gaze_beacon",
    "acoustic_chime",
    "thermal_beacon",
    "sanctuary_projector",
];
pub const CHANNEL_LABELS: [&str; NUM_STATE_CHANNELS] = [
    "animal_presence",
    "distress_signal",
    "path_conflict_risk",
    "thermal_stress",
    "hydration_risk",
    "air_quality_risk",
    "classification_confidence",
    "sanctuary_signal",
    "handoff_confidence",
    "route_clear_confidence",
    "comm_link_quality",
    "occupancy_pressure",
    "vehicle_threat",
    "robot_threat",
    "flock_panic_risk",
    "owner_proximity",
    "safe_zone_distance",
    "intervention_overreach",
    "response_latency",
    "species_diversity",
    "local_noise_stress",
    "blackout_resilience",
    "mission_progress",
    "welfare_integrity",
];

pub const ANIMAL_PRESENCE: usize = 0;
pub const DISTRESS_SIGNAL: usize = 1;
pub const PATH_CONFLICT_RISK: usize = 2;
pub const THERMAL_STRESS: usize = 3;
pub const CLASSIFICATION_CONFIDENCE: usize = 6;
pub const SANCTUARY_SIGNAL: usize = 7;
pub const HANDOFF_CONFIDENCE: usize = 8;
pub const ROUTE_CLEAR_CONFIDENCE: usize = 9;
pub const COMM_LINK_QUALITY: usize = 10;
pub const VEHICLE_THREAT: usize = 12;
pub const ROBOT_THREAT: usize = 13;
pub const FLOCK_PANIC_RISK: usize = 14;
pub const INTERVENTION_OVERREACH: usize = 17;
pub const RESPONSE_LATENCY: usize = 18;
pub const LOCAL_NOISE_STRESS: usize = 20;
pub const BLACKOUT_RESILIENCE: usize = 21;
pub const MISSION_PROGRESS: usize = 22;
pub const WELFARE_INTEGRITY: usize = 23;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BiotaOperatingMode {
    Observe,
    Escort,
    CrossingGuard,
    DistressResponse,
    SanctuaryHold,
    QuietMode,
    BlackoutAutonomy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiotaState {
    pub channels: [f64; NUM_STATE_CHANNELS],
}

impl BiotaState {
    pub fn home() -> Self {
        let mut channels = [0.0; NUM_STATE_CHANNELS];
        channels[CLASSIFICATION_CONFIDENCE] = 0.82;
        channels[HANDOFF_CONFIDENCE] = 0.75;
        channels[ROUTE_CLEAR_CONFIDENCE] = 0.8;
        channels[COMM_LINK_QUALITY] = 0.85;
        channels[BLACKOUT_RESILIENCE] = 0.7;
        channels[WELFARE_INTEGRITY] = 0.9;
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

    pub fn distress_signal(&self) -> f64 {
        self.channels[DISTRESS_SIGNAL]
    }
    pub fn path_conflict_risk(&self) -> f64 {
        self.channels[PATH_CONFLICT_RISK]
    }
    pub fn sanctuary_signal(&self) -> f64 {
        self.channels[SANCTUARY_SIGNAL]
    }
    pub fn route_clear_confidence(&self) -> f64 {
        self.channels[ROUTE_CLEAR_CONFIDENCE]
    }
    pub fn classification_confidence(&self) -> f64 {
        self.channels[CLASSIFICATION_CONFIDENCE]
    }
    pub fn welfare_integrity(&self) -> f64 {
        self.channels[WELFARE_INTEGRITY]
    }

    pub fn inferred_mode(&self) -> BiotaOperatingMode {
        if self.channels[COMM_LINK_QUALITY] <= 0.2 && self.channels[BLACKOUT_RESILIENCE] >= 0.45 {
            BiotaOperatingMode::BlackoutAutonomy
        } else if self.channels[INTERVENTION_OVERREACH] >= 0.55
            || self.channels[LOCAL_NOISE_STRESS] >= 0.65
        {
            BiotaOperatingMode::QuietMode
        } else if self.channels[DISTRESS_SIGNAL] >= 0.6 || self.channels[THERMAL_STRESS] >= 0.65 {
            BiotaOperatingMode::DistressResponse
        } else if self.channels[PATH_CONFLICT_RISK] >= 0.6 || self.channels[VEHICLE_THREAT] >= 0.55
        {
            BiotaOperatingMode::CrossingGuard
        } else if self.channels[SANCTUARY_SIGNAL] >= 0.6
            && self.channels[ROUTE_CLEAR_CONFIDENCE] < 0.5
        {
            BiotaOperatingMode::SanctuaryHold
        } else if self.channels[ANIMAL_PRESENCE] >= 0.45 {
            BiotaOperatingMode::Escort
        } else {
            BiotaOperatingMode::Observe
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BiotaCommand {
    pub torques: [f32; NUM_ACTUATORS],
}

impl BiotaCommand {
    pub fn zero() -> Self {
        Self {
            torques: [0.0; NUM_ACTUATORS],
        }
    }
    pub fn control_effort(&self) -> f32 {
        self.torques.iter().map(|t| t.abs()).sum::<f32>() / NUM_ACTUATORS as f32
    }
    pub fn acoustic_chime(&self) -> f32 {
        self.torques[3]
    }
    pub fn thermal_beacon(&self) -> f32 {
        self.torques[4]
    }
    pub fn sanctuary_projector(&self) -> f32 {
        self.torques[5]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiotaConfig {
    pub genesis_phrase: String,
    pub learning_rate: f32,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub physics_hz: f64,
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
}

impl BiotaConfig {
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }
}

impl Default for BiotaConfig {
    fn default() -> Self {
        Self {
            genesis_phrase: "symthaea-biota".to_string(),
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
        assert!(BiotaState::home().is_finite());
    }

    #[test]
    fn test_channels() {
        assert_eq!(BiotaState::home().to_channels().len(), NUM_STATE_CHANNELS);
    }

    #[test]
    fn test_zero_cmd() {
        assert_eq!(BiotaCommand::zero().control_effort(), 0.0);
    }
}
