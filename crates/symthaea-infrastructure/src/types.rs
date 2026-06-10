// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

pub const NUM_ACTUATORS: usize = 8;
pub const NUM_STATE_CHANNELS: usize = 28;
pub const ACTUATOR_LABELS: [&str; NUM_ACTUATORS] = [
    "charge_bus",
    "discharge_bus",
    "cooling_loop",
    "heating_loop",
    "routing_north",
    "routing_south",
    "routing_east",
    "routing_west",
];

/// Cross-platform habitat demand from civic atmosphere / building homeostasis systems.
///
/// This allows Clime to request utility conservation or emergency isolation
/// without embedding civic channels directly into the infrastructure encoder.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ClimeInfrastructureSignal {
    /// Smoke or fire-adjacent contamination urgency (0..1).
    pub smoke_risk: f32,
    /// Airborne contamination or filtration breach urgency (0..1).
    pub contamination_risk: f32,
    /// Public-health severity for the current habitat zone (0..1).
    pub public_health_risk: f32,
    /// Remaining habitat utility reserve (0..1).
    pub utility_reserve: f32,
    /// Confidence that the affected zone can be isolated safely (0..1).
    pub zone_isolation_confidence: f32,
    /// Confidence the upstream habitat assessment is valid (0..1).
    pub sensor_confidence: f32,
}

impl ClimeInfrastructureSignal {
    pub fn clear() -> Self {
        Self {
            smoke_risk: 0.0,
            contamination_risk: 0.0,
            public_health_risk: 0.0,
            utility_reserve: 1.0,
            zone_isolation_confidence: 1.0,
            sensor_confidence: 1.0,
        }
    }

    pub fn should_emergency_red(&self) -> bool {
        self.sensor_confidence >= 0.45
            && ((self.smoke_risk >= 0.7 && self.zone_isolation_confidence >= 0.55)
                || self.contamination_risk >= 0.8
                || self.public_health_risk >= 0.85)
    }

    pub fn should_isolate_orange(&self) -> bool {
        self.sensor_confidence >= 0.35
            && (self.smoke_risk >= 0.45
                || self.contamination_risk >= 0.55
                || self.public_health_risk >= 0.6
                || (self.utility_reserve <= 0.22 && self.zone_isolation_confidence >= 0.4))
    }

    pub fn should_conserve_yellow(&self) -> bool {
        self.sensor_confidence >= 0.25
            && (self.utility_reserve <= 0.35
                || self.public_health_risk >= 0.35
                || self.contamination_risk >= 0.25)
    }
}
pub const CHANNEL_LABELS: [&str; NUM_STATE_CHANNELS] = [
    "storage_ratio",
    "thermal_load",
    "inbound_power_kw",
    "outbound_power_kw",
    "queue_depth",
    "relay_health",
    "voltage_stability",
    "coolant_temp_c",
    "heating_demand",
    "uptime_ratio",
    "north_flow",
    "south_flow",
    "east_flow",
    "west_flow",
    "maintenance_backlog",
    "islanding_capability",
    "switchgear_wear",
    "grid_stress",
    "community_demand",
    "brownout_risk",
    "shed_load_ratio",
    "critical_load_fraction",
    "unserved_demand_ratio",
    "deadlock_risk",
    "islanding_risk",
    "service_integrity",
    "recovery_margin",
    "thermal_runaway_risk",
];

pub const STORAGE_RATIO: usize = 0;
pub const THERMAL_LOAD: usize = 1;
pub const QUEUE_DEPTH: usize = 4;
pub const RELAY_HEALTH: usize = 5;
pub const VOLTAGE_STABILITY: usize = 6;
pub const COOLANT_TEMP_C: usize = 7;
pub const GRID_STRESS: usize = 17;
pub const COMMUNITY_DEMAND: usize = 18;
pub const BROWNOUT_RISK: usize = 19;
pub const SHED_LOAD_RATIO: usize = 20;
pub const CRITICAL_LOAD_FRACTION: usize = 21;
pub const UNSERVED_DEMAND_RATIO: usize = 22;
pub const DEADLOCK_RISK: usize = 23;
pub const ISLANDING_RISK: usize = 24;
pub const SERVICE_INTEGRITY: usize = 25;
pub const RECOVERY_MARGIN: usize = 26;
pub const THERMAL_RUNAWAY_RISK: usize = 27;
pub const ISLANDING_CAPABILITY: usize = 15;
pub const SWITCHGEAR_WEAR: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InfrastructureOperatingMode {
    Balanced,
    LoadShedding,
    CoolingRecovery,
    Islanding,
    DeadlockRecovery,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfrastructureState {
    pub channels: [f64; NUM_STATE_CHANNELS],
}

impl InfrastructureState {
    pub fn home() -> Self {
        let mut channels = [0.0; NUM_STATE_CHANNELS];
        channels[STORAGE_RATIO] = 0.75;
        channels[RELAY_HEALTH] = 1.0;
        channels[VOLTAGE_STABILITY] = 0.95;
        channels[COOLANT_TEMP_C] = 22.0;
        channels[9] = 1.0;
        channels[ISLANDING_CAPABILITY] = 0.8;
        channels[SERVICE_INTEGRITY] = 0.95;
        channels[RECOVERY_MARGIN] = 0.8;
        channels[CRITICAL_LOAD_FRACTION] = 0.35;
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

    pub fn storage_ratio(&self) -> f64 {
        self.channels[STORAGE_RATIO]
    }
    pub fn thermal_load(&self) -> f64 {
        self.channels[THERMAL_LOAD]
    }
    pub fn voltage_stability(&self) -> f64 {
        self.channels[VOLTAGE_STABILITY]
    }
    pub fn relay_health(&self) -> f64 {
        self.channels[RELAY_HEALTH]
    }
    pub fn brownout_risk(&self) -> f64 {
        self.channels[BROWNOUT_RISK]
    }
    pub fn shed_load_ratio(&self) -> f64 {
        self.channels[SHED_LOAD_RATIO]
    }
    pub fn unserved_demand_ratio(&self) -> f64 {
        self.channels[UNSERVED_DEMAND_RATIO]
    }
    pub fn deadlock_risk(&self) -> f64 {
        self.channels[DEADLOCK_RISK]
    }
    pub fn islanding_risk(&self) -> f64 {
        self.channels[ISLANDING_RISK]
    }
    pub fn service_integrity(&self) -> f64 {
        self.channels[SERVICE_INTEGRITY]
    }
    pub fn recovery_margin(&self) -> f64 {
        self.channels[RECOVERY_MARGIN]
    }
    pub fn thermal_runaway_risk(&self) -> f64 {
        self.channels[THERMAL_RUNAWAY_RISK]
    }

    pub fn inferred_mode(&self) -> InfrastructureOperatingMode {
        if self.thermal_runaway_risk() >= 0.8
            || self.brownout_risk() >= 0.85
            || self.service_integrity() <= 0.25
        {
            InfrastructureOperatingMode::Emergency
        } else if self.thermal_runaway_risk() >= 0.55 {
            InfrastructureOperatingMode::CoolingRecovery
        } else if self.deadlock_risk() >= 0.6 {
            InfrastructureOperatingMode::DeadlockRecovery
        } else if self.islanding_risk() >= 0.65 {
            InfrastructureOperatingMode::Islanding
        } else if self.shed_load_ratio() >= 0.2 || self.brownout_risk() >= 0.5 {
            InfrastructureOperatingMode::LoadShedding
        } else {
            InfrastructureOperatingMode::Balanced
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InfrastructureCommand {
    pub torques: [f32; NUM_ACTUATORS],
}

impl InfrastructureCommand {
    pub fn zero() -> Self {
        Self {
            torques: [0.0; NUM_ACTUATORS],
        }
    }
    pub fn control_effort(&self) -> f32 {
        self.torques.iter().map(|t| t.abs()).sum::<f32>() / NUM_ACTUATORS as f32
    }
    pub fn charge_bus(&self) -> f32 {
        self.torques[0]
    }
    pub fn discharge_bus(&self) -> f32 {
        self.torques[1]
    }
    pub fn cooling_loop(&self) -> f32 {
        self.torques[2]
    }
    pub fn heating_loop(&self) -> f32 {
        self.torques[3]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfrastructureConfig {
    pub genesis_phrase: String,
    pub learning_rate: f32,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub physics_hz: f64,
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
}

impl InfrastructureConfig {
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }
}

impl Default for InfrastructureConfig {
    fn default() -> Self {
        Self {
            genesis_phrase: "symthaea-infrastructure".to_string(),
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
        assert!(InfrastructureState::home().is_finite());
    }
    #[test]
    fn test_channels() {
        assert_eq!(
            InfrastructureState::home().to_channels().len(),
            NUM_STATE_CHANNELS
        );
    }
    #[test]
    fn test_zero_cmd() {
        assert_eq!(InfrastructureCommand::zero().control_effort(), 0.0);
    }
    #[test]
    fn test_clime_signal_thresholds() {
        let red = ClimeInfrastructureSignal {
            smoke_risk: 0.8,
            contamination_risk: 0.2,
            public_health_risk: 0.3,
            utility_reserve: 0.7,
            zone_isolation_confidence: 0.8,
            sensor_confidence: 0.8,
        };
        assert!(red.should_emergency_red());

        let orange = ClimeInfrastructureSignal {
            smoke_risk: 0.4,
            contamination_risk: 0.6,
            public_health_risk: 0.4,
            utility_reserve: 0.2,
            zone_isolation_confidence: 0.6,
            sensor_confidence: 0.7,
        };
        assert!(orange.should_isolate_orange());

        let yellow = ClimeInfrastructureSignal {
            smoke_risk: 0.1,
            contamination_risk: 0.3,
            public_health_risk: 0.2,
            utility_reserve: 0.3,
            zone_isolation_confidence: 0.2,
            sensor_confidence: 0.5,
        };
        assert!(yellow.should_conserve_yellow());
    }
}
