// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

pub const NUM_ACTUATORS: usize = 6;
/// Learned primary actuator channels exposed by the HDC-LTC controller.
pub const NUM_PRIMARY_ACTUATORS: usize = NUM_ACTUATORS;
/// Deterministic recovery actuators owned by the safety supervisor.
pub const NUM_RECOVERY_ACTUATORS: usize = 4;
/// Total physical actuator count reported by the platform.
pub const NUM_PHYSICAL_ACTUATORS: usize = NUM_PRIMARY_ACTUATORS + NUM_RECOVERY_ACTUATORS;
pub const CUTTER_HEAD: usize = 0;
pub const AUGER_FEED: usize = 1;
pub const LEFT_TRACK: usize = 2;
pub const RIGHT_TRACK: usize = 3;
pub const BALLAST_TRIM: usize = 4;
pub const THERMAL_PUMP: usize = 5;
pub const NUM_STATE_CHANNELS: usize = 32;
pub const ACTUATOR_LABELS: [&str; NUM_ACTUATORS] = [
    "cutter_head",
    "auger_feed",
    "left_track",
    "right_track",
    "ballast_trim",
    "thermal_pump",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(usize)]
pub enum SubterraneanActuator {
    CutterHead = CUTTER_HEAD,
    AugerFeed = AUGER_FEED,
    LeftTrack = LEFT_TRACK,
    RightTrack = RIGHT_TRACK,
    BallastTrim = BALLAST_TRIM,
    ThermalPump = THERMAL_PUMP,
}

impl SubterraneanActuator {
    pub const fn index(self) -> usize {
        self as usize
    }
}

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

/// Valid physical range for each state channel, in CHANNEL_LABELS order.
pub const STATE_CHANNEL_RANGES: [(f64, f64); NUM_STATE_CHANNELS] = [
    (0.0, 200.0),
    (-2.0, 2.0),
    (-0.6, 0.6),
    (-0.5, 0.5),
    (0.0, 180.0),
    (0.0, 160.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 2.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
];

pub const DEPTH_M: usize = 0;
pub const FORWARD_VELOCITY_MPS: usize = 1;
pub const PITCH_RAD: usize = 2;
pub const ROLL_RAD: usize = 3;
pub const CUTTER_TEMP_C: usize = 4;
pub const MOTOR_TEMP_C: usize = 5;
pub const SPOIL_BUFFER_FILL: usize = 6;
pub const BATTERY_RATIO: usize = 7;
pub const COMM_SIGNAL: usize = 8;
pub const SLIP_RATIO: usize = 9;
pub const SOIL_DENSITY: usize = 10;
pub const VIBRATION_LEVEL: usize = 11;
pub const HUMIDITY: usize = 12;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateIntegrityReport {
    pub invalid_count: usize,
    pub first_invalid_channel: Option<usize>,
}

impl StateIntegrityReport {
    pub const fn is_valid(self) -> bool {
        self.invalid_count == 0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubterraneanState {
    pub channels: [f64; NUM_STATE_CHANNELS],
}

impl SubterraneanState {
    pub fn home() -> Self {
        let mut channels = [0.0; NUM_STATE_CHANNELS];
        channels[CUTTER_TEMP_C] = 20.0;
        channels[MOTOR_TEMP_C] = 20.0;
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

    pub fn integrity_report(&self) -> StateIntegrityReport {
        let mut invalid_count = 0usize;
        let mut first_invalid_channel = None;
        for (index, value) in self.channels.iter().enumerate() {
            let (minimum, maximum) = STATE_CHANNEL_RANGES[index];
            if !value.is_finite() || *value < minimum || *value > maximum {
                invalid_count += 1;
                if first_invalid_channel.is_none() {
                    first_invalid_channel = Some(index);
                }
            }
        }
        StateIntegrityReport {
            invalid_count,
            first_invalid_channel,
        }
    }

    /// Replace malformed observations with conservative, physically bounded
    /// values. Callers must assess integrity before sanitizing so the safety
    /// supervisor can latch the sensor fault.
    pub fn sanitize_fail_closed(&mut self) -> StateIntegrityReport {
        let report = self.integrity_report();
        for (index, value) in self.channels.iter_mut().enumerate() {
            let (minimum, maximum) = STATE_CHANNEL_RANGES[index];
            if value.is_finite() && *value >= minimum && *value <= maximum {
                continue;
            }
            *value = match index {
                CUTTER_TEMP_C => maximum,
                MOTOR_TEMP_C => maximum,
                SPOIL_BUFFER_FILL | WATER_INGRESS_RATIO | AQUIFER_RISK | GAS_RISK | HULL_STRESS
                | SLURRY_LOAD | ABORT_RECOMMENDATION => maximum,
                BATTERY_RATIO
                | COMM_SIGNAL
                | MAPPING_CONFIDENCE
                | THERMAL_MARGIN
                | RETURN_PATH_CONFIDENCE
                | ROOF_STABILITY
                | ESCAPE_CONFIDENCE
                | LOCALIZATION_CONFIDENCE
                | RELAY_LINK_QUALITY
                | SEAL_INTEGRITY => minimum,
                _ => ((minimum + maximum) * 0.5).clamp(minimum, maximum),
            };
        }
        report
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
        } else if self.water_ingress_ratio() >= 0.25 || self.aquifer_risk() >= 0.75 {
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RecoveryCommand {
    /// Remove ingress water and slurry from the hull.
    pub dewatering_pump: f32,
    /// Inject finite sealant reserve into a damaged pressure boundary.
    pub sealant_injector: f32,
    /// Deploy one finite communications/localization relay when commanded.
    pub relay_deployer: f32,
    /// Deploy one finite roof-support cartridge when commanded.
    pub roof_support: f32,
}

impl RecoveryCommand {
    pub const fn zero() -> Self {
        Self {
            dewatering_pump: 0.0,
            sealant_injector: 0.0,
            relay_deployer: 0.0,
            roof_support: 0.0,
        }
    }

    pub fn sanitize(&mut self) {
        for value in [
            &mut self.dewatering_pump,
            &mut self.sealant_injector,
            &mut self.relay_deployer,
            &mut self.roof_support,
        ] {
            *value = if value.is_finite() {
                value.clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
    }

    pub fn effort(&self) -> f32 {
        (self.dewatering_pump.abs()
            + self.sealant_injector.abs()
            + self.relay_deployer.abs()
            + self.roof_support.abs())
            / NUM_RECOVERY_ACTUATORS as f32
    }
}

impl Default for RecoveryCommand {
    fn default() -> Self {
        Self::zero()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SubterraneanCommand {
    pub torques: [f32; NUM_ACTUATORS],
    #[serde(default)]
    pub recovery: RecoveryCommand,
}

impl SubterraneanCommand {
    pub fn zero() -> Self {
        Self {
            torques: [0.0; NUM_ACTUATORS],
            recovery: RecoveryCommand::zero(),
        }
    }
    pub fn get(&self, actuator: SubterraneanActuator) -> f32 {
        self.torques[actuator.index()]
    }
    pub fn set(&mut self, actuator: SubterraneanActuator, value: f32) {
        self.torques[actuator.index()] = value;
    }
    pub fn control_effort(&self) -> f32 {
        let primary = self.torques.iter().map(|t| t.abs()).sum::<f32>();
        let recovery = self.recovery.dewatering_pump.abs()
            + self.recovery.sealant_injector.abs()
            + self.recovery.relay_deployer.abs()
            + self.recovery.roof_support.abs();
        (primary + recovery) / NUM_PHYSICAL_ACTUATORS as f32
    }
    pub fn cutter_head(&self) -> f32 {
        self.torques[CUTTER_HEAD]
    }
    pub fn auger_feed(&self) -> f32 {
        self.torques[AUGER_FEED]
    }
    pub fn left_track(&self) -> f32 {
        self.torques[LEFT_TRACK]
    }
    pub fn right_track(&self) -> f32 {
        self.torques[RIGHT_TRACK]
    }
    pub fn ballast_trim(&self) -> f32 {
        self.torques[BALLAST_TRIM]
    }
    pub fn thermal_pump(&self) -> f32 {
        self.torques[THERMAL_PUMP]
    }
    pub fn set_cutter_head(&mut self, value: f32) {
        self.torques[CUTTER_HEAD] = value;
    }
    pub fn set_auger_feed(&mut self, value: f32) {
        self.torques[AUGER_FEED] = value;
    }
    pub fn set_left_track(&mut self, value: f32) {
        self.torques[LEFT_TRACK] = value;
    }
    pub fn set_right_track(&mut self, value: f32) {
        self.torques[RIGHT_TRACK] = value;
    }
    pub fn set_ballast_trim(&mut self, value: f32) {
        self.torques[BALLAST_TRIM] = value;
    }
    pub fn set_thermal_pump(&mut self, value: f32) {
        self.torques[THERMAL_PUMP] = value;
    }
    pub fn sanitize(&mut self) {
        for value in &mut self.torques {
            *value = if value.is_finite() {
                value.clamp(-1.0, 1.0)
            } else {
                0.0
            };
        }
        self.recovery.sanitize();
    }
    pub fn limit_magnitude(&mut self, limit: f32) {
        let limit = limit.abs().min(1.0);
        for value in &mut self.torques {
            *value = value.clamp(-limit, limit);
        }
    }

    /// Blend learner and teacher commands for deterministic DAgger-style
    /// rollouts. `teacher_ratio=1` executes the teacher, `0` the learner.
    pub fn blend(learner: Self, teacher: Self, teacher_ratio: f32) -> Self {
        let teacher_ratio = teacher_ratio.clamp(0.0, 1.0);
        let learner_ratio = 1.0 - teacher_ratio;
        let mut command = Self::zero();
        for index in 0..NUM_ACTUATORS {
            command.torques[index] =
                learner.torques[index] * learner_ratio + teacher.torques[index] * teacher_ratio;
        }
        command.recovery.dewatering_pump = learner.recovery.dewatering_pump * learner_ratio
            + teacher.recovery.dewatering_pump * teacher_ratio;
        command.recovery.sealant_injector = learner.recovery.sealant_injector * learner_ratio
            + teacher.recovery.sealant_injector * teacher_ratio;
        command.recovery.relay_deployer = learner.recovery.relay_deployer * learner_ratio
            + teacher.recovery.relay_deployer * teacher_ratio;
        command.recovery.roof_support = learner.recovery.roof_support * learner_ratio
            + teacher.recovery.roof_support * teacher_ratio;
        command.sanitize();
        command
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    EmptyGenesisPhrase,
    InvalidLearningRate,
    InvalidNetworkLayers,
    InvalidNeuronsPerLayer,
    InvalidPhysicsRate,
    ZeroCognitiveInterval,
    ZeroEpisodeLength,
    InvalidEvidenceCapacity,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::EmptyGenesisPhrase => "genesis_phrase must not be empty",
            Self::InvalidLearningRate => "learning_rate must be finite and in (0, 1]",
            Self::InvalidNetworkLayers => "network_layers must be in 1..=16",
            Self::InvalidNeuronsPerLayer => "neurons_per_layer must be in 1..=4096",
            Self::InvalidPhysicsRate => "physics_hz must be finite and in 1..=10_000",
            Self::ZeroCognitiveInterval => "cognitive_interval must be greater than zero",
            Self::ZeroEpisodeLength => "steps_per_episode must be greater than zero",
            Self::InvalidEvidenceCapacity => "evidence_capacity must be in 1..=1_000_000",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ConfigError {}

const fn default_evidence_capacity() -> usize {
    4096
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
    #[serde(default = "default_evidence_capacity")]
    pub evidence_capacity: usize,
}

impl SubterraneanConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.genesis_phrase.trim().is_empty() {
            return Err(ConfigError::EmptyGenesisPhrase);
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 || self.learning_rate > 1.0
        {
            return Err(ConfigError::InvalidLearningRate);
        }
        if !(1..=16).contains(&self.network_layers) {
            return Err(ConfigError::InvalidNetworkLayers);
        }
        if !(1..=4096).contains(&self.neurons_per_layer) {
            return Err(ConfigError::InvalidNeuronsPerLayer);
        }
        if !self.physics_hz.is_finite() || !(1.0..=10_000.0).contains(&self.physics_hz) {
            return Err(ConfigError::InvalidPhysicsRate);
        }
        if self.cognitive_interval == 0 {
            return Err(ConfigError::ZeroCognitiveInterval);
        }
        if self.steps_per_episode == 0 {
            return Err(ConfigError::ZeroEpisodeLength);
        }
        if !(1..=1_000_000).contains(&self.evidence_capacity) {
            return Err(ConfigError::InvalidEvidenceCapacity);
        }
        Ok(())
    }

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
            evidence_capacity: default_evidence_capacity(),
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
    #[test]
    fn invalid_config_is_rejected() {
        let mut config = SubterraneanConfig::default();
        config.cognitive_interval = 0;
        assert_eq!(config.validate(), Err(ConfigError::ZeroCognitiveInterval));

        config = SubterraneanConfig::default();
        config.physics_hz = f64::NAN;
        assert_eq!(config.validate(), Err(ConfigError::InvalidPhysicsRate));
    }

    #[test]
    fn typed_actuator_access_matches_named_accessors() {
        let mut command = SubterraneanCommand::zero();
        command.set(SubterraneanActuator::ThermalPump, 0.75);
        assert_eq!(command.get(SubterraneanActuator::ThermalPump), 0.75);
        assert_eq!(command.thermal_pump(), 0.75);
    }

    #[test]
    fn malformed_state_is_detected_and_sanitized_conservatively() {
        let mut state = SubterraneanState::home();
        state.channels[GAS_RISK] = f64::NAN;
        state.channels[BATTERY_RATIO] = 2.0;
        let report = state.sanitize_fail_closed();
        assert_eq!(report.invalid_count, 2);
        assert_eq!(state.channels[GAS_RISK], 1.0);
        assert_eq!(state.channels[BATTERY_RATIO], 0.0);
        assert!(state.integrity_report().is_valid());
    }
}
