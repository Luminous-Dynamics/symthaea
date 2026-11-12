//! Provenance-bound visual + proprioceptive input for driving experiments.
//!
//! This bridge composes the existing Vision Manifold and vehicle HDC encoder.
//! It does not execute a policy or actuator and does not assert that an old
//! proprioceptive-only checkpoint is valid for the fused representation.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};
use symthaea_vehicle::{VehicleHdcEncoder, VehicleState};
use symthaea_vision_manifold::{VisionConfig, VisionManifold, VisionTelemetry};

pub const RECEIPT_SCHEMA_VERSION: u32 = 1;
pub const MAX_ID_BYTES: usize = 256;
pub const MAX_FRAME_BYTES_HARD_LIMIT: usize = 64 * 1024 * 1024;
pub const MAX_FRAME_DIMENSION: u32 = 4_096;
pub const MAX_VEHICLE_ENCODER_LEVELS: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DrivingRunIdentity {
    pub run_id: String,
    pub episode_id: String,
    pub scenario_id: String,
    pub agent_id: String,
    pub camera_id: String,
    pub simulator_revision: String,
    pub policy_revision: String,
}

impl DrivingRunIdentity {
    fn validate(&self) -> Result<(), BridgeError> {
        for (name, value) in [
            ("run_id", &self.run_id),
            ("episode_id", &self.episode_id),
            ("scenario_id", &self.scenario_id),
            ("agent_id", &self.agent_id),
            ("camera_id", &self.camera_id),
            ("simulator_revision", &self.simulator_revision),
            ("policy_revision", &self.policy_revision),
        ] {
            if value.trim().is_empty() {
                return Err(BridgeError::MissingIdentity(name));
            }
            if value.len() > MAX_ID_BYTES {
                return Err(BridgeError::IdentityTooLong(name));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgeConfig {
    pub identity: DrivingRunIdentity,
    pub max_width: u32,
    pub max_height: u32,
    pub max_frame_bytes: usize,
    pub initial_dt_us: u64,
    pub maximum_frame_gap_us: u64,
    /// Maximum age from camera capture to bridge delivery, in microseconds.
    /// Zero is invalid; stale observations are rejected before temporal state advances.
    pub maximum_observation_delivery_age_us: u64,
    pub maximum_sensor_clock_skew_us: u64,
    /// Weight of the visual manifold state in parts per million.
    /// The proprioceptive weight is 1,000,000 minus this value.
    pub visual_weight_ppm: u32,
    pub vehicle_encoder_seed_phrase: String,
    /// Optional exact camera calibration pin. When set, observations with a
    /// different calibration digest are rejected before temporal state advances.
    #[serde(default)]
    pub expected_camera_calibration_sha256: Option<String>,
}

impl BridgeConfig {
    pub fn validate(&self) -> Result<(), BridgeError> {
        self.identity.validate()?;
        if self.max_width == 0
            || self.max_height == 0
            || self.max_width > MAX_FRAME_DIMENSION
            || self.max_height > MAX_FRAME_DIMENSION
        {
            return Err(BridgeError::InvalidConfig("maximum frame dimensions must be in 1..=4096"));
        }
        if self.max_frame_bytes == 0 || self.max_frame_bytes > MAX_FRAME_BYTES_HARD_LIMIT {
            return Err(BridgeError::InvalidConfig("frame-byte bound must be in 1..=64 MiB"));
        }
        if self.initial_dt_us == 0 || self.maximum_frame_gap_us < self.initial_dt_us {
            return Err(BridgeError::InvalidConfig("invalid initial timestep or maximum frame gap"));
        }
        if self.maximum_observation_delivery_age_us == 0 {
            return Err(BridgeError::InvalidConfig("maximum observation delivery age must be non-zero"));
        }
        if self.visual_weight_ppm > 1_000_000 {
            return Err(BridgeError::InvalidConfig("visual weight must be <= 1,000,000 ppm"));
        }
        if self.expected_camera_calibration_sha256.as_deref().is_some_and(|digest| !is_sha256_hex(digest)) {
            return Err(BridgeError::InvalidConfig("expected camera calibration digest must be lowercase SHA-256"));
        }
        if self.vehicle_encoder_seed_phrase.trim().is_empty()
            || self.vehicle_encoder_seed_phrase.len() > 1_024
        {
            return Err(BridgeError::InvalidConfig("invalid vehicle encoder seed phrase"));
        }
        Ok(())
    }
}

/// Policy-visible metadata copied from the upstream Symtropy camera receipt.
/// Digest fields are integrity pointers; this type does not authenticate the issuer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpstreamCameraObservation {
    pub schema_version: u32,
    pub observation_id: String,
    pub pixel_format: String,
    pub frame_byte_len: usize,
    pub receipt_sha256: String,
    pub frame_sha256: String,
    pub episode_id: String,
    pub agent_id: String,
    pub clock_id: String,
    pub tick: u64,
    pub sequence: u64,
    pub camera_id: String,
    pub camera_calibration_sha256: String,
    pub renderer_revision: String,
    pub capture_timestamp_us: u64,
    pub render_completed_timestamp_us: u64,
    pub delivered_timestamp_us: u64,
    #[serde(rename = "frame_width")]
    pub width: u32,
    #[serde(rename = "frame_height")]
    pub height: u32,
    pub channels: usize,
    pub applied_action_receipt_id: Option<String>,
    pub applied_action_receipt_sha256: Option<String>,
    pub applied_action_command_id: Option<String>,
    pub applied_action_tick: Option<u64>,
}

impl UpstreamCameraObservation {
    /// Recompute the renderer's observation identity from policy-visible fields.
    /// This verifies internal consistency, not publisher authenticity.
    pub fn calculated_observation_id(&self) -> String {
        let mut hasher = Sha256::new();
        for value in [&self.episode_id, &self.agent_id, &self.clock_id] {
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        hasher.update(self.tick.to_le_bytes());
        hasher.update(self.sequence.to_le_bytes());
        for value in [
            self.applied_action_receipt_id.as_deref().unwrap_or(""),
            self.applied_action_receipt_sha256.as_deref().unwrap_or(""),
            self.frame_sha256.as_str(),
        ] {
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        digest_to_hex(&hasher.finalize())
    }

    /// Recompute the renderer's policy-receipt commitment. Pixel format is
    /// currently pinned to RGB8 sRGB (enum discriminant zero).
    pub fn calculated_receipt_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.to_le_bytes());
        let observation_id = self.observation_id.as_str();
        hasher.update((observation_id.len() as u64).to_le_bytes());
        hasher.update(observation_id.as_bytes());
        hasher.update([0_u8]);
        for value in [&self.episode_id, &self.agent_id, &self.clock_id] {
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        hasher.update(self.tick.to_le_bytes());
        hasher.update(self.sequence.to_le_bytes());
        for value in [
            self.applied_action_receipt_id.as_deref().unwrap_or(""),
            self.applied_action_receipt_sha256.as_deref().unwrap_or(""),
            self.applied_action_command_id.as_deref().unwrap_or(""),
            self.camera_id.as_str(),
            self.camera_calibration_sha256.as_str(),
            self.renderer_revision.as_str(),
            self.frame_sha256.as_str(),
        ] {
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        hasher.update([u8::from(self.applied_action_tick.is_some())]);
        hasher.update(self.applied_action_tick.unwrap_or(0).to_le_bytes());
        for value in [
            self.capture_timestamp_us,
            self.render_completed_timestamp_us,
            self.delivered_timestamp_us,
            u64::from(self.width),
            u64::from(self.height),
            self.channels as u64,
            self.frame_byte_len as u64,
        ] {
            hasher.update(value.to_le_bytes());
        }
        digest_to_hex(&hasher.finalize())
    }
}

#[derive(Debug, Clone)]
pub struct CapturedDrivingObservation<'a> {
    pub sequence: u64,
    pub timestamp_us: u64,
    pub vehicle_state_timestamp_us: u64,
    pub pixels: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub vehicle_state: &'a VehicleState,
    pub upstream: UpstreamCameraObservation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeError {
    InvalidConfig(&'static str),
    MissingIdentity(&'static str),
    IdentityTooLong(&'static str),
    HdcDimensionMismatch,
    DepthInputRequired,
    UnsupportedChannels,
    InvalidGeometry,
    FrameGeometryOverflow,
    FrameExceedsConfiguredCapacity,
    FrameExceedsByteBound,
    FrameLengthMismatch,
    SequenceRegression,
    VehicleTimestampRegression,
    TimestampRegression,
    FrameGapTooLarge,
    SensorClockSkewExceeded,
    NonFiniteVehicleState,
    ManifoldRejected,
    NonFinitePerceptionOutput,
    LearningNotFrozen,
    UpstreamReceiptMalformed,
    UpstreamReceiptCommitmentMismatch,
    UpstreamIdentityMismatch,
    UpstreamCalibrationMismatch,
    UpstreamFrameMismatch,
    UpstreamSequenceMismatch,
    UpstreamTimestampMismatch,
    UpstreamGeometryMismatch,
    UpstreamClockOrderInvalid,
    UpstreamDeliveryAgeExceeded,
    UpstreamActionReferenceInvalid,
    BridgeFaulted,
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for BridgeError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisionTelemetryReceipt {
    pub prediction_error: f32,
    pub manifold_coherence: f32,
    pub attention_entropy: f32,
    pub motion_surprise: f32,
    pub num_salient_patches: usize,
    pub frame_sequence: u64,
    pub training_triggered: bool,
}

impl TryFrom<&VisionTelemetry> for VisionTelemetryReceipt {
    type Error = BridgeError;

    fn try_from(value: &VisionTelemetry) -> Result<Self, Self::Error> {
        let finite = [
            value.prediction_error,
            value.manifold_coherence,
            value.attention_entropy,
            value.motion_surprise,
        ];
        if !finite.iter().all(|metric| metric.is_finite()) {
            return Err(BridgeError::NonFinitePerceptionOutput);
        }
        Ok(Self {
            prediction_error: value.prediction_error,
            manifold_coherence: value.manifold_coherence,
            attention_entropy: value.attention_entropy,
            motion_surprise: value.motion_surprise,
            num_salient_patches: value.num_salient_patches,
            frame_sequence: value.frame_sequence,
            training_triggered: value.training_triggered,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrivingObservationReceipt {
    pub schema_version: u32,
    pub identity: DrivingRunIdentity,
    pub capture_sequence: u64,
    pub capture_timestamp_us: u64,
    pub vehicle_state_timestamp_us: u64,
    pub derived_dt_us: u64,
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub visual_weight_ppm: u32,
    pub vision_config_sha256: String,
    pub vehicle_encoder_seed_sha256: String,
    pub hdc_dimension: usize,
    pub vehicle_encoder_levels: usize,
    pub upstream_observation_id: String,
    pub upstream_receipt_sha256: String,
    pub upstream_frame_sha256: String,
    pub upstream_clock_id: String,
    pub upstream_tick: u64,
    pub upstream_agent_id: String,
    pub upstream_camera_calibration_sha256: String,
    pub upstream_renderer_revision: String,
    pub upstream_capture_timestamp_us: u64,
    pub upstream_render_completed_timestamp_us: u64,
    pub upstream_delivered_timestamp_us: u64,
    pub upstream_applied_action_receipt_id: Option<String>,
    pub upstream_applied_action_receipt_sha256: Option<String>,
    pub upstream_applied_action_command_id: Option<String>,
    pub upstream_applied_action_tick: Option<u64>,
    pub previous_receipt_sha256: Option<String>,
    pub raw_frame_sha256: String,
    pub visual_state_sha256: String,
    pub vehicle_state_sha256: String,
    pub previous_vehicle_state_sha256: Option<String>,
    pub previous_visual_state_sha256: Option<String>,
    pub proprioceptive_hv_sha256: String,
    pub fused_policy_input_sha256: String,
    pub vision_telemetry: VisionTelemetryReceipt,
    pub vision_learning_frozen: bool,
    pub vision_training_steps_before: u64,
    pub vision_training_steps_after: u64,
    /// SHA-256 commitment to every preceding receipt field.
    pub receipt_sha256: String,
}

pub struct FusedDrivingInput {
    /// The 16,384D vector that a compatible, separately trained policy may consume.
    pub policy_input: ContinuousHV,
    pub receipt: DrivingObservationReceipt,
}

pub struct CameraDrivingBridge {
    config: BridgeConfig,
    vision: VisionManifold,
    vehicle_encoder: VehicleHdcEncoder,
    vision_config_sha256: String,
    vehicle_encoder_seed_sha256: String,
    last_sequence: Option<u64>,
    last_timestamp_us: Option<u64>,
    last_vehicle_timestamp_us: Option<u64>,
    last_vehicle_state_sha256: Option<String>,
    last_visual_state_sha256: Option<String>,
    last_receipt_sha256: Option<String>,
    last_upstream_clock_id: Option<String>,
    last_upstream_tick: Option<u64>,
    faulted: bool,
}

impl CameraDrivingBridge {
    pub fn try_new(config: BridgeConfig, vision_config: VisionConfig) -> Result<Self, BridgeError> {
        config.validate()?;
        if vision_config.hdc_dim != HDC_DIMENSION {
            return Err(BridgeError::HdcDimensionMismatch);
        }
        if vision_config.enable_depth {
            return Err(BridgeError::DepthInputRequired);
        }
        vision_config.validate().map_err(|_| BridgeError::InvalidConfig("invalid VisionConfig"))?;
        let vision_config_bytes = serde_json::to_vec(&vision_config)
            .map_err(|_| BridgeError::InvalidConfig("VisionConfig serialization failed"))?;
        let vision_config_sha256 = digest_bytes(&vision_config_bytes);
        let vehicle_encoder_seed_sha256 = digest_bytes(config.vehicle_encoder_seed_phrase.as_bytes());
        let mut vision = VisionManifold::try_new(
            vision_config,
            config.max_width,
            config.max_height,
        )
        .map_err(|_| BridgeError::InvalidConfig("VisionManifold construction rejected"))?;
        vision.freeze_learning(true);
        let seed = GenesisSeed::from_phrase(&config.vehicle_encoder_seed_phrase);
        let vehicle_encoder = VehicleHdcEncoder::new(&seed, MAX_VEHICLE_ENCODER_LEVELS);
        Ok(Self {
            config,
            vision,
            vehicle_encoder,
            vision_config_sha256,
            vehicle_encoder_seed_sha256,
            last_sequence: None,
            last_timestamp_us: None,
            last_vehicle_timestamp_us: None,
            last_vehicle_state_sha256: None,
            last_visual_state_sha256: None,
            last_receipt_sha256: None,
            last_upstream_clock_id: None,
            last_upstream_tick: None,
            faulted: false,
        })
    }

    /// Validate all caller-controlled fields before advancing either temporal encoder.
    pub fn observe(
        &mut self,
        observation: CapturedDrivingObservation<'_>,
    ) -> Result<FusedDrivingInput, BridgeError> {
        if self.faulted {
            return Err(BridgeError::BridgeFaulted);
        }
        let dt_us = self.validate_observation(&observation)?;
        let state_channels = observation.vehicle_state.to_channels();
        if !observation.vehicle_state.timestamp.is_finite()
            || !state_channels.iter().all(|channel| channel.is_finite())
        {
            return Err(BridgeError::NonFiniteVehicleState);
        }

        let previous_vehicle_state_sha256 = self.last_vehicle_state_sha256.clone();
        let previous_visual_state_sha256 = self.last_visual_state_sha256.clone();
        let previous_receipt_sha256 = self.last_receipt_sha256.clone();
        let vehicle_state_sha256 = digest_vehicle_state(observation.vehicle_state, &state_channels);
        let training_steps_before = self.vision.training_steps();
        let dt = (dt_us as f64 / 1_000_000.0) as f32;
        let telemetry = match self.vision.observe_frame_checked(
            observation.pixels,
            observation.width,
            observation.height,
            observation.channels,
            dt,
        ) {
            Ok(telemetry) => telemetry,
            Err(_) => return Err(BridgeError::ManifoldRejected),
        };

        if !self.vision.is_learning_frozen() {
            self.faulted = true;
            return Err(BridgeError::LearningNotFrozen);
        }
        let training_steps_after = self.vision.training_steps();
        if training_steps_before != training_steps_after || telemetry.training_triggered {
            self.faulted = true;
            return Err(BridgeError::LearningNotFrozen);
        }

        let visual_state = self.vision.state().clone().normalize();
        let proprioceptive_hv = self.vehicle_encoder.encode(observation.vehicle_state).normalize();
        let visual_weight = self.config.visual_weight_ppm as f32 / 1_000_000.0;
        let proprioceptive_weight = 1.0 - visual_weight;
        let policy_input = if self.config.visual_weight_ppm == 0 {
            proprioceptive_hv.clone()
        } else if self.config.visual_weight_ppm == 1_000_000 {
            visual_state.clone()
        } else {
            ContinuousHV::weighted_bundle(
                &[&visual_state, &proprioceptive_hv],
                &[visual_weight, proprioceptive_weight],
            )
            .normalize()
        };

        if !visual_state.as_slice().iter().all(|value| value.is_finite())
            || !proprioceptive_hv.as_slice().iter().all(|value| value.is_finite())
            || !policy_input.as_slice().iter().all(|value| value.is_finite())
        {
            self.faulted = true;
            return Err(BridgeError::NonFinitePerceptionOutput);
        }

        let vision_telemetry = match VisionTelemetryReceipt::try_from(&telemetry) {
            Ok(receipt) => receipt,
            Err(error) => {
                self.faulted = true;
                return Err(error);
            }
        };
        let raw_frame_sha256 = digest_bytes(observation.pixels);
        let visual_state_sha256 = digest_hv(&visual_state);
        let proprioceptive_hv_sha256 = digest_hv(&proprioceptive_hv);
        let fused_policy_input_sha256 = digest_hv(&policy_input);

        let mut receipt = DrivingObservationReceipt {
            schema_version: RECEIPT_SCHEMA_VERSION,
            identity: self.config.identity.clone(),
            capture_sequence: observation.sequence,
            capture_timestamp_us: observation.timestamp_us,
            vehicle_state_timestamp_us: observation.vehicle_state_timestamp_us,
            derived_dt_us: dt_us,
            width: observation.width,
            height: observation.height,
            channels: observation.channels,
            visual_weight_ppm: self.config.visual_weight_ppm,
            vision_config_sha256: self.vision_config_sha256.clone(),
            vehicle_encoder_seed_sha256: self.vehicle_encoder_seed_sha256.clone(),
            hdc_dimension: HDC_DIMENSION,
            vehicle_encoder_levels: MAX_VEHICLE_ENCODER_LEVELS,
            upstream_observation_id: observation.upstream.observation_id.clone(),
            upstream_receipt_sha256: observation.upstream.receipt_sha256.clone(),
            upstream_frame_sha256: observation.upstream.frame_sha256.clone(),
            upstream_clock_id: observation.upstream.clock_id.clone(),
            upstream_tick: observation.upstream.tick,
            upstream_agent_id: observation.upstream.agent_id.clone(),
            upstream_camera_calibration_sha256: observation.upstream.camera_calibration_sha256.clone(),
            upstream_renderer_revision: observation.upstream.renderer_revision.clone(),
            upstream_capture_timestamp_us: observation.upstream.capture_timestamp_us,
            upstream_render_completed_timestamp_us: observation.upstream.render_completed_timestamp_us,
            upstream_delivered_timestamp_us: observation.upstream.delivered_timestamp_us,
            upstream_applied_action_receipt_id: observation.upstream.applied_action_receipt_id.clone(),
            upstream_applied_action_receipt_sha256: observation.upstream.applied_action_receipt_sha256.clone(),
            upstream_applied_action_command_id: observation.upstream.applied_action_command_id.clone(),
            upstream_applied_action_tick: observation.upstream.applied_action_tick,
            previous_receipt_sha256,
            raw_frame_sha256,
            visual_state_sha256: visual_state_sha256.clone(),
            vehicle_state_sha256: vehicle_state_sha256.clone(),
            previous_vehicle_state_sha256,
            previous_visual_state_sha256,
            proprioceptive_hv_sha256,
            fused_policy_input_sha256,
            vision_telemetry,
            vision_learning_frozen: self.vision.is_learning_frozen(),
            vision_training_steps_before: training_steps_before,
            vision_training_steps_after: training_steps_after,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = digest_receipt(&receipt);

        self.last_sequence = Some(observation.sequence);
        self.last_timestamp_us = Some(observation.timestamp_us);
        self.last_vehicle_timestamp_us = Some(observation.vehicle_state_timestamp_us);
        self.last_vehicle_state_sha256 = Some(vehicle_state_sha256);
        self.last_visual_state_sha256 = Some(visual_state_sha256);
        self.last_receipt_sha256 = Some(receipt.receipt_sha256.clone());
        self.last_upstream_clock_id = Some(observation.upstream.clock_id.clone());
        self.last_upstream_tick = Some(observation.upstream.tick);

        Ok(FusedDrivingInput {
            policy_input,
            receipt,
        })
    }

    fn validate_observation(
        &self,
        observation: &CapturedDrivingObservation<'_>,
    ) -> Result<u64, BridgeError> {
        if observation.width == 0 || observation.height == 0 {
            return Err(BridgeError::InvalidGeometry);
        }
        if observation.width > self.config.max_width || observation.height > self.config.max_height {
            return Err(BridgeError::FrameExceedsConfiguredCapacity);
        }
        if !matches!(observation.channels, 1 | 3 | 4) {
            return Err(BridgeError::UnsupportedChannels);
        }
        let expected_len = (observation.width as usize)
            .checked_mul(observation.height as usize)
            .and_then(|pixels| pixels.checked_mul(observation.channels))
            .ok_or(BridgeError::FrameGeometryOverflow)?;
        if expected_len > self.config.max_frame_bytes {
            return Err(BridgeError::FrameExceedsByteBound);
        }
        if expected_len > MAX_FRAME_BYTES_HARD_LIMIT {
            return Err(BridgeError::FrameExceedsByteBound);
        }
        if observation.pixels.len() != expected_len {
            return Err(BridgeError::FrameLengthMismatch);
        }
        self.validate_upstream_observation(observation)?;
        if self.last_sequence.is_some_and(|last| observation.sequence <= last) {
            return Err(BridgeError::SequenceRegression);
        }
        let dt_us = match self.last_timestamp_us {
            Some(previous) => observation
                .timestamp_us
                .checked_sub(previous)
                .filter(|delta| *delta > 0)
                .ok_or(BridgeError::TimestampRegression)?,
            None => self.config.initial_dt_us,
        };
        if dt_us > self.config.maximum_frame_gap_us {
            return Err(BridgeError::FrameGapTooLarge);
        }
        if self
            .last_vehicle_timestamp_us
            .is_some_and(|last| observation.vehicle_state_timestamp_us <= last)
        {
            return Err(BridgeError::VehicleTimestampRegression);
        }
        if observation.timestamp_us.abs_diff(observation.vehicle_state_timestamp_us)
            > self.config.maximum_sensor_clock_skew_us
        {
            return Err(BridgeError::SensorClockSkewExceeded);
        }
        Ok(dt_us)
    }

    fn validate_upstream_observation(
        &self,
        observation: &CapturedDrivingObservation<'_>,
    ) -> Result<(), BridgeError> {
        let upstream = &observation.upstream;
        for value in [
            upstream.observation_id.as_str(),
            upstream.episode_id.as_str(),
            upstream.agent_id.as_str(),
            upstream.clock_id.as_str(),
            upstream.camera_id.as_str(),
            upstream.renderer_revision.as_str(),
        ] {
            if value.trim().is_empty() || value.len() > MAX_ID_BYTES {
                return Err(BridgeError::UpstreamReceiptMalformed);
            }
        }
        if !is_sha256_hex(&upstream.observation_id)
            || !is_sha256_hex(&upstream.receipt_sha256)
            || !is_sha256_hex(&upstream.frame_sha256)
            || !is_sha256_hex(&upstream.camera_calibration_sha256)
        {
            return Err(BridgeError::UpstreamReceiptMalformed);
        }
        if upstream.renderer_revision.len() > 128
            || upstream.schema_version != RECEIPT_SCHEMA_VERSION
            || upstream.pixel_format != "rgb8_srgb"
            || upstream.channels != 3
        {
            return Err(BridgeError::UpstreamReceiptMalformed);
        }
        let expected_frame_bytes = (upstream.width as usize)
            .checked_mul(upstream.height as usize)
            .and_then(|pixels| pixels.checked_mul(upstream.channels))
            .ok_or(BridgeError::UpstreamReceiptMalformed)?;
        if upstream.frame_byte_len != expected_frame_bytes {
            return Err(BridgeError::UpstreamReceiptMalformed);
        }
        if upstream.sequence == 0 {
            return Err(BridgeError::UpstreamSequenceMismatch);
        }
        if upstream.episode_id != self.config.identity.episode_id
            || upstream.agent_id != self.config.identity.agent_id
            || upstream.camera_id != self.config.identity.camera_id
        {
            return Err(BridgeError::UpstreamIdentityMismatch);
        }
        if self
            .config
            .expected_camera_calibration_sha256
            .as_deref()
            .is_some_and(|expected| expected != upstream.camera_calibration_sha256)
        {
            return Err(BridgeError::UpstreamCalibrationMismatch);
        }
        if self
            .last_upstream_clock_id
            .as_deref()
            .is_some_and(|last| last != upstream.clock_id)
            || self
                .last_upstream_tick
                .is_some_and(|last| upstream.tick < last)
        {
            return Err(BridgeError::UpstreamIdentityMismatch);
        }
        if upstream.sequence != observation.sequence {
            return Err(BridgeError::UpstreamSequenceMismatch);
        }
        if upstream.capture_timestamp_us != observation.timestamp_us {
            return Err(BridgeError::UpstreamTimestampMismatch);
        }
        if upstream.width != observation.width
            || upstream.height != observation.height
            || upstream.channels != observation.channels
        {
            return Err(BridgeError::UpstreamGeometryMismatch);
        }
        if upstream.render_completed_timestamp_us < upstream.capture_timestamp_us
            || upstream.delivered_timestamp_us < upstream.render_completed_timestamp_us
        {
            return Err(BridgeError::UpstreamClockOrderInvalid);
        }
        let delivery_age_us = upstream
            .delivered_timestamp_us
            .checked_sub(upstream.capture_timestamp_us)
            .ok_or(BridgeError::UpstreamClockOrderInvalid)?;
        if delivery_age_us > self.config.maximum_observation_delivery_age_us {
            return Err(BridgeError::UpstreamDeliveryAgeExceeded);
        }
        if upstream.tick == 0 {
            if upstream.applied_action_receipt_id.is_some()
                || upstream.applied_action_receipt_sha256.is_some()
                || upstream.applied_action_command_id.is_some()
                || upstream.applied_action_tick.is_some()
            {
                return Err(BridgeError::UpstreamActionReferenceInvalid);
            }
        } else if upstream
            .applied_action_receipt_id
            .as_deref()
            .is_none_or(|value| value.is_empty() || value.len() > MAX_ID_BYTES)
            || upstream
                .applied_action_command_id
                .as_deref()
                .is_none_or(|value| value.is_empty() || value.len() > MAX_ID_BYTES)
            || upstream
                .applied_action_receipt_sha256
                .as_deref()
                .is_none_or(|value| !is_sha256_hex(value))
            || upstream.applied_action_tick != Some(upstream.tick)
        {
            return Err(BridgeError::UpstreamActionReferenceInvalid);
        }
        if digest_bytes(observation.pixels) != upstream.frame_sha256 {
            return Err(BridgeError::UpstreamFrameMismatch);
        }
        if upstream.calculated_observation_id() != upstream.observation_id
            || upstream.calculated_receipt_sha256() != upstream.receipt_sha256
        {
            return Err(BridgeError::UpstreamReceiptCommitmentMismatch);
        }
        Ok(())
    }

    pub fn frames_processed(&self) -> u64 {
        self.vision.frame_count()
    }

    pub fn learning_is_frozen(&self) -> bool {
        self.vision.is_learning_frozen()
    }

    pub fn training_steps(&self) -> u64 {
        self.vision.training_steps()
    }

    pub fn is_faulted(&self) -> bool {
        self.faulted
    }
}

fn digest_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest_to_hex(&digest)
}

fn digest_hv(hv: &ContinuousHV) -> String {
    let mut hasher = Sha256::new();
    hasher.update((hv.dim() as u64).to_le_bytes());
    for value in hv.as_slice() {
        hasher.update(value.to_bits().to_le_bytes());
    }
    digest_to_hex(&hasher.finalize())
}

fn digest_vehicle_state(state: &VehicleState, channels: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for channel in channels {
        hasher.update(channel.to_bits().to_le_bytes());
    }
    hasher.update(state.timestamp.to_bits().to_le_bytes());
    digest_to_hex(&hasher.finalize())
}

fn digest_receipt(receipt: &DrivingObservationReceipt) -> String {
    let mut hasher = Sha256::new();
    hasher.update(receipt.schema_version.to_le_bytes());
    hasher.update([u8::from(receipt.previous_vehicle_state_sha256.is_some())]);
    hasher.update([u8::from(receipt.previous_visual_state_sha256.is_some())]);
    hasher.update([u8::from(receipt.previous_receipt_sha256.is_some())]);
    let previous_vehicle_state = receipt.previous_vehicle_state_sha256.as_deref().unwrap_or("");
    let previous_visual_state = receipt.previous_visual_state_sha256.as_deref().unwrap_or("");
    let previous_receipt = receipt.previous_receipt_sha256.as_deref().unwrap_or("");
    for value in [
        receipt.identity.run_id.as_str(),
        receipt.identity.episode_id.as_str(),
        receipt.identity.scenario_id.as_str(),
        receipt.identity.camera_id.as_str(),
        receipt.identity.simulator_revision.as_str(),
        receipt.identity.policy_revision.as_str(),
        receipt.vision_config_sha256.as_str(),
        receipt.vehicle_encoder_seed_sha256.as_str(),
        receipt.upstream_observation_id.as_str(),
        receipt.upstream_receipt_sha256.as_str(),
        receipt.upstream_frame_sha256.as_str(),
        receipt.upstream_clock_id.as_str(),
        receipt.upstream_agent_id.as_str(),
        receipt.upstream_camera_calibration_sha256.as_str(),
        receipt.upstream_renderer_revision.as_str(),
        receipt.upstream_applied_action_receipt_id.as_deref().unwrap_or(""),
        receipt.upstream_applied_action_receipt_sha256.as_deref().unwrap_or(""),
        receipt.upstream_applied_action_command_id.as_deref().unwrap_or(""),
        previous_receipt,
        receipt.raw_frame_sha256.as_str(),
        receipt.visual_state_sha256.as_str(),
        receipt.vehicle_state_sha256.as_str(),
        previous_vehicle_state,
        previous_visual_state,
        receipt.proprioceptive_hv_sha256.as_str(),
        receipt.fused_policy_input_sha256.as_str(),
    ] {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    for value in [
        receipt.capture_sequence,
        receipt.capture_timestamp_us,
        receipt.vehicle_state_timestamp_us,
        receipt.derived_dt_us,
        u64::from(receipt.width),
        u64::from(receipt.height),
        receipt.channels as u64,
        u64::from(receipt.visual_weight_ppm),
        receipt.hdc_dimension as u64,
        receipt.upstream_tick,
        receipt.capture_sequence,
        receipt.upstream_capture_timestamp_us,
        receipt.upstream_render_completed_timestamp_us,
        receipt.upstream_delivered_timestamp_us,
        u64::from(receipt.upstream_applied_action_tick.unwrap_or(0)),
        receipt.vehicle_encoder_levels as u64,
    ] {
        hasher.update(value.to_le_bytes());
    }
    for value in [
        receipt.vision_telemetry.prediction_error,
        receipt.vision_telemetry.manifold_coherence,
        receipt.vision_telemetry.attention_entropy,
        receipt.vision_telemetry.motion_surprise,
    ] {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher.update((receipt.vision_telemetry.num_salient_patches as u64).to_le_bytes());
    hasher.update(receipt.vision_telemetry.frame_sequence.to_le_bytes());
    hasher.update([u8::from(receipt.vision_telemetry.training_triggered)]);
    hasher.update([u8::from(receipt.vision_learning_frozen)]);
    hasher.update(receipt.vision_training_steps_before.to_le_bytes());
    hasher.update(receipt.vision_training_steps_after.to_le_bytes());
    digest_to_hex(&hasher.finalize())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_to_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> DrivingRunIdentity {
        DrivingRunIdentity {
            run_id: "run-001".to_string(),
            episode_id: "episode-001".to_string(),
            scenario_id: "scenario-001".to_string(),
            agent_id: "ego-001".to_string(),
            camera_id: "front-camera".to_string(),
            simulator_revision: "sim-rev-001".to_string(),
            policy_revision: "policy-rev-001".to_string(),
        }
    }

    fn config() -> BridgeConfig {
        BridgeConfig {
            identity: identity(),
            max_width: 32,
            max_height: 32,
            max_frame_bytes: 4096,
            initial_dt_us: 10_000,
            maximum_frame_gap_us: 100_000,
            maximum_observation_delivery_age_us: 5_000,
            maximum_sensor_clock_skew_us: 2_000,
            visual_weight_ppm: 500_000,
            vehicle_encoder_seed_phrase: "symthaea-camera-driving-bridge-test-seed".to_string(),
        }
    }

    fn observation<'a>(
        pixels: &'a [u8],
        sequence: u64,
        timestamp_us: u64,
        vehicle_state: &'a VehicleState,
    ) -> CapturedDrivingObservation<'a> {
        let tick = sequence;
        let mut captured = CapturedDrivingObservation {
            sequence,
            timestamp_us,
            vehicle_state_timestamp_us: timestamp_us,
            pixels,
            width: 16,
            height: 16,
            channels: 3,
            vehicle_state,
            upstream: UpstreamCameraObservation {
                schema_version: 1,
                observation_id: String::new(),
                pixel_format: "rgb8_srgb".to_string(),
                frame_byte_len: pixels.len(),
                receipt_sha256: String::new(),
                frame_sha256: digest_bytes(pixels),
                episode_id: "episode-001".to_string(),
                agent_id: "ego-001".to_string(),
                clock_id: "test-sim-clock".to_string(),
                tick,
                sequence,
                camera_id: "front-camera".to_string(),
                camera_calibration_sha256: "d".repeat(64),
                renderer_revision: "software-camera-q16-rgb-v1".to_string(),
                capture_timestamp_us: timestamp_us,
                render_completed_timestamp_us: timestamp_us + 200,
                delivered_timestamp_us: timestamp_us + 400,
                width: 16,
                height: 16,
                channels: 3,
                applied_action_receipt_id: Some(format!("action-receipt-{sequence}")),
                applied_action_receipt_sha256: Some("e".repeat(64)),
                applied_action_command_id: Some(format!("action-command-{sequence}")),
                applied_action_tick: Some(tick),
            },
        };
        captured.upstream.observation_id = captured.upstream.calculated_observation_id();
        captured.upstream.receipt_sha256 = captured.upstream.calculated_receipt_sha256();
        captured
    }

    fn bridge() -> CameraDrivingBridge {
        CameraDrivingBridge::try_new(config(), VisionConfig::default()).unwrap()
    }

    #[test]
    fn upstream_camera_receipt_accepts_renderer_policy_receipt_field_names() {
        let json = serde_json::json!({
            "schema_version": 1,
            "observation_id": "a".repeat(64),
            "pixel_format": "rgb8_srgb",
            "episode_id": "episode-1",
            "scenario_id": "scenario-1",
            "agent_id": "ego",
            "clock_id": "sim-clock-1",
            "tick": 1,
            "sequence": 1,
            "applied_action_receipt_id": "action-1",
            "applied_action_receipt_sha256": "b".repeat(64),
            "applied_action_command_id": "command-1",
            "applied_action_tick": 1,
            "camera_id": "front",
            "camera_calibration_sha256": "c".repeat(64),
            "renderer_revision": "software-camera-q16-rgb-v1",
            "capture_timestamp_us": 20_000,
            "render_completed_timestamp_us": 20_800,
            "delivered_timestamp_us": 21_000,
            "frame_width": 16,
            "frame_height": 16,
            "channels": 3,
            "frame_byte_len": 768,
            "frame_sha256": "d".repeat(64),
            "receipt_sha256": "e".repeat(64)
        });
        let parsed: UpstreamCameraObservation = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.width, 16);
        assert_eq!(parsed.height, 16);
        assert_eq!(parsed.channels, 3);
        assert_eq!(parsed.frame_sha256, "d".repeat(64));
        assert_eq!(parsed.applied_action_tick, Some(1));
    }

    #[test]
    fn rejects_unpinned_camera_calibration_before_advancing_temporal_state() {
        let mut config = config();
        config.expected_camera_calibration_sha256 = Some("a".repeat(64));
        let mut bridge = CameraDrivingBridge::try_new(config, VisionConfig::default()).unwrap();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let item = observation(&pixels, 1, 10_000, &state);
        assert_eq!(
            bridge.observe(item).err(),
            Some(BridgeError::UpstreamCalibrationMismatch)
        );
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_tampered_upstream_receipt_before_advancing_temporal_state() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut item = observation(&pixels, 1, 10_000, &state);
        item.upstream.receipt_sha256 = "f".repeat(64);
        assert_eq!(
            bridge.observe(item).err(),
            Some(BridgeError::UpstreamReceiptCommitmentMismatch)
        );
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_tampered_upstream_observation_identity_before_advancing_state() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut item = observation(&pixels, 1, 10_000, &state);
        item.upstream.observation_id = "f".repeat(64);
        assert_eq!(
            bridge.observe(item).err(),
            Some(BridgeError::UpstreamReceiptCommitmentMismatch)
        );
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_upstream_frame_byte_length_mismatch_before_advancing_state() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut item = observation(&pixels, 1, 10_000, &state);
        item.upstream.frame_byte_len += 1;
        item.upstream.receipt_sha256 = item.upstream.calculated_receipt_sha256();
        assert_eq!(
            bridge.observe(item).err(),
            Some(BridgeError::UpstreamReceiptMalformed)
        );
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_over_age_camera_delivery_before_advancing_temporal_state() {
        let mut config = config();
        config.maximum_observation_delivery_age_us = 1_000;
        let mut bridge =
            CameraDrivingBridge::try_new(config, VisionConfig::default()).unwrap();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut item = observation(&pixels, 1, 10_000, &state);
        item.upstream.delivered_timestamp_us = item.upstream.capture_timestamp_us + 1_001;
        item.upstream.receipt_sha256 = item.upstream.calculated_receipt_sha256();

        assert_eq!(
            bridge.observe(item).err(),
            Some(BridgeError::UpstreamDeliveryAgeExceeded)
        );
        assert_eq!(bridge.frames_processed(), 0);
        assert_eq!(bridge.training_steps(), 0);
    }

    #[test]
    fn accepts_camera_delivery_at_the_declared_freshness_boundary() {
        let mut config = config();
        config.maximum_observation_delivery_age_us = 1_000;
        let mut bridge =
            CameraDrivingBridge::try_new(config, VisionConfig::default()).unwrap();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut item = observation(&pixels, 1, 10_000, &state);
        item.upstream.delivered_timestamp_us = item.upstream.capture_timestamp_us + 1_000;
        item.upstream.receipt_sha256 = item.upstream.calculated_receipt_sha256();

        let accepted = bridge.observe(item).unwrap();
        assert_eq!(bridge.frames_processed(), 1);
        assert!(accepted.receipt.vision_learning_frozen);
    }

    #[test]
    fn rejects_zero_delivery_age_budget_as_invalid_bridge_config() {
        let mut config = config();
        config.maximum_observation_delivery_age_us = 0;
        assert!(matches!(
            CameraDrivingBridge::try_new(config, VisionConfig::default()),
            Err(BridgeError::InvalidConfig(_))
        ));
    }

    #[test]
    fn constructs_frozen_vision_pipeline() {
        let bridge = bridge();
        assert!(bridge.learning_is_frozen());
        assert_eq!(bridge.training_steps(), 0);
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn emits_dimensioned_fused_policy_input_and_receipt() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let result = bridge.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        assert_eq!(result.policy_input.dim(), HDC_DIMENSION);
        assert_eq!(result.receipt.schema_version, RECEIPT_SCHEMA_VERSION);
        assert_eq!(result.receipt.capture_sequence, 1);
        assert_eq!(result.receipt.derived_dt_us, 10_000);
        assert!(result.receipt.vision_learning_frozen);
        assert_eq!(result.receipt.vision_training_steps_before, 0);
        assert_eq!(result.receipt.vision_training_steps_after, 0);
        assert_eq!(result.receipt.receipt_sha256, digest_receipt(&result.receipt));
    }

    #[test]
    fn same_inputs_and_seed_produce_identical_receipts_and_policy_inputs() {
        let state = VehicleState::cruising(12.0);
        let pixels = vec![64; 16 * 16 * 3];
        let mut first = bridge();
        let mut second = bridge();
        let a = first.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        let b = second.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        assert_eq!(a.receipt, b.receipt);
        assert!(a.policy_input.similarity(&b.policy_input) > 0.999_999);
    }

    #[test]
    fn changed_frame_bytes_change_frame_and_fused_digests() {
        let state = VehicleState::cruising(10.0);
        let first_pixels = vec![10; 16 * 16 * 3];
        let second_pixels = vec![11; 16 * 16 * 3];
        let mut first = bridge();
        let mut second = bridge();
        let a = first.observe(observation(&first_pixels, 1, 10_000, &state)).unwrap();
        let b = second.observe(observation(&second_pixels, 1, 10_000, &state)).unwrap();
        assert_ne!(a.receipt.raw_frame_sha256, b.receipt.raw_frame_sha256);
        assert_ne!(a.receipt.visual_state_sha256, b.receipt.visual_state_sha256);
        assert_ne!(a.receipt.fused_policy_input_sha256, b.receipt.fused_policy_input_sha256);
    }

    #[test]
    fn changed_vehicle_state_changes_proprioceptive_and_fused_digests() {
        let pixels = vec![90; 16 * 16 * 3];
        let state_a = VehicleState::cruising(5.0);
        let state_b = VehicleState::cruising(15.0);
        let mut first = bridge();
        let mut second = bridge();
        let a = first.observe(observation(&pixels, 1, 10_000, &state_a)).unwrap();
        let b = second.observe(observation(&pixels, 1, 10_000, &state_b)).unwrap();
        assert_ne!(a.receipt.vehicle_state_sha256, b.receipt.vehicle_state_sha256);
        assert_ne!(a.receipt.proprioceptive_hv_sha256, b.receipt.proprioceptive_hv_sha256);
        assert_ne!(a.receipt.fused_policy_input_sha256, b.receipt.fused_policy_input_sha256);
    }

    #[test]
    fn vehicle_encoder_seed_is_bound_to_receipt_and_policy_input() {
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut first_config = config();
        let mut second_config = config();
        second_config.vehicle_encoder_seed_phrase = "different-vehicle-seed".to_string();
        let mut first = CameraDrivingBridge::try_new(first_config, VisionConfig::default()).unwrap();
        let mut second = CameraDrivingBridge::try_new(second_config, VisionConfig::default()).unwrap();
        let a = first.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        let b = second.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        assert_ne!(a.receipt.vehicle_encoder_seed_sha256, b.receipt.vehicle_encoder_seed_sha256);
        assert_ne!(a.receipt.proprioceptive_hv_sha256, b.receipt.proprioceptive_hv_sha256);
        assert_ne!(a.receipt.fused_policy_input_sha256, b.receipt.fused_policy_input_sha256);
    }

    #[test]
    fn vision_configuration_is_bound_to_receipt_and_visual_representation() {
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut vision_a = VisionConfig::default();
        let mut vision_b = VisionConfig::default();
        vision_b.seed += 1;
        let mut first = CameraDrivingBridge::try_new(config(), vision_a).unwrap();
        let mut second = CameraDrivingBridge::try_new(config(), vision_b).unwrap();
        let a = first.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        let b = second.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        assert_ne!(a.receipt.vision_config_sha256, b.receipt.vision_config_sha256);
        assert_ne!(a.receipt.visual_state_sha256, b.receipt.visual_state_sha256);
    }

    #[test]
    fn receipt_chain_binds_previous_visual_and_vehicle_states() {
        let mut bridge = bridge();
        let state_a = VehicleState::cruising(10.0);
        let state_b = VehicleState::cruising(11.0);
        let first_pixels = vec![20; 16 * 16 * 3];
        let second_pixels = vec![30; 16 * 16 * 3];
        let first = bridge.observe(observation(&first_pixels, 1, 10_000, &state_a)).unwrap();
        assert_eq!(first.receipt.previous_vehicle_state_sha256, None);
        assert_eq!(first.receipt.previous_visual_state_sha256, None);
        let second = bridge.observe(observation(&second_pixels, 2, 20_000, &state_b)).unwrap();
        assert_eq!(
            second.receipt.previous_vehicle_state_sha256.as_deref(),
            Some(first.receipt.vehicle_state_sha256.as_str())
        );
        assert_eq!(
            second.receipt.previous_visual_state_sha256.as_deref(),
            Some(first.receipt.visual_state_sha256.as_str())
        );
        assert_eq!(
            second.receipt.previous_receipt_sha256.as_deref(),
            Some(first.receipt.receipt_sha256.as_str())
        );
        assert_eq!(
            second.receipt.upstream_observation_id,
            digest_bytes(b"observation-2")
        );
        assert_eq!(second.receipt.upstream_receipt_sha256, "c".repeat(64));
        assert_eq!(second.receipt.receipt_sha256, digest_receipt(&second.receipt));
    }

    #[test]
    fn rejects_upstream_frame_digest_mismatch_before_advancing_vision() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut item = observation(&pixels, 1, 10_000, &state);
        item.upstream.frame_sha256 = "f".repeat(64);
        assert_eq!(
            bridge.observe(item).err(),
            Some(BridgeError::UpstreamFrameMismatch)
        );
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_upstream_receipt_digest_malformed_before_advancing_vision() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut item = observation(&pixels, 1, 10_000, &state);
        item.upstream.receipt_sha256 = "A".repeat(64);
        assert_eq!(
            bridge.observe(item).err(),
            Some(BridgeError::UpstreamReceiptMalformed)
        );
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_upstream_camera_or_agent_identity_mismatch() {
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut camera_bridge = bridge();
        let mut wrong_camera = observation(&pixels, 1, 10_000, &state);
        wrong_camera.upstream.camera_id = "rear-camera".to_string();
        assert_eq!(
            camera_bridge.observe(wrong_camera).err(),
            Some(BridgeError::UpstreamIdentityMismatch)
        );
        assert_eq!(camera_bridge.frames_processed(), 0);

        let mut agent_bridge = bridge();
        let mut wrong_agent = observation(&pixels, 1, 10_000, &state);
        wrong_agent.upstream.agent_id = "other-ego".to_string();
        assert_eq!(
            agent_bridge.observe(wrong_agent).err(),
            Some(BridgeError::UpstreamIdentityMismatch)
        );
        assert_eq!(agent_bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_upstream_sequence_timestamp_and_geometry_mismatch() {
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];

        let mut sequence_bridge = bridge();
        let mut wrong_sequence = observation(&pixels, 1, 10_000, &state);
        wrong_sequence.upstream.sequence = 2;
        assert_eq!(
            sequence_bridge.observe(wrong_sequence).err(),
            Some(BridgeError::UpstreamSequenceMismatch)
        );
        assert_eq!(sequence_bridge.frames_processed(), 0);

        let mut timestamp_bridge = bridge();
        let mut wrong_timestamp = observation(&pixels, 1, 10_000, &state);
        wrong_timestamp.upstream.capture_timestamp_us += 1;
        assert_eq!(
            timestamp_bridge.observe(wrong_timestamp).err(),
            Some(BridgeError::UpstreamTimestampMismatch)
        );
        assert_eq!(timestamp_bridge.frames_processed(), 0);

        let mut geometry_bridge = bridge();
        let mut wrong_geometry = observation(&pixels, 1, 10_000, &state);
        wrong_geometry.upstream.width = 8;
        assert_eq!(
            geometry_bridge.observe(wrong_geometry).err(),
            Some(BridgeError::UpstreamGeometryMismatch)
        );
        assert_eq!(geometry_bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_upstream_action_reference_tick_mismatch() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut item = observation(&pixels, 1, 10_000, &state);
        item.upstream.applied_action_tick = Some(0);
        assert_eq!(
            bridge.observe(item).err(),
            Some(BridgeError::UpstreamActionReferenceInvalid)
        );
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_upstream_clock_order_regression() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut item = observation(&pixels, 1, 10_000, &state);
        item.upstream.render_completed_timestamp_us = 9_999;
        assert_eq!(
            bridge.observe(item).err(),
            Some(BridgeError::UpstreamClockOrderInvalid)
        );
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn derives_timestep_from_capture_clock() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        bridge.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        let result = bridge.observe(observation(&pixels, 2, 35_000, &state)).unwrap();
        assert_eq!(result.receipt.derived_dt_us, 25_000);
    }

    #[test]
    fn rejects_sequence_regression_without_consuming_clock() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        bridge.observe(observation(&pixels, 4, 10_000, &state)).unwrap();
        assert!(matches!(
            bridge.observe(observation(&pixels, 4, 20_000, &state)),
            Err(BridgeError::SequenceRegression)
        ));
        let result = bridge.observe(observation(&pixels, 5, 20_000, &state)).unwrap();
        assert_eq!(result.receipt.derived_dt_us, 10_000);
        assert_eq!(bridge.frames_processed(), 2);
    }

    #[test]
    fn rejects_timestamp_regression_without_advancing_manifold() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        bridge.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        assert!(matches!(
            bridge.observe(observation(&pixels, 2, 9_000, &state)),
            Err(BridgeError::TimestampRegression)
        ));
        assert_eq!(bridge.frames_processed(), 1);
        let result = bridge.observe(observation(&pixels, 2, 20_000, &state)).unwrap();
        assert_eq!(result.receipt.derived_dt_us, 10_000);
    }

    #[test]
    fn rejects_overlong_frame_gap_without_advancing_manifold() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        bridge.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        assert!(matches!(
            bridge.observe(observation(&pixels, 2, 200_001, &state)),
            Err(BridgeError::FrameGapTooLarge)
        ));
        assert_eq!(bridge.frames_processed(), 1);
    }

    #[test]
    fn rejects_bad_frame_length_without_advancing_manifold() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 12];
        assert!(matches!(
            bridge.observe(observation(&pixels, 1, 10_000, &state)),
            Err(BridgeError::FrameLengthMismatch)
        ));
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_vehicle_state_clock_regression() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        bridge.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        let mut stale = observation(&pixels, 2, 20_000, &state);
        stale.vehicle_state_timestamp_us = 9_500;
        assert_eq!(
            bridge.observe(stale).err(),
            Some(BridgeError::VehicleTimestampRegression)
        );
        assert_eq!(bridge.frames_processed(), 1);
        let valid = CapturedDrivingObservation {
            sequence: 2,
            timestamp_us: 20_000,
            vehicle_state_timestamp_us: 20_000,
            pixels: &pixels,
            width: 16,
            height: 16,
            channels: 3,
            vehicle_state: &state,
        };
        assert!(bridge.observe(valid).is_ok());
    }

    #[test]
    fn rejects_sensor_clock_skew() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut item = observation(&pixels, 1, 10_000, &state);
        item.vehicle_state_timestamp_us = 20_000;
        assert!(matches!(
            bridge.observe(item),
            Err(BridgeError::SensorClockSkewExceeded)
        ));
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_non_finite_vehicle_state() {
        let mut bridge = bridge();
        let mut state = VehicleState::cruising(10.0);
        state.yaw_rate = f64::NAN;
        let pixels = vec![128; 16 * 16 * 3];
        assert!(matches!(
            bridge.observe(observation(&pixels, 1, 10_000, &state)),
            Err(BridgeError::NonFiniteVehicleState)
        ));
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_depth_enabled_vision_configuration_without_depth_source() {
        let mut vision_config = VisionConfig::default();
        vision_config.enable_depth = true;
        assert_eq!(
            CameraDrivingBridge::try_new(config(), vision_config).err(),
            Some(BridgeError::DepthInputRequired)
        );
    }

    #[test]
    fn rejects_noncanonical_hdc_dimension() {
        let mut vision_config = VisionConfig::default();
        vision_config.hdc_dim = 8_192;
        assert_eq!(
            CameraDrivingBridge::try_new(config(), vision_config).err(),
            Some(BridgeError::HdcDimensionMismatch)
        );
    }

    #[test]
    fn rejects_visual_weight_above_one_million_ppm() {
        let mut invalid = config();
        invalid.visual_weight_ppm = 1_000_001;
        assert_eq!(
            CameraDrivingBridge::try_new(invalid, VisionConfig::default()).err(),
            Some(BridgeError::InvalidConfig("visual weight must be <= 1,000,000 ppm"))
        );
    }

    #[test]
    fn visual_weight_zero_uses_only_proprioceptive_hv() {
        let mut cfg = config();
        cfg.visual_weight_ppm = 0;
        let mut bridge = CameraDrivingBridge::try_new(cfg, VisionConfig::default()).unwrap();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let result = bridge.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        assert_eq!(
            result.receipt.fused_policy_input_sha256,
            result.receipt.proprioceptive_hv_sha256
        );
    }

    #[test]
    fn visual_weight_one_million_uses_only_visual_hv() {
        let mut cfg = config();
        cfg.visual_weight_ppm = 1_000_000;
        let mut bridge = CameraDrivingBridge::try_new(cfg, VisionConfig::default()).unwrap();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let result = bridge.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        assert_eq!(
            result.receipt.fused_policy_input_sha256,
            result.receipt.visual_state_sha256
        );
    }

    #[test]
    fn rejects_unsupported_channel_layout() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 2];
        let mut item = observation(&pixels, 1, 10_000, &state);
        item.channels = 2;
        assert_eq!(bridge.observe(item).err(), Some(BridgeError::UnsupportedChannels));
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_frame_over_configured_allocation_bound() {
        let mut cfg = config();
        cfg.max_frame_bytes = 512;
        let mut bridge = CameraDrivingBridge::try_new(cfg, VisionConfig::default()).unwrap();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        assert_eq!(
            bridge.observe(observation(&pixels, 1, 10_000, &state)).err(),
            Some(BridgeError::FrameExceedsByteBound)
        );
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_unbounded_vision_manifold_allocation_config() {
        let mut cfg = config();
        cfg.max_width = u32::MAX;
        assert_eq!(
            CameraDrivingBridge::try_new(cfg, VisionConfig::default()).err(),
            Some(BridgeError::InvalidConfig("maximum frame dimensions must be in 1..=4096"))
        );
    }

    #[test]
    fn rejects_missing_run_identity() {
        let mut cfg = config();
        cfg.identity.camera_id.clear();
        assert_eq!(
            CameraDrivingBridge::try_new(cfg, VisionConfig::default()).err(),
            Some(BridgeError::MissingIdentity("camera_id"))
        );
    }

    #[test]
    fn receipt_json_round_trip_preserves_commitment() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let result = bridge.observe(observation(&pixels, 1, 10_000, &state)).unwrap();
        let json = serde_json::to_string(&result.receipt).unwrap();
        let decoded: DrivingObservationReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, result.receipt);
        assert_eq!(decoded.receipt_sha256, digest_receipt(&decoded));
    }
}
