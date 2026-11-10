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
pub const MAX_VEHICLE_ENCODER_LEVELS: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DrivingRunIdentity {
    pub run_id: String,
    pub episode_id: String,
    pub scenario_id: String,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BridgeConfig {
    pub identity: DrivingRunIdentity,
    pub max_width: u32,
    pub max_height: u32,
    pub max_frame_bytes: usize,
    pub initial_dt_us: u64,
    pub maximum_frame_gap_us: u64,
    pub maximum_sensor_clock_skew_us: u64,
    /// Weight of the visual manifold state in parts per million.
    /// The proprioceptive weight is 1,000,000 minus this value.
    pub visual_weight_ppm: u32,
    pub vehicle_encoder_seed_phrase: String,
}

impl BridgeConfig {
    pub fn validate(&self) -> Result<(), BridgeError> {
        self.identity.validate()?;
        if self.max_width == 0 || self.max_height == 0 {
            return Err(BridgeError::InvalidConfig("maximum frame dimensions must be non-zero"));
        }
        if self.max_frame_bytes == 0 || self.max_frame_bytes > MAX_FRAME_BYTES_HARD_LIMIT {
            return Err(BridgeError::InvalidConfig("frame-byte bound must be in 1..=64 MiB"));
        }
        if self.initial_dt_us == 0 || self.maximum_frame_gap_us < self.initial_dt_us {
            return Err(BridgeError::InvalidConfig("invalid initial timestep or maximum frame gap"));
        }
        if self.visual_weight_ppm > 1_000_000 {
            return Err(BridgeError::InvalidConfig("visual weight must be <= 1,000,000 ppm"));
        }
        if self.vehicle_encoder_seed_phrase.trim().is_empty()
            || self.vehicle_encoder_seed_phrase.len() > 1_024
        {
            return Err(BridgeError::InvalidConfig("invalid vehicle encoder seed phrase"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CapturedDrivingObservation<'a> {
    pub sequence: u64,
    pub timestamp_us: u64,
    pub vehicle_state_timestamp_us: u64,
    pub pixels: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub vehicle_state: &'a VehicleState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    TimestampRegression,
    FrameGapTooLarge,
    SensorClockSkewExceeded,
    NonFiniteVehicleState,
    ManifoldRejected,
    NonFinitePerceptionOutput,
    LearningNotFrozen,
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
    pub raw_frame_sha256: String,
    pub visual_state_sha256: String,
    pub vehicle_state_sha256: String,
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
    last_sequence: Option<u64>,
    last_timestamp_us: Option<u64>,
    faulted: bool,
}

impl CameraDrivingBridge {
    pub fn try_new(config: BridgeConfig, mut vision_config: VisionConfig) -> Result<Self, BridgeError> {
        config.validate()?;
        if vision_config.hdc_dim != HDC_DIMENSION {
            return Err(BridgeError::HdcDimensionMismatch);
        }
        if vision_config.enable_depth {
            return Err(BridgeError::DepthInputRequired);
        }
        vision_config.validate().map_err(|_| BridgeError::InvalidConfig("invalid VisionConfig"))?;
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
            last_sequence: None,
            last_timestamp_us: None,
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
        let policy_input = ContinuousHV::weighted_bundle(
            &[&visual_state, &proprioceptive_hv],
            &[visual_weight, proprioceptive_weight],
        )
        .normalize();

        if !visual_state.as_slice().iter().all(|value| value.is_finite())
            || !proprioceptive_hv.as_slice().iter().all(|value| value.is_finite())
            || !policy_input.as_slice().iter().all(|value| value.is_finite())
        {
            self.faulted = true;
            return Err(BridgeError::NonFinitePerceptionOutput);
        }

        let vision_telemetry = VisionTelemetryReceipt::try_from(&telemetry)?;
        let raw_frame_sha256 = digest_bytes(observation.pixels);
        let visual_state_sha256 = digest_hv(&visual_state);
        let vehicle_state_sha256 = digest_vehicle_state(observation.vehicle_state, &state_channels);
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
            raw_frame_sha256,
            visual_state_sha256,
            vehicle_state_sha256,
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
        if observation.timestamp_us.abs_diff(observation.vehicle_state_timestamp_us)
            > self.config.maximum_sensor_clock_skew_us
        {
            return Err(BridgeError::SensorClockSkewExceeded);
        }
        Ok(dt_us)
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
    digest_to_hex(Sha256::digest(bytes).as_slice())
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
    for value in [
        &receipt.identity.run_id,
        &receipt.identity.episode_id,
        &receipt.identity.scenario_id,
        &receipt.identity.camera_id,
        &receipt.identity.simulator_revision,
        &receipt.identity.policy_revision,
        &receipt.raw_frame_sha256,
        &receipt.visual_state_sha256,
        &receipt.vehicle_state_sha256,
        &receipt.proprioceptive_hv_sha256,
        &receipt.fused_policy_input_sha256,
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
            maximum_sensor_clock_skew_us: 2_000,
            visual_weight_ppm: 500_000,
            vehicle_encoder_seed_phrase: "symthaea-camera-driving-bridge-test-seed".to_string(),
        }
    }

    fn frame(value: u8, sequence: u64, timestamp_us: u64, state: &VehicleState) -> CapturedDrivingObservation<'_> {
        let pixels = vec![value; 16 * 16 * 3];
        // This helper's return value must borrow the supplied pixels, so each test
        // constructs its observation inline via observation().
        let _ = pixels;
        let _ = (sequence, timestamp_us, state);
        unreachable!("use observation() so the frame buffer outlives the bridge call")
    }

    fn observation<'a>(
        pixels: &'a [u8],
        sequence: u64,
        timestamp_us: u64,
        vehicle_state: &'a VehicleState,
    ) -> CapturedDrivingObservation<'a> {
        CapturedDrivingObservation {
            sequence,
            timestamp_us,
            vehicle_state_timestamp_us: timestamp_us,
            pixels,
            width: 16,
            height: 16,
            channels: 3,
            vehicle_state,
        }
    }

    fn bridge() -> CameraDrivingBridge {
        CameraDrivingBridge::try_new(config(), VisionConfig::default()).unwrap()
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
        assert_eq!(
            bridge.observe(observation(&pixels, 4, 20_000, &state)),
            Err(BridgeError::SequenceRegression)
        );
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
        assert_eq!(
            bridge.observe(observation(&pixels, 2, 9_000, &state)),
            Err(BridgeError::TimestampRegression)
        );
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
        assert_eq!(
            bridge.observe(observation(&pixels, 2, 200_001, &state)),
            Err(BridgeError::FrameGapTooLarge)
        );
        assert_eq!(bridge.frames_processed(), 1);
    }

    #[test]
    fn rejects_bad_frame_length_without_advancing_manifold() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 12];
        assert_eq!(
            bridge.observe(observation(&pixels, 1, 10_000, &state)),
            Err(BridgeError::FrameLengthMismatch)
        );
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_sensor_clock_skew() {
        let mut bridge = bridge();
        let state = VehicleState::cruising(10.0);
        let pixels = vec![128; 16 * 16 * 3];
        let mut item = observation(&pixels, 1, 10_000, &state);
        item.vehicle_state_timestamp_us = 20_000;
        assert_eq!(
            bridge.observe(item),
            Err(BridgeError::SensorClockSkewExceeded)
        );
        assert_eq!(bridge.frames_processed(), 0);
    }

    #[test]
    fn rejects_non_finite_vehicle_state() {
        let mut bridge = bridge();
        let mut state = VehicleState::cruising(10.0);
        state.yaw_rate = f64::NAN;
        let pixels = vec![128; 16 * 16 * 3];
        assert_eq!(
            bridge.observe(observation(&pixels, 1, 10_000, &state)),
            Err(BridgeError::NonFiniteVehicleState)
        );
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
