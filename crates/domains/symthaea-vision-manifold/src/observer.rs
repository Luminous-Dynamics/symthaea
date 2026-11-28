//! Capability-restricted temporal observer and versioned evidence witness.
//!
//! This module exposes no learning, FEP action-selection, memory, geodesic,
//! dilation, or arbitrary checkpoint-mutation APIs. HDC/CfC outputs are
//! representation diagnostics, not pose probabilities or physical truth.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::manifold::VisionManifold;
use crate::types::{ManifoldState, VisionConfig, VisionTelemetry, VisualModality};

/// Stable observer profile identifier for this wire-contract revision.
pub const TEMPORAL_OBSERVER_PROFILE_ID_V0_1: &str = "symthaea-spacecraft-temporal-observer-v0.1";
/// Current serialized temporal observer checkpoint schema.
pub const TEMPORAL_OBSERVER_CHECKPOINT_SCHEMA_V0_1: u32 = 1;
/// Current emitted temporal witness schema.
pub const TEMPORAL_PERCEPTION_WITNESS_SCHEMA_V0_1: u32 = 1;
const Q16_16_SCALE: f32 = 65_536.0;
const DEFAULT_MAX_FRAME_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_MAX_INTERFRAME_GAP_US: u64 = 1_000_000;

/// Immutable resource and timing contract for one scored episode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalObserverProfileV0_1 {
    pub profile_id: String,
    pub expected_frame_count: u64,
    pub fixed_horizon_us: u64,
    pub max_interframe_gap_us: u64,
    pub max_frame_bytes: u64,
    /// Fixed constructor dimensions are committed into the profile digest.
    #[serde(default)]
    pub frame_width: Option<u32>,
    #[serde(default)]
    pub frame_height: Option<u32>,
}

impl Default for TemporalObserverProfileV0_1 {
    fn default() -> Self {
        Self {
            profile_id: TEMPORAL_OBSERVER_PROFILE_ID_V0_1.to_string(),
            expected_frame_count: 128,
            fixed_horizon_us: 30_000_000,
            max_interframe_gap_us: DEFAULT_MAX_INTERFRAME_GAP_US,
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            frame_width: None,
            frame_height: None,
        }
    }
}

impl TemporalObserverProfileV0_1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.profile_id != TEMPORAL_OBSERVER_PROFILE_ID_V0_1 {
            return Err("unsupported temporal observer profile id".to_string());
        }
        if self.expected_frame_count < 2 {
            return Err("observer expected_frame_count must be at least two".to_string());
        }
        if self.fixed_horizon_us == 0 || self.max_interframe_gap_us == 0 {
            return Err("observer horizon and inter-frame gap must be non-zero".to_string());
        }
        if self.max_interframe_gap_us > self.fixed_horizon_us {
            return Err("observer inter-frame gap exceeds fixed horizon".to_string());
        }
        if self.max_frame_bytes == 0 || self.max_frame_bytes > 256 * 1024 * 1024 {
            return Err("observer max_frame_bytes must be in 1..=268435456".to_string());
        }
        match (self.frame_width, self.frame_height) {
            (None, None) => {}
            (Some(width), Some(height)) if width > 0 && height > 0 => {}
            _ => {
                return Err(
                    "observer fixed frame width/height must be both absent or both non-zero"
                        .to_string(),
                );
            }
        }
        Ok(())
    }
}

/// Per-frame identity and upstream provenance claims.
///
/// All SHA-256 fields are syntactically checked. Their presence does not
/// authenticate the sensor, clock, calibration, or upstream evidence source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObserverFrameIdentityV0_1 {
    pub episode_id: String,
    pub target_id: String,
    pub session_id: String,
    pub trajectory_id: String,
    pub frame_id: String,
    pub sensor_id: String,
    pub source_evidence_id: String,
    pub sequence: u64,
    /// Acquisition time in microseconds in the declared clock domain.
    pub timestamp_us: u64,
    pub modality: VisualModality,
    pub clock_profile_sha256: String,
    pub calibration_sha256: String,
    pub roi_sha256: String,
    pub preprocessing_sha256: String,
}

impl ObserverFrameIdentityV0_1 {
    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("episode_id", self.episode_id.as_str()),
            ("target_id", self.target_id.as_str()),
            ("session_id", self.session_id.as_str()),
            ("trajectory_id", self.trajectory_id.as_str()),
            ("frame_id", self.frame_id.as_str()),
            ("sensor_id", self.sensor_id.as_str()),
            ("source_evidence_id", self.source_evidence_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(format!("observer identity {name} must not be empty"));
            }
        }
        if self.timestamp_us == 0 {
            return Err("observer timestamp_us must be non-zero".to_string());
        }
        if self.modality != VisualModality::Visible {
            return Err("observer v0.1 accepts visible image frames only".to_string());
        }
        for (name, digest) in [
            ("clock_profile_sha256", &self.clock_profile_sha256),
            ("calibration_sha256", &self.calibration_sha256),
            ("roi_sha256", &self.roi_sha256),
            ("preprocessing_sha256", &self.preprocessing_sha256),
        ] {
            if !is_sha256_hex(digest) {
                return Err(format!(
                    "observer identity {name} is not lowercase SHA-256 hex"
                ));
            }
        }
        Ok(())
    }

    fn same_episode_context(&self, next: &Self) -> bool {
        self.episode_id == next.episode_id
            && self.target_id == next.target_id
            && self.session_id == next.session_id
            && self.trajectory_id == next.trajectory_id
            && self.sensor_id == next.sensor_id
            && self.modality == next.modality
            && self.clock_profile_sha256 == next.clock_profile_sha256
            && self.calibration_sha256 == next.calibration_sha256
            && self.roi_sha256 == next.roi_sha256
            && self.preprocessing_sha256 == next.preprocessing_sha256
    }
}

/// Whether a temporal forecast history is available for this observation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TemporalObserverStatusV0_1 {
    WarmingUp,
    Observed,
}

/// Quantized representation telemetry. Values are unsigned Q16.16; all units
/// and claim limits are carried by the containing schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalObserverMetricsV0_1 {
    /// None until a prior frame exists; never encode missing forecast history as zero.
    pub prediction_error_q16_16: Option<u32>,
    pub manifold_coherence_q16_16: u32,
    pub attention_entropy_q16_16: u32,
    pub motion_surprise_q16_16: u32,
    pub cross_scale_prediction_error_q16_16: Option<u32>,
    pub salient_patch_count: u32,
    pub encode_time_us: u64,
    pub evolve_time_us: u64,
}

/// Versioned, non-authoritative temporal perception witness.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalPerceptionWitnessV0_1 {
    pub schema_version: u32,
    pub producer: String,
    pub producer_version: String,
    pub observer_profile_id: String,
    pub observer_profile_sha256: String,
    pub frozen_model_sha256: String,
    pub raw_frame_sha256: String,
    pub identity: ObserverFrameIdentityV0_1,
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub accepted_frame_count: u64,
    pub expected_frame_count: u64,
    pub fixed_horizon_us: u64,
    pub horizon_elapsed_us: u64,
    pub status: TemporalObserverStatusV0_1,
    pub metrics: TemporalObserverMetricsV0_1,
    pub observer_only: bool,
    pub learning_frozen: bool,
    pub raw_frame_retained: bool,
    pub actuation_authority: bool,
    pub pose_probability_claim: bool,
    pub source_authentication_claim: bool,
    pub physical_truth_claim: bool,
}

/// Fixed per-episode image geometry. The observer never silently accepts a
/// resolution, aspect-ratio or channel-layout switch mid-episode.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalObserverFrameGeometryV0_1 {
    pub width: u32,
    pub height: u32,
    pub channels: usize,
}

/// Complete resumable observer checkpoint. The underlying mutable manifold is
/// kept inside the checkpoint type and is only restored through this wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalObserverCheckpointV0_1 {
    pub schema_version: u32,
    pub profile: TemporalObserverProfileV0_1,
    pub config_sha256: String,
    pub observer_profile_sha256: String,
    pub frozen_model_sha256: String,
    pub first_timestamp_us: Option<u64>,
    pub last_identity: Option<ObserverFrameIdentityV0_1>,
    pub frame_geometry: Option<TemporalObserverFrameGeometryV0_1>,
    pub manifold_state: ManifoldState,
}

pub struct ObserverOnlyVisionManifold {
    inner: VisionManifold,
    profile: TemporalObserverProfileV0_1,
    config_sha256: String,
    observer_profile_sha256: String,
    frozen_model_sha256: String,
    first_timestamp_us: Option<u64>,
    last_identity: Option<ObserverFrameIdentityV0_1>,
    frame_geometry: Option<TemporalObserverFrameGeometryV0_1>,
    cross_scale_enabled: bool,
    poisoned: bool,
}

impl ObserverOnlyVisionManifold {
    /// Construct a frozen observer with the default v0.1 resource profile.
    pub fn try_new(config: VisionConfig, width: u32, height: u32) -> Result<Self, String> {
        Self::try_new_with_profile(
            config,
            width,
            height,
            TemporalObserverProfileV0_1::default(),
        )
    }

    /// Construct a frozen observer with an explicit, immutable episode profile.
    pub fn try_new_with_profile(
        config: VisionConfig,
        width: u32,
        height: u32,
        mut profile: TemporalObserverProfileV0_1,
    ) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err("observer constructor dimensions must be non-zero".to_string());
        }
        if profile
            .frame_width
            .is_some_and(|expected| expected != width)
            || profile
                .frame_height
                .is_some_and(|expected| expected != height)
        {
            return Err(
                "observer profile frame geometry conflicts with constructor dimensions".to_string(),
            );
        }
        profile.frame_width = Some(width);
        profile.frame_height = Some(height);
        profile.validate()?;
        if config.enable_predictive_hierarchy {
            return Err(
                "observer v0.1 rejects the adaptive predictive hierarchy until its weights and state can be frozen and attested"
                    .to_string(),
            );
        }
        if config.learning.adaptive_levels {
            return Err(
                "observer v0.1 rejects adaptive encoder quantization; encoding boundaries must remain fixed"
                    .to_string(),
            );
        }
        let config_sha256 = sha256_json(&config)?;
        let cross_scale_enabled = false;
        let mut inner = VisionManifold::try_new(config, width, height)?;
        inner.enable_observer_only()?;
        let state = inner.save_state();
        validate_observer_state(&state)?;
        let frozen_model_sha256 = model_sha256(&state)?;
        let observer_profile_sha256 =
            profile_sha256(&profile, &config_sha256, &frozen_model_sha256)?;
        Ok(Self {
            inner,
            profile,
            config_sha256,
            observer_profile_sha256,
            frozen_model_sha256,
            first_timestamp_us: None,
            last_identity: None,
            frame_geometry: None,
            cross_scale_enabled,
            poisoned: false,
        })
    }

    /// Restore only a checkpoint whose profile, configuration, model and
    /// non-retention invariants are all internally consistent.
    pub fn from_checkpoint(
        config: VisionConfig,
        width: u32,
        height: u32,
        checkpoint: &TemporalObserverCheckpointV0_1,
    ) -> Result<Self, String> {
        if checkpoint.schema_version != TEMPORAL_OBSERVER_CHECKPOINT_SCHEMA_V0_1 {
            return Err("unsupported temporal observer checkpoint schema".to_string());
        }
        checkpoint.profile.validate()?;
        if checkpoint.profile.frame_width != Some(width)
            || checkpoint.profile.frame_height != Some(height)
        {
            return Err(
                "observer checkpoint constructor geometry does not match committed profile"
                    .to_string(),
            );
        }
        if let Some(geometry) = checkpoint.frame_geometry {
            if geometry.width != width || geometry.height != height {
                return Err(
                    "observer checkpoint image dimensions do not match constructor geometry"
                        .to_string(),
                );
            }
        }
        let config_sha256 = sha256_json(&config)?;
        if config_sha256 != checkpoint.config_sha256 {
            return Err("observer checkpoint config digest mismatch".to_string());
        }
        validate_observer_state(&checkpoint.manifold_state)?;
        let frozen_model_sha256 = model_sha256(&checkpoint.manifold_state)?;
        if frozen_model_sha256 != checkpoint.frozen_model_sha256 {
            return Err("observer checkpoint frozen-model digest mismatch".to_string());
        }
        let observer_profile_sha256 =
            profile_sha256(&checkpoint.profile, &config_sha256, &frozen_model_sha256)?;
        if observer_profile_sha256 != checkpoint.observer_profile_sha256 {
            return Err("observer checkpoint profile digest mismatch".to_string());
        }
        validate_run_state(
            &checkpoint.profile,
            checkpoint.first_timestamp_us,
            checkpoint.last_identity.as_ref(),
            checkpoint.frame_geometry,
            checkpoint.manifold_state.frame_count,
        )?;

        let mut observer =
            Self::try_new_with_profile(config.clone(), width, height, checkpoint.profile.clone())?;
        observer.inner.load_state(&checkpoint.manifold_state)?;
        validate_observer_state(&observer.inner.save_state())?;
        if model_sha256(&observer.inner.save_state())? != checkpoint.frozen_model_sha256 {
            return Err("restored observer model differs from checkpoint model".to_string());
        }
        observer.config_sha256 = config_sha256;
        observer.frozen_model_sha256 = frozen_model_sha256;
        observer.observer_profile_sha256 = observer_profile_sha256;
        observer.first_timestamp_us = checkpoint.first_timestamp_us;
        observer.last_identity = checkpoint.last_identity.clone();
        observer.frame_geometry = checkpoint.frame_geometry;
        observer.cross_scale_enabled = false;
        Ok(observer)
    }

    /// Validate identity/time/budget first, then observe a frame and emit an
    /// exact-byte-bound witness. The raw image is never included in the witness
    /// or retained in the manifold checkpoint.
    pub fn observe_frame(
        &mut self,
        identity: ObserverFrameIdentityV0_1,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
    ) -> Result<TemporalPerceptionWitnessV0_1, String> {
        if self.poisoned {
            return Err("observer instance is poisoned after an invariant failure".to_string());
        }
        identity.validate()?;
        self.validate_next_identity(&identity, pixels.len(), width, height, channels)?;

        let dt = match &self.last_identity {
            Some(previous) => (identity.timestamp_us - previous.timestamp_us) as f32 / 1_000_000.0,
            None => 0.0,
        };
        let raw_frame_sha256 = sha256_bytes(pixels);
        let telemetry = match self
            .inner
            .observe_frame_checked(pixels, width, height, channels, dt)
        {
            Ok(telemetry) => telemetry,
            Err(error) => {
                return Err(format!(
                    "observer rejected frame without state advance: {error}"
                ));
            }
        };

        let state = self.inner.save_state();
        if let Err(error) = validate_observer_state(&state) {
            self.poisoned = true;
            return Err(format!(
                "observer invariant failure after observation: {error}"
            ));
        }
        let model_digest_after = match model_sha256(&state) {
            Ok(digest) => digest,
            Err(error) => {
                self.poisoned = true;
                return Err(error);
            }
        };
        if model_digest_after != self.frozen_model_sha256 {
            self.poisoned = true;
            return Err("frozen observer model changed during evaluation".to_string());
        }

        let first_timestamp_us = self.first_timestamp_us.unwrap_or(identity.timestamp_us);
        let horizon_elapsed_us = identity.timestamp_us - first_timestamp_us;
        let accepted_frame_count = state.frame_count;
        let status = if accepted_frame_count <= 1 {
            TemporalObserverStatusV0_1::WarmingUp
        } else {
            TemporalObserverStatusV0_1::Observed
        };
        let prediction_valid = status == TemporalObserverStatusV0_1::Observed;
        let metrics =
            match quantize_telemetry(&telemetry, prediction_valid, self.cross_scale_enabled) {
                Ok(metrics) => metrics,
                Err(error) => {
                    self.poisoned = true;
                    return Err(format!(
                        "observer telemetry could not be represented: {error}"
                    ));
                }
            };

        self.first_timestamp_us = Some(first_timestamp_us);
        self.last_identity = Some(identity.clone());
        self.frame_geometry = Some(TemporalObserverFrameGeometryV0_1 {
            width,
            height,
            channels,
        });
        Ok(TemporalPerceptionWitnessV0_1 {
            schema_version: TEMPORAL_PERCEPTION_WITNESS_SCHEMA_V0_1,
            producer: "symthaea-vision-manifold".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            observer_profile_id: self.profile.profile_id.clone(),
            observer_profile_sha256: self.observer_profile_sha256.clone(),
            frozen_model_sha256: self.frozen_model_sha256.clone(),
            raw_frame_sha256,
            identity,
            width,
            height,
            channels,
            accepted_frame_count,
            expected_frame_count: self.profile.expected_frame_count,
            fixed_horizon_us: self.profile.fixed_horizon_us,
            horizon_elapsed_us,
            status,
            metrics,
            observer_only: true,
            learning_frozen: true,
            raw_frame_retained: false,
            actuation_authority: false,
            pose_probability_claim: false,
            source_authentication_claim: false,
            physical_truth_claim: false,
        })
    }

    fn validate_next_identity(
        &self,
        identity: &ObserverFrameIdentityV0_1,
        byte_len: usize,
        width: u32,
        height: u32,
        channels: usize,
    ) -> Result<(), String> {
        if width == 0 || height == 0 || !matches!(channels, 1 | 3 | 4) {
            return Err("observer frame geometry or channel count is invalid".to_string());
        }
        if let Some(previous) = self.frame_geometry {
            if previous
                != (TemporalObserverFrameGeometryV0_1 {
                    width,
                    height,
                    channels,
                })
            {
                return Err(
                    "observer frame geometry changed within the fixed episode profile".to_string(),
                );
            }
        }
        let expected_bytes = (width as usize)
            .checked_mul(height as usize)
            .and_then(|count| count.checked_mul(channels))
            .ok_or_else(|| "observer frame geometry overflow".to_string())?;
        if expected_bytes != byte_len {
            return Err(format!(
                "observer frame byte length mismatch: expected {expected_bytes}, got {byte_len}"
            ));
        }
        if byte_len as u64 > self.profile.max_frame_bytes {
            return Err("observer frame exceeds configured byte budget".to_string());
        }
        let accepted = self.inner.frame_count();
        if accepted >= self.profile.expected_frame_count {
            return Err("observer expected frame budget is exhausted".to_string());
        }
        match &self.last_identity {
            None => {
                if identity.sequence != 0 {
                    return Err("observer first frame sequence must be zero".to_string());
                }
            }
            Some(previous) => {
                if !previous.same_episode_context(identity) {
                    return Err(
                        "observer episode, target, sensor, clock or calibration scope changed"
                            .to_string(),
                    );
                }
                if previous.frame_id == identity.frame_id {
                    return Err("observer frame_id must not repeat within an episode".to_string());
                }
                if previous.source_evidence_id == identity.source_evidence_id {
                    return Err(
                        "observer source_evidence_id must not repeat within an episode".to_string(),
                    );
                }
                let expected_sequence = previous
                    .sequence
                    .checked_add(1)
                    .ok_or_else(|| "observer sequence overflow".to_string())?;
                if identity.sequence != expected_sequence {
                    return Err(format!(
                        "observer sequence mismatch: expected {expected_sequence}, got {}",
                        identity.sequence
                    ));
                }
                if identity.timestamp_us <= previous.timestamp_us {
                    return Err("observer timestamps must increase strictly".to_string());
                }
                let gap = identity.timestamp_us - previous.timestamp_us;
                if gap > self.profile.max_interframe_gap_us {
                    return Err("observer inter-frame gap exceeds declared profile".to_string());
                }
            }
        }
        let first_timestamp = self.first_timestamp_us.unwrap_or(identity.timestamp_us);
        if identity.timestamp_us - first_timestamp > self.profile.fixed_horizon_us {
            return Err("observer frame exceeds fixed episode horizon".to_string());
        }
        Ok(())
    }

    /// Export a resumable checkpoint that carries the stream identity and
    /// temporal cursor in addition to the frozen manifold state.
    pub fn save_checkpoint(&self) -> Result<TemporalObserverCheckpointV0_1, String> {
        if self.poisoned {
            return Err("cannot checkpoint a poisoned observer instance".to_string());
        }
        let manifold_state = self.inner.save_state();
        validate_observer_state(&manifold_state)?;
        validate_run_state(
            &self.profile,
            self.first_timestamp_us,
            self.last_identity.as_ref(),
            self.frame_geometry,
            manifold_state.frame_count,
        )?;
        let frozen_model_sha256 = model_sha256(&manifold_state)?;
        if frozen_model_sha256 != self.frozen_model_sha256 {
            return Err("observer model digest changed before checkpoint".to_string());
        }
        Ok(TemporalObserverCheckpointV0_1 {
            schema_version: TEMPORAL_OBSERVER_CHECKPOINT_SCHEMA_V0_1,
            profile: self.profile.clone(),
            config_sha256: self.config_sha256.clone(),
            observer_profile_sha256: self.observer_profile_sha256.clone(),
            frozen_model_sha256,
            first_timestamp_us: self.first_timestamp_us,
            last_identity: self.last_identity.clone(),
            frame_geometry: self.frame_geometry,
            manifold_state,
        })
    }

    pub fn telemetry(&self) -> &VisionTelemetry {
        self.inner.telemetry()
    }

    pub fn frame_count(&self) -> u64 {
        self.inner.frame_count()
    }

    pub fn is_observer_only(&self) -> bool {
        !self.poisoned && self.inner.is_observer_only() && self.inner.is_learning_frozen()
    }

    pub fn profile_sha256(&self) -> &str {
        &self.observer_profile_sha256
    }

    pub fn model_sha256(&self) -> &str {
        &self.frozen_model_sha256
    }
}

fn validate_run_state(
    profile: &TemporalObserverProfileV0_1,
    first_timestamp_us: Option<u64>,
    last_identity: Option<&ObserverFrameIdentityV0_1>,
    frame_geometry: Option<TemporalObserverFrameGeometryV0_1>,
    frame_count: u64,
) -> Result<(), String> {
    match (first_timestamp_us, last_identity, frame_geometry) {
        (None, None, None) if frame_count == 0 => Ok(()),
        (Some(first), Some(last), Some(geometry)) => {
            if geometry.width == 0
                || geometry.height == 0
                || !matches!(geometry.channels, 1 | 3 | 4)
            {
                return Err("observer checkpoint frame geometry is invalid".to_string());
            }
            last.validate()?;
            if frame_count == 0 || frame_count > profile.expected_frame_count {
                return Err("observer checkpoint frame count is outside profile budget".to_string());
            }
            if last.sequence.checked_add(1) != Some(frame_count) {
                return Err("observer checkpoint sequence/frame count mismatch".to_string());
            }
            if first > last.timestamp_us || last.timestamp_us - first > profile.fixed_horizon_us {
                return Err("observer checkpoint cursor exceeds fixed horizon".to_string());
            }
            Ok(())
        }
        _ => Err("observer checkpoint cursor or fixed geometry is incomplete".to_string()),
    }
}

fn validate_observer_state(state: &ManifoldState) -> Result<(), String> {
    if !state.observer_only || !state.learning_frozen {
        return Err("observer checkpoint is not frozen observer-only state".to_string());
    }
    if state.last_observed_frame.is_some()
        || state.scene_memory.is_some()
        || state.object_memory.is_some()
        || state.working_memory.is_some()
        || state.scene_graph_enabled
        || state.predictive_state.is_some()
        || state.last_imagination.is_some()
        || !state.last_object_hypotheses.is_empty()
        || !state.last_geodesic.is_empty()
        || state
            .last_intent_hv
            .as_ref()
            .is_some_and(|intent| intent.iter().any(|value| *value != 0.0))
        || state.modality_contexts.iter().any(|context| {
            context.object_memory.is_some()
                || context.working_memory.is_some()
                || context.scene_graph_enabled
                || !context.last_object_hypotheses.is_empty()
        })
    {
        return Err(
            "observer checkpoint violates no-retention, no-learning or no-action invariants"
                .to_string(),
        );
    }
    Ok(())
}

fn quantize_telemetry(
    telemetry: &VisionTelemetry,
    prediction_valid: bool,
    cross_scale_enabled: bool,
) -> Result<TemporalObserverMetricsV0_1, String> {
    let quantize = |name: &str, value: f32| -> Result<u32, String> {
        if !value.is_finite() || value < 0.0 || value > 65_535.0 {
            return Err(format!("observer telemetry {name} is outside Q16.16 range"));
        }
        let scaled = (value * Q16_16_SCALE).round();
        if !scaled.is_finite() || scaled < 0.0 || scaled > u32::MAX as f32 {
            return Err(format!("observer telemetry {name} quantization overflow"));
        }
        Ok(scaled as u32)
    };
    Ok(TemporalObserverMetricsV0_1 {
        prediction_error_q16_16: if prediction_valid {
            Some(quantize("prediction_error", telemetry.prediction_error)?)
        } else {
            None
        },
        manifold_coherence_q16_16: quantize("manifold_coherence", telemetry.manifold_coherence)?,
        attention_entropy_q16_16: quantize("attention_entropy", telemetry.attention_entropy)?,
        motion_surprise_q16_16: quantize("motion_surprise", telemetry.motion_surprise)?,
        cross_scale_prediction_error_q16_16: if cross_scale_enabled && prediction_valid {
            Some(quantize(
                "cross_scale_prediction_error",
                telemetry.cross_scale_prediction_error,
            )?)
        } else {
            None
        },
        salient_patch_count: u32::try_from(telemetry.num_salient_patches)
            .map_err(|_| "observer salient patch count exceeds u32".to_string())?,
        encode_time_us: telemetry.encode_time_us,
        evolve_time_us: telemetry.evolve_time_us,
    })
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("cannot serialize observer digest input: {error}"))?;
    Ok(sha256_bytes(&bytes))
}

fn model_sha256(state: &ManifoldState) -> Result<String, String> {
    sha256_json(&(
        &state.weight_hv,
        &state.feature_weights,
        state.tau_base,
        state.training_steps,
        &state.trainer_state,
    ))
}

fn profile_sha256(
    profile: &TemporalObserverProfileV0_1,
    config_sha256: &str,
    frozen_model_sha256: &str,
) -> Result<String, String> {
    sha256_json(&(profile, config_sha256, frozen_model_sha256))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(sequence: u64, timestamp_us: u64) -> ObserverFrameIdentityV0_1 {
        ObserverFrameIdentityV0_1 {
            episode_id: "episode-1".to_string(),
            target_id: "target-1".to_string(),
            session_id: "session-1".to_string(),
            trajectory_id: "trajectory-1".to_string(),
            frame_id: format!("frame-{sequence}"),
            sensor_id: "rgb-camera-1".to_string(),
            source_evidence_id: format!("evidence-{sequence}"),
            sequence,
            timestamp_us,
            modality: VisualModality::Visible,
            clock_profile_sha256: "1".repeat(64),
            calibration_sha256: "2".repeat(64),
            roi_sha256: "3".repeat(64),
            preprocessing_sha256: "4".repeat(64),
        }
    }

    fn observer() -> ObserverOnlyVisionManifold {
        let profile = TemporalObserverProfileV0_1 {
            expected_frame_count: 4,
            fixed_horizon_us: 100_000,
            max_interframe_gap_us: 50_000,
            ..TemporalObserverProfileV0_1::default()
        };
        ObserverOnlyVisionManifold::try_new_with_profile(VisionConfig::default(), 32, 32, profile)
            .unwrap()
    }

    #[test]
    fn witness_json_matches_v0_1_wire_contract_and_serde_enum_names() {
        let mut observer = observer();
        let frame = vec![96_u8; 32 * 32];
        let witness = observer
            .observe_frame(identity(0, 1_000_000), &frame, 32, 32, 1)
            .unwrap();
        let value = serde_json::to_value(&witness).unwrap();

        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["producer"], "symthaea-vision-manifold");
        assert_eq!(
            value["observer_profile_id"],
            TEMPORAL_OBSERVER_PROFILE_ID_V0_1
        );
        assert_eq!(value["identity"]["modality"], "Visible");
        assert_eq!(value["status"], "warming_up");
        assert!(value["metrics"]["prediction_error_q16_16"].is_null());
        assert_eq!(value["observer_only"], true);
        assert_eq!(value["learning_frozen"], true);
        assert_eq!(value["raw_frame_retained"], false);
        assert_eq!(value["actuation_authority"], false);
        assert_eq!(value["pose_probability_claim"], false);
        assert_eq!(value["source_authentication_claim"], false);
        assert_eq!(value["physical_truth_claim"], false);
    }

    #[test]
    fn adaptive_encoder_profiles_are_rejected_before_construction() {
        let mut predictive = VisionConfig::default();
        predictive.enable_predictive_hierarchy = true;
        assert!(
            ObserverOnlyVisionManifold::try_new(predictive, 32, 32).is_err(),
            "adaptive predictive hierarchy must be frozen and attested before observer use"
        );

        let mut adaptive_levels = VisionConfig::default();
        adaptive_levels.learning.adaptive_levels = true;
        assert!(
            ObserverOnlyVisionManifold::try_new(adaptive_levels, 32, 32).is_err(),
            "adaptive quantization boundaries must be excluded from v0.1"
        );
    }

    #[test]
    fn frame_geometry_is_committed_in_profile_digest() {
        let first = ObserverOnlyVisionManifold::try_new(VisionConfig::default(), 32, 32).unwrap();
        let second = ObserverOnlyVisionManifold::try_new(VisionConfig::default(), 16, 64).unwrap();
        assert_ne!(first.profile_sha256(), second.profile_sha256());
    }

    #[test]
    fn duplicate_frame_and_source_evidence_ids_are_rejected() {
        let mut observer = observer();
        let frame = vec![64_u8; 32 * 32];
        observer
            .observe_frame(identity(0, 1_000_000), &frame, 32, 32, 1)
            .unwrap();

        let mut duplicate_frame = identity(1, 1_033_000);
        duplicate_frame.frame_id = "frame-0".to_string();
        assert!(
            observer
                .observe_frame(duplicate_frame, &frame, 32, 32, 1)
                .is_err()
        );

        let mut duplicate_evidence = identity(1, 1_033_000);
        duplicate_evidence.source_evidence_id = "evidence-0".to_string();
        assert!(
            observer
                .observe_frame(duplicate_evidence, &frame, 32, 32, 1)
                .is_err()
        );

        assert_eq!(observer.frame_count(), 1);
    }

    #[test]
    fn frame_geometry_is_fixed_and_checkpoint_bound() {
        let mut observer = observer();
        let frame = vec![64_u8; 32 * 32];
        observer
            .observe_frame(identity(0, 1_000_000), &frame, 32, 32, 1)
            .unwrap();
        let count = observer.frame_count();

        let changed_shape = vec![64_u8; 16 * 64];
        assert!(
            observer
                .observe_frame(identity(1, 1_033_000), &changed_shape, 16, 64, 1)
                .is_err()
        );
        assert_eq!(observer.frame_count(), count);

        let checkpoint = observer.save_checkpoint().unwrap();
        assert_eq!(
            checkpoint.frame_geometry,
            Some(TemporalObserverFrameGeometryV0_1 {
                width: 32,
                height: 32,
                channels: 1,
            })
        );
    }

    #[test]
    fn witness_binds_exact_frame_bytes_identity_and_frozen_profile() {
        let mut observer = observer();
        let frame = vec![96_u8; 32 * 32];
        let first = observer
            .observe_frame(identity(0, 1_000_000), &frame, 32, 32, 1)
            .unwrap();
        assert_eq!(first.schema_version, 1);
        assert_eq!(first.status, TemporalObserverStatusV0_1::WarmingUp);
        assert!(first.metrics.prediction_error_q16_16.is_none());
        assert_eq!(first.raw_frame_sha256, sha256_bytes(&frame));
        assert!(first.observer_only && first.learning_frozen);
        assert!(!first.raw_frame_retained);
        assert!(!first.actuation_authority);
        assert!(!first.pose_probability_claim);
        assert!(!first.source_authentication_claim);
        assert!(!first.physical_truth_claim);

        let second = observer
            .observe_frame(identity(1, 1_033_000), &frame, 32, 32, 1)
            .unwrap();
        assert_eq!(second.status, TemporalObserverStatusV0_1::Observed);
        assert!(second.metrics.prediction_error_q16_16.is_some());
        assert_eq!(
            first.observer_profile_sha256,
            second.observer_profile_sha256
        );
        assert_eq!(first.frozen_model_sha256, second.frozen_model_sha256);

        let state = observer.save_checkpoint().unwrap();
        assert!(state.manifold_state.last_observed_frame.is_none());
    }

    #[test]
    fn invalid_sequence_or_scope_does_not_advance_manifold() {
        let mut observer = observer();
        let frame = vec![64_u8; 32 * 32];
        observer
            .observe_frame(identity(0, 1_000_000), &frame, 32, 32, 1)
            .unwrap();
        let count = observer.frame_count();

        assert!(
            observer
                .observe_frame(identity(2, 1_033_000), &frame, 32, 32, 1)
                .is_err()
        );
        assert_eq!(observer.frame_count(), count);

        let mut changed_calibration = identity(1, 1_033_000);
        changed_calibration.calibration_sha256 = "5".repeat(64);
        assert!(
            observer
                .observe_frame(changed_calibration, &frame, 32, 32, 1)
                .is_err()
        );
        assert_eq!(observer.frame_count(), count);
    }

    #[test]
    fn invalid_bytes_or_timestamp_are_rejected_before_mutation() {
        let mut observer = observer();
        let wrong_size = vec![1_u8; 3];
        assert!(
            observer
                .observe_frame(identity(0, 1_000_000), &wrong_size, 32, 32, 1)
                .is_err()
        );
        assert_eq!(observer.frame_count(), 0);

        let frame = vec![64_u8; 32 * 32];
        observer
            .observe_frame(identity(0, 1_000_000), &frame, 32, 32, 1)
            .unwrap();
        assert!(
            observer
                .observe_frame(identity(1, 1_000_000), &frame, 32, 32, 1)
                .is_err()
        );
        assert_eq!(observer.frame_count(), 1);
    }

    #[test]
    fn checkpoint_roundtrip_preserves_identity_cursor_and_rejects_tampering() {
        let mut observer = observer();
        let frame = vec![128_u8; 32 * 32];
        observer
            .observe_frame(identity(0, 1_000_000), &frame, 32, 32, 1)
            .unwrap();
        observer
            .observe_frame(identity(1, 1_033_000), &frame, 32, 32, 1)
            .unwrap();
        let checkpoint = observer.save_checkpoint().unwrap();

        let mut restored = ObserverOnlyVisionManifold::from_checkpoint(
            VisionConfig::default(),
            32,
            32,
            &checkpoint,
        )
        .unwrap();
        assert_eq!(restored.frame_count(), 2);
        let third = restored
            .observe_frame(identity(2, 1_066_000), &frame, 32, 32, 1)
            .unwrap();
        assert_eq!(third.status, TemporalObserverStatusV0_1::Observed);

        let mut tampered = checkpoint;
        tampered.manifold_state.last_observed_frame = Some(frame);
        assert!(ObserverOnlyVisionManifold::from_checkpoint(
            VisionConfig::default(),
            32,
            32,
            &tampered,
        )
        .is_err());
    }

    #[test]
    fn checkpoint_constructor_rejects_different_initial_image_geometry() {
        let mut observer = observer();
        let frame = vec![128_u8; 32 * 32];
        observer
            .observe_frame(identity(0, 1_000_000), &frame, 32, 32, 1)
            .unwrap();
        let checkpoint = observer.save_checkpoint().unwrap();

        assert!(
            ObserverOnlyVisionManifold::from_checkpoint(
                VisionConfig::default(),
                16,
                64,
                &checkpoint,
            )
            .is_err()
        );
    }

    #[test]
    fn profile_or_model_digest_tampering_is_rejected() {
        let observer = observer();
        let mut checkpoint = observer.save_checkpoint().unwrap();
        checkpoint.observer_profile_sha256 = "f".repeat(64);
        assert!(
            ObserverOnlyVisionManifold::from_checkpoint(
                VisionConfig::default(),
                32,
                32,
                &checkpoint,
            )
            .is_err()
        );
    }
}
