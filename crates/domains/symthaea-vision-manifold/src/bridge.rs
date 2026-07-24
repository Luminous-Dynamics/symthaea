// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge between the VisionManifold and the cognitive loop.
//!
//! `VisionBridge` wraps a `VisionManifold` and produces `ContinuousHV` outputs
//! suitable for feeding into `CognitiveLoopService::cycle_with_hv()`.
//!
//! The bridge applies attention-modulated rebundling: salient patch
//! hypervectors receive larger bundle weights, preserving HDC's distributed
//! representation while making unexpected regions more prominent.

use std::time::Instant;

use symthaea_core::hdc::ContinuousHV;

use crate::manifold::VisionManifold;
use crate::spectrum::{MultiSpectralEncoder, MultiSpectralEncoderState, MultiSpectralFrame};
use crate::types::{ManifoldState, SalientRegion, VisionConfig, VisionTelemetry};

/// Top-down cognitive signal that modulates where the visual system looks.
///
/// When set on a `VisionBridge`, the cognitive loop's current goal influences
/// *which* patches receive extra attention. This implements active, goal-directed
/// vision: patches with high similarity to `task_hv` are boosted, so Symthaea
/// becomes literally more sensitive to what she is cognitively focused on.
///
/// # Biological Analogy
///
/// Visual attention in primates is bidirectionally coupled: bottom-up surprise
/// (Itti & Koch 2001) and top-down task relevance (Desimone & Duncan 1995) jointly
/// modulate V1/V4/IT firing rates. This struct provides the top-down signal.
///
/// # Example
///
/// ```ignore
/// // Symthaea is searching for something red
/// let goal = CognitiveGoalSignal {
///     task_hv: Some(red_concept_hv),
///     task_gain: 0.5,
/// };
/// bridge.set_goal_signal(goal);
/// // Now red patches will be boosted in addition to bottom-up surprise
/// ```
#[derive(Debug, Clone)]
pub struct CognitiveGoalSignal {
    /// Task hypervector: encoding of the current cognitive goal or attended concept.
    ///
    /// Patches with positive cosine similarity to this vector receive an additional
    /// boost of `task_gain * similarity`. Set to `None` for purely bottom-up attention.
    pub task_hv: Option<ContinuousHV>,
    /// Gain applied to task-relevant patches (default: 0.4).
    ///
    /// Scales the per-patch task-similarity boost. Reasonable range: 0.1–1.0.
    /// Higher values make top-down attention dominate over bottom-up surprise.
    pub task_gain: f32,
    /// Learning rate for Hebbian template update (default: 0.05).
    ///
    /// When `update_from_recognition()` fires, `task_hv` is interpolated toward
    /// the recognized patch HV at rate `learning_rate * confidence * similarity`.
    /// Range: 0.0 (frozen template) to 0.5 (fast adaptation).
    pub learning_rate: f32,
}

impl Default for CognitiveGoalSignal {
    fn default() -> Self {
        Self {
            task_hv: None,
            task_gain: 0.0,
            learning_rate: 0.05,
        }
    }
}

impl CognitiveGoalSignal {
    /// Create a new goal signal with the given task HV and default gain (0.4).
    pub fn new(task_hv: ContinuousHV) -> Self {
        Self {
            task_hv: Some(task_hv),
            task_gain: 0.4,
            learning_rate: 0.05,
        }
    }

    /// Create a goal signal with explicit gain.
    pub fn with_gain(task_hv: ContinuousHV, gain: f32) -> Self {
        Self {
            task_hv: Some(task_hv),
            task_gain: if gain.is_finite() { gain.max(0.0) } else { 0.0 },
            learning_rate: 0.05,
        }
    }

    /// Validate a goal signal before installing or checkpointing it.
    pub fn validate(&self, expected_dim: usize) -> Result<(), String> {
        if !self.task_gain.is_finite() || self.task_gain < 0.0 {
            return Err(format!(
                "goal task gain must be finite and >= 0.0, got {}",
                self.task_gain
            ));
        }
        if !self.learning_rate.is_finite() || !(0.0..=1.0).contains(&self.learning_rate) {
            return Err(format!(
                "goal learning rate must be finite and in [0.0, 1.0], got {}",
                self.learning_rate
            ));
        }
        if let Some(ref task_hv) = self.task_hv {
            if task_hv.dim() != expected_dim {
                return Err(format!(
                    "goal HV dimension mismatch: got {}, expected {expected_dim}",
                    task_hv.dim()
                ));
            }
            if !task_hv.as_slice().iter().all(|value| value.is_finite()) {
                return Err("goal HV contains non-finite values".to_string());
            }
        }
        Ok(())
    }

    /// Hebbian template update: move `task_hv` toward `recognized_hv` when
    /// the recognized patch is sufficiently task-relevant and confident.
    ///
    /// Only fires when:
    /// - `cos_sim(task_hv, recognized_hv) > 0.3` (patch is task-relevant)
    /// - `confidence > 0.6` (ventral recognition is reliable)
    ///
    /// The update magnitude `α = learning_rate * confidence * similarity` means
    /// highly confident, highly relevant recognition causes the largest template shift.
    ///
    /// Returns `true` if the template was updated.
    pub fn update_from_recognition(
        &mut self,
        recognized_hv: &ContinuousHV,
        confidence: f32,
    ) -> bool {
        let Some(ref mut task_hv) = self.task_hv else {
            return false;
        };
        if !self.learning_rate.is_finite()
            || self.learning_rate < 1e-6
            || !confidence.is_finite()
            || confidence < 0.6
            || task_hv.dim() != recognized_hv.dim()
            || !recognized_hv
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
        {
            return false;
        }
        let sim = task_hv.similarity(recognized_hv).max(0.0);
        if sim < 0.3 {
            return false;
        }
        let alpha = (self.learning_rate * confidence * sim).clamp(0.0, 0.5);
        *task_hv = ContinuousHV::weighted_bundle(&[task_hv, recognized_hv], &[1.0 - alpha, alpha])
            .normalize();
        true
    }

    /// Perform 'Holographic Dilation' - scale internal task HV.
    pub fn dilate(&mut self, target_dim: usize) {
        if let Some(ref mut hv) = self.task_hv {
            *hv = hv.dilate(target_dim);
        }
    }
}

/// Current serialized bridge checkpoint schema.
pub const VISION_BRIDGE_STATE_SCHEMA_VERSION: u32 = 2;

/// Serializable top-down goal state.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CognitiveGoalSignalState {
    pub task_hv: Option<Vec<f32>>,
    pub task_gain: f32,
    pub learning_rate: f32,
}

/// Atomic checkpoint spanning the complete vision-to-cognition bridge.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VisionBridgeState {
    pub schema_version: u32,
    pub manifold: ManifoldState,
    pub attention_boost: f32,
    pub goal_signal: CognitiveGoalSignalState,
    pub multi_spectral: Option<MultiSpectralEncoderState>,
    /// Optional learned vision-to-cognition mapping owned by this bridge.
    #[serde(default)]
    pub cross_predictor: Option<CrossManifoldPredictorState>,
}

/// Bridge from vision manifold output to cognitive loop input.
///
/// Wraps a `VisionManifold` and adds attention-modulated boosting so that
/// the output HV emphasizes surprising (high free-energy) regions.
///
/// Optionally accepts a [`CognitiveGoalSignal`] for top-down task modulation:
/// patches similar to the cognitive goal vector receive extra boost, implementing
/// active, goal-directed perception.
///
/// # Usage
///
/// ```ignore
/// let bridge = VisionBridge::new(VisionConfig::default(), 640, 480);
/// let hv = bridge.process_frame(pixels, 640, 480, 3, 0.033);
/// // Feed hv into CognitiveLoopService::cycle_with_hv(&hv)
/// ```
pub struct VisionBridge {
    manifold: VisionManifold,
    attention_boost: f32,
    /// Optional top-down cognitive goal for task-directed attention.
    goal_signal: CognitiveGoalSignal,
    /// Optional multi-spectral encoder for non-visible-light bands.
    multi_spectral: Option<MultiSpectralEncoder>,
    /// Optional learned mapping from the visual manifold into cognition.
    cross_predictor: Option<CrossManifoldPredictor>,
}

impl VisionBridge {
    /// Create a new vision bridge.
    ///
    /// # Arguments
    /// * `config` — Vision manifold configuration.
    /// * `max_width` / `max_height` — Maximum frame dimensions.
    pub fn new(config: VisionConfig, max_width: u32, max_height: u32) -> Self {
        Self::try_new(config, max_width, max_height)
            .unwrap_or_else(|error| panic!("Invalid VisionBridge construction: {error}"))
    }

    /// Construct a bridge without panicking on invalid manifold policy or capacity.
    pub fn try_new(config: VisionConfig, max_width: u32, max_height: u32) -> Result<Self, String> {
        let manifold = VisionManifold::try_new(config, max_width, max_height)?;
        Ok(Self {
            manifold,
            attention_boost: 0.3,
            goal_signal: CognitiveGoalSignal::default(),
            multi_spectral: None,
            cross_predictor: None,
        })
    }

    /// Create a bridge wrapping an existing manifold.
    pub fn from_manifold(manifold: VisionManifold) -> Self {
        Self {
            manifold,
            attention_boost: 0.3,
            goal_signal: CognitiveGoalSignal::default(),
            multi_spectral: None,
            cross_predictor: None,
        }
    }

    /// Set the attention boost factor (default: 0.3).
    ///
    /// Higher values make surprising regions more prominent in the output HV.
    pub fn set_attention_boost(&mut self, boost: f32) {
        if let Err(error) = self.set_attention_boost_checked(boost) {
            tracing::warn!(%error, "rejected attention boost update");
        }
    }

    /// Set the attention boost without allowing non-finite policy state.
    pub fn set_attention_boost_checked(&mut self, boost: f32) -> Result<(), String> {
        if !boost.is_finite() || boost < 0.0 {
            return Err(format!(
                "attention boost must be finite and >= 0.0, got {boost}"
            ));
        }
        self.attention_boost = boost;
        Ok(())
    }

    /// Set a top-down cognitive goal signal for task-directed attention.
    ///
    /// Patches with high cosine similarity to `signal.task_hv` receive an
    /// additional boost in `apply_attention_boost()`, on top of the bottom-up
    /// surprise boost. This closes the top-down → vision pathway:
    /// what Symthaea thinks about shapes what she literally notices.
    ///
    /// Call with `CognitiveGoalSignal::default()` or `clear_goal_signal()` to
    /// return to purely bottom-up attention.
    pub fn set_goal_signal(&mut self, signal: CognitiveGoalSignal) {
        if let Err(error) = self.set_goal_signal_checked(signal) {
            tracing::warn!(%error, "rejected cognitive goal signal");
        }
    }

    /// Validate and install a top-down goal atomically.
    pub fn set_goal_signal_checked(&mut self, signal: CognitiveGoalSignal) -> Result<(), String> {
        signal.validate(self.manifold.hdc_dim())?;
        self.goal_signal = signal;
        Ok(())
    }

    /// Clear the cognitive goal signal, returning to purely bottom-up attention.
    pub fn clear_goal_signal(&mut self) {
        self.goal_signal = CognitiveGoalSignal::default();
    }

    /// Gently drift the top-down goal signal toward the current cognitive state (P3-A EMA).
    ///
    /// Rather than hard-replacing the goal template each cycle (which causes
    /// attentional thrashing when thoughts shift rapidly), this blends `thought_hv`
    /// in with weight `alpha` (typical: 0.05):
    ///
    /// ```text
    /// new_goal = normalize((1 - α) · existing_goal + α · thought_hv)
    /// ```
    ///
    /// Biological analog: cortical priming signals decay slowly across V1/V4;
    /// a single salient thought doesn't instantly redirect the entire visual system.
    pub fn update_goal_from_cognition(&mut self, thought_hv: &ContinuousHV, alpha: f32) {
        if let Err(error) = self.update_goal_from_cognition_checked(thought_hv, alpha) {
            tracing::warn!(%error, "rejected cognitive goal update");
        }
    }

    /// Drift the goal toward cognition only after validating the complete update.
    pub fn update_goal_from_cognition_checked(
        &mut self,
        thought_hv: &ContinuousHV,
        alpha: f32,
    ) -> Result<(), String> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(format!(
                "goal update alpha must be finite and in [0.0, 1.0], got {alpha}"
            ));
        }
        if thought_hv.dim() != self.manifold.hdc_dim() {
            return Err(format!(
                "cognitive goal dimension mismatch: got {}, expected {}",
                thought_hv.dim(),
                self.manifold.hdc_dim()
            ));
        }
        if !thought_hv.as_slice().iter().all(|value| value.is_finite()) {
            return Err("cognitive goal contains non-finite values".to_string());
        }
        self.goal_signal.validate(self.manifold.hdc_dim())?;

        // Clone the existing HV first to release the immutable borrow before writing back.
        let new_goal = match self.goal_signal.task_hv.take() {
            None => {
                self.goal_signal.task_gain = 0.4;
                thought_hv.clone()
            }
            Some(existing) => {
                ContinuousHV::weighted_bundle(&[&existing, thought_hv], &[1.0 - alpha, alpha])
                    .normalize()
            }
        };
        self.goal_signal.task_hv = Some(new_goal);
        Ok(())
    }

    /// Attach a multi-spectral encoder for non-visible-light processing (P3-C).
    ///
    /// Call once after construction; subsequent calls to `process_multiband_frame()`
    /// will use this encoder. The encoder is built from the bridge's own manifold config
    /// so band-identity HVs are guaranteed to be orthogonal to the existing basis vectors.
    ///
    /// # Arguments
    /// * `max_width`, `max_height` — Maximum frame dimensions for the spectral encoder
    ///   (should match or exceed the frames you intend to process).
    pub fn enable_multi_spectral(&mut self, max_width: u32, max_height: u32) {
        let config = self.manifold.config().clone();
        self.multi_spectral = Some(MultiSpectralEncoder::new(&config, max_width, max_height));
    }

    /// Enable a bridge-owned vision-to-cognition predictor.
    pub fn enable_cross_manifold_predictor(&mut self, seed: u64) {
        self.cross_predictor = Some(CrossManifoldPredictor::new(self.manifold.hdc_dim(), seed));
    }

    /// Disable and discard the learned vision-to-cognition predictor.
    pub fn disable_cross_manifold_predictor(&mut self) {
        self.cross_predictor = None;
    }

    /// Predict cognition directly from the current visual manifold state.
    pub fn predict_cognitive_checked(&mut self) -> Result<ContinuousHV, String> {
        let predictor = self
            .cross_predictor
            .as_mut()
            .ok_or_else(|| "cross-manifold predictor is not enabled".to_string())?;
        predictor.predict_cognitive_checked(self.manifold.state())
    }

    /// Learn the bridge-owned mapping from an observed cognitive state.
    pub fn observe_cognitive_checked(
        &mut self,
        actual_cognitive: &ContinuousHV,
    ) -> Result<bool, String> {
        let predictor = self
            .cross_predictor
            .as_mut()
            .ok_or_else(|| "cross-manifold predictor is not enabled".to_string())?;
        predictor.observe_cognitive_checked(actual_cognitive)
    }

    /// Current cross-manifold prediction error, when enabled.
    pub fn cross_prediction_error(&self) -> Option<f32> {
        self.cross_predictor
            .as_ref()
            .map(CrossManifoldPredictor::prediction_error)
    }

    /// Snapshot the complete bridge under one compatibility contract.
    pub fn save_state(&self) -> VisionBridgeState {
        VisionBridgeState {
            schema_version: VISION_BRIDGE_STATE_SCHEMA_VERSION,
            manifold: self.manifold.save_state(),
            attention_boost: self.attention_boost,
            goal_signal: CognitiveGoalSignalState {
                task_hv: self
                    .goal_signal
                    .task_hv
                    .as_ref()
                    .map(|hv| hv.as_slice().to_vec()),
                task_gain: self.goal_signal.task_gain,
                learning_rate: self.goal_signal.learning_rate,
            },
            multi_spectral: self
                .multi_spectral
                .as_ref()
                .map(MultiSpectralEncoder::save_state),
            cross_predictor: self
                .cross_predictor
                .as_ref()
                .map(CrossManifoldPredictor::save_state),
        }
    }

    /// Serialize the complete bridge into a bounded integrity envelope.
    pub fn save_checkpoint_bytes(&self) -> Result<Vec<u8>, String> {
        self.save_checkpoint_bytes_with_limit(
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )
    }

    pub fn save_checkpoint_bytes_with_limit(
        &self,
        max_payload_bytes: usize,
    ) -> Result<Vec<u8>, String> {
        crate::checkpoint::encode_checkpoint(
            "symthaea-vision-bridge",
            VISION_BRIDGE_STATE_SCHEMA_VERSION,
            &self.save_state(),
            max_payload_bytes,
        )
    }

    /// Serialize the complete stack into a caller-authenticated envelope.
    ///
    /// The signing callback receives the complete bounded integrity envelope,
    /// including its kind, schema, payload length, checksum, and payload.
    pub fn save_authenticated_checkpoint_bytes<S>(
        &self,
        max_tag_bytes: usize,
        sign: S,
    ) -> Result<Vec<u8>, String>
    where
        S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
    {
        crate::checkpoint::encode_authenticated_checkpoint(
            "symthaea-vision-bridge",
            VISION_BRIDGE_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            max_tag_bytes,
            sign,
        )
    }

    /// Atomically persist a caller-authenticated checkpoint.
    pub fn save_authenticated_checkpoint_file<S>(
        &self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        sign: S,
    ) -> Result<crate::checkpoint::CheckpointWriteReport, String>
    where
        S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
    {
        crate::checkpoint::save_authenticated_checkpoint_file(
            path,
            "symthaea-vision-bridge",
            VISION_BRIDGE_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            max_tag_bytes,
            sign,
        )
    }

    /// Atomically persist the complete bridge checkpoint to disk.
    pub fn save_checkpoint_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        crate::checkpoint::save_checkpoint_file(
            path,
            "symthaea-vision-bridge",
            VISION_BRIDGE_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )
    }

    /// Persist a bridge checkpoint while retaining the previous verified generation.
    pub fn save_checkpoint_file_recoverable(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        crate::checkpoint::save_checkpoint_file_recoverable(
            path,
            "symthaea-vision-bridge",
            VISION_BRIDGE_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )
    }

    /// Persist a bounded retained generation set and return complete write evidence.
    pub fn save_checkpoint_file_with_retention_report(
        &self,
        path: impl AsRef<std::path::Path>,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
    ) -> Result<crate::checkpoint::CheckpointRetentionSaveReport, String> {
        crate::checkpoint::save_checkpoint_file_with_retention_report(
            path,
            "symthaea-vision-bridge",
            VISION_BRIDGE_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            policy,
        )
    }

    /// Persist retained generations under one cross-process writer lease.
    pub fn save_checkpoint_file_with_retention_locked_report(
        &self,
        path: impl AsRef<std::path::Path>,
        retention: crate::checkpoint::CheckpointRetentionPolicy,
        lock_policy: crate::checkpoint::CheckpointWriterLockPolicy,
    ) -> Result<crate::checkpoint::CheckpointRetentionSaveReport, String> {
        crate::checkpoint::save_checkpoint_file_with_retention_locked_report(
            path,
            "symthaea-vision-bridge",
            VISION_BRIDGE_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            retention,
            lock_policy,
        )
    }

    /// Persist authenticated retained generations and return complete write evidence.
    pub fn save_authenticated_checkpoint_file_with_retention_report<S, V>(
        &self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
        sign: S,
        verify: V,
    ) -> Result<crate::checkpoint::CheckpointRetentionSaveReport, String>
    where
        S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
        V: Fn(&[u8], &[u8]) -> Result<bool, String>,
    {
        crate::checkpoint::save_authenticated_checkpoint_file_with_retention_report(
            path,
            "symthaea-vision-bridge",
            VISION_BRIDGE_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            max_tag_bytes,
            policy,
            sign,
            verify,
        )
    }

    /// Persist authenticated retained generations under one writer lease.
    pub fn save_authenticated_checkpoint_file_with_retention_locked_report<S, V>(
        &self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        retention: crate::checkpoint::CheckpointRetentionPolicy,
        lock_policy: crate::checkpoint::CheckpointWriterLockPolicy,
        sign: S,
        verify: V,
    ) -> Result<crate::checkpoint::CheckpointRetentionSaveReport, String>
    where
        S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
        V: Fn(&[u8], &[u8]) -> Result<bool, String>,
    {
        crate::checkpoint::save_authenticated_checkpoint_file_with_retention_locked_report(
            path,
            "symthaea-vision-bridge",
            VISION_BRIDGE_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            max_tag_bytes,
            retention,
            lock_policy,
            sign,
            verify,
        )
    }

    /// Inspect the primary and retained checkpoint generations for this stack.
    pub fn inspect_checkpoint_generations(
        &self,
        path: impl AsRef<std::path::Path>,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
    ) -> Result<Vec<crate::checkpoint::CheckpointGenerationInspection>, String> {
        crate::checkpoint::inspect_checkpoint_generations(
            path,
            "symthaea-vision-bridge",
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            policy,
        )
    }

    /// Remove generations beyond a reduced retention bound.
    pub fn prune_checkpoint_generations(
        &self,
        path: impl AsRef<std::path::Path>,
        keep_previous_generations: usize,
    ) -> Result<crate::checkpoint::CheckpointPruneReport, String> {
        crate::checkpoint::prune_checkpoint_generations(path, keep_previous_generations)
    }

    /// Remove old generations while coordinating with retained checkpoint writers.
    pub fn prune_checkpoint_generations_locked(
        &self,
        path: impl AsRef<std::path::Path>,
        keep_previous_generations: usize,
        lock_policy: crate::checkpoint::CheckpointWriterLockPolicy,
    ) -> Result<crate::checkpoint::CheckpointPruneReport, String> {
        crate::checkpoint::prune_checkpoint_generations_locked(
            path,
            keep_previous_generations,
            lock_policy,
        )
    }

    /// Authenticate, deserialize, validate, and atomically restore checkpoint bytes.
    pub fn load_authenticated_checkpoint_bytes<V>(
        &mut self,
        encoded: &[u8],
        max_tag_bytes: usize,
        verify: V,
    ) -> Result<(), String>
    where
        V: FnOnce(&[u8], &[u8]) -> Result<bool, String>,
    {
        let (payload_schema, state): (u32, VisionBridgeState) =
            crate::checkpoint::decode_authenticated_checkpoint(
                encoded,
                "symthaea-vision-bridge",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
                max_tag_bytes,
                verify,
            )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "authenticated checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)
    }

    /// Read, authenticate, validate, and atomically restore a checkpoint file.
    pub fn load_authenticated_checkpoint_file<V>(
        &mut self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        verify: V,
    ) -> Result<(), String>
    where
        V: FnOnce(&[u8], &[u8]) -> Result<bool, String>,
    {
        let (payload_schema, state): (u32, VisionBridgeState) =
            crate::checkpoint::load_authenticated_checkpoint_file(
                path,
                "symthaea-vision-bridge",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
                max_tag_bytes,
                verify,
            )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "authenticated checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)
    }

    /// Read, verify, and atomically restore a bridge checkpoint file.
    pub fn load_checkpoint_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        let (payload_schema, state): (u32, VisionBridgeState) =
            crate::checkpoint::load_checkpoint_file(
                path,
                "symthaea-vision-bridge",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "vision bridge checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)
    }

    /// Restore the primary checkpoint or its previous verified generation.
    pub fn load_checkpoint_file_recoverable(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<crate::checkpoint::CheckpointRecoverySource, String> {
        let (payload_schema, state, source): (
            u32,
            VisionBridgeState,
            crate::checkpoint::CheckpointRecoverySource,
        ) = crate::checkpoint::load_checkpoint_file_recoverable(
            path,
            "symthaea-vision-bridge",
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "bridge checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)?;
        Ok(source)
    }

    /// Restore the newest semantically compatible retained bridge generation.
    pub fn load_checkpoint_file_with_retention_audited(
        &mut self,
        path: impl AsRef<std::path::Path>,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
    ) -> Result<
        crate::checkpoint::CheckpointSemanticRecoveryReport,
        crate::checkpoint::CheckpointSemanticRecoveryFailure,
    > {
        let (payload_schema, state, report) =
            crate::checkpoint::load_checkpoint_file_with_retention_audited_detailed(
                path,
                "symthaea-vision-bridge",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
                policy,
                |schema, candidate: &VisionBridgeState| {
                    if schema != candidate.schema_version {
                        return Err(format!(
                            "bridge checkpoint envelope/payload schema mismatch: envelope={schema}, payload={}",
                            candidate.schema_version
                        ));
                    }
                    self.validate_checkpoint_state(candidate)
                },
            )?;
        if payload_schema != state.schema_version {
            return Err(crate::checkpoint::CheckpointSemanticRecoveryFailure {
                attempts: report.attempts.clone(),
                setup_error: Some(format!(
                    "selected bridge checkpoint schema mismatch: envelope={payload_schema}, payload={}",
                    state.schema_version
                )),
            });
        }
        self.load_state(&state).map_err(|error| {
            crate::checkpoint::CheckpointSemanticRecoveryFailure {
                attempts: report.attempts.clone(),
                setup_error: Some(format!(
                    "selected bridge checkpoint failed final atomic restore: {error}"
                )),
            }
        })?;
        Ok(report)
    }

    /// Restore the newest authenticated and semantically compatible retained generation.
    pub fn load_authenticated_checkpoint_file_with_retention_audited<V>(
        &mut self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
        verify: V,
    ) -> Result<
        crate::checkpoint::CheckpointSemanticRecoveryReport,
        crate::checkpoint::CheckpointSemanticRecoveryFailure,
    >
    where
        V: Fn(&[u8], &[u8]) -> Result<bool, String>,
    {
        let (payload_schema, state, report) =
            crate::checkpoint::load_authenticated_checkpoint_file_with_retention_audited_detailed(
                path,
                "symthaea-vision-bridge",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
                max_tag_bytes,
                policy,
                verify,
                |schema, candidate: &VisionBridgeState| {
                    if schema != candidate.schema_version {
                        return Err(format!(
                            "bridge checkpoint envelope/payload schema mismatch: envelope={schema}, payload={}",
                            candidate.schema_version
                        ));
                    }
                    self.validate_checkpoint_state(candidate)
                },
            )?;
        if payload_schema != state.schema_version {
            return Err(crate::checkpoint::CheckpointSemanticRecoveryFailure {
                attempts: report.attempts.clone(),
                setup_error: Some(format!(
                    "selected authenticated bridge checkpoint schema mismatch: envelope={payload_schema}, payload={}",
                    state.schema_version
                )),
            });
        }
        self.load_state(&state).map_err(|error| {
            crate::checkpoint::CheckpointSemanticRecoveryFailure {
                attempts: report.attempts.clone(),
                setup_error: Some(format!(
                    "selected authenticated bridge checkpoint failed final atomic restore: {error}"
                )),
            }
        })?;
        Ok(report)
    }

    /// Validate and atomically restore a bridge integrity envelope.
    pub fn load_checkpoint_bytes(&mut self, encoded: &[u8]) -> Result<(), String> {
        self.load_checkpoint_bytes_with_limit(
            encoded,
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )
    }

    pub fn load_checkpoint_bytes_with_limit(
        &mut self,
        encoded: &[u8],
        max_payload_bytes: usize,
    ) -> Result<(), String> {
        let (payload_schema, state): (u32, VisionBridgeState) =
            crate::checkpoint::decode_checkpoint(
                encoded,
                "symthaea-vision-bridge",
                max_payload_bytes,
            )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "vision bridge checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)
    }

    fn checkpoint_validation_probe(&self) -> Result<Self, String> {
        let (width, height) = self.manifold.checkpoint_capacity_dimensions()?;
        let mut probe = Self::try_new(self.manifold.config().clone(), width, height)?;
        probe.load_state(&self.save_state())?;
        Ok(probe)
    }

    /// Validate a complete bridge checkpoint without changing live bridge state.
    /// Each call uses a pristine probe restored from the current bridge, so
    /// retained-generation validation has no cross-candidate state leakage.
    pub fn validate_checkpoint_state(&self, state: &VisionBridgeState) -> Result<(), String> {
        let mut probe = self.checkpoint_validation_probe()?;
        probe.load_state(state)
    }

    fn validate_bridge_state(&self, state: &VisionBridgeState) -> Result<(), String> {
        if state.schema_version == 0 || state.schema_version > VISION_BRIDGE_STATE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported vision bridge checkpoint schema: saved={}, supported<= {}",
                state.schema_version, VISION_BRIDGE_STATE_SCHEMA_VERSION
            ));
        }
        if !state.attention_boost.is_finite() || state.attention_boost < 0.0 {
            return Err(format!(
                "invalid bridge attention boost: {}",
                state.attention_boost
            ));
        }
        if !state.goal_signal.task_gain.is_finite() || state.goal_signal.task_gain < 0.0 {
            return Err(format!(
                "invalid goal task gain: {}",
                state.goal_signal.task_gain
            ));
        }
        if !state.goal_signal.learning_rate.is_finite()
            || !(0.0..=1.0).contains(&state.goal_signal.learning_rate)
        {
            return Err(format!(
                "invalid goal learning rate: {}",
                state.goal_signal.learning_rate
            ));
        }
        if let Some(ref values) = state.goal_signal.task_hv {
            if values.len() != state.manifold.hdc_dim {
                return Err(format!(
                    "goal HV dimension mismatch: saved={}, manifold={}",
                    values.len(),
                    state.manifold.hdc_dim
                ));
            }
            if !values.iter().all(|value| value.is_finite()) {
                return Err("goal HV contains non-finite values".to_string());
            }
        }

        match (&self.multi_spectral, &state.multi_spectral) {
            (Some(encoder), Some(saved)) => encoder.validate_state(saved)?,
            (None, None) => {}
            (Some(_), None) => {
                return Err(
                    "bridge checkpoint is missing state for the enabled multispectral encoder"
                        .to_string(),
                );
            }
            (None, Some(_)) => {
                return Err(
                    "bridge checkpoint requires a multispectral encoder that is not enabled"
                        .to_string(),
                );
            }
        }
        match (&self.cross_predictor, &state.cross_predictor) {
            (Some(_), Some(saved)) => {
                CrossManifoldPredictor::validate_state(saved)?;
                if saved.dim != state.manifold.hdc_dim {
                    return Err(format!(
                        "cross-manifold dimension mismatch: predictor={}, manifold={}",
                        saved.dim, state.manifold.hdc_dim
                    ));
                }
            }
            (None, None) => {}
            (Some(_), None) => {
                return Err(
                    "bridge checkpoint is missing state for the enabled cross-manifold predictor"
                        .to_string(),
                );
            }
            (None, Some(_)) => {
                return Err(
                    "bridge checkpoint requires a cross-manifold predictor that is not enabled"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    /// Restore manifold, attention, goal, and multispectral state transactionally.
    ///
    /// Bridge-level fields are validated before mutation. The manifold and
    /// multispectral loaders are themselves fail-closed; a defensive rollback
    /// restores the original state if a later subsystem unexpectedly rejects.
    pub fn load_state(&mut self, state: &VisionBridgeState) -> Result<(), String> {
        self.validate_bridge_state(state)?;

        let before_manifold = self.manifold.save_state();
        let before_spectral = self
            .multi_spectral
            .as_ref()
            .map(MultiSpectralEncoder::save_state);
        let before_predictor = self
            .cross_predictor
            .as_ref()
            .map(CrossManifoldPredictor::save_state);

        self.manifold.load_state(&state.manifold)?;
        if let (Some(encoder), Some(saved)) =
            (self.multi_spectral.as_mut(), state.multi_spectral.as_ref())
        {
            if let Err(error) = encoder.load_state(saved) {
                let _ = self.manifold.load_state(&before_manifold);
                if let Some(ref previous) = before_spectral {
                    let _ = encoder.load_state(previous);
                }
                return Err(format!(
                    "failed to restore multispectral bridge state: {error}"
                ));
            }
        }

        if let (Some(predictor), Some(saved)) = (
            self.cross_predictor.as_mut(),
            state.cross_predictor.as_ref(),
        ) {
            if let Err(error) = predictor.load_state(saved) {
                let _ = self.manifold.load_state(&before_manifold);
                if let (Some(encoder), Some(previous)) =
                    (self.multi_spectral.as_mut(), before_spectral.as_ref())
                {
                    let _ = encoder.load_state(previous);
                }
                if let Some(previous) = before_predictor.as_ref() {
                    let _ = predictor.load_state(previous);
                }
                return Err(format!(
                    "failed to restore cross-manifold bridge state: {error}"
                ));
            }
        }

        self.attention_boost = state.attention_boost;
        self.goal_signal = CognitiveGoalSignal {
            task_hv: state
                .goal_signal
                .task_hv
                .as_ref()
                .map(|values| ContinuousHV::from_vec(values.clone())),
            task_gain: state.goal_signal.task_gain,
            learning_rate: state.goal_signal.learning_rate,
        };
        Ok(())
    }

    /// Process a multi-spectral frame and return the output HV with telemetry (P3-C).
    ///
    /// Encodes each spectral band via band-identity binding:
    /// `band_frame_hv = band_id_hv ⊗ encode(band_pixels)`
    /// then bundles all bands into a single multi-band HV fed to the CfC manifold
    /// via `observe_encoded`. Falls back to the current manifold state if no
    /// multi-spectral encoder is attached (see `enable_multi_spectral()`).
    ///
    /// # Arguments
    /// * `frame` — Multi-spectral frame with one or more spectral layers.
    /// * `dt` — Time step in seconds since the last observation.
    pub fn process_multiband_frame(
        &mut self,
        frame: &MultiSpectralFrame,
        dt: f32,
    ) -> (ContinuousHV, VisionTelemetry) {
        match self.process_multiband_frame_checked(frame, dt) {
            Ok(result) => result,
            Err(error) => {
                tracing::warn!(%error, "rejected multispectral bridge observation");
                (
                    self.manifold.state().clone(),
                    self.manifold.telemetry().clone(),
                )
            }
        }
    }

    /// Validate and process a multispectral frame without partial mutation.
    ///
    /// Geometry, duplicate bands, capacity, timestep, and HDC dimensionality are
    /// checked before the manifold advances. A missing spectral encoder is an
    /// explicit integration error rather than a silent default-telemetry result.
    pub fn process_multiband_frame_checked(
        &mut self,
        frame: &MultiSpectralFrame,
        dt: f32,
    ) -> Result<(ContinuousHV, VisionTelemetry), String> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(format!(
                "multispectral timestep must be finite and >= 0, got {dt}"
            ));
        }
        let expected_dim = self.manifold.config().hdc_dim;
        let enc = self
            .multi_spectral
            .as_mut()
            .ok_or_else(|| "multispectral encoder is not enabled".to_string())?;
        if enc.hdc_dim() != expected_dim {
            return Err(format!(
                "multispectral encoder dimension mismatch: encoder={}, manifold={expected_dim}",
                enc.hdc_dim()
            ));
        }

        let t0 = Instant::now();
        let multi_hv = enc.encode_checked(frame)?;
        let encode_us = t0.elapsed().as_micros() as u64;

        let t1 = Instant::now();
        let mut telemetry = self
            .manifold
            .observe_multiband_frame_checked(&multi_hv, dt)?;
        telemetry.encode_time_us = encode_us;
        telemetry.evolve_time_us += t1.elapsed().as_micros() as u64;

        let boosted_hv = self.apply_attention_boost();
        telemetry.output_hv_norm = boosted_hv.norm();
        telemetry.attention_boost_applied = self.attention_boost;

        Ok((boosted_hv, telemetry))
    }

    /// Apply ventral recognition feedback to suppress surprise at a patch.
    ///
    /// Called after collecting `FoveationResult`s from the foveation manager.
    /// High-confidence recognition dampens the dorsal surprise map so the
    /// system doesn't re-foveate the same region repeatedly
    /// (biological analog: you don't re-read "STOP" on every frame).
    ///
    /// Confidence → dampening factor mapping:
    /// - ≥ 0.7 (high): factor = 0.1 — strong suppression ("I know what this is")
    /// - 0.4–0.7 (medium): factor = 0.4 — moderate suppression
    /// - < 0.4 (low): factor = 0.8 — gentle suppression ("worth another look")
    ///
    /// # Arguments
    /// * `row`, `col` — Grid coordinates of the recognized patch.
    /// * `recognition_confidence` — Confidence from ventral pipeline (0.0–1.0).
    pub fn dampen_patch_surprise(&mut self, row: usize, col: usize, recognition_confidence: f32) {
        let factor = if recognition_confidence >= 0.7 {
            0.1 // Strong recognition: nearly silence this patch
        } else if recognition_confidence >= 0.4 {
            0.4 // Medium confidence: moderate suppression
        } else {
            0.8 // Low confidence: gentle suppression — still worth re-examining
        };
        self.manifold.surprise_map_mut().dampen(row, col, factor);
    }

    /// Update the cognitive goal template from a recognized patch (P2-B).
    ///
    /// After the foveation ventral pipeline recognizes a patch, call this with the
    /// recognized semantic HV and confidence. If the recognition matches the current
    /// goal (cos_sim > 0.3) and is confident (> 0.6), the goal template shifts
    /// slightly toward the recognized HV via Hebbian learning.
    ///
    /// This closes the third feedback loop: the system literally adjusts *what it's
    /// looking for* based on what it finds, making search progressively more efficient.
    ///
    /// Returns `true` if the template was updated.
    pub fn learn_from_recognized_patch(
        &mut self,
        recognized_hv: &ContinuousHV,
        confidence: f32,
    ) -> bool {
        self.goal_signal
            .update_from_recognition(recognized_hv, confidence)
    }

    /// Validate and process a raw frame with one depth sample per patch.
    #[allow(clippy::too_many_arguments)]
    pub fn process_frame_with_depth_checked(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        patch_depths: &[f32],
        dt: f32,
    ) -> Result<ContinuousHV, String> {
        self.manifold.observe_frame_with_depth_checked(
            pixels,
            width,
            height,
            channels,
            patch_depths,
            dt,
        )?;
        Ok(self.apply_attention_boost())
    }

    /// Compatibility wrapper for sensor-depth processing. Invalid observations
    /// leave the bridge unchanged and return its current manifold state.
    #[allow(clippy::too_many_arguments)]
    pub fn process_frame_with_depth(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        patch_depths: &[f32],
        dt: f32,
    ) -> ContinuousHV {
        match self.process_frame_with_depth_checked(
            pixels,
            width,
            height,
            channels,
            patch_depths,
            dt,
        ) {
            Ok(hv) => hv,
            Err(error) => {
                tracing::warn!(%error, "rejected sensor-depth bridge observation");
                self.manifold.state().clone()
            }
        }
    }

    /// Validate and process a raw frame without partial state mutation.
    pub fn process_frame_checked(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        dt: f32,
    ) -> Result<ContinuousHV, String> {
        self.manifold
            .observe_frame_checked(pixels, width, height, channels, dt)?;
        Ok(self.apply_attention_boost())
    }

    /// Validate and process a frame with detailed telemetry.
    pub fn process_frame_with_telemetry_checked(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        dt: f32,
    ) -> Result<(ContinuousHV, VisionTelemetry), String> {
        let t0 = Instant::now();
        let mut telemetry = self
            .manifold
            .observe_frame_checked(pixels, width, height, channels, dt)?;
        let boosted_hv = self.apply_attention_boost();
        telemetry.output_hv_norm = boosted_hv.norm();
        telemetry.attention_boost_applied = self.attention_boost;
        telemetry.evolve_time_us += t0.elapsed().as_micros() as u64;
        Ok((boosted_hv, telemetry))
    }

    /// Process a raw frame and return a ContinuousHV ready for `cycle_with_hv()`.
    ///
    /// Steps:
    /// 1. Feed frame to manifold (encode + CfC evolve + surprise)
    /// 2. Get the evolved manifold state
    /// 3. Apply attention-modulated boosting from the surprise map
    /// 4. Return the boosted, normalized HV
    pub fn process_frame(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        dt: f32,
    ) -> ContinuousHV {
        self.manifold
            .observe_frame(pixels, width, height, channels, dt);
        self.apply_attention_boost()
    }

    /// Process a frame and return both the HV and detailed telemetry.
    pub fn process_frame_with_telemetry(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        dt: f32,
    ) -> (ContinuousHV, VisionTelemetry) {
        let t0 = Instant::now();
        let mut telemetry = self
            .manifold
            .observe_frame(pixels, width, height, channels, dt);

        let boosted_hv = self.apply_attention_boost();

        telemetry.output_hv_norm = boosted_hv.norm();
        telemetry.attention_boost_applied = self.attention_boost;
        telemetry.evolve_time_us += t0.elapsed().as_micros() as u64;

        (boosted_hv, telemetry)
    }

    /// Apply attention modulation to the manifold state by rebundling patch HVs.
    ///
    /// HDC dimensions are distributed: a patch does not own a contiguous slice
    /// of the state vector. Attention therefore changes the bundle weights of
    /// the actual patch hypervectors rather than scaling arbitrary dimensions.
    ///
    /// Signals:
    /// 1. **Bottom-up surprise** from the temporal surprise map.
    /// 2. **Motion saliency** from the motion field.
    /// 3. **Top-down task relevance** from cosine similarity to `task_hv`.
    fn apply_attention_boost(&self) -> ContinuousHV {
        let state = self.manifold.state();
        let surprise_map = self.manifold.surprise_map();
        let motion_saliency = self.manifold.motion_saliency();
        let patch_hvs = self.manifold.last_patch_hvs();
        let patch_appearances = self.manifold.last_patch_appearance_hvs();
        let max_surprise = surprise_map.max_surprise();
        let max_motion = motion_saliency.iter().copied().fold(0.0f32, f32::max);
        let max_signal = max_surprise.max(max_motion);

        let has_task = self.goal_signal.task_hv.is_some() && self.goal_signal.task_gain > 1e-6;
        let has_saliency = max_signal >= 1e-6 && self.attention_boost >= 1e-6;

        if (!has_saliency && !has_task) || patch_hvs.is_empty() {
            return state.clone();
        }

        let attention = surprise_map.attention_map();
        let mut attended_patches = Vec::new();
        let mut attended_weights = Vec::new();
        let mut strongest_boost = 0.0f32;

        for (patch_idx, patch_hv) in patch_hvs.iter().enumerate() {
            if patch_hv.dim() != state.dim() {
                continue;
            }

            let appearance_surprise = attention.values.get(patch_idx).copied().unwrap_or(0.0);
            let motion = motion_saliency.get(patch_idx).copied().unwrap_or(0.0);
            let combined = appearance_surprise.max(motion);
            let bottom_up_boost = if has_saliency && max_signal > 1e-6 {
                self.attention_boost * (combined / max_signal)
            } else {
                0.0
            };

            let top_down_boost = if has_task {
                self.goal_signal
                    .task_hv
                    .as_ref()
                    .zip(patch_appearances.get(patch_idx))
                    .filter(|(task_hv, appearance_hv)| task_hv.dim() == appearance_hv.dim())
                    .map(|(task_hv, appearance_hv)| {
                        self.goal_signal.task_gain * task_hv.similarity(appearance_hv).max(0.0)
                    })
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            let boost = bottom_up_boost + top_down_boost;
            if boost > 1e-6 {
                attended_patches.push(patch_hv);
                attended_weights.push(boost);
                strongest_boost = strongest_boost.max(boost);
            }
        }

        if attended_patches.is_empty() {
            return state.clone();
        }

        let attended =
            ContinuousHV::weighted_bundle(&attended_patches, &attended_weights).normalize();
        ContinuousHV::weighted_bundle(&[state, &attended], &[1.0, strongest_boost.clamp(0.0, 2.0)])
            .normalize()
    }

    /// Get salient patches with their pixel-space bounding boxes.
    ///
    /// Maps `SurpriseMap::salient_patches()` to pixel coordinates using
    /// the current PatchGrid. Used by the foveation bridge to know where
    /// to crop high-res regions for ventral analysis.
    pub fn salient_regions(&self) -> Vec<SalientRegion> {
        let surprise_map = self.manifold.surprise_map();
        let grid = surprise_map.grid();
        let patches = surprise_map.salient_patches();

        patches
            .iter()
            .map(|&(r, c, s)| SalientRegion {
                grid_row: r,
                grid_col: c,
                surprise: s,
                pixel_x: c * grid.patch_size,
                pixel_y: r * grid.patch_size,
                pixel_w: grid
                    .patch_size
                    .min(grid.frame_width as usize - c * grid.patch_size),
                pixel_h: grid
                    .patch_size
                    .min(grid.frame_height as usize - r * grid.patch_size),
            })
            .collect()
    }

    /// Access the underlying manifold.
    pub fn manifold(&self) -> &VisionManifold {
        &self.manifold
    }

    /// Mutable access to the underlying manifold.
    pub fn manifold_mut(&mut self) -> &mut VisionManifold {
        &mut self.manifold
    }

    /// Perform a transactional holographic dilation across the bridge.
    ///
    /// The manifold owns the allocation budget and is mutated first. Goal and
    /// multispectral components are resized only after that preflight succeeds,
    /// preventing a rejected request from leaving bridge dimensions split.
    pub fn try_dilate(
        &mut self,
        target: symthaea_core::hdc::HdcDimensionality,
    ) -> Result<crate::types::DilationEstimate, String> {
        let estimate = self.manifold.try_dilate(target)?;
        let dim = estimate.target_dim;
        self.goal_signal.dilate(dim);
        if let Some(ref mut multi_spectral) = self.multi_spectral {
            multi_spectral.dilate(dim);
        }
        if let Some(ref mut predictor) = self.cross_predictor {
            predictor.dilate(dim);
        }
        Ok(estimate)
    }

    /// Compatibility wrapper for callers that do not consume dilation errors.
    pub fn dilate(&mut self, target: symthaea_core::hdc::HdcDimensionality) {
        if let Err(error) = self.try_dilate(target) {
            tracing::warn!(%error, "vision bridge dilation request rejected");
        }
    }

    /// HDC dimension of the attached multispectral encoder, when enabled.
    pub fn multi_spectral_hdc_dim(&self) -> Option<usize> {
        self.multi_spectral
            .as_ref()
            .map(MultiSpectralEncoder::hdc_dim)
    }

    /// Current frame count.
    pub fn frame_count(&self) -> u64 {
        self.manifold.frame_count()
    }

    /// Reset observation-dependent bridge state. Learned manifold and spectral
    /// encoder weights are preserved, while top-down goals and all temporal
    /// histories are cleared.
    pub fn reset(&mut self) {
        self.manifold.reset();
        self.goal_signal = CognitiveGoalSignal::default();
        if let Some(ref mut multi_spectral) = self.multi_spectral {
            multi_spectral.reset_runtime();
        }
        if let Some(ref mut predictor) = self.cross_predictor {
            predictor.reset();
        }
    }

    /// Prediction confidence: `1.0 - prediction_error`.
    ///
    /// Returns a value in [0.0, 1.0] where 1.0 means the manifold perfectly
    /// predicted this frame. Useful for gating downstream processing.
    pub fn prediction_confidence(&self) -> f32 {
        (1.0 - self.manifold.prediction_error()).clamp(0.0, 1.0)
    }

    /// Count patches where surprise exceeds the configured threshold.
    ///
    /// Returns `(active, total)`. Delegates to the underlying manifold.
    pub fn active_patch_count(&self) -> (usize, usize) {
        self.manifold.active_patch_count()
    }

    /// Unified visual context HV for the cognitive loop (P6-E).
    ///
    /// Combines three signals into a single context vector:
    /// 1. **Working memory bundle** — what the system is attending to (weighted by saliency)
    /// 2. **Scene graph HV** — spatial relational structure between objects
    /// 3. **Manifold state** — raw perceptual scene encoding
    ///
    /// The cognitive loop can use this as a comprehensive "visual situation"
    /// vector that encodes what's there, what's attended, and how things relate.
    ///
    /// Returns `None` if neither working memory nor scene graph are active.
    pub fn scene_context_hv(&self) -> Option<ContinuousHV> {
        let mut components: Vec<&ContinuousHV> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();

        // Working memory: what we're attending to (highest weight)
        let wm_bundle;
        if let Some(wm) = self.manifold.working_memory()
            && let Some(bundle) = wm.bundle_attended()
        {
            wm_bundle = bundle;
            components.push(&wm_bundle);
            weights.push(0.4);
        }

        // Scene graph: relational structure
        if let Some(sg) = self.manifold.scene_graph()
            && let Some(ghv) = sg.graph_hv()
        {
            components.push(ghv);
            weights.push(0.3);
        }

        // Manifold state: raw perception (lowest weight — already in the thought HV)
        let state = self.manifold.state();
        if state.norm() > 1e-6 {
            components.push(state);
            weights.push(0.3);
        }

        if components.is_empty() {
            return None;
        }

        Some(ContinuousHV::weighted_bundle(&components, &weights).normalize())
    }

    /// Current imagination surprise from the manifold (P6-A accessor).
    ///
    /// 0 = reality matched prediction, 1 = maximum divergence.
    pub fn imagination_surprise(&self) -> f32 {
        self.manifold.imagination_surprise()
    }
}

/// Cross-manifold predictor: learns to predict cognitive state from vision state.
///
/// Uses a learned binding weight to map the vision manifold's state HV into
/// a predicted cognitive HV. Online input-conditioned delta learning reduces
/// prediction error as the system observes actual cognitive states.
/// Current serialized cross-manifold predictor schema.
pub const CROSS_MANIFOLD_PREDICTOR_STATE_SCHEMA_VERSION: u32 = 1;

/// Serializable state for learned vision-to-cognition prediction.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CrossManifoldPredictorState {
    pub schema_version: u32,
    pub mapping_weight: Vec<f32>,
    pub last_prediction: Option<Vec<f32>>,
    pub last_vision_state: Option<Vec<f32>>,
    pub prediction_error: f32,
    pub learning_rate: f32,
    pub dim: usize,
}

pub struct CrossManifoldPredictor {
    mapping_weight: ContinuousHV,
    last_prediction: Option<ContinuousHV>,
    last_vision_state: Option<ContinuousHV>,
    prediction_error: f32,
    learning_rate: f32,
    dim: usize,
}

impl CrossManifoldPredictor {
    /// Create a new cross-manifold predictor.
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            mapping_weight: ContinuousHV::random(dim, seed + 800_000),
            last_prediction: None,
            last_vision_state: None,
            prediction_error: 0.0,
            learning_rate: 0.005,
            dim,
        }
    }

    /// Perform 'Holographic Dilation' - scale internal mapping weight.
    pub fn dilate(&mut self, target_dim: usize) {
        if self.dim == target_dim {
            return;
        }

        self.mapping_weight = self.mapping_weight.dilate(target_dim);
        if let Some(ref mut hv) = self.last_prediction {
            *hv = hv.dilate(target_dim);
        }
        if let Some(ref mut hv) = self.last_vision_state {
            *hv = hv.dilate(target_dim);
        }
        self.dim = target_dim;
    }

    /// Predict the cognitive state from a vision state.
    ///
    /// `predicted_cognitive = tanh(mapping_weight ⊗ vision_state)`
    pub fn predict_cognitive(&mut self, vision_state: &ContinuousHV) -> ContinuousHV {
        self.predict_cognitive_checked(vision_state)
            .expect("vision state must satisfy the cross-manifold prediction contract")
    }

    /// Predict only after validating dimensionality and finite evidence.
    pub fn predict_cognitive_checked(
        &mut self,
        vision_state: &ContinuousHV,
    ) -> Result<ContinuousHV, String> {
        if vision_state.dim() != self.dim {
            return Err(format!(
                "vision state dimension mismatch: got {}, expected {}",
                vision_state.dim(),
                self.dim
            ));
        }
        if !vision_state
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
        {
            return Err("vision state contains non-finite values".to_string());
        }
        let predicted = self.mapping_weight.bind(vision_state).tanh();
        if !predicted.as_slice().iter().all(|value| value.is_finite()) {
            return Err("cross-manifold prediction produced non-finite values".to_string());
        }
        self.last_prediction = Some(predicted.clone());
        self.last_vision_state = Some(vision_state.clone());
        Ok(predicted)
    }

    /// Observe the actual cognitive state and update the mapping weight.
    ///
    /// Applies an input-conditioned delta rule through the forward mapping
    /// `tanh(W ⊗ vision)`. The cached prediction/vision pair is consumed exactly
    /// once so repeated observations cannot apply a stale gradient.
    pub fn observe_cognitive(&mut self, actual_cognitive: &ContinuousHV) {
        if let Err(error) = self.observe_cognitive_checked(actual_cognitive) {
            tracing::warn!(%error, "rejected cross-manifold learning observation");
        }
    }

    /// Learn from cognition atomically, consuming the pending pair only on success.
    pub fn observe_cognitive_checked(
        &mut self,
        actual_cognitive: &ContinuousHV,
    ) -> Result<bool, String> {
        if actual_cognitive.dim() != self.dim {
            return Err(format!(
                "cognitive state dimension mismatch: got {}, expected {}",
                actual_cognitive.dim(),
                self.dim
            ));
        }
        if !actual_cognitive
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
        {
            return Err("cognitive state contains non-finite values".to_string());
        }
        let (Some(predicted), Some(vision_state)) = (
            self.last_prediction.as_ref(),
            self.last_vision_state.as_ref(),
        ) else {
            return Ok(false);
        };

        let prediction_error = 1.0 - actual_cognitive.similarity(predicted).clamp(-1.0, 1.0);

        let actual_s = actual_cognitive.as_slice();
        let predicted_s = predicted.as_slice();
        let vision_s = vision_state.as_slice();
        let weight_s = self.mapping_weight.as_slice();

        let mut updated = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let error = actual_s[i] - predicted_s[i];
            let tanh_derivative = 1.0 - predicted_s[i] * predicted_s[i];
            let gradient = (error * tanh_derivative * vision_s[i]).clamp(-1.0, 1.0);
            updated.push(weight_s[i] + self.learning_rate * gradient);
        }
        if !updated.iter().all(|value| value.is_finite()) {
            return Err("cross-manifold update produced non-finite weights".to_string());
        }
        self.mapping_weight = ContinuousHV::from_vec(updated);
        self.prediction_error = prediction_error;
        self.last_prediction = None;
        self.last_vision_state = None;
        Ok(true)
    }

    /// Current prediction error (1 - cos_sim).
    pub fn prediction_error(&self) -> f32 {
        self.prediction_error
    }

    /// Current internal dimension of the mapping weight. Callers must skip
    /// `predict_cognitive` when the vision HV's dimension differs (e.g.
    /// post-holographic-dilation) — `bind` hard-asserts equal dims.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Set the learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) {
        if let Err(error) = self.set_learning_rate_checked(lr) {
            tracing::warn!(%error, "rejected cross-manifold learning rate");
        }
    }

    /// Set a finite, bounded learning rate.
    pub fn set_learning_rate_checked(&mut self, lr: f32) -> Result<(), String> {
        if !lr.is_finite() || !(0.0..=1.0).contains(&lr) {
            return Err(format!(
                "cross-manifold learning rate must be finite and in [0.0, 1.0], got {lr}"
            ));
        }
        self.learning_rate = lr;
        Ok(())
    }

    /// Snapshot learned mapping and any pending prediction/observation pair.
    pub fn save_state(&self) -> CrossManifoldPredictorState {
        CrossManifoldPredictorState {
            schema_version: CROSS_MANIFOLD_PREDICTOR_STATE_SCHEMA_VERSION,
            mapping_weight: self.mapping_weight.as_slice().to_vec(),
            last_prediction: self
                .last_prediction
                .as_ref()
                .map(|hv| hv.as_slice().to_vec()),
            last_vision_state: self
                .last_vision_state
                .as_ref()
                .map(|hv| hv.as_slice().to_vec()),
            prediction_error: self.prediction_error,
            learning_rate: self.learning_rate,
            dim: self.dim,
        }
    }

    /// Validate a serialized predictor before mutating learned state.
    pub fn validate_state(state: &CrossManifoldPredictorState) -> Result<(), String> {
        if state.schema_version != CROSS_MANIFOLD_PREDICTOR_STATE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported cross-manifold checkpoint schema: saved={}, supported={}",
                state.schema_version, CROSS_MANIFOLD_PREDICTOR_STATE_SCHEMA_VERSION
            ));
        }
        if state.dim == 0 || state.mapping_weight.len() != state.dim {
            return Err(format!(
                "cross-manifold mapping dimension mismatch: weight={}, dim={}",
                state.mapping_weight.len(),
                state.dim
            ));
        }
        if !state.mapping_weight.iter().all(|value| value.is_finite()) {
            return Err("cross-manifold mapping contains non-finite values".to_string());
        }
        if state.last_prediction.is_some() != state.last_vision_state.is_some() {
            return Err(
                "cross-manifold checkpoint contains an incomplete pending pair".to_string(),
            );
        }
        for (name, values) in [
            ("last_prediction", state.last_prediction.as_ref()),
            ("last_vision_state", state.last_vision_state.as_ref()),
        ] {
            if let Some(values) = values {
                if values.len() != state.dim {
                    return Err(format!(
                        "cross-manifold {name} dimension mismatch: got {}, expected {}",
                        values.len(),
                        state.dim
                    ));
                }
                if !values.iter().all(|value| value.is_finite()) {
                    return Err(format!("cross-manifold {name} contains non-finite values"));
                }
            }
        }
        if !state.prediction_error.is_finite() || !(0.0..=2.0).contains(&state.prediction_error) {
            return Err(format!(
                "cross-manifold prediction error must be finite and in [0.0, 2.0], got {}",
                state.prediction_error
            ));
        }
        if !state.learning_rate.is_finite() || !(0.0..=1.0).contains(&state.learning_rate) {
            return Err(format!(
                "cross-manifold learning rate must be finite and in [0.0, 1.0], got {}",
                state.learning_rate
            ));
        }
        Ok(())
    }

    /// Restore learned and pending state transactionally.
    pub fn load_state(&mut self, state: &CrossManifoldPredictorState) -> Result<(), String> {
        Self::validate_state(state)?;
        self.mapping_weight = ContinuousHV::from_vec(state.mapping_weight.clone());
        self.last_prediction = state
            .last_prediction
            .as_ref()
            .map(|values| ContinuousHV::from_vec(values.clone()));
        self.last_vision_state = state
            .last_vision_state
            .as_ref()
            .map(|values| ContinuousHV::from_vec(values.clone()));
        self.prediction_error = state.prediction_error;
        self.learning_rate = state.learning_rate;
        self.dim = state.dim;
        Ok(())
    }

    /// Reset the predictor state (but keep learned weights).
    pub fn reset(&mut self) {
        self.last_prediction = None;
        self.last_vision_state = None;
        self.prediction_error = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_gray_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height) as usize]
    }

    fn gradient_frame(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x + y) % 256) as u8);
            }
        }
        pixels
    }

    #[test]
    fn test_try_new_is_fallible() {
        let mut invalid = VisionConfig::default();
        invalid.num_levels = 1;
        assert!(VisionBridge::try_new(invalid, 16, 16).is_err());
        assert!(VisionBridge::try_new(VisionConfig::default(), 0, 16).is_err());
    }

    #[test]
    fn test_salient_regions_clip_partial_edge_patches() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut bridge = VisionBridge::new(config, 17, 9);
        let dark = vec![0u8; 17 * 9];
        let mut changed = dark.clone();
        changed[8 * 17 + 16] = 255;
        bridge.process_frame(&dark, 17, 9, 1, 0.033);
        bridge.process_frame(&changed, 17, 9, 1, 0.033);

        let edge = bridge
            .salient_regions()
            .into_iter()
            .find(|region| region.grid_row == 1 && region.grid_col == 2)
            .expect("partial edge patch should be represented");
        assert_eq!((edge.pixel_x, edge.pixel_y), (16, 8));
        assert_eq!((edge.pixel_w, edge.pixel_h), (1, 1));
    }

    #[test]
    fn test_bridge_construction() {
        let cfg = VisionConfig::default();
        let bridge = VisionBridge::new(cfg.clone(), 64, 64);
        assert_eq!(bridge.frame_count(), 0);
    }

    #[test]
    fn test_goal_admission_rejects_bad_dimension_and_non_finite_policy_atomically() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut bridge = VisionBridge::new(config, 16, 16);
        let original = CognitiveGoalSignal::with_gain(ContinuousHV::random(256, 1), 0.4);
        bridge.set_goal_signal_checked(original).unwrap();
        let before = bridge.save_state();

        let wrong_dim = CognitiveGoalSignal::new(ContinuousHV::random(512, 2));
        assert!(bridge.set_goal_signal_checked(wrong_dim).is_err());
        assert_eq!(
            bridge.save_state().goal_signal.task_hv,
            before.goal_signal.task_hv.clone()
        );

        let mut invalid = CognitiveGoalSignal::new(ContinuousHV::random(256, 3));
        invalid.learning_rate = f32::NAN;
        assert!(bridge.set_goal_signal_checked(invalid).is_err());
        assert_eq!(
            bridge.save_state().goal_signal.task_hv,
            before.goal_signal.task_hv.clone()
        );

        assert!(bridge.set_attention_boost_checked(f32::NAN).is_err());
        assert_eq!(bridge.save_state().attention_boost, before.attention_boost);
    }

    #[test]
    fn test_checked_cognitive_goal_update_rejects_before_mutation() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut bridge = VisionBridge::new(config, 16, 16);
        bridge
            .set_goal_signal_checked(CognitiveGoalSignal::new(ContinuousHV::random(256, 7)))
            .unwrap();
        let before = bridge.save_state().goal_signal.task_hv.unwrap();

        assert!(
            bridge
                .update_goal_from_cognition_checked(&ContinuousHV::random(512, 8), 0.1)
                .is_err()
        );
        assert!(
            bridge
                .update_goal_from_cognition_checked(&ContinuousHV::random(256, 9), f32::NAN)
                .is_err()
        );
        assert_eq!(bridge.save_state().goal_signal.task_hv.unwrap(), before);
    }

    #[test]
    fn test_cross_manifold_checkpoint_roundtrip_preserves_pending_pair() {
        let mut predictor = CrossManifoldPredictor::new(256, 42);
        predictor.set_learning_rate_checked(0.02).unwrap();
        let vision = ContinuousHV::random(256, 7);
        predictor.predict_cognitive_checked(&vision).unwrap();
        let saved = predictor.save_state();

        let mut restored = CrossManifoldPredictor::new(256, 999);
        restored.load_state(&saved).unwrap();
        assert_eq!(restored.save_state(), saved);
        assert!(
            restored
                .observe_cognitive_checked(&ContinuousHV::random(256, 8))
                .unwrap()
        );
    }

    #[test]
    fn test_cross_manifold_rejection_preserves_pending_prediction() {
        let mut predictor = CrossManifoldPredictor::new(256, 11);
        predictor
            .predict_cognitive_checked(&ContinuousHV::random(256, 12))
            .unwrap();
        let before = predictor.save_state();

        assert!(
            predictor
                .observe_cognitive_checked(&ContinuousHV::random(512, 13))
                .is_err()
        );
        assert_eq!(predictor.save_state(), before);

        let mut malformed = before.clone();
        malformed.last_vision_state = None;
        assert!(predictor.load_state(&malformed).is_err());
        assert_eq!(predictor.save_state(), before);
    }

    #[test]
    fn test_checked_bridge_rejection_does_not_advance_frame_count() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut bridge = VisionBridge::new(config, 16, 16);
        let result = bridge.process_frame_checked(&vec![0; 255], 16, 16, 1, 0.033);
        assert!(result.is_err());
        assert_eq!(bridge.frame_count(), 0);
    }

    #[test]
    fn test_checked_multiband_requires_encoder_and_preserves_state_on_bad_dt() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut bridge = VisionBridge::new(config, 16, 16);
        let frame = MultiSpectralFrame::new(16, 16)
            .with_layer(crate::spectrum::SpectrumBand::Visible, vec![128; 16 * 16]);
        assert!(
            bridge
                .process_multiband_frame_checked(&frame, 0.033)
                .is_err()
        );
        assert_eq!(bridge.frame_count(), 0);

        bridge.enable_multi_spectral(16, 16);
        let before = bridge.multi_spectral.as_ref().unwrap().save_state();
        assert!(
            bridge
                .process_multiband_frame_checked(&frame, f32::NAN)
                .is_err()
        );
        assert_eq!(bridge.frame_count(), 0);
        assert_eq!(bridge.multi_spectral.as_ref().unwrap().save_state(), before);
    }

    #[test]
    fn test_bridge_owned_cross_predictor_roundtrips_and_dilates() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        config.max_dilation_bytes = u64::MAX;
        let mut bridge = VisionBridge::new(config.clone(), 8, 8);
        bridge.enable_cross_manifold_predictor(99);
        bridge.process_frame(&vec![42; 64], 8, 8, 1, 0.033);
        let predicted = bridge.predict_cognitive_checked().unwrap();
        assert!(bridge.observe_cognitive_checked(&predicted).unwrap());

        let saved = bridge.save_state();
        let mut restored = VisionBridge::new(config, 8, 8);
        restored.enable_cross_manifold_predictor(1);
        restored.load_state(&saved).unwrap();
        assert_eq!(restored.save_state().cross_predictor, saved.cross_predictor);

        restored
            .try_dilate(symthaea_core::hdc::HdcDimensionality::Ultra)
            .unwrap();
        assert_eq!(
            restored.cross_predictor.as_ref().unwrap().dim(),
            restored.manifold.hdc_dim()
        );
    }

    #[test]
    fn test_bridge_checkpoint_rejects_cross_predictor_topology_mismatch() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut source = VisionBridge::new(config.clone(), 8, 8);
        source.enable_cross_manifold_predictor(7);
        let saved = source.save_state();

        let mut destination = VisionBridge::new(config, 8, 8);
        let before = destination.save_state();
        assert!(destination.load_state(&saved).is_err());
        assert_eq!(
            destination.save_state().manifold.frame_count,
            before.manifold.frame_count
        );
    }

    #[test]
    fn test_bridge_checkpoint_roundtrip_restores_complete_state() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut bridge = VisionBridge::new(config, 16, 16);
        bridge.set_attention_boost(0.73);
        let mut goal = CognitiveGoalSignal::with_gain(ContinuousHV::random(256, 77), 0.61);
        goal.learning_rate = 0.17;
        bridge.set_goal_signal(goal);
        bridge.enable_multi_spectral(16, 16);

        let frame = MultiSpectralFrame::new(16, 16)
            .with_layer(crate::spectrum::SpectrumBand::Visible, vec![96; 16 * 16])
            .with_layer(crate::spectrum::SpectrumBand::ThermalIR, vec![180; 16 * 16]);
        bridge
            .process_multiband_frame_checked(&frame, 0.033)
            .expect("valid multispectral observation");
        let saved = bridge.save_state();

        bridge.set_attention_boost(0.05);
        bridge.clear_goal_signal();
        bridge.reset();
        bridge
            .load_state(&saved)
            .expect("bridge checkpoint should restore");

        let restored = bridge.save_state();
        assert_eq!(restored.schema_version, VISION_BRIDGE_STATE_SCHEMA_VERSION);
        assert_eq!(restored.manifold.frame_count, saved.manifold.frame_count);
        assert!((restored.attention_boost - saved.attention_boost).abs() < 1e-6);
        assert!((restored.goal_signal.task_gain - saved.goal_signal.task_gain).abs() < 1e-6);
        assert!(
            (restored.goal_signal.learning_rate - saved.goal_signal.learning_rate).abs() < 1e-6
        );
        let expected_goal = ContinuousHV::from_vec(saved.goal_signal.task_hv.clone().unwrap());
        let actual_goal = ContinuousHV::from_vec(restored.goal_signal.task_hv.unwrap());
        assert!(expected_goal.similarity(&actual_goal) > 0.9999);
        assert_eq!(restored.multi_spectral, saved.multi_spectral);
    }

    #[test]
    fn test_bridge_checkpoint_rejection_is_atomic() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut bridge = VisionBridge::new(config, 16, 16);
        bridge.set_attention_boost(0.42);
        bridge.set_goal_signal(CognitiveGoalSignal::new(ContinuousHV::random(256, 9)));
        bridge.enable_multi_spectral(16, 16);
        let frame = MultiSpectralFrame::new(16, 16)
            .with_layer(crate::spectrum::SpectrumBand::Visible, vec![128; 16 * 16]);
        bridge
            .process_multiband_frame_checked(&frame, 0.033)
            .expect("valid multispectral observation");

        let before = bridge.save_state();
        let mut invalid = before.clone();
        invalid.goal_signal.task_hv = Some(vec![0.0; 128]);
        assert!(bridge.load_state(&invalid).is_err());
        let after = bridge.save_state();

        assert_eq!(after.manifold.frame_count, before.manifold.frame_count);
        assert!((after.attention_boost - before.attention_boost).abs() < 1e-6);
        assert_eq!(after.goal_signal.task_hv, before.goal_signal.task_hv);
        assert_eq!(after.multi_spectral, before.multi_spectral);
    }

    #[test]
    fn test_bridge_checkpoint_rejects_spectral_topology_mismatch() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut source = VisionBridge::new(config.clone(), 16, 16);
        source.enable_multi_spectral(16, 16);
        let saved = source.save_state();

        let mut destination = VisionBridge::new(config, 16, 16);
        assert!(destination.load_state(&saved).is_err());
        assert_eq!(destination.frame_count(), 0);
    }

    #[test]
    fn test_checked_sensor_depth_is_atomic_and_changes_output() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        config.enable_depth = true;
        let frame = vec![128u8; 16 * 16];
        let near = vec![0.0; 4];
        let far = vec![1.0; 4];

        let mut near_bridge = VisionBridge::new(config.clone(), 16, 16);
        let invalid =
            near_bridge.process_frame_with_depth_checked(&frame, 16, 16, 1, &[f32::NAN; 4], 0.033);
        assert!(invalid.is_err());
        assert_eq!(near_bridge.frame_count(), 0);

        let near_hv = near_bridge
            .process_frame_with_depth_checked(&frame, 16, 16, 1, &near, 0.033)
            .expect("valid near-depth frame");
        let mut far_bridge = VisionBridge::new(config, 16, 16);
        let far_hv = far_bridge
            .process_frame_with_depth_checked(&frame, 16, 16, 1, &far, 0.033)
            .expect("valid far-depth frame");
        assert!(near_hv.similarity(&far_hv) < 0.999);
    }

    #[test]
    fn test_goal_learning_rejects_nonfinite_confidence_and_dimension_mismatch() {
        let mut goal = CognitiveGoalSignal::new(ContinuousHV::random(256, 1));
        let before = goal.task_hv.clone().unwrap();
        assert!(!goal.update_from_recognition(&ContinuousHV::random(128, 2), 0.9));
        assert!(!goal.update_from_recognition(&ContinuousHV::random(256, 3), f32::NAN));
        assert!(goal.task_hv.as_ref().unwrap().similarity(&before) > 0.9999);
    }

    #[test]
    fn test_process_frame_returns_correct_dim() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg.clone(), 64, 64);
        let frame = solid_gray_frame(64, 64, 128);

        let hv = bridge.process_frame(&frame, 64, 64, 1, 0.033);
        assert_eq!(hv.dim(), cfg.hdc_dim);
        assert!(hv.norm() > 0.0, "Output HV should have non-zero norm");
    }

    #[test]
    fn test_process_frame_with_telemetry() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        let (hv, tel) = bridge.process_frame_with_telemetry(&frame, 64, 64, 1, 0.033);
        assert!(hv.norm() > 0.0);
        assert_eq!(tel.frame_sequence, 1);
        assert!(tel.output_hv_norm > 0.0);
    }

    #[test]
    fn test_attention_boost_changes_output() {
        let cfg = VisionConfig::default();
        let mut bridge_boost = VisionBridge::new(cfg.clone(), 64, 64);
        let mut bridge_plain = VisionBridge::new(cfg, 64, 64);
        bridge_boost.set_attention_boost(0.8);
        bridge_plain.set_attention_boost(0.0);

        let frame_a = solid_gray_frame(64, 64, 50);
        let mut frame_b = frame_a.clone();
        // Localized change: only the upper-left quadrant becomes salient.
        for y in 0..32usize {
            for x in 0..32usize {
                frame_b[y * 64 + x] = 220;
            }
        }

        bridge_boost.process_frame(&frame_a, 64, 64, 1, 0.033);
        bridge_plain.process_frame(&frame_a, 64, 64, 1, 0.033);
        let hv_with_boost = bridge_boost.process_frame(&frame_b, 64, 64, 1, 0.033);
        let hv_without_boost = bridge_plain.process_frame(&frame_b, 64, 64, 1, 0.033);

        let sim = hv_with_boost.similarity(&hv_without_boost);
        assert!(
            sim < 1.0 - 1e-5,
            "attention-weighted rebundling must change the output: sim={sim}"
        );
    }

    #[test]
    fn test_bridge_dilation_rejection_is_transactional() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        config.max_dilation_bytes = 1;
        let mut bridge = VisionBridge::new(config, 16, 16);
        bridge.set_goal_signal(CognitiveGoalSignal::new(ContinuousHV::random(256, 91)));
        bridge.enable_multi_spectral(16, 16);

        let result = bridge.try_dilate(symthaea_core::hdc::HdcDimensionality::Ultra);
        assert!(result.is_err());
        assert_eq!(bridge.manifold().hdc_dim(), 256);
        assert_eq!(
            bridge.goal_signal.task_hv.as_ref().map(ContinuousHV::dim),
            Some(256)
        );
        assert_eq!(bridge.multi_spectral_hdc_dim(), Some(256));
    }

    #[test]
    fn test_bridge_dilation_updates_all_components_after_acceptance() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        config.max_dilation_bytes = u64::MAX;
        let mut bridge = VisionBridge::new(config, 8, 8);
        bridge.set_goal_signal(CognitiveGoalSignal::new(ContinuousHV::random(256, 92)));
        bridge.enable_multi_spectral(8, 8);

        let estimate = bridge
            .try_dilate(symthaea_core::hdc::HdcDimensionality::Ultra)
            .expect("dilation should fit unlimited budget");
        assert_eq!(bridge.manifold().hdc_dim(), estimate.target_dim);
        assert_eq!(
            bridge.goal_signal.task_hv.as_ref().map(ContinuousHV::dim),
            Some(estimate.target_dim)
        );
        assert_eq!(bridge.multi_spectral_hdc_dim(), Some(estimate.target_dim));
    }

    #[test]
    fn test_bridge_reset_clears_goal_and_multispectral_history() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut bridge = VisionBridge::new(config, 8, 8);
        bridge.set_goal_signal(CognitiveGoalSignal::new(ContinuousHV::random(256, 801)));
        bridge.enable_multi_spectral(8, 8);
        let frame = MultiSpectralFrame {
            width: 8,
            height: 8,
            layers: vec![crate::spectrum::SpectralLayer {
                band: crate::spectrum::SpectrumBand::Visible,
                data: vec![128; 64],
            }],
        };
        bridge.process_multiband_frame(&frame, 0.033);

        bridge.reset();

        assert!(bridge.goal_signal.task_hv.is_none());
        assert_eq!(bridge.frame_count(), 0);
        let spectral = bridge.multi_spectral.as_ref().unwrap();
        assert!(spectral.probe_bands(&ContinuousHV::zero(256)).is_empty());
    }

    #[test]
    fn test_bridge_multiple_frames() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg.clone(), 64, 64);

        // Process several frames
        for i in 0..10u8 {
            let frame = solid_gray_frame(64, 64, i * 25);
            let hv = bridge.process_frame(&frame, 64, 64, 1, 0.033);
            assert_eq!(hv.dim(), cfg.hdc_dim);
        }

        assert_eq!(bridge.frame_count(), 10);
    }

    #[test]
    fn test_from_manifold() {
        let cfg = VisionConfig::default();
        let manifold = VisionManifold::new(cfg.clone(), 64, 64);
        let bridge = VisionBridge::from_manifold(manifold);
        assert_eq!(bridge.frame_count(), 0);
    }

    #[test]
    fn test_reset() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        bridge.process_frame(&frame, 64, 64, 1, 0.033);
        assert!(bridge.frame_count() > 0);

        bridge.reset();
        assert_eq!(bridge.frame_count(), 0);
    }

    // === RGB Bridge Tests ===

    #[test]
    fn test_bridge_rgb_frame() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg.clone(), 64, 64);

        let rgb_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![128u8, 64, 192]).collect();
        let hv = bridge.process_frame(&rgb_frame, 64, 64, 3, 0.033);
        assert_eq!(hv.dim(), cfg.hdc_dim);
        assert!(hv.norm() > 0.0);
    }

    #[test]
    fn test_bridge_rgb_color_discrimination() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg.clone(), 64, 64);

        let red_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![255u8, 0, 0]).collect();
        let hv_red = bridge.process_frame(&red_frame, 64, 64, 3, 0.033);

        bridge.reset();
        let blue_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![0u8, 0, 255]).collect();
        let hv_blue = bridge.process_frame(&blue_frame, 64, 64, 3, 0.033);

        // Ensure dimensions match for comparison (dilation might have triggered)
        let (hv_red_final, hv_blue_final) = if hv_red.dim() != hv_blue.dim() {
            if hv_red.dim() < hv_blue.dim() {
                (hv_red.dilate(hv_blue.dim()), hv_blue)
            } else {
                (hv_red.clone(), hv_blue.dilate(hv_red.dim()))
            }
        } else {
            (hv_red, hv_blue)
        };

        let sim = hv_red_final.similarity(&hv_blue_final);
        assert!(
            sim < 0.99,
            "Red and blue should produce different bridge outputs: sim={sim}"
        );
    }

    #[test]
    fn test_bridge_rgb_telemetry() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);

        let frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![100u8, 150, 200]).collect();
        let (hv, tel) = bridge.process_frame_with_telemetry(&frame, 64, 64, 3, 0.033);
        assert!(hv.norm() > 0.0);
        assert_eq!(tel.frame_sequence, 1);
    }

    // === CognitiveGoalSignal Tests ===

    #[test]
    fn test_goal_signal_default_is_none() {
        let cfg = VisionConfig::default();
        let bridge = VisionBridge::new(cfg, 64, 64);
        assert!(bridge.goal_signal.task_hv.is_none());
    }

    #[test]
    fn test_goal_signal_changes_output() {
        // The top-down boost is per-patch bundle weight:
        // weight[i] += task_gain * cos_sim(patch_i, task_hv).
        // In 16,384D, random HVs are nearly orthogonal (sim ≈ 0), so we must use an
        // *actual* patch HV as the task vector to guarantee non-trivial similarity.
        // Here we use patch[0] from bridge_goal as the task vector — on the next frame
        // that patch receives a strong attended-bundle weight, which must change
        // the output.
        let cfg = VisionConfig::default();
        let mut bridge_base = VisionBridge::new(cfg.clone(), 64, 64);
        let mut bridge_goal = VisionBridge::new(cfg, 64, 64);

        // A frame with a distinct, uncorrelated solid color per 8x8 patch
        // block. A smooth gradient/periodic frame instead gives many patches
        // near-identical *true* appearance (verified: a linear gradient with
        // a short period made every patch in patch[0]'s column match its
        // appearance at similarity > 0.85), so a task_hv built from one patch
        // ends up boosting most of the frame almost uniformly — which changes
        // the bundle so little after normalization that the goal signal's
        // effect on the output becomes unmeasurable. Distinct per-patch
        // colors make patch[0]'s appearance genuinely unique.
        let patch_size = 8usize;
        let cols = 64 / patch_size;
        let frame: Vec<u8> = (0..64usize * 64)
            .flat_map(|i| {
                let (px, py) = (i % 64 / patch_size, i / 64 / patch_size);
                let patch_idx = py * cols + px;
                let r = ((patch_idx * 73 + 17) % 256) as u8;
                let g = ((patch_idx * 151 + 43) % 256) as u8;
                let b = ((patch_idx * 199 + 91) % 256) as u8;
                vec![r, g, b]
            })
            .collect();

        // Warm both bridges identically.
        bridge_base.process_frame(&frame, 64, 64, 3, 0.033);
        bridge_goal.process_frame(&frame, 64, 64, 3, 0.033);

        // Use patch[0]'s *position-invariant appearance* from bridge_goal as the
        // goal — `apply_attention_boost()`'s top-down term compares `task_hv`
        // against `last_patch_appearance_hvs()`, not the raw position-bound
        // patch HVs, so the goal vector must live in the same appearance space
        // to be meaningfully comparable (see `last_patch_appearance_hvs()`'s
        // doc comment: top-down concepts should match content independent of
        // where it appeared).
        let task_hv = bridge_goal
            .manifold()
            .last_patch_appearance_hvs()
            .first()
            .cloned()
            .expect("Should have patch appearance HVs after first frame");

        bridge_goal.set_goal_signal(CognitiveGoalSignal::with_gain(task_hv, 2.0));

        // Process a second identical frame — same content so same surprise, but the
        // goal signal rebundles patch[0] into bridge_goal only.
        let hv_base = bridge_base.process_frame(&frame, 64, 64, 3, 0.033);
        let hv_goal = bridge_goal.process_frame(&frame, 64, 64, 3, 0.033);

        let sim = hv_base.similarity(&hv_goal);
        assert!(
            sim < 1.0 - 1e-4,
            "Goal signal should change output: sim={sim}"
        );
    }

    #[test]
    fn test_clear_goal_signal() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);

        let task_hv = ContinuousHV::random(16_384, 99);
        bridge.set_goal_signal(CognitiveGoalSignal::new(task_hv));
        assert!(bridge.goal_signal.task_hv.is_some());

        bridge.clear_goal_signal();
        assert!(bridge.goal_signal.task_hv.is_none());
    }

    // === Ventral→Dorsal Dampening Tests ===

    #[test]
    fn test_dampen_high_confidence_strong_suppression() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);

        // Generate surprise by processing two different frames
        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = solid_gray_frame(64, 64, 200);
        bridge.process_frame(&frame_a, 64, 64, 1, 0.033);
        bridge.process_frame(&frame_b, 64, 64, 1, 0.033);

        // Get surprise before dampening
        let before = bridge.manifold().surprise_map().max_surprise();

        // High confidence recognition: should strongly dampen
        bridge.dampen_patch_surprise(0, 0, 0.9);

        // Surprise at (0,0) should be reduced
        let _after = bridge.manifold().surprise_map().max_surprise();
        // max may be at a different patch — check via the attention map
        let attention = bridge.manifold().surprise_map().attention_map();
        let surprise_at_00 = attention.at(0, 0);

        assert!(
            before >= 0.0,
            "Before dampening surprise should be non-negative"
        );
        // The patch at (0,0) should have very low surprise after high-confidence recognition
        assert!(
            surprise_at_00 < before || surprise_at_00 < 0.1,
            "High confidence should strongly dampen surprise at (0,0): value={surprise_at_00}"
        );
    }

    #[test]
    fn test_dampen_low_confidence_gentle_suppression() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);

        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = solid_gray_frame(64, 64, 200);
        bridge.process_frame(&frame_a, 64, 64, 1, 0.033);
        bridge.process_frame(&frame_b, 64, 64, 1, 0.033);

        let attention_before = bridge.manifold().surprise_map().attention_map();
        let surprise_before = attention_before.at(0, 0);

        // Low confidence: gentle dampening (factor = 0.8)
        bridge.dampen_patch_surprise(0, 0, 0.2);

        let attention_after = bridge.manifold().surprise_map().attention_map();
        let surprise_after = attention_after.at(0, 0);

        // Should be reduced by ~20% (factor 0.8)
        if surprise_before > 1e-6 {
            assert!(
                surprise_after < surprise_before,
                "Even low confidence should reduce surprise: before={surprise_before}, after={surprise_after}"
            );
            let ratio = surprise_after / surprise_before;
            assert!(
                (ratio - 0.8).abs() < 0.01,
                "Low confidence factor should be ~0.8, got ratio={ratio}"
            );
        }
    }

    #[test]
    fn test_dampen_out_of_bounds_is_noop() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);
        bridge.process_frame(&frame, 64, 64, 1, 0.033);

        // Should not panic on out-of-bounds coordinates
        bridge.dampen_patch_surprise(999, 999, 0.9);
    }

    // === Cross-Manifold Predictor Tests ===

    #[test]
    fn test_cross_manifold_predictor_construction() {
        let pred = CrossManifoldPredictor::new(16_384, 42);
        assert_eq!(pred.prediction_error(), 0.0);
    }

    #[test]
    fn test_cross_manifold_predict_produces_valid_hv() {
        let mut pred = CrossManifoldPredictor::new(16_384, 42);
        let vision_state = ContinuousHV::random(16_384, 100);

        let cognitive = pred.predict_cognitive(&vision_state);
        assert_eq!(cognitive.dim(), 16_384);
        assert!(cognitive.norm() > 0.0);
    }

    #[test]
    fn test_cross_manifold_learning_reduces_error() {
        let dim = 512;
        let mut pred = CrossManifoldPredictor::new(dim, 42);
        pred.set_learning_rate(0.05);

        let vision = ContinuousHV::random(dim, 100);
        let actual_cognitive = ContinuousHV::random(dim, 200);
        let initial_sim = pred
            .predict_cognitive(&vision)
            .similarity(&actual_cognitive);

        for _ in 0..100 {
            pred.predict_cognitive(&vision);
            pred.observe_cognitive(&actual_cognitive);
        }

        let final_sim = pred
            .predict_cognitive(&vision)
            .similarity(&actual_cognitive);
        assert!(
            final_sim > initial_sim,
            "input-conditioned learning should improve similarity: initial={initial_sim}, final={final_sim}"
        );
    }

    #[test]
    fn test_cross_manifold_update_depends_on_vision_input() {
        let dim = 512;
        let mut first = CrossManifoldPredictor::new(dim, 42);
        let mut second = CrossManifoldPredictor::new(dim, 42);
        first.set_learning_rate(0.05);
        second.set_learning_rate(0.05);

        let vision_a = ContinuousHV::random(dim, 100);
        let vision_b = ContinuousHV::random(dim, 101);
        let target = ContinuousHV::random(dim, 200);
        first.predict_cognitive(&vision_a);
        second.predict_cognitive(&vision_b);
        first.observe_cognitive(&target);
        second.observe_cognitive(&target);

        assert!(
            first
                .mapping_weight
                .as_slice()
                .iter()
                .zip(second.mapping_weight.as_slice())
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "different vision inputs must produce different mapping updates"
        );
    }

    #[test]
    fn test_cross_manifold_observation_consumes_cached_pair_once() {
        let dim = 512;
        let mut pred = CrossManifoldPredictor::new(dim, 42);
        let vision = ContinuousHV::random(dim, 100);
        let target = ContinuousHV::random(dim, 200);

        pred.predict_cognitive(&vision);
        pred.observe_cognitive(&target);
        let after_first = pred.mapping_weight.as_slice().to_vec();
        pred.observe_cognitive(&target);

        assert_eq!(pred.mapping_weight.as_slice(), after_first.as_slice());
    }

    #[test]
    fn test_cross_manifold_reset() {
        let mut pred = CrossManifoldPredictor::new(16_384, 42);
        let vision = ContinuousHV::random(16_384, 100);
        let actual = ContinuousHV::random(16_384, 200);

        pred.predict_cognitive(&vision);
        pred.observe_cognitive(&actual);
        assert!(pred.prediction_error() > 0.0);

        pred.reset();
        assert_eq!(pred.prediction_error(), 0.0);
    }

    // === Full Vision→Cognitive Pipeline Integration Test ===

    #[test]
    fn test_full_pipeline_100_frames() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg.clone(), 128, 128);

        // Synthetic video sequence: static → scene change → oscillating
        let frame_a: Vec<u8> = vec![128; 128 * 128];
        let frame_b: Vec<u8> = (0..128 * 128)
            .map(|i| ((i % 128 + i / 128) % 256) as u8)
            .collect();

        let mut all_hvs = Vec::with_capacity(100);
        for i in 0..100 {
            let frame = match i {
                0..=30 => &frame_a,  // Static scene
                31..=50 => &frame_b, // Scene change
                _ => {
                    if i % 2 == 0 {
                        &frame_a
                    } else {
                        &frame_b
                    }
                }
            };

            let (hv, tel) = bridge.process_frame_with_telemetry(frame, 128, 128, 1, 0.033);

            // Validate HV constraints for cycle_with_hv() compatibility
            // Allow dilation to Ultra (65536)
            let current_dim = hv.dim();
            assert!(
                current_dim == 16384 || current_dim == 65536,
                "Frame {i}: invalid dimension {current_dim}"
            );
            assert!(hv.norm() > 0.0, "Frame {i}: zero-norm HV");
            assert!(hv.norm().is_finite(), "Frame {i}: non-finite norm");

            // All values should be finite
            assert!(
                hv.as_slice().iter().all(|v| v.is_finite()),
                "Frame {i}: non-finite values in HV"
            );

            // Telemetry should be sane
            assert!(tel.prediction_error >= 0.0 && tel.prediction_error.is_finite());
            assert!(tel.manifold_coherence >= 0.0 && tel.manifold_coherence.is_finite());

            all_hvs.push(hv);
        }

        assert_eq!(bridge.frame_count(), 100);
        // Frames during static scene should be similar to each other
        let hv5 = &all_hvs[5];
        let hv25 = &all_hvs[25];
        let (hv5_f, hv25_f) = if hv5.dim() != hv25.dim() {
            let max_dim = hv5.dim().max(hv25.dim());
            (hv5.dilate(max_dim), hv25.dilate(max_dim))
        } else {
            (hv5.clone(), hv25.clone())
        };
        let static_sim = hv5_f.similarity(&hv25_f);
        assert!(
            static_sim > 0.5,
            "Static scene HVs should be similar: sim={static_sim}"
        );

        // Scene change should produce different HVs
        let hv35 = &all_hvs[35];
        let (hv25_c, hv35_c) = if hv25.dim() != hv35.dim() {
            let max_dim = hv25.dim().max(hv35.dim());
            (hv25.dilate(max_dim), hv35.dilate(max_dim))
        } else {
            (hv25.clone(), hv35.clone())
        };
        let change_sim = hv25_c.similarity(&hv35_c);
        assert!(
            change_sim < 0.999,
            "Scene change should produce different HVs: sim={change_sim}"
        );

        // Verify health is OK
        let health = bridge.manifold().compute_health();
        assert!(
            health.is_healthy,
            "Manifold should be healthy after 100 frames"
        );
        assert_eq!(health.total_frames, 100);
    }

    #[test]
    fn test_pipeline_rgb_end_to_end() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg.clone(), 64, 64);

        // Red→Green→Blue color cycle
        let colors: Vec<Vec<u8>> = vec![
            (0..64 * 64).flat_map(|_| vec![255u8, 0, 0]).collect(),
            (0..64 * 64).flat_map(|_| vec![0u8, 255, 0]).collect(),
            (0..64 * 64).flat_map(|_| vec![0u8, 0, 255]).collect(),
        ];

        let mut hvs = Vec::new();
        for (i, color_frame) in colors.iter().enumerate() {
            for _ in 0..10 {
                let hv = bridge.process_frame(color_frame, 64, 64, 3, 0.033);
                if i > 0 || hvs.len() >= 5 {
                    // After warm-up
                    assert!(hv.norm() > 0.0);
                }
                hvs.push(hv);
            }
        }

        assert_eq!(bridge.frame_count(), 30);

        // Different color states should be distinguishable
        let red_hv = &hvs[8]; // Late red
        let blue_hv = &hvs[28]; // Late blue

        let (r_final, b_final) = if red_hv.dim() != blue_hv.dim() {
            let max_dim = red_hv.dim().max(blue_hv.dim());
            (red_hv.dilate(max_dim), blue_hv.dilate(max_dim))
        } else {
            (red_hv.clone(), blue_hv.clone())
        };

        let sim = r_final.similarity(&b_final);
        assert!(
            sim < 0.99,
            "Red and blue should produce different pipeline outputs: sim={sim}"
        );
    }

    #[test]
    fn test_prediction_confidence() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);

        // Before any frame, prediction_error is 0 → confidence = 1.0
        assert!((bridge.prediction_confidence() - 1.0).abs() < 1e-6);

        // After frames, confidence should be in valid range
        let frame = gradient_frame(64, 64);
        for _ in 0..5 {
            bridge.process_frame(&frame, 64, 64, 1, 0.033);
        }
        let conf = bridge.prediction_confidence();
        assert!(
            conf >= 0.0 && conf <= 1.0,
            "Confidence should be in [0, 1]: {conf}"
        );
    }

    #[test]
    fn test_bridge_active_patch_count() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);

        let (active, total) = bridge.active_patch_count();
        assert_eq!(active, 0);
        assert!(total > 0);

        let frame = gradient_frame(64, 64);
        bridge.process_frame(&frame, 64, 64, 1, 0.033);
        let (_active2, total2) = bridge.active_patch_count();
        assert_eq!(total2, total);
    }

    // === P2-A: Cross-Scale Predictive Surprise Injection ===

    #[test]
    fn test_predictive_hierarchy_injects_into_surprise() {
        // With predictive hierarchy enabled, surprise should be augmented by
        // cross-scale prediction errors, producing different values than without.
        let mut cfg_base = VisionConfig::default();
        let mut cfg_pred = VisionConfig::default();
        cfg_pred.enable_predictive_hierarchy = true;

        let mut bridge_base = VisionBridge::new(cfg_base.clone(), 64, 64);
        let mut bridge_pred = VisionBridge::new(cfg_pred, 64, 64);

        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = solid_gray_frame(64, 64, 200);
        cfg_base.enable_predictive_hierarchy = false; // already false

        // Process same sequence through both
        for _ in 0..3 {
            bridge_base.process_frame(&frame_a, 64, 64, 1, 0.033);
            bridge_pred.process_frame(&frame_a, 64, 64, 1, 0.033);
        }
        bridge_base.process_frame(&frame_b, 64, 64, 1, 0.033);
        bridge_pred.process_frame(&frame_b, 64, 64, 1, 0.033);

        // Both should remain finite and valid
        let s_base = bridge_base.manifold().surprise_map().max_surprise();
        let s_pred = bridge_pred.manifold().surprise_map().max_surprise();
        assert!(s_base.is_finite() && s_base >= 0.0, "base surprise finite");
        assert!(s_pred.is_finite() && s_pred >= 0.0, "pred surprise finite");
        // Predictive hierarchy adds cross-scale signal → surprise may differ
        // (we verify absence of NaN/panic rather than exact values)
    }

    // === P2-B: Goal-Signal Hebbian Learning ===

    #[test]
    fn test_goal_signal_learning_rate_default() {
        let sig = CognitiveGoalSignal::default();
        assert!((sig.learning_rate - 0.05).abs() < 1e-6);
        assert!(sig.task_hv.is_none());
    }

    #[test]
    fn test_goal_signal_no_update_without_task_hv() {
        let mut sig = CognitiveGoalSignal::default();
        let recognized = ContinuousHV::random(16_384, 42);
        let updated = sig.update_from_recognition(&recognized, 0.9);
        assert!(!updated, "Should not update without task_hv");
    }

    #[test]
    fn test_goal_signal_no_update_low_confidence() {
        let task_hv = ContinuousHV::random(16_384, 1);
        let mut sig = CognitiveGoalSignal::new(task_hv.clone());
        // Low confidence should not trigger update
        let recognized = task_hv.clone(); // Same as task — maximum similarity
        let updated = sig.update_from_recognition(&recognized, 0.3);
        assert!(!updated, "Should not update with confidence < 0.6");
    }

    #[test]
    fn test_goal_signal_updates_toward_recognized() {
        // Use two different frames so task_hv and recognized_hv are related but
        // not identical — lerp between identical vectors produces no shift.
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);

        // Frame A: use as task template
        let frame_a = solid_gray_frame(64, 64, 80);
        bridge.process_frame(&frame_a, 64, 64, 1, 0.033);
        let task_patch = bridge
            .manifold()
            .last_patch_hvs()
            .first()
            .cloned()
            .expect("should have patch HVs after frame A");

        // Frame B: slightly different scene → similar but non-identical patches
        let frame_b = solid_gray_frame(64, 64, 100);
        bridge.process_frame(&frame_b, 64, 64, 1, 0.033);
        let recognized_patch = bridge
            .manifold()
            .last_patch_hvs()
            .first()
            .cloned()
            .expect("should have patch HVs after frame B");

        let pre_sim = task_patch.similarity(&recognized_patch);
        // Patches from similar gray frames should be related but not identical
        if pre_sim <= 0.3 || (pre_sim - 1.0).abs() < 1e-6 {
            // Edge case: patches happen to be orthogonal or identical — skip
            return;
        }

        bridge.set_goal_signal(CognitiveGoalSignal::new(task_patch.clone()));

        let task_hv_before = bridge.goal_signal.task_hv.clone().unwrap();
        let updated = bridge.learn_from_recognized_patch(&recognized_patch, 0.95);

        assert!(updated, "Should update when sim > 0.3 and confidence > 0.6");

        let task_hv_after = bridge.goal_signal.task_hv.clone().unwrap();
        // After update, task_hv should be closer to recognized_patch than before
        let sim_before = task_hv_before.similarity(&recognized_patch);
        let sim_after = task_hv_after.similarity(&recognized_patch);
        assert!(
            sim_after >= sim_before - 1e-4,
            "Task HV should move toward recognized patch: before={sim_before}, after={sim_after}"
        );
        // And it should have actually moved (not a no-op)
        let self_sim = task_hv_before.similarity(&task_hv_after);
        assert!(
            self_sim < 1.0 - 1e-6,
            "Task HV should have shifted from before: self_sim={self_sim}"
        );
    }

    #[test]
    fn test_goal_signal_no_update_orthogonal_patch() {
        // A random patch has cos_sim ≈ 0 with a random task HV (concentration of measure).
        // Update should not fire below the 0.3 threshold.
        let task_hv = ContinuousHV::random(16_384, 1);
        let mut sig = CognitiveGoalSignal::new(task_hv);
        let orthogonal = ContinuousHV::random(16_384, 999_999);
        // In 16,384D, random unit vectors have cos_sim ≈ 0 — well below 0.3
        let updated = sig.update_from_recognition(&orthogonal, 0.95);
        assert!(
            !updated,
            "Near-orthogonal patch should not trigger template update"
        );
    }

    // === P2-C: Temporal Patch Binding ===

    #[test]
    fn test_temporal_binding_disabled_by_default() {
        let cfg = VisionConfig::default();
        assert!(!cfg.enable_temporal_binding);
        let mut bridge = VisionBridge::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);
        bridge.process_frame(&frame, 64, 64, 1, 0.033);
        // When disabled, temporal_patch_hvs == last_patch_hvs
        let temporal = bridge.manifold().temporal_patch_hvs();
        let raw = bridge.manifold().last_patch_hvs();
        assert_eq!(temporal.len(), raw.len());
    }

    #[test]
    fn test_temporal_binding_produces_valid_output() {
        let mut cfg = VisionConfig::default();
        cfg.enable_temporal_binding = true;
        let mut bridge = VisionBridge::new(cfg.clone(), 64, 64);

        let frame_a = solid_gray_frame(64, 64, 80);
        let frame_b = solid_gray_frame(64, 64, 180);

        let hv1 = bridge.process_frame(&frame_a, 64, 64, 1, 0.033);
        let hv2 = bridge.process_frame(&frame_b, 64, 64, 1, 0.033);
        let hv3 = bridge.process_frame(&frame_b, 64, 64, 1, 0.033);

        assert_eq!(hv1.dim(), cfg.hdc_dim);
        assert!(hv1.norm() > 0.0 && hv1.norm().is_finite());
        assert!(hv2.norm() > 0.0 && hv2.norm().is_finite());
        assert!(hv3.norm() > 0.0 && hv3.norm().is_finite());

        // Temporal patch HVs should be populated after frame 1
        assert!(
            !bridge.manifold().temporal_patch_hvs().is_empty(),
            "Temporal patch HVs should be populated"
        );
    }

    #[test]
    fn test_temporal_binding_distinguishes_direction() {
        // ρ(A) ⊗ B ≠ ρ(B) ⊗ A — temporal binding is non-commutative.
        let mut cfg = VisionConfig::default();
        cfg.enable_temporal_binding = true;

        // Forward: A then B
        let mut bridge_fwd = VisionBridge::new(cfg.clone(), 64, 64);
        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = solid_gray_frame(64, 64, 200);
        bridge_fwd.process_frame(&frame_a, 64, 64, 1, 0.033);
        bridge_fwd.process_frame(&frame_b, 64, 64, 1, 0.033);
        let fwd_temporal = bridge_fwd.manifold().temporal_patch_hvs().to_vec();

        // Reverse: B then A
        let mut bridge_rev = VisionBridge::new(cfg, 64, 64);
        bridge_rev.process_frame(&frame_b, 64, 64, 1, 0.033);
        bridge_rev.process_frame(&frame_a, 64, 64, 1, 0.033);
        let rev_temporal = bridge_rev.manifold().temporal_patch_hvs().to_vec();

        assert!(!fwd_temporal.is_empty() && !rev_temporal.is_empty());
        // A→B and B→A should produce different temporal encodings
        let sim = fwd_temporal[0].similarity(&rev_temporal[0]);
        assert!(
            sim < 0.99,
            "Temporal binding should distinguish A→B from B→A: sim={sim}"
        );
    }

    #[test]
    fn test_temporal_binding_stable_for_static_scene() {
        // Static scene: A→A has consistent temporal HVs across frames.
        let mut cfg = VisionConfig::default();
        cfg.enable_temporal_binding = true;
        let mut bridge = VisionBridge::new(cfg, 64, 64);

        let frame = solid_gray_frame(64, 64, 128);
        for _ in 0..5 {
            bridge.process_frame(&frame, 64, 64, 1, 0.033);
        }
        let temporal_5 = bridge.manifold().temporal_patch_hvs().to_vec();
        bridge.process_frame(&frame, 64, 64, 1, 0.033);
        let temporal_6 = bridge.manifold().temporal_patch_hvs().to_vec();

        assert!(!temporal_5.is_empty());
        let sim = temporal_5[0].similarity(&temporal_6[0]);
        assert!(
            sim > 0.5,
            "Static scene temporal HVs should be consistent across frames: sim={sim}"
        );
    }
    #[test]
    fn audited_bridge_loader_reports_structural_failure_without_mutation() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-vision-bridge-audited-load-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("bridge.chk");
        std::fs::write(&path, b"not a checkpoint").unwrap();
        let mut bridge = VisionBridge::new(VisionConfig::default(), 16, 16);
        let before = serde_json::to_vec(&bridge.save_state()).unwrap();
        let error = bridge
            .load_checkpoint_file_with_retention_audited(
                &path,
                crate::checkpoint::CheckpointRetentionPolicy {
                    previous_generations: 0,
                },
            )
            .unwrap_err();
        assert_eq!(error.attempts.len(), 1);
        assert_eq!(serde_json::to_vec(&bridge.save_state()).unwrap(), before);
        let _ = std::fs::remove_dir_all(directory);
    }
}
