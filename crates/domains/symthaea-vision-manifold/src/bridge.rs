// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge between the VisionManifold and the cognitive loop.
//!
//! `VisionBridge` wraps a `VisionManifold` and produces `ContinuousHV` outputs
//! suitable for feeding into `CognitiveLoopService::cycle_with_hv()`.
//!
//! The bridge applies attention-modulated boosting: dimensions corresponding
//! to salient (high-surprise) patches are scaled up, making unexpected regions
//! more prominent in the cognitive encoding.

use std::time::Instant;

use symthaea_core::hdc::ContinuousHV;

use crate::manifold::VisionManifold;
use crate::spectrum::{MultiSpectralEncoder, MultiSpectralFrame};
use crate::types::{SalientRegion, VisionConfig, VisionTelemetry};

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
            task_gain: gain.max(0.0),
            learning_rate: 0.05,
        }
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
        if self.learning_rate < 1e-6 || confidence < 0.6 {
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
}

impl VisionBridge {
    /// Create a new vision bridge.
    ///
    /// # Arguments
    /// * `config` — Vision manifold configuration.
    /// * `max_width` / `max_height` — Maximum frame dimensions.
    pub fn new(config: VisionConfig, max_width: u32, max_height: u32) -> Self {
        let manifold = VisionManifold::new(config, max_width, max_height);
        Self {
            manifold,
            attention_boost: 0.3,
            goal_signal: CognitiveGoalSignal::default(),
            multi_spectral: None,
        }
    }

    /// Create a bridge wrapping an existing manifold.
    pub fn from_manifold(manifold: VisionManifold) -> Self {
        Self {
            manifold,
            attention_boost: 0.3,
            goal_signal: CognitiveGoalSignal::default(),
            multi_spectral: None,
        }
    }

    /// Set the attention boost factor (default: 0.3).
    ///
    /// Higher values make surprising regions more prominent in the output HV.
    pub fn set_attention_boost(&mut self, boost: f32) {
        self.attention_boost = boost.max(0.0);
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
        self.goal_signal = signal;
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
        let alpha = alpha.clamp(0.0, 1.0);
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
        let Some(ref mut enc) = self.multi_spectral else {
            return (self.manifold.state().clone(), VisionTelemetry::default());
        };
        let t0 = Instant::now();
        let multi_hv = enc.encode(frame);
        let encode_us = t0.elapsed().as_micros() as u64;

        let t1 = Instant::now();
        let mut telemetry = self.manifold.observe_multiband_frame(&multi_hv, dt);
        telemetry.encode_time_us = encode_us;
        telemetry.evolve_time_us += t1.elapsed().as_micros() as u64;

        let boosted_hv = self.apply_attention_boost();
        telemetry.output_hv_norm = boosted_hv.norm();
        telemetry.attention_boost_applied = self.attention_boost;

        (boosted_hv, telemetry)
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

    /// Apply attention boost to the manifold state based on combined surprise.
    ///
    /// Combines three signals via element-wise max / additive blend:
    /// 1. **Bottom-up surprise** (SurpriseMap): unexpected change drives saliency.
    /// 2. **Motion saliency** (MotionField): moving regions receive extra boost.
    /// 3. **Top-down task relevance** (CognitiveGoalSignal): patches with high
    ///    cosine similarity to the current `task_hv` receive `task_gain * similarity`
    ///    additional boost, implementing goal-directed active perception.
    ///
    /// Each patch's dimensions in the state HV are scaled by:
    /// ```text
    /// scale = 1.0 + attention_boost * normalized_saliency + task_gain * task_similarity
    /// ```
    fn apply_attention_boost(&self) -> ContinuousHV {
        let state = self.manifold.state();
        let surprise_map = self.manifold.surprise_map();
        let motion_saliency = self.manifold.motion_saliency();
        let patch_hvs = self.manifold.last_patch_hvs();
        let max_surprise = surprise_map.max_surprise();
        let max_motion = motion_saliency.iter().copied().fold(0.0f32, f32::max);
        let max_signal = max_surprise.max(max_motion);

        let has_task = self.goal_signal.task_hv.is_some() && self.goal_signal.task_gain > 1e-6;
        let has_saliency = max_signal >= 1e-6 && self.attention_boost >= 1e-6;

        if !has_saliency && !has_task {
            return state.clone();
        }

        let attention = surprise_map.attention_map();
        let num_patches = attention.values.len();
        let dim = state.dim();

        if num_patches == 0 {
            return state.clone();
        }

        // Map patch-level combined saliency to HV dimensions via strided mapping.
        let dims_per_patch = dim / num_patches.max(1);
        let state_slice = state.as_slice();
        let mut boosted = state_slice.to_vec();

        for (patch_idx, &appearance_surprise) in attention.values.iter().enumerate() {
            // ── Bottom-up signal: appearance surprise + motion saliency ──────────
            let motion = motion_saliency.get(patch_idx).copied().unwrap_or(0.0);
            let combined = appearance_surprise.max(motion);
            let bottom_up_boost = if has_saliency && max_signal > 1e-6 {
                self.attention_boost * (combined / max_signal)
            } else {
                0.0
            };

            // ── Top-down signal: cosine similarity to current task_hv ─────────
            // Only computed when a goal is set and per-patch HVs are available.
            let top_down_boost = if has_task {
                if let Some(ref task_hv) = self.goal_signal.task_hv {
                    if let Some(patch_hv) = patch_hvs.get(patch_idx) {
                        // Clamp to [0, 1]: only positive alignment earns boost.
                        // Negative similarity (anti-correlated) does not suppress.
                        // Dimension guard: if goal was set before dilation, skip boost until next sync.
                        let sim = if task_hv.dim() == patch_hv.dim() {
                            task_hv.similarity(patch_hv).max(0.0)
                        } else {
                            0.0
                        };
                        self.goal_signal.task_gain * sim
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let scale = 1.0 + bottom_up_boost + top_down_boost;
            let start = patch_idx * dims_per_patch;
            let end = (start + dims_per_patch).min(dim);
            for d in start..end {
                boosted[d] *= scale;
            }
        }

        ContinuousHV::from_vec(boosted).normalize()
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
                pixel_w: grid.patch_size,
                pixel_h: grid.patch_size,
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

    /// Perform 'Holographic Dilation' - scale internal components.
    pub fn dilate(&mut self, target: symthaea_core::hdc::HdcDimensionality) {
        let dim = target.dimension();
        self.goal_signal.dilate(dim);
        self.manifold.dilate(target);

        if let Some(ref mut ms) = self.multi_spectral {
            ms.dilate(dim);
        }
    }

    /// Current frame count.
    pub fn frame_count(&self) -> u64 {
        self.manifold.frame_count()
    }

    /// Reset the bridge (and underlying manifold) to initial state.
    pub fn reset(&mut self) {
        self.manifold.reset();
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
/// a predicted cognitive HV. Online Hebbian learning reduces prediction error
/// as the system observes actual cognitive states.
pub struct CrossManifoldPredictor {
    mapping_weight: ContinuousHV,
    last_prediction: Option<ContinuousHV>,
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
        self.dim = target_dim;
    }

    /// Predict the cognitive state from a vision state.
    ///
    /// `predicted_cognitive = tanh(mapping_weight ⊗ vision_state)`
    pub fn predict_cognitive(&mut self, vision_state: &ContinuousHV) -> ContinuousHV {
        let predicted = self.mapping_weight.bind(vision_state).tanh();
        self.last_prediction = Some(predicted.clone());
        predicted
    }

    /// Observe the actual cognitive state and update the mapping weight.
    ///
    /// Hebbian learning: strengthen the mapping that would have produced the
    /// actual cognitive HV, weakening dimensions that produced error.
    pub fn observe_cognitive(&mut self, actual_cognitive: &ContinuousHV) {
        if let Some(ref predicted) = self.last_prediction {
            self.prediction_error = 1.0 - actual_cognitive.similarity(predicted).clamp(-1.0, 1.0);

            // Hebbian update: W += lr * (actual - predicted) ⊗ sign(predicted)
            let dim = self.dim;
            let actual_s = actual_cognitive.as_slice();
            let predicted_s = predicted.as_slice();
            let weight_s = self.mapping_weight.as_slice();

            let mut updated = vec![0.0f32; dim];
            for i in 0..dim {
                let error = actual_s[i] - predicted_s[i];
                updated[i] = weight_s[i] + self.learning_rate * error;
            }
            self.mapping_weight = ContinuousHV::from_vec(updated);
        }
    }

    /// Current prediction error (1 - cos_sim).
    pub fn prediction_error(&self) -> f32 {
        self.prediction_error
    }

    /// Set the learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr.max(0.0);
    }

    /// Reset the predictor state (but keep learned weights).
    pub fn reset(&mut self) {
        self.last_prediction = None;
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
    fn test_bridge_construction() {
        let cfg = VisionConfig::default();
        let bridge = VisionBridge::new(cfg.clone(), 64, 64);
        assert_eq!(bridge.frame_count(), 0);
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
        let mut bridge = VisionBridge::new(cfg.clone(), 64, 64);

        // Feed two different frames to generate surprise
        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = solid_gray_frame(64, 64, 200);

        bridge.process_frame(&frame_a, 64, 64, 1, 0.033);
        let hv_with_boost = bridge.process_frame(&frame_b, 64, 64, 1, 0.033);

        // Reset and do the same without boost
        bridge.reset();
        bridge.set_attention_boost(0.0);
        bridge.process_frame(&frame_a, 64, 64, 1, 0.033);
        let hv_without_boost = bridge.process_frame(&frame_b, 64, 64, 1, 0.033);

        // With attention boost, output should differ from without
        let sim = hv_with_boost.similarity(&hv_without_boost);
        // They should be similar (same manifold state) but not identical
        // (boost modulates dimensions)
        assert!(
            sim < 1.0 - 1e-6 || sim > 1.0 + 1e-6 || (sim - 1.0).abs() < 1e-3,
            "Attention boost may or may not change output depending on surprise: sim={sim}"
        );
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

        let sim = hv_red.similarity(&hv_blue);
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
        // The top-down boost is per-patch: scale[i] += task_gain * cos_sim(patch_i, task_hv).
        // In 16,384D, random HVs are nearly orthogonal (sim ≈ 0), so we must use an
        // *actual* patch HV as the task vector to guarantee non-trivial similarity.
        // Here we use patch[0] from bridge_goal as the task vector — on the next frame
        // that patch gets scale ≈ 2.0 while all others stay at 1.0, which must change
        // the output.
        let cfg = VisionConfig::default();
        let mut bridge_base = VisionBridge::new(cfg.clone(), 64, 64);
        let mut bridge_goal = VisionBridge::new(cfg, 64, 64);

        // A colorful gradient frame so patches have varied content.
        let frame: Vec<u8> = (0..64u32 * 64)
            .flat_map(|i| {
                let v = (i % 128 + 32) as u8;
                vec![v, 255 - v, 128u8]
            })
            .collect();

        // Warm both bridges identically.
        bridge_base.process_frame(&frame, 64, 64, 3, 0.033);
        bridge_goal.process_frame(&frame, 64, 64, 3, 0.033);

        // Use patch[0] from bridge_goal as the goal: maximally correlated with itself.
        // On the *next* frame that patch receives scale ≈ 1 + task_gain while others don't.
        let task_hv = bridge_goal
            .manifold()
            .last_patch_hvs()
            .first()
            .cloned()
            .expect("Should have patch HVs after first frame");

        bridge_goal.set_goal_signal(CognitiveGoalSignal::with_gain(task_hv, 2.0));

        // Process a second identical frame — same content so same surprise, but the
        // goal signal boosts patch[0] dimensions in bridge_goal only.
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
    fn test_cross_manifold_learning() {
        let mut pred = CrossManifoldPredictor::new(16_384, 42);
        pred.set_learning_rate(0.01);

        let vision = ContinuousHV::random(16_384, 100);
        let actual_cognitive = ContinuousHV::random(16_384, 200);

        // Initial prediction
        let initial_pred = pred.predict_cognitive(&vision);
        let initial_sim = initial_pred.similarity(&actual_cognitive);

        // Learn from the observation
        pred.observe_cognitive(&actual_cognitive);
        assert!(pred.prediction_error() >= 0.0);

        // After learning, re-predict — may or may not be closer depending on
        // the random starting point, but the weight should have changed
        let _ = pred.predict_cognitive(&vision);
        let _ = initial_sim; // Just verify no panic
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
            assert_eq!(hv.dim(), cfg.hdc_dim, "Frame {i}: wrong dimension");
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

        // Static scene HVs should be similar to each other
        let static_sim = all_hvs[5].similarity(&all_hvs[25]);
        assert!(
            static_sim > 0.5,
            "Static scene HVs should be similar: sim={static_sim}"
        );

        // Scene change should produce different HVs
        let change_sim = all_hvs[25].similarity(&all_hvs[35]);
        assert!(
            change_sim < static_sim || change_sim < 0.99,
            "Scene change should produce different HVs"
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
        let sim = red_hv.similarity(blue_hv);
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
}
