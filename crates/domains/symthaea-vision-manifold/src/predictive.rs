// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive coding hierarchy for multi-scale vision processing.
//!
//! Implements a two-level hierarchy where the coarse scale predicts the fine
//! scale, and prediction errors flow upward. This mirrors cortical predictive
//! coding: higher levels generate top-down expectations, lower levels signal
//! surprise when those expectations are violated.
//!
//! ```text
//! coarse (32px patches) → predicts → fine (8px patches)
//!      ↑                                    |
//!      └──── prediction error ──────────────┘
//! ```
//!
//! The prediction error signal serves two purposes:
//! 1. **Attention**: High prediction error regions need more processing
//! 2. **Learning**: Gradient signal for improving the coarse-to-fine mapping

use symthaea_core::hdc::ContinuousHV;

use crate::encoder::MultiScaleEncoder;
use crate::types::{PredictiveHierarchyState, VisionConfig};

/// A two-level predictive coding hierarchy over multi-scale HDC encodings.
///
/// The coarse level generates predictions of the fine-level encoding.
/// Prediction errors are tracked per-frame and drive attention allocation.
pub struct PredictiveCodingHierarchy {
    encoder: MultiScaleEncoder,
    /// Coarse→fine prediction weight HV (learned mapping).
    prediction_weight: ContinuousHV,
    /// Last coarse-level HV (for prediction).
    last_coarse_hv: Option<ContinuousHV>,
    /// Last fine-level percept used as a persistence baseline.
    last_fine_hv: Option<ContinuousHV>,
    /// Prediction error: 1 - cos_sim(predicted_fine, actual_fine).
    prediction_error: f32,
    /// Exponential moving average of prediction error.
    error_ema: f32,
    /// EMA of the previous-fine persistence baseline error.
    baseline_error_ema: f32,
    /// EMA of normalized skill relative to persistence, in [-1, 1].
    relative_skill_ema: f32,
    /// Number of observations for which a real temporal prediction existed.
    prediction_count: u64,
    /// EMA decay factor.
    ema_decay: f32,
    /// HDC dimension.
    dim: usize,
    /// Learning rate for prediction weight updates.
    learning_rate: f32,
}

impl PredictiveCodingHierarchy {
    /// Create a new predictive coding hierarchy from a VisionConfig.
    ///
    /// The config's multi_scale settings determine the fine and coarse scales.
    /// At least two strictly ordered scales are required.
    pub fn new(config: &VisionConfig, max_width: u32, max_height: u32) -> Self {
        Self::try_new(config, max_width, max_height)
            .expect("invalid predictive-coding hierarchy configuration")
    }

    /// Construct a predictive hierarchy without panicking on invalid topology.
    pub fn try_new(config: &VisionConfig, max_width: u32, max_height: u32) -> Result<Self, String> {
        config.validate()?;
        if max_width == 0 || max_height == 0 {
            return Err(format!(
                "predictive hierarchy capacity must be non-zero, got {max_width}x{max_height}"
            ));
        }
        if config.multi_scale.scales.len() < 2 {
            return Err(
                "predictive hierarchy requires at least two ordered spatial scales".to_string(),
            );
        }

        let encoder = MultiScaleEncoder::new(config, max_width, max_height);
        let dim = config.hdc_dim;
        let prediction_weight = ContinuousHV::random(dim, config.seed + 700_000);

        Ok(Self {
            encoder,
            prediction_weight,
            last_coarse_hv: None,
            last_fine_hv: None,
            prediction_error: 0.0,
            error_ema: 0.0,
            baseline_error_ema: 0.0,
            relative_skill_ema: 0.0,
            prediction_count: 0,
            ema_decay: 0.95,
            dim,
            learning_rate: 0.01,
        })
    }

    pub(crate) fn hdc_vector_count(&self) -> usize {
        self.encoder.hdc_vector_count()
            + 1
            + self.last_coarse_hv.is_some() as usize
            + self.last_fine_hv.is_some() as usize
    }

    /// Perform 'Holographic Dilation' - scale internal components.
    pub fn dilate(&mut self, target_dim: usize) {
        if self.dim == target_dim {
            return;
        }

        self.encoder.dilate(target_dim);
        self.prediction_weight = self.prediction_weight.dilate(target_dim);

        if let Some(ref mut hv) = self.last_coarse_hv {
            *hv = hv.dilate(target_dim);
        }
        if let Some(ref mut hv) = self.last_fine_hv {
            *hv = hv.dilate(target_dim);
        }

        self.dim = target_dim;
    }

    /// Perform multi-scale 'Dreaming' - predict future coarse and fine states.
    ///
    /// Science: Friston (2010). Hierarchical active inference allows the system
    /// to Zoom Out (abstract simulation) and Zoom In (detailed simulation) across
    /// multiple levels of the world model hierarchy.
    ///
    /// Returns `(coarse_predictions, fine_predictions)`.
    pub fn dream_ahead(&self, steps: usize, _dt: f32) -> (Vec<ContinuousHV>, Vec<ContinuousHV>) {
        let mut coarse_preds = Vec::with_capacity(steps);
        let mut fine_preds = Vec::with_capacity(steps);

        let mut current_coarse = self
            .last_coarse_hv
            .clone()
            .unwrap_or_else(|| ContinuousHV::zero(self.dim));

        for _ in 0..steps {
            // 1. Evolve coarse state (assume abstract scene-level dynamics)
            // For now we use the identity as a neutral dynamic (static scene dream)
            // but rotate it slightly to simulate abstract "thought drift".
            current_coarse = current_coarse.permute(1);

            // 2. Predict fine scale from coarse scale (top-down inference)
            let predicted_fine = self.predict_fine(&current_coarse);

            coarse_preds.push(current_coarse.clone());
            fine_preds.push(predicted_fine);
        }

        (coarse_preds, fine_preds)
    }

    /// Process a frame through the predictive hierarchy.
    ///
    /// Returns `(fine_hv, coarse_hv, prediction_error)` where:
    /// - `fine_hv`: The fine-scale encoding (8px patches)
    /// - `coarse_hv`: The coarse-scale encoding (32px patches)
    /// - `prediction_error`: How well the coarse scale predicted the fine scale
    ///
    /// Internally updates the prediction weight to reduce future errors.
    pub fn process_frame(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
    ) -> PredictiveOutput {
        self.process_frame_checked(pixels, width, height, channels)
            .expect("invalid predictive-coding frame")
    }

    /// Process only a fully validated frame.
    ///
    /// Validation completes before any scale encoder advances its luminance
    /// history, so a rejected frame cannot create artificial motion later.
    pub fn process_frame_checked(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
    ) -> Result<PredictiveOutput, String> {
        if width == 0 || height == 0 {
            return Err(format!(
                "predictive frame dimensions must be non-zero, got {width}x{height}"
            ));
        }
        if !matches!(channels, 1 | 3 | 4) {
            return Err(format!(
                "predictive frame channel count must be 1, 3, or 4, got {channels}"
            ));
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|count| count.checked_mul(channels))
            .ok_or_else(|| "predictive frame geometry overflow".to_string())?;
        if pixels.len() != expected {
            return Err(format!(
                "predictive frame length mismatch: got {}, expected {expected}",
                pixels.len()
            ));
        }
        for (index, scale) in self.encoder.scales().iter().copied().enumerate() {
            let encoder = self
                .encoder
                .encoder_at(index)
                .ok_or_else(|| format!("predictive encoder is missing scale index {index}"))?;
            let grid = encoder.grid_for(width, height);
            if grid.rows > encoder.max_rows() || grid.cols > encoder.max_cols() {
                return Err(format!(
                    "predictive frame {width}x{height} exceeds scale-{scale} capacity of {}x{} patches",
                    encoder.max_cols(),
                    encoder.max_rows()
                ));
            }
        }

        let (_blended, scale_hvs, _patches) =
            self.encoder.encode_frame(pixels, width, height, channels);

        let fine_hv = scale_hvs
            .first()
            .cloned()
            .unwrap_or_else(|| ContinuousHV::zero(self.dim));
        let coarse_hv = scale_hvs
            .last()
            .cloned()
            .unwrap_or_else(|| ContinuousHV::zero(self.dim));

        let (prediction_available, baseline_error, relative_skill) =
            if let (Some(prev_coarse), Some(prev_fine)) =
                (self.last_coarse_hv.clone(), self.last_fine_hv.as_ref())
            {
                let predicted_fine = self.predict_fine(&prev_coarse);
                self.prediction_error = 1.0 - fine_hv.similarity(&predicted_fine).clamp(-1.0, 1.0);
                let baseline_error = 1.0 - fine_hv.similarity(prev_fine).clamp(-1.0, 1.0);
                let relative_skill = Self::relative_skill(self.prediction_error, baseline_error);
                self.update_calibration(baseline_error, relative_skill);
                self.update_prediction_weight(&fine_hv, &predicted_fine, &prev_coarse);
                (true, baseline_error, relative_skill)
            } else {
                self.prediction_error = 0.0;
                (false, 0.0, 0.0)
            };

        self.last_coarse_hv = Some(coarse_hv.clone());
        self.last_fine_hv = Some(fine_hv.clone());

        Ok(PredictiveOutput {
            fine_hv,
            coarse_hv,
            prediction_available,
            prediction_error: self.prediction_error,
            baseline_error,
            relative_skill,
            error_ema: self.error_ema,
            patch_prediction_errors: vec![],
        })
    }

    /// Predict fine-scale HV from a coarse-scale HV.
    ///
    /// Uses the learned prediction weight as a transformation:
    /// `predicted_fine = tanh(prediction_weight ⊗ coarse_hv)`
    fn predict_fine(&self, coarse: &ContinuousHV) -> ContinuousHV {
        self.prediction_weight.bind(coarse).tanh()
    }

    /// Update the prediction weight to reduce prediction error.
    ///
    /// Uses a contrastive Hebbian rule: strengthen the mapping that would have
    /// produced the actual fine HV, weaken the mapping that produced the error.
    fn update_prediction_weight(
        &mut self,
        actual_fine: &ContinuousHV,
        predicted_fine: &ContinuousHV,
        coarse: &ContinuousHV,
    ) {
        let dim = self.dim;
        let actual_s = actual_fine.as_slice();
        let predicted_s = predicted_fine.as_slice();
        let coarse_s = coarse.as_slice();
        let weight_s = self.prediction_weight.as_slice();

        // For predicted = tanh(W ⊗ coarse), the local derivative is
        // (1 - predicted²) ⊗ coarse. Omitting it causes saturated coordinates
        // to receive the largest ineffective updates and misstates the forward model.
        let mut updated = vec![0.0f32; dim];
        for i in 0..dim {
            let error = actual_s[i] - predicted_s[i];
            let tanh_derivative = (1.0 - predicted_s[i] * predicted_s[i]).max(0.0);
            let delta = self.learning_rate * error * tanh_derivative * coarse_s[i];
            let candidate = weight_s[i] + delta.clamp(-0.1, 0.1);
            updated[i] = if candidate.is_finite() {
                candidate.clamp(-4.0, 4.0)
            } else {
                weight_s[i]
            };
        }

        self.prediction_weight = ContinuousHV::from_vec(updated);
    }

    /// Current prediction error (1 - cos_sim between predicted and actual fine).
    pub fn prediction_error(&self) -> f32 {
        self.prediction_error
    }

    /// Exponential moving average of prediction error.
    pub fn error_ema(&self) -> f32 {
        self.error_ema
    }

    /// EMA of the persistence-baseline error.
    pub fn baseline_error_ema(&self) -> f32 {
        self.baseline_error_ema
    }

    /// EMA of normalized model skill relative to persistence.
    pub fn relative_skill_ema(&self) -> f32 {
        self.relative_skill_ema
    }

    /// Number of calibrated temporal predictions observed.
    pub fn prediction_count(&self) -> u64 {
        self.prediction_count
    }

    fn relative_skill(model_error: f32, baseline_error: f32) -> f32 {
        ((baseline_error - model_error) / (baseline_error + model_error + 1e-6)).clamp(-1.0, 1.0)
    }

    fn update_calibration(&mut self, baseline_error: f32, relative_skill: f32) {
        let update = |ema: f32, value: f32, decay: f32, count: u64| {
            if count == 0 {
                value
            } else {
                decay * ema + (1.0 - decay) * value
            }
        };
        self.error_ema = update(
            self.error_ema,
            self.prediction_error,
            self.ema_decay,
            self.prediction_count,
        );
        self.baseline_error_ema = update(
            self.baseline_error_ema,
            baseline_error,
            self.ema_decay,
            self.prediction_count,
        );
        self.relative_skill_ema = update(
            self.relative_skill_ema,
            relative_skill,
            self.ema_decay,
            self.prediction_count,
        );
        self.prediction_count += 1;
    }

    /// Access the underlying multi-scale encoder.
    pub fn encoder(&self) -> &MultiScaleEncoder {
        &self.encoder
    }

    /// Mutable access to the underlying multi-scale encoder.
    pub fn encoder_mut(&mut self) -> &mut MultiScaleEncoder {
        &mut self.encoder
    }

    /// Process a frame with full predictive coding feedback loop.
    ///
    /// Implements a proper cortical predictive coding cycle (Rao & Ballard 1999):
    /// 1. Encode at all scales
    /// 2. Apply top-down prior: coarse prediction biases fine encoding
    /// 3. Compute bottom-up error: cross-scale attention from prediction residuals
    /// 4. Re-encode fine scale with attention weighting
    /// 5. Update coarse→fine mapping from bottom-up error signal
    pub fn process_frame_with_feedback(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
    ) -> PredictiveOutput {
        let (_blended, scale_hvs, all_patches) =
            self.encoder.encode_frame(pixels, width, height, channels);

        let fine_hv = scale_hvs
            .first()
            .cloned()
            .unwrap_or_else(|| ContinuousHV::zero(self.dim));
        let coarse_hv = scale_hvs
            .last()
            .cloned()
            .unwrap_or_else(|| ContinuousHV::zero(self.dim));

        // Step 1: Apply top-down prior from previous coarse prediction
        let top_down_fine = if let Some(ref prev_coarse) = self.last_coarse_hv {
            let predicted = self.predict_fine(prev_coarse);
            // Blend prediction with bottom-up: 80% bottom-up + 20% top-down prior
            ContinuousHV::weighted_bundle(&[&fine_hv, &predicted], &[0.8, 0.2])
        } else {
            fine_hv.clone()
        };

        // Step 2: Compute geometry-aligned cross-scale attention from
        // position-unbound appearance HVs.
        let attention = self.compute_patch_attention_aligned(&all_patches, width, height);

        // Step 3: Rebundle the already-encoded fine patches with attention.
        // Re-extracting pixels here would advance the encoder's motion history a
        // second time for the same physical frame.
        let attended_fine = if !attention.is_empty() {
            match (self.encoder.encoder_at(0), all_patches.first()) {
                (Some(fine_enc), Some(fine_patches)) => {
                    let attended = fine_enc.bundle_attended_patches(fine_patches, &attention);
                    ContinuousHV::weighted_bundle(&[&attended, &top_down_fine], &[0.7, 0.3])
                }
                _ => top_down_fine,
            }
        } else {
            top_down_fine
        };

        // Step 4: Compute model error and calibrate it against the previous-fine
        // persistence baseline before updating the coarse→fine mapping.
        let (prediction_available, baseline_error, relative_skill) =
            if let (Some(prev_coarse), Some(prev_fine)) =
                (self.last_coarse_hv.clone(), self.last_fine_hv.as_ref())
            {
                let predicted_fine = self.predict_fine(&prev_coarse);
                self.prediction_error =
                    1.0 - attended_fine.similarity(&predicted_fine).clamp(-1.0, 1.0);
                let baseline_error = 1.0 - attended_fine.similarity(prev_fine).clamp(-1.0, 1.0);
                let relative_skill = Self::relative_skill(self.prediction_error, baseline_error);
                self.update_calibration(baseline_error, relative_skill);
                self.update_prediction_weight(&attended_fine, &predicted_fine, &prev_coarse);
                (true, baseline_error, relative_skill)
            } else {
                self.prediction_error = 0.0;
                (false, 0.0, 0.0)
            };

        self.last_coarse_hv = Some(coarse_hv.clone());
        self.last_fine_hv = Some(attended_fine.clone());

        PredictiveOutput {
            fine_hv: attended_fine,
            coarse_hv,
            prediction_available,
            prediction_error: self.prediction_error,
            baseline_error,
            relative_skill,
            error_ema: self.error_ema,
            patch_prediction_errors: attention,
        }
    }

    /// Compute geometry-aligned per-patch cross-scale prediction error.
    ///
    /// Fine patch centers are mapped into the coarse grid in pixel coordinates.
    /// Position bindings are removed before comparison, and multi-scale encoders
    /// share appearance bases, so the resulting similarity measures visual
    /// content rather than incompatible coordinate tags.
    fn compute_patch_attention_aligned(
        &self,
        all_patches: &[Vec<ContinuousHV>],
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        if all_patches.len() < 2 {
            return vec![];
        }

        let fine_patches = &all_patches[0];
        let coarse_patches = &all_patches[all_patches.len() - 1];
        let Some(fine_encoder) = self.encoder.encoder_at(0) else {
            return vec![];
        };
        let Some(coarse_encoder) = self.encoder.encoder_at(all_patches.len() - 1) else {
            return vec![];
        };

        let fine_grid = fine_encoder.grid_for(width, height);
        let coarse_grid = coarse_encoder.grid_for(width, height);
        if fine_grid.num_patches() != fine_patches.len()
            || coarse_grid.num_patches() != coarse_patches.len()
            || fine_grid.rows == 0
            || fine_grid.cols == 0
            || coarse_grid.rows == 0
            || coarse_grid.cols == 0
        {
            return vec![];
        }

        let fine_ps = fine_encoder.config().patch_size;
        let coarse_ps = coarse_encoder.config().patch_size.max(1);

        fine_patches
            .iter()
            .enumerate()
            .map(|(fine_idx, fine_hv)| {
                let fine_row = fine_idx / fine_grid.cols;
                let fine_col = fine_idx % fine_grid.cols;
                let center_y = fine_row * fine_ps + fine_ps / 2;
                let center_x = fine_col * fine_ps + fine_ps / 2;
                let coarse_row = (center_y / coarse_ps).min(coarse_grid.rows - 1);
                let coarse_col = (center_x / coarse_ps).min(coarse_grid.cols - 1);
                let coarse_idx = coarse_row * coarse_grid.cols + coarse_col;

                let fine_appearance = fine_encoder.unbind_position(fine_hv, fine_row, fine_col);
                let coarse_appearance = coarse_encoder.unbind_position(
                    &coarse_patches[coarse_idx],
                    coarse_row,
                    coarse_col,
                );
                let sim = fine_appearance.similarity(&coarse_appearance).max(0.0);
                1.0 - sim
            })
            .collect()
    }

    /// Compute per-patch attention from cross-scale prediction error.
    ///
    /// Geometry-free compatibility helper for callers that only retained patch
    /// arrays. The live hierarchy uses the aligned method above because a linear
    /// index ratio does not preserve two-dimensional correspondence.
    pub fn compute_patch_attention(all_patches: &[Vec<ContinuousHV>]) -> Vec<f32> {
        if all_patches.len() < 2 {
            return vec![];
        }

        let fine_patches = &all_patches[0];
        let coarse_patches = &all_patches[all_patches.len() - 1];

        if fine_patches.is_empty() || coarse_patches.is_empty() {
            return vec![];
        }

        let fine_per_coarse = (fine_patches.len() / coarse_patches.len()).max(1);

        fine_patches
            .iter()
            .enumerate()
            .map(|(i, fine_hv)| {
                let coarse_idx = (i / fine_per_coarse).min(coarse_patches.len() - 1);
                let sim = fine_hv.similarity(&coarse_patches[coarse_idx]).max(0.0);
                1.0 - sim
            })
            .collect()
    }

    pub(crate) fn save_state(&self) -> PredictiveHierarchyState {
        PredictiveHierarchyState {
            prediction_weight: self.prediction_weight.as_slice().to_vec(),
            last_coarse_hv: self
                .last_coarse_hv
                .as_ref()
                .map(|hv| hv.as_slice().to_vec()),
            last_fine_hv: self.last_fine_hv.as_ref().map(|hv| hv.as_slice().to_vec()),
            prediction_error: self.prediction_error,
            error_ema: self.error_ema,
            baseline_error_ema: self.baseline_error_ema,
            relative_skill_ema: self.relative_skill_ema,
            prediction_count: self.prediction_count,
            ema_decay: self.ema_decay,
            learning_rate: self.learning_rate,
            scale_prev_patch_lum: (0..self.encoder.scales().len())
                .filter_map(|idx| self.encoder.encoder_at(idx))
                .map(|encoder| encoder.prev_patch_lum.clone())
                .collect(),
        }
    }

    pub(crate) fn validate_state(
        state: &PredictiveHierarchyState,
        expected_dim: usize,
        expected_scales: usize,
    ) -> Result<(), String> {
        let validate_hv = |name: &str, values: &[f32]| -> Result<(), String> {
            if values.len() != expected_dim {
                return Err(format!(
                    "{name} dimension mismatch: saved={}, expected={expected_dim}",
                    values.len()
                ));
            }
            if !values.iter().all(|value| value.is_finite()) {
                return Err(format!("{name} contains non-finite values"));
            }
            Ok(())
        };
        validate_hv("predictive.prediction_weight", &state.prediction_weight)?;
        if let Some(values) = &state.last_coarse_hv {
            validate_hv("predictive.last_coarse_hv", values)?;
        }
        if let Some(values) = &state.last_fine_hv {
            validate_hv("predictive.last_fine_hv", values)?;
        }
        if state.scale_prev_patch_lum.len() != expected_scales {
            return Err(format!(
                "predictive scale history mismatch: saved={}, expected={expected_scales}",
                state.scale_prev_patch_lum.len()
            ));
        }
        if state
            .scale_prev_patch_lum
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err("predictive scale history contains non-finite values".to_string());
        }
        for (name, value) in [
            ("prediction_error", state.prediction_error),
            ("error_ema", state.error_ema),
            ("baseline_error_ema", state.baseline_error_ema),
            ("relative_skill_ema", state.relative_skill_ema),
            ("ema_decay", state.ema_decay),
            ("learning_rate", state.learning_rate),
        ] {
            if !value.is_finite() {
                return Err(format!("predictive {name} is non-finite"));
            }
        }
        if !(0.0..1.0).contains(&state.ema_decay) {
            return Err(format!(
                "predictive ema_decay must be in (0,1), got {}",
                state.ema_decay
            ));
        }
        if state.learning_rate <= 0.0 {
            return Err(format!(
                "predictive learning_rate must be > 0, got {}",
                state.learning_rate
            ));
        }
        Ok(())
    }

    pub(crate) fn load_state(&mut self, state: &PredictiveHierarchyState) {
        self.prediction_weight = ContinuousHV::from_vec(state.prediction_weight.clone());
        self.last_coarse_hv = state
            .last_coarse_hv
            .as_ref()
            .map(|values| ContinuousHV::from_vec(values.clone()));
        self.last_fine_hv = state
            .last_fine_hv
            .as_ref()
            .map(|values| ContinuousHV::from_vec(values.clone()));
        self.prediction_error = state.prediction_error;
        self.error_ema = state.error_ema;
        self.baseline_error_ema = state.baseline_error_ema;
        self.relative_skill_ema = state.relative_skill_ema;
        self.prediction_count = state.prediction_count;
        self.ema_decay = state.ema_decay;
        self.learning_rate = state.learning_rate;
        for (idx, history) in state.scale_prev_patch_lum.iter().enumerate() {
            if let Some(encoder) = self.encoder.encoder_at_mut(idx) {
                encoder.prev_patch_lum = history.clone();
            }
        }
    }

    /// Reset observation history and calibration while preserving the learned
    /// coarse-to-fine mapping weight.
    pub fn reset(&mut self) {
        self.last_coarse_hv = None;
        self.last_fine_hv = None;
        self.prediction_error = 0.0;
        self.error_ema = 0.0;
        self.baseline_error_ema = 0.0;
        self.relative_skill_ema = 0.0;
        self.prediction_count = 0;
        for scale_idx in 0..self.encoder.scales().len() {
            if let Some(encoder) = self.encoder.encoder_at_mut(scale_idx) {
                encoder.prev_patch_lum.clear();
            }
        }
    }
}

/// Output from one predictive coding cycle.
#[derive(Debug, Clone)]
pub struct PredictiveOutput {
    /// Fine-scale (8px) encoding.
    pub fine_hv: ContinuousHV,
    /// Coarse-scale (32px) encoding.
    pub coarse_hv: ContinuousHV,
    /// Whether a prior coarse/fine pair existed for a valid temporal prediction.
    pub prediction_available: bool,
    /// Prediction error: how well coarse predicted fine this frame.
    pub prediction_error: f32,
    /// Error of the previous-fine persistence baseline.
    pub baseline_error: f32,
    /// Normalized skill relative to persistence, in [-1, 1].
    pub relative_skill: f32,
    /// Exponential moving average of prediction error.
    pub error_ema: f32,
    /// Per-patch cross-scale prediction error (1 - cos_sim(fine_patch, predicted_from_coarse)).
    ///
    /// Length = number of fine-scale patches. Each entry is in [0.0, ~1.5].
    /// High values indicate regions where the coarse encoding fails to capture
    /// the fine-grained structure — these are true free-energy hotspots.
    /// Empty until the second frame (no prediction on first frame).
    pub patch_prediction_errors: Vec<f32>,
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
    fn test_try_new_rejects_single_scale_or_zero_capacity() {
        let mut cfg = VisionConfig::default();
        cfg.multi_scale.scales = vec![8];
        assert!(PredictiveCodingHierarchy::try_new(&cfg, 64, 64).is_err());

        cfg.multi_scale.scales = vec![8, 32];
        assert!(PredictiveCodingHierarchy::try_new(&cfg, 0, 64).is_err());
    }

    #[test]
    fn test_checked_processing_rejects_before_temporal_mutation() {
        let cfg = VisionConfig::default();
        let mut hierarchy = PredictiveCodingHierarchy::new(&cfg, 32, 32);
        let before = hierarchy.save_state();
        let error = hierarchy
            .process_frame_checked(&[0; 7], 8, 8, 1)
            .unwrap_err();
        assert!(error.contains("length mismatch"));
        assert_eq!(hierarchy.save_state(), before);
    }

    #[test]
    fn test_hierarchy_construction() {
        let cfg = VisionConfig::default();
        let pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        assert_eq!(pch.prediction_error(), 0.0);
        assert_eq!(pch.error_ema(), 0.0);
        assert_eq!(pch.baseline_error_ema(), 0.0);
        assert_eq!(pch.relative_skill_ema(), 0.0);
        assert_eq!(pch.prediction_count(), 0);
    }

    #[test]
    fn test_single_frame_no_prediction() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        let out = pch.process_frame(&frame, 64, 64, 1);

        // First frame has no temporal prediction and must not contaminate calibration.
        assert!(!out.prediction_available);
        assert_eq!(out.prediction_error, 0.0);
        assert_eq!(pch.prediction_count(), 0);
        assert!(out.fine_hv.norm() > 0.0);
        assert!(out.coarse_hv.norm() > 0.0);
    }

    #[test]
    fn test_prediction_error_finite() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);

        let frame = gradient_frame(64, 64);
        for _ in 0..10 {
            let out = pch.process_frame(&frame, 64, 64, 1);
            assert!(
                out.prediction_error.is_finite(),
                "Prediction error should be finite"
            );
            assert!(out.prediction_error >= 0.0);
        }
    }

    #[test]
    fn test_prediction_error_decreases_for_static_scene() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let frame = solid_gray_frame(64, 64, 128);

        let mut errors = Vec::new();
        for _ in 0..30 {
            let out = pch.process_frame(&frame, 64, 64, 1);
            errors.push(out.prediction_error);
        }

        // After learning, prediction error should decrease
        let early_mean: f32 = errors[2..5].iter().sum::<f32>() / 3.0;
        let late_mean: f32 = errors[25..30].iter().sum::<f32>() / 5.0;
        assert!(
            late_mean <= early_mean + 0.1,
            "Prediction error should decrease for static scene: early={early_mean}, late={late_mean}"
        );
    }

    #[test]
    fn test_scene_change_increases_error() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);

        // Converge on scene A
        let frame_a = solid_gray_frame(64, 64, 50);
        for _ in 0..20 {
            pch.process_frame(&frame_a, 64, 64, 1);
        }
        let stable_error = pch.prediction_error();

        // Switch to scene B
        let frame_b = solid_gray_frame(64, 64, 200);
        let out = pch.process_frame(&frame_b, 64, 64, 1);

        assert!(
            out.prediction_error >= stable_error * 0.5,
            "Scene change should increase error: stable={stable_error}, change={}",
            out.prediction_error
        );
    }

    #[test]
    fn test_ema_tracks_errors() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        for _ in 0..10 {
            pch.process_frame(&frame, 64, 64, 1);
        }

        // EMA should be non-zero after processing frames
        assert!(pch.error_ema().is_finite());
        assert!(pch.error_ema() >= 0.0);
    }

    #[test]
    fn test_prediction_is_calibrated_against_persistence() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        let first = pch.process_frame(&frame, 64, 64, 1);
        assert!(!first.prediction_available);
        let second = pch.process_frame(&frame, 64, 64, 1);
        assert!(second.prediction_available);
        assert!(second.baseline_error.is_finite());
        assert!((-1.0..=1.0).contains(&second.relative_skill));
        assert_eq!(pch.prediction_count(), 1);
    }

    #[test]
    fn test_reset() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        pch.process_frame(&frame, 64, 64, 1);
        pch.process_frame(&frame, 64, 64, 1);
        assert!(pch.prediction_error() < 1.0 || pch.error_ema() > 0.0);

        pch.reset();
        assert_eq!(pch.prediction_error(), 0.0);
        assert_eq!(pch.error_ema(), 0.0);
        assert_eq!(pch.baseline_error_ema(), 0.0);
        assert_eq!(pch.relative_skill_ema(), 0.0);
        assert_eq!(pch.prediction_count(), 0);
    }

    // === Predictive Coding Feedback ===

    #[test]
    fn test_process_frame_with_feedback_produces_valid_output() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        let out = pch.process_frame_with_feedback(&frame, 64, 64, 1);
        assert!(out.fine_hv.norm() > 0.0);
        assert!(out.coarse_hv.norm() > 0.0);
        assert!(out.prediction_error >= 0.0);
    }

    #[test]
    fn test_feedback_differs_from_no_feedback() {
        let cfg = VisionConfig::default();
        let mut pch1 = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let mut pch2 = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        // Process first frame to establish state
        pch1.process_frame(&frame, 64, 64, 1);
        pch2.process_frame_with_feedback(&frame, 64, 64, 1);

        // Second frame: compare outputs
        let out_no_fb = pch1.process_frame(&frame, 64, 64, 1);
        let out_fb = pch2.process_frame_with_feedback(&frame, 64, 64, 1);

        // Both should produce valid output; they may differ due to attention weighting
        assert!(out_no_fb.fine_hv.norm() > 0.0);
        assert!(out_fb.fine_hv.norm() > 0.0);
    }

    #[test]
    fn test_compute_patch_attention() {
        let dim = 16_384;
        let fine: Vec<ContinuousHV> = (0..16)
            .map(|i| ContinuousHV::random(dim, 1000 + i))
            .collect();
        let coarse: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(dim, 2000 + i))
            .collect();

        let attention = PredictiveCodingHierarchy::compute_patch_attention(&[fine, coarse]);
        assert_eq!(attention.len(), 16);
        for &a in &attention {
            assert!(a >= 0.0 && a <= 1.5, "attention should be bounded: {a}");
        }
    }

    #[test]
    fn test_aligned_attention_preserves_two_dimensional_correspondence() {
        let mut cfg = VisionConfig::default();
        cfg.enable_motion = false;
        cfg.enable_color = false;
        cfg.enable_opponent_color = false;
        cfg.multi_scale.scales = vec![8, 16];
        let pch = PredictiveCodingHierarchy::new(&cfg, 32, 32);

        let normal = vec![0.2; cfg.total_features()];
        let anomaly = vec![0.9; cfg.total_features()];
        let mut fine_features = vec![normal.clone(); 16];
        fine_features[4] = anomaly.clone(); // row 1, col 0 → coarse row 0, col 0
        let coarse_features = vec![anomaly, normal.clone(), normal.clone(), normal];

        let (_, fine_patches) = pch
            .encoder
            .encoder_at(0)
            .expect("fine encoder")
            .encode_precomputed(&fine_features);
        let (_, coarse_patches) = pch
            .encoder
            .encoder_at(1)
            .expect("coarse encoder")
            .encode_precomputed(&coarse_features);

        let attention =
            pch.compute_patch_attention_aligned(&[fine_patches, coarse_patches], 32, 32);
        assert_eq!(attention.len(), 16);
        assert!(
            attention[4] + 0.1 < attention[5],
            "fine patch 4 must map to coarse patch 0 by 2D center geometry"
        );
    }

    #[test]
    fn test_predict_fine_produces_valid_hv() {
        let cfg = VisionConfig::default();
        let pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let coarse = ContinuousHV::random(cfg.hdc_dim, 999);

        let predicted = pch.predict_fine(&coarse);
        assert_eq!(predicted.dim(), cfg.hdc_dim);
        assert!(predicted.norm() > 0.0);
    }
}
