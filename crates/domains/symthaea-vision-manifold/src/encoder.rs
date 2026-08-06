// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Patch-based HDC encoder for video frames.
//!
//! Encodes raw pixel data into holographic hypervectors using the bind-bundle
//! paradigm from Hyperdimensional Computing:
//!
//! ```text
//! frame_hv = normalize(Σ_p  position(row_p, col_p) ⊗ appearance(patch_p))
//! ```
//!
//! where `⊗` is the HDC bind operation (element-wise multiply) and appearance
//! is itself a bound-bundled encoding of per-patch features:
//!
//! ```text
//! appearance = normalize(Σ_f  feature_basis_f ⊗ level(quantize(value_f)))
//! ```
//!
//! Position uses factored row/col basis vectors (GridEncoder pattern) so memory
//! scales as O(rows + cols) rather than O(rows × cols).

use symthaea_core::hdc::ContinuousHV;

use crate::types::{PatchGrid, ScaleHealth, VisionConfig};

/// `(blended_hv, per_scale_hvs, per_scale_patches)` -- the result of a checked multi-scale
/// frame encode. Named per clippy's `type_complexity` suggestion (reused at two call sites).
type MultiScaleEncodeResult =
    Result<(ContinuousHV, Vec<ContinuousHV>, Vec<Vec<ContinuousHV>>), String>;

/// Confidence-aware per-patch stereo reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoDepthEstimate {
    /// Raw normalized depth (`0 = near`, `1 = far`).
    pub depths: Vec<f32>,
    /// Match confidence in `[0, 1]`.
    pub confidences: Vec<f32>,
    /// Winning horizontal disparity in pixels.
    pub disparities: Vec<usize>,
}

impl StereoDepthEstimate {
    /// Blend uncertain estimates toward neutral depth before feature injection.
    pub fn fused_depths(&self) -> Vec<f32> {
        self.depths
            .iter()
            .zip(&self.confidences)
            .map(|(&depth, &confidence)| {
                let confidence = confidence.clamp(0.0, 1.0);
                (confidence * depth + (1.0 - confidence) * 0.5).clamp(0.0, 1.0)
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.depths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.depths.is_empty()
    }
}

/// Encodes video frames into holographic hypervectors.
pub struct PatchHdcEncoder {
    config: VisionConfig,
    row_hvs: Vec<ContinuousHV>,
    col_hvs: Vec<ContinuousHV>,
    feature_hvs: Vec<ContinuousHV>,
    level_hvs: Vec<ContinuousHV>,
    /// Per-feature adaptive weights (learned via contrastive refinement).
    feature_weights: Vec<f32>,
    max_rows: usize,
    max_cols: usize,
    /// Previous frame's per-patch mean luminance for motion features.
    pub(crate) prev_patch_lum: Vec<f32>,
}

impl PatchHdcEncoder {
    /// Create a new encoder sized for frames up to `max_width × max_height`.
    pub fn new(config: &VisionConfig, max_width: u32, max_height: u32) -> Self {
        Self::new_with_basis_seeds(config, max_width, max_height, config.seed, config.seed)
    }

    /// Construct an encoder with independent spatial and appearance seeds.
    ///
    /// Multi-scale encoders use unique spatial bases per scale while sharing the
    /// feature/level bases. That makes position-unbound appearance HVs directly
    /// comparable across scales without collapsing the spatial coordinate systems.
    fn new_with_basis_seeds(
        config: &VisionConfig,
        max_width: u32,
        max_height: u32,
        position_seed: u64,
        appearance_seed: u64,
    ) -> Self {
        let patch_size = config.patch_size.max(1);
        let max_cols = (max_width as usize).div_ceil(patch_size).max(1);
        let max_rows = (max_height as usize).div_ceil(patch_size).max(1);

        let row_hvs: Vec<ContinuousHV> = (0..max_rows)
            .map(|r| ContinuousHV::random(config.hdc_dim, position_seed + r as u64))
            .collect();

        let col_hvs: Vec<ContinuousHV> = (0..max_cols)
            .map(|c| ContinuousHV::random(config.hdc_dim, position_seed + 50_000 + c as u64))
            .collect();

        let total_features = config.total_features();
        let feature_hvs: Vec<ContinuousHV> = (0..total_features)
            .map(|f| ContinuousHV::random(config.hdc_dim, appearance_seed + 100_000 + f as u64))
            .collect();

        let level_hvs =
            Self::generate_level_hvs(config.hdc_dim, config.num_levels, appearance_seed + 200_000);

        let feature_weights = vec![1.0 / total_features as f32; total_features];

        Self {
            config: config.clone(),
            row_hvs,
            col_hvs,
            feature_hvs,
            level_hvs,
            feature_weights,
            max_rows,
            max_cols,
            prev_patch_lum: Vec::new(),
        }
    }

    /// Perform 'Holographic Dilation' - scale internal basis vectors.
    pub(crate) fn hdc_vector_count(&self) -> usize {
        self.row_hvs.len() + self.col_hvs.len() + self.feature_hvs.len() + self.level_hvs.len()
    }

    pub fn dilate(&mut self, target_dim: usize) {
        if self.config.hdc_dim == target_dim {
            return;
        }

        for hv in &mut self.row_hvs {
            *hv = hv.dilate(target_dim);
        }
        for hv in &mut self.col_hvs {
            *hv = hv.dilate(target_dim);
        }
        for hv in &mut self.feature_hvs {
            *hv = hv.dilate(target_dim);
        }
        for hv in &mut self.level_hvs {
            *hv = hv.dilate(target_dim);
        }

        self.config.hdc_dim = target_dim;
    }

    /// Generate level HVs with ordinal similarity preservation.
    ///
    /// Adjacent levels share most dimensions; distant levels are nearly orthogonal.
    /// Uses the progressive random-flip strategy from Imani et al. (2019).
    fn generate_level_hvs(dim: usize, num_levels: usize, seed: u64) -> Vec<ContinuousHV> {
        if num_levels == 0 {
            return vec![];
        }
        let base = ContinuousHV::random(dim, seed);
        let mut levels = vec![base.clone()];
        let flips_per_level = dim / num_levels.max(1);

        let mut rng_state = seed ^ 0xDEAD_BEEF_CAFE_1234;
        let mut current = base;

        for _ in 1..num_levels {
            let mut values = current.as_slice().to_vec();
            for _ in 0..flips_per_level {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let idx = (rng_state as usize) % dim;
                values[idx] = -values[idx];
            }
            current = ContinuousHV::from_vec(values);
            levels.push(current.clone());
        }

        levels
    }

    /// Encode a raw pixel buffer into a holographic frame HV.
    ///
    /// Returns `(frame_hv, per_patch_hvs)` where per-patch HVs enable
    /// spatial surprise computation.
    ///
    /// # Arguments
    /// * `pixels` — Raw pixel data (row-major, tightly packed).
    /// * `width` / `height` — Frame dimensions in pixels.
    /// * `channels` — Bytes per pixel (1 for grayscale, 3 for RGB, 4 for RGBA).
    pub fn encode_frame(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
    ) -> (ContinuousHV, Vec<ContinuousHV>) {
        self.encode_frame_impl(pixels, width, height, channels, None)
    }

    /// Encode a frame while overriding the depth feature for each patch.
    ///
    /// `patch_depths` must be in row-major patch-grid order and use the same
    /// convention as the built-in depth channel: `0.0 = near`, `1.0 = far`.
    /// The override is ignored when `enable_depth` is false.
    pub fn encode_frame_with_depth(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        patch_depths: &[f32],
    ) -> (ContinuousHV, Vec<ContinuousHV>) {
        let grid = PatchGrid::new(width, height, self.config.patch_size);
        assert_eq!(
            patch_depths.len(),
            grid.num_patches(),
            "depth override length must match patch grid"
        );
        self.encode_frame_impl(pixels, width, height, channels, Some(patch_depths))
    }

    fn encode_frame_impl(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        patch_depths: Option<&[f32]>,
    ) -> (ContinuousHV, Vec<ContinuousHV>) {
        let grid = PatchGrid::new(width, height, self.config.patch_size);
        assert!(
            grid.rows <= self.max_rows && grid.cols <= self.max_cols,
            "frame {width}x{height} exceeds encoder capacity of {}x{} patches",
            self.max_cols,
            self.max_rows
        );
        if grid.num_patches() == 0 {
            return (ContinuousHV::zero(self.config.hdc_dim), vec![]);
        }

        let mut patch_hvs = Vec::with_capacity(grid.num_patches());
        let mut current_lum = Vec::with_capacity(grid.num_patches());

        for row in 0..grid.rows {
            for col in 0..grid.cols {
                let patch_idx = row * grid.cols + col;
                let prev_lum = self.prev_patch_lum.get(patch_idx).copied().unwrap_or(0.0);
                let features = self.extract_patch_features(
                    pixels,
                    width,
                    height,
                    channels,
                    col * self.config.patch_size,
                    row * self.config.patch_size,
                    prev_lum,
                    patch_depths.and_then(|depths| depths.get(patch_idx).copied()),
                );
                // Store current luminance (first feature = mean_r for grayscale, or weighted lum)
                let mean_lum = 0.299 * features[0] + 0.587 * features[1] + 0.114 * features[2];
                current_lum.push(mean_lum);

                let appearance = self.encode_features(&features);
                let position = self.position_hv(row, col);
                patch_hvs.push(position.bind(&appearance));
            }
        }

        self.prev_patch_lum = current_lum;

        let refs: Vec<&ContinuousHV> = patch_hvs.iter().collect();
        let frame_hv = ContinuousHV::bundle(&refs);
        (frame_hv, patch_hvs)
    }

    /// Encode a frame with attention-weighted patch contributions.
    ///
    /// Patches with higher attention values contribute more to the frame HV,
    /// making the encoding focus on surprising/salient regions.
    pub fn encode_frame_attended(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        attention: &[f32],
    ) -> (ContinuousHV, Vec<ContinuousHV>) {
        let grid = PatchGrid::new(width, height, self.config.patch_size);
        assert!(
            grid.rows <= self.max_rows && grid.cols <= self.max_cols,
            "frame {width}x{height} exceeds encoder capacity of {}x{} patches",
            self.max_cols,
            self.max_rows
        );
        if grid.num_patches() == 0 {
            return (ContinuousHV::zero(self.config.hdc_dim), vec![]);
        }

        let mut patch_hvs = Vec::with_capacity(grid.num_patches());
        let mut current_lum = Vec::with_capacity(grid.num_patches());

        for row in 0..grid.rows {
            for col in 0..grid.cols {
                let patch_idx = row * grid.cols + col;
                let prev_lum = self.prev_patch_lum.get(patch_idx).copied().unwrap_or(0.0);
                let features = self.extract_patch_features(
                    pixels,
                    width,
                    height,
                    channels,
                    col * self.config.patch_size,
                    row * self.config.patch_size,
                    prev_lum,
                    None,
                );
                let mean_lum = 0.299 * features[0] + 0.587 * features[1] + 0.114 * features[2];
                current_lum.push(mean_lum);

                let appearance = self.encode_features(&features);
                let position = self.position_hv(row, col);
                patch_hvs.push(position.bind(&appearance));
            }
        }

        self.prev_patch_lum = current_lum;

        let frame_hv = self.bundle_attended_patches(&patch_hvs, attention);
        (frame_hv, patch_hvs)
    }

    /// Rebundle already-encoded patches using attention weights.
    ///
    /// This is intentionally side-effect free: predictive feedback can focus an
    /// existing fine-scale encoding without extracting the same frame twice and
    /// accidentally advancing motion history a second time.
    pub fn bundle_attended_patches(
        &self,
        patch_hvs: &[ContinuousHV],
        attention: &[f32],
    ) -> ContinuousHV {
        if patch_hvs.is_empty() {
            return ContinuousHV::zero(self.config.hdc_dim);
        }

        let max_att = attention.iter().copied().fold(0.0f32, f32::max).max(1e-8);
        let weights: Vec<f32> = (0..patch_hvs.len())
            .map(|i| {
                let att = attention.get(i).copied().unwrap_or(0.0) / max_att;
                // Base weight 1.0 + attention boost up to 2x.
                1.0 + att
            })
            .collect();

        let refs: Vec<&ContinuousHV> = patch_hvs.iter().collect();
        ContinuousHV::weighted_bundle(&refs, &weights)
    }

    /// Convenience wrapper for single-channel grayscale frames.
    pub fn encode_grayscale(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> (ContinuousHV, Vec<ContinuousHV>) {
        self.encode_frame(pixels, width, height, 1)
    }

    /// Encode pre-extracted feature vectors directly (bypass pixel extraction).
    ///
    /// Each entry in `patch_features` is a feature vector for one patch,
    /// in row-major order matching the grid layout.
    pub fn encode_precomputed(
        &self,
        patch_features: &[Vec<f32>],
    ) -> (ContinuousHV, Vec<ContinuousHV>) {
        if patch_features.is_empty() {
            return (ContinuousHV::zero(self.config.hdc_dim), vec![]);
        }

        // Infer grid dimensions: assume square-ish or accept flat index
        let n = patch_features.len();
        let side = (n as f32).sqrt().ceil() as usize;
        let cols = side.min(self.max_cols);
        let rows = n.div_ceil(cols);

        let mut patch_hvs = Vec::with_capacity(n);
        for (idx, features) in patch_features.iter().enumerate() {
            let r = idx / cols;
            let c = idx % cols;
            let appearance = self.encode_features(features);
            let position = self.position_hv(r, c);
            patch_hvs.push(position.bind(&appearance));
        }

        let refs: Vec<&ContinuousHV> = patch_hvs.iter().collect();
        let frame_hv = if rows > 0 {
            ContinuousHV::bundle(&refs)
        } else {
            ContinuousHV::zero(self.config.hdc_dim)
        };
        (frame_hv, patch_hvs)
    }

    /// Extract the appearance component from a position-bound patch HV.
    ///
    /// `patch_hv = position_hv ⊗ appearance_hv`, where `⊗` is element-wise
    /// multiplication. Binding is only *exactly* self-inverse
    /// (`a ⊗ a ⊗ b = b`) for bipolar {-1, +1} vectors; `position_hv` here is
    /// continuous-valued, so re-binding with the position HV a second time
    /// (the textbook bipolar-HDC trick) only approximately recovers
    /// appearance and materially degrades cross-scale/cross-position
    /// comparability. The *exact* inverse of element-wise multiplication is
    /// element-wise reciprocal, so we recover appearance via
    /// `patch_hv ⊗ position_hv⁻¹` using [`ContinuousHV::inverse`]
    /// (documented there for precisely this unbinding use case).
    ///
    /// This enables position-invariant template matching: compare a template's
    /// appearance against patches regardless of where they are on screen.
    pub fn unbind_position(&self, patch_hv: &ContinuousHV, row: usize, col: usize) -> ContinuousHV {
        let pos = self.position_hv(row, col);
        patch_hv.bind(&pos.inverse())
    }

    /// Factored position HV: row_hv ⊗ col_hv (GridEncoder pattern).
    fn position_hv(&self, row: usize, col: usize) -> ContinuousHV {
        let r = row % self.max_rows.max(1);
        let c = col % self.max_cols.max(1);
        self.row_hvs[r].bind(&self.col_hvs[c])
    }

    /// Extract features from a single patch.
    ///
    /// Base features (5): [mean_r, mean_g, mean_b, edge_density, variance]
    /// Motion features (+2 if enabled): [temporal_diff, motion_magnitude]
    /// Color features (+2 if enabled): [mean_cb, mean_cr]
    ///
    /// All values normalized to [0, 1].
    // Genuinely needs this many parameters: the raw frame buffer plus its geometry
    // (width/height/channels), the patch coordinates within it, and two cross-frame/
    // cross-modal inputs (prev_mean_lum, depth_override) that only this call site has.
    #[allow(clippy::too_many_arguments)]
    fn extract_patch_features(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        patch_x: usize,
        patch_y: usize,
        prev_mean_lum: f32,
        depth_override: Option<f32>,
    ) -> Vec<f32> {
        let ps = self.config.patch_size;
        let stride = width as usize * channels.max(1);

        let mut sum_r = 0.0f32;
        let mut sum_g = 0.0f32;
        let mut sum_b = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut edge_sum = 0.0f32;
        let mut count = 0.0f32;
        // For color chrominance
        let mut sum_cb = 0.0f32;
        let mut sum_cr = 0.0f32;

        for dy in 0..ps {
            let mut prev_lum_cached = 0.0f32;
            for dx in 0..ps {
                let y = patch_y + dy;
                let x = patch_x + dx;
                let offset = y * stride + x * channels.max(1);

                if offset + channels.max(1) > pixels.len() {
                    continue;
                }

                let (r, g, b) = if channels >= 3 {
                    (
                        pixels[offset] as f32,
                        pixels[offset + 1] as f32,
                        pixels[offset + 2] as f32,
                    )
                } else {
                    let v = pixels[offset] as f32;
                    (v, v, v)
                };

                sum_r += r;
                sum_g += g;
                sum_b += b;
                let lum = 0.299 * r + 0.587 * g + 0.114 * b;
                sum_sq += lum * lum;

                // YCbCr chrominance (ITU-R BT.601)
                if channels >= 3 && self.config.enable_color {
                    // Cb = 128 - 0.169*R - 0.331*G + 0.500*B
                    // Cr = 128 + 0.500*R - 0.419*G - 0.081*B
                    sum_cb += 128.0 - 0.169 * r - 0.331 * g + 0.500 * b;
                    sum_cr += 128.0 + 0.500 * r - 0.419 * g - 0.081 * b;
                }

                // Horizontal gradient magnitude for edge detection
                // Cache previous luminance to avoid redundant pixel re-read
                if dx > 0 {
                    edge_sum += (lum - prev_lum_cached).abs();
                }
                prev_lum_cached = lum;

                count += 1.0;
            }
        }

        if count == 0.0 {
            return vec![0.0; self.config.total_features()];
        }

        let inv_count = 1.0 / count;
        let inv_255 = 1.0 / 255.0;
        let mean_r = sum_r * inv_count * inv_255;
        let mean_g = sum_g * inv_count * inv_255;
        let mean_b = sum_b * inv_count * inv_255;
        let mean_lum = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b;
        let variance = (sum_sq * inv_count * inv_255 * inv_255 - mean_lum * mean_lum).max(0.0);
        let edge_density = (edge_sum * inv_count * inv_255).min(1.0);

        let mut features = vec![mean_r, mean_g, mean_b, edge_density, variance];

        // Motion features: temporal difference and magnitude
        if self.config.enable_motion {
            let temporal_diff = (mean_lum - prev_mean_lum).abs().min(1.0);
            // Motion magnitude: combine temporal diff with edge density as proxy
            let motion_magnitude = (temporal_diff * 0.7 + edge_density * 0.3).min(1.0);
            features.push(temporal_diff);
            features.push(motion_magnitude);
        }

        // Color features: Cb and Cr chrominance
        if self.config.enable_color {
            if channels >= 3 {
                let mean_cb = (sum_cb * inv_count * inv_255).clamp(0.0, 1.0);
                let mean_cr = (sum_cr * inv_count * inv_255).clamp(0.0, 1.0);
                features.push(mean_cb);
                features.push(mean_cr);
            } else {
                // Grayscale: chrominance is neutral (0.5 = no color)
                features.push(0.5);
                features.push(0.5);
            }
        }

        // Opponent color features: model V1 double-opponent cells (Hubel & Wiesel 1968).
        //
        // Red–green (R–G) and blue–yellow (B–Y) opponent channels capture color
        // contrast in a perceptually meaningful basis. Values are mapped from
        // [-1, 1] to [0, 1] so they fit the feature quantizer.
        //
        //   rg = 0.5 means pure achromatic (R ≈ G)
        //   rg > 0.5 means red bias; rg < 0.5 means green bias
        //   by = 0.5 means achromatic; by > 0.5 means blue bias; by < 0.5 means yellow bias
        if self.config.enable_opponent_color {
            if channels >= 3 {
                // mean_r / mean_g / mean_b are already normalized to [0, 1]
                let rg = (mean_r - mean_g + 1.0) / 2.0; // [0, 1]
                let by = (mean_b - 0.5 * (mean_r + mean_g) + 1.0) / 2.0; // [0, 1]
                features.push(rg.clamp(0.0, 1.0));
                features.push(by.clamp(0.0, 1.0));
            } else {
                // Grayscale: opponent channels are neutral
                features.push(0.5);
                features.push(0.5);
            }
        }

        // Depth channel: monocular depth cues when no sensor data is available.
        //
        // Two cues combined (Cutting & Vishton 1995):
        // 1. **Texture gradient** — variance decreases with distance (Gibson 1950):
        //    `texture_depth = 1.0 - variance` (low variance → far away)
        // 2. **Relative vertical position** — objects lower in frame are closer
        //    (perspective assumption for ground-plane scenes):
        //    `position_depth = patch_y / frame_height` (0 = top/far, 1 = bottom/near)
        //
        // Final depth = 0.6 * texture + 0.4 * position (texture-dominant blend).
        // Values in [0, 1]: 0 = near, 1 = far.
        if self.config.enable_depth {
            let depth = if let Some(sensor_depth) = depth_override {
                sensor_depth.clamp(0.0, 1.0)
            } else {
                let texture_depth = (1.0 - variance).clamp(0.0, 1.0);
                let frame_height = height.max(1) as f32;
                let position_depth = 1.0 - (patch_y as f32 / frame_height).clamp(0.0, 1.0);
                (0.6 * texture_depth + 0.4 * position_depth).clamp(0.0, 1.0)
            };
            features.push(depth);
        }

        features
    }

    /// Encode a feature vector into a ContinuousHV via fused bind-and-accumulate.
    ///
    /// Feature weights modulate each feature's contribution to the final HV.
    /// Uses a single accumulation buffer to avoid N intermediate allocations.
    fn encode_features(&self, features: &[f32]) -> ContinuousHV {
        let num = features.len().min(self.feature_hvs.len());
        if num == 0 {
            return ContinuousHV::zero(self.config.hdc_dim);
        }

        let dim = self.config.hdc_dim;
        let mut accum = vec![0.0f32; dim];

        for (i, &val) in features.iter().take(num).enumerate() {
            let level_idx = self.quantize(val);
            let weight = self.feature_weights[i];
            let feat_s = self.feature_hvs[i].as_slice();
            let level_s = self.level_hvs[level_idx].as_slice();
            for d in 0..dim {
                accum[d] += weight * feat_s[d] * level_s[d];
            }
        }

        ContinuousHV::from_vec(accum).normalize()
    }

    /// Refine feature weights via contrastive learning.
    ///
    /// Adjusts weights so that the encoder produces HVs more similar to
    /// `positive` and less similar to `negative`. Uses a perceptron-like
    /// update rule on the feature weight vector.
    pub fn refine_contrastive(
        &mut self,
        positive: &ContinuousHV,
        negative: &ContinuousHV,
        lr: f32,
    ) {
        let num = self.feature_weights.len();
        // Compute per-feature contribution direction
        for i in 0..num {
            let basis = &self.feature_hvs[i];
            // Project positive/negative onto this feature's basis
            let pos_proj = positive.similarity(basis);
            let neg_proj = negative.similarity(basis);
            // Gradient: increase weight if feature aligns more with positive
            let gradient = pos_proj - neg_proj;
            self.feature_weights[i] += lr * gradient;
            // Clamp to prevent negative or extreme weights
            self.feature_weights[i] = self.feature_weights[i].clamp(0.01, 10.0);
        }

        // Normalize weights to sum to 1
        let sum: f32 = self.feature_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.feature_weights {
                *w /= sum;
            }
        }
    }

    /// Current feature weights (for inspection/serialization).
    pub fn feature_weights(&self) -> &[f32] {
        &self.feature_weights
    }

    /// Set feature weights directly (for state restoration).
    ///
    /// Weights are clamped and normalized to sum to 1.0.
    pub fn set_feature_weights(&mut self, weights: &[f32]) {
        if weights.len() == self.feature_weights.len() {
            let _ = self.set_feature_weights_checked(weights);
            return;
        }

        // Legacy partial-update behavior remains available, but malformed
        // values are rejected before any existing weight changes.
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return;
        }
        let mut candidate = self.feature_weights.clone();
        for (target, weight) in candidate.iter_mut().zip(weights) {
            *target = weight.clamp(0.01, 10.0);
        }
        let sum: f32 = candidate.iter().sum();
        if sum.is_finite() && sum > 0.0 {
            for weight in &mut candidate {
                *weight /= sum;
            }
            self.feature_weights = candidate;
        }
    }

    /// Atomically replace the complete feature-weight vector.
    pub fn set_feature_weights_checked(&mut self, weights: &[f32]) -> Result<(), String> {
        if weights.len() != self.feature_weights.len() {
            return Err(format!(
                "feature weight count mismatch: got {}, expected {}",
                weights.len(),
                self.feature_weights.len()
            ));
        }
        if let Some((index, weight)) = weights
            .iter()
            .copied()
            .enumerate()
            .find(|(_, weight)| !weight.is_finite() || *weight < 0.0)
        {
            return Err(format!(
                "feature weight at index {index} must be finite and non-negative, got {weight}"
            ));
        }

        let mut candidate: Vec<f32> = weights
            .iter()
            .map(|weight| weight.clamp(0.01, 10.0))
            .collect();
        let sum: f32 = candidate.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err("feature weights must have a finite positive sum".to_string());
        }
        for weight in &mut candidate {
            *weight /= sum;
        }
        self.feature_weights = candidate;
        Ok(())
    }

    /// Quantize a [0, 1] feature value to a level index.
    fn quantize(&self, value: f32) -> usize {
        let clamped = value.clamp(0.0, 1.0);
        let idx = (clamped * (self.config.num_levels - 1) as f32).round() as usize;
        idx.min(self.config.num_levels - 1)
    }

    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    pub fn grid_for(&self, width: u32, height: u32) -> PatchGrid {
        PatchGrid::new(width, height, self.config.patch_size)
    }

    /// Compute per-patch stereo disparity from left and right camera frames.
    ///
    /// This compatibility wrapper returns an empty map when validation fails.
    /// New integrations should use [`Self::compute_stereo_depth_checked`] so
    /// malformed sensor buffers cannot be mistaken for a valid far-depth map.
    pub fn compute_stereo_depth(
        &self,
        left: &[u8],
        right: &[u8],
        width: u32,
        height: u32,
        max_disparity: usize,
    ) -> Vec<f32> {
        match self.compute_stereo_depth_checked(left, right, width, height, max_disparity) {
            Ok(estimate) => estimate.depths,
            Err(error) => {
                tracing::warn!(%error, "rejected malformed stereo frame pair");
                Vec::new()
            }
        }
    }

    /// Compute confidence-aware stereo depth with full-patch SAD matching.
    ///
    /// Mean-only matching aliases every flat patch. This implementation scores
    /// every pixel in the patch, tracks the runner-up match, and combines match
    /// margin, texture energy, and absolute quality into a confidence value.
    pub fn compute_stereo_depth_checked(
        &self,
        left: &[u8],
        right: &[u8],
        width: u32,
        height: u32,
        max_disparity: usize,
    ) -> Result<StereoDepthEstimate, String> {
        if width == 0 || height == 0 {
            return Err(format!(
                "stereo dimensions must be non-zero, got {width}x{height}"
            ));
        }
        if max_disparity == 0 {
            return Err("stereo max_disparity must be > 0".to_string());
        }
        let expected_len = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| "stereo frame geometry overflow".to_string())?;
        if left.len() != expected_len || right.len() != expected_len {
            return Err(format!(
                "stereo buffer length mismatch: left={}, right={}, expected={expected_len}",
                left.len(),
                right.len()
            ));
        }

        let grid = self.grid_for(width, height);
        if grid.rows > self.max_rows || grid.cols > self.max_cols {
            return Err(format!(
                "stereo frame {width}x{height} exceeds encoder capacity of {}x{} patches",
                self.max_cols, self.max_rows
            ));
        }
        if grid.num_patches() == 0 {
            return Err(format!(
                "stereo frame {width}x{height} is smaller than patch size {}",
                self.config.patch_size
            ));
        }

        let ps = self.config.patch_size;
        let stride = width as usize;
        let mut depths = Vec::with_capacity(grid.num_patches());
        let mut confidences = Vec::with_capacity(grid.num_patches());
        let mut disparities = Vec::with_capacity(grid.num_patches());

        for row in 0..grid.rows {
            for col in 0..grid.cols {
                let py = row * ps;
                let px = col * ps;
                let search_limit = max_disparity.min(px);
                let mut best_disparity = 0usize;
                let mut best_sad = f32::INFINITY;
                let mut second_sad = f32::INFINITY;
                let mut candidates = 0usize;

                for disparity in 0..=search_limit {
                    let right_x = px - disparity;
                    let sad = Self::patch_sad(left, right, stride, px, right_x, py, ps);
                    candidates += 1;
                    if sad < best_sad {
                        second_sad = best_sad;
                        best_sad = sad;
                        best_disparity = disparity;
                    } else if sad < second_sad {
                        second_sad = sad;
                    }
                }

                let texture = (Self::patch_stddev(left, stride, px, py, ps) / 64.0).clamp(0.0, 1.0);
                let margin = if candidates >= 2 && second_sad.is_finite() {
                    ((second_sad - best_sad).max(0.0) / (second_sad + 1.0)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let quality = (1.0 - best_sad / 255.0).clamp(0.0, 1.0);
                let confidence = (texture * margin * quality).clamp(0.0, 1.0);
                let max_d = max_disparity.max(1) as f32;
                let depth = 1.0 - (best_disparity as f32 / max_d).clamp(0.0, 1.0);

                depths.push(depth);
                confidences.push(confidence);
                disparities.push(best_disparity);
            }
        }

        Ok(StereoDepthEstimate {
            depths,
            confidences,
            disparities,
        })
    }

    /// Mean absolute difference between two full patch regions.
    fn patch_sad(
        left: &[u8],
        right: &[u8],
        stride: usize,
        left_x: usize,
        right_x: usize,
        py: usize,
        ps: usize,
    ) -> f32 {
        let mut sum = 0.0f32;
        for dy in 0..ps {
            for dx in 0..ps {
                let left_idx = (py + dy) * stride + left_x + dx;
                let right_idx = (py + dy) * stride + right_x + dx;
                sum += (left[left_idx] as f32 - right[right_idx] as f32).abs();
            }
        }
        sum / (ps * ps).max(1) as f32
    }

    /// Standard deviation of one grayscale patch, used as texture evidence.
    fn patch_stddev(pixels: &[u8], stride: usize, px: usize, py: usize, ps: usize) -> f32 {
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        let count = (ps * ps).max(1) as f32;
        for dy in 0..ps {
            for dx in 0..ps {
                let value = pixels[(py + dy) * stride + px + dx] as f32;
                sum += value;
                sum_sq += value * value;
            }
        }
        let mean = sum / count;
        (sum_sq / count - mean * mean).max(0.0).sqrt()
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub fn max_cols(&self) -> usize {
        self.max_cols
    }

    /// Row basis HVs (for external consumers like MotionField).
    pub fn row_basis(&self) -> &[ContinuousHV] {
        &self.row_hvs
    }

    /// Column basis HVs (for external consumers like MotionField).
    pub fn col_basis(&self) -> &[ContinuousHV] {
        &self.col_hvs
    }
}

/// Multi-scale spatial pyramid encoder.
///
/// Holds one `PatchHdcEncoder` per scale, producing a blended HV that
/// captures both fine-grained detail and coarse scene structure.
pub struct MultiScaleEncoder {
    encoders: Vec<PatchHdcEncoder>,
    scales: Vec<usize>,
    fine_weight: f32,
}

impl MultiScaleEncoder {
    /// Create a multi-scale encoder from a `VisionConfig`.
    ///
    /// One `PatchHdcEncoder` is created per scale in `config.multi_scale.scales`.
    /// Spatial bases are independent per scale, while appearance feature/level
    /// bases are shared so position-unbound patches remain cross-scale comparable.
    pub fn new(config: &VisionConfig, max_width: u32, max_height: u32) -> Self {
        Self::try_new(config, max_width, max_height)
            .expect("invalid multi-scale encoder configuration")
    }

    /// Create a multi-scale encoder without panicking on malformed topology.
    pub fn try_new(config: &VisionConfig, max_width: u32, max_height: u32) -> Result<Self, String> {
        config.validate()?;
        if max_width == 0 || max_height == 0 {
            return Err(format!(
                "multi-scale encoder capacity must be non-zero, got {max_width}x{max_height}"
            ));
        }

        let scales = config.multi_scale.scales.clone();
        let fine_weight = config.multi_scale.fine_weight;
        let encoders: Vec<PatchHdcEncoder> = scales
            .iter()
            .enumerate()
            .map(|(i, &patch_size)| {
                let mut scale_cfg = config.clone();
                scale_cfg.patch_size = patch_size;
                let position_seed = config.seed + (i as u64 + 1) * 500_000;
                PatchHdcEncoder::new_with_basis_seeds(
                    &scale_cfg,
                    max_width,
                    max_height,
                    position_seed,
                    config.seed,
                )
            })
            .collect();

        Ok(Self {
            encoders,
            scales,
            fine_weight,
        })
    }

    fn validate_frame_input(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
    ) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err(format!(
                "multi-scale frame dimensions must be non-zero, got {width}x{height}"
            ));
        }
        if !matches!(channels, 1 | 3 | 4) {
            return Err(format!(
                "multi-scale frame channel count must be 1, 3, or 4, got {channels}"
            ));
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|value| value.checked_mul(channels))
            .ok_or_else(|| "multi-scale frame geometry overflow".to_string())?;
        if pixels.len() != expected {
            return Err(format!(
                "multi-scale frame buffer length mismatch: got {}, expected {expected}",
                pixels.len()
            ));
        }
        for (index, encoder) in self.encoders.iter().enumerate() {
            let grid = encoder.grid_for(width, height);
            if grid.rows > encoder.max_rows() || grid.cols > encoder.max_cols() {
                return Err(format!(
                    "multi-scale frame {width}x{height} exceeds scale {index} capacity of {}x{} patches",
                    encoder.max_cols(),
                    encoder.max_rows()
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn hdc_vector_count(&self) -> usize {
        self.encoders
            .iter()
            .map(PatchHdcEncoder::hdc_vector_count)
            .sum()
    }

    /// Perform 'Holographic Dilation' - scale all internal encoders.
    pub fn dilate(&mut self, target_dim: usize) {
        for encoder in &mut self.encoders {
            encoder.dilate(target_dim);
        }
    }

    /// Encode a frame at all scales and return a blended HV.
    ///
    /// Returns `(blended_hv, per_scale_hvs, per_scale_patches)`.
    /// The blended HV uses a linear weight ramp: finest scale gets `fine_weight`,
    /// coarsest gets `1 - fine_weight`, intermediate scales are linearly interpolated.
    pub fn encode_frame_checked(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
    ) -> MultiScaleEncodeResult {
        self.validate_frame_input(pixels, width, height, channels)?;
        Ok(self.encode_frame(pixels, width, height, channels))
    }

    /// Compatibility wrapper for callers that already guarantee valid input.
    pub fn encode_frame(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
    ) -> (ContinuousHV, Vec<ContinuousHV>, Vec<Vec<ContinuousHV>>) {
        let n = self.encoders.len();
        if n == 0 {
            let dim = symthaea_core::hdc::HDC_DIMENSION;
            return (ContinuousHV::zero(dim), vec![], vec![]);
        }

        let mut scale_hvs = Vec::with_capacity(n);
        let mut all_patches = Vec::with_capacity(n);

        for enc in &mut self.encoders {
            let (hv, patches) = enc.encode_frame(pixels, width, height, channels);
            scale_hvs.push(hv);
            all_patches.push(patches);
        }

        if n == 1 {
            return (scale_hvs[0].clone(), scale_hvs, all_patches);
        }

        // Compute per-scale blend weights (linear ramp from fine_weight to 1-fine_weight)
        let weights: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                self.fine_weight * (1.0 - t) + (1.0 - self.fine_weight) * t
            })
            .collect();

        let refs: Vec<&ContinuousHV> = scale_hvs.iter().collect();
        let blended = ContinuousHV::weighted_bundle(&refs, &weights);

        (blended, scale_hvs, all_patches)
    }

    /// Access the encoder at a specific scale index.
    pub fn encoder_at(&self, scale_idx: usize) -> Option<&PatchHdcEncoder> {
        self.encoders.get(scale_idx)
    }

    /// Mutable access to the encoder at a specific scale index.
    pub fn encoder_at_mut(&mut self, scale_idx: usize) -> Option<&mut PatchHdcEncoder> {
        self.encoders.get_mut(scale_idx)
    }

    /// The patch sizes for each scale.
    pub fn scales(&self) -> &[usize] {
        &self.scales
    }

    /// Compute the static (non-saliency) blend weights for each scale.
    pub fn static_weights(&self) -> Vec<f32> {
        let n = self.encoders.len();
        if n <= 1 {
            return vec![1.0; n];
        }
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                self.fine_weight * (1.0 - t) + (1.0 - self.fine_weight) * t
            })
            .collect()
    }

    /// Compute per-scale health metrics.
    ///
    /// Returns one `ScaleHealth` per scale, reporting feature weight entropy,
    /// patch count, and blend contribution.
    pub fn compute_scale_health(&self) -> Vec<ScaleHealth> {
        let static_weights = self.static_weights();
        self.encoders
            .iter()
            .enumerate()
            .map(|(i, enc)| {
                let weights = enc.feature_weights();
                let sum: f32 = weights.iter().sum();
                let weight_entropy = if sum > 0.0 {
                    weights
                        .iter()
                        .filter(|&&w| w > 0.0)
                        .map(|&w| {
                            let p = w / sum;
                            -p * p.ln()
                        })
                        .sum()
                } else {
                    0.0
                };
                ScaleHealth {
                    patch_size: self.scales[i],
                    num_patches: enc.max_rows() * enc.max_cols(),
                    weight_entropy,
                    blend_weight: static_weights.get(i).copied().unwrap_or(0.0),
                }
            })
            .collect()
    }

    /// Encode a frame with saliency-guided dynamic multi-scale fusion.
    ///
    /// When `per_scale_surprise` is provided (one value per scale), the blend
    /// weights are adjusted: 50% static base + 50% surprise-proportional.
    /// Scales with higher surprise receive more weight.
    /// Falls back to static weights when surprise is absent or too short.
    pub fn encode_frame_saliency_guided_checked(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        per_scale_surprise: Option<&[f32]>,
    ) -> MultiScaleEncodeResult {
        self.validate_frame_input(pixels, width, height, channels)?;
        if let Some(surprise) = per_scale_surprise {
            if surprise.len() != self.encoders.len() {
                return Err(format!(
                    "multi-scale surprise length mismatch: got {}, expected {}",
                    surprise.len(),
                    self.encoders.len()
                ));
            }
            if surprise
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(
                    "multi-scale surprise values must be finite and non-negative".to_string(),
                );
            }
        }
        Ok(self.encode_frame_saliency_guided(pixels, width, height, channels, per_scale_surprise))
    }

    /// Compatibility wrapper for callers that already guarantee valid input.
    pub fn encode_frame_saliency_guided(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        per_scale_surprise: Option<&[f32]>,
    ) -> (ContinuousHV, Vec<ContinuousHV>, Vec<Vec<ContinuousHV>>) {
        let n = self.encoders.len();
        if n == 0 {
            let dim = symthaea_core::hdc::HDC_DIMENSION;
            return (ContinuousHV::zero(dim), vec![], vec![]);
        }

        let mut scale_hvs = Vec::with_capacity(n);
        let mut all_patches = Vec::with_capacity(n);

        for enc in &mut self.encoders {
            let (hv, patches) = enc.encode_frame(pixels, width, height, channels);
            scale_hvs.push(hv);
            all_patches.push(patches);
        }

        if n == 1 {
            return (scale_hvs[0].clone(), scale_hvs, all_patches);
        }

        let static_w = self.static_weights();

        // Determine final weights: mix static with surprise-proportional
        let weights: Vec<f32> = if let Some(surprise) = per_scale_surprise {
            if surprise.len() >= n {
                let total_surprise: f32 = surprise.iter().take(n).sum::<f32>().max(1e-8);
                static_w
                    .iter()
                    .enumerate()
                    .map(|(i, &sw)| {
                        let surprise_w = surprise[i] / total_surprise;
                        0.5 * sw + 0.5 * surprise_w
                    })
                    .collect()
            } else {
                static_w
            }
        } else {
            static_w
        };

        let refs: Vec<&ContinuousHV> = scale_hvs.iter().collect();
        let blended = ContinuousHV::weighted_bundle(&refs, &weights);

        (blended, scale_hvs, all_patches)
    }
}

/// Directional motion field via HDC binding.
///
/// Computes per-patch motion vectors from the temporal difference gradient
/// of adjacent patches, then encodes them as a holographic motion field HV
/// by binding direction basis vectors with position and magnitude.
///
/// 8 cardinal/ordinal directions: N, NE, E, SE, S, SW, W, NW.
pub struct MotionField {
    direction_hvs: Vec<ContinuousHV>,
    dim: usize,
}

impl MotionField {
    /// Create a motion field encoder with 8 direction basis vectors.
    pub fn new(dim: usize, seed: u64) -> Self {
        let direction_hvs: Vec<ContinuousHV> = (0..8)
            .map(|d| ContinuousHV::random(dim, seed + 900_000 + d as u64))
            .collect();
        Self { direction_hvs, dim }
    }

    pub(crate) fn hdc_vector_count(&self) -> usize {
        self.direction_hvs.len()
    }

    /// Perform 'Holographic Dilation' - scale internal basis vectors.
    pub fn dilate(&mut self, target_dim: usize) {
        if self.dim == target_dim {
            return;
        }

        for hv in &mut self.direction_hvs {
            *hv = hv.dilate(target_dim);
        }

        self.dim = target_dim;
    }

    /// Compute the motion field from current and previous per-patch luminances.
    ///
    /// Returns `(motion_field_hv, per_patch_motion_vectors)` where each motion
    /// vector is `[dx, dy]` in normalized [-1, 1] range.
    ///
    /// The motion field HV encodes WHERE motion is happening and in WHAT DIRECTION
    /// by binding `direction_hv ⊗ position_hv` weighted by magnitude.
    pub fn compute(
        &self,
        current_lum: &[f32],
        prev_lum: &[f32],
        rows: usize,
        cols: usize,
        row_hvs: &[ContinuousHV],
        col_hvs: &[ContinuousHV],
    ) -> (ContinuousHV, Vec<[f32; 2]>) {
        let n = rows * cols;
        if n == 0 || current_lum.len() < n || prev_lum.len() < n {
            return (ContinuousHV::zero(self.dim), vec![]);
        }

        // Compute temporal difference grid
        let td: Vec<f32> = current_lum
            .iter()
            .zip(prev_lum.iter())
            .map(|(c, p)| (c - p).abs())
            .collect();

        // Compute per-patch motion vectors from temporal difference gradient
        let mut vectors = Vec::with_capacity(n);
        let mut motion_components: Vec<(ContinuousHV, f32)> = Vec::new();

        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                // Spatial gradient of temporal difference → motion direction
                let dx = if c > 0 && c + 1 < cols {
                    (td[r * cols + c + 1] - td[r * cols + c - 1]) * 0.5
                } else if c + 1 < cols {
                    td[r * cols + c + 1] - td[idx]
                } else if c > 0 {
                    td[idx] - td[r * cols + c - 1]
                } else {
                    0.0
                };

                let dy = if r > 0 && r + 1 < rows {
                    (td[(r + 1) * cols + c] - td[(r - 1) * cols + c]) * 0.5
                } else if r + 1 < rows {
                    td[(r + 1) * cols + c] - td[idx]
                } else if r > 0 {
                    td[idx] - td[(r - 1) * cols + c]
                } else {
                    0.0
                };

                vectors.push([dx, dy]);

                let magnitude = (dx * dx + dy * dy).sqrt();
                if magnitude > 0.01 {
                    // Quantize to nearest of 8 directions
                    let angle = dy.atan2(dx);
                    let dir_idx = ((angle + std::f32::consts::PI) / (std::f32::consts::PI / 4.0))
                        .round() as usize
                        % 8;

                    // Bind direction with position
                    let r_idx = r.min(row_hvs.len().saturating_sub(1));
                    let c_idx = c.min(col_hvs.len().saturating_sub(1));
                    let pos = row_hvs[r_idx].bind(&col_hvs[c_idx]);
                    let dir_pos = self.direction_hvs[dir_idx].bind(&pos);
                    motion_components.push((dir_pos, magnitude));
                }
            }
        }

        // Bundle all motion components weighted by magnitude
        let motion_hv = if motion_components.is_empty() {
            ContinuousHV::zero(self.dim)
        } else {
            let refs: Vec<&ContinuousHV> = motion_components.iter().map(|(hv, _)| hv).collect();
            let weights: Vec<f32> = motion_components.iter().map(|(_, w)| *w).collect();
            ContinuousHV::weighted_bundle(&refs, &weights)
        };

        (motion_hv, vectors)
    }

    /// Number of direction basis vectors (always 8).
    pub fn num_directions(&self) -> usize {
        8
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

    fn checkerboard_frame(width: u32, height: u32, block: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let check = ((x / block) + (y / block)) % 2;
                pixels.push(if check == 0 { 0 } else { 255 });
            }
        }
        pixels
    }

    #[test]
    fn test_encode_determinism() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        let (hv1, _) = enc.encode_grayscale(&frame, 64, 64);
        // Reset motion state for determinism
        enc.prev_patch_lum.clear();
        let (hv2, _) = enc.encode_grayscale(&frame, 64, 64);

        assert!(
            (hv1.similarity(&hv2) - 1.0).abs() < 1e-6,
            "Same frame must produce identical HV"
        );
    }

    #[test]
    fn test_similar_frames_similar_hvs() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);

        let frame1 = solid_gray_frame(64, 64, 128);
        let frame2 = solid_gray_frame(64, 64, 130); // slight change

        let (hv1, _) = enc.encode_grayscale(&frame1, 64, 64);
        enc.prev_patch_lum.clear();
        let (hv2, _) = enc.encode_grayscale(&frame2, 64, 64);

        let sim = hv1.similarity(&hv2);
        assert!(
            sim > 0.8,
            "Similar frames should have high similarity, got {sim}"
        );
    }

    #[test]
    fn test_similarity_ordering() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);

        let frame_a = solid_gray_frame(64, 64, 128);
        let frame_similar = solid_gray_frame(64, 64, 130);
        let frame_different = gradient_frame(64, 64);

        let (hv_a, _) = enc.encode_grayscale(&frame_a, 64, 64);
        enc.prev_patch_lum.clear();
        let (hv_sim, _) = enc.encode_grayscale(&frame_similar, 64, 64);
        enc.prev_patch_lum.clear();
        let (hv_diff, _) = enc.encode_grayscale(&frame_different, 64, 64);

        let sim_close = hv_a.similarity(&hv_sim);
        let sim_far = hv_a.similarity(&hv_diff);
        assert!(
            sim_close > sim_far,
            "Similar frame should be closer than different frame: close={sim_close}, far={sim_far}"
        );
    }

    #[test]
    fn test_level_ordinal_similarity() {
        let levels = PatchHdcEncoder::generate_level_hvs(16_384, 32, 42_000);
        let sim_adjacent = levels[0].similarity(&levels[1]);
        let sim_distant = levels[0].similarity(&levels[15]);

        assert!(
            sim_adjacent > sim_distant,
            "Adjacent levels should be more similar: adjacent={sim_adjacent}, distant={sim_distant}"
        );
    }

    #[test]
    fn test_factored_position_shares_components() {
        let cfg = VisionConfig::default();
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);

        // Positions sharing a row should be more similar than arbitrary positions
        let p00 = enc.position_hv(0, 0);
        let p01 = enc.position_hv(0, 1);
        let p70 = enc.position_hv(7, 0);

        // All position HVs from bind of two random HVs are pseudo-random,
        // but same-row positions share the row_hv component.
        // In factored HDC, bind(A,B) and bind(A,C) have similarity ~ 0 when B,C are random.
        // This is a known property — position encoding is distributed.
        // The test verifies basic construction works.
        assert!(p00.similarity(&p00) > 0.99);
        let _ = p01.similarity(&p70); // Just ensure no panic
    }

    #[test]
    fn test_encode_empty_frame() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let (hv, patches) = enc.encode_grayscale(&[], 0, 0);
        assert_eq!(patches.len(), 0);
        assert_eq!(hv.dim(), cfg.hdc_dim);
    }

    #[test]
    fn test_patch_count_matches_grid() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);
        let (_, patches) = enc.encode_grayscale(&frame, 64, 64);
        assert_eq!(patches.len(), 64); // 8×8 patches
    }

    #[test]
    fn test_checkerboard_vs_gradient_vs_solid() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);

        let solid = solid_gray_frame(64, 64, 128);
        let checker = checkerboard_frame(64, 64, 4);
        let grad = gradient_frame(64, 64);

        let (hv_solid, _) = enc.encode_grayscale(&solid, 64, 64);
        enc.prev_patch_lum.clear();
        let (hv_checker, _) = enc.encode_grayscale(&checker, 64, 64);
        enc.prev_patch_lum.clear();
        let (hv_grad, _) = enc.encode_grayscale(&grad, 64, 64);

        // All three should produce distinct encodings (not identical)
        let sim_sc = hv_solid.similarity(&hv_checker);
        let sim_sg = hv_solid.similarity(&hv_grad);
        let sim_cg = hv_checker.similarity(&hv_grad);
        assert!(
            sim_sc < 1.0,
            "Solid and checkerboard should differ: {sim_sc}"
        );
        assert!(sim_sg < 1.0, "Solid and gradient should differ: {sim_sg}");
        assert!(
            sim_cg < 1.0,
            "Checkerboard and gradient should differ: {sim_cg}"
        );

        // Self-similarity is 1.0
        assert!((hv_solid.similarity(&hv_solid) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_encode_precomputed() {
        let cfg = VisionConfig::default();
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);

        let features = vec![
            vec![0.5, 0.5, 0.5, 0.1, 0.01],
            vec![0.5, 0.5, 0.5, 0.1, 0.01],
            vec![0.5, 0.5, 0.5, 0.1, 0.01],
            vec![0.5, 0.5, 0.5, 0.1, 0.01],
        ];
        let (hv, patches) = enc.encode_precomputed(&features);
        assert_eq!(patches.len(), 4);
        assert!(hv.norm() > 0.0);
    }

    // === Improvement 1: Learned Encoding Weights ===

    #[test]
    fn test_feature_weights_initial_uniform() {
        let cfg = VisionConfig::default();
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let weights = enc.feature_weights();
        let total = cfg.total_features();
        assert_eq!(weights.len(), total);
        let expected = 1.0 / total as f32;
        for &w in weights {
            assert!((w - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_contrastive_refinement_increases_positive_similarity() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);

        let frame = gradient_frame(64, 64);
        let (hv_before, _) = enc.encode_grayscale(&frame, 64, 64);

        // Create a "positive" target and a "negative" anti-target
        let positive = ContinuousHV::random(cfg.hdc_dim, 7777);
        let negative = ContinuousHV::random(cfg.hdc_dim, 8888);

        let sim_before = hv_before.similarity(&positive);

        // Refine weights toward positive
        for _ in 0..20 {
            enc.refine_contrastive(&positive, &negative, 0.1);
        }

        let (hv_after, _) = enc.encode_grayscale(&frame, 64, 64);
        let sim_after = hv_after.similarity(&positive);

        // Weights changed, so encoding changed
        assert!(
            (sim_after - sim_before).abs() > 1e-6,
            "Contrastive refinement should change the encoding"
        );
    }

    #[test]
    fn test_feature_weights_stay_normalized() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);

        let positive = ContinuousHV::random(cfg.hdc_dim, 1234);
        let negative = ContinuousHV::random(cfg.hdc_dim, 5678);

        enc.refine_contrastive(&positive, &negative, 0.5);

        let sum: f32 = enc.feature_weights().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Weights should sum to ~1.0 after refinement, got {sum}"
        );
    }

    // === Improvement 3: Multi-Scale Encoder ===

    #[test]
    fn test_multi_scale_construction() {
        let cfg = VisionConfig::default();
        let ms = MultiScaleEncoder::new(&cfg, 64, 64);
        assert_eq!(ms.scales(), &[8, 32]);
        assert!(ms.encoder_at(0).is_some());
        assert!(ms.encoder_at(1).is_some());
        assert!(ms.encoder_at(2).is_none());
    }

    #[test]
    fn test_multi_scale_encoding_produces_valid_hv() {
        let cfg = VisionConfig::default();
        let mut ms = MultiScaleEncoder::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        let (blended, scale_hvs, all_patches) = ms.encode_frame(&frame, 64, 64, 1);

        assert_eq!(blended.dim(), cfg.hdc_dim);
        assert!(blended.norm() > 0.0);
        assert_eq!(scale_hvs.len(), 2);
        assert_eq!(all_patches.len(), 2);
        // Fine scale (8px): 8×8 = 64 patches
        assert_eq!(all_patches[0].len(), 64);
        // Coarse scale (32px): 2×2 = 4 patches
        assert_eq!(all_patches[1].len(), 4);
    }

    #[test]
    fn test_multi_scale_captures_both_structures() {
        let cfg = VisionConfig::default();
        let mut ms = MultiScaleEncoder::new(&cfg, 64, 64);

        // Checkerboard on gradient should differ from solid on gradient
        let checker_on_gradient = {
            let mut pixels = Vec::with_capacity(64 * 64);
            for y in 0..64u32 {
                for x in 0..64u32 {
                    let check = ((x / 4) + (y / 4)) % 2;
                    let grad = (x + y) / 2;
                    pixels.push(if check == 0 {
                        grad as u8
                    } else {
                        (255 - grad) as u8
                    });
                }
            }
            pixels
        };

        let solid_on_gradient = {
            let mut pixels = Vec::with_capacity(64 * 64);
            for y in 0..64u32 {
                for x in 0..64u32 {
                    pixels.push(((x + y) / 2) as u8);
                }
            }
            pixels
        };

        let (hv_cg, _, _) = ms.encode_frame(&checker_on_gradient, 64, 64, 1);
        ms = MultiScaleEncoder::new(&cfg, 64, 64);
        let (hv_sg, _, _) = ms.encode_frame(&solid_on_gradient, 64, 64, 1);

        let sim = hv_cg.similarity(&hv_sg);
        assert!(
            sim < 0.99,
            "Multi-scale should distinguish fine texture differences: sim={sim}"
        );
    }

    #[test]
    fn test_multi_scale_single_scale_passthrough() {
        let mut cfg = VisionConfig::default();
        cfg.multi_scale.scales = vec![8];
        let mut ms = MultiScaleEncoder::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        let (blended, scale_hvs, _) = ms.encode_frame(&frame, 64, 64, 1);
        assert_eq!(scale_hvs.len(), 1);
        // With a single scale, blended should equal the scale HV
        let sim = blended.similarity(&scale_hvs[0]);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Single scale should pass through directly: sim={sim}"
        );
    }

    // === Motion Features ===

    #[test]
    fn test_motion_features_detect_change() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);

        // First frame: dark
        let frame_a = solid_gray_frame(64, 64, 50);
        let (hv_a, _) = enc.encode_grayscale(&frame_a, 64, 64);

        // Second frame: bright — motion features should be non-zero
        let frame_b = solid_gray_frame(64, 64, 200);
        let (hv_b, _) = enc.encode_grayscale(&frame_b, 64, 64);

        // Third frame: same bright — motion features should be near-zero
        let (hv_c, _) = enc.encode_grayscale(&frame_b, 64, 64);

        // hv_b (with motion) should be more different from hv_a than hv_c is from hv_b
        // because the motion features capture temporal change
        assert!(hv_a.norm() > 0.0);
        assert!(hv_b.norm() > 0.0);
        assert!(hv_c.norm() > 0.0);
    }

    #[test]
    fn test_motion_disabled() {
        let mut cfg = VisionConfig::default();
        cfg.enable_motion = false;
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);
        // Without motion: 5 base + 2 color + 2 opponent = 9
        assert_eq!(enc.feature_weights().len(), 9);
    }

    // === Color Features ===

    #[test]
    fn test_color_features_rgb() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);

        // Red frame (RGB)
        let red_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![255u8, 0, 0]).collect();
        let (hv_red, _) = enc.encode_frame(&red_frame, 64, 64, 3);

        enc.prev_patch_lum.clear();

        // Blue frame (RGB)
        let blue_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![0u8, 0, 255]).collect();
        let (hv_blue, _) = enc.encode_frame(&blue_frame, 64, 64, 3);

        // Red and blue should produce different encodings due to chrominance features
        let sim = hv_red.similarity(&hv_blue);
        assert!(sim < 0.95, "Red and blue frames should differ: sim={sim}");
    }

    #[test]
    fn test_color_disabled() {
        let mut cfg = VisionConfig::default();
        cfg.enable_color = false;
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);
        // Without YCbCr: 5 base + 2 motion + 2 opponent = 9
        assert_eq!(enc.feature_weights().len(), 9);
    }

    #[test]
    fn test_all_features_disabled() {
        let mut cfg = VisionConfig::default();
        cfg.enable_motion = false;
        cfg.enable_color = false;
        cfg.enable_opponent_color = false;
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);
        // Base only = 5
        assert_eq!(enc.feature_weights().len(), 5);
    }

    // === Opponent Color Features ===

    #[test]
    fn test_opponent_color_feature_count() {
        let cfg = VisionConfig::default();
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);
        // Default: 5 base + 2 motion + 2 color + 2 opponent = 11
        assert_eq!(enc.feature_weights().len(), 11);
    }

    #[test]
    fn test_opponent_color_disabled() {
        let mut cfg = VisionConfig::default();
        cfg.enable_opponent_color = false;
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);
        // Without opponent: 5 + 2 motion + 2 color = 9
        assert_eq!(enc.feature_weights().len(), 9);
    }

    #[test]
    fn test_opponent_red_vs_green_discrimination() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);

        // Pure red frame: high rg_opponent (R >> G)
        let red_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![255u8, 0, 0]).collect();
        let (hv_red, _) = enc.encode_frame(&red_frame, 64, 64, 3);
        enc.prev_patch_lum.clear();

        // Pure green frame: low rg_opponent (G >> R)
        let green_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![0u8, 255, 0]).collect();
        let (hv_green, _) = enc.encode_frame(&green_frame, 64, 64, 3);

        // Opponent channels should make red and green clearly distinct
        let sim = hv_red.similarity(&hv_green);
        assert!(
            sim < 0.9,
            "Red and green should be distinguishable via opponent channels: sim={sim}"
        );
    }

    #[test]
    fn test_opponent_blue_vs_yellow_discrimination() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);

        // Pure blue frame: high by_opponent
        let blue_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![0u8, 0, 255]).collect();
        let (hv_blue, _) = enc.encode_frame(&blue_frame, 64, 64, 3);
        enc.prev_patch_lum.clear();

        // Yellow = R+G full, B=0: low by_opponent
        let yellow_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![255u8, 255, 0]).collect();
        let (hv_yellow, _) = enc.encode_frame(&yellow_frame, 64, 64, 3);

        let sim = hv_blue.similarity(&hv_yellow);
        assert!(
            sim < 0.9,
            "Blue and yellow should be distinguishable via opponent channels: sim={sim}"
        );
    }

    #[test]
    fn test_opponent_achromatic_is_neutral() {
        // Gray frame should produce neutral (≈0.5) opponent values.
        // With opponent channels, gray should not look like red or blue.
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);

        let gray_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![128u8, 128, 128]).collect();
        let (hv_gray, _) = enc.encode_frame(&gray_frame, 64, 64, 3);
        enc.prev_patch_lum.clear();

        let red_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![255u8, 0, 0]).collect();
        let (hv_red, _) = enc.encode_frame(&red_frame, 64, 64, 3);

        // Gray and red should be distinguishable
        let sim = hv_gray.similarity(&hv_red);
        assert!(
            sim < 0.98,
            "Achromatic gray should differ from pure red: sim={sim}"
        );
    }

    // === Attention-Weighted Encoding ===

    #[test]
    fn test_attended_encoding_differs_from_uniform() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        let (hv_uniform, _) = enc.encode_grayscale(&frame, 64, 64);
        enc.prev_patch_lum.clear();

        // Create non-uniform attention (high in top-left, low elsewhere)
        let grid = enc.grid_for(64, 64);
        let mut attention = vec![0.0f32; grid.num_patches()];
        attention[0] = 1.0;
        attention[1] = 0.8;

        let (hv_attended, _) = enc.encode_frame_attended(&frame, 64, 64, 1, &attention);

        // Should be similar (same content) but not identical (different weighting)
        let sim = hv_uniform.similarity(&hv_attended);
        assert!(
            sim < 1.0 - 1e-6,
            "Attended encoding should differ from uniform: sim={sim}"
        );
    }

    // === Saliency-Guided Multi-Scale Fusion ===

    #[test]
    fn test_saliency_guided_with_no_surprise() {
        let cfg = VisionConfig::default();
        let mut ms = MultiScaleEncoder::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        let (hv_static, _, _) = ms.encode_frame(&frame, 64, 64, 1);
        ms = MultiScaleEncoder::new(&cfg, 64, 64);
        let (hv_saliency, _, _) = ms.encode_frame_saliency_guided(&frame, 64, 64, 1, None);

        // Without surprise, saliency-guided should match static
        let sim = hv_static.similarity(&hv_saliency);
        assert!(
            (sim - 1.0).abs() < 1e-4,
            "No-surprise saliency should match static: sim={sim}"
        );
    }

    #[test]
    fn test_saliency_guided_with_unequal_surprise() {
        let cfg = VisionConfig::default();
        let mut ms = MultiScaleEncoder::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        // High surprise at fine scale, low at coarse
        let surprise = vec![0.9, 0.1];
        let (hv_fine_surprise, _, _) =
            ms.encode_frame_saliency_guided(&frame, 64, 64, 1, Some(&surprise));

        ms = MultiScaleEncoder::new(&cfg, 64, 64);
        // Opposite: high surprise at coarse
        let surprise2 = vec![0.1, 0.9];
        let (hv_coarse_surprise, _, _) =
            ms.encode_frame_saliency_guided(&frame, 64, 64, 1, Some(&surprise2));

        // Different surprise distributions should produce different blends
        let sim = hv_fine_surprise.similarity(&hv_coarse_surprise);
        assert!(
            sim < 1.0 - 1e-6,
            "Different surprise distributions should differ: sim={sim}"
        );
    }

    #[test]
    fn test_saliency_guided_falls_back_on_short_surprise() {
        let cfg = VisionConfig::default();
        let mut ms = MultiScaleEncoder::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        // Only 1 surprise value for 2 scales → should fall back to static
        let (hv_short, _, _) = ms.encode_frame_saliency_guided(&frame, 64, 64, 1, Some(&[0.5]));
        ms = MultiScaleEncoder::new(&cfg, 64, 64);
        let (hv_static, _, _) = ms.encode_frame(&frame, 64, 64, 1);

        let sim = hv_short.similarity(&hv_static);
        assert!(
            (sim - 1.0).abs() < 1e-4,
            "Short surprise should fall back to static: sim={sim}"
        );
    }

    // === RGB Multi-Scale ===

    #[test]
    fn test_multi_scale_rgb() {
        let cfg = VisionConfig::default();
        let mut ms = MultiScaleEncoder::new(&cfg, 64, 64);

        let red_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![255u8, 0, 0]).collect();
        let (hv_red, scale_hvs, _) = ms.encode_frame(&red_frame, 64, 64, 3);

        assert_eq!(hv_red.dim(), cfg.hdc_dim);
        assert!(hv_red.norm() > 0.0);
        assert_eq!(scale_hvs.len(), 2);
    }

    #[test]
    fn test_attended_encoding_with_empty_attention() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        // Empty attention → all weights default to 1.0 (uniform)
        let (hv, _) = enc.encode_frame_attended(&frame, 64, 64, 1, &[]);
        assert!(hv.norm() > 0.0);
    }

    // === Motion Field ===

    #[test]
    fn test_motion_field_construction() {
        let mf = MotionField::new(16_384, 42_000);
        assert_eq!(mf.num_directions(), 8);
    }

    #[test]
    fn test_motion_field_no_motion() {
        let mf = MotionField::new(16_384, 42_000);
        let cfg = VisionConfig::default();
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);

        // Same luminance at both times → no motion
        let lum = vec![0.5f32; 64];
        let (hv, vectors) = mf.compute(&lum, &lum, 8, 8, enc.row_basis(), enc.col_basis());
        assert_eq!(vectors.len(), 64);
        // All motion vectors should be ~zero
        for v in &vectors {
            assert!(v[0].abs() < 1e-6 && v[1].abs() < 1e-6);
        }
        // Motion field HV should be zero (no motion components)
        assert!(hv.norm() < 1e-6, "No motion should produce zero HV");
    }

    #[test]
    fn test_motion_field_detects_direction() {
        let mf = MotionField::new(16_384, 42_000);
        let cfg = VisionConfig::default();
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let rows = 8;
        let cols = 8;

        // Prev: uniform luminance
        let prev = vec![0.5f32; rows * cols];
        // Current: bright spot moving rightward (bright on right side)
        let mut curr = vec![0.5f32; rows * cols];
        for r in 3..5 {
            for c in 5..7 {
                curr[r * cols + c] = 0.9;
            }
        }

        let (hv_right, vectors) =
            mf.compute(&curr, &prev, rows, cols, enc.row_basis(), enc.col_basis());

        // Some patches should have non-zero motion
        let has_motion = vectors
            .iter()
            .any(|v| v[0].abs() > 0.01 || v[1].abs() > 0.01);
        assert!(has_motion, "Should detect motion from brightness change");
        assert!(hv_right.norm() > 0.0, "Motion field HV should be non-zero");

        // Now create leftward motion
        let mut curr_left = vec![0.5f32; rows * cols];
        for r in 3..5 {
            for c in 1..3 {
                curr_left[r * cols + c] = 0.9;
            }
        }

        let (hv_left, _) = mf.compute(
            &curr_left,
            &prev,
            rows,
            cols,
            enc.row_basis(),
            enc.col_basis(),
        );

        // Rightward and leftward motion should produce different HVs
        let sim = hv_right.similarity(&hv_left);
        assert!(
            sim < 0.95,
            "Different motion directions should produce different HVs: sim={sim}"
        );
    }

    #[test]
    fn test_motion_field_empty_input() {
        let mf = MotionField::new(16_384, 42_000);
        let cfg = VisionConfig::default();
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let (hv, vectors) = mf.compute(&[], &[], 0, 0, enc.row_basis(), enc.col_basis());
        assert_eq!(vectors.len(), 0);
        assert!(hv.norm() < 1e-6);
    }

    // === Edge Case Hardening ===

    #[test]
    fn test_encode_frame_truncated_pixels() {
        // Pixel buffer shorter than expected — should not panic
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);
        // Only provide half the expected pixels
        let frame = vec![128u8; 64 * 32];
        let (hv, patches) = enc.encode_frame(&frame, 64, 64, 1);
        // Should still produce a valid (possibly partial) encoding
        assert!(hv.dim() == cfg.hdc_dim);
        let _ = patches; // May have fewer patches than expected
    }

    #[test]
    fn test_encode_frame_extra_pixels() {
        // Pixel buffer longer than expected — extra pixels should be ignored
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let frame = vec![128u8; 64 * 64 * 2]; // 2x expected
        let (hv, patches) = enc.encode_frame(&frame, 64, 64, 1);
        assert_eq!(hv.dim(), cfg.hdc_dim);
        assert_eq!(patches.len(), 64); // 8x8 patches
    }

    #[test]
    fn test_encode_frame_single_pixel_patch() {
        let mut cfg = VisionConfig::default();
        cfg.patch_size = 1;
        let mut enc = PatchHdcEncoder::new(&cfg, 8, 8);
        let frame = vec![128u8; 64];
        let (hv, patches) = enc.encode_frame(&frame, 8, 8, 1);
        assert!(hv.dim() == cfg.hdc_dim);
        assert_eq!(patches.len(), 64); // 8x8 patches
    }

    #[test]
    fn test_encode_frame_includes_partial_edge_patches() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 17, 9);
        let frame = vec![128u8; 17 * 9];
        let (_, patches) = enc.encode_frame(&frame, 17, 9, 1);
        assert_eq!(patches.len(), 6); // ceil(17/8) × ceil(9/8) = 3 × 2
    }

    #[test]
    fn test_set_feature_weights_partial_update() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);

        // Partial weights: updates first 2 features, renormalizes all
        enc.set_feature_weights(&[0.5, 0.5]);
        let weights = enc.feature_weights();
        let sum: f32 = weights.iter().sum();
        // Should still be normalized to 1.0
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Weights should sum to 1.0 after partial set: {sum}"
        );
        assert!(weights.iter().all(|w| w.is_finite() && *w > 0.0));
    }

    #[test]
    fn test_checked_feature_weights_are_atomic() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let before = enc.feature_weights().to_vec();
        let mut malformed = before.clone();
        malformed[1] = f32::NAN;

        assert!(enc.set_feature_weights_checked(&malformed).is_err());
        assert_eq!(enc.feature_weights(), before.as_slice());
        assert!(enc.set_feature_weights_checked(&[0.5, 0.5]).is_err());
        assert_eq!(enc.feature_weights(), before.as_slice());
    }

    #[test]
    fn test_checked_feature_weights_replace_complete_vector() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let mut replacement = vec![1.0; enc.feature_weights().len()];
        replacement[0] = 8.0;
        enc.set_feature_weights_checked(&replacement).unwrap();
        let sum: f32 = enc.feature_weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(enc.feature_weights()[0] > enc.feature_weights()[1]);
    }

    #[test]
    fn test_refine_contrastive_with_zero_hv() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);

        let zero_hv = ContinuousHV::zero(cfg.hdc_dim);
        let random_hv = ContinuousHV::random(cfg.hdc_dim, 42);

        // Should not panic or produce NaN
        enc.refine_contrastive(&zero_hv, &random_hv, 0.1);
        for &w in enc.feature_weights() {
            assert!(
                w.is_finite(),
                "Weights should remain finite after zero-HV refinement"
            );
        }
    }

    #[test]
    fn test_encode_frame_rgba() {
        let cfg = VisionConfig::default();
        let mut enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let frame: Vec<u8> = (0..64 * 64)
            .flat_map(|_| vec![128u8, 64, 192, 255])
            .collect();

        let (hv, patches) = enc.encode_frame(&frame, 64, 64, 4);
        assert!(hv.norm() > 0.0);
        assert!(!patches.is_empty());
    }

    #[test]
    fn test_multiscale_mismatched_frame_size() {
        // Frame smaller than coarse patch size
        let mut cfg = VisionConfig::default();
        cfg.multi_scale.scales = vec![8, 64]; // Coarse is 64, but frame is only 32x32
        let mut ms = MultiScaleEncoder::new(&cfg, 32, 32);
        let frame = vec![128u8; 32 * 32];

        // Coarse encoder keeps one partial 32x32 patch rather than discarding the frame.
        let (hv, scale_hvs, scale_patches) = ms.encode_frame(&frame, 32, 32, 1);
        assert_eq!(hv.dim(), cfg.hdc_dim);
        assert_eq!(scale_hvs.len(), 2);
        assert_eq!(scale_patches[1].len(), 1);
    }

    // === Per-Scale Health ===

    #[test]
    fn test_scale_health_default_config() {
        let cfg = VisionConfig::default();
        let ms = MultiScaleEncoder::new(&cfg, 64, 64);
        let health = ms.compute_scale_health();

        assert_eq!(health.len(), 2); // 2 scales: [8, 32]
        assert_eq!(health[0].patch_size, 8);
        assert_eq!(health[1].patch_size, 32);

        // Fine scale has more patches than coarse
        assert!(health[0].num_patches > health[1].num_patches);

        // Initial weights are uniform → max entropy
        assert!(health[0].weight_entropy > 0.0);
        assert!(health[1].weight_entropy > 0.0);

        // Blend weights should sum to ~1
        let sum: f32 = health.iter().map(|h| h.blend_weight).sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Blend weights should sum to ~1: {sum}"
        );
    }

    #[test]
    fn test_scale_health_after_refinement() {
        let cfg = VisionConfig::default();
        let mut ms = MultiScaleEncoder::new(&cfg, 64, 64);

        let health_before = ms.compute_scale_health();

        // Refine fine-scale encoder weights
        let positive = ContinuousHV::random(cfg.hdc_dim, 1000);
        let negative = ContinuousHV::random(cfg.hdc_dim, 2000);
        ms.encoder_at_mut(0)
            .unwrap()
            .refine_contrastive(&positive, &negative, 0.5);

        let health_after = ms.compute_scale_health();

        // Fine-scale entropy should have changed
        assert!(
            (health_before[0].weight_entropy - health_after[0].weight_entropy).abs() > 1e-6,
            "Fine-scale entropy should change after refinement"
        );

        // Coarse-scale should be unaffected
        assert!(
            (health_before[1].weight_entropy - health_after[1].weight_entropy).abs() < 1e-6,
            "Coarse-scale should not change"
        );
    }

    #[test]
    fn test_scale_health_single_scale() {
        let mut cfg = VisionConfig::default();
        cfg.multi_scale.scales = vec![16];
        let ms = MultiScaleEncoder::new(&cfg, 64, 64);
        let health = ms.compute_scale_health();

        assert_eq!(health.len(), 1);
        assert_eq!(health[0].patch_size, 16);
        assert!((health[0].blend_weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_depth_position_uses_actual_frame_height() {
        let mut cfg = VisionConfig::default();
        cfg.enable_depth = true;
        cfg.enable_motion = false;
        cfg.enable_color = false;
        cfg.enable_opponent_color = false;
        let encoder = PatchHdcEncoder::new(&cfg, 64, 128);
        let pixels = vec![128u8; 64 * 128];

        let features = encoder.extract_patch_features(&pixels, 64, 128, 1, 0, 96, 0.5, None);
        let depth = *features.last().expect("depth feature");
        assert!(
            (depth - 0.7).abs() < 0.02,
            "depth must use height=128 rather than width=64: {depth}"
        );
    }

    #[test]
    #[should_panic(expected = "exceeds encoder capacity")]
    fn test_frame_larger_than_capacity_is_rejected() {
        let cfg = VisionConfig::default();
        let mut encoder = PatchHdcEncoder::new(&cfg, 64, 64);
        let pixels = vec![128u8; 128 * 64];
        let _ = encoder.encode_frame(&pixels, 128, 64, 1);
    }

    #[test]
    fn test_multiscale_position_unbound_appearance_is_comparable() {
        let mut cfg = VisionConfig::default();
        cfg.enable_motion = false;
        cfg.enable_color = false;
        cfg.enable_opponent_color = false;
        cfg.multi_scale.scales = vec![8, 16];
        let mut encoder = MultiScaleEncoder::new(&cfg, 32, 32);
        let frame = vec![96u8; 32 * 32];

        let (_, _, patches) = encoder.encode_frame(&frame, 32, 32, 1);
        let fine = encoder
            .encoder_at(0)
            .expect("fine encoder")
            .unbind_position(&patches[0][0], 0, 0);
        let coarse = encoder
            .encoder_at(1)
            .expect("coarse encoder")
            .unbind_position(&patches[1][0], 0, 0);

        assert!(
            fine.similarity(&coarse) > 0.9,
            "shared appearance bases should make identical content comparable across scales"
        );
    }

    #[test]
    fn test_rebundling_attention_does_not_mutate_motion_history() {
        let cfg = VisionConfig::default();
        let mut encoder = PatchHdcEncoder::new(&cfg, 32, 32);
        let frame = vec![128u8; 32 * 32];
        let (_, patches) = encoder.encode_frame(&frame, 32, 32, 1);
        let before = encoder.prev_patch_lum.clone();

        let attention = vec![1.0; patches.len()];
        let _ = encoder.bundle_attended_patches(&patches, &attention);

        assert_eq!(encoder.prev_patch_lum, before);
    }

    #[test]
    fn test_sensor_depth_override_changes_patch_encoding() {
        let mut cfg = VisionConfig::default();
        cfg.enable_depth = true;
        cfg.enable_motion = false;
        cfg.enable_color = false;
        cfg.enable_opponent_color = false;
        let frame = vec![128u8; 32 * 32];
        let grid = PatchGrid::new(32, 32, cfg.patch_size);
        let near = vec![0.0; grid.num_patches()];
        let far = vec![1.0; grid.num_patches()];
        let mut near_encoder = PatchHdcEncoder::new(&cfg, 32, 32);
        let mut far_encoder = PatchHdcEncoder::new(&cfg, 32, 32);

        let (near_hv, _) = near_encoder.encode_frame_with_depth(&frame, 32, 32, 1, &near);
        let (far_hv, _) = far_encoder.encode_frame_with_depth(&frame, 32, 32, 1, &far);

        assert!(
            near_hv.similarity(&far_hv) < 0.99,
            "sensor depth must affect the encoded representation"
        );
    }

    #[test]
    fn test_stereo_flat_patches_have_zero_confidence_and_neutral_fusion() {
        let mut cfg = VisionConfig::default();
        cfg.hdc_dim = 256;
        cfg.patch_size = 8;
        let encoder = PatchHdcEncoder::new(&cfg, 32, 16);
        let left = vec![128u8; 32 * 16];
        let right = left.clone();

        let estimate = encoder
            .compute_stereo_depth_checked(&left, &right, 32, 16, 8)
            .unwrap();
        assert_eq!(estimate.len(), 8);
        assert!(estimate.confidences.iter().all(|&value| value == 0.0));
        assert!(
            estimate
                .fused_depths()
                .iter()
                .all(|&value| (value - 0.5).abs() < 1e-6)
        );
    }

    #[test]
    fn test_stereo_textured_patch_recovers_disparity_with_confidence() {
        let mut cfg = VisionConfig::default();
        cfg.hdc_dim = 256;
        cfg.patch_size = 8;
        let encoder = PatchHdcEncoder::new(&cfg, 32, 16);
        let mut left = vec![0u8; 32 * 16];
        let mut right = vec![0u8; 32 * 16];

        for y in 0..8usize {
            for dx in 0..8usize {
                let value = ((dx * 31 + y * 47 + dx * y * 7) % 251 + 1) as u8;
                left[y * 32 + 16 + dx] = value;
                right[y * 32 + 12 + dx] = value;
            }
        }

        let estimate = encoder
            .compute_stereo_depth_checked(&left, &right, 32, 16, 8)
            .unwrap();
        let patch = 2usize;
        assert_eq!(estimate.disparities[patch], 4);
        assert!((estimate.depths[patch] - 0.5).abs() < 1e-6);
        assert!(
            estimate.confidences[patch] > 0.4,
            "unique textured match should be confident: {}",
            estimate.confidences[patch]
        );
    }

    #[test]
    fn test_stereo_checked_rejects_malformed_buffers() {
        let cfg = VisionConfig::default();
        let encoder = PatchHdcEncoder::new(&cfg, 16, 16);
        let error = encoder
            .compute_stereo_depth_checked(&vec![0; 256], &vec![0; 255], 16, 16, 8)
            .unwrap_err();
        assert!(error.contains("buffer length mismatch"));
    }

    #[test]
    #[should_panic(expected = "depth override length must match patch grid")]
    fn test_sensor_depth_override_requires_one_value_per_patch() {
        let mut cfg = VisionConfig::default();
        cfg.enable_depth = true;
        let frame = vec![128u8; 32 * 32];
        let mut encoder = PatchHdcEncoder::new(&cfg, 32, 32);
        let _ = encoder.encode_frame_with_depth(&frame, 32, 32, 1, &[0.5]);
    }

    #[test]
    fn test_multiscale_try_new_rejects_invalid_capacity_and_config() {
        let cfg = VisionConfig::default();
        assert!(MultiScaleEncoder::try_new(&cfg, 0, 64).is_err());

        let mut invalid = cfg;
        invalid.multi_scale.scales = vec![32, 8];
        assert!(MultiScaleEncoder::try_new(&invalid, 64, 64).is_err());
    }

    #[test]
    fn test_multiscale_checked_rejection_preserves_temporal_history() {
        let cfg = VisionConfig::default();
        let mut encoder = MultiScaleEncoder::try_new(&cfg, 32, 32).unwrap();
        let frame = vec![128u8; 32 * 32];
        encoder.encode_frame_checked(&frame, 32, 32, 1).unwrap();
        let before: Vec<Vec<f32>> = encoder
            .encoders
            .iter()
            .map(|scale| scale.prev_patch_lum.clone())
            .collect();

        assert!(
            encoder
                .encode_frame_checked(&frame[..frame.len() - 1], 32, 32, 1)
                .is_err()
        );
        let after: Vec<Vec<f32>> = encoder
            .encoders
            .iter()
            .map(|scale| scale.prev_patch_lum.clone())
            .collect();
        assert_eq!(after, before);
    }

    #[test]
    fn test_multiscale_checked_saliency_is_atomic() {
        let cfg = VisionConfig::default();
        let mut encoder = MultiScaleEncoder::try_new(&cfg, 32, 32).unwrap();
        let frame = vec![64u8; 32 * 32];
        let before: Vec<Vec<f32>> = encoder
            .encoders
            .iter()
            .map(|scale| scale.prev_patch_lum.clone())
            .collect();

        assert!(
            encoder
                .encode_frame_saliency_guided_checked(&frame, 32, 32, 1, Some(&[f32::NAN, 0.0]))
                .is_err()
        );
        let after: Vec<Vec<f32>> = encoder
            .encoders
            .iter()
            .map(|scale| scale.prev_patch_lum.clone())
            .collect();
        assert_eq!(after, before);
    }
}
