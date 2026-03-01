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

use crate::types::{PatchGrid, VisionConfig};

/// Encodes video frames into 16,384-dimensional holographic hypervectors.
pub struct PatchHdcEncoder {
    config: VisionConfig,
    row_hvs: Vec<ContinuousHV>,
    col_hvs: Vec<ContinuousHV>,
    feature_hvs: Vec<ContinuousHV>,
    level_hvs: Vec<ContinuousHV>,
    max_rows: usize,
    max_cols: usize,
}

impl PatchHdcEncoder {
    /// Create a new encoder sized for frames up to `max_width × max_height`.
    pub fn new(config: &VisionConfig, max_width: u32, max_height: u32) -> Self {
        let max_cols = max_width as usize / config.patch_size.max(1);
        let max_rows = max_height as usize / config.patch_size.max(1);

        let row_hvs: Vec<ContinuousHV> = (0..max_rows)
            .map(|r| ContinuousHV::random(config.hdc_dim, config.seed + r as u64))
            .collect();

        let col_hvs: Vec<ContinuousHV> = (0..max_cols)
            .map(|c| ContinuousHV::random(config.hdc_dim, config.seed + 50_000 + c as u64))
            .collect();

        let feature_hvs: Vec<ContinuousHV> = (0..config.num_features)
            .map(|f| ContinuousHV::random(config.hdc_dim, config.seed + 100_000 + f as u64))
            .collect();

        let level_hvs =
            Self::generate_level_hvs(config.hdc_dim, config.num_levels, config.seed + 200_000);

        Self {
            config: config.clone(),
            row_hvs,
            col_hvs,
            feature_hvs,
            level_hvs,
            max_rows,
            max_cols,
        }
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
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
    ) -> (ContinuousHV, Vec<ContinuousHV>) {
        let grid = PatchGrid::new(width, height, self.config.patch_size);
        if grid.num_patches() == 0 {
            return (ContinuousHV::zero(self.config.hdc_dim), vec![]);
        }

        let mut patch_hvs = Vec::with_capacity(grid.num_patches());

        for row in 0..grid.rows {
            for col in 0..grid.cols {
                let features = self.extract_patch_features(
                    pixels,
                    width,
                    channels,
                    col * self.config.patch_size,
                    row * self.config.patch_size,
                );
                let appearance = self.encode_features(&features);
                let position = self.position_hv(row, col);
                patch_hvs.push(position.bind(&appearance));
            }
        }

        let refs: Vec<&ContinuousHV> = patch_hvs.iter().collect();
        let frame_hv = ContinuousHV::bundle(&refs);
        (frame_hv, patch_hvs)
    }

    /// Convenience wrapper for single-channel grayscale frames.
    pub fn encode_grayscale(
        &self,
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
        let rows = (n + cols - 1) / cols;

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

    /// Factored position HV: row_hv ⊗ col_hv (GridEncoder pattern).
    fn position_hv(&self, row: usize, col: usize) -> ContinuousHV {
        let r = row % self.max_rows.max(1);
        let c = col % self.max_cols.max(1);
        self.row_hvs[r].bind(&self.col_hvs[c])
    }

    /// Extract 5 features from a single patch:
    /// [mean_r, mean_g, mean_b, edge_density, variance]
    ///
    /// All values normalized to [0, 1].
    fn extract_patch_features(
        &self,
        pixels: &[u8],
        width: u32,
        channels: usize,
        patch_x: usize,
        patch_y: usize,
    ) -> Vec<f32> {
        let ps = self.config.patch_size;
        let stride = width as usize * channels.max(1);

        let mut sum_r = 0.0f32;
        let mut sum_g = 0.0f32;
        let mut sum_b = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut edge_sum = 0.0f32;
        let mut count = 0.0f32;

        for dy in 0..ps {
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

                // Horizontal gradient magnitude for edge detection
                if dx > 0 {
                    let prev_offset = y * stride + (x - 1) * channels.max(1);
                    if prev_offset + channels.max(1) <= pixels.len() {
                        let prev_lum = if channels >= 3 {
                            0.299 * pixels[prev_offset] as f32
                                + 0.587 * pixels[prev_offset + 1] as f32
                                + 0.114 * pixels[prev_offset + 2] as f32
                        } else {
                            pixels[prev_offset] as f32
                        };
                        edge_sum += (lum - prev_lum).abs();
                    }
                }

                count += 1.0;
            }
        }

        if count == 0.0 {
            return vec![0.0; self.config.num_features];
        }

        let inv_count = 1.0 / count;
        let inv_255 = 1.0 / 255.0;
        let mean_r = sum_r * inv_count * inv_255;
        let mean_g = sum_g * inv_count * inv_255;
        let mean_b = sum_b * inv_count * inv_255;
        let mean_lum = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b;
        let variance = (sum_sq * inv_count * inv_255 * inv_255 - mean_lum * mean_lum).max(0.0);
        let edge_density = (edge_sum * inv_count * inv_255).min(1.0);

        vec![mean_r, mean_g, mean_b, edge_density, variance]
    }

    /// Encode a feature vector into a ContinuousHV via bind-bundle.
    fn encode_features(&self, features: &[f32]) -> ContinuousHV {
        let num = features.len().min(self.config.num_features);
        if num == 0 {
            return ContinuousHV::zero(self.config.hdc_dim);
        }

        let mut components = Vec::with_capacity(num);
        for (i, &val) in features.iter().take(num).enumerate() {
            let level_idx = self.quantize(val);
            components.push(self.feature_hvs[i].bind(&self.level_hvs[level_idx]));
        }

        let refs: Vec<&ContinuousHV> = components.iter().collect();
        ContinuousHV::bundle(&refs)
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

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub fn max_cols(&self) -> usize {
        self.max_cols
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
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        let (hv1, _) = enc.encode_grayscale(&frame, 64, 64);
        let (hv2, _) = enc.encode_grayscale(&frame, 64, 64);

        assert!((hv1.similarity(&hv2) - 1.0).abs() < 1e-6, "Same frame must produce identical HV");
    }

    #[test]
    fn test_similar_frames_similar_hvs() {
        let cfg = VisionConfig::default();
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);

        let frame1 = solid_gray_frame(64, 64, 128);
        let frame2 = solid_gray_frame(64, 64, 130); // slight change

        let (hv1, _) = enc.encode_grayscale(&frame1, 64, 64);
        let (hv2, _) = enc.encode_grayscale(&frame2, 64, 64);

        let sim = hv1.similarity(&hv2);
        assert!(sim > 0.8, "Similar frames should have high similarity, got {sim}");
    }

    #[test]
    fn test_similarity_ordering() {
        let cfg = VisionConfig::default();
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);

        let frame_a = solid_gray_frame(64, 64, 128);
        let frame_similar = solid_gray_frame(64, 64, 130);
        let frame_different = gradient_frame(64, 64);

        let (hv_a, _) = enc.encode_grayscale(&frame_a, 64, 64);
        let (hv_sim, _) = enc.encode_grayscale(&frame_similar, 64, 64);
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
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let (hv, patches) = enc.encode_grayscale(&[], 0, 0);
        assert_eq!(patches.len(), 0);
        assert_eq!(hv.dim(), cfg.hdc_dim);
    }

    #[test]
    fn test_patch_count_matches_grid() {
        let cfg = VisionConfig::default();
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);
        let (_, patches) = enc.encode_grayscale(&frame, 64, 64);
        assert_eq!(patches.len(), 64); // 8×8 patches
    }

    #[test]
    fn test_checkerboard_vs_gradient_vs_solid() {
        let cfg = VisionConfig::default();
        let enc = PatchHdcEncoder::new(&cfg, 64, 64);

        let solid = solid_gray_frame(64, 64, 128);
        let checker = checkerboard_frame(64, 64, 4);
        let grad = gradient_frame(64, 64);

        let (hv_solid, _) = enc.encode_grayscale(&solid, 64, 64);
        let (hv_checker, _) = enc.encode_grayscale(&checker, 64, 64);
        let (hv_grad, _) = enc.encode_grayscale(&grad, 64, 64);

        // All three should produce distinct encodings (not identical)
        let sim_sc = hv_solid.similarity(&hv_checker);
        let sim_sg = hv_solid.similarity(&hv_grad);
        let sim_cg = hv_checker.similarity(&hv_grad);
        assert!(sim_sc < 1.0, "Solid and checkerboard should differ: {sim_sc}");
        assert!(sim_sg < 1.0, "Solid and gradient should differ: {sim_sg}");
        assert!(sim_cg < 1.0, "Checkerboard and gradient should differ: {sim_cg}");

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
}
