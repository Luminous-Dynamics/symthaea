//! Shared types for the vision manifold pipeline.

use serde::{Deserialize, Serialize};

/// Configuration for the vision manifold pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    /// HDC dimension (default: 16,384).
    pub hdc_dim: usize,
    /// Patch size in pixels (default: 8).
    pub patch_size: usize,
    /// Number of quantization levels for pixel features (default: 32).
    pub num_levels: usize,
    /// Number of features extracted per patch (default: 5).
    pub num_features: usize,
    /// Base time constant for CfC dynamics in seconds (default: 0.5).
    pub tau_base: f32,
    /// Surprise threshold for spatial attention (default: 0.3).
    pub surprise_threshold: f32,
    /// Exponential decay for surprise history (default: 0.9).
    pub surprise_decay: f32,
    /// Seed for deterministic basis vector generation.
    pub seed: u64,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hdc_dim: symthaea_core::hdc::HDC_DIMENSION,
            patch_size: 8,
            num_levels: 32,
            num_features: 5,
            tau_base: 0.5,
            surprise_threshold: 0.3,
            surprise_decay: 0.9,
            seed: 42_000,
        }
    }
}

/// Describes the grid of patches extracted from a frame.
#[derive(Debug, Clone)]
pub struct PatchGrid {
    pub cols: usize,
    pub rows: usize,
    pub patch_size: usize,
    pub frame_width: u32,
    pub frame_height: u32,
}

impl PatchGrid {
    pub fn new(frame_width: u32, frame_height: u32, patch_size: usize) -> Self {
        let cols = frame_width as usize / patch_size.max(1);
        let rows = frame_height as usize / patch_size.max(1);
        Self { cols, rows, patch_size, frame_width, frame_height }
    }

    pub fn num_patches(&self) -> usize {
        self.cols * self.rows
    }

    pub fn patch_index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }
}

/// Telemetry from one manifold observation cycle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VisionTelemetry {
    /// Time spent encoding the frame (microseconds).
    pub encode_time_us: u64,
    /// Time spent evolving the CfC manifold (microseconds).
    pub evolve_time_us: u64,
    /// Prediction error (1 - cosine similarity with predicted frame).
    pub prediction_error: f32,
    /// Manifold coherence (cosine similarity between state and frame encoding).
    pub manifold_coherence: f32,
    /// Shannon entropy of the attention/surprise map.
    pub attention_entropy: f32,
    /// Number of patches exceeding the surprise threshold.
    pub num_salient_patches: usize,
    /// Frame sequence number.
    pub frame_sequence: u64,
}

/// Per-patch spatial attention/surprise map.
#[derive(Debug, Clone)]
pub struct AttentionMap {
    pub values: Vec<f32>,
    pub grid: PatchGrid,
}

impl AttentionMap {
    pub fn new(grid: PatchGrid) -> Self {
        let n = grid.num_patches();
        Self { values: vec![0.0; n], grid }
    }

    /// Get surprise value at a specific grid position.
    pub fn at(&self, row: usize, col: usize) -> f32 {
        let idx = self.grid.patch_index(row, col);
        self.values.get(idx).copied().unwrap_or(0.0)
    }

    /// Return all patches exceeding the given surprise threshold.
    pub fn salient_patches(&self, threshold: f32) -> Vec<(usize, usize, f32)> {
        let mut result = Vec::new();
        for r in 0..self.grid.rows {
            for c in 0..self.grid.cols {
                let v = self.at(r, c);
                if v > threshold {
                    result.push((r, c, v));
                }
            }
        }
        result
    }

    /// Shannon entropy of the surprise distribution.
    pub fn entropy(&self) -> f32 {
        let sum: f32 = self.values.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }
        let mut ent = 0.0f32;
        for &v in &self.values {
            if v > 0.0 {
                let p = v / sum;
                ent -= p * p.ln();
            }
        }
        ent
    }

    pub fn max_surprise(&self) -> f32 {
        self.values.iter().copied().fold(0.0f32, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_config_default() {
        let cfg = VisionConfig::default();
        assert_eq!(cfg.hdc_dim, 16_384);
        assert_eq!(cfg.patch_size, 8);
        assert_eq!(cfg.num_levels, 32);
    }

    #[test]
    fn test_patch_grid() {
        let grid = PatchGrid::new(64, 64, 8);
        assert_eq!(grid.cols, 8);
        assert_eq!(grid.rows, 8);
        assert_eq!(grid.num_patches(), 64);
        assert_eq!(grid.patch_index(2, 3), 19);
    }

    #[test]
    fn test_patch_grid_truncates() {
        // 65x65 with patch_size=8 → 8 cols, 8 rows (last pixel row/col unused)
        let grid = PatchGrid::new(65, 65, 8);
        assert_eq!(grid.cols, 8);
        assert_eq!(grid.rows, 8);
    }

    #[test]
    fn test_attention_map_entropy() {
        let grid = PatchGrid::new(16, 16, 8);
        let mut map = AttentionMap::new(grid);
        // Uniform distribution → max entropy
        map.values = vec![1.0; 4];
        let uniform_ent = map.entropy();

        // Concentrated → lower entropy
        map.values = vec![10.0, 0.0, 0.0, 0.0];
        let concentrated_ent = map.entropy();

        assert!(uniform_ent > concentrated_ent);
    }

    #[test]
    fn test_salient_patches() {
        let grid = PatchGrid::new(16, 16, 8);
        let mut map = AttentionMap::new(grid);
        map.values = vec![0.1, 0.5, 0.2, 0.8];
        let salient = map.salient_patches(0.3);
        assert_eq!(salient.len(), 2);
        assert_eq!(salient[0], (0, 1, 0.5));
        assert_eq!(salient[1], (1, 1, 0.8));
    }
}
