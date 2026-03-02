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
    /// Number of base features extracted per patch (default: 5).
    ///
    /// Total features = base + motion (2, if enabled) + color (2, if enabled).
    pub num_features: usize,
    /// Enable motion features (temporal_diff, motion_magnitude). Default: true.
    pub enable_motion: bool,
    /// Enable color features (mean_cb, mean_cr from YCbCr). Default: true.
    pub enable_color: bool,
    /// Base time constant for CfC dynamics in seconds (default: 0.5).
    pub tau_base: f32,
    /// Surprise threshold for spatial attention (default: 0.3).
    pub surprise_threshold: f32,
    /// Exponential decay for surprise history (default: 0.9).
    pub surprise_decay: f32,
    /// Seed for deterministic basis vector generation.
    pub seed: u64,
    /// Learning configuration for adaptive encoding weights.
    pub learning: LearningConfig,
    /// Multi-scale configuration for spatial pyramid encoding.
    pub multi_scale: MultiScaleConfig,
    /// Training configuration for CfC temporal learning.
    pub training: TrainingConfig,
}

impl VisionConfig {
    /// Total number of features per patch (base + motion + color).
    pub fn total_features(&self) -> usize {
        let mut n = self.num_features;
        if self.enable_motion {
            n += 2; // temporal_diff, motion_magnitude
        }
        if self.enable_color {
            n += 2; // mean_cb, mean_cr
        }
        n
    }
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hdc_dim: symthaea_core::hdc::HDC_DIMENSION,
            patch_size: 8,
            num_levels: 32,
            num_features: 5,
            enable_motion: true,
            enable_color: true,
            tau_base: 0.5,
            surprise_threshold: 0.3,
            surprise_decay: 0.9,
            seed: 42_000,
            learning: LearningConfig::default(),
            multi_scale: MultiScaleConfig::default(),
            training: TrainingConfig::default(),
        }
    }
}

/// Configuration for adaptive/learned encoding weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Enable adaptive level quantization boundaries.
    pub adaptive_levels: bool,
    /// Learning rate for contrastive weight refinement.
    pub contrastive_lr: f32,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            adaptive_levels: false,
            contrastive_lr: 0.01,
        }
    }
}

/// Configuration for multi-scale spatial pyramid encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleConfig {
    /// Patch sizes at each scale (default: [8, 32]).
    pub scales: Vec<usize>,
    /// Blend weight for finest scale (0..1, default: 0.6 = fine-dominant).
    pub fine_weight: f32,
}

impl Default for MultiScaleConfig {
    fn default() -> Self {
        Self {
            scales: vec![8, 32],
            fine_weight: 0.6,
        }
    }
}

/// Training method for CfC temporal dynamics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingMethod {
    /// Analytical gradient through closed-form CfC.
    Bptt,
    /// Zeroth-order Simultaneous Perturbation Stochastic Approximation.
    Spsa,
    /// BPTT with SPSA fallback when gradients are unstable.
    #[default]
    BpttWithSpsaFallback,
}

/// Configuration for CfC temporal training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Base learning rate.
    pub learning_rate: f32,
    /// Scale factor for weight HV learning rate (relative to base).
    pub weight_lr_scale: f32,
    /// Scale factor for tau learning rate (slower, 0.1x default).
    pub tau_lr_scale: f32,
    /// Gradient clipping threshold.
    pub grad_clip: f32,
    /// SPSA perturbation magnitude.
    pub spsa_epsilon: f32,
    /// SPSA gain sequence constant.
    pub spsa_c: f32,
    /// Training method selection.
    pub method: TrainingMethod,
    /// Prediction error threshold to trigger training (adaptive).
    pub error_threshold: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            weight_lr_scale: 1.0,
            tau_lr_scale: 0.1,
            grad_clip: 1.0,
            spsa_epsilon: 0.01,
            spsa_c: 0.1,
            method: TrainingMethod::default(),
            error_threshold: 0.1,
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
    /// Whether a training step was triggered this cycle.
    pub training_triggered: bool,
    /// Training loss after this cycle's training step (if any).
    pub training_loss: Option<f32>,
    /// Output HV norm (bridge diagnostic).
    pub output_hv_norm: f32,
    /// Attention boost applied (bridge diagnostic).
    pub attention_boost_applied: f32,
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

/// Serializable snapshot of the manifold's learned state.
///
/// Captures everything needed to resume from a trained checkpoint:
/// weight_hv, tau_base, feature_weights, and training step count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifoldState {
    /// Learned CfC weight hypervector.
    pub weight_hv: Vec<f32>,
    /// Learned time constant.
    pub tau_base: f32,
    /// Per-feature encoder weights.
    pub feature_weights: Vec<f32>,
    /// Total training steps completed.
    pub training_steps: u64,
    /// Config snapshot for compatibility checking.
    pub hdc_dim: usize,
    /// Number of base features.
    pub num_features: usize,
}

/// Result of a scene recognition query against stored landmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneMatch {
    /// Index of the matched scene in the memory buffer.
    pub scene_id: usize,
    /// Cosine similarity between current state and the matched scene.
    pub similarity: f32,
    /// Frame number at which this scene was last stored.
    pub stored_at_frame: u64,
    /// Number of frames since the scene was last stored.
    pub frames_since_stored: u64,
}

/// Health diagnostics for the vision manifold.
///
/// Tracks drift, stability, and training quality metrics for monitoring.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ManifoldHealth {
    /// Cosine similarity between current weight_hv and initial weight_hv.
    /// Values near 1.0 = minimal drift; low values = significant adaptation.
    pub weight_drift: f32,
    /// Current tau_base value (should stay in [0.01, 10.0]).
    pub tau_value: f32,
    /// Shannon entropy of encoder feature weights (higher = more uniform).
    pub encoder_weight_entropy: f32,
    /// Fraction of recent frames that triggered training.
    pub training_frequency: f32,
    /// Mean prediction error over recent frames.
    pub mean_prediction_error: f32,
    /// Mean coherence over recent frames.
    pub mean_coherence: f32,
    /// Total frames observed.
    pub total_frames: u64,
    /// Total training steps performed.
    pub total_training_steps: u64,
    /// Whether the manifold appears healthy (heuristic).
    pub is_healthy: bool,
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
        assert!(cfg.enable_motion);
        assert!(cfg.enable_color);
        assert_eq!(cfg.total_features(), 9); // 5 base + 2 motion + 2 color
    }

    #[test]
    fn test_total_features_combinations() {
        let mut cfg = VisionConfig::default();
        assert_eq!(cfg.total_features(), 9);

        cfg.enable_motion = false;
        assert_eq!(cfg.total_features(), 7); // 5 + 2 color

        cfg.enable_color = false;
        assert_eq!(cfg.total_features(), 5); // base only

        cfg.enable_motion = true;
        assert_eq!(cfg.total_features(), 7); // 5 + 2 motion
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
