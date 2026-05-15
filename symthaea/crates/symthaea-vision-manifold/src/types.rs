// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    /// Enable opponent color features (rg_opponent, by_opponent). Default: true.
    ///
    /// Models V1 double-opponent cells: red–green and blue–yellow channels.
    /// These capture color contrast in a perceptually meaningful basis and
    /// directly improve saliency detection for color-distinctive regions
    /// (e.g., a red apple on a green table).
    ///
    /// Adds 2 features when enabled. Values mapped to [0, 1]:
    /// - `rg_opponent = (mean_r - mean_g + 1.0) / 2.0`
    /// - `by_opponent = (mean_b - 0.5*(mean_r + mean_g) + 1.0) / 2.0`
    pub enable_opponent_color: bool,
    /// Base time constant for CfC dynamics in seconds (default: 0.5).
    ///
    /// Valid range: (0.001, 100.0). Controls how quickly the manifold state
    /// responds to new input vs retaining memory of past states.
    pub tau_base: f32,
    /// Surprise threshold for spatial attention (default: 0.3).
    ///
    /// Valid range: (0.0, 1.0]. Patches with surprise above this threshold
    /// are considered salient and trigger attention-guided processing.
    pub surprise_threshold: f32,
    /// Exponential decay for surprise history (default: 0.9).
    ///
    /// Valid range: (0.0, 1.0). Higher values = longer surprise memory.
    /// Steady-state max surprise ≈ 1.0 / (1.0 - decay).
    pub surprise_decay: f32,
    /// Seed for deterministic basis vector generation.
    pub seed: u64,
    /// Weight of current observation in CfC equilibrium (default: 0.7).
    ///
    /// State blend is derived as `1.0 - input_blend`. Higher values (e.g. 0.9)
    /// favor responsiveness to new input; lower values (e.g. 0.3) increase
    /// temporal persistence/memory. Valid range: [0.1, 0.9].
    pub input_blend: f32,
    /// Enable the predictive coding hierarchy in `observe_frame()` (default: false).
    ///
    /// When enabled, a two-level predictive coding hierarchy (Rao & Ballard 1999)
    /// processes each frame: coarse scale predicts fine scale, and cross-scale
    /// prediction errors are injected into the surprise map as additional free-energy
    /// signal. Adds ~2x compute cost per frame.
    pub enable_predictive_hierarchy: bool,
    /// Enable temporal patch binding in CfC equilibrium (default: false).
    ///
    /// When enabled, consecutive patch HVs are bound with cyclic permutation:
    /// `temporal_patch[i] = ρ(prev_patch[i]) ⊗ curr_patch[i]`
    ///
    /// This gives the manifold temporal identity — the same object at the same
    /// location across frames produces consistent HVs even under minor appearance
    /// variation. Non-commutativity (`A⊗B ≠ B⊗A`) encodes temporal direction.
    ///
    /// Reference: Plate (1995) holographic reduced representations;
    /// Kanerva (2009) hyperdimensional computing.
    pub enable_temporal_binding: bool,
    /// Enable depth channel per patch (default: false).
    ///
    /// When enabled, adds 1 feature per patch representing estimated depth
    /// (0.0 = near, 1.0 = far). Without a depth sensor the encoder uses a
    /// neutral stub value of 0.5. Connect real depth data via
    /// `observe_frame_with_depth()` on the manifold or bridge.
    pub enable_depth: bool,
    /// Enable object-level relational binding in `observe_frame()` (default: false).
    ///
    /// When enabled, the scene HV is computed via object-centric binding:
    /// `scene_hv = bundle(position_hv[centroid] ⊗ object_hv)` for each
    /// spatial cluster, rather than a plain patch bag-of-words. This encodes
    /// *where* each perceptual object is, not just *what* patches are present.
    ///
    /// Patches are grouped into object hypotheses by spatial proximity and
    /// HDC cosine similarity. Each cluster contributes one bound HV.
    pub enable_object_binding: bool,
    /// Learning configuration for adaptive encoding weights.
    pub learning: LearningConfig,
    /// Multi-scale configuration for spatial pyramid encoding.
    pub multi_scale: MultiScaleConfig,
    /// Training configuration for CfC temporal learning.
    pub training: TrainingConfig,
}

impl VisionConfig {
    /// Total number of features per patch (base + motion + color + opponent).
    pub fn total_features(&self) -> usize {
        let mut n = self.num_features;
        if self.enable_motion {
            n += 2; // temporal_diff, motion_magnitude
        }
        if self.enable_color {
            n += 2; // mean_cb, mean_cr
        }
        if self.enable_opponent_color {
            n += 2; // rg_opponent, by_opponent (V1 double-opponent cells)
        }
        if self.enable_depth {
            n += 1; // mean_z (depth estimate, stub = 0.5 without a sensor)
        }
        n
    }

    /// Validate config parameters. Returns `Err` with a descriptive message on invalid config.
    ///
    /// Called automatically by `VisionManifold::new()`.
    pub fn validate(&self) -> Result<(), String> {
        if self.hdc_dim == 0 || self.hdc_dim < 256 {
            return Err(format!("hdc_dim must be >= 256, got {}", self.hdc_dim));
        }
        if self.patch_size == 0 || self.patch_size > 64 {
            return Err(format!(
                "patch_size must be in [1, 64], got {}",
                self.patch_size
            ));
        }
        if self.tau_base <= 0.001 || self.tau_base >= 100.0 {
            return Err(format!(
                "tau_base must be in (0.001, 100.0), got {}",
                self.tau_base
            ));
        }
        if self.surprise_threshold <= 0.0 || self.surprise_threshold > 1.0 {
            return Err(format!(
                "surprise_threshold must be in (0.0, 1.0], got {}",
                self.surprise_threshold
            ));
        }
        if self.surprise_decay <= 0.0 || self.surprise_decay >= 1.0 {
            return Err(format!(
                "surprise_decay must be in (0.0, 1.0), got {}",
                self.surprise_decay
            ));
        }
        if self.training.error_threshold <= 0.0 || self.training.error_threshold > 1.0 {
            return Err(format!(
                "training.error_threshold must be in (0.0, 1.0], got {}",
                self.training.error_threshold
            ));
        }
        if self.input_blend < 0.1 || self.input_blend > 0.9 {
            return Err(format!(
                "input_blend must be in [0.1, 0.9], got {}",
                self.input_blend
            ));
        }
        if self.training.learning_rate <= 0.0 || self.training.learning_rate > 1.0 {
            return Err(format!(
                "training.learning_rate must be in (0.0, 1.0], got {}",
                self.training.learning_rate
            ));
        }
        if self.training.grad_clip <= 0.0 {
            return Err(format!(
                "training.grad_clip must be > 0.0, got {}",
                self.training.grad_clip
            ));
        }
        if self.num_features < 3 || self.num_features > 20 {
            return Err(format!(
                "num_features must be in [3, 20], got {}",
                self.num_features
            ));
        }
        if self.multi_scale.scales.is_empty() {
            return Err("multi_scale.scales must be non-empty".to_string());
        }
        for &s in &self.multi_scale.scales {
            if s == 0 {
                return Err("multi_scale.scales must all be > 0".to_string());
            }
        }
        Ok(())
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
            enable_opponent_color: true,
            tau_base: 0.5,
            surprise_threshold: 0.3,
            surprise_decay: 0.9,
            seed: 42_000,
            input_blend: 0.7,
            enable_predictive_hierarchy: false,
            enable_temporal_binding: false,
            enable_depth: false,
            enable_object_binding: false,
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
        Self {
            cols,
            rows,
            patch_size,
            frame_width,
            frame_height,
        }
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
    /// Maximum per-patch motion magnitude this frame (0 = no motion detected).
    pub motion_surprise: f32,
    /// Norm of the holographic motion field HV.
    pub motion_field_norm: f32,
    /// Output HV norm (bridge diagnostic).
    pub output_hv_norm: f32,
    /// Attention boost applied (bridge diagnostic).
    pub attention_boost_applied: f32,
    /// Cross-scale prediction error from the predictive coding hierarchy (if enabled).
    pub cross_scale_prediction_error: f32,
    /// Whether the current scene was recognized from scene memory.
    #[serde(default)]
    pub scene_recognized: bool,
    /// Cosine similarity of the scene recognition match (0.0 if no match).
    #[serde(default)]
    pub scene_recognition_similarity: f32,
    /// Temporal imagination surprise: prediction error between dream_ahead(1)
    /// and actual observation. 0 = perfect prediction, 1 = maximum surprise.
    /// Only populated when imagination comparison is active on the bridge.
    #[serde(default)]
    pub imagination_surprise: f32,
    /// Number of objects currently held in visual working memory.
    #[serde(default)]
    pub working_memory_load: usize,
    /// Number of spatial relations in the scene graph.
    #[serde(default)]
    pub scene_graph_edges: usize,
    /// Variational Free Energy (F = Complexity - Accuracy).
    #[serde(default)]
    pub free_energy: f32,
    /// Model complexity (posterior-prior distance).
    #[serde(default)]
    pub complexity: f32,
    /// Prediction accuracy (1 - error).
    #[serde(default)]
    pub accuracy: f32,
    /// Latest geodesic path on the manifold (mental simulation path).
    /// Stores a sequence of state hypervector values.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub last_geodesic_path: Vec<Vec<f32>>,
    /// Thermodynamic cost of the last geodesic computation.
    #[serde(default)]
    pub last_geodesic_cost: f32,
    /// Number of steps in the last geodesic path.
    #[serde(default)]
    pub last_geodesic_length: usize,
    /// Latest cognitive action selected by the FEP agent.
    #[serde(default)]
    pub last_fep_action: String,
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
        Self {
            values: vec![0.0; n],
            grid,
        }
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

/// A salient region identified by the dorsal stream with pixel coordinates.
///
/// Used by the foveation bridge to know where to crop high-res regions
/// for ventral stream analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalientRegion {
    /// Grid row in the PatchGrid.
    pub grid_row: usize,
    /// Grid col in the PatchGrid.
    pub grid_col: usize,
    /// Surprise value (higher = more unexpected).
    pub surprise: f32,
    /// Pixel X coordinate of the region's top-left corner.
    pub pixel_x: usize,
    /// Pixel Y coordinate of the region's top-left corner.
    pub pixel_y: usize,
    /// Width in pixels.
    pub pixel_w: usize,
    /// Height in pixels.
    pub pixel_h: usize,
}

/// Serializable snapshot of the manifold's learned state.
///
/// Captures everything needed to resume from a trained checkpoint:
/// weight_hv, tau_base, feature_weights, training step count, error_ema,
/// prediction_error, frame_count, and optionally scene memory.
///
/// # Example: round-trip checkpoint/resume
///
/// ```ignore
/// let saved = manifold.save_state();
/// let json = serde_json::to_string(&saved).unwrap();
/// // ... persist to disk ...
/// let loaded: ManifoldState = serde_json::from_str(&json).unwrap();
/// manifold2.load_state(&loaded).unwrap();
/// // manifold2 now has the same learned weights, tau, error_ema, and frame count
/// ```
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
    /// Exponential moving average of prediction error (for adaptive training trigger).
    #[serde(default)]
    pub error_ema: f32,
    /// Current prediction error state.
    #[serde(default)]
    pub prediction_error: f32,
    /// Frame count to resume numbering.
    #[serde(default)]
    pub frame_count: u64,
    /// Previous patch luminances for motion field on first frame after load.
    #[serde(default)]
    pub prev_patch_lum: Option<Vec<f32>>,
    /// Scene memory snapshot (if any).
    #[serde(default)]
    pub scene_memory: Option<SceneMemoryState>,
}

/// Serializable snapshot of scene memory landmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneMemoryState {
    /// Stored landmarks: (hv_values, stored_at_frame).
    pub landmarks: Vec<(Vec<f32>, u64)>,
    /// Maximum capacity.
    pub capacity: usize,
    /// Recognition similarity threshold.
    pub threshold: f32,
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

/// A perceptual object hypothesis formed by clustering spatially-adjacent patches.
///
/// Used by the object-level binding pipeline (`enable_object_binding = true`).
/// Patches are grouped by spatial proximity and HDC similarity; each cluster
/// produces one `ObjectHypothesis` whose centroid determines the position HV
/// used in the scene binding: `position_hv ⊗ object_hv`.
#[derive(Debug, Clone)]
pub struct ObjectHypothesis {
    /// Row of the cluster centroid in the PatchGrid.
    pub centroid_row: usize,
    /// Column of the cluster centroid in the PatchGrid.
    pub centroid_col: usize,
    /// Patch indices belonging to this cluster.
    pub patch_indices: Vec<usize>,
    /// Mean saliency of patches in this cluster.
    pub saliency: f32,
    /// Object HV: bundle of all member patch HVs (normalized).
    pub hv: symthaea_core::hdc::ContinuousHV,
}

/// Spatial relation between two objects in the scene graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpatialRelation {
    Above,
    Below,
    LeftOf,
    RightOf,
    Near,
    Far,
    Overlapping,
}

impl SpatialRelation {
    /// All supported relations.
    pub const ALL: [Self; 7] = [
        Self::Above,
        Self::Below,
        Self::LeftOf,
        Self::RightOf,
        Self::Near,
        Self::Far,
        Self::Overlapping,
    ];
}

/// An edge in the visual scene graph: subject → relation → object.
///
/// Encoded in HDC as `relation_hv = subject_hv ⊗ relation_basis ⊗ object_hv`,
/// giving a holographic relational triple the cognitive loop can reason over.
#[derive(Debug, Clone)]
pub struct SceneGraphEdge {
    /// Track ID of the subject object.
    pub subject_id: u64,
    /// Track ID of the object.
    pub object_id: u64,
    /// Spatial relation (e.g., Above, LeftOf, Near).
    pub relation: SpatialRelation,
    /// HDC encoding of this relational triple.
    pub relation_hv: symthaea_core::hdc::ContinuousHV,
}

/// Per-scale health metrics for multi-scale encoders.
///
/// Reports the state of each scale in a `MultiScaleEncoder`, including
/// feature weight entropy and blend contribution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScaleHealth {
    /// Patch size at this scale (in pixels).
    pub patch_size: usize,
    /// Number of patches at this scale.
    pub num_patches: usize,
    /// Shannon entropy of encoder feature weights at this scale.
    /// Higher = more uniform weighting.
    pub weight_entropy: f32,
    /// This scale's blend weight in the multi-scale fusion.
    pub blend_weight: f32,
}

/// Variational Free Energy (FEP) metrics for the visual manifold.
///
/// Science: Friston (2010). Free energy (F) = Complexity - Accuracy.
/// Minimizing F is equivalent to maximizing the evidence for the system's world model.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct FepMetrics {
    /// Variational Free Energy (surprise).
    pub free_energy: f32,
    /// Model complexity (KL divergence between posterior and prior).
    /// High complexity = overfitting or over-responsive state.
    pub complexity: f32,
    /// Prediction accuracy (log-likelihood of sensory input given state).
    pub accuracy: f32,
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
        assert!(cfg.enable_opponent_color);
        assert_eq!(cfg.total_features(), 11); // 5 base + 2 motion + 2 color + 2 opponent
        assert!((cfg.input_blend - 0.7).abs() < 1e-6);
        assert!(!cfg.enable_predictive_hierarchy);
    }

    #[test]
    fn test_config_validate_default_passes() {
        let cfg = VisionConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_catches_invalid() {
        let mut cfg = VisionConfig::default();

        cfg.hdc_dim = 0;
        assert!(cfg.validate().unwrap_err().contains("hdc_dim"));
        cfg.hdc_dim = 16_384;

        cfg.patch_size = 0;
        assert!(cfg.validate().unwrap_err().contains("patch_size"));
        cfg.patch_size = 8;

        cfg.tau_base = 0.0;
        assert!(cfg.validate().unwrap_err().contains("tau_base"));
        cfg.tau_base = 0.5;

        cfg.surprise_threshold = 0.0;
        assert!(cfg.validate().unwrap_err().contains("surprise_threshold"));
        cfg.surprise_threshold = 0.3;

        cfg.surprise_decay = 1.0;
        assert!(cfg.validate().unwrap_err().contains("surprise_decay"));
        cfg.surprise_decay = 0.9;

        cfg.input_blend = 0.05;
        assert!(cfg.validate().unwrap_err().contains("input_blend"));
        cfg.input_blend = 0.7;

        cfg.training.learning_rate = 0.0;
        assert!(cfg.validate().unwrap_err().contains("learning_rate"));
        cfg.training.learning_rate = 0.001;

        cfg.training.grad_clip = -1.0;
        assert!(cfg.validate().unwrap_err().contains("grad_clip"));
        cfg.training.grad_clip = 1.0;

        cfg.num_features = 2;
        assert!(cfg.validate().unwrap_err().contains("num_features"));
        cfg.num_features = 5;

        cfg.multi_scale.scales = vec![];
        assert!(cfg.validate().unwrap_err().contains("scales"));
        cfg.multi_scale.scales = vec![8, 32];

        cfg.multi_scale.scales = vec![0];
        assert!(cfg.validate().unwrap_err().contains("scales"));
    }

    #[test]
    fn test_total_features_combinations() {
        let mut cfg = VisionConfig::default();
        assert_eq!(cfg.total_features(), 11); // 5 base + 2 motion + 2 color + 2 opponent

        cfg.enable_opponent_color = false;
        assert_eq!(cfg.total_features(), 9); // 5 + 2 motion + 2 color

        cfg.enable_motion = false;
        assert_eq!(cfg.total_features(), 7); // 5 + 2 color

        cfg.enable_color = false;
        assert_eq!(cfg.total_features(), 5); // base only

        cfg.enable_motion = true;
        assert_eq!(cfg.total_features(), 7); // 5 + 2 motion

        cfg.enable_opponent_color = true;
        assert_eq!(cfg.total_features(), 9); // 5 + 2 motion + 2 opponent
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
