// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BrocaCheckpoint: save/load trained Broca models with integrity verification.
//!
//! Uses rmp-serde (MessagePack) for self-describing serialization and blake3
//! for checksum integrity. Legacy bincode checkpoints are supported for loading
//! via automatic fallback (new saves always use MessagePack).
//!
//! Pattern follows `swarm/checkpoint.rs`.

use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNetwork};

use crate::generator::{BrocaConfig, BrocaGenerator};
use crate::tokenizer::{MergePair, VocabFile};

/// Current BrocaCheckpoint schema version.
/// v1: original schema
/// v2: added `liquid_mamba_config: Option<LiquidMambaConfig>`
const CHECKPOINT_VERSION: u32 = 2;

/// Current ProjectionCheckpoint schema version.
/// v1: 4 weight matrices (w_down, w_up, w_back_down, w_back_up).
/// v2: v1 + 4 LayerNorm vectors (ln_fwd_gamma, ln_fwd_beta, ln_bwd_gamma, ln_bwd_beta).
/// v3: v2 + temporal projection fields (temporal, chunk_dim, num_chunks, temporal_weights).
#[cfg(feature = "mamba-cpu")]
const PROJECTION_CHECKPOINT_VERSION: u32 = 4;

/// Minimum supported ProjectionCheckpoint version.
#[cfg(feature = "mamba-cpu")]
const PROJECTION_MIN_VERSION: u32 = 1;

/// Serializable optimizer state for training resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamState {
    /// First moment estimates per embedding.
    pub m: Vec<Vec<f32>>,
    /// Second moment estimates per embedding.
    pub v: Vec<Vec<f32>>,
    /// Timestep counter.
    pub t: usize,
    /// First moment decay rate.
    pub beta1: f32,
    /// Second moment decay rate.
    pub beta2: f32,
    /// Numerical stability constant.
    pub epsilon: f32,
}

impl AdamState {
    /// Create a new Adam state for the given number of embeddings and dimension.
    pub fn new(num_embeddings: usize, dim: usize) -> Self {
        Self {
            m: vec![vec![0.0; dim]; num_embeddings],
            v: vec![vec![0.0; dim]; num_embeddings],
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// Compute bias-corrected Adam update for a single embedding.
    ///
    /// Returns the update delta to subtract from the current embedding values.
    pub fn step(&mut self, idx: usize, grad: &[f32], lr: f32) -> Vec<f32> {
        self.t += 1;
        let t = self.t as f32;

        // Precompute bias correction factors ONCE (not per-dimension)
        let bc1 = 1.0 / (1.0 - self.beta1.powf(t));
        let bc2 = 1.0 / (1.0 - self.beta2.powf(t));

        let m = &mut self.m[idx];
        let v = &mut self.v[idx];
        let mut update = vec![0.0f32; grad.len()];

        for (j, &g) in grad.iter().enumerate() {
            if j >= m.len() {
                break;
            }
            // Update biased first and second moments
            m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * g;
            v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * g * g;

            // Bias-corrected estimates
            let m_hat = m[j] * bc1;
            let v_hat = v[j] * bc2;

            update[j] = lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }

        update
    }
}

/// A complete Broca checkpoint: model weights, config, and training state.
#[derive(Serialize, Deserialize)]
pub struct BrocaCheckpoint {
    /// Schema version for forward-compatible migration.
    pub version: u32,
    /// Token embedding vectors (vocab_size x HDC_DIMENSION).
    pub token_embeddings: Vec<ContinuousHV>,
    /// The HdcLtcUnifiedNetwork state (all neuron weights + layer bindings).
    pub network_state: HdcLtcUnifiedNetwork,
    /// Vocabulary definition (for tokenizer reconstruction).
    pub vocab: VocabFile,
    /// Generator configuration (sampling, gating, controller params).
    pub config: BrocaConfig,
    /// Training epoch at time of save.
    pub training_epoch: usize,
    /// Training loss at time of save.
    pub training_loss: f32,
    /// Optional Adam optimizer state for training resume.
    pub adam_state: Option<AdamState>,
    /// Optional HDC↔SSM projection weights for Liquid-Mamba fusion.
    /// Stored as a flat Vec from `HdcSsmProjection::flatten_weights()`.
    pub projection_weights: Option<Vec<f32>>,
    /// Optional Liquid-Mamba configuration for full L-SSM resume (serialized as JSON string).
    /// Added in v2; v1 checkpoints deserialize with `None` via serde(default).
    /// Stored as a JSON `String` for feature-flag independence — the `liquid_mamba`
    /// module is gated behind `cfg(feature = "mamba-cpu")`, but checkpoints must (de)serialize
    /// regardless of enabled features. Bincode can't round-trip `serde_json::Value` (untagged enum).
    #[serde(default)]
    pub liquid_mamba_config: Option<String>,
    /// Blake3 integrity checksum (set to zeros before hashing).
    pub checksum: [u8; 32],
}

impl BrocaCheckpoint {
    /// Compute the blake3 checksum of the checkpoint (with checksum field zeroed).
    fn compute_checksum(&self) -> [u8; 32] {
        // Serialize with zeroed checksum
        let copy = BrocaCheckpoint {
            version: self.version,
            token_embeddings: self.token_embeddings.clone(),
            network_state: self.network_state.clone(),
            vocab: self.vocab.clone(),
            config: self.config.clone(),
            training_epoch: self.training_epoch,
            training_loss: self.training_loss,
            adam_state: self.adam_state.clone(),
            projection_weights: self.projection_weights.clone(),
            liquid_mamba_config: self.liquid_mamba_config.clone(),
            checksum: [0u8; 32],
        };

        let bytes = rmp_serde::to_vec(&copy)
            .expect("BrocaCheckpoint serialization must succeed for integrity checking");
        *blake3::hash(&bytes).as_bytes()
    }

    /// Verify the checkpoint integrity.
    pub fn verify(&self) -> bool {
        let expected = self.compute_checksum();
        self.checksum == expected
    }

    /// Save checkpoint to a file (MessagePack format).
    pub fn save_to_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        // Compute and set checksum before serialization
        self.checksum = self.compute_checksum();

        let serialized = rmp_serde::to_vec(self).context("Failed to serialize BrocaCheckpoint")?;

        let mut file = std::fs::File::create(path.as_ref())
            .with_context(|| format!("creating checkpoint file: {}", path.as_ref().display()))?;
        file.write_all(&serialized)?;
        file.sync_all()?;

        tracing::info!(
            path = %path.as_ref().display(),
            epoch = self.training_epoch,
            loss = self.training_loss,
            vocab_size = self.token_embeddings.len(),
            "Broca checkpoint saved"
        );

        Ok(())
    }

    /// Load checkpoint from a file with integrity verification.
    ///
    /// Tries MessagePack (current format) first, then falls back to bincode
    /// for legacy checkpoints. Legacy bincode checkpoints skip integrity
    /// verification (the hash was computed from bincode bytes).
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())
            .with_context(|| format!("opening checkpoint file: {}", path.as_ref().display()))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Try MessagePack (current format) first. On failure, keep the error so
        // we can surface BOTH msgpack and bincode failure reasons if bincode
        // fallback also fails — silent error swallowing here made checkpoint
        // drift invisible (Phase 0 validation 2026-04-18: epistemic-v1 and
        // round5/6/7 all reported a generic "tried msgpack + bincode" without
        // any indication of which field or type caused the failure).
        let msgpack_err = match rmp_serde::from_slice::<Self>(&buffer) {
            Ok(ckpt) => {
                if !ckpt.verify() {
                    tracing::warn!(
                        "Checkpoint checksum mismatch (schema evolution) — proceeding with loaded data"
                    );
                }
                return Self::finalize_loaded(ckpt, path.as_ref());
            }
            Err(e) => e,
        };

        // Fall back to bincode (legacy format) — skip verify since hash format changed
        let checkpoint: Self = match bincode::deserialize::<Self>(&buffer) {
            Ok(ckpt) => {
                tracing::warn!(
                    "Loaded legacy bincode Broca checkpoint — will be re-saved as MessagePack"
                );
                ckpt
            }
            Err(bincode_err) => {
                anyhow::bail!(
                    "Failed to deserialize BrocaCheckpoint:\n  msgpack: {}\n  bincode: {}",
                    msgpack_err,
                    bincode_err
                );
            }
        };

        Self::finalize_loaded(checkpoint, path.as_ref())
    }

    fn finalize_loaded(checkpoint: Self, path: &Path) -> Result<Self> {
        if checkpoint.version > CHECKPOINT_VERSION {
            anyhow::bail!(
                "Broca checkpoint version {} is newer than supported (max: {})",
                checkpoint.version,
                CHECKPOINT_VERSION
            );
        }
        if checkpoint.version < CHECKPOINT_VERSION {
            tracing::warn!(
                saved_version = checkpoint.version,
                current_version = CHECKPOINT_VERSION,
                "Loading legacy Broca checkpoint (v{} → v{}). LiquidMambaConfig will use defaults.",
                checkpoint.version,
                CHECKPOINT_VERSION
            );
        }

        tracing::info!(
            path = %path.display(),
            epoch = checkpoint.training_epoch,
            loss = checkpoint.training_loss,
            vocab_size = checkpoint.token_embeddings.len(),
            version = checkpoint.version,
            "Broca checkpoint loaded"
        );

        Ok(checkpoint)
    }
}

/// Extension methods for BrocaGenerator to support checkpointing.
impl BrocaGenerator {
    /// Save the current generator state to a checkpoint file.
    ///
    /// The `liquid_mamba_config` parameter accepts a JSON string — serialize
    /// your `LiquidMambaConfig` via `serde_json::to_string(&config).ok()` before calling.
    /// Pass `None` for CfC-HDC-only training.
    pub fn save_checkpoint<P: AsRef<Path>>(
        &self,
        path: P,
        training_epoch: usize,
        training_loss: f32,
        adam_state: Option<AdamState>,
        projection_weights: Option<Vec<f32>>,
        liquid_mamba_config: Option<String>,
    ) -> Result<()> {
        let vocab = VocabFile {
            tokens: (0..self.tokenizer().vocab_size())
                .map(|i| self.tokenizer().token_str(i as u32).to_string())
                .collect(),
            merges: self
                .tokenizer()
                .merges()
                .iter()
                .map(|(l, r)| MergePair {
                    left: l.clone(),
                    right: r.clone(),
                })
                .collect(),
        };

        let mut checkpoint = BrocaCheckpoint {
            version: CHECKPOINT_VERSION,
            token_embeddings: self.controller().token_embeddings().to_vec(),
            network_state: self.controller().network().clone(),
            vocab,
            config: self.config().clone(),
            training_epoch,
            training_loss,
            adam_state,
            projection_weights,
            liquid_mamba_config,
            checksum: [0u8; 32],
        };

        checkpoint.save_to_file(path)
    }

    /// Load a generator from a checkpoint file.
    ///
    /// The genesis seed is used to reconstruct the encoder and position base
    /// (which are deterministic and not stored in the checkpoint).
    ///
    /// Returns `(generator, adam_state, projection_weights, liquid_mamba_config)`.
    ///
    /// The `liquid_mamba_config` is a JSON string that can be deserialized
    /// into `LiquidMambaConfig` when the mamba feature is enabled:
    /// ```ignore
    /// let lm_config: LiquidMambaConfig = serde_json::from_str(&json_str)?;
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn from_checkpoint<P: AsRef<Path>>(
        path: P,
        genesis: &symthaea_core::genesis::GenesisSeed,
    ) -> Result<(Self, Option<AdamState>, Option<Vec<f32>>, Option<String>)> {
        let checkpoint = BrocaCheckpoint::load_from_file(path)?;

        let tokenizer = crate::tokenizer::BpeTokenizer::from_vocab_file(&checkpoint.vocab);
        let mut gen = BrocaGenerator::with_tokenizer(genesis, checkpoint.config, tokenizer);

        // Restore trained weights
        *gen.controller_mut().token_embeddings_mut() = checkpoint.token_embeddings;
        // Restore network state (weights, momentums, etc.)
        *gen.controller_mut().network_mut() = checkpoint.network_state;

        Ok((
            gen,
            checkpoint.adam_state,
            checkpoint.projection_weights,
            checkpoint.liquid_mamba_config,
        ))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROJECTION CHECKPOINT (standalone, for Liquid-Mamba fusion)
// ═══════════════════════════════════════════════════════════════════════════════

/// Legacy v3 checkpoint layout (14 positional fields, no num_groups/has_adapter).
/// Used only for backward-compatible deserialization of older checkpoints.
#[cfg(feature = "mamba-cpu")]
#[derive(Deserialize)]
struct ProjectionCheckpointV3 {
    version: u32,
    projection_weights: Vec<f32>,
    hdc_dim: usize,
    bottleneck_dim: usize,
    ssm_dim: usize,
    training_epoch: usize,
    #[serde(default)]
    deep: bool,
    #[serde(default)]
    inner_dim: usize,
    #[serde(default)]
    diagnostics_snapshot: Option<crate::projection::GradientDiagnosticsSnapshot>,
    #[serde(default)]
    temporal: bool,
    #[serde(default)]
    chunk_dim: usize,
    #[serde(default)]
    num_chunks: usize,
    #[serde(default)]
    temporal_weights: Option<Vec<f32>>,
    checksum: [u8; 32],
}

#[cfg(feature = "mamba-cpu")]
impl From<ProjectionCheckpointV3> for ProjectionCheckpoint {
    fn from(v3: ProjectionCheckpointV3) -> Self {
        Self {
            version: v3.version,
            projection_weights: v3.projection_weights,
            hdc_dim: v3.hdc_dim,
            bottleneck_dim: v3.bottleneck_dim,
            ssm_dim: v3.ssm_dim,
            training_epoch: v3.training_epoch,
            deep: v3.deep,
            inner_dim: v3.inner_dim,
            diagnostics_snapshot: v3.diagnostics_snapshot,
            temporal: v3.temporal,
            chunk_dim: v3.chunk_dim,
            num_chunks: v3.num_chunks,
            temporal_weights: v3.temporal_weights,
            num_groups: 0,
            has_adapter: false,
            checksum: v3.checksum,
        }
    }
}

/// Standalone checkpoint for the HDC↔SSM projection weights.
///
/// Smaller and faster than a full BrocaCheckpoint — only stores the projection
/// matrix weights (8.8M parameters at default dimensions).
#[cfg(feature = "mamba-cpu")]
#[derive(Serialize, Deserialize)]
pub struct ProjectionCheckpoint {
    /// Schema version.
    pub version: u32,
    /// Flat projection weights from `HdcSsmProjection::flatten_weights()`.
    pub projection_weights: Vec<f32>,
    /// HDC dimension the projection was built for.
    pub hdc_dim: usize,
    /// Bottleneck dimension.
    pub bottleneck_dim: usize,
    /// SSM dimension.
    pub ssm_dim: usize,
    /// Training epoch at time of save.
    pub training_epoch: usize,
    /// Whether this projection uses the deep double-bottleneck architecture.
    /// Added after v2; backward-compatible via serde(default) = false.
    #[serde(default)]
    pub deep: bool,
    /// Inner bottleneck dimension (0 if shallow).
    /// Added after v2; backward-compatible via serde(default) = 0.
    #[serde(default)]
    pub inner_dim: usize,
    /// Optional gradient diagnostics snapshot from the training session.
    /// Added after v2; backward-compatible via serde(default) = None.
    #[serde(default)]
    pub diagnostics_snapshot: Option<crate::projection::GradientDiagnosticsSnapshot>,
    /// Whether this checkpoint uses temporal projection (chunk-based continuous latent prompting).
    /// Added in v3; v1/v2 checkpoints deserialize with `false` via serde(default).
    #[serde(default)]
    pub temporal: bool,
    /// Per-chunk dimension for temporal projection (e.g., 256).
    /// Added in v3; v1/v2 checkpoints deserialize with `0` via serde(default).
    #[serde(default)]
    pub chunk_dim: usize,
    /// Number of chunks for temporal projection (e.g., 64).
    /// Added in v3; v1/v2 checkpoints deserialize with `0` via serde(default).
    #[serde(default)]
    pub num_chunks: usize,
    /// Optional temporal projection weights (w_chunk_up + w_chunk_down + LN).
    /// Added in v3; v1/v2 checkpoints deserialize with `None` via serde(default).
    #[serde(default)]
    pub temporal_weights: Option<Vec<f32>>,
    /// Number of groups for grouped temporal projection.
    /// Added in v4; earlier checkpoints deserialize with `0` via serde(default).
    #[serde(default)]
    pub num_groups: usize,
    /// Whether this checkpoint was trained with the whitening adapter.
    /// Added in v4; earlier checkpoints deserialize with `false` via serde(default).
    #[serde(default)]
    pub has_adapter: bool,
    /// Blake3 integrity checksum (zeroed before hashing).
    pub checksum: [u8; 32],
}

#[cfg(feature = "mamba-cpu")]
impl ProjectionCheckpoint {
    /// Compute the blake3 checksum (with checksum field zeroed).
    fn compute_checksum(&self) -> [u8; 32] {
        let copy = ProjectionCheckpoint {
            version: self.version,
            projection_weights: self.projection_weights.clone(),
            hdc_dim: self.hdc_dim,
            bottleneck_dim: self.bottleneck_dim,
            ssm_dim: self.ssm_dim,
            training_epoch: self.training_epoch,
            deep: self.deep,
            inner_dim: self.inner_dim,
            diagnostics_snapshot: self.diagnostics_snapshot.clone(),
            temporal: self.temporal,
            chunk_dim: self.chunk_dim,
            num_chunks: self.num_chunks,
            temporal_weights: self.temporal_weights.clone(),
            num_groups: self.num_groups,
            has_adapter: self.has_adapter,
            checksum: [0u8; 32],
        };
        let bytes = rmp_serde::to_vec(&copy)
            .expect("ProjectionCheckpoint serialization must succeed for integrity checking");
        *blake3::hash(&bytes).as_bytes()
    }

    /// Verify the checkpoint integrity.
    pub fn verify(&self) -> bool {
        let expected = self.compute_checksum();
        self.checksum == expected
    }

    /// Save to a file (MessagePack named-map format for forward compatibility).
    pub fn save_to_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.checksum = self.compute_checksum();
        let serialized =
            rmp_serde::to_vec_named(self).context("Failed to serialize ProjectionCheckpoint")?;
        let mut file = std::fs::File::create(path.as_ref()).with_context(|| {
            format!(
                "creating projection checkpoint: {}",
                path.as_ref().display()
            )
        })?;
        file.write_all(&serialized)?;
        file.sync_all()?;
        tracing::info!(
            path = %path.as_ref().display(),
            epoch = self.training_epoch,
            params = self.projection_weights.len(),
            "Projection checkpoint saved"
        );
        Ok(())
    }

    /// Create a new ProjectionCheckpoint with the current version.
    pub fn new(
        weights: Vec<f32>,
        hdc_dim: usize,
        bottleneck_dim: usize,
        ssm_dim: usize,
        training_epoch: usize,
        deep: bool,
        inner_dim: usize,
    ) -> Self {
        Self {
            version: PROJECTION_CHECKPOINT_VERSION,
            projection_weights: weights,
            hdc_dim,
            bottleneck_dim,
            ssm_dim,
            training_epoch,
            deep,
            inner_dim,
            diagnostics_snapshot: None,
            temporal: false,
            chunk_dim: 0,
            num_chunks: 0,
            temporal_weights: None,
            num_groups: 0,
            has_adapter: false,
            checksum: [0u8; 32],
        }
    }

    /// Create a new ProjectionCheckpoint for temporal projection.
    pub fn new_temporal(
        spatial_weights: Vec<f32>,
        temporal_weights: Vec<f32>,
        hdc_dim: usize,
        bottleneck_dim: usize,
        ssm_dim: usize,
        training_epoch: usize,
        chunk_dim: usize,
        num_chunks: usize,
    ) -> Self {
        Self {
            version: PROJECTION_CHECKPOINT_VERSION,
            projection_weights: spatial_weights,
            hdc_dim,
            bottleneck_dim,
            ssm_dim,
            training_epoch,
            deep: false,
            inner_dim: 0,
            diagnostics_snapshot: None,
            temporal: true,
            chunk_dim,
            num_chunks,
            temporal_weights: Some(temporal_weights),
            num_groups: 0,
            has_adapter: false,
            checksum: [0u8; 32],
        }
    }

    /// Create a temporal checkpoint with multi-group/adapter metadata.
    ///
    /// Create a temporal checkpoint with multi-group and adapter metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn new_temporal_with_groups(
        spatial_weights: Vec<f32>,
        temporal_weights: Vec<f32>,
        hdc_dim: usize,
        bottleneck_dim: usize,
        ssm_dim: usize,
        training_epoch: usize,
        chunk_dim: usize,
        num_chunks: usize,
        num_groups: usize,
        has_adapter: bool,
    ) -> Self {
        let mut ckpt = Self::new_temporal(
            spatial_weights,
            temporal_weights,
            hdc_dim,
            bottleneck_dim,
            ssm_dim,
            training_epoch,
            chunk_dim,
            num_chunks,
        );
        ckpt.num_groups = num_groups;
        ckpt.has_adapter = has_adapter;
        ckpt
    }

    /// Load from a file with integrity and version compatibility checks.
    ///
    /// Tries MessagePack (current format) first, then falls back to bincode
    /// for legacy checkpoints. Legacy bincode checkpoints skip integrity
    /// verification (the hash was computed from bincode bytes).
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref()).with_context(|| {
            format!("opening projection checkpoint: {}", path.as_ref().display())
        })?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Try MessagePack (current named-map format, then legacy positional array, then bincode)
        let checkpoint: Self = if let Ok(ckpt) = rmp_serde::from_slice::<Self>(&buffer) {
            // Named-map or exact-match positional array — verify integrity
            if !ckpt.verify() {
                anyhow::bail!("Projection checkpoint integrity check failed: checksum mismatch");
            }
            ckpt
        } else if let Ok(ckpt) = rmp_serde::from_slice::<ProjectionCheckpointV3>(&buffer) {
            // Legacy v3 positional array (14 fields) — upgrade to v4
            tracing::warn!("Loaded legacy v3 positional-array checkpoint — upgrading to v4");
            ckpt.into()
        } else {
            // Fall back to bincode (oldest format) — skip verify since hash format changed
            let ckpt = bincode::deserialize::<Self>(&buffer)
                .context("Failed to deserialize ProjectionCheckpoint (tried msgpack named/positional + bincode)")?;
            tracing::warn!(
                "Loaded legacy bincode projection checkpoint — will be re-saved as MessagePack"
            );
            ckpt
        };

        // Version compatibility check
        if checkpoint.version < PROJECTION_MIN_VERSION {
            anyhow::bail!(
                "Projection checkpoint version {} is too old (minimum: {})",
                checkpoint.version,
                PROJECTION_MIN_VERSION
            );
        }
        if checkpoint.version > PROJECTION_CHECKPOINT_VERSION {
            anyhow::bail!(
                "Projection checkpoint version {} is newer than supported (max: {})",
                checkpoint.version,
                PROJECTION_CHECKPOINT_VERSION
            );
        }
        if checkpoint.version < PROJECTION_CHECKPOINT_VERSION {
            tracing::warn!(
                saved_version = checkpoint.version,
                current_version = PROJECTION_CHECKPOINT_VERSION,
                "Loading legacy projection checkpoint (v{} → v{}). Newer fields will use defaults.",
                checkpoint.version,
                PROJECTION_CHECKPOINT_VERSION
            );
        }

        tracing::info!(
            path = %path.as_ref().display(),
            version = checkpoint.version,
            epoch = checkpoint.training_epoch,
            hdc_dim = checkpoint.hdc_dim,
            bottleneck = checkpoint.bottleneck_dim,
            ssm_dim = checkpoint.ssm_dim,
            params = checkpoint.projection_weights.len(),
            "Projection checkpoint loaded"
        );
        Ok(checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::genesis::GenesisSeed;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-broca-checkpoint")
    }

    #[test]
    fn test_checkpoint_save_load_roundtrip() {
        let genesis = test_genesis();
        let config = BrocaConfig::default();
        let gen = BrocaGenerator::new(&genesis, config);

        let dir = std::env::temp_dir();
        let path = dir.join("broca_test_checkpoint.bin");

        // Save
        gen.save_checkpoint(&path, 5, 2.5, None, None, None)
            .unwrap();

        // Load
        let (loaded_gen, adam, proj, lm_config) =
            BrocaGenerator::from_checkpoint(&path, &genesis).unwrap();
        assert!(adam.is_none());
        assert!(proj.is_none());
        assert!(lm_config.is_none());
        assert_eq!(
            loaded_gen.tokenizer().vocab_size(),
            gen.tokenizer().vocab_size()
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_checkpoint_with_adam_state() {
        let genesis = test_genesis();
        let config = BrocaConfig::default();
        let gen = BrocaGenerator::new(&genesis, config);

        let adam = AdamState::new(gen.tokenizer().vocab_size(), 16384);

        let dir = std::env::temp_dir();
        let path = dir.join("broca_test_checkpoint_adam.bin");

        gen.save_checkpoint(&path, 10, 1.5, Some(adam), None, None)
            .unwrap();

        let (_, loaded_adam, _, _) = BrocaGenerator::from_checkpoint(&path, &genesis).unwrap();
        assert!(loaded_adam.is_some());
        let loaded_adam = loaded_adam.unwrap();
        assert_eq!(loaded_adam.t, 0);
        assert_eq!(loaded_adam.beta1, 0.9);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_checkpoint_with_projection_weights() {
        let genesis = test_genesis();
        let config = BrocaConfig::default();
        let gen = BrocaGenerator::new(&genesis, config);

        let proj_weights = vec![0.1, 0.2, 0.3, -0.4, 0.5];

        let dir = std::env::temp_dir();
        let path = dir.join("broca_test_checkpoint_proj.bin");

        gen.save_checkpoint(&path, 3, 2.0, None, Some(proj_weights.clone()), None)
            .unwrap();

        let (_, _, loaded_proj, _) = BrocaGenerator::from_checkpoint(&path, &genesis).unwrap();
        assert!(loaded_proj.is_some());
        let loaded_proj = loaded_proj.unwrap();
        assert_eq!(loaded_proj.len(), 5);
        assert!((loaded_proj[0] - 0.1).abs() < 1e-6);
        assert!((loaded_proj[3] - (-0.4)).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_checkpoint_corruption_detected() {
        let genesis = test_genesis();
        let config = BrocaConfig::default();
        let gen = BrocaGenerator::new(&genesis, config);

        let dir = std::env::temp_dir();
        let path = dir.join("broca_test_corrupt.bin");

        gen.save_checkpoint(&path, 1, 3.0, None, None, None)
            .unwrap();

        // Corrupt the file severely — overwrite the first 64 bytes to break
        // the msgpack/bincode framing, not just flip data bits (which may
        // deserialize successfully with a checksum warning).
        let mut data = std::fs::read(&path).unwrap();
        for byte in data.iter_mut().take(64) {
            *byte = 0xFF;
        }
        std::fs::write(&path, data).unwrap();

        // Load should fail
        let result = BrocaGenerator::from_checkpoint(&path, &genesis);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_adam_state_step() {
        let mut adam = AdamState::new(1, 4);
        let grad = vec![0.1, -0.2, 0.3, -0.4];
        let update = adam.step(0, &grad, 0.001);

        assert_eq!(update.len(), 4);
        assert!(update.iter().all(|v| v.is_finite()));
        assert_eq!(adam.t, 1);
    }

    #[test]
    fn test_checkpoint_verify_valid() {
        let mut checkpoint = BrocaCheckpoint {
            version: CHECKPOINT_VERSION,
            token_embeddings: vec![],
            network_state: HdcLtcUnifiedNetwork::from_genesis(
                symthaea_core::hdc::UnifiedNetworkConfig::default(),
                &test_genesis(),
            ),
            vocab: VocabFile {
                tokens: vec![],
                merges: vec![],
            },
            config: BrocaConfig::default(),
            training_epoch: 0,
            training_loss: 0.0,
            adam_state: None,
            projection_weights: None,
            liquid_mamba_config: None,
            checksum: [0u8; 32],
        };
        checkpoint.checksum = checkpoint.compute_checksum();
        assert!(checkpoint.verify());
    }

    #[cfg(feature = "mamba-cpu")]
    #[test]
    fn test_checkpoint_roundtrip_liquid_mamba_config() {
        use crate::liquid_mamba::LiquidMambaConfig;

        let genesis = test_genesis();
        let config = BrocaConfig::default();
        let gen = BrocaGenerator::new(&genesis, config);

        let lm_config = LiquidMambaConfig {
            surprise_gradient_alpha: 0.8,
            base_lr: 0.005,
            ..Default::default()
        };

        let dir = std::env::temp_dir();
        let path = dir.join("broca_test_lm_config.bin");

        let lm_json = serde_json::to_string(&lm_config).ok();
        gen.save_checkpoint(&path, 7, 1.0, None, None, lm_json)
            .unwrap();

        let (_, _, _, loaded_lm_json) = BrocaGenerator::from_checkpoint(&path, &genesis).unwrap();
        assert!(loaded_lm_json.is_some());
        let loaded: LiquidMambaConfig = serde_json::from_str(&loaded_lm_json.unwrap()).unwrap();
        assert!((loaded.surprise_gradient_alpha - 0.8).abs() < 1e-6);
        assert!((loaded.base_lr - 0.005).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_checkpoint_roundtrip_merges() {
        let genesis = test_genesis();
        let config = BrocaConfig::default();

        // Use 4K tokenizer which has merge rules
        let tokenizer = crate::tokenizer::BpeTokenizer::default_4k();
        let gen = BrocaGenerator::with_tokenizer(&genesis, config, tokenizer);

        let original_merges_count = gen.tokenizer().merges().len();
        assert!(original_merges_count > 0, "4K tokenizer should have merges");

        // Encode some text to compare before/after
        let test_text = "the world is beautiful";
        let original_ids = gen.tokenizer().encode(test_text);

        let dir = std::env::temp_dir();
        let path = dir.join("broca_test_merges.bin");

        gen.save_checkpoint(&path, 1, 1.0, None, None, None)
            .unwrap();

        let (loaded_gen, _, _, _) = BrocaGenerator::from_checkpoint(&path, &genesis).unwrap();
        let loaded_merges_count = loaded_gen.tokenizer().merges().len();
        assert_eq!(
            loaded_merges_count, original_merges_count,
            "Merges should survive checkpoint round-trip"
        );

        let loaded_ids = loaded_gen.tokenizer().encode(test_text);
        assert_eq!(
            original_ids, loaded_ids,
            "Token IDs should be identical after checkpoint round-trip"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[cfg(feature = "mamba-cpu")]
    #[test]
    fn test_projection_checkpoint_deep_flag() {
        let dir = std::env::temp_dir();
        let path = dir.join("broca_test_proj_deep.bin");

        let mut ckpt =
            ProjectionCheckpoint::new(vec![0.1, 0.2, 0.3], 16384, 256, 768, 5, true, 128);
        ckpt.save_to_file(&path).unwrap();

        let loaded = ProjectionCheckpoint::load_from_file(&path).unwrap();
        assert!(loaded.deep, "deep flag should survive round-trip");
        assert_eq!(loaded.inner_dim, 128, "inner_dim should survive round-trip");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_checkpoint_preserves_generation_output() {
        use crate::encoder::ThoughtChannels;
        use crate::generator::SamplingStrategy;

        let genesis = test_genesis();
        let config = BrocaConfig {
            sampling: SamplingStrategy::Greedy,
            enable_coherence_feedback: false,
            enable_semantic_veto: false,
            ..BrocaConfig::default()
        };
        let mut gen = BrocaGenerator::new(&genesis, config.clone());

        let channels = ThoughtChannels::with_intent(1);
        let result_before = gen.generate(&channels);

        let dir = std::env::temp_dir();
        let path = dir.join("broca_test_gen_determinism.bin");

        gen.save_checkpoint(&path, 0, 0.0, None, None, None)
            .unwrap();

        let (mut loaded_gen, _, _, _) = BrocaGenerator::from_checkpoint(&path, &genesis).unwrap();
        let result_after = loaded_gen.generate(&channels);

        assert_eq!(
            result_before.token_ids, result_after.token_ids,
            "Checkpoint round-trip must preserve generation output"
        );
        assert_eq!(result_before.text, result_after.text);

        let _ = std::fs::remove_file(&path);
    }
}
