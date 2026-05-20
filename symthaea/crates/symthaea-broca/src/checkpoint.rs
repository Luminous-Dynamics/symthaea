// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Broca Checkpoint Persistence
//!
//! Handles serialization and deserialization of complete Broca states.
//! Includes CfC-HDC weights, token embeddings, and tokenizer metadata.
//!
//! Uses MessagePack via `rmp-serde` for efficient binary storage with
//! named-map support (enabling forward compatibility when fields are added).

use crate::controller::LanguageControllerConfig;
use crate::encoder::NUM_CHANNELS;
use crate::gating::GatingConfig;
use crate::generator::BrocaConfig;
#[cfg(feature = "mamba-cpu")]
use crate::projection::GradientDiagnosticsSnapshot;
use crate::tokenizer::{MergePair, VocabFile};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::Path;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork;
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Current schema version.
pub const CHECKPOINT_VERSION: u32 = 2;
/// Minimum supported schema version.
pub const MIN_SUPPORTED_VERSION: u32 = 1;

/// Channel schema version (v1 = 20ch, v2 = 24ch, v3 = 43ch/47ch).
pub const CHANNEL_SCHEMA_VERSION: u32 = 3;

/// Magic bytes for enveloped checkpoints.
pub const CHECKPOINT_ENVELOPE_MAGIC: &[u8; 8] = b"BROCA_V2";
/// Header length for enveloped checkpoints (magic + 32-byte hash).
pub const CHECKPOINT_ENVELOPE_HEADER_LEN: usize = 40;

/// Adam optimizer state for a single parameter group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamState {
    pub t: usize,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

impl AdamState {
    pub fn new(vocab_size: usize, hdc_dim: usize) -> Self {
        let size = vocab_size * hdc_dim;
        Self {
            t: 0,
            m: vec![0.0; size],
            v: vec![0.0; size],
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    pub fn step(&mut self, idx: usize, grad: &[f32], lr: f32) -> Vec<f32> {
        self.t += 1;
        let t = self.t as f32;

        let bc1 = 1.0 / (1.0 - self.beta1.powf(t));
        let bc2 = 1.0 / (1.0 - self.beta2.powf(t));

        let mut update = vec![0.0; grad.len()];
        let offset = idx * grad.len();

        for (i, &g) in grad.iter().enumerate() {
            let m_ref = &mut self.m[offset + i];
            let v_ref = &mut self.v[offset + i];

            *m_ref = self.beta1 * (*m_ref) + (1.0 - self.beta1) * g;
            *v_ref = self.beta2 * (*v_ref) + (1.0 - self.beta2) * g * g;

            let m_hat = *m_ref * bc1;
            let v_hat = *v_ref * bc2;

            update[i] = lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }

        update
    }
}

/// Compatibility metadata for refusing accidental cross-schema checkpoint loads.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BrocaCheckpointMetadata {
    #[serde(default)]
    pub vocab_size: usize,
    #[serde(default)]
    pub vocab_hash: [u8; 32],
    #[serde(default)]
    pub embedding_dim: usize,
    #[serde(default)]
    pub channel_count: usize,
    #[serde(default)]
    pub channel_schema_version: u32,
    #[serde(default)]
    pub feature_set: Vec<String>,
    #[serde(default)]
    pub backend: String,
}

/// A complete Broca checkpoint: model weights, config, and training state.
#[derive(Serialize, Deserialize, Clone)]
pub struct BrocaCheckpoint {
    pub version: u32,
    pub token_embeddings: Vec<ContinuousHV>,
    pub network_state: HdcLtcUnifiedNetwork,
    pub vocab: VocabFile,
    pub config: BrocaConfig,
    pub training_epoch: usize,
    pub training_loss: f32,
    pub adam_state: Option<AdamState>,
    pub projection_weights: Option<Vec<f32>>,
    #[serde(default)]
    pub liquid_mamba_config: Option<String>,
    #[serde(default)]
    pub metadata: BrocaCheckpointMetadata,
    pub checksum: [u8; 32],
}

impl BrocaCheckpoint {
    fn metadata_for(
        token_embeddings: &[ContinuousHV],
        vocab: &VocabFile,
        projection_weights: &Option<Vec<f32>>,
        liquid_mamba_config: &Option<String>,
    ) -> BrocaCheckpointMetadata {
        let vocab_bytes = serde_json::to_vec(vocab)
            .expect("VocabFile serialization must succeed for compatibility metadata");
        BrocaCheckpointMetadata {
            vocab_size: vocab.tokens.len(),
            vocab_hash: *blake3::hash(&vocab_bytes).as_bytes(),
            embedding_dim: token_embeddings
                .first()
                .map(ContinuousHV::dim)
                .unwrap_or(symthaea_core::hdc::HDC_DIMENSION),
            channel_count: NUM_CHANNELS,
            channel_schema_version: CHANNEL_SCHEMA_VERSION,
            feature_set: vec![],
            backend: if liquid_mamba_config.is_some() || projection_weights.is_some() {
                "cfc-hdc+liquid-mamba".to_string()
            } else {
                "cfc-hdc".to_string()
            },
        }
    }

    fn compute_checksum(&self) -> [u8; 32] {
        let mut copy = self.clone();
        copy.checksum = [0u8; 32];
        let bytes = rmp_serde::to_vec(&copy)
            .expect("BrocaCheckpoint serialization must succeed for integrity checking");
        *blake3::hash(&bytes).as_bytes()
    }

    pub fn verify(&self) -> bool {
        let expected = self.compute_checksum();
        self.checksum == expected
    }

    pub fn save_to_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.metadata = Self::metadata_for(
            &self.token_embeddings,
            &self.vocab,
            &self.projection_weights,
            &self.liquid_mamba_config,
        );
        self.checksum = [0u8; 32];
        let payload = rmp_serde::to_vec(self).context("Failed to serialize BrocaCheckpoint")?;
        self.checksum = *blake3::hash(&payload).as_bytes();

        let mut file = std::fs::File::create(path.as_ref())?;
        file.write_all(CHECKPOINT_ENVELOPE_MAGIC)?;
        file.write_all(&self.checksum)?;
        file.write_all(&payload)?;
        file.sync_all()?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::load_from_file_inner(path, false)
    }

    pub fn load_from_file_allow_checksum_mismatch<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::load_from_file_inner(path, true)
    }

    fn load_from_file_inner<P: AsRef<Path>>(
        path: P,
        allow_checksum_mismatch: bool,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        if buffer.starts_with(CHECKPOINT_ENVELOPE_MAGIC) {
            let mut expected = [0u8; 32];
            expected.copy_from_slice(&buffer[8..CHECKPOINT_ENVELOPE_HEADER_LEN]);
            let payload = &buffer[CHECKPOINT_ENVELOPE_HEADER_LEN..];
            let observed = *blake3::hash(payload).as_bytes();
            if expected != observed && !allow_checksum_mismatch {
                anyhow::bail!("Broca checkpoint checksum mismatch");
            }

            let mut checkpoint = rmp_serde::from_slice::<Self>(payload)
                .context("Failed to deserialize BrocaCheckpoint payload")?;
            checkpoint.checksum = expected;
            return Ok(checkpoint);
        }

        // Try direct MessagePack
        let mut checkpoint: Self = match rmp_serde::from_slice(&buffer) {
            Ok(ckpt) => ckpt,
            Err(e) => {
                // Fallback to bincode
                match bincode::deserialize(&buffer) {
                    Ok(ckpt) => ckpt,
                    Err(be) => {
                        anyhow::bail!("Failed to deserialize: msgpack={}, bincode={}", e, be)
                    }
                }
            }
        };

        if !checkpoint.verify() && !allow_checksum_mismatch {
            anyhow::bail!("Broca checkpoint checksum mismatch (direct)");
        }
        Ok(checkpoint)
    }
}

/// PROJECTION CHECKPOINT
#[cfg(feature = "mamba-cpu")]
pub const PROJECTION_CHECKPOINT_VERSION: u32 = 5;

#[cfg(feature = "mamba-cpu")]
#[derive(Serialize, Deserialize, Clone)]
pub struct ProjectionCheckpoint {
    pub version: u32,
    pub projection_weights: Vec<f32>,
    pub hdc_dim: usize,
    pub bottleneck_dim: usize,
    pub ssm_dim: usize,
    pub training_epoch: usize,
    #[serde(default)]
    pub deep: bool,
    #[serde(default)]
    pub inner_dim: usize,
    #[serde(default)]
    pub diagnostics_snapshot: Option<GradientDiagnosticsSnapshot>,
    #[serde(default)]
    pub temporal: bool,
    #[serde(default)]
    pub chunk_dim: usize,
    #[serde(default)]
    pub num_chunks: usize,
    #[serde(default)]
    pub temporal_weights: Option<Vec<f32>>,
    #[serde(default)]
    pub num_groups: usize,
    #[serde(default)]
    pub has_adapter: bool,
    #[serde(default)]
    pub full_snapshot: Option<Box<BrocaCheckpoint>>,
    pub checksum: [u8; 32],
}

#[cfg(feature = "mamba-cpu")]
impl ProjectionCheckpoint {
    fn compute_checksum(&self) -> [u8; 32] {
        let mut copy = self.clone();
        copy.checksum = [0u8; 32];
        let bytes = rmp_serde::to_vec(&copy).expect("ProjectionCheckpoint serialization failed");
        *blake3::hash(&bytes).as_bytes()
    }

    pub fn verify(&self) -> bool {
        self.checksum == self.compute_checksum()
    }

    pub fn save_to_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.checksum = self.compute_checksum();
        let serialized = rmp_serde::to_vec_named(self)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    /// Load a projection checkpoint from a file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        // Try named-map first (new format)
        let checkpoint: Self = match rmp_serde::from_slice(&bytes) {
            Ok(ckpt) => ckpt,
            Err(_) => {
                // Fallback to positional array (legacy or some training outputs)
                rmp_serde::decode::from_slice(&bytes).context(
                    "Failed to deserialize ProjectionCheckpoint (tried named and positional)",
                )?
            }
        };

        if !checkpoint.verify() {
            anyhow::bail!("Projection checkpoint checksum mismatch");
        }
        Ok(checkpoint)
    }

    /// Load while allowing MessagePack checksum mismatch.
    pub fn load_from_file_allow_checksum_mismatch<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let checkpoint: Self = match rmp_serde::from_slice(&bytes) {
            Ok(ckpt) => ckpt,
            Err(_) => rmp_serde::decode::from_slice(&bytes)
                .context("Failed to deserialize ProjectionCheckpoint (recovery mode)")?,
        };
        Ok(checkpoint)
    }

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
            num_groups: 1,
            has_adapter: false,
            full_snapshot: None,
            checksum: [0u8; 32],
        }
    }

    pub fn new_temporal(
        weights: Vec<f32>,
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
            projection_weights: weights,
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
            num_groups: 1,
            has_adapter: false,
            full_snapshot: None,
            checksum: [0u8; 32],
        }
    }

    pub fn new_temporal_with_groups(
        weights: Vec<f32>,
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
        Self {
            version: PROJECTION_CHECKPOINT_VERSION,
            projection_weights: weights,
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
            num_groups,
            has_adapter,
            full_snapshot: None,
            checksum: [0u8; 32],
        }
    }
}
