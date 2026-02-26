//! BrocaCheckpoint: save/load trained Broca models with integrity verification.
//!
//! Uses bincode for efficient serialization and blake3 for checksum integrity.
//! Pattern follows `swarm/checkpoint.rs`.

use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNetwork};

use crate::generator::{BrocaConfig, BrocaGenerator};
use crate::tokenizer::VocabFile;

/// Current checkpoint schema version.
const CHECKPOINT_VERSION: u32 = 1;

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

            // Bias correction
            let m_hat = m[j] / (1.0 - self.beta1.powf(t));
            let v_hat = v[j] / (1.0 - self.beta2.powf(t));

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
    /// Blake3 integrity checksum (set to zeros before hashing).
    pub checksum: [u8; 32],
}

impl BrocaCheckpoint {
    /// Compute the blake3 checksum of the checkpoint (with checksum field zeroed).
    fn compute_checksum(&self) -> [u8; 32] {
        // Serialize with zeroed checksum
        let mut copy = BrocaCheckpoint {
            version: self.version,
            token_embeddings: self.token_embeddings.clone(),
            network_state: self.network_state.clone(),
            vocab: self.vocab.clone(),
            config: self.config.clone(),
            training_epoch: self.training_epoch,
            training_loss: self.training_loss,
            adam_state: self.adam_state.clone(),
            projection_weights: self.projection_weights.clone(),
            checksum: [0u8; 32],
        };
        copy.checksum = [0u8; 32];

        let bytes = bincode::serialize(&copy).unwrap_or_default();
        *blake3::hash(&bytes).as_bytes()
    }

    /// Verify the checkpoint integrity.
    pub fn verify(&self) -> bool {
        let expected = self.compute_checksum();
        self.checksum == expected
    }

    /// Save checkpoint to a file.
    pub fn save_to_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        // Compute and set checksum before serialization
        self.checksum = self.compute_checksum();

        let serialized = bincode::serialize(self)
            .context("Failed to serialize BrocaCheckpoint")?;

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
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())
            .with_context(|| format!("opening checkpoint file: {}", path.as_ref().display()))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let checkpoint: Self = bincode::deserialize(&buffer)
            .context("Failed to deserialize BrocaCheckpoint")?;

        if !checkpoint.verify() {
            anyhow::bail!("Checkpoint integrity check failed: checksum mismatch");
        }

        tracing::info!(
            path = %path.as_ref().display(),
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
    pub fn save_checkpoint<P: AsRef<Path>>(
        &self,
        path: P,
        training_epoch: usize,
        training_loss: f32,
        adam_state: Option<AdamState>,
        projection_weights: Option<Vec<f32>>,
    ) -> Result<()> {
        let vocab = VocabFile {
            tokens: (0..self.tokenizer().vocab_size())
                .map(|i| self.tokenizer().token_str(i as u32).to_string())
                .collect(),
            merges: vec![],
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
            checksum: [0u8; 32],
        };

        checkpoint.save_to_file(path)
    }

    /// Load a generator from a checkpoint file.
    ///
    /// The genesis seed is used to reconstruct the encoder and position base
    /// (which are deterministic and not stored in the checkpoint).
    pub fn from_checkpoint<P: AsRef<Path>>(
        path: P,
        genesis: &symthaea_core::genesis::GenesisSeed,
    ) -> Result<(Self, Option<AdamState>, Option<Vec<f32>>)> {
        let checkpoint = BrocaCheckpoint::load_from_file(path)?;

        let tokenizer = crate::tokenizer::BpeTokenizer::from_vocab_file(&checkpoint.vocab);
        let mut gen = BrocaGenerator::with_tokenizer(genesis, checkpoint.config, tokenizer);

        // Restore trained weights
        *gen.controller_mut().token_embeddings_mut() = checkpoint.token_embeddings;
        // Restore network state (weights, momentums, etc.)
        *gen.controller_mut().network_mut() = checkpoint.network_state;

        Ok((gen, checkpoint.adam_state, checkpoint.projection_weights))
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
        gen.save_checkpoint(&path, 5, 2.5, None, None).unwrap();

        // Load
        let (loaded_gen, adam, proj) = BrocaGenerator::from_checkpoint(&path, &genesis).unwrap();
        assert!(adam.is_none());
        assert!(proj.is_none());
        assert_eq!(loaded_gen.tokenizer().vocab_size(), gen.tokenizer().vocab_size());

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

        gen.save_checkpoint(&path, 10, 1.5, Some(adam), None).unwrap();

        let (_, loaded_adam, _) = BrocaGenerator::from_checkpoint(&path, &genesis).unwrap();
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

        gen.save_checkpoint(&path, 3, 2.0, None, Some(proj_weights.clone()))
            .unwrap();

        let (_, _, loaded_proj) = BrocaGenerator::from_checkpoint(&path, &genesis).unwrap();
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

        gen.save_checkpoint(&path, 1, 3.0, None, None).unwrap();

        // Corrupt the file
        let mut data = std::fs::read(&path).unwrap();
        if data.len() > 100 {
            data[50] ^= 0xFF;
            data[51] ^= 0xFF;
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
            vocab: VocabFile { tokens: vec![], merges: vec![] },
            config: BrocaConfig::default(),
            training_epoch: 0,
            training_loss: 0.0,
            adam_state: None,
            projection_weights: None,
            checksum: [0u8; 32],
        };
        checkpoint.checksum = checkpoint.compute_checksum();
        assert!(checkpoint.verify());
    }
}
