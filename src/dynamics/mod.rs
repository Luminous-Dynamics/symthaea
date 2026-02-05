//! # Symthaea Dynamics: Continuous-Time Neural Dynamics
//!
//! This module provides continuous-time neural network components including:
//! - Closed-form Continuous-time (CfC) networks
//! - Liquid Time-Constant (LTC) cells
//! - World model components
//! - Crystallized concept representations
//!
//! ## Key Types
//!
//! - `CfCNetwork` - Closed-form continuous neural network
//! - `CrystalizedConcept` - Stable memory representation
//! - `HierarchicalCfCWorldModel` - Multi-level world modeling

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod cfc;
pub mod cfc_gpu;
pub mod hierarchical_cfc;
pub mod cfc_coherence;
pub mod temporal_signatures;
pub mod world_model;

/// A crystallized concept - a stable, consolidated memory representation
///
/// Crystallized concepts represent knowledge that has been consolidated from
/// episodic memories into semantic knowledge through sleep-like processes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrystalizedConcept {
    /// Unique identifier
    pub id: u64,

    /// Name/label for this concept
    pub name: String,

    /// Semantic description
    pub description: Option<String>,

    /// High-dimensional vector representation
    pub embedding: Vec<f32>,

    /// Associated concepts by ID and strength
    pub associations: HashMap<u64, f32>,

    /// Confidence/certainty in this concept
    pub confidence: f32,

    /// Number of times this concept was activated
    pub activation_count: u64,

    /// Last activation timestamp
    pub last_activated: u64,

    /// Source memories that contributed to this concept
    pub source_memories: Vec<u64>,

    /// Hierarchical level (0 = ground-level, higher = more abstract)
    pub abstraction_level: u8,

    /// Emotional valence (-1.0 to 1.0)
    pub emotional_valence: f32,

    /// Metadata tags
    pub tags: Vec<String>,
}

impl CrystalizedConcept {
    /// Create a new crystallized concept
    pub fn new(id: u64, name: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            id,
            name: name.into(),
            description: None,
            embedding,
            associations: HashMap::new(),
            confidence: 0.5,
            activation_count: 0,
            last_activated: 0,
            source_memories: Vec::new(),
            abstraction_level: 0,
            emotional_valence: 0.0,
            tags: Vec::new(),
        }
    }

    /// Create with full parameters
    pub fn with_details(
        id: u64,
        name: impl Into<String>,
        description: impl Into<String>,
        embedding: Vec<f32>,
        confidence: f32,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            description: Some(description.into()),
            embedding,
            associations: HashMap::new(),
            confidence,
            activation_count: 0,
            last_activated: 0,
            source_memories: Vec::new(),
            abstraction_level: 0,
            emotional_valence: 0.0,
            tags: Vec::new(),
        }
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.embedding.len()
    }

    /// Add an association to another concept
    pub fn add_association(&mut self, other_id: u64, strength: f32) {
        self.associations.insert(other_id, strength.clamp(-1.0, 1.0));
    }

    /// Get association strength with another concept
    pub fn association_strength(&self, other_id: u64) -> f32 {
        self.associations.get(&other_id).copied().unwrap_or(0.0)
    }

    /// Record an activation of this concept
    pub fn activate(&mut self, timestamp: u64) {
        self.activation_count += 1;
        self.last_activated = timestamp;
    }

    /// Calculate similarity with another concept using cosine similarity
    pub fn similarity(&self, other: &CrystalizedConcept) -> f32 {
        if self.embedding.len() != other.embedding.len() {
            return 0.0;
        }

        let dot: f32 = self.embedding.iter()
            .zip(other.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();

        let mag_self: f32 = self.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_other: f32 = other.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_self > 0.0 && mag_other > 0.0 {
            dot / (mag_self * mag_other)
        } else {
            0.0
        }
    }

    /// Add a source memory
    pub fn add_source(&mut self, memory_id: u64) {
        if !self.source_memories.contains(&memory_id) {
            self.source_memories.push(memory_id);
        }
    }

    /// Set abstraction level
    pub fn set_level(&mut self, level: u8) {
        self.abstraction_level = level;
    }

    /// Set emotional valence
    pub fn set_valence(&mut self, valence: f32) {
        self.emotional_valence = valence.clamp(-1.0, 1.0);
    }
}

impl Default for CrystalizedConcept {
    fn default() -> Self {
        Self::new(0, "unnamed", Vec::new())
    }
}

/// Result type for dynamics operations
pub type DynamicsResult<T> = Result<T, DynamicsError>;

/// Error type for dynamics operations
#[derive(Debug, thiserror::Error)]
pub enum DynamicsError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Not initialized")]
    NotInitialized,
}

// Re-export key types
pub use cfc::{
    CfCNetwork, CfCCell, CfCConfig, CfCNetworkConfig,
    OnlineLearningConfig, OnlineLearningStats, NetworkOnlineLearningStats,
    // Phi-gated attention
    PhiGatedConfig, compute_phi_attention_weights,
};
pub use cfc_coherence::{
    CfCCoherenceBridge, CoherenceConfig, TemporalCoherenceMetrics, CoherenceSummary
};
pub use temporal_signatures::{
    TemporalSignatureEncoder, SignatureConfig, ConsciousnessPattern,
    TrajectoryFeatures, TemporalStateSummary
};
pub use world_model::{HierarchicalCfCWorldModel, WorldModelConfig, WorldModelLayer};
pub use hierarchical_cfc::{
    HierarchicalCfC, HierarchicalCfCConfig, HierarchicalOutput,
    DEFAULT_TIME_CONSTANTS
};
pub use cfc_gpu::{GpuCfcNetwork, GpuCfcConfig, GpuCfcStats, GpuBackend};
