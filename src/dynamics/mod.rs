// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
pub mod cfc_coherence;
pub mod cfc_gpu;
pub mod hierarchical_cfc;
pub mod temporal_signatures;
pub mod world_model;

// Sparse LTC, differentiable HDC, resonator networks, concept crystallization
// (Ported from crates/symthaea-dynamics 2026-02-06)
pub mod crate_world_model;
pub mod crystallization;
pub mod differentiable_hdc;
pub mod ltc;
pub mod resonator;

// CfC/LTC stability analysis: Jacobian, Lyapunov exponents, bifurcations
pub mod stability_analysis;

// Advanced ODE solvers for continuous-time neural dynamics (Euler, RK4,
// Dormand-Prince, Implicit Midpoint, Backward Euler, Exponential Integrator)
pub mod ode_solvers;

// Frequency-domain analysis for neural dynamics signals
pub mod spectral_analysis;

// Stochastic differential equations for neural noise modeling
pub mod stochastic_dynamics;

// Hidden Markov Model for temporal state sequence classification
// (Forward-Backward, Viterbi, Baum-Welch EM)
pub mod hmm;

// Wavelet analysis: DWT (Mallat), CWT (Morlet), spindle/burst detection
pub mod wavelet;

// Phase-Amplitude Coupling: cross-frequency neural coupling analysis
// (Modulation Index, Mean Vector Length, comodulogram)
pub mod phase_amplitude_coupling;

// Narrative arc dynamics using HierarchicalCfC
pub mod narrative_dynamics;

// Multi-scene story session (wraps algebra + dynamics)
pub mod story_session;

// Code understanding dynamics (Consciousness-Aware Code)
#[cfg(feature = "code_generation")]
pub mod cfc_code_sequencer;
#[cfg(feature = "code_generation")]
pub mod temporal_code;

/// A crystallized concept - a stable, consolidated memory representation
///
/// Crystallized concepts represent knowledge that has been consolidated from
/// episodic memories into semantic knowledge through sleep-like processes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrystalizedConcept {
    /// Unique identifier
    pub id: u64,

    /// String-based unique identifier (for dream_mode compatibility)
    #[serde(default)]
    pub uid: String,

    /// Name/label for this concept
    pub name: String,

    /// Semantic description
    pub description: Option<String>,

    /// High-dimensional vector representation
    pub embedding: Vec<f32>,

    /// Attractor signature for consciousness patterns (alias for embedding)
    #[serde(default)]
    pub attractor_signature: Vec<f32>,

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
            uid: format!("concept_{id}"),
            name: name.into(),
            description: None,
            embedding: embedding.clone(),
            attractor_signature: embedding,
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
            uid: format!("concept_{id}"),
            name: name.into(),
            description: Some(description.into()),
            embedding: embedding.clone(),
            attractor_signature: embedding,
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
        self.associations
            .insert(other_id, strength.clamp(-1.0, 1.0));
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

        let dot: f32 = self
            .embedding
            .iter()
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

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// Re-export key types
pub use cfc::{
    // Activation types
    ActivationType,
    CfCCell,
    CfCConfig,
    CfCNetwork,
    CfCNetworkConfig,
    NetworkOnlineLearningStats,
    OnlineLearningConfig,
    OnlineLearningStats,
    // Phi-gated attention
    PhiGatedConfig,
    compute_phi_attention_weights,
};
pub use cfc_coherence::{
    CfCCoherenceBridge, CoherenceConfig, CoherenceSummary, TemporalCoherenceMetrics,
};
pub use cfc_gpu::{GpuBackend, GpuCfcConfig, GpuCfcNetwork, GpuCfcStats};
pub use hierarchical_cfc::{
    DEFAULT_TIME_CONSTANTS, HierarchicalCfC, HierarchicalCfCConfig, HierarchicalOutput,
};
pub use temporal_signatures::{
    ConsciousnessPattern, SignatureConfig, TemporalSignatureEncoder, TemporalStateSummary,
    TrajectoryFeatures,
};
pub use world_model::{HierarchicalCfCWorldModel, WorldModelConfig, WorldModelLayer};

// Ported from crates/symthaea-dynamics (2026-02-06)
pub use crystallization::{
    ConceptCrystallizer,
    CrystalizedConcept as AttractorConcept, // Alias to avoid conflict with existing CrystalizedConcept
    CrystallizationConfig,
    RecurrenceAnalyzer,
    StepResult,
    UnifiedLearningMind,
};
pub use differentiable_hdc::{DifferentiableHDCConfig, DifferentiableHDCEncoder, HDCEncoder};
pub use ltc::{CsrMatrix, IntegrationMethod, LiquidNetwork, LiquidNetworkConfig};
pub use resonator::{Codebook, Episode, ResonatorConfig, ResonatorMemory, ResonatorNetwork};

// Stability analysis for CfC/LTC dynamics
pub use stability_analysis::{
    BifurcationPoint, BifurcationType, FixedPoint, FixedPointType, JacobianResult, LyapunovResult,
    StabilityAnalyzer, StabilityConfig,
};

// ODE solvers for continuous-time neural dynamics
pub use ode_solvers::{OdeConfig, OdeResult, OdeSolver, OdeSolverEngine, OdeSystem, newton_solve};

// Frequency-domain spectral analysis
pub use spectral_analysis::{
    BandPower, CoherenceResult, Complex, FrequencySpectrum, SpectralAnalyzer, SpectralConfig,
    WindowType,
};

// Stochastic differential equations
pub use stochastic_dynamics::{
    FokkerPlanckSolver, LangevinDynamics, OrnsteinUhlenbeck, SdeConfig, SdeResult, SdeSolver,
    SdeStatistics, SdeSystem, SimpleRng, StochasticCfC,
};

// Narrative arc dynamics
pub use narrative_dynamics::{NarrativeSignal, StoryArcConfig, StoryArcDynamics};
pub use story_session::{
    CharacterArc, ConflictEntry, SceneRecord, StorySession, StorySessionSnapshot, StoryState,
};

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction tests ──────────────────────────────────────────

    #[test]
    fn construct_dimension_mismatch() {
        let err = DynamicsError::DimensionMismatch {
            expected: 128,
            actual: 64,
        };
        // Verify the structured fields are accessible via Debug
        let debug = format!("{:?}", err);
        assert!(debug.contains("128"));
        assert!(debug.contains("64"));
    }

    #[test]
    fn construct_invalid_config() {
        let err = DynamicsError::InvalidConfig("tau must be positive".into());
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidConfig"));
    }

    #[test]
    fn construct_computation_error() {
        let err = DynamicsError::ComputationError("matrix singular".into());
        let debug = format!("{:?}", err);
        assert!(debug.contains("ComputationError"));
    }

    #[test]
    fn construct_not_initialized() {
        let err = DynamicsError::NotInitialized;
        let debug = format!("{:?}", err);
        assert!(debug.contains("NotInitialized"));
    }

    // ── Display formatting tests ────────────────────────────────────

    #[test]
    fn display_dimension_mismatch_contains_values() {
        let err = DynamicsError::DimensionMismatch {
            expected: 256,
            actual: 32,
        };
        let msg = err.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("256"),
            "display should include expected dimension"
        );
        assert!(
            msg.contains("32"),
            "display should include actual dimension"
        );
        assert!(
            msg.contains("mismatch") || msg.contains("Mismatch"),
            "display should mention mismatch"
        );
    }

    #[test]
    fn display_invalid_config_includes_inner_message() {
        let inner = "hidden_size must be > 0";
        let err = DynamicsError::InvalidConfig(inner.to_string());
        let msg = err.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains(inner),
            "display should include the inner message"
        );
    }

    #[test]
    fn display_computation_error_includes_inner_message() {
        let inner = "NaN detected in activation";
        let err = DynamicsError::ComputationError(inner.to_string());
        let msg = err.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains(inner),
            "display should include the inner message"
        );
    }

    #[test]
    fn display_not_initialized_is_nonempty() {
        let err = DynamicsError::NotInitialized;
        let msg = err.to_string();
        assert!(!msg.is_empty(), "NotInitialized display must not be empty");
    }

    // ── std::error::Error trait tests ───────────────────────────────

    #[test]
    fn all_variants_implement_std_error() {
        // Verify each variant can be treated as &dyn std::error::Error
        let errors: Vec<Box<dyn std::error::Error>> = vec![
            Box::new(DynamicsError::DimensionMismatch {
                expected: 10,
                actual: 5,
            }),
            Box::new(DynamicsError::InvalidConfig("bad".into())),
            Box::new(DynamicsError::ComputationError("fail".into())),
            Box::new(DynamicsError::NotInitialized),
            Box::new(DynamicsError::Io(std::io::Error::other("test io error"))),
        ];
        for err in &errors {
            // std::error::Error::to_string delegates to Display
            assert!(!err.to_string().is_empty());
        }
    }

    // ── DynamicsResult alias test ───────────────────────────────────

    #[test]
    #[allow(clippy::unnecessary_literal_unwrap)]
    fn dynamics_result_ok() {
        let res: DynamicsResult<i32> = Ok(42);
        assert_eq!(res.unwrap(), 42);
    }

    #[test]
    fn dynamics_result_err() {
        let res: DynamicsResult<i32> = Err(DynamicsError::NotInitialized);
        assert!(res.is_err());
    }

    // ── Edge-case: empty strings ────────────────────────────────────

    #[test]
    fn invalid_config_with_empty_string() {
        let err = DynamicsError::InvalidConfig(String::new());
        // Display should still produce something (the prefix at minimum)
        let msg = err.to_string();
        assert!(
            !msg.is_empty(),
            "even with empty inner, display has a prefix"
        );
    }

    #[test]
    fn computation_error_with_empty_string() {
        let err = DynamicsError::ComputationError(String::new());
        let msg = err.to_string();
        assert!(
            !msg.is_empty(),
            "even with empty inner, display has a prefix"
        );
    }

    // ── Dimension mismatch with equal values ────────────────────────

    #[test]
    fn dimension_mismatch_same_values_still_formats() {
        // Pathological but constructible: expected == actual
        let err = DynamicsError::DimensionMismatch {
            expected: 0,
            actual: 0,
        };
        let msg = err.to_string();
        assert!(!msg.is_empty());
    }

    // ── Debug vs Display are distinct ───────────────────────────────

    #[test]
    fn debug_and_display_differ_for_structured_variant() {
        let err = DynamicsError::DimensionMismatch {
            expected: 100,
            actual: 50,
        };
        let debug = format!("{:?}", err);
        let display = format!("{}", err);
        // Debug includes variant name with struct syntax; Display is human-readable
        assert_ne!(
            debug, display,
            "Debug and Display should produce different output"
        );
    }

    #[test]
    fn debug_and_display_differ_for_unit_variant() {
        let err = DynamicsError::NotInitialized;
        let debug = format!("{:?}", err);
        let display = format!("{}", err);
        // Debug is "NotInitialized", Display is "Not initialized"
        assert_ne!(
            debug, display,
            "Debug and Display should produce different output"
        );
    }
}
