//! # Bridges Module: Cross-Representation Translation
//!
//! This module provides bridges between different neural/cognitive representations
//! in Symthaea, enabling bidirectional information flow between:
//!
//! - **HDC semantic vectors** (high-dimensional, symbolic meaning)
//! - **CfC temporal states** (continuous-time dynamics, sequential patterns)
//! - **Consciousness components** (world models, affective systems, etc.)
//!
//! ## Common Bridge Traits
//!
//! All bridges implement the `ConsciousnessBridge` trait which provides:
//! - Unified observation and integration interfaces
//! - Standard error handling
//! - Bridge health and statistics tracking
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                        HDC-CfC BRIDGE                                       │
//! │                                                                             │
//! │    HDC Domain                              CfC Domain                       │
//! │   (Semantic)                              (Temporal)                        │
//! │                                                                             │
//! │  ┌─────────────┐    encode_semantic_to_temporal    ┌─────────────┐          │
//! │  │ ContinuousHV│ ──────────────────────────────▶   │  CfC State  │          │
//! │  │  (16,384D)  │                                   │   (128D)    │          │
//! │  │             │    decode_temporal_to_semantic    │             │          │
//! │  │  concepts   │ ◀──────────────────────────────── │  dynamics   │          │
//! │  └─────────────┘                                   └─────────────┘          │
//! │                                                                             │
//! │  Projection Matrices:                     Attention Mechanism:              │
//! │  - W_encode: [hidden × hdc_dim]          - Query from HDC                   │
//! │  - W_decode: [hdc_dim × hidden]          - Key/Value from CfC               │
//! │  - Both learnable via Adam               - Selective bridging               │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

pub mod hdc_cfc_bridge;

pub use hdc_cfc_bridge::{
    HdcCfcBridge, HdcCfcBridgeConfig, BridgeAttention, AttentionOutput,
};

// ═══════════════════════════════════════════════════════════════════════════════
// BRIDGE ERROR TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Errors that can occur during bridge operations.
#[derive(Debug, Error)]
pub enum BridgeError {
    /// Source component is not available.
    #[error("Source unavailable: {0}")]
    SourceUnavailable(String),

    /// Target component is not available.
    #[error("Target unavailable: {0}")]
    TargetUnavailable(String),

    /// Translation between representations failed.
    #[error("Translation failed: {0}")]
    TranslationFailed(String),

    /// Bridge is not initialized.
    #[error("Bridge not initialized")]
    NotInitialized,

    /// Bridge capacity exceeded.
    #[error("Bridge capacity exceeded: {current} > {max}")]
    CapacityExceeded { current: usize, max: usize },

    /// Configuration error.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for bridge operations.
pub type BridgeResult<T> = Result<T, BridgeError>;

// ═══════════════════════════════════════════════════════════════════════════════
// CORE BRIDGE TRAITS
// ═══════════════════════════════════════════════════════════════════════════════

/// Semantic input to a consciousness bridge.
///
/// This represents any input that can be observed by a consciousness component.
#[derive(Debug, Clone)]
pub struct SemanticInput {
    /// Unique identifier for this input.
    pub id: String,
    /// Input modality (text, visual, audio, etc.)
    pub modality: InputModality,
    /// High-dimensional embedding of the input.
    pub embedding: Vec<f32>,
    /// Salience/importance score (0.0-1.0).
    pub salience: f32,
    /// Timestamp of the input.
    pub timestamp_ms: u64,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

/// Input modality for semantic inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputModality {
    /// Text/language input.
    Text,
    /// Visual input.
    Visual,
    /// Audio input.
    Audio,
    /// Conceptual/abstract input.
    Conceptual,
    /// Emotional/affective input.
    Affective,
    /// Temporal/sequential input.
    Temporal,
    /// Multi-modal input.
    Multimodal,
}

/// Result of observing an input through a bridge.
#[derive(Debug, Clone)]
pub struct ObservationResult {
    /// Whether the observation was successful.
    pub success: bool,
    /// Transformed representation (if applicable).
    pub transformed: Option<Vec<f32>>,
    /// Confidence in the observation (0.0-1.0).
    pub confidence: f32,
    /// Any insights discovered during observation.
    pub insights: Vec<Insight>,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
}

/// An insight discovered during bridge processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insight {
    /// Insight type/category.
    pub category: String,
    /// Brief description.
    pub description: String,
    /// Relevance score (0.0-1.0).
    pub relevance: f32,
    /// Associated concept IDs.
    pub related_concepts: Vec<String>,
}

/// Result of integrating an insight into the system.
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    /// Whether integration was successful.
    pub success: bool,
    /// Number of concepts affected.
    pub concepts_affected: usize,
    /// Change in consciousness coherence.
    pub coherence_delta: f64,
    /// Any warnings during integration.
    pub warnings: Vec<String>,
}

/// Core trait for consciousness bridges.
///
/// All bridges between consciousness components should implement this trait
/// to ensure a consistent API for observation, integration, and health monitoring.
pub trait ConsciousnessBridge {
    /// Bridge name for identification.
    fn name(&self) -> &str;

    /// Observe an input through the bridge.
    ///
    /// This transforms the input from the source representation to the
    /// target representation, returning any observations and insights.
    fn observe(&mut self, input: &SemanticInput) -> BridgeResult<ObservationResult>;

    /// Integrate an insight back into the source system.
    ///
    /// This allows bidirectional flow of information.
    fn integrate(&mut self, insight: Insight) -> BridgeResult<IntegrationResult>;

    /// Check if the bridge is healthy and ready for use.
    fn is_healthy(&self) -> bool;

    /// Get bridge statistics.
    fn stats(&self) -> BridgeStats;

    /// Reset the bridge to its initial state.
    fn reset(&mut self);
}

/// Statistics for bridge monitoring.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BridgeStats {
    /// Total observations processed.
    pub total_observations: u64,
    /// Successful observations.
    pub successful_observations: u64,
    /// Total insights generated.
    pub total_insights: u64,
    /// Total integrations.
    pub total_integrations: u64,
    /// Average processing time in microseconds.
    pub avg_processing_time_us: f64,
    /// Current bridge health (0.0-1.0).
    pub health: f64,
    /// Capacity utilization (0.0-1.0).
    pub capacity_utilization: f64,
}

impl BridgeStats {
    /// Success rate of observations.
    pub fn success_rate(&self) -> f64 {
        if self.total_observations == 0 {
            return 1.0;
        }
        self.successful_observations as f64 / self.total_observations as f64
    }

    /// Insights per observation.
    pub fn insights_per_observation(&self) -> f64 {
        if self.successful_observations == 0 {
            return 0.0;
        }
        self.total_insights as f64 / self.successful_observations as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BIDIRECTIONAL BRIDGE TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for bridges that support bidirectional transformation.
///
/// This extends `ConsciousnessBridge` with explicit encode/decode methods
/// for converting between two representation spaces.
pub trait BidirectionalBridge: ConsciousnessBridge {
    /// Source representation type.
    type Source;
    /// Target representation type.
    type Target;

    /// Encode from source to target representation.
    fn encode(&self, source: &Self::Source) -> BridgeResult<Self::Target>;

    /// Decode from target back to source representation.
    fn decode(&self, target: &Self::Target) -> BridgeResult<Self::Source>;

    /// Round-trip accuracy: encode → decode → compare.
    fn roundtrip_accuracy(&self, source: &Self::Source) -> BridgeResult<f64>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONCEPT BRIDGE TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for bridges that work with crystallized concepts.
///
/// These bridges connect concept crystallization to other systems
/// (attention, affect, memory, etc.).
pub trait ConceptBridge: ConsciousnessBridge {
    /// Process a newly crystallized concept.
    fn on_concept_crystallized(&mut self, concept: &ConceptRef) -> BridgeResult<()>;

    /// Get pending actions for the target system.
    fn pending_actions(&self) -> Vec<BridgeAction>;

    /// Acknowledge that an action was processed.
    fn acknowledge_action(&mut self, action_id: &str) -> BridgeResult<()>;
}

/// Reference to a crystallized concept.
#[derive(Debug, Clone)]
pub struct ConceptRef {
    /// Concept unique ID.
    pub uid: String,
    /// Concept name.
    pub name: String,
    /// Concept embedding.
    pub embedding: Vec<f32>,
    /// Activation count.
    pub activation_count: u64,
    /// Confidence score.
    pub confidence: f32,
}

/// Action generated by a bridge for the target system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeAction {
    /// Action unique ID.
    pub id: String,
    /// Action type.
    pub action_type: String,
    /// Priority (0.0-1.0).
    pub priority: f32,
    /// Payload data.
    pub payload: HashMap<String, String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BRIDGE REGISTRY
// ═══════════════════════════════════════════════════════════════════════════════

/// Registry for managing multiple bridges.
pub struct BridgeRegistry {
    /// Registered bridges by name.
    bridges: HashMap<String, Box<dyn ConsciousnessBridge + Send + Sync>>,
    /// Bridge connection graph (from → to).
    connections: HashMap<String, Vec<String>>,
}

impl BridgeRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            bridges: HashMap::new(),
            connections: HashMap::new(),
        }
    }

    /// Register a bridge.
    pub fn register<B: ConsciousnessBridge + Send + Sync + 'static>(
        &mut self,
        bridge: B,
    ) {
        let name = bridge.name().to_string();
        self.bridges.insert(name, Box::new(bridge));
    }

    /// Connect two bridges (from → to).
    pub fn connect(&mut self, from: &str, to: &str) {
        self.connections
            .entry(from.to_string())
            .or_default()
            .push(to.to_string());
    }

    /// Get a bridge by name.
    pub fn get(&self, name: &str) -> Option<&(dyn ConsciousnessBridge + Send + Sync)> {
        self.bridges.get(name).map(|b| b.as_ref())
    }

    /// Get a mutable bridge by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut (dyn ConsciousnessBridge + Send + Sync + '_)> {
        self.bridges.get_mut(name).map(move |b| &mut **b)
    }

    /// Get all bridge names.
    pub fn names(&self) -> Vec<&str> {
        self.bridges.keys().map(|s| s.as_str()).collect()
    }

    /// Get aggregate stats for all bridges.
    pub fn aggregate_stats(&self) -> BridgeStats {
        let mut total = BridgeStats::default();
        let mut count = 0;

        for bridge in self.bridges.values() {
            let stats = bridge.stats();
            total.total_observations += stats.total_observations;
            total.successful_observations += stats.successful_observations;
            total.total_insights += stats.total_insights;
            total.total_integrations += stats.total_integrations;
            total.health += stats.health;
            count += 1;
        }

        if count > 0 {
            total.health /= count as f64;
        }

        total
    }

    /// Check health of all bridges.
    pub fn all_healthy(&self) -> bool {
        self.bridges.values().all(|b| b.is_healthy())
    }
}

impl Default for BridgeRegistry {
    fn default() -> Self {
        Self::new()
    }
}
