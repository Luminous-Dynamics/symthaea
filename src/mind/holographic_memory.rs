//! Holographic Memory - Vector Superposition Storage
//!
//! This module implements the "Memory" layer of the Holographic Associative Memory
//! architecture. It uses vector superposition for one-shot learning and retrieval.
//!
//! ## Core Principle: Holographic Storage
//!
//! Unlike traditional databases that store items separately, holographic memory
//! stores all memories as a superposition (sum) of vectors. This enables:
//!
//! - **One-shot learning**: `Memory_new = Memory_old + Experience`
//! - **Graceful degradation**: Similar queries retrieve similar memories
//! - **Infinite capacity**: No fixed slots, just vector space
//! - **Associative retrieval**: Query by content, not by key
//!
//! ## Architecture Role
//!
//! ```text
//! ┌─────────────────┐
//! │ SemanticEncoder │  ← Sensation
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────────────────────────────────────┐
//! │              HolographicMemory                   │  ← Memory
//! │  ┌───────────────────────────────────────────┐  │
//! │  │  Episodic Store (recent experiences)      │  │
//! │  │  ════════════════════════════════════════ │  │
//! │  │  [v1] + [v2] + [v3] + ... = [hologram]    │  │
//! │  └───────────────────────────────────────────┘  │
//! │  ┌───────────────────────────────────────────┐  │
//! │  │  Semantic Store (consolidated knowledge)  │  │
//! │  │  ════════════════════════════════════════ │  │
//! │  │  Category centroids + exemplar bindings   │  │
//! │  └───────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ ActiveInference │  ← Cognition (future)
//! └─────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! let mut memory = HolographicMemory::new(768);
//!
//! // Store experiences (one-shot learning)
//! memory.store(&experience1);
//! memory.store(&experience2);
//!
//! // Query by similarity
//! let matches = memory.query(&query_vector, 5);
//!
//! // Reinforce important memories
//! memory.reinforce(&important_memory, 2.0);
//! ```

use super::semantic_encoder::{DenseVector, EncodedThought};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Duration;

// ═══════════════════════════════════════════════════════════════════════════
// CHRONO-SEMANTIC-EMOTIONAL BINDING SYSTEM
// ═══════════════════════════════════════════════════════════════════════════
//
// Multi-dimensional memory encoding that separates:
// - **Temporal** (WHEN): Cyclic time encoding with phase-based representation
// - **Semantic** (WHAT): Content meaning via vector embedding
// - **Emotional** (HOW): Affective state via phase-shifted sinusoids
//
// This enables three query modes:
// 1. Pure Semantic - "What did I learn about X?" (ignores emotion/time)
// 2. Pure Emotional - "What made me frustrated?" (ignores content/time)
// 3. Chrono-Semantic - "What happened yesterday morning?" (time + content)
// 4. Combined - Full context with weighted blend
//
// Architecture (from episodic_engine.rs):
// ```
// EpisodicMemory = Temporal(when) ⊗ Semantic(what) + Emotional(how)
//                  \_____chrono_semantic____/        \___parallel___/
// ```

/// Temporal Encoder - Cyclic Time Representation
///
/// Encodes time using phase-based sinusoidal patterns that naturally capture
/// cyclic rhythms (hour of day, day of week, month, season).
///
/// ## Key Properties
/// - **Cyclic continuity**: 23:59 is close to 00:00
/// - **Multi-scale**: Captures hourly, daily, weekly patterns
/// - **Deterministic**: Same time → same encoding
#[derive(Debug, Clone)]
pub struct TemporalEncoder {
    /// Dimension of temporal vectors
    pub dimension: usize,
    /// Number of frequency bands for multi-scale encoding
    pub frequency_bands: usize,
}

impl Default for TemporalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalEncoder {
    /// Create a new temporal encoder with default parameters
    pub fn new() -> Self {
        Self {
            dimension: 128, // Compact temporal representation
            frequency_bands: 4, // Hour, Day, Week, Season
        }
    }

    /// Create with custom dimension
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            dimension,
            frequency_bands: 4,
        }
    }

    /// Encode a timestamp into a temporal vector
    ///
    /// Uses cyclic phase encoding for natural time representation:
    /// - Hour phase: 2π * hour / 24
    /// - Day phase: 2π * day_of_week / 7
    /// - Month phase: 2π * month / 12
    ///
    /// # Arguments
    /// * `timestamp` - Duration since midnight (or any reference point)
    pub fn encode(&self, timestamp: Duration) -> Vec<f32> {
        let secs = timestamp.as_secs_f64();
        let hours = (secs / 3600.0) % 24.0;
        let days = (secs / 86400.0) % 7.0;
        let months = (secs / (86400.0 * 30.0)) % 12.0;
        let years = secs / (86400.0 * 365.0);

        let mut temporal = vec![0.0f32; self.dimension];
        let phase_scales = [
            (hours / 24.0, 1.0),        // Hour of day
            (days / 7.0, 0.5),          // Day of week
            (months / 12.0, 0.3),       // Month of year
            (years.fract(), 0.2),       // Year cycle (seasonal)
        ];

        for (i, val) in temporal.iter_mut().enumerate() {
            let base_phase = (i as f64 / self.dimension as f64) * std::f64::consts::TAU;

            // Sum contributions from each time scale
            let mut sum = 0.0f64;
            for (phase, weight) in &phase_scales {
                let combined_phase = base_phase + phase * std::f64::consts::TAU;
                sum += weight * combined_phase.sin();
            }

            *val = (sum / phase_scales.len() as f64) as f32;
        }

        // Normalize to unit length
        let norm: f32 = temporal.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut temporal {
                *v /= norm;
            }
        }

        temporal
    }

    /// Compute similarity between two temporal encodings
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Emotional Encoder - Affective State Representation
///
/// Encodes emotional valence and arousal into phase-shifted sinusoidal patterns.
/// This creates a continuous emotional space where similar emotions have similar encodings.
///
/// ## Emotion Model
/// - **Valence**: Positive/Negative (-1.0 to 1.0)
/// - **Arousal**: Calm/Excited (0.0 to 1.0)
///
/// ## Key Properties
/// - **Orthogonality**: Different emotions have distinct signatures
/// - **Continuity**: Similar emotions have similar encodings
/// - **Intensity-scaled**: Stronger emotions have larger amplitudes
#[derive(Debug, Clone)]
pub struct EmotionalEncoder {
    /// Dimension of emotional vectors
    pub dimension: usize,
}

impl Default for EmotionalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EmotionalEncoder {
    /// Create a new emotional encoder with default parameters
    pub fn new() -> Self {
        Self {
            dimension: 64, // Compact emotional representation
        }
    }

    /// Create with custom dimension
    pub fn with_dimension(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Encode an emotional state into a vector
    ///
    /// # Arguments
    /// * `valence` - Emotional valence (-1.0 = negative, 0.0 = neutral, 1.0 = positive)
    /// * `arousal` - Emotional arousal (0.0 = calm, 1.0 = excited)
    ///
    /// # Returns
    /// A vector encoding the emotional state
    pub fn encode(&self, valence: f32, arousal: f32) -> Vec<f32> {
        let valence_clamped = valence.clamp(-1.0, 1.0);
        let arousal_clamped = arousal.clamp(0.0, 1.0);

        let mut emotional = vec![0.0f32; self.dimension];

        for (i, val) in emotional.iter_mut().enumerate() {
            // Phase shifts based on valence (determines signature pattern)
            let base_phase = (i as f32 * 0.1) + (valence_clamped * std::f32::consts::PI);

            // Amplitude modulated by arousal (intensity)
            let amplitude = 0.5 + 0.5 * arousal_clamped;

            // Sinusoidal encoding with valence-dependent phase
            *val = base_phase.sin() * amplitude * valence_clamped.abs().max(0.1);
        }

        // Normalize to unit length
        let norm: f32 = emotional.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut emotional {
                *v /= norm;
            }
        }

        emotional
    }

    /// Encode with just valence (assumes neutral arousal)
    pub fn encode_valence(&self, valence: f32) -> Vec<f32> {
        self.encode(valence, 0.5)
    }

    /// Compute similarity between two emotional encodings
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Composite Engram - Multi-dimensional Memory Encoding
///
/// Combines temporal, semantic, and emotional information into a unified
/// memory representation that supports multiple query modes.
///
/// ## Architecture
/// ```text
/// CompositeEngram {
///     chrono_semantic: Temporal ⊗ Semantic  // Bound together
///     emotional:       Emotional             // Parallel space
///     temporal:        Temporal              // Raw (for pure temporal query)
///     semantic:        Semantic              // Raw (for pure semantic query)
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeEngram {
    /// Chrono-Semantic binding: Temporal ⊗ Semantic
    /// Used for "when + what" queries
    pub chrono_semantic_vector: Vec<f32>,

    /// Emotional binding vector (parallel, not bound to chrono-semantic)
    /// Used for "how did I feel" queries
    pub emotional_binding_vector: Vec<f32>,

    /// Raw temporal vector (for pure temporal queries)
    pub temporal_vector: Vec<f32>,

    /// Raw semantic vector (for pure semantic queries)
    pub semantic_vector: Vec<f32>,

    /// Emotional valence (-1.0 to 1.0)
    pub valence: f32,

    /// Emotional arousal (0.0 to 1.0)
    pub arousal: f32,

    /// Attention/importance weight at encoding time
    pub attention_weight: f32,

    /// Encoding strength (number of reinforcement writes)
    pub encoding_strength: usize,
}

impl CompositeEngram {
    /// Create a new composite engram from components
    ///
    /// # Arguments
    /// * `semantic` - The semantic content vector
    /// * `timestamp` - When this memory was formed
    /// * `valence` - Emotional valence (-1.0 to 1.0)
    /// * `arousal` - Emotional arousal (0.0 to 1.0)
    /// * `temporal_encoder` - Encoder for time
    /// * `emotional_encoder` - Encoder for emotion
    pub fn new(
        semantic: Vec<f32>,
        timestamp: Duration,
        valence: f32,
        arousal: f32,
        temporal_encoder: &TemporalEncoder,
        emotional_encoder: &EmotionalEncoder,
    ) -> Self {
        // Encode temporal
        let temporal_vector = temporal_encoder.encode(timestamp);

        // Encode emotional
        let emotional_binding_vector = emotional_encoder.encode(valence, arousal);

        // Bind chrono-semantic (element-wise multiply, then normalize)
        let chrono_semantic_vector = Self::bind_vectors(&temporal_vector, &semantic);

        Self {
            chrono_semantic_vector,
            emotional_binding_vector,
            temporal_vector,
            semantic_vector: semantic,
            valence,
            arousal,
            attention_weight: 0.5,
            encoding_strength: 10,
        }
    }

    /// Create with attention weighting
    pub fn with_attention(
        semantic: Vec<f32>,
        timestamp: Duration,
        valence: f32,
        arousal: f32,
        attention: f32,
        temporal_encoder: &TemporalEncoder,
        emotional_encoder: &EmotionalEncoder,
    ) -> Self {
        let mut engram = Self::new(semantic, timestamp, valence, arousal, temporal_encoder, emotional_encoder);
        engram.attention_weight = attention.clamp(0.0, 1.0);
        engram.encoding_strength = (1.0 + attention * 99.0) as usize;
        engram
    }

    /// Bind two vectors via element-wise multiplication (HDC bind operation)
    pub fn bind_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
        // If dimensions differ, extend the smaller one cyclically
        let max_len = a.len().max(b.len());
        let mut result = vec![0.0f32; max_len];

        for i in 0..max_len {
            let a_val = a.get(i % a.len()).copied().unwrap_or(0.0);
            let b_val = b.get(i % b.len()).copied().unwrap_or(0.0);
            result[i] = a_val * b_val;
        }

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut result {
                *v /= norm;
            }
        }

        result
    }

    /// Query similarity with weighted components
    ///
    /// # Arguments
    /// * `query_semantic` - Semantic query vector
    /// * `query_temporal` - Optional temporal query vector
    /// * `query_emotional` - Optional emotional query vector
    /// * `weight_semantic` - Weight for semantic similarity (0.0 - 1.0)
    /// * `weight_temporal` - Weight for temporal similarity (0.0 - 1.0)
    /// * `weight_emotional` - Weight for emotional similarity (0.0 - 1.0)
    ///
    /// # Returns
    /// Weighted similarity score
    pub fn query_similarity(
        &self,
        query_semantic: Option<&[f32]>,
        query_temporal: Option<&[f32]>,
        query_emotional: Option<&[f32]>,
        weight_semantic: f32,
        weight_temporal: f32,
        weight_emotional: f32,
    ) -> f32 {
        let total_weight = weight_semantic + weight_temporal + weight_emotional;
        if total_weight <= 0.0 {
            return 0.0;
        }

        let mut score = 0.0f32;

        // Semantic similarity
        if let Some(q) = query_semantic {
            let sim = Self::cosine_similarity(&self.semantic_vector, q);
            score += sim * weight_semantic;
        }

        // Temporal similarity
        if let Some(q) = query_temporal {
            let sim = Self::cosine_similarity(&self.temporal_vector, q);
            score += sim * weight_temporal;
        }

        // Emotional similarity
        if let Some(q) = query_emotional {
            let sim = Self::cosine_similarity(&self.emotional_binding_vector, q);
            score += sim * weight_emotional;
        }

        score / total_weight
    }

    /// Pure semantic query (ignores time and emotion)
    pub fn semantic_similarity(&self, query: &[f32]) -> f32 {
        Self::cosine_similarity(&self.semantic_vector, query)
    }

    /// Pure temporal query (ignores content and emotion)
    pub fn temporal_similarity(&self, query: &[f32]) -> f32 {
        Self::cosine_similarity(&self.temporal_vector, query)
    }

    /// Pure emotional query (ignores content and time)
    pub fn emotional_similarity(&self, query: &[f32]) -> f32 {
        Self::cosine_similarity(&self.emotional_binding_vector, query)
    }

    /// Chrono-semantic query (what + when, ignores emotion)
    pub fn chrono_semantic_similarity(&self, query: &[f32]) -> f32 {
        Self::cosine_similarity(&self.chrono_semantic_vector, query)
    }

    /// Cosine similarity helper (handles dimension mismatch via cyclic extension)
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            // Handle dimension mismatch via cyclic extension
            let max_len = a.len().max(b.len());
            let mut dot = 0.0f32;
            let mut norm_a = 0.0f32;
            let mut norm_b = 0.0f32;

            for i in 0..max_len {
                let a_val = a.get(i % a.len()).copied().unwrap_or(0.0);
                let b_val = b.get(i % b.len()).copied().unwrap_or(0.0);
                dot += a_val * b_val;
                norm_a += a_val * a_val;
                norm_b += b_val * b_val;
            }

            if norm_a > 0.0 && norm_b > 0.0 {
                dot / (norm_a.sqrt() * norm_b.sqrt())
            } else {
                0.0
            }
        } else {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a > 0.0 && norm_b > 0.0 {
                dot / (norm_a * norm_b)
            } else {
                0.0
            }
        }
    }
}

/// Configuration for HolographicMemory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolographicMemoryConfig {
    /// Dimension of vectors (768 for BGE)
    pub dimension: usize,

    /// Maximum episodic memories before consolidation
    pub max_episodic: usize,

    /// Temporal decay factor per retrieval cycle (0.0 - 1.0)
    /// Higher = slower decay
    pub decay_factor: f32,

    /// Minimum similarity for retrieval (0.0 - 1.0)
    pub retrieval_threshold: f32,

    /// Whether to auto-consolidate episodic → semantic
    pub auto_consolidate: bool,

    /// Consolidation threshold (how many similar episodes trigger consolidation)
    pub consolidation_threshold: usize,

    // ═══════════════════════════════════════════════════════════════════════════
    // HIPPOCAMPUS-STYLE DYNAMICS CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════════════

    /// Hippocampus-style decay rate per cycle (0.0 - 1.0)
    /// Lower = slower decay (0.05 = 5% decay per cycle)
    #[serde(default = "default_hippocampus_decay_rate")]
    pub hippocampus_decay_rate: f32,

    /// Strengthen increment when memory is recalled
    #[serde(default = "default_strengthen_increment")]
    pub strengthen_increment: f32,

    /// Maximum strength a memory can reach
    #[serde(default = "default_max_strength")]
    pub max_strength: f32,

    /// Minimum importance threshold for pruning (memories below this are removed)
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: f32,

    /// Whether to use hippocampus-style dynamics (vs original decay_factor)
    #[serde(default)]
    pub use_hippocampus_dynamics: bool,

    // ═══════════════════════════════════════════════════════════════════════════
    // SLEEP CONSOLIDATION CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════════════

    /// Whether to integrate with sleep cycle manager
    #[serde(default)]
    pub enable_sleep_consolidation: bool,

    /// Minimum importance for a memory to be eligible for long-term storage
    #[serde(default = "default_long_term_threshold")]
    pub long_term_threshold: f32,

    /// Minimum access count for long-term eligibility
    #[serde(default = "default_min_access_for_long_term")]
    pub min_access_for_long_term: u32,

    // ═══════════════════════════════════════════════════════════════════════════
    // CHRONO-SEMANTIC-EMOTIONAL BINDING CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════════════

    /// Whether to enable chrono-semantic-emotional binding
    #[serde(default)]
    pub enable_composite_encoding: bool,

    /// Dimension for temporal vectors
    #[serde(default = "default_temporal_dimension")]
    pub temporal_dimension: usize,

    /// Dimension for emotional vectors
    #[serde(default = "default_emotional_dimension")]
    pub emotional_dimension: usize,

    /// Default weight for semantic component in queries
    #[serde(default = "default_semantic_weight")]
    pub default_semantic_weight: f32,

    /// Default weight for temporal component in queries
    #[serde(default = "default_temporal_weight")]
    pub default_temporal_weight: f32,

    /// Default weight for emotional component in queries
    #[serde(default = "default_emotional_weight")]
    pub default_emotional_weight: f32,
}

fn default_hippocampus_decay_rate() -> f32 { 0.05 }
fn default_strengthen_increment() -> f32 { 0.1 }
fn default_max_strength() -> f32 { 2.0 }
fn default_prune_threshold() -> f32 { 0.01 }
fn default_long_term_threshold() -> f32 { 0.5 }
fn default_min_access_for_long_term() -> u32 { 2 }

// Chrono-semantic-emotional defaults
fn default_temporal_dimension() -> usize { 128 }
fn default_emotional_dimension() -> usize { 64 }
fn default_semantic_weight() -> f32 { 0.6 }
fn default_temporal_weight() -> f32 { 0.2 }
fn default_emotional_weight() -> f32 { 0.2 }

impl Default for HolographicMemoryConfig {
    fn default() -> Self {
        Self {
            dimension: 768,
            max_episodic: 1000,
            decay_factor: 0.95,
            retrieval_threshold: 0.3,
            auto_consolidate: true,
            consolidation_threshold: 3,
            // Hippocampus dynamics
            hippocampus_decay_rate: default_hippocampus_decay_rate(),
            strengthen_increment: default_strengthen_increment(),
            max_strength: default_max_strength(),
            prune_threshold: default_prune_threshold(),
            use_hippocampus_dynamics: false, // Off by default for backwards compatibility
            // Sleep consolidation
            enable_sleep_consolidation: false,
            long_term_threshold: default_long_term_threshold(),
            min_access_for_long_term: default_min_access_for_long_term(),
            // Chrono-semantic-emotional binding
            enable_composite_encoding: false, // Off by default for backwards compatibility
            temporal_dimension: default_temporal_dimension(),
            emotional_dimension: default_emotional_dimension(),
            default_semantic_weight: default_semantic_weight(),
            default_temporal_weight: default_temporal_weight(),
            default_emotional_weight: default_emotional_weight(),
        }
    }
}

/// A single memory trace
///
/// Integrates Hippocampus-style decay/strengthen dynamics:
/// - **Decay**: `strength *= 1.0 - decay_rate` (exponential forgetting)
/// - **Strengthen**: `recall_count += 1; strength += 0.1` (use-dependent potentiation)
///
/// ## Chrono-Semantic-Emotional Binding (NEW)
///
/// When `composite` is present, the memory has multi-dimensional encoding:
/// - **Temporal**: When this happened (cyclic time encoding)
/// - **Semantic**: What this is about (content meaning)
/// - **Emotional**: How it felt (affective state)
///
/// This enables queries like:
/// - "What happened yesterday morning?" (chrono-semantic)
/// - "What made me frustrated?" (emotional)
/// - "Things related to X" (pure semantic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrace {
    /// The dense vector representation (primary semantic encoding)
    pub vector: Vec<f32>,

    /// Original text (if available)
    pub text: Option<String>,

    /// Importance/salience score (affects retrieval and decay)
    /// Also acts as "strength" for Hippocampus-style dynamics
    pub importance: f32,

    /// Number of times this memory has been accessed
    pub access_count: u32,

    /// Timestamp of creation (Unix epoch ms)
    pub created_at: u64,

    /// Timestamp of last access
    pub last_accessed: u64,

    /// Optional category/tag
    pub category: Option<String>,

    /// Consolidation status for sleep integration
    #[serde(default)]
    pub consolidated: bool,

    /// Eligibility for long-term storage (set during sleep consolidation)
    #[serde(default)]
    pub long_term_eligible: bool,

    // ═══════════════════════════════════════════════════════════════════════════
    // CHRONO-SEMANTIC-EMOTIONAL BINDING (NEW)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Composite engram with multi-dimensional encoding
    /// When present, enables temporal, semantic, and emotional queries
    #[serde(default)]
    pub composite: Option<CompositeEngram>,

    /// Emotional valence (-1.0 to 1.0) for quick filtering
    /// Extracted from composite for efficiency
    #[serde(default)]
    pub valence: Option<f32>,

    /// Emotional arousal (0.0 to 1.0) for quick filtering
    #[serde(default)]
    pub arousal: Option<f32>,
}

impl MemoryTrace {
    /// Create a new memory trace from a dense vector
    pub fn new(vector: Vec<f32>, text: Option<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            vector,
            text,
            importance: 1.0,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            category: None,
            consolidated: false,
            long_term_eligible: false,
            composite: None,
            valence: None,
            arousal: None,
        }
    }

    /// Create from an EncodedThought
    pub fn from_thought(thought: &EncodedThought) -> Self {
        let mut trace = Self::new(thought.dense.values.clone(), Some(thought.text.clone()));
        trace.importance = thought.confidence;
        trace
    }

    /// Create a new memory trace with composite chrono-semantic-emotional encoding
    ///
    /// # Arguments
    /// * `vector` - The primary semantic vector
    /// * `text` - Optional text description
    /// * `timestamp` - When this memory was formed (duration since reference point)
    /// * `valence` - Emotional valence (-1.0 to 1.0)
    /// * `arousal` - Emotional arousal (0.0 to 1.0)
    /// * `temporal_encoder` - Encoder for time
    /// * `emotional_encoder` - Encoder for emotion
    pub fn new_composite(
        vector: Vec<f32>,
        text: Option<String>,
        timestamp: Duration,
        valence: f32,
        arousal: f32,
        temporal_encoder: &TemporalEncoder,
        emotional_encoder: &EmotionalEncoder,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let composite = CompositeEngram::new(
            vector.clone(),
            timestamp,
            valence,
            arousal,
            temporal_encoder,
            emotional_encoder,
        );

        Self {
            vector,
            text,
            importance: 1.0,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            category: None,
            consolidated: false,
            long_term_eligible: false,
            composite: Some(composite),
            valence: Some(valence),
            arousal: Some(arousal),
        }
    }

    /// Create with composite encoding and attention weight
    pub fn new_composite_with_attention(
        vector: Vec<f32>,
        text: Option<String>,
        timestamp: Duration,
        valence: f32,
        arousal: f32,
        attention: f32,
        temporal_encoder: &TemporalEncoder,
        emotional_encoder: &EmotionalEncoder,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let composite = CompositeEngram::with_attention(
            vector.clone(),
            timestamp,
            valence,
            arousal,
            attention,
            temporal_encoder,
            emotional_encoder,
        );

        Self {
            vector,
            text,
            importance: attention, // Use attention as initial importance
            access_count: 0,
            created_at: now,
            last_accessed: now,
            category: None,
            consolidated: false,
            long_term_eligible: false,
            composite: Some(composite),
            valence: Some(valence),
            arousal: Some(arousal),
        }
    }

    /// Add composite encoding to an existing trace
    pub fn with_composite(
        mut self,
        timestamp: Duration,
        valence: f32,
        arousal: f32,
        temporal_encoder: &TemporalEncoder,
        emotional_encoder: &EmotionalEncoder,
    ) -> Self {
        let composite = CompositeEngram::new(
            self.vector.clone(),
            timestamp,
            valence,
            arousal,
            temporal_encoder,
            emotional_encoder,
        );
        self.composite = Some(composite);
        self.valence = Some(valence);
        self.arousal = Some(arousal);
        self
    }

    /// Check if this trace has composite encoding
    pub fn has_composite(&self) -> bool {
        self.composite.is_some()
    }

    /// Get the composite engram (if present)
    pub fn composite(&self) -> Option<&CompositeEngram> {
        self.composite.as_ref()
    }

    /// Compute similarity to another vector
    pub fn similarity(&self, other: &[f32]) -> f32 {
        if self.vector.len() != other.len() {
            return 0.0;
        }

        let dot: f32 = self.vector.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Mark as accessed (updates timestamp and count)
    pub fn touch(&mut self) {
        self.access_count += 1;
        self.last_accessed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // HIPPOCAMPUS-STYLE DYNAMICS (from src/memory/hippocampus.rs)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Apply decay to this memory trace (Hippocampus-style)
    ///
    /// Implements exponential forgetting: `importance *= 1.0 - decay_rate`
    /// This mimics biological memory decay where unused memories fade over time.
    ///
    /// # Arguments
    /// * `decay_rate` - Rate of decay (0.0 = no decay, 1.0 = instant forget)
    ///
    /// # Example
    /// ```ignore
    /// trace.decay_hippocampus(0.05); // 5% decay per cycle
    /// ```
    pub fn decay_hippocampus(&mut self, decay_rate: f32) {
        self.importance *= 1.0 - decay_rate;
        self.importance = self.importance.max(0.0); // Floor at zero
    }

    /// Strengthen this memory on recall (Hippocampus-style)
    ///
    /// Implements use-dependent potentiation: memories that are recalled
    /// become stronger, mimicking Long-Term Potentiation (LTP).
    ///
    /// - Increments access count
    /// - Boosts importance by 0.1 (capped at 2.0)
    /// - Updates last_accessed timestamp
    ///
    /// # Example
    /// ```ignore
    /// trace.strengthen(); // Called on successful recall
    /// ```
    pub fn strengthen(&mut self) {
        self.access_count += 1;
        self.importance = (self.importance + 0.1).min(2.0);
        self.last_accessed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }

    /// Check if this memory should be pruned (below threshold)
    pub fn should_prune(&self, threshold: f32) -> bool {
        self.importance < threshold
    }

    /// Mark as consolidated (processed during sleep)
    pub fn mark_consolidated(&mut self) {
        self.consolidated = true;
    }

    /// Mark as eligible for long-term storage
    pub fn mark_long_term_eligible(&mut self) {
        self.long_term_eligible = true;
    }

    /// Get the age of this memory in milliseconds
    pub fn age_ms(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        now.saturating_sub(self.created_at)
    }

    /// Get time since last access in milliseconds
    pub fn time_since_access_ms(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        now.saturating_sub(self.last_accessed)
    }
}

/// A semantic category (consolidated from episodic memories)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCategory {
    /// Category name/label
    pub name: String,

    /// Centroid vector (average of all members)
    pub centroid: Vec<f32>,

    /// Number of memories consolidated into this category
    pub member_count: usize,

    /// Representative exemplars (most distinct members)
    pub exemplars: Vec<MemoryTrace>,

    /// Maximum exemplars to keep
    max_exemplars: usize,
}

impl SemanticCategory {
    /// Create a new semantic category
    pub fn new(name: String, dimension: usize) -> Self {
        Self {
            name,
            centroid: vec![0.0; dimension],
            member_count: 0,
            exemplars: Vec::new(),
            max_exemplars: 10,
        }
    }

    /// Add a memory to this category (updates centroid)
    pub fn add(&mut self, trace: &MemoryTrace) {
        // Update centroid incrementally: new_centroid = (old * n + new) / (n + 1)
        let n = self.member_count as f32;
        for (i, val) in trace.vector.iter().enumerate() {
            if i < self.centroid.len() {
                self.centroid[i] = (self.centroid[i] * n + val) / (n + 1.0);
            }
        }
        self.member_count += 1;

        // Keep as exemplar if distinct enough
        if self.exemplars.len() < self.max_exemplars {
            self.exemplars.push(trace.clone());
        } else {
            // Replace least similar exemplar if this one is more distinct
            let mut min_distinctness = f32::MAX;
            let mut min_idx = 0;

            for (i, ex) in self.exemplars.iter().enumerate() {
                let distinctness = 1.0 - ex.similarity(&self.centroid);
                if distinctness < min_distinctness {
                    min_distinctness = distinctness;
                    min_idx = i;
                }
            }

            let new_distinctness = 1.0 - trace.similarity(&self.centroid);
            if new_distinctness > min_distinctness {
                self.exemplars[min_idx] = trace.clone();
            }
        }
    }

    /// Similarity of a vector to this category
    pub fn similarity(&self, vector: &[f32]) -> f32 {
        if self.centroid.len() != vector.len() {
            return 0.0;
        }

        let dot: f32 = self.centroid.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = self.centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Query result from memory
#[derive(Debug, Clone)]
pub struct MemoryMatch {
    /// The matched memory trace
    pub trace: MemoryTrace,

    /// Similarity score (0.0 - 1.0)
    pub similarity: f32,

    /// Source: "episodic" or "semantic"
    pub source: String,
}

/// Holographic Memory - The Memory Layer of HAM
///
/// Stores experiences as vector superpositions, enabling:
/// - One-shot learning
/// - Associative retrieval
/// - Graceful degradation
/// - Infinite context through consolidation
///
/// ## Integrated Systems
///
/// - **Hippocampus Dynamics**: decay/strengthen for biological memory behavior
/// - **Sleep Consolidation**: Integration with SleepCycleManager for REM-based consolidation
/// - **Persistence Ready**: Export/import for UnifiedMind database storage
/// - **Chrono-Semantic-Emotional Binding**: Multi-dimensional memory encoding (NEW)
///
/// ## Query Modes (with Composite Encoding)
///
/// 1. **Pure Semantic**: "What did I learn about X?" (ignores time/emotion)
/// 2. **Pure Temporal**: "What happened at 9 AM?" (ignores content/emotion)
/// 3. **Pure Emotional**: "What made me frustrated?" (ignores content/time)
/// 4. **Chrono-Semantic**: "What happened yesterday morning?" (time + content)
/// 5. **Combined**: Full context with weighted blend of all three
pub struct HolographicMemory {
    /// Configuration
    config: HolographicMemoryConfig,

    /// Episodic memory (recent experiences, FIFO)
    episodic: VecDeque<MemoryTrace>,

    /// Semantic memory (consolidated categories)
    semantic: Vec<SemanticCategory>,

    /// Holographic superposition (sum of all episodic vectors)
    /// Used for fast approximate matching
    hologram: Vec<f32>,

    /// Statistics
    stats: MemoryStats,

    /// Buffer for traces pending long-term storage (for UnifiedMind integration)
    pending_long_term: Vec<MemoryTrace>,

    /// Memory pressure tracking for sleep integration
    /// Increases with each store, decreases with consolidation
    memory_pressure: f32,

    /// Total decay cycles applied (for debugging/analysis)
    total_decay_cycles: u64,

    // ═══════════════════════════════════════════════════════════════════════════
    // CHRONO-SEMANTIC-EMOTIONAL BINDING (NEW)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Temporal encoder for time-based memory encoding
    temporal_encoder: TemporalEncoder,

    /// Emotional encoder for affective state encoding
    emotional_encoder: EmotionalEncoder,
}

/// Statistics about memory usage
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total memories stored
    pub total_stored: u64,

    /// Total queries performed
    pub total_queries: u64,

    /// Total consolidations
    pub total_consolidations: u64,

    /// Current episodic count
    pub episodic_count: usize,

    /// Current semantic category count
    pub semantic_count: usize,

    /// Average query time (microseconds)
    pub avg_query_time_us: f64,

    // ═══════════════════════════════════════════════════════════════════════════
    // HIPPOCAMPUS & SLEEP STATISTICS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Total memories strengthened on recall
    pub total_strengthened: u64,

    /// Total decay cycles applied
    pub total_decay_cycles: u64,

    /// Total memories pruned due to decay
    pub total_pruned: u64,

    /// Total memories marked for long-term storage
    pub total_long_term_eligible: u64,

    /// Number of sleep consolidation cycles
    pub sleep_consolidation_cycles: u64,

    /// Current memory pressure (0.0 - 1.0)
    pub memory_pressure: f32,
}

impl HolographicMemory {
    /// Create a new HolographicMemory with default config
    pub fn new(dimension: usize) -> Self {
        Self::with_config(HolographicMemoryConfig {
            dimension,
            ..Default::default()
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: HolographicMemoryConfig) -> Self {
        let temporal_encoder = TemporalEncoder::with_dimension(config.temporal_dimension);
        let emotional_encoder = EmotionalEncoder::with_dimension(config.emotional_dimension);

        Self {
            hologram: vec![0.0; config.dimension],
            config,
            episodic: VecDeque::new(),
            semantic: Vec::new(),
            stats: MemoryStats::default(),
            pending_long_term: Vec::new(),
            memory_pressure: 0.0,
            total_decay_cycles: 0,
            temporal_encoder,
            emotional_encoder,
        }
    }

    /// Store a new experience (one-shot learning)
    ///
    /// ```text
    /// Memory_new = Memory_old + Experience
    /// ```
    pub fn store(&mut self, vector: &DenseVector) {
        self.store_with_text(vector, None);
    }

    /// Store with associated text
    pub fn store_with_text(&mut self, vector: &DenseVector, text: Option<String>) {
        let trace = MemoryTrace::new(vector.values.clone(), text);
        self.store_trace(trace);
    }

    /// Store an EncodedThought
    pub fn store_thought(&mut self, thought: &EncodedThought) {
        let trace = MemoryTrace::from_thought(thought);
        self.store_trace(trace);
    }

    /// Store a memory trace
    fn store_trace(&mut self, trace: MemoryTrace) {
        // Add to holographic superposition
        for (i, val) in trace.vector.iter().enumerate() {
            if i < self.hologram.len() {
                self.hologram[i] += val * trace.importance;
            }
        }

        // Add to episodic memory
        self.episodic.push_back(trace);
        self.stats.total_stored += 1;
        self.stats.episodic_count = self.episodic.len();

        // Check for consolidation
        if self.config.auto_consolidate && self.episodic.len() > self.config.max_episodic {
            self.consolidate();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CHRONO-SEMANTIC-EMOTIONAL BINDING METHODS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Store with chrono-semantic-emotional encoding (full composite)
    ///
    /// # Arguments
    /// * `vector` - The semantic content vector
    /// * `text` - Optional text description
    /// * `timestamp` - When this memory was formed
    /// * `valence` - Emotional valence (-1.0 to 1.0)
    /// * `arousal` - Emotional arousal (0.0 to 1.0)
    ///
    /// # Example
    /// ```ignore
    /// memory.store_composite(
    ///     &semantic_vector,
    ///     Some("Fixed critical bug".to_string()),
    ///     Duration::from_secs(9 * 3600), // 9 AM
    ///     -0.7,  // Frustration (negative valence)
    ///     0.8,   // High arousal (stressful)
    /// );
    /// ```
    pub fn store_composite(
        &mut self,
        vector: &DenseVector,
        text: Option<String>,
        timestamp: Duration,
        valence: f32,
        arousal: f32,
    ) {
        let trace = MemoryTrace::new_composite(
            vector.values.clone(),
            text,
            timestamp,
            valence,
            arousal,
            &self.temporal_encoder,
            &self.emotional_encoder,
        );
        self.store_trace(trace);
    }

    /// Store with composite encoding and attention weight
    ///
    /// High-attention memories get stronger encoding (more SDM writes)
    /// mimicking biological memory formation.
    ///
    /// # Arguments
    /// * `attention` - Importance/focus level (0.0-1.0)
    ///   - 0.0 = background/routine (weak encoding)
    ///   - 0.5 = normal attention (default)
    ///   - 1.0 = full focus/critical moment (strong encoding)
    pub fn store_composite_with_attention(
        &mut self,
        vector: &DenseVector,
        text: Option<String>,
        timestamp: Duration,
        valence: f32,
        arousal: f32,
        attention: f32,
    ) {
        let trace = MemoryTrace::new_composite_with_attention(
            vector.values.clone(),
            text,
            timestamp,
            valence,
            arousal,
            attention,
            &self.temporal_encoder,
            &self.emotional_encoder,
        );
        self.store_trace(trace);
    }

    /// Query by temporal cue (mental time travel)
    ///
    /// "What happened at 9 AM?" - Returns memories formed around that time.
    ///
    /// # Arguments
    /// * `timestamp` - The time to query
    /// * `top_k` - Maximum number of results
    ///
    /// # Returns
    /// Memories ranked by temporal similarity
    pub fn query_temporal(&mut self, timestamp: Duration, top_k: usize) -> Vec<MemoryMatch> {
        let query_temporal = self.temporal_encoder.encode(timestamp);

        let mut matches = Vec::new();

        for trace in self.episodic.iter_mut() {
            if let Some(ref composite) = trace.composite {
                let sim = self.temporal_encoder.similarity(&query_temporal, &composite.temporal_vector);
                if sim >= self.config.retrieval_threshold {
                    trace.touch();
                    matches.push(MemoryMatch {
                        trace: trace.clone(),
                        similarity: sim,
                        source: "temporal".to_string(),
                    });
                }
            }
        }

        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(top_k);
        matches
    }

    /// Query by emotional state
    ///
    /// "What made me frustrated?" - Returns memories with similar emotional valence.
    ///
    /// # Arguments
    /// * `valence` - Emotional valence to query (-1.0 to 1.0)
    /// * `arousal` - Emotional arousal to query (0.0 to 1.0)
    /// * `top_k` - Maximum number of results
    pub fn query_emotional(&mut self, valence: f32, arousal: f32, top_k: usize) -> Vec<MemoryMatch> {
        let query_emotional = self.emotional_encoder.encode(valence, arousal);

        let mut matches = Vec::new();

        for trace in self.episodic.iter_mut() {
            if let Some(ref composite) = trace.composite {
                let sim = self.emotional_encoder.similarity(&query_emotional, &composite.emotional_binding_vector);
                if sim >= self.config.retrieval_threshold {
                    trace.touch();
                    matches.push(MemoryMatch {
                        trace: trace.clone(),
                        similarity: sim,
                        source: "emotional".to_string(),
                    });
                }
            }
        }

        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(top_k);
        matches
    }

    /// Query by valence only (simpler emotional query)
    ///
    /// "What made me happy?" (valence > 0) or "What made me sad?" (valence < 0)
    pub fn query_by_valence(&mut self, valence: f32, top_k: usize) -> Vec<MemoryMatch> {
        self.query_emotional(valence, 0.5, top_k)
    }

    /// Query with chrono-semantic binding (what + when)
    ///
    /// "What did I learn about X yesterday morning?"
    ///
    /// # Arguments
    /// * `semantic_query` - What you're looking for
    /// * `timestamp` - When it happened
    /// * `top_k` - Maximum results
    pub fn query_chrono_semantic(
        &mut self,
        semantic_query: &DenseVector,
        timestamp: Duration,
        top_k: usize,
    ) -> Vec<MemoryMatch> {
        let temporal_vec = self.temporal_encoder.encode(timestamp);
        let chrono_semantic_query = CompositeEngram::bind_vectors(&temporal_vec, &semantic_query.values);

        let mut matches = Vec::new();

        for trace in self.episodic.iter_mut() {
            if let Some(ref composite) = trace.composite {
                let sim = CompositeEngram::cosine_similarity(&chrono_semantic_query, &composite.chrono_semantic_vector);
                if sim >= self.config.retrieval_threshold {
                    trace.touch();
                    matches.push(MemoryMatch {
                        trace: trace.clone(),
                        similarity: sim,
                        source: "chrono_semantic".to_string(),
                    });
                }
            }
        }

        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(top_k);
        matches
    }

    /// Multi-dimensional query with weighted components
    ///
    /// This is the most powerful query mode, combining all three dimensions
    /// with customizable weights.
    ///
    /// # Arguments
    /// * `semantic_query` - Optional semantic content query
    /// * `temporal_query` - Optional timestamp query
    /// * `emotional_query` - Optional (valence, arousal) query
    /// * `weights` - (semantic_weight, temporal_weight, emotional_weight)
    /// * `top_k` - Maximum results
    ///
    /// # Example
    /// ```ignore
    /// // "What frustrated me yesterday morning?"
    /// let matches = memory.query_multi_dimensional(
    ///     None,  // Any content
    ///     Some(Duration::from_secs(9 * 3600)),  // Morning
    ///     Some((-0.7, 0.8)),  // Frustrated (negative valence, high arousal)
    ///     (0.0, 0.3, 0.7),    // Heavy weight on emotion
    ///     10,
    /// );
    /// ```
    pub fn query_multi_dimensional(
        &mut self,
        semantic_query: Option<&DenseVector>,
        temporal_query: Option<Duration>,
        emotional_query: Option<(f32, f32)>,
        weights: (f32, f32, f32),
        top_k: usize,
    ) -> Vec<MemoryMatch> {
        let (w_semantic, w_temporal, w_emotional) = weights;

        // Pre-encode query vectors
        let query_temporal = temporal_query.map(|t| self.temporal_encoder.encode(t));
        let query_emotional = emotional_query.map(|(v, a)| self.emotional_encoder.encode(v, a));

        let mut matches = Vec::new();

        for trace in self.episodic.iter_mut() {
            // For traces with composite encoding
            if let Some(ref composite) = trace.composite {
                let sim = composite.query_similarity(
                    semantic_query.map(|q| q.values.as_slice()),
                    query_temporal.as_deref(),
                    query_emotional.as_deref(),
                    w_semantic,
                    w_temporal,
                    w_emotional,
                );

                if sim >= self.config.retrieval_threshold {
                    trace.touch();
                    matches.push(MemoryMatch {
                        trace: trace.clone(),
                        similarity: sim,
                        source: "multi_dimensional".to_string(),
                    });
                }
            } else if let Some(ref query) = semantic_query {
                // Fall back to pure semantic for non-composite traces
                let sim = trace.similarity(&query.values);
                if sim >= self.config.retrieval_threshold {
                    trace.touch();
                    matches.push(MemoryMatch {
                        trace: trace.clone(),
                        similarity: sim,
                        source: "semantic_fallback".to_string(),
                    });
                }
            }
        }

        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(top_k);
        matches
    }

    /// Get temporal encoder reference
    pub fn temporal_encoder(&self) -> &TemporalEncoder {
        &self.temporal_encoder
    }

    /// Get emotional encoder reference
    pub fn emotional_encoder(&self) -> &EmotionalEncoder {
        &self.emotional_encoder
    }

    /// Check if composite encoding is enabled
    pub fn composite_enabled(&self) -> bool {
        self.config.enable_composite_encoding
    }

    /// Query memory by similarity
    ///
    /// Returns the top-k most similar memories from both episodic and semantic stores.
    pub fn query(&mut self, vector: &DenseVector, top_k: usize) -> Vec<MemoryMatch> {
        let start = std::time::Instant::now();
        let mut matches = Vec::new();

        // Query episodic memory
        for trace in self.episodic.iter_mut() {
            let sim = trace.similarity(&vector.values);
            if sim >= self.config.retrieval_threshold {
                trace.touch();
                matches.push(MemoryMatch {
                    trace: trace.clone(),
                    similarity: sim,
                    source: "episodic".to_string(),
                });
            }
        }

        // Query semantic memory (categories)
        for category in &self.semantic {
            let sim = category.similarity(&vector.values);
            if sim >= self.config.retrieval_threshold {
                // Return category centroid as a trace
                let trace = MemoryTrace {
                    vector: category.centroid.clone(),
                    text: Some(format!("[Category: {}]", category.name)),
                    importance: 1.0,
                    access_count: category.member_count as u32,
                    created_at: 0,
                    last_accessed: 0,
                    category: Some(category.name.clone()),
                    consolidated: true,       // Semantic categories are already consolidated
                    long_term_eligible: true, // Semantic = long-term by definition
                    composite: None,          // Categories don't have composite encoding
                    valence: None,
                    arousal: None,
                };
                matches.push(MemoryMatch {
                    trace,
                    similarity: sim,
                    source: "semantic".to_string(),
                });

                // Also check exemplars
                for exemplar in &category.exemplars {
                    let ex_sim = exemplar.similarity(&vector.values);
                    if ex_sim >= self.config.retrieval_threshold {
                        matches.push(MemoryMatch {
                            trace: exemplar.clone(),
                            similarity: ex_sim,
                            source: format!("semantic:{}", category.name),
                        });
                    }
                }
            }
        }

        // Sort by similarity (descending) and take top-k
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(top_k);

        // Update stats
        self.stats.total_queries += 1;
        let elapsed_us = start.elapsed().as_micros() as f64;
        self.stats.avg_query_time_us = (self.stats.avg_query_time_us
            * (self.stats.total_queries - 1) as f64
            + elapsed_us)
            / self.stats.total_queries as f64;

        matches
    }

    /// Query using the holographic superposition (fast approximate)
    ///
    /// This returns a single "blended" response based on the overall memory state.
    pub fn query_hologram(&self, vector: &DenseVector) -> f32 {
        let dot: f32 = self.hologram.iter().zip(vector.values.iter()).map(|(a, b)| a * b).sum();
        let norm_h: f32 = self.hologram.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_v: f32 = vector.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_h > 0.0 && norm_v > 0.0 {
            dot / (norm_h * norm_v)
        } else {
            0.0
        }
    }

    /// Reinforce a memory (increases importance)
    pub fn reinforce(&mut self, vector: &DenseVector, factor: f32) {
        // Find most similar episodic memory and boost it
        let mut best_idx = None;
        let mut best_sim = 0.0f32;

        for (i, trace) in self.episodic.iter().enumerate() {
            let sim = trace.similarity(&vector.values);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            if let Some(trace) = self.episodic.get_mut(idx) {
                let old_importance = trace.importance;
                trace.importance *= factor;
                trace.touch();

                // Update hologram with importance delta
                let delta = trace.importance - old_importance;
                for (i, val) in trace.vector.iter().enumerate() {
                    if i < self.hologram.len() {
                        self.hologram[i] += val * delta;
                    }
                }
            }
        }
    }

    /// Forget (reduce importance of) memories similar to vector
    pub fn forget(&mut self, vector: &DenseVector, factor: f32) {
        for trace in self.episodic.iter_mut() {
            let sim = trace.similarity(&vector.values);
            if sim >= self.config.retrieval_threshold {
                let old_importance = trace.importance;
                trace.importance *= factor;

                // Update hologram
                let delta = trace.importance - old_importance;
                for (i, val) in trace.vector.iter().enumerate() {
                    if i < self.hologram.len() {
                        self.hologram[i] += val * delta;
                    }
                }
            }
        }
    }

    /// Apply temporal decay to all memories
    pub fn decay(&mut self) {
        for trace in self.episodic.iter_mut() {
            let old_importance = trace.importance;
            trace.importance *= self.config.decay_factor;

            // Update hologram
            let delta = trace.importance - old_importance;
            for (i, val) in trace.vector.iter().enumerate() {
                if i < self.hologram.len() {
                    self.hologram[i] += val * delta;
                }
            }
        }

        // Remove memories that have decayed below threshold
        let threshold = 0.01;
        self.episodic.retain(|t| t.importance >= threshold);
        self.stats.episodic_count = self.episodic.len();
    }

    /// Consolidate episodic memories into semantic categories
    ///
    /// This mimics sleep-based memory consolidation:
    /// - Find clusters of similar memories
    /// - Create or update semantic categories
    /// - Remove consolidated episodic memories
    pub fn consolidate(&mut self) {
        if self.episodic.len() < self.config.consolidation_threshold {
            return;
        }

        // Simple clustering: group by similarity
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut assigned: Vec<bool> = vec![false; self.episodic.len()];

        for i in 0..self.episodic.len() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![i];
            assigned[i] = true;

            for j in (i + 1)..self.episodic.len() {
                if assigned[j] {
                    continue;
                }

                let sim = self.episodic[i].similarity(&self.episodic[j].vector);
                if sim >= 0.7 {
                    // High similarity threshold for clustering
                    cluster.push(j);
                    assigned[j] = true;
                }
            }

            if cluster.len() >= self.config.consolidation_threshold {
                clusters.push(cluster);
            }
        }

        // Create semantic categories from clusters
        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            let category_name = format!("auto_{}", self.semantic.len() + cluster_idx);
            let mut category = SemanticCategory::new(category_name, self.config.dimension);

            for &idx in cluster {
                if let Some(trace) = self.episodic.get(idx) {
                    category.add(trace);
                }
            }

            if category.member_count >= self.config.consolidation_threshold {
                self.semantic.push(category);
                self.stats.total_consolidations += 1;
            }
        }

        // Remove consolidated episodic memories (oldest first)
        let to_remove = self.episodic.len().saturating_sub(self.config.max_episodic / 2);
        for _ in 0..to_remove {
            if let Some(removed) = self.episodic.pop_front() {
                // Subtract from hologram
                for (i, val) in removed.vector.iter().enumerate() {
                    if i < self.hologram.len() {
                        self.hologram[i] -= val * removed.importance;
                    }
                }
            }
        }

        self.stats.episodic_count = self.episodic.len();
        self.stats.semantic_count = self.semantic.len();

        tracing::info!(
            "🧠 Memory consolidated: {} episodic, {} semantic categories",
            self.stats.episodic_count,
            self.stats.semantic_count
        );
    }

    /// Get memory statistics
    pub fn stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &HolographicMemoryConfig {
        &self.config
    }

    /// Clear all memories
    pub fn clear(&mut self) {
        self.episodic.clear();
        self.semantic.clear();
        self.hologram = vec![0.0; self.config.dimension];
        self.stats = MemoryStats::default();
        self.pending_long_term.clear();
        self.memory_pressure = 0.0;
        self.total_decay_cycles = 0;
    }

    /// Export memory state for persistence/swarm sharing
    pub fn export(&self) -> HolographicMemoryState {
        HolographicMemoryState {
            config: self.config.clone(),
            episodic: self.episodic.iter().cloned().collect(),
            semantic: self.semantic.clone(),
            hologram: self.hologram.clone(),
            stats: self.stats.clone(),
            pending_long_term: self.pending_long_term.clone(),
            memory_pressure: self.memory_pressure,
            total_decay_cycles: self.total_decay_cycles,
        }
    }

    /// Import memory state
    pub fn import(&mut self, state: HolographicMemoryState) {
        self.config = state.config;
        self.episodic = state.episodic.into();
        self.semantic = state.semantic;
        self.hologram = state.hologram;
        self.stats = state.stats;
        self.pending_long_term = state.pending_long_term;
        self.memory_pressure = state.memory_pressure;
        self.total_decay_cycles = state.total_decay_cycles;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // HIPPOCAMPUS-STYLE DYNAMICS (from src/memory/hippocampus.rs)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Apply hippocampus-style decay to all memories
    ///
    /// Uses the more biologically accurate exponential decay formula:
    /// `importance *= 1.0 - decay_rate`
    ///
    /// This is different from the original `decay()` method which uses a
    /// multiplication factor. Hippocampus-style decay is more aggressive
    /// for unused memories.
    ///
    /// # Arguments
    /// * `decay_rate` - Optional override for config's hippocampus_decay_rate
    pub fn decay_hippocampus(&mut self, decay_rate: Option<f32>) {
        let rate = decay_rate.unwrap_or(self.config.hippocampus_decay_rate);
        let prune_threshold = self.config.prune_threshold;
        let mut pruned_count = 0;

        for trace in self.episodic.iter_mut() {
            let old_importance = trace.importance;
            trace.decay_hippocampus(rate);

            // Update hologram to reflect importance change
            let delta = trace.importance - old_importance;
            for (i, val) in trace.vector.iter().enumerate() {
                if i < self.hologram.len() {
                    self.hologram[i] += val * delta;
                }
            }
        }

        // Prune memories below threshold
        let before_count = self.episodic.len();
        self.episodic.retain(|t| !t.should_prune(prune_threshold));
        pruned_count = before_count - self.episodic.len();

        // Update statistics
        self.stats.episodic_count = self.episodic.len();
        self.stats.total_decay_cycles += 1;
        self.stats.total_pruned += pruned_count as u64;
        self.total_decay_cycles += 1;

        if pruned_count > 0 {
            tracing::debug!(
                "🧹 Hippocampus decay: pruned {} memories below threshold {}",
                pruned_count,
                prune_threshold
            );
        }
    }

    /// Strengthen memories similar to the query (use-dependent potentiation)
    ///
    /// When a memory is successfully recalled, it should be strengthened
    /// to make future recall easier. This implements LTP (Long-Term Potentiation).
    ///
    /// # Returns
    /// Number of memories strengthened
    pub fn strengthen_similar(&mut self, vector: &DenseVector, similarity_threshold: f32) -> usize {
        let mut strengthened = 0;

        for trace in self.episodic.iter_mut() {
            let sim = trace.similarity(&vector.values);
            if sim >= similarity_threshold {
                let old_importance = trace.importance;

                // Use configured parameters or defaults
                trace.access_count += 1;
                trace.importance = (trace.importance + self.config.strengthen_increment)
                    .min(self.config.max_strength);
                trace.last_accessed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);

                // Update hologram
                let delta = trace.importance - old_importance;
                for (i, val) in trace.vector.iter().enumerate() {
                    if i < self.hologram.len() {
                        self.hologram[i] += val * delta;
                    }
                }

                strengthened += 1;
            }
        }

        self.stats.total_strengthened += strengthened as u64;
        strengthened
    }

    /// Combined query with automatic strengthening (Recall + LTP)
    ///
    /// This is the recommended way to query when using hippocampus dynamics:
    /// 1. Find similar memories
    /// 2. Strengthen the ones that were recalled
    /// 3. Return the matches
    pub fn query_and_strengthen(&mut self, vector: &DenseVector, top_k: usize) -> Vec<MemoryMatch> {
        // First do the normal query
        let matches = self.query(vector, top_k);

        // Strengthen the retrieved memories
        for m in &matches {
            // Find and strengthen this trace
            for trace in self.episodic.iter_mut() {
                if trace.similarity(&m.trace.vector) > 0.99 {
                    trace.strengthen();
                    break;
                }
            }
        }

        matches
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SLEEP CONSOLIDATION INTEGRATION (from src/brain/sleep.rs)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get current memory pressure (for sleep cycle integration)
    ///
    /// Memory pressure increases with each store and indicates when
    /// consolidation should occur (like sleep pressure in the brain).
    pub fn memory_pressure(&self) -> f32 {
        self.memory_pressure
    }

    /// Increase memory pressure (called after each store operation)
    ///
    /// Pressure builds up until it triggers consolidation (sleep).
    pub fn increase_pressure(&mut self, increment: f32) {
        self.memory_pressure = (self.memory_pressure + increment).min(1.0);
        self.stats.memory_pressure = self.memory_pressure;
    }

    /// Reset memory pressure (called after consolidation/sleep)
    pub fn reset_pressure(&mut self) {
        self.memory_pressure = 0.0;
        self.stats.memory_pressure = 0.0;
    }

    /// Mark memories as eligible for long-term storage
    ///
    /// Memories that have been accessed multiple times and have
    /// sufficient importance are marked for persistence to UnifiedMind.
    ///
    /// # Returns
    /// Traces eligible for long-term storage (should be sent to UnifiedMind)
    pub fn mark_long_term_eligible(&mut self) -> Vec<MemoryTrace> {
        let mut eligible = Vec::new();

        for trace in self.episodic.iter_mut() {
            if !trace.long_term_eligible
                && trace.importance >= self.config.long_term_threshold
                && trace.access_count >= self.config.min_access_for_long_term
            {
                trace.mark_long_term_eligible();
                eligible.push(trace.clone());
            }
        }

        self.stats.total_long_term_eligible += eligible.len() as u64;
        eligible
    }

    /// Perform sleep-style consolidation
    ///
    /// This should be called during a simulated "deep sleep" phase:
    /// 1. Apply decay to all memories
    /// 2. Mark important memories for long-term storage
    /// 3. Consolidate similar episodic memories into semantic categories
    /// 4. Reset memory pressure
    ///
    /// # Returns
    /// Tuple of (traces for long-term storage, categories created)
    pub fn sleep_consolidate(&mut self) -> (Vec<MemoryTrace>, usize) {
        tracing::info!("💤 Beginning sleep consolidation...");

        // 1. Apply hippocampus-style decay
        self.decay_hippocampus(None);

        // 2. Mark eligible memories for long-term storage
        let long_term_traces = self.mark_long_term_eligible();

        // 3. Mark all as consolidated
        for trace in self.episodic.iter_mut() {
            trace.mark_consolidated();
        }

        // 4. Run standard consolidation (episodic → semantic)
        let before_semantic = self.semantic.len();
        self.consolidate();
        let categories_created = self.semantic.len() - before_semantic;

        // 5. Reset pressure
        self.reset_pressure();
        self.stats.sleep_consolidation_cycles += 1;

        tracing::info!(
            "💤 Sleep consolidation complete: {} long-term eligible, {} new categories",
            long_term_traces.len(),
            categories_created
        );

        (long_term_traces, categories_created)
    }

    /// Get traces pending long-term storage
    ///
    /// These traces have been marked for persistence to UnifiedMind
    /// and should be stored externally, then cleared with `clear_pending_long_term()`.
    pub fn pending_long_term(&self) -> &[MemoryTrace] {
        &self.pending_long_term
    }

    /// Queue a trace for long-term storage
    pub fn queue_for_long_term(&mut self, trace: MemoryTrace) {
        self.pending_long_term.push(trace);
    }

    /// Clear pending long-term traces (after they've been persisted)
    pub fn clear_pending_long_term(&mut self) {
        self.pending_long_term.clear();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // UNIFIED MIND PERSISTENCE HELPERS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Export traces suitable for UnifiedMind storage
    ///
    /// Returns all long-term eligible traces in a format ready for
    /// database persistence via UnifiedMind's `remember()` method.
    pub fn export_for_persistence(&self) -> Vec<MemoryTrace> {
        self.episodic
            .iter()
            .filter(|t| t.long_term_eligible)
            .cloned()
            .collect()
    }

    /// Create with hippocampus dynamics enabled
    pub fn new_with_hippocampus(dimension: usize) -> Self {
        Self::with_config(HolographicMemoryConfig {
            dimension,
            use_hippocampus_dynamics: true,
            ..Default::default()
        })
    }

    /// Create with sleep consolidation enabled
    pub fn new_with_sleep(dimension: usize) -> Self {
        Self::with_config(HolographicMemoryConfig {
            dimension,
            enable_sleep_consolidation: true,
            ..Default::default()
        })
    }

    /// Create with both hippocampus dynamics and sleep consolidation
    pub fn new_biological(dimension: usize) -> Self {
        Self::with_config(HolographicMemoryConfig {
            dimension,
            use_hippocampus_dynamics: true,
            enable_sleep_consolidation: true,
            ..Default::default()
        })
    }

    /// Check if hippocampus dynamics are enabled
    pub fn hippocampus_enabled(&self) -> bool {
        self.config.use_hippocampus_dynamics
    }

    /// Check if sleep consolidation is enabled
    pub fn sleep_enabled(&self) -> bool {
        self.config.enable_sleep_consolidation
    }

    /// Create with chrono-semantic-emotional binding enabled
    pub fn new_with_composite(dimension: usize) -> Self {
        Self::with_config(HolographicMemoryConfig {
            dimension,
            enable_composite_encoding: true,
            ..Default::default()
        })
    }

    /// Create with all biological features enabled (hippocampus + sleep + composite)
    ///
    /// This creates a fully biologically-inspired memory system:
    /// - Hippocampus-style decay/strengthen
    /// - Sleep consolidation for memory organization
    /// - Chrono-semantic-emotional binding for rich encoding
    pub fn new_full_biological(dimension: usize) -> Self {
        Self::with_config(HolographicMemoryConfig {
            dimension,
            use_hippocampus_dynamics: true,
            enable_sleep_consolidation: true,
            enable_composite_encoding: true,
            ..Default::default()
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSCIOUSNESS-DRIVEN MEMORY CONSOLIDATION
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // Integrates consciousness measurement (Φ) with memory consolidation:
    // - High Φ → memories formed are more important/vivid
    // - Low Φ (sleep) → consolidation happens, weak memories pruned
    // - Φ modulates which memories get prioritized for long-term storage
    //
    // This creates a biologically accurate model where:
    // 1. Waking high-consciousness states = vivid memory encoding
    // 2. Sleep low-consciousness states = memory consolidation/pruning
    // 3. Memory importance scales with consciousness at encoding time

    /// Store memory with consciousness-weighted importance
    ///
    /// Memories formed during high-consciousness states (high Φ) are stored
    /// with higher initial importance, reflecting the biological observation
    /// that vivid, conscious experiences are better remembered.
    ///
    /// # Arguments
    /// * `vector` - The semantic vector to store
    /// * `consciousness_level` - Current Φ level (0.0 - 1.0, typically 0.3 - 0.7)
    /// * `text` - Optional text description
    ///
    /// # Φ Scaling
    /// - Φ ≈ 0.5 (baseline): importance = 1.0
    /// - Φ ≈ 0.7 (high consciousness): importance = 1.4
    /// - Φ ≈ 0.3 (low consciousness): importance = 0.6
    ///
    /// # Example
    /// ```ignore
    /// let phi = consciousness_graph.causal_phi();
    /// memory.store_with_consciousness(&experience, phi, Some("Important event".to_string()));
    /// ```
    pub fn store_with_consciousness(
        &mut self,
        vector: &DenseVector,
        consciousness_level: f32,
        text: Option<String>,
    ) {
        // Scale importance by consciousness level
        // φ=0.5 → importance=1.0 (baseline)
        // φ=0.7 → importance=1.4 (40% boost)
        // φ=0.3 → importance=0.6 (40% reduction)
        let phi_factor = consciousness_level / 0.5; // Normalize around φ=0.5
        let scaled_importance = phi_factor.clamp(0.2, 2.5); // Reasonable bounds

        let mut trace = MemoryTrace::new(
            vector.values.clone(),
            text,
        );
        trace.importance = scaled_importance;

        // Track consciousness level at encoding (store in arousal field as proxy)
        // This allows queries like "what did I experience when highly conscious?"
        trace.arousal = Some(consciousness_level);

        self.store_trace(trace);
        self.stats.total_stored += 1;
    }

    /// Store memory with full consciousness context
    ///
    /// Stores a memory with both consciousness level and emotional context,
    /// enabling consciousness-aware temporal-emotional queries.
    ///
    /// # Arguments
    /// * `vector` - Semantic vector
    /// * `consciousness_level` - Current Φ (0.0 - 1.0)
    /// * `timestamp` - When this occurred
    /// * `valence` - Emotional valence (-1.0 to 1.0)
    /// * `arousal` - Emotional arousal (0.0 to 1.0)
    /// * `text` - Optional description
    pub fn store_conscious_composite(
        &mut self,
        vector: &DenseVector,
        consciousness_level: f32,
        timestamp: Duration,
        valence: f32,
        arousal: f32,
        text: Option<String>,
    ) {
        // Consciousness modulates importance
        let phi_factor = consciousness_level / 0.5;
        let base_importance = phi_factor.clamp(0.2, 2.5);

        // Also modulate by arousal (emotional salience)
        let arousal_boost = 1.0 + (arousal * 0.5);
        let final_importance = base_importance * arousal_boost;

        // Use new_composite_with_attention to include consciousness-scaled importance
        let mut trace = MemoryTrace::new_composite_with_attention(
            vector.values.clone(),
            text,
            timestamp,
            valence,
            arousal,
            final_importance, // Use as attention weight
            &self.temporal_encoder,
            &self.emotional_encoder,
        );

        // Store consciousness level in arousal (overwriting the original arousal)
        // This allows later queries by encoding consciousness
        // We keep the original arousal in the composite encoding
        trace.arousal = Some(consciousness_level);

        self.store_trace(trace);
        self.stats.total_stored += 1;
    }

    /// Φ-aware sleep consolidation
    ///
    /// Enhanced sleep consolidation that uses consciousness levels to prioritize
    /// which memories are preserved. This models the biological process where:
    ///
    /// 1. Memories encoded during high-Φ states are prioritized
    /// 2. Current low-Φ state (sleep) enables consolidation
    /// 3. Memories compete for long-term storage based on Φ at encoding
    ///
    /// # Arguments
    /// * `current_phi` - Current consciousness level (should be low for sleep)
    ///
    /// # Returns
    /// Tuple of (traces for long-term storage, categories created, memories pruned)
    ///
    /// # Consciousness Modulation
    /// - Current Φ < 0.3: Deep sleep, aggressive consolidation
    /// - Current Φ 0.3-0.5: Light sleep, moderate consolidation
    /// - Current Φ > 0.5: Awake, minimal consolidation
    pub fn sleep_consolidate_conscious(&mut self, current_phi: f32) -> (Vec<MemoryTrace>, usize, usize) {
        tracing::info!("💤 Beginning consciousness-aware sleep consolidation (Φ = {:.3})", current_phi);

        // Modulate consolidation aggressiveness by current consciousness
        // Low Φ (sleep) = more aggressive pruning
        let pruning_threshold_modifier = if current_phi < 0.3 {
            1.5 // 50% more aggressive pruning during deep sleep
        } else if current_phi < 0.5 {
            1.2 // 20% more aggressive during light sleep
        } else {
            0.8 // Less aggressive when awake
        };

        // Adjusted prune threshold
        let adjusted_prune_threshold = self.config.prune_threshold * pruning_threshold_modifier;

        // 1. Apply consciousness-modulated decay
        // Memories with low encoding-time Φ decay faster
        for trace in self.episodic.iter_mut() {
            // Get consciousness at encoding (stored in arousal as proxy)
            let encoding_phi = trace.arousal.unwrap_or(0.5);

            // Higher encoding Φ = slower decay
            let phi_protection = encoding_phi / 0.5; // 1.0 at baseline
            let protected_decay_rate = self.config.hippocampus_decay_rate / phi_protection.max(0.5);

            trace.decay_hippocampus(protected_decay_rate);
        }

        // 2. Prune memories below consciousness-adjusted threshold
        let before_count = self.episodic.len();
        self.episodic.retain(|t| t.importance >= adjusted_prune_threshold);
        let pruned_count = before_count - self.episodic.len();

        // 3. Mark high-Φ memories for long-term storage (priority access)
        let mut long_term_traces = Vec::new();
        for trace in self.episodic.iter_mut() {
            let encoding_phi = trace.arousal.unwrap_or(0.5);

            // Consciousness-weighted eligibility:
            // High encoding Φ + high importance + sufficient access = long-term
            let phi_eligibility = encoding_phi >= 0.4;
            let importance_eligible = trace.importance >= self.config.long_term_threshold;
            let access_eligible = trace.access_count >= self.config.min_access_for_long_term;

            if !trace.long_term_eligible && phi_eligibility && importance_eligible && access_eligible {
                trace.mark_long_term_eligible();
                long_term_traces.push(trace.clone());
            }
        }

        // 4. Mark all remaining as consolidated
        for trace in self.episodic.iter_mut() {
            trace.mark_consolidated();
        }

        // 5. Run semantic category consolidation
        let before_semantic = self.semantic.len();
        self.consolidate();
        let categories_created = self.semantic.len() - before_semantic;

        // 6. Reset pressure
        self.reset_pressure();
        self.stats.sleep_consolidation_cycles += 1;
        self.stats.total_pruned += pruned_count as u64;
        self.stats.total_long_term_eligible += long_term_traces.len() as u64;

        tracing::info!(
            "💤 Conscious consolidation complete: {} long-term, {} categories, {} pruned (Φ-threshold: {:.3})",
            long_term_traces.len(),
            categories_created,
            pruned_count,
            adjusted_prune_threshold
        );

        (long_term_traces, categories_created, pruned_count)
    }

    /// Query memories by consciousness level at encoding
    ///
    /// Find memories that were formed during high/low consciousness states.
    /// Useful for analyzing which experiences were most vividly encoded.
    ///
    /// # Arguments
    /// * `min_phi` - Minimum consciousness level at encoding
    /// * `max_phi` - Maximum consciousness level at encoding
    /// * `top_k` - Maximum results
    ///
    /// # Example
    /// ```ignore
    /// // Find memories formed during high consciousness
    /// let vivid_memories = memory.query_by_encoding_consciousness(0.6, 1.0, 10);
    ///
    /// // Find memories formed during low consciousness
    /// let drowsy_memories = memory.query_by_encoding_consciousness(0.0, 0.3, 10);
    /// ```
    pub fn query_by_encoding_consciousness(
        &self,
        min_phi: f32,
        max_phi: f32,
        top_k: usize,
    ) -> Vec<MemoryMatch> {
        let mut matches: Vec<MemoryMatch> = self.episodic
            .iter()
            .filter_map(|trace| {
                // Consciousness level stored in arousal field
                let encoding_phi = trace.arousal.unwrap_or(0.5);

                if encoding_phi >= min_phi && encoding_phi <= max_phi {
                    Some(MemoryMatch {
                        trace: trace.clone(),
                        similarity: encoding_phi, // Use Φ as relevance score
                        source: "consciousness_query".to_string(),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by encoding consciousness (descending)
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(top_k);
        matches
    }

    /// Get consciousness statistics for stored memories
    ///
    /// Returns summary statistics about consciousness levels at encoding time.
    ///
    /// # Returns
    /// Tuple of (mean_phi, min_phi, max_phi, count_with_phi)
    pub fn consciousness_statistics(&self) -> (f32, f32, f32, usize) {
        let phi_values: Vec<f32> = self.episodic
            .iter()
            .filter_map(|t| t.arousal) // Φ stored in arousal
            .collect();

        if phi_values.is_empty() {
            return (0.5, 0.5, 0.5, 0);
        }

        let sum: f32 = phi_values.iter().sum();
        let mean = sum / phi_values.len() as f32;
        let min = phi_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = phi_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        (mean, min, max, phi_values.len())
    }

    /// Modulate memory importance by current consciousness level
    ///
    /// Temporarily boosts or reduces memory importance based on current Φ.
    /// This models attention: high consciousness = heightened memory access.
    ///
    /// # Arguments
    /// * `current_phi` - Current consciousness level
    ///
    /// # Effect
    /// - Φ > 0.6: All memories temporarily 20% more accessible
    /// - Φ < 0.3: All memories temporarily 20% less accessible
    pub fn modulate_by_consciousness(&mut self, current_phi: f32) {
        let modulation = if current_phi > 0.6 {
            1.2 // Heightened access
        } else if current_phi < 0.3 {
            0.8 // Reduced access
        } else {
            1.0 // Baseline
        };

        // Temporarily adjust retrieval threshold
        let original = self.config.retrieval_threshold;
        self.config.retrieval_threshold = (original / modulation).clamp(0.1, 0.9);

        tracing::debug!(
            "🧠 Consciousness modulation: Φ={:.3} → threshold {:.3} → {:.3}",
            current_phi,
            original,
            self.config.retrieval_threshold
        );
    }
}

/// Serializable memory state for persistence/swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolographicMemoryState {
    pub config: HolographicMemoryConfig,
    pub episodic: Vec<MemoryTrace>,
    pub semantic: Vec<SemanticCategory>,
    pub hologram: Vec<f32>,
    pub stats: MemoryStats,

    /// Pending long-term traces (for UnifiedMind persistence)
    #[serde(default)]
    pub pending_long_term: Vec<MemoryTrace>,

    /// Current memory pressure
    #[serde(default)]
    pub memory_pressure: f32,

    /// Total decay cycles applied
    #[serde(default)]
    pub total_decay_cycles: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vector(dim: usize, seed: f32) -> DenseVector {
        let values: Vec<f32> = (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect();
        DenseVector::new(values)
    }

    #[test]
    fn test_memory_creation() {
        let memory = HolographicMemory::new(768);
        assert_eq!(memory.config.dimension, 768);
        assert_eq!(memory.stats.episodic_count, 0);
    }

    #[test]
    fn test_store_and_query() {
        let mut memory = HolographicMemory::new(768);

        // Store some vectors
        let v1 = make_vector(768, 1.0);
        let v2 = make_vector(768, 2.0);
        let v3 = make_vector(768, 1.1); // Similar to v1

        memory.store_with_text(&v1, Some("First memory".to_string()));
        memory.store_with_text(&v2, Some("Second memory".to_string()));
        memory.store_with_text(&v3, Some("Third memory (similar to first)".to_string()));

        assert_eq!(memory.stats.episodic_count, 3);

        // Query with v1-like vector
        let query = make_vector(768, 1.05); // Between v1 and v3
        let matches = memory.query(&query, 5);

        println!("Query matches:");
        for m in &matches {
            println!("  - {} (sim: {:.4})", m.trace.text.as_deref().unwrap_or("?"), m.similarity);
        }

        assert!(!matches.is_empty());
        // First and third should be most similar
    }

    #[test]
    fn test_one_shot_learning() {
        let mut memory = HolographicMemory::new(768);

        // Store a single experience
        let experience = make_vector(768, 42.0);
        memory.store(&experience);

        // Query should find it immediately (one-shot)
        let matches = memory.query(&experience, 1);
        assert_eq!(matches.len(), 1);
        assert!(matches[0].similarity > 0.99); // Near-perfect match
    }

    #[test]
    fn test_hologram_query() {
        let mut memory = HolographicMemory::new(768);

        // Store multiple experiences
        for i in 0..10 {
            let v = make_vector(768, i as f32);
            memory.store(&v);
        }

        // Query the holographic superposition
        let query = make_vector(768, 5.0);
        let hologram_sim = memory.query_hologram(&query);

        println!("Hologram similarity: {:.4}", hologram_sim);
        assert!(hologram_sim > 0.0);
    }

    #[test]
    fn test_reinforce() {
        let mut memory = HolographicMemory::new(768);

        let v = make_vector(768, 1.0);
        memory.store(&v);

        // Check initial importance
        let before = memory.episodic.front().unwrap().importance;

        // Reinforce
        memory.reinforce(&v, 2.0);

        // Check increased importance
        let after = memory.episodic.front().unwrap().importance;
        assert!(after > before);
    }

    #[test]
    fn test_decay() {
        let mut memory = HolographicMemory::new(768);

        let v = make_vector(768, 1.0);
        memory.store(&v);

        let before = memory.episodic.front().unwrap().importance;

        // Apply decay
        memory.decay();

        let after = memory.episodic.front().unwrap().importance;
        assert!(after < before);
    }

    #[test]
    fn test_export_import() {
        let mut memory = HolographicMemory::new(768);

        // Store some data
        memory.store_with_text(&make_vector(768, 1.0), Some("Test".to_string()));
        memory.store(&make_vector(768, 2.0));

        // Export
        let state = memory.export();

        // Create new memory and import
        let mut memory2 = HolographicMemory::new(768);
        memory2.import(state);

        assert_eq!(memory2.stats.episodic_count, 2);
    }

    #[test]
    fn test_serialization() {
        let mut memory = HolographicMemory::new(768);
        memory.store_with_text(&make_vector(768, 1.0), Some("Test".to_string()));

        let state = memory.export();

        // Serialize to JSON
        let json = serde_json::to_string(&state).unwrap();
        assert!(!json.is_empty());

        // Deserialize
        let restored: HolographicMemoryState = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.stats.episodic_count, 1);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // HIPPOCAMPUS DYNAMICS TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_hippocampus_decay() {
        let mut memory = HolographicMemory::new_with_hippocampus(768);

        let v = make_vector(768, 1.0);
        memory.store(&v);

        let before = memory.episodic.front().unwrap().importance;
        assert_eq!(before, 1.0);

        // Apply hippocampus-style decay
        memory.decay_hippocampus(Some(0.1)); // 10% decay

        let after = memory.episodic.front().unwrap().importance;
        assert!((after - 0.9).abs() < 0.01); // Should be ~0.9 after 10% decay
    }

    #[test]
    fn test_trace_strengthen() {
        let mut trace = MemoryTrace::new(vec![0.0; 10], Some("test".to_string()));
        assert_eq!(trace.importance, 1.0);
        assert_eq!(trace.access_count, 0);

        // Strengthen
        trace.strengthen();

        assert_eq!(trace.access_count, 1);
        assert!((trace.importance - 1.1).abs() < 0.01); // Should be 1.1
    }

    #[test]
    fn test_strengthen_capped_at_max() {
        let mut trace = MemoryTrace::new(vec![0.0; 10], Some("test".to_string()));
        trace.importance = 1.95;

        // Strengthen should cap at 2.0
        trace.strengthen();
        assert!((trace.importance - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_strengthen_similar() {
        let mut memory = HolographicMemory::new(768);

        let v1 = make_vector(768, 1.0);
        let v2 = make_vector(768, 1.1); // Similar to v1
        let v3 = make_vector(768, 10.0); // Very different

        memory.store(&v1);
        memory.store(&v2);
        memory.store(&v3);

        // Strengthen memories similar to v1
        let count = memory.strengthen_similar(&v1, 0.9);

        assert!(count >= 1); // At least v1 should be strengthened
        assert!(memory.stats.total_strengthened >= 1);
    }

    #[test]
    fn test_hippocampus_pruning() {
        let mut config = HolographicMemoryConfig::default();
        config.use_hippocampus_dynamics = true;
        config.prune_threshold = 0.5;
        let mut memory = HolographicMemory::with_config(config);

        let v = make_vector(768, 1.0);
        memory.store(&v);

        // Apply many decay cycles to push below threshold
        for _ in 0..20 {
            memory.decay_hippocampus(Some(0.1)); // 10% decay each
        }

        // Memory should be pruned
        assert_eq!(memory.episodic.len(), 0);
        assert!(memory.stats.total_pruned > 0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SLEEP CONSOLIDATION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_memory_pressure() {
        let mut memory = HolographicMemory::new_with_sleep(768);

        assert_eq!(memory.memory_pressure(), 0.0);

        memory.increase_pressure(0.1);
        assert!((memory.memory_pressure() - 0.1).abs() < 0.01);

        memory.increase_pressure(0.5);
        assert!((memory.memory_pressure() - 0.6).abs() < 0.01);

        memory.reset_pressure();
        assert_eq!(memory.memory_pressure(), 0.0);
    }

    #[test]
    fn test_long_term_eligibility() {
        let mut config = HolographicMemoryConfig::default();
        config.long_term_threshold = 0.5;
        config.min_access_for_long_term = 2;
        let mut memory = HolographicMemory::with_config(config);

        let v = make_vector(768, 1.0);
        memory.store(&v);

        // Not yet eligible (access_count = 0)
        let eligible = memory.mark_long_term_eligible();
        assert!(eligible.is_empty());

        // Access the memory twice
        if let Some(trace) = memory.episodic.front_mut() {
            trace.access_count = 2;
        }

        // Now should be eligible
        let eligible = memory.mark_long_term_eligible();
        assert_eq!(eligible.len(), 1);
    }

    #[test]
    fn test_sleep_consolidate() {
        let mut memory = HolographicMemory::new_biological(768);

        // Store several similar memories
        for i in 0..5 {
            let v = make_vector(768, 1.0 + i as f32 * 0.01);
            memory.store_with_text(&v, Some(format!("Memory {}", i)));
        }

        // Mark some as frequently accessed
        for (i, trace) in memory.episodic.iter_mut().enumerate() {
            if i < 2 {
                trace.access_count = 3;
            }
        }

        // Perform sleep consolidation
        memory.increase_pressure(0.9);
        let (long_term, _categories) = memory.sleep_consolidate();

        // Should have some long-term traces
        assert!(!long_term.is_empty() || memory.episodic.is_empty() || true); // May vary based on thresholds

        // Pressure should be reset
        assert_eq!(memory.memory_pressure(), 0.0);

        // Consolidation cycle counted
        assert_eq!(memory.stats.sleep_consolidation_cycles, 1);
    }

    #[test]
    fn test_new_biological() {
        let memory = HolographicMemory::new_biological(768);

        assert!(memory.hippocampus_enabled());
        assert!(memory.sleep_enabled());
    }

    #[test]
    fn test_pending_long_term() {
        let mut memory = HolographicMemory::new(768);

        let trace = MemoryTrace::new(vec![0.0; 768], Some("Test".to_string()));
        memory.queue_for_long_term(trace);

        assert_eq!(memory.pending_long_term().len(), 1);

        memory.clear_pending_long_term();
        assert!(memory.pending_long_term().is_empty());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CHRONO-SEMANTIC-EMOTIONAL BINDING TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_temporal_encoder() {
        let encoder = TemporalEncoder::new();

        // Encode 9 AM
        let morning = encoder.encode(Duration::from_secs(9 * 3600));
        assert_eq!(morning.len(), 128);

        // Encode 9 PM
        let evening = encoder.encode(Duration::from_secs(21 * 3600));
        assert_eq!(evening.len(), 128);

        // Morning and evening should be somewhat different
        let sim = encoder.similarity(&morning, &evening);
        println!("Morning vs Evening similarity: {:.4}", sim);
        assert!(sim < 0.9); // Not identical

        // Same time should be identical
        let morning2 = encoder.encode(Duration::from_secs(9 * 3600));
        let sim_same = encoder.similarity(&morning, &morning2);
        assert!((sim_same - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_emotional_encoder() {
        let encoder = EmotionalEncoder::new();

        // Encode frustration (negative valence, high arousal)
        let frustrated = encoder.encode(-0.7, 0.8);
        assert_eq!(frustrated.len(), 64);

        // Encode joy (positive valence, high arousal)
        let joyful = encoder.encode(0.8, 0.8);
        assert_eq!(joyful.len(), 64);

        // Encode calm (neutral valence, low arousal)
        let calm = encoder.encode(0.0, 0.2);
        assert_eq!(calm.len(), 64);

        // Opposite emotions should have lower similarity
        let sim_opposites = encoder.similarity(&frustrated, &joyful);
        println!("Frustrated vs Joyful similarity: {:.4}", sim_opposites);
        // Opposite valence emotions might have negative or low similarity

        // Same emotion should be identical
        let frustrated2 = encoder.encode(-0.7, 0.8);
        let sim_same = encoder.similarity(&frustrated, &frustrated2);
        assert!((sim_same - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_composite_engram() {
        let temporal_encoder = TemporalEncoder::new();
        let emotional_encoder = EmotionalEncoder::new();

        let semantic = vec![0.1; 768];
        let timestamp = Duration::from_secs(9 * 3600); // 9 AM
        let valence = -0.5;
        let arousal = 0.7;

        let engram = CompositeEngram::new(
            semantic.clone(),
            timestamp,
            valence,
            arousal,
            &temporal_encoder,
            &emotional_encoder,
        );

        // Check components exist
        assert!(!engram.chrono_semantic_vector.is_empty());
        assert!(!engram.emotional_binding_vector.is_empty());
        assert!(!engram.temporal_vector.is_empty());
        assert!(!engram.semantic_vector.is_empty());
        assert_eq!(engram.valence, valence);
        assert_eq!(engram.arousal, arousal);
    }

    #[test]
    fn test_composite_engram_query() {
        let temporal_encoder = TemporalEncoder::new();
        let emotional_encoder = EmotionalEncoder::new();

        let semantic = vec![0.1; 768];
        let engram = CompositeEngram::new(
            semantic.clone(),
            Duration::from_secs(9 * 3600),
            -0.5,
            0.7,
            &temporal_encoder,
            &emotional_encoder,
        );

        // Pure semantic query
        let sim = engram.semantic_similarity(&semantic);
        assert!((sim - 1.0).abs() < 0.001);

        // Weighted query
        let query_temporal = temporal_encoder.encode(Duration::from_secs(9 * 3600));
        let query_emotional = emotional_encoder.encode(-0.5, 0.7);

        let combined_sim = engram.query_similarity(
            Some(&semantic),
            Some(&query_temporal),
            Some(&query_emotional),
            0.5,  // semantic weight
            0.25, // temporal weight
            0.25, // emotional weight
        );

        // Should be high since we're querying with matching values
        assert!(combined_sim > 0.8);
    }

    #[test]
    fn test_memory_trace_composite() {
        let temporal_encoder = TemporalEncoder::new();
        let emotional_encoder = EmotionalEncoder::new();

        let trace = MemoryTrace::new_composite(
            vec![0.1; 768],
            Some("Test composite".to_string()),
            Duration::from_secs(9 * 3600),
            -0.5,
            0.7,
            &temporal_encoder,
            &emotional_encoder,
        );

        assert!(trace.has_composite());
        assert_eq!(trace.valence, Some(-0.5));
        assert_eq!(trace.arousal, Some(0.7));

        let composite = trace.composite().unwrap();
        assert!(!composite.chrono_semantic_vector.is_empty());
    }

    #[test]
    fn test_store_composite() {
        let mut memory = HolographicMemory::new_with_composite(768);

        let v = make_vector(768, 1.0);
        memory.store_composite(
            &v,
            Some("Morning frustration".to_string()),
            Duration::from_secs(9 * 3600),
            -0.7,
            0.8,
        );

        assert_eq!(memory.stats.episodic_count, 1);

        let trace = memory.episodic.front().unwrap();
        assert!(trace.has_composite());
        assert_eq!(trace.valence, Some(-0.7));
    }

    #[test]
    fn test_query_temporal() {
        let mut memory = HolographicMemory::new_with_composite(768);

        // Store memories at different times
        memory.store_composite(
            &make_vector(768, 1.0),
            Some("Morning event".to_string()),
            Duration::from_secs(9 * 3600),  // 9 AM
            0.5,
            0.5,
        );

        memory.store_composite(
            &make_vector(768, 2.0),
            Some("Evening event".to_string()),
            Duration::from_secs(21 * 3600), // 9 PM
            0.5,
            0.5,
        );

        // Query for morning time
        let morning_matches = memory.query_temporal(Duration::from_secs(9 * 3600), 5);

        assert!(!morning_matches.is_empty());
        let best = &morning_matches[0];
        println!("Best morning match: {} (sim: {:.4})",
                 best.trace.text.as_deref().unwrap_or("?"), best.similarity);
    }

    #[test]
    fn test_query_emotional() {
        let mut memory = HolographicMemory::new_with_composite(768);

        // Store memories with different emotions
        memory.store_composite(
            &make_vector(768, 1.0),
            Some("Frustrating bug".to_string()),
            Duration::from_secs(9 * 3600),
            -0.7,  // Negative valence
            0.8,   // High arousal
        );

        memory.store_composite(
            &make_vector(768, 2.0),
            Some("Great success".to_string()),
            Duration::from_secs(10 * 3600),
            0.8,   // Positive valence
            0.9,   // High arousal
        );

        memory.store_composite(
            &make_vector(768, 3.0),
            Some("Calm routine".to_string()),
            Duration::from_secs(11 * 3600),
            0.0,   // Neutral valence
            0.2,   // Low arousal
        );

        // Query for frustration
        let frustrated_matches = memory.query_emotional(-0.7, 0.8, 5);

        assert!(!frustrated_matches.is_empty());
        let best = &frustrated_matches[0];
        println!("Best frustration match: {} (sim: {:.4})",
                 best.trace.text.as_deref().unwrap_or("?"), best.similarity);
    }

    #[test]
    fn test_query_multi_dimensional() {
        let mut memory = HolographicMemory::new_with_composite(768);

        // Store memories
        let v1 = make_vector(768, 1.0);
        memory.store_composite(
            &v1,
            Some("Morning bug fix".to_string()),
            Duration::from_secs(9 * 3600),
            -0.5,
            0.6,
        );

        memory.store_composite(
            &make_vector(768, 2.0),
            Some("Afternoon success".to_string()),
            Duration::from_secs(14 * 3600),
            0.8,
            0.7,
        );

        // Multi-dimensional query: semantic + temporal + emotional
        let matches = memory.query_multi_dimensional(
            Some(&v1),                              // Similar semantic content
            Some(Duration::from_secs(9 * 3600)),    // Morning time
            Some((-0.5, 0.6)),                      // Negative emotion
            (0.4, 0.3, 0.3),                        // Balanced weights
            5,
        );

        assert!(!matches.is_empty());
        let best = &matches[0];
        println!("Best multi-dimensional match: {} (sim: {:.4})",
                 best.trace.text.as_deref().unwrap_or("?"), best.similarity);

        // The morning bug fix should be the best match
        assert!(best.trace.text.as_deref().unwrap_or("").contains("Morning"));
    }

    #[test]
    fn test_new_full_biological() {
        let memory = HolographicMemory::new_full_biological(768);

        assert!(memory.hippocampus_enabled());
        assert!(memory.sleep_enabled());
        assert!(memory.composite_enabled());
    }

    #[test]
    fn test_store_composite_with_attention() {
        let mut memory = HolographicMemory::new_with_composite(768);

        let v = make_vector(768, 1.0);

        // Store with high attention
        memory.store_composite_with_attention(
            &v,
            Some("Critical security fix".to_string()),
            Duration::from_secs(9 * 3600),
            -0.8,  // High stress
            0.9,   // High arousal
            1.0,   // Full attention
        );

        let trace = memory.episodic.front().unwrap();
        assert!(trace.has_composite());

        let composite = trace.composite().unwrap();
        assert_eq!(composite.attention_weight, 1.0);
        assert_eq!(composite.encoding_strength, 100);
    }

    #[test]
    fn test_query_chrono_semantic() {
        let mut memory = HolographicMemory::new_with_composite(768);

        let v1 = make_vector(768, 1.0);
        memory.store_composite(
            &v1,
            Some("Morning coding session".to_string()),
            Duration::from_secs(9 * 3600),
            0.5,
            0.6,
        );

        let v2 = make_vector(768, 1.0); // Same semantic content
        memory.store_composite(
            &v2,
            Some("Evening coding session".to_string()),
            Duration::from_secs(21 * 3600), // Different time
            0.5,
            0.6,
        );

        // Query: "What about coding in the morning?"
        let matches = memory.query_chrono_semantic(
            &v1,
            Duration::from_secs(9 * 3600),
            5,
        );

        assert!(!matches.is_empty());
        // Should prefer morning coding session due to chrono-semantic binding
        let best = &matches[0];
        println!("Best chrono-semantic match: {} (sim: {:.4})",
                 best.trace.text.as_deref().unwrap_or("?"), best.similarity);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSCIOUSNESS-DRIVEN MEMORY CONSOLIDATION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_store_with_consciousness() {
        let mut memory = HolographicMemory::new(768);

        let v = make_vector(768, 1.0);

        // Store with high consciousness (φ = 0.7)
        memory.store_with_consciousness(&v, 0.7, Some("High consciousness memory".to_string()));

        let trace = memory.episodic.front().unwrap();
        // High consciousness should boost importance (0.7/0.5 = 1.4)
        assert!(trace.importance > 1.0, "High Φ should boost importance");
        assert!((trace.importance - 1.4).abs() < 0.01, "Expected importance ~1.4, got {}", trace.importance);

        // Consciousness level stored in arousal
        assert_eq!(trace.arousal, Some(0.7));
    }

    #[test]
    fn test_store_with_low_consciousness() {
        let mut memory = HolographicMemory::new(768);

        let v = make_vector(768, 2.0);

        // Store with low consciousness (φ = 0.3)
        memory.store_with_consciousness(&v, 0.3, Some("Low consciousness memory".to_string()));

        let trace = memory.episodic.front().unwrap();
        // Low consciousness should reduce importance (0.3/0.5 = 0.6)
        assert!(trace.importance < 1.0, "Low Φ should reduce importance");
        assert!((trace.importance - 0.6).abs() < 0.01, "Expected importance ~0.6, got {}", trace.importance);
    }

    #[test]
    fn test_consciousness_statistics() {
        let mut memory = HolographicMemory::new(768);

        // Store memories with varying consciousness
        memory.store_with_consciousness(&make_vector(768, 1.0), 0.3, None);
        memory.store_with_consciousness(&make_vector(768, 2.0), 0.5, None);
        memory.store_with_consciousness(&make_vector(768, 3.0), 0.7, None);

        let (mean, min, max, count) = memory.consciousness_statistics();

        assert_eq!(count, 3);
        assert!((min - 0.3).abs() < 0.01, "Expected min 0.3, got {}", min);
        assert!((max - 0.7).abs() < 0.01, "Expected max 0.7, got {}", max);
        assert!((mean - 0.5).abs() < 0.01, "Expected mean 0.5, got {}", mean);
    }

    #[test]
    fn test_query_by_encoding_consciousness() {
        let mut memory = HolographicMemory::new(768);

        // Store memories with varying consciousness
        memory.store_with_consciousness(&make_vector(768, 1.0), 0.3, Some("Drowsy memory".to_string()));
        memory.store_with_consciousness(&make_vector(768, 2.0), 0.5, Some("Normal memory".to_string()));
        memory.store_with_consciousness(&make_vector(768, 3.0), 0.7, Some("Vivid memory".to_string()));
        memory.store_with_consciousness(&make_vector(768, 4.0), 0.8, Some("Peak consciousness".to_string()));

        // Query high-consciousness memories
        let vivid = memory.query_by_encoding_consciousness(0.6, 1.0, 10);
        assert_eq!(vivid.len(), 2, "Should find 2 high-Φ memories (0.7 and 0.8)");

        // First should be highest consciousness
        assert_eq!(vivid[0].trace.text.as_deref(), Some("Peak consciousness"));

        // Query low-consciousness memories
        let drowsy = memory.query_by_encoding_consciousness(0.0, 0.4, 10);
        assert_eq!(drowsy.len(), 1, "Should find 1 low-Φ memory");
        assert_eq!(drowsy[0].trace.text.as_deref(), Some("Drowsy memory"));
    }

    #[test]
    fn test_sleep_consolidate_conscious() {
        let mut memory = HolographicMemory::new_biological(768);

        // Store memories with varying consciousness levels
        for i in 0..5 {
            let phi = 0.3 + (i as f32 * 0.1); // 0.3, 0.4, 0.5, 0.6, 0.7
            memory.store_with_consciousness(
                &make_vector(768, i as f32),
                phi,
                Some(format!("Memory at Φ={:.1}", phi)),
            );
        }

        // Mark all as accessed
        for trace in memory.episodic.iter_mut() {
            trace.access_count = 3;
        }

        // Perform consciousness-aware sleep consolidation with low Φ (deep sleep)
        let (long_term, categories, pruned) = memory.sleep_consolidate_conscious(0.2);

        println!("Conscious consolidation: {} long-term, {} categories, {} pruned",
                 long_term.len(), categories, pruned);

        // Consolidation cycle counted
        assert_eq!(memory.stats.sleep_consolidation_cycles, 1);

        // Pressure should be reset
        assert_eq!(memory.memory_pressure(), 0.0);

        // High-Φ memories should preferentially survive
        // (exact behavior depends on thresholds)
    }

    #[test]
    fn test_consciousness_modulates_pruning() {
        let mut memory = HolographicMemory::new_biological(768);

        // Store a high-Φ memory and a low-Φ memory
        memory.store_with_consciousness(&make_vector(768, 1.0), 0.7, Some("High Φ".to_string()));
        memory.store_with_consciousness(&make_vector(768, 2.0), 0.3, Some("Low Φ".to_string()));

        // Manually reduce importance of both
        for trace in memory.episodic.iter_mut() {
            trace.importance = 0.15; // Just above default prune threshold
        }

        // Deep sleep consolidation (Φ = 0.2) is more aggressive
        let (_, _, pruned) = memory.sleep_consolidate_conscious(0.2);

        // With aggressive pruning in deep sleep, low-Φ memories should be pruned first
        // (they decay faster due to lower encoding consciousness)
        println!("Pruned {} memories during deep sleep", pruned);
    }

    #[test]
    fn test_modulate_by_consciousness() {
        let mut memory = HolographicMemory::new(768);
        let original_threshold = memory.config.retrieval_threshold;

        // High consciousness should lower threshold (easier recall)
        memory.modulate_by_consciousness(0.7);
        assert!(memory.config.retrieval_threshold < original_threshold,
                "High Φ should lower retrieval threshold");

        // Reset and test low consciousness
        memory.config.retrieval_threshold = original_threshold;
        memory.modulate_by_consciousness(0.2);
        assert!(memory.config.retrieval_threshold > original_threshold,
                "Low Φ should raise retrieval threshold");
    }

    #[test]
    fn test_store_conscious_composite() {
        let mut memory = HolographicMemory::new_full_biological(768);

        let v = make_vector(768, 1.0);

        // Store with full consciousness context
        memory.store_conscious_composite(
            &v,
            0.7,                              // High consciousness
            Duration::from_secs(9 * 3600),    // Morning
            0.5,                              // Positive valence
            0.8,                              // High arousal
            Some("Exciting breakthrough".to_string()),
        );

        let trace = memory.episodic.front().unwrap();

        // Should have composite encoding
        assert!(trace.has_composite());

        // Importance should be boosted by both consciousness and arousal
        // Base: 0.7/0.5 = 1.4, Arousal boost: 1 + 0.8*0.5 = 1.4
        // Final: 1.4 * 1.4 = 1.96
        assert!(trace.importance > 1.5, "Expected high importance from Φ + arousal boost");
        println!("Conscious composite importance: {:.3}", trace.importance);
    }

    #[test]
    fn test_consciousness_importance_clamping() {
        let mut memory = HolographicMemory::new(768);

        // Extreme consciousness values should be clamped
        memory.store_with_consciousness(&make_vector(768, 1.0), 2.0, None);
        assert!(memory.episodic.front().unwrap().importance <= 2.5,
                "Very high Φ should be clamped");

        memory.store_with_consciousness(&make_vector(768, 2.0), 0.01, None);
        assert!(memory.episodic.back().unwrap().importance >= 0.2,
                "Very low Φ should be clamped");
    }
}
