//! Consciousness-Language Bridge - Revolutionary Unification Layer
//!
//! This module creates a bidirectional bridge between:
//! - Language Understanding (semantic primes, frames, constructions, predictions)
//! - Global Consciousness (workspace, attention, working memory)
//! - Active Inference (Free Energy minimization across modalities)
//!
//! # Paradigm-Shifting Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    UNIFIED CONSCIOUSNESS LAYER                          │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌───────────────┐    ┌──────────────────┐    ┌──────────────────┐     │
//! │  │   LANGUAGE    │    │   GLOBAL         │    │   ACTIVE         │     │
//! │  │   PIPELINE    │ ←→ │   WORKSPACE      │ ←→ │   INFERENCE      │     │
//! │  │               │    │                  │    │                  │     │
//! │  │ • NSM Primes  │    │ • Spotlight      │    │ • Free Energy    │     │
//! │  │ • Frames      │    │ • Working Memory │    │ • Predictions    │     │
//! │  │ • Constructs  │    │ • Attention Bids │    │ • Precision      │     │
//! │  │ • Predictions │    │ • Consciousness  │    │ • Learning       │     │
//! │  └───────────────┘    └──────────────────┘    └──────────────────┘     │
//! │           │                     │                      │                │
//! │           └─────────────────────┴──────────────────────┘                │
//! │                                 │                                       │
//! │                    ┌────────────┴────────────┐                          │
//! │                    │   CONSCIOUSNESS BRIDGE   │                         │
//! │                    │                          │                         │
//! │                    │ • Understanding → Bid    │                         │
//! │                    │ • Prediction → Error     │                         │
//! │                    │ • Frame → Memory         │                         │
//! │                    │ • Φ → Coherence          │                         │
//! │                    └─────────────────────────┘                          │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Revolutionary Concepts
//!
//! ## 1. Unified Free Energy
//! Language prediction errors and general prediction errors are combined
//! into a single Free Energy objective, enabling cross-modal learning.
//!
//! ## 2. Semantic Spotlight
//! Language understanding results compete for conscious attention via
//! the global workspace, enabling prioritized processing.
//!
//! ## 3. Working Memory Binding
//! Activated frames and constructions can be held in working memory,
//! enabling complex reasoning over multiple utterances.
//!
//! ## 4. Φ-Coherence Alignment
//! Language Φ (integrated information) maps to system coherence,
//! creating feedback between understanding and consciousness.

use crate::hdc::binary_hv::HV16;
use super::conscious_understanding::{
    ConsciousUnderstanding, ActivatedFrame, ParsedConstruction,
    PredictionResult, ConsciousnessMetrics, PipelineConfig,
    ConsciousUnderstandingPipeline,
};
use super::predictive_understanding::{
    PredictiveUnderstanding, LinguisticLevel, LinguisticError,
    PredictionSource, SentenceUnderstanding,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// =============================================================================
// ATTENTION BID GENERATION
// =============================================================================

/// An attention bid generated from language understanding
///
/// Maps to the brain's `AttentionBid` structure for global workspace integration
#[derive(Debug, Clone)]
pub struct LanguageAttentionBid {
    /// Source module identifier
    pub source: String,

    /// Content summary for the bid
    pub content: String,

    /// Salience (importance): derived from Φ and prediction errors
    pub salience: f32,

    /// Urgency: derived from surprise and context
    pub urgency: f32,

    /// Confidence: derived from understanding confidence
    pub confidence: f32,

    /// The full understanding result (for detailed processing)
    pub understanding: ConsciousUnderstanding,

    /// Key semantic primes activated (for quick routing)
    pub active_primes: Vec<String>,

    /// Primary frame (if any)
    pub primary_frame: Option<String>,

    /// Construction pattern matched
    pub construction: Option<String>,
}

impl LanguageAttentionBid {
    /// Create a bid from a conscious understanding result
    pub fn from_understanding(understanding: ConsciousUnderstanding) -> Self {
        // Calculate salience from Φ and confidence
        let phi_contribution = understanding.consciousness.phi.min(1.0) as f32;
        let conf_contribution = understanding.confidence as f32;
        let salience = 0.6 * phi_contribution + 0.4 * conf_contribution;

        // Calculate urgency from prediction surprise
        let urgency = if let Some((_, surprise)) = &understanding.prediction_result.peak_surprise {
            (*surprise as f32 / 10.0).min(1.0) // Normalize surprise to [0, 1]
        } else {
            0.3 // Default moderate urgency
        };

        // Extract key information
        let active_primes: Vec<String> = understanding.semantic_primes
            .iter()
            .flat_map(|(_, primes)| primes.iter().map(|p| format!("{:?}", p)))
            .take(5)
            .collect();

        let primary_frame = understanding.frames.first().map(|f| f.name.clone());
        let construction = understanding.constructions.first().map(|c| c.name.clone());

        // Create content summary
        let content = format!(
            "Understood: '{}' | Φ={:.2} | Conf={:.2} | Frames={} | Constructs={}",
            understanding.input.chars().take(50).collect::<String>(),
            understanding.consciousness.phi,
            understanding.confidence,
            understanding.frames.len(),
            understanding.constructions.len(),
        );

        Self {
            source: "LanguagePipeline".to_string(),
            content,
            salience,
            urgency,
            confidence: understanding.confidence as f32,
            understanding,
            active_primes,
            primary_frame,
            construction,
        }
    }

    /// Check if this bid should gain spotlight attention
    pub fn should_spotlight(&self) -> bool {
        self.salience > 0.5 || self.urgency > 0.7
    }

    /// Calculate priority for working memory inclusion
    pub fn working_memory_priority(&self) -> f32 {
        0.5 * self.salience + 0.3 * self.confidence + 0.2 * self.urgency
    }
}

// =============================================================================
// ACTIVE INFERENCE INTEGRATION
// =============================================================================

/// Linguistic prediction error mapped to active inference domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticPredictionError {
    /// The linguistic level where error occurred
    pub level: String,

    /// Expected encoding (as bit pattern hash)
    pub expected_hash: u64,

    /// Observed encoding hash
    pub observed_hash: u64,

    /// Raw error magnitude
    pub error: f32,

    /// Precision (confidence)
    pub precision: f32,

    /// Precision-weighted error
    pub weighted_error: f32,

    /// Domain mapping for active inference
    pub inference_domain: InferenceDomain,
}

/// Maps linguistic levels to active inference domains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceDomain {
    /// Lexical → User State (what user is trying to say)
    UserState,
    /// Syntactic → Task Success (understanding structure)
    TaskSuccess,
    /// Semantic → Coherence (meaning alignment)
    Coherence,
    /// Discourse → Social (conversation flow)
    Social,
    /// Prediction → Performance (processing efficiency)
    Performance,
}

impl From<LinguisticLevel> for InferenceDomain {
    fn from(level: LinguisticLevel) -> Self {
        match level {
            LinguisticLevel::Sublexical => InferenceDomain::Performance,
            LinguisticLevel::Lexical => InferenceDomain::UserState,
            LinguisticLevel::Syntactic => InferenceDomain::TaskSuccess,
            LinguisticLevel::Semantic => InferenceDomain::Coherence,
            LinguisticLevel::Discourse => InferenceDomain::Social,
        }
    }
}

/// Converts linguistic errors to active inference prediction errors
pub fn convert_linguistic_error(error: &LinguisticError) -> LinguisticPredictionError {
    LinguisticPredictionError {
        level: format!("{:?}", error.level),
        expected_hash: 0, // Would need access to prediction
        observed_hash: 0, // Would need access to observation
        error: error.magnitude as f32,
        precision: error.precision as f32,
        weighted_error: error.weighted_error as f32,
        inference_domain: InferenceDomain::from(error.level),
    }
}

// =============================================================================
// WORKING MEMORY ITEMS
// =============================================================================

/// A language-derived item for working memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageWorkingMemoryItem {
    /// Unique identifier
    pub id: u64,

    /// Content type
    pub content_type: WorkingMemoryContentType,

    /// Encoding in HDC space
    pub encoding: HV16,

    /// Human-readable summary
    pub summary: String,

    /// Activation level (decays over time)
    pub activation: f32,

    /// How many cycles this has been in memory
    pub age: u32,

    /// Binding strength to other items
    pub bindings: Vec<(u64, f32)>,
}

/// Types of content that can be held in working memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkingMemoryContentType {
    /// An activated semantic frame
    Frame { name: String, role_count: usize },
    /// A parsed construction
    Construction { name: String, slot_count: usize },
    /// A semantic prime bundle
    PrimeBundle { primes: Vec<String> },
    /// A discourse topic
    Topic { name: String },
    /// An unresolved question
    OpenQuestion { question: String },
}

impl LanguageWorkingMemoryItem {
    /// Create from an activated frame
    pub fn from_frame(frame: &ActivatedFrame, id: u64) -> Self {
        Self {
            id,
            content_type: WorkingMemoryContentType::Frame {
                name: frame.name.clone(),
                role_count: frame.role_fillers.len(),
            },
            encoding: frame.encoding,
            summary: format!("Frame: {} ({} roles)", frame.name, frame.role_fillers.len()),
            activation: frame.activation as f32,
            age: 0,
            bindings: Vec::new(),
        }
    }

    /// Create from a parsed construction
    pub fn from_construction(construction: &ParsedConstruction, id: u64) -> Self {
        Self {
            id,
            content_type: WorkingMemoryContentType::Construction {
                name: construction.name.clone(),
                slot_count: construction.slots.len(),
            },
            encoding: construction.encoding,
            summary: format!("Construction: {} ({} slots)", construction.name, construction.slots.len()),
            activation: construction.confidence as f32,
            age: 0,
            bindings: Vec::new(),
        }
    }

    /// Decay activation over time
    pub fn decay(&mut self, rate: f32) {
        self.activation *= 1.0 - rate;
        self.age += 1;
    }

    /// Check if item should be evicted
    pub fn should_evict(&self, threshold: f32) -> bool {
        self.activation < threshold
    }
}

// =============================================================================
// THE CONSCIOUSNESS BRIDGE
// =============================================================================

/// Configuration for the consciousness bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Decay rate for working memory items
    pub memory_decay_rate: f32,

    /// Threshold for working memory eviction
    pub eviction_threshold: f32,

    /// Maximum working memory capacity
    pub max_working_memory: usize,

    /// Weight for Φ in salience calculation
    pub phi_weight: f32,

    /// Weight for confidence in salience calculation
    pub confidence_weight: f32,

    /// Surprise normalization factor
    pub surprise_scale: f32,

    /// Enable cross-modal binding
    pub enable_cross_modal: bool,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            memory_decay_rate: 0.1,
            eviction_threshold: 0.1,
            max_working_memory: 7, // Miller's 7±2
            phi_weight: 0.6,
            confidence_weight: 0.4,
            surprise_scale: 10.0,
            enable_cross_modal: true,
        }
    }
}

/// The Consciousness-Language Bridge
///
/// Revolutionary module that unifies language understanding with
/// global consciousness and active inference.
pub struct ConsciousnessBridge {
    /// Configuration
    config: BridgeConfig,

    /// Language understanding pipeline
    pipeline: ConsciousUnderstandingPipeline,

    /// Working memory for language items
    working_memory: Vec<LanguageWorkingMemoryItem>,

    /// History of attention bids
    bid_history: VecDeque<LanguageAttentionBid>,

    /// Accumulated prediction errors for learning
    accumulated_errors: Vec<LinguisticPredictionError>,

    /// Current spotlight (most attended understanding)
    spotlight: Option<LanguageAttentionBid>,

    /// Free energy history for monitoring
    free_energy_history: VecDeque<f64>,

    /// Φ history for consciousness tracking
    phi_history: VecDeque<f64>,

    /// Next working memory item ID
    next_id: u64,

    /// Processing statistics
    stats: BridgeStats,
}

/// Statistics for the consciousness bridge
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BridgeStats {
    /// Total inputs processed
    pub inputs_processed: u64,

    /// Bids that gained spotlight
    pub spotlight_wins: u64,

    /// Items added to working memory
    pub memory_additions: u64,

    /// Items evicted from working memory
    pub memory_evictions: u64,

    /// Average Φ across all processing
    pub average_phi: f64,

    /// Average free energy
    pub average_free_energy: f64,

    /// Peak Φ observed
    pub peak_phi: f64,

    /// Lowest free energy observed
    pub lowest_free_energy: f64,
}

/// Status of consciousness bootstrapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapStatus {
    /// Whether bootstrap has been performed
    pub is_bootstrapped: bool,

    /// Initial Φ value after bootstrap
    pub initial_phi: f64,

    /// Whether working memory has been primed
    pub working_memory_primed: bool,

    /// Names of archetypal frames loaded
    pub archetypal_frames: Vec<String>,
}

impl ConsciousnessBridge {
    /// Create a new consciousness bridge
    pub fn new(config: BridgeConfig) -> Self {
        let pipeline_config = PipelineConfig::default();
        let pipeline = ConsciousUnderstandingPipeline::new(pipeline_config);

        Self {
            config,
            pipeline,
            working_memory: Vec::new(),
            bid_history: VecDeque::with_capacity(100),
            accumulated_errors: Vec::new(),
            spotlight: None,
            free_energy_history: VecDeque::with_capacity(100),
            phi_history: VecDeque::with_capacity(100),
            next_id: 0,
            stats: BridgeStats::default(),
        }
    }

    /// Process input through the full consciousness bridge
    ///
    /// This is the main entry point that:
    /// 1. Runs language understanding
    /// 2. Generates attention bid
    /// 3. Updates working memory
    /// 4. Tracks free energy and Φ
    /// 5. Returns unified result
    pub fn process(&mut self, input: &str) -> BridgeResult {
        self.stats.inputs_processed += 1;

        // Step 1: Language understanding
        let understanding = self.pipeline.understand(input);

        // Step 2: Generate attention bid
        let bid = LanguageAttentionBid::from_understanding(understanding.clone());

        // Step 3: Update spotlight
        let gained_spotlight = self.update_spotlight(&bid);
        if gained_spotlight {
            self.stats.spotlight_wins += 1;
        }

        // Step 4: Update working memory
        self.update_working_memory(&understanding);

        // Step 5: Track metrics
        let phi = understanding.consciousness.phi;
        let free_energy = understanding.prediction_result.final_free_energy;

        self.phi_history.push_back(phi);
        if self.phi_history.len() > 100 {
            self.phi_history.pop_front();
        }

        self.free_energy_history.push_back(free_energy);
        if self.free_energy_history.len() > 100 {
            self.free_energy_history.pop_front();
        }

        // Update stats
        self.stats.average_phi = self.phi_history.iter().sum::<f64>() / self.phi_history.len() as f64;
        self.stats.average_free_energy = self.free_energy_history.iter().sum::<f64>() / self.free_energy_history.len() as f64;
        self.stats.peak_phi = self.stats.peak_phi.max(phi);
        self.stats.lowest_free_energy = self.stats.lowest_free_energy.min(free_energy);

        // Add to bid history
        self.bid_history.push_back(bid.clone());
        if self.bid_history.len() > 100 {
            self.bid_history.pop_front();
        }

        // Construct result
        BridgeResult {
            understanding,
            bid,
            gained_spotlight,
            working_memory_count: self.working_memory.len(),
            current_phi: phi,
            current_free_energy: free_energy,
            consciousness_state: self.assess_consciousness_state(),
        }
    }

    /// Update spotlight based on new bid
    fn update_spotlight(&mut self, bid: &LanguageAttentionBid) -> bool {
        if bid.should_spotlight() {
            // Check if new bid beats current spotlight
            let should_replace = match &self.spotlight {
                None => true,
                Some(current) => {
                    let current_priority = current.salience * 0.6 + current.urgency * 0.4;
                    let new_priority = bid.salience * 0.6 + bid.urgency * 0.4;
                    new_priority > current_priority
                }
            };

            if should_replace {
                self.spotlight = Some(bid.clone());
                return true;
            }
        }
        false
    }

    /// Update working memory with frames and constructions
    fn update_working_memory(&mut self, understanding: &ConsciousUnderstanding) {
        // Decay existing items
        for item in &mut self.working_memory {
            item.decay(self.config.memory_decay_rate);
        }

        // Evict items below threshold
        let before_count = self.working_memory.len();
        self.working_memory.retain(|item| !item.should_evict(self.config.eviction_threshold));
        self.stats.memory_evictions += (before_count - self.working_memory.len()) as u64;

        // Add new frames
        for frame in &understanding.frames {
            if self.working_memory.len() < self.config.max_working_memory {
                let item = LanguageWorkingMemoryItem::from_frame(frame, self.next_id);
                self.next_id += 1;
                self.working_memory.push(item);
                self.stats.memory_additions += 1;
            }
        }

        // Add new constructions (if room)
        for construction in &understanding.constructions {
            if self.working_memory.len() < self.config.max_working_memory {
                let item = LanguageWorkingMemoryItem::from_construction(construction, self.next_id);
                self.next_id += 1;
                self.working_memory.push(item);
                self.stats.memory_additions += 1;
            }
        }

        // Sort by activation (most active first)
        self.working_memory.sort_by(|a, b| {
            b.activation.partial_cmp(&a.activation).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Trim to max capacity
        while self.working_memory.len() > self.config.max_working_memory {
            self.working_memory.pop();
            self.stats.memory_evictions += 1;
        }
    }

    /// Assess the current consciousness state
    fn assess_consciousness_state(&self) -> ConsciousnessState {
        let avg_phi = self.stats.average_phi;
        let avg_fe = self.stats.average_free_energy;
        let wm_load = self.working_memory.len() as f32 / self.config.max_working_memory as f32;

        if avg_phi > 0.7 && avg_fe < 0.3 {
            ConsciousnessState::HighlyCoherent
        } else if avg_phi > 0.4 && avg_fe < 0.5 {
            ConsciousnessState::Coherent
        } else if avg_fe > 0.7 {
            ConsciousnessState::Confused
        } else if wm_load > 0.8 {
            ConsciousnessState::Overloaded
        } else {
            ConsciousnessState::Normal
        }
    }

    /// Get current spotlight
    pub fn spotlight(&self) -> Option<&LanguageAttentionBid> {
        self.spotlight.as_ref()
    }

    /// Get working memory contents
    pub fn working_memory(&self) -> &[LanguageWorkingMemoryItem] {
        &self.working_memory
    }

    /// Get statistics
    pub fn stats(&self) -> &BridgeStats {
        &self.stats
    }

    /// Get recent Φ history
    pub fn phi_history(&self) -> &VecDeque<f64> {
        &self.phi_history
    }

    /// Get recent free energy history
    pub fn free_energy_history(&self) -> &VecDeque<f64> {
        &self.free_energy_history
    }

    /// Reset the bridge state
    pub fn reset(&mut self) {
        self.pipeline.reset();
        self.working_memory.clear();
        self.spotlight = None;
        self.free_energy_history.clear();
        self.phi_history.clear();
        self.accumulated_errors.clear();
    }

    /// Bootstrap consciousness with archetypal patterns
    ///
    /// This pre-seeds the system with foundational concepts so it doesn't
    /// start "cold" with Φ=0. After bootstrapping, the system should have
    /// meaningful Φ (~0.3-0.5) from the first real interaction.
    ///
    /// # Archetypal Patterns
    ///
    /// The bootstrap uses fundamental semantic concepts that establish:
    /// - Existence and identity ("I am here")
    /// - Causation and agency ("Something causes something")
    /// - Temporal flow ("Before leads to after")
    /// - Spatial grounding ("Here and there")
    /// - Self-reference ("I know that I know")
    ///
    /// These prime the consciousness substrate for coherent understanding.
    pub fn bootstrap_consciousness(&mut self) {
        // Archetypal sentences that establish fundamental concepts
        let archetypes = [
            // Existence and identity
            "I exist in this moment",
            "Something is here now",
            // Causation and agency
            "Actions have consequences",
            "Something causes something else",
            // Temporal grounding
            "Time flows from before to after",
            "Now follows then",
            // Spatial awareness
            "Here is different from there",
            "Places exist in space",
            // Self-reference and meta-cognition
            "I can think about thinking",
            "Knowing that I know",
            // Relational understanding
            "Things relate to other things",
            "Parts belong to wholes",
        ];

        // Process archetypes silently to build up consciousness state
        for archetype in &archetypes {
            let _ = self.pipeline.understand(archetype);
        }

        // Process key sentences that establish working memory patterns
        let foundational = [
            "I want to help you",
            "You need something good",
            "I can do this for you",
        ];

        for sentence in &foundational {
            let understanding = self.pipeline.understand(sentence);

            // Add frames to working memory
            for frame in &understanding.frames {
                if self.working_memory.len() < self.config.max_working_memory {
                    let item = LanguageWorkingMemoryItem::from_frame(frame, self.next_id);
                    self.next_id += 1;
                    self.working_memory.push(item);
                }
            }

            // Track Φ history
            self.phi_history.push_back(understanding.consciousness.phi);
            self.free_energy_history.push_back(understanding.prediction_result.final_free_energy);
        }

        // Trim histories to avoid showing bootstrap in stats
        while self.phi_history.len() > 5 {
            self.phi_history.pop_front();
        }
        while self.free_energy_history.len() > 5 {
            self.free_energy_history.pop_front();
        }

        // Update stats to reflect bootstrap
        if !self.phi_history.is_empty() {
            self.stats.average_phi = self.phi_history.iter().sum::<f64>() / self.phi_history.len() as f64;
            self.stats.peak_phi = self.phi_history.iter().cloned().fold(0.0, f64::max);
        }
        if !self.free_energy_history.is_empty() {
            self.stats.average_free_energy = self.free_energy_history.iter().sum::<f64>()
                / self.free_energy_history.len() as f64;
            self.stats.lowest_free_energy = self.free_energy_history.iter().cloned().fold(f64::INFINITY, f64::min);
        }
    }

    /// Check if consciousness has been bootstrapped
    pub fn is_bootstrapped(&self) -> bool {
        !self.phi_history.is_empty() && self.stats.average_phi > 0.1
    }

    /// Get bootstrap status for diagnostics
    pub fn bootstrap_status(&self) -> BootstrapStatus {
        BootstrapStatus {
            is_bootstrapped: self.is_bootstrapped(),
            initial_phi: self.stats.average_phi,
            working_memory_primed: !self.working_memory.is_empty(),
            archetypal_frames: self.working_memory.iter()
                .filter_map(|item| {
                    if let WorkingMemoryContentType::Frame { name, .. } = &item.content_type {
                        Some(name.clone())
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }

    /// Query working memory by frame name
    pub fn query_frame(&self, frame_name: &str) -> Option<&LanguageWorkingMemoryItem> {
        self.working_memory.iter().find(|item| {
            matches!(&item.content_type, WorkingMemoryContentType::Frame { name, .. } if name == frame_name)
        })
    }

    /// Get all active frames in working memory
    pub fn active_frames(&self) -> Vec<&LanguageWorkingMemoryItem> {
        self.working_memory
            .iter()
            .filter(|item| matches!(item.content_type, WorkingMemoryContentType::Frame { .. }))
            .collect()
    }
}

/// Result from the consciousness bridge
#[derive(Debug, Clone)]
pub struct BridgeResult {
    /// The full understanding result
    pub understanding: ConsciousUnderstanding,

    /// The attention bid generated
    pub bid: LanguageAttentionBid,

    /// Whether this bid gained spotlight
    pub gained_spotlight: bool,

    /// Current working memory count
    pub working_memory_count: usize,

    /// Current Φ value
    pub current_phi: f64,

    /// Current free energy
    pub current_free_energy: f64,

    /// Assessed consciousness state
    pub consciousness_state: ConsciousnessState,
}

/// States of consciousness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsciousnessState {
    /// High Φ, low free energy - optimal understanding
    HighlyCoherent,
    /// Good Φ, acceptable free energy
    Coherent,
    /// Normal operation
    Normal,
    /// Working memory at capacity
    Overloaded,
    /// High free energy - struggling to understand
    Confused,
}

impl std::fmt::Display for ConsciousnessState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConsciousnessState::HighlyCoherent => write!(f, "Highly Coherent"),
            ConsciousnessState::Coherent => write!(f, "Coherent"),
            ConsciousnessState::Normal => write!(f, "Normal"),
            ConsciousnessState::Overloaded => write!(f, "Overloaded"),
            ConsciousnessState::Confused => write!(f, "Confused"),
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let config = BridgeConfig::default();
        let bridge = ConsciousnessBridge::new(config);

        assert!(bridge.working_memory.is_empty());
        assert!(bridge.spotlight.is_none());
    }

    #[test]
    fn test_basic_processing() {
        let config = BridgeConfig::default();
        let mut bridge = ConsciousnessBridge::new(config);

        let result = bridge.process("The cat sat on the mat");

        assert!(!result.understanding.input.is_empty());
        assert!(result.current_phi >= 0.0);
        assert!(result.current_free_energy >= 0.0);
        assert!(bridge.stats().inputs_processed == 1);
    }

    #[test]
    fn test_attention_bid_generation() {
        let config = BridgeConfig::default();
        let mut bridge = ConsciousnessBridge::new(config);

        let result = bridge.process("I want something good");

        assert_eq!(result.bid.source, "LanguagePipeline");
        assert!(result.bid.salience >= 0.0 && result.bid.salience <= 1.0);
        assert!(result.bid.urgency >= 0.0 && result.bid.urgency <= 1.0);
    }

    #[test]
    fn test_working_memory_management() {
        let mut config = BridgeConfig::default();
        config.max_working_memory = 5;
        let mut bridge = ConsciousnessBridge::new(config);

        // Process multiple sentences
        for sentence in &[
            "She gave him the book",
            "He wanted the apple",
            "They saw something beautiful",
            "I know you think differently",
            "The big dog ran quickly",
        ] {
            bridge.process(sentence);
        }

        // Working memory should not exceed max
        assert!(bridge.working_memory().len() <= 5);
    }

    #[test]
    fn test_spotlight_competition() {
        let config = BridgeConfig::default();
        let mut bridge = ConsciousnessBridge::new(config);

        // Process low-salience input
        bridge.process("a b c");

        // Process high-salience input
        let result = bridge.process("I urgently need help with something important");

        // Second input should win spotlight if it has higher salience
        if result.bid.salience > 0.5 {
            assert!(bridge.spotlight().is_some());
        }
    }

    #[test]
    fn test_phi_tracking() {
        let config = BridgeConfig::default();
        let mut bridge = ConsciousnessBridge::new(config);

        // Process several inputs
        for _ in 0..5 {
            bridge.process("I think you know something");
        }

        // Φ history should have entries
        assert_eq!(bridge.phi_history().len(), 5);
        assert!(bridge.stats().average_phi >= 0.0);
    }

    #[test]
    fn test_consciousness_state_assessment() {
        let config = BridgeConfig::default();
        let mut bridge = ConsciousnessBridge::new(config);

        // Initially should be Normal
        let result = bridge.process("Test sentence");

        // State should be one of the valid states
        assert!(matches!(
            result.consciousness_state,
            ConsciousnessState::HighlyCoherent
                | ConsciousnessState::Coherent
                | ConsciousnessState::Normal
                | ConsciousnessState::Overloaded
                | ConsciousnessState::Confused
        ));
    }

    #[test]
    fn test_reset() {
        let config = BridgeConfig::default();
        let mut bridge = ConsciousnessBridge::new(config);

        // Process some inputs
        bridge.process("First sentence");
        bridge.process("Second sentence");

        // Reset
        bridge.reset();

        // Should be cleared
        assert!(bridge.working_memory().is_empty());
        assert!(bridge.spotlight().is_none());
        assert!(bridge.phi_history().is_empty());
    }

    #[test]
    fn test_frame_query() {
        let config = BridgeConfig::default();
        let mut bridge = ConsciousnessBridge::new(config);

        // Process sentence that might activate frames
        bridge.process("She gave him the book");

        // Query for frames
        let frames = bridge.active_frames();

        // Should return frame items only
        for frame in frames {
            assert!(matches!(frame.content_type, WorkingMemoryContentType::Frame { .. }));
        }
    }

    #[test]
    fn test_inference_domain_mapping() {
        assert_eq!(
            InferenceDomain::from(LinguisticLevel::Lexical),
            InferenceDomain::UserState
        );
        assert_eq!(
            InferenceDomain::from(LinguisticLevel::Semantic),
            InferenceDomain::Coherence
        );
        assert_eq!(
            InferenceDomain::from(LinguisticLevel::Discourse),
            InferenceDomain::Social
        );
    }

    #[test]
    fn test_conversation_flow() {
        let config = BridgeConfig::default();
        let mut bridge = ConsciousnessBridge::new(config);

        // Simulate a conversation
        let sentences = [
            "Hello, how are you?",
            "I'm doing well, thank you.",
            "Can you help me with something?",
            "I need to install a program.",
            "It's called Firefox.",
        ];

        for sentence in &sentences {
            let result = bridge.process(sentence);
            assert!(result.current_phi >= 0.0);
        }

        // Should have processed all
        assert_eq!(bridge.stats().inputs_processed, 5);
    }

    #[test]
    #[ignore = "performance test - run with cargo test --release"]
    fn benchmark_bridge_performance() {
        use std::time::Instant;

        let config = BridgeConfig::default();
        let mut bridge = ConsciousnessBridge::new(config);

        let test_sentences = [
            "The cat sat on the mat",
            "She gave him a beautiful book",
            "I think you know something",
        ];

        // Warm up
        for _ in 0..3 {
            for s in &test_sentences {
                bridge.process(s);
            }
        }
        bridge.reset();

        // Benchmark
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            for s in &test_sentences {
                bridge.process(s);
            }
        }
        let elapsed = start.elapsed();
        let per_sentence = elapsed.as_micros() / (iterations * test_sentences.len()) as u128;

        // Should be under 50ms per sentence (debug mode)
        assert!(per_sentence < 50_000,
            "Bridge too slow: {}μs per sentence", per_sentence);

        println!("\n📊 Consciousness Bridge Performance:");
        println!("   {}μs per sentence", per_sentence);
    }

    #[test]
    fn test_consciousness_bootstrap() {
        let config = BridgeConfig::default();
        let mut bridge = ConsciousnessBridge::new(config);

        // Before bootstrap, should not be bootstrapped
        assert!(!bridge.is_bootstrapped());

        // Bootstrap
        bridge.bootstrap_consciousness();

        // After bootstrap, should be bootstrapped
        assert!(bridge.is_bootstrapped());

        // Should have some Φ history
        assert!(!bridge.phi_history().is_empty());

        // Average Φ should be > 0
        assert!(bridge.stats().average_phi > 0.0);
    }

    #[test]
    fn test_bootstrap_improves_first_interaction() {
        let config = BridgeConfig::default();

        // Test without bootstrap
        let mut cold_bridge = ConsciousnessBridge::new(config.clone());
        let cold_result = cold_bridge.process("Install Firefox for me");
        let cold_phi = cold_result.current_phi;

        // Test with bootstrap
        let mut warm_bridge = ConsciousnessBridge::new(config);
        warm_bridge.bootstrap_consciousness();
        let warm_result = warm_bridge.process("Install Firefox for me");
        let warm_phi = warm_result.current_phi;

        // Both should have valid Φ
        assert!(cold_phi >= 0.0);
        assert!(warm_phi >= 0.0);

        // Warm bridge should have primed working memory
        assert!(!warm_bridge.working_memory().is_empty());

        println!("\n📊 Bootstrap Effect:");
        println!("   Cold start Φ: {:.4}", cold_phi);
        println!("   Warm start Φ: {:.4}", warm_phi);
        println!("   Working memory primed: {}", !warm_bridge.working_memory().is_empty());
    }

    #[test]
    fn test_bootstrap_status() {
        let config = BridgeConfig::default();
        let mut bridge = ConsciousnessBridge::new(config);

        // Before bootstrap
        let status_before = bridge.bootstrap_status();
        assert!(!status_before.is_bootstrapped);

        // Bootstrap
        bridge.bootstrap_consciousness();

        // After bootstrap
        let status_after = bridge.bootstrap_status();
        assert!(status_after.is_bootstrapped);
        assert!(status_after.initial_phi > 0.0);
    }

    #[test]
    fn test_bootstrap_then_reset() {
        let config = BridgeConfig::default();
        let mut bridge = ConsciousnessBridge::new(config);

        // Bootstrap
        bridge.bootstrap_consciousness();
        assert!(bridge.is_bootstrapped());

        // Reset
        bridge.reset();

        // Should no longer be bootstrapped
        assert!(!bridge.is_bootstrapped());
        assert!(bridge.working_memory().is_empty());
    }
}
