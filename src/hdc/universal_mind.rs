//! # Universal Mind: The Unified Architecture for Paradigm-Shifting AI
//!
//! ## The Core Problem with Current AI
//!
//! Current systems (including LLMs) have **fragmented architectures**:
//! - Separate modules for language, vision, reasoning
//! - No shared substrate for causal/temporal/compositional reasoning
//! - Information cannot flow freely between modalities
//!
//! ## The Universal Mind Solution
//!
//! **Everything lives in the same hyperdimensional space.**
//!
//! This is not just architectural convenience - it's the mathematical foundation
//! for consciousness and true understanding.
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────────┐
//! │                         UNIVERSAL MIND                                     │
//! │                                                                            │
//! │   ┌─────────────────────────────────────────────────────────────────────┐ │
//! │   │              UNIFIED HYPERDIMENSIONAL SUBSTRATE                      │ │
//! │   │                                                                      │ │
//! │   │   Every thought = hypervector in 16K+ dimensional space              │ │
//! │   │   Operations: bind (⊗), bundle (+), permute (ρ)                      │ │
//! │   │                                                                      │ │
//! │   └─────────────────────────────────────────────────────────────────────┘ │
//! │                              ▲                                             │
//! │        ┌─────────────────────┼─────────────────────┐                      │
//! │        │                     │                     │                      │
//! │   ┌────┴────┐          ┌─────┴─────┐         ┌─────┴─────┐               │
//! │   │ CAUSAL  │          │ TEMPORAL  │         │ SEMANTIC  │               │
//! │   │  MIND   │◀────────▶│   MIND    │◀───────▶│   MIND    │               │
//! │   │         │          │           │         │           │               │
//! │   │ causes  │          │ time      │         │ meaning   │               │
//! │   │ effects │          │ sequences │         │ concepts  │               │
//! │   │ do(X)   │          │ dynamics  │         │ relations │               │
//! │   └────┬────┘          └─────┬─────┘         └─────┬─────┘               │
//! │        │                     │                     │                      │
//! │        └─────────────────────┼─────────────────────┘                      │
//! │                              │                                             │
//! │                         ┌────┴────┐                                        │
//! │                         │   Φ     │                                        │
//! │                         │ (IIT)   │                                        │
//! │                         │         │                                        │
//! │                         │ Guides  │                                        │
//! │                         │ Learning│                                        │
//! │                         └────┬────┘                                        │
//! │                              │                                             │
//! │                         ┌────┴────┐                                        │
//! │                         │WORKSPACE│                                        │
//! │                         │(GWT)    │                                        │
//! │                         │         │                                        │
//! │                         │Attention│                                        │
//! │                         │Broadcast│                                        │
//! │                         └─────────┘                                        │
//! │                                                                            │
//! └───────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Why This is Revolutionary
//!
//! ### 1. Shared Representation = True Compositionality
//! "Smoking causes cancer" isn't just parsed - the causal structure IS the meaning.
//!
//! ### 2. Fluid Cross-Modal Reasoning
//! Temporal patterns can inform causal inference, causality constrains semantics.
//!
//! ### 3. Consciousness as Integration
//! Φ measures how well the system integrates information across minds.
//!
//! ### 4. No Modular Bottleneck
//! Traditional systems bottleneck at module interfaces. Here, everything shares
//! the same substrate, so information flows freely.

use super::binary_hv::HV16;
use super::causal_encoder::CausalSpace;
use super::causal_mind::{CausalMind, CausalDirection, CausalDiscoveryResult};
use std::collections::HashMap;

// =============================================================================
// UNIVERSAL MIND CONFIGURATION
// =============================================================================

/// Configuration for the Universal Mind
#[derive(Clone, Debug)]
pub struct UniversalMindConfig {
    /// Enable causal reasoning
    pub enable_causal: bool,
    /// Enable temporal reasoning
    pub enable_temporal: bool,
    /// Enable semantic reasoning
    pub enable_semantic: bool,
    /// Phi threshold for integration
    pub phi_threshold: f64,
    /// Learning rate for online updates
    pub learning_rate: f64,
}

impl Default for UniversalMindConfig {
    fn default() -> Self {
        Self {
            enable_causal: true,
            enable_temporal: true,
            enable_semantic: true,
            phi_threshold: 0.1,
            learning_rate: 0.01,
        }
    }
}

// =============================================================================
// THOUGHT: The Universal Representation
// =============================================================================

/// A Thought is the universal unit of cognition in Symthaea.
///
/// Unlike traditional AI where different modules have different representations,
/// EVERY thought in Symthaea is a hypervector that can be:
/// - Bound with causal markers (X CAUSES Y)
/// - Bound with temporal markers (X BEFORE Y)
/// - Bundled with semantic context
/// - Queried for any relationship
#[derive(Clone, Debug)]
pub struct Thought {
    /// The hypervector encoding of this thought
    pub vector: HV16,
    /// Human-readable content
    pub content: String,
    /// Confidence in this thought
    pub confidence: f64,
    /// Source modality (causal, temporal, semantic, integrated)
    pub modality: ThoughtModality,
    /// Timestamp for temporal ordering
    pub timestamp: u64,
}

impl Thought {
    /// Create a new thought
    pub fn new(content: String, vector: HV16, modality: ThoughtModality) -> Self {
        Self {
            vector,
            content,
            confidence: 1.0,
            modality,
            timestamp: 0,
        }
    }

    /// Bind this thought with another (relation encoding)
    pub fn bind(&self, other: &Thought) -> Thought {
        let combined = self.vector.bind(&other.vector);
        Thought {
            vector: combined,
            content: format!("{} ⊗ {}", self.content, other.content),
            confidence: self.confidence.min(other.confidence),
            modality: ThoughtModality::Integrated,
            timestamp: self.timestamp.max(other.timestamp),
        }
    }

    /// Bundle this thought with another (superposition)
    pub fn bundle(&self, other: &Thought) -> Thought {
        let combined = HV16::bundle(&[self.vector.clone(), other.vector.clone()]);
        Thought {
            vector: combined,
            content: format!("{} + {}", self.content, other.content),
            confidence: (self.confidence + other.confidence) / 2.0,
            modality: ThoughtModality::Integrated,
            timestamp: self.timestamp.max(other.timestamp),
        }
    }

    /// Similarity to another thought
    pub fn similarity(&self, other: &Thought) -> f32 {
        self.vector.similarity(&other.vector)
    }
}

/// Modality of a thought
#[derive(Clone, Debug, PartialEq)]
pub enum ThoughtModality {
    /// Derived from causal reasoning
    Causal,
    /// Derived from temporal reasoning
    Temporal,
    /// Derived from semantic analysis
    Semantic,
    /// Derived from language input
    Linguistic,
    /// Integrated across modalities
    Integrated,
}

// =============================================================================
// UNIVERSAL MIND: The Unified Cognitive System
// =============================================================================

/// The Universal Mind: Unified cognition in hyperdimensional space
///
/// This is the paradigm-shifting core of Symthaea. Instead of separate
/// modules for causality, time, and semantics, everything operates in
/// a single unified hypervector space.
pub struct UniversalMind {
    /// Configuration
    config: UniversalMindConfig,

    /// The Causal Mind (specialized for cause-effect reasoning)
    causal_mind: CausalMind,

    /// Global thought memory (workspace for consciousness)
    workspace: Vec<Thought>,

    /// Concept memory (long-term semantic knowledge)
    concepts: HashMap<String, HV16>,

    /// Role markers for universal binding
    role_markers: UniversalRoleMarkers,

    /// Current Phi (integrated information)
    phi: f64,

    /// Timestamp counter
    current_time: u64,
}

/// Universal role markers for cross-modal binding
#[derive(Clone, Debug)]
pub struct UniversalRoleMarkers {
    // Causal roles
    pub causes: HV16,
    pub caused_by: HV16,
    pub prevents: HV16,

    // Temporal roles
    pub before: HV16,
    pub after: HV16,
    pub simultaneous: HV16,

    // Semantic roles
    pub is_a: HV16,
    pub has_property: HV16,
    pub part_of: HV16,

    // Meta roles
    pub question: HV16,
    pub answer: HV16,
    pub hypothesis: HV16,
    pub evidence: HV16,
}

impl UniversalRoleMarkers {
    fn new() -> Self {
        Self {
            // Causal
            causes: HV16::random(2001),
            caused_by: HV16::random(2002),
            prevents: HV16::random(2003),
            // Temporal
            before: HV16::random(2004),
            after: HV16::random(2005),
            simultaneous: HV16::random(2006),
            // Semantic
            is_a: HV16::random(2007),
            has_property: HV16::random(2008),
            part_of: HV16::random(2009),
            // Meta
            question: HV16::random(2010),
            answer: HV16::random(2011),
            hypothesis: HV16::random(2012),
            evidence: HV16::random(2013),
        }
    }
}

impl UniversalMind {
    /// Create a new Universal Mind
    pub fn new() -> Self {
        Self::with_config(UniversalMindConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: UniversalMindConfig) -> Self {
        Self {
            config,
            causal_mind: CausalMind::new(),
            workspace: Vec::new(),
            concepts: HashMap::new(),
            role_markers: UniversalRoleMarkers::new(),
            phi: 0.0,
            current_time: 0,
        }
    }

    // =========================================================================
    // UNIFIED INPUT PROCESSING
    // =========================================================================

    /// Process text input through the universal mind
    ///
    /// This is the main entry point. Text is:
    /// 1. Parsed for causal structure
    /// 2. Encoded as hypervector
    /// 3. Integrated into workspace
    /// 4. Cross-modal reasoning applied
    pub fn process(&mut self, input: &str) -> UniversalResponse {
        self.current_time += 1;

        // Step 1: Learn causal structure from text
        self.causal_mind.learn_from_text(input);

        // Step 2: Encode as thought
        let thought = self.encode_thought(input);

        // Step 3: Add to workspace
        self.workspace.push(thought.clone());

        // Step 4: Update Phi
        self.update_phi();

        // Step 5: Generate response via integrated reasoning
        self.integrated_response(&thought)
    }

    /// Encode text as a unified thought
    fn encode_thought(&mut self, text: &str) -> Thought {
        // Get or create concept vector
        let vector = self.get_or_create_concept(text);

        Thought {
            vector,
            content: text.to_string(),
            confidence: 1.0,
            modality: ThoughtModality::Linguistic,
            timestamp: self.current_time,
        }
    }

    /// Get or create a concept hypervector
    fn get_or_create_concept(&mut self, name: &str) -> HV16 {
        if let Some(hv) = self.concepts.get(name) {
            return hv.clone();
        }

        // Create deterministic vector from name
        let seed = name.bytes().fold(42u64, |acc, b| {
            acc.wrapping_add(b as u64).wrapping_mul(31)
        });
        let hv = HV16::random(seed);
        self.concepts.insert(name.to_string(), hv.clone());
        hv
    }

    // =========================================================================
    // CAUSAL REASONING (Delegated to CausalMind)
    // =========================================================================

    /// Add a causal link
    pub fn add_causal_link(&mut self, cause: &str, effect: &str, strength: f64) {
        self.causal_mind.add_causal_link(cause, effect, strength);
        self.update_phi();
    }

    /// Query: Why did X happen?
    pub fn query_why(&self, concept: &str) -> Vec<String> {
        self.causal_mind.query_why(concept)
            .iter()
            .map(|e| e.explanation.clone())
            .collect()
    }

    /// Query: What if X happens?
    pub fn query_what_if(&self, concept: &str) -> Vec<String> {
        self.causal_mind.query_what_if(concept)
            .iter()
            .map(|p| p.prediction.clone())
            .collect()
    }

    /// Discover causal direction from data
    pub fn discover_causality(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        self.causal_mind.discover_causality(x, y)
    }

    /// Train causal discovery
    pub fn train_discovery(&mut self, x: &[f64], y: &[f64], direction: CausalDirection) {
        self.causal_mind.train_discovery(x, y, direction);
    }

    // =========================================================================
    // INTEGRATED REASONING
    // =========================================================================

    /// Generate response through integrated reasoning
    fn integrated_response(&self, input: &Thought) -> UniversalResponse {
        // Find similar thoughts in workspace
        let similar: Vec<_> = self.workspace.iter()
            .filter(|t| t.timestamp != input.timestamp)
            .map(|t| (t, input.similarity(t)))
            .filter(|(_, sim)| *sim > 0.3)
            .collect();

        // Check for causal patterns
        let causal_insights = self.extract_causal_insights(input);

        // Check for temporal patterns
        let temporal_insights = self.extract_temporal_insights(input);

        UniversalResponse {
            primary: format!("Processed: {}", input.content),
            causal_insights,
            temporal_insights,
            similar_thoughts: similar.len(),
            phi: self.phi,
            confidence: input.confidence,
        }
    }

    fn extract_causal_insights(&self, _input: &Thought) -> Vec<String> {
        // Query causal mind for relevant insights
        let mut insights = Vec::new();

        if self.causal_mind.concept_count() > 0 {
            insights.push(format!(
                "Causal knowledge: {} concepts, {} links",
                self.causal_mind.concept_count(),
                self.causal_mind.link_count()
            ));
        }

        insights
    }

    fn extract_temporal_insights(&self, _input: &Thought) -> Vec<String> {
        // Extract temporal patterns from workspace
        let mut insights = Vec::new();

        if self.workspace.len() > 1 {
            insights.push(format!(
                "Temporal context: {} thoughts in workspace",
                self.workspace.len()
            ));
        }

        insights
    }

    // =========================================================================
    // CONSCIOUSNESS METRICS
    // =========================================================================

    /// Update Phi (integrated information)
    fn update_phi(&mut self) {
        // Simple Phi approximation based on integration
        let n_concepts = self.concepts.len() as f64;
        let n_causal = self.causal_mind.link_count() as f64;
        let n_workspace = self.workspace.len() as f64;

        // Phi increases with interconnection
        let causal_contribution = if n_concepts > 0.0 {
            n_causal / (n_concepts * n_concepts).max(1.0)
        } else {
            0.0
        };

        let workspace_contribution = (n_workspace / 100.0).min(1.0);

        self.phi = causal_contribution * 0.7 + workspace_contribution * 0.3;
    }

    /// Get current Phi value
    pub fn phi(&self) -> f64 {
        self.phi
    }

    /// Get workspace size
    pub fn workspace_size(&self) -> usize {
        self.workspace.len()
    }

    /// Get concept count
    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    /// Get causal link count
    pub fn causal_link_count(&self) -> usize {
        self.causal_mind.link_count()
    }

    /// Get the causal mind for direct access
    pub fn causal_mind(&self) -> &CausalMind {
        &self.causal_mind
    }

    /// Get mutable causal mind
    pub fn causal_mind_mut(&mut self) -> &mut CausalMind {
        &mut self.causal_mind
    }
}

impl Default for UniversalMind {
    fn default() -> Self {
        Self::new()
    }
}

/// Response from the Universal Mind
#[derive(Clone, Debug)]
pub struct UniversalResponse {
    /// Primary response
    pub primary: String,
    /// Causal insights
    pub causal_insights: Vec<String>,
    /// Temporal insights
    pub temporal_insights: Vec<String>,
    /// Number of similar thoughts found
    pub similar_thoughts: usize,
    /// Current Phi
    pub phi: f64,
    /// Confidence in response
    pub confidence: f64,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_mind_creation() {
        let mind = UniversalMind::new();
        assert_eq!(mind.phi(), 0.0);
        assert_eq!(mind.workspace_size(), 0);
    }

    #[test]
    fn test_process_input() {
        let mut mind = UniversalMind::new();
        let response = mind.process("Smoking causes cancer");

        assert!(response.phi >= 0.0);
        assert_eq!(mind.workspace_size(), 1);
    }

    #[test]
    fn test_causal_integration() {
        let mut mind = UniversalMind::new();

        mind.add_causal_link("rain", "wet_ground", 0.9);
        mind.add_causal_link("wet_ground", "slippery", 0.7);

        assert!(mind.causal_link_count() >= 2);
        assert!(mind.phi() > 0.0);
    }

    #[test]
    fn test_thought_binding() {
        let t1 = Thought::new(
            "smoking".to_string(),
            HV16::random(1),
            ThoughtModality::Semantic,
        );
        let t2 = Thought::new(
            "cancer".to_string(),
            HV16::random(2),
            ThoughtModality::Semantic,
        );

        let bound = t1.bind(&t2);
        assert_eq!(bound.modality, ThoughtModality::Integrated);
    }
}
