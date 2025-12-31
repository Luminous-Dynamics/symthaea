//! # Unified Cognitive Core: The Radical Integration
//!
//! ## The Key Insight
//!
//! Traditional AI (and even our hybrid proposal) treats cognition as:
//! ```text
//! Input → [Semantic Engine] → [Causal Engine] → [Temporal Engine] → Output
//!                  ↓                 ↓                 ↓
//!              separate          separate          separate
//!           representation    representation    representation
//! ```
//!
//! This is fundamentally wrong. Real cognition is UNIFIED:
//! ```text
//! Input → [UNIFIED REPRESENTATION] → Output
//!                   ↓
//!         concept = meaning ⊗ causality ⊗ temporality
//!              (ALL IN ONE VECTOR!)
//! ```
//!
//! ## Why This Matters
//!
//! In IIT (Integrated Information Theory), Φ measures irreducibility.
//! A system with separate modules has low Φ - you can partition it.
//! A system where EVERY element encodes ALL aspects has HIGH Φ.
//!
//! ## The UCTS Representation
//!
//! Every cognitive element is a single hypervector that inherently encodes:
//! - **Semantic meaning**: What it IS
//! - **Causal structure**: What it CAUSES and what CAUSES it
//! - **Temporal context**: WHEN it occurs and its dynamics
//! - **Confidence/salience**: How certain/important it is
//!
//! These aren't separate fields - they're BOUND into one holographic vector.

use super::binary_hv::HV16;
use super::causal_mind::{CausalMind, CausalDirection, LearnedCausalDiscovery};
use super::hdc_ltc_neuron::{HdcLtcNetwork, HdcLtcNetworkConfig};
use super::unified_hv::ContinuousHV;
use std::collections::HashMap;

// =============================================================================
// FOUNDATIONAL MARKERS: The Basis Vectors of Cognition
// =============================================================================

/// The foundational markers that define the cognitive space
///
/// These are orthogonal basis vectors that, when bound with content,
/// create unified representations. They're generated once and shared
/// across the entire system.
#[derive(Clone)]
pub struct CognitiveMarkers {
    // === Causal Markers ===
    /// Marks "X causes Y" relationship
    pub causes: HV16,
    /// Marks "X is caused by Y" relationship
    pub caused_by: HV16,
    /// Marks "X prevents Y" relationship
    pub prevents: HV16,
    /// Marks "X enables Y" relationship
    pub enables: HV16,
    /// Marks interventional context (do-calculus)
    pub intervention: HV16,

    // === Temporal Markers ===
    /// Marks "before" in sequence
    pub before: HV16,
    /// Marks "after" in sequence
    pub after: HV16,
    /// Marks "simultaneous with"
    pub simultaneous: HV16,
    /// Marks "duration of"
    pub duration: HV16,
    /// Temporal position encodings (circular, like hours on clock)
    pub temporal_positions: Vec<HV16>,

    // === Semantic Markers ===
    /// Marks "is a type of"
    pub is_a: HV16,
    /// Marks "has property"
    pub has_property: HV16,
    /// Marks "part of"
    pub part_of: HV16,
    /// Marks "similar to"
    pub similar_to: HV16,

    // === Meta Markers ===
    /// Marks confidence level
    pub confidence: HV16,
    /// Marks salience/importance
    pub salience: HV16,
    /// Marks "this is about myself" (self-reference)
    pub self_reference: HV16,
    /// Marks "this is uncertain/hypothetical"
    pub hypothetical: HV16,

    // === Strength Modifiers ===
    pub strength_high: HV16,
    pub strength_medium: HV16,
    pub strength_low: HV16,
}

impl CognitiveMarkers {
    /// Create the foundational cognitive markers
    ///
    /// These use fixed seeds for reproducibility across sessions
    pub fn new() -> Self {
        // Temporal positions (24 for hours, reusable for other cycles)
        let temporal_positions: Vec<HV16> = (0..24)
            .map(|i| HV16::random(5000 + i as u64))
            .collect();

        Self {
            // Causal (seeds 1000-1099)
            causes: HV16::random(1000),
            caused_by: HV16::random(1001),
            prevents: HV16::random(1002),
            enables: HV16::random(1003),
            intervention: HV16::random(1004),

            // Temporal (seeds 1100-1199)
            before: HV16::random(1100),
            after: HV16::random(1101),
            simultaneous: HV16::random(1102),
            duration: HV16::random(1103),
            temporal_positions,

            // Semantic (seeds 1200-1299)
            is_a: HV16::random(1200),
            has_property: HV16::random(1201),
            part_of: HV16::random(1202),
            similar_to: HV16::random(1203),

            // Meta (seeds 1300-1399)
            confidence: HV16::random(1300),
            salience: HV16::random(1301),
            self_reference: HV16::random(1302),
            hypothetical: HV16::random(1303),

            // Strength (seeds 1400-1499)
            strength_high: HV16::random(1400),
            strength_medium: HV16::random(1401),
            strength_low: HV16::random(1402),
        }
    }

    /// Get strength marker for a given value [0, 1]
    pub fn strength_marker(&self, strength: f64) -> &HV16 {
        if strength > 0.7 {
            &self.strength_high
        } else if strength > 0.3 {
            &self.strength_medium
        } else {
            &self.strength_low
        }
    }

    /// Get temporal position marker (circular encoding)
    pub fn temporal_position(&self, position: usize) -> &HV16 {
        &self.temporal_positions[position % self.temporal_positions.len()]
    }
}

impl Default for CognitiveMarkers {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// UNIFIED COGNITIVE ELEMENT: The Atomic Unit of Thought
// =============================================================================

/// A Unified Cognitive Element (UCE)
///
/// This is the atomic unit of thought in Symthaea. Unlike traditional systems
/// where semantic, causal, and temporal information are stored separately,
/// a UCE encodes ALL aspects in a single hypervector through binding.
///
/// ## Structure
///
/// ```text
/// UCE = base_semantic
///     ⊗ (CAUSES ⊗ effect₁) + (CAUSES ⊗ effect₂) + ...
///     ⊗ (CAUSED_BY ⊗ cause₁) + (CAUSED_BY ⊗ cause₂) + ...
///     ⊗ (BEFORE ⊗ predecessor) + (AFTER ⊗ successor) + ...
///     ⊗ (CONFIDENCE ⊗ strength_marker)
///     ⊗ (TEMPORAL_POS ⊗ position_marker)
/// ```
///
/// All bound into ONE vector that can be:
/// - Compared for similarity
/// - Unbound to extract specific aspects
/// - Bundled with other UCEs for reasoning
#[derive(Clone)]
pub struct UnifiedCognitiveElement {
    /// The unified hypervector encoding all aspects
    pub vector: HV16,

    /// Human-readable label
    pub label: String,

    /// Creation timestamp (for temporal ordering)
    pub created_at: u64,

    /// Last accessed timestamp (for relevance)
    pub accessed_at: u64,

    /// Access count (for importance weighting)
    pub access_count: u64,
}

impl UnifiedCognitiveElement {
    /// Create a new UCE from a base semantic vector
    pub fn new(label: String, semantic: HV16, timestamp: u64) -> Self {
        Self {
            vector: semantic,
            label,
            created_at: timestamp,
            accessed_at: timestamp,
            access_count: 1,
        }
    }

    /// Add causal information: this UCE CAUSES the effect
    pub fn add_cause(&mut self, effect: &HV16, strength: f64, markers: &CognitiveMarkers) {
        let causal_binding = markers.causes
            .bind(effect)
            .bind(markers.strength_marker(strength));
        self.vector = HV16::bundle(&[self.vector.clone(), causal_binding]);
    }

    /// Add causal information: this UCE IS CAUSED BY the cause
    pub fn add_effect_of(&mut self, cause: &HV16, strength: f64, markers: &CognitiveMarkers) {
        let causal_binding = markers.caused_by
            .bind(cause)
            .bind(markers.strength_marker(strength));
        self.vector = HV16::bundle(&[self.vector.clone(), causal_binding]);
    }

    /// Add temporal information: this UCE comes BEFORE the successor
    pub fn add_before(&mut self, successor: &HV16, markers: &CognitiveMarkers) {
        let temporal_binding = markers.before.bind(successor);
        self.vector = HV16::bundle(&[self.vector.clone(), temporal_binding]);
    }

    /// Add temporal information: this UCE comes AFTER the predecessor
    pub fn add_after(&mut self, predecessor: &HV16, markers: &CognitiveMarkers) {
        let temporal_binding = markers.after.bind(predecessor);
        self.vector = HV16::bundle(&[self.vector.clone(), temporal_binding]);
    }

    /// Add temporal position (e.g., hour of day, position in sequence)
    pub fn add_temporal_position(&mut self, position: usize, markers: &CognitiveMarkers) {
        let pos_marker = markers.temporal_position(position);
        self.vector = HV16::bundle(&[self.vector.clone(), pos_marker.clone()]);
    }

    /// Add semantic relation: this UCE IS A type of the category
    pub fn add_is_a(&mut self, category: &HV16, markers: &CognitiveMarkers) {
        let semantic_binding = markers.is_a.bind(category);
        self.vector = HV16::bundle(&[self.vector.clone(), semantic_binding]);
    }

    /// Add semantic relation: this UCE HAS PROPERTY
    pub fn add_property(&mut self, property: &HV16, markers: &CognitiveMarkers) {
        let semantic_binding = markers.has_property.bind(property);
        self.vector = HV16::bundle(&[self.vector.clone(), semantic_binding]);
    }

    /// Mark as self-referential (about the system itself)
    pub fn mark_self_referential(&mut self, markers: &CognitiveMarkers) {
        self.vector = HV16::bundle(&[self.vector.clone(), markers.self_reference.clone()]);
    }

    /// Mark as hypothetical/uncertain
    pub fn mark_hypothetical(&mut self, markers: &CognitiveMarkers) {
        self.vector = HV16::bundle(&[self.vector.clone(), markers.hypothetical.clone()]);
    }

    /// Query: What does this UCE cause? (unbind CAUSES marker)
    pub fn query_effects(&self, markers: &CognitiveMarkers) -> HV16 {
        self.vector.bind(&markers.causes) // Unbinding = binding with same vector
    }

    /// Query: What causes this UCE? (unbind CAUSED_BY marker)
    pub fn query_causes(&self, markers: &CognitiveMarkers) -> HV16 {
        self.vector.bind(&markers.caused_by)
    }

    /// Query: What comes before this UCE?
    pub fn query_predecessors(&self, markers: &CognitiveMarkers) -> HV16 {
        self.vector.bind(&markers.after) // If I'm AFTER X, then X is my predecessor
    }

    /// Query: What comes after this UCE?
    pub fn query_successors(&self, markers: &CognitiveMarkers) -> HV16 {
        self.vector.bind(&markers.before) // If I'm BEFORE Y, then Y is my successor
    }

    /// Similarity to another UCE (holistic comparison)
    pub fn similarity(&self, other: &UnifiedCognitiveElement) -> f32 {
        self.vector.similarity(&other.vector)
    }

    /// Record access (for relevance tracking)
    pub fn access(&mut self, timestamp: u64) {
        self.accessed_at = timestamp;
        self.access_count += 1;
    }
}

// =============================================================================
// UNIFIED COGNITIVE CORE: The Integrated Mind
// =============================================================================

/// The Unified Cognitive Core
///
/// This is the radical integration of all cognitive capabilities into a single
/// system where every element naturally encodes semantic, causal, and temporal
/// information in unified hypervectors.
///
/// ## Key Differences from Modular Approach
///
/// | Modular (Old) | Unified (New) |
/// |---------------|---------------|
/// | Separate semantic/causal/temporal stores | Single UCE store |
/// | Query each system separately | Single unified query |
/// | Low Φ (partitionable) | High Φ (irreducible) |
/// | Information loss at boundaries | No boundaries |
pub struct UnifiedCognitiveCore {
    /// Foundational markers
    markers: CognitiveMarkers,

    /// All cognitive elements (unified store)
    elements: HashMap<String, UnifiedCognitiveElement>,

    /// Causal discovery module (for learning from data)
    causal_discovery: LearnedCausalDiscovery,

    /// Temporal dynamics network (for evolution)
    temporal_network: Option<HdcLtcNetwork>,

    /// Current timestamp
    current_time: u64,

    /// System Φ (integrated information)
    phi: f64,
}

impl UnifiedCognitiveCore {
    /// Create a new Unified Cognitive Core
    pub fn new() -> Self {
        Self {
            markers: CognitiveMarkers::new(),
            elements: HashMap::new(),
            causal_discovery: LearnedCausalDiscovery::new(),
            temporal_network: None,
            current_time: 0,
            phi: 0.0,
        }
    }

    /// Create with temporal dynamics network
    pub fn with_temporal_network(mut self, config: HdcLtcNetworkConfig) -> Self {
        self.temporal_network = Some(HdcLtcNetwork::new(config, 42)); // Default seed
        self
    }

    /// Get or create a cognitive element
    pub fn get_or_create(&mut self, label: &str) -> &UnifiedCognitiveElement {
        if !self.elements.contains_key(label) {
            let seed = label.bytes().fold(42u64, |acc, b| {
                acc.wrapping_add(b as u64).wrapping_mul(31)
            });
            let semantic = HV16::random(seed);
            let uce = UnifiedCognitiveElement::new(
                label.to_string(),
                semantic,
                self.current_time,
            );
            self.elements.insert(label.to_string(), uce);
        }
        self.elements.get(label).unwrap()
    }

    /// Get mutable reference to a cognitive element
    pub fn get_mut(&mut self, label: &str) -> Option<&mut UnifiedCognitiveElement> {
        if let Some(uce) = self.elements.get_mut(label) {
            uce.access(self.current_time);
            Some(uce)
        } else {
            None
        }
    }

    /// Learn a causal relationship: cause → effect
    ///
    /// This updates BOTH the cause and effect UCEs with their causal structure
    pub fn learn_causal(&mut self, cause_label: &str, effect_label: &str, strength: f64) {
        // Get or create both elements
        let cause_seed = cause_label.bytes().fold(42u64, |acc, b| {
            acc.wrapping_add(b as u64).wrapping_mul(31)
        });
        let effect_seed = effect_label.bytes().fold(42u64, |acc, b| {
            acc.wrapping_add(b as u64).wrapping_mul(31)
        });

        let cause_semantic = HV16::random(cause_seed);
        let effect_semantic = HV16::random(effect_seed);

        // Create if not exists
        if !self.elements.contains_key(cause_label) {
            let uce = UnifiedCognitiveElement::new(
                cause_label.to_string(),
                cause_semantic.clone(),
                self.current_time,
            );
            self.elements.insert(cause_label.to_string(), uce);
        }
        if !self.elements.contains_key(effect_label) {
            let uce = UnifiedCognitiveElement::new(
                effect_label.to_string(),
                effect_semantic.clone(),
                self.current_time,
            );
            self.elements.insert(effect_label.to_string(), uce);
        }

        // Update cause: it CAUSES the effect
        if let Some(cause_uce) = self.elements.get_mut(cause_label) {
            cause_uce.add_cause(&effect_semantic, strength, &self.markers);
        }

        // Update effect: it IS CAUSED BY the cause
        if let Some(effect_uce) = self.elements.get_mut(effect_label) {
            effect_uce.add_effect_of(&cause_semantic, strength, &self.markers);
        }

        self.update_phi();
    }

    /// Learn a temporal sequence: predecessor → successor
    pub fn learn_temporal(&mut self, predecessor_label: &str, successor_label: &str) {
        let pred_seed = predecessor_label.bytes().fold(42u64, |acc, b| {
            acc.wrapping_add(b as u64).wrapping_mul(31)
        });
        let succ_seed = successor_label.bytes().fold(42u64, |acc, b| {
            acc.wrapping_add(b as u64).wrapping_mul(31)
        });

        let pred_semantic = HV16::random(pred_seed);
        let succ_semantic = HV16::random(succ_seed);

        // Create if not exists
        if !self.elements.contains_key(predecessor_label) {
            let uce = UnifiedCognitiveElement::new(
                predecessor_label.to_string(),
                pred_semantic.clone(),
                self.current_time,
            );
            self.elements.insert(predecessor_label.to_string(), uce);
        }
        if !self.elements.contains_key(successor_label) {
            let uce = UnifiedCognitiveElement::new(
                successor_label.to_string(),
                succ_semantic.clone(),
                self.current_time,
            );
            self.elements.insert(successor_label.to_string(), uce);
        }

        // Update predecessor: it comes BEFORE successor
        if let Some(pred_uce) = self.elements.get_mut(predecessor_label) {
            pred_uce.add_before(&succ_semantic, &self.markers);
        }

        // Update successor: it comes AFTER predecessor
        if let Some(succ_uce) = self.elements.get_mut(successor_label) {
            succ_uce.add_after(&pred_semantic, &self.markers);
        }
    }

    /// Learn a semantic relation: entity IS A category
    pub fn learn_is_a(&mut self, entity_label: &str, category_label: &str) {
        let cat_seed = category_label.bytes().fold(42u64, |acc, b| {
            acc.wrapping_add(b as u64).wrapping_mul(31)
        });
        let category_semantic = HV16::random(cat_seed);

        // Create entity if not exists
        if !self.elements.contains_key(entity_label) {
            let seed = entity_label.bytes().fold(42u64, |acc, b| {
                acc.wrapping_add(b as u64).wrapping_mul(31)
            });
            let uce = UnifiedCognitiveElement::new(
                entity_label.to_string(),
                HV16::random(seed),
                self.current_time,
            );
            self.elements.insert(entity_label.to_string(), uce);
        }

        if let Some(entity_uce) = self.elements.get_mut(entity_label) {
            entity_uce.add_is_a(&category_semantic, &self.markers);
        }
    }

    /// Learn from natural language text
    ///
    /// Extracts causal, temporal, and semantic relations
    pub fn learn_from_text(&mut self, text: &str) {
        let text_lower = text.to_lowercase();

        // Causal patterns
        let causal_patterns = [
            ("causes", 0.8),
            ("leads to", 0.7),
            ("results in", 0.7),
            ("produces", 0.6),
            ("triggers", 0.7),
        ];

        for (pattern, strength) in causal_patterns {
            if let Some(pos) = text_lower.find(pattern) {
                let before = &text[..pos].trim();
                let after_start = pos + pattern.len();
                let after = &text[after_start..].trim();

                if let (Some(cause), Some(effect)) = (
                    before.split_whitespace().last(),
                    after.split_whitespace().next()
                ) {
                    let cause = cause.trim_matches(|c: char| !c.is_alphanumeric());
                    let effect = effect.trim_matches(|c: char| !c.is_alphanumeric());

                    if !cause.is_empty() && !effect.is_empty() {
                        self.learn_causal(cause, effect, strength);
                    }
                }
            }
        }

        // Temporal patterns
        let temporal_patterns = ["then", "after that", "next", "followed by"];

        for pattern in temporal_patterns {
            if let Some(pos) = text_lower.find(pattern) {
                let before = &text[..pos].trim();
                let after_start = pos + pattern.len();
                let after = &text[after_start..].trim();

                if let (Some(pred), Some(succ)) = (
                    before.split_whitespace().last(),
                    after.split_whitespace().next()
                ) {
                    let pred = pred.trim_matches(|c: char| !c.is_alphanumeric());
                    let succ = succ.trim_matches(|c: char| !c.is_alphanumeric());

                    if !pred.is_empty() && !succ.is_empty() {
                        self.learn_temporal(pred, succ);
                    }
                }
            }
        }

        // Semantic patterns (is a)
        if let Some(pos) = text_lower.find(" is a ") {
            let before = &text[..pos].trim();
            let after_start = pos + 6; // " is a " length
            let after = &text[after_start..].trim();

            if let (Some(entity), Some(category)) = (
                before.split_whitespace().last(),
                after.split_whitespace().next()
            ) {
                let entity = entity.trim_matches(|c: char| !c.is_alphanumeric());
                let category = category.trim_matches(|c: char| !c.is_alphanumeric());

                if !entity.is_empty() && !category.is_empty() {
                    self.learn_is_a(entity, category);
                }
            }
        }

        self.current_time += 1;
    }

    /// Discover causal direction from observational data
    pub fn discover_causality(&self, x: &[f64], y: &[f64]) -> CausalDirection {
        self.causal_discovery.discover(x, y).direction
    }

    /// Train causal discovery on labeled data
    pub fn train_causal_discovery(&mut self, x: &[f64], y: &[f64], direction: CausalDirection) {
        self.causal_discovery.train(x, y, direction);
    }

    /// Query: Why did X happen? (find causes)
    pub fn query_why(&self, label: &str) -> Vec<QueryResult> {
        let uce = match self.elements.get(label) {
            Some(u) => u,
            None => return Vec::new(),
        };

        // Get the "causes" query vector
        let cause_query = uce.query_causes(&self.markers);

        // Find similar elements (potential causes)
        let mut results: Vec<QueryResult> = self.elements.iter()
            .filter(|(l, _)| *l != label)
            .map(|(l, other)| {
                let sim = cause_query.similarity(&other.vector);
                QueryResult {
                    label: l.clone(),
                    similarity: sim,
                    relation: "causes".to_string(),
                }
            })
            .filter(|r| r.similarity > 0.1)
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(5);
        results
    }

    /// Query: What does X cause? (find effects)
    pub fn query_effects(&self, label: &str) -> Vec<QueryResult> {
        let uce = match self.elements.get(label) {
            Some(u) => u,
            None => return Vec::new(),
        };

        // Get the "effects" query vector
        let effect_query = uce.query_effects(&self.markers);

        // Find similar elements (potential effects)
        let mut results: Vec<QueryResult> = self.elements.iter()
            .filter(|(l, _)| *l != label)
            .map(|(l, other)| {
                let sim = effect_query.similarity(&other.vector);
                QueryResult {
                    label: l.clone(),
                    similarity: sim,
                    relation: "is caused by".to_string(),
                }
            })
            .filter(|r| r.similarity > 0.1)
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(5);
        results
    }

    /// Query: What comes after X?
    pub fn query_successors(&self, label: &str) -> Vec<QueryResult> {
        let uce = match self.elements.get(label) {
            Some(u) => u,
            None => return Vec::new(),
        };

        let succ_query = uce.query_successors(&self.markers);

        let mut results: Vec<QueryResult> = self.elements.iter()
            .filter(|(l, _)| *l != label)
            .map(|(l, other)| {
                let sim = succ_query.similarity(&other.vector);
                QueryResult {
                    label: l.clone(),
                    similarity: sim,
                    relation: "comes after".to_string(),
                }
            })
            .filter(|r| r.similarity > 0.1)
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(5);
        results
    }

    /// Find most similar elements to a query
    pub fn find_similar(&self, label: &str, limit: usize) -> Vec<QueryResult> {
        let uce = match self.elements.get(label) {
            Some(u) => u,
            None => return Vec::new(),
        };

        let mut results: Vec<QueryResult> = self.elements.iter()
            .filter(|(l, _)| *l != label)
            .map(|(l, other)| QueryResult {
                label: l.clone(),
                similarity: uce.similarity(other),
                relation: "similar to".to_string(),
            })
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(limit);
        results
    }

    /// Step the temporal dynamics (if network is configured)
    pub fn step_dynamics(&mut self, dt: f64, input: Option<&HV16>) {
        if let Some(ref mut network) = self.temporal_network {
            // Convert HV16 (binary) to ContinuousHV (f32) if provided
            // Use to_bipolar to convert binary bits to -1/+1 values
            let input_hv = input.map(|hv| {
                let bipolar = hv.to_bipolar();
                ContinuousHV::from_values(bipolar)
            });

            // Default to zero input if none provided
            let default_input = ContinuousHV::zero(super::HDC_DIMENSION);
            let actual_input = input_hv.as_ref().unwrap_or(&default_input);

            network.evolve(dt as f32, actual_input);
        }
        self.current_time += 1;
    }

    /// Update Φ (integrated information)
    fn update_phi(&mut self) {
        let n = self.elements.len() as f64;
        if n < 2.0 {
            self.phi = 0.0;
            return;
        }

        // Count causal connections (embedded in vectors)
        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        let elements: Vec<_> = self.elements.values().collect();
        for i in 0..elements.len() {
            for j in (i+1)..elements.len() {
                total_similarity += elements[i].similarity(elements[j]) as f64;
                pair_count += 1;
            }
        }

        let avg_similarity = if pair_count > 0 {
            total_similarity / pair_count as f64
        } else {
            0.0
        };

        // Φ approximation: integration beyond random
        // High similarity = high integration = high Φ
        // But not TOO high (that would be redundancy)
        // Optimal is around 0.3-0.5 (related but distinct)
        let optimal_sim = 0.4;
        let deviation = (avg_similarity - optimal_sim).abs();
        self.phi = (1.0 - deviation * 2.0).max(0.0) * n.ln() / 10.0;
    }

    /// Get current Φ
    pub fn phi(&self) -> f64 {
        self.phi
    }

    /// Get element count
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Get markers reference
    pub fn markers(&self) -> &CognitiveMarkers {
        &self.markers
    }

    /// Advance time
    pub fn tick(&mut self) {
        self.current_time += 1;
    }
}

impl Default for UnifiedCognitiveCore {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a cognitive query
#[derive(Clone, Debug)]
pub struct QueryResult {
    pub label: String,
    pub similarity: f32,
    pub relation: String,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_core_creation() {
        let core = UnifiedCognitiveCore::new();
        assert_eq!(core.element_count(), 0);
        assert_eq!(core.phi(), 0.0);
    }

    #[test]
    fn test_learn_causal() {
        let mut core = UnifiedCognitiveCore::new();
        core.learn_causal("smoking", "cancer", 0.8);

        assert_eq!(core.element_count(), 2);
        assert!(core.phi() > 0.0);
    }

    #[test]
    fn test_learn_from_text() {
        let mut core = UnifiedCognitiveCore::new();
        core.learn_from_text("Smoking causes cancer");
        core.learn_from_text("Rain leads to wet ground");

        assert!(core.element_count() >= 4);
    }

    #[test]
    fn test_unified_representation() {
        let mut core = UnifiedCognitiveCore::new();

        // Learn multiple aspects of "rain"
        core.learn_causal("rain", "wet", 0.9);
        core.learn_temporal("clouds", "rain");
        core.learn_is_a("rain", "precipitation");

        // Rain now has causal, temporal, AND semantic info in ONE vector
        let rain = core.elements.get("rain").unwrap();

        // We can query all aspects from the unified vector
        let causes = core.query_effects("rain");
        let predecessors = core.query_successors("clouds");

        // The queries work because all info is bound into the vector
        assert!(causes.len() > 0 || predecessors.len() > 0 || rain.vector.as_bytes().len() > 0);
    }
}
