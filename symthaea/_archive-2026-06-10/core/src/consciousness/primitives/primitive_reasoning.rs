// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Primitive Reasoning: Basic Cognitive Operations
//!
//! Provides foundational reasoning capabilities using hyperdimensional computing.
//! Based on analogical reasoning and concept binding.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::{BinaryHV, Primitive, PrimitiveTier};

// =============================================================================
// Types for Primitive-Consciousness Integration
// =============================================================================

/// Type of transformation applied during reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransformationType {
    /// Bind two vectors (XOR in binary HV)
    Bind,
    /// Bundle multiple vectors (majority vote)
    Bundle,
    /// Permute vector for sequence encoding
    Permute,
    /// Resonance - amplify similar patterns
    Resonate,
    /// Abstract to higher level
    Abstract,
    /// Ground to lower level details
    Ground,
}

/// Type of task for primitive selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    /// Mathematical reasoning
    Mathematical,
    /// Physical simulation
    Physical,
    /// Geometric/spatial reasoning
    Geometric,
    /// Strategic planning
    Strategic,
    /// Meta-cognitive reflection
    MetaCognitive,
    /// Logical/mathematical reasoning (alias for evolution_bridge compatibility)
    Logical,
    /// Causal inference (alias for evolution_bridge compatibility)
    Causal,
    /// Spatial reasoning (alias for evolution_bridge compatibility)
    Spatial,
    /// Social/strategic reasoning (alias for evolution_bridge compatibility)
    Social,
    /// General/unknown task type
    General,
    /// Temporal reasoning
    Temporal,
    /// Generic/unknown task
    Generic,
    /// Code understanding, generation, and transformation
    Code,
}

/// Configuration for tier-aware primitive selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierAwareConfig {
    /// Weight for NSM tier
    pub nsm_weight: f64,
    /// Weight for Mathematical tier
    pub math_weight: f64,
    /// Weight for Physical tier
    pub physical_weight: f64,
    /// Weight for Geometric tier
    pub geometric_weight: f64,
    /// Weight for Strategic tier
    pub strategic_weight: f64,
    /// Weight for MetaCognitive tier
    pub metacognitive_weight: f64,
    /// Weight for Temporal tier
    pub temporal_weight: f64,
    /// Weight for Compositional tier
    pub compositional_weight: f64,
    /// Weight for Consciousness tier
    pub consciousness_weight: f64,
    /// Weight for Code tier
    pub code_weight: f64,
}

impl Default for TierAwareConfig {
    fn default() -> Self {
        Self {
            nsm_weight: 1.0,
            math_weight: 1.0,
            physical_weight: 1.0,
            geometric_weight: 1.0,
            strategic_weight: 1.0,
            metacognitive_weight: 1.0,
            temporal_weight: 1.0,
            compositional_weight: 1.0,
            consciousness_weight: 1.0,
            code_weight: 1.0,
        }
    }
}

impl TierAwareConfig {
    /// Get weight for a specific tier
    pub fn weight_for_tier(&self, tier: PrimitiveTier) -> f64 {
        match tier {
            PrimitiveTier::NSM => self.nsm_weight,
            PrimitiveTier::Mathematical => self.math_weight,
            PrimitiveTier::Physical => self.physical_weight,
            PrimitiveTier::Geometric => self.geometric_weight,
            PrimitiveTier::Strategic => self.strategic_weight,
            PrimitiveTier::MetaCognitive => self.metacognitive_weight,
            PrimitiveTier::Temporal => self.temporal_weight,
            PrimitiveTier::Compositional => self.compositional_weight,
            PrimitiveTier::Consciousness => self.consciousness_weight,
            PrimitiveTier::Code => self.code_weight,
        }
    }
}

/// A single execution step in a reasoning chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveExecution {
    /// The primitive that was executed
    pub primitive: Primitive,
    /// Input vector
    pub input: BinaryHV,
    /// Output vector
    pub output: BinaryHV,
    /// Type of transformation
    pub transformation: TransformationType,
    /// Phi contribution from this step
    pub phi_contribution: f64,
    /// Timestamp
    pub timestamp: f64,
}

/// A chain of reasoning steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChain {
    /// Initial question/query vector
    pub question: BinaryHV,
    /// Sequence of primitive executions
    pub executions: Vec<PrimitiveExecution>,
    /// Current state vector
    pub current_state: BinaryHV,
    /// Total phi accumulated
    pub total_phi: f64,
    /// Chain metadata
    pub metadata: HashMap<String, String>,
}

impl ReasoningChain {
    /// Create a new reasoning chain starting from a question
    pub fn new(question: BinaryHV) -> Self {
        Self {
            question,
            executions: Vec::new(),
            current_state: question,
            total_phi: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Execute a primitive on the current state
    pub fn execute_primitive(
        &mut self,
        primitive: &Primitive,
        transformation: TransformationType,
    ) -> Result<()> {
        let input = self.current_state;

        // Apply transformation
        let output = match transformation {
            TransformationType::Bind => input.bind(&primitive.encoding),
            TransformationType::Bundle => BinaryHV::bundle(&[input, primitive.encoding]),
            TransformationType::Permute => input.permute(1),
            TransformationType::Resonate => {
                let combined = input.bind(&primitive.encoding);
                BinaryHV::bundle(&[combined, input, primitive.encoding])
            }
            TransformationType::Abstract => {
                let elevated = input.permute(2);
                elevated.bind(&primitive.encoding)
            }
            TransformationType::Ground => {
                let grounded = input.permute(16383); // Wrap around
                grounded.bind(&primitive.encoding)
            }
        };

        // Estimate phi contribution (simplified)
        let similarity = input.similarity(&output) as f64;
        let phi_contribution = (1.0 - similarity.abs()) * 0.1;

        let execution = PrimitiveExecution {
            primitive: primitive.clone(),
            input,
            output,
            transformation,
            phi_contribution,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
        };

        self.executions.push(execution);
        self.current_state = output;
        self.total_phi += phi_contribution;

        Ok(())
    }

    /// Get the final answer vector
    pub fn answer(&self) -> &BinaryHV {
        &self.current_state
    }

    /// Get number of steps
    pub fn depth(&self) -> usize {
        self.executions.len()
    }
}

/// Statistics for a primitive-task combination
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrimitiveTaskStats {
    /// Number of times used
    pub usage_count: usize,
    /// Alias for use_count (evolution_bridge compatibility)
    pub use_count: usize,
    /// Total phi contribution
    pub total_phi: f64,
    /// Average phi contribution
    pub avg_phi: f64,
    /// Success rate
    pub success_rate: f64,
}

impl PrimitiveTaskStats {
    /// Get mean phi (alias for avg_phi, for evolution_bridge compatibility)
    pub fn mean_phi(&self) -> f64 {
        self.avg_phi
    }
}

/// Adaptive primitive selector that learns from outcomes
#[derive(Debug, Clone, Default)]
pub struct AdaptivePrimitiveSelector {
    /// Stats for each (primitive, task) pair
    stats: HashMap<(String, TaskType), PrimitiveTaskStats>,
    /// Learning rate
    #[allow(dead_code)]
    learning_rate: f64,
}

impl AdaptivePrimitiveSelector {
    /// Create a new selector
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            learning_rate: 0.1,
        }
    }

    /// Select primitives for a task with tier awareness
    pub fn select_tier_aware<'a>(
        &self,
        task: TaskType,
        primitives: &[&'a Primitive],
        count: usize,
        config: &TierAwareConfig,
        affinity: Option<&PrimitiveAffinityGraph>,
    ) -> Vec<&'a Primitive> {
        // Score each primitive
        let mut scored: Vec<(&'a Primitive, f64)> = primitives
            .iter()
            .map(|p| {
                let tier_weight = config.weight_for_tier(p.tier);
                let task_score = self
                    .stats
                    .get(&(p.name.clone(), task))
                    .map(|s| s.avg_phi)
                    .unwrap_or(0.5);
                let affinity_score = affinity
                    .map(|g| g.get_tier_affinity(p.tier, p.tier))
                    .unwrap_or(0.5);

                let score = tier_weight * task_score * affinity_score;
                (*p, score)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top N
        scored.into_iter().take(count).map(|(p, _)| p).collect()
    }

    /// Update from a completed reasoning chain
    pub fn update_from_chain(&mut self, chain: &ReasoningChain, task: TaskType) {
        for exec in &chain.executions {
            let key = (exec.primitive.name.clone(), task);
            let stats = self.stats.entry(key).or_default();

            stats.usage_count += 1;
            stats.total_phi += exec.phi_contribution;
            stats.avg_phi = stats.total_phi / stats.usage_count as f64;
        }
    }

    /// Get all stats
    pub fn all_stats(&self) -> &HashMap<(String, TaskType), PrimitiveTaskStats> {
        &self.stats
    }

    /// Record a single observation for a primitive (evolution_bridge compatibility)
    pub fn record_observation(&mut self, primitive_name: &str, task: TaskType, phi: f64) {
        let key = (primitive_name.to_string(), task);
        let stats = self.stats.entry(key).or_default();
        stats.usage_count += 1;
        stats.total_phi += phi;
        stats.avg_phi = stats.total_phi / stats.usage_count as f64;
        if phi > 0.0 {
            stats.success_rate = (stats.success_rate * (stats.usage_count - 1) as f64 + 1.0)
                / stats.usage_count as f64;
        } else {
            stats.success_rate =
                (stats.success_rate * (stats.usage_count - 1) as f64) / stats.usage_count as f64;
        }
    }
}

/// Statistics for affinity graph
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AffinityStats {
    /// Total pairs tracked
    pub total_pairs: usize,
    /// Average affinity
    pub avg_affinity: f64,
}

/// Graph of primitive affinities (how well they compose)
#[derive(Debug, Clone, Default)]
pub struct PrimitiveAffinityGraph {
    /// Affinity scores between primitive pairs
    affinities: HashMap<(String, String), f64>,
    /// Tier-level affinities
    tier_affinities: HashMap<(PrimitiveTier, PrimitiveTier), f64>,
}

impl PrimitiveAffinityGraph {
    /// Create a new affinity graph
    pub fn new() -> Self {
        let mut graph = Self::default();

        // Initialize default tier affinities
        let tiers = [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
        ];

        for &t1 in &tiers {
            for &t2 in &tiers {
                // Same tier has high affinity
                let affinity = if t1 == t2 { 0.8 } else { 0.5 };
                graph.tier_affinities.insert((t1, t2), affinity);
            }
        }

        graph
    }

    /// Record a composition result
    pub fn record_composition(&mut self, prim1: &str, prim2: &str, phi: f64) {
        let key = (prim1.to_string(), prim2.to_string());
        let current = self.affinities.get(&key).copied().unwrap_or(0.5);
        let updated = current * 0.9 + phi * 0.1;
        self.affinities.insert(key, updated);
    }

    /// Get affinity between two primitives
    pub fn get_affinity(&self, prim1: &str, prim2: &str) -> f64 {
        self.affinities
            .get(&(prim1.to_string(), prim2.to_string()))
            .copied()
            .unwrap_or(0.5)
    }

    /// Get tier-level affinity
    pub fn get_tier_affinity(&self, t1: PrimitiveTier, t2: PrimitiveTier) -> f64 {
        self.tier_affinities.get(&(t1, t2)).copied().unwrap_or(0.5)
    }

    /// Get best partners for a primitive
    pub fn best_partners(&self, primitive: &str, count: usize) -> Vec<(String, f64)> {
        let mut partners: Vec<(String, f64)> = self
            .affinities
            .iter()
            .filter_map(|((p1, p2), score)| {
                if p1 == primitive {
                    Some((p2.clone(), *score))
                } else if p2 == primitive {
                    Some((p1.clone(), *score))
                } else {
                    None
                }
            })
            .collect();

        partners.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        partners.truncate(count);
        partners
    }

    /// Get statistics
    pub fn stats(&self) -> AffinityStats {
        let total_pairs = self.affinities.len();
        let avg_affinity = if total_pairs > 0 {
            self.affinities.values().sum::<f64>() / total_pairs as f64
        } else {
            0.5
        };

        AffinityStats {
            total_pairs,
            avg_affinity,
        }
    }
}

/// Configuration for the primitive reasoner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasonerConfig {
    /// Similarity threshold for concept matching
    pub similarity_threshold: f32,
    /// Maximum reasoning depth
    pub max_depth: usize,
    /// Whether to use analogical reasoning
    pub use_analogy: bool,
    /// Cache size for memoization
    pub cache_size: usize,
}

impl Default for ReasonerConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_depth: 5,
            use_analogy: true,
            cache_size: 1000,
        }
    }
}

/// Result of a reasoning operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningResult {
    /// The conclusion reached
    pub conclusion: String,
    /// Confidence in the conclusion (0.0-1.0)
    pub confidence: f32,
    /// Chain of reasoning steps
    pub reasoning_chain: Vec<ReasoningStep>,
    /// Evidence supporting the conclusion
    pub evidence: Vec<String>,
    /// Alternative conclusions considered
    pub alternatives: Vec<(String, f32)>,
}

/// A single step in the reasoning chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// The operation performed
    pub operation: ReasoningOperation,
    /// Input concepts
    pub inputs: Vec<String>,
    /// Output concept
    pub output: String,
    /// Confidence of this step
    pub confidence: f32,
}

/// Types of reasoning operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningOperation {
    /// Bind two concepts together
    Binding,
    /// Find similarity between concepts
    Similarity,
    /// Analogical mapping (A:B :: C:?)
    Analogy,
    /// Sequence prediction
    Sequence,
    /// Hierarchical classification
    Classification,
    /// Causal inference
    Causation,
    /// Composition of concepts
    Composition,
    /// Abstraction/generalization
    Abstraction,
}

/// A concept in the reasoning system
#[derive(Debug, Clone)]
pub struct Concept {
    /// Concept name
    pub name: String,
    /// Hyperdimensional representation
    pub hv: ContinuousHV,
    /// Relations to other concepts
    pub relations: HashMap<String, f32>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl Concept {
    /// Create a new concept
    pub fn new(name: impl Into<String>, hv: ContinuousHV) -> Self {
        Self {
            name: name.into(),
            hv,
            relations: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a relation to another concept
    pub fn add_relation(&mut self, target: impl Into<String>, strength: f32) {
        self.relations.insert(target.into(), strength);
    }
}

/// The primitive reasoner
#[derive(Debug)]
pub struct PrimitiveReasoner {
    /// Configuration
    config: ReasonerConfig,
    /// Known concepts
    concepts: HashMap<String, Concept>,
    /// Reasoning cache
    cache: HashMap<String, ReasoningResult>,
    /// Statistics
    stats: ReasonerStats,
}

/// Statistics for the reasoner
#[derive(Debug, Clone, Default)]
pub struct ReasonerStats {
    /// Total reasoning operations
    pub total_operations: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Average confidence
    pub avg_confidence: f32,
    /// Deepest reasoning chain
    pub max_depth_reached: usize,
}

impl PrimitiveReasoner {
    /// Create a new reasoner
    pub fn new(config: ReasonerConfig) -> Self {
        Self {
            config,
            concepts: HashMap::new(),
            cache: HashMap::new(),
            stats: ReasonerStats::default(),
        }
    }

    /// Add a concept to the knowledge base
    pub fn add_concept(&mut self, concept: Concept) {
        self.concepts.insert(concept.name.clone(), concept);
    }

    /// Get a concept by name
    pub fn get_concept(&self, name: &str) -> Option<&Concept> {
        self.concepts.get(name)
    }

    /// Perform reasoning on a query
    pub fn reason(&mut self, query: &str, context: &[String]) -> ReasoningResult {
        self.stats.total_operations += 1;

        // Check cache
        let cache_key = format!("{}:{}", query, context.join(","));
        if let Some(cached) = self.cache.get(&cache_key) {
            self.stats.cache_hits += 1;
            return cached.clone();
        }

        // Perform actual reasoning
        let result = self.perform_reasoning(query, context);

        // Update cache
        if self.cache.len() < self.config.cache_size {
            self.cache.insert(cache_key, result.clone());
        }

        // Update statistics
        let n = self.stats.total_operations as f32;
        self.stats.avg_confidence = (self.stats.avg_confidence * (n - 1.0) + result.confidence) / n;
        self.stats.max_depth_reached = self
            .stats
            .max_depth_reached
            .max(result.reasoning_chain.len());

        result
    }

    /// Internal reasoning implementation
    fn perform_reasoning(&self, query: &str, context: &[String]) -> ReasoningResult {
        let mut chain = Vec::new();
        let mut confidence = 1.0f32;
        let mut alternatives = Vec::new();

        // Parse query and identify concepts
        let query_concepts: Vec<_> = query
            .split_whitespace()
            .filter(|w| self.concepts.contains_key(*w))
            .collect();

        // Build reasoning chain
        for (i, concept_name) in query_concepts.iter().enumerate() {
            if i == 0 {
                continue;
            }

            let prev = query_concepts[i - 1];
            let curr = *concept_name;

            // Try to find relation
            if let Some(concept) = self.concepts.get(prev) {
                if let Some(&rel_strength) = concept.relations.get(curr) {
                    chain.push(ReasoningStep {
                        operation: ReasoningOperation::Binding,
                        inputs: vec![prev.to_string(), curr.to_string()],
                        output: format!("{prev}→{curr}"),
                        confidence: rel_strength,
                    });
                    confidence *= rel_strength;
                }
            }

            // Try analogical reasoning if enabled
            if self.config.use_analogy && chain.is_empty() {
                if let Some(analogy_result) = self.try_analogy(prev, curr) {
                    chain.push(analogy_result.0);
                    confidence = analogy_result.1;
                }
            }
        }

        // Include context
        let evidence: Vec<String> = context
            .iter()
            .filter(|c| query_concepts.iter().any(|qc| c.contains(qc)))
            .cloned()
            .collect();

        // Generate alternatives based on similar concepts
        for concept_name in &query_concepts {
            if let Some(concept) = self.concepts.get(*concept_name) {
                for (related, strength) in &concept.relations {
                    if *strength > 0.5 && *strength < confidence {
                        alternatives.push((related.clone(), *strength));
                    }
                }
            }
        }

        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        alternatives.truncate(5);

        ReasoningResult {
            conclusion: query.to_string(),
            confidence,
            reasoning_chain: chain,
            evidence,
            alternatives,
        }
    }

    /// Try analogical reasoning (A:B :: C:?)
    fn try_analogy(&self, a: &str, c: &str) -> Option<(ReasoningStep, f32)> {
        let concept_a = self.concepts.get(a)?;
        let concept_c = self.concepts.get(c)?;

        // Find the best matching relation
        let mut best_match: Option<(String, f32)> = None;

        for strength_ab in concept_a.relations.values() {
            // Look for similar relation from C
            for (d, strength_cd) in &concept_c.relations {
                let similarity = (strength_ab - strength_cd).abs();
                if similarity < 0.2 {
                    let confidence = 1.0 - similarity;
                    if best_match.as_ref().is_none_or(|(_, c)| confidence > *c) {
                        best_match = Some((d.clone(), confidence));
                    }
                }
            }
        }

        best_match.map(|(d, conf)| {
            (
                ReasoningStep {
                    operation: ReasoningOperation::Analogy,
                    inputs: vec![a.to_string(), c.to_string()],
                    output: d,
                    confidence: conf,
                },
                conf,
            )
        })
    }

    /// Bind two concepts together
    pub fn bind(&mut self, a: &str, b: &str) -> Option<ContinuousHV> {
        let concept_a = self.concepts.get(a)?;
        let concept_b = self.concepts.get(b)?;

        // Bind hypervectors
        let bound = concept_a.hv.bind(&concept_b.hv);
        Some(bound)
    }

    /// Find similar concepts
    pub fn find_similar(&self, query_hv: &ContinuousHV, top_k: usize) -> Vec<(String, f32)> {
        let mut similarities: Vec<(String, f32)> = self
            .concepts
            .iter()
            .map(|(name, concept)| {
                let sim = query_hv.similarity(&concept.hv);
                (name.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    /// Get statistics
    pub fn stats(&self) -> &ReasonerStats {
        &self.stats
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for PrimitiveReasoner {
    fn default() -> Self {
        Self::new(ReasonerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoner_creation() {
        let reasoner = PrimitiveReasoner::default();
        assert!(reasoner.concepts.is_empty());
    }

    #[test]
    fn test_concept_creation() {
        let hv = ContinuousHV::random(512, 42);
        let mut concept = Concept::new("test", hv);
        concept.add_relation("other", 0.8);
        assert_eq!(concept.relations.get("other"), Some(&0.8));
    }

    #[test]
    fn test_basic_reasoning() {
        let mut reasoner = PrimitiveReasoner::default();

        // Add some concepts
        let hv1 = ContinuousHV::random(512, 42);
        let mut c1 = Concept::new("animal", hv1);
        c1.add_relation("dog", 0.9);
        reasoner.add_concept(c1);

        let hv2 = ContinuousHV::random(512, 42);
        let c2 = Concept::new("dog", hv2);
        reasoner.add_concept(c2);

        // Perform reasoning
        let result = reasoner.reason("animal dog", &[]);
        assert!(result.confidence > 0.0);
    }
}
