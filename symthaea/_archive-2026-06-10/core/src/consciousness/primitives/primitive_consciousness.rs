// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Primitive-Consciousness Bridge
//!
//! Connects the 9-tier primitive system to consciousness processing.
//!
//! ## Purpose
//!
//! This module bridges:
//! - HDC primitive encodings ↔ consciousness states
//! - Primitive reasoning ↔ phenomenal binding
//! - Primitive selection ↔ attention allocation
//! - Primitive affinity ↔ causal structure
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                 PRIMITIVE-CONSCIOUSNESS BRIDGE                       │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  ┌─────────────────┐           ┌─────────────────┐                 │
//! │  │  Primitive      │           │ Consciousness   │                 │
//! │  │  System (HDC)   │◄─────────►│ Graph           │                 │
//! │  │  9 Tiers        │           │ (States)        │                 │
//! │  └────────┬────────┘           └────────┬────────┘                 │
//! │           │                             │                          │
//! │           ▼                             ▼                          │
//! │  ┌─────────────────────────────────────────────────┐               │
//! │  │          PrimitiveConsciousnessState            │               │
//! │  │  - Active primitives                             │               │
//! │  │  - Consciousness level                           │               │
//! │  │  - Binding structure                             │               │
//! │  │  - Causal trace                                  │               │
//! │  └─────────────────────────────────────────────────┘               │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use crate::consciousness::primitive_reasoning::{
    AdaptivePrimitiveSelector, PrimitiveAffinityGraph, ReasoningChain, TaskType, TierAwareConfig,
    TransformationType,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::primitive_system::{Primitive, PrimitiveSystem, PrimitiveTier};

// =============================================================================
// PRIMITIVE CONSCIOUSNESS STATE
// =============================================================================

/// A consciousness state expressed in primitive space
///
/// This represents a moment of conscious experience decomposed into
/// the active primitives from each tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveConsciousnessState {
    /// Active primitives organized by tier
    pub active_by_tier: std::collections::HashMap<PrimitiveTier, Vec<ActivePrimitive>>,

    /// Combined encoding of all active primitives
    pub unified_encoding: Option<BinaryHV>,

    /// Overall consciousness level (Φ)
    pub phi: f64,

    /// Timestamp
    pub timestamp: f64,

    /// Binding structure between primitives
    pub bindings: Vec<PrimitiveBinding>,

    /// Causal attribution for this state
    pub causal_source: Option<String>,
}

/// A primitive that is currently active in consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivePrimitive {
    /// The primitive
    pub primitive: Primitive,

    /// Activation strength (0.0 - 1.0)
    pub activation: f64,

    /// Why this primitive is active
    pub activation_reason: ActivationReason,

    /// Duration in consciousness (timesteps)
    pub duration: usize,
}

/// Why a primitive became active
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationReason {
    /// Bottom-up: activated by sensory/semantic input
    BottomUp { input_similarity: f64 },

    /// Top-down: activated by attention/goal
    TopDown { goal: String },

    /// Lateral: activated by another primitive
    Lateral {
        source_primitive: String,
        affinity: f64,
    },

    /// Sustained: maintained from previous state
    Sustained {
        original_reason: Box<ActivationReason>,
    },
}

/// Binding between two primitives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveBinding {
    /// First primitive
    pub prim1: String,

    /// Second primitive
    pub prim2: String,

    /// Binding type (from TransformationType)
    pub binding_type: TransformationType,

    /// Binding strength
    pub strength: f64,

    /// Φ contribution from this binding
    pub phi_contribution: f64,
}

impl PrimitiveConsciousnessState {
    /// Create an empty consciousness state
    pub fn new(timestamp: f64) -> Self {
        Self {
            active_by_tier: std::collections::HashMap::new(),
            unified_encoding: None,
            phi: 0.0,
            timestamp,
            bindings: Vec::new(),
            causal_source: None,
        }
    }

    /// Get all active primitives across all tiers
    pub fn all_active(&self) -> Vec<&ActivePrimitive> {
        self.active_by_tier.values().flatten().collect()
    }

    /// Get active primitives for a specific tier
    pub fn tier_active(&self, tier: PrimitiveTier) -> Vec<&ActivePrimitive> {
        self.active_by_tier
            .get(&tier)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Add an active primitive
    pub fn activate(&mut self, primitive: Primitive, activation: f64, reason: ActivationReason) {
        let tier = primitive.tier;
        let active = ActivePrimitive {
            primitive,
            activation,
            activation_reason: reason,
            duration: 0,
        };

        self.active_by_tier.entry(tier).or_default().push(active);
    }

    /// Compute unified encoding from all active primitives
    pub fn compute_unified_encoding(&mut self) {
        let actives = self.all_active();
        if actives.is_empty() {
            self.unified_encoding = None;
            return;
        }

        // Collect all encodings for bundling
        let encodings: Vec<BinaryHV> = actives.iter().map(|a| a.primitive.encoding).collect();

        // Bundle all active primitives
        let unified = BinaryHV::bundle(&encodings);

        self.unified_encoding = Some(unified);
    }

    /// Get the dominant tier (most active primitives)
    pub fn dominant_tier(&self) -> Option<PrimitiveTier> {
        self.active_by_tier
            .iter()
            .max_by_key(|(_, v)| v.len())
            .map(|(tier, _)| *tier)
    }
}

// =============================================================================
// CONSCIOUSNESS-PRIMITIVE PROCESSOR
// =============================================================================

/// Processes consciousness states using the primitive system
#[derive(Debug)]
pub struct ConsciousnessPrimitiveProcessor {
    /// The primitive system
    primitive_system: PrimitiveSystem,

    /// Adaptive selector for primitive selection
    selector: AdaptivePrimitiveSelector,

    /// Affinity graph for composition
    affinity_graph: PrimitiveAffinityGraph,

    /// Tier-aware selection config
    tier_config: TierAwareConfig,

    /// Current state
    current_state: Option<PrimitiveConsciousnessState>,

    /// State history
    history: Vec<PrimitiveConsciousnessState>,

    /// History limit
    history_size: usize,
}

impl ConsciousnessPrimitiveProcessor {
    /// Create a new processor
    pub fn new() -> Self {
        Self {
            primitive_system: PrimitiveSystem::new(),
            selector: AdaptivePrimitiveSelector::new(),
            affinity_graph: PrimitiveAffinityGraph::new(),
            tier_config: TierAwareConfig::default(),
            current_state: None,
            history: Vec::new(),
            history_size: 100,
        }
    }

    /// Borrow the underlying primitive system (needed by PrimitiveLattice construction).
    pub fn primitive_system(&self) -> &PrimitiveSystem {
        &self.primitive_system
    }

    /// Get all primitives from the system
    fn get_all_primitives(&self) -> Vec<&Primitive> {
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

        tiers
            .iter()
            .flat_map(|tier| self.primitive_system.get_tier(*tier))
            .collect()
    }

    /// Process semantic input to activate primitives
    pub fn process_input(
        &mut self,
        input: &BinaryHV,
        timestamp: f64,
    ) -> PrimitiveConsciousnessState {
        let mut state = PrimitiveConsciousnessState::new(timestamp);

        // Find primitives that resonate with input
        let primitives = self.get_all_primitives();

        for prim in primitives {
            let similarity = input.similarity(&prim.encoding);

            // Activate if similarity exceeds threshold
            if similarity > 0.3 {
                state.activate(
                    prim.clone(),
                    similarity as f64,
                    ActivationReason::BottomUp {
                        input_similarity: similarity as f64,
                    },
                );
            }
        }

        // Compute unified encoding
        state.compute_unified_encoding();

        // Compute phi (simplified - actual phi computation more complex)
        state.phi = self.estimate_phi(&state);

        // Store in history
        self.update_history(state.clone());
        self.current_state = Some(state.clone());

        state
    }

    /// Select primitives for a task
    pub fn select_for_task(&self, task: TaskType, count: usize) -> Vec<&Primitive> {
        let primitives = self.get_all_primitives();
        let prim_refs: Vec<&Primitive> = primitives.into_iter().collect();

        self.selector.select_tier_aware(
            task,
            &prim_refs,
            count,
            &self.tier_config,
            Some(&self.affinity_graph),
        )
    }

    /// Create a reasoning chain for a question
    pub fn reason(
        &self,
        question: BinaryHV,
        task: TaskType,
        depth: usize,
    ) -> Result<ReasoningChain> {
        let mut chain = ReasoningChain::new(question);

        // Select primitives for task
        let primitives = self.select_for_task(task, depth);

        // Execute each primitive in sequence
        for prim in primitives {
            chain.execute_primitive(prim, TransformationType::Bind)?;
        }

        Ok(chain)
    }

    /// Update from a reasoning chain (learn from outcomes)
    pub fn update_from_chain(&mut self, chain: &ReasoningChain, task: TaskType) {
        self.selector.update_from_chain(chain, task);

        // Also update affinity graph from sequential bindings
        let executions = &chain.executions;
        for i in 0..executions.len().saturating_sub(1) {
            let phi = executions[i].phi_contribution + executions[i + 1].phi_contribution;
            self.affinity_graph.record_composition(
                &executions[i].primitive.name,
                &executions[i + 1].primitive.name,
                phi,
            );
        }
    }

    /// Get current consciousness state
    pub fn current(&self) -> Option<&PrimitiveConsciousnessState> {
        self.current_state.as_ref()
    }

    /// Get state history
    pub fn history(&self) -> &[PrimitiveConsciousnessState] {
        &self.history
    }

    /// Estimate phi for a state using real Phi computation
    fn estimate_phi(&self, state: &PrimitiveConsciousnessState) -> f64 {
        use symthaea_core::hdc::unified_hv::ContinuousHV;
        use symthaea_core::phi_engine::{PhiEngine, PhiMethod};

        // Convert active primitive encodings to node representations
        let active = state.all_active();
        if active.is_empty() {
            return 0.0;
        }

        // Use up to 8 active primitives as nodes
        let mut nodes = Vec::new();
        for prim in active.iter().take(8) {
            // Convert encoding to continuous representation
            let bits: Vec<f32> = (0..16)
                .map(|i| {
                    if prim
                        .primitive
                        .encoding
                        .get_bit(i % crate::hdc::HDC_DIMENSION)
                        != 0
                    {
                        1.0f32
                    } else {
                        -1.0f32
                    }
                })
                .collect();
            nodes.push(ContinuousHV::from_vec(bits));
        }

        if nodes.len() < 2 {
            // Need at least 2 nodes for meaningful Phi
            let active_count = active.len();
            return (active_count as f64).sqrt() * 0.1;
        }

        let engine = PhiEngine::new(PhiMethod::Auto);
        let result = engine.compute(&nodes);
        result.phi.min(1.0)
    }

    /// Update history with new state
    fn update_history(&mut self, state: PrimitiveConsciousnessState) {
        self.history.push(state);
        if self.history.len() > self.history_size {
            self.history.remove(0);
        }
    }

    /// Get primitive statistics
    pub fn primitive_stats(&self) -> ProcessorStats {
        let total_primitives = self.get_all_primitives().len();
        let selector_stats = self.selector.all_stats().len();
        let affinity_pairs = self.affinity_graph.stats().total_pairs;

        ProcessorStats {
            total_primitives,
            tracked_primitive_task_pairs: selector_stats,
            learned_affinities: affinity_pairs,
            history_length: self.history.len(),
        }
    }
}

impl Default for ConsciousnessPrimitiveProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorStats {
    pub total_primitives: usize,
    pub tracked_primitive_task_pairs: usize,
    pub learned_affinities: usize,
    pub history_length: usize,
}

// =============================================================================
// CONSCIOUSNESS STATE DECOMPOSITION
// =============================================================================

/// Decomposes consciousness states into primitive representations
pub struct ConsciousnessDecomposer {
    primitive_system: PrimitiveSystem,
}

impl ConsciousnessDecomposer {
    pub fn new() -> Self {
        Self {
            primitive_system: PrimitiveSystem::new(),
        }
    }

    /// Decompose a semantic vector into active primitives
    pub fn decompose(&self, semantic: &[f32]) -> Vec<(Primitive, f64)> {
        // Convert semantic to BinaryHV for comparison
        let hv = self.semantic_to_hv(semantic);

        let mut activations = Vec::new();
        for prim in self.primitive_system.all_primitives() {
            let similarity = hv.similarity(&prim.encoding) as f64;
            if similarity > 0.2 {
                activations.push((prim.clone(), similarity));
            }
        }

        // Sort by activation strength
        activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        activations
    }

    /// Find dominant tier for a semantic representation
    pub fn dominant_tier(&self, semantic: &[f32]) -> Option<PrimitiveTier> {
        let activations = self.decompose(semantic);

        let mut tier_scores: std::collections::HashMap<PrimitiveTier, f64> =
            std::collections::HashMap::new();

        for (prim, score) in activations {
            *tier_scores.entry(prim.tier).or_insert(0.0) += score;
        }

        tier_scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(tier, _)| tier)
    }

    /// Convert semantic vector to BinaryHV
    fn semantic_to_hv(&self, semantic: &[f32]) -> BinaryHV {
        // Use semantic hash as seed for reproducible conversion
        let seed: u64 = semantic
            .iter()
            .enumerate()
            .map(|(i, &v)| (v.to_bits() as u64).wrapping_mul(i as u64 + 1))
            .fold(0u64, |acc, x| acc.wrapping_add(x));

        BinaryHV::random(seed)
    }
}

impl Default for ConsciousnessDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// PHENOMENAL BINDING USING PRIMITIVES
// =============================================================================

/// Binds conscious content using primitive operations
#[derive(Debug)]
pub struct PrimitiveBindingEngine {
    affinity_graph: PrimitiveAffinityGraph,
}

impl PrimitiveBindingEngine {
    pub fn new() -> Self {
        Self {
            affinity_graph: PrimitiveAffinityGraph::new(),
        }
    }

    /// Bind two primitives together
    pub fn bind(
        &mut self,
        prim1: &Primitive,
        prim2: &Primitive,
        binding_type: TransformationType,
    ) -> PrimitiveBinding {
        let result = match binding_type {
            TransformationType::Bind => prim1.encoding.bind(&prim2.encoding),
            TransformationType::Bundle => BinaryHV::bundle(&[prim1.encoding, prim2.encoding]),
            TransformationType::Permute => prim1.encoding.permute(1),
            TransformationType::Resonate => {
                // Resonance amplifies similar patterns
                let combined = prim1.encoding.bind(&prim2.encoding);
                BinaryHV::bundle(&[combined, prim1.encoding, prim2.encoding])
            }
            TransformationType::Abstract => {
                // Abstract to higher-level - bundle with permutation
                let elevated = prim1.encoding.permute(2);
                elevated.bind(&prim2.encoding)
            }
            TransformationType::Ground => {
                // Ground to lower-level details - inverse permutation
                let grounded = prim1.encoding.permute(-1i32 as usize); // Wrap around
                grounded.bind(&prim2.encoding)
            }
        };

        // Estimate phi contribution from binding
        let phi = self.estimate_binding_phi(prim1, prim2, &result);

        // Record in affinity graph
        self.affinity_graph
            .record_composition(&prim1.name, &prim2.name, phi);

        PrimitiveBinding {
            prim1: prim1.name.clone(),
            prim2: prim2.name.clone(),
            binding_type,
            strength: prim1.encoding.similarity(&prim2.encoding) as f64,
            phi_contribution: phi,
        }
    }

    /// Suggest optimal binding partner for a primitive
    pub fn suggest_partner(&self, primitive: &str, count: usize) -> Vec<(String, f64)> {
        self.affinity_graph.best_partners(primitive, count)
    }

    /// Estimate phi contribution from binding
    fn estimate_binding_phi(&self, prim1: &Primitive, prim2: &Primitive, result: &BinaryHV) -> f64 {
        // Tier affinity contributes to phi
        let tier_affinity = self
            .affinity_graph
            .get_tier_affinity(prim1.tier, prim2.tier);

        // Orthogonality contributes to information integration
        let similarity = prim1.encoding.similarity(&prim2.encoding) as f64;
        let orthogonality = 1.0 - similarity.abs();

        // Result distinctiveness
        let result_distinctiveness = 1.0 - result.similarity(&prim1.encoding).abs() as f64;

        // Combine factors
        (tier_affinity * 0.3 + orthogonality * 0.4 + result_distinctiveness * 0.3).min(1.0)
    }
}

impl Default for PrimitiveBindingEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_state_creation() {
        let state = PrimitiveConsciousnessState::new(0.0);
        assert!(state.all_active().is_empty());
        assert_eq!(state.phi, 0.0);
    }

    #[test]
    fn test_primitive_activation() {
        let mut state = PrimitiveConsciousnessState::new(0.0);

        let prim = Primitive {
            name: "TEST".to_string(),
            encoding: BinaryHV::random(42),
            tier: PrimitiveTier::Physical,
            domain: "test".to_string(),
            definition: "Test".to_string(),
            is_base: true,
            derivation: None,
        };

        state.activate(
            prim,
            0.8,
            ActivationReason::BottomUp {
                input_similarity: 0.8,
            },
        );

        assert_eq!(state.all_active().len(), 1);
        assert_eq!(state.tier_active(PrimitiveTier::Physical).len(), 1);
    }

    #[test]
    fn test_processor_creation() {
        let processor = ConsciousnessPrimitiveProcessor::new();
        let stats = processor.primitive_stats();
        assert!(stats.total_primitives > 0);
    }

    #[test]
    fn test_processor_input_processing() {
        let mut processor = ConsciousnessPrimitiveProcessor::new();
        let input = BinaryHV::random(42);

        let state = processor.process_input(&input, 0.0);
        assert!(state.phi >= 0.0);
    }

    #[test]
    fn test_decomposer() {
        let decomposer = ConsciousnessDecomposer::new();
        let semantic = vec![0.1; 64];

        let activations = decomposer.decompose(&semantic);
        // Decomposition should return valid (Primitive, f64) pairs
        for (_prim, score) in &activations {
            assert!(
                score.is_finite(),
                "Activation score must be finite: {score}"
            );
        }
    }

    #[test]
    fn test_binding_engine() {
        let mut engine = PrimitiveBindingEngine::new();

        let prim1 = Primitive {
            name: "A".to_string(),
            encoding: BinaryHV::random(1),
            tier: PrimitiveTier::Physical,
            domain: "test".to_string(),
            definition: "A".to_string(),
            is_base: true,
            derivation: None,
        };

        let prim2 = Primitive {
            name: "B".to_string(),
            encoding: BinaryHV::random(2),
            tier: PrimitiveTier::Physical,
            domain: "test".to_string(),
            definition: "B".to_string(),
            is_base: true,
            derivation: None,
        };

        let binding = engine.bind(&prim1, &prim2, TransformationType::Bind);
        assert!(binding.phi_contribution >= 0.0);
        assert!(binding.phi_contribution <= 1.0);
    }
}
