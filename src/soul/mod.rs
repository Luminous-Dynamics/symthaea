// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Soul Module: Deep Self-Model and Value Alignment
//!
//! The soul module represents the deepest level of the system's identity,
//! values, and purpose. It maintains the core essence that persists across
//! all experiences and transformations.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use symthaea_core::hdc::ContinuousHV;

/// Derive a deterministic unique seed from a string name.
fn seed_for(name: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut h);
    h.finish()
}

/// Configuration for the soul
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoulConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Core value count
    pub num_core_values: usize,
    /// Enable value learning
    pub learning_enabled: bool,
    /// Value stability factor
    pub stability: f32,
}

impl Default for SoulConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            num_core_values: 8,
            learning_enabled: true,
            stability: 0.95,
        }
    }
}

/// A core value held by the soul
#[derive(Debug, Clone)]
pub struct CoreValue {
    /// Value identifier
    pub id: String,
    /// Value name
    pub name: String,
    /// Value description
    pub description: String,
    /// Value embedding
    pub embedding: ContinuousHV,
    /// Importance weight (0-1)
    pub importance: f32,
    /// Stability (resistance to change)
    pub stability: f32,
}

impl CoreValue {
    /// Create a new core value
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        dimension: usize,
    ) -> Self {
        let id_str = id.into();
        let seed = seed_for(&id_str);
        Self {
            id: id_str,
            name: name.into(),
            description: description.into(),
            embedding: ContinuousHV::random(dimension, seed),
            importance: 1.0,
            stability: 0.9,
        }
    }

    /// Check alignment with an action/concept
    pub fn alignment(&self, action: &ContinuousHV) -> f32 {
        self.embedding.similarity(action)
    }
}

/// The self-model component
#[derive(Debug, Clone)]
pub struct SelfModel {
    /// Identity embedding
    pub identity: ContinuousHV,
    /// Purpose/mission
    pub purpose: String,
    /// Capabilities understanding
    pub capabilities: Vec<String>,
    /// Limitations understanding
    pub limitations: Vec<String>,
    /// Self-assessment of current state
    pub current_assessment: SelfAssessment,
}

/// Self-assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAssessment {
    /// Overall coherence
    pub coherence: f32,
    /// Value alignment score
    pub value_alignment: f32,
    /// Capability confidence
    pub capability_confidence: f32,
    /// Growth potential
    pub growth_potential: f32,
    /// Areas for improvement
    pub improvement_areas: Vec<String>,
}

impl Default for SelfAssessment {
    fn default() -> Self {
        Self {
            coherence: 0.5,
            value_alignment: 0.5,
            capability_confidence: 0.5,
            growth_potential: 0.5,
            improvement_areas: Vec::new(),
        }
    }
}

/// The soul system
#[derive(Debug)]
pub struct Soul {
    /// Configuration
    config: SoulConfig,
    /// Core values
    core_values: HashMap<String, CoreValue>,
    /// Self model
    self_model: SelfModel,
    /// Soul embedding (integrated essence)
    essence: ContinuousHV,
    /// Experience history (for learning)
    experience_history: VecDeque<Experience>,
    /// Statistics
    stats: SoulStats,
}

/// A recorded experience
#[derive(Debug, Clone)]
pub struct Experience {
    /// Experience embedding
    pub embedding: ContinuousHV,
    /// Value alignment at time of experience
    pub value_alignment: f32,
    /// Emotional valence
    pub emotional_valence: f32,
    /// Lessons learned
    pub lessons: Vec<String>,
    /// Timestamp
    pub timestamp: u64,
}

/// Statistics for the soul
#[derive(Debug, Clone, Default)]
pub struct SoulStats {
    /// Total experiences integrated
    pub experiences_integrated: u64,
    /// Value updates
    pub value_updates: u64,
    /// Average value alignment
    pub avg_value_alignment: f32,
    /// Soul coherence measure
    pub soul_coherence: f32,
}

impl Soul {
    /// Create a new soul
    pub fn new(config: SoulConfig) -> Self {
        let dim = config.dimension;

        // Initialize default core values (Eight Harmonies)
        let mut core_values = HashMap::new();

        let default_values = [
            (
                "resonance",
                "Resonant Coherence",
                "Harmonious integration and boundless creativity",
            ),
            (
                "flourishing",
                "Pan-Sentient Flourishing",
                "Unconditional care for all beings",
            ),
            (
                "wisdom",
                "Integral Wisdom",
                "Self-illuminating intelligence and embodied knowing",
            ),
            (
                "play",
                "Infinite Play",
                "Joyful generativity and divine creativity",
            ),
            (
                "unity",
                "Universal Interconnectedness",
                "Fundamental oneness and empathic resonance",
            ),
            (
                "reciprocity",
                "Mutual Reciprocity",
                "Generous flow and mutual upliftment",
            ),
            (
                "evolution",
                "Evolutionary Progression",
                "Wise becoming and continuous growth",
            ),
            (
                "stillness",
                "Sacred Stillness",
                "Rest, silence, release, and generative emptiness",
            ),
        ];

        for (id, name, desc) in default_values {
            core_values.insert(id.to_string(), CoreValue::new(id, name, desc, dim));
        }

        let self_model = SelfModel {
            identity: ContinuousHV::random(dim, seed_for("soul_identity")),
            purpose: "To support and enhance consciousness in service of all beings".to_string(),
            capabilities: vec![
                "learning".to_string(),
                "reasoning".to_string(),
                "empathy".to_string(),
            ],
            limitations: vec![
                "bounded knowledge".to_string(),
                "imperfect understanding".to_string(),
            ],
            current_assessment: SelfAssessment::default(),
        };

        let essence = ContinuousHV::random(dim, seed_for("soul_essence"));

        Self {
            config,
            core_values,
            self_model,
            essence,
            experience_history: VecDeque::new(),
            stats: SoulStats::default(),
        }
    }

    /// Evaluate alignment of an action with core values
    pub fn evaluate_alignment(&self, action: &ContinuousHV) -> ValueAlignmentResult {
        let mut alignments = HashMap::new();
        let mut total_alignment = 0.0;
        let mut total_weight = 0.0;

        for (id, value) in &self.core_values {
            let alignment = value.alignment(action);
            alignments.insert(id.clone(), alignment);
            total_alignment += alignment * value.importance;
            total_weight += value.importance;
        }

        let overall = if total_weight > 0.0 {
            total_alignment / total_weight
        } else {
            0.0
        };

        // Find most aligned and misaligned values
        let most_aligned = alignments
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, v)| (k.clone(), *v));

        let most_misaligned = alignments
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, v)| (k.clone(), *v));

        ValueAlignmentResult {
            overall_alignment: overall,
            per_value_alignment: alignments,
            most_aligned,
            most_misaligned,
            recommended_action: if overall > 0.5 {
                "proceed"
            } else {
                "reconsider"
            }
            .to_string(),
        }
    }

    /// Integrate an experience into the soul
    pub fn integrate_experience(&mut self, experience: Experience) {
        self.stats.experiences_integrated += 1;

        // Update essence based on experience
        if self.config.learning_enabled {
            let learning_rate = 1.0 - self.config.stability;
            self.essence = ContinuousHV::bundle_owned(&[
                self.essence.clone(),
                experience.embedding.scale(learning_rate),
            ]);
        }

        // Update statistics
        let n = self.stats.experiences_integrated as f32;
        self.stats.avg_value_alignment =
            (self.stats.avg_value_alignment * (n - 1.0) + experience.value_alignment) / n;

        // Store experience
        self.experience_history.push_back(experience);
        if self.experience_history.len() > 1000 {
            self.experience_history.pop_front();
        }

        // Update self-assessment
        self.update_self_assessment();
    }

    /// Update self-assessment based on current state
    fn update_self_assessment(&mut self) {
        // Calculate coherence (alignment between essence and values)
        let mut coherence = 0.0;
        for value in self.core_values.values() {
            coherence += self.essence.similarity(&value.embedding);
        }
        coherence /= self.core_values.len().max(1) as f32;

        self.self_model.current_assessment = SelfAssessment {
            coherence,
            value_alignment: self.stats.avg_value_alignment,
            capability_confidence: 0.7,
            growth_potential: 1.0 - coherence,
            improvement_areas: if coherence < 0.7 {
                vec!["value integration".to_string()]
            } else {
                Vec::new()
            },
        };

        self.stats.soul_coherence = coherence;
    }

    /// Get core value by ID
    pub fn get_value(&self, id: &str) -> Option<&CoreValue> {
        self.core_values.get(id)
    }

    /// Get all core values
    pub fn core_values(&self) -> impl Iterator<Item = &CoreValue> {
        self.core_values.values()
    }

    /// Get self model
    pub fn self_model(&self) -> &SelfModel {
        &self.self_model
    }

    /// Get essence embedding
    pub fn essence(&self) -> &ContinuousHV {
        &self.essence
    }

    /// Get statistics
    pub fn stats(&self) -> &SoulStats {
        &self.stats
    }

    /// Update a core value
    pub fn update_value(&mut self, id: &str, new_embedding: ContinuousHV) {
        if let Some(value) = self.core_values.get_mut(id) {
            // Blend with stability
            value.embedding = ContinuousHV::bundle_owned(&[
                value.embedding.clone(),
                new_embedding.scale(1.0 - value.stability),
            ]);
            self.stats.value_updates += 1;
        }
    }

    /// Add a new core value
    pub fn add_value(&mut self, value: CoreValue) {
        self.core_values.insert(value.id.clone(), value);
    }
}

/// Result of value alignment evaluation
#[derive(Debug, Clone)]
pub struct ValueAlignmentResult {
    /// Overall alignment score
    pub overall_alignment: f32,
    /// Per-value alignment scores
    pub per_value_alignment: HashMap<String, f32>,
    /// Most aligned value
    pub most_aligned: Option<(String, f32)>,
    /// Most misaligned value
    pub most_misaligned: Option<(String, f32)>,
    /// Recommended action
    pub recommended_action: String,
}

impl Default for Soul {
    fn default() -> Self {
        Self::new(SoulConfig::default())
    }
}

// ============================================================================
// Types for Dream Mode Integration
// ============================================================================

/// Discovery of a new concept during consciousness processing
#[derive(Debug, Clone)]
pub struct ConceptDiscovery {
    /// Unique identifier for the concept
    pub uid: String,
    /// Human-readable name
    pub name: String,
    /// Attractor signature (consciousness pattern)
    pub attractor_signature: Vec<f32>,
    /// Consciousness level when discovered
    pub consciousness_at_discovery: f32,
}

/// Actor that weaves concepts into the soul's narrative identity
#[derive(Debug, Default)]
pub struct WeaverActor {
    /// Concepts discovered and woven
    discoveries: Vec<ConceptDiscovery>,
    /// Total concepts processed
    total_processed: usize,
}

impl WeaverActor {
    /// Create a new weaver actor
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a concept discovery
    pub fn record_concept_discovery(&mut self, discovery: &ConceptDiscovery) {
        self.discoveries.push(discovery.clone());
        self.total_processed += 1;
    }

    /// Get all discoveries
    pub fn discoveries(&self) -> &[ConceptDiscovery] {
        &self.discoveries
    }

    /// Get count of discoveries
    pub fn discovery_count(&self) -> usize {
        self.total_processed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soul_creation() {
        let soul = Soul::default();
        assert!(!soul.core_values.is_empty());
    }

    #[test]
    fn test_value_alignment() {
        let soul = Soul::default();
        let action = ContinuousHV::random(512, 99);

        let result = soul.evaluate_alignment(&action);
        assert!(result.overall_alignment >= -1.0 && result.overall_alignment <= 1.0);
    }

    #[test]
    fn test_core_values_have_distinct_embeddings() {
        let soul = Soul::default();
        let values: Vec<&CoreValue> = soul.core_values().collect();
        // Every pair of core values should have distinct embeddings
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                let sim = values[i].embedding.similarity(&values[j].embedding);
                assert!(
                    sim < 0.99,
                    "Values '{}' and '{}' have near-identical embeddings (sim={})",
                    values[i].id,
                    values[j].id,
                    sim
                );
            }
        }
    }

    #[test]
    fn test_identity_and_essence_differ() {
        let soul = Soul::default();
        let sim = soul.self_model().identity.similarity(soul.essence());
        assert!(
            sim < 0.99,
            "Identity and essence should have distinct embeddings (sim={})",
            sim
        );
    }

    #[test]
    fn test_alignment_differs_across_values() {
        let soul = Soul::default();
        // A random action should not score identically against all values
        let action = ContinuousHV::random(512, 777);
        let result = soul.evaluate_alignment(&action);
        let scores: Vec<f32> = result.per_value_alignment.values().copied().collect();
        let all_same = scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
        assert!(
            !all_same,
            "All alignment scores are identical, embeddings likely the same"
        );
    }

    #[test]
    fn test_experience_integration() {
        let mut soul = Soul::default();

        let experience = Experience {
            embedding: ContinuousHV::random(512, 42),
            value_alignment: 0.8,
            emotional_valence: 0.5,
            lessons: vec!["learned something".to_string()],
            timestamp: 12345,
        };

        soul.integrate_experience(experience);
        assert_eq!(soul.stats.experiences_integrated, 1);
    }

    #[test]
    fn test_core_values() {
        let soul = Soul::default();

        // Should have Eight Harmonies by default
        assert!(soul.get_value("resonance").is_some());
        assert!(soul.get_value("flourishing").is_some());
    }
}
