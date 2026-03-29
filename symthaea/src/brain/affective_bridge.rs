// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Affective Bridge - Connecting Crystallized Concepts to Emotional Valence
//!
//! This module bridges the ConsciousnessWorldModel to the AffectiveConsciousnessAnalyzer,
//! giving crystallized concepts emotional depth and valence.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                       AFFECTIVE BRIDGE                                   │
//! │                                                                          │
//! │  ┌──────────────────────┐          ┌──────────────────────────────┐    │
//! │  │  ConsciousnessWorldModel │ ───▶  │     AffectiveBridge         │    │
//! │  │  (Crystallized Concepts) │        │                              │    │
//! │  └──────────────────────┘          │  • Evaluates concept affect   │    │
//! │                                     │  • Tags emotional valence    │    │
//! │                                     │  • Tracks emotional history  │    │
//! │                                     └──────────────┬───────────────┘    │
//! │                                                    │                     │
//! │                                                    ▼                     │
//! │                       ┌─────────────────────────────────────────────┐   │
//! │                       │      AffectiveConsciousnessAnalyzer         │   │
//! │                       │   (Damasio's Somatic Marker Hypothesis)     │   │
//! │                       │                                             │   │
//! │                       │  "This concept FEELS exciting/threatening"  │   │
//! │                       └─────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## How It Works
//!
//! 1. **Concept Crystallization**: WorldModel crystallizes a new concept
//! 2. **Affect Evaluation**: Bridge evaluates concept's emotional significance
//! 3. **Valence Tagging**: Concept gets tagged with valence/arousal/dominance
//! 4. **History Tracking**: Emotional trajectory of concepts over time

use crate::consciousness::affective_consciousness::{CoreAffect, EmotionCategory};
#[cfg(feature = "full_consciousness")]
use crate::consciousness::recursive_improvement::{ConsciousnessWorldModel, WorldModelStats};
#[cfg(feature = "full_consciousness")]
use crate::dynamics::CrystalizedConcept;
use std::collections::{HashMap, VecDeque};

/// Affective assessment of a crystallized concept
#[derive(Debug, Clone)]
pub struct ConceptAffect {
    /// The concept's unique ID
    pub concept_uid: String,

    /// Core affect (valence, arousal, dominance)
    pub affect: CoreAffect,

    /// Discrete emotion category
    pub category: EmotionCategory,

    /// Emotional intensity (0.0 = neutral, 1.0 = extreme)
    pub intensity: f64,

    /// Whether this is the first assessment
    pub is_novel: bool,

    /// Assessment timestamp
    pub assessed_at: std::time::Instant,
}

/// Bridge between ConsciousnessWorldModel and AffectiveConsciousness
#[derive(Debug)]
pub struct AffectiveBridge {
    /// Configuration
    config: AffectiveBridgeConfig,

    /// Statistics
    stats: AffectiveBridgeStats,

    /// Concept affect history (keyed by concept UID)
    concept_affects: HashMap<String, VecDeque<ConceptAffect>>,

    /// Last known concept count (used by sync() which requires full_consciousness)
    #[cfg(feature = "full_consciousness")]
    last_concept_count: usize,

    /// Global affect state (mood influenced by concepts)
    global_affect: CoreAffect,
}

/// Configuration for the affective bridge
#[derive(Debug, Clone)]
pub struct AffectiveBridgeConfig {
    /// Base valence for new concepts (curiosity = positive)
    pub novelty_valence_boost: f64,

    /// Base arousal for new concepts (activation = high)
    pub novelty_arousal_boost: f64,

    /// How much named concepts shift toward neutral
    pub familiarity_damping: f64,

    /// Maximum affect history per concept
    pub max_history_per_concept: usize,

    /// Global affect decay rate
    pub global_affect_decay: f64,

    /// Minimum consciousness to evaluate affect
    pub min_consciousness_threshold: f64,
}

impl Default for AffectiveBridgeConfig {
    fn default() -> Self {
        Self {
            novelty_valence_boost: 0.4, // New = slightly positive (curiosity)
            novelty_arousal_boost: 0.5, // New = arousing
            familiarity_damping: 0.7,   // Named = 70% toward neutral
            max_history_per_concept: 10,
            global_affect_decay: 0.05,
            min_consciousness_threshold: 0.1,
        }
    }
}

/// Statistics for the affective bridge
#[derive(Debug, Clone, Default)]
pub struct AffectiveBridgeStats {
    /// Total concepts evaluated
    pub concepts_evaluated: u64,

    /// Positive affect concepts
    pub positive_concepts: u64,

    /// Negative affect concepts
    pub negative_concepts: u64,

    /// Neutral affect concepts
    pub neutral_concepts: u64,

    /// Average valence across all concepts
    pub avg_valence: f64,

    /// Average arousal across all concepts
    pub avg_arousal: f64,

    /// Peak emotional intensity observed
    pub peak_intensity: f64,
}

impl Default for AffectiveBridge {
    fn default() -> Self {
        Self::new(AffectiveBridgeConfig::default())
    }
}

impl AffectiveBridge {
    /// Create a new affective bridge
    pub fn new(config: AffectiveBridgeConfig) -> Self {
        Self {
            config,
            stats: AffectiveBridgeStats::default(),
            concept_affects: HashMap::new(),
            #[cfg(feature = "full_consciousness")]
            last_concept_count: 0,
            global_affect: CoreAffect::neutral(),
        }
    }

    /// Sync with world model and evaluate affect for new concepts
    #[cfg(feature = "full_consciousness")]
    pub fn sync(&mut self, world_model: &ConsciousnessWorldModel) -> Vec<ConceptAffect> {
        let stats = world_model.stats();
        let current_count = world_model.pending_concepts.len();

        // Decay global affect
        self.global_affect.decay(self.config.global_affect_decay);

        // Check consciousness threshold
        if stats.consciousness_level < self.config.min_consciousness_threshold {
            return Vec::new();
        }

        let mut new_affects = Vec::new();

        // Detect new concepts
        if current_count > self.last_concept_count {
            let new_concepts = &world_model.pending_concepts[self.last_concept_count..];

            for concept in new_concepts {
                let affect = self.evaluate_concept_affect(concept, &stats);
                new_affects.push(affect);
            }
        }

        self.last_concept_count = current_count;
        new_affects
    }

    /// Evaluate the affective significance of a crystallized concept
    #[cfg(feature = "full_consciousness")]
    fn evaluate_concept_affect(
        &mut self,
        concept: &CrystalizedConcept,
        stats: &WorldModelStats,
    ) -> ConceptAffect {
        let is_novel = concept.activation_count <= 1;
        let is_named = !concept.name.is_empty();

        // Calculate valence
        let mut valence = if is_novel {
            // New concepts trigger SEEKING (curiosity = positive valence)
            self.config.novelty_valence_boost
        } else if is_named {
            // Named concepts are familiar = neutral
            0.0
        } else {
            // Unnamed but active = slight unease
            -0.1
        };

        // Modulate by consciousness level (high consciousness = more intense affect)
        valence *= 0.5 + stats.consciousness_level * 0.5;

        // Calculate arousal
        let mut arousal = if is_novel {
            // New = arousing
            self.config.novelty_arousal_boost
        } else {
            // Familiar = calming
            -0.1 + (concept.activation_count as f64 * 0.05).min(0.3)
        };

        arousal *= 0.5 + stats.consciousness_level * 0.5;

        // Calculate dominance (sense of control)
        let dominance = if is_named {
            // Named = understood = controlled
            0.3
        } else if concept.activation_count > 5 {
            // Frequently appearing but unnamed = confusing
            -0.2
        } else {
            0.0
        };

        let affect = CoreAffect::new(valence as f32, arousal as f32, dominance as f32);
        let category = affect.to_emotion_category();
        let intensity = affect.intensity();

        // Update statistics
        self.stats.concepts_evaluated += 1;
        match category {
            EmotionCategory::Joy | EmotionCategory::Contentment | EmotionCategory::Serenity => {
                self.stats.positive_concepts += 1
            }
            EmotionCategory::Anger
            | EmotionCategory::Fear
            | EmotionCategory::Sadness
            | EmotionCategory::Disgust => self.stats.negative_concepts += 1,
            _ => self.stats.neutral_concepts += 1,
        }

        // Update rolling averages
        let n = self.stats.concepts_evaluated as f64;
        self.stats.avg_valence = (self.stats.avg_valence * (n - 1.0) + valence) / n;
        self.stats.avg_arousal = (self.stats.avg_arousal * (n - 1.0) + arousal) / n;

        if intensity > self.stats.peak_intensity {
            self.stats.peak_intensity = intensity;
        }

        // Update global affect
        self.global_affect = self.global_affect.blend(&affect, 0.2);

        // Store in history
        let concept_affect = ConceptAffect {
            concept_uid: concept.uid.clone(),
            affect,
            category,
            intensity,
            is_novel,
            assessed_at: std::time::Instant::now(),
        };

        let history = self
            .concept_affects
            .entry(concept.uid.clone())
            .or_insert_with(VecDeque::new);

        history.push_back(concept_affect.clone());
        if history.len() > self.config.max_history_per_concept {
            history.pop_front();
        }

        concept_affect
    }

    /// Get current global affect (mood influenced by all concepts)
    pub fn global_affect(&self) -> &CoreAffect {
        &self.global_affect
    }

    /// Evaluate affect from cognitive loop signals (no world model dependency).
    /// Science: Damasio (1994) — somatic markers guide cognition via affective signals
    pub fn evaluate_from_signals(
        &mut self,
        prediction_error: f32,
        surprise_triggered: bool,
        consciousness_proxy: f64,
        moral_score: f32,
    ) -> &CoreAffect {
        let valence = ((0.5 - prediction_error) + moral_score * 0.3).clamp(-1.0, 1.0);
        let arousal = (0.3
            + if surprise_triggered { 0.3 } else { 0.0 }
            + (consciousness_proxy * 0.4) as f32
            + (prediction_error * 0.3).min(0.3))
        .clamp(0.0, 1.0);
        let dominance = ((consciousness_proxy * 0.6) as f32 - 0.3).clamp(-1.0, 1.0);
        let new_affect = CoreAffect::new(valence, arousal, dominance);
        self.global_affect = self.global_affect.blend(&new_affect, 0.3);
        self.global_affect.decay(self.config.global_affect_decay);
        &self.global_affect
    }

    /// Evaluate affect from cognitive + social signals (Damasio + social somatic markers).
    /// Science: Decety & Chaminade (2003) — social context modulates affective processing
    pub fn evaluate_from_signals_with_social(
        &mut self,
        prediction_error: f32,
        surprise_triggered: bool,
        consciousness_proxy: f64,
        moral_score: f32,
        social_trust: f32,
        social_cooperation_rate: f32,
        peer_valence: f32,
    ) -> &CoreAffect {
        // Base affect from cognitive signals
        let mut valence = ((0.5 - prediction_error) + moral_score * 0.3).clamp(-1.0, 1.0);
        let mut arousal = (0.3
            + if surprise_triggered { 0.3 } else { 0.0 }
            + (consciousness_proxy * 0.4) as f32
            + (prediction_error * 0.3).min(0.3))
        .clamp(0.0, 1.0);
        let mut dominance = ((consciousness_proxy * 0.6) as f32 - 0.3).clamp(-1.0, 1.0);

        // Social modulation: emotional contagion (peer valence bleeds into self valence)
        valence = (valence + peer_valence * 0.15).clamp(-1.0, 1.0);

        // Social modulation: cooperation boosts arousal (social engagement)
        arousal = (arousal + social_cooperation_rate * 0.1).clamp(0.0, 1.0);

        // Social modulation: trust increases dominance (sense of control/safety)
        dominance = (dominance + (social_trust - 0.5) * 0.2).clamp(-1.0, 1.0);

        let new_affect = CoreAffect::new(valence, arousal, dominance);
        self.global_affect = self.global_affect.blend(&new_affect, 0.3);
        self.global_affect.decay(self.config.global_affect_decay);
        &self.global_affect
    }

    /// Get affect history for a specific concept
    pub fn concept_history(&self, concept_uid: &str) -> Option<&VecDeque<ConceptAffect>> {
        self.concept_affects.get(concept_uid)
    }

    /// Get the latest affect for a concept
    pub fn latest_affect(&self, concept_uid: &str) -> Option<&ConceptAffect> {
        self.concept_affects.get(concept_uid)?.back()
    }

    /// Get all concepts with positive affect
    pub fn positive_concepts(&self) -> Vec<&ConceptAffect> {
        self.concept_affects
            .values()
            .filter_map(|h| h.back())
            .filter(|a| a.affect.valence > 0.2)
            .collect()
    }

    /// Get all concepts with negative affect
    pub fn negative_concepts(&self) -> Vec<&ConceptAffect> {
        self.concept_affects
            .values()
            .filter_map(|h| h.back())
            .filter(|a| a.affect.valence < -0.2)
            .collect()
    }

    /// Get the emotional category breakdown
    pub fn emotion_breakdown(&self) -> HashMap<EmotionCategory, usize> {
        let mut breakdown = HashMap::new();
        for history in self.concept_affects.values() {
            if let Some(latest) = history.back() {
                *breakdown.entry(latest.category).or_insert(0) += 1;
            }
        }
        breakdown
    }

    /// Get bridge statistics
    pub fn stats(&self) -> &AffectiveBridgeStats {
        &self.stats
    }

    /// Get bridge configuration
    pub fn config(&self) -> &AffectiveBridgeConfig {
        &self.config
    }
}

/// Extension trait for ConsciousnessWorldModel to get affect for concepts
#[cfg(feature = "full_consciousness")]
pub trait ConceptAffectEvaluator {
    /// Evaluate affect for the latest crystallized concept
    fn evaluate_latest_affect(&self) -> Option<ConceptAffect>;
}

#[cfg(feature = "full_consciousness")]
impl ConceptAffectEvaluator for ConsciousnessWorldModel {
    fn evaluate_latest_affect(&self) -> Option<ConceptAffect> {
        let concept = self.pending_concepts.last()?;
        let stats = self.stats();

        let is_novel = concept.activation_count <= 1;
        let is_named = !concept.name.is_empty();

        let valence = if is_novel {
            0.4 * (0.5 + stats.consciousness_level * 0.5)
        } else if is_named {
            0.0
        } else {
            -0.1 * (0.5 + stats.consciousness_level * 0.5)
        };

        let arousal = if is_novel {
            0.5 * (0.5 + stats.consciousness_level * 0.5)
        } else {
            -0.1 + (concept.activation_count as f64 * 0.05).min(0.3)
        };

        let dominance = if is_named {
            0.3
        } else if concept.activation_count > 5 {
            -0.2
        } else {
            0.0
        };

        let affect = CoreAffect::new(valence as f32, arousal as f32, dominance as f32);

        Some(ConceptAffect {
            concept_uid: concept.uid.clone(),
            affect,
            category: affect.to_emotion_category(),
            intensity: affect.intensity(),
            is_novel,
            assessed_at: std::time::Instant::now(),
        })
    }
}

#[cfg(test)]
#[cfg(feature = "full_consciousness")]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = AffectiveBridge::default();
        assert_eq!(bridge.concept_affects.len(), 0);
        assert_eq!(bridge.stats.concepts_evaluated, 0);
    }

    #[test]
    fn test_novel_concept_positive_valence() {
        let concept = CrystalizedConcept {
            id: 1,
            uid: "test-1".to_string(),
            name: String::new(),
            description: None,
            embedding: vec![0.5; 32],
            attractor_signature: vec![0.3; 16],
            associations: HashMap::new(),
            confidence: 0.5,
            activation_count: 1, // Novel!
            last_activated: 12345,
            source_memories: Vec::new(),
            abstraction_level: 0,
            emotional_valence: 0.0,
            tags: Vec::new(),
        };

        let stats = WorldModelStats {
            consciousness_level: 0.7,
            total_transitions: 100,
            total_predictions: 50,
            avg_prediction_error: 0.3,
            concepts_crystallized: 5,
        };

        let mut bridge = AffectiveBridge::default();
        let affect = bridge.evaluate_concept_affect(&concept, &stats);

        assert!(
            affect.affect.valence > 0.0,
            "Novel concepts should be positive (curiosity)"
        );
        assert!(
            affect.affect.arousal > 0.0,
            "Novel concepts should be arousing"
        );
        assert!(affect.is_novel);
    }

    #[test]
    fn test_named_concept_neutral() {
        let concept = CrystalizedConcept {
            id: 2,
            uid: "test-2".to_string(),
            name: "FamiliarConcept".to_string(),
            description: None,
            embedding: vec![0.5; 32],
            attractor_signature: vec![0.3; 16],
            associations: HashMap::new(),
            confidence: 0.5,
            activation_count: 10,
            last_activated: 12345,
            source_memories: Vec::new(),
            abstraction_level: 0,
            emotional_valence: 0.0,
            tags: Vec::new(),
        };

        let stats = WorldModelStats {
            consciousness_level: 0.5,
            total_transitions: 50,
            total_predictions: 25,
            avg_prediction_error: 0.5,
            concepts_crystallized: 3,
        };

        let mut bridge = AffectiveBridge::default();
        let affect = bridge.evaluate_concept_affect(&concept, &stats);

        // Named concepts should have positive dominance (understood)
        assert!(
            affect.affect.dominance > 0.0,
            "Named concepts should feel controlled"
        );
    }

    #[test]
    fn test_global_affect_blending() {
        let mut bridge = AffectiveBridge::default();

        let concept = CrystalizedConcept {
            id: 3,
            uid: "test-3".to_string(),
            name: String::new(),
            description: None,
            embedding: vec![0.5; 32],
            attractor_signature: vec![0.3; 16],
            associations: HashMap::new(),
            confidence: 0.5,
            activation_count: 1,
            last_activated: 12345,
            source_memories: Vec::new(),
            abstraction_level: 0,
            emotional_valence: 0.0,
            tags: Vec::new(),
        };

        let stats = WorldModelStats {
            consciousness_level: 0.8,
            total_transitions: 100,
            total_predictions: 50,
            avg_prediction_error: 0.3,
            concepts_crystallized: 5,
        };

        // Record initial global affect intensity (neutral has arousal=0.5, so
        // intensity is ~0.5 — not zero. We just need to confirm processing shifts it.)
        let initial_intensity = bridge.global_affect().intensity();

        // Process a concept
        bridge.evaluate_concept_affect(&concept, &stats);

        // Global affect should have shifted from the neutral baseline
        let post_intensity = bridge.global_affect().intensity();
        assert!(
            (post_intensity - initial_intensity).abs() > 0.001,
            "Affect should shift after evaluating concept: initial={initial_intensity}, post={post_intensity}"
        );
    }
}
