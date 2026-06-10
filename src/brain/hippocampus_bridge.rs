// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hippocampus Bridge - Connecting Crystallized Concepts to Episodic Memory
//!
//! This module bridges the ConsciousnessWorldModel to the HippocampusActor,
//! allowing crystallized concepts to be stored as episodic memories.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                      HIPPOCAMPUS BRIDGE                                  │
//! │                                                                          │
//! │  ┌──────────────────────┐          ┌──────────────────────────────┐    │
//! │  │  ConsciousnessWorldModel │ ───▶  │    HippocampusBridge        │    │
//! │  │  (Crystallized Concepts) │        │                              │    │
//! │  └──────────────────────┘          │  • Converts concepts to memories│   │
//! │                                     │  • Stores episodic traces     │    │
//! │                                     │  • Enables concept recall     │    │
//! │                                     └──────────────┬───────────────┘    │
//! │                                                    │                     │
//! │                                                    ▼                     │
//! │                              ┌──────────────────────────────────────┐   │
//! │                              │      HippocampusActor               │   │
//! │                              │   (Episodic Memory Storage)         │   │
//! │                              │                                      │   │
//! │                              │  "I remember when this concept      │   │
//! │                              │   first crystallized..."            │   │
//! │                              └──────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## How It Works
//!
//! 1. **Concept Crystallization**: WorldModel crystallizes a new concept
//! 2. **Memory Encoding**: Bridge converts concept into episodic memory trace
//! 3. **Storage**: Hippocampus stores with holographic compression
//! 4. **Recall**: Later, concepts can be recalled by similarity or context

use crate::consciousness::recursive_improvement::{ConsciousnessWorldModel, WorldModelStats};
use crate::dynamics::CrystalizedConcept;
use crate::memory::{EmotionalValence, HippocampusActor, RecallQuery, RecallResult};
use anyhow::Result;
use std::collections::HashMap;

/// Bridge between ConsciousnessWorldModel and HippocampusActor
pub struct HippocampusBridge {
    /// Configuration for memory encoding
    config: HippocampusBridgeConfig,

    /// Statistics
    stats: HippocampusBridgeStats,

    /// Mapping from concept UIDs to memory IDs
    concept_to_memory: HashMap<String, u64>,

    /// Last known concept count (for delta detection)
    last_concept_count: usize,
}

/// Configuration for the hippocampus bridge
#[derive(Debug, Clone)]
pub struct HippocampusBridgeConfig {
    /// Base context tags for all concept memories
    pub base_tags: Vec<String>,

    /// Whether to include concept UID in tags
    pub include_uid_tag: bool,

    /// Minimum consciousness level to encode concepts
    pub min_consciousness_threshold: f64,
}

impl Default for HippocampusBridgeConfig {
    fn default() -> Self {
        Self {
            base_tags: vec![
                "crystallized_concept".to_string(),
                "consciousness".to_string(),
            ],
            include_uid_tag: true,
            min_consciousness_threshold: 0.1,
        }
    }
}

/// Statistics for the hippocampus bridge
#[derive(Debug, Clone, Default)]
pub struct HippocampusBridgeStats {
    /// Total concepts encoded as memories
    pub concepts_encoded: u64,

    /// Total memory recalls performed
    pub recalls_performed: u64,

    /// Average memories returned per recall
    pub avg_recall_count: f32,

    /// Peak consciousness level observed
    pub peak_consciousness: f64,
}

impl Default for HippocampusBridge {
    fn default() -> Self {
        Self::new(HippocampusBridgeConfig::default())
    }
}

impl HippocampusBridge {
    /// Create a new hippocampus bridge
    pub fn new(config: HippocampusBridgeConfig) -> Self {
        Self {
            config,
            stats: HippocampusBridgeStats::default(),
            concept_to_memory: HashMap::new(),
            last_concept_count: 0,
        }
    }

    /// Sync with world model and encode new concepts as memories
    ///
    /// Call this periodically to convert crystallized concepts into episodic memories.
    pub fn sync(
        &mut self,
        world_model: &ConsciousnessWorldModel,
        hippocampus: &mut HippocampusActor,
    ) -> Result<Vec<u64>> {
        let stats = world_model.stats();
        let current_count = world_model.pending_concepts.len();

        // Track peak consciousness
        if stats.consciousness_level > self.stats.peak_consciousness {
            self.stats.peak_consciousness = stats.consciousness_level;
        }

        // Check if below consciousness threshold
        if stats.consciousness_level < self.config.min_consciousness_threshold {
            return Ok(Vec::new());
        }

        // Detect new concepts (delta since last sync)
        let mut new_memory_ids = Vec::new();

        if current_count > self.last_concept_count {
            let new_concepts = &world_model.pending_concepts[self.last_concept_count..];

            for concept in new_concepts {
                // Skip if already encoded
                if self.concept_to_memory.contains_key(&concept.uid) {
                    continue;
                }

                match self.encode_concept_as_memory(concept, &stats, hippocampus) {
                    Ok(memory_id) => {
                        self.concept_to_memory
                            .insert(concept.uid.clone(), memory_id);
                        new_memory_ids.push(memory_id);
                        self.stats.concepts_encoded += 1;
                    }
                    Err(e) => {
                        tracing::warn!(
                            concept_uid = %concept.uid,
                            error = %e,
                            "Failed to encode concept as memory"
                        );
                    }
                }
            }
        }

        self.last_concept_count = current_count;
        Ok(new_memory_ids)
    }

    /// Encode a crystallized concept as an episodic memory
    fn encode_concept_as_memory(
        &self,
        concept: &CrystalizedConcept,
        stats: &WorldModelStats,
        hippocampus: &mut HippocampusActor,
    ) -> Result<u64> {
        // Generate content description
        let name_str = concept.name.as_deref().unwrap_or("unnamed");
        let content = format!(
            "Crystallized concept '{}' [{}] at consciousness level {:.1}%. {}",
            name_str,
            concept.uid,
            stats.consciousness_level * 100.0,
            concept
                .description
                .as_deref()
                .unwrap_or("A newly discovered pattern.")
        );

        // Build context tags
        let mut tags = self.config.base_tags.clone();
        if self.config.include_uid_tag {
            tags.push(format!("concept:{}", concept.uid));
        }
        if let Some(name) = &concept.name {
            tags.push(format!("name:{}", name));
        }
        tags.push(format!("activations:{}", concept.activation_count));

        // Determine emotional valence based on context
        let emotion = self.determine_emotion(concept, stats);

        // Store in hippocampus
        hippocampus.remember(content, tags, emotion)
    }

    /// Determine emotional valence for a concept
    fn determine_emotion(
        &self,
        concept: &CrystalizedConcept,
        stats: &WorldModelStats,
    ) -> EmotionalValence {
        // New concepts with high consciousness = positive (discovery!)
        if concept.activation_count <= 1 && stats.consciousness_level > 0.5 {
            return EmotionalValence::Positive;
        }

        // Named concepts = neutral (familiarity)
        if concept.name.is_some() {
            return EmotionalValence::Neutral;
        }

        // Frequently activated but unnamed = slight negative (needs attention)
        if concept.activation_count > 5 && concept.name.is_none() {
            return EmotionalValence::Negative;
        }

        EmotionalValence::Neutral
    }

    /// Recall memories related to a concept
    pub fn recall_concept_memories(
        &mut self,
        concept_uid: &str,
        hippocampus: &mut HippocampusActor,
        top_k: usize,
    ) -> Result<Vec<RecallResult>> {
        let query = RecallQuery {
            query: format!("concept {} crystallized", concept_uid),
            context_tags: vec![format!("concept:{}", concept_uid)],
            top_k,
            threshold: 0.3,
            ..Default::default()
        };

        let results = hippocampus.recall(query)?;

        self.stats.recalls_performed += 1;
        let n = self.stats.recalls_performed as f32;
        self.stats.avg_recall_count =
            (self.stats.avg_recall_count * (n - 1.0) + results.len() as f32) / n;

        Ok(results)
    }

    /// Recall memories by emotional valence (e.g., "when did we feel frustrated?")
    pub fn recall_by_emotion(
        &mut self,
        emotion: EmotionalValence,
        hippocampus: &mut HippocampusActor,
        top_k: usize,
    ) -> Result<Vec<RecallResult>> {
        let query = RecallQuery {
            query: "crystallized concept consciousness".to_string(),
            emotion_filter: Some(emotion),
            top_k,
            threshold: 0.2,
            ..Default::default()
        };

        let results = hippocampus.recall(query)?;

        self.stats.recalls_performed += 1;
        Ok(results)
    }

    /// Get memory ID for a concept UID
    pub fn get_memory_id(&self, concept_uid: &str) -> Option<u64> {
        self.concept_to_memory.get(concept_uid).copied()
    }

    /// Get all concept-to-memory mappings
    pub fn concept_memory_map(&self) -> &HashMap<String, u64> {
        &self.concept_to_memory
    }

    /// Get bridge statistics
    pub fn stats(&self) -> &HippocampusBridgeStats {
        &self.stats
    }

    /// Get bridge configuration
    pub fn config(&self) -> &HippocampusBridgeConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: HippocampusBridgeConfig) {
        self.config = config;
    }
}

/// Extension trait for ConsciousnessWorldModel to easily create episodic memories
pub trait StoreInHippocampus {
    /// Store the latest crystallized concept in hippocampus
    fn store_latest_in_hippocampus(
        &self,
        hippocampus: &mut HippocampusActor,
    ) -> Result<Option<u64>>;
}

impl StoreInHippocampus for ConsciousnessWorldModel {
    fn store_latest_in_hippocampus(
        &self,
        hippocampus: &mut HippocampusActor,
    ) -> Result<Option<u64>> {
        let Some(concept) = self.pending_concepts.last() else {
            return Ok(None);
        };

        let stats = self.stats();
        let name_str = concept.name.as_deref().unwrap_or("unnamed");
        let content = format!(
            "Crystallized concept '{}' [{}] at consciousness {:.1}%",
            name_str,
            concept.uid,
            stats.consciousness_level * 100.0
        );

        let tags = vec![
            "crystallized_concept".to_string(),
            format!("concept:{}", concept.uid),
        ];

        let emotion = if concept.activation_count <= 1 {
            EmotionalValence::Positive
        } else {
            EmotionalValence::Neutral
        };

        let memory_id = hippocampus.remember(content, tags, emotion)?;
        Ok(Some(memory_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::{
        ConsciousnessAction, ConsciousnessTransition, LatentConsciousnessState, WorldModelConfig,
    };

    #[test]
    fn test_bridge_creation() {
        let bridge = HippocampusBridge::default();
        assert_eq!(bridge.concept_to_memory.len(), 0);
        assert_eq!(bridge.stats.concepts_encoded, 0);
    }

    #[test]
    fn test_determine_emotion() {
        let bridge = HippocampusBridge::default();

        // New concept with high consciousness = positive
        let new_concept = CrystalizedConcept {
            uid: "test-1".to_string(),
            hypervector: vec![0.5; 32],
            name: None,
            description: None,
            crystallized_at: 12345,
            activation_count: 1,
            attractor_signature: vec![0.3; 16],
            primitive_birth: Vec::new(),
        };
        let high_stats = WorldModelStats {
            consciousness_level: 0.7,
            accumulated_surprise: 1.5,
            transitions_observed: 100,
            transitions_imagined: 50,
            counterfactuals_analyzed: 10,
            dreams_completed: 5,
            average_prediction_error: 0.3,
            concepts_crystallized: 5,
            episodes_stored: 20,
        };
        assert_eq!(
            bridge.determine_emotion(&new_concept, &high_stats),
            EmotionalValence::Positive
        );

        // Named concept = neutral
        let named_concept = CrystalizedConcept {
            uid: "test-2".to_string(),
            hypervector: vec![0.5; 32],
            name: Some("TestConcept".to_string()),
            description: None,
            crystallized_at: 12345,
            activation_count: 5,
            attractor_signature: vec![0.3; 16],
            primitive_birth: Vec::new(),
        };
        assert_eq!(
            bridge.determine_emotion(&named_concept, &high_stats),
            EmotionalValence::Neutral
        );

        // Frequently activated but unnamed = negative
        let active_unnamed = CrystalizedConcept {
            uid: "test-3".to_string(),
            hypervector: vec![0.5; 32],
            name: None,
            description: None,
            crystallized_at: 12345,
            activation_count: 10,
            attractor_signature: vec![0.3; 16],
            primitive_birth: Vec::new(),
        };
        let low_stats = WorldModelStats {
            consciousness_level: 0.3,
            accumulated_surprise: 1.0,
            transitions_observed: 50,
            transitions_imagined: 25,
            counterfactuals_analyzed: 5,
            dreams_completed: 2,
            average_prediction_error: 0.5,
            concepts_crystallized: 3,
            episodes_stored: 10,
        };
        assert_eq!(
            bridge.determine_emotion(&active_unnamed, &low_stats),
            EmotionalValence::Negative
        );
    }
}