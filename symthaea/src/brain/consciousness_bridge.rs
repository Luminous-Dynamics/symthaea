// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Bridge - Connecting Heart (World Model) to Brain (Global Workspace)
//!
//! This module bridges the ConsciousnessWorldModel (the "heart" that crystallizes concepts)
//! to the PrefrontalCortex (the "brain" that broadcasts attention).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                      CONSCIOUSNESS BRIDGE                                │
//! │                                                                          │
//! │  ┌──────────────────────┐          ┌──────────────────────────────┐    │
//! │  │  ConsciousnessWorldModel │ ───▶  │    ConsciousnessBridge      │    │
//! │  │  (Heart - Crystallizes   │        │                              │    │
//! │  │   Concepts)              │        │  • Converts concepts to bids │    │
//! │  └──────────────────────┘          │  • Calculates salience        │    │
//! │                                     │  • Injects into Prefrontal   │    │
//! │                                     └──────────────┬───────────────┘    │
//! │                                                    │                     │
//! │                                                    ▼                     │
//! │                              ┌──────────────────────────────────────┐   │
//! │                              │     PrefrontalCortexActor            │   │
//! │                              │     (Brain - Global Workspace)       │   │
//! │                              │                                      │   │
//! │                              │  Spotlight: [Crystallized Concept!]  │   │
//! │                              └──────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## How It Works
//!
//! 1. **Concept Discovery**: ConsciousnessWorldModel crystallizes novel attractors
//! 2. **Bid Generation**: Bridge converts concepts to AttentionBids with salience
//! 3. **Competition**: Bids compete in Global Workspace attention cycle
//! 4. **Broadcast**: Winning concepts get broadcast to all brain modules

use super::prefrontal::{AttentionBid, PrefrontalCortexActor};
use crate::consciousness::recursive_improvement::{
    ActionType, ConsciousnessAction, ConsciousnessTransition, ConsciousnessWorldModel,
    LatentConsciousnessState, WorldModelStats,
};
use crate::dynamics::CrystalizedConcept;
use crate::memory::EmotionalValence;
use std::collections::VecDeque;

/// Bridge between ConsciousnessWorldModel and PrefrontalCortex
///
/// This bridge monitors the world model for crystallized concepts and
/// generates attention bids that can compete in the global workspace.
pub struct ConsciousnessBridge {
    /// Recent concepts converted to bids (buffer)
    pending_bids: VecDeque<AttentionBid>,

    /// Configuration for bid generation
    config: BridgeConfig,

    /// Statistics
    stats: BridgeStats,

    /// Last known concept count (for delta detection)
    last_concept_count: usize,
}

/// Configuration for the consciousness bridge
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Base salience for new concepts (0.0-1.0)
    pub base_salience: f32,

    /// Salience boost per activation count
    pub activation_salience_multiplier: f32,

    /// Base urgency for new concepts
    pub base_urgency: f32,

    /// Maximum bids to buffer
    pub max_pending_bids: usize,

    /// Minimum consciousness level to generate bids
    pub min_consciousness_threshold: f64,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            base_salience: 0.6,
            activation_salience_multiplier: 0.01, // Small boost per activation
            base_urgency: 0.5,
            max_pending_bids: 20,
            min_consciousness_threshold: 0.1,
        }
    }
}

/// Statistics for the bridge
#[derive(Debug, Clone, Default)]
pub struct BridgeStats {
    /// Total concepts converted to bids
    pub concepts_converted: u64,

    /// Total bids submitted to prefrontal
    pub bids_submitted: u64,

    /// Total bids that won attention competition
    pub bids_won_spotlight: u64,

    /// Average salience of generated bids
    pub avg_salience: f32,

    /// Peak consciousness level observed
    pub peak_consciousness: f64,
}

impl Default for ConsciousnessBridge {
    fn default() -> Self {
        Self::new(BridgeConfig::default())
    }
}

impl ConsciousnessBridge {
    /// Create a new consciousness bridge
    pub fn new(config: BridgeConfig) -> Self {
        Self {
            pending_bids: VecDeque::new(),
            config,
            stats: BridgeStats::default(),
            last_concept_count: 0,
        }
    }

    /// Sync with world model and generate bids for new concepts
    ///
    /// Call this periodically to convert crystallized concepts into attention bids.
    pub fn sync(&mut self, world_model: &ConsciousnessWorldModel) {
        let stats = world_model.stats();
        let current_count = world_model.pending_concepts.len();

        // Track peak consciousness
        if stats.consciousness_level > self.stats.peak_consciousness {
            self.stats.peak_consciousness = stats.consciousness_level;
        }

        // Check if below consciousness threshold
        if stats.consciousness_level < self.config.min_consciousness_threshold {
            return;
        }

        // Detect new concepts (delta since last sync)
        if current_count > self.last_concept_count {
            let new_concepts = &world_model.pending_concepts[self.last_concept_count..];

            for concept in new_concepts {
                let bid = self.concept_to_bid(concept, stats);

                // Add to pending buffer
                self.pending_bids.push_back(bid);
                if self.pending_bids.len() > self.config.max_pending_bids {
                    self.pending_bids.pop_front();
                }

                self.stats.concepts_converted += 1;
            }
        }

        self.last_concept_count = current_count;
    }

    /// Convert a crystallized concept to an attention bid
    fn concept_to_bid(
        &mut self,
        concept: &CrystalizedConcept,
        stats: &WorldModelStats,
    ) -> AttentionBid {
        // Calculate salience based on concept properties
        // Use activation_count as a proxy for importance
        let activation_bonus =
            (concept.activation_count as f32 * self.config.activation_salience_multiplier).min(0.3);
        let salience = (self.config.base_salience + activation_bonus).clamp(0.0, 1.0);

        // Calculate urgency based on consciousness level and whether this is a new concept
        let consciousness_factor = stats.consciousness_level as f32 * 0.2;
        let newness_bonus = if concept.activation_count <= 1 {
            0.2
        } else {
            0.0
        }; // New concepts are urgent
        let urgency =
            (self.config.base_urgency + consciousness_factor + newness_bonus).clamp(0.0, 1.0);

        // Determine emotional valence based on concept properties
        let emotion = if concept.activation_count <= 1 {
            // Brand new concept = curiosity/positive
            EmotionalValence::Positive
        } else if concept.name.is_some() {
            // Named concept = familiar/neutral
            EmotionalValence::Neutral
        } else {
            // Unnamed but active = needs attention
            EmotionalValence::Negative
        };

        // Track average salience
        let n = self.stats.concepts_converted as f32;
        self.stats.avg_salience = (self.stats.avg_salience * n + salience) / (n + 1.0);

        // Generate content description
        let name_str = concept.name.as_deref().unwrap_or("unnamed");
        let content = format!(
            "Crystallized Concept [{}]: name='{}', activations={}",
            concept.uid, name_str, concept.activation_count
        );

        AttentionBid::new("ConsciousnessWorldModel", content)
            .with_salience(salience)
            .with_urgency(urgency)
            .with_emotion(emotion)
            .with_tags(vec![
                "concept".to_string(),
                "crystallized".to_string(),
                format!("activations:{}", concept.activation_count),
            ])
    }

    /// Drain pending bids for submission to prefrontal cortex
    ///
    /// Returns all pending bids and clears the buffer.
    pub fn drain_bids(&mut self) -> Vec<AttentionBid> {
        let bids: Vec<_> = self.pending_bids.drain(..).collect();
        self.stats.bids_submitted += bids.len() as u64;
        bids
    }

    /// Get current pending bids (without draining)
    pub fn pending_bids(&self) -> &VecDeque<AttentionBid> {
        &self.pending_bids
    }

    /// Check if there are pending bids
    pub fn has_pending_bids(&self) -> bool {
        !self.pending_bids.is_empty()
    }

    /// Submit bids to prefrontal cortex and run cognitive cycle
    ///
    /// This is a convenience method that drains bids and runs them through
    /// the prefrontal cortex's cognitive cycle.
    pub fn submit_to_prefrontal(
        &mut self,
        prefrontal: &mut PrefrontalCortexActor,
        additional_bids: Vec<AttentionBid>,
    ) -> Option<AttentionBid> {
        let mut bids = self.drain_bids();
        bids.extend(additional_bids);

        let winner = prefrontal.cognitive_cycle(bids);

        // Track if our concept won
        if let Some(ref bid) = winner {
            if bid.source == "ConsciousnessWorldModel" {
                self.stats.bids_won_spotlight += 1;
            }
        }

        winner
    }

    /// Get bridge statistics
    pub fn stats(&self) -> &BridgeStats {
        &self.stats
    }

    /// Get bridge configuration
    pub fn config(&self) -> &BridgeConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: BridgeConfig) {
        self.config = config;
    }
}

/// Extension trait for ConsciousnessWorldModel to easily bridge to prefrontal
pub trait BridgeToPrefrontal {
    /// Generate attention bids from recent concept crystallizations
    fn generate_attention_bids(&self, last_count: usize) -> Vec<AttentionBid>;
}

impl BridgeToPrefrontal for ConsciousnessWorldModel {
    fn generate_attention_bids(&self, last_count: usize) -> Vec<AttentionBid> {
        let current_count = self.pending_concepts.len();
        if current_count <= last_count {
            return Vec::new();
        }

        let stats = self.stats();
        let new_concepts = &self.pending_concepts[last_count..];

        new_concepts
            .iter()
            .map(|concept| {
                let activation_bonus = (concept.activation_count as f32 * 0.01).min(0.3);
                let salience = (0.6 + activation_bonus).clamp(0.0, 1.0);

                let consciousness_factor = stats.consciousness_level as f32 * 0.2;
                let newness_bonus = if concept.activation_count <= 1 {
                    0.2
                } else {
                    0.0
                };
                let urgency = (0.5 + consciousness_factor + newness_bonus).clamp(0.0, 1.0);

                let emotion = if concept.activation_count <= 1 {
                    EmotionalValence::Positive
                } else if concept.name.is_some() {
                    EmotionalValence::Neutral
                } else {
                    EmotionalValence::Negative
                };

                let name_str = concept.name.as_deref().unwrap_or("unnamed");
                let content = format!(
                    "Crystallized Concept [{}]: name='{}', activations={}",
                    concept.uid, name_str, concept.activation_count
                );

                AttentionBid::new("ConsciousnessWorldModel", content)
                    .with_salience(salience)
                    .with_urgency(urgency)
                    .with_emotion(emotion)
                    .with_tags(vec!["concept".to_string(), "crystallized".to_string()])
            })
            .collect()
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
        let bridge = ConsciousnessBridge::default();
        assert_eq!(bridge.pending_bids.len(), 0);
        assert_eq!(bridge.stats.concepts_converted, 0);
    }

    #[test]
    fn test_bridge_sync_with_concepts() {
        let mut bridge = ConsciousnessBridge::default();
        let config = WorldModelConfig {
            min_training_samples: 1,
            ..Default::default()
        };
        let mut world_model = ConsciousnessWorldModel::new(config);

        // Generate some transitions to potentially crystallize concepts
        let state1 = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.8);
        let state2 = LatentConsciousnessState::from_observables(0.6, 0.6, 0.6, 0.8);

        for _i in 0..50 {
            let transition = ConsciousnessTransition {
                from_state: state1.clone(),
                action: ConsciousnessAction::new("focus", ActionType::Integrate),
                to_state: state2.clone(),
                reward: 0.1,
                timestamp: 0,
                surprise: 0.0,
            };
            world_model.observe_transition(transition);
        }

        // Sync bridge
        bridge.sync(&world_model);

        // Bridge should have tracked the concepts
        assert!(bridge.stats.peak_consciousness > 0.0);
    }

    #[test]
    fn test_concept_to_bid_conversion() {
        let concept = CrystalizedConcept {
            uid: "test-concept-123".to_string(),
            hypervector: vec![0.5; 32],
            name: None,
            description: None,
            crystallized_at: 12345,
            activation_count: 1,
            attractor_signature: vec![0.3; 16],
            primitive_birth: Vec::new(),
        };

        let stats = WorldModelStats {
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

        let mut bridge = ConsciousnessBridge::default();
        let bid = bridge.concept_to_bid(&concept, &stats);

        assert_eq!(bid.source, "ConsciousnessWorldModel");
        assert!(bid.content.contains("test-concept-123"));
        assert!(bid.salience > 0.5);
        assert!(bid.urgency > 0.5);
        assert_eq!(bid.emotion, EmotionalValence::Positive); // New concept
    }

    #[test]
    fn test_drain_bids() {
        let mut bridge = ConsciousnessBridge::default();

        // Manually add some bids
        let bid = AttentionBid::new("test", "test content");
        bridge.pending_bids.push_back(bid);

        assert!(bridge.has_pending_bids());

        let drained = bridge.drain_bids();
        assert_eq!(drained.len(), 1);
        assert!(!bridge.has_pending_bids());
        assert_eq!(bridge.stats.bids_submitted, 1);
    }
}