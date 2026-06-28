// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Memory system, causal enhancement, episodic replay, and HDC projection accessors.

use anyhow::Result;

use crate::causal::{CausalGraph, DiscoveredRelationship};
use crate::cognitive_loop::CognitiveLoopService;

use super::super::{CognitiveGoal, EpisodicMemory, WorldModelBridge};

#[allow(dead_code)]
impl CognitiveLoopService {
    cognitive_accessors! {
        // ═══════════════════════════════════════════════════════════════════
        // MEMORY SYSTEM
        // ═══════════════════════════════════════════════════════════════════

        /// Get memory counts (short_term, long_term)
        pub fn memory_counts(&self) -> (usize, usize) { self.fep.episodic_memory.memory_count() }

        /// Get active goals
        pub fn active_goals(&self) -> Vec<&CognitiveGoal> { self.fep.goal_system.active_goals() }

        /// Get the world model bridge reference
        pub fn world_model(&self) -> &WorldModelBridge { &self.fep.world_model }

        /// Get abstract level state from world model
        pub fn world_model_abstract_state(&self) -> &[f32] { self.fep.world_model.abstract_state() }

        /// Get world model prediction errors at each level
        pub fn world_model_level_errors(&self) -> &[f32] { self.fep.world_model.level_errors() }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CAUSAL ENHANCEMENT ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════

    /// Get the current causal graph (if causal enhancement is enabled)
    pub fn causal_graph(&self) -> Option<&CausalGraph> {
        self.memory
            .causal_enhancer
            .as_ref()
            .map(|e| e.current_graph())
    }

    /// Get discovered causal relationships history
    pub fn causal_discoveries(&self) -> Option<&[DiscoveredRelationship]> {
        self.memory
            .causal_enhancer
            .as_ref()
            .map(|e| e.discovered_relationships())
    }

    /// Get causal enhancer statistics
    pub fn causal_stats(&self) -> Option<crate::causal::CausalLoopStats> {
        self.memory
            .causal_enhancer
            .as_ref()
            .map(|e| e.stats().clone())
    }

    /// Check if any causal structure has been discovered
    pub fn has_causal_structure(&self) -> bool {
        self.memory
            .causal_enhancer
            .as_ref()
            .map(|e| e.has_causal_structure())
            .unwrap_or(false)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // EPISODIC REPLAY ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════

    /// Get episodic replay statistics
    pub fn episodic_replay_stats(
        &self,
    ) -> Option<crate::memory::episodic_replay::EpisodicMemoryStats> {
        self.memory
            .episodic_persistence
            .replay
            .as_ref()
            .map(|r| r.stats())
    }

    /// Get the number of stored episodes
    pub fn episodic_replay_count(&self) -> usize {
        self.memory
            .episodic_persistence
            .replay
            .as_ref()
            .map(|r| r.len())
            .unwrap_or(0)
    }

    /// Get top N episodes by Phi (highest consciousness moments)
    pub fn top_phi_episodes(&self, n: usize) -> Vec<crate::memory::episodic_replay::Episode> {
        self.memory
            .episodic_persistence
            .replay
            .as_ref()
            .map(|r| r.get_top_episodes(n))
            .unwrap_or_default()
    }

    /// Project an embedding directly to HDC space, bypassing CfC temporal dynamics.
    pub fn project_embedding_to_hdc(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        let input_dim = self.config.cfc_config.input_dim;

        let compressed = if embedding.len() <= input_dim {
            let mut v = embedding.to_vec();
            v.resize(input_dim, 0.0);
            v
        } else {
            let step = embedding.len() / input_dim;
            embedding
                .iter()
                .step_by(step)
                .take(input_dim)
                .cloned()
                .collect::<Vec<_>>()
        };

        self.temporal_network
            .project_to_hdc_vec(&compressed)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "HDC projection not available (using CfC backend, not HdcLtcBridge)"
                )
            })
    }

    /// Get temporal coherence value (uses cycle-cached value when available)
    pub fn temporal_coherence(&self) -> f32 {
        self.carryover.history.cached_coherence.unwrap_or_else(|| {
            self.language_comm
                .voice_coherence
                .bridge
                .smoothed_coherence()
        })
    }

    /// Recall memories similar to input
    pub fn recall_memories(&mut self, query: &[f32], top_k: usize) -> Vec<(EpisodicMemory, f32)> {
        self.fep.episodic_memory.recall(query, top_k, 0.2)
    }

    /// Retrieve the current binding of proprioceptive state to semantic HDC space.
    pub fn get_proprioceptive_binding(&self) -> Option<Vec<f32>> {
        #[cfg(any(
            feature = "humanoid",
            feature = "helicopter",
            feature = "flight",
            feature = "vehicle",
            feature = "auv",
            feature = "manipulator",
            feature = "exoskeleton",
            feature = "surgical",
            feature = "orbital",
            feature = "quadruped",
            feature = "subterranean",
            feature = "infrastructure",
            feature = "scavenger",
            feature = "agribot",
            feature = "biota",
            feature = "clime",
            feature = "phone"
        ))]
        {
            if let Some(hv) = self
                .sensorimotor
                .embodiment_bridge
                .as_ref()
                .and_then(|b| b.last_perception_hv())
                .or_else(|| self.sensorimotor.last_proprioceptive_hv.clone())
            {
                return Some(hv.to_vec());
            }
        }

        None
    }

    /// Retrieve the current epistemic quality (sensorimotor accuracy) of the embodiment.
    pub fn get_epistemic_quality(&self) -> f64 {
        #[cfg(any(
            feature = "humanoid",
            feature = "helicopter",
            feature = "flight",
            feature = "vehicle",
            feature = "auv",
            feature = "manipulator",
            feature = "exoskeleton",
            feature = "surgical",
            feature = "orbital",
            feature = "quadruped",
            feature = "subterranean",
            feature = "infrastructure",
            feature = "scavenger",
            feature = "agribot",
            feature = "biota",
            feature = "clime",
            feature = "phone"
        ))]
        {
            if let Some(bridge) = self.sensorimotor.embodiment_bridge.as_ref() {
                return bridge.sensorimotor_accuracy() as f64;
            }
        }

        (1.0 - self.sensorimotor.somatic_bridge.systemic_stress()).clamp(0.0, 1.0)
    }

    /// Add a goal to the system
    pub fn add_goal(&mut self, id: &str, description: &str, priority: f32) {
        self.fep
            .goal_system
            .add_goal(CognitiveGoal::new(id, description, priority));
    }
}
