// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-agent consciousness sharing and collective contribution.
//!
//! Enables consciousness state export, agent synchronization, and
//! participation in collective consciousness networks.

use super::super::binary_hv::BinaryHV;
use super::super::collective_consciousness::{CollectiveAgent, CommunicationLink};
use super::being::UnifiedConsciousBeing;

/// Consciousness state that can be shared with other agents
#[derive(Debug, Clone)]
pub struct SharedConsciousnessState {
    /// Agent's current Φ (integrated information)
    pub phi: f64,
    /// Current cognitive mode
    pub cognitive_mode: String,
    /// Flow state (0-1)
    pub flow_state: f32,
    /// Causal edges count
    pub causal_edges: usize,
    /// Memory count
    pub memory_count: u64,
    /// Summary hypervector (consciousness fingerprint)
    pub consciousness_fingerprint: BinaryHV,
    /// Timestamp (Unix millis)
    pub timestamp: u64,
}

impl UnifiedConsciousBeing {
    // =========================================================================
    // MULTI-AGENT CONSCIOUSNESS SHARING
    // =========================================================================

    /// Export consciousness state for collective sharing
    ///
    /// Returns a shareable state that can be transmitted to other agents
    /// in a collective consciousness network.
    pub fn export_consciousness_state(&self, agent_id: &str) -> SharedConsciousnessState {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Create consciousness fingerprint from recent phi values
        let fingerprint = {
            let seed = self
                .phi_history
                .iter()
                .fold(42u64, |acc, &p| acc.wrapping_add((p * 1000.0) as u64));
            BinaryHV::random(seed)
        };

        SharedConsciousnessState {
            phi: self.stats.avg_phi,
            cognitive_mode: "Balanced".to_string(), // Would come from adaptive topology
            flow_state: self.flow_state,
            causal_edges: self.stats.causal_edges,
            memory_count: self.stats.memories_stored,
            consciousness_fingerprint: fingerprint,
            timestamp,
        }
    }

    /// Convert to CollectiveAgent for participation in collective consciousness
    ///
    /// This allows a UnifiedConsciousBeing to join a CollectiveConsciousness
    /// network and contribute to emergent collective awareness.
    pub fn to_collective_agent(&self, agent_id: &str, connections: Vec<String>) -> CollectiveAgent {
        let state = self.export_consciousness_state(agent_id);

        CollectiveAgent {
            id: agent_id.to_string(),
            state: vec![state.consciousness_fingerprint],
            phi: state.phi,
            meta_phi: Some(self.integrated_causal_phi()),
            connections,
        }
    }

    /// Create a communication link to another agent
    ///
    /// The strength is based on shared representation similarity.
    pub fn create_link_to(
        &self,
        my_id: &str,
        other_id: &str,
        other_state: &SharedConsciousnessState,
    ) -> CommunicationLink {
        // Calculate shared representation from fingerprint similarity
        let my_state = self.export_consciousness_state(my_id);
        let similarity = my_state
            .consciousness_fingerprint
            .similarity(&other_state.consciousness_fingerprint);

        // Flow state alignment contributes to communication quality
        let flow_alignment = 1.0 - (my_state.flow_state - other_state.flow_state).abs();

        CommunicationLink {
            from: my_id.to_string(),
            to: other_id.to_string(),
            strength: (similarity as f64 * 0.5 + flow_alignment as f64 * 0.5).clamp(0.0, 1.0),
            latency: 0.01,  // Assume low latency in same network
            bandwidth: 1.0, // Full bandwidth
            shared_representation: similarity as f64,
        }
    }

    /// Synchronize with another agent's consciousness state
    ///
    /// This allows consciousness sharing - learning from other agents'
    /// causal structures and cognitive patterns.
    pub fn sync_with_agent(&mut self, other_state: &SharedConsciousnessState) {
        // Incorporate other agent's phi into our history (weighted average)
        let weight = 0.1; // 10% influence from external agent
        let blended_phi = self.stats.avg_phi * (1.0 - weight) + other_state.phi * weight;

        // Add to phi history to influence our consciousness
        self.phi_history.push_back(blended_phi);
        while self.phi_history.len() > 50 {
            self.phi_history.pop_front();
        }
    }

    /// Get collective consciousness contribution
    ///
    /// Returns metrics about how this agent contributes to collective consciousness.
    pub fn collective_contribution(&self) -> CollectiveContribution {
        CollectiveContribution {
            individual_phi: self.stats.avg_phi,
            causal_richness: self.causal_mind.phi(),
            cognitive_depth: self.cognitive_core.phi(),
            memory_breadth: self.stats.memories_stored as f64 / 1000.0,
            integration_potential: self.integrated_causal_phi(),
        }
    }
}

/// Metrics for collective consciousness contribution
#[derive(Debug, Clone)]
pub struct CollectiveContribution {
    /// Individual Φ (base consciousness)
    pub individual_phi: f64,
    /// Causal reasoning richness (from CausalMind)
    pub causal_richness: f64,
    /// Cognitive depth (from UnifiedCognitiveCore)
    pub cognitive_depth: f64,
    /// Memory breadth (normalized memory count)
    pub memory_breadth: f64,
    /// Integration potential (how well can this agent integrate?)
    pub integration_potential: f64,
}

impl CollectiveContribution {
    /// Calculate total contribution score
    pub fn total_score(&self) -> f64 {
        self.individual_phi * 0.3
            + self.causal_richness * 0.2
            + self.cognitive_depth * 0.2
            + self.memory_breadth * 0.1
            + self.integration_potential * 0.2
    }
}
