// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Social Fabric Manager — Resonance-Based Content CognitiveSubsystem
//!
//! **Requires**: `feature = "social-fabric"`
//!
//! Updates the `ResonanceGraph` from SwarmManager events and provides
//! neuromodulatory feedback based on social connection quality.
//!
//! # Science
//! - Woolley, A. et al. (2010). Evidence for a collective intelligence factor
//!
//! # Interval
//! 31 — co-prime with {7, 11, 13, 19, 23, 29, 37, 41, 43, 47, 53, 67}

use crate::cognitive_loop::subsystem_trait::{CognitiveSubsystem, CycleSnapshot, SubsystemOutput};
use crate::swarm::resonance_graph::{ContentRef, ResonanceGraph, SocialFabricTelemetry};

/// Neuromodulatory gain: high resonance -> oxytocin (positive valence).
const RESONANCE_OXY_GAIN: f64 = 0.03;

/// Neuromodulatory gain: resonance drop -> NE (arousal).
const RESONANCE_DROP_NE_GAIN: f64 = 0.04;

/// Neuromodulatory gain: diversity -> DA (curiosity).
const DIVERSITY_DA_GAIN: f64 = 0.02;

/// Resonance drop threshold (fraction of mean resonance loss).
const RESONANCE_DROP_THRESHOLD: f64 = 0.15;

/// A social fabric event from the swarm.
#[derive(Debug, Clone)]
pub enum SocialEvent {
    /// New content received from a peer.
    ContentReceived(ContentRef),
    /// Affective synchronization with a peer.
    AffectiveSync {
        peer_id: String,
        valence: f32,
        arousal: f32,
    },
}

/// Social Fabric Manager — integrates resonance graph into cognitive loop.
pub struct SocialFabricManager {
    /// The resonance graph.
    graph: ResonanceGraph,
    /// Pending events.
    pending_events: Vec<SocialEvent>,
    /// Whether enabled.
    enabled: bool,
    /// Previous mean resonance for drop detection.
    prev_mean_resonance: f64,
    /// Last telemetry.
    last_telemetry: SocialFabricTelemetry,
}

impl SocialFabricManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 31;

    /// Create a new SocialFabricManager.
    pub fn new(enabled: bool) -> Self {
        Self {
            graph: ResonanceGraph::new(),
            pending_events: Vec::new(),
            enabled,
            prev_mean_resonance: 0.5,
            last_telemetry: SocialFabricTelemetry::default(),
        }
    }

    /// Inject a social event for processing.
    pub fn inject_event(&mut self, event: SocialEvent) {
        if self.enabled {
            self.pending_events.push(event);
        }
    }

    /// Access the resonance graph.
    pub fn graph(&self) -> &ResonanceGraph {
        &self.graph
    }

    /// Mutable access.
    pub fn graph_mut(&mut self) -> &mut ResonanceGraph {
        &mut self.graph
    }

    /// Last telemetry snapshot.
    pub fn telemetry(&self) -> &SocialFabricTelemetry {
        &self.last_telemetry
    }

    fn process_events(&mut self) {
        for event in self.pending_events.drain(..) {
            match event {
                SocialEvent::ContentReceived(content) => {
                    self.graph.record_consumption(&content.source_peer);
                    self.graph.add_content(content);
                }
                SocialEvent::AffectiveSync { .. } => {
                    // Affective sync handled by SwarmManager; tracked here for resonance
                }
            }
        }
    }
}

impl CognitiveSubsystem for SocialFabricManager {
    fn name(&self) -> &'static str {
        "social_fabric_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, _snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        if !self.enabled {
            return output;
        }

        self.process_events();

        // Update our consciousness state in the resonance graph
        // (Using compressed_state as proxy for HDV embedding)
        // In production, this would use the actual BinaryHV from the cycle

        let mean_resonance = self.graph.mean_resonance();
        let diversity = self.graph.diversity();
        let echo_risk = self.graph.echo_chamber_risk();

        // Neuromod: high resonance -> oxytocin (calm connection)
        if mean_resonance > crate::cognitive_loop::thresholds::SOCIAL_RESONANCE_HIGH_THRESHOLD {
            output.valence_delta += (RESONANCE_OXY_GAIN
                * (mean_resonance
                    - crate::cognitive_loop::thresholds::SOCIAL_RESONANCE_HIGH_THRESHOLD)
                / crate::cognitive_loop::thresholds::SOCIAL_RESONANCE_RANGE)
                as f32;
        }

        // Neuromod: resonance drop -> NE (social alertness)
        let resonance_drop = self.prev_mean_resonance - mean_resonance;
        if resonance_drop > RESONANCE_DROP_THRESHOLD {
            output.arousal_delta += (RESONANCE_DROP_NE_GAIN * resonance_drop)
                .min(crate::cognitive_loop::thresholds::SOCIAL_RESONANCE_DROP_AROUSAL_CAP)
                as f32;
        }

        // Neuromod: diversity -> DA (curiosity boost)
        if diversity > crate::cognitive_loop::thresholds::SOCIAL_DIVERSITY_THRESHOLD {
            output.exploration_delta += DIVERSITY_DA_GAIN * diversity;
        }

        // Echo chamber warning
        if echo_risk > crate::cognitive_loop::thresholds::SOCIAL_ECHO_CHAMBER_THRESHOLD {
            output.flags |= crate::cognitive_loop::subsystem_trait::output_flags::ANOMALY_DETECTED;
        }

        self.prev_mean_resonance = mean_resonance;
        self.last_telemetry = self.graph.telemetry();

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        // Layout: [prev_mean_resonance: f64 = 8][enabled: u8 = 1]
        // Total: 9 bytes
        let mut data = Vec::with_capacity(9);
        data.extend_from_slice(&self.prev_mean_resonance.to_le_bytes());
        data.push(self.enabled as u8);
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        const MIN_SIZE: usize = 8 + 1; // 9
        if data.len() < MIN_SIZE {
            return Err(format!(
                "SocialFabricManager checkpoint too short: {} < {}",
                data.len(),
                MIN_SIZE
            ));
        }
        self.prev_mean_resonance = f64::from_le_bytes(
            data[0..8]
                .try_into()
                .map_err(|_| "SocialFabricManager: corrupt prev_mean_resonance bytes")?,
        );
        self.enabled = data[8] != 0;
        Ok(())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::BinaryHV;

    fn default_snapshot() -> CycleSnapshot {
        CycleSnapshot::default()
    }

    fn make_content(peer: &str) -> ContentRef {
        ContentRef {
            source_peer: peer.to_string(),
            content_hash: [0; 32],
            hdv_embedding: BinaryHV::random(rand::random::<u64>()),
            domain: "test".to_string(),
            created_at: 1000,
        }
    }

    #[test]
    fn test_social_fabric_creation() {
        let mgr = SocialFabricManager::new(true);
        assert_eq!(mgr.name(), "social_fabric_manager");
        assert_eq!(mgr.interval(), 31);
    }

    #[test]
    fn test_disabled_returns_neutral() {
        let mut mgr = SocialFabricManager::new(false);
        let output = mgr.process(&default_snapshot());
        assert_eq!(output.arousal_delta, 0.0);
    }

    #[test]
    fn test_content_event_processing() {
        let mut mgr = SocialFabricManager::new(true);
        mgr.inject_event(SocialEvent::ContentReceived(make_content("peer1")));
        mgr.process(&default_snapshot());
        assert_eq!(mgr.graph().peer_count(), 1);
    }

    #[test]
    fn test_telemetry_updates() {
        let mut mgr = SocialFabricManager::new(true);
        mgr.inject_event(SocialEvent::ContentReceived(make_content("p1")));
        mgr.inject_event(SocialEvent::ContentReceived(make_content("p2")));
        mgr.process(&default_snapshot());
        let t = mgr.telemetry();
        assert_eq!(t.peer_reach, 2);
    }

    #[test]
    fn test_disabled_ignores_events() {
        let mut mgr = SocialFabricManager::new(false);
        mgr.inject_event(SocialEvent::ContentReceived(make_content("p")));
        mgr.process(&default_snapshot());
        assert_eq!(mgr.graph().peer_count(), 0);
    }
}
