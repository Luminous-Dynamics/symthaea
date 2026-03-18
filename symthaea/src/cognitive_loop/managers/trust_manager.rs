//! Trust Manager — Web-of-Trust CognitiveSubsystem
//!
//! **Requires**: `feature = "mesh-trust"`
//!
//! Maintains the `TrustGraph` and provides neuromodulatory feedback based on
//! trust network state. Processes SwarmEvents (TrustVerified, PeerJoined)
//! to update the trust graph.
//!
//! # Science
//! - Zak, P. (2012). The Moral Molecule — oxytocin + trust neuroscience
//! - Dunbar, R. (1998). Social brain hypothesis — trust circle size limits
//!
//! # Interval
//! 29 — co-prime with {7, 11, 13, 19, 23, 37, 41, 43, 47, 53, 67}

use crate::cognitive_loop::subsystem_trait::{CognitiveSubsystem, CycleSnapshot, SubsystemOutput};
use crate::swarm::web_of_trust::{TrustGraph, TrustTelemetry};

/// Neuromodulatory gain for trust violation → NE (arousal spike).
/// Basis: Zak (2012) — betrayal triggers cortisol + NE cascade.
const TRUST_VIOLATION_NE_GAIN: f64 = 0.06;

/// Neuromodulatory gain for stable trust → oxytocin (positive valence).
/// Basis: Zak (2012) — stable trust relationships maintain oxytocin baseline.
const STABLE_TRUST_OXY_GAIN: f64 = 0.03;

/// Trust decay rate per tick (0.998 = ~0.2%/cycle).
const TRUST_DECAY_RATE: f64 = 0.998;

/// Minimum average trust for "healthy" network.
const HEALTHY_TRUST_THRESHOLD: f64 = 0.4;

/// A trust-related event from the swarm.
#[derive(Debug, Clone)]
pub enum TrustEvent {
    /// A peer's trust was verified via handshake.
    PeerVerified {
        peer_id: String,
        trust_level: f64,
        pq_verified: bool,
    },
    /// A peer joined the network.
    PeerJoined {
        peer_id: String,
        initial_trust: f64,
    },
    /// A peer left the network.
    PeerLeft {
        peer_id: String,
    },
    /// A trust violation was detected.
    TrustViolation {
        peer_id: String,
        reason: String,
    },
}

/// Trust Manager — integrates web-of-trust into the cognitive loop.
pub struct TrustManager {
    /// Our node identifier for trust graph operations.
    our_node_id: String,
    /// The trust graph.
    graph: TrustGraph,
    /// Pending events from swarm.
    pending_events: Vec<TrustEvent>,
    /// Whether the trust system is enabled.
    enabled: bool,
    /// Last telemetry snapshot.
    last_telemetry: TrustTelemetry,
    /// Violation count since last tick (for neuromod).
    violations_this_cycle: u32,
}

impl TrustManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 29;

    /// Create a new TrustManager.
    pub fn new(our_node_id: String, enabled: bool) -> Self {
        Self {
            our_node_id,
            graph: TrustGraph::new(),
            pending_events: Vec::new(),
            enabled,
            last_telemetry: TrustTelemetry::default(),
            violations_this_cycle: 0,
        }
    }

    /// Inject a trust event for processing on next cycle.
    pub fn inject_event(&mut self, event: TrustEvent) {
        if self.enabled {
            self.pending_events.push(event);
        }
    }

    /// Get the trust graph (read-only).
    pub fn graph(&self) -> &TrustGraph {
        &self.graph
    }

    /// Get mutable trust graph.
    pub fn graph_mut(&mut self) -> &mut TrustGraph {
        &mut self.graph
    }

    /// Get last telemetry snapshot.
    pub fn telemetry(&self) -> &TrustTelemetry {
        &self.last_telemetry
    }

    /// Get our node ID.
    pub fn our_node_id(&self) -> &str {
        &self.our_node_id
    }

    /// Process pending events.
    fn process_events(&mut self) {
        self.violations_this_cycle = 0;
        for event in self.pending_events.drain(..) {
            match event {
                TrustEvent::PeerVerified {
                    peer_id,
                    trust_level,
                    pq_verified,
                } => {
                    self.graph
                        .set_trust(&self.our_node_id, &peer_id, trust_level, pq_verified);
                }
                TrustEvent::PeerJoined {
                    peer_id,
                    initial_trust,
                } => {
                    self.graph
                        .set_trust(&self.our_node_id, &peer_id, initial_trust, false);
                }
                TrustEvent::PeerLeft { peer_id } => {
                    self.graph.remove_trust(&self.our_node_id, &peer_id);
                }
                TrustEvent::TrustViolation { peer_id, .. } => {
                    // Slash trust on violation
                    let current = self.graph.direct_trust(&self.our_node_id, &peer_id);
                    self.graph
                        .set_trust(&self.our_node_id, &peer_id, current * 0.5, false);
                    self.violations_this_cycle += 1;
                }
            }
        }
    }
}

impl CognitiveSubsystem for TrustManager {
    fn name(&self) -> &'static str {
        "trust_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, _snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        if !self.enabled {
            return output;
        }

        // Process pending events
        self.process_events();

        // Tick decay and anomaly detection
        let cycle = _snapshot.cycle_number;
        self.graph.tick(cycle, TRUST_DECAY_RATE);

        // Update telemetry
        self.last_telemetry = self.graph.telemetry();

        // Neuromod: trust violations → arousal (NE)
        if self.violations_this_cycle > 0 {
            output.arousal_delta +=
                (self.violations_this_cycle as f64 * TRUST_VIOLATION_NE_GAIN).min(0.1) as f32;
            output.valence_delta -= 0.02; // Negative valence from betrayal
        }

        // Neuromod: stable healthy trust → positive valence (oxytocin)
        if self.last_telemetry.avg_trust >= HEALTHY_TRUST_THRESHOLD
            && self.last_telemetry.anomaly_count == 0
        {
            output.valence_delta += STABLE_TRUST_OXY_GAIN as f32;
        }

        // Anomaly detection → escalate
        if self.last_telemetry.anomaly_count > 0 {
            output.arousal_delta += 0.03;
            output.flags |= crate::cognitive_loop::subsystem_trait::output_flags::ANOMALY_DETECTED;
        }

        output
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::subsystem_trait::CycleSnapshot;

    fn default_snapshot() -> CycleSnapshot {
        CycleSnapshot::default()
    }

    #[test]
    fn test_trust_manager_creation() {
        let mgr = TrustManager::new("node_0".to_string(), true);
        assert_eq!(mgr.name(), "trust_manager");
        assert_eq!(mgr.interval(), 29);
    }

    #[test]
    fn test_disabled_returns_neutral() {
        let mut mgr = TrustManager::new("node_0".to_string(), false);
        let output = mgr.process(&default_snapshot());
        assert_eq!(output.arousal_delta, 0.0);
    }

    #[test]
    fn test_peer_verified_event() {
        let mut mgr = TrustManager::new("node_0".to_string(), true);
        mgr.inject_event(TrustEvent::PeerVerified {
            peer_id: "peer_1".to_string(),
            trust_level: 0.9,
            pq_verified: true,
        });
        mgr.process(&default_snapshot());
        assert!(mgr.graph().direct_trust("node_0", "peer_1") > 0.8);
    }

    #[test]
    fn test_trust_violation_slashes() {
        let mut mgr = TrustManager::new("node_0".to_string(), true);
        mgr.inject_event(TrustEvent::PeerVerified {
            peer_id: "bad_peer".to_string(),
            trust_level: 0.8,
            pq_verified: false,
        });
        mgr.process(&default_snapshot());
        let before = mgr.graph().direct_trust("node_0", "bad_peer");

        mgr.inject_event(TrustEvent::TrustViolation {
            peer_id: "bad_peer".to_string(),
            reason: "invalid attestation".to_string(),
        });
        mgr.process(&default_snapshot());
        let after = mgr.graph().direct_trust("node_0", "bad_peer");
        assert!(
            after < before,
            "Trust should be slashed: {} -> {}",
            before,
            after
        );
    }

    #[test]
    fn test_violation_triggers_arousal() {
        let mut mgr = TrustManager::new("node_0".to_string(), true);
        mgr.inject_event(TrustEvent::PeerVerified {
            peer_id: "peer".to_string(),
            trust_level: 0.8,
            pq_verified: false,
        });
        mgr.process(&default_snapshot());

        mgr.inject_event(TrustEvent::TrustViolation {
            peer_id: "peer".to_string(),
            reason: "bad sig".to_string(),
        });
        let output = mgr.process(&default_snapshot());
        assert!(
            output.arousal_delta > 0.0,
            "Violation should increase arousal"
        );
    }

    #[test]
    fn test_peer_left_removes_trust() {
        let mut mgr = TrustManager::new("node_0".to_string(), true);
        mgr.inject_event(TrustEvent::PeerJoined {
            peer_id: "peer".to_string(),
            initial_trust: 0.7,
        });
        mgr.process(&default_snapshot());
        assert!(mgr.graph().direct_trust("node_0", "peer") > 0.0);

        mgr.inject_event(TrustEvent::PeerLeft {
            peer_id: "peer".to_string(),
        });
        mgr.process(&default_snapshot());
        assert_eq!(mgr.graph().direct_trust("node_0", "peer"), 0.0);
    }

    #[test]
    fn test_telemetry_updates() {
        let mut mgr = TrustManager::new("node_0".to_string(), true);
        mgr.inject_event(TrustEvent::PeerVerified {
            peer_id: "p1".to_string(),
            trust_level: 0.8,
            pq_verified: true,
        });
        mgr.process(&default_snapshot());
        let t = mgr.telemetry();
        assert_eq!(t.edge_count, 1);
        assert!(t.pq_fraction > 0.0);
    }

    #[test]
    fn test_disabled_ignores_events() {
        let mut mgr = TrustManager::new("node_0".to_string(), false);
        mgr.inject_event(TrustEvent::PeerJoined {
            peer_id: "peer".to_string(),
            initial_trust: 0.5,
        });
        mgr.process(&default_snapshot());
        assert_eq!(mgr.graph().edge_count(), 0);
    }
}
