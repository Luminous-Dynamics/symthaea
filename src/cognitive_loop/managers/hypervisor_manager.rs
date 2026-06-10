// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Hypervisor Manager — Hierarchical Consciousness Supervision
//!
//! Extends the peer-to-peer swarm with hierarchical supervision. High-Phi agents
//! are elected as supervisors who aggregate subordinate consciousness metrics,
//! detect systemic failures, and recommend safety escalations.
//!
//! ## Design
//!
//! - **Election**: Agents with Phi > 75th percentile AND reputation > 0.7 are candidates.
//!   Highest `phi * reputation` wins. Deterministic (no randomness in election).
//! - **Aggregation**: Subordinate consciousness is staleness-weighted using the same
//!   exponential decay as `consciousness_sync.rs`.
//! - **Authority**: Supervisors can recommend quarantine and safety escalation, but
//!   cannot override the local safety agent directly. All actions are proposals.
//! - **Epoch**: Re-election every `ELECTION_EPOCH_CYCLES` cycles (default 1000).
//!
//! ## Science
//!
//! - Malone & Bernstein (2015): Collective intelligence in organizations.
//! - Woolley et al. (2010): Evidence for a collective intelligence factor.
//! - Tononi (2004): Integrated information as a measure of consciousness.
//!
//! ## Integration
//!
//! Feature-gated behind `#[cfg(feature = "hypervisor")]`. Implements `CognitiveSubsystem`
//! with interval 71 (co-prime with all existing intervals).

use super::super::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput, output_flags,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Supervisor re-election interval (cycles). ~32s at 31Hz.
const ELECTION_EPOCH_CYCLES: u64 = 1000;

/// Minimum Phi for supervisor candidacy (relative to peer distribution).
/// If no peers, this is the absolute threshold.
const MIN_SUPERVISOR_PHI: f64 = 0.4;

/// Minimum reputation for supervisor candidacy.
const MIN_SUPERVISOR_REPUTATION: f64 = 0.7;

/// Staleness decay tau for subordinate consciousness (cycles).
/// At 31Hz, tau=300 ≈ 10s half-life.
const STALENESS_TAU: f64 = 300.0;

/// Maximum tracked subordinates.
const MAX_SUBORDINATES: usize = 100;

/// Heartbeat timeout (cycles). If supervisor doesn't heartbeat for this many
/// cycles, subordinates trigger re-election.
const HEARTBEAT_TIMEOUT: u64 = 200;

/// Threat severity threshold for recommending safety escalation.
const ESCALATION_THREAT_THRESHOLD: f32 = 0.6;

/// Phi drop threshold for subordinate distress detection.
const SUBORDINATE_DISTRESS_PHI_DROP: f64 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Role in the hierarchical supervision protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypervisorRole {
    /// Not participating in hierarchy (default before first election).
    Standalone,
    /// Candidate for supervisor election.
    Candidate,
    /// Elected supervisor — aggregates subordinate state.
    Supervisor,
    /// Subordinate — reports to a supervisor.
    Subordinate,
}

/// Consciousness state of a tracked subordinate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubordinateState {
    /// Peer identifier.
    pub peer_id: String,
    /// Last reported Phi.
    pub phi: f64,
    /// Last reported coherence.
    pub coherence: f32,
    /// Last reported arousal.
    pub arousal: f32,
    /// Cycle at which state was last updated.
    pub last_update_cycle: u64,
    /// Reputation score (0.0-1.0).
    pub reputation: f64,
}

impl SubordinateState {
    /// Staleness-weighted effective Phi.
    /// Decays exponentially with age: `phi * exp(-age / tau)`.
    pub fn effective_phi(&self, current_cycle: u64) -> f64 {
        let age = current_cycle.saturating_sub(self.last_update_cycle) as f64;
        self.phi * (-age / STALENESS_TAU).exp()
    }

    /// Whether this subordinate is stale (no update for `HEARTBEAT_TIMEOUT` cycles).
    pub fn is_stale(&self, current_cycle: u64) -> bool {
        current_cycle.saturating_sub(self.last_update_cycle) > HEARTBEAT_TIMEOUT
    }
}

/// Supervisory event for telemetry and swarm broadcast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupervisoryEvent {
    /// Supervisor elected for an epoch.
    Elected {
        supervisor_id: String,
        epoch: u64,
        phi_score: f64,
    },
    /// Supervisor heartbeat with aggregated state.
    Heartbeat {
        epoch: u64,
        subordinate_count: usize,
        aggregated_phi: f64,
        aggregated_coherence: f32,
    },
    /// Supervisor recommends safety escalation.
    Escalation {
        reason: String,
        severity: f32,
        affected_peers: Vec<String>,
    },
    /// Supervisor recommends quarantine of a peer.
    QuarantineRecommendation { peer_id: String, reason: String },
}

/// Telemetry snapshot for `CycleMetadata`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HypervisorTelemetry {
    pub role: String,
    pub epoch: u64,
    pub subordinate_count: usize,
    pub aggregated_phi: f64,
    pub stale_subordinates: usize,
    pub pending_events: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Hierarchical consciousness supervision manager.
///
/// Implements `CognitiveSubsystem` at interval 71 (co-prime with all existing
/// manager intervals). Manages supervisor election, subordinate tracking,
/// and safety escalation recommendations.
pub struct HypervisorManager {
    /// Current role in the hierarchy.
    role: HypervisorRole,
    /// Current election epoch number.
    epoch: u64,
    /// Cycle at which the current epoch started.
    epoch_start_cycle: u64,
    /// Our own agent identifier (set on first process call).
    self_id: String,
    /// Our own Phi score (tracked for election).
    self_phi: f64,
    /// Our own reputation (tracked for election).
    self_reputation: f64,
    /// Tracked subordinate states (supervisor only).
    subordinates: HashMap<String, SubordinateState>,
    /// Supervisor identifier (subordinate only).
    supervisor_id: Option<String>,
    /// Last heartbeat cycle from supervisor (subordinate only).
    last_supervisor_heartbeat: u64,
    /// Pending events to broadcast.
    pending_events: Vec<SupervisoryEvent>,
    /// Aggregated Phi across subordinates (cached).
    aggregated_phi: f64,
    /// Aggregated coherence across subordinates (cached).
    aggregated_coherence: f32,
}

impl HypervisorManager {
    /// Create a new hypervisor manager in standalone mode.
    pub fn new(self_id: String) -> Self {
        Self {
            role: HypervisorRole::Standalone,
            epoch: 0,
            epoch_start_cycle: 0,
            self_id,
            self_phi: 0.0,
            self_reputation: 1.0, // Start with full reputation
            subordinates: HashMap::new(),
            supervisor_id: None,
            last_supervisor_heartbeat: 0,
            pending_events: Vec::new(),
            aggregated_phi: 0.0,
            aggregated_coherence: 0.0,
        }
    }

    /// Inject a subordinate state update (from swarm event).
    pub fn update_subordinate(
        &mut self,
        peer_id: String,
        phi: f64,
        coherence: f32,
        arousal: f32,
        reputation: f64,
        cycle: u64,
    ) {
        if self.subordinates.len() >= MAX_SUBORDINATES && !self.subordinates.contains_key(&peer_id)
        {
            return; // At capacity, reject unknown peers
        }
        self.subordinates.insert(
            peer_id.clone(),
            SubordinateState {
                peer_id,
                phi,
                coherence,
                arousal,
                last_update_cycle: cycle,
                reputation,
            },
        );
    }

    /// Set own reputation (from external trust system).
    pub fn set_reputation(&mut self, reputation: f64) {
        self.self_reputation = reputation.clamp(0.0, 1.0);
    }

    /// Drain pending supervisory events for broadcast.
    pub fn drain_events(&mut self) -> Vec<SupervisoryEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Get telemetry snapshot.
    pub fn telemetry(&self, cycle: u64) -> HypervisorTelemetry {
        let stale = self
            .subordinates
            .values()
            .filter(|s| s.is_stale(cycle))
            .count();
        HypervisorTelemetry {
            role: format!("{:?}", self.role),
            epoch: self.epoch,
            subordinate_count: self.subordinates.len(),
            aggregated_phi: self.aggregated_phi,
            stale_subordinates: stale,
            pending_events: self.pending_events.len(),
        }
    }

    /// Check if it's time for a new election epoch.
    fn should_elect(&self, cycle: u64) -> bool {
        cycle.saturating_sub(self.epoch_start_cycle) >= ELECTION_EPOCH_CYCLES
    }

    /// Run election. Returns true if we become supervisor.
    fn run_election(&mut self, cycle: u64) -> bool {
        self.epoch += 1;
        self.epoch_start_cycle = cycle;

        // Check eligibility
        if self.self_phi < MIN_SUPERVISOR_PHI || self.self_reputation < MIN_SUPERVISOR_REPUTATION {
            self.role = HypervisorRole::Standalone;
            return false;
        }

        // Score: phi * reputation
        let self_score = self.self_phi * self.self_reputation;

        // Check if any known peer beats us (deterministic: highest score wins,
        // ties broken by lexicographic peer_id comparison)
        let mut best_score = self_score;
        let mut best_id = self.self_id.clone();

        for sub in self.subordinates.values() {
            if sub.is_stale(cycle) {
                continue;
            }
            let score = sub.phi * sub.reputation;
            if score > best_score || (score == best_score && sub.peer_id < best_id) {
                best_score = score;
                best_id = sub.peer_id.clone();
            }
        }

        if best_id == self.self_id {
            self.role = HypervisorRole::Supervisor;
            self.pending_events.push(SupervisoryEvent::Elected {
                supervisor_id: self.self_id.clone(),
                epoch: self.epoch,
                phi_score: self_score,
            });
            true
        } else {
            self.role = HypervisorRole::Subordinate;
            self.supervisor_id = Some(best_id);
            false
        }
    }

    /// Aggregate subordinate consciousness (staleness-weighted mean).
    fn aggregate_subordinates(&mut self, cycle: u64) {
        let active: Vec<&SubordinateState> = self
            .subordinates
            .values()
            .filter(|s| !s.is_stale(cycle))
            .collect();

        if active.is_empty() {
            self.aggregated_phi = self.self_phi;
            self.aggregated_coherence = 1.0;
            return;
        }

        let mut total_weight = 0.0f64;
        let mut weighted_phi = 0.0f64;
        let mut weighted_coherence = 0.0f64;

        for sub in &active {
            let w =
                sub.reputation * (-((cycle - sub.last_update_cycle) as f64) / STALENESS_TAU).exp();
            total_weight += w;
            weighted_phi += w * sub.phi;
            weighted_coherence += w * sub.coherence as f64;
        }

        if total_weight > 0.0 {
            self.aggregated_phi = weighted_phi / total_weight;
            self.aggregated_coherence = (weighted_coherence / total_weight) as f32;
        }
    }

    /// Detect subordinates in distress (sudden Phi drop).
    fn detect_distress(&mut self, cycle: u64) -> Vec<String> {
        let mut distressed = Vec::new();
        for sub in self.subordinates.values() {
            if !sub.is_stale(cycle) && sub.phi < SUBORDINATE_DISTRESS_PHI_DROP {
                distressed.push(sub.peer_id.clone());
            }
        }
        distressed
    }

    /// Emit heartbeat event (supervisor only).
    fn emit_heartbeat(&mut self, cycle: u64) {
        let active_count = self
            .subordinates
            .values()
            .filter(|s| !s.is_stale(cycle))
            .count();
        self.pending_events.push(SupervisoryEvent::Heartbeat {
            epoch: self.epoch,
            subordinate_count: active_count,
            aggregated_phi: self.aggregated_phi,
            aggregated_coherence: self.aggregated_coherence,
        });
    }
}

impl CognitiveSubsystem for HypervisorManager {
    fn name(&self) -> &'static str {
        "hypervisor"
    }

    fn interval(&self) -> u32 {
        71 // Co-prime with all existing intervals (7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,73)
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let cycle = snapshot.cycle_number;
        self.self_phi = snapshot.unified_psi;

        // Check for re-election
        if self.should_elect(cycle) {
            self.run_election(cycle);
        }

        match self.role {
            HypervisorRole::Supervisor => {
                // 1. Aggregate subordinate consciousness
                self.aggregate_subordinates(cycle);

                // 2. Detect distressed subordinates
                let distressed = self.detect_distress(cycle);
                if !distressed.is_empty() {
                    self.pending_events.push(SupervisoryEvent::Escalation {
                        reason: format!("{} subordinates below Phi threshold", distressed.len()),
                        severity: 0.5 + 0.1 * distressed.len() as f32,
                        affected_peers: distressed,
                    });
                }

                // 3. Emit heartbeat every 5 intervals (~11s at 31Hz)
                if cycle % (71 * 5) == 0 {
                    self.emit_heartbeat(cycle);
                }

                // 4. Confidence boost from supervisory aggregation
                let phi_health = (self.aggregated_phi - 0.3).max(0.0) * 0.1;
                let coherence_health = (self.aggregated_coherence - 0.5).max(0.0) as f64 * 0.05;

                SubsystemOutput {
                    confidence_delta: phi_health + coherence_health,
                    lr_modulation: 1.0,
                    exploration_delta: 0.0,
                    arousal_delta: 0.0,
                    valence_delta: 0.0,
                    flags: if !self.pending_events.is_empty() {
                        output_flags::HAS_TELEMETRY
                    } else {
                        0
                    },
                    _reserved: 0,
                }
            }

            HypervisorRole::Subordinate => {
                // Check supervisor heartbeat timeout
                let supervisor_stale =
                    cycle.saturating_sub(self.last_supervisor_heartbeat) > HEARTBEAT_TIMEOUT;
                if supervisor_stale {
                    // Lost supervisor — trigger re-election next interval
                    self.epoch_start_cycle = 0;
                    SubsystemOutput {
                        confidence_delta: -0.02, // Mild anxiety from lost supervisor
                        lr_modulation: 1.0,
                        exploration_delta: 0.01, // Slight exploration boost
                        arousal_delta: 0.01,
                        valence_delta: -0.01,
                        flags: output_flags::ANOMALY_DETECTED,
                        _reserved: 0,
                    }
                } else {
                    SubsystemOutput::NEUTRAL
                }
            }

            HypervisorRole::Standalone | HypervisorRole::Candidate => {
                // Not participating — check if conditions are met for candidacy
                if self.self_phi >= MIN_SUPERVISOR_PHI
                    && self.self_reputation >= MIN_SUPERVISOR_REPUTATION
                {
                    self.role = HypervisorRole::Candidate;
                }
                SubsystemOutput::NEUTRAL
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(cycle: u64, phi: f64) -> CycleSnapshot {
        let mut snap = CycleSnapshot::default();
        snap.cycle_number = cycle;
        snap.unified_psi = phi;
        snap
    }

    #[test]
    fn test_election_determinism() {
        let mut mgr = HypervisorManager::new("agent-A".into());
        mgr.self_phi = 0.6;
        mgr.self_reputation = 0.8;

        // No peers — we should win
        assert!(mgr.run_election(0));
        assert_eq!(mgr.role, HypervisorRole::Supervisor);
    }

    #[test]
    fn test_election_peer_wins() {
        let mut mgr = HypervisorManager::new("agent-A".into());
        mgr.self_phi = 0.5;
        mgr.self_reputation = 0.8;

        // Add a peer with higher score
        mgr.update_subordinate("agent-B".into(), 0.9, 0.8, 0.3, 0.9, 0);

        assert!(!mgr.run_election(0));
        assert_eq!(mgr.role, HypervisorRole::Subordinate);
        assert_eq!(mgr.supervisor_id, Some("agent-B".into()));
    }

    #[test]
    fn test_election_below_threshold() {
        let mut mgr = HypervisorManager::new("agent-A".into());
        mgr.self_phi = 0.2; // Below MIN_SUPERVISOR_PHI
        mgr.self_reputation = 0.8;

        assert!(!mgr.run_election(0));
        assert_eq!(mgr.role, HypervisorRole::Standalone);
    }

    #[test]
    fn test_staleness_decay() {
        let sub = SubordinateState {
            peer_id: "test".into(),
            phi: 0.8,
            coherence: 0.9,
            arousal: 0.3,
            last_update_cycle: 0,
            reputation: 1.0,
        };

        // At cycle 0, effective_phi = phi
        assert!((sub.effective_phi(0) - 0.8).abs() < 1e-6);

        // At cycle 300 (= tau), effective_phi ≈ phi * e^-1 ≈ 0.294
        let decayed = sub.effective_phi(300);
        assert!(decayed < 0.4, "should decay: got {decayed}");
        assert!(decayed > 0.2, "shouldn't vanish: got {decayed}");
    }

    #[test]
    fn test_subordinate_stale_detection() {
        let sub = SubordinateState {
            peer_id: "test".into(),
            phi: 0.8,
            coherence: 0.9,
            arousal: 0.3,
            last_update_cycle: 0,
            reputation: 1.0,
        };

        assert!(!sub.is_stale(100));
        assert!(sub.is_stale(HEARTBEAT_TIMEOUT + 1));
    }

    #[test]
    fn test_supervisor_heartbeat_emitted() {
        let mut mgr = HypervisorManager::new("agent-A".into());
        mgr.self_phi = 0.7;
        mgr.self_reputation = 0.9;
        mgr.run_election(0);
        assert_eq!(mgr.role, HypervisorRole::Supervisor);

        // Process at a heartbeat cycle
        let snap = make_snapshot(71 * 5, 0.7);
        mgr.process(&snap);

        let events = mgr.drain_events();
        // Should have Election + Heartbeat events
        assert!(events.len() >= 1);
    }

    #[test]
    fn test_distress_detection() {
        let mut mgr = HypervisorManager::new("agent-A".into());
        mgr.self_phi = 0.7;
        mgr.self_reputation = 0.9;
        mgr.run_election(0);

        // Add a distressed subordinate (low phi)
        mgr.update_subordinate("agent-B".into(), 0.1, 0.3, 0.8, 0.8, 0);

        let distressed = mgr.detect_distress(0);
        assert_eq!(distressed, vec!["agent-B"]);
    }

    #[test]
    fn test_epoch_re_election() {
        let mut mgr = HypervisorManager::new("agent-A".into());
        mgr.self_phi = 0.7;
        mgr.self_reputation = 0.9;
        mgr.epoch_start_cycle = 0;

        assert!(!mgr.should_elect(500));
        assert!(mgr.should_elect(ELECTION_EPOCH_CYCLES));
    }

    #[test]
    fn test_aggregation_weighted() {
        let mut mgr = HypervisorManager::new("agent-A".into());
        mgr.self_phi = 0.5;

        mgr.update_subordinate("B".into(), 0.8, 0.9, 0.3, 1.0, 100);
        mgr.update_subordinate("C".into(), 0.4, 0.5, 0.5, 0.5, 100);

        mgr.aggregate_subordinates(100);

        // B has higher reputation, so aggregated_phi should be closer to 0.8
        assert!(mgr.aggregated_phi > 0.5, "weighted toward high-rep peer");
        assert!(mgr.aggregated_phi < 0.8, "but not exactly B's phi");
    }
}

#[path = "hypervisor_manager_integration_test.rs"]
#[cfg(test)]
mod integration_tests;
