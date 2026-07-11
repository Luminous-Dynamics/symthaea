// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Web of Trust — Unified Consciousness-Aware Trust Graph
//!
//! A directed weighted trust graph that merges:
//! - Direct peer trust from handshake verification (Ed25519/ML-KEM)
//! - Transitive trust via shortest-path with per-hop decay (0.8/hop, ADR-001)
//! - Reputation from Mycelix ConsciousnessProfile
//! - Sybil detection via mass trust change anomaly
//!
//! # References
//! - Zak, P. (2012). The Moral Molecule — oxytocin + trust neuroscience
//! - Dunbar, R. (1998). Social brain hypothesis — trust circle limits

use std::collections::HashMap;

/// Maximum hops for transitive trust resolution.
pub const MAX_TRUST_HOPS: usize = 5;

/// Per-hop trust decay factor (ADR-001 from mycelix-mail).
pub const TRUST_HOP_DECAY: f64 = 0.8;

/// Threshold for Sybil anomaly detection: >30% mass trust change in one cycle.
pub const SYBIL_MASS_CHANGE_THRESHOLD: f64 = 0.3;

/// Trust below this is considered effectively zero.
pub const TRUST_FLOOR: f64 = 0.01;

/// Maximum edges per node (Dunbar-inspired limit).
pub const MAX_EDGES_PER_NODE: usize = 150;

/// A directed trust edge between two peers.
#[derive(Debug, Clone)]
pub struct TrustEdge {
    /// Source peer identifier.
    pub from: String,
    /// Target peer identifier.
    pub to: String,
    /// Direct trust score [0.0, 1.0] from handshake/attestation.
    pub direct_trust: f64,
    /// Whether this edge has been verified with post-quantum crypto.
    pub pq_verified: bool,
    /// Temporal decay factor (decreases over time without refresh).
    pub decay: f64,
    /// Cycle number when this edge was last verified.
    pub last_verified_cycle: u64,
}

impl TrustEdge {
    /// Effective trust = direct_trust × decay.
    pub fn effective_trust(&self) -> f64 {
        (self.direct_trust * self.decay).max(0.0)
    }
}

/// Unified trust score combining all sources.
#[derive(Debug, Clone, Default)]
pub struct UnifiedTrust {
    /// Direct trust from handshake verification.
    pub direct: f64,
    /// Transitive trust via shortest path.
    pub transitive: f64,
    /// Reputation from ConsciousnessProfile.
    pub reputation: f64,
    /// Consciousness tier (0=Observer, 1=Participant, 2=Citizen, 3=Steward, 4=Guardian).
    pub consciousness_tier: u8,
    /// Composite trust score (weighted combination).
    pub composite: f64,
}

impl UnifiedTrust {
    /// Compute composite trust from components.
    ///
    /// Weights: direct 40%, transitive 25%, reputation 25%, tier bonus 10%.
    pub fn compute(direct: f64, transitive: f64, reputation: f64, consciousness_tier: u8) -> Self {
        let tier_bonus = (consciousness_tier as f64 / 4.0).clamp(0.0, 1.0);
        let composite = (direct * 0.4 + transitive * 0.25 + reputation * 0.25 + tier_bonus * 0.1)
            .clamp(0.0, 1.0);
        Self {
            direct,
            transitive,
            reputation,
            consciousness_tier,
            composite,
        }
    }
}

/// Trust graph telemetry.
#[derive(Debug, Clone, Default)]
pub struct TrustTelemetry {
    /// Average trust across all edges.
    pub avg_trust: f64,
    /// Graph density (edges / max possible edges).
    pub graph_density: f64,
    /// Number of Sybil anomalies detected.
    pub anomaly_count: u32,
    /// Fraction of edges with PQ verification.
    pub pq_fraction: f64,
    /// Total nodes in graph.
    pub node_count: usize,
    /// Total edges in graph.
    pub edge_count: usize,
}

/// Sybil anomaly record.
#[derive(Debug, Clone)]
pub struct SybilAnomaly {
    /// Node that triggered the anomaly.
    pub node: String,
    /// Fraction of trust changed.
    pub change_fraction: f64,
    /// Cycle when detected.
    pub cycle: u64,
}

/// Directed weighted trust graph with transitive resolution and Sybil detection.
pub struct TrustGraph {
    /// Adjacency list: node_id → outgoing edges.
    edges: HashMap<String, Vec<TrustEdge>>,
    /// Known nodes set.
    nodes: std::collections::HashSet<String>,
    /// Detected anomalies (ring buffer, cap 32).
    anomalies: Vec<SybilAnomaly>,
    /// Previous trust snapshot for anomaly detection.
    prev_trust_snapshot: HashMap<String, f64>,
    /// Current cycle number.
    current_cycle: u64,
}

impl TrustGraph {
    /// Create an empty trust graph.
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            nodes: std::collections::HashSet::new(),
            anomalies: Vec::new(),
            prev_trust_snapshot: HashMap::new(),
            current_cycle: 0,
        }
    }

    /// Add or update a trust edge.
    pub fn set_trust(&mut self, from: &str, to: &str, direct_trust: f64, pq_verified: bool) {
        let trust = direct_trust.clamp(0.0, 1.0);
        self.nodes.insert(from.to_string());
        self.nodes.insert(to.to_string());

        let outgoing = self.edges.entry(from.to_string()).or_default();

        // Check Dunbar limit
        if outgoing.len() >= MAX_EDGES_PER_NODE {
            // Evict lowest-trust edge
            if let Some(min_idx) = outgoing
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.effective_trust()
                        .partial_cmp(&b.effective_trust())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                if outgoing[min_idx].effective_trust() < trust {
                    outgoing.swap_remove(min_idx);
                } else {
                    return; // New edge is weaker than all existing
                }
            }
        }

        if let Some(edge) = outgoing.iter_mut().find(|e| e.to == to) {
            edge.direct_trust = trust;
            edge.pq_verified = edge.pq_verified || pq_verified;
            edge.decay = 1.0;
            edge.last_verified_cycle = self.current_cycle;
        } else {
            outgoing.push(TrustEdge {
                from: from.to_string(),
                to: to.to_string(),
                direct_trust: trust,
                pq_verified,
                decay: 1.0,
                last_verified_cycle: self.current_cycle,
            });
        }
    }

    /// Remove a trust edge.
    pub fn remove_trust(&mut self, from: &str, to: &str) {
        if let Some(outgoing) = self.edges.get_mut(from) {
            outgoing.retain(|e| e.to != to);
        }
    }

    /// Get direct trust between two nodes.
    pub fn direct_trust(&self, from: &str, to: &str) -> f64 {
        self.edges
            .get(from)
            .and_then(|edges| edges.iter().find(|e| e.to == to))
            .map(|e| e.effective_trust())
            .unwrap_or(0.0)
    }

    /// Compute transitive trust via BFS shortest path with per-hop decay.
    ///
    /// Uses Dijkstra-like algorithm where edge weight = -log(trust) and
    /// max hops = MAX_TRUST_HOPS.
    pub fn transitive_trust(&self, from: &str, to: &str, max_hops: usize) -> f64 {
        if from == to {
            return 1.0;
        }

        // Direct edge check
        let direct = self.direct_trust(from, to);
        if direct > TRUST_FLOOR {
            return direct;
        }

        // BFS with trust propagation
        let max_hops = max_hops.min(MAX_TRUST_HOPS);
        let mut best_trust: HashMap<&str, f64> = HashMap::new();
        let mut queue: Vec<(&str, f64, usize)> = vec![(from, 1.0, 0)];
        best_trust.insert(from, 1.0);

        while let Some((node, trust_so_far, hops)) = queue.pop() {
            if hops >= max_hops {
                continue;
            }
            if let Some(outgoing) = self.edges.get(node) {
                for edge in outgoing {
                    let new_trust = trust_so_far * edge.effective_trust() * TRUST_HOP_DECAY;
                    if new_trust < TRUST_FLOOR {
                        continue;
                    }
                    let existing = best_trust.get(edge.to.as_str()).copied().unwrap_or(0.0);
                    if new_trust > existing {
                        best_trust.insert(&edge.to, new_trust);
                        if edge.to != to {
                            queue.push((&edge.to, new_trust, hops + 1));
                        }
                    }
                }
            }
        }

        best_trust.get(to).copied().unwrap_or(0.0)
    }

    /// Compute unified trust for a target node from our perspective.
    pub fn unified_trust(
        &self,
        from: &str,
        to: &str,
        reputation: f64,
        consciousness_tier: u8,
    ) -> UnifiedTrust {
        let direct = self.direct_trust(from, to);
        let transitive = self.transitive_trust(from, to, MAX_TRUST_HOPS);
        UnifiedTrust::compute(direct, transitive, reputation, consciousness_tier)
    }

    /// Apply temporal decay to all edges and detect Sybil anomalies.
    ///
    /// Call once per cycle to age trust relationships.
    pub fn tick(&mut self, cycle: u64, decay_rate: f64) {
        self.current_cycle = cycle;
        let decay = decay_rate.clamp(0.9, 1.0);

        // Build current trust snapshot for anomaly detection
        let mut current_snapshot: HashMap<String, f64> = HashMap::new();

        for (from, edges) in &mut self.edges {
            for edge in edges.iter_mut() {
                edge.decay *= decay;
            }
            // Remove edges below floor
            edges.retain(|e| e.effective_trust() >= TRUST_FLOOR);

            // Aggregate trust for anomaly detection
            let total: f64 = edges.iter().map(|e| e.effective_trust()).sum();
            current_snapshot.insert(from.clone(), total);
        }

        // Sybil detection: check for sudden mass trust changes
        for (node, current_total) in &current_snapshot {
            if let Some(&prev_total) = self.prev_trust_snapshot.get(node) {
                if prev_total > 0.0 {
                    let change = (current_total - prev_total).abs() / prev_total;
                    if change > SYBIL_MASS_CHANGE_THRESHOLD {
                        let anomaly = SybilAnomaly {
                            node: node.clone(),
                            change_fraction: change,
                            cycle,
                        };
                        if self.anomalies.len() >= 32 {
                            self.anomalies.remove(0);
                        }
                        self.anomalies.push(anomaly);
                    }
                }
            }
        }

        self.prev_trust_snapshot = current_snapshot;
    }

    /// Number of detected anomalies.
    pub fn anomaly_count(&self) -> usize {
        self.anomalies.len()
    }

    /// Recent anomalies.
    pub fn anomalies(&self) -> &[SybilAnomaly] {
        &self.anomalies
    }

    /// Whether `node` has ever been flagged by Sybil-anomaly detection
    /// (`tick`'s sudden-mass-trust-change check).
    ///
    /// This is a coarse, binary signal — "has this node ever tripped
    /// detection," not a graduated trust/reputation score. Used to gate
    /// content admission (e.g. cross-agent cultural-memory sharing) without
    /// requiring positive trust, which would create a cold-start deadlock:
    /// an unknown, never-before-seen node has `direct_trust`/
    /// `transitive_trust` of exactly 0.0, identical to a known bad actor, so
    /// a positive-trust admission requirement would block every peer's
    /// first contact forever. Rejecting only already-flagged nodes stops
    /// caught bad actors without punishing legitimate newcomers.
    pub fn is_flagged(&self, node: &str) -> bool {
        self.anomalies.iter().any(|a| a.node == node)
    }

    /// Total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }

    /// Generate telemetry snapshot.
    pub fn telemetry(&self) -> TrustTelemetry {
        let edge_count = self.edge_count();
        let node_count = self.node_count();

        let (trust_sum, pq_count) =
            self.edges
                .values()
                .flat_map(|v| v.iter())
                .fold((0.0, 0usize), |(sum, pq), e| {
                    (
                        sum + e.effective_trust(),
                        pq + if e.pq_verified { 1 } else { 0 },
                    )
                });

        let avg_trust = if edge_count > 0 {
            trust_sum / edge_count as f64
        } else {
            0.0
        };
        let pq_fraction = if edge_count > 0 {
            pq_count as f64 / edge_count as f64
        } else {
            0.0
        };
        let max_edges = if node_count > 1 {
            node_count * (node_count - 1)
        } else {
            1
        };
        let graph_density = edge_count as f64 / max_edges as f64;

        TrustTelemetry {
            avg_trust,
            graph_density,
            anomaly_count: self.anomalies.len() as u32,
            pq_fraction,
            node_count,
            edge_count,
        }
    }
}

impl Default for TrustGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let g = TrustGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_direct_trust() {
        let mut g = TrustGraph::new();
        g.set_trust("alice", "bob", 0.9, true);
        assert!((g.direct_trust("alice", "bob") - 0.9).abs() < 0.01);
        assert_eq!(g.direct_trust("bob", "alice"), 0.0); // Directed
    }

    #[test]
    fn test_transitive_trust() {
        let mut g = TrustGraph::new();
        g.set_trust("alice", "bob", 0.9, false);
        g.set_trust("bob", "carol", 0.8, false);
        let t = g.transitive_trust("alice", "carol", 5);
        // Expected: 0.9 * 0.8 (hop decay) * 0.8 * 0.8 (hop decay) = 0.4608
        assert!(t > 0.3 && t < 0.6, "Got {}", t);
    }

    #[test]
    fn test_transitive_trust_max_hops() {
        let mut g = TrustGraph::new();
        g.set_trust("a", "b", 0.9, false);
        g.set_trust("b", "c", 0.9, false);
        g.set_trust("c", "d", 0.9, false);
        // With max_hops=1, can't reach d from a
        assert!(g.transitive_trust("a", "d", 1) < TRUST_FLOOR);
        // With max_hops=3, can reach
        assert!(g.transitive_trust("a", "d", 3) > TRUST_FLOOR);
    }

    #[test]
    fn test_self_trust_is_one() {
        let g = TrustGraph::new();
        assert_eq!(g.transitive_trust("alice", "alice", 5), 1.0);
    }

    #[test]
    fn test_trust_update() {
        let mut g = TrustGraph::new();
        g.set_trust("alice", "bob", 0.5, false);
        assert!((g.direct_trust("alice", "bob") - 0.5).abs() < 0.01);
        g.set_trust("alice", "bob", 0.9, true);
        assert!((g.direct_trust("alice", "bob") - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_trust_removal() {
        let mut g = TrustGraph::new();
        g.set_trust("alice", "bob", 0.9, false);
        g.remove_trust("alice", "bob");
        assert_eq!(g.direct_trust("alice", "bob"), 0.0);
    }

    #[test]
    fn test_temporal_decay() {
        let mut g = TrustGraph::new();
        g.set_trust("alice", "bob", 1.0, false);
        // Apply decay for 10 cycles
        for i in 1..=10 {
            g.tick(i, 0.95);
        }
        let trust = g.direct_trust("alice", "bob");
        assert!(trust < 1.0 && trust > 0.0, "Trust should decay: {}", trust);
    }

    #[test]
    fn test_sybil_detection() {
        let mut g = TrustGraph::new();
        // Establish baseline
        g.set_trust("alice", "bob", 0.5, false);
        g.set_trust("alice", "carol", 0.5, false);
        g.tick(1, 0.999);

        // Mass trust change: alice suddenly trusts 10 new nodes
        for i in 0..10 {
            g.set_trust("alice", &format!("sybil_{}", i), 0.9, false);
        }
        g.tick(2, 0.999);

        assert!(g.anomaly_count() > 0, "Should detect Sybil anomaly");
    }

    #[test]
    fn test_is_flagged_reflects_real_sybil_detection() {
        let mut g = TrustGraph::new();
        g.set_trust("alice", "bob", 0.5, false);
        g.set_trust("alice", "carol", 0.5, false);
        g.tick(1, 0.999);
        for i in 0..10 {
            g.set_trust("alice", &format!("sybil_{i}"), 0.9, false);
        }
        g.tick(2, 0.999);

        assert!(
            !g.anomalies().is_empty(),
            "test setup must trigger detection"
        );
        let flagged_node = g.anomalies()[0].node.clone();

        assert!(
            g.is_flagged(&flagged_node),
            "the node the real detection algorithm flagged must read as flagged"
        );
        assert!(
            !g.is_flagged("someone_never_involved"),
            "an unrelated node must not be flagged"
        );
    }

    #[test]
    fn test_unified_trust() {
        let mut g = TrustGraph::new();
        g.set_trust("alice", "bob", 0.8, true);
        let ut = g.unified_trust("alice", "bob", 0.7, 2);
        assert!(ut.composite > 0.5);
        assert!(ut.direct > 0.0);
        assert_eq!(ut.consciousness_tier, 2);
    }

    #[test]
    fn test_telemetry() {
        let mut g = TrustGraph::new();
        g.set_trust("alice", "bob", 0.8, true);
        g.set_trust("bob", "carol", 0.6, false);
        let t = g.telemetry();
        assert_eq!(t.node_count, 3);
        assert_eq!(t.edge_count, 2);
        assert!(t.avg_trust > 0.0);
        assert!(t.pq_fraction > 0.0 && t.pq_fraction < 1.0);
    }

    #[test]
    fn test_trust_clamping() {
        let mut g = TrustGraph::new();
        g.set_trust("a", "b", 1.5, false); // Over 1.0
        assert!((g.direct_trust("a", "b") - 1.0).abs() < 0.01);
        g.set_trust("a", "c", -0.5, false); // Under 0.0
        assert_eq!(g.direct_trust("a", "c"), 0.0);
    }

    #[test]
    fn test_pq_verified_sticky() {
        let mut g = TrustGraph::new();
        g.set_trust("a", "b", 0.5, true);
        g.set_trust("a", "b", 0.7, false); // Update without PQ
        // PQ flag should still be true (sticky)
        let edges = g.edges.get("a").unwrap();
        assert!(edges[0].pq_verified);
    }

    #[test]
    fn test_edge_count_tracking() {
        let mut g = TrustGraph::new();
        g.set_trust("a", "b", 0.5, false);
        g.set_trust("a", "c", 0.5, false);
        g.set_trust("b", "c", 0.5, false);
        assert_eq!(g.edge_count(), 3);
        g.remove_trust("a", "b");
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_dunbar_limit() {
        let mut g = TrustGraph::new();
        for i in 0..MAX_EDGES_PER_NODE {
            g.set_trust("hub", &format!("peer_{}", i), 0.5, false);
        }
        assert_eq!(g.edges.get("hub").unwrap().len(), MAX_EDGES_PER_NODE);
        // Adding one more with higher trust should evict lowest
        g.set_trust("hub", "new_peer", 0.9, false);
        assert!(g.edges.get("hub").unwrap().len() <= MAX_EDGES_PER_NODE);
    }

    #[test]
    fn test_default_impl() {
        let g = TrustGraph::default();
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_anomaly_ring_buffer() {
        let mut g = TrustGraph::new();
        // Fill anomaly buffer
        for i in 0..40 {
            g.anomalies.push(SybilAnomaly {
                node: format!("node_{}", i),
                change_fraction: 0.5,
                cycle: i,
            });
            if g.anomalies.len() > 32 {
                g.anomalies.remove(0);
            }
        }
        assert_eq!(g.anomaly_count(), 32);
    }

    #[test]
    fn test_unified_trust_compute() {
        let ut = UnifiedTrust::compute(0.0, 0.0, 0.0, 0);
        assert_eq!(ut.composite, 0.0);

        let ut = UnifiedTrust::compute(1.0, 1.0, 1.0, 4);
        assert!((ut.composite - 1.0).abs() < 0.01);
    }
}
