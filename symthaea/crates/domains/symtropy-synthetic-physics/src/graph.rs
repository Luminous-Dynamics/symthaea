// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Core graph data structures for the Synthetic Physics Lab.
//!
//! [`SyntheticGraph`] is the mutable substrate that update rules act on.
//! It wraps [`petgraph`] for efficient graph operations.

use std::collections::HashMap;

use petgraph::{
    algo::{connected_components, dijkstra},
    graph::{NodeIndex, UnGraph},
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{UpdateOutcome, circuit_breakers::GraphSafetyGuards, update_rules::UpdateRule};

/// A node in the synthetic graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: u64,
    /// Optional HDC-style hypervector seed (for Phase 4 gauge transport).
    /// Stored as a compact binary mask in v0.1.
    pub hdc_seed: u64,
    /// Tag for semantic type (used in visualization).
    pub semantic_type: NodeSemanticType,
}

/// A directed edge in the synthetic graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: u64,
    /// Edge weight — interpreted as resistance / distance proxy.
    pub weight: f64,
    /// Confidence in this edge (evidence strength).
    pub confidence: f64,
    /// Optional edge transform seed for Phase 4 gauge-HDC holonomy.
    pub transform_seed: u64,
}

/// Semantic node types for visualization color mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeSemanticType {
    /// Generic graph node (white/clean).
    Generic,
    /// High-centrality hub (amber — durable, chronicle-like).
    Hub,
    /// Boundary / membrane node (violet — memory, topology boundary).
    Boundary,
    /// Quarantined node (red — guard violation).
    Quarantined,
}

/// The mutable synthetic graph substrate.
pub struct SyntheticGraph {
    /// Internal petgraph undirected graph.
    pub(crate) graph: UnGraph<GraphNode, GraphEdge>,
    /// Deterministic RNG for reproducibility.
    pub(crate) rng: StdRng,
    /// Seed used to initialize this graph.
    pub seed: u64,
    /// Current tick.
    pub tick: u64,
    /// Count of consecutive rejected updates (circuit breaker counter).
    pub consecutive_rejections: usize,
    /// Snapshot of graph state before last update (for rollback).
    rollback_snapshot: Option<RollbackSnapshot>,
}

/// Minimal snapshot for rollback (petgraph clone is cheap for small graphs).
struct RollbackSnapshot {
    graph: UnGraph<GraphNode, GraphEdge>,
}

impl SyntheticGraph {
    /// Create a new graph with `initial_nodes` fully randomized nodes, deterministic from `seed`.
    pub fn new_with_seed(seed: u64, initial_nodes: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut graph = UnGraph::new_undirected();

        // Create initial nodes
        let indices: Vec<NodeIndex> = (0..initial_nodes)
            .map(|i| {
                graph.add_node(GraphNode {
                    id: i as u64,
                    hdc_seed: rng.r#gen(),
                    semantic_type: NodeSemanticType::Generic,
                })
            })
            .collect();

        // Connect them as a sparse random graph (Erdős–Rényi p=0.2)
        let mut edge_id = 0u64;
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                if rng.gen_bool(0.2) {
                    graph.add_edge(
                        indices[i],
                        indices[j],
                        GraphEdge {
                            id: edge_id,
                            weight: rng.gen_range(0.1..1.0),
                            confidence: 1.0,
                            transform_seed: rng.r#gen(),
                        },
                    );
                    edge_id += 1;
                }
            }
        }

        Self {
            graph,
            rng,
            seed,
            tick: 0,
            consecutive_rejections: 0,
            rollback_snapshot: None,
        }
    }

    /// Total number of nodes.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Total number of edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Maximum degree of any node.
    pub fn max_degree(&self) -> u32 {
        self.graph
            .node_indices()
            .map(|n| self.graph.neighbors(n).count() as u32)
            .max()
            .unwrap_or(0)
    }

    /// Number of connected components (Betti-0 proxy).
    pub fn connected_components(&self) -> usize {
        connected_components(&self.graph)
    }

    /// Estimated average shortest path length (graph diameter approximation).
    pub fn estimated_diameter(&self) -> usize {
        if self.graph.node_count() == 0 {
            return 0;
        }
        // Sample from first node for O(V+E) estimate
        if let Some(start) = self.graph.node_indices().next() {
            let dists = dijkstra(&self.graph, start, None, |_| 1usize);
            *dists.values().max().unwrap_or(&0)
        } else {
            0
        }
    }

    /// Clustering coefficient (global, Watts-Strogatz style).
    pub fn clustering_coefficient(&self) -> f64 {
        let n = self.graph.node_count();
        if n == 0 {
            return 0.0;
        }
        let total: f64 = self
            .graph
            .node_indices()
            .map(|v| {
                let neighbors: Vec<NodeIndex> = self.graph.neighbors(v).collect();
                let k = neighbors.len();
                if k < 2 {
                    return 0.0;
                }
                let mut triangles = 0usize;
                for i in 0..neighbors.len() {
                    for j in (i + 1)..neighbors.len() {
                        if self.graph.contains_edge(neighbors[i], neighbors[j]) {
                            triangles += 1;
                        }
                    }
                }
                2.0 * triangles as f64 / (k * (k - 1)) as f64
            })
            .sum();
        total / n as f64
    }

    /// Estimate intrinsic dimension from graph structure (Menger curvature proxy).
    ///
    /// Uses the relationship: `dim ≈ log(N) / log(D)` where N = node count and
    /// D = estimated diameter. This is a rough proxy; proper dimension estimation
    /// will use correlation dimension in Phase 4.
    pub fn estimate_dimension(&self) -> f64 {
        let n = self.node_count();
        let d = self.estimated_diameter();
        if n <= 1 || d <= 1 {
            return 1.0;
        }
        (n as f64).ln() / (d as f64).ln()
    }

    /// Graph entropy (Shannon entropy of the degree distribution).
    pub fn degree_entropy(&self) -> f64 {
        let n = self.graph.node_count();
        if n == 0 {
            return 0.0;
        }
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for node in self.graph.node_indices() {
            let deg = self.graph.neighbors(node).count();
            *counts.entry(deg).or_insert(0) += 1;
        }
        -counts
            .values()
            .map(|&c| {
                let p = c as f64 / n as f64;
                if p > 0.0 { p * p.ln() } else { 0.0 }
            })
            .sum::<f64>()
    }

    /// Apply an update rule, checking guards. Returns the outcome.
    pub fn apply_update(
        &mut self,
        rule: &UpdateRule,
        guards: &GraphSafetyGuards,
        delta_f: f64,
    ) -> UpdateOutcome {
        // Take rollback snapshot
        self.rollback_snapshot = Some(RollbackSnapshot {
            graph: self.graph.clone(),
        });

        // Apply the candidate update
        let churn = rule.apply(self);

        // Check guards post-update
        let max_deg = self.max_degree();
        let betti0 = self.connected_components();
        let est_dim = self.estimate_dimension();

        // Spectral radius approximation: sqrt(max_degree * avg_degree)
        let avg_deg = if self.node_count() > 0 {
            2.0 * self.edge_count() as f64 / self.node_count() as f64
        } else {
            0.0
        };
        let spectral_radius_approx = (max_deg as f64 * avg_deg).sqrt();

        // Energy increase factor: use delta_f sign (positive = increase)
        let energy_factor = if delta_f > 0.0 { 1.0 + delta_f } else { 1.0 };

        // Holonomy drift: placeholder for Phase 4 (zero until gauge HDC is wired)
        let holonomy_drift = 0.0f64;

        // Entropy growth rate: compare to previous tick (placeholder)
        let entropy_growth = 0.0f64;

        let guard_result = guards.check(
            max_deg,
            churn,
            est_dim,
            spectral_radius_approx,
            energy_factor,
            betti0,
            holonomy_drift,
            entropy_growth,
        );

        self.tick += 1;

        match guard_result {
            Ok(()) => {
                self.consecutive_rejections = 0;
                self.rollback_snapshot = None;
                UpdateOutcome::Applied
            }
            Err(reason) => {
                // Rollback if configured
                if guards.rollback_on_violation {
                    if let Some(snap) = self.rollback_snapshot.take() {
                        self.graph = snap.graph;
                    }
                }

                self.consecutive_rejections += 1;

                // Trigger quarantine if too many consecutive rejections
                if guards.quarantine_strange_attractor
                    && self.consecutive_rejections >= guards.max_consecutive_rejections
                {
                    return UpdateOutcome::Quarantined {
                        reason: format!(
                            "{} consecutive rejections (last: {})",
                            self.consecutive_rejections, reason
                        ),
                    };
                }

                UpdateOutcome::Rejected { reason }
            }
        }
    }

    /// Repair graph connectivity: ensure Betti₀ = 1 by bridging disconnected components.
    ///
    /// This is called after initialization when the initial random graph is fragmented.
    /// It connects isolated components with minimum-weight spanning edges, guaranteeing
    /// the graph starts as a single connected component.
    ///
    /// **This is the fix for the seed 9999 curse** — some seeds produce Betti₀ = 3 at
    /// tick 0, which causes certain update rules (particularly CurvatureFlow) to fail
    /// before they can rescue the graph.
    pub fn repair_connectivity(&mut self) -> usize {
        let mut bridges_added = 0;
        let max_iterations = self.graph.node_count();

        for _ in 0..max_iterations {
            if connected_components(&self.graph) <= 1 {
                break;
            }

            // Find a node in component 0 and a node in a different component
            let nodes: Vec<NodeIndex> = self.graph.node_indices().collect();
            if nodes.len() < 2 {
                break;
            }

            // BFS from node[0] to find which nodes are reachable
            let start = nodes[0];
            let reachable: std::collections::HashSet<NodeIndex> = {
                let dists = dijkstra(&self.graph, start, None, |_| 1usize);
                dists.into_keys().collect()
            };

            // Find first unreachable node
            let other = nodes.iter().find(|n| !reachable.contains(n)).copied();
            let Some(other) = other else { break };

            // Bridge the gap
            let eid = self.graph.edge_count() as u64;
            self.graph.add_edge(
                start,
                other,
                GraphEdge {
                    id: eid,
                    weight: 1.0,
                    confidence: 0.5, // low confidence — this is a repair edge
                    transform_seed: self.rng.r#gen(),
                },
            );
            bridges_added += 1;
        }
        bridges_added
    }

    /// Create a new graph with guaranteed connectivity (Betti₀ = 1).
    ///
    /// Same as `new_with_seed` but applies `repair_connectivity()` after initialization.
    /// Use this for experiment runs to avoid seed-dependent fragmentation.
    pub fn new_connected(seed: u64, initial_nodes: usize) -> Self {
        let mut g = Self::new_with_seed(seed, initial_nodes);
        g.repair_connectivity();
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_connected_always_betti0_1() {
        // Test the seeds that were previously fragmented (seed 9999 curse)
        for seed in [42u64, 123, 777, 1337, 9999, 0, u64::MAX] {
            let g = SyntheticGraph::new_connected(seed, 16);
            assert_eq!(
                g.connected_components(),
                1,
                "seed {seed}: new_connected() must guarantee Betti₀ = 1"
            );
        }
    }

    #[test]
    fn repair_connectivity_bridges_fragmented_graph() {
        // seed 9999 was observed to produce Betti₀ = 3 at tick 0 in the survey
        let mut g = SyntheticGraph::new_with_seed(9999, 16);
        let before = g.connected_components();
        let bridges = g.repair_connectivity();
        let after = g.connected_components();
        if before > 1 {
            assert!(bridges > 0, "should have added bridges");
        }
        assert_eq!(after, 1, "after repair, graph must be connected");
    }

    #[test]
    fn repair_connectivity_idempotent_on_connected_graph() {
        let mut g = SyntheticGraph::new_connected(42, 16);
        assert_eq!(g.connected_components(), 1);
        let bridges = g.repair_connectivity(); // should add 0 bridges (already connected)
        assert_eq!(bridges, 0, "connected graph needs no repair bridges");
        assert_eq!(g.connected_components(), 1);
    }

    #[test]
    fn new_connected_vs_new_with_seed_node_count() {
        // new_connected must have same or more edges (repair adds bridges) but same nodes
        let g_base = SyntheticGraph::new_with_seed(9999, 16);
        let g_conn = SyntheticGraph::new_connected(9999, 16);
        assert_eq!(
            g_base.node_count(),
            g_conn.node_count(),
            "node count must be the same"
        );
        assert!(
            g_conn.edge_count() >= g_base.edge_count(),
            "connected graph must have >= edges"
        );
    }
}
