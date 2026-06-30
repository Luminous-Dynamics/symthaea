// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Graph update rules — the local rewriting operations whose attractors we study.
//!
//! Each rule defines one local operation. The attractor lab tests which rules
//! produce stable, useful, low-dimensional manifolds.
//!
//! ## Rules (v0.1)
//!
//! 1. [`UpdateRule::NearestNeighborAttachment`] — new nodes attach to lowest-degree neighbor
//! 2. [`UpdateRule::TriangulationPressure`] — edges added to close open triangles
//! 3. [`UpdateRule::DegreeBalancingRemoval`] — high-degree edges pruned probabilistically
//! 4. [`UpdateRule::CurvatureFlow`] — edges rewired to minimize local degree variance
//! 5. [`UpdateRule::FreeEnergyMinimization`] — edges added/removed to minimize F_structural
//!
//! All rules return the edge churn count (adds + removes).

use petgraph::graph::NodeIndex;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::graph::{GraphEdge, GraphNode, NodeSemanticType, SyntheticGraph};

/// A local graph-rewriting rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateRule {
    /// **Rule 1**: Each tick, add one new node connected to the lowest-degree existing node.
    /// Models preferential attachment to underserved regions.
    NearestNeighborAttachment {
        /// Number of new nodes to add per tick.
        nodes_per_tick: usize,
    },

    /// **Rule 2**: For each node, check its neighbor-pairs. If two neighbors are not connected,
    /// add that edge with `probability`. Models triangle-closing / local clustering pressure.
    TriangulationPressure {
        /// Probability of closing each open triangle.
        probability: f64,
    },

    /// **Rule 3**: For each edge connecting a node with degree > `degree_threshold`,
    /// remove it with `probability`. Models degree-balancing / entropy reduction.
    DegreeBalancingRemoval {
        /// Degree threshold above which edges become candidates for removal.
        degree_threshold: u32,
        /// Probability of removing each candidate edge per tick.
        probability: f64,
    },

    /// **Rule 4**: For each node, rewire one edge to a neighbor that reduces local degree variance.
    /// Models curvature flow / homogenization.
    CurvatureFlow {
        /// Fraction of nodes to process per tick.
        fraction: f64,
    },

    /// **Rule 5**: Add or remove edges based on their contribution to structural free energy.
    /// Accepts an edge change only if ΔF_local < 0.
    FreeEnergyMinimization {
        /// Number of candidate edge changes to evaluate per tick.
        candidates_per_tick: usize,
    },
}

impl UpdateRule {
    /// A modifier applied to the structural free energy delta for this rule.
    /// More aggressive rules cost more.
    pub fn complexity_modifier(&self) -> f64 {
        match self {
            UpdateRule::NearestNeighborAttachment { nodes_per_tick } => {
                1.0 + 0.1 * (*nodes_per_tick as f64)
            }
            UpdateRule::TriangulationPressure { probability } => 1.0 + probability,
            UpdateRule::DegreeBalancingRemoval { .. } => 0.8, // reduces complexity
            UpdateRule::CurvatureFlow { fraction } => 1.0 + 0.5 * fraction,
            UpdateRule::FreeEnergyMinimization {
                candidates_per_tick,
            } => 1.0 + 0.05 * (*candidates_per_tick as f64),
        }
    }

    /// Apply this rule to the graph. Returns the edge churn count (adds + removes).
    pub fn apply(&self, graph: &mut SyntheticGraph) -> usize {
        match self {
            UpdateRule::NearestNeighborAttachment { nodes_per_tick } => {
                apply_nearest_neighbor(graph, *nodes_per_tick)
            }
            UpdateRule::TriangulationPressure { probability } => {
                apply_triangulation_pressure(graph, *probability)
            }
            UpdateRule::DegreeBalancingRemoval {
                degree_threshold,
                probability,
            } => apply_degree_balancing(graph, *degree_threshold, *probability),
            UpdateRule::CurvatureFlow { fraction } => apply_curvature_flow(graph, *fraction),
            UpdateRule::FreeEnergyMinimization {
                candidates_per_tick,
            } => apply_free_energy_minimization(graph, *candidates_per_tick),
        }
    }

    /// Human-readable name for logging and visualization.
    pub fn name(&self) -> &'static str {
        match self {
            UpdateRule::NearestNeighborAttachment { .. } => "nearest_neighbor_attachment",
            UpdateRule::TriangulationPressure { .. } => "triangulation_pressure",
            UpdateRule::DegreeBalancingRemoval { .. } => "degree_balancing_removal",
            UpdateRule::CurvatureFlow { .. } => "curvature_flow",
            UpdateRule::FreeEnergyMinimization { .. } => "free_energy_minimization",
        }
    }
}

// ── Rule implementations ──────────────────────────────────────────────────────

fn apply_nearest_neighbor(graph: &mut SyntheticGraph, nodes_per_tick: usize) -> usize {
    let mut churn = 0;
    for _ in 0..nodes_per_tick {
        // Find lowest-degree node
        let target = graph
            .graph
            .node_indices()
            .min_by_key(|&n| graph.graph.neighbors(n).count());

        let Some(target) = target else { break };

        let new_id = graph.graph.node_count() as u64;
        let new_node = graph.graph.add_node(GraphNode {
            id: new_id,
            hdc_seed: graph.rng.r#gen(),
            semantic_type: NodeSemanticType::Generic,
        });

        let edge_id = graph.graph.edge_count() as u64;
        graph.graph.add_edge(
            new_node,
            target,
            GraphEdge {
                id: edge_id,
                weight: graph.rng.gen_range(0.1..1.0),
                confidence: 1.0,
                transform_seed: graph.rng.r#gen(),
            },
        );
        churn += 1;
    }
    churn
}

fn apply_triangulation_pressure(graph: &mut SyntheticGraph, probability: f64) -> usize {
    let mut churn = 0;
    let node_indices: Vec<NodeIndex> = graph.graph.node_indices().collect();

    for &v in &node_indices {
        let neighbors: Vec<NodeIndex> = graph.graph.neighbors(v).collect();
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let a = neighbors[i];
                let b = neighbors[j];
                if !graph.graph.contains_edge(a, b) && graph.rng.gen_bool(probability) {
                    let eid = graph.graph.edge_count() as u64;
                    graph.graph.add_edge(
                        a,
                        b,
                        GraphEdge {
                            id: eid,
                            weight: graph.rng.gen_range(0.1..1.0),
                            confidence: 0.8, // inferred edge — lower confidence
                            transform_seed: graph.rng.r#gen(),
                        },
                    );
                    churn += 1;
                }
            }
        }
    }
    churn
}

fn apply_degree_balancing(
    graph: &mut SyntheticGraph,
    degree_threshold: u32,
    probability: f64,
) -> usize {
    let mut edges_to_remove = vec![];

    for edge in graph.graph.edge_indices() {
        if let Some((a, b)) = graph.graph.edge_endpoints(edge) {
            let deg_a = graph.graph.neighbors(a).count() as u32;
            let deg_b = graph.graph.neighbors(b).count() as u32;
            if (deg_a > degree_threshold || deg_b > degree_threshold)
                && graph.rng.gen_bool(probability)
            {
                edges_to_remove.push(edge);
            }
        }
    }

    let churn = edges_to_remove.len();
    for e in edges_to_remove {
        graph.graph.remove_edge(e);
    }
    churn
}

fn apply_curvature_flow(graph: &mut SyntheticGraph, fraction: f64) -> usize {
    let node_indices: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n_to_process = ((node_indices.len() as f64) * fraction).ceil() as usize;
    let mut churn = 0;

    for &v in node_indices.iter().take(n_to_process) {
        let neighbors: Vec<NodeIndex> = graph.graph.neighbors(v).collect();
        if neighbors.len() < 2 {
            continue;
        }

        // Find highest-degree neighbor
        let max_deg_neighbor = neighbors
            .iter()
            .copied()
            .max_by_key(|&n| graph.graph.neighbors(n).count());

        // Find a non-neighbor that has lower degree (rewiring target)
        let rewire_target = node_indices
            .iter()
            .copied()
            .filter(|&n| {
                n != v
                    && !graph.graph.contains_edge(v, n)
                    && graph.graph.neighbors(n).count() < graph.graph.neighbors(v).count()
            })
            .next();

        if let (Some(old_neighbor), Some(new_target)) = (max_deg_neighbor, rewire_target) {
            if let Some(edge) = graph.graph.find_edge(v, old_neighbor) {
                graph.graph.remove_edge(edge);
                let eid = graph.graph.edge_count() as u64;
                graph.graph.add_edge(
                    v,
                    new_target,
                    GraphEdge {
                        id: eid,
                        weight: graph.rng.gen_range(0.1..1.0),
                        confidence: 0.9,
                        transform_seed: graph.rng.r#gen(),
                    },
                );
                churn += 2; // 1 remove + 1 add
            }
        }
    }
    churn
}

fn apply_free_energy_minimization(graph: &mut SyntheticGraph, candidates_per_tick: usize) -> usize {
    // Simplified: try random edge additions and removals, accept if they reduce
    // a local proxy of structural free energy (degree variance).
    let mut churn = 0;
    let node_indices: Vec<NodeIndex> = graph.graph.node_indices().collect();
    if node_indices.len() < 2 {
        return 0;
    }

    for _ in 0..candidates_per_tick {
        let i = graph.rng.gen_range(0..node_indices.len());
        let j = graph.rng.gen_range(0..node_indices.len());
        if i == j {
            continue;
        }
        let a = node_indices[i];
        let b = node_indices[j];

        // Compute local degree variance before change
        let deg_a_before = graph.graph.neighbors(a).count() as f64;
        let deg_b_before = graph.graph.neighbors(b).count() as f64;
        let variance_before = (deg_a_before - deg_b_before).powi(2);

        if let Some(edge) = graph.graph.find_edge(a, b) {
            // Try removing
            let deg_a_after = deg_a_before - 1.0;
            let deg_b_after = deg_b_before - 1.0;
            let variance_after = (deg_a_after - deg_b_after).powi(2);
            if variance_after < variance_before {
                graph.graph.remove_edge(edge);
                churn += 1;
            }
        } else {
            // Try adding
            let deg_a_after = deg_a_before + 1.0;
            let deg_b_after = deg_b_before + 1.0;
            let variance_after = (deg_a_after - deg_b_after).powi(2);
            if variance_after <= variance_before {
                let eid = graph.graph.edge_count() as u64;
                graph.graph.add_edge(
                    a,
                    b,
                    GraphEdge {
                        id: eid,
                        weight: graph.rng.gen_range(0.1..1.0),
                        confidence: 0.7, // model-driven edge
                        transform_seed: graph.rng.r#gen(),
                    },
                );
                churn += 1;
            }
        }
    }
    churn
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::SyntheticGraph;

    fn run_rule(rule: UpdateRule, ticks: usize) -> SyntheticGraph {
        let mut graph = SyntheticGraph::new_with_seed(42, 16);
        let guards = crate::circuit_breakers::GraphSafetyGuards::default();
        let fe = crate::free_energy::StructuralFreeEnergy::default();
        for _ in 0..ticks {
            let delta_f = fe.compute_delta(&graph, &rule);
            graph.apply_update(&rule, &guards, delta_f);
        }
        graph
    }

    #[test]
    fn nearest_neighbor_grows_graph() {
        let rule = UpdateRule::NearestNeighborAttachment { nodes_per_tick: 2 };
        let g = run_rule(rule, 10);
        assert!(g.node_count() > 16, "graph should grow");
    }

    #[test]
    fn triangulation_adds_edges() {
        let rule = UpdateRule::TriangulationPressure { probability: 0.5 };
        let g_before = SyntheticGraph::new_with_seed(42, 16);
        let e_before = g_before.edge_count();
        let g_after = run_rule(rule, 5);
        assert!(
            g_after.edge_count() >= e_before,
            "triangulation should not reduce edges"
        );
    }

    #[test]
    fn degree_balancing_reduces_hubs() {
        let rule = UpdateRule::DegreeBalancingRemoval {
            degree_threshold: 2,
            probability: 0.8,
        };
        let g = run_rule(rule, 20);
        // After aggressive removal, max degree should be lower
        assert!(g.max_degree() <= 8, "degree balancing should limit hubs");
    }

    #[test]
    fn all_rules_survive_100_ticks() {
        let rules = vec![
            UpdateRule::NearestNeighborAttachment { nodes_per_tick: 1 },
            UpdateRule::TriangulationPressure { probability: 0.1 },
            UpdateRule::DegreeBalancingRemoval {
                degree_threshold: 4,
                probability: 0.3,
            },
            UpdateRule::CurvatureFlow { fraction: 0.2 },
            UpdateRule::FreeEnergyMinimization {
                candidates_per_tick: 5,
            },
        ];
        for rule in rules {
            let name = rule.name();
            let g = run_rule(rule, 100);
            assert!(g.node_count() > 0, "rule '{name}' should not destroy graph");
        }
    }
}
