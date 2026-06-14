// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use petgraph::graph::UnGraph;
use std::collections::HashSet;

/// Advanced topological primitives for structural analysis
pub struct TopologicalAnalyzer;

impl TopologicalAnalyzer {
    /// Calculate Betti number β0 (number of connected components)
    pub fn betti_0(&self, graph: &UnGraph<(), ()>) -> usize {
        let mut visited = HashSet::new();
        let mut count = 0;
        for node in graph.node_indices() {
            if !visited.contains(&node) {
                count += 1;
                let mut stack = vec![node];
                while let Some(n) = stack.pop() {
                    if visited.insert(n) {
                        for neighbor in graph.neighbors(n) {
                            stack.push(neighbor);
                        }
                    }
                }
            }
        }
        count
    }

    /// Calculate cycle proxy (Betti number β1 approximation)
    /// β1 = E - V + β0
    pub fn betti_1_proxy(&self, graph: &UnGraph<(), ()>) -> i64 {
        let e = graph.edge_count() as i64;
        let v = graph.node_count() as i64;
        let b0 = self.betti_0(graph) as i64;
        e - v + b0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_betti_numbers() {
        let mut g = UnGraph::<(), ()>::default();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        let n2 = g.add_node(());
        g.add_edge(n0, n1, ());
        g.add_edge(n1, n2, ());
        g.add_edge(n2, n0, ()); // Cycle

        let analyzer = TopologicalAnalyzer;
        assert_eq!(analyzer.betti_0(&g), 1);
        assert_eq!(analyzer.betti_1_proxy(&g), 1);
    }
}
