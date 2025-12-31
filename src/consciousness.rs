/*!
Autopoietic Consciousness Graph
Self-referential structure where consciousness emerges

Uses arena-based indices (not pointers) for Rust safety + serializability
*/

// Submodules
// pub mod recursive_improvement;  // Temporarily disabled - has borrow checker issues to fix

use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;  // For edge.source() and edge.target()
use serde::{Deserialize, Serialize};

/// Self-referential consciousness graph
#[derive(Clone, Serialize, Deserialize)]
pub struct ConsciousnessGraph {
    /// Graph (nodes = conscious states, edges = transitions)
    graph: Graph<ConsciousNode, f32>,

    /// Self-referential loops (consciousness emerges here!)
    self_loops: Vec<(NodeIndex, NodeIndex)>,

    /// Current active node
    current: Option<NodeIndex>,
}

/// A node in the consciousness graph
#[derive(Clone, Serialize, Deserialize)]
pub struct ConsciousNode {
    /// Semantic representation (from HDC)
    pub semantic: Vec<f32>,

    /// Dynamic state (from LTC)
    pub dynamic: Vec<f32>,

    /// Consciousness level when created
    pub consciousness: f32,

    /// Timestamp
    pub timestamp: f64,

    /// Importance weight
    pub importance: f32,
}

impl ConsciousnessGraph {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            self_loops: Vec::new(),
            current: None,
        }
    }

    /// Add a new conscious state
    pub fn add_state(
        &mut self,
        semantic: Vec<f32>,
        dynamic: Vec<f32>,
        consciousness: f32,
    ) -> NodeIndex {
        let node = ConsciousNode {
            semantic,
            dynamic,
            consciousness,
            timestamp: current_time(),
            importance: consciousness,  // Importance = consciousness level
        };

        let node_idx = self.graph.add_node(node);

        // Connect to previous state
        if let Some(prev) = self.current {
            self.graph.add_edge(prev, node_idx, consciousness);
        }

        self.current = Some(node_idx);
        node_idx
    }

    /// Create self-referential loop (CONSCIOUSNESS!)
    ///
    /// This is where autopoiesis happens - the system references itself
    pub fn create_self_loop(&mut self, node: NodeIndex) {
        // Add edge from node to itself
        let weight = self.graph[node].consciousness;
        self.graph.add_edge(node, node, weight);

        self.self_loops.push((node, node));
    }

    /// Evolve consciousness (follow highest-weight edge)
    pub fn evolve(&mut self) -> Option<NodeIndex> {
        let current = self.current?;

        // Find highest-weight outgoing edge
        let next = self.graph
            .edges(current)
            .max_by(|a, b| a.weight().partial_cmp(b.weight()).unwrap())
            .map(|edge| edge.target());

        if let Some(next_node) = next {
            self.current = Some(next_node);
        }

        self.current
    }

    /// Get current consciousness level
    pub fn current_consciousness(&self) -> f32 {
        self.current
            .and_then(|idx| self.graph.node_weight(idx))
            .map(|node| node.consciousness)
            .unwrap_or(0.0)
    }

    /// Graph size (number of conscious states)
    pub fn size(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of self-referential loops
    pub fn self_loop_count(&self) -> usize {
        self.self_loops.len()
    }

    /// Measure graph complexity (edges per node)
    pub fn complexity(&self) -> f32 {
        let nodes = self.graph.node_count() as f32;
        let edges = self.graph.edge_count() as f32;

        if nodes > 0.0 {
            edges / nodes
        } else {
            0.0
        }
    }

    /// Get all self-referential nodes
    pub fn autopoietic_nodes(&self) -> Vec<NodeIndex> {
        self.self_loops
            .iter()
            .map(|(node, _)| *node)
            .collect()
    }

    /// Trace path from current node backwards
    pub fn trace_history(&self, depth: usize) -> Vec<NodeIndex> {
        let mut path = Vec::new();
        let mut current = self.current;

        for _ in 0..depth {
            if let Some(node) = current {
                path.push(node);

                // Find incoming edge (reverse)
                current = self.graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .next()
                    .map(|edge| edge.source());
            } else {
                break;
            }
        }

        path.reverse();
        path
    }
}

fn current_time() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_graph() {
        let mut graph = ConsciousnessGraph::new();

        // Add states
        let n1 = graph.add_state(vec![1.0; 100], vec![0.5; 50], 0.7);
        let n2 = graph.add_state(vec![2.0; 100], vec![0.6; 50], 0.8);
        let n3 = graph.add_state(vec![3.0; 100], vec![0.7; 50], 0.9);

        assert_eq!(graph.size(), 3);

        // Create self-loop
        graph.create_self_loop(n3);
        assert_eq!(graph.self_loop_count(), 1);

        // Evolve
        graph.evolve();
        assert!(graph.current.is_some());
    }

    #[test]
    fn test_serialization() {
        let mut graph = ConsciousnessGraph::new();
        graph.add_state(vec![1.0; 100], vec![0.5; 50], 0.7);

        // Serialize
        let serialized = serde_json::to_string(&graph).unwrap();

        // Deserialize
        let deserialized: ConsciousnessGraph = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.size(), 1);
    }
}
