// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Knowledge Graph for Epistemic Integration
//!
//! This module provides a lightweight knowledge graph optimized for storing
//! and querying verified claims from web research. It tracks epistemic
//! provenance and confidence scores.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for a node in the knowledge graph
pub type NodeId = u64;

/// Types of nodes in the knowledge graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    /// A concept or entity
    Concept,
    /// A factual claim
    Claim,
    /// A source (URL, document, etc.)
    Source,
    /// An abstract node for high-level relationships
    Abstract,
    /// A query that generated research
    Query,
}

/// Types of edges between nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// General relationship
    RelatedTo,
    /// Supports (claim supports another claim)
    Supports,
    /// Contradicts (claim contradicts another)
    Contradicts,
    /// Sources (claim is sourced from)
    SourcedFrom,
    /// Answers (claim answers a query)
    Answers,
    /// Is-a relationship (concept hierarchy)
    IsA,
    /// Part-of relationship
    PartOf,
}

/// Source of knowledge for provenance tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KnowledgeSource {
    /// Internal knowledge (built-in vocabulary, learned patterns)
    Internal,
    /// External source with URL/identifier
    External(String),
    /// User-provided information
    UserProvided,
    /// Inferred from other knowledge
    Inferred,
}

/// A node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNode {
    /// Unique identifier
    pub id: NodeId,
    /// Node label/name
    pub name: String,
    /// Type of node
    pub node_type: NodeType,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Source of this knowledge
    pub source: KnowledgeSource,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// An edge in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEdge {
    /// Source node ID
    pub from: NodeId,
    /// Target node ID
    pub to: NodeId,
    /// Type of relationship
    pub edge_type: EdgeType,
    /// Weight/strength of relationship (0.0 - 1.0)
    pub weight: f32,
    /// Source of this relationship
    pub source: KnowledgeSource,
}

/// Statistics about the knowledge graph
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphStats {
    /// Total number of nodes
    pub nodes: usize,
    /// Total number of edges
    pub edges: usize,
    /// Breakdown by node type
    pub nodes_by_type: HashMap<String, usize>,
    /// Average confidence across all nodes
    pub avg_confidence: f32,
    /// Number of high-confidence nodes (>0.7)
    pub high_confidence_nodes: usize,
}

/// A simple knowledge graph for epistemic consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// All nodes in the graph
    nodes: HashMap<NodeId, KnowledgeNode>,
    /// Outgoing edges indexed by source node
    outgoing_edges: HashMap<NodeId, Vec<KnowledgeEdge>>,
    /// Incoming edges indexed by target node
    incoming_edges: HashMap<NodeId, Vec<KnowledgeEdge>>,
    /// Next available node ID
    next_id: NodeId,
    /// Name lookup cache
    name_to_id: HashMap<String, NodeId>,
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            outgoing_edges: HashMap::new(),
            incoming_edges: HashMap::new(),
            next_id: 1,
            name_to_id: HashMap::new(),
        }
    }

    /// Add a node to the graph
    ///
    /// Returns the NodeId of the newly created node.
    pub fn add_node(&mut self, name: impl Into<String>, node_type: NodeType) -> NodeId {
        let name = name.into();
        let id = self.next_id;
        self.next_id += 1;

        let node = KnowledgeNode {
            id,
            name: name.clone(),
            node_type,
            confidence: 0.5, // Default confidence
            created_at: chrono::Utc::now(),
            source: KnowledgeSource::Internal,
            metadata: HashMap::new(),
        };

        self.name_to_id.insert(name, id);
        self.nodes.insert(id, node);
        self.outgoing_edges.insert(id, Vec::new());
        self.incoming_edges.insert(id, Vec::new());

        id
    }

    /// Add a node with full configuration
    pub fn add_node_with_config(
        &mut self,
        name: impl Into<String>,
        node_type: NodeType,
        confidence: f32,
        source: KnowledgeSource,
    ) -> NodeId {
        let name = name.into();
        let id = self.next_id;
        self.next_id += 1;

        let node = KnowledgeNode {
            id,
            name: name.clone(),
            node_type,
            confidence: confidence.clamp(0.0, 1.0),
            created_at: chrono::Utc::now(),
            source,
            metadata: HashMap::new(),
        };

        self.name_to_id.insert(name, id);
        self.nodes.insert(id, node);
        self.outgoing_edges.insert(id, Vec::new());
        self.incoming_edges.insert(id, Vec::new());

        id
    }

    /// Add an edge with metadata
    pub fn add_edge_with_meta(
        &mut self,
        from: NodeId,
        edge_type: EdgeType,
        to: NodeId,
        weight: f32,
        source: KnowledgeSource,
    ) {
        if !self.nodes.contains_key(&from) || !self.nodes.contains_key(&to) {
            return; // Silently ignore invalid edges
        }

        let edge = KnowledgeEdge {
            from,
            to,
            edge_type,
            weight: weight.clamp(0.0, 1.0),
            source,
        };

        self.outgoing_edges
            .entry(from)
            .or_default()
            .push(edge.clone());
        self.incoming_edges.entry(to).or_default().push(edge);
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&KnowledgeNode> {
        self.nodes.get(&id)
    }

    /// Get a node by name
    pub fn get_node_by_name(&self, name: &str) -> Option<&KnowledgeNode> {
        self.name_to_id.get(name).and_then(|id| self.nodes.get(id))
    }

    /// Get outgoing edges from a node
    pub fn outgoing_edges(&self, node_id: NodeId) -> &[KnowledgeEdge] {
        self.outgoing_edges
            .get(&node_id)
            .map_or(&[], |v| v.as_slice())
    }

    /// Get incoming edges to a node
    pub fn incoming_edges(&self, node_id: NodeId) -> &[KnowledgeEdge] {
        self.incoming_edges
            .get(&node_id)
            .map_or(&[], |v| v.as_slice())
    }

    /// Update node confidence
    pub fn update_confidence(&mut self, id: NodeId, confidence: f32) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.confidence = confidence.clamp(0.0, 1.0);
        }
    }

    /// Find nodes matching a predicate
    pub fn find_nodes<F>(&self, predicate: F) -> Vec<&KnowledgeNode>
    where
        F: Fn(&KnowledgeNode) -> bool,
    {
        self.nodes.values().filter(|n| predicate(n)).collect()
    }

    /// Find nodes by type
    pub fn nodes_by_type(&self, node_type: NodeType) -> Vec<&KnowledgeNode> {
        self.find_nodes(|n| n.node_type == node_type)
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let mut nodes_by_type: HashMap<String, usize> = HashMap::new();
        let mut total_confidence = 0.0f32;
        let mut high_confidence = 0;

        for node in self.nodes.values() {
            *nodes_by_type
                .entry(format!("{:?}", node.node_type))
                .or_insert(0) += 1;
            total_confidence += node.confidence;
            if node.confidence > 0.7 {
                high_confidence += 1;
            }
        }

        let edge_count: usize = self.outgoing_edges.values().map(|v| v.len()).sum();

        GraphStats {
            nodes: self.nodes.len(),
            edges: edge_count,
            nodes_by_type,
            avg_confidence: if self.nodes.is_empty() {
                0.0
            } else {
                total_confidence / self.nodes.len() as f32
            },
            high_confidence_nodes: high_confidence,
        }
    }

    /// Calculate a simple Phi-like measure based on graph integration
    ///
    /// This is a heuristic measure of "integrated information" in the graph,
    /// based on connectivity patterns and confidence scores.
    pub fn calculate_phi(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        // Measure 1: Connectivity - ratio of edges to maximum possible
        let n = self.nodes.len() as f64;
        let edge_count: usize = self.outgoing_edges.values().map(|v| v.len()).sum();
        let max_edges = n * (n - 1.0);
        let connectivity = if max_edges > 0.0 {
            edge_count as f64 / max_edges
        } else {
            0.0
        };

        // Measure 2: Average confidence
        let avg_confidence: f64 = self
            .nodes
            .values()
            .map(|n| n.confidence as f64)
            .sum::<f64>()
            / n;

        // Measure 3: Type diversity (entropy-like)
        let stats = self.stats();
        let type_diversity = if stats.nodes_by_type.len() > 1 {
            let entropy: f64 = stats
                .nodes_by_type
                .values()
                .map(|&count| {
                    let p = count as f64 / n;
                    if p > 0.0 { -p * p.ln() } else { 0.0 }
                })
                .sum();
            entropy / (stats.nodes_by_type.len() as f64).ln()
        } else {
            0.0
        };

        // Combine measures: Phi = connectivity * confidence * (1 + diversity)
        let phi = connectivity * avg_confidence * (1.0 + type_diversity);
        phi.min(1.0) // Cap at 1.0
    }

    /// Clear all nodes and edges
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.outgoing_edges.clear();
        self.incoming_edges.clear();
        self.name_to_id.clear();
        self.next_id = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_graph_basics() {
        let mut graph = KnowledgeGraph::new();

        let node1 = graph.add_node("rust programming", NodeType::Concept);
        let node2 = graph.add_node("memory safety", NodeType::Concept);

        graph.add_edge_with_meta(
            node1,
            EdgeType::RelatedTo,
            node2,
            0.9,
            KnowledgeSource::Internal,
        );

        let stats = graph.stats();
        assert_eq!(stats.nodes, 2);
        assert_eq!(stats.edges, 1);
    }

    #[test]
    fn test_phi_calculation() {
        let mut graph = KnowledgeGraph::new();

        // Empty graph should have phi = 0
        assert_eq!(graph.calculate_phi(), 0.0);

        // Add some connected nodes
        let n1 = graph.add_node_with_config(
            "A",
            NodeType::Claim,
            0.8,
            KnowledgeSource::External("test".into()),
        );
        let n2 = graph.add_node_with_config(
            "B",
            NodeType::Claim,
            0.9,
            KnowledgeSource::External("test".into()),
        );
        let n3 = graph.add_node_with_config(
            "C",
            NodeType::Source,
            0.7,
            KnowledgeSource::External("test".into()),
        );

        graph.add_edge_with_meta(n1, EdgeType::Supports, n2, 0.8, KnowledgeSource::Internal);
        graph.add_edge_with_meta(
            n2,
            EdgeType::SourcedFrom,
            n3,
            0.9,
            KnowledgeSource::Internal,
        );
        graph.add_edge_with_meta(
            n1,
            EdgeType::SourcedFrom,
            n3,
            0.7,
            KnowledgeSource::Internal,
        );

        let phi = graph.calculate_phi();
        assert!(phi > 0.0, "Connected graph should have positive phi");
    }
}
