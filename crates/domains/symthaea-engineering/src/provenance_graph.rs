//! Shared deterministic provenance graph primitives for the Sol Atlas fixtures.
//!
//! This module is the authoritative in-process representation used by the
//! SWA provenance slice example and downstream evidence bindings. Consumers
//! may project a selected slice into another wire format, but they must not
//! reconstruct the graph topology independently.

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum NodeKind {
    Claim,
    Evidence,
    Prediction,
    Model,
    Parameters,
    Scenario,
    Dataset,
    ContextOfUse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Node {
    pub id: &'static str,
    pub kind: NodeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum EdgeKind {
    DerivedFrom,
    SupportedBy,
    QualifiedBy,
    ContradictedBy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Edge {
    pub from: &'static str,
    pub to: &'static str,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProvenanceGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl ProvenanceGraph {
    pub fn slice(&self, root: &'static str) -> Option<ProvenanceSlice> {
        if !self.nodes.iter().any(|node| node.id == root) {
            return None;
        }

        let mut outgoing: BTreeMap<&'static str, Vec<Edge>> = BTreeMap::new();
        for edge in &self.edges {
            outgoing.entry(edge.from).or_default().push(*edge);
        }

        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::from([root]);

        while let Some(id) = queue.pop_front() {
            if !visited.insert(id) {
                continue;
            }

            if let Some(edges) = outgoing.get(id) {
                for edge in edges {
                    queue.push_back(edge.to);
                }
            }
        }

        let mut nodes = self
            .nodes
            .iter()
            .copied()
            .filter(|node| visited.contains(node.id))
            .collect::<Vec<_>>();
        nodes.sort_by_key(|node| (node.kind, node.id));

        let mut edges = self
            .edges
            .iter()
            .copied()
            .filter(|edge| visited.contains(edge.from) && visited.contains(edge.to))
            .collect::<Vec<_>>();
        edges.sort();

        Some(ProvenanceSlice { root, nodes, edges })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProvenanceSlice {
    pub root: &'static str,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl ProvenanceSlice {
    pub fn contains(&self, id: &str) -> bool {
        self.nodes.iter().any(|node| node.id == id)
    }

    pub fn dependency_frontier(&self) -> Vec<Node> {
        self.nodes
            .iter()
            .copied()
            .filter(|node| {
                matches!(
                    node.kind,
                    NodeKind::Model
                        | NodeKind::Parameters
                        | NodeKind::Scenario
                        | NodeKind::Dataset
                        | NodeKind::ContextOfUse
                )
            })
            .collect()
    }
}

/// The deterministic reference graph used by SWA-009 and downstream
/// cross-system projections. Keep topology here rather than copying it into
/// adapter examples.
pub fn reference_graph() -> ProvenanceGraph {
    ProvenanceGraph {
        nodes: vec![
            Node { id: "claim-001", kind: NodeKind::Claim },
            Node { id: "validation-001", kind: NodeKind::Evidence },
            Node { id: "prediction-001", kind: NodeKind::Prediction },
            Node { id: "model-001", kind: NodeKind::Model },
            Node { id: "parameters-001", kind: NodeKind::Parameters },
            Node { id: "scenario-001", kind: NodeKind::Scenario },
            Node { id: "dataset-001", kind: NodeKind::Dataset },
            Node { id: "context-001", kind: NodeKind::ContextOfUse },
            Node { id: "unrelated-001", kind: NodeKind::Dataset },
        ],
        edges: vec![
            Edge { from: "claim-001", to: "validation-001", kind: EdgeKind::SupportedBy },
            Edge { from: "validation-001", to: "prediction-001", kind: EdgeKind::DerivedFrom },
            Edge { from: "validation-001", to: "context-001", kind: EdgeKind::DerivedFrom },
            Edge { from: "prediction-001", to: "model-001", kind: EdgeKind::DerivedFrom },
            Edge { from: "prediction-001", to: "parameters-001", kind: EdgeKind::DerivedFrom },
            Edge { from: "prediction-001", to: "scenario-001", kind: EdgeKind::DerivedFrom },
            Edge { from: "validation-001", to: "dataset-001", kind: EdgeKind::DerivedFrom },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_slice_is_deterministic_and_excludes_unrelated_nodes() {
        let first = reference_graph().slice("claim-001").expect("claim exists");
        let second = reference_graph().slice("claim-001").expect("claim exists");

        assert_eq!(first, second);
        assert_eq!(first.nodes.len(), 8);
        assert!(!first.contains("unrelated-001"));
    }

    #[test]
    fn contradiction_and_qualification_edges_remain_distinct() {
        let mut graph = reference_graph();
        graph.nodes.push(Node { id: "counter-001", kind: NodeKind::Evidence });
        graph.nodes.push(Node { id: "boundary-001", kind: NodeKind::Evidence });
        graph.edges.push(Edge {
            from: "claim-001",
            to: "counter-001",
            kind: EdgeKind::ContradictedBy,
        });
        graph.edges.push(Edge {
            from: "claim-001",
            to: "boundary-001",
            kind: EdgeKind::QualifiedBy,
        });

        let slice = graph.slice("claim-001").expect("claim exists");
        assert!(slice.edges.iter().any(|edge| edge.kind == EdgeKind::ContradictedBy));
        assert!(slice.edges.iter().any(|edge| edge.kind == EdgeKind::QualifiedBy));
    }

    #[test]
    fn missing_root_is_not_an_empty_slice() {
        assert!(reference_graph().slice("missing").is_none());
    }
}
