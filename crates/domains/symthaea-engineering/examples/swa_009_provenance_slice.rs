//! SWA-009: deterministic provenance slicing.
//!
//! Given a downstream claim, compute the minimal reproducible provenance slice
//! required to reproduce or challenge it. The slice is a graph traversal, not a
//! copy of the entire validation record.

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
enum NodeKind {
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
struct Node {
    id: &'static str,
    kind: NodeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
enum EdgeKind {
    DerivedFrom,
    SupportedBy,
    QualifiedBy,
    ContradictedBy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct Edge {
    from: &'static str,
    to: &'static str,
    kind: EdgeKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ProvenanceGraph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

impl ProvenanceGraph {
    fn slice(&self, root: &'static str) -> Option<ProvenanceSlice> {
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
struct ProvenanceSlice {
    root: &'static str,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

impl ProvenanceSlice {
    fn contains(&self, id: &str) -> bool {
        self.nodes.iter().any(|node| node.id == id)
    }

    fn dependency_frontier(&self) -> Vec<Node> {
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

fn reference_graph() -> ProvenanceGraph {
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

fn main() {
    let graph = reference_graph();
    let slice = graph.slice("claim-001").expect("claim exists");

    assert!(slice.contains("claim-001"));
    assert!(slice.contains("validation-001"));
    assert!(slice.contains("prediction-001"));
    assert!(slice.contains("model-001"));
    assert!(slice.contains("parameters-001"));
    assert!(slice.contains("scenario-001"));
    assert!(slice.contains("dataset-001"));
    assert!(slice.contains("context-001"));
    assert!(!slice.contains("unrelated-001"));

    println!("{}", serde_json::to_string(&slice).expect("slice serializes"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_contains_only_reachable_provenance() {
        let slice = reference_graph().slice("claim-001").expect("claim exists");
        assert!(!slice.contains("unrelated-001"));
        assert_eq!(slice.nodes.len(), 8);
    }

    #[test]
    fn dependency_frontier_is_explicit_and_deterministic() {
        let slice = reference_graph().slice("claim-001").expect("claim exists");
        let first = slice.dependency_frontier();
        let second = slice.dependency_frontier();

        assert_eq!(first, second);
        assert_eq!(first.len(), 5);
        assert!(first.iter().any(|node| node.kind == NodeKind::Model));
        assert!(first.iter().any(|node| node.kind == NodeKind::ContextOfUse));
    }

    #[test]
    fn missing_root_is_not_an_empty_slice() {
        assert!(reference_graph().slice("missing").is_none());
    }

    #[test]
    fn counterevidence_can_be_traversed_without_resolution() {
        let mut graph = reference_graph();
        graph.nodes.push(Node { id: "counter-001", kind: NodeKind::Evidence });
        graph.edges.push(Edge {
            from: "claim-001",
            to: "counter-001",
            kind: EdgeKind::ContradictedBy,
        });

        let slice = graph.slice("claim-001").expect("claim exists");
        assert!(slice.contains("counter-001"));
        assert!(slice.edges.iter().any(|edge| edge.kind == EdgeKind::ContradictedBy));
    }

    #[test]
    fn qualification_is_preserved_as_qualification() {
        let mut graph = reference_graph();
        graph.nodes.push(Node { id: "boundary-001", kind: NodeKind::Evidence });
        graph.edges.push(Edge {
            from: "claim-001",
            to: "boundary-001",
            kind: EdgeKind::QualifiedBy,
        });

        let slice = graph.slice("claim-001").expect("claim exists");
        assert!(slice.edges.iter().any(|edge| edge.kind == EdgeKind::QualifiedBy));
        assert!(!slice.edges.iter().any(|edge| {
            edge.from == "claim-001" && edge.to == "boundary-001" && edge.kind == EdgeKind::ContradictedBy
        }));
    }

    #[test]
    fn slice_is_not_authority() {
        let slice = reference_graph().slice("claim-001").expect("claim exists");
        assert!(slice.contains("claim-001"));
        assert!(!slice.contains("authorization-001"));
    }
}
