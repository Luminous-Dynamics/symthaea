//! SWA-009: deterministic provenance slicing.
//!
//! Given a downstream claim, compute the minimal reproducible provenance slice
//! required to reproduce or challenge it. The slice is a graph traversal, not a
//! copy of the entire validation record.

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::Serialize;
use symthaea_engineering::provenance_graph::{reference_graph, NodeKind, ProvenanceSlice};

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
