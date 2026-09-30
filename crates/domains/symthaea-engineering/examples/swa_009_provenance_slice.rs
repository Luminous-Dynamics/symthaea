//! SWA-009: deterministic provenance slicing.
//!
//! Given a downstream claim, compute the minimal reproducible provenance slice
//! required to reproduce or challenge it. The slice is a graph traversal, not a
//! copy of the entire validation record.

use symthaea_engineering::provenance_graph::{reference_graph, Edge, EdgeKind, Node, NodeKind};

fn main() {
    let graph = reference_graph();
    assert_eq!(graph.validate(), Ok(()));
    let slice = graph.slice("claim-001").expect("claim exists");

    assert!(slice.contains("claim-001"));
    assert!(slice.contains("validation-001"));
    assert!(slice.contains("prediction-001"));
    assert!(slice.contains("model-001"));