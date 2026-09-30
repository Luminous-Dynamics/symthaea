//! SWA-010: claim-specific completeness over an authoritative provenance slice.
//!
//! SWA-028 moved structural traversal completeness into ProvenanceSlice itself.
//! This example therefore checks semantic completeness only: whether the
//! selected claim kind has all required dependency kinds. It does not rebuild
//! graph topology or issue a second traversal certificate.

use std::collections::BTreeSet;
use symthaea_engineering::provenance_graph::{reference_graph, NodeKind, ProvenanceSlice};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaimKind {
    PredictionValidated,
    ApplicabilitySupported,
}

impl ClaimKind {
    fn required_dependencies(self) -> &'static [NodeKind] {
        match self {
            Self::PredictionValidated => &[
                NodeKind::Model,
                NodeKind::Parameters,
                NodeKind::Scenario,
                NodeKind::Dataset,
            ],
            Self::ApplicabilitySupported => &[
                NodeKind::Model,
                NodeKind::Scenario,
                NodeKind::Dataset,
                NodeKind::ContextOfUse,
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompletenessCertificate {
    complete: bool,
    missing: Vec<NodeKind>,
}

fn certify(claim_kind: ClaimKind, slice: &ProvenanceSlice) -> CompletenessCertificate {
    let actual = slice
        .dependency_frontier()
        .into_iter()
        .map(|node| node.kind)
        .collect::<BTreeSet<_>>();
    let required = claim_kind
        .required_dependencies()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();

    let mut missing = required
        .difference(&actual)
        .copied()
        .collect::<Vec<_>>();
    missing.sort();

    CompletenessCertificate {
        complete: slice.boundary.is_complete() && missing.is_empty(),
        missing,
    }
}

fn reference_slice() -> ProvenanceSlice {
    reference_graph()
        .slice("claim-001")
        .expect("authoritative reference graph is valid")
}

fn main() {
    let slice = reference_slice();
    assert!(slice.boundary.is_complete());

    let certificate = certify(ClaimKind::PredictionValidated, &slice);

    assert!(certificate.complete);
    assert!(certificate.missing.is_empty());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_slice_is_accepted() {
        let slice = reference_slice();
        let result = certify(ClaimKind::PredictionValidated, &slice);

        assert!(result.complete);
        assert!(slice.boundary.is_complete());
    }

    #[test]
    fn missing_semantic_dependency_is_rejected() {
        let slice = reference_slice();
        let filtered = ProvenanceSlice {
            nodes: slice
                .nodes
                .iter()
                .copied()
                .filter(|node| node.kind != NodeKind::Parameters)
                .collect(),
            ..slice.clone()
        };

        let result = certify(ClaimKind::PredictionValidated, &filtered);

        assert!(!result.complete);
        assert_eq!(result.missing, vec![NodeKind::Parameters]);
    }

    #[test]
    fn applicability_claim_requires_context() {
        let slice = reference_slice();
        let result = certify(ClaimKind::ApplicabilitySupported, &slice);

        assert!(result.complete);
    }

    #[test]
    fn structural_boundary_is_part_of_slice() {
        let slice = reference_slice();

        assert_eq!(slice.boundary.root, "claim-001");
        assert_eq!(slice.boundary.graph_node_count, 9);
        assert_eq!(slice.boundary.graph_edge_count, 7);
        assert_eq!(slice.boundary.slice_node_count, 8);
        assert_eq!(slice.boundary.slice_edge_count, 7);
        assert!(slice.boundary.frontier_exhausted);
    }
}
