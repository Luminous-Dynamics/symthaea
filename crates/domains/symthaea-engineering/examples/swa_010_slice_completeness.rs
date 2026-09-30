//! SWA-010: provenance-slice completeness and minimality.
//!
//! SWA-009 computes a reachable provenance slice. SWA-010 adds an explicit
//! completeness certificate: a slice is not sufficient merely because it is
//! reachable; it must contain every dependency required by the claim closure,
//! and it must not contain unrelated dependency kinds.

use serde::Serialize;
use std::collections::BTreeSet;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CompletenessCertificate {
    complete: bool,
    missing: Vec<NodeKind>,
    unrelated: Vec<NodeKind>,
}

fn certify(
    claim_kind: ClaimKind,
    slice: &[Node],
) -> CompletenessCertificate {
    let actual = slice.iter().map(|node| node.kind).collect::<BTreeSet<_>>();
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

    let allowed = required
        .union(&BTreeSet::from([NodeKind::Claim, NodeKind::Evidence, NodeKind::Prediction]))
        .copied()
        .collect::<BTreeSet<_>>();

    let mut unrelated = actual
        .difference(&allowed)
        .copied()
        .collect::<Vec<_>>();
    unrelated.sort();

    CompletenessCertificate {
        complete: missing.is_empty() && unrelated.is_empty(),
        missing,
        unrelated,
    }
}

fn reference_slice() -> Vec<Node> {
    vec![
        Node { id: "claim-001", kind: NodeKind::Claim },
        Node { id: "validation-001", kind: NodeKind::Evidence },
        Node { id: "prediction-001", kind: NodeKind::Prediction },
        Node { id: "model-001", kind: NodeKind::Model },
        Node { id: "parameters-001", kind: NodeKind::Parameters },
        Node { id: "scenario-001", kind: NodeKind::Scenario },
        Node { id: "dataset-001", kind: NodeKind::Dataset },
    ]
}

fn main() {
    let certificate = certify(ClaimKind::PredictionValidated, &reference_slice());

    assert!(certificate.complete);
    assert!(certificate.missing.is_empty());
    assert!(certificate.unrelated.is_empty());

    println!(
        "{}",
        serde_json::to_string(&certificate).expect("certificate serializes")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_slice_is_accepted() {
        let result = certify(ClaimKind::PredictionValidated, &reference_slice());
        assert!(result.complete);
    }

    #[test]
    fn missing_dependency_is_rejected() {
        let mut slice = reference_slice();
        slice.retain(|node| node.kind != NodeKind::Parameters);

        let result = certify(ClaimKind::PredictionValidated, &slice);

        assert!(!result.complete);
        assert_eq!(result.missing, vec![NodeKind::Parameters]);
    }

    #[test]
    fn unrelated_dependency_is_rejected() {
        let mut slice = reference_slice();
        slice.push(Node {
            id: "unrelated-context",
            kind: NodeKind::ContextOfUse,
        });

        let result = certify(ClaimKind::PredictionValidated, &slice);

        assert!(!result.complete);
        assert_eq!(result.unrelated, vec![NodeKind::ContextOfUse]);
    }

    #[test]
    fn applicability_claim_requires_context() {
        let mut slice = reference_slice();
        slice.push(Node {
            id: "context-001",
            kind: NodeKind::ContextOfUse,
        });

        let result = certify(ClaimKind::ApplicabilitySupported, &slice);

        assert!(result.complete);
    }

    #[test]
    fn certificate_is_deterministic() {
        let first = certify(ClaimKind::PredictionValidated, &reference_slice());
        let second = certify(ClaimKind::PredictionValidated, &reference_slice());

        assert_eq!(first, second);
        assert_eq!(
            serde_json::to_string(&first).expect("first serializes"),
            serde_json::to_string(&second).expect("second serializes")
        );
    }
}
