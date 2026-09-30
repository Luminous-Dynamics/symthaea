//! SWA-013 — provenance integrity validation.
//!
//! SWA-012 established that support, qualification, and contradiction should
//! remain independently replayable. SWA-013 makes the provenance structure
//! itself subject to deterministic, fail-closed integrity checks.
//!
//! The validator is deliberately narrower than a general PROV implementation:
//! it checks the invariants required by the SWA reference fixture without
//! collapsing contested evidence into a winner.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum NodeKind {
    Claim,
    Evidence,
    Prediction,
    Model,
    Parameters,
    Scenario,
    Dataset,
    ContextOfUse,
    Witness,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub kind: NodeKind,
    /// Dependency identity is a stable logical key; revision is the exact
    /// version used by this provenance record.
    pub dependency_key: Option<String>,
    pub revision: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum EdgeKind {
    DerivedFrom,
    Supports,
    Qualifies,
    Contradicts,
    References,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub kind: EdgeKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum IntegrityError {
    EmptyNodeId,
    DuplicateNodeIdentity { id: String },
    ConflictingDependencyRevision {
        dependency_key: String,
        revisions: Vec<String>,
    },
    DuplicateEdge { from: String, to: String, kind: EdgeKind },
    MissingNode { node_id: String },
    SelfDerivation { node_id: String },
    DerivationCycle { nodes: Vec<String> },
    WitnessReferenceMissing { witness_id: String },
    InvalidWitnessNode { witness_id: String },
    MissingRequiredDependency {
        claim_id: String,
        dependency: NodeKind,
    },
    CompletenessCertificateMismatch { claim_id: String },
    EmptyClaimId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum IntegrityStatus {
    Valid,
    Invalid,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompletenessCertificate {
    pub claim_id: String,
    pub required: BTreeSet<NodeKind>,
    pub observed: BTreeSet<NodeKind>,
}

impl CompletenessCertificate {
    pub fn is_truthful(&self) -> bool {
        self.required == self.observed
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IntegrityReport {
    pub status: IntegrityStatus,
    pub errors: Vec<IntegrityError>,
    pub normalized_node_ids: Vec<String>,
    pub normalized_edges: Vec<Edge>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct ProvenanceGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub certificates: Vec<CompletenessCertificate>,
}

impl ProvenanceGraph {
    pub fn validate(&self) -> IntegrityReport {
        let mut errors = BTreeSet::new();
        let mut nodes_by_id = BTreeMap::<String, &Node>::new();

        for node in &self.nodes {
            if node.id.is_empty() {
                errors.insert(IntegrityError::EmptyNodeId);
            }
            if nodes_by_id.insert(node.id.clone(), node).is_some() {
                errors.insert(IntegrityError::DuplicateNodeIdentity { id: node.id.clone() });
            }
        }

        let mut dependency_revisions = BTreeMap::<String, BTreeSet<String>>::new();
        for node in &self.nodes {
            if let Some(key) = &node.dependency_key {
                dependency_revisions
                    .entry(key.clone())
                    .or_default()
                    .insert(node.revision.clone().unwrap_or_default());
            }
        }
        for (dependency_key, revisions) in dependency_revisions {
            if revisions.len() > 1 {
                errors.insert(IntegrityError::ConflictingDependencyRevision {
                    dependency_key,
                    revisions: revisions.into_iter().collect(),
                });
            }
        }

        let mut seen_edges = BTreeSet::new();
        for edge in &self.edges {
            if !nodes_by_id.contains_key(&edge.from) {
                errors.insert(IntegrityError::MissingNode { node_id: edge.from.clone() });
            }
            if !nodes_by_id.contains_key(&edge.to) {
                errors.insert(IntegrityError::MissingNode { node_id: edge.to.clone() });
            }

            if !seen_edges.insert((edge.from.clone(), edge.to.clone(), edge.kind.clone())) {
                errors.insert(IntegrityError::DuplicateEdge {
                    from: edge.from.clone(),
                    to: edge.to.clone(),
                    kind: edge.kind.clone(),
                });
            }

            if edge.kind == EdgeKind::DerivedFrom && edge.from == edge.to {
                errors.insert(IntegrityError::SelfDerivation { node_id: edge.from.clone() });
            }
        }

        for edge in &self.edges {
            if edge.kind == EdgeKind::References {
                if let Some(node) = nodes_by_id.get(&edge.to) {
                    if node.kind != NodeKind::Witness {
                        errors.insert(IntegrityError::InvalidWitnessNode {
                            witness_id: edge.to.clone(),
                        });
                    }
                } else {
                    errors.insert(IntegrityError::WitnessReferenceMissing {
                        witness_id: edge.to.clone(),
                    });
                }
            }
        }

        for claim in self.nodes.iter().filter(|node| node.kind == NodeKind::Claim) {
            let required = required_dependencies(claim, &self.edges, &nodes_by_id);
            let observed = reachable_kinds(&claim.id, &self.edges, &nodes_by_id);
            let observed_required = observed
                .intersection(&required)
                .cloned()
                .collect::<BTreeSet<_>>();

            if let Some(certificate) = self.certificates.iter().find(|c| c.claim_id == claim.id) {
                if !certificate.is_truthful()
                    || certificate.required != required
                    || certificate.observed != observed_required
                {
                    errors.insert(IntegrityError::CompletenessCertificateMismatch {
                        claim_id: claim.id.clone(),
                    });
                }
            } else if !required.is_empty() {
                errors.insert(IntegrityError::CompletenessCertificateMismatch {
                    claim_id: claim.id.clone(),
                });
            }

            for dependency in required {
                if !observed.contains(&dependency) {
                    errors.insert(IntegrityError::MissingRequiredDependency {
                        claim_id: claim.id.clone(),
                        dependency,
                    });
                }
            }
        }

        for cycle in derivation_cycles(&self.edges) {
            errors.insert(IntegrityError::DerivationCycle { nodes: cycle });
        }

        let mut normalized_nodes = self.nodes.iter().map(|n| n.id.clone()).collect::<Vec<_>>();
        normalized_nodes.sort();
        normalized_nodes.dedup();

        let mut normalized_edges = self.edges.clone();
        normalized_edges.sort_by(|a, b| {
            a.from
                .cmp(&b.from)
                .then_with(|| a.to.cmp(&b.to))
                .then_with(|| a.kind.cmp(&b.kind))
        });

        let status = if errors.is_empty() {
            IntegrityStatus::Valid
        } else {
            IntegrityStatus::Invalid
        };

        IntegrityReport {
            status,
            errors: errors.into_iter().collect(),
            normalized_node_ids: normalized_nodes,
            normalized_edges,
        }
    }
}

fn required_dependencies(
    claim: &Node,
    edges: &[Edge],
    nodes: &BTreeMap<String, &Node>,
) -> BTreeSet<NodeKind> {
    let mut required = BTreeSet::new();
    let outgoing = edges.iter().any(|edge| {
        edge.from == claim.id && edge.kind == EdgeKind::DerivedFrom
    });

    // The SWA reference claim is a prediction-validation claim. Its minimum
    // reproducible dependency closure is explicit rather than inferred from
    // whichever nodes happen to be reachable today.
    if outgoing {
        required.extend([
            NodeKind::Evidence,
            NodeKind::Prediction,
            NodeKind::Model,
            NodeKind::Parameters,
            NodeKind::Scenario,
            NodeKind::Dataset,
        ]);
    }

    let _ = nodes;
    required
}

fn reachable_kinds(
    root: &str,
    edges: &[Edge],
    nodes: &BTreeMap<String, &Node>,
) -> BTreeSet<NodeKind> {
    let mut seen = BTreeSet::new();
    let mut frontier = vec![root.to_string()];

    while let Some(current) = frontier.pop() {
        for edge in edges.iter().filter(|edge| edge.from == current) {
            if seen.insert(edge.to.clone()) {
                frontier.push(edge.to.clone());
            }
        }
    }

    seen.into_iter()
        .filter_map(|id| nodes.get(&id).map(|node| node.kind.clone()))
        .collect()
}

fn derivation_cycles(edges: &[Edge]) -> Vec<Vec<String>> {
    let mut adjacency = BTreeMap::<String, BTreeSet<String>>::new();
    for edge in edges.iter().filter(|edge| edge.kind == EdgeKind::DerivedFrom) {
        adjacency.entry(edge.from.clone()).or_default().insert(edge.to.clone());
    }

    let mut cycles = BTreeSet::<Vec<String>>::new();
    for start in adjacency.keys() {
        let mut path = Vec::new();
        let mut visiting = BTreeSet::new();
        dfs_cycle(
            start,
            &adjacency,
            &mut path,
            &mut visiting,
            &mut cycles,
        );
    }
    cycles.into_iter().collect()
}

fn dfs_cycle(
    current: &str,
    adjacency: &BTreeMap<String, BTreeSet<String>>,
    path: &mut Vec<String>,
    visiting: &mut BTreeSet<String>,
    cycles: &mut BTreeSet<Vec<String>>,
) {
    if !visiting.insert(current.to_string()) {
        if let Some(index) = path.iter().position(|id| id == current) {
            let mut cycle = path[index..].to_vec();
            cycle.sort();
            cycle.dedup();
            cycles.insert(cycle);
        }
        return;
    }

    path.push(current.to_string());
    if let Some(next) = adjacency.get(current) {
        for node in next {
            dfs_cycle(node, adjacency, path, visiting, cycles);
        }
    }
    path.pop();
    visiting.remove(current);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, kind: NodeKind) -> Node {
        Node {
            id: id.into(),
            kind,
            dependency_key: None,
            revision: None,
        }
    }

    fn valid_graph() -> ProvenanceGraph {
        ProvenanceGraph {
            nodes: vec![
                node("claim-001", NodeKind::Claim),
                node("evidence-001", NodeKind::Evidence),
                node("prediction-001", NodeKind::Prediction),
                node("model-001", NodeKind::Model),
                node("parameters-001", NodeKind::Parameters),
                node("scenario-001", NodeKind::Scenario),
                node("dataset-001", NodeKind::Dataset),
                node("witness-001", NodeKind::Witness),
            ],
            edges: vec![
                Edge { from: "claim-001".into(), to: "evidence-001".into(), kind: EdgeKind::DerivedFrom },
                Edge { from: "evidence-001".into(), to: "prediction-001".into(), kind: EdgeKind::DerivedFrom },
                Edge { from: "prediction-001".into(), to: "model-001".into(), kind: EdgeKind::DerivedFrom },
                Edge { from: "prediction-001".into(), to: "parameters-001".into(), kind: EdgeKind::DerivedFrom },
                Edge { from: "prediction-001".into(), to: "scenario-001".into(), kind: EdgeKind::DerivedFrom },
                Edge { from: "evidence-001".into(), to: "dataset-001".into(), kind: EdgeKind::DerivedFrom },
                Edge { from: "claim-001".into(), to: "witness-001".into(), kind: EdgeKind::References },
            ],
            certificates: vec![CompletenessCertificate {
                claim_id: "claim-001".into(),
                required: [
                    NodeKind::Evidence,
                    NodeKind::Prediction,
                    NodeKind::Model,
                    NodeKind::Parameters,
                    NodeKind::Scenario,
                    NodeKind::Dataset,
                ]
                .into_iter()
                .collect(),
                observed: [
                    NodeKind::Evidence,
                    NodeKind::Prediction,
                    NodeKind::Model,
                    NodeKind::Parameters,
                    NodeKind::Scenario,
                    NodeKind::Dataset,
                ]
                .into_iter()
                .collect(),
            }],
        }
    }

    #[test]
    fn valid_reference_graph_is_accepted() {
        assert_eq!(valid_graph().validate().status, IntegrityStatus::Valid);
    }

    #[test]
    fn missing_reference_node_is_invalid() {
        let mut graph = valid_graph();
        graph.edges.push(Edge {
            from: "claim-001".into(),
            to: "missing".into(),
            kind: EdgeKind::References,
        });
        assert_eq!(graph.validate().status, IntegrityStatus::Invalid);
    }

    #[test]
    fn duplicate_node_identity_is_invalid() {
        let mut graph = valid_graph();
        graph.nodes.push(node("claim-001", NodeKind::Claim));
        assert!(graph
            .validate()
            .errors
            .contains(&IntegrityError::DuplicateNodeIdentity { id: "claim-001".into() }));
    }

    #[test]
    fn conflicting_dependency_revisions_are_invalid() {
        let mut graph = valid_graph();
        graph.nodes.push(Node {
            id: "model-002".into(),
            kind: NodeKind::Model,
            dependency_key: Some("thermal-model".into()),
            revision: Some("m7".into()),
        });
        graph.nodes.push(Node {
            id: "model-003".into(),
            kind: NodeKind::Model,
            dependency_key: Some("thermal-model".into()),
            revision: Some("m8".into()),
        });
        assert!(graph.validate().errors.contains(
            &IntegrityError::ConflictingDependencyRevision {
                dependency_key: "thermal-model".into(),
                revisions: vec!["m7".into(), "m8".into()],
            }
        ));
    }

    #[test]
    fn derivation_cycle_is_invalid() {
        let mut graph = valid_graph();
        graph.edges.push(Edge {
            from: "model-001".into(),
            to: "prediction-001".into(),
            kind: EdgeKind::DerivedFrom,
        });
        assert!(graph
            .validate()
            .errors
            .iter()
            .any(|error| matches!(error, IntegrityError::DerivationCycle { .. })));
    }

    #[test]
    fn support_and_contradiction_do_not_form_derivation_cycles() {
        let mut graph = valid_graph();
        graph.edges.push(Edge {
            from: "witness-001".into(),
            to: "claim-001".into(),
            kind: EdgeKind::Contradicts,
        });
        graph.edges.push(Edge {
            from: "claim-001".into(),
            to: "witness-001".into(),
            kind: EdgeKind::Supports,
        });
        assert!(!graph
            .validate()
            .errors
            .iter()
            .any(|error| matches!(error, IntegrityError::DerivationCycle { .. })));
    }

    #[test]
    fn completeness_certificate_cannot_claim_more_than_observed() {
        let mut graph = valid_graph();
        graph.certificates[0].observed.insert(NodeKind::Model);
        assert!(graph
            .validate()
            .errors
            .contains(&IntegrityError::CompletenessCertificateMismatch {
                claim_id: "claim-001".into(),
            }));
    }

    #[test]
    fn normalization_is_order_independent() {
        let mut a = valid_graph();
        let mut b = valid_graph();
        a.nodes.reverse();
        a.edges.reverse();
        assert_eq!(
            a.validate().normalized_node_ids,
            b.validate().normalized_node_ids
        );
        assert_eq!(a.validate().normalized_edges, b.validate().normalized_edges);
    }

    #[test]
    fn historical_integrity_does_not_grant_authority() {
        let report = valid_graph().validate();
        assert_eq!(report.status, IntegrityStatus::Valid);
        // A valid provenance structure describes evidence integrity only; it
        // does not encode an authorization relation or physical control path.
    }
}
