//! Shared deterministic provenance graph primitives for the Sol Atlas fixtures.
//!
//! This module is the authoritative in-process representation used by the
//! SWA provenance slice example and downstream evidence bindings. Consumers
//! may project a selected slice into another wire format, but they must not
//! reconstruct the graph topology independently.

use blake3::Hasher;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const PROVENANCE_GRAPH_REVISION: &str = "sol-atlas-reference-graph@v1";
pub const PROVENANCE_SLICE_TRAVERSAL_POLICY: &str = "outgoing-reachability-bfs@v1";
const PROVENANCE_GRAPH_ENCODING_VERSION: &[u8] = b"symthaea:provenance-graph:v1";

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
    /// Revision is part of provenance identity, not adapter-local metadata.
    pub revision: Option<&'static str>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceGraphError {
    DuplicateNodeId,
    DuplicateEdge,
    MissingEdgeEndpoint,
    InvalidNodeIdentity,
    MissingRoot,
}

impl ProvenanceGraph {
    /// Deterministic identity commitment for the exact authoritative graph.
    pub fn identity_digest(&self) -> String {
        let mut hasher = Hasher::new();
        hasher.update(PROVENANCE_GRAPH_ENCODING_VERSION);
        let mut nodes = self.nodes.clone();
        nodes.sort();
        for node in nodes {
            hasher.update(node.id.as_bytes());
            hasher.update(&[0]);
            hasher.update(format!("{:?}", node.kind).as_bytes());
            hasher.update(&[0]);
            if let Some(revision) = node.revision { hasher.update(revision.as_bytes()); }
            hasher.update(&[0xff]);
        }
        let mut edges = self.edges.clone();
        edges.sort();
        for edge in edges {
            hasher.update(edge.from.as_bytes());
            hasher.update(&[0]);
            hasher.update(edge.to.as_bytes());
            hasher.update(&[0]);
            hasher.update(format!("{:?}", edge.kind).as_bytes());
            hasher.update(&[0xff]);
        }
        hasher.finalize().to_hex().to_string()
    }

    /// Validate graph identity before any downstream projection consumes it.
    ///
    /// Node IDs are semantic identities, so duplicates are ambiguous. Edges
    /// must be unique and must reference nodes in the same graph.
    pub fn validate(&self) -> Result<(), ProvenanceGraphError> {
        let mut node_ids = BTreeSet::new();
        for node in &self.nodes {
            if node.id.is_empty() || node.id.trim() != node.id || node.id.chars().any(char::is_control) {
                return Err(ProvenanceGraphError::InvalidNodeIdentity);
            }
            if let Some(revision) = node.revision {
                if revision.is_empty() || revision.trim() != revision || revision.chars().any(char::is_control) {
                    return Err(ProvenanceGraphError::InvalidNodeIdentity);
                }
            }
            if !node_ids.insert(node.id) {
                return Err(ProvenanceGraphError::DuplicateNodeId);
            }
        }

        let mut edges = BTreeSet::new();
        for edge in &self.edges {
            if !node_ids.contains(edge.from) || !node_ids.contains(edge.to) {
                return Err(ProvenanceGraphError::MissingEdgeEndpoint);
            }
            if !edges.insert(*edge) {
                return Err(ProvenanceGraphError::DuplicateEdge);
            }
        }

        Ok(())
    }


    pub fn slice(&self, root: &'static str) -> Result<ProvenanceSlice, ProvenanceGraphError> {
        self.validate()?;
        if !self.nodes.iter().any(|node| node.id == root) {
            return Err(ProvenanceGraphError::MissingRoot);
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

        let boundary = SliceBoundaryCertificate {
            root,
            graph_revision: PROVENANCE_GRAPH_REVISION,
            graph_digest: self.identity_digest(),
            traversal_policy: PROVENANCE_SLICE_TRAVERSAL_POLICY,
            graph_node_count: self.nodes.len(),
            graph_edge_count: self.edges.len(),
            slice_node_count: nodes.len(),
            slice_edge_count: edges.len(),
            frontier_exhausted: true,
        };

        Ok(ProvenanceSlice {
            root,
            nodes,
            edges,
            boundary,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProvenanceSlice {
    pub root: &'static str,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    /// Structural certificate issued by the authoritative traversal itself.
    /// This is about traversal completeness, not semantic truth: SWA-010
    /// remains responsible for claim-specific dependency requirements.
    pub boundary: SliceBoundaryCertificate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SliceBoundaryCertificate {
    pub root: &'static str,
    /// Versioned identity of the authoritative graph representation.
    pub graph_revision: &'static str,
    pub traversal_policy: &'static str,
    pub graph_node_count: usize,
    pub graph_edge_count: usize,
    pub slice_node_count: usize,
    pub slice_edge_count: usize,
    pub frontier_exhausted: bool,
}

impl SliceBoundaryCertificate {
    pub fn is_complete(&self) -> bool {
        self.frontier_exhausted
            && self.graph_digest.len() == 64
            && self.graph_digest.chars().all(|c| c.is_ascii_hexdigit())
            && self.graph_revision == PROVENANCE_GRAPH_REVISION
            && self.traversal_policy == PROVENANCE_SLICE_TRAVERSAL_POLICY

            && self.slice_node_count <= self.graph_node_count
            && self.slice_edge_count <= self.graph_edge_count
    }
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
            Node { id: "claim-001", kind: NodeKind::Claim, revision: Some("claim@v1") },
            Node { id: "validation-001", kind: NodeKind::Evidence, revision: Some("evidence@v1") },
            Node { id: "prediction-001", kind: NodeKind::Prediction, revision: Some("prediction@v1") },
            Node { id: "model-001", kind: NodeKind::Model, revision: Some("building-twin@fixture") },
            Node { id: "parameters-001", kind: NodeKind::Parameters, revision: Some("parameters@v1") },
            Node { id: "scenario-001", kind: NodeKind::Scenario, revision: Some("intervention-scenario@v1") },
            Node { id: "dataset-001", kind: NodeKind::Dataset, revision: Some("dataset@v1") },
            Node { id: "context-001", kind: NodeKind::ContextOfUse, revision: Some("context@v1") },
            Node { id: "unrelated-001", kind: NodeKind::Dataset, revision: Some("unrelated@v1") },
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
    fn invalid_node_identity_is_rejected() {
        let mut graph = reference_graph();
        graph.nodes[0].revision = Some("bad\nrevision");
        assert_eq!(graph.validate(), Err(ProvenanceGraphError::InvalidNodeIdentity));
    }

    #[test]
    fn reference_graph_is_well_formed() {
        assert_eq!(reference_graph().validate(), Ok(()));
    }

    #[test]
    fn duplicate_node_identity_is_rejected() {
        let mut graph = reference_graph();
        graph.nodes.push(Node { id: "claim-001", kind: NodeKind::Evidence });
        assert_eq!(
            graph.validate(),
            Err(ProvenanceGraphError::DuplicateNodeId)
        );
    }

    #[test]
    fn dangling_edge_is_rejected() {
        let mut graph = reference_graph();
        graph.edges.push(Edge {
            from: "claim-001",
            to: "missing-001",
            kind: EdgeKind::DerivedFrom,
        });
        assert_eq!(
            graph.validate(),
            Err(ProvenanceGraphError::MissingEdgeEndpoint)
        );
    }

    #[test]
    fn revision_is_part_of_authoritative_node_identity() {
        let mut a = reference_graph();
        let mut b = reference_graph();
        a.nodes.iter_mut().find(|n| n.id == "model-001").unwrap().revision = Some("model@v1");
        b.nodes.iter_mut().find(|n| n.id == "model-001").unwrap().revision = Some("model@v2");
        assert_ne!(a, b);
    }

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
        graph.nodes.push(Node { id: "counter-001", kind: NodeKind::Evidence, revision: Some("counter@v1") });
        graph.nodes.push(Node { id: "boundary-001", kind: NodeKind::Evidence, revision: Some("boundary@v1") });
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
    fn invalid_graph_cannot_produce_a_slice() {
        let mut graph = reference_graph();
        graph.edges.push(Edge {
            from: "claim-001",
            to: "missing-001",
            kind: EdgeKind::DerivedFrom,
        });
        assert_eq!(
            graph.slice("claim-001"),
            Err(ProvenanceGraphError::MissingEdgeEndpoint)
        );
    }

    #[test]
    fn missing_root_is_not_an_empty_slice() {
        assert_eq!(reference_graph().slice("missing"), Err(ProvenanceGraphError::MissingRoot));
    }
}
