//! Canonical evidence-slice commitments for cross-system projections.
//!
//! The provenance graph remains the source of semantic truth. This module only
//! defines the deterministic wire commitment emitted from an already-selected
//! slice; it does not build a second provenance graph or interpret evidence.

use blake3;
use serde::Serialize;

use crate::provenance_graph::SliceBoundaryCertificate;

const CANONICAL_ENCODING_VERSION: &[u8] = b"symthaea:evidence-slice:v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CanonicalNodeRef {
    pub id: String,
    pub kind: String,
    pub revision: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CanonicalEdgeRef {
    pub from: String,
    pub to: String,
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidenceSliceManifest {
    pub slice_ref: String,
    pub claim_ref: String,
    pub slice_revision: String,
    pub boundary: SliceBoundaryCertificate,
    pub nodes: Vec<CanonicalNodeRef>,
    pub edges: Vec<CanonicalEdgeRef>,
}

impl EvidenceSliceManifest {
    /// Canonicalize order-insensitive graph members before hashing.
    pub fn canonicalize(mut self) -> Self {
        self.nodes.sort_by(|a, b| {
            a.id.cmp(&b.id)
                .then_with(|| a.kind.cmp(&b.kind))
                .then_with(|| a.revision.cmp(&b.revision))
        });
        self.edges.sort_by(|a, b| {
            a.from.cmp(&b.from)
                .then_with(|| a.to.cmp(&b.to))
                .then_with(|| a.kind.cmp(&b.kind))
        });
        self
    }

    /// Versioned, length-delimited canonical bytes. JSON is intentionally not
    /// part of the commitment format so field framing cannot become ambiguous
    /// if the representation evolves.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let canonical = self.clone().canonicalize();
        let mut bytes = Vec::new();
        append_bytes(&mut bytes, CANONICAL_ENCODING_VERSION);
        append_bytes(&mut bytes, canonical.slice_ref.as_bytes());
        append_bytes(&mut bytes, canonical.claim_ref.as_bytes());
        append_bytes(&mut bytes, canonical.slice_revision.as_bytes());
        append_bytes(&mut bytes, canonical.boundary.root.as_bytes());
        append_bytes(&mut bytes, canonical.boundary.graph_revision.as_bytes());
        append_bytes(&mut bytes, canonical.boundary.graph_digest.as_bytes());
        append_bytes(&mut bytes, canonical.boundary.traversal_policy.as_bytes());
        append_u64(&mut bytes, canonical.boundary.graph_node_count);
        append_u64(&mut bytes, canonical.boundary.graph_edge_count);
        append_u64(&mut bytes, canonical.boundary.slice_node_count);
        append_u64(&mut bytes, canonical.boundary.slice_edge_count);
        bytes.push(u8::from(canonical.boundary.frontier_exhausted));
        append_u64(&mut bytes, canonical.nodes.len());
        for node in canonical.nodes {
            append_bytes(&mut bytes, node.id.as_bytes());
            append_bytes(&mut bytes, node.kind.as_bytes());
            append_optional_bytes(&mut bytes, node.revision.as_deref().map(str::as_bytes));
        }
        append_u64(&mut bytes, canonical.edges.len());
        for edge in canonical.edges {
            append_bytes(&mut bytes, edge.from.as_bytes());
            append_bytes(&mut bytes, edge.to.as_bytes());
            append_bytes(&mut bytes, edge.kind.as_bytes());
        }
        bytes
    }

    pub fn digest(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
    }
}

fn append_u64(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u64).to_be_bytes());
}

fn append_bytes(out: &mut Vec<u8>, value: &[u8]) {
    append_u64(out, value.len());
    out.extend_from_slice(value);
}

fn append_optional_bytes(out: &mut Vec<u8>, value: Option<&[u8]>) {
    match value {
        Some(value) => {
            out.push(1);
            append_bytes(out, value);
        }
        None => out.push(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> EvidenceSliceManifest {
        EvidenceSliceManifest {
            slice_ref: "slice-001".into(),
            claim_ref: "claim-001".into(),
            slice_revision: "slice@v1".into(),
            boundary: SliceBoundaryCertificate {
                root: "claim-001",
                graph_revision: "sol-atlas-reference-graph@v1",
                graph_digest: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
                traversal_policy: "outgoing-reachability-bfs@v1",
                graph_node_count: 2,
                graph_edge_count: 1,
                slice_node_count: 2,
                slice_edge_count: 1,
                frontier_exhausted: true,
            },
            nodes: vec![
                CanonicalNodeRef {
                    id: "model-001".into(),
                    kind: "Model".into(),
                    revision: Some("m7".into()),
                },
                CanonicalNodeRef {
                    id: "claim-001".into(),
                    kind: "Claim".into(),
                    revision: Some("c1".into()),
                },
            ],
            edges: vec![CanonicalEdgeRef {
                from: "claim-001".into(),
                to: "model-001".into(),
                kind: "DerivedFrom".into(),
            }],
        }
    }

    #[test]
    fn canonical_encoding_is_versioned_and_framed() {
        let bytes = manifest().canonical_bytes();
        assert!(bytes.starts_with(&(CANONICAL_ENCODING_VERSION.len() as u64).to_be_bytes()));
        assert!(bytes.windows(CANONICAL_ENCODING_VERSION.len()).any(|w| w == CANONICAL_ENCODING_VERSION));
        assert_ne!(manifest().digest(), blake3::hash(b"{}").to_hex().to_string());
    }

    #[test]
    fn boundary_graph_digest_is_part_of_slice_commitment() {
        let baseline = manifest().digest();
        let mut changed = manifest();
        changed.boundary.graph_digest =
            "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210".into();
        assert_ne!(baseline, changed.digest());
    }

    #[test]
    fn boundary_traversal_policy_is_part_of_slice_commitment() {
        let baseline = manifest().digest();
        let mut changed = manifest();
        changed.boundary.traversal_policy = "different-policy@v1";
        assert_ne!(baseline, changed.digest());
    }

    #[test]
    fn optional_revision_is_distinct_from_empty_revision() {
        let mut absent = manifest();
        absent.nodes[0].revision = None;
        let mut empty = manifest();
        empty.nodes[0].revision = Some(String::new());
        assert_ne!(absent.digest(), empty.digest());
    }

    #[test]
    fn member_order_does_not_change_digest() {
        let a = manifest();
        let mut b = manifest();
        b.nodes.reverse();
        b.edges.reverse();
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn revision_change_changes_digest() {
        let a = manifest();
        let mut b = manifest();
        b.nodes[0].revision = Some("m8".into());
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn edge_change_changes_digest() {
        let a = manifest();
        let mut b = manifest();
        b.edges[0].kind = "Contradicts".into();
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn contradiction_is_committed_not_collapsed() {
        let mut a = manifest();
        a.edges.push(CanonicalEdgeRef {
            from: "claim-001".into(),
            to: "counter-001".into(),
            kind: "Contradicts".into(),
        });
        let mut b = a.clone();
        b.edges.pop();
        assert_ne!(a.digest(), b.digest());
    }
}
