//! Canonical evidence-slice commitments for cross-system projections.
//!
//! The provenance graph remains the source of semantic truth. This module only
//! defines the deterministic wire commitment emitted from an already-selected
//! slice; it does not build a second provenance graph or interpret evidence.

use blake3;
use serde::Serialize;

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

    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(&self.canonicalize()).expect("manifest serializes")
    }

    pub fn digest(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
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
