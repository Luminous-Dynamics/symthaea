// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-addressed ancestry-path receipts for canonical evidence-lineage DKGs.
//!
//! A path receipt proves only that an explicitly ordered sequence of nodes and
//! edges exists in one exact manifest-bound canonical projection. It is not a
//! scientific judgment, consensus signal, authority credential, criterion
//! completion claim, or digital signature.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::evidence_lineage_dkg::{DkgEdgeType, DkgNodeType, EvidenceLineageDkgProjection};
use crate::evidence_lineage_federation::{
    EvidenceLineageDkgFederationManifest, CANONICALIZATION_ID, DIGEST_ALGORITHM,
    PROJECTION_SCHEMA_VERSION,
};

const PATH_RECEIPT_DIGEST_DOMAIN: &[u8] =
    b"symthaea:evidence-lineage-dkg-path-receipt:v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathNode {
    pub node_id: String,
    pub node_type: DkgNodeType,
    pub record_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathEdge {
    pub source_node_id: String,
    pub edge_type: DkgEdgeType,
    pub target_node_id: String,
}

/// An ordered provenance path bound to an exact federation manifest and DKG projection.
///
/// For ordinary ancestry edges the canonical graph points from the descendant
/// toward its parent (for example, observation -> commitment). The receipt may
/// present nodes in root-to-terminal order; verification checks the exact graph
/// edge between adjacent path nodes without rewriting edge direction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLineageDkgPathReceipt {
    pub receipt_version: String,
    pub projection_schema_version: String,
    pub canonicalization_id: String,
    pub digest_algorithm: String,
    pub manifest_digest: String,
    pub projection_digest: String,
    pub nodes: Vec<PathNode>,
    pub edges: Vec<PathEdge>,
    pub receipt_digest: String,
}

impl EvidenceLineageDkgPathReceipt {
    /// Create a receipt for an explicitly supplied ordered node path.
    pub fn path(
        manifest: &EvidenceLineageDkgFederationManifest,
        projection: &EvidenceLineageDkgProjection,
        nodes: Vec<PathNode>,
        edges: Vec<PathEdge>,
    ) -> Result<Self, PathReceiptError> {
        manifest.verify_against(projection)
            .map_err(PathReceiptError::InvalidManifest)?;
        validate_path(projection, &nodes, &edges)?;

        let mut receipt = Self {
            receipt_version: "1.0.0".into(),
            projection_schema_version: PROJECTION_SCHEMA_VERSION.into(),
            canonicalization_id: CANONICALIZATION_ID.into(),
            digest_algorithm: DIGEST_ALGORITHM.into(),
            manifest_digest: manifest.manifest_digest.clone(),
            projection_digest: projection.projection_digest.clone(),
            nodes,
            edges,
            receipt_digest: String::new(),
        };
        receipt.receipt_digest = receipt.compute_digest();
        Ok(receipt)
    }

    /// Create the canonical evidence ancestry from commitment through terminal
    /// criterion evidence. If exactly one audit node is present, append it.
    pub fn full_evidence_lineage(
        manifest: &EvidenceLineageDkgFederationManifest,
        projection: &EvidenceLineageDkgProjection,
    ) -> Result<Self, PathReceiptError> {
        manifest.verify_against(projection)
            .map_err(PathReceiptError::InvalidManifest)?;
        projection.verify_semantic_invariants()
            .map_err(PathReceiptError::InvalidSemanticProjection)?;

        let commitment = unique_node(projection, DkgNodeType::CandidateCommitment)?;
        let observation = unique_node(projection, DkgNodeType::ExternalObservation)?;
        let assessment = unique_node(projection, DkgNodeType::IndependentAssessment)?;
        let replication = unique_node(projection, DkgNodeType::Replication)?;

        let evidence_nodes: Vec<_> = projection.nodes.iter()
            .filter(|n| n.node_type == DkgNodeType::CriterionEvidence)
            .collect();
        let terminal_evidence = evidence_nodes.iter()
            .find(|e| !evidence_nodes.iter().any(|child| {
                child.supersedes_node_id.as_deref() == Some(e.node_id.as_str())
            }))
            .ok_or(PathReceiptError::MissingTerminalEvidence)?;

        let mut nodes = vec![
            path_node(commitment),
            path_node(observation),
            path_node(assessment),
            path_node(replication),
            path_node(terminal_evidence),
        ];
        let mut edges = vec![
            exact_edge(projection, observation.node_id.as_str(), DkgEdgeType::ObservedFrom, commitment.node_id.as_str())?,
            exact_edge(projection, assessment.node_id.as_str(), DkgEdgeType::Assesses, observation.node_id.as_str())?,
            exact_edge(projection, replication.node_id.as_str(), DkgEdgeType::Replicates, assessment.node_id.as_str())?,
            exact_edge(projection, terminal_evidence.node_id.as_str(), DkgEdgeType::EligibleFor, replication.node_id.as_str())?,
        ];

        let audit_nodes: Vec<_> = projection.nodes.iter()
            .filter(|n| n.node_type == DkgNodeType::EvidenceChainAudit)
            .collect();
        if audit_nodes.len() == 1 {
            let audit = audit_nodes[0];
            nodes.push(path_node(audit));
            edges.push(exact_edge(
                projection,
                terminal_evidence.node_id.as_str(),
                DkgEdgeType::AuditedBy,
                audit.node_id.as_str(),
            )?);
        }
        Self::path(manifest, projection, nodes, edges)
    }

    /// Verify the receipt against the exact manifest and canonical projection.
    pub fn verify_against(
        &self,
        manifest: &EvidenceLineageDkgFederationManifest,
        projection: &EvidenceLineageDkgProjection,
    ) -> Result<(), PathReceiptError> {
        manifest.verify_against(projection)
            .map_err(PathReceiptError::InvalidManifest)?;
        if self.receipt_version != "1.0.0" {
            return Err(PathReceiptError::ReceiptVersionMismatch);
        }
        if self.projection_schema_version != PROJECTION_SCHEMA_VERSION
            || self.projection_schema_version != projection.projection_version {
            return Err(PathReceiptError::ProjectionVersionMismatch);
        }
        if self.canonicalization_id != CANONICALIZATION_ID {
            return Err(PathReceiptError::CanonicalizationMismatch);
        }
        if self.digest_algorithm != DIGEST_ALGORITHM {
            return Err(PathReceiptError::DigestAlgorithmMismatch);
        }
        if self.manifest_digest != manifest.manifest_digest {
            return Err(PathReceiptError::ManifestDigestMismatch);
        }
        if self.projection_digest != projection.projection_digest {
            return Err(PathReceiptError::ProjectionDigestMismatch);
        }
        validate_path(projection, &self.nodes, &self.edges)?;
        if self.receipt_digest != self.compute_digest() {
            return Err(PathReceiptError::ReceiptDigestMismatch);
        }
        Ok(())
    }

    pub fn receipt_digest(&self) -> String {
        self.compute_digest()
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(PATH_RECEIPT_DIGEST_DOMAIN);
        put(&mut h, &self.receipt_version);
        put(&mut h, &self.projection_schema_version);
        put(&mut h, &self.canonicalization_id);
        put(&mut h, &self.digest_algorithm);
        put(&mut h, &self.manifest_digest);
        put(&mut h, &self.projection_digest);
        h.update((self.nodes.len() as u64).to_be_bytes());
        for node in &self.nodes {
            put(&mut h, &node.node_id);
            put(&mut h, node_type(node.node_type));
            put(&mut h, &node.record_digest);
        }
        h.update((self.edges.len() as u64).to_be_bytes());
        for edge in &self.edges {
            put(&mut h, &edge.source_node_id);
            put(&mut h, edge_type(edge.edge_type));
            put(&mut h, &edge.target_node_id);
        }
        format!("sha256:{:x}", h.finalize())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathReceiptError {
    InvalidManifest(crate::evidence_lineage_federation::FederationManifestError),
    InvalidSemanticProjection(crate::evidence_lineage_dkg::DkgSemanticError),
    EmptyPath,
    NodeEdgeCountMismatch,
    DuplicatePathNode(String),
    MissingPathNode(String),
    NodeTypeMismatch(String),
    RecordDigestMismatch(String),
    MissingPathEdge { source: String, edge: DkgEdgeType, target: String },
    PathAdjacencyMismatch { index: usize },
    MissingUniqueNode(&'static str),
    MissingTerminalEvidence,
    ReceiptVersionMismatch,
    ProjectionVersionMismatch,
    CanonicalizationMismatch,
    DigestAlgorithmMismatch,
    ManifestDigestMismatch,
    ProjectionDigestMismatch,
    ReceiptDigestMismatch,
}

fn validate_path(
    projection: &EvidenceLineageDkgProjection,
    nodes: &[PathNode],
    edges: &[PathEdge],
) -> Result<(), PathReceiptError> {
    if nodes.is_empty() {
        return Err(PathReceiptError::EmptyPath);
    }
    if edges.len() + 1 != nodes.len() {
        return Err(PathReceiptError::NodeEdgeCountMismatch);
    }

    for (index, path) in nodes.iter().enumerate() {
        if path.node_id.trim().is_empty() || path.record_digest.trim().is_empty() {
            return Err(PathReceiptError::MissingPathNode(path.node_id.clone()));
        }
        if nodes[..index].iter().any(|prior| prior.node_id == path.node_id) {
            return Err(PathReceiptError::DuplicatePathNode(path.node_id.clone()));
        }
        let canonical = projection.nodes.iter()
            .find(|node| node.node_id == path.node_id)
            .ok_or_else(|| PathReceiptError::MissingPathNode(path.node_id.clone()))?;
        if canonical.node_type != path.node_type {
            return Err(PathReceiptError::NodeTypeMismatch(path.node_id.clone()));
        }
        if canonical.record_digest != path.record_digest {
            return Err(PathReceiptError::RecordDigestMismatch(path.node_id.clone()));
        }
    }

    for (index, edge) in edges.iter().enumerate() {
        if edge.source_node_id.trim().is_empty() || edge.target_node_id.trim().is_empty() {
            return Err(PathReceiptError::MissingPathNode(format!("edge:{index}")));
        }
        if projection.edges.iter().find(|candidate| {
            candidate.source_node_id == edge.source_node_id
                && candidate.edge_type == edge.edge_type
                && candidate.target_node_id == edge.target_node_id
        }).is_none() {
            return Err(PathReceiptError::MissingPathEdge {
                source: edge.source_node_id.clone(),
                edge: edge.edge_type,
                target: edge.target_node_id.clone(),
            });
        }

        let current = &nodes[index].node_id;
        let next = &nodes[index + 1].node_id;
        let adjacency_ok = if edge.edge_type == DkgEdgeType::AuditedBy {
            edge.source_node_id == current && edge.target_node_id == next
        } else {
            edge.source_node_id == next && edge.target_node_id == current
        };
        if !adjacency_ok {
            return Err(PathReceiptError::PathAdjacencyMismatch { index });
        }
    }
    Ok(())
}

fn unique_node(
    projection: &EvidenceLineageDkgProjection,
    node_type_value: DkgNodeType,
) -> Result<&crate::evidence_lineage_dkg::DkgNode, PathReceiptError> {
    let mut matches = projection.nodes.iter().filter(|n| n.node_type == node_type_value);
    let first = matches.next().ok_or(PathReceiptError::MissingUniqueNode(node_type(node_type_value)))?;
    if matches.next().is_some() {
        return Err(PathReceiptError::MissingUniqueNode(node_type(node_type_value)));
    }
    Ok(first)
}

fn exact_edge(
    projection: &EvidenceLineageDkgProjection,
    source: &str,
    edge: DkgEdgeType,
    target: &str,
) -> Result<PathEdge, PathReceiptError> {
    projection.edges.iter()
        .find(|candidate| candidate.source_node_id == source
            && candidate.edge_type == edge
            && candidate.target_node_id == target)
        .map(|e| PathEdge {
            source_node_id: e.source_node_id.clone(),
            edge_type: e.edge_type,
            target_node_id: e.target_node_id.clone(),
        })
        .ok_or_else(|| PathReceiptError::MissingPathEdge {
            source: source.into(), edge, target: target.into()
        })
}

fn path_node(node: &crate::evidence_lineage_dkg::DkgNode) -> PathNode {
    PathNode { node_id: node.node_id.clone(), node_type: node.node_type, record_digest: node.record_digest.clone() }
}

fn node_type(value: DkgNodeType) -> &'static str {
    match value {
        DkgNodeType::CandidateCommitment => "candidate_commitment",
        DkgNodeType::ExternalObservation => "external_observation",
        DkgNodeType::IndependentAssessment => "independent_assessment",
        DkgNodeType::Replication => "replication",
        DkgNodeType::CriterionEvidence => "criterion_evidence",
        DkgNodeType::EvidenceChainAudit => "evidence_chain_audit",
    }
}

fn edge_type(value: DkgEdgeType) -> &'static str {
    match value {
        DkgEdgeType::CommitsTo => "commits_to",
        DkgEdgeType::ObservedFrom => "observed_from",
        DkgEdgeType::Assesses => "assesses",
        DkgEdgeType::Replicates => "replicates",
        DkgEdgeType::EligibleFor => "eligible_for",
        DkgEdgeType::AuditedBy => "audited_by",
        DkgEdgeType::Supersedes => "supersedes",
    }
}

fn put(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_lineage_dkg::{DkgEdge, DkgNode};

    fn projection() -> EvidenceLineageDkgProjection {
        EvidenceLineageDkgProjection::new(
            vec![
                DkgNode { node_id: "commitment:1".into(), node_type: DkgNodeType::CandidateCommitment, record_digest: "sha256:commitment".into(), supersedes_node_id: None },
                DkgNode { node_id: "observation:1".into(), node_type: DkgNodeType::ExternalObservation, record_digest: "sha256:observation".into(), supersedes_node_id: None },
                DkgNode { node_id: "assessment:1".into(), node_type: DkgNodeType::IndependentAssessment, record_digest: "sha256:assessment".into(), supersedes_node_id: None },
                DkgNode { node_id: "replication:1".into(), node_type: DkgNodeType::Replication, record_digest: "sha256:replication".into(), supersedes_node_id: None },
                DkgNode { node_id: "evidence:1".into(), node_type: DkgNodeType::CriterionEvidence, record_digest: "sha256:evidence".into(), supersedes_node_id: None },
            ],
            vec![
                DkgEdge { source_node_id: "observation:1".into(), edge_type: DkgEdgeType::ObservedFrom, target_node_id: "commitment:1".into() },
                DkgEdge { source_node_id: "assessment:1".into(), edge_type: DkgEdgeType::Assesses, target_node_id: "observation:1".into() },
                DkgEdge { source_node_id: "replication:1".into(), edge_type: DkgEdgeType::Replicates, target_node_id: "assessment:1".into() },
                DkgEdge { source_node_id: "evidence:1".into(), edge_type: DkgEdgeType::EligibleFor, target_node_id: "replication:1".into() },
            ],
        )
    }

    fn manifest() -> EvidenceLineageDkgFederationManifest {
        let p = projection();
        EvidenceLineageDkgFederationManifest::from_projection(&p, None).unwrap()
    }

    fn nodes() -> Vec<PathNode> {
        vec![
            PathNode { node_id: "commitment:1".into(), node_type: DkgNodeType::CandidateCommitment, record_digest: "sha256:commitment".into() },
            PathNode { node_id: "observation:1".into(), node_type: DkgNodeType::ExternalObservation, record_digest: "sha256:observation".into() },
            PathNode { node_id: "assessment:1".into(), node_type: DkgNodeType::IndependentAssessment, record_digest: "sha256:assessment".into() },
            PathNode { node_id: "replication:1".into(), node_type: DkgNodeType::Replication, record_digest: "sha256:replication".into() },
            PathNode { node_id: "evidence:1".into(), node_type: DkgNodeType::CriterionEvidence, record_digest: "sha256:evidence".into() },
        ]
    }

    fn edges() -> Vec<PathEdge> {
        vec![
            PathEdge { source_node_id: "observation:1".into(), edge_type: DkgEdgeType::ObservedFrom, target_node_id: "commitment:1".into() },
            PathEdge { source_node_id: "assessment:1".into(), edge_type: DkgEdgeType::Assesses, target_node_id: "observation:1".into() },
            PathEdge { source_node_id: "replication:1".into(), edge_type: DkgEdgeType::Replicates, target_node_id: "assessment:1".into() },
            PathEdge { source_node_id: "evidence:1".into(), edge_type: DkgEdgeType::EligibleFor, target_node_id: "replication:1".into() },
        ]
    }

    #[test]
    fn full_lineage_receipt_verifies() {
        let p = projection();
        let m = manifest();
        let r = EvidenceLineageDkgPathReceipt::path(&m, &p, nodes(), edges()).unwrap();
        assert_eq!(r.nodes.len(), 5);
        assert_eq!(r.edges.len(), 4);
        assert!(r.verify_against(&m, &p).is_ok());
    }

    #[test]
    fn full_lineage_constructor_is_content_addressed() {
        let p = projection();
        let m = manifest();
        let r = EvidenceLineageDkgPathReceipt::path(&m, &p, nodes(), edges()).unwrap();
        assert!(r.receipt_digest().starts_with("sha256:"));
    }

    #[test]
    fn skipped_node_is_rejected() {
        let p = projection();
        let m = manifest();
        let mut n = nodes();
        n.remove(2);
        let mut e = edges();
        e.remove(1);
        assert_eq!(
            EvidenceLineageDkgPathReceipt::path(&m, &p, n, e),
            Err(PathReceiptError::PathAdjacencyMismatch { index: 1 })
        );
    }

    #[test]
    fn wrong_edge_is_rejected() {
        let p = projection();
        let m = manifest();
        let mut e = edges();
        e[1].edge_type = DkgEdgeType::Replicates;
        assert!(matches!(
            EvidenceLineageDkgPathReceipt::path(&m, &p, nodes(), e),
            Err(PathReceiptError::MissingPathEdge { .. })
        ));
    }

    #[test]
    fn wrong_record_digest_is_rejected() {
        let p = projection();
        let m = manifest();
        let mut n = nodes();
        n[1].record_digest = "sha256:tampered".into();
        assert_eq!(
            EvidenceLineageDkgPathReceipt::path(&m, &p, n, edges()),
            Err(PathReceiptError::RecordDigestMismatch("observation:1".into()))
        );
    }

    #[test]
    fn reordered_path_is_rejected() {
        let p = projection();
        let m = manifest();
        let mut n = nodes();
        n.swap(0, 1);
        assert_eq!(
            EvidenceLineageDkgPathReceipt::path(&m, &p, n, edges()),
            Err(PathReceiptError::PathAdjacencyMismatch { index: 0 })
        );
    }

    #[test]
    fn tampered_manifest_binding_is_rejected() {
        let p = projection();
        let m = manifest();
        let mut r = EvidenceLineageDkgPathReceipt::path(&m, &p, nodes(), edges()).unwrap();
        r.manifest_digest = "sha256:tampered".into();
        assert_eq!(r.verify_against(&m, &p), Err(PathReceiptError::ManifestDigestMismatch));
    }

    #[test]
    fn tampered_receipt_digest_is_rejected() {
        let p = projection();
        let m = manifest();
        let mut r = EvidenceLineageDkgPathReceipt::path(&m, &p, nodes(), edges()).unwrap();
        r.receipt_digest = "sha256:tampered".into();
        assert_eq!(r.verify_against(&m, &p), Err(PathReceiptError::ReceiptDigestMismatch));
    }

    #[test]
    fn serde_roundtrip_preserves_receipt() {
        let p = projection();
        let m = manifest();
        let r = EvidenceLineageDkgPathReceipt::path(&m, &p, nodes(), edges()).unwrap();
        let encoded = serde_json::to_vec(&r).unwrap();
        let decoded: EvidenceLineageDkgPathReceipt = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(r, decoded);
        assert!(decoded.verify_against(&m, &p).is_ok());
    }
}
