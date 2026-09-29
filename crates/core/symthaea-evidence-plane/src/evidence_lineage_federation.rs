// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducibility manifest for federated evidence-lineage DKG indexers.
//!
//! The manifest is a deterministic description of the exact projection inputs,
//! canonicalization contract, and resulting projection digest. It is metadata
//! for federation and reproducibility only; it does not assert scientific truth.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::evidence_lineage_dkg::{DkgNodeType, EvidenceLineageDkgProjection};

pub const MANIFEST_VERSION: &str = "1.0.0";
pub const PROJECTION_SCHEMA_VERSION: &str = "1.2.0";
pub const CANONICALIZATION_ID: &str = "symthaea:dkg-node-edge-order:v1";
pub const DIGEST_ALGORITHM: &str = "sha256";
pub const PROJECTION_DIGEST_DOMAIN: &str = "symthaea:evidence-lineage-dkg-projection:v3";
pub const AUDIT_DIGEST_DOMAIN: &str = "symthaea:evidence-chain-audit:v1";
const MANIFEST_DIGEST_DOMAIN: &[u8] = b"symthaea:evidence-lineage-dkg-federation-manifest:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditInclusionMode {
    Excluded,
    Included,
}

impl AuditInclusionMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Excluded => "excluded",
            Self::Included => "included",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FederationSourceRecord {
    pub node_id: String,
    pub node_type: DkgNodeType,
    pub record_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLineageDkgFederationManifest {
    pub manifest_version: String,
    pub projection_schema_version: String,
    pub canonicalization_id: String,
    pub digest_algorithm: String,
    pub projection_digest_domain: String,
    pub audit_digest_domain: String,
    pub audit_inclusion: AuditInclusionMode,
    pub source_records: Vec<FederationSourceRecord>,
    pub projection_digest: String,
    pub indexer_metadata: Option<String>,
    pub manifest_digest: String,
}

impl EvidenceLineageDkgFederationManifest {
    /// Build a canonical manifest directly from a verified DKG projection.
    pub fn from_projection(
        projection: &EvidenceLineageDkgProjection,
        indexer_metadata: Option<String>,
    ) -> Result<Self, FederationManifestError> {
        if !projection.verify_integrity() {
            return Err(FederationManifestError::InvalidProjection);
        }
        if projection.verify_semantic_invariants().is_err() {
            return Err(FederationManifestError::InvalidSemanticProjection);
        }
        if projection.projection_version != PROJECTION_SCHEMA_VERSION {
            return Err(FederationManifestError::ProjectionVersionMismatch);
        }

        let audit_count = projection
            .nodes
            .iter()
            .filter(|n| n.node_type == DkgNodeType::EvidenceChainAudit)
            .count();
        if audit_count > 1 {
            return Err(FederationManifestError::InvalidAuditInclusion);
        }

        let mut source_records: Vec<FederationSourceRecord> = projection
            .nodes
            .iter()
            .map(|node| FederationSourceRecord {
                node_id: node.node_id.clone(),
                node_type: node.node_type,
                record_digest: node.record_digest.clone(),
            })
            .collect();
        canonical_sort(&mut source_records);

        let mut manifest = Self {
            manifest_version: MANIFEST_VERSION.into(),
            projection_schema_version: PROJECTION_SCHEMA_VERSION.into(),
            canonicalization_id: CANONICALIZATION_ID.into(),
            digest_algorithm: DIGEST_ALGORITHM.into(),
            projection_digest_domain: PROJECTION_DIGEST_DOMAIN.into(),
            audit_digest_domain: AUDIT_DIGEST_DOMAIN.into(),
            audit_inclusion: if audit_count == 0 {
                AuditInclusionMode::Excluded
            } else {
                AuditInclusionMode::Included
            },
            source_records,
            projection_digest: projection.projection_digest.clone(),
            indexer_metadata,
            manifest_digest: String::new(),
        };
        manifest.manifest_digest = manifest.compute_digest();
        Ok(manifest)
    }

    /// Verify that this manifest describes exactly the supplied projection.
    pub fn verify_against(
        &self,
        projection: &EvidenceLineageDkgProjection,
    ) -> Result<(), FederationManifestError> {
        if !projection.verify_integrity() {
            return Err(FederationManifestError::InvalidProjection);
        }
        if projection.verify_semantic_invariants().is_err() {
            return Err(FederationManifestError::InvalidSemanticProjection);
        }
        if self.manifest_version != MANIFEST_VERSION {
            return Err(FederationManifestError::ManifestVersionMismatch);
        }
        if self.projection_schema_version != projection.projection_version
            || self.projection_schema_version != PROJECTION_SCHEMA_VERSION
        {
            return Err(FederationManifestError::ProjectionVersionMismatch);
        }
        if self.canonicalization_id != CANONICALIZATION_ID {
            return Err(FederationManifestError::CanonicalizationMismatch);
        }
        if self.digest_algorithm != DIGEST_ALGORITHM {
            return Err(FederationManifestError::DigestAlgorithmMismatch);
        }
        if self.projection_digest_domain != PROJECTION_DIGEST_DOMAIN
            || self.audit_digest_domain != AUDIT_DIGEST_DOMAIN
        {
            return Err(FederationManifestError::DigestDomainMismatch);
        }
        if self.projection_digest != projection.projection_digest {
            return Err(FederationManifestError::ProjectionDigestMismatch);
        }

        let audit_count = projection
            .nodes
            .iter()
            .filter(|n| n.node_type == DkgNodeType::EvidenceChainAudit)
            .count();
        let expected_mode = if audit_count == 0 {
            AuditInclusionMode::Excluded
        } else if audit_count == 1 {
            AuditInclusionMode::Included
        } else {
            return Err(FederationManifestError::InvalidAuditInclusion);
        };
        if self.audit_inclusion != expected_mode {
            return Err(FederationManifestError::AuditInclusionMismatch);
        }

        let mut expected = projection
            .nodes
            .iter()
            .map(|node| FederationSourceRecord {
                node_id: node.node_id.clone(),
                node_type: node.node_type,
                record_digest: node.record_digest.clone(),
            })
            .collect::<Vec<_>>();
        canonical_sort(&mut expected);

        if has_duplicate_ids(&self.source_records) {
            return Err(FederationManifestError::DuplicateSourceRecord);
        }
        if !is_canonical(&self.source_records) {
            return Err(FederationManifestError::NonCanonicalSourceOrder);
        }
        if self.source_records != expected {
            return Err(FederationManifestError::SourceSetMismatch);
        }
        if self.manifest_digest != self.compute_digest() {
            return Err(FederationManifestError::ManifestDigestMismatch);
        }
        Ok(())
    }

    pub fn manifest_digest(&self) -> String {
        self.compute_digest()
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(MANIFEST_DIGEST_DOMAIN);
        put(&mut h, &self.manifest_version);
        put(&mut h, &self.projection_schema_version);
        put(&mut h, &self.canonicalization_id);
        put(&mut h, &self.digest_algorithm);
        put(&mut h, &self.projection_digest_domain);
        put(&mut h, &self.audit_digest_domain);
        put(&mut h, self.audit_inclusion.as_str());
        h.update((self.source_records.len() as u64).to_be_bytes());
        for source in &self.source_records {
            put(&mut h, &source.node_id);
            put(&mut h, node_type(source.node_type));
            put(&mut h, &source.record_digest);
        }
        put(&mut h, &self.projection_digest);
        match &self.indexer_metadata {
            Some(metadata) => {
                put(&mut h, "some");
                put(&mut h, metadata);
            }
            None => put(&mut h, "none"),
        }
        format!("sha256:{:x}", h.finalize())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FederationManifestError {
    InvalidProjection,
    InvalidSemanticProjection,
    ManifestVersionMismatch,
    ProjectionVersionMismatch,
    CanonicalizationMismatch,
    DigestAlgorithmMismatch,
    DigestDomainMismatch,
    ProjectionDigestMismatch,
    InvalidAuditInclusion,
    AuditInclusionMismatch,
    DuplicateSourceRecord,
    SourceSetMismatch,
    NonCanonicalSourceOrder,
    ManifestDigestMismatch,
}

fn canonical_sort(records: &mut [FederationSourceRecord]) {
    records.sort_by_key(|record| {
        (
            node_type(record.node_type),
            record.node_id.clone(),
            record.record_digest.clone(),
        )
    });
}

fn is_canonical(records: &[FederationSourceRecord]) -> bool {
    records
        .windows(2)
        .all(|pair| {
            (
                node_type(pair[0].node_type),
                &pair[0].node_id,
                &pair[0].record_digest,
            ) <= (
                node_type(pair[1].node_type),
                &pair[1].node_id,
                &pair[1].record_digest,
            )
        })
}

fn has_duplicate_ids(records: &[FederationSourceRecord]) -> bool {
    records
        .windows(2)
        .any(|pair| pair[0].node_id == pair[1].node_id)
        || {
            let mut ids = std::collections::BTreeSet::new();
            records.iter().any(|record| !ids.insert(record.node_id.as_str()))
        }
}

fn node_type(node_type: DkgNodeType) -> &'static str {
    match node_type {
        DkgNodeType::CandidateCommitment => "candidate_commitment",
        DkgNodeType::ExternalObservation => "external_observation",
        DkgNodeType::IndependentAssessment => "independent_assessment",
        DkgNodeType::Replication => "replication",
        DkgNodeType::CriterionEvidence => "criterion_evidence",
        DkgNodeType::EvidenceChainAudit => "evidence_chain_audit",
    }
}

fn put(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_lineage_dkg::{DkgEdge, DkgEdgeType, DkgNode};

    fn projection(with_audit: bool) -> EvidenceLineageDkgProjection {
        let mut nodes = vec![
            DkgNode { node_id: "commitment:1".into(), node_type: DkgNodeType::CandidateCommitment, record_digest: "sha256:commitment".into(), supersedes_node_id: None },
            DkgNode { node_id: "observation:1".into(), node_type: DkgNodeType::ExternalObservation, record_digest: "sha256:observation".into(), supersedes_node_id: None },
            DkgNode { node_id: "assessment:1".into(), node_type: DkgNodeType::IndependentAssessment, record_digest: "sha256:assessment".into(), supersedes_node_id: None },
            DkgNode { node_id: "replication:1".into(), node_type: DkgNodeType::Replication, record_digest: "sha256:replication".into(), supersedes_node_id: None },
            DkgNode { node_id: "evidence:1".into(), node_type: DkgNodeType::CriterionEvidence, record_digest: "sha256:evidence".into(), supersedes_node_id: None },
        ];
        let mut edges = vec![
            DkgEdge { source_node_id: "observation:1".into(), edge_type: DkgEdgeType::ObservedFrom, target_node_id: "commitment:1".into() },
            DkgEdge { source_node_id: "assessment:1".into(), edge_type: DkgEdgeType::Assesses, target_node_id: "observation:1".into() },
            DkgEdge { source_node_id: "replication:1".into(), edge_type: DkgEdgeType::Replicates, target_node_id: "assessment:1".into() },
            DkgEdge { source_node_id: "evidence:1".into(), edge_type: DkgEdgeType::EligibleFor, target_node_id: "replication:1".into() },
        ];
        if with_audit {
            nodes.push(DkgNode {
                node_id: "audit:sha256:a".into(),
                node_type: DkgNodeType::EvidenceChainAudit,
                record_digest: "sha256:a".into(),
                supersedes_node_id: None,
            });
            edges.push(DkgEdge {
                source_node_id: "evidence:1".into(),
                edge_type: DkgEdgeType::AuditedBy,
                target_node_id: "audit:sha256:a".into(),
            });
        }
        EvidenceLineageDkgProjection::new(nodes, edges)
    }

    #[test]
    fn manifest_is_deterministic_under_node_order() {
        let p1 = projection(false);
        let mut p2 = projection(false);
        p2.nodes.reverse();
        let manifest1 = EvidenceLineageDkgFederationManifest::from_projection(&p1, None).unwrap();
        let manifest2 = EvidenceLineageDkgFederationManifest::from_projection(&p2, None).unwrap();
        assert_eq!(manifest1.source_records, manifest2.source_records);
        assert_eq!(manifest1.manifest_digest, manifest2.manifest_digest);
    }

    #[test]
    fn manifest_verifies_projection() {
        let p = projection(false);
        let manifest = EvidenceLineageDkgFederationManifest::from_projection(&p, Some("indexer-a".into())).unwrap();
        assert!(manifest.verify_against(&p).is_ok());
    }

    #[test]
    fn tampered_source_digest_is_rejected() {
        let p = projection(false);
        let mut manifest = EvidenceLineageDkgFederationManifest::from_projection(&p, None).unwrap();
        manifest.source_records[0].record_digest = "sha256:tampered".into();
        manifest.manifest_digest = manifest.manifest_digest();
        assert_eq!(
            manifest.verify_against(&p),
            Err(FederationManifestError::SourceSetMismatch)
        );
    }

    #[test]
    fn duplicate_source_id_is_rejected() {
        let p = projection(false);
        let mut manifest = EvidenceLineageDkgFederationManifest::from_projection(&p, None).unwrap();
        manifest.source_records.push(manifest.source_records[0].clone());
        manifest.source_records.sort_by_key(|r| (node_type(r.node_type), r.node_id.clone(), r.record_digest.clone()));
        manifest.manifest_digest = manifest.manifest_digest();
        assert_eq!(
            manifest.verify_against(&p),
            Err(FederationManifestError::DuplicateSourceRecord)
        );
    }

    #[test]
    fn audit_inclusion_is_explicit() {
        let p = projection(true);
        let manifest = EvidenceLineageDkgFederationManifest::from_projection(&p, None).unwrap();
        assert_eq!(manifest.audit_inclusion, AuditInclusionMode::Included);
        assert!(manifest.verify_against(&p).is_ok());
    }

    #[test]
    fn projection_digest_tampering_is_rejected() {
        let p = projection(false);
        let mut manifest = EvidenceLineageDkgFederationManifest::from_projection(&p, None).unwrap();
        manifest.projection_digest = "sha256:tampered".into();
        manifest.manifest_digest = manifest.manifest_digest();
        assert_eq!(
            manifest.verify_against(&p),
            Err(FederationManifestError::ProjectionDigestMismatch)
        );
    }

    #[test]
    fn noncanonical_source_order_is_rejected() {
        let p = projection(false);
        let mut manifest = EvidenceLineageDkgFederationManifest::from_projection(&p, None).unwrap();
        manifest.source_records.swap(0, 1);
        assert_eq!(
            manifest.verify_against(&p),
            Err(FederationManifestError::SourceSetMismatch)
        );
    }
}
