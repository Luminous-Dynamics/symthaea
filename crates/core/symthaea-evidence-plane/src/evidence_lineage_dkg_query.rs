// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-addressed query receipts for canonical evidence-lineage DKGs.
//!
//! A receipt proves only the membership/absence of a node in a specific,
//! manifest-bound canonical projection. It is not a scientific judgment,
//! consensus signal, authority credential, or digital signature.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::evidence_lineage_dkg::{DkgNodeType, EvidenceLineageDkgProjection};
use crate::evidence_lineage_federation::{
    EvidenceLineageDkgFederationManifest, CANONICALIZATION_ID, DIGEST_ALGORITHM,
    PROJECTION_SCHEMA_VERSION,
};

const RECEIPT_DIGEST_DOMAIN: &[u8] =
    b"symthaea:evidence-lineage-dkg-query-receipt:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryResult {
    Present,
    Absent,
}

impl QueryResult {
    fn as_str(self) -> &'static str {
        match self {
            Self::Present => "present",
            Self::Absent => "absent",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLineageDkgQueryReceipt {
    pub receipt_version: String,
    pub projection_schema_version: String,
    pub canonicalization_id: String,
    pub digest_algorithm: String,
    pub manifest_digest: String,
    pub projection_digest: String,
    pub node_id: String,
    pub node_type: DkgNodeType,
    pub result: QueryResult,
    pub record_digest: Option<String>,
    pub receipt_digest: String,
}

impl EvidenceLineageDkgQueryReceipt {
    /// Create a receipt from a manifest-bound canonical projection.
    pub fn query(
        manifest: &EvidenceLineageDkgFederationManifest,
        projection: &EvidenceLineageDkgProjection,
        node_id: impl Into<String>,
        node_type: DkgNodeType,
    ) -> Result<Self, QueryReceiptError> {
        manifest
            .verify_against(projection)
            .map_err(QueryReceiptError::InvalidManifest)?;

        let node_id = node_id.into();
        if node_id.trim().is_empty() {
            return Err(QueryReceiptError::MissingNodeId);
        }

        let matching = projection
            .nodes
            .iter()
            .find(|node| node.node_id == node_id);

        let (result, record_digest) = match matching {
            Some(node) if node.node_type == node_type => {
                (QueryResult::Present, Some(node.record_digest.clone()))
            }
            Some(_) => return Err(QueryReceiptError::NodeTypeMismatch),
            None => (QueryResult::Absent, None),
        };

        let mut receipt = Self {
            receipt_version: "1.0.0".into(),
            projection_schema_version: PROJECTION_SCHEMA_VERSION.into(),
            canonicalization_id: CANONICALIZATION_ID.into(),
            digest_algorithm: DIGEST_ALGORITHM.into(),
            manifest_digest: manifest.manifest_digest.clone(),
            projection_digest: projection.projection_digest.clone(),
            node_id,
            node_type,
            result,
            record_digest,
            receipt_digest: String::new(),
        };
        receipt.receipt_digest = receipt.compute_digest();
        Ok(receipt)
    }

    /// Verify the receipt against the exact manifest and canonical projection.
    pub fn verify_against(
        &self,
        manifest: &EvidenceLineageDkgFederationManifest,
        projection: &EvidenceLineageDkgProjection,
    ) -> Result<(), QueryReceiptError> {
        manifest
            .verify_against(projection)
            .map_err(QueryReceiptError::InvalidManifest)?;

        if self.receipt_version != "1.0.0" {
            return Err(QueryReceiptError::ReceiptVersionMismatch);
        }
        if self.projection_schema_version != PROJECTION_SCHEMA_VERSION
            || self.projection_schema_version != projection.projection_version
        {
            return Err(QueryReceiptError::ProjectionVersionMismatch);
        }
        if self.canonicalization_id != CANONICALIZATION_ID {
            return Err(QueryReceiptError::CanonicalizationMismatch);
        }
        if self.digest_algorithm != DIGEST_ALGORITHM {
            return Err(QueryReceiptError::DigestAlgorithmMismatch);
        }
        if self.manifest_digest != manifest.manifest_digest {
            return Err(QueryReceiptError::ManifestDigestMismatch);
        }
        if self.projection_digest != projection.projection_digest {
            return Err(QueryReceiptError::ProjectionDigestMismatch);
        }
        if self.node_id.trim().is_empty() {
            return Err(QueryReceiptError::MissingNodeId);
        }

        let matching = projection.nodes.iter().find(|n| n.node_id == self.node_id);
        match (self.result, matching) {
            (QueryResult::Present, Some(node)) if node.node_type == self.node_type => {
                if self.record_digest.as_deref() != Some(node.record_digest.as_str()) {
                    return Err(QueryReceiptError::RecordDigestMismatch);
                }
            }
            (QueryResult::Present, Some(_)) => return Err(QueryReceiptError::NodeTypeMismatch),
            (QueryResult::Present, None) => return Err(QueryReceiptError::MembershipMismatch),
            (QueryResult::Absent, None) => {
                if self.record_digest.is_some() {
                    return Err(QueryReceiptError::UnexpectedRecordDigest);
                }
            }
            (QueryResult::Absent, Some(_)) => return Err(QueryReceiptError::AbsenceMismatch),
        }

        if self.receipt_digest != self.compute_digest() {
            return Err(QueryReceiptError::ReceiptDigestMismatch);
        }
        Ok(())
    }

    pub fn receipt_digest(&self) -> String {
        self.compute_digest()
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(RECEIPT_DIGEST_DOMAIN);
        put(&mut h, &self.receipt_version);
        put(&mut h, &self.projection_schema_version);
        put(&mut h, &self.canonicalization_id);
        put(&mut h, &self.digest_algorithm);
        put(&mut h, &self.manifest_digest);
        put(&mut h, &self.projection_digest);
        put(&mut h, &self.node_id);
        put(&mut h, node_type(self.node_type));
        put(&mut h, self.result.as_str());
        put(&mut h, self.record_digest.as_deref().unwrap_or(""));
        format!("sha256:{:x}", h.finalize())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryReceiptError {
    InvalidManifest(crate::evidence_lineage_federation::FederationManifestError),
    MissingNodeId,
    ReceiptVersionMismatch,
    ProjectionVersionMismatch,
    CanonicalizationMismatch,
    DigestAlgorithmMismatch,
    ManifestDigestMismatch,
    ProjectionDigestMismatch,
    NodeTypeMismatch,
    MembershipMismatch,
    AbsenceMismatch,
    RecordDigestMismatch,
    UnexpectedRecordDigest,
    ReceiptDigestMismatch,
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

    #[test]
    fn membership_receipt_verifies() {
        let p = projection();
        let m = manifest();
        let r = EvidenceLineageDkgQueryReceipt::query(
            &m, &p, "observation:1", DkgNodeType::ExternalObservation,
        ).unwrap();
        assert_eq!(r.result, QueryResult::Present);
        assert_eq!(r.record_digest.as_deref(), Some("sha256:observation"));
        assert!(r.verify_against(&m, &p).is_ok());
    }

    #[test]
    fn absence_receipt_verifies_against_complete_manifest() {
        let p = projection();
        let m = manifest();
        let r = EvidenceLineageDkgQueryReceipt::query(
            &m, &p, "observation:missing", DkgNodeType::ExternalObservation,
        ).unwrap();
        assert_eq!(r.result, QueryResult::Absent);
        assert!(r.record_digest.is_none());
        assert!(r.verify_against(&m, &p).is_ok());
    }

    #[test]
    fn wrong_type_is_rejected() {
        let p = projection();
        let m = manifest();
        assert_eq!(
            EvidenceLineageDkgQueryReceipt::query(
                &m, &p, "observation:1", DkgNodeType::Replication,
            ),
            Err(QueryReceiptError::NodeTypeMismatch)
        );
    }

    #[test]
    fn tampered_record_digest_is_rejected() {
        let p = projection();
        let m = manifest();
        let mut r = EvidenceLineageDkgQueryReceipt::query(
            &m, &p, "observation:1", DkgNodeType::ExternalObservation,
        ).unwrap();
        r.record_digest = Some("sha256:tampered".into());
        assert_eq!(
            r.verify_against(&m, &p),
            Err(QueryReceiptError::RecordDigestMismatch)
        );
    }

    #[test]
    fn tampered_projection_binding_is_rejected() {
        let p = projection();
        let m = manifest();
        let mut r = EvidenceLineageDkgQueryReceipt::query(
            &m, &p, "observation:1", DkgNodeType::ExternalObservation,
        ).unwrap();
        r.projection_digest = "sha256:tampered".into();
        assert_eq!(
            r.verify_against(&m, &p),
            Err(QueryReceiptError::ProjectionDigestMismatch)
        );
    }

    #[test]
    fn tampered_receipt_digest_is_rejected() {
        let p = projection();
        let m = manifest();
        let mut r = EvidenceLineageDkgQueryReceipt::query(
            &m, &p, "observation:1", DkgNodeType::ExternalObservation,
        ).unwrap();
        r.receipt_digest = "sha256:tampered".into();
        assert_eq!(
            r.verify_against(&m, &p),
            Err(QueryReceiptError::ReceiptDigestMismatch)
        );
    }

    #[test]
    fn serde_roundtrip_preserves_receipt() {
        let p = projection();
        let m = manifest();
        let r = EvidenceLineageDkgQueryReceipt::query(
            &m, &p, "observation:1", DkgNodeType::ExternalObservation,
        ).unwrap();
        let encoded = serde_json::to_vec(&r).unwrap();
        let decoded: EvidenceLineageDkgQueryReceipt = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(r, decoded);
        assert!(decoded.verify_against(&m, &p).is_ok());
    }
}
