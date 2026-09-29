// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit federation links between independent DKGs.
//!
//! A cross-DKG link is a provenance reference, not an import of semantic
//! authority. Each graph remains authoritative for its own records. The link
//! binds exact record identities and digests so downstream projections can
//! connect Mycelix, Symthaea, Sol Atlas, or other DKGs without silently
//! promoting a foreign record into evidence.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const LINK_VERSION: &str = "1.0.0";
const DIGEST_ALGORITHM: &str = "sha256";
const DIGEST_DOMAIN: &[u8] = b"symthaea:cross-dkg-federation-link:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossDkgRelation {
    References,
    DerivedFrom,
    ContextFor,
    SameAs,
    Contradicts,
    Supersedes,
}

impl CrossDkgRelation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::References => "references",
            Self::DerivedFrom => "derived_from",
            Self::ContextFor => "context_for",
            Self::SameAs => "same_as",
            Self::Contradicts => "contradicts",
            Self::Supersedes => "supersedes",
        }
    }
}

/// A content-addressed, authority-neutral edge between records owned by
/// independent DKGs.
///
/// The source and target graphs are identified explicitly. A consumer must not
/// infer that either side inherits the other's truth, qualification, identity,
/// or governance semantics from this link.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossDkgFederationLink {
    pub link_version: String,
    pub digest_algorithm: String,
    pub source_dkg_id: String,
    pub source_graph_digest: String,
    pub source_record_id: String,
    pub source_record_digest: String,
    pub relation: CrossDkgRelation,
    pub target_dkg_id: String,
    pub target_graph_digest: String,
    pub target_record_id: String,
    pub target_record_digest: String,
    pub link_digest: String,
}

impl CrossDkgFederationLink {
    /// Construct and content-address a link.
    pub fn new(
        source_dkg_id: impl Into<String>,
        source_graph_digest: impl Into<String>,
        source_record_id: impl Into<String>,
        source_record_digest: impl Into<String>,
        relation: CrossDkgRelation,
        target_dkg_id: impl Into<String>,
        target_graph_digest: impl Into<String>,
        target_record_id: impl Into<String>,
        target_record_digest: impl Into<String>,
    ) -> Result<Self, CrossDkgLinkError> {
        let mut link = Self {
            link_version: LINK_VERSION.into(),
            digest_algorithm: DIGEST_ALGORITHM.into(),
            source_dkg_id: source_dkg_id.into(),
            source_graph_digest: source_graph_digest.into(),
            source_record_id: source_record_id.into(),
            source_record_digest: source_record_digest.into(),
            relation,
            target_dkg_id: target_dkg_id.into(),
            target_graph_digest: target_graph_digest.into(),
            target_record_id: target_record_id.into(),
            target_record_digest: target_record_digest.into(),
            link_digest: String::new(),
        };
        link.validate_fields()?;
        link.link_digest = link.compute_digest();
        Ok(link)
    }

    /// Verify the link envelope and its content address.
    pub fn verify_integrity(&self) -> Result<(), CrossDkgLinkError> {
        self.validate_fields()?;
        if self.link_digest != self.compute_digest() {
            return Err(CrossDkgLinkError::LinkDigestMismatch);
        }
        Ok(())
    }

    /// Verify that the target side names the exact record in an expected
    /// graph. This is useful when adapting the link into the local evidence
    /// plane without importing source semantics.
    pub fn verify_target(
        &self,
        target_dkg_id: &str,
        target_graph_digest: &str,
        target_record_id: &str,
        target_record_digest: &str,
    ) -> Result<(), CrossDkgLinkError> {
        self.verify_integrity()?;
        if self.target_dkg_id != target_dkg_id {
            return Err(CrossDkgLinkError::TargetDkgMismatch);
        }
        if self.target_graph_digest != target_graph_digest {
            return Err(CrossDkgLinkError::TargetGraphDigestMismatch);
        }
        if self.target_record_id != target_record_id {
            return Err(CrossDkgLinkError::TargetRecordIdMismatch);
        }
        if self.target_record_digest != target_record_digest {
            return Err(CrossDkgLinkError::TargetRecordDigestMismatch);
        }
        Ok(())
    }

    /// Verify that the source side names the exact foreign record.
    pub fn verify_source(
        &self,
        source_dkg_id: &str,
        source_graph_digest: &str,
        source_record_id: &str,
        source_record_digest: &str,
    ) -> Result<(), CrossDkgLinkError> {
        self.verify_integrity()?;
        if self.source_dkg_id != source_dkg_id {
            return Err(CrossDkgLinkError::SourceDkgMismatch);
        }
        if self.source_graph_digest != source_graph_digest {
            return Err(CrossDkgLinkError::SourceGraphDigestMismatch);
        }
        if self.source_record_id != source_record_id {
            return Err(CrossDkgLinkError::SourceRecordIdMismatch);
        }
        if self.source_record_digest != source_record_digest {
            return Err(CrossDkgLinkError::SourceRecordDigestMismatch);
        }
        Ok(())
    }

    pub fn link_digest(&self) -> String {
        self.compute_digest()
    }

    fn validate_fields(&self) -> Result<(), CrossDkgLinkError> {
        if self.link_version != LINK_VERSION {
            return Err(CrossDkgLinkError::VersionMismatch);
        }
        if self.digest_algorithm != DIGEST_ALGORITHM {
            return Err(CrossDkgLinkError::DigestAlgorithmMismatch);
        }
        for value in [
            &self.source_dkg_id,
            &self.source_graph_digest,
            &self.source_record_id,
            &self.source_record_digest,
            &self.target_dkg_id,
            &self.target_graph_digest,
            &self.target_record_id,
            &self.target_record_digest,
        ] {
            if value.trim().is_empty() {
                return Err(CrossDkgLinkError::MissingField);
            }
        }
        if self.source_dkg_id == self.target_dkg_id
            && self.source_record_id == self.target_record_id
            && self.source_record_digest == self.target_record_digest
        {
            return Err(CrossDkgLinkError::SelfLink);
        }
        Ok(())
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(DIGEST_DOMAIN);
        put(&mut h, &self.link_version);
        put(&mut h, &self.digest_algorithm);
        put(&mut h, &self.source_dkg_id);
        put(&mut h, &self.source_graph_digest);
        put(&mut h, &self.source_record_id);
        put(&mut h, &self.source_record_digest);
        put(&mut h, self.relation.as_str());
        put(&mut h, &self.target_dkg_id);
        put(&mut h, &self.target_graph_digest);
        put(&mut h, &self.target_record_id);
        put(&mut h, &self.target_record_digest);
        format!("sha256:{:x}", h.finalize())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossDkgLinkError {
    VersionMismatch,
    DigestAlgorithmMismatch,
    MissingField,
    SelfLink,
    LinkDigestMismatch,
    SourceDkgMismatch,
    SourceGraphDigestMismatch,
    SourceRecordIdMismatch,
    SourceRecordDigestMismatch,
    TargetDkgMismatch,
    TargetGraphDigestMismatch,
    TargetRecordIdMismatch,
    TargetRecordDigestMismatch,
}

fn put(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link() -> CrossDkgFederationLink {
        CrossDkgFederationLink::new(
            "mycelix-knowledge",
            "sha256:source-graph",
            "claim:42",
            "sha256:source-record",
            CrossDkgRelation::References,
            "symthaea-evidence-lineage",
            "sha256:target-graph",
            "observation:7",
            "sha256:target-record",
        )
        .unwrap()
    }

    #[test]
    fn content_addressed_link_verifies() {
        let link = link();
        assert!(link.verify_integrity().is_ok());
        assert!(link
            .verify_source(
                "mycelix-knowledge",
                "sha256:source-graph",
                "claim:42",
                "sha256:source-record",
            )
            .is_ok());
        assert!(link
            .verify_target(
                "symthaea-evidence-lineage",
                "sha256:target-graph",
                "observation:7",
                "sha256:target-record",
            )
            .is_ok());
    }

    #[test]
    fn relation_is_part_of_identity() {
        let a = link();
        let b = CrossDkgFederationLink::new(
            "mycelix-knowledge",
            "sha256:source-graph",
            "claim:42",
            "sha256:source-record",
            CrossDkgRelation::DerivedFrom,
            "symthaea-evidence-lineage",
            "sha256:target-graph",
            "observation:7",
            "sha256:target-record",
        )
        .unwrap();
        assert_ne!(a.link_digest, b.link_digest);
    }

    #[test]
    fn wrong_source_is_rejected() {
        let link = link();
        assert_eq!(
            link.verify_source(
                "sol-atlas",
                "sha256:source-graph",
                "claim:42",
                "sha256:source-record",
            ),
            Err(CrossDkgLinkError::SourceDkgMismatch)
        );
    }

    #[test]
    fn wrong_target_digest_is_rejected() {
        let link = link();
        assert_eq!(
            link.verify_target(
                "symthaea-evidence-lineage",
                "sha256:target-graph",
                "observation:7",
                "sha256:tampered",
            ),
            Err(CrossDkgLinkError::TargetRecordDigestMismatch)
        );
    }

    #[test]
    fn tampering_is_rejected() {
        let mut link = link();
        link.target_record_id = "observation:8".into();
        assert_eq!(
            link.verify_integrity(),
            Err(CrossDkgLinkError::LinkDigestMismatch)
        );
    }

    #[test]
    fn self_link_is_rejected() {
        assert_eq!(
            CrossDkgFederationLink::new(
                "symthaea",
                "sha256:graph",
                "record:1",
                "sha256:record",
                CrossDkgRelation::References,
                "symthaea",
                "sha256:graph",
                "record:1",
                "sha256:record",
            ),
            Err(CrossDkgLinkError::SelfLink)
        );
    }

    #[test]
    fn serde_roundtrip_preserves_link() {
        let link = link();
        let bytes = serde_json::to_vec(&link).unwrap();
        let decoded: CrossDkgFederationLink = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(link, decoded);
        assert!(decoded.verify_integrity().is_ok());
    }
}
