// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact multi-hop catalog lineage built from direct consistency proofs.
//!
//! The existing consistency proof deliberately proves one direct checkpoint
//! transition.  This module composes those direct proofs without weakening
//! them: every intermediate catalog and checkpoint remains explicit and every
//! link is re-audited independently.

use serde::{Deserialize, Serialize};

use crate::evidence_calibration::{
    CalibrationPublicationCatalog, CalibrationPublicationCatalogCheckpoint,
    CalibrationPublicationCatalogConsistencyProof,
    audit_calibration_publication_catalog_checkpoint,
    audit_calibration_publication_catalog_consistency_proof,
    build_calibration_publication_catalog_consistency_proof,
};
use crate::evidence_calibration::sha256::{Sha256, hex as sha256_hex};

pub const CALIBRATION_PUBLICATION_CATALOG_LINEAGE_LINK_VERSION: &str =
    "score-evidence-calibration-publication-catalog-lineage-link-v1";
pub const CALIBRATION_PUBLICATION_CATALOG_LINEAGE_CHAIN_VERSION: &str =
    "score-evidence-calibration-publication-catalog-lineage-chain-v1";
pub const CALIBRATION_PUBLICATION_CATALOG_LINEAGE_AUDIT_VERSION: &str =
    "score-evidence-calibration-publication-catalog-lineage-audit-v1";

const LINK_DOMAIN: &[u8] =
    b"symthaea.score-evidence.publication-catalog-lineage-link.v1\0";
const CHAIN_DOMAIN: &[u8] =
    b"symthaea.score-evidence.publication-catalog-lineage-chain.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationCatalogLineageLink {
    pub link_version: String,
    pub to_catalog: CalibrationPublicationCatalog,
    pub to_checkpoint: CalibrationPublicationCatalogCheckpoint,
    pub consistency_proof: CalibrationPublicationCatalogConsistencyProof,
    pub link_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationCatalogLineageChain {
    pub chain_version: String,
    pub catalog_id: String,
    pub authority_id: String,
    pub anchor_catalog: CalibrationPublicationCatalog,
    pub anchor_checkpoint: CalibrationPublicationCatalogCheckpoint,
    pub links: Vec<CalibrationPublicationCatalogLineageLink>,
    pub chain_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationPublicationCatalogLineageIssueCode {
    ChainVersionMismatch,
    EmptyIdentity,
    AnchorCheckpointInvalid,
    AnchorIdentityMismatch,
    LinkVersionMismatch,
    LinkIdentityMismatch,
    DirectConsistencyInvalid,
    LinkSha256Mismatch,
    CheckpointRepeated,
    ChainSha256Mismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationCatalogLineageIssue {
    pub code: CalibrationPublicationCatalogLineageIssueCode,
    pub link_index: Option<u64>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationCatalogLineageAuditReport {
    pub audit_version: String,
    pub issues: Vec<CalibrationPublicationCatalogLineageIssue>,
}

impl CalibrationPublicationCatalogLineageAuditReport {
    pub fn valid(&self) -> bool {
        self.issues.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "details", rename_all = "snake_case")]
pub enum CalibrationPublicationCatalogLineageError {
    InvalidAnchor,
    InvalidExtension { index: u64, issues: usize },
    InvalidChain { issues: usize },
}

impl std::fmt::Display for CalibrationPublicationCatalogLineageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAnchor => write!(formatter, "catalog lineage anchor is invalid"),
            Self::InvalidExtension { index, issues } => write!(
                formatter,
                "catalog lineage extension {index} failed with {issues} issues",
            ),
            Self::InvalidChain { issues } => {
                write!(formatter, "catalog lineage audit failed with {issues} issues")
            }
        }
    }
}

impl std::error::Error for CalibrationPublicationCatalogLineageError {}

pub fn build_calibration_publication_catalog_lineage_chain(
    anchor_catalog: CalibrationPublicationCatalog,
    anchor_checkpoint: CalibrationPublicationCatalogCheckpoint,
    extensions: Vec<(CalibrationPublicationCatalog, CalibrationPublicationCatalogCheckpoint)>,
) -> Result<CalibrationPublicationCatalogLineageChain, CalibrationPublicationCatalogLineageError> {
    if !audit_calibration_publication_catalog_checkpoint(&anchor_catalog, &anchor_checkpoint).valid() {
        return Err(CalibrationPublicationCatalogLineageError::InvalidAnchor);
    }
    let mut previous_catalog = &anchor_catalog;
    let mut previous_checkpoint = &anchor_checkpoint;
    let mut links = Vec::with_capacity(extensions.len());
    for (index, (to_catalog, to_checkpoint)) in extensions.into_iter().enumerate() {
        let proof = build_calibration_publication_catalog_consistency_proof(
            previous_catalog,
            previous_checkpoint,
            &to_catalog,
            &to_checkpoint,
        )
        .map_err(|_| CalibrationPublicationCatalogLineageError::InvalidExtension {
            index: index as u64,
            issues: 1,
        })?;
        let mut link = CalibrationPublicationCatalogLineageLink {
            link_version: CALIBRATION_PUBLICATION_CATALOG_LINEAGE_LINK_VERSION.into(),
            to_catalog,
            to_checkpoint,
            consistency_proof: proof,
            link_sha256: String::new(),
        };
        link.link_sha256 = calibration_publication_catalog_lineage_link_sha256(&link);
        links.push(link);
        let latest_index = links.len() - 1;
        previous_catalog = &links[latest_index].to_catalog;
        previous_checkpoint = &links[latest_index].to_checkpoint;
    }
    let mut chain = CalibrationPublicationCatalogLineageChain {
        chain_version: CALIBRATION_PUBLICATION_CATALOG_LINEAGE_CHAIN_VERSION.into(),
        catalog_id: anchor_catalog.catalog_id.clone(),
        authority_id: anchor_catalog.authority_id.clone(),
        anchor_catalog,
        anchor_checkpoint,
        links,
        chain_sha256: String::new(),
    };
    chain.chain_sha256 = calibration_publication_catalog_lineage_chain_sha256(&chain);
    let audit = audit_calibration_publication_catalog_lineage_chain(&chain);
    if !audit.valid() {
        return Err(CalibrationPublicationCatalogLineageError::InvalidChain {
            issues: audit.issues.len(),
        });
    }
    Ok(chain)
}

pub fn audit_calibration_publication_catalog_lineage_chain(
    chain: &CalibrationPublicationCatalogLineageChain,
) -> CalibrationPublicationCatalogLineageAuditReport {
    let mut report = CalibrationPublicationCatalogLineageAuditReport {
        audit_version: CALIBRATION_PUBLICATION_CATALOG_LINEAGE_AUDIT_VERSION.into(),
        issues: Vec::new(),
    };
    if chain.chain_version != CALIBRATION_PUBLICATION_CATALOG_LINEAGE_CHAIN_VERSION {
        lineage_issue(&mut report, CalibrationPublicationCatalogLineageIssueCode::ChainVersionMismatch, None, "catalog-lineage chain version mismatch");
    }
    if chain.catalog_id.trim().is_empty() || chain.authority_id.trim().is_empty() {
        lineage_issue(&mut report, CalibrationPublicationCatalogLineageIssueCode::EmptyIdentity, None, "catalog and authority identities must not be empty");
    }
    if !audit_calibration_publication_catalog_checkpoint(
        &chain.anchor_catalog,
        &chain.anchor_checkpoint,
    )
    .valid()
    {
        lineage_issue(&mut report, CalibrationPublicationCatalogLineageIssueCode::AnchorCheckpointInvalid, None, "lineage anchor checkpoint is invalid for its catalog");
    }
    if chain.anchor_catalog.catalog_id != chain.catalog_id
        || chain.anchor_catalog.authority_id != chain.authority_id
        || chain.anchor_checkpoint.catalog_id != chain.catalog_id
        || chain.anchor_checkpoint.authority_id != chain.authority_id
    {
        lineage_issue(&mut report, CalibrationPublicationCatalogLineageIssueCode::AnchorIdentityMismatch, None, "lineage anchor identity differs from the chain identity");
    }
    let mut checkpoint_ids = std::collections::BTreeSet::new();
    checkpoint_ids.insert(chain.anchor_checkpoint.checkpoint_sha256.clone());
    let mut previous_catalog = &chain.anchor_catalog;
    let mut previous_checkpoint = &chain.anchor_checkpoint;
    for (index, link) in chain.links.iter().enumerate() {
        let index = index as u64;
        if link.link_version != CALIBRATION_PUBLICATION_CATALOG_LINEAGE_LINK_VERSION {
            lineage_issue(&mut report, CalibrationPublicationCatalogLineageIssueCode::LinkVersionMismatch, Some(index), "catalog-lineage link version mismatch");
        }
        if link.to_catalog.catalog_id != chain.catalog_id
            || link.to_catalog.authority_id != chain.authority_id
            || link.to_checkpoint.catalog_id != chain.catalog_id
            || link.to_checkpoint.authority_id != chain.authority_id
        {
            lineage_issue(&mut report, CalibrationPublicationCatalogLineageIssueCode::LinkIdentityMismatch, Some(index), "catalog-lineage link changes catalog identity");
        }
        if !audit_calibration_publication_catalog_consistency_proof(
            previous_catalog,
            previous_checkpoint,
            &link.to_catalog,
            &link.to_checkpoint,
            &link.consistency_proof,
        )
        .valid()
        {
            lineage_issue(&mut report, CalibrationPublicationCatalogLineageIssueCode::DirectConsistencyInvalid, Some(index), "catalog-lineage link does not prove one exact direct extension");
        }
        if link.link_sha256 != calibration_publication_catalog_lineage_link_sha256(link) {
            lineage_issue(&mut report, CalibrationPublicationCatalogLineageIssueCode::LinkSha256Mismatch, Some(index), "catalog-lineage link SHA-256 mismatch");
        }
        if !checkpoint_ids.insert(link.to_checkpoint.checkpoint_sha256.clone()) {
            lineage_issue(&mut report, CalibrationPublicationCatalogLineageIssueCode::CheckpointRepeated, Some(index), "catalog-lineage chain repeats a checkpoint identity");
        }
        previous_catalog = &link.to_catalog;
        previous_checkpoint = &link.to_checkpoint;
    }
    if chain.chain_sha256 != calibration_publication_catalog_lineage_chain_sha256(chain) {
        lineage_issue(&mut report, CalibrationPublicationCatalogLineageIssueCode::ChainSha256Mismatch, None, "catalog-lineage chain SHA-256 mismatch");
    }
    report
}

pub fn calibration_publication_catalog_lineage_checkpoint_sha256s(
    chain: &CalibrationPublicationCatalogLineageChain,
) -> Vec<String> {
    let mut checkpoints = Vec::with_capacity(chain.links.len() + 1);
    checkpoints.push(chain.anchor_checkpoint.checkpoint_sha256.clone());
    checkpoints.extend(
        chain
            .links
            .iter()
            .map(|link| link.to_checkpoint.checkpoint_sha256.clone()),
    );
    checkpoints
}

pub fn calibration_publication_catalog_lineage_terminal<'a>(
    chain: &'a CalibrationPublicationCatalogLineageChain,
) -> (&'a CalibrationPublicationCatalog, &'a CalibrationPublicationCatalogCheckpoint) {
    match chain.links.last() {
        Some(link) => (&link.to_catalog, &link.to_checkpoint),
        None => (&chain.anchor_catalog, &chain.anchor_checkpoint),
    }
}

pub fn calibration_publication_catalog_lineage_link_sha256(
    link: &CalibrationPublicationCatalogLineageLink,
) -> String {
    let mut hash = Sha256::new();
    hash.update(LINK_DOMAIN);
    hash_field(&mut hash, &link.link_version);
    hash_field(&mut hash, &link.to_catalog.catalog_sha256);
    hash_field(&mut hash, &link.to_checkpoint.checkpoint_sha256);
    hash_field(&mut hash, &link.consistency_proof.proof_sha256);
    sha256_hex(&hash.finalize())
}

pub fn calibration_publication_catalog_lineage_chain_sha256(
    chain: &CalibrationPublicationCatalogLineageChain,
) -> String {
    let mut hash = Sha256::new();
    hash.update(CHAIN_DOMAIN);
    hash_field(&mut hash, &chain.chain_version);
    hash_field(&mut hash, &chain.catalog_id);
    hash_field(&mut hash, &chain.authority_id);
    hash_field(&mut hash, &chain.anchor_catalog.catalog_sha256);
    hash_field(&mut hash, &chain.anchor_checkpoint.checkpoint_sha256);
    hash.update(&(chain.links.len() as u64).to_le_bytes());
    for link in &chain.links {
        hash_field(&mut hash, &link.link_sha256);
    }
    sha256_hex(&hash.finalize())
}

fn lineage_issue(
    report: &mut CalibrationPublicationCatalogLineageAuditReport,
    code: CalibrationPublicationCatalogLineageIssueCode,
    link_index: Option<u64>,
    detail: impl Into<String>,
) {
    report.issues.push(CalibrationPublicationCatalogLineageIssue {
        code,
        link_index,
        detail: detail.into(),
    });
}

fn hash_field(hash: &mut Sha256, value: &str) {
    hash.update(&(value.len() as u64).to_le_bytes());
    hash.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_calibration::{
        build_calibration_publication_catalog,
        build_calibration_publication_catalog_checkpoint,
    };

    #[test]
    fn empty_chain_is_exact_anchor_lineage() {
        let catalog = build_calibration_publication_catalog("catalog", "authority");
        let checkpoint = build_calibration_publication_catalog_checkpoint(&catalog, None, 1)
            .expect("checkpoint");
        let chain = build_calibration_publication_catalog_lineage_chain(
            catalog.clone(),
            checkpoint.clone(),
            Vec::new(),
        )
        .expect("chain");
        assert!(audit_calibration_publication_catalog_lineage_chain(&chain).valid());
        assert_eq!(calibration_publication_catalog_lineage_terminal(&chain), (&catalog, &checkpoint));
    }

    #[test]
    fn direct_checkpoint_hops_compose() {
        let catalog = build_calibration_publication_catalog("catalog", "authority");
        let first = build_calibration_publication_catalog_checkpoint(&catalog, None, 1)
            .expect("first");
        let second = build_calibration_publication_catalog_checkpoint(&catalog, Some(&first), 2)
            .expect("second");
        let third = build_calibration_publication_catalog_checkpoint(&catalog, Some(&second), 3)
            .expect("third");
        let chain = build_calibration_publication_catalog_lineage_chain(
            catalog.clone(),
            first,
            vec![(catalog.clone(), second), (catalog.clone(), third.clone())],
        )
        .expect("chain");
        assert_eq!(
            calibration_publication_catalog_lineage_terminal(&chain).1.checkpoint_sha256,
            third.checkpoint_sha256
        );
        assert_eq!(calibration_publication_catalog_lineage_checkpoint_sha256s(&chain).len(), 3);
    }

    #[test]
    fn forged_intermediate_link_is_detected() {
        let catalog = build_calibration_publication_catalog("catalog", "authority");
        let first = build_calibration_publication_catalog_checkpoint(&catalog, None, 1)
            .expect("first");
        let second = build_calibration_publication_catalog_checkpoint(&catalog, Some(&first), 2)
            .expect("second");
        let mut chain = build_calibration_publication_catalog_lineage_chain(
            catalog.clone(),
            first,
            vec![(catalog, second)],
        )
        .expect("chain");
        chain.links[0].consistency_proof.to_event_count = 99;
        chain.links[0].link_sha256 = calibration_publication_catalog_lineage_link_sha256(&chain.links[0]);
        chain.chain_sha256 = calibration_publication_catalog_lineage_chain_sha256(&chain);
        assert!(!audit_calibration_publication_catalog_lineage_chain(&chain).valid());
    }
}
