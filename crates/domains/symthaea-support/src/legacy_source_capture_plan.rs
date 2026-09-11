// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deduplicated capture planning for legacy-computing qualification sources.
//!
//! Qualification needs immutable content artifacts, but this crate must not copy
//! vendor manuals into the repository. V2 therefore plans one external/private
//! capture per logical document — not one capture per metadata snapshot — and
//! leaves byte retrieval, licensing checks, storage and receipt generation to an
//! authorized evidence service/operator.

use crate::legacy_computing::{LegacyComputingErrorV1, LegacyComputingPackV1};
use crate::legacy_source_lineage::{
    assess_legacy_source_lineage_v1, LegacySourceLineageErrorV1,
};
use crate::standards_registry::{
    SourceDocumentIdV1, SourceSnapshotIdV1, TechnicalClaimIdV1, TechnicalPublisherV1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const LEGACY_SOURCE_CAPTURE_PLAN_SCHEMA_V2: &str =
    "symthaea-it-legacy-source-capture-plan-v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegacyCaptureRetentionV2 {
    /// Default for vendor material whose redistribution rights are not established.
    PrivateEvidenceStore,
    /// Organization-controlled archive; still not solver-visible by default.
    OrganizationArchive,
    /// Only when redistribution/public archival rights are explicitly established.
    PublicImmutableArchive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegacyCaptureRedistributionPolicyV2 {
    /// Fail-safe default: retain privately unless a separate rights review says otherwise.
    PrivateUnlessExplicitlyPermitted,
    ExplicitlyPublic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySourceCaptureRequestV2 {
    pub document_id: SourceDocumentIdV1,
    pub publisher: TechnicalPublisherV1,
    pub title: String,
    pub canonical_ref: String,
    pub canonical_locator: Option<String>,
    pub observed_versions: BTreeSet<String>,
    /// Historical/metadata captures collapsed into this one logical capture request.
    pub metadata_snapshot_ids: BTreeSet<SourceSnapshotIdV1>,
    pub claim_ids: BTreeSet<TechnicalClaimIdV1>,
    pub content_digest_algorithm: String,
    pub retention: LegacyCaptureRetentionV2,
    pub redistribution_policy: LegacyCaptureRedistributionPolicyV2,
    pub capture_ready: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySourceCapturePlanV2 {
    pub schema_version: String,
    pub logical_documents: usize,
    pub metadata_snapshots: usize,
    pub capture_requests: usize,
    pub duplicate_snapshot_captures_collapsed: usize,
    pub requests_missing_locator: usize,
    pub requests: Vec<LegacySourceCaptureRequestV2>,
}

pub fn plan_legacy_qualification_source_captures_v2(
    pack: &LegacyComputingPackV1,
) -> Result<LegacySourceCapturePlanV2, LegacySourceCapturePlanErrorV2> {
    pack.validate()?;
    let lineage = assess_legacy_source_lineage_v1(pack)?;
    let mut requests = Vec::with_capacity(lineage.groups.len());
    let mut requests_missing_locator = 0usize;

    for group in lineage.groups {
        let document = pack
            .sources
            .document(&group.document_id)
            .ok_or_else(|| LegacySourceCapturePlanErrorV2::UnknownDocument(group.document_id.clone()))?;
        let mut observed_versions = BTreeSet::new();
        for snapshot_id in &group.snapshot_ids {
            let snapshot = pack
                .sources
                .snapshot(snapshot_id)
                .ok_or_else(|| LegacySourceCapturePlanErrorV2::UnknownSnapshot(snapshot_id.clone()))?;
            if let Some(version) = &snapshot.version {
                observed_versions.insert(version.clone());
            }
        }

        let canonical_locator = document
            .canonical_locator
            .as_ref()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let capture_ready = canonical_locator.is_some() && !group.snapshot_ids.is_empty();
        if canonical_locator.is_none() {
            requests_missing_locator += 1;
        }

        let (retention, redistribution_policy) = match &document.publisher {
            // Vendor content remains private by default unless a separate rights
            // review establishes that redistribution is permitted.
            TechnicalPublisherV1::Vendor(_) => (
                LegacyCaptureRetentionV2::PrivateEvidenceStore,
                LegacyCaptureRedistributionPolicyV2::PrivateUnlessExplicitlyPermitted,
            ),
            _ => (
                LegacyCaptureRetentionV2::OrganizationArchive,
                LegacyCaptureRedistributionPolicyV2::PrivateUnlessExplicitlyPermitted,
            ),
        };

        requests.push(LegacySourceCaptureRequestV2 {
            document_id: group.document_id,
            publisher: document.publisher.clone(),
            title: document.title.clone(),
            canonical_ref: document.canonical_ref.clone(),
            canonical_locator,
            observed_versions,
            metadata_snapshot_ids: group.snapshot_ids,
            claim_ids: group.claim_ids,
            content_digest_algorithm: "sha256".into(),
            retention,
            redistribution_policy,
            capture_ready,
        });
    }

    requests.sort_by(|a, b| a.document_id.cmp(&b.document_id));
    let metadata_snapshots = lineage.snapshot_count;
    let logical_documents = lineage.logical_document_count;
    let capture_requests = requests.len();
    let duplicate_snapshot_captures_collapsed = metadata_snapshots.saturating_sub(capture_requests);

    Ok(LegacySourceCapturePlanV2 {
        schema_version: LEGACY_SOURCE_CAPTURE_PLAN_SCHEMA_V2.into(),
        logical_documents,
        metadata_snapshots,
        capture_requests,
        duplicate_snapshot_captures_collapsed,
        requests_missing_locator,
        requests,
    })
}

#[derive(Debug)]
pub enum LegacySourceCapturePlanErrorV2 {
    Pack(LegacyComputingErrorV1),
    Lineage(LegacySourceLineageErrorV1),
    UnknownDocument(SourceDocumentIdV1),
    UnknownSnapshot(SourceSnapshotIdV1),
}

impl fmt::Display for LegacySourceCapturePlanErrorV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pack(err) => write!(f, "legacy capture-plan pack error: {err}"),
            Self::Lineage(err) => write!(f, "legacy capture-plan lineage error: {err}"),
            Self::UnknownDocument(id) => write!(f, "legacy capture plan references unknown document {}", id.0),
            Self::UnknownSnapshot(id) => write!(f, "legacy capture plan references unknown snapshot {}", id.0),
        }
    }
}

impl Error for LegacySourceCapturePlanErrorV2 {}

impl From<LegacyComputingErrorV1> for LegacySourceCapturePlanErrorV2 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::Pack(value)
    }
}

impl From<LegacySourceLineageErrorV1> for LegacySourceCapturePlanErrorV2 {
    fn from(value: LegacySourceLineageErrorV1) -> Self {
        Self::Lineage(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_legacy_five_platform_portfolio_v1;

    fn pack() -> LegacyComputingPackV1 {
        build_legacy_five_platform_portfolio_v1(1_800_000_000_000)
            .unwrap()
            .0
    }

    #[test]
    fn plan_deduplicates_snapshot_capture_by_logical_document() {
        let pack = pack();
        let plan = plan_legacy_qualification_source_captures_v2(&pack).unwrap();
        assert_eq!(plan.schema_version, LEGACY_SOURCE_CAPTURE_PLAN_SCHEMA_V2);
        assert_eq!(plan.capture_requests, plan.logical_documents);
        assert!(plan.metadata_snapshots >= plan.capture_requests);
        assert!(plan.duplicate_snapshot_captures_collapsed >= 1);
    }

    #[test]
    fn duplicated_hpux_install_snapshots_become_one_private_capture_request() {
        let pack = pack();
        let plan = plan_legacy_qualification_source_captures_v2(&pack).unwrap();
        let request = plan
            .requests
            .iter()
            .find(|request| request.document_id.0 == "hpe:hpux-install-update")
            .unwrap();
        assert_eq!(request.metadata_snapshot_ids.len(), 2);
        assert_eq!(request.retention, LegacyCaptureRetentionV2::PrivateEvidenceStore);
        assert_eq!(
            request.redistribution_policy,
            LegacyCaptureRedistributionPolicyV2::PrivateUnlessExplicitlyPermitted
        );
        assert!(request.capture_ready);
    }

    #[test]
    fn vendor_sources_default_to_private_retention() {
        let pack = pack();
        let plan = plan_legacy_qualification_source_captures_v2(&pack).unwrap();
        for request in plan.requests.iter().filter(|request| {
            matches!(request.publisher, TechnicalPublisherV1::Vendor(_))
        }) {
            assert_eq!(request.retention, LegacyCaptureRetentionV2::PrivateEvidenceStore);
            assert_eq!(
                request.redistribution_policy,
                LegacyCaptureRedistributionPolicyV2::PrivateUnlessExplicitlyPermitted
            );
        }
    }

    #[test]
    fn capture_plan_contains_no_vendor_document_bytes() {
        let pack = pack();
        let plan = plan_legacy_qualification_source_captures_v2(&pack).unwrap();
        let encoded = serde_json::to_string(&plan).unwrap();
        assert!(!encoded.contains("document_bytes"));
        assert!(!encoded.contains("base64"));
        assert!(plan
            .requests
            .iter()
            .all(|request| request.content_digest_algorithm == "sha256"));
    }
}
