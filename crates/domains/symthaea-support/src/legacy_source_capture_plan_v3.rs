// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Snapshot-scoped capture planning for qualification-grade legacy evidence.
//!
//! V2 plans one acquisition per logical document, which is useful for discovery
//! and fetch deduplication but too coarse for qualification when one document has
//! multiple immutable observed revisions. V3 therefore treats each immutable
//! source snapshot actually used by a claim or procedure as an independent
//! qualification revision requirement.
//!
//! Shared document/locator observations are exposed only as *potential* fetch-
//! reuse groups. They are never merged automatically. One fetch may later satisfy
//! more than one original snapshot only after explicit content/source verification.
//!
//! ```text
//! logical document lineage -> corroboration ceiling
//! immutable source snapshot -> qualification revision identity
//! shared locator -> possible acquisition optimization, not equivalence proof
//! ```

use crate::legacy_computing::{LegacyComputingErrorV1, LegacyComputingPackV1};
use crate::legacy_source_capture_plan::{
    LegacyCaptureRedistributionPolicyV2, LegacyCaptureRetentionV2,
};
use crate::standards_registry::{
    SourceCaptureV1, SourceDocumentIdV1, SourceSnapshotIdV1, TechnicalClaimIdV1,
    TechnicalPublisherV1, TechnicalSourceDocumentV1, TechnicalSourceSnapshotV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_SOURCE_CAPTURE_PLAN_SCHEMA_V3: &str =
    "symthaea-it-legacy-source-capture-plan-v3";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySourceRevisionCaptureRequestV3 {
    pub document_id: SourceDocumentIdV1,
    /// The immutable historical observation whose provenance must eventually be
    /// backed by exact retained bytes.
    pub original_snapshot_id: SourceSnapshotIdV1,
    /// BLAKE3 over the exact typed document + snapshot metadata. This binds the
    /// request to more than the string IDs.
    pub source_revision_blake3: String,
    pub publisher: TechnicalPublisherV1,
    pub title: String,
    pub canonical_ref: String,
    pub canonical_locator: Option<String>,
    pub observed_version: Option<String>,
    pub claim_ids: BTreeSet<TechnicalClaimIdV1>,
    pub procedure_ids: BTreeSet<String>,
    pub existing_content_bound: bool,
    pub content_digest_algorithm: String,
    pub retention: LegacyCaptureRetentionV2,
    pub redistribution_policy: LegacyCaptureRedistributionPolicyV2,
    pub capture_needed: bool,
    pub capture_ready: bool,
}

/// Advisory acquisition optimization only. Membership does not assert that the
/// represented source revisions contain identical bytes or semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyPotentialFetchReuseGroupV3 {
    pub document_id: SourceDocumentIdV1,
    pub canonical_locator: String,
    pub snapshot_ids: BTreeSet<SourceSnapshotIdV1>,
    pub reuse_requires_post_capture_verification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySourceCapturePlanV3 {
    pub schema_version: String,
    pub required_logical_documents: usize,
    pub required_source_revisions: usize,
    pub already_content_bound_revisions: usize,
    pub capture_needed_revisions: usize,
    pub requests_missing_locator: usize,
    pub potential_fetch_reuse_groups: Vec<LegacyPotentialFetchReuseGroupV3>,
    pub requests: Vec<LegacySourceRevisionCaptureRequestV3>,
}

pub fn plan_legacy_qualification_source_captures_v3(
    pack: &LegacyComputingPackV1,
) -> Result<LegacySourceCapturePlanV3, LegacySourceCapturePlanErrorV3> {
    pack.validate()?;

    let mut required = BTreeMap::<SourceSnapshotIdV1, RequiredUse>::new();
    for claim in pack.sources.claims() {
        required
            .entry(claim.source_snapshot.clone())
            .or_default()
            .claim_ids
            .insert(claim.id.clone());
    }
    for procedure in &pack.procedures {
        for snapshot_id in &procedure.source_snapshots {
            required
                .entry(snapshot_id.clone())
                .or_default()
                .procedure_ids
                .insert(procedure.id.clone());
        }
    }

    let mut requests = Vec::with_capacity(required.len());
    let mut required_documents = BTreeSet::new();
    let mut requests_missing_locator = 0usize;
    let mut already_content_bound_revisions = 0usize;
    let mut reuse_candidates = BTreeMap::<
        (SourceDocumentIdV1, String),
        BTreeSet<SourceSnapshotIdV1>,
    >::new();

    for (snapshot_id, use_info) in required {
        let snapshot = pack
            .sources
            .snapshot(&snapshot_id)
            .ok_or_else(|| LegacySourceCapturePlanErrorV3::UnknownSnapshot(snapshot_id.clone()))?;
        let document = pack
            .sources
            .document(&snapshot.document_id)
            .ok_or_else(|| {
                LegacySourceCapturePlanErrorV3::UnknownDocument(snapshot.document_id.clone())
            })?;
        required_documents.insert(document.id.clone());

        let canonical_locator = document
            .canonical_locator
            .as_ref()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        if let Some(locator) = &canonical_locator {
            reuse_candidates
                .entry((document.id.clone(), locator.clone()))
                .or_default()
                .insert(snapshot.id.clone());
        }

        let existing_content_bound =
            matches!(&snapshot.capture, SourceCaptureV1::ContentDigest { .. });
        if existing_content_bound {
            already_content_bound_revisions += 1;
        } else if canonical_locator.is_none() {
            requests_missing_locator += 1;
        }
        let (retention, redistribution_policy) = capture_policy(document);
        requests.push(LegacySourceRevisionCaptureRequestV3 {
            document_id: document.id.clone(),
            original_snapshot_id: snapshot.id.clone(),
            source_revision_blake3: legacy_source_revision_commitment_v3(document, snapshot)?,
            publisher: document.publisher.clone(),
            title: document.title.clone(),
            canonical_ref: document.canonical_ref.clone(),
            canonical_locator,
            observed_version: snapshot.version.clone(),
            claim_ids: use_info.claim_ids,
            procedure_ids: use_info.procedure_ids,
            existing_content_bound,
            content_digest_algorithm: "sha256".into(),
            retention,
            redistribution_policy,
            capture_needed: !existing_content_bound,
            capture_ready: existing_content_bound || document.canonical_locator.is_some(),
        });
    }

    requests.sort_by(|a, b| a.original_snapshot_id.cmp(&b.original_snapshot_id));
    let potential_fetch_reuse_groups = reuse_candidates
        .into_iter()
        .filter(|(_, snapshots)| snapshots.len() > 1)
        .map(
            |((document_id, canonical_locator), snapshot_ids)| LegacyPotentialFetchReuseGroupV3 {
                document_id,
                canonical_locator,
                snapshot_ids,
                reuse_requires_post_capture_verification: true,
            },
        )
        .collect::<Vec<_>>();

    let required_source_revisions = requests.len();
    let capture_needed_revisions = required_source_revisions - already_content_bound_revisions;
    Ok(LegacySourceCapturePlanV3 {
        schema_version: LEGACY_SOURCE_CAPTURE_PLAN_SCHEMA_V3.into(),
        required_logical_documents: required_documents.len(),
        required_source_revisions,
        already_content_bound_revisions,
        capture_needed_revisions,
        requests_missing_locator,
        potential_fetch_reuse_groups,
        requests,
    })
}

/// Exact typed metadata commitment for the qualification revision represented by
/// one immutable source snapshot. Fetch time and capture class are intentionally
/// included: this is a commitment to the historical observation itself, not a
/// guessed publisher revision label.
pub fn legacy_source_revision_commitment_v3(
    document: &TechnicalSourceDocumentV1,
    snapshot: &TechnicalSourceSnapshotV1,
) -> Result<String, LegacySourceCapturePlanErrorV3> {
    if snapshot.document_id != document.id {
        return Err(LegacySourceCapturePlanErrorV3::DocumentSnapshotMismatch {
            document: document.id.clone(),
            snapshot: snapshot.id.clone(),
        });
    }
    #[derive(Serialize)]
    struct Commitment<'a> {
        document: &'a TechnicalSourceDocumentV1,
        snapshot: &'a TechnicalSourceSnapshotV1,
    }
    let encoded = serde_json::to_vec(&Commitment { document, snapshot })
        .map_err(|err| LegacySourceCapturePlanErrorV3::Serialization(err.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    frame(
        &mut hasher,
        b"schema",
        LEGACY_SOURCE_CAPTURE_PLAN_SCHEMA_V3.as_bytes(),
    );
    frame(&mut hasher, b"source_revision", &encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

fn capture_policy(
    document: &TechnicalSourceDocumentV1,
) -> (LegacyCaptureRetentionV2, LegacyCaptureRedistributionPolicyV2) {
    match &document.publisher {
        TechnicalPublisherV1::Vendor(_) => (
            LegacyCaptureRetentionV2::PrivateEvidenceStore,
            LegacyCaptureRedistributionPolicyV2::PrivateUnlessExplicitlyPermitted,
        ),
        _ => (
            LegacyCaptureRetentionV2::OrganizationArchive,
            LegacyCaptureRedistributionPolicyV2::PrivateUnlessExplicitlyPermitted,
        ),
    }
}

fn frame(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

#[derive(Default)]
struct RequiredUse {
    claim_ids: BTreeSet<TechnicalClaimIdV1>,
    procedure_ids: BTreeSet<String>,
}

#[derive(Debug)]
pub enum LegacySourceCapturePlanErrorV3 {
    Pack(LegacyComputingErrorV1),
    Serialization(String),
    UnknownDocument(SourceDocumentIdV1),
    UnknownSnapshot(SourceSnapshotIdV1),
    DocumentSnapshotMismatch {
        document: SourceDocumentIdV1,
        snapshot: SourceSnapshotIdV1,
    },
}

impl fmt::Display for LegacySourceCapturePlanErrorV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pack(err) => write!(f, "legacy source-capture V3 pack error: {err}"),
            Self::Serialization(message) => {
                write!(f, "legacy source-revision serialization failed: {message}")
            }
            Self::UnknownDocument(id) => {
                write!(f, "legacy capture V3 references unknown document {}", id.0)
            }
            Self::UnknownSnapshot(id) => {
                write!(f, "legacy capture V3 references unknown snapshot {}", id.0)
            }
            Self::DocumentSnapshotMismatch { document, snapshot } => write!(
                f,
                "snapshot {} does not belong to document {}",
                snapshot.0, document.0
            ),
        }
    }
}

impl Error for LegacySourceCapturePlanErrorV3 {}

impl From<LegacyComputingErrorV1> for LegacySourceCapturePlanErrorV3 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::Pack(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_legacy_five_platform_portfolio_v1;
    use crate::standards_registry::{
        TechnicalClaimIdV1, TechnicalKnowledgeClaimV1, TechnicalSourceSnapshotV1,
    };

    fn portfolio() -> LegacyComputingPackV1 {
        build_legacy_five_platform_portfolio_v1(1_800_000_000_000)
            .unwrap()
            .0
    }

    #[test]
    fn qualification_plan_is_snapshot_scoped_not_document_scoped() {
        let pack = portfolio();
        let plan = plan_legacy_qualification_source_captures_v3(&pack).unwrap();
        assert_eq!(plan.schema_version, LEGACY_SOURCE_CAPTURE_PLAN_SCHEMA_V3);
        assert!(plan.required_source_revisions >= plan.required_logical_documents);
        assert_eq!(
            plan.capture_needed_revisions + plan.already_content_bound_revisions,
            plan.required_source_revisions
        );
    }

    #[test]
    fn hpux_duplicate_document_remains_two_requirements_but_one_reuse_candidate() {
        let pack = portfolio();
        let plan = plan_legacy_qualification_source_captures_v3(&pack).unwrap();
        let hpux_requests = plan
            .requests
            .iter()
            .filter(|request| request.document_id.0 == "hpe:hpux-install-update")
            .collect::<Vec<_>>();
        assert_eq!(hpux_requests.len(), 2);
        assert_ne!(
            hpux_requests[0].source_revision_blake3,
            hpux_requests[1].source_revision_blake3
        );
        let reuse = plan
            .potential_fetch_reuse_groups
            .iter()
            .find(|group| group.document_id.0 == "hpe:hpux-install-update")
            .unwrap();
        assert_eq!(reuse.snapshot_ids.len(), 2);
        assert!(reuse.reuse_requires_post_capture_verification);
    }

    #[test]
    fn distinct_future_version_of_same_document_cannot_collapse() {
        let mut pack = portfolio();
        let original_claim = pack
            .sources
            .claim(&TechnicalClaimIdV1("legacy:hpux:install-verification".into()))
            .unwrap()
            .clone();
        let original_snapshot = pack
            .sources
            .snapshot(&original_claim.source_snapshot)
            .unwrap()
            .clone();
        let future_snapshot_id = SourceSnapshotIdV1("hpe:hpux-install-update@future-v4".into());
        pack.sources
            .register_snapshot(TechnicalSourceSnapshotV1 {
                id: future_snapshot_id.clone(),
                document_id: original_snapshot.document_id.clone(),
                version: Some("future-v4".into()),
                lifecycle: original_snapshot.lifecycle,
                authority: original_snapshot.authority,
                stability: original_snapshot.stability,
                published_at_unix_ms: original_snapshot.published_at_unix_ms,
                source_updated_at_unix_ms: Some(1_900_000_000_000),
                fetched_at_unix_ms: 1_900_000_000_001,
                capture: SourceCaptureV1::MetadataOnly,
                relations: original_snapshot.relations.clone(),
            })
            .unwrap();
        pack.sources
            .register_claim(TechnicalKnowledgeClaimV1 {
                id: TechnicalClaimIdV1("legacy:hpux:future-v4-test-claim".into()),
                statement: original_claim.statement.clone(),
                source_snapshot: future_snapshot_id.clone(),
                locator: original_claim.locator.clone(),
                modality: original_claim.modality,
                applicability: original_claim.applicability.clone(),
                extraction_quality: original_claim.extraction_quality,
                category: original_claim.category.clone(),
            })
            .unwrap();
        pack.validate().unwrap();

        let plan = plan_legacy_qualification_source_captures_v3(&pack).unwrap();
        let hpux_requests = plan
            .requests
            .iter()
            .filter(|request| request.document_id.0 == "hpe:hpux-install-update")
            .collect::<Vec<_>>();
        assert_eq!(hpux_requests.len(), 3);
        assert!(hpux_requests
            .iter()
            .any(|request| request.original_snapshot_id == future_snapshot_id));
    }

    #[test]
    fn unused_source_snapshot_does_not_create_qualification_capture_work() {
        let mut pack = portfolio();
        let basis = pack.sources.snapshots().next().unwrap().clone();
        let unused = SourceSnapshotIdV1("legacy:unused:test-snapshot".into());
        pack.sources
            .register_snapshot(TechnicalSourceSnapshotV1 {
                id: unused.clone(),
                document_id: basis.document_id.clone(),
                version: Some("unused-test".into()),
                lifecycle: basis.lifecycle,
                authority: basis.authority,
                stability: basis.stability,
                published_at_unix_ms: None,
                source_updated_at_unix_ms: None,
                fetched_at_unix_ms: 1_800_000_100_000,
                capture: SourceCaptureV1::MetadataOnly,
                relations: basis.relations.clone(),
            })
            .unwrap();
        pack.validate().unwrap();
        let plan = plan_legacy_qualification_source_captures_v3(&pack).unwrap();
        assert!(!plan
            .requests
            .iter()
            .any(|request| request.original_snapshot_id == unused));
    }
}
