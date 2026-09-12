// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Qualification-time continuity gate for legacy source evidence.
//!
//! A later fetch cannot retroactively recover exact bytes for an earlier
//! metadata-only observation merely because product/version metadata still
//! matches. Living vendor documentation may change beneath a stable product
//! version or locator.
//!
//! ```text
//! metadata-only historical observation + later content fetch
//!     -> rebaseline required
//!
//! content-bound historical observation + exact same content identity
//!     -> exact content continuity
//!
//! content-bound historical observation + different content identity
//!     -> content mismatch
//! ```
//!
//! Rebaselining preserves the old observation as history, captures current bytes
//! as a new content-bound snapshot, and requires dependent claims/procedures to
//! be reviewed against that new baseline. This module creates no evidence and
//! grants no authority.

use crate::legacy_computing::LegacyComputingPackV1;
use crate::legacy_qualification_source_ledger_v3::{
    LegacyQualificationSourceLedgerV3, LegacyQualificationSourceSelectionV3,
};
use crate::legacy_source_capture_plan_v3::legacy_source_revision_commitment_v3;
use crate::standards_registry::{SourceCaptureV1, SourceSnapshotIdV1};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const LEGACY_SOURCE_CONTINUITY_SCHEMA_V1: &str =
    "symthaea-it-legacy-source-continuity-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacySourceContinuityStatusV1 {
    ExactContentMatch,
    RebaselineRequired,
    ContentMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySourceContinuityItemV1 {
    pub original_snapshot_id: SourceSnapshotIdV1,
    pub qualifying_snapshot_id: SourceSnapshotIdV1,
    pub status: LegacySourceContinuityStatusV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySourceContinuityAssessmentV1 {
    pub schema_version: String,
    pub total_selections: usize,
    pub exact_content_matches: usize,
    pub rebaseline_required: BTreeSet<SourceSnapshotIdV1>,
    pub content_mismatches: BTreeSet<SourceSnapshotIdV1>,
    pub qualification_safe: bool,
    pub items: Vec<LegacySourceContinuityItemV1>,
}

/// Assess byte-identity continuity for selections already present in the V3
/// qualification source ledger. This is deliberately narrower than full ledger
/// validation: profile qualification first validates the ledger/artifact chain,
/// then applies this independent continuity predicate.
pub fn assess_legacy_source_continuity_v1(
    pack: &LegacyComputingPackV1,
    ledger: &LegacyQualificationSourceLedgerV3,
) -> Result<LegacySourceContinuityAssessmentV1, LegacySourceContinuityErrorV1> {
    pack.validate()
        .map_err(|err| LegacySourceContinuityErrorV1::InvalidPack(err.to_string()))?;

    let mut items = Vec::new();
    let mut exact_content_matches = 0usize;
    let mut rebaseline_required = BTreeSet::new();
    let mut content_mismatches = BTreeSet::new();

    for selection in ledger.selections() {
        let status = continuity_status(pack, selection)?;
        match status {
            LegacySourceContinuityStatusV1::ExactContentMatch => {
                exact_content_matches += 1;
            }
            LegacySourceContinuityStatusV1::RebaselineRequired => {
                rebaseline_required.insert(selection.original_snapshot_id.clone());
            }
            LegacySourceContinuityStatusV1::ContentMismatch => {
                content_mismatches.insert(selection.original_snapshot_id.clone());
            }
        }
        items.push(LegacySourceContinuityItemV1 {
            original_snapshot_id: selection.original_snapshot_id.clone(),
            qualifying_snapshot_id: selection.qualifying_snapshot_id.clone(),
            status,
        });
    }

    items.sort_by(|a, b| a.original_snapshot_id.cmp(&b.original_snapshot_id));
    let total_selections = items.len();
    let qualification_safe = rebaseline_required.is_empty() && content_mismatches.is_empty();

    Ok(LegacySourceContinuityAssessmentV1 {
        schema_version: LEGACY_SOURCE_CONTINUITY_SCHEMA_V1.into(),
        total_selections,
        exact_content_matches,
        rebaseline_required,
        content_mismatches,
        qualification_safe,
        items,
    })
}

pub fn require_legacy_source_continuity_v1(
    pack: &LegacyComputingPackV1,
    ledger: &LegacyQualificationSourceLedgerV3,
) -> Result<LegacySourceContinuityAssessmentV1, LegacySourceContinuityErrorV1> {
    let assessment = assess_legacy_source_continuity_v1(pack, ledger)?;
    if assessment.qualification_safe {
        return Ok(assessment);
    }
    Err(LegacySourceContinuityErrorV1::UnsafeContinuity {
        rebaseline_required: assessment.rebaseline_required,
        content_mismatches: assessment.content_mismatches,
    })
}

fn continuity_status(
    pack: &LegacyComputingPackV1,
    selection: &LegacyQualificationSourceSelectionV3,
) -> Result<LegacySourceContinuityStatusV1, LegacySourceContinuityErrorV1> {
    let original = pack
        .sources
        .snapshot(&selection.original_snapshot_id)
        .ok_or_else(|| {
            LegacySourceContinuityErrorV1::UnknownSnapshot(
                selection.original_snapshot_id.clone(),
            )
        })?;
    let qualifying = pack
        .sources
        .snapshot(&selection.qualifying_snapshot_id)
        .ok_or_else(|| {
            LegacySourceContinuityErrorV1::UnknownSnapshot(
                selection.qualifying_snapshot_id.clone(),
            )
        })?;
    let document = pack.sources.document(&original.document_id).ok_or_else(|| {
        LegacySourceContinuityErrorV1::UnknownDocument(original.document_id.0.clone())
    })?;

    let expected_revision = legacy_source_revision_commitment_v3(document, original)
        .map_err(|err| LegacySourceContinuityErrorV1::Revision(err.to_string()))?;
    if expected_revision != selection.source_revision_blake3.trim().to_ascii_lowercase() {
        return Err(LegacySourceContinuityErrorV1::SourceRevisionMismatch(
            selection.original_snapshot_id.clone(),
        ));
    }

    let SourceCaptureV1::ContentDigest {
        algorithm: qualifying_algorithm,
        digest: qualifying_digest,
    } = &qualifying.capture
    else {
        return Err(LegacySourceContinuityErrorV1::QualifyingSnapshotNotContentBound(
            qualifying.id.clone(),
        ));
    };

    if !qualifying_algorithm.eq_ignore_ascii_case(selection.content_algorithm.trim())
        || qualifying_digest.to_ascii_lowercase()
            != selection.content_digest.trim().to_ascii_lowercase()
    {
        return Err(LegacySourceContinuityErrorV1::SelectionDigestMismatch(
            qualifying.id.clone(),
        ));
    }

    match &original.capture {
        SourceCaptureV1::ContentDigest { algorithm, digest } => {
            if algorithm.eq_ignore_ascii_case(selection.content_algorithm.trim())
                && digest.to_ascii_lowercase()
                    == selection.content_digest.trim().to_ascii_lowercase()
            {
                Ok(LegacySourceContinuityStatusV1::ExactContentMatch)
            } else {
                Ok(LegacySourceContinuityStatusV1::ContentMismatch)
            }
        }
        SourceCaptureV1::MetadataOnly | SourceCaptureV1::MetadataDigest { .. } => {
            Ok(LegacySourceContinuityStatusV1::RebaselineRequired)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegacySourceContinuityErrorV1 {
    InvalidPack(String),
    UnknownDocument(String),
    UnknownSnapshot(SourceSnapshotIdV1),
    Revision(String),
    SourceRevisionMismatch(SourceSnapshotIdV1),
    QualifyingSnapshotNotContentBound(SourceSnapshotIdV1),
    SelectionDigestMismatch(SourceSnapshotIdV1),
    UnsafeContinuity {
        rebaseline_required: BTreeSet<SourceSnapshotIdV1>,
        content_mismatches: BTreeSet<SourceSnapshotIdV1>,
    },
}

impl fmt::Display for LegacySourceContinuityErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPack(err) => write!(f, "legacy source-continuity pack is invalid: {err}"),
            Self::UnknownDocument(id) => {
                write!(f, "legacy source-continuity references unknown document {id}")
            }
            Self::UnknownSnapshot(id) => {
                write!(f, "legacy source-continuity references unknown snapshot {}", id.0)
            }
            Self::Revision(err) => write!(f, "legacy source-revision commitment failed: {err}"),
            Self::SourceRevisionMismatch(id) => write!(
                f,
                "legacy source-continuity revision commitment mismatched for {}",
                id.0
            ),
            Self::QualifyingSnapshotNotContentBound(id) => write!(
                f,
                "legacy source-continuity qualifying snapshot {} is not content-bound",
                id.0
            ),
            Self::SelectionDigestMismatch(id) => write!(
                f,
                "legacy source-continuity selection does not match qualifying snapshot {}",
                id.0
            ),
            Self::UnsafeContinuity {
                rebaseline_required,
                content_mismatches,
            } => write!(
                f,
                "legacy source continuity is unsafe: {} selections require rebaseline and {} content-bound selections mismatch",
                rebaseline_required.len(),
                content_mismatches.len()
            ),
        }
    }
}

impl Error for LegacySourceContinuityErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legacy_qualification_source_ledger_v3::LegacyQualificationSourceSelectionV3;
    use crate::legacy_source_artifacts::{
        LegacyArtifactAccessPolicyV1, LegacyArtifactStorageClassV1,
        LegacySourceArtifactLedgerV1, LegacySourceArtifactRefV1,
    };
    use crate::standards_registry::TechnicalSourceSnapshotV1;
    use crate::{build_legacy_five_platform_portfolio_v1, legacy_source_revision_commitment_v3};

    fn register_selection(
        pack: &mut LegacyComputingPackV1,
        original_id: &str,
        qualifying_suffix: &str,
        digest: &str,
    ) -> (LegacySourceArtifactLedgerV1, LegacyQualificationSourceLedgerV3) {
        let original = pack
            .sources
            .snapshot(&SourceSnapshotIdV1(original_id.into()))
            .unwrap()
            .clone();
        let qualifying_id = SourceSnapshotIdV1(format!("{original_id}:{qualifying_suffix}"));
        pack.sources
            .register_snapshot(TechnicalSourceSnapshotV1 {
                id: qualifying_id.clone(),
                document_id: original.document_id.clone(),
                version: original.version.clone(),
                lifecycle: original.lifecycle,
                authority: original.authority,
                stability: original.stability,
                published_at_unix_ms: original.published_at_unix_ms,
                source_updated_at_unix_ms: original.source_updated_at_unix_ms,
                fetched_at_unix_ms: original.fetched_at_unix_ms + 10,
                capture: SourceCaptureV1::ContentDigest {
                    algorithm: "sha256".into(),
                    digest: digest.into(),
                },
                relations: original.relations.clone(),
            })
            .unwrap();

        let mut artifacts = LegacySourceArtifactLedgerV1::new();
        artifacts
            .register(
                pack,
                LegacySourceArtifactRefV1 {
                    snapshot_id: qualifying_id.clone(),
                    content_algorithm: "sha256".into(),
                    content_digest: digest.into(),
                    byte_length: 1,
                    media_type: "application/octet-stream".into(),
                    retrieved_at_unix_ms: original.fetched_at_unix_ms + 10,
                    artifact_locator: format!("evidence://continuity/{qualifying_suffix}"),
                    storage_class: LegacyArtifactStorageClassV1::PrivateEvidenceStore,
                    access_policy: LegacyArtifactAccessPolicyV1::EvaluatorOnly,
                    retention_receipt_digest: Some("a".repeat(64)),
                },
            )
            .unwrap();

        let document = pack.sources.document(&original.document_id).unwrap();
        let selection = LegacyQualificationSourceSelectionV3 {
            original_snapshot_id: original.id.clone(),
            source_revision_blake3: legacy_source_revision_commitment_v3(document, &original)
                .unwrap(),
            qualifying_snapshot_id: qualifying_id,
            content_algorithm: "sha256".into(),
            content_digest: digest.into(),
            selected_at_unix_ms: original.fetched_at_unix_ms + 11,
        };
        let mut ledger = LegacyQualificationSourceLedgerV3::new();
        ledger
            .register_selection(pack, &artifacts, selection)
            .unwrap();
        (artifacts, ledger)
    }

    #[test]
    fn empty_ledger_is_continuity_safe_but_not_source_ready() {
        let (pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let ledger = LegacyQualificationSourceLedgerV3::new();
        let assessment = require_legacy_source_continuity_v1(&pack, &ledger).unwrap();
        assert!(assessment.qualification_safe);
        assert_eq!(assessment.total_selections, 0);
    }

    #[test]
    fn metadata_only_history_requires_rebaseline_after_later_capture() {
        let (mut pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let (_artifacts, ledger) = register_selection(
            &mut pack,
            "ibm:aix-os-management@7.3",
            "later-content",
            &"1".repeat(64),
        );
        let assessment = assess_legacy_source_continuity_v1(&pack, &ledger).unwrap();
        assert!(!assessment.qualification_safe);
        assert!(assessment.rebaseline_required.contains(&SourceSnapshotIdV1(
            "ibm:aix-os-management@7.3".into()
        )));
        assert_eq!(assessment.content_mismatches.len(), 0);
        assert!(matches!(
            require_legacy_source_continuity_v1(&pack, &ledger),
            Err(LegacySourceContinuityErrorV1::UnsafeContinuity { .. })
        ));
    }

    #[test]
    fn content_bound_original_with_same_digest_is_exact_continuity() {
        let (mut pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let original_id = SourceSnapshotIdV1("ibm:aix-os-management@7.3".into());
        let original = pack.sources.snapshot(&original_id).unwrap().clone();
        let digest = "2".repeat(64);
        let mut content_bound = original.clone();
        content_bound.capture = SourceCaptureV1::ContentDigest {
            algorithm: "sha256".into(),
            digest: digest.clone(),
        };
        pack.sources.replace_snapshot_for_test(content_bound).unwrap();
        let (_artifacts, ledger) =
            register_selection(&mut pack, &original_id.0, "same-content", &digest);
        let assessment = require_legacy_source_continuity_v1(&pack, &ledger).unwrap();
        assert!(assessment.qualification_safe);
        assert_eq!(assessment.exact_content_matches, 1);
    }

    #[test]
    fn content_bound_original_with_different_digest_is_mismatch() {
        let (mut pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let original_id = SourceSnapshotIdV1("ibm:aix-os-management@7.3".into());
        let original = pack.sources.snapshot(&original_id).unwrap().clone();
        let mut content_bound = original.clone();
        content_bound.capture = SourceCaptureV1::ContentDigest {
            algorithm: "sha256".into(),
            digest: "3".repeat(64),
        };
        pack.sources.replace_snapshot_for_test(content_bound).unwrap();
        let (_artifacts, ledger) = register_selection(
            &mut pack,
            &original_id.0,
            "different-content",
            &"4".repeat(64),
        );
        let assessment = assess_legacy_source_continuity_v1(&pack, &ledger).unwrap();
        assert!(assessment.content_mismatches.contains(&original_id));
        assert!(!assessment.qualification_safe);
    }
}
