// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Qualification-readiness checks for legacy-computing source snapshots.
//!
//! A versioned URL is not an immutable content artifact. Legacy vendor pages can
//! change in place, so advisory discovery and reproducible qualification are
//! deliberately separate use classes.

use crate::legacy_computing::{LegacyComputingErrorV1, LegacyComputingPackV1};
use crate::standards_registry::{SourceCaptureV1, SourceSnapshotIdV1};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegacyKnowledgeUseClassV1 {
    /// Search, orientation, and operator-reviewed reasoning may use metadata-only
    /// source snapshots, while preserving their weaker provenance status.
    AdvisoryDiscovery,
    /// Reproducible qualification requires content-digest-bound source material.
    Qualification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegacySourceCaptureClassV1 {
    MetadataOnly,
    MetadataDigestBound,
    ContentDigestBound,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySourceReadinessItemV1 {
    pub snapshot_id: SourceSnapshotIdV1,
    pub capture_class: LegacySourceCaptureClassV1,
    pub qualification_ready: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySourceReadinessAssessmentV1 {
    pub total_snapshots: usize,
    pub content_digest_bound: usize,
    pub metadata_digest_bound: usize,
    pub metadata_only: usize,
    pub qualification_ready: bool,
    pub items: Vec<LegacySourceReadinessItemV1>,
}

impl LegacySourceReadinessAssessmentV1 {
    pub fn blockers(&self) -> Vec<SourceSnapshotIdV1> {
        self.items
            .iter()
            .filter(|item| !item.qualification_ready)
            .map(|item| item.snapshot_id.clone())
            .collect()
    }
}

pub fn assess_legacy_source_readiness_v1(
    pack: &LegacyComputingPackV1,
) -> Result<LegacySourceReadinessAssessmentV1, LegacySourceReadinessErrorV1> {
    pack.validate()?;

    let mut items = Vec::new();
    let mut content_digest_bound = 0usize;
    let mut metadata_digest_bound = 0usize;
    let mut metadata_only = 0usize;

    for snapshot in pack.sources.snapshots() {
        let capture_class = match &snapshot.capture {
            SourceCaptureV1::MetadataOnly => {
                metadata_only += 1;
                LegacySourceCaptureClassV1::MetadataOnly
            }
            SourceCaptureV1::MetadataDigest { .. } => {
                metadata_digest_bound += 1;
                LegacySourceCaptureClassV1::MetadataDigestBound
            }
            SourceCaptureV1::ContentDigest { .. } => {
                content_digest_bound += 1;
                LegacySourceCaptureClassV1::ContentDigestBound
            }
        };
        items.push(LegacySourceReadinessItemV1 {
            snapshot_id: snapshot.id.clone(),
            capture_class,
            qualification_ready: capture_class == LegacySourceCaptureClassV1::ContentDigestBound,
        });
    }
    items.sort_by(|a, b| a.snapshot_id.cmp(&b.snapshot_id));

    let total_snapshots = items.len();
    let qualification_ready = total_snapshots > 0 && content_digest_bound == total_snapshots;
    Ok(LegacySourceReadinessAssessmentV1 {
        total_snapshots,
        content_digest_bound,
        metadata_digest_bound,
        metadata_only,
        qualification_ready,
        items,
    })
}

pub fn admit_legacy_knowledge_use_v1(
    pack: &LegacyComputingPackV1,
    use_class: LegacyKnowledgeUseClassV1,
) -> Result<LegacySourceReadinessAssessmentV1, LegacySourceReadinessErrorV1> {
    let assessment = assess_legacy_source_readiness_v1(pack)?;
    match use_class {
        LegacyKnowledgeUseClassV1::AdvisoryDiscovery => Ok(assessment),
        LegacyKnowledgeUseClassV1::Qualification if assessment.qualification_ready => Ok(assessment),
        LegacyKnowledgeUseClassV1::Qualification => Err(
            LegacySourceReadinessErrorV1::QualificationRequiresContentDigests {
                blockers: assessment.blockers(),
            },
        ),
    }
}

#[derive(Debug)]
pub enum LegacySourceReadinessErrorV1 {
    LegacyPack(LegacyComputingErrorV1),
    QualificationRequiresContentDigests {
        blockers: Vec<SourceSnapshotIdV1>,
    },
}

impl fmt::Display for LegacySourceReadinessErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyPack(err) => write!(f, "invalid legacy computing pack: {err}"),
            Self::QualificationRequiresContentDigests { blockers } => write!(
                f,
                "legacy qualification requires content-digest-bound source snapshots; blockers={:?}",
                blockers.iter().map(|id| id.0.as_str()).collect::<Vec<_>>()
            ),
        }
    }
}

impl Error for LegacySourceReadinessErrorV1 {}

impl From<LegacyComputingErrorV1> for LegacySourceReadinessErrorV1 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::LegacyPack(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed_legacy_computing_pack_v1;

    #[test]
    fn seed_pack_is_explicitly_discovery_ready_but_not_qualification_ready() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let assessment = admit_legacy_knowledge_use_v1(
            &pack,
            LegacyKnowledgeUseClassV1::AdvisoryDiscovery,
        )
        .unwrap();
        assert!(assessment.total_snapshots > 0);
        assert_eq!(assessment.metadata_only, assessment.total_snapshots);
        assert_eq!(assessment.content_digest_bound, 0);
        assert!(!assessment.qualification_ready);
    }

    #[test]
    fn metadata_only_vendor_pages_fail_closed_for_qualification() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let error = admit_legacy_knowledge_use_v1(
            &pack,
            LegacyKnowledgeUseClassV1::Qualification,
        )
        .unwrap_err();
        match error {
            LegacySourceReadinessErrorV1::QualificationRequiresContentDigests { blockers } => {
                assert_eq!(blockers.len(), pack.sources.snapshots().count());
                assert!(!blockers.is_empty());
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
