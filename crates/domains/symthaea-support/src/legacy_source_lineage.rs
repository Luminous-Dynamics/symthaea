// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Logical source-lineage accounting for legacy IT knowledge.
//!
//! A source snapshot is an observation of a logical document, not an independent
//! authority. Multiple snapshots of the same document therefore remain one source
//! lineage for corroboration accounting. This module does not claim that distinct
//! documents are statistically independent; it only prevents the weaker error of
//! counting repeated captures of one document as separate corroboration.

use crate::legacy_computing::{LegacyComputingErrorV1, LegacyComputingPackV1};
use crate::standards_registry::{
    SourceDocumentIdV1, SourceSnapshotIdV1, TechnicalClaimIdV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_SOURCE_LINEAGE_SCHEMA_V1: &str = "symthaea-it-legacy-source-lineage-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySourceLineageGroupV1 {
    pub document_id: SourceDocumentIdV1,
    pub snapshot_ids: BTreeSet<SourceSnapshotIdV1>,
    pub claim_ids: BTreeSet<TechnicalClaimIdV1>,
}

impl LegacySourceLineageGroupV1 {
    pub fn has_multiple_snapshots(&self) -> bool {
        self.snapshot_ids.len() > 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySourceLineageAssessmentV1 {
    pub schema_version: String,
    pub snapshot_count: usize,
    pub logical_document_count: usize,
    pub multi_snapshot_document_count: usize,
    pub groups: Vec<LegacySourceLineageGroupV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyClaimLineageSummaryV1 {
    pub claim_count: usize,
    pub snapshot_count: usize,
    pub logical_document_count: usize,
    pub snapshot_ids: BTreeSet<SourceSnapshotIdV1>,
    pub document_ids: BTreeSet<SourceDocumentIdV1>,
}

/// Inventory source captures by their stable logical document identity.
///
/// This deliberately does *not* label distinct documents as independent sources.
/// Vendor mirrors, derivative documents, and common upstream material may still
/// share a deeper lineage that requires explicit provenance analysis later.
pub fn assess_legacy_source_lineage_v1(
    pack: &LegacyComputingPackV1,
) -> Result<LegacySourceLineageAssessmentV1, LegacySourceLineageErrorV1> {
    pack.validate()?;

    let mut groups = BTreeMap::<SourceDocumentIdV1, LegacySourceLineageGroupV1>::new();
    for snapshot in pack.sources.snapshots() {
        groups
            .entry(snapshot.document_id.clone())
            .or_insert_with(|| LegacySourceLineageGroupV1 {
                document_id: snapshot.document_id.clone(),
                snapshot_ids: BTreeSet::new(),
                claim_ids: BTreeSet::new(),
            })
            .snapshot_ids
            .insert(snapshot.id.clone());
    }

    for claim in pack.sources.claims() {
        let snapshot = pack
            .sources
            .snapshot(&claim.source_snapshot)
            .ok_or_else(|| LegacySourceLineageErrorV1::UnknownSnapshot(claim.source_snapshot.clone()))?;
        let group = groups
            .get_mut(&snapshot.document_id)
            .ok_or_else(|| LegacySourceLineageErrorV1::UnknownDocument(snapshot.document_id.clone()))?;
        group.claim_ids.insert(claim.id.clone());
    }

    let snapshot_count = groups.values().map(|group| group.snapshot_ids.len()).sum();
    let logical_document_count = groups.len();
    let multi_snapshot_document_count = groups
        .values()
        .filter(|group| group.has_multiple_snapshots())
        .count();

    Ok(LegacySourceLineageAssessmentV1 {
        schema_version: LEGACY_SOURCE_LINEAGE_SCHEMA_V1.into(),
        snapshot_count,
        logical_document_count,
        multi_snapshot_document_count,
        groups: groups.into_values().collect(),
    })
}

/// Count the source lineages behind a selected claim set.
///
/// `logical_document_count` is the maximum corroboration count that can be
/// inferred from document identity alone. It is intentionally more conservative
/// than snapshot count and intentionally less strong than an independence claim.
pub fn summarize_legacy_claim_lineage_v1(
    pack: &LegacyComputingPackV1,
    claim_ids: &BTreeSet<TechnicalClaimIdV1>,
) -> Result<LegacyClaimLineageSummaryV1, LegacySourceLineageErrorV1> {
    pack.validate()?;
    if claim_ids.is_empty() {
        return Err(LegacySourceLineageErrorV1::InvalidInput(
            "claim-lineage summary requires at least one claim".into(),
        ));
    }

    let mut snapshot_ids = BTreeSet::new();
    let mut document_ids = BTreeSet::new();
    for claim_id in claim_ids {
        let claim = pack
            .sources
            .claim(claim_id)
            .ok_or_else(|| LegacySourceLineageErrorV1::UnknownClaim(claim_id.clone()))?;
        let snapshot = pack
            .sources
            .snapshot(&claim.source_snapshot)
            .ok_or_else(|| LegacySourceLineageErrorV1::UnknownSnapshot(claim.source_snapshot.clone()))?;
        snapshot_ids.insert(snapshot.id.clone());
        document_ids.insert(snapshot.document_id.clone());
    }

    Ok(LegacyClaimLineageSummaryV1 {
        claim_count: claim_ids.len(),
        snapshot_count: snapshot_ids.len(),
        logical_document_count: document_ids.len(),
        snapshot_ids,
        document_ids,
    })
}

#[derive(Debug)]
pub enum LegacySourceLineageErrorV1 {
    Computing(LegacyComputingErrorV1),
    InvalidInput(String),
    UnknownClaim(TechnicalClaimIdV1),
    UnknownSnapshot(SourceSnapshotIdV1),
    UnknownDocument(SourceDocumentIdV1),
}

impl fmt::Display for LegacySourceLineageErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Computing(err) => write!(f, "legacy source-lineage pack error: {err}"),
            Self::InvalidInput(message) => write!(f, "invalid legacy source-lineage input: {message}"),
            Self::UnknownClaim(id) => write!(f, "unknown legacy source-lineage claim {}", id.0),
            Self::UnknownSnapshot(id) => write!(f, "unknown legacy source-lineage snapshot {}", id.0),
            Self::UnknownDocument(id) => write!(f, "unknown legacy source-lineage document {}", id.0),
        }
    }
}

impl Error for LegacySourceLineageErrorV1 {}

impl From<LegacyComputingErrorV1> for LegacySourceLineageErrorV1 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::Computing(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_legacy_five_platform_portfolio_v1;

    fn portfolio() -> LegacyComputingPackV1 {
        build_legacy_five_platform_portfolio_v1(1_800_000_000_000)
            .unwrap()
            .0
    }

    #[test]
    fn repeated_captures_of_one_document_remain_one_lineage() {
        let pack = portfolio();
        let assessment = assess_legacy_source_lineage_v1(&pack).unwrap();
        assert!(assessment.snapshot_count >= assessment.logical_document_count);
        let install = assessment
            .groups
            .iter()
            .find(|group| group.document_id.0 == "hpe:hpux-install-update")
            .unwrap();
        assert_eq!(install.snapshot_ids.len(), 2);
        assert!(install.has_multiple_snapshots());
        assert!(assessment.multi_snapshot_document_count >= 1);
    }

    #[test]
    fn hpux_install_claims_do_not_gain_false_corroboration_from_two_snapshots() {
        let pack = portfolio();
        let claims = [
            TechnicalClaimIdV1("legacy:hpux:install-verification".into()),
            TechnicalClaimIdV1("legacy:hpux:ignite-recovery-context".into()),
            TechnicalClaimIdV1("legacy:hpux:software-distributor-verification".into()),
        ]
        .into_iter()
        .collect();
        let summary = summarize_legacy_claim_lineage_v1(&pack, &claims).unwrap();
        assert_eq!(summary.claim_count, 3);
        assert_eq!(summary.snapshot_count, 2);
        assert_eq!(summary.logical_document_count, 1);
        assert_eq!(summary.document_ids.len(), 1);
        assert!(summary.document_ids.contains(&SourceDocumentIdV1(
            "hpe:hpux-install-update".into()
        )));
    }

    #[test]
    fn unknown_claim_is_rejected_instead_of_silently_ignored() {
        let pack = portfolio();
        let claims = [TechnicalClaimIdV1("legacy:missing:claim".into())]
            .into_iter()
            .collect();
        assert!(matches!(
            summarize_legacy_claim_lineage_v1(&pack, &claims),
            Err(LegacySourceLineageErrorV1::UnknownClaim(_))
        ));
    }

    #[test]
    fn empty_corroboration_request_is_invalid() {
        let pack = portfolio();
        let claims = BTreeSet::new();
        assert!(matches!(
            summarize_legacy_claim_lineage_v1(&pack, &claims),
            Err(LegacySourceLineageErrorV1::InvalidInput(_))
        ));
    }
}
