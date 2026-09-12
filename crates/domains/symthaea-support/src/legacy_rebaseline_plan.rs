// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, non-authoritative recovery planning for legacy source
//! continuity failures.
//!
//! `RebaselineRequired` and `ContentMismatch` are deliberately different
//! recovery classes:
//!
//! ```text
//! metadata-only history + later bytes
//!   -> new content baseline
//!   -> re-review dependent claims/procedures
//!   -> successor manifest
//!   -> semantic review + external trust verification
//!
//! content-bound history + wrong selected bytes
//!   -> recover exact known historical content
//!   -> repair source selection
//!   -> keep claim/procedure identities unchanged
//! ```
//!
//! This module creates no source snapshot, evidence receipt, semantic judgment,
//! execution authority, or qualification result. It is an ephemeral work plan.

use crate::legacy_computing::{LegacyComputingPackV1, LegacyProcedureV1};
use crate::legacy_qualification_profile_v3::{
    legacy_qualification_manifest_commitment_v1, LegacyQualificationManifestErrorV1,
    LegacyQualificationManifestV1,
};
use crate::legacy_source_continuity::{
    LegacySourceContinuityAssessmentV1, LegacySourceContinuityItemV1,
    LegacySourceContinuityStatusV1, LEGACY_SOURCE_CONTINUITY_SCHEMA_V1,
};
use crate::standards_registry::{SourceSnapshotIdV1, TechnicalClaimIdV1};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const LEGACY_REBASELINE_PLAN_SCHEMA_V1: &str = "symthaea-it-legacy-rebaseline-plan-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacyRebaselineRecoveryKindV1 {
    /// Historical bytes were never captured exactly; current bytes become a new
    /// baseline and dependent knowledge must be reviewed under new identities.
    CreateSuccessorKnowledgeGeneration,
    /// Historical bytes are already known by content identity; the selected
    /// evidence is wrong and should be repaired without rewriting knowledge.
    RepairHistoricalEvidenceSelection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacyRebaselineStageV1 {
    // Rebaseline path.
    CaptureCurrentSourceBytes,
    RegisterContentBoundSuccessorSnapshot,
    ReReviewAffectedClaims,
    ReReviewAffectedProcedures,
    BuildSuccessorQualificationManifest,
    CollectSemanticEquivalenceReviews,
    ExternallyVerifySemanticReviews,
    CollectSuccessorClaimProcedureSourceVerification,
    AssessSuccessorQualificationGeneration,
    // Historical-selection repair path.
    LocateExactHistoricalArtifact,
    RepairQualificationSourceSelection,
    RevalidateCurrentQualificationGeneration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyRebaselineWorkItemV1 {
    pub original_snapshot_id: SourceSnapshotIdV1,
    pub observed_qualifying_snapshot_id: SourceSnapshotIdV1,
    pub recovery_kind: LegacyRebaselineRecoveryKindV1,
    pub affected_claim_ids: BTreeSet<TechnicalClaimIdV1>,
    pub affected_procedure_ids: BTreeSet<String>,
    /// Strictly ordered planning stages. Completion of stage N is a prerequisite
    /// for stage N+1; this vector is not execution authority.
    pub ordered_stages: Vec<LegacyRebaselineStageV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyRebaselineWorkPackageV1 {
    pub schema_version: String,
    pub manifest_generation: u64,
    pub manifest_blake3: String,
    pub continuity_schema_version: String,
    pub qualification_blocked: bool,
    pub requires_successor_generation: bool,
    pub exact_content_matches: usize,
    pub rebaseline_items: Vec<LegacyRebaselineWorkItemV1>,
    pub historical_selection_repair_items: Vec<LegacyRebaselineWorkItemV1>,
    pub claims_requiring_replacement: BTreeSet<TechnicalClaimIdV1>,
    pub procedures_requiring_replacement: BTreeSet<String>,
    pub plan_blake3: String,
}

pub fn plan_legacy_rebaseline_work_v1(
    pack: &LegacyComputingPackV1,
    manifest: &LegacyQualificationManifestV1,
    continuity: &LegacySourceContinuityAssessmentV1,
) -> Result<LegacyRebaselineWorkPackageV1, LegacyRebaselinePlanErrorV1> {
    manifest.validate_basic(pack)?;
    validate_continuity_assessment(continuity)?;

    let manifest_blake3 = legacy_qualification_manifest_commitment_v1(manifest)?;
    let mut rebaseline_items = Vec::new();
    let mut historical_selection_repair_items = Vec::new();
    let mut claims_requiring_replacement = BTreeSet::new();
    let mut procedures_requiring_replacement = BTreeSet::new();

    for item in &continuity.items {
        if item.status == LegacySourceContinuityStatusV1::ExactContentMatch {
            continue;
        }
        let (affected_claim_ids, affected_procedure_ids) =
            affected_active_targets(pack, manifest, &item.original_snapshot_id)?;
        if affected_claim_ids.is_empty() && affected_procedure_ids.is_empty() {
            return Err(LegacyRebaselinePlanErrorV1::ContinuitySnapshotNotActive(
                item.original_snapshot_id.clone(),
            ));
        }

        match item.status {
            LegacySourceContinuityStatusV1::RebaselineRequired => {
                claims_requiring_replacement.extend(affected_claim_ids.iter().cloned());
                procedures_requiring_replacement.extend(affected_procedure_ids.iter().cloned());
                rebaseline_items.push(LegacyRebaselineWorkItemV1 {
                    original_snapshot_id: item.original_snapshot_id.clone(),
                    observed_qualifying_snapshot_id: item.qualifying_snapshot_id.clone(),
                    recovery_kind: LegacyRebaselineRecoveryKindV1::CreateSuccessorKnowledgeGeneration,
                    ordered_stages: rebaseline_stages(
                        !affected_claim_ids.is_empty(),
                        !affected_procedure_ids.is_empty(),
                    ),
                    affected_claim_ids,
                    affected_procedure_ids,
                });
            }
            LegacySourceContinuityStatusV1::ContentMismatch => {
                historical_selection_repair_items.push(LegacyRebaselineWorkItemV1 {
                    original_snapshot_id: item.original_snapshot_id.clone(),
                    observed_qualifying_snapshot_id: item.qualifying_snapshot_id.clone(),
                    recovery_kind: LegacyRebaselineRecoveryKindV1::RepairHistoricalEvidenceSelection,
                    affected_claim_ids,
                    affected_procedure_ids,
                    ordered_stages: vec![
                        LegacyRebaselineStageV1::LocateExactHistoricalArtifact,
                        LegacyRebaselineStageV1::RepairQualificationSourceSelection,
                        LegacyRebaselineStageV1::RevalidateCurrentQualificationGeneration,
                    ],
                });
            }
            LegacySourceContinuityStatusV1::ExactContentMatch => unreachable!(),
        }
    }

    rebaseline_items.sort_by(|a, b| a.original_snapshot_id.cmp(&b.original_snapshot_id));
    historical_selection_repair_items
        .sort_by(|a, b| a.original_snapshot_id.cmp(&b.original_snapshot_id));

    let qualification_blocked = !rebaseline_items.is_empty()
        || !historical_selection_repair_items.is_empty();
    let requires_successor_generation = !rebaseline_items.is_empty();

    let plan_blake3 = rebaseline_plan_binding(
        manifest.generation,
        &manifest_blake3,
        continuity,
        &rebaseline_items,
        &historical_selection_repair_items,
        &claims_requiring_replacement,
        &procedures_requiring_replacement,
    )?;

    Ok(LegacyRebaselineWorkPackageV1 {
        schema_version: LEGACY_REBASELINE_PLAN_SCHEMA_V1.into(),
        manifest_generation: manifest.generation,
        manifest_blake3,
        continuity_schema_version: continuity.schema_version.clone(),
        qualification_blocked,
        requires_successor_generation,
        exact_content_matches: continuity.exact_content_matches,
        rebaseline_items,
        historical_selection_repair_items,
        claims_requiring_replacement,
        procedures_requiring_replacement,
        plan_blake3,
    })
}

fn affected_active_targets(
    pack: &LegacyComputingPackV1,
    manifest: &LegacyQualificationManifestV1,
    snapshot_id: &SourceSnapshotIdV1,
) -> Result<(BTreeSet<TechnicalClaimIdV1>, BTreeSet<String>), LegacyRebaselinePlanErrorV1> {
    if pack.sources.snapshot(snapshot_id).is_none() {
        return Err(LegacyRebaselinePlanErrorV1::UnknownSnapshot(snapshot_id.clone()));
    }

    let affected_claim_ids = manifest
        .claim_ids
        .iter()
        .filter_map(|claim_id| {
            pack.sources
                .claim(claim_id)
                .filter(|claim| &claim.source_snapshot == snapshot_id)
                .map(|claim| claim.id.clone())
        })
        .collect::<BTreeSet<_>>();

    let affected_procedure_ids = manifest
        .procedure_ids
        .iter()
        .filter_map(|procedure_id| {
            find_procedure(pack, procedure_id)
                .filter(|procedure| procedure.source_snapshots.contains(snapshot_id))
                .map(|procedure| procedure.id.clone())
        })
        .collect::<BTreeSet<_>>();

    Ok((affected_claim_ids, affected_procedure_ids))
}

fn find_procedure<'a>(pack: &'a LegacyComputingPackV1, id: &str) -> Option<&'a LegacyProcedureV1> {
    pack.procedures.iter().find(|procedure| procedure.id == id)
}

fn rebaseline_stages(has_claims: bool, has_procedures: bool) -> Vec<LegacyRebaselineStageV1> {
    let mut stages = vec![
        LegacyRebaselineStageV1::CaptureCurrentSourceBytes,
        LegacyRebaselineStageV1::RegisterContentBoundSuccessorSnapshot,
    ];
    if has_claims {
        stages.push(LegacyRebaselineStageV1::ReReviewAffectedClaims);
    }
    if has_procedures {
        stages.push(LegacyRebaselineStageV1::ReReviewAffectedProcedures);
    }
    stages.extend([
        LegacyRebaselineStageV1::BuildSuccessorQualificationManifest,
        LegacyRebaselineStageV1::CollectSemanticEquivalenceReviews,
        LegacyRebaselineStageV1::ExternallyVerifySemanticReviews,
        LegacyRebaselineStageV1::CollectSuccessorClaimProcedureSourceVerification,
        LegacyRebaselineStageV1::AssessSuccessorQualificationGeneration,
    ]);
    stages
}

fn validate_continuity_assessment(
    continuity: &LegacySourceContinuityAssessmentV1,
) -> Result<(), LegacyRebaselinePlanErrorV1> {
    if continuity.schema_version != LEGACY_SOURCE_CONTINUITY_SCHEMA_V1 {
        return Err(LegacyRebaselinePlanErrorV1::UnsupportedContinuitySchema(
            continuity.schema_version.clone(),
        ));
    }
    if continuity.total_selections != continuity.items.len() {
        return Err(LegacyRebaselinePlanErrorV1::InconsistentContinuityAssessment(
            "total selection count does not match items".into(),
        ));
    }

    let mut seen = BTreeSet::new();
    let mut exact_content_matches = 0usize;
    let mut rebaseline_required = BTreeSet::new();
    let mut content_mismatches = BTreeSet::new();
    for item in &continuity.items {
        if !seen.insert(item.original_snapshot_id.clone()) {
            return Err(LegacyRebaselinePlanErrorV1::DuplicateContinuitySnapshot(
                item.original_snapshot_id.clone(),
            ));
        }
        match item.status {
            LegacySourceContinuityStatusV1::ExactContentMatch => exact_content_matches += 1,
            LegacySourceContinuityStatusV1::RebaselineRequired => {
                rebaseline_required.insert(item.original_snapshot_id.clone());
            }
            LegacySourceContinuityStatusV1::ContentMismatch => {
                content_mismatches.insert(item.original_snapshot_id.clone());
            }
        }
    }

    if exact_content_matches != continuity.exact_content_matches
        || rebaseline_required != continuity.rebaseline_required
        || content_mismatches != continuity.content_mismatches
    {
        return Err(LegacyRebaselinePlanErrorV1::InconsistentContinuityAssessment(
            "summary sets/counts do not match continuity items".into(),
        ));
    }
    let expected_safe = rebaseline_required.is_empty() && content_mismatches.is_empty();
    if continuity.qualification_safe != expected_safe {
        return Err(LegacyRebaselinePlanErrorV1::InconsistentContinuityAssessment(
            "qualification_safe does not match continuity blockers".into(),
        ));
    }
    Ok(())
}

fn rebaseline_plan_binding(
    manifest_generation: u64,
    manifest_blake3: &str,
    continuity: &LegacySourceContinuityAssessmentV1,
    rebaseline_items: &[LegacyRebaselineWorkItemV1],
    historical_selection_repair_items: &[LegacyRebaselineWorkItemV1],
    claims_requiring_replacement: &BTreeSet<TechnicalClaimIdV1>,
    procedures_requiring_replacement: &BTreeSet<String>,
) -> Result<String, LegacyRebaselinePlanErrorV1> {
    #[derive(Serialize)]
    struct Binding<'a> {
        manifest_generation: u64,
        manifest_blake3: &'a str,
        continuity: &'a LegacySourceContinuityAssessmentV1,
        rebaseline_items: &'a [LegacyRebaselineWorkItemV1],
        historical_selection_repair_items: &'a [LegacyRebaselineWorkItemV1],
        claims_requiring_replacement: &'a BTreeSet<TechnicalClaimIdV1>,
        procedures_requiring_replacement: &'a BTreeSet<String>,
    }
    let encoded = serde_json::to_vec(&Binding {
        manifest_generation,
        manifest_blake3,
        continuity,
        rebaseline_items,
        historical_selection_repair_items,
        claims_requiring_replacement,
        procedures_requiring_replacement,
    })
    .map_err(|err| LegacyRebaselinePlanErrorV1::Serialization(err.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    frame(
        &mut hasher,
        b"schema",
        LEGACY_REBASELINE_PLAN_SCHEMA_V1.as_bytes(),
    );
    frame(&mut hasher, b"work_package", &encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

fn frame(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

#[derive(Debug)]
pub enum LegacyRebaselinePlanErrorV1 {
    Manifest(LegacyQualificationManifestErrorV1),
    UnsupportedContinuitySchema(String),
    InconsistentContinuityAssessment(String),
    DuplicateContinuitySnapshot(SourceSnapshotIdV1),
    UnknownSnapshot(SourceSnapshotIdV1),
    ContinuitySnapshotNotActive(SourceSnapshotIdV1),
    Serialization(String),
}

impl fmt::Display for LegacyRebaselinePlanErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(err) => write!(f, "legacy rebaseline manifest error: {err}"),
            Self::UnsupportedContinuitySchema(value) => {
                write!(f, "unsupported continuity schema {value}")
            }
            Self::InconsistentContinuityAssessment(value) => {
                write!(f, "inconsistent continuity assessment: {value}")
            }
            Self::DuplicateContinuitySnapshot(id) => {
                write!(f, "duplicate continuity snapshot {}", id.0)
            }
            Self::UnknownSnapshot(id) => write!(f, "unknown continuity snapshot {}", id.0),
            Self::ContinuitySnapshotNotActive(id) => write!(
                f,
                "continuity snapshot {} is not referenced by the active qualification manifest",
                id.0
            ),
            Self::Serialization(value) => {
                write!(f, "legacy rebaseline plan serialization failed: {value}")
            }
        }
    }
}

impl Error for LegacyRebaselinePlanErrorV1 {}

impl From<LegacyQualificationManifestErrorV1> for LegacyRebaselinePlanErrorV1 {
    fn from(value: LegacyQualificationManifestErrorV1) -> Self {
        Self::Manifest(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legacy_qualification_profile_v3::initial_legacy_qualification_manifest_v1;
    use crate::build_legacy_five_platform_portfolio_v1;

    fn one_item_assessment(
        snapshot_id: SourceSnapshotIdV1,
        qualifying_id: SourceSnapshotIdV1,
        status: LegacySourceContinuityStatusV1,
    ) -> LegacySourceContinuityAssessmentV1 {
        let mut rebaseline_required = BTreeSet::new();
        let mut content_mismatches = BTreeSet::new();
        let exact_content_matches = match status {
            LegacySourceContinuityStatusV1::ExactContentMatch => 1,
            LegacySourceContinuityStatusV1::RebaselineRequired => {
                rebaseline_required.insert(snapshot_id.clone());
                0
            }
            LegacySourceContinuityStatusV1::ContentMismatch => {
                content_mismatches.insert(snapshot_id.clone());
                0
            }
        };
        LegacySourceContinuityAssessmentV1 {
            schema_version: LEGACY_SOURCE_CONTINUITY_SCHEMA_V1.into(),
            total_selections: 1,
            exact_content_matches,
            rebaseline_required,
            content_mismatches,
            qualification_safe: status == LegacySourceContinuityStatusV1::ExactContentMatch,
            items: vec![LegacySourceContinuityItemV1 {
                original_snapshot_id: snapshot_id,
                qualifying_snapshot_id: qualifying_id,
                status,
            }],
        }
    }

    fn first_active_snapshot(
        pack: &LegacyComputingPackV1,
        manifest: &LegacyQualificationManifestV1,
    ) -> SourceSnapshotIdV1 {
        let claim_id = manifest.claim_ids.iter().next().unwrap();
        pack.sources.claim(claim_id).unwrap().source_snapshot.clone()
    }

    #[test]
    fn metadata_only_continuity_failure_requires_successor_generation() {
        let (pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        let snapshot_id = first_active_snapshot(&pack, &manifest);
        let continuity = one_item_assessment(
            snapshot_id.clone(),
            SourceSnapshotIdV1(format!("{}:current-capture", snapshot_id.0)),
            LegacySourceContinuityStatusV1::RebaselineRequired,
        );
        let plan = plan_legacy_rebaseline_work_v1(&pack, &manifest, &continuity).unwrap();
        assert!(plan.qualification_blocked);
        assert!(plan.requires_successor_generation);
        assert_eq!(plan.rebaseline_items.len(), 1);
        assert!(plan.historical_selection_repair_items.is_empty());
        assert!(plan
            .claims_requiring_replacement
            .iter()
            .any(|claim_id| pack.sources.claim(claim_id).unwrap().source_snapshot == snapshot_id));
        assert_eq!(
            plan.rebaseline_items[0].ordered_stages.first(),
            Some(&LegacyRebaselineStageV1::CaptureCurrentSourceBytes)
        );
        assert_eq!(
            plan.rebaseline_items[0].ordered_stages.last(),
            Some(&LegacyRebaselineStageV1::AssessSuccessorQualificationGeneration)
        );
    }

    #[test]
    fn content_mismatch_repairs_evidence_without_rewriting_knowledge() {
        let (pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        let snapshot_id = first_active_snapshot(&pack, &manifest);
        let continuity = one_item_assessment(
            snapshot_id.clone(),
            SourceSnapshotIdV1(format!("{}:wrong-content", snapshot_id.0)),
            LegacySourceContinuityStatusV1::ContentMismatch,
        );
        let plan = plan_legacy_rebaseline_work_v1(&pack, &manifest, &continuity).unwrap();
        assert!(plan.qualification_blocked);
        assert!(!plan.requires_successor_generation);
        assert!(plan.rebaseline_items.is_empty());
        assert_eq!(plan.historical_selection_repair_items.len(), 1);
        assert!(plan.claims_requiring_replacement.is_empty());
        assert!(plan.procedures_requiring_replacement.is_empty());
        assert_eq!(
            plan.historical_selection_repair_items[0].ordered_stages,
            vec![
                LegacyRebaselineStageV1::LocateExactHistoricalArtifact,
                LegacyRebaselineStageV1::RepairQualificationSourceSelection,
                LegacyRebaselineStageV1::RevalidateCurrentQualificationGeneration,
            ]
        );
    }

    #[test]
    fn exact_continuity_needs_no_recovery_work() {
        let (pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        let snapshot_id = first_active_snapshot(&pack, &manifest);
        let continuity = one_item_assessment(
            snapshot_id.clone(),
            snapshot_id,
            LegacySourceContinuityStatusV1::ExactContentMatch,
        );
        let plan = plan_legacy_rebaseline_work_v1(&pack, &manifest, &continuity).unwrap();
        assert!(!plan.qualification_blocked);
        assert!(!plan.requires_successor_generation);
        assert!(plan.rebaseline_items.is_empty());
        assert!(plan.historical_selection_repair_items.is_empty());
        assert_eq!(plan.exact_content_matches, 1);
    }

    #[test]
    fn inconsistent_assessment_fails_closed() {
        let (pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        let snapshot_id = first_active_snapshot(&pack, &manifest);
        let mut continuity = one_item_assessment(
            snapshot_id.clone(),
            SourceSnapshotIdV1(format!("{}:current-capture", snapshot_id.0)),
            LegacySourceContinuityStatusV1::RebaselineRequired,
        );
        continuity.rebaseline_required.clear();
        assert!(matches!(
            plan_legacy_rebaseline_work_v1(&pack, &manifest, &continuity),
            Err(LegacyRebaselinePlanErrorV1::InconsistentContinuityAssessment(_))
        ));
    }
}
