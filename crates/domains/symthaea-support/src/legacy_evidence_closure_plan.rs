// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dependency-aware closure planning for legacy qualification evidence.
//!
//! This module does not weaken the conservative V3 qualification theorem. It
//! projects the exact missing source/claim/procedure IDs into an operational
//! work queue so evidence can be closed in dependency order without pretending
//! that completing provenance establishes competence.
//!
//! ```text
//! source revision capture
//!      -> claim verification
//!      -> procedure verification
//!      != scenario / hardware / competence qualification
//! ```

use crate::it_qualification::ItQualificationMatrixV1;
use crate::legacy_computing::{
    LegacyComputingPackV1, LegacyCoverageStateV1, LegacyKnowledgeAreaV1, LegacyPlatformV1,
};
use crate::legacy_qualification_profile::exhaustive_legacy_qualification_profile_v1;
use crate::legacy_qualification_profile_v3::{
    assess_legacy_qualification_profile_v3, LegacyQualificationProfileAssessmentV3,
    LegacyQualificationProfileErrorV3,
};
use crate::legacy_qualification_source_ledger_v3::LegacyQualificationSourceLedgerV3;
use crate::legacy_source_artifacts::LegacySourceArtifactLedgerV1;
use crate::standards_registry::{SourceSnapshotIdV1, TechnicalClaimIdV1};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const LEGACY_EVIDENCE_CLOSURE_PLAN_SCHEMA_V1: &str =
    "symthaea-it-legacy-evidence-closure-plan-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacyEvidenceClosureStageV1 {
    SourceRevisionSelection,
    ClaimVerification,
    ProcedureVerification,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacyEvidenceClosureTargetV1 {
    SourceRevision(SourceSnapshotIdV1),
    Claim(TechnicalClaimIdV1),
    Procedure(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyEvidenceClosureItemV1 {
    pub stage: LegacyEvidenceClosureStageV1,
    pub target: LegacyEvidenceClosureTargetV1,
    /// Platforms whose represented knowledge depends on this evidence item.
    pub platforms: BTreeSet<LegacyPlatformV1>,
    /// Qualification cells to which this item is relevant. Completing the item
    /// does not imply these cells become ready; independent competency blockers
    /// may remain.
    pub affected_areas: BTreeSet<LegacyKnowledgeAreaV1>,
    pub relevant_cells: usize,
    /// Missing source revisions that must be selected before this review item
    /// can be meaningfully completed. Source-selection items have no entries.
    pub prerequisite_source_revisions: BTreeSet<SourceSnapshotIdV1>,
    pub actionable_now: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyEvidenceClosurePlanV1 {
    pub schema_version: String,
    pub source_evidence_ready: bool,
    pub total_missing_items: usize,
    pub actionable_items: usize,
    pub deferred_items: usize,
    pub remaining_source_revisions: usize,
    pub remaining_claims: usize,
    pub remaining_procedures: usize,
    /// Stable priority order: actionable first, then dependency stage, then
    /// largest relevant-cell fanout, then exact target identity.
    pub items: Vec<LegacyEvidenceClosureItemV1>,
}

impl LegacyEvidenceClosurePlanV1 {
    pub fn actionable(&self) -> impl Iterator<Item = &LegacyEvidenceClosureItemV1> {
        self.items.iter().filter(|item| item.actionable_now)
    }

    pub fn deferred(&self) -> impl Iterator<Item = &LegacyEvidenceClosureItemV1> {
        self.items.iter().filter(|item| !item.actionable_now)
    }
}

pub fn legacy_evidence_closure_plan_v1(
    pack: &LegacyComputingPackV1,
    matrix: &ItQualificationMatrixV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    sources: &LegacyQualificationSourceLedgerV3,
) -> Result<LegacyEvidenceClosurePlanV1, LegacyEvidenceClosurePlanErrorV1> {
    let profile = exhaustive_legacy_qualification_profile_v1();
    let assessment =
        assess_legacy_qualification_profile_v3(pack, &profile, matrix, artifacts, sources)?;
    closure_plan_from_assessment_v1(pack, &assessment)
}

/// Pure projection helper for receipts/evaluators that already possess a
/// validated V3 assessment. Unknown IDs fail closed rather than being silently
/// dropped from the work queue.
pub fn closure_plan_from_assessment_v1(
    pack: &LegacyComputingPackV1,
    assessment: &LegacyQualificationProfileAssessmentV3,
) -> Result<LegacyEvidenceClosurePlanV1, LegacyEvidenceClosurePlanErrorV1> {
    pack.validate()
        .map_err(|err| LegacyEvidenceClosurePlanErrorV1::InvalidPack(err.to_string()))?;

    let missing_sources = &assessment.source_readiness.missing_source_selections;
    let mut items = Vec::with_capacity(
        missing_sources.len()
            + assessment.source_readiness.missing_claim_receipts.len()
            + assessment.source_readiness.missing_procedure_receipts.len(),
    );

    for snapshot_id in missing_sources {
        let platforms = platforms_for_snapshot(pack, snapshot_id);
        if platforms.is_empty() {
            return Err(LegacyEvidenceClosurePlanErrorV1::UnscopedSourceRevision(
                snapshot_id.clone(),
            ));
        }
        let (affected_areas, relevant_cells) = relevant_cells_for_source(
            assessment,
            &platforms,
            LegacyCoverageStateV1::SourceMapped,
        );
        items.push(LegacyEvidenceClosureItemV1 {
            stage: LegacyEvidenceClosureStageV1::SourceRevisionSelection,
            target: LegacyEvidenceClosureTargetV1::SourceRevision(snapshot_id.clone()),
            platforms,
            affected_areas,
            relevant_cells,
            prerequisite_source_revisions: BTreeSet::new(),
            actionable_now: true,
        });
    }

    for claim_id in &assessment.source_readiness.missing_claim_receipts {
        let claim = pack
            .sources
            .claim(claim_id)
            .ok_or_else(|| LegacyEvidenceClosurePlanErrorV1::UnknownClaim(claim_id.clone()))?;
        let platforms = platforms_for_snapshot(pack, &claim.source_snapshot);
        if platforms.is_empty() {
            return Err(LegacyEvidenceClosurePlanErrorV1::UnscopedClaim(
                claim_id.clone(),
            ));
        }
        let (affected_areas, relevant_cells) = relevant_cells_for_source(
            assessment,
            &platforms,
            LegacyCoverageStateV1::ClaimSeeded,
        );
        let prerequisite_source_revisions = missing_sources
            .contains(&claim.source_snapshot)
            .then(|| BTreeSet::from([claim.source_snapshot.clone()]))
            .unwrap_or_default();
        items.push(LegacyEvidenceClosureItemV1 {
            stage: LegacyEvidenceClosureStageV1::ClaimVerification,
            target: LegacyEvidenceClosureTargetV1::Claim(claim_id.clone()),
            platforms,
            affected_areas,
            relevant_cells,
            actionable_now: prerequisite_source_revisions.is_empty(),
            prerequisite_source_revisions,
        });
    }

    for procedure_id in &assessment.source_readiness.missing_procedure_receipts {
        let procedure = pack
            .procedures
            .iter()
            .find(|procedure| &procedure.id == procedure_id)
            .ok_or_else(|| {
                LegacyEvidenceClosurePlanErrorV1::UnknownProcedure(procedure_id.clone())
            })?;
        let prerequisite_source_revisions = procedure
            .source_snapshots
            .intersection(missing_sources)
            .cloned()
            .collect::<BTreeSet<_>>();
        let relevant_cells = assessment
            .requirements
            .iter()
            .filter(|requirement| {
                requirement.platform == procedure.platform && requirement.area == procedure.area
            })
            .count();
        items.push(LegacyEvidenceClosureItemV1 {
            stage: LegacyEvidenceClosureStageV1::ProcedureVerification,
            target: LegacyEvidenceClosureTargetV1::Procedure(procedure_id.clone()),
            platforms: BTreeSet::from([procedure.platform]),
            affected_areas: BTreeSet::from([procedure.area]),
            relevant_cells,
            actionable_now: prerequisite_source_revisions.is_empty(),
            prerequisite_source_revisions,
        });
    }

    items.sort_by(compare_items);
    let actionable_items = items.iter().filter(|item| item.actionable_now).count();
    let total_missing_items = items.len();

    Ok(LegacyEvidenceClosurePlanV1 {
        schema_version: LEGACY_EVIDENCE_CLOSURE_PLAN_SCHEMA_V1.into(),
        source_evidence_ready: assessment.source_readiness.source_evidence_ready,
        total_missing_items,
        actionable_items,
        deferred_items: total_missing_items - actionable_items,
        remaining_source_revisions: assessment.source_readiness.missing_source_selections.len(),
        remaining_claims: assessment.source_readiness.missing_claim_receipts.len(),
        remaining_procedures: assessment.source_readiness.missing_procedure_receipts.len(),
        items,
    })
}

fn relevant_cells_for_source(
    assessment: &LegacyQualificationProfileAssessmentV3,
    platforms: &BTreeSet<LegacyPlatformV1>,
    minimum_state: LegacyCoverageStateV1,
) -> (BTreeSet<LegacyKnowledgeAreaV1>, usize) {
    let mut areas = BTreeSet::new();
    let mut cells = 0usize;
    for requirement in &assessment.requirements {
        if platforms.contains(&requirement.platform) && requirement.knowledge_state >= minimum_state {
            areas.insert(requirement.area);
            cells += 1;
        }
    }
    (areas, cells)
}

fn platforms_for_snapshot(
    pack: &LegacyComputingPackV1,
    snapshot_id: &SourceSnapshotIdV1,
) -> BTreeSet<LegacyPlatformV1> {
    pack.profiles
        .iter()
        .filter(|profile| profile.source_snapshots.contains(snapshot_id))
        .map(|profile| profile.platform)
        .collect()
}

fn compare_items(
    left: &LegacyEvidenceClosureItemV1,
    right: &LegacyEvidenceClosureItemV1,
) -> Ordering {
    right
        .actionable_now
        .cmp(&left.actionable_now)
        .then_with(|| left.stage.cmp(&right.stage))
        .then_with(|| right.relevant_cells.cmp(&left.relevant_cells))
        .then_with(|| target_key(&left.target).cmp(&target_key(&right.target)))
}

fn target_key(target: &LegacyEvidenceClosureTargetV1) -> String {
    match target {
        LegacyEvidenceClosureTargetV1::SourceRevision(id) => format!("source:{}", id.0),
        LegacyEvidenceClosureTargetV1::Claim(id) => format!("claim:{}", id.0),
        LegacyEvidenceClosureTargetV1::Procedure(id) => format!("procedure:{id}"),
    }
}

#[derive(Debug)]
pub enum LegacyEvidenceClosurePlanErrorV1 {
    Qualification(LegacyQualificationProfileErrorV3),
    InvalidPack(String),
    UnscopedSourceRevision(SourceSnapshotIdV1),
    UnknownClaim(TechnicalClaimIdV1),
    UnscopedClaim(TechnicalClaimIdV1),
    UnknownProcedure(String),
}

impl fmt::Display for LegacyEvidenceClosurePlanErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "legacy evidence-closure assessment failed: {err}"),
            Self::InvalidPack(err) => write!(f, "legacy evidence-closure pack is invalid: {err}"),
            Self::UnscopedSourceRevision(id) => write!(
                f,
                "missing source revision {} is not owned by any legacy platform profile",
                id.0
            ),
            Self::UnknownClaim(id) => {
                write!(f, "missing claim {} does not exist in the legacy pack", id.0)
            }
            Self::UnscopedClaim(id) => write!(
                f,
                "missing claim {} is not traceable to a platform-owned source revision",
                id.0
            ),
            Self::UnknownProcedure(id) => {
                write!(f, "missing procedure {id} does not exist in the legacy pack")
            }
        }
    }
}

impl Error for LegacyEvidenceClosurePlanErrorV1 {}

impl From<LegacyQualificationProfileErrorV3> for LegacyEvidenceClosurePlanErrorV1 {
    fn from(value: LegacyQualificationProfileErrorV3) -> Self {
        Self::Qualification(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legacy_qualification_profile_v3::assess_legacy_qualification_profile_v3;
    use crate::build_legacy_five_platform_portfolio_v1;

    fn empty_assessment() -> (
        LegacyComputingPackV1,
        LegacyQualificationProfileAssessmentV3,
    ) {
        let (pack, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let sources = LegacyQualificationSourceLedgerV3::new();
        let profile = exhaustive_legacy_qualification_profile_v1();
        let assessment =
            assess_legacy_qualification_profile_v3(&pack, &profile, &matrix, &artifacts, &sources)
                .unwrap();
        (pack, assessment)
    }

    #[test]
    fn empty_source_ledger_makes_only_source_capture_actionable() {
        let (pack, assessment) = empty_assessment();
        let plan = closure_plan_from_assessment_v1(&pack, &assessment).unwrap();

        assert_eq!(plan.schema_version, LEGACY_EVIDENCE_CLOSURE_PLAN_SCHEMA_V1);
        assert!(plan.remaining_source_revisions > 0);
        assert!(plan.remaining_claims > 0);
        assert!(plan.remaining_procedures > 0);
        assert_eq!(plan.actionable_items, plan.remaining_source_revisions);
        assert!(plan.actionable().all(|item| {
            item.stage == LegacyEvidenceClosureStageV1::SourceRevisionSelection
                && item.prerequisite_source_revisions.is_empty()
        }));
        assert!(plan.deferred().all(|item| {
            matches!(
                item.stage,
                LegacyEvidenceClosureStageV1::ClaimVerification
                    | LegacyEvidenceClosureStageV1::ProcedureVerification
            ) && !item.prerequisite_source_revisions.is_empty()
        }));
    }

    #[test]
    fn selecting_aix_source_unlocks_only_its_dependent_reviews_in_projection() {
        let (pack, mut assessment) = empty_assessment();
        let aix_source = SourceSnapshotIdV1("ibm:aix-os-management@7.3".into());
        assert!(assessment
            .source_readiness
            .missing_source_selections
            .remove(&aix_source));
        assessment.source_readiness.selected_source_revisions += 1;

        let plan = closure_plan_from_assessment_v1(&pack, &assessment).unwrap();
        let aix_claim = plan
            .items
            .iter()
            .find(|item| {
                item.target
                    == LegacyEvidenceClosureTargetV1::Claim(TechnicalClaimIdV1(
                        "legacy:aix:os-management-scope".into(),
                    ))
            })
            .unwrap();
        assert!(aix_claim.actionable_now);
        assert!(aix_claim.prerequisite_source_revisions.is_empty());

        let aix_procedure = plan
            .items
            .iter()
            .find(|item| {
                item.target
                    == LegacyEvidenceClosureTargetV1::Procedure(
                        "legacy:aix:first-response".into(),
                    )
            })
            .unwrap();
        assert!(aix_procedure.actionable_now);

        let solaris_claim = plan
            .items
            .iter()
            .find(|item| {
                item.target
                    == LegacyEvidenceClosureTargetV1::Claim(TechnicalClaimIdV1(
                        "legacy:solaris:admin-library-scope".into(),
                    ))
            })
            .unwrap();
        assert!(!solaris_claim.actionable_now);
        assert!(solaris_claim
            .prerequisite_source_revisions
            .contains(&SourceSnapshotIdV1("oracle:solaris-docs@11.4".into())));
    }

    #[test]
    fn closure_items_are_scoped_and_do_not_claim_global_cell_impact() {
        let (pack, assessment) = empty_assessment();
        let plan = closure_plan_from_assessment_v1(&pack, &assessment).unwrap();

        let aix_source = plan
            .items
            .iter()
            .find(|item| {
                item.target
                    == LegacyEvidenceClosureTargetV1::SourceRevision(SourceSnapshotIdV1(
                        "ibm:aix-os-management@7.3".into(),
                    ))
            })
            .unwrap();
        assert_eq!(aix_source.platforms, BTreeSet::from([LegacyPlatformV1::Aix]));
        assert!(aix_source.relevant_cells > 0);
        assert!(aix_source.relevant_cells < assessment.total_requirements);

        let aix_procedure = plan
            .items
            .iter()
            .find(|item| {
                item.target
                    == LegacyEvidenceClosureTargetV1::Procedure(
                        "legacy:aix:first-response".into(),
                    )
            })
            .unwrap();
        assert_eq!(aix_procedure.relevant_cells, 1);
        assert_eq!(
            aix_procedure.affected_areas,
            BTreeSet::from([LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement])
        );
    }

    #[test]
    fn unknown_missing_claim_fails_closed() {
        let (pack, mut assessment) = empty_assessment();
        assessment
            .source_readiness
            .missing_claim_receipts
            .insert(TechnicalClaimIdV1("legacy:unknown:claim".into()));
        let error = closure_plan_from_assessment_v1(&pack, &assessment).unwrap_err();
        assert!(matches!(
            error,
            LegacyEvidenceClosurePlanErrorV1::UnknownClaim(_)
        ));
    }
}
