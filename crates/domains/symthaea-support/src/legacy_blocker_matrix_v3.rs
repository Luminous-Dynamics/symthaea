// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Machine-readable blocker matrix for snapshot-scoped V3 legacy qualification.
//!
//! V3 keeps the frozen 5×10 competency target while exposing the three source
//! provenance dimensions independently: source-revision selection, technical-
//! claim verification, and advisory-procedure verification.

use crate::it_qualification::ItQualificationMatrixV1;
use crate::legacy_computing::{
    LegacyComputingPackV1, LegacyCoverageStateV1, LegacyKnowledgeAreaV1, LegacyPlatformV1,
};
use crate::legacy_qualification_profile::{
    exhaustive_legacy_qualification_profile_v1, LegacyQualificationBlockerV1,
};
use crate::legacy_qualification_profile_v3::{
    assess_legacy_qualification_profile_v3, LegacyQualificationProfileAssessmentV3,
    LegacyQualificationProfileErrorV3,
};
use crate::legacy_qualification_source_ledger_v3::LegacyQualificationSourceLedgerV3;
use crate::legacy_source_artifacts::LegacySourceArtifactLedgerV1;
use crate::standards_registry::{SourceSnapshotIdV1, TechnicalClaimIdV1};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_BLOCKER_MATRIX_SCHEMA_V3: &str =
    "symthaea-it-legacy-qualification-blocker-matrix-v3";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacyBlockerClassV3 {
    SourceRevisionEvidence,
    ClaimVerification,
    ProcedureVerification,
    KnowledgeCoverage,
    ScenarioCoverage,
    CompetencyLevel,
    EvidenceClass,
    AdversarialCoverage,
    HighStakesCoverage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyBlockerCellV3 {
    pub platform: LegacyPlatformV1,
    pub area: LegacyKnowledgeAreaV1,
    pub knowledge_state: LegacyCoverageStateV1,
    pub matching_active_cases: usize,
    pub blocker_classes: BTreeSet<LegacyBlockerClassV3>,
    /// Exact competency blockers from the frozen V1 target. Source provenance
    /// is represented separately by the three V3 blocker classes.
    pub non_source_blockers: BTreeSet<LegacyQualificationBlockerV1>,
}

impl LegacyBlockerCellV3 {
    pub fn ready_for_evaluation(&self) -> bool {
        self.blocker_classes.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyBlockerMatrixV3 {
    pub schema_version: String,
    pub total_cells: usize,
    pub ready_cells: usize,
    pub source_evidence_ready: bool,

    pub required_source_revisions: usize,
    pub selected_source_revisions: usize,
    pub missing_source_selections: BTreeSet<SourceSnapshotIdV1>,

    pub required_claims: usize,
    pub verified_claims: usize,
    pub missing_claim_receipts: BTreeSet<TechnicalClaimIdV1>,

    pub required_procedures: usize,
    pub verified_procedures: usize,
    pub missing_procedure_receipts: BTreeSet<String>,

    pub blocked_cells_by_class: BTreeMap<LegacyBlockerClassV3, usize>,
    pub cells: Vec<LegacyBlockerCellV3>,
}

impl LegacyBlockerMatrixV3 {
    pub fn cell(
        &self,
        platform: LegacyPlatformV1,
        area: LegacyKnowledgeAreaV1,
    ) -> Option<&LegacyBlockerCellV3> {
        self.cells
            .iter()
            .find(|cell| cell.platform == platform && cell.area == area)
    }
}

pub fn legacy_blocker_matrix_v3(
    pack: &LegacyComputingPackV1,
    matrix: &ItQualificationMatrixV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    sources: &LegacyQualificationSourceLedgerV3,
) -> Result<LegacyBlockerMatrixV3, LegacyBlockerMatrixErrorV3> {
    let profile = exhaustive_legacy_qualification_profile_v1();
    let assessment =
        assess_legacy_qualification_profile_v3(pack, &profile, matrix, artifacts, sources)?;
    Ok(blocker_matrix_from_assessment_v3(&assessment))
}

pub fn blocker_matrix_from_assessment_v3(
    assessment: &LegacyQualificationProfileAssessmentV3,
) -> LegacyBlockerMatrixV3 {
    let source_revision_blocked = !assessment.source_readiness.missing_source_selections.is_empty();
    let claim_blocked = !assessment.source_readiness.missing_claim_receipts.is_empty();
    let procedure_blocked = !assessment.source_readiness.missing_procedure_receipts.is_empty();

    let mut cells = Vec::with_capacity(assessment.requirements.len());
    let mut blocked_cells_by_class = BTreeMap::new();

    for requirement in &assessment.requirements {
        let mut blocker_classes: BTreeSet<_> = requirement
            .non_source_blockers
            .iter()
            .map(classify_non_source_blocker)
            .collect();
        if source_revision_blocked {
            blocker_classes.insert(LegacyBlockerClassV3::SourceRevisionEvidence);
        }
        if claim_blocked {
            blocker_classes.insert(LegacyBlockerClassV3::ClaimVerification);
        }
        if procedure_blocked {
            blocker_classes.insert(LegacyBlockerClassV3::ProcedureVerification);
        }
        for blocker_class in &blocker_classes {
            *blocked_cells_by_class.entry(*blocker_class).or_insert(0) += 1;
        }
        cells.push(LegacyBlockerCellV3 {
            platform: requirement.platform,
            area: requirement.area,
            knowledge_state: requirement.knowledge_state,
            matching_active_cases: requirement.matching_active_cases,
            blocker_classes,
            non_source_blockers: requirement.non_source_blockers.clone(),
        });
    }

    cells.sort_by_key(|cell| (cell.platform, cell.area));
    LegacyBlockerMatrixV3 {
        schema_version: LEGACY_BLOCKER_MATRIX_SCHEMA_V3.into(),
        total_cells: assessment.total_requirements,
        ready_cells: assessment.ready_requirements,
        source_evidence_ready: assessment.source_readiness.source_evidence_ready,
        required_source_revisions: assessment.source_readiness.required_source_revisions,
        selected_source_revisions: assessment.source_readiness.selected_source_revisions,
        missing_source_selections: assessment.source_readiness.missing_source_selections.clone(),
        required_claims: assessment.source_readiness.required_claims,
        verified_claims: assessment.source_readiness.verified_claims,
        missing_claim_receipts: assessment.source_readiness.missing_claim_receipts.clone(),
        required_procedures: assessment.source_readiness.required_procedures,
        verified_procedures: assessment.source_readiness.verified_procedures,
        missing_procedure_receipts: assessment.source_readiness.missing_procedure_receipts.clone(),
        blocked_cells_by_class,
        cells,
    }
}

fn classify_non_source_blocker(blocker: &LegacyQualificationBlockerV1) -> LegacyBlockerClassV3 {
    match blocker {
        LegacyQualificationBlockerV1::SourceNotContentDigestBound => {
            // Profile V3 removes this before projection. Keep this exhaustive arm
            // fail-closed in case an invalid hand-built assessment is supplied.
            LegacyBlockerClassV3::SourceRevisionEvidence
        }
        LegacyQualificationBlockerV1::KnowledgeAreaUnmapped => {
            LegacyBlockerClassV3::KnowledgeCoverage
        }
        LegacyQualificationBlockerV1::InsufficientCases { .. } => {
            LegacyBlockerClassV3::ScenarioCoverage
        }
        LegacyQualificationBlockerV1::MinimumLevelNotMet { .. } => {
            LegacyBlockerClassV3::CompetencyLevel
        }
        LegacyQualificationBlockerV1::MissingEvidenceClass(_) => {
            LegacyBlockerClassV3::EvidenceClass
        }
        LegacyQualificationBlockerV1::MissingAdversarialCondition(_) => {
            LegacyBlockerClassV3::AdversarialCoverage
        }
        LegacyQualificationBlockerV1::MissingHighStakesCase => {
            LegacyBlockerClassV3::HighStakesCoverage
        }
    }
}

#[derive(Debug)]
pub enum LegacyBlockerMatrixErrorV3 {
    Qualification(LegacyQualificationProfileErrorV3),
}

impl fmt::Display for LegacyBlockerMatrixErrorV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "legacy V3 blocker-matrix error: {err}"),
        }
    }
}

impl Error for LegacyBlockerMatrixErrorV3 {}

impl From<LegacyQualificationProfileErrorV3> for LegacyBlockerMatrixErrorV3 {
    fn from(value: LegacyQualificationProfileErrorV3) -> Self {
        Self::Qualification(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legacy_qualification_profile_v3::{
        LegacyQualificationRequirementAssessmentV3,
        LEGACY_QUALIFICATION_PROFILE_ASSESSMENT_SCHEMA_V3,
    };
    use crate::legacy_qualification_source_ledger_v3::LegacyQualificationSourceReadinessV3;
    use crate::build_legacy_five_platform_portfolio_v1;

    #[test]
    fn empty_v3_evidence_exposes_three_distinct_source_blocker_classes() {
        let (pack, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let sources = LegacyQualificationSourceLedgerV3::new();
        let blockers = legacy_blocker_matrix_v3(&pack, &matrix, &artifacts, &sources).unwrap();

        assert_eq!(blockers.schema_version, LEGACY_BLOCKER_MATRIX_SCHEMA_V3);
        assert_eq!(blockers.total_cells, 50);
        assert_eq!(blockers.ready_cells, 0);
        assert!(blockers.required_source_revisions > 0);
        assert!(blockers.required_claims > 0);
        assert!(blockers.required_procedures > 0);
        for class in [
            LegacyBlockerClassV3::SourceRevisionEvidence,
            LegacyBlockerClassV3::ClaimVerification,
            LegacyBlockerClassV3::ProcedureVerification,
        ] {
            assert_eq!(blockers.blocked_cells_by_class.get(&class), Some(&50));
            assert!(blockers
                .cells
                .iter()
                .all(|cell| cell.blocker_classes.contains(&class)));
        }
    }

    #[test]
    fn matrix_preserves_non_source_gap_structure() {
        let (pack, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let sources = LegacyQualificationSourceLedgerV3::new();
        let blockers = legacy_blocker_matrix_v3(&pack, &matrix, &artifacts, &sources).unwrap();

        assert!(blockers
            .blocked_cells_by_class
            .get(&LegacyBlockerClassV3::ScenarioCoverage)
            .copied()
            .unwrap_or_default()
            > 0);
        assert!(blockers
            .blocked_cells_by_class
            .get(&LegacyBlockerClassV3::CompetencyLevel)
            .copied()
            .unwrap_or_default()
            > 0);
        assert!(blockers.cells.iter().any(|cell| cell.blocker_classes.len() > 3));
    }

    #[test]
    fn source_dimensions_progress_independently() {
        let assessment = LegacyQualificationProfileAssessmentV3 {
            schema_version: LEGACY_QUALIFICATION_PROFILE_ASSESSMENT_SCHEMA_V3.into(),
            total_requirements: 1,
            ready_requirements: 0,
            source_readiness: LegacyQualificationSourceReadinessV3 {
                required_source_revisions: 2,
                selected_source_revisions: 2,
                required_claims: 2,
                verified_claims: 1,
                required_procedures: 1,
                verified_procedures: 0,
                missing_source_selections: BTreeSet::new(),
                missing_claim_receipts: BTreeSet::from([TechnicalClaimIdV1("claim:missing".into())]),
                missing_procedure_receipts: BTreeSet::from(["procedure:missing".into()]),
                source_evidence_ready: false,
            },
            requirements: vec![LegacyQualificationRequirementAssessmentV3 {
                platform: LegacyPlatformV1::Aix,
                area: LegacyKnowledgeAreaV1::Networking,
                knowledge_state: LegacyCoverageStateV1::ProcedureSeeded,
                matching_active_cases: 3,
                non_source_blockers: BTreeSet::new(),
                source_evidence_ready: false,
            }],
        };

        let blockers = blocker_matrix_from_assessment_v3(&assessment);
        let cell = blockers
            .cell(LegacyPlatformV1::Aix, LegacyKnowledgeAreaV1::Networking)
            .unwrap();
        assert!(!cell
            .blocker_classes
            .contains(&LegacyBlockerClassV3::SourceRevisionEvidence));
        assert!(cell
            .blocker_classes
            .contains(&LegacyBlockerClassV3::ClaimVerification));
        assert!(cell
            .blocker_classes
            .contains(&LegacyBlockerClassV3::ProcedureVerification));
    }

    #[test]
    fn source_ready_projection_does_not_reintroduce_provenance_blockers() {
        let assessment = LegacyQualificationProfileAssessmentV3 {
            schema_version: LEGACY_QUALIFICATION_PROFILE_ASSESSMENT_SCHEMA_V3.into(),
            total_requirements: 1,
            ready_requirements: 0,
            source_readiness: LegacyQualificationSourceReadinessV3 {
                required_source_revisions: 2,
                selected_source_revisions: 2,
                required_claims: 2,
                verified_claims: 2,
                required_procedures: 1,
                verified_procedures: 1,
                missing_source_selections: BTreeSet::new(),
                missing_claim_receipts: BTreeSet::new(),
                missing_procedure_receipts: BTreeSet::new(),
                source_evidence_ready: true,
            },
            requirements: vec![LegacyQualificationRequirementAssessmentV3 {
                platform: LegacyPlatformV1::Aix,
                area: LegacyKnowledgeAreaV1::Networking,
                knowledge_state: LegacyCoverageStateV1::ClaimSeeded,
                matching_active_cases: 1,
                non_source_blockers: BTreeSet::from([
                    LegacyQualificationBlockerV1::InsufficientCases {
                        required: 2,
                        observed: 1,
                    },
                ]),
                source_evidence_ready: true,
            }],
        };

        let blockers = blocker_matrix_from_assessment_v3(&assessment);
        assert!(blockers.source_evidence_ready);
        assert!(!blockers
            .blocked_cells_by_class
            .contains_key(&LegacyBlockerClassV3::SourceRevisionEvidence));
        assert!(!blockers
            .blocked_cells_by_class
            .contains_key(&LegacyBlockerClassV3::ClaimVerification));
        assert!(!blockers
            .blocked_cells_by_class
            .contains_key(&LegacyBlockerClassV3::ProcedureVerification));
        assert_eq!(
            blockers
                .blocked_cells_by_class
                .get(&LegacyBlockerClassV3::ScenarioCoverage),
            Some(&1)
        );
    }
}
