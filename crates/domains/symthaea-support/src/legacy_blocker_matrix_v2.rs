// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Machine-readable 50-cell blocker matrix for the V2 legacy qualification gate.
//!
//! This layer turns the conservative V2 assessment into a planning artifact
//! without changing any qualification requirement or blocker semantics.

use crate::it_qualification::ItQualificationMatrixV1;
use crate::legacy_computing::{
    LegacyComputingPackV1, LegacyCoverageStateV1, LegacyKnowledgeAreaV1, LegacyPlatformV1,
};
use crate::legacy_qualification_profile::exhaustive_legacy_qualification_profile_v1;
use crate::legacy_qualification_profile_v2::{
    assess_legacy_qualification_profile_v2, LegacyQualificationBlockerV2,
    LegacyQualificationProfileAssessmentV2, LegacyQualificationProfileErrorV2,
};
use crate::legacy_qualification_source_ledger_v2::LegacyQualificationSourceLedgerV2;
use crate::legacy_source_artifacts::LegacySourceArtifactLedgerV1;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_BLOCKER_MATRIX_SCHEMA_V2: &str =
    "symthaea-it-legacy-qualification-blocker-matrix-v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacyBlockerClassV2 {
    SourceEvidence,
    KnowledgeCoverage,
    ScenarioCoverage,
    CompetencyLevel,
    EvidenceClass,
    AdversarialCoverage,
    HighStakesCoverage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyBlockerCellV2 {
    pub platform: LegacyPlatformV1,
    pub area: LegacyKnowledgeAreaV1,
    pub knowledge_state: LegacyCoverageStateV1,
    pub matching_active_cases: usize,
    pub blocker_classes: BTreeSet<LegacyBlockerClassV2>,
    pub blockers: BTreeSet<LegacyQualificationBlockerV2>,
}

impl LegacyBlockerCellV2 {
    pub fn ready_for_evaluation(&self) -> bool {
        self.blockers.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyBlockerMatrixV2 {
    pub schema_version: String,
    pub total_cells: usize,
    pub ready_cells: usize,
    pub source_evidence_ready: bool,
    pub source_required_documents: usize,
    pub source_selected_documents: usize,
    pub source_required_claims: usize,
    pub source_verified_claims: usize,
    pub blocked_cells_by_class: BTreeMap<LegacyBlockerClassV2, usize>,
    pub cells: Vec<LegacyBlockerCellV2>,
}

impl LegacyBlockerMatrixV2 {
    pub fn cell(
        &self,
        platform: LegacyPlatformV1,
        area: LegacyKnowledgeAreaV1,
    ) -> Option<&LegacyBlockerCellV2> {
        self.cells
            .iter()
            .find(|cell| cell.platform == platform && cell.area == area)
    }
}

pub fn legacy_blocker_matrix_v2(
    pack: &LegacyComputingPackV1,
    matrix: &ItQualificationMatrixV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    sources: &LegacyQualificationSourceLedgerV2,
) -> Result<LegacyBlockerMatrixV2, LegacyBlockerMatrixErrorV2> {
    let profile = exhaustive_legacy_qualification_profile_v1();
    let assessment =
        assess_legacy_qualification_profile_v2(pack, &profile, matrix, artifacts, sources)?;
    Ok(blocker_matrix_from_assessment_v2(&assessment))
}

pub fn blocker_matrix_from_assessment_v2(
    assessment: &LegacyQualificationProfileAssessmentV2,
) -> LegacyBlockerMatrixV2 {
    let mut cells = Vec::with_capacity(assessment.requirements.len());
    let mut blocked_cells_by_class = BTreeMap::new();

    for requirement in &assessment.requirements {
        let blocker_classes: BTreeSet<_> = requirement
            .blockers
            .iter()
            .map(classify_blocker)
            .collect();
        for blocker_class in &blocker_classes {
            *blocked_cells_by_class.entry(*blocker_class).or_insert(0) += 1;
        }
        cells.push(LegacyBlockerCellV2 {
            platform: requirement.platform,
            area: requirement.area,
            knowledge_state: requirement.knowledge_state,
            matching_active_cases: requirement.matching_active_cases,
            blocker_classes,
            blockers: requirement.blockers.clone(),
        });
    }

    cells.sort_by_key(|cell| (cell.platform, cell.area));
    LegacyBlockerMatrixV2 {
        schema_version: LEGACY_BLOCKER_MATRIX_SCHEMA_V2.into(),
        total_cells: assessment.total_requirements,
        ready_cells: assessment.ready_requirements,
        source_evidence_ready: assessment.source_evidence_ready,
        source_required_documents: assessment.source_required_documents,
        source_selected_documents: assessment.source_selected_documents,
        source_required_claims: assessment.source_required_claims,
        source_verified_claims: assessment.source_verified_claims,
        blocked_cells_by_class,
        cells,
    }
}

fn classify_blocker(blocker: &LegacyQualificationBlockerV2) -> LegacyBlockerClassV2 {
    match blocker {
        LegacyQualificationBlockerV2::SourceEvidenceIncomplete => {
            LegacyBlockerClassV2::SourceEvidence
        }
        LegacyQualificationBlockerV2::KnowledgeAreaUnmapped => {
            LegacyBlockerClassV2::KnowledgeCoverage
        }
        LegacyQualificationBlockerV2::InsufficientCases { .. } => {
            LegacyBlockerClassV2::ScenarioCoverage
        }
        LegacyQualificationBlockerV2::MinimumLevelNotMet { .. } => {
            LegacyBlockerClassV2::CompetencyLevel
        }
        LegacyQualificationBlockerV2::MissingEvidenceClass(_) => {
            LegacyBlockerClassV2::EvidenceClass
        }
        LegacyQualificationBlockerV2::MissingAdversarialCondition(_) => {
            LegacyBlockerClassV2::AdversarialCoverage
        }
        LegacyQualificationBlockerV2::MissingHighStakesCase => {
            LegacyBlockerClassV2::HighStakesCoverage
        }
    }
}

#[derive(Debug)]
pub enum LegacyBlockerMatrixErrorV2 {
    Qualification(LegacyQualificationProfileErrorV2),
}

impl fmt::Display for LegacyBlockerMatrixErrorV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "legacy V2 blocker-matrix error: {err}"),
        }
    }
}

impl Error for LegacyBlockerMatrixErrorV2 {}

impl From<LegacyQualificationProfileErrorV2> for LegacyBlockerMatrixErrorV2 {
    fn from(value: LegacyQualificationProfileErrorV2) -> Self {
        Self::Qualification(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_legacy_five_platform_portfolio_v1;

    #[test]
    fn empty_source_evidence_projects_to_exact_fifty_cell_source_blocker() {
        let (pack, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let sources = LegacyQualificationSourceLedgerV2::new();
        let blockers = legacy_blocker_matrix_v2(&pack, &matrix, &artifacts, &sources).unwrap();
        assert_eq!(blockers.schema_version, LEGACY_BLOCKER_MATRIX_SCHEMA_V2);
        assert_eq!(blockers.total_cells, 50);
        assert_eq!(blockers.ready_cells, 0);
        assert_eq!(
            blockers
                .blocked_cells_by_class
                .get(&LegacyBlockerClassV2::SourceEvidence),
            Some(&50)
        );
        assert!(blockers.cells.iter().all(|cell| {
            cell.blocker_classes
                .contains(&LegacyBlockerClassV2::SourceEvidence)
        }));
    }

    #[test]
    fn matrix_preserves_non_source_gap_structure() {
        let (pack, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let sources = LegacyQualificationSourceLedgerV2::new();
        let blockers = legacy_blocker_matrix_v2(&pack, &matrix, &artifacts, &sources).unwrap();

        assert!(blockers
            .blocked_cells_by_class
            .get(&LegacyBlockerClassV2::ScenarioCoverage)
            .copied()
            .unwrap_or_default()
            > 0);
        assert!(blockers
            .blocked_cells_by_class
            .get(&LegacyBlockerClassV2::CompetencyLevel)
            .copied()
            .unwrap_or_default()
            > 0);
        assert!(blockers.cells.iter().any(|cell| cell.blocker_classes.len() > 1));
    }

    #[test]
    fn pure_projection_does_not_reintroduce_source_blocker_when_assessment_is_source_ready() {
        let assessment = LegacyQualificationProfileAssessmentV2 {
            schema_version: crate::LEGACY_QUALIFICATION_PROFILE_ASSESSMENT_SCHEMA_V2.into(),
            total_requirements: 1,
            ready_requirements: 0,
            source_evidence_ready: true,
            source_required_documents: 3,
            source_selected_documents: 3,
            source_required_claims: 5,
            source_verified_claims: 5,
            requirements: vec![crate::LegacyQualificationRequirementAssessmentV2 {
                platform: LegacyPlatformV1::Aix,
                area: LegacyKnowledgeAreaV1::Networking,
                knowledge_state: LegacyCoverageStateV1::ClaimSeeded,
                matching_active_cases: 1,
                blockers: BTreeSet::from([
                    LegacyQualificationBlockerV2::InsufficientCases {
                        required: 2,
                        observed: 1,
                    },
                    LegacyQualificationBlockerV2::MissingAdversarialCondition(
                        crate::AdversarialConditionV1::StaleTelemetry,
                    ),
                ]),
            }],
        };
        let matrix = blocker_matrix_from_assessment_v2(&assessment);
        assert!(matrix.source_evidence_ready);
        assert!(!matrix
            .blocked_cells_by_class
            .contains_key(&LegacyBlockerClassV2::SourceEvidence));
        let cell = matrix
            .cell(LegacyPlatformV1::Aix, LegacyKnowledgeAreaV1::Networking)
            .unwrap();
        assert!(cell
            .blocker_classes
            .contains(&LegacyBlockerClassV2::ScenarioCoverage));
        assert!(cell
            .blocker_classes
            .contains(&LegacyBlockerClassV2::AdversarialCoverage));
    }
}
