// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Machine-readable blocker matrix for the 5×10 legacy IT qualification target.
//!
//! This module turns a coarse `ready_requirements` count into an exact per-cell
//! explanation of what still prevents evaluation readiness. It does not weaken
//! any qualification requirement and does not infer competence from repository
//! coverage or scenario registration.

use crate::it_qualification::ItQualificationMatrixV1;
use crate::legacy_computing::{
    LegacyComputingPackV1, LegacyCoverageStateV1, LegacyKnowledgeAreaV1, LegacyPlatformV1,
};
use crate::legacy_qualification_profile::{
    assess_legacy_qualification_profile_v1, exhaustive_legacy_qualification_profile_v1,
    LegacyQualificationBlockerV1, LegacyQualificationProfileErrorV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_BLOCKER_MATRIX_SCHEMA_V1: &str = "symthaea-it-legacy-blocker-matrix-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LegacyBlockerClassV1 {
    SourceProvenance,
    KnowledgeCoverage,
    ScenarioCoverage,
    CompetencyLevel,
    EvidenceClass,
    AdversarialCoverage,
    HighStakesCoverage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyBlockerCellV1 {
    pub platform: LegacyPlatformV1,
    pub area: LegacyKnowledgeAreaV1,
    pub knowledge_state: LegacyCoverageStateV1,
    pub matching_active_cases: usize,
    pub blockers: BTreeSet<LegacyQualificationBlockerV1>,
    pub blocker_classes: BTreeSet<LegacyBlockerClassV1>,
}

impl LegacyBlockerCellV1 {
    pub fn ready_for_evaluation(&self) -> bool {
        self.blockers.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyBlockerMatrixV1 {
    pub schema_version: String,
    pub total_cells: usize,
    pub ready_cells: usize,
    pub blocked_cells: usize,
    pub blocker_class_counts: BTreeMap<LegacyBlockerClassV1, usize>,
    pub cells: Vec<LegacyBlockerCellV1>,
}

impl LegacyBlockerMatrixV1 {
    pub fn cell(
        &self,
        platform: LegacyPlatformV1,
        area: LegacyKnowledgeAreaV1,
    ) -> Option<&LegacyBlockerCellV1> {
        self.cells
            .iter()
            .find(|cell| cell.platform == platform && cell.area == area)
    }

    pub fn cells_for_platform(&self, platform: LegacyPlatformV1) -> Vec<&LegacyBlockerCellV1> {
        self.cells
            .iter()
            .filter(|cell| cell.platform == platform)
            .collect()
    }
}

pub fn legacy_blocker_matrix_v1(
    pack: &LegacyComputingPackV1,
    matrix: &ItQualificationMatrixV1,
) -> Result<LegacyBlockerMatrixV1, LegacyBlockerMatrixErrorV1> {
    let profile = exhaustive_legacy_qualification_profile_v1();
    let assessment = assess_legacy_qualification_profile_v1(pack, &profile, matrix)?;

    let mut cells = Vec::with_capacity(assessment.requirements.len());
    let mut blocker_class_counts = BTreeMap::new();
    for requirement in assessment.requirements {
        let blocker_classes: BTreeSet<_> = requirement
            .blockers
            .iter()
            .map(blocker_class)
            .collect();
        for class in &blocker_classes {
            *blocker_class_counts.entry(*class).or_insert(0) += 1;
        }
        cells.push(LegacyBlockerCellV1 {
            platform: requirement.platform,
            area: requirement.area,
            knowledge_state: requirement.knowledge_state,
            matching_active_cases: requirement.matching_active_cases,
            blockers: requirement.blockers,
            blocker_classes,
        });
    }

    cells.sort_by_key(|cell| (cell.platform, cell.area));
    let ready_cells = cells.iter().filter(|cell| cell.ready_for_evaluation()).count();
    let total_cells = cells.len();

    Ok(LegacyBlockerMatrixV1 {
        schema_version: LEGACY_BLOCKER_MATRIX_SCHEMA_V1.into(),
        total_cells,
        ready_cells,
        blocked_cells: total_cells.saturating_sub(ready_cells),
        blocker_class_counts,
        cells,
    })
}

fn blocker_class(blocker: &LegacyQualificationBlockerV1) -> LegacyBlockerClassV1 {
    match blocker {
        LegacyQualificationBlockerV1::SourceNotContentDigestBound => {
            LegacyBlockerClassV1::SourceProvenance
        }
        LegacyQualificationBlockerV1::KnowledgeAreaUnmapped => {
            LegacyBlockerClassV1::KnowledgeCoverage
        }
        LegacyQualificationBlockerV1::InsufficientCases { .. } => {
            LegacyBlockerClassV1::ScenarioCoverage
        }
        LegacyQualificationBlockerV1::MinimumLevelNotMet { .. } => {
            LegacyBlockerClassV1::CompetencyLevel
        }
        LegacyQualificationBlockerV1::MissingEvidenceClass(_) => {
            LegacyBlockerClassV1::EvidenceClass
        }
        LegacyQualificationBlockerV1::MissingAdversarialCondition(_) => {
            LegacyBlockerClassV1::AdversarialCoverage
        }
        LegacyQualificationBlockerV1::MissingHighStakesCase => {
            LegacyBlockerClassV1::HighStakesCoverage
        }
    }
}

#[derive(Debug)]
pub enum LegacyBlockerMatrixErrorV1 {
    Qualification(LegacyQualificationProfileErrorV1),
}

impl fmt::Display for LegacyBlockerMatrixErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "legacy blocker-matrix qualification error: {err}"),
        }
    }
}

impl Error for LegacyBlockerMatrixErrorV1 {}

impl From<LegacyQualificationProfileErrorV1> for LegacyBlockerMatrixErrorV1 {
    fn from(value: LegacyQualificationProfileErrorV1) -> Self {
        Self::Qualification(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_legacy_five_platform_portfolio_v1;

    fn portfolio() -> (LegacyComputingPackV1, ItQualificationMatrixV1) {
        let (pack, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        (pack, matrix)
    }

    #[test]
    fn matrix_contains_exactly_fifty_platform_area_cells() {
        let (pack, qualification) = portfolio();
        let blockers = legacy_blocker_matrix_v1(&pack, &qualification).unwrap();
        assert_eq!(blockers.total_cells, 50);
        assert_eq!(blockers.ready_cells, 0);
        assert_eq!(blockers.blocked_cells, 50);
        for platform in LegacyPlatformV1::ALL {
            assert_eq!(blockers.cells_for_platform(platform).len(), 10);
        }
    }

    #[test]
    fn metadata_only_sources_block_every_cell_without_hiding_other_gaps() {
        let (pack, qualification) = portfolio();
        let blockers = legacy_blocker_matrix_v1(&pack, &qualification).unwrap();
        assert_eq!(
            blockers
                .blocker_class_counts
                .get(&LegacyBlockerClassV1::SourceProvenance),
            Some(&50)
        );
        assert!(blockers.cells.iter().all(|cell| {
            cell.blocker_classes
                .contains(&LegacyBlockerClassV1::SourceProvenance)
        }));
        assert!(blockers.cells.iter().any(|cell| {
            cell.blocker_classes
                .contains(&LegacyBlockerClassV1::ScenarioCoverage)
        }));
    }

    #[test]
    fn blocker_matrix_preserves_existing_case_progress() {
        let (pack, qualification) = portfolio();
        let blockers = legacy_blocker_matrix_v1(&pack, &qualification).unwrap();
        let zos_network = blockers
            .cell(LegacyPlatformV1::Zos, LegacyKnowledgeAreaV1::Networking)
            .unwrap();
        assert!(zos_network.matching_active_cases >= 2);
        assert!(zos_network.knowledge_state >= LegacyCoverageStateV1::ClaimSeeded);

        let hpux_network = blockers
            .cell(LegacyPlatformV1::HpUx, LegacyKnowledgeAreaV1::Networking)
            .unwrap();
        assert_eq!(hpux_network.matching_active_cases, 0);
        assert!(hpux_network
            .blocker_classes
            .contains(&LegacyBlockerClassV1::ScenarioCoverage));
    }

    #[test]
    fn every_raw_blocker_maps_to_an_explicit_planning_class() {
        let (pack, qualification) = portfolio();
        let blockers = legacy_blocker_matrix_v1(&pack, &qualification).unwrap();
        for cell in &blockers.cells {
            for blocker in &cell.blockers {
                assert!(cell.blocker_classes.contains(&blocker_class(blocker)));
            }
        }
    }
}
