// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Legacy qualification profile V3.
//!
//! The competency target remains the existing exhaustive V1 5×10 matrix. V3
//! changes only source-evidence admission: the obsolete global requirement that
//! every historical registry snapshot be content-bound is replaced by V3's exact
//! source-revision + claim + procedure verification readiness.

use crate::it_qualification::ItQualificationMatrixV1;
use crate::legacy_computing::{
    LegacyComputingPackV1, LegacyCoverageStateV1, LegacyKnowledgeAreaV1, LegacyPlatformV1,
};
use crate::legacy_qualification_profile::{
    assess_legacy_qualification_profile_v1, LegacyQualificationBlockerV1,
    LegacyQualificationProfileErrorV1, LegacyQualificationProfileV1,
};
use crate::legacy_qualification_source_ledger_v3::{
    assess_legacy_qualification_source_readiness_v3, LegacyQualificationSourceLedgerErrorV3,
    LegacyQualificationSourceLedgerV3, LegacyQualificationSourceReadinessV3,
};
use crate::legacy_source_artifacts::LegacySourceArtifactLedgerV1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const LEGACY_QUALIFICATION_PROFILE_ASSESSMENT_SCHEMA_V3: &str =
    "symthaea-it-legacy-qualification-profile-assessment-v3";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationRequirementAssessmentV3 {
    pub platform: LegacyPlatformV1,
    pub area: LegacyKnowledgeAreaV1,
    pub knowledge_state: LegacyCoverageStateV1,
    pub matching_active_cases: usize,
    /// All V1 competency blockers except the obsolete global source predicate.
    pub non_source_blockers: BTreeSet<LegacyQualificationBlockerV1>,
    pub source_evidence_ready: bool,
}

impl LegacyQualificationRequirementAssessmentV3 {
    pub fn ready_for_evaluation(&self) -> bool {
        self.source_evidence_ready && self.non_source_blockers.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationProfileAssessmentV3 {
    pub schema_version: String,
    pub total_requirements: usize,
    pub ready_requirements: usize,
    pub source_readiness: LegacyQualificationSourceReadinessV3,
    pub requirements: Vec<LegacyQualificationRequirementAssessmentV3>,
}

pub fn assess_legacy_qualification_profile_v3(
    pack: &LegacyComputingPackV1,
    profile: &LegacyQualificationProfileV1,
    matrix: &ItQualificationMatrixV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    source_ledger: &LegacyQualificationSourceLedgerV3,
) -> Result<LegacyQualificationProfileAssessmentV3, LegacyQualificationProfileErrorV3> {
    let v1 = assess_legacy_qualification_profile_v1(pack, profile, matrix)?;
    let source_readiness =
        assess_legacy_qualification_source_readiness_v3(pack, artifacts, source_ledger)?;

    let requirements = v1
        .requirements
        .into_iter()
        .map(|assessment| {
            let mut non_source_blockers = assessment.blockers;
            non_source_blockers.remove(&LegacyQualificationBlockerV1::SourceNotContentDigestBound);
            LegacyQualificationRequirementAssessmentV3 {
                platform: assessment.platform,
                area: assessment.area,
                knowledge_state: assessment.knowledge_state,
                matching_active_cases: assessment.matching_active_cases,
                non_source_blockers,
                source_evidence_ready: source_readiness.source_evidence_ready,
            }
        })
        .collect::<Vec<_>>();
    let ready_requirements = requirements
        .iter()
        .filter(|assessment| assessment.ready_for_evaluation())
        .count();

    Ok(LegacyQualificationProfileAssessmentV3 {
        schema_version: LEGACY_QUALIFICATION_PROFILE_ASSESSMENT_SCHEMA_V3.into(),
        total_requirements: requirements.len(),
        ready_requirements,
        source_readiness,
        requirements,
    })
}

#[derive(Debug)]
pub enum LegacyQualificationProfileErrorV3 {
    Profile(LegacyQualificationProfileErrorV1),
    Source(LegacyQualificationSourceLedgerErrorV3),
}

impl fmt::Display for LegacyQualificationProfileErrorV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Profile(err) => write!(f, "legacy V3 qualification profile error: {err}"),
            Self::Source(err) => write!(f, "legacy V3 qualification source error: {err}"),
        }
    }
}

impl Error for LegacyQualificationProfileErrorV3 {}

impl From<LegacyQualificationProfileErrorV1> for LegacyQualificationProfileErrorV3 {
    fn from(value: LegacyQualificationProfileErrorV1) -> Self {
        Self::Profile(value)
    }
}

impl From<LegacyQualificationSourceLedgerErrorV3> for LegacyQualificationProfileErrorV3 {
    fn from(value: LegacyQualificationSourceLedgerErrorV3) -> Self {
        Self::Source(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        build_legacy_five_platform_portfolio_v1, exhaustive_legacy_qualification_profile_v1,
    };

    #[test]
    fn empty_v3_source_evidence_blocks_all_cells_without_erasing_other_gaps() {
        let (pack, matrix, _) = build_legacy_five_platform_portfolio_v1(1_800_000_000_000)
            .unwrap();
        let profile = exhaustive_legacy_qualification_profile_v1();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let source_ledger = LegacyQualificationSourceLedgerV3::new();
        let assessment = assess_legacy_qualification_profile_v3(
            &pack,
            &profile,
            &matrix,
            &artifacts,
            &source_ledger,
        )
        .unwrap();

        assert_eq!(assessment.total_requirements, 50);
        assert_eq!(assessment.ready_requirements, 0);
        assert!(!assessment.source_readiness.source_evidence_ready);
        assert!(assessment
            .requirements
            .iter()
            .all(|requirement| !requirement.source_evidence_ready));
        assert!(assessment
            .requirements
            .iter()
            .any(|requirement| !requirement.non_source_blockers.is_empty()));
        assert!(assessment.requirements.iter().all(|requirement| {
            !requirement
                .non_source_blockers
                .contains(&LegacyQualificationBlockerV1::SourceNotContentDigestBound)
        }));
    }

    #[test]
    fn readiness_method_requires_both_source_and_competency_clearance() {
        let assessment = LegacyQualificationRequirementAssessmentV3 {
            platform: LegacyPlatformV1::Aix,
            area: LegacyKnowledgeAreaV1::Networking,
            knowledge_state: LegacyCoverageStateV1::ProcedureSeeded,
            matching_active_cases: 3,
            non_source_blockers: BTreeSet::new(),
            source_evidence_ready: false,
        };
        assert!(!assessment.ready_for_evaluation());
        let ready = LegacyQualificationRequirementAssessmentV3 {
            source_evidence_ready: true,
            ..assessment
        };
        assert!(ready.ready_for_evaluation());
    }
}
