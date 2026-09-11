// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strong legacy qualification admission: case/source coverage plus retained artifacts.
//!
//! The baseline 50-cell assessment remains useful for planning. Certification-grade
//! admission additionally requires the exact content-digest-bound source artifacts to
//! be retained and independently addressable through the artifact ledger.

use crate::it_qualification::ItQualificationMatrixV1;
use crate::legacy_computing::LegacyComputingPackV1;
use crate::legacy_qualification_profile::{
    assess_legacy_qualification_profile_v1, LegacyQualificationProfileAssessmentV1,
    LegacyQualificationProfileErrorV1, LegacyQualificationProfileV1,
};
use crate::legacy_source_artifacts::{
    assess_legacy_source_artifact_readiness_v1, LegacySourceArtifactErrorV1,
    LegacySourceArtifactLedgerV1, LegacySourceArtifactReadinessV1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacyStrongQualificationBlockerV1 {
    BaselineRequirementsIncomplete,
    SourceArtifactsIncomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyStrongQualificationAssessmentV1 {
    pub baseline: LegacyQualificationProfileAssessmentV1,
    pub artifact_readiness: LegacySourceArtifactReadinessV1,
    pub blockers: BTreeSet<LegacyStrongQualificationBlockerV1>,
    /// Number of the baseline's 50 requirements admitted through the stronger
    /// retained-artifact gate. If source artifacts are incomplete this is zero,
    /// even when individual benchmark cells are otherwise ready.
    pub strongly_ready_requirements: usize,
}

impl LegacyStrongQualificationAssessmentV1 {
    pub fn strong_source_admission_ready(&self) -> bool {
        self.blockers.is_empty()
    }
}

pub fn assess_legacy_strong_qualification_v1(
    pack: &LegacyComputingPackV1,
    profile: &LegacyQualificationProfileV1,
    matrix: &ItQualificationMatrixV1,
    artifacts: &LegacySourceArtifactLedgerV1,
) -> Result<LegacyStrongQualificationAssessmentV1, LegacyArtifactQualificationErrorV1> {
    let baseline = assess_legacy_qualification_profile_v1(pack, profile, matrix)?;
    let artifact_readiness = assess_legacy_source_artifact_readiness_v1(pack, artifacts)?;

    let mut blockers = BTreeSet::new();
    if baseline.ready_requirements != baseline.total_requirements {
        blockers.insert(LegacyStrongQualificationBlockerV1::BaselineRequirementsIncomplete);
    }
    if !artifact_readiness.qualification_artifact_ready {
        blockers.insert(LegacyStrongQualificationBlockerV1::SourceArtifactsIncomplete);
    }

    let strongly_ready_requirements = if artifact_readiness.qualification_artifact_ready {
        baseline.ready_requirements
    } else {
        0
    };

    Ok(LegacyStrongQualificationAssessmentV1 {
        baseline,
        artifact_readiness,
        blockers,
        strongly_ready_requirements,
    })
}

#[derive(Debug)]
pub enum LegacyArtifactQualificationErrorV1 {
    Profile(LegacyQualificationProfileErrorV1),
    Artifacts(LegacySourceArtifactErrorV1),
}

impl fmt::Display for LegacyArtifactQualificationErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Profile(err) => write!(f, "legacy qualification profile error: {err}"),
            Self::Artifacts(err) => write!(f, "legacy source artifact error: {err}"),
        }
    }
}

impl Error for LegacyArtifactQualificationErrorV1 {}

impl From<LegacyQualificationProfileErrorV1> for LegacyArtifactQualificationErrorV1 {
    fn from(value: LegacyQualificationProfileErrorV1) -> Self {
        Self::Profile(value)
    }
}

impl From<LegacySourceArtifactErrorV1> for LegacyArtifactQualificationErrorV1 {
    fn from(value: LegacySourceArtifactErrorV1) -> Self {
        Self::Artifacts(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        exhaustive_legacy_qualification_profile_v1, seed_legacy_computing_pack_v1,
        ItQualificationMatrixV1,
    };

    #[test]
    fn current_seed_is_zero_of_fifty_under_strong_admission() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let profile = exhaustive_legacy_qualification_profile_v1();
        let matrix = ItQualificationMatrixV1::new();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let assessment =
            assess_legacy_strong_qualification_v1(&pack, &profile, &matrix, &artifacts).unwrap();

        assert_eq!(assessment.baseline.total_requirements, 50);
        assert_eq!(assessment.baseline.ready_requirements, 0);
        assert_eq!(assessment.strongly_ready_requirements, 0);
        assert!(!assessment.artifact_readiness.qualification_artifact_ready);
        assert!(assessment
            .blockers
            .contains(&LegacyStrongQualificationBlockerV1::BaselineRequirementsIncomplete));
        assert!(assessment
            .blockers
            .contains(&LegacyStrongQualificationBlockerV1::SourceArtifactsIncomplete));
    }

    #[test]
    fn retained_artifact_gate_never_increases_baseline_readiness() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let profile = exhaustive_legacy_qualification_profile_v1();
        let matrix = ItQualificationMatrixV1::new();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let assessment =
            assess_legacy_strong_qualification_v1(&pack, &profile, &matrix, &artifacts).unwrap();
        assert!(assessment.strongly_ready_requirements <= assessment.baseline.ready_requirements);
    }
}
