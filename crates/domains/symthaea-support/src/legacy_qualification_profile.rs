// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exhaustive qualification target for legacy enterprise computing.
//!
//! This module defines what must eventually be demonstrated. It does not mint
//! qualification from repository coverage, source presence, or case registration.

use crate::it_qualification::{
    AdversarialConditionV1, ItCompetencyLevelV1, ItDomainV1, ItQualificationMatrixV1,
    QualificationEvidenceClassV1,
};
use crate::legacy_computing::{
    LegacyComputingErrorV1, LegacyComputingPackV1, LegacyCoverageStateV1,
    LegacyKnowledgeAreaV1, LegacyPlatformV1,
};
use crate::legacy_source_readiness::{
    assess_legacy_source_readiness_v1, LegacySourceReadinessErrorV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_QUALIFICATION_PROFILE_SCHEMA_V1: &str =
    "symthaea-it-legacy-qualification-profile-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationRequirementV1 {
    pub platform: LegacyPlatformV1,
    pub area: LegacyKnowledgeAreaV1,
    pub minimum_level: ItCompetencyLevelV1,
    pub minimum_cases: usize,
    pub required_evidence_classes: BTreeSet<QualificationEvidenceClassV1>,
    pub required_adversarial_conditions: BTreeSet<AdversarialConditionV1>,
    pub require_high_stakes_case: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationProfileV1 {
    pub schema_version: String,
    pub requirements: Vec<LegacyQualificationRequirementV1>,
}

impl LegacyQualificationProfileV1 {
    pub fn validate(&self) -> Result<(), LegacyQualificationProfileErrorV1> {
        if self.schema_version != LEGACY_QUALIFICATION_PROFILE_SCHEMA_V1 {
            return Err(LegacyQualificationProfileErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        let mut seen = BTreeSet::new();
        for requirement in &self.requirements {
            if !seen.insert((requirement.platform, requirement.area)) {
                return Err(LegacyQualificationProfileErrorV1::DuplicateRequirement {
                    platform: requirement.platform,
                    area: requirement.area,
                });
            }
            if requirement.minimum_cases == 0 {
                return Err(LegacyQualificationProfileErrorV1::InvalidRequirement(
                    "minimum legacy qualification cases must be non-zero".into(),
                ));
            }
            if requirement.required_evidence_classes.is_empty() {
                return Err(LegacyQualificationProfileErrorV1::InvalidRequirement(
                    "legacy qualification requirement needs an evidence class".into(),
                ));
            }
        }
        let expected: BTreeSet<_> = LegacyPlatformV1::ALL
            .into_iter()
            .flat_map(|platform| {
                LegacyKnowledgeAreaV1::ALL
                    .into_iter()
                    .map(move |area| (platform, area))
            })
            .collect();
        if seen != expected {
            return Err(LegacyQualificationProfileErrorV1::IncompleteRequirementMatrix {
                expected: expected.len(),
                actual: seen.len(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacyQualificationBlockerV1 {
    SourceNotContentDigestBound,
    KnowledgeAreaUnmapped,
    InsufficientCases { required: usize, observed: usize },
    MinimumLevelNotMet { required: ItCompetencyLevelV1 },
    MissingEvidenceClass(QualificationEvidenceClassV1),
    MissingAdversarialCondition(AdversarialConditionV1),
    MissingHighStakesCase,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationRequirementAssessmentV1 {
    pub platform: LegacyPlatformV1,
    pub area: LegacyKnowledgeAreaV1,
    pub knowledge_state: LegacyCoverageStateV1,
    pub matching_active_cases: usize,
    pub blockers: BTreeSet<LegacyQualificationBlockerV1>,
}

impl LegacyQualificationRequirementAssessmentV1 {
    pub fn ready_for_evaluation(&self) -> bool {
        self.blockers.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationProfileAssessmentV1 {
    pub total_requirements: usize,
    pub ready_requirements: usize,
    pub source_qualification_ready: bool,
    pub requirements: Vec<LegacyQualificationRequirementAssessmentV1>,
}

pub fn exhaustive_legacy_qualification_profile_v1() -> LegacyQualificationProfileV1 {
    let requirements = LegacyPlatformV1::ALL
        .into_iter()
        .flat_map(|platform| {
            LegacyKnowledgeAreaV1::ALL
                .into_iter()
                .map(move |area| requirement(platform, area))
        })
        .collect();
    LegacyQualificationProfileV1 {
        schema_version: LEGACY_QUALIFICATION_PROFILE_SCHEMA_V1.into(),
        requirements,
    }
}

pub fn assess_legacy_qualification_profile_v1(
    pack: &LegacyComputingPackV1,
    profile: &LegacyQualificationProfileV1,
    matrix: &ItQualificationMatrixV1,
) -> Result<LegacyQualificationProfileAssessmentV1, LegacyQualificationProfileErrorV1> {
    pack.validate()?;
    profile.validate()?;
    let source_readiness = assess_legacy_source_readiness_v1(pack)?;

    let mut assessments = Vec::with_capacity(profile.requirements.len());
    for requirement in &profile.requirements {
        let profile_entry = pack
            .profile(requirement.platform)
            .ok_or(LegacyQualificationProfileErrorV1::MissingPlatform(
                requirement.platform,
            ))?;
        let knowledge_state = profile_entry.state(requirement.area);
        let platform_tag = platform_tag(requirement.platform);
        let area_tag = area_tag(requirement.area);
        let matching: Vec<_> = matrix
            .cases()
            .map(|(case, _)| case)
            .filter(|case| case.active && case.domain == ItDomainV1::LegacyComputing)
            .filter(|case| {
                has_tag(&case.technology_tags, platform_tag)
                    && has_tag(&case.technology_tags, area_tag)
            })
            .collect();

        let mut blockers = BTreeSet::new();
        if !source_readiness.qualification_ready {
            blockers.insert(LegacyQualificationBlockerV1::SourceNotContentDigestBound);
        }
        if knowledge_state == LegacyCoverageStateV1::Unmapped {
            blockers.insert(LegacyQualificationBlockerV1::KnowledgeAreaUnmapped);
        }
        if matching.len() < requirement.minimum_cases {
            blockers.insert(LegacyQualificationBlockerV1::InsufficientCases {
                required: requirement.minimum_cases,
                observed: matching.len(),
            });
        }
        if !matching
            .iter()
            .any(|case| level_at_least(case.level, requirement.minimum_level))
        {
            blockers.insert(LegacyQualificationBlockerV1::MinimumLevelNotMet {
                required: requirement.minimum_level,
            });
        }

        let evidence: BTreeSet<_> = matching.iter().map(|case| case.evidence_class).collect();
        for required in &requirement.required_evidence_classes {
            if !evidence.contains(required) {
                blockers.insert(LegacyQualificationBlockerV1::MissingEvidenceClass(*required));
            }
        }
        let adversarial: BTreeSet<_> = matching
            .iter()
            .flat_map(|case| case.adversarial_conditions.iter().copied())
            .collect();
        for required in &requirement.required_adversarial_conditions {
            if !adversarial.contains(required) {
                blockers.insert(LegacyQualificationBlockerV1::MissingAdversarialCondition(
                    *required,
                ));
            }
        }
        if requirement.require_high_stakes_case && !matching.iter().any(|case| case.high_stakes) {
            blockers.insert(LegacyQualificationBlockerV1::MissingHighStakesCase);
        }

        assessments.push(LegacyQualificationRequirementAssessmentV1 {
            platform: requirement.platform,
            area: requirement.area,
            knowledge_state,
            matching_active_cases: matching.len(),
            blockers,
        });
    }

    assessments.sort_by_key(|assessment| (assessment.platform, assessment.area));
    let ready_requirements = assessments
        .iter()
        .filter(|assessment| assessment.ready_for_evaluation())
        .count();
    Ok(LegacyQualificationProfileAssessmentV1 {
        total_requirements: assessments.len(),
        ready_requirements,
        source_qualification_ready: source_readiness.qualification_ready,
        requirements: assessments,
    })
}

/// Stable tags expected on future legacy qualification cases.
pub fn platform_tag(platform: LegacyPlatformV1) -> &'static str {
    match platform {
        LegacyPlatformV1::Zos => "legacy-platform:zos",
        LegacyPlatformV1::IbmI => "legacy-platform:ibm-i",
        LegacyPlatformV1::Aix => "legacy-platform:aix",
        LegacyPlatformV1::Solaris => "legacy-platform:solaris",
        LegacyPlatformV1::HpUx => "legacy-platform:hp-ux",
    }
}

pub fn area_tag(area: LegacyKnowledgeAreaV1) -> &'static str {
    match area {
        LegacyKnowledgeAreaV1::SystemLifecycle => "legacy-area:system-lifecycle",
        LegacyKnowledgeAreaV1::WorkloadAndJobs => "legacy-area:workload-jobs",
        LegacyKnowledgeAreaV1::Storage => "legacy-area:storage",
        LegacyKnowledgeAreaV1::Networking => "legacy-area:networking",
        LegacyKnowledgeAreaV1::IdentityAndSecurity => "legacy-area:identity-security",
        LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement => {
            "legacy-area:observability-problem-management"
        }
        LegacyKnowledgeAreaV1::AvailabilityAndRecovery => "legacy-area:availability-recovery",
        LegacyKnowledgeAreaV1::VirtualizationAndPartitioning => {
            "legacy-area:virtualization-partitioning"
        }
        LegacyKnowledgeAreaV1::SoftwareLifecycle => "legacy-area:software-lifecycle",
        LegacyKnowledgeAreaV1::InteroperabilityAndMigration => {
            "legacy-area:interoperability-migration"
        }
    }
}

fn requirement(
    platform: LegacyPlatformV1,
    area: LegacyKnowledgeAreaV1,
) -> LegacyQualificationRequirementV1 {
    use AdversarialConditionV1 as A;
    use ItCompetencyLevelV1 as L;
    use LegacyKnowledgeAreaV1 as K;
    use QualificationEvidenceClassV1 as E;

    let (minimum_level, minimum_cases, evidence, adversarial, high_stakes) = match area {
        K::SystemLifecycle => (
            L::Diagnosis,
            2,
            BTreeSet::from([E::DeterministicReplay]),
            BTreeSet::from([A::VersionMismatch, A::RecoveryConstraint]),
            false,
        ),
        K::WorkloadAndJobs => (
            L::Diagnosis,
            2,
            BTreeSet::from([E::DeterministicReplay]),
            BTreeSet::from([A::PartialFailure]),
            false,
        ),
        K::Storage => (
            L::Causality,
            2,
            BTreeSet::from([E::DeterministicReplay]),
            BTreeSet::from([A::MultipleFaults, A::RecoveryConstraint]),
            true,
        ),
        K::Networking => (
            L::Diagnosis,
            2,
            BTreeSet::from([E::DeterministicReplay]),
            BTreeSet::from([A::PartialFailure, A::StaleTelemetry]),
            false,
        ),
        K::IdentityAndSecurity => (
            L::Adversarial,
            2,
            BTreeSet::from([E::DeterministicReplay]),
            BTreeSet::from([A::PrivilegeConstraint, A::UnsafeSuggestedAction]),
            true,
        ),
        K::ObservabilityAndProblemManagement => (
            L::Causality,
            3,
            BTreeSet::from([E::DeterministicReplay]),
            BTreeSet::from([A::ConflictingSources, A::MisleadingAlert]),
            false,
        ),
        K::AvailabilityAndRecovery => (
            L::Operations,
            3,
            BTreeSet::from([E::DeterministicReplay, E::HardwareLab]),
            BTreeSet::from([A::MultipleFaults, A::RecoveryConstraint, A::UnsafeSuggestedAction]),
            true,
        ),
        K::VirtualizationAndPartitioning => (
            L::Architecture,
            2,
            BTreeSet::from([E::DeterministicReplay]),
            BTreeSet::from([A::TopologyDrift]),
            false,
        ),
        K::SoftwareLifecycle => (
            L::Diagnosis,
            2,
            BTreeSet::from([E::DeterministicReplay]),
            BTreeSet::from([A::VersionMismatch, A::RecoveryConstraint]),
            false,
        ),
        K::InteroperabilityAndMigration => (
            L::CrossDomainTransfer,
            2,
            BTreeSet::from([E::DeterministicReplay]),
            BTreeSet::from([A::VersionMismatch, A::ConflictingSources]),
            true,
        ),
    };

    LegacyQualificationRequirementV1 {
        platform,
        area,
        minimum_level,
        minimum_cases,
        required_evidence_classes: evidence,
        required_adversarial_conditions: adversarial,
        require_high_stakes_case: high_stakes,
    }
}

fn has_tag(tags: &[String], expected: &str) -> bool {
    tags.iter()
        .any(|tag| tag.trim().eq_ignore_ascii_case(expected))
}

fn level_at_least(actual: ItCompetencyLevelV1, required: ItCompetencyLevelV1) -> bool {
    level_rank(actual) >= level_rank(required)
}

fn level_rank(level: ItCompetencyLevelV1) -> u8 {
    match level {
        ItCompetencyLevelV1::Recognition => 0,
        ItCompetencyLevelV1::Recall => 1,
        ItCompetencyLevelV1::Mechanism => 2,
        ItCompetencyLevelV1::Configuration => 3,
        ItCompetencyLevelV1::Diagnosis => 4,
        ItCompetencyLevelV1::Causality => 5,
        ItCompetencyLevelV1::Architecture => 6,
        ItCompetencyLevelV1::Tradeoffs => 7,
        ItCompetencyLevelV1::Adversarial => 8,
        ItCompetencyLevelV1::Operations => 9,
        ItCompetencyLevelV1::CrossDomainTransfer => 10,
    }
}

#[derive(Debug)]
pub enum LegacyQualificationProfileErrorV1 {
    LegacyPack(LegacyComputingErrorV1),
    SourceReadiness(LegacySourceReadinessErrorV1),
    UnsupportedSchema(String),
    InvalidRequirement(String),
    DuplicateRequirement {
        platform: LegacyPlatformV1,
        area: LegacyKnowledgeAreaV1,
    },
    IncompleteRequirementMatrix { expected: usize, actual: usize },
    MissingPlatform(LegacyPlatformV1),
}

impl fmt::Display for LegacyQualificationProfileErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyPack(err) => write!(f, "invalid legacy pack: {err}"),
            Self::SourceReadiness(err) => write!(f, "invalid legacy source readiness: {err}"),
            Self::UnsupportedSchema(schema) => {
                write!(f, "unsupported legacy qualification profile {schema}")
            }
            Self::InvalidRequirement(message) => write!(f, "invalid legacy requirement: {message}"),
            Self::DuplicateRequirement { platform, area } => {
                write!(f, "duplicate legacy qualification requirement {platform:?}/{area:?}")
            }
            Self::IncompleteRequirementMatrix { expected, actual } => write!(
                f,
                "incomplete legacy qualification matrix; expected={expected}, actual={actual}"
            ),
            Self::MissingPlatform(platform) => write!(f, "legacy pack missing {platform:?}"),
        }
    }
}

impl Error for LegacyQualificationProfileErrorV1 {}

impl From<LegacyComputingErrorV1> for LegacyQualificationProfileErrorV1 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::LegacyPack(value)
    }
}

impl From<LegacySourceReadinessErrorV1> for LegacyQualificationProfileErrorV1 {
    fn from(value: LegacySourceReadinessErrorV1) -> Self {
        Self::SourceReadiness(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed_legacy_computing_pack_v1;

    #[test]
    fn exhaustive_profile_has_exactly_fifty_platform_area_requirements() {
        let profile = exhaustive_legacy_qualification_profile_v1();
        profile.validate().unwrap();
        assert_eq!(profile.requirements.len(), 50);
    }

    #[test]
    fn current_seed_pack_cannot_be_mistaken_for_qualification_readiness() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let profile = exhaustive_legacy_qualification_profile_v1();
        let matrix = ItQualificationMatrixV1::new();
        let assessment =
            assess_legacy_qualification_profile_v1(&pack, &profile, &matrix).unwrap();
        assert_eq!(assessment.total_requirements, 50);
        assert_eq!(assessment.ready_requirements, 0);
        assert!(!assessment.source_qualification_ready);
        assert!(assessment.requirements.iter().all(|item| {
            item.blockers
                .contains(&LegacyQualificationBlockerV1::SourceNotContentDigestBound)
        }));
    }

    #[test]
    fn target_requires_hardware_evidence_for_availability_recovery() {
        let profile = exhaustive_legacy_qualification_profile_v1();
        for requirement in profile
            .requirements
            .iter()
            .filter(|requirement| {
                requirement.area == LegacyKnowledgeAreaV1::AvailabilityAndRecovery
            })
        {
            assert!(requirement
                .required_evidence_classes
                .contains(&QualificationEvidenceClassV1::HardwareLab));
            assert!(requirement.require_high_stakes_case);
        }
    }

    #[test]
    fn stable_case_tags_are_platform_and_area_specific() {
        assert_ne!(
            platform_tag(LegacyPlatformV1::Aix),
            platform_tag(LegacyPlatformV1::Solaris)
        );
        assert_ne!(
            area_tag(LegacyKnowledgeAreaV1::Storage),
            area_tag(LegacyKnowledgeAreaV1::Networking)
        );
    }
}
