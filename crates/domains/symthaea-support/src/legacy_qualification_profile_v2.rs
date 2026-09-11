// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! V2 legacy qualification assessment with selected source evidence.
//!
//! V1's qualification target remains authoritative for competency requirements.
//! V2 changes only the source-evidence gate: immutable metadata history may stay
//! in the registry while qualification selects retained content-bound successors
//! and verifies claims against those exact artifacts.

use crate::it_qualification::{
    AdversarialConditionV1, ItCompetencyLevelV1, ItDomainV1, ItQualificationMatrixV1,
    QualificationEvidenceClassV1,
};
use crate::legacy_computing::{
    LegacyComputingErrorV1, LegacyComputingPackV1, LegacyCoverageStateV1,
    LegacyKnowledgeAreaV1, LegacyPlatformV1,
};
use crate::legacy_qualification_profile::{
    area_tag, platform_tag, LegacyQualificationProfileErrorV1, LegacyQualificationProfileV1,
};
use crate::legacy_qualification_source_ledger_v2::{
    assess_legacy_qualification_source_readiness_v2, LegacyQualificationSourceLedgerErrorV2,
    LegacyQualificationSourceLedgerV2,
};
use crate::legacy_source_artifacts::LegacySourceArtifactLedgerV1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const LEGACY_QUALIFICATION_PROFILE_ASSESSMENT_SCHEMA_V2: &str =
    "symthaea-it-legacy-qualification-profile-assessment-v2";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacyQualificationBlockerV2 {
    SourceEvidenceIncomplete,
    KnowledgeAreaUnmapped,
    InsufficientCases { required: usize, observed: usize },
    MinimumLevelNotMet { required: ItCompetencyLevelV1 },
    MissingEvidenceClass(QualificationEvidenceClassV1),
    MissingAdversarialCondition(AdversarialConditionV1),
    MissingHighStakesCase,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationRequirementAssessmentV2 {
    pub platform: LegacyPlatformV1,
    pub area: LegacyKnowledgeAreaV1,
    pub knowledge_state: LegacyCoverageStateV1,
    pub matching_active_cases: usize,
    pub blockers: BTreeSet<LegacyQualificationBlockerV2>,
}

impl LegacyQualificationRequirementAssessmentV2 {
    pub fn ready_for_evaluation(&self) -> bool {
        self.blockers.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationProfileAssessmentV2 {
    pub schema_version: String,
    pub total_requirements: usize,
    pub ready_requirements: usize,
    pub source_evidence_ready: bool,
    pub source_required_documents: usize,
    pub source_selected_documents: usize,
    pub source_required_claims: usize,
    pub source_verified_claims: usize,
    pub requirements: Vec<LegacyQualificationRequirementAssessmentV2>,
}

pub fn assess_legacy_qualification_profile_v2(
    pack: &LegacyComputingPackV1,
    profile: &LegacyQualificationProfileV1,
    matrix: &ItQualificationMatrixV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    sources: &LegacyQualificationSourceLedgerV2,
) -> Result<LegacyQualificationProfileAssessmentV2, LegacyQualificationProfileErrorV2> {
    pack.validate()?;
    profile.validate()?;
    let source_readiness =
        assess_legacy_qualification_source_readiness_v2(pack, artifacts, sources)?;

    let mut assessments = Vec::with_capacity(profile.requirements.len());
    for requirement in &profile.requirements {
        let profile_entry = pack.profile(requirement.platform).ok_or(
            LegacyQualificationProfileErrorV2::MissingPlatform(requirement.platform),
        )?;
        let knowledge_state = profile_entry.state(requirement.area);
        let expected_platform_tag = platform_tag(requirement.platform);
        let expected_area_tag = area_tag(requirement.area);
        let matching: Vec<_> = matrix
            .cases()
            .map(|(case, _)| case)
            .filter(|case| case.active && case.domain == ItDomainV1::LegacyComputing)
            .filter(|case| {
                has_tag(&case.technology_tags, expected_platform_tag)
                    && has_tag(&case.technology_tags, expected_area_tag)
            })
            .collect();

        let mut blockers = BTreeSet::new();
        if !source_readiness.source_evidence_ready {
            blockers.insert(LegacyQualificationBlockerV2::SourceEvidenceIncomplete);
        }
        if knowledge_state == LegacyCoverageStateV1::Unmapped {
            blockers.insert(LegacyQualificationBlockerV2::KnowledgeAreaUnmapped);
        }
        if matching.len() < requirement.minimum_cases {
            blockers.insert(LegacyQualificationBlockerV2::InsufficientCases {
                required: requirement.minimum_cases,
                observed: matching.len(),
            });
        }
        if !matching
            .iter()
            .any(|case| level_at_least(case.level, requirement.minimum_level))
        {
            blockers.insert(LegacyQualificationBlockerV2::MinimumLevelNotMet {
                required: requirement.minimum_level,
            });
        }

        let evidence: BTreeSet<_> = matching.iter().map(|case| case.evidence_class).collect();
        for required in &requirement.required_evidence_classes {
            if !evidence.contains(required) {
                blockers.insert(LegacyQualificationBlockerV2::MissingEvidenceClass(*required));
            }
        }

        let adversarial: BTreeSet<_> = matching
            .iter()
            .flat_map(|case| case.adversarial_conditions.iter().copied())
            .collect();
        for required in &requirement.required_adversarial_conditions {
            if !adversarial.contains(required) {
                blockers.insert(LegacyQualificationBlockerV2::MissingAdversarialCondition(
                    *required,
                ));
            }
        }
        if requirement.require_high_stakes_case && !matching.iter().any(|case| case.high_stakes) {
            blockers.insert(LegacyQualificationBlockerV2::MissingHighStakesCase);
        }

        assessments.push(LegacyQualificationRequirementAssessmentV2 {
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

    Ok(LegacyQualificationProfileAssessmentV2 {
        schema_version: LEGACY_QUALIFICATION_PROFILE_ASSESSMENT_SCHEMA_V2.into(),
        total_requirements: assessments.len(),
        ready_requirements,
        source_evidence_ready: source_readiness.source_evidence_ready,
        source_required_documents: source_readiness.required_documents,
        source_selected_documents: source_readiness.selected_documents,
        source_required_claims: source_readiness.required_claims,
        source_verified_claims: source_readiness.verified_claims,
        requirements: assessments,
    })
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
pub enum LegacyQualificationProfileErrorV2 {
    LegacyPack(LegacyComputingErrorV1),
    ProfileV1(LegacyQualificationProfileErrorV1),
    SourceEvidence(LegacyQualificationSourceLedgerErrorV2),
    MissingPlatform(LegacyPlatformV1),
}

impl fmt::Display for LegacyQualificationProfileErrorV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyPack(err) => write!(f, "invalid legacy pack: {err}"),
            Self::ProfileV1(err) => write!(f, "invalid legacy V1 qualification target: {err}"),
            Self::SourceEvidence(err) => write!(f, "invalid legacy V2 source evidence: {err}"),
            Self::MissingPlatform(platform) => {
                write!(f, "legacy V2 qualification is missing platform {platform:?}")
            }
        }
    }
}

impl Error for LegacyQualificationProfileErrorV2 {}

impl From<LegacyComputingErrorV1> for LegacyQualificationProfileErrorV2 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::LegacyPack(value)
    }
}

impl From<LegacyQualificationProfileErrorV1> for LegacyQualificationProfileErrorV2 {
    fn from(value: LegacyQualificationProfileErrorV1) -> Self {
        Self::ProfileV1(value)
    }
}

impl From<LegacyQualificationSourceLedgerErrorV2> for LegacyQualificationProfileErrorV2 {
    fn from(value: LegacyQualificationSourceLedgerErrorV2) -> Self {
        Self::SourceEvidence(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_source::KnowledgeLifecycleV1;
    use crate::legacy_qualification_profile::exhaustive_legacy_qualification_profile_v1;
    use crate::legacy_qualification_source_ledger_v2::{
        legacy_claim_verification_binding_v2, LegacyClaimVerificationMethodV2,
        LegacyClaimVerificationReceiptV2, LegacyQualificationSourceSelectionV2,
    };
    use crate::legacy_source_artifacts::{
        LegacyArtifactAccessPolicyV1, LegacyArtifactStorageClassV1,
        LegacySourceArtifactRefV1,
    };
    use crate::legacy_source_capture_plan::plan_legacy_qualification_source_captures_v2;
    use crate::legacy_source_readiness::assess_legacy_source_readiness_v1;
    use crate::standards_registry::{SourceCaptureV1, TechnicalSourceSnapshotV1};
    use crate::build_legacy_five_platform_portfolio_v1;

    fn fully_source_bound_portfolio() -> (
        LegacyComputingPackV1,
        ItQualificationMatrixV1,
        LegacySourceArtifactLedgerV1,
        LegacyQualificationSourceLedgerV2,
    ) {
        let (mut pack, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let plan = plan_legacy_qualification_source_captures_v2(&pack).unwrap();
        let mut artifacts = LegacySourceArtifactLedgerV1::new();
        let mut sources = LegacyQualificationSourceLedgerV2::new();

        for (index, request) in plan.requests.iter().enumerate() {
            let original_id = request.metadata_snapshot_ids.iter().next().unwrap();
            let original = pack.sources.snapshot(original_id).unwrap().clone();
            let qualifying_id = crate::SourceSnapshotIdV1(format!(
                "{}:qualification-content-v2",
                request.document_id.0
            ));
            let digest = format!("{:064x}", index + 1);
            pack.sources
                .register_snapshot(TechnicalSourceSnapshotV1 {
                    id: qualifying_id.clone(),
                    document_id: request.document_id.clone(),
                    version: original.version.clone(),
                    lifecycle: original.lifecycle,
                    authority: original.authority,
                    stability: original.stability,
                    published_at_unix_ms: original.published_at_unix_ms,
                    source_updated_at_unix_ms: original.source_updated_at_unix_ms,
                    fetched_at_unix_ms: 1_800_000_010_000 + index as u64,
                    capture: SourceCaptureV1::ContentDigest {
                        algorithm: "sha256".into(),
                        digest: digest.clone(),
                    },
                    relations: original.relations.clone(),
                })
                .unwrap();
            artifacts
                .register(
                    &pack,
                    LegacySourceArtifactRefV1 {
                        snapshot_id: qualifying_id.clone(),
                        content_algorithm: "sha256".into(),
                        content_digest: digest.clone(),
                        byte_length: 1_000 + index as u64,
                        media_type: "application/octet-stream".into(),
                        retrieved_at_unix_ms: 1_800_000_020_000 + index as u64,
                        artifact_locator: format!(
                            "evidence://legacy/qualification/document-{index}"
                        ),
                        storage_class: LegacyArtifactStorageClassV1::PrivateEvidenceStore,
                        access_policy: LegacyArtifactAccessPolicyV1::EvaluatorOnly,
                        retention_receipt_digest: None,
                    },
                )
                .unwrap();
            sources
                .register_selection(
                    &pack,
                    &artifacts,
                    LegacyQualificationSourceSelectionV2 {
                        document_id: request.document_id.clone(),
                        qualifying_snapshot_id: qualifying_id,
                        content_algorithm: "sha256".into(),
                        content_digest: digest,
                        selected_at_unix_ms: 1_800_000_030_000 + index as u64,
                    },
                )
                .unwrap();
        }

        let claims: Vec<_> = pack.sources.claims().cloned().collect();
        for (index, claim) in claims.into_iter().enumerate() {
            let original = pack.sources.snapshot(&claim.source_snapshot).unwrap();
            let selection = sources.selection(&original.document_id).unwrap().clone();
            let binding = legacy_claim_verification_binding_v2(&claim, &selection).unwrap();
            sources
                .register_claim_receipt(
                    &pack,
                    &artifacts,
                    LegacyClaimVerificationReceiptV2 {
                        claim_id: claim.id.clone(),
                        document_id: original.document_id.clone(),
                        original_snapshot_id: claim.source_snapshot.clone(),
                        qualifying_snapshot_id: selection.qualifying_snapshot_id.clone(),
                        content_algorithm: selection.content_algorithm.clone(),
                        content_digest: selection.content_digest.clone(),
                        verification_method: LegacyClaimVerificationMethodV2::HumanAndDeterministic,
                        verifier_profile: "legacy-source-verifier-v2-test".into(),
                        verified_at_unix_ms: 1_800_000_040_000 + index as u64,
                        claim_binding_blake3: binding,
                    },
                )
                .unwrap();
        }

        (pack, matrix, artifacts, sources)
    }

    #[test]
    fn empty_v2_source_ledger_blocks_all_fifty_cells_without_hiding_other_gaps() {
        let (pack, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let profile = exhaustive_legacy_qualification_profile_v1();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let sources = LegacyQualificationSourceLedgerV2::new();
        let assessment = assess_legacy_qualification_profile_v2(
            &pack,
            &profile,
            &matrix,
            &artifacts,
            &sources,
        )
        .unwrap();
        assert_eq!(assessment.total_requirements, 50);
        assert_eq!(
            assessment
                .requirements
                .iter()
                .filter(|cell| cell.blockers.contains(&LegacyQualificationBlockerV2::SourceEvidenceIncomplete))
                .count(),
            50
        );
        assert!(!assessment.source_evidence_ready);
        assert!(assessment.requirements.iter().any(|cell| cell.blockers.len() > 1));
    }

    #[test]
    fn complete_v2_source_evidence_clears_only_the_provenance_blocker() {
        let (pack, matrix, artifacts, sources) = fully_source_bound_portfolio();
        let profile = exhaustive_legacy_qualification_profile_v1();

        // Historical metadata remains by design, so the old V1 global source
        // predicate is still false.
        let old = assess_legacy_source_readiness_v1(&pack).unwrap();
        assert!(old.metadata_only > 0);
        assert!(!old.qualification_ready);

        let assessment = assess_legacy_qualification_profile_v2(
            &pack,
            &profile,
            &matrix,
            &artifacts,
            &sources,
        )
        .unwrap();
        assert!(assessment.source_evidence_ready);
        assert_eq!(assessment.source_selected_documents, assessment.source_required_documents);
        assert_eq!(assessment.source_verified_claims, assessment.source_required_claims);
        assert!(assessment.requirements.iter().all(|cell| {
            !cell
                .blockers
                .contains(&LegacyQualificationBlockerV2::SourceEvidenceIncomplete)
        }));

        // Source evidence is no longer the blocker, but the current public
        // portfolio still lacks enough scenario/hardware/depth evidence to make
        // every one of the fifty competency cells ready.
        assert!(assessment.ready_requirements < assessment.total_requirements);
    }
}
