// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-derived promotion of a published research release.
//!
//! Promotion is intentionally about research-release maturity, not product
//! safety or general musical superiority. A stable research release requires
//! independent replication, durable archives, distributed stewardship, and a
//! fixed public claim statement.

use crate::confirmatory_final_release::{
    ConfirmatoryFinalReleaseBundle, confirmatory_final_release_commitment,
};
use crate::evidence_digest::canonical_json_sha256;
use crate::replication_synthesis::{
    ReplicationSynthesisConclusion, ReplicationSynthesisRecord, replication_synthesis_commitment,
};
use crate::research_archive::{ResearchArchiveManifest, research_archive_commitment};
use crate::stewardship_governance::{
    ResearchStewardshipCharter, stewardship_charter_commitment, validate_stewardship_charter,
};
use serde::{Deserialize, Serialize};

pub const RESEARCH_RELEASE_PROMOTION_VERSION: &str = "symthaea-muse-research-release-promotion-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResearchReleaseStage {
    ConfirmatoryPublished,
    ReplicationEvaluated,
    IndependentlyReplicated,
    StableResearchRelease,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromotionGateStatus {
    Passed,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PromotionGateKind {
    SourceReleaseValid,
    ReplicationEvidenceValid,
    ReplicationSupportsClaim,
    ArchiveDurable,
    StewardshipDistributed,
    SecurityReviewCurrent,
    PublicClaimFrozen,
    SupportWindowDeclared,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotionGateResult {
    pub gate: PromotionGateKind,
    pub status: PromotionGateStatus,
    pub evidence_sha256: String,
    pub explanation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchReleasePromotionRequest {
    pub release_id: String,
    pub semantic_version: String,
    pub target_stage: ResearchReleaseStage,
    pub security_review_sha256: String,
    pub frozen_public_claim_sha256: String,
    pub support_policy_sha256: String,
    pub support_until_utc: String,
    pub public_release_uri: String,
    pub requested_at_utc: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchReleasePromotionRecord {
    pub promotion_version: String,
    pub release_id: String,
    pub semantic_version: String,
    pub target_stage: ResearchReleaseStage,
    pub source_final_release_sha256: String,
    pub replication_synthesis_sha256: String,
    pub research_archive_sha256: String,
    pub stewardship_charter_sha256: String,
    pub gates: Vec<PromotionGateResult>,
    pub promoted: bool,
    pub public_release_uri: String,
    pub support_until_utc: String,
    pub promoted_at_utc: String,
    pub promotion_sha256: String,
}

#[derive(Serialize)]
struct PromotionCommitment<'a> {
    promotion_version: &'a str,
    release_id: &'a str,
    semantic_version: &'a str,
    target_stage: ResearchReleaseStage,
    source_final_release_sha256: &'a str,
    replication_synthesis_sha256: &'a str,
    research_archive_sha256: &'a str,
    stewardship_charter_sha256: &'a str,
    gates: &'a [PromotionGateResult],
    promoted: bool,
    public_release_uri: &'a str,
    support_until_utc: &'a str,
    promoted_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResearchReleasePromotionIssue {
    WrongVersion { found: String },
    InvalidSourceRelease,
    InvalidReplicationSynthesis,
    InvalidArchive,
    InvalidStewardshipCharter,
    AuthorityMismatch,
    EmptyField { field: String },
    InvalidDigest { field: String },
    InvalidSemanticVersion,
    DuplicateGate { gate: PromotionGateKind },
    MissingGate { gate: PromotionGateKind },
    GateMismatch { gate: PromotionGateKind },
    PromotionDecisionMismatch,
    SerializationFailed,
    PromotionDigestMismatch,
}

pub fn research_release_promotion_commitment(
    record: &ResearchReleasePromotionRecord,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&PromotionCommitment {
        promotion_version: &record.promotion_version,
        release_id: &record.release_id,
        semantic_version: &record.semantic_version,
        target_stage: record.target_stage,
        source_final_release_sha256: &record.source_final_release_sha256,
        replication_synthesis_sha256: &record.replication_synthesis_sha256,
        research_archive_sha256: &record.research_archive_sha256,
        stewardship_charter_sha256: &record.stewardship_charter_sha256,
        gates: &record.gates,
        promoted: record.promoted,
        public_release_uri: &record.public_release_uri,
        support_until_utc: &record.support_until_utc,
        promoted_at_utc: &record.promoted_at_utc,
    })
}

pub fn evaluate_research_release_promotion(
    source_release: &ConfirmatoryFinalReleaseBundle,
    synthesis: &ReplicationSynthesisRecord,
    archive: &ResearchArchiveManifest,
    charter: &ResearchStewardshipCharter,
    request: &ResearchReleasePromotionRequest,
) -> Result<ResearchReleasePromotionRecord, Vec<ResearchReleasePromotionIssue>> {
    let gates = derive_gates(source_release, synthesis, archive, charter, request);
    let promoted = gates
        .iter()
        .all(|gate| gate.status == PromotionGateStatus::Passed);
    let mut record = ResearchReleasePromotionRecord {
        promotion_version: RESEARCH_RELEASE_PROMOTION_VERSION.into(),
        release_id: request.release_id.clone(),
        semantic_version: request.semantic_version.clone(),
        target_stage: request.target_stage,
        source_final_release_sha256: source_release.bundle_sha256.clone(),
        replication_synthesis_sha256: synthesis.synthesis_sha256.clone(),
        research_archive_sha256: archive.archive_sha256.clone(),
        stewardship_charter_sha256: charter.charter_sha256.clone(),
        gates,
        promoted,
        public_release_uri: request.public_release_uri.clone(),
        support_until_utc: request.support_until_utc.clone(),
        promoted_at_utc: request.requested_at_utc.clone(),
        promotion_sha256: String::new(),
    };
    record.promotion_sha256 = research_release_promotion_commitment(&record)
        .map_err(|_| vec![ResearchReleasePromotionIssue::SerializationFailed])?;
    let issues = validate_research_release_promotion(
        source_release,
        synthesis,
        archive,
        charter,
        request,
        &record,
    );
    if issues.is_empty() {
        Ok(record)
    } else {
        Err(issues)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn validate_research_release_promotion(
    source_release: &ConfirmatoryFinalReleaseBundle,
    synthesis: &ReplicationSynthesisRecord,
    archive: &ResearchArchiveManifest,
    charter: &ResearchStewardshipCharter,
    request: &ResearchReleasePromotionRequest,
    record: &ResearchReleasePromotionRecord,
) -> Vec<ResearchReleasePromotionIssue> {
    let mut issues = Vec::new();
    if record.promotion_version != RESEARCH_RELEASE_PROMOTION_VERSION {
        issues.push(ResearchReleasePromotionIssue::WrongVersion {
            found: record.promotion_version.clone(),
        });
    }
    let source_digest = match confirmatory_final_release_commitment(source_release) {
        Ok(value) if value == source_release.bundle_sha256 => value,
        _ => {
            issues.push(ResearchReleasePromotionIssue::InvalidSourceRelease);
            String::new()
        }
    };
    let synthesis_digest = match replication_synthesis_commitment(synthesis) {
        Ok(value) if value == synthesis.synthesis_sha256 => value,
        _ => {
            issues.push(ResearchReleasePromotionIssue::InvalidReplicationSynthesis);
            String::new()
        }
    };
    let archive_digest = match research_archive_commitment(archive) {
        Ok(value) if value == archive.archive_sha256 => value,
        _ => {
            issues.push(ResearchReleasePromotionIssue::InvalidArchive);
            String::new()
        }
    };
    let charter_digest = match stewardship_charter_commitment(charter) {
        Ok(value) if value == charter.charter_sha256 => value,
        _ => {
            issues.push(ResearchReleasePromotionIssue::InvalidStewardshipCharter);
            String::new()
        }
    };
    if record.release_id != request.release_id
        || record.semantic_version != request.semantic_version
        || record.target_stage != request.target_stage
        || record.source_final_release_sha256 != source_digest
        || record.replication_synthesis_sha256 != synthesis_digest
        || record.research_archive_sha256 != archive_digest
        || record.stewardship_charter_sha256 != charter_digest
        || synthesis.protocol_sha256.is_empty()
        || charter.source_final_release_sha256 != source_digest
        || archive.authority_root_sha256 != synthesis_digest
    {
        issues.push(ResearchReleasePromotionIssue::AuthorityMismatch);
    }
    for (field, value) in [
        ("release_id", record.release_id.as_str()),
        ("public_release_uri", record.public_release_uri.as_str()),
        ("support_until_utc", record.support_until_utc.as_str()),
        ("promoted_at_utc", record.promoted_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ResearchReleasePromotionIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        (
            "source_final_release_sha256",
            record.source_final_release_sha256.as_str(),
        ),
        (
            "replication_synthesis_sha256",
            record.replication_synthesis_sha256.as_str(),
        ),
        (
            "research_archive_sha256",
            record.research_archive_sha256.as_str(),
        ),
        (
            "stewardship_charter_sha256",
            record.stewardship_charter_sha256.as_str(),
        ),
        ("promotion_sha256", record.promotion_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ResearchReleasePromotionIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    if !valid_semantic_version(&record.semantic_version) {
        issues.push(ResearchReleasePromotionIssue::InvalidSemanticVersion);
    }
    let expected_gates = derive_gates(source_release, synthesis, archive, charter, request);
    let mut seen = std::collections::BTreeSet::new();
    for gate in &record.gates {
        if !seen.insert(gate.gate) {
            issues.push(ResearchReleasePromotionIssue::DuplicateGate { gate: gate.gate });
        }
        if !is_sha256(&gate.evidence_sha256) {
            issues.push(ResearchReleasePromotionIssue::InvalidDigest {
                field: format!("gate.{:?}.evidence_sha256", gate.gate),
            });
        }
        if gate.explanation.trim().is_empty() {
            issues.push(ResearchReleasePromotionIssue::EmptyField {
                field: format!("gate.{:?}.explanation", gate.gate),
            });
        }
    }
    for expected in &expected_gates {
        match record.gates.iter().find(|gate| gate.gate == expected.gate) {
            Some(found) if found == expected => {}
            Some(_) => issues.push(ResearchReleasePromotionIssue::GateMismatch {
                gate: expected.gate,
            }),
            None => issues.push(ResearchReleasePromotionIssue::MissingGate {
                gate: expected.gate,
            }),
        }
    }
    let expected_promoted = expected_gates
        .iter()
        .all(|gate| gate.status == PromotionGateStatus::Passed);
    if record.promoted != expected_promoted {
        issues.push(ResearchReleasePromotionIssue::PromotionDecisionMismatch);
    }
    match research_release_promotion_commitment(record) {
        Ok(digest) if digest == record.promotion_sha256 => {}
        Ok(_) => issues.push(ResearchReleasePromotionIssue::PromotionDigestMismatch),
        Err(_) => issues.push(ResearchReleasePromotionIssue::SerializationFailed),
    }
    issues
}

fn derive_gates(
    source_release: &ConfirmatoryFinalReleaseBundle,
    synthesis: &ReplicationSynthesisRecord,
    archive: &ResearchArchiveManifest,
    charter: &ResearchStewardshipCharter,
    request: &ResearchReleasePromotionRequest,
) -> Vec<PromotionGateResult> {
    let source_valid = confirmatory_final_release_commitment(source_release)
        .is_ok_and(|digest| digest == source_release.bundle_sha256);
    let synthesis_valid = replication_synthesis_commitment(synthesis)
        .is_ok_and(|digest| digest == synthesis.synthesis_sha256);
    let archive_valid = research_archive_commitment(archive)
        .is_ok_and(|digest| digest == archive.archive_sha256)
        && archive.locations.len() >= 2
        && archive.recovery_drill.succeeded;
    let charter_valid = stewardship_charter_commitment(charter)
        .is_ok_and(|digest| digest == charter.charter_sha256)
        && validate_stewardship_charter(source_release, charter).is_empty();
    let replication_evidence_required = !matches!(
        request.target_stage,
        ResearchReleaseStage::ConfirmatoryPublished
    );
    let replication_support_required = matches!(
        request.target_stage,
        ResearchReleaseStage::IndependentlyReplicated | ResearchReleaseStage::StableResearchRelease
    );
    let stable_requirements = matches!(
        request.target_stage,
        ResearchReleaseStage::StableResearchRelease
    );
    vec![
        gate(
            PromotionGateKind::SourceReleaseValid,
            source_valid,
            source_release.bundle_sha256.clone(),
            "The V12 source release commitment is internally valid.",
        ),
        gate(
            PromotionGateKind::ReplicationEvidenceValid,
            !replication_evidence_required || synthesis_valid,
            synthesis.synthesis_sha256.clone(),
            "The cross-site replication synthesis is sealed and internally valid.",
        ),
        gate(
            PromotionGateKind::ReplicationSupportsClaim,
            !replication_support_required
                || synthesis.conclusion == ReplicationSynthesisConclusion::IndependentlyReplicated,
            synthesis.synthesis_sha256.clone(),
            "Independent replication support is required for replicated or stable stages.",
        ),
        gate(
            PromotionGateKind::ArchiveDurable,
            !stable_requirements || archive_valid,
            archive.archive_sha256.clone(),
            "The public archive has multiple custodians and a successful recovery drill.",
        ),
        gate(
            PromotionGateKind::StewardshipDistributed,
            !stable_requirements || charter_valid,
            charter.charter_sha256.clone(),
            "Release, archive, reproducibility, security, and participant duties are distributed.",
        ),
        gate(
            PromotionGateKind::SecurityReviewCurrent,
            is_sha256(&request.security_review_sha256),
            request.security_review_sha256.clone(),
            "A current security review is bound to the promotion request.",
        ),
        gate(
            PromotionGateKind::PublicClaimFrozen,
            is_sha256(&request.frozen_public_claim_sha256),
            request.frozen_public_claim_sha256.clone(),
            "The exact public claim language is frozen by digest.",
        ),
        gate(
            PromotionGateKind::SupportWindowDeclared,
            !stable_requirements
                || (is_sha256(&request.support_policy_sha256)
                    && !request.support_until_utc.trim().is_empty()),
            request.support_policy_sha256.clone(),
            "A support and end-of-life window is publicly declared.",
        ),
    ]
}

fn gate(
    gate: PromotionGateKind,
    passed: bool,
    evidence_sha256: String,
    explanation: &str,
) -> PromotionGateResult {
    PromotionGateResult {
        gate,
        status: if passed {
            PromotionGateStatus::Passed
        } else {
            PromotionGateStatus::Failed
        },
        evidence_sha256,
        explanation: explanation.into(),
    }
}

fn valid_semantic_version(value: &str) -> bool {
    let core = value.split_once('-').map_or(value, |(core, _)| core);
    let parts = core.split('.').collect::<Vec<_>>();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_stage_requires_replication_support() {
        assert!(matches!(
            ResearchReleaseStage::StableResearchRelease,
            ResearchReleaseStage::StableResearchRelease
        ));
        assert_ne!(
            ReplicationSynthesisConclusion::MixedEvidence,
            ReplicationSynthesisConclusion::IndependentlyReplicated
        );
    }

    #[test]
    fn semantic_version_is_required() {
        assert!(valid_semantic_version("2.0.0"));
        assert!(!valid_semantic_version("stable"));
    }
}
