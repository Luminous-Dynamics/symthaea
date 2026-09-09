// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Final public Reach authority facade using verified source decisions.
//!
//! The lower manifest-aware layer proves behavioral qualification and exact
//! perturbation/protocol identity. Operator and cognitive authority must originate
//! from the strong transport-neutral verifier boundary. Physical and epistemic
//! authority are stricter still: their signed claims must match canonical runtime
//! evidence artifacts before they can reach this motor boundary.

use crate::execution_authority_scope::HumanoidScopedSkillAuthorityReceipt;
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_manifest_authority::{
    HumanoidReachManifestAuthorityApproval, HumanoidReachManifestAuthorityIssueFailure,
    issue_humanoid_reach_manifest_authority_receipt,
};
use crate::skill_permit::HumanoidSkillExecutionPermit;
use crate::verified_authority_artifacts::{
    HumanoidVerifiedEpistemicAuthorityEvidence, HumanoidVerifiedPhysicalAuthorityEvidence,
};
use crate::verified_authority_source::{
    HumanoidVerifiedAuthorityKind, HumanoidVerifiedAuthoritySource,
};

pub use crate::reach_manifest_authority::{
    HumanoidReachBaseOperationalPolicy,
    HumanoidReachBaseStageRequirement,
    HumanoidReachEpisodeCampaignAssessment,
    HumanoidReachEpisodeCampaignFailureKind,
    HumanoidReachEpisodeCampaignPolicy,
    HumanoidReachEpisodePolicy,
    HumanoidReachEpisodeScenarioAssessment,
    HumanoidReachEpisodeScenarioFailureKind,
    HumanoidReachEpisodeScenarioRequirement,
    HumanoidReachManifestBoundEpisodeCase,
    HumanoidReachManifestBoundTrial,
    HumanoidReachManifestEpisodeStageArtifact,
    HumanoidReachManifestOperationalArtifact,
    HumanoidReachManifestOperationalPolicy,
    HumanoidReachManifestPromotionFailure,
    HumanoidReachManifestStageIssueFailure,
    HumanoidReachManifestStageRequirement,
    HumanoidReachManifestStepStageArtifact,
    HumanoidReachManifestTrialBindFailure,
    HumanoidReachScenarioManifestRequirement,
    assess_humanoid_reach_episode_campaign,
    issue_humanoid_reach_manifest_episode_stage,
    issue_humanoid_reach_manifest_step_stage,
    promote_humanoid_reach_manifest_to_operational,
};

/// Operator approval whose source is one exact verifier decision.
pub struct HumanoidReachVerifiedAuthorityApproval {
    operator_statement_digest: crate::evidence_digest::HumanoidEvidenceDigest,
    operator_verification_digest: crate::evidence_digest::HumanoidEvidenceDigest,
    inner: HumanoidReachManifestAuthorityApproval,
}

impl std::fmt::Debug for HumanoidReachVerifiedAuthorityApproval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachVerifiedAuthorityApproval")
            .field("operator_statement_digest", &self.operator_statement_digest)
            .field("operator_verification_digest", &self.operator_verification_digest)
            .field("approval_digest", &self.inner.approval_digest())
            .field("operational_scope_id", &self.inner.operational_scope_id())
            .finish()
    }
}

impl HumanoidReachVerifiedAuthorityApproval {
    #[allow(clippy::too_many_arguments)]
    pub fn bind_verified_operator(
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachManifestOperationalPolicy,
        operational_scope_id: impl Into<String>,
        approval_id: impl Into<String>,
        operator: &HumanoidVerifiedAuthoritySource,
        approved_at_s: f64,
        valid_until_s: f64,
    ) -> Option<Self> {
        if !operator.validate_for(subject, HumanoidVerifiedAuthorityKind::Operator, approved_at_s)
            || !valid_until_s.is_finite()
            || valid_until_s < approved_at_s
            || valid_until_s > operator.valid_until_s()
        {
            return None;
        }
        let inner = HumanoidReachManifestAuthorityApproval::bind_upstream(
            subject,
            policy,
            operational_scope_id,
            approval_id,
            operator.source_snapshot(),
            approved_at_s,
            valid_until_s,
        )?;
        Some(Self {
            operator_statement_digest: operator.statement_digest(),
            operator_verification_digest: operator.verification_digest(),
            inner,
        })
    }

    pub const fn operator_statement_digest(&self) -> crate::evidence_digest::HumanoidEvidenceDigest {
        self.operator_statement_digest
    }

    pub const fn operator_verification_digest(&self) -> crate::evidence_digest::HumanoidEvidenceDigest {
        self.operator_verification_digest
    }

    pub fn operational_scope_id(&self) -> &str {
        self.inner.operational_scope_id()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachVerifiedAuthorityIssueFailure {
    InvalidPhysicalEvidence,
    InvalidEpistemicEvidence,
    InvalidCognitiveSource,
    SourceValidityTooShort,
    Lower(HumanoidReachManifestAuthorityIssueFailure),
}

/// Preferred final motor-authority issuer.
///
/// Physical and epistemic sources are required to be verifier-approved canonical
/// runtime artifacts. Operator and cognitive sources are still generic verifier
/// results; qualification is internally derived from the manifest-aware Reach
/// operational artifact.
#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_verified_authority_receipt(
    subject: &HumanoidQualificationSubject,
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachManifestOperationalArtifact,
    policy: &HumanoidReachManifestOperationalPolicy,
    operator_approval: &HumanoidReachVerifiedAuthorityApproval,
    physical: &HumanoidVerifiedPhysicalAuthorityEvidence,
    epistemic: &HumanoidVerifiedEpistemicAuthorityEvidence,
    cognitive: &HumanoidVerifiedAuthoritySource,
    now_s: f64,
    now_unix_millis: u64,
    requested_valid_until_s: f64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidReachVerifiedAuthorityIssueFailure> {
    if !physical.validate_for(subject, now_s) {
        return Err(HumanoidReachVerifiedAuthorityIssueFailure::InvalidPhysicalEvidence);
    }
    if !epistemic.validate_for(subject, now_s) {
        return Err(HumanoidReachVerifiedAuthorityIssueFailure::InvalidEpistemicEvidence);
    }
    if !cognitive.validate_for(subject, HumanoidVerifiedAuthorityKind::Cognitive, now_s) {
        return Err(HumanoidReachVerifiedAuthorityIssueFailure::InvalidCognitiveSource);
    }
    if !requested_valid_until_s.is_finite()
        || requested_valid_until_s <= now_s
        || requested_valid_until_s > physical.valid_until_s()
        || requested_valid_until_s > epistemic.valid_until_s()
        || requested_valid_until_s > cognitive.valid_until_s()
    {
        return Err(HumanoidReachVerifiedAuthorityIssueFailure::SourceValidityTooShort);
    }

    issue_humanoid_reach_manifest_authority_receipt(
        subject,
        permit,
        qualification,
        policy,
        &operator_approval.inner,
        physical.source_snapshot(),
        epistemic.source_snapshot(),
        cognitive.source_snapshot(),
        now_s,
        now_unix_millis,
        requested_valid_until_s,
    )
    .map_err(HumanoidReachVerifiedAuthorityIssueFailure::Lower)
}
