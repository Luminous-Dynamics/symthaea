// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Final attested operational Reach authority facade.
//!
//! The manifest-aware Reach stack proves what behavior was qualified, under which
//! exact protocols, perturbations, spatial goals, authority decisions and
//! deployment scope. This layer closes the remaining public source-provenance seam:
//! operator, physical, epistemic and cognitive authority must arrive as opaque
//! verifier-produced attestations rather than caller-authored strings/scales.

use crate::authority_source_attestation::{
    HumanoidAuthoritySourceKind, HumanoidVerifiedAuthoritySource,
};
use crate::execution_authority_scope::HumanoidScopedSkillAuthorityReceipt;
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_manifest_authority::{
    HumanoidReachManifestAuthorityApproval, HumanoidReachManifestAuthorityIssueFailure,
    issue_humanoid_reach_manifest_authority_receipt,
};
use crate::skill_permit::HumanoidSkillExecutionPermit;

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

/// Opaque operator-policy approval whose source authority was already authenticated
/// through the transport-neutral attestation verifier boundary.
pub struct HumanoidReachAttestedAuthorityApproval {
    operator_statement_digest: crate::evidence_digest::HumanoidEvidenceDigest,
    operator_verification_digest: crate::evidence_digest::HumanoidEvidenceDigest,
    inner: HumanoidReachManifestAuthorityApproval,
}

impl std::fmt::Debug for HumanoidReachAttestedAuthorityApproval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachAttestedAuthorityApproval")
            .field("operator_statement_digest", &self.operator_statement_digest)
            .field("operator_verification_digest", &self.operator_verification_digest)
            .field("approval_digest", &self.inner.approval_digest())
            .field("operational_scope_id", &self.inner.operational_scope_id())
            .finish()
    }
}

impl HumanoidReachAttestedAuthorityApproval {
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
        if !operator.validate_for(
            subject,
            HumanoidAuthoritySourceKind::Operator,
            approved_at_s,
        ) || valid_until_s > operator.valid_until_s()
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
pub enum HumanoidReachAttestedAuthorityIssueFailure {
    InvalidPhysicalAttestation,
    InvalidEpistemicAttestation,
    InvalidCognitiveAttestation,
    Lower(HumanoidReachManifestAuthorityIssueFailure),
}

/// Preferred final motor-authority issuer. Raw `HumanoidAuthoritySourceSnapshot`
/// values are deliberately absent from this public API.
#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_attested_authority_receipt(
    subject: &HumanoidQualificationSubject,
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachManifestOperationalArtifact,
    policy: &HumanoidReachManifestOperationalPolicy,
    operator_approval: &HumanoidReachAttestedAuthorityApproval,
    physical: &HumanoidVerifiedAuthoritySource,
    epistemic: &HumanoidVerifiedAuthoritySource,
    cognitive: &HumanoidVerifiedAuthoritySource,
    now_s: f64,
    now_unix_millis: u64,
    requested_valid_until_s: f64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidReachAttestedAuthorityIssueFailure> {
    if !physical.validate_for(subject, HumanoidAuthoritySourceKind::Physical, now_s) {
        return Err(HumanoidReachAttestedAuthorityIssueFailure::InvalidPhysicalAttestation);
    }
    if !epistemic.validate_for(subject, HumanoidAuthoritySourceKind::Epistemic, now_s) {
        return Err(HumanoidReachAttestedAuthorityIssueFailure::InvalidEpistemicAttestation);
    }
    if !cognitive.validate_for(subject, HumanoidAuthoritySourceKind::Cognitive, now_s) {
        return Err(HumanoidReachAttestedAuthorityIssueFailure::InvalidCognitiveAttestation);
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
    .map_err(HumanoidReachAttestedAuthorityIssueFailure::Lower)
}
