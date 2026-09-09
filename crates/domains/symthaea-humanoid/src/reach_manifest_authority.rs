// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Final public Reach authority facade.
//!
//! Operational motor authority is issued only from the strongest manifest-bound
//! qualification artifact and an operator approval bound to that exact policy.
//! The resulting skill receipt records the strongest SHA-256 artifact identities,
//! rather than validating a strong wrapper and then auditing only a weaker lower
//! qualification identity.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::execution_authority_scope::{
    HumanoidExecutionAuthorityScopeIssueFailure, HumanoidScopedSkillAuthorityReceipt,
    scope_verified_operational_authority_receipt,
};
use crate::qualification::HumanoidQualificationSubject;
use crate::skill_authority_receipt::{
    HumanoidAuthoritySourceSnapshot, HumanoidSkillAuthorityEvidence,
    HumanoidSkillAuthorityReceiptIssueFailure, issue_humanoid_skill_authority_receipt,
};
use crate::skill_permit::HumanoidSkillExecutionPermit;
use crate::types::{ActuationMode, HumanoidTask};

pub use crate::reach_manifest_bound_promotion::{
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

pub const HUMANOID_REACH_MANIFEST_AUTHORITY_APPROVAL_SCHEMA_VERSION: u32 = 1;

/// Structural upstream-approval binding for the strongest manifest-aware Reach
/// policy. Authentication/signatures/revocation remain the responsibility of the
/// upstream authority service (for example Xenia/Mycelix).
pub struct HumanoidReachManifestAuthorityApproval {
    schema_version: u32,
    approval_id: String,
    subject_digest: HumanoidEvidenceDigest,
    policy_digest: HumanoidEvidenceDigest,
    operational_scope_id: String,
    approved_at_s: f64,
    valid_until_s: f64,
    operator_source: HumanoidAuthoritySourceSnapshot,
    approval_digest: HumanoidEvidenceDigest,
}

impl std::fmt::Debug for HumanoidReachManifestAuthorityApproval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachManifestAuthorityApproval")
            .field("approval_id", &self.approval_id)
            .field("policy_digest", &self.policy_digest)
            .field("operational_scope_id", &self.operational_scope_id)
            .field("approved_at_s", &self.approved_at_s)
            .field("valid_until_s", &self.valid_until_s)
            .field("approval_digest", &self.approval_digest)
            .finish()
    }
}

impl HumanoidReachManifestAuthorityApproval {
    #[allow(clippy::too_many_arguments)]
    pub fn bind_upstream(
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachManifestOperationalPolicy,
        operational_scope_id: impl Into<String>,
        approval_id: impl Into<String>,
        operator_source: HumanoidAuthoritySourceSnapshot,
        approved_at_s: f64,
        valid_until_s: f64,
    ) -> Option<Self> {
        let operational_scope_id = operational_scope_id.into();
        let approval_id = approval_id.into();
        if !policy.validate_for(subject)
            || !valid_id(&operational_scope_id)
            || !valid_id(&approval_id)
            || !approved_at_s.is_finite()
            || approved_at_s < 0.0
            || !valid_until_s.is_finite()
            || valid_until_s < approved_at_s
            || !operator_source.validate_at(approved_at_s)
            || valid_until_s > operator_source.valid_until_s
        {
            return None;
        }
        let mut approval = Self {
            schema_version: HUMANOID_REACH_MANIFEST_AUTHORITY_APPROVAL_SCHEMA_VERSION,
            approval_id,
            subject_digest: digest_subject(subject)?,
            policy_digest: policy.policy_digest(),
            operational_scope_id,
            approved_at_s,
            valid_until_s,
            operator_source,
            approval_digest: HumanoidEvidenceDigest::ZERO,
        };
        approval.approval_digest = digest_approval(&approval);
        approval.validate_at(subject, policy, approved_at_s).then_some(approval)
    }

    pub const fn approval_digest(&self) -> HumanoidEvidenceDigest {
        self.approval_digest
    }

    pub fn operational_scope_id(&self) -> &str {
        &self.operational_scope_id
    }

    pub fn validate_at(
        &self,
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachManifestOperationalPolicy,
        now_s: f64,
    ) -> bool {
        self.schema_version == HUMANOID_REACH_MANIFEST_AUTHORITY_APPROVAL_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && policy.validate_for(subject)
            && self.policy_digest == policy.policy_digest()
            && valid_id(&self.approval_id)
            && valid_id(&self.operational_scope_id)
            && self.approved_at_s.is_finite()
            && self.valid_until_s.is_finite()
            && now_s.is_finite()
            && now_s >= self.approved_at_s
            && now_s <= self.valid_until_s
            && self.operator_source.validate_at(now_s)
            && self.valid_until_s <= self.operator_source.valid_until_s
            && !self.approval_digest.is_zero()
            && self.approval_digest == digest_approval(self)
    }

    fn derived_operator_source(
        &self,
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachManifestOperationalPolicy,
        now_s: f64,
    ) -> Option<HumanoidAuthoritySourceSnapshot> {
        if !self.validate_at(subject, policy, now_s) {
            return None;
        }
        let source = HumanoidAuthoritySourceSnapshot {
            evidence_id: format!(
                "reach-operator-manifest-sha256:{}",
                self.approval_digest.to_hex()
            ),
            scale: self.operator_source.scale,
            evaluated_at_s: self.operator_source.evaluated_at_s.max(self.approved_at_s),
            valid_until_s: self.operator_source.valid_until_s.min(self.valid_until_s),
        };
        source.validate_at(now_s).then_some(source)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachManifestAuthorityIssueFailure {
    InvalidQualification,
    InvalidApproval,
    PermitSubjectMismatch,
    InvalidTime,
    InvalidRequestedValidity,
    Inner(HumanoidSkillAuthorityReceiptIssueFailure),
    Scope(HumanoidExecutionAuthorityScopeIssueFailure),
}

/// Mint the actual purpose-scoped motor authority receipt from the strongest Reach
/// evidence identities.
///
/// `now_s` and `now_unix_millis` are an explicit caller-supplied clock correlation:
/// they must describe the same issuance instant. The humanoid domain does not own
/// a trusted wall-clock/monotonic synchronization service, so this function does
/// not claim to authenticate that correlation. `requested_valid_until_s` is mapped
/// forward by elapsed time into Unix milliseconds; the strongest qualification
/// artifact must remain valid at that mapped future instant or issuance fails
/// closed. A future clock-attestation service can bind the correlation without
/// changing this capability API.
#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_manifest_authority_receipt(
    subject: &HumanoidQualificationSubject,
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachManifestOperationalArtifact,
    policy: &HumanoidReachManifestOperationalPolicy,
    operator_approval: &HumanoidReachManifestAuthorityApproval,
    physical: HumanoidAuthoritySourceSnapshot,
    epistemic: HumanoidAuthoritySourceSnapshot,
    cognitive: HumanoidAuthoritySourceSnapshot,
    now_s: f64,
    now_unix_millis: u64,
    requested_valid_until_s: f64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidReachManifestAuthorityIssueFailure> {
    if !now_s.is_finite() || now_s < 0.0 || now_unix_millis == 0 {
        return Err(HumanoidReachManifestAuthorityIssueFailure::InvalidTime);
    }
    let future_unix_millis = map_monotonic_expiry_to_unix(
        now_s,
        now_unix_millis,
        requested_valid_until_s,
    )
    .ok_or(HumanoidReachManifestAuthorityIssueFailure::InvalidRequestedValidity)?;

    if !qualification.validate_at(subject, policy, now_unix_millis)
        || !qualification.validate_at(subject, policy, future_unix_millis)
    {
        return Err(HumanoidReachManifestAuthorityIssueFailure::InvalidQualification);
    }
    if !operator_approval.validate_at(subject, policy, now_s)
        || !operator_approval.validate_at(subject, policy, requested_valid_until_s)
    {
        return Err(HumanoidReachManifestAuthorityIssueFailure::InvalidApproval);
    }

    let subjects = permit
        .requirements()
        .iter()
        .map(|requirement| requirement.request.subject_fingerprint)
        .collect::<Vec<_>>();
    if subject.task != HumanoidTask::Reach
        || subjects.as_slice() != &[subject.fingerprint()]
        || permit.morphology() != subject.morphology
        || permit.actuation_mode() != subject.actuation_mode
        || permit.backend_profile_id() != subject.backend_profile_id.as_str()
    {
        return Err(HumanoidReachManifestAuthorityIssueFailure::PermitSubjectMismatch);
    }

    let operator = operator_approval
        .derived_operator_source(subject, policy, now_s)
        .ok_or(HumanoidReachManifestAuthorityIssueFailure::InvalidApproval)?;
    let qualification_source = HumanoidAuthoritySourceSnapshot {
        evidence_id: format!(
            "reach-qualified-manifest-sha256:{}",
            qualification.artifact_digest().to_hex()
        ),
        scale: 1.0,
        evaluated_at_s: now_s,
        valid_until_s: requested_valid_until_s,
    };
    if !qualification_source.validate_at(now_s) {
        return Err(HumanoidReachManifestAuthorityIssueFailure::InvalidRequestedValidity);
    }

    let inner = issue_humanoid_skill_authority_receipt(
        permit,
        HumanoidSkillAuthorityEvidence {
            operator,
            qualification: qualification_source,
            physical,
            epistemic,
            cognitive,
        },
        now_s,
    )
    .map_err(HumanoidReachManifestAuthorityIssueFailure::Inner)?;

    scope_verified_operational_authority_receipt(
        inner,
        permit,
        operator_approval.operational_scope_id().to_string(),
        now_s,
    )
    .map_err(HumanoidReachManifestAuthorityIssueFailure::Scope)
}

fn map_monotonic_expiry_to_unix(
    now_s: f64,
    now_unix_millis: u64,
    requested_valid_until_s: f64,
) -> Option<u64> {
    if !requested_valid_until_s.is_finite() || requested_valid_until_s <= now_s {
        return None;
    }
    let delta_millis = (requested_valid_until_s - now_s) * 1000.0;
    if !delta_millis.is_finite() || delta_millis <= 0.0 || delta_millis > u64::MAX as f64 {
        return None;
    }
    let delta_millis = delta_millis.ceil() as u64;
    now_unix_millis.checked_add(delta_millis)
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() || subject.task != HumanoidTask::Reach {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("reach.manifest-authority-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn digest_approval(approval: &HumanoidReachManifestAuthorityApproval) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.manifest-authority-approval.v1");
    h.u32(approval.schema_version)
        .string(&approval.approval_id)
        .digest(approval.subject_digest)
        .digest(approval.policy_digest)
        .string(&approval.operational_scope_id)
        .f64(approval.approved_at_s)
        .f64(approval.valid_until_s)
        .string(&approval.operator_source.evidence_id)
        .f32(approval.operator_source.scale)
        .f64(approval.operator_source.evaluated_at_s)
        .f64(approval.operator_source.valid_until_s);
    h.finish()
}

fn task_id(task: HumanoidTask) -> u64 {
    match task {
        HumanoidTask::Stand => 1,
        HumanoidTask::Walk => 2,
        HumanoidTask::Run => 3,
        HumanoidTask::Reach => 4,
        HumanoidTask::Grasp => 5,
    }
}

fn actuation_mode_id(mode: ActuationMode) -> u64 {
    match mode {
        ActuationMode::NormalizedTorque => 1,
        ActuationMode::TorqueNewtonMetres => 2,
        ActuationMode::NormalizedPosition => 3,
        ActuationMode::PositionTargetRadians => 4,
    }
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expiry_mapping_rounds_up_and_rejects_zero_window() {
        assert_eq!(map_monotonic_expiry_to_unix(10.0, 1_000, 10.001), Some(1_001));
        assert_eq!(map_monotonic_expiry_to_unix(10.0, 1_000, 10.0), None);
    }
}
