// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Final operational Reach authority built on the task-generic live-authority kernel.
//!
//! Reach owns behavioral qualification. The generic kernel owns live Physical,
//! Epistemic and Cognitive evidence, verifier trust roots and scoped Operator
//! approval. This separation lets later skills reuse the same live trust boundary.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::execution_authority_scope::{
    HumanoidExecutionAuthorityScopeIssueFailure, HumanoidScopedSkillAuthorityReceipt,
    scope_verified_operational_authority_receipt,
};
use crate::live_authority_kernel::{
    HumanoidLiveAuthorityApproval, HumanoidLiveAuthorityPolicy,
    HumanoidVerifiedLiveAuthorityEvidence,
};
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_manifest_authority::{
    HumanoidReachManifestOperationalArtifact, HumanoidReachManifestOperationalPolicy,
};
use crate::skill_authority_receipt::{
    HumanoidAuthoritySourceSnapshot, HumanoidSkillAuthorityEvidence,
    HumanoidSkillAuthorityReceiptIssueFailure, issue_humanoid_skill_authority_receipt,
};
use crate::skill_permit::HumanoidSkillExecutionPermit;
use crate::types::{ActuationMode, HumanoidTask};
use crate::verified_authority_source::HumanoidVerifiedAuthoritySource;

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

/// Schema v3 replaces Reach-local live-evidence fields with one generic live policy digest.
pub const HUMANOID_REACH_RUNTIME_EVIDENCE_POLICY_SCHEMA_VERSION: u32 = 3;

/// Reach-specific operational policy = behavioral qualification + generic live authority policy.
pub struct HumanoidReachRuntimeEvidencePolicy {
    schema_version: u32,
    policy_id: String,
    subject_digest: HumanoidEvidenceDigest,
    base: HumanoidReachManifestOperationalPolicy,
    live: HumanoidLiveAuthorityPolicy,
    policy_digest: HumanoidEvidenceDigest,
}

impl std::fmt::Debug for HumanoidReachRuntimeEvidencePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachRuntimeEvidencePolicy")
            .field("policy_id", &self.policy_id)
            .field("base_policy_digest", &self.base.policy_digest())
            .field("live_policy_digest", &self.live.policy_digest())
            .field("policy_digest", &self.policy_digest)
            .finish()
    }
}

impl HumanoidReachRuntimeEvidencePolicy {
    pub fn new(
        subject: &HumanoidQualificationSubject,
        policy_id: impl Into<String>,
        base: HumanoidReachManifestOperationalPolicy,
        live: HumanoidLiveAuthorityPolicy,
    ) -> Option<Self> {
        let mut value = Self {
            schema_version: HUMANOID_REACH_RUNTIME_EVIDENCE_POLICY_SCHEMA_VERSION,
            policy_id: policy_id.into(),
            subject_digest: digest_subject(subject)?,
            base,
            live,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        if !value.validate_shape(subject) {
            return None;
        }
        value.policy_digest = digest_runtime_policy(&value);
        value.validate_for(subject).then_some(value)
    }

    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }

    pub fn base_policy(&self) -> &HumanoidReachManifestOperationalPolicy {
        &self.base
    }

    pub fn live_policy(&self) -> &HumanoidLiveAuthorityPolicy {
        &self.live
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.validate_shape(subject)
            && !self.policy_digest.is_zero()
            && self.policy_digest == digest_runtime_policy(self)
    }

    fn validate_shape(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_REACH_RUNTIME_EVIDENCE_POLICY_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && valid_id(&self.policy_id)
            && self.base.validate_for(subject)
            && self.live.validate_for(subject)
    }
}

/// Convenience binder that prevents callers from supplying the wrong operational
/// policy digest to the generic approval kernel.
#[allow(clippy::too_many_arguments)]
pub fn bind_humanoid_reach_runtime_operator_approval(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachRuntimeEvidencePolicy,
    operational_scope_id: impl Into<String>,
    approval_id: impl Into<String>,
    operator: &HumanoidVerifiedAuthoritySource,
    approved_at_s: f64,
    valid_until_s: f64,
) -> Option<HumanoidLiveAuthorityApproval> {
    if !policy.validate_for(subject) {
        return None;
    }
    HumanoidLiveAuthorityApproval::bind_verified_operator(
        subject,
        policy.live_policy(),
        policy.policy_digest(),
        operational_scope_id,
        approval_id,
        operator,
        approved_at_s,
        valid_until_s,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachRuntimeAuthorityIssueFailure {
    InvalidPolicy,
    InvalidQualification,
    InvalidApproval,
    InvalidLiveEvidence,
    PermitSubjectMismatch,
    InvalidTime,
    InvalidRequestedValidity,
    Inner(HumanoidSkillAuthorityReceiptIssueFailure),
    Scope(HumanoidExecutionAuthorityScopeIssueFailure),
}

/// Mint operational Reach authority from task-specific qualification plus the
/// generic, policy-bound live authority bundle.
#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_runtime_authority_receipt(
    subject: &HumanoidQualificationSubject,
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachManifestOperationalArtifact,
    policy: &HumanoidReachRuntimeEvidencePolicy,
    operator_approval: &HumanoidLiveAuthorityApproval,
    live_evidence: &HumanoidVerifiedLiveAuthorityEvidence,
    now_s: f64,
    now_unix_millis: u64,
    requested_valid_until_s: f64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidReachRuntimeAuthorityIssueFailure> {
    if !policy.validate_for(subject) {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidPolicy);
    }
    if !now_s.is_finite() || now_s < 0.0 || now_unix_millis == 0 {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidTime);
    }

    let future_unix_millis = map_monotonic_expiry_to_unix(
        now_s,
        now_unix_millis,
        requested_valid_until_s,
    )
    .ok_or(HumanoidReachRuntimeAuthorityIssueFailure::InvalidRequestedValidity)?;

    if !qualification.validate_at(subject, policy.base_policy(), now_unix_millis)
        || !qualification.validate_at(subject, policy.base_policy(), future_unix_millis)
    {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidQualification);
    }

    if !operator_approval.validate_at(
        subject,
        policy.live_policy(),
        policy.policy_digest(),
        now_s,
    ) || !operator_approval.validate_at(
        subject,
        policy.live_policy(),
        policy.policy_digest(),
        requested_valid_until_s,
    ) {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidApproval);
    }

    if !live_evidence.validate_at(subject, policy.live_policy(), now_s)
        || !live_evidence.validate_at(
            subject,
            policy.live_policy(),
            requested_valid_until_s,
        )
    {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidLiveEvidence);
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
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::PermitSubjectMismatch);
    }

    if !requested_valid_until_s.is_finite()
        || requested_valid_until_s <= now_s
        || requested_valid_until_s > live_evidence.valid_until_s()
    {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidRequestedValidity);
    }

    let operator = operator_approval
        .derived_operator_source(
            subject,
            policy.live_policy(),
            policy.policy_digest(),
            now_s,
        )
        .ok_or(HumanoidReachRuntimeAuthorityIssueFailure::InvalidApproval)?;

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
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidRequestedValidity);
    }

    let (physical, epistemic, cognitive) = live_evidence.source_snapshots();
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
    .map_err(HumanoidReachRuntimeAuthorityIssueFailure::Inner)?;

    scope_verified_operational_authority_receipt(
        inner,
        permit,
        operator_approval.operational_scope_id().to_string(),
        now_s,
    )
    .map_err(HumanoidReachRuntimeAuthorityIssueFailure::Scope)
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
    now_unix_millis.checked_add(delta_millis.ceil() as u64)
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() || subject.task != HumanoidTask::Reach {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("reach.runtime-evidence-policy-subject.v3");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn digest_runtime_policy(policy: &HumanoidReachRuntimeEvidencePolicy) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.runtime-evidence-policy.v3");
    h.u32(policy.schema_version)
        .string(&policy.policy_id)
        .digest(policy.subject_digest)
        .digest(policy.base.policy_digest())
        .digest(policy.live.policy_digest());
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
