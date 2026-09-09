// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Final operational Reach authority with precommitted live-evidence policies.
//!
//! Behavioral qualification alone does not define which live physical-health or
//! estimator configuration may be trusted at runtime. Likewise, a valid signature
//! is insufficient if operations did not precommit which verifier policy/keyset is
//! authoritative. This facade freezes both before motor authority can be minted.

use crate::authority_evidence_policy::{
    HumanoidPolicyVerifiedEpistemicAuthorityEvidence,
    HumanoidPolicyVerifiedPhysicalAuthorityEvidence,
};
use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::execution_authority_scope::{
    HumanoidExecutionAuthorityScopeIssueFailure, HumanoidScopedSkillAuthorityReceipt,
    scope_verified_operational_authority_receipt,
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

pub const HUMANOID_REACH_RUNTIME_EVIDENCE_POLICY_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_RUNTIME_AUTHORITY_APPROVAL_SCHEMA_VERSION: u32 = 1;

/// Exact live-evidence and trust-root policy for operational Reach.
pub struct HumanoidReachRuntimeEvidencePolicy {
    schema_version: u32,
    policy_id: String,
    subject_digest: HumanoidEvidenceDigest,
    base: HumanoidReachManifestOperationalPolicy,
    physical_evidence_policy_digest: HumanoidEvidenceDigest,
    epistemic_evidence_policy_digest: HumanoidEvidenceDigest,
    operator_verifier_digest: HumanoidEvidenceDigest,
    physical_verifier_digest: HumanoidEvidenceDigest,
    epistemic_verifier_digest: HumanoidEvidenceDigest,
    cognitive_verifier_digest: HumanoidEvidenceDigest,
    policy_digest: HumanoidEvidenceDigest,
}

impl std::fmt::Debug for HumanoidReachRuntimeEvidencePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachRuntimeEvidencePolicy")
            .field("policy_id", &self.policy_id)
            .field("base_policy_digest", &self.base.policy_digest())
            .field("physical_evidence_policy_digest", &self.physical_evidence_policy_digest)
            .field("epistemic_evidence_policy_digest", &self.epistemic_evidence_policy_digest)
            .field("policy_digest", &self.policy_digest)
            .finish()
    }
}

impl HumanoidReachRuntimeEvidencePolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        policy_id: impl Into<String>,
        base: HumanoidReachManifestOperationalPolicy,
        physical_evidence_policy_digest: HumanoidEvidenceDigest,
        epistemic_evidence_policy_digest: HumanoidEvidenceDigest,
        operator_verifier_digest: HumanoidEvidenceDigest,
        physical_verifier_digest: HumanoidEvidenceDigest,
        epistemic_verifier_digest: HumanoidEvidenceDigest,
        cognitive_verifier_digest: HumanoidEvidenceDigest,
    ) -> Option<Self> {
        let mut value = Self {
            schema_version: HUMANOID_REACH_RUNTIME_EVIDENCE_POLICY_SCHEMA_VERSION,
            policy_id: policy_id.into(),
            subject_digest: digest_subject(subject)?,
            base,
            physical_evidence_policy_digest,
            epistemic_evidence_policy_digest,
            operator_verifier_digest,
            physical_verifier_digest,
            epistemic_verifier_digest,
            cognitive_verifier_digest,
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

    pub const fn physical_evidence_policy_digest(&self) -> HumanoidEvidenceDigest {
        self.physical_evidence_policy_digest
    }

    pub const fn epistemic_evidence_policy_digest(&self) -> HumanoidEvidenceDigest {
        self.epistemic_evidence_policy_digest
    }

    pub const fn operator_verifier_digest(&self) -> HumanoidEvidenceDigest {
        self.operator_verifier_digest
    }

    pub const fn physical_verifier_digest(&self) -> HumanoidEvidenceDigest {
        self.physical_verifier_digest
    }

    pub const fn epistemic_verifier_digest(&self) -> HumanoidEvidenceDigest {
        self.epistemic_verifier_digest
    }

    pub const fn cognitive_verifier_digest(&self) -> HumanoidEvidenceDigest {
        self.cognitive_verifier_digest
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
            && !self.physical_evidence_policy_digest.is_zero()
            && !self.epistemic_evidence_policy_digest.is_zero()
            && !self.operator_verifier_digest.is_zero()
            && !self.physical_verifier_digest.is_zero()
            && !self.epistemic_verifier_digest.is_zero()
            && !self.cognitive_verifier_digest.is_zero()
    }
}

/// Operator approval for one exact runtime-evidence policy and deployment scope.
pub struct HumanoidReachRuntimeAuthorityApproval {
    schema_version: u32,
    approval_id: String,
    subject_digest: HumanoidEvidenceDigest,
    runtime_policy_digest: HumanoidEvidenceDigest,
    operational_scope_id: String,
    approved_at_s: f64,
    valid_until_s: f64,
    operator_verification_digest: HumanoidEvidenceDigest,
    operator_source: HumanoidAuthoritySourceSnapshot,
    approval_digest: HumanoidEvidenceDigest,
}

impl std::fmt::Debug for HumanoidReachRuntimeAuthorityApproval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachRuntimeAuthorityApproval")
            .field("approval_id", &self.approval_id)
            .field("runtime_policy_digest", &self.runtime_policy_digest)
            .field("operational_scope_id", &self.operational_scope_id)
            .field("operator_verification_digest", &self.operator_verification_digest)
            .field("approval_digest", &self.approval_digest)
            .finish()
    }
}

impl HumanoidReachRuntimeAuthorityApproval {
    #[allow(clippy::too_many_arguments)]
    pub fn bind_verified_operator(
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachRuntimeEvidencePolicy,
        operational_scope_id: impl Into<String>,
        approval_id: impl Into<String>,
        operator: &HumanoidVerifiedAuthoritySource,
        approved_at_s: f64,
        valid_until_s: f64,
    ) -> Option<Self> {
        let operational_scope_id = operational_scope_id.into();
        let approval_id = approval_id.into();
        if !policy.validate_for(subject)
            || !valid_id(&operational_scope_id)
            || !valid_id(&approval_id)
            || !operator.validate_for(subject, HumanoidVerifiedAuthorityKind::Operator, approved_at_s)
            || operator.verifier_digest() != policy.operator_verifier_digest
            || !valid_until_s.is_finite()
            || valid_until_s < approved_at_s
            || valid_until_s > operator.valid_until_s()
        {
            return None;
        }
        let mut value = Self {
            schema_version: HUMANOID_REACH_RUNTIME_AUTHORITY_APPROVAL_SCHEMA_VERSION,
            approval_id,
            subject_digest: digest_subject(subject)?,
            runtime_policy_digest: policy.policy_digest,
            operational_scope_id,
            approved_at_s,
            valid_until_s,
            operator_verification_digest: operator.verification_digest(),
            operator_source: operator.source_snapshot(),
            approval_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.approval_digest = digest_runtime_approval(&value);
        value.validate_at(subject, policy, approved_at_s).then_some(value)
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
        policy: &HumanoidReachRuntimeEvidencePolicy,
        now_s: f64,
    ) -> bool {
        self.schema_version == HUMANOID_REACH_RUNTIME_AUTHORITY_APPROVAL_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && policy.validate_for(subject)
            && self.runtime_policy_digest == policy.policy_digest
            && valid_id(&self.approval_id)
            && valid_id(&self.operational_scope_id)
            && self.operator_verification_digest != HumanoidEvidenceDigest::ZERO
            && now_s.is_finite()
            && now_s >= self.approved_at_s
            && now_s <= self.valid_until_s
            && self.operator_source.validate_at(now_s)
            && self.valid_until_s <= self.operator_source.valid_until_s
            && !self.approval_digest.is_zero()
            && self.approval_digest == digest_runtime_approval(self)
    }

    fn derived_operator_source(
        &self,
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachRuntimeEvidencePolicy,
        now_s: f64,
    ) -> Option<HumanoidAuthoritySourceSnapshot> {
        if !self.validate_at(subject, policy, now_s) {
            return None;
        }
        let source = HumanoidAuthoritySourceSnapshot {
            evidence_id: format!(
                "reach-operator-runtime-policy-sha256:{}",
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
pub enum HumanoidReachRuntimeAuthorityIssueFailure {
    InvalidPolicy,
    InvalidQualification,
    InvalidApproval,
    InvalidPhysicalEvidencePolicy,
    InvalidEpistemicEvidencePolicy,
    InvalidPhysicalVerifier,
    InvalidEpistemicVerifier,
    InvalidCognitiveVerifier,
    InvalidCognitiveSource,
    PermitSubjectMismatch,
    InvalidTime,
    InvalidRequestedValidity,
    Inner(HumanoidSkillAuthorityReceiptIssueFailure),
    Scope(HumanoidExecutionAuthorityScopeIssueFailure),
}

/// Mint operational motor authority only when live evidence and verifier identities
/// match the precommitted runtime policy.
#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_runtime_authority_receipt(
    subject: &HumanoidQualificationSubject,
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachManifestOperationalArtifact,
    policy: &HumanoidReachRuntimeEvidencePolicy,
    operator_approval: &HumanoidReachRuntimeAuthorityApproval,
    physical: &HumanoidPolicyVerifiedPhysicalAuthorityEvidence,
    epistemic: &HumanoidPolicyVerifiedEpistemicAuthorityEvidence,
    cognitive: &HumanoidVerifiedAuthoritySource,
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
    if !operator_approval.validate_at(subject, policy, now_s)
        || !operator_approval.validate_at(subject, policy, requested_valid_until_s)
    {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidApproval);
    }

    if !physical.validate_for(subject, now_s)
        || physical.evidence_policy_digest() != policy.physical_evidence_policy_digest
    {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidPhysicalEvidencePolicy);
    }
    if !epistemic.validate_for(subject, now_s)
        || epistemic.evidence_policy_digest() != policy.epistemic_evidence_policy_digest
    {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidEpistemicEvidencePolicy);
    }
    if physical.verifier_digest() != policy.physical_verifier_digest {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidPhysicalVerifier);
    }
    if epistemic.verifier_digest() != policy.epistemic_verifier_digest {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidEpistemicVerifier);
    }
    if !cognitive.validate_for(subject, HumanoidVerifiedAuthorityKind::Cognitive, now_s) {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidCognitiveSource);
    }
    if cognitive.verifier_digest() != policy.cognitive_verifier_digest {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidCognitiveVerifier);
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
        || requested_valid_until_s > physical.valid_until_s()
        || requested_valid_until_s > epistemic.valid_until_s()
        || requested_valid_until_s > cognitive.valid_until_s()
    {
        return Err(HumanoidReachRuntimeAuthorityIssueFailure::InvalidRequestedValidity);
    }

    let operator = operator_approval
        .derived_operator_source(subject, policy, now_s)
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

    let inner = issue_humanoid_skill_authority_receipt(
        permit,
        HumanoidSkillAuthorityEvidence {
            operator,
            qualification: qualification_source,
            physical: physical.inner().source_snapshot(),
            epistemic: epistemic.inner().source_snapshot(),
            cognitive: cognitive.source_snapshot(),
        },
        now_s,
    )
    .map_err(HumanoidReachRuntimeAuthorityIssueFailure::Inner)?;

    scope_verified_operational_authority_receipt(
        inner,
        permit,
        operator_approval.operational_scope_id.clone(),
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
    let mut h = HumanoidEvidenceHasher::new("reach.runtime-evidence-policy-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn digest_runtime_policy(policy: &HumanoidReachRuntimeEvidencePolicy) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.runtime-evidence-policy.v1");
    h.u32(policy.schema_version)
        .string(&policy.policy_id)
        .digest(policy.subject_digest)
        .digest(policy.base.policy_digest())
        .digest(policy.physical_evidence_policy_digest)
        .digest(policy.epistemic_evidence_policy_digest)
        .digest(policy.operator_verifier_digest)
        .digest(policy.physical_verifier_digest)
        .digest(policy.epistemic_verifier_digest)
        .digest(policy.cognitive_verifier_digest);
    h.finish()
}

fn digest_runtime_approval(approval: &HumanoidReachRuntimeAuthorityApproval) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.runtime-authority-approval.v1");
    h.u32(approval.schema_version)
        .string(&approval.approval_id)
        .digest(approval.subject_digest)
        .digest(approval.runtime_policy_digest)
        .string(&approval.operational_scope_id)
        .f64(approval.approved_at_s)
        .f64(approval.valid_until_s)
        .digest(approval.operator_verification_digest)
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
