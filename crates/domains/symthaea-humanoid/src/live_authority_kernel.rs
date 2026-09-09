// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Task-generic live authority kernel for humanoid skills.
//!
//! Reach was the first skill to require canonical, authenticated Physical,
//! Epistemic and Cognitive evidence plus a scoped Operator approval. Those
//! invariants are not Reach-specific. This module extracts them so Grasp, Lift,
//! Carry and later humanoid skills can reuse one trust boundary instead of
//! cloning Reach's authority machinery.

use crate::authority_evidence_policy::{
    HumanoidEpistemicAuthorityEvidencePolicy, HumanoidPhysicalAuthorityEvidencePolicy,
    HumanoidPolicyVerifiedEpistemicAuthorityEvidence,
    HumanoidPolicyVerifiedPhysicalAuthorityEvidence,
};
use crate::cognitive_authority_evidence::{
    HumanoidCognitiveAuthorityEvidencePolicy, HumanoidPolicyVerifiedCognitiveAuthorityEvidence,
};
use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::qualification::HumanoidQualificationSubject;
use crate::skill_authority_receipt::HumanoidAuthoritySourceSnapshot;
use crate::types::{ActuationMode, HumanoidTask};
use crate::verified_authority_source::{
    HumanoidVerifiedAuthorityKind, HumanoidVerifiedAuthoritySource,
};

pub const HUMANOID_LIVE_AUTHORITY_POLICY_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_LIVE_AUTHORITY_APPROVAL_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_LIVE_AUTHORITY_BUNDLE_SCHEMA_VERSION: u32 = 1;

/// Precommitted live evidence + verifier trust policy for one exact humanoid skill subject.
pub struct HumanoidLiveAuthorityPolicy {
    schema_version: u32,
    policy_id: String,
    subject_digest: HumanoidEvidenceDigest,
    physical_evidence_policy_digest: HumanoidEvidenceDigest,
    epistemic_evidence_policy_digest: HumanoidEvidenceDigest,
    cognitive_evidence_policy_digest: HumanoidEvidenceDigest,
    operator_verifier_digest: HumanoidEvidenceDigest,
    physical_verifier_digest: HumanoidEvidenceDigest,
    epistemic_verifier_digest: HumanoidEvidenceDigest,
    cognitive_verifier_digest: HumanoidEvidenceDigest,
    policy_digest: HumanoidEvidenceDigest,
}

impl std::fmt::Debug for HumanoidLiveAuthorityPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidLiveAuthorityPolicy")
            .field("policy_id", &self.policy_id)
            .field("subject_digest", &self.subject_digest)
            .field("physical_evidence_policy_digest", &self.physical_evidence_policy_digest)
            .field("epistemic_evidence_policy_digest", &self.epistemic_evidence_policy_digest)
            .field("cognitive_evidence_policy_digest", &self.cognitive_evidence_policy_digest)
            .field("policy_digest", &self.policy_digest)
            .finish()
    }
}

impl HumanoidLiveAuthorityPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        policy_id: impl Into<String>,
        physical_policy: &HumanoidPhysicalAuthorityEvidencePolicy,
        epistemic_policy: &HumanoidEpistemicAuthorityEvidencePolicy,
        cognitive_policy: &HumanoidCognitiveAuthorityEvidencePolicy,
        operator_verifier_digest: HumanoidEvidenceDigest,
        physical_verifier_digest: HumanoidEvidenceDigest,
        epistemic_verifier_digest: HumanoidEvidenceDigest,
        cognitive_verifier_digest: HumanoidEvidenceDigest,
    ) -> Option<Self> {
        if !subject.validate()
            || physical_policy.subject() != subject
            || epistemic_policy.subject() != subject
            || cognitive_policy.subject() != subject
        {
            return None;
        }
        let mut value = Self {
            schema_version: HUMANOID_LIVE_AUTHORITY_POLICY_SCHEMA_VERSION,
            policy_id: policy_id.into(),
            subject_digest: digest_subject(subject)?,
            physical_evidence_policy_digest: physical_policy.policy_digest(),
            epistemic_evidence_policy_digest: epistemic_policy.policy_digest(),
            cognitive_evidence_policy_digest: cognitive_policy.policy_digest(),
            operator_verifier_digest,
            physical_verifier_digest,
            epistemic_verifier_digest,
            cognitive_verifier_digest,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        if !value.validate_shape(subject) {
            return None;
        }
        value.policy_digest = digest_live_policy(&value);
        value.validate_for(subject).then_some(value)
    }

    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }

    pub const fn physical_evidence_policy_digest(&self) -> HumanoidEvidenceDigest {
        self.physical_evidence_policy_digest
    }

    pub const fn epistemic_evidence_policy_digest(&self) -> HumanoidEvidenceDigest {
        self.epistemic_evidence_policy_digest
    }

    pub const fn cognitive_evidence_policy_digest(&self) -> HumanoidEvidenceDigest {
        self.cognitive_evidence_policy_digest
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
            && self.policy_digest == digest_live_policy(self)
    }

    fn validate_shape(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_LIVE_AUTHORITY_POLICY_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && valid_id(&self.policy_id)
            && !self.physical_evidence_policy_digest.is_zero()
            && !self.epistemic_evidence_policy_digest.is_zero()
            && !self.cognitive_evidence_policy_digest.is_zero()
            && !self.operator_verifier_digest.is_zero()
            && !self.physical_verifier_digest.is_zero()
            && !self.epistemic_verifier_digest.is_zero()
            && !self.cognitive_verifier_digest.is_zero()
    }
}

/// Generic operator approval bound to one live policy, one skill operational policy,
/// and one deployment scope.
pub struct HumanoidLiveAuthorityApproval {
    schema_version: u32,
    approval_id: String,
    subject_digest: HumanoidEvidenceDigest,
    live_policy_digest: HumanoidEvidenceDigest,
    operational_policy_digest: HumanoidEvidenceDigest,
    operational_scope_id: String,
    approved_at_s: f64,
    valid_until_s: f64,
    operator_verification_digest: HumanoidEvidenceDigest,
    operator_source: HumanoidAuthoritySourceSnapshot,
    approval_digest: HumanoidEvidenceDigest,
}

impl std::fmt::Debug for HumanoidLiveAuthorityApproval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidLiveAuthorityApproval")
            .field("approval_id", &self.approval_id)
            .field("live_policy_digest", &self.live_policy_digest)
            .field("operational_policy_digest", &self.operational_policy_digest)
            .field("operational_scope_id", &self.operational_scope_id)
            .field("operator_verification_digest", &self.operator_verification_digest)
            .field("approval_digest", &self.approval_digest)
            .finish()
    }
}

impl HumanoidLiveAuthorityApproval {
    #[allow(clippy::too_many_arguments)]
    pub fn bind_verified_operator(
        subject: &HumanoidQualificationSubject,
        live_policy: &HumanoidLiveAuthorityPolicy,
        operational_policy_digest: HumanoidEvidenceDigest,
        operational_scope_id: impl Into<String>,
        approval_id: impl Into<String>,
        operator: &HumanoidVerifiedAuthoritySource,
        approved_at_s: f64,
        valid_until_s: f64,
    ) -> Option<Self> {
        let operational_scope_id = operational_scope_id.into();
        let approval_id = approval_id.into();
        if !live_policy.validate_for(subject)
            || operational_policy_digest.is_zero()
            || !valid_id(&operational_scope_id)
            || !valid_id(&approval_id)
            || !operator.validate_for(subject, HumanoidVerifiedAuthorityKind::Operator, approved_at_s)
            || operator.verifier_digest() != live_policy.operator_verifier_digest()
            || !valid_until_s.is_finite()
            || valid_until_s < approved_at_s
            || valid_until_s > operator.valid_until_s()
        {
            return None;
        }

        let mut value = Self {
            schema_version: HUMANOID_LIVE_AUTHORITY_APPROVAL_SCHEMA_VERSION,
            approval_id,
            subject_digest: digest_subject(subject)?,
            live_policy_digest: live_policy.policy_digest(),
            operational_policy_digest,
            operational_scope_id,
            approved_at_s,
            valid_until_s,
            operator_verification_digest: operator.verification_digest(),
            operator_source: operator.source_snapshot(),
            approval_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.approval_digest = digest_live_approval(&value);
        value
            .validate_at(subject, live_policy, operational_policy_digest, approved_at_s)
            .then_some(value)
    }

    pub const fn approval_digest(&self) -> HumanoidEvidenceDigest {
        self.approval_digest
    }

    pub const fn operational_policy_digest(&self) -> HumanoidEvidenceDigest {
        self.operational_policy_digest
    }

    pub fn operational_scope_id(&self) -> &str {
        &self.operational_scope_id
    }

    pub fn validate_at(
        &self,
        subject: &HumanoidQualificationSubject,
        live_policy: &HumanoidLiveAuthorityPolicy,
        operational_policy_digest: HumanoidEvidenceDigest,
        now_s: f64,
    ) -> bool {
        self.schema_version == HUMANOID_LIVE_AUTHORITY_APPROVAL_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && live_policy.validate_for(subject)
            && self.live_policy_digest == live_policy.policy_digest()
            && !operational_policy_digest.is_zero()
            && self.operational_policy_digest == operational_policy_digest
            && valid_id(&self.approval_id)
            && valid_id(&self.operational_scope_id)
            && now_s.is_finite()
            && now_s >= self.approved_at_s
            && now_s <= self.valid_until_s
            && !self.operator_verification_digest.is_zero()
            && self.operator_source.validate_at(now_s)
            && self.valid_until_s <= self.operator_source.valid_until_s
            && !self.approval_digest.is_zero()
            && self.approval_digest == digest_live_approval(self)
    }

    pub(crate) fn derived_operator_source(
        &self,
        subject: &HumanoidQualificationSubject,
        live_policy: &HumanoidLiveAuthorityPolicy,
        operational_policy_digest: HumanoidEvidenceDigest,
        now_s: f64,
    ) -> Option<HumanoidAuthoritySourceSnapshot> {
        if !self.validate_at(subject, live_policy, operational_policy_digest, now_s) {
            return None;
        }
        let source = HumanoidAuthoritySourceSnapshot {
            evidence_id: format!(
                "humanoid-live-operator-approval-sha256:{}",
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
pub enum HumanoidLiveAuthorityBundleFailure {
    InvalidPolicy,
    InvalidPhysicalEvidence,
    InvalidEpistemicEvidence,
    InvalidCognitiveEvidence,
    InvalidPhysicalVerifier,
    InvalidEpistemicVerifier,
    InvalidCognitiveVerifier,
    EvidenceExpired,
    InvalidBundleDigest,
}

/// Opaque, task-generic bundle of the three canonical machine-derived live sources.
pub struct HumanoidVerifiedLiveAuthorityEvidence {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    live_policy_digest: HumanoidEvidenceDigest,
    physical_verification_digest: HumanoidEvidenceDigest,
    epistemic_verification_digest: HumanoidEvidenceDigest,
    cognitive_verification_digest: HumanoidEvidenceDigest,
    physical: HumanoidAuthoritySourceSnapshot,
    epistemic: HumanoidAuthoritySourceSnapshot,
    cognitive: HumanoidAuthoritySourceSnapshot,
    assembled_at_s: f64,
    valid_until_s: f64,
    bundle_digest: HumanoidEvidenceDigest,
}

impl std::fmt::Debug for HumanoidVerifiedLiveAuthorityEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidVerifiedLiveAuthorityEvidence")
            .field("live_policy_digest", &self.live_policy_digest)
            .field("physical_verification_digest", &self.physical_verification_digest)
            .field("epistemic_verification_digest", &self.epistemic_verification_digest)
            .field("cognitive_verification_digest", &self.cognitive_verification_digest)
            .field("valid_until_s", &self.valid_until_s)
            .field("bundle_digest", &self.bundle_digest)
            .finish()
    }
}

impl HumanoidVerifiedLiveAuthorityEvidence {
    pub fn assemble(
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidLiveAuthorityPolicy,
        physical: &HumanoidPolicyVerifiedPhysicalAuthorityEvidence,
        epistemic: &HumanoidPolicyVerifiedEpistemicAuthorityEvidence,
        cognitive: &HumanoidPolicyVerifiedCognitiveAuthorityEvidence,
        now_s: f64,
    ) -> Result<Self, HumanoidLiveAuthorityBundleFailure> {
        if !policy.validate_for(subject) {
            return Err(HumanoidLiveAuthorityBundleFailure::InvalidPolicy);
        }
        if !physical.validate_for(subject, now_s)
            || physical.evidence_policy_digest() != policy.physical_evidence_policy_digest()
        {
            return Err(HumanoidLiveAuthorityBundleFailure::InvalidPhysicalEvidence);
        }
        if !epistemic.validate_for(subject, now_s)
            || epistemic.evidence_policy_digest() != policy.epistemic_evidence_policy_digest()
        {
            return Err(HumanoidLiveAuthorityBundleFailure::InvalidEpistemicEvidence);
        }
        if !cognitive.validate_for(subject, now_s)
            || cognitive.evidence_policy_digest() != policy.cognitive_evidence_policy_digest()
        {
            return Err(HumanoidLiveAuthorityBundleFailure::InvalidCognitiveEvidence);
        }
        if physical.verifier_digest() != policy.physical_verifier_digest() {
            return Err(HumanoidLiveAuthorityBundleFailure::InvalidPhysicalVerifier);
        }
        if epistemic.verifier_digest() != policy.epistemic_verifier_digest() {
            return Err(HumanoidLiveAuthorityBundleFailure::InvalidEpistemicVerifier);
        }
        if cognitive.verifier_digest() != policy.cognitive_verifier_digest() {
            return Err(HumanoidLiveAuthorityBundleFailure::InvalidCognitiveVerifier);
        }
        if !now_s.is_finite() || now_s < 0.0 {
            return Err(HumanoidLiveAuthorityBundleFailure::EvidenceExpired);
        }

        let valid_until_s = physical
            .valid_until_s()
            .min(epistemic.valid_until_s())
            .min(cognitive.valid_until_s());
        if !valid_until_s.is_finite() || valid_until_s <= now_s {
            return Err(HumanoidLiveAuthorityBundleFailure::EvidenceExpired);
        }

        let mut value = Self {
            schema_version: HUMANOID_LIVE_AUTHORITY_BUNDLE_SCHEMA_VERSION,
            subject_digest: digest_subject(subject)
                .ok_or(HumanoidLiveAuthorityBundleFailure::InvalidPolicy)?,
            live_policy_digest: policy.policy_digest(),
            physical_verification_digest: physical.inner().verification_digest(),
            epistemic_verification_digest: epistemic.inner().verification_digest(),
            cognitive_verification_digest: cognitive.verification_digest(),
            physical: physical.inner().source_snapshot(),
            epistemic: epistemic.inner().source_snapshot(),
            cognitive: cognitive.source_snapshot(),
            assembled_at_s: now_s,
            valid_until_s,
            bundle_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.bundle_digest = digest_live_bundle(&value);
        if value.bundle_digest.is_zero()
            || !value.validate_at(subject, policy, now_s)
        {
            return Err(HumanoidLiveAuthorityBundleFailure::InvalidBundleDigest);
        }
        Ok(value)
    }

    pub const fn bundle_digest(&self) -> HumanoidEvidenceDigest {
        self.bundle_digest
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.valid_until_s
    }

    pub fn validate_at(
        &self,
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidLiveAuthorityPolicy,
        now_s: f64,
    ) -> bool {
        self.schema_version == HUMANOID_LIVE_AUTHORITY_BUNDLE_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && policy.validate_for(subject)
            && self.live_policy_digest == policy.policy_digest()
            && now_s.is_finite()
            && now_s >= self.assembled_at_s
            && now_s <= self.valid_until_s
            && self.physical.validate_at(now_s)
            && self.epistemic.validate_at(now_s)
            && self.cognitive.validate_at(now_s)
            && !self.physical_verification_digest.is_zero()
            && !self.epistemic_verification_digest.is_zero()
            && !self.cognitive_verification_digest.is_zero()
            && !self.bundle_digest.is_zero()
            && self.bundle_digest == digest_live_bundle(self)
    }

    pub(crate) fn source_snapshots(
        &self,
    ) -> (
        HumanoidAuthoritySourceSnapshot,
        HumanoidAuthoritySourceSnapshot,
        HumanoidAuthoritySourceSnapshot,
    ) {
        (
            self.physical.clone(),
            self.epistemic.clone(),
            self.cognitive.clone(),
        )
    }
}

fn digest_live_policy(policy: &HumanoidLiveAuthorityPolicy) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.live-authority-policy.v1");
    h.u32(policy.schema_version)
        .string(&policy.policy_id)
        .digest(policy.subject_digest)
        .digest(policy.physical_evidence_policy_digest)
        .digest(policy.epistemic_evidence_policy_digest)
        .digest(policy.cognitive_evidence_policy_digest)
        .digest(policy.operator_verifier_digest)
        .digest(policy.physical_verifier_digest)
        .digest(policy.epistemic_verifier_digest)
        .digest(policy.cognitive_verifier_digest);
    h.finish()
}

fn digest_live_approval(approval: &HumanoidLiveAuthorityApproval) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.live-authority-approval.v1");
    h.u32(approval.schema_version)
        .string(&approval.approval_id)
        .digest(approval.subject_digest)
        .digest(approval.live_policy_digest)
        .digest(approval.operational_policy_digest)
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

fn digest_live_bundle(bundle: &HumanoidVerifiedLiveAuthorityEvidence) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.live-authority-bundle.v1");
    h.u32(bundle.schema_version)
        .digest(bundle.subject_digest)
        .digest(bundle.live_policy_digest)
        .digest(bundle.physical_verification_digest)
        .digest(bundle.epistemic_verification_digest)
        .digest(bundle.cognitive_verification_digest)
        .string(&bundle.physical.evidence_id)
        .f32(bundle.physical.scale)
        .f64(bundle.physical.evaluated_at_s)
        .f64(bundle.physical.valid_until_s)
        .string(&bundle.epistemic.evidence_id)
        .f32(bundle.epistemic.scale)
        .f64(bundle.epistemic.evaluated_at_s)
        .f64(bundle.epistemic.valid_until_s)
        .string(&bundle.cognitive.evidence_id)
        .f32(bundle.cognitive.scale)
        .f64(bundle.cognitive.evaluated_at_s)
        .f64(bundle.cognitive.valid_until_s)
        .f64(bundle.assembled_at_s)
        .f64(bundle.valid_until_s);
    h.finish()
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("humanoid.live-authority-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
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
