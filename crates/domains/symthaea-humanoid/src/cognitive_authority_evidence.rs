// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical cognitive motor-restriction evidence for operational humanoid authority.
//!
//! This module deliberately does **not** treat Phi or any consciousness metric as
//! evidence of motor competence. It records the existing most-restrictive-wins
//! safety decision used by the humanoid embodiment:
//!
//! `max(phi-derived level, explicit safety override, moral/consent restriction)`
//!
//! The resulting `MotorSafetyLevel::motor_gain()` may only restrict goal-directed
//! authority. Protective fallback remains a separate deterministic path.

use symthaea_core::embodiment::MotorSafetyLevel;

use crate::authority_source_signing::HumanoidAuthorityUnsignedClaim;
use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::qualification::HumanoidQualificationSubject;
use crate::skill_authority_receipt::HumanoidAuthoritySourceSnapshot;
use crate::types::{ActuationMode, HumanoidTask};
use crate::verified_authority_source::{
    HumanoidAuthoritySourceClaim, HumanoidAuthoritySourceVerificationFailure,
    HumanoidAuthoritySourceVerifier, HumanoidVerifiedAuthorityKind,
    HumanoidVerifiedAuthoritySource, verify_humanoid_authority_source,
};

pub const HUMANOID_COGNITIVE_AUTHORITY_POLICY_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_COGNITIVE_AUTHORITY_EVIDENCE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HumanoidCognitiveRestrictionFrame {
    /// Monotonic sequence owned by the motor-safety supervisor.
    pub sequence: u64,
    /// Time at which the supervisor sampled all restriction inputs.
    pub sampled_at_s: f64,
    /// Time at which the composed frame became available to the authority path.
    pub received_at_s: f64,
    /// Current Phi input. Non-finite values are permitted because the canonical
    /// core rule maps them to `MotorSafetyLevel::Red`.
    pub phi: f64,
    /// Explicit operator/system safety override. Absence means no additional
    /// restriction and is represented canonically as Green during composition.
    pub safety_override: Option<MotorSafetyLevel>,
    /// Moral-gate verdict: 0=Safe, 1=Caution, 2=Blocked.
    /// Any other value is rejected rather than interpreted as Safe.
    pub moral_verdict: u8,
    /// Consent violation forces Orange or stricter.
    pub consent_violation: bool,
    /// Ahimsa violation forces Red.
    pub ahimsa_violated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidCognitiveAuthorityEvidenceFailure {
    InvalidSubject,
    InvalidProducerIdentity,
    InvalidMaximumAge,
    InvalidTimestamp,
    InvalidMoralVerdict,
    StaleFrame,
    InvalidDigest,
}

#[derive(Debug, Clone)]
pub struct HumanoidCognitiveAuthorityEvidencePolicy {
    subject: HumanoidQualificationSubject,
    safety_supervisor_artifact_digest: HumanoidEvidenceDigest,
    phi_producer_artifact_digest: HumanoidEvidenceDigest,
    moral_gate_producer_artifact_digest: HumanoidEvidenceDigest,
    override_producer_artifact_digest: HumanoidEvidenceDigest,
    maximum_frame_age_s: f64,
    policy_digest: HumanoidEvidenceDigest,
}

impl HumanoidCognitiveAuthorityEvidencePolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: HumanoidQualificationSubject,
        safety_supervisor_artifact_digest: HumanoidEvidenceDigest,
        phi_producer_artifact_digest: HumanoidEvidenceDigest,
        moral_gate_producer_artifact_digest: HumanoidEvidenceDigest,
        override_producer_artifact_digest: HumanoidEvidenceDigest,
        maximum_frame_age_s: f64,
    ) -> Result<Self, HumanoidCognitiveAuthorityEvidenceFailure> {
        if !subject.validate() {
            return Err(HumanoidCognitiveAuthorityEvidenceFailure::InvalidSubject);
        }
        if [
            safety_supervisor_artifact_digest,
            phi_producer_artifact_digest,
            moral_gate_producer_artifact_digest,
            override_producer_artifact_digest,
        ]
        .into_iter()
        .any(HumanoidEvidenceDigest::is_zero)
        {
            return Err(HumanoidCognitiveAuthorityEvidenceFailure::InvalidProducerIdentity);
        }
        if !maximum_frame_age_s.is_finite() || maximum_frame_age_s <= 0.0 {
            return Err(HumanoidCognitiveAuthorityEvidenceFailure::InvalidMaximumAge);
        }

        let mut value = Self {
            subject,
            safety_supervisor_artifact_digest,
            phi_producer_artifact_digest,
            moral_gate_producer_artifact_digest,
            override_producer_artifact_digest,
            maximum_frame_age_s,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.policy_digest = digest_policy(&value);
        if value.policy_digest.is_zero() {
            return Err(HumanoidCognitiveAuthorityEvidenceFailure::InvalidDigest);
        }
        Ok(value)
    }

    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }

    pub fn subject(&self) -> &HumanoidQualificationSubject {
        &self.subject
    }

    pub const fn maximum_frame_age_s(&self) -> f64 {
        self.maximum_frame_age_s
    }

    pub fn evaluate(
        &self,
        frame: HumanoidCognitiveRestrictionFrame,
        now_s: f64,
    ) -> Result<HumanoidPolicyBoundCognitiveAuthorityEvidence, HumanoidCognitiveAuthorityEvidenceFailure>
    {
        if !frame.sampled_at_s.is_finite()
            || !frame.received_at_s.is_finite()
            || !now_s.is_finite()
            || frame.sampled_at_s < 0.0
            || frame.received_at_s < frame.sampled_at_s
            || now_s < frame.received_at_s
        {
            return Err(HumanoidCognitiveAuthorityEvidenceFailure::InvalidTimestamp);
        }
        if frame.moral_verdict > 2 {
            return Err(HumanoidCognitiveAuthorityEvidenceFailure::InvalidMoralVerdict);
        }

        let valid_until_s = frame.sampled_at_s + self.maximum_frame_age_s;
        if !valid_until_s.is_finite() || valid_until_s <= now_s {
            return Err(HumanoidCognitiveAuthorityEvidenceFailure::StaleFrame);
        }

        let phi_level = MotorSafetyLevel::from_phi(frame.phi);
        let override_level = frame.safety_override.unwrap_or(MotorSafetyLevel::Green);
        let moral_level = moral_restriction_level(frame);
        let effective_level = phi_level.max(override_level).max(moral_level);
        let authority_scale = effective_level.motor_gain();

        let mut artifact = HumanoidPolicyBoundCognitiveAuthorityEvidence {
            subject: self.subject.clone(),
            evidence_policy_digest: self.policy_digest,
            frame,
            phi_level,
            moral_level,
            effective_level,
            authority_scale,
            evaluated_at_s: now_s,
            valid_until_s,
            artifact_digest: HumanoidEvidenceDigest::ZERO,
        };
        artifact.artifact_digest = digest_artifact(&artifact);
        if artifact.artifact_digest.is_zero() {
            return Err(HumanoidCognitiveAuthorityEvidenceFailure::InvalidDigest);
        }
        Ok(artifact)
    }
}

#[derive(Debug, Clone)]
pub struct HumanoidPolicyBoundCognitiveAuthorityEvidence {
    subject: HumanoidQualificationSubject,
    evidence_policy_digest: HumanoidEvidenceDigest,
    frame: HumanoidCognitiveRestrictionFrame,
    phi_level: MotorSafetyLevel,
    moral_level: MotorSafetyLevel,
    effective_level: MotorSafetyLevel,
    authority_scale: f32,
    evaluated_at_s: f64,
    valid_until_s: f64,
    artifact_digest: HumanoidEvidenceDigest,
}

impl HumanoidPolicyBoundCognitiveAuthorityEvidence {
    pub const fn evidence_policy_digest(&self) -> HumanoidEvidenceDigest {
        self.evidence_policy_digest
    }

    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.artifact_digest
    }

    pub const fn effective_level(&self) -> MotorSafetyLevel {
        self.effective_level
    }

    pub const fn authority_scale(&self) -> f32 {
        self.authority_scale
    }

    pub const fn evaluated_at_s(&self) -> f64 {
        self.evaluated_at_s
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.valid_until_s
    }

    pub fn unsigned_claim(
        &self,
        issuer_id: impl Into<String>,
        key_id: impl Into<String>,
        revocation_epoch: u64,
        scheme_id: impl Into<String>,
    ) -> Option<HumanoidAuthorityUnsignedClaim> {
        HumanoidAuthorityUnsignedClaim::new(
            self.subject.clone(),
            HumanoidVerifiedAuthorityKind::Cognitive,
            self.artifact_digest,
            self.authority_scale,
            self.evaluated_at_s,
            self.valid_until_s,
            issuer_id,
            key_id,
            revocation_epoch,
            scheme_id,
        )
    }

    pub fn verify(
        &self,
        claim: &HumanoidAuthoritySourceClaim,
        verifier: &dyn HumanoidAuthoritySourceVerifier,
        now_s: f64,
    ) -> Result<HumanoidPolicyVerifiedCognitiveAuthorityEvidence, HumanoidCognitiveAuthorityVerificationFailure>
    {
        if claim.source_kind() != HumanoidVerifiedAuthorityKind::Cognitive {
            return Err(HumanoidCognitiveAuthorityVerificationFailure::ClaimKindMismatch);
        }
        if claim.evidence_digest() != self.artifact_digest {
            return Err(HumanoidCognitiveAuthorityVerificationFailure::EvidenceDigestMismatch);
        }
        if claim.scale().to_bits() != self.authority_scale.to_bits() {
            return Err(HumanoidCognitiveAuthorityVerificationFailure::AuthorityScaleMismatch);
        }
        if claim.evaluated_at_s().to_bits() != self.evaluated_at_s.to_bits() {
            return Err(HumanoidCognitiveAuthorityVerificationFailure::EvaluationTimeMismatch);
        }
        if claim.valid_until_s().to_bits() != self.valid_until_s.to_bits() {
            return Err(HumanoidCognitiveAuthorityVerificationFailure::ExpiryMismatch);
        }

        let inner = verify_humanoid_authority_source(
            &self.subject,
            HumanoidVerifiedAuthorityKind::Cognitive,
            claim,
            verifier,
            now_s,
        )
        .map_err(HumanoidCognitiveAuthorityVerificationFailure::Verification)?;
        let verifier_digest = inner.verifier_digest();

        Ok(HumanoidPolicyVerifiedCognitiveAuthorityEvidence {
            subject: self.subject.clone(),
            evidence_policy_digest: self.evidence_policy_digest,
            verifier_digest,
            inner,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidCognitiveAuthorityVerificationFailure {
    ClaimKindMismatch,
    EvidenceDigestMismatch,
    AuthorityScaleMismatch,
    EvaluationTimeMismatch,
    ExpiryMismatch,
    Verification(HumanoidAuthoritySourceVerificationFailure),
}

pub struct HumanoidPolicyVerifiedCognitiveAuthorityEvidence {
    subject: HumanoidQualificationSubject,
    evidence_policy_digest: HumanoidEvidenceDigest,
    verifier_digest: HumanoidEvidenceDigest,
    inner: HumanoidVerifiedAuthoritySource,
}

impl std::fmt::Debug for HumanoidPolicyVerifiedCognitiveAuthorityEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidPolicyVerifiedCognitiveAuthorityEvidence")
            .field("evidence_policy_digest", &self.evidence_policy_digest)
            .field("verifier_digest", &self.verifier_digest)
            .field("verification_digest", &self.inner.verification_digest())
            .field("scale", &self.inner.scale())
            .field("valid_until_s", &self.inner.valid_until_s())
            .finish()
    }
}

impl HumanoidPolicyVerifiedCognitiveAuthorityEvidence {
    pub const fn evidence_policy_digest(&self) -> HumanoidEvidenceDigest {
        self.evidence_policy_digest
    }

    pub const fn verifier_digest(&self) -> HumanoidEvidenceDigest {
        self.verifier_digest
    }

    pub const fn verification_digest(&self) -> HumanoidEvidenceDigest {
        self.inner.verification_digest()
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.inner.valid_until_s()
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject, now_s: f64) -> bool {
        &self.subject == subject
            && !self.evidence_policy_digest.is_zero()
            && !self.verifier_digest.is_zero()
            && self.verifier_digest == self.inner.verifier_digest()
            && self
                .inner
                .validate_for(subject, HumanoidVerifiedAuthorityKind::Cognitive, now_s)
    }

    pub(crate) fn source_snapshot(&self) -> HumanoidAuthoritySourceSnapshot {
        self.inner.source_snapshot()
    }
}

fn moral_restriction_level(frame: HumanoidCognitiveRestrictionFrame) -> MotorSafetyLevel {
    if frame.ahimsa_violated || frame.moral_verdict == 2 {
        MotorSafetyLevel::Red
    } else if frame.consent_violation {
        MotorSafetyLevel::Orange
    } else if frame.moral_verdict == 1 {
        MotorSafetyLevel::Yellow
    } else {
        MotorSafetyLevel::Green
    }
}

fn digest_policy(policy: &HumanoidCognitiveAuthorityEvidencePolicy) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.cognitive-authority-policy.v1");
    h.u32(HUMANOID_COGNITIVE_AUTHORITY_POLICY_SCHEMA_VERSION)
        .digest(digest_subject(&policy.subject))
        .digest(policy.safety_supervisor_artifact_digest)
        .digest(policy.phi_producer_artifact_digest)
        .digest(policy.moral_gate_producer_artifact_digest)
        .digest(policy.override_producer_artifact_digest)
        .f64(policy.maximum_frame_age_s);
    h.finish()
}

fn digest_artifact(artifact: &HumanoidPolicyBoundCognitiveAuthorityEvidence) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.cognitive-authority-evidence.v1");
    h.u32(HUMANOID_COGNITIVE_AUTHORITY_EVIDENCE_SCHEMA_VERSION)
        .digest(digest_subject(&artifact.subject))
        .digest(artifact.evidence_policy_digest)
        .u64(artifact.frame.sequence)
        .f64(artifact.frame.sampled_at_s)
        .f64(artifact.frame.received_at_s)
        .bool(artifact.frame.phi.is_finite());
    if artifact.frame.phi.is_finite() {
        h.f64(artifact.frame.phi);
    }
    h.bool(artifact.frame.safety_override.is_some());
    if let Some(level) = artifact.frame.safety_override {
        h.u64(safety_level_id(level));
    }
    h.u64(artifact.frame.moral_verdict as u64)
        .bool(artifact.frame.consent_violation)
        .bool(artifact.frame.ahimsa_violated)
        .u64(safety_level_id(artifact.phi_level))
        .u64(safety_level_id(artifact.moral_level))
        .u64(safety_level_id(artifact.effective_level))
        .f32(artifact.authority_scale)
        .f64(artifact.evaluated_at_s)
        .f64(artifact.valid_until_s);
    h.finish()
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.cognitive-authority-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    h.finish()
}

fn safety_level_id(level: MotorSafetyLevel) -> u64 {
    match level {
        MotorSafetyLevel::Green => 1,
        MotorSafetyLevel::Yellow => 2,
        MotorSafetyLevel::Orange => 3,
        MotorSafetyLevel::Red => 4,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "cognitive-authority-test-backend",
        )
    }

    fn policy() -> HumanoidCognitiveAuthorityEvidencePolicy {
        HumanoidCognitiveAuthorityEvidencePolicy::new(
            subject(),
            HumanoidEvidenceDigest::from_bytes([1; 32]),
            HumanoidEvidenceDigest::from_bytes([2; 32]),
            HumanoidEvidenceDigest::from_bytes([3; 32]),
            HumanoidEvidenceDigest::from_bytes([4; 32]),
            0.05,
        )
        .unwrap()
    }

    fn safe_frame() -> HumanoidCognitiveRestrictionFrame {
        HumanoidCognitiveRestrictionFrame {
            sequence: 7,
            sampled_at_s: 1.0,
            received_at_s: 1.0,
            phi: 0.8,
            safety_override: None,
            moral_verdict: 0,
            consent_violation: false,
            ahimsa_violated: false,
        }
    }

    #[test]
    fn safe_fresh_frame_preserves_full_goal_authority() {
        let evidence = policy().evaluate(safe_frame(), 1.0).unwrap();
        assert_eq!(evidence.effective_level(), MotorSafetyLevel::Green);
        assert_eq!(evidence.authority_scale(), 1.0);
    }

    #[test]
    fn consent_restricts_good_phi_to_orange() {
        let mut frame = safe_frame();
        frame.consent_violation = true;
        let evidence = policy().evaluate(frame, 1.0).unwrap();
        assert_eq!(evidence.effective_level(), MotorSafetyLevel::Orange);
        assert_eq!(evidence.authority_scale(), 0.3);
    }

    #[test]
    fn explicit_override_is_most_restrictive_wins() {
        let mut frame = safe_frame();
        frame.safety_override = Some(MotorSafetyLevel::Yellow);
        let evidence = policy().evaluate(frame, 1.0).unwrap();
        assert_eq!(evidence.effective_level(), MotorSafetyLevel::Yellow);
        assert_eq!(evidence.authority_scale(), 0.6);
    }

    #[test]
    fn non_finite_phi_fails_closed_to_red() {
        let mut frame = safe_frame();
        frame.phi = f64::NAN;
        let evidence = policy().evaluate(frame, 1.0).unwrap();
        assert_eq!(evidence.effective_level(), MotorSafetyLevel::Red);
        assert_eq!(evidence.authority_scale(), 0.0);
    }

    #[test]
    fn unknown_moral_verdict_is_rejected_not_treated_as_safe() {
        let mut frame = safe_frame();
        frame.moral_verdict = 99;
        assert_eq!(
            policy().evaluate(frame, 1.0).unwrap_err(),
            HumanoidCognitiveAuthorityEvidenceFailure::InvalidMoralVerdict
        );
    }

    #[test]
    fn stale_frame_cannot_produce_operational_evidence() {
        assert_eq!(
            policy().evaluate(safe_frame(), 1.05).unwrap_err(),
            HumanoidCognitiveAuthorityEvidenceFailure::StaleFrame
        );
    }
}
