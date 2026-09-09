// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Precommitted producer policies for live humanoid authority evidence.
//!
//! A valid signature over an evidence artifact is not sufficient if an operational
//! caller can silently choose a different monitor build, calibration, health
//! derating config, estimator build, or estimator config. These types freeze those
//! identities before dynamic evidence is observed and carry the policy digest
//! through verification.

use crate::authority_evidence_artifacts::{
    HumanoidAuthorityEvidenceArtifactFailure, HumanoidEpistemicAuthorityEvidenceArtifact,
    HumanoidPhysicalAuthorityEvidenceArtifact,
};
use crate::authority_source_signing::HumanoidAuthorityUnsignedClaim;
use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::physical_health::{HumanoidPhysicalHealthFrame, PhysicalHealthAuthorityConfig};
use crate::qualification::HumanoidQualificationSubject;
use crate::state_estimation::StateEstimatorConfig;
use crate::state_uncertainty::HumanoidStateUncertaintyEnvelope;
use crate::types::{ActuationMode, HumanoidState, HumanoidTask};
use crate::verified_authority_artifacts::{
    HumanoidCanonicalAuthorityVerificationFailure, HumanoidVerifiedEpistemicAuthorityEvidence,
    HumanoidVerifiedPhysicalAuthorityEvidence,
};
use crate::verified_authority_source::{
    HumanoidAuthoritySourceClaim, HumanoidAuthoritySourceVerifier,
};

pub const HUMANOID_AUTHORITY_EVIDENCE_POLICY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidAuthorityEvidencePolicyFailure {
    InvalidSubject,
    InvalidProducerIdentity,
    InvalidCalibrationIdentity,
    InvalidConfig,
    Evidence(HumanoidAuthorityEvidenceArtifactFailure),
}

#[derive(Debug, Clone)]
pub struct HumanoidPhysicalAuthorityEvidencePolicy {
    subject: HumanoidQualificationSubject,
    monitor_artifact_digest: HumanoidEvidenceDigest,
    calibration_artifact_digest: HumanoidEvidenceDigest,
    config: PhysicalHealthAuthorityConfig,
    policy_digest: HumanoidEvidenceDigest,
}

impl HumanoidPhysicalAuthorityEvidencePolicy {
    pub fn new(
        subject: HumanoidQualificationSubject,
        monitor_artifact_digest: HumanoidEvidenceDigest,
        calibration_artifact_digest: HumanoidEvidenceDigest,
        config: PhysicalHealthAuthorityConfig,
    ) -> Result<Self, HumanoidAuthorityEvidencePolicyFailure> {
        if !subject.validate() {
            return Err(HumanoidAuthorityEvidencePolicyFailure::InvalidSubject);
        }
        if monitor_artifact_digest.is_zero() {
            return Err(HumanoidAuthorityEvidencePolicyFailure::InvalidProducerIdentity);
        }
        if calibration_artifact_digest.is_zero() {
            return Err(HumanoidAuthorityEvidencePolicyFailure::InvalidCalibrationIdentity);
        }
        if !valid_physical_config(config) {
            return Err(HumanoidAuthorityEvidencePolicyFailure::InvalidConfig);
        }
        let mut value = Self {
            subject,
            monitor_artifact_digest,
            calibration_artifact_digest,
            config,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.policy_digest = digest_physical_policy(&value);
        if value.policy_digest.is_zero() {
            return Err(HumanoidAuthorityEvidencePolicyFailure::InvalidConfig);
        }
        Ok(value)
    }

    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }

    pub fn subject(&self) -> &HumanoidQualificationSubject {
        &self.subject
    }

    pub fn evaluate(
        &self,
        frame: &HumanoidPhysicalHealthFrame,
        now_s: f64,
    ) -> Result<HumanoidPolicyBoundPhysicalAuthorityEvidence, HumanoidAuthorityEvidencePolicyFailure> {
        let inner = HumanoidPhysicalAuthorityEvidenceArtifact::from_health_frame(
            &self.subject,
            frame,
            self.config,
            self.monitor_artifact_digest,
            self.calibration_artifact_digest,
            now_s,
        )
        .map_err(HumanoidAuthorityEvidencePolicyFailure::Evidence)?;
        Ok(HumanoidPolicyBoundPhysicalAuthorityEvidence {
            subject: self.subject.clone(),
            policy_digest: self.policy_digest,
            inner,
        })
    }
}

#[derive(Debug, Clone)]
pub struct HumanoidEpistemicAuthorityEvidencePolicy {
    subject: HumanoidQualificationSubject,
    estimator_artifact_digest: HumanoidEvidenceDigest,
    config: StateEstimatorConfig,
    policy_digest: HumanoidEvidenceDigest,
}

impl HumanoidEpistemicAuthorityEvidencePolicy {
    pub fn new(
        subject: HumanoidQualificationSubject,
        estimator_artifact_digest: HumanoidEvidenceDigest,
        config: StateEstimatorConfig,
    ) -> Result<Self, HumanoidAuthorityEvidencePolicyFailure> {
        if !subject.validate() {
            return Err(HumanoidAuthorityEvidencePolicyFailure::InvalidSubject);
        }
        if estimator_artifact_digest.is_zero() {
            return Err(HumanoidAuthorityEvidencePolicyFailure::InvalidProducerIdentity);
        }
        if !valid_estimator_config(&config) {
            return Err(HumanoidAuthorityEvidencePolicyFailure::InvalidConfig);
        }
        let mut value = Self {
            subject,
            estimator_artifact_digest,
            config,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.policy_digest = digest_epistemic_policy(&value);
        if value.policy_digest.is_zero() {
            return Err(HumanoidAuthorityEvidencePolicyFailure::InvalidConfig);
        }
        Ok(value)
    }

    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }

    pub fn subject(&self) -> &HumanoidQualificationSubject {
        &self.subject
    }

    pub fn evaluate(
        &self,
        estimated_state: &HumanoidState,
        uncertainty: HumanoidStateUncertaintyEnvelope,
        evaluated_at_s: f64,
    ) -> Result<HumanoidPolicyBoundEpistemicAuthorityEvidence, HumanoidAuthorityEvidencePolicyFailure> {
        let inner = HumanoidEpistemicAuthorityEvidenceArtifact::from_state_estimate(
            &self.subject,
            estimated_state,
            uncertainty,
            &self.config,
            self.estimator_artifact_digest,
            evaluated_at_s,
        )
        .map_err(HumanoidAuthorityEvidencePolicyFailure::Evidence)?;
        Ok(HumanoidPolicyBoundEpistemicAuthorityEvidence {
            subject: self.subject.clone(),
            policy_digest: self.policy_digest,
            inner,
        })
    }
}

#[derive(Debug, Clone)]
pub struct HumanoidPolicyBoundPhysicalAuthorityEvidence {
    subject: HumanoidQualificationSubject,
    policy_digest: HumanoidEvidenceDigest,
    inner: HumanoidPhysicalAuthorityEvidenceArtifact,
}

impl HumanoidPolicyBoundPhysicalAuthorityEvidence {
    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }

    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.inner.artifact_digest()
    }

    pub fn unsigned_claim(
        &self,
        issuer_id: impl Into<String>,
        key_id: impl Into<String>,
        revocation_epoch: u64,
        scheme_id: impl Into<String>,
    ) -> Option<HumanoidAuthorityUnsignedClaim> {
        self.inner.unsigned_claim(
            self.subject.clone(),
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
    ) -> Result<HumanoidPolicyVerifiedPhysicalAuthorityEvidence, HumanoidCanonicalAuthorityVerificationFailure> {
        let inner = HumanoidVerifiedPhysicalAuthorityEvidence::verify(
            &self.subject,
            &self.inner,
            claim,
            verifier,
            now_s,
        )?;
        let verifier_digest = inner.verifier_digest();
        Ok(HumanoidPolicyVerifiedPhysicalAuthorityEvidence {
            subject: self.subject.clone(),
            evidence_policy_digest: self.policy_digest,
            verifier_digest,
            inner,
        })
    }
}

#[derive(Debug, Clone)]
pub struct HumanoidPolicyBoundEpistemicAuthorityEvidence {
    subject: HumanoidQualificationSubject,
    policy_digest: HumanoidEvidenceDigest,
    inner: HumanoidEpistemicAuthorityEvidenceArtifact,
}

impl HumanoidPolicyBoundEpistemicAuthorityEvidence {
    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }

    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.inner.artifact_digest()
    }

    pub fn unsigned_claim(
        &self,
        issuer_id: impl Into<String>,
        key_id: impl Into<String>,
        revocation_epoch: u64,
        scheme_id: impl Into<String>,
    ) -> Option<HumanoidAuthorityUnsignedClaim> {
        self.inner.unsigned_claim(
            self.subject.clone(),
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
    ) -> Result<HumanoidPolicyVerifiedEpistemicAuthorityEvidence, HumanoidCanonicalAuthorityVerificationFailure> {
        let inner = HumanoidVerifiedEpistemicAuthorityEvidence::verify(
            &self.subject,
            &self.inner,
            claim,
            verifier,
            now_s,
        )?;
        let verifier_digest = inner.verifier_digest();
        Ok(HumanoidPolicyVerifiedEpistemicAuthorityEvidence {
            subject: self.subject.clone(),
            evidence_policy_digest: self.policy_digest,
            verifier_digest,
            inner,
        })
    }
}

pub struct HumanoidPolicyVerifiedPhysicalAuthorityEvidence {
    subject: HumanoidQualificationSubject,
    evidence_policy_digest: HumanoidEvidenceDigest,
    verifier_digest: HumanoidEvidenceDigest,
    inner: HumanoidVerifiedPhysicalAuthorityEvidence,
}

impl std::fmt::Debug for HumanoidPolicyVerifiedPhysicalAuthorityEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidPolicyVerifiedPhysicalAuthorityEvidence")
            .field("evidence_policy_digest", &self.evidence_policy_digest)
            .field("verifier_digest", &self.verifier_digest)
            .field("verification_digest", &self.inner.verification_digest())
            .finish()
    }
}

impl HumanoidPolicyVerifiedPhysicalAuthorityEvidence {
    pub const fn evidence_policy_digest(&self) -> HumanoidEvidenceDigest {
        self.evidence_policy_digest
    }

    pub const fn verifier_digest(&self) -> HumanoidEvidenceDigest {
        self.verifier_digest
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.inner.valid_until_s()
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject, now_s: f64) -> bool {
        &self.subject == subject
            && !self.evidence_policy_digest.is_zero()
            && !self.verifier_digest.is_zero()
            && self.verifier_digest == self.inner.verifier_digest()
            && self.inner.validate_for(subject, now_s)
    }

    pub(crate) fn inner(&self) -> &HumanoidVerifiedPhysicalAuthorityEvidence {
        &self.inner
    }
}

pub struct HumanoidPolicyVerifiedEpistemicAuthorityEvidence {
    subject: HumanoidQualificationSubject,
    evidence_policy_digest: HumanoidEvidenceDigest,
    verifier_digest: HumanoidEvidenceDigest,
    inner: HumanoidVerifiedEpistemicAuthorityEvidence,
}

impl std::fmt::Debug for HumanoidPolicyVerifiedEpistemicAuthorityEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidPolicyVerifiedEpistemicAuthorityEvidence")
            .field("evidence_policy_digest", &self.evidence_policy_digest)
            .field("verifier_digest", &self.verifier_digest)
            .field("verification_digest", &self.inner.verification_digest())
            .finish()
    }
}

impl HumanoidPolicyVerifiedEpistemicAuthorityEvidence {
    pub const fn evidence_policy_digest(&self) -> HumanoidEvidenceDigest {
        self.evidence_policy_digest
    }

    pub const fn verifier_digest(&self) -> HumanoidEvidenceDigest {
        self.verifier_digest
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.inner.valid_until_s()
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject, now_s: f64) -> bool {
        &self.subject == subject
            && !self.evidence_policy_digest.is_zero()
            && !self.verifier_digest.is_zero()
            && self.verifier_digest == self.inner.verifier_digest()
            && self.inner.validate_for(subject, now_s)
    }

    pub(crate) fn inner(&self) -> &HumanoidVerifiedEpistemicAuthorityEvidence {
        &self.inner
    }
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.authority-evidence-policy-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    h.finish()
}

fn digest_physical_policy(policy: &HumanoidPhysicalAuthorityEvidencePolicy) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.physical-authority-evidence-policy.v1");
    h.u32(HUMANOID_AUTHORITY_EVIDENCE_POLICY_SCHEMA_VERSION)
        .digest(digest_subject(&policy.subject))
        .digest(policy.monitor_artifact_digest)
        .digest(policy.calibration_artifact_digest)
        .f64(policy.config.full_freshness_age_s)
        .f64(policy.config.maximum_frame_age_s)
        .f64(policy.config.current_derate_start_fraction)
        .f32(policy.config.missing_temperature_authority)
        .f64(policy.config.minimum_state_of_charge)
        .f64(policy.config.full_state_of_charge);
    h.finish()
}

fn digest_epistemic_policy(policy: &HumanoidEpistemicAuthorityEvidencePolicy) -> HumanoidEvidenceDigest {
    let c = &policy.config;
    let mut h = HumanoidEvidenceHasher::new("humanoid.epistemic-authority-evidence-policy.v1");
    h.u32(HUMANOID_AUTHORITY_EVIDENCE_POLICY_SCHEMA_VERSION)
        .digest(digest_subject(&policy.subject))
        .digest(policy.estimator_artifact_digest)
        .f64(c.orientation_correction)
        .f64(c.linear_velocity_correction)
        .f64(c.angular_velocity_correction)
        .f64(c.joint_position_correction)
        .f64(c.joint_velocity_correction)
        .f64(c.double_support_velocity_damping)
        .f64(c.maximum_measurement_age_s)
        .f64(c.maximum_dt_s)
        .f64(c.maximum_orientation_innovation_rad)
        .f64(c.maximum_linear_velocity_innovation_mps)
        .f64(c.maximum_joint_position_innovation_rad);
    h.finish()
}

fn valid_physical_config(config: PhysicalHealthAuthorityConfig) -> bool {
    config.full_freshness_age_s.is_finite()
        && config.maximum_frame_age_s.is_finite()
        && config.full_freshness_age_s >= 0.0
        && config.full_freshness_age_s < config.maximum_frame_age_s
        && config.current_derate_start_fraction.is_finite()
        && (0.0..1.0).contains(&config.current_derate_start_fraction)
        && config.missing_temperature_authority.is_finite()
        && (0.0..=1.0).contains(&config.missing_temperature_authority)
        && config.minimum_state_of_charge.is_finite()
        && config.full_state_of_charge.is_finite()
        && (0.0..1.0).contains(&config.minimum_state_of_charge)
        && (0.0..=1.0).contains(&config.full_state_of_charge)
        && config.minimum_state_of_charge < config.full_state_of_charge
}

fn valid_estimator_config(config: &StateEstimatorConfig) -> bool {
    [
        config.orientation_correction,
        config.linear_velocity_correction,
        config.angular_velocity_correction,
        config.joint_position_correction,
        config.joint_velocity_correction,
        config.double_support_velocity_damping,
        config.maximum_measurement_age_s,
        config.maximum_dt_s,
        config.maximum_orientation_innovation_rad,
        config.maximum_linear_velocity_innovation_mps,
        config.maximum_joint_position_innovation_rad,
    ]
    .into_iter()
    .all(f64::is_finite)
        && config.maximum_measurement_age_s > 0.0
        && config.maximum_dt_s > 0.0
        && config.maximum_orientation_innovation_rad > 0.0
        && config.maximum_linear_velocity_innovation_mps > 0.0
        && config.maximum_joint_position_innovation_rad > 0.0
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
