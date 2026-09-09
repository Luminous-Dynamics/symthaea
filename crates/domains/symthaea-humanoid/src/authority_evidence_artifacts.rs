// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical evidence artifacts for externally authenticated humanoid authority.
//!
//! These artifacts do not authenticate themselves. They canonicalize the exact
//! runtime evidence and restrictive authority scale that an external signer may
//! attest through `HumanoidAuthorityUnsignedClaim` / `HumanoidAuthoritySourceClaim`.

use crate::authority_source_signing::HumanoidAuthorityUnsignedClaim;
use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::physical_health::{
    HumanoidPhysicalHealthEnvelope, HumanoidPhysicalHealthFrame, PhysicalHealthAuthorityConfig,
};
use crate::qualification::HumanoidQualificationSubject;
use crate::state_estimation::StateEstimatorConfig;
use crate::state_uncertainty::{BoundedUncertainty, HumanoidStateUncertaintyEnvelope};
use crate::types::{ActuationMode, HumanoidState, HumanoidTask};
use crate::verified_authority_source::HumanoidVerifiedAuthorityKind;

pub const HUMANOID_PHYSICAL_AUTHORITY_EVIDENCE_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_EPISTEMIC_AUTHORITY_EVIDENCE_SCHEMA_VERSION: u32 = 1;
const TIME_MATCH_TOLERANCE_S: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidAuthorityEvidenceArtifactFailure {
    InvalidSubject,
    InvalidProducerIdentity,
    InvalidCalibrationIdentity,
    InvalidPhysicalEvidence,
    PhysicalEvidenceExpired,
    InvalidEstimatorEvidence,
    EstimatorStateMismatch,
    EpistemicEvidenceExpired,
    InvalidDigest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidPhysicalAuthorityEvidenceArtifact {
    subject_digest: HumanoidEvidenceDigest,
    monitor_artifact_digest: HumanoidEvidenceDigest,
    calibration_artifact_digest: HumanoidEvidenceDigest,
    frame_digest: HumanoidEvidenceDigest,
    config_digest: HumanoidEvidenceDigest,
    evaluated_at_s: f64,
    valid_until_s: f64,
    authority_scale: f32,
    artifact_digest: HumanoidEvidenceDigest,
}

impl HumanoidPhysicalAuthorityEvidenceArtifact {
    pub fn from_health_frame(
        subject: &HumanoidQualificationSubject,
        frame: &HumanoidPhysicalHealthFrame,
        config: PhysicalHealthAuthorityConfig,
        monitor_artifact_digest: HumanoidEvidenceDigest,
        calibration_artifact_digest: HumanoidEvidenceDigest,
        now_s: f64,
    ) -> Result<Self, HumanoidAuthorityEvidenceArtifactFailure> {
        let subject_digest = digest_subject(subject)
            .ok_or(HumanoidAuthorityEvidenceArtifactFailure::InvalidSubject)?;
        if monitor_artifact_digest.is_zero() {
            return Err(HumanoidAuthorityEvidenceArtifactFailure::InvalidProducerIdentity);
        }
        if calibration_artifact_digest.is_zero() {
            return Err(HumanoidAuthorityEvidenceArtifactFailure::InvalidCalibrationIdentity);
        }
        let envelope = frame
            .evaluate(subject.morphology, now_s, config)
            .map_err(|_| HumanoidAuthorityEvidenceArtifactFailure::InvalidPhysicalEvidence)?;
        let valid_until_s = frame.sampled_at_s + config.maximum_frame_age_s;
        if !valid_until_s.is_finite() || valid_until_s <= now_s {
            return Err(HumanoidAuthorityEvidenceArtifactFailure::PhysicalEvidenceExpired);
        }

        let frame_digest = digest_physical_frame(frame);
        let config_digest = digest_physical_config(config);
        if frame_digest.is_zero() || config_digest.is_zero() {
            return Err(HumanoidAuthorityEvidenceArtifactFailure::InvalidDigest);
        }
        let authority_scale = envelope.physical_authority();
        let mut value = Self {
            subject_digest,
            monitor_artifact_digest,
            calibration_artifact_digest,
            frame_digest,
            config_digest,
            evaluated_at_s: now_s,
            valid_until_s,
            authority_scale,
            artifact_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.artifact_digest = digest_physical_artifact(&value, envelope);
        if value.artifact_digest.is_zero() {
            return Err(HumanoidAuthorityEvidenceArtifactFailure::InvalidDigest);
        }
        Ok(value)
    }

    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.artifact_digest
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

    #[allow(clippy::too_many_arguments)]
    pub fn unsigned_claim(
        &self,
        subject: HumanoidQualificationSubject,
        issuer_id: impl Into<String>,
        key_id: impl Into<String>,
        revocation_epoch: u64,
        scheme_id: impl Into<String>,
    ) -> Option<HumanoidAuthorityUnsignedClaim> {
        if digest_subject(&subject) != Some(self.subject_digest) {
            return None;
        }
        HumanoidAuthorityUnsignedClaim::new(
            subject,
            HumanoidVerifiedAuthorityKind::Physical,
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidEpistemicAuthorityEvidenceArtifact {
    subject_digest: HumanoidEvidenceDigest,
    estimator_artifact_digest: HumanoidEvidenceDigest,
    state_digest: HumanoidEvidenceDigest,
    uncertainty_digest: HumanoidEvidenceDigest,
    config_digest: HumanoidEvidenceDigest,
    evaluated_at_s: f64,
    valid_until_s: f64,
    authority_scale: f32,
    artifact_digest: HumanoidEvidenceDigest,
}

impl HumanoidEpistemicAuthorityEvidenceArtifact {
    pub fn from_state_estimate(
        subject: &HumanoidQualificationSubject,
        estimated_state: &HumanoidState,
        uncertainty: HumanoidStateUncertaintyEnvelope,
        config: &StateEstimatorConfig,
        estimator_artifact_digest: HumanoidEvidenceDigest,
        evaluated_at_s: f64,
    ) -> Result<Self, HumanoidAuthorityEvidenceArtifactFailure> {
        let subject_digest = digest_subject(subject)
            .ok_or(HumanoidAuthorityEvidenceArtifactFailure::InvalidSubject)?;
        if estimator_artifact_digest.is_zero() {
            return Err(HumanoidAuthorityEvidenceArtifactFailure::InvalidProducerIdentity);
        }
        if !valid_estimator_config(config)
            || !valid_uncertainty(uncertainty)
            || !uncertainty_matches_config(uncertainty, config)
            || estimated_state.validate_for(subject.morphology).is_err()
            || !evaluated_at_s.is_finite()
            || evaluated_at_s < 0.0
        {
            return Err(HumanoidAuthorityEvidenceArtifactFailure::InvalidEstimatorEvidence);
        }

        let expected_age_s = evaluated_at_s - estimated_state.timestamp;
        if expected_age_s < -TIME_MATCH_TOLERANCE_S
            || (expected_age_s - uncertainty.measurement_age.observed).abs() > TIME_MATCH_TOLERANCE_S
        {
            return Err(HumanoidAuthorityEvidenceArtifactFailure::EstimatorStateMismatch);
        }

        let remaining_age_s =
            uncertainty.measurement_age.rejection_bound - uncertainty.measurement_age.observed;
        if !remaining_age_s.is_finite() || remaining_age_s <= 0.0 {
            return Err(HumanoidAuthorityEvidenceArtifactFailure::EpistemicEvidenceExpired);
        }
        let valid_until_s = evaluated_at_s + remaining_age_s;
        if !valid_until_s.is_finite() || valid_until_s <= evaluated_at_s {
            return Err(HumanoidAuthorityEvidenceArtifactFailure::EpistemicEvidenceExpired);
        }

        let state_digest = digest_state(subject, estimated_state);
        let uncertainty_digest = digest_uncertainty(uncertainty);
        let config_digest = digest_estimator_config(config);
        if state_digest.is_zero() || uncertainty_digest.is_zero() || config_digest.is_zero() {
            return Err(HumanoidAuthorityEvidenceArtifactFailure::InvalidDigest);
        }
        let authority_scale = uncertainty.epistemic_authority();
        let mut value = Self {
            subject_digest,
            estimator_artifact_digest,
            state_digest,
            uncertainty_digest,
            config_digest,
            evaluated_at_s,
            valid_until_s,
            authority_scale,
            artifact_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.artifact_digest = digest_epistemic_artifact(&value);
        if value.artifact_digest.is_zero() {
            return Err(HumanoidAuthorityEvidenceArtifactFailure::InvalidDigest);
        }
        Ok(value)
    }

    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.artifact_digest
    }

    pub const fn state_digest(&self) -> HumanoidEvidenceDigest {
        self.state_digest
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

    #[allow(clippy::too_many_arguments)]
    pub fn unsigned_claim(
        &self,
        subject: HumanoidQualificationSubject,
        issuer_id: impl Into<String>,
        key_id: impl Into<String>,
        revocation_epoch: u64,
        scheme_id: impl Into<String>,
    ) -> Option<HumanoidAuthorityUnsignedClaim> {
        if digest_subject(&subject) != Some(self.subject_digest) {
            return None;
        }
        HumanoidAuthorityUnsignedClaim::new(
            subject,
            HumanoidVerifiedAuthorityKind::Epistemic,
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
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("humanoid.authority-evidence-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn digest_physical_frame(frame: &HumanoidPhysicalHealthFrame) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.physical-health-frame.v1");
    h.string(frame.morphology.schema_id())
        .u64(frame.sequence)
        .f64(frame.sampled_at_s)
        .f64(frame.received_at_s)
        .u64(frame.calibration_fingerprint)
        .usize(frame.actuators.len());
    for actuator in &frame.actuators {
        h.bool(actuator.enabled)
            .bool(actuator.feedback_valid)
            .f64(actuator.current_a)
            .f64(actuator.current_limit_a)
            .bool(actuator.temperature_c.is_some());
        if let Some(value) = actuator.temperature_c {
            h.f64(value);
        }
        h.f64(actuator.warning_temperature_c)
            .f64(actuator.shutdown_temperature_c);
    }
    h.f64(frame.power.bus_voltage_v)
        .f64(frame.power.minimum_bus_voltage_v)
        .f64(frame.power.nominal_bus_voltage_v)
        .f64(frame.power.pack_current_a)
        .f64(frame.power.pack_current_limit_a)
        .f64(frame.power.state_of_charge)
        .bool(frame.latched_fault);
    h.finish()
}

fn digest_physical_config(config: PhysicalHealthAuthorityConfig) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.physical-health-authority-config.v1");
    h.f64(config.full_freshness_age_s)
        .f64(config.maximum_frame_age_s)
        .f64(config.current_derate_start_fraction)
        .f32(config.missing_temperature_authority)
        .f64(config.minimum_state_of_charge)
        .f64(config.full_state_of_charge);
    h.finish()
}

fn digest_physical_artifact(
    value: &HumanoidPhysicalAuthorityEvidenceArtifact,
    envelope: HumanoidPhysicalHealthEnvelope,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.physical-authority-evidence.v1");
    h.u32(HUMANOID_PHYSICAL_AUTHORITY_EVIDENCE_SCHEMA_VERSION)
        .digest(value.subject_digest)
        .digest(value.monitor_artifact_digest)
        .digest(value.calibration_artifact_digest)
        .digest(value.frame_digest)
        .digest(value.config_digest)
        .u64(envelope.sequence)
        .f64(envelope.frame_age_s)
        .f32(envelope.freshness_authority)
        .f32(envelope.actuator_authority)
        .f32(envelope.power_authority)
        .bool(envelope.latched_fault)
        .f64(value.evaluated_at_s)
        .f64(value.valid_until_s)
        .f32(value.authority_scale);
    h.finish()
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

fn valid_bound(value: BoundedUncertainty) -> bool {
    value.observed.is_finite()
        && value.observed >= 0.0
        && value.rejection_bound.is_finite()
        && value.rejection_bound > 0.0
}

fn valid_uncertainty(value: HumanoidStateUncertaintyEnvelope) -> bool {
    valid_bound(value.measurement_age)
        && valid_bound(value.orientation_innovation)
        && valid_bound(value.linear_velocity_innovation)
        && valid_bound(value.maximum_joint_position_innovation)
        && value.contact_trust.is_finite()
        && (0.0..=1.0).contains(&value.contact_trust)
}

fn uncertainty_matches_config(
    value: HumanoidStateUncertaintyEnvelope,
    config: &StateEstimatorConfig,
) -> bool {
    value.measurement_age.rejection_bound.to_bits() == config.maximum_measurement_age_s.to_bits()
        && value.orientation_innovation.rejection_bound.to_bits()
            == config.maximum_orientation_innovation_rad.to_bits()
        && value.linear_velocity_innovation.rejection_bound.to_bits()
            == config.maximum_linear_velocity_innovation_mps.to_bits()
        && value.maximum_joint_position_innovation.rejection_bound.to_bits()
            == config.maximum_joint_position_innovation_rad.to_bits()
}

fn digest_estimator_config(config: &StateEstimatorConfig) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.state-estimator-config.v1");
    h.f64(config.orientation_correction)
        .f64(config.linear_velocity_correction)
        .f64(config.angular_velocity_correction)
        .f64(config.joint_position_correction)
        .f64(config.joint_velocity_correction)
        .f64(config.double_support_velocity_damping)
        .f64(config.maximum_measurement_age_s)
        .f64(config.maximum_dt_s)
        .f64(config.maximum_orientation_innovation_rad)
        .f64(config.maximum_linear_velocity_innovation_mps)
        .f64(config.maximum_joint_position_innovation_rad);
    h.finish()
}

fn digest_uncertainty(value: HumanoidStateUncertaintyEnvelope) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.state-uncertainty-evidence.v1");
    h.u64(value.sequence)
        .bool(value.accepted);
    hash_bound(&mut h, value.measurement_age);
    hash_bound(&mut h, value.orientation_innovation);
    hash_bound(&mut h, value.linear_velocity_innovation);
    hash_bound(&mut h, value.maximum_joint_position_innovation);
    h.f32(value.contact_trust);
    h.finish()
}

fn hash_bound(hasher: &mut HumanoidEvidenceHasher, value: BoundedUncertainty) {
    hasher.f64(value.observed).f64(value.rejection_bound);
}

fn digest_state(
    subject: &HumanoidQualificationSubject,
    state: &HumanoidState,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.fused-state-evidence.v1");
    h.string(subject.morphology.schema_id())
        .f64(state.root_height);
    for value in state.root_position { h.f64(value); }
    for value in state.root_quaternion { h.f64(value); }
    h.usize(state.joint_angles.len());
    for &value in &state.joint_angles { h.f64(value); }
    for value in state.root_linear_velocity { h.f64(value); }
    for value in state.root_angular_velocity { h.f64(value); }
    h.usize(state.joint_velocities.len());
    for &value in &state.joint_velocities { h.f64(value); }
    h.f64(state.head_height);
    for value in state.torso_vertical { h.f64(value); }
    h.usize(state.extremities.len());
    for &value in &state.extremities { h.f64(value); }
    for value in state.com_velocity { h.f64(value); }
    h.f64(state.timestamp);
    h.finish()
}

fn digest_epistemic_artifact(value: &HumanoidEpistemicAuthorityEvidenceArtifact) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.epistemic-authority-evidence.v1");
    h.u32(HUMANOID_EPISTEMIC_AUTHORITY_EVIDENCE_SCHEMA_VERSION)
        .digest(value.subject_digest)
        .digest(value.estimator_artifact_digest)
        .digest(value.state_digest)
        .digest(value.uncertainty_digest)
        .digest(value.config_digest)
        .f64(value.evaluated_at_s)
        .f64(value.valid_until_s)
        .f32(value.authority_scale);
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
