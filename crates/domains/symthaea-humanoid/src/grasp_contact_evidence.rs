// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Non-actuating grasp-contact evidence for humanoid manipulation.
//!
//! This module establishes the physical semantics Grasp needs before any grip
//! command or operational authority is introduced. It records one exact hand's
//! contact with one exact object state, derives normal/tangential wrench and
//! relative-motion metrics, applies an explicit freshness/contact policy, and
//! emits an immutable SHA-256 assessment artifact.
//!
//! It deliberately does not command finger torque, infer an unknown friction
//! coefficient, or claim retention from geometry alone.

use crate::contact_site::HumanoidContactSite;
use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::morphology::HandSide;
use crate::qualification::HumanoidQualificationSubject;
use crate::types::{ActuationMode, HumanoidTask};

pub const HUMANOID_GRASP_CONTACT_OBSERVATION_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_CONTACT_POLICY_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_CONTACT_ASSESSMENT_SCHEMA_VERSION: u32 = 1;
const NORMAL_VECTOR_EPSILON: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HumanoidManipulationContactSource {
    Unavailable,
    KinematicEstimate,
    VisionEstimate,
    SolverWrench,
    ForceTorqueSensor,
    TactileArray,
    FusedMeasured,
}

impl HumanoidManipulationContactSource {
    pub const fn quality_rank(self) -> u8 {
        match self {
            Self::Unavailable => 0,
            Self::KinematicEstimate => 1,
            Self::VisionEstimate => 2,
            Self::SolverWrench => 3,
            Self::ForceTorqueSensor | Self::TactileArray => 4,
            Self::FusedMeasured => 5,
        }
    }

    pub const fn is_measured(self) -> bool {
        matches!(
            self,
            Self::SolverWrench
                | Self::ForceTorqueSensor
                | Self::TactileArray
                | Self::FusedMeasured
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspContactObservationFailure {
    InvalidSubject,
    WrongTask,
    InvalidObjectIdentity,
    InvalidObjectStateDigest,
    InvalidHandSite,
    InvalidTimestamp,
    InvalidContactPoint,
    InvalidNormal,
    InvalidWrench,
    InvalidRelativeVelocity,
    InvalidConfidence,
    InvalidSource,
    InvalidDigest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspContactObservation {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    object_id: String,
    object_state_digest: HumanoidEvidenceDigest,
    hand: HandSide,
    site: HumanoidContactSite,
    in_contact: bool,
    contact_point_world_m: [f64; 3],
    outward_normal_world_unit: [f64; 3],
    force_world_n: [f64; 3],
    torque_world_nm: [f64; 3],
    /// Velocity of the hand contact point relative to the object at the contact
    /// point. Positive projection along the outward normal means separation.
    hand_relative_to_object_velocity_world_mps: [f64; 3],
    confidence: f64,
    source: HumanoidManipulationContactSource,
    timestamp_s: f64,
    observation_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspContactObservation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        object_id: impl Into<String>,
        object_state_digest: HumanoidEvidenceDigest,
        hand: HandSide,
        site: HumanoidContactSite,
        in_contact: bool,
        contact_point_world_m: [f64; 3],
        outward_normal_world: [f64; 3],
        force_world_n: [f64; 3],
        torque_world_nm: [f64; 3],
        hand_relative_to_object_velocity_world_mps: [f64; 3],
        confidence: f64,
        source: HumanoidManipulationContactSource,
        timestamp_s: f64,
    ) -> Result<Self, HumanoidGraspContactObservationFailure> {
        if !subject.validate() {
            return Err(HumanoidGraspContactObservationFailure::InvalidSubject);
        }
        if subject.task != HumanoidTask::Grasp {
            return Err(HumanoidGraspContactObservationFailure::WrongTask);
        }
        let object_id = object_id.into();
        if !valid_id(&object_id) {
            return Err(HumanoidGraspContactObservationFailure::InvalidObjectIdentity);
        }
        if object_state_digest.is_zero() {
            return Err(HumanoidGraspContactObservationFailure::InvalidObjectStateDigest);
        }
        if site != expected_hand_site(hand) {
            return Err(HumanoidGraspContactObservationFailure::InvalidHandSite);
        }
        if !timestamp_s.is_finite() || timestamp_s < 0.0 {
            return Err(HumanoidGraspContactObservationFailure::InvalidTimestamp);
        }
        if !finite_vec3(contact_point_world_m) {
            return Err(HumanoidGraspContactObservationFailure::InvalidContactPoint);
        }
        let outward_normal_world_unit = normalize_vec3(outward_normal_world)
            .ok_or(HumanoidGraspContactObservationFailure::InvalidNormal)?;
        if !finite_vec3(force_world_n) || !finite_vec3(torque_world_nm) {
            return Err(HumanoidGraspContactObservationFailure::InvalidWrench);
        }
        if !finite_vec3(hand_relative_to_object_velocity_world_mps) {
            return Err(HumanoidGraspContactObservationFailure::InvalidRelativeVelocity);
        }
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(HumanoidGraspContactObservationFailure::InvalidConfidence);
        }
        if source == HumanoidManipulationContactSource::Unavailable && in_contact {
            return Err(HumanoidGraspContactObservationFailure::InvalidSource);
        }

        let mut value = Self {
            schema_version: HUMANOID_GRASP_CONTACT_OBSERVATION_SCHEMA_VERSION,
            subject_digest: digest_subject(subject)
                .ok_or(HumanoidGraspContactObservationFailure::InvalidSubject)?,
            object_id,
            object_state_digest,
            hand,
            site,
            in_contact,
            contact_point_world_m,
            outward_normal_world_unit,
            force_world_n,
            torque_world_nm,
            hand_relative_to_object_velocity_world_mps,
            confidence,
            source,
            timestamp_s,
            observation_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.observation_digest = digest_observation(&value);
        if value.observation_digest.is_zero() || !value.validate_for(subject) {
            return Err(HumanoidGraspContactObservationFailure::InvalidDigest);
        }
        Ok(value)
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_GRASP_CONTACT_OBSERVATION_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && valid_id(&self.object_id)
            && !self.object_state_digest.is_zero()
            && self.site == expected_hand_site(self.hand)
            && self.timestamp_s.is_finite()
            && self.timestamp_s >= 0.0
            && finite_vec3(self.contact_point_world_m)
            && finite_vec3(self.outward_normal_world_unit)
            && (norm3(self.outward_normal_world_unit) - 1.0).abs() <= 1.0e-9
            && finite_vec3(self.force_world_n)
            && finite_vec3(self.torque_world_nm)
            && finite_vec3(self.hand_relative_to_object_velocity_world_mps)
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
            && !(self.source == HumanoidManipulationContactSource::Unavailable && self.in_contact)
            && !self.observation_digest.is_zero()
            && self.observation_digest == digest_observation(self)
    }

    pub fn object_id(&self) -> &str { &self.object_id }
    pub const fn object_state_digest(&self) -> HumanoidEvidenceDigest { self.object_state_digest }
    pub const fn hand(&self) -> HandSide { self.hand }
    pub const fn site(&self) -> HumanoidContactSite { self.site }
    pub const fn source(&self) -> HumanoidManipulationContactSource { self.source }
    pub const fn in_contact(&self) -> bool { self.in_contact }
    pub const fn timestamp_s(&self) -> f64 { self.timestamp_s }
    pub const fn confidence(&self) -> f64 { self.confidence }
    pub const fn observation_digest(&self) -> HumanoidEvidenceDigest { self.observation_digest }

    pub fn age_s(&self, now_s: f64) -> f64 {
        if !now_s.is_finite() || now_s < self.timestamp_s {
            return f64::INFINITY;
        }
        now_s - self.timestamp_s
    }

    /// Positive means the hand is pushing into the object, opposite the object's
    /// outward surface normal.
    pub fn normal_force_n(&self) -> f64 {
        -dot3(self.force_world_n, self.outward_normal_world_unit)
    }

    pub fn tangential_force_n(&self) -> f64 {
        let signed_outward = dot3(self.force_world_n, self.outward_normal_world_unit);
        norm3(sub3(
            self.force_world_n,
            scale3(self.outward_normal_world_unit, signed_outward),
        ))
    }

    pub fn separation_speed_mps(&self) -> f64 {
        dot3(
            self.hand_relative_to_object_velocity_world_mps,
            self.outward_normal_world_unit,
        )
        .max(0.0)
    }

    pub fn closing_speed_mps(&self) -> f64 {
        (-dot3(
            self.hand_relative_to_object_velocity_world_mps,
            self.outward_normal_world_unit,
        ))
        .max(0.0)
    }

    pub fn tangential_speed_mps(&self) -> f64 {
        let normal_speed = dot3(
            self.hand_relative_to_object_velocity_world_mps,
            self.outward_normal_world_unit,
        );
        norm3(sub3(
            self.hand_relative_to_object_velocity_world_mps,
            scale3(self.outward_normal_world_unit, normal_speed),
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspContactPolicy {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    hand: HandSide,
    maximum_contact_age_s: f64,
    minimum_confidence: f64,
    minimum_source_quality_rank: u8,
    require_measured_source: bool,
    minimum_normal_force_n: f64,
    maximum_normal_force_n: f64,
    maximum_tangential_speed_mps: f64,
    maximum_separation_speed_mps: f64,
    maximum_closing_speed_mps: f64,
    policy_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspContactPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        hand: HandSide,
        maximum_contact_age_s: f64,
        minimum_confidence: f64,
        minimum_source_quality_rank: u8,
        require_measured_source: bool,
        minimum_normal_force_n: f64,
        maximum_normal_force_n: f64,
        maximum_tangential_speed_mps: f64,
        maximum_separation_speed_mps: f64,
        maximum_closing_speed_mps: f64,
    ) -> Option<Self> {
        if !subject.validate()
            || subject.task != HumanoidTask::Grasp
            || !maximum_contact_age_s.is_finite()
            || maximum_contact_age_s <= 0.0
            || !minimum_confidence.is_finite()
            || !(0.0..=1.0).contains(&minimum_confidence)
            || minimum_source_quality_rank > HumanoidManipulationContactSource::FusedMeasured.quality_rank()
            || !minimum_normal_force_n.is_finite()
            || minimum_normal_force_n < 0.0
            || !maximum_normal_force_n.is_finite()
            || maximum_normal_force_n <= minimum_normal_force_n
            || !maximum_tangential_speed_mps.is_finite()
            || maximum_tangential_speed_mps < 0.0
            || !maximum_separation_speed_mps.is_finite()
            || maximum_separation_speed_mps < 0.0
            || !maximum_closing_speed_mps.is_finite()
            || maximum_closing_speed_mps < 0.0
        {
            return None;
        }
        let mut value = Self {
            schema_version: HUMANOID_GRASP_CONTACT_POLICY_SCHEMA_VERSION,
            subject_digest: digest_subject(subject)?,
            hand,
            maximum_contact_age_s,
            minimum_confidence,
            minimum_source_quality_rank,
            require_measured_source,
            minimum_normal_force_n,
            maximum_normal_force_n,
            maximum_tangential_speed_mps,
            maximum_separation_speed_mps,
            maximum_closing_speed_mps,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.policy_digest = digest_policy(&value);
        value.validate_for(subject).then_some(value)
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_GRASP_CONTACT_POLICY_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && self.maximum_contact_age_s.is_finite()
            && self.maximum_contact_age_s > 0.0
            && self.minimum_confidence.is_finite()
            && (0.0..=1.0).contains(&self.minimum_confidence)
            && self.minimum_source_quality_rank <= HumanoidManipulationContactSource::FusedMeasured.quality_rank()
            && self.minimum_normal_force_n.is_finite()
            && self.minimum_normal_force_n >= 0.0
            && self.maximum_normal_force_n.is_finite()
            && self.maximum_normal_force_n > self.minimum_normal_force_n
            && self.maximum_tangential_speed_mps.is_finite()
            && self.maximum_tangential_speed_mps >= 0.0
            && self.maximum_separation_speed_mps.is_finite()
            && self.maximum_separation_speed_mps >= 0.0
            && self.maximum_closing_speed_mps.is_finite()
            && self.maximum_closing_speed_mps >= 0.0
            && !self.policy_digest.is_zero()
            && self.policy_digest == digest_policy(self)
    }

    pub const fn hand(&self) -> HandSide { self.hand }
    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest { self.policy_digest }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspContactFailureKind {
    HandMismatch,
    NoContact,
    Stale,
    ConfidenceTooLow,
    SourceQualityTooLow,
    SourceNotMeasured,
    NormalForceTooLow,
    NormalForceTooHigh,
    TangentialSlipTooFast,
    SeparationTooFast,
    ClosingTooFast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspContactAssessmentFailure {
    InvalidTime,
    InvalidObservation,
    InvalidPolicy,
    InvalidDigest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspContactAssessment {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    object_id: String,
    object_state_digest: HumanoidEvidenceDigest,
    hand: HandSide,
    observation_digest: HumanoidEvidenceDigest,
    policy_digest: HumanoidEvidenceDigest,
    age_s: f64,
    normal_force_n: f64,
    tangential_force_n: f64,
    tangential_speed_mps: f64,
    separation_speed_mps: f64,
    closing_speed_mps: f64,
    confidence: f64,
    source: HumanoidManipulationContactSource,
    failures: Vec<HumanoidGraspContactFailureKind>,
    accepted: bool,
    assessed_at_s: f64,
    assessment_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspContactAssessment {
    pub fn object_id(&self) -> &str { &self.object_id }
    pub const fn object_state_digest(&self) -> HumanoidEvidenceDigest { self.object_state_digest }
    pub const fn hand(&self) -> HandSide { self.hand }
    pub const fn observation_digest(&self) -> HumanoidEvidenceDigest { self.observation_digest }
    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest { self.policy_digest }
    pub const fn normal_force_n(&self) -> f64 { self.normal_force_n }
    pub const fn tangential_force_n(&self) -> f64 { self.tangential_force_n }
    pub const fn tangential_speed_mps(&self) -> f64 { self.tangential_speed_mps }
    pub fn failures(&self) -> &[HumanoidGraspContactFailureKind] { &self.failures }
    pub const fn accepted(&self) -> bool { self.accepted }
    pub const fn assessment_digest(&self) -> HumanoidEvidenceDigest { self.assessment_digest }

    pub fn validate(
        &self,
        subject: &HumanoidQualificationSubject,
        observation: &HumanoidGraspContactObservation,
        policy: &HumanoidGraspContactPolicy,
    ) -> bool {
        self.schema_version == HUMANOID_GRASP_CONTACT_ASSESSMENT_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && observation.validate_for(subject)
            && policy.validate_for(subject)
            && self.object_id == observation.object_id
            && self.object_state_digest == observation.object_state_digest
            && self.hand == observation.hand
            && self.observation_digest == observation.observation_digest
            && self.policy_digest == policy.policy_digest
            && self.accepted == self.failures.is_empty()
            && !self.assessment_digest.is_zero()
            && self.assessment_digest == digest_assessment(self)
    }
}

pub fn assess_humanoid_grasp_contact(
    subject: &HumanoidQualificationSubject,
    observation: &HumanoidGraspContactObservation,
    policy: &HumanoidGraspContactPolicy,
    now_s: f64,
) -> Result<HumanoidGraspContactAssessment, HumanoidGraspContactAssessmentFailure> {
    if !now_s.is_finite() || now_s < 0.0 {
        return Err(HumanoidGraspContactAssessmentFailure::InvalidTime);
    }
    if !observation.validate_for(subject) {
        return Err(HumanoidGraspContactAssessmentFailure::InvalidObservation);
    }
    if !policy.validate_for(subject) {
        return Err(HumanoidGraspContactAssessmentFailure::InvalidPolicy);
    }

    let age_s = observation.age_s(now_s);
    let normal_force_n = observation.normal_force_n();
    let tangential_force_n = observation.tangential_force_n();
    let tangential_speed_mps = observation.tangential_speed_mps();
    let separation_speed_mps = observation.separation_speed_mps();
    let closing_speed_mps = observation.closing_speed_mps();
    let mut failures = Vec::new();

    if observation.hand != policy.hand || observation.site != expected_hand_site(policy.hand) {
        failures.push(HumanoidGraspContactFailureKind::HandMismatch);
    }
    if !observation.in_contact {
        failures.push(HumanoidGraspContactFailureKind::NoContact);
    }
    if !age_s.is_finite() || age_s > policy.maximum_contact_age_s {
        failures.push(HumanoidGraspContactFailureKind::Stale);
    }
    if observation.confidence < policy.minimum_confidence {
        failures.push(HumanoidGraspContactFailureKind::ConfidenceTooLow);
    }
    if observation.source.quality_rank() < policy.minimum_source_quality_rank {
        failures.push(HumanoidGraspContactFailureKind::SourceQualityTooLow);
    }
    if policy.require_measured_source && !observation.source.is_measured() {
        failures.push(HumanoidGraspContactFailureKind::SourceNotMeasured);
    }
    if normal_force_n < policy.minimum_normal_force_n {
        failures.push(HumanoidGraspContactFailureKind::NormalForceTooLow);
    }
    if normal_force_n > policy.maximum_normal_force_n {
        failures.push(HumanoidGraspContactFailureKind::NormalForceTooHigh);
    }
    if tangential_speed_mps > policy.maximum_tangential_speed_mps {
        failures.push(HumanoidGraspContactFailureKind::TangentialSlipTooFast);
    }
    if separation_speed_mps > policy.maximum_separation_speed_mps {
        failures.push(HumanoidGraspContactFailureKind::SeparationTooFast);
    }
    if closing_speed_mps > policy.maximum_closing_speed_mps {
        failures.push(HumanoidGraspContactFailureKind::ClosingTooFast);
    }

    let mut assessment = HumanoidGraspContactAssessment {
        schema_version: HUMANOID_GRASP_CONTACT_ASSESSMENT_SCHEMA_VERSION,
        subject_digest: digest_subject(subject)
            .ok_or(HumanoidGraspContactAssessmentFailure::InvalidObservation)?,
        object_id: observation.object_id.clone(),
        object_state_digest: observation.object_state_digest,
        hand: observation.hand,
        observation_digest: observation.observation_digest,
        policy_digest: policy.policy_digest,
        age_s,
        normal_force_n,
        tangential_force_n,
        tangential_speed_mps,
        separation_speed_mps,
        closing_speed_mps,
        confidence: observation.confidence,
        source: observation.source,
        accepted: failures.is_empty(),
        failures,
        assessed_at_s: now_s,
        assessment_digest: HumanoidEvidenceDigest::ZERO,
    };
    assessment.assessment_digest = digest_assessment(&assessment);
    if !assessment.validate(subject, observation, policy) {
        return Err(HumanoidGraspContactAssessmentFailure::InvalidDigest);
    }
    Ok(assessment)
}

fn expected_hand_site(hand: HandSide) -> HumanoidContactSite {
    match hand {
        HandSide::Right => HumanoidContactSite::RightHand,
        HandSide::Left => HumanoidContactSite::LeftHand,
    }
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() || subject.task != HumanoidTask::Grasp {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-contact-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn digest_observation(value: &HumanoidGraspContactObservation) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-contact-observation.v1");
    h.u32(value.schema_version)
        .digest(value.subject_digest)
        .string(&value.object_id)
        .digest(value.object_state_digest)
        .u64(hand_id(value.hand))
        .string(value.site.canonical_id())
        .bool(value.in_contact);
    hash_vec3(&mut h, value.contact_point_world_m);
    hash_vec3(&mut h, value.outward_normal_world_unit);
    hash_vec3(&mut h, value.force_world_n);
    hash_vec3(&mut h, value.torque_world_nm);
    hash_vec3(&mut h, value.hand_relative_to_object_velocity_world_mps);
    h.f64(value.confidence)
        .u64(contact_source_id(value.source))
        .f64(value.timestamp_s);
    h.finish()
}

fn digest_policy(value: &HumanoidGraspContactPolicy) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-contact-policy.v1");
    h.u32(value.schema_version)
        .digest(value.subject_digest)
        .u64(hand_id(value.hand))
        .f64(value.maximum_contact_age_s)
        .f64(value.minimum_confidence)
        .u64(value.minimum_source_quality_rank as u64)
        .bool(value.require_measured_source)
        .f64(value.minimum_normal_force_n)
        .f64(value.maximum_normal_force_n)
        .f64(value.maximum_tangential_speed_mps)
        .f64(value.maximum_separation_speed_mps)
        .f64(value.maximum_closing_speed_mps);
    h.finish()
}

fn digest_assessment(value: &HumanoidGraspContactAssessment) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-contact-assessment.v1");
    h.u32(value.schema_version)
        .digest(value.subject_digest)
        .string(&value.object_id)
        .digest(value.object_state_digest)
        .u64(hand_id(value.hand))
        .digest(value.observation_digest)
        .digest(value.policy_digest)
        .f64(value.age_s)
        .f64(value.normal_force_n)
        .f64(value.tangential_force_n)
        .f64(value.tangential_speed_mps)
        .f64(value.separation_speed_mps)
        .f64(value.closing_speed_mps)
        .f64(value.confidence)
        .u64(contact_source_id(value.source))
        .usize(value.failures.len());
    for failure in &value.failures {
        h.u64(failure_id(*failure));
    }
    h.bool(value.accepted).f64(value.assessed_at_s);
    h.finish()
}

fn hash_vec3(h: &mut HumanoidEvidenceHasher, value: [f64; 3]) {
    for component in value { h.f64(component); }
}

fn finite_vec3(value: [f64; 3]) -> bool { value.into_iter().all(f64::is_finite) }

fn normalize_vec3(value: [f64; 3]) -> Option<[f64; 3]> {
    if !finite_vec3(value) { return None; }
    let norm = norm3(value);
    if !norm.is_finite() || norm <= NORMAL_VECTOR_EPSILON { return None; }
    Some([value[0] / norm, value[1] / norm, value[2] / norm])
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn scale3(value: [f64; 3], scale: f64) -> [f64; 3] { [value[0] * scale, value[1] * scale, value[2] * scale] }
fn norm3(value: [f64; 3]) -> f64 { dot3(value, value).sqrt() }

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value.bytes().all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

fn hand_id(hand: HandSide) -> u64 {
    match hand { HandSide::Right => 1, HandSide::Left => 2 }
}

fn contact_source_id(source: HumanoidManipulationContactSource) -> u64 {
    match source {
        HumanoidManipulationContactSource::Unavailable => 0,
        HumanoidManipulationContactSource::KinematicEstimate => 1,
        HumanoidManipulationContactSource::VisionEstimate => 2,
        HumanoidManipulationContactSource::SolverWrench => 3,
        HumanoidManipulationContactSource::ForceTorqueSensor => 4,
        HumanoidManipulationContactSource::TactileArray => 5,
        HumanoidManipulationContactSource::FusedMeasured => 6,
    }
}

fn failure_id(failure: HumanoidGraspContactFailureKind) -> u64 {
    match failure {
        HumanoidGraspContactFailureKind::HandMismatch => 1,
        HumanoidGraspContactFailureKind::NoContact => 2,
        HumanoidGraspContactFailureKind::Stale => 3,
        HumanoidGraspContactFailureKind::ConfidenceTooLow => 4,
        HumanoidGraspContactFailureKind::SourceQualityTooLow => 5,
        HumanoidGraspContactFailureKind::SourceNotMeasured => 6,
        HumanoidGraspContactFailureKind::NormalForceTooLow => 7,
        HumanoidGraspContactFailureKind::NormalForceTooHigh => 8,
        HumanoidGraspContactFailureKind::TangentialSlipTooFast => 9,
        HumanoidGraspContactFailureKind::SeparationTooFast => 10,
        HumanoidGraspContactFailureKind::ClosingTooFast => 11,
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
            HumanoidTask::Grasp,
            ActuationMode::NormalizedTorque,
            "grasp-contact-test-backend",
        )
    }

    fn policy() -> HumanoidGraspContactPolicy {
        HumanoidGraspContactPolicy::new(
            &subject(), HandSide::Right, 0.05, 0.8,
            HumanoidManipulationContactSource::SolverWrench.quality_rank(),
            true, 2.0, 50.0, 0.01, 0.005, 0.02,
        ).unwrap()
    }

    fn observation(relative_velocity: [f64; 3], timestamp_s: f64) -> HumanoidGraspContactObservation {
        HumanoidGraspContactObservation::new(
            &subject(), "object-17", HumanoidEvidenceDigest::from_bytes([7; 32]),
            HandSide::Right, HumanoidContactSite::RightHand, true,
            [0.4, -0.2, 1.1], [1.0, 0.0, 0.0], [-12.0, 0.5, 0.0], [0.0; 3],
            relative_velocity, 0.95, HumanoidManipulationContactSource::SolverWrench, timestamp_s,
        ).unwrap()
    }

    #[test]
    fn stable_measured_contact_is_accepted() {
        let o = observation([0.0, 0.002, 0.0], 1.0);
        let p = policy();
        let assessment = assess_humanoid_grasp_contact(&subject(), &o, &p, 1.01).unwrap();
        assert!(assessment.accepted(), "{:?}", assessment.failures());
        assert!(assessment.normal_force_n() >= 2.0);
        assert!(assessment.validate(&subject(), &o, &p));
    }

    #[test]
    fn tangential_slip_is_rejected() {
        let o = observation([0.0, 0.03, 0.0], 1.0);
        let p = policy();
        let assessment = assess_humanoid_grasp_contact(&subject(), &o, &p, 1.01).unwrap();
        assert!(!assessment.accepted());
        assert!(assessment.failures().contains(&HumanoidGraspContactFailureKind::TangentialSlipTooFast));
    }

    #[test]
    fn stale_contact_is_rejected() {
        let o = observation([0.0; 3], 1.0);
        let p = policy();
        let assessment = assess_humanoid_grasp_contact(&subject(), &o, &p, 1.20).unwrap();
        assert!(assessment.failures().contains(&HumanoidGraspContactFailureKind::Stale));
    }

    #[test]
    fn invalid_assessment_time_is_rejected() {
        let o = observation([0.0; 3], 1.0);
        let p = policy();
        assert_eq!(
            assess_humanoid_grasp_contact(&subject(), &o, &p, f64::NAN),
            Err(HumanoidGraspContactAssessmentFailure::InvalidTime)
        );
    }

    #[test]
    fn wrong_hand_site_fails_observation_construction() {
        let result = HumanoidGraspContactObservation::new(
            &subject(), "object-17", HumanoidEvidenceDigest::from_bytes([7; 32]),
            HandSide::Right, HumanoidContactSite::LeftHand, true,
            [0.0; 3], [1.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [0.0; 3], [0.0; 3],
            1.0, HumanoidManipulationContactSource::SolverWrench, 1.0,
        );
        assert_eq!(result, Err(HumanoidGraspContactObservationFailure::InvalidHandSite));
    }

    #[test]
    fn object_state_changes_observation_identity() {
        let a = observation([0.0; 3], 1.0);
        let b = HumanoidGraspContactObservation::new(
            &subject(), "object-17", HumanoidEvidenceDigest::from_bytes([8; 32]),
            HandSide::Right, HumanoidContactSite::RightHand, true,
            [0.4, -0.2, 1.1], [1.0, 0.0, 0.0], [-12.0, 0.5, 0.0], [0.0; 3], [0.0; 3],
            0.95, HumanoidManipulationContactSource::SolverWrench, 1.0,
        ).unwrap();
        assert_ne!(a.observation_digest(), b.observation_digest());
    }

    #[test]
    fn kinematic_contact_cannot_satisfy_measured_policy() {
        let kinematic = HumanoidGraspContactObservation::new(
            &subject(), "object-17", HumanoidEvidenceDigest::from_bytes([7; 32]),
            HandSide::Right, HumanoidContactSite::RightHand, true,
            [0.4, -0.2, 1.1], [1.0, 0.0, 0.0], [-12.0, 0.0, 0.0], [0.0; 3], [0.0; 3],
            1.0, HumanoidManipulationContactSource::KinematicEstimate, 1.0,
        ).unwrap();
        let p = policy();
        let assessment = assess_humanoid_grasp_contact(&subject(), &kinematic, &p, 1.01).unwrap();
        assert!(!assessment.accepted());
        assert!(assessment.failures().contains(&HumanoidGraspContactFailureKind::SourceNotMeasured));
    }
}
