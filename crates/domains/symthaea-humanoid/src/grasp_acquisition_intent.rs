// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Object-bound, non-actuating acquisition intent for humanoid Grasp.
//!
//! The existing whole-body IR already describes Grasp as an end-effector target
//! plus `ObjectContact(Acquire)`. This module does not invent a second command
//! language. It refines that IR with exact object identity/state and proves that
//! the live spatial permit, Grasp subject, contact/retention policies and every
//! Grasp objective still describe the same episode.
//!
//! The resulting value is planning/evidence state only. It is not motor authority
//! and cannot be lowered directly to actuator commands.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::grasp_contact_evidence::HumanoidGraspContactPolicy;
use crate::grasp_retention_evidence::HumanoidGraspRetentionPolicy;
use crate::morphology::HandSide;
use crate::qualification::HumanoidQualificationSubject;
use crate::skill_runtime::{HumanoidSkillIntent, HumanoidSkillRequirementRole};
use crate::spatial_goal::{
    HumanoidSpatialGoalEvidence, HumanoidSpatialTargetKind, HumanoidSpatiallyBoundSkillPermit,
};
use crate::types::{ActuationMode, HumanoidTask};
use crate::whole_body_intent::{
    HumanoidContactObjectiveMode, HumanoidWholeBodyInvariantIR, HumanoidWholeBodyMotionIntent,
    HumanoidWholeBodyObjectiveIR,
};

pub const HUMANOID_GRASP_OBJECT_BINDING_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_ACQUISITION_POLICY_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_ACQUISITION_INTENT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspObjectBindingFailure {
    InvalidSubject,
    InvalidGoal,
    GoalNotObject,
    InvalidObjectIdentity,
    InvalidObjectState,
    InvalidDigest,
}

/// Cryptographic refinement of one already-admitted object spatial goal.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspObjectBindingEvidence {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    object_id: String,
    object_state_digest: HumanoidEvidenceDigest,
    spatial_goal_id: String,
    hand: HandSide,
    target_world_m: [f64; 3],
    observed_at_s: f64,
    received_at_s: f64,
    confidence: f64,
    binding_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspObjectBindingEvidence {
    pub fn bind(
        subject: &HumanoidQualificationSubject,
        goal: &HumanoidSpatialGoalEvidence,
        object_id: impl Into<String>,
        object_state_digest: HumanoidEvidenceDigest,
    ) -> Result<Self, HumanoidGraspObjectBindingFailure> {
        let subject_digest = digest_subject(subject)
            .ok_or(HumanoidGraspObjectBindingFailure::InvalidSubject)?;
        if !goal.validate() {
            return Err(HumanoidGraspObjectBindingFailure::InvalidGoal);
        }
        if goal.kind != HumanoidSpatialTargetKind::Object {
            return Err(HumanoidGraspObjectBindingFailure::GoalNotObject);
        }
        let object_id = object_id.into();
        if !valid_id(&object_id) {
            return Err(HumanoidGraspObjectBindingFailure::InvalidObjectIdentity);
        }
        if object_state_digest.is_zero() {
            return Err(HumanoidGraspObjectBindingFailure::InvalidObjectState);
        }

        let mut value = Self {
            schema_version: HUMANOID_GRASP_OBJECT_BINDING_SCHEMA_VERSION,
            subject_digest,
            object_id,
            object_state_digest,
            spatial_goal_id: goal.goal_id.clone(),
            hand: goal.hand,
            target_world_m: goal.target_world_m,
            observed_at_s: goal.observed_at_s,
            received_at_s: goal.received_at_s,
            confidence: goal.confidence,
            binding_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.binding_digest = digest_object_binding(&value);
        if !value.validate_for(subject, goal) {
            return Err(HumanoidGraspObjectBindingFailure::InvalidDigest);
        }
        Ok(value)
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        goal: &HumanoidSpatialGoalEvidence,
    ) -> bool {
        self.schema_version == HUMANOID_GRASP_OBJECT_BINDING_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && goal.validate()
            && goal.kind == HumanoidSpatialTargetKind::Object
            && valid_id(&self.object_id)
            && !self.object_state_digest.is_zero()
            && self.spatial_goal_id == goal.goal_id
            && self.hand == goal.hand
            && same_vec3_bits(self.target_world_m, goal.target_world_m)
            && self.observed_at_s.to_bits() == goal.observed_at_s.to_bits()
            && self.received_at_s.to_bits() == goal.received_at_s.to_bits()
            && self.confidence.to_bits() == goal.confidence.to_bits()
            && !self.binding_digest.is_zero()
            && self.binding_digest == digest_object_binding(self)
    }

    pub fn object_id(&self) -> &str {
        &self.object_id
    }

    pub const fn object_state_digest(&self) -> HumanoidEvidenceDigest {
        self.object_state_digest
    }

    pub const fn hand(&self) -> HandSide {
        self.hand
    }

    pub fn spatial_goal_id(&self) -> &str {
        &self.spatial_goal_id
    }

    pub const fn binding_digest(&self) -> HumanoidEvidenceDigest {
        self.binding_digest
    }

    pub const fn confidence(&self) -> f64 {
        self.confidence
    }

    pub const fn observed_at_s(&self) -> f64 {
        self.observed_at_s
    }

    pub const fn received_at_s(&self) -> f64 {
        self.received_at_s
    }
}

/// Admission policy for turning a fresh object-bound Grasp IR into acquisition
/// planning state. These are plan limits, not actuator or product-safety limits.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspAcquisitionPolicy {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    hand: HandSide,
    contact_policy_digest: HumanoidEvidenceDigest,
    retention_policy_digest: HumanoidEvidenceDigest,
    maximum_object_binding_age_s: f64,
    minimum_object_confidence: f64,
    maximum_end_effector_speed_mps: f64,
    minimum_requested_contact_force_n: f64,
    maximum_requested_contact_force_n: f64,
    maximum_resulting_payload_kg: f64,
    policy_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspAcquisitionPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        contact_policy: &HumanoidGraspContactPolicy,
        retention_policy: &HumanoidGraspRetentionPolicy,
        maximum_object_binding_age_s: f64,
        minimum_object_confidence: f64,
        maximum_end_effector_speed_mps: f64,
        minimum_requested_contact_force_n: f64,
        maximum_requested_contact_force_n: f64,
        maximum_resulting_payload_kg: f64,
    ) -> Option<Self> {
        if !subject.validate()
            || subject.task != HumanoidTask::Grasp
            || !contact_policy.validate_for(subject)
            || !retention_policy.validate_for(subject, contact_policy)
            || !maximum_object_binding_age_s.is_finite()
            || maximum_object_binding_age_s <= 0.0
            || !minimum_object_confidence.is_finite()
            || !(0.0..=1.0).contains(&minimum_object_confidence)
            || !maximum_end_effector_speed_mps.is_finite()
            || maximum_end_effector_speed_mps <= 0.0
            || !minimum_requested_contact_force_n.is_finite()
            || minimum_requested_contact_force_n < 0.0
            || !maximum_requested_contact_force_n.is_finite()
            || maximum_requested_contact_force_n < minimum_requested_contact_force_n
            || !maximum_resulting_payload_kg.is_finite()
            || maximum_resulting_payload_kg < 0.0
        {
            return None;
        }

        let mut value = Self {
            schema_version: HUMANOID_GRASP_ACQUISITION_POLICY_SCHEMA_VERSION,
            subject_digest: digest_subject(subject)?,
            hand: contact_policy.hand(),
            contact_policy_digest: contact_policy.policy_digest(),
            retention_policy_digest: retention_policy.policy_digest(),
            maximum_object_binding_age_s,
            minimum_object_confidence,
            maximum_end_effector_speed_mps,
            minimum_requested_contact_force_n,
            maximum_requested_contact_force_n,
            maximum_resulting_payload_kg,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.policy_digest = digest_acquisition_policy(&value);
        value
            .validate_for(subject, contact_policy, retention_policy)
            .then_some(value)
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        contact_policy: &HumanoidGraspContactPolicy,
        retention_policy: &HumanoidGraspRetentionPolicy,
    ) -> bool {
        self.schema_version == HUMANOID_GRASP_ACQUISITION_POLICY_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && contact_policy.validate_for(subject)
            && retention_policy.validate_for(subject, contact_policy)
            && self.hand == contact_policy.hand()
            && self.contact_policy_digest == contact_policy.policy_digest()
            && self.retention_policy_digest == retention_policy.policy_digest()
            && self.maximum_object_binding_age_s.is_finite()
            && self.maximum_object_binding_age_s > 0.0
            && self.minimum_object_confidence.is_finite()
            && (0.0..=1.0).contains(&self.minimum_object_confidence)
            && self.maximum_end_effector_speed_mps.is_finite()
            && self.maximum_end_effector_speed_mps > 0.0
            && self.minimum_requested_contact_force_n.is_finite()
            && self.minimum_requested_contact_force_n >= 0.0
            && self.maximum_requested_contact_force_n.is_finite()
            && self.maximum_requested_contact_force_n >= self.minimum_requested_contact_force_n
            && self.maximum_resulting_payload_kg.is_finite()
            && self.maximum_resulting_payload_kg >= 0.0
            && !self.policy_digest.is_zero()
            && self.policy_digest == digest_acquisition_policy(self)
    }

    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspAcquisitionFailure {
    InvalidSubject,
    InvalidObjectBinding,
    InvalidPolicy,
    InvalidTime,
    ObjectBindingTimestampInFuture,
    ObjectBindingStale,
    ObjectConfidenceTooLow,
    SpatialRoleMismatch,
    PermitSubjectMismatch,
    NotGraspIntent,
    IntentLineageMismatch,
    IntentObjectiveShapeMismatch,
    MissingRequiredInvariant,
    RequestedSpeedTooHigh,
    RequestedContactForceOutOfPolicy,
    RequestedPayloadTooHigh,
    InvalidDigest,
}

/// Opaque acquisition-planning state. This is intentionally not actuator authority.
#[derive(Debug, PartialEq)]
pub struct HumanoidGraspAcquisitionIntent {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    validation_epoch: u64,
    object_binding_digest: HumanoidEvidenceDigest,
    object_id: String,
    object_state_digest: HumanoidEvidenceDigest,
    spatial_goal_id: String,
    hand: HandSide,
    target_root_m: [f64; 3],
    workspace_utilization_sq: f64,
    contact_policy_digest: HumanoidEvidenceDigest,
    retention_policy_digest: HumanoidEvidenceDigest,
    acquisition_policy_digest: HumanoidEvidenceDigest,
    requested_end_effector_speed_mps: f64,
    requested_contact_force_n: f64,
    resulting_total_payload_kg: f64,
    compiled_at_s: f64,
    valid_until_s: f64,
    intent_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspAcquisitionIntent {
    pub fn object_id(&self) -> &str {
        &self.object_id
    }

    pub const fn object_state_digest(&self) -> HumanoidEvidenceDigest {
        self.object_state_digest
    }

    pub const fn hand(&self) -> HandSide {
        self.hand
    }

    pub fn spatial_goal_id(&self) -> &str {
        &self.spatial_goal_id
    }

    pub const fn target_root_m(&self) -> [f64; 3] {
        self.target_root_m
    }

    pub const fn requested_contact_force_n(&self) -> f64 {
        self.requested_contact_force_n
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.valid_until_s
    }

    pub const fn intent_digest(&self) -> HumanoidEvidenceDigest {
        self.intent_digest
    }
}

/// Refine an exact spatially-bound Grasp permit and its already-compiled
/// whole-body IR into object-bound acquisition planning state.
#[allow(clippy::too_many_arguments)]
pub fn compile_humanoid_grasp_acquisition_intent(
    subject: &HumanoidQualificationSubject,
    bound: &HumanoidSpatiallyBoundSkillPermit<'_>,
    whole_body: &HumanoidWholeBodyMotionIntent,
    object_binding: &HumanoidGraspObjectBindingEvidence,
    contact_policy: &HumanoidGraspContactPolicy,
    retention_policy: &HumanoidGraspRetentionPolicy,
    acquisition_policy: &HumanoidGraspAcquisitionPolicy,
    now_s: f64,
) -> Result<HumanoidGraspAcquisitionIntent, HumanoidGraspAcquisitionFailure> {
    if digest_subject(subject).is_none() {
        return Err(HumanoidGraspAcquisitionFailure::InvalidSubject);
    }
    if !object_binding.validate_for(subject, bound.goal()) {
        return Err(HumanoidGraspAcquisitionFailure::InvalidObjectBinding);
    }
    if !acquisition_policy.validate_for(subject, contact_policy, retention_policy) {
        return Err(HumanoidGraspAcquisitionFailure::InvalidPolicy);
    }
    if !now_s.is_finite() || now_s < 0.0 {
        return Err(HumanoidGraspAcquisitionFailure::InvalidTime);
    }
    if object_binding.received_at_s() > now_s || object_binding.observed_at_s() > now_s {
        return Err(HumanoidGraspAcquisitionFailure::ObjectBindingTimestampInFuture);
    }
    let binding_age_s = now_s - object_binding.observed_at_s();
    if binding_age_s > acquisition_policy.maximum_object_binding_age_s {
        return Err(HumanoidGraspAcquisitionFailure::ObjectBindingStale);
    }
    if object_binding.confidence() < acquisition_policy.minimum_object_confidence {
        return Err(HumanoidGraspAcquisitionFailure::ObjectConfidenceTooLow);
    }
    if bound.role() != HumanoidSkillRequirementRole::Manipulation {
        return Err(HumanoidGraspAcquisitionFailure::SpatialRoleMismatch);
    }

    let permit = bound.semantic();
    if permit.morphology() != subject.morphology
        || permit.actuation_mode() != subject.actuation_mode
        || permit.backend_profile_id() != subject.backend_profile_id
    {
        return Err(HumanoidGraspAcquisitionFailure::PermitSubjectMismatch);
    }
    let manipulation_requirements = permit
        .requirements()
        .iter()
        .filter(|requirement| requirement.role == HumanoidSkillRequirementRole::Manipulation)
        .collect::<Vec<_>>();
    if !matches!(manipulation_requirements.as_slice(), [requirement]
        if requirement.request.subject_fingerprint == subject.fingerprint())
    {
        return Err(HumanoidGraspAcquisitionFailure::PermitSubjectMismatch);
    }

    let (
        requested_end_effector_speed_mps,
        requested_contact_force_n,
        resulting_total_payload_kg,
    ) = match permit.intent() {
        HumanoidSkillIntent::Grasp {
            end_effector_speed_mps,
            object_contact_force_n,
            resulting_total_payload_kg,
        } => (
            end_effector_speed_mps,
            object_contact_force_n,
            resulting_total_payload_kg,
        ),
        _ => return Err(HumanoidGraspAcquisitionFailure::NotGraspIntent),
    };

    if whole_body.validation_epoch != permit.epoch()
        || whole_body.morphology != permit.morphology()
        || whole_body.actuation_mode != permit.actuation_mode()
        || whole_body.backend_profile_id != permit.backend_profile_id()
        || whole_body.source_skill != permit.intent()
        || whole_body.spatial_goal_id.as_deref() != Some(bound.goal().goal_id.as_str())
        || whole_body.requirement_subject_fingerprints
            != permit
                .requirements()
                .iter()
                .map(|requirement| requirement.request.subject_fingerprint)
                .collect::<Vec<_>>()
    {
        return Err(HumanoidGraspAcquisitionFailure::IntentLineageMismatch);
    }

    for required in [
        HumanoidWholeBodyInvariantIR::CapabilityEnvelopeRemainsAdmitted,
        HumanoidWholeBodyInvariantIR::QualificationSubjectRemainsStable,
        HumanoidWholeBodyInvariantIR::ProtectiveBehaviorMayPreemptGoal,
        HumanoidWholeBodyInvariantIR::SpatialGoalRemainsBound,
    ] {
        if !whole_body.invariants.contains(&required) {
            return Err(HumanoidGraspAcquisitionFailure::MissingRequiredInvariant);
        }
    }

    if !exact_grasp_objective_shape(
        whole_body,
        object_binding.hand,
        bound.target_root_m(),
        requested_end_effector_speed_mps,
        requested_contact_force_n,
        resulting_total_payload_kg,
    ) {
        return Err(HumanoidGraspAcquisitionFailure::IntentObjectiveShapeMismatch);
    }
    if requested_end_effector_speed_mps > acquisition_policy.maximum_end_effector_speed_mps {
        return Err(HumanoidGraspAcquisitionFailure::RequestedSpeedTooHigh);
    }
    if requested_contact_force_n < acquisition_policy.minimum_requested_contact_force_n
        || requested_contact_force_n > acquisition_policy.maximum_requested_contact_force_n
    {
        return Err(HumanoidGraspAcquisitionFailure::RequestedContactForceOutOfPolicy);
    }
    if resulting_total_payload_kg > acquisition_policy.maximum_resulting_payload_kg {
        return Err(HumanoidGraspAcquisitionFailure::RequestedPayloadTooHigh);
    }

    let valid_until_s = object_binding.observed_at_s() + acquisition_policy.maximum_object_binding_age_s;
    if !valid_until_s.is_finite() || valid_until_s < now_s {
        return Err(HumanoidGraspAcquisitionFailure::ObjectBindingStale);
    }

    let mut value = HumanoidGraspAcquisitionIntent {
        schema_version: HUMANOID_GRASP_ACQUISITION_INTENT_SCHEMA_VERSION,
        subject_digest: digest_subject(subject)
            .ok_or(HumanoidGraspAcquisitionFailure::InvalidSubject)?,
        validation_epoch: permit.epoch(),
        object_binding_digest: object_binding.binding_digest(),
        object_id: object_binding.object_id().to_string(),
        object_state_digest: object_binding.object_state_digest(),
        spatial_goal_id: object_binding.spatial_goal_id().to_string(),
        hand: object_binding.hand(),
        target_root_m: bound.target_root_m(),
        workspace_utilization_sq: bound.workspace_utilization_sq(),
        contact_policy_digest: contact_policy.policy_digest(),
        retention_policy_digest: retention_policy.policy_digest(),
        acquisition_policy_digest: acquisition_policy.policy_digest(),
        requested_end_effector_speed_mps,
        requested_contact_force_n,
        resulting_total_payload_kg,
        compiled_at_s: now_s,
        valid_until_s,
        intent_digest: HumanoidEvidenceDigest::ZERO,
    };
    value.intent_digest = digest_acquisition_intent(&value);
    if value.intent_digest.is_zero() {
        return Err(HumanoidGraspAcquisitionFailure::InvalidDigest);
    }
    Ok(value)
}

fn exact_grasp_objective_shape(
    intent: &HumanoidWholeBodyMotionIntent,
    hand: HandSide,
    target_root_m: [f64; 3],
    requested_speed_mps: f64,
    requested_contact_force_n: f64,
    resulting_total_payload_kg: f64,
) -> bool {
    let mut upright = 0usize;
    let mut end_effector = 0usize;
    let mut object_contact = 0usize;
    let mut payload = 0usize;

    for objective in &intent.objectives {
        match objective {
            HumanoidWholeBodyObjectiveIR::UprightPosture => upright += 1,
            HumanoidWholeBodyObjectiveIR::EndEffectorTarget {
                hand: objective_hand,
                target_root_m: objective_target,
                maximum_speed_mps,
            } => {
                if *objective_hand != hand
                    || !same_vec3_bits(*objective_target, target_root_m)
                    || maximum_speed_mps.to_bits() != requested_speed_mps.to_bits()
                {
                    return false;
                }
                end_effector += 1;
            }
            HumanoidWholeBodyObjectiveIR::ObjectContact {
                hand: objective_hand,
                mode,
                requested_contact_force_n: objective_force,
                resulting_total_payload_kg: objective_payload,
            } => {
                if *objective_hand != hand
                    || *mode != HumanoidContactObjectiveMode::Acquire
                    || objective_force.to_bits() != requested_contact_force_n.to_bits()
                    || objective_payload.to_bits() != resulting_total_payload_kg.to_bits()
                {
                    return false;
                }
                object_contact += 1;
            }
            HumanoidWholeBodyObjectiveIR::Payload { total_payload_kg } => {
                if resulting_total_payload_kg <= 0.0
                    || total_payload_kg.to_bits() != resulting_total_payload_kg.to_bits()
                {
                    return false;
                }
                payload += 1;
            }
            HumanoidWholeBodyObjectiveIR::LocomotionVelocity { .. }
            | HumanoidWholeBodyObjectiveIR::HumanContact { .. } => return false,
        }
    }

    upright == 1
        && end_effector == 1
        && object_contact == 1
        && payload == usize::from(resulting_total_payload_kg > 0.0)
        && intent.objectives.len() == 3 + payload
}

fn digest_object_binding(value: &HumanoidGraspObjectBindingEvidence) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-object-binding.v1");
    h.u32(value.schema_version)
        .digest(value.subject_digest)
        .string(&value.object_id)
        .digest(value.object_state_digest)
        .string(&value.spatial_goal_id)
        .u64(hand_id(value.hand));
    for component in value.target_world_m {
        h.f64(component);
    }
    h.f64(value.observed_at_s)
        .f64(value.received_at_s)
        .f64(value.confidence);
    h.finish()
}

fn digest_acquisition_policy(value: &HumanoidGraspAcquisitionPolicy) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-acquisition-policy.v1");
    h.u32(value.schema_version)
        .digest(value.subject_digest)
        .u64(hand_id(value.hand))
        .digest(value.contact_policy_digest)
        .digest(value.retention_policy_digest)
        .f64(value.maximum_object_binding_age_s)
        .f64(value.minimum_object_confidence)
        .f64(value.maximum_end_effector_speed_mps)
        .f64(value.minimum_requested_contact_force_n)
        .f64(value.maximum_requested_contact_force_n)
        .f64(value.maximum_resulting_payload_kg);
    h.finish()
}

fn digest_acquisition_intent(value: &HumanoidGraspAcquisitionIntent) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-acquisition-intent.v1");
    h.u32(value.schema_version)
        .digest(value.subject_digest)
        .u64(value.validation_epoch)
        .digest(value.object_binding_digest)
        .string(&value.object_id)
        .digest(value.object_state_digest)
        .string(&value.spatial_goal_id)
        .u64(hand_id(value.hand));
    for component in value.target_root_m {
        h.f64(component);
    }
    h.f64(value.workspace_utilization_sq)
        .digest(value.contact_policy_digest)
        .digest(value.retention_policy_digest)
        .digest(value.acquisition_policy_digest)
        .f64(value.requested_end_effector_speed_mps)
        .f64(value.requested_contact_force_n)
        .f64(value.resulting_total_payload_kg)
        .f64(value.compiled_at_s)
        .f64(value.valid_until_s);
    h.finish()
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() || subject.task != HumanoidTask::Grasp {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-acquisition-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn same_vec3_bits(left: [f64; 3], right: [f64; 3]) -> bool {
    left.into_iter()
        .zip(right)
        .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

fn hand_id(hand: HandSide) -> u64 {
    match hand {
        HandSide::Right => 1,
        HandSide::Left => 2,
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
    use crate::actuator_controllability::{
        ContactWrenchAxis, ContactWrenchAxisMargin, ContactWrenchMarginAssessment,
        HumanoidActuationControllabilityAssessment,
    };
    use crate::capability_envelope::{
        HumanInteractionEvidence, HumanoidCapabilityRestriction, HumanoidNominalCapabilityProfile,
        derive_humanoid_capability_envelope,
    };
    use crate::contact_site::HumanoidContactSite;
    use crate::execution::HumanoidAuthorityEnvelope;
    use crate::full_dynamics::DynamicsComponentSource;
    use crate::grasp_contact_evidence::{
        HumanoidGraspContactPolicy, HumanoidManipulationContactSource,
    };
    use crate::grasp_retention_evidence::HumanoidGraspRetentionPolicy;
    use crate::morphology::HumanoidMorphology;
    use crate::skill_actuation_guard::{
        HumanoidSkillActuationEvidenceEntry, HumanoidSkillActuationPolicyEntry,
    };
    use crate::skill_permit::{HumanoidPermitSkillExecutive, HumanoidSkillValidationCycle};
    use crate::skill_runtime::{
        HumanoidSkillQualificationSet, compile_humanoid_skill_contract,
    };
    use crate::spatial_goal::{
        HumanoidReachWorkspaceProfile, HumanoidSpatialGoalAdmissionConfig,
        bind_humanoid_spatial_goal,
    };
    use crate::typed_actuation_capability::{
        HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION, TypedHumanoidActuationCapabilityPolicy,
        TypedHumanoidContactActuationRequirement,
    };
    use crate::types::HumanoidState;
    use crate::whole_body_intent::compile_spatial_whole_body_intent;

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Grasp,
            ActuationMode::NormalizedTorque,
            "grasp-acquisition-test-backend",
        )
    }

    fn profile() -> HumanoidNominalCapabilityProfile {
        HumanoidNominalCapabilityProfile::new(
            &subject(), 0.0, 0.0, 0.0, 1.0, 0.2, 5.0, 40.0, 10.0, 0.8, true,
        )
    }

    fn envelope() -> crate::capability_envelope::HumanoidCapabilityEnvelope {
        derive_humanoid_capability_envelope(
            &subject(),
            &profile(),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence::no_human_present(),
        )
    }

    fn actuation_policy() -> HumanoidSkillActuationPolicyEntry {
        let subject = subject();
        HumanoidSkillActuationPolicyEntry {
            role: HumanoidSkillRequirementRole::Manipulation,
            subject: subject.clone(),
            profile: profile(),
            policy: TypedHumanoidActuationCapabilityPolicy {
                schema_version: HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION,
                subject_fingerprint: subject.fingerprint(),
                authority_profile_id: "grasp-acq-joints-v1".into(),
                calibration_fingerprint: 5,
                dynamics_model_id: "grasp-acq-dynamics-v1".into(),
                requirements: vec![TypedHumanoidContactActuationRequirement {
                    site: HumanoidContactSite::RightHand,
                    axis: ContactWrenchAxis::ForceZ,
                    nominal_min_retained_fraction: 0.8,
                    degraded_min_retained_fraction: 0.5,
                    nominal_min_retained_wrench: 20.0,
                    degraded_min_retained_wrench: 10.0,
                }],
                degraded_restriction: HumanoidCapabilityRestriction {
                    max_horizontal_speed_mps: 0.0,
                    max_turn_rate_rad_s: 0.0,
                    max_end_effector_speed_mps: 0.3,
                    max_payload_kg: 3.0,
                    max_object_contact_force_n: 20.0,
                    max_human_contact_force_n: 5.0,
                },
            },
        }
    }

    fn actuation_evidence() -> HumanoidSkillActuationEvidenceEntry {
        HumanoidSkillActuationEvidenceEntry {
            subject_fingerprint: subject().fingerprint(),
            assessment: HumanoidActuationControllabilityAssessment {
                morphology: HumanoidMorphology::Dexterous53,
                authority_sequence: 1,
                authority_age_s: 0.01,
                authority_profile_id: "grasp-acq-joints-v1".into(),
                calibration_fingerprint: 5,
                dynamics_model_id: "grasp-acq-dynamics-v1".into(),
                sites: vec![ContactWrenchMarginAssessment {
                    site_id: HumanoidContactSite::RightHand.canonical_id().into(),
                    contact_confidence: 1.0,
                    jacobian_source: DynamicsComponentSource::SimulatorSolver,
                    actuator_limit_source: DynamicsComponentSource::SystemIdentification,
                    axes: vec![ContactWrenchAxisMargin {
                        axis: ContactWrenchAxis::ForceZ,
                        actuated_support_present: true,
                        nominal_limit: Some(30.0),
                        retained_limit: Some(27.0),
                        retained_fraction: 0.9,
                        limiting_joint: Some(15),
                    }],
                    minimum_retained_fraction: 0.9,
                }],
            },
        }
    }

    fn cycle<'a>(
        executive: &'a mut HumanoidPermitSkillExecutive,
    ) -> HumanoidSkillValidationCycle<'a> {
        let qualifications = HumanoidSkillQualificationSet::new(vec![subject()]);
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Grasp {
                end_effector_speed_mps: 0.1,
                object_contact_force_n: 10.0,
                resulting_total_payload_kg: 0.0,
            },
            &qualifications,
        )
        .unwrap();
        executive
            .start_and_issue(
                contract,
                &[envelope()],
                &[actuation_policy()],
                &[actuation_evidence()],
                crate::skill_executive::HumanoidSkillRuntimeEvidence {
                    goal_authority_valid: true,
                    load_retained: false,
                    human_proximity_valid: true,
                    human_contact_consent: false,
                    protective_preempted: false,
                    objective_satisfied: false,
                },
            )
            .unwrap()
    }

    fn state() -> HumanoidState {
        let mut state = HumanoidState::standing_for(HumanoidMorphology::Dexterous53);
        state.root_position = [1.0, 2.0, 1.0];
        state.root_quaternion = [1.0, 0.0, 0.0, 0.0];
        state.timestamp = 10.0;
        state
    }

    fn workspace() -> HumanoidReachWorkspaceProfile {
        HumanoidReachWorkspaceProfile::from_subject(
            &subject(),
            HandSide::Right,
            [0.3, -0.2, 0.2],
            [0.4, 0.4, 0.4],
        )
    }

    fn goal(target_world_m: [f64; 3]) -> HumanoidSpatialGoalEvidence {
        HumanoidSpatialGoalEvidence {
            goal_id: "goal-grasp-object-a".into(),
            kind: HumanoidSpatialTargetKind::Object,
            hand: HandSide::Right,
            target_world_m,
            observed_at_s: 10.0,
            received_at_s: 10.0,
            confidence: 0.95,
        }
    }

    fn contact_policy() -> HumanoidGraspContactPolicy {
        HumanoidGraspContactPolicy::new(
            &subject(),
            HandSide::Right,
            0.05,
            0.8,
            HumanoidManipulationContactSource::SolverWrench.quality_rank(),
            true,
            2.0,
            50.0,
            15.0,
            2.0,
            0.02,
            0.01,
            0.02,
        )
        .unwrap()
    }

    fn retention_policy(contact: &HumanoidGraspContactPolicy) -> HumanoidGraspRetentionPolicy {
        HumanoidGraspRetentionPolicy::new(&subject(), contact, 3, 0.04, 5, 0.08, 0.03, 64, 2.0)
            .unwrap()
    }

    fn acquisition_policy(
        contact: &HumanoidGraspContactPolicy,
        retention: &HumanoidGraspRetentionPolicy,
    ) -> HumanoidGraspAcquisitionPolicy {
        HumanoidGraspAcquisitionPolicy::new(
            &subject(),
            contact,
            retention,
            0.2,
            0.8,
            0.2,
            2.0,
            15.0,
            1.0,
        )
        .unwrap()
    }

    #[test]
    fn exact_object_bound_grasp_compiles() {
        let mut executive = HumanoidPermitSkillExecutive::new();
        let cycle = cycle(&mut executive);
        let goal = goal([1.3, 1.8, 1.2]);
        let bound = bind_humanoid_spatial_goal(
            &cycle,
            HumanoidSkillRequirementRole::Manipulation,
            &subject(),
            &workspace(),
            &goal,
            &actuation_policy(),
            &actuation_evidence(),
            &state(),
            10.05,
            HumanoidSpatialGoalAdmissionConfig {
                maximum_goal_age_s: 0.2,
                maximum_state_age_s: 0.2,
                minimum_goal_confidence: 0.8,
            },
        )
        .unwrap();
        let whole_body = compile_spatial_whole_body_intent(&bound).unwrap();
        let binding = HumanoidGraspObjectBindingEvidence::bind(
            &subject(),
            bound.goal(),
            "object-a",
            HumanoidEvidenceDigest::from_bytes([9; 32]),
        )
        .unwrap();
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let acquisition = acquisition_policy(&contact, &retention);

        let intent = compile_humanoid_grasp_acquisition_intent(
            &subject(),
            &bound,
            &whole_body,
            &binding,
            &contact,
            &retention,
            &acquisition,
            10.05,
        )
        .unwrap();
        assert_eq!(intent.object_id(), "object-a");
        assert_eq!(intent.requested_contact_force_n(), 10.0);
        assert!(!intent.intent_digest().is_zero());
    }

    #[test]
    fn stale_object_binding_is_rejected() {
        let mut executive = HumanoidPermitSkillExecutive::new();
        let cycle = cycle(&mut executive);
        let goal = goal([1.3, 1.8, 1.2]);
        let bound = bind_humanoid_spatial_goal(
            &cycle,
            HumanoidSkillRequirementRole::Manipulation,
            &subject(),
            &workspace(),
            &goal,
            &actuation_policy(),
            &actuation_evidence(),
            &state(),
            10.05,
            HumanoidSpatialGoalAdmissionConfig {
                maximum_goal_age_s: 0.2,
                maximum_state_age_s: 0.2,
                minimum_goal_confidence: 0.8,
            },
        )
        .unwrap();
        let whole_body = compile_spatial_whole_body_intent(&bound).unwrap();
        let binding = HumanoidGraspObjectBindingEvidence::bind(
            &subject(),
            bound.goal(),
            "object-a",
            HumanoidEvidenceDigest::from_bytes([9; 32]),
        )
        .unwrap();
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let acquisition = acquisition_policy(&contact, &retention);
        assert_eq!(
            compile_humanoid_grasp_acquisition_intent(
                &subject(),
                &bound,
                &whole_body,
                &binding,
                &contact,
                &retention,
                &acquisition,
                10.25,
            ),
            Err(HumanoidGraspAcquisitionFailure::ObjectBindingStale)
        );
    }

    #[test]
    fn caller_modified_contact_objective_is_rejected() {
        let mut executive = HumanoidPermitSkillExecutive::new();
        let cycle = cycle(&mut executive);
        let goal = goal([1.3, 1.8, 1.2]);
        let bound = bind_humanoid_spatial_goal(
            &cycle,
            HumanoidSkillRequirementRole::Manipulation,
            &subject(),
            &workspace(),
            &goal,
            &actuation_policy(),
            &actuation_evidence(),
            &state(),
            10.05,
            HumanoidSpatialGoalAdmissionConfig {
                maximum_goal_age_s: 0.2,
                maximum_state_age_s: 0.2,
                minimum_goal_confidence: 0.8,
            },
        )
        .unwrap();
        let mut whole_body = compile_spatial_whole_body_intent(&bound).unwrap();
        for objective in &mut whole_body.objectives {
            if let HumanoidWholeBodyObjectiveIR::ObjectContact {
                requested_contact_force_n,
                ..
            } = objective
            {
                *requested_contact_force_n = 11.0;
            }
        }
        let binding = HumanoidGraspObjectBindingEvidence::bind(
            &subject(),
            bound.goal(),
            "object-a",
            HumanoidEvidenceDigest::from_bytes([9; 32]),
        )
        .unwrap();
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let acquisition = acquisition_policy(&contact, &retention);
        assert_eq!(
            compile_humanoid_grasp_acquisition_intent(
                &subject(),
                &bound,
                &whole_body,
                &binding,
                &contact,
                &retention,
                &acquisition,
                10.05,
            ),
            Err(HumanoidGraspAcquisitionFailure::IntentObjectiveShapeMismatch)
        );
    }

    #[test]
    fn object_state_changes_binding_identity() {
        let goal = goal([1.3, 1.8, 1.2]);
        let a = HumanoidGraspObjectBindingEvidence::bind(
            &subject(),
            &goal,
            "object-a",
            HumanoidEvidenceDigest::from_bytes([1; 32]),
        )
        .unwrap();
        let b = HumanoidGraspObjectBindingEvidence::bind(
            &subject(),
            &goal,
            "object-a",
            HumanoidEvidenceDigest::from_bytes([2; 32]),
        )
        .unwrap();
        assert_ne!(a.binding_digest(), b.binding_digest());
    }
}
