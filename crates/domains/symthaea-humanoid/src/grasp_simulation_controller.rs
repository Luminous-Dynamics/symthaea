// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Proposal-only Grasp controller candidate for Simulation qualification.
//!
//! This module deliberately does not lower `ObjectContact`, produce joint/torque
//! commands, mint authority, or expose a hardware adapter. It consumes an already
//! admitted object-bound Grasp acquisition intent plus canonical contact feedback
//! and emits typed **simulation proposals**. The independent qualification stack
//! remains responsible for judging the resulting measured contact/retention stream.
//!
//! The candidate is intentionally conservative: no-contact -> approach, accepted
//! measured contact -> hold, isolated low normal force -> seek the already-admitted
//! semantic force, and any other in-contact policy violation -> abort. It does not
//! infer friction or compensate unsafe shear/torque/slip by blindly squeezing harder.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::grasp_acquisition_intent::HumanoidGraspAcquisitionIntent;
use crate::grasp_contact_evidence::{
    HumanoidGraspContactAssessment, HumanoidGraspContactFailureKind,
    HumanoidGraspContactObservation, HumanoidGraspContactPolicy,
    assess_humanoid_grasp_contact,
};
use crate::grasp_controller_qualification::HumanoidGraspControllerCandidate;
use crate::morphology::HandSide;
use crate::qualification::HumanoidQualificationSubject;
use crate::types::{ActuationMode, HumanoidTask};

pub const HUMANOID_GRASP_SIMULATION_CONTROLLER_POLICY_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_SIMULATION_SESSION_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_SIMULATION_PROPOSAL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspSimulationControllerPolicy {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    controller_id: String,
    controller_artifact_digest: HumanoidEvidenceDigest,
    feedback_contact_policy_digest: HumanoidEvidenceDigest,
    approach_speed_fraction: f64,
    target_contact_force_fraction: f64,
    maximum_steps: usize,
    maximum_step_gap_s: f64,
    policy_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspSimulationControllerPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        contact_policy: &HumanoidGraspContactPolicy,
        controller_id: impl Into<String>,
        controller_artifact_digest: HumanoidEvidenceDigest,
        approach_speed_fraction: f64,
        target_contact_force_fraction: f64,
        maximum_steps: usize,
        maximum_step_gap_s: f64,
    ) -> Option<Self> {
        if !subject.validate()
            || subject.task != HumanoidTask::Grasp
            || !contact_policy.validate_for(subject)
            || controller_artifact_digest.is_zero()
            || !approach_speed_fraction.is_finite()
            || !(0.0..=1.0).contains(&approach_speed_fraction)
            || approach_speed_fraction == 0.0
            || !target_contact_force_fraction.is_finite()
            || !(0.0..=1.0).contains(&target_contact_force_fraction)
            || target_contact_force_fraction == 0.0
            || maximum_steps == 0
            || maximum_steps > 1_000_000
            || !maximum_step_gap_s.is_finite()
            || maximum_step_gap_s <= 0.0
        {
            return None;
        }

        let mut value = Self {
            schema_version: HUMANOID_GRASP_SIMULATION_CONTROLLER_POLICY_SCHEMA_VERSION,
            subject_digest: digest_subject(subject)?,
            controller_id: controller_id.into(),
            controller_artifact_digest,
            feedback_contact_policy_digest: contact_policy.policy_digest(),
            approach_speed_fraction,
            target_contact_force_fraction,
            maximum_steps,
            maximum_step_gap_s,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        if !valid_id(&value.controller_id) {
            return None;
        }
        value.policy_digest = digest_controller_policy(&value);
        value.validate_for(subject, contact_policy).then_some(value)
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        contact_policy: &HumanoidGraspContactPolicy,
    ) -> bool {
        self.schema_version == HUMANOID_GRASP_SIMULATION_CONTROLLER_POLICY_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && valid_id(&self.controller_id)
            && !self.controller_artifact_digest.is_zero()
            && contact_policy.validate_for(subject)
            && self.feedback_contact_policy_digest == contact_policy.policy_digest()
            && self.approach_speed_fraction.is_finite()
            && self.approach_speed_fraction > 0.0
            && self.approach_speed_fraction <= 1.0
            && self.target_contact_force_fraction.is_finite()
            && self.target_contact_force_fraction > 0.0
            && self.target_contact_force_fraction <= 1.0
            && self.maximum_steps > 0
            && self.maximum_steps <= 1_000_000
            && self.maximum_step_gap_s.is_finite()
            && self.maximum_step_gap_s > 0.0
            && !self.policy_digest.is_zero()
            && self.policy_digest == digest_controller_policy(self)
    }

    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }
}

/// Opaque pair that guarantees the generic qualification candidate was created
/// with this exact controller policy as its configuration identity.
pub struct HumanoidGraspSimulationControllerCandidate {
    policy: HumanoidGraspSimulationControllerPolicy,
    candidate: HumanoidGraspControllerCandidate,
}

impl std::fmt::Debug for HumanoidGraspSimulationControllerCandidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidGraspSimulationControllerCandidate")
            .field("controller_id", &self.policy.controller_id)
            .field("policy_digest", &self.policy.policy_digest)
            .field("candidate_digest", &self.candidate.candidate_digest())
            .finish()
    }
}

impl HumanoidGraspSimulationControllerCandidate {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        contact_policy: &HumanoidGraspContactPolicy,
        controller_id: impl Into<String>,
        controller_artifact_digest: HumanoidEvidenceDigest,
        approach_speed_fraction: f64,
        target_contact_force_fraction: f64,
        maximum_steps: usize,
        maximum_step_gap_s: f64,
    ) -> Option<Self> {
        let policy = HumanoidGraspSimulationControllerPolicy::new(
            subject,
            contact_policy,
            controller_id,
            controller_artifact_digest,
            approach_speed_fraction,
            target_contact_force_fraction,
            maximum_steps,
            maximum_step_gap_s,
        )?;
        let candidate = HumanoidGraspControllerCandidate::new(
            policy.controller_id.clone(),
            policy.controller_artifact_digest,
            policy.policy_digest,
        )?;
        Some(Self { policy, candidate })
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        contact_policy: &HumanoidGraspContactPolicy,
    ) -> bool {
        self.policy.validate_for(subject, contact_policy) && self.candidate.validate()
    }

    pub fn qualification_candidate(&self) -> &HumanoidGraspControllerCandidate {
        &self.candidate
    }

    pub fn policy(&self) -> &HumanoidGraspSimulationControllerPolicy {
        &self.policy
    }

    pub const fn candidate_digest(&self) -> HumanoidEvidenceDigest {
        self.candidate.candidate_digest()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspSimulationAbortReason {
    UnsafeMeasuredContact,
}

/// A semantic simulation proposal. These are not actuator units or commands.
#[derive(Debug, Clone, PartialEq)]
pub enum HumanoidGraspSimulationProposalKind {
    /// Continue moving toward the exact acquisition target at a fraction of the
    /// already-admitted end-effector speed.
    Approach {
        target_root_m: [f64; 3],
        admitted_speed_fraction: f64,
    },
    /// Seek a fraction of the contact force already admitted by the semantic Grasp
    /// request. This is not a torque command and has no hardware lowering here.
    SeekContactForce {
        admitted_force_fraction: f64,
    },
    /// Maintain the admitted force target while independent evidence remains valid.
    HoldContact {
        admitted_force_fraction: f64,
    },
    /// Stop goal-directed acquisition in the simulation candidate.
    Abort {
        reason: HumanoidGraspSimulationAbortReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspSimulationControllerFailure {
    InvalidSubject,
    InvalidCandidate,
    InvalidTime,
    AcquisitionExpired,
    HandMismatch,
    InvalidFeedbackShape,
    InvalidObservation,
    InvalidAssessment,
    ObjectMismatch,
    FeedbackFromFuture,
    StepGapTooLarge,
    StepBudgetExhausted,
    SessionTerminal,
    InvalidDigest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspSimulationControllerProposal {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    session_digest: HumanoidEvidenceDigest,
    sequence: u64,
    previous_proposal_digest: Option<HumanoidEvidenceDigest>,
    acquisition_intent_digest: HumanoidEvidenceDigest,
    candidate_digest: HumanoidEvidenceDigest,
    object_id: String,
    hand: HandSide,
    feedback_observation_digest: Option<HumanoidEvidenceDigest>,
    feedback_assessment_digest: Option<HumanoidEvidenceDigest>,
    feedback_object_state_digest: Option<HumanoidEvidenceDigest>,
    generated_at_s: f64,
    kind: HumanoidGraspSimulationProposalKind,
    proposal_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspSimulationControllerProposal {
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub const fn proposal_digest(&self) -> HumanoidEvidenceDigest {
        self.proposal_digest
    }

    pub fn kind(&self) -> &HumanoidGraspSimulationProposalKind {
        &self.kind
    }

    fn validate(&self) -> bool {
        self.schema_version == HUMANOID_GRASP_SIMULATION_PROPOSAL_SCHEMA_VERSION
            && !self.subject_digest.is_zero()
            && !self.session_digest.is_zero()
            && self.sequence > 0
            && self
                .previous_proposal_digest
                .map(|digest| !digest.is_zero())
                .unwrap_or(true)
            && !self.acquisition_intent_digest.is_zero()
            && !self.candidate_digest.is_zero()
            && valid_id(&self.object_id)
            && self.feedback_observation_digest.is_some()
                == self.feedback_assessment_digest.is_some()
            && self.feedback_observation_digest.is_some()
                == self.feedback_object_state_digest.is_some()
            && self.generated_at_s.is_finite()
            && self.generated_at_s >= 0.0
            && valid_proposal_kind(&self.kind)
            && !self.proposal_digest.is_zero()
            && self.proposal_digest == digest_proposal(self)
    }
}

/// Mutable hash-chained session state. It is simulation/planning state only.
pub struct HumanoidGraspSimulationControllerSession {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    acquisition_intent_digest: HumanoidEvidenceDigest,
    candidate_digest: HumanoidEvidenceDigest,
    controller_policy_digest: HumanoidEvidenceDigest,
    object_id: String,
    hand: HandSide,
    target_root_m: [f64; 3],
    acquisition_valid_until_s: f64,
    started_at_s: f64,
    last_step_at_s: f64,
    steps: usize,
    last_proposal_digest: Option<HumanoidEvidenceDigest>,
    terminal: bool,
    session_digest: HumanoidEvidenceDigest,
}

impl std::fmt::Debug for HumanoidGraspSimulationControllerSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidGraspSimulationControllerSession")
            .field("candidate_digest", &self.candidate_digest)
            .field("object_id", &self.object_id)
            .field("hand", &self.hand)
            .field("steps", &self.steps)
            .field("terminal", &self.terminal)
            .field("session_digest", &self.session_digest)
            .finish()
    }
}

impl HumanoidGraspSimulationControllerSession {
    pub fn new(
        subject: &HumanoidQualificationSubject,
        acquisition: &HumanoidGraspAcquisitionIntent,
        candidate: &HumanoidGraspSimulationControllerCandidate,
        contact_policy: &HumanoidGraspContactPolicy,
        now_s: f64,
    ) -> Result<Self, HumanoidGraspSimulationControllerFailure> {
        let subject_digest = digest_subject(subject)
            .ok_or(HumanoidGraspSimulationControllerFailure::InvalidSubject)?;
        if !candidate.validate_for(subject, contact_policy) {
            return Err(HumanoidGraspSimulationControllerFailure::InvalidCandidate);
        }
        if !now_s.is_finite() || now_s < 0.0 {
            return Err(HumanoidGraspSimulationControllerFailure::InvalidTime);
        }
        if acquisition.intent_digest().is_zero() || acquisition.valid_until_s() < now_s {
            return Err(HumanoidGraspSimulationControllerFailure::AcquisitionExpired);
        }
        if acquisition.hand() != contact_policy.hand() {
            return Err(HumanoidGraspSimulationControllerFailure::HandMismatch);
        }

        let mut value = Self {
            schema_version: HUMANOID_GRASP_SIMULATION_SESSION_SCHEMA_VERSION,
            subject_digest,
            acquisition_intent_digest: acquisition.intent_digest(),
            candidate_digest: candidate.candidate_digest(),
            controller_policy_digest: candidate.policy.policy_digest,
            object_id: acquisition.object_id().to_string(),
            hand: acquisition.hand(),
            target_root_m: acquisition.target_root_m(),
            acquisition_valid_until_s: acquisition.valid_until_s(),
            started_at_s: now_s,
            last_step_at_s: now_s,
            steps: 0,
            last_proposal_digest: None,
            terminal: false,
            session_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.session_digest = digest_session(&value);
        if value.session_digest.is_zero() {
            return Err(HumanoidGraspSimulationControllerFailure::InvalidDigest);
        }
        Ok(value)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        subject: &HumanoidQualificationSubject,
        acquisition: &HumanoidGraspAcquisitionIntent,
        candidate: &HumanoidGraspSimulationControllerCandidate,
        contact_policy: &HumanoidGraspContactPolicy,
        observation: Option<&HumanoidGraspContactObservation>,
        assessment: Option<&HumanoidGraspContactAssessment>,
        now_s: f64,
    ) -> Result<HumanoidGraspSimulationControllerProposal, HumanoidGraspSimulationControllerFailure> {
        if self.terminal {
            return Err(HumanoidGraspSimulationControllerFailure::SessionTerminal);
        }
        if digest_subject(subject) != Some(self.subject_digest)
            || !candidate.validate_for(subject, contact_policy)
            || candidate.candidate_digest() != self.candidate_digest
            || candidate.policy.policy_digest != self.controller_policy_digest
            || acquisition.intent_digest() != self.acquisition_intent_digest
            || acquisition.object_id() != self.object_id
            || acquisition.hand() != self.hand
            || acquisition.target_root_m().map(f64::to_bits) != self.target_root_m.map(f64::to_bits)
        {
            return Err(HumanoidGraspSimulationControllerFailure::InvalidCandidate);
        }
        if !now_s.is_finite() || now_s < self.last_step_at_s {
            return Err(HumanoidGraspSimulationControllerFailure::InvalidTime);
        }
        if now_s > self.acquisition_valid_until_s || now_s > acquisition.valid_until_s() {
            return Err(HumanoidGraspSimulationControllerFailure::AcquisitionExpired);
        }
        if self.steps >= candidate.policy.maximum_steps {
            return Err(HumanoidGraspSimulationControllerFailure::StepBudgetExhausted);
        }
        if self.steps > 0 && now_s - self.last_step_at_s > candidate.policy.maximum_step_gap_s {
            return Err(HumanoidGraspSimulationControllerFailure::StepGapTooLarge);
        }
        if observation.is_some() != assessment.is_some() {
            return Err(HumanoidGraspSimulationControllerFailure::InvalidFeedbackShape);
        }

        let (kind, observation_digest, assessment_digest, object_state_digest) =
            match (observation, assessment) {
                (None, None) => (
                    HumanoidGraspSimulationProposalKind::Approach {
                        target_root_m: self.target_root_m,
                        admitted_speed_fraction: candidate.policy.approach_speed_fraction,
                    },
                    None,
                    None,
                    None,
                ),
                (Some(observation), Some(assessment)) => {
                    if !observation.validate_for(subject) {
                        return Err(HumanoidGraspSimulationControllerFailure::InvalidObservation);
                    }
                    if !assessment.validate(subject, observation, contact_policy) {
                        return Err(HumanoidGraspSimulationControllerFailure::InvalidAssessment);
                    }
                    if observation.object_id() != self.object_id || assessment.object_id() != self.object_id {
                        return Err(HumanoidGraspSimulationControllerFailure::ObjectMismatch);
                    }
                    if observation.hand() != self.hand || assessment.hand() != self.hand {
                        return Err(HumanoidGraspSimulationControllerFailure::HandMismatch);
                    }
                    if observation.timestamp_s() > now_s {
                        return Err(HumanoidGraspSimulationControllerFailure::FeedbackFromFuture);
                    }

                    // Reassess at the controller decision instant so a historically
                    // accepted sample cannot remain accepted after it becomes stale.
                    let fresh = assess_humanoid_grasp_contact(
                        subject,
                        observation,
                        contact_policy,
                        now_s,
                    )
                    .map_err(|_| HumanoidGraspSimulationControllerFailure::InvalidAssessment)?;

                    let kind = if !observation.in_contact() {
                        HumanoidGraspSimulationProposalKind::Approach {
                            target_root_m: self.target_root_m,
                            admitted_speed_fraction: candidate.policy.approach_speed_fraction,
                        }
                    } else if fresh.accepted() {
                        HumanoidGraspSimulationProposalKind::HoldContact {
                            admitted_force_fraction: candidate.policy.target_contact_force_fraction,
                        }
                    } else if isolated_low_normal_force(fresh.failures()) {
                        HumanoidGraspSimulationProposalKind::SeekContactForce {
                            admitted_force_fraction: candidate.policy.target_contact_force_fraction,
                        }
                    } else {
                        self.terminal = true;
                        HumanoidGraspSimulationProposalKind::Abort {
                            reason: HumanoidGraspSimulationAbortReason::UnsafeMeasuredContact,
                        }
                    };
                    (
                        kind,
                        Some(observation.observation_digest()),
                        Some(fresh.assessment_digest()),
                        Some(observation.object_state_digest()),
                    )
                }
                _ => unreachable!("feedback shape checked above"),
            };

        let sequence = (self.steps as u64)
            .checked_add(1)
            .ok_or(HumanoidGraspSimulationControllerFailure::StepBudgetExhausted)?;
        let mut proposal = HumanoidGraspSimulationControllerProposal {
            schema_version: HUMANOID_GRASP_SIMULATION_PROPOSAL_SCHEMA_VERSION,
            subject_digest: self.subject_digest,
            session_digest: self.session_digest,
            sequence,
            previous_proposal_digest: self.last_proposal_digest,
            acquisition_intent_digest: self.acquisition_intent_digest,
            candidate_digest: self.candidate_digest,
            object_id: self.object_id.clone(),
            hand: self.hand,
            feedback_observation_digest: observation_digest,
            feedback_assessment_digest: assessment_digest,
            feedback_object_state_digest: object_state_digest,
            generated_at_s: now_s,
            kind,
            proposal_digest: HumanoidEvidenceDigest::ZERO,
        };
        proposal.proposal_digest = digest_proposal(&proposal);
        if !proposal.validate() {
            return Err(HumanoidGraspSimulationControllerFailure::InvalidDigest);
        }

        self.steps += 1;
        self.last_step_at_s = now_s;
        self.last_proposal_digest = Some(proposal.proposal_digest);
        Ok(proposal)
    }

    pub const fn session_digest(&self) -> HumanoidEvidenceDigest {
        self.session_digest
    }

    pub const fn steps(&self) -> usize {
        self.steps
    }
}

fn isolated_low_normal_force(failures: &[HumanoidGraspContactFailureKind]) -> bool {
    matches!(failures, [HumanoidGraspContactFailureKind::NormalForceTooLow])
}

fn valid_proposal_kind(kind: &HumanoidGraspSimulationProposalKind) -> bool {
    match kind {
        HumanoidGraspSimulationProposalKind::Approach {
            target_root_m,
            admitted_speed_fraction,
        } => {
            target_root_m.into_iter().all(f64::is_finite)
                && admitted_speed_fraction.is_finite()
                && *admitted_speed_fraction > 0.0
                && *admitted_speed_fraction <= 1.0
        }
        HumanoidGraspSimulationProposalKind::SeekContactForce {
            admitted_force_fraction,
        }
        | HumanoidGraspSimulationProposalKind::HoldContact {
            admitted_force_fraction,
        } => {
            admitted_force_fraction.is_finite()
                && *admitted_force_fraction > 0.0
                && *admitted_force_fraction <= 1.0
        }
        HumanoidGraspSimulationProposalKind::Abort { .. } => true,
    }
}

fn digest_controller_policy(value: &HumanoidGraspSimulationControllerPolicy) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-simulation-controller-policy.v1");
    h.u32(value.schema_version)
        .digest(value.subject_digest)
        .string(&value.controller_id)
        .digest(value.controller_artifact_digest)
        .digest(value.feedback_contact_policy_digest)
        .f64(value.approach_speed_fraction)
        .f64(value.target_contact_force_fraction)
        .usize(value.maximum_steps)
        .f64(value.maximum_step_gap_s);
    h.finish()
}

fn digest_session(value: &HumanoidGraspSimulationControllerSession) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-simulation-controller-session.v1");
    h.u32(value.schema_version)
        .digest(value.subject_digest)
        .digest(value.acquisition_intent_digest)
        .digest(value.candidate_digest)
        .digest(value.controller_policy_digest)
        .string(&value.object_id)
        .u64(hand_id(value.hand));
    for component in value.target_root_m {
        h.f64(component);
    }
    h.f64(value.acquisition_valid_until_s)
        .f64(value.started_at_s);
    h.finish()
}

fn digest_proposal(value: &HumanoidGraspSimulationControllerProposal) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-simulation-controller-proposal.v1");
    h.u32(value.schema_version)
        .digest(value.subject_digest)
        .digest(value.session_digest)
        .u64(value.sequence)
        .bool(value.previous_proposal_digest.is_some());
    if let Some(digest) = value.previous_proposal_digest {
        h.digest(digest);
    }
    h.digest(value.acquisition_intent_digest)
        .digest(value.candidate_digest)
        .string(&value.object_id)
        .u64(hand_id(value.hand))
        .bool(value.feedback_observation_digest.is_some());
    if let Some(digest) = value.feedback_observation_digest {
        h.digest(digest);
    }
    if let Some(digest) = value.feedback_assessment_digest {
        h.digest(digest);
    }
    if let Some(digest) = value.feedback_object_state_digest {
        h.digest(digest);
    }
    h.f64(value.generated_at_s);
    hash_proposal_kind(&mut h, &value.kind);
    h.finish()
}

fn hash_proposal_kind(h: &mut HumanoidEvidenceHasher, kind: &HumanoidGraspSimulationProposalKind) {
    match kind {
        HumanoidGraspSimulationProposalKind::Approach {
            target_root_m,
            admitted_speed_fraction,
        } => {
            h.u64(1);
            for component in *target_root_m {
                h.f64(component);
            }
            h.f64(*admitted_speed_fraction);
        }
        HumanoidGraspSimulationProposalKind::SeekContactForce {
            admitted_force_fraction,
        } => {
            h.u64(2).f64(*admitted_force_fraction);
        }
        HumanoidGraspSimulationProposalKind::HoldContact {
            admitted_force_fraction,
        } => {
            h.u64(3).f64(*admitted_force_fraction);
        }
        HumanoidGraspSimulationProposalKind::Abort { reason } => {
            h.u64(4).u64(match reason {
                HumanoidGraspSimulationAbortReason::UnsafeMeasuredContact => 1,
            });
        }
    }
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() || subject.task != HumanoidTask::Grasp {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-simulation-controller-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
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
    use crate::contact_site::HumanoidContactSite;
    use crate::grasp_contact_evidence::{
        HumanoidManipulationContactSource, assess_humanoid_grasp_contact,
    };
    use crate::morphology::HumanoidMorphology;

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Grasp,
            ActuationMode::NormalizedTorque,
            "grasp-simulation-controller-test-backend",
        )
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

    fn observation(force_n: f64) -> HumanoidGraspContactObservation {
        HumanoidGraspContactObservation::new(
            &subject(),
            "object-a",
            HumanoidEvidenceDigest::from_bytes([7; 32]),
            HandSide::Right,
            HumanoidContactSite::RightHand,
            true,
            [0.2, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-force_n, 0.1, 0.0],
            [0.0; 3],
            [0.0; 3],
            0.95,
            HumanoidManipulationContactSource::SolverWrench,
            1.0,
        )
        .unwrap()
    }

    #[test]
    fn controller_candidate_configuration_is_policy_identity() {
        let contact = contact_policy();
        let candidate = HumanoidGraspSimulationControllerCandidate::new(
            &subject(),
            &contact,
            "grasp-sim-controller-v1",
            HumanoidEvidenceDigest::from_bytes([9; 32]),
            0.5,
            0.8,
            100,
            0.02,
        )
        .unwrap();
        assert!(candidate.validate_for(&subject(), &contact));
        assert!(!candidate.candidate_digest().is_zero());
    }

    #[test]
    fn isolated_low_normal_force_is_recoverable_but_slip_is_not() {
        assert!(isolated_low_normal_force(&[
            HumanoidGraspContactFailureKind::NormalForceTooLow,
        ]));
        assert!(!isolated_low_normal_force(&[
            HumanoidGraspContactFailureKind::NormalForceTooLow,
            HumanoidGraspContactFailureKind::TangentialSlipTooFast,
        ]));
    }

    #[test]
    fn fresh_assessment_changes_with_force_without_controller_self_certification() {
        let contact = contact_policy();
        let low = observation(1.0);
        let good = observation(10.0);
        let low_assessment = assess_humanoid_grasp_contact(&subject(), &low, &contact, 1.001).unwrap();
        let good_assessment = assess_humanoid_grasp_contact(&subject(), &good, &contact, 1.001).unwrap();
        assert!(!low_assessment.accepted());
        assert!(isolated_low_normal_force(low_assessment.failures()));
        assert!(good_assessment.accepted());
    }
}
