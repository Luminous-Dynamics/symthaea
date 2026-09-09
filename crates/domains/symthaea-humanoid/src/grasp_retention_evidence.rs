// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Temporal, non-actuating retention evidence for humanoid Grasp.
//!
//! One favorable contact frame is not a grasp. This module binds validated
//! contact observations/assessments into immutable samples, then requires an
//! uninterrupted measured-contact run with bounded inter-sample gaps, minimum
//! sample counts and minimum durations before a grasp can be called `Retained`.
//!
//! A continuity break resets the current proof run rather than permanently
//! poisoning the episode: the hand may reacquire and establish a new retained
//! run. A grasp that was retained and later lost is **not** accepted at episode
//! end unless a later run re-establishes retention.
//!
//! Object state digests are allowed to evolve because a held object can move.
//! Every state digest is committed into the episode lineage. This module does not
//! yet claim to validate object-pose dynamics; that requires a future typed object
//! track/trajectory model.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::grasp_contact_evidence::{
    HumanoidGraspContactAssessment, HumanoidGraspContactObservation, HumanoidGraspContactPolicy,
    HumanoidManipulationContactSource,
};
use crate::morphology::HandSide;
use crate::qualification::HumanoidQualificationSubject;
use crate::types::{ActuationMode, HumanoidTask};

pub const HUMANOID_GRASP_RETENTION_SAMPLE_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_RETENTION_POLICY_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_RETENTION_EPISODE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspRetentionSampleFailure {
    InvalidSubject,
    InvalidContactEvidence,
    UnmeasuredAcceptedContact,
    InvalidDigest,
}

/// Immutable bridge from one validated contact assessment into temporal evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspRetentionSample {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    object_id: String,
    object_state_digest: HumanoidEvidenceDigest,
    hand: HandSide,
    contact_policy_digest: HumanoidEvidenceDigest,
    observation_digest: HumanoidEvidenceDigest,
    assessment_digest: HumanoidEvidenceDigest,
    source: HumanoidManipulationContactSource,
    sampled_at_s: f64,
    accepted_contact: bool,
    sample_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspRetentionSample {
    pub fn bind(
        subject: &HumanoidQualificationSubject,
        observation: &HumanoidGraspContactObservation,
        assessment: &HumanoidGraspContactAssessment,
        contact_policy: &HumanoidGraspContactPolicy,
    ) -> Result<Self, HumanoidGraspRetentionSampleFailure> {
        let subject_digest = digest_subject(subject)
            .ok_or(HumanoidGraspRetentionSampleFailure::InvalidSubject)?;
        if !observation.validate_for(subject)
            || !contact_policy.validate_for(subject)
            || !assessment.validate(subject, observation, contact_policy)
        {
            return Err(HumanoidGraspRetentionSampleFailure::InvalidContactEvidence);
        }
        if assessment.accepted() && !observation.source().is_measured() {
            return Err(HumanoidGraspRetentionSampleFailure::UnmeasuredAcceptedContact);
        }

        let mut value = Self {
            schema_version: HUMANOID_GRASP_RETENTION_SAMPLE_SCHEMA_VERSION,
            subject_digest,
            object_id: observation.object_id().to_string(),
            object_state_digest: observation.object_state_digest(),
            hand: observation.hand(),
            contact_policy_digest: contact_policy.policy_digest(),
            observation_digest: observation.observation_digest(),
            assessment_digest: assessment.assessment_digest(),
            source: observation.source(),
            sampled_at_s: observation.timestamp_s(),
            accepted_contact: assessment.accepted(),
            sample_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.sample_digest = digest_sample(&value);
        if !value.validate_for(subject, contact_policy) {
            return Err(HumanoidGraspRetentionSampleFailure::InvalidDigest);
        }
        Ok(value)
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        contact_policy: &HumanoidGraspContactPolicy,
    ) -> bool {
        self.schema_version == HUMANOID_GRASP_RETENTION_SAMPLE_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && contact_policy.validate_for(subject)
            && self.contact_policy_digest == contact_policy.policy_digest()
            && !self.object_id.trim().is_empty()
            && !self.object_state_digest.is_zero()
            && self.hand == contact_policy.hand()
            && !self.observation_digest.is_zero()
            && !self.assessment_digest.is_zero()
            && self.sampled_at_s.is_finite()
            && self.sampled_at_s >= 0.0
            && (!self.accepted_contact || self.source.is_measured())
            && !self.sample_digest.is_zero()
            && self.sample_digest == digest_sample(self)
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

    pub const fn contact_policy_digest(&self) -> HumanoidEvidenceDigest {
        self.contact_policy_digest
    }

    pub const fn source(&self) -> HumanoidManipulationContactSource {
        self.source
    }

    pub const fn sampled_at_s(&self) -> f64 {
        self.sampled_at_s
    }

    pub const fn accepted_contact(&self) -> bool {
        self.accepted_contact
    }

    pub const fn sample_digest(&self) -> HumanoidEvidenceDigest {
        self.sample_digest
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspRetentionPolicy {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    hand: HandSide,
    contact_policy_digest: HumanoidEvidenceDigest,
    minimum_stabilization_samples: usize,
    minimum_stabilization_duration_s: f64,
    minimum_retention_samples: usize,
    minimum_retention_duration_s: f64,
    maximum_inter_sample_gap_s: f64,
    maximum_episode_samples: usize,
    maximum_episode_duration_s: f64,
    policy_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspRetentionPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        contact_policy: &HumanoidGraspContactPolicy,
        minimum_stabilization_samples: usize,
        minimum_stabilization_duration_s: f64,
        minimum_retention_samples: usize,
        minimum_retention_duration_s: f64,
        maximum_inter_sample_gap_s: f64,
        maximum_episode_samples: usize,
        maximum_episode_duration_s: f64,
    ) -> Option<Self> {
        if !subject.validate()
            || subject.task != HumanoidTask::Grasp
            || !contact_policy.validate_for(subject)
            || minimum_stabilization_samples < 2
            || minimum_retention_samples <= minimum_stabilization_samples
            || !minimum_stabilization_duration_s.is_finite()
            || minimum_stabilization_duration_s <= 0.0
            || !minimum_retention_duration_s.is_finite()
            || minimum_retention_duration_s <= minimum_stabilization_duration_s
            || !maximum_inter_sample_gap_s.is_finite()
            || maximum_inter_sample_gap_s <= 0.0
            || maximum_episode_samples < minimum_retention_samples
            || maximum_episode_samples > 1_000_000
            || !maximum_episode_duration_s.is_finite()
            || maximum_episode_duration_s < minimum_retention_duration_s
        {
            return None;
        }

        let mut value = Self {
            schema_version: HUMANOID_GRASP_RETENTION_POLICY_SCHEMA_VERSION,
            subject_digest: digest_subject(subject)?,
            hand: contact_policy.hand(),
            contact_policy_digest: contact_policy.policy_digest(),
            minimum_stabilization_samples,
            minimum_stabilization_duration_s,
            minimum_retention_samples,
            minimum_retention_duration_s,
            maximum_inter_sample_gap_s,
            maximum_episode_samples,
            maximum_episode_duration_s,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.policy_digest = digest_policy(&value);
        value.validate_for(subject, contact_policy).then_some(value)
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        contact_policy: &HumanoidGraspContactPolicy,
    ) -> bool {
        self.schema_version == HUMANOID_GRASP_RETENTION_POLICY_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && contact_policy.validate_for(subject)
            && self.hand == contact_policy.hand()
            && self.contact_policy_digest == contact_policy.policy_digest()
            && self.minimum_stabilization_samples >= 2
            && self.minimum_retention_samples > self.minimum_stabilization_samples
            && self.minimum_stabilization_duration_s.is_finite()
            && self.minimum_stabilization_duration_s > 0.0
            && self.minimum_retention_duration_s.is_finite()
            && self.minimum_retention_duration_s > self.minimum_stabilization_duration_s
            && self.maximum_inter_sample_gap_s.is_finite()
            && self.maximum_inter_sample_gap_s > 0.0
            && self.maximum_episode_samples >= self.minimum_retention_samples
            && self.maximum_episode_samples <= 1_000_000
            && self.maximum_episode_duration_s.is_finite()
            && self.maximum_episode_duration_s >= self.minimum_retention_duration_s
            && !self.policy_digest.is_zero()
            && self.policy_digest == digest_policy(self)
    }

    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspRetentionPhase {
    Approach,
    ContactAcquired,
    Stabilizing,
    Retained,
    Lost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspRetentionEpisodeFailure {
    InvalidSubject,
    InvalidPolicy,
    EmptyEpisode,
    TooManySamples,
    InvalidSample,
    ObjectSubstitution,
    HandSubstitution,
    ContactPolicySubstitution,
    TimestampNotStrictlyIncreasing,
    EpisodeTooLong,
    InvalidDigest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspRetentionFailureKind {
    NoMeasuredContactAtEnd,
    StabilizationSamplesInsufficient,
    StabilizationDurationInsufficient,
    RetentionSamplesInsufficient,
    RetentionDurationInsufficient,
    RetentionLostAtEnd,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspRetentionEpisode {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    object_id: String,
    hand: HandSide,
    contact_policy_digest: HumanoidEvidenceDigest,
    retention_policy_digest: HumanoidEvidenceDigest,
    sample_digests: Vec<HumanoidEvidenceDigest>,
    object_state_digests: Vec<HumanoidEvidenceDigest>,
    started_at_s: f64,
    ended_at_s: f64,
    final_run_started_at_s: Option<f64>,
    final_continuous_samples: usize,
    final_continuous_duration_s: f64,
    continuity_breaks: usize,
    reacquisitions: usize,
    ever_stabilized: bool,
    ever_retained: bool,
    final_phase: HumanoidGraspRetentionPhase,
    failures: Vec<HumanoidGraspRetentionFailureKind>,
    retained_at_end: bool,
    episode_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspRetentionEpisode {
    pub fn object_id(&self) -> &str {
        &self.object_id
    }

    pub const fn hand(&self) -> HandSide {
        self.hand
    }

    pub const fn final_phase(&self) -> HumanoidGraspRetentionPhase {
        self.final_phase
    }

    pub const fn final_continuous_samples(&self) -> usize {
        self.final_continuous_samples
    }

    pub const fn final_continuous_duration_s(&self) -> f64 {
        self.final_continuous_duration_s
    }

    pub const fn continuity_breaks(&self) -> usize {
        self.continuity_breaks
    }

    pub const fn reacquisitions(&self) -> usize {
        self.reacquisitions
    }

    pub const fn ever_retained(&self) -> bool {
        self.ever_retained
    }

    pub fn failures(&self) -> &[HumanoidGraspRetentionFailureKind] {
        &self.failures
    }

    pub const fn retained_at_end(&self) -> bool {
        self.retained_at_end
    }

    pub const fn episode_digest(&self) -> HumanoidEvidenceDigest {
        self.episode_digest
    }

    pub fn object_state_digests(&self) -> &[HumanoidEvidenceDigest] {
        &self.object_state_digests
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        contact_policy: &HumanoidGraspContactPolicy,
        retention_policy: &HumanoidGraspRetentionPolicy,
    ) -> bool {
        self.schema_version == HUMANOID_GRASP_RETENTION_EPISODE_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && contact_policy.validate_for(subject)
            && retention_policy.validate_for(subject, contact_policy)
            && self.hand == contact_policy.hand()
            && self.contact_policy_digest == contact_policy.policy_digest()
            && self.retention_policy_digest == retention_policy.policy_digest()
            && !self.object_id.trim().is_empty()
            && !self.sample_digests.is_empty()
            && self.sample_digests.len() == self.object_state_digests.len()
            && self.sample_digests.iter().all(|digest| !digest.is_zero())
            && self.object_state_digests.iter().all(|digest| !digest.is_zero())
            && self.started_at_s.is_finite()
            && self.ended_at_s.is_finite()
            && self.ended_at_s >= self.started_at_s
            && self.final_continuous_samples <= self.sample_digests.len()
            && self.final_continuous_duration_s.is_finite()
            && self.final_continuous_duration_s >= 0.0
            && self.retained_at_end == (self.final_phase == HumanoidGraspRetentionPhase::Retained)
            && self.retained_at_end == self.failures.is_empty()
            && !self.episode_digest.is_zero()
            && self.episode_digest == digest_episode(self)
    }
}

pub fn evaluate_humanoid_grasp_retention(
    subject: &HumanoidQualificationSubject,
    samples: &[HumanoidGraspRetentionSample],
    contact_policy: &HumanoidGraspContactPolicy,
    retention_policy: &HumanoidGraspRetentionPolicy,
) -> Result<HumanoidGraspRetentionEpisode, HumanoidGraspRetentionEpisodeFailure> {
    if digest_subject(subject).is_none() {
        return Err(HumanoidGraspRetentionEpisodeFailure::InvalidSubject);
    }
    if !contact_policy.validate_for(subject)
        || !retention_policy.validate_for(subject, contact_policy)
    {
        return Err(HumanoidGraspRetentionEpisodeFailure::InvalidPolicy);
    }
    if samples.is_empty() {
        return Err(HumanoidGraspRetentionEpisodeFailure::EmptyEpisode);
    }
    if samples.len() > retention_policy.maximum_episode_samples {
        return Err(HumanoidGraspRetentionEpisodeFailure::TooManySamples);
    }

    let object_id = samples[0].object_id.clone();
    let hand = samples[0].hand;
    let started_at_s = samples[0].sampled_at_s;
    let mut previous_time_s: Option<f64> = None;
    let mut current_run_start_s: Option<f64> = None;
    let mut current_run_samples = 0usize;
    let mut continuity_breaks = 0usize;
    let mut acquisition_count = 0usize;
    let mut ever_stabilized = false;
    let mut ever_retained = false;
    let mut final_phase = HumanoidGraspRetentionPhase::Approach;
    let mut sample_digests = Vec::with_capacity(samples.len());
    let mut object_state_digests = Vec::with_capacity(samples.len());

    for sample in samples {
        if !sample.validate_for(subject, contact_policy) {
            return Err(HumanoidGraspRetentionEpisodeFailure::InvalidSample);
        }
        if sample.object_id != object_id {
            return Err(HumanoidGraspRetentionEpisodeFailure::ObjectSubstitution);
        }
        if sample.hand != hand || hand != retention_policy.hand {
            return Err(HumanoidGraspRetentionEpisodeFailure::HandSubstitution);
        }
        if sample.contact_policy_digest != retention_policy.contact_policy_digest {
            return Err(HumanoidGraspRetentionEpisodeFailure::ContactPolicySubstitution);
        }
        if let Some(previous) = previous_time_s {
            if sample.sampled_at_s <= previous {
                return Err(HumanoidGraspRetentionEpisodeFailure::TimestampNotStrictlyIncreasing);
            }
        }

        let gap_break = previous_time_s
            .map(|previous| {
                sample.sampled_at_s - previous > retention_policy.maximum_inter_sample_gap_s
            })
            .unwrap_or(false);
        let gap_broke_active_run = gap_break && current_run_start_s.is_some();
        let accepted_measured = sample.accepted_contact && sample.source.is_measured();

        if gap_broke_active_run {
            continuity_breaks += 1;
            current_run_start_s = None;
            current_run_samples = 0;
            final_phase = HumanoidGraspRetentionPhase::Lost;
        }

        if accepted_measured {
            if current_run_start_s.is_none() {
                acquisition_count += 1;
                current_run_start_s = Some(sample.sampled_at_s);
                current_run_samples = 1;
                final_phase = HumanoidGraspRetentionPhase::ContactAcquired;
            } else {
                current_run_samples += 1;
                final_phase = HumanoidGraspRetentionPhase::Stabilizing;
            }

            let run_duration_s = sample.sampled_at_s - current_run_start_s.unwrap();
            let stabilized_now = current_run_samples >= retention_policy.minimum_stabilization_samples
                && run_duration_s >= retention_policy.minimum_stabilization_duration_s;
            if stabilized_now {
                ever_stabilized = true;
                final_phase = HumanoidGraspRetentionPhase::Stabilizing;
            }
            let retained_now = stabilized_now
                && current_run_samples >= retention_policy.minimum_retention_samples
                && run_duration_s >= retention_policy.minimum_retention_duration_s;
            if retained_now {
                ever_retained = true;
                final_phase = HumanoidGraspRetentionPhase::Retained;
            }
        } else {
            // A large gap already records the continuity break for this sample;
            // do not count the same physical discontinuity twice when the sample
            // is also rejected.
            if current_run_start_s.is_some() {
                continuity_breaks += 1;
                final_phase = HumanoidGraspRetentionPhase::Lost;
            } else if acquisition_count > 0 {
                final_phase = HumanoidGraspRetentionPhase::Lost;
            } else {
                final_phase = HumanoidGraspRetentionPhase::Approach;
            }
            current_run_start_s = None;
            current_run_samples = 0;
        }

        sample_digests.push(sample.sample_digest);
        object_state_digests.push(sample.object_state_digest);
        previous_time_s = Some(sample.sampled_at_s);
    }

    let ended_at_s = samples.last().unwrap().sampled_at_s;
    if ended_at_s - started_at_s > retention_policy.maximum_episode_duration_s {
        return Err(HumanoidGraspRetentionEpisodeFailure::EpisodeTooLong);
    }
    let final_continuous_duration_s = current_run_start_s
        .map(|start| ended_at_s - start)
        .unwrap_or(0.0);
    let reacquisitions = acquisition_count.saturating_sub(1);

    let mut failures = Vec::new();
    if current_run_samples == 0 {
        failures.push(HumanoidGraspRetentionFailureKind::NoMeasuredContactAtEnd);
    }
    if current_run_samples < retention_policy.minimum_stabilization_samples {
        failures.push(HumanoidGraspRetentionFailureKind::StabilizationSamplesInsufficient);
    }
    if final_continuous_duration_s < retention_policy.minimum_stabilization_duration_s {
        failures.push(HumanoidGraspRetentionFailureKind::StabilizationDurationInsufficient);
    }
    if current_run_samples < retention_policy.minimum_retention_samples {
        failures.push(HumanoidGraspRetentionFailureKind::RetentionSamplesInsufficient);
    }
    if final_continuous_duration_s < retention_policy.minimum_retention_duration_s {
        failures.push(HumanoidGraspRetentionFailureKind::RetentionDurationInsufficient);
    }
    if ever_retained && final_phase != HumanoidGraspRetentionPhase::Retained {
        failures.push(HumanoidGraspRetentionFailureKind::RetentionLostAtEnd);
    }

    let retained_at_end =
        failures.is_empty() && final_phase == HumanoidGraspRetentionPhase::Retained;

    let mut episode = HumanoidGraspRetentionEpisode {
        schema_version: HUMANOID_GRASP_RETENTION_EPISODE_SCHEMA_VERSION,
        subject_digest: digest_subject(subject)
            .ok_or(HumanoidGraspRetentionEpisodeFailure::InvalidSubject)?,
        object_id,
        hand,
        contact_policy_digest: contact_policy.policy_digest(),
        retention_policy_digest: retention_policy.policy_digest(),
        sample_digests,
        object_state_digests,
        started_at_s,
        ended_at_s,
        final_run_started_at_s: current_run_start_s,
        final_continuous_samples: current_run_samples,
        final_continuous_duration_s,
        continuity_breaks,
        reacquisitions,
        ever_stabilized,
        ever_retained,
        final_phase,
        failures,
        retained_at_end,
        episode_digest: HumanoidEvidenceDigest::ZERO,
    };
    episode.episode_digest = digest_episode(&episode);
    if !episode.validate_for(subject, contact_policy, retention_policy) {
        return Err(HumanoidGraspRetentionEpisodeFailure::InvalidDigest);
    }
    Ok(episode)
}

fn digest_sample(sample: &HumanoidGraspRetentionSample) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-retention-sample.v1");
    h.u32(sample.schema_version)
        .digest(sample.subject_digest)
        .string(&sample.object_id)
        .digest(sample.object_state_digest)
        .u64(hand_id(sample.hand))
        .digest(sample.contact_policy_digest)
        .digest(sample.observation_digest)
        .digest(sample.assessment_digest)
        .u64(source_id(sample.source))
        .f64(sample.sampled_at_s)
        .bool(sample.accepted_contact);
    h.finish()
}

fn digest_policy(policy: &HumanoidGraspRetentionPolicy) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-retention-policy.v1");
    h.u32(policy.schema_version)
        .digest(policy.subject_digest)
        .u64(hand_id(policy.hand))
        .digest(policy.contact_policy_digest)
        .usize(policy.minimum_stabilization_samples)
        .f64(policy.minimum_stabilization_duration_s)
        .usize(policy.minimum_retention_samples)
        .f64(policy.minimum_retention_duration_s)
        .f64(policy.maximum_inter_sample_gap_s)
        .usize(policy.maximum_episode_samples)
        .f64(policy.maximum_episode_duration_s);
    h.finish()
}

fn digest_episode(episode: &HumanoidGraspRetentionEpisode) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-retention-episode.v1");
    h.u32(episode.schema_version)
        .digest(episode.subject_digest)
        .string(&episode.object_id)
        .u64(hand_id(episode.hand))
        .digest(episode.contact_policy_digest)
        .digest(episode.retention_policy_digest)
        .usize(episode.sample_digests.len());
    for (sample_digest, state_digest) in episode
        .sample_digests
        .iter()
        .zip(episode.object_state_digests.iter())
    {
        h.digest(*sample_digest).digest(*state_digest);
    }
    h.f64(episode.started_at_s)
        .f64(episode.ended_at_s)
        .bool(episode.final_run_started_at_s.is_some());
    if let Some(start) = episode.final_run_started_at_s {
        h.f64(start);
    }
    h.usize(episode.final_continuous_samples)
        .f64(episode.final_continuous_duration_s)
        .usize(episode.continuity_breaks)
        .usize(episode.reacquisitions)
        .bool(episode.ever_stabilized)
        .bool(episode.ever_retained)
        .u64(phase_id(episode.final_phase))
        .usize(episode.failures.len());
    for failure in &episode.failures {
        h.u64(failure_id(*failure));
    }
    h.bool(episode.retained_at_end);
    h.finish()
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() || subject.task != HumanoidTask::Grasp {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-retention-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn phase_id(phase: HumanoidGraspRetentionPhase) -> u64 {
    match phase {
        HumanoidGraspRetentionPhase::Approach => 1,
        HumanoidGraspRetentionPhase::ContactAcquired => 2,
        HumanoidGraspRetentionPhase::Stabilizing => 3,
        HumanoidGraspRetentionPhase::Retained => 4,
        HumanoidGraspRetentionPhase::Lost => 5,
    }
}

fn failure_id(failure: HumanoidGraspRetentionFailureKind) -> u64 {
    match failure {
        HumanoidGraspRetentionFailureKind::NoMeasuredContactAtEnd => 1,
        HumanoidGraspRetentionFailureKind::StabilizationSamplesInsufficient => 2,
        HumanoidGraspRetentionFailureKind::StabilizationDurationInsufficient => 3,
        HumanoidGraspRetentionFailureKind::RetentionSamplesInsufficient => 4,
        HumanoidGraspRetentionFailureKind::RetentionDurationInsufficient => 5,
        HumanoidGraspRetentionFailureKind::RetentionLostAtEnd => 6,
    }
}

fn hand_id(hand: HandSide) -> u64 {
    match hand {
        HandSide::Right => 1,
        HandSide::Left => 2,
    }
}

fn source_id(source: HumanoidManipulationContactSource) -> u64 {
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
        HumanoidGraspContactPolicy, HumanoidManipulationContactSource,
        assess_humanoid_grasp_contact,
    };
    use crate::morphology::HumanoidMorphology;

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Grasp,
            ActuationMode::NormalizedTorque,
            "grasp-retention-test-backend",
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

    fn retention_policy(contact: &HumanoidGraspContactPolicy) -> HumanoidGraspRetentionPolicy {
        HumanoidGraspRetentionPolicy::new(&subject(), contact, 3, 0.04, 5, 0.08, 0.03, 64, 2.0)
            .unwrap()
    }

    fn sample(
        contact: &HumanoidGraspContactPolicy,
        timestamp_s: f64,
        state_byte: u8,
        accepted_contact: bool,
    ) -> HumanoidGraspRetentionSample {
        let source = if accepted_contact {
            HumanoidManipulationContactSource::ForceTorqueSensor
        } else {
            HumanoidManipulationContactSource::KinematicEstimate
        };
        let observation = HumanoidGraspContactObservation::new(
            &subject(),
            "object-a",
            HumanoidEvidenceDigest::from_bytes([state_byte; 32]),
            HandSide::Right,
            HumanoidContactSite::RightHand,
            accepted_contact,
            [0.2, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            if accepted_contact {
                [-10.0, 0.2, 0.0]
            } else {
                [0.0; 3]
            },
            [0.0; 3],
            [0.0; 3],
            0.95,
            source,
            timestamp_s,
        )
        .unwrap();
        let assessment =
            assess_humanoid_grasp_contact(&subject(), &observation, contact, timestamp_s + 0.001)
                .unwrap();
        HumanoidGraspRetentionSample::bind(&subject(), &observation, &assessment, contact).unwrap()
    }

    #[test]
    fn retention_policy_must_be_stricter_than_stabilization() {
        let contact = contact_policy();
        assert!(
            HumanoidGraspRetentionPolicy::new(
                &subject(),
                &contact,
                3,
                0.04,
                3,
                0.04,
                0.03,
                64,
                2.0,
            )
            .is_none()
        );
    }

    #[test]
    fn one_good_contact_cannot_prove_retention() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let samples = vec![sample(&contact, 1.0, 1, true)];
        let episode =
            evaluate_humanoid_grasp_retention(&subject(), &samples, &contact, &retention).unwrap();
        assert!(!episode.retained_at_end());
        assert_eq!(
            episode.final_phase(),
            HumanoidGraspRetentionPhase::ContactAcquired
        );
    }

    #[test]
    fn continuous_measured_contact_proves_retention() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let samples = [1.00, 1.02, 1.04, 1.06, 1.08]
            .into_iter()
            .enumerate()
            .map(|(index, time)| sample(&contact, time, index as u8 + 1, true))
            .collect::<Vec<_>>();
        let episode =
            evaluate_humanoid_grasp_retention(&subject(), &samples, &contact, &retention).unwrap();
        assert!(episode.retained_at_end(), "{:?}", episode.failures());
        assert_eq!(episode.final_phase(), HumanoidGraspRetentionPhase::Retained);
        assert_eq!(episode.final_continuous_samples(), 5);
        assert_eq!(episode.object_state_digests().len(), 5);
    }

    #[test]
    fn retained_then_lost_is_not_accepted_at_end() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let mut samples = [1.00, 1.02, 1.04, 1.06, 1.08]
            .into_iter()
            .enumerate()
            .map(|(index, time)| sample(&contact, time, index as u8 + 1, true))
            .collect::<Vec<_>>();
        samples.push(sample(&contact, 1.10, 9, false));
        let episode =
            evaluate_humanoid_grasp_retention(&subject(), &samples, &contact, &retention).unwrap();
        assert!(!episode.retained_at_end());
        assert!(episode.ever_retained());
        assert_eq!(episode.final_phase(), HumanoidGraspRetentionPhase::Lost);
        assert!(
            episode
                .failures()
                .contains(&HumanoidGraspRetentionFailureKind::RetentionLostAtEnd)
        );
    }

    #[test]
    fn contact_may_reacquire_after_loss() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let mut samples = vec![
            sample(&contact, 1.00, 1, true),
            sample(&contact, 1.02, 2, true),
            sample(&contact, 1.04, 3, false),
        ];
        // Keep the final sample comfortably beyond the 0.08 s retention boundary
        // so this test exercises reacquisition semantics, not binary-float edge
        // representation of an exactly-threshold duration.
        for (index, time) in [1.06, 1.08, 1.10, 1.12, 1.15].into_iter().enumerate() {
            samples.push(sample(&contact, time, index as u8 + 10, true));
        }
        let episode =
            evaluate_humanoid_grasp_retention(&subject(), &samples, &contact, &retention).unwrap();
        assert!(episode.retained_at_end(), "{:?}", episode.failures());
        assert_eq!(episode.reacquisitions(), 1);
    }

    #[test]
    fn large_sample_gap_breaks_continuity_once() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let samples = vec![
            sample(&contact, 1.00, 1, true),
            sample(&contact, 1.02, 2, true),
            sample(&contact, 1.10, 3, false),
            sample(&contact, 1.12, 4, true),
            sample(&contact, 1.14, 5, true),
        ];
        let episode =
            evaluate_humanoid_grasp_retention(&subject(), &samples, &contact, &retention).unwrap();
        assert_eq!(episode.continuity_breaks(), 1);
        assert!(!episode.retained_at_end());
    }

    #[test]
    fn object_substitution_fails_structurally() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let first = sample(&contact, 1.0, 1, true);
        let observation = HumanoidGraspContactObservation::new(
            &subject(),
            "object-b",
            HumanoidEvidenceDigest::from_bytes([2; 32]),
            HandSide::Right,
            HumanoidContactSite::RightHand,
            true,
            [0.2, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0],
            [0.0; 3],
            [0.0; 3],
            0.95,
            HumanoidManipulationContactSource::ForceTorqueSensor,
            1.02,
        )
        .unwrap();
        let assessment = assess_humanoid_grasp_contact(&subject(), &observation, &contact, 1.021)
            .unwrap();
        let second =
            HumanoidGraspRetentionSample::bind(&subject(), &observation, &assessment, &contact)
                .unwrap();
        assert_eq!(
            evaluate_humanoid_grasp_retention(
                &subject(),
                &[first, second],
                &contact,
                &retention,
            ),
            Err(HumanoidGraspRetentionEpisodeFailure::ObjectSubstitution)
        );
    }

    #[test]
    fn non_monotonic_sample_time_fails_structurally() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let samples = vec![
            sample(&contact, 1.02, 1, true),
            sample(&contact, 1.01, 2, true),
        ];
        assert_eq!(
            evaluate_humanoid_grasp_retention(&subject(), &samples, &contact, &retention),
            Err(HumanoidGraspRetentionEpisodeFailure::TimestampNotStrictlyIncreasing)
        );
    }
}
