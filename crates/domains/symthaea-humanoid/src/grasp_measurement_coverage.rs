// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent-reference measurement coverage for Grasp controller qualification.
//!
//! A qualification trial may contain only valid contact samples and still be
//! under-observed: a sparse stream can miss short force/slip transients. This
//! module adds a separate, precommitted reference-measurement chain and temporal
//! observability policy. It atomically mints the existing controller trial and a
//! coverage artifact from the same observation/assessment slices.
//!
//! The types here prove exact measurement-chain identity and evidence coverage;
//! they do not prove organizational independence of the measurement producer.
//! That remains an operational/provenance requirement outside this crate.
//! Nothing in this module grants motor authority or enables ObjectContact lowering.

use std::collections::BTreeSet;

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::execution_authority_scope::HumanoidExecutionPurpose;
use crate::grasp_contact_evidence::{
    HumanoidGraspContactAssessment, HumanoidGraspContactObservation, HumanoidGraspContactPolicy,
    HumanoidManipulationContactSource,
};
use crate::grasp_controller_qualification::{
    HumanoidGraspControllerCandidate, HumanoidGraspControllerQualificationPolicy,
    HumanoidGraspControllerQualificationTrial, HumanoidGraspControllerTrialBindFailure,
    HumanoidGraspControllerTrialContext, bind_humanoid_grasp_controller_qualification_trial,
};
use crate::grasp_retention_evidence::HumanoidGraspRetentionPolicy;
use crate::qualification::HumanoidQualificationSubject;
use crate::types::{ActuationMode, HumanoidTask};

pub const HUMANOID_GRASP_MEASUREMENT_CHAIN_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_MEASUREMENT_COVERAGE_POLICY_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_MEASUREMENT_COVERAGE_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_MEASURED_GRASP_CONTROLLER_TRIAL_SCHEMA_VERSION: u32 = 1;

/// Reference-measurement mode appropriate to one qualification environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspQualificationMeasurementKind {
    SimulationGroundTruth,
    HilReferenceInstrumentation,
    PhysicalReferenceInstrumentation,
}

impl HumanoidGraspQualificationMeasurementKind {
    pub const fn admits(self, purpose: HumanoidExecutionPurpose) -> bool {
        matches!(
            (self, purpose),
            (
                Self::SimulationGroundTruth,
                HumanoidExecutionPurpose::SimulationQualification
            ) | (
                Self::HilReferenceInstrumentation,
                HumanoidExecutionPurpose::HilQualification
            ) | (
                Self::PhysicalReferenceInstrumentation,
                HumanoidExecutionPurpose::PhysicalQualification
            )
        )
    }
}

/// Exact identity of the reference chain used to judge a candidate controller.
///
/// `reference_artifact_digest` identifies the reference instrument/model itself;
/// `reference_configuration_digest` covers calibration or simulator-ground-truth
/// configuration; acquisition software and clock configuration are separate so
/// either can be changed without silently reusing prior qualification evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspQualificationMeasurementChain {
    schema_version: u32,
    chain_id: String,
    kind: HumanoidGraspQualificationMeasurementKind,
    reference_artifact_digest: HumanoidEvidenceDigest,
    reference_configuration_digest: HumanoidEvidenceDigest,
    acquisition_software_digest: HumanoidEvidenceDigest,
    clock_configuration_digest: HumanoidEvidenceDigest,
    chain_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspQualificationMeasurementChain {
    pub fn new(
        chain_id: impl Into<String>,
        kind: HumanoidGraspQualificationMeasurementKind,
        reference_artifact_digest: HumanoidEvidenceDigest,
        reference_configuration_digest: HumanoidEvidenceDigest,
        acquisition_software_digest: HumanoidEvidenceDigest,
        clock_configuration_digest: HumanoidEvidenceDigest,
    ) -> Option<Self> {
        let mut value = Self {
            schema_version: HUMANOID_GRASP_MEASUREMENT_CHAIN_SCHEMA_VERSION,
            chain_id: chain_id.into(),
            kind,
            reference_artifact_digest,
            reference_configuration_digest,
            acquisition_software_digest,
            clock_configuration_digest,
            chain_digest: HumanoidEvidenceDigest::ZERO,
        };
        if !value.base_valid() {
            return None;
        }
        value.chain_digest = digest_measurement_chain(&value);
        value.validate().then_some(value)
    }

    fn base_valid(&self) -> bool {
        self.schema_version == HUMANOID_GRASP_MEASUREMENT_CHAIN_SCHEMA_VERSION
            && valid_id(&self.chain_id)
            && !self.reference_artifact_digest.is_zero()
            && !self.reference_configuration_digest.is_zero()
            && !self.acquisition_software_digest.is_zero()
            && !self.clock_configuration_digest.is_zero()
    }

    pub fn validate(&self) -> bool {
        self.base_valid()
            && !self.chain_digest.is_zero()
            && self.chain_digest == digest_measurement_chain(self)
    }

    pub const fn kind(&self) -> HumanoidGraspQualificationMeasurementKind {
        self.kind
    }

    pub const fn chain_digest(&self) -> HumanoidEvidenceDigest {
        self.chain_digest
    }
}

/// Precommitted temporal-observability policy for qualification measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspMeasurementCoveragePolicy {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    controller_candidate_digest: HumanoidEvidenceDigest,
    controller_qualification_policy_digest: HumanoidEvidenceDigest,
    contact_policy_digest: HumanoidEvidenceDigest,
    retention_policy_digest: HumanoidEvidenceDigest,
    measurement_chain_digest: HumanoidEvidenceDigest,
    minimum_observations: usize,
    maximum_observation_gap_s: f64,
    allowed_contact_sources: Vec<HumanoidManipulationContactSource>,
    policy_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspMeasurementCoveragePolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        candidate: &HumanoidGraspControllerCandidate,
        controller_policy: &HumanoidGraspControllerQualificationPolicy,
        contact_policy: &HumanoidGraspContactPolicy,
        retention_policy: &HumanoidGraspRetentionPolicy,
        measurement_chain: &HumanoidGraspQualificationMeasurementChain,
        minimum_observations: usize,
        maximum_observation_gap_s: f64,
        mut allowed_contact_sources: Vec<HumanoidManipulationContactSource>,
    ) -> Option<Self> {
        if !subject.validate()
            || subject.task != HumanoidTask::Grasp
            || !candidate.validate()
            || !controller_policy.validate_for(subject, candidate, contact_policy, retention_policy)
            || !measurement_chain.validate()
            || minimum_observations < 2
            || minimum_observations > 1_000_000
            || !maximum_observation_gap_s.is_finite()
            || maximum_observation_gap_s <= 0.0
            || allowed_contact_sources.is_empty()
            || allowed_contact_sources.iter().any(|source| !source.is_measured())
        {
            return None;
        }
        allowed_contact_sources.sort();
        allowed_contact_sources.dedup();

        let mut value = Self {
            schema_version: HUMANOID_GRASP_MEASUREMENT_COVERAGE_POLICY_SCHEMA_VERSION,
            subject_digest: digest_subject(subject)?,
            controller_candidate_digest: candidate.candidate_digest(),
            controller_qualification_policy_digest: controller_policy.policy_digest(),
            contact_policy_digest: contact_policy.policy_digest(),
            retention_policy_digest: retention_policy.policy_digest(),
            measurement_chain_digest: measurement_chain.chain_digest(),
            minimum_observations,
            maximum_observation_gap_s,
            allowed_contact_sources,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.policy_digest = digest_measurement_policy(&value);
        value
            .validate_for(
                subject,
                candidate,
                controller_policy,
                contact_policy,
                retention_policy,
                measurement_chain,
            )
            .then_some(value)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        candidate: &HumanoidGraspControllerCandidate,
        controller_policy: &HumanoidGraspControllerQualificationPolicy,
        contact_policy: &HumanoidGraspContactPolicy,
        retention_policy: &HumanoidGraspRetentionPolicy,
        measurement_chain: &HumanoidGraspQualificationMeasurementChain,
    ) -> bool {
        self.schema_version == HUMANOID_GRASP_MEASUREMENT_COVERAGE_POLICY_SCHEMA_VERSION
            && digest_subject(subject) == Some(self.subject_digest)
            && candidate.validate()
            && self.controller_candidate_digest == candidate.candidate_digest()
            && controller_policy.validate_for(subject, candidate, contact_policy, retention_policy)
            && self.controller_qualification_policy_digest == controller_policy.policy_digest()
            && contact_policy.validate_for(subject)
            && self.contact_policy_digest == contact_policy.policy_digest()
            && retention_policy.validate_for(subject, contact_policy)
            && self.retention_policy_digest == retention_policy.policy_digest()
            && measurement_chain.validate()
            && self.measurement_chain_digest == measurement_chain.chain_digest()
            && self.minimum_observations >= 2
            && self.minimum_observations <= 1_000_000
            && self.maximum_observation_gap_s.is_finite()
            && self.maximum_observation_gap_s > 0.0
            && !self.allowed_contact_sources.is_empty()
            && self.allowed_contact_sources.iter().all(|source| source.is_measured())
            && self
                .allowed_contact_sources
                .windows(2)
                .all(|window| window[0] < window[1])
            && !self.policy_digest.is_zero()
            && self.policy_digest == digest_measurement_policy(self)
    }

    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspMeasurementCoverageFailureKind {
    InsufficientObservations,
    ObservationGapTooLarge,
    DisallowedContactSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspMeasurementCoverageBindFailure {
    InvalidSubject,
    InvalidMeasurementPolicy,
    MeasurementPurposeMismatch,
    BaseTrial(HumanoidGraspControllerTrialBindFailure),
    EvidenceLengthMismatch,
    EmptyEvidence,
    InvalidObservationOrAssessment,
    TimestampNotStrictlyIncreasing,
    InvalidDigest,
}

/// Immutable evidence that the independent/reference stream was sufficiently
/// observable for the associated exact controller qualification trial.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspMeasurementCoverageArtifact {
    schema_version: u32,
    trial_digest: HumanoidEvidenceDigest,
    policy_digest: HumanoidEvidenceDigest,
    measurement_chain_digest: HumanoidEvidenceDigest,
    observation_digests: Vec<HumanoidEvidenceDigest>,
    assessment_digests: Vec<HumanoidEvidenceDigest>,
    sample_count: usize,
    first_timestamp_s: f64,
    last_timestamp_s: f64,
    maximum_observed_gap_s: f64,
    gap_violation_count: usize,
    disallowed_source_count: usize,
    failures: Vec<HumanoidGraspMeasurementCoverageFailureKind>,
    accepted: bool,
    artifact_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspMeasurementCoverageArtifact {
    pub fn validate_for(
        &self,
        trial: &HumanoidGraspControllerQualificationTrial,
        policy: &HumanoidGraspMeasurementCoveragePolicy,
        measurement_chain: &HumanoidGraspQualificationMeasurementChain,
    ) -> bool {
        self.schema_version == HUMANOID_GRASP_MEASUREMENT_COVERAGE_SCHEMA_VERSION
            && self.trial_digest == trial.trial_digest()
            && !self.trial_digest.is_zero()
            && self.policy_digest == policy.policy_digest()
            && self.measurement_chain_digest == measurement_chain.chain_digest()
            && self.sample_count == self.observation_digests.len()
            && self.sample_count == self.assessment_digests.len()
            && self.sample_count > 0
            && self.observation_digests.iter().all(|digest| !digest.is_zero())
            && self.assessment_digests.iter().all(|digest| !digest.is_zero())
            && self.first_timestamp_s.is_finite()
            && self.last_timestamp_s.is_finite()
            && self.last_timestamp_s >= self.first_timestamp_s
            && self.maximum_observed_gap_s.is_finite()
            && self.maximum_observed_gap_s >= 0.0
            && self.gap_violation_count <= self.sample_count.saturating_sub(1)
            && self.disallowed_source_count <= self.sample_count
            && self.accepted == self.failures.is_empty()
            && !self.artifact_digest.is_zero()
            && self.artifact_digest == digest_measurement_coverage(self)
    }

    pub const fn accepted(&self) -> bool {
        self.accepted
    }

    pub const fn maximum_observed_gap_s(&self) -> f64 {
        self.maximum_observed_gap_s
    }

    pub const fn gap_violation_count(&self) -> usize {
        self.gap_violation_count
    }

    pub const fn disallowed_source_count(&self) -> usize {
        self.disallowed_source_count
    }

    pub fn failures(&self) -> &[HumanoidGraspMeasurementCoverageFailureKind] {
        &self.failures
    }

    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.artifact_digest
    }
}

/// Atomic result: both artifacts were derived from the exact same evidence slices.
#[derive(Debug)]
pub struct HumanoidMeasuredGraspControllerQualificationTrial {
    schema_version: u32,
    trial: HumanoidGraspControllerQualificationTrial,
    measurement: HumanoidGraspMeasurementCoverageArtifact,
    bundle_digest: HumanoidEvidenceDigest,
}

impl HumanoidMeasuredGraspControllerQualificationTrial {
    pub fn trial(&self) -> &HumanoidGraspControllerQualificationTrial {
        &self.trial
    }

    pub fn measurement(&self) -> &HumanoidGraspMeasurementCoverageArtifact {
        &self.measurement
    }

    pub fn qualification_evidence_accepted(&self) -> bool {
        self.trial.trial_accepted() && self.measurement.accepted()
    }

    pub const fn bundle_digest(&self) -> HumanoidEvidenceDigest {
        self.bundle_digest
    }

    pub fn validate_for(
        &self,
        measurement_policy: &HumanoidGraspMeasurementCoveragePolicy,
        measurement_chain: &HumanoidGraspQualificationMeasurementChain,
    ) -> bool {
        self.schema_version == HUMANOID_MEASURED_GRASP_CONTROLLER_TRIAL_SCHEMA_VERSION
            && self.measurement.validate_for(&self.trial, measurement_policy, measurement_chain)
            && !self.bundle_digest.is_zero()
            && self.bundle_digest == digest_measured_trial(self)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bind_humanoid_measured_grasp_controller_qualification_trial(
    subject: &HumanoidQualificationSubject,
    candidate: &HumanoidGraspControllerCandidate,
    controller_policy: &HumanoidGraspControllerQualificationPolicy,
    contact_policy: &HumanoidGraspContactPolicy,
    retention_policy: &HumanoidGraspRetentionPolicy,
    measurement_chain: &HumanoidGraspQualificationMeasurementChain,
    measurement_policy: &HumanoidGraspMeasurementCoveragePolicy,
    context: &HumanoidGraspControllerTrialContext,
    observations: &[HumanoidGraspContactObservation],
    assessments: &[HumanoidGraspContactAssessment],
) -> Result<HumanoidMeasuredGraspControllerQualificationTrial, HumanoidGraspMeasurementCoverageBindFailure>
{
    if !subject.validate() || subject.task != HumanoidTask::Grasp {
        return Err(HumanoidGraspMeasurementCoverageBindFailure::InvalidSubject);
    }
    if !measurement_policy.validate_for(
        subject,
        candidate,
        controller_policy,
        contact_policy,
        retention_policy,
        measurement_chain,
    ) {
        return Err(HumanoidGraspMeasurementCoverageBindFailure::InvalidMeasurementPolicy);
    }
    if !measurement_chain.kind().admits(context.execution_purpose) {
        return Err(HumanoidGraspMeasurementCoverageBindFailure::MeasurementPurposeMismatch);
    }

    let trial = bind_humanoid_grasp_controller_qualification_trial(
        subject,
        candidate,
        controller_policy,
        contact_policy,
        retention_policy,
        context,
        observations,
        assessments,
    )
    .map_err(HumanoidGraspMeasurementCoverageBindFailure::BaseTrial)?;

    let measurement = assess_measurement_coverage(
        subject,
        &trial,
        contact_policy,
        measurement_chain,
        measurement_policy,
        observations,
        assessments,
    )?;

    let mut value = HumanoidMeasuredGraspControllerQualificationTrial {
        schema_version: HUMANOID_MEASURED_GRASP_CONTROLLER_TRIAL_SCHEMA_VERSION,
        trial,
        measurement,
        bundle_digest: HumanoidEvidenceDigest::ZERO,
    };
    value.bundle_digest = digest_measured_trial(&value);
    if !value.validate_for(measurement_policy, measurement_chain) {
        return Err(HumanoidGraspMeasurementCoverageBindFailure::InvalidDigest);
    }
    Ok(value)
}

#[allow(clippy::too_many_arguments)]
fn assess_measurement_coverage(
    subject: &HumanoidQualificationSubject,
    trial: &HumanoidGraspControllerQualificationTrial,
    contact_policy: &HumanoidGraspContactPolicy,
    measurement_chain: &HumanoidGraspQualificationMeasurementChain,
    measurement_policy: &HumanoidGraspMeasurementCoveragePolicy,
    observations: &[HumanoidGraspContactObservation],
    assessments: &[HumanoidGraspContactAssessment],
) -> Result<HumanoidGraspMeasurementCoverageArtifact, HumanoidGraspMeasurementCoverageBindFailure>
{
    if observations.len() != assessments.len() {
        return Err(HumanoidGraspMeasurementCoverageBindFailure::EvidenceLengthMismatch);
    }
    let Some(first) = observations.first() else {
        return Err(HumanoidGraspMeasurementCoverageBindFailure::EmptyEvidence);
    };

    let mut observation_digests = Vec::with_capacity(observations.len());
    let mut assessment_digests = Vec::with_capacity(assessments.len());
    let mut previous_timestamp_s: Option<f64> = None;
    let mut maximum_observed_gap_s = 0.0_f64;
    let mut gap_violation_count = 0usize;
    let mut disallowed_source_count = 0usize;

    for (observation, assessment) in observations.iter().zip(assessments) {
        if !observation.validate_for(subject)
            || !assessment.validate(subject, observation, contact_policy)
        {
            return Err(HumanoidGraspMeasurementCoverageBindFailure::InvalidObservationOrAssessment);
        }
        if let Some(previous) = previous_timestamp_s {
            if observation.timestamp_s() <= previous {
                return Err(HumanoidGraspMeasurementCoverageBindFailure::TimestampNotStrictlyIncreasing);
            }
            let gap_s = observation.timestamp_s() - previous;
            maximum_observed_gap_s = maximum_observed_gap_s.max(gap_s);
            if gap_s > measurement_policy.maximum_observation_gap_s {
                gap_violation_count += 1;
            }
        }
        if !measurement_policy.allowed_contact_sources.contains(&observation.source()) {
            disallowed_source_count += 1;
        }
        observation_digests.push(observation.observation_digest());
        assessment_digests.push(assessment.assessment_digest());
        previous_timestamp_s = Some(observation.timestamp_s());
    }

    let mut failures = Vec::new();
    if observations.len() < measurement_policy.minimum_observations {
        failures.push(HumanoidGraspMeasurementCoverageFailureKind::InsufficientObservations);
    }
    if gap_violation_count > 0 {
        failures.push(HumanoidGraspMeasurementCoverageFailureKind::ObservationGapTooLarge);
    }
    if disallowed_source_count > 0 {
        failures.push(HumanoidGraspMeasurementCoverageFailureKind::DisallowedContactSource);
    }

    let mut value = HumanoidGraspMeasurementCoverageArtifact {
        schema_version: HUMANOID_GRASP_MEASUREMENT_COVERAGE_SCHEMA_VERSION,
        trial_digest: trial.trial_digest(),
        policy_digest: measurement_policy.policy_digest(),
        measurement_chain_digest: measurement_chain.chain_digest(),
        observation_digests,
        assessment_digests,
        sample_count: observations.len(),
        first_timestamp_s: first.timestamp_s(),
        last_timestamp_s: observations.last().unwrap().timestamp_s(),
        maximum_observed_gap_s,
        gap_violation_count,
        disallowed_source_count,
        accepted: failures.is_empty(),
        failures,
        artifact_digest: HumanoidEvidenceDigest::ZERO,
    };
    value.artifact_digest = digest_measurement_coverage(&value);
    if !value.validate_for(trial, measurement_policy, measurement_chain) {
        return Err(HumanoidGraspMeasurementCoverageBindFailure::InvalidDigest);
    }
    Ok(value)
}

fn digest_measurement_chain(
    value: &HumanoidGraspQualificationMeasurementChain,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-qualification-measurement-chain.v1");
    h.u32(value.schema_version)
        .string(&value.chain_id)
        .u64(measurement_kind_id(value.kind))
        .digest(value.reference_artifact_digest)
        .digest(value.reference_configuration_digest)
        .digest(value.acquisition_software_digest)
        .digest(value.clock_configuration_digest);
    h.finish()
}

fn digest_measurement_policy(
    value: &HumanoidGraspMeasurementCoveragePolicy,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-measurement-coverage-policy.v1");
    h.u32(value.schema_version)
        .digest(value.subject_digest)
        .digest(value.controller_candidate_digest)
        .digest(value.controller_qualification_policy_digest)
        .digest(value.contact_policy_digest)
        .digest(value.retention_policy_digest)
        .digest(value.measurement_chain_digest)
        .usize(value.minimum_observations)
        .f64(value.maximum_observation_gap_s)
        .usize(value.allowed_contact_sources.len());
    for source in &value.allowed_contact_sources {
        h.u64(contact_source_id(*source));
    }
    h.finish()
}

fn digest_measurement_coverage(
    value: &HumanoidGraspMeasurementCoverageArtifact,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-measurement-coverage.v1");
    h.u32(value.schema_version)
        .digest(value.trial_digest)
        .digest(value.policy_digest)
        .digest(value.measurement_chain_digest)
        .usize(value.observation_digests.len());
    for digest in &value.observation_digests {
        h.digest(*digest);
    }
    h.usize(value.assessment_digests.len());
    for digest in &value.assessment_digests {
        h.digest(*digest);
    }
    h.usize(value.sample_count)
        .f64(value.first_timestamp_s)
        .f64(value.last_timestamp_s)
        .f64(value.maximum_observed_gap_s)
        .usize(value.gap_violation_count)
        .usize(value.disallowed_source_count)
        .usize(value.failures.len());
    for failure in &value.failures {
        h.u64(measurement_failure_id(*failure));
    }
    h.bool(value.accepted);
    h.finish()
}

fn digest_measured_trial(
    value: &HumanoidMeasuredGraspControllerQualificationTrial,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.measured-grasp-controller-trial.v1");
    h.u32(value.schema_version)
        .digest(value.trial.trial_digest())
        .digest(value.measurement.artifact_digest());
    h.finish()
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() || subject.task != HumanoidTask::Grasp {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-measurement-coverage-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn measurement_kind_id(kind: HumanoidGraspQualificationMeasurementKind) -> u64 {
    match kind {
        HumanoidGraspQualificationMeasurementKind::SimulationGroundTruth => 1,
        HumanoidGraspQualificationMeasurementKind::HilReferenceInstrumentation => 2,
        HumanoidGraspQualificationMeasurementKind::PhysicalReferenceInstrumentation => 3,
    }
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

fn measurement_failure_id(failure: HumanoidGraspMeasurementCoverageFailureKind) -> u64 {
    match failure {
        HumanoidGraspMeasurementCoverageFailureKind::InsufficientObservations => 1,
        HumanoidGraspMeasurementCoverageFailureKind::ObservationGapTooLarge => 2,
        HumanoidGraspMeasurementCoverageFailureKind::DisallowedContactSource => 3,
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

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact_site::HumanoidContactSite;
    use crate::grasp_contact_evidence::assess_humanoid_grasp_contact;
    use crate::grasp_controller_qualification::{
        HumanoidGraspControllerScenarioCell, HumanoidGraspControllerScenarioRequirement,
    };
    use crate::morphology::{HandSide, HumanoidMorphology};

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Grasp,
            ActuationMode::NormalizedTorque,
            "grasp-measurement-test-backend",
        )
    }

    fn candidate() -> HumanoidGraspControllerCandidate {
        HumanoidGraspControllerCandidate::new(
            "grasp-controller-v1",
            HumanoidEvidenceDigest::from_bytes([1; 32]),
            HumanoidEvidenceDigest::from_bytes([2; 32]),
        )
        .unwrap()
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

    fn scenario() -> HumanoidGraspControllerScenarioRequirement {
        HumanoidGraspControllerScenarioRequirement {
            cell: HumanoidGraspControllerScenarioCell {
                scenario_id: "center".into(),
                hand: HandSide::Right,
                minimum_workspace_utilization_sq: 0.0,
                maximum_workspace_utilization_sq: 0.5,
                object_fixture_class_id: "rigid-small-v1".into(),
                object_fixture_class_digest: HumanoidEvidenceDigest::from_bytes([7; 32]),
                perturbation_profile_id: "nominal-v1".into(),
                perturbation_profile_digest: HumanoidEvidenceDigest::from_bytes([5; 32]),
                sensor_fault_profile_id: "sensors-nominal-v1".into(),
                sensor_fault_profile_digest: HumanoidEvidenceDigest::from_bytes([6; 32]),
            },
            minimum_trials: 1,
            maximum_trial_failure_rate: 0.0,
            maximum_false_retention_rate: 0.0,
            maximum_false_negative_rate: 0.0,
            maximum_retention_loss_rate: 0.0,
            maximum_continuity_break_trial_rate: 0.0,
            minimum_distinct_object_fixtures: 1,
            minimum_distinct_environment_digests: 1,
            require_unique_trial_seeds: true,
        }
    }

    fn controller_policy(
        contact: &HumanoidGraspContactPolicy,
        retention: &HumanoidGraspRetentionPolicy,
    ) -> HumanoidGraspControllerQualificationPolicy {
        HumanoidGraspControllerQualificationPolicy::new(
            &subject(),
            &candidate(),
            contact,
            retention,
            "grasp-measurement-campaign-v1",
            HumanoidExecutionPurpose::SimulationQualification,
            vec![scenario()],
        )
        .unwrap()
    }

    fn chain(kind: HumanoidGraspQualificationMeasurementKind) -> HumanoidGraspQualificationMeasurementChain {
        HumanoidGraspQualificationMeasurementChain::new(
            "reference-chain-v1",
            kind,
            HumanoidEvidenceDigest::from_bytes([20; 32]),
            HumanoidEvidenceDigest::from_bytes([21; 32]),
            HumanoidEvidenceDigest::from_bytes([22; 32]),
            HumanoidEvidenceDigest::from_bytes([23; 32]),
        )
        .unwrap()
    }

    fn measurement_policy(
        contact: &HumanoidGraspContactPolicy,
        retention: &HumanoidGraspRetentionPolicy,
        controller: &HumanoidGraspControllerQualificationPolicy,
        chain: &HumanoidGraspQualificationMeasurementChain,
        maximum_observation_gap_s: f64,
        allowed_sources: Vec<HumanoidManipulationContactSource>,
    ) -> HumanoidGraspMeasurementCoveragePolicy {
        HumanoidGraspMeasurementCoveragePolicy::new(
            &subject(),
            &candidate(),
            controller,
            contact,
            retention,
            chain,
            5,
            maximum_observation_gap_s,
            allowed_sources,
        )
        .unwrap()
    }

    fn context() -> HumanoidGraspControllerTrialContext {
        HumanoidGraspControllerTrialContext {
            trial_id: "trial-center-1".into(),
            trial_seed: 42,
            scenario_id: "center".into(),
            execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
            backend_profile_id: subject().backend_profile_id,
            workspace_utilization_sq: 0.25,
            object_fixture_id: "fixture-a".into(),
            object_fixture_class_id: "rigid-small-v1".into(),
            object_fixture_class_digest: HumanoidEvidenceDigest::from_bytes([7; 32]),
            object_fixture_digest: HumanoidEvidenceDigest::from_bytes([3; 32]),
            environment_digest: HumanoidEvidenceDigest::from_bytes([4; 32]),
            perturbation_profile_id: "nominal-v1".into(),
            perturbation_profile_digest: HumanoidEvidenceDigest::from_bytes([5; 32]),
            sensor_fault_profile_id: "sensors-nominal-v1".into(),
            sensor_fault_profile_digest: HumanoidEvidenceDigest::from_bytes([6; 32]),
            controller_reported_retained: true,
        }
    }

    fn evidence(
        source: HumanoidManipulationContactSource,
        times: &[f64],
    ) -> (
        Vec<HumanoidGraspContactObservation>,
        Vec<HumanoidGraspContactAssessment>,
    ) {
        let contact = contact_policy();
        let mut observations = Vec::new();
        let mut assessments = Vec::new();
        for (index, time) in times.iter().copied().enumerate() {
            let observation = HumanoidGraspContactObservation::new(
                &subject(),
                "object-a",
                HumanoidEvidenceDigest::from_bytes([30 + index as u8; 32]),
                HandSide::Right,
                HumanoidContactSite::RightHand,
                true,
                [0.2, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [-10.0, 0.1, 0.0],
                [0.0; 3],
                [0.0; 3],
                0.95,
                source,
                time,
            )
            .unwrap();
            let assessment = assess_humanoid_grasp_contact(
                &subject(),
                &observation,
                &contact,
                time + 0.001,
            )
            .unwrap();
            observations.push(observation);
            assessments.push(assessment);
        }
        (observations, assessments)
    }

    #[test]
    fn dense_reference_stream_can_be_accepted() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let controller = controller_policy(&contact, &retention);
        let chain = chain(HumanoidGraspQualificationMeasurementKind::SimulationGroundTruth);
        let measurement = measurement_policy(
            &contact,
            &retention,
            &controller,
            &chain,
            0.015,
            vec![HumanoidManipulationContactSource::SolverWrench],
        );
        let (observations, assessments) = evidence(
            HumanoidManipulationContactSource::SolverWrench,
            &[1.000, 1.012, 1.024, 1.036, 1.048, 1.060, 1.072, 1.085],
        );
        let bundle = bind_humanoid_measured_grasp_controller_qualification_trial(
            &subject(),
            &candidate(),
            &controller,
            &contact,
            &retention,
            &chain,
            &measurement,
            &context(),
            &observations,
            &assessments,
        )
        .unwrap();
        assert!(bundle.trial().trial_accepted());
        assert!(bundle.measurement().accepted());
        assert!(bundle.qualification_evidence_accepted());
        assert!(bundle.validate_for(&measurement, &chain));
    }

    #[test]
    fn retention_success_does_not_hide_sparse_measurement_coverage() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let controller = controller_policy(&contact, &retention);
        let chain = chain(HumanoidGraspQualificationMeasurementKind::SimulationGroundTruth);
        let measurement = measurement_policy(
            &contact,
            &retention,
            &controller,
            &chain,
            0.015,
            vec![HumanoidManipulationContactSource::SolverWrench],
        );
        let (observations, assessments) = evidence(
            HumanoidManipulationContactSource::SolverWrench,
            &[1.000, 1.020, 1.040, 1.060, 1.085],
        );
        let bundle = bind_humanoid_measured_grasp_controller_qualification_trial(
            &subject(),
            &candidate(),
            &controller,
            &contact,
            &retention,
            &chain,
            &measurement,
            &context(),
            &observations,
            &assessments,
        )
        .unwrap();
        assert!(bundle.trial().trial_accepted());
        assert!(!bundle.measurement().accepted());
        assert!(bundle.measurement().gap_violation_count() > 0);
        assert!(!bundle.qualification_evidence_accepted());
    }

    #[test]
    fn disallowed_measurement_source_blocks_qualification_evidence() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let controller = controller_policy(&contact, &retention);
        let chain = chain(HumanoidGraspQualificationMeasurementKind::SimulationGroundTruth);
        let measurement = measurement_policy(
            &contact,
            &retention,
            &controller,
            &chain,
            0.015,
            vec![HumanoidManipulationContactSource::SolverWrench],
        );
        let (observations, assessments) = evidence(
            HumanoidManipulationContactSource::ForceTorqueSensor,
            &[1.000, 1.012, 1.024, 1.036, 1.048, 1.060, 1.072, 1.085],
        );
        let bundle = bind_humanoid_measured_grasp_controller_qualification_trial(
            &subject(),
            &candidate(),
            &controller,
            &contact,
            &retention,
            &chain,
            &measurement,
            &context(),
            &observations,
            &assessments,
        )
        .unwrap();
        assert!(bundle.trial().trial_accepted());
        assert!(!bundle.measurement().accepted());
        assert!(bundle.measurement().disallowed_source_count() > 0);
    }

    #[test]
    fn measurement_kind_must_match_qualification_environment() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let controller = controller_policy(&contact, &retention);
        let chain = chain(HumanoidGraspQualificationMeasurementKind::PhysicalReferenceInstrumentation);
        let measurement = measurement_policy(
            &contact,
            &retention,
            &controller,
            &chain,
            0.015,
            vec![HumanoidManipulationContactSource::SolverWrench],
        );
        let (observations, assessments) = evidence(
            HumanoidManipulationContactSource::SolverWrench,
            &[1.000, 1.012, 1.024, 1.036, 1.048, 1.060, 1.072, 1.085],
        );
        assert_eq!(
            bind_humanoid_measured_grasp_controller_qualification_trial(
                &subject(),
                &candidate(),
                &controller,
                &contact,
                &retention,
                &chain,
                &measurement,
                &context(),
                &observations,
                &assessments,
            ),
            Err(HumanoidGraspMeasurementCoverageBindFailure::MeasurementPurposeMismatch)
        );
    }

    #[test]
    fn reference_configuration_change_changes_chain_identity() {
        let a = chain(HumanoidGraspQualificationMeasurementKind::SimulationGroundTruth);
        let b = HumanoidGraspQualificationMeasurementChain::new(
            "reference-chain-v1",
            HumanoidGraspQualificationMeasurementKind::SimulationGroundTruth,
            HumanoidEvidenceDigest::from_bytes([20; 32]),
            HumanoidEvidenceDigest::from_bytes([99; 32]),
            HumanoidEvidenceDigest::from_bytes([22; 32]),
            HumanoidEvidenceDigest::from_bytes([23; 32]),
        )
        .unwrap();
        assert_ne!(a.chain_digest(), b.chain_digest());
    }
}
