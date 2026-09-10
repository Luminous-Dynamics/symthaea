// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Causal evidence binding for the proposal-only Grasp simulation controller.
//!
//! This module does **not** translate a semantic proposal into `HumanoidCommand`,
//! call `HumanoidPhysicsSimulator::step`, or expose any physical/hardware path.
//! It defines a two-phase evidence protocol around a separately reviewed simulation
//! adapter: precommit the exact proposal/harness/pre-state, then bind the exact
//! post-state and same-frame canonical contact observation.
//!
//! Complete traces consume the controller session when sealed. That matters: a
//! hash-contiguous prefix is not complete evidence if the same session can later
//! emit omitted proposals. Session consumption makes closure type-level final for
//! that session value. A terminal `Abort` proposal is recorded as a final decision
//! but is never treated as a plant step.
//!
//! This is structural causal lineage, not authenticated proof that an external
//! simulator process actually executed; producer/executor attestation remains a
//! separate evidence-provenance concern.

use super::{
    HumanoidGraspSimulationControllerCandidate, HumanoidGraspSimulationControllerProposal,
    HumanoidGraspSimulationControllerSession, HumanoidGraspSimulationProposalKind,
};
use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::execution_authority_scope::HumanoidExecutionPurpose;
use crate::grasp_acquisition_intent::HumanoidGraspAcquisitionIntent;
use crate::grasp_contact_evidence::HumanoidGraspContactObservation;
use crate::morphology::HandSide;
use crate::qualification::HumanoidQualificationSubject;
use crate::types::{ActuationMode, HumanoidState, HumanoidTask};

pub const HUMANOID_GRASP_SIMULATION_HARNESS_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_SIMULATION_STEP_COMMITMENT_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_SIMULATION_STEP_RESULT_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_GRASP_SIMULATION_CAUSAL_TRACE_SCHEMA_VERSION: u32 = 2;
const MAX_INTEGRATION_SUBSTEPS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspSimulationDeterminismMode {
    DeterministicReplay,
    SeededStochastic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspSimulationTraceClosureKind {
    QualificationWindowClosed,
    ControllerAborted,
}

/// Exact identity of the simulation environment that turns one semantic proposal
/// into one simulation transition and extracts the canonical contact observation.
///
/// The proposal adapter is identified here but intentionally not implemented here.
/// Proposal -> simulator-command lowering remains its own review boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspSimulationHarnessIdentity {
    schema_version: u32,
    harness_id: String,
    simulator_backend_id: String,
    simulator_artifact_digest: HumanoidEvidenceDigest,
    simulator_model_digest: HumanoidEvidenceDigest,
    simulator_configuration_digest: HumanoidEvidenceDigest,
    proposal_adapter_artifact_digest: HumanoidEvidenceDigest,
    proposal_adapter_configuration_digest: HumanoidEvidenceDigest,
    contact_extractor_artifact_digest: HumanoidEvidenceDigest,
    contact_extractor_configuration_digest: HumanoidEvidenceDigest,
    randomization_profile_digest: HumanoidEvidenceDigest,
    determinism_mode: HumanoidGraspSimulationDeterminismMode,
    harness_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspSimulationHarnessIdentity {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        harness_id: impl Into<String>,
        simulator_backend_id: impl Into<String>,
        simulator_artifact_digest: HumanoidEvidenceDigest,
        simulator_model_digest: HumanoidEvidenceDigest,
        simulator_configuration_digest: HumanoidEvidenceDigest,
        proposal_adapter_artifact_digest: HumanoidEvidenceDigest,
        proposal_adapter_configuration_digest: HumanoidEvidenceDigest,
        contact_extractor_artifact_digest: HumanoidEvidenceDigest,
        contact_extractor_configuration_digest: HumanoidEvidenceDigest,
        randomization_profile_digest: HumanoidEvidenceDigest,
        determinism_mode: HumanoidGraspSimulationDeterminismMode,
    ) -> Option<Self> {
        let mut value = Self {
            schema_version: HUMANOID_GRASP_SIMULATION_HARNESS_SCHEMA_VERSION,
            harness_id: harness_id.into(),
            simulator_backend_id: simulator_backend_id.into(),
            simulator_artifact_digest,
            simulator_model_digest,
            simulator_configuration_digest,
            proposal_adapter_artifact_digest,
            proposal_adapter_configuration_digest,
            contact_extractor_artifact_digest,
            contact_extractor_configuration_digest,
            randomization_profile_digest,
            determinism_mode,
            harness_digest: HumanoidEvidenceDigest::ZERO,
        };
        if !value.base_valid() {
            return None;
        }
        value.harness_digest = digest_harness(&value);
        value.validate().then_some(value)
    }

    fn base_valid(&self) -> bool {
        self.schema_version == HUMANOID_GRASP_SIMULATION_HARNESS_SCHEMA_VERSION
            && valid_id(&self.harness_id)
            && valid_id(&self.simulator_backend_id)
            && !self.simulator_artifact_digest.is_zero()
            && !self.simulator_model_digest.is_zero()
            && !self.simulator_configuration_digest.is_zero()
            && !self.proposal_adapter_artifact_digest.is_zero()
            && !self.proposal_adapter_configuration_digest.is_zero()
            && !self.contact_extractor_artifact_digest.is_zero()
            && !self.contact_extractor_configuration_digest.is_zero()
            && !self.randomization_profile_digest.is_zero()
    }

    pub fn validate(&self) -> bool {
        self.base_valid()
            && !self.harness_digest.is_zero()
            && self.harness_digest == digest_harness(self)
    }

    pub const fn harness_digest(&self) -> HumanoidEvidenceDigest {
        self.harness_digest
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspSimulationCausalFailure {
    InvalidSubject,
    InvalidHarness,
    InvalidProposal,
    ProposalSessionMismatch,
    ProposalAcquisitionMismatch,
    ProposalCandidateMismatch,
    ProposalObjectMismatch,
    ProposalHandMismatch,
    ProposalNotLatestSessionDecision,
    TerminalProposalHasNoPlantStep,
    WrongExecutionPurpose,
    InvalidPreState,
    InvalidPostState,
    InvalidObjectState,
    ProposalStateTimeMismatch,
    NonMonotonicSimulatorTime,
    InvalidObservation,
    ObservationTimestampMismatch,
    ObservationObjectStateMismatch,
    ObservationObjectMismatch,
    ObservationHandMismatch,
    InvalidIntegrationSubsteps,
    InvalidCommitment,
    InvalidResult,
    EmptyTrace,
    TraceDoesNotStartAtFirstProposal,
    TraceDiscontinuity,
    MissingTerminalAbortProposal,
    UnexpectedTerminalProposal,
    SessionClosureMismatch,
    InvalidDigest,
}

/// Pre-step commitment created before a simulation adapter is allowed to execute
/// the semantic proposal. No simulator command or actuator representation exists
/// in this value.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspSimulationStepCommitment {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    execution_purpose: HumanoidExecutionPurpose,
    session_digest: HumanoidEvidenceDigest,
    acquisition_intent_digest: HumanoidEvidenceDigest,
    candidate_digest: HumanoidEvidenceDigest,
    proposal_digest: HumanoidEvidenceDigest,
    previous_proposal_digest: Option<HumanoidEvidenceDigest>,
    proposal_sequence: u64,
    harness_digest: HumanoidEvidenceDigest,
    object_id: String,
    hand: HandSide,
    pre_state_digest: HumanoidEvidenceDigest,
    pre_object_state_digest: HumanoidEvidenceDigest,
    episode_seed: u64,
    step_index: u64,
    proposal_generated_at_s: f64,
    pre_state_timestamp_s: f64,
    commitment_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspSimulationStepCommitment {
    fn validate_internal(&self) -> bool {
        self.schema_version == HUMANOID_GRASP_SIMULATION_STEP_COMMITMENT_SCHEMA_VERSION
            && !self.subject_digest.is_zero()
            && self.execution_purpose == HumanoidExecutionPurpose::SimulationQualification
            && !self.session_digest.is_zero()
            && !self.acquisition_intent_digest.is_zero()
            && !self.candidate_digest.is_zero()
            && !self.proposal_digest.is_zero()
            && self
                .previous_proposal_digest
                .map(|digest| !digest.is_zero())
                .unwrap_or(true)
            && self.proposal_sequence > 0
            && self.proposal_sequence == self.step_index.saturating_add(1)
            && !self.harness_digest.is_zero()
            && valid_id(&self.object_id)
            && !self.pre_state_digest.is_zero()
            && !self.pre_object_state_digest.is_zero()
            && self.proposal_generated_at_s.is_finite()
            && self.proposal_generated_at_s >= 0.0
            && self.pre_state_timestamp_s.is_finite()
            && self.pre_state_timestamp_s >= 0.0
            && self.proposal_generated_at_s.to_bits() == self.pre_state_timestamp_s.to_bits()
            && (self.proposal_sequence != 1 || self.previous_proposal_digest.is_none())
            && (self.proposal_sequence == 1 || self.previous_proposal_digest.is_some())
            && !self.commitment_digest.is_zero()
            && self.commitment_digest == digest_step_commitment(self)
    }

    pub const fn commitment_digest(&self) -> HumanoidEvidenceDigest {
        self.commitment_digest
    }

    pub const fn proposal_digest(&self) -> HumanoidEvidenceDigest {
        self.proposal_digest
    }

    pub const fn step_index(&self) -> u64 {
        self.step_index
    }
}

/// Precommit one exact proposal against one exact simulation state.
///
/// `pre_state.timestamp` must exactly equal the controller decision time. This
/// rejects ambiguous traces where a proposal is paired with another pre-step frame.
#[allow(clippy::too_many_arguments)]
pub fn prepare_humanoid_grasp_simulation_step(
    subject: &HumanoidQualificationSubject,
    acquisition: &HumanoidGraspAcquisitionIntent,
    candidate: &HumanoidGraspSimulationControllerCandidate,
    session: &HumanoidGraspSimulationControllerSession,
    proposal: &HumanoidGraspSimulationControllerProposal,
    harness: &HumanoidGraspSimulationHarnessIdentity,
    pre_state: &HumanoidState,
    pre_object_state_digest: HumanoidEvidenceDigest,
    episode_seed: u64,
    step_index: u64,
) -> Result<HumanoidGraspSimulationStepCommitment, HumanoidGraspSimulationCausalFailure> {
    if !subject.validate() || subject.task != HumanoidTask::Grasp {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidSubject);
    }
    if !harness.validate() {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidHarness);
    }
    if !proposal.validate() {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidProposal);
    }
    if proposal.execution_purpose != HumanoidExecutionPurpose::SimulationQualification
        || session.execution_purpose != HumanoidExecutionPurpose::SimulationQualification
    {
        return Err(HumanoidGraspSimulationCausalFailure::WrongExecutionPurpose);
    }
    if super::digest_subject(subject) != Some(proposal.subject_digest)
        || proposal.subject_digest != session.subject_digest
    {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidSubject);
    }
    if proposal.session_digest != session.session_digest {
        return Err(HumanoidGraspSimulationCausalFailure::ProposalSessionMismatch);
    }
    if proposal.acquisition_intent_digest != acquisition.intent_digest()
        || session.acquisition_intent_digest != acquisition.intent_digest()
    {
        return Err(HumanoidGraspSimulationCausalFailure::ProposalAcquisitionMismatch);
    }
    if proposal.candidate_digest != candidate.candidate_digest()
        || session.candidate_digest != candidate.candidate_digest()
    {
        return Err(HumanoidGraspSimulationCausalFailure::ProposalCandidateMismatch);
    }
    if proposal.object_id.as_str() != acquisition.object_id()
        || session.object_id.as_str() != acquisition.object_id()
    {
        return Err(HumanoidGraspSimulationCausalFailure::ProposalObjectMismatch);
    }
    if proposal.hand != acquisition.hand() || session.hand != acquisition.hand() {
        return Err(HumanoidGraspSimulationCausalFailure::ProposalHandMismatch);
    }
    if session.last_proposal_digest != Some(proposal.proposal_digest)
        || session.steps as u64 != proposal.sequence
    {
        return Err(HumanoidGraspSimulationCausalFailure::ProposalNotLatestSessionDecision);
    }
    if matches!(&proposal.kind, HumanoidGraspSimulationProposalKind::Abort { .. }) {
        return Err(HumanoidGraspSimulationCausalFailure::TerminalProposalHasNoPlantStep);
    }
    if pre_state.validate_for(subject.morphology).is_err() {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidPreState);
    }
    let Some(pre_state_digest) = digest_humanoid_grasp_simulation_state(subject, pre_state) else {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidPreState);
    };
    if pre_object_state_digest.is_zero() {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidObjectState);
    }
    if proposal.generated_at_s.to_bits() != pre_state.timestamp.to_bits() {
        return Err(HumanoidGraspSimulationCausalFailure::ProposalStateTimeMismatch);
    }
    if proposal.sequence != step_index.saturating_add(1) {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidProposal);
    }
    if proposal.sequence == 1 && pre_object_state_digest != acquisition.object_state_digest() {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidObjectState);
    }

    let mut value = HumanoidGraspSimulationStepCommitment {
        schema_version: HUMANOID_GRASP_SIMULATION_STEP_COMMITMENT_SCHEMA_VERSION,
        subject_digest: proposal.subject_digest,
        execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
        session_digest: proposal.session_digest,
        acquisition_intent_digest: proposal.acquisition_intent_digest,
        candidate_digest: proposal.candidate_digest,
        proposal_digest: proposal.proposal_digest,
        previous_proposal_digest: proposal.previous_proposal_digest,
        proposal_sequence: proposal.sequence,
        harness_digest: harness.harness_digest(),
        object_id: proposal.object_id.clone(),
        hand: proposal.hand,
        pre_state_digest,
        pre_object_state_digest,
        episode_seed,
        step_index,
        proposal_generated_at_s: proposal.generated_at_s,
        pre_state_timestamp_s: pre_state.timestamp,
        commitment_digest: HumanoidEvidenceDigest::ZERO,
    };
    value.commitment_digest = digest_step_commitment(&value);
    value
        .validate_internal()
        .then_some(value)
        .ok_or(HumanoidGraspSimulationCausalFailure::InvalidDigest)
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspSimulationStepResult {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    execution_purpose: HumanoidExecutionPurpose,
    session_digest: HumanoidEvidenceDigest,
    acquisition_intent_digest: HumanoidEvidenceDigest,
    candidate_digest: HumanoidEvidenceDigest,
    proposal_digest: HumanoidEvidenceDigest,
    previous_proposal_digest: Option<HumanoidEvidenceDigest>,
    proposal_sequence: u64,
    harness_digest: HumanoidEvidenceDigest,
    commitment_digest: HumanoidEvidenceDigest,
    object_id: String,
    hand: HandSide,
    pre_state_digest: HumanoidEvidenceDigest,
    post_state_digest: HumanoidEvidenceDigest,
    pre_object_state_digest: HumanoidEvidenceDigest,
    post_object_state_digest: HumanoidEvidenceDigest,
    observation_digest: HumanoidEvidenceDigest,
    episode_seed: u64,
    step_index: u64,
    pre_state_timestamp_s: f64,
    post_state_timestamp_s: f64,
    integration_substeps: usize,
    result_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspSimulationStepResult {
    fn validate_internal(&self) -> bool {
        self.schema_version == HUMANOID_GRASP_SIMULATION_STEP_RESULT_SCHEMA_VERSION
            && !self.subject_digest.is_zero()
            && self.execution_purpose == HumanoidExecutionPurpose::SimulationQualification
            && !self.session_digest.is_zero()
            && !self.acquisition_intent_digest.is_zero()
            && !self.candidate_digest.is_zero()
            && !self.proposal_digest.is_zero()
            && self.proposal_sequence > 0
            && self.proposal_sequence == self.step_index.saturating_add(1)
            && !self.harness_digest.is_zero()
            && !self.commitment_digest.is_zero()
            && valid_id(&self.object_id)
            && !self.pre_state_digest.is_zero()
            && !self.post_state_digest.is_zero()
            && !self.pre_object_state_digest.is_zero()
            && !self.post_object_state_digest.is_zero()
            && !self.observation_digest.is_zero()
            && self.pre_state_timestamp_s.is_finite()
            && self.pre_state_timestamp_s >= 0.0
            && self.post_state_timestamp_s.is_finite()
            && self.post_state_timestamp_s > self.pre_state_timestamp_s
            && self.integration_substeps > 0
            && self.integration_substeps <= MAX_INTEGRATION_SUBSTEPS
            && !self.result_digest.is_zero()
            && self.result_digest == digest_step_result(self)
    }

    pub const fn result_digest(&self) -> HumanoidEvidenceDigest {
        self.result_digest
    }

    pub const fn observation_digest(&self) -> HumanoidEvidenceDigest {
        self.observation_digest
    }
}

/// Bind the post-step state and same-frame canonical contact observation to an
/// already-created pre-step commitment.
#[allow(clippy::too_many_arguments)]
pub fn bind_humanoid_grasp_simulation_step_result(
    subject: &HumanoidQualificationSubject,
    harness: &HumanoidGraspSimulationHarnessIdentity,
    commitment: &HumanoidGraspSimulationStepCommitment,
    pre_state: &HumanoidState,
    post_state: &HumanoidState,
    post_object_state_digest: HumanoidEvidenceDigest,
    observation: &HumanoidGraspContactObservation,
    integration_substeps: usize,
) -> Result<HumanoidGraspSimulationStepResult, HumanoidGraspSimulationCausalFailure> {
    if !subject.validate() || subject.task != HumanoidTask::Grasp {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidSubject);
    }
    if !harness.validate() || commitment.harness_digest != harness.harness_digest() {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidHarness);
    }
    if !commitment.validate_internal() {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidCommitment);
    }
    if super::digest_subject(subject) != Some(commitment.subject_digest) {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidSubject);
    }
    if pre_state.validate_for(subject.morphology).is_err()
        || digest_humanoid_grasp_simulation_state(subject, pre_state)
            != Some(commitment.pre_state_digest)
        || pre_state.timestamp.to_bits() != commitment.pre_state_timestamp_s.to_bits()
    {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidPreState);
    }
    if post_state.validate_for(subject.morphology).is_err() {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidPostState);
    }
    if post_state.timestamp <= pre_state.timestamp {
        return Err(HumanoidGraspSimulationCausalFailure::NonMonotonicSimulatorTime);
    }
    let Some(post_state_digest) = digest_humanoid_grasp_simulation_state(subject, post_state) else {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidPostState);
    };
    if post_object_state_digest.is_zero() {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidObjectState);
    }
    if integration_substeps == 0 || integration_substeps > MAX_INTEGRATION_SUBSTEPS {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidIntegrationSubsteps);
    }
    if !observation.validate_for(subject) {
        return Err(HumanoidGraspSimulationCausalFailure::InvalidObservation);
    }
    if observation.timestamp_s().to_bits() != post_state.timestamp.to_bits() {
        return Err(HumanoidGraspSimulationCausalFailure::ObservationTimestampMismatch);
    }
    if observation.object_state_digest() != post_object_state_digest {
        return Err(HumanoidGraspSimulationCausalFailure::ObservationObjectStateMismatch);
    }
    if observation.object_id() != commitment.object_id.as_str() {
        return Err(HumanoidGraspSimulationCausalFailure::ObservationObjectMismatch);
    }
    if observation.hand() != commitment.hand {
        return Err(HumanoidGraspSimulationCausalFailure::ObservationHandMismatch);
    }

    let mut value = HumanoidGraspSimulationStepResult {
        schema_version: HUMANOID_GRASP_SIMULATION_STEP_RESULT_SCHEMA_VERSION,
        subject_digest: commitment.subject_digest,
        execution_purpose: commitment.execution_purpose,
        session_digest: commitment.session_digest,
        acquisition_intent_digest: commitment.acquisition_intent_digest,
        candidate_digest: commitment.candidate_digest,
        proposal_digest: commitment.proposal_digest,
        previous_proposal_digest: commitment.previous_proposal_digest,
        proposal_sequence: commitment.proposal_sequence,
        harness_digest: commitment.harness_digest,
        commitment_digest: commitment.commitment_digest,
        object_id: commitment.object_id.clone(),
        hand: commitment.hand,
        pre_state_digest: commitment.pre_state_digest,
        post_state_digest,
        pre_object_state_digest: commitment.pre_object_state_digest,
        post_object_state_digest,
        observation_digest: observation.observation_digest(),
        episode_seed: commitment.episode_seed,
        step_index: commitment.step_index,
        pre_state_timestamp_s: commitment.pre_state_timestamp_s,
        post_state_timestamp_s: post_state.timestamp,
        integration_substeps,
        result_digest: HumanoidEvidenceDigest::ZERO,
    };
    value.result_digest = digest_step_result(&value);
    value
        .validate_internal()
        .then_some(value)
        .ok_or(HumanoidGraspSimulationCausalFailure::InvalidResult)
}

/// Complete, sealed closed-loop simulation lineage for one controller session.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspSimulationCausalTrace {
    schema_version: u32,
    subject_digest: HumanoidEvidenceDigest,
    execution_purpose: HumanoidExecutionPurpose,
    session_digest: HumanoidEvidenceDigest,
    acquisition_intent_digest: HumanoidEvidenceDigest,
    candidate_digest: HumanoidEvidenceDigest,
    harness_digest: HumanoidEvidenceDigest,
    object_id: String,
    hand: HandSide,
    episode_seed: u64,
    closed_session_steps: usize,
    closure_kind: HumanoidGraspSimulationTraceClosureKind,
    terminal_proposal_digest: Option<HumanoidEvidenceDigest>,
    steps: Vec<HumanoidGraspSimulationStepResult>,
    trace_digest: HumanoidEvidenceDigest,
}

/// Consume the controller session and seal the complete transition trace.
///
/// A non-terminal session must have exactly one plant result for every proposal.
/// A terminal session must supply its final `Abort` proposal separately; that
/// proposal must follow the final plant transition but is not itself executed.
pub fn seal_humanoid_grasp_simulation_causal_trace(
    session: HumanoidGraspSimulationControllerSession,
    terminal_abort: Option<HumanoidGraspSimulationControllerProposal>,
    results: Vec<HumanoidGraspSimulationStepResult>,
) -> Result<HumanoidGraspSimulationCausalTrace, HumanoidGraspSimulationCausalFailure> {
    if results.is_empty() {
        return Err(HumanoidGraspSimulationCausalFailure::EmptyTrace);
    }
    if !valid_result_chain(&results) {
        return Err(HumanoidGraspSimulationCausalFailure::TraceDiscontinuity);
    }
    let first = &results[0];
    let last = results.last().expect("nonempty checked");
    if first.step_index != 0
        || first.proposal_sequence != 1
        || first.previous_proposal_digest.is_some()
    {
        return Err(HumanoidGraspSimulationCausalFailure::TraceDoesNotStartAtFirstProposal);
    }
    if session.execution_purpose != HumanoidExecutionPurpose::SimulationQualification
        || session.subject_digest != first.subject_digest
        || session.session_digest != first.session_digest
        || session.acquisition_intent_digest != first.acquisition_intent_digest
        || session.candidate_digest != first.candidate_digest
        || session.object_id != first.object_id
        || session.hand != first.hand
    {
        return Err(HumanoidGraspSimulationCausalFailure::SessionClosureMismatch);
    }

    let (closure_kind, terminal_proposal_digest) = if session.terminal {
        let Some(abort) = terminal_abort else {
            return Err(HumanoidGraspSimulationCausalFailure::MissingTerminalAbortProposal);
        };
        if !abort.validate()
            || !matches!(&abort.kind, HumanoidGraspSimulationProposalKind::Abort { .. })
            || abort.execution_purpose != HumanoidExecutionPurpose::SimulationQualification
            || abort.subject_digest != session.subject_digest
            || abort.session_digest != session.session_digest
            || abort.acquisition_intent_digest != session.acquisition_intent_digest
            || abort.candidate_digest != session.candidate_digest
            || abort.object_id != session.object_id
            || abort.hand != session.hand
            || session.last_proposal_digest != Some(abort.proposal_digest)
            || session.steps != results.len().saturating_add(1)
            || abort.sequence != session.steps as u64
            || abort.previous_proposal_digest != Some(last.proposal_digest)
            || abort.sequence != last.proposal_sequence.saturating_add(1)
        {
            return Err(HumanoidGraspSimulationCausalFailure::SessionClosureMismatch);
        }
        (
            HumanoidGraspSimulationTraceClosureKind::ControllerAborted,
            Some(abort.proposal_digest),
        )
    } else {
        if terminal_abort.is_some() {
            return Err(HumanoidGraspSimulationCausalFailure::UnexpectedTerminalProposal);
        }
        if session.steps != results.len()
            || session.last_proposal_digest != Some(last.proposal_digest)
            || session.steps as u64 != last.proposal_sequence
        {
            return Err(HumanoidGraspSimulationCausalFailure::SessionClosureMismatch);
        }
        (
            HumanoidGraspSimulationTraceClosureKind::QualificationWindowClosed,
            None,
        )
    };

    let mut value = HumanoidGraspSimulationCausalTrace {
        schema_version: HUMANOID_GRASP_SIMULATION_CAUSAL_TRACE_SCHEMA_VERSION,
        subject_digest: first.subject_digest,
        execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
        session_digest: first.session_digest,
        acquisition_intent_digest: first.acquisition_intent_digest,
        candidate_digest: first.candidate_digest,
        harness_digest: first.harness_digest,
        object_id: first.object_id.clone(),
        hand: first.hand,
        episode_seed: first.episode_seed,
        closed_session_steps: session.steps,
        closure_kind,
        terminal_proposal_digest,
        steps: results,
        trace_digest: HumanoidEvidenceDigest::ZERO,
    };
    value.trace_digest = digest_causal_trace(&value);
    value
        .validate()
        .then_some(value)
        .ok_or(HumanoidGraspSimulationCausalFailure::InvalidDigest)
}

impl HumanoidGraspSimulationCausalTrace {
    pub fn validate(&self) -> bool {
        if self.schema_version != HUMANOID_GRASP_SIMULATION_CAUSAL_TRACE_SCHEMA_VERSION
            || self.execution_purpose != HumanoidExecutionPurpose::SimulationQualification
            || self.subject_digest.is_zero()
            || self.session_digest.is_zero()
            || self.acquisition_intent_digest.is_zero()
            || self.candidate_digest.is_zero()
            || self.harness_digest.is_zero()
            || !valid_id(&self.object_id)
            || self.steps.is_empty()
            || self.trace_digest.is_zero()
            || !valid_result_chain(&self.steps)
        {
            return false;
        }
        let first = &self.steps[0];
        if first.step_index != 0
            || first.proposal_sequence != 1
            || first.previous_proposal_digest.is_some()
            || self.steps.iter().any(|result| {
                result.subject_digest != self.subject_digest
                    || result.session_digest != self.session_digest
                    || result.acquisition_intent_digest != self.acquisition_intent_digest
                    || result.candidate_digest != self.candidate_digest
                    || result.harness_digest != self.harness_digest
                    || result.object_id != self.object_id
                    || result.hand != self.hand
                    || result.episode_seed != self.episode_seed
            })
        {
            return false;
        }
        match self.closure_kind {
            HumanoidGraspSimulationTraceClosureKind::QualificationWindowClosed => {
                if self.terminal_proposal_digest.is_some()
                    || self.closed_session_steps != self.steps.len()
                {
                    return false;
                }
            }
            HumanoidGraspSimulationTraceClosureKind::ControllerAborted => {
                if self
                    .terminal_proposal_digest
                    .map(|digest| digest.is_zero())
                    .unwrap_or(true)
                    || self.closed_session_steps != self.steps.len().saturating_add(1)
                {
                    return false;
                }
            }
        }
        self.trace_digest == digest_causal_trace(self)
    }

    pub const fn trace_digest(&self) -> HumanoidEvidenceDigest {
        self.trace_digest
    }

    pub const fn closure_kind(&self) -> HumanoidGraspSimulationTraceClosureKind {
        self.closure_kind
    }

    pub const fn terminal_proposal_digest(&self) -> Option<HumanoidEvidenceDigest> {
        self.terminal_proposal_digest
    }

    pub fn steps(&self) -> &[HumanoidGraspSimulationStepResult] {
        &self.steps
    }

    pub fn observation_digests(&self) -> Vec<HumanoidEvidenceDigest> {
        self.steps.iter().map(|step| step.observation_digest).collect()
    }
}

fn valid_result_chain(results: &[HumanoidGraspSimulationStepResult]) -> bool {
    if results.is_empty() || results.iter().any(|result| !result.validate_internal()) {
        return false;
    }
    let first = &results[0];
    if results.iter().any(|result| {
        result.subject_digest != first.subject_digest
            || result.execution_purpose != HumanoidExecutionPurpose::SimulationQualification
            || result.session_digest != first.session_digest
            || result.acquisition_intent_digest != first.acquisition_intent_digest
            || result.candidate_digest != first.candidate_digest
            || result.harness_digest != first.harness_digest
            || result.object_id != first.object_id
            || result.hand != first.hand
            || result.episode_seed != first.episode_seed
    }) {
        return false;
    }
    !results.windows(2).any(|pair| {
        let previous = &pair[0];
        let next = &pair[1];
        next.step_index != previous.step_index.saturating_add(1)
            || next.proposal_sequence != previous.proposal_sequence.saturating_add(1)
            || next.previous_proposal_digest != Some(previous.proposal_digest)
            || next.pre_state_digest != previous.post_state_digest
            || next.pre_object_state_digest != previous.post_object_state_digest
            || next.pre_state_timestamp_s.to_bits() != previous.post_state_timestamp_s.to_bits()
    })
}

/// Canonical privileged simulation-state identity for causal Grasp evidence.
/// Unlike policy observation channels, this intentionally includes world root
/// position and simulation timestamp because exact transition replay needs them.
pub fn digest_humanoid_grasp_simulation_state(
    subject: &HumanoidQualificationSubject,
    state: &HumanoidState,
) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate()
        || subject.task != HumanoidTask::Grasp
        || state.validate_for(subject.morphology).is_err()
    {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-simulation-causal-state.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id)
        .f64(state.root_height);
    hash_f64_slice(&mut h, &state.root_position);
    hash_f64_slice(&mut h, &state.root_quaternion);
    hash_f64_slice(&mut h, &state.joint_angles);
    hash_f64_slice(&mut h, &state.root_linear_velocity);
    hash_f64_slice(&mut h, &state.root_angular_velocity);
    hash_f64_slice(&mut h, &state.joint_velocities);
    h.f64(state.head_height);
    hash_f64_slice(&mut h, &state.torso_vertical);
    hash_f64_slice(&mut h, &state.extremities);
    hash_f64_slice(&mut h, &state.com_velocity);
    h.f64(state.timestamp);
    Some(h.finish())
}

fn digest_harness(value: &HumanoidGraspSimulationHarnessIdentity) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-simulation-harness.v1");
    h.u32(value.schema_version)
        .string(&value.harness_id)
        .string(&value.simulator_backend_id)
        .digest(value.simulator_artifact_digest)
        .digest(value.simulator_model_digest)
        .digest(value.simulator_configuration_digest)
        .digest(value.proposal_adapter_artifact_digest)
        .digest(value.proposal_adapter_configuration_digest)
        .digest(value.contact_extractor_artifact_digest)
        .digest(value.contact_extractor_configuration_digest)
        .digest(value.randomization_profile_digest)
        .u64(determinism_mode_id(value.determinism_mode));
    h.finish()
}

fn digest_step_commitment(value: &HumanoidGraspSimulationStepCommitment) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-simulation-step-commitment.v1");
    h.u32(value.schema_version)
        .digest(value.subject_digest)
        .u64(execution_purpose_id(value.execution_purpose))
        .digest(value.session_digest)
        .digest(value.acquisition_intent_digest)
        .digest(value.candidate_digest)
        .digest(value.proposal_digest)
        .bool(value.previous_proposal_digest.is_some());
    if let Some(digest) = value.previous_proposal_digest {
        h.digest(digest);
    }
    h.u64(value.proposal_sequence)
        .digest(value.harness_digest)
        .string(&value.object_id)
        .u64(hand_id(value.hand))
        .digest(value.pre_state_digest)
        .digest(value.pre_object_state_digest)
        .u64(value.episode_seed)
        .u64(value.step_index)
        .f64(value.proposal_generated_at_s)
        .f64(value.pre_state_timestamp_s);
    h.finish()
}

fn digest_step_result(value: &HumanoidGraspSimulationStepResult) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-simulation-step-result.v1");
    h.u32(value.schema_version)
        .digest(value.subject_digest)
        .u64(execution_purpose_id(value.execution_purpose))
        .digest(value.session_digest)
        .digest(value.acquisition_intent_digest)
        .digest(value.candidate_digest)
        .digest(value.proposal_digest)
        .bool(value.previous_proposal_digest.is_some());
    if let Some(digest) = value.previous_proposal_digest {
        h.digest(digest);
    }
    h.u64(value.proposal_sequence)
        .digest(value.harness_digest)
        .digest(value.commitment_digest)
        .string(&value.object_id)
        .u64(hand_id(value.hand))
        .digest(value.pre_state_digest)
        .digest(value.post_state_digest)
        .digest(value.pre_object_state_digest)
        .digest(value.post_object_state_digest)
        .digest(value.observation_digest)
        .u64(value.episode_seed)
        .u64(value.step_index)
        .f64(value.pre_state_timestamp_s)
        .f64(value.post_state_timestamp_s)
        .usize(value.integration_substeps);
    h.finish()
}

fn digest_causal_trace(value: &HumanoidGraspSimulationCausalTrace) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-simulation-causal-trace.v2");
    h.u32(value.schema_version)
        .digest(value.subject_digest)
        .u64(execution_purpose_id(value.execution_purpose))
        .digest(value.session_digest)
        .digest(value.acquisition_intent_digest)
        .digest(value.candidate_digest)
        .digest(value.harness_digest)
        .string(&value.object_id)
        .u64(hand_id(value.hand))
        .u64(value.episode_seed)
        .usize(value.closed_session_steps)
        .u64(closure_kind_id(value.closure_kind))
        .bool(value.terminal_proposal_digest.is_some());
    if let Some(digest) = value.terminal_proposal_digest {
        h.digest(digest);
    }
    h.usize(value.steps.len());
    for step in &value.steps {
        h.digest(step.result_digest);
    }
    h.finish()
}

fn hash_f64_slice(h: &mut HumanoidEvidenceHasher, values: &[f64]) {
    h.usize(values.len());
    for value in values {
        h.f64(*value);
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

fn determinism_mode_id(mode: HumanoidGraspSimulationDeterminismMode) -> u64 {
    match mode {
        HumanoidGraspSimulationDeterminismMode::DeterministicReplay => 1,
        HumanoidGraspSimulationDeterminismMode::SeededStochastic => 2,
    }
}

fn closure_kind_id(kind: HumanoidGraspSimulationTraceClosureKind) -> u64 {
    match kind {
        HumanoidGraspSimulationTraceClosureKind::QualificationWindowClosed => 1,
        HumanoidGraspSimulationTraceClosureKind::ControllerAborted => 2,
    }
}

fn execution_purpose_id(purpose: HumanoidExecutionPurpose) -> u64 {
    match purpose {
        HumanoidExecutionPurpose::SimulationQualification => 1,
        HumanoidExecutionPurpose::HilQualification => 2,
        HumanoidExecutionPurpose::PhysicalQualification => 3,
        HumanoidExecutionPurpose::Operational => 4,
    }
}

fn hand_id(hand: HandSide) -> u64 {
    match hand {
        HandSide::Right => 1,
        HandSide::Left => 2,
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
    use crate::grasp_contact_evidence::HumanoidManipulationContactSource;
    use crate::morphology::HumanoidMorphology;

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dmc21,
            HumanoidTask::Grasp,
            ActuationMode::NormalizedTorque,
            "causal-trace-test-backend",
        )
    }

    fn harness() -> HumanoidGraspSimulationHarnessIdentity {
        HumanoidGraspSimulationHarnessIdentity::new(
            "grasp-harness-v1",
            "sim-backend-v1",
            HumanoidEvidenceDigest::from_bytes([1; 32]),
            HumanoidEvidenceDigest::from_bytes([2; 32]),
            HumanoidEvidenceDigest::from_bytes([3; 32]),
            HumanoidEvidenceDigest::from_bytes([4; 32]),
            HumanoidEvidenceDigest::from_bytes([5; 32]),
            HumanoidEvidenceDigest::from_bytes([6; 32]),
            HumanoidEvidenceDigest::from_bytes([7; 32]),
            HumanoidEvidenceDigest::from_bytes([8; 32]),
            HumanoidGraspSimulationDeterminismMode::DeterministicReplay,
        )
        .unwrap()
    }

    fn state(timestamp: f64, x: f64) -> HumanoidState {
        let mut state = HumanoidState::default_for(HumanoidMorphology::Dmc21);
        state.root_position[0] = x;
        state.timestamp = timestamp;
        state
    }

    fn synthetic_session(
        steps: usize,
        last_proposal_digest: Option<HumanoidEvidenceDigest>,
        terminal: bool,
    ) -> HumanoidGraspSimulationControllerSession {
        let mut session = HumanoidGraspSimulationControllerSession {
            schema_version: super::super::HUMANOID_GRASP_SIMULATION_SESSION_SCHEMA_VERSION,
            subject_digest: super::super::digest_subject(&subject()).unwrap(),
            acquisition_intent_digest: HumanoidEvidenceDigest::from_bytes([11; 32]),
            candidate_digest: HumanoidEvidenceDigest::from_bytes([12; 32]),
            controller_policy_digest: HumanoidEvidenceDigest::from_bytes([13; 32]),
            execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
            object_id: "object-a".into(),
            hand: HandSide::Right,
            target_root_m: [0.2, 0.0, 1.0],
            acquisition_valid_until_s: 10.0,
            started_at_s: 1.0,
            last_step_at_s: 1.0,
            steps,
            last_proposal_digest,
            terminal,
            session_digest: HumanoidEvidenceDigest::ZERO,
        };
        session.session_digest = super::super::digest_session(&session);
        session
    }

    fn session_digest() -> HumanoidEvidenceDigest {
        synthetic_session(0, None, false).session_digest
    }

    fn synthetic_commitment(
        step_index: u64,
        proposal_digest: HumanoidEvidenceDigest,
        previous_proposal_digest: Option<HumanoidEvidenceDigest>,
        pre_state: &HumanoidState,
        pre_object_state_digest: HumanoidEvidenceDigest,
    ) -> HumanoidGraspSimulationStepCommitment {
        let mut value = HumanoidGraspSimulationStepCommitment {
            schema_version: HUMANOID_GRASP_SIMULATION_STEP_COMMITMENT_SCHEMA_VERSION,
            subject_digest: super::super::digest_subject(&subject()).unwrap(),
            execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
            session_digest: session_digest(),
            acquisition_intent_digest: HumanoidEvidenceDigest::from_bytes([11; 32]),
            candidate_digest: HumanoidEvidenceDigest::from_bytes([12; 32]),
            proposal_digest,
            previous_proposal_digest,
            proposal_sequence: step_index + 1,
            harness_digest: harness().harness_digest(),
            object_id: "object-a".into(),
            hand: HandSide::Right,
            pre_state_digest: digest_humanoid_grasp_simulation_state(&subject(), pre_state).unwrap(),
            pre_object_state_digest,
            episode_seed: 42,
            step_index,
            proposal_generated_at_s: pre_state.timestamp,
            pre_state_timestamp_s: pre_state.timestamp,
            commitment_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.commitment_digest = digest_step_commitment(&value);
        assert!(value.validate_internal());
        value
    }

    fn observation(
        timestamp: f64,
        object_state_digest: HumanoidEvidenceDigest,
    ) -> HumanoidGraspContactObservation {
        HumanoidGraspContactObservation::new(
            &subject(),
            "object-a",
            object_state_digest,
            HandSide::Right,
            HumanoidContactSite::RightHand,
            true,
            [0.2, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-10.0, 0.1, 0.0],
            [0.0; 3],
            [0.0; 3],
            0.99,
            HumanoidManipulationContactSource::SolverWrench,
            timestamp,
        )
        .unwrap()
    }

    fn synthetic_abort(
        sequence: u64,
        previous: HumanoidEvidenceDigest,
    ) -> HumanoidGraspSimulationControllerProposal {
        let session = synthetic_session(0, None, false);
        let mut proposal = HumanoidGraspSimulationControllerProposal {
            schema_version: super::super::HUMANOID_GRASP_SIMULATION_PROPOSAL_SCHEMA_VERSION,
            subject_digest: session.subject_digest,
            session_digest: session.session_digest,
            execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
            sequence,
            previous_proposal_digest: Some(previous),
            acquisition_intent_digest: session.acquisition_intent_digest,
            candidate_digest: session.candidate_digest,
            object_id: session.object_id,
            hand: session.hand,
            feedback_observation_digest: Some(HumanoidEvidenceDigest::from_bytes([60; 32])),
            feedback_assessment_digest: Some(HumanoidEvidenceDigest::from_bytes([61; 32])),
            feedback_object_state_digest: Some(HumanoidEvidenceDigest::from_bytes([62; 32])),
            generated_at_s: 1.02,
            kind: HumanoidGraspSimulationProposalKind::Abort {
                reason: super::super::HumanoidGraspSimulationAbortReason::UnsafeMeasuredContact,
            },
            proposal_digest: HumanoidEvidenceDigest::ZERO,
        };
        proposal.proposal_digest = super::super::digest_proposal(&proposal);
        assert!(proposal.validate());
        proposal
    }

    fn one_result() -> (HumanoidGraspSimulationStepResult, HumanoidEvidenceDigest) {
        let pre = state(1.0, 0.0);
        let post = state(1.01, 0.01);
        let object0 = HumanoidEvidenceDigest::from_bytes([20; 32]);
        let object1 = HumanoidEvidenceDigest::from_bytes([21; 32]);
        let proposal0 = HumanoidEvidenceDigest::from_bytes([30; 32]);
        let commitment = synthetic_commitment(0, proposal0, None, &pre, object0);
        let result = bind_humanoid_grasp_simulation_step_result(
            &subject(),
            &harness(),
            &commitment,
            &pre,
            &post,
            object1,
            &observation(1.01, object1),
            1,
        )
        .unwrap();
        (result, proposal0)
    }

    #[test]
    fn harness_identity_changes_when_adapter_configuration_changes() {
        let a = harness();
        let b = HumanoidGraspSimulationHarnessIdentity::new(
            "grasp-harness-v1",
            "sim-backend-v1",
            HumanoidEvidenceDigest::from_bytes([1; 32]),
            HumanoidEvidenceDigest::from_bytes([2; 32]),
            HumanoidEvidenceDigest::from_bytes([3; 32]),
            HumanoidEvidenceDigest::from_bytes([4; 32]),
            HumanoidEvidenceDigest::from_bytes([55; 32]),
            HumanoidEvidenceDigest::from_bytes([6; 32]),
            HumanoidEvidenceDigest::from_bytes([7; 32]),
            HumanoidEvidenceDigest::from_bytes([8; 32]),
            HumanoidGraspSimulationDeterminismMode::DeterministicReplay,
        )
        .unwrap();
        assert_ne!(a.harness_digest(), b.harness_digest());
    }

    #[test]
    fn privileged_state_digest_binds_world_position_and_time() {
        let a = state(1.0, 0.0);
        let b = state(1.0, 0.1);
        let c = state(1.1, 0.0);
        assert_ne!(
            digest_humanoid_grasp_simulation_state(&subject(), &a),
            digest_humanoid_grasp_simulation_state(&subject(), &b)
        );
        assert_ne!(
            digest_humanoid_grasp_simulation_state(&subject(), &a),
            digest_humanoid_grasp_simulation_state(&subject(), &c)
        );
    }

    #[test]
    fn result_requires_observation_from_exact_post_state_frame() {
        let pre = state(1.0, 0.0);
        let post = state(1.01, 0.01);
        let pre_object = HumanoidEvidenceDigest::from_bytes([20; 32]);
        let post_object = HumanoidEvidenceDigest::from_bytes([21; 32]);
        let commitment = synthetic_commitment(
            0,
            HumanoidEvidenceDigest::from_bytes([30; 32]),
            None,
            &pre,
            pre_object,
        );
        let wrong_time = observation(1.02, post_object);
        assert_eq!(
            bind_humanoid_grasp_simulation_step_result(
                &subject(),
                &harness(),
                &commitment,
                &pre,
                &post,
                post_object,
                &wrong_time,
                1,
            ),
            Err(HumanoidGraspSimulationCausalFailure::ObservationTimestampMismatch)
        );
    }

    #[test]
    fn sealed_trace_rejects_tail_omission() {
        let (result, proposal0) = one_result();
        let session = synthetic_session(2, Some(proposal0), false);
        assert_eq!(
            seal_humanoid_grasp_simulation_causal_trace(session, None, vec![result]),
            Err(HumanoidGraspSimulationCausalFailure::SessionClosureMismatch)
        );
    }

    #[test]
    fn complete_trace_rejects_state_chain_discontinuity_even_with_rehashed_result() {
        let pre0 = state(1.0, 0.0);
        let post0 = state(1.01, 0.01);
        let object0 = HumanoidEvidenceDigest::from_bytes([20; 32]);
        let object1 = HumanoidEvidenceDigest::from_bytes([21; 32]);
        let proposal0 = HumanoidEvidenceDigest::from_bytes([30; 32]);
        let commitment0 = synthetic_commitment(0, proposal0, None, &pre0, object0);
        let result0 = bind_humanoid_grasp_simulation_step_result(
            &subject(), &harness(), &commitment0, &pre0, &post0, object1,
            &observation(1.01, object1), 2,
        )
        .unwrap();

        let post1 = state(1.02, 0.02);
        let object2 = HumanoidEvidenceDigest::from_bytes([22; 32]);
        let proposal1 = HumanoidEvidenceDigest::from_bytes([31; 32]);
        let commitment1 = synthetic_commitment(1, proposal1, Some(proposal0), &post0, object1);
        let mut result1 = bind_humanoid_grasp_simulation_step_result(
            &subject(), &harness(), &commitment1, &post0, &post1, object2,
            &observation(1.02, object2), 2,
        )
        .unwrap();
        result1.pre_state_digest = HumanoidEvidenceDigest::from_bytes([99; 32]);
        result1.result_digest = digest_step_result(&result1);
        assert!(result1.validate_internal());
        let session = synthetic_session(2, Some(proposal1), false);
        assert_eq!(
            seal_humanoid_grasp_simulation_causal_trace(session, None, vec![result0, result1]),
            Err(HumanoidGraspSimulationCausalFailure::TraceDiscontinuity)
        );
    }

    #[test]
    fn nonterminal_session_is_consumed_into_closed_trace() {
        let (result, proposal0) = one_result();
        let session = synthetic_session(1, Some(proposal0), false);
        let trace = seal_humanoid_grasp_simulation_causal_trace(session, None, vec![result]).unwrap();
        assert!(trace.validate());
        assert_eq!(
            trace.closure_kind(),
            HumanoidGraspSimulationTraceClosureKind::QualificationWindowClosed
        );
        assert_eq!(trace.steps().len(), 1);
        assert_eq!(trace.observation_digests().len(), 1);
    }

    #[test]
    fn abort_is_sealed_as_terminal_decision_not_plant_step() {
        let (result, proposal0) = one_result();
        let abort = synthetic_abort(2, proposal0);
        let abort_digest = abort.proposal_digest;
        let session = synthetic_session(2, Some(abort_digest), true);
        let trace = seal_humanoid_grasp_simulation_causal_trace(
            session,
            Some(abort),
            vec![result],
        )
        .unwrap();
        assert!(trace.validate());
        assert_eq!(
            trace.closure_kind(),
            HumanoidGraspSimulationTraceClosureKind::ControllerAborted
        );
        assert_eq!(trace.terminal_proposal_digest(), Some(abort_digest));
        assert_eq!(trace.steps().len(), 1);
    }
}
