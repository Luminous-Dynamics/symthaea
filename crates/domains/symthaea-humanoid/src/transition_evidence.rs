// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causally aligned humanoid transition evidence for future HDC/LTC dynamics work.
//!
//! R3.13 is deliberately *not* a learner. It binds the ingredients that a
//! predictive model would otherwise be tempted to assemble implicitly:
//!
//! `semantic state at t + exact backend-applied action + backend-applied dt
//!  + semantic state at t+dt`.
//!
//! State source, clock domain, and episode identity remain explicit metadata.
//! A semantic state produced from privileged simulator truth is therefore not
//! silently interchangeable with a delayed/noisy policy observation merely
//! because both use the same physical address vocabulary.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::Write as _;

use symthaea_core::hdc::sensorimotor_contingencies::{
    MissingObservationReasonV1, SensorimotorObservationV1,
};

use crate::backend_action_evidence::{
    BackendAppliedActionEvidenceErrorV1, BackendAppliedActionEvidenceProvider,
    BackendAppliedActionSnapshotV1, InstrumentedSimpleHumanoidSimulator,
};
use crate::morphology::HumanoidMorphology;
use crate::semantic_state::{
    SemanticHumanoidEncoderV1, SemanticHumanoidErrorV1, SemanticHumanoidFrameV1,
};
use crate::simulator::HumanoidPhysicsSimulator;
use crate::types::{ActuationMode, CommandValidationError, HumanoidCommand};

const TRANSITION_EVIDENCE_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.transition-evidence.v1\0";

/// Provenance class of the semantic state entering a transition record.
///
/// This is intentionally not encoded into each physical observation address.
/// The same physical role can be observed through different epistemic paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidTransitionStateSourceV1 {
    /// Privileged backend/simulator state used for scientific ground truth.
    PrivilegedBackendTruth,
    /// The state frame actually exposed through the policy observation path.
    PolicyObservation,
}

impl HumanoidTransitionStateSourceV1 {
    const fn token(self) -> &'static str {
        match self {
            Self::PrivilegedBackendTruth => "privileged_backend_truth",
            Self::PolicyObservation => "policy_observation",
        }
    }
}

/// One versioned, content-addressed causal transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidTransitionEvidenceV1 {
    pub schema_id: String,
    pub morphology: HumanoidMorphology,
    /// Identity of the clock whose timestamps are carried by both semantic
    /// frames. Same units alone are insufficient to establish comparability.
    pub clock_domain_id: String,
    /// Reset/run identity. Timestamps may restart at zero across episodes.
    pub episode_id: String,
    pub pre_state_source: HumanoidTransitionStateSourceV1,
    pub post_state_source: HumanoidTransitionStateSourceV1,
    pub pre_state: SemanticHumanoidFrameV1,
    pub backend_action: BackendAppliedActionSnapshotV1,
    pub post_state: SemanticHumanoidFrameV1,
    /// Domain-separated commitment over state semantics, values, causal action
    /// evidence, timing identities, and state-source labels.
    pub transition_digest_hex: String,
}

impl HumanoidTransitionEvidenceV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.humanoid.transition-evidence.v1";

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        morphology: HumanoidMorphology,
        clock_domain_id: impl Into<String>,
        episode_id: impl Into<String>,
        pre_state_source: HumanoidTransitionStateSourceV1,
        post_state_source: HumanoidTransitionStateSourceV1,
        pre_state: SemanticHumanoidFrameV1,
        backend_action: BackendAppliedActionSnapshotV1,
        post_state: SemanticHumanoidFrameV1,
    ) -> Result<Self, HumanoidTransitionEvidenceErrorV1> {
        let mut transition = Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            morphology,
            clock_domain_id: clock_domain_id.into(),
            episode_id: episode_id.into(),
            pre_state_source,
            post_state_source,
            pre_state,
            backend_action,
            post_state,
            transition_digest_hex: String::new(),
        };
        transition.validate_without_digest()?;
        transition.transition_digest_hex = transition.compute_digest_hex()?;
        transition.validate()?;
        Ok(transition)
    }

    pub fn validate(&self) -> Result<(), HumanoidTransitionEvidenceErrorV1> {
        self.validate_without_digest()?;
        if self.transition_digest_hex != self.compute_digest_hex()? {
            return Err(HumanoidTransitionEvidenceErrorV1::DigestMismatch);
        }
        Ok(())
    }

    fn validate_without_digest(&self) -> Result<(), HumanoidTransitionEvidenceErrorV1> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err(HumanoidTransitionEvidenceErrorV1::SchemaMismatch);
        }
        if self.clock_domain_id.trim().is_empty() {
            return Err(HumanoidTransitionEvidenceErrorV1::EmptyClockDomain);
        }
        if self.episode_id.trim().is_empty() {
            return Err(HumanoidTransitionEvidenceErrorV1::EmptyEpisodeId);
        }

        self.pre_state
            .validate()
            .map_err(HumanoidTransitionEvidenceErrorV1::PreState)?;
        self.post_state
            .validate()
            .map_err(HumanoidTransitionEvidenceErrorV1::PostState)?;
        self.backend_action
            .validate()
            .map_err(HumanoidTransitionEvidenceErrorV1::BackendAction)?;

        if self.pre_state.morphology != self.morphology
            || self.post_state.morphology != self.morphology
            || self.backend_action.morphology != self.morphology
        {
            return Err(HumanoidTransitionEvidenceErrorV1::MorphologyMismatch);
        }

        // R3.13 forms an exact causal tuple, not a nearest-neighbor temporal
        // join. Delayed/asynchronous observations belong behind R3.9 alignment
        // evidence rather than being silently snapped onto this transition.
        if self.pre_state.timestamp_seconds.to_bits()
            != self.backend_action.pre_step_timestamp_seconds.to_bits()
        {
            return Err(HumanoidTransitionEvidenceErrorV1::PreTimestampMismatch);
        }
        if self.post_state.timestamp_seconds.to_bits()
            != self.backend_action.post_step_timestamp_seconds.to_bits()
        {
            return Err(HumanoidTransitionEvidenceErrorV1::PostTimestampMismatch);
        }
        if self.post_state.timestamp_seconds <= self.pre_state.timestamp_seconds {
            return Err(HumanoidTransitionEvidenceErrorV1::NonForwardTransition);
        }

        let semantic_dt = self.post_state.timestamp_seconds - self.pre_state.timestamp_seconds;
        if semantic_dt.to_bits() != self.backend_action.applied_dt_seconds.to_bits() {
            return Err(HumanoidTransitionEvidenceErrorV1::AppliedDtMismatch);
        }
        Ok(())
    }

    pub fn applied_dt_seconds(&self) -> f64 {
        self.backend_action.applied_dt_seconds
    }

    pub fn requested_dt_seconds(&self) -> f64 {
        self.backend_action.requested_dt_seconds
    }

    pub fn step_sequence(&self) -> u64 {
        self.backend_action.step_sequence
    }

    pub fn homogeneous_state_source(&self) -> bool {
        self.pre_state_source == self.post_state_source
    }

    /// Conservative readiness predicate for the first R4 simulator experiments.
    /// It deliberately requires privileged truth on both sides; observation-path
    /// prediction should be qualified separately with explicit temporal alignment.
    pub fn r4_privileged_truth_ready(&self) -> bool {
        self.homogeneous_state_source()
            && self.pre_state_source == HumanoidTransitionStateSourceV1::PrivilegedBackendTruth
            && self.validate().is_ok()
    }

    fn compute_digest_hex(&self) -> Result<String, HumanoidTransitionEvidenceErrorV1> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(TRANSITION_EVIDENCE_DOMAIN_V1);
        feed_str(&mut hasher, &self.schema_id);
        feed_str(&mut hasher, self.morphology.schema_id());
        feed_str(&mut hasher, &self.clock_domain_id);
        feed_str(&mut hasher, &self.episode_id);
        feed_str(&mut hasher, self.pre_state_source.token());
        feed_str(&mut hasher, self.post_state_source.token());
        feed_semantic_frame(&mut hasher, &self.pre_state)?;
        feed_str(&mut hasher, &self.backend_action.evidence_digest_hex);
        feed_semantic_frame(&mut hasher, &self.post_state)?;
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

/// Capture one exact privileged-truth transition from the evidence-transparent
/// simple backend. Invalid requested action/time is rejected before the backend
/// is mutated.
pub fn capture_privileged_transition_v1(
    simulator: &mut InstrumentedSimpleHumanoidSimulator,
    requested_command: &HumanoidCommand,
    requested_dt_seconds: f64,
    clock_domain_id: impl Into<String>,
    episode_id: impl Into<String>,
) -> Result<HumanoidTransitionEvidenceV1, HumanoidTransitionEvidenceErrorV1> {
    if !requested_dt_seconds.is_finite() || requested_dt_seconds <= 0.0 {
        return Err(HumanoidTransitionEvidenceErrorV1::InvalidRequestedDt);
    }

    let morphology = simulator.morphology();
    requested_command
        .validate_for(morphology.num_actuators(), ActuationMode::NormalizedTorque)
        .map_err(HumanoidTransitionEvidenceErrorV1::RequestedCommand)?;

    let encoder = SemanticHumanoidEncoderV1::new();
    let pre_state = encoder
        .frame(simulator.true_state(), morphology)
        .map_err(HumanoidTransitionEvidenceErrorV1::PreState)?;

    simulator.step(requested_command, requested_dt_seconds);

    let backend_action = simulator
        .backend_applied_action()
        .cloned()
        .ok_or(HumanoidTransitionEvidenceErrorV1::MissingBackendAction)?;
    let post_state = encoder
        .frame(simulator.true_state(), morphology)
        .map_err(HumanoidTransitionEvidenceErrorV1::PostState)?;

    HumanoidTransitionEvidenceV1::new(
        morphology,
        clock_domain_id,
        episode_id,
        HumanoidTransitionStateSourceV1::PrivilegedBackendTruth,
        HumanoidTransitionStateSourceV1::PrivilegedBackendTruth,
        pre_state,
        backend_action,
        post_state,
    )
}

fn feed_semantic_frame(
    hasher: &mut blake3::Hasher,
    frame: &SemanticHumanoidFrameV1,
) -> Result<(), HumanoidTransitionEvidenceErrorV1> {
    feed_str(hasher, &frame.schema_id);
    feed_str(hasher, frame.morphology.schema_id());
    feed_str(hasher, &frame.source_observation_schema_id);
    hasher.update(&frame.timestamp_seconds.to_bits().to_le_bytes());
    hasher.update(&(frame.policy_observations.len() as u64).to_le_bytes());

    for observation in &frame.policy_observations {
        let address_digest = observation
            .address()
            .semantic_digest()
            .map_err(HumanoidTransitionEvidenceErrorV1::Sensorimotor)?;
        hasher.update(&address_digest);
        match observation {
            SensorimotorObservationV1::Measured(measurement) => {
                hasher.update(&[1]);
                hasher.update(&measurement.value.to_bits().to_le_bytes());
            }
            SensorimotorObservationV1::Missing { reason, .. } => {
                hasher.update(&[0]);
                feed_str(hasher, missing_reason_token(reason));
            }
        }
    }

    for value in &frame.privileged_root_position_world_m {
        match value {
            Some(value) => {
                hasher.update(&[1]);
                hasher.update(&value.to_bits().to_le_bytes());
            }
            None => hasher.update(&[0]),
        }
    }
    Ok(())
}

fn missing_reason_token(reason: &MissingObservationReasonV1) -> &'static str {
    match reason {
        MissingObservationReasonV1::NotPresent => "not_present",
        MissingObservationReasonV1::NotObserved => "not_observed",
        MissingObservationReasonV1::Stale => "stale",
        MissingObservationReasonV1::Invalid => "invalid",
        MissingObservationReasonV1::Disabled => "disabled",
        MissingObservationReasonV1::Unknown => "unknown",
    }
}

fn feed_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[derive(Debug)]
pub enum HumanoidTransitionEvidenceErrorV1 {
    SchemaMismatch,
    EmptyClockDomain,
    EmptyEpisodeId,
    InvalidRequestedDt,
    RequestedCommand(CommandValidationError),
    PreState(SemanticHumanoidErrorV1),
    PostState(SemanticHumanoidErrorV1),
    BackendAction(BackendAppliedActionEvidenceErrorV1),
    MissingBackendAction,
    MorphologyMismatch,
    PreTimestampMismatch,
    PostTimestampMismatch,
    NonForwardTransition,
    AppliedDtMismatch,
    Sensorimotor(&'static str),
    DigestMismatch,
}

impl fmt::Display for HumanoidTransitionEvidenceErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaMismatch => write!(f, "unsupported humanoid transition schema"),
            Self::EmptyClockDomain => write!(f, "transition clock domain must be non-empty"),
            Self::EmptyEpisodeId => write!(f, "transition episode id must be non-empty"),
            Self::InvalidRequestedDt => write!(f, "requested transition dt must be finite and positive"),
            Self::RequestedCommand(error) => write!(f, "invalid requested command: {error:?}"),
            Self::PreState(error) => write!(f, "invalid pre-state: {error}"),
            Self::PostState(error) => write!(f, "invalid post-state: {error}"),
            Self::BackendAction(error) => write!(f, "invalid backend action: {error:?}"),
            Self::MissingBackendAction => write!(f, "backend did not expose applied-action evidence"),
            Self::MorphologyMismatch => write!(f, "transition morphology mismatch"),
            Self::PreTimestampMismatch => write!(f, "pre-state timestamp does not match backend action"),
            Self::PostTimestampMismatch => write!(f, "post-state timestamp does not match backend action"),
            Self::NonForwardTransition => write!(f, "transition must advance forward in time"),
            Self::AppliedDtMismatch => write!(f, "semantic state delta does not match backend-applied dt"),
            Self::Sensorimotor(message) => write!(f, "sensorimotor schema: {message}"),
            Self::DigestMismatch => write!(f, "transition digest mismatch"),
        }
    }
}

impl std::error::Error for HumanoidTransitionEvidenceErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;

    fn requested_command(morphology: HumanoidMorphology) -> HumanoidCommand {
        let mut torques = vec![0.0f32; morphology.num_actuators()];
        torques[0] = 0.5;
        if torques.len() > 6 {
            torques[6] = -0.2;
        }
        HumanoidCommand { torques }
    }

    #[test]
    fn exact_transition_binds_pre_action_dt_and_post_state() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut simulator =
            InstrumentedSimpleHumanoidSimulator::new_for(morphology).with_actuator_noise(0.03);
        simulator.reset_with_perturbation(0.0, 77);

        let transition = capture_privileged_transition_v1(
            &mut simulator,
            &requested_command(morphology),
            0.025,
            "simple-sim-clock:test",
            "episode:77",
        )
        .unwrap();

        transition.validate().unwrap();
        assert!(transition.r4_privileged_truth_ready());
        assert_eq!(transition.step_sequence(), 1);
        assert_eq!(
            transition.pre_state.timestamp_seconds.to_bits(),
            transition.backend_action.pre_step_timestamp_seconds.to_bits()
        );
        assert_eq!(
            transition.post_state.timestamp_seconds.to_bits(),
            transition.backend_action.post_step_timestamp_seconds.to_bits()
        );
        assert_eq!(
            transition.applied_dt_seconds().to_bits(),
            (transition.post_state.timestamp_seconds - transition.pre_state.timestamp_seconds)
                .to_bits()
        );
        assert!(transition.backend_action.backend_adaptation_applied);
    }

    #[test]
    fn deterministic_run_reproduces_transition_digest() {
        let morphology = HumanoidMorphology::Dmc21;
        let command = requested_command(morphology);
        let mut left =
            InstrumentedSimpleHumanoidSimulator::new_for(morphology).with_actuator_noise(0.03);
        let mut right =
            InstrumentedSimpleHumanoidSimulator::new_for(morphology).with_actuator_noise(0.03);
        left.reset_with_perturbation(0.0, 9);
        right.reset_with_perturbation(0.0, 9);

        let left_transition = capture_privileged_transition_v1(
            &mut left,
            &command,
            0.025,
            "simple-sim-clock:deterministic",
            "episode:9",
        )
        .unwrap();
        let right_transition = capture_privileged_transition_v1(
            &mut right,
            &command,
            0.025,
            "simple-sim-clock:deterministic",
            "episode:9",
        )
        .unwrap();

        assert_eq!(
            left_transition.transition_digest_hex,
            right_transition.transition_digest_hex
        );
    }

    #[test]
    fn invalid_capture_input_does_not_mutate_backend() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut simulator = InstrumentedSimpleHumanoidSimulator::new_for(morphology);
        simulator.reset();
        let before = simulator.true_state().timestamp.to_bits();

        assert!(matches!(
            capture_privileged_transition_v1(
                &mut simulator,
                &requested_command(morphology),
                0.0,
                "clock:invalid-dt",
                "episode:invalid-dt",
            ),
            Err(HumanoidTransitionEvidenceErrorV1::InvalidRequestedDt)
        ));
        assert_eq!(simulator.true_state().timestamp.to_bits(), before);
        assert!(simulator.backend_applied_action().is_none());

        let malformed = HumanoidCommand::zero_for(morphology.num_actuators() - 1);
        assert!(matches!(
            capture_privileged_transition_v1(
                &mut simulator,
                &malformed,
                0.025,
                "clock:invalid-command",
                "episode:invalid-command",
            ),
            Err(HumanoidTransitionEvidenceErrorV1::RequestedCommand(_))
        ));
        assert_eq!(simulator.true_state().timestamp.to_bits(), before);
        assert!(simulator.backend_applied_action().is_none());
    }

    #[test]
    fn timestamp_tampering_fails_before_digest_is_trusted() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut simulator = InstrumentedSimpleHumanoidSimulator::new_for(morphology);
        simulator.reset();
        let mut transition = capture_privileged_transition_v1(
            &mut simulator,
            &requested_command(morphology),
            0.025,
            "simple-sim-clock:tamper",
            "episode:tamper",
        )
        .unwrap();

        transition.pre_state.timestamp_seconds += 0.001;
        assert!(matches!(
            transition.validate(),
            Err(HumanoidTransitionEvidenceErrorV1::PreTimestampMismatch)
        ));
    }

    #[test]
    fn action_tampering_invalidates_transition_via_nested_evidence() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut simulator = InstrumentedSimpleHumanoidSimulator::new_for(morphology);
        simulator.reset();
        let mut transition = capture_privileged_transition_v1(
            &mut simulator,
            &requested_command(morphology),
            0.025,
            "simple-sim-clock:action-tamper",
            "episode:action-tamper",
        )
        .unwrap();

        transition.backend_action.command.torques[0] += 0.001;
        assert!(matches!(
            transition.validate(),
            Err(HumanoidTransitionEvidenceErrorV1::BackendAction(
                BackendAppliedActionEvidenceErrorV1::DigestMismatch
            ))
        ));
    }

    #[test]
    fn clock_and_episode_identity_participate_in_transition_commitment() {
        let morphology = HumanoidMorphology::Dmc21;
        let command = requested_command(morphology);
        let mut first = InstrumentedSimpleHumanoidSimulator::new_for(morphology);
        let mut second = InstrumentedSimpleHumanoidSimulator::new_for(morphology);
        first.reset();
        second.reset();

        let first_transition = capture_privileged_transition_v1(
            &mut first,
            &command,
            0.025,
            "clock:a",
            "episode:1",
        )
        .unwrap();
        let second_transition = capture_privileged_transition_v1(
            &mut second,
            &command,
            0.025,
            "clock:b",
            "episode:1",
        )
        .unwrap();

        assert_ne!(
            first_transition.transition_digest_hex,
            second_transition.transition_digest_hex
        );
    }

    #[test]
    fn empty_clock_or_episode_fails_closed() {
        let morphology = HumanoidMorphology::Dmc21;
        let command = requested_command(morphology);
        let mut simulator = InstrumentedSimpleHumanoidSimulator::new_for(morphology);
        simulator.reset();

        let pre_encoder = SemanticHumanoidEncoderV1::new();
        let pre = pre_encoder.frame(simulator.true_state(), morphology).unwrap();
        simulator.step(&command, 0.025);
        let action = simulator.backend_applied_action().unwrap().clone();
        let post = pre_encoder.frame(simulator.true_state(), morphology).unwrap();

        assert!(matches!(
            HumanoidTransitionEvidenceV1::new(
                morphology,
                "",
                "episode:1",
                HumanoidTransitionStateSourceV1::PrivilegedBackendTruth,
                HumanoidTransitionStateSourceV1::PrivilegedBackendTruth,
                pre.clone(),
                action.clone(),
                post.clone(),
            ),
            Err(HumanoidTransitionEvidenceErrorV1::EmptyClockDomain)
        ));

        assert!(matches!(
            HumanoidTransitionEvidenceV1::new(
                morphology,
                "clock:1",
                "   ",
                HumanoidTransitionStateSourceV1::PrivilegedBackendTruth,
                HumanoidTransitionStateSourceV1::PrivilegedBackendTruth,
                pre,
                action,
                post,
            ),
            Err(HumanoidTransitionEvidenceErrorV1::EmptyEpisodeId)
        ));
    }
}
