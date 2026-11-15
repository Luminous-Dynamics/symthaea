// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact linkage from camera observations to applied-action supervised labels.
//!
//! This module validates training-target provenance only. It does not call
//! VehicleController::train_step, invoke a policy, or emit actuator commands.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_core::hdc::ContinuousHV;
use symthaea_vehicle::VehicleCommand;

use crate::{
    DrivingObservationReceipt, FusedDrivingInput,
    policy_compatibility::{CameraPolicyPrivilegeProfile, vehicle_action_schema_sha256},
};

pub const APPLIED_ACTION_TRAINING_RECEIPT_SCHEMA_VERSION: u32 = 1;
const MAX_LABEL_ID_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AppliedActionDisposition {
    Applied,
    Proposed,
    Rejected,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AppliedActionSourceKind {
    HumanDemonstration,
    ClassicalController,
    PrivilegedExpert,
    GroundTruthOracle,
    Unknown,
}

impl AppliedActionSourceKind {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::HumanDemonstration => "HUMAN_DEMONSTRATION",
            Self::ClassicalController => "CLASSICAL_CONTROLLER",
            Self::PrivilegedExpert => "PRIVILEGED_EXPERT",
            Self::GroundTruthOracle => "GROUND_TRUTH_ORACLE",
            Self::Unknown => "UNKNOWN",
        }
    }
}

/// Canonical payload supplied by the simulator/recorder for an action that
/// was actually applied, not merely proposed by a policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppliedVehicleAction {
    pub steering: f32,
    pub throttle: f32,
    pub brake: f32,
}

impl AppliedVehicleAction {
    fn into_vehicle_command(self) -> VehicleCommand {
        VehicleCommand {
            steering: self.steering,
            throttle: self.throttle,
            brake: self.brake,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppliedActionTrainingReceipt {
    pub schema_version: u32,
    pub receipt_id: String,
    pub episode_id: String,
    pub agent_id: String,
    pub clock_id: String,
    pub command_id: String,
    pub tick: u64,
    pub applied_timestamp_us: u64,
    pub disposition: AppliedActionDisposition,
    pub action_schema_sha256: String,
    pub source_kind: AppliedActionSourceKind,
    pub source_identity: String,
    pub privilege_profile: CameraPolicyPrivilegeProfile,
    pub action: AppliedVehicleAction,
    pub receipt_sha256: String,
}

fn update_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn fused_policy_input_sha256(input: &ContinuousHV) -> String {
    let mut hasher = Sha256::new();
    hasher.update((input.dim() as u64).to_le_bytes());
    for value in input.as_slice() {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn privilege_profile_name(value: CameraPolicyPrivilegeProfile) -> &'static str {
    match value {
        CameraPolicyPrivilegeProfile::NoPrivilegedInputs => "NO_PRIVILEGED_INPUTS",
        CameraPolicyPrivilegeProfile::PrivilegedExpert => "PRIVILEGED_EXPERT",
        CameraPolicyPrivilegeProfile::GroundTruthPerception => "GROUND_TRUTH_PERCEPTION",
        CameraPolicyPrivilegeProfile::MixedDeclared => "MIXED_DECLARED",
        CameraPolicyPrivilegeProfile::Unknown => "UNKNOWN",
    }
}

impl AppliedActionTrainingReceipt {
    /// Canonical, domain-separated digest over every receipt field except the
    /// digest itself. This checks internal consistency, not issuer identity.
    pub fn calculated_receipt_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea-applied-action-training-receipt-v1\0");
        hasher.update(self.schema_version.to_le_bytes());
        update_string(&mut hasher, &self.receipt_id);
        update_string(&mut hasher, &self.episode_id);
        update_string(&mut hasher, &self.agent_id);
        update_string(&mut hasher, &self.clock_id);
        update_string(&mut hasher, &self.command_id);
        hasher.update(self.tick.to_le_bytes());
        hasher.update(self.applied_timestamp_us.to_le_bytes());
        update_string(&mut hasher, match self.disposition {
            AppliedActionDisposition::Applied => "APPLIED",
            AppliedActionDisposition::Proposed => "PROPOSED",
            AppliedActionDisposition::Rejected => "REJECTED",
            AppliedActionDisposition::Unknown => "UNKNOWN",
        });
        update_string(&mut hasher, &self.action_schema_sha256);
        update_string(&mut hasher, self.source_kind.canonical_name());
        update_string(&mut hasher, &self.source_identity);
        update_string(&mut hasher, privilege_profile_name(self.privilege_profile));
        hasher.update(self.action.steering.to_bits().to_le_bytes());
        hasher.update(self.action.throttle.to_bits().to_le_bytes());
        hasher.update(self.action.brake.to_bits().to_le_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Validated action target and declared privilege/source metadata. This value
/// can be supplied to a separately scoped trainer, but this module does not
/// perform any training.
#[derive(Debug, Clone)]
pub struct ValidatedAppliedActionLabel {
    pub target: VehicleCommand,
    pub receipt_id: String,
    pub receipt_sha256: String,
    pub command_id: String,
    pub episode_id: String,
    pub agent_id: String,
    pub clock_id: String,
    pub tick: u64,
    pub applied_timestamp_us: u64,
    pub dt_seconds: f32,
    pub source_kind: AppliedActionSourceKind,
    pub source_identity: String,
    pub privilege_profile: CameraPolicyPrivilegeProfile,
    pub observation_receipt_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingLabelError {
    UnsupportedSchemaVersion,
    MissingIdentity,
    IdentityTooLong,
    InvalidReceiptDigest,
    ObservationReceiptMismatch,
    PolicyInputDigestMismatch,
    InvalidObservationDeltaTime,
    ActionSchemaMismatch,
    MissingAppliedActionReference,
    AppliedActionReferenceMismatch,
    EpisodeMismatch,
    AgentMismatch,
    ClockMismatch,
    TickMismatch,
    FutureAppliedAction,
    UnknownSourceKind,
    UnknownPrivilegeProfile,
    ActionNotApplied,
    NonFiniteAction,
    ActionOutOfRange,
}

impl std::fmt::Display for TrainingLabelError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for TrainingLabelError {}

fn validate_identity(value: &str) -> Result<(), TrainingLabelError> {
    if value.trim().is_empty() {
        return Err(TrainingLabelError::MissingIdentity);
    }
    if value.len() > MAX_LABEL_ID_BYTES {
        return Err(TrainingLabelError::IdentityTooLong);
    }
    Ok(())
}

/// Validate a receipt against the exact applied-action reference committed by
/// the upstream camera observation. The returned target is never inferred by
/// nearest timestamp, episode proximity, command proposal or policy output.
pub fn validate_applied_action_training_label(
    observation: &FusedDrivingInput,
    label: &AppliedActionTrainingReceipt,
) -> Result<ValidatedAppliedActionLabel, TrainingLabelError> {
    if label.schema_version != APPLIED_ACTION_TRAINING_RECEIPT_SCHEMA_VERSION {
        return Err(TrainingLabelError::UnsupportedSchemaVersion);
    }

    for identity in [
        label.receipt_id.as_str(),
        label.episode_id.as_str(),
        label.agent_id.as_str(),
        label.clock_id.as_str(),
        label.command_id.as_str(),
        label.source_identity.as_str(),
    ] {
        validate_identity(identity)?;
    }

    if label.receipt_sha256.len() != 64
        || !label
            .receipt_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || label.receipt_sha256 != label.calculated_receipt_sha256()
    {
        return Err(TrainingLabelError::InvalidReceiptDigest);
    }

    if label.disposition != AppliedActionDisposition::Applied {
        return Err(TrainingLabelError::ActionNotApplied);
    }
    if label.action_schema_sha256 != vehicle_action_schema_sha256() {
        return Err(TrainingLabelError::ActionSchemaMismatch);
    }
    if label.source_kind == AppliedActionSourceKind::Unknown {
        return Err(TrainingLabelError::UnknownSourceKind);
    }
    if label.privilege_profile == CameraPolicyPrivilegeProfile::Unknown {
        return Err(TrainingLabelError::UnknownPrivilegeProfile);
    }

    let receipt: &DrivingObservationReceipt = &observation.receipt;
    if receipt.receipt_sha256 != crate::digest_receipt(receipt)
        || receipt.capture_timestamp_us != receipt.upstream_capture_timestamp_us
    {
        return Err(TrainingLabelError::ObservationReceiptMismatch);
    }
    if receipt.fused_policy_input_sha256 != fused_policy_input_sha256(&observation.policy_input) {
        return Err(TrainingLabelError::PolicyInputDigestMismatch);
    }
    if receipt.derived_dt_us == 0 {
        return Err(TrainingLabelError::InvalidObservationDeltaTime);
    }
    let (
        Some(reference_id),
        Some(reference_sha256),
        Some(reference_command_id),
        Some(reference_tick),
    ) = (
        receipt.upstream_applied_action_receipt_id.as_deref(),
        receipt.upstream_applied_action_receipt_sha256.as_deref(),
        receipt.upstream_applied_action_command_id.as_deref(),
        receipt.upstream_applied_action_tick,
    )
    else {
        return Err(TrainingLabelError::MissingAppliedActionReference);
    };

    if reference_id != label.receipt_id
        || reference_sha256 != label.receipt_sha256
        || reference_command_id != label.command_id
    {
        return Err(TrainingLabelError::AppliedActionReferenceMismatch);
    }
    if reference_tick != label.tick {
        return Err(TrainingLabelError::TickMismatch);
    }
    if receipt.identity.episode_id != label.episode_id {
        return Err(TrainingLabelError::EpisodeMismatch);
    }
    if receipt.identity.agent_id != label.agent_id {
        return Err(TrainingLabelError::AgentMismatch);
    }
    if receipt.upstream_clock_id != label.clock_id {
        return Err(TrainingLabelError::ClockMismatch);
    }
    if label.applied_timestamp_us > receipt.capture_timestamp_us {
        return Err(TrainingLabelError::FutureAppliedAction);
    }

    let action = label.action;
    if !action.steering.is_finite()
        || !action.throttle.is_finite()
        || !action.brake.is_finite()
    {
        return Err(TrainingLabelError::NonFiniteAction);
    }
    if !(-1.0..=1.0).contains(&action.steering)
        || !(0.0..=1.0).contains(&action.throttle)
        || !(0.0..=1.0).contains(&action.brake)
    {
        return Err(TrainingLabelError::ActionOutOfRange);
    }

    Ok(ValidatedAppliedActionLabel {
        target: action.into_vehicle_command(),
        receipt_id: label.receipt_id.clone(),
        receipt_sha256: label.receipt_sha256.clone(),
        command_id: label.command_id.clone(),
        episode_id: label.episode_id.clone(),
        agent_id: label.agent_id.clone(),
        clock_id: label.clock_id.clone(),
        tick: label.tick,
        applied_timestamp_us: label.applied_timestamp_us,
        policy_input_sha256: receipt.fused_policy_input_sha256.clone(),
        dt_seconds: (receipt.derived_dt_us as f64 / 1_000_000.0) as f32,
        source_kind: label.source_kind,
        source_identity: label.source_identity.clone(),
        privilege_profile: label.privilege_profile,
        observation_receipt_sha256: receipt.receipt_sha256.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BridgeConfig, CameraDrivingBridge, CapturedDrivingObservation, DrivingRunIdentity,
        UpstreamCameraObservation, digest_bytes,
    };
    use symthaea_vehicle::VehicleState;
    use symthaea_vision_manifold::VisionConfig;

    const EPISODE: &str = "episode-applied-action-1";
    const AGENT: &str = "ego";
    const CLOCK: &str = "clock-applied-action-1";
    const RECEIPT_ID: &str = "applied-action-receipt-4";
    const COMMAND_ID: &str = "command-4";
    const TICK: u64 = 4;
    const CAPTURE_US: u64 = 10_000;

    fn label() -> AppliedActionTrainingReceipt {
        let mut receipt = AppliedActionTrainingReceipt {
            schema_version: APPLIED_ACTION_TRAINING_RECEIPT_SCHEMA_VERSION,
            receipt_id: RECEIPT_ID.into(),
            episode_id: EPISODE.into(),
            agent_id: AGENT.into(),
            clock_id: CLOCK.into(),
            command_id: COMMAND_ID.into(),
            tick: TICK,
            applied_timestamp_us: 9_900,
            disposition: AppliedActionDisposition::Applied,
            action_schema_sha256: vehicle_action_schema_sha256(),
            source_kind: AppliedActionSourceKind::ClassicalController,
            source_identity: "teacher-controller-r3".into(),
            privilege_profile: CameraPolicyPrivilegeProfile::NoPrivilegedInputs,
            action: AppliedVehicleAction {
                steering: 0.25,
                throttle: 0.6,
                brake: 0.0,
            },
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = receipt.calculated_receipt_sha256();
        receipt
    }

    fn fused_input(label: &AppliedActionTrainingReceipt) -> FusedDrivingInput {
        let pixels = vec![127_u8; 16 * 16 * 3];
        let identity = DrivingRunIdentity {
            run_id: "run-applied-action-1".into(),
            episode_id: EPISODE.into(),
            scenario_id: "scenario-applied-action-1".into(),
            agent_id: AGENT.into(),
            camera_id: "front".into(),
            simulator_revision: "sim-r1".into(),
            policy_revision: "policy-r1".into(),
        };
        let config = BridgeConfig {
            identity,
            max_width: 16,
            max_height: 16,
            max_frame_bytes: pixels.len(),
            initial_dt_us: 10_000,
            maximum_frame_gap_us: 20_000,
            maximum_observation_delivery_age_us: 5_000,
            maximum_sensor_clock_skew_us: 0,
            visual_weight_ppm: 600_000,
            vehicle_encoder_seed_phrase: "applied-action-label-test".into(),
            expected_camera_calibration_sha256: None,
        };
        let mut upstream = UpstreamCameraObservation {
            schema_version: 1,
            observation_id: String::new(),
            pixel_format: "rgb8_srgb".into(),
            frame_byte_len: pixels.len(),
            receipt_sha256: String::new(),
            frame_sha256: digest_bytes(&pixels),
            episode_id: EPISODE.into(),
            agent_id: AGENT.into(),
            clock_id: CLOCK.into(),
            tick: TICK,
            sequence: 1,
            camera_id: "front".into(),
            camera_calibration_sha256: "c".repeat(64),
            renderer_revision: "renderer-r1".into(),
            capture_timestamp_us: CAPTURE_US,
            render_completed_timestamp_us: CAPTURE_US + 100,
            delivered_timestamp_us: CAPTURE_US + 200,
            width: 16,
            height: 16,
            channels: 3,
            applied_action_receipt_id: Some(label.receipt_id.clone()),
            applied_action_receipt_sha256: Some(label.receipt_sha256.clone()),
            applied_action_command_id: Some(label.command_id.clone()),
            applied_action_tick: Some(label.tick),
        };
        upstream.observation_id = upstream.calculated_observation_id();
        upstream.receipt_sha256 = upstream.calculated_receipt_sha256();

        let mut state = VehicleState::cruising(4.0);
        state.timestamp = 0.01;
        let mut bridge =
            CameraDrivingBridge::try_new(config, VisionConfig::default()).expect("bridge config");
        bridge
            .observe(CapturedDrivingObservation {
                sequence: 1,
                timestamp_us: CAPTURE_US,
                vehicle_state_timestamp_us: CAPTURE_US,
                pixels: &pixels,
                width: 16,
                height: 16,
                channels: 3,
                vehicle_state: &state,
                upstream,
            })
            .expect("fused observation")
    }

    #[test]
    fn exact_applied_action_receipt_produces_a_validated_training_target() {
        let label = label();
        let observation = fused_input(&label);
        let validated = validate_applied_action_training_label(&observation, &label).unwrap();
        assert_eq!(validated.target.steering, 0.25);
        assert_eq!(validated.target.throttle, 0.6);
        assert_eq!(validated.target.brake, 0.0);
        assert_eq!(validated.tick, TICK);
        assert_eq!(validated.dt_seconds, 0.01);
        assert_eq!(validated.source_kind, AppliedActionSourceKind::ClassicalController);
    }

    #[test]
    fn proposed_or_rejected_action_cannot_become_a_training_target() {
        let mut value = label();
        value.disposition = AppliedActionDisposition::Proposed;
        value.receipt_sha256 = value.calculated_receipt_sha256();
        assert_eq!(
            validate_applied_action_training_label(&fused_input(&value), &value).unwrap_err(),
            TrainingLabelError::ActionNotApplied
        );
        value.disposition = AppliedActionDisposition::Rejected;
        value.receipt_sha256 = value.calculated_receipt_sha256();
        assert_eq!(
            validate_applied_action_training_label(&fused_input(&value), &value).unwrap_err(),
            TrainingLabelError::ActionNotApplied
        );
    }

    #[test]
    fn validated_target_is_bound_to_exact_fused_input_digest() {
        let value = label();
        let observation = fused_input(&value);
        let validated = validate_applied_action_training_label(&observation, &value).unwrap();
        assert_eq!(validated.policy_input_sha256, observation.receipt.fused_policy_input_sha256);
        assert_eq!(validated.observation_receipt_sha256, observation.receipt.receipt_sha256);
    }

    #[test]
    fn mutated_fused_policy_vector_is_rejected_even_when_receipt_is_unchanged() {
        let valid = label();
        let mut observation = fused_input(&valid);
        observation.policy_input =
            ContinuousHV::from_vec(vec![0.0; observation.policy_input.dim()]);
        assert_eq!(
            validate_applied_action_training_label(&observation, &valid).unwrap_err(),
            TrainingLabelError::PolicyInputDigestMismatch
        );
    }

    #[test]
    fn receipt_payload_mutation_is_rejected() {
        let valid = label();
        let observation = fused_input(&valid);
        let mut changed = valid;
        changed.action.throttle = 0.1;
        assert_eq!(
            validate_applied_action_training_label(&observation, &changed).unwrap_err(),
            TrainingLabelError::InvalidReceiptDigest
        );
    }

    #[test]
    fn receipt_reference_mismatch_is_rejected_even_with_a_recomputed_label_digest() {
        let valid = label();
        let observation = fused_input(&valid);
        let mut changed = valid;
        changed.receipt_id = "different-receipt".into();
        changed.receipt_sha256 = changed.calculated_receipt_sha256();
        assert_eq!(
            validate_applied_action_training_label(&observation, &changed).unwrap_err(),
            TrainingLabelError::AppliedActionReferenceMismatch
        );
    }

    #[test]
    fn future_action_timestamp_is_rejected() {
        let mut value = label();
        value.applied_timestamp_us = CAPTURE_US + 1;
        value.receipt_sha256 = value.calculated_receipt_sha256();
        let observation = fused_input(&value);
        assert_eq!(
            validate_applied_action_training_label(&observation, &value).unwrap_err(),
            TrainingLabelError::FutureAppliedAction
        );
    }

    #[test]
    fn wrong_episode_agent_clock_and_tick_are_rejected() {
        let valid = label();

        let mut wrong_episode_observation = fused_input(&valid);
        wrong_episode_observation.receipt.identity.episode_id = "another-episode".into();
        wrong_episode_observation.receipt.receipt_sha256 =
            crate::digest_receipt(&wrong_episode_observation.receipt);
        assert_eq!(
            validate_applied_action_training_label(&wrong_episode_observation, &valid).unwrap_err(),
            TrainingLabelError::EpisodeMismatch
        );

        let mut wrong_agent_observation = fused_input(&valid);
        wrong_agent_observation.receipt.identity.agent_id = "another-agent".into();
        wrong_agent_observation.receipt.receipt_sha256 =
            crate::digest_receipt(&wrong_agent_observation.receipt);
        assert_eq!(
            validate_applied_action_training_label(&wrong_agent_observation, &valid).unwrap_err(),
            TrainingLabelError::AgentMismatch
        );

        let mut wrong_clock_observation = fused_input(&valid);
        wrong_clock_observation.receipt.upstream_clock_id = "another-clock".into();
        wrong_clock_observation.receipt.receipt_sha256 =
            crate::digest_receipt(&wrong_clock_observation.receipt);
        assert_eq!(
            validate_applied_action_training_label(&wrong_clock_observation, &valid).unwrap_err(),
            TrainingLabelError::ClockMismatch
        );

        let mut wrong_tick_observation = fused_input(&valid);
        wrong_tick_observation.receipt.upstream_applied_action_tick = Some(TICK + 1);
        wrong_tick_observation.receipt.receipt_sha256 =
            crate::digest_receipt(&wrong_tick_observation.receipt);
        assert_eq!(
            validate_applied_action_training_label(&wrong_tick_observation, &valid).unwrap_err(),
            TrainingLabelError::TickMismatch
        );
    }

    #[test]
    fn missing_upstream_applied_action_reference_is_rejected() {
        let valid = label();
        let mut observation = fused_input(&valid);
        observation.receipt.upstream_applied_action_receipt_id = None;
        observation.receipt.upstream_applied_action_receipt_sha256 = None;
        observation.receipt.upstream_applied_action_command_id = None;
        observation.receipt.upstream_applied_action_tick = None;
        observation.receipt.receipt_sha256 = crate::digest_receipt(&observation.receipt);
        assert_eq!(
            validate_applied_action_training_label(&observation, &valid).unwrap_err(),
            TrainingLabelError::MissingAppliedActionReference
        );
    }

    #[test]
    fn observation_receipt_must_remain_internally_consistent() {
        let valid = label();
        let mut observation = fused_input(&valid);
        observation.receipt.upstream_clock_id = "tampered-clock".into();
        assert_eq!(
            validate_applied_action_training_label(&observation, &valid).unwrap_err(),
            TrainingLabelError::ObservationReceiptMismatch
        );
    }

    #[test]
    fn empty_training_source_identity_is_rejected() {
        let mut value = label();
        value.source_identity.clear();
        value.receipt_sha256 = value.calculated_receipt_sha256();
        let observation = fused_input(&value);
        assert_eq!(
            validate_applied_action_training_label(&observation, &value).unwrap_err(),
            TrainingLabelError::MissingIdentity
        );
    }

    #[test]
    fn unknown_source_or_privilege_is_not_admitted() {
        let mut value = label();
        let observation = fused_input(&value);
        value.source_kind = AppliedActionSourceKind::Unknown;
        value.receipt_sha256 = value.calculated_receipt_sha256();
        assert_eq!(
            validate_applied_action_training_label(&observation, &value).unwrap_err(),
            TrainingLabelError::UnknownSourceKind
        );

        value = label();
        value.privilege_profile = CameraPolicyPrivilegeProfile::Unknown;
        value.receipt_sha256 = value.calculated_receipt_sha256();
        assert_eq!(
            validate_applied_action_training_label(&fused_input(&value), &value).unwrap_err(),
            TrainingLabelError::UnknownPrivilegeProfile
        );
    }

    #[test]
    fn action_schema_and_action_ranges_are_enforced() {
        let mut value = label();
        value.action_schema_sha256 = "0".repeat(64);
        value.receipt_sha256 = value.calculated_receipt_sha256();
        assert_eq!(
            validate_applied_action_training_label(&fused_input(&value), &value).unwrap_err(),
            TrainingLabelError::ActionSchemaMismatch
        );

        value = label();
        value.action.steering = f32::NAN;
        value.receipt_sha256 = value.calculated_receipt_sha256();
        assert_eq!(
            validate_applied_action_training_label(&fused_input(&value), &value).unwrap_err(),
            TrainingLabelError::NonFiniteAction
        );

        value = label();
        value.action.throttle = 1.1;
        value.receipt_sha256 = value.calculated_receipt_sha256();
        assert_eq!(
            validate_applied_action_training_label(&fused_input(&value), &value).unwrap_err(),
            TrainingLabelError::ActionOutOfRange
        );
    }
}
