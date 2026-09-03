// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Protocol-neutral physical-effect envelope and device-side semantic acceptance.
//!
//! This crate closes a different boundary from host authorization. A perfectly
//! authorized host command is still unsafe if a device endpoint accepts arbitrary
//! bytes, replays an old generation after reboot, ignores expiry, or fails open on
//! local firmware/safety state.
//!
//! The serialized [`PhysicalEffectEnvelopeV1`] is deliberately **not authority**.
//! It is canonical semantic content that a future Xenia/device transport must
//! authenticate before any actuator layer can combine it with device-side semantic
//! acceptance. This crate never treats possession of envelope bytes as identity.
//!
//! Device semantic ordering is crash-conservative:
//!
//! ```text
//! authenticated envelope bytes (future transport layer)
//!   -> local semantic checks
//!   -> burn monotonically increasing device sequence
//!   -> build successor DeviceSemanticCheckpointV1
//!   -> persist checkpoint + external head
//!   -> SemanticallyAcceptedEffect   (still non-authorizing)
//!   -> future authenticated-transport + interlock composition
//!   -> actuator
//! ```
//!
//! Sequence numbers intentionally never roll back, including when the physical
//! effect later proves not to have happened.

#![deny(unsafe_code)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use symthaea_authority::{Digest32, Operation, ResourceRef};
use symthaea_iot_authority::{DeviceCommand, DeviceRuntimeState, SafetyEnvelope};
use symthaea_iot_durable_runtime::{DurableIoTHead, DurableUnknownPhysicalEffect};
use symthaea_iot_egress_guard::PostureBoundEgressPermit;
use symthaea_iot_policy::ActuationPolicyHead;
use symthaea_iot_posture::VerifierTrustHead;
use thiserror::Error;

pub const PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION: u16 = 1;
pub const DEVICE_ENFORCEMENT_CONFIG_SCHEMA_VERSION: u16 = 1;
pub const DEVICE_SEMANTIC_CHECKPOINT_SCHEMA_VERSION: u16 = 1;
pub const MAX_HOST_TO_DEVICE_EGRESS_WINDOW_S: u64 = 10;
pub const MAX_DEVICE_ENVELOPE_LIFETIME_S: u64 = 30;

const PHYSICAL_EFFECT_ENVELOPE_DOMAIN: &[u8] = b"symthaea-iot-physical-effect-envelope-v1\0";
const DEVICE_CONFIG_DOMAIN: &[u8] = b"symthaea-iot-device-enforcement-config-v1\0";
const DEVICE_CHECKPOINT_DOMAIN: &[u8] = b"symthaea-iot-device-semantic-checkpoint-v1\0";

/// Canonical semantic payload sent toward one physical device.
///
/// This object is serializable because it is wire data. It is not signed here and
/// is not an authorization token. The future transport adapter must authenticate
/// the exact bytes/digest independently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalEffectEnvelopeV1 {
    pub schema_version: u16,
    pub command: DeviceCommand,
    pub proposal_digest: Digest32,
    pub policy_digest: Digest32,
    pub policy_registry_head: ActuationPolicyHead,
    pub durable_host_head: DurableIoTHead,
    pub posture_result_digest: Digest32,
    pub posture_evidence_digest: Digest32,
    pub posture_reference_values_digest: Digest32,
    pub posture_appraisal_policy_digest: Digest32,
    pub posture_challenge_digest: Digest32,
    pub posture_verifier_trust_head: VerifierTrustHead,
    pub posture_expires_at_unix_s: u64,
    pub host_preflight_at_unix_s: u64,
    pub send_not_after_unix_s: u64,
}

impl PhysicalEffectEnvelopeV1 {
    pub fn validate_structure(&self) -> Result<(), DeviceProtocolError> {
        if self.schema_version != PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION {
            return Err(DeviceProtocolError::UnsupportedEnvelopeSchema);
        }
        if self.command.command_id.is_empty() {
            return Err(DeviceProtocolError::MalformedCommand);
        }
        if self.command.expires_at_unix_s < self.command.issued_at_unix_s {
            return Err(DeviceProtocolError::MalformedCommand);
        }
        if self.host_preflight_at_unix_s < self.command.issued_at_unix_s
            || self.host_preflight_at_unix_s > self.command.expires_at_unix_s
        {
            return Err(DeviceProtocolError::InvalidHostPreflightTime);
        }
        let egress_window = self
            .send_not_after_unix_s
            .checked_sub(self.host_preflight_at_unix_s)
            .ok_or(DeviceProtocolError::InvalidEgressWindow)?;
        if egress_window == 0 || egress_window > MAX_HOST_TO_DEVICE_EGRESS_WINDOW_S {
            return Err(DeviceProtocolError::InvalidEgressWindow);
        }
        if self.send_not_after_unix_s > self.command.expires_at_unix_s
            || self.send_not_after_unix_s > self.posture_expires_at_unix_s
        {
            return Err(DeviceProtocolError::EgressOutlivesAuthorityOrPosture);
        }
        for digest in [
            self.proposal_digest,
            self.policy_digest,
            self.posture_result_digest,
            self.posture_evidence_digest,
            self.posture_reference_values_digest,
            self.posture_appraisal_policy_digest,
            self.posture_challenge_digest,
        ] {
            if digest == Digest32([0; 32]) {
                return Err(DeviceProtocolError::ZeroSecurityDigest);
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, DeviceProtocolError> {
        self.validate_structure()?;
        let mut h = blake3::Hasher::new();
        h.update(PHYSICAL_EFFECT_ENVELOPE_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        update_digest(&mut h, self.command.digest());
        update_digest(&mut h, self.proposal_digest);
        update_digest(&mut h, self.policy_digest);
        h.update(&self.policy_registry_head.sequence.to_be_bytes());
        update_digest(&mut h, self.policy_registry_head.digest);
        h.update(&self.durable_host_head.action_head.sequence.to_be_bytes());
        update_digest(&mut h, self.durable_host_head.action_head.digest);
        update_digest(&mut h, self.durable_host_head.digest);
        update_digest(&mut h, self.posture_result_digest);
        update_digest(&mut h, self.posture_evidence_digest);
        update_digest(&mut h, self.posture_reference_values_digest);
        update_digest(&mut h, self.posture_appraisal_policy_digest);
        update_digest(&mut h, self.posture_challenge_digest);
        h.update(&self.posture_verifier_trust_head.sequence.to_be_bytes());
        update_digest(&mut h, self.posture_verifier_trust_head.digest);
        h.update(&self.posture_expires_at_unix_s.to_be_bytes());
        h.update(&self.host_preflight_at_unix_s.to_be_bytes());
        h.update(&self.send_not_after_unix_s.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }
}

/// Host-side affine state carrying the only posture-bound permit alongside its exact
/// canonical device envelope.
#[derive(Debug)]
pub struct PreparedDeviceEgress {
    permit: PostureBoundEgressPermit,
    envelope: PhysicalEffectEnvelopeV1,
    envelope_digest: Digest32,
}

impl PreparedDeviceEgress {
    pub fn envelope(&self) -> &PhysicalEffectEnvelopeV1 {
        &self.envelope
    }

    pub fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    /// Transport/device outcome is ambiguous. Host consequence capacity remains charged.
    pub fn into_unknown(self) -> DurableUnknownPhysicalEffect {
        self.permit.into_unknown()
    }
}

/// Consume the final posture-bound host permit into exactly one canonical envelope.
pub fn prepare_device_egress(
    permit: PostureBoundEgressPermit,
    send_not_after_unix_s: u64,
) -> Result<PreparedDeviceEgress, DeviceProtocolError> {
    let envelope = PhysicalEffectEnvelopeV1 {
        schema_version: PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION,
        command: permit.command().clone(),
        proposal_digest: permit.proposal_digest(),
        policy_digest: permit.policy_digest(),
        policy_registry_head: permit.policy_registry_head(),
        durable_host_head: permit.armed_head(),
        posture_result_digest: permit.posture_result_digest(),
        posture_evidence_digest: permit.posture_evidence_digest(),
        posture_reference_values_digest: permit.posture_reference_values_digest(),
        posture_appraisal_policy_digest: permit.posture_appraisal_policy_digest(),
        posture_challenge_digest: permit.posture_challenge_digest(),
        posture_verifier_trust_head: permit.posture_trust_head(),
        posture_expires_at_unix_s: permit.posture_expires_at_unix_s(),
        host_preflight_at_unix_s: permit.validated_at_unix_s(),
        send_not_after_unix_s,
    };
    let envelope_digest = envelope.digest()?;
    Ok(PreparedDeviceEgress {
        permit,
        envelope,
        envelope_digest,
    })
}

/// Device-local semantic configuration independent of host claims.
///
/// A device can provision this from secure local configuration/firmware. It does not
/// trust the envelope's policy digest, firmware or operation merely because they are
/// present on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceEnforcementConfigV1 {
    pub schema_version: u16,
    pub device: ResourceRef,
    pub operation: Operation,
    pub exact_policy_digest: Digest32,
    pub minimum_policy_registry_sequence: u64,
    pub safety: SafetyEnvelope,
    pub maximum_envelope_lifetime_s: u64,
}

impl DeviceEnforcementConfigV1 {
    pub fn validate(&self) -> Result<(), DeviceProtocolError> {
        if self.schema_version != DEVICE_ENFORCEMENT_CONFIG_SCHEMA_VERSION {
            return Err(DeviceProtocolError::UnsupportedDeviceConfigSchema);
        }
        if self.exact_policy_digest == Digest32([0; 32]) {
            return Err(DeviceProtocolError::ZeroSecurityDigest);
        }
        if self.minimum_policy_registry_sequence == 0 {
            return Err(DeviceProtocolError::PolicyRegistrySequenceZero);
        }
        if self.safety.device != self.device || self.safety.operation != self.operation {
            return Err(DeviceProtocolError::LocalSafetyBindingMismatch);
        }
        if self.safety.allowed_firmware.is_empty()
            || self
                .safety
                .parameter_ranges
                .values()
                .any(|range| !range.is_valid())
            || self
                .safety
                .required_observations
                .values()
                .any(|range| !range.is_valid())
        {
            return Err(DeviceProtocolError::MalformedLocalSafetyEnvelope);
        }
        if self.maximum_envelope_lifetime_s == 0
            || self.maximum_envelope_lifetime_s > MAX_DEVICE_ENVELOPE_LIFETIME_S
        {
            return Err(DeviceProtocolError::InvalidDeviceEnvelopeLifetime);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, DeviceProtocolError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(DEVICE_CONFIG_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        update_string(&mut h, &self.device.0);
        update_string(&mut h, &self.operation.0);
        update_digest(&mut h, self.exact_policy_digest);
        h.update(&self.minimum_policy_registry_sequence.to_be_bytes());
        update_digest(&mut h, self.safety.digest());
        h.update(&self.maximum_envelope_lifetime_s.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceSemanticHead {
    pub generation: u64,
    pub digest: Digest32,
}

/// Device-local replay journal. Persist this before any later actuator authority is minted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceSemanticCheckpointV1 {
    pub schema_version: u16,
    pub generation: u64,
    pub previous_checkpoint_digest: Option<Digest32>,
    pub device: ResourceRef,
    pub highest_accepted_sequence: Option<u64>,
    pub last_envelope_digest: Option<Digest32>,
}

impl DeviceSemanticCheckpointV1 {
    pub fn genesis(device: ResourceRef) -> Self {
        Self {
            schema_version: DEVICE_SEMANTIC_CHECKPOINT_SCHEMA_VERSION,
            generation: 0,
            previous_checkpoint_digest: None,
            device,
            highest_accepted_sequence: None,
            last_envelope_digest: None,
        }
    }

    pub fn validate(&self) -> Result<(), DeviceProtocolError> {
        if self.schema_version != DEVICE_SEMANTIC_CHECKPOINT_SCHEMA_VERSION {
            return Err(DeviceProtocolError::UnsupportedDeviceCheckpointSchema);
        }
        if self.generation == 0 {
            if self.previous_checkpoint_digest.is_some()
                || self.highest_accepted_sequence.is_some()
                || self.last_envelope_digest.is_some()
            {
                return Err(DeviceProtocolError::MalformedDeviceGenesis);
            }
        } else if self.previous_checkpoint_digest.is_none()
            || self.highest_accepted_sequence.is_none()
            || self.last_envelope_digest.is_none()
        {
            return Err(DeviceProtocolError::IncompleteDeviceCheckpoint);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, DeviceProtocolError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(DEVICE_CHECKPOINT_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.generation.to_be_bytes());
        optional_digest(&mut h, self.previous_checkpoint_digest);
        update_string(&mut h, &self.device.0);
        optional_u64(&mut h, self.highest_accepted_sequence);
        optional_digest(&mut h, self.last_envelope_digest);
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    pub fn head(&self) -> Result<DeviceSemanticHead, DeviceProtocolError> {
        Ok(DeviceSemanticHead {
            generation: self.generation,
            digest: self.digest()?,
        })
    }

    pub fn verify_as_trusted_head(
        &self,
        trusted: DeviceSemanticHead,
    ) -> Result<(), DeviceProtocolError> {
        if self.head()? != trusted {
            return Err(DeviceProtocolError::TrustedDeviceHeadMismatch);
        }
        Ok(())
    }
}

/// Pending semantic acceptance waiting for the exact device replay journal to be durable.
#[derive(Debug)]
pub struct PendingSemanticAcceptance {
    envelope: PhysicalEffectEnvelopeV1,
    envelope_digest: Digest32,
    checkpoint: DeviceSemanticCheckpointV1,
    expected_head: DeviceSemanticHead,
}

impl PendingSemanticAcceptance {
    pub fn checkpoint(&self) -> &DeviceSemanticCheckpointV1 {
        &self.checkpoint
    }

    pub const fn expected_head(&self) -> DeviceSemanticHead {
        self.expected_head
    }

    pub fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    pub fn confirm_persisted(
        self,
        durable_head: DeviceSemanticHead,
    ) -> Result<SemanticallyAcceptedEffect, Box<PendingSemanticAcceptance>> {
        if durable_head != self.expected_head {
            return Err(Box::new(self));
        }
        Ok(SemanticallyAcceptedEffect {
            envelope: self.envelope,
            envelope_digest: self.envelope_digest,
            device_head: self.expected_head,
        })
    }
}

/// Opaque result of device semantic checks and durable replay-state advancement.
///
/// This is explicitly **not an actuator permit**. It proves no transport identity
/// and no physical interlock state. A future product adapter must combine it with an
/// authenticated Xenia session/message before exposing actuator I/O.
#[derive(Debug)]
pub struct SemanticallyAcceptedEffect {
    envelope: PhysicalEffectEnvelopeV1,
    envelope_digest: Digest32,
    device_head: DeviceSemanticHead,
}

impl SemanticallyAcceptedEffect {
    pub fn command(&self) -> &DeviceCommand {
        &self.envelope.command
    }

    pub fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    pub const fn device_head(&self) -> DeviceSemanticHead {
        self.device_head
    }
}

/// Validate semantic content against device-local configuration/state and burn the
/// sequence in a successor checkpoint. This does not authenticate the envelope.
pub fn prepare_semantic_acceptance(
    envelope: PhysicalEffectEnvelopeV1,
    config: &DeviceEnforcementConfigV1,
    local_runtime: &DeviceRuntimeState,
    current_checkpoint: &DeviceSemanticCheckpointV1,
    trusted_current_head: DeviceSemanticHead,
    now_unix_s: u64,
) -> Result<PendingSemanticAcceptance, DeviceProtocolError> {
    envelope.validate_structure()?;
    config.validate()?;
    current_checkpoint.verify_as_trusted_head(trusted_current_head)?;
    if current_checkpoint.device != config.device {
        return Err(DeviceProtocolError::CheckpointDeviceMismatch);
    }
    validate_envelope_against_device(&envelope, config, local_runtime, now_unix_s)?;

    let sequence = envelope.command.sequence;
    if current_checkpoint
        .highest_accepted_sequence
        .is_some_and(|highest| sequence <= highest)
    {
        return Err(DeviceProtocolError::DeviceSequenceReplay {
            proposed: sequence,
            highest: current_checkpoint.highest_accepted_sequence.unwrap_or_default(),
        });
    }
    if local_runtime
        .last_accepted_sequence
        .is_some_and(|highest| sequence <= highest)
    {
        return Err(DeviceProtocolError::RuntimeSequenceReplay {
            proposed: sequence,
            highest: local_runtime.last_accepted_sequence.unwrap_or_default(),
        });
    }

    let envelope_digest = envelope.digest()?;
    let generation = current_checkpoint
        .generation
        .checked_add(1)
        .ok_or(DeviceProtocolError::DeviceGenerationOverflow)?;
    let checkpoint = DeviceSemanticCheckpointV1 {
        schema_version: DEVICE_SEMANTIC_CHECKPOINT_SCHEMA_VERSION,
        generation,
        previous_checkpoint_digest: Some(current_checkpoint.digest()?),
        device: config.device.clone(),
        highest_accepted_sequence: Some(sequence),
        last_envelope_digest: Some(envelope_digest),
    };
    let expected_head = checkpoint.head()?;

    Ok(PendingSemanticAcceptance {
        envelope,
        envelope_digest,
        checkpoint,
        expected_head,
    })
}

fn validate_envelope_against_device(
    envelope: &PhysicalEffectEnvelopeV1,
    config: &DeviceEnforcementConfigV1,
    local_runtime: &DeviceRuntimeState,
    now_unix_s: u64,
) -> Result<(), DeviceProtocolError> {
    if now_unix_s < envelope.host_preflight_at_unix_s
        || now_unix_s > envelope.send_not_after_unix_s
    {
        return Err(DeviceProtocolError::EnvelopeNotFresh);
    }
    let envelope_lifetime = envelope
        .send_not_after_unix_s
        .checked_sub(envelope.host_preflight_at_unix_s)
        .ok_or(DeviceProtocolError::InvalidEgressWindow)?;
    if envelope_lifetime > config.maximum_envelope_lifetime_s {
        return Err(DeviceProtocolError::EnvelopeLifetimeExceedsDevicePolicy);
    }
    if envelope.command.device != config.device {
        return Err(DeviceProtocolError::EnvelopeDeviceMismatch);
    }
    if envelope.command.operation != config.operation {
        return Err(DeviceProtocolError::EnvelopeOperationMismatch);
    }
    if envelope.policy_digest != config.exact_policy_digest {
        return Err(DeviceProtocolError::EnvelopePolicyMismatch);
    }
    if envelope.policy_registry_head.sequence < config.minimum_policy_registry_sequence {
        return Err(DeviceProtocolError::PolicyRegistryGenerationTooOld);
    }
    if local_runtime.running_firmware != envelope.command.expected_firmware {
        return Err(DeviceProtocolError::LocalFirmwareMismatch);
    }
    if !config
        .safety
        .allowed_firmware
        .contains(&local_runtime.running_firmware)
    {
        return Err(DeviceProtocolError::LocalFirmwareNotAllowed);
    }

    for name in config.safety.parameter_ranges.keys() {
        if !envelope.command.parameters.contains_key(name) {
            return Err(DeviceProtocolError::MissingCommandParameter(name.clone()));
        }
    }
    for (name, value) in &envelope.command.parameters {
        let Some(range) = config.safety.parameter_ranges.get(name) else {
            return Err(DeviceProtocolError::UnknownCommandParameter(name.clone()));
        };
        if !range.contains(*value) {
            return Err(DeviceProtocolError::CommandParameterOutOfRange {
                name: name.clone(),
                value: *value,
            });
        }
    }
    for (name, range) in &config.safety.required_observations {
        let Some(value) = local_runtime.observations.get(name) else {
            return Err(DeviceProtocolError::MissingLocalObservation(name.clone()));
        };
        if !range.contains(*value) {
            return Err(DeviceProtocolError::LocalObservationOutOfRange {
                name: name.clone(),
                value: *value,
            });
        }
    }
    Ok(())
}

fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

fn update_digest(h: &mut blake3::Hasher, Digest32(value): Digest32) {
    h.update(&value);
}

fn optional_digest(h: &mut blake3::Hasher, value: Option<Digest32>) {
    match value {
        Some(value) => {
            h.update(&[1]);
            update_digest(h, value);
        }
        None => {
            h.update(&[0]);
        }
    }
}

fn optional_u64(h: &mut blake3::Hasher, value: Option<u64>) {
    match value {
        Some(value) => {
            h.update(&[1]);
            h.update(&value.to_be_bytes());
        }
        None => {
            h.update(&[0]);
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DeviceProtocolError {
    #[error("unsupported physical-effect envelope schema")]
    UnsupportedEnvelopeSchema,
    #[error("malformed physical command")]
    MalformedCommand,
    #[error("invalid host preflight timestamp")]
    InvalidHostPreflightTime,
    #[error("invalid host-to-device egress window")]
    InvalidEgressWindow,
    #[error("envelope outlives command authority or verified posture")]
    EgressOutlivesAuthorityOrPosture,
    #[error("zero security commitment in physical-effect envelope/config")]
    ZeroSecurityDigest,
    #[error("unsupported device enforcement config schema")]
    UnsupportedDeviceConfigSchema,
    #[error("device policy registry sequence must be non-zero")]
    PolicyRegistrySequenceZero,
    #[error("device-local safety policy binds another device/operation")]
    LocalSafetyBindingMismatch,
    #[error("malformed device-local safety envelope")]
    MalformedLocalSafetyEnvelope,
    #[error("invalid device envelope lifetime ceiling")]
    InvalidDeviceEnvelopeLifetime,
    #[error("unsupported device semantic checkpoint schema")]
    UnsupportedDeviceCheckpointSchema,
    #[error("malformed generation-zero device checkpoint")]
    MalformedDeviceGenesis,
    #[error("non-genesis device checkpoint missing replay commitments")]
    IncompleteDeviceCheckpoint,
    #[error("persisted device checkpoint does not match trusted head")]
    TrustedDeviceHeadMismatch,
    #[error("device checkpoint belongs to another device")]
    CheckpointDeviceMismatch,
    #[error("device semantic generation overflow")]
    DeviceGenerationOverflow,
    #[error("envelope is outside its device receive window")]
    EnvelopeNotFresh,
    #[error("envelope lifetime exceeds device-local ceiling")]
    EnvelopeLifetimeExceedsDevicePolicy,
    #[error("envelope targets another device")]
    EnvelopeDeviceMismatch,
    #[error("envelope requests another semantic operation")]
    EnvelopeOperationMismatch,
    #[error("envelope policy commitment is not locally configured")]
    EnvelopePolicyMismatch,
    #[error("host policy registry generation is older than device minimum")]
    PolicyRegistryGenerationTooOld,
    #[error("local running firmware differs from command expectation")]
    LocalFirmwareMismatch,
    #[error("local running firmware is outside device safety policy")]
    LocalFirmwareNotAllowed,
    #[error("missing command parameter {0}")]
    MissingCommandParameter(String),
    #[error("unknown command parameter {0}")]
    UnknownCommandParameter(String),
    #[error("command parameter {name} is outside local range: {value}")]
    CommandParameterOutOfRange { name: String, value: i64 },
    #[error("missing local safety observation {0}")]
    MissingLocalObservation(String),
    #[error("local observation {name} is outside safe range: {value}")]
    LocalObservationOutOfRange { name: String, value: i64 },
    #[error("device semantic sequence replay: proposed {proposed} <= durable {highest}")]
    DeviceSequenceReplay { proposed: u64, highest: u64 },
    #[error("runtime/device sequence replay: proposed {proposed} <= runtime {highest}")]
    RuntimeSequenceReplay { proposed: u64, highest: u64 },
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use symthaea_authority::RiskBudget;
    use symthaea_iot_authority::{
        DEVICE_COMMAND_SCHEMA_VERSION, InclusiveRangeI64, SAFETY_ENVELOPE_SCHEMA_VERSION,
    };
    use symthaea_iot_durable_runtime::DurableIoTHead;
    use symthaea_iot_posture::VerifierTrustHead;

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn safety() -> SafetyEnvelope {
        SafetyEnvelope {
            schema_version: SAFETY_ENVELOPE_SCHEMA_VERSION,
            policy_id: "device-local-safe-open".into(),
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            allowed_firmware: BTreeSet::from([digest(7)]),
            parameter_ranges: BTreeMap::from([(
                "duration_ms".into(),
                InclusiveRangeI64 {
                    min: 1_000,
                    max: 120_000,
                },
            )]),
            required_observations: BTreeMap::from([(
                "pressure_x100".into(),
                InclusiveRangeI64 {
                    min: 100,
                    max: 350_000,
                },
            )]),
        }
    }

    fn config() -> DeviceEnforcementConfigV1 {
        DeviceEnforcementConfigV1 {
            schema_version: DEVICE_ENFORCEMENT_CONFIG_SCHEMA_VERSION,
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            exact_policy_digest: digest(20),
            minimum_policy_registry_sequence: 5,
            safety: safety(),
            maximum_envelope_lifetime_s: 5,
        }
    }

    fn runtime(last: Option<u64>) -> DeviceRuntimeState {
        DeviceRuntimeState {
            running_firmware: digest(7),
            last_accepted_sequence: last,
            observations: BTreeMap::from([("pressure_x100".into(), 210_000)]),
        }
    }

    fn envelope(sequence: u64) -> PhysicalEffectEnvelopeV1 {
        PhysicalEffectEnvelopeV1 {
            schema_version: PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION,
            command: DeviceCommand {
                schema_version: DEVICE_COMMAND_SCHEMA_VERSION,
                command_id: format!("cmd-{sequence}"),
                actor: symthaea_authority::PrincipalId("agent:irrigation".into()),
                executor: symthaea_authority::PrincipalId("gateway:field-a".into()),
                task: None,
                device: ResourceRef("iot:valve:72".into()),
                operation: Operation("valve.open".into()),
                expected_firmware: digest(7),
                sequence,
                issued_at_unix_s: 4_990,
                expires_at_unix_s: 5_010,
                parameters: BTreeMap::from([("duration_ms".into(), 60_000)]),
            },
            proposal_digest: digest(10),
            policy_digest: digest(20),
            policy_registry_head: ActuationPolicyHead {
                sequence: 5,
                digest: digest(21),
            },
            durable_host_head: DurableIoTHead {
                action_head: symthaea_action_checkpoint::CheckpointHead {
                    sequence: 9,
                    digest: digest(30),
                },
                digest: digest(31),
            },
            posture_result_digest: digest(40),
            posture_evidence_digest: digest(41),
            posture_reference_values_digest: digest(42),
            posture_appraisal_policy_digest: digest(43),
            posture_challenge_digest: digest(44),
            posture_verifier_trust_head: VerifierTrustHead {
                sequence: 3,
                digest: digest(45),
            },
            posture_expires_at_unix_s: 5_010,
            host_preflight_at_unix_s: 5_000,
            send_not_after_unix_s: 5_005,
        }
    }

    #[test]
    fn semantic_acceptance_burns_sequence_before_token_exists() {
        let cfg = config();
        let state = DeviceSemanticCheckpointV1::genesis(cfg.device.clone());
        let head = state.head().unwrap();
        let pending = prepare_semantic_acceptance(
            envelope(7),
            &cfg,
            &runtime(None),
            &state,
            head,
            5_001,
        )
        .unwrap();
        assert_eq!(pending.checkpoint().highest_accepted_sequence, Some(7));
        assert_eq!(pending.checkpoint().generation, 1);
        let accepted = pending
            .confirm_persisted(pending.expected_head())
            .expect("exact device head");
        assert_eq!(accepted.command().sequence, 7);
    }

    #[test]
    fn wrong_device_head_cannot_mint_semantic_token() {
        let cfg = config();
        let state = DeviceSemanticCheckpointV1::genesis(cfg.device.clone());
        let pending = prepare_semantic_acceptance(
            envelope(7),
            &cfg,
            &runtime(None),
            &state,
            state.head().unwrap(),
            5_001,
        )
        .unwrap();
        let wrong = DeviceSemanticHead {
            generation: pending.expected_head().generation,
            digest: digest(99),
        };
        assert!(pending.confirm_persisted(wrong).is_err());
    }

    #[test]
    fn persisted_sequence_cannot_be_replayed() {
        let cfg = config();
        let state = DeviceSemanticCheckpointV1::genesis(cfg.device.clone());
        let first = prepare_semantic_acceptance(
            envelope(7),
            &cfg,
            &runtime(None),
            &state,
            state.head().unwrap(),
            5_001,
        )
        .unwrap();
        let next = first.checkpoint().clone();
        let error = prepare_semantic_acceptance(
            envelope(7),
            &cfg,
            &runtime(None),
            &next,
            next.head().unwrap(),
            5_001,
        )
        .unwrap_err();
        assert!(matches!(error, DeviceProtocolError::DeviceSequenceReplay { .. }));
    }

    #[test]
    fn expired_or_wrong_policy_envelope_fails_closed() {
        let cfg = config();
        let state = DeviceSemanticCheckpointV1::genesis(cfg.device.clone());
        let head = state.head().unwrap();
        assert!(matches!(
            prepare_semantic_acceptance(
                envelope(7),
                &cfg,
                &runtime(None),
                &state,
                head,
                5_006,
            ),
            Err(DeviceProtocolError::EnvelopeNotFresh)
        ));

        let mut wrong = envelope(8);
        wrong.policy_digest = digest(99);
        assert!(matches!(
            prepare_semantic_acceptance(
                wrong,
                &cfg,
                &runtime(None),
                &state,
                head,
                5_001,
            ),
            Err(DeviceProtocolError::EnvelopePolicyMismatch)
        ));
    }

    #[test]
    fn unsafe_local_observation_dominates_host_claims() {
        let cfg = config();
        let state = DeviceSemanticCheckpointV1::genesis(cfg.device.clone());
        let mut local = runtime(None);
        local.observations.insert("pressure_x100".into(), 500_000);
        assert!(matches!(
            prepare_semantic_acceptance(
                envelope(7),
                &cfg,
                &local,
                &state,
                state.head().unwrap(),
                5_001,
            ),
            Err(DeviceProtocolError::LocalObservationOutOfRange { .. })
        ));
    }

    #[test]
    fn envelope_commitment_changes_with_posture_or_deadline() {
        let a = envelope(7);
        let mut b = a.clone();
        b.posture_result_digest = digest(99);
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
        let mut c = a.clone();
        c.send_not_after_unix_s -= 1;
        assert_ne!(a.digest().unwrap(), c.digest().unwrap());
    }

    #[test]
    fn config_commitment_changes_with_local_policy() {
        let a = config();
        let mut b = a.clone();
        b.exact_policy_digest = digest(22);
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }

    #[test]
    fn risk_budget_type_remains_unrelated_to_device_semantic_auth() {
        // Compile-time/documentation guard: device semantic acceptance does not
        // receive a host RiskBudget or CapabilityGrant as an authentication proxy.
        let _ = RiskBudget::default();
    }
}
