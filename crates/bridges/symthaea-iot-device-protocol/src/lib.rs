// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Protocol-neutral physical-effect envelope and device-side semantic acceptance.
//!
//! The serialized envelope is semantic content, not authority. A future Xenia/device
//! transport must authenticate the exact envelope independently. Device semantic
//! acceptance also remains non-authorizing until composed with authenticated transport
//! and device-local physical interlocks.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_action_runtime::GrantAccount;
use symthaea_authority::{CapabilityGrant, Digest32, Operation, ResourceRef};
use symthaea_iot_actuation::ActuationError;
use symthaea_iot_authority::{
    DEVICE_COMMAND_SCHEMA_VERSION, DeviceCommand, DeviceRuntimeState,
    SAFETY_ENVELOPE_SCHEMA_VERSION, SafetyEnvelope,
};
use symthaea_iot_durable_runtime::{
    DurableEffectTransition, DurableIoTHead, DurableUnknownPhysicalEffect,
};
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
        if self.command.schema_version != DEVICE_COMMAND_SCHEMA_VERSION {
            return Err(DeviceProtocolError::UnsupportedCommandSchema);
        }
        if self.command.command_id.is_empty()
            || self.command.sequence == 0
            || self.command.expires_at_unix_s < self.command.issued_at_unix_s
        {
            return Err(DeviceProtocolError::MalformedCommand);
        }
        if self.policy_registry_head.sequence == 0
            || self.posture_verifier_trust_head.sequence == 0
        {
            return Err(DeviceProtocolError::SecurityGenerationZero);
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
            self.policy_registry_head.digest,
            self.durable_host_head.action_head.digest,
            self.durable_host_head.digest,
            self.posture_result_digest,
            self.posture_evidence_digest,
            self.posture_reference_values_digest,
            self.posture_appraisal_policy_digest,
            self.posture_challenge_digest,
            self.posture_verifier_trust_head.digest,
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

    pub fn into_unknown(self) -> DurableUnknownPhysicalEffect {
        self.permit.into_unknown()
    }

    pub fn observed_applied(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<DurableEffectTransition, ActuationError> {
        self.permit.observed_applied(account, grant)
    }

    pub fn proven_not_dispatched(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<DurableEffectTransition, ActuationError> {
        self.permit.proven_not_dispatched(account, grant)
    }
}

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
        if self.safety.schema_version != SAFETY_ENVELOPE_SCHEMA_VERSION {
            return Err(DeviceProtocolError::UnsupportedLocalSafetySchema);
        }
        if self.safety.device != self.device || self.safety.operation != self.operation {
            return Err(DeviceProtocolError::LocalSafetyBindingMismatch);
        }
        if self.safety.policy_id.is_empty()
            || self.safety.allowed_firmware.is_empty()
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

/// Device-local replay journal bound to the exact local enforcement configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceSemanticCheckpointV1 {
    pub schema_version: u16,
    pub generation: u64,
    pub previous_checkpoint_digest: Option<Digest32>,
    pub device: ResourceRef,
    pub config_digest: Digest32,
    pub highest_accepted_sequence: Option<u64>,
    pub last_envelope_digest: Option<Digest32>,
}

impl DeviceSemanticCheckpointV1 {
    pub fn genesis(config: &DeviceEnforcementConfigV1) -> Result<Self, DeviceProtocolError> {
        config.validate()?;
        Ok(Self {
            schema_version: DEVICE_SEMANTIC_CHECKPOINT_SCHEMA_VERSION,
            generation: 0,
            previous_checkpoint_digest: None,
            device: config.device.clone(),
            config_digest: config.digest()?,
            highest_accepted_sequence: None,
            last_envelope_digest: None,
        })
    }

    pub fn validate(&self) -> Result<(), DeviceProtocolError> {
        if self.schema_version != DEVICE_SEMANTIC_CHECKPOINT_SCHEMA_VERSION {
            return Err(DeviceProtocolError::UnsupportedDeviceCheckpointSchema);
        }
        if self.config_digest == Digest32([0; 32]) {
            return Err(DeviceProtocolError::ZeroSecurityDigest);
        }
        if self.generation == 0 {
            if self.previous_checkpoint_digest.is_some()
                || self.highest_accepted_sequence.is_some()
                || self.last_envelope_digest.is_some()
            {
                return Err(DeviceProtocolError::MalformedDeviceGenesis);
            }
        } else {
            if self.previous_checkpoint_digest.is_none() {
                return Err(DeviceProtocolError::IncompleteDeviceCheckpoint);
            }
            match (self.highest_accepted_sequence, self.last_envelope_digest) {
                (None, None) => {}
                (Some(sequence), Some(_)) if sequence > 0 => {}
                _ => return Err(DeviceProtocolError::IncompleteDeviceCheckpoint),
            }
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
        update_digest(&mut h, self.config_digest);
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

    /// Explicitly migrate local device policy without resetting replay state.
    ///
    /// The policy-registry floor may only stay equal or increase. Replacing the exact
    /// host-policy digest requires advancing that floor, preventing a same-generation
    /// policy substitution from being disguised as local configuration maintenance.
    pub fn migrate_config(
        &self,
        trusted_current_head: DeviceSemanticHead,
        current: &DeviceEnforcementConfigV1,
        next: &DeviceEnforcementConfigV1,
    ) -> Result<Self, DeviceProtocolError> {
        self.verify_as_trusted_head(trusted_current_head)?;
        current.validate()?;
        next.validate()?;
        if self.device != current.device || self.config_digest != current.digest()? {
            return Err(DeviceProtocolError::CheckpointConfigMismatch);
        }
        if next.device != current.device {
            return Err(DeviceProtocolError::ConfigMigrationDeviceChanged);
        }
        if next.minimum_policy_registry_sequence < current.minimum_policy_registry_sequence {
            return Err(DeviceProtocolError::ConfigPolicyGenerationRollback);
        }
        if next.exact_policy_digest != current.exact_policy_digest
            && next.minimum_policy_registry_sequence <= current.minimum_policy_registry_sequence
        {
            return Err(DeviceProtocolError::ConfigPolicyChangedWithoutGenerationAdvance);
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(DeviceProtocolError::DeviceGenerationOverflow)?;
        Ok(Self {
            schema_version: DEVICE_SEMANTIC_CHECKPOINT_SCHEMA_VERSION,
            generation,
            previous_checkpoint_digest: Some(self.digest()?),
            device: self.device.clone(),
            config_digest: next.digest()?,
            highest_accepted_sequence: self.highest_accepted_sequence,
            last_envelope_digest: self.last_envelope_digest,
        })
    }
}

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

/// Opaque semantic acceptance after durable replay-state advancement.
///
/// This is not an actuator permit: it carries no transport identity and no proof of a
/// physical interlock. The future device product adapter must compose both separately.
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
    if current_checkpoint.config_digest != config.digest()? {
        return Err(DeviceProtocolError::CheckpointConfigMismatch);
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
        config_digest: current_checkpoint.config_digest,
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
    let lifetime = envelope
        .send_not_after_unix_s
        .checked_sub(envelope.host_preflight_at_unix_s)
        .ok_or(DeviceProtocolError::InvalidEgressWindow)?;
    if lifetime > config.maximum_envelope_lifetime_s {
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
    #[error("unsupported physical command schema")]
    UnsupportedCommandSchema,
    #[error("malformed physical command")]
    MalformedCommand,
    #[error("security generation must be non-zero")]
    SecurityGenerationZero,
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
    #[error("unsupported device-local safety schema")]
    UnsupportedLocalSafetySchema,
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
    #[error("non-genesis device checkpoint is internally inconsistent")]
    IncompleteDeviceCheckpoint,
    #[error("persisted device checkpoint does not match trusted head")]
    TrustedDeviceHeadMismatch,
    #[error("device checkpoint belongs to another device")]
    CheckpointDeviceMismatch,
    #[error("device checkpoint was created under another local configuration")]
    CheckpointConfigMismatch,
    #[error("device semantic generation overflow")]
    DeviceGenerationOverflow,
    #[error("device configuration migration changed physical device identity")]
    ConfigMigrationDeviceChanged,
    #[error("device configuration policy-registry floor rolled backward")]
    ConfigPolicyGenerationRollback,
    #[error("device policy digest changed without advancing policy-registry generation")]
    ConfigPolicyChangedWithoutGenerationAdvance,
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
    use symthaea_iot_authority::InclusiveRangeI64;

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
                symthaea_iot_authority::InclusiveRangeI64 {
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
    fn semantic_acceptance_burns_sequence_and_config() {
        let cfg = config();
        let state = DeviceSemanticCheckpointV1::genesis(&cfg).unwrap();
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
        assert_eq!(pending.checkpoint().config_digest, cfg.digest().unwrap());
        let expected = pending.expected_head();
        let accepted = pending.confirm_persisted(expected).expect("exact device head");
        assert_eq!(accepted.command().sequence, 7);
    }

    #[test]
    fn persisted_sequence_cannot_be_replayed() {
        let cfg = config();
        let state = DeviceSemanticCheckpointV1::genesis(&cfg).unwrap();
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
    fn silent_config_substitution_fails() {
        let cfg = config();
        let state = DeviceSemanticCheckpointV1::genesis(&cfg).unwrap();
        let mut changed = cfg.clone();
        changed.maximum_envelope_lifetime_s = 6;
        assert!(matches!(
            prepare_semantic_acceptance(
                envelope(7),
                &changed,
                &runtime(None),
                &state,
                state.head().unwrap(),
                5_001,
            ),
            Err(DeviceProtocolError::CheckpointConfigMismatch)
        ));
    }

    #[test]
    fn explicit_config_migration_retains_replay_state() {
        let cfg = config();
        let state = DeviceSemanticCheckpointV1::genesis(&cfg).unwrap();
        let first = prepare_semantic_acceptance(
            envelope(7),
            &cfg,
            &runtime(None),
            &state,
            state.head().unwrap(),
            5_001,
        )
        .unwrap();
        let accepted_state = first.checkpoint().clone();
        let mut next_cfg = cfg.clone();
        next_cfg.minimum_policy_registry_sequence = 6;
        next_cfg.exact_policy_digest = digest(22);
        let migrated = accepted_state
            .migrate_config(accepted_state.head().unwrap(), &cfg, &next_cfg)
            .unwrap();
        assert_eq!(migrated.highest_accepted_sequence, Some(7));
        assert_eq!(migrated.config_digest, next_cfg.digest().unwrap());
        assert_eq!(migrated.generation, accepted_state.generation + 1);
    }

    #[test]
    fn policy_change_without_generation_advance_is_rejected() {
        let cfg = config();
        let state = DeviceSemanticCheckpointV1::genesis(&cfg).unwrap();
        let mut next_cfg = cfg.clone();
        next_cfg.exact_policy_digest = digest(22);
        assert!(matches!(
            state.migrate_config(state.head().unwrap(), &cfg, &next_cfg),
            Err(DeviceProtocolError::ConfigPolicyChangedWithoutGenerationAdvance)
        ));
    }

    #[test]
    fn unsafe_local_observation_dominates_host_claims() {
        let cfg = config();
        let state = DeviceSemanticCheckpointV1::genesis(&cfg).unwrap();
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
    fn envelope_commitment_binds_posture_and_deadline() {
        let a = envelope(7);
        let mut b = a.clone();
        b.posture_result_digest = digest(99);
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
        let mut c = a.clone();
        c.send_not_after_unix_s -= 1;
        assert_ne!(a.digest().unwrap(), c.digest().unwrap());
    }
}
