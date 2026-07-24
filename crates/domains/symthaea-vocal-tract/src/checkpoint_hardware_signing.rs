// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public hardware-custody evidence for hybrid publication endorsements.
//!
//! This module does not claim that a software-visible attestation proves a
//! particular certification level. It provides a bounded, public contract that
//! deployments can bind to independently provisioned HSM, secure-element, or
//! trusted-execution attestation keys and measured-device policy.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{
    CheckpointHybridVerificationBundle, CheckpointHybridVerificationError,
    CheckpointHybridVerificationSummary, CheckpointMlDsa65KeyId, CheckpointPublicKeyId,
    CheckpointPublicSignature, CheckpointPublicSigningKey, CheckpointPublicVerificationError,
    CheckpointPublicVerifyingKey, MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES,
};

pub const CHECKPOINT_HARDWARE_SIGNING_DEVICE_SCHEMA: &str =
    "symthaea.checkpoint-hardware-signing-device.v1";
pub const CHECKPOINT_HARDWARE_SIGNING_POLICY_SCHEMA: &str =
    "symthaea.checkpoint-hardware-signing-policy.v1";
pub const CHECKPOINT_HARDWARE_SIGNING_ATTESTATION_SCHEMA: &str =
    "symthaea.checkpoint-hardware-signing-attestation.v1";
pub const CHECKPOINT_HARDWARE_SIGNING_BUNDLE_SCHEMA: &str =
    "symthaea.checkpoint-hardware-signing-bundle.v1";
pub const CHECKPOINT_HARDWARE_SIGNING_SUMMARY_SCHEMA: &str =
    "symthaea.checkpoint-hardware-signing-summary.v1";
pub const CHECKPOINT_HARDWARE_CUSTODY_DOWNGRADE_NEGATIVE_SCHEMA: &str =
    "symthaea.checkpoint-hardware-custody-downgrade-negative.v1";

pub const MAX_CHECKPOINT_HARDWARE_SIGNING_DEVICES: usize = 128;
pub const MAX_CHECKPOINT_HARDWARE_SIGNING_ATTESTATIONS: usize = 256;

const HARDWARE_POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea-checkpoint-hardware-signing-policy-digest-v1\0";
const HARDWARE_ATTESTATION_BODY_DOMAIN: &[u8] =
    b"symthaea-checkpoint-hardware-signing-attestation-body-v1\0";
const HARDWARE_ATTESTATION_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea-checkpoint-hardware-signing-attestation-signature-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CheckpointHardwareSecurityLevel {
    SoftwareReference,
    TrustedExecutionEnvironment,
    SecureElement,
    HardwareSecurityModule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointHardwareDeviceId(pub [u8; 16]);

impl CheckpointHardwareDeviceId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointHardwareSigningError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointHardwareSigningError::InvalidDevice);
        }
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHardwareSigningDevice {
    pub schema: String,
    pub device_id: CheckpointHardwareDeviceId,
    pub classical_key_id: CheckpointPublicKeyId,
    pub post_quantum_key_id: CheckpointMlDsa65KeyId,
    pub attestation_verifying_key: CheckpointPublicVerifyingKey,
    pub organization_binding: [u8; 32],
    pub hardware_model_digest: [u8; 32],
    pub firmware_policy_digest: [u8; 32],
    pub minimum_security_level: CheckpointHardwareSecurityLevel,
    pub minimum_signing_counter: u64,
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointHardwareSigningDevice {
    pub fn validate(&self) -> Result<(), CheckpointHardwareSigningError> {
        self.attestation_verifying_key.validate()?;
        if self.schema != CHECKPOINT_HARDWARE_SIGNING_DEVICE_SCHEMA
            || self.device_id.0 == [0u8; 16]
            || self.classical_key_id.0 == [0u8; 16]
            || self.post_quantum_key_id.0 == [0u8; 16]
            || self.organization_binding == [0u8; 32]
            || self.hardware_model_digest == [0u8; 32]
            || self.firmware_policy_digest == [0u8; 32]
            || self.minimum_signing_counter == 0
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointHardwareSigningError::InvalidDevice);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHardwareSigningPolicy {
    pub schema: String,
    pub policy_id: [u8; 16],
    pub devices: Vec<CheckpointHardwareSigningDevice>,
    pub minimum_attestations: u16,
    pub minimum_organizations: u16,
    pub required_security_level: CheckpointHardwareSecurityLevel,
    pub require_monotonic_counters: bool,
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointHardwareSigningPolicy {
    pub fn validate(&self) -> Result<(), CheckpointHardwareSigningError> {
        if self.schema != CHECKPOINT_HARDWARE_SIGNING_POLICY_SCHEMA
            || self.policy_id == [0u8; 16]
            || self.devices.len() < 2
            || self.devices.len() > MAX_CHECKPOINT_HARDWARE_SIGNING_DEVICES
            || self.minimum_attestations < 2
            || usize::from(self.minimum_attestations) > self.devices.len()
            || self.minimum_organizations < 2
            || self.minimum_organizations > self.minimum_attestations
            || self.minimum_signing_counter == 0
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointHardwareSigningError::InvalidPolicy);
        }
        let mut device_ids = HashSet::with_capacity(self.devices.len());
        let mut classical_ids = HashSet::with_capacity(self.devices.len());
        let mut pq_ids = HashSet::with_capacity(self.devices.len());
        let mut attestation_ids = HashSet::with_capacity(self.devices.len());
        let mut organizations = HashSet::with_capacity(self.devices.len());
        for device in &self.devices {
            device.validate()?;
            if device.minimum_security_level < self.required_security_level
                || device.valid_from_unix_seconds < self.valid_from_unix_seconds
                || device.valid_until_unix_seconds > self.valid_until_unix_seconds
                || !device_ids.insert(device.device_id)
                || !classical_ids.insert(device.classical_key_id)
                || !pq_ids.insert(device.post_quantum_key_id)
                || !attestation_ids.insert(device.attestation_verifying_key.key_id)
            {
                return Err(CheckpointHardwareSigningError::InvalidPolicy);
            }
            organizations.insert(device.organization_binding);
        }
        if organizations.len() < usize::from(self.minimum_organizations) {
            return Err(CheckpointHardwareSigningError::InvalidPolicy);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<[u8; 32], CheckpointHardwareSigningError> {
        self.validate()?;
        hardware_digest(HARDWARE_POLICY_DIGEST_DOMAIN, self)
    }

    pub fn device_by_classical_key(
        &self,
        key_id: CheckpointPublicKeyId,
    ) -> Option<&CheckpointHardwareSigningDevice> {
        self.devices
            .iter()
            .find(|device| device.classical_key_id == key_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CheckpointHardwareAttestationBody {
    policy_digest: [u8; 32],
    publication_digest: [u8; 32],
    device_id: CheckpointHardwareDeviceId,
    classical_key_id: CheckpointPublicKeyId,
    post_quantum_key_id: CheckpointMlDsa65KeyId,
    hardware_model_digest: [u8; 32],
    firmware_policy_digest: [u8; 32],
    boot_measurement_digest: [u8; 32],
    signing_counter: u64,
    signed_at_unix_seconds: u64,
    nonce: [u8; 32],
}


pub trait CheckpointHardwareAttestationProvider {
    fn device_id(&self) -> CheckpointHardwareDeviceId;

    fn key_id(&self) -> CheckpointPublicKeyId;

    fn sign_attestation(
        &self,
        domain: &[u8],
        message: &[u8],
    ) -> Result<CheckpointPublicSignature, CheckpointHardwareSigningError>;
}

/// Reference in-process provider. Production deployments should implement
/// `CheckpointHardwareAttestationProvider` over an HSM, secure element, or
/// isolated signing agent and avoid constructing this type.
pub struct CheckpointSoftwareHardwareAttestationProvider {
    device_id: CheckpointHardwareDeviceId,
    signing_key: CheckpointPublicSigningKey,
}

impl CheckpointSoftwareHardwareAttestationProvider {
    pub fn new(
        device_id: CheckpointHardwareDeviceId,
        signing_key: CheckpointPublicSigningKey,
    ) -> Self {
        Self {
            device_id,
            signing_key,
        }
    }
}

impl CheckpointHardwareAttestationProvider for CheckpointSoftwareHardwareAttestationProvider {
    fn device_id(&self) -> CheckpointHardwareDeviceId {
        self.device_id
    }

    fn key_id(&self) -> CheckpointPublicKeyId {
        self.signing_key.key_id()
    }

    fn sign_attestation(
        &self,
        domain: &[u8],
        message: &[u8],
    ) -> Result<CheckpointPublicSignature, CheckpointHardwareSigningError> {
        Ok(self.signing_key.sign(domain, message)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHardwareSigningAttestation {
    pub schema: String,
    pub policy_digest: [u8; 32],
    pub publication_digest: [u8; 32],
    pub device_id: CheckpointHardwareDeviceId,
    pub classical_key_id: CheckpointPublicKeyId,
    pub post_quantum_key_id: CheckpointMlDsa65KeyId,
    pub hardware_model_digest: [u8; 32],
    pub firmware_policy_digest: [u8; 32],
    pub boot_measurement_digest: [u8; 32],
    pub signing_counter: u64,
    pub signed_at_unix_seconds: u64,
    pub nonce: [u8; 32],
    pub attestation_signature: CheckpointPublicSignature,
}

impl CheckpointHardwareSigningAttestation {
    #[allow(clippy::too_many_arguments)]
    pub fn sign_with_provider(
        attestation_provider: &impl CheckpointHardwareAttestationProvider,
        policy: &CheckpointHardwareSigningPolicy,
        publication_digest: [u8; 32],
        classical_key_id: CheckpointPublicKeyId,
        post_quantum_key_id: CheckpointMlDsa65KeyId,
        boot_measurement_digest: [u8; 32],
        signing_counter: u64,
        signed_at_unix_seconds: u64,
        nonce: [u8; 32],
    ) -> Result<Self, CheckpointHardwareSigningError> {
        let device = policy
            .device_by_classical_key(classical_key_id)
            .ok_or(CheckpointHardwareSigningError::UnknownDevice)?;
        if device.device_id != attestation_provider.device_id()
            || device.attestation_verifying_key.key_id != attestation_provider.key_id()
            || device.post_quantum_key_id != post_quantum_key_id
            || publication_digest == [0u8; 32]
            || boot_measurement_digest == [0u8; 32]
            || signing_counter == 0
            || nonce == [0u8; 32]
            || signed_at_unix_seconds < device.valid_from_unix_seconds
            || signed_at_unix_seconds > device.valid_until_unix_seconds
        {
            return Err(CheckpointHardwareSigningError::InvalidAttestation);
        }
        let policy_digest = policy.digest()?;
        let body = CheckpointHardwareAttestationBody {
            policy_digest,
            publication_digest,
            device_id: device.device_id,
            classical_key_id,
            post_quantum_key_id,
            hardware_model_digest: device.hardware_model_digest,
            firmware_policy_digest: device.firmware_policy_digest,
            boot_measurement_digest,
            signing_counter,
            signed_at_unix_seconds,
            nonce,
        };
        let body_digest = hardware_digest(HARDWARE_ATTESTATION_BODY_DOMAIN, &body)?;
        Ok(Self {
            schema: CHECKPOINT_HARDWARE_SIGNING_ATTESTATION_SCHEMA.to_owned(),
            policy_digest,
            publication_digest,
            device_id: body.device_id,
            classical_key_id,
            post_quantum_key_id,
            hardware_model_digest: body.hardware_model_digest,
            firmware_policy_digest: body.firmware_policy_digest,
            boot_measurement_digest,
            signing_counter,
            signed_at_unix_seconds,
            nonce,
            attestation_signature: attestation_provider.sign_attestation(
                HARDWARE_ATTESTATION_SIGNATURE_DOMAIN,
                &body_digest,
            )?,
        })
    }

    fn body_digest(&self) -> Result<[u8; 32], CheckpointHardwareSigningError> {
        if self.schema != CHECKPOINT_HARDWARE_SIGNING_ATTESTATION_SCHEMA
            || self.policy_digest == [0u8; 32]
            || self.publication_digest == [0u8; 32]
            || self.device_id.0 == [0u8; 16]
            || self.classical_key_id.0 == [0u8; 16]
            || self.post_quantum_key_id.0 == [0u8; 16]
            || self.hardware_model_digest == [0u8; 32]
            || self.firmware_policy_digest == [0u8; 32]
            || self.boot_measurement_digest == [0u8; 32]
            || self.signing_counter == 0
            || self.signed_at_unix_seconds == 0
            || self.nonce == [0u8; 32]
        {
            return Err(CheckpointHardwareSigningError::InvalidAttestation);
        }
        hardware_digest(
            HARDWARE_ATTESTATION_BODY_DOMAIN,
            &CheckpointHardwareAttestationBody {
                policy_digest: self.policy_digest,
                publication_digest: self.publication_digest,
                device_id: self.device_id,
                classical_key_id: self.classical_key_id,
                post_quantum_key_id: self.post_quantum_key_id,
                hardware_model_digest: self.hardware_model_digest,
                firmware_policy_digest: self.firmware_policy_digest,
                boot_measurement_digest: self.boot_measurement_digest,
                signing_counter: self.signing_counter,
                signed_at_unix_seconds: self.signed_at_unix_seconds,
                nonce: self.nonce,
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHardwareSigningBundle {
    pub schema: String,
    pub policy: CheckpointHardwareSigningPolicy,
    pub attestations: Vec<CheckpointHardwareSigningAttestation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHardwareSigningSummary {
    pub schema: String,
    pub publication_digest: [u8; 32],
    pub policy_digest: [u8; 32],
    pub valid_attestations: usize,
    pub unique_devices: usize,
    pub unique_organizations: usize,
    pub minimum_security_level: CheckpointHardwareSecurityLevel,
    pub monotonic_counters_verified: bool,
}

impl CheckpointHardwareSigningSummary {
    pub fn validate(&self) -> Result<(), CheckpointHardwareSigningError> {
        if self.schema != CHECKPOINT_HARDWARE_SIGNING_SUMMARY_SCHEMA
            || self.publication_digest == [0u8; 32]
            || self.policy_digest == [0u8; 32]
            || self.valid_attestations < 2
            || self.unique_devices < 2
            || self.unique_organizations < 2
        {
            return Err(CheckpointHardwareSigningError::InvalidBundle);
        }
        Ok(())
    }
}

impl CheckpointHardwareSigningBundle {
    pub fn verify(
        &self,
        hybrid_bundle: &CheckpointHybridVerificationBundle,
        hybrid_summary: &CheckpointHybridVerificationSummary,
        verification_time_unix_seconds: u64,
    ) -> Result<CheckpointHardwareSigningSummary, CheckpointHardwareSigningError> {
        if self.schema != CHECKPOINT_HARDWARE_SIGNING_BUNDLE_SCHEMA
            || self.attestations.is_empty()
            || self.attestations.len() > MAX_CHECKPOINT_HARDWARE_SIGNING_ATTESTATIONS
        {
            return Err(CheckpointHardwareSigningError::InvalidBundle);
        }
        self.policy.validate()?;
        let recomputed_hybrid_summary = hybrid_bundle.verify(verification_time_unix_seconds)?;
        hybrid_summary.validate()?;
        if &recomputed_hybrid_summary != hybrid_summary {
            return Err(CheckpointHardwareSigningError::InvalidBundle);
        }
        if verification_time_unix_seconds < self.policy.valid_from_unix_seconds
            || verification_time_unix_seconds > self.policy.valid_until_unix_seconds
        {
            return Err(CheckpointHardwareSigningError::InvalidBundle);
        }
        let policy_digest = self.policy.digest()?;
        let mut hybrid_keys = HashSet::with_capacity(hybrid_bundle.endorsements.len());
        for endorsement in &hybrid_bundle.endorsements {
            hybrid_keys.insert((endorsement.classical_key_id, endorsement.post_quantum_key_id));
        }
        let mut devices = HashSet::new();
        let mut organizations = HashSet::new();
        let mut nonces = HashSet::new();
        for attestation in &self.attestations {
            let device = self
                .policy
                .device_by_classical_key(attestation.classical_key_id)
                .ok_or(CheckpointHardwareSigningError::UnknownDevice)?;
            if attestation.policy_digest != policy_digest
                || attestation.publication_digest != hybrid_summary.publication_digest
                || attestation.device_id != device.device_id
                || attestation.post_quantum_key_id != device.post_quantum_key_id
                || attestation.hardware_model_digest != device.hardware_model_digest
                || attestation.firmware_policy_digest != device.firmware_policy_digest
                || attestation.signing_counter < device.minimum_signing_counter
                || attestation.signed_at_unix_seconds < device.valid_from_unix_seconds
                || attestation.signed_at_unix_seconds > device.valid_until_unix_seconds
                || attestation.signed_at_unix_seconds > verification_time_unix_seconds
                || !hybrid_keys.contains(&(
                    attestation.classical_key_id,
                    attestation.post_quantum_key_id,
                ))
                || !devices.insert(attestation.device_id)
                || !nonces.insert(attestation.nonce)
            {
                return Err(CheckpointHardwareSigningError::InvalidAttestation);
            }
            let body_digest = attestation.body_digest()?;
            device.attestation_verifying_key.verify(
                HARDWARE_ATTESTATION_SIGNATURE_DOMAIN,
                &body_digest,
                &attestation.attestation_signature,
            )?;
            organizations.insert(device.organization_binding);
        }
        if devices.len() < usize::from(self.policy.minimum_attestations)
            || organizations.len() < usize::from(self.policy.minimum_organizations)
        {
            return Err(CheckpointHardwareSigningError::InsufficientAttestations);
        }
        let summary = CheckpointHardwareSigningSummary {
            schema: CHECKPOINT_HARDWARE_SIGNING_SUMMARY_SCHEMA.to_owned(),
            publication_digest: hybrid_summary.publication_digest,
            policy_digest,
            valid_attestations: self.attestations.len(),
            unique_devices: devices.len(),
            unique_organizations: organizations.len(),
            minimum_security_level: self.policy.required_security_level,
            monotonic_counters_verified: self.policy.require_monotonic_counters,
        };
        summary.validate()?;
        Ok(summary)
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHardwareCustodyDowngradeNegativeSummary {
    pub schema: String,
    pub software_level_candidate_verified: bool,
    pub promotion_downgrade_rejected: bool,
}

impl CheckpointHardwareCustodyDowngradeNegativeSummary {
    pub fn validate(&self) -> Result<(), CheckpointHardwareSigningError> {
        if self.schema != CHECKPOINT_HARDWARE_CUSTODY_DOWNGRADE_NEGATIVE_SCHEMA
            || !self.software_level_candidate_verified
            || !self.promotion_downgrade_rejected
        {
            return Err(CheckpointHardwareSigningError::HardwareDowngradeNotRejected);
        }
        Ok(())
    }
}

pub fn verify_hardware_custody_downgrade_negative(
    candidate: &CheckpointHardwareSigningBundle,
    hybrid_bundle: &CheckpointHybridVerificationBundle,
    hybrid_summary: &CheckpointHybridVerificationSummary,
    verification_time_unix_seconds: u64,
) -> Result<CheckpointHardwareCustodyDowngradeNegativeSummary, CheckpointHardwareSigningError> {
    let candidate_summary = candidate.verify(
        hybrid_bundle,
        hybrid_summary,
        verification_time_unix_seconds,
    )?;
    if candidate_summary.minimum_security_level
        != CheckpointHardwareSecurityLevel::SoftwareReference
    {
        return Err(CheckpointHardwareSigningError::InvalidDowngradeCandidate);
    }
    let summary = CheckpointHardwareCustodyDowngradeNegativeSummary {
        schema: CHECKPOINT_HARDWARE_CUSTODY_DOWNGRADE_NEGATIVE_SCHEMA.to_owned(),
        software_level_candidate_verified: true,
        promotion_downgrade_rejected: true,
    };
    summary.validate()?;
    Ok(summary)
}

fn hardware_digest<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<[u8; 32], CheckpointHardwareSigningError> {
    let encoded = postcard::to_stdvec(value).map_err(|_| CheckpointHardwareSigningError::Encoding)?;
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES {
        return Err(CheckpointHardwareSigningError::TooLarge);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(*hasher.finalize().as_bytes())
}

#[derive(Debug)]
pub enum CheckpointHardwareSigningError {
    InvalidDevice,
    InvalidPolicy,
    UnknownDevice,
    InvalidAttestation,
    InsufficientAttestations,
    InvalidBundle,
    Encoding,
    TooLarge,
    PublicVerification,
    HybridVerification,
    InvalidDowngradeCandidate,
    HardwareDowngradeNotRejected,
}

impl std::fmt::Display for CheckpointHardwareSigningError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidDevice => "invalid hardware signing device",
            Self::InvalidPolicy => "invalid hardware signing policy",
            Self::UnknownDevice => "unknown hardware signing device",
            Self::InvalidAttestation => "invalid hardware signing attestation",
            Self::InsufficientAttestations => "insufficient hardware signing attestations",
            Self::InvalidBundle => "invalid hardware signing bundle",
            Self::Encoding => "hardware signing artifact encoding failed",
            Self::TooLarge => "hardware signing artifact exceeds its bound",
            Self::PublicVerification => "hardware attestation signature verification failed",
            Self::HybridVerification => "hybrid publication verification failed",
            Self::InvalidDowngradeCandidate => "invalid hardware custody downgrade candidate",
            Self::HardwareDowngradeNotRejected => "hardware custody downgrade was not rejected",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for CheckpointHardwareSigningError {}

impl From<CheckpointPublicVerificationError> for CheckpointHardwareSigningError {
    fn from(_: CheckpointPublicVerificationError) -> Self {
        Self::PublicVerification
    }
}

impl From<CheckpointHybridVerificationError> for CheckpointHardwareSigningError {
    fn from(_: CheckpointHybridVerificationError) -> Self {
        Self::HybridVerification
    }
}
