// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Challenge-bound observation of one currently installed safety configuration.
//!
//! [`SafetyConfigurationManifest`] defines what a safety-relevant configuration is.
//! This module does not create a second configuration schema. Instead it defines the
//! canonical raw evidence that one designated configuration observer claims to have
//! observed one exact manifest digest on one node for one exact challenge.
//!
//! This is intentionally not trusted state by itself. A caller must still verify an
//! external signature/attestation over [`SafetyConfigurationObservation::canonical_signing_bytes`],
//! enforce observer-root policy, bind the challenge to the commissioning transaction,
//! and enforce freshness/monotone-generation policy before treating the observation
//! as evidence of what is installed now.

use crate::ConfigurationDigest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SAFETY_CONFIGURATION_OBSERVATION_SCHEMA_V1: &str =
    "symthaea-safety-configuration-observation-v1";
const DOMAIN_SEPARATOR: &[u8] = b"symthaea:safety-configuration-observation:v1\0";

/// Fingerprint of the separately provisioned configuration-observer verifier key.
///
/// V1 uses BLAKE3-256 over exactly the raw verifier public-key bytes, matching the
/// root-fingerprint convention used by the profile and commissioning authority
/// contracts while retaining a distinct Rust type so these powers cannot be mixed
/// accidentally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConfigurationObserverRootDigest {
    Blake3_256([u8; 32]),
}

impl ConfigurationObserverRootDigest {
    pub fn from_raw_verifier_key(verifier_key_bytes: &[u8]) -> Self {
        Self::Blake3_256(ConfigurationDigest::blake3_256(verifier_key_bytes).into_blake3_256())
    }

    pub fn into_blake3_256(self) -> [u8; 32] {
        match self {
            Self::Blake3_256(bytes) => bytes,
        }
    }
}

/// Unpredictable transaction challenge that binds a configuration observation to
/// one commissioning/qualification attempt.
///
/// This type does not generate randomness. The caller must obtain the bytes from an
/// appropriate CSPRNG or hardware challenge source. V1 rejects the all-zero value so
/// accidentally omitting challenge generation fails closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConfigurationObservationChallenge([u8; 32]);

impl ConfigurationObservationChallenge {
    pub fn new(bytes: [u8; 32]) -> Result<Self, SafetyConfigurationObservationError> {
        if bytes == [0; 32] {
            return Err(SafetyConfigurationObservationError::ZeroChallenge);
        }
        Ok(Self(bytes))
    }

    pub fn into_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Exact BLAKE3-256 identity of one canonical configuration observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyConfigurationObservationDigest {
    Blake3_256([u8; 32]),
}

impl SafetyConfigurationObservationDigest {
    pub fn blake3_256(bytes: &[u8]) -> Self {
        Self::Blake3_256(ConfigurationDigest::blake3_256(bytes).into_blake3_256())
    }

    pub fn into_blake3_256(self) -> [u8; 32] {
        match self {
            Self::Blake3_256(bytes) => bytes,
        }
    }
}

/// Raw canonical claim that one observer saw one exact safety configuration on one
/// node for one challenge at one observer-local generation/time.
///
/// `observation_generation` is monotone within the observer/node lineage. It is not
/// a substitute for the challenge or trusted time: all three serve different replay
/// and freshness purposes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyConfigurationObservation {
    schema_version: String,
    observation_id: String,
    observer_id: String,
    subject_node_id: String,
    observation_generation: u64,
    observed_at_unix_ms: i64,
    observer_root_digest: ConfigurationObserverRootDigest,
    challenge: ConfigurationObservationChallenge,
    configuration_digest: ConfigurationDigest,
}

impl SafetyConfigurationObservation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        observation_id: impl Into<String>,
        observer_id: impl Into<String>,
        subject_node_id: impl Into<String>,
        observation_generation: u64,
        observed_at_unix_ms: i64,
        observer_root_digest: ConfigurationObserverRootDigest,
        challenge: ConfigurationObservationChallenge,
        configuration_digest: ConfigurationDigest,
    ) -> Result<Self, SafetyConfigurationObservationError> {
        let observation = Self {
            schema_version: SAFETY_CONFIGURATION_OBSERVATION_SCHEMA_V1.to_owned(),
            observation_id: observation_id.into(),
            observer_id: observer_id.into(),
            subject_node_id: subject_node_id.into(),
            observation_generation,
            observed_at_unix_ms,
            observer_root_digest,
            challenge,
            configuration_digest,
        };
        observation.validate()?;
        Ok(observation)
    }

    pub fn validate(&self) -> Result<(), SafetyConfigurationObservationError> {
        if self.schema_version != SAFETY_CONFIGURATION_OBSERVATION_SCHEMA_V1 {
            return Err(SafetyConfigurationObservationError::UnsupportedSchemaVersion(
                self.schema_version.clone(),
            ));
        }
        if self.observation_id.trim().is_empty() {
            return Err(SafetyConfigurationObservationError::EmptyObservationId);
        }
        if self.observer_id.trim().is_empty() {
            return Err(SafetyConfigurationObservationError::EmptyObserverId);
        }
        if self.subject_node_id.trim().is_empty() {
            return Err(SafetyConfigurationObservationError::EmptySubjectNodeId);
        }
        if self.observation_generation == 0 {
            return Err(SafetyConfigurationObservationError::ZeroObservationGeneration);
        }
        if self.challenge.0 == [0; 32] {
            return Err(SafetyConfigurationObservationError::ZeroChallenge);
        }
        Ok(())
    }

    /// Fixed-order, domain-separated bytes intended for external authentication.
    ///
    /// Encoding v1:
    /// - fixed ASCII domain separator
    /// - schema, observation ID, observer ID, and node ID as u32-length-prefixed UTF-8
    /// - observation generation as big-endian u64
    /// - signed observation time as big-endian i64 Unix milliseconds
    /// - observer root digest with explicit digest-algorithm tag
    /// - exact 32-byte challenge
    /// - configuration digest with explicit digest-algorithm tag
    pub fn canonical_signing_bytes(&self) -> Result<Vec<u8>, SafetyConfigurationObservationError> {
        self.validate()?;
        let mut out = Vec::with_capacity(256);
        out.extend_from_slice(DOMAIN_SEPARATOR);
        push_string(&mut out, "schema_version", &self.schema_version)?;
        push_string(&mut out, "observation_id", &self.observation_id)?;
        push_string(&mut out, "observer_id", &self.observer_id)?;
        push_string(&mut out, "subject_node_id", &self.subject_node_id)?;
        out.extend_from_slice(&self.observation_generation.to_be_bytes());
        out.extend_from_slice(&self.observed_at_unix_ms.to_be_bytes());
        push_observer_root_digest(&mut out, self.observer_root_digest);
        out.extend_from_slice(self.challenge.as_bytes());
        push_configuration_digest(&mut out, self.configuration_digest);
        Ok(out)
    }

    pub fn observation_digest(
        &self,
    ) -> Result<SafetyConfigurationObservationDigest, SafetyConfigurationObservationError> {
        Ok(SafetyConfigurationObservationDigest::blake3_256(
            &self.canonical_signing_bytes()?,
        ))
    }

    pub fn observation_id(&self) -> &str {
        &self.observation_id
    }

    pub fn observer_id(&self) -> &str {
        &self.observer_id
    }

    pub fn subject_node_id(&self) -> &str {
        &self.subject_node_id
    }

    pub fn observation_generation(&self) -> u64 {
        self.observation_generation
    }

    pub fn observed_at_unix_ms(&self) -> i64 {
        self.observed_at_unix_ms
    }

    pub fn observer_root_digest(&self) -> ConfigurationObserverRootDigest {
        self.observer_root_digest
    }

    pub fn challenge(&self) -> ConfigurationObservationChallenge {
        self.challenge
    }

    pub fn configuration_digest(&self) -> ConfigurationDigest {
        self.configuration_digest
    }
}

fn push_string(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &str,
) -> Result<(), SafetyConfigurationObservationError> {
    let len = u32::try_from(value.len())
        .map_err(|_| SafetyConfigurationObservationError::StringTooLong(field))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_observer_root_digest(out: &mut Vec<u8>, digest: ConfigurationObserverRootDigest) {
    match digest {
        ConfigurationObserverRootDigest::Blake3_256(bytes) => {
            out.push(1);
            out.extend_from_slice(&bytes);
        }
    }
}

fn push_configuration_digest(out: &mut Vec<u8>, digest: ConfigurationDigest) {
    match digest {
        ConfigurationDigest::Blake3_256(bytes) => {
            out.push(1);
            out.extend_from_slice(&bytes);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyConfigurationObservationError {
    #[error("unsupported safety configuration observation schema version {0}")]
    UnsupportedSchemaVersion(String),
    #[error("safety configuration observation id must not be empty")]
    EmptyObservationId,
    #[error("safety configuration observer id must not be empty")]
    EmptyObserverId,
    #[error("safety configuration observation subject node id must not be empty")]
    EmptySubjectNodeId,
    #[error("safety configuration observation generation must be greater than zero")]
    ZeroObservationGeneration,
    #[error("safety configuration observation challenge must not be all zero")]
    ZeroChallenge,
    #[error("safety configuration observation string field {0} exceeds canonical u32 length")]
    StringTooLong(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    const PINNED_V1_HEX: &str = "73796d74686165613a7361666574792d636f6e66696775726174696f6e2d6f62736572766174696f6e3a7631000000002c73796d74686165612d7361666574792d636f6e66696775726174696f6e2d6f62736572766174696f6e2d7631000000056f62732d310000000a6f627365727665722d31000000047261636b000000000000000700000000000005dc0144444444444444444444444444444444444444444444444444444444444444445555555555555555555555555555555555555555555555555555555555555555016666666666666666666666666666666666666666666666666666666666666666";

    fn config(byte: u8) -> ConfigurationDigest {
        ConfigurationDigest::Blake3_256([byte; 32])
    }

    fn root(byte: u8) -> ConfigurationObserverRootDigest {
        ConfigurationObserverRootDigest::Blake3_256([byte; 32])
    }

    fn challenge(byte: u8) -> ConfigurationObservationChallenge {
        ConfigurationObservationChallenge::new([byte; 32]).unwrap()
    }

    fn observation() -> SafetyConfigurationObservation {
        SafetyConfigurationObservation::new(
            "obs-1",
            "observer-1",
            "rack",
            7,
            1_500,
            root(0x44),
            challenge(0x55),
            config(0x66),
        )
        .unwrap()
    }

    #[test]
    fn pinned_v1_vector_is_byte_exact() {
        assert_eq!(hex(&observation().canonical_signing_bytes().unwrap()), PINNED_V1_HEX);
    }

    #[test]
    fn identical_observations_have_identical_identity() {
        let first = observation();
        let second = first.clone();
        assert_eq!(
            first.canonical_signing_bytes().unwrap(),
            second.canonical_signing_bytes().unwrap()
        );
        assert_eq!(
            first.observation_digest().unwrap(),
            second.observation_digest().unwrap()
        );
    }

    #[test]
    fn challenge_is_part_of_observation_identity() {
        let first = observation();
        let second = SafetyConfigurationObservation::new(
            "obs-1",
            "observer-1",
            "rack",
            7,
            1_500,
            root(0x44),
            challenge(0x56),
            config(0x66),
        )
        .unwrap();
        assert_ne!(first.observation_digest().unwrap(), second.observation_digest().unwrap());
    }

    #[test]
    fn configuration_is_part_of_observation_identity() {
        let first = observation();
        let second = SafetyConfigurationObservation::new(
            "obs-1",
            "observer-1",
            "rack",
            7,
            1_500,
            root(0x44),
            challenge(0x55),
            config(0x67),
        )
        .unwrap();
        assert_ne!(first.observation_digest().unwrap(), second.observation_digest().unwrap());
    }

    #[test]
    fn observer_root_and_generation_are_bound() {
        let first = observation();
        let different_root = SafetyConfigurationObservation::new(
            "obs-1",
            "observer-1",
            "rack",
            7,
            1_500,
            root(0x45),
            challenge(0x55),
            config(0x66),
        )
        .unwrap();
        let different_generation = SafetyConfigurationObservation::new(
            "obs-1",
            "observer-1",
            "rack",
            8,
            1_500,
            root(0x44),
            challenge(0x55),
            config(0x66),
        )
        .unwrap();
        assert_ne!(first.observation_digest().unwrap(), different_root.observation_digest().unwrap());
        assert_ne!(
            first.observation_digest().unwrap(),
            different_generation.observation_digest().unwrap()
        );
    }

    #[test]
    fn zero_generation_and_zero_challenge_fail_closed() {
        assert_eq!(
            ConfigurationObservationChallenge::new([0; 32]),
            Err(SafetyConfigurationObservationError::ZeroChallenge)
        );
        assert_eq!(
            SafetyConfigurationObservation::new(
                "obs-1",
                "observer-1",
                "rack",
                0,
                1_500,
                root(0x44),
                challenge(0x55),
                config(0x66),
            ),
            Err(SafetyConfigurationObservationError::ZeroObservationGeneration)
        );
    }

    #[test]
    fn observer_key_fingerprint_is_raw_key_hash() {
        let raw_key = [0x42; 1952];
        assert_eq!(
            ConfigurationObserverRootDigest::from_raw_verifier_key(&raw_key).into_blake3_256(),
            ConfigurationDigest::blake3_256(&raw_key).into_blake3_256()
        );
    }

    fn hex(bytes: &[u8]) -> String {
        const TABLE: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            out.push(TABLE[(byte >> 4) as usize] as char);
            out.push(TABLE[(byte & 0x0f) as usize] as char);
        }
        out
    }
}
