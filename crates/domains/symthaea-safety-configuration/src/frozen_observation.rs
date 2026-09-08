// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical observation evidence bound to one exact configuration freeze lineage.
//!
//! [`crate::observation::SafetyConfigurationObservation`] proves only what one
//! observer claims to have seen. This wrapper additionally binds that claim to the
//! exact local configuration epoch and freeze generation under which the observation
//! was produced. It does not replace the original observation format.
//!
//! The constructor requires the supplied freeze token to still be the exact active
//! freeze in [`crate::state::SafetyConfigurationState`] and requires node, challenge,
//! and configuration digest equality between the observation and freeze token.
//!
//! This remains raw serializable evidence. Deserialization does not recreate an
//! active local freeze, and callers must still verify the canonical signing bytes,
//! apply freshness/root policy, and recheck the local state before commissioning.

use crate::observation::{
    SafetyConfigurationObservation, SafetyConfigurationObservationDigest,
    SafetyConfigurationObservationError,
};
use crate::state::{
    SafetyConfigurationFreezeToken, SafetyConfigurationState, SafetyConfigurationStateError,
};
use crate::ConfigurationDigest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SAFETY_CONFIGURATION_FROZEN_OBSERVATION_SCHEMA_V1: &str =
    "symthaea-safety-configuration-frozen-observation-v1";
const DOMAIN_SEPARATOR: &[u8] = b"symthaea:safety-configuration-frozen-observation:v1\0";

/// BLAKE3-256 identity of one canonical frozen-configuration observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyConfigurationFrozenObservationDigest {
    Blake3_256([u8; 32]),
}

impl SafetyConfigurationFrozenObservationDigest {
    pub fn blake3_256(bytes: &[u8]) -> Self {
        Self::Blake3_256(ConfigurationDigest::blake3_256(bytes).into_blake3_256())
    }

    pub fn into_blake3_256(self) -> [u8; 32] {
        match self {
            Self::Blake3_256(bytes) => bytes,
        }
    }
}

/// Raw observer claim tied to one exact configuration freeze lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyConfigurationFrozenObservation {
    schema_version: String,
    configuration_epoch: u64,
    freeze_generation: u64,
    observation: SafetyConfigurationObservation,
}

impl SafetyConfigurationFrozenObservation {
    pub fn new(
        observation: SafetyConfigurationObservation,
        state: &SafetyConfigurationState,
        freeze: &SafetyConfigurationFreezeToken,
    ) -> Result<Self, SafetyConfigurationFrozenObservationError> {
        observation.validate()?;
        state.require_active_freeze(freeze)?;

        if observation.subject_node_id() != state.node_id()
            || freeze.node_id() != state.node_id()
        {
            return Err(SafetyConfigurationFrozenObservationError::NodeMismatch);
        }
        if observation.configuration_digest() != state.configuration_digest()
            || freeze.configuration_digest() != state.configuration_digest()
        {
            return Err(SafetyConfigurationFrozenObservationError::ConfigurationDigestMismatch);
        }
        if observation.challenge() != freeze.challenge() {
            return Err(SafetyConfigurationFrozenObservationError::ChallengeMismatch);
        }
        if freeze.configuration_epoch() != state.configuration_epoch() {
            return Err(SafetyConfigurationFrozenObservationError::ConfigurationEpochMismatch {
                expected: state.configuration_epoch(),
                observed: freeze.configuration_epoch(),
            });
        }
        if freeze.freeze_generation() != state.freeze_generation() {
            return Err(SafetyConfigurationFrozenObservationError::FreezeGenerationMismatch {
                expected: state.freeze_generation(),
                observed: freeze.freeze_generation(),
            });
        }

        let frozen = Self {
            schema_version: SAFETY_CONFIGURATION_FROZEN_OBSERVATION_SCHEMA_V1.to_owned(),
            configuration_epoch: freeze.configuration_epoch(),
            freeze_generation: freeze.freeze_generation(),
            observation,
        };
        frozen.validate()?;
        Ok(frozen)
    }

    /// Structural validation for raw wire evidence.
    ///
    /// This intentionally cannot prove that the freeze is still active now. That
    /// requires comparison against a live [`SafetyConfigurationState`].
    pub fn validate(&self) -> Result<(), SafetyConfigurationFrozenObservationError> {
        if self.schema_version != SAFETY_CONFIGURATION_FROZEN_OBSERVATION_SCHEMA_V1 {
            return Err(
                SafetyConfigurationFrozenObservationError::UnsupportedSchemaVersion(
                    self.schema_version.clone(),
                ),
            );
        }
        if self.configuration_epoch == 0 {
            return Err(SafetyConfigurationFrozenObservationError::ZeroConfigurationEpoch);
        }
        if self.freeze_generation == 0 {
            return Err(SafetyConfigurationFrozenObservationError::ZeroFreezeGeneration);
        }
        self.observation.validate()?;
        Ok(())
    }

    /// Fixed-order, domain-separated bytes intended for verifier authentication.
    ///
    /// Encoding v1:
    /// - fixed ASCII domain separator
    /// - schema as u32-length-prefixed UTF-8
    /// - configuration epoch as big-endian u64
    /// - freeze generation as big-endian u64
    /// - nested canonical observation length as big-endian u32
    /// - complete canonical observation bytes
    pub fn canonical_signing_bytes(
        &self,
    ) -> Result<Vec<u8>, SafetyConfigurationFrozenObservationError> {
        self.validate()?;
        let observation_bytes = self.observation.canonical_signing_bytes()?;
        let observation_len = u32::try_from(observation_bytes.len())
            .map_err(|_| SafetyConfigurationFrozenObservationError::ObservationTooLarge)?;
        let schema_len = u32::try_from(self.schema_version.len())
            .map_err(|_| SafetyConfigurationFrozenObservationError::SchemaTooLong)?;

        let mut out = Vec::with_capacity(DOMAIN_SEPARATOR.len() + 4 + self.schema_version.len() + 20 + observation_bytes.len());
        out.extend_from_slice(DOMAIN_SEPARATOR);
        out.extend_from_slice(&schema_len.to_be_bytes());
        out.extend_from_slice(self.schema_version.as_bytes());
        out.extend_from_slice(&self.configuration_epoch.to_be_bytes());
        out.extend_from_slice(&self.freeze_generation.to_be_bytes());
        out.extend_from_slice(&observation_len.to_be_bytes());
        out.extend_from_slice(&observation_bytes);
        Ok(out)
    }

    pub fn frozen_observation_digest(
        &self,
    ) -> Result<SafetyConfigurationFrozenObservationDigest, SafetyConfigurationFrozenObservationError>
    {
        Ok(SafetyConfigurationFrozenObservationDigest::blake3_256(
            &self.canonical_signing_bytes()?,
        ))
    }

    pub fn configuration_epoch(&self) -> u64 {
        self.configuration_epoch
    }

    pub fn freeze_generation(&self) -> u64 {
        self.freeze_generation
    }

    pub fn observation(&self) -> &SafetyConfigurationObservation {
        &self.observation
    }

    pub fn observation_digest(
        &self,
    ) -> Result<SafetyConfigurationObservationDigest, SafetyConfigurationFrozenObservationError> {
        Ok(self.observation.observation_digest()?)
    }

    /// Recheck the raw evidence against the exact currently active local freeze.
    pub fn require_live_freeze(
        &self,
        state: &SafetyConfigurationState,
        freeze: &SafetyConfigurationFreezeToken,
    ) -> Result<(), SafetyConfigurationFrozenObservationError> {
        state.require_active_freeze(freeze)?;
        if self.observation.subject_node_id() != state.node_id()
            || freeze.node_id() != state.node_id()
        {
            return Err(SafetyConfigurationFrozenObservationError::NodeMismatch);
        }
        if self.observation.configuration_digest() != state.configuration_digest()
            || freeze.configuration_digest() != state.configuration_digest()
        {
            return Err(SafetyConfigurationFrozenObservationError::ConfigurationDigestMismatch);
        }
        if self.observation.challenge() != freeze.challenge() {
            return Err(SafetyConfigurationFrozenObservationError::ChallengeMismatch);
        }
        if self.configuration_epoch != state.configuration_epoch()
            || freeze.configuration_epoch() != state.configuration_epoch()
        {
            return Err(SafetyConfigurationFrozenObservationError::ConfigurationEpochMismatch {
                expected: state.configuration_epoch(),
                observed: self.configuration_epoch,
            });
        }
        if self.freeze_generation != state.freeze_generation()
            || freeze.freeze_generation() != state.freeze_generation()
        {
            return Err(SafetyConfigurationFrozenObservationError::FreezeGenerationMismatch {
                expected: state.freeze_generation(),
                observed: self.freeze_generation,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyConfigurationFrozenObservationError {
    #[error(transparent)]
    Observation(#[from] SafetyConfigurationObservationError),
    #[error(transparent)]
    State(#[from] SafetyConfigurationStateError),
    #[error("unsupported frozen configuration observation schema version {0}")]
    UnsupportedSchemaVersion(String),
    #[error("frozen configuration observation configuration epoch must be greater than zero")]
    ZeroConfigurationEpoch,
    #[error("frozen configuration observation freeze generation must be greater than zero")]
    ZeroFreezeGeneration,
    #[error("frozen configuration observation node does not match the active freeze state")]
    NodeMismatch,
    #[error("frozen configuration observation digest does not match the active freeze state")]
    ConfigurationDigestMismatch,
    #[error("frozen configuration observation challenge does not match the active freeze")]
    ChallengeMismatch,
    #[error("frozen configuration observation epoch mismatch: expected {expected}, observed {observed}")]
    ConfigurationEpochMismatch { expected: u64, observed: u64 },
    #[error("frozen configuration observation freeze generation mismatch: expected {expected}, observed {observed}")]
    FreezeGenerationMismatch { expected: u64, observed: u64 },
    #[error("frozen configuration observation schema string exceeds canonical u32 length")]
    SchemaTooLong,
    #[error("nested configuration observation exceeds canonical u32 length")]
    ObservationTooLarge,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::{
        ConfigurationObservationChallenge, ConfigurationObserverRootDigest,
    };

    const PINNED_V1_HEX: &str = "73796d74686165613a7361666574792d636f6e66696775726174696f6e2d66726f7a656e2d6f62736572766174696f6e3a7631000000003373796d74686165612d7361666574792d636f6e66696775726174696f6e2d66726f7a656e2d6f62736572766174696f6e2d763100000000000000010000000000000001000000ee73796d74686165613a7361666574792d636f6e66696775726174696f6e2d6f62736572766174696f6e3a7631000000002c73796d74686165612d7361666574792d636f6e66696775726174696f6e2d6f62736572766174696f6e2d7631000000056f62732d310000000a6f627365727665722d31000000047261636b000000000000000700000000000005dc0144444444444444444444444444444444444444444444444444444444444444445555555555555555555555555555555555555555555555555555555555555555016666666666666666666666666666666666666666666666666666666666666666";

    fn digest(byte: u8) -> ConfigurationDigest {
        ConfigurationDigest::Blake3_256([byte; 32])
    }

    fn root(byte: u8) -> ConfigurationObserverRootDigest {
        ConfigurationObserverRootDigest::Blake3_256([byte; 32])
    }

    fn challenge(byte: u8) -> ConfigurationObservationChallenge {
        ConfigurationObservationChallenge::new([byte; 32]).unwrap()
    }

    fn observation(challenge_value: u8, config: ConfigurationDigest) -> SafetyConfigurationObservation {
        SafetyConfigurationObservation::new(
            "obs-1",
            "observer-1",
            "rack",
            7,
            1_500,
            root(0x44),
            challenge(challenge_value),
            config,
        )
        .unwrap()
    }

    fn frozen() -> (
        SafetyConfigurationState,
        SafetyConfigurationFreezeToken,
        SafetyConfigurationFrozenObservation,
    ) {
        let mut state = SafetyConfigurationState::initialize("rack", digest(0x66)).unwrap();
        let freeze = state.begin_freeze(challenge(0x55)).unwrap();
        let frozen = SafetyConfigurationFrozenObservation::new(
            observation(0x55, digest(0x66)),
            &state,
            &freeze,
        )
        .unwrap();
        (state, freeze, frozen)
    }

    #[test]
    fn pinned_v1_vector_is_byte_exact() {
        let (_, _, frozen) = frozen();
        assert_eq!(hex(&frozen.canonical_signing_bytes().unwrap()), PINNED_V1_HEX);
    }

    #[test]
    fn exact_active_freeze_produces_frozen_observation() {
        let (state, freeze, frozen) = frozen();
        assert_eq!(frozen.configuration_epoch(), 1);
        assert_eq!(frozen.freeze_generation(), 1);
        frozen.require_live_freeze(&state, &freeze).unwrap();
    }

    #[test]
    fn wrong_challenge_is_rejected() {
        let mut state = SafetyConfigurationState::initialize("rack", digest(0x66)).unwrap();
        let freeze = state.begin_freeze(challenge(0x55)).unwrap();
        assert_eq!(
            SafetyConfigurationFrozenObservation::new(
                observation(0x56, digest(0x66)),
                &state,
                &freeze,
            ),
            Err(SafetyConfigurationFrozenObservationError::ChallengeMismatch)
        );
    }

    #[test]
    fn wrong_configuration_is_rejected() {
        let mut state = SafetyConfigurationState::initialize("rack", digest(0x66)).unwrap();
        let freeze = state.begin_freeze(challenge(0x55)).unwrap();
        assert_eq!(
            SafetyConfigurationFrozenObservation::new(
                observation(0x55, digest(0x77)),
                &state,
                &freeze,
            ),
            Err(SafetyConfigurationFrozenObservationError::ConfigurationDigestMismatch)
        );
    }

    #[test]
    fn release_reacquire_changes_signed_freeze_lineage_even_with_same_digest_and_challenge() {
        let mut state = SafetyConfigurationState::initialize("rack", digest(0x66)).unwrap();
        let first_token = state.begin_freeze(challenge(0x55)).unwrap();
        let first = SafetyConfigurationFrozenObservation::new(
            observation(0x55, digest(0x66)),
            &state,
            &first_token,
        )
        .unwrap();
        state.end_freeze(&first_token).unwrap();
        let second_token = state.begin_freeze(challenge(0x55)).unwrap();
        let second = SafetyConfigurationFrozenObservation::new(
            observation(0x55, digest(0x66)),
            &state,
            &second_token,
        )
        .unwrap();
        assert_ne!(first.freeze_generation(), second.freeze_generation());
        assert_ne!(
            first.canonical_signing_bytes().unwrap(),
            second.canonical_signing_bytes().unwrap()
        );
    }

    #[test]
    fn external_mutation_invalidates_live_freeze_proof() {
        let (mut state, freeze, frozen) = frozen();
        state.record_external_mutation(1, digest(0x77)).unwrap();
        assert_eq!(
            frozen.require_live_freeze(&state, &freeze),
            Err(SafetyConfigurationFrozenObservationError::State(
                SafetyConfigurationStateError::NotFrozen
            ))
        );
    }

    fn hex(bytes: &[u8]) -> String {
        const LUT: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            out.push(LUT[(byte >> 4) as usize] as char);
            out.push(LUT[(byte & 0x0f) as usize] as char);
        }
        out
    }
}