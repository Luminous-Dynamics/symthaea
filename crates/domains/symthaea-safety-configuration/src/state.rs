// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Monotone local state for the currently installed safety configuration.
//!
//! A fresh signed observation proves what one observer saw at one time. It does not
//! by itself prevent the configuration from changing immediately afterward. This
//! module adds a cooperating local mutation boundary with two independent epochs:
//!
//! - `configuration_epoch` advances on every declared safety-relevant mutation,
//!   even when the resulting canonical configuration digest is unchanged.
//! - `freeze_generation` advances on every commissioning freeze acquisition, so a
//!   release/reacquire ABA cycle cannot recreate an old freeze token.
//!
//! Managed mutations are rejected while a freeze is active. Out-of-band mutations
//! cannot be prevented by this software abstraction; when detected they must be
//! recorded through [`SafetyConfigurationState::record_external_mutation`], which
//! advances the configuration epoch and invalidates any active freeze. Hardware,
//! BMC, firmware, provisioning, and physical-observation tooling therefore remains
//! responsible for surfacing external changes into this state machine.
//!
//! This module is local coordination state, not cryptographic attestation. A future
//! configuration-state observation should bind the exact configuration epoch and
//! freeze generation into verifier-authenticated evidence before commissioning.

use crate::ConfigurationDigest;
use crate::observation::ConfigurationObservationChallenge;
use thiserror::Error;

/// How one safety-relevant configuration transition entered the local state model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigurationMutationOrigin {
    Managed,
    ExternalObservation,
}

/// Exact token for one active commissioning freeze.
///
/// Fields are private and the type is intentionally non-Serde. The token can only
/// be created by successfully freezing one [`SafetyConfigurationState`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyConfigurationFreezeToken {
    node_id: String,
    configuration_epoch: u64,
    configuration_digest: ConfigurationDigest,
    freeze_generation: u64,
    challenge: ConfigurationObservationChallenge,
}

impl SafetyConfigurationFreezeToken {
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub fn configuration_epoch(&self) -> u64 {
        self.configuration_epoch
    }

    pub fn configuration_digest(&self) -> ConfigurationDigest {
        self.configuration_digest
    }

    pub fn freeze_generation(&self) -> u64 {
        self.freeze_generation
    }

    pub fn challenge(&self) -> ConfigurationObservationChallenge {
        self.challenge
    }
}

/// Result of recording one safety-relevant configuration mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyConfigurationMutation {
    origin: ConfigurationMutationOrigin,
    previous_epoch: u64,
    new_epoch: u64,
    previous_digest: ConfigurationDigest,
    new_digest: ConfigurationDigest,
    invalidated_freeze: Option<SafetyConfigurationFreezeToken>,
}

impl SafetyConfigurationMutation {
    pub fn origin(&self) -> ConfigurationMutationOrigin {
        self.origin
    }

    pub fn previous_epoch(&self) -> u64 {
        self.previous_epoch
    }

    pub fn new_epoch(&self) -> u64 {
        self.new_epoch
    }

    pub fn previous_digest(&self) -> ConfigurationDigest {
        self.previous_digest
    }

    pub fn new_digest(&self) -> ConfigurationDigest {
        self.new_digest
    }

    pub fn invalidated_freeze(&self) -> Option<&SafetyConfigurationFreezeToken> {
        self.invalidated_freeze.as_ref()
    }
}

/// Current local view of one node's safety-relevant configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyConfigurationState {
    node_id: String,
    configuration_epoch: u64,
    configuration_digest: ConfigurationDigest,
    freeze_generation: u64,
    active_freeze: Option<SafetyConfigurationFreezeToken>,
}

impl SafetyConfigurationState {
    pub fn initialize(
        node_id: impl Into<String>,
        configuration_digest: ConfigurationDigest,
    ) -> Result<Self, SafetyConfigurationStateError> {
        let node_id = node_id.into();
        if node_id.trim().is_empty() {
            return Err(SafetyConfigurationStateError::EmptyNodeId);
        }
        Ok(Self {
            node_id,
            configuration_epoch: 1,
            configuration_digest,
            freeze_generation: 0,
            active_freeze: None,
        })
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub fn configuration_epoch(&self) -> u64 {
        self.configuration_epoch
    }

    pub fn configuration_digest(&self) -> ConfigurationDigest {
        self.configuration_digest
    }

    pub fn freeze_generation(&self) -> u64 {
        self.freeze_generation
    }

    pub fn active_freeze(&self) -> Option<&SafetyConfigurationFreezeToken> {
        self.active_freeze.as_ref()
    }

    pub fn begin_freeze(
        &mut self,
        challenge: ConfigurationObservationChallenge,
    ) -> Result<SafetyConfigurationFreezeToken, SafetyConfigurationStateError> {
        if self.active_freeze.is_some() {
            return Err(SafetyConfigurationStateError::AlreadyFrozen);
        }
        let freeze_generation = self
            .freeze_generation
            .checked_add(1)
            .ok_or(SafetyConfigurationStateError::FreezeGenerationExhausted)?;
        let token = SafetyConfigurationFreezeToken {
            node_id: self.node_id.clone(),
            configuration_epoch: self.configuration_epoch,
            configuration_digest: self.configuration_digest,
            freeze_generation,
            challenge,
        };
        self.freeze_generation = freeze_generation;
        self.active_freeze = Some(token.clone());
        Ok(token)
    }

    pub fn end_freeze(
        &mut self,
        token: &SafetyConfigurationFreezeToken,
    ) -> Result<(), SafetyConfigurationStateError> {
        match self.active_freeze.as_ref() {
            Some(active) if active == token => {
                self.active_freeze = None;
                Ok(())
            }
            Some(_) => Err(SafetyConfigurationStateError::FreezeTokenMismatch),
            None => Err(SafetyConfigurationStateError::NotFrozen),
        }
    }

    pub fn require_active_freeze(
        &self,
        token: &SafetyConfigurationFreezeToken,
    ) -> Result<(), SafetyConfigurationStateError> {
        match self.active_freeze.as_ref() {
            Some(active) if active == token => Ok(()),
            Some(_) => Err(SafetyConfigurationStateError::FreezeTokenMismatch),
            None => Err(SafetyConfigurationStateError::NotFrozen),
        }
    }

    pub fn apply_managed_mutation(
        &mut self,
        expected_epoch: u64,
        new_digest: ConfigurationDigest,
    ) -> Result<SafetyConfigurationMutation, SafetyConfigurationStateError> {
        if self.active_freeze.is_some() {
            return Err(SafetyConfigurationStateError::MutationWhileFrozen);
        }
        self.require_epoch(expected_epoch)?;
        let new_epoch = self.next_configuration_epoch()?;
        Ok(self.apply_mutation_unchecked(
            new_epoch,
            new_digest,
            ConfigurationMutationOrigin::Managed,
            None,
        ))
    }

    /// Record a safety-relevant change discovered outside the cooperating writer.
    ///
    /// Reality wins over the software freeze. Every fallible precondition is checked
    /// before the active freeze is taken, so a failed recording leaves state intact.
    pub fn record_external_mutation(
        &mut self,
        expected_epoch: u64,
        new_digest: ConfigurationDigest,
    ) -> Result<SafetyConfigurationMutation, SafetyConfigurationStateError> {
        self.require_epoch(expected_epoch)?;
        let new_epoch = self.next_configuration_epoch()?;
        let invalidated_freeze = self.active_freeze.take();
        Ok(self.apply_mutation_unchecked(
            new_epoch,
            new_digest,
            ConfigurationMutationOrigin::ExternalObservation,
            invalidated_freeze,
        ))
    }

    fn require_epoch(&self, observed: u64) -> Result<(), SafetyConfigurationStateError> {
        if observed != self.configuration_epoch {
            return Err(SafetyConfigurationStateError::ConfigurationEpochMismatch {
                expected: self.configuration_epoch,
                observed,
            });
        }
        Ok(())
    }

    fn next_configuration_epoch(&self) -> Result<u64, SafetyConfigurationStateError> {
        self.configuration_epoch
            .checked_add(1)
            .ok_or(SafetyConfigurationStateError::ConfigurationEpochExhausted)
    }

    fn apply_mutation_unchecked(
        &mut self,
        new_epoch: u64,
        new_digest: ConfigurationDigest,
        origin: ConfigurationMutationOrigin,
        invalidated_freeze: Option<SafetyConfigurationFreezeToken>,
    ) -> SafetyConfigurationMutation {
        let previous_epoch = self.configuration_epoch;
        let previous_digest = self.configuration_digest;
        self.configuration_epoch = new_epoch;
        self.configuration_digest = new_digest;
        SafetyConfigurationMutation {
            origin,
            previous_epoch,
            new_epoch,
            previous_digest,
            new_digest,
            invalidated_freeze,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyConfigurationStateError {
    #[error("safety configuration state node id must not be empty")]
    EmptyNodeId,
    #[error("safety configuration is already frozen for a commissioning transaction")]
    AlreadyFrozen,
    #[error("safety configuration is not currently frozen")]
    NotFrozen,
    #[error("freeze token does not match the exact currently active freeze")]
    FreezeTokenMismatch,
    #[error("managed safety-relevant configuration mutation is forbidden while frozen")]
    MutationWhileFrozen,
    #[error("safety configuration epoch mismatch: current {expected}, caller supplied {observed}")]
    ConfigurationEpochMismatch { expected: u64, observed: u64 },
    #[error("safety configuration epoch space exhausted")]
    ConfigurationEpochExhausted,
    #[error("safety configuration freeze-generation space exhausted")]
    FreezeGenerationExhausted,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> ConfigurationDigest {
        ConfigurationDigest::Blake3_256([byte; 32])
    }

    fn challenge(byte: u8) -> ConfigurationObservationChallenge {
        ConfigurationObservationChallenge::new([byte; 32]).unwrap()
    }

    #[test]
    fn initialize_starts_at_epoch_one_and_unfrozen() {
        let state = SafetyConfigurationState::initialize("rack", digest(0x11)).unwrap();
        assert_eq!(state.node_id(), "rack");
        assert_eq!(state.configuration_epoch(), 1);
        assert_eq!(state.configuration_digest(), digest(0x11));
        assert_eq!(state.freeze_generation(), 0);
        assert!(state.active_freeze().is_none());
    }

    #[test]
    fn managed_mutation_advances_epoch_even_when_digest_is_unchanged() {
        let mut state = SafetyConfigurationState::initialize("rack", digest(0x11)).unwrap();
        let mutation = state.apply_managed_mutation(1, digest(0x11)).unwrap();
        assert_eq!(mutation.origin(), ConfigurationMutationOrigin::Managed);
        assert_eq!(mutation.previous_epoch(), 1);
        assert_eq!(mutation.new_epoch(), 2);
        assert_eq!(state.configuration_epoch(), 2);
        assert_eq!(state.configuration_digest(), digest(0x11));
    }

    #[test]
    fn freeze_binds_epoch_digest_generation_and_challenge() {
        let mut state = SafetyConfigurationState::initialize("rack", digest(0x11)).unwrap();
        let token = state.begin_freeze(challenge(0x44)).unwrap();
        assert_eq!(token.node_id(), "rack");
        assert_eq!(token.configuration_epoch(), 1);
        assert_eq!(token.configuration_digest(), digest(0x11));
        assert_eq!(token.freeze_generation(), 1);
        assert_eq!(token.challenge(), challenge(0x44));
        state.require_active_freeze(&token).unwrap();
    }

    #[test]
    fn managed_mutation_fails_closed_while_frozen() {
        let mut state = SafetyConfigurationState::initialize("rack", digest(0x11)).unwrap();
        state.begin_freeze(challenge(0x44)).unwrap();
        assert_eq!(
            state.apply_managed_mutation(1, digest(0x22)),
            Err(SafetyConfigurationStateError::MutationWhileFrozen)
        );
        assert_eq!(state.configuration_epoch(), 1);
        assert_eq!(state.configuration_digest(), digest(0x11));
    }

    #[test]
    fn external_mutation_invalidates_freeze_and_advances_epoch() {
        let mut state = SafetyConfigurationState::initialize("rack", digest(0x11)).unwrap();
        let token = state.begin_freeze(challenge(0x44)).unwrap();
        let mutation = state.record_external_mutation(1, digest(0x22)).unwrap();
        assert_eq!(mutation.origin(), ConfigurationMutationOrigin::ExternalObservation);
        assert_eq!(mutation.invalidated_freeze(), Some(&token));
        assert_eq!(state.configuration_epoch(), 2);
        assert_eq!(state.configuration_digest(), digest(0x22));
        assert!(state.active_freeze().is_none());
        assert_eq!(
            state.require_active_freeze(&token),
            Err(SafetyConfigurationStateError::NotFrozen)
        );
    }

    #[test]
    fn stale_external_mutation_does_not_destroy_active_freeze() {
        let mut state = SafetyConfigurationState::initialize("rack", digest(0x11)).unwrap();
        let token = state.begin_freeze(challenge(0x44)).unwrap();
        assert_eq!(
            state.record_external_mutation(9, digest(0x22)),
            Err(SafetyConfigurationStateError::ConfigurationEpochMismatch {
                expected: 1,
                observed: 9,
            })
        );
        state.require_active_freeze(&token).unwrap();
        assert_eq!(state.configuration_epoch(), 1);
    }

    #[test]
    fn freeze_release_reacquire_has_new_generation_and_rejects_old_token() {
        let mut state = SafetyConfigurationState::initialize("rack", digest(0x11)).unwrap();
        let first = state.begin_freeze(challenge(0x44)).unwrap();
        state.end_freeze(&first).unwrap();
        let second = state.begin_freeze(challenge(0x44)).unwrap();
        assert_eq!(first.configuration_epoch(), second.configuration_epoch());
        assert_eq!(first.configuration_digest(), second.configuration_digest());
        assert_ne!(first.freeze_generation(), second.freeze_generation());
        assert_eq!(
            state.require_active_freeze(&first),
            Err(SafetyConfigurationStateError::FreezeTokenMismatch)
        );
        state.require_active_freeze(&second).unwrap();
    }

    #[test]
    fn nested_freeze_is_rejected() {
        let mut state = SafetyConfigurationState::initialize("rack", digest(0x11)).unwrap();
        state.begin_freeze(challenge(0x44)).unwrap();
        assert_eq!(
            state.begin_freeze(challenge(0x55)),
            Err(SafetyConfigurationStateError::AlreadyFrozen)
        );
    }

    #[test]
    fn stale_epoch_cannot_mutate_state() {
        let mut state = SafetyConfigurationState::initialize("rack", digest(0x11)).unwrap();
        state.apply_managed_mutation(1, digest(0x22)).unwrap();
        assert_eq!(
            state.apply_managed_mutation(1, digest(0x33)),
            Err(SafetyConfigurationStateError::ConfigurationEpochMismatch {
                expected: 2,
                observed: 1,
            })
        );
        assert_eq!(state.configuration_epoch(), 2);
        assert_eq!(state.configuration_digest(), digest(0x22));
    }
}