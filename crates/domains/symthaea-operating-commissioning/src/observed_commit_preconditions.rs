// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TOCTOU-resistant commit preconditions for observed commissioning.
//!
//! This is the final non-cryptographic composition boundary between commissioning
//! authority and current-configuration evidence. It combines the existing
//! commissioning commit capsule with the frozen-observation commit capsule and the
//! exact outer observed-commissioning policy witness.
//!
//! A fresh write is allowed only when both inner capsules remain `ReadyToCommit` at
//! the same observation and no outer marker already exists. Historical retry is
//! acknowledged only when both inner writes are already committed and the exact
//! outer observed-commissioning marker is present. Mixed/partial states fail closed.
//!
//! This module proves no signature and performs no persistence. Production must
//! additionally require independent verifier-owned proof for the exact frozen
//! observation under the observer root and the exact outer observed-commissioning
//! bytes under the commissioning root before atomically writing all markers/state.

use crate::admission::{CommissioningAuthorityRootSnapshot, CommissioningAuthorizationHead};
use crate::commit_preconditions::{
    CommissioningAuthorizationCommitError, CommissioningAuthorizationCommitPreconditions,
    CommissioningAuthorizationCommitState,
};
use crate::identity::CommissioningRecordDigest;
use crate::observed_admission::PolicyCheckedObservedCommissioningAuthorization;
use crate::observed_authorization::ObservedCommissioningAuthorizationDigest;
use symthaea_safety_configuration::frozen_observation::
    SafetyConfigurationFrozenObservationDigest;
use symthaea_safety_configuration::observation::{
    ConfigurationObservationChallenge, SafetyConfigurationObservationDigest,
};
use symthaea_safety_configuration::state::SafetyConfigurationState;
use symthaea_safety_profile::commit_preconditions::ProfileAuthorityRootSnapshot;
use symthaea_safety_profile::lifecycle::SafetyProfileAuthorizationLifecycleState;
use symthaea_safety_profile::trusted_time::TrustedAuthorizationClockObservation;
use symthaea_safety_qualification::frozen_observation_commit_preconditions::{
    FrozenSafetyConfigurationObservationCommitError,
    FrozenSafetyConfigurationObservationCommitPreconditions,
    FrozenSafetyConfigurationObservationCommitState,
};
use symthaea_safety_qualification::observation_admission::{
    ConfigurationObserverRootSnapshot, SafetyConfigurationObservationHead,
};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservedCommissioningAuthorizationCommitState {
    ReadyToCommit,
    /// Historical acknowledgement only. All exact durable markers prove this
    /// composite transaction was committed previously; current authority/freshness
    /// is deliberately not implied.
    AlreadyCommitted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservedCommissioningAuthorizationCommitPreconditions {
    commissioning: CommissioningAuthorizationCommitPreconditions,
    frozen_observation: FrozenSafetyConfigurationObservationCommitPreconditions,
    canonical_observed_authorization_bytes: Vec<u8>,
    observed_authorization_digest: ObservedCommissioningAuthorizationDigest,
}

impl ObservedCommissioningAuthorizationCommitPreconditions {
    pub fn from_policy_checked(
        checked: &PolicyCheckedObservedCommissioningAuthorization,
        expected_profile_root: ProfileAuthorityRootSnapshot,
    ) -> Result<Self, ObservedCommissioningAuthorizationCommitError> {
        let commissioning = CommissioningAuthorizationCommitPreconditions::from_policy_checked(
            checked.checked_commissioning(),
            expected_profile_root,
        )?;
        let frozen_observation =
            FrozenSafetyConfigurationObservationCommitPreconditions::from_policy_checked(
                checked.checked_frozen_observation(),
            );

        let commissioning_configuration = commissioning.configuration_digest();
        let frozen_configuration = frozen_observation.expected_freeze().configuration_digest();
        if commissioning_configuration != frozen_configuration {
            return Err(
                ObservedCommissioningAuthorizationCommitError::ConfigurationIdentityMismatch,
            );
        }

        Ok(Self {
            commissioning,
            frozen_observation,
            canonical_observed_authorization_bytes: checked
                .canonical_observed_authorization_bytes()
                .to_vec(),
            observed_authorization_digest: checked.observed_authorization_digest(),
        })
    }

    pub fn commissioning(&self) -> &CommissioningAuthorizationCommitPreconditions {
        &self.commissioning
    }

    pub fn frozen_observation(
        &self,
    ) -> &FrozenSafetyConfigurationObservationCommitPreconditions {
        &self.frozen_observation
    }

    pub fn canonical_observed_authorization_bytes(&self) -> &[u8] {
        &self.canonical_observed_authorization_bytes
    }

    pub fn observed_authorization_digest(&self) -> ObservedCommissioningAuthorizationDigest {
        self.observed_authorization_digest
    }

    /// Recheck one complete commissioning + observer + physical-configuration
    /// transaction immediately before durable commit.
    ///
    /// A successful fresh store must atomically commit the commissioning candidate,
    /// the observation candidate and both observation markers, plus the exact outer
    /// observed-commissioning marker keyed by the candidate commissioning transition.
    #[allow(clippy::too_many_arguments)]
    pub fn recheck_commit_observation(
        &self,
        current_clock: &TrustedAuthorizationClockObservation,
        current_commissioning_root: &CommissioningAuthorityRootSnapshot,
        current_profile_root: &ProfileAuthorityRootSnapshot,
        current_commissioning_head: &CommissioningAuthorizationHead,
        current_profile_lifecycle: &SafetyProfileAuthorizationLifecycleState,
        candidate_record_digest_if_committed: Option<CommissioningRecordDigest>,
        current_observer_root: &ConfigurationObserverRootSnapshot,
        current_observation_head: &SafetyConfigurationObservationHead,
        current_challenge: ConfigurationObservationChallenge,
        candidate_observation_marker: Option<SafetyConfigurationObservationDigest>,
        candidate_frozen_observation_marker: Option<SafetyConfigurationFrozenObservationDigest>,
        current_configuration_state: &SafetyConfigurationState,
        candidate_observed_authorization_marker: Option<ObservedCommissioningAuthorizationDigest>,
    ) -> Result<
        ObservedCommissioningAuthorizationCommitState,
        ObservedCommissioningAuthorizationCommitError,
    > {
        if let Some(observed) = candidate_observed_authorization_marker {
            if observed != self.observed_authorization_digest {
                return Err(
                    ObservedCommissioningAuthorizationCommitError::ObservedAuthorizationMarkerConflict {
                        expected: self.observed_authorization_digest,
                        observed,
                    },
                );
            }
        }

        // The production-oriented path no longer accepts a naked configuration
        // digest: commissioning sees the digest owned by the same live state machine
        // whose exact freeze is rechecked by the frozen-observation capsule below.
        let commissioning_state = self.commissioning.recheck_commit_observation(
            current_clock,
            current_commissioning_root,
            current_profile_root,
            current_commissioning_head,
            current_profile_lifecycle,
            current_configuration_state.configuration_digest(),
            candidate_record_digest_if_committed,
        )?;

        let frozen_state = self.frozen_observation.recheck_commit_observation(
            current_clock,
            current_observer_root,
            current_observation_head,
            current_challenge,
            candidate_observation_marker,
            candidate_frozen_observation_marker,
            current_configuration_state,
        )?;

        match (commissioning_state, frozen_state) {
            (
                CommissioningAuthorizationCommitState::ReadyToCommit,
                FrozenSafetyConfigurationObservationCommitState::ReadyToCommit,
            ) => {
                if candidate_observed_authorization_marker.is_some() {
                    return Err(
                        ObservedCommissioningAuthorizationCommitError::OrphanObservedAuthorizationMarker,
                    );
                }
                Ok(ObservedCommissioningAuthorizationCommitState::ReadyToCommit)
            }
            (
                CommissioningAuthorizationCommitState::AlreadyCommitted,
                FrozenSafetyConfigurationObservationCommitState::AlreadyCommitted,
            ) => {
                if candidate_observed_authorization_marker.is_none() {
                    return Err(
                        ObservedCommissioningAuthorizationCommitError::CommittedComponentsMissingObservedMarker,
                    );
                }
                Ok(ObservedCommissioningAuthorizationCommitState::AlreadyCommitted)
            }
            (commissioning, observation) => Err(
                ObservedCommissioningAuthorizationCommitError::PartialCompositeCommit {
                    commissioning,
                    observation,
                },
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ObservedCommissioningAuthorizationCommitError {
    #[error(transparent)]
    Commissioning(#[from] CommissioningAuthorizationCommitError),
    #[error(transparent)]
    FrozenObservation(#[from] FrozenSafetyConfigurationObservationCommitError),
    #[error("commissioning and frozen-observation preconditions bind different configuration identities")]
    ConfigurationIdentityMismatch,
    #[error("a different observed-commissioning marker is already committed for this candidate transition")]
    ObservedAuthorizationMarkerConflict {
        expected: ObservedCommissioningAuthorizationDigest,
        observed: ObservedCommissioningAuthorizationDigest,
    },
    #[error("observed-commissioning marker exists before its commissioning and observation components are committed")]
    OrphanObservedAuthorizationMarker,
    #[error("commissioning and frozen-observation components are committed without their atomically paired outer observed-commissioning marker")]
    CommittedComponentsMissingObservedMarker,
    #[error("observed commissioning is only partially committed: commissioning {commissioning:?}, observation {observation:?}")]
    PartialCompositeCommit {
        commissioning: CommissioningAuthorizationCommitState,
        observation: FrozenSafetyConfigurationObservationCommitState,
    },
}
