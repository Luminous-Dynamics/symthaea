// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TOCTOU-resistant commit preconditions for frozen configuration observations.
//!
//! The ordinary observation commit layer persists one accepted observation lineage
//! head plus its exact inner observation digest. Once observations are bound to a
//! local configuration freeze, that is no longer enough: the same inner observation
//! could otherwise lose the configuration-epoch/freeze-generation identity carried
//! by the stronger frozen wrapper.
//!
//! This module therefore composes the existing observation commit preconditions with
//! exact live-freeze continuity and a second durable marker for the canonical frozen
//! observation. It proves no signature and performs no persistence.

use crate::frozen_observation_admission::PolicyCheckedFrozenSafetyConfigurationObservation;
use crate::observation_admission::{
    ConfigurationObserverRootSnapshot, SafetyConfigurationObservationHead,
};
use crate::observation_commit_preconditions::{
    SafetyConfigurationObservationCommitError, SafetyConfigurationObservationCommitPreconditions,
    SafetyConfigurationObservationCommitState,
};
use symthaea_safety_configuration::frozen_observation::
    SafetyConfigurationFrozenObservationDigest;
use symthaea_safety_configuration::observation::{
    ConfigurationObservationChallenge, SafetyConfigurationObservationDigest,
};
use symthaea_safety_configuration::state::{
    SafetyConfigurationFreezeToken, SafetyConfigurationState, SafetyConfigurationStateError,
};
use symthaea_safety_profile::trusted_time::TrustedAuthorizationClockObservation;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrozenSafetyConfigurationObservationCommitState {
    ReadyToCommit,
    /// Historical acknowledgement only: both exact markers prove this write already
    /// happened. It does not claim the observation remains fresh or the freeze live.
    AlreadyCommitted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenSafetyConfigurationObservationCommitPreconditions {
    inner: SafetyConfigurationObservationCommitPreconditions,
    canonical_frozen_observation_bytes: Vec<u8>,
    frozen_observation_digest: SafetyConfigurationFrozenObservationDigest,
    expected_freeze: SafetyConfigurationFreezeToken,
}

impl FrozenSafetyConfigurationObservationCommitPreconditions {
    pub fn from_policy_checked(
        checked: &PolicyCheckedFrozenSafetyConfigurationObservation,
    ) -> Self {
        Self {
            inner: SafetyConfigurationObservationCommitPreconditions::from_policy_checked(
                checked.checked_observation(),
            ),
            canonical_frozen_observation_bytes: checked
                .canonical_frozen_observation_bytes()
                .to_vec(),
            frozen_observation_digest: checked.frozen_observation_digest(),
            expected_freeze: checked.expected_freeze().clone(),
        }
    }

    pub fn inner(&self) -> &SafetyConfigurationObservationCommitPreconditions {
        &self.inner
    }

    pub fn canonical_frozen_observation_bytes(&self) -> &[u8] {
        &self.canonical_frozen_observation_bytes
    }

    pub fn frozen_observation_digest(&self) -> SafetyConfigurationFrozenObservationDigest {
        self.frozen_observation_digest
    }

    pub fn expected_freeze(&self) -> &SafetyConfigurationFreezeToken {
        &self.expected_freeze
    }

    /// Recheck one atomic observer/lifecycle snapshot immediately before persistence.
    ///
    /// A successful new write must atomically:
    /// - advance the ordinary observation head,
    /// - store the exact inner observation marker, and
    /// - store the exact frozen-observation marker for that same candidate generation.
    #[allow(clippy::too_many_arguments)]
    pub fn recheck_commit_observation(
        &self,
        current_clock: &TrustedAuthorizationClockObservation,
        current_observer_root: &ConfigurationObserverRootSnapshot,
        current_observation_head: &SafetyConfigurationObservationHead,
        current_challenge: ConfigurationObservationChallenge,
        candidate_observation_marker: Option<SafetyConfigurationObservationDigest>,
        candidate_frozen_observation_marker: Option<SafetyConfigurationFrozenObservationDigest>,
        current_configuration_state: &SafetyConfigurationState,
    ) -> Result<
        FrozenSafetyConfigurationObservationCommitState,
        FrozenSafetyConfigurationObservationCommitError,
    > {
        if let Some(observed) = candidate_frozen_observation_marker {
            if observed != self.frozen_observation_digest {
                return Err(
                    FrozenSafetyConfigurationObservationCommitError::FrozenObservationMarkerConflict {
                        expected: self.frozen_observation_digest,
                        observed,
                    },
                );
            }
        }

        let inner_state = self.inner.recheck_commit_observation(
            current_clock,
            current_observer_root,
            current_observation_head,
            current_challenge,
            candidate_observation_marker,
        )?;

        match inner_state {
            SafetyConfigurationObservationCommitState::AlreadyCommitted => {
                if candidate_frozen_observation_marker.is_none() {
                    return Err(
                        FrozenSafetyConfigurationObservationCommitError::CommittedObservationMissingFrozenMarker,
                    );
                }
                Ok(FrozenSafetyConfigurationObservationCommitState::AlreadyCommitted)
            }
            SafetyConfigurationObservationCommitState::ReadyToCommit => {
                if candidate_frozen_observation_marker.is_some() {
                    return Err(
                        FrozenSafetyConfigurationObservationCommitError::OrphanFrozenObservationMarker,
                    );
                }
                current_configuration_state.require_active_freeze(&self.expected_freeze)?;
                Ok(FrozenSafetyConfigurationObservationCommitState::ReadyToCommit)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FrozenSafetyConfigurationObservationCommitError {
    #[error(transparent)]
    ObservationCommit(#[from] SafetyConfigurationObservationCommitError),
    #[error(transparent)]
    ConfigurationState(#[from] SafetyConfigurationStateError),
    #[error("committed observation head/marker is missing its atomically paired frozen-observation marker")]
    CommittedObservationMissingFrozenMarker,
    #[error("frozen-observation marker exists before the inner observation write is committed")]
    OrphanFrozenObservationMarker,
    #[error("a different frozen-observation marker is already committed for this observation generation")]
    FrozenObservationMarkerConflict {
        expected: SafetyConfigurationFrozenObservationDigest,
        observed: SafetyConfigurationFrozenObservationDigest,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frozen_observation_admission::admit_frozen_safety_configuration_observation;
    use crate::observation_admission::SafetyConfigurationObservationAdmissionPolicy;
    use crate::qualify_safety_configuration;
    use symthaea_safety_configuration::frozen_observation::SafetyConfigurationFrozenObservation;
    use symthaea_safety_configuration::observation::{
        ConfigurationObserverRootDigest, SafetyConfigurationObservation,
    };
    use symthaea_safety_configuration::{
        ConfigurationComponent, ConfigurationDigest, SafetyConfigurationManifest,
        SAFETY_CONFIGURATION_SCHEMA_V1,
    };
    use symthaea_safety_profile::{
        ComponentRequirement, SafetyConfigurationProfile,
        SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
    };

    fn digest(byte: u8) -> ConfigurationDigest {
        ConfigurationDigest::Blake3_256([byte; 32])
    }

    fn observer_root(byte: u8) -> ConfigurationObserverRootDigest {
        ConfigurationObserverRootDigest::Blake3_256([byte; 32])
    }

    fn challenge(byte: u8) -> ConfigurationObservationChallenge {
        ConfigurationObservationChallenge::new([byte; 32]).unwrap()
    }

    fn clock(earliest: i64, latest: i64) -> TrustedAuthorizationClockObservation {
        TrustedAuthorizationClockObservation::new("secure-rtc-v1", 7, earliest, latest).unwrap()
    }

    fn profile() -> SafetyConfigurationProfile {
        SafetyConfigurationProfile {
            schema_version: SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1.to_owned(),
            profile_id: "compute-commons-autonomous-node-v1".to_owned(),
            hardware_inventory: ComponentRequirement::Required,
            firmware: ComponentRequirement::Required,
            software_closure: ComponentRequirement::Required,
            electrical_topology: ComponentRequirement::Required,
            thermal_topology: ComponentRequirement::Required,
            protection_settings: ComponentRequirement::Required,
            sensor_map: ComponentRequirement::Required,
            actuator_map: ComponentRequirement::Required,
            calibration: ComponentRequirement::Required,
            network_topology: ComponentRequirement::Required,
        }
    }

    fn qualified() -> crate::QualifiedSafetyConfiguration {
        let profile = profile();
        qualify_safety_configuration(
            &profile,
            SafetyConfigurationManifest {
                schema_version: SAFETY_CONFIGURATION_SCHEMA_V1.to_owned(),
                node_id: "rack".to_owned(),
                profile_id: profile.profile_id.clone(),
                profile_digest: profile.digest().unwrap(),
                hardware_inventory: ConfigurationComponent::Digest(digest(0x01)),
                firmware: ConfigurationComponent::Digest(digest(0x02)),
                software_closure: ConfigurationComponent::Digest(digest(0x03)),
                electrical_topology: ConfigurationComponent::Digest(digest(0x04)),
                thermal_topology: ConfigurationComponent::Digest(digest(0x05)),
                protection_settings: ConfigurationComponent::Digest(digest(0x06)),
                sensor_map: ConfigurationComponent::Digest(digest(0x07)),
                actuator_map: ConfigurationComponent::Digest(digest(0x08)),
                calibration: ConfigurationComponent::Digest(digest(0x09)),
                network_topology: ConfigurationComponent::Digest(digest(0x0a)),
            },
        )
        .unwrap()
    }

    fn setup() -> (
        FrozenSafetyConfigurationObservationCommitPreconditions,
        ConfigurationObserverRootSnapshot,
        SafetyConfigurationState,
    ) {
        let qualified = qualified();
        let root = ConfigurationObserverRootSnapshot::new(
            "observer-1",
            observer_root(0x44),
            3,
        )
        .unwrap();
        let mut state =
            SafetyConfigurationState::initialize("rack", qualified.configuration_digest()).unwrap();
        let freeze = state.begin_freeze(challenge(0x55)).unwrap();
        let observation = SafetyConfigurationObservation::new(
            "obs-1",
            "observer-1",
            "rack",
            1,
            1_500,
            observer_root(0x44),
            challenge(0x55),
            qualified.configuration_digest(),
        )
        .unwrap();
        let frozen = SafetyConfigurationFrozenObservation::new(observation, &state, &freeze).unwrap();
        let policy = SafetyConfigurationObservationAdmissionPolicy::new(
            "rack",
            root.clone(),
            SafetyConfigurationObservationHead::Uninitialized,
            challenge(0x55),
            500,
        )
        .unwrap();
        let checked = admit_frozen_safety_configuration_observation(
            &policy,
            &clock(1_550, 1_600),
            &frozen,
            &state,
            &freeze,
            &qualified,
        )
        .unwrap();
        (
            FrozenSafetyConfigurationObservationCommitPreconditions::from_policy_checked(&checked),
            root,
            state,
        )
    }

    #[test]
    fn exact_live_state_is_ready_to_commit() {
        let (preconditions, root, state) = setup();
        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock(1_600, 1_650),
                    &root,
                    preconditions.inner().expected_predecessor_head(),
                    challenge(0x55),
                    None,
                    None,
                    &state,
                )
                .unwrap(),
            FrozenSafetyConfigurationObservationCommitState::ReadyToCommit
        );
    }

    #[test]
    fn out_of_band_mutation_invalidates_fresh_commit() {
        let (preconditions, root, mut state) = setup();
        state
            .record_external_mutation(state.configuration_epoch(), digest(0xee))
            .unwrap();
        assert!(matches!(
            preconditions.recheck_commit_observation(
                &clock(1_600, 1_650),
                &root,
                preconditions.inner().expected_predecessor_head(),
                challenge(0x55),
                None,
                None,
                &state,
            ),
            Err(FrozenSafetyConfigurationObservationCommitError::ConfigurationState(_))
        ));
    }

    #[test]
    fn exact_dual_markers_are_historical_idempotent_acknowledgement() {
        let (preconditions, root, mut state) = setup();
        let old_freeze = preconditions.expected_freeze().clone();
        state.end_freeze(&old_freeze).unwrap();
        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock(9_000, 9_100),
                    &root,
                    preconditions.inner().candidate_head(),
                    challenge(0x99),
                    Some(preconditions.inner().observation_digest()),
                    Some(preconditions.frozen_observation_digest()),
                    &state,
                )
                .unwrap(),
            FrozenSafetyConfigurationObservationCommitState::AlreadyCommitted
        );
    }

    #[test]
    fn committed_inner_observation_without_frozen_marker_is_inconsistent() {
        let (preconditions, root, state) = setup();
        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock(1_600, 1_650),
                &root,
                preconditions.inner().candidate_head(),
                challenge(0x55),
                Some(preconditions.inner().observation_digest()),
                None,
                &state,
            ),
            Err(
                FrozenSafetyConfigurationObservationCommitError::CommittedObservationMissingFrozenMarker
            )
        );
    }

    #[test]
    fn orphan_frozen_marker_fails_closed() {
        let (preconditions, root, state) = setup();
        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock(1_600, 1_650),
                &root,
                preconditions.inner().expected_predecessor_head(),
                challenge(0x55),
                None,
                Some(preconditions.frozen_observation_digest()),
                &state,
            ),
            Err(FrozenSafetyConfigurationObservationCommitError::OrphanFrozenObservationMarker)
        );
    }

    #[test]
    fn different_frozen_marker_is_equivocation() {
        let (preconditions, root, state) = setup();
        assert!(matches!(
            preconditions.recheck_commit_observation(
                &clock(1_600, 1_650),
                &root,
                preconditions.inner().candidate_head(),
                challenge(0x55),
                Some(preconditions.inner().observation_digest()),
                Some(SafetyConfigurationFrozenObservationDigest::Blake3_256([0xff; 32])),
                &state,
            ),
            Err(
                FrozenSafetyConfigurationObservationCommitError::FrozenObservationMarkerConflict {
                    ..
                }
            )
        ));
    }
}
