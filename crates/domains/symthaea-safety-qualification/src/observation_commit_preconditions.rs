// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TOCTOU-resistant commit preconditions for current-configuration observations.
//!
//! Observation admission proves that one raw observation is locally admissible at
//! one instant. Detached cryptographic verification may take time. During that
//! interval the observer root, observation lineage, transaction challenge, trusted
//! clock lineage, or freshness status may change. This module captures the exact
//! admitted state and requires it to remain current immediately before persistence.
//!
//! This module proves no signature and performs no persistence. A production store
//! must additionally require verifier-owned proof over
//! [`SafetyConfigurationObservationCommitPreconditions::canonical_observation_bytes`]
//! under the exact captured observer root before atomically writing the candidate
//! observation head and exact observation marker.

use crate::observation_admission::{
    ConfigurationObserverRootSnapshot, PolicyCheckedCurrentSafetyConfigurationObservation,
    SafetyConfigurationObservationHead,
};
use symthaea_safety_configuration::ConfigurationDigest;
use symthaea_safety_configuration::observation::{
    ConfigurationObservationChallenge, SafetyConfigurationObservationDigest,
};
use symthaea_safety_profile::trusted_time::TrustedAuthorizationClockObservation;
use thiserror::Error;

/// Result of rechecking one atomic observation-store snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyConfigurationObservationCommitState {
    /// Every captured policy precondition remains current. Persistence still
    /// requires verifier-owned cryptographic proof over the exact observation bytes.
    ReadyToCommit,
    /// The exact candidate observation was already durably accepted. Historical
    /// acknowledgement does not imply the observation remains fresh enough for a
    /// new commissioning action now.
    AlreadyCommitted,
}

/// Exact non-cryptographic state that must remain stable while one observation is
/// cryptographically verified and committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyConfigurationObservationCommitPreconditions {
    canonical_observation_bytes: Vec<u8>,
    observation_digest: SafetyConfigurationObservationDigest,
    expected_observer_root: ConfigurationObserverRootSnapshot,
    expected_predecessor_head: SafetyConfigurationObservationHead,
    candidate_head: SafetyConfigurationObservationHead,
    expected_challenge: ConfigurationObservationChallenge,
    expected_clock_source_id: String,
    expected_clock_epoch: u64,
    observed_at_unix_ms: i64,
    max_age_ms: i64,
    configuration_digest: ConfigurationDigest,
}

impl SafetyConfigurationObservationCommitPreconditions {
    pub fn from_policy_checked(
        checked: &PolicyCheckedCurrentSafetyConfigurationObservation,
    ) -> Self {
        Self {
            canonical_observation_bytes: checked.canonical_observation_bytes().to_vec(),
            observation_digest: checked.observation_digest(),
            expected_observer_root: checked.expected_observer_root().clone(),
            expected_predecessor_head: checked.expected_predecessor_head().clone(),
            candidate_head: checked.candidate_head().clone(),
            expected_challenge: checked.expected_challenge(),
            expected_clock_source_id: checked.expected_clock_source_id().to_owned(),
            expected_clock_epoch: checked.expected_clock_epoch(),
            observed_at_unix_ms: checked.observation().observed_at_unix_ms(),
            max_age_ms: checked.max_age_ms(),
            configuration_digest: checked.configuration_digest(),
        }
    }

    pub fn canonical_observation_bytes(&self) -> &[u8] {
        &self.canonical_observation_bytes
    }

    pub fn observation_digest(&self) -> SafetyConfigurationObservationDigest {
        self.observation_digest
    }

    pub fn expected_observer_root(&self) -> &ConfigurationObserverRootSnapshot {
        &self.expected_observer_root
    }

    pub fn expected_predecessor_head(&self) -> &SafetyConfigurationObservationHead {
        &self.expected_predecessor_head
    }

    pub fn candidate_head(&self) -> &SafetyConfigurationObservationHead {
        &self.candidate_head
    }

    pub fn expected_challenge(&self) -> ConfigurationObservationChallenge {
        self.expected_challenge
    }

    pub fn observed_at_unix_ms(&self) -> i64 {
        self.observed_at_unix_ms
    }

    pub fn max_age_ms(&self) -> i64 {
        self.max_age_ms
    }

    pub fn configuration_digest(&self) -> ConfigurationDigest {
        self.configuration_digest
    }

    /// Recheck one atomic observer-root + observation-head + challenge + clock
    /// snapshot immediately before durable acceptance.
    ///
    /// `candidate_observation_marker` is the durable observation digest stored for
    /// this exact candidate generation, if any. A successful write must atomically
    /// advance the observation head and store the exact marker.
    pub fn recheck_commit_observation(
        &self,
        current_clock: &TrustedAuthorizationClockObservation,
        current_observer_root: &ConfigurationObserverRootSnapshot,
        current_observation_head: &SafetyConfigurationObservationHead,
        current_challenge: ConfigurationObservationChallenge,
        candidate_observation_marker: Option<SafetyConfigurationObservationDigest>,
    ) -> Result<
        SafetyConfigurationObservationCommitState,
        SafetyConfigurationObservationCommitError,
    > {
        // Historical exact retry acknowledgement is intentionally independent of
        // current freshness/root/challenge state. It only says this exact write
        // already happened.
        if let Some(observed) = candidate_observation_marker {
            return if observed == self.observation_digest {
                Ok(SafetyConfigurationObservationCommitState::AlreadyCommitted)
            } else {
                Err(SafetyConfigurationObservationCommitError::CandidateMarkerConflict {
                    expected: self.observation_digest,
                    observed,
                })
            };
        }

        if current_observation_head == &self.candidate_head {
            return Err(
                SafetyConfigurationObservationCommitError::CandidateHeadMissingObservationMarker,
            );
        }
        if current_observer_root != &self.expected_observer_root {
            return Err(SafetyConfigurationObservationCommitError::ObserverRootSnapshotChanged);
        }
        if current_observation_head != &self.expected_predecessor_head {
            return Err(SafetyConfigurationObservationCommitError::ObservationHeadChanged);
        }
        if current_challenge != self.expected_challenge {
            return Err(SafetyConfigurationObservationCommitError::ChallengeChanged);
        }
        if current_clock.source_id() != self.expected_clock_source_id
            || current_clock.epoch() != self.expected_clock_epoch
        {
            return Err(SafetyConfigurationObservationCommitError::ClockLineageChanged {
                expected_source_id: self.expected_clock_source_id.clone(),
                expected_epoch: self.expected_clock_epoch,
                observed_source_id: current_clock.source_id().to_owned(),
                observed_epoch: current_clock.epoch(),
            });
        }

        validate_freshness(self.observed_at_unix_ms, self.max_age_ms, current_clock)?;
        Ok(SafetyConfigurationObservationCommitState::ReadyToCommit)
    }
}

fn validate_freshness(
    observed_at_unix_ms: i64,
    max_age_ms: i64,
    clock: &TrustedAuthorizationClockObservation,
) -> Result<(), SafetyConfigurationObservationCommitError> {
    if observed_at_unix_ms > clock.latest_unix_ms() {
        return Err(SafetyConfigurationObservationCommitError::DefinitelyFromFuture {
            observed_at_unix_ms,
            latest_unix_ms: clock.latest_unix_ms(),
        });
    }
    if observed_at_unix_ms > clock.earliest_unix_ms() {
        return Err(
            SafetyConfigurationObservationCommitError::ClockUncertaintyCrossesObservationTime {
                observed_at_unix_ms,
                earliest_unix_ms: clock.earliest_unix_ms(),
                latest_unix_ms: clock.latest_unix_ms(),
            },
        );
    }
    let worst_case_age_ms =
        i128::from(clock.latest_unix_ms()) - i128::from(observed_at_unix_ms);
    if worst_case_age_ms > i128::from(max_age_ms) {
        return Err(SafetyConfigurationObservationCommitError::ObservationTooOld {
            observed_at_unix_ms,
            latest_unix_ms: clock.latest_unix_ms(),
            max_age_ms,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyConfigurationObservationCommitError {
    #[error("configuration-observer root snapshot changed before observation commit")]
    ObserverRootSnapshotChanged,
    #[error("configuration observation lineage head changed before observation commit")]
    ObservationHeadChanged,
    #[error("configuration observation transaction challenge changed before observation commit")]
    ChallengeChanged,
    #[error("trusted clock lineage changed before observation commit: expected {expected_source_id}@{expected_epoch}, observed {observed_source_id}@{observed_epoch}")]
    ClockLineageChanged {
        expected_source_id: String,
        expected_epoch: u64,
        observed_source_id: String,
        observed_epoch: u64,
    },
    #[error("candidate observation head is current without its atomically paired observation marker")]
    CandidateHeadMissingObservationMarker,
    #[error("a different observation marker is already committed for this candidate generation")]
    CandidateMarkerConflict {
        expected: SafetyConfigurationObservationDigest,
        observed: SafetyConfigurationObservationDigest,
    },
    #[error("configuration observation timestamp {observed_at_unix_ms} is definitely after latest possible current time {latest_unix_ms}")]
    DefinitelyFromFuture {
        observed_at_unix_ms: i64,
        latest_unix_ms: i64,
    },
    #[error("trusted clock uncertainty [{earliest_unix_ms}, {latest_unix_ms}] crosses observation time {observed_at_unix_ms}")]
    ClockUncertaintyCrossesObservationTime {
        observed_at_unix_ms: i64,
        earliest_unix_ms: i64,
        latest_unix_ms: i64,
    },
    #[error("configuration observation at {observed_at_unix_ms} is older than maximum age {max_age_ms}ms at latest possible time {latest_unix_ms}")]
    ObservationTooOld {
        observed_at_unix_ms: i64,
        latest_unix_ms: i64,
        max_age_ms: i64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation_admission::SafetyConfigurationObservationAdmissionPolicy;
    use crate::qualify_safety_configuration;
    use symthaea_safety_configuration::observation::{
        ConfigurationObserverRootDigest, SafetyConfigurationObservation,
    };
    use symthaea_safety_configuration::{
        ConfigurationComponent, SafetyConfigurationManifest, SAFETY_CONFIGURATION_SCHEMA_V1,
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

    fn manifest(profile: &SafetyConfigurationProfile) -> SafetyConfigurationManifest {
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
        }
    }

    fn qualified() -> crate::QualifiedSafetyConfiguration {
        let profile = profile();
        qualify_safety_configuration(&profile, manifest(&profile)).unwrap()
    }

    fn root_snapshot(epoch: u64) -> ConfigurationObserverRootSnapshot {
        ConfigurationObserverRootSnapshot::new("observer-1", observer_root(0x44), epoch).unwrap()
    }

    fn observation(config: ConfigurationDigest) -> SafetyConfigurationObservation {
        SafetyConfigurationObservation::new(
            "obs-1",
            "observer-1",
            "rack",
            1,
            1_500,
            observer_root(0x44),
            challenge(0x55),
            config,
        )
        .unwrap()
    }

    fn clock(source: &str, epoch: u64, earliest: i64, latest: i64) -> TrustedAuthorizationClockObservation {
        TrustedAuthorizationClockObservation::new(source, epoch, earliest, latest).unwrap()
    }

    fn preconditions() -> SafetyConfigurationObservationCommitPreconditions {
        let qualified = qualified();
        let observation = observation(qualified.configuration_digest());
        let policy = SafetyConfigurationObservationAdmissionPolicy::new(
            "rack",
            root_snapshot(3),
            SafetyConfigurationObservationHead::Uninitialized,
            challenge(0x55),
            500,
        )
        .unwrap();
        let checked = policy
            .check(&clock("secure-rtc-v1", 7, 1_550, 1_600), &observation, &qualified)
            .unwrap();
        SafetyConfigurationObservationCommitPreconditions::from_policy_checked(&checked)
    }

    #[test]
    fn exact_state_is_ready_to_commit() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock("secure-rtc-v1", 7, 1_600, 1_650),
                    &root_snapshot(3),
                    preconditions.expected_predecessor_head(),
                    challenge(0x55),
                    None,
                )
                .unwrap(),
            SafetyConfigurationObservationCommitState::ReadyToCommit
        );
    }

    #[test]
    fn same_key_observer_root_reprovisioning_invalidates_commit() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_600, 1_650),
                &root_snapshot(4),
                preconditions.expected_predecessor_head(),
                challenge(0x55),
                None,
            ),
            Err(SafetyConfigurationObservationCommitError::ObserverRootSnapshotChanged)
        );
    }

    #[test]
    fn challenge_change_invalidates_commit() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_600, 1_650),
                &root_snapshot(3),
                preconditions.expected_predecessor_head(),
                challenge(0x56),
                None,
            ),
            Err(SafetyConfigurationObservationCommitError::ChallengeChanged)
        );
    }

    #[test]
    fn observation_lineage_race_invalidates_commit() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_600, 1_650),
                &root_snapshot(3),
                preconditions.candidate_head(),
                challenge(0x55),
                None,
            ),
            Err(
                SafetyConfigurationObservationCommitError::CandidateHeadMissingObservationMarker
            )
        );
    }

    #[test]
    fn exact_history_marker_is_idempotent_acknowledgement() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock("other-clock", 99, 9_000, 9_100),
                    &root_snapshot(99),
                    preconditions.candidate_head(),
                    challenge(0x56),
                    Some(preconditions.observation_digest()),
                )
                .unwrap(),
            SafetyConfigurationObservationCommitState::AlreadyCommitted
        );
    }

    #[test]
    fn conflicting_history_marker_fails_closed() {
        let preconditions = preconditions();
        let observed = SafetyConfigurationObservationDigest::Blake3_256([0xee; 32]);
        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_600, 1_650),
                &root_snapshot(3),
                preconditions.expected_predecessor_head(),
                challenge(0x55),
                Some(observed),
            ),
            Err(SafetyConfigurationObservationCommitError::CandidateMarkerConflict {
                expected: preconditions.observation_digest(),
                observed,
            })
        );
    }

    #[test]
    fn freshness_is_rechecked_after_crypto_delay() {
        let preconditions = preconditions();
        assert!(matches!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_950, 2_001),
                &root_snapshot(3),
                preconditions.expected_predecessor_head(),
                challenge(0x55),
                None,
            ),
            Err(SafetyConfigurationObservationCommitError::ObservationTooOld { .. })
        ));
    }

    #[test]
    fn clock_lineage_change_invalidates_commit() {
        let preconditions = preconditions();
        assert!(matches!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 8, 1_600, 1_650),
                &root_snapshot(3),
                preconditions.expected_predecessor_head(),
                challenge(0x55),
                None,
            ),
            Err(SafetyConfigurationObservationCommitError::ClockLineageChanged { .. })
        ));
    }
}
