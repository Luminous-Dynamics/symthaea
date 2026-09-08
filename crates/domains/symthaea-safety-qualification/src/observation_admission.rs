// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed local admission for challenge-bound safety-configuration observations.
//!
//! A raw [`SafetyConfigurationObservation`] is only a claim. This module binds one
//! observation to the exact locally provisioned observer root, exact commissioning
//! challenge, monotone observer/node history, exact verifier-owned qualified
//! configuration, and uncertainty-aware trusted time.
//!
//! Successful admission remains non-cryptographic and non-serializable. A future
//! Xenia adapter must authenticate the exact canonical observation bytes under the
//! captured observer root before the observation may be used as trusted current
//! configuration evidence.

use crate::QualifiedSafetyConfiguration;
use symthaea_safety_configuration::ConfigurationDigest;
use symthaea_safety_configuration::observation::{
    ConfigurationObservationChallenge, ConfigurationObserverRootDigest,
    SafetyConfigurationObservation, SafetyConfigurationObservationDigest,
    SafetyConfigurationObservationError,
};
use symthaea_safety_profile::trusted_time::TrustedAuthorizationClockObservation;
use thiserror::Error;

/// Exact locally provisioned configuration-observer trust root.
///
/// `epoch` changes whenever provisioning continuity is lost or replaced, even if
/// the same verifier key is installed again. A later commit-precondition layer can
/// therefore invalidate an in-flight observation on same-key reprovisioning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigurationObserverRootSnapshot {
    observer_id: String,
    root_digest: ConfigurationObserverRootDigest,
    epoch: u64,
}

impl ConfigurationObserverRootSnapshot {
    pub fn new(
        observer_id: impl Into<String>,
        root_digest: ConfigurationObserverRootDigest,
        epoch: u64,
    ) -> Result<Self, SafetyConfigurationObservationAdmissionError> {
        let observer_id = observer_id.into();
        if observer_id.trim().is_empty() {
            return Err(SafetyConfigurationObservationAdmissionError::EmptyExpectedObserverId);
        }
        if epoch == 0 {
            return Err(SafetyConfigurationObservationAdmissionError::ZeroObserverRootEpoch);
        }
        Ok(Self {
            observer_id,
            root_digest,
            epoch,
        })
    }

    pub fn observer_id(&self) -> &str {
        &self.observer_id
    }

    pub fn root_digest(&self) -> ConfigurationObserverRootDigest {
        self.root_digest
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// Complete identity of one locally admitted configuration observation.
///
/// All fields are derived from one validated observation so a caller cannot pair a
/// generation from one observation with the digest/root/node of another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyConfigurationObservationHeadIdentity {
    generation: u64,
    observation_digest: SafetyConfigurationObservationDigest,
    observer_id: String,
    subject_node_id: String,
    observer_root_digest: ConfigurationObserverRootDigest,
}

impl SafetyConfigurationObservationHeadIdentity {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn observation_digest(&self) -> SafetyConfigurationObservationDigest {
        self.observation_digest
    }

    pub fn observer_id(&self) -> &str {
        &self.observer_id
    }

    pub fn subject_node_id(&self) -> &str {
        &self.subject_node_id
    }

    pub fn observer_root_digest(&self) -> ConfigurationObserverRootDigest {
        self.observer_root_digest
    }
}

/// Exact locally persisted head of one observer/node observation lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafetyConfigurationObservationHead {
    Uninitialized,
    Current(SafetyConfigurationObservationHeadIdentity),
}

impl SafetyConfigurationObservationHead {
    pub fn from_observation(
        observation: &SafetyConfigurationObservation,
    ) -> Result<Self, SafetyConfigurationObservationAdmissionError> {
        observation.validate()?;
        Ok(Self::Current(SafetyConfigurationObservationHeadIdentity {
            generation: observation.observation_generation(),
            observation_digest: observation.observation_digest()?,
            observer_id: observation.observer_id().to_owned(),
            subject_node_id: observation.subject_node_id().to_owned(),
            observer_root_digest: observation.observer_root_digest(),
        }))
    }

    pub fn identity(&self) -> Option<&SafetyConfigurationObservationHeadIdentity> {
        match self {
            Self::Uninitialized => None,
            Self::Current(identity) => Some(identity),
        }
    }
}

/// Trusted local policy inputs for one current-configuration observation attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyConfigurationObservationAdmissionPolicy {
    expected_subject_node_id: String,
    expected_observer_root: ConfigurationObserverRootSnapshot,
    current_head: SafetyConfigurationObservationHead,
    expected_challenge: ConfigurationObservationChallenge,
    max_age_ms: i64,
}

impl SafetyConfigurationObservationAdmissionPolicy {
    pub fn new(
        expected_subject_node_id: impl Into<String>,
        expected_observer_root: ConfigurationObserverRootSnapshot,
        current_head: SafetyConfigurationObservationHead,
        expected_challenge: ConfigurationObservationChallenge,
        max_age_ms: i64,
    ) -> Result<Self, SafetyConfigurationObservationAdmissionError> {
        let expected_subject_node_id = expected_subject_node_id.into();
        if expected_subject_node_id.trim().is_empty() {
            return Err(SafetyConfigurationObservationAdmissionError::EmptyExpectedSubjectNodeId);
        }
        if max_age_ms <= 0 {
            return Err(SafetyConfigurationObservationAdmissionError::InvalidMaximumAge {
                max_age_ms,
            });
        }

        if let Some(identity) = current_head.identity() {
            if identity.subject_node_id() != expected_subject_node_id {
                return Err(
                    SafetyConfigurationObservationAdmissionError::PersistedHeadNodeMismatch {
                        expected: expected_subject_node_id,
                        observed: identity.subject_node_id().to_owned(),
                    },
                );
            }
            if identity.observer_id() != expected_observer_root.observer_id() {
                return Err(
                    SafetyConfigurationObservationAdmissionError::PersistedHeadObserverIdMismatch {
                        expected: expected_observer_root.observer_id().to_owned(),
                        observed: identity.observer_id().to_owned(),
                    },
                );
            }
            if identity.observer_root_digest() != expected_observer_root.root_digest() {
                return Err(
                    SafetyConfigurationObservationAdmissionError::PersistedHeadObserverRootMismatch,
                );
            }
        }

        Ok(Self {
            expected_subject_node_id,
            expected_observer_root,
            current_head,
            expected_challenge,
            max_age_ms,
        })
    }

    pub fn expected_subject_node_id(&self) -> &str {
        &self.expected_subject_node_id
    }

    pub fn expected_observer_root(&self) -> &ConfigurationObserverRootSnapshot {
        &self.expected_observer_root
    }

    pub fn current_head(&self) -> &SafetyConfigurationObservationHead {
        &self.current_head
    }

    pub fn expected_challenge(&self) -> ConfigurationObservationChallenge {
        self.expected_challenge
    }

    pub fn max_age_ms(&self) -> i64 {
        self.max_age_ms
    }

    /// Admit one raw observation against exact observer, node, challenge,
    /// qualification, monotone generation, and trusted-time freshness policy.
    pub fn check(
        &self,
        current_clock: &TrustedAuthorizationClockObservation,
        observation: &SafetyConfigurationObservation,
        qualified: &QualifiedSafetyConfiguration,
    ) -> Result<
        PolicyCheckedCurrentSafetyConfigurationObservation,
        SafetyConfigurationObservationAdmissionError,
    > {
        observation.validate()?;

        if observation.subject_node_id() != self.expected_subject_node_id {
            return Err(SafetyConfigurationObservationAdmissionError::SubjectNodeMismatch {
                expected: self.expected_subject_node_id.clone(),
                observed: observation.subject_node_id().to_owned(),
            });
        }
        if qualified.node_id() != self.expected_subject_node_id {
            return Err(SafetyConfigurationObservationAdmissionError::QualificationNodeMismatch {
                expected: self.expected_subject_node_id.clone(),
                observed: qualified.node_id().to_owned(),
            });
        }
        if observation.observer_id() != self.expected_observer_root.observer_id() {
            return Err(SafetyConfigurationObservationAdmissionError::ObserverIdMismatch {
                expected: self.expected_observer_root.observer_id().to_owned(),
                observed: observation.observer_id().to_owned(),
            });
        }
        if observation.observer_root_digest() != self.expected_observer_root.root_digest() {
            return Err(SafetyConfigurationObservationAdmissionError::ObserverRootDigestMismatch);
        }
        if observation.challenge() != self.expected_challenge {
            return Err(SafetyConfigurationObservationAdmissionError::ChallengeMismatch);
        }
        if observation.configuration_digest() != qualified.configuration_digest() {
            return Err(
                SafetyConfigurationObservationAdmissionError::ConfigurationDigestMismatch {
                    expected: qualified.configuration_digest(),
                    observed: observation.configuration_digest(),
                },
            );
        }

        check_observation_lineage(&self.current_head, observation)?;
        validate_freshness(
            observation.observed_at_unix_ms(),
            self.max_age_ms,
            current_clock,
        )?;

        let candidate_head = SafetyConfigurationObservationHead::from_observation(observation)?;

        Ok(PolicyCheckedCurrentSafetyConfigurationObservation {
            observation: observation.clone(),
            canonical_observation_bytes: observation.canonical_signing_bytes()?,
            observation_digest: observation.observation_digest()?,
            expected_observer_root: self.expected_observer_root.clone(),
            expected_predecessor_head: self.current_head.clone(),
            candidate_head,
            expected_challenge: self.expected_challenge,
            expected_clock_source_id: current_clock.source_id().to_owned(),
            expected_clock_epoch: current_clock.epoch(),
            max_age_ms: self.max_age_ms,
            configuration_digest: qualified.configuration_digest(),
        })
    }
}

fn check_observation_lineage(
    current_head: &SafetyConfigurationObservationHead,
    observation: &SafetyConfigurationObservation,
) -> Result<(), SafetyConfigurationObservationAdmissionError> {
    match current_head {
        SafetyConfigurationObservationHead::Uninitialized => {
            if observation.observation_generation() != 1 {
                return Err(
                    SafetyConfigurationObservationAdmissionError::ExpectedBootstrapGeneration {
                        observed: observation.observation_generation(),
                    },
                );
            }
        }
        SafetyConfigurationObservationHead::Current(identity) => {
            let expected = identity.generation().checked_add(1).ok_or(
                SafetyConfigurationObservationAdmissionError::GenerationExhausted {
                    current: identity.generation(),
                },
            )?;
            if observation.observation_generation() != expected {
                return Err(
                    SafetyConfigurationObservationAdmissionError::GenerationNotSuccessor {
                        current: identity.generation(),
                        expected,
                        observed: observation.observation_generation(),
                    },
                );
            }
        }
    }
    Ok(())
}

/// Require the observation to be unambiguously in the past and no older than the
/// configured maximum age for every possible current time in the trusted interval.
fn validate_freshness(
    observed_at_unix_ms: i64,
    max_age_ms: i64,
    clock: &TrustedAuthorizationClockObservation,
) -> Result<(), SafetyConfigurationObservationAdmissionError> {
    if observed_at_unix_ms > clock.latest_unix_ms() {
        return Err(SafetyConfigurationObservationAdmissionError::DefinitelyFromFuture {
            observed_at_unix_ms,
            latest_unix_ms: clock.latest_unix_ms(),
        });
    }
    if observed_at_unix_ms > clock.earliest_unix_ms() {
        return Err(
            SafetyConfigurationObservationAdmissionError::ClockUncertaintyCrossesObservationTime {
                observed_at_unix_ms,
                earliest_unix_ms: clock.earliest_unix_ms(),
                latest_unix_ms: clock.latest_unix_ms(),
            },
        );
    }

    // i128 avoids overflow even for extreme signed Unix-millisecond inputs.
    let worst_case_age_ms =
        i128::from(clock.latest_unix_ms()) - i128::from(observed_at_unix_ms);
    if worst_case_age_ms > i128::from(max_age_ms) {
        return Err(SafetyConfigurationObservationAdmissionError::ObservationTooOld {
            observed_at_unix_ms,
            latest_unix_ms: clock.latest_unix_ms(),
            max_age_ms,
        });
    }
    Ok(())
}

/// Non-serializable result of exact local observation policy checks.
///
/// This value proves no signature. A future Xenia adapter must authenticate
/// [`Self::canonical_observation_bytes`] under [`Self::expected_observer_root`]
/// before it may stand in for the current installed configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyCheckedCurrentSafetyConfigurationObservation {
    observation: SafetyConfigurationObservation,
    canonical_observation_bytes: Vec<u8>,
    observation_digest: SafetyConfigurationObservationDigest,
    expected_observer_root: ConfigurationObserverRootSnapshot,
    expected_predecessor_head: SafetyConfigurationObservationHead,
    candidate_head: SafetyConfigurationObservationHead,
    expected_challenge: ConfigurationObservationChallenge,
    expected_clock_source_id: String,
    expected_clock_epoch: u64,
    max_age_ms: i64,
    configuration_digest: ConfigurationDigest,
}

impl PolicyCheckedCurrentSafetyConfigurationObservation {
    pub fn observation(&self) -> &SafetyConfigurationObservation {
        &self.observation
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

    pub fn expected_clock_source_id(&self) -> &str {
        &self.expected_clock_source_id
    }

    pub fn expected_clock_epoch(&self) -> u64 {
        self.expected_clock_epoch
    }

    pub fn max_age_ms(&self) -> i64 {
        self.max_age_ms
    }

    pub fn configuration_digest(&self) -> ConfigurationDigest {
        self.configuration_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyConfigurationObservationAdmissionError {
    #[error(transparent)]
    Observation(#[from] SafetyConfigurationObservationError),
    #[error("expected safety-configuration observation node id must not be empty")]
    EmptyExpectedSubjectNodeId,
    #[error("expected configuration observer id must not be empty")]
    EmptyExpectedObserverId,
    #[error("configuration-observer root epoch must be greater than zero")]
    ZeroObserverRootEpoch,
    #[error("configuration observation maximum age must be positive, observed {max_age_ms}")]
    InvalidMaximumAge { max_age_ms: i64 },
    #[error("persisted observation head belongs to node {observed}, expected {expected}")]
    PersistedHeadNodeMismatch { expected: String, observed: String },
    #[error("persisted observation head observer id mismatch: expected {expected}, observed {observed}")]
    PersistedHeadObserverIdMismatch { expected: String, observed: String },
    #[error("persisted observation head belongs to a different observer root")]
    PersistedHeadObserverRootMismatch,
    #[error("configuration observation node mismatch: expected {expected}, observed {observed}")]
    SubjectNodeMismatch { expected: String, observed: String },
    #[error("qualified configuration node mismatch: expected {expected}, observed {observed}")]
    QualificationNodeMismatch { expected: String, observed: String },
    #[error("configuration observer id mismatch: expected {expected}, observed {observed}")]
    ObserverIdMismatch { expected: String, observed: String },
    #[error("configuration observation root digest does not match provisioned observer root")]
    ObserverRootDigestMismatch,
    #[error("configuration observation challenge does not match this transaction")]
    ChallengeMismatch,
    #[error("observed configuration digest does not match the qualified configuration")]
    ConfigurationDigestMismatch {
        expected: ConfigurationDigest,
        observed: ConfigurationDigest,
    },
    #[error("first admitted configuration observation must use generation 1, observed {observed}")]
    ExpectedBootstrapGeneration { observed: u64 },
    #[error("configuration observation generation space exhausted at {current}")]
    GenerationExhausted { current: u64 },
    #[error("configuration observation must be exact successor of generation {current}: expected {expected}, observed {observed}")]
    GenerationNotSuccessor {
        current: u64,
        expected: u64,
        observed: u64,
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
    use symthaea_safety_configuration::{
        ConfigurationComponent, SafetyConfigurationManifest, SAFETY_CONFIGURATION_SCHEMA_V1,
    };
    use symthaea_safety_profile::{
        ComponentRequirement, SafetyConfigurationProfile,
        SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
    };
    use crate::qualify_safety_configuration;

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

    fn qualified() -> QualifiedSafetyConfiguration {
        let profile = profile();
        qualify_safety_configuration(&profile, manifest(&profile)).unwrap()
    }

    fn root_snapshot(epoch: u64) -> ConfigurationObserverRootSnapshot {
        ConfigurationObserverRootSnapshot::new("observer-1", observer_root(0x44), epoch).unwrap()
    }

    fn observation(
        generation: u64,
        observed_at: i64,
        challenge_byte: u8,
        config: ConfigurationDigest,
    ) -> SafetyConfigurationObservation {
        SafetyConfigurationObservation::new(
            format!("obs-{generation}"),
            "observer-1",
            "rack",
            generation,
            observed_at,
            observer_root(0x44),
            challenge(challenge_byte),
            config,
        )
        .unwrap()
    }

    fn clock(earliest: i64, latest: i64) -> TrustedAuthorizationClockObservation {
        TrustedAuthorizationClockObservation::new("secure-rtc-v1", 7, earliest, latest).unwrap()
    }

    #[test]
    fn exact_fresh_challenge_bound_observation_is_admitted() {
        let qualified = qualified();
        let observation = observation(1, 1_500, 0x55, qualified.configuration_digest());
        let policy = SafetyConfigurationObservationAdmissionPolicy::new(
            "rack",
            root_snapshot(3),
            SafetyConfigurationObservationHead::Uninitialized,
            challenge(0x55),
            500,
        )
        .unwrap();

        let checked = policy
            .check(&clock(1_550, 1_600), &observation, &qualified)
            .unwrap();
        assert_eq!(checked.configuration_digest(), qualified.configuration_digest());
        assert_eq!(checked.expected_challenge(), challenge(0x55));
        assert_eq!(checked.expected_clock_source_id(), "secure-rtc-v1");
        assert_eq!(checked.expected_clock_epoch(), 7);
        assert!(checked.candidate_head().identity().is_some());
    }

    #[test]
    fn challenge_replay_into_different_transaction_fails() {
        let qualified = qualified();
        let observation = observation(1, 1_500, 0x54, qualified.configuration_digest());
        let policy = SafetyConfigurationObservationAdmissionPolicy::new(
            "rack",
            root_snapshot(3),
            SafetyConfigurationObservationHead::Uninitialized,
            challenge(0x55),
            500,
        )
        .unwrap();
        assert_eq!(
            policy.check(&clock(1_550, 1_600), &observation, &qualified),
            Err(SafetyConfigurationObservationAdmissionError::ChallengeMismatch)
        );
    }

    #[test]
    fn unqualified_configuration_digest_cannot_be_substituted() {
        let qualified = qualified();
        let observation = observation(1, 1_500, 0x55, digest(0xee));
        let policy = SafetyConfigurationObservationAdmissionPolicy::new(
            "rack",
            root_snapshot(3),
            SafetyConfigurationObservationHead::Uninitialized,
            challenge(0x55),
            500,
        )
        .unwrap();
        assert!(matches!(
            policy.check(&clock(1_550, 1_600), &observation, &qualified),
            Err(SafetyConfigurationObservationAdmissionError::ConfigurationDigestMismatch { .. })
        ));
    }

    #[test]
    fn clock_uncertainty_must_prove_observation_is_not_future() {
        let qualified = qualified();
        let observation = observation(1, 1_575, 0x55, qualified.configuration_digest());
        let policy = SafetyConfigurationObservationAdmissionPolicy::new(
            "rack",
            root_snapshot(3),
            SafetyConfigurationObservationHead::Uninitialized,
            challenge(0x55),
            500,
        )
        .unwrap();
        assert!(matches!(
            policy.check(&clock(1_550, 1_600), &observation, &qualified),
            Err(SafetyConfigurationObservationAdmissionError::ClockUncertaintyCrossesObservationTime { .. })
        ));
    }

    #[test]
    fn worst_case_clock_age_must_fit_freshness_window() {
        let qualified = qualified();
        let observation = observation(1, 1_000, 0x55, qualified.configuration_digest());
        let policy = SafetyConfigurationObservationAdmissionPolicy::new(
            "rack",
            root_snapshot(3),
            SafetyConfigurationObservationHead::Uninitialized,
            challenge(0x55),
            500,
        )
        .unwrap();
        assert!(matches!(
            policy.check(&clock(1_450, 1_501), &observation, &qualified),
            Err(SafetyConfigurationObservationAdmissionError::ObservationTooOld { .. })
        ));
    }

    #[test]
    fn exact_successor_generation_is_required() {
        let qualified = qualified();
        let previous = observation(1, 1_300, 0x40, qualified.configuration_digest());
        let previous_head = SafetyConfigurationObservationHead::from_observation(&previous).unwrap();
        let candidate = observation(3, 1_500, 0x55, qualified.configuration_digest());
        let policy = SafetyConfigurationObservationAdmissionPolicy::new(
            "rack",
            root_snapshot(3),
            previous_head,
            challenge(0x55),
            500,
        )
        .unwrap();
        assert!(matches!(
            policy.check(&clock(1_550, 1_600), &candidate, &qualified),
            Err(SafetyConfigurationObservationAdmissionError::GenerationNotSuccessor {
                current: 1,
                expected: 2,
                observed: 3,
            })
        ));
    }

    #[test]
    fn persisted_head_must_belong_to_current_observer_root() {
        let qualified = qualified();
        let previous = observation(1, 1_300, 0x40, qualified.configuration_digest());
        let previous_head = SafetyConfigurationObservationHead::from_observation(&previous).unwrap();
        assert_eq!(
            SafetyConfigurationObservationAdmissionPolicy::new(
                "rack",
                ConfigurationObserverRootSnapshot::new(
                    "observer-1",
                    observer_root(0x45),
                    3,
                )
                .unwrap(),
                previous_head,
                challenge(0x55),
                500,
            ),
            Err(SafetyConfigurationObservationAdmissionError::PersistedHeadObserverRootMismatch)
        );
    }
}
