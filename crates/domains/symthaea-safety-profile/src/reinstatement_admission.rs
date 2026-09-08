// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed local admission for revocation-bound profile reinstatement.
//!
//! A valid reinstatement artifact is still only raw evidence. Admission requires
//! the exact locally persisted lifecycle state to be the revoked predecessor named
//! by the artifact, under the exact provisioned profile-authority root, exact
//! profile artifact, and one uncertainty-aware trusted-clock observation.
//!
//! Success is deliberately non-cryptographic and non-serializable. A future Xenia
//! adapter must authenticate the exact reinstatement canonical bytes before any
//! lifecycle state can be committed from Revoked(N, R) to Active(N+1).

use crate::admission::SafetyProfileAuthorizationHead;
use crate::commit_preconditions::ProfileAuthorityRootSnapshot;
use crate::lifecycle::{
    SafetyProfileAuthorizationLifecycleError, SafetyProfileAuthorizationLifecycleState,
};
use crate::reinstatement::{
    SafetyProfileAuthorizationReinstatementError,
    SafetyProfileAuthorizationReinstatementTransition,
};
use crate::revocation::{
    SafetyProfileAuthorizationRevocationDigest, SafetyProfileAuthorizationRevocationError,
};
use crate::transition::SafetyProfileAuthorizationTransitionError;
use crate::trusted_time::TrustedAuthorizationClockObservation;
use crate::{SafetyConfigurationProfile, SafetyConfigurationProfileError};
use symthaea_safety_configuration::ConfigurationDigest;
use thiserror::Error;

/// Trusted local policy inputs for one reinstatement admission attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyProfileAuthorizationReinstatementAdmissionPolicy {
    expected_subject_node_id: String,
    expected_root: ProfileAuthorityRootSnapshot,
    current_lifecycle: SafetyProfileAuthorizationLifecycleState,
}

impl SafetyProfileAuthorizationReinstatementAdmissionPolicy {
    pub fn new(
        expected_subject_node_id: impl Into<String>,
        expected_root: ProfileAuthorityRootSnapshot,
        current_lifecycle: SafetyProfileAuthorizationLifecycleState,
    ) -> Result<Self, SafetyProfileAuthorizationReinstatementAdmissionError> {
        let expected_subject_node_id = expected_subject_node_id.into();
        if expected_subject_node_id.trim().is_empty() {
            return Err(
                SafetyProfileAuthorizationReinstatementAdmissionError::EmptyExpectedSubjectNodeId,
            );
        }
        if !current_lifecycle.is_revoked() {
            return Err(
                SafetyProfileAuthorizationReinstatementAdmissionError::CurrentLifecycleNotRevoked,
            );
        }

        let identity = current_lifecycle.authorization_head().identity().ok_or(
            SafetyProfileAuthorizationReinstatementAdmissionError::CurrentHeadNotInitialized,
        )?;
        if identity.subject_node_id() != expected_subject_node_id {
            return Err(
                SafetyProfileAuthorizationReinstatementAdmissionError::PersistedHeadNodeMismatch {
                    expected: expected_subject_node_id,
                    observed: identity.subject_node_id().to_owned(),
                },
            );
        }
        if identity.authority_root_id() != expected_root.root_id() {
            return Err(
                SafetyProfileAuthorizationReinstatementAdmissionError::PersistedHeadAuthorityRootIdMismatch {
                    expected: expected_root.root_id().to_owned(),
                    observed: identity.authority_root_id().to_owned(),
                },
            );
        }
        if identity.authority_root_digest() != expected_root.root_digest() {
            return Err(
                SafetyProfileAuthorizationReinstatementAdmissionError::PersistedHeadAuthorityRootDigestMismatch,
            );
        }

        Ok(Self {
            expected_subject_node_id,
            expected_root,
            current_lifecycle,
        })
    }

    pub fn expected_subject_node_id(&self) -> &str {
        &self.expected_subject_node_id
    }

    pub fn expected_root(&self) -> &ProfileAuthorityRootSnapshot {
        &self.expected_root
    }

    pub fn current_lifecycle(&self) -> &SafetyProfileAuthorizationLifecycleState {
        &self.current_lifecycle
    }

    /// Check one reinstatement against exact revoked lifecycle state, root, profile,
    /// and uncertainty-aware trusted time.
    pub fn check(
        &self,
        current_clock: &TrustedAuthorizationClockObservation,
        reinstatement: &SafetyProfileAuthorizationReinstatementTransition,
        profile: &SafetyConfigurationProfile,
    ) -> Result<
        PolicyCheckedSafetyProfileAuthorizationReinstatement,
        SafetyProfileAuthorizationReinstatementAdmissionError,
    > {
        reinstatement.validate()?;
        profile.validate()?;

        let successor_subject = reinstatement.successor().subject();
        if successor_subject.subject_node_id() != self.expected_subject_node_id {
            return Err(
                SafetyProfileAuthorizationReinstatementAdmissionError::SubjectNodeMismatch {
                    expected: self.expected_subject_node_id.clone(),
                    observed: successor_subject.subject_node_id().to_owned(),
                },
            );
        }
        if successor_subject.authority_root_id() != self.expected_root.root_id() {
            return Err(
                SafetyProfileAuthorizationReinstatementAdmissionError::AuthorityRootIdMismatch {
                    expected: self.expected_root.root_id().to_owned(),
                    observed: successor_subject.authority_root_id().to_owned(),
                },
            );
        }
        if successor_subject.authority_root_digest() != self.expected_root.root_digest() {
            return Err(
                SafetyProfileAuthorizationReinstatementAdmissionError::AuthorityRootDigestMismatch,
            );
        }

        let expected_revocation_digest = self.current_lifecycle.revocation_digest().ok_or(
            SafetyProfileAuthorizationReinstatementAdmissionError::CurrentLifecycleNotRevoked,
        )?;
        let observed_revocation_digest = reinstatement.revocation_digest()?;
        if observed_revocation_digest != expected_revocation_digest {
            return Err(
                SafetyProfileAuthorizationReinstatementAdmissionError::RevocationDigestMismatch {
                    expected: expected_revocation_digest,
                    observed: observed_revocation_digest,
                },
            );
        }
        if !reinstatement
            .revocation()
            .targets_exact_head(self.current_lifecycle.authorization_head())?
        {
            return Err(
                SafetyProfileAuthorizationReinstatementAdmissionError::RevocationTargetHeadMismatch,
            );
        }

        if successor_subject.profile_id() != profile.profile_id {
            return Err(
                SafetyProfileAuthorizationReinstatementAdmissionError::ProfileIdMismatch {
                    expected: successor_subject.profile_id().to_owned(),
                    observed: profile.profile_id.clone(),
                },
            );
        }
        let profile_digest = profile.digest()?;
        if successor_subject.profile_digest() != profile_digest {
            return Err(
                SafetyProfileAuthorizationReinstatementAdmissionError::ProfileDigestMismatch {
                    expected: successor_subject.profile_digest(),
                    observed: profile_digest,
                },
            );
        }

        validate_entire_successor_interval(
            successor_subject.valid_from_unix_ms(),
            successor_subject.valid_until_unix_ms(),
            current_clock,
        )?;

        let candidate_head =
            SafetyProfileAuthorizationHead::from_transition(reinstatement.successor())?;
        let candidate_lifecycle =
            SafetyProfileAuthorizationLifecycleState::active(candidate_head)?;

        Ok(PolicyCheckedSafetyProfileAuthorizationReinstatement {
            reinstatement: reinstatement.clone(),
            profile: profile.clone(),
            canonical_reinstatement_bytes: reinstatement.canonical_signing_bytes()?,
            expected_root: self.expected_root.clone(),
            expected_lifecycle: self.current_lifecycle.clone(),
            candidate_lifecycle,
            expected_clock_source_id: current_clock.source_id().to_owned(),
            expected_clock_epoch: current_clock.epoch(),
        })
    }
}

/// Non-serializable result of exact local reinstatement policy checks.
///
/// This value proves no signature. Its canonical bytes must still be authenticated
/// under the exact captured root before the candidate lifecycle may be persisted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyCheckedSafetyProfileAuthorizationReinstatement {
    reinstatement: SafetyProfileAuthorizationReinstatementTransition,
    profile: SafetyConfigurationProfile,
    canonical_reinstatement_bytes: Vec<u8>,
    expected_root: ProfileAuthorityRootSnapshot,
    expected_lifecycle: SafetyProfileAuthorizationLifecycleState,
    candidate_lifecycle: SafetyProfileAuthorizationLifecycleState,
    expected_clock_source_id: String,
    expected_clock_epoch: u64,
}

impl PolicyCheckedSafetyProfileAuthorizationReinstatement {
    pub fn reinstatement(&self) -> &SafetyProfileAuthorizationReinstatementTransition {
        &self.reinstatement
    }

    pub fn profile(&self) -> &SafetyConfigurationProfile {
        &self.profile
    }

    pub fn canonical_reinstatement_bytes(&self) -> &[u8] {
        &self.canonical_reinstatement_bytes
    }

    pub fn expected_root(&self) -> &ProfileAuthorityRootSnapshot {
        &self.expected_root
    }

    pub fn expected_lifecycle(&self) -> &SafetyProfileAuthorizationLifecycleState {
        &self.expected_lifecycle
    }

    pub fn candidate_lifecycle(&self) -> &SafetyProfileAuthorizationLifecycleState {
        &self.candidate_lifecycle
    }

    pub fn expected_clock_source_id(&self) -> &str {
        &self.expected_clock_source_id
    }

    pub fn expected_clock_epoch(&self) -> u64 {
        self.expected_clock_epoch
    }
}

fn validate_entire_successor_interval(
    valid_from_unix_ms: i64,
    valid_until_unix_ms: i64,
    clock: &TrustedAuthorizationClockObservation,
) -> Result<(), SafetyProfileAuthorizationReinstatementAdmissionError> {
    if clock.latest_unix_ms() < valid_from_unix_ms {
        return Err(
            SafetyProfileAuthorizationReinstatementAdmissionError::DefinitelyNotYetValid {
                latest_unix_ms: clock.latest_unix_ms(),
                valid_from_unix_ms,
            },
        );
    }
    if clock.earliest_unix_ms() >= valid_until_unix_ms {
        return Err(
            SafetyProfileAuthorizationReinstatementAdmissionError::DefinitelyExpired {
                earliest_unix_ms: clock.earliest_unix_ms(),
                valid_until_unix_ms,
            },
        );
    }
    if clock.earliest_unix_ms() < valid_from_unix_ms
        || clock.latest_unix_ms() >= valid_until_unix_ms
    {
        return Err(
            SafetyProfileAuthorizationReinstatementAdmissionError::UncertaintyCrossesValidityBoundary {
                earliest_unix_ms: clock.earliest_unix_ms(),
                latest_unix_ms: clock.latest_unix_ms(),
                valid_from_unix_ms,
                valid_until_unix_ms,
            },
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyProfileAuthorizationReinstatementAdmissionError {
    #[error(transparent)]
    Reinstatement(#[from] SafetyProfileAuthorizationReinstatementError),
    #[error(transparent)]
    Transition(#[from] SafetyProfileAuthorizationTransitionError),
    #[error(transparent)]
    Revocation(#[from] SafetyProfileAuthorizationRevocationError),
    #[error(transparent)]
    Profile(#[from] SafetyConfigurationProfileError),
    #[error(transparent)]
    Lifecycle(#[from] SafetyProfileAuthorizationLifecycleError),
    #[error("expected reinstatement subject node id must not be empty")]
    EmptyExpectedSubjectNodeId,
    #[error("profile reinstatement requires the current lifecycle to be revoked")]
    CurrentLifecycleNotRevoked,
    #[error("revoked lifecycle unexpectedly has no current authorization head")]
    CurrentHeadNotInitialized,
    #[error("persisted authorization head belongs to node {observed}, expected {expected}")]
    PersistedHeadNodeMismatch { expected: String, observed: String },
    #[error("persisted authorization head root id mismatch: expected {expected}, observed {observed}")]
    PersistedHeadAuthorityRootIdMismatch { expected: String, observed: String },
    #[error("persisted authorization head root digest does not match provisioned root")]
    PersistedHeadAuthorityRootDigestMismatch,
    #[error("reinstatement successor node mismatch: expected {expected}, observed {observed}")]
    SubjectNodeMismatch { expected: String, observed: String },
    #[error("reinstatement authority-root id mismatch: expected {expected}, observed {observed}")]
    AuthorityRootIdMismatch { expected: String, observed: String },
    #[error("reinstatement authority-root digest does not match provisioned root")]
    AuthorityRootDigestMismatch,
    #[error("reinstatement revocation digest does not match exact persisted revoked lifecycle")]
    RevocationDigestMismatch {
        expected: SafetyProfileAuthorizationRevocationDigest,
        observed: SafetyProfileAuthorizationRevocationDigest,
    },
    #[error("reinstatement revocation does not target exact persisted authorization head")]
    RevocationTargetHeadMismatch,
    #[error("reinstatement successor profile id mismatch: expected {expected}, observed {observed}")]
    ProfileIdMismatch { expected: String, observed: String },
    #[error("reinstatement successor profile digest mismatch: expected {expected:?}, observed {observed:?}")]
    ProfileDigestMismatch {
        expected: ConfigurationDigest,
        observed: ConfigurationDigest,
    },
    #[error("reinstatement successor is definitely not yet valid: latest possible time {latest_unix_ms} < valid-from {valid_from_unix_ms}")]
    DefinitelyNotYetValid {
        latest_unix_ms: i64,
        valid_from_unix_ms: i64,
    },
    #[error("reinstatement successor is definitely expired: earliest possible time {earliest_unix_ms} >= valid-until {valid_until_unix_ms}")]
    DefinitelyExpired {
        earliest_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
    #[error("trusted clock uncertainty [{earliest_unix_ms}, {latest_unix_ms}] crosses reinstatement successor validity [{valid_from_unix_ms}, {valid_until_unix_ms})")]
    UncertaintyCrossesValidityBoundary {
        earliest_unix_ms: i64,
        latest_unix_ms: i64,
        valid_from_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::SafetyProfileAuthorizationHead;
    use crate::authorization::{ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject};
    use crate::reinstatement::SafetyProfileAuthorizationReinstatementTransition;
    use crate::revocation::SafetyProfileAuthorizationRevocationSubject;
    use crate::transition::SafetyProfileAuthorizationTransition;
    use crate::{
        ComponentRequirement, SafetyConfigurationProfile,
        SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
    };

    fn root(byte: u8) -> ProfileAuthorityRootDigest {
        ProfileAuthorityRootDigest::Blake3_256([byte; 32])
    }

    fn profile(id: &str) -> SafetyConfigurationProfile {
        SafetyConfigurationProfile {
            schema_version: SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1.to_owned(),
            profile_id: id.to_owned(),
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

    fn subject(
        id: &str,
        generation: u64,
        valid_from: i64,
        valid_until: i64,
        profile: &SafetyConfigurationProfile,
    ) -> SafetyProfileAuthorizationSubject {
        SafetyProfileAuthorizationSubject::new(
            id,
            "facility-profile-root-v1",
            root(0x33),
            "compute-campus",
            generation,
            valid_from,
            valid_until,
            profile,
        )
        .unwrap()
    }

    fn bootstrap(
        id: &str,
        profile: &SafetyConfigurationProfile,
    ) -> SafetyProfileAuthorizationTransition {
        SafetyProfileAuthorizationTransition::bootstrap(subject(
            id, 1, 1_000, 4_000, profile,
        ))
        .unwrap()
    }

    fn successor(
        predecessor: &SafetyProfileAuthorizationTransition,
        profile: &SafetyConfigurationProfile,
    ) -> SafetyProfileAuthorizationTransition {
        SafetyProfileAuthorizationTransition::successor(
            subject("auth-2", 2, 1_500, 4_000, profile),
            predecessor,
        )
        .unwrap()
    }

    fn root_snapshot() -> ProfileAuthorityRootSnapshot {
        ProfileAuthorityRootSnapshot::new("facility-profile-root-v1", root(0x33), 4).unwrap()
    }

    fn clock(earliest: i64, latest: i64) -> TrustedAuthorizationClockObservation {
        TrustedAuthorizationClockObservation::new("secure-rtc-v1", 7, earliest, latest).unwrap()
    }

    fn fixture() -> (
        SafetyProfileAuthorizationTransition,
        SafetyProfileAuthorizationRevocationSubject,
        SafetyProfileAuthorizationReinstatementTransition,
        SafetyConfigurationProfile,
    ) {
        let old_profile = profile("profile-v1");
        let new_profile = profile("profile-v2");
        let first = bootstrap("auth-1", &old_profile);
        let revocation = SafetyProfileAuthorizationRevocationSubject::for_transition(
            "revoke-1",
            &first,
            1_500,
        )
        .unwrap();
        let second = successor(&first, &new_profile);
        let reinstatement = SafetyProfileAuthorizationReinstatementTransition::new(
            "reinstate-1",
            &first,
            &revocation,
            &second,
        )
        .unwrap();
        (first, revocation, reinstatement, new_profile)
    }

    fn revoked_state(
        head_transition: &SafetyProfileAuthorizationTransition,
        digest: SafetyProfileAuthorizationRevocationDigest,
    ) -> SafetyProfileAuthorizationLifecycleState {
        SafetyProfileAuthorizationLifecycleState::revoked(
            SafetyProfileAuthorizationHead::from_transition(head_transition).unwrap(),
            digest,
        )
        .unwrap()
    }

    #[test]
    fn exact_revoked_lifecycle_root_profile_and_time_are_admitted() {
        let (first, revocation, reinstatement, new_profile) = fixture();
        let lifecycle = revoked_state(&first, revocation.revocation_digest().unwrap());
        let policy = SafetyProfileAuthorizationReinstatementAdmissionPolicy::new(
            "compute-campus",
            root_snapshot(),
            lifecycle.clone(),
        )
        .unwrap();

        let checked = policy
            .check(&clock(1_600, 1_700), &reinstatement, &new_profile)
            .unwrap();

        assert_eq!(checked.expected_lifecycle(), &lifecycle);
        assert!(checked.candidate_lifecycle().is_active());
        assert_eq!(
            checked.candidate_lifecycle().authorization_head(),
            &SafetyProfileAuthorizationHead::from_transition(reinstatement.successor()).unwrap()
        );
        assert_eq!(
            checked.canonical_reinstatement_bytes(),
            reinstatement.canonical_signing_bytes().unwrap()
        );
    }

    #[test]
    fn active_lifecycle_cannot_enter_reinstatement_policy() {
        let (first, _, _, _) = fixture();
        let active = SafetyProfileAuthorizationLifecycleState::active(
            SafetyProfileAuthorizationHead::from_transition(&first).unwrap(),
        )
        .unwrap();
        assert_eq!(
            SafetyProfileAuthorizationReinstatementAdmissionPolicy::new(
                "compute-campus",
                root_snapshot(),
                active,
            ),
            Err(SafetyProfileAuthorizationReinstatementAdmissionError::CurrentLifecycleNotRevoked)
        );
    }

    #[test]
    fn wrong_revocation_digest_fails_closed() {
        let (first, _, reinstatement, new_profile) = fixture();
        let lifecycle = revoked_state(
            &first,
            SafetyProfileAuthorizationRevocationDigest::Blake3_256([0xee; 32]),
        );
        let policy = SafetyProfileAuthorizationReinstatementAdmissionPolicy::new(
            "compute-campus",
            root_snapshot(),
            lifecycle,
        )
        .unwrap();

        assert!(matches!(
            policy.check(&clock(1_600, 1_700), &reinstatement, &new_profile),
            Err(SafetyProfileAuthorizationReinstatementAdmissionError::RevocationDigestMismatch { .. })
        ));
    }

    #[test]
    fn same_generation_fork_lifecycle_cannot_accept_other_branch_reinstatement() {
        let (_, revocation, reinstatement, new_profile) = fixture();
        let fork_profile = profile("profile-v1");
        let fork = bootstrap("auth-fork", &fork_profile);
        let lifecycle = revoked_state(&fork, revocation.revocation_digest().unwrap());
        let policy = SafetyProfileAuthorizationReinstatementAdmissionPolicy::new(
            "compute-campus",
            root_snapshot(),
            lifecycle,
        )
        .unwrap();

        assert_eq!(
            policy.check(&clock(1_600, 1_700), &reinstatement, &new_profile),
            Err(SafetyProfileAuthorizationReinstatementAdmissionError::RevocationTargetHeadMismatch)
        );
    }

    #[test]
    fn wrong_profile_artifact_fails_closed() {
        let (first, revocation, reinstatement, _) = fixture();
        let lifecycle = revoked_state(&first, revocation.revocation_digest().unwrap());
        let policy = SafetyProfileAuthorizationReinstatementAdmissionPolicy::new(
            "compute-campus",
            root_snapshot(),
            lifecycle,
        )
        .unwrap();
        let wrong_profile = profile("wrong-profile");

        assert!(matches!(
            policy.check(&clock(1_600, 1_700), &reinstatement, &wrong_profile),
            Err(SafetyProfileAuthorizationReinstatementAdmissionError::ProfileIdMismatch { .. })
        ));
    }

    #[test]
    fn uncertainty_crossing_successor_validity_fails_closed() {
        let (first, revocation, reinstatement, new_profile) = fixture();
        let lifecycle = revoked_state(&first, revocation.revocation_digest().unwrap());
        let policy = SafetyProfileAuthorizationReinstatementAdmissionPolicy::new(
            "compute-campus",
            root_snapshot(),
            lifecycle,
        )
        .unwrap();

        assert!(matches!(
            policy.check(&clock(1_400, 1_600), &reinstatement, &new_profile),
            Err(SafetyProfileAuthorizationReinstatementAdmissionError::UncertaintyCrossesValidityBoundary { .. })
        ));
    }

    #[test]
    fn root_alias_with_same_digest_is_rejected_before_check() {
        let (first, revocation, _, _) = fixture();
        let lifecycle = revoked_state(&first, revocation.revocation_digest().unwrap());
        let aliased_root = ProfileAuthorityRootSnapshot::new(
            "alias-profile-root",
            root(0x33),
            4,
        )
        .unwrap();

        assert!(matches!(
            SafetyProfileAuthorizationReinstatementAdmissionPolicy::new(
                "compute-campus",
                aliased_root,
                lifecycle,
            ),
            Err(SafetyProfileAuthorizationReinstatementAdmissionError::PersistedHeadAuthorityRootIdMismatch { .. })
        ));
    }
}
