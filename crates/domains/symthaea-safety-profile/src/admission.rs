// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed local admission for lineage-bearing safety-profile authorizations.
//!
//! Cryptographic verification and policy admission are intentionally distinct.
//! A valid signature proves who authenticated a transition; this module proves
//! whether that transition is admissible for the exact local node, trust root,
//! current time, profile artifact, and persisted authorization head.

use crate::authorization::ProfileAuthorityRootDigest;
use crate::transition::{
    SafetyProfileAuthorizationPredecessor, SafetyProfileAuthorizationTransition,
    SafetyProfileAuthorizationTransitionDigest, SafetyProfileAuthorizationTransitionError,
};
use crate::{SafetyConfigurationProfile, SafetyConfigurationProfileError};
use symthaea_safety_configuration::ConfigurationDigest;
use thiserror::Error;

/// Exact identity of one admitted authorization head.
///
/// All fields are private and the only production constructor derives them from
/// one validated [`SafetyProfileAuthorizationTransition`]. Callers therefore
/// cannot pair the generation from one transition with the digest or authority
/// identity from another and present that inconsistent tuple as local state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyProfileAuthorizationHeadIdentity {
    generation: u64,
    transition_digest: SafetyProfileAuthorizationTransitionDigest,
    subject_node_id: String,
    authority_root_id: String,
    authority_root_digest: ProfileAuthorityRootDigest,
}

impl SafetyProfileAuthorizationHeadIdentity {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn transition_digest(&self) -> SafetyProfileAuthorizationTransitionDigest {
        self.transition_digest
    }

    pub fn subject_node_id(&self) -> &str {
        &self.subject_node_id
    }

    pub fn authority_root_id(&self) -> &str {
        &self.authority_root_id
    }

    pub fn authority_root_digest(&self) -> ProfileAuthorityRootDigest {
        self.authority_root_digest
    }
}

/// Exact persisted authorization head observed before evaluating a candidate.
///
/// A generation number alone is insufficient because two conflicting transitions
/// can share the same generation. A current head therefore carries one derived,
/// internally consistent transition identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafetyProfileAuthorizationHead {
    /// Trusted authority root is provisioned, but no profile authorization has
    /// been admitted yet. Only a generation-1 Bootstrap transition can enter.
    Uninitialized,
    /// Exact currently admitted transition identity.
    Current(SafetyProfileAuthorizationHeadIdentity),
}

impl SafetyProfileAuthorizationHead {
    /// Derive a current-head identity from one exact validated transition.
    ///
    /// This is intentionally the only production path to `Current`: runtime code
    /// must not synthesize generation/digest/root metadata independently.
    pub fn from_transition(
        transition: &SafetyProfileAuthorizationTransition,
    ) -> Result<Self, SafetyProfileAuthorizationTransitionError> {
        transition.validate()?;
        let subject = transition.subject();
        Ok(Self::Current(SafetyProfileAuthorizationHeadIdentity {
            generation: transition.generation(),
            transition_digest: transition.transition_digest()?,
            subject_node_id: subject.subject_node_id().to_owned(),
            authority_root_id: subject.authority_root_id().to_owned(),
            authority_root_digest: subject.authority_root_digest(),
        }))
    }

    pub fn identity(&self) -> Option<&SafetyProfileAuthorizationHeadIdentity> {
        match self {
            Self::Uninitialized => None,
            Self::Current(identity) => Some(identity),
        }
    }
}

/// Trusted local policy inputs for authorization admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyProfileAuthorizationAdmissionPolicy {
    expected_subject_node_id: String,
    trusted_authority_root_digest: ProfileAuthorityRootDigest,
    current_head: SafetyProfileAuthorizationHead,
}

impl SafetyProfileAuthorizationAdmissionPolicy {
    pub fn new(
        expected_subject_node_id: impl Into<String>,
        trusted_authority_root_digest: ProfileAuthorityRootDigest,
        current_head: SafetyProfileAuthorizationHead,
    ) -> Result<Self, SafetyProfileAuthorizationAdmissionError> {
        let expected_subject_node_id = expected_subject_node_id.into();
        if expected_subject_node_id.trim().is_empty() {
            return Err(SafetyProfileAuthorizationAdmissionError::EmptyExpectedNodeId);
        }

        if let Some(identity) = current_head.identity() {
            if identity.subject_node_id() != expected_subject_node_id {
                return Err(
                    SafetyProfileAuthorizationAdmissionError::PersistedHeadNodeMismatch {
                        expected: expected_subject_node_id,
                        observed: identity.subject_node_id().to_owned(),
                    },
                );
            }
            if identity.authority_root_digest() != trusted_authority_root_digest {
                return Err(
                    SafetyProfileAuthorizationAdmissionError::PersistedHeadAuthorityRootMismatch,
                );
            }
        }

        Ok(Self {
            expected_subject_node_id,
            trusted_authority_root_digest,
            current_head,
        })
    }

    pub fn expected_subject_node_id(&self) -> &str {
        &self.expected_subject_node_id
    }

    pub fn trusted_authority_root_digest(&self) -> ProfileAuthorityRootDigest {
        self.trusted_authority_root_digest
    }

    pub fn current_head(&self) -> &SafetyProfileAuthorizationHead {
        &self.current_head
    }

    /// Check one transition against exact local state and the exact profile artifact
    /// it claims to authorize.
    ///
    /// Validity is half-open: `valid_from <= now < valid_until`.
    ///
    /// Success is deliberately **not** executable authority. The returned value is
    /// a non-serializable policy result whose canonical transition bytes must still
    /// be matched against verifier-owned cryptographic proof before a runtime head
    /// may be committed.
    pub fn check(
        &self,
        now_unix_ms: i64,
        transition: &SafetyProfileAuthorizationTransition,
        profile: &SafetyConfigurationProfile,
    ) -> Result<PolicyCheckedSafetyProfileAuthorization, SafetyProfileAuthorizationAdmissionError>
    {
        transition.validate()?;
        profile.validate()?;
        let subject = transition.subject();

        if subject.subject_node_id() != self.expected_subject_node_id {
            return Err(SafetyProfileAuthorizationAdmissionError::SubjectNodeMismatch {
                expected: self.expected_subject_node_id.clone(),
                observed: subject.subject_node_id().to_owned(),
            });
        }
        if transition.authority_root_digest() != self.trusted_authority_root_digest {
            return Err(
                SafetyProfileAuthorizationAdmissionError::TrustedAuthorityRootMismatch,
            );
        }

        if now_unix_ms < subject.valid_from_unix_ms() {
            return Err(SafetyProfileAuthorizationAdmissionError::NotYetValid {
                now_unix_ms,
                valid_from_unix_ms: subject.valid_from_unix_ms(),
            });
        }
        if now_unix_ms >= subject.valid_until_unix_ms() {
            return Err(SafetyProfileAuthorizationAdmissionError::Expired {
                now_unix_ms,
                valid_until_unix_ms: subject.valid_until_unix_ms(),
            });
        }

        match &self.current_head {
            SafetyProfileAuthorizationHead::Uninitialized => {
                if transition.generation() != 1
                    || transition.predecessor()
                        != SafetyProfileAuthorizationPredecessor::Bootstrap
                {
                    return Err(SafetyProfileAuthorizationAdmissionError::ExpectedBootstrap {
                        observed_generation: transition.generation(),
                        observed_predecessor: transition.predecessor(),
                    });
                }
            }
            SafetyProfileAuthorizationHead::Current(identity) => {
                let expected_generation = identity.generation().checked_add(1).ok_or(
                    SafetyProfileAuthorizationAdmissionError::GenerationExhausted {
                        current: identity.generation(),
                    },
                )?;
                if transition.generation() != expected_generation {
                    return Err(
                        SafetyProfileAuthorizationAdmissionError::GenerationNotSuccessor {
                            current: identity.generation(),
                            expected: expected_generation,
                            observed: transition.generation(),
                        },
                    );
                }

                let expected_predecessor = SafetyProfileAuthorizationPredecessor::Previous(
                    identity.transition_digest(),
                );
                if transition.predecessor() != expected_predecessor {
                    return Err(
                        SafetyProfileAuthorizationAdmissionError::PredecessorHeadMismatch {
                            expected: expected_predecessor,
                            observed: transition.predecessor(),
                        },
                    );
                }

                if subject.authority_root_id() != identity.authority_root_id() {
                    return Err(
                        SafetyProfileAuthorizationAdmissionError::AuthorityRootIdMismatch {
                            expected: identity.authority_root_id().to_owned(),
                            observed: subject.authority_root_id().to_owned(),
                        },
                    );
                }
            }
        }

        if subject.profile_id() != profile.profile_id {
            return Err(SafetyProfileAuthorizationAdmissionError::ProfileIdMismatch {
                expected: subject.profile_id().to_owned(),
                observed: profile.profile_id.clone(),
            });
        }
        let profile_digest = profile.digest()?;
        if subject.profile_digest() != profile_digest {
            return Err(SafetyProfileAuthorizationAdmissionError::ProfileDigestMismatch {
                expected: subject.profile_digest(),
                observed: profile_digest,
            });
        }

        let canonical_transition_bytes = transition.canonical_signing_bytes()?;
        let candidate_head = SafetyProfileAuthorizationHead::from_transition(transition)?;

        Ok(PolicyCheckedSafetyProfileAuthorization {
            transition: transition.clone(),
            profile: profile.clone(),
            canonical_transition_bytes,
            expected_predecessor_head: self.current_head.clone(),
            candidate_head,
        })
    }
}

/// Non-serializable result of exact local admission checks.
///
/// It has no public constructor and does not claim cryptographic verification.
/// A future Xenia adapter must require an opaque verifier proof that matches
/// [`Self::canonical_transition_bytes`] and the externally trusted root before
/// turning this into committed runtime authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyCheckedSafetyProfileAuthorization {
    transition: SafetyProfileAuthorizationTransition,
    profile: SafetyConfigurationProfile,
    canonical_transition_bytes: Vec<u8>,
    expected_predecessor_head: SafetyProfileAuthorizationHead,
    candidate_head: SafetyProfileAuthorizationHead,
}

impl PolicyCheckedSafetyProfileAuthorization {
    pub fn transition(&self) -> &SafetyProfileAuthorizationTransition {
        &self.transition
    }

    pub fn profile(&self) -> &SafetyConfigurationProfile {
        &self.profile
    }

    pub fn canonical_transition_bytes(&self) -> &[u8] {
        &self.canonical_transition_bytes
    }

    /// Exact head that must still be current at commit time. A persistent registry
    /// should compare-and-swap this value rather than performing a separate
    /// unchecked read followed by a write.
    pub fn expected_predecessor_head(&self) -> &SafetyProfileAuthorizationHead {
        &self.expected_predecessor_head
    }

    /// Complete candidate head derived from the exact checked transition.
    pub fn candidate_head(&self) -> &SafetyProfileAuthorizationHead {
        &self.candidate_head
    }

    pub fn candidate_generation(&self) -> u64 {
        self.transition.generation()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyProfileAuthorizationAdmissionError {
    #[error(transparent)]
    Transition(#[from] SafetyProfileAuthorizationTransitionError),
    #[error(transparent)]
    Profile(#[from] SafetyConfigurationProfileError),
    #[error("expected safety-profile authorization node id must not be empty")]
    EmptyExpectedNodeId,
    #[error("persisted safety-profile authorization head belongs to node {observed}, expected {expected}")]
    PersistedHeadNodeMismatch { expected: String, observed: String },
    #[error("persisted safety-profile authorization head does not use the externally trusted root")]
    PersistedHeadAuthorityRootMismatch,
    #[error("safety-profile authorization node mismatch: expected {expected}, observed {observed}")]
    SubjectNodeMismatch { expected: String, observed: String },
    #[error("safety-profile authorization root does not match externally trusted root")]
    TrustedAuthorityRootMismatch,
    #[error("safety-profile authorization root id mismatch with persisted lineage: expected {expected}, observed {observed}")]
    AuthorityRootIdMismatch { expected: String, observed: String },
    #[error("safety-profile authorization is not yet valid: now {now_unix_ms}, valid from {valid_from_unix_ms}")]
    NotYetValid {
        now_unix_ms: i64,
        valid_from_unix_ms: i64,
    },
    #[error("safety-profile authorization is expired: now {now_unix_ms}, valid until {valid_until_unix_ms}")]
    Expired {
        now_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
    #[error("uninitialized authorization head requires generation-1 Bootstrap transition")]
    ExpectedBootstrap {
        observed_generation: u64,
        observed_predecessor: SafetyProfileAuthorizationPredecessor,
    },
    #[error("safety-profile authorization generation space exhausted at {current}")]
    GenerationExhausted { current: u64 },
    #[error("safety-profile authorization generation must be exact successor of {current}: expected {expected}, observed {observed}")]
    GenerationNotSuccessor {
        current: u64,
        expected: u64,
        observed: u64,
    },
    #[error("safety-profile authorization predecessor does not match exact persisted head")]
    PredecessorHeadMismatch {
        expected: SafetyProfileAuthorizationPredecessor,
        observed: SafetyProfileAuthorizationPredecessor,
    },
    #[error("safety-profile authorization profile id mismatch: expected {expected}, observed {observed}")]
    ProfileIdMismatch { expected: String, observed: String },
    #[error("safety-profile authorization profile digest mismatch")]
    ProfileDigestMismatch {
        expected: ConfigurationDigest,
        observed: ConfigurationDigest,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authorization::{ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject};
    use crate::{ComponentRequirement, SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1};

    fn root(byte: u8) -> ProfileAuthorityRootDigest {
        ProfileAuthorityRootDigest::Blake3_256([byte; 32])
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

    fn subject(
        id: &str,
        generation: u64,
        root_digest: ProfileAuthorityRootDigest,
        node: &str,
        profile: &SafetyConfigurationProfile,
    ) -> SafetyProfileAuthorizationSubject {
        SafetyProfileAuthorizationSubject::new(
            id,
            "facility-profile-root-v1",
            root_digest,
            node,
            generation,
            1_000,
            2_000,
            profile,
        )
        .unwrap()
    }

    fn bootstrap(
        id: &str,
        profile: &SafetyConfigurationProfile,
    ) -> SafetyProfileAuthorizationTransition {
        SafetyProfileAuthorizationTransition::bootstrap(subject(
            id,
            1,
            root(0x33),
            "compute-campus",
            profile,
        ))
        .unwrap()
    }

    fn current_head(
        transition: &SafetyProfileAuthorizationTransition,
    ) -> SafetyProfileAuthorizationHead {
        SafetyProfileAuthorizationHead::from_transition(transition).unwrap()
    }

    fn policy(
        head: SafetyProfileAuthorizationHead,
    ) -> SafetyProfileAuthorizationAdmissionPolicy {
        SafetyProfileAuthorizationAdmissionPolicy::new("compute-campus", root(0x33), head).unwrap()
    }

    #[test]
    fn current_head_identity_is_derived_from_one_exact_transition() {
        let profile = profile();
        let transition = bootstrap("auth-1", &profile);
        let head = current_head(&transition);
        let identity = head.identity().unwrap();

        assert_eq!(identity.generation(), 1);
        assert_eq!(identity.transition_digest(), transition.transition_digest().unwrap());
        assert_eq!(identity.subject_node_id(), "compute-campus");
        assert_eq!(identity.authority_root_id(), "facility-profile-root-v1");
        assert_eq!(identity.authority_root_digest(), root(0x33));
    }

    #[test]
    fn exact_bootstrap_is_policy_checked() {
        let profile = profile();
        let transition = bootstrap("auth-1", &profile);
        let checked = policy(SafetyProfileAuthorizationHead::Uninitialized)
            .check(1_000, &transition, &profile)
            .unwrap();

        assert_eq!(
            checked.expected_predecessor_head(),
            &SafetyProfileAuthorizationHead::Uninitialized
        );
        assert_eq!(checked.candidate_generation(), 1);
        assert_eq!(checked.candidate_head(), &current_head(&transition));
        assert_eq!(
            checked.canonical_transition_bytes(),
            transition.canonical_signing_bytes().unwrap()
        );
    }

    #[test]
    fn exact_persisted_head_digest_is_required_for_successor() {
        let profile = profile();
        let head_a = bootstrap("auth-a", &profile);
        let head_b = bootstrap("auth-b", &profile);
        assert_ne!(
            head_a.transition_digest().unwrap(),
            head_b.transition_digest().unwrap()
        );

        let candidate = SafetyProfileAuthorizationTransition::successor(
            subject("auth-2", 2, root(0x33), "compute-campus", &profile),
            &head_a,
        )
        .unwrap();

        policy(current_head(&head_a))
            .check(1_500, &candidate, &profile)
            .unwrap();

        assert!(matches!(
            policy(current_head(&head_b)).check(1_500, &candidate, &profile),
            Err(SafetyProfileAuthorizationAdmissionError::PredecessorHeadMismatch { .. })
        ));
    }

    #[test]
    fn policy_constructor_rejects_head_from_different_node_or_root() {
        let profile = profile();
        let other_node = SafetyProfileAuthorizationTransition::bootstrap(subject(
            "other-node-auth",
            1,
            root(0x33),
            "other-node",
            &profile,
        ))
        .unwrap();
        assert!(matches!(
            SafetyProfileAuthorizationAdmissionPolicy::new(
                "compute-campus",
                root(0x33),
                current_head(&other_node),
            ),
            Err(SafetyProfileAuthorizationAdmissionError::PersistedHeadNodeMismatch { .. })
        ));

        let other_root = SafetyProfileAuthorizationTransition::bootstrap(subject(
            "other-root-auth",
            1,
            root(0x44),
            "compute-campus",
            &profile,
        ))
        .unwrap();
        assert_eq!(
            SafetyProfileAuthorizationAdmissionPolicy::new(
                "compute-campus",
                root(0x33),
                current_head(&other_root),
            ),
            Err(SafetyProfileAuthorizationAdmissionError::PersistedHeadAuthorityRootMismatch)
        );
    }

    #[test]
    fn replay_is_rejected_even_with_correct_root() {
        let profile = profile();
        let head = bootstrap("auth-1", &profile);
        let replay = bootstrap("auth-replay", &profile);

        assert!(matches!(
            policy(current_head(&head)).check(1_500, &replay, &profile),
            Err(SafetyProfileAuthorizationAdmissionError::GenerationNotSuccessor { .. })
        ));
    }

    #[test]
    fn node_root_time_and_profile_all_fail_closed() {
        let profile = profile();
        let transition = bootstrap("auth-1", &profile);

        let wrong_node = SafetyProfileAuthorizationAdmissionPolicy::new(
            "other-node",
            root(0x33),
            SafetyProfileAuthorizationHead::Uninitialized,
        )
        .unwrap();
        assert!(matches!(
            wrong_node.check(1_500, &transition, &profile),
            Err(SafetyProfileAuthorizationAdmissionError::SubjectNodeMismatch { .. })
        ));

        let wrong_root = SafetyProfileAuthorizationAdmissionPolicy::new(
            "compute-campus",
            root(0x44),
            SafetyProfileAuthorizationHead::Uninitialized,
        )
        .unwrap();
        assert_eq!(
            wrong_root.check(1_500, &transition, &profile),
            Err(SafetyProfileAuthorizationAdmissionError::TrustedAuthorityRootMismatch)
        );

        let exact = policy(SafetyProfileAuthorizationHead::Uninitialized);
        assert!(matches!(
            exact.check(999, &transition, &profile),
            Err(SafetyProfileAuthorizationAdmissionError::NotYetValid { .. })
        ));
        assert!(matches!(
            exact.check(2_000, &transition, &profile),
            Err(SafetyProfileAuthorizationAdmissionError::Expired { .. })
        ));

        let mut different_profile = profile.clone();
        different_profile.firmware = ComponentRequirement::NotApplicable;
        assert!(matches!(
            exact.check(1_500, &transition, &different_profile),
            Err(SafetyProfileAuthorizationAdmissionError::ProfileDigestMismatch { .. })
        ));
    }

    #[test]
    fn candidate_head_is_bound_to_checked_transition() {
        let profile = profile();
        let first = bootstrap("auth-1", &profile);
        let second = SafetyProfileAuthorizationTransition::successor(
            subject("auth-2", 2, root(0x33), "compute-campus", &profile),
            &first,
        )
        .unwrap();

        let checked = policy(current_head(&first))
            .check(1_500, &second, &profile)
            .unwrap();
        assert_eq!(checked.candidate_head(), &current_head(&second));
        assert_eq!(
            checked.expected_predecessor_head(),
            &current_head(&first)
        );
    }

    #[test]
    fn generation_overflow_fails_closed() {
        let profile = profile();
        let transition = bootstrap("auth-1", &profile);
        let forged_for_boundary_test = SafetyProfileAuthorizationHead::Current(
            SafetyProfileAuthorizationHeadIdentity {
                generation: u64::MAX,
                transition_digest: transition.transition_digest().unwrap(),
                subject_node_id: "compute-campus".to_owned(),
                authority_root_id: "facility-profile-root-v1".to_owned(),
                authority_root_digest: root(0x33),
            },
        );

        assert_eq!(
            policy(forged_for_boundary_test).check(1_500, &transition, &profile),
            Err(SafetyProfileAuthorizationAdmissionError::GenerationExhausted {
                current: u64::MAX,
            })
        );
    }
}
