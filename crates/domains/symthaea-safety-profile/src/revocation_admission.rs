// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed local admission for safety-profile authorization revocations.
//!
//! A canonical revocation claim is still only evidence. This module checks whether
//! one revocation is admissible against the exact locally provisioned profile root,
//! exact current authorization head, and one uncertainty-aware trusted-time
//! observation. Successful admission remains non-cryptographic and cannot mutate
//! persistent state.

use crate::admission::SafetyProfileAuthorizationHead;
use crate::commit_preconditions::ProfileAuthorityRootSnapshot;
use crate::revocation::{
    SafetyProfileAuthorizationRevocationDigest, SafetyProfileAuthorizationRevocationError,
    SafetyProfileAuthorizationRevocationSubject,
};
use crate::trusted_time::TrustedAuthorizationClockObservation;
use thiserror::Error;

/// Trusted local policy inputs for admitting one profile-authorization revocation.
///
/// Revocation is intentionally defined only for an exact current authorization
/// head. An uninitialized node has nothing to revoke, and stale/forked revocations
/// cannot be applied merely because the same root signed them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyProfileAuthorizationRevocationAdmissionPolicy {
    expected_subject_node_id: String,
    expected_root: ProfileAuthorityRootSnapshot,
    current_head: SafetyProfileAuthorizationHead,
}

impl SafetyProfileAuthorizationRevocationAdmissionPolicy {
    pub fn new(
        expected_subject_node_id: impl Into<String>,
        expected_root: ProfileAuthorityRootSnapshot,
        current_head: SafetyProfileAuthorizationHead,
    ) -> Result<Self, SafetyProfileAuthorizationRevocationAdmissionError> {
        let expected_subject_node_id = expected_subject_node_id.into();
        if expected_subject_node_id.trim().is_empty() {
            return Err(
                SafetyProfileAuthorizationRevocationAdmissionError::EmptyExpectedSubjectNodeId,
            );
        }

        let identity = current_head.identity().ok_or(
            SafetyProfileAuthorizationRevocationAdmissionError::NoCurrentAuthorizationHead,
        )?;
        if identity.subject_node_id() != expected_subject_node_id {
            return Err(
                SafetyProfileAuthorizationRevocationAdmissionError::PersistedHeadNodeMismatch {
                    expected: expected_subject_node_id,
                    observed: identity.subject_node_id().to_owned(),
                },
            );
        }
        if identity.authority_root_id() != expected_root.root_id() {
            return Err(
                SafetyProfileAuthorizationRevocationAdmissionError::PersistedHeadAuthorityRootIdMismatch {
                    expected: expected_root.root_id().to_owned(),
                    observed: identity.authority_root_id().to_owned(),
                },
            );
        }
        if identity.authority_root_digest() != expected_root.root_digest() {
            return Err(
                SafetyProfileAuthorizationRevocationAdmissionError::PersistedHeadAuthorityRootDigestMismatch,
            );
        }

        Ok(Self {
            expected_subject_node_id,
            expected_root,
            current_head,
        })
    }

    pub fn expected_subject_node_id(&self) -> &str {
        &self.expected_subject_node_id
    }

    pub fn expected_root(&self) -> &ProfileAuthorityRootSnapshot {
        &self.expected_root
    }

    pub fn current_head(&self) -> &SafetyProfileAuthorizationHead {
        &self.current_head
    }

    /// Check one raw revocation against exact local state.
    ///
    /// The revocation becomes admissible only when trusted time proves it is
    /// definitely effective. If the uncertainty interval straddles the signed
    /// effective-from boundary, admission fails closed.
    ///
    /// Success does **not** prove the revocation signature and does not change
    /// runtime state. The returned value has no public constructor and carries the
    /// exact canonical bytes a verifier must authenticate later.
    pub fn check(
        &self,
        current_clock: &TrustedAuthorizationClockObservation,
        revocation: &SafetyProfileAuthorizationRevocationSubject,
    ) -> Result<PolicyCheckedSafetyProfileAuthorizationRevocation, SafetyProfileAuthorizationRevocationAdmissionError>
    {
        revocation.validate()?;

        if revocation.subject_node_id() != self.expected_subject_node_id {
            return Err(
                SafetyProfileAuthorizationRevocationAdmissionError::SubjectNodeMismatch {
                    expected: self.expected_subject_node_id.clone(),
                    observed: revocation.subject_node_id().to_owned(),
                },
            );
        }
        if revocation.authority_root_id() != self.expected_root.root_id() {
            return Err(
                SafetyProfileAuthorizationRevocationAdmissionError::AuthorityRootIdMismatch {
                    expected: self.expected_root.root_id().to_owned(),
                    observed: revocation.authority_root_id().to_owned(),
                },
            );
        }
        if revocation.authority_root_digest() != self.expected_root.root_digest() {
            return Err(
                SafetyProfileAuthorizationRevocationAdmissionError::AuthorityRootDigestMismatch,
            );
        }
        if !revocation.targets_exact_head(&self.current_head)? {
            return Err(
                SafetyProfileAuthorizationRevocationAdmissionError::TargetHeadMismatch,
            );
        }

        let effective_from_unix_ms = revocation.effective_from_unix_ms();
        if current_clock.latest_unix_ms() < effective_from_unix_ms {
            return Err(
                SafetyProfileAuthorizationRevocationAdmissionError::NotYetEffective {
                    latest_unix_ms: current_clock.latest_unix_ms(),
                    effective_from_unix_ms,
                },
            );
        }
        if current_clock.earliest_unix_ms() < effective_from_unix_ms {
            return Err(
                SafetyProfileAuthorizationRevocationAdmissionError::UncertaintyCrossesEffectiveBoundary {
                    earliest_unix_ms: current_clock.earliest_unix_ms(),
                    latest_unix_ms: current_clock.latest_unix_ms(),
                    effective_from_unix_ms,
                },
            );
        }

        Ok(PolicyCheckedSafetyProfileAuthorizationRevocation {
            revocation: revocation.clone(),
            canonical_revocation_bytes: revocation.canonical_signing_bytes()?,
            revocation_digest: revocation.revocation_digest()?,
            expected_root: self.expected_root.clone(),
            expected_current_head: self.current_head.clone(),
            expected_clock_source_id: current_clock.source_id().to_owned(),
            expected_clock_epoch: current_clock.epoch(),
        })
    }
}

/// Non-serializable result of exact local revocation-admission checks.
///
/// This is deliberately not a proof of cryptographic verification. A future Xenia
/// adapter must authenticate [`Self::canonical_revocation_bytes`] under the exact
/// root captured here before a persistent revoked-head marker may be committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyCheckedSafetyProfileAuthorizationRevocation {
    revocation: SafetyProfileAuthorizationRevocationSubject,
    canonical_revocation_bytes: Vec<u8>,
    revocation_digest: SafetyProfileAuthorizationRevocationDigest,
    expected_root: ProfileAuthorityRootSnapshot,
    expected_current_head: SafetyProfileAuthorizationHead,
    expected_clock_source_id: String,
    expected_clock_epoch: u64,
}

impl PolicyCheckedSafetyProfileAuthorizationRevocation {
    pub fn revocation(&self) -> &SafetyProfileAuthorizationRevocationSubject {
        &self.revocation
    }

    pub fn canonical_revocation_bytes(&self) -> &[u8] {
        &self.canonical_revocation_bytes
    }

    pub fn revocation_digest(&self) -> SafetyProfileAuthorizationRevocationDigest {
        self.revocation_digest
    }

    pub fn expected_root(&self) -> &ProfileAuthorityRootSnapshot {
        &self.expected_root
    }

    pub fn expected_current_head(&self) -> &SafetyProfileAuthorizationHead {
        &self.expected_current_head
    }

    pub fn expected_clock_source_id(&self) -> &str {
        &self.expected_clock_source_id
    }

    pub fn expected_clock_epoch(&self) -> u64 {
        self.expected_clock_epoch
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyProfileAuthorizationRevocationAdmissionError {
    #[error(transparent)]
    Revocation(#[from] SafetyProfileAuthorizationRevocationError),
    #[error("expected revocation subject node id must not be empty")]
    EmptyExpectedSubjectNodeId,
    #[error("cannot revoke a profile authorization when no current authorization head exists")]
    NoCurrentAuthorizationHead,
    #[error("persisted authorization head belongs to node {observed}, expected {expected}")]
    PersistedHeadNodeMismatch { expected: String, observed: String },
    #[error("persisted authorization head root id mismatch: expected {expected}, observed {observed}")]
    PersistedHeadAuthorityRootIdMismatch { expected: String, observed: String },
    #[error("persisted authorization head root digest does not match the provisioned root")]
    PersistedHeadAuthorityRootDigestMismatch,
    #[error("revocation subject node mismatch: expected {expected}, observed {observed}")]
    SubjectNodeMismatch { expected: String, observed: String },
    #[error("revocation authority-root id mismatch: expected {expected}, observed {observed}")]
    AuthorityRootIdMismatch { expected: String, observed: String },
    #[error("revocation authority-root digest does not match the provisioned root")]
    AuthorityRootDigestMismatch,
    #[error("revocation does not target the exact current authorization head")]
    TargetHeadMismatch,
    #[error("revocation is definitely not yet effective: latest possible time {latest_unix_ms} < effective-from {effective_from_unix_ms}")]
    NotYetEffective {
        latest_unix_ms: i64,
        effective_from_unix_ms: i64,
    },
    #[error("trusted clock uncertainty [{earliest_unix_ms}, {latest_unix_ms}] crosses revocation effective-from {effective_from_unix_ms}")]
    UncertaintyCrossesEffectiveBoundary {
        earliest_unix_ms: i64,
        latest_unix_ms: i64,
        effective_from_unix_ms: i64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::SafetyProfileAuthorizationHead;
    use crate::authorization::{ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject};
    use crate::revocation::SafetyProfileAuthorizationRevocationSubject;
    use crate::transition::SafetyProfileAuthorizationTransition;
    use crate::{
        ComponentRequirement, SafetyConfigurationProfile, SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
    };

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

    fn transition(id: &str, root_id: &str, root_digest: ProfileAuthorityRootDigest) -> SafetyProfileAuthorizationTransition {
        let profile = profile();
        let subject = SafetyProfileAuthorizationSubject::new(
            id,
            root_id,
            root_digest,
            "compute-campus",
            1,
            1_000,
            2_000,
            &profile,
        )
        .unwrap();
        SafetyProfileAuthorizationTransition::bootstrap(subject).unwrap()
    }

    fn clock(earliest: i64, latest: i64) -> TrustedAuthorizationClockObservation {
        TrustedAuthorizationClockObservation::new("secure-rtc-v1", 7, earliest, latest).unwrap()
    }

    fn policy_for(
        target: &SafetyProfileAuthorizationTransition,
    ) -> SafetyProfileAuthorizationRevocationAdmissionPolicy {
        SafetyProfileAuthorizationRevocationAdmissionPolicy::new(
            "compute-campus",
            ProfileAuthorityRootSnapshot::new("facility-profile-root-v1", root(0x33), 4).unwrap(),
            SafetyProfileAuthorizationHead::from_transition(target).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn exact_current_head_and_effective_time_are_admitted() {
        let target = transition("auth-a", "facility-profile-root-v1", root(0x33));
        let revocation =
            SafetyProfileAuthorizationRevocationSubject::for_transition("revoke-a", &target, 1_400)
                .unwrap();
        let checked = policy_for(&target)
            .check(&clock(1_500, 1_600), &revocation)
            .unwrap();

        assert_eq!(checked.canonical_revocation_bytes(), revocation.canonical_signing_bytes().unwrap());
        assert_eq!(checked.revocation_digest(), revocation.revocation_digest().unwrap());
        assert_eq!(checked.expected_current_head(), &SafetyProfileAuthorizationHead::from_transition(&target).unwrap());
        assert_eq!(checked.expected_root().epoch(), 4);
        assert_eq!(checked.expected_clock_source_id(), "secure-rtc-v1");
        assert_eq!(checked.expected_clock_epoch(), 7);
    }

    #[test]
    fn same_generation_fork_cannot_be_revoked_through_wrong_head() {
        let target = transition("auth-a", "facility-profile-root-v1", root(0x33));
        let other = transition("auth-b", "facility-profile-root-v1", root(0x33));
        let revocation =
            SafetyProfileAuthorizationRevocationSubject::for_transition("revoke-a", &target, 1_400)
                .unwrap();

        assert_eq!(
            policy_for(&other).check(&clock(1_500, 1_600), &revocation),
            Err(SafetyProfileAuthorizationRevocationAdmissionError::TargetHeadMismatch)
        );
    }

    #[test]
    fn root_alias_with_same_digest_fails_closed() {
        let target = transition("auth-a", "alias-root", root(0x33));
        let current_head = SafetyProfileAuthorizationHead::from_transition(&target).unwrap();
        assert!(matches!(
            SafetyProfileAuthorizationRevocationAdmissionPolicy::new(
                "compute-campus",
                ProfileAuthorityRootSnapshot::new("facility-profile-root-v1", root(0x33), 4).unwrap(),
                current_head,
            ),
            Err(SafetyProfileAuthorizationRevocationAdmissionError::PersistedHeadAuthorityRootIdMismatch { .. })
        ));
    }

    #[test]
    fn future_revocation_is_not_yet_admissible() {
        let target = transition("auth-a", "facility-profile-root-v1", root(0x33));
        let revocation =
            SafetyProfileAuthorizationRevocationSubject::for_transition("revoke-a", &target, 1_700)
                .unwrap();
        assert!(matches!(
            policy_for(&target).check(&clock(1_500, 1_600), &revocation),
            Err(SafetyProfileAuthorizationRevocationAdmissionError::NotYetEffective { .. })
        ));
    }

    #[test]
    fn uncertainty_crossing_effective_boundary_fails_closed() {
        let target = transition("auth-a", "facility-profile-root-v1", root(0x33));
        let revocation =
            SafetyProfileAuthorizationRevocationSubject::for_transition("revoke-a", &target, 1_550)
                .unwrap();
        assert!(matches!(
            policy_for(&target).check(&clock(1_500, 1_600), &revocation),
            Err(SafetyProfileAuthorizationRevocationAdmissionError::UncertaintyCrossesEffectiveBoundary { .. })
        ));
    }

    #[test]
    fn uninitialized_head_cannot_be_revoked() {
        assert_eq!(
            SafetyProfileAuthorizationRevocationAdmissionPolicy::new(
                "compute-campus",
                ProfileAuthorityRootSnapshot::new("facility-profile-root-v1", root(0x33), 4).unwrap(),
                SafetyProfileAuthorizationHead::Uninitialized,
            ),
            Err(SafetyProfileAuthorizationRevocationAdmissionError::NoCurrentAuthorizationHead)
        );
    }
}
