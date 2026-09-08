// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Revocation-bound reinstatement transitions for safety-profile authorization.
//!
//! Ordinary authorization successors deliberately cannot clear a committed
//! revocation: their signed bytes bind only the predecessor authorization
//! transition, not a later revocation. Reinstatement is therefore a separate
//! authenticated artifact that binds the exact revocation being superseded and the
//! exact successor authorization transition that will become eligible.
//!
//! This module defines raw canonical evidence only. It performs no signature
//! verification, local admission, persistence, or runtime activation.

use crate::admission::SafetyProfileAuthorizationHead;
use crate::authorization::ProfileAuthorityRootDigest;
use crate::revocation::{
    SafetyProfileAuthorizationRevocationDigest, SafetyProfileAuthorizationRevocationError,
    SafetyProfileAuthorizationRevocationSubject,
};
use crate::transition::{
    SafetyProfileAuthorizationPredecessor, SafetyProfileAuthorizationTransition,
    SafetyProfileAuthorizationTransitionDigest, SafetyProfileAuthorizationTransitionError,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SAFETY_PROFILE_AUTHORIZATION_REINSTATEMENT_SCHEMA_V1: &str =
    "symthaea-safety-profile-authorization-reinstatement-v1";
const DOMAIN_SEPARATOR: &[u8] =
    b"symthaea:safety-profile-authorization-reinstatement:v1\0";

/// Exact digest of one canonical reinstatement transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyProfileAuthorizationReinstatementDigest {
    Blake3_256([u8; 32]),
}

impl SafetyProfileAuthorizationReinstatementDigest {
    pub fn blake3_256(bytes: &[u8]) -> Self {
        Self::Blake3_256(
            symthaea_safety_configuration::ConfigurationDigest::blake3_256(bytes)
                .into_blake3_256(),
        )
    }

    pub fn into_blake3_256(self) -> [u8; 32] {
        match self {
            Self::Blake3_256(bytes) => bytes,
        }
    }
}

/// One exact, revocation-aware restoration of profile authorization eligibility.
///
/// The embedded revocation identifies the exact revoked predecessor head. The
/// embedded ordinary authorization transition must be generation N+1, name that
/// exact predecessor transition digest, preserve node/root identity, and begin no
/// earlier than the revocation's authenticated effective-from boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyProfileAuthorizationReinstatementTransition {
    schema_version: String,
    reinstatement_id: String,
    revocation: SafetyProfileAuthorizationRevocationSubject,
    successor: SafetyProfileAuthorizationTransition,
}

impl SafetyProfileAuthorizationReinstatementTransition {
    /// Construct a reinstatement from the exact revoked predecessor, the exact raw
    /// revocation that targets it, and the exact ordinary successor transition.
    ///
    /// The predecessor itself is not duplicated in the serialized artifact because
    /// both the revocation and successor already bind its exact transition digest.
    pub fn new(
        reinstatement_id: impl Into<String>,
        revoked_predecessor: &SafetyProfileAuthorizationTransition,
        revocation: &SafetyProfileAuthorizationRevocationSubject,
        successor: &SafetyProfileAuthorizationTransition,
    ) -> Result<Self, SafetyProfileAuthorizationReinstatementError> {
        revoked_predecessor.validate()?;
        revocation.validate()?;
        successor.validate()?;

        let predecessor_head = SafetyProfileAuthorizationHead::from_transition(revoked_predecessor)?;
        if !revocation.targets_exact_head(&predecessor_head)? {
            return Err(SafetyProfileAuthorizationReinstatementError::RevocationTargetMismatch);
        }

        let reinstatement = Self {
            schema_version: SAFETY_PROFILE_AUTHORIZATION_REINSTATEMENT_SCHEMA_V1.to_owned(),
            reinstatement_id: reinstatement_id.into(),
            revocation: revocation.clone(),
            successor: successor.clone(),
        };
        reinstatement.validate()?;
        Ok(reinstatement)
    }

    /// Validate all relationships that remain available in the serialized artifact.
    ///
    /// The revocation's target transition digest becomes the exact predecessor that
    /// the successor must name. This prevents deserialized evidence from pairing a
    /// valid revocation with an unrelated generation-N+1 authorization.
    pub fn validate(&self) -> Result<(), SafetyProfileAuthorizationReinstatementError> {
        if self.schema_version != SAFETY_PROFILE_AUTHORIZATION_REINSTATEMENT_SCHEMA_V1 {
            return Err(
                SafetyProfileAuthorizationReinstatementError::UnsupportedSchemaVersion(
                    self.schema_version.clone(),
                ),
            );
        }
        if self.reinstatement_id.trim().is_empty() {
            return Err(SafetyProfileAuthorizationReinstatementError::EmptyReinstatementId);
        }

        self.revocation.validate()?;
        self.successor.validate()?;

        let expected_generation = self
            .revocation
            .target_generation()
            .checked_add(1)
            .ok_or(SafetyProfileAuthorizationReinstatementError::GenerationExhausted {
                current: self.revocation.target_generation(),
            })?;
        if self.successor.generation() != expected_generation {
            return Err(
                SafetyProfileAuthorizationReinstatementError::SuccessorGenerationMismatch {
                    revoked_generation: self.revocation.target_generation(),
                    expected: expected_generation,
                    observed: self.successor.generation(),
                },
            );
        }

        let expected_predecessor = SafetyProfileAuthorizationPredecessor::Previous(
            self.revocation.target_transition_digest(),
        );
        if self.successor.predecessor() != expected_predecessor {
            return Err(
                SafetyProfileAuthorizationReinstatementError::SuccessorPredecessorMismatch {
                    expected: expected_predecessor,
                    observed: self.successor.predecessor(),
                },
            );
        }

        let successor_subject = self.successor.subject();
        if successor_subject.subject_node_id() != self.revocation.subject_node_id() {
            return Err(SafetyProfileAuthorizationReinstatementError::SubjectNodeChanged {
                revoked: self.revocation.subject_node_id().to_owned(),
                successor: successor_subject.subject_node_id().to_owned(),
            });
        }
        if successor_subject.authority_root_id() != self.revocation.authority_root_id() {
            return Err(SafetyProfileAuthorizationReinstatementError::AuthorityRootIdChanged {
                revoked: self.revocation.authority_root_id().to_owned(),
                successor: successor_subject.authority_root_id().to_owned(),
            });
        }
        if successor_subject.authority_root_digest() != self.revocation.authority_root_digest() {
            return Err(SafetyProfileAuthorizationReinstatementError::AuthorityRootChanged);
        }

        if successor_subject.valid_from_unix_ms() < self.revocation.effective_from_unix_ms() {
            return Err(
                SafetyProfileAuthorizationReinstatementError::SuccessorStartsBeforeRevocation {
                    successor_valid_from_unix_ms: successor_subject.valid_from_unix_ms(),
                    revocation_effective_from_unix_ms: self.revocation.effective_from_unix_ms(),
                },
            );
        }

        Ok(())
    }

    /// Fixed-order, domain-separated bytes to be authenticated by the profile
    /// authority before reinstatement admission.
    ///
    /// The complete canonical revocation and complete canonical successor
    /// transition are embedded as length-prefixed opaque byte strings so no field
    /// used by either lower-level contract can be omitted by an integration layer.
    pub fn canonical_signing_bytes(
        &self,
    ) -> Result<Vec<u8>, SafetyProfileAuthorizationReinstatementError> {
        self.validate()?;
        let revocation_bytes = self.revocation.canonical_signing_bytes()?;
        let successor_bytes = self.successor.canonical_signing_bytes()?;
        let mut out = Vec::with_capacity(768);
        out.extend_from_slice(DOMAIN_SEPARATOR);
        push_bytes(
            &mut out,
            "schema_version",
            self.schema_version.as_bytes(),
        )?;
        push_bytes(
            &mut out,
            "reinstatement_id",
            self.reinstatement_id.as_bytes(),
        )?;
        push_bytes(&mut out, "revocation", &revocation_bytes)?;
        push_bytes(&mut out, "successor", &successor_bytes)?;
        Ok(out)
    }

    pub fn reinstatement_digest(
        &self,
    ) -> Result<SafetyProfileAuthorizationReinstatementDigest, SafetyProfileAuthorizationReinstatementError>
    {
        Ok(SafetyProfileAuthorizationReinstatementDigest::blake3_256(
            &self.canonical_signing_bytes()?,
        ))
    }

    pub fn reinstatement_id(&self) -> &str {
        &self.reinstatement_id
    }

    pub fn revocation(&self) -> &SafetyProfileAuthorizationRevocationSubject {
        &self.revocation
    }

    pub fn revocation_digest(
        &self,
    ) -> Result<SafetyProfileAuthorizationRevocationDigest, SafetyProfileAuthorizationReinstatementError>
    {
        Ok(self.revocation.revocation_digest()?)
    }

    pub fn successor(&self) -> &SafetyProfileAuthorizationTransition {
        &self.successor
    }

    pub fn revoked_generation(&self) -> u64 {
        self.revocation.target_generation()
    }

    pub fn revoked_transition_digest(&self) -> SafetyProfileAuthorizationTransitionDigest {
        self.revocation.target_transition_digest()
    }

    pub fn successor_generation(&self) -> u64 {
        self.successor.generation()
    }

    pub fn subject_node_id(&self) -> &str {
        self.successor.subject().subject_node_id()
    }

    pub fn authority_root_id(&self) -> &str {
        self.successor.subject().authority_root_id()
    }

    pub fn authority_root_digest(&self) -> ProfileAuthorityRootDigest {
        self.successor.subject().authority_root_digest()
    }
}

fn push_bytes(
    out: &mut Vec<u8>,
    field: &'static str,
    bytes: &[u8],
) -> Result<(), SafetyProfileAuthorizationReinstatementError> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| SafetyProfileAuthorizationReinstatementError::FieldTooLong(field))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyProfileAuthorizationReinstatementError {
    #[error(transparent)]
    Transition(#[from] SafetyProfileAuthorizationTransitionError),
    #[error(transparent)]
    Revocation(#[from] SafetyProfileAuthorizationRevocationError),
    #[error("unsupported safety-profile authorization reinstatement schema version {0}")]
    UnsupportedSchemaVersion(String),
    #[error("safety-profile authorization reinstatement id must not be empty")]
    EmptyReinstatementId,
    #[error("reinstatement revocation does not target the exact supplied predecessor transition")]
    RevocationTargetMismatch,
    #[error("reinstatement generation space exhausted at revoked generation {current}")]
    GenerationExhausted { current: u64 },
    #[error("reinstatement successor generation mismatch: revoked {revoked_generation}, expected {expected}, observed {observed}")]
    SuccessorGenerationMismatch {
        revoked_generation: u64,
        expected: u64,
        observed: u64,
    },
    #[error("reinstatement successor predecessor mismatch")]
    SuccessorPredecessorMismatch {
        expected: SafetyProfileAuthorizationPredecessor,
        observed: SafetyProfileAuthorizationPredecessor,
    },
    #[error("reinstatement successor node changed from revoked node {revoked} to {successor}")]
    SubjectNodeChanged { revoked: String, successor: String },
    #[error("reinstatement successor authority-root id changed from {revoked} to {successor}")]
    AuthorityRootIdChanged { revoked: String, successor: String },
    #[error("reinstatement successor authority-root digest changed")]
    AuthorityRootChanged,
    #[error("reinstatement successor validity begins at {successor_valid_from_unix_ms}, before revocation effective-from {revocation_effective_from_unix_ms}")]
    SuccessorStartsBeforeRevocation {
        successor_valid_from_unix_ms: i64,
        revocation_effective_from_unix_ms: i64,
    },
    #[error("safety-profile authorization reinstatement field {0} exceeds canonical u32 length")]
    FieldTooLong(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authorization::{ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject};
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
        profile: &SafetyConfigurationProfile,
    ) -> SafetyProfileAuthorizationSubject {
        SafetyProfileAuthorizationSubject::new(
            id,
            "facility-profile-root-v1",
            root(0x33),
            "compute-campus",
            generation,
            valid_from,
            4_000,
            profile,
        )
        .unwrap()
    }

    fn bootstrap(id: &str, profile: &SafetyConfigurationProfile) -> SafetyProfileAuthorizationTransition {
        SafetyProfileAuthorizationTransition::bootstrap(subject(id, 1, 1_000, profile)).unwrap()
    }

    fn successor(
        id: &str,
        predecessor: &SafetyProfileAuthorizationTransition,
        valid_from: i64,
        profile: &SafetyConfigurationProfile,
    ) -> SafetyProfileAuthorizationTransition {
        SafetyProfileAuthorizationTransition::successor(
            subject(id, predecessor.generation() + 1, valid_from, profile),
            predecessor,
        )
        .unwrap()
    }

    fn revocation(
        id: &str,
        target: &SafetyProfileAuthorizationTransition,
        effective_from: i64,
    ) -> SafetyProfileAuthorizationRevocationSubject {
        SafetyProfileAuthorizationRevocationSubject::for_transition(
            id,
            target,
            effective_from,
        )
        .unwrap()
    }

    #[test]
    fn exact_revoked_predecessor_and_successor_form_reinstatement() {
        let original_profile = profile("profile-v1");
        let replacement_profile = profile("profile-v2");
        let first = bootstrap("auth-1", &original_profile);
        let revoke = revocation("revoke-1", &first, 1_500);
        let second = successor("auth-2", &first, 1_500, &replacement_profile);

        let reinstatement = SafetyProfileAuthorizationReinstatementTransition::new(
            "reinstate-1",
            &first,
            &revoke,
            &second,
        )
        .unwrap();

        assert_eq!(reinstatement.revoked_generation(), 1);
        assert_eq!(reinstatement.successor_generation(), 2);
        assert_eq!(
            reinstatement.revoked_transition_digest(),
            first.transition_digest().unwrap()
        );
        assert_eq!(reinstatement.revocation_digest().unwrap(), revoke.revocation_digest().unwrap());
        assert_eq!(reinstatement.successor(), &second);
        assert!(!reinstatement.canonical_signing_bytes().unwrap().is_empty());
    }

    #[test]
    fn same_generation_fork_revocation_cannot_reinstate_other_branch() {
        let p = profile("profile-v1");
        let first = bootstrap("auth-a", &p);
        let fork = bootstrap("auth-b", &p);
        let revoke_first = revocation("revoke-a", &first, 1_500);
        let successor_of_fork = successor("auth-b2", &fork, 1_500, &p);

        assert!(matches!(
            SafetyProfileAuthorizationReinstatementTransition::new(
                "reinstate-wrong-fork",
                &fork,
                &revoke_first,
                &successor_of_fork,
            ),
            Err(SafetyProfileAuthorizationReinstatementError::RevocationTargetMismatch)
        ));
    }

    #[test]
    fn successor_must_name_exact_revoked_predecessor() {
        let p = profile("profile-v1");
        let first = bootstrap("auth-a", &p);
        let other = bootstrap("auth-b", &p);
        let revoke = revocation("revoke-a", &first, 1_500);
        let successor_of_other = successor("auth-b2", &other, 1_500, &p);

        let artifact = SafetyProfileAuthorizationReinstatementTransition {
            schema_version: SAFETY_PROFILE_AUTHORIZATION_REINSTATEMENT_SCHEMA_V1.to_owned(),
            reinstatement_id: "reinstate-cross-branch".to_owned(),
            revocation: revoke,
            successor: successor_of_other,
        };
        assert!(matches!(
            artifact.validate(),
            Err(SafetyProfileAuthorizationReinstatementError::SuccessorPredecessorMismatch { .. })
        ));
    }

    #[test]
    fn successor_cannot_begin_before_revocation_effective_boundary() {
        let p = profile("profile-v1");
        let first = bootstrap("auth-1", &p);
        let revoke = revocation("revoke-1", &first, 1_700);
        let second = successor("auth-2", &first, 1_600, &p);

        assert!(matches!(
            SafetyProfileAuthorizationReinstatementTransition::new(
                "reinstate-early",
                &first,
                &revoke,
                &second,
            ),
            Err(SafetyProfileAuthorizationReinstatementError::SuccessorStartsBeforeRevocation { .. })
        ));
    }

    #[test]
    fn changing_revocation_changes_reinstatement_identity() {
        let p = profile("profile-v1");
        let first = bootstrap("auth-1", &p);
        let second = successor("auth-2", &first, 1_500, &p);
        let revoke_a = revocation("revoke-a", &first, 1_500);
        let revoke_b = revocation("revoke-b", &first, 1_500);

        let a = SafetyProfileAuthorizationReinstatementTransition::new(
            "reinstate-1",
            &first,
            &revoke_a,
            &second,
        )
        .unwrap();
        let b = SafetyProfileAuthorizationReinstatementTransition::new(
            "reinstate-1",
            &first,
            &revoke_b,
            &second,
        )
        .unwrap();

        assert_ne!(a.revocation_digest().unwrap(), b.revocation_digest().unwrap());
        assert_ne!(a.reinstatement_digest().unwrap(), b.reinstatement_digest().unwrap());
        assert_ne!(a.canonical_signing_bytes().unwrap(), b.canonical_signing_bytes().unwrap());
    }

    #[test]
    fn changing_successor_changes_reinstatement_identity() {
        let p = profile("profile-v1");
        let first = bootstrap("auth-1", &p);
        let revoke = revocation("revoke-1", &first, 1_500);
        let second_a = successor("auth-2a", &first, 1_500, &p);
        let second_b = successor("auth-2b", &first, 1_500, &p);

        let a = SafetyProfileAuthorizationReinstatementTransition::new(
            "reinstate-1",
            &first,
            &revoke,
            &second_a,
        )
        .unwrap();
        let b = SafetyProfileAuthorizationReinstatementTransition::new(
            "reinstate-1",
            &first,
            &revoke,
            &second_b,
        )
        .unwrap();

        assert_ne!(a.reinstatement_digest().unwrap(), b.reinstatement_digest().unwrap());
    }
}
