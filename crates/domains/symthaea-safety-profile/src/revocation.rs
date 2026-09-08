// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical revocation claims for safety-profile authorization.
//!
//! V1 profile authorization grants eligibility to use one exact safety profile for
//! configuration qualification and commissioning. It is **not** continuous
//! actuator authority. Revoking a profile authorization prevents further use of
//! that exact authorization head for new qualification/commissioning; it does not
//! itself revoke an already commissioned local-safety envelope. Emergency stop or
//! commissioning revocation belongs to the corresponding runtime authority layer.
//!
//! This module defines raw evidence only. A revocation must still be authenticated
//! by the externally trusted profile-authority root and admitted against the exact
//! current authorization head before it can affect persistent state.

use crate::admission::{SafetyProfileAuthorizationHead, SafetyProfileAuthorizationHeadIdentity};
use crate::authorization::ProfileAuthorityRootDigest;
use crate::transition::{
    SafetyProfileAuthorizationTransition, SafetyProfileAuthorizationTransitionDigest,
    SafetyProfileAuthorizationTransitionError,
};
use serde::{Deserialize, Serialize};
use symthaea_safety_configuration::ConfigurationDigest;
use thiserror::Error;

/// Normative semantic purpose of the v1 authorization schema.
///
/// The schema version itself fixes this meaning. This constant exists so policy,
/// UI, audit, and future bridge code can surface it without inventing alternate
/// interpretations.
pub const SAFETY_PROFILE_AUTHORIZATION_PURPOSE_V1: &str =
    "configuration-qualification-and-commissioning-eligibility";

pub const SAFETY_PROFILE_AUTHORIZATION_REVOCATION_SCHEMA_V1: &str =
    "symthaea-safety-profile-authorization-revocation-v1";
const DOMAIN_SEPARATOR: &[u8] = b"symthaea:safety-profile-authorization-revocation:v1\0";

/// Exact digest of one canonical profile-authorization revocation claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyProfileAuthorizationRevocationDigest {
    Blake3_256([u8; 32]),
}

impl SafetyProfileAuthorizationRevocationDigest {
    pub fn blake3_256(bytes: &[u8]) -> Self {
        Self::Blake3_256(ConfigurationDigest::blake3_256(bytes).into_blake3_256())
    }

    pub fn into_blake3_256(self) -> [u8; 32] {
        match self {
            Self::Blake3_256(bytes) => bytes,
        }
    }
}

/// Immutable revocation message to be authenticated by the same externally
/// trusted profile-authority root as the authorization it targets.
///
/// The target is an exact authorization transition digest, not merely a profile ID
/// or generation. A revocation for one fork therefore cannot silently disable a
/// different same-generation branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyProfileAuthorizationRevocationSubject {
    schema_version: String,
    revocation_id: String,
    authority_root_id: String,
    authority_root_digest: ProfileAuthorityRootDigest,
    subject_node_id: String,
    target_generation: u64,
    target_transition_digest: SafetyProfileAuthorizationTransitionDigest,
    effective_from_unix_ms: i64,
}

impl SafetyProfileAuthorizationRevocationSubject {
    /// Construct a revocation directly from the exact transition being revoked.
    ///
    /// Node, root, generation, and transition digest are derived rather than
    /// caller-supplied so the normal construction path cannot create internally
    /// inconsistent target metadata.
    pub fn for_transition(
        revocation_id: impl Into<String>,
        target: &SafetyProfileAuthorizationTransition,
        effective_from_unix_ms: i64,
    ) -> Result<Self, SafetyProfileAuthorizationRevocationError> {
        target.validate()?;
        let subject = target.subject();
        let revocation = Self {
            schema_version: SAFETY_PROFILE_AUTHORIZATION_REVOCATION_SCHEMA_V1.to_owned(),
            revocation_id: revocation_id.into(),
            authority_root_id: subject.authority_root_id().to_owned(),
            authority_root_digest: subject.authority_root_digest(),
            subject_node_id: subject.subject_node_id().to_owned(),
            target_generation: target.generation(),
            target_transition_digest: target.transition_digest()?,
            effective_from_unix_ms,
        };
        revocation.validate()?;
        Ok(revocation)
    }

    pub fn validate(&self) -> Result<(), SafetyProfileAuthorizationRevocationError> {
        if self.schema_version != SAFETY_PROFILE_AUTHORIZATION_REVOCATION_SCHEMA_V1 {
            return Err(
                SafetyProfileAuthorizationRevocationError::UnsupportedSchemaVersion(
                    self.schema_version.clone(),
                ),
            );
        }
        if self.revocation_id.trim().is_empty() {
            return Err(SafetyProfileAuthorizationRevocationError::EmptyRevocationId);
        }
        if self.authority_root_id.trim().is_empty() {
            return Err(SafetyProfileAuthorizationRevocationError::EmptyAuthorityRootId);
        }
        if self.subject_node_id.trim().is_empty() {
            return Err(SafetyProfileAuthorizationRevocationError::EmptySubjectNodeId);
        }
        if self.target_generation == 0 {
            return Err(SafetyProfileAuthorizationRevocationError::ZeroTargetGeneration);
        }
        Ok(())
    }

    /// Fixed-order, domain-separated bytes that the profile authority must
    /// authenticate.
    ///
    /// V1 encoding:
    /// - revocation domain separator
    /// - schema, revocation ID, authority-root ID, and node ID as big-endian u32
    ///   length-prefixed UTF-8
    /// - target generation as big-endian u64
    /// - effective-from as big-endian signed i64 Unix milliseconds
    /// - authority-root digest algorithm tag `1` + 32 digest bytes
    /// - target-transition digest algorithm tag `1` + 32 digest bytes
    pub fn canonical_signing_bytes(
        &self,
    ) -> Result<Vec<u8>, SafetyProfileAuthorizationRevocationError> {
        self.validate()?;
        let mut out = Vec::with_capacity(256);
        out.extend_from_slice(DOMAIN_SEPARATOR);
        push_string(&mut out, "schema_version", &self.schema_version)?;
        push_string(&mut out, "revocation_id", &self.revocation_id)?;
        push_string(&mut out, "authority_root_id", &self.authority_root_id)?;
        push_string(&mut out, "subject_node_id", &self.subject_node_id)?;
        out.extend_from_slice(&self.target_generation.to_be_bytes());
        out.extend_from_slice(&self.effective_from_unix_ms.to_be_bytes());
        push_root_digest(&mut out, self.authority_root_digest);
        push_transition_digest(&mut out, self.target_transition_digest);
        Ok(out)
    }

    pub fn revocation_digest(
        &self,
    ) -> Result<SafetyProfileAuthorizationRevocationDigest, SafetyProfileAuthorizationRevocationError>
    {
        Ok(SafetyProfileAuthorizationRevocationDigest::blake3_256(
            &self.canonical_signing_bytes()?,
        ))
    }

    /// Check whether this raw revocation names one exact current authorization
    /// head. This is a structural/policy check only and does not authenticate the
    /// revocation signature.
    pub fn targets_exact_head(
        &self,
        head: &SafetyProfileAuthorizationHead,
    ) -> Result<bool, SafetyProfileAuthorizationRevocationError> {
        self.validate()?;
        let Some(identity) = head.identity() else {
            return Ok(false);
        };
        Ok(self.matches_identity(identity))
    }

    fn matches_identity(&self, identity: &SafetyProfileAuthorizationHeadIdentity) -> bool {
        self.target_generation == identity.generation()
            && self.target_transition_digest == identity.transition_digest()
            && self.subject_node_id == identity.subject_node_id()
            && self.authority_root_id == identity.authority_root_id()
            && self.authority_root_digest == identity.authority_root_digest()
    }

    pub fn revocation_id(&self) -> &str {
        &self.revocation_id
    }

    pub fn authority_root_id(&self) -> &str {
        &self.authority_root_id
    }

    pub fn authority_root_digest(&self) -> ProfileAuthorityRootDigest {
        self.authority_root_digest
    }

    pub fn subject_node_id(&self) -> &str {
        &self.subject_node_id
    }

    pub fn target_generation(&self) -> u64 {
        self.target_generation
    }

    pub fn target_transition_digest(&self) -> SafetyProfileAuthorizationTransitionDigest {
        self.target_transition_digest
    }

    pub fn effective_from_unix_ms(&self) -> i64 {
        self.effective_from_unix_ms
    }
}

fn push_string(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &str,
) -> Result<(), SafetyProfileAuthorizationRevocationError> {
    let len = u32::try_from(value.len())
        .map_err(|_| SafetyProfileAuthorizationRevocationError::StringTooLong(field))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_root_digest(out: &mut Vec<u8>, digest: ProfileAuthorityRootDigest) {
    match digest {
        ProfileAuthorityRootDigest::Blake3_256(bytes) => {
            out.push(1);
            out.extend_from_slice(&bytes);
        }
    }
}

fn push_transition_digest(
    out: &mut Vec<u8>,
    digest: SafetyProfileAuthorizationTransitionDigest,
) {
    match digest {
        SafetyProfileAuthorizationTransitionDigest::Blake3_256(bytes) => {
            out.push(1);
            out.extend_from_slice(&bytes);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyProfileAuthorizationRevocationError {
    #[error(transparent)]
    Transition(#[from] SafetyProfileAuthorizationTransitionError),
    #[error("unsupported safety-profile authorization revocation schema version {0}")]
    UnsupportedSchemaVersion(String),
    #[error("safety-profile authorization revocation id must not be empty")]
    EmptyRevocationId,
    #[error("safety-profile authorization revocation root id must not be empty")]
    EmptyAuthorityRootId,
    #[error("safety-profile authorization revocation subject node id must not be empty")]
    EmptySubjectNodeId,
    #[error("safety-profile authorization revocation target generation must be greater than zero")]
    ZeroTargetGeneration,
    #[error("safety-profile authorization revocation string field {0} exceeds canonical u32 length")]
    StringTooLong(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authorization::SafetyProfileAuthorizationSubject;
    use crate::{ComponentRequirement, SafetyConfigurationProfile, SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1};

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

    fn bootstrap(id: &str) -> SafetyProfileAuthorizationTransition {
        let profile = profile();
        let subject = SafetyProfileAuthorizationSubject::new(
            id,
            "facility-profile-root-v1",
            root(0x33),
            "compute-campus",
            1,
            1_000,
            2_000,
            &profile,
        )
        .unwrap();
        SafetyProfileAuthorizationTransition::bootstrap(subject).unwrap()
    }

    #[test]
    fn constructor_binds_exact_transition_identity() {
        let target = bootstrap("auth-a");
        let revocation = SafetyProfileAuthorizationRevocationSubject::for_transition(
            "revoke-a",
            &target,
            1_500,
        )
        .unwrap();

        assert_eq!(revocation.target_generation(), 1);
        assert_eq!(
            revocation.target_transition_digest(),
            target.transition_digest().unwrap()
        );
        assert_eq!(revocation.subject_node_id(), "compute-campus");
        assert_eq!(revocation.authority_root_id(), "facility-profile-root-v1");
        assert_eq!(revocation.authority_root_digest(), root(0x33));
        assert_eq!(
            SAFETY_PROFILE_AUTHORIZATION_PURPOSE_V1,
            "configuration-qualification-and-commissioning-eligibility"
        );
    }

    #[test]
    fn revocation_matches_only_the_exact_authorization_head() {
        let target = bootstrap("auth-a");
        let other = bootstrap("auth-b");
        let revocation = SafetyProfileAuthorizationRevocationSubject::for_transition(
            "revoke-a",
            &target,
            1_500,
        )
        .unwrap();

        let target_head = SafetyProfileAuthorizationHead::from_transition(&target).unwrap();
        let other_head = SafetyProfileAuthorizationHead::from_transition(&other).unwrap();
        assert!(revocation.targets_exact_head(&target_head).unwrap());
        assert!(!revocation.targets_exact_head(&other_head).unwrap());
        assert!(!revocation
            .targets_exact_head(&SafetyProfileAuthorizationHead::Uninitialized)
            .unwrap());
    }

    #[test]
    fn same_generation_fork_has_different_revocation_identity() {
        let first = bootstrap("auth-a");
        let second = bootstrap("auth-b");
        let first_revocation = SafetyProfileAuthorizationRevocationSubject::for_transition(
            "revoke",
            &first,
            1_500,
        )
        .unwrap();
        let second_revocation = SafetyProfileAuthorizationRevocationSubject::for_transition(
            "revoke",
            &second,
            1_500,
        )
        .unwrap();

        assert_ne!(
            first_revocation.canonical_signing_bytes().unwrap(),
            second_revocation.canonical_signing_bytes().unwrap()
        );
        assert_ne!(
            first_revocation.revocation_digest().unwrap(),
            second_revocation.revocation_digest().unwrap()
        );
    }

    #[test]
    fn effective_time_is_authenticated() {
        let target = bootstrap("auth-a");
        let first = SafetyProfileAuthorizationRevocationSubject::for_transition(
            "revoke-a",
            &target,
            1_500,
        )
        .unwrap();
        let second = SafetyProfileAuthorizationRevocationSubject::for_transition(
            "revoke-a",
            &target,
            1_501,
        )
        .unwrap();
        assert_ne!(
            first.canonical_signing_bytes().unwrap(),
            second.canonical_signing_bytes().unwrap()
        );
    }

    #[test]
    fn revocation_id_is_authenticated() {
        let target = bootstrap("auth-a");
        let first = SafetyProfileAuthorizationRevocationSubject::for_transition(
            "revoke-a",
            &target,
            1_500,
        )
        .unwrap();
        let second = SafetyProfileAuthorizationRevocationSubject::for_transition(
            "revoke-b",
            &target,
            1_500,
        )
        .unwrap();
        assert_ne!(
            first.canonical_signing_bytes().unwrap(),
            second.canonical_signing_bytes().unwrap()
        );
    }

    #[test]
    fn empty_revocation_id_fails_closed() {
        let target = bootstrap("auth-a");
        assert_eq!(
            SafetyProfileAuthorizationRevocationSubject::for_transition("", &target, 1_500),
            Err(SafetyProfileAuthorizationRevocationError::EmptyRevocationId)
        );
    }
}
