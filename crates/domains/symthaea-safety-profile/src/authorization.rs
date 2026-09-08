// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical signing subjects for safety-profile authorization.
//!
//! This module deliberately defines only the message that an external trust
//! verifier must authenticate. A syntactically valid authorization subject is not
//! executable authority and this module does not provide a "verified" wrapper.
//! Cryptographic verification belongs at the Xenia / deployment trust boundary.

use crate::{SafetyConfigurationProfile, SafetyConfigurationProfileError};
use serde::{Deserialize, Serialize};
use symthaea_safety_configuration::ConfigurationDigest;
use thiserror::Error;

pub const SAFETY_PROFILE_AUTHORIZATION_SCHEMA_V1: &str =
    "symthaea-safety-profile-authorization-v1";
const DOMAIN_SEPARATOR: &[u8] = b"symthaea:safety-profile-authorization:v1\0";

/// Fingerprint of the externally provisioned profile-authority trust root.
///
/// This is identity, not proof of possession. A verifier must establish that a
/// signature/capability actually chains to this root before treating a subject as
/// authorized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProfileAuthorityRootDigest {
    Blake3_256([u8; 32]),
}

/// Immutable, serializable message to be authenticated by the profile authority.
///
/// The subject binds one exact profile artifact to one exact infrastructure node
/// under one exact externally provisioned authority root. Generations and validity
/// bounds are included in the signed message so replay/rotation policy can be
/// enforced by the future verifier without relying on mutable side metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyProfileAuthorizationSubject {
    schema_version: String,
    authorization_id: String,
    authority_root_id: String,
    authority_root_digest: ProfileAuthorityRootDigest,
    subject_node_id: String,
    generation: u64,
    valid_from_unix_ms: i64,
    valid_until_unix_ms: i64,
    profile_id: String,
    profile_digest: ConfigurationDigest,
}

impl SafetyProfileAuthorizationSubject {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        authorization_id: impl Into<String>,
        authority_root_id: impl Into<String>,
        authority_root_digest: ProfileAuthorityRootDigest,
        subject_node_id: impl Into<String>,
        generation: u64,
        valid_from_unix_ms: i64,
        valid_until_unix_ms: i64,
        profile: &SafetyConfigurationProfile,
    ) -> Result<Self, SafetyProfileAuthorizationError> {
        profile.validate()?;
        let subject = Self {
            schema_version: SAFETY_PROFILE_AUTHORIZATION_SCHEMA_V1.to_owned(),
            authorization_id: authorization_id.into(),
            authority_root_id: authority_root_id.into(),
            authority_root_digest,
            subject_node_id: subject_node_id.into(),
            generation,
            valid_from_unix_ms,
            valid_until_unix_ms,
            profile_id: profile.profile_id.clone(),
            profile_digest: profile.digest()?,
        };
        subject.validate()?;
        Ok(subject)
    }

    pub fn validate(&self) -> Result<(), SafetyProfileAuthorizationError> {
        if self.schema_version != SAFETY_PROFILE_AUTHORIZATION_SCHEMA_V1 {
            return Err(SafetyProfileAuthorizationError::UnsupportedSchemaVersion(
                self.schema_version.clone(),
            ));
        }
        if self.authorization_id.trim().is_empty() {
            return Err(SafetyProfileAuthorizationError::EmptyAuthorizationId);
        }
        if self.authority_root_id.trim().is_empty() {
            return Err(SafetyProfileAuthorizationError::EmptyAuthorityRootId);
        }
        if self.subject_node_id.trim().is_empty() {
            return Err(SafetyProfileAuthorizationError::EmptySubjectNodeId);
        }
        if self.generation == 0 {
            return Err(SafetyProfileAuthorizationError::ZeroGeneration);
        }
        if self.valid_from_unix_ms >= self.valid_until_unix_ms {
            return Err(SafetyProfileAuthorizationError::InvalidValidityWindow {
                valid_from_unix_ms: self.valid_from_unix_ms,
                valid_until_unix_ms: self.valid_until_unix_ms,
            });
        }
        if self.profile_id.trim().is_empty() {
            return Err(SafetyProfileAuthorizationError::EmptyProfileId);
        }
        Ok(())
    }

    /// Fixed-order, domain-separated bytes to sign or otherwise authenticate.
    ///
    /// V1 encoding:
    /// - domain separator
    /// - schema, authorization ID, authority-root ID, and subject node ID as
    ///   big-endian u32 length-prefixed UTF-8
    /// - generation as big-endian u64
    /// - validity bounds as big-endian signed i64 Unix milliseconds
    /// - profile ID as big-endian u32 length-prefixed UTF-8
    /// - authority-root digest algorithm tag + 32 bytes
    /// - profile digest algorithm tag + 32 bytes
    pub fn canonical_signing_bytes(&self) -> Result<Vec<u8>, SafetyProfileAuthorizationError> {
        self.validate()?;
        let mut out = Vec::with_capacity(256);
        out.extend_from_slice(DOMAIN_SEPARATOR);
        push_string(&mut out, "schema_version", &self.schema_version)?;
        push_string(&mut out, "authorization_id", &self.authorization_id)?;
        push_string(&mut out, "authority_root_id", &self.authority_root_id)?;
        push_string(&mut out, "subject_node_id", &self.subject_node_id)?;
        out.extend_from_slice(&self.generation.to_be_bytes());
        out.extend_from_slice(&self.valid_from_unix_ms.to_be_bytes());
        out.extend_from_slice(&self.valid_until_unix_ms.to_be_bytes());
        push_string(&mut out, "profile_id", &self.profile_id)?;
        push_root_digest(&mut out, self.authority_root_digest);
        push_configuration_digest(&mut out, self.profile_digest);
        Ok(out)
    }

    pub fn authorization_id(&self) -> &str {
        &self.authorization_id
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

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn valid_from_unix_ms(&self) -> i64 {
        self.valid_from_unix_ms
    }

    pub fn valid_until_unix_ms(&self) -> i64 {
        self.valid_until_unix_ms
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn profile_digest(&self) -> ConfigurationDigest {
        self.profile_digest
    }
}

fn push_string(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &str,
) -> Result<(), SafetyProfileAuthorizationError> {
    let len = u32::try_from(value.len())
        .map_err(|_| SafetyProfileAuthorizationError::StringTooLong(field))?;
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

fn push_configuration_digest(out: &mut Vec<u8>, digest: ConfigurationDigest) {
    match digest {
        ConfigurationDigest::Blake3_256(bytes) => {
            out.push(1);
            out.extend_from_slice(&bytes);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyProfileAuthorizationError {
    #[error(transparent)]
    Profile(#[from] SafetyConfigurationProfileError),
    #[error("unsupported safety-profile authorization schema version {0}")]
    UnsupportedSchemaVersion(String),
    #[error("safety-profile authorization id must not be empty")]
    EmptyAuthorizationId,
    #[error("profile-authority root id must not be empty")]
    EmptyAuthorityRootId,
    #[error("safety-profile authorization subject node id must not be empty")]
    EmptySubjectNodeId,
    #[error("safety-profile authorization generation must be greater than zero")]
    ZeroGeneration,
    #[error(
        "invalid safety-profile authorization validity window: {valid_from_unix_ms} >= {valid_until_unix_ms}"
    )]
    InvalidValidityWindow {
        valid_from_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
    #[error("safety-profile authorization profile id must not be empty")]
    EmptyProfileId,
    #[error("safety-profile authorization string field {0} exceeds canonical u32 length")]
    StringTooLong(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComponentRequirement, SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1};

    fn config_digest(byte: u8) -> ConfigurationDigest {
        ConfigurationDigest::Blake3_256([byte; 32])
    }

    fn root_digest(byte: u8) -> ProfileAuthorityRootDigest {
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

    fn subject() -> SafetyProfileAuthorizationSubject {
        SafetyProfileAuthorizationSubject::new(
            "auth-compute-campus-g1",
            "facility-profile-root-v1",
            root_digest(0x33),
            "compute-campus",
            1,
            1_000,
            2_000,
            &profile(),
        )
        .unwrap()
    }

    #[test]
    fn constructor_binds_exact_profile_identity() {
        let profile = profile();
        let subject = SafetyProfileAuthorizationSubject::new(
            "auth-1",
            "root-1",
            root_digest(0x33),
            "node",
            1,
            1_000,
            2_000,
            &profile,
        )
        .unwrap();
        assert_eq!(subject.profile_id(), profile.profile_id);
        assert_eq!(subject.profile_digest(), profile.digest().unwrap());
    }

    #[test]
    fn root_subject_generation_validity_and_profile_are_all_bound() {
        let first = subject();

        let mut root_changed = first.clone();
        root_changed.authority_root_digest = root_digest(0x44);
        assert_ne!(
            first.canonical_signing_bytes().unwrap(),
            root_changed.canonical_signing_bytes().unwrap()
        );

        let mut node_changed = first.clone();
        node_changed.subject_node_id = "other-node".to_owned();
        assert_ne!(
            first.canonical_signing_bytes().unwrap(),
            node_changed.canonical_signing_bytes().unwrap()
        );

        let mut generation_changed = first.clone();
        generation_changed.generation = 2;
        assert_ne!(
            first.canonical_signing_bytes().unwrap(),
            generation_changed.canonical_signing_bytes().unwrap()
        );

        let mut validity_changed = first.clone();
        validity_changed.valid_until_unix_ms = 3_000;
        assert_ne!(
            first.canonical_signing_bytes().unwrap(),
            validity_changed.canonical_signing_bytes().unwrap()
        );

        let mut profile_changed = first.clone();
        profile_changed.profile_digest = config_digest(0xee);
        assert_ne!(
            first.canonical_signing_bytes().unwrap(),
            profile_changed.canonical_signing_bytes().unwrap()
        );
    }

    #[test]
    fn invalid_generation_and_validity_fail_closed() {
        let mut zero = subject();
        zero.generation = 0;
        assert_eq!(
            zero.validate(),
            Err(SafetyProfileAuthorizationError::ZeroGeneration)
        );

        let mut reversed = subject();
        reversed.valid_from_unix_ms = 2_000;
        reversed.valid_until_unix_ms = 2_000;
        assert_eq!(
            reversed.validate(),
            Err(SafetyProfileAuthorizationError::InvalidValidityWindow {
                valid_from_unix_ms: 2_000,
                valid_until_unix_ms: 2_000,
            })
        );
    }

    #[test]
    fn empty_authority_or_subject_identity_fails_closed() {
        let mut missing_root = subject();
        missing_root.authority_root_id = "  ".to_owned();
        assert_eq!(
            missing_root.validate(),
            Err(SafetyProfileAuthorizationError::EmptyAuthorityRootId)
        );

        let mut missing_subject = subject();
        missing_subject.subject_node_id.clear();
        assert_eq!(
            missing_subject.validate(),
            Err(SafetyProfileAuthorizationError::EmptySubjectNodeId)
        );
    }

    #[test]
    fn canonical_signing_bytes_have_pinned_cross_tool_v1_vector() {
        let pinned = SafetyProfileAuthorizationSubject {
            schema_version: SAFETY_PROFILE_AUTHORIZATION_SCHEMA_V1.to_owned(),
            authorization_id: "auth-1".to_owned(),
            authority_root_id: "root-1".to_owned(),
            authority_root_digest: root_digest(0x33),
            subject_node_id: "node".to_owned(),
            generation: 1,
            valid_from_unix_ms: 1_000,
            valid_until_unix_ms: 2_000,
            profile_id: "test-profile-v1".to_owned(),
            profile_digest: config_digest(0x22),
        };

        let expected = hex_bytes(
            "73796d74686165613a7361666574792d70726f66696c652d617574686f72697a6174696f6e3a7631000000002873796d74686165612d7361666574792d70726f66696c652d617574686f72697a6174696f6e2d763100000006617574682d3100000006726f6f742d31000000046e6f6465000000000000000100000000000003e800000000000007d00000000f746573742d70726f66696c652d7631013333333333333333333333333333333333333333333333333333333333333333012222222222222222222222222222222222222222222222222222222222222222",
        );
        assert_eq!(pinned.canonical_signing_bytes().unwrap(), expected);
    }

    fn hex_bytes(hex: &str) -> Vec<u8> {
        assert_eq!(hex.len() % 2, 0);
        hex.as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let hi = from_hex(pair[0]);
                let lo = from_hex(pair[1]);
                (hi << 4) | lo
            })
            .collect()
    }

    fn from_hex(byte: u8) -> u8 {
        match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            _ => panic!("invalid hex byte"),
        }
    }
}
