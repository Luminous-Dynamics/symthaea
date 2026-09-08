// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical signing subject for one exact commissioning action.
//!
//! Commissioning is a stronger power than authorizing a safety profile: it binds a
//! concrete configuration and a concrete local survival envelope into operational
//! state. This module therefore uses a distinct commissioning-authority root and
//! requires provenance through both a verifier-owned qualified configuration and
//! the exact profile-authorization transition that authorized that profile.
//!
//! This module defines raw evidence only. A syntactically valid subject is not
//! executable commissioning authority. Cryptographic verification, current-profile
//! admission, trusted-time admission, and atomic persistence remain separate gates.

use crate::identity::{
    CommissioningIdentityError, CommissioningRecordDigest, commissioning_record_digest,
};
use crate::{CommissioningRecord, ConfigurationDigest};
use serde::{Deserialize, Serialize};
use symthaea_resource_hierarchy::ResourceHierarchy;
use symthaea_safety_profile::transition::{
    SafetyProfileAuthorizationTransition, SafetyProfileAuthorizationTransitionDigest,
    SafetyProfileAuthorizationTransitionError,
};
use symthaea_safety_qualification::QualifiedSafetyConfiguration;
use thiserror::Error;

pub const COMMISSIONING_AUTHORIZATION_SCHEMA_V1: &str =
    "symthaea-commissioning-authorization-v1";
const DOMAIN_SEPARATOR: &[u8] = b"symthaea:commissioning-authorization:v1\0";

/// Fingerprint of the separately provisioned commissioning-authority verifier key.
///
/// V1 uses the same byte-level fingerprint convention as the profile-authority
/// contract -- BLAKE3-256 of the raw verifier public-key bytes -- while retaining a
/// distinct Rust type so profile authority cannot be passed where commissioning
/// authority is required by accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommissioningAuthorityRootDigest {
    Blake3_256([u8; 32]),
}

impl CommissioningAuthorityRootDigest {
    pub fn from_raw_verifier_key(verifier_key_bytes: &[u8]) -> Self {
        Self::Blake3_256(
            ConfigurationDigest::blake3_256(verifier_key_bytes).into_blake3_256(),
        )
    }

    pub fn into_blake3_256(self) -> [u8; 32] {
        match self {
            Self::Blake3_256(bytes) => bytes,
        }
    }
}

/// Immutable message to be authenticated by the commissioning authority.
///
/// Every provenance field after the commissioning-root identity is derived from
/// the three supplied evidence/proof objects. Callers cannot independently choose
/// a node, configuration, profile, generation, or profile-authorization lineage
/// and accidentally sign an internally inconsistent combination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommissioningAuthorizationSubject {
    schema_version: String,
    authorization_id: String,
    authority_root_id: String,
    authority_root_digest: CommissioningAuthorityRootDigest,
    subject_node_id: String,
    commissioning_generation: u64,
    valid_from_unix_ms: i64,
    valid_until_unix_ms: i64,
    commissioning_record_digest: CommissioningRecordDigest,
    configuration_digest: ConfigurationDigest,
    profile_id: String,
    profile_digest: ConfigurationDigest,
    profile_authorization_generation: u64,
    profile_authorization_transition_digest: SafetyProfileAuthorizationTransitionDigest,
}

impl CommissioningAuthorizationSubject {
    /// Construct one commissioning-authorization subject from proof-carrying
    /// qualification, exact profile lineage, and the complete commissioning record.
    ///
    /// The validity window bounds the *commissioning action*. It does not add an
    /// expiry to the locally commissioned survival envelope after a successful
    /// commissioning commit.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        authorization_id: impl Into<String>,
        authority_root_id: impl Into<String>,
        authority_root_digest: CommissioningAuthorityRootDigest,
        valid_from_unix_ms: i64,
        valid_until_unix_ms: i64,
        record: &CommissioningRecord,
        hierarchy: &ResourceHierarchy,
        qualified: &QualifiedSafetyConfiguration,
        profile_authorization: &SafetyProfileAuthorizationTransition,
    ) -> Result<Self, CommissioningAuthorizationError> {
        profile_authorization.validate()?;
        let record_digest = commissioning_record_digest(record, hierarchy)?;
        let profile_subject = profile_authorization.subject();
        let record_configuration = record.commissioned().binding().configuration_digest();

        if record.subject_node_id() != qualified.node_id() {
            return Err(CommissioningAuthorizationError::RecordQualificationNodeMismatch {
                record: record.subject_node_id().to_owned(),
                qualified: qualified.node_id().to_owned(),
            });
        }
        if record_configuration != qualified.configuration_digest() {
            return Err(
                CommissioningAuthorizationError::RecordQualificationConfigurationMismatch {
                    record: record_configuration,
                    qualified: qualified.configuration_digest(),
                },
            );
        }
        if profile_subject.subject_node_id() != qualified.node_id() {
            return Err(
                CommissioningAuthorizationError::ProfileAuthorizationNodeMismatch {
                    authorized: profile_subject.subject_node_id().to_owned(),
                    qualified: qualified.node_id().to_owned(),
                },
            );
        }
        if profile_subject.profile_id() != qualified.profile_id() {
            return Err(
                CommissioningAuthorizationError::ProfileAuthorizationIdMismatch {
                    authorized: profile_subject.profile_id().to_owned(),
                    qualified: qualified.profile_id().to_owned(),
                },
            );
        }
        if profile_subject.profile_digest() != qualified.profile_digest() {
            return Err(CommissioningAuthorizationError::ProfileAuthorizationDigestMismatch);
        }

        let subject = Self {
            schema_version: COMMISSIONING_AUTHORIZATION_SCHEMA_V1.to_owned(),
            authorization_id: authorization_id.into(),
            authority_root_id: authority_root_id.into(),
            authority_root_digest,
            subject_node_id: qualified.node_id().to_owned(),
            commissioning_generation: record.generation(),
            valid_from_unix_ms,
            valid_until_unix_ms,
            commissioning_record_digest: record_digest,
            configuration_digest: qualified.configuration_digest(),
            profile_id: qualified.profile_id().to_owned(),
            profile_digest: qualified.profile_digest(),
            profile_authorization_generation: profile_authorization.generation(),
            profile_authorization_transition_digest: profile_authorization.transition_digest()?,
        };
        subject.validate()?;
        Ok(subject)
    }

    pub fn validate(&self) -> Result<(), CommissioningAuthorizationError> {
        if self.schema_version != COMMISSIONING_AUTHORIZATION_SCHEMA_V1 {
            return Err(CommissioningAuthorizationError::UnsupportedSchemaVersion(
                self.schema_version.clone(),
            ));
        }
        if self.authorization_id.trim().is_empty() {
            return Err(CommissioningAuthorizationError::EmptyAuthorizationId);
        }
        if self.authority_root_id.trim().is_empty() {
            return Err(CommissioningAuthorizationError::EmptyAuthorityRootId);
        }
        if self.subject_node_id.trim().is_empty() {
            return Err(CommissioningAuthorizationError::EmptySubjectNodeId);
        }
        if self.commissioning_generation == 0 {
            return Err(CommissioningAuthorizationError::ZeroCommissioningGeneration);
        }
        if self.profile_authorization_generation == 0 {
            return Err(CommissioningAuthorizationError::ZeroProfileAuthorizationGeneration);
        }
        if self.profile_id.trim().is_empty() {
            return Err(CommissioningAuthorizationError::EmptyProfileId);
        }
        if self.valid_from_unix_ms >= self.valid_until_unix_ms {
            return Err(CommissioningAuthorizationError::InvalidValidityWindow {
                valid_from_unix_ms: self.valid_from_unix_ms,
                valid_until_unix_ms: self.valid_until_unix_ms,
            });
        }
        Ok(())
    }

    /// Fixed-order, domain-separated bytes for external authentication.
    ///
    /// V1 binds the complete canonical commissioning-record digest and repeats the
    /// key provenance digests/generations explicitly for cross-tool auditability.
    /// Redundant fields are constructor-derived and therefore cannot disagree in
    /// the normal construction path.
    pub fn canonical_signing_bytes(&self) -> Result<Vec<u8>, CommissioningAuthorizationError> {
        self.validate()?;
        let mut out = Vec::with_capacity(320);
        out.extend_from_slice(DOMAIN_SEPARATOR);
        push_string(&mut out, "schema_version", &self.schema_version)?;
        push_string(&mut out, "authorization_id", &self.authorization_id)?;
        push_string(&mut out, "authority_root_id", &self.authority_root_id)?;
        push_string(&mut out, "subject_node_id", &self.subject_node_id)?;
        out.extend_from_slice(&self.commissioning_generation.to_be_bytes());
        out.extend_from_slice(&self.valid_from_unix_ms.to_be_bytes());
        out.extend_from_slice(&self.valid_until_unix_ms.to_be_bytes());
        push_commissioning_root_digest(&mut out, self.authority_root_digest);
        push_commissioning_record_digest(&mut out, self.commissioning_record_digest);
        push_configuration_digest(&mut out, self.configuration_digest);
        push_string(&mut out, "profile_id", &self.profile_id)?;
        push_configuration_digest(&mut out, self.profile_digest);
        out.extend_from_slice(&self.profile_authorization_generation.to_be_bytes());
        push_profile_transition_digest(&mut out, self.profile_authorization_transition_digest);
        Ok(out)
    }

    pub fn authorization_id(&self) -> &str {
        &self.authorization_id
    }

    pub fn authority_root_id(&self) -> &str {
        &self.authority_root_id
    }

    pub fn authority_root_digest(&self) -> CommissioningAuthorityRootDigest {
        self.authority_root_digest
    }

    pub fn subject_node_id(&self) -> &str {
        &self.subject_node_id
    }

    pub fn commissioning_generation(&self) -> u64 {
        self.commissioning_generation
    }

    pub fn valid_from_unix_ms(&self) -> i64 {
        self.valid_from_unix_ms
    }

    pub fn valid_until_unix_ms(&self) -> i64 {
        self.valid_until_unix_ms
    }

    pub fn commissioning_record_digest(&self) -> CommissioningRecordDigest {
        self.commissioning_record_digest
    }

    pub fn configuration_digest(&self) -> ConfigurationDigest {
        self.configuration_digest
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn profile_digest(&self) -> ConfigurationDigest {
        self.profile_digest
    }

    pub fn profile_authorization_generation(&self) -> u64 {
        self.profile_authorization_generation
    }

    pub fn profile_authorization_transition_digest(&self) -> SafetyProfileAuthorizationTransitionDigest {
        self.profile_authorization_transition_digest
    }
}

fn push_string(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &str,
) -> Result<(), CommissioningAuthorizationError> {
    let len = u32::try_from(value.len())
        .map_err(|_| CommissioningAuthorizationError::StringTooLong(field))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_commissioning_root_digest(out: &mut Vec<u8>, digest: CommissioningAuthorityRootDigest) {
    match digest {
        CommissioningAuthorityRootDigest::Blake3_256(bytes) => {
            out.push(1);
            out.extend_from_slice(&bytes);
        }
    }
}

fn push_commissioning_record_digest(out: &mut Vec<u8>, digest: CommissioningRecordDigest) {
    match digest {
        CommissioningRecordDigest::Blake3_256(bytes) => {
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

fn push_profile_transition_digest(
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

#[derive(Debug, Clone, PartialEq, Error)]
pub enum CommissioningAuthorizationError {
    #[error(transparent)]
    Identity(#[from] CommissioningIdentityError),
    #[error(transparent)]
    ProfileAuthorization(#[from] SafetyProfileAuthorizationTransitionError),
    #[error("unsupported commissioning-authorization schema version {0}")]
    UnsupportedSchemaVersion(String),
    #[error("commissioning authorization id must not be empty")]
    EmptyAuthorizationId,
    #[error("commissioning-authority root id must not be empty")]
    EmptyAuthorityRootId,
    #[error("commissioning authorization subject node id must not be empty")]
    EmptySubjectNodeId,
    #[error("commissioning generation must be greater than zero")]
    ZeroCommissioningGeneration,
    #[error("profile-authorization generation must be greater than zero")]
    ZeroProfileAuthorizationGeneration,
    #[error("commissioning authorization profile id must not be empty")]
    EmptyProfileId,
    #[error("invalid commissioning authorization validity window: {valid_from_unix_ms} >= {valid_until_unix_ms}")]
    InvalidValidityWindow {
        valid_from_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
    #[error("commissioning record node {record} does not match qualified configuration node {qualified}")]
    RecordQualificationNodeMismatch { record: String, qualified: String },
    #[error("commissioning record configuration does not match qualified configuration")]
    RecordQualificationConfigurationMismatch {
        record: ConfigurationDigest,
        qualified: ConfigurationDigest,
    },
    #[error("profile authorization node {authorized} does not match qualified configuration node {qualified}")]
    ProfileAuthorizationNodeMismatch { authorized: String, qualified: String },
    #[error("profile authorization id {authorized} does not match qualified profile id {qualified}")]
    ProfileAuthorizationIdMismatch { authorized: String, qualified: String },
    #[error("profile authorization digest does not match qualified profile digest")]
    ProfileAuthorizationDigestMismatch,
    #[error("commissioning authorization string field {0} exceeds canonical u32 length")]
    StringTooLong(&'static str),
}
