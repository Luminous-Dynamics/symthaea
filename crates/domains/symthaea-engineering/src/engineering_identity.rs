// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical identity for recursively composed engineering objects.
//!
//! Identity is deliberately separate from provenance, currentness, applicability,
//! epistemic confidence, and operational authority. The digest carried by an
//! EngineeringObjectId is a content identity supplied by the owning system;
//! this module does not infer physical execution or qualification from it.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

pub const ENGINEERING_OBJECT_ID_SCHEMA: &str = "symthaea.engineering-object-id.v1";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EngineeringObjectId {
    pub namespace: String,
    pub object_kind: String,
    pub canonical_identifier: String,
    pub version: String,
    pub content_digest: String,
}

impl EngineeringObjectId {
    pub fn new(
        namespace: impl Into<String>,
        object_kind: impl Into<String>,
        canonical_identifier: impl Into<String>,
        version: impl Into<String>,
        content_digest: impl Into<String>,
    ) -> Result<Self, IdentityError> {
        let id = Self {
            namespace: namespace.into(),
            object_kind: object_kind.into(),
            canonical_identifier: canonical_identifier.into(),
            version: version.into(),
            content_digest: content_digest.into(),
        };
        id.validate()?;
        Ok(id)
    }

    pub fn validate(&self) -> Result<(), IdentityError> {
        for (field, value) in [
            ("namespace", self.namespace.as_str()),
            ("object_kind", self.object_kind.as_str()),
            ("canonical_identifier", self.canonical_identifier.as_str()),
            ("version", self.version.as_str()),
        ] {
            if value.is_empty() || value.trim() != value {
                return Err(IdentityError::InvalidField(field));
            }
        }

        if self.content_digest.len() != 64
            || !self.content_digest.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(IdentityError::InvalidDigest);
        }

        Ok(())
    }

    /// Canonical, length-delimited representation suitable for hashing or storage.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        fn push_field(out: &mut Vec<u8>, value: &str) {
            out.extend_from_slice(value.len().to_string().as_bytes());
            out.push(b':');
            out.extend_from_slice(value.as_bytes());
        }

        let mut out = Vec::new();
        push_field(&mut out, ENGINEERING_OBJECT_ID_SCHEMA);
        push_field(&mut out, &self.namespace);
        push_field(&mut out, &self.object_kind);
        push_field(&mut out, &self.canonical_identifier);
        push_field(&mut out, &self.version);
        push_field(&mut out, &self.content_digest);
        out
    }

    pub fn identity_digest(&self) -> String {
        let digest = Sha256::digest(self.canonical_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl fmt::Display for EngineeringObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}@{}#{}",
            self.namespace,
            self.object_kind,
            self.canonical_identifier,
            self.version,
            self.content_digest
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityError {
    InvalidField(&'static str),
    InvalidDigest,
}

impl fmt::Display for IdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidField(field) => write!(f, "invalid engineering identity field: {field}"),
            Self::InvalidDigest => write!(f, "content_digest must be exactly 64 hexadecimal characters"),
        }
    }
}

impl std::error::Error for IdentityError {}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn sample() -> EngineeringObjectId {
        EngineeringObjectId::new("mycelix", "runtime", "runtime-build", "1", DIGEST).unwrap()
    }

    #[test]
    fn canonical_identity_is_deterministic() {
        assert_eq!(sample().canonical_bytes(), sample().canonical_bytes());
        assert_eq!(sample().identity_digest(), sample().identity_digest());
    }

    #[test]
    fn one_tuple_component_changes_identity_digest() {
        let base = sample();
        let changed =
            EngineeringObjectId::new("mycelix", "runtime", "runtime-build", "2", DIGEST).unwrap();
        assert_ne!(base.identity_digest(), changed.identity_digest());
    }

    #[test]
    fn content_digest_is_not_recomputed_or_promoted() {
        let id = sample();
        assert_eq!(id.content_digest, DIGEST);
        assert_ne!(id.content_digest, id.identity_digest());
    }

    #[test]
    fn malformed_digest_is_rejected() {
        let err = EngineeringObjectId::new("mycelix", "runtime", "x", "1", "abc").unwrap_err();
        assert_eq!(err, IdentityError::InvalidDigest);
    }

    #[test]
    fn whitespace_and_empty_fields_are_rejected() {
        assert_eq!(
            EngineeringObjectId::new("mycelix", "runtime", " x", "1", DIGEST).unwrap_err(),
            IdentityError::InvalidField("canonical_identifier")
        );
        assert_eq!(
            EngineeringObjectId::new("mycelix", "runtime", "", "1", DIGEST).unwrap_err(),
            IdentityError::InvalidField("canonical_identifier")
        );
    }

    #[test]
    fn serde_round_trip_preserves_identity() {
        let id = sample();
        let json = serde_json::to_string(&id).unwrap();
        let restored: EngineeringObjectId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }
}
