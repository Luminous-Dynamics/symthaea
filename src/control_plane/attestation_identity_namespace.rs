// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical identity-namespace semantics for attestation authority mappings.
//!
//! Bare identity strings are not globally meaningful. This module makes the
//! namespace and interpretation of principal/authority identifiers explicit and
//! representation-independent before they are used in diversity decisions.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

pub const ATTESTATION_IDENTITY_NAMESPACE_VERSION_V1: u16 = 1;
pub const MAX_ATTESTATION_IDENTITY_NAMESPACE_BYTES_V1: usize = 128;
pub const MAX_ATTESTATION_QUALIFIED_ID_BYTES_V1: usize = 256;

pub const ATTESTATION_IDENTITY_NAMESPACE_DESCRIPTOR_DOMAIN_V1: &[u8] =
    b"symthaea.control-plane.attestation-identity-namespace-descriptor.v1\0";
pub const ATTESTATION_IDENTITY_NAMESPACE_SET_DOMAIN_V1: &[u8] =
    b"symthaea.control-plane.attestation-identity-namespace-set.v1\0";
pub const ATTESTATION_QUALIFIED_IDENTITY_DOMAIN_V1: &[u8] =
    b"symthaea.control-plane.attestation-qualified-identity.v1\0";

/// Stable semantic kind of an attestation identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AttestationIdentityKindV1 {
    Principal,
    Authority,
}

impl AttestationIdentityKindV1 {
    pub const fn canonical_tag(self) -> u8 {
        match self {
            Self::Principal => 1,
            Self::Authority => 2,
        }
    }
}

/// Exact namespace and semantic interpretation for one class of identity IDs.
///
/// `namespace_id` is human/operationally useful. `semantics_commitment` binds the
/// exact canonicalization, provenance, aliasing and interpretation rules behind
/// that namespace. Reusing the friendly ID with changed semantics must therefore
/// produce a different namespace commitment.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttestationIdentityNamespaceV1 {
    pub namespace_id: String,
    pub semantics_commitment: [u8; 32],
}

impl AttestationIdentityNamespaceV1 {
    pub fn new(
        namespace_id: impl Into<String>,
        semantics_commitment: [u8; 32],
    ) -> Result<Self, AttestationIdentityNamespaceErrorV1> {
        let namespace = Self {
            namespace_id: namespace_id.into(),
            semantics_commitment,
        };
        namespace.validate()?;
        Ok(namespace)
    }

    pub fn validate(&self) -> Result<(), AttestationIdentityNamespaceErrorV1> {
        validate_bounded_nonempty(
            "namespace_id",
            &self.namespace_id,
            MAX_ATTESTATION_IDENTITY_NAMESPACE_BYTES_V1,
        )?;
        if self.semantics_commitment == [0; 32] {
            return Err(AttestationIdentityNamespaceErrorV1::ZeroCommitment(
                "semantics_commitment",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes_v1(
        &self,
        kind: AttestationIdentityKindV1,
    ) -> Result<Vec<u8>, AttestationIdentityNamespaceErrorV1> {
        self.validate()?;
        let mut out = Vec::with_capacity(
            ATTESTATION_IDENTITY_NAMESPACE_DESCRIPTOR_DOMAIN_V1.len()
                + 1
                + 2
                + MAX_ATTESTATION_IDENTITY_NAMESPACE_BYTES_V1
                + 32,
        );
        out.extend_from_slice(ATTESTATION_IDENTITY_NAMESPACE_DESCRIPTOR_DOMAIN_V1);
        out.push(kind.canonical_tag());
        append_bounded_utf8(
            &mut out,
            "namespace_id",
            &self.namespace_id,
            MAX_ATTESTATION_IDENTITY_NAMESPACE_BYTES_V1,
        )?;
        out.extend_from_slice(&self.semantics_commitment);
        Ok(out)
    }

    pub fn canonical_commitment_v1(
        &self,
        kind: AttestationIdentityKindV1,
    ) -> Result<[u8; 32], AttestationIdentityNamespaceErrorV1> {
        Ok(*blake3::hash(&self.canonical_bytes_v1(kind)?).as_bytes())
    }
}

/// Namespace set selected for one authority-binding resolver profile.
///
/// Principal identity is mandatory. Authority identity is optional because some
/// deployments intentionally qualify principal diversity only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttestationIdentityNamespaceSetV1 {
    pub protocol_version: u16,
    pub principal: AttestationIdentityNamespaceV1,
    pub authority: Option<AttestationIdentityNamespaceV1>,
}

impl AttestationIdentityNamespaceSetV1 {
    pub fn new(
        principal: AttestationIdentityNamespaceV1,
        authority: Option<AttestationIdentityNamespaceV1>,
    ) -> Result<Self, AttestationIdentityNamespaceErrorV1> {
        let set = Self {
            protocol_version: ATTESTATION_IDENTITY_NAMESPACE_VERSION_V1,
            principal,
            authority,
        };
        set.validate()?;
        Ok(set)
    }

    pub fn validate(&self) -> Result<(), AttestationIdentityNamespaceErrorV1> {
        if self.protocol_version != ATTESTATION_IDENTITY_NAMESPACE_VERSION_V1 {
            return Err(
                AttestationIdentityNamespaceErrorV1::UnsupportedProtocolVersion(
                    self.protocol_version,
                ),
            );
        }
        self.principal.validate()?;
        if let Some(authority) = &self.authority {
            authority.validate()?;
        }
        Ok(())
    }

    pub fn canonical_bytes_v1(&self) -> Result<Vec<u8>, AttestationIdentityNamespaceErrorV1> {
        self.validate()?;
        let principal_commitment = self
            .principal
            .canonical_commitment_v1(AttestationIdentityKindV1::Principal)?;
        let authority_commitment = self
            .authority
            .as_ref()
            .map(|namespace| {
                namespace.canonical_commitment_v1(AttestationIdentityKindV1::Authority)
            })
            .transpose()?;

        let mut out = Vec::with_capacity(
            ATTESTATION_IDENTITY_NAMESPACE_SET_DOMAIN_V1.len() + 2 + 32 + 1 + 32,
        );
        out.extend_from_slice(ATTESTATION_IDENTITY_NAMESPACE_SET_DOMAIN_V1);
        out.extend_from_slice(&self.protocol_version.to_be_bytes());
        out.extend_from_slice(&principal_commitment);
        match authority_commitment {
            Some(commitment) => {
                out.push(1);
                out.extend_from_slice(&commitment);
            }
            None => out.push(0),
        }
        Ok(out)
    }

    pub fn canonical_commitment_v1(
        &self,
    ) -> Result<[u8; 32], AttestationIdentityNamespaceErrorV1> {
        Ok(*blake3::hash(&self.canonical_bytes_v1()?).as_bytes())
    }

    pub fn qualify_principal(
        &self,
        principal_id: impl Into<String>,
    ) -> Result<QualifiedAttestationIdentityV1, AttestationIdentityNamespaceErrorV1> {
        self.validate()?;
        QualifiedAttestationIdentityV1::new(
            AttestationIdentityKindV1::Principal,
            self.principal
                .canonical_commitment_v1(AttestationIdentityKindV1::Principal)?,
            principal_id,
        )
    }

    pub fn qualify_authority(
        &self,
        authority_id: impl Into<String>,
    ) -> Result<QualifiedAttestationIdentityV1, AttestationIdentityNamespaceErrorV1> {
        self.validate()?;
        let authority = self
            .authority
            .as_ref()
            .ok_or(AttestationIdentityNamespaceErrorV1::AuthorityNamespaceUnavailable)?;
        QualifiedAttestationIdentityV1::new(
            AttestationIdentityKindV1::Authority,
            authority.canonical_commitment_v1(AttestationIdentityKindV1::Authority)?,
            authority_id,
        )
    }
}

/// Exact namespace-qualified identifier suitable for equality/distinctness tests.
///
/// This is identity metadata, not proof of human/legal/organizational independence.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedAttestationIdentityV1 {
    pub kind: AttestationIdentityKindV1,
    pub namespace_commitment: [u8; 32],
    pub id: String,
}

impl QualifiedAttestationIdentityV1 {
    pub fn new(
        kind: AttestationIdentityKindV1,
        namespace_commitment: [u8; 32],
        id: impl Into<String>,
    ) -> Result<Self, AttestationIdentityNamespaceErrorV1> {
        let identity = Self {
            kind,
            namespace_commitment,
            id: id.into(),
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(&self) -> Result<(), AttestationIdentityNamespaceErrorV1> {
        if self.namespace_commitment == [0; 32] {
            return Err(AttestationIdentityNamespaceErrorV1::ZeroCommitment(
                "namespace_commitment",
            ));
        }
        validate_bounded_nonempty(
            "qualified_identity.id",
            &self.id,
            MAX_ATTESTATION_QUALIFIED_ID_BYTES_V1,
        )
    }

    pub fn canonical_bytes_v1(&self) -> Result<Vec<u8>, AttestationIdentityNamespaceErrorV1> {
        self.validate()?;
        let mut out = Vec::with_capacity(
            ATTESTATION_QUALIFIED_IDENTITY_DOMAIN_V1.len()
                + 1
                + 32
                + 2
                + MAX_ATTESTATION_QUALIFIED_ID_BYTES_V1,
        );
        out.extend_from_slice(ATTESTATION_QUALIFIED_IDENTITY_DOMAIN_V1);
        out.push(self.kind.canonical_tag());
        out.extend_from_slice(&self.namespace_commitment);
        append_bounded_utf8(
            &mut out,
            "qualified_identity.id",
            &self.id,
            MAX_ATTESTATION_QUALIFIED_ID_BYTES_V1,
        )?;
        Ok(out)
    }

    pub fn canonical_commitment_v1(
        &self,
    ) -> Result<[u8; 32], AttestationIdentityNamespaceErrorV1> {
        Ok(*blake3::hash(&self.canonical_bytes_v1()?).as_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationIdentityNamespaceErrorV1 {
    EmptyField(&'static str),
    FieldTooLong {
        field: &'static str,
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    ZeroCommitment(&'static str),
    UnsupportedProtocolVersion(u16),
    AuthorityNamespaceUnavailable,
}

impl fmt::Display for AttestationIdentityNamespaceErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must be non-empty"),
            Self::FieldTooLong {
                field,
                actual_bytes,
                maximum_bytes,
            } => write!(
                f,
                "{field} is {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::ZeroCommitment(field) => write!(f, "{field} must not be the zero commitment"),
            Self::UnsupportedProtocolVersion(version) => {
                write!(f, "unsupported attestation identity namespace version {version}")
            }
            Self::AuthorityNamespaceUnavailable => {
                write!(f, "authority identity namespace is not configured")
            }
        }
    }
}

impl Error for AttestationIdentityNamespaceErrorV1 {}

fn validate_bounded_nonempty(
    field: &'static str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), AttestationIdentityNamespaceErrorV1> {
    if value.is_empty() {
        return Err(AttestationIdentityNamespaceErrorV1::EmptyField(field));
    }
    if value.len() > maximum_bytes {
        return Err(AttestationIdentityNamespaceErrorV1::FieldTooLong {
            field,
            actual_bytes: value.len(),
            maximum_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn principal_namespace() -> AttestationIdentityNamespaceV1 {
        AttestationIdentityNamespaceV1::new("example-principals-v1", [0x11; 32]).unwrap()
    }

    fn namespaces() -> AttestationIdentityNamespaceSetV1 {
        AttestationIdentityNamespaceSetV1::new(
            principal_namespace(),
            Some(
                AttestationIdentityNamespaceV1::new("example-authorities-v1", [0x22; 32])
                    .unwrap(),
            ),
        )
        .unwrap()
    }

    #[test]
    fn same_text_in_different_namespaces_is_not_same_identity() {
        let left = AttestationIdentityNamespaceSetV1::new(
            AttestationIdentityNamespaceV1::new("resolver-a", [0x31; 32]).unwrap(),
            None,
        )
        .unwrap()
        .qualify_principal("alice")
        .unwrap();
        let right = AttestationIdentityNamespaceSetV1::new(
            AttestationIdentityNamespaceV1::new("resolver-b", [0x32; 32]).unwrap(),
            None,
        )
        .unwrap()
        .qualify_principal("alice")
        .unwrap();

        assert_ne!(left, right);
        assert_ne!(
            left.canonical_commitment_v1().unwrap(),
            right.canonical_commitment_v1().unwrap()
        );
    }

    #[test]
    fn same_namespace_name_with_different_semantics_is_not_same_identity() {
        let left = AttestationIdentityNamespaceSetV1::new(
            AttestationIdentityNamespaceV1::new("resolver", [0x41; 32]).unwrap(),
            None,
        )
        .unwrap()
        .qualify_principal("alice")
        .unwrap();
        let right = AttestationIdentityNamespaceSetV1::new(
            AttestationIdentityNamespaceV1::new("resolver", [0x42; 32]).unwrap(),
            None,
        )
        .unwrap()
        .qualify_principal("alice")
        .unwrap();

        assert_ne!(left, right);
    }

    #[test]
    fn principal_and_authority_kinds_do_not_alias() {
        let shared_namespace = AttestationIdentityNamespaceV1::new("shared", [0x51; 32]).unwrap();
        let set = AttestationIdentityNamespaceSetV1::new(
            shared_namespace.clone(),
            Some(shared_namespace),
        )
        .unwrap();

        let principal = set.qualify_principal("same-id").unwrap();
        let authority = set.qualify_authority("same-id").unwrap();
        assert_ne!(principal, authority);
        assert_ne!(principal.namespace_commitment, authority.namespace_commitment);
    }

    #[test]
    fn authority_qualification_fails_without_authority_namespace() {
        let set = AttestationIdentityNamespaceSetV1::new(principal_namespace(), None).unwrap();
        assert!(matches!(
            set.qualify_authority("org-1"),
            Err(AttestationIdentityNamespaceErrorV1::AuthorityNamespaceUnavailable)
        ));
    }

    #[test]
    fn changing_namespace_semantics_changes_set_commitment() {
        let first = namespaces();
        let mut second = first.clone();
        second.authority.as_mut().unwrap().semantics_commitment[0] ^= 0xFF;
        assert_ne!(
            first.canonical_commitment_v1().unwrap(),
            second.canonical_commitment_v1().unwrap()
        );
    }

    #[test]
    fn zero_semantics_commitment_is_rejected() {
        assert!(matches!(
            AttestationIdentityNamespaceV1::new("resolver", [0; 32]),
            Err(AttestationIdentityNamespaceErrorV1::ZeroCommitment(
                "semantics_commitment"
            ))
        ));
    }

    #[test]
    fn wrong_protocol_version_is_rejected() {
        let mut set = namespaces();
        set.protocol_version += 1;
        assert!(matches!(
            set.validate(),
            Err(AttestationIdentityNamespaceErrorV1::UnsupportedProtocolVersion(_))
        ));
    }
}
