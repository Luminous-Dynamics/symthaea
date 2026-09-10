// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed scope and stable subject identity for computing continuity.
//!
//! Enterprise continuity is not always machine-scoped. A transition can concern
//! one service, a quorum-bearing cluster, a storage set, a network device, an
//! entire fabric, a site, or a fleet. This module gives those subjects one small,
//! deterministic identity vocabulary without granting any migration, verification,
//! or execution authority.
//!
//! Core theorem:
//!
//! `ContinuitySubjectV1 != ContinuityContract != QualifiedWitness != ExecutionAuthority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Stable schema for one continuity subject identity.
pub const CONTINUITY_SUBJECT_SCHEMA_V1: &str = "symthaea-continuity-subject-v1";

const SUBJECT_DOMAIN: &[u8] = b"symthaea.continuity.subject.v1\0";
const MAX_TEXT_BYTES: usize = 1024;

/// Stable content identity of one continuity subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContinuitySubjectId([u8; 32]);

impl ContinuitySubjectId {
    /// Raw BLAKE3-256 identity bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// The operational scope whose continuity is being described.
///
/// The enum is intentionally small. Product- or vendor-specific semantics belong
/// in observations, requirements, verifier profiles, and adapters rather than in
/// this trust-neutral identity layer. `Custom` permits a namespaced extension
/// without pretending that the core schema understands its semantics.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ContinuityScopeV1 {
    Machine,
    Service,
    Cluster,
    StorageSet,
    NetworkDevice,
    NetworkFabric,
    Site,
    Fleet,
    ManagedService,
    ExternalDependency,
    Custom { kind_id: String },
}

impl ContinuityScopeV1 {
    /// Construct a validated custom scope kind.
    pub fn custom(kind_id: impl Into<String>) -> Result<Self, ContinuitySubjectError> {
        Ok(Self::Custom {
            kind_id: checked_text("custom scope kind_id", kind_id.into())?,
        })
    }

    fn normalize(self) -> Result<Self, ContinuitySubjectError> {
        match self {
            Self::Custom { kind_id } => Self::custom(kind_id),
            other => Ok(other),
        }
    }

    fn validate(&self) -> Result<(), ContinuitySubjectError> {
        if let Self::Custom { kind_id } = self {
            let canonical = checked_text("custom scope kind_id", kind_id.clone())?;
            if canonical != *kind_id {
                return Err(ContinuitySubjectError::NonCanonicalText {
                    field: "custom scope kind_id",
                });
            }
        }
        Ok(())
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Machine => out.push(1),
            Self::Service => out.push(2),
            Self::Cluster => out.push(3),
            Self::StorageSet => out.push(4),
            Self::NetworkDevice => out.push(5),
            Self::NetworkFabric => out.push(6),
            Self::Site => out.push(7),
            Self::Fleet => out.push(8),
            Self::ManagedService => out.push(9),
            Self::ExternalDependency => out.push(10),
            Self::Custom { kind_id } => {
                out.push(255);
                put_str(out, kind_id);
            }
        }
    }
}

/// Stable, serializable identity for the thing whose continuity is at stake.
///
/// `namespace` and `logical_id` are identifiers only. They do not establish
/// ownership, authentication, trust, or authorization. `parent_subject_id` is a
/// containment/context edge, not an authority edge; for example, a machine can be
/// scoped beneath a cluster or a network device beneath a fabric without implying
/// that either subject may authorize the other.
///
/// The stored `subject_id` is always revalidated against the canonical fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuitySubjectV1 {
    schema_version: String,
    namespace: String,
    logical_id: String,
    scope: ContinuityScopeV1,
    parent_subject_id: Option<ContinuitySubjectId>,
    subject_id: ContinuitySubjectId,
}

impl ContinuitySubjectV1 {
    /// Construct one canonical continuity subject.
    pub fn new(
        namespace: impl Into<String>,
        logical_id: impl Into<String>,
        scope: ContinuityScopeV1,
        parent_subject_id: Option<ContinuitySubjectId>,
    ) -> Result<Self, ContinuitySubjectError> {
        let namespace = checked_text("subject namespace", namespace.into())?;
        let logical_id = checked_text("subject logical_id", logical_id.into())?;
        let scope = scope.normalize()?;
        let subject_id = ContinuitySubjectId(hash_subject_fields(
            &namespace,
            &logical_id,
            &scope,
            parent_subject_id,
        ));

        Ok(Self {
            schema_version: CONTINUITY_SUBJECT_SCHEMA_V1.to_owned(),
            namespace,
            logical_id,
            scope,
            parent_subject_id,
            subject_id,
        })
    }

    /// Revalidate schema, canonical text, scope, and stored content identity.
    ///
    /// Validation establishes only structural/content identity. It does not prove
    /// that this subject exists, is currently observed, is owned by the caller, or
    /// is authorized for any transition.
    pub fn validate(&self) -> Result<(), ContinuitySubjectError> {
        if self.schema_version != CONTINUITY_SUBJECT_SCHEMA_V1 {
            return Err(ContinuitySubjectError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }

        let namespace = checked_text("subject namespace", self.namespace.clone())?;
        if namespace != self.namespace {
            return Err(ContinuitySubjectError::NonCanonicalText {
                field: "subject namespace",
            });
        }

        let logical_id = checked_text("subject logical_id", self.logical_id.clone())?;
        if logical_id != self.logical_id {
            return Err(ContinuitySubjectError::NonCanonicalText {
                field: "subject logical_id",
            });
        }

        self.scope.validate()?;

        let expected = ContinuitySubjectId(hash_subject_fields(
            &self.namespace,
            &self.logical_id,
            &self.scope,
            self.parent_subject_id,
        ));
        if expected != self.subject_id {
            return Err(ContinuitySubjectError::SubjectIdentityMismatch);
        }
        Ok(())
    }

    /// Exact subject identity.
    pub fn id(&self) -> ContinuitySubjectId {
        self.subject_id
    }

    /// Administrative/application namespace used only for stable naming.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Stable logical identifier within the namespace.
    pub fn logical_id(&self) -> &str {
        &self.logical_id
    }

    /// Typed continuity scope.
    pub fn scope(&self) -> &ContinuityScopeV1 {
        &self.scope
    }

    /// Optional containment/context parent. This is not an authority relation.
    pub fn parent_subject_id(&self) -> Option<ContinuitySubjectId> {
        self.parent_subject_id
    }

    /// Canonical identity bytes before domain-separated hashing.
    ///
    /// These bytes are useful for independent implementations and deterministic
    /// fixtures. They are not authentication payloads and confer no authority.
    pub fn canonical_identity_bytes(&self) -> Result<Vec<u8>, ContinuitySubjectError> {
        self.validate()?;
        Ok(encode_subject_fields(
            &self.namespace,
            &self.logical_id,
            &self.scope,
            self.parent_subject_id,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ContinuitySubjectError {
    #[error("unsupported continuity subject schema: {0}")]
    UnsupportedSchema(String),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds {MAX_TEXT_BYTES} bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("{field} is not canonically trimmed")]
    NonCanonicalText { field: &'static str },
    #[error("stored continuity subject identity does not match canonical fields")]
    SubjectIdentityMismatch,
}

fn checked_text(field: &'static str, value: String) -> Result<String, ContinuitySubjectError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ContinuitySubjectError::BlankText { field });
    }
    if trimmed.len() > MAX_TEXT_BYTES {
        return Err(ContinuitySubjectError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(ContinuitySubjectError::ControlCharacters { field });
    }
    Ok(trimmed.to_owned())
}

fn hash_subject_fields(
    namespace: &str,
    logical_id: &str,
    scope: &ContinuityScopeV1,
    parent_subject_id: Option<ContinuitySubjectId>,
) -> [u8; 32] {
    let bytes = encode_subject_fields(namespace, logical_id, scope, parent_subject_id);
    let mut hasher = blake3::Hasher::new();
    hasher.update(SUBJECT_DOMAIN);
    hasher.update(&bytes);
    *hasher.finalize().as_bytes()
}

fn encode_subject_fields(
    namespace: &str,
    logical_id: &str,
    scope: &ContinuityScopeV1,
    parent_subject_id: Option<ContinuitySubjectId>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(160);
    put_str(&mut out, CONTINUITY_SUBJECT_SCHEMA_V1);
    put_str(&mut out, namespace);
    put_str(&mut out, logical_id);
    scope.encode(&mut out);
    match parent_subject_id {
        Some(parent) => {
            out.push(1);
            out.extend_from_slice(parent.as_bytes());
        }
        None => out.push(0),
    }
    out
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_subject_fields_produce_same_identity() {
        let a = ContinuitySubjectV1::new(
            "org.example",
            "payments-api",
            ContinuityScopeV1::Service,
            None,
        )
        .unwrap();
        let b = ContinuitySubjectV1::new(
            "org.example",
            "payments-api",
            ContinuityScopeV1::Service,
            None,
        )
        .unwrap();

        assert_eq!(a.id(), b.id());
        assert_eq!(
            a.canonical_identity_bytes().unwrap(),
            b.canonical_identity_bytes().unwrap()
        );
    }

    #[test]
    fn scope_is_part_of_subject_identity() {
        let service =
            ContinuitySubjectV1::new("org.example", "edge-1", ContinuityScopeV1::Service, None)
                .unwrap();
        let device = ContinuitySubjectV1::new(
            "org.example",
            "edge-1",
            ContinuityScopeV1::NetworkDevice,
            None,
        )
        .unwrap();

        assert_ne!(service.id(), device.id());
    }

    #[test]
    fn namespace_is_part_of_subject_identity() {
        let a = ContinuitySubjectV1::new("org.a", "db-primary", ContinuityScopeV1::Machine, None)
            .unwrap();
        let b = ContinuitySubjectV1::new("org.b", "db-primary", ContinuityScopeV1::Machine, None)
            .unwrap();

        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn parent_context_is_part_of_subject_identity() {
        let fabric_a = ContinuitySubjectV1::new(
            "org.example",
            "fabric-a",
            ContinuityScopeV1::NetworkFabric,
            None,
        )
        .unwrap();
        let fabric_b = ContinuitySubjectV1::new(
            "org.example",
            "fabric-b",
            ContinuityScopeV1::NetworkFabric,
            None,
        )
        .unwrap();

        let leaf_a = ContinuitySubjectV1::new(
            "org.example",
            "leaf-01",
            ContinuityScopeV1::NetworkDevice,
            Some(fabric_a.id()),
        )
        .unwrap();
        let leaf_b = ContinuitySubjectV1::new(
            "org.example",
            "leaf-01",
            ContinuityScopeV1::NetworkDevice,
            Some(fabric_b.id()),
        )
        .unwrap();

        assert_ne!(leaf_a.id(), leaf_b.id());
    }

    #[test]
    fn custom_scope_is_canonicalized() {
        let subject = ContinuitySubjectV1::new(
            "org.example",
            "control-plane-a",
            ContinuityScopeV1::custom("  mainframe-lpar  ").unwrap(),
            None,
        )
        .unwrap();

        assert_eq!(
            subject.scope(),
            &ContinuityScopeV1::Custom {
                kind_id: "mainframe-lpar".to_owned()
            }
        );
        subject.validate().unwrap();
    }

    #[test]
    fn blank_or_control_bearing_identifiers_fail_closed() {
        assert!(matches!(
            ContinuitySubjectV1::new("   ", "node-1", ContinuityScopeV1::Machine, None),
            Err(ContinuitySubjectError::BlankText { .. })
        ));

        assert!(matches!(
            ContinuitySubjectV1::new("org.example", "node\n1", ContinuityScopeV1::Machine, None),
            Err(ContinuitySubjectError::ControlCharacters { .. })
        ));
    }

    #[test]
    fn mutated_transport_identity_is_rejected() {
        let mut subject =
            ContinuitySubjectV1::new("org.example", "cluster-a", ContinuityScopeV1::Cluster, None)
                .unwrap();

        subject.logical_id = "cluster-b".to_owned();
        assert_eq!(
            subject.validate(),
            Err(ContinuitySubjectError::SubjectIdentityMismatch)
        );
    }

    #[test]
    fn noncanonical_transport_text_is_rejected() {
        let mut subject =
            ContinuitySubjectV1::new("org.example", "site-a", ContinuityScopeV1::Site, None)
                .unwrap();

        subject.namespace = " org.example ".to_owned();
        assert_eq!(
            subject.validate(),
            Err(ContinuitySubjectError::NonCanonicalText {
                field: "subject namespace"
            })
        );
    }

    #[test]
    fn subject_identity_does_not_encode_authority_state() {
        let subject = ContinuitySubjectV1::new(
            "org.example",
            "payments-cluster",
            ContinuityScopeV1::Cluster,
            None,
        )
        .unwrap();

        subject.validate().unwrap();
        assert!(!subject.canonical_identity_bytes().unwrap().is_empty());
    }
}
