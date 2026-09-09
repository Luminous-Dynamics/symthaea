// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical capability identity and prerequisite expressions.
//!
//! A capability definition is a declared model. It does not establish local
//! realization, current availability, verified sufficiency, or authority.
//!
//! Core theorem:
//!
//! `CapabilityDefinitionV1 != CapabilityRealization != CurrentAvailability
//! != VerifiedSufficiency != Authority`.
//!
//! Capability identity is intentionally distinct from definition identity.
//! Stable capability identities may therefore participate in cyclic dependency
//! structures without requiring recursively self-referential content hashes.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Stable schema for one capability definition.
pub const CAPABILITY_DEFINITION_SCHEMA_V1: &str = "symthaea-continuity-capability-definition-v1";

const CAPABILITY_ID_DOMAIN: &[u8] = b"symthaea.continuity.capability-id.v1\0";
const CAPABILITY_DEFINITION_DOMAIN: &[u8] = b"symthaea.continuity.capability-definition.v1\0";
const MAX_TEXT_BYTES: usize = 1024;
const MAX_REQUIREMENT_DEPTH: usize = 32;
const MAX_REQUIREMENT_NODES: usize = 4096;

/// Stable semantic identity of a capability.
///
/// This identity commits only to the canonical namespace and logical id. It is
/// not a content hash of the full definition. Keeping the stable semantic key
/// separate from [`CapabilityDefinitionId`] permits cyclic capability graphs:
/// A can reference B while B references A without recursive hashes.
///
/// The identity does not establish that the capability exists, is realized,
/// available, sufficient, safe, or authorized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CapabilityId([u8; 32]);

impl CapabilityId {
    /// Construct a stable capability identity from a canonical semantic key.
    pub fn new(
        namespace: impl Into<String>,
        logical_id: impl Into<String>,
    ) -> Result<Self, CapabilityError> {
        let namespace = checked_text("capability namespace", namespace.into())?;
        let logical_id = checked_text("capability logical_id", logical_id.into())?;
        Ok(Self(hash_capability_key(&namespace, &logical_id)))
    }

    /// Raw BLAKE3-256 identity bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Exact content identity of one canonical capability-definition revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CapabilityDefinitionId([u8; 32]);

impl CapabilityDefinitionId {
    /// Raw BLAKE3-256 identity bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Pure AND/OR prerequisite expression.
///
/// Leaves reference stable [`CapabilityId`] values. `AllOf` and `AnyOf` are
/// canonicalized as unordered, duplicate-free sets. Nested groups of the same
/// kind are flattened, and one-child groups collapse to the child.
///
/// This is dependency structure only. It does not establish that a referenced
/// capability exists in any particular graph snapshot or is currently usable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CapabilityRequirementV1 {
    Leaf { capability_id: CapabilityId },
    AllOf {
        requirements: Vec<CapabilityRequirementV1>,
    },
    AnyOf {
        requirements: Vec<CapabilityRequirementV1>,
    },
}

impl CapabilityRequirementV1 {
    /// One prerequisite capability.
    pub fn leaf(capability_id: CapabilityId) -> Self {
        Self::Leaf { capability_id }
    }

    /// All child requirements must be satisfied.
    ///
    /// The returned expression is canonical. Empty groups are rejected and a
    /// one-child group collapses to that child.
    pub fn all_of(
        requirements: Vec<CapabilityRequirementV1>,
    ) -> Result<Self, CapabilityError> {
        Self::AllOf { requirements }.canonicalize()
    }

    /// Any one child requirement may satisfy this prerequisite.
    ///
    /// The returned expression is canonical. Empty groups are rejected and a
    /// one-child group collapses to that child.
    pub fn any_of(
        requirements: Vec<CapabilityRequirementV1>,
    ) -> Result<Self, CapabilityError> {
        Self::AnyOf { requirements }.canonicalize()
    }

    /// Revalidate that a transported expression is already canonical.
    pub fn validate(&self) -> Result<(), CapabilityError> {
        let canonical = self.clone().canonicalize()?;
        if canonical != *self {
            return Err(CapabilityError::NonCanonicalRequirement);
        }
        Ok(())
    }

    pub(crate) fn collect_capability_refs(&self, out: &mut Vec<CapabilityId>) {
        match self {
            Self::Leaf { capability_id } => out.push(*capability_id),
            Self::AllOf { requirements } | Self::AnyOf { requirements } => {
                for requirement in requirements {
                    requirement.collect_capability_refs(out);
                }
            }
        }
    }

    fn canonicalize(self) -> Result<Self, CapabilityError> {
        let mut nodes = 0usize;
        self.canonicalize_inner(0, &mut nodes)
    }

    fn canonicalize_inner(
        self,
        depth: usize,
        nodes: &mut usize,
    ) -> Result<Self, CapabilityError> {
        if depth > MAX_REQUIREMENT_DEPTH {
            return Err(CapabilityError::RequirementTooDeep);
        }
        *nodes = nodes.saturating_add(1);
        if *nodes > MAX_REQUIREMENT_NODES {
            return Err(CapabilityError::TooManyRequirementNodes);
        }

        match self {
            Self::Leaf { capability_id } => Ok(Self::Leaf { capability_id }),
            Self::AllOf { requirements } => {
                canonicalize_group(requirements, true, depth, nodes)
            }
            Self::AnyOf { requirements } => {
                canonicalize_group(requirements, false, depth, nodes)
            }
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Leaf { capability_id } => {
                out.push(1);
                out.extend_from_slice(capability_id.as_bytes());
            }
            Self::AllOf { requirements } => {
                out.push(2);
                put_len(out, requirements.len());
                for requirement in requirements {
                    requirement.encode(out);
                }
            }
            Self::AnyOf { requirements } => {
                out.push(3);
                put_len(out, requirements.len());
                for requirement in requirements {
                    requirement.encode(out);
                }
            }
        }
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }
}

fn canonicalize_group(
    requirements: Vec<CapabilityRequirementV1>,
    all_of: bool,
    depth: usize,
    nodes: &mut usize,
) -> Result<CapabilityRequirementV1, CapabilityError> {
    if requirements.is_empty() {
        return Err(CapabilityError::EmptyRequirementGroup);
    }

    let mut canonical = Vec::new();
    for requirement in requirements {
        let requirement = requirement.canonicalize_inner(depth + 1, nodes)?;
        match (all_of, requirement) {
            (true, CapabilityRequirementV1::AllOf { requirements }) => {
                canonical.extend(requirements);
            }
            (false, CapabilityRequirementV1::AnyOf { requirements }) => {
                canonical.extend(requirements);
            }
            (_, requirement) => canonical.push(requirement),
        }
    }

    canonical.sort_by_key(CapabilityRequirementV1::canonical_bytes);
    canonical.dedup();

    match canonical.len() {
        0 => Err(CapabilityError::EmptyRequirementGroup),
        1 => Ok(canonical.remove(0)),
        _ if all_of => Ok(CapabilityRequirementV1::AllOf {
            requirements: canonical,
        }),
        _ => Ok(CapabilityRequirementV1::AnyOf {
            requirements: canonical,
        }),
    }
}

/// One exact, canonical definition revision for a stable capability.
///
/// `capability_id` is stable across definition revisions with the same namespace
/// and logical id. `definition_id` changes whenever the canonical prerequisite
/// expression changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityDefinitionV1 {
    schema_version: String,
    namespace: String,
    logical_id: String,
    requirements: Option<CapabilityRequirementV1>,
    capability_id: CapabilityId,
    definition_id: CapabilityDefinitionId,
}

impl CapabilityDefinitionV1 {
    /// Construct one canonical capability definition.
    pub fn new(
        namespace: impl Into<String>,
        logical_id: impl Into<String>,
        requirements: Option<CapabilityRequirementV1>,
    ) -> Result<Self, CapabilityError> {
        let namespace = checked_text("capability namespace", namespace.into())?;
        let logical_id = checked_text("capability logical_id", logical_id.into())?;
        let requirements = requirements
            .map(CapabilityRequirementV1::canonicalize)
            .transpose()?;
        let capability_id = CapabilityId(hash_capability_key(&namespace, &logical_id));
        let definition_id = CapabilityDefinitionId(hash_definition(
            &namespace,
            &logical_id,
            requirements.as_ref(),
        ));

        Ok(Self {
            schema_version: CAPABILITY_DEFINITION_SCHEMA_V1.to_owned(),
            namespace,
            logical_id,
            requirements,
            capability_id,
            definition_id,
        })
    }

    /// Revalidate schema, canonical text, canonical prerequisites, and stored
    /// identities.
    pub fn validate(&self) -> Result<(), CapabilityError> {
        if self.schema_version != CAPABILITY_DEFINITION_SCHEMA_V1 {
            return Err(CapabilityError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }

        let namespace = checked_text("capability namespace", self.namespace.clone())?;
        if namespace != self.namespace {
            return Err(CapabilityError::NonCanonicalText {
                field: "capability namespace",
            });
        }

        let logical_id = checked_text("capability logical_id", self.logical_id.clone())?;
        if logical_id != self.logical_id {
            return Err(CapabilityError::NonCanonicalText {
                field: "capability logical_id",
            });
        }

        if let Some(requirements) = &self.requirements {
            requirements.validate()?;
        }

        let expected_capability_id =
            CapabilityId(hash_capability_key(&self.namespace, &self.logical_id));
        if expected_capability_id != self.capability_id {
            return Err(CapabilityError::CapabilityIdentityMismatch);
        }

        let expected_definition_id = CapabilityDefinitionId(hash_definition(
            &self.namespace,
            &self.logical_id,
            self.requirements.as_ref(),
        ));
        if expected_definition_id != self.definition_id {
            return Err(CapabilityError::DefinitionIdentityMismatch);
        }

        Ok(())
    }

    /// Stable semantic capability identity.
    pub fn id(&self) -> CapabilityId {
        self.capability_id
    }

    /// Exact identity of this definition revision.
    pub fn definition_id(&self) -> CapabilityDefinitionId {
        self.definition_id
    }

    /// Administrative/application namespace used only for stable naming.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Stable logical capability identifier within the namespace.
    pub fn logical_id(&self) -> &str {
        &self.logical_id
    }

    /// Canonical prerequisite expression, if this definition has capability
    /// prerequisites.
    pub fn requirements(&self) -> Option<&CapabilityRequirementV1> {
        self.requirements.as_ref()
    }

    /// Exact canonical definition bytes before domain-separated hashing.
    pub fn canonical_definition_bytes(&self) -> Result<Vec<u8>, CapabilityError> {
        self.validate()?;
        Ok(encode_definition(
            &self.namespace,
            &self.logical_id,
            self.requirements.as_ref(),
        ))
    }

    pub(crate) fn referenced_capabilities(&self) -> Vec<CapabilityId> {
        let mut refs = Vec::new();
        if let Some(requirements) = &self.requirements {
            requirements.collect_capability_refs(&mut refs);
        }
        refs.sort_unstable();
        refs.dedup();
        refs
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CapabilityError {
    #[error("unsupported capability definition schema: {0}")]
    UnsupportedSchema(String),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds {MAX_TEXT_BYTES} bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("{field} is not canonically trimmed")]
    NonCanonicalText { field: &'static str },
    #[error("capability requirement group must not be empty")]
    EmptyRequirementGroup,
    #[error("capability requirement nesting exceeds {MAX_REQUIREMENT_DEPTH}")]
    RequirementTooDeep,
    #[error("capability requirement tree exceeds {MAX_REQUIREMENT_NODES} nodes")]
    TooManyRequirementNodes,
    #[error("capability requirement expression is not canonical")]
    NonCanonicalRequirement,
    #[error("stored capability identity does not match canonical semantic key")]
    CapabilityIdentityMismatch,
    #[error("stored capability definition identity does not match canonical fields")]
    DefinitionIdentityMismatch,
}

fn checked_text(field: &'static str, value: String) -> Result<String, CapabilityError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(CapabilityError::BlankText { field });
    }
    if trimmed.len() > MAX_TEXT_BYTES {
        return Err(CapabilityError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(CapabilityError::ControlCharacters { field });
    }
    Ok(trimmed.to_owned())
}

fn hash_capability_key(namespace: &str, logical_id: &str) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(namespace.len() + logical_id.len() + 24);
    put_str(&mut bytes, namespace);
    put_str(&mut bytes, logical_id);
    domain_hash(CAPABILITY_ID_DOMAIN, &bytes)
}

fn hash_definition(
    namespace: &str,
    logical_id: &str,
    requirements: Option<&CapabilityRequirementV1>,
) -> [u8; 32] {
    let bytes = encode_definition(namespace, logical_id, requirements);
    domain_hash(CAPABILITY_DEFINITION_DOMAIN, &bytes)
}

fn encode_definition(
    namespace: &str,
    logical_id: &str,
    requirements: Option<&CapabilityRequirementV1>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(192);
    put_str(&mut out, CAPABILITY_DEFINITION_SCHEMA_V1);
    put_str(&mut out, namespace);
    put_str(&mut out, logical_id);
    match requirements {
        Some(requirements) => {
            out.push(1);
            requirements.encode(&mut out);
        }
        None => out.push(0),
    }
    out
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
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

    fn id(name: &str) -> CapabilityId {
        CapabilityId::new("org.example", name).unwrap()
    }

    #[test]
    fn stable_capability_identity_is_independent_of_definition_revision() {
        let bearing = id("bearing-production");
        let v1 = CapabilityDefinitionV1::new("org.example", "pump-repair", None).unwrap();
        let v2 = CapabilityDefinitionV1::new(
            "org.example",
            "pump-repair",
            Some(CapabilityRequirementV1::leaf(bearing)),
        )
        .unwrap();

        assert_eq!(v1.id(), v2.id());
        assert_ne!(v1.definition_id(), v2.definition_id());
    }

    #[test]
    fn namespace_is_part_of_capability_identity() {
        let a = CapabilityId::new("org.a", "potable-water").unwrap();
        let b = CapabilityId::new("org.b", "potable-water").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn all_of_child_order_is_canonicalized() {
        let a = CapabilityRequirementV1::leaf(id("a"));
        let b = CapabilityRequirementV1::leaf(id("b"));

        let left = CapabilityRequirementV1::all_of(vec![a.clone(), b.clone()]).unwrap();
        let right = CapabilityRequirementV1::all_of(vec![b, a]).unwrap();

        assert_eq!(left, right);
    }

    #[test]
    fn duplicate_and_nested_same_kind_requirements_canonicalize() {
        let a = CapabilityRequirementV1::leaf(id("a"));
        let b = CapabilityRequirementV1::leaf(id("b"));
        let nested = CapabilityRequirementV1::all_of(vec![a.clone(), b.clone()]).unwrap();

        let expression =
            CapabilityRequirementV1::all_of(vec![nested, a.clone(), b.clone()]).unwrap();

        assert_eq!(
            expression,
            CapabilityRequirementV1::all_of(vec![a, b]).unwrap()
        );
    }

    #[test]
    fn all_of_and_any_of_are_distinct_definition_revisions() {
        let a = CapabilityRequirementV1::leaf(id("a"));
        let b = CapabilityRequirementV1::leaf(id("b"));

        let all = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(CapabilityRequirementV1::all_of(vec![a.clone(), b.clone()]).unwrap()),
        )
        .unwrap();
        let any = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(CapabilityRequirementV1::any_of(vec![a, b]).unwrap()),
        )
        .unwrap();

        assert_eq!(all.id(), any.id());
        assert_ne!(all.definition_id(), any.definition_id());
    }

    #[test]
    fn cyclic_references_do_not_require_recursive_content_hashes() {
        let a_id = id("a");
        let b_id = id("b");

        let a = CapabilityDefinitionV1::new(
            "org.example",
            "a",
            Some(CapabilityRequirementV1::leaf(b_id)),
        )
        .unwrap();
        let b = CapabilityDefinitionV1::new(
            "org.example",
            "b",
            Some(CapabilityRequirementV1::leaf(a_id)),
        )
        .unwrap();

        assert_eq!(a.id(), a_id);
        assert_eq!(b.id(), b_id);
        assert_ne!(a.definition_id(), b.definition_id());
    }

    #[test]
    fn transported_noncanonical_requirement_is_rejected() {
        let a = CapabilityRequirementV1::leaf(id("a"));
        let mut definition =
            CapabilityDefinitionV1::new("org.example", "target", Some(a.clone())).unwrap();

        definition.requirements = Some(CapabilityRequirementV1::AllOf {
            requirements: vec![a],
        });

        assert_eq!(
            definition.validate(),
            Err(CapabilityError::NonCanonicalRequirement)
        );
    }

    #[test]
    fn canonical_definition_bytes_are_deterministic() {
        let requirement = CapabilityRequirementV1::any_of(vec![
            CapabilityRequirementV1::leaf(id("manual-pump")),
            CapabilityRequirementV1::leaf(id("electric-pump")),
        ])
        .unwrap();

        let a =
            CapabilityDefinitionV1::new("org.example", "pumping", Some(requirement.clone())).unwrap();
        let b = CapabilityDefinitionV1::new("org.example", "pumping", Some(requirement)).unwrap();

        assert_eq!(a.definition_id(), b.definition_id());
        assert_eq!(
            a.canonical_definition_bytes().unwrap(),
            b.canonical_definition_bytes().unwrap()
        );
    }
}
