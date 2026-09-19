// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure content-addressed data-lineage contracts for the Interaction Fabric.
//!
//! This crate tracks exact content provenance plus monotone policy-restriction
//! and trust-disposition inheritance across derivations. It deliberately
//! performs no networking, policy evaluation, declassification, credential
//! access, authority checks, persistence, or external effects.
//!
//! Core separation:
//!
//! ```text
//! cognitive ProvenanceTag
//!     != exact DataLineage
//!
//! DataLineage
//!     != truth
//!     != declassification
//!     != data-egress authority
//!     != current execution authority
//!     != DispatchPermit
//! ```

#![deny(unsafe_code)]

use sha2::{Digest as ShaDigest, Sha256};
use std::error::Error;
use std::fmt;
use symthaea_interaction_core::{Digest32, PrincipalRef, ResourceRef};

pub const LINEAGE_SCHEMA_VERSION: u16 = 1;

const SOURCE_DOMAIN: &[u8] = b"symthaea.interaction.lineage.source.v1\0";
const TRANSFORM_DOMAIN: &[u8] = b"symthaea.interaction.lineage.transform.v1\0";
const LINEAGE_DOMAIN: &[u8] = b"symthaea.interaction.lineage.node.v1\0";
const DISCLOSURE_DOMAIN: &[u8] = b"symthaea.interaction.lineage.disclosure.v1\0";

const MAX_ATOM_LEN: usize = 256;
const MAX_SET_ITEMS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageError {
    EmptyText(&'static str),
    TextTooLong {
        field: &'static str,
        length: usize,
        maximum: usize,
    },
    NonCanonicalText {
        field: &'static str,
        byte: u8,
    },
    EmptyParents,
    TooManyItems {
        field: &'static str,
        count: usize,
        maximum: usize,
    },
    DuplicateDigest(&'static str),
}

impl fmt::Display for LineageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyText(field) => write!(formatter, "{field} must not be empty"),
            Self::TextTooLong {
                field,
                length,
                maximum,
            } => write!(
                formatter,
                "{field} length {length} exceeds maximum {maximum}"
            ),
            Self::NonCanonicalText { field, byte } => write!(
                formatter,
                "{field} contains non-canonical byte 0x{byte:02x}; identity text must be printable ASCII"
            ),
            Self::EmptyParents => write!(formatter, "derived lineage requires at least one parent"),
            Self::TooManyItems {
                field,
                count,
                maximum,
            } => write!(
                formatter,
                "{field} contains {count} items; maximum is {maximum}"
            ),
            Self::DuplicateDigest(field) => {
                write!(formatter, "{field} contains a duplicate digest")
            }
        }
    }
}

impl Error for LineageError {}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CanonicalText(String);

impl CanonicalText {
    fn new(field: &'static str, value: &str) -> Result<Self, LineageError> {
        if value.is_empty() {
            return Err(LineageError::EmptyText(field));
        }
        if value.len() > MAX_ATOM_LEN {
            return Err(LineageError::TextTooLong {
                field,
                length: value.len(),
                maximum: MAX_ATOM_LEN,
            });
        }
        if let Some(byte) = value
            .as_bytes()
            .iter()
            .copied()
            .find(|byte| !byte.is_ascii_graphic())
        {
            return Err(LineageError::NonCanonicalText { field, byte });
        }
        Ok(Self(value.to_owned()))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

/// Broad source category for the first content-bearing node in a lineage.
///
/// This is descriptive provenance, not a truth or authority score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginKind {
    ExternalObservation,
    UserInput,
    LocalData,
    Memory,
    Generated,
    Imported,
    Unknown,
}

impl OriginKind {
    const fn code(self) -> u16 {
        match self {
            Self::ExternalObservation => 0,
            Self::UserInput => 1,
            Self::LocalData => 2,
            Self::Memory => 3,
            Self::Generated => 4,
            Self::Imported => 5,
            Self::Unknown => 6,
        }
    }
}

/// Content/control-plane disposition carried with provenance.
///
/// A disposition is metadata, never authority. Multiple dispositions may be
/// present on one derived lineage. For example, a model-generated summary of
/// an external page carries both `ExternalContent` and `InternalDerived`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ContentDisposition {
    ExternalContent,
    UserContent,
    InternalDerived,
    PolicyFactCandidate,
    ControlPlaneCandidate,
    GeneratedProposal,
    Unknown,
}

impl ContentDisposition {
    const fn code(self) -> u16 {
        match self {
            Self::ExternalContent => 0,
            Self::UserContent => 1,
            Self::InternalDerived => 2,
            Self::PolicyFactCandidate => 3,
            Self::ControlPlaneCandidate => 4,
            Self::GeneratedProposal => 5,
            Self::Unknown => 6,
        }
    }
}

/// Exact provenance bindings for an origin node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceBinding {
    origin_kind: OriginKind,
    disposition: ContentDisposition,
    source_resource: Option<Digest32>,
    source_observation: Option<Digest32>,
    principal: Option<Digest32>,
    classification_refs: Vec<Digest32>,
}

impl SourceBinding {
    pub fn new(
        origin_kind: OriginKind,
        disposition: ContentDisposition,
        source_resource: Option<&ResourceRef>,
        source_observation: Option<Digest32>,
        principal: Option<&PrincipalRef>,
        classification_refs: Vec<Digest32>,
    ) -> Result<Self, LineageError> {
        Ok(Self {
            origin_kind,
            disposition,
            source_resource: source_resource.map(ResourceRef::digest),
            source_observation,
            principal: principal.map(PrincipalRef::digest),
            classification_refs: normalize_digest_set("classification_refs", classification_refs)?,
        })
    }

    pub const fn disposition(&self) -> ContentDisposition {
        self.disposition
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(SOURCE_DOMAIN);
        transcript.u16(LINEAGE_SCHEMA_VERSION);
        transcript.u16(self.origin_kind.code());
        transcript.u16(self.disposition.code());
        transcript.optional_digest(self.source_resource);
        transcript.optional_digest(self.source_observation);
        transcript.optional_digest(self.principal);
        transcript.digest_set(&self.classification_refs);
        transcript.finish()
    }
}

/// Exact deterministic transformation identity for a derived lineage node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformRef {
    profile: CanonicalText,
    implementation_digest: Option<Digest32>,
}

impl TransformRef {
    pub fn new(
        profile: &str,
        implementation_digest: Option<Digest32>,
    ) -> Result<Self, LineageError> {
        Ok(Self {
            profile: CanonicalText::new("transform profile", profile)?,
            implementation_digest,
        })
    }

    pub fn profile(&self) -> &str {
        self.profile.as_str()
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(TRANSFORM_DOMAIN);
        transcript.u16(LINEAGE_SCHEMA_VERSION);
        transcript.text(self.profile.as_str());
        transcript.optional_digest(self.implementation_digest);
        transcript.finish()
    }
}

/// Content-addressed provenance node.
///
/// Restrictions are opaque policy/classification commitments. Restrictions and
/// dispositions both propagate monotonically through `derive`; this tranche
/// intentionally has no API that removes a parent restriction/disposition or
/// performs declassification/control-plane promotion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataLineage {
    content_commitment: Digest32,
    parents: Vec<Digest32>,
    source: Option<SourceBinding>,
    transform: Option<TransformRef>,
    dispositions: Vec<ContentDisposition>,
    restrictions: Vec<Digest32>,
}

impl DataLineage {
    pub fn origin(
        content_commitment: Digest32,
        source: SourceBinding,
        restrictions: Vec<Digest32>,
    ) -> Result<Self, LineageError> {
        let disposition = source.disposition();
        Ok(Self {
            content_commitment,
            parents: Vec::new(),
            source: Some(source),
            transform: None,
            dispositions: vec![disposition],
            restrictions: normalize_digest_set("restrictions", restrictions)?,
        })
    }

    pub fn derive(
        content_commitment: Digest32,
        parents: &[DataLineage],
        transform: TransformRef,
        added_restrictions: Vec<Digest32>,
    ) -> Result<Self, LineageError> {
        if parents.is_empty() {
            return Err(LineageError::EmptyParents);
        }
        if parents.len() > MAX_SET_ITEMS {
            return Err(LineageError::TooManyItems {
                field: "parents",
                count: parents.len(),
                maximum: MAX_SET_ITEMS,
            });
        }

        let parent_ids = normalize_digest_set(
            "parents",
            parents.iter().map(DataLineage::digest).collect(),
        )?;

        let mut dispositions = vec![ContentDisposition::InternalDerived];
        let mut restrictions = Vec::new();
        for parent in parents {
            dispositions.extend_from_slice(&parent.dispositions);
            restrictions.extend_from_slice(&parent.restrictions);
        }
        dispositions.sort();
        dispositions.dedup();

        restrictions.extend(added_restrictions);
        restrictions.sort();
        restrictions.dedup();
        if restrictions.len() > MAX_SET_ITEMS {
            return Err(LineageError::TooManyItems {
                field: "restrictions",
                count: restrictions.len(),
                maximum: MAX_SET_ITEMS,
            });
        }

        Ok(Self {
            content_commitment,
            parents: parent_ids,
            source: None,
            transform: Some(transform),
            dispositions,
            restrictions,
        })
    }

    pub fn dispositions(&self) -> &[ContentDisposition] {
        &self.dispositions
    }

    pub fn restrictions(&self) -> &[Digest32] {
        &self.restrictions
    }

    pub fn content_commitment(&self) -> Digest32 {
        self.content_commitment
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(LINEAGE_DOMAIN);
        transcript.u16(LINEAGE_SCHEMA_VERSION);
        transcript.digest(self.content_commitment);
        transcript.digest_set(&self.parents);
        transcript.optional_digest(self.source.as_ref().map(SourceBinding::digest));
        transcript.optional_digest(self.transform.as_ref().map(TransformRef::digest));
        transcript.disposition_set(&self.dispositions);
        transcript.digest_set(&self.restrictions);
        transcript.finish()
    }
}

/// Immutable record of one disclosure sink binding.
///
/// This object records identity only. It does not establish that disclosure
/// was allowed, occurred, or completed successfully.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisclosureBinding {
    lineage: Digest32,
    destination: Digest32,
    outbound_payload_commitment: Digest32,
    purpose_context: Digest32,
    decision_receipt: Option<Digest32>,
    profile: CanonicalText,
}

impl DisclosureBinding {
    pub fn new(
        lineage: &DataLineage,
        destination: &ResourceRef,
        outbound_payload_commitment: Digest32,
        purpose_context: Digest32,
        decision_receipt: Option<Digest32>,
        profile: &str,
    ) -> Result<Self, LineageError> {
        Ok(Self {
            lineage: lineage.digest(),
            destination: destination.digest(),
            outbound_payload_commitment,
            purpose_context,
            decision_receipt,
            profile: CanonicalText::new("disclosure profile", profile)?,
        })
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(DISCLOSURE_DOMAIN);
        transcript.u16(LINEAGE_SCHEMA_VERSION);
        transcript.digest(self.lineage);
        transcript.digest(self.destination);
        transcript.digest(self.outbound_payload_commitment);
        transcript.digest(self.purpose_context);
        transcript.optional_digest(self.decision_receipt);
        transcript.text(self.profile.as_str());
        transcript.finish()
    }
}

fn normalize_digest_set(
    field: &'static str,
    mut values: Vec<Digest32>,
) -> Result<Vec<Digest32>, LineageError> {
    if values.len() > MAX_SET_ITEMS {
        return Err(LineageError::TooManyItems {
            field,
            count: values.len(),
            maximum: MAX_SET_ITEMS,
        });
    }
    values.sort();
    if values.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(LineageError::DuplicateDigest(field));
    }
    Ok(values)
}

struct Transcript {
    hasher: Sha256,
}

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        Self { hasher }
    }

    fn u8(&mut self, value: u8) {
        self.hasher.update([value]);
    }

    fn u16(&mut self, value: u16) {
        self.hasher.update(value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.hasher.update(value.to_be_bytes());
    }

    fn text(&mut self, value: &str) {
        self.u32(value.len() as u32);
        self.hasher.update(value.as_bytes());
    }

    fn digest(&mut self, value: Digest32) {
        self.hasher.update(value.as_bytes());
    }

    fn optional_digest(&mut self, value: Option<Digest32>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.digest(value);
            }
            None => self.u8(0),
        }
    }

    fn digest_set(&mut self, values: &[Digest32]) {
        self.u32(values.len() as u32);
        for value in values {
            self.digest(*value);
        }
    }

    fn disposition_set(&mut self, values: &[ContentDisposition]) {
        self.u32(values.len() as u32);
        for value in values {
            self.u16(value.code());
        }
    }

    fn finish(self) -> Digest32 {
        let digest = self.hasher.finalize();
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&digest);
        Digest32::new(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_interaction_core::{IdentityComponent, IdentityOrdering, NamespaceId};

    fn component(name: &str, value: &str) -> IdentityComponent {
        IdentityComponent::new(name, value).expect("component")
    }

    fn resource(namespace: &str, name: &str) -> ResourceRef {
        ResourceRef::new(
            NamespaceId::new(namespace).expect("namespace"),
            "object",
            IdentityOrdering::NamedSet,
            vec![component("name", name)],
        )
        .expect("resource")
    }

    fn origin(restriction: u8) -> DataLineage {
        DataLineage::origin(
            Digest32::new([0x11; 32]),
            SourceBinding::new(
                OriginKind::ExternalObservation,
                ContentDisposition::ExternalContent,
                Some(&resource("web/http", "source")),
                Some(Digest32::new([0x22; 32])),
                None,
                vec![Digest32::new([0x33; 32])],
            )
            .expect("source"),
            vec![Digest32::new([restriction; 32])],
        )
        .expect("origin")
    }

    #[test]
    fn origin_vector_is_stable() {
        assert_eq!(
            origin(0x44).digest().to_hex(),
            "8118245e56a36f4c803615e67c93ecba35f501c8bb7ce4581c7c8660837b9227"
        );
    }

    #[test]
    fn derived_vector_is_stable() {
        let derived = DataLineage::derive(
            Digest32::new([0x66; 32]),
            &[origin(0x44), origin(0x55)],
            TransformRef::new("summarize/v1", Some(Digest32::new([0x77; 32])))
                .expect("transform"),
            vec![Digest32::new([0x88; 32])],
        )
        .expect("derived");
        assert_eq!(
            derived.digest().to_hex(),
            "efe6ab1a262bd04d52aad1bc9271a771a0fb70c9c33aa698d4c4981b3e2768b0"
        );
    }

    #[test]
    fn derived_lineage_inherits_all_parent_restrictions() {
        let left = origin(0x44);
        let right = origin(0x55);
        let derived = DataLineage::derive(
            Digest32::new([0x66; 32]),
            &[left, right],
            TransformRef::new("summarize/v1", Some(Digest32::new([0x77; 32])))
                .expect("transform"),
            vec![Digest32::new([0x88; 32])],
        )
        .expect("derived");

        assert_eq!(derived.restrictions().len(), 3);
        assert!(derived.restrictions().contains(&Digest32::new([0x44; 32])));
        assert!(derived.restrictions().contains(&Digest32::new([0x55; 32])));
        assert!(derived.restrictions().contains(&Digest32::new([0x88; 32])));
    }

    #[test]
    fn derived_lineage_preserves_external_disposition_locally() {
        let derived = DataLineage::derive(
            Digest32::new([0x66; 32]),
            &[origin(0x44)],
            TransformRef::new("summarize/v1", None).expect("transform"),
            Vec::new(),
        )
        .expect("derived");

        assert_eq!(
            derived.dispositions(),
            &[
                ContentDisposition::ExternalContent,
                ContentDisposition::InternalDerived,
            ]
        );
    }

    #[test]
    fn parent_order_is_non_semantic() {
        let left = origin(0x44);
        let right = origin(0x55);
        let transform = TransformRef::new("combine/v1", None).expect("transform");
        let a = DataLineage::derive(
            Digest32::new([0x66; 32]),
            &[left.clone(), right.clone()],
            transform.clone(),
            Vec::new(),
        )
        .expect("a");
        let b = DataLineage::derive(
            Digest32::new([0x66; 32]),
            &[right, left],
            transform,
            Vec::new(),
        )
        .expect("b");
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn duplicate_parent_fails_closed() {
        let parent = origin(0x44);
        let error = DataLineage::derive(
            Digest32::new([0x66; 32]),
            &[parent.clone(), parent],
            TransformRef::new("combine/v1", None).expect("transform"),
            Vec::new(),
        )
        .expect_err("duplicate parent must fail");
        assert_eq!(error, LineageError::DuplicateDigest("parents"));
    }

    #[test]
    fn derived_content_commitment_changes_identity() {
        let parent = origin(0x44);
        let transform = TransformRef::new("summarize/v1", None).expect("transform");
        let a = DataLineage::derive(
            Digest32::new([0x66; 32]),
            std::slice::from_ref(&parent),
            transform.clone(),
            Vec::new(),
        )
        .expect("a");
        let b = DataLineage::derive(
            Digest32::new([0x67; 32]),
            &[parent],
            transform,
            Vec::new(),
        )
        .expect("b");
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn disclosure_destination_and_payload_are_bound() {
        let lineage = origin(0x44);
        let destination_a = resource("web/http", "public-api");
        let destination_b = resource("web/http", "other-api");
        let a = DisclosureBinding::new(
            &lineage,
            &destination_a,
            Digest32::new([0x90; 32]),
            Digest32::new([0x91; 32]),
            Some(Digest32::new([0x92; 32])),
            "network-egress/v1",
        )
        .expect("a");
        let b = DisclosureBinding::new(
            &lineage,
            &destination_b,
            Digest32::new([0x90; 32]),
            Digest32::new([0x91; 32]),
            Some(Digest32::new([0x92; 32])),
            "network-egress/v1",
        )
        .expect("b");
        let c = DisclosureBinding::new(
            &lineage,
            &destination_a,
            Digest32::new([0x93; 32]),
            Digest32::new([0x91; 32]),
            Some(Digest32::new([0x92; 32])),
            "network-egress/v1",
        )
        .expect("c");
        assert_ne!(a.digest(), b.digest());
        assert_ne!(a.digest(), c.digest());
    }

    #[test]
    fn duplicate_restrictions_fail_closed_at_origin() {
        let restriction = Digest32::new([0x44; 32]);
        let error = DataLineage::origin(
            Digest32::new([0x11; 32]),
            SourceBinding::new(
                OriginKind::LocalData,
                ContentDisposition::UserContent,
                None,
                None,
                None,
                Vec::new(),
            )
            .expect("source"),
            vec![restriction, restriction],
        )
        .expect_err("duplicate restriction must fail");
        assert_eq!(error, LineageError::DuplicateDigest("restrictions"));
    }
}
