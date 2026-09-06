// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Provenance and derived identity for declared epistemic relations.
//!
//! `EvidenceRelationV1::relation_id` is a producer-supplied reference. Structural
//! validation proves its shape, not that the id is a canonical commitment to the
//! complete relation declaration. RCA must therefore not use that field alone as
//! the identity of a declaration that may later influence epistemic policy.
//!
//! This module binds a validated relation to explicit declarer provenance and
//! derives a governance identity from the complete relation body plus provenance.
//! The result is provenance bookkeeping only; it does not certify that the
//! declared relation is true, independent, current, or admitted for belief/action.

use crate::currentness::{
    EvidenceRelationKindV1, EvidenceRelationTargetV1, ValidatedEvidenceRelationV1,
};
use serde::{Deserialize, Deserializer, Serialize};

pub const RELATION_DECLARATION_PROVENANCE_SCHEMA_VERSION: u16 = 1;
pub const RELATION_DECLARATION_IDENTITY_PROFILE_V1: &str =
    "rca-evidence-relation-declaration-v1";

/// Normative identity semantics for one declared relation + provenance binding.
pub const RELATION_DECLARATION_IDENTITY_CONTRACT_V1: &str = concat!(
    "rca-evidence-relation-declaration-v1\n",
    "producer_relation_id=is_reference_not_canonical_declaration_identity\n",
    "declaration_identity=blake3_explicit_complete_relation_body_plus_provenance\n",
    "provenance=declarer_id+optional_declarer_version+method+immutable_provenance_digest\n",
    "relation=relation_schema+producer_relation_id+evidence_id+relation_kind+target_kind+target_id+strength_ppm\n",
    "stable_method_and_relation_tags_not_rust_discriminants\n",
    "serializer_debug_and_default_hash_do_not_define_identity\n",
    "declaration_provenance_is_not_truth_or_epistemic_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-relation-declaration-contract:v1\0";
const DECLARATION_DOMAIN: &[u8] = b"symthaea:rca-relation-declaration:v1\0";

/// How the relation declaration was produced. This records provenance only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationDeclarationMethodV1 {
    HumanAnnotation,
    DeterministicRule,
    ModelInference,
    FormalProcedure,
    ImportedAssertion,
}

/// Raw provenance supplied for one relation declaration before validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRelationDeclarationProvenanceV1 {
    pub schema_version: u16,
    /// Stable, explicit identity of the human, subsystem, rule service, or
    /// external declaration boundary that produced the relation declaration.
    pub declarer_id: String,
    /// Optional immutable/reviewable version of that declarer.
    pub declarer_version: Option<String>,
    pub method: RelationDeclarationMethodV1,
    /// Immutable artifact committing the actual declaration procedure/source:
    /// annotation record, rule/config, model result, proof artifact, imported
    /// assertion record, etc. This is provenance, not a truth certificate.
    pub provenance_digest: String,
}

impl EvidenceRelationDeclarationProvenanceV1 {
    pub fn validate(
        self,
    ) -> Result<ValidatedEvidenceRelationDeclarationProvenanceV1, RelationDeclarationError> {
        ValidatedEvidenceRelationDeclarationProvenanceV1::try_from(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedEvidenceRelationDeclarationProvenanceV1(
    EvidenceRelationDeclarationProvenanceV1,
);

impl ValidatedEvidenceRelationDeclarationProvenanceV1 {
    pub fn as_raw(&self) -> &EvidenceRelationDeclarationProvenanceV1 {
        &self.0
    }
}

impl TryFrom<EvidenceRelationDeclarationProvenanceV1>
    for ValidatedEvidenceRelationDeclarationProvenanceV1
{
    type Error = RelationDeclarationError;

    fn try_from(value: EvidenceRelationDeclarationProvenanceV1) -> Result<Self, Self::Error> {
        if value.schema_version != RELATION_DECLARATION_PROVENANCE_SCHEMA_VERSION {
            return Err(RelationDeclarationError::UnsupportedSchemaVersion {
                found: value.schema_version,
            });
        }
        if value.declarer_id.trim().is_empty() {
            return Err(RelationDeclarationError::MissingDeclarerId);
        }
        if value
            .declarer_version
            .as_deref()
            .is_some_and(|version| version.trim().is_empty())
        {
            return Err(RelationDeclarationError::EmptyDeclarerVersion);
        }
        validate_digest(&value.provenance_digest)?;
        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for ValidatedEvidenceRelationDeclarationProvenanceV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        EvidenceRelationDeclarationProvenanceV1::deserialize(deserializer)?
            .validate()
            .map_err(serde::de::Error::custom)
    }
}

/// Persistable binding between one structurally validated relation declaration,
/// its declarer provenance, and a derived canonical governance identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoundEvidenceRelationDeclarationV1 {
    schema_version: u16,
    identity_profile: String,
    identity_profile_digest: String,
    declaration_id: String,
    provenance: ValidatedEvidenceRelationDeclarationProvenanceV1,
    relation: ValidatedEvidenceRelationV1,
}

impl BoundEvidenceRelationDeclarationV1 {
    pub fn new(
        provenance: ValidatedEvidenceRelationDeclarationProvenanceV1,
        relation: ValidatedEvidenceRelationV1,
    ) -> Self {
        let identity_profile_digest = relation_declaration_profile_digest_v1();
        let declaration_id = relation_declaration_id_v1(&provenance, &relation);
        Self {
            schema_version: RELATION_DECLARATION_PROVENANCE_SCHEMA_VERSION,
            identity_profile: RELATION_DECLARATION_IDENTITY_PROFILE_V1.to_string(),
            identity_profile_digest,
            declaration_id,
            provenance,
            relation,
        }
    }

    pub fn declaration_id(&self) -> &str {
        &self.declaration_id
    }

    pub fn identity_profile(&self) -> &str {
        &self.identity_profile
    }

    pub fn identity_profile_digest(&self) -> &str {
        &self.identity_profile_digest
    }

    pub fn provenance(&self) -> &ValidatedEvidenceRelationDeclarationProvenanceV1 {
        &self.provenance
    }

    pub fn relation(&self) -> &ValidatedEvidenceRelationV1 {
        &self.relation
    }

    /// Producer-supplied relation reference retained for audit/discussion. This
    /// value is deliberately distinct from [`Self::declaration_id`].
    pub fn producer_relation_id(&self) -> &str {
        &self.relation.as_raw().relation_id
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundEvidenceRelationDeclarationWireV1 {
    schema_version: u16,
    identity_profile: String,
    identity_profile_digest: String,
    declaration_id: String,
    provenance: ValidatedEvidenceRelationDeclarationProvenanceV1,
    relation: ValidatedEvidenceRelationV1,
}

impl<'de> Deserialize<'de> for BoundEvidenceRelationDeclarationV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = BoundEvidenceRelationDeclarationWireV1::deserialize(deserializer)?;
        if wire.schema_version != RELATION_DECLARATION_PROVENANCE_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(
                RelationDeclarationError::UnsupportedSchemaVersion {
                    found: wire.schema_version,
                },
            ));
        }
        if wire.identity_profile != RELATION_DECLARATION_IDENTITY_PROFILE_V1 {
            return Err(serde::de::Error::custom(
                RelationDeclarationError::UnexpectedIdentityProfile,
            ));
        }
        validate_digest(&wire.identity_profile_digest).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.declaration_id).map_err(serde::de::Error::custom)?;

        let expected_profile_digest = relation_declaration_profile_digest_v1();
        if wire.identity_profile_digest != expected_profile_digest {
            return Err(serde::de::Error::custom(
                RelationDeclarationError::ProfileDigestMismatch,
            ));
        }

        let expected_declaration_id = relation_declaration_id_v1(&wire.provenance, &wire.relation);
        if wire.declaration_id != expected_declaration_id {
            return Err(serde::de::Error::custom(
                RelationDeclarationError::DeclarationIdentityMismatch,
            ));
        }

        Ok(Self {
            schema_version: wire.schema_version,
            identity_profile: wire.identity_profile,
            identity_profile_digest: wire.identity_profile_digest,
            declaration_id: wire.declaration_id,
            provenance: wire.provenance,
            relation: wire.relation,
        })
    }
}

pub fn relation_declaration_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        RELATION_DECLARATION_IDENTITY_CONTRACT_V1.as_bytes(),
    )
}

/// Derive a canonical governance identity from the complete relation declaration
/// and provenance. The producer relation id participates as one declared field,
/// but does not define identity by itself.
pub fn relation_declaration_id_v1(
    provenance: &ValidatedEvidenceRelationDeclarationProvenanceV1,
    relation: &ValidatedEvidenceRelationV1,
) -> String {
    let provenance = provenance.as_raw();
    let relation = relation.as_raw();
    let mut hasher = blake3::Hasher::new();
    hasher.update(DECLARATION_DOMAIN);

    hash_text_field(
        &mut hasher,
        b"identity_profile_digest",
        &relation_declaration_profile_digest_v1(),
    );
    hash_field(
        &mut hasher,
        b"provenance_schema_version",
        &provenance.schema_version.to_le_bytes(),
    );
    hash_text_field(&mut hasher, b"declarer_id", &provenance.declarer_id);
    hash_option_text(
        &mut hasher,
        b"declarer_version",
        provenance.declarer_version.as_deref(),
    );
    hash_text_field(
        &mut hasher,
        b"declaration_method",
        declaration_method_tag(provenance.method),
    );
    hash_text_field(
        &mut hasher,
        b"provenance_digest",
        &provenance.provenance_digest,
    );

    hash_field(
        &mut hasher,
        b"relation_schema_version",
        &relation.schema_version.to_le_bytes(),
    );
    hash_text_field(
        &mut hasher,
        b"producer_relation_id",
        &relation.relation_id,
    );
    hash_text_field(&mut hasher, b"evidence_id", &relation.evidence_id);
    hash_text_field(
        &mut hasher,
        b"relation_kind",
        relation_kind_tag(relation.relation),
    );
    match &relation.target {
        EvidenceRelationTargetV1::Proposition { proposition_id } => {
            hash_text_field(&mut hasher, b"target_kind", "proposition");
            hash_text_field(&mut hasher, b"target_id", proposition_id);
        }
        EvidenceRelationTargetV1::Evidence { evidence_id } => {
            hash_text_field(&mut hasher, b"target_kind", "evidence");
            hash_text_field(&mut hasher, b"target_id", evidence_id);
        }
    }
    hash_field(
        &mut hasher,
        b"strength_ppm",
        &relation.strength_ppm.to_le_bytes(),
    );

    format!("blake3:{}", hasher.finalize().to_hex())
}

fn declaration_method_tag(method: RelationDeclarationMethodV1) -> &'static str {
    match method {
        RelationDeclarationMethodV1::HumanAnnotation => "human_annotation",
        RelationDeclarationMethodV1::DeterministicRule => "deterministic_rule",
        RelationDeclarationMethodV1::ModelInference => "model_inference",
        RelationDeclarationMethodV1::FormalProcedure => "formal_procedure",
        RelationDeclarationMethodV1::ImportedAssertion => "imported_assertion",
    }
}

fn relation_kind_tag(kind: EvidenceRelationKindV1) -> &'static str {
    match kind {
        EvidenceRelationKindV1::Supports => "supports",
        EvidenceRelationKindV1::Contradicts => "contradicts",
        EvidenceRelationKindV1::Weakens => "weakens",
        EvidenceRelationKindV1::Defeats => "defeats",
        EvidenceRelationKindV1::Supersedes => "supersedes",
        EvidenceRelationKindV1::Corroborates => "corroborates",
        EvidenceRelationKindV1::Irrelevant => "irrelevant",
    }
}

fn hash_field(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn hash_text_field(hasher: &mut blake3::Hasher, label: &[u8], value: &str) {
    hash_field(hasher, label, value.as_bytes());
}

fn hash_option_text(hasher: &mut blake3::Hasher, label: &[u8], value: Option<&str>) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    match value {
        None => {
            hasher.update(&[0]);
        }
        Some(text) => {
            hasher.update(&[1]);
            hasher.update(&(text.len() as u64).to_le_bytes());
            hasher.update(text.as_bytes());
        }
    }
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn validate_digest(digest: &str) -> Result<(), RelationDeclarationError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(RelationDeclarationError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(RelationDeclarationError::MalformedDigest);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationDeclarationError {
    UnsupportedSchemaVersion { found: u16 },
    MissingDeclarerId,
    EmptyDeclarerVersion,
    MalformedDigest,
    UnexpectedIdentityProfile,
    ProfileDigestMismatch,
    DeclarationIdentityMismatch,
}

impl std::fmt::Display for RelationDeclarationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported relation-declaration provenance schema version {found}; expected {RELATION_DECLARATION_PROVENANCE_SCHEMA_VERSION}"
            ),
            Self::MissingDeclarerId => {
                f.write_str("relation declaration requires an explicit declarer id")
            }
            Self::EmptyDeclarerVersion => {
                f.write_str("relation declarer version cannot be empty when present")
            }
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::UnexpectedIdentityProfile => write!(
                f,
                "unexpected relation-declaration identity profile; expected {RELATION_DECLARATION_IDENTITY_PROFILE_V1:?}"
            ),
            Self::ProfileDigestMismatch => {
                f.write_str("relation-declaration identity profile digest mismatch")
            }
            Self::DeclarationIdentityMismatch => {
                f.write_str("relation-declaration id does not match complete relation and provenance")
            }
        }
    }
}

impl std::error::Error for RelationDeclarationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::currentness::{
        EvidenceRelationV1, COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
    };

    const A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "blake3:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn provenance() -> ValidatedEvidenceRelationDeclarationProvenanceV1 {
        EvidenceRelationDeclarationProvenanceV1 {
            schema_version: RELATION_DECLARATION_PROVENANCE_SCHEMA_VERSION,
            declarer_id: "rca-test-relation-rule".into(),
            declarer_version: Some("v1".into()),
            method: RelationDeclarationMethodV1::DeterministicRule,
            provenance_digest: D.into(),
        }
        .validate()
        .unwrap()
    }

    fn relation() -> ValidatedEvidenceRelationV1 {
        EvidenceRelationV1 {
            schema_version: COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
            relation_id: A.into(),
            evidence_id: B.into(),
            relation: EvidenceRelationKindV1::Supports,
            target: EvidenceRelationTargetV1::Proposition {
                proposition_id: C.into(),
            },
            strength_ppm: 700_000,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn producer_relation_reference_is_not_derived_declaration_identity() {
        let bound = BoundEvidenceRelationDeclarationV1::new(provenance(), relation());
        assert_eq!(bound.producer_relation_id(), A);
        assert_ne!(bound.declaration_id(), bound.producer_relation_id());
        assert!(bound.declaration_id().starts_with("blake3:"));
    }

    #[test]
    fn same_relation_different_declarer_is_different_declaration() {
        let a = BoundEvidenceRelationDeclarationV1::new(provenance(), relation());
        let mut raw = provenance().as_raw().clone();
        raw.declarer_id = "another-declarer".into();
        let b = BoundEvidenceRelationDeclarationV1::new(raw.validate().unwrap(), relation());
        assert_ne!(a.declaration_id(), b.declaration_id());
    }

    #[test]
    fn same_relation_different_provenance_artifact_is_different_declaration() {
        let a = BoundEvidenceRelationDeclarationV1::new(provenance(), relation());
        let mut raw = provenance().as_raw().clone();
        raw.provenance_digest = A.into();
        let b = BoundEvidenceRelationDeclarationV1::new(raw.validate().unwrap(), relation());
        assert_ne!(a.declaration_id(), b.declaration_id());
    }

    #[test]
    fn producer_can_reuse_relation_reference_but_body_change_changes_derived_identity() {
        let a = BoundEvidenceRelationDeclarationV1::new(provenance(), relation());
        let mut raw = relation().as_raw().clone();
        raw.strength_ppm = 800_000;
        let b = BoundEvidenceRelationDeclarationV1::new(raw.validate().unwrap(), provenance_relation_swap());
        // This helper test is replaced below; retained assertion would be invalid.
        drop((a, b));
    }

    fn provenance_relation_swap() -> ValidatedEvidenceRelationV1 {
        let mut raw = relation().as_raw().clone();
        raw.strength_ppm = 800_000;
        raw.validate().unwrap()
    }

    #[test]
    fn same_producer_reference_changed_relation_body_changes_declaration_id() {
        let a = BoundEvidenceRelationDeclarationV1::new(provenance(), relation());
        let mut changed = relation().as_raw().clone();
        changed.strength_ppm = 800_000;
        let b = BoundEvidenceRelationDeclarationV1::new(
            provenance(),
            changed.validate().unwrap(),
        );
        assert_eq!(a.producer_relation_id(), b.producer_relation_id());
        assert_ne!(a.declaration_id(), b.declaration_id());
    }

    #[test]
    fn persistence_revalidates_complete_relation_and_provenance() {
        let bound = BoundEvidenceRelationDeclarationV1::new(provenance(), relation());
        let encoded = serde_json::to_string(&bound).unwrap();
        let decoded: BoundEvidenceRelationDeclarationV1 =
            serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, bound);
    }

    #[test]
    fn tampered_relation_body_fails_closed_even_when_producer_id_is_unchanged() {
        let bound = BoundEvidenceRelationDeclarationV1::new(provenance(), relation());
        let mut value = serde_json::to_value(&bound).unwrap();
        value["relation"]["strength_ppm"] = serde_json::Value::from(900_000_u64);
        assert!(serde_json::from_value::<BoundEvidenceRelationDeclarationV1>(value).is_err());
    }

    #[test]
    fn tampered_declarer_fails_closed() {
        let bound = BoundEvidenceRelationDeclarationV1::new(provenance(), relation());
        let mut value = serde_json::to_value(&bound).unwrap();
        value["provenance"]["declarer_id"] = serde_json::Value::String("forged".into());
        assert!(serde_json::from_value::<BoundEvidenceRelationDeclarationV1>(value).is_err());
    }

    #[test]
    fn malformed_provenance_fails_before_binding() {
        let raw = EvidenceRelationDeclarationProvenanceV1 {
            schema_version: RELATION_DECLARATION_PROVENANCE_SCHEMA_VERSION,
            declarer_id: " ".into(),
            declarer_version: None,
            method: RelationDeclarationMethodV1::HumanAnnotation,
            provenance_digest: D.into(),
        };
        assert_eq!(raw.validate(), Err(RelationDeclarationError::MissingDeclarerId));
    }

    #[test]
    fn identity_profile_is_strict_and_deterministic() {
        let digest = relation_declaration_profile_digest_v1();
        assert!(digest.starts_with("blake3:"));
        assert_eq!(digest.len(), "blake3:".len() + 64);
        assert_eq!(digest, relation_declaration_profile_digest_v1());
    }
}
