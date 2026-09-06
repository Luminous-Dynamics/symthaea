// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Use-specific qualification for relation declarers.
//!
//! Relation-declaration provenance proves who/what declared a relation and binds
//! the declaration body. It does not prove that the declarer is qualified for a
//! particular epistemic use. This module adds that separate boundary.
//!
//! The persistable qualification record is proposition-, method-, relation-kind-,
//! evaluator-, policy-, artifact-, time-, and use-specific. It grants no use by
//! itself. A non-deserializable eligibility capability is issued only after an
//! exact join against one bound relation declaration and an explicit use context.

use crate::{
    currentness::{EvidenceRelationKindV1, EvidenceRelationTargetV1},
    relation_provenance::{
        BoundEvidenceRelationDeclarationV1, RelationDeclarationMethodV1,
    },
};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashSet;

pub const RELATION_DECLARER_QUALIFICATION_SCHEMA_VERSION: u16 = 1;
pub const RELATION_DECLARER_QUALIFICATION_PROFILE_V1: &str =
    "rca-relation-declarer-qualification-v1";
pub const RELATION_DECLARATION_ELIGIBILITY_SCHEMA_VERSION: u16 = 1;
pub const RELATION_DECLARATION_ELIGIBILITY_PROFILE_V1: &str =
    "rca-relation-declaration-eligibility-v1";

/// Normative qualification semantics.
pub const RELATION_DECLARER_QUALIFICATION_CONTRACT_V1: &str = concat!(
    "rca-relation-declarer-qualification-v1\n",
    "qualification_is_use_specific_not_global_trust\n",
    "scope=exact_proposition_id_v1\n",
    "subject=declarer_id+optional_version+declaration_method\n",
    "qualifier_must_not_equal_subject_declarer\n",
    "qualification=qualifier+evaluator+policy_digest+artifact_digest\n",
    "allowed_relation_kinds=canonical_unique_set_without_supersedes\n",
    "validity=qualified_at_unix_ms<=now<=valid_until_unix_ms\n",
    "permitted_use=shadow_runtime_disposition_only_v1\n",
    "qualification_id=blake3_explicit_complete_record_v1\n",
    "registered_qualification_is_not_relation_eligibility\n",
    "eligibility_requires_exact_declaration_subject_method_proposition_kind_use_time_join\n",
    "eligible_result=is_issued_private_non_deserializable_capability\n",
    "qualification_and_eligibility_are_not_truth_belief_action_or_promotion_authority\n",
);

const QUALIFICATION_PROFILE_DOMAIN: &[u8] =
    b"symthaea:rca-relation-declarer-qualification-contract:v1\0";
const QUALIFICATION_ID_DOMAIN: &[u8] =
    b"symthaea:rca-relation-declarer-qualification:v1\0";
const ELIGIBILITY_PROFILE_DOMAIN: &[u8] =
    b"symthaea:rca-relation-declaration-eligibility-contract:v1\0";
const ELIGIBILITY_CONTEXT_DOMAIN: &[u8] =
    b"symthaea:rca-relation-declaration-eligibility-context:v1\0";
const ELIGIBILITY_ID_DOMAIN: &[u8] =
    b"symthaea:rca-relation-declaration-eligibility:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationDeclarationUseV1 {
    ShadowRuntimeDisposition,
}

/// Persistable raw qualification record. Validation canonicalizes the allowed
/// relation-kind set and derives a stable qualification identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationDeclarerQualificationV1 {
    pub schema_version: u16,
    pub subject_declarer_id: String,
    pub subject_declarer_version: Option<String>,
    pub subject_method: RelationDeclarationMethodV1,
    /// Authority responsible for approving this qualification. V1 structurally
    /// requires this identity to differ from the subject declarer identity. That
    /// inequality is not, by itself, proof of organizational independence.
    pub qualifier_id: String,
    pub qualifier_version: Option<String>,
    /// Evaluator/harness that produced the qualification evidence.
    pub evaluator_id: String,
    pub evaluator_version: Option<String>,
    /// V1 intentionally scopes qualification to one exact proposition identity.
    pub proposition_id: String,
    pub allowed_relation_kinds: Vec<EvidenceRelationKindV1>,
    pub permitted_use: RelationDeclarationUseV1,
    pub qualification_policy_digest: String,
    pub qualification_artifact_digest: String,
    pub qualified_at_unix_ms: u64,
    pub valid_until_unix_ms: u64,
}

impl RelationDeclarerQualificationV1 {
    pub fn register(
        self,
    ) -> Result<RegisteredRelationDeclarerQualificationV1, RelationQualificationError> {
        RegisteredRelationDeclarerQualificationV1::try_from(self)
    }
}

/// Persistable, revalidated qualification record with a derived governance id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredRelationDeclarerQualificationV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    qualification_id: String,
    record: RelationDeclarerQualificationV1,
}

impl RegisteredRelationDeclarerQualificationV1 {
    pub fn qualification_id(&self) -> &str {
        &self.qualification_id
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn profile_contract_digest(&self) -> &str {
        &self.profile_contract_digest
    }

    pub fn record(&self) -> &RelationDeclarerQualificationV1 {
        &self.record
    }
}

impl TryFrom<RelationDeclarerQualificationV1> for RegisteredRelationDeclarerQualificationV1 {
    type Error = RelationQualificationError;

    fn try_from(mut value: RelationDeclarerQualificationV1) -> Result<Self, Self::Error> {
        validate_and_canonicalize_qualification(&mut value)?;
        let profile_contract_digest = relation_declarer_qualification_profile_digest_v1();
        let qualification_id = relation_declarer_qualification_id_v1(
            &profile_contract_digest,
            &value,
        );
        Ok(Self {
            schema_version: RELATION_DECLARER_QUALIFICATION_SCHEMA_VERSION,
            profile: RELATION_DECLARER_QUALIFICATION_PROFILE_V1.to_string(),
            profile_contract_digest,
            qualification_id,
            record: value,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisteredRelationDeclarerQualificationWireV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    qualification_id: String,
    record: RelationDeclarerQualificationV1,
}

impl<'de> Deserialize<'de> for RegisteredRelationDeclarerQualificationV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RegisteredRelationDeclarerQualificationWireV1::deserialize(deserializer)?;
        if wire.schema_version != RELATION_DECLARER_QUALIFICATION_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(
                RelationQualificationError::UnsupportedSchemaVersion {
                    found: wire.schema_version,
                },
            ));
        }
        if wire.profile != RELATION_DECLARER_QUALIFICATION_PROFILE_V1 {
            return Err(serde::de::Error::custom(
                RelationQualificationError::UnexpectedQualificationProfile,
            ));
        }
        validate_digest(&wire.profile_contract_digest).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.qualification_id).map_err(serde::de::Error::custom)?;

        let expected = wire
            .record
            .clone()
            .register()
            .map_err(serde::de::Error::custom)?;
        if wire.profile_contract_digest != expected.profile_contract_digest
            || wire.qualification_id != expected.qualification_id
            || wire.record != expected.record
        {
            return Err(serde::de::Error::custom(
                RelationQualificationError::QualificationIdentityMismatch,
            ));
        }
        Ok(expected)
    }
}

/// Explicit context for deciding whether one exact bound relation declaration is
/// eligible for one use now.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationDeclarationEligibilityContextV1 {
    pub schema_version: u16,
    pub proposition_id: String,
    pub use_case: RelationDeclarationUseV1,
    pub now_unix_ms: u64,
}

impl RelationDeclarationEligibilityContextV1 {
    pub fn validate(
        self,
    ) -> Result<ValidatedRelationDeclarationEligibilityContextV1, RelationQualificationError> {
        if self.schema_version != RELATION_DECLARATION_ELIGIBILITY_SCHEMA_VERSION {
            return Err(RelationQualificationError::UnsupportedEligibilitySchemaVersion {
                found: self.schema_version,
            });
        }
        validate_digest(&self.proposition_id)?;
        Ok(ValidatedRelationDeclarationEligibilityContextV1(self))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedRelationDeclarationEligibilityContextV1(
    RelationDeclarationEligibilityContextV1,
);

impl ValidatedRelationDeclarationEligibilityContextV1 {
    pub fn as_raw(&self) -> &RelationDeclarationEligibilityContextV1 {
        &self.0
    }

    pub fn commitment(&self) -> String {
        eligibility_context_commitment_v1(&self.0)
    }
}

impl<'de> Deserialize<'de> for ValidatedRelationDeclarationEligibilityContextV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        RelationDeclarationEligibilityContextV1::deserialize(deserializer)?
            .validate()
            .map_err(serde::de::Error::custom)
    }
}

/// Issued use-eligibility capability. Private fields and no `Deserialize` are
/// deliberate: archived JSON cannot recreate current eligibility.
#[must_use = "eligible relation declarations are scoped shadow capabilities and should be inspected"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DispositionEligibleRelationDeclarationV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    eligibility_id: String,
    context_commitment: String,
    declaration: BoundEvidenceRelationDeclarationV1,
    qualification: RegisteredRelationDeclarerQualificationV1,
}

impl DispositionEligibleRelationDeclarationV1 {
    pub fn eligibility_id(&self) -> &str {
        &self.eligibility_id
    }

    pub fn context_commitment(&self) -> &str {
        &self.context_commitment
    }

    pub fn declaration(&self) -> &BoundEvidenceRelationDeclarationV1 {
        &self.declaration
    }

    pub fn qualification(&self) -> &RegisteredRelationDeclarerQualificationV1 {
        &self.qualification
    }

    pub fn proposition_id(&self) -> &str {
        &self.qualification.record.proposition_id
    }
}

/// Issue exact-use eligibility after joining one declaration, one registered
/// qualification, and one explicit current use context.
pub fn admit_relation_declaration_for_use_v1(
    declaration: &BoundEvidenceRelationDeclarationV1,
    qualification: &RegisteredRelationDeclarerQualificationV1,
    context: &ValidatedRelationDeclarationEligibilityContextV1,
) -> Result<DispositionEligibleRelationDeclarationV1, RelationQualificationError> {
    let provenance = declaration.provenance().as_raw();
    let relation = declaration.relation().as_raw();
    let record = qualification.record();
    let context_raw = context.as_raw();

    if provenance.declarer_id != record.subject_declarer_id
        || provenance.declarer_version != record.subject_declarer_version
        || provenance.method != record.subject_method
    {
        return Err(RelationQualificationError::DeclarationSubjectMismatch);
    }

    if record.proposition_id != context_raw.proposition_id {
        return Err(RelationQualificationError::QualificationPropositionMismatch);
    }

    let relation_proposition = match &relation.target {
        EvidenceRelationTargetV1::Proposition { proposition_id } => proposition_id,
        EvidenceRelationTargetV1::Evidence { .. } => {
            return Err(RelationQualificationError::DispositionRequiresPropositionRelation);
        }
    };
    if relation_proposition != &context_raw.proposition_id {
        return Err(RelationQualificationError::DeclarationPropositionMismatch);
    }

    if record.permitted_use != context_raw.use_case {
        return Err(RelationQualificationError::UseNotPermitted);
    }
    if !record.allowed_relation_kinds.contains(&relation.relation) {
        return Err(RelationQualificationError::RelationKindNotQualified {
            relation: relation.relation,
        });
    }
    if context_raw.now_unix_ms < record.qualified_at_unix_ms {
        return Err(RelationQualificationError::QualificationNotYetValid {
            qualified_at_unix_ms: record.qualified_at_unix_ms,
            now_unix_ms: context_raw.now_unix_ms,
        });
    }
    if context_raw.now_unix_ms > record.valid_until_unix_ms {
        return Err(RelationQualificationError::QualificationExpired {
            valid_until_unix_ms: record.valid_until_unix_ms,
            now_unix_ms: context_raw.now_unix_ms,
        });
    }

    let profile_contract_digest = relation_declaration_eligibility_profile_digest_v1();
    let context_commitment = context.commitment();
    let eligibility_id = relation_declaration_eligibility_id_v1(
        &profile_contract_digest,
        declaration.declaration_id(),
        qualification.qualification_id(),
        &context_commitment,
    );

    Ok(DispositionEligibleRelationDeclarationV1 {
        schema_version: RELATION_DECLARATION_ELIGIBILITY_SCHEMA_VERSION,
        profile: RELATION_DECLARATION_ELIGIBILITY_PROFILE_V1.to_string(),
        profile_contract_digest,
        eligibility_id,
        context_commitment,
        declaration: declaration.clone(),
        qualification: qualification.clone(),
    })
}

pub fn relation_declarer_qualification_profile_digest_v1() -> String {
    domain_hash(
        QUALIFICATION_PROFILE_DOMAIN,
        RELATION_DECLARER_QUALIFICATION_CONTRACT_V1.as_bytes(),
    )
}

pub fn relation_declaration_eligibility_profile_digest_v1() -> String {
    domain_hash(
        ELIGIBILITY_PROFILE_DOMAIN,
        RELATION_DECLARER_QUALIFICATION_CONTRACT_V1.as_bytes(),
    )
}

fn validate_and_canonicalize_qualification(
    value: &mut RelationDeclarerQualificationV1,
) -> Result<(), RelationQualificationError> {
    if value.schema_version != RELATION_DECLARER_QUALIFICATION_SCHEMA_VERSION {
        return Err(RelationQualificationError::UnsupportedSchemaVersion {
            found: value.schema_version,
        });
    }
    validate_nonempty(&value.subject_declarer_id, RelationQualificationError::MissingSubjectDeclarerId)?;
    validate_optional_nonempty(
        value.subject_declarer_version.as_deref(),
        RelationQualificationError::EmptySubjectDeclarerVersion,
    )?;
    validate_nonempty(&value.qualifier_id, RelationQualificationError::MissingQualifierId)?;
    validate_optional_nonempty(
        value.qualifier_version.as_deref(),
        RelationQualificationError::EmptyQualifierVersion,
    )?;
    validate_nonempty(&value.evaluator_id, RelationQualificationError::MissingEvaluatorId)?;
    validate_optional_nonempty(
        value.evaluator_version.as_deref(),
        RelationQualificationError::EmptyEvaluatorVersion,
    )?;
    if value.qualifier_id == value.subject_declarer_id {
        return Err(RelationQualificationError::SelfQualification);
    }
    validate_digest(&value.proposition_id)?;
    validate_digest(&value.qualification_policy_digest)?;
    validate_digest(&value.qualification_artifact_digest)?;
    if value.valid_until_unix_ms < value.qualified_at_unix_ms {
        return Err(RelationQualificationError::ValidityEndsBeforeQualification {
            qualified_at_unix_ms: value.qualified_at_unix_ms,
            valid_until_unix_ms: value.valid_until_unix_ms,
        });
    }
    if value.allowed_relation_kinds.is_empty() {
        return Err(RelationQualificationError::EmptyAllowedRelationKinds);
    }

    let mut seen = HashSet::with_capacity(value.allowed_relation_kinds.len());
    for kind in &value.allowed_relation_kinds {
        if *kind == EvidenceRelationKindV1::Supersedes {
            return Err(RelationQualificationError::SupersedesNotDispositionRelation);
        }
        if !seen.insert(*kind) {
            return Err(RelationQualificationError::DuplicateAllowedRelationKind {
                relation: *kind,
            });
        }
    }
    value.allowed_relation_kinds.sort_by_key(|kind| relation_kind_rank(*kind));
    Ok(())
}

fn relation_declarer_qualification_id_v1(
    profile_contract_digest: &str,
    value: &RelationDeclarerQualificationV1,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_ID_DOMAIN);
    hash_text(&mut hasher, b"profile_contract_digest", profile_contract_digest);
    hash_bytes(
        &mut hasher,
        b"schema_version",
        &value.schema_version.to_le_bytes(),
    );
    hash_text(&mut hasher, b"subject_declarer_id", &value.subject_declarer_id);
    hash_option_text(
        &mut hasher,
        b"subject_declarer_version",
        value.subject_declarer_version.as_deref(),
    );
    hash_text(
        &mut hasher,
        b"subject_method",
        declaration_method_tag(value.subject_method),
    );
    hash_text(&mut hasher, b"qualifier_id", &value.qualifier_id);
    hash_option_text(
        &mut hasher,
        b"qualifier_version",
        value.qualifier_version.as_deref(),
    );
    hash_text(&mut hasher, b"evaluator_id", &value.evaluator_id);
    hash_option_text(
        &mut hasher,
        b"evaluator_version",
        value.evaluator_version.as_deref(),
    );
    hash_text(&mut hasher, b"proposition_id", &value.proposition_id);
    hash_count(
        &mut hasher,
        b"allowed_relation_kind_count",
        value.allowed_relation_kinds.len(),
    );
    for kind in &value.allowed_relation_kinds {
        hash_text(
            &mut hasher,
            b"allowed_relation_kind",
            relation_kind_tag(*kind),
        );
    }
    hash_text(
        &mut hasher,
        b"permitted_use",
        relation_use_tag(value.permitted_use),
    );
    hash_text(
        &mut hasher,
        b"qualification_policy_digest",
        &value.qualification_policy_digest,
    );
    hash_text(
        &mut hasher,
        b"qualification_artifact_digest",
        &value.qualification_artifact_digest,
    );
    hash_bytes(
        &mut hasher,
        b"qualified_at_unix_ms",
        &value.qualified_at_unix_ms.to_le_bytes(),
    );
    hash_bytes(
        &mut hasher,
        b"valid_until_unix_ms",
        &value.valid_until_unix_ms.to_le_bytes(),
    );
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn eligibility_context_commitment_v1(
    context: &RelationDeclarationEligibilityContextV1,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(ELIGIBILITY_CONTEXT_DOMAIN);
    hash_text(
        &mut hasher,
        b"eligibility_profile_digest",
        &relation_declaration_eligibility_profile_digest_v1(),
    );
    hash_bytes(
        &mut hasher,
        b"schema_version",
        &context.schema_version.to_le_bytes(),
    );
    hash_text(&mut hasher, b"proposition_id", &context.proposition_id);
    hash_text(
        &mut hasher,
        b"use_case",
        relation_use_tag(context.use_case),
    );
    hash_bytes(
        &mut hasher,
        b"now_unix_ms",
        &context.now_unix_ms.to_le_bytes(),
    );
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn relation_declaration_eligibility_id_v1(
    profile_contract_digest: &str,
    declaration_id: &str,
    qualification_id: &str,
    context_commitment: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(ELIGIBILITY_ID_DOMAIN);
    hash_text(&mut hasher, b"profile_contract_digest", profile_contract_digest);
    hash_text(&mut hasher, b"declaration_id", declaration_id);
    hash_text(&mut hasher, b"qualification_id", qualification_id);
    hash_text(&mut hasher, b"context_commitment", context_commitment);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn relation_kind_rank(kind: EvidenceRelationKindV1) -> u8 {
    match kind {
        EvidenceRelationKindV1::Supports => 0,
        EvidenceRelationKindV1::Contradicts => 1,
        EvidenceRelationKindV1::Weakens => 2,
        EvidenceRelationKindV1::Defeats => 3,
        EvidenceRelationKindV1::Corroborates => 4,
        EvidenceRelationKindV1::Irrelevant => 5,
        EvidenceRelationKindV1::Supersedes => 6,
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

fn declaration_method_tag(method: RelationDeclarationMethodV1) -> &'static str {
    match method {
        RelationDeclarationMethodV1::HumanAnnotation => "human_annotation",
        RelationDeclarationMethodV1::DeterministicRule => "deterministic_rule",
        RelationDeclarationMethodV1::ModelInference => "model_inference",
        RelationDeclarationMethodV1::FormalProcedure => "formal_procedure",
        RelationDeclarationMethodV1::ImportedAssertion => "imported_assertion",
    }
}

fn relation_use_tag(use_case: RelationDeclarationUseV1) -> &'static str {
    match use_case {
        RelationDeclarationUseV1::ShadowRuntimeDisposition => "shadow_runtime_disposition",
    }
}

fn validate_nonempty(
    value: &str,
    error: RelationQualificationError,
) -> Result<(), RelationQualificationError> {
    if value.trim().is_empty() {
        Err(error)
    } else {
        Ok(())
    }
}

fn validate_optional_nonempty(
    value: Option<&str>,
    error: RelationQualificationError,
) -> Result<(), RelationQualificationError> {
    if value.is_some_and(|value| value.trim().is_empty()) {
        Err(error)
    } else {
        Ok(())
    }
}

fn validate_digest(digest: &str) -> Result<(), RelationQualificationError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(RelationQualificationError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(RelationQualificationError::MalformedDigest);
    }
    Ok(())
}

fn hash_count(hasher: &mut blake3::Hasher, label: &[u8], count: usize) {
    hash_bytes(hasher, label, &(count as u64).to_le_bytes());
}

fn hash_text(hasher: &mut blake3::Hasher, label: &[u8], value: &str) {
    hash_bytes(hasher, label, value.as_bytes());
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

fn hash_bytes(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationQualificationError {
    UnsupportedSchemaVersion { found: u16 },
    UnsupportedEligibilitySchemaVersion { found: u16 },
    UnexpectedQualificationProfile,
    QualificationIdentityMismatch,
    MissingSubjectDeclarerId,
    EmptySubjectDeclarerVersion,
    MissingQualifierId,
    EmptyQualifierVersion,
    MissingEvaluatorId,
    EmptyEvaluatorVersion,
    SelfQualification,
    MalformedDigest,
    ValidityEndsBeforeQualification {
        qualified_at_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
    EmptyAllowedRelationKinds,
    DuplicateAllowedRelationKind { relation: EvidenceRelationKindV1 },
    SupersedesNotDispositionRelation,
    DeclarationSubjectMismatch,
    QualificationPropositionMismatch,
    DeclarationPropositionMismatch,
    DispositionRequiresPropositionRelation,
    UseNotPermitted,
    RelationKindNotQualified { relation: EvidenceRelationKindV1 },
    QualificationNotYetValid {
        qualified_at_unix_ms: u64,
        now_unix_ms: u64,
    },
    QualificationExpired {
        valid_until_unix_ms: u64,
        now_unix_ms: u64,
    },
}

impl std::fmt::Display for RelationQualificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported relation-declarer qualification schema version {found}; expected {RELATION_DECLARER_QUALIFICATION_SCHEMA_VERSION}"
            ),
            Self::UnsupportedEligibilitySchemaVersion { found } => write!(
                f,
                "unsupported relation-declaration eligibility schema version {found}; expected {RELATION_DECLARATION_ELIGIBILITY_SCHEMA_VERSION}"
            ),
            Self::UnexpectedQualificationProfile => {
                f.write_str("unexpected relation-declarer qualification profile")
            }
            Self::QualificationIdentityMismatch => {
                f.write_str("registered qualification does not match its complete normalized record")
            }
            Self::MissingSubjectDeclarerId => f.write_str("qualification requires subject declarer id"),
            Self::EmptySubjectDeclarerVersion => f.write_str("subject declarer version cannot be empty when present"),
            Self::MissingQualifierId => f.write_str("qualification requires qualifier id"),
            Self::EmptyQualifierVersion => f.write_str("qualifier version cannot be empty when present"),
            Self::MissingEvaluatorId => f.write_str("qualification requires evaluator id"),
            Self::EmptyEvaluatorVersion => f.write_str("evaluator version cannot be empty when present"),
            Self::SelfQualification => f.write_str("declarer cannot be its own sole qualification authority"),
            Self::MalformedDigest => f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>"),
            Self::ValidityEndsBeforeQualification {
                qualified_at_unix_ms,
                valid_until_unix_ms,
            } => write!(
                f,
                "qualification validity {valid_until_unix_ms} precedes qualification time {qualified_at_unix_ms}"
            ),
            Self::EmptyAllowedRelationKinds => f.write_str("qualification requires at least one allowed relation kind"),
            Self::DuplicateAllowedRelationKind { relation } => write!(f, "duplicate allowed relation kind {relation:?}"),
            Self::SupersedesNotDispositionRelation => f.write_str("Supersedes targets evidence and is not a v1 shadow-disposition proposition relation"),
            Self::DeclarationSubjectMismatch => f.write_str("declaration provenance does not match qualified subject declarer/method"),
            Self::QualificationPropositionMismatch => f.write_str("qualification does not cover the requested proposition"),
            Self::DeclarationPropositionMismatch => f.write_str("declaration does not target the requested proposition"),
            Self::DispositionRequiresPropositionRelation => f.write_str("shadow runtime disposition requires a proposition-targeting relation"),
            Self::UseNotPermitted => f.write_str("qualification does not permit the requested use"),
            Self::RelationKindNotQualified { relation } => write!(f, "relation kind {relation:?} is not qualified for this declarer/use"),
            Self::QualificationNotYetValid {
                qualified_at_unix_ms,
                now_unix_ms,
            } => write!(f, "qualification begins at {qualified_at_unix_ms}, requested at {now_unix_ms}"),
            Self::QualificationExpired {
                valid_until_unix_ms,
                now_unix_ms,
            } => write!(f, "qualification expired at {valid_until_unix_ms}, requested at {now_unix_ms}"),
        }
    }
}

impl std::error::Error for RelationQualificationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        currentness::{
            EvidenceRelationV1, COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
        },
        relation_provenance::{
            EvidenceRelationDeclarationProvenanceV1,
            RELATION_DECLARATION_PROVENANCE_SCHEMA_VERSION,
        },
    };

    const PROPOSITION: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OTHER_PROPOSITION: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const RELATION_ID: &str =
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const EVIDENCE_ID: &str =
        "blake3:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const PROVENANCE: &str =
        "blake3:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const POLICY: &str =
        "blake3:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const ARTIFACT: &str =
        "blake3:1111111111111111111111111111111111111111111111111111111111111111";

    fn declaration(
        relation_kind: EvidenceRelationKindV1,
        declarer_id: &str,
    ) -> BoundEvidenceRelationDeclarationV1 {
        let relation = EvidenceRelationV1 {
            schema_version: COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
            relation_id: RELATION_ID.into(),
            evidence_id: EVIDENCE_ID.into(),
            relation: relation_kind,
            target: EvidenceRelationTargetV1::Proposition {
                proposition_id: PROPOSITION.into(),
            },
            strength_ppm: 700_000,
        }
        .validate()
        .unwrap();
        let provenance = EvidenceRelationDeclarationProvenanceV1 {
            schema_version: RELATION_DECLARATION_PROVENANCE_SCHEMA_VERSION,
            declarer_id: declarer_id.into(),
            declarer_version: Some("v1".into()),
            method: RelationDeclarationMethodV1::DeterministicRule,
            provenance_digest: PROVENANCE.into(),
        }
        .validate()
        .unwrap();
        BoundEvidenceRelationDeclarationV1::new(provenance, relation)
    }

    fn qualification(
        relation_kinds: Vec<EvidenceRelationKindV1>,
    ) -> RegisteredRelationDeclarerQualificationV1 {
        RelationDeclarerQualificationV1 {
            schema_version: RELATION_DECLARER_QUALIFICATION_SCHEMA_VERSION,
            subject_declarer_id: "rule-a".into(),
            subject_declarer_version: Some("v1".into()),
            subject_method: RelationDeclarationMethodV1::DeterministicRule,
            qualifier_id: "independent-qualifier".into(),
            qualifier_version: Some("2026-09".into()),
            evaluator_id: "relation-qualification-harness".into(),
            evaluator_version: Some("v1".into()),
            proposition_id: PROPOSITION.into(),
            allowed_relation_kinds: relation_kinds,
            permitted_use: RelationDeclarationUseV1::ShadowRuntimeDisposition,
            qualification_policy_digest: POLICY.into(),
            qualification_artifact_digest: ARTIFACT.into(),
            qualified_at_unix_ms: 100,
            valid_until_unix_ms: 200,
        }
        .register()
        .unwrap()
    }

    fn context(
        proposition_id: &str,
        now_unix_ms: u64,
    ) -> ValidatedRelationDeclarationEligibilityContextV1 {
        RelationDeclarationEligibilityContextV1 {
            schema_version: RELATION_DECLARATION_ELIGIBILITY_SCHEMA_VERSION,
            proposition_id: proposition_id.into(),
            use_case: RelationDeclarationUseV1::ShadowRuntimeDisposition,
            now_unix_ms,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn qualification_kind_set_is_canonical_not_input_order_identity() {
        let a = qualification(vec![
            EvidenceRelationKindV1::Defeats,
            EvidenceRelationKindV1::Supports,
        ]);
        let b = qualification(vec![
            EvidenceRelationKindV1::Supports,
            EvidenceRelationKindV1::Defeats,
        ]);
        assert_eq!(a, b);
        assert_eq!(a.qualification_id(), b.qualification_id());
    }

    #[test]
    fn producer_cannot_be_sole_qualifier() {
        let raw = RelationDeclarerQualificationV1 {
            schema_version: RELATION_DECLARER_QUALIFICATION_SCHEMA_VERSION,
            subject_declarer_id: "rule-a".into(),
            subject_declarer_version: Some("v1".into()),
            subject_method: RelationDeclarationMethodV1::DeterministicRule,
            qualifier_id: "rule-a".into(),
            qualifier_version: None,
            evaluator_id: "harness".into(),
            evaluator_version: None,
            proposition_id: PROPOSITION.into(),
            allowed_relation_kinds: vec![EvidenceRelationKindV1::Supports],
            permitted_use: RelationDeclarationUseV1::ShadowRuntimeDisposition,
            qualification_policy_digest: POLICY.into(),
            qualification_artifact_digest: ARTIFACT.into(),
            qualified_at_unix_ms: 100,
            valid_until_unix_ms: 200,
        };
        assert_eq!(raw.register(), Err(RelationQualificationError::SelfQualification));
    }

    #[test]
    fn registered_qualification_revalidates_after_persistence() {
        let registered = qualification(vec![EvidenceRelationKindV1::Supports]);
        let encoded = serde_json::to_string(&registered).unwrap();
        let decoded: RegisteredRelationDeclarerQualificationV1 =
            serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, registered);
    }

    #[test]
    fn qualification_tampering_fails_closed() {
        let registered = qualification(vec![EvidenceRelationKindV1::Supports]);
        let mut value = serde_json::to_value(&registered).unwrap();
        value["record"]["evaluator_id"] = serde_json::Value::String("forged".into());
        assert!(serde_json::from_value::<RegisteredRelationDeclarerQualificationV1>(value).is_err());
    }

    #[test]
    fn exact_join_issues_non_persisted_eligibility() {
        let declaration = declaration(EvidenceRelationKindV1::Supports, "rule-a");
        let qualification = qualification(vec![EvidenceRelationKindV1::Supports]);
        let eligible = admit_relation_declaration_for_use_v1(
            &declaration,
            &qualification,
            &context(PROPOSITION, 150),
        )
        .unwrap();
        assert_eq!(eligible.declaration().declaration_id(), declaration.declaration_id());
        assert_eq!(eligible.qualification().qualification_id(), qualification.qualification_id());
        assert_eq!(eligible.proposition_id(), PROPOSITION);
        assert!(eligible.eligibility_id().starts_with("blake3:"));
        let encoded = serde_json::to_string(&eligible).unwrap();
        assert!(encoded.contains(eligible.eligibility_id()));
    }

    #[test]
    fn wrong_subject_fails_exact_join() {
        let declaration = declaration(EvidenceRelationKindV1::Supports, "other-rule");
        let error = admit_relation_declaration_for_use_v1(
            &declaration,
            &qualification(vec![EvidenceRelationKindV1::Supports]),
            &context(PROPOSITION, 150),
        )
        .unwrap_err();
        assert_eq!(error, RelationQualificationError::DeclarationSubjectMismatch);
    }

    #[test]
    fn wrong_proposition_fails_exact_join() {
        let declaration = declaration(EvidenceRelationKindV1::Supports, "rule-a");
        let error = admit_relation_declaration_for_use_v1(
            &declaration,
            &qualification(vec![EvidenceRelationKindV1::Supports]),
            &context(OTHER_PROPOSITION, 150),
        )
        .unwrap_err();
        assert_eq!(error, RelationQualificationError::QualificationPropositionMismatch);
    }

    #[test]
    fn unqualified_relation_kind_cannot_become_eligible() {
        let declaration = declaration(EvidenceRelationKindV1::Defeats, "rule-a");
        let error = admit_relation_declaration_for_use_v1(
            &declaration,
            &qualification(vec![EvidenceRelationKindV1::Supports]),
            &context(PROPOSITION, 150),
        )
        .unwrap_err();
        assert_eq!(
            error,
            RelationQualificationError::RelationKindNotQualified {
                relation: EvidenceRelationKindV1::Defeats
            }
        );
    }

    #[test]
    fn expired_or_future_qualification_cannot_become_eligible() {
        let declaration = declaration(EvidenceRelationKindV1::Supports, "rule-a");
        let qualification = qualification(vec![EvidenceRelationKindV1::Supports]);
        assert!(matches!(
            admit_relation_declaration_for_use_v1(
                &declaration,
                &qualification,
                &context(PROPOSITION, 99),
            ),
            Err(RelationQualificationError::QualificationNotYetValid { .. })
        ));
        assert!(matches!(
            admit_relation_declaration_for_use_v1(
                &declaration,
                &qualification,
                &context(PROPOSITION, 201),
            ),
            Err(RelationQualificationError::QualificationExpired { .. })
        ));
    }

    #[test]
    fn supersedes_cannot_be_qualified_for_v1_disposition() {
        let mut raw = qualification(vec![EvidenceRelationKindV1::Supports])
            .record()
            .clone();
        raw.allowed_relation_kinds = vec![EvidenceRelationKindV1::Supersedes];
        assert_eq!(
            raw.register(),
            Err(RelationQualificationError::SupersedesNotDispositionRelation)
        );
    }

    #[test]
    fn qualification_identity_binds_policy_artifact_and_validity() {
        let a = qualification(vec![EvidenceRelationKindV1::Supports]);
        let mut changed = a.record().clone();
        changed.qualification_artifact_digest = PROVENANCE.into();
        let b = changed.register().unwrap();
        assert_ne!(a.qualification_id(), b.qualification_id());

        let mut changed = a.record().clone();
        changed.valid_until_unix_ms = 250;
        let c = changed.register().unwrap();
        assert_ne!(a.qualification_id(), c.qualification_id());
    }
}
