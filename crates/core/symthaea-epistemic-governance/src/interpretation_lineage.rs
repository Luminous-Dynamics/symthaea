// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Interpretation lineage and qualified interpretation-root independence.
//!
//! Evidence-root independence answers whether observations share evidence roots.
//! This module answers a different question: whether evidence -> proposition
//! judgments share an interpretation root. Distinct interpreter identities are
//! **not** automatically independent. Independence is established only when an
//! exact, current, proposition/use-scoped qualification joins the root pair.
//!
//! The graph is root-normalized: declarations map to interpretation roots, each
//! root is represented once, and each distinct unordered root pair is assessed
//! exactly once. Same-root declarations share identity; they do not create a
//! synthetic same-root edge. Edge count is not an independent-root-set witness.

use crate::{
    relation_provenance::RelationDeclarationMethodV1,
    relation_qualification::{
        DispositionEligibleRelationDeclarationV1, RelationDeclarationUseV1,
        ValidatedRelationDeclarationEligibilityContextV1,
    },
};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet};

pub const INTERPRETATION_LINEAGE_SCHEMA_VERSION: u16 = 1;
pub const INTERPRETATION_LINEAGE_PROFILE_V1: &str = "rca-interpretation-lineage-v1";
pub const INTERPRETATION_INDEPENDENCE_QUALIFICATION_SCHEMA_VERSION: u16 = 1;
pub const INTERPRETATION_INDEPENDENCE_QUALIFICATION_PROFILE_V1: &str =
    "rca-interpretation-independence-qualification-v1";

pub const INTERPRETATION_LINEAGE_CONTRACT_V1: &str = concat!(
    "rca-interpretation-lineage-v1\n",
    "input=currently_eligible_relation_declarations+one_exact_eligibility_context\n",
    "interpretation_root=declarer_id+optional_declarer_version+declaration_method\n",
    "same_declarer_version_method=one_interpretation_root\n",
    "graph_is_normalized_to_unique_interpretation_roots\n",
    "same_root_declarations_do_not_create_pair_edges\n",
    "each_distinct_unordered_root_pair_is_assessed_exactly_once\n",
    "distinct_interpretation_roots_do_not_imply_independence\n",
    "distinct_roots_without_exact_current_qualification=independence_unknown\n",
    "independence_qualified_only_by_exact_root_pair+proposition+use+time_join\n",
    "all_eligible_declarations_must_share_exact_context_commitment\n",
    "lineage_identity=blake3_explicit_entries+roots+root_pair_assessments+context\n",
    "issued_lineage=is_private_non_deserializable_shadow_report\n",
    "root_pair_topology_is_exposed_but_qualified_pair_count_is_not_an_api\n",
    "interpretation_independence_is_not_truth_or_evidence_independence\n",
    "lineage_is_not_belief_workspace_action_or_promotion_authority\n",
);

pub const INTERPRETATION_INDEPENDENCE_QUALIFICATION_CONTRACT_V1: &str = concat!(
    "rca-interpretation-independence-qualification-v1\n",
    "subject=canonical_unordered_pair_of_distinct_interpretation_root_ids\n",
    "scope=exact_proposition_id+exact_permitted_use\n",
    "qualification=qualifier+evaluator+policy_digest+artifact_digest\n",
    "validity=qualified_at_unix_ms<=now<=valid_until_unix_ms\n",
    "registration_is_persistable_and_revalidated\n",
    "registration_does_not_by_itself_make_roots_independent\n",
    "assembly_rejects_direct_self_qualification_by_either_root_owner\n",
    "root_pair_order_does_not_change_qualification_identity\n",
    "qualification_is_not_truth_belief_action_or_promotion_authority\n",
);

const LINEAGE_PROFILE_DOMAIN: &[u8] = b"symthaea:rca-interpretation-lineage-contract:v1\0";
const ROOT_ID_DOMAIN: &[u8] = b"symthaea:rca-interpretation-root:v1\0";
const LINEAGE_ID_DOMAIN: &[u8] = b"symthaea:rca-interpretation-lineage:v1\0";
const INDEPENDENCE_PROFILE_DOMAIN: &[u8] =
    b"symthaea:rca-interpretation-independence-qualification-contract:v1\0";
const INDEPENDENCE_ID_DOMAIN: &[u8] =
    b"symthaea:rca-interpretation-independence-qualification:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpretationIndependenceStatusV1 {
    DistinctRootsIndependenceUnknown,
    IndependenceQualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterpretationIndependenceQualificationV1 {
    pub schema_version: u16,
    pub left_interpretation_root_id: String,
    pub right_interpretation_root_id: String,
    pub proposition_id: String,
    pub permitted_use: RelationDeclarationUseV1,
    pub qualifier_id: String,
    pub qualifier_version: Option<String>,
    pub evaluator_id: String,
    pub evaluator_version: Option<String>,
    pub qualification_policy_digest: String,
    pub qualification_artifact_digest: String,
    pub qualified_at_unix_ms: u64,
    pub valid_until_unix_ms: u64,
}

impl InterpretationIndependenceQualificationV1 {
    pub fn register(
        self,
    ) -> Result<RegisteredInterpretationIndependenceQualificationV1, InterpretationLineageError>
    {
        RegisteredInterpretationIndependenceQualificationV1::try_from(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredInterpretationIndependenceQualificationV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    qualification_id: String,
    record: InterpretationIndependenceQualificationV1,
}

impl RegisteredInterpretationIndependenceQualificationV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn profile_contract_digest(&self) -> &str {
        &self.profile_contract_digest
    }

    pub fn qualification_id(&self) -> &str {
        &self.qualification_id
    }

    pub fn record(&self) -> &InterpretationIndependenceQualificationV1 {
        &self.record
    }
}

impl TryFrom<InterpretationIndependenceQualificationV1>
    for RegisteredInterpretationIndependenceQualificationV1
{
    type Error = InterpretationLineageError;

    fn try_from(mut value: InterpretationIndependenceQualificationV1) -> Result<Self, Self::Error> {
        validate_and_canonicalize_independence_qualification(&mut value)?;
        let profile_contract_digest = interpretation_independence_profile_digest_v1();
        let qualification_id =
            interpretation_independence_qualification_id_v1(&profile_contract_digest, &value);
        Ok(Self {
            schema_version: INTERPRETATION_INDEPENDENCE_QUALIFICATION_SCHEMA_VERSION,
            profile: INTERPRETATION_INDEPENDENCE_QUALIFICATION_PROFILE_V1.to_string(),
            profile_contract_digest,
            qualification_id,
            record: value,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisteredInterpretationIndependenceQualificationWireV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    qualification_id: String,
    record: InterpretationIndependenceQualificationV1,
}

impl<'de> Deserialize<'de> for RegisteredInterpretationIndependenceQualificationV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RegisteredInterpretationIndependenceQualificationWireV1::deserialize(deserializer)?;
        if wire.schema_version != INTERPRETATION_INDEPENDENCE_QUALIFICATION_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(
                InterpretationLineageError::UnsupportedIndependenceQualificationSchemaVersion {
                    found: wire.schema_version,
                },
            ));
        }
        if wire.profile != INTERPRETATION_INDEPENDENCE_QUALIFICATION_PROFILE_V1 {
            return Err(serde::de::Error::custom(
                InterpretationLineageError::UnexpectedIndependenceQualificationProfile,
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
                InterpretationLineageError::IndependenceQualificationIdentityMismatch,
            ));
        }
        Ok(expected)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InterpretationLineageEntryV1 {
    declaration_id: String,
    eligibility_id: String,
    interpretation_root_id: String,
}

impl InterpretationLineageEntryV1 {
    pub fn declaration_id(&self) -> &str {
        &self.declaration_id
    }

    pub fn eligibility_id(&self) -> &str {
        &self.eligibility_id
    }

    pub fn interpretation_root_id(&self) -> &str {
        &self.interpretation_root_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InterpretationRootV1 {
    interpretation_root_id: String,
    declarer_id: String,
    declarer_version: Option<String>,
    declaration_method: RelationDeclarationMethodV1,
}

impl InterpretationRootV1 {
    pub fn interpretation_root_id(&self) -> &str {
        &self.interpretation_root_id
    }

    pub fn declarer_id(&self) -> &str {
        &self.declarer_id
    }

    pub fn declarer_version(&self) -> Option<&str> {
        self.declarer_version.as_deref()
    }

    pub const fn declaration_method(&self) -> RelationDeclarationMethodV1 {
        self.declaration_method
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InterpretationRootPairAssessmentV1 {
    left_interpretation_root_id: String,
    right_interpretation_root_id: String,
    status: InterpretationIndependenceStatusV1,
    qualification_id: Option<String>,
}

impl InterpretationRootPairAssessmentV1 {
    pub fn left_interpretation_root_id(&self) -> &str {
        &self.left_interpretation_root_id
    }

    pub fn right_interpretation_root_id(&self) -> &str {
        &self.right_interpretation_root_id
    }

    pub const fn status(&self) -> InterpretationIndependenceStatusV1 {
        self.status
    }

    pub fn qualification_id(&self) -> Option<&str> {
        self.qualification_id.as_deref()
    }
}

#[must_use = "interpretation lineage is shadow epistemic evidence and should be inspected"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InterpretationLineageV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    lineage_id: String,
    proposition_id: String,
    eligibility_context_commitment: String,
    entries: Vec<InterpretationLineageEntryV1>,
    roots: Vec<InterpretationRootV1>,
    root_pair_assessments: Vec<InterpretationRootPairAssessmentV1>,
}

impl InterpretationLineageV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn profile_contract_digest(&self) -> &str {
        &self.profile_contract_digest
    }

    pub fn lineage_id(&self) -> &str {
        &self.lineage_id
    }

    pub fn proposition_id(&self) -> &str {
        &self.proposition_id
    }

    pub fn eligibility_context_commitment(&self) -> &str {
        &self.eligibility_context_commitment
    }

    /// Canonical declaration -> root mappings, ordered by declaration id.
    pub fn entries(&self) -> &[InterpretationLineageEntryV1] {
        &self.entries
    }

    /// Canonical unique interpretation roots, ordered by root id.
    pub fn roots(&self) -> &[InterpretationRootV1] {
        &self.roots
    }

    /// Exact unique-root pair topology only. There is deliberately no public
    /// helper that counts qualified edges: edge count is not an independent-root
    /// set proof.
    pub fn root_pair_assessments(&self) -> &[InterpretationRootPairAssessmentV1] {
        &self.root_pair_assessments
    }
}

pub fn interpretation_lineage_profile_digest_v1() -> String {
    domain_hash(
        LINEAGE_PROFILE_DOMAIN,
        INTERPRETATION_LINEAGE_CONTRACT_V1.as_bytes(),
    )
}

pub fn interpretation_independence_profile_digest_v1() -> String {
    domain_hash(
        INDEPENDENCE_PROFILE_DOMAIN,
        INTERPRETATION_INDEPENDENCE_QUALIFICATION_CONTRACT_V1.as_bytes(),
    )
}

pub fn assemble_interpretation_lineage_v1(
    eligible_declarations: &[DispositionEligibleRelationDeclarationV1],
    eligibility_context: &ValidatedRelationDeclarationEligibilityContextV1,
    independence_qualifications: &[RegisteredInterpretationIndependenceQualificationV1],
) -> Result<InterpretationLineageV1, InterpretationLineageError> {
    if eligible_declarations.is_empty() {
        return Err(InterpretationLineageError::EmptyEligibleDeclarationSet);
    }

    let context_raw = eligibility_context.as_raw();
    let context_commitment = eligibility_context.commitment();
    let mut seen_declarations = HashSet::with_capacity(eligible_declarations.len());
    let mut seen_eligibility_ids = HashSet::with_capacity(eligible_declarations.len());
    let mut entries = Vec::with_capacity(eligible_declarations.len());
    let mut roots_by_id: HashMap<String, InterpretationRootV1> = HashMap::new();

    for eligible in eligible_declarations {
        if eligible.proposition_id() != context_raw.proposition_id {
            return Err(InterpretationLineageError::EligiblePropositionMismatch);
        }
        if eligible.context_commitment() != context_commitment {
            return Err(InterpretationLineageError::EligibilityContextMismatch);
        }
        let declaration_id = eligible.declaration().declaration_id();
        if !seen_declarations.insert(declaration_id.to_string()) {
            return Err(InterpretationLineageError::DuplicateDeclarationId {
                declaration_id: declaration_id.to_string(),
            });
        }
        if !seen_eligibility_ids.insert(eligible.eligibility_id().to_string()) {
            return Err(InterpretationLineageError::DuplicateEligibilityId {
                eligibility_id: eligible.eligibility_id().to_string(),
            });
        }

        let provenance = eligible.declaration().provenance().as_raw();
        let interpretation_root_id = interpretation_root_id_v1(
            &provenance.declarer_id,
            provenance.declarer_version.as_deref(),
            provenance.method,
        );
        let root = InterpretationRootV1 {
            interpretation_root_id: interpretation_root_id.clone(),
            declarer_id: provenance.declarer_id.clone(),
            declarer_version: provenance.declarer_version.clone(),
            declaration_method: provenance.method,
        };
        match roots_by_id.get(&interpretation_root_id) {
            None => {
                roots_by_id.insert(interpretation_root_id.clone(), root);
            }
            Some(existing) if existing == &root => {}
            Some(_) => return Err(InterpretationLineageError::InterpretationRootIdentityCollision),
        }
        entries.push(InterpretationLineageEntryV1 {
            declaration_id: declaration_id.to_string(),
            eligibility_id: eligible.eligibility_id().to_string(),
            interpretation_root_id,
        });
    }
    entries.sort_by(|a, b| a.declaration_id.cmp(&b.declaration_id));

    let mut roots = roots_by_id.into_values().collect::<Vec<_>>();
    roots.sort_by(|a, b| a.interpretation_root_id.cmp(&b.interpretation_root_id));
    let root_lookup: HashMap<&str, &InterpretationRootV1> = roots
        .iter()
        .map(|root| (root.interpretation_root_id.as_str(), root))
        .collect();

    let mut qualifications_by_pair: HashMap<
        (&str, &str),
        &RegisteredInterpretationIndependenceQualificationV1,
    > = HashMap::with_capacity(independence_qualifications.len());

    for qualification in independence_qualifications {
        let record = qualification.record();
        if record.proposition_id != context_raw.proposition_id {
            return Err(InterpretationLineageError::IndependenceQualificationPropositionMismatch);
        }
        if record.permitted_use != context_raw.use_case {
            return Err(InterpretationLineageError::IndependenceQualificationUseMismatch);
        }
        let Some(left_owner) = root_lookup.get(record.left_interpretation_root_id.as_str()) else {
            return Err(InterpretationLineageError::UnexpectedIndependenceQualificationPair);
        };
        let Some(right_owner) = root_lookup.get(record.right_interpretation_root_id.as_str()) else {
            return Err(InterpretationLineageError::UnexpectedIndependenceQualificationPair);
        };
        if context_raw.now_unix_ms < record.qualified_at_unix_ms {
            return Err(InterpretationLineageError::IndependenceQualificationNotYetValid {
                qualified_at_unix_ms: record.qualified_at_unix_ms,
                now_unix_ms: context_raw.now_unix_ms,
            });
        }
        if context_raw.now_unix_ms > record.valid_until_unix_ms {
            return Err(InterpretationLineageError::IndependenceQualificationExpired {
                valid_until_unix_ms: record.valid_until_unix_ms,
                now_unix_ms: context_raw.now_unix_ms,
            });
        }
        if record.qualifier_id == left_owner.declarer_id
            || record.qualifier_id == right_owner.declarer_id
        {
            return Err(InterpretationLineageError::DirectInterpretationSelfQualification);
        }
        let key = (
            record.left_interpretation_root_id.as_str(),
            record.right_interpretation_root_id.as_str(),
        );
        if qualifications_by_pair.insert(key, qualification).is_some() {
            return Err(InterpretationLineageError::DuplicateIndependenceQualificationPair);
        }
    }

    let mut root_pair_assessments = Vec::with_capacity(pair_count(roots.len()));
    for left_index in 0..roots.len() {
        for right_index in (left_index + 1)..roots.len() {
            let left = &roots[left_index];
            let right = &roots[right_index];
            debug_assert!(left.interpretation_root_id < right.interpretation_root_id);
            let qualification = qualifications_by_pair
                .get(&(
                    left.interpretation_root_id.as_str(),
                    right.interpretation_root_id.as_str(),
                ))
                .copied();
            root_pair_assessments.push(InterpretationRootPairAssessmentV1 {
                left_interpretation_root_id: left.interpretation_root_id.clone(),
                right_interpretation_root_id: right.interpretation_root_id.clone(),
                status: if qualification.is_some() {
                    InterpretationIndependenceStatusV1::IndependenceQualified
                } else {
                    InterpretationIndependenceStatusV1::DistinctRootsIndependenceUnknown
                },
                qualification_id: qualification.map(|value| value.qualification_id().to_string()),
            });
        }
    }

    let profile_contract_digest = interpretation_lineage_profile_digest_v1();
    let lineage_id = interpretation_lineage_id_v1(
        &profile_contract_digest,
        &context_raw.proposition_id,
        &context_commitment,
        &entries,
        &roots,
        &root_pair_assessments,
    );

    Ok(InterpretationLineageV1 {
        schema_version: INTERPRETATION_LINEAGE_SCHEMA_VERSION,
        profile: INTERPRETATION_LINEAGE_PROFILE_V1.to_string(),
        profile_contract_digest,
        lineage_id,
        proposition_id: context_raw.proposition_id.clone(),
        eligibility_context_commitment: context_commitment,
        entries,
        roots,
        root_pair_assessments,
    })
}

fn validate_and_canonicalize_independence_qualification(
    value: &mut InterpretationIndependenceQualificationV1,
) -> Result<(), InterpretationLineageError> {
    if value.schema_version != INTERPRETATION_INDEPENDENCE_QUALIFICATION_SCHEMA_VERSION {
        return Err(
            InterpretationLineageError::UnsupportedIndependenceQualificationSchemaVersion {
                found: value.schema_version,
            },
        );
    }
    validate_digest(&value.left_interpretation_root_id)?;
    validate_digest(&value.right_interpretation_root_id)?;
    if value.left_interpretation_root_id == value.right_interpretation_root_id {
        return Err(InterpretationLineageError::SameRootCannotBeQualifiedIndependent);
    }
    if value.left_interpretation_root_id > value.right_interpretation_root_id {
        std::mem::swap(
            &mut value.left_interpretation_root_id,
            &mut value.right_interpretation_root_id,
        );
    }
    validate_digest(&value.proposition_id)?;
    validate_nonempty(&value.qualifier_id, InterpretationLineageError::MissingQualifierId)?;
    validate_optional_nonempty(
        value.qualifier_version.as_deref(),
        InterpretationLineageError::EmptyQualifierVersion,
    )?;
    validate_nonempty(&value.evaluator_id, InterpretationLineageError::MissingEvaluatorId)?;
    validate_optional_nonempty(
        value.evaluator_version.as_deref(),
        InterpretationLineageError::EmptyEvaluatorVersion,
    )?;
    validate_digest(&value.qualification_policy_digest)?;
    validate_digest(&value.qualification_artifact_digest)?;
    if value.valid_until_unix_ms < value.qualified_at_unix_ms {
        return Err(InterpretationLineageError::ValidityEndsBeforeQualification {
            qualified_at_unix_ms: value.qualified_at_unix_ms,
            valid_until_unix_ms: value.valid_until_unix_ms,
        });
    }
    Ok(())
}

fn interpretation_root_id_v1(
    declarer_id: &str,
    declarer_version: Option<&str>,
    method: RelationDeclarationMethodV1,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(ROOT_ID_DOMAIN);
    hash_text(
        &mut hasher,
        b"lineage_profile_digest",
        &interpretation_lineage_profile_digest_v1(),
    );
    hash_text(&mut hasher, b"declarer_id", declarer_id);
    hash_option_text(&mut hasher, b"declarer_version", declarer_version);
    hash_text(
        &mut hasher,
        b"declaration_method",
        declaration_method_tag(method),
    );
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn interpretation_independence_qualification_id_v1(
    profile_contract_digest: &str,
    value: &InterpretationIndependenceQualificationV1,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(INDEPENDENCE_ID_DOMAIN);
    hash_text(
        &mut hasher,
        b"profile_contract_digest",
        profile_contract_digest,
    );
    hash_bytes(
        &mut hasher,
        b"schema_version",
        &value.schema_version.to_le_bytes(),
    );
    hash_text(
        &mut hasher,
        b"left_interpretation_root_id",
        &value.left_interpretation_root_id,
    );
    hash_text(
        &mut hasher,
        b"right_interpretation_root_id",
        &value.right_interpretation_root_id,
    );
    hash_text(&mut hasher, b"proposition_id", &value.proposition_id);
    hash_text(
        &mut hasher,
        b"permitted_use",
        relation_use_tag(value.permitted_use),
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

fn interpretation_lineage_id_v1(
    profile_contract_digest: &str,
    proposition_id: &str,
    context_commitment: &str,
    entries: &[InterpretationLineageEntryV1],
    roots: &[InterpretationRootV1],
    root_pairs: &[InterpretationRootPairAssessmentV1],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(LINEAGE_ID_DOMAIN);
    hash_text(
        &mut hasher,
        b"profile_contract_digest",
        profile_contract_digest,
    );
    hash_bytes(
        &mut hasher,
        b"schema_version",
        &INTERPRETATION_LINEAGE_SCHEMA_VERSION.to_le_bytes(),
    );
    hash_text(&mut hasher, b"proposition_id", proposition_id);
    hash_text(
        &mut hasher,
        b"eligibility_context_commitment",
        context_commitment,
    );
    hash_count(&mut hasher, b"entry_count", entries.len());
    for entry in entries {
        hash_text(&mut hasher, b"declaration_id", &entry.declaration_id);
        hash_text(&mut hasher, b"eligibility_id", &entry.eligibility_id);
        hash_text(
            &mut hasher,
            b"entry_interpretation_root_id",
            &entry.interpretation_root_id,
        );
    }
    hash_count(&mut hasher, b"root_count", roots.len());
    for root in roots {
        hash_text(
            &mut hasher,
            b"interpretation_root_id",
            &root.interpretation_root_id,
        );
        hash_text(&mut hasher, b"declarer_id", &root.declarer_id);
        hash_option_text(
            &mut hasher,
            b"declarer_version",
            root.declarer_version.as_deref(),
        );
        hash_text(
            &mut hasher,
            b"declaration_method",
            declaration_method_tag(root.declaration_method),
        );
    }
    hash_count(&mut hasher, b"root_pair_count", root_pairs.len());
    for pair in root_pairs {
        hash_text(
            &mut hasher,
            b"left_interpretation_root_id",
            &pair.left_interpretation_root_id,
        );
        hash_text(
            &mut hasher,
            b"right_interpretation_root_id",
            &pair.right_interpretation_root_id,
        );
        hash_text(
            &mut hasher,
            b"independence_status",
            independence_status_tag(pair.status),
        );
        hash_option_text(
            &mut hasher,
            b"independence_qualification_id",
            pair.qualification_id.as_deref(),
        );
    }
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

fn relation_use_tag(use_case: RelationDeclarationUseV1) -> &'static str {
    match use_case {
        RelationDeclarationUseV1::ShadowRuntimeDisposition => "shadow_runtime_disposition",
    }
}

fn independence_status_tag(status: InterpretationIndependenceStatusV1) -> &'static str {
    match status {
        InterpretationIndependenceStatusV1::DistinctRootsIndependenceUnknown => {
            "distinct_roots_independence_unknown"
        }
        InterpretationIndependenceStatusV1::IndependenceQualified => "independence_qualified",
    }
}

fn pair_count(root_count: usize) -> usize {
    root_count.saturating_mul(root_count.saturating_sub(1)) / 2
}

fn validate_nonempty(
    value: &str,
    error: InterpretationLineageError,
) -> Result<(), InterpretationLineageError> {
    if value.trim().is_empty() {
        Err(error)
    } else {
        Ok(())
    }
}

fn validate_optional_nonempty(
    value: Option<&str>,
    error: InterpretationLineageError,
) -> Result<(), InterpretationLineageError> {
    if value.is_some_and(|value| value.trim().is_empty()) {
        Err(error)
    } else {
        Ok(())
    }
}

fn validate_digest(digest: &str) -> Result<(), InterpretationLineageError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(InterpretationLineageError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(InterpretationLineageError::MalformedDigest);
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
pub enum InterpretationLineageError {
    UnsupportedIndependenceQualificationSchemaVersion { found: u16 },
    UnexpectedIndependenceQualificationProfile,
    IndependenceQualificationIdentityMismatch,
    MalformedDigest,
    MissingQualifierId,
    EmptyQualifierVersion,
    MissingEvaluatorId,
    EmptyEvaluatorVersion,
    SameRootCannotBeQualifiedIndependent,
    ValidityEndsBeforeQualification {
        qualified_at_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
    EmptyEligibleDeclarationSet,
    EligiblePropositionMismatch,
    EligibilityContextMismatch,
    DuplicateDeclarationId { declaration_id: String },
    DuplicateEligibilityId { eligibility_id: String },
    InterpretationRootIdentityCollision,
    IndependenceQualificationPropositionMismatch,
    IndependenceQualificationUseMismatch,
    UnexpectedIndependenceQualificationPair,
    DuplicateIndependenceQualificationPair,
    DirectInterpretationSelfQualification,
    IndependenceQualificationNotYetValid {
        qualified_at_unix_ms: u64,
        now_unix_ms: u64,
    },
    IndependenceQualificationExpired {
        valid_until_unix_ms: u64,
        now_unix_ms: u64,
    },
}

impl std::fmt::Display for InterpretationLineageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedIndependenceQualificationSchemaVersion { found } => write!(
                f,
                "unsupported interpretation-independence qualification schema version {found}; expected {INTERPRETATION_INDEPENDENCE_QUALIFICATION_SCHEMA_VERSION}"
            ),
            Self::UnexpectedIndependenceQualificationProfile => {
                f.write_str("unexpected interpretation-independence qualification profile")
            }
            Self::IndependenceQualificationIdentityMismatch => {
                f.write_str("interpretation-independence qualification identity mismatch")
            }
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::MissingQualifierId => {
                f.write_str("independence qualification requires qualifier id")
            }
            Self::EmptyQualifierVersion => {
                f.write_str("qualifier version cannot be empty when present")
            }
            Self::MissingEvaluatorId => {
                f.write_str("independence qualification requires evaluator id")
            }
            Self::EmptyEvaluatorVersion => {
                f.write_str("evaluator version cannot be empty when present")
            }
            Self::SameRootCannotBeQualifiedIndependent => f.write_str(
                "the same interpretation root cannot be qualified independent from itself",
            ),
            Self::ValidityEndsBeforeQualification {
                qualified_at_unix_ms,
                valid_until_unix_ms,
            } => write!(
                f,
                "independence validity {valid_until_unix_ms} precedes qualification time {qualified_at_unix_ms}"
            ),
            Self::EmptyEligibleDeclarationSet => {
                f.write_str("interpretation lineage requires at least one eligible declaration")
            }
            Self::EligiblePropositionMismatch => {
                f.write_str("eligible declaration proposition differs from lineage context")
            }
            Self::EligibilityContextMismatch => f.write_str(
                "eligible declarations must share the exact lineage eligibility context",
            ),
            Self::DuplicateDeclarationId { declaration_id } => write!(
                f,
                "duplicate declaration id in interpretation lineage: {declaration_id}"
            ),
            Self::DuplicateEligibilityId { eligibility_id } => write!(
                f,
                "duplicate eligibility id in interpretation lineage: {eligibility_id}"
            ),
            Self::InterpretationRootIdentityCollision => {
                f.write_str("interpretation root identity resolved to inconsistent owner metadata")
            }
            Self::IndependenceQualificationPropositionMismatch => f.write_str(
                "interpretation-independence qualification proposition mismatch",
            ),
            Self::IndependenceQualificationUseMismatch => {
                f.write_str("interpretation-independence qualification use mismatch")
            }
            Self::UnexpectedIndependenceQualificationPair => f.write_str(
                "independence qualification references roots outside this lineage",
            ),
            Self::DuplicateIndependenceQualificationPair => f.write_str(
                "duplicate interpretation-independence qualification for root pair",
            ),
            Self::DirectInterpretationSelfQualification => f.write_str(
                "an interpretation root owner cannot directly qualify its own independence pair",
            ),
            Self::IndependenceQualificationNotYetValid {
                qualified_at_unix_ms,
                now_unix_ms,
            } => write!(
                f,
                "interpretation-independence qualification begins at {qualified_at_unix_ms}, requested at {now_unix_ms}"
            ),
            Self::IndependenceQualificationExpired {
                valid_until_unix_ms,
                now_unix_ms,
            } => write!(
                f,
                "interpretation-independence qualification expired at {valid_until_unix_ms}, requested at {now_unix_ms}"
            ),
        }
    }
}

impl std::error::Error for InterpretationLineageError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        currentness::{
            EvidenceRelationKindV1, EvidenceRelationTargetV1, EvidenceRelationV1,
            COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
        },
        relation_provenance::{
            BoundEvidenceRelationDeclarationV1, EvidenceRelationDeclarationProvenanceV1,
            RELATION_DECLARATION_PROVENANCE_SCHEMA_VERSION,
        },
        relation_qualification::{
            admit_relation_declaration_for_use_v1, RelationDeclarationEligibilityContextV1,
            RelationDeclarerQualificationV1,
            RELATION_DECLARATION_ELIGIBILITY_SCHEMA_VERSION,
            RELATION_DECLARER_QUALIFICATION_SCHEMA_VERSION,
        },
    };

    const PROPOSITION: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const POLICY: &str =
        "blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const ARTIFACT: &str =
        "blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn context(now: u64) -> ValidatedRelationDeclarationEligibilityContextV1 {
        RelationDeclarationEligibilityContextV1 {
            schema_version: RELATION_DECLARATION_ELIGIBILITY_SCHEMA_VERSION,
            proposition_id: PROPOSITION.into(),
            use_case: RelationDeclarationUseV1::ShadowRuntimeDisposition,
            now_unix_ms: now,
        }
        .validate()
        .unwrap()
    }

    fn eligible(
        declarer_id: &str,
        declarer_version: &str,
        evidence_fill: char,
        relation_fill: char,
        now: u64,
    ) -> DispositionEligibleRelationDeclarationV1 {
        let evidence_id = format!("blake3:{}", evidence_fill.to_string().repeat(64));
        let relation_id = format!("sha256:{}", relation_fill.to_string().repeat(64));
        let provenance_digest = format!("blake3:{}", relation_fill.to_string().repeat(64));
        let relation = EvidenceRelationV1 {
            schema_version: COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
            relation_id,
            evidence_id,
            relation: EvidenceRelationKindV1::Supports,
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
            declarer_version: Some(declarer_version.into()),
            method: RelationDeclarationMethodV1::DeterministicRule,
            provenance_digest,
        }
        .validate()
        .unwrap();
        let declaration = BoundEvidenceRelationDeclarationV1::new(provenance, relation);
        let qualification = RelationDeclarerQualificationV1 {
            schema_version: RELATION_DECLARER_QUALIFICATION_SCHEMA_VERSION,
            subject_declarer_id: declarer_id.into(),
            subject_declarer_version: Some(declarer_version.into()),
            subject_method: RelationDeclarationMethodV1::DeterministicRule,
            qualifier_id: "declarer-qualification-authority".into(),
            qualifier_version: Some("v1".into()),
            evaluator_id: "declarer-qualification-harness".into(),
            evaluator_version: Some("v1".into()),
            proposition_id: PROPOSITION.into(),
            allowed_relation_kinds: vec![EvidenceRelationKindV1::Supports],
            permitted_use: RelationDeclarationUseV1::ShadowRuntimeDisposition,
            qualification_policy_digest: POLICY.into(),
            qualification_artifact_digest: ARTIFACT.into(),
            qualified_at_unix_ms: 100,
            valid_until_unix_ms: 300,
        }
        .register()
        .unwrap();
        admit_relation_declaration_for_use_v1(&declaration, &qualification, &context(now)).unwrap()
    }

    fn independence_qualification(
        left_root: &str,
        right_root: &str,
        qualifier_id: &str,
    ) -> RegisteredInterpretationIndependenceQualificationV1 {
        InterpretationIndependenceQualificationV1 {
            schema_version: INTERPRETATION_INDEPENDENCE_QUALIFICATION_SCHEMA_VERSION,
            left_interpretation_root_id: left_root.into(),
            right_interpretation_root_id: right_root.into(),
            proposition_id: PROPOSITION.into(),
            permitted_use: RelationDeclarationUseV1::ShadowRuntimeDisposition,
            qualifier_id: qualifier_id.into(),
            qualifier_version: Some("v1".into()),
            evaluator_id: "interpretation-independence-harness".into(),
            evaluator_version: Some("v1".into()),
            qualification_policy_digest: POLICY.into(),
            qualification_artifact_digest: ARTIFACT.into(),
            qualified_at_unix_ms: 100,
            valid_until_unix_ms: 300,
        }
        .register()
        .unwrap()
    }

    #[test]
    fn same_declarer_version_method_is_one_interpretation_root() {
        let a = eligible("rule-a", "v1", '1', '2', 150);
        let b = eligible("rule-a", "v1", '3', '4', 150);
        let lineage = assemble_interpretation_lineage_v1(&[a, b], &context(150), &[]).unwrap();
        assert_eq!(lineage.entries().len(), 2);
        assert_eq!(lineage.roots().len(), 1);
        assert_eq!(
            lineage.entries()[0].interpretation_root_id(),
            lineage.entries()[1].interpretation_root_id()
        );
        assert!(lineage.root_pair_assessments().is_empty());
    }

    #[test]
    fn distinct_roots_default_to_independence_unknown() {
        let a = eligible("rule-a", "v1", '1', '2', 150);
        let b = eligible("rule-b", "v1", '3', '4', 150);
        let lineage = assemble_interpretation_lineage_v1(&[a, b], &context(150), &[]).unwrap();
        assert_eq!(lineage.roots().len(), 2);
        assert_eq!(lineage.root_pair_assessments().len(), 1);
        assert_eq!(
            lineage.root_pair_assessments()[0].status(),
            InterpretationIndependenceStatusV1::DistinctRootsIndependenceUnknown
        );
    }

    #[test]
    fn exact_current_pair_qualification_establishes_independence() {
        let a = eligible("rule-a", "v1", '1', '2', 150);
        let b = eligible("rule-b", "v1", '3', '4', 150);
        let baseline =
            assemble_interpretation_lineage_v1(&[a.clone(), b.clone()], &context(150), &[])
                .unwrap();
        let qualification = independence_qualification(
            baseline.roots()[0].interpretation_root_id(),
            baseline.roots()[1].interpretation_root_id(),
            "independence-authority",
        );
        let lineage = assemble_interpretation_lineage_v1(
            &[a, b],
            &context(150),
            &[qualification.clone()],
        )
        .unwrap();
        assert_eq!(
            lineage.root_pair_assessments()[0].status(),
            InterpretationIndependenceStatusV1::IndependenceQualified
        );
        assert_eq!(
            lineage.root_pair_assessments()[0].qualification_id(),
            Some(qualification.qualification_id())
        );
    }

    #[test]
    fn multiple_declarations_do_not_duplicate_root_pair_edges() {
        let a1 = eligible("rule-a", "v1", '1', '2', 150);
        let a2 = eligible("rule-a", "v1", '3', '4', 150);
        let b1 = eligible("rule-b", "v1", '5', '6', 150);
        let b2 = eligible("rule-b", "v1", '7', '8', 150);
        let lineage =
            assemble_interpretation_lineage_v1(&[a1, a2, b1, b2], &context(150), &[]).unwrap();
        assert_eq!(lineage.entries().len(), 4);
        assert_eq!(lineage.roots().len(), 2);
        assert_eq!(lineage.root_pair_assessments().len(), 1);
    }

    #[test]
    fn pair_order_does_not_change_independence_qualification_identity() {
        let a = eligible("rule-a", "v1", '1', '2', 150);
        let b = eligible("rule-b", "v1", '3', '4', 150);
        let baseline = assemble_interpretation_lineage_v1(&[a, b], &context(150), &[]).unwrap();
        let left = baseline.roots()[0].interpretation_root_id();
        let right = baseline.roots()[1].interpretation_root_id();
        let forward = independence_qualification(left, right, "independence-authority");
        let reverse = independence_qualification(right, left, "independence-authority");
        assert_eq!(forward, reverse);
        assert_eq!(forward.qualification_id(), reverse.qualification_id());
    }

    #[test]
    fn same_root_cannot_be_registered_as_independent() {
        let a = eligible("rule-a", "v1", '1', '2', 150);
        let lineage =
            assemble_interpretation_lineage_v1(std::slice::from_ref(&a), &context(150), &[])
                .unwrap();
        let root = lineage.roots()[0].interpretation_root_id();
        let raw = InterpretationIndependenceQualificationV1 {
            schema_version: INTERPRETATION_INDEPENDENCE_QUALIFICATION_SCHEMA_VERSION,
            left_interpretation_root_id: root.into(),
            right_interpretation_root_id: root.into(),
            proposition_id: PROPOSITION.into(),
            permitted_use: RelationDeclarationUseV1::ShadowRuntimeDisposition,
            qualifier_id: "authority".into(),
            qualifier_version: None,
            evaluator_id: "harness".into(),
            evaluator_version: None,
            qualification_policy_digest: POLICY.into(),
            qualification_artifact_digest: ARTIFACT.into(),
            qualified_at_unix_ms: 100,
            valid_until_unix_ms: 300,
        };
        assert_eq!(
            raw.register(),
            Err(InterpretationLineageError::SameRootCannotBeQualifiedIndependent)
        );
    }

    #[test]
    fn root_owner_cannot_directly_qualify_pair_independence() {
        let a = eligible("rule-a", "v1", '1', '2', 150);
        let b = eligible("rule-b", "v1", '3', '4', 150);
        let baseline =
            assemble_interpretation_lineage_v1(&[a.clone(), b.clone()], &context(150), &[])
                .unwrap();
        let qualification = independence_qualification(
            baseline.roots()[0].interpretation_root_id(),
            baseline.roots()[1].interpretation_root_id(),
            "rule-a",
        );
        assert_eq!(
            assemble_interpretation_lineage_v1(&[a, b], &context(150), &[qualification]),
            Err(InterpretationLineageError::DirectInterpretationSelfQualification)
        );
    }

    #[test]
    fn mixed_eligibility_contexts_fail_closed() {
        let a = eligible("rule-a", "v1", '1', '2', 150);
        let b = eligible("rule-b", "v1", '3', '4', 151);
        assert_eq!(
            assemble_interpretation_lineage_v1(&[a, b], &context(150), &[]),
            Err(InterpretationLineageError::EligibilityContextMismatch)
        );
    }

    #[test]
    fn expired_independence_qualification_cannot_establish_independence() {
        let a = eligible("rule-a", "v1", '1', '2', 250);
        let b = eligible("rule-b", "v1", '3', '4', 250);
        let baseline =
            assemble_interpretation_lineage_v1(&[a.clone(), b.clone()], &context(250), &[])
                .unwrap();
        let mut raw = independence_qualification(
            baseline.roots()[0].interpretation_root_id(),
            baseline.roots()[1].interpretation_root_id(),
            "independence-authority",
        )
        .record()
        .clone();
        raw.valid_until_unix_ms = 200;
        let expired = raw.register().unwrap();
        assert!(matches!(
            assemble_interpretation_lineage_v1(&[a, b], &context(250), &[expired]),
            Err(InterpretationLineageError::IndependenceQualificationExpired { .. })
        ));
    }

    #[test]
    fn registered_independence_qualification_revalidates_after_persistence() {
        let a = eligible("rule-a", "v1", '1', '2', 150);
        let b = eligible("rule-b", "v1", '3', '4', 150);
        let baseline = assemble_interpretation_lineage_v1(&[a, b], &context(150), &[]).unwrap();
        let registered = independence_qualification(
            baseline.roots()[0].interpretation_root_id(),
            baseline.roots()[1].interpretation_root_id(),
            "independence-authority",
        );
        let encoded = serde_json::to_string(&registered).unwrap();
        let decoded: RegisteredInterpretationIndependenceQualificationV1 =
            serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, registered);
    }

    #[test]
    fn lineage_is_order_independent_and_content_addressed() {
        let a = eligible("rule-a", "v1", '1', '2', 150);
        let b = eligible("rule-b", "v1", '3', '4', 150);
        let ab =
            assemble_interpretation_lineage_v1(&[a.clone(), b.clone()], &context(150), &[])
                .unwrap();
        let ba = assemble_interpretation_lineage_v1(&[b, a], &context(150), &[]).unwrap();
        assert_eq!(ab, ba);
        assert_eq!(ab.lineage_id(), ba.lineage_id());
        assert!(ab.lineage_id().starts_with("blake3:"));
    }

    #[test]
    fn issued_lineage_serializes_for_audit_only() {
        let a = eligible("rule-a", "v1", '1', '2', 150);
        let lineage = assemble_interpretation_lineage_v1(&[a], &context(150), &[]).unwrap();
        let encoded = serde_json::to_string(&lineage).unwrap();
        assert!(encoded.contains(lineage.lineage_id()));
        assert!(encoded.contains(lineage.profile_contract_digest()));
    }
}
