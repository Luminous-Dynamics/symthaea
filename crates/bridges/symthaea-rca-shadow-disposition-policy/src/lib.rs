// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! RCA-003b.3: preregistered shadow-disposition policy identity.
//!
//! This crate intentionally contains **no disposition engine**. It freezes the
//! complete policy surface that a later pure engine may consume so thresholds,
//! profile bindings, abstention behavior, defeater treatment, preregistration,
//! and resource ceilings cannot be introduced as hidden implementation constants.

#![deny(unsafe_code)]

use serde::{Deserialize, Deserializer, Serialize};
use symthaea_epistemic_governance::{
    evidence_set_witness::independent_evidence_set_witness_profile_digest_v1,
    experiment_contract::EXPERIMENT_CONTRACT_SCHEMA_VERSION,
    interpretation_lineage::interpretation_lineage_profile_digest_v1,
    relation_qualification::relation_declaration_eligibility_profile_digest_v1,
};
use symthaea_rca_bound_shadow_case::bound_shadow_evidence_case_profile_digest_v1;

pub const SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION: u16 = 1;
pub const SHADOW_DISPOSITION_POLICY_PROFILE_V1: &str = "rca-shadow-disposition-policy-v1";

pub const SHADOW_DISPOSITION_POLICY_CONTRACT_V1: &str = concat!(
    "rca-shadow-disposition-policy-v1\n",
    "policy_is_registered_before_result_bearing_case_evaluation\n",
    "scope=one_exact_proposition_id_v1\n",
    "profiles=exact_bound_case+evidence_set_witness+relation_eligibility+interpretation_lineage\n",
    "evidence_threshold_semantics=issued_pairwise_independent_evidence_items_v1\n",
    "interpretation_threshold_semantics=pairwise_independent_interpretation_root_set_v1\n",
    "distinct_evidence_root_count_does_not_satisfy_evidence_item_threshold\n",
    "candidate_module_root_id_and_pair_edge_counts_do_not_substitute_for_required_witnesses\n",
    "strength_treatment=diagnostic_only_v1_no_arithmetic\n",
    "defeater_mode=qualified_current_blocker_only_v1\n",
    "unknown_interpretation_independence=force_underdetermined_v1\n",
    "contestation=qualified_support_and_opposition_must_survive\n",
    "evaluation_binding=exact_registered_rca_experiment_contract_schema+digest\n",
    "evaluation_details_are_transitively_bound_by_registered_experiment_contract_not_duplicated\n",
    "resource_ceilings=explicit_case_items+interpretation_pairs_and_must_make_thresholds_feasible\n",
    "policy_id=blake3_explicit_complete_normalized_policy_v1\n",
    "post_result_policy_change_requires_new_policy_identity\n",
    "policy_registration_is_not_case_evaluation_or_disposition\n",
    "policy_is_not_belief_workspace_action_or_promotion_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-shadow-disposition-policy-contract:v1\0";
const POLICY_ID_DOMAIN: &[u8] = b"symthaea:rca-shadow-disposition-policy:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSetSemanticsV1 {
    /// A later engine may satisfy an evidence requirement only with an issued
    /// `IndependentEvidenceSetWitnessV1` whose selected **item** cardinality meets
    /// the threshold. Distinct ancestry-root count is provenance, not item count.
    IssuedPairwiseIndependentItems,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpretationRootSetSemanticsV1 {
    /// A threshold N means there must exist at least N interpretation roots where
    /// every distinct root pair is qualified independent in the exact lineage.
    PairwiseIndependentRoots,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationStrengthTreatmentV1 {
    DiagnosticOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefeaterModeV1 {
    QualifiedCurrentBlocker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownInterpretationIndependenceModeV1 {
    ForceUnderdetermined,
}

/// Topology requirements for one future shadow-disposition outcome condition.
/// Evidence cardinality is selected evidence **items** in an issued independent
/// set witness. Interpretation cardinality is a pairwise-independent root set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeTopologyRequirementsV1 {
    pub min_pairwise_independent_evidence_items: u16,
    pub min_pairwise_independent_interpretation_roots: u16,
}

/// Exact preregistration lineage under which this policy was frozen.
/// `RegisteredExperimentContractV1::contract_digest()` already commits corpus,
/// seed plan, evaluator, metrics, thresholds, falsification criteria, allowed
/// outcomes, and experiment resource ceilings. This policy references that one
/// canonical artifact instead of duplicating those fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispositionPolicyEvaluationBindingV1 {
    pub experiment_contract_schema_version: u16,
    pub registered_experiment_contract_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispositionPolicyResourceCeilingsV1 {
    pub max_case_items: u32,
    pub max_interpretation_pairs: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowDispositionPolicyV1 {
    pub schema_version: u16,
    pub proposition_id: String,
    pub bound_case_profile_digest: String,
    pub evidence_set_witness_profile_digest: String,
    pub relation_eligibility_profile_digest: String,
    pub interpretation_lineage_profile_digest: String,

    pub evidence_set_semantics: EvidenceSetSemanticsV1,
    pub interpretation_root_set_semantics: InterpretationRootSetSemanticsV1,
    pub tentative_support_requirements: OutcomeTopologyRequirementsV1,
    pub support_requirements: OutcomeTopologyRequirementsV1,
    pub tentative_opposition_requirements: OutcomeTopologyRequirementsV1,
    pub opposition_requirements: OutcomeTopologyRequirementsV1,
    pub defeater_requirements: OutcomeTopologyRequirementsV1,
    pub contested_side_requirements: OutcomeTopologyRequirementsV1,

    pub strength_treatment: RelationStrengthTreatmentV1,
    pub defeater_mode: DefeaterModeV1,
    pub unknown_interpretation_independence_mode: UnknownInterpretationIndependenceModeV1,
    pub contested_requires_qualified_support_and_opposition: bool,

    pub evaluation: DispositionPolicyEvaluationBindingV1,
    pub resources: DispositionPolicyResourceCeilingsV1,
}

impl ShadowDispositionPolicyV1 {
    pub fn register(
        self,
    ) -> Result<RegisteredShadowDispositionPolicyV1, ShadowDispositionPolicyError> {
        RegisteredShadowDispositionPolicyV1::try_from(self)
    }
}

/// Persistable/revalidated preregistered policy. No case-evaluation API exists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredShadowDispositionPolicyV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    policy_id: String,
    policy: ShadowDispositionPolicyV1,
}

impl RegisteredShadowDispositionPolicyV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn profile_contract_digest(&self) -> &str {
        &self.profile_contract_digest
    }

    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    pub fn policy(&self) -> &ShadowDispositionPolicyV1 {
        &self.policy
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisteredShadowDispositionPolicyWireV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    policy_id: String,
    policy: ShadowDispositionPolicyV1,
}

impl<'de> Deserialize<'de> for RegisteredShadowDispositionPolicyV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RegisteredShadowDispositionPolicyWireV1::deserialize(deserializer)?;
        if wire.schema_version != SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(
                ShadowDispositionPolicyError::UnsupportedSchemaVersion {
                    found: wire.schema_version,
                },
            ));
        }
        if wire.profile != SHADOW_DISPOSITION_POLICY_PROFILE_V1 {
            return Err(serde::de::Error::custom(
                ShadowDispositionPolicyError::UnexpectedProfile,
            ));
        }
        validate_digest(&wire.profile_contract_digest).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.policy_id).map_err(serde::de::Error::custom)?;
        let expected = wire
            .policy
            .clone()
            .register()
            .map_err(serde::de::Error::custom)?;
        if wire.profile_contract_digest != expected.profile_contract_digest
            || wire.policy_id != expected.policy_id
            || wire.policy != expected.policy
        {
            return Err(serde::de::Error::custom(
                ShadowDispositionPolicyError::PolicyIdentityMismatch,
            ));
        }
        Ok(expected)
    }
}

pub fn shadow_disposition_policy_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        SHADOW_DISPOSITION_POLICY_CONTRACT_V1.as_bytes(),
    )
}

pub fn current_bound_case_profile_digest_v1() -> String {
    bound_shadow_evidence_case_profile_digest_v1()
}

pub fn current_evidence_set_witness_profile_digest_v1() -> String {
    independent_evidence_set_witness_profile_digest_v1()
}

pub fn current_relation_eligibility_profile_digest_v1() -> String {
    relation_declaration_eligibility_profile_digest_v1()
}

pub fn current_interpretation_lineage_profile_digest_v1() -> String {
    interpretation_lineage_profile_digest_v1()
}

impl TryFrom<ShadowDispositionPolicyV1> for RegisteredShadowDispositionPolicyV1 {
    type Error = ShadowDispositionPolicyError;

    fn try_from(value: ShadowDispositionPolicyV1) -> Result<Self, Self::Error> {
        validate_policy_v1(&value)?;
        let profile_contract_digest = shadow_disposition_policy_profile_digest_v1();
        let policy_id = shadow_disposition_policy_id_v1(&profile_contract_digest, &value);
        Ok(Self {
            schema_version: SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION,
            profile: SHADOW_DISPOSITION_POLICY_PROFILE_V1.to_string(),
            profile_contract_digest,
            policy_id,
            policy: value,
        })
    }
}

fn validate_policy_v1(value: &ShadowDispositionPolicyV1) -> Result<(), ShadowDispositionPolicyError> {
    if value.schema_version != SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION {
        return Err(ShadowDispositionPolicyError::UnsupportedSchemaVersion {
            found: value.schema_version,
        });
    }
    validate_digest(&value.proposition_id)?;

    if value.bound_case_profile_digest != current_bound_case_profile_digest_v1() {
        return Err(ShadowDispositionPolicyError::BoundCaseProfileMismatch);
    }
    if value.evidence_set_witness_profile_digest != current_evidence_set_witness_profile_digest_v1()
    {
        return Err(ShadowDispositionPolicyError::EvidenceSetWitnessProfileMismatch);
    }
    if value.relation_eligibility_profile_digest != current_relation_eligibility_profile_digest_v1()
    {
        return Err(ShadowDispositionPolicyError::RelationEligibilityProfileMismatch);
    }
    if value.interpretation_lineage_profile_digest
        != current_interpretation_lineage_profile_digest_v1()
    {
        return Err(ShadowDispositionPolicyError::InterpretationLineageProfileMismatch);
    }

    match value.evidence_set_semantics {
        EvidenceSetSemanticsV1::IssuedPairwiseIndependentItems => {}
    }
    match value.interpretation_root_set_semantics {
        InterpretationRootSetSemanticsV1::PairwiseIndependentRoots => {}
    }
    match value.strength_treatment {
        RelationStrengthTreatmentV1::DiagnosticOnly => {}
    }
    match value.defeater_mode {
        DefeaterModeV1::QualifiedCurrentBlocker => {}
    }
    match value.unknown_interpretation_independence_mode {
        UnknownInterpretationIndependenceModeV1::ForceUnderdetermined => {}
    }

    for (name, requirements) in all_requirements(value) {
        validate_topology_requirements(name, requirements)?;
    }
    if !requirements_at_least(
        value.support_requirements,
        value.tentative_support_requirements,
    ) {
        return Err(ShadowDispositionPolicyError::SupportWeakerThanTentativeSupport);
    }
    if !requirements_at_least(
        value.opposition_requirements,
        value.tentative_opposition_requirements,
    ) {
        return Err(ShadowDispositionPolicyError::OppositionWeakerThanTentativeOpposition);
    }
    if !value.contested_requires_qualified_support_and_opposition {
        return Err(ShadowDispositionPolicyError::ContestedMustPreserveBothSides);
    }

    validate_evaluation_binding(&value.evaluation)?;
    validate_resource_feasibility(value)?;
    Ok(())
}

fn all_requirements(
    value: &ShadowDispositionPolicyV1,
) -> [(&'static str, OutcomeTopologyRequirementsV1); 6] {
    [
        ("tentative_support", value.tentative_support_requirements),
        ("support", value.support_requirements),
        ("tentative_opposition", value.tentative_opposition_requirements),
        ("opposition", value.opposition_requirements),
        ("defeater", value.defeater_requirements),
        ("contested_side", value.contested_side_requirements),
    ]
}

fn validate_topology_requirements(
    outcome: &'static str,
    value: OutcomeTopologyRequirementsV1,
) -> Result<(), ShadowDispositionPolicyError> {
    if value.min_pairwise_independent_evidence_items == 0 {
        return Err(ShadowDispositionPolicyError::ZeroEvidenceItemRequirement { outcome });
    }
    if value.min_pairwise_independent_interpretation_roots == 0 {
        return Err(ShadowDispositionPolicyError::ZeroInterpretationRootRequirement { outcome });
    }
    Ok(())
}

fn requirements_at_least(
    stronger: OutcomeTopologyRequirementsV1,
    weaker: OutcomeTopologyRequirementsV1,
) -> bool {
    stronger.min_pairwise_independent_evidence_items
        >= weaker.min_pairwise_independent_evidence_items
        && stronger.min_pairwise_independent_interpretation_roots
            >= weaker.min_pairwise_independent_interpretation_roots
}

fn validate_evaluation_binding(
    value: &DispositionPolicyEvaluationBindingV1,
) -> Result<(), ShadowDispositionPolicyError> {
    if value.experiment_contract_schema_version != EXPERIMENT_CONTRACT_SCHEMA_VERSION {
        return Err(ShadowDispositionPolicyError::ExperimentContractSchemaMismatch {
            found: value.experiment_contract_schema_version,
        });
    }
    validate_digest(&value.registered_experiment_contract_digest)
}

fn validate_resource_feasibility(
    value: &ShadowDispositionPolicyV1,
) -> Result<(), ShadowDispositionPolicyError> {
    if value.resources.max_case_items == 0 {
        return Err(ShadowDispositionPolicyError::ZeroMaxCaseItems);
    }

    let max_evidence_items = all_requirements(value)
        .into_iter()
        .map(|(_, requirement)| requirement.min_pairwise_independent_evidence_items as u32)
        .max()
        .unwrap_or(0);
    if value.resources.max_case_items < max_evidence_items {
        return Err(ShadowDispositionPolicyError::CaseCeilingBelowEvidenceItemRequirement);
    }

    let max_interpretation_roots = all_requirements(value)
        .into_iter()
        .map(|(_, requirement)| requirement.min_pairwise_independent_interpretation_roots as u32)
        .max()
        .unwrap_or(0);
    let required_pairs = pair_count_for_roots(max_interpretation_roots)?;
    if value.resources.max_interpretation_pairs < required_pairs {
        return Err(ShadowDispositionPolicyError::InterpretationPairCeilingBelowRootRequirement {
            required_pairs,
            configured_pairs: value.resources.max_interpretation_pairs,
        });
    }
    Ok(())
}

fn pair_count_for_roots(root_count: u32) -> Result<u32, ShadowDispositionPolicyError> {
    let product = root_count
        .checked_mul(root_count.saturating_sub(1))
        .ok_or(ShadowDispositionPolicyError::InterpretationPairCountOverflow)?;
    Ok(product / 2)
}

fn shadow_disposition_policy_id_v1(
    profile_contract_digest: &str,
    value: &ShadowDispositionPolicyV1,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(POLICY_ID_DOMAIN);
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
    hash_text(&mut hasher, b"proposition_id", &value.proposition_id);
    hash_text(
        &mut hasher,
        b"bound_case_profile_digest",
        &value.bound_case_profile_digest,
    );
    hash_text(
        &mut hasher,
        b"evidence_set_witness_profile_digest",
        &value.evidence_set_witness_profile_digest,
    );
    hash_text(
        &mut hasher,
        b"relation_eligibility_profile_digest",
        &value.relation_eligibility_profile_digest,
    );
    hash_text(
        &mut hasher,
        b"interpretation_lineage_profile_digest",
        &value.interpretation_lineage_profile_digest,
    );
    hash_text(
        &mut hasher,
        b"evidence_set_semantics",
        evidence_set_semantics_tag(value.evidence_set_semantics),
    );
    hash_text(
        &mut hasher,
        b"interpretation_root_set_semantics",
        interpretation_root_set_semantics_tag(value.interpretation_root_set_semantics),
    );

    for (name, requirements) in all_requirements(value) {
        hash_requirements(&mut hasher, name.as_bytes(), requirements);
    }

    hash_text(
        &mut hasher,
        b"strength_treatment",
        strength_treatment_tag(value.strength_treatment),
    );
    hash_text(
        &mut hasher,
        b"defeater_mode",
        defeater_mode_tag(value.defeater_mode),
    );
    hash_text(
        &mut hasher,
        b"unknown_interpretation_independence_mode",
        unknown_independence_tag(value.unknown_interpretation_independence_mode),
    );
    hash_bool(
        &mut hasher,
        b"contested_requires_qualified_support_and_opposition",
        value.contested_requires_qualified_support_and_opposition,
    );
    hash_bytes(
        &mut hasher,
        b"experiment_contract_schema_version",
        &value.evaluation.experiment_contract_schema_version.to_le_bytes(),
    );
    hash_text(
        &mut hasher,
        b"registered_experiment_contract_digest",
        &value.evaluation.registered_experiment_contract_digest,
    );
    hash_bytes(
        &mut hasher,
        b"max_case_items",
        &value.resources.max_case_items.to_le_bytes(),
    );
    hash_bytes(
        &mut hasher,
        b"max_interpretation_pairs",
        &value.resources.max_interpretation_pairs.to_le_bytes(),
    );
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn hash_requirements(
    hasher: &mut blake3::Hasher,
    outcome: &[u8],
    value: OutcomeTopologyRequirementsV1,
) {
    hash_bytes(hasher, b"outcome", outcome);
    hash_bytes(
        hasher,
        b"min_pairwise_independent_evidence_items",
        &value.min_pairwise_independent_evidence_items.to_le_bytes(),
    );
    hash_bytes(
        hasher,
        b"min_pairwise_independent_interpretation_roots",
        &value.min_pairwise_independent_interpretation_roots.to_le_bytes(),
    );
}

fn evidence_set_semantics_tag(value: EvidenceSetSemanticsV1) -> &'static str {
    match value {
        EvidenceSetSemanticsV1::IssuedPairwiseIndependentItems => {
            "issued_pairwise_independent_items"
        }
    }
}

fn interpretation_root_set_semantics_tag(
    value: InterpretationRootSetSemanticsV1,
) -> &'static str {
    match value {
        InterpretationRootSetSemanticsV1::PairwiseIndependentRoots => {
            "pairwise_independent_roots"
        }
    }
}

fn strength_treatment_tag(value: RelationStrengthTreatmentV1) -> &'static str {
    match value {
        RelationStrengthTreatmentV1::DiagnosticOnly => "diagnostic_only",
    }
}

fn defeater_mode_tag(value: DefeaterModeV1) -> &'static str {
    match value {
        DefeaterModeV1::QualifiedCurrentBlocker => "qualified_current_blocker",
    }
}

fn unknown_independence_tag(value: UnknownInterpretationIndependenceModeV1) -> &'static str {
    match value {
        UnknownInterpretationIndependenceModeV1::ForceUnderdetermined => "force_underdetermined",
    }
}

fn validate_digest(digest: &str) -> Result<(), ShadowDispositionPolicyError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(ShadowDispositionPolicyError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ShadowDispositionPolicyError::MalformedDigest);
    }
    Ok(())
}

fn hash_bool(hasher: &mut blake3::Hasher, label: &[u8], value: bool) {
    hash_bytes(hasher, label, &[u8::from(value)]);
}

fn hash_text(hasher: &mut blake3::Hasher, label: &[u8], value: &str) {
    hash_bytes(hasher, label, value.as_bytes());
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
pub enum ShadowDispositionPolicyError {
    UnsupportedSchemaVersion { found: u16 },
    UnexpectedProfile,
    PolicyIdentityMismatch,
    MalformedDigest,
    BoundCaseProfileMismatch,
    EvidenceSetWitnessProfileMismatch,
    RelationEligibilityProfileMismatch,
    InterpretationLineageProfileMismatch,
    ZeroEvidenceItemRequirement { outcome: &'static str },
    ZeroInterpretationRootRequirement { outcome: &'static str },
    SupportWeakerThanTentativeSupport,
    OppositionWeakerThanTentativeOpposition,
    ContestedMustPreserveBothSides,
    ExperimentContractSchemaMismatch { found: u16 },
    ZeroMaxCaseItems,
    CaseCeilingBelowEvidenceItemRequirement,
    InterpretationPairCountOverflow,
    InterpretationPairCeilingBelowRootRequirement {
        required_pairs: u32,
        configured_pairs: u32,
    },
}

impl std::fmt::Display for ShadowDispositionPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported shadow-disposition policy schema version {found}; expected {SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION}"
            ),
            Self::UnexpectedProfile => f.write_str("unexpected shadow-disposition policy profile"),
            Self::PolicyIdentityMismatch => {
                f.write_str("registered shadow-disposition policy identity mismatch")
            }
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::BoundCaseProfileMismatch => {
                f.write_str("policy does not bind the current bound-case profile")
            }
            Self::EvidenceSetWitnessProfileMismatch => {
                f.write_str("policy does not bind the current independent-evidence-set witness profile")
            }
            Self::RelationEligibilityProfileMismatch => {
                f.write_str("policy does not bind the current relation-eligibility profile")
            }
            Self::InterpretationLineageProfileMismatch => {
                f.write_str("policy does not bind the current interpretation-lineage profile")
            }
            Self::ZeroEvidenceItemRequirement { outcome } => write!(
                f,
                "{outcome} requires at least one pairwise-independent evidence item"
            ),
            Self::ZeroInterpretationRootRequirement { outcome } => write!(
                f,
                "{outcome} requires at least one pairwise-independent interpretation root"
            ),
            Self::SupportWeakerThanTentativeSupport => f.write_str(
                "Supported topology requirements cannot be weaker than TentativelySupported requirements",
            ),
            Self::OppositionWeakerThanTentativeOpposition => f.write_str(
                "Opposed topology requirements cannot be weaker than TentativelyOpposed requirements",
            ),
            Self::ContestedMustPreserveBothSides => {
                f.write_str("v1 Contested must require qualified support and opposition")
            }
            Self::ExperimentContractSchemaMismatch { found } => write!(
                f,
                "policy expects RCA experiment-contract schema {EXPERIMENT_CONTRACT_SCHEMA_VERSION}, found {found}"
            ),
            Self::ZeroMaxCaseItems => f.write_str("policy max_case_items must be non-zero"),
            Self::CaseCeilingBelowEvidenceItemRequirement => f.write_str(
                "policy max_case_items cannot be below its largest pairwise-independent evidence-item requirement",
            ),
            Self::InterpretationPairCountOverflow => {
                f.write_str("interpretation-root pair requirement overflowed u32")
            }
            Self::InterpretationPairCeilingBelowRootRequirement {
                required_pairs,
                configured_pairs,
            } => write!(
                f,
                "policy requires at least {required_pairs} interpretation-root pair assessments but ceiling is {configured_pairs}"
            ),
        }
    }
}

impl std::error::Error for ShadowDispositionPolicyError {}

#[cfg(test)]
mod tests {
    use super::*;

    const PROPOSITION: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const EXPERIMENT: &str =
        "blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const EXPERIMENT_2: &str =
        "blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn requirements(evidence_items: u16, interpretations: u16) -> OutcomeTopologyRequirementsV1 {
        OutcomeTopologyRequirementsV1 {
            min_pairwise_independent_evidence_items: evidence_items,
            min_pairwise_independent_interpretation_roots: interpretations,
        }
    }

    fn policy() -> ShadowDispositionPolicyV1 {
        ShadowDispositionPolicyV1 {
            schema_version: SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION,
            proposition_id: PROPOSITION.into(),
            bound_case_profile_digest: current_bound_case_profile_digest_v1(),
            evidence_set_witness_profile_digest: current_evidence_set_witness_profile_digest_v1(),
            relation_eligibility_profile_digest: current_relation_eligibility_profile_digest_v1(),
            interpretation_lineage_profile_digest: current_interpretation_lineage_profile_digest_v1(),
            evidence_set_semantics: EvidenceSetSemanticsV1::IssuedPairwiseIndependentItems,
            interpretation_root_set_semantics:
                InterpretationRootSetSemanticsV1::PairwiseIndependentRoots,
            tentative_support_requirements: requirements(1, 1),
            support_requirements: requirements(2, 2),
            tentative_opposition_requirements: requirements(1, 1),
            opposition_requirements: requirements(2, 2),
            defeater_requirements: requirements(1, 1),
            contested_side_requirements: requirements(1, 1),
            strength_treatment: RelationStrengthTreatmentV1::DiagnosticOnly,
            defeater_mode: DefeaterModeV1::QualifiedCurrentBlocker,
            unknown_interpretation_independence_mode:
                UnknownInterpretationIndependenceModeV1::ForceUnderdetermined,
            contested_requires_qualified_support_and_opposition: true,
            evaluation: DispositionPolicyEvaluationBindingV1 {
                experiment_contract_schema_version: EXPERIMENT_CONTRACT_SCHEMA_VERSION,
                registered_experiment_contract_digest: EXPERIMENT.into(),
            },
            resources: DispositionPolicyResourceCeilingsV1 {
                max_case_items: 32,
                max_interpretation_pairs: 496,
            },
        }
    }

    #[test]
    fn policy_registers_against_exact_current_profiles() {
        let registered = policy().register().unwrap();
        assert_eq!(registered.policy().proposition_id, PROPOSITION);
        assert_eq!(
            registered.policy().bound_case_profile_digest,
            current_bound_case_profile_digest_v1()
        );
        assert_eq!(
            registered.policy().evidence_set_witness_profile_digest,
            current_evidence_set_witness_profile_digest_v1()
        );
        assert_eq!(
            registered.policy().relation_eligibility_profile_digest,
            current_relation_eligibility_profile_digest_v1()
        );
        assert_eq!(
            registered.policy().interpretation_lineage_profile_digest,
            current_interpretation_lineage_profile_digest_v1()
        );
    }

    #[test]
    fn evidence_item_and_interpretation_root_thresholds_are_distinct() {
        let registered = policy().register().unwrap();
        assert_eq!(
            registered.policy().evidence_set_semantics,
            EvidenceSetSemanticsV1::IssuedPairwiseIndependentItems
        );
        assert_eq!(
            registered.policy().interpretation_root_set_semantics,
            InterpretationRootSetSemanticsV1::PairwiseIndependentRoots
        );
        assert!(SHADOW_DISPOSITION_POLICY_CONTRACT_V1.contains(
            "distinct_evidence_root_count_does_not_satisfy_evidence_item_threshold"
        ));
    }

    #[test]
    fn evaluation_binding_uses_exact_registered_experiment_contract_identity() {
        let registered = policy().register().unwrap();
        assert_eq!(
            registered.policy().evaluation.experiment_contract_schema_version,
            EXPERIMENT_CONTRACT_SCHEMA_VERSION
        );
        assert_eq!(
            registered.policy().evaluation.registered_experiment_contract_digest,
            EXPERIMENT
        );
    }

    #[test]
    fn wrong_experiment_contract_schema_fails_closed() {
        let mut raw = policy();
        raw.evaluation.experiment_contract_schema_version =
            EXPERIMENT_CONTRACT_SCHEMA_VERSION.saturating_add(1);
        assert!(matches!(
            raw.register(),
            Err(ShadowDispositionPolicyError::ExperimentContractSchemaMismatch { .. })
        ));
    }

    #[test]
    fn strength_semantics_are_diagnostic_only_v1() {
        assert_eq!(
            policy().strength_treatment,
            RelationStrengthTreatmentV1::DiagnosticOnly
        );
    }

    #[test]
    fn supported_cannot_be_weaker_than_tentative_support() {
        let mut raw = policy();
        raw.tentative_support_requirements = requirements(2, 2);
        raw.support_requirements = requirements(1, 2);
        assert_eq!(
            raw.register(),
            Err(ShadowDispositionPolicyError::SupportWeakerThanTentativeSupport)
        );
    }

    #[test]
    fn opposed_cannot_be_weaker_than_tentative_opposition() {
        let mut raw = policy();
        raw.tentative_opposition_requirements = requirements(2, 2);
        raw.opposition_requirements = requirements(2, 1);
        assert_eq!(
            raw.register(),
            Err(ShadowDispositionPolicyError::OppositionWeakerThanTentativeOpposition)
        );
    }

    #[test]
    fn zero_topology_requirements_fail_closed() {
        let mut raw = policy();
        raw.defeater_requirements = requirements(0, 1);
        assert!(matches!(
            raw.register(),
            Err(ShadowDispositionPolicyError::ZeroEvidenceItemRequirement {
                outcome: "defeater"
            })
        ));
    }

    #[test]
    fn lower_layer_profile_drift_requires_new_policy() {
        let mut raw = policy();
        raw.evidence_set_witness_profile_digest = EXPERIMENT.into();
        assert_eq!(
            raw.register(),
            Err(ShadowDispositionPolicyError::EvidenceSetWitnessProfileMismatch)
        );
    }

    #[test]
    fn contested_must_preserve_qualified_support_and_opposition() {
        let mut raw = policy();
        raw.contested_requires_qualified_support_and_opposition = false;
        assert_eq!(
            raw.register(),
            Err(ShadowDispositionPolicyError::ContestedMustPreserveBothSides)
        );
    }

    #[test]
    fn experiment_and_resource_identity_are_policy_bearing() {
        let a = policy().register().unwrap();
        let mut changed = policy();
        changed.evaluation.registered_experiment_contract_digest = EXPERIMENT_2.into();
        let b = changed.register().unwrap();
        assert_ne!(a.policy_id(), b.policy_id());

        let mut changed = policy();
        changed.resources.max_case_items = 64;
        let c = changed.register().unwrap();
        assert_ne!(a.policy_id(), c.policy_id());
    }

    #[test]
    fn threshold_change_requires_new_policy_identity() {
        let a = policy().register().unwrap();
        let mut changed = policy();
        changed.support_requirements = requirements(3, 2);
        let b = changed.register().unwrap();
        assert_ne!(a.policy_id(), b.policy_id());
    }

    #[test]
    fn registered_policy_revalidates_after_persistence() {
        let registered = policy().register().unwrap();
        let encoded = serde_json::to_string(&registered).unwrap();
        let decoded: RegisteredShadowDispositionPolicyV1 =
            serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, registered);
    }

    #[test]
    fn policy_tampering_fails_closed() {
        let registered = policy().register().unwrap();
        let mut value = serde_json::to_value(&registered).unwrap();
        value["policy"]["support_requirements"]
            ["min_pairwise_independent_evidence_items"] = serde_json::Value::from(7_u64);
        assert!(serde_json::from_value::<RegisteredShadowDispositionPolicyV1>(value).is_err());
    }

    #[test]
    fn evidence_resource_ceiling_must_make_item_threshold_feasible() {
        let mut raw = policy();
        raw.support_requirements = requirements(33, 2);
        assert_eq!(
            raw.register(),
            Err(ShadowDispositionPolicyError::CaseCeilingBelowEvidenceItemRequirement)
        );
    }

    #[test]
    fn interpretation_pair_ceiling_must_make_root_set_feasible() {
        let mut raw = policy();
        raw.support_requirements = requirements(2, 4);
        raw.resources.max_interpretation_pairs = 5;
        assert_eq!(
            raw.register(),
            Err(
                ShadowDispositionPolicyError::InterpretationPairCeilingBelowRootRequirement {
                    required_pairs: 6,
                    configured_pairs: 5,
                }
            )
        );
    }

    #[test]
    fn policy_profile_has_strict_identity() {
        let digest = shadow_disposition_policy_profile_digest_v1();
        assert!(digest.starts_with("blake3:"));
        assert_eq!(digest.len(), "blake3:".len() + 64);
        assert_eq!(digest, shadow_disposition_policy_profile_digest_v1());
    }
}
