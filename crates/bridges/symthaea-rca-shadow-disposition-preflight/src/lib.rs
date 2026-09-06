// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! RCA-003b.3b: exact cross-artifact preflight for shadow disposition.
//!
//! This crate performs no disposition. It proves that already-issued RCA
//! artifacts belong to one exact proposition/case/currentness/qualification/
//! policy lineage before a future pure shadow engine may inspect them together.
//!
//! The issued result is Serialize-only. Archived bytes are audit material and
//! cannot restore current evaluation eligibility.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_epistemic_governance::{
    currentness::EvidenceRelationKindV1,
    evidence_set_witness::IndependentEvidenceSetWitnessV1,
    experiment_contract::{RegisteredExperimentContractV1, EXPERIMENT_CONTRACT_SCHEMA_VERSION},
    interpretation_lineage::InterpretationLineageV1,
    interpretation_set_witness::IndependentInterpretationRootSetWitnessV1,
    relation_qualification::DispositionEligibleRelationDeclarationV1,
};
use symthaea_rca_bound_shadow_case::BoundShadowEvidenceCaseV1;
use symthaea_rca_effective_disposition_policy::RegisteredEffectiveShadowDispositionPolicyV1;

pub const SHADOW_DISPOSITION_PREFLIGHT_SCHEMA_VERSION: u16 = 1;
pub const SHADOW_DISPOSITION_PREFLIGHT_PROFILE_V1: &str =
    "rca-shadow-disposition-preflight-v1";

pub const SHADOW_DISPOSITION_PREFLIGHT_CONTRACT_V1: &str = concat!(
    "rca-shadow-disposition-preflight-v1\n",
    "input=bound_case+bounded_evidence_witness_slots+eligible_declarations+interpretation_lineage+bounded_interpretation_witness_slots+effective_policy+registered_experiment_contract\n",
    "one_exact_proposition_across_case+eligibility+lineage+witnesses+policy\n",
    "case_declaration_set_must_exactly_equal_eligible_declaration_set\n",
    "lineage_declaration_to_eligibility_entries_must_exactly_equal_current_eligibility_set\n",
    "all_eligible_declarations_must_share_exact_lineage_context_commitment\n",
    "evidence_witness_items_must_be_current_case_candidates_with_exact_single_observation_root_binding_v1\n",
    "composite_multiroot_evidence_is_out_of_scope_for_this_preflight_profile\n",
    "evidence_witness_slot_relations_are_support_or_opposition_or_defeater_scoped\n",
    "interpretation_witness_requires_corresponding_evidence_witness_same_slot\n",
    "interpretation_witness_root_set_must_exactly_equal_roots_interpreting_same_slot_evidence_items\n",
    "artifact_profiles_must_match_effective_preregistered_policy_bindings\n",
    "actual_case_items_and_interpretation_pairs_must_respect_policy_resource_ceilings\n",
    "actual_registered_experiment_contract_must_verify_and_match_policy_digest\n",
    "preflight_performs_no_threshold_comparison_or_disposition\n",
    "preflight_id=blake3_explicit_complete_cross_artifact_binding_v1\n",
    "issued_preflight=is_private_non_deserializable_shadow_capability\n",
    "preflight_is_not_disposition_belief_workspace_action_or_promotion_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-shadow-disposition-preflight-contract:v1\0";
const PREFLIGHT_ID_DOMAIN: &[u8] = b"symthaea:rca-shadow-disposition-preflight:v1\0";

/// Fixed semantic slots prevent an unbounded witness list from becoming an
/// unregistered resource dimension. Reusing one exact witness in several slots
/// does not multiply evidence because each slot stores the same witness identity.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShadowDispositionEvidenceWitnessSlotsV1<'a> {
    pub support: Option<&'a IndependentEvidenceSetWitnessV1>,
    pub opposition: Option<&'a IndependentEvidenceSetWitnessV1>,
    pub defeater: Option<&'a IndependentEvidenceSetWitnessV1>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ShadowDispositionInterpretationWitnessSlotsV1<'a> {
    pub support: Option<&'a IndependentInterpretationRootSetWitnessV1>,
    pub opposition: Option<&'a IndependentInterpretationRootSetWitnessV1>,
    pub defeater: Option<&'a IndependentInterpretationRootSetWitnessV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ShadowDispositionEligibilityBindingV1 {
    declaration_id: String,
    eligibility_id: String,
}

impl ShadowDispositionEligibilityBindingV1 {
    pub fn declaration_id(&self) -> &str {
        &self.declaration_id
    }

    pub fn eligibility_id(&self) -> &str {
        &self.eligibility_id
    }
}

/// Exact current input binding that a future pure shadow-disposition engine may
/// require. It intentionally has no `Deserialize` implementation.
#[must_use = "shadow-disposition preflight is a current evaluation capability and should be inspected"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ShadowDispositionPreflightV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    preflight_id: String,
    proposition_id: String,
    case_id: String,
    case_scope_digest: String,
    evidence_lineage_graph_id: String,
    relation_eligibility_context_commitment: String,
    support_evidence_witness_id: Option<String>,
    opposition_evidence_witness_id: Option<String>,
    defeater_evidence_witness_id: Option<String>,
    support_interpretation_witness_id: Option<String>,
    opposition_interpretation_witness_id: Option<String>,
    defeater_interpretation_witness_id: Option<String>,
    eligibility_bindings: Vec<ShadowDispositionEligibilityBindingV1>,
    interpretation_lineage_id: String,
    effective_policy_id: String,
    registered_experiment_contract_digest: String,
}

impl ShadowDispositionPreflightV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }
    pub fn profile(&self) -> &str {
        &self.profile
    }
    pub fn profile_contract_digest(&self) -> &str {
        &self.profile_contract_digest
    }
    pub fn preflight_id(&self) -> &str {
        &self.preflight_id
    }
    pub fn proposition_id(&self) -> &str {
        &self.proposition_id
    }
    pub fn case_id(&self) -> &str {
        &self.case_id
    }
    pub fn case_scope_digest(&self) -> &str {
        &self.case_scope_digest
    }
    pub fn evidence_lineage_graph_id(&self) -> &str {
        &self.evidence_lineage_graph_id
    }
    pub fn relation_eligibility_context_commitment(&self) -> &str {
        &self.relation_eligibility_context_commitment
    }
    pub fn support_evidence_witness_id(&self) -> Option<&str> {
        self.support_evidence_witness_id.as_deref()
    }
    pub fn opposition_evidence_witness_id(&self) -> Option<&str> {
        self.opposition_evidence_witness_id.as_deref()
    }
    pub fn defeater_evidence_witness_id(&self) -> Option<&str> {
        self.defeater_evidence_witness_id.as_deref()
    }
    pub fn support_interpretation_witness_id(&self) -> Option<&str> {
        self.support_interpretation_witness_id.as_deref()
    }
    pub fn opposition_interpretation_witness_id(&self) -> Option<&str> {
        self.opposition_interpretation_witness_id.as_deref()
    }
    pub fn defeater_interpretation_witness_id(&self) -> Option<&str> {
        self.defeater_interpretation_witness_id.as_deref()
    }
    pub fn eligibility_bindings(&self) -> &[ShadowDispositionEligibilityBindingV1] {
        &self.eligibility_bindings
    }
    pub fn interpretation_lineage_id(&self) -> &str {
        &self.interpretation_lineage_id
    }
    pub fn effective_policy_id(&self) -> &str {
        &self.effective_policy_id
    }
    pub fn registered_experiment_contract_digest(&self) -> &str {
        &self.registered_experiment_contract_digest
    }
}

pub fn shadow_disposition_preflight_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        SHADOW_DISPOSITION_PREFLIGHT_CONTRACT_V1.as_bytes(),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CaseItemFactV1 {
    candidate_id: String,
    observation_root_id: String,
    declaration_id: String,
    relation: EvidenceRelationKindV1,
    current_runtime_relevant: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WitnessSideV1 {
    Support,
    Opposition,
    Defeater,
}

/// Prove all exact cross-artifact joins required before a future pure shadow
/// disposition engine may inspect these inputs together.
///
/// This function does **not** compare witness cardinality to policy thresholds and
/// does not emit a disposition. Empty witness slots are valid preflight inputs and
/// leave later epistemic interpretation underdetermined.
pub fn preflight_shadow_disposition_inputs_v1(
    bound_case: &BoundShadowEvidenceCaseV1,
    evidence_witnesses: ShadowDispositionEvidenceWitnessSlotsV1<'_>,
    eligible_declarations: &[DispositionEligibleRelationDeclarationV1],
    interpretation_lineage: &InterpretationLineageV1,
    interpretation_witnesses: ShadowDispositionInterpretationWitnessSlotsV1<'_>,
    effective_policy: &RegisteredEffectiveShadowDispositionPolicyV1,
    experiment_contract: &RegisteredExperimentContractV1,
) -> Result<ShadowDispositionPreflightV1, ShadowDispositionPreflightError> {
    let structural_case = bound_case.structural_case();
    let proposition_id = structural_case.proposition_id();
    let base_policy = effective_policy.base_policy();
    let policy = base_policy.policy();

    validate_policy_scope(bound_case, interpretation_lineage, effective_policy)?;
    validate_experiment_contract(policy, experiment_contract)?;
    validate_resource_ceilings(bound_case, interpretation_lineage, policy)?;

    let case_by_candidate = build_case_facts(bound_case)?;
    let case_by_declaration = case_by_candidate
        .values()
        .map(|fact| (fact.declaration_id.clone(), fact.clone()))
        .collect::<BTreeMap<_, _>>();
    if case_by_declaration.len() != case_by_candidate.len() {
        return Err(ShadowDispositionPreflightError::DuplicateCaseDeclarationId);
    }

    let eligibility_bindings = validate_eligibility_join(
        proposition_id,
        policy.relation_eligibility_profile_digest.as_str(),
        eligible_declarations,
        &case_by_declaration,
        interpretation_lineage,
    )?;
    let lineage_root_by_declaration = interpretation_lineage
        .entries()
        .iter()
        .map(|entry| {
            (
                entry.declaration_id().to_string(),
                entry.interpretation_root_id().to_string(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    let support_evidence = validate_evidence_witness_slot(
        evidence_witnesses.support,
        WitnessSideV1::Support,
        policy.evidence_set_witness_profile_digest.as_str(),
        &case_by_candidate,
    )?;
    let opposition_evidence = validate_evidence_witness_slot(
        evidence_witnesses.opposition,
        WitnessSideV1::Opposition,
        policy.evidence_set_witness_profile_digest.as_str(),
        &case_by_candidate,
    )?;
    let defeater_evidence = validate_evidence_witness_slot(
        evidence_witnesses.defeater,
        WitnessSideV1::Defeater,
        policy.evidence_set_witness_profile_digest.as_str(),
        &case_by_candidate,
    )?;

    validate_interpretation_witness_slot(
        interpretation_witnesses.support,
        support_evidence.as_ref(),
        WitnessSideV1::Support,
        proposition_id,
        effective_policy.interpretation_set_witness_profile_digest(),
        interpretation_lineage,
        &case_by_candidate,
        &lineage_root_by_declaration,
    )?;
    validate_interpretation_witness_slot(
        interpretation_witnesses.opposition,
        opposition_evidence.as_ref(),
        WitnessSideV1::Opposition,
        proposition_id,
        effective_policy.interpretation_set_witness_profile_digest(),
        interpretation_lineage,
        &case_by_candidate,
        &lineage_root_by_declaration,
    )?;
    validate_interpretation_witness_slot(
        interpretation_witnesses.defeater,
        defeater_evidence.as_ref(),
        WitnessSideV1::Defeater,
        proposition_id,
        effective_policy.interpretation_set_witness_profile_digest(),
        interpretation_lineage,
        &case_by_candidate,
        &lineage_root_by_declaration,
    )?;

    let profile_contract_digest = shadow_disposition_preflight_profile_digest_v1();
    let support_evidence_witness_id = witness_id(evidence_witnesses.support);
    let opposition_evidence_witness_id = witness_id(evidence_witnesses.opposition);
    let defeater_evidence_witness_id = witness_id(evidence_witnesses.defeater);
    let support_interpretation_witness_id = interpretation_witness_id(interpretation_witnesses.support);
    let opposition_interpretation_witness_id =
        interpretation_witness_id(interpretation_witnesses.opposition);
    let defeater_interpretation_witness_id =
        interpretation_witness_id(interpretation_witnesses.defeater);

    let identity = PreflightIdentityInputsV1 {
        profile_contract_digest: &profile_contract_digest,
        proposition_id,
        case_id: bound_case.case_id(),
        case_scope_digest: structural_case.case_scope_digest(),
        evidence_lineage_graph_id: structural_case.lineage_graph_id(),
        relation_eligibility_context_commitment: interpretation_lineage
            .eligibility_context_commitment(),
        support_evidence_witness_id: support_evidence_witness_id.as_deref(),
        opposition_evidence_witness_id: opposition_evidence_witness_id.as_deref(),
        defeater_evidence_witness_id: defeater_evidence_witness_id.as_deref(),
        support_interpretation_witness_id: support_interpretation_witness_id.as_deref(),
        opposition_interpretation_witness_id: opposition_interpretation_witness_id.as_deref(),
        defeater_interpretation_witness_id: defeater_interpretation_witness_id.as_deref(),
        eligibility_bindings: &eligibility_bindings,
        interpretation_lineage_id: interpretation_lineage.lineage_id(),
        effective_policy_id: effective_policy.effective_policy_id(),
        experiment_contract_digest: experiment_contract.contract_digest(),
    };
    let preflight_id = preflight_id_v1(&identity);

    Ok(ShadowDispositionPreflightV1 {
        schema_version: SHADOW_DISPOSITION_PREFLIGHT_SCHEMA_VERSION,
        profile: SHADOW_DISPOSITION_PREFLIGHT_PROFILE_V1.to_string(),
        profile_contract_digest,
        preflight_id,
        proposition_id: proposition_id.to_string(),
        case_id: bound_case.case_id().to_string(),
        case_scope_digest: structural_case.case_scope_digest().to_string(),
        evidence_lineage_graph_id: structural_case.lineage_graph_id().to_string(),
        relation_eligibility_context_commitment: interpretation_lineage
            .eligibility_context_commitment()
            .to_string(),
        support_evidence_witness_id,
        opposition_evidence_witness_id,
        defeater_evidence_witness_id,
        support_interpretation_witness_id,
        opposition_interpretation_witness_id,
        defeater_interpretation_witness_id,
        eligibility_bindings,
        interpretation_lineage_id: interpretation_lineage.lineage_id().to_string(),
        effective_policy_id: effective_policy.effective_policy_id().to_string(),
        registered_experiment_contract_digest: experiment_contract.contract_digest().to_string(),
    })
}

fn validate_policy_scope(
    bound_case: &BoundShadowEvidenceCaseV1,
    interpretation_lineage: &InterpretationLineageV1,
    effective_policy: &RegisteredEffectiveShadowDispositionPolicyV1,
) -> Result<(), ShadowDispositionPreflightError> {
    let proposition_id = bound_case.structural_case().proposition_id();
    let policy = effective_policy.base_policy().policy();
    if policy.proposition_id.as_str() != proposition_id {
        return Err(ShadowDispositionPreflightError::PolicyPropositionMismatch);
    }
    if bound_case.profile_contract_digest() != policy.bound_case_profile_digest.as_str() {
        return Err(ShadowDispositionPreflightError::BoundCaseProfileMismatch);
    }
    if interpretation_lineage.profile_contract_digest()
        != policy.interpretation_lineage_profile_digest.as_str()
    {
        return Err(ShadowDispositionPreflightError::InterpretationLineageProfileMismatch);
    }
    if interpretation_lineage.proposition_id() != proposition_id {
        return Err(ShadowDispositionPreflightError::InterpretationLineagePropositionMismatch);
    }
    Ok(())
}

fn validate_experiment_contract(
    policy: &symthaea_rca_shadow_disposition_policy::ShadowDispositionPolicyV1,
    experiment_contract: &RegisteredExperimentContractV1,
) -> Result<(), ShadowDispositionPreflightError> {
    experiment_contract
        .verify_integrity()
        .map_err(|error| ShadowDispositionPreflightError::ExperimentContractIntegrity(error.to_string()))?;
    if policy.evaluation.experiment_contract_schema_version != EXPERIMENT_CONTRACT_SCHEMA_VERSION {
        return Err(ShadowDispositionPreflightError::ExperimentContractSchemaMismatch);
    }
    if policy.evaluation.registered_experiment_contract_digest.as_str()
        != experiment_contract.contract_digest()
    {
        return Err(ShadowDispositionPreflightError::ExperimentContractDigestMismatch);
    }
    Ok(())
}

fn validate_resource_ceilings(
    bound_case: &BoundShadowEvidenceCaseV1,
    interpretation_lineage: &InterpretationLineageV1,
    policy: &symthaea_rca_shadow_disposition_policy::ShadowDispositionPolicyV1,
) -> Result<(), ShadowDispositionPreflightError> {
    let actual_case_items = bound_case.structural_case().items().len();
    if actual_case_items > policy.resources.max_case_items as usize {
        return Err(ShadowDispositionPreflightError::CaseItemCeilingExceeded {
            actual: actual_case_items,
            ceiling: policy.resources.max_case_items,
        });
    }
    let actual_pairs = interpretation_lineage.root_pair_assessments().len();
    if actual_pairs > policy.resources.max_interpretation_pairs as usize {
        return Err(ShadowDispositionPreflightError::InterpretationPairCeilingExceeded {
            actual: actual_pairs,
            ceiling: policy.resources.max_interpretation_pairs,
        });
    }
    Ok(())
}

fn build_case_facts(
    bound_case: &BoundShadowEvidenceCaseV1,
) -> Result<BTreeMap<String, CaseItemFactV1>, ShadowDispositionPreflightError> {
    let items = bound_case.structural_case().items();
    let declarations = bound_case.relation_declarations();
    if items.len() != declarations.len() {
        return Err(ShadowDispositionPreflightError::BoundCaseInternalCardinalityMismatch);
    }

    let mut result = BTreeMap::new();
    let mut declaration_ids = BTreeSet::new();
    for (item, declaration) in items.iter().zip(declarations) {
        let relation = declaration.relation().as_raw();
        if relation.evidence_id.as_str() != item.candidate_id() {
            return Err(ShadowDispositionPreflightError::CaseDeclarationCandidateMismatch {
                candidate_id: item.candidate_id().to_string(),
                relation_evidence_id: relation.evidence_id.clone(),
            });
        }
        if relation.relation != item.declared_relation() {
            return Err(ShadowDispositionPreflightError::CaseDeclarationRelationMismatch {
                candidate_id: item.candidate_id().to_string(),
            });
        }
        if !declaration_ids.insert(declaration.declaration_id().to_string()) {
            return Err(ShadowDispositionPreflightError::DuplicateCaseDeclarationId);
        }
        let fact = CaseItemFactV1 {
            candidate_id: item.candidate_id().to_string(),
            observation_root_id: item.observation_root_id().to_string(),
            declaration_id: declaration.declaration_id().to_string(),
            relation: item.declared_relation(),
            current_runtime_relevant: item.current_runtime_relevant(),
        };
        if result.insert(fact.candidate_id.clone(), fact).is_some() {
            return Err(ShadowDispositionPreflightError::DuplicateCaseCandidateId);
        }
    }
    Ok(result)
}

fn validate_eligibility_join(
    proposition_id: &str,
    expected_profile_digest: &str,
    eligible_declarations: &[DispositionEligibleRelationDeclarationV1],
    case_by_declaration: &BTreeMap<String, CaseItemFactV1>,
    interpretation_lineage: &InterpretationLineageV1,
) -> Result<Vec<ShadowDispositionEligibilityBindingV1>, ShadowDispositionPreflightError> {
    if eligible_declarations.len() != case_by_declaration.len() {
        return Err(ShadowDispositionPreflightError::EligibilitySetCardinalityMismatch);
    }

    let expected_context = interpretation_lineage.eligibility_context_commitment();
    let mut eligibility_by_declaration = BTreeMap::<String, String>::new();
    let mut eligibility_ids = BTreeSet::new();
    for eligible in eligible_declarations {
        if eligible.profile_contract_digest() != expected_profile_digest {
            return Err(ShadowDispositionPreflightError::RelationEligibilityProfileMismatch);
        }
        if eligible.proposition_id() != proposition_id {
            return Err(ShadowDispositionPreflightError::EligibleDeclarationPropositionMismatch);
        }
        if eligible.context_commitment() != expected_context {
            return Err(ShadowDispositionPreflightError::EligibilityContextMismatch);
        }
        let declaration_id = eligible.declaration().declaration_id();
        if !case_by_declaration.contains_key(declaration_id) {
            return Err(ShadowDispositionPreflightError::UnexpectedEligibleDeclaration);
        }
        if !eligibility_ids.insert(eligible.eligibility_id().to_string()) {
            return Err(ShadowDispositionPreflightError::DuplicateEligibilityId);
        }
        if eligibility_by_declaration
            .insert(declaration_id.to_string(), eligible.eligibility_id().to_string())
            .is_some()
        {
            return Err(ShadowDispositionPreflightError::DuplicateEligibleDeclarationId);
        }
    }

    if interpretation_lineage.entries().len() != eligibility_by_declaration.len() {
        return Err(ShadowDispositionPreflightError::InterpretationEntryCardinalityMismatch);
    }
    for entry in interpretation_lineage.entries() {
        let Some(expected_eligibility_id) = eligibility_by_declaration.get(entry.declaration_id())
        else {
            return Err(ShadowDispositionPreflightError::UnexpectedInterpretationDeclaration);
        };
        if expected_eligibility_id.as_str() != entry.eligibility_id() {
            return Err(ShadowDispositionPreflightError::InterpretationEligibilityMismatch);
        }
    }

    Ok(eligibility_by_declaration
        .into_iter()
        .map(|(declaration_id, eligibility_id)| ShadowDispositionEligibilityBindingV1 {
            declaration_id,
            eligibility_id,
        })
        .collect())
}

/// Returns the exact selected candidate set for cross-binding to the same slot's
/// interpretation witness.
fn validate_evidence_witness_slot(
    witness: Option<&IndependentEvidenceSetWitnessV1>,
    side: WitnessSideV1,
    expected_profile_digest: &str,
    case_by_candidate: &BTreeMap<String, CaseItemFactV1>,
) -> Result<Option<BTreeSet<String>>, ShadowDispositionPreflightError> {
    let Some(witness) = witness else {
        return Ok(None);
    };
    if witness.profile_contract_digest() != expected_profile_digest {
        return Err(ShadowDispositionPreflightError::EvidenceWitnessProfileMismatch);
    }

    let mut selected = BTreeSet::new();
    for witness_item in witness.items() {
        let Some(case_item) = case_by_candidate.get(witness_item.evidence_id()) else {
            return Err(ShadowDispositionPreflightError::EvidenceWitnessUnknownCaseItem);
        };
        if !case_item.current_runtime_relevant {
            return Err(ShadowDispositionPreflightError::EvidenceWitnessItemNotCurrent);
        }
        if !relation_matches_side(case_item.relation, side) {
            return Err(ShadowDispositionPreflightError::EvidenceWitnessRelationSideMismatch);
        }
        // Current runtime candidates have one exact observation-event root. A
        // future composite-evidence preflight must use a new profile rather than
        // silently widening this V1 meaning.
        if witness_item.root_ids().len() != 1
            || witness_item.root_ids()[0].as_str() != case_item.observation_root_id.as_str()
        {
            return Err(ShadowDispositionPreflightError::EvidenceWitnessRootBindingMismatch);
        }
        selected.insert(witness_item.evidence_id().to_string());
    }
    Ok(Some(selected))
}

fn validate_interpretation_witness_slot(
    witness: Option<&IndependentInterpretationRootSetWitnessV1>,
    evidence_candidates: Option<&BTreeSet<String>>,
    _side: WitnessSideV1,
    proposition_id: &str,
    expected_profile_digest: &str,
    interpretation_lineage: &InterpretationLineageV1,
    case_by_candidate: &BTreeMap<String, CaseItemFactV1>,
    lineage_root_by_declaration: &BTreeMap<String, String>,
) -> Result<(), ShadowDispositionPreflightError> {
    let Some(witness) = witness else {
        return Ok(());
    };
    let Some(evidence_candidates) = evidence_candidates else {
        return Err(ShadowDispositionPreflightError::InterpretationWitnessMissingEvidenceWitness);
    };
    if witness.profile_contract_digest() != expected_profile_digest {
        return Err(ShadowDispositionPreflightError::InterpretationWitnessProfileMismatch);
    }
    if witness.proposition_id() != proposition_id {
        return Err(ShadowDispositionPreflightError::InterpretationWitnessPropositionMismatch);
    }
    if witness.interpretation_lineage_id() != interpretation_lineage.lineage_id() {
        return Err(ShadowDispositionPreflightError::InterpretationWitnessLineageMismatch);
    }

    let mut expected_roots = BTreeSet::new();
    for candidate_id in evidence_candidates {
        let case_item = case_by_candidate
            .get(candidate_id)
            .ok_or(ShadowDispositionPreflightError::EvidenceWitnessUnknownCaseItem)?;
        let root_id = lineage_root_by_declaration
            .get(case_item.declaration_id.as_str())
            .ok_or(ShadowDispositionPreflightError::MissingInterpretationRootForCaseDeclaration)?;
        expected_roots.insert(root_id.clone());
    }
    let actual_roots = witness.root_ids().iter().cloned().collect::<BTreeSet<_>>();
    if actual_roots != expected_roots {
        return Err(ShadowDispositionPreflightError::InterpretationWitnessRootSetMismatch);
    }
    Ok(())
}

fn relation_matches_side(relation: EvidenceRelationKindV1, side: WitnessSideV1) -> bool {
    match side {
        WitnessSideV1::Support => matches!(
            relation,
            EvidenceRelationKindV1::Supports | EvidenceRelationKindV1::Corroborates
        ),
        WitnessSideV1::Opposition => matches!(
            relation,
            EvidenceRelationKindV1::Contradicts
                | EvidenceRelationKindV1::Weakens
                | EvidenceRelationKindV1::Defeats
        ),
        WitnessSideV1::Defeater => relation == EvidenceRelationKindV1::Defeats,
    }
}

fn witness_id(witness: Option<&IndependentEvidenceSetWitnessV1>) -> Option<String> {
    witness.map(|value| value.witness_id().to_string())
}

fn interpretation_witness_id(
    witness: Option<&IndependentInterpretationRootSetWitnessV1>,
) -> Option<String> {
    witness.map(|value| value.witness_id().to_string())
}

struct PreflightIdentityInputsV1<'a> {
    profile_contract_digest: &'a str,
    proposition_id: &'a str,
    case_id: &'a str,
    case_scope_digest: &'a str,
    evidence_lineage_graph_id: &'a str,
    relation_eligibility_context_commitment: &'a str,
    support_evidence_witness_id: Option<&'a str>,
    opposition_evidence_witness_id: Option<&'a str>,
    defeater_evidence_witness_id: Option<&'a str>,
    support_interpretation_witness_id: Option<&'a str>,
    opposition_interpretation_witness_id: Option<&'a str>,
    defeater_interpretation_witness_id: Option<&'a str>,
    eligibility_bindings: &'a [ShadowDispositionEligibilityBindingV1],
    interpretation_lineage_id: &'a str,
    effective_policy_id: &'a str,
    experiment_contract_digest: &'a str,
}

fn preflight_id_v1(value: &PreflightIdentityInputsV1<'_>) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFLIGHT_ID_DOMAIN);
    hash_text(&mut hasher, b"profile_contract_digest", value.profile_contract_digest);
    hash_bytes(
        &mut hasher,
        b"schema_version",
        &SHADOW_DISPOSITION_PREFLIGHT_SCHEMA_VERSION.to_le_bytes(),
    );
    hash_text(&mut hasher, b"proposition_id", value.proposition_id);
    hash_text(&mut hasher, b"case_id", value.case_id);
    hash_text(&mut hasher, b"case_scope_digest", value.case_scope_digest);
    hash_text(
        &mut hasher,
        b"evidence_lineage_graph_id",
        value.evidence_lineage_graph_id,
    );
    hash_text(
        &mut hasher,
        b"relation_eligibility_context_commitment",
        value.relation_eligibility_context_commitment,
    );
    hash_option_text(
        &mut hasher,
        b"support_evidence_witness_id",
        value.support_evidence_witness_id,
    );
    hash_option_text(
        &mut hasher,
        b"opposition_evidence_witness_id",
        value.opposition_evidence_witness_id,
    );
    hash_option_text(
        &mut hasher,
        b"defeater_evidence_witness_id",
        value.defeater_evidence_witness_id,
    );
    hash_option_text(
        &mut hasher,
        b"support_interpretation_witness_id",
        value.support_interpretation_witness_id,
    );
    hash_option_text(
        &mut hasher,
        b"opposition_interpretation_witness_id",
        value.opposition_interpretation_witness_id,
    );
    hash_option_text(
        &mut hasher,
        b"defeater_interpretation_witness_id",
        value.defeater_interpretation_witness_id,
    );
    hash_count(
        &mut hasher,
        b"eligibility_binding_count",
        value.eligibility_bindings.len(),
    );
    for binding in value.eligibility_bindings {
        hash_text(&mut hasher, b"declaration_id", &binding.declaration_id);
        hash_text(&mut hasher, b"eligibility_id", &binding.eligibility_id);
    }
    hash_text(
        &mut hasher,
        b"interpretation_lineage_id",
        value.interpretation_lineage_id,
    );
    hash_text(&mut hasher, b"effective_policy_id", value.effective_policy_id);
    hash_text(
        &mut hasher,
        b"registered_experiment_contract_digest",
        value.experiment_contract_digest,
    );
    format!("blake3:{}", hasher.finalize().to_hex())
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
pub enum ShadowDispositionPreflightError {
    PolicyPropositionMismatch,
    BoundCaseProfileMismatch,
    InterpretationLineageProfileMismatch,
    InterpretationLineagePropositionMismatch,
    ExperimentContractIntegrity(String),
    ExperimentContractSchemaMismatch,
    ExperimentContractDigestMismatch,
    CaseItemCeilingExceeded { actual: usize, ceiling: u32 },
    InterpretationPairCeilingExceeded { actual: usize, ceiling: u32 },
    BoundCaseInternalCardinalityMismatch,
    DuplicateCaseCandidateId,
    DuplicateCaseDeclarationId,
    CaseDeclarationCandidateMismatch { candidate_id: String, relation_evidence_id: String },
    CaseDeclarationRelationMismatch { candidate_id: String },
    EligibilitySetCardinalityMismatch,
    RelationEligibilityProfileMismatch,
    EligibleDeclarationPropositionMismatch,
    EligibilityContextMismatch,
    UnexpectedEligibleDeclaration,
    DuplicateEligibilityId,
    DuplicateEligibleDeclarationId,
    InterpretationEntryCardinalityMismatch,
    UnexpectedInterpretationDeclaration,
    InterpretationEligibilityMismatch,
    EvidenceWitnessProfileMismatch,
    EvidenceWitnessUnknownCaseItem,
    EvidenceWitnessItemNotCurrent,
    EvidenceWitnessRelationSideMismatch,
    EvidenceWitnessRootBindingMismatch,
    InterpretationWitnessMissingEvidenceWitness,
    InterpretationWitnessProfileMismatch,
    InterpretationWitnessPropositionMismatch,
    InterpretationWitnessLineageMismatch,
    MissingInterpretationRootForCaseDeclaration,
    InterpretationWitnessRootSetMismatch,
}

impl std::fmt::Display for ShadowDispositionPreflightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PolicyPropositionMismatch => f.write_str("policy proposition does not match bound case"),
            Self::BoundCaseProfileMismatch => f.write_str("bound-case profile does not match policy"),
            Self::InterpretationLineageProfileMismatch => f.write_str("interpretation-lineage profile does not match policy"),
            Self::InterpretationLineagePropositionMismatch => f.write_str("interpretation lineage proposition does not match case"),
            Self::ExperimentContractIntegrity(error) => write!(f, "registered experiment contract failed integrity: {error}"),
            Self::ExperimentContractSchemaMismatch => f.write_str("policy experiment-contract schema mismatch"),
            Self::ExperimentContractDigestMismatch => f.write_str("actual registered experiment contract does not match policy"),
            Self::CaseItemCeilingExceeded { actual, ceiling } => write!(f, "case has {actual} items; policy ceiling is {ceiling}"),
            Self::InterpretationPairCeilingExceeded { actual, ceiling } => write!(f, "interpretation lineage has {actual} root pairs; policy ceiling is {ceiling}"),
            Self::BoundCaseInternalCardinalityMismatch => f.write_str("bound case item/declaration cardinality mismatch"),
            Self::DuplicateCaseCandidateId => f.write_str("bound case contains duplicate candidate id"),
            Self::DuplicateCaseDeclarationId => f.write_str("bound case contains duplicate declaration id"),
            Self::CaseDeclarationCandidateMismatch { candidate_id, relation_evidence_id } => write!(f, "case candidate {candidate_id} is bound to relation evidence {relation_evidence_id}"),
            Self::CaseDeclarationRelationMismatch { candidate_id } => write!(f, "case candidate {candidate_id} relation kind differs from bound declaration"),
            Self::EligibilitySetCardinalityMismatch => f.write_str("eligible declaration set does not exactly cover case declarations"),
            Self::RelationEligibilityProfileMismatch => f.write_str("relation eligibility profile does not match policy"),
            Self::EligibleDeclarationPropositionMismatch => f.write_str("eligible declaration targets a different proposition"),
            Self::EligibilityContextMismatch => f.write_str("eligible declaration was issued under a different context"),
            Self::UnexpectedEligibleDeclaration => f.write_str("eligible declaration is not in the bound case"),
            Self::DuplicateEligibilityId => f.write_str("duplicate eligibility id"),
            Self::DuplicateEligibleDeclarationId => f.write_str("duplicate eligible declaration id"),
            Self::InterpretationEntryCardinalityMismatch => f.write_str("interpretation entries do not exactly cover eligible declarations"),
            Self::UnexpectedInterpretationDeclaration => f.write_str("interpretation lineage contains an unexpected declaration"),
            Self::InterpretationEligibilityMismatch => f.write_str("interpretation lineage binds the wrong current eligibility id"),
            Self::EvidenceWitnessProfileMismatch => f.write_str("evidence witness profile does not match policy"),
            Self::EvidenceWitnessUnknownCaseItem => f.write_str("evidence witness contains a non-case item"),
            Self::EvidenceWitnessItemNotCurrent => f.write_str("evidence witness contains a non-current case item"),
            Self::EvidenceWitnessRelationSideMismatch => f.write_str("evidence witness contains an item from the wrong relation side"),
            Self::EvidenceWitnessRootBindingMismatch => f.write_str("evidence witness does not bind the exact single case observation root"),
            Self::InterpretationWitnessMissingEvidenceWitness => f.write_str("interpretation witness has no corresponding evidence witness slot"),
            Self::InterpretationWitnessProfileMismatch => f.write_str("interpretation witness profile does not match effective policy"),
            Self::InterpretationWitnessPropositionMismatch => f.write_str("interpretation witness targets a different proposition"),
            Self::InterpretationWitnessLineageMismatch => f.write_str("interpretation witness belongs to a different interpretation lineage"),
            Self::MissingInterpretationRootForCaseDeclaration => f.write_str("case declaration has no interpretation-root mapping"),
            Self::InterpretationWitnessRootSetMismatch => f.write_str("interpretation witness roots do not exactly interpret the same-slot evidence witness items"),
        }
    }
}
impl std::error::Error for ShadowDispositionPreflightError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_opposition_and_defeater_relation_classes_are_explicit() {
        assert!(relation_matches_side(EvidenceRelationKindV1::Supports, WitnessSideV1::Support));
        assert!(relation_matches_side(EvidenceRelationKindV1::Corroborates, WitnessSideV1::Support));
        assert!(!relation_matches_side(EvidenceRelationKindV1::Weakens, WitnessSideV1::Support));
        assert!(relation_matches_side(EvidenceRelationKindV1::Contradicts, WitnessSideV1::Opposition));
        assert!(relation_matches_side(EvidenceRelationKindV1::Weakens, WitnessSideV1::Opposition));
        assert!(relation_matches_side(EvidenceRelationKindV1::Defeats, WitnessSideV1::Opposition));
        assert!(relation_matches_side(EvidenceRelationKindV1::Defeats, WitnessSideV1::Defeater));
        assert!(!relation_matches_side(EvidenceRelationKindV1::Weakens, WitnessSideV1::Defeater));
    }

    #[test]
    fn canonical_set_equality_is_order_independent() {
        let left = ["b".to_string(), "a".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>();
        let right = ["a".to_string(), "b".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert_eq!(left, right);
    }

    #[test]
    fn preflight_profile_identity_is_stable() {
        let first = shadow_disposition_preflight_profile_digest_v1();
        assert_eq!(first, shadow_disposition_preflight_profile_digest_v1());
        assert!(first.starts_with("blake3:"));
    }
}
