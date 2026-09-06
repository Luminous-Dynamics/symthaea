// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! RCA-003b.3b: exact cross-artifact preflight for shadow disposition.
//!
//! This crate performs no disposition. It proves that individually issued RCA
//! artifacts belong to one exact proposition/case/currentness/qualification/
//! policy lineage before a future pure engine may inspect them together.
//!
//! The issued preflight is deliberately non-deserializable. Archived bytes are
//! audit material; current evaluation eligibility must be recomputed from current
//! issued inputs.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::{HashMap, HashSet};
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
    "evidence_witness_items_must_be_current_case_candidates_with_exact_observation_root_binding\n",
    "evidence_witness_slot_relations_are_support_or_opposition_or_defeater_scoped\n",
    "interpretation_witnesses_must_bind_exact_lineage_and_have_current_slot_compatible_declarations\n",
    "artifact_profiles_must_match_effective_preregistered_policy_bindings\n",
    "actual_case_items_and_interpretation_pairs_must_respect_policy_resource_ceilings\n",
    "actual_registered_experiment_contract_must_verify_and_match_policy_digest\n",
    "preflight_id=blake3_explicit_complete_cross_artifact_binding_v1\n",
    "issued_preflight=is_private_non_deserializable_shadow_capability\n",
    "preflight_is_not_disposition_belief_workspace_action_or_promotion_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-shadow-disposition-preflight-contract:v1\0";
const PREFLIGHT_ID_DOMAIN: &[u8] = b"symthaea:rca-shadow-disposition-preflight:v1\0";

/// Fixed semantic slots prevent an unbounded witness collection from becoming an
/// unregistered resource dimension. The same issued witness may be supplied in
/// more than one slot; reuse does not multiply evidence.
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

/// Issued exact input binding for a future pure shadow-disposition engine.
///
/// Private fields plus no `Deserialize` are deliberate. A stored report cannot
/// restore current eligibility; all joins must be rerun from currently issued
/// artifacts.
#[must_use = "shadow-disposition preflight is an issued evaluation capability and should be inspected"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ShadowDispositionPreflightV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    preflight_id: String,
    proposition_id: String,
    case_id: String,
    case_scope_digest: String,
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

#[derive(Debug, Clone)]
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

/// Perform all cross-artifact joins needed before a future disposition engine may
/// inspect these artifacts together. No threshold comparison or disposition is
/// performed here.
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

    if policy.proposition_id != proposition_id {
        return Err(ShadowDispositionPreflightError::PolicyPropositionMismatch);
    }
    if bound_case.profile_contract_digest() != policy.bound_case_profile_digest {
        return Err(ShadowDispositionPreflightError::BoundCaseProfileMismatch);
    }
    if interpretation_lineage.profile_contract_digest()
        != policy.interpretation_lineage_profile_digest
    {
        return Err(ShadowDispositionPreflightError::InterpretationLineageProfileMismatch);
    }
    if interpretation_lineage.proposition_id() != proposition_id {
        return Err(ShadowDispositionPreflightError::InterpretationLineagePropositionMismatch);
    }

    experiment_contract
        .verify_integrity()
        .map_err(|error| ShadowDispositionPreflightError::ExperimentContractIntegrity(error.to_string()))?;
    if policy.evaluation.experiment_contract_schema_version != EXPERIMENT_CONTRACT_SCHEMA_VERSION {
        return Err(ShadowDispositionPreflightError::ExperimentContractSchemaMismatch);
    }
    if policy.evaluation.registered_experiment_contract_digest != experiment_contract.contract_digest()
    {
        return Err(ShadowDispositionPreflightError::ExperimentContractDigestMismatch);
    }

    if structural_case.items().len() > policy.resources.max_case_items as usize {
        return Err(ShadowDispositionPreflightError::CaseItemCeilingExceeded {
            actual: structural_case.items().len(),
            ceiling: policy.resources.max_case_items,
        });
    }
    if interpretation_lineage.root_pair_assessments().len()
        > policy.resources.max_interpretation_pairs as usize
    {
        return Err(ShadowDispositionPreflightError::InterpretationPairCeilingExceeded {
            actual: interpretation_lineage.root_pair_assessments().len(),
            ceiling: policy.resources.max_interpretation_pairs,
        });
    }

    let case_items = build_case_item_facts(bound_case)?;
    let case_by_candidate = case_items
        .iter()
        .map(|item| (item.candidate_id.as_str(), item))
        .collect::<HashMap<_, _>>();
    let case_by_declaration = case_items
        .iter()
        .map(|item| (item.declaration_id.as_str(), item))
        .collect::<HashMap<_, _>>();

    let eligibility_bindings = validate_eligibility_join(
        proposition_id,
        &policy.relation_eligibility_profile_digest,
        eligible_declarations,
        &case_by_declaration,
        interpretation_lineage,
    )?;

    validate_evidence_witness_slot(
        evidence_witnesses.support,
        WitnessSideV1::Support,
        &policy.evidence_set_witness_profile_digest,
        &case_by_candidate,
    )?;
    validate_evidence_witness_slot(
        evidence_witnesses.opposition,
        WitnessSideV1::Opposition,
        &policy.evidence_set_witness_profile_digest,
        &case_by_candidate,
    )?;
    validate_evidence_witness_slot(
        evidence_witnesses.defeater,
        WitnessSideV1::Defeater,
        &policy.evidence_set_witness_profile_digest,
        &case_by_candidate,
    )?;

    validate_interpretation_witness_slot(
        interpretation_witnesses.support,
        WitnessSideV1::Support,
        proposition_id,
        effective_policy.interpretation_set_witness_profile_digest(),
        interpretation_lineage,
        &case_by_declaration,
    )?;
    validate_interpretation_witness_slot(
        interpretation_witnesses.opposition,
        WitnessSideV1::Opposition,
        proposition_id,
        effective_policy.interpretation_set_witness_profile_digest(),
        interpretation_lineage,
        &case_by_declaration,
    )?;
    validate_interpretation_witness_slot(
        interpretation_witnesses.defeater,
        WitnessSideV1::Defeater,
        proposition_id,
        effective_policy.interpretation_set_witness_profile_digest(),
        interpretation_lineage,
        &case_by_declaration,
    )?;

    let profile_contract_digest = shadow_disposition_preflight_profile_digest_v1();
    let relation_eligibility_context_commitment =
        interpretation_lineage.eligibility_context_commitment().to_string();
    let support_evidence_witness_id = evidence_witnesses
        .support
        .map(|witness| witness.witness_id().to_string());
    let opposition_evidence_witness_id = evidence_witnesses
        .opposition
        .map(|witness| witness.witness_id().to_string());
    let defeater_evidence_witness_id = evidence_witnesses
        .defeater
        .map(|witness| witness.witness_id().to_string());
    let support_interpretation_witness_id = interpretation_witnesses
        .support
        .map(|witness| witness.witness_id().to_string());
    let opposition_interpretation_witness_id = interpretation_witnesses
        .opposition
        .map(|witness| witness.witness_id().to_string());
    let defeater_interpretation_witness_id = interpretation_witnesses
        .defeater
        .map(|witness| witness.witness_id().to_string());

    let preflight_id = shadow_disposition_preflight_id_v1(
        &profile_contract_digest,
        proposition_id,
        bound_case.case_id(),
        structural_case.case_scope_digest(),
        &relation_eligibility_context_commitment,
        support_evidence_witness_id.as_deref(),
        opposition_evidence_witness_id.as_deref(),
        defeater_evidence_witness_id.as_deref(),
        support_interpretation_witness_id.as_deref(),
        opposition_interpretation_witness_id.as_deref(),
        defeater_interpretation_witness_id.as_deref(),
        &eligibility_bindings,
        interpretation_lineage.lineage_id(),
        effective_policy.effective_policy_id(),
        experiment_contract.contract_digest(),
    );

    Ok(ShadowDispositionPreflightV1 {
        schema_version: SHADOW_DISPOSITION_PREFLIGHT_SCHEMA_VERSION,
        profile: SHADOW_DISPOSITION_PREFLIGHT_PROFILE_V1.to_string(),
        profile_contract_digest,
        preflight_id,
        proposition_id: proposition_id.to_string(),
        case_id: bound_case.case_id().to_string(),
        case_scope_digest: structural_case.case_scope_digest().to_string(),
        relation_eligibility_context_commitment,
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

fn build_case_item_facts(
    bound_case: &BoundShadowEvidenceCaseV1,
) -> Result<Vec<CaseItemFactV1>, ShadowDispositionPreflightError> {
    let items = bound_case.structural_case().items();
    let declarations = bound_case.relation_declarations();
    if items.len() != declarations.len() {
        return Err(ShadowDispositionPreflightError::BoundCaseInternalCardinalityMismatch);
    }

    let mut seen_candidates = HashSet::with_capacity(items.len());
    let mut seen_declarations = HashSet::with_capacity(items.len());
    let mut facts = Vec::with_capacity(items.len());

    for (item, declaration) in items.iter().zip(declarations) {
        let relation = declaration.relation().as_raw();
        if relation.evidence_id != item.candidate_id() {
            return Err(ShadowDispositionPreflightError::CaseDeclarationCandidateMismatch {
                candidate_id: item.candidate_id().to_string(),
                relation_evidence_id: relation.evidence_id.clone(),
            });
        }
        if !seen_candidates.insert(item.candidate_id()) {
            return Err(ShadowDispositionPreflightError::DuplicateCaseCandidateId {
                candidate_id: item.candidate_id().to_string(),
            });
        }
        if !seen_declarations.insert(declaration.declaration_id()) {
            return Err(ShadowDispositionPreflightError::DuplicateCaseDeclarationId {
                declaration_id: declaration.declaration_id().to_string(),
            });
        }
        facts.push(CaseItemFactV1 {
            candidate_id: item.candidate_id().to_string(),
            observation_root_id: item.observation_root_id().to_string(),
            declaration_id: declaration.declaration_id().to_string(),
            relation: item.declared_relation(),
            current_runtime_relevant: item.current_runtime_relevant(),
        });
    }

    facts.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    Ok(facts)
}

fn validate_eligibility_join(
    proposition_id: &str,
    expected_profile_digest: &str,
    eligible_declarations: &[DispositionEligibleRelationDeclarationV1],
    case_by_declaration: &HashMap<&str, &CaseItemFactV1>,
    interpretation_lineage: &InterpretationLineageV1,
) -> Result<Vec<ShadowDispositionEligibilityBindingV1>, ShadowDispositionPreflightError> {
    if eligible_declarations.len() != case_by_declaration.len() {
        return Err(ShadowDispositionPreflightError::EligibilitySetCardinalityMismatch {
            case_declarations: case_by_declaration.len(),
            eligible_declarations: eligible_declarations.len(),
        });
    }

    let mut seen_declarations = HashSet::with_capacity(eligible_declarations.len());
    let mut seen_eligibility_ids = HashSet::with_capacity(eligible_declarations.len());
    let mut eligibility_by_declaration = HashMap::with_capacity(eligible_declarations.len());
    let expected_context = interpretation_lineage.eligibility_context_commitment();

    for eligible in eligible_declarations {
        if eligible.profile_contract_digest() != expected_profile_digest {
            return Err(ShadowDispositionPreflightError::RelationEligibilityProfileMismatch {
                eligibility_id: eligible.eligibility_id().to_string(),
            });
        }
        if eligible.proposition_id() != proposition_id {
            return Err(ShadowDispositionPreflightError::EligibleDeclarationPropositionMismatch {
                eligibility_id: eligible.eligibility_id().to_string(),
            });
        }
        if eligible.context_commitment() != expected_context {
            return Err(ShadowDispositionPreflightError::EligibilityContextMismatch {
                eligibility_id: eligible.eligibility_id().to_string(),
            });
        }

        let declaration_id = eligible.declaration().declaration_id();
        if !case_by_declaration.contains_key(declaration_id) {
            return Err(ShadowDispositionPreflightError::UnexpectedEligibleDeclaration {
                declaration_id: declaration_id.to_string(),
            });
        }
        if !seen_declarations.insert(declaration_id) {
            return Err(ShadowDispositionPreflightError::DuplicateEligibleDeclarationId {
                declaration_id: declaration_id.to_string(),
            });
        }
        if !seen_eligibility_ids.insert(eligible.eligibility_id()) {
            return Err(ShadowDispositionPreflightError::DuplicateEligibilityId {
                eligibility_id: eligible.eligibility_id().to_string(),
            });
        }
        eligibility_by_declaration.insert(declaration_id, eligible.eligibility_id());
    }

    if interpretation_lineage.entries().len() != eligibility_by_declaration.len() {
        return Err(ShadowDispositionPreflightError::InterpretationEntryCardinalityMismatch {
            lineage_entries: interpretation_lineage.entries().len(),
            eligible_declarations: eligibility_by_declaration.len(),
        });
    }

    for entry in interpretation_lineage.entries() {
        let Some(expected_eligibility_id) = eligibility_by_declaration.get(entry.declaration_id()) else {
            return Err(ShadowDispositionPreflightError::UnexpectedInterpretationDeclaration {
                declaration_id: entry.declaration_id().to_string(),
            });
        };
        if *expected_eligibility_id != entry.eligibility_id() {
            return Err(ShadowDispositionPreflightError::InterpretationEligibilityMismatch {
                declaration_id: entry.declaration_id().to_string(),
                expected_eligibility_id: (*expected_eligibility_id).to_string(),
                found_eligibility_id: entry.eligibility_id().to_string(),
            });
        }
    }

    let mut bindings = eligibility_by_declaration
        .into_iter()
        .map(|(declaration_id, eligibility_id)| ShadowDispositionEligibilityBindingV1 {
            declaration_id: declaration_id.to_string(),
            eligibility_id: eligibility_id.to_string(),
        })
        .collect::<Vec<_>>();
    bindings.sort_by(|left, right| left.declaration_id.cmp(&right.declaration_id));
    Ok(bindings)
}

fn validate_evidence_witness_slot(
    witness: Option<&IndependentEvidenceSetWitnessV1>,
    side: WitnessSideV1,
    expected_profile_digest: &str,
    case_by_candidate: &HashMap<&str, &CaseItemFactV1>,
) -> Result<(), ShadowDispositionPreflightError> {
    let Some(witness) = witness else {
        return Ok(());
    };
    if witness.profile_contract_digest() != expected_profile_digest {
        return Err(ShadowDispositionPreflightError::EvidenceWitnessProfileMismatch {
            witness_id: witness.witness_id().to_string(),
        });
    }

    for witness_item in witness.items() {
        let Some(case_item) = case_by_candidate.get(witness_item.evidence_id()) else {
            return Err(ShadowDispositionPreflightError::EvidenceWitnessUnknownCaseItem {
                witness_id: witness.witness_id().to_string(),
                evidence_id: witness_item.evidence_id().to_string(),
            });
        };
        if !case_item.current_runtime_relevant {
            return Err(ShadowDispositionPreflightError::EvidenceWitnessItemNotCurrent {
                witness_id: witness.witness_id().to_string(),
                evidence_id: witness_item.evidence_id().to_string(),
            });
        }
        if !relation_matches_side(case_item.relation, side) {
            return Err(ShadowDispositionPreflightError::EvidenceWitnessRelationSideMismatch {
                witness_id: witness.witness_id().to_string(),
                evidence_id: witness_item.evidence_id().to_string(),
            });
        }
        if witness_item.root_ids().len() != 1
            || witness_item.root_ids()[0] != case_item.observation_root_id
        {
            return Err(ShadowDispositionPreflightError::EvidenceWitnessRootBindingMismatch {
                witness_id: witness.witness_id().to_string(),
                evidence_id: witness_item.evidence_id().to_string(),
                expected_observation_root_id: case_item.observation_root_id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_interpretation_witness_slot(
    witness: Option<&IndependentInterpretationRootSetWitnessV1>,
    side: WitnessSideV1,
    proposition_id: &str,
    expected_profile_digest: &str,
    interpretation_lineage: &InterpretationLineageV1,
    case_by_declaration: &HashMap<&str, &CaseItemFactV1>,
) -> Result<(), ShadowDispositionPreflightError> {
    let Some(witness) = witness else {
        return Ok(());
    };
    if witness.profile_contract_digest() != expected_profile_digest {
        return Err(ShadowDispositionPreflightError::InterpretationWitnessProfileMismatch {
            witness_id: witness.witness_id().to_string(),
        });
    }
    if witness.proposition_id() != proposition_id {
        return Err(ShadowDispositionPreflightError::InterpretationWitnessPropositionMismatch {
            witness_id: witness.witness_id().to_string(),
        });
    }
    if witness.interpretation_lineage_id() != interpretation_lineage.lineage_id() {
        return Err(ShadowDispositionPreflightError::InterpretationWitnessLineageMismatch {
            witness_id: witness.witness_id().to_string(),
        });
    }

    for root_id in witness.root_ids() {
        let compatible = interpretation_lineage.entries().iter().any(|entry| {
            if entry.interpretation_root_id() != root_id {
                return false;
            }
            case_by_declaration
                .get(entry.declaration_id())
                .is_some_and(|case_item| {
                    case_item.current_runtime_relevant && relation_matches_side(case_item.relation, side)
                })
        });
        if !compatible {
            return Err(
                ShadowDispositionPreflightError::InterpretationWitnessRootHasNoCurrentSideDeclaration {
                    witness_id: witness.witness_id().to_string(),
                    interpretation_root_id: root_id.clone(),
                },
            );
        }
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

#[allow(clippy::too_many_arguments)]
fn shadow_disposition_preflight_id_v1(
    profile_contract_digest: &str,
    proposition_id: &str,
    case_id: &str,
    case_scope_digest: &str,
    relation_eligibility_context_commitment: &str,
    support_evidence_witness_id: Option<&str>,
    opposition_evidence_witness_id: Option<&str>,
    defeater_evidence_witness_id: Option<&str>,
    support_interpretation_witness_id: Option<&str>,
    opposition_interpretation_witness_id: Option<&str>,
    defeater_interpretation_witness_id: Option<&str>,
    eligibility_bindings: &[ShadowDispositionEligibilityBindingV1],
    interpretation_lineage_id: &str,
    effective_policy_id: &str,
    experiment_contract_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFLIGHT_ID_DOMAIN);
    hash_text(
        &mut hasher,
        b"profile_contract_digest",
        profile_contract_digest,
    );
    hash_bytes(
        &mut hasher,
        b"schema_version",
        &SHADOW_DISPOSITION_PREFLIGHT_SCHEMA_VERSION.to_le_bytes(),
    );
    hash_text(&mut hasher, b"proposition_id", proposition_id);
    hash_text(&mut hasher, b"case_id", case_id);
    hash_text(&mut hasher, b"case_scope_digest", case_scope_digest);
    hash_text(
        &mut hasher,
        b"relation_eligibility_context_commitment",
        relation_eligibility_context_commitment,
    );
    hash_option_text(
        &mut hasher,
        b"support_evidence_witness_id",
        support_evidence_witness_id,
    );
    hash_option_text(
        &mut hasher,
        b"opposition_evidence_witness_id",
        opposition_evidence_witness_id,
    );
    hash_option_text(
        &mut hasher,
        b"defeater_evidence_witness_id",
        defeater_evidence_witness_id,
    );
    hash_option_text(
        &mut hasher,
        b"support_interpretation_witness_id",
        support_interpretation_witness_id,
    );
    hash_option_text(
        &mut hasher,
        b"opposition_interpretation_witness_id",
        opposition_interpretation_witness_id,
    );
    hash_option_text(
        &mut hasher,
        b"defeater_interpretation_witness_id",
        defeater_interpretation_witness_id,
    );
    hash_count(
        &mut hasher,
        b"eligibility_binding_count",
        eligibility_bindings.len(),
    );
    for binding in eligibility_bindings {
        hash_text(&mut hasher, b"declaration_id", &binding.declaration_id);
        hash_text(&mut hasher, b"eligibility_id", &binding.eligibility_id);
    }
    hash_text(
        &mut hasher,
        b"interpretation_lineage_id",
        interpretation_lineage_id,
    );
    hash_text(&mut hasher, b"effective_policy_id", effective_policy_id);
    hash_text(
        &mut hasher,
        b"registered_experiment_contract_digest",
        experiment_contract_digest,
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
    CaseItemCeilingExceeded {
        actual: usize,
        ceiling: u32,
    },
    InterpretationPairCeilingExceeded {
        actual: usize,
        ceiling: u32,
    },
    BoundCaseInternalCardinalityMismatch,
    CaseDeclarationCandidateMismatch {
        candidate_id: String,
        relation_evidence_id: String,
    },
    DuplicateCaseCandidateId {
        candidate_id: String,
    },
    DuplicateCaseDeclarationId {
        declaration_id: String,
    },
    EligibilitySetCardinalityMismatch {
        case_declarations: usize,
        eligible_declarations: usize,
    },
    RelationEligibilityProfileMismatch {
        eligibility_id: String,
    },
    EligibleDeclarationPropositionMismatch {
        eligibility_id: String,
    },
    EligibilityContextMismatch {
        eligibility_id: String,
    },
    UnexpectedEligibleDeclaration {
        declaration_id: String,
    },
    DuplicateEligibleDeclarationId {
        declaration_id: String,
    },
    DuplicateEligibilityId {
        eligibility_id: String,
    },
    InterpretationEntryCardinalityMismatch {
        lineage_entries: usize,
        eligible_declarations: usize,
    },
    UnexpectedInterpretationDeclaration {
        declaration_id: String,
    },
    InterpretationEligibilityMismatch {
        declaration_id: String,
        expected_eligibility_id: String,
        found_eligibility_id: String,
    },
    EvidenceWitnessProfileMismatch {
        witness_id: String,
    },
    EvidenceWitnessUnknownCaseItem {
        witness_id: String,
        evidence_id: String,
    },
    EvidenceWitnessItemNotCurrent {
        witness_id: String,
        evidence_id: String,
    },
    EvidenceWitnessRelationSideMismatch {
        witness_id: String,
        evidence_id: String,
    },
    EvidenceWitnessRootBindingMismatch {
        witness_id: String,
        evidence_id: String,
        expected_observation_root_id: String,
    },
    InterpretationWitnessProfileMismatch {
        witness_id: String,
    },
    InterpretationWitnessPropositionMismatch {
        witness_id: String,
    },
    InterpretationWitnessLineageMismatch {
        witness_id: String,
    },
    InterpretationWitnessRootHasNoCurrentSideDeclaration {
        witness_id: String,
        interpretation_root_id: String,
    },
}

impl std::fmt::Display for ShadowDispositionPreflightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PolicyPropositionMismatch => f.write_str("policy proposition does not match bound case"),
            Self::BoundCaseProfileMismatch => f.write_str("bound-case profile does not match preregistered policy"),
            Self::InterpretationLineageProfileMismatch => f.write_str("interpretation-lineage profile does not match preregistered policy"),
            Self::InterpretationLineagePropositionMismatch => f.write_str("interpretation lineage proposition does not match bound case"),
            Self::ExperimentContractIntegrity(error) => write!(f, "registered experiment contract failed integrity: {error}"),
            Self::ExperimentContractSchemaMismatch => f.write_str("policy experiment-contract schema mismatch"),
            Self::ExperimentContractDigestMismatch => f.write_str("actual registered experiment contract does not match policy binding"),
            Self::CaseItemCeilingExceeded { actual, ceiling } => write!(f, "case contains {actual} items but policy ceiling is {ceiling}"),
            Self::InterpretationPairCeilingExceeded { actual, ceiling } => write!(f, "interpretation lineage contains {actual} root pairs but policy ceiling is {ceiling}"),
            Self::BoundCaseInternalCardinalityMismatch => f.write_str("bound case item/declaration cardinality mismatch"),
            Self::CaseDeclarationCandidateMismatch { candidate_id, relation_evidence_id } => write!(f, "case candidate {candidate_id} is bound to declaration evidence {relation_evidence_id}"),
            Self::DuplicateCaseCandidateId { candidate_id } => write!(f, "duplicate case candidate id: {candidate_id}"),
            Self::DuplicateCaseDeclarationId { declaration_id } => write!(f, "duplicate case declaration id: {declaration_id}"),
            Self::EligibilitySetCardinalityMismatch { case_declarations, eligible_declarations } => write!(f, "case has {case_declarations} declarations but preflight received {eligible_declarations} eligible declarations"),
            Self::RelationEligibilityProfileMismatch { eligibility_id } => write!(f, "eligibility {eligibility_id} uses a non-policy profile"),
            Self::EligibleDeclarationPropositionMismatch { eligibility_id } => write!(f, "eligibility {eligibility_id} targets a different proposition"),
            Self::EligibilityContextMismatch { eligibility_id } => write!(f, "eligibility {eligibility_id} was issued under a different context"),
            Self::UnexpectedEligibleDeclaration { declaration_id } => write!(f, "eligible declaration {declaration_id} is not in the bound case"),
            Self::DuplicateEligibleDeclarationId { declaration_id } => write!(f, "duplicate eligible declaration id: {declaration_id}"),
            Self::DuplicateEligibilityId { eligibility_id } => write!(f, "duplicate eligibility id: {eligibility_id}"),
            Self::InterpretationEntryCardinalityMismatch { lineage_entries, eligible_declarations } => write!(f, "interpretation lineage has {lineage_entries} entries but eligibility set has {eligible_declarations}"),
            Self::UnexpectedInterpretationDeclaration { declaration_id } => write!(f, "interpretation lineage contains unexpected declaration {declaration_id}"),
            Self::InterpretationEligibilityMismatch { declaration_id, expected_eligibility_id, found_eligibility_id } => write!(f, "interpretation entry {declaration_id} binds eligibility {found_eligibility_id}; expected {expected_eligibility_id}"),
            Self::EvidenceWitnessProfileMismatch { witness_id } => write!(f, "evidence witness {witness_id} uses a non-policy profile"),
            Self::EvidenceWitnessUnknownCaseItem { witness_id, evidence_id } => write!(f, "evidence witness {witness_id} contains non-case item {evidence_id}"),
            Self::EvidenceWitnessItemNotCurrent { witness_id, evidence_id } => write!(f, "evidence witness {witness_id} contains non-current case item {evidence_id}"),
            Self::EvidenceWitnessRelationSideMismatch { witness_id, evidence_id } => write!(f, "evidence witness {witness_id} item {evidence_id} does not belong to its declared slot"),
            Self::EvidenceWitnessRootBindingMismatch { witness_id, evidence_id, expected_observation_root_id } => write!(f, "evidence witness {witness_id} item {evidence_id} does not bind exact case observation root {expected_observation_root_id}"),
            Self::InterpretationWitnessProfileMismatch { witness_id } => write!(f, "interpretation witness {witness_id} uses a non-effective-policy profile"),
            Self::InterpretationWitnessPropositionMismatch { witness_id } => write!(f, "interpretation witness {witness_id} targets a different proposition"),
            Self::InterpretationWitnessLineageMismatch { witness_id } => write!(f, "interpretation witness {witness_id} belongs to a different interpretation lineage"),
            Self::InterpretationWitnessRootHasNoCurrentSideDeclaration { witness_id, interpretation_root_id } => write!(f, "interpretation witness {witness_id} root {interpretation_root_id} has no current case declaration compatible with its slot"),
        }
    }
}

impl std::error::Error for ShadowDispositionPreflightError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_and_opposition_relation_classes_are_explicit() {
        assert!(relation_matches_side(
            EvidenceRelationKindV1::Supports,
            WitnessSideV1::Support
        ));
        assert!(relation_matches_side(
            EvidenceRelationKindV1::Corroborates,
            WitnessSideV1::Support
        ));
        assert!(!relation_matches_side(
            EvidenceRelationKindV1::Contradicts,
            WitnessSideV1::Support
        ));
        assert!(relation_matches_side(
            EvidenceRelationKindV1::Contradicts,
            WitnessSideV1::Opposition
        ));
        assert!(relation_matches_side(
            EvidenceRelationKindV1::Defeats,
            WitnessSideV1::Opposition
        ));
    }

    #[test]
    fn defeater_slot_accepts_only_defeats() {
        assert!(relation_matches_side(
            EvidenceRelationKindV1::Defeats,
            WitnessSideV1::Defeater
        ));
        assert!(!relation_matches_side(
            EvidenceRelationKindV1::Weakens,
            WitnessSideV1::Defeater
        ));
    }

    #[test]
    fn preflight_profile_identity_is_stable() {
        let first = shadow_disposition_preflight_profile_digest_v1();
        let second = shadow_disposition_preflight_profile_digest_v1();
        assert_eq!(first, second);
        assert!(first.starts_with("blake3:"));
    }

    #[test]
    fn eligibility_binding_order_changes_never_enter_identity_after_canonicalization() {
        let mut bindings = [
            ShadowDispositionEligibilityBindingV1 {
                declaration_id: "blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
                eligibility_id: "blake3:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into(),
            },
            ShadowDispositionEligibilityBindingV1 {
                declaration_id: "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
                eligibility_id: "blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into(),
            },
        ];
        bindings.sort_by(|left, right| left.declaration_id.cmp(&right.declaration_id));
        assert!(bindings[0].declaration_id < bindings[1].declaration_id);
    }
}
