// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! RCA-003a.2: provenance-bound, content-addressed shadow evidence cases.
//!
//! `symthaea-rca-shadow-case` deliberately produces a lower-level structural
//! diagnostic case from candidates, one relevance context, lineage, and declared
//! relations. This crate adds the next boundary required before any truth or
//! disposition policy may consume that case:
//!
//! 1. every relation must be a `BoundEvidenceRelationDeclarationV1` with derived
//!    declarer/provenance identity;
//! 2. the complete issued structural case and exact declaration set receive one
//!    serializer-independent BLAKE3 `case_id`.
//!
//! A bound case is still not canonical evidence admission, truth, belief,
//! workspace authority, action authority, or recursive-improvement promotion.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::{HashMap, HashSet};
use symthaea_epistemic_governance::{
    currentness::EvidenceRelationKindV1,
    lineage::EvidenceIndependenceV1,
    relation_provenance::{
        BoundEvidenceRelationDeclarationV1, RelationDeclarationMethodV1,
    },
};
use symthaea_rca_evidence_bridge::{
    InstrumentedRuntimeEvidenceCandidateV1, ShadowObservationFieldV1,
};
use symthaea_rca_shadow_case::{
    assemble_shadow_evidence_case_v1, ShadowDeclaredRelationTopologyV1,
    ShadowEvidenceCaseError, ShadowEvidenceCaseV1,
};
use symthaea_rca_shadow_epistemics::{
    RuntimeRelevanceDefectV1, ValidatedCurrentRuntimeRelevanceContextV1,
};

pub const BOUND_SHADOW_EVIDENCE_CASE_SCHEMA_VERSION: u16 = 1;
pub const BOUND_SHADOW_EVIDENCE_CASE_PROFILE_V1: &str =
    "rca-bound-shadow-evidence-case-v1";

pub const BOUND_SHADOW_EVIDENCE_CASE_CONTRACT_V1: &str = concat!(
    "rca-bound-shadow-evidence-case-v1\n",
    "input=structural_case_inputs+bound_relation_declarations\n",
    "one_bound_declaration_per_exact_candidate\n",
    "producer_relation_reference_is_not_case_provenance_identity\n",
    "relation_declaration_id_is_identity_bearing\n",
    "structural_case_is_recomputed_internally\n",
    "case_id=blake3_explicit_complete_case_content_v1\n",
    "case_id_binds=profile+scope+lineage+items+relevance+declarations+independence+topology\n",
    "case_id_is_serializer_debug_and_rust_hash_independent\n",
    "input_order_does_not_change_case_identity\n",
    "bound_case=is_issued_private_non_deserializable_shadow_artifact\n",
    "raw_structural_case_is_not_eligible_for_future_disposition_without_binding\n",
    "bound_case_is_not_truth_belief_or_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-bound-shadow-case-contract:v1\0";
const CASE_ID_DOMAIN: &[u8] = b"symthaea:rca-bound-shadow-case:v1\0";

/// Issued provenance-bound case intended to be the only input class accepted by
/// a future RCA-003b disposition policy.
///
/// Fields are private and this type intentionally does not implement
/// `Deserialize`. Archive serialization is audit material only. Trusted binding
/// must be recomputed from revalidated candidates, context, and declarations.
#[must_use = "bound shadow cases are epistemic diagnostics and should be inspected"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BoundShadowEvidenceCaseV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    case_id: String,
    structural_case: ShadowEvidenceCaseV1,
    relation_declarations: Vec<BoundEvidenceRelationDeclarationV1>,
}

impl BoundShadowEvidenceCaseV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn profile_contract_digest(&self) -> &str {
        &self.profile_contract_digest
    }

    pub fn case_id(&self) -> &str {
        &self.case_id
    }

    pub fn structural_case(&self) -> &ShadowEvidenceCaseV1 {
        &self.structural_case
    }

    /// Declarations are stored in the same candidate-id order as
    /// `structural_case().items()`.
    pub fn relation_declarations(&self) -> &[BoundEvidenceRelationDeclarationV1] {
        &self.relation_declarations
    }
}

pub fn bound_shadow_evidence_case_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        BOUND_SHADOW_EVIDENCE_CASE_CONTRACT_V1.as_bytes(),
    )
}

/// Bind one structural RCA shadow case to declarer provenance and a canonical
/// complete-case identity.
///
/// The caller cannot supply a prebuilt `ShadowEvidenceCaseV1`: the structural
/// case is recomputed internally from the same candidate/context inputs whose
/// relation declarations are being bound.
pub fn bind_shadow_evidence_case_v1(
    proposition_id: &str,
    candidates: &[InstrumentedRuntimeEvidenceCandidateV1],
    relevance_context: &ValidatedCurrentRuntimeRelevanceContextV1,
    declarations: &[BoundEvidenceRelationDeclarationV1],
) -> Result<BoundShadowEvidenceCaseV1, BoundShadowEvidenceCaseError> {
    let candidate_ids: HashSet<&str> = candidates
        .iter()
        .map(|candidate| candidate.candidate_id())
        .collect();

    let mut declaration_by_candidate = HashMap::with_capacity(declarations.len());
    let mut declaration_ids = HashSet::with_capacity(declarations.len());
    for declaration in declarations {
        let candidate_id = declaration.relation().as_raw().evidence_id.as_str();
        if !candidate_ids.contains(candidate_id) {
            return Err(BoundShadowEvidenceCaseError::UnexpectedDeclarationCandidate {
                candidate_id: candidate_id.to_string(),
            });
        }
        if declaration_by_candidate
            .insert(candidate_id, declaration)
            .is_some()
        {
            return Err(BoundShadowEvidenceCaseError::DuplicateDeclarationCandidate {
                candidate_id: candidate_id.to_string(),
            });
        }
        if !declaration_ids.insert(declaration.declaration_id()) {
            return Err(BoundShadowEvidenceCaseError::DuplicateDeclarationId {
                declaration_id: declaration.declaration_id().to_string(),
            });
        }
    }

    for candidate in candidates {
        if !declaration_by_candidate.contains_key(candidate.candidate_id()) {
            return Err(BoundShadowEvidenceCaseError::MissingDeclaration {
                candidate_id: candidate.candidate_id().to_string(),
            });
        }
    }

    let relations = declarations
        .iter()
        .map(|declaration| declaration.relation().clone())
        .collect::<Vec<_>>();
    let structural_case = assemble_shadow_evidence_case_v1(
        proposition_id,
        candidates,
        relevance_context,
        &relations,
    )
    .map_err(BoundShadowEvidenceCaseError::StructuralCase)?;

    // Canonical declaration order follows the already-canonical candidate order
    // of the issued structural case, not caller input order.
    let mut ordered_declarations = Vec::with_capacity(structural_case.items().len());
    for item in structural_case.items() {
        let declaration = declaration_by_candidate
            .get(item.candidate_id())
            .ok_or_else(|| BoundShadowEvidenceCaseError::MissingDeclaration {
                candidate_id: item.candidate_id().to_string(),
            })?;
        ordered_declarations.push((*declaration).clone());
    }

    let profile_contract_digest = bound_shadow_evidence_case_profile_digest_v1();
    let case_id = complete_case_id_v1(
        &profile_contract_digest,
        &structural_case,
        &ordered_declarations,
    );

    Ok(BoundShadowEvidenceCaseV1 {
        schema_version: BOUND_SHADOW_EVIDENCE_CASE_SCHEMA_VERSION,
        profile: BOUND_SHADOW_EVIDENCE_CASE_PROFILE_V1.to_string(),
        profile_contract_digest,
        case_id,
        structural_case,
        relation_declarations: ordered_declarations,
    })
}

fn complete_case_id_v1(
    profile_contract_digest: &str,
    case: &ShadowEvidenceCaseV1,
    declarations: &[BoundEvidenceRelationDeclarationV1],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CASE_ID_DOMAIN);

    hash_text(
        &mut hasher,
        b"binding_profile_digest",
        profile_contract_digest,
    );
    hash_bytes(
        &mut hasher,
        b"binding_schema_version",
        &BOUND_SHADOW_EVIDENCE_CASE_SCHEMA_VERSION.to_le_bytes(),
    );
    hash_text(&mut hasher, b"structural_profile", case.profile());
    hash_text(
        &mut hasher,
        b"structural_profile_digest",
        case.profile_contract_digest(),
    );
    hash_text(&mut hasher, b"proposition_id", case.proposition_id());
    hash_text(
        &mut hasher,
        b"relevance_context_commitment",
        case.relevance_context_commitment(),
    );
    hash_text(&mut hasher, b"case_scope_digest", case.case_scope_digest());
    hash_text(&mut hasher, b"lineage_graph_id", case.lineage_graph_id());

    hash_count(&mut hasher, b"item_count", case.items().len());
    for (item, declaration) in case.items().iter().zip(declarations) {
        hash_text(&mut hasher, b"item_candidate_id", item.candidate_id());
        hash_text(
            &mut hasher,
            b"item_observation_root_id",
            item.observation_root_id(),
        );
        hash_text(&mut hasher, b"item_claim_digest", item.claim_digest());
        hash_text(&mut hasher, b"item_field", field_tag(item.field()));
        hash_text(&mut hasher, b"producer_relation_id", item.relation_id());
        hash_text(
            &mut hasher,
            b"relation_declaration_id",
            declaration.declaration_id(),
        );
        hash_text(
            &mut hasher,
            b"relation_declaration_profile_digest",
            declaration.identity_profile_digest(),
        );
        let provenance = declaration.provenance().as_raw();
        hash_text(&mut hasher, b"declarer_id", &provenance.declarer_id);
        hash_option_text(
            &mut hasher,
            b"declarer_version",
            provenance.declarer_version.as_deref(),
        );
        hash_text(
            &mut hasher,
            b"declaration_method",
            declaration_method_tag(provenance.method),
        );
        hash_text(
            &mut hasher,
            b"declaration_provenance_digest",
            &provenance.provenance_digest,
        );
        hash_text(
            &mut hasher,
            b"declared_relation",
            relation_kind_tag(item.declared_relation()),
        );
        hash_bytes(
            &mut hasher,
            b"declared_relation_strength_ppm",
            &item.declared_relation_strength_ppm().to_le_bytes(),
        );
        hash_bool(
            &mut hasher,
            b"current_runtime_relevant",
            item.current_runtime_relevant(),
        );
        hash_count(
            &mut hasher,
            b"relevance_defect_count",
            item.relevance_defects().len(),
        );
        for defect in item.relevance_defects() {
            hash_relevance_defect(&mut hasher, defect);
        }
    }

    hash_count(
        &mut hasher,
        b"independence_pair_count",
        case.independence_pairs().len(),
    );
    for pair in case.independence_pairs() {
        hash_text(
            &mut hasher,
            b"independence_left_candidate_id",
            pair.left_candidate_id(),
        );
        hash_text(
            &mut hasher,
            b"independence_right_candidate_id",
            pair.right_candidate_id(),
        );
        hash_text(
            &mut hasher,
            b"independence_assessment",
            independence_tag(pair.assessment()),
        );
    }

    hash_text(
        &mut hasher,
        b"declared_relation_topology",
        topology_tag(case.declared_relation_topology()),
    );
    hash_bool(
        &mut hasher,
        b"has_declared_current_runtime_defeater",
        case.has_declared_current_runtime_defeater(),
    );

    format!("blake3:{}", hasher.finalize().to_hex())
}

fn field_tag(field: ShadowObservationFieldV1) -> &'static str {
    match field {
        ShadowObservationFieldV1::CycleTimeUs => "cycle_time_us",
        ShadowObservationFieldV1::PredictionErrorPpm => "prediction_error_ppm",
        ShadowObservationFieldV1::PeakAttentionBits => "peak_attention_bits",
        ShadowObservationFieldV1::LearningOccurred => "learning_occurred",
        ShadowObservationFieldV1::DetectedPrimitiveCount => "detected_primitive_count",
        ShadowObservationFieldV1::OutputDigest => "output_digest",
        ShadowObservationFieldV1::ThoughtDigest => "thought_digest",
        ShadowObservationFieldV1::MetadataDigest => "metadata_digest",
        ShadowObservationFieldV1::LanguageOutput => "language_output",
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

fn independence_tag(assessment: EvidenceIndependenceV1) -> &'static str {
    match assessment {
        EvidenceIndependenceV1::SameEvidence => "same_evidence",
        EvidenceIndependenceV1::Derived => "derived",
        EvidenceIndependenceV1::SameRoot => "same_root",
        EvidenceIndependenceV1::PartiallyShared => "partially_shared",
        EvidenceIndependenceV1::Independent => "independent",
    }
}

fn topology_tag(topology: ShadowDeclaredRelationTopologyV1) -> &'static str {
    match topology {
        ShadowDeclaredRelationTopologyV1::NoCurrentRuntimeRelevantItems => {
            "no_current_runtime_relevant_items"
        }
        ShadowDeclaredRelationTopologyV1::OnlyNeutralRelations => "only_neutral_relations",
        ShadowDeclaredRelationTopologyV1::SupportSideOnly => "support_side_only",
        ShadowDeclaredRelationTopologyV1::OppositionSideOnly => "opposition_side_only",
        ShadowDeclaredRelationTopologyV1::MixedSupportAndOpposition => {
            "mixed_support_and_opposition"
        }
    }
}

fn hash_relevance_defect(hasher: &mut blake3::Hasher, defect: &RuntimeRelevanceDefectV1) {
    match defect {
        RuntimeRelevanceDefectV1::SourceGenerationMismatch { observed, current } => {
            hash_text(hasher, b"relevance_defect_kind", "source_generation_mismatch");
            hash_text(hasher, b"relevance_defect_observed", observed);
            hash_text(hasher, b"relevance_defect_current", current);
        }
        RuntimeRelevanceDefectV1::ExecutionLineageMismatch { observed, current } => {
            hash_text(hasher, b"relevance_defect_kind", "execution_lineage_mismatch");
            hash_text(hasher, b"relevance_defect_observed", observed);
            hash_text(hasher, b"relevance_defect_current", current);
        }
        RuntimeRelevanceDefectV1::AdapterProfileMismatch { observed, current } => {
            hash_text(hasher, b"relevance_defect_kind", "adapter_profile_mismatch");
            hash_text(hasher, b"relevance_defect_observed", observed);
            hash_text(hasher, b"relevance_defect_current", current);
        }
        RuntimeRelevanceDefectV1::AdapterContractMismatch { observed, current } => {
            hash_text(hasher, b"relevance_defect_kind", "adapter_contract_mismatch");
            hash_text(hasher, b"relevance_defect_observed", observed);
            hash_text(hasher, b"relevance_defect_current", current);
        }
        RuntimeRelevanceDefectV1::FutureObservation {
            observed_cycle,
            current_cycle,
        } => {
            hash_text(hasher, b"relevance_defect_kind", "future_observation");
            hash_bytes(
                hasher,
                b"relevance_defect_observed_cycle",
                &observed_cycle.to_le_bytes(),
            );
            hash_bytes(
                hasher,
                b"relevance_defect_current_cycle",
                &current_cycle.to_le_bytes(),
            );
        }
        RuntimeRelevanceDefectV1::StaleByCycleLag { lag, max_cycle_lag } => {
            hash_text(hasher, b"relevance_defect_kind", "stale_by_cycle_lag");
            hash_bytes(hasher, b"relevance_defect_lag", &lag.to_le_bytes());
            hash_bytes(
                hasher,
                b"relevance_defect_max_cycle_lag",
                &max_cycle_lag.to_le_bytes(),
            );
        }
    }
}

fn hash_count(hasher: &mut blake3::Hasher, label: &[u8], count: usize) {
    hash_bytes(hasher, label, &(count as u64).to_le_bytes());
}

fn hash_bool(hasher: &mut blake3::Hasher, label: &[u8], value: bool) {
    hash_bytes(hasher, label, &[u8::from(value)]);
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
pub enum BoundShadowEvidenceCaseError {
    DuplicateDeclarationCandidate { candidate_id: String },
    DuplicateDeclarationId { declaration_id: String },
    UnexpectedDeclarationCandidate { candidate_id: String },
    MissingDeclaration { candidate_id: String },
    StructuralCase(ShadowEvidenceCaseError),
}

impl std::fmt::Display for BoundShadowEvidenceCaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateDeclarationCandidate { candidate_id } => write!(
                f,
                "multiple bound relation declarations reference candidate {candidate_id}"
            ),
            Self::DuplicateDeclarationId { declaration_id } => {
                write!(f, "duplicate relation declaration id {declaration_id}")
            }
            Self::UnexpectedDeclarationCandidate { candidate_id } => write!(
                f,
                "relation declaration references candidate {candidate_id} outside the case"
            ),
            Self::MissingDeclaration { candidate_id } => {
                write!(f, "candidate {candidate_id} lacks a bound relation declaration")
            }
            Self::StructuralCase(error) => write!(f, "structural shadow case failed: {error}"),
        }
    }
}

impl std::error::Error for BoundShadowEvidenceCaseError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_governance::{
        currentness::{
            EvidenceRelationTargetV1, EvidenceRelationV1,
            COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
        },
        relation_provenance::{
            EvidenceRelationDeclarationProvenanceV1,
            RELATION_DECLARATION_PROVENANCE_SCHEMA_VERSION,
        },
    };
    use symthaea_rca_shadow::{
        FrozenCycleObservationV1, FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION,
    };
    use symthaea_rca_shadow_epistemics::{
        CurrentRuntimeRelevanceContextV1, RUNTIME_RELEVANCE_SCHEMA_VERSION,
    };

    const PROPOSITION: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const REL_A: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const REL_B: &str =
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SOURCE: &str =
        "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const LINEAGE: &str =
        "blake3:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const ADAPTER: &str =
        "blake3:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const OUTPUT: &str =
        "blake3:1111111111111111111111111111111111111111111111111111111111111111";
    const PROVENANCE_A: &str =
        "blake3:2222222222222222222222222222222222222222222222222222222222222222";
    const PROVENANCE_B: &str =
        "blake3:3333333333333333333333333333333333333333333333333333333333333333";

    fn observation(cycle: u64) -> symthaea_rca_shadow::ValidatedFrozenCycleObservationV1 {
        FrozenCycleObservationV1 {
            schema_version: FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION,
            source_generation_digest: SOURCE.into(),
            execution_lineage_digest: LINEAGE.into(),
            adapter_profile: "adapter-v1".into(),
            adapter_contract_digest: ADAPTER.into(),
            cycle_index: cycle,
            cycle_time_us: 10_000,
            prediction_error_ppm: 200_000,
            peak_attention_bits: 1.5_f32.to_bits(),
            learning_occurred: false,
            detected_primitive_count: 2,
            output_digest: OUTPUT.into(),
            thought_digest: LINEAGE.into(),
            metadata_digest: SOURCE.into(),
            language_output_digest: None,
            language_source: None,
        }
        .validate()
        .unwrap()
    }

    fn candidate(
        cycle: u64,
        field: ShadowObservationFieldV1,
    ) -> InstrumentedRuntimeEvidenceCandidateV1 {
        InstrumentedRuntimeEvidenceCandidateV1::new(observation(cycle), field)
    }

    fn context(current_cycle: u64, max_lag: u64) -> ValidatedCurrentRuntimeRelevanceContextV1 {
        CurrentRuntimeRelevanceContextV1 {
            schema_version: RUNTIME_RELEVANCE_SCHEMA_VERSION,
            source_generation_digest: SOURCE.into(),
            execution_lineage_digest: LINEAGE.into(),
            adapter_profile: "adapter-v1".into(),
            adapter_contract_digest: ADAPTER.into(),
            current_cycle_index: current_cycle,
            max_cycle_lag: max_lag,
        }
        .validate()
        .unwrap()
    }

    fn declaration(
        relation_id: &str,
        candidate: &InstrumentedRuntimeEvidenceCandidateV1,
        kind: EvidenceRelationKindV1,
        strength_ppm: u32,
        declarer_id: &str,
        provenance_digest: &str,
    ) -> BoundEvidenceRelationDeclarationV1 {
        let relation = EvidenceRelationV1 {
            schema_version: COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
            relation_id: relation_id.into(),
            evidence_id: candidate.candidate_id().into(),
            relation: kind,
            target: EvidenceRelationTargetV1::Proposition {
                proposition_id: PROPOSITION.into(),
            },
            strength_ppm,
        }
        .validate()
        .unwrap();
        let provenance = EvidenceRelationDeclarationProvenanceV1 {
            schema_version: RELATION_DECLARATION_PROVENANCE_SCHEMA_VERSION,
            declarer_id: declarer_id.into(),
            declarer_version: Some("v1".into()),
            method: RelationDeclarationMethodV1::DeterministicRule,
            provenance_digest: provenance_digest.into(),
        }
        .validate()
        .unwrap();
        BoundEvidenceRelationDeclarationV1::new(provenance, relation)
    }

    #[test]
    fn exact_inputs_produce_deterministic_content_addressed_case() {
        let item = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let declaration = declaration(
            REL_A,
            &item,
            EvidenceRelationKindV1::Supports,
            700_000,
            "rule-a",
            PROVENANCE_A,
        );
        let a = bind_shadow_evidence_case_v1(
            PROPOSITION,
            std::slice::from_ref(&item),
            &context(10, 0),
            std::slice::from_ref(&declaration),
        )
        .unwrap();
        let b = bind_shadow_evidence_case_v1(
            PROPOSITION,
            &[item],
            &context(10, 0),
            &[declaration],
        )
        .unwrap();
        assert_eq!(a, b);
        assert!(a.case_id().starts_with("blake3:"));
    }

    #[test]
    fn same_structural_case_different_declarer_provenance_changes_bound_case_id() {
        let item = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let a_decl = declaration(
            REL_A,
            &item,
            EvidenceRelationKindV1::Supports,
            700_000,
            "rule-a",
            PROVENANCE_A,
        );
        let b_decl = declaration(
            REL_A,
            &item,
            EvidenceRelationKindV1::Supports,
            700_000,
            "rule-b",
            PROVENANCE_B,
        );
        let a = bind_shadow_evidence_case_v1(
            PROPOSITION,
            std::slice::from_ref(&item),
            &context(10, 0),
            &[a_decl],
        )
        .unwrap();
        let b = bind_shadow_evidence_case_v1(
            PROPOSITION,
            &[item],
            &context(10, 0),
            &[b_decl],
        )
        .unwrap();
        assert_eq!(a.structural_case(), b.structural_case());
        assert_ne!(a.case_id(), b.case_id());
    }

    #[test]
    fn caller_declaration_order_does_not_change_case_identity() {
        let a = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let b = candidate(11, ShadowObservationFieldV1::LearningOccurred);
        let a_decl = declaration(
            REL_A,
            &a,
            EvidenceRelationKindV1::Supports,
            600_000,
            "rule-a",
            PROVENANCE_A,
        );
        let b_decl = declaration(
            REL_B,
            &b,
            EvidenceRelationKindV1::Contradicts,
            500_000,
            "rule-b",
            PROVENANCE_B,
        );
        let first = bind_shadow_evidence_case_v1(
            PROPOSITION,
            &[a.clone(), b.clone()],
            &context(11, 1),
            &[a_decl.clone(), b_decl.clone()],
        )
        .unwrap();
        let second = bind_shadow_evidence_case_v1(
            PROPOSITION,
            &[b, a],
            &context(11, 1),
            &[b_decl, a_decl],
        )
        .unwrap();
        assert_eq!(first.case_id(), second.case_id());
        assert_eq!(first, second);
    }

    #[test]
    fn changed_relation_body_changes_bound_case_identity() {
        let item = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let a = declaration(
            REL_A,
            &item,
            EvidenceRelationKindV1::Supports,
            500_000,
            "rule-a",
            PROVENANCE_A,
        );
        let b = declaration(
            REL_A,
            &item,
            EvidenceRelationKindV1::Supports,
            900_000,
            "rule-a",
            PROVENANCE_A,
        );
        let case_a = bind_shadow_evidence_case_v1(
            PROPOSITION,
            std::slice::from_ref(&item),
            &context(10, 0),
            &[a],
        )
        .unwrap();
        let case_b = bind_shadow_evidence_case_v1(
            PROPOSITION,
            &[item],
            &context(10, 0),
            &[b],
        )
        .unwrap();
        assert_ne!(case_a.case_id(), case_b.case_id());
    }

    #[test]
    fn every_candidate_requires_exactly_one_bound_declaration() {
        let a = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let b = candidate(11, ShadowObservationFieldV1::LearningOccurred);
        let a_decl = declaration(
            REL_A,
            &a,
            EvidenceRelationKindV1::Supports,
            500_000,
            "rule-a",
            PROVENANCE_A,
        );
        let missing = bind_shadow_evidence_case_v1(
            PROPOSITION,
            &[a.clone(), b],
            &context(11, 1),
            std::slice::from_ref(&a_decl),
        )
        .unwrap_err();
        assert!(matches!(
            missing,
            BoundShadowEvidenceCaseError::MissingDeclaration { .. }
        ));

        let duplicate = bind_shadow_evidence_case_v1(
            PROPOSITION,
            &[a],
            &context(10, 0),
            &[a_decl.clone(), a_decl],
        )
        .unwrap_err();
        assert!(matches!(
            duplicate,
            BoundShadowEvidenceCaseError::DuplicateDeclarationCandidate { .. }
        ));
    }

    #[test]
    fn declaration_for_candidate_outside_case_fails_closed() {
        let inside = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let outside = candidate(11, ShadowObservationFieldV1::LearningOccurred);
        let outside_decl = declaration(
            REL_B,
            &outside,
            EvidenceRelationKindV1::Supports,
            500_000,
            "rule-b",
            PROVENANCE_B,
        );
        let error = bind_shadow_evidence_case_v1(
            PROPOSITION,
            &[inside],
            &context(10, 0),
            &[outside_decl],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BoundShadowEvidenceCaseError::UnexpectedDeclarationCandidate { .. }
        ));
    }

    #[test]
    fn relevance_context_is_case_identity_bearing() {
        let item = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let declaration = declaration(
            REL_A,
            &item,
            EvidenceRelationKindV1::Supports,
            500_000,
            "rule-a",
            PROVENANCE_A,
        );
        let now = bind_shadow_evidence_case_v1(
            PROPOSITION,
            std::slice::from_ref(&item),
            &context(10, 0),
            std::slice::from_ref(&declaration),
        )
        .unwrap();
        let later = bind_shadow_evidence_case_v1(
            PROPOSITION,
            &[item],
            &context(11, 1),
            &[declaration],
        )
        .unwrap();
        assert_ne!(now.case_id(), later.case_id());
    }

    #[test]
    fn issued_bound_case_serializes_for_audit() {
        let item = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let declaration = declaration(
            REL_A,
            &item,
            EvidenceRelationKindV1::Supports,
            500_000,
            "rule-a",
            PROVENANCE_A,
        );
        let case = bind_shadow_evidence_case_v1(
            PROPOSITION,
            &[item],
            &context(10, 0),
            &[declaration],
        )
        .unwrap();
        let encoded = serde_json::to_string(&case).unwrap();
        assert!(encoded.contains(case.case_id()));
        assert!(encoded.contains(case.profile_contract_digest()));
        assert_eq!(case.relation_declarations().len(), 1);
    }

    #[test]
    fn profile_has_strict_identity() {
        let digest = bound_shadow_evidence_case_profile_digest_v1();
        assert!(digest.starts_with("blake3:"));
        assert_eq!(digest.len(), "blake3:".len() + 64);
        assert_eq!(digest, bound_shadow_evidence_case_profile_digest_v1());
    }
}
