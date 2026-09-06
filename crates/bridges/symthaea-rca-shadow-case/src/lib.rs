// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! RCA-003a: causally inert shadow evidence-case assembly.
//!
//! This crate joins already-detached runtime evidence candidates with one
//! validated current-runtime relevance context and typed **declared** proposition
//! relations. It recomputes relevance internally, reconstructs candidate lineage
//! from the candidates themselves, derives pairwise independence from that
//! closed lineage graph, and preserves relation topology without converting it
//! into a truth judgment.
//!
//! It deliberately does **not** aggregate evidence into truth, confidence,
//! posterior probability, canonical evidence admission, belief/workspace state,
//! action authority, or self-improvement promotion.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::{HashMap, HashSet};
use symthaea_epistemic_governance::{
    currentness::{
        EvidenceRelationKindV1, EvidenceRelationTargetV1, ValidatedEvidenceRelationV1,
    },
    lineage::{
        CognitiveLineageError, EvidenceIndependenceV1, EvidenceLineageGraphV1,
        ValidatedEvidenceLineageGraphV1, COGNITIVE_LINEAGE_SCHEMA_VERSION,
    },
};
use symthaea_rca_evidence_bridge::{
    InstrumentedRuntimeEvidenceCandidateV1, ShadowObservationFieldV1,
};
use symthaea_rca_shadow_epistemics::{
    assess_current_runtime_relevance, RuntimeRelevanceDefectV1,
    ValidatedCurrentRuntimeRelevanceContextV1,
};

pub const SHADOW_EVIDENCE_CASE_SCHEMA_VERSION: u16 = 1;
pub const SHADOW_EVIDENCE_CASE_PROFILE_V1: &str = "rca-shadow-evidence-case-v1";

/// Normative RCA-003a case-assembly semantics.
pub const SHADOW_EVIDENCE_CASE_CONTRACT_V1: &str = concat!(
    "rca-shadow-evidence-case-v1\n",
    "inputs=instrumented_runtime_candidates+one_validated_runtime_relevance_context+validated_declared_proposition_relations\n",
    "runtime_relevance=recomputed_internally_for_every_candidate\n",
    "one_case_one_relevance_context_commitment\n",
    "case_scope=exact_proposition_digest+exact_relevance_context_commitment\n",
    "candidate_relation_join=exact_candidate_id_one_to_one\n",
    "relation_target=one_exact_case_proposition_digest\n",
    "relation_semantics=caller_declared_not_truth_verified\n",
    "lineage=reconstructed_only_from_candidate_lineage_fragments\n",
    "same_observation_candidates_share_root_and_are_not_independent\n",
    "independence=derived_pairwise_from_closed_lineage_graph\n",
    "corroborates_label_does_not_imply_independence\n",
    "relation_strength_is_preserved_not_summed_or_converted_to_probability\n",
    "declared_relation_topology_is_diagnostic_not_truth_disposition\n",
    "case_result=is_issued_private_non_deserializable_shadow_report\n",
    "no_supported_tentative_belief_posterior_or_confidence_output\n",
    "no_canonical_evidence_workspace_action_or_self_improvement_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-shadow-evidence-case-contract:v1\0";
const CASE_SCOPE_DOMAIN: &[u8] = b"symthaea:rca-shadow-case-scope:v1\0";
const LINEAGE_GRAPH_DOMAIN: &[u8] = b"symthaea:rca-shadow-case-lineage-graph:v1\0";

/// Diagnostic shape of caller-declared proposition relations among candidates
/// that are relevant to the one exact runtime context used for this case.
///
/// This is not a truth, belief, probability, or evidence-admission disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowDeclaredRelationTopologyV1 {
    NoCurrentRuntimeRelevantItems,
    OnlyNeutralRelations,
    SupportSideOnly,
    OppositionSideOnly,
    MixedSupportAndOpposition,
}

/// One joined candidate/relevance/declared-relation item inside an issued case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ShadowEvidenceCaseItemV1 {
    candidate_id: String,
    observation_root_id: String,
    claim_digest: String,
    field: ShadowObservationFieldV1,
    relation_id: String,
    declared_relation: EvidenceRelationKindV1,
    declared_relation_strength_ppm: u32,
    current_runtime_relevant: bool,
    relevance_defects: Vec<RuntimeRelevanceDefectV1>,
}

impl ShadowEvidenceCaseItemV1 {
    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn observation_root_id(&self) -> &str {
        &self.observation_root_id
    }

    pub fn claim_digest(&self) -> &str {
        &self.claim_digest
    }

    pub const fn field(&self) -> ShadowObservationFieldV1 {
        self.field
    }

    pub fn relation_id(&self) -> &str {
        &self.relation_id
    }

    pub const fn declared_relation(&self) -> EvidenceRelationKindV1 {
        self.declared_relation
    }

    pub const fn declared_relation_strength_ppm(&self) -> u32 {
        self.declared_relation_strength_ppm
    }

    pub const fn current_runtime_relevant(&self) -> bool {
        self.current_runtime_relevant
    }

    pub fn relevance_defects(&self) -> &[RuntimeRelevanceDefectV1] {
        &self.relevance_defects
    }
}

/// Pairwise lineage result preserved separately from declared relation labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ShadowEvidenceIndependencePairV1 {
    left_candidate_id: String,
    right_candidate_id: String,
    assessment: EvidenceIndependenceV1,
}

impl ShadowEvidenceIndependencePairV1 {
    pub fn left_candidate_id(&self) -> &str {
        &self.left_candidate_id
    }

    pub fn right_candidate_id(&self) -> &str {
        &self.right_candidate_id
    }

    pub const fn assessment(&self) -> EvidenceIndependenceV1 {
        self.assessment
    }
}

/// Issued, audit-serializable shadow case.
///
/// Fields are private and the type intentionally has no `Deserialize`
/// implementation. Archived reports cannot be rehydrated as trusted case
/// assembly; callers must reload/revalidate candidates/relations/context and
/// call [`assemble_shadow_evidence_case_v1`] again.
#[must_use = "shadow evidence cases preserve epistemic diagnostics and should be inspected"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ShadowEvidenceCaseV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    proposition_id: String,
    relevance_context_commitment: String,
    case_scope_digest: String,
    lineage_graph_id: String,
    items: Vec<ShadowEvidenceCaseItemV1>,
    independence_pairs: Vec<ShadowEvidenceIndependencePairV1>,
    declared_relation_topology: ShadowDeclaredRelationTopologyV1,
    has_declared_current_runtime_defeater: bool,
}

impl ShadowEvidenceCaseV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn profile_contract_digest(&self) -> &str {
        &self.profile_contract_digest
    }

    pub fn proposition_id(&self) -> &str {
        &self.proposition_id
    }

    pub fn relevance_context_commitment(&self) -> &str {
        &self.relevance_context_commitment
    }

    pub fn case_scope_digest(&self) -> &str {
        &self.case_scope_digest
    }

    pub fn lineage_graph_id(&self) -> &str {
        &self.lineage_graph_id
    }

    pub fn items(&self) -> &[ShadowEvidenceCaseItemV1] {
        &self.items
    }

    pub fn independence_pairs(&self) -> &[ShadowEvidenceIndependencePairV1] {
        &self.independence_pairs
    }

    pub const fn declared_relation_topology(&self) -> ShadowDeclaredRelationTopologyV1 {
        self.declared_relation_topology
    }

    pub const fn has_declared_current_runtime_defeater(&self) -> bool {
        self.has_declared_current_runtime_defeater
    }
}

pub fn shadow_evidence_case_profile_digest_v1() -> String {
    domain_hash(PROFILE_DOMAIN, SHADOW_EVIDENCE_CASE_CONTRACT_V1.as_bytes())
}

/// Assemble one causally inert RCA shadow evidence case.
///
/// Relevance is recomputed for every candidate using the **same validated
/// context**, preventing mixed-currentness reports from being assembled into one
/// topology. The proposition digest and context commitment are jointly bound as
/// `case_scope_digest`.
///
/// This function performs exact joins and derives lineage/independence. It does
/// not aggregate relation strengths, compute a posterior, decide truth, admit
/// canonical evidence, modify workspace state, or return an authority token.
pub fn assemble_shadow_evidence_case_v1(
    proposition_id: &str,
    candidates: &[InstrumentedRuntimeEvidenceCandidateV1],
    relevance_context: &ValidatedCurrentRuntimeRelevanceContextV1,
    relations: &[ValidatedEvidenceRelationV1],
) -> Result<ShadowEvidenceCaseV1, ShadowEvidenceCaseError> {
    validate_digest(proposition_id)?;
    if candidates.is_empty() {
        return Err(ShadowEvidenceCaseError::EmptyCase);
    }

    let mut candidate_map = HashMap::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate_map
            .insert(candidate.candidate_id(), candidate)
            .is_some()
        {
            return Err(ShadowEvidenceCaseError::DuplicateCandidateId {
                candidate_id: candidate.candidate_id().to_string(),
            });
        }
    }

    let mut relation_map = HashMap::with_capacity(relations.len());
    for relation in relations {
        let raw = relation.as_raw();
        if !candidate_map.contains_key(raw.evidence_id.as_str()) {
            return Err(ShadowEvidenceCaseError::UnexpectedRelationCandidate {
                candidate_id: raw.evidence_id.clone(),
            });
        }
        if relation_map
            .insert(raw.evidence_id.as_str(), relation)
            .is_some()
        {
            return Err(ShadowEvidenceCaseError::DuplicateRelationCandidate {
                candidate_id: raw.evidence_id.clone(),
            });
        }
    }

    for candidate in candidates {
        if !relation_map.contains_key(candidate.candidate_id()) {
            return Err(ShadowEvidenceCaseError::MissingRelation {
                candidate_id: candidate.candidate_id().to_string(),
            });
        }
    }

    let lineage = reconstruct_candidate_lineage(candidates)?;
    let relevance_context_commitment = relevance_context.commitment();
    let case_scope_digest = case_scope_digest(proposition_id, &relevance_context_commitment);

    let mut ordered: Vec<&InstrumentedRuntimeEvidenceCandidateV1> = candidates.iter().collect();
    ordered.sort_by(|a, b| a.candidate_id().cmp(b.candidate_id()));

    let mut items = Vec::with_capacity(ordered.len());
    for candidate in &ordered {
        let relation = relation_map[candidate.candidate_id()];
        let raw_relation = relation.as_raw();

        let relation_target = match &raw_relation.target {
            EvidenceRelationTargetV1::Proposition { proposition_id } => proposition_id,
            EvidenceRelationTargetV1::Evidence { .. } => {
                return Err(ShadowEvidenceCaseError::RelationMustTargetProposition {
                    candidate_id: candidate.candidate_id().to_string(),
                });
            }
        };
        if relation_target != proposition_id {
            return Err(ShadowEvidenceCaseError::RelationTargetMismatch {
                candidate_id: candidate.candidate_id().to_string(),
                expected_proposition_id: proposition_id.to_string(),
                found_proposition_id: relation_target.clone(),
            });
        }

        // The closed lineage is reconstructed from each candidate's own
        // observation-root binding. This is a consistency assertion, not a
        // caller-supplied lineage claim.
        let roots = lineage
            .root_ids(candidate.candidate_id())
            .map_err(ShadowEvidenceCaseError::Lineage)?;
        if roots.len() != 1 || !roots.contains(candidate.observation_root_id()) {
            return Err(ShadowEvidenceCaseError::CandidateLineageMismatch {
                candidate_id: candidate.candidate_id().to_string(),
                expected_observation_root_id: candidate.observation_root_id().to_string(),
            });
        }

        // Relevance is issued here from the one shared validated context. The
        // caller cannot splice per-item reports generated under different
        // current cycles, lag policies, source generations, or executions.
        let relevance = assess_current_runtime_relevance(candidate, relevance_context);

        items.push(ShadowEvidenceCaseItemV1 {
            candidate_id: candidate.candidate_id().to_string(),
            observation_root_id: candidate.observation_root_id().to_string(),
            claim_digest: candidate.claim_digest().to_string(),
            field: candidate.field(),
            relation_id: raw_relation.relation_id.clone(),
            declared_relation: raw_relation.relation,
            declared_relation_strength_ppm: raw_relation.strength_ppm,
            current_runtime_relevant: relevance.is_relevant(),
            relevance_defects: relevance.defects().to_vec(),
        });
    }

    let mut independence_pairs = Vec::new();
    for left_index in 0..ordered.len() {
        for right_index in (left_index + 1)..ordered.len() {
            let left = ordered[left_index];
            let right = ordered[right_index];
            let assessment = lineage
                .assess_independence(left.candidate_id(), right.candidate_id())
                .map_err(ShadowEvidenceCaseError::Lineage)?;
            independence_pairs.push(ShadowEvidenceIndependencePairV1 {
                left_candidate_id: left.candidate_id().to_string(),
                right_candidate_id: right.candidate_id().to_string(),
                assessment,
            });
        }
    }

    let declared_relation_topology = declared_relation_topology(&items);
    let has_declared_current_runtime_defeater = items.iter().any(|item| {
        item.current_runtime_relevant && item.declared_relation == EvidenceRelationKindV1::Defeats
    });

    Ok(ShadowEvidenceCaseV1 {
        schema_version: SHADOW_EVIDENCE_CASE_SCHEMA_VERSION,
        profile: SHADOW_EVIDENCE_CASE_PROFILE_V1.to_string(),
        profile_contract_digest: shadow_evidence_case_profile_digest_v1(),
        proposition_id: proposition_id.to_string(),
        relevance_context_commitment,
        case_scope_digest,
        lineage_graph_id: lineage_graph_id(candidates),
        items,
        independence_pairs,
        declared_relation_topology,
        has_declared_current_runtime_defeater,
    })
}

fn reconstruct_candidate_lineage(
    candidates: &[InstrumentedRuntimeEvidenceCandidateV1],
) -> Result<ValidatedEvidenceLineageGraphV1, ShadowEvidenceCaseError> {
    let mut ordered: Vec<&InstrumentedRuntimeEvidenceCandidateV1> = candidates.iter().collect();
    ordered.sort_by(|a, b| a.candidate_id().cmp(b.candidate_id()));

    let mut seen_roots = HashSet::new();
    let mut nodes = Vec::with_capacity(candidates.len() * 2);
    for candidate in ordered {
        let fragment = candidate
            .lineage_fragment()
            .map_err(ShadowEvidenceCaseError::Lineage)?;
        if seen_roots.insert(candidate.observation_root_id().to_string()) {
            nodes.push(fragment.observation_root().clone());
        }
        nodes.push(fragment.candidate_node().clone());
    }

    EvidenceLineageGraphV1 {
        schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
        graph_id: lineage_graph_id(candidates),
        nodes,
    }
    .validate()
    .map_err(ShadowEvidenceCaseError::Lineage)
}

fn case_scope_digest(proposition_id: &str, relevance_context_commitment: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CASE_SCOPE_DOMAIN);
    hash_field(
        &mut hasher,
        b"case_profile_digest",
        shadow_evidence_case_profile_digest_v1().as_bytes(),
    );
    hash_field(&mut hasher, b"proposition_id", proposition_id.as_bytes());
    hash_field(
        &mut hasher,
        b"relevance_context_commitment",
        relevance_context_commitment.as_bytes(),
    );
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn lineage_graph_id(candidates: &[InstrumentedRuntimeEvidenceCandidateV1]) -> String {
    let mut ids: Vec<&str> = candidates
        .iter()
        .map(|candidate| candidate.candidate_id())
        .collect();
    ids.sort_unstable();
    let mut hasher = blake3::Hasher::new();
    hasher.update(LINEAGE_GRAPH_DOMAIN);
    hash_field(
        &mut hasher,
        b"case_profile_digest",
        shadow_evidence_case_profile_digest_v1().as_bytes(),
    );
    for candidate_id in ids {
        hash_field(&mut hasher, b"candidate_id", candidate_id.as_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn declared_relation_topology(items: &[ShadowEvidenceCaseItemV1]) -> ShadowDeclaredRelationTopologyV1 {
    let mut any_relevant = false;
    let mut support_side = false;
    let mut opposition_side = false;

    for item in items.iter().filter(|item| item.current_runtime_relevant) {
        any_relevant = true;
        match item.declared_relation {
            EvidenceRelationKindV1::Supports | EvidenceRelationKindV1::Corroborates => {
                support_side = true;
            }
            EvidenceRelationKindV1::Contradicts
            | EvidenceRelationKindV1::Weakens
            | EvidenceRelationKindV1::Defeats => {
                opposition_side = true;
            }
            EvidenceRelationKindV1::Irrelevant | EvidenceRelationKindV1::Supersedes => {}
        }
    }

    if !any_relevant {
        ShadowDeclaredRelationTopologyV1::NoCurrentRuntimeRelevantItems
    } else {
        match (support_side, opposition_side) {
            (false, false) => ShadowDeclaredRelationTopologyV1::OnlyNeutralRelations,
            (true, false) => ShadowDeclaredRelationTopologyV1::SupportSideOnly,
            (false, true) => ShadowDeclaredRelationTopologyV1::OppositionSideOnly,
            (true, true) => ShadowDeclaredRelationTopologyV1::MixedSupportAndOpposition,
        }
    }
}

fn validate_digest(digest: &str) -> Result<(), ShadowEvidenceCaseError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(ShadowEvidenceCaseError::MalformedPropositionDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ShadowEvidenceCaseError::MalformedPropositionDigest);
    }
    Ok(())
}

fn hash_field(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
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
pub enum ShadowEvidenceCaseError {
    MalformedPropositionDigest,
    EmptyCase,
    DuplicateCandidateId {
        candidate_id: String,
    },
    DuplicateRelationCandidate {
        candidate_id: String,
    },
    UnexpectedRelationCandidate {
        candidate_id: String,
    },
    MissingRelation {
        candidate_id: String,
    },
    RelationMustTargetProposition {
        candidate_id: String,
    },
    RelationTargetMismatch {
        candidate_id: String,
        expected_proposition_id: String,
        found_proposition_id: String,
    },
    CandidateLineageMismatch {
        candidate_id: String,
        expected_observation_root_id: String,
    },
    Lineage(CognitiveLineageError),
}

impl std::fmt::Display for ShadowEvidenceCaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedPropositionDigest => {
                f.write_str("proposition id must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::EmptyCase => f.write_str("shadow evidence case requires at least one candidate"),
            Self::DuplicateCandidateId { candidate_id } => {
                write!(f, "duplicate shadow candidate id {candidate_id}")
            }
            Self::DuplicateRelationCandidate { candidate_id } => {
                write!(f, "duplicate declared proposition relation for candidate {candidate_id}")
            }
            Self::UnexpectedRelationCandidate { candidate_id } => {
                write!(f, "declared proposition relation references unknown candidate {candidate_id}")
            }
            Self::MissingRelation { candidate_id } => {
                write!(f, "missing declared proposition relation for candidate {candidate_id}")
            }
            Self::RelationMustTargetProposition { candidate_id } => {
                write!(f, "candidate {candidate_id} relation must target the case proposition")
            }
            Self::RelationTargetMismatch {
                candidate_id,
                expected_proposition_id,
                found_proposition_id,
            } => write!(
                f,
                "candidate {candidate_id} relation targets {found_proposition_id}, expected {expected_proposition_id}"
            ),
            Self::CandidateLineageMismatch {
                candidate_id,
                expected_observation_root_id,
            } => write!(
                f,
                "candidate {candidate_id} lineage does not resolve exclusively to observation root {expected_observation_root_id}"
            ),
            Self::Lineage(error) => write!(f, "lineage validation failed: {error}"),
        }
    }
}

impl std::error::Error for ShadowEvidenceCaseError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_governance::currentness::{
        EvidenceRelationV1, COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
    };
    use symthaea_rca_shadow::{
        FrozenCycleObservationV1, FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION,
    };
    use symthaea_rca_shadow_epistemics::{
        CurrentRuntimeRelevanceContextV1, RUNTIME_RELEVANCE_SCHEMA_VERSION,
    };

    const PROPOSITION_A: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const PROPOSITION_B: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SOURCE: &str =
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const LINEAGE: &str =
        "blake3:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const ADAPTER: &str =
        "blake3:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const OUTPUT: &str =
        "blake3:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn observation(cycle_index: u64) -> symthaea_rca_shadow::ValidatedFrozenCycleObservationV1 {
        FrozenCycleObservationV1 {
            schema_version: FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION,
            source_generation_digest: SOURCE.into(),
            execution_lineage_digest: LINEAGE.into(),
            adapter_profile: "adapter-v1".into(),
            adapter_contract_digest: ADAPTER.into(),
            cycle_index,
            cycle_time_us: 11_000,
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
        cycle_index: u64,
        field: ShadowObservationFieldV1,
    ) -> InstrumentedRuntimeEvidenceCandidateV1 {
        InstrumentedRuntimeEvidenceCandidateV1::new(observation(cycle_index), field)
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

    fn relation(
        relation_id: &str,
        candidate: &InstrumentedRuntimeEvidenceCandidateV1,
        proposition_id: &str,
        kind: EvidenceRelationKindV1,
        strength_ppm: u32,
    ) -> ValidatedEvidenceRelationV1 {
        EvidenceRelationV1 {
            schema_version: COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
            relation_id: relation_id.into(),
            evidence_id: candidate.candidate_id().into(),
            relation: kind,
            target: EvidenceRelationTargetV1::Proposition {
                proposition_id: proposition_id.into(),
            },
            strength_ppm,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn same_observation_candidates_preserve_same_root_not_independence() {
        let pe = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let learning = candidate(10, ShadowObservationFieldV1::LearningOccurred);
        let relations = vec![
            relation(
                PROPOSITION_B,
                &pe,
                PROPOSITION_A,
                EvidenceRelationKindV1::Supports,
                800_000,
            ),
            relation(
                SOURCE,
                &learning,
                PROPOSITION_A,
                EvidenceRelationKindV1::Corroborates,
                700_000,
            ),
        ];

        let case = assemble_shadow_evidence_case_v1(
            PROPOSITION_A,
            &[pe, learning],
            &context(10, 0),
            &relations,
        )
        .unwrap();

        assert_eq!(case.items().len(), 2);
        assert_eq!(case.independence_pairs().len(), 1);
        assert_eq!(
            case.independence_pairs()[0].assessment(),
            EvidenceIndependenceV1::SameRoot
        );
        assert_eq!(
            case.declared_relation_topology(),
            ShadowDeclaredRelationTopologyV1::SupportSideOnly
        );
    }

    #[test]
    fn distinct_observation_events_can_be_independent() {
        let a = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let b = candidate(11, ShadowObservationFieldV1::PredictionErrorPpm);
        let relations = vec![
            relation(
                PROPOSITION_B,
                &a,
                PROPOSITION_A,
                EvidenceRelationKindV1::Supports,
                500_000,
            ),
            relation(
                SOURCE,
                &b,
                PROPOSITION_A,
                EvidenceRelationKindV1::Supports,
                500_000,
            ),
        ];
        let case = assemble_shadow_evidence_case_v1(
            PROPOSITION_A,
            &[a, b],
            &context(11, 1),
            &relations,
        )
        .unwrap();
        assert_eq!(
            case.independence_pairs()[0].assessment(),
            EvidenceIndependenceV1::Independent
        );
    }

    #[test]
    fn stale_candidate_is_preserved_but_not_counted_in_current_topology() {
        let item = candidate(5, ShadowObservationFieldV1::PredictionErrorPpm);
        let relations = vec![relation(
            PROPOSITION_B,
            &item,
            PROPOSITION_A,
            EvidenceRelationKindV1::Supports,
            1_000_000,
        )];
        let case = assemble_shadow_evidence_case_v1(
            PROPOSITION_A,
            &[item],
            &context(10, 1),
            &relations,
        )
        .unwrap();
        assert_eq!(
            case.declared_relation_topology(),
            ShadowDeclaredRelationTopologyV1::NoCurrentRuntimeRelevantItems
        );
        assert!(!case.items()[0].current_runtime_relevant());
        assert!(!case.items()[0].relevance_defects().is_empty());
    }

    #[test]
    fn support_and_contradiction_remain_declared_mixed_topology_not_truth_result() {
        let a = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let b = candidate(11, ShadowObservationFieldV1::LearningOccurred);
        let relations = vec![
            relation(
                PROPOSITION_B,
                &a,
                PROPOSITION_A,
                EvidenceRelationKindV1::Supports,
                900_000,
            ),
            relation(
                SOURCE,
                &b,
                PROPOSITION_A,
                EvidenceRelationKindV1::Contradicts,
                900_000,
            ),
        ];
        let case = assemble_shadow_evidence_case_v1(
            PROPOSITION_A,
            &[a, b],
            &context(11, 1),
            &relations,
        )
        .unwrap();
        assert_eq!(
            case.declared_relation_topology(),
            ShadowDeclaredRelationTopologyV1::MixedSupportAndOpposition
        );
    }

    #[test]
    fn declared_defeater_is_preserved_without_becoming_belief_or_admission() {
        let item = candidate(10, ShadowObservationFieldV1::LearningOccurred);
        let relations = vec![relation(
            PROPOSITION_B,
            &item,
            PROPOSITION_A,
            EvidenceRelationKindV1::Defeats,
            950_000,
        )];
        let case = assemble_shadow_evidence_case_v1(
            PROPOSITION_A,
            &[item],
            &context(10, 0),
            &relations,
        )
        .unwrap();
        assert!(case.has_declared_current_runtime_defeater());
        assert_eq!(
            case.declared_relation_topology(),
            ShadowDeclaredRelationTopologyV1::OppositionSideOnly
        );
    }

    #[test]
    fn relation_target_must_match_exact_case_proposition() {
        let item = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let relations = vec![relation(
            PROPOSITION_B,
            &item,
            PROPOSITION_B,
            EvidenceRelationKindV1::Supports,
            500_000,
        )];
        let error = assemble_shadow_evidence_case_v1(
            PROPOSITION_A,
            &[item],
            &context(10, 0),
            &relations,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ShadowEvidenceCaseError::RelationTargetMismatch { .. }
        ));
    }

    #[test]
    fn relation_must_join_exact_candidate_once() {
        let a = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let b = candidate(11, ShadowObservationFieldV1::PredictionErrorPpm);
        let relations = vec![relation(
            PROPOSITION_B,
            &a,
            PROPOSITION_A,
            EvidenceRelationKindV1::Supports,
            500_000,
        )];
        let error = assemble_shadow_evidence_case_v1(
            PROPOSITION_A,
            &[a, b],
            &context(11, 1),
            &relations,
        )
        .unwrap_err();
        assert!(matches!(error, ShadowEvidenceCaseError::MissingRelation { .. }));
    }

    #[test]
    fn relevance_is_recomputed_from_one_context_for_all_candidates() {
        let a = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let b = candidate(11, ShadowObservationFieldV1::LearningOccurred);
        let relations = vec![
            relation(
                PROPOSITION_B,
                &a,
                PROPOSITION_A,
                EvidenceRelationKindV1::Supports,
                500_000,
            ),
            relation(
                SOURCE,
                &b,
                PROPOSITION_A,
                EvidenceRelationKindV1::Supports,
                500_000,
            ),
        ];
        let relevance_context = context(11, 0);
        let case = assemble_shadow_evidence_case_v1(
            PROPOSITION_A,
            &[a, b],
            &relevance_context,
            &relations,
        )
        .unwrap();

        assert_eq!(
            case.relevance_context_commitment(),
            relevance_context.commitment()
        );
        assert!(!case.items()[0].current_runtime_relevant() || !case.items()[1].current_runtime_relevant());
        assert_eq!(
            case.items().iter().filter(|item| item.current_runtime_relevant()).count(),
            1
        );
    }

    #[test]
    fn case_scope_binds_proposition_and_relevance_context() {
        let item = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let relations_a = vec![relation(
            PROPOSITION_B,
            &item,
            PROPOSITION_A,
            EvidenceRelationKindV1::Supports,
            500_000,
        )];
        let case_a = assemble_shadow_evidence_case_v1(
            PROPOSITION_A,
            std::slice::from_ref(&item),
            &context(10, 0),
            &relations_a,
        )
        .unwrap();

        let relations_b = vec![relation(
            SOURCE,
            &item,
            PROPOSITION_B,
            EvidenceRelationKindV1::Supports,
            500_000,
        )];
        let case_b = assemble_shadow_evidence_case_v1(
            PROPOSITION_B,
            &[item],
            &context(10, 0),
            &relations_b,
        )
        .unwrap();

        assert_ne!(case_a.case_scope_digest(), case_b.case_scope_digest());

        let item2 = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let relations_c = vec![relation(
            PROPOSITION_B,
            &item2,
            PROPOSITION_A,
            EvidenceRelationKindV1::Supports,
            500_000,
        )];
        let case_c = assemble_shadow_evidence_case_v1(
            PROPOSITION_A,
            &[item2],
            &context(11, 1),
            &relations_c,
        )
        .unwrap();
        assert_ne!(case_a.case_scope_digest(), case_c.case_scope_digest());
    }

    #[test]
    fn declared_relation_strength_is_preserved_without_aggregation() {
        let item = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let relations = vec![relation(
            PROPOSITION_B,
            &item,
            PROPOSITION_A,
            EvidenceRelationKindV1::Supports,
            123_456,
        )];
        let case = assemble_shadow_evidence_case_v1(
            PROPOSITION_A,
            &[item],
            &context(10, 0),
            &relations,
        )
        .unwrap();
        assert_eq!(
            case.items()[0].declared_relation_strength_ppm(),
            123_456
        );
    }

    #[test]
    fn issued_case_serializes_for_audit() {
        let item = candidate(10, ShadowObservationFieldV1::PredictionErrorPpm);
        let relations = vec![relation(
            PROPOSITION_B,
            &item,
            PROPOSITION_A,
            EvidenceRelationKindV1::Supports,
            500_000,
        )];
        let case = assemble_shadow_evidence_case_v1(
            PROPOSITION_A,
            &[item],
            &context(10, 0),
            &relations,
        )
        .unwrap();
        let encoded = serde_json::to_string(&case).unwrap();
        assert!(encoded.contains(PROPOSITION_A));
        assert!(encoded.contains(case.profile_contract_digest()));
        assert!(encoded.contains(case.relevance_context_commitment()));
    }

    #[test]
    fn profile_has_strict_identity() {
        let digest = shadow_evidence_case_profile_digest_v1();
        assert!(digest.starts_with("blake3:"));
        assert_eq!(digest.len(), "blake3:".len() + 64);
        assert_eq!(digest, shadow_evidence_case_profile_digest_v1());
    }
}
