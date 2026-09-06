// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! RCA-003b.3c: canonical evidence-lineage generation binding for an already
//! issued cross-artifact shadow-disposition preflight.
//!
//! The raw preflight proves cross-artifact coherence and stores the structural
//! case's historical/local lineage reference. This layer adds the stronger #578
//! invariant: every evidence witness must have been issued from the exact
//! canonical complete lineage generation reconstructed from the bound case.
//!
//! This crate performs no disposition and compares no policy thresholds.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_epistemic_governance::{
    evidence_set_witness::IndependentEvidenceSetWitnessV1,
    lineage::{
        CognitiveDerivationKindV1, CognitiveLineageError, EvidenceLineageGraphV1,
        EvidenceLineageNodeV1, ValidatedEvidenceLineageGraphV1,
        COGNITIVE_LINEAGE_SCHEMA_VERSION,
    },
    lineage_identity::{
        canonical_evidence_lineage_graph_id_v1, CanonicalLineageIdentityError,
    },
};
use symthaea_rca_bound_shadow_case::BoundShadowEvidenceCaseV1;
use symthaea_rca_shadow_disposition_preflight::{
    ShadowDispositionEvidenceWitnessSlotsV1, ShadowDispositionPreflightV1,
};

pub const LINEAGE_BOUND_PREFLIGHT_SCHEMA_VERSION: u16 = 1;
pub const LINEAGE_BOUND_PREFLIGHT_PROFILE_V1: &str =
    "rca-lineage-bound-shadow-disposition-preflight-v1";

pub const LINEAGE_BOUND_PREFLIGHT_CONTRACT_V1: &str = concat!(
    "rca-lineage-bound-shadow-disposition-preflight-v1\n",
    "input=issued_shadow_preflight+exact_bound_case+same_evidence_witness_slots\n",
    "raw_preflight_case_scope_must_equal_exact_bound_case\n",
    "raw_preflight_witness_slot_ids_must_equal_supplied_witness_slot_ids\n",
    "case_local_lineage_reference_is_not_canonical_generation_authority\n",
    "canonical_case_lineage=v1_observation_roots+transformation_candidate_children\n",
    "canonical_case_lineage_identity=canonical_evidence_lineage_graph_id_v1\n",
    "every_supplied_evidence_witness_lineage_graph_id_must_equal_canonical_case_generation\n",
    "subset_superset_or_alternate_lineage_generation_fails_closed\n",
    "binding_id=blake3_exact_raw_preflight+canonical_lineage_generation_v1\n",
    "issued_binding=is_private_non_deserializable_shadow_capability\n",
    "binding_performs_no_threshold_comparison_or_disposition\n",
    "binding_is_not_belief_workspace_action_or_promotion_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-lineage-bound-preflight-contract:v1\0";
const BINDING_ID_DOMAIN: &[u8] = b"symthaea:rca-lineage-bound-preflight:v1\0";

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeLineageFactV1 {
    candidate_id: String,
    observation_root_id: String,
}

/// Issued current capability proving that one exact raw preflight is additionally
/// bound to the canonical complete evidence-lineage generation reconstructed from
/// its exact bound case.
///
/// Private fields and the absence of `Deserialize` are deliberate. Archived
/// bytes are audit material; current trust requires recomputation from current
/// issued inputs.
#[must_use = "lineage-bound preflight is a current shadow evaluation capability and should be inspected"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LineageBoundShadowDispositionPreflightV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    binding_id: String,
    canonical_evidence_lineage_graph_id: String,
    preflight: ShadowDispositionPreflightV1,
}

impl LineageBoundShadowDispositionPreflightV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn profile_contract_digest(&self) -> &str {
        &self.profile_contract_digest
    }

    pub fn binding_id(&self) -> &str {
        &self.binding_id
    }

    pub fn canonical_evidence_lineage_graph_id(&self) -> &str {
        &self.canonical_evidence_lineage_graph_id
    }

    pub fn preflight(&self) -> &ShadowDispositionPreflightV1 {
        &self.preflight
    }
}

pub fn lineage_bound_preflight_profile_digest_v1() -> String {
    domain_hash(PROFILE_DOMAIN, LINEAGE_BOUND_PREFLIGHT_CONTRACT_V1.as_bytes())
}

/// Bind an already-issued raw preflight to the exact canonical evidence-lineage
/// generation of its bound runtime case.
///
/// The caller must supply the same evidence-witness slots that were bound into
/// the raw preflight. A witness may be reused across compatible slots, but its
/// identity is never multiplied.
pub fn bind_preflight_to_canonical_lineage_v1(
    preflight: ShadowDispositionPreflightV1,
    bound_case: &BoundShadowEvidenceCaseV1,
    evidence_witnesses: ShadowDispositionEvidenceWitnessSlotsV1<'_>,
) -> Result<LineageBoundShadowDispositionPreflightV1, LineageBoundPreflightError> {
    validate_preflight_case_binding(&preflight, bound_case)?;
    validate_exact_witness_slot(
        "support",
        preflight.support_evidence_witness_id(),
        evidence_witnesses.support.map(IndependentEvidenceSetWitnessV1::witness_id),
    )?;
    validate_exact_witness_slot(
        "opposition",
        preflight.opposition_evidence_witness_id(),
        evidence_witnesses
            .opposition
            .map(IndependentEvidenceSetWitnessV1::witness_id),
    )?;
    validate_exact_witness_slot(
        "defeater",
        preflight.defeater_evidence_witness_id(),
        evidence_witnesses.defeater.map(IndependentEvidenceSetWitnessV1::witness_id),
    )?;

    let canonical_evidence_lineage_graph_id = canonical_runtime_case_lineage_graph_id(bound_case)?;
    validate_witness_lineage(
        "support",
        evidence_witnesses.support,
        &canonical_evidence_lineage_graph_id,
    )?;
    validate_witness_lineage(
        "opposition",
        evidence_witnesses.opposition,
        &canonical_evidence_lineage_graph_id,
    )?;
    validate_witness_lineage(
        "defeater",
        evidence_witnesses.defeater,
        &canonical_evidence_lineage_graph_id,
    )?;

    let profile_contract_digest = lineage_bound_preflight_profile_digest_v1();
    let binding_id = lineage_bound_preflight_id_v1(
        &profile_contract_digest,
        preflight.preflight_id(),
        &canonical_evidence_lineage_graph_id,
    );

    Ok(LineageBoundShadowDispositionPreflightV1 {
        schema_version: LINEAGE_BOUND_PREFLIGHT_SCHEMA_VERSION,
        profile: LINEAGE_BOUND_PREFLIGHT_PROFILE_V1.to_string(),
        profile_contract_digest,
        binding_id,
        canonical_evidence_lineage_graph_id,
        preflight,
    })
}

fn validate_preflight_case_binding(
    preflight: &ShadowDispositionPreflightV1,
    bound_case: &BoundShadowEvidenceCaseV1,
) -> Result<(), LineageBoundPreflightError> {
    let structural = bound_case.structural_case();
    if preflight.case_id() != bound_case.case_id() {
        return Err(LineageBoundPreflightError::CaseIdMismatch);
    }
    if preflight.proposition_id() != structural.proposition_id() {
        return Err(LineageBoundPreflightError::PropositionMismatch);
    }
    if preflight.case_scope_digest() != structural.case_scope_digest() {
        return Err(LineageBoundPreflightError::CaseScopeMismatch);
    }
    if preflight.evidence_lineage_graph_id() != structural.lineage_graph_id() {
        return Err(LineageBoundPreflightError::LocalLineageReferenceMismatch);
    }
    Ok(())
}

fn validate_exact_witness_slot(
    slot: &'static str,
    expected: Option<&str>,
    supplied: Option<&str>,
) -> Result<(), LineageBoundPreflightError> {
    if expected != supplied {
        return Err(LineageBoundPreflightError::EvidenceWitnessSlotMismatch { slot });
    }
    Ok(())
}

fn validate_witness_lineage(
    slot: &'static str,
    witness: Option<&IndependentEvidenceSetWitnessV1>,
    canonical_case_generation: &str,
) -> Result<(), LineageBoundPreflightError> {
    if let Some(witness) = witness {
        if witness.lineage_graph_id() != canonical_case_generation {
            return Err(LineageBoundPreflightError::EvidenceWitnessLineageGenerationMismatch {
                slot,
                witness_lineage_graph_id: witness.lineage_graph_id().to_string(),
                canonical_case_lineage_graph_id: canonical_case_generation.to_string(),
            });
        }
    }
    Ok(())
}

fn canonical_runtime_case_lineage_graph_id(
    bound_case: &BoundShadowEvidenceCaseV1,
) -> Result<String, LineageBoundPreflightError> {
    let facts = bound_case
        .structural_case()
        .items()
        .iter()
        .map(|item| RuntimeLineageFactV1 {
            candidate_id: item.candidate_id().to_string(),
            observation_root_id: item.observation_root_id().to_string(),
        })
        .collect::<Vec<_>>();
    canonical_runtime_lineage_graph_id_from_facts(
        &facts,
        bound_case.structural_case().lineage_graph_id(),
    )
}

fn canonical_runtime_lineage_graph_id_from_facts(
    facts: &[RuntimeLineageFactV1],
    legacy_local_graph_id: &str,
) -> Result<String, LineageBoundPreflightError> {
    if facts.is_empty() {
        return Err(LineageBoundPreflightError::EmptyCaseLineage);
    }

    let mut seen_roots = BTreeSet::new();
    let mut seen_candidates = BTreeSet::new();
    let mut nodes = Vec::with_capacity(facts.len().saturating_mul(2));

    for fact in facts {
        if !seen_candidates.insert(fact.candidate_id.clone()) {
            return Err(LineageBoundPreflightError::DuplicateCandidateId {
                candidate_id: fact.candidate_id.clone(),
            });
        }
        if seen_roots.insert(fact.observation_root_id.clone()) {
            nodes.push(
                EvidenceLineageNodeV1 {
                    schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
                    evidence_id: fact.observation_root_id.clone(),
                    parent_ids: Vec::new(),
                    derivation_kind: CognitiveDerivationKindV1::RootObservation,
                }
                .validate()
                .map_err(LineageBoundPreflightError::Lineage)?,
            );
        }
        nodes.push(
            EvidenceLineageNodeV1 {
                schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
                evidence_id: fact.candidate_id.clone(),
                parent_ids: vec![fact.observation_root_id.clone()],
                derivation_kind: CognitiveDerivationKindV1::Transformation,
            }
            .validate()
            .map_err(LineageBoundPreflightError::Lineage)?,
        );
    }

    let graph: ValidatedEvidenceLineageGraphV1 = EvidenceLineageGraphV1 {
        schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
        // Legacy/local case reference is retained only to satisfy the v1 wire
        // shape. #578 canonical identity deliberately excludes this field.
        graph_id: legacy_local_graph_id.to_string(),
        nodes,
    }
    .validate()
    .map_err(LineageBoundPreflightError::Lineage)?;

    canonical_evidence_lineage_graph_id_v1(&graph)
        .map_err(LineageBoundPreflightError::CanonicalLineageIdentity)
}

fn lineage_bound_preflight_id_v1(
    profile_contract_digest: &str,
    preflight_id: &str,
    canonical_evidence_lineage_graph_id: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(BINDING_ID_DOMAIN);
    hash_text(
        &mut hasher,
        b"profile_contract_digest",
        profile_contract_digest,
    );
    hash_bytes(
        &mut hasher,
        b"schema_version",
        &LINEAGE_BOUND_PREFLIGHT_SCHEMA_VERSION.to_le_bytes(),
    );
    hash_text(&mut hasher, b"preflight_id", preflight_id);
    hash_text(
        &mut hasher,
        b"canonical_evidence_lineage_graph_id",
        canonical_evidence_lineage_graph_id,
    );
    format!("blake3:{}", hasher.finalize().to_hex())
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
pub enum LineageBoundPreflightError {
    CaseIdMismatch,
    PropositionMismatch,
    CaseScopeMismatch,
    LocalLineageReferenceMismatch,
    EvidenceWitnessSlotMismatch {
        slot: &'static str,
    },
    EvidenceWitnessLineageGenerationMismatch {
        slot: &'static str,
        witness_lineage_graph_id: String,
        canonical_case_lineage_graph_id: String,
    },
    EmptyCaseLineage,
    DuplicateCandidateId {
        candidate_id: String,
    },
    Lineage(CognitiveLineageError),
    CanonicalLineageIdentity(CanonicalLineageIdentityError),
}

impl std::fmt::Display for LineageBoundPreflightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CaseIdMismatch => f.write_str("raw preflight case id does not match exact bound case"),
            Self::PropositionMismatch => {
                f.write_str("raw preflight proposition does not match exact bound case")
            }
            Self::CaseScopeMismatch => {
                f.write_str("raw preflight case scope does not match exact bound case")
            }
            Self::LocalLineageReferenceMismatch => f.write_str(
                "raw preflight local lineage reference does not match exact bound case",
            ),
            Self::EvidenceWitnessSlotMismatch { slot } => {
                write!(f, "{slot} evidence witness differs from raw preflight binding")
            }
            Self::EvidenceWitnessLineageGenerationMismatch {
                slot,
                witness_lineage_graph_id,
                canonical_case_lineage_graph_id,
            } => write!(
                f,
                "{slot} evidence witness lineage generation {witness_lineage_graph_id} does not match canonical case generation {canonical_case_lineage_graph_id}"
            ),
            Self::EmptyCaseLineage => f.write_str("bound case has no runtime lineage facts"),
            Self::DuplicateCandidateId { candidate_id } => {
                write!(f, "duplicate candidate id while reconstructing case lineage: {candidate_id}")
            }
            Self::Lineage(error) => write!(f, "case lineage reconstruction failed: {error}"),
            Self::CanonicalLineageIdentity(error) => {
                write!(f, "canonical case lineage identity failed: {error}")
            }
        }
    }
}

impl std::error::Error for LineageBoundPreflightError {}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E: &str = "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn fact(candidate_id: &str, observation_root_id: &str) -> RuntimeLineageFactV1 {
        RuntimeLineageFactV1 {
            candidate_id: candidate_id.to_string(),
            observation_root_id: observation_root_id.to_string(),
        }
    }

    #[test]
    fn runtime_lineage_reconstruction_is_order_independent() {
        let first = canonical_runtime_lineage_graph_id_from_facts(
            &[fact(C, A), fact(D, B)],
            E,
        )
        .unwrap();
        let second = canonical_runtime_lineage_graph_id_from_facts(
            &[fact(D, B), fact(C, A)],
            B,
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn same_observation_root_is_deduplicated() {
        let two_fields_same_observation = canonical_runtime_lineage_graph_id_from_facts(
            &[fact(C, A), fact(D, A)],
            E,
        )
        .unwrap();
        let reordered = canonical_runtime_lineage_graph_id_from_facts(
            &[fact(D, A), fact(C, A)],
            B,
        )
        .unwrap();
        assert_eq!(two_fields_same_observation, reordered);
    }

    #[test]
    fn unrelated_candidate_changes_complete_generation() {
        let first = canonical_runtime_lineage_graph_id_from_facts(&[fact(C, A)], E).unwrap();
        let second = canonical_runtime_lineage_graph_id_from_facts(
            &[fact(C, A), fact(D, B)],
            E,
        )
        .unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn local_legacy_graph_reference_does_not_define_generation() {
        let facts = [fact(C, A), fact(D, B)];
        let first = canonical_runtime_lineage_graph_id_from_facts(&facts, E).unwrap();
        let second = canonical_runtime_lineage_graph_id_from_facts(&facts, B).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn witness_slot_binding_is_exact() {
        assert!(validate_exact_witness_slot("support", Some(A), Some(A)).is_ok());
        assert!(matches!(
            validate_exact_witness_slot("support", Some(A), Some(B)),
            Err(LineageBoundPreflightError::EvidenceWitnessSlotMismatch { .. })
        ));
        assert!(matches!(
            validate_exact_witness_slot("support", Some(A), None),
            Err(LineageBoundPreflightError::EvidenceWitnessSlotMismatch { .. })
        ));
        assert!(matches!(
            validate_exact_witness_slot("support", None, Some(A)),
            Err(LineageBoundPreflightError::EvidenceWitnessSlotMismatch { .. })
        ));
    }

    #[test]
    fn binding_profile_identity_is_stable() {
        let first = lineage_bound_preflight_profile_digest_v1();
        assert_eq!(first, lineage_bound_preflight_profile_digest_v1());
        assert!(first.starts_with("blake3:"));
    }
}
