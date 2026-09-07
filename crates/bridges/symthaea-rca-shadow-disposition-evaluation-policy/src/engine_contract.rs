// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Non-result-bearing contract for the future RCA-003b.4 pure shadow-disposition
//! engine.
//!
//! This module intentionally implements no classifier and accepts no evaluation
//! artifacts. It freezes the outcome taxonomy, precedence, cardinality semantics,
//! reason-trace requirements, and authority boundary *before* result-bearing code
//! exists.

#![deny(unsafe_code)]

pub const SHADOW_DISPOSITION_ENGINE_CONTRACT_SCHEMA_VERSION: u16 = 1;
pub const SHADOW_DISPOSITION_ENGINE_CONTRACT_PROFILE_V1: &str =
    "rca-pure-shadow-disposition-engine-contract-v1";

pub const SHADOW_DISPOSITION_ENGINE_CONTRACT_V1: &str = concat!(
    "rca-pure-shadow-disposition-engine-contract-v1\n",
    "contract_precedes_result_bearing_engine_implementation\n",
    "future_input=registered_evaluation_policy+lineage_bound_preflight+exact_evidence_witness_slots+exact_interpretation_witness_slots\n",
    "future_engine_must_not_accept_raw_case_raw_preflight_eligibility_lineage_or_experiment_as_separate_inputs\n",
    "identity_join=effective_policy_id+raw_preflight_profile+lineage_bound_preflight_profile+exact_embedded_witness_ids\n",
    "invalid_identity_profile_or_witness_join_is_engine_error_not_epistemic_outcome\n",
    "evidence_cardinality=one_exact_issued_evidence_witness_item_count_per_slot\n",
    "interpretation_cardinality=one_exact_issued_interpretation_witness_root_count_per_slot\n",
    "ancestry_root_pair_edge_candidate_declaration_module_and_strength_counts_are_not_cardinality_substitutes\n",
    "threshold_predicates=D,CS,CO,S,TS,O,TO\n",
    "registered_invariants=S_implies_TS+O_implies_TO\n",
    "precedence=qualified_defeater>qualified_contestation>bilateral_qualifying_disagreement>unilateral_full>unilateral_tentative>underdetermined\n",
    "qualified_defeater_primary=blocked_by_qualified_defeater_not_false_or_refuted\n",
    "bilateral_tentative_or_stronger_disagreement_below_contestation=underdetermined\n",
    "count_margin_vote_strength_posterior_and_winner_take_all_tiebreakers_forbidden\n",
    "unknown_interpretation_independence_is_scoped_to_witness_that_requires_it_not_global_case_poison\n",
    "reason_trace=typed_predicates+typed_rule_id+primary_class+identity_lineage+slot_cardinalities\n",
    "reason_trace_preserves_all_simultaneously_true_predicates_under_higher_precedence_rule\n",
    "result_identity=domain_separated_serializer_independent_complete_normalized_result_v1\n",
    "future_issued_result=private_serialize_only_recompute_for_current_result\n",
    "engine=pure_deterministic_stateless_clock_rng_network_filesystem_callback_free\n",
    "engine_has_no_belief_workspace_gwt_action_or_recursive_improvement_promotion_authority\n",
);

/// Stable primary outcome tags. These are taxonomy only; this module does not
/// construct a result-bearing value.
pub const SHADOW_DISPOSITION_CLASS_TAGS_V1: &[&str] = &[
    "blocked_by_qualified_defeater",
    "contested",
    "supported",
    "tentatively_supported",
    "opposed",
    "tentatively_opposed",
    "underdetermined",
];

/// Stable decision-rule identifiers in exact V1 precedence order.
pub const SHADOW_DISPOSITION_RULE_PRECEDENCE_V1: &[&str] = &[
    "qualified_defeater_blocker",
    "qualified_contestation",
    "bilateral_qualified_disagreement_below_contestation",
    "full_support",
    "full_opposition",
    "tentative_support",
    "tentative_opposition",
    "insufficient_topology",
];

/// Stable boolean predicate identifiers retained in the future reason trace.
pub const SHADOW_DISPOSITION_PREDICATE_TAGS_V1: &[&str] = &[
    "defeater_qualified",
    "support_contested_side_qualified",
    "opposition_contested_side_qualified",
    "support_full_qualified",
    "support_tentative_qualified",
    "opposition_full_qualified",
    "opposition_tentative_qualified",
];

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-pure-shadow-disposition-engine-contract:v1\0";

/// Content identity for the exact non-result-bearing engine semantics.
///
/// The digest binds the normative contract and every stable machine-readable tag
/// table. Changing taxonomy, predicates, rule IDs, or precedence therefore
/// requires a new evaluation-policy identity before any result-bearing engine may
/// run.
pub fn shadow_disposition_engine_contract_profile_digest_v1() -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PROFILE_DOMAIN);
    hash_bytes(
        &mut hasher,
        b"contract",
        SHADOW_DISPOSITION_ENGINE_CONTRACT_V1.as_bytes(),
    );
    hash_bytes(
        &mut hasher,
        b"schema_version",
        &SHADOW_DISPOSITION_ENGINE_CONTRACT_SCHEMA_VERSION.to_le_bytes(),
    );
    hash_text(
        &mut hasher,
        b"profile",
        SHADOW_DISPOSITION_ENGINE_CONTRACT_PROFILE_V1,
    );
    hash_tags(
        &mut hasher,
        b"class_tags",
        SHADOW_DISPOSITION_CLASS_TAGS_V1,
    );
    hash_tags(
        &mut hasher,
        b"rule_precedence",
        SHADOW_DISPOSITION_RULE_PRECEDENCE_V1,
    );
    hash_tags(
        &mut hasher,
        b"predicate_tags",
        SHADOW_DISPOSITION_PREDICATE_TAGS_V1,
    );
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn hash_tags(hasher: &mut blake3::Hasher, label: &[u8], tags: &[&str]) {
    hash_bytes(hasher, label, &(tags.len() as u64).to_le_bytes());
    for tag in tags {
        hash_text(hasher, b"tag", tag);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_contract_has_expected_primary_taxonomy() {
        assert_eq!(SHADOW_DISPOSITION_CLASS_TAGS_V1.len(), 7);
        assert_eq!(
            SHADOW_DISPOSITION_CLASS_TAGS_V1[0],
            "blocked_by_qualified_defeater"
        );
        assert_eq!(
            SHADOW_DISPOSITION_CLASS_TAGS_V1.last().copied(),
            Some("underdetermined")
        );
    }

    #[test]
    fn rule_precedence_preserves_bilateral_disagreement_before_unilateral_outcomes() {
        let bilateral = SHADOW_DISPOSITION_RULE_PRECEDENCE_V1
            .iter()
            .position(|tag| *tag == "bilateral_qualified_disagreement_below_contestation")
            .unwrap();
        let support = SHADOW_DISPOSITION_RULE_PRECEDENCE_V1
            .iter()
            .position(|tag| *tag == "full_support")
            .unwrap();
        let opposition = SHADOW_DISPOSITION_RULE_PRECEDENCE_V1
            .iter()
            .position(|tag| *tag == "full_opposition")
            .unwrap();
        assert!(bilateral < support);
        assert!(bilateral < opposition);
    }

    #[test]
    fn defeater_is_primary_rule() {
        assert_eq!(
            SHADOW_DISPOSITION_RULE_PRECEDENCE_V1.first().copied(),
            Some("qualified_defeater_blocker")
        );
    }

    #[test]
    fn contract_profile_is_content_addressed() {
        let digest = shadow_disposition_engine_contract_profile_digest_v1();
        assert!(digest.starts_with("blake3:"));
        assert_eq!(digest.len(), "blake3:".len() + 64);
    }

    #[test]
    fn contract_exposes_no_evaluator() {
        // Compile-time shape test: this module intentionally exposes only static
        // contract data and its profile digest. Result-bearing behavior belongs in
        // a later crate after qualification.
        assert!(!SHADOW_DISPOSITION_ENGINE_CONTRACT_V1.contains("engine_implements_evaluate"));
    }
}
