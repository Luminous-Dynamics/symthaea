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
    "identity_join=evaluation_policy_engine_contract+effective_policy+preflight_profiles+exact_embedded_witness_ids\n",
    "invalid_identity_profile_or_witness_join_is_engine_error_not_epistemic_outcome\n",
    "evidence_cardinality=one_exact_issued_evidence_witness_item_count_per_slot\n",
    "interpretation_cardinality=one_exact_issued_interpretation_witness_root_count_per_slot\n",
    "ancestry_root_pair_edge_candidate_declaration_module_and_strength_counts_are_not_cardinality_substitutes\n",
    "threshold_predicates=D,CS,CO,S,TS,O,TO\n",
    "registered_invariants=S_implies_TS+O_implies_TO\n",
    "precedence=qualified_defeater>qualified_contestation>bilateral_qualifying_disagreement>unilateral_full>unilateral_tentative>underdetermined\n",
    "rule_to_primary_class_mapping_is_exact_and_profile_bearing\n",
    "decision_lattice_total_over_all_72_admissible_predicate_states_v1\n",
    "qualified_defeater_primary=blocked_by_qualified_defeater_not_false_or_refuted\n",
    "bilateral_tentative_or_stronger_disagreement_below_contestation=underdetermined\n",
    "count_margin_vote_strength_posterior_and_winner_take_all_tiebreakers_forbidden\n",
    "unknown_interpretation_independence_is_scoped_to_witness_that_requires_it_not_global_case_poison\n",
    "reason_trace_fields_are_exact_and_profile_bearing\n",
    "reason_trace=identity_lineage+slot_facts+typed_predicates+typed_rule_id+primary_class\n",
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

/// Exact V1 mapping from every decision-rule identifier to its primary class.
/// The mapping is profile-bearing so a future implementation cannot keep the
/// same precedence table while silently changing what a selected rule means.
pub const SHADOW_DISPOSITION_RULE_TO_CLASS_V1: &[(&str, &str)] = &[
    (
        "qualified_defeater_blocker",
        "blocked_by_qualified_defeater",
    ),
    ("qualified_contestation", "contested"),
    (
        "bilateral_qualified_disagreement_below_contestation",
        "underdetermined",
    ),
    ("full_support", "supported"),
    ("full_opposition", "opposed"),
    ("tentative_support", "tentatively_supported"),
    ("tentative_opposition", "tentatively_opposed"),
    ("insufficient_topology", "underdetermined"),
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

/// Exact identity/provenance fields retained in every future result reason trace.
pub const SHADOW_DISPOSITION_IDENTITY_TRACE_TAGS_V1: &[&str] = &[
    "engine_implementation_profile_digest",
    "engine_contract_profile_digest",
    "evaluation_policy_id",
    "effective_policy_id",
    "base_policy_id",
    "lineage_bound_preflight_binding_id",
    "raw_preflight_id",
    "proposition_id",
    "case_id",
    "canonical_evidence_lineage_graph_id",
    "registered_experiment_contract_digest",
];

/// Exact per-slot evidence/interpretation cardinality facts retained in every
/// future result reason trace.
pub const SHADOW_DISPOSITION_SLOT_TRACE_TAGS_V1: &[&str] = &[
    "support_evidence_witness_id",
    "support_evidence_item_count",
    "support_interpretation_witness_id",
    "support_interpretation_root_count",
    "opposition_evidence_witness_id",
    "opposition_evidence_item_count",
    "opposition_interpretation_witness_id",
    "opposition_interpretation_root_count",
    "defeater_evidence_witness_id",
    "defeater_evidence_item_count",
    "defeater_interpretation_witness_id",
    "defeater_interpretation_root_count",
];

/// Exact final decision fields retained in every future result reason trace.
pub const SHADOW_DISPOSITION_DECISION_TRACE_TAGS_V1: &[&str] =
    &["decision_rule_id", "primary_class"];

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-pure-shadow-disposition-engine-contract:v1\0";

/// Content identity for the exact non-result-bearing engine semantics.
///
/// The digest binds the normative contract and every stable machine-readable tag
/// table. Changing taxonomy, predicates, rule IDs, precedence, rule meaning, or
/// reason-trace schema therefore requires a new evaluation-policy identity before
/// any result-bearing engine may run.
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
    hash_rule_mapping(
        &mut hasher,
        b"rule_to_class",
        SHADOW_DISPOSITION_RULE_TO_CLASS_V1,
    );
    hash_tags(
        &mut hasher,
        b"predicate_tags",
        SHADOW_DISPOSITION_PREDICATE_TAGS_V1,
    );
    hash_tags(
        &mut hasher,
        b"identity_trace_tags",
        SHADOW_DISPOSITION_IDENTITY_TRACE_TAGS_V1,
    );
    hash_tags(
        &mut hasher,
        b"slot_trace_tags",
        SHADOW_DISPOSITION_SLOT_TRACE_TAGS_V1,
    );
    hash_tags(
        &mut hasher,
        b"decision_trace_tags",
        SHADOW_DISPOSITION_DECISION_TRACE_TAGS_V1,
    );
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn hash_tags(hasher: &mut blake3::Hasher, label: &[u8], tags: &[&str]) {
    hash_bytes(hasher, label, &(tags.len() as u64).to_le_bytes());
    for tag in tags {
        hash_text(hasher, b"tag", tag);
    }
}

fn hash_rule_mapping(hasher: &mut blake3::Hasher, label: &[u8], mappings: &[(&str, &str)]) {
    hash_bytes(hasher, label, &(mappings.len() as u64).to_le_bytes());
    for (rule, class) in mappings {
        hash_text(hasher, b"rule", rule);
        hash_text(hasher, b"class", class);
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

    #[derive(Clone, Copy)]
    struct PredicateState {
        defeater: bool,
        support_contested: bool,
        opposition_contested: bool,
        support_full: bool,
        support_tentative: bool,
        opposition_full: bool,
        opposition_tentative: bool,
    }

    fn frozen_rule_for(state: PredicateState) -> &'static str {
        if state.defeater {
            "qualified_defeater_blocker"
        } else if state.support_contested && state.opposition_contested {
            "qualified_contestation"
        } else if state.support_tentative && state.opposition_tentative {
            "bilateral_qualified_disagreement_below_contestation"
        } else if state.support_full {
            "full_support"
        } else if state.opposition_full {
            "full_opposition"
        } else if state.support_tentative {
            "tentative_support"
        } else if state.opposition_tentative {
            "tentative_opposition"
        } else {
            "insufficient_topology"
        }
    }

    fn class_for_rule(rule: &str) -> &'static str {
        SHADOW_DISPOSITION_RULE_TO_CLASS_V1
            .iter()
            .find_map(|(candidate, class)| (*candidate == rule).then_some(*class))
            .expect("every frozen rule must map to one primary class")
    }

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
    fn rule_to_class_mapping_covers_every_rule_exactly_once() {
        assert_eq!(
            SHADOW_DISPOSITION_RULE_TO_CLASS_V1.len(),
            SHADOW_DISPOSITION_RULE_PRECEDENCE_V1.len()
        );
        for rule in SHADOW_DISPOSITION_RULE_PRECEDENCE_V1 {
            assert_eq!(
                SHADOW_DISPOSITION_RULE_TO_CLASS_V1
                    .iter()
                    .filter(|(candidate, _)| candidate == rule)
                    .count(),
                1,
                "rule {rule} must have exactly one primary class"
            );
            assert!(SHADOW_DISPOSITION_CLASS_TAGS_V1.contains(&class_for_rule(rule)));
        }
    }

    #[test]
    fn reason_trace_schema_is_exact_and_identity_bearing() {
        assert!(SHADOW_DISPOSITION_IDENTITY_TRACE_TAGS_V1.contains(&"evaluation_policy_id"));
        assert!(SHADOW_DISPOSITION_IDENTITY_TRACE_TAGS_V1.contains(&"case_id"));
        assert!(SHADOW_DISPOSITION_IDENTITY_TRACE_TAGS_V1
            .contains(&"canonical_evidence_lineage_graph_id"));
        assert_eq!(SHADOW_DISPOSITION_SLOT_TRACE_TAGS_V1.len(), 12);
        assert_eq!(
            SHADOW_DISPOSITION_DECISION_TRACE_TAGS_V1,
            &["decision_rule_id", "primary_class"]
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
    fn decision_lattice_is_total_over_all_72_admissible_states() {
        let mut admissible = 0_u16;
        for bits in 0_u16..128 {
            let state = PredicateState {
                defeater: bits & 1 != 0,
                support_contested: bits & 2 != 0,
                opposition_contested: bits & 4 != 0,
                support_full: bits & 8 != 0,
                support_tentative: bits & 16 != 0,
                opposition_full: bits & 32 != 0,
                opposition_tentative: bits & 64 != 0,
            };
            if state.support_full && !state.support_tentative {
                continue;
            }
            if state.opposition_full && !state.opposition_tentative {
                continue;
            }
            admissible += 1;
            let rule = frozen_rule_for(state);
            assert!(SHADOW_DISPOSITION_RULE_PRECEDENCE_V1.contains(&rule));
            let class = class_for_rule(rule);
            assert!(SHADOW_DISPOSITION_CLASS_TAGS_V1.contains(&class));
        }
        assert_eq!(admissible, 72);
    }

    #[test]
    fn contract_profile_is_content_addressed() {
        let digest = shadow_disposition_engine_contract_profile_digest_v1();
        assert!(digest.starts_with("blake3:"));
        assert_eq!(digest.len(), "blake3:".len() + 64);
    }

    #[test]
    fn contract_exposes_no_evaluator() {
        // The classifier above exists only in #[cfg(test)] to prove the frozen
        // contract is total. Production exposes static contract data and its
        // profile digest only; result-bearing behavior belongs in a later crate.
        assert!(!SHADOW_DISPOSITION_ENGINE_CONTRACT_V1.contains("engine_implements_evaluate"));
    }
}
